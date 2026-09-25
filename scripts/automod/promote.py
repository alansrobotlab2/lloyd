"""Land a gated candidate on the live tree, reversibly.

Ordering is the whole design, and one step is non-negotiable:

**The rollback point is written and read back BEFORE anything is mutated.**
`scripts/autoresearch/promote.py::snapshot_current_prompts` mkdirs
unconditionally, never verifies the copy landed, and `promote()` overwrites
live state regardless — which is why 26 of 83 historical promotions have no
snapshot and therefore no way back (see
`tests/test_autoresearch_promotion.py:362`, an xfail documenting the live
defect). Here `write_verified` raises unless the state file round-trips, and
nothing touches the tree until it has.

The promoter never advances `last_known_good`. It writes `current.json` with
an observation window; the **guardian** promotes that to LKG only after the
window passes clean. So "last known good" means *observed healthy in
production*, and a rollback always targets a commit that already survived a
full window.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.supervisor_client import process_info, restart_process, start_process, stop_process
from scripts.automod import ram_gate, state as S, worktree as W

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

IDLE_POLL_SECONDS = 2.0
IDLE_QUIET_POLLS = 3
IDLE_MAX_WAIT = 900.0
# The ceiling on the whole wait, however it is extended. Above the longest
# worker `max_duration_seconds` (3600): once the pool is paused nothing new
# starts, so what is in flight is finite and waiting it out always ends.
IDLE_HARD_MAX_WAIT = 4500.0
DRAIN_TTL = 180.0
DRAIN_REFRESH_SECONDS = 60.0   # re-arm well inside the TTL while waiting for idle
# The observation window, and the number that decides how much of the loop's
# day is spent serialized behind it. Liveness and the error rate are watched
# for ALL of it, not for some shorter sub-window: a build that crashes at
# minute ten is exactly as bad as one that crashes at minute one. There was a
# `LIVENESS_WINDOW = 120.0` here and a `liveness_until_ts` written into every
# promotion record, and nothing ever read either — the guardian applies
# liveness whenever a promotion is under observation. A constant that looks
# like a bound and bounds nothing is worse than no constant, which is also why
# `errors_window_s` is *read* below: it has sat in `automod.landing` since that
# block was written and nothing has ever read it, exactly like the two idle
# keys found dead on 2026-09-17.
#
# 900 until 2026-09-20. Every rollback the window has ever caused fired within
# 5.5 minutes of the landing — 09-07 `error_rate` at 4 s, 09-06 `crash` at
# 147 s, 09-10 `data_damage` at 262 s, 09-17 `data_damage` at 327 s — and all
# four were false positives or misattributions (the 09-10 one blamed the
# `bench_010` vault wipe on whichever promotion happened to be observing, the
# 09-17 one a `.git` repack the file counter did not skip). Against that, on a
# 54-promotion day the window cost 13.5 h of the 24, and it is what every
# landing queues behind. 450 cleared the longest signal ever observed with
# margin, at half the serialization.
#
# 300 since 2026-09-24 (Alan's ruling, with the land train). Of the four, the
# crash fired at 147 s and the error spike at 4 s; the 327 s trip was the `.git`
# repack miscount and the 262 s one the bench_010 wipe — and a wipe of the
# data home or the vault is caught on EVERY guardian tick, inside a window or
# not (`check_data`, `check_vault`: halt and alert), so the window is not the
# only thing between such damage and a human.
ERRORS_WINDOW = 300.0
# A landing that restarted no service gets a shorter one. The guardian already
# skips liveness and the error rate for such a promotion (`unrestarted` in
# `guardian.tick`): the code that could crash is the code that was already
# running, and the error log belongs to a build that never booted. What it
# still judges is data damage, because a script the landing changed can be run
# by a job inside the window. That is a weak guard by construction — a nightly
# changed at 14:00 runs at 02:00, outside any window this loop would tolerate —
# so the window is shortened here rather than removed. 24 of the 55 promotions
# on 2026-09-20 restarted nothing.
ERRORS_WINDOW_UNRESTARTED = 120.0
# Bounds on what config.yaml may set. A window under the floor cannot judge
# anything — the guardian ticks every 5 s and `crash` needs three consecutive
# failed probes — and one that settles with nothing observed still advances the
# LKG, quietly turning "last known good" into "last landed". Above the ceiling
# the loop stops landing. A value outside either is clamped, not obeyed.
ERRORS_WINDOW_FLOOR = 60.0
ERRORS_WINDOW_CEILING = 3600.0
RESTART_LEASE = 120.0

SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
SUPERVISORCTL = Path.home() / ".local/share/uv/tools/supervisor/bin/supervisorctl"
SUPERVISORD_CONF = LIVE_ROOT / "agent-services" / "supervisor" / "supervisord.conf"

BACKEND = "http://127.0.0.1:8080"
MCP_HEALTH = "http://127.0.0.1:8500/health"
# The primary engine, restartable through `round restart --only
# agent-llm-primary` since 2026-09-15 so a KV or venv change goes through the
# same lease, pool pause and drain as a backend restart. Three things make
# its leg different from the two above, all measured on this box:
#   - the boot takes 240-265 s warm and 775 s after a venv change (the
#     program's startsecs is 900), so the lease is refreshed while waiting;
#   - `supervisorctl stop` returns before the kernel has reclaimed the
#     95.37 GiB host-RAM n-gram table, and a second boot started too soon
#     put two of them on a 251 GiB box — systemd-oomd then killed the whole
#     supervisord unit (2026-09-08, twice). The leg waits for MemAvailable
#     to pass the boot gate's floor and refuses to boot below its abort line;
#   - its `environment=` lives in the program's conf, so the leg runs
#     `reread` + `update` before starting, or an edited KV budget is ignored.
PRIMARY_PROGRAM = "agent-llm-primary"
PRIMARY_HEALTH = "http://127.0.0.1:8096/health"
PRIMARY_HEALTH_BUDGET = 1200.0
# The boot-gate thresholds are READ here, never held: the file that defines them
# is `agent-services/bin/ram-boot-gate.sh`, which the A/B sweep route shells in
# and this route reads through `scripts.automod.ram_gate`. The numbers, the oomd
# history that set them, and why the sweep pair sits lower than this pair all
# live in that one header. #1340 closed the state where 180/150 stood here and
# 150/120 stood in the arm script with nothing between them, and where the only
# pointer to the production pair — in the restart skill — was 596 lines off. The
# names are bound into this module so the restart-leg tests can still patch them;
# the values are the definition file's, and a broken definition raises at import
# rather than falling back to a remembered number.
PRIMARY_RAM_FLOOR_GIB = ram_gate.PRIMARY_RAM_FLOOR_GIB
PRIMARY_RAM_ABORT_GIB = ram_gate.PRIMARY_RAM_ABORT_GIB
PRIMARY_RAM_WAIT_SECONDS = ram_gate.PRIMARY_RAM_WAIT_SECONDS
# Vite dev server, serving the live tree over HTTPS with a private cert. Not
# restarted by a landing: HMR picks the fast-forward up on its own.
FRONTEND_URL = "https://127.0.0.1:5173/"


class PromoteError(RuntimeError):
    pass


def _frontend_alive(url: str = FRONTEND_URL, budget: float = 30.0) -> tuple[bool, str]:
    """Does the Vite dev server still answer after a frontend landing?

    Liveness only, and deliberately so: a broken `src` change is a
    browser-side error the dev server serves with a 200. The gate's `vite
    build` is what verifies the change; this catches the one thing the
    landing itself could do to the frontend — leave it unreachable."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    deadline = time.time() + budget
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0, context=ctx) as resp:
                if 200 <= resp.status < 400:
                    return True, f"HTTP {resp.status}"
                last = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except Exception as exc:  # refused, TLS, timeout
            last = type(exc).__name__
        time.sleep(2.0)
    return False, last or "no answer"


def _get(url: str, timeout: float = 5.0):
    """`(status, body)`. The status is reported whenever the server answered,
    JSON body or not: vLLM's `/health` is a bare 200 with an empty body, and
    reading that as `(None, None)` kept the primary's restart leg waiting on
    an engine that had been serving for minutes (2026-09-15)."""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(url), timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw) if raw.strip() else None
            except ValueError:
                return r.status, None
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "replace"))
        except Exception:
            return e.code, None
    except Exception:
        return None, None


def _post(url: str, payload: dict, timeout: float = 5.0) -> bool:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def set_drain(on: bool, ttl: float = DRAIN_TTL) -> bool:
    return _post(f"{BACKEND}/api/automod/drain", {"on": on, "ttl_s": ttl})


def pool_paused() -> bool | None:
    """The worker pool's pause flag, or None if the backend cannot say."""
    status, body = _get(f"{BACKEND}/api/workers/status")
    if status != 200 or not isinstance(body, dict):
        return None
    pool = body.get("pool") if isinstance(body.get("pool"), dict) else body
    value = pool.get("paused")
    return None if value is None else bool(value)


def set_pool_paused(paused: bool) -> bool:
    # `owner: automod`: a transient pause the restart clears. An unowned one is
    # an operator's, persisted across restarts — and a landing's pause made
    # durable would leave the pool paused after every landing.
    return _post(f"{BACKEND}/api/workers/pause", {"paused": paused, "owner": "automod"})


# True while a pause THIS promoter set is in force. It is sent as
# `owner: automod`, which the backend keeps in memory only (an operator's pause
# is persisted), so a successful landing's restart clears it; the paths that
# matter are the ones that never restart — give-up, and every PromoteError
# before the restart — where a pause we set must be released. A pause a human
# set is never touched: Alan pauses the queue by hand before restarts, and a
# promoter that silently resumed it would be undoing him.
_POOL_PAUSED_BY_US = False


def release_pool_pause() -> None:
    global _POOL_PAUSED_BY_US
    if _POOL_PAUSED_BY_US:
        set_pool_paused(False)
        _POOL_PAUSED_BY_US = False


def pool_in_flight() -> list[dict] | None:
    """The worker jobs the pool is running now (`source`, `started_at`), or
    None when the backend cannot say."""
    status, body = _get(f"{BACKEND}/api/workers/status")
    if status != 200 or not isinstance(body, dict):
        return None
    pool = body.get("pool") if isinstance(body.get("pool"), dict) else {}
    jobs = pool.get("in_flight")
    if not isinstance(jobs, dict):
        return None
    return [dict(v) for v in jobs.values() if isinstance(v, dict)]


def _landing_num(key: str, default: float) -> float:
    """One scalar out of `automod.landing`, `default` on anything unreadable.

    Read per call rather than bound as a default argument: `round.land`
    imports this module once per process, so a default would freeze the value
    at import time. `landing_cfg` caches on the file's mtime, so the call is
    ~0.03 ms and an edit is still picked up by the next landing.
    """
    try:
        return float(S.landing_cfg(LIVE_ROOT).get(key, default))
    except (TypeError, ValueError):
        return default


def drain_ttl() -> float:
    """`automod.landing.drain_ttl_s` — how long a drain stays armed."""
    return _landing_num("drain_ttl_s", DRAIN_TTL)


def idle_quiet_polls() -> float:
    """`automod.landing.idle_quiet_polls` — consecutive quiet polls = idle."""
    return _landing_num("idle_quiet_polls", IDLE_QUIET_POLLS)


def _idle_budget(max_wait: float | None) -> tuple[float, float]:
    """`(quiet budget, hard ceiling)` from `automod.landing`, arguments first."""
    budget = (float(max_wait) if max_wait is not None
              else _landing_num("idle_max_wait_s", IDLE_MAX_WAIT))
    return budget, max(budget, _landing_num("idle_hard_max_wait_s", IDLE_HARD_MAX_WAIT))


def errors_window(restart: bool = True) -> float:
    """How long this promotion is observed, by whether it restarted anything.

    `automod.landing.errors_window_s` and `.errors_window_unrestarted_s`, each
    falling back to its constant on anything unreadable and clamped to
    `[ERRORS_WINDOW_FLOOR, ERRORS_WINDOW_CEILING]`. Clamped rather than obeyed
    because both ends fail silently in the same direction the guardian cannot
    see: a zero-length window settles a promotion nothing ever judged and
    advances the LKG anyway, so `last_known_good` would come to mean "last
    landed" while every record still looked healthy.

    `restart` is the promotion's own `restart_needed` verdict, not the
    caller's preference. Read at stamp time rather than at import, so a change
    to config.yaml reaches the next landing without a restart — the promoter
    runs fresh per landing, but `land` imports this module once per process.
    """
    value = _landing_num("errors_window_s" if restart else "errors_window_unrestarted_s",
                         ERRORS_WINDOW if restart else ERRORS_WINDOW_UNRESTARTED)
    return max(ERRORS_WINDOW_FLOOR, min(ERRORS_WINDOW_CEILING, value))


# How many polls in a row must fail to read the pool before `wait_for_rounds`
# stops waiting. At the 10 s poll that is a minute of a backend that cannot
# say, which is a backend the drain's own wait should be judging.
ROUNDS_UNREADABLE_POLLS = 6


def wait_for_rounds(ceiling: float, *, source: str = "autocode",
                    poll: float = 10.0) -> tuple[bool, str]:
    """Wait, WITHOUT pausing or draining, until no `source` job is in flight.

    With more than one round at a time (`workers.sources.autocode.max_inflight`)
    a landing's idle wait is a wait for the OTHER round's turn — a median 24
    minutes, up to an hour. Pausing the pool for that long would stall triage
    and every scheduled task behind a turn that does not need them stopped:
    `current.json` already reads `landing`, so `_loop_is_free` starts no new
    round, and everything else may keep running until the real drain begins.
    Unknown (backend cannot say) returns at once: the drain that follows is
    the wait that matters, and it is the one with the full rules.

    **Never under the automod lock.** `round start` takes the same lock, so a
    turn that has not yet opened its round cannot while a landing holds it —
    and a landing that holds it while waiting for that turn waits on a turn
    it is itself blocking. The first concurrent landing did exactly that
    (2026-09-17, #1204 over #1210's turn, Lloyd's own #1215): the turn got
    `LockHeld` from `automod_start`, gave up after 14 minutes with nothing
    done, and only then did the landing proceed. `round.land` calls this
    before `_land_lock`; its land marker is what keeps a NEW round from
    starting meanwhile (`autocode._rounds_about_to_land`).
    """
    deadline = time.time() + ceiling
    unreadable = 0
    while time.time() < deadline:
        jobs = pool_in_flight()
        if jobs is None:
            # One probe that did not answer is a busy event loop, not a
            # backend that cannot say: `_get` gives up at 5 s, and with four
            # turns streaming a single stall inside a twenty-minute wait was
            # enough to send the landing straight into the drain under its
            # siblings (2026-09-18: seven drains of 12-36 minutes).
            unreadable += 1
            if unreadable >= ROUNDS_UNREADABLE_POLLS:
                return True, (f"pool state unreadable for {unreadable} consecutive polls; "
                              f"going straight to the drain")
        else:
            unreadable = 0
            if not any(j.get("source") == source for j in jobs):
                return True, f"no {source} turn in flight"
        time.sleep(poll)
    return False, f"a {source} turn was still in flight after {ceiling:.0f}s"


def wait_idle(max_wait: float | None = None, *, drain: bool = True,
              pause_pool: bool = True) -> tuple[bool, str]:
    """Drain, then require N consecutive quiet polls before touching anything.

    "Quiet" means all three counters are zero. `harness_runs` is the one that
    was missing: it counts agent loops in flight by ANY caller, and worker jobs
    never enter a session queue. A landing that restarts the backend during a
    ten-minute research job kills it, and the connection failures it logs on
    the way down land inside the window the error-rate detector is watching —
    so the promotion is reverted for the damage its own landing caused.

    **The drain comes first.** Without it, idle is a lottery against the worker
    pool: a research or distill job starts every few minutes, three quiet
    polls in a row never arrive, and the budget runs out. The first landing of
    the unattended era (SM_20260907_233449) spent its whole 900 s watching
    `harness_runs` flicker between 1 and 2 and never drained at all — the
    drain used to be armed only AFTER idle, which is the one moment it is no
    longer needed. Armed here, nothing new starts (chat turns and worker jobs
    both honour it), what is in flight finishes, and zero arrives. It is
    re-armed every `DRAIN_REFRESH_SECONDS` because its TTL is shorter than this
    wait, and released on give-up so a failed landing does not leave the
    backend refusing turns for another three minutes.

    **And the worker pool is paused, not merely drained.** The drain makes a
    dispatched worker job *fail* — each refusal counts an attempt, and three
    attempts poison the job — whereas a paused pool simply starts nothing and
    what is in flight finishes. Only a pause this promoter set is released.

    **The budget does not burn while a worker job is what we are waiting on.**
    With the pool paused nothing new starts, so a job in flight is a bounded
    wait, and giving up on it throws away a gated round to save minutes. On
    2026-09-17 #1204 passed all nine rungs and its landing gave up at 900 s
    with `harness_runs=1` — scheduled task #74, which takes ~37 minutes every
    night and finished 112 seconds later. `max_wait` now counts only the time
    the POOL is empty and the backend is still busy (a chat turn, a leaked
    counter); `automod.landing.idle_hard_max_wait_s` bounds the whole wait.
    """
    global _POOL_PAUSED_BY_US
    max_wait, hard_max = _idle_budget(max_wait)
    if pause_pool and pool_paused() is False and set_pool_paused(True):
        _POOL_PAUSED_BY_US = True
    started = time.time()
    deadline = started + max_wait
    quiet = 0
    # Both read once, before the loop, so a config.yaml edit cannot change the
    # rules underneath a wait already in progress: this polls every 2 s for up
    # to 75 minutes, and `landing_cfg` picks an edit up on its next stat.
    quiet_needed = idle_quiet_polls()
    ttl = drain_ttl()
    busiest = ""
    armed_at = 0.0
    while time.time() < min(deadline, started + hard_max):
        if drain and time.time() - armed_at >= DRAIN_REFRESH_SECONDS:
            set_drain(True, ttl)
            armed_at = time.time()
        status, body = _get(f"{BACKEND}/health")
        if status == 200 and body:
            turns = body.get("turns") or {}
            busy = (turns.get("active", 1) or turns.get("queued", 1)
                    or turns.get("harness_runs", 0))
            # A pool job that is not an agent turn never shows in `turns`: it
            # is a thread running `subprocess.run`, or pure Python. The idle
            # gate learned to count `harness_runs` on 2026-09-06 and stopped
            # there, so a landing restarted the backend under the paired
            # regression check every time one was running — the check died,
            # its pinned qmd daemon did not, and the promotion went unmeasured
            # (8 of 17 measured on 2026-09-18). With the pool paused by this
            # promoter a job in flight is finite, exactly as a harness job is.
            jobs_quiet = pool_in_flight() if (pause_pool and not busy) else None
            if not busy and jobs_quiet:
                quiet = 0
                deadline = time.time() + max_wait
                busiest = "no turn in flight; waiting on pool job(s): " + ", ".join(
                    sorted({str(j.get("source")) for j in jobs_quiet}))
            elif not busy:
                quiet += 1
                if quiet >= quiet_needed:
                    return True, f"idle for {quiet} consecutive polls"
            else:
                quiet = 0  # a turn appearing resets the counter
                busiest = (f"active={turns.get('active')} queued={turns.get('queued')} "
                           f"harness_runs={turns.get('harness_runs')}")
                jobs = pool_in_flight() if pause_pool else None
                if jobs:
                    # A paused pool's job in flight is finite: wait it out.
                    deadline = time.time() + max_wait
                    busiest += " waiting on " + ", ".join(
                        sorted({str(j.get("source")) for j in jobs}))
        else:
            quiet = 0
            # A backend supervisord holds stopped can never go idle: the
            # second `round restart` of 2026-09-24 sat in this loop on the
            # backend the first one had left STOPPED until a human started it
            # by hand. Asked only when /health did not answer at all, so a
            # busy event loop is never the question, and supervisord's own
            # state decides — an unreachable supervisord keeps the old wait.
            state = _program_state("lloyd-backend") if status is None else None
            if state in _DEAD_STATES:
                if drain:
                    set_drain(False)
                release_pool_pause()
                return False, (
                    f"lloyd-backend is {state} in supervisord — a backend that is not "
                    f"running never goes idle, so there is nothing to wait for. Start it: "
                    f"{supervisorctl_hint('start', 'lloyd-backend')} — or restart over it "
                    f"with `round restart --skip-idle`")
        time.sleep(IDLE_POLL_SECONDS)
    if drain:
        set_drain(False)
    release_pool_pause()
    return False, (f"backend never went idle within {time.time() - started:.0f}s"
                   + (f" (last: {busiest})" if busiest else ""))


def count_kg_rows() -> int | None:
    import sqlite3
    from app.paths import production_data_root
    db = production_data_root() / "_pipeline" / "vault-derived" / "kg.sqlite"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            return sum(con.execute(f"SELECT count(*) FROM '{n}'").fetchone()[0] for n in names)
        finally:
            con.close()
    except Exception:
        return None


def count_vault_files(root: Path | None = None) -> int | None:
    """The pre-promotion vault count the guardian will compare against.

    Counted by the guardian's own `vaultwatch.measure`, loaded from the file,
    so `.git/**` is excluded on both sides of the comparison. A private
    `rglob` here counted git's loose objects, and a repack during the
    observation window then read as a 6.7% loss of notes (2026-09-17, #1206).
    """
    root = root or (Path.home() / "obsidian")
    if not root.is_dir():
        return None
    try:
        snap = _guardian_vaultwatch().measure(str(root))
    except Exception:
        return None
    return None if snap is None else snap.total


def _guardian_vaultwatch():
    import importlib.util
    path = LIVE_ROOT / "agent-services" / "guardian" / "vaultwatch.py"
    spec_ = importlib.util.spec_from_file_location("_g_vaultwatch", path)
    mod = importlib.util.module_from_spec(spec_)
    # `@dataclass` resolves its postponed annotations through sys.modules.
    sys.modules[spec_.name] = mod
    spec_.loader.exec_module(mod)
    return mod


def _run(argv: list, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True,
                          timeout=timeout, check=False)


def _apply_service_changes(changed: list[str]) -> list[str]:
    """Make changed service DEFINITIONS take effect. Returns human-readable notes.

    `spec.py` calls the supervisor confs, the systemd units and the guardian
    "protected" — allowed to change, provided the drill passes. But nothing
    ever *applied* such a change: supervisord includes conf.d straight out of
    the repo and needs `reread`/`update` to notice, systemd units are copies
    under ~/.config and the repo edit reached nothing at all, and the guardian
    runs from a pinned snapshot that is only re-staged on a unit restart. So a
    round could pass a full drill, land, be marked healthy, and leave the
    running system on the old definition indefinitely — the change looked
    delivered and was not.

    Never fatal to the promotion: the code is already live and verified by the
    time this runs, and a failure to reload a conf is worth an alert, not a
    revert.
    """
    notes: list[str] = []
    touched_supervisor = any(p.startswith("agent-services/supervisor/") for p in changed)
    touched_units = [p for p in changed if p.startswith("agent-services/systemd/")]
    touched_guardian = any(p.startswith("agent-services/guardian/") for p in changed)

    if touched_supervisor and SUPERVISORCTL.exists():
        for verb in ("reread", "update"):
            r = _run([SUPERVISORCTL, "-c", SUPERVISORD_CONF, verb], timeout=120)
            notes.append(f"supervisorctl {verb}: "
                         f"{'ok' if r.returncode == 0 else r.stderr.strip()[:120]}")

    for rel in touched_units:
        src = LIVE_ROOT / rel
        if not src.is_file():
            continue
        try:
            SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, SYSTEMD_USER_DIR / src.name)
            notes.append(f"installed {src.name}")
        except OSError as exc:
            notes.append(f"FAILED to install {src.name}: {exc}")
    if touched_units:
        r = _run(["systemctl", "--user", "daemon-reload"])
        notes.append("daemon-reload: "
                     f"{'ok' if r.returncode == 0 else r.stderr.strip()[:120]}")

    # The guardian runs a pinned snapshot, re-staged by ExecStartPre. Without
    # this a landed guardian change is inert until something else restarts the
    # unit. Staging declines a candidate that does not compile or fails its own
    # selftest, so the worst case is that the previous snapshot keeps running.
    if touched_guardian or any(p.endswith("lloyd-guardian.service") for p in touched_units):
        r = _run(["systemctl", "--user", "restart", "lloyd-guardian"], timeout=120)
        notes.append("guardian restarted: "
                     f"{'ok' if r.returncode == 0 else r.stderr.strip()[:120]}")
    return notes


def _start_regression_runner() -> str:
    """Start the detached regression runner for what just landed. Never raises.

    The check used to wait for the worker pool to offer it, and the pool holds
    it back while a round is in flight — which, with rounds always in flight,
    left only the seconds after a restart. Started from here it begins the
    moment the landing is verified, in its own session, and the next landing
    cannot kill it (`workers/sources/automod_regression.py::execute`). The
    pool source stays as the safety net for a runner that could not start.
    """
    try:
        python = LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
        pid = S.spawn_detached([python, "-m", "scripts.automod.regression_runner", "run"],
                               S.STATE_DIR / "regression.log", cwd=LIVE_ROOT)
        return f"started (pid {pid})"
    except Exception as exc:  # noqa: BLE001 — a measurement is never the landing
        return f"not started: {exc!r}"[:200]


def promotion_announcement(title: str, n_files: int,
                           window_s: float | None = None) -> tuple[str, str]:
    """The toast head and body for a landing. Pure, so it can be pinned.

    Never the round id: it is in the `promoted` ledger row for anyone who
    needs it, and read aloud it is a date one digit at a time. The title is
    the item's name — the one thing a person in the room can act on.

    `window_s` is the window this promotion actually got, which since
    2026-09-20 is one of two and is read from config. Formatting the constant
    instead is how the announcement comes to state a number nothing used —
    the same defect as the `errors_window_s` key that sat unread in
    config.yaml. Always minutes, one decimal only when it does not divide
    evenly: this line is read aloud as well as shown, and mixed units across
    consecutive landings are worse to listen to than a fraction.
    """
    seconds = float(ERRORS_WINDOW if window_s is None else window_s)
    watching = (f"{int(seconds // 60)} minutes" if seconds % 60 == 0
                else f"{seconds / 60:.1f} minutes")
    head = f"Landed: {title}" if title else "Landed a change"
    body = f"{n_files} file{'' if n_files == 1 else 's'} changed. Watching for {watching}."
    return head, body


def _announce_promoted(round_id: str, commit: str, changed: list, title: str = "",
                       window_s: float | None = None) -> None:
    """Say out loud that the loop just landed code on itself.

    Until now the self-modification loop only ever spoke when it *failed*:
    every notify-send in the tree hung off a guardian alert. A loop that can
    rewrite the running system in the background and is silent when it works
    is the wrong way round — the successful landings are the ones nobody is
    watching a terminal for.

    Routed through the guardian's `Notifier.announce` rather than a private
    notify-send so it shares the one fan-out, and guarded end to end: an
    announcement must never be able to fail a promotion that already
    succeeded and is being observed.
    """
    head, body = promotion_announcement(title, len(changed), window_s)
    announce(head, body)


def announce(head: str, body: str) -> None:
    """News through the guardian's one fan-out (journal, toast, voice), never
    an `alert` — no ledger row, no backlog task. Guarded end to end: an
    announcement must never fail the thing it announces."""
    try:
        import sys
        gdir = Path(__file__).resolve().parents[2] / "agent-services" / "guardian"
        if not (gdir / "notify.py").is_file():
            return
        if str(gdir) not in sys.path:
            sys.path.insert(0, str(gdir))
        import gstate, notify as notify_mod, policy
        notifier = notify_mod.Notifier(
            ledger=gstate.AutomodState(Path(policy.AUTOMOD_STATE)).ledger,
            state_dir=Path(policy.GUARDIAN_STATE),
            vault_root=policy.VAULT_ROOT,
            voice_window=policy.VOICE_REPEAT_SECONDS,
        )
        notifier.announce(head, body)
    except Exception:
        pass


def vault_commits_for(round_id: str) -> list[str]:
    """Vault shas this round landed, newest last.

    A `mixed` backlog item lands its vault half through `automod_vault_land`
    and its code half through this promoter, and until now the two halves were
    recorded in different places with nothing joining them. That matters at
    rollback: reverting #377's code commit alone would restore the Python
    `ANTICOMPLIANCE_DIRECTIVE` constant while `SOUL.md` kept its condensed
    section — recreating the exact doubled-frame state (#465) the round
    existed to remove, and doing it silently.

    Joined on time rather than on an id, because `vault_land` records the
    backlog `item_id` and not the round: the ledger's own `round_start` event
    is the only anchor both halves share. Events are ordered, so "after this
    round opened" is exactly the window, and a round that landed no vault
    change returns [].
    """
    events = S.read_events(limit=500)
    start_ts = None
    start_idx = -1
    for i, e in enumerate(events):
        if e.get("event") == "round_start" and e.get("round_id") == round_id:
            start_ts, start_idx = e.get("ts"), i
    if start_ts is None:
        return []
    # Closed at the next round to open, not left running to now. Only one
    # round holds the lock at a time, so the next `round_start` is this
    # round's end whether it landed or was abandoned — without that bound an
    # aborted round retroactively claims the next round's vault commit, which
    # is precisely backwards for a field a rollback acts on.
    end_ts = next((e.get("ts") for e in events[start_idx + 1:]
                   if e.get("event") == "round_start"), None)
    return [e["commit"] for e in events
            if e.get("event") == "vault_land" and e.get("ok") and e.get("commit")
            and (e.get("ts") or 0) >= start_ts
            and (end_ts is None or (e.get("ts") or 0) < end_ts)]


# A landing can be refused after every rung passed, and until 2026-09-09 that
# refusal was invisible to the backlog: `backlog.implement_outcomes` read only
# `gate` events, so a round that lost a race with `main` at landing looked
# exactly like one that landed. It also spent the item. `land_failed` is the
# verdict event for that case; `external` says the cause was the tree, not the
# diff, and the item keeps its attempt.
def _post_json(url: str, payload: dict, *, headers: dict | None = None,
               timeout: float = 10.0) -> tuple[int | None, dict | None]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8", "replace") or "null")
            return r.status, body if isinstance(body, dict) else None
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


# Non-Python paths no running process holds: read fresh by whatever uses them,
# or by nothing at runtime at all. An allowlist, so an unrecognised file is a
# reason to restart. `.md` anywhere is the rule `bless` already used.
INERT_PREFIXES = ("tests/", "architecture/", "eval/")
# Python here is never "not loaded, so inert": the guardian runs from a staged
# snapshot and the service scripts belong to supervisord programs, and both
# reach the running system through `_apply_service_changes`, not an import.
ALWAYS_RESTART_PREFIXES = ("agent-services/",)


def _loaded_in(name: str, url: str, paths: list[str], headers: dict | None = None,
               timeout: float = 10.0):
    """The members of `paths` a server has loaded, or None when it cannot say."""
    status, body = _post_json(url, {"paths": paths}, headers=headers, timeout=timeout)
    if status != 200 or not isinstance(body, dict) or not isinstance(body.get("loaded"), list):
        return None
    return [str(p) for p in body["loaded"]]


def restart_needed(changed: list[str], *, in_backend: bool = False) -> tuple[bool, str]:
    """`(a landing of these paths must restart the services, why)`.

    A landing drains the backend, waits for every sibling round's turn to end
    and restarts two services so that what landed is what runs. When no
    changed file is one either process has loaded, what runs is unchanged by
    the merge and all of that buys nothing: on the night of 2026-09-18 nine of
    nineteen promotions changed only tests, `eval/`, docs and scripts that run
    fresh per invocation, and each still waited on up to three other turns.

    Python is judged by asking the two processes (`app/loaded_paths.py`), not
    by directory: the backend imports `scripts/automod/**` and a list would
    not have known. Everything else is an allowlist. **Fails closed** at every
    step — a server that does not answer, an answer of the wrong shape, a path
    nothing here recognises, the kill switch (`automod.landing.skip_restart`)
    — because a needless restart costs minutes and a skipped one that was
    needed leaves `main` and the running code disagreeing until someone
    notices.

    `in_backend`: the caller IS the backend (the implement source, deciding
    whether a gated round holds new rounds back), on its event loop. It reads
    its own `sys.modules` instead of requesting itself over HTTP — a request
    the blocked loop could never answer — and gives the aggregator two
    seconds rather than ten.
    """
    if not bool(S.landing_cfg(LIVE_ROOT).get("skip_restart", True)):
        return True, "automod.landing.skip_restart is off"
    if not changed:
        return True, "no changed paths named"
    python: list[str] = []
    for p in changed:
        if p.startswith(ALWAYS_RESTART_PREFIXES):
            return True, f"{p} is a service definition or guardian file"
        if p.endswith(".py"):
            python.append(p)
        elif not (p.endswith(".md") or p.startswith(INERT_PREFIXES)):
            return True, f"{p} is not a path known to be inert"
    if python:
        try:
            from agent_mcp import aggregator_auth
            mcp_headers = aggregator_auth.auth_headers()
        except Exception:  # noqa: BLE001
            mcp_headers = {}
        for name, url, headers in (
                ("backend", f"{BACKEND}/api/automod/loaded", None),
                ("aggregator", MCP_HEALTH.rsplit("/health", 1)[0] + "/loaded", mcp_headers)):
            if in_backend and name == "backend":
                try:
                    from app.loaded_paths import loaded
                    hits = loaded(python)
                except Exception:  # noqa: BLE001
                    hits = None
            else:
                hits = _loaded_in(name, url, python, headers,
                                  timeout=2.0 if in_backend else 10.0)
            if hits is None:
                return True, f"the {name} could not say which modules it has loaded"
            if hits:
                return True, (f"the {name} has loaded {hits[0]}"
                              + (f" and {len(hits) - 1} more" if len(hits) > 1 else ""))
    return False, (f"none of the {len(changed)} changed path(s) is loaded by the backend "
                   f"or the aggregator")


def squash_enabled() -> bool:
    """`automod.landing.squash`, default on. Off: every commit of the round's
    branch is fast-forwarded onto `main`, as before 2026-09-17."""
    return bool(S.landing_cfg(LIVE_ROOT).get("squash", True))


def squash_message(round_id: str, title: str, worktree: Path, base: str) -> str:
    """The one commit's message: the round's title, then what it was made of.

    The child subjects are kept in the body because they are the only
    narrative of how the round got there once the branch is gone; the shas
    are reachable under `refs/automod/rounds/<round>`.
    """
    log = W.git(worktree, "log", "--reverse", "--format=%h %s", f"{base}..HEAD")
    children = [ln for ln in (log.stdout or "").splitlines() if ln.strip()]
    trailers = W.git(worktree, "log", "--format=%(trailers:key=Co-Authored-By,unfold)", f"{base}..HEAD")
    coauthors = list(dict.fromkeys(ln.strip() for ln in (trailers.stdout or "").splitlines() if ln.strip()))
    subject = " ".join((title or "").split())[:140] or (children[-1].split(" ", 1)[-1] if children else round_id)
    body = [subject, "", f"Round {round_id}, squashed at landing from {len(children)} commit(s)",
            f"(kept at refs/automod/rounds/{round_id}):", ""]
    body += [f"  {c}" for c in children]
    if coauthors:
        body += [""] + coauthors
    return "\n".join(body) + "\n"


def _land_failed(round_id: str, why: str, *, external: bool, **extra) -> None:
    S.append_event({"event": "land_failed", "round_id": round_id, "ok": False,
                    "external_blocker": bool(external), "detail": why[:500], **extra})
    raise PromoteError(why)


# The chamber (`automod.chamber`). The next round may run its turn and gate
# while the previous promotion is under observation; only the landing needs
# the window closed. So a landing that finds a promotion observed waits for
# it to settle — up to the window plus slack — instead of refusing at once
# and throwing away a gated round. A landing is written `current.json` with
# its own window only after the restart verifies, so the whole window can
# still be ahead of it.
SETTLE_SLACK = 120.0
SETTLE_POLL_SECONDS = 10.0
# How many promotions in a row one landing will queue behind, each with a
# full window. Four is an hour; past that something else is wrong.
SETTLE_QUEUE_CAP = 4


def settle_max_wait() -> float:
    """The longest one landing waits for the promotion ahead of it to settle.

    The *larger* of the two windows, never the waiting round's own: what it is
    waiting on is somebody else's promotion, and the window belongs to that
    one. A landing that changes no loaded file would otherwise be held to the
    120 s of its own window while waiting out a restarted promotion's 450, and
    be refused with six minutes still to run.
    """
    return max(errors_window(True), errors_window(False)) + SETTLE_SLACK


def _settled(commit: str) -> bool:
    """Whether the guardian recorded `settled` for this commit."""
    return any(e.get("event") == "settled" and str(e.get("commit") or "") == commit
               for e in S.read_events(limit=400))


def wait_for_settle(max_wait: float | None = None, *, poll: float | None = None,
                    round_id: str = "", observed: dict | None = None) -> dict | None:
    """Wait while a promotion is recorded in `current.json`; None once it
    clears, `PromoteError` if it never does. `promote` keeps its own "still
    under observation" refusal behind this, for any caller that did not wait.

    Every way this ends without a landing is someone else's promotion, not
    this round's change, so with a `round_id` it is recorded as an external
    `land_failed` and the item keeps its attempt — a gated round refused
    because the promotion ahead of it was rolled back is not a verdict.

    Called by `round.land` BEFORE it takes the automod lock and before
    `wait_idle` pauses the pool, so neither a round start nor the worker pool
    is held up for the quarter hour this can take. After the wait, the three
    things that can change underneath it are asked again: a halt, BROKEN, and
    a pending rollback request — the observed promotion may have been the one
    rolled back, and landing on top of a revert in flight is the one case a
    wait must never end in. A moved `main` needs nothing here: `promote`
    compares live HEAD with the gated base and re-gates (`_regate_after_move`).
    """
    max_wait = settle_max_wait() if max_wait is None else float(max_wait)
    poll = SETTLE_POLL_SECONDS if poll is None else float(poll)
    started = time.monotonic()
    deadline = started + max_wait
    # `observed` is the record the caller decided to wait on. Read again here
    # regardless, and remember the commit from whichever saw it: a promotion
    # that clears between the caller's read and this one must still be proved
    # settled, not assumed.
    waited_on = str((observed or {}).get("commit") or "")
    observed = S.read_current()
    waited_on = waited_on or str((observed or {}).get("commit") or "")
    # Every promotion waited behind, in order. Landings queue: while this one
    # waits for A, another gated round is waiting too, and when A settles one
    # of them lands B within seconds. The one that lost that race is now
    # behind B, which has its whole window ahead of it — and used to be held
    # to what was left of A's. On 2026-09-19 `854144ab` settled at 21:45:25,
    # `0c15ba3d` was promoted at 21:45:32, and at 21:47:22 a green, gated round
    # was refused "0c15ba3d is still under observation after waiting 17 min"
    # and reaped. Each new promotion restarts the clock, `SETTLE_QUEUE_CAP`
    # times at most; every one of them must have settled, not merely cleared.
    behind = [waited_on] if waited_on else []
    while observed and time.monotonic() < deadline:
        time.sleep(poll)
        observed = S.read_current()
        now_on = str((observed or {}).get("commit") or "")
        if now_on and now_on not in behind:
            if len(behind) >= SETTLE_QUEUE_CAP:
                break
            behind.append(now_on)
            deadline = time.monotonic() + max_wait
    why = ""
    if S.is_halted():
        why = f"promotions are halted: {S.HALTED_PATH}"
    elif S.is_broken():
        why = f"guardian is in a BROKEN state: {S.BROKEN_PATH}"
    elif S.read_rollback_request():
        why = "a rollback request is pending — not landing on top of a revert"
    elif observed:
        why = (f"{str(observed.get('commit'))[:8]} is still under observation "
               f"({observed.get('state')}) after waiting {max_wait / 60:.0f} min for it to settle"
               + (f", behind {len(behind)} promotion(s) in a row" if len(behind) > 1 else ""))
    else:
        # `current.json` also clears when the guardian ROLLS BACK: it deletes
        # a rollback request the moment it reads one, and its own error-window
        # rollbacks never write one, so an empty request file proves nothing.
        # Only a `settled` row for a commit waited on says it survived.
        unsettled = [c for c in behind if not _settled(c)]
        if unsettled:
            why = (f"{unsettled[0][:8]} left observation without settling (rolled back?) — "
                   f"not landing a round that ran on top of it")
    waited_s = round(time.monotonic() - started, 1)
    if why:
        if round_id:
            _land_failed(round_id, why, external=True, waited_for_settle=True,
                         waited_s=waited_s, behind=len(behind))
        raise PromoteError(why)
    # A successful settle wait recorded NOTHING until 2026-09-20, so the cost
    # this whole change is about was invisible: `land_wait_rounds` has its own
    # row and `wait_for_settle` had none, and the only way to see the window
    # serializing landings was to difference `promoted` stamps by hand. Mirrors
    # that row's shape so the scorecard can read them side by side.
    if round_id:
        S.append_event({
            "event": "land_wait_settle", "round_id": round_id, "ok": True,
            "waited_s": waited_s, "behind": len(behind),
            "windows": {"restarted": errors_window(True),
                        "unrestarted": errors_window(False)},
            "detail": (f"waited {waited_s:.0f}s for {len(behind)} promotion(s) to settle"
                       if behind else "nothing under observation by the first read")})
    return None


def _regate_after_move(round_id: str, worktree: Path, live: Path, base: str,
                       live_head: str) -> tuple[str, str, dict]:
    """`main` moved since the gate ran. Rebase the round onto it and retest.

    Not a second implementation of the rebase: `Gate.rung_preflight` already
    does it, names conflicts, and fails closed. Running the gate with the OLD
    base is what triggers that path, and the report that comes back carries
    the new base and the new head. Every rung then judged the round's change
    on top of what landed — which is the only build that was ever going to be
    live, and the one nothing had tested until now.

    Returns `(base, head, gate_report)`. Persists both records the next reader
    depends on: `gate.json`, which `land` reads its head from, and the run
    spec's base, which the next `run_gate` reads its diff from.
    """
    from scripts.automod import gate as G   # lazy: gate → canary → ports; not needed elsewhere here
    # The item, so the chase's gate can run its review rung. Without it the
    # rung records `review: skipped` and a landing can be judged on a review
    # of a commit that no longer exists — the rebase moved the head. The
    # review's patch-id reuse then answers the common case (a clean rebase of
    # an identical diff) from the ledger without a second grading turn.
    item_id = None
    try:
        import yaml
        spec = yaml.safe_load(
            (S.ROUNDS_DIR / round_id / "run_spec.yaml").read_text()) or {}
        item_id = (spec.get("item") or {}).get("id")
    except Exception as exc:  # noqa: BLE001 — a missing spec is not fatal here
        print(f"[warn] could not read the run spec for {round_id}: {exc}")
    report = G.Gate(round_id, Path(worktree), base, live_root=live,
                    item_id=item_id).run()
    rep = report.to_dict()
    S.write_gate_report(round_id, rep)
    if report.base != base:
        S.update_run_spec_base(round_id, report.base)
    if not report.ok:
        failed = next((r for r in rep.get("rungs", []) if not r.get("ok")), {})
        _land_failed(round_id,
                     f"main moved to {live_head[:8]} since the gate ran; rebased and "
                     f"retested, and the `{failed.get('name')}` rung failed: "
                     f"{failed.get('detail', '')[:300]}",
                     external=True, rung=failed.get("name"), moved_to=live_head)
    return report.base, report.head, rep


def swap_candidate_venv(worktree: Path, live: Path, gate_report: dict | None,
                        current: dict | None = None) -> bool:
    """Install the venv the gate actually built, in place of the live one.

    Returns True when a candidate was swapped in, False when the gate named
    none — which is every requirements-unchanged landing, the common case.
    Raises `PromoteError` when the gate DID name one and it cannot be installed
    safely: code whose dependencies were verified against a candidate must not
    go live against the old venv, and the caller's except path rolls the merge
    back, which is what the old crash did too.

    The trigger used to be a filesystem probe — `Path(worktree)/".venvs"/"lloyd"
    .exists()` — not the gate's report. Those agree only while a `.venvs/lloyd`
    inside a worktree can have exactly one cause: `Gate.rung_venv` cloning the
    live venv and recording it as `GateReport.venv`. #692's proposed fix (symlink
    the whole `.venvs` into the round so the documented relative interpreter path
    resolves) breaks that agreement for every round at once: the probe reads True
    through the symlink, the promoter renames the LIVE venv aside as `lloyd.prev`,
    and its second rename raises `FileNotFoundError` — with `git merge --ff-only`
    already having moved `main`. Measured in a scratch tree: probe True, live venv
    renamed aside, `rename RAISED: [Errno 2] No such file or directory`. A guard
    acting on a number it cannot re-measure, and the expensive kind: it spends the
    item's one unattended attempt on a rollback of a build that was fine.

    So: the report decides, and the named path must be a real directory inside
    the worktree that does not resolve into the live tree. A `.venvs` shared
    with, or symlinked into, the live checkout can therefore never be the thing
    that renames the live venv aside.
    """
    named = str((gate_report or {}).get("venv") or "").strip()
    if not named:
        return False

    import os
    clone, live_venv = Path(named), Path(live) / ".venvs" / "lloyd"
    wt_root, live_root = Path(worktree).resolve(), Path(live).resolve()
    resolved = clone.resolve()
    inside_worktree = clone.is_dir() and resolved.is_relative_to(wt_root)
    if not inside_worktree or resolved.is_relative_to(live_root):
        raise PromoteError(
            f"the gate named candidate venv {named}, but it is not an isolated directory "
            f"inside the worktree (resolves to {resolved}, worktree {wt_root}, live "
            f"{live_root}) — refusing to touch {live_venv}")
    if not live_venv.is_dir():
        raise PromoteError(f"cannot swap in candidate venv {named}: no live venv at {live_venv}")

    prev = live_venv.parent / "lloyd.prev"
    if prev.exists():
        raise PromoteError(f"cannot swap in candidate venv {named}: {prev} already exists")

    os.rename(live_venv, prev)
    try:
        os.rename(clone, live_venv)
    except OSError as exc:
        os.rename(prev, live_venv)   # never leave the live tree without its venv
        raise PromoteError(f"candidate venv {named} could not be moved into place: {exc}; "
                           "the live venv was put back")
    if current is not None:
        current["venv_swapped"] = True
    return True


def promote(round_id: str, worktree: Path, base: str, *,
            gate_report: dict | None = None, dry_run: bool = False) -> dict:
    live = LIVE_ROOT
    head = W.head(Path(worktree))
    if not head:
        raise PromoteError("cannot read the candidate HEAD")

    if S.is_halted():
        raise PromoteError(f"promotions are halted: {S.HALTED_PATH}")
    if S.is_broken():
        raise PromoteError(f"guardian is in a BROKEN state: {S.BROKEN_PATH}")

    # The gate judged a specific commit. Anything committed into the worktree
    # afterwards would land completely ungated: `land` passes the base along
    # but the candidate HEAD was re-read from the worktree here, so an extra
    # commit made after a passing gate was indistinguishable from the one that
    # passed it.
    gate_head = (gate_report or {}).get("head")
    if gate_head and gate_head != head:
        raise PromoteError(
            f"worktree HEAD moved since the gate ran ({head[:8]} != gated "
            f"{gate_head[:8]}) — re-gate before landing")

    # The land train (`automod.landing.defer_restart`, and only while the
    # STAGED guardian can judge a batch): a landing that need not restart
    # anything NOW is merged and left for the flush. With the train off every
    # landing is eager and what follows is the landing path unchanged; with it
    # on, an eager one also carries whatever the train is holding (`batch`).
    train = restart_deferred()[0]
    if train and not landing_is_eager(W.changed_paths(Path(worktree), base), gate_report):
        return merge_round(round_id, worktree, base, gate_report=gate_report,
                           dry_run=dry_run)

    # One promotion under observation at a time. A second landing overwrote
    # `current.json`, so the first promotion never settled, never advanced the
    # LKG, and — worse — the new record's rollback target became a commit that
    # had never survived a window, quietly breaking the invariant the whole
    # promoter/guardian split exists to guarantee.
    observed = S.read_current()
    if observed and observed.get("state") in ("landing", "observing"):
        until = float(observed.get("errors_until_ts") or 0)
        left = max(0.0, until - time.time())
        raise PromoteError(
            f"{str(observed.get('commit'))[:8]} is still under observation "
            f"({observed.get('state')}, {left / 60:.1f} min left) — a promotion must "
            "settle before the next one lands")

    changed_preview = W.changed_paths(Path(worktree), base)
    tree_hash = S.changed_tree_hash(worktree, head, changed_preview)
    if S.is_denied(commit=head, tree_hash=tree_hash):
        raise PromoteError(
            f"{head[:8]} is on the rollback denylist"
            + (" (matched by content, not SHA — this change was reverted before "
               "and has been re-derived)" if tree_hash and not S.is_denied(commit=head)
               else ""))
    live_head = subprocess.run(["git", "-C", str(live), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
    if live_head != base:
        # The tree is shared. Rebase and retest rather than refuse — the
        # gate's own preflight does the rebase; see `_regate_after_move`.
        base, head, gate_report = _regate_after_move(round_id, Path(worktree), live,
                                                     base, live_head)
        changed_preview = W.changed_paths(Path(worktree), base)
        tree_hash = S.changed_tree_hash(worktree, head, changed_preview)
        # The denylist was checked against the head that existed before the
        # rebase. Content is what it matches on, and a clean rebase over a
        # same-file, non-overlapping commit changes that content — so check
        # again against the head that will actually land.
        if S.is_denied(commit=head, tree_hash=tree_hash):
            _land_failed(round_id, f"{head[:8]} (rebased) is on the rollback denylist",
                         external=False)

    changed = W.changed_paths(Path(worktree), base)
    # The item's name, for the toast, the spoken line and the guardian's
    # rollback alert — the guardian is stdlib-only and reads it off this record.
    from scripts.automod import backlog as B
    title = B.work_title_for_round(S.LEDGER_PATH, round_id)

    # Uncommitted edits in production are tolerated when they are disjoint
    # from this diff: `merge --ff-only` never touches a file it is not
    # merging, so they stay exactly where they are, in the editor they are
    # open in. Overlap is the hazard — two writers on one file — and git
    # would refuse the merge anyway; refusing here says which files, before
    # the pool is paused and the drain armed. Still not the round's fault.
    live_dirty = W.dirty_paths(live)
    overlap = sorted(set(live_dirty) & set(changed))
    if overlap:
        # Same contract as the gate's preflight refusal, same reason the wording
        # is pinned by a test (#1038). This string is what a person reads in the
        # ledger days later to find the other author, so it names every
        # overlapping path and asks for a report. It must never suggest getting
        # the dirt out of the way: that route is the live checkout's one global
        # LIFO working-tree stack, shared by every author, from which a round
        # implementing an unrelated item already popped #573's only recovered
        # copy of a 136-line diff on 2026-09-11.
        _land_failed(round_id,
                     f"live tree has uncommitted edits in paths this round also changes: "
                     f"{overlap} — two writers on one file. Report the paths and who is "
                     f"editing them; the live edit is not yours to commit and not yours "
                     f"to move out of the tree, and nothing here needs a clean live "
                     f"tree to land again",
                     external=True, overlap=overlap)

    result: dict = {"round_id": round_id, "commit": head, "parent": live_head,
                    "changed_paths": changed, "dry_run": dry_run,
                    "live_dirty_paths": live_dirty[:20], "title": title}
    if dry_run:
        result["would_promote"] = True
        return result

    # An eager landing on the train restarts for everything merged before it,
    # so its record is the batch: `_write_record` stamps `commits` and the
    # oldest entry's parent as the rollback target. Empty with the train off,
    # and then every write below is exactly `S.write_verified` as before.
    batch = _live_pending(live) if train else []
    restart_began = False
    idle_waited_s = 0.0

    # ── the rollback point, verified before anything moves ─────────────
    current = {
        "schema": 1,
        "round_id": round_id,
        "title": title,
        "commit": head,
        "parent": live_head,
        "rollback_target": live_head,
        "branch": f"automod/{round_id}",
        "state": "landing",
        "landed_at": None,
        "landed_ts": None,
        # Deliberately not started yet. The idle gate can wait up to 15
        # minutes for a turn to finish, and starting the observation clock
        # here would burn most of the window before the code is even live —
        # the guardian would then settle a promotion it had barely watched.
        # Both are set after the restart verifies, below.
        "errors_until_ts": None,
        "changed_paths": changed,
        # What was uncommitted in production when this landed. The guardian's
        # observation window will blame errors on the promotion; if a human's
        # half-finished edit was live in the same process, this is how a
        # reader tells the two apart.
        "live_dirty_paths": live_dirty[:20],
        "vault_commits": vault_commits_for(round_id),
        "tree_hash": tree_hash,
        "venv_swapped": False,
        "touched_guardian": any(p.startswith("agent-services/guardian/") for p in changed),
        "kg_rows": count_kg_rows(),
        "vault_files": count_vault_files(),
        "gate": (gate_report or {}).get("rungs"),
    }
    _, body = _get(f"{BACKEND}/health")
    current["boot_id"] = (body or {}).get("boot_id")
    _write_record(current, batch)   # raises unless it round-trips

    # ── idle gate + drain ──────────────────────────────────────────────
    # The other round's turn was waited out by `round.land` BEFORE it took the
    # automod lock (`wait_for_rounds`): under the lock that wait deadlocks.
    #
    # Unless nothing that runs would change (`restart_needed`): then there is
    # no restart to protect a turn from, so nothing is paused, drained or
    # waited for, and the merge below is the whole landing.
    restart, restart_why = restart_needed(changed)
    if batch and not restart and any(e.get("restart") for e in batch):
        # What the train holds needs the restart even if this landing alone would not.
        restart, restart_why = True, "the train holds landings that need a restart"
    current["restart"] = result["restart"] = restart
    result["restart_why"] = restart_why
    idle_from = time.time()
    ok, why = wait_idle() if restart else (True, restart_why)
    idle_waited_s = time.time() - idle_from
    if not ok:
        S.clear_current()
        # A landing that never got the backend idle is the infrastructure's
        # failure, not the round's, and it has to be SAID on the ledger: a
        # bare PromoteError here left the finished implement row claiming
        # `landed: true` with no promotion behind it, `implement_outcomes`
        # read that as a spent attempt, and on 2026-09-16 nine gate-passed
        # rounds were parked, re-triaged and re-implemented from scratch
        # while their commits sat on kept branches. As a `land_failed` with
        # `external_blocker`, the item keeps its attempt and its branch.
        _land_failed(round_id, why, external=True, waited_idle=True)
    if restart:
        set_drain(True, drain_ttl())
    merged = False
    try:
        status, body = _get(f"{BACKEND}/health")
        turns = (body or {}).get("turns") or {}
        if restart and (turns.get("active") or turns.get("queued") or turns.get("harness_runs")):
            raise PromoteError(f"a turn started during the drain handshake: {turns}")

        # The idle wait can take fifteen minutes, and the human is still
        # committing. One more chase, inside the drain so nothing starts
        # underneath the retest; re-arm the drain afterwards because a gate
        # run can outlast its TTL.
        live_now = subprocess.run(["git", "-C", str(live), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
        if live_now != base:
            base, head, gate_report = _regate_after_move(round_id, Path(worktree), live,
                                                         base, live_now)
            if restart:
                set_drain(True, drain_ttl())
            live_head = live_now
            changed = W.changed_paths(Path(worktree), base)
            tree_hash = S.changed_tree_hash(worktree, head, changed)
            if S.is_denied(commit=head, tree_hash=tree_hash):
                _land_failed(round_id, f"{head[:8]} (rebased) is on the rollback denylist",
                             external=False)
            current.update({"commit": head, "parent": live_head, "rollback_target": live_head,
                            "changed_paths": changed, "tree_hash": tree_hash,
                            "gate": gate_report.get("rungs")})
            result.update({"commit": head, "parent": live_head, "changed_paths": changed})
            _write_record(current, batch)

        # ── one commit per landing ─────────────────────────────────────
        # After the last gate, before the fast-forward, and only ever to a
        # commit whose tree is the gated one (`W.squash_onto` proves it). The
        # record is rewritten BEFORE the merge: `current.json` is what the
        # guardian rolls back by, and `/health.commit` is checked against it.
        if squash_enabled():
            squashed, note = W.squash_onto(
                Path(worktree), live_head,
                squash_message(round_id, title, Path(worktree), live_head),
                keep_ref=f"refs/automod/rounds/{round_id}")
            if squashed:
                result["squashed_from"] = current["squashed_from"] = head
                head = squashed
                current["commit"] = result["commit"] = head
                _write_record(current, batch)
            result["squash"] = note

        # ── land ───────────────────────────────────────────────────────
        if restart:
            S.set_pause(RESTART_LEASE)   # the guardian must not read our own restart as a crash
        merge = subprocess.run(
            ["git", "-C", str(live), "merge", "--ff-only", f"automod/{round_id}"],
            capture_output=True, text=True)
        if merge.returncode != 0:
            # Chased twice and still not a fast-forward: `main` is moving faster
            # than the loop can retest. Say so and stop; the item comes back.
            _land_failed(round_id,
                         f"fast-forward failed after rebasing and retesting: "
                         f"{merge.stderr.strip()[:300]}",
                         external=True)
        merged = True

        if not restart:
            # Asked again now that the files have moved: a module imported
            # between the first answer and the merge may have loaded the old
            # file. Then this is an ordinary landing after all, late — the
            # drain and the restart below, with the tree already in place.
            again, again_why = restart_needed(changed)
            if again:
                restart = current["restart"] = result["restart"] = True
                result["restart_why"] = f"after the merge: {again_why}"
                _write_record(current, batch)
                ok, why = wait_idle()
                if not ok:
                    raise PromoteError(f"{again_why}, and {why}")
                set_drain(True, drain_ttl())
                S.set_pause(RESTART_LEASE)

        # The gate's report, not a path probe: see `swap_candidate_venv`.
        if swap_candidate_venv(Path(worktree), live, gate_report, current):
            _write_record(current, batch)

        # Service definitions the diff changed must reach the running system
        # BEFORE the restart, or the restart re-reads the old ones.
        service_notes = _apply_service_changes(changed)

        # ── restart, MCP first ─────────────────────────────────────────
        restart_began = restart
        for program, health in ((("lloyd-mcp", MCP_HEALTH),
                                 ("lloyd-backend", f"{BACKEND}/health")) if restart else ()):
            # Refresh the lease before each leg. The lease is 120s and this
            # loop can legitimately spend 90s per service waiting on health,
            # so a single lease taken before the merge could expire mid-restart
            # and hand the guardian its own deploy to judge.
            S.set_pause(RESTART_LEASE)
            ok, msg = _restart_program(program)
            if not ok:
                raise PromoteError(f"restart {program} failed: {msg}")
            if not _wait_health(health, 90.0):
                raise PromoteError(f"{program} never became healthy after restart")

        # ── prove the RUNNING code changed ─────────────────────────────
        # Polled, not probed once: on 2026-09-16 16:31Z a single `_get`
        # right after the health wait came back with no body, the check
        # read `commit None`, and a landing that had merged, restarted and
        # booted the new code was rolled back as `promote_failed` — the
        # loop's every rollback has been a false positive, and this was one
        # more. A body that names a commit is the answer; the wrong commit
        # is a real failure; no answer within the budget is reported as such.
        if restart:
            body = _wait_for_commit(f"{BACKEND}/health", VERIFY_COMMIT_BUDGET)
            actual = (body or {}).get("commit")
            if actual != head:
                raise PromoteError(f"backend reports commit {actual}, expected {head} "
                                   "— the restart did not pick up the new code")
            if current.get("boot_id") and (body or {}).get("boot_id") == current["boot_id"]:
                raise PromoteError("backend boot_id unchanged — the process was never replaced")
        else:
            # No process was replaced, so `/health.commit` still names the boot
            # commit and proves nothing here. What a landing without a restart
            # has to prove is that the tree moved and the services it did not
            # touch are still answering.
            on_disk = subprocess.run(["git", "-C", str(live), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
            if on_disk != head:
                raise PromoteError(f"live HEAD is {on_disk[:8]}, expected {head[:8]} after the merge")
            status, body = _get(f"{BACKEND}/health")
            if status != 200:
                raise PromoteError(f"backend /health answered {status} after a landing that "
                                   "restarted nothing")

        if any(p.startswith("web/") for p in changed):
            alive, note = _frontend_alive()
            if not alive:
                raise PromoteError(f"frontend unreachable after landing ({note}) at {FRONTEND_URL}")

        # The code is live and verified: start the clock now.
        landed = time.time()
        current["state"] = "observing"
        current["landed_at"] = S.now_iso()
        current["landed_ts"] = landed
        window = errors_window(restart)
        current["errors_until_ts"] = landed + window
        current["boot_id"] = (body or {}).get("boot_id")
        current["service_changes"] = service_notes
        _write_record(current, batch)
        result["service_changes"] = service_notes
        S.append_event({"event": "promoted", "round_id": round_id, "title": title,
                        "commit": head, "parent": live_head, "changed_paths": changed,
                        "vault_commits": current.get("vault_commits") or [],
                        "tree_hash": tree_hash, "service_changes": service_notes,
                        "errors_until": current["errors_until_ts"],
                        # The window this promotion actually got, beside the
                        # flag that chose it — a landing is judged for as long
                        # as this says and the scorecard should not have to
                        # re-derive it from a constant that has since moved.
                        "errors_window_s": window,
                        # False: the merge was the landing, nothing was drained
                        # or restarted (`restart_needed`).
                        "restarted": restart, "restart_why": result.get("restart_why", ""),
                        **({"flushed_pending": [e["commit"] for e in batch]} if batch else {})})
        if batch:
            _record_flushed(batch + [{"commit": head, "round_id": round_id,
                                      "restart": restart}],
                            reason=f"eager landing of {round_id}", by="land",
                            restarted=restart, idle_waited_s=idle_waited_s, window=window)
            S.remove_pending([e["commit"] for e in batch])
            result["flushed_pending"] = [e["commit"] for e in batch]
        _announce_promoted(round_id, head, changed, title, window)
        result["regression_runner"] = _start_regression_runner()
        result["promoted"] = True
        return result

    except Exception:
        S.clear_pause()
        if not merged:
            # Nothing has moved. A failure before the merge — the drain
            # handshake, a retest that failed, a refused fast-forward — used to
            # take this same path and stop, restore and restart the services
            # for a tree that was already exactly where it belonged. With
            # uncommitted edits tolerated in that tree, the restore would also
            # have stashed them out from under the human's editor.
            S.clear_current()
            raise
        # Any failure between the merge and the verification: revert now
        # rather than waiting for the guardian's next tick. Uncommitted edits
        # in the tree survive as `broken/<stamp>/dirty.patch` — the guardian's
        # `preserve_evidence` contract — and the event says where.
        try:
            if batch and restart_began:
                # The restart ran everything the train held, so any of it may
                # be what failed: the batch goes, by the route its range allows.
                _undo_batch_and_record(live, current, trigger="promote_failed",
                                       round_id=round_id)
            else:
                evidence = _rollback_inline(live, live_head)
                S.append_event({"event": "rollback_succeeded", "trigger": "promote_failed",
                                "commit": head, "restored": live_head, "round_id": round_id,
                                "stash": evidence.get("patch"), "tag": evidence.get("tag")})
        except Exception as exc:
            S.append_event({"event": "rollback_failed", "trigger": "promote_failed",
                            "commit": head, "round_id": round_id, "error": str(exc)[:400]})
        # Say it on the ledger as the landing's failure, not the round's:
        # the round passed every rung and the tree was put back. Without
        # this row the finished implement row's `landed: true` stood alone
        # and `implement_outcomes` read the attempt as spent (the 2026-09-16
        # 16:31Z rollback of #658). `rolled_back_rounds` cannot help here —
        # it joins through a `promoted` row that a pre-observation failure
        # never writes.
        S.append_event({"event": "land_failed", "round_id": round_id, "ok": False,
                        "external_blocker": True, "rolled_back": True,
                        "detail": f"landing rolled back after the merge: {sys.exc_info()[1]}"[:500]})
        S.clear_current()
        raise
    finally:
        set_drain(False)
        release_pool_pause()
        S.clear_pause()


# supervisord states in which nothing serves and nothing will until somebody
# starts the program. BACKOFF is not here: supervisord is still retrying it.
_DEAD_STATES = frozenset({"STOPPED", "EXITED", "FATAL"})
START_AGAIN_BUDGET = 30.0


class RestartInterrupted(PromoteError):
    """A signal arrived while a program was between its stop and its start.
    Raised only once the program has been started again."""


def _program_state(program: str) -> str | None:
    """supervisord's statename for `program`, upper-cased; None when it cannot say."""
    try:
        return str(process_info(program).get("statename") or "").upper() or None
    except Exception:
        return None


def supervisorctl_hint(verb: str, program: str) -> str:
    """The exact command a human pastes, group-qualified the way supervisord
    wants it (`lloyd-mc:lloyd-backend`) — the bare name is `BAD_NAME` there."""
    name = f"lloyd-mc:{program}" if program in ("lloyd-backend", "lloyd-mcp",
                                               "lloyd-frontend") else program
    return f"{SUPERVISORCTL} -c {SUPERVISORD_CONF} {verb} {name}"


def _start_again(program: str, budget: float = START_AGAIN_BUDGET) -> tuple[bool, str]:
    """Start `program` if supervisord holds it stopped, and wait for it to be
    RUNNING or STARTING. Never raises: this runs on the way out of a failure."""
    try:
        state = _program_state(program)
        if state in ("RUNNING", "STARTING"):
            return True, f"{program} is {state}"
        ok, msg = start_process(program, wait=False)
        if not ok:
            return False, f"start {program}: {msg}"
        deadline = time.monotonic() + budget
        while True:
            state = _program_state(program)
            if state in ("RUNNING", "STARTING"):
                return True, f"{program} started again ({state})"
            if state == "FATAL" or time.monotonic() >= deadline:
                return False, f"{program} is {state or 'unreadable'} after a start"
            time.sleep(0.5)
    except BaseException as exc:   # a second Ctrl-C here must not hide the first
        return False, f"start {program} interrupted: {type(exc).__name__}"


class _SignalsHeld:
    """Hold SIGINT/SIGTERM/SIGHUP while a program is between stop and start.

    `restart_process` is stop-then-start, two RPCs, and nothing in between is
    atomic. On 2026-09-24 23:39Z a `round restart --only lloyd-backend` left
    the backend STOPPED: supervisord logged the stop and never a start, and
    the lease was cleared within a second — so Python unwound through
    `restart_stack`'s `finally` (a hard kill would have left the lease), and
    `start_process` catches every `Exception`, so only a `BaseException`
    (a Ctrl-C's `KeyboardInterrupt`) can have skipped the start. With these
    held the leg finishes — the program is started again — and the signal is
    then raised as `RestartInterrupted`. Main thread only; elsewhere
    `signal.signal` refuses and the leg's `except BaseException` is the net.
    """

    SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")

    def __init__(self, program: str):
        self.program = program
        self.received: list[int] = []
        self._previous: dict = {}

    def _handler(self, signum, frame):
        self.received.append(int(signum))
        print(f"round restart: signal {int(signum)} received while {self.program} is being "
              f"restarted; it takes effect once {self.program} is started again",
              file=sys.stderr, flush=True)

    def __enter__(self):
        import signal
        try:
            for name in self.SIGNALS:
                sig = getattr(signal, name)
                self._previous[sig] = signal.signal(sig, self._handler)
        except ValueError:            # not the main thread
            self._restore()
        return self

    def _restore(self):
        import signal
        for sig, old in self._previous.items():
            try:
                signal.signal(sig, signal.SIG_DFL if old is None else old)
            except ValueError:
                pass
        self._previous = {}

    def __exit__(self, *exc):
        self._restore()
        return False


def _restart_program(program: str, *, hold_signals: bool = False) -> tuple[bool, str]:
    """`restart_process`, and a program it stopped is never left stopped.

    Any exit — a failed or timed-out start, an exception, an interrupt — ends
    with one more `start` attempt when supervisord holds the program stopped.
    A start that works after a failed one is a success with a note; one that
    does not says so with the command to run by hand.
    """
    held = _SignalsHeld(program) if hold_signals else None
    with held or contextlib.nullcontext():
        try:
            ok, msg = restart_process(program)
        except BaseException:
            _start_again(program)
            raise
        if not ok:
            again_ok, again = _start_again(program)
            if again_ok:
                ok, msg = True, f"{msg}; {again}"
            else:
                msg = (f"{msg}; {again} — {program} may be left stopped: "
                       f"{supervisorctl_hint('start', program)}")
    if held and held.received:
        raise RestartInterrupted(f"interrupted by signal {held.received[0]} during the "
                                 f"restart of {program} ({msg}); stopped after that leg")
    return ok, msg


def restart_stack(programs: tuple[str, ...] = ("lloyd-mcp", "lloyd-backend"), *,
                  reason: str = "", max_wait: float = IDLE_MAX_WAIT,
                  force: bool = False, skip_idle: bool = False) -> dict:
    """Restart the live services the way the promoter does, for a human.

    A `supervisorctl restart` by hand is indistinguishable from a crash to the
    guardian: "Service down, but no promotion to revert" fired four times on
    2026-09-09 for four deliberate restarts, each through every channel —
    ledger, ALERT.md, journal, toast, voice, vault note. The promoter avoids
    that with the pause lease (`S.set_pause`), and it pauses the worker pool
    and drains the backend first so a research job is not killed mid-flight
    and its connection errors do not land in someone's observation window.
    CLAUDE.md describes that procedure in prose as five manual steps. This is
    the procedure.

    Refuses while a promotion is under observation unless forced: the guardian
    is judging that build, and restarting it underneath the window is the
    promoter's job to avoid, not a human's to repeat. Pool and drain are
    released whatever happens; the lease is cleared once the services are
    healthy, so the guardian is blind for exactly the restart and nothing
    after it.

    **With the land train holding merged landings this is a flush**
    (`flush_pending(force_restart=True)`): a restart of the backend or the
    aggregator makes every pending landing live, and one that did so without
    opening their window would leave them running unjudged. `skip_idle` maps to
    `--now`. The primary, if asked for too, gets its own leg afterwards.
    """
    if S.read_pending() and {"lloyd-mcp", "lloyd-backend"} & set(programs):
        out = flush_pending(f"human restart: {reason}" if reason else "human restart",
                            kill_turns=skip_idle, by="restart", force_restart=True)
        if not out.get("flushed"):
            # Everything pending had already settled or left `main`: an
            # ordinary restart, below.
            return restart_stack(programs, reason=reason, max_wait=max_wait,
                                 force=force, skip_idle=skip_idle)
        rest = tuple(p for p in programs if p not in ("lloyd-mcp", "lloyd-backend"))
        if rest:
            out["also"] = restart_stack(rest, reason=reason, max_wait=max_wait,
                                        force=force, skip_idle=skip_idle)
        return {"restarted": ["lloyd-mcp", "lloyd-backend"] + list(rest), "flush": out,
                "reason": reason}
    observed = S.read_current()
    if observed and observed.get("state") in ("landing", "observing") and not force:
        raise PromoteError(
            f"{str(observed.get('commit'))[:8]} is under observation "
            f"({observed.get('state')}) — let it settle, or pass force=True")
    health_for = {"lloyd-mcp": MCP_HEALTH, "lloyd-backend": f"{BACKEND}/health",
                  PRIMARY_PROGRAM: PRIMARY_HEALTH}
    budget_for = {PRIMARY_PROGRAM: PRIMARY_HEALTH_BUDGET}
    unknown = [p for p in programs if p not in health_for]
    if unknown:
        raise PromoteError(f"no health probe for {unknown}; restart those by hand")

    S.set_pause(RESTART_LEASE)   # the guardian must not read this as a crash
    if skip_idle:
        # The emergency form: pause the pool so nothing new starts, and
        # restart over whatever is in flight — every turn dies and is
        # re-offered. For the case where the idle wait itself is what is
        # broken: on 2026-09-16 a leaked `harness_runs` count held the
        # backend "busy" for nine hours, and the restart that would have
        # cleared the leak waited on the leak.
        if pool_paused() is False:
            set_pool_paused(True)
        ok, why = True, "idle wait skipped"
    else:
        ok, why = wait_idle(max_wait)   # pauses the pool, arms the drain, waits for quiet
    if not ok:
        S.clear_pause()          # wait_idle already released the pool and the drain
        S.append_event({"event": "restart_failed", "stage": "idle", "programs": list(programs),
                        "by": "human", "reason": reason[:300], "error": why[:400],
                        "left_stopped": [p for p in programs
                                         if _program_state(p) in _DEAD_STATES]})
        raise PromoteError(why)
    started = time.time()
    done: list[str] = []
    leg = None
    try:
        for program in programs:
            leg = program
            S.set_pause(RESTART_LEASE)   # refreshed per leg, as the promoter does
            if program == PRIMARY_PROGRAM:
                # Never started again on a failure: below the RAM abort line
                # leaving the engine stopped is the leg's deliberate answer.
                ok, msg = _restart_primary()
            else:
                ok, msg = _restart_program(program, hold_signals=True)
            if not ok:
                raise PromoteError(f"restart {program} failed: {msg}")
            if not _wait_health(health_for[program], budget_for.get(program, 90.0)):
                raise PromoteError(f"{program} never became healthy after restart")
            done.append(program)
        S.append_event({"event": "restart", "programs": list(done), "by": "human",
                        "reason": reason[:300], "seconds": round(time.time() - started, 1)})
        return {"restarted": done, "seconds": round(time.time() - started, 1),
                "idle": why, "reason": reason}
    except BaseException as exc:
        # The 2026-09-24 23:39Z restart wrote nothing: the ledger heard of a
        # restart only when one succeeded, so the stopped backend read as a
        # crash nobody had asked for. Every other exit says what it left down.
        left = [p for p in programs if _program_state(p) in _DEAD_STATES]
        try:
            S.append_event({"event": "restart_failed", "stage": "restart",
                            "programs": list(programs), "restarted": list(done),
                            "failed_at": leg, "by": "human", "reason": reason[:300],
                            "error": (str(exc) or type(exc).__name__)[:400],
                            "left_stopped": left,
                            "seconds": round(time.time() - started, 1)})
        except Exception:
            pass
        if left:
            print("round restart: LEFT STOPPED: " + ", ".join(left) + "\n  start with: "
                  + "\n  start with: ".join(supervisorctl_hint("start", p) for p in left),
                  file=sys.stderr, flush=True)
            if isinstance(exc, PromoteError):
                raise PromoteError(f"{exc} — left stopped: {', '.join(left)}; "
                                   f"{supervisorctl_hint('start', left[0])}") from exc
        raise
    finally:
        set_drain(False)
        release_pool_pause()
        S.clear_pause()


def _wait_health(url: str, budget: float) -> bool:
    """Poll `url` until it answers 200 or `budget` runs out. A budget longer
    than the lease (the primary's boot) refreshes the lease on the way, so
    the guardian does not wake up to a stopped engine halfway through."""
    deadline = time.time() + budget
    refreshed = time.time()
    while time.time() < deadline:
        status, _ = _get(url, 3.0)
        if status == 200:
            return True
        if budget > RESTART_LEASE and time.time() - refreshed > RESTART_LEASE / 2:
            S.set_pause(RESTART_LEASE)
            refreshed = time.time()
        time.sleep(1.0)
    return False


VERIFY_COMMIT_BUDGET = 60.0


def _wait_for_commit(url: str, budget: float) -> dict | None:
    """Poll `url` until it answers with a JSON body that names a `commit`,
    or `budget` runs out (then the last body, which may be None). A 200
    with no body, a refused connection while the process is replaced, and
    a `starting` 503 are all "not yet", never "no"."""
    deadline = time.time() + budget
    body = None
    while True:
        status, body = _get(url, 5.0)
        if isinstance(body, dict) and body.get("commit"):
            return body
        if time.time() >= deadline:
            return body if isinstance(body, dict) else None
        time.sleep(1.0)


def _host_ram_available_gib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1048576
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _wait_host_ram(floor_gib: int, budget: float) -> int:
    """Wait for the stopped engine's host-RAM table to be reclaimed. Returns
    the last reading, whatever it was; the caller decides the abort line."""
    deadline = time.time() + budget
    avail = _host_ram_available_gib()
    while avail < floor_gib and time.time() < deadline:
        S.set_pause(RESTART_LEASE)
        time.sleep(5.0)
        avail = _host_ram_available_gib()
    return avail


def _restart_primary() -> tuple[bool, str]:
    """Stop the engine, wait for its host-RAM table to go, pick up any conf
    change, start it. Health is the caller's wait (`PRIMARY_HEALTH_BUDGET`)."""
    ok, msg = stop_process(PRIMARY_PROGRAM, wait=False)
    if not ok:
        return False, f"stop failed: {msg}"
    deadline = time.time() + 120.0   # stopwaitsecs is 90
    while time.time() < deadline:
        try:
            state = process_info(PRIMARY_PROGRAM).get("statename", "").upper()
        except Exception:
            break
        if state in ("STOPPED", "EXITED", "FATAL"):
            break
        time.sleep(2.0)
    avail = _wait_host_ram(PRIMARY_RAM_FLOOR_GIB, PRIMARY_RAM_WAIT_SECONDS)
    if avail < PRIMARY_RAM_ABORT_GIB:
        return False, (f"only {avail} GiB host RAM available after the stop; a 170 GiB load "
                       f"would risk the oomd kill of 2026-09-08 — the engine is left stopped, "
                       f"start it by hand once MemAvailable is over {PRIMARY_RAM_FLOOR_GIB} GiB")
    if SUPERVISORCTL.exists():
        for verb in (["reread"], ["update", PRIMARY_PROGRAM]):
            r = _run([SUPERVISORCTL, "-c", SUPERVISORD_CONF, *verb], timeout=120)
            if r.returncode != 0:
                return False, f"supervisorctl {' '.join(verb)} failed: {(r.stderr or r.stdout)[:200]}"
    S.set_pause(RESTART_LEASE)
    ok, msg = start_process(PRIMARY_PROGRAM, wait=False)
    if not ok:
        return False, f"start failed: {msg}"
    return True, f"started after {avail} GiB host RAM free"


def _rollback_inline(live: Path, target: str) -> dict:
    """Reuse the guardian's rollback rather than reimplementing it.

    Returns `preserve_evidence`'s record — tag, and the patch any uncommitted
    live edits were written to before `reset --hard` — so the caller can put
    that path in front of whoever was editing.
    """
    import importlib.util
    guardian_dir = live / "agent-services" / "guardian"
    spec_ = importlib.util.spec_from_file_location("_g_rollback", guardian_dir / "rollback.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)

    for program in ("lloyd-backend", "lloyd-mcp"):
        stop_process(program, wait=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    evidence = mod.preserve_evidence(str(live), S.BROKEN_DIR / stamp, f"guardian-broken-{stamp}")
    mod.restore_tree(str(live), target, ("app", "agent_mcp", "workers", "scripts",
                                         "eval", "tests"),
                     ("app", "agent_mcp", "workers", "scripts"))
    mod.verify_tree(str(live), target)
    for program, health in (("lloyd-mcp", MCP_HEALTH), ("lloyd-backend", f"{BACKEND}/health")):
        restart_process(program)
        _wait_health(health, 90.0)
    return evidence or {}


# ═══════════════════════════════════════════════════════════════════════════
# The land train: a merge is a landing, a restart is a flush (2026-09-24)
# ═══════════════════════════════════════════════════════════════════════════
#
# Every restarting landing used to drain the loop to zero first — wait out
# sibling turns, pause the pool, wait for idle, restart — 172 of them a week,
# 32 of the loop's 58 idle hours. With `automod.landing.defer_restart` on, a
# landing that need not restart anything NOW fast-forwards `main`, writes its
# `promoted` row, starts the regression check, and appends itself to
# `pending_restart.json`; it opens no window. A flush then restarts ONCE for
# everything pending and writes the one `current.json` the guardian judges,
# with `commits: [oldest..newest]` and the oldest entry's parent as the
# rollback target. `architecture/automod.md` §3.2i is the long version.

RESTART_AFTER_S = 2700.0
RESTART_BATCH_MAX = 4
BATCH_SUSPEND_AFTER_ROLLBACK_S = 21600.0
FLUSH_LOCK_MAX_WAIT = 2 * 3600.0


class FlushNotStarted(PromoteError):
    """A flush that stopped before anything was restarted: nothing to undo,
    the pending entries stay pending for the next trigger."""


def restart_deferred() -> tuple[bool, str]:
    """`(the land train is on, why)`. Three things must all hold:

    `automod.landing.defer_restart` (ships false); the STAGED guardian declares
    `BATCH_SCHEMA >= 2` (`S.guardian_batch_aware`) — the interlock, so no batch
    record is ever written for a guardian that would reset over a foreign
    commit to undo it; and no batch rollback in the last
    `batch_suspend_after_rollback_s`, so re-landings are attributed one at a
    time for a while (`S.batch_suspended_until`)."""
    if not bool(S.landing_cfg(LIVE_ROOT).get("defer_restart", False)):
        return False, "automod.landing.defer_restart is off"
    if not S.guardian_batch_aware():
        return False, ("the staged guardian cannot roll back a batch "
                       f"({S.guardian_bin_dir() / 'gstate.py'} declares no BATCH_SCHEMA >= "
                       f"{S.GUARDIAN_BATCH_SCHEMA}) — restage it and run the batch drill")
    until = S.batch_suspended_until(
        _landing_num("batch_suspend_after_rollback_s", BATCH_SUSPEND_AFTER_ROLLBACK_S))
    if until > time.time():
        return False, (f"batch landings are suspended for {(until - time.time()) / 60:.0f} "
                       "more min after a batch rollback: landing singly so each change is "
                       "judged on its own window")
    return True, "restart deferred to the next flush"


def landing_is_eager(changed: list[str], gate_report: dict | None) -> list[str]:
    """Why this landing must restart the services itself, now — `[]` when it
    can merge and leave the restart to the flush.

    With the train off every landing is eager (the reason says so), which is
    the landing path as it always was. With it on: a candidate venv (the swap
    and the restart are one act), and anything under `agent-services/` (a
    service definition or the guardian, applied by `_apply_service_changes`
    before a restart). Those flush inline, carrying the batch."""
    on, why = restart_deferred()
    if not on:
        return [why]
    reasons: list[str] = []
    if str((gate_report or {}).get("venv") or "").strip():
        reasons.append("the gate built a candidate venv; swapping it is a restart")
    svc = [p for p in changed or [] if str(p).startswith(ALWAYS_RESTART_PREFIXES)]
    if svc:
        reasons.append(f"{svc[0]} is a service definition or guardian file")
    return reasons


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    if not ancestor or not descendant:
        return False
    if ancestor == descendant:
        return True
    return subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                           ancestor, descendant], capture_output=True).returncode == 0


def _live_head(live: Path) -> str:
    return subprocess.run(["git", "-C", str(live), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _live_pending(live: Path) -> list[dict]:
    """The pending entries still worth a window, oldest first. Under the lock.

    An entry the guardian already settled (a flush that died after its record
    was judged) or whose commit is no longer in `main`'s history (reverted, or
    a merge that never happened after its entry was written) is dropped and
    said on the ledger — a flush that re-observed a reverted commit would give
    its `rollback_target` a meaning the tree no longer has."""
    head = _live_head(live)
    keep, dropped = [], []
    for e in S.read_pending():
        c = str(e.get("commit") or "")
        (keep if c and not _settled(c) and _is_ancestor(live, c, head) else dropped).append(e)
    if dropped:
        S.remove_pending([e.get("commit") for e in dropped])
        S.append_event({"event": "pending_dropped", "ok": True,
                        "commits": [e.get("commit") for e in dropped],
                        "round_ids": [e.get("round_id") for e in dropped],
                        "detail": "settled already, or no longer in main's history"})
    return keep


def _stamp_batch(current: dict, batch: list[dict]) -> dict:
    """The batch fields on a promotion record: `commits` oldest..newest ending
    in the record's own commit, the oldest entry's parent as the rollback
    target, and the per-commit entries the guardian denylists by content."""
    if not batch:
        return current
    commits = [str(e["commit"]) for e in batch if str(e["commit"]) != current.get("commit")]
    commits.append(str(current.get("commit")))
    current["schema"] = 2
    current["commits"] = commits
    current["rollback_target"] = current["parent"] = str(batch[0].get("parent") or "")
    current["entries"] = [_compact_entry(e) for e in batch] + (
        [{"commit": current.get("commit"), "round_id": current.get("round_id"),
          "changed_paths": list(current.get("changed_paths") or []),
          "tree_hash": current.get("tree_hash")}]
        if current.get("commit") not in {e.get("commit") for e in batch} else [])
    return current


def _write_record(current: dict, batch: list[dict]) -> dict:
    """`S.write_verified` of the promotion record, with the batch stamped on
    when there is one. With none it IS `S.write_verified(CURRENT_PATH, …)`."""
    if batch:
        _stamp_batch(current, batch)
    return S.write_verified(S.CURRENT_PATH, current)


def _compact_entry(e: dict) -> dict:
    return {k: e.get(k) for k in ("commit", "parent", "round_id", "title", "changed_paths",
                                  "tree_hash", "restart", "merged_ts")}


def merge_announcement(title: str, n_files: int, pending: int, restart: bool) -> tuple[str, str]:
    """Toast head and body for a merge the train holds. Pure, like
    `promotion_announcement`, and it states no window: none is open yet."""
    head = f"Landed: {title}" if title else "Landed a change"
    files = f"{n_files} file{'' if n_files == 1 else 's'} changed."
    if restart:
        tail = (f" Goes live at the next restart ({pending} landing"
                f"{'' if pending == 1 else 's'} waiting).")
    else:
        tail = " Nothing running needed a restart."
    return head, files + tail


def merge_round(round_id: str, worktree: Path, base: str, *,
                gate_report: dict | None = None, dry_run: bool = False) -> dict:
    """Land a round by merging it, and leave the restart to the flush.

    The same refusals, chase and squash as `promote`, in the same order —
    halt, BROKEN, the gated head, the denylist before and after a rebase,
    overlapping live dirt — and none of the rest: no settle wait (a promotion
    under observation is running OTHER code; this merge changes nothing that
    runs), no idle wait, no drain, no pool pause, no restart lease, no window.
    `main` moves, `promoted` is written with `restart_pending`, the regression
    check starts, and the entry goes on the train.

    The rollback point is still written BEFORE the tree moves: the pending
    entry, verified, carries the parent. A process killed between the merge
    and the end leaves an entry whose commit IS in `main`, which the flush
    treats as any other; killed before the merge, an entry whose commit is
    not, which the flush drops. A `landing` record (a flush or an eager
    landing mid-restart) still refuses — though under the automod lock that
    record's writer cannot be running, so it is a dead one."""
    live = LIVE_ROOT
    head = W.head(Path(worktree))
    if not head:
        raise PromoteError("cannot read the candidate HEAD")
    observed = S.read_current()
    if observed and observed.get("state") == "landing":
        raise PromoteError(
            f"{str(observed.get('commit'))[:8]} is mid-landing ({observed.get('state')}) — "
            "a restart is in progress; not merging underneath it")

    changed = W.changed_paths(Path(worktree), base)
    tree_hash = S.changed_tree_hash(worktree, head, changed)
    if S.is_denied(commit=head, tree_hash=tree_hash):
        raise PromoteError(
            f"{head[:8]} is on the rollback denylist"
            + (" (matched by content, not SHA — this change was reverted before "
               "and has been re-derived)" if tree_hash and not S.is_denied(commit=head)
               else ""))
    live_head = _live_head(live)
    if live_head != base:
        base, head, gate_report = _regate_after_move(round_id, Path(worktree), live,
                                                     base, live_head)
        changed = W.changed_paths(Path(worktree), base)
        tree_hash = S.changed_tree_hash(worktree, head, changed)
        if S.is_denied(commit=head, tree_hash=tree_hash):
            _land_failed(round_id, f"{head[:8]} (rebased) is on the rollback denylist",
                         external=False)
        if landing_is_eager(changed, gate_report):
            # The rebase brought in a service definition or a venv: this is an
            # eager landing now, and `promote` is the path that restarts.
            _land_failed(round_id, "after rebasing onto a moved main the landing must "
                         "restart the services itself; land it again", external=True)

    from scripts.automod import backlog as B
    title = B.work_title_for_round(S.LEDGER_PATH, round_id)
    live_dirty = W.dirty_paths(live)
    overlap = sorted(set(live_dirty) & set(changed))
    if overlap:
        _land_failed(round_id,
                     f"live tree has uncommitted edits in paths this round also changes: "
                     f"{overlap} — two writers on one file. Report the paths and who is "
                     f"editing them; the live edit is not yours to commit and not yours "
                     f"to move out of the tree, and nothing here needs a clean live "
                     f"tree to land again",
                     external=True, overlap=overlap)

    restart, restart_why = restart_needed(changed)
    result: dict = {"round_id": round_id, "commit": head, "parent": live_head,
                    "changed_paths": changed, "dry_run": dry_run,
                    "live_dirty_paths": live_dirty[:20], "title": title,
                    "deferred": True, "restart": False, "restart_pending": restart,
                    "restart_why": restart_why}
    if dry_run:
        result["would_promote"] = True
        return result

    _, body = _get(f"{BACKEND}/health")
    entry = {"round_id": round_id, "title": title, "commit": head, "parent": live_head,
             "changed_paths": changed, "tree_hash": tree_hash, "restart": restart,
             "restart_why": restart_why, "vault_commits": vault_commits_for(round_id),
             "live_dirty_paths": live_dirty[:20], "kg_rows": count_kg_rows(),
             "vault_files": count_vault_files(), "boot_id": (body or {}).get("boot_id"),
             "gate": (gate_report or {}).get("rungs"), "merged_ts": None,
             "queued_ts": time.time()}
    S.append_pending(entry)   # the rollback point: verified before the tree moves

    merged = False
    try:
        if squash_enabled():
            squashed, note = W.squash_onto(
                Path(worktree), live_head,
                squash_message(round_id, title, Path(worktree), live_head),
                keep_ref=f"refs/automod/rounds/{round_id}")
            if squashed:
                result["squashed_from"] = entry["squashed_from"] = head
                head = entry["commit"] = result["commit"] = squashed
                S.append_pending(entry)
            result["squash"] = note
        merge = subprocess.run(
            ["git", "-C", str(live), "merge", "--ff-only", f"automod/{round_id}"],
            capture_output=True, text=True)
        if merge.returncode != 0:
            _land_failed(round_id, f"fast-forward failed after rebasing and retesting: "
                                   f"{merge.stderr.strip()[:300]}", external=True)
        merged = True
        on_disk = _live_head(live)
        if on_disk != head:
            raise PromoteError(f"live HEAD is {on_disk[:8]}, expected {head[:8]} after the merge")
        # Vite serves the live tree and HMR picks a merge up at once, so a
        # frontend change is live now whatever the train holds.
        if any(p.startswith("web/") for p in changed):
            alive, note = _frontend_alive()
            if not alive:
                raise PromoteError(f"frontend unreachable after landing ({note}) at {FRONTEND_URL}")
        entry["merged_ts"] = time.time()
        pending = S.append_pending(entry)
    except Exception:
        S.remove_pending([entry["commit"]])
        if not merged:
            raise
        try:
            evidence = _rollback_inline(live, live_head)
            S.append_event({"event": "rollback_succeeded", "trigger": "promote_failed",
                            "commit": head, "restored": live_head, "round_id": round_id,
                            "stash": evidence.get("patch"), "tag": evidence.get("tag")})
        except Exception as exc:
            S.append_event({"event": "rollback_failed", "trigger": "promote_failed",
                            "commit": head, "round_id": round_id, "error": str(exc)[:400]})
        S.append_event({"event": "land_failed", "round_id": round_id, "ok": False,
                        "external_blocker": True, "rolled_back": True,
                        "detail": f"merge rolled back: {sys.exc_info()[1]}"[:500]})
        raise

    S.append_event({"event": "promoted", "round_id": round_id, "title": title,
                    "commit": head, "parent": live_head, "changed_paths": changed,
                    "vault_commits": entry["vault_commits"], "tree_hash": tree_hash,
                    "service_changes": [], "errors_until": None, "errors_window_s": None,
                    # Merged, not restarted: the flush that restarts it writes
                    # `restart_flushed` and the window. `restarted` is False only
                    # when nothing it changed is loaded, as before.
                    "restarted": None if restart else False, "restart_why": restart_why,
                    "deferred": True, "restart_pending": restart,
                    "pending": len(pending)})
    head_line, body_line = merge_announcement(
        title, len(changed), sum(1 for e in pending if e.get("restart")), restart)
    announce(head_line, body_line)
    result["regression_runner"] = _start_regression_runner()
    result["pending"] = len(pending)
    result["promoted"] = True
    return result


def flush_due(rounds_in_flight: int | None = None, *, now: float | None = None) -> tuple[bool, str]:
    """`(a flush should start now, why)`. Cheap: two small files and a marker.

    Never while a flush or a landing is running, a promotion is recorded, or
    promotions are halted, BROKEN or a rollback is pending — the flush would
    only wait on it. Otherwise, in order: nothing pending needs a restart (a
    flush then restarts nothing and costs nothing); the oldest entry has waited
    `restart_after_s` (45 min); `restart_batch_max` (4) entries need one; or the
    loop is at a natural gap — `rounds_in_flight == 0`, where the drain is
    cheapest because no implement turn is there to wait out. `None` means the
    caller cannot say, and the gap trigger is not used."""
    entries = S.read_pending()
    if not entries:
        return False, "nothing pending"
    now = time.time() if now is None else float(now)
    if S.flush_in_progress():
        return False, "a flush is already running"
    if S.is_halted() or S.is_broken():
        return False, "promotions are halted or the guardian is BROKEN"
    if S.read_rollback_request():
        return False, "a rollback request is pending"
    cur = S.read_current()
    if cur and cur.get("state") in ("landing", "observing"):
        return False, f"{str(cur.get('commit'))[:8]} is {cur.get('state')}; flushing after it settles"
    if S.rounds_landing():
        return False, "a landing is running"
    if len(entries) >= 2 and not S.guardian_batch_aware():
        return False, (f"{len(entries)} landings are pending but the staged guardian cannot "
                       "judge a batch — restage it (guardian-stage.sh) before flushing")
    needs = [e for e in entries if e.get("restart")]
    if not needs:
        return True, f"{len(entries)} pending landing(s) need no restart"
    stamps = [float(e.get("merged_ts") or e.get("queued_ts") or now) for e in needs]
    age = now - min(stamps)
    after = _landing_num("restart_after_s", RESTART_AFTER_S)
    if age >= after:
        return True, f"the oldest pending landing has waited {age / 60:.0f} min (>= {after / 60:.0f})"
    batch_max = int(_landing_num("restart_batch_max", RESTART_BATCH_MAX))
    if len(needs) >= max(1, batch_max):
        return True, f"{len(needs)} landings need a restart (batch max {batch_max})"
    if rounds_in_flight == 0:
        return True, "no implement turn in flight: a natural gap"
    return False, (f"{len(needs)} landing(s) wait for a restart, oldest {age / 60:.0f} min; "
                   f"{rounds_in_flight if rounds_in_flight is not None else '?'} turn(s) in flight")


def pending_summary(rounds_in_flight: int | None = None) -> dict:
    """What `round status` and `automod_status` say about the train.
    `rounds_in_flight` is the caller's count of implement turns (None when it
    could not read one), so the natural-gap trigger is judged here exactly as
    the backend's `autocode._maybe_flush` judges it."""
    entries = S.read_pending()
    now = time.time()
    needs = [e for e in entries if e.get("restart")]
    try:
        due = flush_due(rounds_in_flight)
    except Exception as exc:  # noqa: BLE001 — a status is never the thing that fails
        due = (False, f"unreadable: {exc}")
    return {"train": dict(zip(("on", "why"), restart_deferred())),
            "count": len(entries), "restart_needed": len(needs),
            "oldest_age_s": (round(now - min(float(e.get("merged_ts") or e.get("queued_ts") or now)
                                              for e in entries), 1) if entries else None),
            "entries": [{k: e.get(k) for k in ("round_id", "title", "commit", "restart",
                                               "merged_ts")} for e in entries],
            "flush_due": due[0], "flush_why": due[1],
            "rounds_in_flight": rounds_in_flight,
            "flush_running": S.flush_in_progress()}


def _record_flushed(batch: list[dict], *, reason: str, by: str, restarted: bool,
                    idle_waited_s: float = 0.0, rounds_waited_s: float = 0.0,
                    settle_waited_s: float = 0.0, already_live: bool = False,
                    window: float | None = None) -> None:
    """The `restart_flushed` ledger row: one per restart that went live for
    the train, with the idle wait it spent (`waited_s`, what scorecard row 14
    reads beside the two landing waits)."""
    S.append_event({"event": "restart_flushed", "ok": True, "by": by,
                    "reason": str(reason)[:300],
                    "commits": [e.get("commit") for e in batch],
                    "round_ids": [e.get("round_id") for e in batch],
                    "batch": len(batch), "restarted": bool(restarted),
                    "already_live": bool(already_live),
                    "waited_s": round(float(idle_waited_s), 1),
                    "waited_rounds_s": round(float(rounds_waited_s), 1),
                    "waited_settle_s": round(float(settle_waited_s), 1),
                    "errors_window_s": window})


def _guardian_rollback_module(live: Path):
    import importlib.util
    guardian_dir = live / "agent-services" / "guardian"
    spec_ = importlib.util.spec_from_file_location("_g_rollback", guardian_dir / "rollback.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def _undo_batch(live: Path, commits: list[str], target: str) -> dict:
    """Take a batch the promoter restarted off `main`, inline, by the route
    its range allows: `reset --hard target` only when HEAD is the newest batch
    commit and `target..HEAD` is exactly the batch, every commit reverted in
    place otherwise — the guardian's rule, through the guardian's code."""
    mod = _guardian_rollback_module(live)
    head = _live_head(live)
    if head == commits[-1] and mod.range_is_exactly(str(live), target, head, commits):
        evidence = dict(_rollback_inline(live, target) or {})
        evidence.update(route="reset", restored=target)
        return evidence
    for program in ("lloyd-backend", "lloyd-mcp"):
        stop_process(program, wait=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    evidence = dict(mod.preserve_evidence(str(live), S.BROKEN_DIR / stamp,
                                          f"guardian-broken-{stamp}") or {})
    restored = mod.revert_commits(str(live), list(commits))
    for program, health in (("lloyd-mcp", MCP_HEALTH), ("lloyd-backend", f"{BACKEND}/health")):
        restart_process(program)
        _wait_health(health, 90.0)
    evidence.update(route="revert", restored=restored)
    return evidence


def _undo_batch_and_record(live: Path, record: dict, *, trigger: str,
                           round_id: str = "") -> None:
    """`_undo_batch` for a record, with the ledger row; a batch that cannot be
    undone inline is handed to the guardian as one request naming every
    commit, which escalates to a human if it cannot either."""
    commits = list(record.get("commits") or [record.get("commit")])
    target = str(record.get("rollback_target") or "")
    try:
        ev = _undo_batch(live, commits, target)
        S.append_event({"event": "rollback_succeeded", "trigger": trigger,
                        "commit": commits[-1], "commits": commits, "batch": len(commits),
                        "restored": ev.get("restored"), "route": ev.get("route"),
                        "round_id": round_id or record.get("round_id"),
                        "stash": ev.get("patch"), "tag": ev.get("tag")})
    except Exception as exc:  # noqa: BLE001 — the guardian is the escalation path
        S.request_rollback(reason=f"{trigger}: the promoter could not undo the batch "
                                  f"inline ({exc})"[:1000],
                           trigger=trigger, target=target or None, commit=commits[-1],
                           commits=commits,
                           changed_paths=list(record.get("changed_paths") or []))
    finally:
        S.remove_pending(commits)


def flush_pending(reason: str = "", *, kill_turns: bool = False, by: str = "",
                  force_restart: bool = False) -> dict:
    """Restart once for everything the train holds, and open its one window.

    Today's post-merge landing block, applied to the batch: `wait_for_rounds`
    and `wait_for_settle` OUTSIDE the automod lock (#1215), then under it the
    pending list re-read and trimmed (`_live_pending`), the record written in
    state `landing` before anything moves, `wait_idle` (pool paused, drain
    armed), the restart legs under the guardian's lease, and the proof that
    the running code now contains the newest batch commit. The record then
    reads `observing` with `commits`, the pending entries go, and
    `restart_flushed` carries the waits.

    Already live: when the backend's `/health.commit` already contains every
    entry that needed a restart (the guardian's own rollback restart, a human
    restart, a crash), nothing is restarted and the window opens on what is
    running. Nothing pending needs a restart and not `force_restart`: a
    `restart: false` record with the short window, and no drain at all.

    `kill_turns` (`round flush --now`): no wait for implement turns and no
    idle wait — the pool is paused and every turn in flight dies with the
    restart and is re-offered. Failure before the restart leaves everything
    pending (`flush_failed`); after it, the batch is undone
    (`_undo_batch_and_record`) and `flush_failed` says so."""
    import os
    if not S.read_pending():
        return {"flushed": False, "detail": "nothing pending"}
    if S.is_halted():
        raise PromoteError(f"promotions are halted: {S.HALTED_PATH}")
    if S.is_broken():
        raise PromoteError(f"guardian is in a BROKEN state: {S.BROKEN_PATH}")
    me = os.getpid()
    other = S.flush_in_progress()
    if other and int(other.get("pid") or 0) != me:
        raise PromoteError(f"a flush is already running (pid {other.get('pid')}, "
                           f"since {other.get('started_iso')})")
    S.write_flush_marker(pid=me, by=by)
    try:
        return _flush(reason, kill_turns=kill_turns, by=by, force_restart=force_restart)
    finally:
        S.clear_flush_marker(pid=me)


def _flush_failed(detail: str, **extra) -> None:
    S.append_event({"event": "flush_failed", "ok": False, "detail": str(detail)[:500], **extra})


def _flush(reason: str, *, kill_turns: bool, by: str, force_restart: bool) -> dict:
    live = LIVE_ROOT
    entries = S.read_pending()
    needs = force_restart or any(e.get("restart") for e in entries)
    rounds_s = settle_s = 0.0
    if needs and not kill_turns:
        t0 = time.time()
        ok, why = wait_for_rounds(_idle_budget(None)[1])
        rounds_s = time.time() - t0
        if not ok:
            _flush_failed(why, waited_rounds=True, waited_rounds_s=round(rounds_s, 1))
            raise FlushNotStarted(why)
    observed = S.read_current()
    if observed and observed.get("state") == "observing":
        t0 = time.time()
        try:
            wait_for_settle(observed=observed)
        except PromoteError as exc:
            _flush_failed(str(exc), waited_for_settle=True)
            raise FlushNotStarted(str(exc)) from exc
        settle_s = time.time() - t0
    try:
        lock = S.Lock(owner="flush").acquire_wait(FLUSH_LOCK_MAX_WAIT, poll=10.0)
    except S.LockHeld as exc:
        _flush_failed(f"the automod lock stayed held: {exc}")
        raise FlushNotStarted(str(exc)) from exc
    try:
        return _flush_locked(live, reason, kill_turns=kill_turns, by=by,
                             force_restart=force_restart, rounds_s=rounds_s, settle_s=settle_s)
    finally:
        lock.release()


def _flush_locked(live: Path, reason: str, *, kill_turns: bool, by: str,
                  force_restart: bool, rounds_s: float, settle_s: float) -> dict:
    global _POOL_PAUSED_BY_US
    keep = _live_pending(live)
    if not keep:
        return {"flushed": False, "detail": "nothing left pending after the trim"}
    if len(keep) >= 2 and not S.guardian_batch_aware():
        why = ("the staged guardian cannot judge a batch; not writing a record it would "
               "roll back by resetting over whatever sits between the commits")
        _flush_failed(why, commits=[e["commit"] for e in keep])
        raise FlushNotStarted(why)
    cur = S.read_current()
    if cur and cur.get("state") in ("landing", "observing"):
        why = f"{str(cur.get('commit'))[:8]} is {cur.get('state')}; a flush must not overwrite it"
        _flush_failed(why)
        raise FlushNotStarted(why)

    needs = force_restart or any(e.get("restart") for e in keep)
    commits = [str(e["commit"]) for e in keep]
    _, body = _get(f"{BACKEND}/health")
    running = str((body or {}).get("commit") or "")
    boot_before = (body or {}).get("boot_id")
    already_live = bool(needs and not force_restart and running and all(
        _is_ancestor(live, str(e["commit"]), running) for e in keep if e.get("restart")))

    changed: list[str] = []
    for e in keep:
        changed += [p for p in e.get("changed_paths") or [] if p not in changed]
    first, last = keep[0], keep[-1]
    record: dict = {
        "schema": 2 if len(keep) > 1 else 1,
        "round_id": last.get("round_id"),
        "title": (last.get("title") if len(keep) == 1
                  else f"{len(keep)} landings: " + "; ".join(
                      str(e.get("title") or e.get("round_id")) for e in keep))[:300],
        "commit": commits[-1], "commits": commits,
        "parent": first.get("parent"), "rollback_target": first.get("parent"),
        "entries": [_compact_entry(e) for e in keep],
        "branch": None, "state": "landing", "landed_at": None, "landed_ts": None,
        "errors_until_ts": None, "changed_paths": changed,
        "live_dirty_paths": W.dirty_paths(live)[:20],
        "vault_commits": [c for e in keep for c in (e.get("vault_commits") or [])],
        "tree_hash": last.get("tree_hash"), "venv_swapped": False,
        "touched_guardian": False,
        # The counts from before the batch's FIRST merge: a script merged an
        # hour ago can already have run, and the flush's own count would hide it.
        "kg_rows": first.get("kg_rows") if first.get("kg_rows") is not None else count_kg_rows(),
        "vault_files": (first.get("vault_files") if first.get("vault_files") is not None
                        else count_vault_files()),
        "restart": needs, "boot_id": boot_before,
        "flush": {"reason": str(reason)[:300], "by": by, "already_live": already_live},
    }
    S.write_verified(S.CURRENT_PATH, record)   # the rollback point, before anything moves

    restart_began = False
    idle_s = 0.0
    try:
        if needs and not already_live:
            t0 = time.time()
            if kill_turns:
                if pool_paused() is False and set_pool_paused(True):
                    _POOL_PAUSED_BY_US = True
                ok, why = True, "turns in flight are killed (--now)"
            else:
                ok, why = wait_idle()
            idle_s = time.time() - t0
            if not ok:
                raise FlushNotStarted(why)
            set_drain(True, drain_ttl())
            if not kill_turns:
                _, h = _get(f"{BACKEND}/health")
                turns = (h or {}).get("turns") or {}
                if turns.get("active") or turns.get("queued") or turns.get("harness_runs"):
                    raise FlushNotStarted(f"a turn started during the drain handshake: {turns}")
            restart_began = True
            for program, health in (("lloyd-mcp", MCP_HEALTH),
                                    ("lloyd-backend", f"{BACKEND}/health")):
                S.set_pause(RESTART_LEASE)
                ok, msg = _restart_program(program)
                if not ok:
                    raise PromoteError(f"restart {program} failed: {msg}")
                if not _wait_health(health, 90.0):
                    raise PromoteError(f"{program} never became healthy after restart")
            body = _wait_for_commit(f"{BACKEND}/health", VERIFY_COMMIT_BUDGET)
            actual = str((body or {}).get("commit") or "")
            if not _is_ancestor(live, commits[-1], actual):
                raise PromoteError(f"backend reports commit {actual or None}, which does not "
                                   f"contain {commits[-1][:8]} — the restart did not pick up "
                                   "the batch")
            if boot_before and (body or {}).get("boot_id") == boot_before:
                raise PromoteError("backend boot_id unchanged — the process was never replaced")
        else:
            status, body = _get(f"{BACKEND}/health")
            if status != 200:
                raise FlushNotStarted(f"backend /health answered {status}; not opening a "
                                      "window on a service that is not answering")

        landed = time.time()
        window = errors_window(needs)
        record.update(state="observing", landed_at=S.now_iso(), landed_ts=landed,
                      errors_until_ts=landed + window, boot_id=(body or {}).get("boot_id"))
        S.write_verified(S.CURRENT_PATH, record)
        S.remove_pending(commits)
        restarted = bool(needs and not already_live)
        _record_flushed(keep, reason=reason, by=by, restarted=restarted,
                        idle_waited_s=idle_s, rounds_waited_s=rounds_s,
                        settle_waited_s=settle_s, already_live=already_live, window=window)
        n = len(keep)
        announce(f"Restarted for {n} landing{'' if n == 1 else 's'}" if restarted
                 else f"Watching {n} landing{'' if n == 1 else 's'}",
                 f"{len(changed)} files across {n} round{'' if n == 1 else 's'}. "
                 f"Watching for {window / 60:.1f} minutes.")
        return {"flushed": True, "commits": commits, "restarted": restarted,
                "already_live": already_live, "errors_window_s": window,
                "waited_s": round(idle_s, 1), "reason": reason}
    except Exception as exc:
        S.clear_pause()
        if not restart_began:
            S.clear_current()
            _flush_failed(str(exc), commits=commits, restarted=False)
            if isinstance(exc, PromoteError):
                raise
            raise FlushNotStarted(str(exc)) from exc
        _undo_batch_and_record(live, record, trigger="flush_failed")
        S.clear_current()
        _flush_failed(f"undone after the restart: {exc}", commits=commits, restarted=True,
                      rolled_back=True)
        raise
    finally:
        set_drain(False)
        release_pool_pause()
        S.clear_pause()
