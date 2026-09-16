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

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.supervisor_client import process_info, restart_process, start_process, stop_process
from scripts.automod import state as S, worktree as W

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

IDLE_POLL_SECONDS = 2.0
IDLE_QUIET_POLLS = 3
IDLE_MAX_WAIT = 900.0
DRAIN_TTL = 180.0
DRAIN_REFRESH_SECONDS = 60.0   # re-arm well inside the TTL while waiting for idle
# The observation window. Liveness is watched for ALL of it, not for some
# shorter sub-window: a build that crashes at minute ten is exactly as bad as
# one that crashes at minute one. There was a `LIVENESS_WINDOW = 120.0` here
# and a `liveness_until_ts` written into every promotion record, and nothing
# ever read either — the guardian applies liveness whenever a promotion is
# under observation. A constant that looks like a bound and bounds nothing is
# worse than no constant.
ERRORS_WINDOW = 900.0
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
#     to pass PRIMARY_RAM_FLOOR_GIB and refuses to boot below the abort line;
#   - its `environment=` lives in the program's conf, so the leg runs
#     `reread` + `update` before starting, or an edited KV budget is ignored.
PRIMARY_PROGRAM = "agent-llm-primary"
PRIMARY_HEALTH = "http://127.0.0.1:8096/health"
PRIMARY_HEALTH_BUDGET = 1200.0
# 150/120 on the first cut, and the first real use (2026-09-15 23:47Z) got
# through them and still lost the unit: a 16 GiB qemu VM had joined the
# desktop, MemAvailable read 57 GiB with the engine up, the stop freed the
# table to just past the floor, and the boot's own transient took the box
# to pressure — systemd-oomd killed agent-supervisord.service at 23:52:46Z
# and every program under it came back on autorestart (memwatch snapshot
# 20260915_235555). The floor is now what a boot actually needs on top of
# whatever else the desktop holds, and the abort line is the old floor.
PRIMARY_RAM_FLOOR_GIB = 180
PRIMARY_RAM_ABORT_GIB = 150
PRIMARY_RAM_WAIT_SECONDS = 600.0
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
    return _post(f"{BACKEND}/api/workers/pause", {"paused": paused})


# True while a pause THIS promoter set is in force. The pause is in-memory in
# the backend, so a successful landing's restart clears it; the paths that
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


def wait_idle(max_wait: float = IDLE_MAX_WAIT, *, drain: bool = True,
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
    """
    global _POOL_PAUSED_BY_US
    if pause_pool and pool_paused() is False and set_pool_paused(True):
        _POOL_PAUSED_BY_US = True
    deadline = time.time() + max_wait
    quiet = 0
    busiest = ""
    armed_at = 0.0
    while time.time() < deadline:
        if drain and time.time() - armed_at >= DRAIN_REFRESH_SECONDS:
            set_drain(True, DRAIN_TTL)
            armed_at = time.time()
        status, body = _get(f"{BACKEND}/health")
        if status == 200 and body:
            turns = body.get("turns") or {}
            busy = (turns.get("active", 1) or turns.get("queued", 1)
                    or turns.get("harness_runs", 0))
            if not busy:
                quiet += 1
                if quiet >= IDLE_QUIET_POLLS:
                    return True, f"idle for {quiet} consecutive polls"
            else:
                quiet = 0  # a turn appearing resets the counter
                busiest = (f"active={turns.get('active')} queued={turns.get('queued')} "
                           f"harness_runs={turns.get('harness_runs')}")
        else:
            quiet = 0
        time.sleep(IDLE_POLL_SECONDS)
    if drain:
        set_drain(False)
    release_pool_pause()
    return False, (f"backend never went idle within {max_wait:.0f}s"
                   + (f" (last: {busiest})" if busiest else ""))


def count_kg_rows() -> int | None:
    import sqlite3
    db = LIVE_ROOT / "_pipeline" / "vault-derived" / "kg.sqlite"
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


def count_vault_files() -> int | None:
    root = Path.home() / "obsidian"
    if not root.is_dir():
        return None
    try:
        return sum(1 for p in root.rglob("*") if p.is_file())
    except OSError:
        return None


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


def promotion_announcement(title: str, n_files: int) -> tuple[str, str]:
    """The toast head and body for a landing. Pure, so it can be pinned.

    Never the round id: it is in the `promoted` ledger row for anyone who
    needs it, and read aloud it is a date one digit at a time. The title is
    the item's name — the one thing a person in the room can act on.
    """
    head = f"Landed: {title}" if title else "Landed a change"
    body = f"{n_files} file{'' if n_files == 1 else 's'} changed. Watching for {int(ERRORS_WINDOW // 60)} minutes."
    return head, body


def _announce_promoted(round_id: str, commit: str, changed: list, title: str = "") -> None:
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
    head, body = promotion_announcement(title, len(changed))
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
def _land_failed(round_id: str, why: str, *, external: bool, **extra) -> None:
    S.append_event({"event": "land_failed", "round_id": round_id, "ok": False,
                    "external_blocker": bool(external), "detail": why[:500], **extra})
    raise PromoteError(why)


# How many times one landing will chase a moving `main`. Each chase is a full
# gate run — minutes — and the third miss means someone is committing faster
# than the loop can retest, which is a reason to stop and say so, not to keep
# up. The round is left rebased and gated; the reaper closes it, the branch is
# kept, and the item comes back.
MAX_REBASES_PER_LANDING = 2

# The chamber (`automod.chamber`). The next round may run its turn and gate
# while the previous promotion is under observation; only the landing needs
# the window closed. So a landing that finds a promotion observed waits for
# it to settle — up to the window plus slack — instead of refusing at once
# and throwing away a gated round. A landing is written `current.json` with
# its own window only after the restart verifies, so the whole window can
# still be ahead of it.
SETTLE_MAX_WAIT = ERRORS_WINDOW + 120.0
SETTLE_POLL_SECONDS = 10.0


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
    max_wait = SETTLE_MAX_WAIT if max_wait is None else float(max_wait)
    poll = SETTLE_POLL_SECONDS if poll is None else float(poll)
    deadline = time.monotonic() + max_wait
    # `observed` is the record the caller decided to wait on. Read again here
    # regardless, and remember the commit from whichever saw it: a promotion
    # that clears between the caller's read and this one must still be proved
    # settled, not assumed.
    waited_on = str((observed or {}).get("commit") or "")
    observed = S.read_current()
    waited_on = waited_on or str((observed or {}).get("commit") or "")
    while observed and time.monotonic() < deadline:
        time.sleep(poll)
        observed = S.read_current()
    why = ""
    if S.is_halted():
        why = f"promotions are halted: {S.HALTED_PATH}"
    elif S.is_broken():
        why = f"guardian is in a BROKEN state: {S.BROKEN_PATH}"
    elif S.read_rollback_request():
        why = "a rollback request is pending — not landing on top of a revert"
    elif observed:
        why = (f"{str(observed.get('commit'))[:8]} is still under observation "
               f"({observed.get('state')}) after waiting {max_wait / 60:.0f} min for it to settle")
    elif waited_on and not _settled(waited_on):
        # `current.json` also clears when the guardian ROLLS BACK: it deletes
        # a rollback request the moment it reads one, and its own error-window
        # rollbacks never write one, so an empty request file proves nothing.
        # Only a `settled` row for the commit waited on says it survived.
        why = (f"{waited_on[:8]} left observation without settling (rolled back?) — "
               f"not landing a round that ran on top of it")
    if why:
        if round_id:
            _land_failed(round_id, why, external=True, waited_for_settle=True)
        raise PromoteError(why)
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
        _land_failed(round_id,
                     f"live tree has uncommitted edits in paths this round also changes: "
                     f"{overlap} — commit or stash the live edit, then land again",
                     external=True, overlap=overlap)

    result: dict = {"round_id": round_id, "commit": head, "parent": live_head,
                    "changed_paths": changed, "dry_run": dry_run,
                    "live_dirty_paths": live_dirty[:20], "title": title}
    if dry_run:
        result["would_promote"] = True
        return result

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
    S.write_verified(S.CURRENT_PATH, current)   # raises unless it round-trips

    # ── idle gate + drain ──────────────────────────────────────────────
    ok, why = wait_idle()
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
    set_drain(True, DRAIN_TTL)
    merged = False
    try:
        status, body = _get(f"{BACKEND}/health")
        turns = (body or {}).get("turns") or {}
        if turns.get("active") or turns.get("queued") or turns.get("harness_runs"):
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
            set_drain(True, DRAIN_TTL)
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
            S.write_verified(S.CURRENT_PATH, current)

        # ── land ───────────────────────────────────────────────────────
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

        venv_clone = Path(worktree) / ".venvs" / "lloyd"
        if venv_clone.exists():
            import os
            venvs = live / ".venvs"
            os.rename(venvs / "lloyd", venvs / "lloyd.prev")
            os.rename(venv_clone, venvs / "lloyd")
            current["venv_swapped"] = True
            S.write_verified(S.CURRENT_PATH, current)

        # Service definitions the diff changed must reach the running system
        # BEFORE the restart, or the restart re-reads the old ones.
        service_notes = _apply_service_changes(changed)

        # ── restart, MCP first ─────────────────────────────────────────
        for program, health in (("lloyd-mcp", MCP_HEALTH), ("lloyd-backend", f"{BACKEND}/health")):
            # Refresh the lease before each leg. The lease is 120s and this
            # loop can legitimately spend 90s per service waiting on health,
            # so a single lease taken before the merge could expire mid-restart
            # and hand the guardian its own deploy to judge.
            S.set_pause(RESTART_LEASE)
            ok, msg = restart_process(program)
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
        body = _wait_for_commit(f"{BACKEND}/health", VERIFY_COMMIT_BUDGET)
        actual = (body or {}).get("commit")
        if actual != head:
            raise PromoteError(f"backend reports commit {actual}, expected {head} "
                               "— the restart did not pick up the new code")
        if current.get("boot_id") and (body or {}).get("boot_id") == current["boot_id"]:
            raise PromoteError("backend boot_id unchanged — the process was never replaced")

        if any(p.startswith("web/") for p in changed):
            alive, note = _frontend_alive()
            if not alive:
                raise PromoteError(f"frontend unreachable after landing ({note}) at {FRONTEND_URL}")

        # The code is live and verified: start the clock now.
        landed = time.time()
        current["state"] = "observing"
        current["landed_at"] = S.now_iso()
        current["landed_ts"] = landed
        current["errors_until_ts"] = landed + ERRORS_WINDOW
        current["boot_id"] = (body or {}).get("boot_id")
        current["service_changes"] = service_notes
        S.write_verified(S.CURRENT_PATH, current)
        result["service_changes"] = service_notes
        S.append_event({"event": "promoted", "round_id": round_id, "title": title,
                        "commit": head, "parent": live_head, "changed_paths": changed,
                        "vault_commits": current.get("vault_commits") or [],
                        "tree_hash": tree_hash, "service_changes": service_notes,
                        "errors_until": current["errors_until_ts"]})
        _announce_promoted(round_id, head, changed, title)
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
    """
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
        raise PromoteError(why)
    started = time.time()
    done: list[str] = []
    try:
        for program in programs:
            S.set_pause(RESTART_LEASE)   # refreshed per leg, as the promoter does
            if program == PRIMARY_PROGRAM:
                ok, msg = _restart_primary()
            else:
                ok, msg = restart_process(program)
            if not ok:
                raise PromoteError(f"restart {program} failed: {msg}")
            if not _wait_health(health_for[program], budget_for.get(program, 90.0)):
                raise PromoteError(f"{program} never became healthy after restart")
            done.append(program)
        S.append_event({"event": "restart", "programs": list(done), "by": "human",
                        "reason": reason[:300], "seconds": round(time.time() - started, 1)})
        return {"restarted": done, "seconds": round(time.time() - started, 1),
                "idle": why, "reason": reason}
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
