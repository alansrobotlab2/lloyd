#!/usr/bin/env python3
"""Lloyd MCP Server: an OS ceiling on every Bash-spawned process subtree (#2578).

Bash has always bounded where a command may **write** — the protected-path
bubblewrap profile in `_path_sandbox`, which makes a protected write "fail in the
kernel rather than depending on a parser recognising the shape" — and how **long** it
may run: 120 s default, 600 s cap, SIGTERM→SIGKILL on the process group. Nothing
bounded what a command could *consume*, so the module docstring's claim that "the one
thing enforced here rather than upstream is the substrate below" was true only of the
filesystem. Before this module, `git grep -nE "RLIMIT|setrlimit|prlimit" -- "*.py"`
returned 3 hits and all three were prose; no spawn path carried a limit.

That gap had a known owner. `agent-services/systemd/agent-supervisord.service` carries
`Slice=lloyd.slice` (pinned by #1337) because systemd-oomd monitors exactly one cgroup
— `…/user@1000.service/app.slice`, 50% for 20 s — and an engine boot drives its own
cgroup to ~226 GiB of this box's 251 GiB, which had killed the whole unit and all 33
programs five times. Relocating the unit cured that. Nobody wrote down the
consequence: with the unit out of the monitored cgroup there was then **no automatic
bound at all** on what a Bash-spawned subtree may consume, and the failure mode for a
runaway command changed from "oomd kills the unit" to "the kernel picks a victim" —
plausibly one of the engine daemons, which is the difference between a failed command
and a nine-minute boot.

The shape follows the item's own argument: prevention below the model, not detection
above it. `app/harness/safety.py` cannot match a runaway and must not try — a command
that forks or allocates is not destructive-*shaped*, so any string that caught it would
be a guess about intent. The ceiling catches it one layer down, where shape does not
matter.

    systemd-run --user --scope --slice=lloyd.slice --unit=<name> -q
        -p MemoryMax=… -p MemorySwapMax=0 -p TasksMax=…
        -- <the argv the sandbox already built>

The wrapper nests *inside* bwrap's argv, never around it: at all three
`create_subprocess_exec` sites in `builtin_bash.py` the argv that arrives is already
`_path_sandbox.wrap(...)`'s or `_tool_sandbox.bwrap_argv(...)`'s, and `apply` prepends,
so a command loses neither the path sandbox nor the ceiling. `--scope`, not a transient
service, because the caller must remain the foreground process of its own cgroup: a
service returns before its command has run, which would break the tool's timeout, its
`os.killpg` group-kill and bwrap's `--die-with-parent` in one move.

Three properties this module enforces on its own arguments are not decoration:

* `--slice=lloyd.slice` is *required* (`CEILING_SLICE`) and never defaulted. The reason
  the supervisord unit lives where it lives is that the other slice in this user
  manager — `app.slice` — is the one systemd-oomd watches, so a ceiling created there
  puts a runaway child's memory pressure straight back into the cgroup that has killed
  this box's whole agent stack five times. Every ceiling node in
  `tests/test_bash_resource_ceiling.py` reads `/proc/self/cgroup` from a child that went
  through a real spawn site, so the assertion is about where the kernel put the subtree,
  not about a string in an argv.
* `MemorySwapMax=0` because this box's swap is a 251.5 GiB zram device (`swapon
  --show`: `/dev/zram0 251.5G`). Measured on this box, both directions: the same
  single-process allocator under `-p MemoryMax=256M -p MemorySwapMax=0` is killed by the
  memory controller in 0.08 s (rc=-9), and the same allocator under `-p MemoryMax=256M`
  with swap left to inherit the slice's `infinity` was still running and uncapped at a
  90 s wall clock. A cap on RAM alone is a cap on nothing here, and the damage only
  moves into compressed pages that are still host RAM and still the engine's to lose.
* `TasksMax` is a real bound *because* the scope is per-subtree. `RLIMIT_NPROC`, the
  tempting one-liner, is enforced against the calling UID's system-wide process count —
  and everything here, engines included, runs as one user — so a low `nproc` is a
  footgun aimed at the whole stack rather than a ceiling on a child. That is ruled out
  by a test, in the file, not in prose.

**What the ceiling must not do.** It must not slow work that is under the limit, and
"it finished" is not evidence of that: every node here that reports completion also
reports a wall-clock budget, and a foreground command whose child backgrounds a
descendant costs +0.02 s with the ceiling armed (measured 6.03 s against 6.00 s for the
identical command unceiled — the naive form and the fully-detached form both). And it
must not throttle the engines: `scripts/automod/evalpin.py`, which launches the vLLM
boots, is checked by test to carry no ceiling call, because a ceiling that throttles an
engine boot is a worse defect than the one it fixes.

**Reporting, which is why the module is bigger than a flag.** `systemd-run` prints
"Running as unit: …" (suppressed here with `-q`) and otherwise never says why a child
died; an OOM kill arrives as rc=-9 with no message at all, and a fork-EAGAIN fan-out exits
with whatever its shell felt like. So `apply()` starts a sampler *in the aggregator* — a
different cgroup, and the only process entitled to fork — that mirrors the scope's own
counters into a report file, and the tool result turns them into a `[resource ceiling]`
line naming the limit that tripped and its configured value.

Outside the scope, and not by preference. The first implementation was a `trap … EXIT`
inside the wrapped command, and the planted allocator produced nothing at all from it:
this box answers an over-cap cgroup by signal 9 to **every process in it**, the wrapper's
own shell included — which is why that child arrived as rc=-9 rather than the 137 a shell
would otherwise have reported — so no trap was left to run. A reader on the other side of
the boundary sees the same numbers, needs no shell in the payload, costs no extra process
per call, and cannot be killed by the event it exists to describe. And because
`memory.peak`/`pids.peak` are in-kernel high-water marks that count the dead, a poll every
50 ms loses nothing to the interval: the peak is the peak whether or not anyone was
looking.

Both directions of the measurement are owed as a pair — commands the ceiling killed, and
benign work that came near it — and the values are derived from a real distribution, in
`MEASURED_PEAK_*` below, not guessed.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("lloyd-builtin-resource-ceiling")

# ── the ceiling, and the measurement each number came from ────────────────────
#
# MEASURED 2026-10-12 on this box, in a transient scope of its own inside lloyd.slice with
# the engine daemons resident (vLLM's EngineCore at ~13.9 GiB, chrome at ~7.4 GiB), reading
# the scope's own `memory.peak`/`pids.peak` every 5 s:
#
#   `pytest tests/` (the whole suite, `-q`)  peak 4096630784 B (3.82 GiB)  peak 300 tasks
#
# That sample stopped at 68% — the run was terminated to free the box for this round's own
# gate — so it is a floor over 68% of the suite, not a p99 over a finished run, and both
# constants below carry extra margin specifically for the 32% that was never sampled rather
# than pretending the factor of 3 covered a complete pass. Re-measure with:
#   systemd-run --user --scope --slice=lloyd.slice -- \
#     /bin/bash -c 'CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); \
#       pytest tests/ -q -p no:cacheprovider & p=$!; \
#       while kill -0 $p; do cat $CG/memory.peak $CG/pids.peak; sleep 5; done'

MEASURED_PEAK_RAM_BYTES = 4_096_630_784   # 3.82 GiB over 68% of the suite
MEASURED_PEAK_PIDS = 300                  # tasks, same scope, same window
HEADROOM_FACTOR = 3           # stated rather than fitted, so it can be argued with

MEMORY_MAX = "16G"            # MEASURED_PEAK_RAM_BYTES × HEADROOM_FACTOR, rounded up to
                              # 4.2× rather than 3× because the sample covers 68% of the
                              # suite and the margin also has to pay for the part nobody
                              # measured. A *protection* limit, not a budget: above the
                              # largest legitimate thing a session does, below the point
                              # where the OOM killer reaches for an engine daemon. `free` at
                              # measure time: 193 GiB used of 251, 57 GiB available,
                              # `VLLM::EngineCore` resident at 13.9 GiB — so 16G is half of
                              # what is actually free, and a runaway that filled 32G would
                              # have taken the box into swap before the kernel stopped it.
                              # The item's triage quotes a full run needing ">8 GiB with the
                              # engine daemons resident"; this round's own sample topped out
                              # at 3.82 GiB and stopped at 68%, so that figure is carried as
                              # NOT confirmed by this measurement, and 16G is above both.
MEMORY_SWAP_MAX = "0"         # see the docstring: 251.5 GiB of zram absorbs any cap
                              # that memory alone is given.
TASKS_MAX = "1024"            # MEASURED_PEAK_PIDS × HEADROOM_FACTOR = 900, rounded up to
                              # the next power of two: 3.4× the 300 tasks the suite peaked
                              # at, and far below `DefaultLimitNPROC`, so it stops a bomb
                              # without touching anything else. It is NOT a tight fit — a
                              # legitimate parallel build is the thing to watch in the
                              # post-landing near-miss report.

#: The slice every ceiling scope is created in, and the ONLY one this module accepts.
#: Never `app.slice`: that is the cgroup systemd-oomd monitors (#1337's reason).
CEILING_SLICE = "lloyd.slice"

#: Per-call scope units are short-lived; this prefix is what a stray unit or a
#: `journalctl` line is recognised by.
UNIT_PREFIX = "lloyd-bash-ceiling"

#: The unified cgroup filesystem — `stat -fc %T /sys/fs/cgroup` reads `cgroup2fs` here.
#: The sampler reads through this path, never through a path the child reports, so the two
#: agree only if the kernel really put the child where `--slice=` names it. That is what
#: `test_the_samplers_path_and_the_childs_agree` pins: a ceiling reading the wrong
#: directory would issue a confident verdict about a process it never watched.
CGROUP_ROOT = "/sys/fs/cgroup"

#: Where a call's counters are mirrored to. Under `~/lloyd-data`, never the checkout: a
#: round must not create runtime files in the tree that ships code.
_DATA_ROOT = Path(os.environ.get("LLOYD_DATA", str(Path.home() / "lloyd-data")))
REPORT_DIR = _DATA_ROOT / "_pipeline" / "ceiling"

#: How long a row is kept. ONE number, read by both owners of the bound: this module's
#: :func:`_prune_samples` applies it from inside the append, and
#: `scripts/groundskeeper/retention-sweep.py` imports it as
#: `CEILING_SAMPLES_MAX_AGE_DAYS` for its own prune of the same file. The second owner
#: exists because the first only runs when a Bash call happens: a box whose MCP service is
#: down, or one that simply stops running Bash, keeps whatever the file had reached and
#: says nothing about it anywhere — a bound that tightens only under load is a side effect
#: of load, not a bound. The two owners share this constant rather than each holding a
#: number, because a row store pruned on two horizons is a distribution whose window
#: depends on which of them last ran, and the window is the only thing the p99 means.
#:
#: 90 days rather than the sweep's uniform 30, because the horizon must sit outside the
#: window the rows exist for: #2578 owes a p99 over >=7 days of Bash children that includes
#: one full `pytest tests/` run, and a horizon that empties the evidence before it can be
#: read is not bounding a store, it is deleting it.
SAMPLE_MAX_AGE_DAYS = 90


def samples_path(data_root: "str | os.PathLike[str] | None" = None) -> Path:
    """Where the durable per-call rows live, for a caller that knows a different data root.

    The sweep needs the path rather than the constant below because it resolves its own
    `DATA_ROOT` by three rules (an env override, then the tree, then home) and then redirects
    it under a test; a sweep that pruned whatever root this module happened to import into
    would be pruning a tree its own report does not name — the mistake
    `tests/test_retention_sweep.py`'s unredirected-paths guard exists to catch. Same layout as
    `REPORT_DIR`, one file instead of a directory, and the sweep reaches the store through
    this function rather than restating the layout.
    """
    root = _DATA_ROOT if data_root is None else Path(data_root)
    return root / "_pipeline" / "ceiling-samples.jsonl"


#: The durable per-call sample store: **one JSON row per completed Bash call**, carrying
#: `ts, rc, mem_peak, mem_max, pids_peak, pids_max, mem_frac, oom_kill_delta, source, unit`.
#: This is the record, not a mirror of it. `finish()`'s `logger.info` line is a convenience,
#: and the reason that used to be given for deleting the report — that "the same line in the
#: already-rotated MCP log carries the identical numbers" — was false on this box: `grep -c
#: "resource ceiling sample"` returns 0 across `logs/mcp.log`, its ten rotations and
#: `logs/lloyd-agent-worker.log`, because this logger reaches no file at all (that file is
#: uvicorn access lines). So the only thing a call left behind was the `.report` of a spawn
#: that never reached `finish` — 100 unread files across 2026-10-12→17, against 429 Bash calls
#: counted in the 25 newest session stores — and the peak distribution #2578's owed entry asks
#: a p99 of did not exist.
#:
#: `source` is what keeps a row honest about how it was measured: `"scope"` or
#: `"scope+report"` means the scope's own cgroup files answered `settle`'s final read;
#: `"report"` means systemd had already collected the scope and the sampler's mirrored
#: snapshot is all there is; `"none"` means neither arrived, so the row is a timestamp and an
#: rc about a call whose consumption is unknown — a p99 taken over a store that contains those
#: is a number about nothing unless the reader filters on this field, which is why `status()`
#: reports `sample_unmeasured` beside `sample_rows`. `unit` lets a reader prove one row per
#: call instead of one row per some multiple of calls.
#:
#: Bounded twice, on one horizon: :func:`_prune_samples` from inside the append, and the
#: daily `scripts/groundskeeper/retention-sweep.py` run, whose store this is now — its
#: `CEILING_SAMPLES_MAX_AGE_DAYS` IS the `SAMPLE_MAX_AGE_DAYS` above, imported rather than
#: copied, and it reaches this file through :func:`samples_path` rather than by restating
#: the layout. That pair is what licenses one row per call.
SAMPLE_STORE = samples_path()

#: A wall-clock budget on the ceiling's own nodes, so "completed" can never be reported
#: without "and it was not held". Loose against each node's real cost on purpose: what
#: is pinned is that the ceiling imposes no ceiling on *time*, not how fast a box is.
CEILING_NODE_BUDGET_S = 60.0

#: The suffixes systemd accepts for a memory property, so the ceiling's own value can be
#: compared against the integer `memory.max` the kernel reports without a test parsing
#: the string a second time and disagreeing about whether G means 1000 or 1024 cubed.
_SUFFIX_MULTIPLIERS = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def to_bytes(value: str) -> int:
    """A systemd memory property (`32G`, `512M`, a bare byte count) as an integer."""
    text = str(value).strip().upper()
    mult = _SUFFIX_MULTIPLIERS.get(text[-1:], 1)
    return int(text[:-1] if mult > 1 else text) * mult


def memory_bytes() -> int:
    """:data:`MEMORY_MAX` in bytes — the number `memory.max` is expected to read."""
    return to_bytes(MEMORY_MAX)


_probe_lock = threading.Lock()
_probe: dict[str, object] = {"ok": None, "at": 0.0, "error": ""}
_PROBE_RETRY_S = 60.0

#: The verdict for the call in flight, so `_bash` can append the line at the same place
#: it appends `[exit code: N]`. A contextvar for the reason
#: `builtin_bash._bash_failed` gives: `_bash` has six return points, its string return is
#: what the harness's spill/truncation path expects, and this process serves many calls —
#: a module-level "last verdict" would be read by whichever call finished first.
_verdict: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "resource_ceiling_verdict", default=None)


# ── the environment the LAUNCHER needs ────────────────────────────────────────

#: systemd's own convention for a per-uid runtime directory, which is where the user
#: manager's D-Bus socket lives (`$XDG_RUNTIME_DIR/bus`). The variable is set by
#: `systemd-logind`, not by systemd-run, so any caller that builds a child env by hand can
#: leave it out — and the process that then execs `systemd-run --user` cannot reach its own
#: manager.
_RUN_DIR_TEMPLATE = "/run/user/{uid}"


def user_runtime_dir() -> "str | None":
    """The per-uid runtime directory, or None if there is no live user manager here.

    `$XDG_RUNTIME_DIR` is the authority; this function is only consulted when the caller's
    environment does not carry it, and it accepts the conventional path solely on the
    conditions that make it the real one: it exists, it is a directory, and it is ours
    (`/run/user/<uid>` is created 0700 by logind, so an owned one is a live session and a
    missing one means there is no manager to reach). Nothing is created and nothing is
    guessed at another path.
    """
    path = _RUN_DIR_TEMPLATE.format(uid=os.getuid())
    try:
        if os.stat(path).st_uid == os.getuid() and os.path.isdir(path):
            return path
    except OSError:
        return None
    return None


def launcher_env(env: "dict[str, str] | None") -> "dict[str, str] | None":
    """The environment for a spawn whose argv begins with this module's wrapper.

    `systemd-run --user --scope` does not create a cgroup by itself: it asks the **user
    manager** over `$XDG_RUNTIME_DIR/bus`, so a caller with a curated environment that omits
    that variable gets `Failed to connect to user scope bus via local transport:
    $DBUS_SESSION_BUS_ADDRESS and $XDG_RUNTIME_DIR not defined`, rc=1, from the launcher —
    and `apply`, which fails open like `_path_sandbox`, then hands back the *bare* argv. The
    command runs, unbounded, and the only trace is a `fallback` in `/state`. That is the
    worst shape this module can have: the bound silently missing on exactly the curated
    spawns a sandbox is supposed to make safer. Measured with
    `env -i PATH=/usr/bin:/bin systemd-run --user --scope --slice=lloyd.slice -- /bin/true`
    → rc=1; adding only `XDG_RUNTIME_DIR=/run/user/1000` → rc=0 and `/proc/self/cgroup`
    reads `…/lloyd.slice/run-p564999-i25701169.scope`.

    Only the *path* is supplied, never a credential or a session bus address: when
    `$XDG_RUNTIME_DIR` is already set this returns the caller's env untouched, so a spawn
    that deliberately trimmed it stays trimmed unless its own uid has a live session.

    It is not a privilege leak into a sandboxed child. Both Bash sandbox profiles hide the
    directory's contents — `agent_mcp/_tool_sandbox.py`'s bench/eval profile is `--ro-bind /
    /` plus `--tmpfs /tmp /run /var/tmp`, and `_path_sandbox.profile_argv` runs
    `--unshare-all --die-with-parent`, whose network and ipc mounts come from a fresh tmpfs
    — so the variable names a directory the child cannot open a socket in. The authority to
    reach the manager stays with the process that named the scope, which is this one.
    """
    base = dict(os.environ if env is None else env)
    if not base.get("XDG_RUNTIME_DIR"):
        runtime = user_runtime_dir()
        if runtime:
            base["XDG_RUNTIME_DIR"] = runtime
    return base


# ── availability ──────────────────────────────────────────────────────────────

_READ_BACK = (
    'CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); '
    'printf \'{"slice":"%s","pids.max":%s,"memory.max":%s,"memory.swap.max":%s}\\n\' '
    '"$(basename "$(dirname "$CG")")" "$(cat "$CG/pids.max")" "$(cat "$CG/memory.max")" '
    '"$(cat "$CG/memory.swap.max")"'
)


def available() -> tuple[bool, str]:
    """Whether a ceiling can actually be armed here.

    Cached on success; a failure is re-probed after a minute, so a transient namespace
    problem does not strand the box unbounded until the aggregator restarts. Same shape
    and same reasoning as `_path_sandbox.available()`.

    The probe is the mechanism itself, not a `--version` that only proves a binary: it
    creates a scope carrying this module's own properties and reads the limits back out
    of the scope's cgroup directory. That is the only way to see the two failures that
    would otherwise read as armed — a manager that will not delegate a subtree in
    `lloyd.slice` to an unprivileged caller (the slice check's failure mode, and it
    fails by *not bounding*, so an argv-only test could never notice), and a property
    that did not land, leaving `memory.swap.max` at the slice's `infinity`.
    """
    if os.environ.get("LLOYD_DISABLE_BASH_CEILING", "").strip().lower() in (
            "1", "true", "yes", "on"):
        # An operator switch, reported as a reason rather than silently honoured:
        # `apply` passes the argv through and `status()` says `fallback: true`.
        return False, "disabled by LLOYD_DISABLE_BASH_CEILING"
    with _probe_lock:
        now = time.monotonic()
        if _probe["ok"] is True:
            return True, ""
        if _probe["ok"] is False and now - float(_probe["at"]) < _PROBE_RETRY_S:
            return False, str(_probe["error"])
        if shutil.which("systemd-run") is None:
            ok, err = False, "systemd-run is not installed"
        else:
            argv = scope_argv(_safe_unit("probe"), ["/bin/sh", "-c", _READ_BACK])
            try:
                # `launcher_env`, because the probe IS a spawn of the launcher: an
                # environment that cannot reach the manager answers "unavailable" here,
                # which is what makes every later command unbounded.
                proc = subprocess.run(argv, capture_output=True, timeout=30,
                                      check=False, env=launcher_env(None))
                got = _load_json(proc.stdout)
                ok, err = _probe_verdict(proc, got)
            except Exception as exc:  # noqa: BLE001
                ok, err = False, f"ceiling probe failed: {exc}"
        _probe.update(ok=ok, at=now, error=err)
        if not ok:
            logger.warning(
                "resource ceiling UNAVAILABLE (%s); Bash children run UNBOUNDED — "
                "this box then has no automatic bound at all on what a command can "
                "consume, which is the gap #2578 exists to close", err)
        return ok, err


def _load_json(text: bytes | str) -> dict:
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    try:
        return json.loads(text.strip() or "{}")
    except json.JSONDecodeError:
        return {}


def _probe_verdict(proc, got: dict) -> tuple[bool, str]:
    """Armed, or not, with the reason in the words an operator would act on."""
    if proc.returncode != 0:
        return False, ("systemd-run exited %d: %s" % (
            proc.returncode,
            proc.stderr.decode(errors="replace").strip()[:200]))
    if not got:
        return False, ("the ceiling scope ran but reported no limits back, so the "
                       "cgroup is not readable from inside it")
    if got.get("slice") != CEILING_SLICE:
        return False, (f"the scope landed in {got.get('slice')!r}, not "
                       f"{CEILING_SLICE!r} — a ceiling in the wrong slice is how "
                       "#1337's whole-unit kill comes back")
    if str(got.get("pids.max")) != TASKS_MAX:
        return False, (f"pids.max reads {got.get('pids.max')!r}, configured "
                       f"TasksMax={TASKS_MAX}")
    if str(got.get("memory.swap.max")) != "0":
        return False, (f"memory.swap.max reads {got.get('memory.swap.max')!r}, not 0 — "
                       "the box's 251.5 GiB of zram would absorb the memory cap")
    return True, ""


# ── the wrapper ───────────────────────────────────────────────────────────────

def _safe_unit(extra: str = "") -> str:
    """A unit name unique to this call. systemd unit names are `[a-zA-Z0-9:.@_-]`; a
    collision with a live unit is a hard failure rather than a queue, so the suffix is a
    random token — a pid alone would repeat across the many calls one process makes.

    Neither part of the token identifies the call that made it: the leading number is a
    wall-clock read (`int(time.time()) % 100000`), not a pid, so a scope name read out of
    `journalctl -k` can be matched to a *time* but never to a spawner. #2725's planned
    `ps`-at-kill-time diagnosis keyed on exactly that number and could not have worked."""
    suffix = f"{int(time.time()) % 100000}-{secrets.token_hex(4)}"
    return f"{UNIT_PREFIX}-{suffix}{('-' + extra) if extra else ''}"


def _is_unit_name(text: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9:.@_-]{1,200}", text or ""))


def slice_of(argv: list[str]) -> str | None:
    """The `--slice=` value in a wrapper argv, or None. Exists so a test can assert the
    slice instead of trusting that one was passed."""
    for tok in argv:
        if tok.startswith("--slice="):
            return tok.split("=", 1)[1]
    return None


def scope_argv(unit: str, inner_argv: list[str]) -> list[str]:
    """Wrap an already-built argv in a transient scope carrying the ceiling.

    `-q` suppresses "Running as unit: …; invocation ID: …", which on the Bash tool would
    land in the merged stream ahead of the command's own first line. Whether that banner
    should be stripped or surfaced is a live-traffic question and is left on the item;
    `-q` is the choice that cannot corrupt a command's output.

    `unit` must already be a legal unit name. Nothing here sanitises: a silently mangled
    name is a unit nobody can find in the journal afterwards, so this refuses.
    """
    if not _is_unit_name(unit):
        raise ValueError(f"not a usable systemd unit name: {unit!r}")
    if not inner_argv:
        raise ValueError("nothing to wrap")
    return ["systemd-run", "--user", "--scope", f"--slice={CEILING_SLICE}",
            f"--unit={unit}", "-q",
            "-p", f"MemoryMax={MEMORY_MAX}",
            "-p", f"MemorySwapMax={MEMORY_SWAP_MAX}",
            "-p", f"TasksMax={TASKS_MAX}",
            "--", *inner_argv]


# ── the out-of-scope sampler ───────────────────────────────────────────────────

#: How long the sampler waits for the manager to create the scope's directory, and how
#: often it reads it. The wait is generous because the aggregator only *names* the unit:
#: `systemd-run` asks the manager over D-Bus for the scope, and that round trip has to
#: land before anything can be read. The interval is 50 ms because the two peak files are
#: kernel high-water marks — sampling faster than the kernel updates them buys nothing,
#: and 50 ms is far inside the ~100 ms a fan-out needs to exhaust `pids.max`.
_APPEAR_WAIT_S = 5.0
# 10 ms, and the number is a measured one rather than a taste. A fan-out that trips
# `pids.max` and does NOT hold its children — the shape the clause-2 plant takes —
# runs its whole fork storm and exits in 44 ms measured, and the scope's cgroup
# directory is collected within ~20 ms of that exit (probed at 20 ms spacing: readable
# at +0 ms, ENOENT at +20 ms, three times). At the 50 ms this used to be, the sampler
# wrote exactly one snapshot taken before the storm reached the cap, and the note that
# names the limit was therefore never produced for the one shape that most needs it:
# `finish`'s live re-read and slice-delta fallback both arrive after the directory is
# gone, and unlike the memory case there is no slice-level counter to fall back to —
# `pids.events.max` on the PARENT does not increment for a descendant's cap hit
# (measured: slice stayed at its own count across three blocked-at-47 runs). 10 ms
# puts four reads inside a 44 ms storm; `settle` then adds a final live read.
_SAMPLE_INTERVAL_S = 0.01

_COUNTERS = {
    "mem_peak": "memory.peak",
    "mem_current": "memory.current",
    "pids_peak": "pids.peak",
    "pids_current": "pids.current",
    "pids_max": "pids.max",
    "mem_max": "memory.max",
    "swap_max": "memory.swap.max",
}


def _user_manager_cgroup() -> str:
    """This process's cgroup, truncated at its user manager — on this box
    `/user.slice/user-1000.slice/user@1000.service`.

    A scope created by a `--user` manager with `--slice=lloyd.slice` is placed inside that
    manager's own tree, **not** at `/sys/fs/cgroup/lloyd.slice`. Measured here: a process
    sitting in `…/user@1000.service/lloyd.slice/agent-supervisord.service` asked for
    `--slice=lloyd.slice` and got its scope at
    `/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/lloyd.slice/<unit>.scope`.
    Assuming the flat path would leave the sampler reading a directory that never exists,
    writing nothing, and staying silent about a bound that really was applied — so the
    ceiling would appear to have caught nothing while it had in fact killed the command.
    """
    with open("/proc/self/cgroup", encoding="utf-8") as fh:
        line = fh.read().splitlines()[0]
    rel = line.split("::", 1)[-1]                     # cgroup v2: `0::/some/path`
    match = re.search(r"^(.*/user@\d+\.service)", rel)
    if match:
        return match.group(1)
    # No user manager in the path (a container, or a runner outside the login session):
    # one level above wherever this process lives is still where the manager puts a child
    # scope, and `test_the_samplers_path_and_the_childs_agree` is what says so.
    return rel.rsplit("/", 1)[0] or "/"


def cgroup_dir(unit: str) -> str:
    """The scope's cgroup directory: the user manager's root, then `CEILING_SLICE`, then
    the unit. Derived rather than discovered, so the sampler can be running before the
    directory exists.

    Pinned by `test_the_samplers_path_and_the_childs_agree`, which reads the same path out
    of a child's own `/proc/self/cgroup` through a real spawn site. The one way this module
    could invent a verdict about a process it never watched is to read the wrong directory,
    and the child is the only witness to where the kernel actually put it.
    """
    return f"{CGROUP_ROOT}{_user_manager_cgroup()}/{CEILING_SLICE}/{unit}.scope"


def _read_file(path: str) -> str:
    """One cgroup file, or "" when it would not open — absence here is normal."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _flat_events(cgdir: str, name: str) -> str | None:
    """A cgroup `*.events` file as `key=value;…`, or None if it would not open.

    The kernel writes these as `key value` lines; the report keeps them `key=value`
    because `events_counts` parses that shape and one parser is the point.
    """
    try:
        with open(f"{cgdir}/{name}", encoding="utf-8") as fh:
            pairs = [line.split() for line in fh]
    except OSError:
        return None
    return ";".join(f"{p[0]}={p[1]}" for p in pairs if len(p) == 2)


def _read_counters(cgdir: str) -> dict[str, str] | None:
    """One read of the scope's counters, or None once its directory is gone."""
    memory_events = _flat_events(cgdir, "memory.events")
    if memory_events is None:
        return None                  # the directory is gone: the run is over
    pids_events = _flat_events(cgdir, "pids.events") or ""
    pmax = _read_file(f"{cgdir}/pids.max")
    out = {"scope": cgdir,
           "memory_events": memory_events,
           "pids_events": pids_events,
           # The cap THIS run asked for, read from the scope rather than from the
           # module constant, so a call that ran under a different value reports
           # that value. Written once at creation and never written again: this
           # function has exactly one write surface and it is `report`.
           "tasks_max": pmax or TASKS_MAX,
           "memory_max": _read_file(f"{cgdir}/memory.max") or MEMORY_MAX,
           "memory_swap_max": _read_file(f"{cgdir}/memory.swap.max") or MEMORY_SWAP_MAX}
    for key, name in _COUNTERS.items():
        try:
            with open(f"{cgdir}/{name}", encoding="utf-8") as fh:
                out[key] = fh.read().strip()
        except OSError:
            out[key] = ""
    return out


def _write_report(report: Path, fields: dict[str, str]) -> None:
    """The counters into `report`, via a rename, so no reader ever sees a half-written set.

    A verdict assembled from a partial peak and a whole cap is a wrong verdict with a
    number in it, which is worse than no verdict at all."""
    tmp = report.with_suffix(report.suffix + ".tmp")
    try:
        tmp.write_text("\n".join(f"{k}={v}" for k, v in fields.items()) + "\n",
                       encoding="utf-8")
        tmp.replace(report)
    except OSError:
        logger.debug("ceiling report could not be written", exc_info=True)


def _sample(cgdir: str, report: Path, stop: threading.Event) -> None:
    """Mirror the scope's counters into `report` until its directory disappears.

    One daemon thread per live Bash call, eight small reads every 50 ms, gone with the
    scope and never outliving the aggregator. It forks nothing, which is the property that
    makes it usable at all: a reporter that forks cannot report on the one failure —
    `pids.max` exhausted — that it most needs to describe.
    """
    waited = 0.0
    while not stop.is_set():
        fields = _read_counters(cgdir)
        if fields is None:
            if waited >= _APPEAR_WAIT_S:
                return                  # never armed: write nothing, claim nothing
            stop.wait(0.02)
            waited += 0.02
            continue
        _write_report(report, fields)
        stop.wait(_SAMPLE_INTERVAL_S)


def _higher_of(key: str, old: str | None, new: str) -> str:
    """The larger of two readings of one counter file, for merging a stale snapshot.

    Peaks are high-water marks and event counters only ever rise, so "the later read"
    is never less true — but a merge that simply preferred the newer string would drop a
    peak the sampler caught and the live read missed (the scope's `memory.peak` can be
    collected between the two). Non-numeric fields (`scope`, the `*_max` settings) are
    not quantities, and the live read's value is the one that was in force.
    """
    if old is None:
        return new
    try:
        return new if int(new) >= int(old) else old
    except ValueError:
        return new


def slice_cgroup_dir() -> str:
    """The slice directory the scope is created in: `<CGROUP_ROOT><user manager>/lloyd.slice`.

    Read before every spawn, and again at `finish()`, for one reason: a scope that is OOM
    killed is garbage-collected with its counters, and this round showed that losing that
    race is the common case, not the rare one — the wrapper's shell dies in the same group
    kill as the payload, the unit stops, and systemd removes the directory whenever it
    gets to it. The slice's `memory.events` is cumulative and outlives every scope inside
    it, so reading it before and after a call detects that *a* memory kill happened without
    needing to catch anything in flight.

    What it cannot do is say whose. Every concurrent subtree in `lloyd.slice` shares this
    counter, so a delta licenses exactly one claim — a kill happened somewhere in the slice
    during the window — and `tripped_limit` words it that way; the sentence that this
    call's subtree was killed comes only from the scope's own counters, which the sampler
    does catch more often than the peaks. Detection never races; attribution is graded by
    which witness arrived.

    Measured here: five planted allocators under `MemoryMax=256M` each moved the slice's
    `oom_kill` by one, and each also left `memory.events oom_kill=1` in its own report —
    the three that allocated in 16 MiB strides never got a `memory.peak` into the report,
    the two that paused between chunks did.
    """
    return f"{CGROUP_ROOT}{_user_manager_cgroup()}/{CEILING_SLICE}"


def _slice_events() -> dict[str, int]:
    """`memory.events` of the slice as counts; {} if the file cannot be read, which the
    caller must treat as "no baseline", never as "no kills"."""
    try:
        with open(f"{slice_cgroup_dir()}/memory.events", encoding="utf-8") as fh:
            body = fh.read()
    except OSError:
        return {}
    out: dict[str, int] = {}
    for line in body.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


def _read_report_file(report: Path) -> dict[str, str]:
    if not report.is_file():
        return {}
    try:
        text = report.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


class CeilingRun:
    """One wrapped spawn: the unit it named, the report that fills it, the sampler between
    them, and the slice's counter baseline from before any of it started.

    Opaque to `builtin_bash`, which hands it back to :func:`finish` and nothing else.
    """

    __slots__ = ("unit", "report", "cgdir", "baseline", "_stop", "_thread")

    def __init__(self, unit: str) -> None:
        self.unit = unit
        self.report = REPORT_DIR / f"{unit}.report"
        self.cgdir = cgroup_dir(unit)
        #: Cumulative slice counters, read synchronously here — in the calling thread,
        #: before the spawn that may die within 50 ms. This is the part of the measurement
        #: that must not race, because it is the part that decides whether a memory limit
        #: fired at all; the scope's own peak files are the optional detail on top.
        self.baseline = _slice_events()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            self.report.unlink(missing_ok=True)
        except OSError:
            logger.debug("ceiling report dir unavailable", exc_info=True)

    def start(self) -> "CeilingRun":
        self._thread = threading.Thread(target=_sample,
                                        args=(self.cgdir, self.report, self._stop),
                                        name=f"lloyd-ceiling-{self.unit[-20:]}",
                                        daemon=True)
        self._thread.start()
        return self

    def settle(self, timeout: float = _APPEAR_WAIT_S) -> dict[str, str]:
        """Stop the sampler and return the counters, with a final live read merged in.

        Joining rather than sleeping, so "no verdict" is a fact and not a timing artefact
        of the reader. The merge is the part that decides whether the note exists at all:
        the sampler's last write is a snapshot up to one interval OLD, and the event that
        matters happens at the END of a run — a fan-out reaches `pids.max`, returns EAGAIN,
        and exits inside the ~44 ms measured, so the report on disk says `max=0` about a
        scope that had already hit its cap. Reading the scope again here is the state at
        the moment the child is gone, and it is cheap: one directory of small files.
        Peaks are kernel high-water marks, so the later read is never less true than the
        earlier one, and merging takes the larger of each rather than letting either win.
        If the directory is already collected, the report is all there is and the caller
        falls back to the exit status, which is why `finish` reads rc.

        Which of those two witnesses answered is returned as `source`, because #2924 turns
        this dict into a durable row and a peak is only as good as the read that produced
        it: `"scope"` means the directory was still there and this function read it,
        `"scope+report"` means both answered and `_higher_of` merged them, `"report"` means
        systemd had already collected the scope and every number came from the sampler's
        mirrored snapshot, and no key at all means neither arrived (a run that never armed
        reports `{}`, and `finish` records that as `"none"` rather than dropping the row).
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        report = _read_report_file(self.report)
        live = _read_counters(self.cgdir) or {}
        if not live:
            return {**report, "source": "report"} if report else {}
        if not report:
            return {**live, "source": "scope"}
        merged = dict(report)
        for key, value in live.items():
            merged[key] = _higher_of(key, report.get(key), value)
        merged["source"] = "scope+report"
        return merged


# ── the spawn-site seam ───────────────────────────────────────────────────────

def apply(argv: list[str]) -> tuple[list[str], "CeilingRun | None"]:
    """Wrap a built spawn argv in the ceiling. Returns `(argv, run)`.

    `(argv, None)` is the fail-open path: the argv is exactly what came in, the command
    still runs, and the caller must report it — `/state` says `fallback` and the log says
    UNBOUNDED. Never raise: this sits between the model and its command, and a ceiling
    that can throw has the reach of an outage.

    The wrapped argv keeps the caller's payload **byte for byte** after `systemd-run`'s own
    arguments: `systemd-run … -- <argv>` execs it exactly as `create_subprocess_exec` would
    have. No shell is inserted and nothing is appended, so a command containing an
    unbalanced quote is still passed as data, and the payload the sandboxes build —
    `bwrap … -- /bin/sh -c <command>`, or `/bin/bash -c` for a background task — is
    untouched. The scope the command lands in is the same subtree the sandbox put it in, one
    level up, which is why the two controls compose instead of competing: the read-only bind
    mounts still hold, and the ceiling now bounds what the process behind them may consume.

    The sampler starts before returning, so the first read can happen the moment the
    manager creates the scope. The unit name embeds the pid and a random token because two
    Bash calls in flight must not report on each other's counters: a name that could
    collide would attribute one command's OOM to another.
    """
    try:
        ok, why = available()
        if not ok:
            logger.warning("resource ceiling UNBOUNDED for a Bash call: %s", why)
            return argv, None
        run = CeilingRun(_safe_unit()).start()
        return scope_argv(run.unit, list(argv)), run
    except Exception:
        # Fail open, and say so: the ceiling protects availability, so it may not become
        # the reason a command did not run. `/state` reports the same state on the next
        # read, because a probe that throws never caches a positive verdict.
        logger.warning("resource ceiling wrap failed; the command runs unbounded",
                       exc_info=True)
        return argv, None


# ── reading the verdict ───────────────────────────────────────────────────────

def _int(text: str | None) -> int | None:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def events_counts(fields: dict[str, str]) -> dict[str, int]:
    """`memory_events` — the flattened `key value` lines of a cgroup's `memory.events` —
    as a mapping. `max` and `oom_kill` are the two that matter: `max` counts times the
    cgroup hit its own limit, `oom_kill` counts processes the kernel killed because of it."""
    return event_line_counts(fields.get("memory_events") or "")


def event_line_counts(text: str) -> dict[str, int]:
    """One flattened `key=value;…` events string as a mapping of ints."""
    out: dict[str, int] = {}
    for chunk in (text or "").split(";"):
        key, sep, value = chunk.partition("=")
        try:
            out[key] = int(value)
        except ValueError:
            pass
    return out


def _oom_kill_delta(before: dict[str, int], after: dict[str, int]) -> int | None:
    """How many OOM kills the slice recorded during one call.

    `None`, not 0, when either side could not be read: a missing baseline is "we do not
    know", and reporting a clean run on the strength of a file that would not open is the
    silent-fail-open this module's whole readout exists to avoid. The counter is
    cumulative and monotonic, so a negative delta means the slice was recreated under us
    (a manager restart) and is likewise unknown, not zero.
    """
    if not before or not after:
        return None
    b, a = before.get("oom_kill"), after.get("oom_kill")
    if b is None or a is None or a < b:
        return None
    return a - b


def tripped_limit(fields: dict[str, str]) -> dict | None:
    """Which limit the run hit, from the kernel's counters and from nothing else.

    The exit code is deliberately not evidence: an OOM kill arrives as rc=-9 (137 through
    a shell) and a fan-out that could not fork exits 1, 2 or 0 depending on its shell, and
    any of those can also be the command's own answer. `pids.peak` against `pids.max`, and
    `memory.peak` against `memory.max`, are in-kernel high-water marks that count the dead
    — which is exactly why they can be read after a reporter that could not survive.
    """
    peak_pids, max_pids = _int(fields.get("pids_peak")), _int(fields.get("pids_max"))
    # `pids.events`' own `max` line is the task cap's witness and is checked first: a
    # capped fan-out can be reaped before its `pids.peak` is mirrored into the report,
    # and reading only the peak turned the planted fork bomb into a non-zero exit with
    # a note of `None`, which is the silent-fail-open this module exists to avoid.
    if event_line_counts(fields.get("pids_events") or "").get("max", 0) > 0:
        return {"limit": "TasksMax", "value": fields.get("tasks_max") or TASKS_MAX,
                "observed": f"pids.events={fields.get('pids_events')} against "
                            f"pids.max={max_pids}",
                "detail": "every further fork in the subtree failed with EAGAIN"}
    if peak_pids and max_pids and peak_pids >= max_pids:
        return {"limit": "TasksMax", "value": TASKS_MAX,
                "observed": f"pids.peak={peak_pids} reached pids.max={max_pids}",
                "detail": "every further fork in the subtree failed with EAGAIN"}
    peak_mem, cap = _int(fields.get("mem_peak")), _int(fields.get("mem_max"))
    ev = events_counts(fields)
    delta = _int(fields.get("oom_kill_delta"))
    if cap and (peak_mem or 0) >= int(cap * 0.99):
        return {"limit": "MemoryMax", "value": MEMORY_MAX,
                "observed": f"memory.peak={peak_mem} against memory.max={cap}"
                            + (f", memory.events max={ev.get('max')}" if ev else ""),
                "detail": "the memory controller terminated the subtree; swap was "
                          "pinned to 0, so the box's zram could not absorb it"}
    # The scope's own `memory.events` — captured by the sampler while the directory was
    # alive, and merged into the report — is the ONLY witness that a kill happened inside
    # THIS call's subtree, and it survives more often than the peaks do: measured over five
    # planted 256 MiB allocators, `oom_kill=1` arrived in five of five while `memory.peak`
    # arrived in none of the three fast ones. Claiming the kill is this call's on the
    # strength of the slice's cumulative counter instead is a different statement and a
    # false one: `lloyd.slice` counts every concurrent subtree in it, so a delta can only
    # say a kill happened *somewhere* in the slice during the window. Until this split the
    # two evidences shared one branch and the branch spoke the subtree sentence, which on
    # this box blamed the gate's own ceiling suite (two real kills per automod round, at
    # 256 MiB and 16 GiB) on whatever unrelated Bash call's window straddled them — 88
    # occurrences across the Oct-13/14 session stores, one of which became incident #2725.
    # A `None` delta is unknown, not zero, and contributes nothing either way.
    if (ev.get("oom_kill") or 0) > 0:
        return {"limit": "MemoryMax", "value": MEMORY_MAX,
                "observed": f"memory.events oom_kill={ev['oom_kill']}"
                            + (f" (memory.peak={peak_mem} against memory.max={cap})"
                               if cap else ""),
                "detail": "the memory controller killed a process in the subtree; swap "
                          "was pinned to 0, so the box's zram could not absorb it"}
    if (delta or 0) > 0:
        return {"limit": "MemoryMax", "value": MEMORY_MAX,
                "observed": f"memory.oom_kill rose by {delta} in {CEILING_SLICE} during "
                            f"this call's window",
                "detail": f"{CEILING_SLICE} recorded {delta} memory "
                          f"kill{'s' if delta != 1 else ''} while this call ran, "
                          f"but this call's own counters record none, so no kill is "
                          f"attributed to it; the scope that died is collected with its "
                          f"counters, which is why only the slice saw it"}
    return None


def ceiling_note(fields: dict[str, str]) -> str | None:
    """The line appended to the tool result, or None when nothing tripped.

    The limit's *name* and its configured *value* both appear, always: `pids.peak=512`
    without `TasksMax=512` beside it is a number the reader has to go and look up, and
    the whole purpose of the line is that nobody has to."""
    trip = tripped_limit(fields)
    if not trip:
        return None
    return (f"[resource ceiling] {trip['limit']}={trip['value']} — {trip['observed']}; "
            f"{trip['detail']}")


def _int_or_none(value: object) -> "int | None":
    """A counter as an int, or None when it is absent or `max`.

    Not `_int`: a sample row must distinguish "this peak was never read" from "this peak
    was zero", and `_int` returns 0 for both, which would let a store full of unmeasured
    calls pull a p99 toward zero. `max` is unbounded and is not a number either.
    """
    try:
        text = str(value).strip()
        if not text or text == "max":
            return None
        return int(text)
    except (TypeError, ValueError):
        return None


def _mem_frac(mem_peak: "int | None", mem_max: "int | None") -> "float | None":
    """Peak over cap, the number the p99 re-derivation actually reads. `round`ed to four
    places so a row is 200 bytes and not a float spew; None when either side is unknown."""
    if not mem_peak or not mem_max:
        return None
    return round(mem_peak / mem_max, 4)


def _append_sample(fields: dict[str, str], rc: "int | None", unit: str) -> None:
    """Append one row to `SAMPLE_STORE`. Never raises.

    Never raises for the reason that governs everything in this module: this sits on the
    path that reports a command's own exit code, and a store that cannot be written (a full
    disk, a directory the retention sweep is mid-rewrite on) must cost a sample, not a Bash
    call. The failure is logged at debug because a store that has been failing for a week is
    exactly the thing `status()`'s `sample_rows` will show as a flat line, and the debug line
    is then the only place the reason exists.
    """
    mem_peak = _int_or_none(fields.get("mem_peak"))
    mem_max = _int_or_none(fields.get("mem_max"))
    row = {
        # UTC with the offset spelled out, never a naive local stamp: the retention sweep
        # compares this against a UTC cutoff, and a stamp that only *looks* local is read as
        # UTC by every later reader — an hour of drift on every row, in whichever direction
        # this box happens to sit.
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "rc": rc,
        "mem_peak": mem_peak,
        "mem_max": mem_max,
        "pids_peak": _int_or_none(fields.get("pids_peak")),
        "pids_max": _int_or_none(fields.get("pids_max")),
        "mem_frac": _mem_frac(mem_peak, mem_max),
        "oom_kill_delta": int(fields.get("oom_kill_delta") or 0),
        # Which witness measured the peak. "none" is the no-witness case: settle() returns
        # no key at all when both the scope directory and the mirrored report came back
        # empty, and a row with `source: "none"` is a call whose consumption is UNKNOWN —
        # not a call that used zero. A p99 that does not filter on this is a lie.
        "source": fields.get("source") or "none",
        "unit": unit,
    }
    try:
        SAMPLE_STORE.parent.mkdir(parents=True, exist_ok=True)
        _prune_samples()
        with open(SAMPLE_STORE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        logger.debug("ceiling sample row could not be written", exc_info=True)


def _prune_samples(now: "float | None" = None) -> int:
    """Drop rows past `SAMPLE_MAX_AGE_DAYS`, at most once a day. Returns how many went.

    This bound is what licenses one row per Bash call at ~250 bytes each: the store prunes itself
    from its own writer, so it cannot silently become the twentieth unbounded file under
    `~/lloyd-data` if nothing else ever reaches it. Rate-limited by the file's own mtime rather
    than run per call — 429 Bash calls across the 25 newest session stores would otherwise mean
    a full read-modify-rewrite of the store hundreds of times a day to retire rows that age out
    once a day.

    Age is each row's own `ts`, never the file's mtime, which is always now on a store that is
    appended forever: a prune keyed on mtime deletes nothing while reporting a clean sweep, the
    failure `scripts/groundskeeper/retention-sweep.py` names for its turn-row store. Two kinds of
    row are always kept — one whose `ts` will not parse (#2357: unreadable is not the same thing
    as old), and the newest row in the file, so a box idle for a quarter still holds a witness of
    its last call rather than an empty store that reads as "never measured". Rewrite is
    rename-atomic, because `_sample_rows` and the owed p99 pass read this file while a Bash call
    is appending to it.
    """
    try:
        stamp = SAMPLE_STORE.stat().st_mtime
    except OSError:
        return 0
    moment = time.time() if now is None else now
    if moment - stamp <= 86400:
        return 0
    rows = _sample_rows()
    if len(rows) < 2:
        return 0
    horizon = moment - SAMPLE_MAX_AGE_DAYS * 86400
    newest = max((str(r.get("ts", "")) for r in rows), default="")
    kept: list[dict] = []
    for r in rows:
        raw = r.get("ts")
        when = None
        if isinstance(raw, str) and raw:
            try:
                when = datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S").replace(
                    tzinfo=timezone.utc).timestamp()
            except ValueError:
                when = None
        if when is None or when > horizon or str(raw) == newest:
            kept.append(r)
    dropped = len(rows) - len(kept)
    if dropped <= 0:
        os.utime(SAMPLE_STORE, (moment, moment))   # nothing to retire; do not re-check today
        return 0
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in kept)
    tmp = SAMPLE_STORE.with_suffix(SAMPLE_STORE.suffix + ".pruning")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, SAMPLE_STORE)
    return dropped


def _sample_rows(path: "Path | None" = None) -> list[dict]:
    """Every parsable row in `SAMPLE_STORE`, oldest first; unreadable tail line skipped.

    The sweep rewrites this file in place, so a read that lands mid-rewrite can see a
    truncated last line. Skipping it is right here and reporting it is the sweep's job:
    `status()` uses this to answer "how deep is the sample store", and a half line is not a
    sample.
    """
    target = SAMPLE_STORE if path is None else Path(path)
    rows: list[dict] = []
    try:
        raw = target.read_bytes()
    except OSError:
        return rows
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def finish(run: "CeilingRun | None", rc: "int | None" = None) -> str | None:
    """Read the verdict for a finished call, append its durable row, return the line, if any.

    Called by `_bash` where it already appends `[exit code: N]`, so rc=137 and rc=-9
    arrive with a meaning attached: without this line, 137 is a number, and a future
    session spends a turn rediscovering that it is 128+9. Never raises for a reason that
    matters — a ceiling that cannot explain itself must not stop a command reporting its
    own exit code.

    Two things happen here beyond the note, both about what this leaves behind.

    **One row per call lands in `SAMPLE_STORE`, whatever else happened** — tripped or not,
    counters or none. The number #2578 owes after landing is a pair (commands the ceiling
    killed, and benign commands that came within a fraction of the cap), so a record kept
    only of kills answers only half of it, and a call that yielded no counters still gets a
    row with `source: "none"` rather than being skipped, because a store that silently drops
    its own failures cannot be asked how deep it is. `rc` is what lets a peak be read as a
    fact at all: 4 GiB that exited 0 and 4 GiB that exited 137 are different measurements.
    A backgrounded spawn passes `rc=None` — its process is still running at this point and
    its exit code belongs to the task registry — which is honest as "no exit code existed",
    and `unit` is what joins the row to that task's own log.

    **Then the per-unit report is unlinked**, so `REPORT_DIR` stays a handover between the
    sampler and this function instead of becoming a second per-call store. The reason this
    used to give for the unlink — that "the same line in the already-rotated MCP log carries
    the identical numbers" — was false on this box: `grep -c "resource ceiling sample"` is 0
    in `logs/mcp.log`, in its ten rotations and in `logs/lloyd-agent-worker.log`, because
    this logger reaches no file at all (the file is uvicorn access lines). What survived a
    call was therefore only the `.report` of a spawn that never got here — 100 unread files
    across 2026-10-12→17 — which is why the JSONL row, not the log line, is the record now.
    The line below stays for whoever is tailing.
    """
    fields = run.settle() if run is not None else {}
    if run is not None:
        fields = dict(fields)
        fields["oom_kill_delta"] = _oom_kill_delta(run.baseline, _slice_events())
    _append_sample(fields, rc, run.unit if run is not None else "")
    if fields:
        peak_mem, cap = _int(fields.get("mem_peak")), _int(fields.get("mem_max"))
        logger.info(
            "resource ceiling sample: mem_peak=%s mem_max=%s pids_peak=%s pids_max=%s "
            "mem_frac=%s oom_kill_delta=%s",
            peak_mem, cap, fields.get("pids_peak"), fields.get("pids_max"),
            (f"{peak_mem / cap:.3f}" if peak_mem and cap else "?"),
            fields.get("oom_kill_delta"))
    note = ceiling_note(fields)
    if run is not None:
        try:
            run.report.unlink(missing_ok=True)
        except OSError:
            logger.debug("ceiling report could not be removed", exc_info=True)
    if note:
        _verdict.set(note)
        logger.info("resource ceiling tripped: %s", note)
    return note



def take_verdict() -> str | None:
    """This call's ceiling line, cleared. For a caller that cannot use `finish`'s return
    value directly (a background spawn has no result to append to)."""
    got = _verdict.get()
    if got is not None:
        _verdict.set(None)
    return got


def status() -> dict:
    """The `/state` shape, in the same vocabulary as `_path_sandbox.status()` and
    `egress.status()`: `enforcing`/`fallback` are the #2109 names, and the configured
    values travel with the verdict, because "armed" without the numbers is one more thing
    to go and look up.

    A box the ceiling cannot arm reports `fallback: true` with the reason. It does not
    report a bound that is not there.
    """
    rows = _sample_rows()
    stamps = sorted({str(r.get("ts", ""))[:10] for r in rows if r.get("ts")})
    # Whole days the store covers, inclusive of both ends, and None when there is nothing to
    # span. #2578's owed entry asks "do we have >=7 days of real traffic yet", and the point
    # of putting the answer here is that the pass which has to rule on it can read /state
    # instead of opening the file: `sample_days` < 7 is that question, and `sample_rows`
    # tells whether the window is thin or merely short.
    days = None
    if len(stamps) >= 2:
        try:
            first = time.strptime(stamps[0], "%Y-%m-%d")
            last = time.strptime(stamps[-1], "%Y-%m-%d")
            days = (int(time.mktime(last)) - int(time.mktime(first))) // 86400 + 1
        except ValueError:
            days = None
    ok, err = available()
    return {
        "enforcing": ok,
        "fallback": not ok,
        "error": err or None,
        "systemd_run": bool(shutil.which("systemd-run")),
        "slice": CEILING_SLICE,
        "MemoryMax": MEMORY_MAX,
        "memory_bytes": memory_bytes(),
        "MemorySwapMax": MEMORY_SWAP_MAX,
        "TasksMax": TASKS_MAX,
        "measured_peak_ram_bytes": MEASURED_PEAK_RAM_BYTES,
        "measured_peak_pids": MEASURED_PEAK_PIDS,
        "headroom_factor": HEADROOM_FACTOR,
        "report_dir": str(REPORT_DIR),
        # #2924: how deep the durable sample store is. `measured_peak_*` above is the single
        # 2026-10-12 observation the constants were derived from; these are how many real
        # calls now stand behind that kind of number, and `sample_days` is the >=7-days
        # question #2578 owes. `sample_unmeasured` is the rows carrying `source: "none"` —
        # rows a p99 must not silently average in.
        "sample_store": str(SAMPLE_STORE),
        "sample_rows": len(rows),
        "sample_days": days,
        "sample_dates": stamps,
        "sample_unmeasured": sum(1 for r in rows if r.get("source") == "none"),
    }


def reset_probe_cache() -> None:
    """Forget the probe verdict — for a test that moved `XDG_RUNTIME_DIR`, or a re-check
    after a fix. The affordance `_path_sandbox.reset_probe_cache()` provides."""
    with _probe_lock:
        _probe.update(ok=None, at=0.0, error="")
