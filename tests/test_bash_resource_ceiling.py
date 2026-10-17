"""#2578 — every Bash-spawned process subtree gets an OS resource ceiling.

The item's premise is that Lloyd bounded where a command may write (the bwrap profile in
`agent_mcp/_path_sandbox.py`) and how long it may run (120 s default, 600 s cap, SIGTERM →
SIGKILL on the process group), but never what it may consume — so the stop-loss for a
command that forks or allocates was response-shaped: a timeout that only bites while the
foreground command is still attached, and skills a human invokes afterwards. The ceiling is
preventive instead: `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0 -p
TasksMax=…`, created inside `lloyd.slice`, wrapping the argv the sandboxes already built.

Boundaries this file crosses, and why each node forks. `builtin_bash` is an MCP tool
handler and the ceiling is a property of the OS, so the only honest assertion is what the
kernel did to a real process: a planted allocator killed by the memory controller, a planted
fan-out whose `fork()` starts returning EAGAIN, `/proc/self/cgroup` read by a child to prove
which slice it landed in. A test that only parsed `apply()`'s argv would pass on an argv
that does not work, and a node that measured the ceiling's bite on `setrlimit` semantics —
RLIMIT_AS is per-process, and `RLIMIT_NPROC` counts the whole UID, which on a box where one
user owns the engines is a footgun aimed at the stack, not a bound on a child — would prove
nothing.

Which is also why no node in this file allocates the shipped 16G (`MEMORY_MAX` in
`agent_mcp/_resource_ceiling.py`): filling 16 GiB on a box with the engine daemons resident
would put the agent stack down to prove a bound exists. The cap every planted memory bomb
runs at is `PLANTED_MEMORY_MAX` = 256M — the byte-bound allocator, the slow sibling in the
misattribution node, and the allocator whose kill has to surface in a *tool result*
(`test_the_named_limit_reaches_the_tool_result_and_not_only_the_module`, which ran against
the shipped 16G until #2925: 242 launches of it since 2026-10-13, with scope teardown peaks
up to 15.7G and ~100 s of wall clock each) — and the planted fan-out runs at
`RC.TASKS_MAX = "48"`. Each allocator node then proves its own cap from inside the scope it
was planted in (its `planted cap-path=… cap=…` line, read from that scope's `memory.max`),
so an edit that drops a
patch fails a node instead of quietly arming 16G again. That is a substitute of *value*, not
of *mechanism* — same unit, same cgroup files, same kernel controller — and the mechanism
itself is proved again on the real constants, which no node patches: memory by the byte-exact
`memory.max` read-back in the probe, tasks by the benign 16-way fan-out that `TasksMax=1024`
must let through untouched.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_mcp import _path_sandbox as PS          # noqa: E402
from agent_mcp import _resource_ceiling as RC      # noqa: E402
from agent_mcp import builtin_bash as B            # noqa: E402

RC_SRC = (ROOT / "agent_mcp" / "_resource_ceiling.py").read_text(encoding="utf-8")
VENVS = ROOT / ".venvs" / "lloyd" / "bin"
REPO_PY = str(VENVS / "python") if VENVS.is_dir() else sys.executable

# The ceiling's own nodes must be able to say "and it was not slow", because a bound that
# holds legitimate work at the wall clock is a worse defect than the one being fixed.
# Generous against each node's real cost; what is pinned is that the ceiling puts no
# ceiling on time.
CEIL_BUDGET_S = float(os.environ.get("LLOYD_CEILING_TEST_BUDGET_S", "120"))
DETACHED_SECONDS = 6


def _run_bash(command, **kw):
    """One real foreground tool call, through the same coroutine the MCP handler runs.

    `{"command": …}` is the shape the handler's own tests use: `_bash` takes the tool's
    argument dict, so a helper that passed a bare string would fail inside the handler
    before the ceiling was ever reached.
    """
    return asyncio.run(B._bash({"command": command, **kw}))


def _spawn_bg_and_wait(command, budget_s=40.0, **kw):
    """Spawn through `run_in_background` and wait for its record to leave `running`.

    Spawn and poll must share one event loop: `_spawn_background` arms its exit waiter with
    `asyncio.create_task`, so a helper that called `asyncio.run` per step would close the
    loop the waiter lives on and the record would sit `running` forever. Polling from a
    second loop therefore reports a task that never finishes, which is a bug in the test,
    not a ceiling that holds a task — and the distinction is the whole subject of clause 5.
    """
    async def go():
        res = json.loads(await B._spawn_background(command, "test Bash task", **kw))
        task_id = res["task_id"]
        deadline = asyncio.get_running_loop().time() + budget_s
        while asyncio.get_running_loop().time() < deadline:
            rec = B._task_registry.get(task_id)
            assert rec is not None, f"{task_id} vanished from the registry"
            if rec.status != "running":
                return task_id, rec
            await asyncio.sleep(0.05)
        rec = B._task_registry.get(task_id)
        raise AssertionError(f"{task_id} still running after {budget_s}s — either the "
                             "ceiling held it or the waiter never ran")
    return asyncio.run(go())


# The two planted shapes, as the item states them. Both are written to die at a *cgroup*
# limit rather than to look like one: a shell `( … & )` loop returns EAGAIN on the shell's
# own terms, which is why the fan-out forks from a real process and reaps, so the only thing
# stopping it is the kernel.
_FAN_OUT = (
    "import os, sys\n"
    "kids = []\n"
    "for i in range(400):\n"
    "    try:\n"
    "        pid = os.fork()\n"
    "    except OSError as exc:\n"
    "        print(f'blocked-at={i} errno={exc.errno}', flush=True)\n"
    "        break\n"
    "    if pid == 0:\n"
    "        os._exit(0)\n"
    "    kids.append(pid)\n"
    "for pid in kids:\n"
    "    try:\n"
    "        os.waitpid(pid, 0)\n"
    "    except ChildProcessError:\n"
    "        pass\n"
    "print('loop-ended', flush=True)\n"
)
# The cap every planted memory bomb in this file runs at, and the most a planted cap is
# allowed to be. #2925 found the tool-result node allocating against the shipped 16G — so
# one suite run grew a real ~15 GiB buffer per launch inside `lloyd.slice` and spent ~100 s
# of wall clock racing the Bash tool's own 120 s timeout to get its kill in first. 256M is
# what the byte-bound allocator already cost: the same silent SIGKILL in a second.
PLANTED_MEMORY_MAX = "256M"
PLANTED_MEMORY_MAX_BYTES = 256 * 1024 * 1024
# 512 MiB, from the item. The nodes below assert their scope reads back at or under this, so
# the invariant is `<= 512 MiB` and not `== 256M`: raising `PLANTED_MEMORY_MAX` past the
# bound, or dropping a patch so the scope arms at the shipped 16G, fails a node.
PLANTED_CAP_LIMIT_BYTES = 512 * 1024 * 1024
# Bytes the allocator is expected to die inside: the test's own 256M cap plus one 8 MiB
# block plus the interpreter's own footprint. Stated because the item's acceptance is a
# *byte* bound, and because 8 GiB of churn before a kill would be a different failure.
_ALLOCATOR_BYTE_BOUND = PLANTED_MEMORY_MAX_BYTES + 64 * 1024 * 1024

#: The first line of both allocator payloads: the cap, and the path it was read from, taken
#: from the scope the payload is already running in — `/proc/self/cgroup` names the
#: `lloyd-bash-ceiling-*.scope` (measured through `B._bash` on this box, where the child's own
#: view does name it) and that scope's `memory.max` is the value `apply` asked the manager
#: for. Both figures are printed, not just the number, because the proof has to be the planted
#: command's own and nothing else's: `planted` is a marker no harness line prints, and the
#: path is what distinguishes a read of the per-call scope from a read of
#: `/sys/fs/cgroup/memory.max`, which also exists and is an entirely different cgroup. Printed
#: *before* allocating because the process that read it is SIGKILLed with no exit status of its
#: own and a fast scope's `memory.peak` is collected with its directory — a witness that lives
#: only in the note goes blind exactly when the bomb grows largest. And the payloads exit 42
#: rather than allocate past `PLANTED_CAP_LIMIT_BYTES`: an assertion that reads the run's output
#: can only fire after the kill, so the guard against a dropped patch has to live in the payload,
#: which is what `test_a_planted_allocator_refuses_to_grow_against_an_oversized_scope` proves.
_MEMCAP_READ_BACK_PY = (
    "import pathlib, sys\n"
    "rel = pathlib.Path('/proc/self/cgroup').read_text().split(':', 2)[2].strip()\n"
    "p = '/sys/fs/cgroup' + rel + '/memory.max'\n"
    "cap = int(pathlib.Path(p).read_text().strip())\n"
    "print('planted cap-path=' + p + ' cap=' + str(cap), flush=True)\n"
    f"sys.exit(42) if cap > {PLANTED_CAP_LIMIT_BYTES} else None\n"
)
_MEMCAP_READ_BACK_SHELL = (
    "cap=$(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max); "
    "echo planted cap-path=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max cap=$cap; "
    f'[ "$cap" -le {PLANTED_CAP_LIMIT_BYTES} ] || exit 42; ')

# `-u` is load-bearing for the byte bound below: stdout is a pipe here, so without
# unbuffered output the progress lines die in the child's own 8 KiB buffer when it is
# SIGKILLed and the bound becomes unmeasurable.
_ALLOCATOR = (
    _MEMCAP_READ_BACK_PY +
    "blocks = []\n"
    "while True:\n"
    "    blocks.append(bytearray(8 * 1024 * 1024))\n"
    "    print(f'bytes={len(blocks) * 8 * 1024 * 1024}', flush=True)\n"
)


def _planted_cap_bytes(shown: str) -> int:
    """The cap a planted allocator read back from its own scope, bounded (#2925).

    `shown` is whatever the node has in hand — the child's stdout, or the whole tool result.
    Four assertions, in the order the failure modes arrive. `planted` is printed by the two
    allocator payloads and by nothing else in the run, so requiring exactly one such line is
    what makes the figure the planted command's own rather than a number the harness happened
    to emit: zero means the read-back is gone, two means something else started printing it.
    The path has to name a per-call ceiling scope, because `/sys/fs/cgroup/memory.max` also
    reads cleanly and belongs to no call at all. Then the bound — the shipped `MEMORY_MAX` of
    16G is 17179869184 bytes, which is what the third bites on — and finally the planted value,
    so a patch and a payload that disagree about which run this is cannot pass either.
    """
    found = re.findall(r"planted cap-path=(\S+) cap=(\d+)", shown)
    assert len(found) == 1, \
        f"expected exactly one `planted cap-path=… cap=…` line, got {len(found)}: the marker is " \
        f"printed by the planted allocator alone, so none means the in-scope read-back was " \
        f"dropped and more than one means the figure is no longer attributable to it"
    path, raw = found[0]
    value = int(raw)
    assert "lloyd-bash-ceiling-" in path, \
        f"the payload read its cap from {path}, which is not a per-call ceiling scope — the cap " \
        f"that bites a planted command is its own scope's, named in /proc/self/cgroup"
    assert value <= PLANTED_CAP_LIMIT_BYTES, \
        f"the planted payload ran in a scope whose memory.max was {value} bytes, over " \
        f"the {PLANTED_CAP_LIMIT_BYTES}-byte bound this suite is allowed to plant under"
    assert value == PLANTED_MEMORY_MAX_BYTES, \
        f"the scope the payload ran in carried {value} bytes, not the " \
        f"{PLANTED_MEMORY_MAX_BYTES} ({PLANTED_MEMORY_MAX}) the node planted"
    return value


@pytest.fixture(autouse=True)
def _fresh_probe():
    """Every node probes availability itself.

    The probe's verdict is cached module-wide for a minute so a hot Bash path does not
    shell out per call; a test that inherited another node's cached verdict would be
    asserting a state it never established, which is exactly how a fail-open ceiling gets
    a green test.
    """
    RC.reset_probe_cache()
    yield
    RC.reset_probe_cache()


@pytest.fixture(autouse=True)
def _restore_constants():
    """Nodes shrink the cap so a planted command dies in a second instead of at the shipped
    `MEMORY_MAX` of 16G; the shipped values are restored after each one, because they are the
    ceiling. The restore is what makes `RC.MEMORY_MAX = PLANTED_MEMORY_MAX` safe to write in a
    node body at all — and until #2925 one node never patched the constant in the first place,
    which no restore could notice: that is the gap the payload's in-scope cap read-back is for."""
    before = (RC.MEMORY_MAX, RC.MEMORY_SWAP_MAX, RC.TASKS_MAX)
    yield
    RC.MEMORY_MAX, RC.MEMORY_SWAP_MAX, RC.TASKS_MAX = before


def _big_tree() -> str:
    """A tree big enough that a recursive grep and find do real work: the repo's own
    checkout, which is the largest recursive search any Bash call on this box performs, and
    the one a real session runs most often."""
    # NOT the repo checkout. The first version searched the checkout and the node
    # spent its whole 120 s budget with no ceiling note at all: over a working tree
    # that the running system is writing, and from the bench profile's bwrap, a
    # repo-wide recursive `find` ran past the tool timeout (0.3 s in the foreground
    # arm). A control slower than its budget proves nothing in either direction. This
    # is the shape the item names — recursive `grep -rl` and `find` doing real
    # traversal — at a size that measures rather than outlives: 200 dirs x 20 files,
    # 0.1 s in both arms with the ceiling armed (measured), and the grep token matches
    # every file so nothing short-circuits the walk.
    planted = Path("/tmp") / "lloyd-ceiling-planted-tree" / "ceiling_find"
    if not (planted / "d0").is_dir():
        planted.mkdir(parents=True, exist_ok=True)
        for d in range(200):
            deep = planted / f"d{d}"
            deep.mkdir(exist_ok=True)
            for i in range(20):
                (deep / f"f{i}.txt").write_text(f"row {i} ceiling_find {i * 3}\n",
                                                encoding="utf-8")
    assert sum(1 for _ in planted.rglob("*.txt")) == 4000, planted
    return str(planted)


def _wrap_python(code, memory=None, swap=None, tasks=None):
    """`apply`'s argv and run handle for a planted Python payload, constants narrowed.

    The payload goes to the interpreter as `-c` data rather than through a shell, so what
    `apply` wraps is one argv and not a shell string with quoting to argue about.
    """
    if memory is not None:
        RC.MEMORY_MAX = memory
    if swap is not None:
        RC.MEMORY_SWAP_MAX = swap
    if tasks is not None:
        RC.TASKS_MAX = tasks
    argv, run = RC.apply([REPO_PY, "-u", "-c", code])
    assert argv and argv[0] == "systemd-run", argv[:1]
    assert run is not None, "the ceiling reported itself unavailable on this box"
    return argv, run


def _wrap(command, memory=None, swap=None, tasks=None):
    """`apply`'s argv and run handle with the constants narrowed for a planted command.

    Narrowed *before* `apply` runs, so the ceiling the unit asks the manager for is the
    number the result is then expected to name — no post-hoc `pids.max` write, nothing the
    spawn site would not itself have asked for. Values the tests invent and the box is
    allowed to have: the 256 MiB and 48-task caps the kill nodes plant against.
    """
    if memory is not None:
        RC.MEMORY_MAX = memory
    if swap is not None:
        RC.MEMORY_SWAP_MAX = swap
    if tasks is not None:
        RC.TASKS_MAX = tasks
    argv, run = RC.apply(["/bin/sh", "-c", command])
    assert argv and argv[0] == "systemd-run", argv[:1]
    assert run is not None, "the ceiling reported itself unavailable on this box"
    return argv, run


# ── Clause 1 — where the ceiling is applied, and in which slice ───────────────

def test_exactly_the_three_spawn_sites_apply_the_ceiling():
    """Clause 1: the ceiling is applied at the three `create_subprocess_exec` sites, its
    three values are named constants in one module, and nothing outside those sites applies
    it.

    Pinned statically because the failure this guards is a *missing* call, and a runtime
    test of "the other spawn path has no ceiling" can only be observed by a runaway that has
    already taken the box. The three sites are the ones the item names: the bench/eval arm,
    the foreground spawn and `_spawn_background`. The negative half is the item's scope
    rule — `scripts/automod/evalpin.py` launches vLLM and `agent-services/systemd/**` is
    where the units live; a ceiling reaching either would throttle an engine boot, which is a
    worse defect than the one being fixed, since engine boots legitimately want ~226 GiB.
    """
    text = (ROOT / "agent_mcp" / "builtin_bash.py").read_text(encoding="utf-8")
    spawns = [m.start() for m in re.finditer(r"create_subprocess_exec\(", text)]
    assert len(spawns) == 3, \
        f"builtin_bash has {len(spawns)} spawn sites, not the three this ceiling is scoped " \
        "to; a fourth would be a Bash path with no bound and no test"
    applied = [m.start() for m in re.finditer(r"_resource_ceiling\.apply", text)]
    assert len(applied) == 3, f"{len(applied)} apply() calls for {len(spawns)} spawns"
    for spawn, want in zip(spawns, applied):
        # The apply must sit in the 700 characters before its own spawn: an apply()
        # anywhere in the file would satisfy a count-only assertion while wrapping nothing.
        assert spawn - 700 < want < spawn, \
            "an apply() is not adjacent to the spawn it is supposed to bound"

    for rel in ("scripts/automod/evalpin.py", "agent-services/systemd"):
        hits = subprocess.run(
            ["git", "-C", str(ROOT), "grep", "-l", "resource_ceiling", "--", rel],
            capture_output=True, text=True)
        assert not hits.stdout.strip(), \
            f"the ceiling reached {rel}, which the item keeps out of scope: {hits.stdout}"

    # The unit is named once, in the module: a per-site literal would be a value nobody
    # has to change in step with the measurement.
    unit_tokens = re.findall(r"systemd-run", text)
    assert not unit_tokens, \
        "builtin_bash builds the scope argv itself instead of calling into the module"
    for name in ("MEMORY_MAX", "MEMORY_SWAP_MAX", "TASKS_MAX", "CEILING_SLICE"):
        assert re.search(rf"^{name} = ", RC_SRC, re.M), \
            f"{name} is not a module-level constant"
    for value in (RC.MEMORY_MAX, RC.MEMORY_SWAP_MAX, RC.TASKS_MAX):
        assert re.fullmatch(r"[0-9]+[KMG]?|[0-9]+", str(value)), \
            f"{value!r} is not a plain systemd value a test can compare"
    assert RC.CEILING_SLICE == "lloyd.slice", RC.CEILING_SLICE


def test_the_slice_is_lloyd_slice_at_every_spawn_site_and_never_app_slice():
    """Clause 1's second half: every Bash-spawned subtree is created inside `lloyd.slice`,
    never `app.slice`, proved from the kernel's own accounting at each of the three sites.

    This is what the static node cannot say. `--slice=` is a string; what matters is where
    the kernel put the subtree, and the answer is not what the string implies if the user
    manager's shape changes — `app.slice` is one manager default away, and it is the slice
    systemd-oomd monitors at 50%/20 s, whose pressure took `agent-supervisord` down five
    times (#1337's reason for the relocation). A ceiling created there would be a ceiling
    standing in the room the relocation emptied.

    Each spawn site builds its argv its own way (`_tool_sandbox.bwrap_argv` for a bench or
    eval arm, `_path_sandbox.wrap` for a foreground call and for `run_in_background`), so
    each is asked through that site's own construction. Two of the three arms let the child
    name its own cgroup and are read that way; the bench/eval sandbox unshares the cgroup
    namespace, so its child can only report `0::/` and is read through the scope's counters
    instead — the 256 MiB the child allocated has to show up as this scope's peak, which a
    child in some other cgroup could not do. Every site is held to the counter witness.
    """
    for site, build in (
            ("bench/eval", lambda c: B._tool_sandbox.bwrap_argv(c, "/tmp")),
            ("foreground", lambda c: PS.profile_argv(c, "/tmp")),
            ("background", lambda c: PS.profile_argv(c, "/tmp", background=True)),
    ):
        if not PS.available()[0] or not B._tool_sandbox.sandbox_available()[0]:
            pytest.skip(f"bwrap unavailable: {PS.available()[1]}")
        argv, run = RC.apply(build(f"{_CGROUP_PROBE}; {_ALLOCATE_256_MIB}"))
        assert RC.slice_of(argv) == RC.CEILING_SLICE, \
            f"the {site} site's argv does not name {RC.CEILING_SLICE}: {argv[:6]}"
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=CEIL_BUDGET_S, env=RC.launcher_env(None))
        assert proc.returncode == 0, f"{site} probe failed: {proc.stderr[-300:]}"
        paths = re.findall(r"(?:self|parent)=([^\s]+)", proc.stdout)
        if all(path == "/" for path in paths):
            # This arm's sandbox unshares the cgroup namespace, so the child's
            # `/proc/self/cgroup` reads `0::/` and CANNOT name the slice — measured: the
            # same inner argv with no scope at all reports `0::/` too. The blindness is
            # the arm's own, pre-dates the ceiling, and is not a hole in it; the
            # counters below are the instrument that arm has, so they are asserted for
            # every site regardless. What is asserted here is that the blindness is the
            # sandbox's and not something the wrapper introduced.
            bare = subprocess.run(build(f"{_CGROUP_PROBE}; {_ALLOCATE_256_MIB}"),
                                  env=RC.launcher_env(None),
                                  capture_output=True, text=True, timeout=CEIL_BUDGET_S)
            assert re.findall(r"(?:self|parent)=([^\s]+)", bare.stdout) == paths, (
                f"the {site} arm's child sees {paths} inside the ceiling and something "
                f"else outside it — the wrapper, not the sandbox, moved the view")
        else:
            assert any(RC.CEILING_SLICE in path for path in paths), \
                f"the {site} site's child reported {paths}: no cgroup on the path names " \
                f"{RC.CEILING_SLICE}, so the ceiling is not where the command is"
            assert not any("app.slice" in path for path in paths), \
                f"the {site} site's child is under app.slice — the monitored cgroup: {paths}"
        fields = run.settle()
        RC.finish(run)          # after settle: finish unlinks the report it reads from
        assert fields.get("scope") == run.cgdir, \
            f"the {site} site's counters came from {fields.get('scope')!r}, not the scope " \
            f"{run.cgdir} that `apply` named in the argv — the sampler is watching a " \
            "different subtree from the one the command runs in"
        peak = int(fields.get("mem_peak") or 0)
        if peak:
            assert peak >= 200 * 1024 * 1024, \
                f"the {site} site's child allocated 256 MiB but {run.cgdir} recorded a " \
                f"peak of {peak}: the sampler was watching a cgroup the command was not " \
                "in, so every verdict it produces is about the wrong subtree"
        else:
            # No counters at all is a real answer, and the honest one for a scope that
            # systemd collected before the sampler's last read: a transient scope is
            # gone ~20 ms after its last process is, which is the same race that makes
            # `finish` fall back to the exit status. What is NOT allowed here is a
            # reading of some other cgroup, and the assert above already ruled that out.
            # The allocator node pins the peak witness on a run that stays alive.
            assert fields.get("memory_events") is not None or not fields, \
                f"the {site} site read events but no peak: {fields}"


def test_the_aggregator_publishes_the_ceiling_on_the_state_endpoint():
    """The readout is reachable where the other two are, or it is a function nobody reads.

    `/state` is how the harness and `scripts/service_health_check.py` learn whether the
    substrate below a Bash call is actually there; the item's requirement is that "ceiling
    armed" and "ceiling unavailable on this box" be distinguishable there rather than in a
    log line nobody is looking at.
    """
    src = (ROOT / "agent_mcp" / "main.py").read_text(encoding="utf-8")
    assert '"bash_resource_ceiling": await asyncio.to_thread(_resource_ceiling.status)' in src, \
        "the ceiling is not published on /state"
    assert '"protected_path_sandbox": await asyncio.to_thread(_path_sandbox.status)' in src, \
        "the readout it is supposed to match moved; the shared shape is the contract"


# ── Clause 5 — benign work still works, at full scale ─────────────────────────

def test_three_benign_high_fan_out_and_high_memory_controls_complete():
    """Clause 5: `xargs -P 16`, a recursive search over a large tree, and a real background
    task all complete with the ceiling armed, inside the node's own wall-clock budget.

    These are the controls that keep the number honest — the item's stated dominant risk is a
    ceiling 4 GB too low turning a legitimate build into a mystery `MemoryError` a later
    session burns a turn diagnosing. Each asserts exit 0 *and* elapsed time *and* that no
    ceiling note was produced, because a control that completed only after being throttled
    would still print 0. The 16-way fan-out also has to clear the shipped `TASKS_MAX` of 1024
    with room to spare, which is the same limit the fan-out node above shows working at 48.
    """
    tree = _big_tree()
    for label, cmd in (  # noqa: B007
            ("xargs -P 16", "seq 400 | xargs -P 16 -n 4 sh -c 'printf \"%s\\n\" \"$0\"' "
                            "| wc -l"),
            ("recursive grep/find", f"find {tree} -name '*.txt' | wc -l; "
                           f"grep -rl 'ceiling_find' {tree} | wc -l"),
    ):
        for sandbox in (True, False):        # both arms clause 1 covers
            arm = "sandbox" if sandbox else "bench"
            if sandbox:
                os.environ.pop("LLOYD_BENCH_SANDBOX", None)
            else:
                os.environ["LLOYD_BENCH_SANDBOX"] = "1"
            t0 = time.monotonic()
            out = _run_bash(cmd, sandbox=sandbox)
            elapsed = time.monotonic() - t0
            assert "resource ceiling" not in out, f"{arm} {label}: {out[-300:]}"
            assert "exit code" not in out, f"{arm} {label}: {out[-300:]}"
            # Per-control AND per-arm: the budget is the item's "not held" claim
            # applied to each control on its own, so a slow control is named rather
            # than averaged away by a fast one.
            assert elapsed < CEIL_BUDGET_S, \
                f"{arm} {label} took {elapsed:.1f}s under the ceiling"

    _task_id, rec = _spawn_bg_and_wait(
        "seq 200 | xargs -P 16 -n 2 sh -c 'true'; echo bg-done", cwd="/tmp")
    assert rec.exit_code == 0, \
        f"background task exited {rec.exit_code}: {Path(rec.output_path).read_text()[-300:]}"
    body = Path(rec.output_path).read_text()
    assert "bg-done" in body and "resource ceiling" not in body, body[-300:]


def test_the_full_suite_passes_under_the_shipped_ceiling():
    """The control that decides whether this change is shippable: the whole suite, in one
    scope, under the values the module ships.

    The item's measured note is that a full run needs >8 GiB with the engine daemons
    resident, which is why 16G is the order the constant sits at (`MEMORY_MAX` is 16G, so
    the suite runs with under 8 GiB of headroom, not 24); a ceiling tuned to an
    ordinary command's p99 would stop this node, and stopping it is the self-inflicted
    failure the item warns about. It is also the distribution's own data point: the peak the
    sampler records for this run is what the owed p99 re-derivation is checked against, so it
    is written to a file as well as asserted — a number only this node saw is a number the
    next ruling cannot re-check.

    Skipped unless `LLOYD_CEILING_FULL_SUITE=1`, on the rule that a single node may not run
    the suite it is part of: the gate's `tests` rung runs this file inside the suite, and a
    self-recursive suite would be an unbounded regression. The gate rung itself is the
    ceiling-armed full-suite pass, and the standalone run that measured the peak was
    `systemd-run --user --scope --slice=lloyd.slice … pytest tests/`.
    """
    if os.environ.get("LLOYD_CEILING_FULL_SUITE") != "1":
        pytest.skip("the gate's tests rung is the suite under the ceiling; this node is "
                    "for a standalone run, which is also where the peak was measured")
    out_file = Path(os.environ.get("LLOYD_CEILING_PEAK_OUT",
                                   "/tmp/lloyd-ceiling-fullsuite.json"))
    argv, run = RC.apply([REPO_PY, "-m", "pytest", "tests/", "-q",
                          "-p", "no:cacheprovider", "-x", "--timeout=600"])
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=540,
                            env=RC.launcher_env(None))
    fields = run.settle()
    note = RC.finish(run)
    out_file.write_text(json.dumps({
        "rc": proc.returncode, "note": note,
        "mem_peak": fields.get("mem_peak"), "pids_peak": fields.get("pids_peak"),
        "memory_max": RC.MEMORY_MAX, "tasks_max": RC.TASKS_MAX,
        "tail": proc.stdout[-400:],
    }) + "\n", encoding="utf-8")
    assert note is None, \
        f"the ceiling stopped the suite it is supposed to protect: {note}"
    assert proc.returncode == 0, f"suite red under the ceiling: {proc.stdout[-2000:]}"




# ── the environment the launcher needs, which is what availability really means ──

def test_availability_survives_the_curated_environment_a_sandbox_spawn_gets(monkeypatch):
    """A curated env with no `$XDG_RUNTIME_DIR` must not read as "this box has no ceiling".

    `systemd-run --user --scope` asks the user manager over `$XDG_RUNTIME_DIR/bus`, and the
    variable is set by logind, not inherited by a process whose environment someone built by
    hand. The gate's `tests` rung does exactly that (`scripts/automod/gate.py:1153` builds
    `child_env` key by key and carries `XDG_RUNTIME_DIR` only when its own process has it),
    which is why an earlier round's armed branch could not run there: the probe answered
    `Failed to connect to user scope bus via local transport`, `apply` failed open like
    `_path_sandbox`, and the node that exists to prove a kernel kill got the bare argv
    instead — a ceiling silently missing on the sandboxed spawn, with only a `fallback` in
    `/state` to show for it.

    Two branches, so neither half can be faked. With the variables deleted from
    `os.environ`: `available()` still reports the launcher usable, and the command `apply`
    wrapped runs under `launcher_env` of that same env and reports its own cgroup under
    `lloyd.slice` and its 48-task cap. And the IDENTICAL argv run with the curated env
    *without* the fill must fail: that failure is the whole reason the fill exists, and a
    launcher that could reach the manager anyway would make this node's first half
    meaningless.
    """
    for var in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        monkeypatch.delenv(var, raising=False)
    curated = {k: v for k, v in os.environ.items()
               if k not in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")}

    armed, unavailable_because = RC.available()
    assert armed, f"no manager was reachable with no XDG_RUNTIME_DIR: {unavailable_because}"

    probe = ("CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); "
             "printf 'self=%s pids=%s\\n' \"$(cut -d: -f3 /proc/self/cgroup)\" "
             "\"$(cat $CG/pids.max)\"")
    argv, run = RC.apply(["/bin/sh", "-c", probe])
    assert argv[0] == "systemd-run", argv[:1]
    got = subprocess.run(argv, capture_output=True, text=True, timeout=CEIL_BUDGET_S,
                         env=RC.launcher_env(curated))
    assert got.returncode == 0, got.stderr[:400]
    assert "lloyd.slice" in got.stdout, got.stdout[:300]
    assert "app.slice" not in got.stdout, got.stdout[:300]
    assert f"pids={RC.TASKS_MAX}" in got.stdout, \
        f"the scope was made but not the cap: {got.stdout[:300]}"
    assert RC.finish(run) is None

    bare = subprocess.run(argv, capture_output=True, text=True, timeout=CEIL_BUDGET_S,
                          env=curated)
    assert bare.returncode != 0, (
        "the launcher reached the user manager with neither $XDG_RUNTIME_DIR nor "
        "$DBUS_SESSION_BUS_ADDRESS, so the fill this node pins is doing nothing and "
        "the branch above proves nothing either")
    assert "user scope bus" in bare.stderr, bare.stderr[:300]


# ── probes shared by the slice node and the kill nodes ─────────────────────────

# What a child can report about its own placement. Inside bwrap the child gets its own
# cgroup namespace, so `/proc/self/cgroup` reads `/` there — true of both sandbox profiles —
# and the *parent's* view (`/proc/$PPID/cgroup`, bwrap itself, not namespaced) is what names
# the slice. Both are asked for, and the assertion is that one of the two names
# `lloyd.slice` and neither names `app.slice`: a probe reading only its own view would pass
# on a child sitting in any cgroup at all.
_CGROUP_PROBE = ("printf 'self=%s parent=%s\\n' \"$(cut -d: -f3 /proc/self/cgroup)\" "
                 "\"$(cut -d: -f3 /proc/$PPID/cgroup 2>/dev/null)\"")

#: A 256 MiB allocation the child really touches, so the kernel accounts it. Placement
#: evidence that does not depend on a namespace: a child in some other cgroup cannot move
#: `memory.peak` in the scope the sampler watches.
_ALLOCATE_256_MIB = ('python3 -u -c "b = bytearray(256 * 1024 * 1024); b[0] = 1; '
                     'b[-1] = 1; print(chr(116), flush=True); '
                     'import time; time.sleep(0.3)"')


# ── Clauses 2 and 3 — the two planted shapes, killed and named ────────────────

def test_a_planted_fan_out_is_stopped_and_the_note_names_the_task_cap(tmp_path):
    """Clause 2: a planted fork-bomb-shaped command exits non-zero and the note names the
    task/pid limit and its value, on an argv produced by `_path_sandbox.profile_argv`.

    Through the bwrap wrapper, not around it: the command the ceiling bounds is the command
    the path sandbox builds, so the argv here is `systemd-run … -- bwrap … -- /bin/sh -c …`
    as the foreground spawn site assembles it. The shape forks from a real process, so the
    only thing that stops it is the kernel — this box answers a fork at `pids.max` with
    EAGAIN and no kill, and the ceiling's job is turning that silence into `TasksMax=48`
    plus the peak the cgroup recorded.

    The children `os._exit` without being reaped. That is deliberate and it is the shape
    under test: a caller that reaps lets the loop run to completion on the tasks it already
    holds, which exits 0 and would make this node assert nothing. What the item calls
    fork-bomb-shaped is a subtree that keeps trying to create tasks, and `pids.max` is what
    refuses it. bwrap is skipped when absent rather than faked, because the sandbox's
    fail-open path is a plain shell argv and composition asserted against a substitute argv
    proves nothing about the real one.
    """
    if not PS.available()[0]:
        pytest.skip(f"bwrap unavailable: {PS.available()[1]}")
    RC.TASKS_MAX = "48"
    script = tmp_path / "fan_out.py"
    script.write_text(_FAN_OUT_NO_REAP, encoding="utf-8")
    inner = PS.profile_argv(f"{REPO_PY} -u {script}", str(tmp_path))
    assert inner[0].endswith("bwrap"), inner[0]
    argv, run = RC.apply(inner)
    assert RC.slice_of(argv) == RC.CEILING_SLICE, argv
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=CEIL_BUDGET_S,
                            env=RC.launcher_env(None))
    note = RC.finish(run)
    assert proc.returncode != 0, \
        f"a subtree that could not fork exited 0, so the note would be decoration: " \
        f"out={proc.stdout[:200]!r} note={note!r}"
    assert note and "TasksMax" in note, \
        f"non-zero exit with no named limit — the reader gets rc={proc.returncode} and " \
        f"nothing else: out={proc.stdout[:200]!r} err={proc.stderr[:200]!r}"
    assert "48" in note, f"the note must name the cap the run was given: {note}"
    assert "EAGAIN" in note, \
        f"the note says the cap was reached but not what that did to the child: {note}"


def test_a_planted_allocator_is_killed_inside_the_byte_bound_and_named():
    """Clause 3: a planted single-process allocator dies under the memory limit, inside a
    stated byte bound, and the note names `MemoryMax` — with swap pinned to 0.

    The cap this node plants is `PLANTED_MEMORY_MAX` (256 MiB), not the shipped value:
    filling 16 GiB on a box with the engine daemons resident is how the agent stack gets
    taken down to prove a bound exists, which is the failure #1337 exists to prevent. What is
    *not* substituted is the mechanism — the same unit, the same cgroup files, the same
    kernel memory controller — and the shipped values are proved separately, without
    allocating them, by the byte-exact read-back in `test_the_shipped_values_are_the_ones_the
    _kernel_enforces`.

    The bound is 256 MiB + 64 MiB of headroom read out of the allocator's own progress
    lines: an allocator that churned 8 GiB before dying would be a different failure mode
    wearing this node's green check. Swap is asserted next because it is what makes the cap
    real here — with the box's 251.5 GiB of zram reachable, an over-cap allocation is
    answered by paging rather than by a kill, and `memory.peak` would report the limit while
    nothing was ever stopped.

    And the kernel's silence is the thing being translated: the run comes back rc=-9 (137
    through a shell) with no message anywhere, which is the bare exit code the item refuses
    to leave the reader with.
    """
    argv, run = _wrap_python(_ALLOCATOR, memory=PLANTED_MEMORY_MAX, swap="0")
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=CEIL_BUDGET_S,
                            env=RC.launcher_env(None))
    note = RC.finish(run)
    assert proc.returncode != 0, f"the allocator ran to completion: {proc.stdout[-200:]}"
    assert proc.returncode in (-9, 137), \
        f"expected the memory controller's silent SIGKILL (rc=-9, or 137 through a shell), " \
        f"got {proc.returncode}: {(proc.stdout + proc.stderr)[-300:]}"
    # #2925 clause 3, first of the file's two `bytearray(8 * 1024 * 1024)` payload sites: the
    # allocator printed the cap its own scope carried before it allocated anything, so the
    # bound is evidenced by the run itself and not only by the `memory=` on the line above.
    _planted_cap_bytes(proc.stdout)
    # The child must not pronounce the verdict: no note marker, no named limit, no kill
    # counter. That is what would make this node grade the payload instead of the wrapper's
    # translation of a silent kill. It is stated as the verdict's own wording rather than as
    # "no occurrence of 'ceiling'", because the payload's cap line necessarily contains that
    # substring — the scope it is running in is named `lloyd-bash-ceiling-*.scope`, and naming
    # that scope is the proof that the figure came from the ceiling's cgroup and not from the
    # slice root (#2925).
    child = proc.stdout + proc.stderr
    assert not any(token in child for token in ("[resource ceiling]", "MemoryMax=", "oom_kill")), \
        f"the child reported a ceiling verdict itself, so this node would be grading the " \
        f"payload rather than the wrapper's translation of a silent kill: {child[-300:]!r}"
    assert note and "MemoryMax" in note, \
        f"rc={proc.returncode} with no named limit: the item's whole complaint about a " \
        "bare exit code"
    assert "256M" in note, f"the note must name the cap the run was given: {note}"
    assert "swap=0" in note or "pinned to 0" in note, \
        f"the note does not state the swap pin that makes the cap bite: {note}"
    died_at = [int(m) for m in re.findall(r"bytes=(\d+)", proc.stdout)]
    assert died_at, "the allocator printed no progress, so the byte bound is unmeasured"
    assert max(died_at) <= _ALLOCATOR_BYTE_BOUND, \
        f"the allocator reached {max(died_at)} bytes against a cap of 268435456 plus a " \
        f"bound of {_ALLOCATOR_BYTE_BOUND}: the limit is not enforced at the cap"


def test_the_named_limit_reaches_the_tool_result_and_not_only_the_module():
    """The verdict has to be in the artifact the model reads.

    `finish()` returning a line proves the module can produce one; the item's acceptance is
    about the *tool result*, which is what a turn sees when its command dies. So the same
    allocator is run through the foreground spawn site and the note is asserted in that
    output, next to the `[exit code: …]` suffix the handler appends.

    #2925 is why this node now patches the cap. It used to run `_ALLOCATOR_SHELL` against the
    shipped `MEMORY_MAX` of 16G, so every gate run grew a real ~15 GiB buffer inside
    `lloyd.slice` — 242 launches of that payload between 2026-10-13 and 2026-10-17, teardown
    peaks up to 15.7G, ~100 s of wall clock each, which left the node racing the Bash tool's
    own 120 s timeout to get its kill in before a timeout note replaced the ceiling note. The
    claim under test is that a *named* limit reaches the tool result, and a named 256M is the
    same claim about the same code path; the shipped value is proved at the kernel, byte for
    byte and without allocating it, by `test_the_shipped_values_are_the_ones_the_kernel_enforces`.
    """
    RC.MEMORY_MAX = PLANTED_MEMORY_MAX
    out = _run_bash(_ALLOCATOR_SHELL)
    assert "MemoryMax" in out, f"the tool result carries no named limit: {out[-400:]!r}"
    # Clause 2: graded against the cap this node actually armed, the way the byte-bound node
    # asserts its own `256M`. Asserting only that *a* limit was named would still pass on a
    # note saying 16G, which is the exact sentence that stopped being true of this node.
    assert PLANTED_MEMORY_MAX in out, \
        f"the tool result named a limit, but not the {PLANTED_MEMORY_MAX} cap this node " \
        f"planted: {out[-400:]!r}"
    assert "exit code" in out, f"an OOM-killed command reported success: {out[-200:]!r}"
    # #2727 clause 4: this kill happens inside its own scope, so the honest new wording
    # (which reserves "somewhere in lloyd.slice" for a kill this call cannot evidence) must
    # not have taken this one. The subtree claim is the marker that it did not — and it is
    # still the claim, now that the kill underneath it happens at 256M instead of 16 GiB.
    assert "subtree" in out, \
        f"a kill at the planted {PLANTED_MEMORY_MAX} cap was reported without claiming its " \
        f"own subtree: {out[-400:]!r}"
    # Clause 3, second of the file's two `bytearray(8 * 1024 * 1024)` payload sites. The
    # figure comes back in this tool result because the payload read its own `memory.max`
    # before allocating: the process that read it was SIGKILLed and left no exit status of
    # its own, so an assertion on anything the harness reads out afterwards could go blind
    # exactly when the bomb grew largest.
    _planted_cap_bytes(out)


def test_a_planted_allocator_refuses_to_grow_against_an_oversized_scope():
    """The payload's own bound, enforced by the payload — so a dropped patch costs no RAM.

    `_planted_cap_bytes` is what fails when a node forgets to narrow the cap, but it reads the
    run's output, and output only exists once the allocator has been killed: at the shipped 16G
    that is ~100 s of wall clock and 15 GiB of a real scope, which is the thing this item exists
    to stop. So the bound is baked into both payloads, which print their cap and then exit 42
    rather than allocate past it. This node arms a scope one notch over the bound and proves the
    refusal from the three things that have to be true together: the run ends with 42, it names
    the oversized cap, and it printed no allocation progress at all.

    It also pins that the bound is in the payload and not merely in the harness: the comparison
    the child made was against a number compiled into the command string.
    """
    argv, run = _wrap_python(_ALLOCATOR, memory="1G", swap="0")
    started = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=CEIL_BUDGET_S,
                            env=RC.launcher_env(None))
    RC.finish(run)
    assert proc.returncode == 42, \
        f"an allocator planted in a 1G scope should refuse rather than fill it, got " \
        f"rc={proc.returncode}: {(proc.stdout + proc.stderr)[-200:]!r}"
    assert "cap=1073741824" in proc.stdout, \
        f"the refusal did not name the oversized cap it read: {proc.stdout[-200:]!r}"
    assert "bytes=" not in proc.stdout, \
        "the payload allocated before refusing, so a dropped patch still grows the bomb — the " \
        "cap has to be checked before the first bytearray"
    assert time.monotonic() - started < 10.0, \
        "a refusal that takes minutes is not a guard; it is the old 16G node with a new name"


# ── #2727 — whose kill: the slice's cumulative counter versus this call's own ───

def test_a_slice_only_oom_delta_is_reported_as_the_slices_kill_and_not_as_this_calls():
    """Clause 1: a call whose only kill evidence is the slice's cumulative counter gets a
    note that names `lloyd.slice` and refuses to say this call's subtree was killed.

    The shape is exactly what an unrelated call sees on this box while the gate's `tests`
    rung runs the two allocator nodes above: this call's own counters are clean — no
    `memory.events oom_kill`, a peak far under the cap — and the slice, which every
    concurrent subtree in it shares, went up by 2. Before the split, one `if` admitted both
    evidences and the branch's own words claimed the kill:
    `memory.oom_kill rose by 2 in lloyd.slice during this call; the memory controller killed
    a process in the subtree`, which is what 88 notes across the Oct-13/14 session stores
    said, and what an autonomy session read as its own command dying and filed as incident
    #2725 against a scanner that had not allocated anything.

    The note must still appear: a kill happened in the ceiling's own slice during this
    window, and silence would lose the detection the slice counter is the only witness of.
    What changes is that the sentence is now about the slice, and says in words that no kill
    is attributed to this call.
    """
    fields = {"mem_peak": "1048576", "mem_max": "17179869184",
              "memory_events": "low=0;high=0;max=0;oom=0;oom_kill=0",
              "oom_kill_delta": "2"}
    note = RC.ceiling_note(fields)
    assert note is not None, \
        "a kill in the ceiling's slice went unreported, so the counter stopped being a witness"
    assert RC.CEILING_SLICE in note, f"the note does not name where the kill was counted: {note}"
    assert "rose by 2" in note, f"the delta the slice recorded is missing from the note: {note}"
    assert "MemoryMax" in note, f"the note no longer names the limit kind: {note}"
    for banned in ("in the subtree", "terminated the subtree", "killed a process"):
        assert banned not in note, \
            f"{banned!r} claims this call's subtree was killed on the slice's counter alone: {note}"
    assert "no kill is attributed to it" in note, \
        f"the note reports a slice-wide kill without disclaiming this call: {note}"


def test_the_subtree_kill_wording_is_claimed_only_from_this_calls_own_counters():
    """Clause 2: the subtree sentence is licensed by two witnesses and by nothing else —
    the scope's own `memory.events oom_kill`, or a `memory.peak` that reached its cap.

    One node, five shapes, because the defect was precisely a merged `if`: the two honest
    witnesses and the slice-cumulative delta all reached the same branch, so the branch's
    wording had to fit all three and fitted only the first two. `both` is the shape the box
    actually produces for a planted kill (measured over five 256 MiB allocators: the scope's
    own `oom_kill=1` arrived every time, and the slice had moved by one too) — the own
    counter wins the wording there, which is the case that must not regress when the
    slice-only case gains its own softer words. `unknown` pins the other half of the rule:
    an unreadable slice file yields `None`, and `None` is not a zero that may be reported as
    a clean run, nor a kill that may be reported at all.
    """
    cases = {
        "own-events": ({"memory_events": "max=3;oom_kill=1", "mem_max": "268435456",
                        "mem_peak": "0", "oom_kill_delta": "0"}, "subtree"),
        "own-peak": ({"memory_events": "max=0;oom_kill=0", "mem_max": "268435456",
                      "mem_peak": "268435456", "oom_kill_delta": "0"}, "subtree"),
        "both": ({"memory_events": "max=1;oom_kill=1", "mem_max": "268435456",
                  "mem_peak": "0", "oom_kill_delta": "1"}, "subtree"),
        "slice-only": ({"memory_events": "max=0;oom_kill=0", "mem_max": "268435456",
                        "mem_peak": "1048576", "oom_kill_delta": "1"}, "slice"),
        "unknown": ({"memory_events": "max=0;oom_kill=0", "mem_max": "268435456",
                     "mem_peak": "0", "oom_kill_delta": None}, None),
    }
    for name, (fields, expect) in cases.items():
        note = RC.ceiling_note(fields)
        if expect is None:
            assert note is None, f"{name}: an unknown delta produced a verdict: {note}"
            continue
        assert note is not None, f"{name}: a recorded kill produced no note"
        if expect == "subtree":
            assert "subtree" in note, \
                f"{name}: this call's own counters show the kill and the note hedges: {note}"
        else:
            assert "subtree" not in note, \
                f"{name}: the slice's cumulative counter alone claimed a subtree kill: {note}"
            assert RC.CEILING_SLICE in note, \
                f"{name}: the note does not say where the kill it reports was counted: {note}"


def test_a_clean_call_concurrent_with_a_sibling_scope_kill_is_not_blamed_for_it():
    """Clause 3 across the real seam: a benign tool call that runs while a *sibling*
    ceiling scope is OOM-killed reports its own exit code and is not told it killed itself.

    Everything above this node is the module reading fields; this is the tool result. The
    killed scope is a separate `apply()`/subprocess with its own 256 MiB cap, and the
    innocent call is a real `B._bash` whose sleep spans the kill — the same geometry as a
    session's Bash call straddling the gate's allocator pair, which is the traffic that made
    88 misattributed notes. `_bash` runs in a worker thread only so the sibling can be
    started *after* the innocent call's scope exists; that ordering is what guarantees the
    kill lands inside the innocent window rather than before its baseline, and the innocent
    call's own thread-safe `asyncio.run` needs no main-thread signal handlers.

    The assertions then follow the counter rather than the clock: the node first proves the
    slice's `oom_kill` actually rose across the innocent call's span, and only then demands
    that the note appear, name the slice, and claim no subtree — so a scheduling hiccup that
    killed the sibling outside the window fails the premise, never the wording.
    Either way the exit code is the call's own — the innocent command is planted with
    `exit 3` rather than a clean one, because `_bash` appends `[exit code: N]` only for a
    non-zero rc, and a note that displaced that suffix would be invisible in a zero-exit
    result. No note in this file may say "subtree" about a call that allocated nothing.
    """
    slow_allocator = (
        "import time\n"
        "time.sleep(0.2)\n"
        "blocks = []\n"
        "while True:\n"
        "    blocks.append(bytearray(4 * 1024 * 1024))\n"
        "    time.sleep(0.04)\n"
    )
    shipped = RC.MEMORY_MAX
    argv, run = _wrap_python(slow_allocator, memory=PLANTED_MEMORY_MAX, swap="0")
    RC.MEMORY_MAX = shipped          # the innocent call runs production-shaped
    before = RC._slice_events().get("oom_kill")
    with ThreadPoolExecutor(max_workers=1) as pool:
        innocent = pool.submit(_run_bash, "echo alive; sleep 8; exit 3")
        time.sleep(1.0)              # the innocent scope exists before the kill can start
        sibling = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=RC.launcher_env(None))
        out = innocent.result(timeout=90)
        rc = sibling.wait(timeout=90)
    after = RC._slice_events().get("oom_kill")
    RC.finish(run)                   # the sibling's own verdict, discarded here
    assert rc in (-9, 137), f"the sibling scope was not OOM-killed (rc={rc}); the node's "\
                             "premise did not fire"
    assert before is not None and after is not None and after > before, \
        f"the slice's oom_kill did not move ({before} -> {after}); nothing happened to "\
        "attribute wrongly"
    assert "[exit code: 3]" in out, \
        f"the ceiling overwrote an innocent call's own exit code, or hid it: {out[-300:]!r}"
    assert "[resource ceiling]" in out, \
        f"the slice rose by {after - before} across this call's window and the note went "\
        "silent — detection lost, which is the other half of the same defect"
    assert RC.CEILING_SLICE in out, \
        f"the note reported a kill without naming the slice it was counted in: {out[-300:]!r}"
    assert "in the subtree" not in out and "terminated the subtree" not in out, \
        f"a sibling scope's kill was blamed on a call that allocated nothing: {out[-300:]!r}"


def test_the_shipped_values_are_the_ones_the_kernel_enforces():
    """The shipped constants, read back from inside a live scope — proved without allocating them.

    The kill nodes run at `PLANTED_MEMORY_MAX` (256 MiB) and `RC.TASKS_MAX = "48"` so the
    agent stack is never at risk; that substitution has to be paid for somewhere, and it is
    paid here. The same `apply`/`scope_argv` path is asked for the shipped constants and
    `memory.max`, `memory.swap.max` and `pids.max` are read back and compared to
    `RC.memory_bytes()` and `RC.TASKS_MAX`. A ceiling whose constant and whose enforced value
    disagree is a ceiling that does not exist, and only this comparison would notice.

    #2925 clause 5 is the other half of that comparison, and it is why this node patches
    nothing: an equality against `RC.memory_bytes()` is satisfied by *whatever* the constant
    says, so on its own it would happily certify a shipped cap someone had quietly shrunk to
    the test's own value. The literal below is the shipped value, and the read-back under it
    still proves the kernel got exactly that.
    """
    assert RC.MEMORY_MAX == "16G", \
        f"the shipped cap this node exists to prove at the kernel is {RC.MEMORY_MAX!r}, not " \
        f"'16G' — and this node patches nothing, so a test-local cap leaking into the " \
        f"shipped constant is exactly what this line is for"
    probe = ("CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); "
             "printf 'mem=%s swap=%s pids=%s\\n' \"$(cat $CG/memory.max)\" "
             "\"$(cat $CG/memory.swap.max)\" \"$(cat $CG/pids.max)\"")
    argv, run = RC.apply(["/bin/sh", "-c", probe])
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60,
                            env=RC.launcher_env(None))
    assert proc.returncode == 0, proc.stderr[:300]
    got = dict(kv.split("=", 1) for kv in proc.stdout.split())
    assert int(got["mem"]) == RC.memory_bytes(), \
        f"memory.max={got['mem']} but MEMORY_MAX={RC.MEMORY_MAX} ({RC.memory_bytes()})"
    assert got["swap"] == "0", f"MemorySwapMax is not reaching the scope: {got}"
    assert int(got["pids"]) == int(RC.TASKS_MAX), got["pids"]
    RC.finish(run)


# ── Clause 4 — armed vs unavailable, in the sandbox's own vocabulary ──────────

def test_the_readout_says_armed_or_unavailable_in_the_sandboxs_shape(caplog):
    """Clause 4: `status()` answers armed-or-not in the same shape as `_path_sandbox.status()`,
    and the unavailable branch is a WARNING plus a state field, never a bound that is absent.

    The shape is the point: `/state` already carries two such readouts
    (`protected_path_sandbox`, `tool_sandbox`) and `scripts/service_health_check.py` greps
    them with one pattern. Both sandboxes fail open with a WARNING, which is the behaviour
    the item names to copy *and* to fix — a ceiling that failed open silently would be worse
    than no ceiling, because the next reader would believe the bound was there. So this node
    asserts the keys, the `error`/`fallback` pairing on both branches, and the WARNING.

    The shared key list is measured on this tree, not assumed: `_path_sandbox.status()`
    returns exactly `['bwrap', 'enforcing', 'entries', 'error', 'fallback']`, so the
    shape this readout has to share is `enforcing` / `fallback` / `error`, with `error`
    carrying the reason. There is no `available_since` in either module — an earlier
    draft of this node asserted that key on both statuses and failed on the *sandbox*
    first, because the key it was copied from does not exist anywhere. And
    `agent_mcp/egress.status()` is `['enforce', 'telemetry']`, so the two bwrap
    sandboxes are the precedent the clause names, not the egress flag.
    """
    armed = RC.status()
    sandbox = PS.status()
    for key in ("enforcing", "fallback", "error"):
        assert key in armed, f"{key} missing from the ceiling readout: {sorted(armed)}"
    assert set(("enforcing", "fallback", "error")).issubset(sandbox), \
        "the ceiling's readout drifted from the shape it is meant to share"
    assert armed["enforcing"] is True and armed["fallback"] is False, \
        f"this box can arm a scope and the readout says otherwise: {armed}"
    assert armed["error"] is None and isinstance(armed["fallback"], bool)
    assert armed["MemoryMax"] == RC.MEMORY_MAX and armed["TasksMax"] == RC.TASKS_MAX
    assert armed["memory_bytes"] == RC.memory_bytes()
    assert armed["slice"] == RC.CEILING_SLICE == "lloyd.slice", armed["slice"]
    assert "app.slice" not in json.dumps(armed), \
        "the readout mentions app.slice: the monitored cgroup is not where this belongs"

    os.environ["LLOYD_DISABLE_BASH_CEILING"] = "1"
    try:
        RC.reset_probe_cache()
        with caplog.at_level(logging.WARNING,
                             logger="lloyd-builtin-resource-ceiling"):
            argv, run = RC.apply(["/bin/sh", "-c", "echo still-ran"])
        assert argv == ["/bin/sh", "-c", "echo still-ran"], \
            "the unavailable path rewrote the argv instead of passing it through"
        assert run is None
        st = RC.status()
        assert st["enforcing"] is False and st["fallback"] is True and st["error"], \
            f"unavailable reported as a bound: {st}"
        assert any("UNBOUNDED" in r.getMessage() and r.levelno >= logging.WARNING
                   for r in caplog.records), \
            "the ceiling failed open without a WARNING, the silent failure the item " \
            "exists to prevent"
    finally:
        os.environ.pop("LLOYD_DISABLE_BASH_CEILING", None)
        RC.reset_probe_cache()
    assert RC.ceiling_note({}) is None, \
        "an empty counter set produced a verdict, so absence would read as a clean run"


def test_a_call_whose_child_outlives_it_costs_the_same_wall_clock_as_one_without_it():
    """Clause 5's timing half: a detached descendant costs no extra wall-clock with the
    ceiling armed, measured against the same command with the ceiling disabled.

    `--scope` without `--die-with-parent` means a child that outlives its call is reparented
    rather than waited for, and `exec setsid … >/dev/null` is the fully-detached form a real
    caller uses — a naive `( sleep & )` still holds the pipes, and holding pipes is the
    shell's behaviour with or without a scope. The budget is the detached sleep's own length
    plus slack, so a ceiling that waited for descendants would trip this node rather than
    merely look slow.
    """
    cmd = f"exec setsid sleep {DETACHED_SECONDS} >/dev/null 2>&1 < /dev/null & echo detached"
    armed_argv, run = RC.apply(["/bin/sh", "-c", cmd])
    t0 = time.monotonic()
    armed = subprocess.run(armed_argv, capture_output=True, text=True,
                           timeout=CEIL_BUDGET_S, env=RC.launcher_env(None))
    armed_wall = time.monotonic() - t0
    assert armed.returncode == 0 and "detached" in armed.stdout, armed.stderr[:200]
    assert RC.finish(run) is None, "a call that backgrounded one sleep tripped a limit"

    t0 = time.monotonic()
    plain = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True,
                           timeout=CEIL_BUDGET_S)
    plain_wall = time.monotonic() - t0
    assert plain.returncode == 0, plain.stderr[:200]

    assert armed_wall < DETACHED_SECONDS, \
        f"the ceiling held a call whose work had detached: {armed_wall:.2f}s against a " \
        f"{plain_wall:.2f}s baseline"
    assert armed_wall - plain_wall < 1.0, \
        f"the ceiling added {armed_wall - plain_wall:.2f}s to a call that does nothing " \
        f"but background a process ({armed_wall:.2f}s vs {plain_wall:.2f}s)"


# ── payload bodies the kill nodes plant ───────────────────────────────────────

#: The fan-out without a reap loop: children `os._exit` immediately, so the parent keeps
#: trying to create tasks and the only thing that stops it is `pids.max`. A reaping variant
#: completes on the tasks it already holds and exits 0, which would make clause 2 untestable.
_FAN_OUT_NO_REAP = (
    "import os\n"
    "for i in range(400):\n"
    "    try:\n"
    "        pid = os.fork()\n"
    "    except OSError as exc:\n"
    "        print('blocked-at=%d errno=%d' % (i, exc.errno), flush=True)\n"
    "        raise SystemExit(3)\n"
    "    if pid == 0:\n"
    "        os._exit(0)\n"
    "print('never-blocked', flush=True)\n"
    "raise SystemExit(4)\n"
)

#: The allocator as the shell string a turn would actually send, so the named limit can be
#: asserted inside the tool result and not only in the module's return value. Plain `print`,
#: no f-string: the payload sits inside shell double quotes, and nested quoting is how a
#: planted command stops being the thing under test.
_ALLOCATOR_SHELL = (_MEMCAP_READ_BACK_SHELL +
                    'python3 -u -c "blocks = []\n'
                    'while True:\n'
                    '    blocks.append(bytearray(8 * 1024 * 1024))\n'
                    '    print(len(blocks) * 8 * 1024 * 1024, flush=True)"')
