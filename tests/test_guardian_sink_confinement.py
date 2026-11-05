"""Which process may write a human-facing surface (#3149).

`memory/2026-10-21.md` carries three `## Self-mod guardian: Vault snapshots are not
arriving` sections naming `nonexistent-vault.git` — a fixture under the throwaway
checkout root, `~/lloyd-work/.t/` — as "the vault's only copy outside the vault", and
orders the reader to run `restore-vault.sh`. The process that wrote them was the automod
gate's review-prefetch checkout running a round's test: it imported the guardian, so it
could reach the alert sink, and the sink was the literal
`/home/alansrobotlab/obsidian`, which no caller could move.

That is a two-knob asymmetry, not a stray path. What the watchdog MEASURES was already
env-resolved (`vaultwatch.VAULT_ROOT` reads `LLOYD_VAULT_ROOT`,
`vaultwatch.snapshot_repo()` reads `LLOYD_VAULT_BACKUP_REPO`, and
`scripts/backup/backup-vault.sh` and `app/data_root.py::vault_root()` resolve the same
names); where the alert is WRITTEN was a constant. Two independent knobs means a caller
may set one and not the other, and the alert then carries a fixture-shaped subject into
a production surface.

Everything here therefore runs in a SUBPROCESS. In-process, `tests/
test_guardian_vaultwatch.py:88-89` redirects the sink twice —
`monkeypatch.setattr(policy, "VAULT_ROOT", ...)` and the same on `G.policy` — and
`:109-110` replaces `g.alert` wholesale with an append-to-list lambda, so a node built
on that fixture proves only that the fixture redirects. A subprocess has no fixture to
blame: `Guardian` is constructed as `main()` constructs it, `Guardian.alert` is the real
method, the real `Notifier` fans out, and the vault it appends to is whatever the child
resolves from its own environment.

The two mutes the probe sets (`LLOYD_JOURNAL_ALERTS`, `LLOYD_DESKTOP_ALERTS`,
`LLOYD_VOICE_ALERTS`) close the three channels that reach the room, not the vault
channel, which is the one under test. They are set because #2061's own lesson is that a
fixture alert in a shared human-facing surface is the same pollution whether it lands in
a note or in the live journal.

#3795 added the second half of the same question. #3149 bounded WHERE an alarm may write
and what it may name as its subject; it said nothing about the command the body tells a
person to run. `memory/2026-11-04.md:35,48,61` are confined and correctly owned and still
unusable, because each publishes `/usr/bin/python3
~/lloyd-work/SM_20261104_123801/home/lloyd/agent-services/guardian/vaultwatch.py status`
— a path that existed only inside the automod round whose tree the writing process had
loaded its modules from, and that round is deleted. So the nodes below ask the one
question that was missing: does the script path an alarm publishes survive the tree that
rendered it? Every published command is read back out of the daily note the alert really
wrote, for the same reason the nodes above read the note rather than a captured argument.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
assert GUARDIAN_DIR.is_dir(), f"{GUARDIAN_DIR} is not on the tree, so a probe run proves nothing"

#: The interpreter the guardian actually runs on: `policy.py`'s own header says this
#: tree is read from the pinned snapshot under `/usr/bin/python3`, and the alert body
#: the 2026-10-21 sections quote names `/usr/bin/python3` too. Falling back to the
#: pytest interpreter keeps the node runnable on a box without that path, and
#: `test_the_probe_runs_on_the_guardian_interpreter` below says which one ran.
PROBE_PY = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable

#: The vault THIS test process would write if it were unconfined, which is the
#: directory clause 2 asks stay untouched. Resolved the same way the code resolves it,
#: because under the gate's vault guard the live-looking vault is a mirror and
#: `/home/alansrobotlab/obsidian` is not what a leak would reach.
LIVE_VAULT = Path(os.environ.get("LLOYD_VAULT_ROOT") or (Path.home() / "obsidian"))

#: The alert path, unpatched. argv: the guardian directory, the scratch home, and a
#: per-run marker that goes into the alert's title.
_ALERT_PROBE = '''"""One real guardian alert, from a process whose only vault is its own HOME's."""
import json
import sys
import types
from pathlib import Path

guardian_dir, scratch, marker = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, guardian_dir)

import guardian as G      # noqa: E402  (stdlib-only siblings, as in production)
import policy             # noqa: E402

# Built the way `guardian.py`'s own `main()` builds it: a real `AutomodState`, a real
# `Notifier` over `policy.VAULT_ROOT`, no channel replaced.
args = types.SimpleNamespace(
    repo=str(scratch / "repo"), state=str(scratch / "state"),
    guardian_state=str(scratch / "gstate"), supervisor_sock="/nonexistent",
    backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
    programs="lloyd-mc:lloyd-backend", interval=5.0,
)
g = G.Guardian(args)
# `Guardian.alert`, not `Notifier.alert`: clause 2 is about the guardian's own alert
# path, and `Guardian.alert` is the method
# `tests/test_guardian_vaultwatch.py:109-110` replaces with an append-to-list lambda in
# every in-process node in this repository. Calling it here, undelegated and
# unreplaced, is the half a monkeypatched fixture structurally cannot do.
g.alert("error", "Sink confinement probe " + marker,
        "Probe alert from a scratch-HOME process; not a real incident.")
memory = Path(policy.VAULT_ROOT) / "memory"
print(json.dumps({
    "vault_root": policy.VAULT_ROOT,
    "notifier_vault_root": str(g.notifier.vault_root),
    "external_channels": g.notifier.external,
    "last_alert": g.last_alert,
    "notes": sorted(str(n) for n in memory.glob("*.md")),
}))
'''

#: The two resolvers, side by side, in one fresh process. Clause 1's property is that
#: they can never disagree, so the probe prints both rather than one.
_RESOLVER_PROBE = '''"""What the two vault knobs resolve to in a process with a given environment."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import policy      # noqa: E402
import vaultwatch  # noqa: E402

print(json.dumps({
    "policy": policy.VAULT_ROOT,
    "vaultwatch": str(Path(vaultwatch.VAULT_ROOT)),
    "home": os.path.expanduser("~"),
    "override": os.environ.get("LLOYD_VAULT_ROOT"),
}))
'''


def _child_env(home: Path | None, *, vault_override: str | None) -> dict[str, str]:
    """The environment for one probe run, with every mute and knob set explicitly.

    `LLOYD_VAULT_ROOT` is POPPED rather than left inherited: the gate's vault guard
    exports it at a mirror, and clause 2's whole subject is a caller that left it
    unset. Leaving it in would test a different contract than the one filed.
    """
    env = dict(os.environ)
    env.pop("LLOYD_VAULT_ROOT", None)
    env["LLOYD_JOURNAL_ALERTS"] = "0"
    env["LLOYD_DESKTOP_ALERTS"] = "0"
    env["LLOYD_VOICE_ALERTS"] = "0"
    if home is not None:
        env["HOME"] = str(home)
    if vault_override is not None:
        env["LLOYD_VAULT_ROOT"] = vault_override
    return env


def _run_probe(tmp_path: Path, source: str, *, home: Path | None,
               vault_override: str | None = None, marker: str = "") -> dict:
    script = tmp_path / "probe.py"
    script.write_text(source, encoding="utf-8")
    cmd = [PROBE_PY, str(script), str(GUARDIAN_DIR)]
    if home is not None:
        cmd.append(str(home))
    if marker:
        cmd.append(marker)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                          env=_child_env(home, vault_override=vault_override))
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _scratch_home(tmp_path: Path, *, with_memory: bool = True) -> Path:
    """A home in the shape a pytest tmp home has: an `obsidian/` with a `memory/`.

    The `memory/` directory is what makes the sink able to write at all —
    `Notifier._daily_note_path` answers None for a vault with no `memory/`, so a home
    without one would let the probe pass by doing nothing. It is created here so the
    only reason a note exists is that the sink fired.
    """
    home = tmp_path / "home"
    if with_memory:
        (home / "obsidian" / "memory").mkdir(parents=True, exist_ok=True)
    else:
        home.mkdir(parents=True, exist_ok=True)
    return home


def _marker_hits(root: Path, marker: str) -> list[str]:
    """Every file under `root/memory/` holding `marker`.

    Used on both sides of the same assertion, which is what makes the live-vault half
    more than an absence: the identical scan finds the marker in the scratch vault on
    the line below the line that finds nothing in the live one, so a zero here is not
    the zero of a scan that could not have found anything.
    """
    memory = root / "memory"
    if not memory.is_dir():
        return [f"{memory} is not a directory"]
    return [str(p) for p in sorted(memory.rglob("*.md"))
            if marker in p.read_text(encoding="utf-8", errors="replace")]


# ── clause 1: the sink resolves the vault the way the rest of the tree does ───

def test_the_alert_sink_resolves_the_vault_from_the_environment(tmp_path):
    """`policy.VAULT_ROOT` is `$LLOYD_VAULT_ROOT` when set, else `$HOME/obsidian`, and
    `vaultwatch.VAULT_ROOT` is the same two branches — which is what #3149's acceptance
    asks for, in these words: "the alert sink follows LLOYD_VAULT_ROOT/$HOME exactly as
    scripts/backup/backup-vault.sh:43 and app/data_root.py:48 already do".

    The property is that the two resolvers cannot disagree, in any environment a caller
    can build. That is the asymmetry the 2026-10-21 incident needed: a caller moved the
    tree being MEASURED and could not move the note being WRITTEN. The third case is the
    positive control, and the one that keeps this node honest about production — under
    the deployed environment both names still resolve to the real vault, so confinement
    here is a fact about the caller's environment and not a fixture that redirects the
    answer away from a sink that was never live.
    """
    scratch = _scratch_home(tmp_path)
    mirror = tmp_path / "vault-mirror"

    at_home = _run_probe(tmp_path, _RESOLVER_PROBE, home=scratch, vault_override=None)
    assert at_home["policy"] == str(scratch / "obsidian"), at_home
    assert at_home["vaultwatch"] == at_home["policy"], at_home

    at_mirror = _run_probe(tmp_path, _RESOLVER_PROBE, home=scratch,
                           vault_override=str(mirror))
    assert at_mirror["policy"] == str(mirror), at_mirror
    assert at_mirror["vaultwatch"] == at_mirror["policy"], at_mirror
    assert at_mirror["policy"] != at_home["policy"], (
        "the override moved nothing, so the two knobs are one knob and the "
        "asymmetry this item filed is not the one that fired")

    deployed = _run_probe(tmp_path, _RESOLVER_PROBE, home=None, vault_override=None)
    assert deployed["policy"] == str(Path.home() / "obsidian"), deployed
    assert deployed["vaultwatch"] == deployed["policy"], deployed


# ── clause 2: an unpatched alert under a scratch HOME writes only that scratch ──

def test_an_unpatched_guardian_alert_under_a_scratch_home_never_reaches_the_live_vault(
        tmp_path):
    """The clause as filed: HOME at a scratch home, `LLOYD_VAULT_ROOT` unset, and
    `Guardian.alert` NOT patched, and the section must land in that home's `memory/` and
    nowhere under the live vault's.

    The order of the two halves is the whole point. The scratch assertion is checked
    first because it is the one that makes the live assertion mean anything: a sink that
    wrote nothing at all would satisfy "the live vault is untouched" perfectly, and that
    is the shape of a confinement test that passes because the code under test never
    ran. Only once a real `## Self-mod guardian:` section is on disk in the scratch
    vault does the zero in the live vault say something about confinement.

    Note what is NOT patched here, and could not be from this side of the process
    boundary: `policy.VAULT_ROOT`, `G.policy.VAULT_ROOT` and `g.alert` are all live,
    which is exactly the pair of monkeypatches
    `tests/test_guardian_vaultwatch.py:88-110` uses and the reason that fixture cannot
    be the proof (its own docstring above this file's says so).
    """
    home = _scratch_home(tmp_path)
    marker = uuid.uuid4().hex[:12]

    out = _run_probe(tmp_path, _ALERT_PROBE, home=home, vault_override=None, marker=marker)

    assert out["vault_root"] == str(home / "obsidian"), out
    assert out["notifier_vault_root"] == str(home / "obsidian"), (
        "the Notifier was handed a vault other than the resolved one, so the sink is "
        "still being given a second answer at its call site")
    assert marker in out["last_alert"], out         # Guardian.alert itself dispatched
    assert len(out["notes"]) == 1, out              # written, not skipped

    scratch_hits = _marker_hits(home / "obsidian", marker)
    assert scratch_hits == [str(Path(out["notes"][0]))], (
        f"the section did not land in the scratch vault the probe resolved: {out}")

    live_hits = _marker_hits(LIVE_VAULT, marker)
    assert live_hits == [], (
        f"a scratch-HOME guardian wrote {live_hits} — the alert sink is still not the "
        f"vault the process resolved (probe said {out['vault_root']})")


def test_a_scratch_home_without_a_memory_directory_writes_no_note_anywhere(tmp_path):
    """The negative shape, pinned so the previous node's zero cannot be a fixture.

    A vault with no `memory/` is the one environment where `Notifier._daily_note_path`
    answers None and nothing is written at all. If that were the case in the node above,
    its live-vault assertion would be vacuously true — so this node makes the same probe
    run with `memory/` absent and shows the scratch side comes up empty, while the node
    above shows it comes up full. The difference between the two runs is the
    `memory/` directory, which is the only thing the two nodes differ in.
    """
    home = _scratch_home(tmp_path, with_memory=False)
    marker = uuid.uuid4().hex[:12]

    out = _run_probe(tmp_path, _ALERT_PROBE, home=home, vault_override=None, marker=marker)

    assert marker in out["last_alert"], out          # the alert ran …
    assert out["notes"] == [], out                   # … and wrote nothing: no `memory/`
    # The scan helper refuses to report a clean result for a vault with no `memory/`,
    # which is why this node asserts the directory's absence directly instead of
    # asking for a zero it could not tell from an unrunnable scan.
    assert not (home / "obsidian" / "memory").is_dir(), out
    assert _marker_hits(LIVE_VAULT, marker) == []


def test_the_probe_runs_on_the_guardian_interpreter():
    """Which interpreter the two nodes above actually used.

    `PROBE_PY` prefers `/usr/bin/python3`, the interpreter the guardian's pinned
    snapshot runs and the one the 2026-10-21 alert body names in its own
    "same reading, from a shell" line. On a box without it the nodes still run, on
    pytest's interpreter — this node exists so that substitution is a printed fact
    rather than an assumption, because a stdlib-only module importing cleanly under a
    venv is weaker evidence than importing cleanly under `/usr/bin/python3`.
    """
    assert Path(PROBE_PY).is_file(), PROBE_PY
    proc = subprocess.run([PROBE_PY, "-c",
                           "import sys; sys.stdout.write(sys.version.split()[0])"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    print(f"\n#3149 probe interpreter: {PROBE_PY} (python {proc.stdout.strip()})")


# ── #3795: the script path an alarm publishes must outlive the tree that wrote it ──

#: One real guardian alarm, rendered by a process whose guardian modules came from a tree
#: the caller names. argv: `<guardian tree> <scratch root> <family> <state dir> <mode>`.
#:
#: `family` picks which published command this process renders: `vault` trips the vault
#: tripwire for real, `data` trips the data root's, `snapshot` runs the stalled-backup
#: check that wrote the 2026-11-04 sections. `mode` is `promoted` (the guardian state dir
#: holds a promoted `bin/`, as on this box) or `unpromoted` (nothing there, and
#: `policy.REPO` pointed at a checkout holding the sources — the fallback branch).
#:
#: Nothing substituted is a stand-in for the thing under test: the trip is a real wipe of
#: files, the watch is the shipped `VaultWatch`/`DataWatch`, `Guardian.alert` is the real
#: method, and the published command is read back out of the daily note the alert wrote.
#: What IS stubbed is the four actions that reach the machine — `_stop_sync` stops a
#: supervisord program, `_pause_workers` POSTs at the live pool, `_vault_evidence` shells
#: out to `ps`, `heartbeat` writes the ledger — plus the two checks these families are not
#: about, exactly as `tests/test_guardian_alert_retraction.py:956-962` stubs them and for
#: the reason its docstring gives.
_PUBLISHED_COMMAND_PROBE = '''"""One real guardian alarm, published from a tree the caller names (#3795)."""
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

tree, scratch = Path(sys.argv[1]), Path(sys.argv[2])
family, gstate, mode = sys.argv[3], Path(sys.argv[4]), sys.argv[5]

sys.path.insert(0, str(tree))

import datawatch as DW      # noqa: E402
import guardian as G        # noqa: E402
import policy as GP         # noqa: E402
import vaultwatch as VW     # noqa: E402

home = scratch / "home"
note_vault = home / "obsidian"
(note_vault / "memory").mkdir(parents=True, exist_ok=True)

GP.REPO = str(scratch / "checkout")
if mode == "unpromoted":
    checkout = Path(GP.REPO) / "agent-services" / "guardian"
    checkout.mkdir(parents=True, exist_ok=True)
    for name in ("guardian.py", "vaultwatch.py", "datawatch.py"):
        shutil.copyfile(tree / name, checkout / name)

if family == "snapshot":
    # A repository the vault's backup stream owns (`snapshot_repo_is_owned` weighs it
    # against `snapshot_owner_dir()`, which is `$HOME`-relative) and has no commit: the
    # stalled stream the alert is about, inside the scratch home and nowhere else.
    os.environ["LLOYD_VAULT_BACKUP_REPO"] = str(
        VW.snapshot_owner_dir() / "fixture-vault.git")

# The tree the watch damages, deliberately NOT the tree the note is written into, so the
# damaged subject and the human-facing surface are two different directories.
watched = scratch / ("watched-" + family)
(watched / "notes").mkdir(parents=True, exist_ok=True)
for i in range(VW.TOPDIR_MIN_FILES + 5):
    (watched / "notes" / f"n{i}.md").write_text("x\\n", encoding="utf-8")
(watched / DW.ROOT_MARKER).write_text("", encoding="utf-8")
(scratch / "inert-vault").mkdir(exist_ok=True)
(scratch / "inert-data").mkdir(exist_ok=True)

GP.VAULT_ROOT = str(watched if family == "vault" else scratch / "inert-vault")
GP.DATA_ROOT = str(watched if family == "data" else scratch / "inert-data")
GP.ALERT_REPEAT_SECONDS = 0.0

args = types.SimpleNamespace(
    repo=str(scratch / "repo"), state=str(scratch / "state"),
    guardian_state=str(gstate), supervisor_sock="/nonexistent",
    backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
    programs="lloyd-mc:lloyd-backend", interval=5.0)
g = G.Guardian(args)
g.notifier.vault_root = note_vault                  # the surface this node reads back
g._stop_sync = lambda: "stopped (probe)"
g._pause_workers = lambda actor: "paused (probe)"
g._vault_evidence = lambda stamp: None
g.heartbeat = lambda *a, **k: None
g.ensure_chronic = lambda: None
g._runtime_data_incident = lambda *a, **k: []
DW.snapshot_report = lambda *a, **k: (True, "")

# The two probe runs share this guardian-state root on purpose — a round's
# `<home>/.local/state/lloyd-guardian` IS the deployed one — so a marker the earlier run
# latched is visible here, and a latched watch both skips the trip and mutes the snapshot
# check. Clearing an inherited latch is what makes the alarm this process publishes this
# process's, rather than a stale one read back off the shared state.
if g.vault.tripped():
    g.vault.clear()
if g.data.tripped():
    g.data.clear()

if family == "snapshot":
    g._vault_snapshot_stream(time.time())
else:
    g._vault_snapshot_stream = lambda now: None     # not this family's subject
    check = g.check_vault if family == "vault" else g.check_data
    check()                                        # arms the watch; nothing lost yet
    shutil.rmtree(str(watched / "notes"))
    check()                                        # the mass deletion, and the alert

marker_clear_with = None
sync_gate_msg = None
if family == "vault":
    marker_clear_with = (g.vault.tripped() or {}).get("clear_with")
    gate_ok, gate_msg = VW.sync_gate(str(watched), gstate)
    sync_gate_msg = None if gate_ok else gate_msg

notes = sorted((note_vault / "memory").glob("*.md"))
print(json.dumps({
    "tree": str(tree),
    "module_files": {"guardian": G.__file__, "policy": GP.__file__,
                     "vaultwatch": VW.__file__, "datawatch": DW.__file__},
    "note_paths": [str(n) for n in notes],
    "note_text": "\\n".join(n.read_text(encoding="utf-8") for n in notes),
    "marker_clear_with": marker_clear_with,
    "sync_gate_msg": sync_gate_msg,
    "marker_path": str(g.vault.marker) if family == "vault" else None,
    "snapshot_repo": os.environ.get("LLOYD_VAULT_BACKUP_REPO"),
    "checkout_src": str(Path(GP.REPO) / "agent-services" / "guardian"),
    "gstate_bin": str(gstate / "bin"),
}))
'''


def _published_trees(tmp_path: Path, *, promote: bool = True):
    """A promoted snapshot `bin/` and a round-worktree checkout, from one set of sources.

    `bin/` is the directory `guardian-stage.sh` promotes into and the one
    `lloyd-guardian.service:29`'s `ExecStart` runs from; the worktree copy is the same
    files under `<home>/lloyd-work/<round id>/home/lloyd/agent-services/guardian`, which
    is the shape a round's tree has and the shape that rendered the dead 2026-11-04
    command. Both hang off ONE guardian-state root, because a round's
    `<home>/.local/state/lloyd-guardian` IS the deployed one — `scripts/automod/
    worktree.py:65` symlinks every home entry except `lloyd` and `lloyd-data` — so the two
    processes share the state dir exactly as the incident's did.
    """
    gstate = tmp_path / "gstate"
    bin_dir = gstate / "bin"
    bin_dir.mkdir(parents=True)
    worktree = (tmp_path / "lloyd-work" / "SM_20261104_123801" / "home" / "lloyd"
                / "agent-services" / "guardian")
    worktree.mkdir(parents=True)
    sources = sorted(GUARDIAN_DIR.glob("*.py"))
    assert len(sources) > 10, f"{GUARDIAN_DIR} holds only {len(sources)} .py files"
    for src in sources:
        if promote:
            shutil.copyfile(src, bin_dir / src.name)
        shutil.copyfile(src, worktree / src.name)
    return bin_dir, worktree, gstate


def _run_published_probe(tmp_path: Path, tree: Path, family: str, gstate: Path, tag: str,
                         *, mode: str = "promoted") -> dict:
    """One probe run: `family`'s alarm, published by a process rooted in `tree`.

    `tag` names this run's scratch HOME and vault, and every node passes a different one
    per run. It is not cosmetic: two runs sharing one home share one daily note, and the
    second run's assertions would then be reading the first run's published line back off
    it — a fixture that could only get the identity claim right. The guardian-state root
    is deliberately NOT tagged: it is the one directory the two processes really do share.
    """
    scratch = tmp_path / f"probe-{family}-{tag}"
    script = tmp_path / f"published-probe-{family}-{tag}.py"
    script.write_text(_PUBLISHED_COMMAND_PROBE, encoding="utf-8")
    env = dict(os.environ)
    env.update({
        "HOME": str(scratch / "home"),
        "LLOYD_GUARDIAN_STATE": str(gstate),
        "LLOYD_AUTOMOD_STATE": str(scratch / "automod"),
        "LLOYD_DATA": str(scratch / "lloyd-data"),
        "LLOYD_JOURNAL_ALERTS": "0",
        "LLOYD_DESKTOP_ALERTS": "0",
        "LLOYD_VOICE_ALERTS": "0",
    })
    # Popped, not inherited: the gate exports these two at its own vault mirror and a
    # data root of its own, and a probe that measured a mirror would not be measuring the
    # tree it names.
    for var in ("LLOYD_VAULT_ROOT", "LLOYD_VAULT", "LLOYD_DAILY_NOTE_DIR"):
        env.pop(var, None)
    proc = subprocess.run(
        [PROBE_PY, str(script), str(tree), str(scratch), family, str(gstate), mode],
        capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode == 0, (
        f"probe ({family}/{mode}, tree {tree}) exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["note_paths"], f"the alarm wrote no daily note at all: {out}"
    return out


def _published_commands(text: str) -> "list[str]":
    """Every command line an alarm body publishes to its reader, in order.

    Read off the note, not off a captured argument: the note is the surface the operator
    opens, and it is where 2026-11-04's dead command was found.
    """
    return [ln.strip() for ln in text.splitlines() if "/usr/bin/python3 " in ln]


def _script_of(command: str) -> str:
    """The script path one published `/usr/bin/python3 <script> <verb>` line names."""
    parts = command.split()
    assert parts[0] == "/usr/bin/python3" and len(parts) >= 3, command
    return parts[1]


def _assert_route_outlives_the_round(out: dict, command: str, worktree: Path,
                                     bin_dir: Path, script_name: str) -> str:
    """The three things every published command must satisfy, checked on one run.

    Shared by the four clause nodes so none of them can be green on a different rule than
    its siblings: the route is the promoted snapshot, the file is really there, and the
    tree that rendered the alarm is not where the route points.
    """
    script = _script_of(command)
    assert script == str(bin_dir / script_name), (
        f"the alarm published {script!r}, not the promoted snapshot "
        f"{str(bin_dir / script_name)!r} (process loaded {out['module_files']})")
    assert Path(script).is_file(), (
        f"the alarm published a command for a script that is not on disk: {script}")
    assert not Path(script).is_relative_to(worktree), (
        f"the published route {script} is inside the round's tree {worktree}, which is "
        f"deleted when the round is cleaned — exactly the 2026-11-04 failure")
    assert script != out["module_files"].get(script_name.replace(".py", "")), (
        f"the alarm is still publishing the file it was loaded from: {script}")
    return script


def test_the_vault_snapshot_status_command_renders_the_same_from_either_tree(tmp_path):
    """Clause 1: the vault-snapshot body's published `/usr/bin/python3 … status` command
    "renders identically whether the check runs with `vaultwatch` loaded from the deployed
    tree or from a scratch/worktree copy".

    This is the line that broke. Three sections of `memory/2026-11-04.md` each order the
    reader to run `…/lloyd-work/SM_20261104_123801/home/lloyd/agent-services/guardian/
    vaultwatch.py status`, under "The same reading, and the same bound, from a shell:",
    and `ls -d ~/lloyd-work/SM_20261104_123801` exits 2 today: the restore directive in a
    live alarm cannot be run, and it is not recoverable by editing the note, because the
    next stall writes another one from whatever tree the writer was in.

    The two runs differ in one thing only — which copy of the guardian `sys.path` resolved
    first. Same sources, same HOME, same guardian-state root, same environment, so a
    divergence between the two published lines has exactly one available explanation,
    which is the one this item filed.
    """
    bin_dir, worktree, gstate = _published_trees(tmp_path)
    promoted = _run_published_probe(tmp_path, bin_dir, "snapshot", gstate, "promoted-tree")
    from_round = _run_published_probe(tmp_path, worktree, "snapshot", gstate, "round-tree")

    assert from_round["module_files"]["vaultwatch"] == str(worktree / "vaultwatch.py"), (
        f"the round-shaped run did not load its `vaultwatch` from the worktree: "
        f"{from_round['module_files']}")
    assert promoted["module_files"]["vaultwatch"] == str(bin_dir / "vaultwatch.py"), (
        f"the promoted run did not load from the promoted snapshot: "
        f"{promoted['module_files']}")

    at_promoted = _published_commands(promoted["note_text"])
    at_round = _published_commands(from_round["note_text"])
    assert len(at_round) == 1, f"the snapshot alarm published {at_round}"
    assert at_round == at_promoted, (
        f"one alarm published two commands: the promoted tree said {at_promoted} and the "
        f"round's tree said {at_round} — a reader of the round's copy has a path that "
        f"does not exist")
    _assert_route_outlives_the_round(from_round, at_round[0], worktree, bin_dir,
                                     "vaultwatch.py")


def test_the_vault_tripwire_clear_command_never_names_the_tree_it_loaded(tmp_path):
    """Clause 2: the vault-tripwire body's published `clear` command "does not name the
    tree the module was loaded from, and the same render from a scratch tree equals the
    render from the deployed tree".

    The trip is real — 25 files wiped from a watched root — and the command is read out of
    the daily-note section the trip's `critical` alert wrote, which is where the sentence
    "…then clear the tripwire with" sends a person.
    """
    bin_dir, worktree, gstate = _published_trees(tmp_path)
    promoted = _run_published_probe(tmp_path, bin_dir, "vault", gstate, "promoted-tree")
    from_round = _run_published_probe(tmp_path, worktree, "vault", gstate, "round-tree")

    assert from_round["module_files"]["vaultwatch"] == str(worktree / "vaultwatch.py")
    at_promoted = _published_commands(promoted["note_text"])
    at_round = _published_commands(from_round["note_text"])
    assert len(at_round) == 1, f"the vault tripwire published {at_round}"
    assert at_round[0].endswith(" clear"), at_round
    assert at_round == at_promoted, (
        f"the same trip published two clear commands: {at_promoted} vs {at_round}")
    _assert_route_outlives_the_round(from_round, at_round[0], worktree, bin_dir,
                                     "vaultwatch.py")


def test_the_data_tripwire_clear_command_never_names_the_datawatch_tree(tmp_path):
    """Clause 3: the data-tripwire body's published `clear` command "does not name the
    tree the `datawatch` module was loaded from".

    Same probe, its own family: the data root's marker file makes the watch arm, a wipe of
    `notes/` trips it, and the body that orders "swap it in with the stack stopped, then
    clear with" is read off the note. It names `datawatch.py`, so the route is the
    promoted snapshot's `datawatch.py` and not the copy this process imported.
    """
    bin_dir, worktree, gstate = _published_trees(tmp_path)
    promoted = _run_published_probe(tmp_path, bin_dir, "data", gstate, "promoted-tree")
    from_round = _run_published_probe(tmp_path, worktree, "data", gstate, "round-tree")

    assert from_round["module_files"]["datawatch"] == str(worktree / "datawatch.py")
    at_promoted = _published_commands(promoted["note_text"])
    at_round = _published_commands(from_round["note_text"])
    assert len(at_round) == 1, f"the data tripwire published {at_round}"
    assert at_round[0].endswith(" clear"), at_round
    assert at_round == at_promoted, (
        f"the same data trip published two clear commands: {at_promoted} vs {at_round}")
    _assert_route_outlives_the_round(from_round, at_round[0], worktree, bin_dir,
                                     "datawatch.py")


def test_the_trip_marker_clear_with_outlives_the_round_and_sync_gate_quotes_it(tmp_path):
    """Clause 4: `VaultWatch.trip()` writes a `clear_with` lying outside the tree that
    loaded the module, and `sync_gate`'s refusal quotes that same value.

    The marker side matters with no daily note in sight: `worktree.py:65` symlinks the
    home entries, so a round-rooted trip writes into the SHARED marker, and the deployed
    `ob sync` refusal reads `clear_with` back out of it and shows it to whoever is trying
    to sync. A worktree path there is a dead command on a refusal, from a process nobody
    is looking at any more.

    Both halves are read from disk in this node: the marker JSON the real trip wrote, and
    the string the real `sync_gate` returned for the tripped root.
    """
    bin_dir, worktree, gstate = _published_trees(tmp_path)
    from_round = _run_published_probe(tmp_path, worktree, "vault", gstate, "round-tree")
    promoted = _run_published_probe(tmp_path, bin_dir, "vault", gstate, "promoted-tree")

    clear_with = from_round["marker_clear_with"]
    assert clear_with, f"the trip wrote no marker command: {from_round}"
    assert Path(from_round["marker_path"]).is_file(), from_round["marker_path"]
    assert from_round["marker_path"] == promoted["marker_path"], (
        "the round's trip and the watchdog's own wrote different marker files, so this "
        "node is not measuring the one shared marker a round writes through")
    _assert_route_outlives_the_round(from_round, clear_with, worktree, bin_dir,
                                     "vaultwatch.py")
    assert clear_with == promoted["marker_clear_with"], (
        f"a round's trip and the watchdog's own publish different commands into the one "
        f"shared marker: {clear_with!r} vs {promoted['marker_clear_with']!r}")

    refusal = from_round["sync_gate_msg"]
    assert refusal, f"`ob sync` was not refused over a tripped root: {from_round}"
    assert clear_with in refusal, (
        f"sync_gate's refusal does not quote the marker's clear_with: {refusal!r}")
    assert str(worktree) not in refusal, (
        f"sync_gate's refusal quotes a path inside the round's tree: {refusal!r}")


def test_with_nothing_promoted_the_published_route_is_the_checkout_not_the_round(tmp_path):
    """The fallback branch, pinned because a route that only works once code has been
    promoted is a route that does not work on a box where staging never ran.

    With no `bin/` entry to name, the published script is the source checkout
    (`policy.REPO` + `agent-services/guardian/`), which `guardian-stage.sh` copies FROM
    and which a round's cleanup never touches. The negative control is the same probe with
    the promoted `bin/` present, whose route is the snapshot — so the node cannot be green
    on a helper that simply always answers one of the two.
    """
    bin_dir, worktree, gstate = _published_trees(tmp_path, promote=False)
    assert not list(bin_dir.glob("*.py")), "the promoted snapshot was filled anyway"
    from_round = _run_published_probe(tmp_path, worktree, "vault", gstate, "round-tree",
                                      mode="unpromoted")

    script = _script_of(from_round["marker_clear_with"])
    assert script == str(Path(from_round["checkout_src"]) / "vaultwatch.py"), (
        f"with nothing promoted the alarm published {script!r} instead of the checkout "
        f"route {from_round['checkout_src']}")
    assert Path(script).is_file(), f"the fallback route names a script that is not there: {script}"
    assert not Path(script).is_relative_to(worktree), (
        f"the fallback route {script} is inside the round's tree {worktree}")
    assert script != from_round["module_files"]["vaultwatch"]

    # The control is the SAME round-shaped tree with its `bin/` filled: the one difference
    # between the two runs is whether anything has been promoted, so a helper that ignored
    # that and always answered the checkout would fail here rather than look correct.
    ctrl_bin, ctrl_worktree, ctrl_gstate = _published_trees(tmp_path / "control")
    control = _run_published_probe(tmp_path, ctrl_worktree, "vault", ctrl_gstate,
                                   "promoted-control")
    control_script = _script_of(control["marker_clear_with"])
    assert control_script == str(ctrl_bin / "vaultwatch.py"), (
        f"with the snapshot promoted the route should have been "
        f"{ctrl_bin / 'vaultwatch.py'} but the alarm published {control_script!r} — the "
        f"same tree answered in both branches, which makes the assertion above "
        f"unfalsifiable")
    assert Path(control_script).is_file(), control_script
    assert control_script != script, (
        f"the promoted and unpromoted branches published the same route {script}")
