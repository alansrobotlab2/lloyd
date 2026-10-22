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
"""

from __future__ import annotations

import json
import os
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
