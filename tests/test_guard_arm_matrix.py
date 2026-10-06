"""#1963 — the guard-by-dispatch-path matrix is derived from the tree.

`architecture/guard-coverage.md` hand-maintains which dispatch path arms which
guard, and its own review found the prose wrong where a call-site grep is blind:
an arm made inside another installer's body. `app/harness/guard_arm_matrix.py`
reads installer bodies; `scripts/maintenance/guard_arm_matrix.py` prints it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.harness import guard_arm_matrix as G
from app.harness import outbound_content as OC

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "maintenance" / "guard_arm_matrix.py"
INTERACTIVE = "app/routers/turn_options.py"


# ── clause 1: four guards, every dispatch path, the live tree ────────────────

def test_the_matrix_covers_the_four_hook_installed_guards():
    assert G.GUARDS == ("safety", "policy", "outbound_content", "action_review")
    names = {n for group in G.GUARD_INSTALLERS.values() for n in group}
    assert {"install_default_safety_hook", "install_policy_hook",
            "install_outbound_content_gate", "_install_action_review"} <= names
    assert set(OC.INSTALLERS) < names, (
        "the matrix is no wider than the one-gate pair it was meant to widen")


def test_every_dispatch_path_has_a_row_and_the_interactive_builder_arms_all_four():
    matrix = G.guard_arm_matrix(ROOT)
    assert set(matrix) == OC.dispatch_registry_sites_files(ROOT), (
        "the rows are not the gate roster's own denominator")
    assert len(matrix) >= 5, sorted(matrix)
    assert matrix[INTERACTIVE] == {g: True for g in G.GUARDS}, matrix[INTERACTIVE]


# ── clause 2: an arm is read from the installer's body ───────────────────────

def test_the_transitive_arm_is_seen_where_no_call_site_exists():
    """`turn_options.py` never names the outbound gate. The floor's body does."""
    source = (ROOT / INTERACTIVE).read_text(encoding="utf-8")
    assert "install_outbound_content_gate(" not in source, (
        "the file now calls the gate directly; pick another witness for the "
        "transitive arm or this test proves nothing")
    assert "install_outbound_content_gate" in G.installer_bodies(ROOT)[
        "install_default_safety_hook"]
    assert G.guard_arm_matrix(ROOT)[INTERACTIVE]["outbound_content"] is True


def test_the_matrix_agrees_with_the_one_gate_derivation_it_generalises():
    """On the outbound gate the two derivations must give one answer: every file
    the existing check calls armed is a row that arms `outbound_content`."""
    matrix = G.guard_arm_matrix(ROOT)
    armed = OC.armed_files(ROOT)
    assert armed, "the one-gate derivation found nothing; no comparison was made"
    assert {rel for rel in armed if not matrix[rel]["outbound_content"]} == set()


def test_a_floor_only_path_is_not_credited_with_a_guard_it_never_reaches():
    matrix = G.guard_arm_matrix(ROOT)
    floor_only = [rel for rel, cells in matrix.items()
                  if cells["safety"] and not cells["policy"]]
    assert floor_only, "no floor-only path on this tree; the negative is untested"
    for rel in floor_only:
        assert "install_policy_hook(" not in (ROOT / rel).read_text(encoding="utf-8")
    # And the worker builder, which installs no Bash hook, is not handed one.
    assert matrix["workers/sources/_common.py"]["safety"] is False
    assert matrix["workers/sources/_common.py"]["policy"] is True


def _tree(tmp_path: Path, *, floor_arms_gate: bool) -> Path:
    """A three-file tree: an installer module and two turn builders."""
    (tmp_path / "app").mkdir()
    gate_call = "    install_outbound_content_gate(hooks)\n" if floor_arms_gate else ""
    (tmp_path / "app" / "guards.py").write_text(
        "def install_outbound_content_gate(hooks):\n    hooks.add('gate')\n\n"
        "def install_policy_hook(hooks):\n    hooks.add('policy')\n\n"
        "def install_default_safety_hook(hooks):\n    hooks.add('floor')\n" + gate_call
        + "\ndef _install_wrapper(hooks):\n    install_policy_hook(hooks)\n")
    for name, call in (("chat.py", "install_default_safety_hook(hooks)"),
                       ("worker.py", "_install_wrapper(hooks)")):
        (tmp_path / "app" / name).write_text(
            "from app.guards import *\n\n"
            "def build():\n    hooks = HookRegistry()\n    " + call + "\n"
            "    return RunOptions(mcp_servers={'lloyd': 1}, hooks=hooks)\n")
    return tmp_path


def test_on_a_tree_this_file_writes_the_body_decides_the_arm(tmp_path):
    (tmp_path / "a").mkdir()
    with_gate = G.guard_arm_matrix(_tree(tmp_path / "a", floor_arms_gate=True))
    assert with_gate["app/chat.py"] == {
        "safety": True, "policy": False, "outbound_content": True,
        "action_review": False}
    assert with_gate["app/worker.py"] == {
        "safety": False, "policy": True, "outbound_content": False,
        "action_review": False}, "a wrapper's body is followed; nothing else is credited"
    assert "app/guards.py" not in with_gate, (
        "calls inside an installer's own definition belong to its callers")

    (tmp_path / "b").mkdir()
    without = G.guard_arm_matrix(_tree(tmp_path / "b", floor_arms_gate=False))
    assert without["app/chat.py"]["outbound_content"] is False, (
        "the same call site, a different body: the arm is gone — which a grep "
        "of the call site cannot see in either direction")


# ── clause 3: a non-test caller ──────────────────────────────────────────────

def test_the_script_prints_one_row_per_dispatch_path_and_exits_zero():
    out = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    rows = [ln for ln in out.stdout.splitlines() if ln.startswith("| `")]
    paths = [ln.split("`")[1] for ln in rows]
    assert sorted(paths) == sorted(G.guard_arm_matrix(ROOT)), paths
    interactive = next(ln for ln in rows if f"`{INTERACTIVE}`" in ln)
    assert interactive.rstrip().endswith(
        "| safety, policy, outbound_content, action_review |"), interactive


def test_the_script_fails_on_a_tree_with_no_dispatch_path(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path)],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert out.returncode == 1 and "no dispatch path" in out.stderr


# ── #2269: the capability envelope rides the same roster machinery ───────────
#
# A source's tool envelope is the other per-dispatch-path fact this module
# exists to keep honest, and the risk is the same one #1828 was written about:
# a fact held in a hand-maintained table reads as clean for the row nobody
# added. So the denominator below is `workers/sources/__init__.py`'s own import
# list read from the AST, and the two nodes that follow are one positive and one
# forged negative — the negative exists because "no source trips the finding" is
# equally true of a check that can never fire.

def test_the_capability_report_enumerates_the_source_roster():
    m = G.capability_matrix(ROOT)
    assert set(m) == G.worker_source_names(ROOT), (
        "the capability report and the roster's import list disagree: a source "
        "exists that the report never looked at")
    assert len(m) >= 10, f"only {len(m)} sources enumerated"
    # The AST read is the whole drift mechanism, so it is checked against the
    # package the tests already import, rather than trusted.
    from workers.sources import SOURCE_REGISTRY
    assert set(m) == set(SOURCE_REGISTRY), (
        f"AST {sorted(set(m) ^ set(SOURCE_REGISTRY))} vs the live registry")
    # Every un-narrowed source is a row, named, not an absent line.
    found = {f.target for f in G.capability_findings(ROOT)
             if f.reason == G.FINDING_UNDECLARED_ENVELOPE}
    assert found == {s for s, c in m.items() if not c["declared"]}
    assert found, "no source is undeclared, which means this column stopped "  \
                  "being a denominator and became a decoration"


def test_the_report_prints_reach_and_names_a_declaration_that_grants_a_sender(
        monkeypatch):
    m = G.capability_matrix(ROOT)
    text = G.render_capabilities(m)
    for source in ("youtube-digest", "deep-research", "session-distill"):
        row = next(ln for ln in text.splitlines() if f"`{source}`" in ln)
        assert m[source]["declared"] and m[source]["durable_reach"] == []
        assert "| none |" in row, row
    # The un-narrowed rows print `—`, not `0`: a cell that reads as zero reach
    # for a source whose reach was never measured is the lie `_counting` and
    # `_completeness` above exist to prevent.
    assert "— |" in text
    # The forged negative is the node: "no source trips the finding" is equally
    # true of a check that can never fire, and this whole file exists because of
    # a guard whose denominator could be empty (#1828). One declared name added,
    # one durable sender, and the report has to name the source AND the name.
    import app.harness.capabilities as caps
    monkeypatch.setitem(caps.SOURCE_CAPABILITIES, "deep-research",
                        tuple(caps.SOURCE_CAPABILITIES["deep-research"])
                        + ("email_send",))
    hits = [f for f in G.capability_findings(ROOT)
            if f.reason == G.FINDING_INGEST_REACHES_DURABLE]
    assert [f.target for f in hits] == ["deep-research"], hits
    assert "email_send" in hits[0].extra


def test_the_report_surface_prints_the_envelopes_on_demand():
    """The run command a cold reader is told to run, run.

    The default output is unchanged on purpose — `architecture/guard-coverage.md`
    and the node above pin it — so the capability table is opt-in, and this is
    what makes it reachable rather than a function nothing calls.
    """
    out = subprocess.run([sys.executable, str(SCRIPT), "--capabilities"],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "worker sources with a capability envelope" in out.stdout
    assert "`youtube-digest`" in out.stdout
    assert "worker sources with a capability envelope: 15" in out.stdout
    assert "| 24 durable-external reach" not in out.stdout
    # Every un-narrowed source is named in the findings block, once.
    assert "capability findings" in out.stdout
    assert out.stdout.count("worker-source-declares-no-capability-set") >= 10
