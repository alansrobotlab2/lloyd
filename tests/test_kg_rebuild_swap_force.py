"""kg_rebuild.py `swap --force` — the override lives on the surface it overrode.

The 2026-09-23 rebuild swapped in a tree whose gate had FAILED:
`rebuild-state.json` carries `gate: false` beside `swapped_at: 2026-09-23T14:08Z`,
and `gate.json` for that run shows `eval_category_regression` at -0.051 against
the -0.05 limit — the same check class that made the 2026-09-03 rebuild be
abandoned. That override was legitimately governed, but every word of the
governance sat in a hand-written `FORCE-SWAP-REASON.md` in the run directory.
`cmd_swap` recorded no marker at all, so `status`, `rollback` and the next
operator reading the state saw `gate: false` + `swapped_at` and could not tell a
governed override from a bug. These pin that a forced swap now says that it
overrode, why, and how many hand-stated rows the carry-over guard had found.
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import kg_rebuild  # noqa: E402

# Every seeded row is created after this, so both carry-over guards see it and
# none of this depends on the wall clock.
EXPORTED_AT = "2020-01-01T00:00:00+00:00"
WRITTEN_AT = "2026-09-04T01:00:00+00:00"
REASON = ('gate.json says eval_category_regression -0.049 vs the -0.05 limit: '
          '"re-measured ABBA on 2026-10-17, rule fixed before the result"')


def _args(**kw):
    """What `swap` really gets from argparse: --dry-run, --force, --reason."""
    fields = {"dry_run": False, "force": False, "reason": None}
    fields.update(kw)
    return SimpleNamespace(**fields)


def _world(tmp_path, monkeypatch, *, gate: bool, stated_facts: int = 0,
           carry_edges: int = 0):
    """A live tree + store and a rebuild tree + store, with the module's path
    constants pointed into `tmp_path`.

    `stated_facts` are hand-stated facts (`provenance="STATED"`) and
    `carry_edges` are carry-worthy edges (`origin="manual"`, which is in
    `CARRY_EDGE_ORIGINS`, and an extractive provenance so the edges half is
    matched on origin alone). Both are written after the recorded export, which
    is exactly what the carry-over guard refuses a swap over.
    """
    from app.kg_store import KGStore

    derived = tmp_path / "vault-derived"
    live, rebuild = derived / "facts", derived / "facts-rebuild"
    (live / "Old").mkdir(parents=True)
    (live / "Old" / "Old-state.md").write_text(
        "---\ntype: facts\nentity: Old\ncategory: state\nfacts:\n"
        "- id: stat-001\n  fact: the old tree\n  confidence: 0.9\n---\n\n# Old\n",
        encoding="utf-8")
    (rebuild / "New").mkdir(parents=True)
    (rebuild / "New" / "New-state.md").write_text(
        "---\ntype: facts\nentity: New\ncategory: state\nfacts:\n"
        "- id: stat-001\n  fact: the rebuilt tree\n  confidence: 0.9\n"
        "  created_at: '2026-09-04T00:00:00+00:00'\n  source_doc: knowledge/x.md\n"
        "---\n\n# New\n", encoding="utf-8")

    live_db, rebuild_db = derived / "kg.sqlite", derived / "kg-rebuild.sqlite"
    a = KGStore(live_db)
    a.entities.register("Old")
    with a.transaction() as c:
        for i in range(stated_facts):
            c.execute(
                "INSERT INTO facts_idx(entity, category, fact_id, text_hash, fact, "
                "created_at, provenance, file_path) VALUES (?,?,?,?,?,?,?,?)",
                ("Old", "state", f"stat-{i:03d}", f"h-stated-{i}",
                 f"stated during the rebuild #{i}", WRITTEN_AT, "STATED",
                 "Old/Old-state.md"))
    for i in range(carry_edges):
        a.edges.add({"source": "Old", "target": f"Partner {i}", "type": "uses",
                     "provenance": "EXTRACTED"}, origin="manual")
    a.close()

    b = KGStore(rebuild_db)
    b.entities.register("New")
    b.close()

    monkeypatch.setattr(kg_rebuild, "VAULT_DERIVED_ROOT", derived)
    monkeypatch.setattr(kg_rebuild, "VAULT_FACTS_ROOT", live)
    monkeypatch.setattr(kg_rebuild, "REBUILD_FACTS", rebuild)
    monkeypatch.setattr(kg_rebuild, "VAULT_KG_DB", live_db)
    monkeypatch.setattr(kg_rebuild, "REBUILD_DB", rebuild_db)
    monkeypatch.setattr(kg_rebuild, "STATE_PATH", derived / "rebuild-state.json")
    writes = []
    monkeypatch.setattr(kg_rebuild, "_set_write_enabled", writes.append)

    kg_rebuild.save_state(gate=gate, export={"exported_at": EXPORTED_AT})
    return derived, live, rebuild, live_db, writes


def _state():
    return json.loads(kg_rebuild.STATE_PATH.read_text(encoding="utf-8"))


# ── clause 1: the marker is there when it overrode, absent when it did not ─────

def test_a_forced_swap_records_the_override_beside_the_swap(tmp_path, monkeypatch):
    """The 2026-09-23 shape: gate false, swap performed anyway.

    `swapped_at` and the fact that it was forced must be one record, not a state
    file and a prose file in a run directory nothing reads.
    """
    _world(tmp_path, monkeypatch, gate=False)

    assert kg_rebuild.cmd_swap(_args(force=True, reason=REASON)) == 0

    state = _state()
    assert state["forced"] is True
    assert state["swapped_at"], "the override sits beside the swap it authorised"
    assert state["gate"] is False, "the failed verdict is not rewritten by the override"
    assert state["force_overrode"] == ["gate"]


def test_a_swap_that_needed_no_override_records_no_forced_key(tmp_path, monkeypatch):
    """The negative half, and the reason `save_state` needs `_drop`.

    `save_state` merges, so a `forced` left by an earlier run would survive into
    the next clean swap and relabel a swap that overrode nothing as one that did.
    """
    _world(tmp_path, monkeypatch, gate=True)
    kg_rebuild.save_state(forced=True, force_reason="an earlier run's override",
                          force_overrode=["gate"],
                          force_destroyed={"hand_stated_facts": 9,
                                           "carryworthy_edges": 9})

    assert kg_rebuild.cmd_swap(_args()) == 0

    state = _state()
    assert state["swapped_at"]
    for key in kg_rebuild.OVERRIDE_KEYS:
        assert key not in state, f"{key} survived from a previous run"


# ── clause 2: --force is not enough; it has to say why ────────────────────────

@pytest.mark.parametrize("reason", [None, "", "   "])
def test_force_without_a_reason_refuses_and_moves_nothing(tmp_path, monkeypatch,
                                                          reason):
    """A bare --force is exactly the 2026-09-23 audit gap, so it must not swap.

    Whitespace is no more a reason than nothing is: it would satisfy a
    `if not reason` check while recording nothing readable.
    """
    derived, live, rebuild, _live_db, writes = _world(tmp_path, monkeypatch,
                                                      gate=False)

    rc = kg_rebuild.cmd_swap(_args(force=True, reason=reason))

    assert rc != 0, "an undocumented override has to be refused"
    assert (live / "Old" / "Old-state.md").exists(), "no rename happened"
    assert (rebuild / "New" / "New-state.md").exists()
    assert not list((derived).glob("facts-quarantine-*"))
    assert not list((derived).glob("kg-quarantine-*"))
    assert "forced" not in _state()
    assert writes == [], "a refused swap did not re-enable fact writes"


def test_a_reason_without_force_records_nothing(tmp_path, monkeypatch):
    """The inverse typo: `--reason` alone must not read as a recorded override."""
    _world(tmp_path, monkeypatch, gate=True)

    assert kg_rebuild.cmd_swap(_args(reason=REASON)) == 0

    state = _state()
    assert state["swapped_at"]
    assert "force_reason" not in state, "nothing was overridden to record against"


# ── clause 3: the rationale is the text the operator typed ───────────────────

def test_the_reason_is_stored_verbatim_as_force_reason(tmp_path, monkeypatch):
    """The whole point is that a later reader gets the rationale itself.

    The fixture reason carries a colon, a hyphenated key, a decimal and embedded
    double quotes: a re-quoted or normalised copy would no longer be what the
    person who ran the swap wrote.
    """
    _world(tmp_path, monkeypatch, gate=False)

    assert kg_rebuild.cmd_swap(_args(force=True, reason=REASON)) == 0

    assert _state()["force_reason"] == REASON


# ── clause 4: what the override destroyed is counted, not just announced ──────

def test_a_forced_swap_over_the_carry_over_guard_counts_the_rows_it_destroyed(
        tmp_path, monkeypatch):
    """The gap is wider than the marker: those rows are gone, permanently.

    The guard refuses on 3 hand-stated facts and 2 carry-worthy edges (asserted
    first, rc 3, so the numbers below are the guard's own finding and not the
    fixture's). The forced swap that crosses it records the same counts, and the
    quarantined store still holds those rows — which is what makes "destroyed"
    the honest word.
    """
    _world(tmp_path, monkeypatch, gate=True, stated_facts=3, carry_edges=2)

    assert kg_rebuild.cmd_swap(_args()) == 3, "the guard must refuse on those rows"

    assert kg_rebuild.cmd_swap(_args(force=True, reason=REASON)) == 0

    state = _state()
    assert state["force_overrode"] == ["carry_over"]
    assert state["force_destroyed"] == {"hand_stated_facts": 3, "carryworthy_edges": 2}

    from app.kg_store import KGStore
    st = KGStore(state["quarantine_db"])
    try:
        held = st._query("SELECT COUNT(*) n FROM facts_idx WHERE provenance='STATED'")[0]
        assert held["n"] == 3, "the counted rows are the ones the swap quarantined"
    finally:
        st.close()


def test_a_gate_only_override_counts_nothing_destroyed(tmp_path, monkeypatch):
    """Zero is a real answer here: it says no hand-stated row was lost."""
    _world(tmp_path, monkeypatch, gate=False)

    assert kg_rebuild.cmd_swap(_args(force=True, reason=REASON)) == 0

    assert _state()["force_destroyed"] == {"hand_stated_facts": 0,
                                           "carryworthy_edges": 0}


# ── clause 5: the force branch says out loud what it is doing ─────────────────

def test_a_forced_swap_prints_that_it_overrode_a_refusal_and_recorded_a_reason(
        tmp_path, monkeypatch, capsys):
    """Before this the override path printed nothing at all about overriding.

    One line has to name both the refusal that was crossed and the fact that a
    reason was recorded, and the pre-rename line has to come before the rows are
    destroyed rather than only in the summary. The receipt rides stdout beside
    the `swapped.` line it explains; the pre-flight warning is on stderr with the
    other refusals. Clause 5 accepts either stream, so this reads both.
    """
    _world(tmp_path, monkeypatch, gate=False, stated_facts=1, carry_edges=1)

    assert kg_rebuild.cmd_swap(_args(force=True, reason=REASON)) == 0

    captured = capsys.readouterr()
    printed = captured.out + captured.err
    recorded = [ln for ln in printed.splitlines() if "OVERRIDE RECORDED" in ln]
    assert len(recorded) == 1, f"expected one receipt line, got {recorded}"
    line = recorded[0]
    for fragment in ("overriding", "failed gate verdict", "carry-over guard",
                     "forced: true", REASON):
        assert fragment in line, f"{fragment!r} missing from the receipt"
    assert any(ln.startswith("FORCE: overriding") for ln in captured.err.splitlines()), \
        "the operator is told before the renames, not only in the summary"


def test_a_clean_swap_prints_no_override_receipt(tmp_path, monkeypatch, capsys):
    """A receipt with no override behind it would train a reader to ignore it."""
    _world(tmp_path, monkeypatch, gate=True)

    assert kg_rebuild.cmd_swap(_args()) == 0

    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert "OVERRIDE RECORDED" not in printed
    assert "FORCE:" not in printed


# ── the CLI seam: the flags the clauses name exist on the subcommand ─────────

def test_the_swap_subcommand_takes_force_with_a_reason():
    """The reason has to be reachable from a terminal, not only from Python.

    Before this `--force` was a bare store_true with no channel for a rationale,
    which is why the 2026-09-23 one ended up in a hand-written markdown file.
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "memory" / "kg_rebuild.py"),
         "swap", "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, proc.stderr
    assert "--force" in proc.stdout
    assert "--reason" in proc.stdout
