"""#1246 clause 4: apply-classifications-v4 does not re-type a pair whose
verdict is already live.

The extractor used to mint a `mentions` row beside an active classifier-typed
edge on the same (source, target) (the guard is `_index_and_link`'s, pinned in
`tests/test_fact_extractor.py`), and that row put the pair back through
`retype`: the live `uses` was expired and a `uses` inserted again, 3-29% of
each nightly apply's output. Such a record is now counted as `already_typed`
and gets no retype in either mode. Under `--apply` the redundant `mentions`
row is retired on its own, so the pair converges to the one active relation
`retype` guarantees instead of being counted again on every run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.kg_store import KGStore  # noqa: E402


def _load():
    path = ROOT / "scripts" / "memory" / "apply-classifications-v4.py"
    spec = importlib.util.spec_from_file_location("apply_classifications_v4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


apply_mod = _load()


@pytest.fixture
def store(tmp_path):
    """A tmp store holding one pair the classifier already typed (with the
    extractor's redundant mentions row beside it) and one bare mentions pair."""
    db = tmp_path / "kg.sqlite"
    st = KGStore(db)
    typed_id = st.edges.add({"source": "Lloyd", "target": "vLLM", "type": "uses",
                             "confidence": 0.9, "provenance": "EXTRACTED_CLASSIFIER_V4"},
                            origin="classifier")
    redundant_id = st.edges.add({"source": "Lloyd", "target": "vLLM", "type": "mentions",
                                 "confidence": 0.8, "provenance": "EXTRACTED"},
                                origin="extractor")
    bare_id = st.edges.add({"source": "Lloyd", "target": "Isaac Lab", "type": "mentions",
                            "confidence": 0.8, "provenance": "EXTRACTED"},
                           origin="extractor")
    st.close()
    classified = tmp_path / "classified"
    classified.mkdir()
    rows = [
        {"source": "Lloyd", "target": "vLLM", "new_type": "uses", "confidence": 0.95,
         "classified_at": "2026-09-24T00:00:00+00:00", "reason": "already live"},
        {"source": "Lloyd", "target": "Isaac Lab", "new_type": "uses", "confidence": 0.95,
         "classified_at": "2026-09-24T00:00:00+00:00", "reason": "real upgrade"},
    ]
    (classified / "classified-v4-batch.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return {"db": db, "classified": classified, "typed_id": typed_id,
            "redundant_id": redundant_id, "bare_id": bare_id}


def _run(monkeypatch, store, *flags):
    monkeypatch.setattr(sys, "argv", ["apply-classifications-v4.py",
                                      "--classified-dir", str(store["classified"]),
                                      "--db", str(store["db"]), *flags])
    return apply_mod.main()


def _stat(out, name):
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == name:
            return int(parts[1])
    raise AssertionError(f"no `{name}` row in the plan:\n{out}")


def test_dry_run_reports_the_already_typed_pair_and_writes_nothing(
        monkeypatch, store, capsys):
    assert _run(monkeypatch, store, "--dry-run") == 0
    out = capsys.readouterr().out
    assert _stat(out, "already_typed") == 1, out
    assert _stat(out, "upgrades") == 1, "the bare pair is still a real upgrade"

    st = KGStore(store["db"])
    on_pair = {(e["id"], e["type"]) for e in st.edges.active(source="Lloyd", target="vLLM")}
    assert on_pair == {(store["typed_id"], "uses"), (store["redundant_id"], "mentions")}, (
        "a dry run wrote something")
    assert st.edges.by_id(store["bare_id"])["expired_at"] is None
    st.close()


def test_apply_keeps_the_live_verdict_and_retires_only_the_redundant_row(
        monkeypatch, store, capsys):
    """Against the pre-fix tree the typed `uses` is expired with reason
    `... pair re-typed as uses` and a new `uses` row takes its id's place —
    the churn the item measured. Now the typed edge is untouched, the
    redundant mentions row is retired with a reason that says why, and the
    bare pair is upgraded as before."""
    assert _run(monkeypatch, store, "--apply") == 0
    out = capsys.readouterr().out
    assert _stat(out, "already_typed") == 1, out

    st = KGStore(store["db"])
    typed = st.edges.by_id(store["typed_id"])
    assert typed["expired_at"] is None and typed["type"] == "uses"
    on_pair = st.edges.active(source="Lloyd", target="vLLM")
    assert [e["id"] for e in on_pair] == [store["typed_id"]], on_pair
    redundant = st.edges.by_id(store["redundant_id"])
    assert redundant["expired_at"] is not None
    assert "already typed as uses" in redundant["expired_reason"], redundant
    assert "re-typed" not in (typed.get("expired_reason") or "")

    bare = st.edges.by_id(store["bare_id"])
    assert bare["expired_at"] is not None, "the bare pair still gets its upgrade"
    upgraded = st.edges.active(source="Lloyd", target="Isaac Lab")
    assert [e["type"] for e in upgraded] == ["uses"]
    assert upgraded[0]["superseded_edge_id"] == store["bare_id"]
    st.close()
