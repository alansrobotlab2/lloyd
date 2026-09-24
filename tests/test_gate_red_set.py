"""The per-base red set: a base probe is asked once per base, not once per round.

Since 2026-09-24 the `tests` rung passes on failures that reproduce at the
round's base. Deciding that costs a throwaway worktree and a pytest run of the
failing files at the base — and during a red episode every round on that base
asks the same question and gets the same answer. `state.red_set.json` keeps
the answer per exact base sha, for `automod.gate.red_set_max_age_s`.

The set only ever applies to nodes that failed at HEAD, so a stale or wrong
entry can skip a probe for a node it names; it can never hide a failure.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.automod import gate as G
from scripts.automod import state as S

RED = "tests/test_uptake.py::test_red"
OTHER = "tests/test_other.py::test_also_red"


@pytest.fixture(autouse=True)
def private_red_set(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RED_SET_PATH", tmp_path / "red_set.json")
    return tmp_path / "red_set.json"


def _gate(tmp_path, monkeypatch, base, *, failing, probe=None):
    """A Gate whose suite fails on `failing` and whose base probe is a spy."""
    monkeypatch.setattr(G.S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(G, "PYTEST_MIN_COLLECTED", 1)
    monkeypatch.setattr(G, "PYTEST_MIN_PASSED", 1)
    wt = tmp_path / "wt"
    (wt / "tests").mkdir(parents=True, exist_ok=True)
    g = G.Gate("SM_RS", wt, base, live_root=tmp_path)
    g.python = Path(sys.executable)
    monkeypatch.setattr(g, "_child_env", lambda root=None, **kw: {})
    monkeypatch.setattr(g, "_touched_paths", lambda: set())
    text = "".join(f"FAILED {n} - assert False\n" for n in failing) + \
        f"{len(failing)} failed, 40 passed in 1s\n"
    monkeypatch.setattr(g, "_run_suite", lambda only: (
        subprocess.CompletedProcess([], 1, text, ""), text, G._parse_pytest_summary(text)))
    monkeypatch.setattr(G, "_reconfirm_candidate_failures",
                        lambda *a, **k: ([], "failed every repeat"))
    probed: list[list[str]] = []

    def at_base(python, live, b, node_ids, scratch, env=None):
        probed.append(list(node_ids))
        if probe is not None:
            return probe(node_ids)
        return set(node_ids), f"probed 1 file(s) at base {b[:8]}: {len(node_ids)} already failing"
    monkeypatch.setattr(G, "_failures_at_base", at_base)
    return g, probed


def test_the_same_base_and_the_same_failure_is_probed_once(tmp_path, monkeypatch):
    base = "a" * 40
    g, probed = _gate(tmp_path, monkeypatch, base, failing=[RED])
    ok, _detail, data = g.rung_tests()
    assert ok is True and probed == [[RED]] and data["red_set_cached"] is False
    g2, probed2 = _gate(tmp_path, monkeypatch, base, failing=[RED])
    ok, detail, data = g2.rung_tests()
    assert ok is True and probed2 == [], "the second round on that base re-probed"
    assert data["red_set_cached"] is True and "red set cached" in data["base_probe"]
    assert data["pre_existing_failures"] == [RED]


def test_a_different_base_is_probed_again(tmp_path, monkeypatch):
    g, _ = _gate(tmp_path, monkeypatch, "a" * 40, failing=[RED])
    g.rung_tests()
    g2, probed = _gate(tmp_path, monkeypatch, "b" * 40, failing=[RED])
    g2.rung_tests()
    assert probed == [[RED]], "a rebase is a new base, and a new entry"


def test_a_node_the_set_does_not_name_is_probed(tmp_path, monkeypatch):
    base = "a" * 40
    g, _ = _gate(tmp_path, monkeypatch, base, failing=[RED])
    g.rung_tests()
    g2, probed = _gate(tmp_path, monkeypatch, base, failing=[RED, OTHER])
    g2.rung_tests()
    assert probed == [[RED, OTHER]], "a failure outside the cached set must be asked"
    assert set(S.read_red_set(base)["nodes"]) == {RED, OTHER}, "a conclusive probe merges in"


def test_a_stale_entry_is_probed_again(tmp_path, monkeypatch):
    base = "a" * 40
    S.write_red_set(base, [RED], by="SM_OLD")
    d = S.read_json(S.RED_SET_PATH)
    d["bases"][base]["ts"] = time.time() - G.RED_SET_MAX_AGE_S - 60
    S.write_json(S.RED_SET_PATH, d)
    g, probed = _gate(tmp_path, monkeypatch, base, failing=[RED])
    g.rung_tests()
    assert probed == [[RED]]


def test_an_inconclusive_probe_writes_nothing(tmp_path, monkeypatch):
    base = "a" * 40
    g, _ = _gate(tmp_path, monkeypatch, base, failing=[RED],
                 probe=lambda ids: (set(), "baseline probe INCONCLUSIVE at base aaaaaaaa — no summary"))
    ok, _detail, _data = g.rung_tests()
    assert ok is False
    assert S.read_red_set(base) is None


def test_a_full_green_run_records_an_empty_set_for_its_base(tmp_path, monkeypatch):
    base = "a" * 40
    S.write_red_set(base, [RED], by="SM_OLD")
    g, _ = _gate(tmp_path, monkeypatch, base, failing=[])
    text = "45 passed in 1s\n"
    monkeypatch.setattr(g, "_run_suite", lambda only: (
        subprocess.CompletedProcess([], 0, text, ""), text, G._parse_pytest_summary(text)))
    ok, _detail, _data = g.rung_tests()
    assert ok is True
    assert S.read_red_set(base)["nodes"] == [], "a green run replaces, it does not merge"


def test_the_set_keeps_the_newest_eight_bases():
    for n in range(S.RED_SET_KEEP + 3):
        S.write_red_set(f"{n:040d}", [RED], by=f"SM_{n}")
        time.sleep(0.002)
    kept = set((S.read_json(S.RED_SET_PATH) or {})["bases"])
    assert len(kept) == S.RED_SET_KEEP
    assert f"{0:040d}" not in kept and f"{S.RED_SET_KEEP + 2:040d}" in kept


def test_merge_unions_and_replace_does_not():
    base = "c" * 40
    S.write_red_set(base, [RED], by="x")
    S.write_red_set(base, [OTHER], by="y")
    assert set(S.read_red_set(base)["nodes"]) == {RED, OTHER}
    S.write_red_set(base, [], by="z", merge=False)
    assert S.read_red_set(base)["nodes"] == []
    assert S.read_red_set(base, max_age_s=-1) is None, "a max age is honoured"
