"""The two commands that close out the plan's remaining manual steps.

`sync_tool_overrides` turns "remember to hand-edit a gitignored file after
merging" into one idempotent command. `measure_parallel_dispatch` turns "soak
it and see" into a number — and the number has to be taken BEFORE the flag is
flipped, because once batches run concurrently their per-iteration duration no
longer describes the sequential baseline.

The measurement tests are mostly about the two ways to count it wrong, both of
which produce a confident and completely false answer.
"""

from __future__ import annotations

import json

import pytest
import yaml

from eval.measure_parallel_dispatch import iterations, measure, qualifies
from scripts.maintenance.sync_tool_overrides import drift, sync


# ── sync_tool_overrides ─────────────────────────────────────────────────────

def _tree(tmp_path, tracked_baseline, served_baseline=None, extra=None):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "config.yaml").write_text(yaml.dump(
        {"harness": {"tool_search": {"enabled": False, "threshold_tools": 30,
                                     "baseline_tools": tracked_baseline}}}))
    if served_baseline is not None:
        payload = {"harness": {"tool_search": {
            "enabled": False, "threshold_tools": 30,
            "baseline_tools": served_baseline,
            "max_results_default": 5}}}
        payload.update(extra or {})
        (tmp_path / "data" / "tool_overrides.yaml").write_text(yaml.dump(payload))
    return tmp_path


def test_it_reports_the_added_names_and_syncs(tmp_path, capsys):
    root = _tree(tmp_path, ["Read", "Edit", "graph_explain"], ["Read", "Edit"])
    assert drift(root)[2]
    assert sync(root) == 0
    out = capsys.readouterr().out
    assert "+ graph_explain" in out
    assert drift(root)[2] == {}, "still drifting after a sync"


def test_it_is_idempotent(tmp_path):
    root = _tree(tmp_path, ["Read", "graph_explain"], ["Read"])
    sync(root)
    assert sync(root) == 0
    assert drift(root)[2] == {}


def test_check_reports_without_writing(tmp_path):
    root = _tree(tmp_path, ["Read", "graph_explain"], ["Read"])
    before = (root / "data" / "tool_overrides.yaml").read_text()
    assert sync(root, check=True) == 1, "--check must exit non-zero on drift"
    assert (root / "data" / "tool_overrides.yaml").read_text() == before


def test_check_is_quiet_and_zero_when_in_sync(tmp_path):
    root = _tree(tmp_path, ["Read"], ["Read"])
    assert sync(root, check=True) == 0


def test_every_other_key_in_the_override_survives(tmp_path):
    """The file also holds the UI's tool disables and the worker switch."""
    root = _tree(tmp_path, ["Read", "graph_explain"], ["Read"],
                 extra={"mcp_servers": {"lloyd-mcp": {"disabled_tools": ["discord_send"]}},
                        "workers": {"enabled": True}})
    sync(root)
    after = yaml.safe_load((root / "data" / "tool_overrides.yaml").read_text())
    assert after["mcp_servers"]["lloyd-mcp"]["disabled_tools"] == ["discord_send"]
    assert after["workers"]["enabled"] is True


def test_a_missing_override_is_not_an_error(tmp_path):
    """A fresh clone has none; config.yaml is the served state there."""
    root = _tree(tmp_path, ["Read"])
    assert sync(root) == 0
    assert not (root / "data" / "tool_overrides.yaml").exists()


def test_it_only_compares_keys_the_override_actually_shadows(tmp_path):
    """A key present only in config.yaml is not drift — the merge keeps it."""
    root = _tree(tmp_path, ["Read"], ["Read"])
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    cfg["harness"]["tool_search"]["max_results_cap"] = 20
    (root / "config.yaml").write_text(yaml.dump(cfg))
    assert drift(root)[2] == {}


def test_the_live_tree_is_in_sync_right_now():
    """Guards the merge step: if this fails in live, run the script."""
    from app.paths import LLOYD_HOME
    if not (LLOYD_HOME / "data" / "tool_overrides.yaml").exists():
        pytest.skip("no override file (a worktree or a fresh clone)")
    assert drift(LLOYD_HOME)[2] == {}, \
        "run `python -m scripts.maintenance.sync_tool_overrides`"


# ── measure_parallel_dispatch ───────────────────────────────────────────────

def _asst(name, iteration, ms=100):
    return {"role": "assistant",
            "tool_calls": [{"function": {"name": name, "arguments": "{}"}}],
            "stats": {"iteration": iteration, "duration_ms": ms}}


def _tool():
    return {"role": "tool", "content": [{"type": "text", "text": "r"}]}


def test_a_multi_call_iteration_is_reassembled_from_its_per_pair_rows():
    """`messages.py` writes one assistant message PER CALL. Counting
    len(tool_calls) reports zero multi-call iterations across the whole
    corpus — an artifact that reads exactly like a finding."""
    session = {"messages": [
        _asst("Read", 1), _tool(), _asst("Grep", 1), _tool(),
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]}
    got = iterations(session)
    assert len(got) == 1 and [c["name"] for c in got[0]] == ["Read", "Grep"]


def test_two_consecutive_single_call_turns_are_not_one_batch():
    """Both turns start at iteration 1; grouping on the number alone merges
    them into a fake batch of two."""
    session = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        _asst("Read", 1), _tool(),
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
        _asst("Read", 1), _tool(),
    ]}
    got = iterations(session)
    assert [len(g) for g in got] == [1, 1], got


def test_adjacent_iterations_within_one_turn_are_split():
    session = {"messages": [
        _asst("Read", 1), _tool(), _asst("Grep", 2), _tool(),
    ]}
    assert [len(g) for g in iterations(session)] == [1, 1]


def test_qualification_matches_the_loops_rule():
    assert qualifies([{"name": "Read"}, {"name": "Grep"}])
    assert qualifies([{"name": "Read"}, {"name": "ToolSearch"}])
    assert not qualifies([{"name": "Read"}, {"name": "Bash"}])
    assert not qualifies([{"name": "Edit"}, {"name": "Write"}])


def test_measure_counts_and_attributes_blockers(tmp_path):
    (tmp_path / "s1.json").write_text(json.dumps({"messages": [
        _asst("Read", 1, 500), _tool(), _asst("Grep", 1, 500), _tool(),
        {"role": "user", "content": []},
        _asst("Read", 1, 900), _tool(), _asst("Bash", 1, 900), _tool(),
        {"role": "user", "content": []},
        _asst("Read", 1, 100), _tool(),
    ]}))
    r = measure(tmp_path)
    assert r["iterations_with_tool_calls"] == 3
    assert r["multi_call"] == 2
    assert r["qualifying"] == 1 and r["mixed"] == 1
    assert r["qualifying_duration_ms"]["median"] == 500
    assert dict(r["top_blockers"])["Bash"] == 1


def test_an_unreadable_session_is_skipped_not_fatal(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "ok.json").write_text(json.dumps({"messages": [_asst("Read", 1)]}))
    assert measure(tmp_path)["iterations_with_tool_calls"] == 1
