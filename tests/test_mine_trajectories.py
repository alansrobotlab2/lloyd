"""mine-trajectories.py — the miner's own contract, in its own test file.

The miner was asserted only from other modules' test files
(`test_trajectory_extraction.py`, `test_skill_verdicts.py`). This file holds
what is the miner's alone:

- **#494 — the `--agent` selector.** Its default was `worker`, a value no row
  has ever carried, so the documented default invocation mined nothing. The
  default is `all` now, and every value the selector accepts is one the
  extractor can actually emit.
- **#389 / #511 — the corroboration gate, miner side.** A step whose only
  failure signal is a keyword in its own output reaches no error pattern and
  counts as no failure; `protocol` or a persisted non-zero `exit_code` does.
  Moved here from `test_trajectory_extraction.py`.
- **#511 — the label scheme.** One `normalize_tool_name`, and its labels are
  pinned because they are baked into the verdict ledger's `seq-*` keys.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINER_PATH = ROOT / "scripts" / "mine-trajectories.py"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mt = _load("mine_trajectories_own", "scripts/mine-trajectories.py")
et = _load("extract_trajectories_for_miner", "scripts/extract-trajectories.py")

BUCKET = "2026-09-12.jsonl"


def _row(key, agent_id="lloyd", cls="interactive"):
    return {"session_key": key, "agent_id": agent_id, "session_class": cls,
            "timestamp": "2026-09-12T10:10:10Z", "tool_count": 0,
            "error_count": 0, "has_errors": False, "tools": [],
            "error_tools": [], "signals": []}


def _corpus(tmp_path, rows):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / BUCKET).write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                            encoding="utf-8")
    return d


def _run(tmp_path, corpus, *extra):
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    cmd = [sys.executable, str(MINER_PATH), "--trajectory-dir", str(corpus),
           "--sessions-dir", str(store), "--days", "9999",
           "--output-dir", str(tmp_path / "out"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180)


def _session(tmp_path, stem):
    path = tmp_path / f"{stem}.json"
    path.write_text(json.dumps({
        "session_id": stem, "session_start": "2026-09-12T10:00:00Z",
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "c0", "function": {"name": "Read",
                                          "arguments": json.dumps({"file_path": "/x"})}}]},
            {"role": "tool", "tool_call_id": "c0",
             "content": [{"type": "text", "text": "ok"}]},
        ],
    }), encoding="utf-8")
    return path


# ── #494: the --agent selector ───────────────────────────────────────────────

def test_the_default_invocation_loads_a_non_empty_corpus(tmp_path):
    """Clause 1: no `--agent` at all, over a corpus of what the extractor
    writes today (`agent_id: lloyd` on every row), loads rows."""
    corpus = _corpus(tmp_path, [_row("i1"), _row("i2")])
    proc = _run(tmp_path, corpus, "--stats")
    report = proc.stdout + proc.stderr
    assert proc.returncode == 0, report
    assert "Loaded 2 trajectory(ies)" in report, report


def test_the_default_matches_the_nightly_runbooks_override(tmp_path):
    """Clause 5: the runbook passes `--agent all`; the default must select the
    same corpus, so the override and the default cannot drift apart."""
    assert mt.DEFAULT_AGENT == "all"
    corpus = _corpus(tmp_path, [_row("i1"), _row("i2"), _row("a1", "autonomy")])
    bare = _run(tmp_path, corpus, "--stats")
    runbook = _run(tmp_path, corpus, "--stats", "--agent", "all")
    loaded = [l for p in (bare, runbook) for l in (p.stdout + p.stderr).splitlines()
              if "Loaded" in l]
    assert len(loaded) == 2 and loaded[0] == loaded[1], loaded


def test_each_advertised_selector_returns_its_matching_records(tmp_path, monkeypatch):
    """Clauses 2 and 3: a fixture with an unattended-shaped row (`autonomy`,
    what an `autonomy_*` session file yields) and an interactive one, and each
    selector value the help advertises returns its own records and not the
    other's."""
    traj = tmp_path / "t"
    traj.mkdir()
    (traj / BUCKET).write_text(
        json.dumps(_row("i1")) + "\n" + json.dumps(_row("a1", "autonomy")) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", traj)

    def keys(agent):
        rows = mt.load_trajectories(days=9999, agent_filter=agent,
                                    exclude_machine=False)
        return sorted(r["session_key"] for r in rows)
    assert keys("all") == ["a1", "i1"]
    assert keys("lloyd") == ["i1"]
    assert keys("main") == ["i1"]
    assert keys("autonomy") == ["a1"]
    assert keys("worker") == ["a1"]


def test_the_help_advertises_no_value_that_matches_nothing():
    """Clause 2: `worker` is no longer the advertised default or the advertised
    choice, in `--help` or in either copy of the index's command block."""
    proc = subprocess.run([sys.executable, str(MINER_PATH), "--help"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    agent_help = " ".join(proc.stdout.split("--agent AGENT")[-1].split("--threshold")[0].split())
    assert "default: all" in agent_help and "worker" not in agent_help, agent_help
    for path in (MINER_PATH, ROOT / "scripts" / "rebuild-skill-candidates-index.py"):
        text = path.read_text(encoding="utf-8")
        assert "mine-trajectories.py --days 7 --agent worker" not in text, path


def test_every_accepted_agent_id_is_one_the_extractor_emits(tmp_path):
    """Clause 4: the selector's accepted set is a subset of what
    `parse_session` can write, measured by running it on both filename shapes
    it distinguishes rather than restating its literals."""
    emitted = {et.parse_session(_session(tmp_path, stem))["agent_id"]
               for stem in ("20260912_100000_ab12", "autonomy_20260912_100000")}
    accepted = set().union(*mt.AGENT_SELECTORS.values())
    assert accepted <= emitted, (accepted, emitted)


# ── #389 / #511: the corroboration gate, miner side ──────────────────────────

def error_traj(session_key, name, error_type, source, params=None):
    return {
        "session_key": session_key,
        "timestamp": "2026-09-08T18:00:00Z",
        "tool_count": 1,
        "error_count": 1,
        "has_errors": True,
        "tools": [{"name": name, "is_error": True, "error_source": source,
                   "params_summary": params or {"path": "/x"}, "sequence": 0}],
        "error_tools": [{"name": name, "sequence": 0, "error_type": error_type,
                         "error_source": source,
                         "params_summary": params or {"path": "/x"}}],
        "signals": [],
    }


def write_session(tmp_path, calls, name="sess-corroboration"):
    """A session file in the shape `parse_session` consumes; `calls` is a list
    of (tool_name, arguments, result_text, stats_is_error)."""
    messages = []
    for i, (tool_name, args, result, stats_error) in enumerate(calls):
        messages.append({"role": "assistant", "tool_calls": [
            {"id": f"call_{i}", "function": {"name": tool_name,
                                             "arguments": json.dumps(args)}}]})
        message = {"role": "tool", "tool_call_id": f"call_{i}",
                   "content": [{"type": "text", "text": result}]}
        if stats_error is not None:
            message["stats"] = {"result_chars": len(result), "is_error": stats_error}
        messages.append(message)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"session_id": name,
                                "session_start": "2026-09-08T18:00:00Z",
                                "messages": messages}), encoding="utf-8")
    return path


def test_mining_ignores_a_keyword_only_error():
    """Rows written before #389 carry `error_source: "semantic"`; they must not
    reach skill authoring either."""
    traj = [error_traj(f"s{i}", "Read", "timeout", "semantic") for i in (1, 2)]
    assert mt.mine_error_patterns(traj, threshold=2) == []


def test_mining_keeps_a_corroborated_error():
    traj = [error_traj(f"s{i}", "Bash", "timeout", "protocol") for i in (1, 2)]
    patterns = mt.mine_error_patterns(traj, threshold=2)
    assert len(patterns) == 1
    assert patterns[0]["tool_name"] == "Bash"
    assert patterns[0]["error_type"] == "timeout"


def test_mining_treats_a_persisted_nonzero_exit_code_as_corroborating():
    traj = []
    for i in (1, 2):
        row = error_traj(f"s{i}", "Bash", "logic", None)
        row["error_tools"][0]["exit_code"] = 1
        traj.append(row)
    assert len(mt.mine_error_patterns(traj, threshold=2)) == 1


def test_an_error_candidate_renders_the_persisted_message(tmp_path):
    """#492: the candidate used to show a label and no message, because the
    example dict was built without `result_summary`."""
    message = "ERROR: grep: warning: no match  [exit code: 1]"
    traj = []
    for i in (1, 2):
        row = error_traj(f"s{i}", "Bash", "logic", "exit_code")
        row["error_tools"][0]["exit_code"] = 1
        row["error_tools"][0]["result_summary"] = message
        traj.append(row)
    patterns = mt.mine_error_patterns(traj, threshold=2)
    assert len(patterns) == 1
    assert patterns[0]["examples"][0]["result_summary"] == message
    out = tmp_path / "candidates"
    path = mt.write_candidate_file(patterns[0], out, verdict_store=tmp_path / "v.jsonl")
    assert path is not None
    assert f"- **Message:** {message}" in Path(path).read_text(encoding="utf-8")
    # A row written before the mirror carried the message says so, rather
    # than rendering an empty line.
    legacy = [error_traj(f"l{i}", "Bash", "logic", "exit_code") for i in (1, 2)]
    for row in legacy:
        row["error_tools"][0]["exit_code"] = 1
    lpath = mt.write_candidate_file(mt.mine_error_patterns(legacy, threshold=2)[0], out,
                                    verdict_store=tmp_path / "v.jsonl")
    assert "- **Message:** N/A" in Path(lpath).read_text(encoding="utf-8")


def test_a_read_timeout_candidate_cannot_be_emitted(tmp_path):
    """The concrete phantom from the 09-06 window: `Read` has no timeout path,
    the word came from the file. End to end — extract, then mine."""
    out = []
    for i in (1, 2):
        path = write_session(
            tmp_path,
            [("Read", {"file_path": f"/x/{i}.py"}, "request timed out after 30s", False)],
            name=f"read-timeout-{i}",
        )
        out.append(et.parse_session(path))
    assert mt.mine_error_patterns(out, threshold=2) == []


def test_success_mining_does_not_count_a_keyword_only_step_as_a_failure(tmp_path):
    calls = [("Bash", {"command": "pytest"}, "Error: 0 warnings\n755 passed\n", False)]
    new = [et.parse_session(write_session(tmp_path, calls, name=f"ok{i}")) for i in (1, 2)]
    legacy = [error_traj(f"legacy{i}", "Bash", "validation", "semantic") for i in (1, 2)]
    for rows in (new, legacy):
        patterns = mt.mine_success_patterns(rows, threshold=2)
        assert len(patterns) == 1
        assert patterns[0]["error_count"] == 0
        assert patterns[0]["error_rate"] == 0.0


# ── #511: one label function, and its labels pinned ─────────────────────────

def test_exactly_one_normalize_tool_name_and_the_dead_tables_are_gone():
    src = MINER_PATH.read_text(encoding="utf-8")
    assert src.count("\ndef normalize_tool_name(") == 1
    for dead in ("BASH_CMD_CATEGORIES", "MCP_PREFIX_MAP", "BUILTIN_TOOLS"):
        assert not hasattr(mt, dead), dead
    for live in ("_BASH_CMD_CATEGORIES", "_MCP_PREFIX_MAP", "_SIMPLE_TOOL_MAP"):
        assert hasattr(mt, live), live


def test_the_ledger_label_scheme_is_pinned():
    """Changing any of these re-keys the verdict ledger's `seq-*` rows."""
    n = mt.normalize_tool_name
    assert n({"name": "Bash", "params_summary": {"command": "ls -la"}}) == "bash:explore"
    assert n({"name": "mcp____discord_send"}) == "mcp:other"
    assert n({"name": "mcp____vault_search"}) == "mcp:vault"
    assert n({"name": "Read"}) == "read"
    assert n({"name": "Edit", "is_error": True}) == "edit:ERR"
    assert n({"name": "NotebookEdit"}) == "notebookedit"
    seq = " → ".join([n({"name": "Edit", "is_error": True}), n({"name": "Read"})])
    assert mt.sequence_pattern_key(2, seq) == "seq-2-edit-err-read"


def test_the_miner_is_pyflakes_clean():
    proc = subprocess.run([sys.executable, "-m", "pyflakes", str(MINER_PATH)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and not proc.stdout.strip(), proc.stdout + proc.stderr
