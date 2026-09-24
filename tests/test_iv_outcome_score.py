"""#833 — Inner Voice interventions scored against turn outcomes from the event log.

A throwaway fixture root holds a `usage.db` with the real table schema and an
`event_logs/` directory; the shipped script is run against it. Pinned: the join
on (session_id, turn_id) and the per-class line (1), the label coming from the
event log and never from an IV column (2), guard injects as their own class (3),
counts without rates under 30 scored rows (4), and a read-only run (5).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import iv_outcome_score as S  # noqa: E402

_SCHEMA = """CREATE TABLE inner_voice_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
    sequence_in_turn INTEGER NOT NULL, trigger TEXT NOT NULL,
    action TEXT NOT NULL, reason TEXT, content TEXT, related_tool TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER,
    cache_create INTEGER, latency_ms INTEGER, model TEXT, error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, safeguard TEXT)"""


def _root(tmp_path, rows, turns, blobs=None):
    """rows: (sid, tid, action, reason, safeguard, created_at);
    turns: {(sid, tid): (stop_reason|None, [tool results])}."""
    db = tmp_path / "usage.db"
    with sqlite3.connect(db) as c:
        c.execute(_SCHEMA)
        for i, (sid, tid, action, reason, guard, at) in enumerate(rows):
            c.execute("INSERT INTO inner_voice_observations (session_id, turn_id, "
                      "sequence_in_turn, trigger, action, reason, safeguard, created_at) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (sid, tid, i, "assistant_message", action, reason, guard, at))
    logs = tmp_path / "event_logs"
    (logs / "blobs").mkdir(parents=True)
    for sha, text in (blobs or {}).items():
        (logs / "blobs" / f"{sha}.txt").write_text(text)
    by_session: dict[str, list[str]] = {}
    for (sid, tid), (stop, results) in turns.items():
        lines = by_session.setdefault(sid, [])
        for res in results:
            lines.append(json.dumps({"session_id": sid, "turn_id": tid,
                                     "event": "brain1.tool_result_received",
                                     "data": {"tool_call_id": "x", "result": res}}))
        if stop is not None:
            lines.append(json.dumps({"session_id": sid, "turn_id": tid,
                                     "event": "brain1.result_message",
                                     "data": {"stop_reason": stop}}))
    for sid, lines in by_session.items():
        (logs / f"{sid}.events.jsonl").write_text("\n".join(lines) + "\n")
    return db, logs


def _run(capsys, db, logs, *extra):
    assert S.main(["--db", str(db), "--event-logs", str(logs), "--json", *extra]) == 0
    return json.loads(capsys.readouterr().out)


T0 = "2026-09-20T10:00:00.000000"


def test_join_on_session_and_turn_prints_the_per_class_line(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "keep going", None, T0),
            ("s1", "t2", "inject", "keep going", None, T0),
            ("s1", "t3", "noop", "fine", None, T0),
            ("s2", "t1", "noop", "fine", None, T0)]
    turns = {("s1", "t1"): ("max_turns", []), ("s1", "t2"): ("stop", []),
             ("s1", "t3"): ("stop", []), ("s2", "t1"): ("max_turns", [])}
    db, logs = _root(tmp_path, rows, turns)
    rep = _run(capsys, db, logs)
    inj = rep["classes"]["inject[model]"]
    assert (inj["n"], inj["tp"], inj["fp"]) == (2, 1, 1)
    # s2/t1 ended badly with no intervention — the FN pool; the same turn id in
    # another session must not borrow s1/t1's inject.
    assert rep["bad_turns"] == 2 and rep["fn_pool"] == 1

    assert S.main(["--db", str(db), "--event-logs", str(logs)]) == 0
    text = capsys.readouterr().out
    for word in ("n", "TP", "FP", "precision", "recall", "FN pool"):
        assert word in text
    assert "inject[model]" in text


def test_since_until_bound_the_window(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "a", None, "2026-09-19T23:00:00.0"),
            ("s1", "t2", "inject", "b", None, "2026-09-20T12:00:00.0"),
            ("s1", "t3", "inject", "c", None, "2026-09-21T01:00:00.0")]
    turns = {k: ("stop", []) for k in (("s1", "t1"), ("s1", "t2"), ("s1", "t3"))}
    db, logs = _root(tmp_path, rows, turns)
    rep = _run(capsys, db, logs, "--since", "2026-09-20 00:00:00", "--until", "2026-09-21T00:00:00")
    assert rep["classes"]["inject[model]"]["n"] == 1


def test_outcome_comes_from_the_event_log_not_the_iv_reason(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "task succeeded, all tests pass", None, T0)]
    db, logs = _root(tmp_path, rows, {("s1", "t1"): ("max_turns", [])})
    rep = _run(capsys, db, logs)
    assert rep["classes"]["inject[model]"]["tp"] == 1
    assert rep["classes"]["inject[model]"]["fp"] == 0


def test_tool_error_label_reads_the_result_prefix_including_blobs(tmp_path, capsys):
    sha = "ab" * 32
    rows = [("s1", "t1", "inject", "x", None, T0), ("s1", "t2", "inject", "x", None, T0)]
    turns = {("s1", "t1"): ("stop", [{"$blob": sha, "size": 9000}]),
             ("s1", "t2"): ("stop", ["ok", "  fine"])}
    db, logs = _root(tmp_path, rows, turns, blobs={sha: "Traceback (most recent call last):\n"})
    rep = _run(capsys, db, logs, "--label", "tool_error")
    assert rep["label"] == "tool_error"
    assert (rep["classes"]["inject[model]"]["tp"], rep["classes"]["inject[model]"]["fp"]) == (1, 1)


def test_a_turn_with_no_result_message_is_unlabelled(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "x", None, T0)]
    db, logs = _root(tmp_path, rows, {("s1", "t1"): (None, ["ok"])})
    rep = _run(capsys, db, logs)
    c = rep["classes"]["inject[model]"]
    assert c["n"] == 0 and c["unlabelled"] == 1 and rep["turns_unlabelled"] == 1


def test_guard_injects_are_their_own_class(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "deterministic: 3 near-identical Read calls", None, T0),
            ("s1", "t2", "inject", "you skipped the test run", None, T0),
            ("s1", "t3", "inject", "stall", "stall_rescue", T0)]
    turns = {k: ("stop", []) for k in (("s1", "t1"), ("s1", "t2"), ("s1", "t3"))}
    db, logs = _root(tmp_path, rows, turns)
    rep = _run(capsys, db, logs)
    assert rep["classes"]["inject[guard]"]["n"] == 2   # prose fallback + safeguard key
    assert rep["classes"]["inject[model]"]["n"] == 1


def test_a_small_class_prints_counts_and_no_rate(tmp_path, capsys):
    rows = [("s1", f"c{i}", "cancel", "stop", None, T0) for i in range(3)]
    rows += [("s1", f"i{i}", "inject", "go", None, T0) for i in range(30)]
    turns = {("s1", f"c{i}"): ("cancelled", []) for i in range(3)}
    turns.update({("s1", f"i{i}"): ("max_turns" if i < 6 else "stop", []) for i in range(30)})
    db, logs = _root(tmp_path, rows, turns)
    rep = _run(capsys, db, logs)
    cancel, inj = rep["classes"]["cancel"], rep["classes"]["inject[model]"]
    assert (cancel["n"], cancel["tp"]) == (3, 3)
    assert cancel["precision"] is None and cancel["recall"] is None and cancel["lift"] is None
    assert inj["precision"] == pytest.approx(0.2) and inj["recall"] == pytest.approx(6 / 9, abs=1e-3)

    assert S.main(["--db", str(db), "--event-logs", str(logs)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.strip().startswith("cancel"))
    assert "—" in line and "1.000" not in line


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_read_only_run_leaves_the_fixture_root_byte_identical(tmp_path):
    rows = [("s1", "t1", "inject", "x", None, T0), ("s1", "t1", "cancel", "x", None, T0)]
    db, logs = _root(tmp_path, rows, {("s1", "t1"): ("max_turns", ["Error: nope"])})
    before = _tree(tmp_path)
    for extra in ([], ["--json"], ["--label", "tool_error"]):
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "iv_outcome_score.py"),
                               "--db", str(db), "--event-logs", str(logs), *extra],
                              capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr
    assert _tree(tmp_path) == before
    assert "mode=ro" in (ROOT / "scripts" / "iv_outcome_score.py").read_text()
