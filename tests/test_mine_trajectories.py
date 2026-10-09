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
- **#2045 — the window denominator.** `window` carries the newest dated bucket's
  completeness, and the miner prints it, so a run over a bucket holding only the
  first hour of its day cannot report a full window.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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


# ── #2045: the window denominator names the newest bucket's completeness ──────

PRIOR_DAY = "2026-09-11"
STUB_DAY = "2026-09-12"
STALE_DAY = "2026-09-10"
#: The bucket #57 mined on 2026-10-02, committed at `backlog/data/2026-10-01.jsonl`
#: in the vault. The date and the newest-row stamp are facts about those bytes, not
#: numbers a fixture chose: `wc -l` on them prints the 27 the item quotes.
WITNESS_DAY = "2026-10-01"
WITNESS_ROWS = 27
WITNESS_NEWEST_ROW = "2026-10-01T01:09:19.486248-07:00"
# The stub #2449 is about, from `~/lloyd-data/_pipeline/trajectories/2026-10-08.jsonl`:
# 9 rows, newest `2026-10-08T01:00:33.512979-07:00`, written by #56's 01:00:49-local
# pass for the day that pass ran IN. Only the newest row is reproduced here — the flag
# reads one row and the row count, and this is a flag fixture, not a copy.
WITNESS_OPEN_DAY = "2026-10-08"
WITNESS_OPEN_NEWEST = "01:00:33.512979"      # -07:00 local, PDT
WITNESS_ZONE = "America/Los_Angeles"
SKILL_PATH = ("skills", "trajectory-skill-mining", "SKILL.md")
WITNESS_PATH = ("backlog", "data", "2026-10-01.jsonl")


@contextlib.contextmanager
def _frozen_zone(zone: str = "UTC"):
    """Freeze this process's local zone to `zone`, then restore it.

    A bucket's completeness is measured against the close of the LOCAL day the
    bucket is named for — the same zone `extract-trajectories.py` buckets rows into
    (#1154) — so leaving the host zone in play would put the boundary an hour or
    eight hours away depending on where the suite runs, and a test could agree with
    the implementation by luck. `os.environ` is set too, so a miner run as a child
    process by `_run` inherits the same frozen zone.
    """
    saved = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved
        time.tzset()


def _past_day(offset: int) -> str:
    """The local date `offset` days before now, spelled as a bucket name.

    The two tests that run the miner as a subprocess cannot inject a run instant
    (`main()` passes none), so their buckets are dated FROM the clock rather than
    as fixed dates: a fixed `2026-09-12` would make the verdict depend on the
    suite's clock being later than 2026-09-13, and a wrong clock would then read as
    a broken miner. Several days back, the bucket's own close is always in the
    past, which is the case those tests are about.
    """
    return (datetime.now(timezone.utc) - timedelta(days=offset)).strftime("%Y-%m-%d")


def _row_at(key, day, hhmmss, offset="+00:00"):
    row = _row(key)
    row["timestamp"] = f"{day}T{hhmmss}{offset}"
    return row


def _multi_bucket_corpus(tmp_path, buckets):
    """A corpus directory holding one dated bucket per key of `buckets`."""
    d = tmp_path / "multi"
    d.mkdir(exist_ok=True)
    for day, rows in buckets.items():
        (d / f"{day}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return d


def _load_window(tmp_path, buckets, now_iso, monkeypatch, zone="UTC"):
    """`window` as `load_trajectories` fills it, with the run instant fixed — the
    second operand of `earlier of (that bucket's local-day close, the run instant)`
    has to be a stated instant, not whatever the clock said when the suite ran."""
    monkeypatch.setattr(mt, "TRAJECTORY_DIR",
                        _multi_bucket_corpus(tmp_path, buckets))
    window: dict = {}
    with _frozen_zone(zone):
        mt.load_trajectories(days=9999, agent_filter="all", window=window,
                             now=datetime.fromisoformat(now_iso))
    return window


def test_a_newest_bucket_holding_only_the_front_of_its_day_is_marked_partial(tmp_path, monkeypatch):
    """Clause 1, first half — the live shape of #2045. On 2026-10-02 the newest
    bucket `2026-10-01.jsonl` held 27 rows ending 01:09 local while
    `2026-09-30.jsonl` held 382 across the full day (the committed witness,
    `wc -l < backlog/data/2026-10-01.jsonl` = 27), and the run still printed
    `2768 rows in 7 daily files (no missing day)`. The bucket count cannot see that;
    the completeness flag must."""
    window = _load_window(tmp_path, {
        PRIOR_DAY: [_row_at("p1", PRIOR_DAY, "09:00:00"),
                    _row_at("p2", PRIOR_DAY, "23:40:00")],
        STUB_DAY: [_row_at("s1", STUB_DAY, "00:03:13"),
                   _row_at("s2", STUB_DAY, "01:09:19")],
    }, "2026-09-12T23:00:00+00:00", monkeypatch)
    assert window["files"] == 2 and window["rows"] == 4, window
    newest = window["newest"]
    assert newest["date"] == STUB_DAY, newest
    assert newest["rows"] == 2, newest
    assert newest["partial"] is True, newest


def test_a_finalised_prior_bucket_is_not_marked_partial(tmp_path, monkeypatch):
    """Clause 1, second half — the corpus #56 leaves once it has finalised a day:
    last row 22:10, that local day closed 1h50m before the run instant, well inside
    the two-hour bound, so the honest answer is `not partial`."""
    window = _load_window(tmp_path, {
        PRIOR_DAY: [_row_at("p1", PRIOR_DAY, "09:00:00"),
                    _row_at("p2", PRIOR_DAY, "22:10:00")],
    }, "2026-09-13T01:00:00+00:00", monkeypatch)
    newest = window["newest"]
    assert newest["date"] == PRIOR_DAY and newest["rows"] == 2, newest
    assert newest["partial"] is False, newest


def test_the_partial_bound_is_more_than_two_hours(tmp_path, monkeypatch):
    """Clause 1's number, pinned from both sides: `exceeds two hours`, so one second
    inside the bound is complete and one second past it is partial."""
    for last_row, expected in (("22:00:00", False), ("21:59:59", True)):
        window = _load_window(
            tmp_path, {PRIOR_DAY: [_row_at("p1", PRIOR_DAY, last_row)]},
            "2026-09-13T01:00:00+00:00", monkeypatch)
        assert window["newest"]["partial"] is expected, (last_row, window)


def test_a_run_earlier_than_the_days_close_measures_the_gap_to_the_run_instant(tmp_path, monkeypatch):
    """Amended by #2449; the `min` half of this node's claim still holds, the flag
    half did not. Both buckets here are dated STUB_DAY and the run instant is local
    15:50 on that SAME day, so `min(local_day_close(day), run_instant)` takes
    `run_instant` in both cases — which is still exactly right for the number: the
    day's close is 8 h in the future and measuring against it would report a bucket
    whose newest row is 1h50m old as "9.2 h of its day uncovered".

    What did not hold is the verdict on the first case. A bucket dated the run's own
    local day is a day still being written — #56 finalises D-1 and creates the day-D
    stub in one 01:0x-local pass — so a 1h50m gap cannot mean the day is covered, and
    `complete` there was the flag #2045's owed-check ruling of 2026-10-02T09:11Z named
    as "what is actually broken … so fix the measure". `active` is therefore True
    now, with its hours still 1.8: the number measures the run instant, the flag
    measures the day.
    """
    active = _load_window(
        tmp_path, {STUB_DAY: [_row_at("s1", STUB_DAY, "21:00:00")]},
        "2026-09-12T22:50:00+00:00", monkeypatch)
    assert active["newest"]["partial"] is True, active
    assert active["newest"]["uncovered_hours"] == 1.8, (
        "the hours figure stopped being the gap to the run instant, which is the half "
        "of this node's original claim that was correct: "
        f"{active['newest']}")
    quiet = _load_window(
        tmp_path, {STUB_DAY: [_row_at("s1", STUB_DAY, "20:00:00")]},
        "2026-09-12T22:50:00+00:00", monkeypatch)
    assert quiet["newest"]["partial"] is True, quiet
    assert quiet["newest"]["uncovered_hours"] == 2.8, quiet


def test_the_run_own_day_bucket_is_never_complete_at_any_run_instant(tmp_path, monkeypatch):
    """The shape #56 actually leaves behind, on the witness bytes' own clock: the
    stub #2449 is about is `2026-10-08.jsonl`, 9 rows whose newest is
    `2026-10-08T01:00:33.512979-07:00` — the day-D file the 01:0x-local pass creates
    for the day it is running in.

    Before #2449 the shipped function called that stub `complete` for any run landing
    within NEWEST_BUCKET_PARTIAL_HOURS of the write, because with the day not yet
    closed `min` always picks the run instant and 01:00→01:30 is a 0.5 h gap under
    the 2 h bar. The item's own reproduction, at run instants local 01:30 / 02:30 /
    03:30 on the same day, printed `complete (0.5 h)`, `complete (1.5 h)` and
    `LAST DAY PARTIAL (2.5 h)` — the first two lying, and the third honest only by
    arithmetic.
    """
    for run_utc, hours, hhmm in (("08:30", 0.5, "01:30"),
                                 ("09:30", 1.5, "02:30"),
                                 ("10:30", 2.5, "03:30")):
        window = _load_window(
            tmp_path, {WITNESS_OPEN_DAY: [_row_at("s1", WITNESS_OPEN_DAY,
                                                  WITNESS_OPEN_NEWEST, "-07:00")]},
            f"2026-10-08T{run_utc}:00+00:00", monkeypatch, zone=WITNESS_ZONE)
        newest = window["newest"]
        assert newest["date"] == WITNESS_OPEN_DAY, newest
        assert newest["partial"] is True, (
            f"a 9-row stub of the day still being written read complete at local "
            f"{hhmm}: {newest}")
        assert newest["uncovered_hours"] == hours, (
            f"local {hhmm} must still report the gap to its newest row "
            f"({WITNESS_OPEN_NEWEST} → {hhmm} = {hours} h), got {newest}")


def test_the_run_own_day_bucket_is_partial_across_the_whole_local_day(tmp_path, monkeypatch):
    """"at any run instant", checked rather than asserted: every 7 minutes of the
    local day the stub belongs to, from 00:00 to 23:59 PDT.

    The sweep deliberately includes instants BEFORE the stub's newest row, where the
    clamped figure legitimately reads 0.0 h — a row the run has not seen yet is not
    uncovered time — so the invariant is the flag, and the 0.0 reading is kept
    honest by the pairing: no instant in the day is complete, and no instant whose
    newest row predates it reports 0.0.
    """
    saw_zero_gap = saw_gap = 0
    minute = 0
    while minute < 24 * 60:
        run = datetime(2026, 10, 8, 7, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
        window = _load_window(
            tmp_path, {WITNESS_OPEN_DAY: [_row_at("s1", WITNESS_OPEN_DAY,
                                                  WITNESS_OPEN_NEWEST, "-07:00")]},
            run.isoformat(), monkeypatch, zone=WITNESS_ZONE)
        newest = window["newest"]
        assert newest["partial"] is True, (f"run {run.isoformat()} read complete: {newest}")
        if newest["uncovered_hours"] == 0.0:
            saw_zero_gap += 1
        else:
            saw_gap += 1
            assert newest["uncovered_hours"] > 0.0, newest
        minute += 7
    assert saw_zero_gap and saw_gap, (
        "the sweep stopped covering both sides of the stub write, so it no longer "
        f"tests the instant the flag used to flip on: {saw_zero_gap}/{saw_gap}")


def test_no_test_in_this_file_asserts_the_open_day_is_complete():
    """The rail #2449 leaves behind, because the node it had to amend
    (`test_a_run_earlier_than_the_days_close_measures_the_gap_to_the_run_instant`)
    was green for a week while pinning the defect exactly. "The suite is green" is
    therefore not the guarantee: a future node can re-assert `partial is False`
    beside a bucket dated the run's own local day, and every such node will look
    like a reasonable reading of `min(local_day_close, run_instant)`.

    Static and syntactic, resolved the way `_load_window` runs it: each
    `_load_window` call inside a node that asserts `partial … is False` gives up its
    bucket dates (dict keys, literal or module constant), its run instant, and the
    `zone=` it freezes TZ to; the instant is converted into THAT zone's local date,
    which is the date the function compares against. A call whose pieces cannot be
    read statically fails rather than being skipped — a rail that silently passes on
    the shapes it cannot parse is how #2449's defect stayed green in the first place.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }

    def resolve(node):
        """A bucket-date expression as its string, or None if not static."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        return None

    checked = 0
    offenders = []
    for test in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        asserts = [a for a in ast.walk(test) if isinstance(a, ast.Assert)
                   and "partial" in ast.unparse(a) and "is False" in ast.unparse(a)]
        if not asserts:
            continue
        for call in (c for c in ast.walk(test)
                     if isinstance(c, ast.Call)
                     and getattr(c.func, "id", None) == "_load_window"):
            buckets, now_iso, zone = call.args[1], call.args[2], "UTC"
            for kw in call.keywords:
                if kw.arg == "zone" and isinstance(kw.value, ast.Constant):
                    zone = kw.value.value
            assert isinstance(now_iso, ast.Constant) and isinstance(now_iso.value, str), (
                f"{test.name}: a node asserting `partial is False` passes a run instant "
                "this rail cannot read, so the open-day rule is unchecked there")
            run_local_date = datetime.fromisoformat(
                now_iso.value).astimezone(ZoneInfo(zone)).date().isoformat()
            for key in (buckets.keys if isinstance(buckets, ast.Dict) else []):
                day = resolve(key)
                assert day is not None, (
                    f"{test.name}: a bucket date this rail cannot read, so the "
                    "open-day rule is unchecked there")
                checked += 1
                if day == run_local_date:
                    offenders.append((test.name, day, now_iso.value, zone))
    assert checked >= 4, (
        f"the rail resolved only {checked} open-day-candidate bucket(s); it stopped "
        "covering the nodes it was written for")
    assert not offenders, (
        "these nodes assert a bucket dated the run's own LOCAL day is complete, which "
        f"is exactly the flag #2449 removed: {sorted(set(offenders))}")


def test_the_flag_describes_the_newest_bucket_not_an_older_stub(tmp_path, monkeypatch):
    """The flag belongs to the newest dated bucket alone. An older bucket that only
    holds its first hour is a fact about history; flagging it would make every
    future window read as short even after #56 catches up."""
    window = _load_window(tmp_path, {
        STALE_DAY: [_row_at("o1", STALE_DAY, "01:00:00")],
        PRIOR_DAY: [_row_at("p1", PRIOR_DAY, "22:10:00")],
    }, "2026-09-13T01:00:00+00:00", monkeypatch)
    assert window["files"] == 2, window
    assert window["newest"]["date"] == PRIOR_DAY, window
    assert window["newest"]["partial"] is False, window


def test_the_boundary_is_the_local_day_close_not_an_hour_from_utc(tmp_path, monkeypatch):
    """Clause 1's words are `its own local-day close`, and the two halves here are
    the same wall-clock stamp on two days that differ in one respect: the machine's
    offset on 2026-11-01 changes at 02:00 local, so that bucket's close is 25 clock
    hours after its own midnight, not 24.

    The stamp carries `-07:00`, the offset the day began with — which is the shape a
    transition-day bucket really holds for every row written before the clock change.
    Measuring to `local midnight + 24 h`, or to any offset pinned from one instant,
    puts that boundary an hour early and reports the bucket as complete: the correct
    gap is 2h30m, the naive one 1h30m, and the bound is 2 h, so the two disagree in
    the only direction that matters — a stub read as a finalised day."""
    transition = _load_window(
        tmp_path, {"2026-11-01": [_row_at("t1", "2026-11-01", "22:30:00",
                                          offset="-07:00")]},
        "2026-11-02T12:00:00+00:00", monkeypatch, zone="America/Los_Angeles")
    assert transition["newest"]["partial"] is True, transition
    ordinary = _load_window(
        tmp_path, {PRIOR_DAY: [_row_at("p1", PRIOR_DAY, "22:30:00",
                                       offset="-07:00")]},
        "2026-09-13T12:00:00+00:00", monkeypatch, zone="America/Los_Angeles")
    assert ordinary["newest"]["partial"] is False, ordinary


def test_the_miners_stderr_line_names_the_newest_bucket_and_its_flag(tmp_path, monkeypatch):
    """Clause 2: the line that carries the file/row denominator must also carry the
    newest bucket's date, its own row count and the flag — so a run over a
    first-hour bucket cannot emit the bare `N daily files` claim that #2045 was
    filed about."""
    stub, full = _past_day(3), _past_day(4)
    corpus = _multi_bucket_corpus(tmp_path, {
        full: [_row_at("f1", full, "09:00:00"), _row_at("f2", full, "23:40:00"),
               _row_at("f3", full, "23:55:00")],
        stub: [_row_at("s1", stub, "00:03:13"), _row_at("s2", stub, "01:09:19")],
    })
    with _frozen_zone():
        proc = _run(tmp_path, corpus, "--stats")
    lines = [l for l in proc.stderr.splitlines() if "newest bucket" in l]
    assert len(lines) == 1, proc.stderr
    assert f"newest bucket {stub}: 2 row(s)" in lines[0], lines[0]
    assert mt.PARTIAL_FLAG in lines[0], lines[0]


def test_a_denominator_cannot_travel_without_its_completeness_flag(tmp_path, monkeypatch):
    """Clause 2's mechanism, stated as the property the item needs: any line that
    prints the window's bucket/row count must print the flag beside it. It is the
    count alone that was read as a day count on 2026-10-02, so a line that printed
    the count and left the flag somewhere else would still be quotable as `7 daily
    files`."""
    stub = _past_day(3)
    corpus = _multi_bucket_corpus(tmp_path, {
        stub: [_row_at("s1", stub, "00:03:13"), _row_at("s2", stub, "01:09:19")],
    })
    with _frozen_zone():
        proc = _run(tmp_path, corpus, "--stats")
    denoms = [l for l in proc.stderr.splitlines() if "dated bucket(s)" in l]
    assert denoms, proc.stderr
    for line in denoms:
        assert (mt.PARTIAL_FLAG in line or mt.COMPLETE_FLAG in line), line


def test_the_miners_stderr_line_calls_a_finalised_newest_bucket_complete(tmp_path, monkeypatch):
    """Clause 2's other half: the same line must say `complete`, with that bucket's
    own row count, over a corpus whose newest day was finalised."""
    old, full = _past_day(5), _past_day(3)
    corpus = _multi_bucket_corpus(tmp_path, {
        old: [_row_at("o1", old, "01:00:00")],
        full: [_row_at("f1", full, "09:00:00"), _row_at("f2", full, "22:10:00")],
    })
    with _frozen_zone():
        proc = _run(tmp_path, corpus, "--stats")
    lines = [l for l in proc.stderr.splitlines() if "newest bucket" in l]
    assert len(lines) == 1, proc.stderr
    assert f"newest bucket {full}: 2 row(s)" in lines[0], lines[0]
    assert mt.COMPLETE_FLAG in lines[0] and mt.PARTIAL_FLAG not in lines[0], lines[0]


def _vault_file(rel, what):
    """A file in the vault, hard-asserted present.

    No skip when it is missing: this clause's subject is a tree outside the gated
    repo, and a guard that could go quietly unverified would repeat the defect it is
    guarding — the same reasoning `tests/test_trajectory_extraction.py` records for
    its own runbook assertions, and `tests/test_automod_doc_claims.py` for reading
    the same tree unguarded."""
    from app.paths import VAULT_ROOT

    path = VAULT_ROOT.joinpath(*rel)
    assert path.is_file(), f"{what} is absent: {path} does not exist"
    return path


def _skill_step2():
    """The text of step 2, sliced by its own headings so an assertion cannot be
    satisfied by a sentence somewhere else in the runbook."""
    text = _vault_file(SKILL_PATH, "step 2, the clause's subject").read_text(
        encoding="utf-8")
    start = text.index("### 2.")
    return text[start:text.index("### 3.", start)]


def test_the_runbook_reports_the_flag_instead_of_counting_buckets():
    """Clause 3: step 2's denominator reconciliation must tell the run to carry the
    newest bucket's completeness flag into its result line and to say the window is
    short of a day when it is flagged, instead of counting buckets.

    The last assertion is the one that makes this more than a grep of prose: the
    sample line the runbook quotes is built from the flag constant the miner prints,
    so a rename in `mine-trajectories.py` that left the runbook quoting the old words
    turns this red. Prose that tells a run to look for a token the code stopped
    printing is the failure mode worth pinning — the run would then read `complete`
    in the flag column and say so."""
    section = _skill_step2()
    assert "completeness flag" in section and "result line" in section, section
    assert "short of a day" in section, section
    assert mt.PARTIAL_FLAG in section, section
    quoted = f"newest bucket {WITNESS_DAY}: {WITNESS_ROWS} row(s), {mt.PARTIAL_FLAG}"
    assert quoted in section, section
    at = section.index("no missing day")
    around = section[max(0, at - 220): at + 140]
    assert "never" in around and "daily files" in around, (
        f"the prohibition is gone; step 2 reads:\n{around}")


def test_the_committed_witness_bytes_are_the_window_the_item_quotes(tmp_path, monkeypatch):
    """Clause 4: re-derive the quoted report from the committed bytes rather than
    from the live corpus the item names. `wc -l` on
    `backlog/data/2026-10-01.jsonl` is the figure #2045 quotes (27), its newest row
    is the stamp the item quotes, and those bytes under the flag rule are the stub
    the item describes — so the sentence the runbook teaches a run to print is the
    sentence these bytes produce, not one a fixture arranged."""
    path = _vault_file(WITNESS_PATH, "the witness bucket")
    rows = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == WITNESS_ROWS, len(rows)
    stamps = [json.loads(l)["timestamp"] for l in rows]
    assert max(stamps) == WITNESS_NEWEST_ROW, max(stamps)

    corpus = {WITNESS_DAY: [json.loads(l) for l in rows]}
    window = _load_window(tmp_path, corpus, "2026-10-02T12:00:00+00:00",
                          monkeypatch, zone="America/Los_Angeles")
    assert window["rows"] == WITNESS_ROWS and window["newest"]["partial"] is True, window
    printed = mt.window_denominator_line(window)
    assert (f"newest bucket {WITNESS_DAY}: {WITNESS_ROWS} row(s), "
            f"{mt.PARTIAL_FLAG}") in printed, printed
    assert f"newest bucket {WITNESS_DAY}: {WITNESS_ROWS} row(s), " in _skill_step2(), \
        "step 2 no longer quotes the denominator these bytes print"
