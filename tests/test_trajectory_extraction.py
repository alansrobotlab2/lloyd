"""Trajectory extraction — local-date bucketing, idempotent append, scrubbing.

Why this file exists
--------------------
`scripts/extract-trajectories.py` had no tests, and it carries two defects that
burned six reflection cycles each:

  * date bucketing by UTC misfiled every session after 17:00 PDT one day late,
    so `YYYY-MM-DD.jsonl` stopped lining up with `memory/learnings/YYYY-MM-DD.md`
    (fixed 2026-08-21 by bucketing in America/Los_Angeles);
  * backfill re-covers sessions a prior run already wrote, and a plain append
    produced byte-identical duplicate lines — measured at 26% entry inflation
    (fixed 2026-09-01 by dedup-on-write keyed on `session_key`).

Both fixes exist and neither was asserted. A file with a dash in its name is not
importable, so it is loaded by path.

`OUTPUT_DIR` / `WATERMARK_PATH` are module-level and point at the live
`_pipeline`, so the autouse fixture redirects them; nothing here reads or writes
production trajectory data except the one read-only integrity guard.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "extract_trajectories", _ROOT / "scripts" / "extract-trajectories.py"
)
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)

# The miner consumes the extractor's error flag, so the corroboration contract
# is pinned on both sides of the boundary in this one file.
_mspec = importlib.util.spec_from_file_location(
    "mine_trajectories", _ROOT / "scripts" / "mine-trajectories.py"
)
mt = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(mt)

LOCAL_TZ = et.LOCAL_TZ

# Dedup-on-write landed 2026-09-01 (commit 9b9450c). Bucket files dated on or
# after this must never contain a repeated session_key.
DEDUP_FIX_DATE = "2026-08-28"


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    out = tmp_path / "trajectories"
    out.mkdir()
    monkeypatch.setattr(et, "OUTPUT_DIR", out)
    monkeypatch.setattr(et, "WATERMARK_PATH", out / ".watermark.json")
    return out


def traj(key, ts):
    return {"session_key": key, "timestamp": ts}


# ── date bucketing ───────────────────────────────────────────────────────────

def test_daytime_utc_timestamp_buckets_to_the_same_local_date():
    assert et.trajectory_date_key(traj("s", "2026-09-04T18:00:00Z")) == "2026-09-04"


def test_evening_utc_timestamp_buckets_to_the_previous_local_date():
    """The 17:00-24:00 PDT window that misfiled sessions one day late.
    2026-09-04T05:00Z is 2026-09-03 22:00 PDT."""
    assert et.trajectory_date_key(traj("s", "2026-09-04T05:00:00Z")) == "2026-09-03"


def test_local_date_not_utc_is_the_bucket():
    """A session at 2026-08-31T23:30Z is 2026-08-31 16:30 PDT: same day either
    way. The discriminating case is the early-UTC one above."""
    assert et.trajectory_date_key(traj("s", "2026-08-31T23:30:00Z")) == "2026-08-31"


def test_utc_offset_timestamps_are_honoured():
    assert et.trajectory_date_key(traj("s", "2026-09-04T01:00:00+00:00")) == "2026-09-03"


def test_pacific_summer_vs_winter_offsets_both_bucket_correctly():
    """2026-07-04T02:00Z is 2026-07-03 19:00 PDT; 2026-01-04T02:00Z is
    2026-01-03 18:00 PST. Both must land on the earlier local date."""
    assert et.trajectory_date_key(traj("s", "2026-07-04T02:00:00Z")) == "2026-07-03"
    assert et.trajectory_date_key(traj("s", "2026-01-04T02:00:00Z")) == "2026-01-03"


def test_missing_timestamp_falls_back_to_today_local():
    expected = datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")
    assert et.trajectory_date_key(traj("s", "")) == expected
    assert et.trajectory_date_key({"session_key": "s"}) == expected


def test_malformed_timestamp_falls_back_instead_of_raising():
    expected = datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")
    assert et.trajectory_date_key(traj("s", "not-a-date")) == expected
    assert et.trajectory_date_key(traj("s", "2026-13-45T99:99:99Z")) == expected


def test_non_string_timestamp_does_not_raise():
    expected = datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")
    assert et.trajectory_date_key({"session_key": "s", "timestamp": 12345}) == expected


def test_bucketing_matches_the_local_timezone_the_daily_notes_use():
    assert str(LOCAL_TZ) == "America/Los_Angeles"


def test_a_session_spanning_midnight_is_bucketed_by_its_timestamp():
    """Two sessions minutes apart across the local midnight boundary must not
    share a bucket."""
    before = et.trajectory_date_key(traj("a", "2026-09-04T06:59:00Z"))   # 23:59 PDT 09-03
    after = et.trajectory_date_key(traj("b", "2026-09-04T07:01:00Z"))    # 00:01 PDT 09-04
    assert (before, after) == ("2026-09-03", "2026-09-04")


# ── idempotent append ────────────────────────────────────────────────────────

def test_first_append_creates_the_bucket(isolated_output):
    et.append_trajectories([traj("s1", "2026-09-04T18:00:00Z")])
    lines = (isolated_output / "2026-09-04.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["session_key"] == "s1"


def test_re_appending_the_same_session_writes_nothing(isolated_output):
    """The backfill defect: re-covering a session used to duplicate it."""
    et.append_trajectories([traj("s1", "2026-09-04T18:00:00Z")])
    et.append_trajectories([traj("s1", "2026-09-04T18:00:00Z")])
    assert len((isolated_output / "2026-09-04.jsonl").read_text().strip().splitlines()) == 1


def test_backfill_never_inflates_the_entry_count(isolated_output):
    sessions = [traj(f"s{i}", "2026-09-04T18:00:00Z") for i in range(5)]
    et.append_trajectories(sessions)
    for _ in range(3):
        et.append_trajectories(sessions)
    assert len((isolated_output / "2026-09-04.jsonl").read_text().strip().splitlines()) == 5


def test_backfill_with_one_new_session_appends_only_the_new_one(isolated_output):
    et.append_trajectories([traj("s1", "2026-09-04T18:00:00Z")])
    et.append_trajectories([
        traj("s1", "2026-09-04T18:00:00Z"),
        traj("s2", "2026-09-04T19:00:00Z"),
    ])
    keys = [json.loads(l)["session_key"]
            for l in (isolated_output / "2026-09-04.jsonl").read_text().splitlines()]
    assert keys == ["s1", "s2"]


def test_dedup_is_per_bucket_not_global(isolated_output):
    """The same session_key bucketed to two dates is written to both files.
    Characterized: a session re-bucketed by a corrected timestamp grows rather
    than moves, so a bucketing fix needs a rewrite pass, not an append."""
    et.append_trajectories([traj("s1", "2026-09-04T18:00:00Z")])
    et.append_trajectories([traj("s1", "2026-09-05T18:00:00Z")])
    assert (isolated_output / "2026-09-04.jsonl").exists()
    assert (isolated_output / "2026-09-05.jsonl").exists()


def test_one_batch_spanning_local_midnight_writes_two_buckets(isolated_output):
    et.append_trajectories([
        traj("a", "2026-09-04T06:59:00Z"),
        traj("b", "2026-09-04T07:01:00Z"),
    ])
    assert (isolated_output / "2026-09-03.jsonl").exists()
    assert (isolated_output / "2026-09-04.jsonl").exists()


def test_entries_with_no_session_key_are_kept(isolated_output):
    """Characterized: dedup is keyed on session_key, and a keyless entry cannot
    be deduped — re-appending it duplicates it. `parse_session` always sets the
    key, so this is a malformed-input path."""
    et.append_trajectories([{"timestamp": "2026-09-04T18:00:00Z"}])
    et.append_trajectories([{"timestamp": "2026-09-04T18:00:00Z"}])
    assert len((isolated_output / "2026-09-04.jsonl").read_text().strip().splitlines()) == 2


def test_corrupt_existing_lines_do_not_break_dedup(isolated_output):
    target = isolated_output / "2026-09-04.jsonl"
    target.write_text("this is not json\n\n" + json.dumps(traj("s1", "x")) + "\n",
                      encoding="utf-8")
    et.append_trajectories([traj("s1", "2026-09-04T18:00:00Z")])
    assert len(target.read_text().strip().splitlines()) == 3      # junk + s1, no dupe


def test_append_never_rewrites_existing_bytes(isolated_output):
    target = isolated_output / "2026-09-04.jsonl"
    et.append_trajectories([traj("s1", "2026-09-04T18:00:00Z")])
    before = target.read_bytes()
    et.append_trajectories([traj("s2", "2026-09-04T19:00:00Z")])
    assert target.read_bytes().startswith(before)


def test_non_ascii_session_content_survives_append(isolated_output):
    et.append_trajectories([{"session_key": "s1", "timestamp": "2026-09-04T18:00:00Z",
                             "summary": "Δ mean → 均值"}])
    line = (isolated_output / "2026-09-04.jsonl").read_text(encoding="utf-8")
    assert "Δ" in line and "\\u0394" not in line          # ensure_ascii=False


def test_rewrite_mode_replaces_the_bucket(isolated_output):
    et.append_trajectories([traj("old", "2026-09-04T18:00:00Z")])
    et.rewrite_trajectories([traj("new", "2026-09-04T18:00:00Z")])
    keys = [json.loads(l)["session_key"]
            for l in (isolated_output / "2026-09-04.jsonl").read_text().splitlines()]
    assert keys == ["new"]


# ── live-data integrity guard ────────────────────────────────────────────────

def test_production_buckets_since_the_fix_have_no_duplicate_keys():
    """Read-only. The defect that ran six cycles must not be running now."""
    live = _ROOT / "_pipeline" / "trajectories"
    if not live.exists():
        pytest.skip("no trajectories dir on this machine")
    offending = {}
    for path in sorted(live.glob("*.jsonl")):
        if path.stem < DEDUP_FIX_DATE:
            continue                                  # pre-fix legacy data
        keys = [json.loads(l).get("session_key")
                for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(keys) != len(set(keys)):
            offending[path.name] = (len(keys), len(set(keys)))
    assert not offending, f"duplicate session_keys reappeared post-fix: {offending}"


def test_pre_fix_duplicate_buckets_are_frozen_not_growing():
    """Legacy files (before the 2026-09-01 dedup fix) are allowed to carry
    duplicates, but their duplicate counts are pinned so nothing new appends to
    them. Re-run the extractor with --rewrite to clean them, then delete this."""
    known = {
        "2026-08-22.jsonl": (6, 3),
        "2026-08-23.jsonl": (11, 10),
        "2026-08-25.jsonl": (5, 2),
        "2026-08-26.jsonl": (7, 3),
        "2026-08-27.jsonl": (6, 3),
    }
    live = _ROOT / "_pipeline" / "trajectories"
    if not live.exists():
        pytest.skip("no trajectories dir on this machine")
    found = {}
    for path in sorted(live.glob("*.jsonl")):
        if path.stem >= DEDUP_FIX_DATE:
            continue
        keys = [json.loads(l).get("session_key")
                for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(keys) != len(set(keys)):
            found[path.name] = (len(keys), len(set(keys)))
    assert found == known, (
        f"legacy duplicate buckets changed: {found} — if you rewrote them, empty "
        "the `known` map; if new dupes appeared post-fix, that is a regression"
    )


# ── scrubbing ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("secret", [
    "sk-ant-api03-abcdefghijklmnop",
    "ghp_" + "A" * 36,
    "xoxb-1234-5678-abcdefghij",
    "Authorization: Bearer abc.def_ghi~+jkl=",
    "api_key: hunter2secret",
    "APIKEY=s3cr3tvalue",
    "password: letmein123",
])
def test_secrets_are_masked(secret):
    out = et.mask_sensitive(f"before {secret} after")
    assert "[MASKED]" in out
    assert secret not in out


def test_ordinary_text_is_untouched():
    text = "read tests/test_kg_store.py and found 21 tests"
    assert et.mask_sensitive(text) == text


def test_masking_is_repeatable():
    once = et.mask_sensitive("token=abcdefghij123456")
    assert et.mask_sensitive(once) == once


def test_bearer_token_is_masked_but_the_word_bearer_is_not_required():
    out = et.mask_sensitive("Bearer sk-abcdefghijklmnop")
    assert out.count("[MASKED]") >= 1


def test_content_keys_truncate_aggressively():
    out = et.scrub_value("content", "x" * (et.MAX_FILE_CONTENT_LEN + 500))
    assert len(out) < et.MAX_FILE_CONTENT_LEN + 60
    assert "[truncated:" in out


def test_other_keys_truncate_at_the_generic_limit():
    out = et.scrub_value("path_note", "y" * (et.MAX_STRING_LEN + 10))
    assert out.startswith("[truncated:")
    assert f"{et.MAX_STRING_LEN + 10} chars" in out


def test_short_values_pass_through():
    assert et.scrub_value("file_path", "/tmp/x.py") == "/tmp/x.py"


def test_non_string_values_are_returned_unchanged():
    for v in (7, 7.5, True, None, ["a"], {"k": "v"}):
        assert et.scrub_value("anything", v) == v


def test_scrub_params_masks_and_truncates_per_key():
    out = et.scrub_params({
        "file_path": "/tmp/a.py",
        "token": "sk-abcdefghijklmnop1234",
        "content": "z" * 5000,
    })
    assert out["file_path"] == "/tmp/a.py"
    assert "[MASKED]" in out["token"]
    assert "[truncated:" in out["content"]


def test_parameter_names_do_not_trigger_masking():
    """Characterized gap: masking matches the *value*, never the key, so a bare
    secret stored under a sensitive key passes through untouched. `mask_sensitive`
    needs a recognisable shape (`sk-…`, `Bearer …`, `key=value`)."""
    assert et.scrub_value("api_key", "hunter2") == "hunter2"


def test_scrub_params_survives_non_dict_input():
    assert et.scrub_params(None) == {}
    assert et.scrub_params("a string") == {}
    assert et.scrub_params([1, 2]) == {}


def test_content_key_matching_is_case_insensitive():
    assert "[truncated:" in et.scrub_value("CONTENT", "q" * 4000)


# ── error classification ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text,category", [
    ("Permission denied: /etc/shadow", "permission"),
    ("EACCES: permission denied", "permission"),
    ("404 Not Found", "not_found"),
    ("FileNotFoundError: no such file", "not_found"),
    ("request timed out after 30s", "timeout"),
    ("connection refused by 127.0.0.1", "network"),
    ("invalid JSON payload", "validation"),
    ("syntax error in config", "validation"),
    ("out of memory while loading", "resource"),
])
def test_error_categories(text, category):
    assert et.categorize_error(text) == category


def test_python_exception_names_are_not_categorised_by_themselves():
    """Characterized gap: the patterns match the phrase `syntax error`, not the
    exception name `SyntaxError`, which is what tracebacks actually contain."""
    assert et.categorize_error("SyntaxError") == "logic"
    assert et.categorize_error("TypeError: unsupported operand") == "logic"


def test_uncategorized_errors_default_to_logic():
    assert et.categorize_error("the assertion compared the wrong field") == "logic"
    assert et.categorize_error("") == "logic"


def test_category_precedence_follows_the_declared_order():
    """Characterized: 'permission denied' also contains 'denied', and a message
    matching several patterns takes the first declared category."""
    assert et.categorize_error("timeout: permission denied") == "permission"


@pytest.mark.parametrize("text", [
    "Traceback (most recent call last):", "ValueError: bad input",
    "bash: foo: command not found", "No such file or directory",
    "npm ERR! code ELIFECYCLE", "FAILED tests/test_x.py", "fatal error: oops",
])
def test_semantic_errors_are_detected_without_an_is_error_flag(text):
    assert et.has_semantic_error(text) is True


def test_a_bare_exit_n_in_prose_is_no_longer_sweep_vocabulary():
    """#389 fix step 3: `exit [1-9]` was dropped from the sweep. It matched
    prose like "the script exit 1 on bad input" and, worse, duplicated what the
    Bash tool already reports as a structured trailer — see EXIT_CODE_RE."""
    assert et.has_semantic_error("exit 1") is False
    assert et.parse_exit_code("boom\n\n[exit code: 1]") == 1


@pytest.mark.parametrize("text", [
    "exit 0", "All 755 tests passed", "wrote 12 lines", "",
    "the word errorless here",
])
def test_healthy_output_is_not_flagged_as_an_error(text):
    assert et.has_semantic_error(text) is False


# ── result summaries ─────────────────────────────────────────────────────────

def test_ok_result_summary_reports_only_the_length():
    assert et.result_summary("hello world", False) == "OK: 11 chars"


def test_error_result_summary_includes_a_flattened_preview():
    out = et.result_summary("boom\nline two", True)
    assert out.startswith("ERROR: boom line two")
    assert "\n" not in out


def test_error_preview_is_capped():
    out = et.result_summary("e" * 5000, True)
    assert len(out) <= et.MAX_ERROR_LEN + len("ERROR: ")


# ── filters ──────────────────────────────────────────────────────────────────

def test_mtime_filter_keeps_only_recent_files(tmp_path):
    import os
    old = tmp_path / "old.json"; old.write_text("{}")
    new = tmp_path / "new.json"; new.write_text("{}")
    os.utime(old, (0, 1_600_000_000))
    assert et.filter_by_mtime([old, new], 1_700_000_000) == [new]


def test_mtime_filter_with_no_cutoff_keeps_everything(tmp_path):
    a = tmp_path / "a.json"; a.write_text("{}")
    assert et.filter_by_mtime([a], None) == [a]


def test_days_filter_excludes_ancient_files(tmp_path):
    import os
    old = tmp_path / "old.json"; old.write_text("{}")
    os.utime(old, (0, 1_400_000_000))
    assert et.filter_by_days([old], days=1) == []


# ── watermark ────────────────────────────────────────────────────────────────

def test_watermark_round_trips(isolated_output):
    et.save_watermark({"last_run": "2026-09-04T00:00:00Z", "count": 3})
    assert et.load_watermark() == {"last_run": "2026-09-04T00:00:00Z", "count": 3}


def test_missing_watermark_loads_the_default_state(isolated_output):
    assert et.load_watermark() == {
        "last_run": None, "sessions_processed": 0, "last_session_mtime": None,
    }


def test_corrupt_watermark_falls_back_to_the_default(isolated_output):
    """A corrupt watermark means 'process everything', i.e. a full re-scan — the
    backfill path that produced the duplicate-append defect, which is why the
    dedup tests above matter."""
    et.WATERMARK_PATH.write_text("{not json", encoding="utf-8")
    assert et.load_watermark() == {
        "last_run": None, "sessions_processed": 0, "last_session_mtime": None,
    }


# ── error corroboration (backlog #389) ───────────────────────────────────────
#
# `is_error` is the input to skill authoring: `trajectory-skill-mining` opens a
# skill-writing branch on ">= 2 pending error candidates". Keyword-matching the
# *text a tool returned* made that counter fiction — in the 2026-09-06→08 window
# 234 steps were flagged and 188 had nothing behind them (review verdicts:
# `_pipeline/skills/candidates/REVIEW-LOG.md`), including a `Read/timeout`
# candidate for a tool with no timeout path, where the word came from the file
# being read. Authoring off those would have emitted the miner's own hardcoded
# mitigation strings as skills, the damage class that got 11 skills archived on
# 2026-09-04.
#
# So: a step is an error only if something other than its own output says so —
# the harness's `stats.is_error`, a non-zero exit state, or a structured
# error body. The keyword sweep survives as `output_mentions_errors`, which
# promotes nothing.

def write_session(tmp_path, calls, name="sess-corroboration"):
    """Write a session file in the shape `parse_session` consumes.

    `calls` is a list of (tool_name, arguments, result_text, stats_is_error);
    pass stats_is_error=None to simulate a session written before tool
    messages carried `stats`.
    """
    messages = []
    for i, (tool_name, args, result, stats_error) in enumerate(calls):
        messages.append({
            "role": "assistant",
            "tool_calls": [{"id": f"call_{i}",
                            "function": {"name": tool_name,
                                         "arguments": json.dumps(args)}}],
        })
        message = {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": [{"type": "text", "text": result}],
        }
        if stats_error is not None:
            message["stats"] = {"result_chars": len(result), "is_error": stats_error}
        messages.append(message)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "session_id": name,
        "session_start": "2026-09-08T18:00:00Z",
        "messages": messages,
    }), encoding="utf-8")
    return path


def first_tool(tmp_path, calls, **kw):
    traj = et.parse_session(write_session(tmp_path, calls, **kw))
    assert traj is not None
    return traj


def test_reading_a_file_that_mentions_exceptions_does_not_flag_is_error(tmp_path):
    """The regression that produced the phantom `Read/timeout` candidate: read
    tooling returns file content, and file content says 'Exception'."""
    body = ("def handler(x):\n"
            "    try:\n"
            "        return x.run()\n"
            "    except Exception as e:\n"
            "        raise ValueError(f'Error: {e}') from e\n")
    traj = first_tool(tmp_path, [("Read", {"file_path": "/x/handler.py"}, body, False)])
    tool = traj["tools"][0]
    assert tool["is_error"] is False
    assert tool["error_source"] is None
    assert traj["error_count"] == 0
    # Content-returning tools report the mention flag as False by design: their
    # result *is* arbitrary text, so the field would be true for most reads.
    assert tool["output_mentions_errors"] is False


def test_a_grep_whose_output_contains_assertionerror_does_not_flag(tmp_path):
    traj = first_tool(tmp_path, [
        ("Grep", {"pattern": "AssertionError", "path": "~/lloyd/tests"},
         "tests/test_x.py:12:    raise AssertionError('boom')\n", False),
    ])
    assert traj["tools"][0]["is_error"] is False
    assert traj["error_count"] == 0


@pytest.mark.parametrize("tool_name", [
    "Read", "Grep", "Glob", "LS", "NotebookRead",
    "mcp____vault_read", "mcp____memory_read", "mcp____skills_read",
])
def test_read_only_tools_are_never_flagged_from_result_text(tmp_path, tool_name):
    """Content-returning tools carry zero failure information in their result,
    whatever the content says."""
    traj = first_tool(tmp_path, [
        (tool_name, {"path": "/x/y"},
         "Traceback (most recent call last):\n    ValueError: Error: no such file\n",
         False),
    ])
    tool = traj["tools"][0]
    assert tool["is_error"] is False
    assert tool["output_mentions_errors"] is False
    assert traj["error_count"] == 0


@pytest.mark.parametrize("tool_name,expected", [
    ("Read", True), ("Grep", True), ("Glob", True), ("LS", True),
    ("NotebookRead", True),
    ("mcp____vault_read", True), ("mcp____vault_write", False),
    ("mcp____memory_read", True), ("mcp____browser_snapshot", True),
    ("mcp____autonomy_run_task", False),
    ("Bash", False), ("Edit", False), ("Write", False),
    ("mcp____backlog_write_task", False), ("mcp____email_delete", False),
])
def test_read_only_classification_names_content_tools_and_no_writers(tool_name, expected):
    """The suffix rule must not quietly claim a writer — a write tool treated as
    read-only would stop reporting content-corroborated failures."""
    assert et.is_read_only_tool(tool_name) is expected


def test_read_only_tools_are_still_flagged_when_the_harness_said_so(tmp_path):
    """The read-only rule demotes keyword matching, not the authoritative flag:
    a Read of a path that does not exist really did fail."""
    traj = first_tool(tmp_path, [
        ("Read", {"file_path": "/x/missing.py"}, "File does not exist.", True),
    ])
    tool = traj["tools"][0]
    assert tool["is_error"] is True
    assert tool["error_source"] == "protocol"
    # Characterized: `categorize_error` matches the phrase "no such file", not
    # the harness's wording "File does not exist", so this lands in `logic`.
    # Retyping it is a categorization change, outside #389.
    assert traj["error_tools"][0]["error_type"] == "logic"


def test_bash_output_that_merely_prints_the_word_error_is_not_an_error(tmp_path):
    """The 153 semantic-only Bash flags in the 09-06→08 window."""
    traj = first_tool(tmp_path, [
        ("Bash", {"command": "pytest -q"}, "Error: plugin warning emitted\n755 passed\n", False),
    ])
    tool = traj["tools"][0]
    assert tool["is_error"] is False
    assert tool["output_mentions_errors"] is True
    assert tool["error_source"] is None


def test_a_nonzero_exit_state_is_a_corroborated_error(tmp_path):
    traj = first_tool(tmp_path, [
        ("Bash", {"command": "grep -rn foo ~/lloyd"},
         "grep: no match\n\n[exit code: 1]", False),
    ])
    tool = traj["tools"][0]
    assert tool["exit_code"] == 1
    assert tool["is_error"] is True
    assert tool["error_source"] == "exit_code"
    assert traj["error_tools"][0]["name"] == "Bash"


def test_a_zero_exit_code_corroborates_nothing(tmp_path):
    traj = first_tool(tmp_path, [
        ("Bash", {"command": "echo hi"}, "hi\n\n[exit code: 0]", False),
    ])
    assert traj["tools"][0]["is_error"] is False
    assert traj["tools"][0]["exit_code"] == 0


def test_the_exit_state_is_read_from_full_output_not_the_truncated_preview(tmp_path):
    """#492: `result_summary()` keeps a 200-char prefix and the Bash marker sits
    at the end of the output, so corroboration cannot be recovered from the
    persisted preview — 44 of the 67 non-zero exits in the 09-06→08 window lose
    their marker there. It is parsed from the whole result and persisted as a
    field instead."""
    body = "x" * 4000 + "\n\n[exit code: 2]"
    traj = first_tool(tmp_path, [("Bash", {"command": "long"}, body, False)])
    tool = traj["tools"][0]
    assert "[exit code: 2]" not in tool["result_summary"]
    assert tool["exit_code"] == 2
    assert tool["is_error"] is True


def test_a_structured_error_body_corroborates_when_stats_is_absent(tmp_path):
    """Sessions predating tool-message `stats` keep the old fallback: a body
    that opens with an error field."""
    traj = first_tool(tmp_path, [
        ("Bash", {"command": "true"}, '{"error": "no server claims tool \'name\'"}', None),
    ])
    tool = traj["tools"][0]
    assert tool["is_error"] is True
    assert tool["error_source"] == "protocol"


def test_a_zero_code_in_a_structured_body_corroborates_nothing(tmp_path):
    traj = first_tool(tmp_path, [
        ("Bash", {"command": "true"}, '{"code": 0, "stdout": "fine"}', None),
    ])
    assert traj["tools"][0]["is_error"] is False


def test_the_harness_error_flag_is_authoritative_over_a_benign_looking_body(tmp_path):
    traj = first_tool(tmp_path, [("Edit", {"path": "/x/y"}, "ok", True)])
    tool = traj["tools"][0]
    assert tool["is_error"] is True
    assert tool["error_source"] == "protocol"
    assert tool["output_mentions_errors"] is False


def test_error_count_counts_only_corroborated_failures(tmp_path):
    traj = first_tool(tmp_path, [
        ("Read", {"file_path": "/a"}, "except Exception as e:", False),
        ("Grep", {"pattern": "x"}, "AssertionError in output", False),
        ("Bash", {"command": "false"}, "boom\n\n[exit code: 1]", True),
    ])
    assert [t["is_error"] for t in traj["tools"]] == [False, False, True]
    assert traj["error_count"] == 1
    assert traj["has_errors"] is True


# ── the miner must inherit the same contract ─────────────────────────────────

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


# ── signature keys and candidate emission (backlog #391) ─────────────────────
#
# `scrub_value` replaced an over-long `command` string with a bare
# `[truncated: N chars]` placeholder, and `normalize_params_signature` took the
# first whitespace token of what was left — so every long command in the corpus,
# whatever it did, keyed as `Bash/[truncated:_signature`: the #2 pattern in the
# 2026-09-08 set by occurrences (625) over 70 sessions, an unsorted union of
# youtube-transcript-api calls, awk, nvidia-smi and git. A skill mined from it
# describes the extractor.
#
# The second defect is the class those keys belong to. `*_signature` names the
# tool plus the *set of argument names* a call carried
# (`Read/file_path_limit_offset_signature` = "Read called with file_path, limit
# and offset"), and its body is a frequency count of successful calls. That is a
# parameter contract restated, which authoring rule 5 forbids; 11 skills were
# archived 2026-09-04 for exactly that content and one of them was re-mined this
# month. On 2026-09-08 the class was 93 of 103 candidate keys and all 20 of the
# top 20 by occurrences, topped by `Bash/cd_signature` (1,129) — the shell
# chaining artefact.

TRUNC = "[truncated: 900 chars]"
LONG = "y" * (et.MAX_STRING_LEN + 1)


def test_a_truncated_bash_command_keeps_its_leading_verb():
    out = et.scrub_value("command", "nvidia-smi " + LONG)
    assert out.startswith("nvidia-smi ")
    assert "[truncated:" in out


def test_only_command_arguments_keep_a_verb():
    """The verb is worth preserving because it is the grouping key of a shell
    call; a path or a url has no equivalent, and inventing one there would be a
    new key class nobody asked for."""
    assert et.scrub_value("file_path", LONG).startswith("[truncated:")


def test_a_command_that_needs_no_truncation_is_untouched():
    assert et.scrub_value("command", "ls -la") == "ls -la"


def test_the_verb_survives_the_end_to_end_scrub(tmp_path):
    traj = first_tool(tmp_path, [
        ("Bash", {"command": "awk '{print $1}' " + LONG}, "ok", False),
    ])
    assert traj["tools"][0]["params_summary"]["command"].startswith("awk ")


def test_unrelated_long_commands_no_longer_share_one_key():
    a = et.scrub_value("command", "youtube-transcript-api " + "u" * 3000)
    b = et.scrub_value("command", "nvidia-smi " + "v" * 3000)
    assert mt.normalize_params_signature({"command": a}) == "cmd:youtube-transcript-api"
    assert mt.normalize_params_signature({"command": b}) == "cmd:nvidia-smi"


def test_the_program_is_keyed_by_basename():
    assert mt.normalize_params_signature(
        {"command": f"/usr/bin/python3 {TRUNC}"}) == "cmd:python3"


def test_a_command_scrubbed_before_the_fix_is_dropped_from_signature_mining():
    """Rows already on disk have no recoverable verb. The item's other option —
    drop the call from signature mining — is what the empty key means to the
    miner; it must not fall back into one bucket."""
    assert mt.normalize_params_signature({"command": TRUNC}) == ""
    assert mt.normalize_params_signature({"command": "[MASKED] something"}) == ""


def unkeyed_traj(session_key):
    """One un-keyable Bash call and one normal Read call, both successful."""
    return {
        "session_key": session_key,
        "timestamp": "2026-09-08T18:00:00Z",
        "tool_count": 2,
        "error_count": 0,
        "has_errors": False,
        "tools": [
            {"name": "Bash", "is_error": False, "sequence": 0,
             "params_summary": {"command": TRUNC}, "result_summary": "ok"},
            {"name": "Read", "is_error": False, "sequence": 1,
             "params_summary": {"file_path": "/x", "limit": 100},
             "result_summary": "ok"},
        ],
        "error_tools": [],
        "signals": [],
    }


def test_an_unkeyable_command_is_dropped_not_bucketed():
    rows = [unkeyed_traj(f"s{i}") for i in (1, 2)]
    patterns = mt.mine_success_patterns(rows, threshold=2)
    assert [(p["tool_name"], p["params_signature"]) for p in patterns] == [
        ("Read", "file_path_limit_signature")]


def test_an_unkeyable_command_error_keeps_its_error_signal():
    """Dropping is a *signature-mining* rule. Error mining groups on
    (tool, error_type, signature) and emits (tool, error_type), so an
    un-keyable command still belongs in the error table — dropping it there
    would lose the failure, which is the opposite of what #389 was for."""
    traj = [error_traj(f"s{i}", "Bash", "not_found", "protocol",
                       params={"command": TRUNC}) for i in (1, 2)]
    patterns = mt.mine_error_patterns(traj, threshold=2)
    assert len(patterns) == 1
    assert patterns[0]["params_signature"] == "generic"


def success_pattern(sig, tool="Read", occ=9):
    return {
        "type": "success", "tool_name": tool, "params_signature": sig,
        "sessions": {"s1", "s2"}, "examples": [], "dates": {"2026-09-08"},
        "total_calls": occ, "error_count": 0, "error_rate": 0.0,
        "first_seen": "2026-09-08", "last_seen": "2026-09-08",
    }


def test_a_parameter_key_set_is_not_emittable():
    """The key names which arguments a call carried, and the body is a count of
    successes. Nothing in it is a decision, so the candidate cannot be authored
    as a skill (authoring rule 5)."""
    assert mt.is_emittable(success_pattern("file_path_limit_offset_signature")) is False
    assert mt.is_emittable(success_pattern("file_path_signature")) is False


def test_a_bare_program_name_is_not_emittable_either():
    """`cmd:` is a better grouping key than a truncation placeholder, but a
    candidate that says "Bash was called with cd" is the same non-skill, and
    `Bash/cd_signature` was the #1 pattern today at 1,129 occurrences."""
    assert mt.is_emittable(success_pattern("cmd:cd", tool="Bash")) is False
    assert mt.is_emittable(success_pattern("cmd:ls", tool="Bash")) is False


def test_a_candidate_with_no_key_at_all_is_not_emittable():
    assert mt.is_emittable(success_pattern("")) is False
    assert mt.is_emittable(success_pattern("generic")) is False


def test_the_emission_gate_leaves_room_for_a_value_shape_key():
    """The item's alternative to stopping emission is redefining the class onto
    a parameter *value* shape that carries a decision — `Read` always passing
    `limit` on a >2,000-line file. Such a key must pass, or the gate would be a
    way of deleting the class rather than a way of holding it to rule 5."""
    assert mt.is_emittable(
        success_pattern("read_limit_on_files_over_2000_lines")) is True


def mined_error_pattern():
    """One `type: error` pattern exactly as `mine_error_patterns` builds it, so
    the key set the emission gate sees is the miner's, not a hand-written one."""
    traj = [error_traj(f"s{i}", "Bash", "not_found", "protocol") for i in (1, 2)]
    patterns = mt.mine_error_patterns(traj, threshold=2)
    assert len(patterns) == 1, "the fixture must mine exactly one error pattern"
    return patterns[0]


def recovering_traj(session_key):
    """One session whose Bash step fails and is followed by successful steps.

    The flag the emission gate reads is derived from the steps, not declared, so a
    test that needs a sequence to *reach* a file needs this shape — `aliased_traj`
    carries no failing step at all and mines nothing the gate admits.

    Mined at threshold 2 across three copies this shape yields both sequence
    kinds the gate now separates: the 2-gram `bash:fs:ERR → read` and the
    3-gram `read → bash:fs:ERR → read` carry an `:ERR` step followed by a
    non-error step and are flagged `has_error_recovery: true`, while `bash:fs →
    read`, `read → bash:fs:ERR` and `bash:fs → read → bash:fs:ERR` end on or
    before the failure and are flagged false. `error_tools` is empty on purpose:
    error mining has its own fixture (`error_traj`) and adding a row here would
    add an error pattern these assertions do not need.
    """
    return {
        "session_key": session_key,
        "timestamp": "2026-09-16T18:00:00Z",
        "tool_count": 4,
        "error_count": 1,
        "has_errors": True,
        "tools": [
            {"name": "Bash", "is_error": False, "sequence": 0,
             "params_summary": {"command": "mkdir -p /tmp/rec"}, "result_summary": "ok"},
            {"name": "Read", "is_error": False, "sequence": 1,
             "params_summary": {"file_path": "/tmp/rec/a"}, "result_summary": "ok"},
            {"name": "Bash", "is_error": True, "sequence": 2,
             "params_summary": {"command": "mkdir -p /tmp/rec"},
             "result_summary": "boom: read-only"},
            {"name": "Read", "is_error": False, "sequence": 3,
             "params_summary": {"file_path": "/tmp/rec/b"}, "result_summary": "ok"},
        ],
        "error_tools": [],
        "signals": [],
    }


def test_a_sequence_with_no_recovery_in_it_is_not_emittable():
    """Clause 1 (#1181). A sequence's `has_error_recovery` is derived from its own
    n-gram, so a pattern flagged false contains no failure at all — the item's
    falsifier keys were `seq-2-calendar-events-email-recent` (157 sessions, 6 of
    6 steps OK) and `seq-2-write-read` (209 sessions, 6 of 6 OK), which the old
    exemption defended as "losing those would mean losing failures". 672 of the
    780 actionable candidate keys on 2026-09-16 read false in their own front
    matter; at the runbook's 5 patterns a night that is ~156 nights of hand
    adjudication to reach "no skill here" on every one of them."""
    assert mt.is_emittable({"type": "sequence", "ngram_size": 3,
                            "has_error_recovery": False}) is False
    assert mt.is_emittable({"type": "sequence", "ngram_size": 3,
                            "has_error_recovery": True}) is True
    # Absent is not False: a pattern that reaches the gate with no flag stays
    # emittable, which is what keeps an `error` dict (clause 3) and a hand-built
    # or legacy sequence dict from being suppressed by absence.
    assert mt.is_emittable({"type": "sequence", "ngram_size": 3}) is True


def test_a_mined_error_pattern_is_still_emittable():
    """Clause 3 (#1181). `mine_error_patterns` builds its dicts from tool_name,
    error_type, params_signature, sessions, examples, dates, total_calls,
    first_seen and last_seen — never `has_error_recovery` — so the gate tests the
    flag with `is False`. A falsy test would have suppressed the error table
    alongside the sequences it was meant to, and the table is the one thing the
    miner exists to find."""
    pattern = mined_error_pattern()
    assert pattern["type"] == "error"
    assert "has_error_recovery" not in pattern, "the miner added the flag; the falsy trap is live"
    assert mt.is_emittable(pattern) is True
    assert mt.is_emittable(
        {"type": "error", "tool_name": "Bash", "error_type": "not_found"}) is True


def test_a_mining_call_writes_only_the_recovering_sequence(tmp_path):
    """Clause 2 (#1181), through the miner rather than a hand-built dict.

    Three things hold at once, and each fails under a different half-fix:
      * `write_candidate_file` refuses a false-flagged sequence on its own, so a
        caller that skips `emit_candidates` cannot re-open the hole;
      * `emit_candidates` writes exactly one file per recovering key mined in the
        same call — filtering the returned list without filtering `sequence_keys`
        (or the reverse) trips the one-key-one-file assertion inside it (#1131);
      * an `error` pattern mined alongside them still gets its file.
    """
    rows = [recovering_traj(f"rec-{i}") for i in (1, 2, 3)]
    seqs = mt.mine_sequence_patterns(rows, threshold=2)
    refusing = [p for p in seqs if p["has_error_recovery"] is False]
    recovering = [p for p in seqs if p["has_error_recovery"] is True]
    assert refusing and recovering, "the fixture must mine both kinds of sequence"

    nowhere = tmp_path / "must-stay-empty"
    for pattern in refusing:
        assert mt.write_candidate_file(pattern, nowhere) is None, (
            f"{mt.candidate_pattern_key(pattern)}: writer opened the hole again")
    assert not nowhere.exists() or list(nowhere.iterdir()) == []

    out = tmp_path / "cands"
    written = mt.emit_candidates(seqs + [mined_error_pattern()], out)
    seq_paths = [p for p in written if p.name.startswith("candidate-seq-")]
    seq_files = sorted(p.name for p in out.glob("candidate-seq-*.md"))

    assert len(seq_files) == len(recovering) == len(seq_paths), (
        f"{len(seq_files)} files for {len(recovering)} recovering keys")
    assert {pattern_field_of(p) for p in seq_paths} == {
        mt.candidate_pattern_key(p) for p in recovering}
    assert len([p for p in written if not p.name.startswith("candidate-seq-")]) == 1, (
        "the error pattern mined alongside them lost its file")


def test_the_emission_gate_docstring_names_the_flag_that_survives():
    """Clause 4 (#1181). The parenthetical that justified the exemption is the
    reason a later reader would widen the gate back open, and a docstring is read
    far more often than it is written — so the corrected claim is pinned as text,
    the same way the #561 guard pins a function name it resolves."""
    doc = mt.is_emittable.__doc__ or ""
    assert "losing those would mean losing failures" not in doc
    assert "has_error_recovery: true" in doc, (
        "the docstring must say the surviving sequences are the ones flagged true")


def test_write_candidate_file_writes_nothing_for_the_key_set_class(tmp_path):
    """The guarantee lives in the writer, so no caller can re-open the hole."""
    out = tmp_path / "cands"
    out.mkdir()
    assert mt.write_candidate_file(
        success_pattern("file_path_limit_signature"), out) is None
    assert list(out.iterdir()) == []


def test_a_day_of_emission_carries_neither_defect(tmp_path):
    """The acceptance check, end to end: after mining, no written candidate
    file has a `pattern:` key containing `truncated`, and none ends in
    `_signature` (93 of 103 keys, and 20 of the top 20, on 2026-09-08)."""
    rows = [unkeyed_traj(f"s{i}") for i in (1, 2)]
    for i in (1, 2):
        rows.append({
            "session_key": f"b{i}", "timestamp": "2026-09-08T18:00:00Z",
            "tool_count": 2, "error_count": 2, "has_errors": True,
            "tools": [
                {"name": "Bash", "is_error": True, "error_source": "protocol",
                 "sequence": 0, "params_summary": {"command": "pytest tests/"},
                 "result_summary": "boom"},
                {"name": "Read", "is_error": False, "sequence": 1,
                 "params_summary": {"file_path": "/x", "limit": 100},
                 "result_summary": "ok"},
            ],
            "error_tools": [{"name": "Bash", "sequence": 0, "error_type": "not_found",
                             "error_source": "protocol",
                             "params_summary": {"command": "pytest tests/"}}],
            "signals": [],
        })
    patterns = (mt.mine_error_patterns(rows, threshold=2)
                + mt.mine_success_patterns(rows, threshold=2)
                + mt.mine_sequence_patterns(rows, threshold=2))
    written = mt.emit_candidates(patterns, tmp_path)
    keys = []
    for path in written:
        body = path.read_text(encoding="utf-8")
        keys.append(re.search(r"^pattern: (.+)$", body, re.MULTILINE).group(1))
    assert keys, "the run emitted nothing, so the assertions below are vacuous"
    assert not [k for k in keys if "truncated" in k.lower()]
    assert not [k for k in keys if k.endswith("_signature")]


# ── a sequence's file must be injective on its key (backlog #1131) ───────────
#
# `candidate_pattern_key()` ran the n-gram through `slugify`, which cuts at 50
# characters, and `write_candidate_file()` then cut the finished key at the same
# 50 for the filename. Two distinct n-grams sharing their first 50 slug
# characters therefore got one filename and one `pattern:` value; the second
# write silently replaced the first, and which one survived was dict iteration
# order. Measured on the live 7-day corpus on 2026-09-15: 1674 mined sequence
# patterns returned 1674 paths, 1671 of them distinct, 1671 files on disk — three
# names carrying two patterns each — and reversing the input list flipped the
# `pattern:` field inside `candidate-seq-3-backlog-write-task-bash-fs-…`. The run
# also printed 1234 `Written:` lines over 1169 files, because the same duplicated
# list feeds the summary and INDEX.md.

ALIASED_HEAD = ("automod_gate_wait", "backlog_write_task")
ALIASED_TAILS = ("automod_land", "automod_abort")


def aliased_traj(session_key):
    """One session carrying both trigrams whose *filenames* aliased.

    Their keys under the filing rule were 55 and 56 characters — `…-automod-land`
    and `…-automod-abort`, distinct — and the alias came from the filename re-slug:
    `write_candidate_file` cut the finished key at 50 characters, and the first 50
    of both are `seq-3-automod-gate-wait-backlog-write-task-automod`, so the pair
    wrote one file. The n-grams' own slugs are 49 and 50 characters, which is why
    the cap has to be measured on the whole key: prepending `seq-3-` costs 6, so
    even the 49-character one is past the cut as a key.
    """
    names = [ALIASED_HEAD[0], ALIASED_HEAD[1], ALIASED_TAILS[0],
             ALIASED_HEAD[0], ALIASED_HEAD[1], ALIASED_TAILS[1]]
    return {
        "session_key": session_key,
        "timestamp": "2026-09-15T10:10:10Z",
        "tool_count": len(names), "error_count": 0, "has_errors": False,
        "tools": [{"name": n, "is_error": False, "sequence": i,
                   "params_summary": {}, "result_summary": "ok"}
                  for i, n in enumerate(names)],
        "error_tools": [], "signals": [], "session_class": "interactive",
    }


def aliased_pair():
    """The two sequence patterns whose *filename* aliased, mined through the real
    miner. Their keys were already distinct at filing — asserted by
    `test_the_aliased_pair_shares_one_plain_slug_but_not_one_candidate_name`; the
    pair whose keys aliased is the 5-gram fixture in
    `test_two_ngrams_differing_only_past_the_cap_get_distinct_keys`."""
    rows = [aliased_traj("alias-a"), aliased_traj("alias-b")]
    pair = [p for p in mt.mine_sequence_patterns(rows, threshold=2)
            if tuple(p["sequence"][:2]) == ALIASED_HEAD
            and p["sequence"][-1] in ALIASED_TAILS]
    assert len(pair) == 2, "the fixture must mine both aliased n-grams"
    return pair


def recovering(patterns: list[dict]) -> list[dict]:
    """The same patterns with the emission gate's one input set to `True`.

    The cap/filename tests below pin *which file a pattern is written to*, and
    since #1181 a sequence reaches a file at all only when it is flagged
    `has_error_recovery: true`. Left as the miner flags them — these fixtures
    carry no failing step, so they are flagged `False` — the writer refuses them
    and every `len(written) == 2` under this helper passes on an empty list,
    which is a test that can no longer fail. Setting the flag is the smallest
    change that keeps those assertions live; the flag's own behaviour is pinned
    by `test_a_sequence_with_no_recovery_in_it_is_not_emittable` and
    `test_emit_candidates_writes_no_file_for_a_non_recovery_sequence`."""
    return [{**p, "has_error_recovery": True} for p in patterns]


def pattern_field_of(path: Path) -> str:
    return re.search(r"^pattern: (.+)$", path.read_text(encoding="utf-8"),
                     re.MULTILINE).group(1)


def test_the_aliased_pair_shares_one_plain_slug_but_not_one_candidate_name():
    """Clause 1's mechanism and clause 2's prefix rule, through the miner rather
    than a hand-built dict.

    Measured at filing, these two n-grams already had *distinct keys* —
    `seq-3-…-automod-land` (55 characters) and `seq-3-…-automod-abort` (56) — and
    what aliased was the *filename*: `write_candidate_file` ran the finished key
    through `slugify`, whose 50-character cut left both on
    `seq-3-automod-gate-wait-backlog-write-task-automod`, so they wrote one file.
    The pair that aliased one level deeper, at the key itself, is the 5-gram fixture
    in `test_two_ngrams_differing_only_past_the_cap_get_distinct_keys`.

    Three things have to hold at once here, and each fails under a different
    wrong fix:
      * the plain `slugify` of the two keys is still the same 50 characters — the
        cut is unchanged and this pair still straddles it (raising `SLUG_CAP`
        breaks this line);
      * `slug_for` no longer agrees with it, so the two keys get two filenames
        (reverting the filename site to plain `slugify` breaks this line);
      * the key *with* the suffix is the one from the 49-character n-gram slug too.
        `… → automod_land`'s n-gram slug is 49 characters — under the cap as
        measured on the n-gram alone, and still cut, once `seq-3-` is prepended.
        Measuring the cap on the whole key is what disambiguates it (measuring on
        the n-gram string breaks this line).
    """
    cut = "seq-3-automod-gate-wait-backlog-write-task-automod"
    key_a, key_b = (mt.candidate_pattern_key(p) for p in aliased_pair())
    assert key_a != key_b, "two n-grams share one pattern key"
    assert mt.slugify(key_a) == mt.slugify(key_b) == cut, (
        "the pair no longer straddles the 50-character cut, so it pins nothing")
    assert mt.slug_for(key_a) != mt.slug_for(key_b), "one filename for two keys"
    suffix = re.compile(rf"^{re.escape(cut)}-[0-9a-f]{{8}}$")
    assert all(suffix.match(k) for k in (key_a, key_b)), (
        f"{key_a} / {key_b}: a key whose n-gram slug is 49 characters ("
        "`automod_land`, one byte under the cap on its own) must still be "
        "disambiguated, because `seq-3-` spends 6 of the 50")


def test_two_ngrams_that_alias_under_the_cap_get_one_file_each(tmp_path):
    """Clause 1 for the mechanism: two patterns, two files, two distinct
    `pattern:` fields — where pre-fix it was one file holding whichever n-gram
    the dict happened to yield last."""
    written = mt.emit_candidates(recovering(aliased_pair()), tmp_path)
    assert len(written) == 2, [p.name for p in written]
    assert len(set(written)) == 2, "the returned list named one path twice"

    fields = {pattern_field_of(p) for p in written}
    assert fields == {mt.candidate_pattern_key(p) for p in aliased_pair()}
    assert len(fields) == 2
    assert len(list(tmp_path.glob("candidate-seq-*.md"))) == 2


def test_reversing_the_pattern_list_writes_the_same_bytes(tmp_path):
    """Clause 3: the surviving evidence must not be decided by iteration order.
    Every file from the forward emission exists from the reversed one and is
    byte-identical to it — a name-only fix would pass the first assertion and
    leave the last writer winning inside the file."""
    pair = recovering(aliased_pair())
    forward, backward = tmp_path / "fwd", tmp_path / "rev"
    f_written = mt.emit_candidates(pair, forward)
    r_written = mt.emit_candidates(list(reversed(pair)), backward)

    assert sorted(p.name for p in f_written) == sorted(p.name for p in r_written)
    by_name = {p.name: p for p in r_written}
    for path in f_written:
        assert path.read_bytes() == by_name[path.name].read_bytes(), path.name


def test_two_ngrams_differing_only_past_the_cap_get_distinct_keys(tmp_path):
    """Clause 2's key-level case: two 5-grams whose n-grams share their first 50
    slug characters and differ only in the final tool. Pre-fix both returned
    `seq-5-backlog-write-task-bash-fs-automod-gate-wait-autom` — one key, so one
    `pattern:` field and one verdict-ledger row for two different loops, and a
    filename-only hash could not separate them because their keys were equal.

    The cut is on the whole key, so the shared prefix the two keys keep is the
    50-character `seq-5-backlog-write-task-bash-fs-automod-gate-wait`, six
    characters shorter than the shared n-gram slug: the `seq-5-` prefix spends
    part of the cap, which is the only measurement that agrees with what
    `write_candidate_file` and the ledger actually consume.
    """
    head = "backlog_write_task → bash:fs → automod_gate_wait → automod_land"
    five_a = {"type": "sequence", "ngram_size": 5, "sessions": {"s1", "s2"},
              "sequence": tuple(head.split(" → ")) + ("backlog_tasks",),
              "sequence_str": f"{head} → backlog_tasks",
              "has_error_recovery": False, "first_seen": "2026-09-14",
              "last_seen": "2026-09-15", "examples": []}
    five_b = {**five_a,
              "sequence": tuple(head.split(" → ")) + ("research_stats",),
              "sequence_str": f"{head} → research_stats"}

    shared = "seq-5-backlog-write-task-bash-fs-automod-gate-wait"
    assert len(shared) == mt.SLUG_CAP, "the fixture no longer straddles the cap"
    key_a, key_b = (mt.candidate_pattern_key(p) for p in (five_a, five_b))
    assert key_a != key_b, "two n-grams share one pattern key"
    assert (key_a[:mt.SLUG_CAP], key_b[:mt.SLUG_CAP]) == (shared, shared)
    assert key_a != key_a[:mt.SLUG_CAP], "the key itself must survive past the cap"

    written = mt.emit_candidates(recovering([five_a, five_b]), tmp_path)
    assert len(written) == 2, [p.name for p in written]
    assert len({p.name for p in written}) == 2, "two keys, one filename"
    assert {pattern_field_of(p) for p in written} == {key_a, key_b}

    # Re-emitting the same pair over the same directory must still be two files: a
    # night that wrote one then the other is how the alias used to hide.
    mt.emit_candidates([five_a, five_b], tmp_path)
    assert len(list(tmp_path.glob("candidate-seq-*.md"))) == 2


def test_a_key_under_the_cap_is_unchanged_by_the_disambiguator():
    """The widening is scoped to the cap, and this is the unit half of that
    scoping: below the cap the key rule is the byte-identical identity
    `seq-{n}-{slugify(sequence_str)}`, which is the only reason the rows the verdict
    ledger already stores still mean what they meant (the ledger half is pinned in
    `test_skill_verdicts.py::test_every_stored_sequence_verdict_still_resolves…`).

    Asserted as the identity over a spread of shapes rather than on one example:
    the claim is about every key under the cap, and a single n-gram cannot show a
    rule that only misbehaves on, say, an uppercase or punctuation-heavy name.
    """
    shapes = ["bash:fs → backlog_write_task",
              "Bash → Read → Write",
              "mcp:vault → bash:cmd:date → grep",
              "a → b → c → d → e",
              "weird!!name → x"]
    checked = 0
    for seq in shapes:
        n = seq.count(" → ") + 1
        old = f"seq-{n}-{mt.slugify(seq)}"
        if len(mt.slugify(old)) < mt.SLUG_CAP:
            checked += 1
            assert mt.sequence_pattern_key(n, seq) == old, (
                f"an under-cap key changed meaning: {old!r} -> "
                f"{mt.sequence_pattern_key(n, seq)!r}")
    assert checked >= 3, (
        f"only {checked} of {len(shapes)} shapes are under the cap, so the identity "
        "above is checking almost nothing")

    short = {"type": "sequence", "ngram_size": 2,
             "sessions": {"s1", "s2"},
             "sequence": ("bash:fs", "backlog_write_task"),
             "sequence_str": "bash:fs → backlog_write_task",
             "has_error_recovery": False, "first_seen": "2026-09-01",
             "last_seen": "2026-09-09", "examples": []}
    key = mt.candidate_pattern_key(short)
    assert key == "seq-2-bash-fs-backlog-write-task"
    assert mt.slug_for(key) == mt.slugify(key), "no hash suffix below the cap"


def test_the_nightly_run_reports_one_line_per_file_it_wrote(tmp_path):
    """Clause 5, over the command the nightly actually runs. `Written:` lines and
    INDEX.md's candidate count are counts of files, so they must equal the files
    on disk; pre-fix the same list carried one entry per write attempt — 1234
    lines over 1169 files. `Suppressed as non-skill candidates` came out of the
    same arithmetic — `len(all_patterns) - len(written)` — and agreed with the gate
    on 2026-09-15 (1482 - 1234 = 248, and 248 patterns refused by `is_emittable`)
    only because the old list held one entry per write attempt; it was never a
    measurement of the gate, and once the list counts files the subtraction starts
    reporting an aliased pattern as suppressed."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [aliased_traj(f"alias-{i}") for i in (1, 2, 3)]
    # The pair that reaches a file is a *recovering* pair. Since #1181 an
    # n-gram with no failing step in it is refused by the emission gate, so
    # `aliased_traj` alone — 6 successful steps, nothing to recover from — mines
    # only suppressed patterns, and `main()` would report zero sequence files
    # while still honouring the arithmetic this test pins. The aliased pair stays
    # in the fixture for the Suppressed count; `recovering_traj` supplies the
    # patterns that actually reach a file.
    rows += [recovering_traj(f"rec-{i}") for i in (1, 2, 3)]
    # A refused-by-gate success pattern: `_signature` keys are counted as mined
    # and never emitted, which is what the Suppressed line is for.
    for i in (1, 2, 3):
        rows.append({
            "session_key": f"succ-{i}", "session_class": "interactive",
            "timestamp": "2026-09-15T10:10:10Z",
            "tool_count": 1, "error_count": 0, "has_errors": False,
            "tools": [{"name": "Read", "is_error": False, "sequence": 0,
                       "params_summary": {"file_path": "/x", "limit": 100},
                       "result_summary": "ok"}],
            "error_tools": [], "signals": []})
    # Two error variants of one (tool, error_type): distinct mined patterns, one
    # candidate file (#515's coarse key, unlanded). The report must count the
    # file once.
    for i in (1, 2, 3):
        rows.append(error_traj(f"err-{i}", "Bash", "not_found", "protocol",
                               params={"command": "pytest tests/"}))
        rows.append(error_traj(f"err-{i}", "Bash", "not_found", "protocol",
                               params={"path": "/missing"}))
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = tmp_path / "cands"
    proc = run_miner(corpus, out, extra_args=("--include-machine",))
    assert proc.returncode == 0, proc.stderr
    report = proc.stdout + proc.stderr

    written_lines = [l for l in report.splitlines()
                     if l.strip().startswith("Written:")]
    files_on_disk = list(out.glob("candidate-*.md"))
    assert len(written_lines) == len(files_on_disk), (
        f"{len(written_lines)} Written: lines for {len(files_on_disk)} files")

    index = (out / "INDEX.md").read_text(encoding="utf-8")
    index_total = int(re.search(r"\*\*Total candidates:\*\* (\d+)", index).group(1))
    assert index_total == len(files_on_disk), index_total

    mined = (mt.mine_error_patterns(rows, threshold=2)
             + mt.mine_success_patterns(rows, threshold=2)
             # The rule `main()` applies, not the CLI's flag: sequences mine at
             # `max(3, threshold)`, so anything else compares two different sets.
             + mt.mine_sequence_patterns(rows, threshold=max(3, 2)))
    refused = [p for p in mined if not mt.is_emittable(p)]
    assert refused, "the corpus must contain a pattern the gate refuses"
    suppressed = int(re.search(
        r"Suppressed as non-skill candidates: (\d+)", report).group(1))
    assert suppressed == len(refused), (suppressed, len(refused))

    # Sequence files counted against the patterns that earned them. Comparing a
    # glob with its own filenames cannot fail, so the denominator is the mined key
    # set: on this corpus the base rule wrote ONE file for the two aliased
    # n-grams, which fails the equality below.
    seq_keys = {mt.candidate_pattern_key(p) for p in mined
                if p["type"] == "sequence" and mt.is_emittable(p)}
    assert seq_keys, "the fixture mined no sequence pattern, so this proves nothing"
    seq_files = [p for p in files_on_disk if p.name.startswith("candidate-seq-")]
    assert len(seq_files) == len(seq_keys), (
        f"{len(seq_files)} candidate-seq files for {len(seq_keys)} distinct sequence "
        "keys: an n-gram is still sharing a file with another")


def test_the_live_corpus_emits_one_file_per_sequence_pattern(tmp_path):
    """The acceptance check itself, over the corpus the nightly reads: 7 days,
    every class, threshold 2.

    #1131's invariant is that one n-gram owns one file; #1181 narrowed *which*
    n-grams are admitted, so the denominator here is the mined sequences that
    clear the emission gate, and the refused ones are asserted to write nothing
    at all — otherwise "one file per admitted key" would also be satisfied by a
    rule that admitted nothing. Measured 2026-09-17 over this window: 1718 mined
    sequence patterns, 317 flagged `has_error_recovery: true`, 1401 refused, and
    317 files."""
    # `LIVE_CORPUS` is defined further down this file, against the real data root
    # rather than the checkout — `_pipeline/` is gitignored, so a worktree-relative
    # path reads as absent forever. Asserted, never skipped: clause 1 *is* a count
    # over this corpus, and a check that can pass by not finding the data is not a
    # check. Read-only; the emitted files go into `tmp_path`.
    assert LIVE_CORPUS.is_dir(), f"live corpus absent: {LIVE_CORPUS}"
    rows = mt.load_trajectories(days=7, agent_filter="all", exclude_machine=False)
    assert rows, f"live corpus at {LIVE_CORPUS} loaded no rows"
    seqs = mt.mine_sequence_patterns(rows, threshold=2)
    assert seqs, "no sequence patterns mined, so the assertions below are vacuous"
    admitted = [p for p in seqs if mt.is_emittable(p)]
    refused = [p for p in seqs if not mt.is_emittable(p)]
    assert admitted and refused, (
        f"the live window must hold both kinds — {len(admitted)} admitted, "
        f"{len(refused)} refused — or this test compares one empty set with another")

    target = tmp_path / "candidates"
    paths = mt.emit_candidates(seqs, target)
    seq_paths = [p for p in paths if p.name.startswith("candidate-seq-")]
    seq_files = list(target.glob("candidate-seq-*.md"))

    assert len(seq_paths) == len(set(seq_paths)), "a path appeared twice"
    assert len(seq_files) == len(admitted) == len(seq_paths), (
        f"{len(seq_files)} files for {len(admitted)} admitted patterns")
    fields = [pattern_field_of(p) for p in seq_files]
    assert len(set(fields)) == len(fields), "two files share a pattern: field"

    # The refused half, stated over the corpus rather than over a fixture: a key
    # the gate refused must not appear as a `pattern:` field in any file that run
    # wrote. A rule that admitted every sequence and a rule that admitted none
    # both keep the equality above plausible; this is what separates them.
    refused_keys = {mt.candidate_pattern_key(p) for p in refused}
    assert refused_keys, "the corpus mined no refused sequence pattern"
    assert not (refused_keys & set(fields)), (
        f"{sorted(refused_keys & set(fields))[:3]}: a sequence with no failing "
        "step in it still reached a candidate file")


# ── live-data guard: the sweep must not survive in regenerated data ──────────
#
# #392: the extractor used to derive `has_errors` / `error_tools` by regexing a
# tool's result *text*, so a `Read` of any file containing the word `Error`
# became a "failed step", and `stats.is_error` — the harness's own answer to
# "did this call fail", present on every tool message — was never read. The
# mining chain is ordered "error trajectories first", so the phantom flags
# steered skill mining at prose-reading sessions and away from real failures.
# The derivation now reads the persisted flag (`extract-trajectories.py:355`);
# the sweep survives only as `output_mentions_errors`, which promotes nothing.
#
# The unit pins above prove the CODE cannot promote the sweep. These two prove
# the DATA on this machine was written by that code. The regression they catch
# is the one the unit tests cannot: a change that re-introduces the old
# derivation and then re-extracts — every unit pin stays green while the
# buckets the miner actually reads go back to phantom failures.

ERROR_REGENERATED_FROM = "2026-09-05"   # start of the window re-extracted for #392
KEYWORD_ONLY_SOURCE = "semantic"        # the pre-fix `error_source` value
CORROBORATED_SOURCES = mt.CORROBORATED_ERROR_SOURCES


def _live_buckets(since=None):
    live = _ROOT / "_pipeline" / "trajectories"
    if not live.exists():
        pytest.skip("no trajectories dir on this machine")
    return [p for p in sorted(live.glob("*.jsonl"))
            if since is None or p.stem >= since]


def _entries(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def test_no_regenerated_bucket_promotes_the_keyword_sweep():
    """Read-only. Acceptance #392: `error_source: "semantic"`-only flags across
    the regenerated window reach 0 — measured at 196 of 234 before the fix."""
    offending = {}
    for path in _live_buckets(since=ERROR_REGENERATED_FROM):
        hits = sum(1 for entry in _entries(path)
                   for flag in (entry.get("error_tools") or [])
                   if flag.get("error_source") == KEYWORD_ONLY_SOURCE)
        if hits:
            offending[path.name] = hits
    assert not offending, (
        "result-text keyword matching is being promoted as a failure again "
        f"(buckets still carrying `error_source: {KEYWORD_ONLY_SOURCE!r}`): {offending}"
    )


def test_the_regenerated_window_flags_only_persisted_flag_failures():
    """Read-only. #392's reproduction clause, machine-checked: 2026-09-05
    carried 30 `error_tools` — 7 / 5 / 18 across three sessions — against 6 real
    `stats.is_error` failures (3 in iv5174, 2 in iv2314, 1 in ivbf4f). ~20 of
    the 30 were `Read`/`Grep` output quoting source code that merely contains
    the word `error` or `Warning`, and all three sessions were flagged."""
    buckets = {p.stem: p for p in _live_buckets()}
    path = buckets.get(ERROR_REGENERATED_FROM)
    if path is None:
        pytest.skip(f"{ERROR_REGENERATED_FROM} bucket not on this machine")
    expected = {"iv5174": 3, "iv2314": 2, "ivbf4f": 1}
    found: dict[str, int] = {}
    uncorroborated: list[str] = []
    for entry in _entries(path):
        key = str(entry.get("session_key", ""))
        flags = entry.get("error_tools") or []
        uncorroborated += [f"{key}:{f.get('name')}={f.get('error_source')}"
                           for f in flags
                           if f.get("error_source") not in CORROBORATED_SOURCES]
        if not flags and entry.get("has_errors"):
            uncorroborated.append(f"{key}: has_errors with no error_tools")
        for tag, want in expected.items():
            if tag in key:
                found[tag] = len(flags)
    assert not uncorroborated, (
        f"`error_tools` entries with no corroborating signal: {uncorroborated}")
    assert found == expected, (
        f"2026-09-05 flagged steps per session are {found}, expected {expected} "
        "— the extractor is not deriving them from `stats.is_error`")
    assert sum(len(e.get("error_tools") or []) for e in _entries(path)) == 6


# ── session class: the frequency gate must not rank the loop (#493) ──────────
#
# The gate qualifies a pattern on *distinct sessions*, so a corpus that is mostly
# the loop's own traffic ranks the loop. Measured 2026-09-12: 938 of 1083 corpus
# sessions across 09-02→09-12 were machine sessions (86.6%, monotone per day),
# and 10,611 of 11,813 `Sessions Affected` bullets in that day's candidates were
# machine sessions — `candidate-automod-gate-logic-20260912.md` and
# `candidate-bash-logic-20260912.md` are 10/10 loop traffic.
#
# The classifier that was supposed to see this keyed on the filename
# (`path.stem.startswith("autonomy_")`), and 0 of the live session files match it
# — the loop renamed itself to `youtubed_*` / `autocode_*` / `autotriage_*` /
# `benchmine_*` around 09-09 — so every row was `agent_id: "lloyd"` and the class
# was unknowable downstream. The class now comes from fields every session JSON
# already carries: `platform`, `source`, `inner_voice`.
#
# These fixtures are the part a live-data test cannot prove: two session files
# with the SAME filename stem, one human and one machine. Anything that reads the
# name gives the same answer for both.

CLASS_STEM = "20260912_101010_autocode_beef"   # loop-shaped, used for both classes

# (platform, inner_voice, expected class). `interactive` appears exactly once, on
# the clause-2 condition; every other row is a class that is NOT human-initiated
# work. `browser` stays its own class rather than folding into `inner-voice` even
# though all 50 browser sessions in the store carry `inner_voice: true` (measured
# 2026-09-14), because
# #493's open scope question is precisely whether browser/inner-voice turns join
# the interactive pool — that decision has to be movable in one line here without
# re-extracting the corpus, and collapsing two platforms into one class would
# destroy the signal it needs.
SESSION_CLASS_TABLE = [
    ("mission-control", False, "interactive"),
    ("mission-control", True, "inner-voice"),
    ("browser", True, "browser"),
    ("browser", False, "browser"),
    ("worker", False, "worker"),
    ("worker", True, "worker"),
    ("autonomy", False, "autonomy"),
    ("autonomy", True, "autonomy"),
    ("e2e-harness", False, "smoke"),
    (None, False, "unknown"),
]


def write_class_session(dir_, stem, platform, inner_voice=False, source=None):
    """Write a session JSON with the fields the classifier reads.

    One corroborated `Bash` failure, so `parse_session` returns a row rather than
    None for a tool-less session.
    """
    d = Path(dir_)
    d.mkdir(parents=True, exist_ok=True)
    body = {
        "session_id": stem,
        "session_start": "2026-09-12T10:10:10Z",
        "inner_voice": inner_voice,
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "call_0", "function": {
                    "name": "Bash",
                    "arguments": json.dumps({"command": "pytest tests/"})}}]},
            {"role": "tool", "tool_call_id": "call_0",
             "content": [{"type": "text", "text": "boom: no such file"}],
             "stats": {"result_chars": 18, "is_error": True}},
        ],
    }
    if platform is not None:
        body["platform"] = platform
    if source is not None:
        body["source"] = source
    path = d / f"{stem}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# ── clause 1: the class cannot come from the filename ────────────────────────

def test_a_worker_session_is_not_interactive_under_a_loop_shaped_stem(tmp_path):
    """Same stem, two classes — the discriminating case for `path.stem`."""
    worker = et.parse_session(write_class_session(
        tmp_path / "a", CLASS_STEM, "worker", source="autocode"))
    human = et.parse_session(write_class_session(
        tmp_path / "b", CLASS_STEM, "mission-control"))
    assert worker["session_class"] == "worker"
    assert human["session_class"] == "interactive"
    # `agent_id` is still derived from the stem and is identical for both, so the
    # class below it provably was not.
    assert worker["agent_id"] == human["agent_id"] == "lloyd"


def test_an_autonomy_prefixed_stem_does_not_make_a_session_non_interactive(tmp_path):
    """The old rule's own positive case, reversed: the filename says autonomy,
    the session JSON says a human drove it from Mission Control."""
    traj = et.parse_session(write_class_session(
        tmp_path, "autonomy_task68", "mission-control"))
    assert traj["session_class"] == "interactive"
    assert traj["agent_id"] == "autonomy"   # legacy field keeps its old meaning


def test_classify_session_needs_no_filename_at_all():
    """The classifier's input is the parsed JSON, so there is no path to read."""
    assert et.classify_session({"platform": "worker"}) == "worker"
    assert et.classify_session({}) == "unknown"


# ── clause 2: interactive == mission-control and not inner_voice ─────────────

@pytest.mark.parametrize("platform,inner_voice,expected", SESSION_CLASS_TABLE)
def test_the_class_comes_from_platform_and_inner_voice(platform, inner_voice, expected):
    assert et.classify_session(
        {"platform": platform, "inner_voice": inner_voice}) == expected


def test_interactive_is_exactly_mission_control_without_inner_voice():
    """Clause 2 in both directions, derived from the classifier and nothing else:
    every platform the backend records (plus one it does not) crossed with every
    truthiness outcome of `inner_voice`, and exactly one pair is interactive.

    Not read out of SESSION_CLASS_TABLE — that table is hand-written, so comparing
    it against a literal set would pass with a classifier that called every session
    interactive. A classifier that dropped the `inner_voice` condition would fail
    here, and would fail the live-store oracle below for the same reason.
    """
    platforms = ["mission-control", "worker", "autonomy", "browser",
                 "e2e-harness", "slack", None]
    inner_voice_values = [True, False, 1, 0, "", None, "true"]
    interactive = {(p, bool(iv))
                   for p in platforms for iv in inner_voice_values
                   if et.classify_session({"platform": p,
                                           "inner_voice": iv})
                   == et.INTERACTIVE_CLASS}
    assert interactive == {("mission-control", False)}


def test_the_classified_session_records_its_source_producer(tmp_path):
    """`source` names the loop that ran the session (`autotriage`, `autocode`,
    `autonomy-task:68`), which is what makes a dropped session attributable."""
    traj = et.parse_session(write_class_session(
        tmp_path, CLASS_STEM, "worker", source="autotriage"))
    assert traj["session_source"] == "autotriage"


# ── clause 3: the exclusion is one flag over one corpus ──────────────────────

MACHINE_ROWS = [("i1", "interactive"), ("i2", "interactive"),
                ("w1", "worker"), ("w2", "worker"), ("a1", "autonomy"),
                ("b1", "browser"), ("v1", "inner-voice")]

CORPUS_BUCKET = "2026-09-12.jsonl"


def write_traj_corpus(dir_, classed_rows):
    """Write a trajectory bucket the way the extractor writes one."""
    d = Path(dir_)
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"session_key": key, "agent_id": "lloyd",
                         "session_class": cls,
                         "timestamp": "2026-09-12T10:10:10Z",
                         "tool_count": 0, "error_count": 0,
                         "has_errors": False, "tools": [], "error_tools": [],
                         "signals": []})
             for key, cls in classed_rows]
    (d / CORPUS_BUCKET).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_load_trajectories_keeps_only_interactive_rows_when_the_exclusion_is_on(
        tmp_path, monkeypatch):
    corpus = write_traj_corpus(tmp_path / "corpus", MACHINE_ROWS)
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    kept = mt.load_trajectories(days=9999, agent_filter="all")
    assert [t["session_key"] for t in kept] == ["i1", "i2"]


def test_load_trajectories_returns_every_row_when_the_exclusion_is_off(
        tmp_path, monkeypatch):
    """Same corpus, one flag — the exclusion is a switch, not a new corpus."""
    corpus = write_traj_corpus(tmp_path / "corpus", MACHINE_ROWS)
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    kept = mt.load_trajectories(days=9999, agent_filter="all",
                                exclude_machine=False)
    assert [t["session_key"] for t in kept] == [k for k, _ in MACHINE_ROWS]


def test_the_exclusion_counts_what_it_dropped_by_class(tmp_path, monkeypatch):
    corpus = write_traj_corpus(tmp_path / "corpus", MACHINE_ROWS)
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert counts["dropped"] == {"worker": 2, "autonomy": 1, "browser": 1,
                                 "inner-voice": 1}
    assert counts["kept"] == {"interactive": 2}


def test_a_row_with_no_session_class_is_not_treated_as_interactive(
        tmp_path, monkeypatch):
    """The corpus on disk predates the field, so absence must not read as
    human-initiated work — and it must be attributable, not silent.

    The store is redirected to an empty directory: this row's last defence is that
    nothing behind it answers, which is only the case on a machine with no session
    file named `old1.json`.
    """
    corpus = write_traj_corpus(tmp_path / "corpus", [("old1", None)])
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    empty_store = tmp_path / "emptystore"
    empty_store.mkdir()
    monkeypatch.setattr(mt, "SESSION_STORE_DIR", empty_store)
    counts: dict = {}
    kept = mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert kept == []
    assert counts["dropped"] == {"uncoded": 1}


# ── the corpus↔store join: which side wins ──────────────────────────────────
#
# Every live corpus row was written without `session_class` (0 of 1,411 carry it as
# at 2026-09-14), so the join is not a fallback for old data — it is the path the
# nightly runs on today, and the only thing standing between the exclusion and a
# blank corpus. These tests redirect `mt.SESSION_STORE_DIR`, because the branch
# that decides whether a legacy row is human work needs a store it can be told
# about.

def isolated_store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mt, "SESSION_STORE_DIR", store)
    return store


def test_a_class_less_row_is_rescued_as_interactive_by_the_store(
        tmp_path, monkeypatch):
    """The positive branch of the join: a legacy row with no emitted class is kept
    when the session JSON behind it says a human drove it from Mission Control.
    Without it the exclusion would blank the 7-day window — every live row is
    class-less — and emit nothing, which is clause 6's forbidden outcome."""
    store = isolated_store(tmp_path, monkeypatch)
    write_class_session(store, "20260910_090000_human_aaaa", "mission-control")
    corpus = write_traj_corpus(tmp_path / "corpus", [("20260910_090000_human_aaaa", None)])
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    kept = mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert [t["session_key"] for t in kept] == ["20260910_090000_human_aaaa"]
    assert kept[0]["session_class"] == "interactive"
    assert counts == {"kept": {"interactive": 1}}


def test_a_class_less_row_is_dropped_when_the_store_says_worker(
        tmp_path, monkeypatch):
    """The same join, other direction: the loop's own row is dropped by what the
    session JSON says, not by the name on the file — the fixture's stem is the
    loop-shaped `2026*_autocode_*` name the old filename rule never matched."""
    store = isolated_store(tmp_path, monkeypatch)
    write_class_session(store, CLASS_STEM, "worker", source="autocode")
    corpus = write_traj_corpus(tmp_path / "corpus", [(CLASS_STEM, None)])
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    kept = mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert kept == []
    assert counts["dropped"] == {"worker": 1}


def test_the_store_outvotes_an_interactive_stamp_on_a_machine_session(
        tmp_path, monkeypatch):
    """A row stamped `interactive` by an older extractor must not survive as human
    work when the session JSON behind it says `platform: worker` — that row is
    precisely the one #493 exists to exclude, so the corpus is treated as a derived
    cache and the store is authoritative."""
    store = isolated_store(tmp_path, monkeypatch)
    write_class_session(store, CLASS_STEM, "worker", source="autotriage")
    corpus = write_traj_corpus(tmp_path / "corpus", [(CLASS_STEM, "interactive")])
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    kept = mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert kept == []
    assert counts["dropped"] == {"worker": 1}


def test_the_store_outvotes_a_machine_stamp_on_a_human_session(
        tmp_path, monkeypatch):
    """The exclusion is not a ratchet that only ever shrinks the pool: a row the
    extractor mis-stamped `autonomy` is restored to interactive by the store, so
    real work is not silently lost to a stale field."""
    store = isolated_store(tmp_path, monkeypatch)
    write_class_session(store, "20260911_090000_human_bbbb", "mission-control")
    corpus = write_traj_corpus(
        tmp_path / "corpus", [("20260911_090000_human_bbbb", "autonomy")])
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    kept = mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert [t["session_key"] for t in kept] == ["20260911_090000_human_bbbb"]
    assert kept[0]["session_class"] == "interactive"


def test_the_extractor_writes_the_class_the_miner_reads(tmp_path, monkeypatch):
    """The seam is the JSONL line: the two scripts are separate processes that
    never import each other, so `session_class` surviving the write is the whole
    contract between them."""
    traj = et.parse_session(write_class_session(
        tmp_path / "sessions", CLASS_STEM, "worker", source="autocode"))
    et.append_trajectories([traj])
    buckets = list(et.OUTPUT_DIR.glob("*.jsonl"))
    assert len(buckets) == 1, "the extractor wrote no bucket to read back"
    assert json.loads(buckets[0].read_text().strip())["session_class"] == "worker"
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", et.OUTPUT_DIR)
    assert mt.load_trajectories(days=9999, agent_filter="all") == []
    kept = mt.load_trajectories(days=9999, agent_filter="all",
                                exclude_machine=False)
    assert [t["session_key"] for t in kept] == [CLASS_STEM]


def test_the_miner_and_the_extractor_name_the_interactive_class_alike():
    """The miner compares a string, so the two modules must agree on it."""
    assert mt.INTERACTIVE_CLASS == et.INTERACTIVE_CLASS == "interactive"


# ── clause 5: the class histogram is no longer a single value ────────────────

def histogram(out, header):
    """Class names listed under `header`, up to the block's blank line.

    Empty list when the header is absent — an exclusion section that did not
    happen must read as zero rows, not as an error.
    """
    if f"{header}\n" not in out:
        return []
    block = out.split(f"{header}\n", 1)[1]
    return re.findall(r"^  (\S+) +\d+$", block.split("\n\n")[0], re.MULTILINE)


def test_miner_stats_reports_more_than_one_session_class(tmp_path, monkeypatch,
                                                         capsys):
    corpus = write_traj_corpus(tmp_path / "corpus", MACHINE_ROWS)
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    rows = mt.load_trajectories(days=9999, agent_filter="all",
                                exclude_machine=False, class_counts=counts)
    mt.print_stats(rows, class_counts=counts)
    out = capsys.readouterr().out
    assert "By session class:" in out, out
    listed = histogram(out, "By session class:")
    assert set(listed) >= {"interactive", "worker", "autonomy", "browser",
                           "inner-voice"}, listed
    assert len(listed) > 1, listed
    assert histogram(out, "By session class dropped by the exclusion:") == [], (
        "nothing was excluded, so the run must not claim an exclusion count")


def test_extractor_stats_reports_more_than_one_session_class(tmp_path, capsys):
    et.append_trajectories([
        et.parse_session(write_class_session(tmp_path, "human_one",
                                             "mission-control")),
        et.parse_session(write_class_session(tmp_path, CLASS_STEM, "worker",
                                             source="autocode")),
    ])
    et.print_stats()
    out = capsys.readouterr().out
    assert "By session class:" in out, out
    assert set(histogram(out, "By session class:")) >= {"interactive", "worker"}, out


# ── clauses 4 + 6: the command the nightly actually runs ────────────────────

MINER_PATH = _ROOT / "scripts" / "mine-trajectories.py"
NIGHTLY_AGENT = "all"   # skills/trajectory-skill-mining/SKILL.md:46


def miner_row(key, cls, tool="Graphex493"):
    """One corroborated failure per session, shared across sessions so the
    pattern qualifies at `--threshold 2`. `cls=None` writes a row with no emitted
    class at all, which is what the corpus on disk carries for its first weeks."""
    row = {
        "session_key": key, "agent_id": "lloyd",
        "timestamp": "2026-09-12T10:10:10Z",
        "tool_count": 1, "error_count": 1, "has_errors": True,
        "tools": [{"name": tool, "is_error": True, "error_source": "protocol",
                   "sequence": 0, "params_summary": {"command": "pytest tests/"},
                   "result_summary": "boom: no such file"}],
        "error_tools": [{"name": tool, "sequence": 0, "error_type": "not_found",
                         "error_source": "protocol",
                         "params_summary": {"command": "pytest tests/"}}],
        "signals": [],
    }
    if cls is not None:
        row["session_class"] = cls      # None = no emitted field at all
    return row


def run_miner(corpus, out_dir, extra_args=()):
    cmd = [sys.executable, str(MINER_PATH), "--trajectory-dir", str(corpus),
           "--agent", NIGHTLY_AGENT, "--days", "9999", "--threshold", "2",
           "--output-dir", str(out_dir), *extra_args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180)


def graphex_candidate(out_dir):
    files = [p for p in Path(out_dir).glob("*.md") if "graphex493" in p.name]
    assert len(files) == 1, [p.name for p in Path(out_dir).glob("*.md")]
    return files[0].read_text(encoding="utf-8")


def test_the_nightly_mining_run_excludes_machine_sessions_by_default(tmp_path):
    """Two interactive sessions and three machine sessions carry the SAME
    failure. The candidate must report 2 sessions, not 5 — the loop's cadence is
    the thing that used to push patterns over the threshold."""
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = ([miner_row("i1", "interactive"), miner_row("i2", "interactive"),
             miner_row("w1", "worker"), miner_row("w2", "worker"),
             miner_row("a1", "autonomy")])
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    proc = run_miner(corpus, out)
    assert proc.returncode == 0, proc.stderr
    report = proc.stdout + proc.stderr
    assert "Machine-class sessions dropped: 3" in report, report
    fm = graphex_candidate(out)
    assert re.search(r"^sessions: 2$", fm, re.MULTILINE), fm
    assert re.search(r"^occurrences: 2$", fm, re.MULTILINE), fm
    assert "- i1" in fm and "- i2" in fm and "- w1" not in fm


def test_excluding_machine_sessions_does_not_stop_the_gate_emitting_candidates(
        tmp_path):
    """Purpose preserved: dropping 3 of 5 sessions must still write the
    candidate the 2 remaining independent sessions justify."""
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = [miner_row("i1", "interactive"), miner_row("i2", "interactive")]
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    proc = run_miner(corpus, out)
    assert proc.returncode == 0, proc.stderr
    assert "Machine-class sessions dropped: 0" in (proc.stdout + proc.stderr)
    assert re.search(r"^sessions: 2$", graphex_candidate(out), re.MULTILINE)


def test_the_exclusion_can_be_turned_off_for_a_full_corpus_view(tmp_path):
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = ([miner_row("i1", "interactive"), miner_row("i2", "interactive"),
             miner_row("w1", "worker"), miner_row("w2", "worker"),
             miner_row("a1", "autonomy")])
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    proc = run_miner(corpus, out, extra_args=("--include-machine",))
    assert proc.returncode == 0, proc.stderr
    report = proc.stdout + proc.stderr
    assert "Machine-class sessions dropped: 0" in report, report
    assert re.search(r"^sessions: 5$", graphex_candidate(out), re.MULTILINE)


# ── live-data guard: the classifier over the real corpus ────────────────────
#
# Same shape as the #392 guards above: the unit pins prove the CODE cannot label
# loop traffic as human work, this one proves the corpus on this machine is not
# labelled that way. It is the acceptance check for #493 as written — join
# `_pipeline/trajectories/*.jsonl` onto the session store and look for a row whose
# effective class is `interactive` while its session's `platform` is a machine
# platform — and it is the only test here that can see a regression introduced by
# re-extracting with a broken classifier.

# Resolved against the real data root, not the checkout: `_pipeline/` and
# `sessions/` are gitignored, so a worktree copy of this file would skip forever
# and the clause it pins would never be graded. Read-only against both.
LIVE_CORPUS = Path.home() / "lloyd" / "_pipeline" / "trajectories"
LIVE_STORE = Path.home() / "lloyd" / "sessions"
MACHINE_PLATFORMS = {"worker", "autonomy", "e2e-harness"}


def test_no_live_corpus_row_is_interactive_on_a_machine_platform():
    """Read-only. The acceptance check for #493 as written: join the corpus onto the
    session store and require zero rows whose effective class is `interactive` while
    the session's own `platform` is a machine platform.

    Measured over all 20 corpus buckets on 2026-09-14: 1,411 rows classify as 1,012
    worker / 237 autonomy / 79 inner-voice / 54 browser / 26 interactive / 3 smoke,
    and 0 rows violate the rule. Before the change all 1,411 carried `agent_id:
    lloyd` and no class at all, which is how 938 of 1,083 loop sessions (the
    2026-09-02→12 window measured at triage) reached the frequency gate as if they
    were Alan.

    Checked on two independent notions of the class, because one of them is
    vacuous today: `mt.effective_session_class` is store-authoritative, so on its
    own it could only ever fail through a broken classifier, while the row's
    *emitted* `session_class` is the extractor's own claim and is what a bad
    re-extraction would corrupt — no live row carries it yet (0 of 1,411 as at
    2026-09-14), so that half starts empty and earns its keep after the next
    extraction run. `machine_rows` is asserted non-zero so neither half can pass on
    an empty store.
    """
    # No skip here, unlike the pre-existing guards in this file: these two are the
    # only tests that read the real corpus, so a skip would mean the acceptance
    # check was never graded and nothing downstream could tell.
    assert LIVE_CORPUS.is_dir(), f"live corpus absent: {LIVE_CORPUS}"
    assert LIVE_STORE.is_dir(), f"session store absent: {LIVE_STORE}"
    cache: dict = {}
    violations: list[str] = []
    emitted_violations: list[str] = []
    rows = 0
    machine_rows = 0
    for bucket in sorted(LIVE_CORPUS.glob("*.jsonl")):
        for line in bucket.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            session_path = LIVE_STORE / f"{row.get('session_key')}.json"
            if not session_path.is_file():
                continue
            platform = json.loads(
                session_path.read_text(encoding="utf-8", errors="replace")
            ).get("platform")
            machine = platform in MACHINE_PLATFORMS
            machine_rows += machine
            if mt.effective_session_class(row, cache) == mt.INTERACTIVE_CLASS and machine:
                violations.append(f"{bucket.name}:{row.get('session_key')}={platform}")
            if row.get("session_class") == mt.INTERACTIVE_CLASS and machine:
                emitted_violations.append(
                    f"{bucket.name}:{row.get('session_key')}={platform}")
    assert rows > 0, "no corpus rows to check, so the assertion below is vacuous"
    assert machine_rows > 0, (
        "no corpus row joined to a machine-platform session, so the join itself is "
        "untested here and the zero below proves nothing")
    assert not violations, (
        f"corpus rows labelled interactive over a machine platform: {violations[:5]}")
    assert not emitted_violations, (
        "the extractor wrote `session_class: interactive` onto a machine-platform "
        f"session: {emitted_violations[:5]}")


def test_the_classifier_agrees_with_the_stored_fields_over_every_live_session():
    """Read-only oracle over the real session store: for every session JSON on this
    machine, the classifier's answer must equal clause 2 stated directly —
    `platform == "mission-control"` and `inner_voice` falsy.

    This is the check that survives the corpus being re-extracted: it is computed
    from the stored session files themselves, not from anything the extractor
    already wrote, so a classifier mutation trips it even while every corpus row is
    class-less — verified by mutating the classifier to ignore `inner_voice`, which
    flags 89 of the 1,688 files on this machine. Measured over those 1,688 files on
    2026-09-14: 44 classify interactive, and the `inner_voice` flag is what
    separates them from the rest — 89 of the 133 `mission-control` sessions are
    inner-voice turns, as are all 50 `browser` ones. `platform` alone would call 133
    interactive and fail here.
    """
    assert LIVE_STORE.is_dir(), f"session store absent: {LIVE_STORE}"
    files = sorted(LIVE_STORE.glob("*.json"))
    assert len(files) > 500, f"only {len(files)} session files, so this is vacuous"
    violations: list[str] = []
    interactive = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        expected = et.INTERACTIVE_CLASS if (
            data.get("platform") == "mission-control" and not data.get("inner_voice")
        ) else "not-interactive"
        got = et.classify_session(data)
        if (got == et.INTERACTIVE_CLASS) != (expected == et.INTERACTIVE_CLASS):
            violations.append(f"{path.name}: platform={data.get('platform')!r} "
                              f"inner_voice={data.get('inner_voice')!r} -> {got}")
        interactive += got == et.INTERACTIVE_CLASS
    assert not violations, f"classifier disagrees with clause 2: {violations[:5]}"
    assert 0 < interactive < len(files), (
        f"interactive={interactive} of {len(files)}: a store with none or all "
        "interactive means the oracle above cannot discriminate either direction")


def test_a_row_the_store_cannot_answer_is_dropped_as_uncoded_and_says_so(
        tmp_path):
    """Provenance of the last fallback, across the real command.

    A row with no emitted `session_class` whose session JSON is not in the store is
    dropped as `uncoded` — absence never reads as human-initiated work — and the run
    names the class it dropped rather than just the total, so a shrinking candidate
    set can be attributed to the rows that have no class rather than to the
    exclusion swallowing real work.
    """
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = [miner_row("i1", "interactive"), miner_row("i2", "interactive"),
            miner_row("nostoreentry493", None)]  # no stored session matches this key
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    proc = run_miner(corpus, out)
    assert proc.returncode == 0, proc.stderr
    report = proc.stdout + proc.stderr
    assert "Machine-class sessions dropped: 1" in report, report
    assert "dropped uncoded: 1" in report, report
    assert re.search(r"^sessions: 2$", graphex_candidate(out), re.MULTILINE)
