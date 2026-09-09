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


def test_error_and_sequence_candidates_are_unaffected():
    assert mt.is_emittable(
        {"type": "error", "tool_name": "Bash", "error_type": "not_found"}) is True
    assert mt.is_emittable({"type": "sequence", "ngram_size": 3}) is True


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
