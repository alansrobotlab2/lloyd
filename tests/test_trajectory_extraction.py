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
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "extract_trajectories", _ROOT / "scripts" / "extract-trajectories.py"
)
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)

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
    "exit 1", "Traceback (most recent call last):", "ValueError: bad input",
    "bash: foo: command not found", "No such file or directory",
    "npm ERR! code ELIFECYCLE", "FAILED tests/test_x.py", "fatal error: oops",
])
def test_semantic_errors_are_detected_without_an_is_error_flag(text):
    assert et.has_semantic_error(text) is True


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
