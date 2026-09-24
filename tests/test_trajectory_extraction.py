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
import inspect
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests._live_data import require_live_data, require_live_volume
from app.paths import production_data_root  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
#: This module's own namespace, so a pin test can redirect a live root by name and
#: the guard under test reads the redirected value on its next global lookup.
_THIS = sys.modules[__name__]
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


# ── the error label comes from the error signal, not the stdout prose (#1055) ─
#
# `error_type` is half the skill-candidate pattern key, so a keyword sitting in
# a program's own printed output picks a candidate's identity. Two rows from
# the 2026-09-13 mining run, verbatim, both minted as `Bash/permission`
# (occ=2) and both adjudicated `rejected_false_positive`: a skill-lint pass
# whose command ended in `echo "--- forbidden ---"; grep -c -E ...` — grep
# matched nothing, exit 1, i.e. a *successful* check reporting a zero count —
# and a mutation test reporting an intended kill. Neither record contains
# `EPERM`, `EACCES` or `Permission denied`.

@pytest.mark.parametrize("text", [
    "ERROR: 1 3 1 --- forbidden --- 0  [exit code: 1]",
    "FAILS  M2 forbidden rule always says clean",
])
def test_the_word_forbidden_in_command_output_is_not_a_permission_error(text):
    assert et.categorize_error(text) != "permission"


@pytest.mark.parametrize("text,category", [
    ("Permission denied: /etc/shadow", "permission"),
    ("open /x: access denied", "permission"),
    ("unlink /mnt: Operation not permitted (EPERM)", "permission"),
    ("chmod: failed, errno EACCES", "permission"),
])
def test_a_permission_label_still_needs_a_permission_form(text, category):
    """Dropping bare `forbidden` must not narrow what a real denial maps to:
    each of these is the form the item names as still required."""
    assert et.categorize_error(text) == category


@pytest.mark.parametrize("text", [
    "not found",
    "timeout",
    "DNS",
    "schema",
    "the key was not found in the mapping, which is expected",
    "timeout handler registered for SIGTERM",
    "DNS names appear in the echoed config header",
    "schema column listed in the migration report",
])
def test_a_bare_prose_word_does_not_choose_the_label(text):
    """Each of these words appears in ordinary program output — an echoed
    header, a test name, a column list — so none of them may select a class.
    Every alternative left in `ERROR_CATEGORIES` is an errno or a phrase that
    only a failure says."""
    assert et.categorize_error(text) == "logic"


@pytest.mark.parametrize("text,category", [
    # The labels pinned by `test_error_categories` above, restated here so this
    # diff is the thing that proves they survived the narrowing.
    ("404 Not Found", "not_found"),
    ("request timed out after 30s", "timeout"),
    ("invalid JSON payload", "validation"),
    # Phrase forms the bare words used to cover by accident.
    ("bash: rg: command not found", "not_found"),
    ("open /x: ETIMEDOUT", "timeout"),
    ("curl: (6) Could not resolve host: example.com", "network"),
])
def test_errno_and_phrase_forms_keep_their_labels(text, category):
    assert et.categorize_error(text) == category


# The real shape of the MCP transport failure the same run hit: eleven copies
# of this body across 2026-09-10/11, `error_source=protocol`, categorised
# `logic` because nothing in `ERROR_CATEGORIES` matched — so a protocol failure
# presented itself as a behaviour problem in the tool's own logic. It is
# already covered by the installed skill `mcp-transport-error-recovery`, whose
# Pattern 2 names `automod_gate`.
TRANSPORT_BODY = ("automod_gate: transport error: unhandled errors "
                  "in a TaskGroup (1 sub-exception)")
# How that body reaches the trajectory record: `result_summary()` prefixes it.
TRANSPORT_SUMMARY = f"ERROR: {TRANSPORT_BODY}"


def test_a_transport_failure_gets_a_label_of_its_own():
    assert et.categorize_error(TRANSPORT_SUMMARY) == "transport"


def test_a_lost_dispatch_is_labelled_transport_end_to_end(tmp_path):
    """Across the extractor seam: the label on the row the miner keys on, not
    just the helper's return value."""
    traj = first_tool(tmp_path, [("automod_gate", {}, TRANSPORT_BODY, True)])
    tool = traj["tools"][0]
    assert tool["error_source"] == "protocol"
    assert tool["result_summary"] == TRANSPORT_SUMMARY
    assert traj["error_tools"][0]["error_type"] == "transport"


# The label only matters because of what happens one process further out: the
# miner groups on `(tool_name, error_type)` and `candidate_pattern_key` turns
# that into the string `skill_verdicts.py` joins the adjudication ledger on.
# These two cross that boundary — extract, then mine, then key — because the
# item's whole claim is about the key, not the helper's return value.

def test_the_prose_rows_mine_no_bash_permission_candidate(tmp_path):
    """The 2026-09-13 run emitted `candidate-bash-permission-20260913.md`
    (`Bash/permission`, occ=2, `status: pending_review`) off exactly these two
    bodies. Mined again after the fix neither can reach that key: the mutation
    test's body keys as `Bash/logic`, a label an adjudicator can derive from
    the record in front of them, and the lint pass's `[exit code: 1]` row is
    `nonzero_exit`, which #500's mining gate rejects before keying.
    `Bash/permission` is left to bodies that actually say `Permission denied`."""
    lint_body = "ERROR: 1 3 1 --- forbidden --- 0  [exit code: 1]"
    mutant_body = "FAILS  M2 forbidden rule always says clean"

    lint = et.parse_session(write_session(
        tmp_path, [("Bash", {"command": "./lint.sh"}, lint_body, True)],
        name="forbidden-prose-lint"))
    assert lint["error_tools"][0]["error_type"] == "logic"
    assert not mt.is_corroborated_error(lint["error_tools"][0])   # the #500 gate

    rows = [et.parse_session(write_session(
               tmp_path, [("Bash", {"command": f"./mutants.py {n}"}, mutant_body, True)],
               name=f"forbidden-prose-mutant-{n}"))
            for n in (1, 2)]
    assert [r["error_tools"][0]["error_type"] for r in rows] == ["logic", "logic"]
    keys = {mt.candidate_pattern_key(p)
            for p in mt.mine_error_patterns(rows + [lint], threshold=2)}
    assert keys == {"Bash/logic"}


def test_a_transport_row_keys_its_own_pattern_key(tmp_path):
    """The other direction of the same fall-through: eleven TaskGroup failures
    on 2026-09-10/11 reached the key `automod_gate/logic`, which is what made a
    lost dispatch read as a behaviour problem. After the fix the mined key is
    `automod_gate/transport` — the key a person can match against the installed
    `mcp-transport-error-recovery`, whose Pattern 2 names this tool."""
    rows = [et.parse_session(write_session(
               tmp_path, [("automod_gate", {}, TRANSPORT_BODY, True)],
               name=f"transport-lost-dispatch-{n}"))
            for n in (1, 2)]
    keys = {mt.candidate_pattern_key(p)
            for p in mt.mine_error_patterns(rows, threshold=2)}
    assert keys == {"automod_gate/transport"}


@pytest.mark.parametrize("body,expected", [
    ("active\n", "logic"),
    ("Permission denied: /etc/shadow\n", "permission"),
])
def test_the_protocol_marker_never_selects_the_label(tmp_path, body, expected):
    """#500: `error_source="protocol"` is `stats.is_error` — a non-zero-exit
    turn marker that reads `protocol` on effectively every flagged step, clean
    health-check output included. If the label consulted it, every failed probe
    would become a transport failure; these rows carry the marker and the
    phrase decides nothing."""
    traj = first_tool(tmp_path, [("Bash", {"command": "true"}, body, True)])
    tool = traj["tools"][0]
    assert tool["error_source"] == "protocol"
    assert traj["error_tools"][0]["error_type"] == expected
    assert traj["error_tools"][0]["error_type"] != "transport"


def test_a_keyword_past_the_persisted_prefix_does_not_choose_the_label(tmp_path):
    """`result_summary()` keeps MAX_ERROR_LEN characters, so a keyword beyond
    that cap is invisible to everyone who later reads the record — including
    the person adjudicating the candidate the label keys. #1055 found exactly
    that on a 2026-09-14 row labelled `resource` whose stored summary matches no
    pattern at all."""
    assert et.MAX_ERROR_LEN == 200
    body = "health check output " * 20 + "out of memory while loading model"
    assert "out of memory" in body[et.MAX_ERROR_LEN:]
    traj = first_tool(tmp_path, [("Bash", {"command": "./probe.sh"}, body, True)])
    tool = traj["tools"][0]
    assert "out of memory" not in tool["result_summary"]
    assert traj["error_tools"][0]["error_type"] == "logic"
    assert (traj["error_tools"][0]["error_type"]
            == et.categorize_error(tool["result_summary"]))


def test_the_miner_keeps_no_second_error_categorizer():
    """`scripts/mine-trajectories.py` carried `ERROR_CATEGORIES` and a pair of
    categorizers with zero call sites — it keys on the extractor's
    `error_tool["error_type"]` — so a fix applied there changed nothing while
    the copy sat looking like the thing to edit. It must not come back: a
    second derivation of the value the verdict ledger joins on would disagree
    with the first. The module still has to import."""
    path = _ROOT / "scripts" / "mine-trajectories.py"
    source = path.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "mine_trajectories_no_categorizer", path
    )
    miner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(miner)
    assert not hasattr(miner, "ERROR_CATEGORIES")
    assert not hasattr(miner, "categorize_error")
    assert not hasattr(miner, "categorize_result_summary")
    # The clause's own check is this literal grep, run over the whole file and
    # not over a filtered slice of it: it is clean today because the comment
    # above the deletion names the two functions without call parentheses, so
    # the pattern can only start matching again by a matcher coming back.
    assert not re.search(r"(categorize_error|categorize_result_summary)\(", source)


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


# ── the class of a flagged step (#500) ───────────────────────────────────────
#
# `stats.is_error` answers "the harness flagged this", and it is set for any
# non-zero shell exit. `error_source` records which channel flagged it, and there
# is effectively one channel, so it cannot discriminate either. Measured over the
# flagged tool messages in `~/lloyd/sessions/*.json` (read 2026-09-19, 3,752
# rows): 1,732 carry a positive exit code and nothing else, 1,586 report failure
# as a JSON body, 39 are harness refusals, 29 are signal deaths, and 366 are a
# flag with no shape behind it. The first bucket contains a `grep` that matched
# nothing (exit 1) and an `ls` of a missing path (exit 2) — outcomes, not
# failures. `failure_class` is what separates them, and the miner's gate keys on
# it, so the split below is the split that decides what reaches skill authoring.

def flagged(tmp_path, calls, **kw):
    """Parse one session and return its single flagged step in both shapes.

    Returns `(tool_entry, error_tool_entry)` — the class must be persisted on
    both, because the miner reads `error_tools[]` for error patterns and
    `tools[]` for the error rate on a signature.
    """
    traj = first_tool(tmp_path, calls, **kw)
    assert traj["error_count"] == 1
    seq = traj["error_tools"][0]["sequence"]
    tool = next(t for t in traj["tools"] if t["sequence"] == seq)
    return tool, traj["error_tools"][0]


def test_a_json_body_reporting_failure_is_classed_structured_error(tmp_path):
    """Clause 1 (#500), first half: a body that reports failure as data, not as
    prose, on a step the harness flagged."""
    tool, err = flagged(tmp_path, [
        ("mcp____tools_email_search", {"query": "invoice"},
         '{"error": "goal is required"}', True),
    ])
    assert tool["failure_class"] == "structured_error"
    assert err["failure_class"] == "structured_error"
    assert err["exit_code"] is None


def test_a_negative_exit_code_is_classed_timeout_or_signal(tmp_path):
    """Clause 1 (#500), second half: the signed parse landed in #389, and the
    class is what makes it mean something. A killed command is the real Bash
    timeout signature — 29 such rows in the 8-day window read 2026-09-19."""
    tool, err = flagged(tmp_path, [
        ("Bash", {"command": "sleep 900"},
         "command timed out after 120000ms\n\n[exit code: -15]", True),
    ])
    assert tool["exit_code"] == -15
    assert tool["failure_class"] == "timeout_or_signal"
    assert err["failure_class"] == "timeout_or_signal"


def test_a_signal_death_outranks_the_error_body_it_carries(tmp_path):
    """Precedence, because the two signals genuinely co-occur: a killed call's
    body can itself read as a structured error, and `was the process killed?` is
    the question a mitigation answers. Losing that flips the class into the one
    the miner cannot distinguish from a `grep` no-match."""
    _tool, err = flagged(tmp_path, [
        ("Bash", {"command": "sleep 900"}, '{"error": "killed"}\n\n[exit code: -9]', True),
    ])
    assert err["failure_class"] == "timeout_or_signal"


def test_a_flagged_step_with_only_an_exit_marker_is_classed_nonzero_exit(tmp_path):
    """Clause 2 (#500): the class that has to appear in written rows for the
    miner's narrowing to have anything to reject. Today the corpus carries 1,082
    steps with a positive exit code and 3 whose `error_source` is `exit_code`,
    because the harness flag takes precedence over the marker on every step that
    has both — so an exit-derived class was invisible exactly when it mattered."""
    tool, err = flagged(tmp_path, [
        ("Bash", {"command": "grep -rn TODO scripts/"},
         "no matches found\n\n[exit code: 1]", True),
    ])
    assert tool["exit_code"] == 1
    assert tool["failure_class"] == "nonzero_exit"
    assert err["failure_class"] == "nonzero_exit"
    # The step is still flagged, and still flagged from the same channel — the
    # class is a narrowing of what the gate promotes, not a suppression of the
    # record.
    assert tool["is_error"] is True
    assert tool["error_source"] == "protocol"


def test_a_structured_body_with_a_positive_exit_code_is_not_the_exit_class(tmp_path):
    """The class is the payload's shape first: a tool that returned a JSON error
    and a non-zero exit is reporting a failure, not exiting 1 on a search."""
    _tool, err = flagged(tmp_path, [
        ("Bash", {"command": "deploy --check"},
         '{"error": "upstream unreachable", "code": 3}\n\n[exit code: 3]', True),
    ])
    assert err["exit_code"] == 3
    assert err["failure_class"] == "structured_error"


def test_a_harness_refusal_is_not_the_expected_nonzero_exit_class(tmp_path):
    """A refusal, a bad dispatch or a lost connection: the call did not run.
    39 such rows in the 8-day window read 2026-09-19 — dispatch failures
    (`Tool 'Bash' is disabled by configuration.`), unkeyable dispatches
    (`no server claims tool 'name'`), `Input validation error: ...`, and
    transport errors."""
    _tool, err = flagged(tmp_path, [
        ("Bash", {"command": "ls"}, "Tool 'Bash' is disabled by configuration.", True),
    ])
    assert err["failure_class"] == "harness_block"
    assert _tool["failure_class"] == "harness_block"


def test_a_flag_with_no_shape_behind_it_is_classed_protocol_flagged(tmp_path):
    """The residual, named rather than folded into a class with a claimed
    mechanism: the harness lost the call and the body is a plain string. 366 rows
    in the 8-day window read 2026-09-19."""
    _tool, err = flagged(tmp_path, [
        ("Bash", {"command": "ls"}, "total 11472\ndrwxr-xr-x 1 a a 4096 .", True),
    ])
    assert err["failure_class"] == "protocol_flagged"
    assert err["exit_code"] is None


def test_an_unflagged_step_carries_no_failure_class(tmp_path):
    """The field describes a failure. A step that was not flagged says nothing
    about a class, and a default of `nonzero_exit` here would put the whole
    corpus into the class the miner rejects."""
    traj = first_tool(tmp_path, [
        ("Bash", {"command": "true"}, "done\n\n[exit code: 0]", False),
        ("Read", {"file_path": "/a"}, "except Exception as e:", False),
    ])
    assert [t["failure_class"] for t in traj["tools"]] == [None, None]
    assert traj["error_tools"] == []


def test_a_pre_stats_session_gets_a_class_from_the_fallback_flag(tmp_path):
    """A tool message predating `stats` is flagged only by the structured-body
    fallback, and must still be classified — otherwise the miner's legacy
    no-`error_source` branch is the only reading a historical row ever gets."""
    _tool, err = flagged(tmp_path, [
        ("Bash", {"command": "name"}, '{"error": "no server claims tool \'name\'"}', None),
    ])
    assert err["failure_class"] == "structured_error"


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


# The five miner-side corroboration tests (`test_mining_ignores_a_keyword_only_error`
# and its siblings) live in tests/test_mine_trajectories.py since #511.


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


def adjacent_next_step_traj(session_key):
    """One session whose failed `mkdir` is followed by an unrelated `Read`.

    This is the shape #1327 says is *not* a recovery: `bash:fs:ERR → read`, where
    the successor neither re-attempts the failed `mkdir` nor names its target (the
    `Read` opens `/tmp/rec/b`, the failed call named `/tmp/rec`). Under the
    adjacency rule it was flagged `has_error_recovery: true` — the item's own
    falsifier pairs were a failed `grep` followed by `date -u …` and a traceback
    followed by a fresh `ls` — and since the narrowing every key mined from this
    row is flagged false.

    Mined at threshold 2 across three copies it yields the false half of the
    gate: the 2-grams `bash:fs → read`, `read → bash:fs:ERR` and
    `bash:fs:ERR → read` (the trigrams `bash:fs → read → bash:fs:ERR` and
    `read → bash:fs:ERR → read` are mined too but never emitted — a windowed key
    owes its sessions to its own suffix, see
    `test_a_windowed_key_does_not_bill_sessions_its_suffix_already_counted`).
    `retrying_traj` is the true half. `error_tools` is empty on purpose: error
    mining has its own fixture (`error_traj`) and adding a row here would add an
    error pattern these assertions do not need.
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


def retrying_traj(session_key: str, command: str = "mkdir -p /tmp/rec") -> dict:
    """One session whose failed call is re-attempted and succeeds (#1327).

    `mkdir -p /tmp/rec` errors and the very next step is the same `mkdir` again,
    OK: the bigram is `bash:fs:ERR → bash:fs`, and its successor *is* the failed
    call run again — the same tool and the same Bash command category, which is
    what the label carries. That is a re-attempt, so the flag is true; under the
    old rule this fixture and `adjacent_next_step_traj` were indistinguishable.
    """
    return {
        "session_key": session_key,
        "timestamp": "2026-09-21T18:00:00Z",
        "tool_count": 2, "error_count": 1, "has_errors": True,
        "tools": [
            {"name": "Bash", "is_error": True, "sequence": 0,
             "params_summary": {"command": command},
             "result_summary": "mkdir: cannot create directory: Read-only file system"},
            {"name": "Bash", "is_error": False, "sequence": 1,
             "params_summary": {"command": command},
             "result_summary": "ok"},
        ],
        "error_tools": [], "signals": [],
    }


def edit_then_read_traj(session_key: str, failed_target: str, read_target: str) -> dict:
    """One session whose failed `Edit` is followed by a `Read`.

    Which way the flag falls is decided by the *argument*, not the tool: the
    bigram is `edit:ERR → read` either way, and it is a recovery only when the
    `Read` names the object the failure named (the agent going back to the file
    the edit could not touch) and not when it opens something else.
    """
    return {
        "session_key": session_key,
        "timestamp": "2026-09-21T18:00:00Z",
        "tool_count": 2, "error_count": 1, "has_errors": True,
        "tools": [
            {"name": "Edit", "is_error": True, "sequence": 0,
             "params_summary": {"file_path": failed_target},
             "result_summary": "File has been modified since read"},
            {"name": "Read", "is_error": False, "sequence": 1,
             "params_summary": {"file_path": read_target},
             "result_summary": "ok"},
        ],
        "error_tools": [], "signals": [],
    }


def windowed_traj(session_key: str, with_prefix: bool) -> dict:
    """One session from the 17⊂20 pair #1327 measured, built from `mkdir`.

    `with_prefix=True` emits `bash:fs → bash:fs:ERR → bash:fs` (17 sessions in the
    item's shape); `with_prefix=False` emits just the suffix bigram
    `bash:fs:ERR → bash:fs`, which is what makes the other 3 of 20. The real pair
    was `seq-3-bash-fs-bash-fs-err-bash-other` (17 sessions) inside
    `seq-2-bash-fs-err-bash-other` (20), and it is not a coincidence of these
    fixtures: `mine_sequence_patterns` walks `for n in (2, 3)` over one collapsed
    label stream, so a 3-gram at position *i* always feeds its suffix bigram at
    *i+1* in the same session — every `seq-3` key's sessions are a subset of its
    suffix's. Both keys here are flagged true by `ngram_shows_recovery`, so what
    keeps the 3-gram out is the session arithmetic and nothing else.
    """
    steps = ([{"command": "mkdir -p /tmp/w", "is_error": False}]
             if with_prefix else [])
    steps += [{"command": "mkdir -p /tmp/w", "is_error": True},
              {"command": "mkdir -p /tmp/w", "is_error": False}]
    return {
        "session_key": session_key,
        "timestamp": "2026-09-21T18:00:00Z",
        "tool_count": len(steps), "error_count": 1, "has_errors": True,
        "tools": [{"name": "Bash", "is_error": step["is_error"],
                   "sequence": i, "params_summary": {"command": step["command"]},
                   "result_summary": "boom" if step["is_error"] else "ok"}
                  for i, step in enumerate(steps)],
        "error_tools": [], "signals": [],
    }


def test_an_unrelated_next_step_is_not_a_recovery_and_a_retry_is():
    """Clause 1 (#1327), both directions, computed by the miner.

    The flag used to be set by adjacency alone: `scripts/mine-trajectories.py`
    walked the n-gram and set it on any `:ERR` step followed by any non-`:ERR`
    step, so `bash:fs:ERR → read` — a failed `mkdir` followed by a read of a
    *different* file — was a "recovery", and consolidation hand-adjudicated keys
    on that evidence. Now the successor has to address the failure, and the two
    directions are pinned from the same mining call so a rule that merely always
    sets the flag, or never sets it, fails one side.
    """
    adjacency = mt.mine_sequence_patterns(
        [adjacent_next_step_traj(f"adj-{i}") for i in (1, 2, 3)], threshold=2)
    retried = mt.mine_sequence_patterns(
        [retrying_traj(f"retry-{i}") for i in (1, 2, 3)], threshold=2)

    by_seq = {p["sequence_str"]: p for p in adjacency + retried}
    for seq in ("bash:fs:ERR → read", "read → bash:fs:ERR"):
        assert seq in by_seq, f"the adjacency fixture lost {seq}: {sorted(by_seq)}"
        assert by_seq[seq]["has_error_recovery"] is False, (
            f"{seq}: adjacency is still setting the flag")
    assert "bash:fs:ERR → bash:fs" in by_seq, (
        f"the retry fixture mined no retry bigram: {sorted(by_seq)}")
    assert by_seq["bash:fs:ERR → bash:fs"]["has_error_recovery"] is True, (
        "a re-attempt of the failed call is not being read as recovery")
    assert mt.is_emittable(by_seq["bash:fs:ERR → read"]) is False
    assert mt.is_emittable(by_seq["bash:fs:ERR → bash:fs"]) is True


def test_a_successor_that_names_the_failed_target_is_a_recovery():
    """Clause 1's other accepted shape: `edit:ERR → read` where both calls name the
    same path is the agent going back to the object the failure named."""
    seqs = mt.mine_sequence_patterns(
        [edit_then_read_traj(f"same-{i}", "/tmp/rec/a.py", "/tmp/rec/a.py")
         for i in (1, 2, 3)], threshold=2)
    pat = [p for p in seqs if p["sequence_str"] == "edit:ERR → read"]
    assert len(pat) == 1, f"no `edit:ERR → read` mined: {[p['sequence_str'] for p in seqs]}"
    assert pat[0]["has_error_recovery"] is True


def test_a_successor_that_names_a_different_target_is_not_a_recovery():
    """The same two tools and the same labels, with a different path in the
    successor: the labels cannot tell these two fixtures apart, which is exactly
    why the flag has to read the arguments."""
    seqs = mt.mine_sequence_patterns(
        [edit_then_read_traj(f"other-{i}", "/tmp/rec/a.py", "/tmp/other/b.py")
         for i in (1, 2, 3)], threshold=2)
    pat = [p for p in seqs if p["sequence_str"] == "edit:ERR → read"]
    assert len(pat) == 1, f"no `edit:ERR → read` mined: {[p['sequence_str'] for p in seqs]}"
    assert pat[0]["has_error_recovery"] is False


def test_a_shared_mode_setting_does_not_make_a_shared_target():
    """`limit: 200` on both calls names no object. Matching argument values
    without asking whether the value *can* name one would rebuild the adjacency
    bug out of the arguments — every `Read` in the corpus shares `limit` with
    every other."""
    rows = [{
        "session_key": f"mode-{i}", "timestamp": "2026-09-21T18:00:00Z",
        "tool_count": 2, "error_count": 1, "has_errors": True,
        "tools": [
            {"name": "Read", "is_error": True, "sequence": 0,
             "params_summary": {"file_path": "/tmp/rec/a.py", "limit": 200},
             "result_summary": "File is a binary file"},
            {"name": "Read", "is_error": False, "sequence": 1,
             "params_summary": {"file_path": "/tmp/rec/b.py", "limit": 200},
             "result_summary": "ok"},
        ],
        "error_tools": [], "signals": [],
    } for i in (1, 2, 3)]
    pat = [p for p in mt.mine_sequence_patterns(rows, threshold=2)
           if p["sequence_str"] == "read:ERR → read"]
    # The same label twice with only the error flag between them is a re-attempt
    # (`read:ERR → read`), so this key is true for that reason whatever the
    # arguments say — the mode-value question is pinned below at the helper, where
    # the two calls genuinely differ.
    assert len(pat) == 1 and pat[0]["has_error_recovery"] is True
    assert mt.ngram_shows_recovery(
        ["edit:ERR", "read"],
        [{"params_summary": {"file_path": "/tmp/rec/a.py", "limit": 200}},
         {"params_summary": {"file_path": "/tmp/rec/b.py", "limit": 200}}]) is False


def test_a_windowed_key_does_not_bill_sessions_its_suffix_already_counted():
    """Clause 3 (#1327), on the item's own 17⊂20 shape with the flag ruled out.

    Both keys here are flagged `has_error_recovery: true`, so the 3-gram's absence
    can only come from the session arithmetic: its 17 sessions are 17 of the 20
    its suffix bigram counted, which is every one of them, so it has no session of
    its own and clears no `sessions >= 3` gate. Before this it reached a candidate
    file — and consolidation's nightly `sessions >= 3` evidence gate — on a session
    set that was one event set counted at two window sizes.
    """
    rows = ([windowed_traj(f"tri-{i}", with_prefix=True) for i in range(17)]
            + [windowed_traj(f"bi-{i}", with_prefix=False) for i in range(3)])
    bigram_seq, trigram_seq = "bash:fs:ERR → bash:fs", "bash:fs → bash:fs:ERR → bash:fs"
    for threshold in (2, 3):
        seqs = mt.mine_sequence_patterns(rows, threshold=threshold)
        by_seq = {p["sequence_str"]: p for p in seqs}
        assert bigram_seq in by_seq, (
            f"threshold {threshold}: the suffix bigram itself vanished: {sorted(by_seq)}")
        bigram = by_seq[bigram_seq]
        assert len(bigram["sessions"]) == bigram["total_sessions"] == 20, (
            f"threshold {threshold}: the suffix key lost sessions: "
            f"{len(bigram['sessions'])} of {bigram['total_sessions']}")
        assert bigram["borrowed_sessions"] == 0, "a bigram has no shorter suffix"
        assert bigram["has_error_recovery"] is True
        assert trigram_seq not in by_seq, (
            f"threshold {threshold}: a windowed key was emitted on "
            f"{len(by_seq[trigram_seq]['sessions'])} own sessions borrowed from "
            "its suffix key")
    assert [p for p in mt.mine_sequence_patterns(rows, threshold=1)
            if p["sequence_str"] == trigram_seq] == [], (
        "a key with no session of its own clears the gate at threshold 1, so it "
        "is still being counted at its own window size")


def test_the_window_dedup_subtracts_only_the_suffix_key_set():
    """The arithmetic clause 3 rests on, where the mined sets cannot show it.

    Under `for n in (2, 3)` a 3-gram's session set is always a *subset* of its
    suffix's, so no real key can exhibit a partial subtraction — which is why the
    count needs a direct check, and why the consequence is that whole `seq-3`
    keys stop being emitted. The disjoint `("a", "b")` key is the prefix, not the
    suffix: subtracting it would bill sessions nobody counted for this window.
    """
    data = {
        ("a", "b"): {"sessions": {"s9"}},
        ("b", "c"): {"sessions": {"s1", "s2", "s3", "s4"}},
        ("a", "b", "c"): {"sessions": {"s1", "s2", "s5"}},
    }
    assert mt._suffix_sessions(("a", "b", "c"), data) == {"s1", "s2", "s3", "s4"}
    assert data[("a", "b", "c")]["sessions"] - mt._suffix_sessions(
        ("a", "b", "c"), data) == {"s5"}
    assert mt._suffix_sessions(("b", "c"), data) == set(), (
        "a 2-gram has no strictly shorter suffix and must never be subtracted")


def test_the_candidate_file_separates_own_sessions_from_the_observed_total(tmp_path):
    """A windowed key that *did* have its own sessions must say so, or its
    `sessions:` count and its `## Sessions Affected` list disagree with the corpus
    that produced them. No key reaches this through the miner today — every
    3-gram's set is a subset of its suffix's, so its own count is 0 and it is not
    emitted — so the dict is built here, against the writer that would print it."""
    pat = {
        "type": "sequence", "sequence": ("a", "b", "c"),
        "sequence_str": "a → b → c", "ngram_size": 3,
        "sessions": {"s5", "s6", "s7"}, "total_sessions": 5,
        "borrowed_sessions": 2, "examples": [], "dates": {"2026-09-21"},
        "has_error_recovery": True, "first_seen": "2026-09-21",
        "last_seen": "2026-09-21",
    }
    path = mt.write_candidate_file(pat, tmp_path)
    text = Path(path).read_text(encoding="utf-8")
    assert "sessions: 3" in text.split("---")[1], "the count billed the borrowed set"
    assert len(re.findall(r"^- s\d$", text, re.MULTILINE)) == 3, (
        "the Sessions Affected list and the sessions: count disagree")
    assert "2 further session(s) carried this window too" in text
    assert "(5 observed in total)" in text


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
    rows = [adjacent_next_step_traj(f"rec-{i}") for i in (1, 2, 3)]
    # The true half has to come from `retrying_traj`: `adjacent_next_step_traj`
    # is the shape #1327 narrowed the flag *against*, so on its own it now mines
    # nothing flagged true and `recovering` below would be empty — a test that
    # asserts "both kinds" and silently holds one.
    rows += [retrying_traj(f"retry-{i}") for i in (1, 2, 3)]
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


def sequence_pattern(ngram: tuple[str, ...], sessions: set[str]) -> dict:
    """One mined-shaped sequence pattern, for the writer tests that need a
    specific n-gram in hand. Field-for-field what `mine_sequence_patterns`
    returns; the flag is false because these fixtures carry no failing step."""
    return {
        "type": "sequence", "sequence": ngram,
        "sequence_str": " → ".join(ngram), "ngram_size": len(ngram),
        "sessions": set(sessions), "total_sessions": len(sessions),
        "borrowed_sessions": 0, "examples": [], "dates": {"2026-09-15"},
        "has_error_recovery": False, "first_seen": "2026-09-15",
        "last_seen": "2026-09-15",
    }


def aliased_pair():
    """The two sequence patterns whose *filename* aliased, from the same row the
    miner is fed. Their keys were already distinct at filing — asserted by
    `test_the_aliased_pair_shares_one_plain_slug_but_not_one_candidate_name`; the
    pair whose keys aliased is the 5-gram fixture in
    `test_two_ngrams_differing_only_past_the_cap_get_distinct_keys`.

    Built from the row's labels rather than pulled out of
    `mine_sequence_patterns`'s return, because since #1327 it no longer returns
    them: these are trigrams, and a trigram's session set is a subset of its own
    suffix bigram's, so it has no session of its own and clears no threshold. The
    alias needs a key past the 50-character slug cap, which no 2-gram has, so
    there is no pair to mine at a size the window rule leaves emitted. The n-grams
    come from `aliased_traj` through `normalize_tool_name` — the miner's own label
    step — so the collision these tests grade is still the one the writer gets."""
    labels = [mt.normalize_tool_name(tool)
              for tool in sorted(aliased_traj("alias-a")["tools"],
                                 key=lambda tool: tool.get("sequence", 0))]
    pair = [tuple(labels[i:i + 3]) for i in range(len(labels) - 2)
            if tuple(labels[i:i + 2]) == ALIASED_HEAD
            and labels[i + 2] in ALIASED_TAILS]
    assert len(pair) == 2, "the fixture must yield both aliased n-grams"
    return [sequence_pattern(ngram, {"alias-a", "alias-b"}) for ngram in pair]


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
    # The pair that reaches a file has to carry a real recovery. Since #1181 an
    # n-gram with no failing step in it is refused by the emission gate, so
    # `aliased_traj` alone — 6 successful steps, nothing to recover from — mines
    # only suppressed patterns, and `main()` would report zero sequence files
    # while still honouring the arithmetic this test pins. Since #1327 so does
    # `adjacent_next_step_traj`: its failing `mkdir` is followed by a `Read` of a
    # different file, which is adjacency, not recovery. The aliased pair stays in
    # the fixture for the Suppressed count; `retrying_traj` supplies the patterns
    # that actually reach a file.
    rows += [retrying_traj(f"retry-{i}") for i in (1, 2, 3)]
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
    # path reads as absent forever. Absent skips and the reason names the path
    # (#1377); *present* always counts, which is what
    # `test_a_present_corpus_that_admits_nothing_fails_rather_than_skips` pins, so no
    # run goes green on the corpus not having been read. Read-only; the
    # emitted files go into `tmp_path`.
    #
    # Checking that root is not the same act as reading it, and until #1403 this
    # guard did the first and then the other. `load_trajectories` takes no path
    # argument: it reads `mt.TRAJECTORY_DIR`, which `scripts/mine-trajectories.py`
    # derives from `app.paths.PIPELINE_DIR` and which is therefore
    # CHECKOUT-relative — in a round's worktree that is
    # `<worktree>/.lloyd-data/_pipeline/trajectories`, absent while the real
    # corpus two directories away is full. The comment above, that the
    # worktree-relative read was fixed, was true of `LIVE_CORPUS` and false of
    # what the guard then loaded, and the node red-blocked every promotion from a
    # worktree on 2026-09-23 reading 0 rows (#1403 cause B). Its three sibling
    # live guards bind the miner; this is that binding, and
    # `test_a_guard_that_reads_a_different_path_than_it_checks_is_caught` is what
    # keeps it from silently un-binding again.
    require_live_data(LIVE_CORPUS, "live trajectory corpus")
    previous = mt.TRAJECTORY_DIR
    mt.set_trajectory_dir(LIVE_CORPUS)
    try:
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
    finally:
        mt.set_trajectory_dir(previous)


# ── one candidate per key, merged evidence (#515) ────────────────────────────
#
# `mine_error_patterns` groups on `(tool_name, error_type, params_signature)`;
# `candidate_pattern_key` deliberately drops the signature, because `Tool/error_type`
# is the shape the verdict ledger adjudicates (#530). Before this section the two
# granularities met at the writer, so 73 mined error patterns went to 25 paths and
# the last writer won: `Bash/logic` carried one bucket's numbers under a key that
# gates 19 buckets, and the run's own `Written:` count reported 73. The fix is a
# merge at emission (`merge_error_patterns`) — mining keeps its finer telemetry, the
# ledger keeps its coarse join, and each key gets exactly one file whose totals are
# the sum and the union over its buckets.

MERGE_TOOL = "Mergetest515"


def merge_rows():
    """Two signature buckets under one key, each corroborated in two sessions.

    `cmd:pytest` and `cmd:ls` are distinct `params_signature`s; both are
    `Mergetest515/not_found`, which is the collision the item measured as
    `Edit/not_found` on 2026-09-08 and `Bash/logic` (19 buckets) on 2026-09-14.
    """
    rows = []
    for i in (1, 2):
        rows.append(error_traj(f"pytest{i}", MERGE_TOOL, "not_found", "protocol",
                               {"command": "pytest tests/"}))
    for i in (1, 2, 3):
        rows.append(error_traj(f"ls{i}", MERGE_TOOL, "not_found", "protocol",
                               {"command": "ls -la"}))
    return rows


def merge_rows_two_keys():
    """`merge_rows()` plus a SECOND colliding key, so the mined corpus reaches exactly
    `MERGE_MULTI_SIGNATURE_KEYS_MIN` = 2 keys with more than one signature behind them:
    `Mergetest515/not_found` and `Mergetest1403/logic`, each with two distinct
    `params_signature`s corroborated across separate sessions.

    This is the smallest corpus on which the merged-total assertions of
    `test_a_live_week_emits_one_merged_candidate_per_error_key` are evidence at all,
    which is precisely why it is the corpus the volume skip must NOT fire on. It is the
    positive control beside `test_a_corpus_whose_error_keys_collide_on_one_key_skips_naming_both_numbers`:
    one fixture skips, this one must run, and a floor that was lowered to 1 or 2
    arbitrarily would let both pass for the wrong reason.
    """
    rows = merge_rows()
    for i in (1, 2):
        rows.append(error_traj(f"head{i}", "Mergetest1403", "logic", "protocol",
                               {"command": "head -5 notes.md"}))
    for i in (1, 2, 3):
        rows.append(error_traj(f"wc{i}", "Mergetest1403", "logic", "protocol",
                               {"command": "wc -l notes.md"}))
    return rows


def sequence_rows():
    """Rows that give `test_the_live_corpus_emits_one_file_per_sequence_pattern` both
    halves of its own vacuity guard: `retrying_traj` n-grams clear the #1181 emission
    gate (a failing step followed by the same call again), `adjacent_next_step_traj`
    n-grams are the refused half (a failing step followed by a DIFFERENT call touching
    the same file). Three sessions each, so both mine at the guard's threshold of 2.

    Used by `test_a_guard_that_reads_a_different_path_than_it_checks_is_caught`, where
    the point is that every assertion in the guard CAN pass — so the only thing left to
    break it is reading a directory other than the one it checked.
    """
    return ([retrying_traj(f"seqpin1403_{i}") for i in (1, 2, 3)]
            + [adjacent_next_step_traj(f"seqpin1403_{i}") for i in (1, 2, 3)])


def frontmatter_field(body: str, field: str) -> str:
    fm = body.split("---", 2)[1]
    line = next((ln for ln in fm.splitlines() if ln.startswith(f"{field}:")), "")
    return line.split(":", 1)[1].strip()


def test_two_signatures_under_one_error_key_mine_as_two_buckets():
    """The premise, pinned: the miner still sees the finer granularity. Merging is
    an emission decision, not a mining one — `--stats` must keep the breakdown."""
    patterns = mt.mine_error_patterns(merge_rows(), threshold=2)
    assert len(patterns) == 2
    assert {p["params_signature"] for p in patterns} == {"cmd:pytest", "cmd:ls"}
    assert len({mt.candidate_pattern_key(p) for p in patterns}) == 1
    assert mt.candidate_pattern_key(patterns[0]) == f"{MERGE_TOOL}/not_found"


def test_the_two_buckets_emit_one_file_with_merged_totals(tmp_path):
    """Clauses 1 + 2 + 6: one returned path, one file on disk, `occurrences:` the
    sum over the buckets and `sessions:` the size of their union — 5 sessions and
    5 calls here, not one bucket's 2."""
    patterns = mt.mine_error_patterns(merge_rows(), threshold=2)
    expected_occ = sum(p["total_calls"] for p in patterns)
    expected_sessions = len(set().union(*(set(p["sessions"]) for p in patterns)))

    written = mt.emit_candidates(patterns, tmp_path)
    files = sorted(tmp_path.glob("candidate-*.md"))

    assert len(written) == len(set(written)) == len(files) == 1, [str(p) for p in written]
    body = files[0].read_text(encoding="utf-8")
    assert frontmatter_field(body, "pattern") == f"{MERGE_TOOL}/not_found"
    assert int(frontmatter_field(body, "occurrences")) == expected_occ == 5
    assert int(frontmatter_field(body, "sessions")) == expected_sessions == 5
    assert frontmatter_field(body, "signature_buckets") == "2"
    # The breakdown is printed inside the file, so a reader can see how many
    # distinct mined patterns the one verdict over this key is standing in front of.
    assert "cmd:pytest" in body and "cmd:ls" in body


def test_reversing_the_mined_pattern_list_writes_byte_identical_candidates(tmp_path):
    """Clause 3: which bucket's evidence survived used to be dict iteration order."""
    patterns = mt.mine_error_patterns(merge_rows(), threshold=2)
    forward, backward = tmp_path / "fwd", tmp_path / "rev"
    mt.emit_candidates(patterns, forward)
    mt.emit_candidates(list(reversed(patterns)), backward)

    names = sorted(p.name for p in forward.glob("candidate-*.md"))
    assert names == sorted(p.name for p in backward.glob("candidate-*.md"))
    assert len(names) == 1
    for name in names:
        assert (forward / name).read_bytes() == (backward / name).read_bytes()


def test_merge_error_patterns_leaves_success_and_sequence_patterns_alone():
    """The merge is scoped to error patterns; `emit_candidates`'s other two feeds
    would lose their per-pattern rows if it were applied to them."""
    seq = {"type": "sequence", "ngram_size": 2, "sequence_str": "read -> edit",
           "sessions": {"s1", "s2"}, "total_calls": 4, "examples": [],
           "first_seen": "2026-09-08", "last_seen": "2026-09-08"}
    out = mt.merge_error_patterns([
        {"type": "success", "tool_name": "Read", "params_signature": "read_limit",
         "sessions": {"s1", "s2"}, "total_calls": 3, "examples": []}, seq])
    assert len(out) == 2, [p["type"] for p in out]
    assert [p["type"] for p in out] == ["success", "sequence"]
    assert "merged_from" not in out[0] and "merged_buckets" not in out[1]


def test_the_emitted_run_reports_one_line_per_key_with_its_merge_count(tmp_path):
    """Clause 4, through the CLI the nightly invokes (`run_miner`: the same flags as
    skills/trajectory-skill-mining/SKILL.md:46 over a synthetic corpus): one
    `Written:` line per distinct key, each naming how many mined patterns merged into
    it, and an `INDEX.md` with one entry per file written."""
    def row(key, command):
        return {
            "session_key": key, "agent_id": "lloyd", "session_class": "interactive",
            "timestamp": "2026-09-12T10:10:10Z",
            "tool_count": 1, "error_count": 1, "has_errors": True,
            "tools": [{"name": MERGE_TOOL, "is_error": True, "error_source": "protocol",
                       "sequence": 0, "params_summary": {"command": command},
                       "result_summary": "boom: no such file"}],
            "error_tools": [{"name": MERGE_TOOL, "sequence": 0,
                             "error_type": "not_found", "error_source": "protocol",
                             "params_summary": {"command": command}}],
            "signals": [],
        }

    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = ([row("p1", "pytest tests/"), row("p2", "pytest tests/"),
             row("l1", "ls -la"), row("l2", "ls -la")])
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    proc = run_miner(corpus, out)
    assert proc.returncode == 0, proc.stderr
    report = proc.stdout + proc.stderr

    written_lines = [ln for ln in report.splitlines() if "  Written: " in ln]
    files = [p for p in out.glob("candidate-*.md")]
    assert len(written_lines) == len(files), report
    key_line = [ln for ln in written_lines if MERGE_TOOL.lower() in ln.lower()]
    assert len(key_line) == 1, written_lines
    # Before emission, at the threshold the nightly actually runs: BOTH signature
    # buckets must qualify. Without this, the count asserted below could pass on a
    # corpus the threshold mines down to one bucket, where there is no collision to
    # merge at all and `1` would be the honest number.
    previous_dir = mt.TRAJECTORY_DIR
    mt.set_trajectory_dir(corpus)
    try:
        buckets = [q for q in mt.mine_error_patterns(
            mt.load_trajectories(days=9999, agent_filter=NIGHTLY_AGENT), threshold=2)
            if mt.candidate_pattern_key(q) == f"{MERGE_TOOL}/not_found"]
    finally:
        mt.set_trajectory_dir(previous_dir)
    assert len(buckets) == 2, [mt.candidate_pattern_key(q) for q in buckets]

    # The line states the count in units of the thing that was merged: 2 mined
    # patterns became 1 file. A bare `1` is indistinguishable from a run that mined
    # one pattern and merged nothing.
    assert f"{MERGE_TOOL}/not_found: 2 mined patterns merged onto this key" \
        in key_line[0], key_line[0]
    assert re.search(r"^occurrences: 4$", files[0].read_text(), re.MULTILINE)
    index = (out / "INDEX.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^- \[candidate-", index, re.MULTILINE)) == len(files)

    # And the counts reconcile: 4 mined buckets, 1 key for this tool, so exactly the
    # buckets that folded into another are reported as merged — never also as
    # suppressed, which is what a single `mined - written` number conflated.
    merge_lines = [ln for ln in report.splitlines()
                   if "merged onto a shared key" in ln]
    assert len(merge_lines) == 2, report          # stderr progress + SUMMARY line
    assert all(ln.rstrip().endswith(": 1") for ln in merge_lines), merge_lines
    assert "Candidates written:   1" in report, report


def test_the_verdict_ledger_join_still_resolves_every_coarse_error_key():
    """Clause 5: the join field is unchanged, so no already-decided candidate is
    orphaned. Read-only over the live ledger.

    Every coarse `Tool/error_type` key in `_pipeline/skills/reviews/verdicts.jsonl`
    must still be what `candidate_pattern_key` derives for a pattern with that tool,
    that error type and *some* signature, and `verdict_for` must return the row
    exactly when the ledger says it binds (terminal, and not lifted by the reopen
    rules). Widen the key to `tool/error_type/<signature>` and the derived key stops
    matching the stored one, so this fails — which is the point: that widening needs
    a migration of the existing rows, not just of the filename.
    """
    ledger = LIVE_VERDICT_LEDGER
    # Absent skips with the path in the reason (#1377) — an append-only ledger that
    # was never regenerated is a fact about the machine, and the same absence that
    # made this node red at base made every other round's `tests` rung red too. What
    # still fails is a ledger that EXISTS and has lost rows, which is the migration
    # defect this clause guards: pinned by
    # `test_a_present_ledger_below_its_coarse_key_floor_still_fails`.
    require_live_data(ledger, "verdict ledger", kind="file")
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    coarse = {}
    for row in rows:
        key = str(row.get("pattern_key") or "")
        if key.count("/") == 1 and not key.startswith("seq-"):
            coarse[key] = row          # latest-wins, as the store does
    assert len(coarse) >= 21, (
        f"only {len(coarse)} coarse keys in a ledger that held 21 on 2026-09-13 — a"
        " migration that dropped rows must not pass as a key-shape change")

    sv = mt._verdicts_module()
    # Which of these rows SHOULD resolve is decided here from the ledger text and the
    # module's two constants, never by calling `terminal_verdict`/`is_terminal` back:
    # asserting `(verdict_for(p) is not None) is sv.terminal_verdict(k) and ...` restates
    # the function's own internals and can only fail on a store break, so it would
    # certify any semantics the implementation happens to have (advisory on round
    # SM_20260914_140724). What the clause actually pins is the IDENTITY of the row a
    # key resolves to, which is asserted per key below.
    terminal = set(sv.TERMINAL_VERDICTS)
    window = timedelta(days=sv.REOPEN_AFTER_DAYS)
    now = datetime.now(tz=timezone.utc)
    resolved = 0
    for key, row in coarse.items():
        tool, error_type = key.split("/")
        pattern = {"type": "error", "tool_name": tool, "error_type": error_type,
                   "params_signature": "cmd:something", "total_calls": 0,
                   "sessions": {"s1"}}
        assert mt.candidate_pattern_key(pattern) == key, key
        verdict = mt.verdict_for(pattern, store=ledger)
        if verdict is None:
            continue
        resolved += 1
        assert verdict.get("pattern_key") == key, (key, verdict.get("pattern_key"))
        assert verdict.get("verdict") == row.get("verdict"), (key, verdict, row)
        assert str(verdict.get("decided_at")) == str(row.get("decided_at")), key
        assert (row.get("verdict") or "") in terminal, (key, row.get("verdict"))
        decided = row.get("decided_at")
        assert decided, key  # an absent decided_at never expires; pin the shape
        age = now - datetime.fromisoformat(str(decided).replace("Z", "+00:00"))
        assert age <= window, (key, str(decided), age.days)
    # Non-vacuity from the ledger text: 14 of the 21 coarse rows were terminal and
    # inside the window on 2026-09-21, so a change that stopped resolving any of them
    # drops this count. A ledger whose terminal rows have all expired would make the
    # loop above certify nothing, and this floor is what says so.
    assert resolved >= 12, (
        f"only {resolved} of {len(coarse)} coarse keys resolved a verdict; the ledger's"
        " terminal rows are close to expiring, and this test proves nothing at 0")


# ── live-data guard: one candidate per key over the real corpus ───────────────


def test_a_merged_key_does_not_reopen_on_growth_measured_in_other_units(tmp_path):
    """The >10x growth reopen compares a live count against a baseline the ledger
    recorded from ONE bucket's file (#530 seeded it before #515 merged buckets), so the
    two numbers have different denominators. Acting on that comparison would reopen
    every multi-bucket key on the first merged run and re-adjudicate a stack of
    patterns nobody asked to reopen — the decision Alan reserved on #515. So
    `verdict_for` skips the growth trigger for a unit with MORE THAN ONE bucket behind
    its key, while terminality and the 60-day expiry still bind. A key with one bucket
    keeps the rule, merged or not: there the live count and the stored baseline are the
    same denominator, so zeroing it would suppress a key entitled to reopen."""
    store = tmp_path / "verdicts.jsonl"
    sv = mt._verdicts_module()
    sv.record_verdict(store, pattern_key="Bash/logic",
                      verdict="rejected_false_positive", reason="one signature judged",
                      evidence_cmd="python3 -c 'print(1)'", occurrences=1,
                      decided_by="unit-test")
    merged = mt.merge_error_patterns([
        {"type": "error", "tool_name": "Bash", "error_type": "logic",
         "params_signature": f"cmd:prog{i}", "sessions": {f"s{i}", f"s{i}b"},
         "examples": [], "dates": {"2026-09-10"}, "total_calls": 10 + i}
        for i in range(12)])
    assert len(merged) == 1 and merged[0]["total_calls"] > 10, (
        "the fixture no longer exceeds 10x the stored baseline of 1")
    assert mt.verdict_for(merged[0], store=store) is not None, (
        "the merged unit stopped suppressing a key whose verdict was never revisited")
    one_bucket = {"type": "error", "tool_name": "Bash", "error_type": "logic",
                  "params_signature": "cmd:prog0", "sessions": {"s0"},
                  "examples": [], "dates": {"2026-09-10"}, "total_calls": 100}
    assert mt.verdict_for(one_bucket, store=store) is None, (
        "the per-bucket growth rule changed, which is not this item's to change")
    # The same single bucket, run through the merge: one bucket behind the key means
    # the denominators match, so the growth reopen must still fire. Passing every
    # merged unit through the skip — which `merged_buckets` also marks — would widen
    # suppression beyond the multi-bucket keys this item touches.
    passed_through = mt.merge_error_patterns([dict(one_bucket)])
    assert len(passed_through) == 1 and len(passed_through[0]["merged_buckets"]) == 1
    assert mt.verdict_for(passed_through[0], store=store) is None, (
        "a one-bucket key stopped evaluating the growth rule just because it was "
        "copied through the merge")
    # `write_candidate_file`, not `emit_candidates`, and with the tmp store passed:
    # otherwise `store_path` falls back to $SKILL_VERDICTS_STORE / the live ledger and
    # the `superseded_by_verdict` asserted below could be production's answer rather
    # than the one this test seeded.
    body = Path(mt.write_candidate_file(merged[0], tmp_path / "c",
                                        verdict_store=store)).read_text(encoding="utf-8")
    assert "status: superseded_by_verdict" in body, body[:400]
    assert "growth reopen does not evaluate" in body
    # The same unit against an empty ledger is `pending_review`: proof that the status
    # above came from the store this test wrote, not from the machine it runs on.
    blank = tmp_path / "empty-ledger.jsonl"
    free_body = Path(mt.write_candidate_file(merged[0], tmp_path / "d",
                                             verdict_store=blank)).read_text(encoding="utf-8")
    assert "status: pending_review" in free_body, free_body[:400]


def test_the_verdict_checker_reads_a_merged_candidate_as_one_unit(tmp_path):
    """The seam in the other program: `skill_verdicts.py` reads the frontmatter this
    miner writes, in its own process, and decides SKIP / REOPEN / PROCEED from it.

    A merged unit carries `occurrences:` summed over every signature bucket behind the
    key, while the ledger's `occurrences_at_decision` for the same key came from ONE
    bucket's file (#530 seeded the ledger before buckets were ever merged). Compared as
    a ratio the two numbers have different denominators, and the reader applied the >10x
    growth reopen to them anyway — flipping `SKIP` (the verdict binds) to `REOPEN`
    (Alan's reserved re-adjudication, arriving by arithmetic). `signature_buckets:` is
    what tells the two cases apart, so the reader has to see it.
    """
    store = tmp_path / "verdicts.jsonl"
    sv = mt._verdicts_module()
    sv.record_verdict(store, pattern_key="Bash/logic",
                      verdict="rejected_false_positive", reason="one signature judged",
                      evidence_cmd="python3 -c 'print(1)'", occurrences=1,
                      decided_by="unit-test")
    merged = mt.merge_error_patterns([
        {"type": "error", "tool_name": "Bash", "error_type": "logic",
         "params_signature": f"cmd:prog{i}", "sessions": {f"s{i}", f"s{i}b"},
         "examples": [], "dates": {"2026-09-10"}, "total_calls": 10 + i}
        for i in range(12)])
    assert merged[0]["total_calls"] > 10, "fixture no longer exceeds 10x baseline 1"
    cands = tmp_path / "cands"
    mt.write_candidate_file(merged[0], cands, verdict_store=store)

    # Same file, through the other program's own frontmatter reader.
    file = next(cands.glob("candidate-*.md"))
    key, status, occurrences, buckets = sv.read_candidate(file)
    assert (key, status) == ("Bash/logic", "superseded_by_verdict")
    assert occurrences == merged[0]["total_calls"]
    assert buckets == 12, "the reader cannot see how many patterns this file stands for"

    # And through the CLI a consolidation agent actually runs: separate process.
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"), "check",
         "--candidates", str(cands), "--store", str(store)],
        capture_output=True, text=True, timeout=120)
    out = proc.stdout
    assert proc.returncode == 0, proc.stderr
    assert "REOPEN Bash/logic" not in out, (
        f"the merged sum reopened a verdict through the checker:\n{out}")
    assert "SKIP Bash/logic" in out, out
    assert "12 mined patterns under this key" in out, (
        f"a coarse verdict must say how many patterns it gates:\n{out}")
    assert "skipped_by_verdict: 1" in out, out


def test_a_one_bucket_candidate_still_reopens_the_checker_on_growth(tmp_path):
    """The narrowing, from the reader's side: with one signature behind the key,
    `occurrences:` and the stored baseline share a denominator, so the >10x growth
    reopen must still fire. A field that blanket-disabled growth would silently widen
    suppression across keys this item has no business touching."""
    store = tmp_path / "verdicts.jsonl"
    sv = mt._verdicts_module()
    sv.record_verdict(store, pattern_key="Bash/logic",
                      verdict="rejected_false_positive", reason="small at decision",
                      evidence_cmd="python3 -c 'print(1)'", occurrences=1,
                      decided_by="unit-test")
    lone = {"type": "error", "tool_name": "Bash", "error_type": "logic",
            "params_signature": "cmd:prog0",
            "sessions": {f"s{i}" for i in range(30)},
            "examples": [], "dates": {"2026-09-10"}, "total_calls": 100,
            "first_seen": "2026-09-10", "last_seen": "2026-09-10"}
    # Through `merge_error_patterns`, which is where a real emission unit gets its
    # `first_seen`/`last_seen`: a one-bucket key comes back with a one-row breakdown.
    lone_unit = mt.merge_error_patterns([lone])[0]
    assert len(lone_unit["merged_buckets"]) == 1
    cands = tmp_path / "cands"
    mt.write_candidate_file(lone_unit, cands, verdict_store=store)
    assert sv.read_candidate(next(cands.glob("candidate-*.md")))[3] == 1
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"), "check",
         "--candidates", str(cands), "--store", str(store)],
        capture_output=True, text=True, timeout=120)
    assert "REOPEN Bash/logic" in proc.stdout, proc.stdout
    assert "skipped_by_verdict: 0" in proc.stdout, proc.stdout


def test_the_merge_moves_nobody_out_of_suppression():
    """The suppressed SET, not only the ledger join: every mined pattern a verdict
    gates today must still be gated after the merge. If the merge let any of them
    resolve to `pending_review`, already-rejected work would be back in the review
    queue (clause 5 read as a set, not only as row reachability).

    A bucket is identified by the triple `mine_error_patterns` groups on, not by its
    `params_signature`: measured over the live 7-day corpus on 2026-09-21, the string
    `file_md_signature` is carried by buckets under four different keys
    (`Grep/not_found`, `Read/logic`, `vault_read/not_found`, `vault_write/validation`),
    two of them gated and two not, so comparing a gated set of *signature strings*
    against a gated set of *keys* reports the two ungated ones as escapes. The first
    version of this test did exactly that and failed on a corpus where nothing leaked.
    """
    def bucket_id(q: dict) -> tuple:
        return (q["tool_name"], q["error_type"], q["params_signature"])

    # Absence of the corpus used to arrive here as `assert mined` failing on an empty
    # list, which reads the same as a corpus that mined nothing (#1377). Skipping on
    # the path, not on the count, is what keeps those two states distinct: an
    # existing corpus that yields no pattern still fails below.
    require_live_data(LIVE_CORPUS, "live trajectory corpus")
    previous = mt.TRAJECTORY_DIR
    mt.set_trajectory_dir(LIVE_CORPUS)
    try:
        mined = mt.mine_error_patterns(
            mt.load_trajectories(days=7, agent_filter="all",
                                 exclude_machine=False), threshold=2)
    finally:
        mt.set_trajectory_dir(previous)
    assert mined, ("no qualifying error pattern mined — the merge suppressed nobody "
                   "because there was nobody to suppress")
    assert len({bucket_id(q) for q in mined}) == len(mined), (
        "mine_error_patterns returned two buckets with the same grouping triple, so "
        "the identity below identifies nothing")
    gated_before = {bucket_id(q) for q in mined if mt.verdict_for(q)}
    assert gated_before, (
        f"the ledger gates none of the {len(mined)} mined patterns, so this compares "
        "two empty sets; a ledger whose coarse rows have all expired is a finding")
    gated_keys = {mt.candidate_pattern_key(q)
                  for q in mt.merge_error_patterns(mined) if mt.verdict_for(q)}
    leaked = sorted(f"{t}/{e} ({sig})" for t, e, sig in gated_before
                    if f"{t}/{e}" not in gated_keys)
    assert not leaked, (f"{len(leaked)} judged patterns re-arrived as pending_review: "
                        f"{leaked[:5]}")
    # The other direction, floored: a coarse key that gates buckets must not end up
    # gating none of them, and every gated bucket's key is one of the 30 keys the merge
    # emits — so the count of gated buckets cannot fall below the count that was gated.
    assert sum(1 for q in mined
               if mt.candidate_pattern_key(q) in gated_keys) >= len(gated_before)


def test_a_live_week_emits_one_merged_candidate_per_error_key(tmp_path):
    """Read-only over `_pipeline/trajectories`; emission goes to a temp dir.

    The item's acceptance check as written: paths returned == distinct paths ==
    files on disk, and every written error candidate's `occurrences:` / `sessions:`
    equal the sum and the union over the mined buckets that share its key. On
    2026-09-14 the same run returned 73 paths for 25 files and `Bash/logic` reported
    one bucket while gating 19.

    Two tiers of "not enough corpus", with two different answers, and conflating
    them is what made this node block every promotion for five days. No corpus at
    all: skip by name (#1377 — `_pipeline/` is gitignored, and after the 2026-09-22
    wipe it did not exist on the machine that runs this). A corpus that exists but
    cannot discriminate the merge: skip naming the floor and the observed count
    (#1403). Both are skips; neither is a pass. What stays fatal either way is a
    corpus deep enough to collide and totals that do not add up, which is what
    `test_a_two_key_fixture_corpus_runs_the_merge_guard_rather_than_skipping` pins,
    and a corpus that mines no qualifying error pattern at all is still a finding,
    asserted below.
    `exclude_machine=False` is
    deliberate — since #493 the nightly default drops worker/autonomy/inner-voice/browser
    sessions, and over a quiet week that corpus holds no qualifying error pattern at all.
    """
    require_live_data(LIVE_CORPUS, "live trajectory corpus")
    previous = mt.TRAJECTORY_DIR
    mt.set_trajectory_dir(LIVE_CORPUS)
    try:
        rows = mt.load_trajectories(days=7, agent_filter="all", exclude_machine=False)
        patterns = mt.mine_error_patterns(rows, threshold=2)
        assert patterns, ("no qualifying error pattern in a 7-day all-classes window — "
                          "a collision cannot be vacuously absent, so this is a finding")
        by_key = {}
        for pattern in patterns:
            by_key.setdefault(mt.candidate_pattern_key(pattern), []).append(pattern)
        multi = {k: v for k, v in by_key.items() if len(v) > 1}
        # Below the floor the merge cannot be discriminated EITHER way, and that is a
        # statement about how deep this machine's corpus happens to be, not about the
        # merge. `_pipeline/trajectories` restarted at zero with the 2026-09-22
        # deletion and regrows one bucket a day, so a hard floor of two here, asserted
        # rather than skipped on, red-blocked every promotion while the corpus refilled
        # (#1403 cause B);
        # `tests/_live_data.py::require_live_volume` is the rule for that state —
        # skip, naming the floor and the observed count beside the path. The floor is
        # this node's own `>= 2`, not truthy, lifted into
        # `MERGE_MULTI_SIGNATURE_KEYS_MIN` so a skip can never be softer than the
        # assertion it replaces, and it is NOT lowered to 1: one multi-bucket key
        # would let a merge that happened to fold a single pair pass, and the
        # merged-total assertions below are only evidence across a population
        # (measured 15 of 30 keys on the whole corpus on 2026-09-21, 1 of 12 keys on
        # the 2-file corpus on 2026-09-23). What the skip does not license is a silent
        # pass: at or above the floor every assertion below still runs and is fatal,
        # which is what
        # `test_a_two_key_fixture_corpus_runs_the_merge_guard_rather_than_skipping` pins.
        require_live_volume(list(multi), MERGE_MULTI_SIGNATURE_KEYS_MIN, LIVE_CORPUS,
                            "mined error keys with more than one signature behind them",
                            noun="keys")

        written = mt.emit_candidates(patterns, tmp_path)
        files = sorted(tmp_path.glob("candidate-*.md"))
        assert len(written) == len(set(written)) == len(files)

        error_keys = 0
        for body_files in files:
            body = body_files.read_text(encoding="utf-8")
            key = frontmatter_field(body, "pattern")
            if frontmatter_field(body, "type") != "error":
                continue
            error_keys += 1
            buckets = by_key[key]
            assert int(frontmatter_field(body, "occurrences")) == sum(
                p["total_calls"] for p in buckets), key
            assert int(frontmatter_field(body, "sessions")) == len(
                set().union(*(set(p["sessions"]) for p in buckets))), key
        assert error_keys == len(by_key)
    finally:
        mt.set_trajectory_dir(previous)


def test_a_full_live_week_emits_one_file_per_path_across_every_class(tmp_path):
    """Clauses 1 and 3 over the WHOLE emission, not over the error class alone.

    The review refused round SM_20260914_140724 on exactly this: the error class was
    already 1:1 (25 patterns returned / 25 distinct / 25 files) and the guard that
    proves it was scoped to `type == "error"` paths, while the same commit's full run
    returned 1073 paths for 1072 files — `candidate-seq-3-backlog-write-task-…-err-ba-
    20260914.md` written twice, silently, and its surviving bytes decided by input
    order. Clause 1 says "no path appears more than once in the list returned by
    `emit_candidates()`" about the LIST, and clause 3 says "each written candidate",
    so both are asserted here across every class the nightly feeds the writer.

    Read-only over `_pipeline/trajectories`; both emissions go to a temp dir. The
    corpus is skipped when the path is absent (#1377) and every assertion below still
    fires on a corpus that exists but writes no sequence candidate — which is the
    review's own failure shape from round SM_20260914_140724, and is pinned by
    `test_a_corpus_that_writes_no_sequence_candidate_fails_rather_than_skips`.
    """
    require_live_data(LIVE_CORPUS, "live trajectory corpus")
    previous = mt.TRAJECTORY_DIR
    mt.set_trajectory_dir(LIVE_CORPUS)
    try:
        rows = mt.load_trajectories(days=7, agent_filter="all", exclude_machine=False)
        # The same concatenation `main()` performs, at the runbook's threshold.
        patterns = (mt.mine_error_patterns(rows, threshold=2)
                    + mt.mine_success_patterns(rows, threshold=2)
                    + mt.mine_sequence_patterns(rows, threshold=2))
        assert patterns, "a 7-day all-classes window mined nothing: vacuous below"
        forward, backward = tmp_path / "fwd", tmp_path / "rev"
        written = mt.emit_candidates(patterns, forward)
        reversed_written = mt.emit_candidates(list(reversed(patterns)), backward)

        # Clause 1: the returned list, all classes, has no repeats, and its length is
        # the number of files that exist — the two halves of "1073 returned / 1072 on
        # disk", which no per-class assertion can see.
        counts: dict = {}
        for path in written:
            counts[path] = counts.get(path, 0) + 1
        repeated = sorted(str(p.name) for p, n in counts.items() if n > 1)
        assert not repeated, (
            f"{len(written)} paths returned, {len(counts)} distinct; repeated: "
            f"{repeated[:3]}")
        files = sorted(forward.glob("candidate-*.md"))
        assert len(written) == len(files), (
            f"emit_candidates returned {len(written)} paths and {len(files)} "
            "candidate-*.md exist on disk")

        # Non-vacuity: the class the review caught must be in this emission, or the
        # all-class claim above is the error-class claim under a different name. The
        # class is read out of each file's own `type:` field, not off the filename,
        # so a mislabelled writer cannot make the split look populated.
        kinds = [frontmatter_field(f.read_text(encoding="utf-8"), "type")
                 for f in files]
        assert kinds.count("sequence") >= 1, (
            f"no sequence candidate was written ({kinds.count('error')} error, "
            f"{len(files)} total), so this run never touched the class whose file aliased")
        assert kinds.count("error") >= 1, "no error candidate was written"

        # Clause 3, every class: the reversed input writes the same names and the same
        # bytes, so which pattern's evidence survives is not an iteration-order answer.
        back_files = sorted(backward.glob("candidate-*.md"))
        assert [f.name for f in files] == [f.name for f in back_files]
        differing = [a.name for a, b in zip(files, back_files)
                     if a.read_bytes() != b.read_bytes()]
        assert not differing, (
            f"{len(differing)} of {len(files)} candidates differ between the forward "
            f"and reversed run: {differing[:3]}")
        assert sorted(p.name for p in reversed_written) == sorted(p.name for p in written)
    finally:
        mt.set_trajectory_dir(previous)


def test_a_colliding_name_raises_before_a_single_candidate_is_written(tmp_path):
    """The guard's scope and its timing, both from the review's finding.

    Scope: the first guard looked only at error paths, so the aliased
    `candidate-seq-3-…` file in the refusing run was written twice with the guard
    silent. A post-merge alias in a NON-error class must therefore trip it, and the
    only alias left in the sequence class is two patterns whose n-grams are equal —
    which `mine_sequence_patterns` cannot produce (its dict is keyed by the n-gram),
    so the fixture supplies it the way a future caller would.

    Timing: the old guard raised AFTER the loop, so the collision had already
    replaced one file's bytes and the raise merely reported the loss. The item's
    requirement is "an error rather than a lost file", so nothing may reach disk.
    """
    def ngram(seq):
        return {"type": "sequence", "ngram_size": 2, "sequence_str": seq,
                "has_error_recovery": True, "sessions": {"s1"}, "total_calls": 4,
                "examples": [], "first_seen": "2026-09-20", "last_seen": "2026-09-20"}

    pair = [ngram("read -> edit"), ngram("read -> edit")]
    assert mt.candidate_filename(pair[0]) == mt.candidate_filename(pair[1])
    with pytest.raises(AssertionError, match="last-writer-wins"):
        mt.emit_candidates(pair, tmp_path)
    assert not list(tmp_path.glob("candidate-*.md")), (
        "the guard fired after the writes, so the first candidate's evidence was "
        "already gone — the collision was reported, not refused")


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
# already carries: `platform`, `source`, and the session id's shape (#1143).
#
# #1143 amended #493's rule on one point: `inner_voice` was read as "the observer
# took this turn", and it is not — it is the switch that turns the observer on for
# a human's chat, so the class it produced (`inner-voice`, 103 sessions on this
# machine, and all 52 `browser` ones) excluded the chats people actually type into
# while keeping the scripted ids. Admission is `app.sessions_io.is_user_session`'s
# decision now; the extractor keeps no second list of machine platforms.
#
# These fixtures are the part a live-data test cannot prove: two session files
# with the SAME filename stem, one human and one machine. Anything that reads the
# name gives the same answer for both.

CLASS_STEM = "20260912_101010_autocode_beef"   # loop-shaped, used for both classes

# (platform, inner_voice, expected class) for a session whose id IS the three-part
# chat shape. Admission is `app.sessions_io.is_user_session`'s, so `inner_voice`
# appears in this table only to prove it changes nothing: every row with the same
# platform and a different `inner_voice` has the same class (#1143 clause 1). The
# `inner-voice` class of #493 is gone — those 103 sessions on this machine are
# Mission Control chats typed by a person, which is the misclassification #1143
# exists to fix. `browser` stays its own class rather than folding into
# `interactive`: it is human-initiated, and it is still reportable separately.
SESSION_CLASS_TABLE = [
    ("mission-control", False, "interactive"),
    ("mission-control", True, "interactive"),
    ("browser", True, "browser"),
    ("browser", False, "browser"),
    ("worker", False, "worker"),
    ("worker", True, "worker"),
    ("autonomy", False, "autonomy"),
    ("autonomy", True, "autonomy"),
    ("e2e-harness", False, "smoke"),
    (None, False, "unknown"),
]

#: A chat-shaped id, so a class in this table is never the id rule's doing.
CHAT_ID = "20260912_101010_9f2a1c"


def write_class_session(dir_, stem, platform, inner_voice=False, source=None,
                        session_id=None):
    """Write a session JSON with the fields the classifier reads.

    One corroborated `Bash` failure, so `parse_session` returns a row rather than
    None for a tool-less session. `session_id` defaults to `stem`; passing it
    deliberately is what separates clause 3's id-shape claim from clause 1's — the
    classifier reads the field, never the filename (#493 clauses 1-2).
    """
    d = Path(dir_)
    d.mkdir(parents=True, exist_ok=True)
    body = {
        "session_id": session_id if session_id is not None else stem,
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
    """Same stem, two classes — the discriminating case for `path.stem`.

    Both files are named `20260912_101010_autocode_beef`; only the fields inside
    differ, so any rule that opened the path would answer for both.
    """
    worker = et.parse_session(write_class_session(
        tmp_path / "a", CLASS_STEM, "worker", source="autocode",
        session_id=CHAT_ID))
    human = et.parse_session(write_class_session(
        tmp_path / "b", CLASS_STEM, "mission-control", session_id=CHAT_ID))
    assert worker["session_class"] == "worker"
    assert human["session_class"] == "interactive"
    # `agent_id` is still derived from the stem and is identical for both, so the
    # class below it provably was not.
    assert worker["agent_id"] == human["agent_id"] == "lloyd"


def test_an_autonomy_prefixed_stem_does_not_make_a_session_non_interactive(tmp_path):
    """The old rule's own positive case, reversed: the filename says autonomy, the
    session JSON says a human drove it from Mission Control. The id inside the file
    is the chat shape, so the id rule does not overrule that (#1143 clause 3)."""
    traj = et.parse_session(write_class_session(
        tmp_path, "autonomy_task68", "mission-control", session_id=CHAT_ID))
    assert traj["session_class"] == "interactive"
    assert traj["agent_id"] == "autonomy"   # legacy field keeps its old meaning


def test_classify_session_needs_no_filename_at_all():
    """The classifier's input is the parsed JSON, so there is no path to read."""
    assert et.classify_session({"platform": "worker"}) == "worker"
    assert et.classify_session({}) == "unknown"


# ── clause 1 (#1143): inner_voice is not a class, and browser is human ────────

@pytest.mark.parametrize("platform,inner_voice,expected", SESSION_CLASS_TABLE)
def test_the_class_comes_from_platform_and_not_from_inner_voice(
        platform, inner_voice, expected):
    """The same chat-shaped id at both settings of `inner_voice`.

    `inner_voice: true` is what the Inner Voice tab and the Chrome extension send
    when a person starts a chat (`InnerVoicePage.tsx:146`,
    `chrome-extension/src/background/lloyd-client.ts:15`), so a classifier that
    read it as an observer turn classified Alan's chats away. #493 asserted the
    opposite of every row in this table with `inner_voice` true.
    """
    assert et.classify_session(
        {"platform": platform, "inner_voice": inner_voice},
        session_id=CHAT_ID) == expected


def test_interactive_is_every_human_platform_whatever_inner_voice_says():
    """Clause 1 in both directions, derived from the classifier and nothing else:
    every platform the backend records (plus one it does not) crossed with every
    truthiness outcome of `inner_voice`. The interactive set is exactly the
    chat-shaped Mission Control sessions at all seven `inner_voice` values, and no
    worker, autonomy, harness or machine-platform session is ever in it.

    Not read out of SESSION_CLASS_TABLE — that table is hand-written, so comparing
    it against a literal set would pass with a classifier that called every session
    interactive.
    """
    platforms = ["mission-control", "worker", "autonomy", "browser",
                 "e2e-harness", "slack", None]
    inner_voice_values = [True, False, 1, 0, "", None, "true"]
    interactive = {(p, bool(iv))
                   for p in platforms for iv in inner_voice_values
                   if et.classify_session({"platform": p, "inner_voice": iv},
                                          session_id=CHAT_ID)
                   == et.INTERACTIVE_CLASS}
    assert interactive == {("mission-control", b)
                           for b in (True, False)}, interactive
    assert et.classify_session({"platform": "browser"},
                               session_id=CHAT_ID) in mt.HUMAN_CLASSES


@pytest.mark.parametrize("platform", ["browser", "mission-control"])
def test_admission_is_is_user_sessions_and_nothing_else(monkeypatch, platform):
    """Clause 2: the classifier holds no second list of machine platforms.

    The fixture is a platform the classifier admits *today*, which is the only
    kind that can discriminate. Patching in a platform nobody had heard of (`slack`
    was the first attempt here) returns `unknown` before the patch and `unknown`
    after it, so that test still passes with the `is_user_session` call deleted
    outright, or swapped for a deny-list of this module's own. `browser` answers a
    human class before the patch and a non-human class on the very next call after
    `NON_USER_PLATFORMS` gains it, and no private rule can produce that flip.

    Mutating the upstream set instead of the extractor is the claim itself: a
    platform the chat listings and the ambient-delivery path already refuse has to
    stop being mined with no change to this file, because a mining pool that
    disagreed with them would be mining sessions nobody reads. The read is through
    `is_user_session`, so the patch is visible at call time and not cached at
    import.
    """
    from app import sessions_io
    data = {"platform": platform, "inner_voice": True}
    before = et.classify_session(data, session_id=CHAT_ID)
    assert before in et.HUMAN_CLASSES, (
        f"{platform} classified {before} before the patch; the discriminating "
        "fixture has to start out admitted")
    monkeypatch.setattr(sessions_io, "NON_USER_PLATFORMS",
                        frozenset({"autonomy", "worker", platform}))
    after = et.classify_session(data, session_id=CHAT_ID)
    assert after not in et.HUMAN_CLASSES, (
        f"{platform} still classified {after} after NON_USER_PLATFORMS gained it: "
        "admission came from a rule of this module's own, not is_user_session's")
    assert after == et.UNKNOWN_CLASS, (
        f"{platform} -> {after}: a refused platform keeps its SESSION_CLASS human "
        "label instead of falling out of it")
    monkeypatch.undo()


@pytest.mark.parametrize("platform", sorted(
    __import__("app.sessions_io", fromlist=["x"]).NON_USER_PLATFORMS))
def test_every_non_user_platform_is_excluded_from_a_class_fixture(
        tmp_path, platform):
    """One fixture per platform in NON_USER_PLATFORMS, at both `inner_voice`
    settings: none of them is ever a human class, and none survives the default
    exclusion."""
    for iv in (True, False):
        cls = et.classify_session({"platform": platform, "inner_voice": iv},
                                  session_id=CHAT_ID)
        assert cls not in et.HUMAN_CLASSES, f"{platform} inner_voice={iv} -> {cls}"
        traj = et.parse_session(write_class_session(
            tmp_path, CLASS_STEM, platform, inner_voice=iv, session_id=CHAT_ID))
        assert traj["session_class"] == cls


def test_the_classified_session_records_its_source_producer(tmp_path):
    """`source` names the loop that ran the session (`autotriage`, `autocode`,
    `autonomy-task:68`), which is what makes a dropped session attributable."""
    traj = et.parse_session(write_class_session(
        tmp_path, CLASS_STEM, "worker", source="autotriage"))
    assert traj["session_source"] == "autotriage"


# ── clause 3 (#1143): a human platform with a scripted id is `test` ──────────

#: Ids created through `POST /api/sessions/create` with `platform:
#: mission-control` by test and harness code: the sandbox four-part shape
#: (`<8>_<6>_<slug>_<4hex>`) and the hand-named prefixes seen in the store.
SCRIPTED_IDS = [
    "20260909_153841_autonomy_1a2b",   # sandbox experiment, four parts
    "v2-bench-mcp-01",
    "dbg_tool_caps",
    "soak_20260905_1200",
    "20260906_113956_selftest_read",
    "20260910_101010_ab12cd_extra",    # five parts
]


@pytest.mark.parametrize("sid", SCRIPTED_IDS)
@pytest.mark.parametrize("platform", ["mission-control", "browser"])
def test_a_non_chat_shaped_id_on_a_human_platform_classifies_test(platform, sid):
    """Clause 3: the shape, not the prefix, is the rule.

    Every one of these rides a human platform, which is why `is_user_session`
    alone cannot exclude them — the census that found them is the one in #1143:
    all 42 sessions the old classifier kept as `interactive` were ids like these.
    """
    assert et.classify_session({"platform": platform}, session_id=sid) == "test"


#: The three suffixes the store actually holds on a chat-shaped id, censused over
#: `~/lloyd/sessions` on 2026-09-18: 163 files named `<8 digits>_<6 digits>_<6
#: chars>`, of which 152 are `iv<4 hex>` (the Inner Voice tab, the sidebar chat and
#: the Chrome extension all mint them through `POST /api/sessions/create`), 9 are
#: plain `<6 hex>` (the chat path), and 2 are `obs<hex>` — the latter two carry
#: `platform: mission-control`, `inner_voice: true` and a real user turn. An id
#: rule narrowed to the plain-hex form would drop exactly the chats #1143 exists to
#: recover, so this list is the negative control on clause 3.
LIVE_CHAT_IDS = [
    "20260912_101010_9f2a1c",     # chat path
    "20260912_101010_iv0484",     # POST /api/sessions/create, observer enabled
    "20260904_211536_obs40b",     # observer-named chat, 1 user turn on disk
]


@pytest.mark.parametrize("sid", LIVE_CHAT_IDS)
@pytest.mark.parametrize("platform,expected",
                         [("mission-control", et.INTERACTIVE_CLASS),
                          ("browser", "browser")])
def test_every_chat_id_shape_the_store_holds_stays_human(platform, expected, sid):
    """Clause 3's other half: the id rule fires on scripted ids and on nothing else.

    Both human platforms, each id shape a person's chat really has on disk. The
    assertion is membership in `HUMAN_CLASSES` as well as the exact label, because
    what the nightly exclusion reads is the set, not the string.
    """
    cls = et.classify_session({"platform": platform, "inner_voice": True},
                              session_id=sid)
    assert cls == expected
    assert cls in mt.HUMAN_CLASSES


def test_e2e_harness_sessions_are_still_smoke_whatever_their_ids_look_like():
    """The harness platform is decided before the id shape is consulted: the 3
    live `e2e-harness` sessions are named `e2e_selfmod*_1788*`, and they were
    already `smoke` under #493. A change to the id rule must not silently re-label
    a class the nightly report already counts."""
    for sid in ("e2e_selfmod_1788756141", "e2e_selfmod363_1788758995", CHAT_ID):
        assert et.classify_session({"platform": "e2e-harness"},
                                   session_id=sid) == "smoke"
        assert "smoke" not in et.HUMAN_CLASSES


def test_the_id_rule_reaches_a_legacy_corpus_row_through_the_store_join(
        tmp_path, monkeypatch):
    """The seam #1143 depends on for a backfill-free rollout.

    A corpus row with no emitted `session_class` is classified by the session JSON
    behind it, so a scripted id written off as interactive by the OLD extractor is
    reclassified to `test` at read time — every historical row, 08-22 onwards,
    without rewriting a byte of the corpus. The key is the four-part sandbox shape
    and the JSON carries no id field at all, which is why the join passes the key.
    """
    store = isolated_store(tmp_path, monkeypatch)
    (store / f"{CLASS_STEM}.json").write_text(
        json.dumps({"platform": "mission-control", "inner_voice": False}),
        encoding="utf-8")
    corpus = write_traj_corpus(tmp_path / "corpus", [(CLASS_STEM, "interactive")])
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    kept = mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert kept == [], "a scripted id rode the reclassification into the corpus"
    assert counts["dropped"] == {"test": 1}


# ── clause 3 (#493): the exclusion is one flag over one corpus ───────────────

MACHINE_ROWS = [("i1", "interactive"), ("i2", "interactive"),
                ("b1", "browser"),
                ("w1", "worker"), ("w2", "worker"), ("a1", "autonomy"),
                ("t1", "test"), ("v1", "inner-voice")]

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


def test_load_trajectories_keeps_only_human_rows_when_the_exclusion_is_on(
        tmp_path, monkeypatch):
    """Both human classes are kept and everything else is dropped.

    The `inner-voice` row is a pre-#1143 emitted class that no classifier produces
    any more; it stays dropped, so a corpus written by the old extractor cannot
    leak an observer-ish row into the pool while the rows are being re-read.
    """
    corpus = write_traj_corpus(tmp_path / "corpus", MACHINE_ROWS)
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    kept = mt.load_trajectories(days=9999, agent_filter="all")
    assert [t["session_key"] for t in kept] == ["i1", "i2", "b1"]


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
    assert counts["dropped"] == {"worker": 2, "autonomy": 1, "test": 1,
                                 "inner-voice": 1}
    assert counts["kept"] == {"interactive": 2, "browser": 1}


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


#: Chat-shaped corpus keys for the store-join tests. Since #1143 the id shape is
#: half of what the join decides, and the join passes the corpus key as the id, so
#: a bare `i9` would classify `test` for a reason the test is not about.
#: Chat-shaped corpus keys for the store-join tests. Since #1143 the id shape is
#: half of what the join decides and the join passes the corpus key as the id, so a
#: four-part stem would classify `test` for a reason the test is not about.
CHAT_HUMAN = "20260910_090000_aaaaaa"
CHAT_HUMAN2 = "20260911_090000_bbbbbb"


def test_a_class_less_row_is_rescued_as_interactive_by_the_store(
        tmp_path, monkeypatch):
    """The positive branch of the join: a legacy row with no emitted class is kept
    when the session JSON behind it says a human drove it from Mission Control.
    Without it the exclusion would blank the 7-day window — every live row is
    class-less — and emit nothing, which is clause 6's forbidden outcome."""
    store = isolated_store(tmp_path, monkeypatch)
    write_class_session(store, CHAT_HUMAN, "mission-control")
    corpus = write_traj_corpus(tmp_path / "corpus", [(CHAT_HUMAN, None)])
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    kept = mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert [t["session_key"] for t in kept] == [CHAT_HUMAN]
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
    write_class_session(store, CHAT_HUMAN2, "mission-control")
    corpus = write_traj_corpus(
        tmp_path / "corpus", [(CHAT_HUMAN2, "autonomy")])
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", corpus)
    counts: dict = {}
    kept = mt.load_trajectories(days=9999, agent_filter="all", class_counts=counts)
    assert [t["session_key"] for t in kept] == [CHAT_HUMAN2]
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
    """The miner compares a string, so the two modules must agree on it — and the
    human/non-human split is the extractor's set, read through the loaded module
    rather than restated in the miner."""
    assert mt.INTERACTIVE_CLASS == et.INTERACTIVE_CLASS == "interactive"
    # `==`, not `is`: the miner loads the extractor by path, so it holds its own
    # module object and its own copy of the set. Agreement across the boundary is
    # the claim; object identity is not available and not needed.
    assert mt.HUMAN_CLASSES == et.HUMAN_CLASSES
    assert not (mt.HUMAN_CLASSES & et.MACHINE_CLASSES), (
        "a class that is both admitted and dropped: the exclusion's answer depends "
        "on which branch it reaches first")


def test_a_by_path_boot_of_the_extractor_imports_its_admission_rule(tmp_path):
    """The process boundary #1143 adds, with a test across it.

    The nightly runs `python3 ~/lloyd/scripts/extract-trajectories.py`: `scripts/`
    is not a package, so that puts `scripts/` and not the repo root on `sys.path`,
    and the classifier's admission rule now lives in `app.sessions_io`. Without the
    path insertion the failure is a `ModuleNotFoundError` at import — the whole
    extraction step dies before it reads a session, and it dies in the nightly,
    where pytest has not put the checkout on the path. Hence a real subprocess, from
    a working directory that is not the checkout.

    The second half is the other half of clause 2: the name `is_user_session` in the
    extractor is the upstream function itself, not a local stand-in for it.
    """
    from app import sessions_io
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "extract-trajectories.py"),
         "--help"],
        capture_output=True, text=True, timeout=120, cwd=tmp_path)
    report = proc.stdout + proc.stderr
    assert proc.returncode == 0, report
    assert "ModuleNotFoundError" not in report and "Traceback" not in report, report
    assert et.is_user_session is sessions_io.is_user_session


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
                                             "mission-control",
                                             session_id=CHAT_ID)),
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


def run_miner(corpus, out_dir, extra_args=(), sessions_dir=None, agent=None):
    """Run the real command line the nightly runs.

    `--sessions-dir` defaults to an empty directory, not the live
    `~/lloyd/sessions`: since #1143 every row's class is resolved through that
    store, so a subprocess test that left it alone would take its expected counts
    from whatever happened to be on the machine. Pass `sessions_dir` to say where
    the sessions came from.

    `agent` overrides `--agent`, which defaults to the nightly skill's value. It
    is a real flag of the command line, not a shortcut: #998 is about what the
    exit status does when a filter is mis-set, so the mis-set filter has to be
    the flag the operator typed.
    """
    store = sessions_dir if sessions_dir is not None else (corpus.parent / "store")
    store.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(MINER_PATH), "--trajectory-dir", str(corpus),
           "--sessions-dir", str(store),
           "--agent", agent or NIGHTLY_AGENT, "--days", "9999", "--threshold", "2",
           "--output-dir", str(out_dir), *extra_args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180)


def graphex_candidate(out_dir):
    files = [p for p in Path(out_dir).glob("*.md") if "graphex493" in p.name]
    assert len(files) == 1, [p.name for p in Path(out_dir).glob("*.md")]
    return files[0].read_text(encoding="utf-8")


def test_the_nightly_mining_run_excludes_machine_sessions_by_default(tmp_path):
    """Two interactive sessions and three machine sessions carry the SAME
    failure. The candidate must report 2 sessions, not 5 — the loop's cadence is
    the thing that used to push patterns over the threshold.

    The second half is #998: the same corpus with the two human rows removed is a
    window that holds rows and selects none of them. Silence in the *log* on a
    machine-only window is fine — the per-class tally is the log — but exiting 0
    while rewriting the candidate index is not, so that run must fail.
    """
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

    machine_only = write_traj_corpus(tmp_path / "machine-only", [])
    (machine_only / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows[2:]) + "\n", encoding="utf-8")
    mo_out = tmp_path / "machine-only-cands"
    mo_proc = run_miner(machine_only, mo_out)
    mo_report = mo_proc.stdout + mo_proc.stderr
    assert mo_proc.returncode != 0, mo_report
    # The tallies the passing run prints are still printed by the failing one:
    # the exit status is added, the diagnosis is not replaced by it.
    assert "Machine-class sessions dropped: 3" in mo_report, mo_report
    assert "dropped worker: 2" in mo_report, mo_report
    assert "dropped autonomy: 1" in mo_report, mo_report
    assert not (mo_out / "INDEX.md").exists(), (
        "a run that selected nothing still rewrote the candidate index")


def test_a_filter_that_selects_nothing_from_a_non_empty_window_fails_loudly(
        tmp_path):
    """#998 clause 1, across the real command line.

    Three human sessions, all written by the extractor with `agent_id: lloyd`, and
    the documented `--agent worker` filter: the session-class exclusion admits all
    three rows and the agent filter then rejects every one of them. That is a
    mis-set selector, not a quiet day, so the run must exit non-zero and name both
    the filter it was given and the number it excluded — which is the case the
    live nightly hits, since #493 made `worker` a machine class and no live row
    carries `agent_id` worker at all (#494).
    """
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = [miner_row(f"i{n}", "interactive") for n in (1, 2, 3)]
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    proc = run_miner(corpus, out, agent="worker")
    report = proc.stdout + proc.stderr
    assert proc.returncode == mt.FILTER_SELECTED_NOTHING_EXIT, report
    # Names the filter value...
    assert "--agent worker" in report, report
    # ...and the count it excluded, in both halves of the selection: the rows the
    # agent filter rejected and the rows the session-class rule rejected.
    assert "selected 0 of 3 row(s)" in report, report
    assert "excluded by --agent worker: 3 row(s)" in report, report
    assert "agent-excluded lloyd: 3" in report, report
    # Not the empty-window verdict, which would tell the reader to wait for data.
    assert "Window empty" not in report, report
    assert "SUMMARY" not in report, (
        "a dead run still printed the ordinary SUMMARY")


def test_an_empty_window_is_a_quiet_day_and_not_a_failure(tmp_path):
    """#998 clause 2: a window with no trajectory rows at all exits 0 and says so
    in one line that is unmistakably different from the filter-selected-nothing
    message, so a reader can tell 'nothing happened' from 'your filter is wrong'."""
    corpus = tmp_path / "corpus"          # exists, holds no bucket file at all
    corpus.mkdir()
    out = tmp_path / "cands"
    proc = run_miner(corpus, out)
    report = proc.stdout + proc.stderr
    assert proc.returncode == 0, report
    empty_lines = [l for l in report.splitlines() if "Window empty" in l]
    assert len(empty_lines) == 1, report
    assert "no trajectory rows" in empty_lines[0], empty_lines[0]
    assert "ERROR" not in report, report
    assert "--agent" not in report, report


def test_a_failed_selection_leaves_the_index_byte_identical(tmp_path):
    """#998 clause 3: `INDEX.md` is neither created nor modified by a run that
    exits under the filter-selected-nothing verdict.

    Both halves matter. The live index at
    `_pipeline/skills/candidates/INDEX.md` is gitignored (`.gitignore:25`), so a
    rewrite that drops 3,920 rows to `Total candidates: 0` is invisible to `git
    status` and to any diff-based review — the only guard possible is that the run
    does not write it. And a scratch `--output-dir` must not gain an index it did
    not have, because `write_index` is what a downstream agent reads next.
    """
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = [miner_row("i1", "interactive"), miner_row("i2", "interactive")]
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = tmp_path / "cands"
    out.mkdir()
    sentinel = "---\ntype: index\n---\n\n- **Total candidates:** 2\n"
    index = out / "INDEX.md"
    index.write_text(sentinel, encoding="utf-8")

    proc = run_miner(corpus, out, agent="worker")
    assert proc.returncode == mt.FILTER_SELECTED_NOTHING_EXIT, (
        proc.stdout + proc.stderr)
    assert index.read_text(encoding="utf-8") == sentinel, (
        "the index was rewritten by a run that loaded nothing")

    fresh = tmp_path / "fresh"
    proc2 = run_miner(corpus, fresh, agent="worker")
    assert proc2.returncode == mt.FILTER_SELECTED_NOTHING_EXIT, (
        proc2.stdout + proc2.stderr)
    assert not (fresh / "INDEX.md").exists(), "INDEX.md was created by a dead run"
    assert not list(fresh.glob("candidate-*.md")), (
        "a dead run emitted candidate files")


def test_stats_gives_the_same_verdict_as_mining_for_both_empties(tmp_path):
    """#998 clause 4: `--stats` is a read-only mode, not a mute button.

    Before this it exited 0 on both empties, so the one mode an operator runs to
    ask 'is the pipeline alive' answered the question exactly as a productive run
    would. The mining path and the stats path must agree on both cases.
    """
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = [miner_row("i1", "interactive"), miner_row("i2", "interactive")]
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Non-empty window, filter selected nothing -> same failure as mining.
    stats_dead = run_miner(corpus, tmp_path / "stats-dead",
                           agent="worker", extra_args=("--stats",))
    mining_dead = run_miner(corpus, tmp_path / "mine-dead", agent="worker")
    report = stats_dead.stdout + stats_dead.stderr
    assert stats_dead.returncode == mining_dead.returncode, report
    assert stats_dead.returncode == mt.FILTER_SELECTED_NOTHING_EXIT, report
    assert "--agent worker" in report, report
    assert "excluded by --agent worker: 2 row(s)" in report, report

    # Genuinely empty window -> exit 0 on both paths, with the quiet-day line.
    empty_corpus = tmp_path / "empty-corpus"
    empty_corpus.mkdir()
    stats_empty = run_miner(empty_corpus, tmp_path / "stats-empty",
                            extra_args=("--stats",))
    mining_empty = run_miner(empty_corpus, tmp_path / "mine-empty")
    assert stats_empty.returncode == 0, stats_empty.stdout + stats_empty.stderr
    assert mining_empty.returncode == 0, mining_empty.stdout + mining_empty.stderr
    assert "Window empty" in stats_empty.stdout + stats_empty.stderr
    assert "ERROR" not in stats_empty.stdout + stats_empty.stderr


def test_the_inner_voice_chats_are_the_corpus_the_default_nightly_run_mines(
        tmp_path):
    """#1143 clause 4, across the real command line and the store join.

    Two Mission Control chats typed with the observer switched on and two worker
    runs share ONE failure signature. The default nightly path — no
    `--include-machine` — must write the candidate on the strength of the two human
    sessions, and the drop tally must name only the two worker sessions. Before
    #1143 the same corpus emitted nothing at all: both human rows were labelled
    `inner-voice` and excluded, so the threshold had one class left to reach and it
    was machine traffic.
    """
    chat_one, chat_two = "20260912_101010_h00001", "20260912_101010_h00002"
    work_one, work_two = "20260912_101010_w00001", "20260912_101010_w00002"
    store = tmp_path / "store"
    for key in (chat_one, chat_two):
        write_class_session(store, key, "mission-control", inner_voice=True)
    for key in (work_one, work_two):
        write_class_session(store, key, "worker", source="autocode")
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = [miner_row(k, None) for k in (chat_one, chat_two, work_one, work_two)]
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    proc = run_miner(corpus, out, sessions_dir=store)
    assert proc.returncode == 0, proc.stderr
    report = proc.stdout + proc.stderr
    # Only the machine sessions are dropped; neither chat appears in the tally.
    assert "Machine-class sessions dropped: 2" in report, report
    assert re.search(r"^\s+dropped worker: 2$", report, re.MULTILINE), report
    assert "dropped interactive" not in report, report
    assert chat_one not in report and chat_two not in report, report
    fm = graphex_candidate(out)
    assert re.search(r"^sessions: 2$", fm, re.MULTILINE), fm
    assert chat_one in fm and chat_two in fm, fm
    assert work_one not in fm, fm


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
# and the clause it pins would never be graded. Read-only against both, and each is
# gated by `tests/_live_data.py`: absent -> named skip, present -> every assertion
# below still runs (#1377).
LIVE_CORPUS = production_data_root() / "_pipeline" / "trajectories"
LIVE_STORE = production_data_root() / "sessions"
LIVE_VERDICT_LEDGER = (production_data_root() / "_pipeline" / "skills"
                       / "reviews" / "verdicts.jsonl")
MACHINE_PLATFORMS = {"worker", "autonomy", "e2e-harness"}
#: The smallest session store the classifier oracle below can discriminate on.
#: #1143 wrote it as `assert len(files) > 500`; as a floor that is 501, and this is
#: the number a skip reason quotes. Measured 3,000 files on 2026-09-18, 58 after the
#: 2026-09-22 wipe.
SESSION_STORE_MIN_FILES = 501
#: The fewest mined error keys with MORE THAN ONE signature behind them that the
#: merged-total assertions of `test_a_live_week_emits_one_merged_candidate_per_error_key`
#: can discriminate on. It is that node's own `>= 2` — not truthy, because one
#: multi-bucket key would let a merge that happened to fold a single pair pass —
#: lifted out of the assertion so the skip and the assertion quote one number
#: instead of two, the same reason `SESSION_STORE_MIN_FILES` exists above.
#: Measured 15 of 30 keys on the 7-day corpus on 2026-09-21; 1 of 12 keys on the
#: 2-file corpus this machine has held since the 2026-09-22 wipe, which is the
#: reading that made this a red node at base rather than a finding (#1403).
MERGE_MULTI_SIGNATURE_KEYS_MIN = 2
#: The fewest sessions the classifier ADMITS as human work that the store must hold
#: before the observer control in
#: `test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session` can be
#: judged at all, and the number that skip reason quotes. It is a denominator rule, not
#: a second volume claim: the control below is a share, and at N admitted sessions one
#: conversation moves it by 1/N, so below 5 the verdict belongs to whether the observer
#: happened to be switched on for ONE chat rather than to the classifier. It is NOT the
#: same floor as `SESSION_STORE_MIN_FILES` above — that counts files in the store, this
#: counts the files admitted — and `require_live_volume` states the skip so the floor
#: and the observed human count appear in the one reason, the shape #1377 required.
#: Measured 163 admitted on 2026-09-18 and 7 on 2026-09-24, both above it (#1448).
ADMITTED_HUMAN_SESSIONS_MIN = 5
#: The fewest of the admitted human sessions that must carry `inner_voice: true`, as a
#: SHARE of the admitted set: #1143's positive control, rewritten scale-invariantly by
#: #1448. Its bound used to be the absolute `> 20`, which was a measurement of one
#: machine's store on one day — it started failing on 2026-09-24, the day the store
#: regrew past `SESSION_STORE_MIN_FILES` after the 2026-09-22 wipe, with the classifier
#: agreeing with the ruled rule on every file. What the control is for is directional
#: and stays exactly as loud: pre-#1143 the classifier admitted only observer-OFF chats,
#: so the share was 0 of 45, and 0 is under this bound whatever the store's size. Half,
#: not more, because a person typing a chat with the observer off is legitimate and must
#: not read as the corpus being dropped. Measured 152 of 163 on 2026-09-18, 7 of 7 on
#: 2026-09-24.
OBSERVER_ON_SHARE_MIN = 0.5


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
    # Absent roots skip and the reason names them both (#1377), which is what let a
    # wiped `_pipeline` red-block every promotion for a day. Present roots always run
    # the join, and `rows`/`machine_rows` below are asserted non-zero, so a corpus that
    # exists but joins to nothing still fails — and a row labelled interactive over a
    # machine platform still fails:
    # `test_a_corpus_row_labelled_interactive_on_a_machine_platform_fails`.
    require_live_data(LIVE_CORPUS, "live trajectory corpus")
    require_live_data(LIVE_STORE, "session store")
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


def test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session():
    """Read-only oracle over the real session store: for every session JSON on this
    machine, the classifier's answer must equal the rule #1143 states directly —
    a platform `is_user_session` admits, mapped by `SESSION_CLASS`, and demoted to
    `test` when the id is not the three-part chat shape. `inner_voice` is not in
    this expression at all, which is the whole point of #1143.

    This is the check that survives the corpus being re-extracted: it is computed
    from the stored session files themselves, not from anything the extractor
    already wrote, so a classifier mutation trips it even while every corpus row is
    class-less.

    Every bound below is scale-invariant, and that is #1448. The store is gitignored
    data whose size on this machine is an accident of the day it was last written and
    the day it was last wiped, so a bound written as a COUNT — of files, or of
    observer-on chats — was never a property of the classifier. It was a description of
    one snapshot, and it turned red on 2026-09-24 the day the store regrew past
    `SESSION_STORE_MIN_FILES`, with the classifier agreeing with the ruled rule on every
    file in it. A systematic drop is 0 of any set, which is why the observer control
    survives as a share of the admitted set behind `ADMITTED_HUMAN_SESSIONS_MIN`; each
    number lives on its own constant above and is never restated here as an inline
    comparison, which is what
    `test_the_observer_control_states_its_share_and_floors_once` checks. Snapshot of
    this machine on 2026-09-24, a snapshot and not a threshold — the store grows daily:
    514 files, 7 admitted (5 `mission-control`, 2 `browser`), all 7 with `inner_voice`
    true, 0 disagreements.
    """
    from app import sessions_io
    # Absent store -> named skip; a store that exists but is too thin in FILES to
    # discriminate -> a named skip carrying BOTH the floor and the observed count
    # (#1377 clause 2); a store that is full in files but too thin in ADMITTED chats for
    # a share to mean anything -> the same reason shape, naming the admitted floor and
    # the observed human count (#1448 clause 2). None of the three can mask a disagreeing
    # classifier: the loop below runs over every file, the disagreement assert is the
    # first thing that fires, and the admitted-volume skip sits after it — pinned by
    # `test_a_full_store_with_a_disagreeing_classifier_fails_rather_than_skips`, which
    # puts the pre-#1143 rule back over a store above the floor and must still FAIL.
    require_live_data(LIVE_STORE, "session store")
    files = sorted(LIVE_STORE.glob("*.json"))
    require_live_volume(files, SESSION_STORE_MIN_FILES, LIVE_STORE, "session store")
    violations: list[str] = []
    admitted: list[str] = []
    human_with_observer_on = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        platform = data.get("platform")
        expected = et.SESSION_CLASS.get(platform) or et.UNKNOWN_CLASS
        if not sessions_io.is_user_session(data):
            expected = (et.UNKNOWN_CLASS if expected in et.HUMAN_CLASSES
                        else expected)
        elif expected in et.HUMAN_CLASSES:
            sid = str(data.get("session_id") or path.stem)
            if sid and not re.match(r"^\d{8}_\d{6}_[A-Za-z0-9]+$", sid):
                expected = "test"
        # The id is passed the way `parse_session` passes it — the file's own field,
        # the stem only when the field is absent — so this oracle calls the
        # classifier with exactly the extractor's inputs.
        got = et.classify_session(data, session_id=data.get("session_id", path.stem))
        if got != expected:
            violations.append(f"{path.name}: platform={platform!r} "
                              f"inner_voice={data.get('inner_voice')!r} "
                              f"-> {got}, expected {expected}")
        if got in et.HUMAN_CLASSES:
            admitted.append(path.name)
            if data.get("inner_voice"):
                human_with_observer_on += 1
    human = len(admitted)
    assert not violations, (
        f"classifier disagrees with the ruled rule: {violations[:5]}")
    assert 0 < human < len(files), (
        f"human={human} of {len(files)}: a store with none or all human means the "
        "oracle above cannot discriminate either direction")
    # The positive control #493 could not have: the chats typed with the observer
    # switched on have to be IN the admitted set, or a zero there is indistinguishable
    # from the misclassification #1143 exists to close. It is a SHARE of the admitted
    # set, because a systematic drop is 0% of a set of any size while an absolute count
    # is only true of the store that was measured (#1448). Below
    # `ADMITTED_HUMAN_SESSIONS_MIN` the denominator cannot carry a share, and the skip
    # names the floor and the observed human count — which is the state a wiped store
    # reaches, and the state this node used to answer with a red run.
    require_live_volume(admitted, ADMITTED_HUMAN_SESSIONS_MIN, LIVE_STORE,
                        "admitted human sessions", noun="sessions")
    assert human_with_observer_on / human >= OBSERVER_ON_SHARE_MIN, (
        f"only {human_with_observer_on} of {human} admitted sessions carry "
        f"`inner_voice: true`, under the {OBSERVER_ON_SHARE_MIN:.0%} this control "
        "allows: chats typed with the observer switched on are dropping out of the "
        "admitted set again")


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


# ── clause 5 (#1143): the architecture page must name who actually typed ─────

ARCH_PAGE = _ROOT / "architecture" / "background-runs.md"


def test_the_architecture_page_attributes_browser_sessions_to_the_extension():
    """`architecture/background-runs.md` said the `platform: browser` sessions were
    "written by ... the Inner Voice bench harness". They are not: the Chrome
    extension's side panel creates them and a person types follow-ups into them,
    which is half of why #1143's corpus was empty of human chats.

    Asserted as text over the whole page, because the sentence is the artifact: the
    count of those sessions moves with every extension turn, so what must not come
    back is the attribution, and the name of the thing that really mints them.
    """
    assert ARCH_PAGE.is_file(), f"missing {ARCH_PAGE}"
    text = " ".join(ARCH_PAGE.read_text(encoding="utf-8").split())
    assert "chrome-extension/src/background/lloyd-client.ts" in text, text[:400]
    assert "Inner Voice bench harness" not in text
    assert "inner voice bench harness" not in text.lower()
    block = [b for b in text.split("- **") if "platform: browser" in b]
    assert block and "extension" in block[0].lower(), block


# ── #1238: a signal is what the run emitted, not what its skill said ──────────
#
# `SIGNAL_RE` used to run over every message's content with no role filter. In an
# autonomy run the first `user` message IS the dispatched SKILL.md body, so a skill
# documenting `SIGNAL:BLOCKED` in prose handed that token to every one of its runs
# whatever happened: measured on 2026-09-17, all 9 `BLOCKED` rows in the corpus were
# inherited (6 from `autonomy-task:24`'s dispatched body, 3 from tool results quoting
# the skill file the autotriage run was reading) and no session emitted one — while
# `TASK_COMPLETE`, the same field on the success axis, appeared once in a day holding
# 249 runs. `collect_signals` now splits hits by role: `signals` holds only what an
# `EMITTED_SIGNAL_ROLES` message said, `inherited_signals` keeps the rest with the
# role that carried them.

# The instruction half of an autonomy run: `skills/autonomy-data-pipeline/SKILL.md`
# documents this token three times (its lines 154, 396, 397) and the run is handed
# the file as its first `user` message.
DISPATCHED_BODY = (
    "# SKILL: autonomy-data-pipeline\n"
    "- HARD RULE - NEVER modify `~/lloyd/.venvs/`. If a dep is missing, "
    "SIGNAL:BLOCKED and exit.\n"
    "- If the script fails, log the error and SIGNAL:BLOCKED - missing dep.\n"
    "SIGNAL:BLOCKED - missing dep\n"
)


def call_pair(result_text, i=0):
    """One assistant tool_call plus its tool result.

    `parse_session` returns None for a session with no tool calls, so every signal
    fixture needs at least one pair before its prose; it is also the source of the
    `tool`-role text the rule must exclude.
    """
    return [
        {"role": "assistant",
         "tool_calls": [{"id": f"call_{i}",
                         "function": {"name": "Bash",
                                      "arguments": json.dumps(
                                          {"command": "pytest tests/"})}}]},
        {"role": "tool", "tool_call_id": f"call_{i}",
         "content": [{"type": "text", "text": result_text}]},
    ]


def signal_row(tmp_path, messages, name="sess-signals"):
    """Parse a hand-authored session into a trajectory row."""
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "session_id": name,
        "session_start": "2026-09-17T03:34:17Z",
        "messages": messages,
    }), encoding="utf-8")
    traj = et.parse_session(path)
    assert traj is not None, f"fixture {name} produced no row (needs a tool call)"
    return traj


def inherited_only_session(tmp_path, name):
    """The 2026-09-17 shape: `SIGNAL:BLOCKED` twice in the dispatched body and once
    in a tool result, and the run's own last message reporting a clean finish."""
    return signal_row(tmp_path, [
        {"role": "user", "content": DISPATCHED_BODY},
        *call_pair("skills/autonomy-data-pipeline/SKILL.md:154: If the script fails, "
                   "log the error and SIGNAL:BLOCKED - missing dep"),
        {"role": "assistant",
         "content": "All four steps ran, failed=0. Loaded-memory shrink guard passed."},
    ], name=name)


def stats_signal_counts(out):
    """The `Signals seen:` block of `--stats` output as a dict.

    A token with zero genuine emissions prints no row, which is how "0 BLOCKED"
    appears: absence, not a zero.
    """
    assert "Signals seen:\n" in out, out
    block = out.split("Signals seen:\n", 1)[1].split("\n\n")[0]
    return {name: int(n)
            for name, n in re.findall(r"^  (\S+) +(\d+)$", block, re.MULTILINE)}


# clause 1: an assistant-role token is collected
def test_a_signal_in_assistant_text_is_collected(tmp_path):
    traj = signal_row(tmp_path, [
        {"role": "user", "content": DISPATCHED_BODY},
        *call_pair("42 passed"),
        {"role": "assistant",
         "content": [{"type": "text", "text": "Done. SIGNAL:TASK_COMPLETE"}]},
    ])
    assert traj["signals"] == ["TASK_COMPLETE"]


# clause 2: injected body + tool results alone yield nothing ...
def test_a_token_only_in_the_skill_body_and_tool_results_is_not_collected(tmp_path):
    traj = inherited_only_session(tmp_path, "sess-inherited")
    assert traj["signals"] == [], (
        "an inherited skill-body token must not enter `signals` on any axis")


# ... and the same fixture with the token moved into assistant text yields it
def test_the_same_session_with_the_token_in_assistant_text_yields_it(tmp_path):
    traj = signal_row(tmp_path, [
        {"role": "user", "content": "Run the autonomy data pipeline."},
        *call_pair("42 passed"),
        {"role": "assistant", "content": "Step 2 died on ImportError. SIGNAL:BLOCKED"},
    ])
    assert traj["signals"] == ["BLOCKED"]


# clause 3: excluded hits stay on the row, naming the role that carried them
def test_excluded_hits_stay_on_the_row_naming_their_role(tmp_path):
    """Three `BLOCKED` hits, none of them the run's: the point is that the row says
    so instead of reading as a session with no signal at all."""
    traj = inherited_only_session(tmp_path, "sess-inherited-roles")
    assert traj["signals"] == []
    assert traj["inherited_signals"] == {"user": ["BLOCKED"], "tool": ["BLOCKED"]}


def test_an_emitted_token_is_not_also_reported_as_inherited(tmp_path):
    """The body said `BLOCKED` and the run said it too: `inherited` is for what the
    rule dropped, and this one was not dropped."""
    traj = signal_row(tmp_path, [
        {"role": "user", "content": DISPATCHED_BODY},
        *call_pair("42 passed"),
        {"role": "assistant", "content": "Dependency missing. SIGNAL:BLOCKED"},
    ], name="sess-both")
    assert traj["signals"] == ["BLOCKED"]
    assert traj["inherited_signals"] == {}


def test_a_token_in_an_assistant_tool_call_argument_is_not_an_emission(tmp_path):
    """The dropped real channel, kept recoverable. A dispatched subprocess that
    prints `SIGNAL:TASK_COMPLETE` to stdout (`scripts/memory/process-groundskeeper-queue.py`)
    reaches the row as tool-role text, so an assistant-only rule loses it - and the
    `echo` command that prints it is assistant text whose *argument* carries the
    token, which is not an emission either. Both stay inspectable."""
    traj = signal_row(tmp_path, [
        {"role": "user", "content": "Drain the groundskeeper queue."},
        {"role": "assistant", "tool_calls": [{"id": "call_0", "function": {
            "name": "Bash",
            "arguments": json.dumps({"command": 'echo "SIGNAL:TASK_COMPLETE"'})}}]},
        {"role": "tool", "tool_call_id": "call_0",
         "content": [{"type": "text", "text": "SIGNAL:TASK_COMPLETE\n"}]},
    ], name="sess-echoed")
    assert traj["signals"] == []
    assert traj["inherited_signals"] == {"tool": ["TASK_COMPLETE"]}


def test_the_checkpoint_form_is_collected_as_its_outer_token(tmp_path):
    """`SIGNAL_RE`'s capture stops at the next colon, so the checkpoint form
    documented in `skills/.archived/python-library-dev/SKILL.md` is collected as
    `CHECKPOINT` and the specific checkpoint is lost. #1238 changed which roles are
    read, not what one hit captures, so this pins the quirk rather than silently
    moving it."""
    traj = signal_row(tmp_path, [
        {"role": "user", "content": "Build the library."},
        *call_pair("ok"),
        {"role": "assistant",
         "content": "SIGNAL:CHECKPOINT:PLAN_COMPLETE and SIGNAL:TASK_COMPLETE"},
    ], name="sess-checkpoint")
    assert traj["signals"] == ["CHECKPOINT", "TASK_COMPLETE"]


# clause 4: the extractor's own histogram reports emissions only
def test_stats_signal_histogram_counts_only_run_emitted_tokens(tmp_path, capsys):
    et.append_trajectories([
        inherited_only_session(tmp_path, "sess-hist-inherited"),
        signal_row(tmp_path, [
            {"role": "user", "content": DISPATCHED_BODY},
            *call_pair("42 passed"),
            {"role": "assistant", "content": "Finished. SIGNAL:TASK_COMPLETE"},
        ], name="sess-hist-emitted"),
    ])
    et.print_stats()
    counts = stats_signal_counts(capsys.readouterr().out)
    assert counts.get("TASK_COMPLETE") == 1, counts
    assert "BLOCKED" not in counts, (
        "the bucket holds one session that inherited BLOCKED and none that emitted "
        f"it, so the histogram must not report it: {counts}")


def test_the_new_field_survives_the_bucket_write(tmp_path):
    """`inherited_signals` is a dict of lists, and the row is written and read back
    as one JSON line by both `--stats` and any skill reading the corpus."""
    et.append_trajectories([inherited_only_session(tmp_path, "sess-roundtrip")])
    bucket = next(et.OUTPUT_DIR.glob("*.jsonl"))
    row = json.loads(bucket.read_text(encoding="utf-8").splitlines()[0])
    assert row["signals"] == []
    assert row["inherited_signals"] == {"user": ["BLOCKED"], "tool": ["BLOCKED"]}


# ── #1238 seams: the row is written here and read over there ──────────────────
#
# Two boundaries the fix crosses as a process, not as a call. The `--stats`
# histogram is read by whoever runs the command (`nightly-skills-management` among
# them), and the bucket file is the miner's input, so the schema gaining a field is
# an output-format change for both.

EXTRACTOR_PATH = _ROOT / "scripts" / "extract-trajectories.py"
SIGNAL_BUCKET = "2026-09-17.jsonl"


def test_the_stats_command_prints_only_run_emitted_tokens(tmp_path, monkeypatch):
    """The real command line, in a real process, over a bucket the real parser wrote.

    Both halves are the code under test: the rows come from `parse_session` over two
    hand-authored sessions (one that inherited `BLOCKED` from its dispatched skill
    body, one that emitted `TASK_COMPLETE` in assistant text), and `--stats` runs as
    a subprocess. `OUTPUT_DIR` derives from the data root, so `LLOYD_DATA` is pointed
    at a directory holding what the extractor wrote. Expected histogram: 1
    `TASK_COMPLETE`, and no `BLOCKED` row at all — which is how "0 BLOCKED" prints.
    """
    home = tmp_path / "home"
    bucket_dir = home / "lloyd-data" / "_pipeline" / "trajectories"
    bucket_dir.mkdir(parents=True)
    monkeypatch.setattr(et, "OUTPUT_DIR", bucket_dir)
    et.append_trajectories([
        inherited_only_session(tmp_path, "sess-cli-inherited"),
        signal_row(tmp_path, [
            {"role": "user", "content": DISPATCHED_BODY},
            *call_pair("42 passed", i=1),
            {"role": "assistant", "content": "Finished. SIGNAL:TASK_COMPLETE"},
        ], name="sess-cli-emitted"),
    ])
    assert list(bucket_dir.glob("*.jsonl")), "extractor wrote no bucket"

    proc = subprocess.run(
        [sys.executable, str(EXTRACTOR_PATH), "--stats"],
        capture_output=True, text=True, timeout=180, cwd=tmp_path,
        env={**os.environ, "HOME": str(home), "LLOYD_DATA": str(home / "lloyd-data")})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    counts = stats_signal_counts(proc.stdout)
    assert counts.get("TASK_COMPLETE") == 1, counts
    assert "BLOCKED" not in counts, counts


def test_the_miner_still_mines_rows_that_carry_the_new_field(tmp_path):
    """`mine-trajectories.py` reads the same JSONL rows and ignores both signal
    fields (its only mention of either is a prose comment), so the schema addition
    must pass through it: the same two interactive failures, now carrying
    `inherited_signals`, still mine one candidate reporting 2 sessions."""
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = []
    for key in ("i1", "i2"):
        row = miner_row(key, "interactive")
        row["signals"] = []
        row["inherited_signals"] = {"user": ["BLOCKED"], "tool": ["BLOCKED"]}
        rows.append(row)
    (corpus / SIGNAL_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    proc = run_miner(corpus, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fm = graphex_candidate(out)
    assert re.search(r"^sessions: 2$", fm, re.MULTILINE), fm


# ── INDEX.md indexes the corpus, not the run that wrote it (backlog #720) ────
#
# `write_index(candidate_files, output_dir)` was fed `[os.path.basename(p) for p in
# written]` — only the files the *current* run wrote — so the one document Phase 1.1
# of the consolidation runbook reads first reported that batch as the corpus.
# Measured on the live tree the hour this round opened: `INDEX.md` carried
# `**Total candidates:** 3` over 3 rows while the same directory held **3,936**
# `candidate-*.md` (`ls … | wc -l`). Task #58 had logged the same mismatch on
# 2026-09-17 with the batch empty: `Total candidates: 0` against 3,919 files. So the
# defect is the row-set source, not the miner's output volume, and it reproduces
# today precisely because a nightly run writes almost nothing.
#
# Three further defects lived in the same function:
#   * the Summary counted a filename containing the substring `error`, and **0** of
#     the live corpus's filenames contain it (`ls candidate-*.md | grep -c error` → 0),
#     so `Error patterns` was structurally 0 and every error candidate was charged to
#     `Success patterns`: run over the 3,936-file store the old lines report
#     `Error patterns: 0 / Success patterns: 466`, where the counts read from the files'
#     own `type:` keys are 126 error and 340 success, and both sum to the same 466
#     (#516, closed by an expiry sweep on 2026-09-15 and never fixed);
#   * the input list was emitted row-per-entry with no dedupe, so one file named
#     twice produced two rows (the emitter de-dupes upstream since #1131 clause 5,
#     but `scripts/rebuild-skill-candidates-index.py` builds its own list);
#   * a file whose front matter has no `status:` line rendered `Status: ?`.
#
# The denominator everywhere below is `candidate-*.md`, never `*.md`: the latter
# counts `INDEX.md` itself and would be off by one forever.

CANDIDATE_GLOB = "candidate-*.md"


def seed_candidate(out_dir, name, *, type_, pattern="Bash/timeout", sessions=3,
                   status=None):
    """Write one candidate file the way `write_candidate_file` writes one: YAML
    front matter, then a body. `status=None` omits the `status:` key entirely,
    which is the state clause 4 is about."""
    lines = ["---", "candidate: true", f"pattern: {pattern}",
             f"type: {type_}", f"sessions: {sessions}"]
    if status is not None:
        lines.append(f"status: {status}")
    lines += ["---", "", "# Skill Candidate: seeded for #720"]
    (Path(out_dir) / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def index_field(index_text, label):
    m = re.search(rf"\*\*{re.escape(label)}:\*\* (\d+)", index_text)
    assert m, f"{label!r} missing from the index Summary:\n{index_text[:400]}"
    return int(m.group(1))


def index_rows(index_text, name=None):
    """The candidate rows, optionally narrowed to one filename."""
    pat = r"^- \[candidate-" if name is None else r"^- \[" + re.escape(name) + r"\]"
    return re.findall(pat, index_text, re.MULTILINE)


def index_statuses(index_text):
    """basename -> the Status column of its row, for every candidate row."""
    return {m.group(1): m.group(2) for m in re.finditer(
        r"^- \[([^\]]+)\]\([^\)]*\) — Pattern: `[^`]*` — Sessions: \S+ — Status: (.+)$",
        index_text, re.MULTILINE)}


def test_the_index_lists_every_candidate_on_disk_when_this_run_wrote_none(tmp_path):
    """Clause 1, and the exact reproduction triage left behind: three
    `candidate-*.md` files already in the output dir, an empty batch handed to the
    emitter. Pre-fix this printed `Total candidates: 0` over 0 rows with 3 files on
    disk — which is how the live index came to report 3 against 3,936."""
    out = tmp_path / "cands"
    out.mkdir()
    seed_candidate(out, "candidate-bash-timeout-20260101.md", type_="error")
    seed_candidate(out, "candidate-read-logic-20260102.md", type_="error",
                   status="reviewed_no_skill")
    seed_candidate(out, "candidate-seq-3-a-b-c-20260103.md", type_="sequence",
                   pattern="seq-3-a-b-c")

    mt.write_index([], out)

    index = (out / "INDEX.md").read_text(encoding="utf-8")
    on_disk = len(list(out.glob(CANDIDATE_GLOB)))
    assert on_disk == 3, "the fixture seeded the wrong number of files"
    assert index_field(index, "Total candidates") == on_disk, index
    assert len(index_rows(index)) == on_disk, index


def test_the_index_type_counts_come_from_each_indexed_files_own_type_field(tmp_path):
    """Clause 2. The old Summary grepped the *filename* for the substring `error`,
    and no candidate filename contains it (0 of the live corpus's 3,936), so every
    error candidate was charged to `Success patterns` and `Error patterns` was
    structurally 0. The counts now come from each indexed file's own `type:` front
    matter — the field `write_candidate_file` already writes — and sum to Total."""
    errs = tmp_path / "err-only"
    errs.mkdir()
    for day in ("20260101", "20260102", "20260103"):
        seed_candidate(errs, f"candidate-bash-timeout-{day}.md", type_="error")
    mt.write_index([], errs)
    index = (errs / "INDEX.md").read_text(encoding="utf-8")
    assert index_field(index, "Error patterns") == 3, index
    assert index_field(index, "Success patterns") == 0, index
    assert index_field(index, "Sequence patterns") == 0, index
    assert (index_field(index, "Error patterns")
            + index_field(index, "Success patterns")
            + index_field(index, "Sequence patterns")) == index_field(
                index, "Total candidates"), index

    # All three classes in one dir, the shape the live corpus is actually in
    # (126 error / 340 success / 3470 sequence measured 2026-09-22).
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    seed_candidate(mixed, "candidate-bash-timeout-20260101.md", type_="error")
    seed_candidate(mixed, "candidate-read-usage-20260101.md", type_="success",
                   pattern="Read/usage")
    seed_candidate(mixed, "candidate-write-usage-20260102.md", type_="success",
                   pattern="Write/usage")
    for day in ("20260101", "20260102", "20260103", "20260104"):
        seed_candidate(mixed, f"candidate-seq-3-a-b-c-{day}.md", type_="sequence",
                       pattern="seq-3-a-b-c")
    mt.write_index([], mixed)
    index = (mixed / "INDEX.md").read_text(encoding="utf-8")
    assert (index_field(index, "Error patterns"),
            index_field(index, "Success patterns"),
            index_field(index, "Sequence patterns")) == (1, 2, 4), index
    assert index_field(index, "Total candidates") == 7, index

    # The falsifier that the counts read the field and not the name: a file named
    # like a sequence whose own `type:` says error, and the converse. A filename
    # heuristic is wrong in both directions here; the front matter is right in both.
    odd = tmp_path / "odd"
    odd.mkdir()
    seed_candidate(odd, "candidate-seq-3-a-b-c-20260101.md", type_="error")
    seed_candidate(odd, "candidate-read-usage-20260102.md", type_="sequence",
                   pattern="seq-2-read-edit")
    mt.write_index([], odd)
    index = (odd / "INDEX.md").read_text(encoding="utf-8")
    assert (index_field(index, "Error patterns"),
            index_field(index, "Success patterns"),
            index_field(index, "Sequence patterns")) == (1, 0, 1), index


def test_one_candidate_name_named_twice_by_the_input_still_yields_one_row(tmp_path):
    """Clause 3. `emit_candidates` de-dupes upstream since #1131 clause 5, so
    `main()` cannot hand a repeated path today, but the emitter's own loop had no
    guard and `write_index` has a second caller-shaped writer —
    `scripts/rebuild-skill-candidates-index.py` builds its own list from a glob.
    Pre-fix, one name passed twice emitted 2 rows for 1 file."""
    out = tmp_path / "cands"
    out.mkdir()
    name = "candidate-bash-timeout-20260101.md"
    seed_candidate(out, name, type_="error")

    mt.write_index([name, name], out)

    index = (out / "INDEX.md").read_text(encoding="utf-8")
    assert len(index_rows(index, name)) == 1, index
    assert index_field(index, "Total candidates") == 1, index
    assert len(index_rows(index)) == len(list(out.glob(CANDIDATE_GLOB))), index


def test_a_candidate_with_no_status_line_is_listed_not_yet_dispositioned(tmp_path):
    """Clause 4. A candidate whose front matter carries no `status:` line is one
    nobody has dispositioned, and `Status: ?` read as a parsing artifact rather
    than a verdict. A file that does carry one still reports its own value — this
    is about the missing key, never about overriding a decision."""
    out = tmp_path / "cands"
    out.mkdir()
    seed_candidate(out, "candidate-read-logic-20260101.md", type_="error")
    seed_candidate(out, "candidate-bash-timeout-20260102.md", type_="error",
                   status="reviewed_no_skill")

    mt.write_index([], out)

    index = (out / "INDEX.md").read_text(encoding="utf-8")
    listed = index_statuses(index)
    assert listed == {"candidate-read-logic-20260101.md": "not-yet-dispositioned",
                      "candidate-bash-timeout-20260102.md": "reviewed_no_skill"}, index
    assert "Status: ?" not in index, index
    assert "pending_review" not in index, index


def test_a_live_mining_run_indexes_the_candidates_earlier_runs_left_behind(tmp_path):
    """#720 across the process boundary the nightly actually crosses: the real
    command line, its own candidate files, the index it regenerates. Two files from
    an earlier run sit in the output dir untouched — one already dispositioned
    (`superseded_by_verdict`), one never dispositioned. Pre-fix the regenerated
    index listed only this run's batch, so the store's three files appeared as one
    and the older evidence vanished from the only document Phase 1.1 reads."""
    corpus = write_traj_corpus(tmp_path / "corpus", [])
    rows = [miner_row("i1", "interactive"), miner_row("i2", "interactive")]
    (corpus / CORPUS_BUCKET).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cands"
    out.mkdir()
    seeded = {"candidate-olderror-20260101.md": "error",
              "candidate-oldseq-20260101.md": "sequence"}
    seed_candidate(out, "candidate-olderror-20260101.md", type_="error",
                   pattern="Old/error", status="superseded_by_verdict")
    seed_candidate(out, "candidate-oldseq-20260101.md", type_="sequence",
                   pattern="seq-2-old-error")

    proc = run_miner(corpus, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    index = (out / "INDEX.md").read_text(encoding="utf-8")
    on_disk = sorted(p.name for p in out.glob(CANDIDATE_GLOB))
    new = [n for n in on_disk if n not in seeded]
    assert len(new) == 1, f"expected the corpus to mine exactly one candidate, got {new}"
    assert index_field(index, "Total candidates") == len(on_disk), index
    assert len(index_rows(index)) == len(on_disk), index
    # Both older files are still indexed, with their own status.
    assert len(index_rows(index, "candidate-olderror-20260101.md")) == 1, index
    assert len(index_rows(index, "candidate-oldseq-20260101.md")) == 1, index
    listed = index_statuses(index)
    assert listed["candidate-olderror-20260101.md"] == "superseded_by_verdict", index
    assert listed["candidate-oldseq-20260101.md"] == "not-yet-dispositioned", index
    # The two seeded files plus the one this run wrote (`Graphex493/not_found`, an
    # error pattern), attributed by their own `type:` fields across the boundary.
    assert "type: error" in graphex_candidate(out), graphex_candidate(out)[:200]
    assert (index_field(index, "Error patterns"),
            index_field(index, "Success patterns"),
            index_field(index, "Sequence patterns")) == (2, 0, 1), index


def test_the_consolidation_runbook_says_which_index_columns_are_advisory():
    """Clause 5. §1.1 of `skills/nightly-skill-consolidation/SKILL.md` is a bare
    `cat …/INDEX.md`, and the Status column it presents is a pre-disposition
    snapshot by design: §5.1 writes dispositions *after* the mining run regenerated
    the index, so that column is a lower bound however correct the generator is.
    The clause asks for the `cat` to stay with a caveat naming the advisory column,
    so the assertion is that §1.1 names Status and says it is advisory — and that
    the `cat` is still there, because replacing it was the other allowed answer and
    a test that passed on both would not pin the caveat.

    No skip when the file is missing: this clause's subject is a tree outside the
    gated repo, and a guard that could go quietly unverified would repeat the defect
    it is guarding (`tests/test_automod_doc_claims.py` reads the same tree
    unguarded for the same reason)."""
    from app.paths import VAULT_ROOT

    skill = VAULT_ROOT / "skills" / "nightly-skill-consolidation" / "SKILL.md"
    assert skill.is_file(), f"clause 5's subject is absent: {skill} does not exist"
    text = skill.read_text(encoding="utf-8")
    section = text[text.index("### 1.1"):text.index("### 1.2")]
    assert "INDEX.md" in section, section
    assert "Status" in section and "advisory" in section.lower(), (
        "§1.1 still presents the index as authoritative with no caveat naming the "
        f"advisory column:\n{section}")


# ── #1377: the absence rule, pinned from both directions ────────────────────
#
# Seven guards in this file, plus one in `test_skill_verdicts.py`, read data the
# checkout does not hold: `.gitignore:22` ignores `/sessions/` and `.gitignore:25`
# ignores `/_pipeline/`, so neither path exists in a round's worktree — and after the
# 2026-09-22 fixture-teardown wipe neither exists on the live tree either. Each of them
# hard-asserted that data's presence, so each reproduced as a failure *at base* in every
# round: `base_probe: probed 21 file(s) at base 1842b8cf: 106 already failing` with
# `external_blocker: true`, copied from
# `~/.local/state/lloyd-automod/rounds/SM_20260922_201206/gate.json`. That is how eight
# nodes about a missing directory came to block every promotion, #1055 included.
#
# The rule itself is `tests/_live_data.py`: absence of a live-data root becomes a named
# skip, everything else stays a failure. The tests below pin BOTH edges. That matters
# more than the red-run it fixes: the review's own finding on round SM_20260914_140724
# was that a corpus guard skipping over present-but-empty data reported green while
# executing nothing, so a skip that could fire over present data would trade eight loud
# failures for eight silent nothings.

#: Each live root: the name this module holds it under, and the miner's own mirror of
#: the same path. `load_trajectories` reads `mt.TRAJECTORY_DIR` and
#: `effective_session_class` reads `mt.SESSION_STORE_DIR`, so redirecting a root has to
#: move the code beneath the guard as well as the guard's own statement of it.
LIVE_ROOTS = {
    "corpus": ("LIVE_CORPUS", "TRAJECTORY_DIR"),
    "store": ("LIVE_STORE", "SESSION_STORE_DIR"),
    "ledger": ("LIVE_VERDICT_LEDGER", None),
}


def redirect_live_root(monkeypatch, root: str, path) -> Path:
    """Point one live root, and the miner's copy of it, at `path`."""
    here, mirror = LIVE_ROOTS[root]
    target = Path(path)
    monkeypatch.setattr(_THIS, here, target)
    if mirror is not None:
        monkeypatch.setattr(mt, mirror, target)
    return target


def guard_arguments(fn, tmp_path) -> dict:
    """How to invoke a guard whose body is being run directly. Every live-data guard
    here declares at most one fixture, `tmp_path`, and the directories it writes are
    created by the code under test, not by pytest."""
    return ({"tmp_path": tmp_path / "out"}
            if "tmp_path" in inspect.signature(fn).parameters else {})


def utc_bucket() -> str:
    """Today's date: `load_trajectories` keeps a `YYYY-MM-DD.jsonl` file whose date is
    not before `utc now - days`, so a bucket named this is inside every window below."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def synthetic_corpus(tmp_path, rows, name=None) -> Path:
    """A `_pipeline/trajectories` stand-in in a tmp root, holding `rows`."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / f"{name or utc_bucket()}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return root


def synthetic_store(tmp_path, count: int, **fields) -> Path:
    """A `~/lloyd/sessions` stand-in of `count` chat-shaped session files.

    Files are named `<session_id>.json` because that is how the real store names them
    and how both readers reach a session: `effective_session_class` opens
    `SESSION_STORE_DIR / f"{session_key}.json"` and the #493 join does the same."""
    root = tmp_path / "sessions"
    root.mkdir()
    for i in range(count):
        data = {"platform": "mission-control", "inner_voice": True}
        data.update(fields)
        data["session_id"] = data.get("session_id") or f"20260920_120000_chat{i}"
        (root / f"{data['session_id']}.json").write_text(
            json.dumps(data), encoding="utf-8")
    return root


def synthetic_mixed_store(tmp_path, name: str, machine: int, human: int,
                          inner_voice: bool) -> Path:
    """A session store of `machine` `worker`-platform files plus `human` chat-shaped
    mission-control chats typed with the observer in state `inner_voice`.

    `synthetic_store` above writes an all-human store, which cannot show a guard
    anything about a SHARE of the admitted set: `assert 0 < human < len(files)` needs
    both classes present, and #1448's observer control is judged over the admitted slice
    alone. The machine files are `worker`, the one machine platform both the backend's
    deny-list and `SESSION_CLASS` name, so the oracle's expected value agrees with the
    classifier for every one of them and the store isolates the observer control."""
    root = tmp_path / name
    root.mkdir()
    for i in range(machine):
        data = {"platform": "worker", "inner_voice": False,
                "session_id": f"20260920_120000_worker{i}"}
        (root / f"{data['session_id']}.json").write_text(
            json.dumps(data), encoding="utf-8")
    for i in range(human):
        data = {"platform": "mission-control", "inner_voice": inner_voice,
                "session_id": f"20260920_120000_human{i}"}
        (root / f"{data['session_id']}.json").write_text(
            json.dumps(data), encoding="utf-8")
    return root


def synthetic_ledger(tmp_path, rows) -> Path:
    """A `_pipeline/skills/reviews/verdicts.jsonl` stand-in."""
    root = tmp_path / "reviews"
    root.mkdir()
    path = root / "verdicts.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


#: A present stand-in for every root, so an absence row can leave exactly ONE root
#: missing. Each builder writes to its own directory name under `tmp_path`, so one call
#: per root per test never collides. The synthetic keys are names no real ledger holds.
_PRESENT_ROOTS = {
    "corpus": lambda tp: synthetic_corpus(tp, [
        error_traj(f"present1377_{i}", "Present1377", "not_found", "protocol")
        for i in (1, 2)]),
    "store": lambda tp: synthetic_store(tp, 1),
    "ledger": lambda tp: synthetic_ledger(tp, [
        {"pattern_key": "Present1377/not_found", "verdict": "rejected",
         "decided_at": "2026-09-22T00:00:00+00:00"}]),
}


#: One row per (guard, root) edge that used to be a bare `assert` — the eight nodes the
#: 2026-09-22 `base_probe` listed, with `test_no_live_corpus_row_...` carrying two
#: because it reads both the corpus and the session store.
ABSENCE_EDGES = [
    pytest.param(test_the_live_corpus_emits_one_file_per_sequence_pattern, "corpus",
                 id="sequence-pattern-per-file/corpus"),
    pytest.param(test_the_verdict_ledger_join_still_resolves_every_coarse_error_key,
                 "ledger", id="verdict-ledger-join/ledger"),
    pytest.param(test_the_merge_moves_nobody_out_of_suppression, "corpus",
                 id="merge-suppression/corpus"),
    pytest.param(test_a_live_week_emits_one_merged_candidate_per_error_key, "corpus",
                 id="merged-candidate-per-error-key/corpus"),
    pytest.param(test_a_full_live_week_emits_one_file_per_path_across_every_class,
                 "corpus", id="one-file-per-path-every-class/corpus"),
    pytest.param(test_no_live_corpus_row_is_interactive_on_a_machine_platform, "corpus",
                 id="classifier-join/corpus"),
    pytest.param(test_no_live_corpus_row_is_interactive_on_a_machine_platform, "store",
                 id="classifier-join/store"),
    pytest.param(test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session,
                 "store", id="classifier-oracle/store"),
]


@pytest.mark.parametrize("guard,root", ABSENCE_EDGES)
def test_an_absent_live_root_skips_every_guard_that_reads_it_by_name(
        guard, root, tmp_path, monkeypatch):
    """Clauses 1 and 4, as one row per edge: redirecting the root a guard reads to a
    path that does not exist must produce a SKIP whose reason NAMES that path.

    Run by name rather than by a marker, so a guard that hard-asserts again fails here
    with `DID NOT RAISE Skipped` instead of silently joining the skips, and so the
    reason text — which is the only thing a reader of a skipped run gets — is checked,
    not assumed. The root is redirected, not merely assumed-missing, so this pins the
    rule on a machine that has the corpus too.
    """
    # The other roots are put in place first, so the reason can only name THIS one:
    # with the corpus absent as well, a guard that reads both would have skipped on the
    # corpus and this row would pass having said nothing about the store.
    for other in LIVE_ROOTS:
        if other != root:
            redirect_live_root(monkeypatch, other, _PRESENT_ROOTS[other](tmp_path))
    missing = redirect_live_root(monkeypatch, root, tmp_path / "gone" / root)
    with pytest.raises(pytest.skip.Exception) as caught:
        guard(**guard_arguments(guard, tmp_path))
    assert str(missing) in str(caught.value), (
        f"{guard.__name__} skipped for a reason that does not name the missing root "
        f"{missing}: {caught.value}")


def test_a_present_but_empty_corpus_directory_fails_rather_than_skips(
        tmp_path, monkeypatch):
    """The sharpest form of clause 3: the root EXISTS and is empty.

    `require_live_data` is satisfied — the path is there, and it is a directory — so the
    guard has to reach its own `loaded no rows` assert. This is the case a
    `if not live.exists(): skip` written against the directory *contents* would get
    wrong: emptying `_pipeline/trajectories` would then read exactly like losing it, and
    a guard that cannot tell those two states apart is not reporting on the data."""
    empty = tmp_path / "corpus"
    empty.mkdir()
    redirect_live_root(monkeypatch, "corpus", empty)
    with pytest.raises(AssertionError) as caught:
        test_the_live_corpus_emits_one_file_per_sequence_pattern(tmp_path / "out")
    assert "loaded no rows" in str(caught.value), caught.value


def test_the_rule_skips_on_absence_and_fails_on_the_wrong_kind_of_path(tmp_path):
    """The helper's own two answers, so a guard cannot get them by writing its own
    `if not path.exists()` and losing the second one."""
    missing = tmp_path / "absent-corpus"
    with pytest.raises(pytest.skip.Exception) as caught:
        require_live_data(missing, "live trajectory corpus")
    assert str(missing) in str(caught.value)

    a_file = tmp_path / "corpus"
    a_file.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="not a directory"):
        require_live_data(a_file, "live trajectory corpus")

    a_dir = tmp_path / "reviews"
    a_dir.mkdir()
    with pytest.raises(AssertionError, match="not a file"):
        require_live_data(a_dir, "verdict ledger", kind="file")


def test_the_volume_skip_reason_carries_the_floor_and_the_observed_count(tmp_path):
    """Clause 2's reason shape: `assert len(files) > 500, "so this is vacuous"` said
    neither number, so a 58-file store and an empty one were indistinguishable in the
    log. The reason has to hold both, beside the root it counted."""
    root = tmp_path / "sessions"
    with pytest.raises(pytest.skip.Exception) as caught:
        require_live_volume([1, 2, 3], SESSION_STORE_MIN_FILES, root, "session store")
    reason = str(caught.value)
    assert "holds 3 files" in reason, reason
    assert f"the {SESSION_STORE_MIN_FILES}-file floor" in reason, reason
    assert str(root) in reason, reason

    # Same shape for the corpus-volume floor #1403 added (`noun="keys"`, because a bare
    # `count` would have produced `the 2-count floor`): a reader of a skipped run gets
    # one line, and that line has to say which floor, how many, and where.
    keys = tmp_path / "trajectories"
    with pytest.raises(pytest.skip.Exception) as caught:
        require_live_volume([1], MERGE_MULTI_SIGNATURE_KEYS_MIN, keys, "mined error "
                            "keys with more than one signature behind them",
                            noun="keys")
    reason = str(caught.value)
    assert "holds 1 keys" in reason, reason
    assert f"the {MERGE_MULTI_SIGNATURE_KEYS_MIN}-key floor" in reason, reason
    assert str(keys) in reason, reason


def test_the_merge_guard_routes_through_the_one_floor_helper_and_states_its_floor_once():
    """The floor lives in `MERGE_MULTI_SIGNATURE_KEYS_MIN` and nowhere else, and the
    below-floor decision is made by `require_live_volume`, not by an inline comparison.

    #1403 clause 5 is that the guard skips BY NAME of the floor and the count. A guard
    that kept its own `assert len(multi) >= 2` and added a skip above it would satisfy
    the observable behaviour today and rot the moment one number was edited and the
    other was not — which is the same defect `SESSION_STORE_MIN_FILES` was lifted out
    of `#1143`'s inline `> 500` to prevent. So this pins the wiring, over the function's
    own source: the call is there, it names the constant, and the duplicated comparison
    is not.
    """
    src = inspect.getsource(test_a_live_week_emits_one_merged_candidate_per_error_key)
    assert "require_live_volume(" in src, (
        "the merge guard decides corpus depth on its own again, so its skip reason "
        "carries whatever wording this file happens to use rather than the one shape "
        "the volume helper gives every live guard")
    assert "MERGE_MULTI_SIGNATURE_KEYS_MIN" in src, (
        "the floor is a literal again: the skip and the assertion it replaces are two "
        "numbers that an edit can move apart")
    dupes = re.findall(r"assert\s+len\(multi\)\s*[<>=]", src)
    assert not dupes, (
        f"an inline comparison survived the move into the helper, so the floor is "
        f"stated {len(dupes) + 1} times over: {dupes}")


def test_a_store_below_the_floor_skips_naming_both_numbers(tmp_path, monkeypatch):
    """Clause 2, first half, over the real guard: the post-wipe store (3 synthetic
    files) skips, and the skip says the floor and the count."""
    redirect_live_root(monkeypatch, "store", synthetic_store(tmp_path, 3))
    with pytest.raises(pytest.skip.Exception) as caught:
        test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session()
    reason = str(caught.value)
    assert "holds 3 files" in reason, reason
    assert f"the {SESSION_STORE_MIN_FILES}-file floor" in reason, reason


def test_a_full_store_with_a_disagreeing_classifier_fails_rather_than_skips(
        tmp_path, monkeypatch):
    """Clause 2, second half, and the one that keeps the two skips honest.

    A synthetic store ABOVE the floor — `SESSION_STORE_MIN_FILES` chat-shaped session
    files, every one a mission-control chat typed with the observer on, which is the
    shape the ruled rule admits as `interactive` — and the classifier replaced by the
    rule #1143 overturned: `interactive` only when the observer switch is off. Every
    row then disagrees with the rule the guard reads, so the guard must FAIL. A skip
    that could fire here would mean the oracle stopped being an oracle.
    """
    redirect_live_root(
        monkeypatch, "store", synthetic_store(tmp_path, SESSION_STORE_MIN_FILES))

    def pre_1143_rule(data, session_id=None):
        if data.get("platform") == "mission-control" and not data.get("inner_voice"):
            return et.INTERACTIVE_CLASS
        return et.UNKNOWN_CLASS

    monkeypatch.setattr(et, "classify_session", pre_1143_rule)
    with pytest.raises(AssertionError) as caught:
        test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session()
    assert "classifier disagrees with the ruled rule" in str(caught.value), caught.value


# ── #1448: the observer control on a scale a wiped store survives ─────────────
#
# `SESSION_STORE_MIN_FILES` did its job and the store regrew past it, which is what
# made the guard's OTHER bound — `assert human_with_observer_on > 20`, calibrated on the
# store of 2026-09-18 — fire in every round's `tests` rung on a classifier that agreed
# with the ruled rule on all 514 files. Verbatim from
# `~/.local/state/lloyd-automod/rounds/SM_20260924_114808/gate.json`: "every failure
# reproduces at base 391f03f4 with this round's diff absent — PRE-EXISTING BREAKAGE".
# The three nodes below pin the new control's three states from the outside, and each
# one refuses a SKIP where a failure or a pass is owed, because a red node that became a
# silently-skipped one would have closed this item without ever guarding anything again.


def test_a_store_with_too_few_admitted_chats_skips_naming_floor_and_count(
        tmp_path, monkeypatch):
    """#1448 clause 2: full in files, too thin in admitted chats to judge a share.

    `SESSION_STORE_MIN_FILES` machine files and 4 observer-on chats — above the file
    floor, below `ADMITTED_HUMAN_SESSIONS_MIN`. That is the state a wiped-then-regrown
    human slice legitimately reaches, so the node skips rather than reporting a verdict
    it cannot justify, and the reason names BOTH the admitted floor and the observed
    human count, the attribution #1377 required of the file-volume skip."""
    thin = ADMITTED_HUMAN_SESSIONS_MIN - 1
    redirect_live_root(monkeypatch, "store", synthetic_mixed_store(
        tmp_path, "thin-human", SESSION_STORE_MIN_FILES, thin, inner_voice=True))
    with pytest.raises(pytest.skip.Exception) as caught:
        test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session()
    reason = str(caught.value)
    assert "admitted human sessions" in reason, (
        f"the skip named the file floor, not the admitted-session floor, so a reader "
        f"of a skipped run cannot tell which of the two the store missed: {reason}")
    assert f"holds {thin} sessions" in reason, reason
    assert f"the {ADMITTED_HUMAN_SESSIONS_MIN}-session floor" in reason, reason


def test_a_full_store_whose_admitted_chats_all_lack_inner_voice_fails(
        tmp_path, monkeypatch):
    """#1448 clause 3, the purpose clause: the state #1143 was built to catch, on the
    new scale.

    A store ABOVE the file floor and above `ADMITTED_HUMAN_SESSIONS_MIN`, whose admitted
    human chats every one carries `inner_voice: false` — the admitted set containing no
    chat typed with the observer on, which is precisely what pre-#1143 looked like (0 of
    45). It must FAIL. A share that skipped here would trade the guard this item exists
    to keep for a number it can no longer reach."""
    redirect_live_root(monkeypatch, "store", synthetic_mixed_store(
        tmp_path, "observer-off", SESSION_STORE_MIN_FILES,
        ADMITTED_HUMAN_SESSIONS_MIN, inner_voice=False))
    try:
        test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session()
    except pytest.skip.Exception as exc:
        pytest.fail(f"the observer control skipped over a full store instead of failing "
                    f"on an admitted set with no observer-on chat in it: {exc}")
    except AssertionError as exc:
        assert "inner_voice: true" in str(exc), (
            f"the guard failed for a reason other than the observer control, so the "
            f"control itself is untested here: {exc}")
    else:
        pytest.fail("the observer control passed a store whose admitted human chats "
                    "all carry `inner_voice: false`")


def test_a_full_store_with_a_live_shaped_human_slice_passes_without_skipping(
        tmp_path, monkeypatch):
    """#1448 clause 4: the new skip is for a store that cannot be judged, not for a
    store that is merely the size this machine's is.

    The shape measured on the live store on 2026-09-24 — above `SESSION_STORE_MIN_FILES`
    files, 7 admitted chats, all observer-on — must RUN and PASS. It is exactly the
    store the absolute `> 20` failed, and a guard that skipped here would have answered
    a red promotion-blocking node by executing nothing."""
    redirect_live_root(monkeypatch, "store", synthetic_mixed_store(
        tmp_path, "live-shape", SESSION_STORE_MIN_FILES,
        ADMITTED_HUMAN_SESSIONS_MIN + 2, inner_voice=True))
    try:
        test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session()
    except pytest.skip.Exception as exc:
        pytest.fail(f"the guard skipped over a full store whose admitted slice is above "
                    f"the floor instead of passing it: {exc}")


def test_the_observer_control_states_its_share_and_floors_once():
    """#1448 clauses 1, 2 and 4, pinned against the guard's own source.

    The three nodes beside this one pin the behaviour; this pins the SHAPE, because
    #1448 is the failure in which a number that began as a measurement quietly became a
    bound nobody could meet. So: the observer control compares to `OBSERVER_ON_SHARE_MIN`
    and to no absolute count; the below-floor decision is `require_live_volume` naming
    `ADMITTED_HUMAN_SESSIONS_MIN` rather than an inline comparison, so the skip reason and
    the floor cannot be moved apart; the two assertions #1143 wrote are still in force
    unreduced; and `SESSION_STORE_MIN_FILES` is still 501 — #1143's `assert len(files) >
    500` stated as a floor — and not the count of files the store happens to hold today."""
    src = inspect.getsource(
        test_the_classifier_agrees_with_the_ruled_rule_over_every_live_session)
    assert "assert not violations" in src, (
        "the per-file classifier-vs-ruled-rule assertion is gone, so the oracle has "
        "stopped being an oracle and every bound under it proves nothing")
    assert "0 < human < len(files)" in src, (
        "#1143's admitted-set guard was dropped or narrowed")
    assert "OBSERVER_ON_SHARE_MIN" in src, (
        "the observer control no longer compares to its share constant, so the bound "
        "and the skip are two numbers again")
    assert not re.search(r"human_with_observer_on\s*(?:<=?|>=?)\s*\d", src), (
        "an absolute count of observer-on chats is the bound again: it describes the "
        "store that was measured, and on the day that store is wiped and regrown it "
        "fails on a classifier that never changed")
    # The CALL form, not the bare name: the name also appears in the oracle's own
    # docstring, so a pin that only looked for it kept passing after the floor moved
    # back inline — the reviewer's finding on round SM_20260924_121936.
    assert re.search(
            r"require_live_volume\(\s*admitted\s*,\s*ADMITTED_HUMAN_SESSIONS_MIN", src), (
        "the admitted-session floor is no longer the floor argument of a "
        "require_live_volume call over the admitted set, so the skip reason can "
        "disagree with the bound")
    assert not re.search(r"human\s*<\s*\d+", src), (
        "an inline admitted-session comparison duplicates the floor constant")
    assert SESSION_STORE_MIN_FILES == 501, (
        "the file floor moved: lowering it toward the observed count is how a wiped "
        "store gets a quieter oracle instead of a named skip")


def test_a_present_corpus_that_admits_nothing_fails_rather_than_skips(
        tmp_path, monkeypatch):
    """Clause 3 for `test_the_live_corpus_emits_one_file_per_sequence_pattern`.

    Present corpus, three copies of `adjacent_next_step_traj`: its n-grams are the
    refused half of the #1181 gate (the next step after a failure is a different call
    that merely touches the same file), so every mined sequence is refused, the mined
    and written sequence sets are both empty, and the counts that guard compares would
    agree. It must fail on `the live window must hold both kinds`, not skip.
    """
    rows = [adjacent_next_step_traj(f"alias1377_{i}") for i in (1, 2, 3)]
    redirect_live_root(monkeypatch, "corpus", synthetic_corpus(tmp_path, rows))
    with pytest.raises(AssertionError) as caught:
        test_the_live_corpus_emits_one_file_per_sequence_pattern(tmp_path / "out")
    assert "must hold both kinds" in str(caught.value), caught.value


def test_a_guard_that_reads_a_different_path_than_it_checks_is_caught(
        tmp_path, monkeypatch):
    """Clause 4 of #1403: checking a corpus and reading it are two different
    statements, and until this round the sequence-pattern guard made only the first.

    Here `LIVE_CORPUS` points at a synthetic corpus that WOULD satisfy every
    assertion in the guard, while the miner's own `TRAJECTORY_DIR` — what
    `load_trajectories` actually opens, since it takes no path argument — is left on
    an empty directory. That is exactly the state a round's worktree produces: the
    production corpus is real two directories away, `<worktree>/.lloyd-data/_pipeline/
    trajectories` is not, and the guard loaded 0 rows and failed with `loaded no rows`
    while the `require_live_data` check above it had passed. So the corpus is present
    here, the rows load, and the ONLY way this fails is a guard that checked one path
    and read another.

    The node fails three ways, all fatal, none silently-skipping: the guard raises
    (the #1403 regression or any other), the guard skips (a guard that skips on a
    present, readable corpus is not checking what it validated), or the guard returns
    without writing the files it is named for.
    """
    corpus = synthetic_corpus(tmp_path, sequence_rows())
    redirect_live_root(monkeypatch, "corpus", corpus)
    worktree_mirror = tmp_path / "worktree" / "_pipeline" / "trajectories"
    worktree_mirror.mkdir(parents=True)
    monkeypatch.setattr(mt, "TRAJECTORY_DIR", worktree_mirror)
    try:
        test_the_live_corpus_emits_one_file_per_sequence_pattern(tmp_path / "out")
    except pytest.skip.Exception as why:
        pytest.fail(f"the sequence-pattern guard skipped over a corpus that exists and "
                    f"has rows in it, so it is not reading what it validated: {why}")
    except AssertionError as why:
        pytest.fail(f"the sequence-pattern guard checked {corpus} and then read "
                    f"{worktree_mirror} — the #1403 missing-binding regression, in "
                    f"whatever form, is what this message means: {why}")
    assert list(((tmp_path / "out") / "candidates").glob("candidate-seq-*.md")), (
        "the guard returned without writing the per-sequence files it is named for, "
        "so it ran over nothing")


def test_a_present_ledger_below_its_coarse_key_floor_still_fails(
        tmp_path, monkeypatch):
    """Clause 3 for the verdict-ledger join: a ledger that EXISTS but has lost rows is
    the defect that guard was written for, so 3 coarse keys against the recorded 21 must
    fail, not skip. `pattern_key`s are synthetic so no real verdict can satisfy it."""
    rows = [{"pattern_key": f"Guardtest1377{i}/not_found", "verdict": "rejected",
             "decided_at": "2026-09-22T00:00:00+00:00"} for i in range(3)]
    redirect_live_root(monkeypatch, "ledger", synthetic_ledger(tmp_path, rows))
    with pytest.raises(AssertionError) as caught:
        test_the_verdict_ledger_join_still_resolves_every_coarse_error_key()
    assert "coarse keys in a ledger that held 21" in str(caught.value), caught.value


def test_a_corpus_the_verdict_ledger_gates_none_of_fails_rather_than_skips(
        tmp_path, monkeypatch):
    """Clause 3 for the suppression-set guard: one mined error pattern whose key no
    ledger row knows, so the gated-before set is empty. The guard's own invariant is
    that a judged pattern is still gated after the merge, and an ungated corpus cannot
    demonstrate that — it must fail on `the ledger gates none of`, not skip."""
    rows = [error_traj(f"gate1377_{i}", "Ungated1377", "not_found", "protocol")
            for i in (1, 2)]
    redirect_live_root(monkeypatch, "corpus", synthetic_corpus(tmp_path, rows))
    with pytest.raises(AssertionError) as caught:
        test_the_merge_moves_nobody_out_of_suppression()
    assert "gates none of" in str(caught.value), caught.value


def test_a_corpus_whose_error_keys_collide_on_one_key_skips_naming_both_numbers(
        tmp_path, monkeypatch):
    """Clause 5 of #1403, and the deliberate reversal of one #1377 clause-3 node.

    That node — `test_a_corpus_whose_error_keys_never_collide_fails_rather_than_skips`,
    deleted here — demanded a FAILURE on `more than one signature` for exactly this
    corpus, on the argument that a present corpus too thin to collide was a finding
    rather than an environment. #1403 is the evidence that the argument was wrong about
    this one guard: `_pipeline/trajectories` restarted at zero with the 2026-09-22
    deletion and regrows one bucket a day, so that assertion spent five days red at
    base and refused every promotion through the gate's `tests` rung while a corpus
    refilled itself. A corpus that is present, readable and merely shallow is a
    measurement the machine cannot make YET, and `require_live_volume` is the rule for
    that state: skip, naming the floor and the observed count beside the path.

    What the reversal does NOT give back is the failure the old node protected: the
    guard still fails on a corpus deep enough to collide whose merged totals do not add
    up, and `test_a_two_key_fixture_corpus_runs_the_merge_guard_rather_than_skipping`
    below pins that this node's fixture skips while a two-key fixture runs. A node that
    skipped over BOTH corpora would keep this one green — that is the failure mode the
    second node exists to catch.
    """
    rows = ([error_traj(f"collide1377_{i}", "Solo1377a", "not_found", "protocol")
             for i in (1, 2)]
            + [error_traj(f"collide1377_{i}", "Solo1377b", "not_found", "protocol")
               for i in (1, 2)])
    corpus = synthetic_corpus(tmp_path, rows)
    redirect_live_root(monkeypatch, "corpus", corpus)
    with pytest.raises(pytest.skip.Exception) as caught:
        test_a_live_week_emits_one_merged_candidate_per_error_key(tmp_path / "out")
    reason = str(caught.value)
    assert "holds 0 keys" in reason, (
        f"a one-signature-per-key corpus must be reported as 0 discriminating keys, "
        f"not counted some other way: {reason}")
    assert f"the {MERGE_MULTI_SIGNATURE_KEYS_MIN}-key floor" in reason, (
        f"the skip must name the floor it applied, which is the guard's own `>= 2`: "
        f"{reason}")
    assert str(corpus) in reason, (
        f"and the path it counted, or a skipped run says nothing about which corpus "
        f"was too shallow: {reason}")


def test_a_corpus_whose_merge_floor_is_one_key_short_skips_naming_the_count(
        tmp_path, monkeypatch):
    """The boundary itself: ONE key with more than one signature behind it — 1 against
    the floor of 2 — must skip, not fail and not pass.

    This is the live reading this machine has had since 2026-09-22 (`Bash/timeout` over
    a 2-file corpus), so it is also the reproduction of the red-at-base node #1403 was
    filed for. Asserting `holds 1 keys` rather than merely `below the floor` is what
    stops the helper from reporting the number it did not measure.
    """
    corpus = synthetic_corpus(tmp_path, merge_rows())
    redirect_live_root(monkeypatch, "corpus", corpus)
    with pytest.raises(pytest.skip.Exception) as caught:
        test_a_live_week_emits_one_merged_candidate_per_error_key(tmp_path / "out")
    reason = str(caught.value)
    assert "holds 1 keys" in reason, (
        f"the skip has to state the observed count beside the floor: {reason}")
    assert f"the {MERGE_MULTI_SIGNATURE_KEYS_MIN}-key floor" in reason, reason
    assert str(corpus) in reason, reason


def test_a_two_key_fixture_corpus_runs_the_merge_guard_rather_than_skipping(
        tmp_path, monkeypatch):
    """The positive control that keeps clause 5 honest: a corpus with
    `MERGE_MULTI_SIGNATURE_KEYS_MIN` = 2 colliding keys is enough to discriminate the
    merge, so the guard must RUN — every merged-total assertion below the skip fires,
    fatally — and may not skip.

    Without this node, routing the guard through the volume helper is unfalsifiable in
    the flattering direction: a floor that ignored its argument, or was quietly raised
    to 3, would skip the production run on every machine forever and the clause-3
    failure this guard exists to catch would never be observable again. So the two
    fixtures are the pair: 1 discriminating key skips (node above), 2 runs (this one),
    and the boundary is the floor, not a constant chosen to make today green.
    """
    corpus = synthetic_corpus(tmp_path, merge_rows_two_keys())
    redirect_live_root(monkeypatch, "corpus", corpus)
    out = tmp_path / "out"
    out.mkdir()
    try:
        test_a_live_week_emits_one_merged_candidate_per_error_key(out)
    except pytest.skip.Exception as why:
        pytest.fail(
            f"the merge guard skipped over a corpus with "
            f"{MERGE_MULTI_SIGNATURE_KEYS_MIN} colliding keys, so it skipped on "
            f"something other than the merge floor and the totals below it are never "
            f"checked on live data: {why}")
    error_bodies = [p.read_text(encoding="utf-8")
                    for p in out.glob("candidate-*.md")]
    assert len(error_bodies) >= MERGE_MULTI_SIGNATURE_KEYS_MIN, (
        f"the guard returned but wrote {len(error_bodies)} candidates for a corpus "
        f"mining {MERGE_MULTI_SIGNATURE_KEYS_MIN} merged error keys")
    assert sum(1 for b in error_bodies
               if int(frontmatter_field(b, "occurrences")) > 2) >= (
        MERGE_MULTI_SIGNATURE_KEYS_MIN), (
        "no candidate's `occurrences` exceeds any single signature bucket's own total, "
        "so nothing was actually merged even though the guard claims it ran")


def test_a_corpus_that_writes_no_sequence_candidate_fails_rather_than_skips(
        tmp_path, monkeypatch):
    """Clause 3 for the every-class path guard, in the review's own words.

    Round SM_20260914_140724 refused this guard because a corpus that mined error
    patterns and no sequence pattern left `assert "error" in kinds` green while the
    sequence half of the check never ran. So: a present corpus of single-tool error
    rows — error patterns mine, sequences cannot exist by construction — must fail on
    `no sequence candidate was written`, the assert that half of that finding put
    there."""
    rows = [error_traj(f"noseq1377_{i}", "Solo1377", "not_found", "protocol")
            for i in (1, 2)]
    redirect_live_root(monkeypatch, "corpus", synthetic_corpus(tmp_path, rows))
    with pytest.raises(AssertionError) as caught:
        test_a_full_live_week_emits_one_file_per_path_across_every_class(tmp_path / "out")
    assert "no sequence candidate was written" in str(caught.value), caught.value


def test_a_corpus_row_labelled_interactive_on_a_machine_platform_fails(
        tmp_path, monkeypatch):
    """Clause 3 for the #493 join guard, on a present corpus and a present store: one
    row stamped `session_class: interactive` whose session file says `platform: worker`.
    That is a bad re-extraction, the exact thing this guard exists to catch, so it must
    fail on the emitted-class half of its own check rather than skip."""
    key = "20260920_120000_guard1377"
    row = dict(adjacent_next_step_traj(key), session_class="interactive")
    redirect_live_root(monkeypatch, "corpus", synthetic_corpus(tmp_path, [row]))
    redirect_live_root(monkeypatch, "store",
                       synthetic_store(tmp_path, 1, session_id=key, platform="worker"))
    with pytest.raises(AssertionError) as caught:
        test_no_live_corpus_row_is_interactive_on_a_machine_platform()
    assert "onto a machine-platform session" in str(caught.value), caught.value
    assert key in str(caught.value), caught.value
