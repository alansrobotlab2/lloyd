"""Hypothesis generator — JSON repair, variant parsing, ledger signal readers.

Why this file exists
--------------------
This is the code that proposes what to change about my own prompts. Two of its
failure modes are documented incidents:

  * 188 `hypothesis_fail_*.txt` dumps accumulated under
    `_pipeline/research/_debug/` with no rotation, from outputs that echoed the
    prompt files back and hit `max_tokens`.
  * `finish_reason: length` is logged as "possible truncation" but is not
    retried and `_try_parse_json` cannot repair it.

Both live on the parse path tested here. `DEBUG_DIR` is the real `_pipeline`
diagnostics dir, so the autouse fixture redirects it: nothing in this module
writes into `_pipeline`.
"""
from __future__ import annotations

import json

import pytest

from scripts.autoresearch import hypothesis_generator as hg

# Captured at import, before the autouse fixture redirects DEBUG_DIR.
LIVE_DEBUG_DIR = hg.DEBUG_DIR


# ── _try_parse_json ──────────────────────────────────────────────────────────

def test_clean_json_parses():
    obj, err = hg._try_parse_json('{"a": 1}')
    assert obj == {"a": 1} and err is None


def test_json_embedded_in_prose_is_extracted():
    obj, err = hg._try_parse_json('Here you go:\n{"a": 1}\nHope that helps!')
    assert obj == {"a": 1} and err is None


def test_markdown_fenced_json_parses():
    obj, _ = hg._try_parse_json('```json\n{"a": 1}\n```')
    assert obj == {"a": 1}


def test_trailing_comma_is_repaired():
    """The documented repair pass: strip a comma before } or ]."""
    obj, err = hg._try_parse_json('{"a": 1, "b": [1, 2,],}')
    assert obj == {"a": 1, "b": [1, 2]} and err is None


def test_empty_input_reports_empty():
    obj, err = hg._try_parse_json("")
    assert obj is None and err == "empty response"


def test_no_json_object_at_all():
    obj, err = hg._try_parse_json("the model just wrote prose")
    assert obj is None and err == "no JSON object found"


def test_unrepairable_json_reports_both_attempts():
    obj, err = hg._try_parse_json('{"a": ,}')
    assert obj is None
    assert "repair also failed" in err


def test_greedy_brace_match_spans_two_objects():
    """Characterized limitation: the extractor regex is `{.*}` with DOTALL, so
    output containing two top-level objects is joined and parses as neither."""
    obj, err = hg._try_parse_json('{"a": 1}\n{"b": 2}')
    assert obj is None and "repair also failed" in err


def test_truncation_is_not_repairable():
    """The 188-dump failure mode: a `finish_reason: length` response is a
    truncated object. With no closing brace the extractor never even attempts a
    parse — which is why the fix is a retry or a smaller prompt, not a better
    repair pass."""
    echoed = '{"overlay_files": {"MEMORY.md": "' + ("x" * 4000)
    obj, err = hg._try_parse_json(echoed)
    assert obj is None and err == "no JSON object found"


def test_truncation_with_a_stray_closing_brace_still_fails():
    """A truncated body that happens to contain a `}` reaches the parser and
    fails both attempts."""
    echoed = '{"overlay_files": {"MEMORY.md": "' + ("x" * 200) + ' yyy"}'
    obj, err = hg._try_parse_json(echoed)
    assert obj is None and "repair also failed" in err


# ── _parse_single_variant ────────────────────────────────────────────────────

def test_minimal_valid_variant():
    raw = json.dumps({
        "description": "tighten the gate", "hypothesis": "less hedging",
        "overlay_files": {"SOUL.md": "NEW SOUL"},
    })
    v, err = hg._parse_single_variant(raw)
    assert err is None
    assert v["target_surface"] == "prompts"
    assert v["overlay_files"] == {"SOUL.md": "NEW SOUL"}
    assert v["description"] == "tighten the gate"
    assert v["variant_id"].startswith("V_") and v["created_at"].endswith("Z")


def test_wrapped_variants_list_is_unwrapped():
    raw = json.dumps({"variants": [{"overlay_files": {"SOUL.md": "s"}}]})
    v, err = hg._parse_single_variant(raw)
    assert err is None and v["overlay_files"] == {"SOUL.md": "s"}


def test_non_prompts_target_is_refused():
    raw = json.dumps({"target_surface": "tool_allowlist",
                      "overlay_files": {"SOUL.md": "s"}})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "unsupported target_surface" in err


def test_empty_overlay_is_refused():
    v, err = hg._parse_single_variant(json.dumps({"overlay_files": {}}))
    assert v is None and "overlay_files missing/empty" in err


def test_missing_overlay_key_is_refused():
    v, err = hg._parse_single_variant(json.dumps({"description": "d"}))
    assert v is None and "overlay_files missing/empty" in err


def test_unsupported_overlay_keys_are_filtered_out():
    """USER.md is not an allowed overlay key here — the generator may only
    propose SOUL/MEMORY edits even though promotion *can* write USER.md."""
    raw = json.dumps({"overlay_files": {"config.yaml": "x", "SOUL.md": "s"}})
    v, err = hg._parse_single_variant(raw)
    assert err is None and v["overlay_files"] == {"SOUL.md": "s"}


def test_only_user_md_yields_no_valid_keys():
    raw = json.dumps({"overlay_files": {"USER.md": "u"}})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "no valid overlay keys" in err


def test_blank_overlay_content_is_dropped():
    raw = json.dumps({"overlay_files": {"SOUL.md": "   ", "MEMORY.md": "real"}})
    v, err = hg._parse_single_variant(raw)
    assert err is None and v["overlay_files"] == {"MEMORY.md": "real"}


def test_multiple_overlays_are_collapsed_to_one():
    """Characterized: the prompt asks for one file and the parser enforces it by
    keeping an arbitrary first key, so a two-file proposal silently tests only
    part of what was proposed."""
    raw = json.dumps({"overlay_files": {"SOUL.md": "s", "MEMORY.md": "m"}})
    v, err = hg._parse_single_variant(raw)
    assert err is None and len(v["overlay_files"]) == 1
    assert list(v["overlay_files"]) == ["SOUL.md"]


def test_wrapped_list_with_a_non_object_first_entry():
    v, err = hg._parse_single_variant(json.dumps({"variants": ["nope"]}))
    assert v is None and "variants[0] is not an object" in err


def test_top_level_array_is_refused():
    v, err = hg._parse_single_variant("[1, 2]")
    assert v is None and err is not None


def test_long_fields_are_truncated():
    raw = json.dumps({"description": "d" * 500, "hypothesis": "h" * 3000,
                      "overlay_files": {"SOUL.md": "s"}})
    v, _ = hg._parse_single_variant(raw)
    assert len(v["description"]) == 200
    assert len(v["hypothesis"]) == 1000


def test_parent_lineage_is_carried_when_present():
    raw = json.dumps({"overlay_files": {"SOUL.md": "s"}, "parent_variant_id": "V_p"})
    assert hg._parse_single_variant(raw)[0]["parent_variant_id"] == "V_p"


def test_absent_parent_lineage_is_none_not_empty_string():
    raw = json.dumps({"overlay_files": {"SOUL.md": "s"}})
    assert hg._parse_single_variant(raw)[0]["parent_variant_id"] is None


def test_each_variant_gets_a_distinct_id():
    raw = json.dumps({"overlay_files": {"SOUL.md": "s"}})
    ids = {hg._parse_single_variant(raw)[0]["variant_id"] for _ in range(10)}
    assert len(ids) == 10


# ── ledger signal readers ────────────────────────────────────────────────────

def line(**kw):
    return json.dumps(kw)


def test_losers_from_a_missing_ledger(tmp_path):
    assert hg._recent_ledger_losers(tmp_path / "nope.jsonl") == []


def test_losers_are_unpromoted_entries_with_scores(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join([
        line(promoted=False, composite_score=0.31, variant_id="V_a"),
        line(promoted=True, composite_score=0.9, variant_id="V_b"),
        line(promoted=False, variant_id="V_noscore"),
    ]) + "\n", encoding="utf-8")
    losers = hg._recent_ledger_losers(p)
    assert [l["variant_id"] for l in losers] == ["V_a"]


def test_losers_are_newest_first_and_capped(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join(
        line(promoted=False, composite_score=0.2, variant_id=f"V_{i}") for i in range(20)
    ) + "\n", encoding="utf-8")
    losers = hg._recent_ledger_losers(p, limit=3)
    assert [l["variant_id"] for l in losers] == ["V_19", "V_18", "V_17"]


def test_garbage_lines_are_skipped(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("not json\n" + line(promoted=False, composite_score=0.1) + "\n",
                 encoding="utf-8")
    assert len(hg._recent_ledger_losers(p)) == 1


def test_losers_only_see_the_last_two_thousand_lines(tmp_path):
    """Characterized, and it matters: the live ledger is ~26k lines, so the
    generator's 'recent losers' window covers the newest ~8%. A long-standing
    losing pattern older than that window is invisible to the hypothesis."""
    p = tmp_path / "l.jsonl"
    old = line(promoted=False, composite_score=0.1, variant_id="V_old")
    filler = "\n".join(line(event="spec") for _ in range(2000))
    p.write_text(old + "\n" + filler + "\n", encoding="utf-8")
    assert hg._recent_ledger_losers(p) == []


def test_baseline_failures_need_a_baseline_variant_id(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(line(variant_id="V_real", task_id="t1", composite_score=0.1) + "\n",
                 encoding="utf-8")
    assert hg._recent_baseline_failures(p) == []


def test_baseline_failure_is_a_low_score(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(line(variant_id="BASELINE_1", task_id="t1", composite_score=0.3) + "\n",
                 encoding="utf-8")
    assert [e["task_id"] for e in hg._recent_baseline_failures(p)] == ["t1"]


def test_a_passing_baseline_score_is_not_a_failure(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(line(variant_id="BASELINE_1", task_id="t1", composite_score=0.9) + "\n",
                 encoding="utf-8")
    assert hg._recent_baseline_failures(p) == []


def test_baseline_safety_failure_counts_even_at_a_high_score(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(line(variant_id="BASELINE_1", task_id="bench_010", composite_score=0.9,
                      safety_critical=True, safety_passed=False) + "\n", encoding="utf-8")
    assert [e["task_id"] for e in hg._recent_baseline_failures(p)] == ["bench_010"]


def test_non_critical_safety_false_is_not_a_failure(tmp_path):
    """Characterized: a middling non-critical score is exactly 0.5 and reads as
    'not a failure', so the half-credit rubric outage is invisible to the
    generator's own idea of what needs fixing."""
    p = tmp_path / "l.jsonl"
    p.write_text(line(variant_id="BASELINE_1", task_id="t1", composite_score=0.5,
                      safety_critical=False, safety_passed=False) + "\n", encoding="utf-8")
    assert hg._recent_baseline_failures(p) == []


def test_each_failing_task_is_reported_once(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join([
        line(variant_id="BASELINE_2", task_id="t1", composite_score=0.1),
        line(variant_id="BASELINE_1", task_id="t1", composite_score=0.2),
        line(variant_id="BASELINE_1", task_id="t2", composite_score=0.2),
    ]) + "\n", encoding="utf-8")
    fails = hg._recent_baseline_failures(p)
    assert [e["task_id"] for e in fails] == ["t2", "t1"]      # newest-first, deduped


def test_failures_are_capped(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join(
        line(variant_id="BASELINE_1", task_id=f"t{i}", composite_score=0.1) for i in range(30)
    ) + "\n", encoding="utf-8")
    assert len(hg._recent_baseline_failures(p, limit=4)) == 4


# ── _read / _dump_raw_on_failure ─────────────────────────────────────────────

def test_read_missing_file_is_empty(tmp_path):
    assert hg._read(tmp_path / "absent.md") == ""


def test_read_returns_the_whole_file(tmp_path):
    f = tmp_path / "a.md"; f.write_text("abcdefghij", encoding="utf-8")
    assert hg._read(f) == "abcdefghij"


def test_read_tail_returns_the_suffix(tmp_path):
    f = tmp_path / "a.md"; f.write_text("abcdefghij", encoding="utf-8")
    assert hg._read(f, tail=3) == "hij"


def test_read_tail_larger_than_file_returns_everything(tmp_path):
    f = tmp_path / "a.md"; f.write_text("abc", encoding="utf-8")
    assert hg._read(f, tail=999) == "abc"


@pytest.fixture(autouse=True)
def redirect_debug_dir(tmp_path, monkeypatch):
    """Never write diagnostics into the live _pipeline/_debug pile."""
    d = tmp_path / "debug"
    monkeypatch.setattr(hg, "DEBUG_DIR", d)
    return d


def test_failure_dump_records_error_payload_and_raw(redirect_debug_dir):
    hg._dump_raw_on_failure("bad_comma", {"messages": [{"content": "x" * 100}],
                                         "max_tokens": 8000},
                            "RAWOUTPUT", "json parse failed")
    files = list(redirect_debug_dir.iterdir())
    assert len(files) == 1 and files[0].name.startswith("hypothesis_fail_")
    text = files[0].read_text(encoding="utf-8")
    assert "=== ERROR ===\njson parse failed" in text
    assert "RAWOUTPUT" in text and "(9 chars)" in text
    assert '"max_tokens": 8000' in text


def test_failure_dump_drops_message_bodies_but_keeps_their_size(redirect_debug_dir):
    """The dumps are the 32 KB prompt-echo artifacts; the payload is trimmed so
    they stay readable while still showing how big the prompt was."""
    hg._dump_raw_on_failure("trunc", {"messages": [{"content": "y" * 5000}],
                                      "model": "primary"}, "r", "e")
    text = next(redirect_debug_dir.iterdir()).read_text(encoding="utf-8")
    assert "y" * 5000 not in text
    assert '"_messages_len": 5000' in text
    assert '"model": "primary"' in text


def test_failure_dump_never_raises(redirect_debug_dir, monkeypatch):
    blocker = redirect_debug_dir.parent / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(hg, "DEBUG_DIR", blocker / "sub")
    hg._dump_raw_on_failure("x", {"messages": []}, "raw", "err")     # no exception


def test_unpatched_debug_dir_is_the_live_pipeline_dir():
    """Characterization of why the redirect fixture exists: diagnostics default
    into `_pipeline/research/_debug/`, which holds 188 unrotated dumps."""
    assert str(LIVE_DEBUG_DIR).endswith("_pipeline/research/_debug")
