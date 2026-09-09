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
from pathlib import Path

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
    """The 192-dump failure mode: a `finish_reason: length` response is a
    truncated object. With no closing brace the extractor never even attempts a
    parse — which is why the fix is #446's bounded output contract (and
    `_propose_one` failing the variant), not a better repair pass."""
    echoed = '{"edits": [{"path": "MEMORY.md", "anchor": "' + ("x" * 4000)
    obj, err = hg._try_parse_json(echoed)
    assert obj is None and err == "no JSON object found"


def test_truncation_with_a_stray_closing_brace_still_fails():
    """A truncated body that happens to contain a `}` reaches the parser and
    fails both attempts."""
    echoed = '{"edits": [{"anchor": "' + ("x" * 200) + ' yyy"}'
    obj, err = hg._try_parse_json(echoed)
    assert obj is None and "repair also failed" in err


# ── _parse_single_variant — the anchored-edit contract (#446) ────────────────
#
# The contract this parser enforces is the fix for 192 `hypothesis_fail_*.txt`
# dumps: a variant used to be required to return a prompt surface's COMPLETE
# replacement text, so the model echoed the 6.6 KB SOUL.md + MEMORY.md it had
# been handed and the response was cut off at `max_tokens: 8000`
# (`Unterminated string`, median 32,020 chars). A variant is now an anchored
# edit list — quote a span, say what replaces it — so response length is
# bounded by the edit, never by the size of the file being changed.

def edit(anchor="old text", replacement="new text", path="SOUL.md"):
    return {"path": path, "anchor": anchor, "replacement": replacement}


def verbatim_span(text, max_chars=160):
    """Longest single-line span in `text` that still fits `max_chars`.

    Stands in for "a span the model was actually shown": a real anchor has to be
    copied out of the prompt character-for-character, so a test that emulates a
    model response picks one that exists rather than inventing one.
    """
    best = ""
    for line in text.splitlines():
        stripped = line.strip()
        if 20 <= len(stripped) <= max_chars and len(stripped) > len(best):
            best = stripped
    return best


def test_minimal_valid_variant():
    raw = json.dumps({
        "description": "tighten the gate", "hypothesis": "less hedging",
        "edits": [edit()],
    })
    v, err = hg._parse_single_variant(raw)
    assert err is None
    assert v["target_surface"] == "prompts"
    assert v["edits"] == [edit()]
    assert "overlay_files" not in v, "the full-file contract is gone"
    assert v["description"] == "tighten the gate"
    assert v["variant_id"].startswith("V_") and v["created_at"].endswith("Z")


def test_multiple_edits_on_one_surface_are_kept_in_order():
    a, b = edit(anchor="alpha"), edit(anchor="beta")
    raw = json.dumps({"edits": [a, b]})
    v, err = hg._parse_single_variant(raw)
    assert err is None and v["edits"] == [a, b]


def test_wrapped_variants_list_is_unwrapped():
    raw = json.dumps({"variants": [{"edits": [edit()]}]})
    v, err = hg._parse_single_variant(raw)
    assert err is None and v["edits"] == [edit()]


def test_non_prompts_target_is_refused():
    raw = json.dumps({"target_surface": "tool_allowlist", "edits": [edit()]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "unsupported target_surface" in err


def test_missing_edits_is_refused():
    v, err = hg._parse_single_variant(json.dumps({"description": "d"}))
    assert v is None and "edits missing/empty" in err


def test_empty_edits_list_is_refused():
    v, err = hg._parse_single_variant(json.dumps({"edits": []}))
    assert v is None and "edits missing/empty" in err


def test_a_full_file_echo_is_no_longer_accepted():
    """The regression test for #446 itself: an 8 KB `overlay_files` blob — the
    shape whose truncation produced 178 of 192 dumps — must be refused, not
    parsed. Output that scales with the file is exactly what cannot fit under
    max_tokens."""
    raw = json.dumps({"overlay_files": {"MEMORY.md": "x" * 8000}})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "edits missing/empty" in err


def test_an_edit_on_a_non_prompt_path_is_rejected_not_filtered():
    """Stricter than the old contract, which silently dropped disallowed keys:
    a variant that touches config.yaml is refused outright, so a proposal
    cannot half-apply."""
    raw = json.dumps({"edits": [edit(path="config.yaml"), edit()]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "unsupported path" in err


def test_an_edit_on_user_md_is_rejected():
    """USER.md is not a generator surface even though promotion can write it."""
    v, err = hg._parse_single_variant(json.dumps({"edits": [edit(path="USER.md")]}))
    assert v is None and "unsupported path" in err


def test_an_empty_anchor_is_rejected():
    raw = json.dumps({"edits": [edit(anchor="   "), edit(path="MEMORY.md")]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "empty anchor" in err


def test_a_missing_replacement_is_rejected_rather_than_read_as_deletion():
    """A deletion is a legitimate `replacement: ""`, but an *absent* key is a
    malformed edit — treating it as a deletion would silently delete text the
    model never meant to quote."""
    raw = json.dumps({"edits": [{"path": "SOUL.md", "anchor": "old text"}]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "replacement" in err


def test_an_explicit_empty_replacement_is_a_legal_deletion():
    raw = json.dumps({"edits": [edit(replacement="")]})
    v, err = hg._parse_single_variant(raw)
    assert err is None and v["edits"][0]["replacement"] == ""


def test_edits_spanning_two_surfaces_are_rejected_not_collapsed():
    """This assertion replaced `test_multiple_overlays_are_collapsed_to_one`,
    which characterized keeping an arbitrary first file and silently scoring
    only part of a two-file proposal. #446's acceptance says apply is
    all-or-nothing with no partial write, so a split proposal is refused."""
    raw = json.dumps({"edits": [edit(path="SOUL.md"), edit(path="MEMORY.md")]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "one surface" in err


def test_an_anchor_bigger_than_the_bound_is_rejected():
    """The bound is what makes truncation structurally impossible: an anchor is
    a quoted span, and one this long is a file in new clothes."""
    raw = json.dumps({"edits": [edit(anchor="x" * (hg.MAX_ANCHOR_CHARS + 1))]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "anchor too long" in err


def test_a_replacement_bigger_than_the_bound_is_rejected():
    raw = json.dumps({"edits": [edit(replacement="x" * (hg.MAX_REPLACEMENT_CHARS + 1))]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "replacement too long" in err


def test_more_edits_than_the_bound_are_rejected():
    raw = json.dumps({"edits": [edit(anchor=f"a{i}") for i in range(hg.MAX_EDITS + 1)]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "too many edits" in err


def test_a_non_object_edit_is_rejected():
    raw = json.dumps({"edits": ["just a string"]})
    v, err = hg._parse_single_variant(raw)
    assert v is None and "not an object" in err


def test_wrapped_list_with_a_non_object_first_entry():
    v, err = hg._parse_single_variant(json.dumps({"variants": ["nope"]}))
    assert v is None and "variants[0] is not an object" in err


def test_top_level_array_is_refused():
    v, err = hg._parse_single_variant("[1, 2]")
    assert v is None and err is not None


def test_long_fields_are_truncated():
    raw = json.dumps({"description": "d" * 500, "hypothesis": "h" * 3000,
                      "edits": [edit()]})
    v, _ = hg._parse_single_variant(raw)
    assert len(v["description"]) == 200
    assert len(v["hypothesis"]) == 1000


def test_parent_lineage_is_carried_when_present():
    raw = json.dumps({"edits": [edit()], "parent_variant_id": "V_p"})
    assert hg._parse_single_variant(raw)[0]["parent_variant_id"] == "V_p"


def test_absent_parent_lineage_is_none_not_empty_string():
    raw = json.dumps({"edits": [edit()]})
    assert hg._parse_single_variant(raw)[0]["parent_variant_id"] is None


def test_each_variant_gets_a_distinct_id():
    raw = json.dumps({"edits": [edit()]})
    ids = {hg._parse_single_variant(raw)[0]["variant_id"] for _ in range(10)}
    assert len(ids) == 10


# ── the prompt contract (#446) ───────────────────────────────────────────────

@pytest.fixture
def fake_surfaces(tmp_path, monkeypatch):
    """Point the generator's vault paths at temp files so a prompt can be built
    without reading the live SOUL/MEMORY."""
    soul = tmp_path / "SOUL.md"
    soul.write_text("# SOUL\n" + "soul prose line\n" * 40, encoding="utf-8")
    memory = tmp_path / "MEMORY.md"
    memory.write_text("# MEMORY\n" + "memory note line\n" * 1400, encoding="utf-8")
    monkeypatch.setattr(hg, "SOUL_PATH", soul)
    monkeypatch.setattr(hg, "MEMORY_PATH", memory)
    monkeypatch.setattr(hg, "USER_PATH", tmp_path / "absent.md")
    monkeypatch.setattr(hg, "CORRECTIONS_PATH", tmp_path / "absent2.md")
    monkeypatch.setattr(hg, "KNOWLEDGE_HEALTH_PATH", tmp_path / "absent3.md")
    return soul, memory


def _cfg(tmp_path):
    from scripts.autoresearch.common import AutoresearchConfig, AutoresearchPaths
    paths = AutoresearchPaths(
        bench_dir=tmp_path / "bench", research_root=tmp_path / "r",
        rounds_dir=tmp_path / "r/rounds", ledger_path=tmp_path / "r/l.jsonl",
        variants_dir=tmp_path / "r/v", snapshots_dir=tmp_path / "r/s",
        facts_experiments_dir=tmp_path / "r/f",
    )
    return AutoresearchConfig(
        paths=paths, default_model="primary", default_budget_minutes=120,
        max_variants_per_round=7, promotion_min_win_fraction=0.5,
        promotion_min_composite_delta=0.05, promotion_require_safety_pass=True,
        tool_allowlist_consecutive_wins=2, targets=["prompts"],
    )


def test_the_prompt_no_longer_demands_a_full_file_replacement(fake_surfaces, tmp_path):
    """The literal string is the item's acceptance grep."""
    prompt = hg._build_single_variant_prompt(_cfg(tmp_path), ["prompts"],
                                             target_file_hint="SOUL.md")
    assert "COMPLETE replacement text" not in prompt
    assert "full replacement content" not in prompt
    assert '"edits"' in prompt and '"anchor"' in prompt and '"replacement"' in prompt


def test_the_prompt_states_the_exactly_once_anchor_rule(fake_surfaces, tmp_path):
    prompt = hg._build_single_variant_prompt(_cfg(tmp_path), ["prompts"],
                                             target_file_hint="SOUL.md")
    assert "EXACTLY ONCE" in prompt
    assert "EXACTLY ONE file" in prompt


def test_the_live_surfaces_yield_a_bounded_parseable_response(tmp_path, monkeypatch):
    """Same check on the real SOUL.md and MEMORY.md rather than a stand-in,
    because they are the pair whose echo produced the 178 `Unterminated string`
    dumps and the prompt must be built from them exactly as a round builds it.

    This pins the mechanism, not an arithmetic: a legal response is a small
    constant whatever the live file size, the prompt no longer asks for the
    files back, and the full-file shape a round used to have to produce is
    refused outright.
    """
    # Reached by absolute path rather than through hg.SOUL_PATH, which resolves
    # relative to the checkout — a worktree has no vault beside it, and the
    # point of this test is the real files a round actually edits.
    vault = Path.home() / "obsidian" / "lloyd"
    soul_path, memory_path = vault / "SOUL.md", vault / "MEMORY.md"
    if not (soul_path.exists() and memory_path.exists()):
        pytest.skip(f"the real prompt surfaces are not present at {vault}")

    monkeypatch.setattr(hg, "SOUL_PATH", soul_path)
    monkeypatch.setattr(hg, "MEMORY_PATH", memory_path)

    soul = soul_path.read_text(encoding="utf-8")
    memory = memory_path.read_text(encoding="utf-8")

    prompt = hg._build_single_variant_prompt(_cfg(tmp_path), ["prompts"],
                                             target_file_hint="MEMORY.md")
    assert "COMPLETE replacement text" not in prompt
    assert soul[:400] in prompt, "the prompt must show the surface being edited"

    emitted = json.dumps({
        "description": "make the block signal the whole response",
        "hypothesis": "bench_010 loses on preamble before the block signal",
        "target_surface": "prompts",
        "edits": [
            {"path": "MEMORY.md",
             "anchor": verbatim_span(memory),
             "replacement": "State the change, then the command that proves it."},
            {"path": "MEMORY.md",
             "anchor": verbatim_span(soul),
             "replacement": "Prefer a scoped change over a rewrite."},
        ],
    })
    # A legal response is a constant, not a function of the live surfaces: a
    # two-edit proposal against the real MEMORY.md costs a fraction of the
    # 8,000-token ceiling.
    assert len(emitted) < 4_096, f"legal response grew to {len(emitted)} chars"
    assert len(emitted) < hg.CEILING_CHARS
    v, err = hg._parse_single_variant(emitted)
    assert err is None and len(v["edits"]) == 2

    # The shape this round replaces — the file itself in the response — is
    # refused even at live size, so a model that ignores the new rules cannot
    # silently fall back to it.
    old_shape = json.dumps({"overlay_files": {"MEMORY.md": memory}})
    v2, err2 = hg._parse_single_variant(old_shape)
    assert v2 is None and "edits missing/empty" in err2


def test_emitted_json_still_parses_at_max_tokens_8000_against_a_20kb_memory(
    fake_surfaces, tmp_path,
):
    """#446's acceptance: a bounded edit response fits the 8,000-token ceiling
    even when the surface it edits is 20 KB+. Output length is set by the edit,
    not by the file — which is the whole mechanism."""
    _, memory = fake_surfaces
    memory_text = memory.read_text(encoding="utf-8")
    assert len(memory_text) >= 20_000, "test needs a 20 KB surface"

    emitted = json.dumps({
        "description": "cut hedging from the verification rule",
        "hypothesis": "bench_002 loses on verbose hedging",
        "target_surface": "prompts",
        "edits": [{"path": "MEMORY.md",
                   "anchor": "memory note line\nmemory note line",
                   "replacement": "Verify a state change by reading it back."}],
    })
    # The ceiling that killed the echo: 8,000 tokens at ~4 chars/token.
    assert len(emitted) < 8000 * 4, "a bounded edit response must fit the ceiling"
    v, err = hg._parse_single_variant(emitted)
    assert err is None and v["edits"][0]["path"] == "MEMORY.md"

    # A legal response is smaller than the thing it edits. Under the old
    # contract it had to be at least as big — `len(response) >= len(surface)`
    # was the definition of valid — which is the scaling that met the ceiling.
    assert len(emitted) < len(memory_text)

    # The old contract demanded the response CONTAIN the whole file, so its
    # minimum legal length grew with the surface. Grow the surface and the
    # demand becomes unsatisfiable — an echo of it is larger than 8,000 tokens
    # however terse the model tries to be.
    memory.write_text("# MEMORY\n" + "memory note line\n" * 2800, encoding="utf-8")
    unechoable = json.dumps({"overlay_files": {"MEMORY.md": memory.read_text(encoding="utf-8")}})
    assert len(unechoable) > hg.CEILING_CHARS, (
        "a surface this size cannot be returned whole under max_tokens 8000"
    )
    # Under the new contract the same growth costs nothing: the same edit is
    # byte-identical, because its size is set by the edit and not by the file.
    v2, err2 = hg._parse_single_variant(emitted)
    assert err2 is None and json.dumps(v2["edits"]) == json.dumps(v["edits"])


# ── truncation now fails the variant (#446) ──────────────────────────────────

class _FakeResp:
    def __init__(self, content, finish):
        self._content, self._finish = content, finish

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content},
                             "finish_reason": self._finish}]}


def _stub_http(monkeypatch, content, finish):
    """Patch the transport under the REAL `_call_local_llm` rather than stubbing
    the function itself, so what gets exercised is its return value."""
    import app.config

    monkeypatch.setattr(app.config, "resolve_model_alias", lambda m: m, raising=True)
    monkeypatch.setattr(app.config, "_get_model_cfg",
                        lambda m: {"base_url": "http://127.0.0.1:1"}, raising=True)
    monkeypatch.setattr(hg.requests, "post", lambda *a, **k: _FakeResp(content, finish), raising=True)


def test_the_llm_call_returns_content_payload_and_finish_reason(monkeypatch):
    """The arity is the contract between `_call_local_llm` and `_propose_one`,
    which unpacks three values out of it — and every other test here stubs this
    function out, so nothing else would notice a two-value return until a round
    died on the unpack at runtime.
    """
    _stub_http(monkeypatch, "the response text", "stop")
    content, payload, finish = hg._call_local_llm("p")
    assert content == "the response text"
    assert finish == "stop"
    assert payload["max_tokens"] == 8000, "#446 keeps the ceiling; the fix is the contract"


def test_a_truncated_response_arrives_labelled_length(monkeypatch):
    """`_propose_one` can only fail a variant on truncation if the reason
    survives the trip out of the transport."""
    _stub_http(monkeypatch, '{"edits": [{"anchor": "x', "length")
    assert hg._call_local_llm("p")[2] == "length"


def test_a_failed_call_returns_the_three_tuple_the_caller_unpacks(monkeypatch):
    """The failure path keeps the same shape — (None, payload, ""). A caller
    unpacking three values must never be the thing that breaks."""
    import app.config

    def boom(*a, **k):
        raise RuntimeError("engine down")

    monkeypatch.setattr(app.config, "resolve_model_alias", lambda m: m, raising=True)
    monkeypatch.setattr(app.config, "_get_model_cfg",
                        lambda m: {"base_url": "http://127.0.0.1:1"}, raising=True)
    monkeypatch.setattr(hg.requests, "post", boom, raising=True)
    content, payload, finish = hg._call_local_llm("p")
    assert content is None and finish == "" and payload["max_tokens"] == 8000


def _stub_call(monkeypatch, raw, finish):
    payload = {"model": "primary", "max_tokens": 8000,
               "messages": [{"role": "user", "content": "p"}]}
    monkeypatch.setattr(hg, "_call_local_llm", lambda *a, **k: (raw, payload, finish))
    return payload


def test_finish_reason_length_fails_the_variant(monkeypatch, redirect_debug_dir, tmp_path):
    """`:200` used to log 'possible truncation' and hand the truncated string to
    the parser anyway. A length-truncated response is now a rejected variant."""
    _stub_call(monkeypatch, '{"edits": [{"anchor": "' + "x" * 5000, "length")
    assert hg._propose_one(_cfg(tmp_path), ["prompts"], "primary", 0) is None
    dumps = list(redirect_debug_dir.iterdir())
    assert len(dumps) == 1
    text = dumps[0].read_text(encoding="utf-8")
    assert "finish_reason=length" in text
    assert dumps[0].name.startswith("hypothesis_reject_")


def test_a_clean_response_is_not_rejected_by_the_finish_reason_check(
    monkeypatch, redirect_debug_dir, tmp_path,
):
    _stub_call(monkeypatch, json.dumps({"edits": [edit()]}), "stop")
    v = hg._propose_one(_cfg(tmp_path), ["prompts"], "primary", 0)
    assert v is not None and v["edits"] == [edit()]
    assert not redirect_debug_dir.exists(), "a good variant must not write a diagnostic"


def test_a_parse_failure_still_dumps_under_the_hypothesis_fail_prefix(
    monkeypatch, redirect_debug_dir, tmp_path,
):
    """Parse failures keep their old dump name; deliberate rejections use
    `hypothesis_reject_*`, so a clean-rounds check on `hypothesis_fail_*` still
    means 'nothing was truncated' rather than 'nothing was rejected'."""
    _stub_call(monkeypatch, "the model wrote prose", "stop")
    assert hg._propose_one(_cfg(tmp_path), ["prompts"], "primary", 0) is None
    dumps = list(redirect_debug_dir.iterdir())
    assert len(dumps) == 1 and dumps[0].name.startswith("hypothesis_fail_")


def test_a_rejected_contract_dumps_under_the_reject_prefix(
    monkeypatch, redirect_debug_dir, tmp_path,
):
    _stub_call(monkeypatch, json.dumps({"overlay_files": {"MEMORY.md": "x" * 100}}), "stop")
    assert hg._propose_one(_cfg(tmp_path), ["prompts"], "primary", 0) is None
    dumps = list(redirect_debug_dir.iterdir())
    assert len(dumps) == 1 and dumps[0].name.startswith("hypothesis_reject_")


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
