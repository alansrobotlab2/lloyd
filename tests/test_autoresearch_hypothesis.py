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

import ast
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import time
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


# `live_vault`, added for #797: this reads ~/obsidian/lloyd/SOUL.md and
# MEMORY.md as they are right now, which no round under test controls. The
# absolute path below is reached precisely so a worktree cannot dodge it, so it
# still resolves when the gate runs its `tests` rung at `cwd=self.worktree` —
# which is where a round that opened no prompt file used to be judged on it. The
# hazard is the anchor: `verbatim_span` accepts only a single line of 20-160
# chars, and this file is rewritten by the nightly reflection job. The count of
# qualifying lines in `lloyd/MEMORY.md` has read 1 at a vault commit and 22 at
# today's, and at 0 the anchor is `""`, `_parse_single_variant` refuses it (that
# refusal is pinned by `test_an_empty_anchor_is_rejected`), and a hard promotion
# rung fails on a prose rewrap. The gate runs `-m "not live_vault"`; the nightly
# `live_vault` rung that `skills/nightly-skills-management/SKILL.md` runs still
# runs this node, and `tests/test_prompt_surface_guard.py` pins that it runs
# green there. What stayed on the hard rung: the tmp-surface node below, which
# demonstrates the same bounded-edit-response contract against a surface the test
# writes itself, plus `test_verbatim_span_bounds_a_single_line_of_20_to_160_chars`
# and the two structural pins after this node. Assertions unchanged — only where
# this runs moved.
@pytest.mark.live_vault
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


# ── what the `live_vault` mark above leaves on the hard rung (#797) ──────────
#
# The mark moves one node off the gate's `tests` rung. These pins keep that
# honest without reading the vault, so they run under `-m "not live_vault"` on
# every rung themselves: the bound the moved node's anchor is drawn from, the
# structure of what stayed behind, and the gate's own selection over this file.

LIVE_SURFACES_NODE = "test_the_live_surfaces_yield_a_bounded_parseable_response"
TMP_SURFACE_NODE = "test_emitted_json_still_parses_at_max_tokens_8000_against_a_20kb_memory"
# Root-relative, because that is the only form pytest resolves from `cwd=root`.
REPO_ROOT = Path(__file__).resolve().parent.parent
THIS_FILE = str(Path(__file__).resolve().relative_to(REPO_ROOT))

# The six assertions the moved node carried at base 8161a4e8, captured with
# `ast.unparse`. This is what "no assertion was relaxed" means exactly: relaxing
# one has to delete its entry here as well, in the same commit, in front of a
# reader.
LIVE_SURFACES_ASSERTIONS = [
    "assert 'COMPLETE replacement text' not in prompt",
    "assert soul[:400] in prompt, 'the prompt must show the surface being edited'",
    "assert len(emitted) < 4096, f'legal response grew to {len(emitted)} chars'",
    "assert len(emitted) < hg.CEILING_CHARS",
    "assert err is None and len(v['edits']) == 2",
    "assert v2 is None and 'edits missing/empty' in err2",
]


def _top_level_functions() -> dict:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _normalized_asserts(nodes) -> list[str]:
    """Asserts as text, whitespace- and quote-normalised.

    Normalising makes this a check on *what* is asserted rather than on how
    CPython formatted it that release: `ast.unparse` picks whichever quote style
    does not clash with the literal's own content, and that is free to change
    between versions while the assertion underneath does not.
    """
    return sorted(re.sub(r"['\"]", "'", re.sub(r"\s+", " ", ast.unparse(n)))
                  for n in nodes)


def test_verbatim_span_bounds_a_single_line_of_20_to_160_chars():
    """The window the moved node's anchor comes from, pinned with no vault in it.

    `verbatim_span` is the only reason that node is vault-sensitive at all: it
    answers with the longest single line that is at least 20 and at most 160
    characters, so a file whose lines all run longer than 160 — which is how the
    nightly reflection job writes `lloyd/MEMORY.md` — yields the empty string, and
    an empty anchor is a refused variant. Both bounds are pinned here against
    synthetic text, so the bound itself stays enforced on a hermetic rung while
    the number of live lines that happen to satisfy it today is reporting, which
    is what the `live_vault` group is for.
    """
    assert verbatim_span("x" * 19) == "", "19 chars is under the floor"
    assert verbatim_span("x" * 20) == "x" * 20, "20 chars is the floor"
    assert verbatim_span("x" * 160) == "x" * 160, "160 chars is the ceiling"
    assert verbatim_span("x" * 161) == "", "161 chars is over the ceiling"
    assert verbatim_span("z" * 4_000) == "", "one unwrapped paragraph yields no anchor"
    # Inside the window the answer is the longest single line, never two joined.
    assert verbatim_span("a" * 30 + "\n" + "b" * 160 + "\n" + "c" * 40) == "b" * 160
    # Surrounding whitespace is not part of the span: a real anchor has to be
    # copyable out of the prompt character-for-character.
    assert verbatim_span("   " + "d" * 25 + "  ") == "d" * 25
    assert inspect.signature(verbatim_span).parameters["max_chars"].default == 160


def test_marking_the_live_surfaces_node_left_the_hard_rung_enforcement_intact():
    """Marking a node is allowed; leaving the mechanism unpinned on a code change is not.

    #797's own instruction — move where the live-vault check runs, keep what it
    checks. So the pin is structural. `TMP_SURFACE_NODE` demonstrates the same
    bounded-edit-response contract against a 20 KB surface the test writes itself
    and carries no marker of any kind, which is what keeps the mechanism on the
    gate's `-m "not live_vault"` rung; and the marked node still asserts the six
    checks it asserted before it was marked, with the mark as its only decorator —
    no `skip`, no `xfail`, no `skipif` standing in for the deletion the nightly
    `live_vault` rung forbids.
    """
    fns = _top_level_functions()
    assert fns[TMP_SURFACE_NODE].decorator_list == [], (
        f"{TMP_SURFACE_NODE} gained a decorator: with the live node deselected, the "
        "bounded-edit-response contract would have no enforcement point on a code change"
    )
    assert [ast.unparse(d) for d in fns[LIVE_SURFACES_NODE].decorator_list] == [
        "pytest.mark.live_vault"
    ], "the moved node must carry exactly the mark and nothing that would skip it"
    expected = _normalized_asserts([ast.parse(a).body[0]
                                    for a in LIVE_SURFACES_ASSERTIONS])
    actual = _normalized_asserts(n for n in ast.walk(fns[LIVE_SURFACES_NODE])
                                 if isinstance(n, ast.Assert))
    assert actual == expected, (
        "the assertions of the node moved off the hard rung changed; #797 moves the "
        "mark and nothing else — restore them, or change LIVE_SURFACES_ASSERTIONS "
        "with the reason in the commit message"
    )


def test_the_gate_selection_over_this_file_keeps_the_tmp_surface_node():
    """Clause 3 across the seam that matters: pytest's own selection, not a grep.

    The gate's `tests` rung builds an argv, hands it to a child interpreter and
    reads an exit code, so that is what gets asserted here — `-m "not live_vault"`
    over this file must still collect the tmp-surface node and run it green. A
    grep would prove the decorator was typed; this proves the deselection and the
    surviving enforcement point, and it fails if either node's marker moves.
    """
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--collect-only", "-m", "not live_vault", THIS_FILE],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    listed = collect.stdout + collect.stderr
    assert f"::{TMP_SURFACE_NODE}" in listed, (
        "the tmp-surface node left the hard rung:\n" + listed[-1200:])
    assert f"::{LIVE_SURFACES_NODE}" not in listed, (
        "the live-vault node is back on the hard rung:\n" + listed[-1200:])

    ran = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-m", "not live_vault", f"{THIS_FILE}::{TMP_SURFACE_NODE}"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert ran.returncode == 0 and "1 passed" in ran.stdout + ran.stderr, (
        "the surviving enforcement point is not passing on the hard rung:\n"
        + (ran.stdout + ran.stderr)[-1200:])


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


# ── #680: MEMORY.md is shown in full, so an anchor can come from anywhere ─────
#
# `_apply_anchored_edits` refuses an anchor that does not match the canonical
# text exactly once, and the only text the model is given is what this prompt
# shows. MEMORY.md used to arrive as `_read(MEMORY_PATH, tail=4000)` — its newest
# 4,000 chars and nothing else, out of a live file several times that long — while
# `_propose_one` seeded half of every round's variants to that file. The tail
# existed to keep the RESPONSE small; the response has been bounded by
# MAX_EDITS/MAX_ANCHOR_CHARS/MAX_REPLACEMENT_CHARS since #446, so the tail only
# ever cost reach.

#: The pre-fix window, spelled out so a reader can see the size being replaced.
OLD_TAIL_CHARS = 4_000

#: Tokens of context on the smallest engine the generator may be pointed at: the
#: primary vLLM serve, `agent-services/bin/start-27b-nvfp4-sakamakismile-mtp-tuned.sh`
#: `--max-model-len 131072` (its un-tuned sibling runs 262144). `_call_local_llm`
#: POSTs to the configured `base_url`, so a round shares this window with
#: interactive traffic and up to `max_variants_per_round` parallel seeds.
SMALLEST_ENGINE_CONTEXT = 131_072


@pytest.fixture
def tall_memory_with_a_leading_sentinel(tmp_path, monkeypatch):
    """A MEMORY.md bigger than the old 4,000-char window whose FIRST line is a
    sentinel. Under the tail read the sentinel is the oldest content in the file
    and cannot appear in the prompt; under a full read it must."""
    sentinel = "SENTINEL-680-OLDEST-NOTE: prefer the scoped change over the rewrite."
    memory = tmp_path / "MEMORY.md"
    body = "# MEMORY\n" + sentinel + "\n" + ("an older infrastructure note\n" * 800)
    assert len(body) > 4 * OLD_TAIL_CHARS, "the file must be well past the old window"
    memory.write_text(body, encoding="utf-8")
    soul = tmp_path / "SOUL.md"
    soul.write_text("# SOUL\n" + "soul prose line\n" * 40, encoding="utf-8")
    monkeypatch.setattr(hg, "SOUL_PATH", soul)
    monkeypatch.setattr(hg, "MEMORY_PATH", memory)
    monkeypatch.setattr(hg, "USER_PATH", tmp_path / "absent.md")
    monkeypatch.setattr(hg, "CORRECTIONS_PATH", tmp_path / "absent2.md")
    monkeypatch.setattr(hg, "KNOWLEDGE_HEALTH_PATH", tmp_path / "absent3.md")
    return memory, sentinel


def test_the_memory_prompt_shows_the_file_in_full_not_a_tail(
        tall_memory_with_a_leading_sentinel, tmp_path):
    """Clause 1. A sentinel in the first 1,000 chars of a MEMORY.md 5.8x the old
    window (23,278 fixture chars vs the 4,000 shown; the live vault file was
    33,704 when this round's acceptance probe ran) reaches the rendered prompt,
    and the section stops calling the text a tail.

    Against the pre-fix read this fails twice over: the sentinel sits outside the
    last 4,000 chars, and the heading literally says `tail`.
    """
    memory, sentinel = tall_memory_with_a_leading_sentinel
    memory_text = memory.read_text(encoding="utf-8")
    # The premise as data, not as fixture decoration: this text is inside the
    # window the generator used to show and outside it, so the prompt assertions
    # below are the only thing that can tell the two reads apart.
    assert sentinel not in memory_text[-OLD_TAIL_CHARS:], (
        "the sentinel must be in the region the pre-fix `tail=4000` read dropped; "
        "if this ever flips, this test stops testing anything"
    )

    assert hg._read(memory) == memory_text, "the no-tail read returns the whole file"
    assert hg._read(memory, tail=OLD_TAIL_CHARS) == memory_text[-OLD_TAIL_CHARS:]

    prompt = hg._build_single_variant_prompt(_cfg(tmp_path), ["prompts"],
                                             target_file_hint="MEMORY.md")
    assert sentinel in prompt
    assert memory_text in prompt, "the whole canonical surface is what gets shown"
    assert "Current MEMORY.md tail" not in prompt
    assert "Current MEMORY.md (long-term notes, shown in full)" in prompt
    # USER.md and the signal files are still excerpts — they are context, not a
    # surface an anchor may quote, so shrinking them costs nothing and this round
    # does not widen them.
    assert "## Current USER.md tail" in prompt


def _memory_section(prompt: str) -> str:
    """The text of the prompt's MEMORY.md block — the only thing a model may copy.

    Spelled out as a slice rather than `assert span in prompt` because the
    response is told to quote from THIS block: a span that merely appears
    somewhere in the prompt (in an instruction, in USER.md's excerpt) would not
    be copyable in the way the anchor rule means.
    """
    heading = "## Current MEMORY.md (long-term notes, shown in full)\n"
    start = prompt.index(heading) + len(heading)
    end = prompt.index("\n## ", start)
    return prompt[start:end]


def test_an_anchor_copied_out_of_the_rendered_prompt_survives_the_parser(
        tall_memory_with_a_leading_sentinel, tmp_path):
    """The seam, run as one chain: render the prompt, copy the oldest span out of
    the rendered text, hand that copy to the parser.

    The copy is what makes this the real mechanism rather than a re-assertion of
    the fixture. A model's anchor is a `str` sliced out of the prompt string, so
    the anchor here comes from `prompt[...]` and must survive parsing byte for
    byte. Under the pre-fix `tail=4000` read this fails: the block is the last
    4,000 chars, its first line is one of the middle filler notes, and the
    sentinel — the first line of the file, absent from the last 4,000 — never
    appears in the prompt at all.
    """
    prompt = hg._build_single_variant_prompt(_cfg(tmp_path), ["prompts"],
                                             target_file_hint="MEMORY.md")
    block = _memory_section(prompt)
    copied = next(line for line in block.splitlines() if "SENTINEL-680" in line)
    assert copied == tall_memory_with_a_leading_sentinel[1], (
        "what the model sees and what it would copy are byte-identical"
    )

    raw = json.dumps({
        "description": "tighten the leading rule",
        "hypothesis": "the oldest note is the one that still causes preamble",
        "target_surface": "prompts",
        "edits": [{"path": "MEMORY.md", "anchor": copied,
                   "replacement": "Prefer a scoped change over a rewrite."}],
    })
    v, err = hg._parse_single_variant(raw)
    assert err is None, f"an anchor quoted from the prompt is in contract: {err}"
    assert len(v["edits"]) == 1
    assert v["edits"][0]["anchor"] == copied
    # The rendered prompt is what grew, and this threshold is what says so: the
    # same fixture rendered ~9,000 chars pre-fix, when the block was the last
    # 4,000 of a 23,278-char file. #446's claim that the RESPONSE stays a
    # constant whatever the file size is pinned where it is actually a source
    # claim — in test_the_bounded_response_contract_holds_while_the_shown_surface_grows
    # below, off the payload and the three edit bounds — not here, where a
    # response-size assertion could only ever be a property of this test body.
    assert len(prompt) > 11_000, "the prompt got the whole surface, not a summary of it"


def test_the_bounded_response_contract_holds_while_the_shown_surface_grows(
        tall_memory_with_a_leading_sentinel, tmp_path, monkeypatch):
    """Clause 4: what is shown grows, what a response may be does not.

    The three bounds and the engine's 8,000-token ceiling are #446's whole fix;
    this pins them at their shipped values and shows the arithmetic that keeps a
    worst-case legal response inside the ceiling — MAX_EDITS x (MAX_ANCHOR_CHARS
    + MAX_REPLACEMENT_CHARS) = 19,200 chars against a 32,000-char ceiling.
    """
    assert hg.MAX_EDITS == 6
    assert hg.MAX_ANCHOR_CHARS == 1_200
    assert hg.MAX_REPLACEMENT_CHARS == 2_000
    assert hg.MAX_EDITS * (hg.MAX_ANCHOR_CHARS + hg.MAX_REPLACEMENT_CHARS) < hg.CEILING_CHARS

    # On the stubbed transport, so this reads the payload a real call would send
    # without reaching the engine a round shares with interactive traffic.
    _stub_http(monkeypatch, "ok", "stop")
    _content, payload, _finish = hg._call_local_llm("prompt under test")
    assert payload["max_tokens"] == 8000, (
        "showing more of the file must not raise the response ceiling; the ceiling "
        "is what makes a bigger prompt affordable"
    )

    memory, _ = tall_memory_with_a_leading_sentinel
    big = hg._build_single_variant_prompt(_cfg(tmp_path), ["prompts"],
                                          target_file_hint="MEMORY.md")
    memory.write_text("# MEMORY\nshort file\n", encoding="utf-8")
    small = hg._build_single_variant_prompt(_cfg(tmp_path), ["prompts"],
                                            target_file_hint="MEMORY.md")
    assert len(big) > len(small), "the prompt carries the growth"

    # The growth has to stay affordable, measured not narrated. At ~4 chars per
    # token, a whole-surface prompt plus a worst-case legal response costs
    # (len(big) + CEILING_CHARS) / 4 tokens against SMALLEST_ENGINE_CONTEXT — the
    # smallest `--max-model-len` any engine on this box runs with — so this fails
    # on the source the day CEILING_CHARS rises, a second surface is shown whole,
    # or MEMORY.md grows past the headroom. The fixture is deliberately a few times
    # the live file's length, so the headroom it leaves is the number that matters,
    # not the fixture's own size.
    headroom = SMALLEST_ENGINE_CONTEXT - (len(big) + hg.CEILING_CHARS) / 4
    assert headroom > 100_000, (
        f"the shown surface is within {(len(big) + hg.CEILING_CHARS) / 4:.0f} tokens "
        f"of the {SMALLEST_ENGINE_CONTEXT}-token engine context; raising "
        "MAX_* or showing another surface whole has eaten the headroom this round "
        "bought, and the item's fallback (a size-triggered section outline) is now "
        "the live problem rather than a future one"
    )


# ── the dump pile is bounded per prefix (#681) ───────────────────────────────



def _seed(d, prefix, n, base):
    """n dumps of one prefix, one minute apart, oldest first from `base`."""
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"{prefix}seed{i:03d}.txt"
        p.write_text("x")
        t = base + 60 * i
        os.utime(p, (t, t))
        out.append(p)
    return out


def _round(rounds, when):
    name = time.strftime("R_%Y%m%d_%H%M%S", time.gmtime(when))
    (rounds / name).mkdir(parents=True)


def test_prune_keeps_the_newest_fifty_per_prefix(redirect_debug_dir, tmp_path):
    base = time.time() - 30 * 86400
    fails = _seed(redirect_debug_dir, hg.FAIL_DUMP_PREFIX, 60, base)
    rejects = _seed(redirect_debug_dir, hg.REJECT_DUMP_PREFIX, 60, base)
    rounds = tmp_path / "rounds"
    _round(rounds, base + 86400 * 20)          # a round after every seeded dump
    for prefix in (hg.FAIL_DUMP_PREFIX, hg.REJECT_DUMP_PREFIX):
        assert hg.prune_debug_dumps(prefix, rounds_dir=rounds) == 10
    assert {p for p in fails if p.exists()} == set(fails[10:])
    assert {p for p in rejects if p.exists()} == set(rejects[10:])


def test_prune_never_deletes_a_dump_newer_than_the_newest_round(
        redirect_debug_dir, tmp_path):
    base = time.time() - 30 * 86400
    fails = _seed(redirect_debug_dir, hg.FAIL_DUMP_PREFIX, 60, base)
    rounds = tmp_path / "rounds"
    _round(rounds, base - 3600)                 # an older round, ignored
    _round(rounds, base + 60 * 5)               # newest round starts at dump #5
    assert hg.prune_debug_dumps(hg.FAIL_DUMP_PREFIX, rounds_dir=rounds) == 5
    assert all(p.exists() for p in fails[5:])
    assert not any(p.exists() for p in fails[:5])


def test_the_51st_dump_bounds_its_own_prefix_only(redirect_debug_dir, tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(hg, "ROUNDS_DIR", tmp_path / "no-rounds-yet")
    base = time.time() - 30 * 86400
    _seed(redirect_debug_dir, hg.FAIL_DUMP_PREFIX, 50, base)
    rejects = _seed(redirect_debug_dir, hg.REJECT_DUMP_PREFIX, 70, base)
    hg._dump_raw_on_failure("v1", {"messages": []}, "raw", "bad comma")
    assert len(list(redirect_debug_dir.glob(f"{hg.FAIL_DUMP_PREFIX}*"))) == 50
    assert all(p.exists() for p in rejects)


def test_prune_under_the_bound_or_on_no_dir_is_a_no_op(redirect_debug_dir, tmp_path):
    rounds = tmp_path / "rounds"
    assert hg.prune_debug_dumps(hg.FAIL_DUMP_PREFIX, rounds_dir=rounds) == 0
    assert not redirect_debug_dir.exists()
    fails = _seed(redirect_debug_dir, hg.FAIL_DUMP_PREFIX, 50, time.time() - 86400)
    assert hg.prune_debug_dumps(hg.FAIL_DUMP_PREFIX, rounds_dir=rounds) == 0
    assert all(p.exists() for p in fails)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert hg.prune_debug_dumps(hg.FAIL_DUMP_PREFIX, debug_dir=empty, rounds_dir=rounds) == 0
    assert empty.is_dir()


def test_a_prune_logs_prefix_count_and_newest_retained(redirect_debug_dir, tmp_path,
                                                       caplog):
    base = time.time() - 30 * 86400
    _seed(redirect_debug_dir, hg.REJECT_DUMP_PREFIX, 53, base)
    with caplog.at_level(logging.INFO, logger="autoresearch.hypothesis"):
        hg.prune_debug_dumps(hg.REJECT_DUMP_PREFIX, rounds_dir=tmp_path / "rounds")
    msgs = [r.getMessage() for r in caplog.records if r.name == "autoresearch.hypothesis"]
    newest = time.strftime("%Y-%m-%dT%H:%M", time.gmtime(base + 60 * 52))
    assert any("pruned 3 hypothesis_reject_*" in m and newest in m for m in msgs), msgs
