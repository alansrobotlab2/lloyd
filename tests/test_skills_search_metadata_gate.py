"""`skills_search` must not advertise a search it does not run (backlog #4028).

The MCP registration said it "Searches skill names, descriptions, tags, and body
content". The handler does not: it scores with `_score_skill(...,
require_metadata_hit=True)`, and that gate returns `0.0` for any skill whose
`name_hits + desc_hits + tag_hits + pq_hits == 0`
(`agent_mcp/skills.py`, `_score_skill`). Body prose therefore contributes score
only *after* metadata already matched, so a body-only match is unreachable by
this tool — measured 2026-11-09 over the live corpus: of 862 tokens that occur in
some skill's body and in none of that skill's name/description/tags, the 80
queried one at a time through `_skills_search` returned `total == 0` for 75, and
the other 5 returned only *other* skills whose metadata matched (the body-owning
skill absent every time).

Why the wording is not a cosmetic problem: this is the tool a caller reaches for
precisely when the injected index did not surface something. A caller that
believed "body content" reads a null result as *no skill covers this task*, when
the only thing the number shows is that the phrasing missed the metadata.

Triage took option 1 — make the description tell the truth and say so in the
payload — and rejected option 2 (score bodies when metadata misses) on
measurement: over the 51 labelled turns of `eval/skill_match_queries.yaml` the
count of the 208 skills scoring `> 0` is mean 39 / median 29 with the gate on and
mean 147 / median 155 / max 207 — nearly the whole corpus — with it off, which is
the "12× powerpoint on graph-classifier work" failure the gate exists for and
would re-open #311. So the last test below pins the gate itself unchanged: the
description moved to the code, not the code to the description, and that is why
the recall numbers cannot move under this change.

Seams crossed: the registered `Tool.description` reaches a caller through
`agent_mcp.main.list_tools()` over MCP, and the payload reaches it through
`agent_mcp.main.call_tool` — both are driven here through the aggregator, not
just the module.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_mcp import main as M
from agent_mcp import skills as S

#: A token no skill carries in either metadata or body, so the live-corpus
#: search below cannot hit and must explain itself.
_UNKNOWABLE = "sesquipedalian"

#: A token planted in a synthetic skill's BODY and nowhere in its metadata — the
#: exact shape the description used to promise and the scorer cannot deliver.
_BODY_ONLY = "brontopsism"


def _description(name: str) -> str:
    """The description of a served tool, read from the live registry."""
    tools = asyncio.run(M.list_tools())
    return next(t.description for t in tools if t.name == name)


def _search_payload(name: str, arguments: dict) -> dict:
    """Dispatch through the aggregator and parse what the caller would parse."""
    result = asyncio.run(M.call_tool(name, arguments))
    blocks = result.content if hasattr(result, "content") else result
    return json.loads(blocks[0].text)


def _skill(name: str, description: str, tags: list[str], body: str) -> dict:
    """A loaded skill shaped as `_load_skill` returns it."""
    return {
        "name": name,
        "description": description,
        "category": "testing",
        "tags": tags,
        "body": body,
        "raw": "",
        "path": None,
    }


@pytest.fixture
def one_body_only_skill(monkeypatch):
    """A corpus of one: `body` carries _BODY_ONLY, the metadata does not."""
    skill = _skill(
        name="bronto-thermostat",
        description="Calibrate the workshop PID controller",
        tags=["workshop"],
        body=f"Turn the {_BODY_ONLY} screw a quarter turn clockwise and wait.",
    )
    monkeypatch.setattr(S, "_iter_skills", lambda *a, **k: iter([skill]))
    return skill


# ── Clause 1: the registered description tells the truth ──────────────────────


def test_the_registered_description_no_longer_claims_body_search():
    """The phrase that lied is gone, from the surface callers actually read."""
    desc = _description("skills_search")
    assert "body content" not in desc, desc
    # The old claim's shape, not just its words: a clause that puts the body
    # beside name/description/tags as something the query is matched against.
    assert "tags, and body" not in desc, desc
    assert "and body content" not in desc, desc


def test_the_registered_description_states_the_metadata_requirement():
    """Correcting a lie by omission is the same defect in a new outfit: a
    description that merely drops the word "body" still leaves a caller unable
    to tell a null search from an empty library. It has to carry the gate — and
    carry it completely. `_score_skill` admits a skill on `name_hits + desc_hits
    + tag_hits + pq_hits`, and `_pseudo_query_token_sets` deliberately subtracts
    the skill's own metadata from the cached queries, so a fourth qualifier
    exists by construction and an enumeration that stops at the three
    human-readable fields is a third wording misdescribing what qualifies."""
    desc = _description("skills_search")
    assert "metadata" in desc.lower(), desc
    assert "only metadata qualifies" in desc.lower(), desc
    for field in ("name", "description", "tags", "indexed queries"):
        assert field in desc, f"{field!r} not named: {desc}"
    # The pointer to the payload field, so a caller who only reads the catalog
    # still learns a null needs interpreting.
    assert "zero_result_reason" in desc, desc


def test_a_body_prose_mention_is_always_inside_a_qualifier():
    """Every surviving mention of the body must be the qualifier, never a
    promise: "body" may only appear next to prose/excerpt language, and the
    sentence that put it beside name/description/tags is gone."""
    desc = _description("skills_search")
    assert "body prose" in desc and "excerpts" in desc, desc
    assert "searches skill names" not in desc.lower(), desc


def test_the_correction_cost_the_catalog_nothing():
    """`tests/test_mcp_layer.py::test_advertised_catalog_stays_under_its_token_ceiling`
    ships with 12 estimated tokens of room over 105 tools (#1555 records why that
    is a human call, not a per-round one). A description rewrite that crossed it
    would spend a shared budget the item never asked to spend, so the honest
    wording was measured down to below the old string's cost: 215 chars /
    22,488 estimated catalog tokens before, 204 chars / 22,487 after (the
    estimator and the advertised-tool list are the ones
    `test_advertised_catalog_stays_under_its_token_ceiling` builds). This node
    keeps the honesty from being bought back later as a ceiling raise."""
    desc = _description("skills_search")
    assert len(desc) <= 215, f"description grew to {len(desc)} chars; the catalog has no room"


# ── Clause 2: a body-only query is never a silent miss ────────────────────────


def test_a_query_matching_only_body_prose_returns_nothing_with_a_reason(
        one_body_only_skill):
    """Acceptance #2's "nothing-with-an-explained-reason" branch, pinned on a
    corpus of one so the result cannot be an accident of what is installed."""
    payload = json.loads(S._skills_search({"query": _BODY_ONLY}))
    assert payload["total"] == 0, payload
    assert payload["results"] == [], payload
    reason = payload["zero_result_reason"]
    assert "metadata" in reason.lower(), reason
    # Names the requirement AND the way out, or the caller is back to guessing.
    assert "name" in reason and "description" in reason, reason
    assert "never qualifies" in reason, reason


def test_the_reason_appears_only_when_the_search_came_back_empty(
        one_body_only_skill):
    """A hit must not be padded with a paragraph about why it might be missing —
    and the metadata path still returns the skill, so the gate is a qualifier
    rule, not a broken search."""
    hit = json.loads(S._skills_search({"query": "calibrate workshop controller"}))
    assert hit["total"] == 1, hit
    assert hit["results"][0]["name"] == one_body_only_skill["name"]
    assert "zero_result_reason" not in hit, hit


def test_a_zero_from_the_live_corpus_explains_itself():
    """The same field on the real corpus, through the handler the MCP process
    runs — the case in the item: `_skills_search({"query": "sesquipedalian"})`
    used to answer `{"query", "results", "total"}` and nothing else."""
    payload = json.loads(S._skills_search({"query": _UNKNOWABLE}))
    assert payload["total"] == 0, payload
    assert "zero_result_reason" in payload, payload
    assert "metadata" in payload["zero_result_reason"].lower()


def test_the_mcp_seam_carries_the_reason_and_the_honest_description():
    """The process boundary: the caller is a different process reading
    `tools/list` and `tools/call`, so a field added to the handler's dict is
    worth nothing until the aggregator serves it. Reads both halves of the
    surface through `agent_mcp.main`."""
    payload = _search_payload("skills_search", {"query": _UNKNOWABLE})
    assert payload["total"] == 0, payload
    assert "metadata" in payload["zero_result_reason"].lower(), payload

    served = _description("skills_search")
    module = next(t for t in asyncio.run(S.list_tools()) if t.name == "skills_search")
    assert served == module.description, (
        "the aggregator serves a different description than the module "
        "registers, so the test above graded a string no caller sees"
    )
    assert "body content" not in served


# ── Option 1, not option 2: the scorer is unchanged, so recall cannot move ────


def test_the_metadata_gate_the_description_now_admits_is_still_the_shipped_one(
        one_body_only_skill):
    """If `require_metadata_hit` were relaxed, this tool would start surfacing
    body-only skills and the description's new wording would become the lie —
    and the offer set would jump to nearly the whole corpus (mean 147 of 208 per
    turn on the eval set versus mean 39 gated), re-opening #311. So the gate
    stays, and the description is what moved to meet it: the offline recall
    numbers (hit@1 0.5098, hit@5 0.7255 measured 2026-11-09) are evidence for
    this round only in that they do not move."""
    tokens = S._query_tokens(_BODY_ONLY)
    gated = S._score_skill(one_body_only_skill, tokens, require_metadata_hit=True)
    ungated = S._score_skill(one_body_only_skill, tokens, require_metadata_hit=False)
    assert gated == 0.0, gated
    assert ungated > 0.0, ungated
    # And the handler is the gated path, not the ungated one.
    assert json.loads(S._skills_search({"query": _BODY_ONLY}))["total"] == 0
