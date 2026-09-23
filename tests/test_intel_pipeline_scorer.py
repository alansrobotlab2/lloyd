"""Backlog #570 — the intelligence pipeline must have a real scorer.

Found by autonomy task #30 on 2026-09-09 and confirmed at triage on 2026-09-11:
`stage2_score()` derived relevance from `max_weight * 10` with every weight
hardcoded to 1.0, so the 1-10 scale was mathematically binary — measured
{1:68,10:13} / {1:90,10:13} / {1:16,10:5} over three days. `stage1_filter()`
kept any item carrying `source_tags`, which both scanners attach to every item,
so it passed 103/103. The vault writer had no relevance floor, which is how
`knowledge/feeds/youtube-uncategorized.md` reached 2,545 sections of 1/10 noise.
The GitHub scanner had no token and no 403 branch, so a rate limit printed an
error and yielded 0 items — identical in the run report to a quiet day.

Each test pins one acceptance clause. Nothing here touches the real vault or
`_pipeline`: every module holds its paths as module-level names resolved from
`Path.home()` at import, so the subprocess tests redirect HOME and
`redirect_paths` rebinds the in-process ones.

Two assertions were tightened after the promotion review of SM_20260911_132720:
a call site is proven by the request it puts on the wire, not by grepping the
module's text (a docstring satisfies a grep), and a fallback assertion names the
number it expects instead of a range it could never leave.

Note on the `^## ` counters: `write_item_to_vault` appends one `## <date>` block
*per item*, which is why the noise file reached 2,545 sections in ~5 months and
why counting those headings is the clause's growth measure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INTEL_DIR = REPO_ROOT / "scripts" / "intel-pipeline"
if str(INTEL_DIR) not in sys.path:
    sys.path.insert(0, str(INTEL_DIR))

from intel_pipeline import models as models_mod  # noqa: E402
from intel_pipeline import profile as profile_mod  # noqa: E402
from intel_pipeline import scoring as scoring_mod  # noqa: E402
from intel_pipeline import state as state_mod  # noqa: E402
from intel_pipeline import vault_writer as vw_mod  # noqa: E402
from intel_pipeline.models import FeedItem, ScoredItem  # noqa: E402
from intel_pipeline.scanners import github_scanner as gh_mod  # noqa: E402
from intel_pipeline.scanners import youtube_scanner as yt_mod  # noqa: E402

# Keyword fallback for a topic at weight 0.9: int(round(0.9 * 10)). Tests name
# this number so "the model was consulted" and "the model was ignored" cannot
# both satisfy the same assertion.
KW_SCORE_09 = 9


@pytest.fixture
def redirect_paths(tmp_path, monkeypatch):
    """Point every package module at tmp_path instead of the live vault/_pipeline.

    `_paths` derives from Path.home() at import time and each module imported the
    result by name, so rebinding the per-module names is what actually moves them.
    """
    vault = tmp_path / "obsidian"
    feeds = tmp_path / "lloyd" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    (vault / "knowledge").mkdir(parents=True)

    monkeypatch.setattr(state_mod, "RAW_DIR", feeds / "raw")
    monkeypatch.setattr(state_mod, "STATE_FILE", feeds / "scanner-state.json")
    monkeypatch.setattr(vw_mod, "SCORED_FEEDS_DIR", feeds)
    monkeypatch.setattr(vw_mod, "VAULT_WRITTEN_STATE", feeds / "vault-written.json")
    monkeypatch.setattr(vw_mod, "KNOWLEDGE_DIR", vault / "knowledge")
    monkeypatch.setattr(vw_mod, "VAULT_ROOT", vault)
    monkeypatch.setattr(profile_mod, "PROFILE_FILE", vault / "interests.md")
    monkeypatch.delenv("INTEL_DISABLE_LLM", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(gh_mod, "CONFIG_PATH", tmp_path / "absent-github-config.yml")
    return tmp_path


# A profile in the shape the loader is supposed to read. Weights sit deliberately
# on both sides of the 0.3 stage-2 threshold (0.2 below; 0.4 and 0.9 above),
# because that band is a decision rather than a switch — and interests.md ships no
# weights at all, so this fixture is the only place the band exists to test.
PROFILE_MD = """---
title: Interests
---
# Interests

## Robotics
**Weight:** 0.7
**Projects:** Alfie
humanoid, actuator, gripper

## AI & LLMs
**Weight:** 0.9
**Projects:** Lloyd
**Keywords:** agent, llm, mcp, vllm

## Wearables
**Weight:** 0.2
**Keywords:** imu, wrist

## Flight controllers
**Weight:** 0.4
**Keywords:** px4, betaflight
"""

TOPIC_WEIGHTS = {"robotics": 0.7, "ai-llms": 0.9, "wearables": 0.2,
                 "flight-controllers": 0.4}


def _profile(tmp_path):
    (tmp_path / "obsidian" / "interests.md").write_text(PROFILE_MD)
    return profile_mod.load_profile()


def _item(item_id="i1", source="youtube", title="", summary="", tags=()):
    return FeedItem(
        id=item_id,
        source=source,
        title=title,
        url=f"https://example.com/{item_id}",
        summary=summary,
        discovered_at="2026-09-11T00:00:00Z",
        authors=[],
        source_tags=list(tags),
    )


class RecordingLLM:
    """Stand-in for the local model. Records every prompt it was asked to score."""

    def __init__(self, relevance=6, projects=("Lloyd",), category="ai-llms"):
        self.relevance = relevance
        self.projects = list(projects)
        self.category = category
        self.asked = []

    def __call__(self, prompt):
        self.asked.append(prompt)
        return json.dumps({
            "relevance": self.relevance,
            "urgency": "morning",
            "why": "Directly relevant to the agent-architecture thread",
            "projects": self.projects,
            "category": self.category,
        })


def _reply(**fields):
    return lambda prompt: json.dumps(fields)


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- clause 1: the scorer really calls the model, and only above 0.3 ---------

def test_call_local_llm_posts_to_the_local_chat_completions_endpoint(monkeypatch):
    """The call site is proven by the request it puts on the wire.

    #570's own check is a grep over scoring.py, and a grep is exactly what a
    docstring can fake — so this asserts URL, method, payload and reply handling.
    """
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode())
        seen["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": '{"relevance": 5}'}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = scoring_mod.call_local_llm("grade this item")

    assert out == '{"relevance": 5}'
    assert seen["method"] == "POST"
    assert "/v1/chat/completions" in seen["url"]
    assert "localhost" in seen["url"], "the scorer must aim at the local engine"
    assert seen["body"]["model"], "no model named means no model answers"
    assert seen["body"]["messages"][1]["content"] == "grade this item"
    assert seen["timeout"] == scoring_mod.LLM_TIMEOUT_SECONDS


def test_call_local_llm_raises_on_an_empty_reply(monkeypatch):
    """A silent model must not read as a quiet day — that is defect 5's shape."""
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: FakeResponse(
            {"choices": [{"message": {"content": "   "}}]}))

    with pytest.raises(RuntimeError):
        scoring_mod.call_local_llm("grade this")


def test_call_local_llm_raises_when_the_endpoint_is_down(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(urllib.error.URLError):
        scoring_mod.call_local_llm("grade this")


# Recorded from a real POST to the live engine at :8096 on 2026-09-11T14:31Z
# while verifying this item, not written by hand. The *keys* are the live
# response's keys verbatim; only the two large arrays (`prompt_token_ids`,
# `metrics`) are shortened, since a client that reads `content` must be
# indifferent to their size. This fixture exists because the seam a promotion
# review named is exactly this one: a hand-built
# `{"choices":[{"message":{"content": ...}}]}` stub cannot fail on anything the
# engine actually sends — a `reasoning`/`annotations`/`audio`/`function_call`/
# `refusal` set sitting beside `content`, a `routed_experts`/`stop_reason`/
# `token_ids` set beside `message`, an echoed model name
# (`Qwen3.8-Flash-Next-nvfp4`) that is *not* the `"primary"` the request asked
# for, and content that is JSON padded with newlines rather than a bare object.
_LIVE_REPLY = {
    "id": "cmpl-abc123",
    "object": "chat.completion",
    "created": 1789137060,
    "model": "Qwen3.8-Flash-Next-nvfp4",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": '{\n  "relevance": 10,\n  "why": "Speculative decoding is '
                       'a critical optimization technique for vLLM to significantly '
                       'reduce inference latency.",\n  "projects": [],\n  '
                       '"category": "llm-optimization"\n}',
            "refusal": None,
            "annotations": None,
            "audio": None,
            "function_call": None,
            "reasoning": None,
        },
        "logprobs": None,
        "finish_reason": "stop",
        "stop_reason": None,
        "token_ids": None,
        "routed_experts": None,
    }],
    "usage": {"prompt_tokens": 64, "total_tokens": 120, "completion_tokens": 56,
              "prompt_tokens_details": {"cached_tokens": 0,
                                        "created_cache_tokens": 0,
                                        "multimodal_tokens": None},
              "completion_tokens_details": {"reasoning_tokens": 0}},
    "system_fingerprint": "fp_local",
    "service_tier": "default",
    "prompt_logprobs": None,
    "prompt_text": "",
    "prompt_token_ids": [0] * 64,
    "kv_transfer_params": None,
    "ec_transfer_params": None,
    "metrics": {},
}


def test_call_local_llm_reads_the_reply_the_live_engine_actually_sends(monkeypatch):
    """The seam: parse the recorded live reply, not a stub's idealisation of it.

    The two ways this could break in production both read as a quiet day — the
    reply's `content` living somewhere else, or the engine's own keys confusing
    the reader — so the assertion runs the reply through the same two helpers
    `stage2_score` uses and demands the grade come out the other side.
    """
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: FakeResponse(_LIVE_REPLY))

    content = scoring_mod.call_local_llm("grade this item")

    graded = scoring_mod._parse_score_json(content)
    assert graded is not None, "the live reply's content is not parseable as a grade"
    assert scoring_mod._clamp_relevance(graded.get("relevance")) == 10
    assert graded["category"] == "llm-optimization"


def test_an_engine_that_answers_without_a_grade_is_reported_and_not_mistaken_for_a_quiet_day(
        redirect_paths, capsys):
    """The failure mode the reply-shape seam above guards: the engine answers, but
    with prose instead of a grade. `stage2_score` then keeps the keyword score —
    which is only honest if the run says out loud that the model judged nothing.
    Asserted through `run_scoring_pipeline`, because that line is the run output."""
    profile = _profile(redirect_paths)

    def rambles(prompt):
        return "I'd be happy to help with that request."

    scored = scoring_mod.run_scoring_pipeline(
        [_item(title="vllm inference", summary="vllm")], profile, llm_call=rambles)

    out = capsys.readouterr().out
    assert scored[0].relevance == KW_SCORE_09, "a wasted call must not lose the item"
    assert "LLM scored 0 of 1" in out, (
        "the run must report grades, not calls made\n" + out)
    assert "WARNING" in out and "keyword-only" in out, (
        "a model that graded nothing must be named, not averaged into a quiet day\n" + out)


def test_stage2_score_requests_the_model_above_the_threshold(redirect_paths):
    profile = _profile(redirect_paths)
    llm = RecordingLLM(relevance=7)
    item = _item(title="A vllm inference note", summary="about vllm inference")

    scored = scoring_mod.stage2_score([item], profile, llm_call=llm)

    assert len(llm.asked) == 1, "keyword score above 0.3 must consult the model"
    assert scored[0].relevance == 7, "the model's grade is the relevance, not the keyword 9"
    assert scored[0].category == "ai-llms"
    assert scored[0].projects == ["Lloyd"], "the model's project match must survive"
    assert "vllm" in llm.asked[0], "the prompt must carry the item, not just a greeting"


def test_stage2_score_skips_the_model_in_the_band_below_the_threshold(redirect_paths):
    """0.2 is a keyword match under the threshold — the band that exists only
    because the loader now reads weights (interests.md ships none)."""
    profile = _profile(redirect_paths)
    llm = RecordingLLM(relevance=9)
    item = _item(title="A new imu for the wrist rig",
                 summary="imu drift calibration")

    scored = scoring_mod.stage2_score([item], profile, llm_call=llm)

    # The score is taken on the same string stage2_score scores, not on a
    # paraphrase of it — otherwise this asserts a property of a string the item
    # under test does not carry.
    assert profile_mod.keyword_score(f"{item.title} {item.summary}", profile) == 0.2
    assert llm.asked == [], "0.2 is a match but under the 0.3 threshold"
    assert scored[0].relevance == 2, "int(round(0.2 * 10)), not a model's 9"


def test_stage2_score_asks_at_the_low_end_of_the_band(redirect_paths):
    """The other side of the same line: 0.4 clears it, and only a weight makes
    this distinguishable from the 0.2 case above."""
    profile = _profile(redirect_paths)
    llm = RecordingLLM(relevance=3)
    item = _item(title="px4 rates loop rework", summary="betaflight comparison")

    scored = scoring_mod.stage2_score([item], profile, llm_call=llm)

    assert len(llm.asked) == 1
    assert scored[0].relevance == 3


def test_stage2_score_skips_the_model_on_no_keyword_match(redirect_paths):
    profile = _profile(redirect_paths)
    llm = RecordingLLM(relevance=9)
    item = _item(title="Why green rebar?", summary="concrete reinforcement")

    scored = scoring_mod.stage2_score([item], profile, llm_call=llm)

    assert llm.asked == [], "a non-match must not spend a model call"
    assert scored[0].relevance == 1


def test_stage2_score_falls_back_to_the_keyword_score_on_garbage(redirect_paths):
    """A rambling model degrades to a named number, not to an arbitrary one."""
    profile = _profile(redirect_paths)
    item = _item(title="A vllm inference note", summary="vllm")

    scored = scoring_mod.stage2_score([item], profile,
                                      llm_call=lambda p: "not json at all")

    assert scored[0].relevance == KW_SCORE_09
    assert "vllm" in scored[0].why


def test_stage2_score_falls_back_when_the_model_call_raises(redirect_paths):
    """Model down is not a day with nothing in it: items keep their keyword score."""
    profile = _profile(redirect_paths)

    def boom(prompt):
        raise urllib.error.URLError("connection refused")

    scored = scoring_mod.stage2_score([_item(title="vllm", summary="vllm")],
                                      profile, llm_call=boom)

    assert scored[0].relevance == KW_SCORE_09


def test_stage2_score_rejects_an_out_of_range_grade_instead_of_clamping_it(redirect_paths):
    """97 is a failed grade. Clamping it to 10 would print a confident 10/10 for a
    reply the parser never validated, so the expected value is the keyword score."""
    profile = _profile(redirect_paths)

    scored = scoring_mod.stage2_score(
        [_item(title="vllm", summary="vllm")], profile, llm_call=_reply(relevance=97))

    assert scored[0].relevance == KW_SCORE_09, "an out-of-range grade must not clamp to 10"


def test_stage2_score_keeps_profile_projects_when_the_model_omits_them(redirect_paths):
    profile = _profile(redirect_paths)
    item = _item(title="Alfie wrist firmware", summary="vllm")

    scored = scoring_mod.stage2_score([item], profile, llm_call=_reply(relevance=5))

    assert scored[0].projects == ["Alfie"]


def test_stage2_score_stops_calling_the_model_past_its_budget(redirect_paths):
    """LLM_MAX_CALLS exists because task #30 has a 1800 s timeout: items past the
    budget keep the keyword score rather than stalling the run."""
    profile = _profile(redirect_paths)
    llm = RecordingLLM(relevance=6)
    items = [_item(f"v{i}", title="vllm thing", summary="vllm") for i in range(5)]

    scored = scoring_mod.stage2_score(items, profile, llm_call=llm, max_llm_calls=2)

    assert len(llm.asked) == 2, "the budget is the point of the budget"
    assert [s.relevance for s in scored] == [6, 6, KW_SCORE_09, KW_SCORE_09, KW_SCORE_09]


def test_disabling_the_model_by_env_leaves_every_item_on_keyword_scores(redirect_paths, monkeypatch):
    """INTEL_DISABLE_LLM=1 is the escape hatch for a batch run with the engine
    offline; it must reach the decision, not merely the client."""
    profile = _profile(redirect_paths)
    monkeypatch.setenv("INTEL_DISABLE_LLM", "1")
    calls = []

    def spy(prompt):
        calls.append(prompt)
        return '{"relevance": 6}'

    monkeypatch.setattr(scoring_mod, "call_local_llm", spy)

    scored = scoring_mod.stage2_score([_item(title="vllm", summary="vllm")], profile)

    assert calls == []
    assert scored[0].relevance == KW_SCORE_09


def test_match_projects_matches_against_declared_projects(redirect_paths):
    """`match_projects()` could never return anything while `get_all_projects()`
    could only return [] — clause 3's downstream consequence."""
    profile = _profile(redirect_paths)
    projects = profile_mod.get_all_projects(profile)

    assert {"Alfie", "Lloyd"} <= set(projects)
    assert scoring_mod.match_projects(
        _item(title="Alfie wrist firmware", summary=""), projects) == ["Alfie"]


def test_run_scoring_pipeline_reports_how_many_items_the_model_saw(redirect_paths, capsys):
    profile = _profile(redirect_paths)
    items = [
        _item("hit", title="vllm thing", summary="vllm"),
        _item("miss", title="green rebar", summary="rebar"),
    ]

    scored = scoring_mod.run_scoring_pipeline(items, profile,
                                              llm_call=RecordingLLM(5))

    out = capsys.readouterr().out
    # The whole line, graded-then-calls: "LLM scored 1" alone is also printed by a
    # run that made one call and got no grade out of it.
    assert "LLM scored 1 of 1" in out, (
        "the run output must say how many items the model graded out of how many "
        "it was asked about\n" + out)
    assert [s.relevance for s in scored] == [5], (
        "only the keyword-matching item reaches stage 2, and it carries the "
        f"model's grade: {[s.relevance for s in scored]}")


# --- clause 2: relevance is graded, not binary ------------------------------

def test_graded_relevance_survives_the_day_file_round_trip(redirect_paths, tmp_path):
    """`intel-<date>.jsonl` must be able to hold a value between 2 and 9.

    The measured defect was a Counter of exactly {1, 10} across three days.
    """
    profile = _profile(redirect_paths)
    scored = scoring_mod.stage2_score(
        [_item(title="vllm inference", summary="vllm")], profile, llm_call=RecordingLLM(6))
    day = tmp_path / "intel-test-day.jsonl"
    day.write_text("\n".join(i.to_json() for i in scored) + "\n")

    rows = [json.loads(l) for l in day.read_text().splitlines() if l.strip()]

    assert {r["relevance"] for r in rows} == {6}, (
        f"the day file does not hold the grade the model returned: {rows}")


# --- clause 3: weights and projects are parsed ------------------------------

def test_load_profile_parses_weight_and_projects(redirect_paths):
    profile = _profile(redirect_paths)
    robotics = profile["topics"][0]

    assert robotics["weight"] == 0.7, "**Weight:** must be read, not hardcoded 1.0"
    assert robotics["projects"] == ["Alfie"]
    assert "Alfie" in profile_mod.get_all_projects(profile), (
        "get_all_projects returning only [] is what made match_projects() dead")
    assert "actuator" in robotics["keywords"], "keywords survive the field block"


def test_load_profile_reads_an_explicit_keywords_field(redirect_paths):
    profile = _profile(redirect_paths)

    assert profile["topics"][1]["weight"] == 0.9
    assert "mcp" in profile["topics"][1]["keywords"]


def test_load_profile_reads_every_weight(redirect_paths):
    """One weight is a coincidence; four straddling the threshold is the parse."""
    profile = _profile(redirect_paths)

    assert {t["name"]: t["weight"] for t in profile["topics"]} == TOPIC_WEIGHTS


def test_load_profile_defaults_when_no_fields_are_present(redirect_paths):
    path = redirect_paths / "obsidian" / "bare.md"
    path.write_text("## Robotics\nhumanoid, actuator\n")

    profile = profile_mod.load_profile(str(path))

    assert profile["topics"][0]["weight"] == 1.0, "an unweighted topic still counts"
    assert profile["topics"][0]["keywords"] == ["humanoid", "actuator"]
    assert profile["topics"][0]["projects"] == []


def test_field_lines_do_not_leak_into_keywords(redirect_paths):
    """A `**Weight:** 0.7` line kept as a keyword would displace a real one and
    match nothing."""
    profile = _profile(redirect_paths)

    assert not any("**" in kw or "weight" in kw.lower()
                   for kw in profile["all_keywords"])


def test_an_unparseable_weight_leaves_the_default(redirect_paths):
    path = redirect_paths / "obsidian" / "bad.md"
    path.write_text("## Robotics\n**Weight:** high\nhumanoid\n")

    profile = profile_mod.load_profile(str(path))

    assert profile["topics"][0]["weight"] == 1.0


# --- clause 4: stage1_filter actually filters -------------------------------

def test_stage1_filter_drops_repo_watch_items_that_match_no_keyword(redirect_paths):
    """The `or item.source_tags` escape passed every scanner item.

    Mirrors raw/2026-09-10.jsonl's shape: 103 items, 90 carrying source_tags with
    no keyword hit, 13 matching. The assertion is on *which* items survived, not
    how many the fixture built — a count of the test's own literals cannot fail.
    """
    profile = _profile(redirect_paths)
    watched = [_item(f"watch{i}", source="github", title=f"commit {i}",
                     summary="routine refactor", tags=["commit"]) for i in range(90)]
    matching = [_item(f"hit{i}", source="github", title="gripper PR",
                      summary="new gripper", tags=["pr"]) for i in range(13)]

    kept = scoring_mod.stage1_filter(watched + matching, profile)

    assert {i.id for i in kept} == {item.id for item in matching}, (
        "repo traffic alone is not an interest hit; kept "
        f"{sorted(i.id for i in kept if i.id.startswith('watch'))[:5]}")


def test_stage1_filter_keeps_a_matching_repo_item(redirect_paths):
    profile = _profile(redirect_paths)
    items = [_item("hit", source="github", title="gripper calibration",
                   summary="", tags=["commit"])]

    assert len(scoring_mod.stage1_filter(items, profile)) == 1


def test_stage1_filter_ignores_source_tags_entirely(redirect_paths):
    """Tags must not matter at all — that is the escape hatch being gone rather
    than merely narrowed."""
    profile = _profile(redirect_paths)
    tagged = _item("t", source="github", title="green rebar", summary="rebar",
                   tags=["commit"])
    untagged = _item("u", source="github", title="green rebar", summary="rebar")

    assert scoring_mod.stage1_filter([tagged, untagged], profile) == []


# --- clause 5: the writer refuses the noise floor ---------------------------

def _write_day(tmp_path, scored, day="2026-09-11"):
    (tmp_path / "lloyd" / "_pipeline" / "vault-derived" / "memory" / "feeds"
     / f"intel-{day}.jsonl").write_text(
         "\n".join(i.to_json() for i in scored) + "\n")
    return day


def _scored(item, relevance):
    return ScoredItem(**{**item.to_dict(), "relevance": relevance, "urgency": "low",
                         "why": "Matches: vllm", "projects": [],
                         "category": "ai-llms"})


def _sections(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.startswith("## "))


def _uncategorized_sections(tmp_path):
    return _sections(tmp_path / "obsidian" / "knowledge" / "feeds"
                     / "youtube-uncategorized.md")


def _knowledge_sections(tmp_path):
    root = tmp_path / "obsidian" / "knowledge"
    return sum(_sections(p) for p in root.rglob("*.md"))


def test_write_all_to_vault_skips_items_below_the_floor(redirect_paths):
    """The clause's measure is the noise file's `^## ` count, unchanged.

    So the file has to exist first — counting zero sections in a file that was
    never created would reward absence instead of restraint. The live file holds
    2,545 such sections; one of them is enough to make "unchanged" mean something.
    """
    noise_file = (redirect_paths / "obsidian" / "knowledge" / "feeds"
                  / "youtube-uncategorized.md")
    noise_file.parent.mkdir(parents=True, exist_ok=True)
    noise_file.write_text("# YouTube\n\n## 2026-09-10\n\n### An older entry\n\n"
                          "[Link](https://example.com/older)\n\n---\n\n")
    before = _sections(noise_file)
    assert before == 1, "the file must exist before its count can be held"

    noise = [_scored(_item(f"n{i}", source="youtube",
                           title=f"AI Just Did the Impossible {i}",
                           summary="thumbnail bait"), 1) for i in range(20)]
    day = _write_day(redirect_paths, noise)

    written = vw_mod.write_all_to_vault(day)

    assert written == 0, "a rel=1-only day must write nothing"
    assert _sections(noise_file) == before, "the noise file grew anyway"


def test_write_all_to_vault_still_writes_items_at_or_above_the_floor(redirect_paths):
    """Guards the previous test against passing by never writing anything."""
    good = _scored(_item("g1", source="youtube", title="vllm speculative decoding",
                         summary="real content"), 6)
    day = _write_day(redirect_paths, [good])

    written = vw_mod.write_all_to_vault(day)

    assert written == 1
    assert _uncategorized_sections(redirect_paths) == 1


def test_write_all_to_vault_treats_the_floor_as_inclusive(redirect_paths):
    mixed = [_scored(_item("keep", source="youtube", title="vllm thing",
                           summary="vllm"), vw_mod.RELEVANCE_FLOOR),
             _scored(_item("drop", source="youtube", title="Why green rebar?",
                           summary="rebar"), vw_mod.RELEVANCE_FLOOR - 1)]
    day = _write_day(redirect_paths, mixed)

    written = vw_mod.write_all_to_vault(day)

    assert written == 1
    assert _knowledge_sections(redirect_paths) == 1


def test_writer_declares_its_floor_in_the_run_output(redirect_paths, capsys):
    noise = [_scored(_item("n1", source="youtube", title="Why green rebar?",
                           summary="rebar"), 1)]
    day = _write_day(redirect_paths, noise)

    vw_mod.write_all_to_vault(day)

    out = capsys.readouterr().out
    assert "below relevance floor" in out.lower()
    assert str(vw_mod.RELEVANCE_FLOOR) in out


def test_below_floor_refuses_an_unscored_item():
    """An item with no usable relevance is not evidence of relevance."""
    item = _scored(_item("x", source="youtube", title="t", summary="s"), 6)
    item.relevance = None

    assert vw_mod.below_floor(item)


# --- clause 6: a rate limit is not a quiet day ------------------------------

def _install_fake_urlopen(monkeypatch, captured, *, fail_403=False):
    """Replace urllib's urlopen as seen through the scanner module.

    A 403 is raised exactly as urllib raises it — HTTPError, not a return value —
    because the defect under test is a broad `except Exception` swallowing it.
    """
    def fake(req, timeout=None):
        captured.append(dict(req.headers))
        if fail_403:
            raise urllib.error.HTTPError(
                req.full_url, 403, "rate limit exceeded",
                {"X-RateLimit-Remaining": "0", "X-RateLimit-Limit": "60"}, None)
        return FakeResponse([])

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def test_github_scanner_sends_an_authorization_header_when_a_token_is_set(redirect_paths, monkeypatch, tmp_path):
    cfg = tmp_path / "github-repos.yml"
    cfg.write_text("token: ghp_testtoken\nrepos:\n  - owner: openclaw\n    repo: openclaw\n"
                   "    track: [commits]\n")
    monkeypatch.setattr(gh_mod, "CONFIG_PATH", cfg)
    captured = []
    _install_fake_urlopen(monkeypatch, captured)

    gh_mod.fetch_commits("openclaw", "openclaw")

    headers = captured[0]
    assert "Authorization" in headers, "unauthenticated calls run against 60 req/hour"
    assert "ghp_testtoken" in headers["Authorization"]


def test_github_scanner_sends_no_authorization_header_without_a_token(redirect_paths, monkeypatch, tmp_path):
    cfg = tmp_path / "github-repos.yml"
    cfg.write_text("repos:\n  - owner: openclaw\n    repo: openclaw\n    track: [commits]\n")
    monkeypatch.setattr(gh_mod, "CONFIG_PATH", cfg)
    captured = []
    _install_fake_urlopen(monkeypatch, captured)

    gh_mod.fetch_commits("openclaw", "openclaw")

    assert "Authorization" not in captured[0]


def test_token_prefers_the_environment_over_the_config(redirect_paths, monkeypatch, tmp_path):
    cfg = tmp_path / "github-repos.yml"
    cfg.write_text("token: ghp_from_config\nrepos: []\n")
    monkeypatch.setattr(gh_mod, "CONFIG_PATH", cfg)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_from_env")

    assert gh_mod.load_github_token() == "ghp_from_env"


def test_http_403_raises_a_named_rate_limit_error(redirect_paths, monkeypatch, tmp_path):
    cfg = tmp_path / "github-repos.yml"
    cfg.write_text("repos:\n  - owner: openclaw\n    repo: openclaw\n    track: [commits]\n")
    monkeypatch.setattr(gh_mod, "CONFIG_PATH", cfg)
    captured = []
    _install_fake_urlopen(monkeypatch, captured, fail_403=True)

    with pytest.raises(gh_mod.GitHubRateLimitError) as exc:
        gh_mod.fetch_commits("openclaw", "openclaw")

    assert exc.value.status == 403
    assert exc.value.remaining == "0"


def _rate_limit_scan_setup(redirect_paths, monkeypatch, tmp_path, capsys):
    cfg = tmp_path / "github-repos.yml"
    cfg.write_text("repos:\n  - owner: openclaw\n    repo: openclaw\n"
                   "    track: [commits, issues]\n")
    monkeypatch.setattr(gh_mod, "CONFIG_PATH", cfg)
    captured = []
    _install_fake_urlopen(monkeypatch, captured, fail_403=True)
    state_dir = tmp_path / "state"
    monkeypatch.setattr(gh_mod.state, "RAW_DIR", state_dir / "raw")
    monkeypatch.setattr(gh_mod.state, "STATE_FILE", state_dir / "scanner-state.json")
    return captured


def test_a_rate_limited_scan_says_so_in_the_run_output(redirect_paths, monkeypatch, tmp_path, capsys):
    """Before: a 403 became a printed error and 0 items, indistinguishable from a
    repo that had nothing new. The report said `Pipeline Complete` either way."""
    _rate_limit_scan_setup(redirect_paths, monkeypatch, tmp_path, capsys)

    items = gh_mod.scan_github_repos()

    out = capsys.readouterr().out
    assert items == []
    assert "GITHUB_RATE_LIMIT" in out, "a rate limit must be a named failure, not 0 items"
    assert "403" in out


def test_a_rate_limited_repo_does_not_advance_its_state(redirect_paths, monkeypatch, tmp_path, capsys):
    """Stamping a repo we could not read would drop every item updated during the
    quota window, so the loss would outlive the rate limit."""
    _rate_limit_scan_setup(redirect_paths, monkeypatch, tmp_path, capsys)

    gh_mod.scan_github_repos()

    saved = gh_mod.state.load_state()
    assert "openclaw/openclaw" not in saved.get(gh_mod.GITHUB_STATE_KEY, {})


def test_a_rate_limited_scan_leaves_yesterdays_state_where_it_was(
        redirect_paths, monkeypatch, tmp_path, capsys):
    """The realistic loss, which the empty-state case above cannot see.

    With nothing stored, a wiped sha loses nothing and the assertion is only as
    strong as the wipe being wrong. Seeded with a sha from a scan that succeeded
    yesterday, the same run has something to destroy: anything that stamps the
    repo from a response it never read re-drops every item published during the
    quota window, and the next run sees no gap to recover them from."""
    _rate_limit_scan_setup(redirect_paths, monkeypatch, tmp_path, capsys)
    stored = {gh_mod.GITHUB_STATE_KEY: {"openclaw/openclaw:commits": "9f3a1c2"},
              yt_mod.YOUTUBE_STATE_KEY: {"some-channel": "vid-1"}}
    gh_mod.state.save_state(stored)

    items = gh_mod.scan_github_repos()

    out = capsys.readouterr().out
    after = gh_mod.state.load_state()
    assert items == []
    assert after.get(gh_mod.GITHUB_STATE_KEY) == {"openclaw/openclaw:commits": "9f3a1c2"}, (
        f"a quota-exhausted scan rewrote stored scan state\n{out}")
    assert after.get(yt_mod.YOUTUBE_STATE_KEY) == {"some-channel": "vid-1"}, (
        "the youtube half of the state is not this failure's to touch")


class _GitHubStub(BaseHTTPRequestHandler):
    """Answers like GitHub's REST API — 200 with a body, or a real 403 carrying
    the X-RateLimit headers — over a socket rather than a patched urlopen."""

    def do_GET(self):  # noqa: N802
        if self.headers.get("Authorization") != "Bearer ghp_stub":
            body = b'{"message":"API rate limit exceeded"}'
            self.send_response(403)
            self.send_header("X-RateLimit-Remaining", "0")
            self.send_header("X-RateLimit-Limit", "60")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.server.paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"[]")

    def log_message(self, *args):
        pass


@pytest.fixture
def github_stub():
    """A loopback HTTP server standing in for api.github.com.

    This is as far as the GitHub seam can be crossed inside a promotion gate:
    api.github.com is unreachable to a test that must run offline and must not
    spend real quota. What it does prove is that the header and the 403 mapping
    survive a real urllib exchange and a real status line, not merely a Request
    object assembled in-process.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GitHubStub)
    server.paths = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_the_scanner_carries_the_token_over_a_real_http_exchange(redirect_paths, monkeypatch, github_stub):
    monkeypatch.setattr(gh_mod, "GITHUB_API_URL",
                        f"http://127.0.0.1:{github_stub.server_port}")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_stub")

    assert gh_mod.fetch_commits("openclaw", "openclaw") == []
    assert any("/repos/openclaw/openclaw/commits" in p for p in github_stub.paths), (
        "the request never arrived with the token, so nothing was served")


def test_a_real_http_403_becomes_a_named_rate_limit_error(redirect_paths, monkeypatch, github_stub):
    """No token -> the stub answers 403 exactly as the rate limit does, through
    urllib's real HTTPError path rather than a raised stand-in."""
    monkeypatch.setattr(gh_mod, "GITHUB_API_URL",
                        f"http://127.0.0.1:{github_stub.server_port}")

    with pytest.raises(gh_mod.GitHubRateLimitError) as exc:
        gh_mod.fetch_commits("openclaw", "openclaw")

    assert exc.value.status == 403
    assert exc.value.remaining == "0", "the headers must ride along with the error"


class _LowerCaseRateLimitStub(BaseHTTPRequestHandler):
    """403 with the quota headers in lower case.

    Whether the real api.github.com spells its quota headers a particular way is
    a question a gate must not answer with the network; the *property* that
    matters is that the reader is not case-sensitive, because HTTP/2 and some
    proxies put every field name on the wire in lower case. So this stub sends
    them that way over a real socket and the code must still name the quota.
    """

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        body = b'{"message":"API rate limit exceeded for user ID 0."}'
        self.send_response(403)
        self.send_header("x-ratelimit-remaining", "0")
        self.send_header("x-ratelimit-limit", "60")
        self.send_header("x-ratelimit-reset", "1789139000")
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def lowercase_rate_limit_stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LowerCaseRateLimitStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_the_quota_is_named_whatever_casing_the_wire_used(
        redirect_paths, monkeypatch, lowercase_rate_limit_stub):
    """Clause 6's header seam, closed on a socket instead of a hand-built dict."""
    monkeypatch.setattr(gh_mod, "GITHUB_API_URL",
                        f"http://127.0.0.1:{lowercase_rate_limit_stub.server_port}")

    with pytest.raises(gh_mod.GitHubRateLimitError) as exc:
        gh_mod.fetch_commits("openclaw", "openclaw")

    assert exc.value.remaining == "0", "the quota was not read off a lower-cased wire"
    assert exc.value.limit == "60"


# --- seams: the CLI as the autonomy task actually runs it -------------------

class _StubModel(BaseHTTPRequestHandler):
    """Answers /v1/chat/completions the way the local engine does.

    `server.relevance` lets a test choose what the model thinks, so a
    graded-but-below-floor day is expressible.
    """

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"choices": [{"message": {
            "content": json.dumps({
                "relevance": getattr(self.server, "relevance", 6),
                "urgency": "morning",
                "why": "matches an active interest",
                "projects": ["Lloyd"],
                "category": "ai-llms"})}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _cli_home(tmp_path):
    """A scratch HOME with the paths `_paths`/PROFILE/config resolve to."""
    home = tmp_path / "home"
    feeds = home / "lloyd-data" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    (home / "obsidian").mkdir(parents=True)
    (home / "obsidian" / "interests.md").write_text(PROFILE_MD)
    (home / "lloyd" / "scripts" / "intel-pipeline" / "config").mkdir(parents=True)
    return home, feeds


def _today_str() -> str:
    """The calendar-day key the CLI derives for itself, as this process sees it.

    The subprocess computes its own `today`; a test that names a *different* day
    asserts about both files, so one helper has to be where that key comes from.
    Across UTC midnight the two processes can disagree — the pre-existing hazard of
    this harness, not one these tests add.
    """
    return __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d")


def _run_cli(home, relevance, *flags, extra_env=None):
    today = _today_str()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubModel)
    server.relevance = relevance
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = dict(os.environ, HOME=str(home), LLOYD_DATA=str(home / "lloyd-data"),
                   INTEL_DISABLE_LLM="0",
                   INTEL_LLM_URL=f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
        env.update(extra_env or {})
        proc = subprocess.run(
            [sys.executable, "-m", "intel_pipeline", *flags],
            cwd=str(INTEL_DIR), env=env, capture_output=True, text=True, timeout=180)
    finally:
        server.shutdown()
        thread.join(timeout=5)
    return today, proc


_RAW_DAY = "\n".join([
    _item("yt1", source="youtube", title="vllm speculative decoding for agents",
          summary="vllm").to_json(),
    _item("yt2", source="youtube", title="Why green rebar?", summary="rebar").to_json(),
]) + "\n"


def test_cli_run_scores_a_day_end_to_end_over_the_http_seam(tmp_path):
    """Crosses the two seams the unit tests cannot: the `python -m intel_pipeline`
    subprocess the autonomy worker spawns, and the loopback POST to the model."""
    home, feeds = _cli_home(tmp_path)
    today = _today_str()
    (feeds / "raw" / f"{today}.jsonl").write_text(_RAW_DAY)

    day, proc = _run_cli(home, 6, "--score")

    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = [json.loads(l) for l in
            (feeds / f"intel-{day}.jsonl").read_text().splitlines() if l.strip()]

    assert {r["relevance"] for r in rows} == {6}, (
        f"the CLI did not put the model's grade in the day file: {rows}\n"
        f"{proc.stdout[-2000:]}")
    assert "LLM scored 1" in proc.stdout, (
        "the junk item must have been dropped before scoring\n" + proc.stdout[-2000:])


def _seed_noise_file(home: Path) -> Path:
    """Put one entry in the noise file before a run.

    Clause 5 says the file's `^## ` count is *unchanged across a run*, and every
    `== 0` assertion below would also be satisfied by the file never existing —
    which is a different fact. Seeding makes "0 new sections" mean 0 new
    sections against a file that really is there.
    """
    noise = home / "obsidian" / "knowledge" / "feeds" / "youtube-uncategorized.md"
    noise.parent.mkdir(parents=True, exist_ok=True)
    noise.write_text("---\ntype: note\n---\n\n# Feed\n\n"
                     "## Old entry — already in there before this run\n\n"
                     "Why: Matches: old\n\nRelevance: 1/10 | Urgency: low | "
                     "Date: 2026-09-01\n\n")
    return noise


def test_cli_run_writes_only_what_clears_the_floor(tmp_path):
    """A graded day end to end: the 6/10 item is written, and the keyword-matched
    item the model put below the floor is not — while a noise file that already
    exists does not grow by one section, which is how it reached 2,545."""
    home, feeds = _cli_home(tmp_path)
    today = _today_str()
    (feeds / "raw" / f"{today}.jsonl").write_text(_RAW_DAY)
    noise = _seed_noise_file(home)

    day, proc = _run_cli(home, 2, "--score", "--write")

    assert proc.returncode == 0, proc.stderr[-2000:]
    knowledge = home / "obsidian" / "knowledge"

    assert _knowledge_sections_for(knowledge) == 1, (
        f"a 2/10 day wrote to the vault anyway; only the seeded noise entry may "
        f"be there\n{proc.stdout[-2500:]}")
    assert _sections(noise) == 1, (
        "the noise file grew this run\n" + noise.read_text()[:1500])
    assert {json.loads(l)["relevance"] for l in
            (feeds / f"intel-{day}.jsonl").read_text().splitlines() if l.strip()} == {2}


def test_cli_run_writes_a_graded_item_it_liked(tmp_path):
    """The other side of that floor, over the same seam: a 6/10 item lands, and a
    noise file that already exists comes out of the run the same size it went in.

    The noise file is seeded with one prior entry on purpose. Asserting `== 0`
    would be satisfied by the file never being created, which is a different
    fact from the one clause 5 states — the count is *unchanged across a run*.
    """
    home, feeds = _cli_home(tmp_path)
    today = _today_str()
    (feeds / "raw" / f"{today}.jsonl").write_text(_RAW_DAY)
    noise = _seed_noise_file(home)

    day, proc = _run_cli(home, 6, "--score", "--write")

    assert proc.returncode == 0, proc.stderr[-2000:]
    knowledge = home / "obsidian" / "knowledge"
    written = _knowledge_sections_for(knowledge)

    assert written == 2, (
        f"the prior noise entry plus the one item above the floor\n"
        f"{proc.stdout[-2500:]}")
    assert _sections(noise) == 1, (
        "the noise file must leave a run exactly as tall as it entered it\n"
        f"{noise.read_text()[:1500]}")


def _knowledge_sections_for(knowledge: Path) -> int:
    return sum(_sections(p) for p in knowledge.rglob("*.md"))


# --- backlog #853: `--date D` must select day D for EVERY stage --------------
#
# The scoring stage keyed its raw read and its scored write on `today`
# (`__main__.py` `state.load_raw_items(today)` / `intel-{today}.jsonl`) while the
# write stage keyed on `date_str` (`write_all_to_vault(date_str)`, whose
# `load_scored_items(date_str)` reads `intel-<date_str>.jsonl`). So
# `--date D --score --write` scored today into `intel-<today>.jsonl` and then read
# `intel-D.jsonl`: the day just scored never reached the vault, and a stale day's
# scored file could be re-published instead. Back-filling a day — the obvious
# recovery move after a bad run — silently paired two days in one invocation.
# These tests hold the whole CLI in one subprocess, HOME redirected, so both ends
# of the day key are visible: which raw file was opened, and which intel file and
# which vault notes came out of it.

_NAMED_DAY = "2026-09-10"
_NAMED_DAY_ITEMS = "\n".join([
    _item("d1", source="youtube", title="vllm continuous batching",
          summary="vllm").to_json(),
]) + "\n"
_TODAY_ITEMS = "\n".join([
    _item("t1", source="youtube", title="vllm chunked prefill",
          summary="vllm").to_json(),
]) + "\n"


def test_cli_date_names_the_day_the_scorer_reads(tmp_path):
    """Clause 1: `--date <D> --score` scores the items in D's raw file.

    The scratch HOME holds a distinct raw file for the named day and for today, so
    only one of the two titles can legitimately appear — and before the fix the
    named day's title could not, because the stage opened today's file.
    """
    home, feeds = _cli_home(tmp_path)
    (feeds / "raw" / f"{_NAMED_DAY}.jsonl").write_text(_NAMED_DAY_ITEMS)
    (feeds / "raw" / f"{_today_str()}.jsonl").write_text(_TODAY_ITEMS)

    _, proc = _run_cli(home, 6, "--date", _NAMED_DAY, "--score")

    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (feeds / f"intel-{_NAMED_DAY}.jsonl").exists(), (
        f"the named day's scored file was never written; stdout said:\n"
        f"{proc.stdout[-2000:]}")
    rows = [json.loads(l) for l in
            (feeds / f"intel-{_NAMED_DAY}.jsonl").read_text().splitlines()
            if l.strip()]
    titles = {r["title"] for r in rows}

    assert titles == {"vllm continuous batching"}, (
        f"the run must score the named day's items and none of today's: {rows}\n"
        f"{proc.stdout[-2000:]}")


def test_cli_date_writes_the_scored_file_the_writer_will_read(tmp_path):
    """Clause 2: scoring emits `intel-<D>.jsonl` and creates no `intel-<today>.jsonl`.

    The day key has to be one value across the whole invocation, not merely a
    correct read: `write_all_to_vault(D)` calls `load_scored_items(D)`, so the
    scored file scoring produced has to BE `intel-D.jsonl`. Seeding a stale
    `intel-D.jsonl` makes the re-publish failure visible — the run must overwrite
    it with the day's fresh scores.
    """
    home, feeds = _cli_home(tmp_path)
    (feeds / "raw" / f"{_NAMED_DAY}.jsonl").write_text(_NAMED_DAY_ITEMS)
    (feeds / "raw" / f"{_today_str()}.jsonl").write_text(_TODAY_ITEMS)
    stale = feeds / f"intel-{_NAMED_DAY}.jsonl"
    stale.write_text(ScoredItem(**{**_item("old", source="youtube",
                                           title="STALE DAY SCORE",
                                           summary="vllm").to_dict(),
                                   "relevance": 9, "urgency": "low",
                                   "why": "written by an earlier run",
                                   "projects": ["Lloyd"],
                                   "category": "ai-llms"}).to_json() + "\n")

    _, proc = _run_cli(home, 6, "--date", _NAMED_DAY, "--score")

    assert proc.returncode == 0, proc.stderr[-2000:]
    titles = {json.loads(l)["title"] for l in stale.read_text().splitlines()
              if l.strip()}

    assert titles == {"vllm continuous batching"}, (
        f"the day's scored file must hold the scores this run produced: "
        f"{titles}\n{proc.stdout[-2000:]}")
    assert [q.name for q in feeds.glob("intel-*.jsonl")] == [
            f"intel-{_NAMED_DAY}.jsonl"], (
        "a run that named one day left a second scored file behind — the write stage "
        "picks a day by filename, so a stray file is a day it would publish unasked. "
        "Asserted against the directory listing rather than `intel-<today>` by name so "
        "it cannot pass or fail on a UTC midnight boundary\n"
        f"{proc.stdout[-2000:]}")


def test_cli_date_write_stage_publishes_the_day_it_named(tmp_path):
    """The two halves meeting: `--date D --score --write` reaches the vault.

    The defect's worse consequence — narrower than "re-scores the wrong day":
    the writer read `intel-D.jsonl`, which scoring had not written, so today's
    fresh scores never reached the vault at all (`No scored items found for D`).
    After the fix the named day's item is in the knowledge tree, and today's item
    is nowhere in it.
    """
    home, feeds = _cli_home(tmp_path)
    (feeds / "raw" / f"{_NAMED_DAY}.jsonl").write_text(_NAMED_DAY_ITEMS)
    (feeds / "raw" / f"{_today_str()}.jsonl").write_text(_TODAY_ITEMS)

    _, proc = _run_cli(home, 6, "--date", _NAMED_DAY, "--score", "--write")

    assert proc.returncode == 0, proc.stderr[-2000:]
    bodies = "\n".join(p.read_text()
                       for p in (home / "obsidian" / "knowledge").rglob("*.md"))

    assert "vllm continuous batching" in bodies, (
        f"the day the run named never reached the vault\n{proc.stdout[-2500:]}")
    assert "vllm chunked prefill" not in bodies, (
        f"today's item was written on a run that named {_NAMED_DAY}\n"
        f"{proc.stdout[-2500:]}")


def test_cli_named_day_with_no_raw_file_says_so(tmp_path):
    """A named day that has no raw file is announced, and no scored file is written.

    Writing an empty `intel-D.jsonl` would be the louder-seeming fix and the worse
    one: for a day whose raw file has rotated away, that file is the last record of
    what was scored. So the run prints the day it could not score — the silence was
    what let a stale scored file reach the vault unnoticed — and leaves it standing.
    """
    home, feeds = _cli_home(tmp_path)
    (feeds / "raw" / f"{_today_str()}.jsonl").write_text(_TODAY_ITEMS)

    _, proc = _run_cli(home, 6, "--date", _NAMED_DAY, "--score", "--write")

    assert proc.returncode == 0, proc.stderr[-2000:]
    assert f"NO RAW ITEMS FOR {_NAMED_DAY}" in proc.stdout, (
        f"a day with nothing to score must be named, not skipped silently\n"
        f"{proc.stdout[-2000:]}")
    assert not (feeds / f"intel-{_NAMED_DAY}.jsonl").exists(), (
        "an empty scored file would have replaced whatever the day had")


def test_cli_refuses_a_malformed_date_before_running_any_stage(tmp_path):
    """`--date` is not a day → exit 2, before any stage runs.

    The value became part of a filename two directories deep (`raw/<X>.jsonl`,
    `intel-<X>.jsonl`) with nothing checking it, so a typo'd day — `09-10-2026` —
    walked the stages, found nothing, and still printed `Pipeline Complete`.
    """
    home, feeds = _cli_home(tmp_path)
    (feeds / "raw" / f"{_today_str()}.jsonl").write_text(_TODAY_ITEMS)

    _, proc = _run_cli(home, 6, "--date", "09-10-2026", "--score", "--write")

    assert proc.returncode == 2, (
        f"a malformed --date must be refused, not run: rc={proc.returncode}\n"
        f"{proc.stdout[-1500:]}")
    assert "expected YYYY-MM-DD" in proc.stdout, proc.stdout[-1500:]
    assert "Pipeline Complete" not in proc.stdout, (
        f"the stages ran anyway:\n{proc.stdout[-1500:]}")
    assert not list(feeds.glob("intel-*.jsonl")), (
        f"a refused run wrote scored output: "
        f"{sorted(p.name for p in feeds.iterdir())}")


# --- backlog #739: YouTube feed coverage is a signal, not a silence ---------

"""The 2026-09-10 defect, in the shape the tests below pin.

Upstream answers a *well-formed* channel id with 404 or 500 intermittently — 55
of 64 feeds lost on 09-10, 60/64 on 09-09, 37/64 on 09-12, all 64 on 09-13 — and
the scanner made exactly one attempt per channel behind `except Exception: return
[]`. `if not videos: continue` then merged "unreachable" with "idle", so a
blackout and a quiet night produced the same run report: `YouTube scanner: 0
items` and `=== Pipeline Complete ===` at exit 0.

Three things had to change, and each test below pins one: a transient failure is
retried with increasing waits; coverage is counted from fetch outcomes *before*
the empty-feed branch and persisted to `scanner-state.json`; and a run whose
coverage is under half the channels attempted exits non-zero without printing the
success banner.

Feed outcomes in the scan-level tests come from a loopback stub rather than a
patched transport, because "the endpoint was unreachable" and "the feed was empty"
are different HTTP responses — a fake that returns a Python value cannot tell
them apart, which is precisely the confusion under test.
"""

_ATOM_NS = ('xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:media="http://search.yahoo.com/mrss/" '
            'xmlns:yt="http://www.youtube.com/xml/schemas/2015"')


def _atom_body(video_ids):
    """A feed body in the shape the scanner parses: one <entry> per video id."""
    entries = "".join(
        f"<entry><id>yt:video:{v}</id><title>Video {v}</title>"
        f"<published>2026-09-19T00:00:00Z</published>"
        f'<link rel="alternate" href="https://youtu.be/{v}"/>'
        f"<media:description>desc {v}</media:description></entry>"
        for v in video_ids)
    return (f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<feed {_ATOM_NS}><title>channel</title>{entries}</feed>')


def _atom_bytes(video_ids):
    return _atom_body(video_ids).encode()


class _StubFeedsHandler(BaseHTTPRequestHandler):
    """Answers the RSS endpoint the way the real one does, per channel id.

    `server.outcomes` maps a channel id either to an HTTP status to answer with
    (an unreachable feed is an *error response*, not an empty one) or to the list
    of video ids to serve, which may be empty for a channel that published
    nothing. Anything not listed answers 500. `server.hits` records the channel id
    of every request, so attempts are counted on the wire rather than trusted from
    the scanner's own printout.
    """

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        channel = self.path.split("channel_id=")[-1]
        outcome = self.server.outcomes.get(channel, 500)
        self.server.hits.append(channel)
        body = b"<html>no such feed</html>" if isinstance(outcome, int) else _atom_bytes(outcome)
        self.send_response(outcome if isinstance(outcome, int) else 200)
        self.send_header("Content-Type", "application/atom+xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def feed_stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubFeedsHandler)
    server.outcomes = {}
    server.hits = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _stub_rss_url(server):
    return f"http://127.0.0.1:{server.server_port}/feeds/videos.xml"


def _channels(*ids):
    return [{"channel_id": i, "name": i, "handle": f"@{i}"} for i in ids]


def _state_json():
    """The state file the redirected `state` module is currently writing."""
    return json.loads(state_mod.STATE_FILE.read_text())


def test_a_feed_that_404s_twice_is_retried_and_its_entries_survive(monkeypatch):
    """Clause 1: the 404 that was fatal on one attempt is a transient on the third.

    The waits go through the module-level `sleep` so the ladder is observable
    without spending wall clock, and are asserted as the exact increasing series
    rather than "some positive numbers".
    """
    calls, waits = [], []
    monkeypatch.setattr(yt_mod, "sleep", waits.append)

    def flaky_get(url, headers=None, timeout=None):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _atom_body(["abc123"])

    monkeypatch.setattr(yt_mod, "_http_get", flaky_get)

    videos, fetched = yt_mod.fetch_channel_rss("UCrDwWp7EBBv4NwvScIpBDOA")

    assert fetched is True, "a feed recovered on retry must count as fetched"
    assert [v["id"] for v in videos] == ["abc123"]
    assert len(calls) == 3, f"expected 3 HTTP attempts, got {len(calls)}"
    assert waits == [5.0, 10.0], f"expected increasing waits, got {waits}"


def test_a_feed_that_never_recovers_is_attempted_four_times_then_failed(monkeypatch):
    """Clause 1's ceiling, and clause 2's failure half at the fetch seam."""
    calls, waits = [], []
    monkeypatch.setattr(yt_mod, "sleep", waits.append)

    def dead_get(url, headers=None, timeout=None):
        calls.append(url)
        raise urllib.error.HTTPError(url, 500, "Server Error", {}, None)

    monkeypatch.setattr(yt_mod, "_http_get", dead_get)

    videos, fetched = yt_mod.fetch_channel_rss("UCrDwWp7EBBv4NwvScIpBDOA")

    assert videos == []
    assert fetched is False, "an exhausted channel must be a coverage failure"
    assert len(calls) == yt_mod.FETCH_ATTEMPTS == 4
    assert waits == [5.0, 10.0, 20.0], f"expected 3 increasing waits, got {waits}"


def test_an_unreachable_feed_is_a_coverage_failure_while_an_idle_one_is_not(
        redirect_paths, monkeypatch, feed_stub):
    """Clause 2: the distinction `if not videos: continue` used to erase.

    One channel answers 404 to every attempt, the other answers 200 with a feed
    holding zero entries. Both yield zero videos; only one is a coverage failure.
    """
    monkeypatch.setattr(yt_mod, "RSS_FEED_URL", _stub_rss_url(feed_stub))
    monkeypatch.setattr(yt_mod, "sleep", lambda s: None)
    monkeypatch.setattr(yt_mod, "load_youtube_channels_config",
                        lambda: _channels("UCdead", "UCidle"))
    feed_stub.outcomes = {"UCdead": 404, "UCidle": []}

    items, coverage = yt_mod.scan_youtube_channels()

    assert items == [], "neither channel had anything to report"
    assert (coverage.fetched, coverage.attempted) == (1, 2), (
        "the idle channel must be counted as fetched, the 404 channel must not")
    # Exactly half is not *strictly below* half, which is where clause 4 draws the
    # line; the degradation itself is pinned over the CLI at 1 fetched of 3.
    assert coverage.degraded is False, "half the feeds reachable is at, not under, the floor"


def test_the_run_persists_fetched_versus_attempted_counts_to_state(
        redirect_paths, monkeypatch, feed_stub):
    """Clause 3: the number has to outlive the run, in the file a reader checks."""
    monkeypatch.setattr(yt_mod, "RSS_FEED_URL", _stub_rss_url(feed_stub))
    monkeypatch.setattr(yt_mod, "sleep", lambda s: None)
    monkeypatch.setattr(yt_mod, "load_youtube_channels_config",
                        lambda: _channels("UCdead", "UCidle"))
    feed_stub.outcomes = {"UCdead": 404, "UCidle": []}

    yt_mod.scan_youtube_channels()

    persisted = _state_json()["youtube_coverage"]
    assert persisted == {"fetched": 1, "attempted": 2}, (
        "one feed failed every attempt, so the state file must read 1 ok / 2 total")
    # The denominator is channels that produced an outcome, not lines of config:
    # nothing here was skipped for a missing id, so attempted equals the config size.
    assert feed_stub.hits.count("UCdead") == yt_mod.FETCH_ATTEMPTS, (
        "the failing channel must be retried on the wire, not just in intent")


# --- Backlog #1281: one cumulative backoff budget per scan ---------------------
#
# Measured 2026-09-20T03:36Z with the feed endpoint blacked out: two dead
# channels cost 100 s, so ~50 s per dead channel — 35 s of sleeping at 5/10/20 s
# plus 4 fast 404 round-trips plus the 0.2 s inter-channel pace. Across the 64
# channels in config/youtube-channels.yml the sleeping alone is 64 × 35 = 2240 s,
# already past the 1800 s `timeout_seconds` of autonomy task #30, the only job
# that runs this pipeline, so a blackout killed the run mid-stage and it wrote
# nothing to the vault. These tests pin the bound, not the hand lever
# (`INTEL_YOUTUBE_RETRY_WAIT_SECONDS`) that was being set per-run to dodge it.


def _shipped_channel_count() -> int:
    """How many channels the shipped config declares, counted as triage counted it.

    `load_youtube_channels_config` is monkeypatched in the scan tests and resolves
    from `Path.home()` in production, so a bound over "the full config" has to
    count the file the scheduled run actually reads: 64 lines carry `channel_id:`.
    """
    cfg = INTEL_DIR / "config" / "youtube-channels.yml"
    return sum(1 for line in cfg.read_text().splitlines() if "channel_id:" in line)


def _dead_channels_scan(monkeypatch, feed_stub, n, budget):
    """Arm a scan over `n` unreachable channels with `budget` seconds to sleep in.

    Waits are captured on the module-level `sleep` so the ladder is observable
    without wall clock; the retry-wait and budget env overrides are removed so the
    scenario is the default ladder under a declared budget, and the stub's `hits`
    are the record of how many requests each channel really got.
    """
    monkeypatch.setattr(yt_mod, "RSS_FEED_URL", _stub_rss_url(feed_stub))
    monkeypatch.delenv(yt_mod.RETRY_WAIT_ENV, raising=False)
    monkeypatch.delenv(yt_mod.BACKOFF_BUDGET_ENV, raising=False)
    monkeypatch.setattr(yt_mod, "BACKOFF_BUDGET_SECONDS", budget)
    waits = []
    monkeypatch.setattr(yt_mod, "sleep", waits.append)
    ids = tuple(f"UCdead{n}" for n in range(n))
    monkeypatch.setattr(yt_mod, "load_youtube_channels_config", lambda: _channels(*ids))
    feed_stub.outcomes = {i: 404 for i in ids}
    return ids, waits


def test_the_default_backoff_budget_keeps_a_full_blackout_under_the_task_timeout(monkeypatch):
    """Clause 2: the worst case is now a declared number, and it fits in 1800 s.

    A scan stops sleeping once cumulative waiting reaches the module budget, so
    the most it can sleep over the whole shipped config is
    `min(channels × per-channel ladder, BACKOFF_BUDGET_SECONDS)` — 240 s today,
    against the 64 × 35 = 2240 s the unbounded ladder would spend.
    """
    monkeypatch.delenv(yt_mod.RETRY_WAIT_ENV, raising=False)
    monkeypatch.delenv(yt_mod.BACKOFF_BUDGET_ENV, raising=False)

    channels = _shipped_channel_count()
    assert channels >= 64, f"expected the 64-channel config, counted {channels}"
    ladder_per_dead_channel = sum(
        yt_mod.RETRY_BASE_WAIT_SECONDS * (2 ** n) for n in range(yt_mod.FETCH_ATTEMPTS - 1))
    assert ladder_per_dead_channel == 35.0, (
        f"a dead channel's ladder is the 5/10/20 s the comments measure, got "
        f"{ladder_per_dead_channel} s")
    unbounded = channels * ladder_per_dead_channel
    assert unbounded > 1800, (
        f"{unbounded} s of unbounded sleeping no longer exceeds task #30's timeout, so "
        "the budget is no longer what bounds a blackout — re-derive this clause")
    bounded = min(unbounded, yt_mod.BACKOFF_BUDGET_SECONDS)
    assert bounded < 1800, (
        f"max sleeping across a scan is {bounded} s, not under the 1800 s that a "
        "scheduled run of #30 is allowed")

    comment = Path(yt_mod.__file__).read_text()
    assert "37 minutes" not in comment, (
        "the module still states the unbounded ladder as the cost a blackout pays")
    assert f"{yt_mod.BACKOFF_BUDGET_SECONDS:g} s" in comment, (
        "the module comment must state the bounded figure this test just computed")


def test_a_blackout_stops_backing_off_once_the_cumulative_budget_is_spent(
        redirect_paths, monkeypatch, feed_stub):
    """Clause 1: sleeping is bounded by the budget, not by the channel count.

    Twelve unreachable channels at the default ladder would sleep 12 × 35 = 420 s.
    With a 20 s budget the scan may sleep at most 20 s in total — 5 s and 10 s
    still fit, so the ladder runs while it fits, and the 20 s wait that would
    breach the bound is refused instead of taken. Every channel after the refusal
    is asked once on the wire, which is the count the stub recorded, not one the
    scanner asserts about itself.
    """
    ids, waits = _dead_channels_scan(monkeypatch, feed_stub, 12, budget=20.0)

    items, coverage = yt_mod.scan_youtube_channels()

    assert sum(waits) <= 20.0, (
        f"the scan slept {sum(waits)} s against a 20 s budget: {waits}")
    backoff_waits = [w for w in waits if w >= yt_mod.RETRY_BASE_WAIT_SECONDS]
    assert backoff_waits == [5.0, 10.0], (
        f"backoff must run while the budget lasts and stop at the wait that will not "
        f"fit it, got {backoff_waits}")
    assert feed_stub.hits.count(ids[0]) == 3, (
        f"the first dead channel gets the attempts the budget still allowed, got "
        f"{feed_stub.hits.count(ids[0])} of {yt_mod.FETCH_ATTEMPTS}")
    one_attempt = [i for i in ids[1:] if feed_stub.hits.count(i) == 1]
    assert len(one_attempt) == 11, (
        f"channels after the budget fired must each be tried once, got {one_attempt}")
    assert len(feed_stub.hits) == 14, (
        f"3 + 11 single-attempt requests, not 12 × 4: {len(feed_stub.hits)} requests")
    assert items == []
    assert (coverage.fetched, coverage.attempted) == (0, 12)


def test_firing_the_budget_says_which_channels_it_reduced_and_keeps_the_denominator(
        redirect_paths, monkeypatch, feed_stub, capsys):
    """Clause 3: the bound is legible in the run and coverage keeps every channel.

    A reader of the run record has to be able to tell "we stopped asking" from
    "the endpoint is dead", and the degraded signal still needs the full
    denominator — a budget that quietly dropped channels from `attempted` would
    turn a blackout into a healthy-looking 0 of 1.
    """
    ids, _waits = _dead_channels_scan(monkeypatch, feed_stub, 12, budget=20.0)

    _items, coverage = yt_mod.scan_youtube_channels()

    out = capsys.readouterr().out
    reduced = sum(1 for i in ids if feed_stub.hits.count(i) == 1)
    assert reduced == 11, "the wire must show 11 channels reduced to one attempt"
    assert "stopped backing off" in out, out[-1500:]
    assert f"{reduced} remaining channel" in out, (
        f"the budget line must name how many channels it reduced; expected {reduced}\n"
        + out[-1500:])
    assert coverage.describe() == "fetched 0 feeds of 12 attempted", (
        "every tried channel stays in the denominator, so the CLI's degraded line "
        "still sees a blackout and not a 0-of-1 quiet night")


def test_a_scan_that_stays_inside_the_budget_retries_exactly_as_before(
        redirect_paths, monkeypatch, capsys):
    """Clause 4: the budget bounds the cumulative wait, never a channel inside it.

    Triage measured 8/12 channels recovering on one attempt and 10/12 within four
    (youtube_scanner.py:66-68), so recovery is the thing the budget must not
    touch. Here a dead channel burns 35 s of the default 240 s budget and a second
    channel still recovers on its 3rd attempt with the untouched 5/10 s waits —
    and nothing prints about the budget, because nothing reached it.
    """
    monkeypatch.delenv(yt_mod.RETRY_WAIT_ENV, raising=False)
    monkeypatch.delenv(yt_mod.BACKOFF_BUDGET_ENV, raising=False)
    waits = []
    monkeypatch.setattr(yt_mod, "sleep", waits.append)
    hits = {}

    def flaky_get(url, headers=None, timeout=None):
        channel = url.split("channel_id=")[-1]
        hits[channel] = hits.get(channel, 0) + 1
        if channel == "UCflaky" and hits[channel] < 3:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if channel == "UCflaky":
            return _atom_body(["abc123"])
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(yt_mod, "_http_get", flaky_get)
    monkeypatch.setattr(yt_mod, "load_youtube_channels_config",
                        lambda: _channels("UCdead", "UCflaky"))

    items, coverage = yt_mod.scan_youtube_channels()

    out = capsys.readouterr().out
    assert hits == {"UCdead": 4, "UCflaky": 3}, (
        f"a dead channel still gets all four attempts and a flaky one still gets "
        f"three inside an unspent budget, got {hits}")
    assert waits == [5.0, 10.0, 20.0, yt_mod.CHANNEL_PACING_SECONDS, 5.0, 10.0], (
        f"the ladder and the pace must be unchanged while the budget lasts, got {waits}")
    assert sum(waits) == pytest.approx(50.2), (
        "35 s of the dead channel plus one pace plus 15 s of the flaky channel")
    assert [i.title for i in items] == ["Video abc123"], (
        "a feed that recovers on the 3rd attempt must still yield its video")
    assert (coverage.fetched, coverage.attempted) == (1, 2)
    assert "stopped backing off" not in out, (
        "an unspent budget must not print a reduction line")


def test_a_blackout_run_over_the_cli_fires_the_budget_and_still_fails_on_coverage(
        tmp_path, feed_stub):
    """Clause 3 and 5 across the process seam: the bound, then the degraded exit.

    Five channels, none reachable, and a 0.3 s budget: the budget has to fire
    inside the run, and firing it must not soften the #739 signal — the run still
    prints `YouTube stage DEGRADED` with both counts after the writer and exits
    non-zero (it is deliberately exit 1, so the skill forbids wrapping this
    command as `a || b || c`).
    """
    home, _feeds = _cli_home(tmp_path)
    _cli_channels(home, "UCa", "UCb", "UCc", "UCd", "UCe")
    feed_stub.outcomes = {i: 404 for i in ("UCa", "UCb", "UCc", "UCd", "UCe")}
    env = _cli_feed_env(feed_stub)
    env["INTEL_YOUTUBE_BACKOFF_BUDGET_SECONDS"] = "0.3"

    _day, proc = _run_cli(home, 6, "--scan", extra_env=env)

    assert "stopped backing off" in proc.stdout, proc.stdout[-2500:]
    assert "4 remaining channel" in proc.stdout, (
        "the first channel spent the budget, the other four are one-attempt each\n"
        + proc.stdout[-2500:])
    assert proc.returncode != 0, (
        f"firing the budget must not rescue a blackout run into exit 0\n{proc.stdout[-2500:]}")
    assert "Pipeline Complete" not in proc.stdout, proc.stdout[-2500:]
    lowered = proc.stdout.lower()
    assert "degraded" in lowered, proc.stdout[-2500:]
    assert "0 feeds of 5 attempted" in lowered, (
        f"the degraded line must still name both counts\n{proc.stdout[-2500:]}")
    assert feed_stub.hits.count("UCa") == 3, (
        "the budget fires after the waits that fit, on the wire as well as in the "
        f"printout; UCa got {feed_stub.hits.count('UCa')} requests")
    assert all(feed_stub.hits.count(i) == 1 for i in ("UCb", "UCc", "UCd", "UCe")), (
        f"later channels are one attempt each: {[feed_stub.hits.count(i) for i in ('UCb', 'UCc', 'UCd', 'UCe')]}")


def _cli_channels(home, *ids):
    """Write the channels config where the scanner looks for it, under scratch HOME."""
    cfg = home / "lloyd" / "scripts" / "intel-pipeline" / "config" / "youtube-channels.yml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("channels:\n" + "".join(
        f"  - channel_id: {i}\n    name: {i}\n    handle: '@{i}'\n" for i in ids))


def _cli_feed_env(feed_stub):
    """Point the CLI's YouTube scanner at the stub, with retry waits that are not 5 s.

    Both are read as environment by the package, which is what lets the *real*
    `python -m intel_pipeline` process — not an in-process call — talk to the stub.
    """
    return {"INTEL_YOUTUBE_RSS_URL": _stub_rss_url(feed_stub),
            "INTEL_YOUTUBE_RETRY_WAIT_SECONDS": "0.05"}


def test_a_run_that_lost_most_feeds_fails_and_names_both_counts(tmp_path, feed_stub):
    """Clause 4, over the CLI seam: exit status plus a line a reader can act on.

    Three channels, two unreachable: coverage 1/3 is strictly under half. Before
    this the same run printed `YouTube scanner: 0 items` and `=== Pipeline
    Complete ===` and exited 0 (backlog #739, measured on 2026-09-10 at 1/64).
    """
    home, _feeds = _cli_home(tmp_path)
    _cli_channels(home, "UCa", "UCb", "UCc")
    feed_stub.outcomes = {"UCa": ["vid_a"], "UCb": 404, "UCc": 500}

    _day, proc = _run_cli(home, 6, "--scan", extra_env=_cli_feed_env(feed_stub))

    assert proc.returncode != 0, (
        f"a run that reached 1 of 3 feeds must not exit 0\n{proc.stdout[-2500:]}")
    assert "Pipeline Complete" not in proc.stdout, (
        "the success banner is what made a blackout indistinguishable\n"
        + proc.stdout[-2500:])
    lowered = proc.stdout.lower()
    assert "degraded" in lowered, proc.stdout[-2500:]
    assert "1" in lowered and "3" in lowered, (
        f"the degraded line must name both counts\n{proc.stdout[-2500:]}")


def test_a_healthy_run_still_exits_zero_and_still_prints_complete(tmp_path, feed_stub):
    """Clause 5: the new failure signal must not fire on a partially-idle day.

    Every configured feed answers 200; two of the three hold no entries, which is
    what an idle channel looks like and is not degradation. Coverage 3/3.
    """
    home, feeds = _cli_home(tmp_path)
    _cli_channels(home, "UCa", "UCb", "UCc")
    feed_stub.outcomes = {"UCa": ["vid_a"], "UCb": [], "UCc": []}

    _day, proc = _run_cli(home, 6, "--scan", extra_env=_cli_feed_env(feed_stub))

    assert proc.returncode == 0, (
        f"three feeds fetched, two idle, must be a normal run\n{proc.stdout[-2500:]}\n"
        f"{proc.stderr[-1500:]}")
    assert "Pipeline Complete" in proc.stdout, proc.stdout[-2500:]
    # The subprocess wrote its state under the scratch HOME, not the in-process
    # `state_mod.STATE_FILE`, so the coverage key is read from where it landed.
    persisted = json.loads((feeds / "scanner-state.json").read_text())
    assert persisted["youtube_coverage"] == {"fetched": 3, "attempted": 3}


# --- Backlog #856: a keyword matches whole words, not substrings --------------

# The `interests.md` topic lines verbatim as of vault commit `752b85e6^` — the
# document one commit *before* #852 appended five bare AI words to `## AI & LLMs`
# — so 64 keywords, duplicates included, loaded by the real loader. It is pinned
# at that shape rather than re-synchronised with today's vault file: every count in
# this section (36 containment admits → 33 whole-word keeps) was measured against
# it, and the #852 section below compares its own copy against it, so the two
# documents differ by exactly the change #852 landed and by nothing else.
# `PROFILE_MD` above cannot
# express this defect: none of its keywords is short enough to hide inside an
# English word, and the defect is exactly that — the robotics keyword `DOF`
# matched the `dof` inside "handoff", which admitted an openclaw Linux-update
# commit, filed it under `robotics`, and scored it 10/10 urgent on the keyword
# path.
INTERESTS_LIVE_MD = """---
title: Interests
---
# Interests

## Robotics
humanoid,actuator,servo,DOF,gait,locomotion,bipedal,quadruped,legged,robot dog,\
unitree,go2,gripper,compliant mechanism,3d printed robot,ros2,gr00t,isaac lab,\
sim to real,imitation learning,behavior cloning,whole-body control,\
dexterous manipulation,gr00t,unitree,nvidia omniverse,nvidia isaacsim,\
unitree go2,unitree r1,unitree g1

## AI & LLMs
qwen,vllm,quantization,gguf,inference,speculative decoding,mixture of experts,\
moe,local llm,llama.cpp,mcp protocol,agent framework,tool use,function calling,\
agentic,orchestration,foundation model,reinforcement learning,embodied ai,\
openclaw

## Voice & TTS
text to speech,tts,voice cloning,voice synthesis,qwen-tts,speech synthesis,\
speech recognition,asr

## Hardware
3d printing,PETG,planetary gear,actuator design,printed robotics,inmoov
"""

# Frozen 2026-09-14/15/16 raw corpus: the items the containment rule admitted,
# with `provenance` and the counts it was cut to. It is a file rather than the
# live `_pipeline` tree because that tree keeps growing — triage counted 21
# admits over the same three dates on 09-16, mid-write, while the settled files
# admit 36 of 267. JSONL, not JSON: `.gitignore` ignores `*.json` and is a
# denied path, so a `.json` fixture would sit untracked and the test would only
# pass in the worktree that generated it.
KEYWORD_ADMITS_FIXTURE = (REPO_ROOT / "tests" / "fixtures" / "intel_pipeline"
                          / "keyword_admits_2026-09-14_16.jsonl")

# The two items that reached the pipeline *only* because a keyword was a
# substring of a bigger word, plus the one true positive that must survive.
HANDOFF_CRON_ITEM = "github:openclaw/openclaw:commit:eec1712a"
HANDOFF_UI_ITEM = "github:openclaw/openclaw:commit:ead35525"
OPENCLAW_ENV_KEYS_ITEM = "github:openclaw/openclaw:issue:146645"
AGIBOT_DOF_ITEM = "github:isaac-sim/IsaacLab:issue:7789"

AGIBOT_TITLE = "Agibot/g2 ik7d teleop"
AGIBOT_SUMMARY = ("Integrates AgiBot's `ik_7d` 7-DoF redundant-arm IK solver as an "
                  "out-of-tree Isaac Lab teleoperation environment "
                  "(`Isaac-Teleop-G2-Ik7d-v0`).")
HANDOFF_CRON_SUMMARY = (
    "* fix(cron): honor tool allowlists across harnesses\n"
    "* test(update): simplify managed handoff fixture\n"
    "* test(update): preserve timeout narrowing in fixture cleanup")


def _live_profile(tmp_path):
    """The live keyword lists, through the real loader, from a redirected HOME."""
    (tmp_path / "obsidian" / "interests.md").write_text(INTERESTS_LIVE_MD)
    return profile_mod.load_profile()


def _fixture_corpus():
    """Header record first, then one item per line — see the fixture's own comment."""
    lines = KEYWORD_ADMITS_FIXTURE.read_text().splitlines()
    header = json.loads(lines[0])
    header["items"] = [json.loads(l) for l in lines[1:] if l.strip()]
    return header, [FeedItem.from_dict(r) for r in header["items"]]


def _containment_admits(items, profile):
    """The pre-fix rule, restated so the delta is measured and not remembered."""
    keywords = profile_mod.get_all_keywords(profile)
    return [it for it in items
            if any(k.lower() in f"{it.title} {it.summary}".lower() for k in keywords)]


def test_a_short_keyword_buried_in_a_bigger_word_stops_matching(redirect_paths):
    """Clause 1: the commit message that fired the defect matches nothing.

    Pre-fix this returned the robotics topic with `matched_keywords: ['DOF']`,
    which is what made an openclaw update look like a robotics item.
    """
    profile = _live_profile(redirect_paths)

    matched = profile_mod.keyword_match(
        "fix(update): allow config and service locks before handoff storage "
        "is initiated", profile)

    assert matched == [], (
        "a keyword embedded inside the word 'handoff' still matched: "
        f"{[(t['name'], t['matched_keywords']) for t in matched]}")


def test_a_whole_word_dof_still_matches_the_robotics_topic(redirect_paths):
    """Clause 2: the boundary rule must cost no true positive.

    Both texts are live. `AGIBOT_TITLE`/`AGIBOT_SUMMARY` is the corpus's one
    genuine `DoF` hit (2026-09-15, written as `7-DoF`), and the hyphenated
    `6-DoF` is the spelling the field writes the spec in.
    """
    profile = _live_profile(redirect_paths)

    matched = profile_mod.keyword_match(f"{AGIBOT_TITLE} {AGIBOT_SUMMARY}", profile)
    robotics = [t for t in matched if t["name"] == "robotics"]
    hyphenated = profile_mod.keyword_match("a 6-DoF redundant arm", profile)

    assert robotics, f"the genuine DoF item stopped matching: {matched}"
    assert "DOF" in robotics[0]["matched_keywords"]
    assert [t["matched_keywords"] for t in hyphenated] == [["DOF"]], (
        "the hyphenated spelling must match the robotics topic and nothing else")


def test_a_multi_word_phrase_still_matches_without_a_boundary(redirect_paths):
    """Clause 3: phrases keep plain containment, so prose inflection is not a miss.

    `\\brobot dog\\b` cannot match "robot dogs"; the space is already the
    boundary, and a phrase split by a hyphen ("sim-to-real") is prose too.
    """
    profile = _live_profile(redirect_paths)

    matched = profile_mod.keyword_match(
        "everything we know about robot dogs in 2026", profile)

    assert [t["name"] for t in matched] == ["robotics"]
    assert matched[0]["matched_keywords"] == ["robot dog"]


def test_a_punctuation_bearing_keyword_still_matches(redirect_paths):
    """`llama.cpp` and `qwen-tts` are the profile's keywords with punctuation
    inside. Their edge characters are word characters, so a `\\b` at each end
    bounds the whole keyword — the shape the boundary rule is claimed for.
    """
    profile = _live_profile(redirect_paths)
    keywords = profile_mod.get_all_keywords(profile)

    assert "llama.cpp" in profile_mod.match_keywords(
        "quantising llama.cpp models for the desktop", keywords)
    assert "qwen-tts" in profile_mod.match_keywords(
        "qwen-tts finished the clone in 2.3s", keywords)


def test_stage1_filter_admits_every_item_that_matched_on_a_whole_word(redirect_paths):
    """Clause 4: over the frozen 09-14→09-16 corpus, admission drops from 36 to
    33 and the three losses are the substring-only items — no more, no less.

    The denominator is the fixture's own count, re-measured against the pre-fix
    rule in the same call, so a fixture that quietly changed size cannot leave
    this test green.
    """
    profile = _live_profile(redirect_paths)
    payload, items = _fixture_corpus()

    kept = scoring_mod.stage1_filter(items, profile)
    contained = _containment_admits(items, profile)
    lost = {it.id for it in items} - {it.id for it in kept}

    assert payload["matched_keywords"] == profile_mod.get_all_keywords(profile), (
        "the keyword list the corpus was frozen against and the keyword list "
        "INTERESTS_LIVE_MD loads are no longer the same list, so the counts "
        "below describe a different profile")
    assert payload["counts"]["containment_admits"] == 36 == len(contained), (
        f"the pre-fix rule admits {len(contained)} of {len(items)} fixture items, "
        f"not the 36 the fixture records")
    assert len(kept) == 33, (
        f"admission must drop 36 → 33, losing only substring-only items; "
        f"{len(kept)} kept, lost {sorted(lost)}")
    assert lost == {HANDOFF_CRON_ITEM, HANDOFF_UI_ITEM, OPENCLAW_ENV_KEYS_ITEM}, (
        f"the three substring-only admits must be the only losses: {sorted(lost)}")
    assert AGIBOT_DOF_ITEM in {it.id for it in kept}, (
        "the genuine 7-DoF item must still be admitted")


def test_a_substring_only_item_scores_below_the_vault_floor_without_the_model(
        redirect_paths, monkeypatch):
    """Clause 5: on the keyword-fallback path the fabricated match *was* the score.

    Pre-fix, with `INTEL_DISABLE_LLM=1`, this item came back `relevance 10 |
    urgency urgent | category robotics | why 'Matches: DOF'` — above
    `RELEVANCE_FLOOR = 4`, so it would have been written to the vault under
    `knowledge/robotics/`. The model overwrites these fields when it answers
    (clause 5 is about the path where it does not).
    """
    profile = _live_profile(redirect_paths)
    monkeypatch.setenv("INTEL_DISABLE_LLM", "1")
    item = _item(HANDOFF_CRON_ITEM, source="github",
                 title="fix(cron): honor tool allowlists across harnesses (#149375)",
                 summary=HANDOFF_CRON_SUMMARY)

    scored = scoring_mod.stage2_score([item], profile)[0]

    assert vw_mod.below_floor(scored), (
        f"a substring-only match still clears the floor: {scored.relevance}/10 "
        f"{scored.urgency} {scored.category} — {scored.why}")
    assert scored.relevance < vw_mod.RELEVANCE_FLOOR
    assert scored.urgency == "low"
    assert scored.category == "general"
    assert "DOF" not in scored.why


def test_a_whole_word_item_still_scores_urgent_on_the_keyword_path(redirect_paths,
                                                                   monkeypatch):
    """The control clause 5 needs: below-floor must not mean everything-scores-1.

    The same env-disabled path, on the real robotics item: robotics has no
    declared weight, so the keyword fallback is weight 1.0 × 10 = 10, urgent,
    category robotics, and it is written.
    """
    profile = _live_profile(redirect_paths)
    monkeypatch.setenv("INTEL_DISABLE_LLM", "1")
    item = _item(AGIBOT_DOF_ITEM, source="github",
                 title=AGIBOT_TITLE, summary=AGIBOT_SUMMARY)

    scored = scoring_mod.stage2_score([item], profile)[0]

    assert scored.relevance == 10 and not vw_mod.below_floor(scored)
    assert scored.urgency == "urgent" and scored.category == "robotics"
    # generate_why emits `list(set(...))`, whose order varies with the process
    # hash seed, so the set is what is pinned — this item's text carries both
    # `DOF` and `Isaac Lab`.
    assert scored.why.startswith("Matches: ")
    assert set(scored.why[len("Matches: "):].split(", ")) == {"isaac lab", "DOF"}


def test_a_vault_path_no_longer_files_a_substring_only_item_under_robotics(redirect_paths):
    """The other route a fabricated match took: `determine_vault_path` also
    consults `keyword_match`, so a false robotics match misfiles an unrelated
    YouTube note into `knowledge/robotics/`."""
    profile = _live_profile(redirect_paths)
    substring_item = ScoredItem(
        id="yt-handoff", source="youtube",
        title="Fixing handoff storage locks in the update service",
        url="https://example.com/yt-handoff",
        summary="why managed updates fail during service activation",
        discovered_at="2026-09-11T00:00:00Z", relevance=6, urgency="morning",
        why="Matches: DOF", category="robotics")
    whole_word_item = ScoredItem(
        id="yt-dof", source="youtube", title="A 6-DoF arm teleop session",
        url="https://example.com/yt-dof", summary="teleop retargeting",
        discovered_at="2026-09-11T00:00:00Z", relevance=6, urgency="morning",
        why="Matches: DOF", category="robotics")

    substring_path = vw_mod.determine_vault_path(substring_item, profile)
    whole_word_path = vw_mod.determine_vault_path(whole_word_item, profile)

    assert substring_path.parts[-2:] == ("feeds", "youtube-uncategorized.md"), (
        f"a substring-only match still routes to {substring_path}")
    assert whole_word_path.parts[-2:] == ("robotics", "youtube-digest.md"), (
        f"the genuine match must keep its topic note, got {whole_word_path}")


def test_the_cli_drops_a_substring_only_item_before_the_day_file(tmp_path):
    """The process boundary the unit tests cannot cross: the `python -m
    intel_pipeline` subprocess autonomy task #30 spawns, its own HOME, the
    interests file read from disk by the real loader, and the keyword-fallback
    path with the engine off. One raw day, two items — only the genuine 7-DoF
    one may reach the day file.
    """
    home, feeds = _cli_home(tmp_path)
    (home / "obsidian" / "interests.md").write_text(INTERESTS_LIVE_MD)
    today = _today_str()
    (feeds / "raw" / f"{today}.jsonl").write_text("\n".join([
        _item(HANDOFF_CRON_ITEM, source="github",
              title="fix(cron): honor tool allowlists across harnesses (#149375)",
              summary=HANDOFF_CRON_SUMMARY).to_json(),
        _item(AGIBOT_DOF_ITEM, source="github",
              title=AGIBOT_TITLE, summary=AGIBOT_SUMMARY).to_json(),
    ]) + "\n")

    day, proc = _run_cli(home, 6, "--score", extra_env={"INTEL_DISABLE_LLM": "1"})

    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = [json.loads(l) for l in
            (feeds / f"intel-{day}.jsonl").read_text().splitlines() if l.strip()]

    assert [r["id"] for r in rows] == [AGIBOT_DOF_ITEM], (
        "the handoff-substring item must be dropped before scoring, and the "
        f"genuine DoF item must survive: {rows}\n{proc.stdout[-2000:]}")
    assert (rows[0]["relevance"], rows[0]["category"]) == (10, "robotics"), (
        f"the surviving item keeps its keyword-fallback grade: {rows[0]}")


# --- Backlog #852: the keyword list has to reach AI-aggregate titles -----------
#
# `## AI & LLMs` in `~/obsidian/interests.md` listed only compound phrases ("agent
# framework", "mcp protocol", "local llm"), and `keyword_match` tests each listed
# phrase literally, so a well-formed AI headline that never spelled a phrase out
# was dropped by `stage1_filter` before `stage2_score` — the model #570 exists to
# add — could ever see it. Measured on the settled raw day files beforehand: stage 1
# kept 0 of 35 YouTube items on 09-10 and 0 of 9 on 09-17, and all 18 YouTube
# titles carrying an AI word across 09-10/09-13/09-16 matched zero keywords — on the
# title alone *and* on title + summary, which is the text `stage1_filter` tests.
# YouTube is where it bites hardest, because #1155 leaves those rows with an empty
# summary, so the title is the only thing that can match.
#
# Vault commit `752b85e6` (2026-09-20) appended `agent`, `llm`, `mcp`, `openai`,
# `ai` to that line — 69 keywords, 67 distinct — and the whole-word rule #856
# landed is what makes listing a two-letter keyword like `ai` safe at all. The same
# probes afterwards: YouTube kept 8 of 24 on 09-16 (was 3) and 4 of 9 on 09-17
# (was 0), and the 8 AI-titled 09-09 items that printed False all print True.
#
# What is pinned below is the code half the vault edit depends on: the loader
# reading the real file, the gate's keep/drop on the two titles the acceptance
# names, and the `python -m intel_pipeline` process autonomy task #30 spawns. No
# `**Weight:**` is set anywhere here — a weight below 0.4 skips the stage-2 model
# call *and* cannot clear `vault_writer.RELEVANCE_FLOOR`, so it is an off switch,
# and relative weighting is Alan's call (#570 defect 3 says the same about the
# vocabulary itself, which is why the needs-human half of this item stays open
# after these tests land).

# INTERESTS_LIVE_MD plus the five words. Building the widened document as a
# substitution on the narrower one — rather than pasting a second interests.md —
# is what makes each comparison below one vocabulary against its own predecessor.
INTERESTS_852_MD = INTERESTS_LIVE_MD.replace(
    "reinforcement learning,embodied ai,openclaw\n",
    "reinforcement learning,embodied ai,openclaw,agent,llm,mcp,openai,ai\n")

# The five, and nothing more. `agi` and `skill` were deliberately not added: #570
# defect 3 warns that a broad AI vocabulary admits AI-headline noise, so widening
# this list further is a decision about what the pipeline cares about, not a
# recall fix.
NEW_AI_WORDS = ("agent", "llm", "mcp", "openai", "ai")

# The three titles the acceptance names, each one a real feed row with the
# `id`, empty `summary` (#1155) and channel-derived `source_tags` the scanner
# wrote. "Explains" and "Training" both contain `ai` inside a word — they are the
# two junk items the triage measured when it warned that the cheap half of this fix
# was unsafe without a boundary rule. "Why green rebar?" (2026-09-09, a YouTube
# Short about concrete reinforcement) matches no keyword under either vocabulary,
# and it arrives carrying a `source_tag`, which is the field #570 found opening the
# gate for every item a scanner emitted.
MCP_APPS_ITEM = "youtube:UCLKPca3kwwd-B59HNr-_lvA:waI44NP1abk"
MCP_APPS_TITLE = "Rebuilding the web for agents — Liad Yosef, MCP Apps"
LANTERN_ITEM = "youtube:UCTAgbu2l6_rBKdbTvEodEDw:coGsFZBfFz4"
LANTERN_TITLE = "LANTERNS Explains Why Guy Gardner Is the Green Lantern"
GREEN_REBAR_ITEM = "youtube:UCKqKiBDpq9j29bXbBx0cfOw:H-3SHxqsXlQ"
GREEN_REBAR_TITLE = "Why green rebar?"

# All 8 titles the item's probe names — every 2026-09-09 YouTube title carrying an
# AI word that matched zero keywords, verbatim from
# `_pipeline/.../feeds/intel-2026-09-09.jsonl`. Each row's `summary` is empty
# (#1155), so the title is the whole match text.
PROBE_09_09_TITLES = [
    "AI Just Did the Impossible: Reversed Human Aging",
    "Should you ask Astra to do this? #AGI #thisisAGI #openai #astra",
    "OpenAI JUST solved math....",
    "Abstraction Agent: LLM Engineers New State Space Geometry",
    "The Universal Remote Control for AI — Alex Hancock, Block",
    "MCP Apps: Give the Model Data, Give the User a UI — Dustin Mihalik, Indeed",
    "How long can your skills be before your agent forgets what you told it? "
    "— Laurie Voss, Arize AI",
    "500 Skills, Zero Fine-Tuning: LinkedIn's Playbook for AI Agents "
    "— Ajay Prakash, LinkedIn",
]


def _ai852_profile(tmp_path):
    """`load_profile()` on the interests.md #852 landed, from a redirected HOME."""
    (tmp_path / "obsidian" / "interests.md").write_text(INTERESTS_852_MD)
    return profile_mod.load_profile()


def _feed_video(item_id, title, tags):
    """One of the three real rows above, reconstructed with the fields the feed
    row actually carries: no summary, and the channel's own source tag."""
    return _item(item_id, source="youtube", title=title, summary="", tags=[tags])


@pytest.mark.live_vault
def test_the_interests_md_the_pipeline_loads_carries_the_bare_ai_words():
    """Clause 1, through the real loader from the real vault file.

    Not from a copy in this file: the defect was that the vocabulary *on disk*
    could not reach an AI-aggregate title, so the assertion has to run against the
    file `load_profile()` reads. Marked `live_vault` because the gate's hard
    `tests` rung deselects it — interests.md is under a person's pen between
    rounds, and this is the one assertion here that is not closed under the diff.
    Each of the five must also sit on the `## AI & LLMs` topic specifically: a bare
    word listed only under another topic files an AI item in the wrong note.
    """
    profile = profile_mod.load_profile()
    keywords = {k.lower() for k in profile_mod.get_all_keywords(profile)}
    ai_line = {k.lower() for k in
               [t["keywords"] for t in profile["topics"]
                if t["name"] == "ai-llms"][0]}

    missing = [w for w in NEW_AI_WORDS if w not in keywords]
    assert not missing, (
        f"interests.md loads without {missing}, so a title like "
        f"{MCP_APPS_TITLE!r} is dropped by stage1_filter before stage2_score ever "
        "sees it")
    off_topic = [w for w in NEW_AI_WORDS if w not in ai_line]
    assert not off_topic, (
        f"{off_topic} must belong to the `## AI & LLMs` topic, not merely to some "
        f"topic, or the item is filed under the wrong note: {sorted(ai_line)[-8:]}")


def test_the_five_bare_ai_words_match_their_own_words_and_nobody_elses(
        redirect_paths):
    """Clause 2 under the widened vocabulary: `ai` is a keyword now, and the
    whole-word rule is the only reason that is not a substring wildcard.

    `ai` (2 characters) and `mcp` (3) are the shapes the clause names. The two junk
    titles both contain `ai` inside a word and must match nothing; `local llm` must
    still match as a phrase, so the boundary rule was not bought by breaking every
    multi-word keyword.
    """
    profile = _ai852_profile(redirect_paths)
    keywords = profile_mod.get_all_keywords(profile)

    assert len(keywords) == 69, (
        "this file's copy of interests.md must hold INTERESTS_LIVE_MD's 64 "
        f"keywords plus the five, or it is not the vocabulary #852 landed: {len(keywords)}")

    on_topic = profile_mod.keyword_match(
        "Connect AI to Billions of Legal Documents", profile)
    assert [t["matched_keywords"] for t in on_topic] == [["ai"]], (
        f"the two-letter keyword must match its own word and no other topic: {on_topic}")
    assert on_topic[0]["name"] == "ai-llms"
    assert profile_mod.keyword_match("MCP Apps for the browser", profile)[0][
        "matched_keywords"] == ["mcp"], (
        "the three-letter keyword must match its own word")

    for junk, item_id in ((GREEN_REBAR_TITLE, GREEN_REBAR_ITEM),
                          (LANTERN_TITLE, LANTERN_ITEM),
                          ("Training Taste", "training-taste")):
        assert profile_mod.keyword_match(junk, profile) == [], (
            f"{junk!r} is the item the containment rule admitted via `ai` inside a "
            f"word ({item_id}); it must still match nothing")

    phrase = profile_mod.keyword_match(
        "quantising a local llm for the desktop", profile)
    assert [t["matched_keywords"] for t in phrase] == [["local llm", "llm"]], (
        "the existing phrase must still match verbatim — joined here by the new "
        f"bare `llm` inside it, which is the expected overlap: {phrase}")


def test_the_item_probe_titles_that_printed_false_now_all_match(redirect_paths):
    """The 8-item probe this item was filed on, re-run against the widened list.

    These are the titles from `feeds/intel-2026-09-09.jsonl` whose `False` the item
    recorded ("all 21 YouTube items scored 1/10, and 8 of those 21 titles contain an
    obvious AI/agent word while matching zero keywords"). Re-running the item's own
    command against today's vault prints True for all 8; this pins the same claim
    without reaching for `_pipeline`, which keeps moving.
    """
    profile = _ai852_profile(redirect_paths)

    unmatched = [t for t in PROBE_09_09_TITLES
                 if not profile_mod.keyword_match(t, profile)]
    assert not unmatched, (
        f"these on-topic AI titles still match no keyword: {unmatched}")


def test_the_loaded_profile_keeps_the_ai_video_and_still_drops_the_rebar(
        redirect_paths):
    """Clause 3 and the surviving half of clause 4, across the gate.

    `load_profile()` on the interests.md #852 landed, then `stage1_filter` on the
    three real feed rows. The MCP Apps title spells "agents" with an s and never
    says "agent framework" or "mcp protocol", so under INTERESTS_LIVE_MD's 64
    keywords it is dropped — the premise leg below is what stops this test passing
    if the hole ever closes somewhere else. The rebar Short is the item the gate
    exists to catch, and it carries a `source_tag`: an item whose title and summary
    match no keyword must be dropped even so.
    """
    after = _ai852_profile(redirect_paths)
    before = _live_profile(redirect_paths)
    videos = [_feed_video(MCP_APPS_ITEM, MCP_APPS_TITLE, "aiDotEngineer"),
              _feed_video(LANTERN_ITEM, LANTERN_TITLE, "Nerdist"),
              _feed_video(GREEN_REBAR_ITEM, GREEN_REBAR_TITLE, "buildwitt")]

    kept_before = {it.id for it in scoring_mod.stage1_filter(videos, before)}
    kept_after = {it.id for it in scoring_mod.stage1_filter(videos, after)}

    assert MCP_APPS_ITEM not in kept_before, (
        "the premise stopped holding: the 64-keyword vocabulary already keeps this "
        "item, so the recall hole this section pins is closed elsewhere")
    assert kept_after == {MCP_APPS_ITEM}, (
        f"only the AI video may survive stage 1; the rebar Short and the "
        f"substring-only `ai` titles must not: {sorted(kept_after)}")
    assert kept_before <= kept_after, (
        "widening a keyword list may only add admits — every new word is ANDed "
        f"with a whole-word test, never ORed with a wider match; lost "
        f"{sorted(kept_before - kept_after)}")


def test_the_cli_keeps_the_ai_video_into_the_day_file(tmp_path):
    """The process boundary the unit tests cannot cross.

    `python -m intel_pipeline --score`: the subprocess autonomy task #30 spawns,
    its own HOME, the interests file read from disk by the real loader, and the
    keyword-fallback path with the engine off. One raw day, the three real feed
    rows — only the AI video may reach the day file, and it must keep the grade the
    keyword path gives it rather than being silently down-rated on its way to the
    writer.
    """
    home, feeds = _cli_home(tmp_path)
    (home / "obsidian" / "interests.md").write_text(INTERESTS_852_MD)
    (feeds / "raw" / f"{_today_str()}.jsonl"
     ).write_text("\n".join([
         _feed_video(MCP_APPS_ITEM, MCP_APPS_TITLE, "aiDotEngineer").to_json(),
         _feed_video(LANTERN_ITEM, LANTERN_TITLE, "Nerdist").to_json(),
         _feed_video(GREEN_REBAR_ITEM, GREEN_REBAR_TITLE, "buildwitt").to_json(),
     ]) + "\n")

    day, proc = _run_cli(home, 6, "--score", extra_env={"INTEL_DISABLE_LLM": "1"})

    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = [json.loads(line) for line in
            (feeds / f"intel-{day}.jsonl").read_text().splitlines() if line.strip()]

    assert [r["id"] for r in rows] == [MCP_APPS_ITEM], (
        f"the AI video must reach the day file and the two junk-shaped ones must "
        f"be dropped before scoring: {rows}\n{proc.stdout[-2000:]}")
    assert (rows[0]["relevance"], rows[0]["category"]) == (10, "ai-llms"), (
        f"the surviving item keeps the keyword-fallback grade, filed under its own "
        f"topic: {rows[0]}")

# --- autonomy task #30's own text (#853 finding 2) -------------------------------
#
# `autonomy/30-intelligence-pipeline-scan-score.md` is the operating instruction the
# scheduled worker reads, and its prose is what clauses 3 and 4 are about. Same
# convention as tests/test_automod_doc_claims.py: fail when the vault is present and
# the claim is stale, skip only when the vault is genuinely absent.

AUTONOMY_TASK_30 = Path.home() / "obsidian" / "autonomy" / (
    "30-intelligence-pipeline-scan-score.md")


def _task_30_text() -> str:
    if not AUTONOMY_TASK_30.exists():
        pytest.skip(f"no vault checkout at {AUTONOMY_TASK_30.parent}")
    return AUTONOMY_TASK_30.read_text(encoding="utf-8")


def test_autonomy_task_30_names_the_scorer_that_ships():
    """Clause 3: the task file names the model that scores, and claims no 122B.

    `scoring.py:43-45` sends model `"primary"` to `localhost:8096/v1/chat/completions`,
    and that endpoint answers `{"id": "Qwen3.8-Flash-Next-nvfp4"}` (re-read live at
    `/v1/models` on 2026-09-22). The worker running this task reads this file, so a
    stale model name in it is what makes a later run re-derive #570 from scratch.
    """
    text = _task_30_text()

    assert "122B" not in text, (
        "the task still names the retired scorer:\n"
        + "\n".join(line for line in text.splitlines() if "122B" in line))
    assert "`primary`" in text, "the task never names the model that scores"
    assert "Qwen3.8-Flash-Next-nvfp4" in text, (
        "the task does not name the model behind `primary`")
    front = text.split("---")[1] if text.startswith("---") else ""
    assert "primary" in front, (
        "the front-matter `description:` is what the scheduler board and the MCP "
        f"reader show, and it does not name the scorer:\n{front}")


def test_autonomy_task_30_states_the_write_floor():
    """Clause 4: stage 4 and Notes state the floor gate; the stale lines are gone.

    `vault_writer.py:44` is the 4, and the writer prints
    `Held N item(s) below relevance floor 4` — the file has to say the same thing.
    """
    text = _task_30_text()

    for stale in ("writes all scored items", "No threshold"):
        assert stale not in text, (
            f"the task still teaches {stale!r}:\n"
            + "\n".join(line for line in text.splitlines() if stale in line))
    assert "vault_writer.RELEVANCE_FLOOR" in text, (
        "the task never names the constant that gates writes")
    assert "(4)" in text, "the task does not state the floor's value"


def test_autonomy_task_30_still_parses_as_scheduler_config():
    """The seam another process reads: the prose edit must not damage the schema.

    This file is live configuration — `autonomy._parse_task_file` feeds the scheduler
    and the Mission Control board. A rewrite that renames `id:` to `task_id:` or drops
    `timeout_seconds:` leaves the task dispatching on defaults, or listed on a board
    with values only one reader can see, with nothing red about it. So the schema is
    asserted, not assumed.
    """
    if not AUTONOMY_TASK_30.exists():
        pytest.skip(f"no vault checkout at {AUTONOMY_TASK_30.parent}")
    import autonomy

    parsed = autonomy._parse_task_file(AUTONOMY_TASK_30)
    assert parsed is not None, "the scheduler's parser returns nothing for task 30"
    assert not parsed.get("_yaml_broken"), (
        "task 30 parses degraded: the scheduler then recovers fields by regex, and a "
        "degraded read can silently drop the schedule or the turn budget")

    for field, expected in {"id": 30, "skill_name": "intelligence-pipeline",
                            "frequency": "daily", "timeout_seconds": 1800,
                            "agent_id": "researcher"}.items():
        assert parsed.get(field) == expected, (
            f"task 30 lost or changed {field}: want {expected!r}, "
            f"got {parsed.get(field)!r}")
    assert parsed["body"].strip(), "task 30 has no body for the worker to read"


# --- Backlog #1380: an item the call cap left ungraded is not a written item ----
#
# `stage2_score` assigned `_keyword_fallback`'s score *before* it decided whether to
# spend a model call, and the fallback is `max_weight * 10` against a profile where
# no topic sets `**Weight:**` — so every whole-word match scored the loader's 1.0
# default × 10 = 10. Measured over `intel-2026-09-22.jsonl`: 258 stage-1 survivors
# against `LLM_MAX_CALLS = 40`, relevance `{10: 222, 9: 10, 8: 5, 6: 2, 3: 2, 2: 12,
# 1: 5}`, 40 rows carrying a model `why` and 218 carrying `Matches: <topic>`. Cross-
# referenced against `vault-written.json`: 104 writes, **101 of them keyword-only at
# relevance 10**, and of the 19 items `RELEVANCE_FLOOR` held, 0 were keyword-only —
# the floor is arithmetically inert on an ungraded item, which scores exactly 10 or
# exactly 1 and never anything between 4 and 10. The run summary made that day read
# as fully graded, because `LLM scored {graded} of {calls}` prints "40 of 40":
# `calls` is the budget, not the survivor count.
#
# The fix is by *cause*, not magnitude: each scored item carries `grade_source`
# naming what produced its relevance, and the writer bars exactly
# `call_cap` — eligible for a call, budget already spent. `no_usable_grade` and
# `keyword` keep writing, because the case below named `INTEL_DISABLE_LLM=1` is what
# the keyword fallback exists for, and a bar that reached it turns an engine outage
# into a zero-write day.
#
# Deliberately untouched: `interests.md`. A `**Weight:**` below ~0.4 skips the
# stage-2 call *and* cannot clear the floor, so weights are an off switch and their
# relative values are Alan's call (see the #852 note above).

CAP_OUTCOMES = ("model", "no_usable_grade", "call_cap", "keyword")


def _knowledge_text(tmp_path) -> str:
    root = tmp_path / "obsidian" / "knowledge"
    return "\n".join(p.read_text() for p in root.rglob("*.md")) if root.exists() else ""


def test_stage2_score_names_which_outcome_produced_each_relevance(redirect_paths):
    """Clause 1: four outcomes, one run, and two of them score identically.

    `max_llm_calls=2` over four items: the first is asked and answers with a usable
    5; the second is asked and answers 97, which `_clamp_relevance` refuses as a
    failed grade; the third is eligible but the budget is spent; the fourth matches
    only the 0.2-weight Wearables topic, so stage 2 never considers asking. Their
    relevance is 5, 9, 9, 2 — and the two 9s are the point: identical numbers with
    different causes, one of which must never be written.
    """
    profile = _profile(redirect_paths)

    def model(prompt):
        model.n += 1
        return _reply(relevance=97 if model.n == 2 else 5)(prompt)
    model.n = 0

    scored = scoring_mod.stage2_score(
        [_item("asked-clean", source="youtube", title="vllm speculative decoding",
               summary="vllm"),
         _item("asked-junk", source="youtube", title="vllm mixture of experts",
               summary="vllm"),
         _item("cap-refused", source="youtube", title="vllm quantization notes",
               summary="vllm"),
         _item("never-eligible", source="youtube", title="IMU wrist calibration",
               summary="imu drift on the wrist mount")],
        profile, llm_call=model, max_llm_calls=2)

    assert [s.grade_source for s in scored] == list(CAP_OUTCOMES)
    assert [s.relevance for s in scored] == [5, KW_SCORE_09, KW_SCORE_09, 2], (
        "the junk grade must keep the fallback score, as it did before #1380; only "
        f"the cause it is filed under is new: {[s.relevance for s in scored]}")
    assert scoring_mod.stage2_score.last_cap_refused == 1
    assert scoring_mod.stage2_score.last_llm_calls == 2
    assert set(CAP_OUTCOMES) == {models_mod.GRADE_MODEL,
                                 models_mod.GRADE_NO_USABLE_GRADE,
                                 models_mod.GRADE_CALL_CAP,
                                 models_mod.GRADE_KEYWORD}


def test_write_all_to_vault_refuses_the_item_the_call_cap_left_ungraded(redirect_paths,
                                                                       capsys):
    """Clause 2, across the seam the unit tests cannot fake: the day file.

    One item graded 6 and one left at the keyword 9 by a 1-call budget, both above
    `RELEVANCE_FLOOR = 4` — so the floor is *not* what holds the second one, and the
    only thing that can is `refused_by_call_cap`. The score then goes out through
    `to_json` and comes back through `load_scored_items`, because in production the
    writer is a different process from the scorer.
    """
    profile = _profile(redirect_paths)
    llm = RecordingLLM(relevance=6)
    scored = scoring_mod.stage2_score(
        [_item("graded", source="youtube", title="vllm speculative decoding",
               summary="vllm"),
         _item("ungraded", source="youtube", title="vllm mixture of experts",
               summary="vllm")],
        profile, llm_call=llm, max_llm_calls=1)

    assert [s.grade_source for s in scored] == ["model", "call_cap"]
    assert [s.relevance for s in scored] == [6, KW_SCORE_09]

    day = _write_day(redirect_paths, scored)
    reloaded = vw_mod.load_scored_items(day)
    assert [vw_mod.refused_by_call_cap(i) for i in reloaded] == [False, True], (
        "the cause did not survive the JSONL round trip")

    written = vw_mod.write_all_to_vault(day)

    assert written == 1, "the ungraded item was written"
    assert _knowledge_sections(redirect_paths) == 1
    text = _knowledge_text(redirect_paths)
    assert "https://example.com/ungraded" not in text
    assert "https://example.com/graded" in text
    out = capsys.readouterr().out
    assert "Held 1 item(s) the stage-2 model was never asked about" in out, out


def test_run_summary_names_the_survivors_the_call_cap_never_asked(redirect_paths,
                                                                 capsys):
    """Clause 3: `LLM scored 40 of 40` was the sentence that hid 218 items.

    Four eligible survivors, a 2-call budget. The old summary printed "LLM scored 2
    of 2 items it was asked about", which is true and useless: it names the budget,
    not the day. The gap must be its own figure, and the graded line must not be
    able to read as a full grading of the four.
    """
    profile = _profile(redirect_paths)
    items = [_item(f"v{i}", source="youtube", title="vllm thing", summary="vllm")
             for i in range(4)]

    scored = scoring_mod.run_scoring_pipeline(items, profile,
                                              llm_call=RecordingLLM(relevance=6),
                                              max_llm_calls=2)

    out = capsys.readouterr().out
    assert len(scored) == 4
    assert "LLM scored 2 of 2 items it was asked about" in out
    assert "NEVER ASKED: 2 of 4" in out, out
    assert "stage-2 call budget of 2" in out, out
    assert "LLM scored 4" not in out, "the graded line still reads as a full grading"


def test_a_day_the_call_cap_reaches_no_prints_no_unasked_figure(redirect_paths, capsys):
    """Guards the line above against passing by always printing something.

    The same four items with a 4-call budget: every survivor is asked, so there is
    no gap to name and the summary must say nothing about one.
    """
    profile = _profile(redirect_paths)
    items = [_item(f"v{i}", source="youtube", title="vllm thing", summary="vllm")
             for i in range(4)]

    scoring_mod.run_scoring_pipeline(items, profile,
                                     llm_call=RecordingLLM(relevance=6),
                                     max_llm_calls=4)

    out = capsys.readouterr().out
    assert "LLM scored 4 of 4 items it was asked about" in out
    assert "NEVER ASKED" not in out, out


def test_a_model_outage_still_writes_every_keyword_graded_item(redirect_paths,
                                                              monkeypatch, capsys):
    """Clause 4, the purpose clause: the bar is scoped to cap overflow.

    Five eligible items, a 2-call budget, and the engine off by env — the shape of a
    batch run taken while the model is down. Not one of the five may be filed as
    `call_cap`, because the model was not run short of calls, it was never asked to
    run at all; and all five must reach the vault at the keyword fallback score the
    sibling test above pins at 10/urgent for the scorer alone. A bar that leaked into
    this path would turn an engine outage into a zero-write day.
    """
    profile = _profile(redirect_paths)
    monkeypatch.setenv("INTEL_DISABLE_LLM", "1")
    items = [_item(f"k{i}", source="youtube", title="vllm speculative decoding",
                   summary="vllm") for i in range(5)]

    scored = scoring_mod.stage2_score(items, profile, max_llm_calls=2)

    assert scoring_mod.stage2_score.last_llm_calls == 0
    assert scoring_mod.stage2_score.last_cap_refused == 0
    assert {s.grade_source for s in scored} == {"keyword"}

    written = vw_mod.write_all_to_vault(_write_day(redirect_paths, scored))

    assert written == 5, "an engine outage must not become a zero-write day"
    out = capsys.readouterr().out
    assert "never asked about" not in out.lower(), out


def test_a_feed_file_predating_the_grade_cause_still_writes(redirect_paths):
    """The default on a missing `grade_source` is a behaviour, not a label.

    `intel-<date>.jsonl` files written before #1380 carry no cause at all. Were the
    missing key read as `call_cap`, re-running `--write` over such a day would write
    nothing — the zero-write day this change must never cause — so both directions
    are pinned: a cause that is there survives the round trip, and one that is not
    there is `keyword`, which is writable.
    """
    graded = ScoredItem(**{**_item("rt1", source="youtube", title="vllm thing",
                                   summary="vllm").to_dict(),
                           "relevance": 6, "urgency": "morning",
                           "why": "Directly relevant", "projects": [],
                           "category": "ai-llms", "grade_source": "model"})
    assert "call_cap" not in graded.to_json()
    assert ScoredItem.from_json(graded.to_json()).grade_source == "model"

    legacy = {**_item("rt2", source="youtube", title="vllm thing",
                      summary="vllm").to_dict(), "relevance": 10,
              "urgency": "urgent", "why": "Matches: vllm", "projects": [],
              "category": "ai-llms"}
    assert "grade_source" not in legacy
    day = "2026-09-11"
    (redirect_paths / "lloyd" / "_pipeline" / "vault-derived" / "memory" / "feeds"
     / f"intel-{day}.jsonl").write_text(json.dumps(legacy) + "\n")

    reloaded = vw_mod.load_scored_items(day)

    assert [i.grade_source for i in reloaded] == ["keyword"]
    assert not vw_mod.refused_by_call_cap(reloaded[0])
    assert vw_mod.write_all_to_vault(day) == 1


def test_cli_run_past_the_call_cap_writes_only_what_the_model_graded(tmp_path):
    """The whole acceptance check, across both real process boundaries.

    `python -m intel_pipeline --score` is the process autonomy task #30 spawns, and
    the model is a loopback HTTP peer; `--write` then runs as a **second** process,
    which is what makes this the seam test rather than a unit test with extra steps:
    `write_all_to_vault` re-reads `intel-<date>.jsonl` from disk, so a cause held
    only in the scoring process would be absent exactly where the refusal has to
    act. `LLM_MAX_CALLS + 2` eligible items, the stub grading each 6 — above the
    floor — so every refusal is attributable to the cap and to nothing else.
    """
    home, feeds = _cli_home(tmp_path)
    today = _today_str()
    cap = scoring_mod.LLM_MAX_CALLS
    items = [_item(f"yt{i}", source="youtube", title="vllm speculative decoding",
                   summary="vllm") for i in range(cap + 2)]
    (feeds / "raw" / f"{today}.jsonl").write_text(
        "\n".join(i.to_json() for i in items) + "\n")

    day, score_proc = _run_cli(home, 6, "--score")

    assert score_proc.returncode == 0, score_proc.stderr[-2000:]
    rows = [json.loads(l) for l in
            (feeds / f"intel-{day}.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == cap + 2
    causes = [r["grade_source"] for r in rows]
    assert causes.count("model") == cap, "the budget did not get spent before it cut off"
    assert causes.count("call_cap") == 2
    assert f"NEVER ASKED: 2 of {cap + 2}" in score_proc.stdout, score_proc.stdout[-1500:]

    day2, write_proc = _run_cli(home, 6, "--write")

    assert day2 == day and write_proc.returncode == 0, write_proc.stderr[-2000:]
    assert "Held 2 item(s) the stage-2 model was never asked about" in write_proc.stdout
    knowledge = home / "obsidian" / "knowledge"
    assert sum(_sections(p) for p in knowledge.rglob("*.md")) == cap, (
        "the digest grew past the graded set")
    text = "\n".join(p.read_text() for p in knowledge.rglob("*.md"))
    refused_urls = [r["url"] for r in rows if r["grade_source"] == "call_cap"]
    assert len(refused_urls) == 2
    assert not [u for u in refused_urls if u in text], (
        f"a cap-refused item reached the vault: {refused_urls}")

    # The triage measure itself: written ids cross-referenced against the day file,
    # split on the `Matches:` prefix that only the keyword fallback emits. Today it
    # is 101; after this change it must be 0.
    written_ids = json.loads((feeds / "vault-written.json").read_text())["written"]
    by_id = {r["id"]: r for r in rows}
    assert [i for i in written_ids
            if (by_id[i].get("why") or "").startswith("Matches:")] == [], (
        f"keyword-only writes reached the vault: {written_ids}")


def test_write_item_to_vault_refuses_the_cap_item_on_its_own_surface(redirect_paths,
                                                                    capsys):
    """The refusal belongs on every route into the vault, not just the batch one.

    `write_all_to_vault` filters the day before it prints the held count, but
    `write_item_to_vault` is public and reachable with no filter in front of it, so
    a guard living only on the batch path is a guard on one of two write surfaces.
    Called directly with a cap-refused item, it must write nothing and leave no
    digest behind — and the floor must not be what stops it, or the test would pass
    for the wrong reason.
    """
    profile = _profile(redirect_paths)
    ungraded = ScoredItem(**{**_item("cap1", source="youtube",
                                     title="vllm mixture of experts",
                                     summary="vllm").to_dict(),
                             "relevance": KW_SCORE_09, "urgency": "urgent",
                             "why": "Matches: vllm", "projects": [],
                             "category": "ai-llms",
                             "grade_source": models_mod.GRADE_CALL_CAP})
    assert not vw_mod.below_floor(ungraded), (
        "a floor-refused item cannot test the call-cap refusal")

    assert vw_mod.write_item_to_vault(ungraded, profile) is False
    assert _knowledge_sections(redirect_paths) == 0, (
        "the direct write surface wrote an ungraded item")
    assert "https://example.com/cap1" not in _knowledge_text(redirect_paths)
    assert "Refusing" in capsys.readouterr().out


def test_a_spent_budget_refuses_the_unasked_and_keeps_the_asked_junk(redirect_paths,
                                                                    capsys):
    """Where clause 2 stops and clause 4 begins, told apart by cause alone.

    Four eligible survivors, a 2-call budget, and a model answering 97 — a grade
    `_clamp_relevance` refuses, so the two asked items keep the keyword 9 and the
    two the budget never reached carry the same 9. All four scores are identical,
    which is the point: the only difference between writing and refusing is whether
    a call was ever made. The asked-but-junk pair must still be written, because
    refusing them is what turns a degraded engine into a zero-write day; the
    unasked pair must not be.
    """
    profile = _profile(redirect_paths)
    scored = scoring_mod.stage2_score(
        [_item(f"j{i}", source="youtube", title="vllm thing", summary="vllm")
         for i in range(4)],
        profile, llm_call=RecordingLLM(relevance=97), max_llm_calls=2)

    assert [s.grade_source for s in scored] == [
        models_mod.GRADE_NO_USABLE_GRADE, models_mod.GRADE_NO_USABLE_GRADE,
        models_mod.GRADE_CALL_CAP, models_mod.GRADE_CALL_CAP]
    assert [s.relevance for s in scored] == [KW_SCORE_09] * 4, (
        "if the four differ by score the writer's split is not being attributed to "
        f"the cause: {[s.relevance for s in scored]}")

    written = vw_mod.write_all_to_vault(_write_day(redirect_paths, scored))

    assert written == 2, f"the asked-but-junk items did not survive: {written}"
    assert _knowledge_sections(redirect_paths) == 2
    out = capsys.readouterr().out
    assert "Held 2 item(s) the stage-2 model was never asked about" in out, out
