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
    feeds = home / "lloyd" / "_pipeline" / "vault-derived" / "memory" / "feeds"
    (feeds / "raw").mkdir(parents=True)
    (home / "obsidian").mkdir(parents=True)
    (home / "obsidian" / "interests.md").write_text(PROFILE_MD)
    (home / "lloyd" / "scripts" / "intel-pipeline" / "config").mkdir(parents=True)
    return home, feeds


def _run_cli(home, relevance, *flags, extra_env=None):
    today = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubModel)
    server.relevance = relevance
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = dict(os.environ, HOME=str(home),
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
    today = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d")
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
    today = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d")
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
    today = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d")
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
