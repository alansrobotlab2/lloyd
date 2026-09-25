"""djev as the fast path for a decision, said where the model decides.

In djev's first four days the model called a djev tool from one session of
~900, and never from a worker turn. The tools' descriptions are read at call
time, after the model has already chosen to reason a ranking or a
classification out in tokens, which is the asymmetry the web-lookup and
code-navigation paragraphs were written for. Three things fix it, and each is
pinned here:

1. a constant system-prompt paragraph naming djev for rank/classify/triage,
   on chat and worker turns both;
2. the trigger in the FIRST sentence of each description, which is what the
   ToolSearch catalog gist keeps (`app/harness/tests/test_tool_search.py`);
3. a compact `djev_decide` result, so reading the answer back does not cost
   what the call saved.
"""

from __future__ import annotations

import asyncio
import json
import re

import prompt_builder as pb
from agent_mcp import djev as tools
from app import djev
from app.harness.tool_search import _gist


def _para(platform: str = "") -> str:
    s = pb.build_system_prompt(include_skills_index=False, platform=platform)
    start = s.find("Fast decisions:")
    assert start != -1, f"no fast-decisions paragraph on platform {platform!r}"
    end = s.find("\n\n", start)
    return s[start:end if end != -1 else None]


# ── 1. the paragraph ────────────────────────────────────────────────────────

def test_chat_and_worker_turns_both_carry_the_paragraph():
    """Workers are where batches of judgements live (triage, dedupe, review),
    so a chat-only paragraph would miss the turns that need it most."""
    for platform in ("", "mission-control", "worker", "autonomy"):
        assert "djev_decide" in _para(platform)


def test_paragraph_names_both_tools_and_the_shapes_they_fit():
    para = _para()
    assert "djev_decide" in para and "djev_rank" in para
    for shape in ("rank", "classify", "triage"):
        assert shape in para


def test_paragraph_says_what_not_to_trust_and_when_not_to_call():
    """Overshoot is the risk of naming a tool: a call for a judgement already
    in hand costs more than the answer. And the scores are not calibrated."""
    para = _para()
    assert "0.5" in para and "never" in para
    assert "needs no tool" in para
    assert "error" in para, "must say what to do when djev does not answer"


def test_paragraph_sits_with_the_other_tool_affordances():
    s = pb.build_system_prompt(include_skills_index=False)
    nav, fast, turn = (s.find("Code navigation:"), s.find("Fast decisions:"),
                       s.find("Turn discipline:"))
    assert -1 not in (nav, fast, turn)
    assert nav < fast < turn


def test_paragraph_carries_no_per_turn_value():
    """A per-turn value in the system prompt re-prefills it every turn."""
    para = _para()
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", para)
    assert not re.search(r"\d+ ?ms\b", para), "a latency figure goes stale"


def test_every_tool_name_in_the_paragraph_is_advertised():
    advertised = {t.name for t in asyncio.run(tools.list_tools())}
    named = set(re.findall(r"\bdjev_[a-z]+", _para()))
    assert named and named <= advertised, named - advertised


# ── 2. the trigger survives the catalog gist ────────────────────────────────

def test_decide_and_rank_gists_say_to_use_them_instead_of_reasoning():
    by_name = {t.name: t.description for t in asyncio.run(tools.list_tools())}
    for name in ("djev_decide", "djev_rank"):
        assert "instead of" in _gist(by_name[name]), (name, _gist(by_name[name]))


# ── 3. the compact result ───────────────────────────────────────────────────

def _answers(**over):
    base = dict(answers={
        "urgent": djev.Answer(id="urgent", type="noul", value=0.98123,
                              label="yes", confidence=0.98123,
                              probabilities={"yes": 0.98123, "no": 0.01877},
                              label_mass=0.9998, argmax_is_label=True),
        "area": djev.Answer(id="area", type="choice", value="storage",
                            label="storage", confidence=0.9,
                            probabilities={"storage": 0.9, "network": 0.1},
                            label_mass=0.95, argmax_is_label=True,
                            low_trust=True)},
        latency_ms=260.0, server_ms=250.0, prompt_tokens=80,
        chunks=[["urgent", "area"]], uninformative=False, floor=None)
    base.update(over)
    return djev.Answers(**base)


def _decide(monkeypatch, answers, **args):
    async def _ask(*a, **k):
        return answers
    monkeypatch.setattr(djev, "ask", _ask)
    res = asyncio.run(tools.call_tool("djev_decide", {
        "state": "s", "questions": {"urgent": {"type": "noul"}}, **args}))
    return json.loads(res.content[0].text)


def test_compact_is_the_default_and_keeps_what_a_caller_acts_on(monkeypatch):
    out = _decide(monkeypatch, _answers())
    urgent, area = out["answers"]["urgent"], out["answers"]["area"]
    assert urgent == {"label": "yes", "value": 0.981,
                      "p": {"yes": 0.981, "no": 0.019}}
    # A choice's value is its label, so it is not sent twice.
    assert area["label"] == "storage" and "value" not in area
    # A trust problem is still said, and only when there is one.
    assert area["low_trust"] is True and "low_trust" not in urgent
    assert out["min_label_mass"] == 0.95
    assert "calibrated" in out["note"]
    for noise in ("latency_ms", "chunks", "argmax_is_label", "seam"):
        assert noise not in json.dumps(out), noise


def test_compact_is_much_smaller_than_verbose(monkeypatch):
    compact = _decide(monkeypatch, _answers())
    verbose = _decide(monkeypatch, _answers(), verbose=True)
    assert len(json.dumps(compact)) < 0.6 * len(json.dumps(verbose))


def test_verbose_returns_every_diagnostic(monkeypatch):
    out = _decide(monkeypatch, _answers(), verbose=True)
    assert out["answers"]["urgent"]["label_mass"] == 0.9998
    assert out["latency_ms"] == 260.0 and "chunks" in out


def test_compact_keeps_both_warnings(monkeypatch):
    """The two answers-are-not-answers cases must survive compaction."""
    out = _decide(monkeypatch, _answers(chunks=[["urgent"], ["area"]],
                                        uninformative=True))
    assert "MUST NOT be sorted" in out["warning"]
    assert "degenerate" in out["warning_uninformative"]


def test_compact_keys_a_score_by_level_name(monkeypatch):
    """The verbose form carries a legend for a score's index keys; compact
    drops the legend, so the names have to move into the keys."""
    sev = djev.Answer(id="sev", type="score", value=2.24, label="major",
                      confidence=0.75,
                      probabilities={"0": 0.0, "1": 0.0, "2": 0.755, "3": 0.245},
                      label_mass=0.99, argmax_is_label=True,
                      legend={"0": "cosmetic", "1": "minor", "2": "major",
                              "3": "critical"})
    out = _decide(monkeypatch, _answers(answers={"sev": sev}, chunks=[["sev"]]))
    assert out["answers"]["sev"]["p"] == {"cosmetic": 0.0, "minor": 0.0,
                                          "major": 0.755, "critical": 0.245}
    assert out["answers"]["sev"]["value"] == 2.24
