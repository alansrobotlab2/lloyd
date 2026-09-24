"""#562 — the per-section size map in prefetch, and the prefetch cost eval.

Hermetic: no engine, no qmd, no aggregator. `run_query` and the judge are
stubbed wherever a test reaches them.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import prefetch  # noqa: E402


def _load_eval():
    spec = importlib.util.spec_from_file_location(
        "run_prefetch_cost_eval", REPO / "eval" / "run_prefetch_cost_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load_eval()

SKILL = {"name": "demo-skill", "raw": "# Demo\nDo the thing.\n" * 5}
VAULT_HIT = [{"title": "Note", "snippet": "the answer is 42", "score": 0.9,
              "file": "qmd://obsidian/projects/lloyd/note.md"}]


# ── prefetch: the size map (clause 5) ────────────────────────────────────────

def test_a_two_section_block_gives_a_two_entry_size_map():
    block = prefetch._format_context([(9.0, SKILL)], [], vault_results=VAULT_HIT,
                                     show_skill_hint=False)
    sizes = prefetch.injected_section_sizes(block)
    assert set(sizes) == {"skill", "vault-context"}
    skill_open = block.index("<skill ")
    skill_close = block.index("</skill>") + len("</skill>")
    assert sizes["skill"] == skill_close - skill_open
    # The envelope is attributed to nobody.
    assert sum(sizes.values()) < len(block)


def test_skill_hint_is_not_read_as_a_skill():
    block = prefetch._format_context([], [], vault_results=VAULT_HIT)
    sizes = prefetch.injected_section_sizes(block)
    assert "skill-hint" in sizes and "skill" not in sizes


def test_a_tag_quoted_inside_a_section_belongs_to_that_section():
    block = ("<context>\n<skill name=\"x\" score=\"9.0\">\nwrite <facts> here\n"
             "</facts> too\n</skill>\n<facts>\n- a\n</facts>\n</context>")
    sizes = prefetch.injected_section_sizes(block)
    assert set(sizes) == {"skill", "facts"}
    assert sizes["facts"] == len("<facts>\n- a\n</facts>")


def test_an_unclosed_section_runs_to_the_end():
    block = "<context>\n<vault-context>\n- x"
    assert prefetch.injected_section_sizes(block) == {
        "vault-context": len(block) - len("<context>\n")}


def test_drop_sections_removes_the_section_and_its_line():
    q = "what is the answer?"
    block = prefetch._format_context([(9.0, SKILL)], [], vault_results=VAULT_HIT,
                                     show_skill_hint=False)
    rendered = block + "\n\n" + q
    out = prefetch.drop_sections(rendered, ["skill"])
    assert "<skill" not in out and "<vault-context>" in out
    assert out == prefetch._format_context([], [], vault_results=VAULT_HIT,
                                           show_skill_hint=False) + "\n\n" + q
    # Dropping everything leaves the bare message, not an empty envelope.
    assert prefetch.drop_sections(rendered, ["skill", "vault-context"]) == q
    assert prefetch.drop_sections(rendered, []) == rendered


def test_split_injected_prices_the_prefix_and_zero_for_no_block():
    q = "hello there, what is up"
    block = prefetch._format_context([], ["- fact"], show_skill_hint=False)
    s = prefetch.split_injected(block + "\n\n" + q, q)
    assert s["injected_chars"] == len(block) + 2
    assert set(s["injected_sections"]) == {"facts"}
    assert prefetch.split_injected(q, q)["injected_chars"] == 0


def test_the_budget_line_carries_sections_and_not_the_message(caplog):
    q = "<facts> is a word the user typed"
    block = prefetch._format_context([], ["- f"], vault_results=VAULT_HIT,
                                     show_skill_hint=False)
    with caplog.at_level(logging.INFO, logger=prefetch.logger.name):
        report = prefetch.log_turn_prompt_budget(block + "\n\n" + q, session_id="s1")
    assert set(report["context_sections"]) == {"facts", "vault-context"}
    line = next(r.getMessage() for r in caplog.records if "PROMPT_BUDGET" in r.getMessage())
    assert "sections=" in line and "vault-context=" in line


# ── the eval's pure halves ───────────────────────────────────────────────────

def test_arm_prompts_come_from_one_rendering():
    q = "what is the answer?"
    rendered = prefetch._format_context([(9.0, SKILL)], [], vault_results=VAULT_HIT,
                                        show_skill_hint=False) + "\n\n" + q
    assert ev.arm_prompt(rendered, q, ev.ARM_INJECTED) == rendered
    assert ev.arm_prompt(rendered, q, ev.ARM_AA) == rendered
    assert ev.arm_prompt(rendered, q, ev.ARM_SUPPRESSED) == q
    no_skill = ev.arm_prompt(rendered, q, ev.ARM_NO_SKILL)
    assert "<skill" not in no_skill and no_skill.endswith(q)


def test_prefill_charges_only_what_each_request_added():
    its = [{"input_tokens": 1000, "cache_read": 900, "output_tokens": 50},
           {"input_tokens": 1300, "cache_read": 1000, "output_tokens": 40},
           {"input_tokens": 1200, "cache_read": 0, "output_tokens": 10}]  # compacted
    c = ev.arm_cost(its, system_prefix_tokens=800)
    assert c["prefill_tokens"] == 200 + 300 + 0
    assert c["uncached_prompt_tokens"] == (1000 + 1300 + 1200) - (900 + 1000)
    assert c["iterations"] == 3 and c["output_tokens"] == 100
    assert ev.arm_cost([], system_prefix_tokens=800)["prefill_tokens"] == 0


def test_marginal_counts_a_hit_that_cost_more_and_bought_nothing():
    inj = {"prefill_tokens": 5000, "judge_score": 2}
    sup_cheaper_same = {"prefill_tokens": 4000, "judge_score": 2}
    sup_cheaper_worse = {"prefill_tokens": 4000, "judge_score": 1}
    m = ev.marginal(inj, sup_cheaper_same, hit=True)
    assert m["negative_marginal"] and m["negative_marginal_strict"] and m["delta"] == 1000
    m = ev.marginal(inj, sup_cheaper_worse, hit=True)
    assert m["negative_marginal"] and not m["negative_marginal_strict"]
    # A tie is neutral, and a miss is never a negative-marginal HIT.
    assert not ev.marginal(inj, {"prefill_tokens": 5000, "judge_score": 2}, hit=True)["negative"]
    assert not ev.marginal(inj, sup_cheaper_same, hit=False)["negative_marginal"]


def _rec(qid, hit, inj, sup, aa, inj_tok=100):
    def arm(p, j, t):
        return {"prefill_tokens": p, "uncached_prompt_tokens": p, "iterations": 3,
                "tool_calls": 2, "output_tokens": 10, "seconds": 1.0, "judge_score": j,
                "injected_tokens": t, "reached_expected": True, "completed": True,
                "error": None}
    arms = {ev.ARM_INJECTED: arm(inj[0], inj[1], inj_tok),
            ev.ARM_SUPPRESSED: arm(sup[0], sup[1], 0),
            ev.ARM_AA: arm(aa, inj[1], inj_tok)}
    r = {"id": qid, "doc_hit": hit, "arms": arms}
    r["marginal"] = ev.marginal(arms[ev.ARM_INJECTED], arms[ev.ARM_SUPPRESSED], hit=hit)
    return r


def test_summary_reports_the_headline_columns():
    recs = [_rec("a", True, (5000, 2), (4000, 2), 5100),
            _rec("b", True, (3000, 2), (6000, 2), 3300),
            _rec("c", False, (4000, 1), (3500, 2), 4000),
            _rec("d", True, (2000, 2), (2500, 1), 2200)]
    s = ev.summarize(recs)
    assert s["negative_marginal_hit_count"] == 1          # a
    assert s["negative_marginal_hit_count_strict"] == 1   # a
    assert s["net_negative_count_any"] == 2               # a, c
    assert s["doc_hit_queries"] == 3
    # mean(sup - inj) prefill = (-1000 + 3000 - 500 + 500) / 4 = 500, per 100 tokens.
    assert s["marginal_value_per_injected_token"] == 5.0
    assert s["arms"][ev.ARM_INJECTED]["injected_tokens_avg"] == 100
    assert s["aa_noise_prefill_abs_median"] == 150
    assert s["negative_marginal_hit_count_beyond_noise"] == 1   # a: 1000 > p90 300
    c = s["contrasts"]["suppressed_minus_injected"]["prefill_tokens"]
    assert c["diff"] == 500 and c["n"] == 4


def test_summary_skips_arms_that_errored():
    r = _rec("a", True, (5000, 2), (4000, 2), 5000)
    r["arms"][ev.ARM_SUPPRESSED]["error"] = "timeout after 480s"
    s = ev.summarize([r, _rec("b", True, (3000, 2), (6000, 2), 3000)])
    assert s["arms"][ev.ARM_SUPPRESSED]["errors"] == 1
    assert s["contrasts"]["suppressed_minus_injected"] == {}  # one clean pair is no CI


def test_every_session_id_is_sandboxed():
    from agent_mcp._tool_sandbox import is_sandboxed_session

    for arm in ev.ARMS:
        assert is_sandboxed_session(ev.new_session_id("kg-maintenance-tasks", arm))


def test_run_arm_records_the_turn_without_an_engine(monkeypatch):
    events = [
        {"type": "assistant_message", "text": "", "usage": {"input_tokens": 100, "cache_read": 90}},
        {"type": "tool_call", "name": "Read", "args_json": '{"path": "projects/x.md"}'},
        {"type": "tool_result", "name": "Read", "content": "body"},
        {"type": "assistant_message", "text": "It is 42.",
         "usage": {"input_tokens": 150, "cache_read": 100}},
        {"type": "result", "stop_reason": "stop", "response_text": "It is 42."},
    ]

    async def fake_run_query(messages, options):
        for e in events:
            yield e

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", fake_run_query)
    out = asyncio.run(ev.run_arm("q", options=None, timeout_s=5,
                                 expect_docs=["projects/x.md"]))
    assert out["tool_calls"] == 1 and out["completed"] and out["reached_expected"]
    assert out["answer"] == "It is 42." and len(out["_iterations"]) == 2
    assert out["error"] is None


def test_an_empty_answer_is_judged_zero_without_a_call():
    out = asyncio.run(ev.judge("http://127.0.0.1:1", "m", "q", "gold", "  "))
    assert out["score"] == 0


def test_an_unreachable_judge_is_recorded_not_raised():
    out = asyncio.run(ev.judge("http://127.0.0.1:1", "m", "q", "gold", "an answer"))
    assert out["score"] is None and "judge failed" in out["why"]


@pytest.mark.parametrize("arm", sorted(ev.ABLATIONS))
def test_every_ablation_names_a_known_section(arm):
    assert set(ev.ABLATIONS[arm]) <= set(prefetch.CONTEXT_SECTION_TAGS)


def test_merge_joins_arms_by_query_and_drops_the_heavy_fields():
    a = {"label": "a", "records": [dict(_rec("q", True, (5000, 2), (4000, 2), 5000),
                                        rendered="<context>...")]}
    extra = {ev.ARM_NO_SKILL: dict(a["records"][0]["arms"][ev.ARM_INJECTED],
                                   prefill_tokens=4500, answer="long answer")}
    b = {"label": "b", "records": [{"id": "q", "doc_hit": True, "arms": extra}]}
    m = ev.merge_artifacts([a, b])
    (rec,) = m["records"]
    assert set(rec["arms"]) == {ev.ARM_INJECTED, ev.ARM_SUPPRESSED, ev.ARM_AA, ev.ARM_NO_SKILL}
    assert "rendered" not in rec and "answer" not in rec["arms"][ev.ARM_NO_SKILL]
    assert m["summary"]["negative_marginal_hit_count"] == 1
