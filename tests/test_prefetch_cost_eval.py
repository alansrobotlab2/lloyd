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

from app import prefetch  # noqa: E402


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



# ---------------------------------------------------------------------------
# #4027 — the arm that varies WHICH skills are advertised, and the population split
# that keeps it honest. All six earlier arms render the whole library, so nothing in
# this file could ever have tested the claim in
# knowledge/video-summaries/more-skills-worse-agents.md that a big skill library
# degrades an agent, and `skills-index-2026-09-25.md` is not evidence about it either
# way. These nodes are what makes the claim reachable; the tail split is what stops a
# smaller set from being scored only on the queries its surviving skills cover.

def test_the_pruned_arm_is_selectable_and_advertises_a_strict_subset() -> None:
    """Clause 1: the arm exists, is accepted by --arms, and its index is a strict subset.

    The set figures are the live library as this round measured it and do not move with
    the event log: 208 skills and 11,998 characters of index for production, 140 skills
    at PRUNE_ADVERTISE_TOP_N, so exactly 68 names stop being advertised and every one of
    those 68 is still a production name.

    The CHARACTER figure is deliberately NOT asserted to fall, and that is the round's
    most useful finding rather than a hedge: `skills.index.budget_chars: 12000` is
    already saturated by the full library, so removing 68 lines frees budget that the
    renderer immediately spends on descriptions for the survivors. Measured 2026-11-09
    over the live 30-day event log, the advertised index went 11,998 -> 11,999
    characters — the arm cut a third of the names and bought one character back — while
    the system+tools prefix measured 68 tokens cheaper. What that says about pruning as
    a prefill optimisation is the arm's first result, and the assertion below is what
    stops a future reader from claiming the cut bought space it did not buy.
    """
    assert ev.ARM_INJECTED_PRUNED in ev.ARMS, ev.ARMS
    prod, pruned = ev.advertised_skill_names(ev.ARM_INJECTED), \
        ev.advertised_skill_names(ev.ARM_INJECTED_PRUNED)
    assert len(prod) == 208, f"the live library moved off the 208 this round measured: {len(prod)}"
    assert len(pruned) == ev.PRUNE_ADVERTISE_TOP_N == 140, len(pruned)
    assert pruned < prod, f"the arm invented skills: {sorted(pruned - prod)}"
    assert len(prod - pruned) == 68, sorted(prod - pruned)[:5]

    prod_block, pruned_block = (ev.advertised_index_block(ev.ARM_INJECTED),
                                ev.advertised_index_block(ev.ARM_INJECTED_PRUNED))
    assert prod_block and pruned_block, "the <available_skills> fence moved; the reader is blind"
    assert len(prod_block.splitlines()) - len(pruned_block.splitlines()) == 68, \
        (len(prod_block.splitlines()), len(pruned_block.splitlines()))
    assert abs(len(pruned_block) - len(prod_block)) <= 0.02 * len(prod_block), (
        f"the char gap widened past the budget-refill effect "
        f"(prod {len(prod_block)} vs pruned {len(pruned_block)}): either the renderer "
        f"changed or the claim written here is now false")


def test_the_pruned_arm_changes_only_the_advertised_set() -> None:
    """Clause 2: one variable, and the library it prunes from is intact.

    Three things must hold for the arm's number to mean anything. The two config keys the
    variant arms already swap — `skills.index.descriptions`, `prefetch.skills.push` — have
    to be at their production values after this arm builds a prompt, or the arm is a
    description experiment wearing a set experiment's clothes. The rendered TURN must be
    byte-identical to `injected`'s, because everything outside the system prompt is the
    same measurement. And the pruned skill must still be on disk and walkable by the
    library's own walker: it is un-advertised, not deleted, which is the only reason a
    tail query is a deferred-task control instead of an ablation.

    Which skill is in the tail is not hardcoded: the pruned set is ranked by the 30-day
    event log, so a name that is in today's tail is out of it next week (`voice-mode`
    sits at rank 194 of 208 on an empty log and inside the top 140 on the live one). The
    node derives one from the difference and asserts the invariant that does not move.
    """
    import yaml
    from app.config import CONFIG
    from agent_mcp.skills import iter_active_skills
    from app.prompt_builder import _skill_walk_roots

    prod, pruned = ev.system_prompt_for(ev.ARM_INJECTED), \
        ev.system_prompt_for(ev.ARM_INJECTED_PRUNED)
    assert prod != pruned
    # Each prompt is cut at its OWN fence: the two blocks are not the same length (the
    # pruned one is 11,999 chars against production's 11,988), so slicing both at one
    # offset would compare a suffix against itself shifted by a character and read as a
    # difference the arms do not have.
    open_tag, close_tag = "<available_skills>", "</available_skills>"
    for name, text in (("production", prod), ("pruned", pruned)):
        assert text.count(open_tag) == 1 and text.count(close_tag) == 1, name
    assert prod[:prod.index(open_tag)] == pruned[:pruned.index(open_tag)], \
        "the arms differ BEFORE the advertised index"
    assert prod[prod.index(close_tag):] == pruned[pruned.index(close_tag):], \
        "the arms differ AFTER the advertised index"

    live = ((CONFIG.get("skills") or {}).get("index") or {})
    cfg_path = Path(ev.__file__).resolve().parents[1] / "config.yaml"
    disk = yaml.safe_load(cfg_path.read_text("utf-8"))
    on_disk = ((disk.get("skills") or {}).get("index") or {})
    assert live.get("descriptions") == on_disk.get("descriptions") is True, live
    assert "advertise_top_n" not in on_disk, "the arm leaked its overlay into config.yaml"
    assert ((disk.get("prefetch") or {}).get("skills") or {}).get("push") is True, \
        "prefetch.skills.push moved out from under the arm"
    assert live.get("advertise_top_n") is None, \
        "the pruned set became production's default: the production arm is no longer production"

    query = "why is the TTS still cutting off mid-sentence"
    rendered = "<ctx/>\n<skill>x</skill>"
    assert ev.arm_prompt(rendered, query, ev.ARM_INJECTED_PRUNED) \
        == ev.arm_prompt(rendered, query, ev.ARM_INJECTED)

    tail = sorted(ev.advertised_skill_names(ev.ARM_INJECTED)
                  - ev.advertised_skill_names(ev.ARM_INJECTED_PRUNED))[0]
    on_disk_names = {s.name for s in iter_active_skills(roots=_skill_walk_roots())}
    assert tail in on_disk_names, f"{tail} pruned out of the LIBRARY; the control is void"
    assert tail not in ev.advertised_skill_names(ev.ARM_INJECTED_PRUNED), \
        "the control's skill is advertised again, so there is no deferred task to defer"

def test_the_summary_reports_advertised_and_tail_populations_separately() -> None:
    """Clause 3: the headline cannot be the surviving-skill half wearing a mean's clothes.

    `summarize` is fed records shaped exactly as `run` writes them — `expected_skills` on
    the RECORD, the flags inside `arms[arm]` — because the failure this pins is a
    plausible-looking summary whose split is silently empty: labels read from the arm's
    dict are absent there, which would make every arm report `tail_n: 0` forever and leave
    the deferred-task control reporting "no tail" while the risk it guards stays open.
    Production's arm must show an EMPTY tail population (n 0, rate None — not a 0.0, which
    would read as "the tail was reached perfectly"), and the pruned arm must show the tail
    this round measured: 25 of the 54 labelled queries, `smq078` among them.
    """

    def rec(qid: str, skills: list[str], *, reached: bool) -> dict:
        flags = {"completed": 1, "judge_score": 2, "doc_hit": 0, "doc_gold": 0,
                 "duration_s": 0.5, "injected_tokens": 100, "reached_expected": 1,
                 "right_skill_reached": int(reached), "iterations": 1, "tool_names": []}
        return {"id": qid, "category": "skill", "expected_skills": skills, "query": qid,
                "arms": {ev.ARM_INJECTED: dict(flags),
                         ev.ARM_INJECTED_PRUNED: dict(flags)}}

    # Which skill is in the tail moves with the 30-day telemetry, so both populations are
    # derived from the live advertised sets rather than named here.
    advertised = sorted(ev.advertised_skill_names(ev.ARM_INJECTED_PRUNED))[0]
    tail_skill = sorted(ev.advertised_skill_names(ev.ARM_INJECTED)
                        - ev.advertised_skill_names(ev.ARM_INJECTED_PRUNED))[0]
    kept_q, tail_q = "qA", "qB"
    records = [rec(kept_q, [advertised], reached=True),      # expected skill survives the cut
               rec(tail_q, [tail_skill], reached=False)]     # expected skill pruned away
    summary = ev.summarize(records)
    prod, arm = summary["arms"][ev.ARM_INJECTED], summary["arms"][ev.ARM_INJECTED_PRUNED]

    for key in ("right_skill_reached_advertised_rate", "right_skill_reached_advertised_n",
                "right_skill_reached_tail_rate", "right_skill_reached_tail_n",
                "tail_query_ids", "advertised_index_chars", "advertised_skill_count"):
        assert key in arm, f"{key} missing from the pruned arm's summary: {sorted(arm)}"
        assert key in prod, f"{key} missing from production's summary: {sorted(prod)}"

    assert prod["right_skill_reached_tail_n"] == 0
    assert prod["right_skill_reached_tail_rate"] is None, "an empty tail is not a rate of 0.0"
    assert prod["right_skill_reached_advertised_n"] == 2
    assert prod["advertised_skill_count"] == 208

    assert arm["right_skill_reached_tail_n"] == 1
    assert arm["tail_query_ids"] == [tail_q]
    assert arm["right_skill_reached_tail_rate"] == 0.0, "the control must be able to FAIL"
    assert arm["right_skill_reached_advertised_n"] == 1
    assert arm["right_skill_reached_advertised_rate"] == 1.0
    assert arm["right_skill_reached_rate"] == 0.5, \
        "the headline is both populations, not the surviving one alone"
    assert arm["advertised_skill_count"] == ev.PRUNE_ADVERTISE_TOP_N
    # The cut is measured in NAMES, not in characters. This node used to assert the chars shrank,
    # a direction the round's own published table contradicts (11,998 → 11,999, because the freed
    # budget re-fills with descriptions for the survivors) — it held only on a synthetic library
    # where the budget is not already saturated, which is a test that passes for a reason the
    # reader cannot reproduce. The strict inequality clause 1 asks for is the line count.
    assert arm["advertised_skill_count"] < prod["advertised_skill_count"]
    for stats in (prod, arm):
        k, t = (stats["right_skill_reached_advertised_n"], stats["right_skill_reached_tail_n"])
        head = ((stats["right_skill_reached_advertised_rate"] * k
                 + (stats["right_skill_reached_tail_rate"] or 0.0) * t) / max(1, k + t))
        assert stats["right_skill_reached_rate"] == head, (stats["right_skill_reached_rate"], head)


def test_the_arms_help_lists_the_pruned_arm() -> None:
    """Clause 1's second half, at the process boundary: an operator can DISCOVER the arm.

    The help string is generated from `ARMS`, so this is satisfied by construction and is
    the cheapest node here — which is why it earns its line. The day the help stops
    interpolating `ARMS`, the arm stops being runnable by anyone who has not read the
    source, and a mechanism that only its author can invoke measures nothing.
    """
    import subprocess

    out = subprocess.run([sys.executable, str(REPO / "eval" / "run_prefetch_cost_eval.py"),
                          "--help"], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-400:]
    assert "injected_pruned" in out.stdout, out.stdout[-600:]
