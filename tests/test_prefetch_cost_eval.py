"""#562 — the per-section size map in prefetch, and the prefetch cost eval.

Hermetic: no engine, no qmd, no aggregator. `run_query` and the judge are
stubbed wherever a test reaches them.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
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


# ─────────────── #4375: the arm that varies description TEXT (`desc_frontload`) ─
#
# Every arm this file shipped before this item varied WHO is granted a description
# (`skills.index.descriptions`), whether the body is pushed (`prefetch.skills.push`),
# or WHICH skills are advertised (`advertise_top_n`). None varied the text inside the
# 100 characters `clip_skill_description` keeps, so the premise #4373 rests on — that a
# trigger inside the clip beats the same trigger at character 800 — could not be asked
# of any instrument in the repo.
#
# Numbers below are re-measured on this tree, not copied from the item, because the
# item's own triage records that both of its quoted figures had aged: the labelled set
# is 78 records / 24 expected-empty, not the 51 the two published notes ran, and the
# names-only population is ~140 of 209 index lines, not 137. Neither is asserted here;
# only the arm's invariants are. What IS measured live: production's index is 209 lines
# and 11,989 characters with 73 skills carrying a description, and the rewrite fires on
# 21 of the 208 descriptions the render sees.
class _FixtureSkill:
    """Just enough of an `iter_active_skills` record for `_skills_index_lines`."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.frontmatter = {"description": description}


BURIED_TRIGGER = "use when the nightly census job has failed or gone silent"
#: Long enough that the trigger sits past character 100 of the folded description.
BURIED_HEAD = ("Renders the weekly fleet census across the supervisor sockets and "
               "writes the rollup into the vault knowledge tree with the counts "
               "table and the per-task gap columns, ")


def _fixture_skills():
    buried = BURIED_HEAD + BURIED_TRIGGER + "."
    assert len(BURIED_HEAD) > 100, (
        "the fixture only exercises the clip if its trigger is past character 100")
    assert trigger_pos(buried) >= 100
    leading = "Use when a systemd unit is missing, to file the socket instead."
    return [(_FixtureSkill("census-skill", buried), BURIED_HEAD + BURIED_TRIGGER + "."),
            (_FixtureSkill("unit-skill", leading), leading)]


def trigger_pos(text: str) -> int:
    ev = _load_eval()
    return ev.trigger_position(text)


def _fixture_index(arm: str) -> list[str]:
    """Render the fixture through the REAL renderer under one arm's treatment."""
    ev = _load_eval()
    from app import prompt_builder
    settings = prompt_builder._skills_index_settings()
    skills = [skill for skill, _ in _fixture_skills()]
    if arm == ev.ARM_DESC_FRONTLOAD:
        with ev.frontloaded_index_render():
            return prompt_builder._skills_index_lines(
                skills, {}, budget=settings["budget_chars"],
                max_desc=settings["max_description_chars"])
    return prompt_builder._skills_index_lines(
        skills, {}, budget=settings["budget_chars"],
        max_desc=settings["max_description_chars"])


def test_desc_frontload_is_a_selectable_arm_and_a_typo_still_exits():
    """Clause 1. The name is in `ARMS`, which is what `--help` prints; `--arms
    desc_frontload` resolves; and an unknown name still exits through argparse with
    `unknown arms [...]`. Arm selection is extracted into `resolve_arms(raw, fail)` so
    this is checkable without running the eval — and `fail` is `ArgumentParser.error`
    in production, so the refusal is the same message and exit code it was before the
    extraction, which is what every operator script here matches on."""
    ev = _load_eval()
    assert ev.ARM_DESC_FRONTLOAD == "desc_frontload"
    assert "desc_frontload" in ev.ARMS, f"ARMS = {ev.ARMS}"
    assert ev.resolve_arms("desc_frontload,desc_push", lambda m: None) == (
        "desc_frontload", "desc_push")
    refused: list[str] = []
    with pytest.raises(SystemExit):
        ev.resolve_arms("desc_frontloa", _fail_with(refused))
    assert refused and "unknown arms ['desc_frontloa']" in refused[0], refused


def test_the_real_cli_still_refuses_an_unknown_arm_name() -> None:
    """Clause 1's other half, across the process boundary an operator actually crosses.

    `resolve_arms(raw, fail)` is production's arm selection now, and the node above hands
    it its own `fail` — which proves the function refuses but not that `main` still passes
    `ArgumentParser.error` as that `fail`. If the wiring were dropped, unit tests stay
    green while a run under a mis-spelled arm name silently proceeds with the default
    arms, and the exit code 2 plus the `unknown arms [...]` text that operator scripts
    match on are gone. So launch the script the way a measured run is launched, with an
    arm name one character off the new one, and read the refusal off the process.
    """
    import subprocess

    out = subprocess.run([sys.executable,
                          str(REPO / "eval" / "run_prefetch_cost_eval.py"),
                          "--arms", "desc_frontloa"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 2, f"exit {out.returncode}, stderr {out.stderr[-400:]}"
    assert "unknown arms ['desc_frontloa']" in out.stderr, out.stderr[-400:]


def _fail_with(sink):
    def fail(message: str):
        sink.append(message)
        raise SystemExit(2)
    return fail


def test_the_frontload_line_leads_with_the_trigger_and_other_lines_are_unchanged():
    """Clause 2, over a fixture: production's line buries the trigger outside the
    clip, the arm's line opens with it, and the line whose trigger already led is
    untouched.

    `_skills_index_lines` is the production renderer, called with the settings
    production reads, so this grades the real clip path rather than a re-implementation
    of it. Description text length is asserted per line because the arm's whole claim to
    be measuring text POSITION is that it changed nothing the budget walk can see."""
    ev = _load_eval()
    from app import prompt_builder
    clip = prompt_builder._skills_index_settings()["max_description_chars"]
    # [0] is the index header; the skill lines follow it alphabetically.
    production = _fixture_index(ev.ARM_DESC_PUSH)[1:]
    variant = _fixture_index(ev.ARM_DESC_FRONTLOAD)[1:]
    assert len(production) == len(variant) == 2, production

    prod_desc = production[0].split(" — ", 1)[1]
    var_desc = variant[0].split(" — ", 1)[1]
    # The position that matters is where the trigger sits in the UNCLIPPED description
    # the renderer was handed; production's line is the clip of that, so the assertion
    # is "beyond the clip", not "at index >= 100 of the line", which a truncated line
    # can never satisfy.
    full_description = BURIED_HEAD + BURIED_TRIGGER + "."
    assert ev.trigger_position(full_description) >= clip, (
        f"the fixture must bury its trigger past the clip (got "
        f"{ev.trigger_position(full_description)} against clip {clip})")
    assert ev.trigger_position(prod_desc) < 0, (
        f"production's clipped line must not contain the trigger at all: {prod_desc!r}")
    assert ev.trigger_position(var_desc) < clip, (
        f"the arm's line must put the trigger inside the clip: {var_desc!r}")
    assert var_desc.startswith(BURIED_TRIGGER[:12]), var_desc
    # What is length-exact is the DESCRIPTION the renderer was handed, and that is the
    # quantity the budget walk measures. The rendered LINE can differ by up to two
    # characters even when nothing was re-spaced: the clip rstrips its cut point, so
    # moving a clause can land the cut on a space or on a word. Fixture: production's
    # cut falls after "vault" (rstripped to 99), the variant's mid-word at 99+ellipsis.
    assert len(full_description) == len(
        ev.frontload_description(full_description)), (
        "the rewrite changed a description's length, so the budget walk could move "
        "who is described and the comparison stops being about text position")
    assert abs(len(var_desc) - len(prod_desc)) <= 2, (
        f"a rendered line may move only by the clip's rstrip artefact: "
        f"{len(var_desc)} vs {len(prod_desc)}")
    assert variant[1] == production[1], (
        "a line whose trigger already leads must come through byte for byte — the arm "
        "is not a rewrite-everything arm")


def test_the_rewrite_is_applied_around_the_build_and_restored_after_it():
    """Clause 3: the patch swaps one symbol for the duration of one build.

    Three failure modes, three asserts. (a) A spy on the renderer sees the swapped
    symbol only while the arm's own prompt is being built — proof the arm reaches the
    renderer instead of post-editing a string. (b) Every OTHER arm's prompt is
    byte-identical before and after the arm builds its own: a leaked patch would show up
    as `desc_push` quietly gaining the rewrite, which is the failure that voids a
    published comparison without any test of the variant noticing. (c) The skills tree
    is untouched — the item's constraint is that the rewrite lives in memory, so a run
    of this arm can never be mistaken for the vault edit #4373 has not been measured
    for."""
    import hashlib

    ev = _load_eval()
    from app import prompt_builder

    original = prompt_builder.skill_index_description
    real_lines = prompt_builder._skills_index_lines
    seen: list[str] = []

    def spy(active, counts, **kw):
        seen.append(getattr(prompt_builder.skill_index_description, "__name__", "?"))
        return real_lines(active, counts, **kw)

    def fingerprints() -> dict[str, str]:
        root = Path.home() / "obsidian" / "skills"
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("SKILL.md"))}

    def other_arms() -> dict[str, str]:
        return {arm: ev.system_prompt_for(arm)
                for arm in ev.ARMS if arm != ev.ARM_DESC_FRONTLOAD}

    vault_before = fingerprints()
    before = other_arms()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(prompt_builder, "_skills_index_lines", spy)
    try:
        arm_prompt = ev.system_prompt_for(ev.ARM_DESC_FRONTLOAD)
        ev.system_prompt_for(ev.ARM_DESC_PUSH)
    finally:
        monkey.undo()

    assert seen == ["frontloaded_index_description", "skill_index_description"], (
        f"the renderer must see the swapped symbol during the arm's build and the real "
        f"one during every other arm's: {seen}")
    assert prompt_builder.skill_index_description is original, (
        "the patch leaked: the symbol is still swapped after the arm finished, so "
        "every later prompt would carry the rewrite")
    assert other_arms() == before, (
        "another arm's prompt changed once the new arm existed — a control that "
        "absorbs the variant's text measures nothing")
    assert fingerprints() == vault_before, (
        "the arm wrote under ~/obsidian/skills/")
    assert "Available skills" in arm_prompt


def test_the_frontload_arm_describes_exactly_the_skills_desc_push_describes():
    """Clause 4, over the live corpus: the set of skills carrying a description is
    production's set, the block's total length is unchanged, and the arm rewrites
    something — the last being what stops the first two being satisfied by an arm that
    does nothing.

    The confound the item names: `budget_chars` is 12,000 and descriptions are granted
    in usage-rank order while the whole index fits, each candidate measured by its
    clipped length. A rewrite that shortened 30 descriptions would let more skills in,
    and a measured gain would then be about who got described — the ~140 names-only
    question — not about where the trigger sits. The rewrite is a permutation, so
    description lengths are identical by construction; the one residual is the clip's
    rstrip artefact on a cut line, pinned at two characters per line and netting to zero
    across the block."""
    ev = _load_eval()
    production = ev.advertised_index_block(ev.ARM_DESC_PUSH)
    variant = ev.advertised_index_block(ev.ARM_DESC_FRONTLOAD)
    prod_named = ev.described_skill_names(production)
    var_named = ev.described_skill_names(variant)

    assert len(prod_named) > 20, (
        f"production grants a description to only {len(prod_named)} skills — the "
        "comparison would be vacuous")
    assert var_named == prod_named, (
        f"the arm moved WHO is described (+{sorted(var_named - prod_named)} "
        f"-{sorted(prod_named - var_named)}); the measured difference would be about "
        f"the budget walk, not about description text")
    assert len(variant) == len(production), (
        f"the advertised block changed total length ({len(variant)} vs "
        f"{len(production)}) while the described set stayed equal")
    worst = max((abs(len(a) - len(b)) for a, b in
                 zip(production.splitlines(), variant.splitlines())), default=0)
    assert worst <= 2, (
        f"a rendered line moved by {worst} characters; the only permitted drift is the "
        f"clip's rstrip of its cut point, which is at most two")

    changed = sum(1 for a, b in zip(production.splitlines(), variant.splitlines())
                  if a != b)
    tally = dict(ev.FRONTLOAD_LAST_TALLY)
    assert changed > 0 and tally.get("rewritten", 0) > 0, (
        f"the arm rewrote nothing ({changed} lines changed, tally {tally}) — an arm "
        f"that never fires would satisfy every other clause in this file")


def _frontload_records(*, var_reached: int = 14, prod_judge: int = 3,
                       var_judge: int = 4, with_judge: bool = True) -> list[dict]:
    """Twenty labelled turns, shaped exactly as `run` writes them, for the #4375 verdict.

    Every number the verdict reads is one `summarize()` COMPUTES from these records: the
    rates from `right_skill_reached` flags (injected 12/20, `injected_aa` 11/20 for the
    A/A spread, `desc_push` 12/20, the variant `var_reached`/20) and the judge from
    `judge_score`. That is deliberate, and it is the correction the review rung forced:
    the first fixture here was a hand-typed summary dict that invented the key
    `judge_mean`, while `summarize()` writes `judge_score_avg` — so the report read a key
    no artifact has, both judges came back None, the judge condition passed vacuously, and
    the test stayed green on a rule that could not fail against a real run. Building the
    input through `summarize()` means a key rename here goes red instead of silently
    testing a shape no artifact will ever have.
    """
    expected = sorted(ev.advertised_skill_names(ev.ARM_INJECTED))[0]
    reached = {ev.ARM_INJECTED: 12, ev.ARM_AA: 11, ev.ARM_DESC_PUSH: 12,
               ev.ARM_DESC_FRONTLOAD: var_reached}
    judge = {ev.ARM_INJECTED: prod_judge, ev.ARM_AA: prod_judge,
             ev.ARM_DESC_PUSH: prod_judge, ev.ARM_DESC_FRONTLOAD: var_judge}
    records: list[dict] = []
    for i in range(20):
        arms: dict[str, dict] = {}
        for arm, hits in reached.items():
            arm_rec: dict = {"completed": 1, "doc_hit": 0, "injected_tokens": 100,
                             "iterations": 1, "tool_calls": 0, "seconds": 0.5,
                             "right_skill_reached": int(i < hits)}
            if with_judge:
                arm_rec["judge_score"] = judge[arm]
            arms[arm] = arm_rec
        records.append({"id": f"q{i:02d}", "category": "skill", "query": f"q{i:02d}",
                        "expected_skills": [expected], "arms": arms})
    return records


def test_a_disabled_rewrite_renders_production_and_the_report_says_so():
    """Clause 5. With the rewrite off the arm's prompt is production's bytes AND the
    report says so; with it on, the per-arm report carries the metrics the shipped rule
    names and states that rule — including that a judge it cannot see is never read as a
    judge that passed.

    The byte-equality is the anti-no-op proof: an arm whose transformation silently
    stopped firing would otherwise keep reporting a number forever. The report has to SAY
    that equality out loud too, because a table whose variant row equals production's,
    with no sentence naming which case produced it, reads as a null result when it is a
    disabled arm. The disabled flag it reads is the one `summarize()` computed for itself
    under `FRONTLOAD_ENABLED = False`, not one this test typed in.
    """
    ev = _load_eval()

    monkey = pytest.MonkeyPatch()
    monkey.setattr(ev, "FRONTLOAD_ENABLED", False)
    disabled_prompt = ev.system_prompt_for(ev.ARM_DESC_FRONTLOAD)
    disabled_summary = ev.summarize(_frontload_records())
    monkey.undo()

    assert disabled_prompt == ev.system_prompt_for(ev.ARM_DESC_PUSH), (
        "a disabled variant must equal production byte for byte; if it differs, the arm "
        "differs from production in something besides description text")
    computed = disabled_summary["arms"][ev.ARM_DESC_FRONTLOAD]
    assert computed["frontload_prompt_equals_production"] is True, (
        "summarize() did not itself compute the disabled arm's prompt as production's, so "
        "the flag the report reads would be a test's assertion rather than a measurement")
    assert computed["frontload_rewrite_enabled"] is False

    # FRONTLOAD_ENABLED is back to True here: the report has to reach the disabled
    # sentence from the flag IN the artifact, which is what a real `--frontload off` run
    # publishes and its reader sees.
    shown = ev.frontload_report(disabled_summary)
    shown_text = "\n".join(shown["lines"])
    assert shown["verdict"].startswith("not measured"), shown["verdict"]
    assert "THE REWRITE WAS DISABLED" in shown_text, shown_text
    assert "byte-identical" in shown_text.lower(), shown_text

    summary = ev.summarize(_frontload_records())
    report = ev.frontload_report(summary)
    text = "\n".join(report["lines"])
    arm = summary["arms"][ev.ARM_DESC_FRONTLOAD]
    prod = summary["arms"][ev.ARM_DESC_PUSH]
    # The rates the verdict is built on, read back out of what summarize() computed:
    # injected 12/20 and injected_aa 11/20 make a 0.05 A/A spread, desc_push 12/20 is
    # production's 0.6, and the variant's 14/20 is a 0.1 gain over it.
    assert report["production_rate"] == prod["right_skill_reached_rate"] == 0.6, report
    assert report["variant_rate"] == arm["right_skill_reached_rate"] == 0.7, report
    assert report["aa_spread"] == 0.05, report
    assert report["production_judge"] == prod["judge_score_avg"] == 3.0, report
    assert report["variant_judge"] == arm["judge_score_avg"] == 4.0, report
    assert report["verdict"].startswith("better"), report["verdict"]
    assert "more than the A/A spread" in report["verdict"], report["verdict"]
    for needle in (f"right_skill_reached_rate {report['production_rate']} -> "
                   f"{report['variant_rate']}",
                   f"A/A noise spread {report['aa_spread']}",
                   f"judge {report['production_judge']} -> {report['variant_judge']}",
                   f"(n={arm['right_skill_reached_tail_n']})",
                   f"described skills: {prod['described_skill_count']} -> "
                   f"{arm['described_skill_count']}"):
        assert needle in text, f"missing from the report: {needle!r}\n{text}"

    # A gain of exactly the A/A spread is one prompt measured twice, not a result.
    inside = ev.frontload_report(ev.summarize(_frontload_records(var_reached=13)))
    assert inside["verdict"].startswith("not better"), inside["verdict"]
    assert "inside the A/A spread" in inside["verdict"], inside["verdict"]

    # Judge lower: the rule is a conjunction, and this is the case the old
    # `(var_judge or 0) >= (prod_judge or 0)` could not fail on a real artifact.
    lower = ev.frontload_report(ev.summarize(
        _frontload_records(var_reached=14, var_judge=2)))
    assert lower["verdict"].startswith("not better"), lower["verdict"]
    assert "judge fell" in lower["verdict"], lower["verdict"]

    # The hole the review found: an artifact whose judge never came back has
    # `judge_score_avg: None` on both sides, and must NEVER read as a win.
    no_judge = ev.frontload_report(ev.summarize(_frontload_records(with_judge=False)))
    assert no_judge["production_judge"] is None and no_judge["variant_judge"] is None
    assert not no_judge["verdict"].startswith("better"), no_judge["verdict"]
    assert no_judge["verdict"].startswith("incomplete"), no_judge["verdict"]


def test_main_calls_only_helpers_defined_above_the_entrypoint() -> None:
    """`main()` runs AT the `if __name__ == "__main__"` line in script mode, so every
    helper it calls has to be bound ABOVE that line — which means the guard has to be the
    last statement in the file.

    This is #4375's own bug, found on this round and not in the item. `frontload_report`
    was appended below the guard, so `--arms desc_frontload` raised
    `NameError: name 'frontload_report' is not defined` at the one statement that prints
    the verdict — AFTER the run had spent its GPU turns and written its artifact. Every
    test above passes regardless, because importing the module for a test executes the
    whole body including the late def; only script mode hits it. Script mode is how the
    measured run #4375 owes will actually be invoked.

    Pinned structurally rather than by launching the script: the failure costs a full
    eval run to reach, and a subprocess of this script inside the suite would either run
    turns through an engine or depend on query-set contents to reach the print.
    """
    import ast

    tree = ast.parse((REPO / "eval" / "run_prefetch_cost_eval.py").read_text())

    def is_main_guard(node: ast.stmt) -> bool:
        return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and any(isinstance(c, ast.Constant) and c.value == "__main__"
                        for c in node.test.comparators))

    guard = next((i for i, n in enumerate(tree.body) if is_main_guard(n)), None)
    assert guard is not None, "the script's entrypoint guard is gone from the file"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "main"
               for n in tree.body[:guard]), "main() is not defined above the guard"
    late = [getattr(n, "name", type(n).__name__)
            for n in tree.body[guard + 1:]
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert not late, (
        f"{late} sit below the `if __name__ == \"__main__\"` guard, so script mode has "
        f"not bound them when main() runs — a run under `--arms desc_frontload` dies at "
        f"the line that prints its verdict, after the artifact is already on disk")


# ── #4394: this runner's plural `--arms` is the correct one ──────────────────

def test_config_yaml_still_prints_this_runners_plural_arms():
    """#4394 clause 4: the fix that re-pointed a config comment at `--arm` must leave
    THIS command line alone.

    `config.yaml` carries two prefetch A/B command lines and they disagree on purpose.
    The follow-up runner declares `ap.add_argument("--arm", dest="arms", ...)`; this one
    declares the plural, so `--arms injected,desc_push,desc_pull` is not a typo and a
    reader who sweep-replaced the token would have broken a working command line while
    fixing a broken one. That is the trap this item named, so the surviving line is
    pinned from the config's bytes and from this runner's own parser surface, read off
    its AST rather than from a grep of its text: the claim being graded is that the flag
    the comment prints is the flag the script accepts, and only the declaration says so.

    `--arms raw,focus` may not appear in config.yaml at all — that is
    `tests/test_arch_review_config_comment_claims.py`'s file-wide ban — but the count
    assertion belongs here, because "exactly one `--arms` is left in the file" is a
    claim about this line being the one that survived.
    """
    import ast

    cfg = (REPO / "config.yaml").read_text(encoding="utf-8")
    assert "eval/run_prefetch_cost_eval.py --arms injected,desc_push,desc_pull" in cfg, (
        "config.yaml no longer prints the cost-eval command line: #4394's fix was aimed "
        "at a different runner's flag")
    assert cfg.count("--arms") == 1, (
        f"{cfg.count('--arms')} `--arms` tokens are left in config.yaml; exactly this "
        "runner's line should survive #4394's change, so a second one means the wrong "
        "string was edited or a new one was written")
    assert "--arms raw,focus" not in cfg

    tree = ast.parse((REPO / "eval" / "run_prefetch_cost_eval.py").read_text(
        encoding="utf-8"))
    flags = {a.value for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "add_argument"
             for a in node.args
             if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    assert "--arms" in flags, (
        "this runner no longer declares the plural this config line prints, so the line "
        "above has become the bug #4394 fixed somewhere else")
    assert "--arm" not in flags, (
        "this runner now declares both `--arm` and `--arms`, so the config line's flag is "
        "no longer unambiguous and the pin above means nothing")


# ------------------------------------------------------------------------ #
# The desc_frontload NON-RUN note (#4375's run, published by #4460)
# ------------------------------------------------------------------------ #
#
# The note is a null with a bound on it: the 78 x 4 measured run was RULED NOT SPENT, so
# there is no result to pin and everything checkable is either a property of the prose (it
# says a non-run, it dates and instruments each figure, its arithmetic closes) or a number
# that a live render can still contradict. The rotating counts — the described set, the
# 209/210 line totals, the 11,970-char block — are deliberately NOT asserted equal to
# today's values: `app/prompt_builder.py` grants descriptions in rank order against the
# 12,000-char budget, so they move with the daily chat telemetry and an equality pin goes
# red on the first rotation. What is pinned instead is the shape of the claim and the
# direction that matters: the published render delta is a CEILING, so a live delta that
# grows past it reddens this file and one that shrinks does not.

NON_RUN_NOTE = REPO / "eval" / "measurements" / "desc-frontload-4375-2026-11-21.md"

#: The runtime probe the note cites as script-mode proof. Untracked, outside the repo:
#: naming the path is the #3989 ruling shape, so this is a string, never a read. It is
#: the path's tail because the note may name it absolute or with `~`, and the claim
#: pinned below is that it names it at all — not which notation it chose.
PROBE_ARTIFACT = "eval/baselines/frontload-probe-4375-20261121-014037.json"


def _note_text() -> str:
    assert NON_RUN_NOTE.exists(), (
        f"{NON_RUN_NOTE.relative_to(REPO)} is the published standing of #4375's "
        "ruled-not-spent run; without it a later pass re-proposes the hours")
    return NON_RUN_NOTE.read_text(encoding="utf-8")


def _note_register() -> dict[str, str]:
    """The note's `key: value` register, so a figure is read and not re-parsed."""
    for block in re.findall(r"```\n(.*?)\n```", _note_text(), re.S):
        if "render_delta_ceiling:" in block:
            return {k.strip(): v.strip()
                    for k, _, v in (line.partition(":")
                                    for line in block.splitlines() if ":" in line)}
    raise AssertionError("the note lost its machine-readable register block")


def _live_render_pair(ev) -> tuple[int, int, int]:
    """(differing index lines, desc_push chars, desc_frontload chars), as the arms run.

    Forces the switch on rather than inheriting it: a sibling node exercises the
    disabled render and restores it in a `finally`, and the ceiling compares the two
    ENABLED renders.
    """
    import difflib

    was = ev.FRONTLOAD_ENABLED
    ev.FRONTLOAD_ENABLED = True
    try:
        ev._INDEX_BLOCK_CACHE.clear()
        push = ev.advertised_index_block("desc_push")
        front = ev.advertised_index_block("desc_frontload")
    finally:
        ev.FRONTLOAD_ENABLED = was
        ev._INDEX_BLOCK_CACHE.clear()
    pl, fl = push.splitlines(), front.splitlines()
    changed = sum(max(i2 - i1, j2 - j1)
                  for tag, i1, i2, j1, j2
                  in difflib.SequenceMatcher(None, pl, fl).get_opcodes()
                  if tag != "equal")
    return changed, len(push), len(front)


def test_the_desc_frontload_note_opens_with_a_ruled_not_spent_non_run():
    """#4460 clause 1: the note exists, is tracked, and says NON-RUN in its opening.

    The half that can rot silently is the second one: a note that quietly grew a
    per-arm table would read like a measured run, and nothing else in the repo could
    tell. So this node refuses any table and the two outcome figures a non-run cannot
    have — `right_skill_reached_rate` and `judge_score_avg` — while the tail-n rows the
    note DOES carry are statements about construction, quoted by name below.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch",
         str(NON_RUN_NOTE.relative_to(REPO))],
        capture_output=True, text=True, check=False)
    assert tracked.returncode == 0, (
        f"{NON_RUN_NOTE.relative_to(REPO)} is on disk but not tracked, so it has no "
        f"history and the null it publishes can be swept: {tracked.stderr.strip()}")

    text = _note_text()
    head = "\n".join(text.splitlines()[:16])
    for needle in ("NON-RUN", "RULED NOT SPENT", "78 queries", "4 arms", "primary",
                   "#4375", "never been executed"):
        assert needle in head, (
            f"the opening of the note must state {needle!r} — a reader who stops at "
            f"the first screen has to learn it is a non-run there, not on page two; "
            f"opening was: {head[:320]!r}")

    tables = [ln for ln in text.splitlines() if ln.lstrip().startswith("|")]
    assert not tables, (
        f"the note carries {len(tables)} table row(s); a per-arm table is exactly what "
        "a run that never happened cannot have, and its presence would read as a "
        f"measured result. First row: {tables[:1]}")
    for forbidden in ("right_skill_reached_rate", "judge_score_avg"):
        assert forbidden not in text, (
            f"{forbidden} is an outcome figure and this run was not spent; the note "
            "may cite it only as 'none exists', which is how it cites the tail rows")


def test_every_figure_in_the_note_is_dated_and_names_its_instrument():
    """#4460 clause 2: each published count is dated and instrumented.

    The failure this guards is a carried-forward number read as a measurement — the
    note's own clause 6 says no count may arrive without the day it was measured and
    the tool that printed it, so the Provenance section must carry one bullet per
    figure naming both.
    """
    text = _note_text()
    assert "re-measured on 2026-11-25" in text, (
        "the note must state the day its counts were taken, distinct from the "
        "2026-11-21 date in its own filename (the ruling's date)")

    marker = "## Provenance of every figure in this note"
    assert marker in text, "the note lost the section that dates its figures"
    prov = text.split(marker, 1)[1].split("\n## ", 1)[0]
    bullets = [b for b in prov.split("\n- ") if b.strip()]
    instruments = ("#4373", ".venvs/lloyd/bin/python", "advertised_index_block",
                   "difflib", "frontload-probe-4375-20261121-014037.json",
                   "wc -l -c", "SM_20261125_222702")
    for figure in ("clipped 68", "trigger-leading 37", "11,970", "11,989",
                   "199.5", "42.5", "concurrency 4", "195 10521"):
        holder = [b for b in bullets if figure in b]
        assert holder, (
            f"no Provenance bullet covers the figure {figure!r}, so a reader cannot "
            "tell what printed it or when — clause 6 forbids carrying it")
        assert any("2026-11-" in b for b in holder), (
            f"the bullet covering {figure!r} names no date")
        assert any(any(tok in b for tok in instruments) for b in holder), (
            f"the bullet covering {figure!r} names no instrument")


def test_the_published_cost_bound_is_self_consistent_and_caveats_one_sample():
    """#4460 clause 3: 78 x 4 = 312 arm-runs at 199.5 s / concurrency 4 closes to 4.3 h.

    ±0.2 h, parsed from the note rather than restated here. The second half is the
    caveat the figure needs: 199.5 s is ONE turn of the slower arm that hit
    --max-turns 8 and did not complete, so the note must also publish the faster
    arm's own sample (42.5 s) and the ~0.9 h it implies, or a later pass re-derives
    0.9 h and files this note as wrong.
    """
    reg = _note_register()
    queries, arms = int(reg["cost_queries"]), int(reg["cost_arms"])
    runs = int(reg["cost_arm_runs"])
    seconds = float(reg["cost_seconds_per_arm_run"])
    concurrency = int(reg["cost_concurrency"])
    hours = float(reg["cost_hours_upper"])
    faster = float(reg["cost_seconds_faster_sample"])
    lower = float(reg["cost_hours_lower"])

    assert queries == 78 and arms == 4
    assert queries * arms == runs == 312
    assert abs(runs * seconds / concurrency / 3600.0 - hours) <= 0.2, (
        f"{runs} arm-runs x {seconds} s / {concurrency} = "
        f"{runs * seconds / concurrency / 3600.0:.2f} h, not the published {hours} h")
    assert abs(runs * faster / concurrency / 3600.0 - lower) <= 0.2, (
        f"the faster arm's sample works out to "
        f"{runs * faster / concurrency / 3600.0:.2f} h, not the published {lower} h")

    bound = [b for b in _note_text().split("\n- ") if "199.5" in b]
    assert bound and any("one sample" in b and "slower" in b and "injected" in b
                         and "max_turns" in b for b in bound), (
        "the bullet publishing 199.5 s must say it is a single sample of the slower "
        "`injected` arm run to --max-turns 8; without that it reads as a measured mean")
    text = _note_text()
    assert "upper bound" in text and "42.5" in text, (
        "the note must call the bound an upper bound and name the other sample")


def test_the_published_render_delta_ceiling_holds_against_a_live_rerender():
    """#4460 clause 4: the live delta stays <= the published ceiling, char-neutral.

    Re-renders both blocks exactly as the two arms do and diffs them with difflib. The
    published ceiling is the largest delta reachable on a tree that cannot see
    production's usage rank (a round worktree: `app.paths` anchors the data root inside
    itself, so `_skill_rank_counts()` is empty and the granted set is the fallback
    order), which is every tree a test run opens — production's own reading today is 1
    line of 209. So growth past the ceiling means the granted set moved and the note
    needs a dated re-publication; shrink is free.
    """
    reg = _note_register()
    ceiling = int(reg["render_delta_ceiling"])
    changed, push_chars, front_chars = _live_render_pair(_load_eval())

    assert int(reg["push_block_chars"]) == int(reg["frontload_block_chars"]), (
        "the note publishes two unequal block totals, which would mean the arm "
        "changes the index's length — it does not: the rewrite is length-exact")
    assert push_chars == front_chars, (
        f"the live renders differ in length ({push_chars} vs {front_chars}); the arm's "
        "rewrite is length-exact, so this is a real regression, not a rotation")
    assert changed <= ceiling, (
        f"live <available_skills> delta is {changed} lines but the note publishes a "
        f"ceiling of {ceiling}: the granted set has moved. Re-publish the number with "
        "a new date (owed on #4460) — this node is the alarm, not the bug")
    assert 1 <= int(reg["render_delta_live"]) <= ceiling
    assert ceiling <= int(reg["render_delta_structural_cap"]), (
        "the ceiling cannot exceed the tally's rewritten count: no line differs whose "
        "description the arm does not rewrite")
    text = _note_text()
    assert "advertised_index_block" in text and "difflib" in text, (
        "clause 4 requires the note to say WHICH two renders were compared")


def test_the_note_names_the_line_the_arm_renders_worse_and_what_cannot_be_answered():
    """#4460 clause 5: the worse line, the by-construction zero, the attribution.

    Four separate claims, each pinned: the named skill whose front-load is a caution
    rather than a trigger (re-derived live from the arm's own rewriter, so the note
    cannot outlive the behaviour it describes); `right_skill_reached_tail_n` 0 by
    construction with only `described_skill_count` bearing on who gets a description;
    trigger-leading 24 -> 36 credited to #4373's landed vault rewrite; and the probe
    artifact cited as script-mode proof while the `--frontload off` byte-identity is
    attributed to the in-process render comparison, because that artifact ran ENABLED
    (`frontload_prompt_equals_production: false`) and so proves nothing about it.
    """
    text = _note_text()
    sections = text.split("\n## ")
    worse = [s for s in sections if "fact-improvement" in s]
    assert worse and any("worse" in s and "caution" in s and "not a trigger" in s
                         for s in worse), (
        "the note must name the skill whose line the arm changes and say the arm's "
        "version is worse because the front-loaded clause is a caution, not a trigger")

    tail = [s for s in sections if "right_skill_reached_tail_n" in s]
    assert tail and any("by construction" in s and "described_skill_count" in s
                        for s in tail), (
        "the tail population is empty for an arm that advertises every skill, so the "
        "note has to say the zero is by construction and that only "
        "described_skill_count bears on who gets a description")

    assert "24 → 36" in text and "#4373" in text and "20e4764a8" in text, (
        "trigger-leading 24 → 36 must be attributed to #4373's landed vault rewrite "
        "(vault commit 20e4764a8), which is what collapsed this arm's population")

    assert PROBE_ARTIFACT in text, (
        f"the note must cite {PROBE_ARTIFACT} as the script-mode proof")
    off = [b for b in text.split("\n- ") if "--frontload off" in b]
    assert off and any(("in-process" in b or "FRONTLOAD_ENABLED = False" in b)
                       and "frontload_prompt_equals_production: false" in b
                       for b in off), (
        "the byte-identical disabled render belongs to the in-process comparison, not "
        "to the probe: the probe ran with the rewrite enabled")

    # The 'worse' claim, re-derived rather than trusted: the arm's own new leading
    # clause for fact-improvement must fail #4373's trigger-leading test.
    from agent_mcp.skills import iter_active_skills
    from app.prompt_builder import _skill_walk_roots, skill_index_description

    described = {a.name: skill_index_description(a.frontmatter)
                 for a in iter_active_skills(overlay=None, roots=_skill_walk_roots())}
    assert "fact-improvement" in described, (
        "fact-improvement is not in the live skill set, so the note's named line and "
        "this node's check have both gone stale")
    out = _load_eval().frontload_description(described["fact-improvement"])
    assert out != described["fact-improvement"], (
        "the arm no longer rewrites fact-improvement, so the note's named worse line "
        "is stale and needs a dated re-publication")
    assert len(out) == len(described["fact-improvement"])
    assert not re.match(r"^\s*Use (when|to|for)\b", out, re.I), (
        "the clause the arm now front-loads onto fact-improvement is trigger-leading, "
        "so 'the arm's version is worse' is no longer the right claim")
