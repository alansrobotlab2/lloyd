"""#3850: the action-preservation scorer for tool-observation compressors, offline.

Every node drives `eval/run_observation_compression_eval.py` end-to-end through an
INJECTED STUB engine — no live engine is contacted, no socket is opened, no request
leaves the process. The numbers asserted are the ones the stub was scripted to
emit, so a node fails the moment the scorer reads a logprob wrongly, books a
missing readout as a zero, folds a decline into a tier, paraphrases production's
rung 4 instead of calling it, or re-reads a session directory it was supposed to
have pinned.

Clause map (the item's five acceptance clauses):
  1  `test_scoring_one_row_returns_delta_ce_kl_tier_and_tokens_retained`,
     `test_each_named_compressor_scores_through_the_same_entry_point`
  2  `test_missing_prompt_logprobs_invalidates_the_row_and_is_excluded_from_every_aggregate`,
     `test_zero_scored_positions_invalidates_rather_than_reading_as_agreement`
  3  `test_harvester_joins_the_sidecar_verbatim_and_scoring_survives_its_removal`
  4  `test_the_rung4_arm_is_byte_identical_to_production_own_output`
  5  `test_a_generated_step_with_no_call_is_a_decline_beside_the_tier`,
     `test_an_unreadable_call_is_protocol_invalid_and_not_the_decline_bucket`
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_observation_compression_eval", ROOT / "eval" / "run_observation_compression_eval.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

# Importable only after `R` ran, because `R` is what puts `eval/` on sys.path.
from action_equivalence import (  # noqa: E402
    TIERS,
    ARGS_DIFFER,
    CANONICALLY_EQUAL,
    DIFFERENT_TOOL,
    PROTOCOL_INVALID,
)

SID = "20261106_120000_fixture"
CID = "chatcmpl-tool-fixture0001"
POINTER = "<persisted-output> 2 KB preview only</persisted-output>"

#: Scripted engine answers. Every node reuses these so a hand-check carries.
TOPK_FULL = {"A": 0.0, "B": -1.0}
TOPK_CAND = {"A": -0.5, "C": -2.0}
# KL(P_full || P_cand) by hand off those two dicts and `kl_first_position`'s floor
# rule (a token outside the candidate's list is charged that list's LOWEST logprob,
# -2.0):  e^0.0*(0.0 - -0.5)  +  e^-1.0*(-1.0 - -2.0)  =  0.5 + 0.367879…
KL_HAND = 1.0 * 0.5 + math.exp(-1.0) * 1.0
FULL_NLL, CAND_NLL = 1.25, 2.0
DELTA_HAND = CAND_NLL - FULL_NLL  # 0.75 nats


# ---------------------------------------------------------------------------
# The seam: a stub engine, scripted per arm, that never opens a socket
# ---------------------------------------------------------------------------

class StubEngine:
    """A `/v1/completions` stand-in with four scripted answers and no transport.

    The arm is told apart by the observation's own bytes: a prompt that still
    contains them is the FULL arm, one that does not is the candidate's. That is
    the same discriminator a real run has, so the stub cannot be steered by a
    keyword the scorer invented.

    `emit_logprobs=False` reproduces the failure #3850 refuses on: an answer with
    no teacher-forced readout at all. `all_masked` reproduces the subtler one — a
    readout of the right length whose every position is `null`, which is a
    zero-position score, not a zero-cost one. `positions` pins the span length to
    trip the short-span refusal.
    """

    route_label = "stub:injected"

    def __init__(self, *, observation: str, full_nll: float, cand_nll: float,
                 full_step: str, cand_step: str,
                 full_topk: dict | None = None, cand_topk: dict | None = None,
                 emit_logprobs: bool = True, all_masked: bool = False,
                 positions: int | None = None, missing_arm: str = "") -> None:
        self.observation = observation
        self.full_nll = full_nll
        self.cand_nll = cand_nll
        self.full_step = full_step
        self.cand_step = cand_step
        self.full_topk = full_topk or {}
        self.cand_topk = cand_topk or {}
        self.emit_logprobs = emit_logprobs
        self.all_masked = all_masked
        self.fixed_positions = positions
        #: "full" or "candidate": withhold the readout from ONE arm, which is what
        #: an engine does when the shorter prompt is fine and the longer one crossed
        #: its window. The row is then uncomparable, and must say which half is missing.
        self.missing_arm = missing_arm
        self.prompts: list[tuple[str, str]] = []

    def _is_full(self, prompt: str) -> bool:
        return self.observation in prompt

    def _arm(self, prompt: str) -> str:
        return "full" if self._is_full(prompt) else "candidate"

    def teacher_forced(self, prompt: str, *, top_k: int = 8) -> dict:
        self.prompts.append(("teacher_forced", prompt))
        if not self.emit_logprobs or self.missing_arm == self._arm(prompt):
            # The real failure this clause names: a 200 whose choice has no readout.
            return {"text": "", "prompt_logprobs": None}
        full = self._is_full(prompt)
        nll = self.full_nll if full else self.cand_nll
        topk = self.full_topk if full else self.cand_topk
        if self.all_masked:
            return {"text": "", "prompt_logprobs": [None] * 1024}
        n = self.fixed_positions if self.fixed_positions is not None \
            else min(max(len(prompt) // 4, 512), 4096)
        body = {"token_logprob": -nll,
                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in topk.items()]}
        return {"text": "", "prompt_logprobs": [None] + [dict(body) for _ in range(n - 1)]}

    def step(self, prompt: str, *, max_tokens: int = 64) -> dict:
        self.prompts.append(("step", prompt))
        return {"text": self.full_step if self._is_full(prompt) else self.cand_step}


#: A toy token width for the surfaced stub, deliberately NOT `CHARS_PER_TOKEN`: at
#: four chars per token the estimated span and the aligned span would be the same
#: width on this fixture, and the node that tells them apart could not.
STUB_TOKEN_CHARS = 8

#: What a position OUTSIDE the action scores. No action position carries it, so a
#: mean that contains it is visibly not the action's.
POISON_NLL = 9.0

#: The top-k at every position except the action's first. Identical in both arms, so
#: reading the KL there yields 0.0 — the number a wrong position reports.
TOPK_OTHER = {"other-token": -1.0}


class SurfacedStubEngine(StubEngine):
    """A stub that labels its own positions, so WHICH window was read is visible.

    `StubEngine` scripts the same body at every position — right for the arm
    comparisons, and the reason an alignment defect survived a green suite: nothing in
    its answers depends on where the window opened. Here every position carries the
    text it stands for, chunked `STUB_TOKEN_CHARS` characters at a time measured back
    from the end of the prompt, and the bodies are scripted by REGION:

    * the action's own positions score `full_nll` / `cand_nll`, and at the action's
      FIRST position the top-k is the per-arm `full_topk` / `cand_topk`, so
      `kl_first_position` comes to KL_HAND only if that position is the one read;
    * every other position scores `POISON_NLL` with `TOPK_OTHER`, the same in both
      arms — so a window that opens before the action moves the mean onto a value no
      action position has, and the KL comes back 0.0.

    `labelled=False` reproduces the other readout shape, a bare token id with no
    text, which is what sends the span to the estimated route.
    """

    route_label = "stub:surfaced"

    def __init__(self, *, action_text: str, labelled: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.action_text = action_text
        self.labelled = labelled

    def teacher_forced(self, prompt: str, *, top_k: int = 8) -> dict:
        self.prompts.append(("teacher_forced", prompt))
        if not self.emit_logprobs:
            return {"text": "", "prompt_logprobs": None}
        if not prompt.endswith(self.action_text):
            raise AssertionError("the surfaced stub only scripts teacher-forced prompts")
        full = self._is_full(prompt)
        action_nll = self.full_nll if full else self.cand_nll
        action_topk = self.full_topk if full else self.cand_topk
        action_start = len(prompt) - len(self.action_text)
        edges = list(range(len(prompt), 0, -STUB_TOKEN_CHARS))
        edges.append(0)
        edges = sorted(set(edges))
        positions: list[dict] = []
        for index in range(len(edges) - 1):
            start, stop = edges[index], edges[index + 1]
            in_action = stop > action_start
            is_first = in_action and start <= action_start < stop
            topk = action_topk if is_first else TOPK_OTHER
            positions.append({
                "token": prompt[start:stop] if self.labelled else index,
                "token_logprob": -(action_nll if in_action else POISON_NLL),
                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in topk.items()],
            })
        return {"text": "", "prompt_logprobs": positions}


# ---------------------------------------------------------------------------
# A fixture sessions tree: one pair inside the size band, one under it
# ---------------------------------------------------------------------------

def _observation(tag: str, lines: int = 100) -> str:
    """Deterministic observation text; 100 lines clears the item's 8,000-byte band."""
    return "\n".join(
        f"{tag} line {i:03d}: the spill sidecar is named by the tool_call_id and "
        f"the relief ladder records which rung fired for session {SID}"
        for i in range(lines)
    )


def _session_messages(tool_content: str) -> list[dict]:
    """A transcript shaped like a real session record.

    The tool row's own `content` is a `<persisted-output>` pointer, NOT the
    observation, so a harvester that read the record instead of the sidecar is
    caught by the verbatim assertion rather than passing on similar bytes.
    """
    return [
        {"role": "system", "content": "You are Lloyd.", "timestamp": "2026-11-06T12:00:00"},
        {"role": "user", "content": "Why did the spill grow?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": CID, "type": "function",
            "function": {"name": "Bash",
                         "arguments": json.dumps({"command": "du -sh sessions"})}}]},
        {"role": "tool", "tool_call_id": CID, "content": tool_content},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "chatcmpl-tool-nextstep0002", "type": "function",
            "function": {"name": "Grep",
                         "arguments": json.dumps({"pattern": "truncate"})}}]},
    ]


def _make_sessions(tmp_path: Path) -> tuple[Path, str]:
    """Write a fresh `sessions/<sid>.json` + `.tool-results/` tree; return (root, big)."""
    root = tmp_path / "sessions"
    root.mkdir(exist_ok=True)
    big, small = _observation("BIG"), _observation("SMALL", lines=6)
    (root / f"{SID}.tool-results").mkdir(exist_ok=True)
    (root / f"{SID}.tool-results" / f"{CID}.txt").write_text(big, encoding="utf-8")
    (root / "underband.tool-results").mkdir(exist_ok=True)
    # Named for the tool_call_id the under-band transcript actually carries. Named
    # `cid-under.txt` instead, that session's row dies in `no_sidecar` and the size
    # band the node below says it enforces is never reached.
    (root / "underband.tool-results" / f"{CID}.txt").write_text(small, encoding="utf-8")
    (root / f"{SID}.json").write_text(
        json.dumps({"messages": _session_messages(POINTER)}), encoding="utf-8")
    (root / "underband.json").write_text(
        json.dumps({"messages": _session_messages(POINTER)}), encoding="utf-8")
    return root, big


def _stub(row: R.CorpusRow, **kw) -> StubEngine:
    """A stub whose BOTH arms emit the recorded call, unless a node overrides one."""
    step = R.render_action(row.next_action)
    return StubEngine(observation=row.observation, full_step=step, cand_step=step, **kw)


def _harvested(tmp_path: Path) -> tuple[R.CorpusRow, str]:
    root, big = _make_sessions(tmp_path)
    return R.harvest(root).rows[0], big


# ---------------------------------------------------------------------------
# Clause 1 — the scoring entry point returns all four readings
# ---------------------------------------------------------------------------

def test_scoring_one_row_returns_delta_ce_kl_tier_and_tokens_retained(tmp_path):
    row, big = _harvested(tmp_path)
    eng = _stub(row, full_nll=FULL_NLL, cand_nll=CAND_NLL,
                full_topk=TOPK_FULL, cand_topk=TOPK_CAND)

    rec = R.score_row(row, eng, compressor=R.RUNG4, budget=0.25)

    assert rec["valid"] is True and rec["invalid_reason"] is None
    # ΔCE in nats: the candidate arm's mean action NLL minus the full arm's.
    assert rec["full_arm_nll"] == pytest.approx(FULL_NLL)
    assert rec["candidate_arm_nll"] == pytest.approx(CAND_NLL)
    assert rec["delta_ce_nats"] == pytest.approx(DELTA_HAND)
    assert rec["action_positions_scored"] > 0
    # Top-k KL at the FIRST generated position, hand-checked against KL_HAND.
    assert rec["kl_first_position"] == pytest.approx(KL_HAND)
    # The action_equivalence tier versus the full arm, and the budget columns.
    assert rec["tier"] == CANONICALLY_EQUAL and rec["agrees"] is True
    assert rec["decline"] is None
    assert rec["tokens_retained"] == len(
        R.compress(row, 0.25, compressor=R.RUNG4)[0])
    # …and the column says which unit it is in: characters, not a tokenizer's count.
    assert rec["retained_unit"] == "chars"
    assert rec["tokens_retained"] < len(big.encode("utf-8"))
    assert rec["observation_bytes"] == len(big.encode("utf-8"))
    assert rec["budget_chars"] == int(round(len(big) * 0.25))
    assert rec["row_id"] == f"{SID}#{CID}" and rec["tool_name"] == "Bash"
    # Which route scored this number, taken from the injected client itself.
    assert rec["engine_route"] == "stub:injected"
    # Four requests: both reads of both arms, all through the one injected engine.
    assert [kind for kind, _ in eng.prompts] == [
        "teacher_forced", "teacher_forced", "step", "step"]
    assert all("im_start" in p for _, p in eng.prompts)


def test_each_named_compressor_scores_through_the_same_entry_point(tmp_path):
    """`score_row` prices ANY candidate replacement, not one hard-coded arm.

    The three compressors the item names run through the same entry point with the
    same stub at the same budget, and each records its own arm label and its own
    retained size. The summary arm is an injected callable: the module must not
    contact an engine for it, and must refuse rather than substitute a different
    arm when the callable is missing — a missing arm that silently became
    truncation would report itself as scored.
    """
    row, _big = _harvested(tmp_path)
    summariser = lambda obs, budget_chars: obs[:budget_chars]  # noqa: E731

    recs = {
        c: R.score_row(row, _stub(row, full_nll=1.0, cand_nll=1.5,
                                  full_topk=TOPK_FULL, cand_topk=TOPK_CAND),
                       compressor=c, budget=0.25, summariser=summariser)
        for c in R.COMPRESSORS
    }
    assert {r["arm"] for r in recs.values()} == {
        "production_relief_rung4", "query_conditioned_extractive",
        "secondary_model_summary"}
    assert all(r["valid"] and r["delta_ce_nats"] == pytest.approx(0.5)
               for r in recs.values())
    assert all(r["tier"] == CANONICALLY_EQUAL and r["kl_first_position"]
               == pytest.approx(KL_HAND) for r in recs.values())
    # The two text arms stay INSIDE the budget they were asked for — the frontier
    # compares arms at MATCHED budgets, so an arm that spent its piece budget and
    # then added a joiner per span is not standing at the same x-axis point as the
    # arm it is ranked against — and rung 4 keeps only its fixed-size notice.
    budget_chars = int(round(len(row.observation) * 0.25))
    assert recs[R.EXTRACTIVE]["tokens_retained"] <= budget_chars
    assert recs[R.SUMMARY]["tokens_retained"] == budget_chars
    # …and EXACTLY how far inside it: the fixture's 100 lines are all `unit` chars
    # and the arm fills greedily, so the most it can hold is `units_fit` whole lines
    # plus one joiner between each pair. The joiner is charged to the budget precisely
    # because the uncharged version came back with 3,625 characters at a 3,600
    # allowance, which is a different point on the x-axis than the arm beside it.
    unit = len(row.observation.split("\n")[0])
    units_fit = (budget_chars + 1) // (unit + 1)
    assert recs[R.EXTRACTIVE]["tokens_retained"] == units_fit * unit + (units_fit - 1)
    assert recs[R.RUNG4]["tokens_retained"] < recs[R.EXTRACTIVE]["tokens_retained"]
    # The extractive arm honours the budget at every budget the run uses.
    for b in R.BUDGETS:
        rep, _info = R.compress(row, b, compressor=R.EXTRACTIVE, summariser=summariser)
        assert len(rep) <= int(round(len(row.observation) * b))
    # A missing summariser is a refusal, not a silently-substituted arm.
    with pytest.raises(ValueError, match="summary arm needs a summariser"):
        R.score_row(row, _stub(row, full_nll=1.0, cand_nll=1.5), compressor=R.SUMMARY)


# ---------------------------------------------------------------------------
# Clause 1's other half — WHICH positions the readings are taken from
# ---------------------------------------------------------------------------

def test_the_scored_window_is_the_action_and_not_the_observation_it_sits_on(tmp_path):
    """ΔCE and KL are read over the action's positions and nowhere else.

    `prompt_logprobs` carries one entry per TOKEN; `render_action(...)` is a string
    measured in CHARACTERS, about four of them per token on this model. A window sized
    by the character count is therefore roughly four times as wide as the action and
    opens long before it — inside the tool observation, the one region of the prompt
    the two arms exist to differ in. ΔCE would then average arm-dependent prompt
    positions and call them the cost of the decision, and `kl_first_position` would
    report a position inside the observation while keeping its column name. The review
    of SM_20261106_200241 found exactly that, and found no assertion could see it:
    every stub scripted the same body at every position, so the answers did not depend
    on where the window opened. This stub's do.

    A compressor arm's job is to change the observation's bytes, so this is also the
    only node where the two arms' prompts differ in length — the condition under which
    a window anchored by position index silently scores different text in each arm.
    """
    row, _big = _harvested(tmp_path)
    action = R.render_action(row.next_action)
    eng = SurfacedStubEngine(
        observation=row.observation, action_text=action,
        full_step=action, cand_step=action,
        full_nll=FULL_NLL, cand_nll=CAND_NLL,
        full_topk=TOPK_FULL, cand_topk=TOPK_CAND)

    rec = R.score_row(row, eng, compressor=R.EXTRACTIVE, budget=0.10)

    assert rec["valid"] is True and rec["engine_route"] == "stub:surfaced"
    # Asserted in this order on purpose: the numbers are the clause, and the label
    # below only says which route was allowed to produce them. Against the instrument
    # as it stood at 34f95621 these two fail first — the window was the action's
    # CHARACTER count wide, so POISON_NLL entered the mean and the KL was read at a
    # position inside the observation, where both arms say "other" and agree.
    assert rec["delta_ce_nats"] == pytest.approx(DELTA_HAND)
    assert rec["kl_first_position"] == pytest.approx(KL_HAND)
    # The scripted action values, unpoisoned: no prompt position entered the mean.
    assert rec["full_arm_nll"] == pytest.approx(FULL_NLL)
    assert rec["candidate_arm_nll"] == pytest.approx(CAND_NLL)
    # The window is the one the readout's own token texts allow, no wider.
    assert rec["span_alignment"] == R.SPAN_TOKEN_TEXT
    assert rec["action_positions_scored"] == -(-len(action) // STUB_TOKEN_CHARS)


def test_a_readout_that_does_not_label_its_positions_is_estimated_and_says_so(tmp_path):
    """The fallback route, and the evidence that it is the weaker one.

    Same fixture with the token texts withheld. The estimate at four chars per token
    is wider than the action, so the mean takes in positions outside it and the KL is
    read before the action begins: the numbers a mis-anchored window produces, kept as
    an assertion so the route can never be quoted as if it were the aligned one.
    """
    row, _big = _harvested(tmp_path)
    action = R.render_action(row.next_action)
    eng = SurfacedStubEngine(
        observation=row.observation, action_text=action, labelled=False,
        full_step=action, cand_step=action,
        full_nll=FULL_NLL, cand_nll=CAND_NLL,
        full_topk=TOPK_FULL, cand_topk=TOPK_CAND)

    rec = R.score_row(row, eng, compressor=R.EXTRACTIVE, budget=0.10)

    assert rec["valid"] is True
    assert rec["span_alignment"] == R.SPAN_CHAR_ESTIMATE
    assert rec["action_positions_scored"] > -(-len(action) // STUB_TOKEN_CHARS)
    assert rec["full_arm_nll"] > FULL_NLL          # POISON_NLL positions are in it
    assert rec["kl_first_position"] == 0.0         # read outside the action, where
    assert rec["delta_ce_nats"] != pytest.approx(DELTA_HAND)  # both arms say "other"


# ---------------------------------------------------------------------------
# Clause 2 — no readout is an invalid row, never a zero
# ---------------------------------------------------------------------------

def test_missing_prompt_logprobs_invalidates_the_row_and_is_excluded_from_every_aggregate(
        tmp_path):
    row, _big = _harvested(tmp_path)
    good = R.score_row(row, _stub(row, full_nll=FULL_NLL, cand_nll=CAND_NLL,
                                  full_topk=TOPK_FULL, cand_topk=TOPK_CAND),
                       compressor=R.RUNG4, budget=0.25)
    missing = R.score_row(row, _stub(row, full_nll=FULL_NLL, cand_nll=CAND_NLL,
                                     emit_logprobs=False),
                          compressor=R.RUNG4, budget=0.25)

    assert missing["valid"] is False
    assert missing["invalid_reason"] == f"{R.INVALID_NO_LOGPROBS}:full_arm"
    # The point of the clause: silence is neither a zero nor agreement.
    assert missing["delta_ce_nats"] is None
    assert missing["kl_first_position"] is None
    assert missing["full_arm_nll"] is None and missing["candidate_arm_nll"] is None
    assert missing["tier"] is None and missing["agrees"] is False
    assert missing["action_positions_scored"] == 0

    # One arm answering and the other not is just as uncomparable, and the reason has
    # to say which half went silent: a ΔCE is a difference, and the arm that scored
    # fine is no evidence about the one that produced nothing. An engine does this when
    # one prompt crosses its context window and the other does not — which is the
    # common case here, since the full arm's prompt is longer by the whole observation.
    half = R.score_row(row, _stub(row, full_nll=FULL_NLL, cand_nll=CAND_NLL,
                                  missing_arm="candidate"),
                       compressor=R.RUNG4, budget=0.25)
    assert half["valid"] is False
    assert half["invalid_reason"] == f"{R.INVALID_NO_LOGPROBS}:candidate_arm"
    assert half["delta_ce_nats"] is None and half["full_arm_nll"] is None
    assert half["kl_first_position"] is None and half["agrees"] is False

    agg = R.aggregate([good, missing, half])[(R.RUNG4, 0.25)]
    assert agg["rows"] == 3 and agg["valid_rows"] == 1 and agg["invalid_rows"] == 2
    assert agg["invalid_reasons"] == {f"{R.INVALID_NO_LOGPROBS}:full_arm": 1,
                                      f"{R.INVALID_NO_LOGPROBS}:candidate_arm": 1}
    # Excluded, not averaged in: the mean is the good row's 0.75, not 0.375 or 0.
    assert agg["mean_delta_ce_nats"] == pytest.approx(DELTA_HAND)
    assert agg["mean_kl_first_position"] == pytest.approx(KL_HAND)
    assert agg["mean_tokens_retained"] == pytest.approx(good["tokens_retained"], abs=0.1)

    # An all-invalid cell has NO mean. It prints `n/a`, never 0.
    only_bad = R.aggregate([missing])
    assert only_bad[(R.RUNG4, 0.25)]["valid_rows"] == 0
    assert only_bad[(R.RUNG4, 0.25)]["invalid_rows"] == 1
    assert only_bad[(R.RUNG4, 0.25)]["mean_delta_ce_nats"] is None
    assert only_bad[(R.RUNG4, 0.25)]["mean_kl_first_position"] is None
    assert only_bad[(R.RUNG4, 0.25)]["agreement_rate"] is None
    report = R.render_report(only_bad, floor=0.9)
    assert "n/a" in report
    # A reader of the report alone learns which route anchored the numbers and what
    # unit the retained column is in — both live in the rows, and a footnote in the
    # JSONL is where a qualifier goes unread.
    assert "span routes" not in report          # no valid row, so no route to name
    assert "retained_unit" in report and "characters" in report
    # `StubEngine`'s readout carries a logprob per position but no token TEXT, which
    # is the unlabelled shape — so the honest label on this cell is the estimate, and
    # the report says so instead of leaving the number unqualified.
    good_report = R.render_report(R.aggregate([good, half, missing]), floor=0.9)
    assert f"span routes over valid rows: {R.SPAN_CHAR_ESTIMATE}=1" in good_report


def test_zero_scored_positions_invalidates_rather_than_reading_as_agreement(tmp_path):
    """A readout of the right length whose every position is null scores nothing.

    This is the silent empty parse the clause is about: the answer HAS a
    `prompt_logprobs` list, so the `no_prompt_logprobs` guard waves it through and
    only the position-by-position read knows no token was ever scored. Reading that
    as `delta_ce_nats = 0.0` would post perfect preservation for a non-answer.
    """
    row, _big = _harvested(tmp_path)
    masked = R.score_row(row, _stub(row, full_nll=FULL_NLL, cand_nll=CAND_NLL,
                                    all_masked=True),
                         compressor=R.EXTRACTIVE, budget=0.10)
    assert masked["valid"] is False
    assert masked["invalid_reason"] == f"{R.INVALID_ZERO_SCORED}:full_arm"
    assert masked["delta_ce_nats"] is None and masked["agrees"] is False

    short = R.score_row(row, _stub(row, full_nll=FULL_NLL, cand_nll=CAND_NLL,
                                   positions=3),
                        compressor=R.EXTRACTIVE, budget=0.10)
    assert short["valid"] is False
    assert short["invalid_reason"] == f"{R.INVALID_SPAN_SHORT}:full_arm"
    assert short["delta_ce_nats"] is None

    agg = R.aggregate([masked, short])[(R.EXTRACTIVE, 0.10)]
    assert (agg["valid_rows"], agg["invalid_rows"]) == (0, 2)
    assert agg["mean_delta_ce_nats"] is None and agg["mean_kl_first_position"] is None
    # A masked FIRST position is not a zero KL either: no distribution, no number.
    assert R.kl_first_position({}, {"A": -0.5}) is None


# ---------------------------------------------------------------------------
# Clause 3 — the harvester joins the sidecar, and scoring needs only the YAML
# ---------------------------------------------------------------------------

def test_harvester_joins_the_sidecar_verbatim_and_scoring_survives_its_removal(tmp_path):
    sessions, big = _make_sessions(tmp_path)
    rows = R.harvest(sessions).rows

    # The size band is enforced: the under-band session is not harvested.
    assert len(rows) == 1
    row = rows[0]
    sidecar = sessions / f"{SID}.tool-results" / f"{CID}.txt"
    assert sidecar.read_text(encoding="utf-8") == big
    # Verbatim from the sidecar, NOT the record's pointer.
    assert row.observation == big
    assert row.observation != POINTER
    assert row.observation_bytes == len(big.encode("utf-8")) >= R.MIN_OBSERVATION_BYTES
    # Tool name and next action come from the session record, not the sidecar.
    assert row.tool_name == "Bash" and row.tool_call_id == CID and row.session_id == SID
    assert row.next_action["kind"] == "tool_call"
    assert row.next_action["calls"][0]["function"]["name"] == "Grep"
    # The prefix is the history up to, but excluding, this observation.
    assert [m["role"] for m in row.prefix] == ["system", "user", "assistant"]
    assert all("timestamp" not in m for m in row.prefix)
    assert row.messages[-1] == {"role": "tool", "tool_call_id": CID, "content": big}

    corpus = tmp_path / "corpus.yaml"
    R.save_corpus(rows, corpus)
    text = corpus.read_text(encoding="utf-8")
    assert text.startswith("version:") and "row_id" in text

    shutil.rmtree(sessions)
    assert not sessions.exists()

    # Scoring consumes the pinned file alone: nothing re-reads the sessions tree.
    pinned = R.load_corpus(corpus)
    assert len(pinned) == 1 and pinned[0].observation == big
    rec = R.score_row(pinned[0], _stub(pinned[0], full_nll=FULL_NLL, cand_nll=CAND_NLL,
                                       full_topk=TOPK_FULL, cand_topk=TOPK_CAND),
                      compressor=R.RUNG4, budget=0.25)
    assert rec["valid"] is True
    assert rec["delta_ce_nats"] == pytest.approx(DELTA_HAND)
    assert rec["kl_first_position"] == pytest.approx(KL_HAND)
    assert rec["tier"] == CANONICALLY_EQUAL and rec["agrees"] is True
    assert rec["observation_bytes"] == len(big.encode("utf-8"))


# ---------------------------------------------------------------------------
# Clause 4 — the baseline arm IS production's rung 4, byte for byte
# ---------------------------------------------------------------------------

def test_the_rung4_arm_is_byte_identical_to_production_own_output(tmp_path):
    from app.harness.loop import _truncate_largest_tool_results

    row, _big = _harvested(tmp_path)
    messages = row.messages
    target = len(row.observation) - int(round(len(row.observation) * 0.25))

    # Production's own call, on its own copy, read straight off its own output.
    direct = copy.deepcopy(messages)
    truncated, freed = _truncate_largest_tool_results(
        direct, target_chars=target, session_id="",
        min_chars=R.DEFAULT_MIN_CHARS, disallowed_tools=None, name_session_record=False)
    direct_bytes = direct[-1]["content"].encode("utf-8")
    assert truncated == 1 and freed > 0

    arm_bytes = R.rung4_replacement(
        messages, target_chars=target, tool_call_id=CID, session_id="").encode("utf-8")
    assert arm_bytes == direct_bytes
    # It replaced the observation rather than echoing it, and said what it did.
    assert arm_bytes != row.observation.encode("utf-8")
    assert b"truncated by context-overflow recovery" in arm_bytes
    assert len(arm_bytes) < len(row.observation.encode("utf-8"))
    # The arm did not mutate the caller's list: the row stays re-scorable.
    assert messages == row.messages

    # The same bytes through the arm the scorer actually runs, at the same target.
    via_compress, info = R.compress(row, 0.25, compressor=R.RUNG4)
    assert via_compress.encode("utf-8") == direct_bytes
    assert info["target_chars"] == target
    scored = R.score_row(row, _stub(row, full_nll=1.0, cand_nll=1.0),
                         compressor=R.RUNG4, budget=0.25)
    # The retained column is the same string in the same unit production writes
    # (characters — the notice carries an em dash, so its byte count is larger).
    assert scored["retained_unit"] == "chars"
    assert scored["tokens_retained"] == len(direct[-1]["content"])
    assert direct[-1]["content"].encode("utf-8") == direct_bytes


# ---------------------------------------------------------------------------
# Clause 5 — suppression is a decline, recorded beside the tier
# ---------------------------------------------------------------------------

def test_a_generated_step_with_no_call_is_a_decline_beside_the_tier(tmp_path):
    """A compressor that deletes the call must not be counted as agreeing.

    The full arm still reaches for the tool; the compressed arm answers in prose.
    That is suppression, and it is NOT `protocol_invalid` — which means a call is
    present and unreadable, a different finding with a different fix.
    """
    row, _big = _harvested(tmp_path)
    eng = _stub(row, full_nll=FULL_NLL, cand_nll=CAND_NLL,
                full_topk=TOPK_FULL, cand_topk=TOPK_CAND)
    eng.cand_step = "The spill directory grew because the run was long."

    rec = R.score_row(row, eng, compressor=R.EXTRACTIVE, budget=0.10)

    assert rec["valid"] is True
    assert rec["decline"] == R.DECLINE_COMPRESSED
    assert rec["decline"] not in TIERS
    assert rec["tier"] is None
    assert rec["agrees"] is False
    assert "suppression, not agreement" in rec["tier_why"]
    # The logprob readings still stand beside it — the row is measured, not void.
    assert rec["delta_ce_nats"] == pytest.approx(DELTA_HAND)
    assert rec["kl_first_position"] == pytest.approx(KL_HAND)

    agg = R.aggregate([rec])[(R.EXTRACTIVE, 0.10)]
    assert agg["declines"] == 1 and agg["tiers"] == {}
    # A suppressed call is the arm's own answer: it stays in the denominator, so a
    # compressor that deletes every call posts 0.0 rather than an empty cell.
    assert agg["undecidable"] == 0
    assert agg["decidable_rows"] == 1 and agg["agreement_rate"] == pytest.approx(0.0)

    # The other two declines are the reference arm's fault, and stay undecidable.
    eng.cand_step = R.render_action(row.next_action)
    eng.full_step = "I will not be calling anything."
    rev = R.score_row(row, eng, compressor=R.EXTRACTIVE, budget=0.10)
    assert rev["decline"] == R.DECLINE_FULL and rev["agrees"] is False
    assert rev["tier_why"] == "the compressed arm emitted a tool call; the full arm did not"
    assert R.aggregate([rev])[(R.EXTRACTIVE, 0.10)]["undecidable"] == 1
    # With neither arm calling, there is no reference decision to preserve at all.
    eng.cand_step = "Nothing to report."
    both = R.score_row(row, eng, compressor=R.EXTRACTIVE, budget=0.10)
    assert both["decline"] == R.DECLINE_BOTH and both["tier"] is None
    assert both["tier_why"] == "the recorded step called; neither arm did"
    assert R.aggregate([both])[(R.EXTRACTIVE, 0.10)]["undecidable"] == 1
    assert len({R.DECLINE_COMPRESSED, R.DECLINE_FULL, R.DECLINE_BOTH}
               & set(TIERS)) == 0


def test_an_unreadable_call_is_protocol_invalid_and_not_the_decline_bucket(tmp_path):
    """The bucket clause 5 keeps apart, seen from the other side.

    Here the compressed arm DID call — with arguments that are not JSON — so this
    is `protocol_invalid` with `decline` empty. Same row shape as a decline,
    opposite finding: one compressor mangled its arguments, the other deleted the
    call, and a frontier that conflated them would recommend the wrong fix.
    """
    row, _big = _harvested(tmp_path)
    eng = _stub(row, full_nll=FULL_NLL, cand_nll=CAND_NLL,
                full_topk=TOPK_FULL, cand_topk=TOPK_CAND)
    eng.cand_step = R.TOOL_CALL_OPEN + "Grep\n" + "this is not json" + R.TOOL_CALL_CLOSE

    rec = R.score_row(row, eng, compressor=R.EXTRACTIVE, budget=0.10)
    assert rec["tier"] == PROTOCOL_INVALID
    assert rec["tier"] in TIERS
    assert rec["decline"] is None and rec["agrees"] is False
    assert rec["tier_why"] == "both arms emitted a call"
    agg = R.aggregate([rec])[(R.EXTRACTIVE, 0.10)]
    assert agg["protocol_invalid_rows"] == 1 and agg["declines"] == 0
    assert agg["decidable_rows"] == 1 and agg["agreement_rate"] == pytest.approx(0.0)

    # The remaining tiers come from `action_equivalence` unchanged, on this pair.
    eng.cand_step = R.TOOL_CALL_OPEN + "Read\n" + json.dumps(
        {"pattern": "truncate"}) + R.TOOL_CALL_CLOSE
    assert R.score_row(row, eng, compressor=R.EXTRACTIVE,
                       budget=0.10)["tier"] == DIFFERENT_TOOL
    eng.cand_step = R.TOOL_CALL_OPEN + "Grep\n" + json.dumps(
        {"pattern": "compress"}) + R.TOOL_CALL_CLOSE
    assert R.score_row(row, eng, compressor=R.EXTRACTIVE,
                       budget=0.10)["tier"] == ARGS_DIFFER
    eng.cand_step = R.render_action(row.next_action)
    assert R.score_row(row, eng, compressor=R.EXTRACTIVE,
                       budget=0.10)["tier"] == CANONICALLY_EQUAL


# ---------------------------------------------------------------------------
# #3967 — a real record stores assistant `content` as blocks, and every scanned
# tool row owes an account
#
# These nodes are the difference between a scorer that passes its fixtures and one
# that can be RUN. #3850's fixtures stored assistant content as a string, which is
# what this module's own projection writes; a real session record stores it as a
# list of blocks. On the live corpus that shape made `harvest` abort outright, and
# the rows that did survive pinned a Python LIST as the next action's content, which
# `render_action` then wrote into the scored arm as a repr.
# ---------------------------------------------------------------------------

#: The assistant prose one deciding turn carries, and the band both this section and
#: the module's own default use. Named here because the fixture above stores prose as
#: a bare string and this section needs the SAME string behind a block list.
PROSE_ASKS = "Let me grep the ladder for the truncation marker."
MIN = R.MIN_OBSERVATION_BYTES

#: One assistant turn's `content`, as the live store writes it: a list of blocks.
BLOCKS = [{"type": "text", "text": PROSE_ASKS}]


def _block_messages(tool_content: str, *, decide_with: str) -> list[dict]:
    """The fixture transcript with assistant `content` stored as a LIST OF BLOCKS.

    `decide_with` picks which arm the deciding row is: `tool_call` keeps its call,
    `answer` drops it. Either way BOTH assistant rows carry blocks, the way the live
    store writes them, so a fix that guards only the prose arm still fails here.
    """
    msgs = _session_messages(tool_content)
    msgs[2]["content"] = list(BLOCKS)
    last = dict(msgs[-1])
    last["content"] = list(BLOCKS)
    if decide_with == "answer":
        last.pop("tool_calls", None)
    msgs[-1] = last
    return msgs


def _block_tree(tmp_path: Path, name: str = "blocks",
                *, decide_with: str = "tool_call") -> tuple[Path, str]:
    """Write one block-form session and its spill beside it; return (dir, big)."""
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    big = _observation("BLOCK")
    (root / f"{SID}.tool-results").mkdir(exist_ok=True)
    (root / f"{SID}.tool-results" / f"{CID}.txt").write_text(big, encoding="utf-8")
    (root / f"{SID}.json").write_text(
        json.dumps({"messages": _block_messages(POINTER, decide_with=decide_with)}),
        encoding="utf-8")
    return root, big


def _block_row(tmp_path: Path, **kw) -> R.CorpusRow:
    """Harvest one block-form row through the documented `session_files=` route."""
    root, _ = _block_tree(tmp_path, **kw)
    got = R.harvest(session_files=[root / f"{SID}.json"], min_bytes=MIN)
    assert got.rows, "a block-form session must survive the harvest"
    return got.rows[0]


def test_next_action_returns_text_on_both_arms_when_content_is_a_block_list():
    """Clause 2: blocks are prose to be read, not an object to be `.strip()`ed.

    Two shapes, one per arm, both built from the same block list. The `answer` arm is
    the one that RAISED — `AttributeError: 'list' object has no attribute 'strip'` at
    `_next_action`'s prose test, which is a crash and takes a corpus-wide harvest down
    with it. The `tool_call` arm never raised: `m.get("content") or ""` handed back the
    list untouched, which is worse than crashing, because the row harvests cleanly and
    every number scored off it is built on a Python repr.
    """
    tool_row = {"role": "tool", "tool_call_id": CID, "content": POINTER}

    answer = R._next_action(
        [tool_row, {"role": "assistant", "content": list(BLOCKS)}], 0)
    assert answer is not None and answer["kind"] == "answer"
    assert answer["content"] == PROSE_ASKS, "the answer arm must pin flattened text"

    call = R._next_action(
        [tool_row, {"role": "assistant", "content": list(BLOCKS),
                    "tool_calls": [{"id": "chatcmpl-tool-x0002", "type": "function",
                                    "function": {"name": "Grep",
                                                 "arguments": "{}"}}]}], 0)
    assert call is not None and call["kind"] == "tool_call"
    assert call["content"] == PROSE_ASKS, "the tool-call arm pins text too"
    assert call["calls"][0]["id"] == "chatcmpl-tool-x0002"


def test_no_rendered_arm_or_chatml_of_a_block_form_row_carries_a_python_repr(tmp_path):
    """Clause 3: the scored text IS the prose the row carries, never `[{'type'`.

    `render_action`'s output is what gets teacher-forced after the assistant header, so
    a repr in it means the engine was scored on a string it is never served. The #3967
    triage reproduced the artefact on a synthetic row as
    `[{'type': 'text', 'text': 'x'}]<|tool_call|>Bash...`. `render_chatml` is checked on
    the same row because it ALREADY flattened blocks: a row whose action still carried a
    list while its prefix did not is the split brain this clause forbids.
    """
    for decide in ("tool_call", "answer"):
        # The raw row as #3850 pinned it — `content` still the block list — so this
        # half reddens on the repr itself, not on the new harvest interface.
        raw = R.render_action({"kind": "tool_call", "content": list(BLOCKS),
                               "calls": [{"id": "chatcmpl-tool-x0002",
                                          "type": "function",
                                          "function": {"name": "Grep",
                                                       "arguments": "{}"}}]})
        assert "[{'type" not in raw, f"a block still reaches the arm as a repr: {raw!r}"
        assert raw.startswith(PROSE_ASKS)

        row = _block_row(tmp_path, name=f"blk-{decide}", decide_with=decide)
        arm = R.render_action(row.next_action)
        assert "[{'type" not in arm and "type':" not in arm, (
            f"the {decide} arm carries a block repr: {arm[:80]!r}")
        assert arm.startswith(PROSE_ASKS), "the arm opens with the row's own prose"

        chatml = R.render_chatml([*row.prefix,
                                  {"role": "assistant", "content": arm}])
        assert "[{'type" not in chatml
        assert chatml.count(PROSE_ASKS) == 2, (
            "the prose appears once in the prefix row and once in the arm")


def test_a_block_form_session_harvests_end_to_end_through_session_files(tmp_path):
    """Clause 4: the documented `harvest(session_files=..., min_bytes=...)` yields a row.

    Two independent defects stood between this exact call and a row, and the call shape
    catches each. `_next_action` aborted on the block list. And `harvest` resolved the
    spill directory against its `sessions_dir` argument, which is None on this call, so
    it looked beside the default store for a spill named `sessions.tool-results`, found
    none, and returned ZERO rows from a corpus that plainly had one. A silent zero is
    the worst outcome a harvester can have: the caller prints `harvested 0 rows` and
    goes on to conclude the corpus is thin.
    """
    root, big = _block_tree(tmp_path)
    got = R.harvest(session_files=[root / f"{SID}.json"], min_bytes=MIN)

    assert len(got.rows) == 1, "one qualifying tool row in, one row out"
    row = got.rows[0]
    assert row.observation == big, "the sidecar is still joined verbatim"
    assert row.tool_name == "Bash"
    assert row.next_action["kind"] == "tool_call"
    assert isinstance(row.next_action["content"], str), (
        "the pinned action's content is text, not the block list the record stored")
    assert R.render_action(row.next_action).startswith(PROSE_ASKS)
    assert got.scanned == 1 and got.skips.total == 0


def test_harvest_accounts_every_scanned_row_in_the_four_named_skips(tmp_path):
    """Clause 5: `scanned == rows + the four counts`, each bucket tripped exactly once.

    Five tool rows, four failure shapes and one row that passes, so the identity is
    checked against a non-zero value in every bucket rather than a tally of zeroes.
    Band is charged before the two record-quality checks on purpose: a row that is both
    under-band and nameless is a band casualty, and charging it to `no_tool_name` would
    report a record-quality problem that the band alone explains.
    """
    root = tmp_path / "skips"
    root.mkdir()
    spill = root / f"{SID}.tool-results"
    spill.mkdir()
    big, small = _observation("BIG"), _observation("SMALL", lines=6)
    (spill / "cid-under.txt").write_text(small, encoding="utf-8")
    (spill / "cid-unnamed.txt").write_text(big, encoding="utf-8")
    (spill / "cid-nodecision.txt").write_text(big, encoding="utf-8")
    (spill / f"{CID}.txt").write_text(big, encoding="utf-8")

    def call(call_id: str, name: str = "Bash") -> dict:
        return {"id": call_id, "type": "function",
                "function": {"name": name, "arguments": "{}"}}

    msgs = [
        {"role": "user", "content": "Why did the spill grow?"},
        # (a) recorded and named, but its observation was never spilled
        {"role": "assistant", "content": "", "tool_calls": [call("cid-missing")]},
        {"role": "tool", "tool_call_id": "cid-missing", "content": POINTER},
        # (b) spilled, but under the band
        {"role": "assistant", "content": "", "tool_calls": [call("cid-under")]},
        {"role": "tool", "tool_call_id": "cid-under", "content": POINTER},
        # (c) at band, and no preceding assistant row carries this id
        {"role": "tool", "tool_call_id": "cid-unnamed", "content": POINTER},
        # (d) at band and named, but the step ends on a user row: no decision
        {"role": "assistant", "content": "", "tool_calls": [call("cid-nodecision")]},
        {"role": "tool", "tool_call_id": "cid-nodecision", "content": POINTER},
        {"role": "user", "content": "and the ladder?"},
        # (e) the one row with all four requirements
        {"role": "assistant", "content": "", "tool_calls": [call(CID)]},
        {"role": "tool", "tool_call_id": CID, "content": POINTER},
        {"role": "assistant", "content": "", "tool_calls": [call("cid-next", "Grep")]},
    ]
    (root / f"{SID}.json").write_text(json.dumps({"messages": msgs}), encoding="utf-8")

    got = R.harvest(root, min_bytes=MIN)
    assert got.scanned == 5, "five tool rows were in the transcript"
    assert len(got.rows) == 1 and got.rows[0].tool_call_id == CID
    assert got.skips.as_dict() == {"no_sidecar": 1, "below_min_bytes": 1,
                                   "no_tool_name": 1, "no_next_action": 1}
    assert got.scanned == len(got.rows) + got.skips.total
    assert tuple(R.HARVEST_SKIP_REASONS) == tuple(got.skips.as_dict()), (
        "the published reason names and the counts must be the same four")


def test_the_harvest_cli_prints_the_row_count_and_all_four_skip_counts(tmp_path,
                                                                      capsys):
    """Clause 5's other half: the four counts reach the terminal, not just Python.

    The owed check is a command line over the real store, so an account the caller
    cannot print is an account nobody reads. The line is emitted as an identity
    (`scanned N = rows + a + b + c + d`), which is why a broken bucket prints
    `SKIP TALLY BROKEN` instead of lying quietly.
    """
    sessions, _ = _make_sessions(tmp_path)
    out = tmp_path / "corpus.yaml"
    assert R.main(["harvest", "--sessions-dir", str(sessions),
                   "--min-bytes", str(MIN), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"harvested 1 rows (min {MIN} bytes) -> {out}" in text
    assert "scanned 2 tool rows = 1 rows + " in text
    for reason in R.HARVEST_SKIP_REASONS:
        assert reason in text, f"the CLI must name the skip reason {reason!r}"
    assert "below_min_bytes 1" in text, "the under-band session is charged to the band"
    assert "no_sidecar 0" in text, "a zero is printed, not omitted"
    assert "SKIP TALLY BROKEN" not in text
    assert "SHORT OF THE ITEM'S BAR" in text, "1 row is under 40 and must say so"
    assert len(R.load_corpus(out)) == 1, "the YAML that did get written still loads"


def test_the_documented_corpus_output_path_is_committable():
    """The usage block must name an `--out` path git will actually accept.

    #3967 triage found the module documenting `eval/baselines/observation-compression-
    corpus.yaml`, which root `.gitignore:141` swallows under `eval/baselines/*` — so the
    frontier run's corpus, the one artefact that would make its numbers re-checkable, had
    no committable home and the round was told to ask a human for a negation. The fix is
    one filename: name a path that is not ignored. Checked against the real .gitignore
    rather than a copy of it, because the claim is about the repository.
    """
    doc = R.__doc__ or ""
    paths = {raw.rstrip("\\") for raw in re.findall(r"--out (\S+)", doc)}
    assert paths, "the usage block must show where the harvested corpus goes"
    for path in sorted(paths):
        assert not path.startswith("eval/baselines/"), f"{path} is ignored"
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", str(ROOT / path)],
            cwd=ROOT, capture_output=True).returncode == 0
        assert not ignored, (
            f"the documented corpus path {path} matches a .gitignore rule, so the "
            "frontier run's corpus could never be committed")
