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
import inspect
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


# ---------------------------------------------------------------------------
# #4499 — the teacher-forced readout asks for the OBSERVED TOKEN, not for a top-k
# ---------------------------------------------------------------------------
#
# The premise these nodes pin: vLLM's `prompt_logprobs` map is the top-K candidates
# PLUS the observed token appended at its true rank when that rank exceeds K, so at
# K >= 1 the entry that says `rank: 1` is the argmax of the returned candidates and
# the token the caller actually sent is somewhere in that map at a rank nothing in
# the entries discloses. `prompt_logprobs: 0` is the shape that fixes it: one entry
# per scored position, and that entry is the prompt's own token, so identity is
# settled by cardinality. The committed capture below is the proof, from this engine,
# on a prompt the caller typed — no node here asks the engine anything.

#: Both arms of one row, teacher-forced at `prompt_logprobs: 0` and captured off the
#: live engine. `tests/fixtures/.gitignore` carries the negation that lets it into
#: the tree, and the last node here re-checks that rule every run, because a hidden
#: fixture is a silently un-run module rather than a red suite (#2282).
K0_CAPTURE = ROOT / "tests" / "fixtures" / "vllm_prompt_logprobs_k0_2026-11-22.json"


def _k0_capture() -> tuple[dict, "R.CorpusRow"]:
    """The committed capture, plus the corpus row its two prompts were built from."""
    fx = json.loads(K0_CAPTURE.read_text(encoding="utf-8"))
    row_json = fx["row"]
    row = R.CorpusRow(
        row_id=row_json["row_id"], session_id=row_json["session_id"],
        tool_call_id=row_json["tool_call_id"], tool_name=row_json["tool_name"],
        observation=row_json["observation"],
        observation_bytes=row_json["observation_bytes"],
        prefix=tuple(row_json["prefix"]), next_action=row_json["next_action"])
    return fx, row


def _sole(pos: dict) -> dict:
    """The one entry of a one-entry position. Test-side twin of `R._sole_entry`."""
    assert len(pos) == 1, f"{len(pos)} entries: not the shape this capture claims"
    return next(iter(pos.values()))


class K0CaptureClient(R.LogprobClient):
    """A `LogprobClient` whose transport is the committed capture, so no socket opens.

    The request half is NOT replaced: `_body` is the real one and the client is built
    with the model name the capture records, so every teacher-forced body `score_row`
    produces has to equal the `request_body` stored beside that arm's readouts — every
    field, byte for byte. That is what makes the capture a witness of THIS module's
    route rather than of some K=0 request someone typed: the fixture's readouts are
    the engine's answer to exactly these bodies (see its `provenance`), so a change to
    the body — a width, a `temperature`, a `chat_template_kwargs` — fails here against
    the stored bytes instead of quietly scoring a request nobody sends. `_post` is the
    only thing stubbed. An unknown prompt fails loudly too, because arms that no longer
    match the captured ones would make every number below meaningless.

    `swap_arms` answers the two captured prompts with each other's readouts. It exists
    for one purpose: to produce a second VALID record for the same row whose ΔCE is the
    exact negation of the real one, so an aggregate over the pair has a mean the test
    can predict from arithmetic rather than from the record it is checking.
    """

    route_label = "stub:k0_capture"

    def __init__(self, fx: dict, step_text: str, *, swap_arms: bool = False) -> None:
        bodies = [arm["request_body"] for arm in fx["arms"]]
        prompts = [arm["prompt_sent"] for arm in fx["arms"]]
        assert [b["prompt"] for b in bodies] == prompts, (
            "each arm's stored body must carry the prompt that arm was scored on")
        super().__init__(base_url="http://stub.invalid/v1", model=bodies[0]["model"])
        self.bodies: list[dict] = []
        self.step_text = step_text
        readouts = [arm["readout"] for arm in fx["arms"]]
        if swap_arms:
            readouts = readouts[::-1]
        self.expected = {b["prompt"]: b for b in bodies}
        self.readouts = dict(zip(prompts, readouts))

    def _post(self, body: dict) -> dict:
        self.bodies.append(dict(body))
        if "prompt_logprobs" not in body:
            return {"choices": [{"text": self.step_text}]}
        if body["prompt"] not in self.expected:
            raise AssertionError(
                "score_row built a prompt that is not in the committed K=0 capture, so "
                "the arms it is scoring are not the arms that were captured")
        assert body == self.expected[body["prompt"]], (
            f"posted body is not the one the capture was taken with: "
            f"{ {k: v for k, v in body.items() if k != 'prompt'} } against "
            f"{ {k: v for k, v in self.expected[body['prompt']].items() if k != 'prompt'} }")
        # `prompt_token_ids` rides along as None because that is what this route
        # serves on `choices[0]`: the id list is not available to shortcut identity,
        # which is why the K=0 shape is the fix rather than a lookup.
        return {"choices": [{"text": "", "prompt_logprobs": self.readouts[body["prompt"]],
                             "prompt_token_ids": None}]}


def _score_captured_row(*, swap_arms: bool = False) -> tuple[
        dict, "R.CorpusRow", K0CaptureClient, dict]:
    """Score the captured row through the capture. One call per node, no engine.

    `swap_arms` keeps the row, the prompts and the request bodies exactly as they are
    and hands each arm the OTHER arm's readout, so the returned record is a second
    VALID row for the same (compressor, budget) whose ΔCE is the negation of the real
    one — which is what lets the aggregate node predict its mean without quoting a
    record back at itself.
    """
    fx, row = _k0_capture()
    eng = K0CaptureClient(fx, R.render_action(row.next_action), swap_arms=swap_arms)
    rec = R.score_row(row, eng, compressor=fx["row"]["compressor"],
                      budget=fx["row"]["budget"])
    return fx, row, eng, rec


def test_the_teacher_forced_request_asks_for_the_observed_token_only():
    """Both arms POST `prompt_logprobs: 0`, not a candidate width.

    #4499's clause 1. The defect was named in the code itself: `teacher_forced`
    defaulted `top_k: int = TOP_K` at 8, so every arm asked for a top-k map, and a
    top-k map is where rank 1 stops being the served token. This node drives
    `score_row` through the real `_body` and reads the JSON back off the transport,
    so what it pins is the bytes on the wire for BOTH arms rather than a constant in
    a signature. `max_tokens: 1` and the step requests are asserted beside it: the
    fix is that the readout asks for less, not that it asks differently everywhere.
    """
    fx, _row, eng, rec = _score_captured_row()
    assert rec["valid"] is True, rec["invalid_reason"]
    teacher_forced = [b for b in eng.bodies if "prompt_logprobs" in b]
    assert len(teacher_forced) == 2, (
        f"one teacher-forced readout per arm; bodies were {len(eng.bodies)}")
    assert [b["prompt_logprobs"] for b in teacher_forced] == [0, 0]
    assert [b["logprobs"] for b in teacher_forced] == [0, 0], (
        "the sampled-arm width must not ride along on a teacher-forced request")
    assert [b["max_tokens"] for b in teacher_forced] == [1, 1]
    assert teacher_forced[0]["prompt"] != teacher_forced[1]["prompt"], (
        "the two bodies must be the two different arms, not one prompt twice")
    steps = [b for b in eng.bodies if "prompt_logprobs" not in b]
    assert len(steps) == 2 and all("logprobs" not in b for b in steps), (
        "the step requests are untouched by this change and must not start asking "
        "for logprobs it has no reader for")
    assert R.OBSERVED_TOKENS_ONLY == 0
    for fn in (R.LogprobClient.teacher_forced, R.score_row):
        default = inspect.signature(fn).parameters["top_k"].default
        assert default == R.OBSERVED_TOKENS_ONLY, (
            f"{fn.__qualname__} still defaults to a K >= 1 readout")
    # The capture's own provenance: the bodies the stub matched above are the ones the
    # fixture says were POSTed to get these readouts, so this node's claim about the
    # wire is a claim about the bytes that produced the numbers every other node here
    # checks — not about a request shape the test author assumed.
    assert fx["provenance"]["request_bodies"] == [
        arm["request_body"] for arm in fx["arms"]], (
        "the capture's provenance must carry the same bodies its arms were scored with")
    assert [b["prompt_logprobs"] for b in fx["provenance"]["request_bodies"]] == [0, 0]


def test_a_one_entry_map_scores_and_a_wider_map_is_refused():
    """The observed token is identified by cardinality, never by rank (#4499, cl. 2).

    The item's own counterexample is the canonical shape here: the observed token
    `' capital'` arrives as the only entry of its position at rank 1434, and it is
    nevertheless the token that was sent — so its negated logprob is the position's
    NLL. A position with more than one entry is a `prompt_logprobs: K >= 1` readout,
    where the observed token is one candidate among several at a rank nothing
    discloses; `_position_nll` returns None for it, because reading rank 1 there
    reports the argmax candidate's loss as though it were the recorded action's. The
    wide position below is a clone of a real one from the capture with a rank-1
    candidate added beside the observed entry, which is exactly the bytes such a
    readout would carry.
    """
    assert R._position_nll({"6511": {"logprob": -9.32, "rank": 1434,
                                     "decoded_token": " capital"}}) == pytest.approx(9.32)
    assert R._position_nll(None) is None
    assert R._position_nll({}) is None
    assert R._position_nll({"token": " capital"}) is None
    # The OpenAI/chat shapes keep scoring exactly as before, raw `nll` included.
    assert R._position_nll({"token_logprob": -1.25}) == pytest.approx(1.25)
    assert R._position_nll({"logprob": -0.5}) == pytest.approx(0.5)
    assert R._position_nll({"nll": 3.5}) == pytest.approx(3.5)

    fx, _row = _k0_capture()
    for arm in fx["arms"]:
        assert R._position_nll(arm["readout"][0]) is None, (
            "position 0 is the masked ChatML opener: null means unscored")
        for pos in arm["readout"]:
            if pos is not None:
                assert R._position_nll(pos) == pytest.approx(-_sole(pos)["logprob"])

    scored = [p for p in fx["arms"][0]["readout"] if p]
    widest = max(scored, key=lambda p: _sole(p)["rank"])
    observed = _sole(widest)
    assert observed["rank"] > 5, "the capture must carry a deep-rank position"
    wide = dict(widest)
    wide["1"] = {"logprob": -0.01, "rank": 1, "decoded_token": " "}
    assert R._position_nll(wide) is None, (
        "a multi-entry map must be refused; rank 1 of it is a candidate, not the "
        "token the caller sent")
    assert R._token_surface(wide) is None
    assert R._position_topk(wide) == {}, (
        "a token-id map is not a top-k distribution either way: no branch here reads it")


def test_the_captured_surfaces_concatenate_to_the_prompt_they_scored():
    """`token_text` alignment becomes honest, because the surfaces ARE the prompt (#cl. 3).

    The capture stores both the bytes that were POSTED (`prompt_sent`) and the region
    the readout labels (`prompt_labelled`). They differ by exactly the leading ChatML
    opener: the engine masks position 0 and never decodes that token, so no readout
    can name it, and this is the only gap. Walking `_token_surface` over the scored
    positions in position order reproduces the labelled prompt character for character
    — 4,172 and 1,346 characters on the two arms. Under the rank-1 semantics this
    change removes, the same walk would have inserted `' '`-shaped argmax
    candidates at positions whose observed token sat at rank 235,315 and 72,159.
    """
    fx, _row = _k0_capture()
    for arm in fx["arms"]:
        join = "".join(R._token_surface(p) for p in arm["readout"] if p)
        assert join == arm["prompt_labelled"], (
            f"{arm['label']}: surfaces differ from the labelled prompt by "
            f"{len(join) - len(arm['prompt_labelled'])} chars")
        assert arm["prompt_sent"] == arm["unlabelled_prefix"] + arm["prompt_labelled"]
        assert arm["unlabelled_prefix"] == "<|" + "im_start" + "|>"
        assert fx["action"] in arm["prompt_labelled"], (
            "the action has to be inside the labelled region for token_text to mean "
            "anything on this arm")
    deepest = max((p for p in fx["arms"][0]["readout"] if p),
                  key=lambda p: _sole(p)["rank"])
    assert _sole(deepest)["rank"] > 1, (
        "at this position the observed token is not anybody's first choice, so the "
        "sole entry is the only reading that can be the prompt's")
    assert R._token_surface(deepest) == _sole(deepest)["decoded_token"]


def test_score_row_over_the_capture_scores_the_action_from_its_own_logprobs():
    """ΔCE comes out non-null, `token_text`-aligned, and equals the capture's arithmetic.

    #4499's clause 4. The expected value is computed HERE from the capture's own
    sole-entry logprobs over the action span — no scripted shift constant, which is
    the hole #4395's clause 4 left open ("it validates the arithmetic, not the token
    identity"). The span is located in this node from the surfaces and the action
    string, independently of `_align_action_span`, so an agreement between the two is
    evidence rather than a tautology — and the two definitions are asserted to COINCIDE,
    position by position: this node's window opens at the action's first token and runs
    to the end of the readout, while `_align_action_span` walks backward from the end
    until the action's character count is covered, so `start == first + 1` (the `+1` is
    the single masked position 0, which is dropped from this node's surface list) plus
    the per-arm mean equality plus `action_positions_scored == len(window)` is what
    proves those two definitions are the same window and not two that happen to agree
    here.

    No direction is asserted for the delta, and none may be. #4499's owed check records
    that per-request logprob magnitudes on this engine drift — one of six single
    requests at the same K differed by 0.6 nats on one position — so a fixture that
    asserted a sign would be asserting a measurement, and a frontier number is only
    ever allowed to come from `>=3` same-arm repeats, never from bytes frozen here.
    What is asserted is that the two arms scored DIFFERENT readouts whose means are
    therefore different, so a delta of zero would be a bug rather than a result.
    """
    fx, _row, _eng, rec = _score_captured_row()
    action = fx["action"]
    assert rec["valid"] is True, rec["invalid_reason"]
    assert rec["span_alignment"] == R.SPAN_TOKEN_TEXT
    assert rec["action_kind"] == "tool_call"

    expected_mean: dict[str, float] = {}
    for arm in fx["arms"]:
        start, route, reason = R._align_action_span(arm["readout"], action)
        assert reason is None and route == R.SPAN_TOKEN_TEXT, (
            f"{arm['label']}: the capture's own surfaces must locate the action")
        masked = [i for i, p in enumerate(arm["readout"]) if p is None]
        assert masked == [0], (
            f"{arm['label']}: this node's index offset below holds only for a readout "
            f"whose sole masked position is the leading one; got {masked}")
        surfaces, nlls = [], []
        for pos in arm["readout"]:
            if pos is None:
                continue
            surfaces.append(_sole(pos)["decoded_token"])
            nlls.append(-_sole(pos)["logprob"])
        start_char = arm["prompt_labelled"].index(action)
        offset, first = 0, None
        for i, surface in enumerate(surfaces):
            if offset + len(surface) > start_char:
                first = i
                break
            offset += len(surface)
        assert first is not None and surfaces[first] == action[0], (
            "the action is meant to begin exactly on a token boundary in this capture")
        assert start == first + 1, (
            f"{arm['label']}: `_align_action_span` opened the window at readout index "
            f"{start}, this node's forward walk opened it at {first} of the scored "
            f"positions (readout index {first + 1}) — two different spans")
        window = nlls[first:]
        expected_mean[arm["label"]] = sum(window) / len(window)
        assert rec["action_positions_scored"] == len(window)

    assert expected_mean["full"] != expected_mean["candidate"], (
        "the two arms must have scored different bytes, or a zero delta would be the "
        "stub answering both prompts with one readout")
    assert rec["full_arm_nll"] is not None
    assert rec["candidate_arm_nll"] is not None
    assert rec["full_arm_nll"] == pytest.approx(expected_mean["full"])
    assert rec["candidate_arm_nll"] == pytest.approx(expected_mean["candidate"])
    assert rec["delta_ce_nats"] is not None
    assert rec["delta_ce_nats"] == pytest.approx(
        expected_mean["candidate"] - expected_mean["full"])
    assert rec["delta_ce_nats"] != 0.0, (
        "same bytes on both arms: the delta would be measuring nothing")


def test_the_k0_route_reports_no_kl_and_records_its_readout_shape():
    """No candidate distribution is borrowed: KL is None and `top_k` says why (cl. 5).

    A `prompt_logprobs: 0` readout carries the observed token and nothing else, so
    `first_topk` is empty on both arms and `kl_first_position` has no distribution to
    diverge. Reporting None is the honest outcome; the alternative this refuses is a
    KL lifted from a second, K >= 1 request, which is a different quantity from a
    different readout sitting in a column about this one. Whether to buy that KL back
    at twice the requests is ruled on in #4499's owed checks. The record's existing
    `top_k` field carries the shape — 0 — so every published ΔCE names the readout
    that produced it, and a K=0 mean is never folded into a K>=1 mean.
    """
    fx, _row, _eng, rec = _score_captured_row()
    assert rec["valid"] is True, rec["invalid_reason"]
    assert rec["kl_first_position"] is None
    assert rec["top_k"] == 0
    for arm in fx["arms"]:
        for pos in arm["readout"]:
            if pos is not None:
                assert R._position_topk(pos) == {}, (
                    "a one-entry observed map is not a candidate distribution; if it "
                    "ever read as one, this None would become a fabricated KL")
    assert R.kl_first_position({}, {}) is None

    # The aggregate half, over THREE valid rows for one (compressor, budget) cell: the
    # real record, the same row again, and the arm-swapped one whose ΔCE is the real
    # one's negation (each arm scored the other arm's readout, so its candidate mean is
    # this record's full mean and the reverse — arithmetic, not a number quoted back
    # from a record). `aggregate` rounds to 6 decimals, so a cell's mean is compared
    # against a value predicted that way, never against a record's own field.
    swapped = _score_captured_row(swap_arms=True)[3]
    assert swapped["valid"] is True, swapped["invalid_reason"]
    assert swapped["kl_first_position"] is None
    assert swapped["delta_ce_nats"] == pytest.approx(-rec["delta_ce_nats"])

    pair = R.aggregate([rec, swapped])
    assert len(pair) == 1
    cell = next(iter(pair.values()))
    assert (cell["rows"], cell["valid_rows"], cell["invalid_rows"]) == (2, 2, 0)
    assert cell["mean_delta_ce_nats"] == pytest.approx(0.0, abs=1e-6), (
        "one row and its arm-swapped twin average to zero by arithmetic; anything else "
        "means the cell is not summing the rows it was given")
    assert cell["mean_kl_first_position"] is None, (
        "an aggregate must not invent a KL for a route that cannot measure one, and two "
        "valid rows is not a reason to")
    assert cell["span_alignments"] == {R.SPAN_TOKEN_TEXT: 2}, cell["span_alignments"]

    # And the mean is not hard-wired to zero: with the real row counted twice the cell
    # must move off zero, which is what proves the pair above was 0.0 because of the
    # rows fed to it rather than because nothing was summed.
    triple = next(iter(R.aggregate([rec, rec, swapped]).values()))
    assert (triple["rows"], triple["valid_rows"]) == (3, 3)
    assert triple["mean_delta_ce_nats"] != 0.0
    assert triple["mean_kl_first_position"] is None
    assert rec["top_k"] == swapped["top_k"] == 0, (
        "every row folded into that cell names the K=0 shape")


def test_the_k0_capture_fixture_is_tracked_not_ignored():
    """The root `*.json` rule must not hide the capture this module scores against.

    #2282's trap, in its loud form: `tests/fixtures/.gitignore` carries the negation,
    so the bytes are committable — and `git check-ignore --no-index` answers exit 0
    for a NEGATION too, which is why this node reads the RULE it was printed rather
    than the exit status. Three things are checked: the last matching pattern for the
    capture is a negation from `tests/fixtures/.gitignore`; a sibling name outside the
    family is still excluded by the root `*.json`, so the negation is a family and not
    a blanket lift; and the file is actually tracked, so a fresh checkout has it. That
    last one is what makes the guard real — without the bytes, the nodes above fail
    here, and a fixture that was never committed would otherwise leave this module's
    reader coverage silently unrun while the suite stayed green.
    """
    rel = str(K0_CAPTURE.relative_to(ROOT))
    probe = subprocess.run(["git", "check-ignore", "--no-index", "-v", rel],
                           cwd=ROOT, capture_output=True, text=True)
    assert probe.returncode == 0, f"no ignore rule matched {rel}: {probe.stderr.strip()}"
    rule = probe.stdout.split("\t")[0]
    source, pattern = rule.rsplit(":", 1)
    assert pattern.startswith("!"), f"{rel} is excluded by {rule}, not negated"
    assert "tests/fixtures/.gitignore" in source, rule

    control = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v",
         "tests/fixtures/engine_answer_not_a_capture.json"],
        cwd=ROOT, capture_output=True, text=True)
    assert control.returncode == 0, "an unnamed engine answer must still be ignored"
    assert control.stdout.split("\t")[0].rsplit(":", 1)[1] == "*.json", control.stdout

    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                             cwd=ROOT, capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{rel} is not in the tree ({tracked.stderr.strip()}), so the nodes that read "
        "it would be skipped-or-broken on a fresh checkout instead of pinning anything")


# ---------------------------------------------------------------------------
# #4508 — publishing the note: the `report` route from a run record to disk
# ---------------------------------------------------------------------------
#
# The six fields this section pins have been carried unpublished since #3967, and the
# reason is structural: `main()` shipped two subcommands (`harvest`, `score`) and the
# only place the frontier verdict, the span tally and the skip counts ever existed was
# `score`'s stdout, which no one commits. The nodes below drive the third subcommand
# over a run-record JSONL written the way `score --out` writes one, so the render path
# is pinned BEFORE the frontier run exists — the owed run then publishes a note rather
# than inventing one. Every number asserted here is the number the scripted stub was
# told to emit; none is quoted off the note it is checking.

#: The floor every node passes explicitly. `report` has no default floor: a silently
#: zero floor would publish "preserves the action" for a compressor whose agreement
#: collapsed to 0.0, so the argument is required and the note prints it.
NOTE_FLOOR = 1.0

#: A floor above any rate a 4-decimal aggregate can post, used to exhibit the case
#: where NO cell clears — the case the item says must print a sentence, not a blank.
NOTE_FLOOR_UNREACHABLE = 1.0001


def _note_stub(row: "R.CorpusRow", *, cand_nll: float = 1.5) -> SurfacedStubEngine:
    """Both arms emit the recorded call; the candidate arm scores `cand_nll`.

    `full_topk` and `cand_topk` are EMPTY maps, which is what a real `prompt_logprobs: 0`
    readout is — `test_the_k0_route_reports_no_kl_and_records_its_readout_shape` proves
    that shape leaves `_position_topk` `{}` — so the records this fixture writes carry
    `kl_first_position: null` and the note's readout-shape line is true of them.
    """
    step = R.render_action(row.next_action)
    return SurfacedStubEngine(action_text=step, observation=row.observation,
                              full_step=step, cand_step=step,
                              full_nll=1.0, cand_nll=cand_nll,
                              full_topk={}, cand_topk={})


def _note_corpus(tmp_path: Path) -> tuple[list["R.CorpusRow"], Path]:
    """Two pinned rows and the corpus file they were written to.

    The second row is the harvested one under its own `row_id`: the roster the note
    publishes is only pinned if it survives two rows, since a one-row roster passes
    even when the renderer prints a single hard-coded line.
    """
    root, _big = _make_sessions(tmp_path)
    base = R.harvest(root).rows[0]
    second = R.CorpusRow(**{**base.to_dict(), "row_id": base.row_id + "-b"})
    return [base, second], R.save_corpus([base, second], tmp_path / "corpus.yaml")


def _note_run(tmp_path: Path, *, cand_nll_by_compressor: dict | None = None,
              ) -> tuple[Path, Path]:
    """Score every (row, compressor, budget) and write the JSONL `score` would write.

    `cand_nll_by_compressor` prices each arm's candidate span differently: the stub
    scores the full arm at 1.0 nats on every route, so a cell's mean ΔCE is exactly
    `cand_nll - 1.0`, and two compressors priced identically could never disagree about
    dominance.
    """
    rows, corpus = _note_corpus(tmp_path)
    prices = cand_nll_by_compressor or {}
    summariser = lambda obs, budget_chars: obs[:budget_chars]  # noqa: E731
    recs = [
        R.score_row(row, _note_stub(row, cand_nll=prices.get(c, 1.5)),
                    compressor=c, budget=b, summariser=summariser)
        for row in rows for c in R.COMPRESSORS for b in R.BUDGETS
    ]
    assert all(r["valid"] for r in recs), [r["invalid_reason"] for r in recs if not r["valid"]]
    assert all(r["top_k"] == 0 and r["kl_first_position"] is None for r in recs), (
        "the fixture stopped looking like the shipped K=0 readout, so the note's "
        "readout-shape line would be asserting nothing about these bytes")
    run = tmp_path / "run.jsonl"
    with run.open("w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return run, corpus


def _note_skips_file(tmp_path: Path, **over) -> Path:
    """A `HarvestSkips.as_dict()` written to disk — the file `report --skips` reads."""
    counts = R.HarvestSkips(no_sidecar=0, below_min_bytes=12, no_tool_name=2,
                            no_next_action=3, **over).as_dict()
    path = tmp_path / "skips.json"
    path.write_text(json.dumps(counts), encoding="utf-8")
    return path


def _note_text(tmp_path: Path, run: Path, corpus: Path, **kw) -> str:
    out_dir = tmp_path / "measurements"
    path = R.write_note(run, corpus, out_dir=out_dir,
                        floor=kw.pop("floor", NOTE_FLOOR), **kw)
    return path.read_text(encoding="utf-8")


def test_the_report_subcommand_renders_the_dated_note_and_prints_its_path(tmp_path, capsys):
    """#4508 clause 1: a third subcommand, over a run record and a corpus, on disk.

    `report --run <the JSONL score wrote> --corpus <the corpus it was scored from>`
    writes `observation-compression-<date>.md` into the repo's own `eval/measurements/`
    and prints that path, because an unpublished note is the state this item exists to
    end. The date comes from `--date` so the run is re-checkable; the default directory
    is asserted against the REAL `.gitignore`, since a note that lands in an ignored
    path is the same loss as no renderer at all (the same trap #3967 found for the
    corpus at `eval/baselines/*`).
    """
    run, corpus = _note_run(tmp_path)
    rc = R.main(["report", "--run", str(run), "--corpus", str(corpus),
                 "--date", "2026-11-26", "--out-dir", str(tmp_path / "measurements"),
                 "--floor", str(NOTE_FLOOR)])
    assert rc == 0, "the CLI refused a run record it was given"
    printed = capsys.readouterr().out.strip().splitlines()
    assert printed[-1] == str(tmp_path / "measurements"
                              / "observation-compression-2026-11-26.md"), printed
    assert Path(printed[-1]).is_file(), "the printed path names no file"
    assert printed[-1].startswith(str(tmp_path)), "the node's own out-dir is what printed"

    default = R.NOTE_DIR
    assert default == ROOT / "eval" / "measurements", default
    assert default.is_dir(), "the repo has no eval/measurements to publish into"
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index",
         str(default / "observation-compression-2026-11-26.md")],
        cwd=ROOT, capture_output=True).returncode
    assert ignored != 0, (
        "a .gitignore rule swallows the published note, so the round could land the "
        "renderer and still never publish the measurement")


def test_the_note_header_names_every_row_id_with_its_corpus_path_and_size(tmp_path):
    """#4508 clause 2: a published row id is pinable or it is decoration.

    Both row ids in the run record appear, and each beside the corpus path and that
    corpus's byte size measured off the file — #4508's own triage had to record two
    corpora in `~/lloyd-data/eval-out/` at 11,646,966 and 1,245,716,690 bytes precisely
    because a row id with no corpus named cannot be re-checked. The size is the file's,
    not a sum of the rows' `observation_bytes`: the question is which file to go read.
    """
    rows, corpus = _note_corpus(tmp_path)
    run = _note_run(tmp_path)[0]
    note = _note_text(tmp_path, run, corpus,
                      skips=_note_skips_file(tmp_path), date="2026-11-26")
    for row in rows:
        assert row.row_id in note, f"{row.row_id} missing from the roster"
    assert str(corpus) in note, "the note names no corpus path"
    size = corpus.stat().st_size
    assert f"{size:,}" in note or f"{size} bytes" in note, (
        f"the corpus byte size {size} is not in the note")
    assert note.count("corpus") >= 2, "the roster collapsed to one mention"


def test_the_note_names_the_smallest_preserving_budget_or_says_none_beats_rung4(tmp_path):
    """#4508 clause 3: the frontier verdict, per compressor, or the explicit sentence.

    Three compressors x three budgets at a floor of 1.0. The stub scores every full arm
    at 1.0 nats, so the priced candidate arms give rung 4 a mean ΔCE of +0.5, `extractive`
    +0.2 and `summary` +1.5 — and all three post agreement 1.0, because both arms are
    scripted to emit the recorded call. That is the shape the note has to tell apart: two
    compressors that clear the floor, one of them better than production at matched
    budgets and one worse, so `beats rung4_truncate at matched budget` is True for
    `extractive` and False for `summary`. Both clear 1.0, so both get a smallest
    preserving budget, and it is the LOWEST budget on the axis (0.10), not whichever the
    dict happened to iterate first.
    """
    run, corpus = _note_run(tmp_path, cand_nll_by_compressor={
        R.RUNG4: 1.5, R.EXTRACTIVE: 1.2, R.SUMMARY: 2.5})
    note = _note_text(tmp_path, run, corpus,
                      skips=_note_skips_file(tmp_path), date="2026-11-26")
    for c in R.COMPRESSORS:
        assert f"{c}: smallest preserving budget: {R.BUDGETS[0]:.2f}" in note, (
            f"{c} cleared the floor at 0.10 and the note does not say so")
    assert f"{R.EXTRACTIVE}: beats {R.RUNG4} at matched budget: True" in note, note
    assert f"{R.SUMMARY}: beats {R.RUNG4} at matched budget: False" in note, note
    assert R.NO_PRESERVING_BUDGET_LINE not in note, (
        "extractive beats rung 4 here, so the nothing-beats-it sentence must not print")


def test_a_compressor_with_no_cell_over_the_floor_prints_the_matched_budget_sentence(
        tmp_path):
    """#4508 clause 3's other arm: no blank cell, a sentence.

    Two facts fail together here, and the note must say both. The floor is 1.0001, one
    ten-thousandth above the 1.000 every cell posts, so `frontier()` reports
    `clearing_budget: None` for all three compressors — a full-arm self-consistency
    spread above perfect self-agreement would read exactly like this. And the candidate
    arms are priced worse than production (+1.5 nats both, against rung 4's +0.5), so no
    compressor beats rung 4 at a matched budget either, which is the case the item says
    must be stated in the open rather than left as an empty table.
    """
    run, corpus = _note_run(tmp_path, cand_nll_by_compressor={
        R.RUNG4: 1.5, R.EXTRACTIVE: 2.5, R.SUMMARY: 2.5})
    note = _note_text(tmp_path, run, corpus, floor=NOTE_FLOOR_UNREACHABLE,
                      skips=_note_skips_file(tmp_path), date="2026-11-26")
    assert R.NO_PRESERVING_BUDGET_LINE in note, note
    for c in R.COMPRESSORS:
        assert f"{c}: smallest preserving budget" not in note, note
        assert f"{c}: beats {R.RUNG4} at matched budget: True" not in note, note
    assert f"{R.EXTRACTIVE}: {R.NO_PRESERVING_BUDGET_LINE}" not in note, (
        "the sentence is the note's one verdict, not a per-compressor filler")


def test_the_note_carries_the_span_route_tally_and_all_four_skip_counts(tmp_path):
    """#4508 clause 4: the two tallies, in the file, not only on a terminal.

    The span-route line is the SAME string `render_report` prints — both read one
    `aggregate()` — so the published note and the scored stdout cannot drift into two
    different accounts of which route sized the windows. The four harvest buckets are
    `HarvestSkips.as_dict()`'s keys in its order, each with its number: `no_sidecar 0`
    is printed because a zero is a result, exactly as the harvest CLI prints it.
    """
    run, corpus = _note_run(tmp_path)
    skips = _note_skips_file(tmp_path)
    note = _note_text(tmp_path, run, corpus, skips=skips, date="2026-11-26")
    recs = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines()]
    stdout_line = next(ln for ln in R.render_report(
        R.aggregate(recs), floor=NOTE_FLOOR).splitlines()
        if ln.startswith("span routes over valid rows: "))
    assert stdout_line in note, (
        f"the note's route tally is not stdout's: {stdout_line!r}")
    counts = json.loads(skips.read_text(encoding="utf-8"))
    for reason in R.HARVEST_SKIP_REASONS:
        assert f"{reason} {counts[reason]}" in note, f"{reason} missing from the note"
    assert R.SKIP_GAP_LABEL not in note, "a reported count rendered as a gap"


def test_a_skip_count_the_run_never_reported_prints_a_named_gap(tmp_path):
    """#4508 clause 4's other arm: an unmeasured bucket is a gap, not an omission.

    Two shapes of "not measured" reach a publisher: no skips file at all, and a skips
    object missing one key — `harvest-2026-11-16.log` is 0 bytes, so the first shape is
    the one this repo actually has. Either way all four names must still print, each
    missing one under the named gap label: a note that silently drops `no_sidecar` reads
    as a harvest that skipped nothing, which is the exact fiction the four buckets were
    added to prevent.
    """
    run, corpus = _note_run(tmp_path)
    no_file = _note_text(tmp_path, run, corpus, skips=None, date="2026-11-26")
    assert no_file.count(R.SKIP_GAP_LABEL) == 4, no_file
    for reason in R.HARVEST_SKIP_REASONS:
        assert f"{reason} {R.SKIP_GAP_LABEL}" in no_file, reason

    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"below_min_bytes": 12}), encoding="utf-8")
    note = _note_text(tmp_path, run, corpus, skips=partial, date="2026-11-26")
    assert "below_min_bytes 12" in note, note
    assert note.count(R.SKIP_GAP_LABEL) == 3, note


def test_every_published_delta_names_the_readout_shape_and_no_stable_logprob(tmp_path):
    """#4508 clause 5: no number arrives without the shape that produced it.

    Every note line carrying a ΔCE figure also names `prompt_logprobs: 0, top_k=0,
    kl_first_position null`, and the note states once that no per-request numeric logprob
    is stable on this route. Each rung-4 mean is rendered beside that budget's no-op
    count — rows whose compressed arm kept every byte, where the arm measured the
    observation it was handed — and beside its decline and undecidable counts. The no-op
    count is one, not zero, because one rung-4 record was rewritten to 100% retained:
    a count that always prints 0 would pass this node and fail the real run.
    """
    run, corpus = _note_run(tmp_path)
    recs = [json.loads(ln) for ln in run.read_text(encoding="utf-8").splitlines()]
    once = False
    for rec in recs:
        if not once and rec["compressor"] == R.RUNG4 and rec["budget"] == R.BUDGETS[0]:
            rec["tokens_retained_pct"] = 100.0   # exactly one untouched rung-4 row
            once = True
    assert once, "the fixture found no rung-4 row at the smallest budget to rewrite" 
    run.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                   encoding="utf-8")
    note = _note_text(tmp_path, run, corpus,
                      skips=_note_skips_file(tmp_path), date="2026-11-26")
    deltas = [ln for ln in note.splitlines() if ln.startswith("- ") and "ΔCE mean" in ln]
    assert len(deltas) == len(R.COMPRESSORS) * len(R.BUDGETS), (
        f"{len(deltas)} delta lines for {len(R.COMPRESSORS)}x{len(R.BUDGETS)} cells")
    for ln in deltas:
        assert R.READOUT_SHAPE in ln, f"a delta line names no readout shape: {ln!r}"
        assert "no-op rows " in ln and "declines " in ln and "undecidable " in ln, ln
    assert R.NO_STABLE_LOGPROB_LINE in note, note
    rung4_low = next(ln for ln in deltas
                     if ln.startswith(f"- {R.RUNG4} budget {R.BUDGETS[0]:.2f}"))
    assert "no-op rows 1" in rung4_low, rung4_low
    assert f"KL@1 {R.NO_KL_CELL}" in rung4_low, (
        "a K=0 cell must print the null KL as the null it is")


def test_a_rung4_mean_with_no_no_op_count_is_refused(tmp_path):
    """#4508 clause 5's refusal: a mean nobody can size is not published.

    A run record that carries no `tokens_retained_pct` cannot say how many of its rows
    the compressor left untouched, so its rung-4 mean is a number over an unknown
    population — the rung-4 row is the BASELINE every other arm is ranked against, so
    the ambiguity is not recoverable downstream. The renderer raises rather than
    publishing, and the CLI turns that into a printed refusal and exit 2, not a note.
    """
    run, corpus = _note_run(tmp_path)
    recs = [json.loads(ln) for ln in run.read_text(encoding="utf-8").splitlines()]
    for rec in recs:
        if rec["compressor"] == R.RUNG4:
            del rec["tokens_retained_pct"]
    run.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                   encoding="utf-8")
    with pytest.raises(R.ReportRefused) as got:
        R.write_note(run, corpus, out_dir=tmp_path / "measurements",
                     floor=NOTE_FLOOR, skips=None, date="2026-11-26")
    assert R.RUNG4 in str(got.value) and "tokens_retained_pct" in str(got.value), got.value

    unwritten = tmp_path / "measurements" / "observation-compression-2026-11-26.md"
    assert not unwritten.exists(), "the refusal wrote a note anyway"
