"""#529 — long worker turns run on a schema-validated execution state.

The behaviour under test is the division of labour, not the token count:
**the model decides which facts from an observation deserve to survive, and
the harness only decides whether the patch is structurally legal and
permitted.** Every test below attacks one half of that sentence, because each
half fails independently:

- a merge that is not null-deleting quietly keeps facts the model deleted;
- a patch merged without validation is the "silently-applied invalid patch"
  the acceptance counts to zero;
- a driver that keeps passing the prior transcript forward has rebuilt the
  append-only chat transcript with extra steps, which is the thing being
  replaced;
- a patch request that drops the tools array re-prefills the whole segment
  (`app/harness/finalizer.py` docstring is the record of that trap), so the
  re-ask must resend the identical array;
- and (#2493) the re-ask must be asked with the budget of the attempt it
  replaces: omitting `max_tokens` there inherited `run_finalizer`'s 1024 while
  the segment's own guided request went out at the template's
  `finalizer_max_tokens` (8192), so the repair could truncate a verdict the same
  request had just produced legally.

No test here touches vLLM: the driver is exercised against a fake
`run_query`, and the token/prefill numbers the item asks for come from
`scripts/replay_run_state.py` against the real engine.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import app.harness
from app.harness.options import RunOptions
from app.harness import run_state as RS
from app.harness.run_state import RunState, RunStateStepError

ROOT = Path(__file__).resolve().parent.parent

STATE_SCHEMA = {
    "title": "distill_run_state",
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "next_action": {"type": "string"},
        "read_of_file": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "steps_done": {"type": "integer"},
        "confidence": {"type": "number"},
    },
    "additionalProperties": False,
}


def make_state(**kw) -> RunState:
    kw.setdefault("job", "session-distill")
    kw.setdefault("schema", STATE_SCHEMA)
    kw.setdefault("max_state_chars", 4_000)
    return RunState(**kw)


def out(reasoning="because", patch=None, action="continue", done=False) -> dict:
    return {
        "reasoning": reasoning,
        "state_patch": {} if patch is None else patch,
        "action": action,
        "done": done,
    }


# ---------------------------------------------------------------------------
# The state object itself
# ---------------------------------------------------------------------------


def test_patch_is_merged_with_null_deletion():
    """`Σ_{t+1} = Σ_t ⊕ ΔΣ_t`, where a null in ΔΣ DELETES the key.

    Without null-deletion the model has no way to forget, and Σ re-becomes
    the transcript this whole module exists to stop being.
    """
    st = make_state()
    st.apply_patch({"goal": "distill", "next_action": "read", "steps_done": 1})
    assert st.values["goal"] == "distill"
    st.apply_patch({"next_action": None, "steps_done": 2})
    assert "next_action" not in st.values, "null must delete, not store"
    assert st.values["goal"] == "distill"
    assert st.values["steps_done"] == 2


def test_invalid_patch_is_rejected_and_never_merged():
    """A wrong-typed patch must not half-apply.

    The paper's open-weight failure taxonomy is 68% premature overwrite /
    20% schema-type / 12% JSON syntax, so schema-type errors are not a
    corner case: they are a fifth of what unconstrained decoding produces.
    """
    st = make_state()
    st.apply_patch({"steps_done": 3, "goal": "keep me"})
    err = st.validate_patch({"steps_done": "three"})
    assert err, "a string in an integer field must be rejected"
    assert st.values["steps_done"] == 3
    assert st.render().count("three") == 0


def test_unknown_keys_are_rejected():
    """`additionalProperties: false` is what makes Σ a schema and not a dump."""
    st = make_state()
    assert st.validate_patch({"scratchpad": "everything I saw"})
    assert "scratchpad" not in st.values


def test_partial_patch_is_valid_but_empty_patch_is_too():
    """ΔΣ is a *patch*: it may name one key, or none (the model adds nothing)."""
    st = make_state()
    assert st.validate_patch({"goal": "one key only"}) == ""
    assert st.validate_patch({}) == ""


def test_state_size_cap_fires_as_a_rejection_not_a_truncation():
    """An unbounded Σ quietly re-becomes the transcript.

    So the cap is enforced as a *rejection* with the current size in the
    message: the model is told to evict, and the eviction is measured rather
    than a silent truncation that drops findings mid-sentence — the same
    truncation-shaped failure the item is replacing.
    """
    st = make_state(max_state_chars=120)
    st.apply_patch({"goal": "short"})
    big = "x" * 2_000
    err = st.validate_patch({"findings": [big]})
    assert err and "120" in err, "the rejection must name the cap it hit"
    assert big not in st.render()
    # Eviction is available to the model: null the key, then write less.
    st.apply_patch({"findings": None})
    assert st.validate_patch({"findings": ["short"]}) == ""


def test_state_persists_as_json_in_the_run_dir(tmp_path):
    st = make_state(run_dir=tmp_path)
    st.apply_patch({"goal": "distill session 42"})
    written = st.save()
    assert written.exists()
    assert json.loads(written.read_text())["values"]["goal"] == "distill session 42"


# ---------------------------------------------------------------------------
# The per-step prompt
# ---------------------------------------------------------------------------


def test_step_prompt_carries_skill_state_and_observation_only():
    """Per-step size is |P| + |Σ| + |O| — nothing carried from earlier steps."""
    st = make_state()
    st.apply_patch({"goal": "distill", "steps_done": 4})
    prompt = RS.build_step_prompt(
        skill_text="THE SKILL BODY",
        state=st,
        observation="the latest tool result",
        step=5,
        max_steps=8,
    )
    assert "THE SKILL BODY" in prompt
    assert '"goal":"distill"' in prompt.replace(" ", "").replace("\n", "")
    assert "the latest tool result" in prompt
    assert "5" in prompt and "8" in prompt


# ---------------------------------------------------------------------------
# The step driver
# ---------------------------------------------------------------------------


@dataclass
class FakeCall:
    messages: list = field(default_factory=list)
    options: Any = None


class FakeQuery:
    """A stand-in `run_query`: yields a segment's events, records its inputs."""

    def __init__(self, segments: list[dict]) -> None:
        self.segments = segments
        self.calls: list[FakeCall] = []

    def __call__(self, messages, options):
        self.calls.append(FakeCall(list(messages), options))
        seg = self.segments[min(len(self.calls) - 1, len(self.segments) - 1)]

        async def gen():
            for evt in seg["events"]:
                yield evt
        return gen()


def am(text="", usage=None, **kw) -> dict:
    return {"type": "assistant_message", "text": text, "tool_calls": [],
            "usage": usage or {"input_tokens": 1000, "output_tokens": 50,
                               "cache_read": 900},
            "iteration": 1, "finish_reason": kw.get("finish_reason", "stop")}


def seg(events=None, structured=None, stop="stop", usage=None) -> dict:
    events = events if events is not None else [am()]
    events = events + [{"type": "result", "stop_reason": stop,
                        "usage": usage or {"input_tokens": 1000,
                                          "finalizer_input_tokens": 900},
                        "num_turns": 1, "structured": structured,
                        "structured_error": ""}]
    return {"events": events}


def template(**kw) -> RunOptions:
    kw.setdefault("model", "primary")
    kw.setdefault("base_url", "http://127.0.0.1:8096")
    kw.setdefault("system_prompt", "SYSTEM")
    return RunOptions(**kw)


def tool_result(content: str, name: str = "Read") -> dict:
    return {"type": "tool_result", "call_id": "c1", "name": name,
            "content": content, "is_error": False}


@pytest.fixture
def no_engine(monkeypatch):
    """The driver must never reach the network in a unit test."""
    def boom(*a, **k):
        raise AssertionError("run_finalizer must not be called here")
    monkeypatch.setattr(RS, "run_finalizer", boom)


def test_turn_runs_steps_until_done_and_carries_only_the_latest_observation(
        monkeypatch, tmp_path, no_engine):
    """The one measurement that says this worked at all.

    `O_t` is the *latest* observation, so step 2 legitimately sees step 1's
    last tool result — what it must not see is step 1's earlier tool result,
    step 1's reasoning, or step 1's message list. If those leak forward the
    driver has rebuilt the append-only transcript with extra steps, and the
    quadratic prompt cost is back.
    """
    q = FakeQuery([
        seg(events=[
            am("step one, looking at the whole transcript"),
            tool_result("SUPERSEDED READ OF THE SESSION FILE"),
            am("still looking"),
            tool_result("LATEST READ OF THE SESSION FILE"),
        ], structured=out(reasoning="REASONING THAT MUST NOT BE REPLAYED",
                          patch={"steps_done": 1, "next_action": "keep going"})),
        seg(structured=out(patch={"steps_done": 2}, action="stop", done=True)),
    ])
    monkeypatch.setattr(app.harness, "run_query", q)
    st = make_state(run_dir=tmp_path)

    res = _run(monkeypatch, st, tmp_path)

    assert res.done is True
    assert len(q.calls) == 2
    step2_blob = json.dumps(q.calls[1].messages)
    assert "LATEST READ OF THE SESSION FILE" in step2_blob, "O_t must carry forward"
    assert "SUPERSEDED READ OF THE SESSION FILE" not in step2_blob
    assert "REASONING THAT MUST NOT BE REPLAYED" not in step2_blob
    assert "looking at the whole transcript" not in step2_blob
    assert "keep going" in step2_blob, "the state must carry the prior step forward"
    # Step 2 starts from ONE message, not from step 1's transcript.
    assert len(q.calls[1].messages) == 1


def test_every_step_asks_for_its_patch_under_the_guided_schema(monkeypatch, tmp_path, no_engine):
    """Patches ride the existing finalizer path, step by step, not one at the end."""
    q = FakeQuery([
        seg(structured=out(patch={"steps_done": 1})),
        seg(structured=out(patch={"steps_done": 2}, done=True)),
    ])
    monkeypatch.setattr(app.harness, "run_query", q)
    _run(monkeypatch, make_state(run_dir=tmp_path), tmp_path)
    envelope = RS.envelope_schema_for(STATE_SCHEMA)
    for call in q.calls:
        assert call.options.final_schema == envelope
        assert call.options.final_schema_prompt
        assert call.options.chat_messages_handle is not None
        assert call.options.visible_tools_capture is not None


def test_invalid_patch_gets_one_reask_then_fails_the_step_loudly(monkeypatch, tmp_path):
    """rejected-invalid → one retry → loud step failure. Never a silent merge,
    and never a silent skip: a step that cannot say what it learned stops."""
    reasks = []

    async def reask(**kw):
        reasks.append(kw)
        return None, "engine says no", {}
    monkeypatch.setattr(RS, "run_finalizer", reask)

    q = FakeQuery([seg(structured=out(patch={"steps_done": "not an int"}))])
    monkeypatch.setattr(app.harness, "run_query", q)
    st = make_state(run_dir=tmp_path)

    with pytest.raises(RunStateStepError) as exc:
        _run(monkeypatch, st, tmp_path, max_steps=1)

    assert len(reasks) == 1, f"exactly one retry expected, got {len(reasks)}"
    assert exc.value.step == 1
    assert st.values == {}, "a rejected patch must not have been merged"
    lines = _trace_lines(tmp_path)
    kinds = [l["kind"] for l in lines]
    assert kinds.count("patch_rejected") == 2, "both attempts are on the record"
    assert "step_failed" in kinds


def test_reask_resends_the_identical_tools_list(monkeypatch, tmp_path):
    """`finalizer.py`: dropping `tools` re-renders the prompt from token zero
    and vLLM re-prefills the whole segment — the exact cost this design is
    meant to avoid, on the error path where it hurts most."""
    captured = [{"type": "function", "function": {"name": "Read"}}]
    asks: list[dict] = []

    async def reask(**kw):
        asks.append(kw)
        return out(patch={"steps_done": 2}, done=True), "", {"input_tokens": 700}
    monkeypatch.setattr(RS, "run_finalizer", reask)

    # The segment carries a visible reply because #867 made one a precondition
    # of `done` being honoured: `if res.done` here is the harness accepting the
    # patch, and a run with nothing to write no longer gets that far (see
    # `test_a_done_with_no_reply_...` below). This test's own subject — the
    # re-ask's byte-identical tools array — is unchanged.
    q = FakeQuery([seg(events=[am("THE NOTE BODY")],
                       structured=out(patch={"steps_done": "nope"}))])

    def query_with_tools(messages, options):
        # The loop's job in production: write the array it actually sent into
        # the caller's capture list, so a later request can be byte-identical.
        q.calls.append(FakeCall(list(messages), options))
        if options.visible_tools_capture is not None:
            options.visible_tools_capture[:] = captured

        async def gen():
            for evt in q.segments[0]["events"]:
                yield evt
        return gen()
    monkeypatch.setattr(app.harness, "run_query", query_with_tools)

    res = _run(monkeypatch, make_state(run_dir=tmp_path), tmp_path, max_steps=1)

    assert res.done is True, "the re-ask's patch was applied"
    assert len(asks) == 1
    assert asks[0]["tools"] == captured, "the re-ask must resend the same array"
    assert asks[0]["chat_messages"] is q.calls[0].options.chat_messages_handle


# ---------------------------------------------------------------------------
# #2493 — the re-ask's budget, and what the retry's trigger still is
# ---------------------------------------------------------------------------


def _one_rejected_patch(monkeypatch, tmp_path, tmpl):
    """Run one step whose first patch parses but fails validation, capture the re-ask.

    The patch is `{"steps_done": "not an int"}` — invalid-but-parsed, the case
    this retry exists for — and the re-ask answers with an engine failure, so the
    step goes on to fail loudly. Returns the captured kwargs and the raised
    `RunStateStepError`.
    """
    import asyncio
    asks: list[dict] = []

    async def reask(**kw):
        asks.append(kw)
        return None, "engine says no", {}
    monkeypatch.setattr(RS, "run_finalizer", reask)

    q = FakeQuery([seg(structured=out(patch={"steps_done": "not an int"}))])
    monkeypatch.setattr(app.harness, "run_query", q)

    with pytest.raises(RS.RunStateStepError) as exc:
        asyncio.run(RS.run_state_turn(
            job="session-distill",
            skill_text="THE SKILL BODY",
            task_block="the task",
            state=make_state(run_dir=tmp_path),
            run_dir=tmp_path,
            template=tmpl,
            max_steps=1,
            iterations_per_step=3,
        ))
    assert len(asks) == 1, f"exactly one re-ask expected, got {len(asks)}"
    return asks[0], exc.value


def test_the_reask_asks_for_the_budget_of_the_attempt_it_replaces(monkeypatch, tmp_path):
    """#2493 clause 1: a default template's re-ask is a 8192-token request, not 1024.

    The segment's own guided request goes out at `options.finalizer_max_tokens`
    (`app/harness/loop.py:1397`; 8192 per the `RunOptions` default at
    `app/harness/options.py:374`). Before this change the re-ask named no budget,
    so Python supplied `run_finalizer`'s `max_tokens: int = 1024`
    (`app/harness/finalizer.py:205`) — a quarter of the budget, on the request
    whose entire job is to produce the verdict the first one failed to shape. The
    1024 default is asserted here as well as the 8192, because the contrast is the
    defect: if the default were ever raised to match, this node would be proving
    nothing and the assert naming it says so.
    """
    kwargs, _err = _one_rejected_patch(monkeypatch, tmp_path, template())

    from app.harness.finalizer import run_finalizer as real_finalizer
    default_budget = inspect.signature(real_finalizer).parameters["max_tokens"].default
    assert default_budget == 1024, (
        "run_finalizer's own default is what the omission used to fall through to;"
        f" it moved to {default_budget}, so this node's contrast needs re-reading")
    assert kwargs["max_tokens"] == 8192, kwargs
    assert kwargs["max_tokens"] != default_budget
    assert kwargs["max_tokens"] == RunOptions(model="m").finalizer_max_tokens, (
        "the re-ask must read the template field, not a literal")


def test_the_reask_follows_a_non_default_finalizer_budget(monkeypatch, tmp_path):
    """#2493 clause 2: the retry tracks the attempt it replaces, not a constant.

    A template built at 256 must produce a 256-token re-ask. Without this, a fix
    that hard-coded 8192 at the call site would satisfy clause 1 and still
    diverge from any job that sets its own finalizer budget — which is what
    `workers/sources/_common.py` builds per job.
    """
    kwargs, _err = _one_rejected_patch(
        monkeypatch, tmp_path, template(finalizer_max_tokens=256))

    assert kwargs["max_tokens"] == 256, kwargs


def test_the_retry_trigger_is_unchanged_and_the_loud_failure_keeps_attempts_two(
        monkeypatch, tmp_path):
    """#2493 clause 3: the trigger is the invalid patch, and the failure stays loud.

    #2370's `should_resample_divergence` is deliberately NOT wired into this path:
    `envelope_schema_for` emits unbounded `reasoning`/`action` strings, so
    `_schema_is_bounded` is False and the predicate can never return True here —
    importing it would delete the re-ask outright and take
    `test_invalid_patch_gets_one_reask_then_fails_the_step_loudly` with it. What
    stays is the original contract: one rejected patch draws exactly one re-ask,
    both attempts are on the record, and the step fails with `attempts=2`.
    """
    _kwargs, err = _one_rejected_patch(monkeypatch, tmp_path, template())

    lines = _trace_lines(tmp_path)
    rejected = [l for l in lines if l["kind"] == "patch_rejected"]
    failed = [l for l in lines if l["kind"] == "step_failed"]
    assert [r["attempt"] for r in rejected] == [1, 2], rejected
    assert len(failed) == 1, failed
    assert failed[0]["attempts"] == 2, failed[0]
    assert err.step == 1

    src = (ROOT / "app" / "harness" / "run_state.py").read_text()
    assert "should_resample_divergence" not in src, (
        "the divergence predicate belongs to the loop path; on this one the"
        " envelope is unbounded so it could only ever switch the re-ask off")


def test_the_loop_path_finalizer_contract_is_untouched():
    """#2493 clause 4: the fix is one call site, not a redefinition of a budget.

    Two defaults hold the line elsewhere and neither may move for this change to
    mean anything: `RunOptions.finalizer_max_tokens` stays 8192 — the value
    `tests/test_board_steward.py::test_the_finalizer_budget_stays_8192_on_both_sides_of_the_reader`
    pins on both sides of the reader — and `run_finalizer`'s own parameter stays
    1024, so the guided-decoding helper still has the small default every other
    caller relies on and this path is the one that overrides it.
    """
    assert RunOptions(model="m").finalizer_max_tokens == 8192
    sig = inspect.signature(RS.run_finalizer)
    assert sig.parameters["max_tokens"].default == 1024
    loop_src = (ROOT / "app" / "harness" / "loop.py").read_text()
    assert "max_tokens=options.finalizer_max_tokens" in loop_src


def test_applied_patch_and_discarded_reasoning_are_traced(monkeypatch, tmp_path, no_engine):
    """Reasoning leaves context but not the record (#525's evidence shape)."""
    # Each segment carries a visible reply: since #867 a `done` with no reply is
    # refused and the step repeats, which would add a third `patch_applied` row
    # and make this count measure the refusal loop instead of the trace.
    q = FakeQuery([
        seg(events=[am("step one notes")],
            structured=out(reasoning="the transcript mentions a GPU crash",
                           patch={"steps_done": 1, "findings": ["GPU crash"]},
                           action="continue")),
        seg(events=[am("the finished note")],
            structured=out(reasoning="enough", patch={"steps_done": 2}, done=True)),
    ])
    monkeypatch.setattr(app.harness, "run_query", q)
    _run(monkeypatch, make_state(run_dir=tmp_path), tmp_path)

    lines = _trace_lines(tmp_path)
    assert any("GPU crash" in json.dumps(l) for l in lines)
    assert any("the transcript mentions a GPU crash" in json.dumps(l) for l in lines)
    assert [l["kind"] for l in lines].count("patch_applied") == 2


def test_cumulative_prompt_tokens_and_prefill_are_reported(monkeypatch, tmp_path, no_engine):
    """The acceptance's go/no-go number is cumulative prompt tokens plus total
    prefill seconds per run — so the driver has to report both."""
    # Both segments reply: a `done` with no visible reply is refused since #867,
    # which would run a third step and inflate the very totals this pins.
    q = FakeQuery([
        seg(events=[am("reading", usage={"input_tokens": 2000, "output_tokens": 10,
                                         "cache_read": 1500})],
            structured=out(patch={"steps_done": 1})),
        seg(events=[am("the note", usage={"input_tokens": 2100, "output_tokens": 10,
                                          "cache_read": 1600})],
            structured=out(patch={"steps_done": 2}, done=True)),
    ])
    monkeypatch.setattr(app.harness, "run_query", q)
    res = _run(monkeypatch, make_state(run_dir=tmp_path), tmp_path)

    assert res.prompt_tokens == 4100 + 1800        # iterations + finalizer asks
    assert res.cached_tokens == 3100
    assert res.uncached_prompt_tokens == res.prompt_tokens - res.cached_tokens
    assert res.prefill_seconds > 0


def test_state_is_saved_per_step_and_survives_a_failing_later_step(monkeypatch, tmp_path):
    """A run that dies at step 2 leaves step 1's state on disk for the retry."""
    async def reask(**kw):
        return None, "no", {}
    monkeypatch.setattr(RS, "run_finalizer", reask)
    q = FakeQuery([
        seg(structured=out(patch={"steps_done": 1, "goal": "half done"})),
        seg(structured=out(patch={"steps_done": "bad"})),
    ])
    monkeypatch.setattr(app.harness, "run_query", q)
    st = make_state(run_dir=tmp_path)
    with pytest.raises(RunStateStepError):
        _run(monkeypatch, st, tmp_path, max_steps=2)

    saved = json.loads((tmp_path / "run_state.json").read_text())
    assert saved["values"]["steps_done"] == 1
    assert saved["values"]["goal"] == "half done"


# ---------------------------------------------------------------------------
# #867 — `done` needs a deliverable
#
# Three shapes of the same hole, all reproduced on live `main` before this was
# written, each with its own assertion below:
#   * a clean `stop_reason="stop"` segment that answers `done=true` and says
#     nothing ends the run with a zero-character reply at attempt 1;
#   * a segment that exhausts `iterations_per_step` has no assistant text by
#     construction (its in-band finalizer is skipped on purpose), so its re-ask
#     answering `done=true` ends the run the same way — at any ceiling value;
#   * `result.text` was assigned from the last segment, so a note written at
#     step 1 and followed by a silent ending step was thrown away.
# `session_distill` then had `turn.text == ""` to write a note from, while the
# run record said the model had stopped cleanly.
# ---------------------------------------------------------------------------


def test_a_done_with_no_reply_is_refused_and_the_run_continues(
        monkeypatch, tmp_path, no_engine):
    """The rule `PATCH_PROMPT` already stated, now enforced by the harness.

    One valid envelope, `done=true`, no assistant text, `stop_reason="stop"` —
    nothing about it is malformed, which is why this needed no engine and no
    ceiling to reproduce. The run must not report completion, and the model has
    to be told what is missing rather than left to repeat itself.
    """
    q = FakeQuery([seg(structured=out(patch={"steps_done": 1}, action="stop",
                                      done=True))])
    monkeypatch.setattr(app.harness, "run_query", q)

    res = _run(monkeypatch, make_state(run_dir=tmp_path), tmp_path, max_steps=3)

    assert res.done is False, "a done with no deliverable was honoured"
    assert not res.text.strip()
    assert res.done_refusals == 3, "every step's done was refused"
    assert len(res.steps) == 3, "the run continued instead of ending"
    # The refusal is said out loud, not just counted: a withheld `done` with no
    # instruction gets the same `done` back.
    assert "Write the deliverable" in RS.NO_DELIVERABLE_OBSERVATION
    assert "harness refused" in json.dumps([c.messages for c in q.calls[1:]]), \
        "the refusal never reached the model"
    kinds = [l["kind"] for l in _trace_lines(tmp_path)]
    assert kinds.count("done_refused") == 3, "the refusals are on the record"


def test_a_done_with_a_reply_still_ends_the_run(monkeypatch, tmp_path, no_engine):
    """The gate is on the deliverable, not on `done` in general.

    Without this pairing the fix could be satisfied by simply never honouring
    `done`, which would turn every successful state-carried run into a
    `max_steps` failure.
    """
    q = FakeQuery([
        seg(events=[am("## Struggles\n- nothing")],
            structured=out(patch={"steps_done": 1}, action="stop", done=True)),
    ])
    monkeypatch.setattr(app.harness, "run_query", q)

    res = _run(monkeypatch, make_state(run_dir=tmp_path), tmp_path, max_steps=4)

    assert res.done is True
    assert res.done_refusals == 0
    assert len(q.calls) == 1, "a delivering done must end the run at once"
    assert res.text == "## Struggles\n- nothing"


@pytest.mark.parametrize("iterations_per_step", [1, 2, 4])
def test_a_ceiling_segment_whose_reask_says_done_delivers_nothing_and_is_refused(
        monkeypatch, tmp_path, iterations_per_step):
    """The shipped failure, at every ceiling value rather than the one observed.

    A segment that burns its whole iteration budget on tool calls ends
    `stop_reason="max_turns"` with its in-band finalizer skipped by design, so
    the assistant text is empty and the patch can only come from the re-ask. In
    round SM_20260911_170801's replay that re-ask answered `done=true` after ONE
    step of a 500,934-byte session — and the run returned a zero-character
    reply while reporting `stop_reason=stop`.
    """
    async def reask(**kw):
        return out(patch={"steps_done": 1}, action="stop", done=True), "", \
            {"input_tokens": 700}
    monkeypatch.setattr(RS, "run_finalizer", reask)

    events = []
    for _ in range(iterations_per_step):
        events.append({"type": "assistant_message", "text": "", "tool_calls":
                       [{"id": "c1", "name": "Read", "input": {}}],
                       "usage": {"input_tokens": 900, "output_tokens": 40,
                                 "cache_read": 0},
                       "iteration": 1, "finish_reason": "tool_calls"})
        events.append(tool_result("a chunk of the session file"))
    q = FakeQuery([seg(events=events, structured=None, stop="max_turns")])
    monkeypatch.setattr(app.harness, "run_query", q)

    res = _run(monkeypatch, make_state(run_dir=tmp_path), tmp_path,
               max_steps=3, iterations_per_step=iterations_per_step)

    assert not (res.done and not res.text.strip()), \
        "the run reported completion with an empty deliverable"
    assert res.done is False
    assert res.done_refusals == 3
    assert all(s.attempts == 2 for s in res.steps), \
        "the ceiling segment's missing patch went through the re-ask"


def test_the_deliverable_is_the_last_nonempty_text_of_the_run_not_of_the_last_step(
        monkeypatch, tmp_path, no_engine):
    """A note written at step 1 survives a silent ending step.

    `result.text = segment["text"]` made the last segment the whole answer, so
    the run in this test — note at step 1, `done=true` with no text at step 2 —
    returned `text=''` and lost a deliverable that had existed for a full step.
    """
    q = FakeQuery([
        seg(events=[am("## Struggles\n- the real note body\n\n## Confidence\n"
                       "0.8: fine")],
            structured=out(patch={"steps_done": 1})),
        seg(events=[am("")],
            structured=out(patch={"steps_done": 2}, action="stop", done=True)),
    ])
    monkeypatch.setattr(app.harness, "run_query", q)

    res = _run(monkeypatch, make_state(run_dir=tmp_path), tmp_path)

    assert res.done is True, "the deliverable existed, so the done is real"
    assert res.text == ("## Struggles\n- the real note body\n\n"
                        "## Confidence\n0.8: fine")
    assert res.done_refusals == 0, "the carried note already satisfied the gate"


# ---------------------------------------------------------------------------
# The worker turn path actually imports it
# ---------------------------------------------------------------------------


def test_worker_turn_path_imports_the_state_module():
    """`git log -S'RunState'` empty was the premise. This is the assertion
    that keeps it from quietly becoming true again."""
    src = (ROOT / "workers" / "sources" / "_common.py").read_text()
    assert "from app.harness.run_state import" in src
    import workers.sources._common as C
    assert C.RunState is RunState
    assert callable(C.run_prompt_with_run_state)


def test_state_turns_are_banned_from_the_automod_tools_like_any_worker_turn(
        monkeypatch, tmp_path, no_engine):
    """A stateful worker turn is still a worker turn: same tool ban."""
    q = FakeQuery([seg(structured=out(patch={}, done=True))])
    monkeypatch.setattr(app.harness, "run_query", q)
    from workers.sources._common import WORKER_AUTOMOD_BAN
    _run_common(monkeypatch, tmp_path)
    opts = q.calls[0].options
    for tool in WORKER_AUTOMOD_BAN:
        assert tool in opts.disallowed_tools


def test_loop_records_the_tools_it_sent_and_the_finalizer_input_tokens():
    """Two additive facts the driver depends on, both invisible before #529:
    the exact tools array a segment used, and the guided-decoding request's
    own prompt tokens — which used to be dropped, under-reporting every
    structured verdict and flattering any comparison against a baseline."""
    from app.harness.options import RunOptions as O
    assert O(model="m").visible_tools_capture is None
    loop_src = (ROOT / "app" / "harness" / "loop.py").read_text()
    assert "visible_tools_capture" in loop_src
    assert "finalizer_input_tokens" in loop_src


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reask_calls(mod) -> list:
    return getattr(mod, "_test_reasks", [])


def _run(monkeypatch, state: RunState, run_dir: Path, **kw):
    import asyncio
    kw.setdefault("max_steps", 4)
    kw.setdefault("iterations_per_step", 3)
    return asyncio.run(RS.run_state_turn(
        job="session-distill",
        skill_text="THE SKILL BODY",
        task_block="the task",
        state=state,
        run_dir=run_dir,
        template=template(),
        **kw,
    ))


def _run_common(monkeypatch, run_dir: Path):
    import asyncio
    from workers.sources._common import run_prompt_with_run_state
    st = make_state(run_dir=run_dir)
    return asyncio.run(run_prompt_with_run_state(
        "do the thing",
        job="session-distill",
        skill_text="THE SKILL BODY",
        state=st,
        run_dir=run_dir,
        max_steps=1,
    ))


def _trace_lines(run_dir: Path) -> list[dict]:
    path = run_dir / "state-trace.ndjson"
    assert path.exists(), "the run's NDJSON trace is the only rescue path"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
