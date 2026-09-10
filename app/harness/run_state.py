"""Schema-validated execution state for long worker turns (backlog #529).

What this replaces
------------------
A worker turn enters the harness as a single user message
(`workers/sources/_common.py`) and then only ever grows. `loop.py` appends
every assistant turn and every tool result to `chat_messages` for the whole
life of the turn, and the only thing that ever gives characters back is
pressure-triggered microcompaction, which fires at 0.8 of the truncation
threshold and so never engages on a mid-size turn. The prompt at iteration 20
is therefore the prompt at iteration 1 plus everything the model saw in
between, cumulative prompt cost is quadratic in steps, and — worse than
expensive — a fact read at iteration 2 is still being asserted at iteration
20 whether or not the thing it describes is still true.

What this does instead
----------------------
SKILL.state (arXiv 2608.26263) splits a turn into three inputs per step —
the immutable skill specification `P`, a structured state `Σ_t` owned by the
harness, and the latest observation `O_t` — and three outputs: transient
reasoning `R_t`, a state *patch* `ΔΣ_t`, and the next action. The harness
validates the patch against a declared schema, merges it with null-deletion
(`Σ_{t+1} = Σ_t ⊕ ΔΣ_t`), applies the action, and **discards R_t** instead of
replaying it. Per-step prompt size is `|P| + |Σ| + |O|`, so cumulative cost is
linear in steps.

The division of labour is the load-bearing part, and it is asymmetric on
purpose: **the model decides which facts from an observation deserve to
survive; the harness only decides whether a patch is structurally legal and
permitted.** Nothing here knows what a distillation finding means. It knows
`steps_done` is not an integer, and it refuses the patch.

Where each piece lives in Lloyd
------------------------------
A *step* here is one bounded call into `run_query`, with a message list
containing only this step's prompt. The patch arrives through the turn's
existing structured-verdict path (`options.final_schema` →
`app/harness.finalizer`), which is the only guided-decoding hook this harness
has: a grammar constrains `content`, and mid-turn the model has to stay free
to emit qwen3_xml tool calls, so the patch cannot ride the turn itself. That
is also why a rejected patch is retried with the **identical tools array** —
`finalizer.py` documents the trap, and the error path is where re-prefilling
the segment would cost the most.

The limits, stated where they can be hit
---------------------------------------
This trades long context full of errors for **state projection errors**, and
the trade is only safe when the schema is known up front and the task is
closer to a linear chain than to open-ended research: once the evidence that
would have contradicted a wrong projection has been discarded from context,
the run cannot notice it. Two things mitigate that and neither is optional:
`Σ` carries claims with sources, not facts, and the step prompt tells the
model to re-read a source when a carried value is load-bearing; and every
discarded `R_t` plus every applied and rejected patch goes to the run's
NDJSON (`state-trace.ndjson`), which is the only rescue path afterwards.

Do not use this on the interactive loop. The position-0 append design there
is backlog #520's subject, and rewriting `Σ` every step is directly opposed
to keeping a KV prefix stable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Module-level so tests can patch `run_state.run_finalizer`, and so there is
# exactly one guided-decoding client in the tree rather than a second one
# invented here.
from app.harness.finalizer import run_finalizer

logger = logging.getLogger("lloyd-run-state")

#: Prompt throughput measured on the primary (vLLM `prompt_tokens_total`
#: rate, 4,501 tok/s at the 2026-09-09 bundle). Only used to express the
#: uncached prompt tokens a run consumed as seconds; the authoritative
#: number is vLLM's own `vllm:request_prefill_time_seconds` delta, which
#: `scripts/replay_run_state.py` reads around a run.
PREFILL_TOKENS_PER_SECOND = 4500.0

STATE_FILENAME = "run_state.json"
TRACE_FILENAME = "state-trace.ndjson"

PATCH_PROMPT = (
    "Emit the state patch for the step you just completed. Three fields "
    "matter: `reasoning` is your working — it is recorded and then discarded, "
    "it is not replayed to you, so write down anything the next step must "
    "know. `state_patch` names ONLY the keys of the execution state that this "
    "step's observation changed or established; omit a key to leave it alone, "
    "set it to null to delete it, and never restate a key you did not "
    "actually re-check against the observation. `action` is what you are doing "
    "next. Set `done` true only when the deliverable the skill asks for is "
    "complete, and then make your visible reply that deliverable."
)

REPAIR_PROMPT_TEMPLATE = (
    "Your previous answer was rejected by the harness and nothing was "
    "recorded. The rejection was: {error}\n\n"
    "Emit the state patch again, valid against the schema. Change only what "
    "the rejection requires. If the state is genuinely full, DELETE keys by "
    "setting them to null rather than adding new ones."
)


class RunStateError(RuntimeError):
    """The harness refused something. Always loud, never merged anyway."""


class RunStateStepError(RunStateError):
    """A step's patch was invalid on both attempts.

    This raises rather than continuing with an unstored step: a run that
    carries on while its state is not being written is the append-only
    transcript again, with the schema as decoration.
    """

    def __init__(self, job: str, step: int, error: str) -> None:
        super().__init__(
            f"{job}: step {step} produced no valid state patch after one retry "
            f"({error}) — run stopped, nothing from this step was merged")
        self.job = job
        self.step = step
        self.error = error


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _nullable(spec: dict) -> dict:
    """Allow a declared field to be null, which is how a patch says 'delete'.

    Declaring the nullability rather than accepting any type is what keeps
    guided decoding usable: vLLM builds the grammar from this schema, so the
    shape it constrains is the shape the model is physically able to produce.
    """
    if spec.get("type") == "null":
        return dict(spec)
    return {"anyOf": [dict(spec), {"type": "null"}]}


def patch_schema_for(state_schema: dict) -> dict:
    """The `ΔΣ` schema: every declared key optional, nullable, nothing else."""
    props = {
        name: _nullable(spec)
        for name, spec in (state_schema.get("properties") or {}).items()
    }
    return {
        "title": f"{state_schema.get('title') or 'run_state'}_patch",
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }


def envelope_schema_for(state_schema: dict) -> dict:
    """The transient per-step object: `{reasoning, state_patch, action, done}`.

    `reasoning` is in the schema because the model has to emit it somewhere
    for it to exist, and putting it beside the patch keeps the discard honest:
    the harness receives it, traces it, and never feeds it back.
    """
    return {
        "title": f"{state_schema.get('title') or 'run_state'}_step",
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "What this step learned, in your own words.",
            },
            "state_patch": patch_schema_for(state_schema),
            "action": {
                "type": "string",
                "description": "What you are doing next, one sentence.",
            },
            "done": {
                "type": "boolean",
                "description": "True only when the deliverable is complete.",
            },
        },
        "required": ["reasoning", "state_patch", "action"],
        "additionalProperties": False,
    }


def _validate(instance: dict, schema: dict) -> str:
    """"", or the first validation error. Missing jsonschema is an error.

    `jsonschema` is not declared in requirements.txt — it is in the
    environment transitively via `mcp`. That is a debt (backlog item filed
    while implementing #529), and the debt is repaid by refusing to validate
    rather than by skipping validation: an unvalidated merge is precisely the
    silently-applied invalid patch this module exists to prevent.
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover — the env has it via mcp
        raise RunStateError(
            "jsonschema is not importable, so state patches cannot be "
            "validated; refusing rather than merging unvalidated patches"
        ) from exc
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(p) for p in exc.absolute_path) or "(root)"
        return f"{path}: {exc.message}"[:400]
    return ""


# ---------------------------------------------------------------------------
# Σ — the harness-owned execution state
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    """`Σ`: a JSON object the harness owns, validates, merges, and renders.

    The model proposes; this owns. `apply_patch` validates before it merges,
    so there is no code path here that merges something invalid — "zero
    silently-applied invalid patches" is structural, not a convention.
    """

    job: str
    schema: dict
    values: dict = field(default_factory=dict)
    #: An unbounded `Σ` quietly re-becomes the transcript, so the cap is part
    #: of the schema's contract. Overflow is a rejection with the size in the
    #: message, never a truncation: the model is told to evict, and how often
    #: it had to is measurable from the trace.
    max_state_chars: int = 4_000
    run_dir: Path | None = None
    trace: "RunStateTrace | None" = None
    steps_applied: int = 0
    rejections: int = 0

    def __post_init__(self) -> None:
        if self.run_dir is not None and self.trace is None:
            Path(self.run_dir).mkdir(parents=True, exist_ok=True)
            self.trace = RunStateTrace(Path(self.run_dir) / TRACE_FILENAME)

    @property
    def patch_schema(self) -> dict:
        return patch_schema_for(self.schema)

    @property
    def envelope_schema(self) -> dict:
        return envelope_schema_for(self.schema)

    def _merged(self, patch: dict) -> dict:
        """`Σ ⊕ ΔΣ` without mutating `Σ` — the dry run the cap is checked on."""
        out = json.loads(json.dumps(self.values)) if self.values else {}
        for key, value in patch.items():
            if value is None:
                out.pop(key, None)
            elif isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = value
        return out

    def size_chars(self, values: dict | None = None) -> int:
        return len(_compact(values if values is not None else self.values))

    def validate_patch(self, patch: Any) -> str:
        """"" on success, else the reason this patch may not be merged."""
        if not isinstance(patch, dict):
            return f"state_patch is {type(patch).__name__}, expected an object"
        err = _validate(patch, self.patch_schema)
        if err:
            return err
        merged = self._merged(patch)
        size = len(_compact(merged))
        if size > self.max_state_chars:
            return (
                f"applying this patch would make the execution state {size} "
                f"characters, over the {self.max_state_chars}-character cap. "
                f"Delete what you no longer need by setting keys to null "
                f"(keys currently in state: {sorted(self.values)}) and try again."
            )
        return ""

    def apply_patch(self, patch: dict) -> dict:
        """Merge `ΔΣ` into `Σ`. Raises rather than merging an illegal patch."""
        err = self.validate_patch(patch)
        if err:
            self.rejections += 1
            raise RunStateError(f"{self.job}: refused state patch — {err}")
        before = set(self.values)
        self.values = self._merged(patch)
        self.steps_applied += 1
        after = set(self.values)
        return {
            "added": sorted(after - before),
            "removed": sorted(before - after),
            "changed": sorted(
                k for k in (after & before)
                if _compact(self.values[k]) != _compact(patch.get(k, self.values[k]))
            ),
        }

    def render(self) -> str:
        """`Σ` as the model sees it: compact JSON, no whitespace to pay for."""
        return _compact(self.values)

    def save(self) -> Path | None:
        """Persist `Σ` to the run dir, one file, after every applied patch.

        Per-step rather than at the end: the point of a state object is that a
        run which dies at step 11 leaves step 10's state readable.
        """
        if self.run_dir is None:
            return None
        path = Path(self.run_dir) / STATE_FILENAME
        path.write_text(json.dumps({
            "job": self.job,
            "schema_title": self.schema.get("title"),
            "values": self.values,
            "state_chars": self.size_chars(),
            "max_state_chars": self.max_state_chars,
            "steps_applied": self.steps_applied,
            "rejections": self.rejections,
            "updated_at": _now(),
        }, indent=2), encoding="utf-8")
        return path


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _compact(values: dict) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunStateTrace:
    """The run's NDJSON: every reasoning block that left context, every patch
    that was applied, and every patch that was refused.

    Reasoning is discarded from *context*, not from the record. That is the
    same evidence-binding shape as #525: the run stays cheap to re-read and
    still auditable afterwards, which is the only thing standing between this
    design and an unrecoverable wrong projection.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields: Any) -> dict:
        rec = {"ts": _now(), "kind": kind}
        rec.update(fields)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


# ---------------------------------------------------------------------------
# The per-step prompt
# ---------------------------------------------------------------------------


def build_step_prompt(*, skill_text: str, state: RunState, observation: str,
                      step: int, max_steps: int) -> str:
    """`P + Σ + O` — and nothing else.

    The staleness line is not decoration. With the transcript gone, `Σ` is
    the only thing asserting what earlier steps found, so a value carried in
    state that describes a file or service is a claim about the past. Telling
    the model that, and naming the re-read as the remedy, is what turns the
    drift case from an invisible failure into a recoverable one.
    """
    return "\n".join([
        f'[SYSTEM: You are running the "{state.job}" worker job, step {step} of '
        f"at most {max_steps}. Work autonomously and do not ask for "
        "confirmation. You see only the skill, the execution state, and the "
        "latest observation — earlier steps are not repeated for you, so "
        "anything the next step needs must be put into the state patch.]",
        "",
        "SKILL (immutable specification):",
        skill_text.strip(),
        "",
        f"EXECUTION STATE (Σ, {state.size_chars()} of {state.max_state_chars} "
        "characters; owned by the harness, proposed only by you):",
        state.render(),
        "",
        "The state is a set of CLAIMS WITH SOURCES, not facts. A value that "
        "describes a file, service or endpoint is stale the moment that thing "
        "changes, and nothing above tells you when it was written: re-read the "
        "source whenever a carried value decides what you are about to do.",
        "",
        "LATEST OBSERVATION (O — what the last step's action produced; this is "
        "the freshest thing you have, and it outranks Σ):",
        observation.strip() or "(no observation yet — this is the first step)",
    ])


# ---------------------------------------------------------------------------
# The step driver
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    prompt_tokens: int = 0
    cached_tokens: int = 0
    finalizer_prompt_tokens: int = 0
    iterations: int = 0
    action: str = ""
    done: bool = False
    state_chars: int = 0
    attempts: int = 1
    error: str = ""


@dataclass
class RunStateResult:
    state: RunState
    steps: list[StepRecord] = field(default_factory=list)
    text: str = ""
    done: bool = False
    prompt_tokens: int = 0
    cached_tokens: int = 0
    finalizer_prompt_tokens: int = 0
    iterations: int = 0
    prefill_seconds: float = 0.0

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def num_turns(self) -> int:
        return sum(s.iterations for s in self.steps)

    def failure_summary(self) -> str:
        return (f"run-state turn ended after {len(self.steps)} step(s), "
                f"done={self.done}, prompt_tokens={self.prompt_tokens}")


def _repair_prompt(error: str) -> str:
    return REPAIR_PROMPT_TEMPLATE.format(error=error)


async def _run_segment(*, run_query_fn, messages: list[dict], options: Any) -> dict:
    """One bounded call into the harness. Returns what the segment cost and said."""
    out: dict[str, Any] = {
        "text": "", "prompt_tokens": 0, "cached_tokens": 0,
        "finalizer_prompt_tokens": 0, "iterations": 0, "iterations_text": "",
        "structured": None, "structured_error": "", "stop_reason": None,
        "observation": "",
    }
    async for evt in run_query_fn(messages, options):
        kind = evt.get("type")
        if kind == "assistant_message":
            usage = evt.get("usage") or {}
            out["prompt_tokens"] += int(usage.get("input_tokens") or 0)
            out["cached_tokens"] += int(usage.get("cache_read") or 0)
            out["iterations"] += 1
            if evt.get("text"):
                out["iterations_text"] = evt["text"]
        elif kind == "tool_result":
            # O_{t+1}: only the newest observation survives into the next step.
            out["observation"] = str(evt.get("content") or "")
        elif kind == "result":
            out["structured"] = evt.get("structured")
            out["structured_error"] = str(evt.get("structured_error") or "")
            out["stop_reason"] = evt.get("stop_reason")
            usage = evt.get("usage") or {}
            out["finalizer_prompt_tokens"] += int(
                usage.get("finalizer_input_tokens") or 0)
            if not out["iterations_text"] and evt.get("response_text"):
                out["iterations_text"] = str(evt["response_text"])
    out["text"] = out.pop("iterations_text") or ""
    if not out["observation"]:
        out["observation"] = out["text"]
    return out


def _refuse(state: RunState, step: int, attempt: int, envelope_out: Any,
            error: str, segment: dict) -> None:
    """A rejection is a record, not a shrug.

    `rejections` on the state and `patch_rejected` lines in the NDJSON are
    what make "zero silently-applied invalid patches" checkable: the count is
    either in the log or it is not.
    """
    state.rejections += 1
    if state.trace is not None:
        state.trace.write(
            "patch_rejected", job=state.job, step=step, attempt=attempt,
            error=error,
            reasoning=(envelope_out or {}).get("reasoning", "")
            if isinstance(envelope_out, dict) else "",
            state_patch=(envelope_out or {}).get("state_patch")
            if isinstance(envelope_out, dict) else envelope_out,
            stop_reason=segment.get("stop_reason"),
            structured_error=segment.get("structured_error", ""),
            prompt_tokens=segment.get("prompt_tokens", 0),
        )
    logger.warning("run-state %s step %d attempt %d: patch refused — %s",
                   state.job, step, attempt, error)


async def run_state_turn(
    *,
    job: str,
    skill_text: str,
    task_block: str,
    state: RunState,
    run_dir: Path,
    template: Any,
    max_steps: int = 6,
    iterations_per_step: int = 4,
    prefill_tokens_per_second: float = PREFILL_TOKENS_PER_SECOND,
) -> RunStateResult:
    """Run one worker job as a sequence of state-carried steps.

    `template` is a prepared `RunOptions` — the caller owns tool policy, and
    `workers/sources/_common.py` is what bakes the automod ban in. This owns
    the per-step parts: a fresh message list, the segment's iteration budget,
    and the guided-decoding request for the patch.

    `iterations_per_step` is the segment budget. A segment that exhausts it
    ends with `stop_reason="max_turns"`, and the finalizer skips itself there
    on purpose (a turn that ran out of budget has no verdict to restate) — so
    a missing patch lands in the same refuse → one retry → loud failure path
    as an invalid one, rather than silently ending the run early.
    """
    from app.harness import run_query  # late: patchable, and avoids an edge at import

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if state.trace is None:
        state.trace = RunStateTrace(run_dir / TRACE_FILENAME)
    envelope = state.envelope_schema

    result = RunStateResult(state=state)
    observation = task_block

    for step in range(1, max_steps + 1):
        handle: list[dict] = [{"role": "user", "content": build_step_prompt(
            skill_text=skill_text, state=state, observation=observation,
            step=step, max_steps=max_steps)}]
        # The loop writes the tools array it actually sent in here. The re-ask
        # below needs that exact array: see app/harness/finalizer.py.
        captured_tools: list[dict] = []
        options = replace(
            template,
            chat_messages_handle=handle,
            visible_tools_capture=captured_tools,
            max_turns=iterations_per_step,
            final_schema=envelope,
            final_schema_prompt=PATCH_PROMPT,
        )

        segment = await _run_segment(
            run_query_fn=run_query, messages=list(handle), options=options)

        patch_out = segment["structured"]
        error = _validate_output(state, patch_out, segment)
        attempts = 1

        if error:
            # One retry, through the same guided-decoding path, against the
            # same prefix and the same tools array.
            attempts = 2
            _refuse(state, step, 1, patch_out, error, segment)
            retry_out, retry_err, _usage = await run_finalizer(
                base_url=template.base_url,
                model=template.model,
                chat_messages=handle,
                tools=captured_tools or None,
                schema=envelope,
                prompt=_repair_prompt(error),
                api_key=template.api_key,
                priority=template.priority,
                cancel_event=template.cancel_event,
            )
            error = retry_err or _validate_output(state, retry_out, segment)
            patch_out = retry_out

        if error:
            _refuse(state, step, attempts, patch_out, error, segment)
            if state.trace is not None:
                state.trace.write(
                    "step_failed", job=job, step=step, error=error,
                    attempts=attempts, prompt_tokens=segment["prompt_tokens"])
            raise RunStateStepError(job, step, error)

        assert isinstance(patch_out, dict)
        patch = patch_out.get("state_patch") or {}
        evicted = state.apply_patch(patch)
        saved = state.save()

        record = StepRecord(
            step=step,
            prompt_tokens=segment["prompt_tokens"],
            cached_tokens=segment["cached_tokens"],
            finalizer_prompt_tokens=segment["finalizer_prompt_tokens"],
            iterations=segment["iterations"],
            action=str(patch_out.get("action") or ""),
            done=bool(patch_out.get("done")),
            state_chars=state.size_chars(),
            attempts=attempts,
        )
        result.steps.append(record)
        result.prompt_tokens += (segment["prompt_tokens"]
                                 + segment["finalizer_prompt_tokens"])
        result.cached_tokens += segment["cached_tokens"]
        result.finalizer_prompt_tokens += segment["finalizer_prompt_tokens"]
        result.iterations += segment["iterations"]
        result.text = segment["text"]
        result.done = record.done

        if state.trace is not None:
            # The reasoning block is written here and never sent again.
            state.trace.write(
                "patch_applied", job=job, step=step, attempts=attempts,
                reasoning=str(patch_out.get("reasoning") or ""),
                state_patch=patch, changed=evicted, action=record.action,
                done=record.done, state_chars=record.state_chars,
                prompt_tokens=segment["prompt_tokens"],
                cached_tokens=segment["cached_tokens"],
                finalizer_prompt_tokens=segment["finalizer_prompt_tokens"],
                iterations=segment["iterations"],
                state_file=str(saved) if saved else "",
            )

        if record.done:
            break
        observation = segment["observation"]

    uncached = result.uncached_prompt_tokens
    result.prefill_seconds = round(
        uncached / max(1.0, float(prefill_tokens_per_second)), 3)
    return result


def _validate_output(state: RunState, envelope_out: Any, segment: dict) -> str:
    """Validate the whole transient object, then the patch inside it.

    Guided decoding should make both checks trivial; the fallback path in
    `finalizer.py` is that the engine may accept the schema under one spelling
    and ignore it under the other, and an ignored grammar is exactly the
    unconstrained decoding the paper's failure taxonomy measures.
    """
    if not isinstance(envelope_out, dict):
        why = segment.get("structured_error") or "no structured patch returned"
        return f"{why} (stop_reason={segment.get('stop_reason')})"
    err = _validate(envelope_out, state.envelope_schema)
    if err:
        return err
    return state.validate_patch(envelope_out.get("state_patch"))
