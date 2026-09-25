"""A scripted engine and a scripted MCP pool for driving the REAL `run_query`.

Not a `test_*.py` file, so pytest does not collect it; a broken fixture is a
collection error in every file that imports it rather than a quiet red run.

Why this exists (P13.0). The loop's ordering properties — position 0 inserted
once, history in wire order, an inject lifted past the batch, captions counted
in wire order, `_pre_dispatch` sequential — used to be pinned with
source-substring checks on `run_query`. Those certify that a line of
text is still there, not that the loop still behaves; a refactor that keeps the
text and breaks the property passes them, and one that fixes the property with
different wording fails them. These fakes replace only the two process seams the
loop crosses, both inside the loop module's own namespace:

  - `app.harness.loop.stream_chat` — the HTTP boundary to vLLM. `ReplayEngine`
    yields the chunk shapes the real client yields (`client.stream_chat` hands
    back the parsed `data:` objects): reasoning deltas under `reasoning` (vLLM
    0.23+) or `reasoning_content`, content deltas, `tool_calls` deltas split
    into a name frame and argument fragments, the finish frame, and a trailing
    `choices: []` usage chunk. Every request is recorded — a snapshot of the
    messages as sent and the exact `tools` object — because the invariants are
    about what went on the wire.
  - `app.harness.loop._build_pool` — MCP discovery and dispatch. `ReplayPool`
    advertises tools with the server's own `readOnlyHint`, answers `call_tool`
    from a dict (a string, an exception to raise, or a callable), sleeps
    `delay_by_call_id[call_id]` first so a test can reorder completions, and
    records start order, completion order, cancellations and peak overlap.

`install(monkeypatch, engine, pool)` wires both; `drive(options, messages)`
collects the events of one turn.
"""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from app.harness import loop as L


# ── the engine ──────────────────────────────────────────────────────────────

@dataclass
class Step:
    """One scripted completion.

    `tool_calls` entries are `{"id", "name", "arguments": dict}`. `raise_` is
    an exception raised when the request is made, before any chunk — the shape
    of a rejected prompt (`ContextOverflowError`) — and the step is consumed.
    `reasoning_key` picks the spelling the engine streams reasoning under.
    """

    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raise_: BaseException | None = None
    reasoning_key: str = "reasoning"


def tool_call(call_id: str, name: str, summary: str | None = None,
              **arguments: Any) -> dict[str, Any]:
    """A scripted tool call. `summary` is the injected caption argument; left
    as None the call carries none, which is what the caption ratchet counts."""
    args = dict(arguments)
    if summary is not None:
        args = {"summary": summary, **args}
    return {"id": call_id, "name": name, "arguments": args}


@dataclass
class Request:
    """One request the loop sent: messages as they were at send time."""

    messages: list[dict[str, Any]]
    tools: Any
    kwargs: dict[str, Any]


class ReplayEngine:
    """Scripted `stream_chat`. The last step repeats if the loop asks for more,
    so a `max_turns` guard never changes what a test observes."""

    def __init__(self, steps: list[Step]):
        self.steps = list(steps) or [Step(text="done")]
        self.requests: list[Request] = []

    def __call__(self, **kwargs: Any):
        idx = min(len(self.requests), len(self.steps) - 1)
        self.requests.append(Request(
            messages=copy.deepcopy(kwargs.get("messages") or []),
            tools=kwargs.get("tools"),
            kwargs={k: v for k, v in kwargs.items() if k != "messages"},
        ))
        return self._gen(self.steps[idx])

    async def _gen(self, step: Step):
        if step.raise_ is not None:
            raise step.raise_
        if step.reasoning:
            # Two frames, so the loop's accumulation is exercised, not a
            # single assignment.
            half = max(1, len(step.reasoning) // 2)
            for part in (step.reasoning[:half], step.reasoning[half:]):
                if part:
                    yield {"choices": [{"delta": {step.reasoning_key: part}}]}
        if step.text:
            yield {"choices": [{"delta": {"content": step.text}}]}
        for i, tc in enumerate(step.tool_calls):
            raw = json.dumps(tc.get("arguments") or {})
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"], "arguments": ""},
            }]}}]}
            cut = len(raw) // 2
            for frag in (raw[:cut], raw[cut:]):
                if frag:
                    yield {"choices": [{"delta": {"tool_calls": [{
                        "index": i, "function": {"arguments": frag},
                    }]}}]}
        finish = step.finish_reason or ("tool_calls" if step.tool_calls else "stop")
        yield {"choices": [{"delta": {}, "finish_reason": finish}]}
        yield {"choices": [], "usage": step.usage or {
            "prompt_tokens": 10, "completion_tokens": 5}}


# ── the pool ────────────────────────────────────────────────────────────────

Answer = Any  # str | BaseException | Callable[[str, dict], str | dict]


class ReplayPool:
    """Scripted MCP pool.

    `tools` maps a tool name to its `readOnlyHint` (or to a full tool dict,
    for a tool that declares its own schema). `answers` maps a call id or a
    tool name to what `call_tool` returns: a string, a `{"content",
    "is_error"}` dict, an exception instance to raise, or a callable of
    `(name, args)` returning either.
    """

    def __init__(self, tools: dict[str, Any] | None = None, *,
                 answers: dict[str, Answer] | None = None,
                 delay_by_call_id: dict[str, float] | None = None,
                 server: str = "lloyd-mcp"):
        tools = tools if tools is not None else {
            "Read": True, "Grep": True, "Glob": True,
            "Bash": False, "Edit": False, "Write": False,
        }
        self._discovered = [(server, [self._tool(n, spec)
                                      for n, spec in tools.items()])]
        self.answers = dict(answers or {})
        self.delay_by_call_id = dict(delay_by_call_id or {})
        self.started: list[str] = []
        self.completed: list[str] = []
        self.cancelled: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.inflight = 0
        self.max_inflight = 0
        # Anything a test wants interleaved with dispatch (hook fires, etc.)
        # appends here too, so one list shows the global order.
        self.timeline: list[str] = []

    @staticmethod
    def _tool(name: str, spec: Any) -> dict[str, Any]:
        if isinstance(spec, dict):
            return {"name": name, "description": "", **spec}
        return {"name": name, "description": "",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {"readOnlyHint": bool(spec)}}

    @property
    def discovered(self):
        return self._discovered

    async def call_tool(self, name: str, args: dict[str, Any], **kw: Any):
        call_id = kw.get("call_id") or name
        self.calls.append({"name": name, "args": dict(args), **kw})
        self.started.append(call_id)
        self.timeline.append(f"start:{call_id}")
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay_by_call_id.get(call_id, 0))
        except asyncio.CancelledError:
            self.cancelled.append(call_id)
            self.timeline.append(f"cancel:{call_id}")
            raise
        finally:
            self.inflight -= 1
        answer = self.answers.get(call_id, self.answers.get(name, f"RESULT[{name}]"))
        if callable(answer) and not isinstance(answer, BaseException):
            answer = answer(name, args)
        self.completed.append(call_id)
        self.timeline.append(f"done:{call_id}")
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, dict):
            return answer
        return {"content": answer, "is_error": False}


# ── wiring ──────────────────────────────────────────────────────────────────

def install(monkeypatch, engine: ReplayEngine, pool: ReplayPool) -> None:
    """Replace the loop's two seams for the life of the test."""

    async def _ready(*_a, **_kw):
        return pool

    monkeypatch.setattr(L, "_build_pool", _ready)
    monkeypatch.setattr(L, "stream_chat", engine)


async def drive(options, messages: list[dict[str, Any]] | None = None,
                *, on_event: Callable[[dict], Any] | None = None) -> list[dict]:
    """Run one turn and return its events. `on_event` returning True stops
    consuming and closes the generator, the way a caller that breaks does."""
    msgs = messages if messages is not None else [{"role": "user", "content": "go"}]
    out: list[dict] = []
    agen = L.run_query(msgs, options)
    try:
        async for evt in agen:
            out.append(evt)
            if on_event is not None and on_event(evt):
                break
    finally:
        await agen.aclose()
    return out


def of_type(events: list[dict], kind: str) -> list[dict]:
    return [e for e in events if e.get("type") == kind]
