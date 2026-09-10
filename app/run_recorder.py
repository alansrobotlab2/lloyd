"""Persist a background run's turn, without owning it.

Three paths run an agent loop in this system and, until this module, only one
of them left anything behind:

| path                     | transcript | event log | session |
|--------------------------|-----------|-----------|---------|
| `autonomy.run_task`      | no        | no        | no      |
| `_common.run_prompt_on_primary` | no | no       | no      |
| `_common.run_prompt_in_session` | yes | yes      | yes     |

The first two call `run_query` directly and collect the text, so the record of
what a scheduled task actually *did* was a 200-character summary in an
`autonomy-runs/*.md` file. On 2026-09-10 an autonomy task was the prime
suspect in a full vault wipe and there was no way to confirm or clear it: the
tool calls it made had never been written down anywhere.

The obvious fix — route those paths through `/api/message/stream` like the
session-backed workers do — is the wrong one, and the sandbox branch that
tried it says why. `run_task` installs the #534 authority-grant gate on its
own `HookRegistry`, and the chat endpoint installs Inner Voice, the
destructive-Bash safety hook and skill dispatch but *not* that gate. Going
over the wire would have dropped it, and would have had to re-plumb
`saw_tool_call`, `tool_errors` and the timeout partial across HTTP to get back
what a direct `async for` already has.

So this is a **passthrough**: an async generator that persists each event and
re-yields it unchanged. The caller's loop is untouched, its options are
untouched, its hooks are untouched. Recording is a second reader of the
stream, not a new owner of it.

Two properties are load-bearing:

  * **Incremental, not end-of-run.** A run killed at its deadline — which is
    how the interesting ones end — must still leave the transcript up to the
    moment it died. Nothing is buffered until the `result` event that a killed
    run never emits.

  * **Recording may never break the run.** Every persistence step is wrapped:
    a full disk, a permissions error or a bug in here costs the *record*, and
    the run continues. An edit with no diagnostics beats an edit that failed
    because the linter did.

Entry shapes come from `app/transcript_entries.py`, which `app/routers/
messages.py` also calls, so the transcript a background run writes is the same
transcript the chat path writes and the Inner Voice reader can render either.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any, AsyncIterator

from app import event_log as _event_log
from app import prefix_miss as _prefix_miss
from app.sessions_io import _append_messages
from app.transcript_entries import (
    build_assistant_text_entry,
    build_thinking_entry,
    build_tool_call,
    build_tool_call_entry,
    build_tool_result_entry,
    build_user_entry,
    truncate_tool_result,
)

logger = logging.getLogger("lloyd-server")


def recording_enabled() -> bool:
    """`harness.background_recording.enabled`, defaulting to on.

    Read per call rather than cached at import: the kill switch is worth
    having only if flipping it takes effect on the next run.
    """
    try:
        from app.config import CONFIG
        block = (CONFIG.get("harness") or {}).get("background_recording") or {}
        return bool(block.get("enabled", True))
    except Exception:
        return True


class _RunRecorder:
    """Mirrors `_run_turn`'s persistence, minus everything that is about SSE.

    The state here is deliberately the same shape as the router's — a text
    accumulator, a thinking accumulator, a tool-call log and the set of pairs
    already on disk — because the two write the same rows and a divergence in
    the bookkeeping is a divergence in the transcript.
    """

    def __init__(self, session_id: str, turn_id: str, *, model: str = "",
                 source: str = "") -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.model = model
        self.source = source
        self.text = ""
        self.thinking = ""
        self.thinking_ms = 0
        self.thinking_seq = 0
        self.iteration = 0
        self.iteration_stats: dict[str, Any] = {}
        self.tool_calls: list[dict] = []
        self.results_by_id: dict[str, str] = {}
        self.persisted_pairs: set[str] = set()
        self.saw_result = False
        self.stop_reason: str | None = None
        # Prefix-cache misses, folded exactly as the chat path folds them
        # (app/prefix_miss.py) — one definition, two writers.
        self.miss = _prefix_miss.TurnMissTracker.for_turn(
            session_id, turn_id, label=source)

    # -- helpers ---------------------------------------------------------
    async def _append(self, entries: list[dict]) -> None:
        if not entries:
            return
        try:
            await _append_messages(self.session_id, entries)
        except Exception as exc:      # noqa: BLE001 — recording is not the run
            logger.warning("run_recorder: append failed for %s: %s",
                           self.session_id, exc)

    def _log(self, event: str, data: dict) -> None:
        try:
            _event_log.log_event(self.session_id, event, data,
                                 turn_id=self.turn_id)
        except Exception as exc:      # noqa: BLE001
            logger.warning("run_recorder: log_event failed for %s: %s",
                           self.session_id, exc)

    def _record_usage(self, stats: dict) -> None:
        """A usage row for a direct background run, as every chat turn has.

        Until prefix-miss accounting these two paths wrote no usage row at
        all, so `usage.db` — and the dashboard's token panel — never saw an
        autonomy task or a `run_prompt_on_primary` job, and a per-turn miss
        count would have had nowhere to live for one of the three paths that
        run an agent loop.
        """
        if not (stats.get("input_tokens") or stats.get("output_tokens")):
            return
        try:
            import usage_store
            usage_store.record_usage(
                session_id=self.session_id,
                model=self.model or "primary",
                input_tokens=int(stats.get("input_tokens") or 0),
                output_tokens=int(stats.get("output_tokens") or 0),
                cache_create=int(stats.get("cache_create") or 0),
                cache_read=int(stats.get("cache_read") or 0),
                cost_usd=0.0,
                duration_ms=stats.get("duration_ms"),
                num_turns=stats.get("num_turns"),
                reprefill_tokens=stats.get("reprefill_tokens"),
                prefix_misses=stats.get("prefix_misses"),
            )
        except Exception as exc:      # noqa: BLE001 — accounting is not the run
            logger.warning("run_recorder: usage row failed for %s: %s",
                           self.session_id, exc)

    # -- lifecycle -------------------------------------------------------
    async def open(self, prompt: str) -> None:
        if prompt:
            await self._append([build_user_entry(
                prompt, timestamp=datetime.now().isoformat(),
                source=self.source or "background")])
        self._log("brain1.user_prompt_received", {
            "prompt": prompt, "prompt_chars": len(prompt),
            "source": self.source,
        })
        self._log("brain1.query_started", {
            "model": self.model, "prompt_chars": len(prompt),
            "background": True, "source": self.source,
        })

    async def handle(self, evt: dict) -> None:
        etype = evt.get("type")
        if etype == "text_delta":
            self.text += evt.get("text", "") or ""

        elif etype == "thinking_delta":
            self.thinking += evt.get("text", "") or ""

        elif etype == "thinking_done":
            text = evt.get("text", "") or ""
            self.thinking = text
            self.thinking_ms = int(evt.get("duration_ms") or 0)
            self._log("brain1.thinking_block_emitted", {
                "thinking": text, "chars": len(text),
                "duration_ms": self.thinking_ms,
            })
            if text:
                await self._append([build_thinking_entry(
                    self.turn_id, text, self.thinking_ms, self.thinking_seq,
                    self.iteration + 1, datetime.now().isoformat(),
                )])
                self.thinking_seq += 1
                # On disk now; what accumulates after this is an unflushed
                # partial, which is exactly what the interrupted path saves.
                self.thinking = ""
                self.thinking_ms = 0

        elif etype == "assistant_message":
            self.iteration = evt.get("iteration") or (self.iteration + 1)
            usage = evt.get("usage") or {}
            self.iteration_stats = {
                "input_tokens": usage.get("input_tokens")
                    or usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("output_tokens")
                    or usage.get("completion_tokens", 0),
                "cache_read": usage.get("cache_read", 0)
                    or usage.get("prompt_tokens_cached", 0) or 0,
                "cache_create": usage.get("cache_create", 0) or 0,
                "duration_ms": evt.get("duration_ms", 0),
                "iteration": evt.get("iteration", 0),
                "model": self.model,
            }
            _prefix_miss.record_iteration(
                self.miss, int(evt.get("iteration") or self.iteration), usage,
                duration_ms=int(evt.get("duration_ms") or 0), log=self._log)
            if evt.get("tool_calls") and self.text.strip():
                await self._append([build_assistant_text_entry(
                    self.text, timestamp=datetime.now().isoformat(),
                    stats=dict(self.iteration_stats),
                    reasoning=self.thinking, reasoning_ms=self.thinking_ms,
                )])
                self.text = ""
                self.thinking = ""
                self.thinking_ms = 0

        elif etype == "tool_call":
            call_id = evt.get("call_id", "")
            tc = build_tool_call(call_id, evt.get("name", ""),
                                 evt.get("args_json", "{}"),
                                 evt.get("summary") or "")
            self.tool_calls.append(tc)
            self._log("brain1.tool_call_proposed", {
                "tool_call_id": call_id, "name": evt.get("name", ""),
                "args": evt.get("args_json", "{}"),
                "summary": evt.get("summary") or "",
            })

        elif etype == "tool_result":
            call_id = evt.get("call_id", "")
            result = truncate_tool_result(str(evt.get("content", "") or ""))
            self.results_by_id[call_id] = result
            self._log("brain1.tool_result_received", {
                "tool_call_id": call_id, "result": result,
                "result_chars": len(result),
                "is_error": bool(evt.get("is_error", False)),
            })
            tc = next((t for t in self.tool_calls
                       if t["call_id"] == call_id), None)
            if tc and call_id not in self.persisted_pairs:
                self.persisted_pairs.add(call_id)
                ts = datetime.now().isoformat()
                await self._append([
                    build_tool_call_entry(tc, timestamp=ts,
                                          stats=self.iteration_stats),
                    build_tool_result_entry(
                        call_id, result, timestamp=ts,
                        is_error=bool(evt.get("is_error", False))),
                ])

        elif etype == "result":
            self.saw_result = True
            self.stop_reason = evt.get("stop_reason", "stop")
            usage = evt.get("usage") or {}
            stats = {
                "input_tokens": usage.get("input_tokens")
                    or usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("output_tokens")
                    or usage.get("completion_tokens", 0),
                "cache_read": usage.get("cache_read", 0) or 0,
                "cache_create": usage.get("cache_create", 0) or 0,
                "duration_ms": evt.get("duration_ms", 0),
                "num_turns": evt.get("num_turns", 0),
                "model": self.model,
                **_prefix_miss.finish(self.miss, log=self._log),
            }
            self._log("brain1.result_message", {
                "usage": {k: stats[k] for k in
                          ("input_tokens", "output_tokens",
                           "cache_read", "cache_create")},
                "stop_reason": self.stop_reason,
                "duration_ms": stats["duration_ms"],
                "num_turns": stats["num_turns"],
                "response_chars": len(self.text),
                "had_tool_calls": bool(self.tool_calls),
            })
            self._record_usage(stats)
            tail = self._unpersisted_pairs(datetime.now().isoformat())
            final_text = self.text or str(evt.get("response_text") or "")
            if final_text.strip():
                tail.append(build_assistant_text_entry(
                    final_text, timestamp=datetime.now().isoformat(),
                    stats=stats, reasoning=self.thinking,
                    reasoning_ms=self.thinking_ms,
                    structured=evt.get("structured"),
                    cancelled=self.stop_reason == "cancelled",
                    source=self.source or "background",
                ))
            await self._append(tail)
            self.text = ""
            self.thinking = ""

    def _unpersisted_pairs(self, ts: str) -> list[dict]:
        """Tool calls whose result never came back, or came back after the
        pair had already been written by someone else. Same reconstruction
        the router does, and for the same reason: a call with no row is a
        call the transcript denies was ever made."""
        tail: list[dict] = []
        for tc in self.tool_calls:
            cid = tc["call_id"]
            if cid in self.persisted_pairs:
                continue
            self.persisted_pairs.add(cid)
            result = self.results_by_id.get(cid, "")
            tail.append(build_tool_call_entry(tc, timestamp=ts,
                                              stats=self.iteration_stats))
            tail.append(build_tool_result_entry(cid, result, timestamp=ts))
        return tail

    async def close_interrupted(self, reason: str) -> None:
        """The stream ended without a `result` event.

        A timeout, a cancellation, or an exception in the model client. This
        is the case the whole module exists for — the run that ends badly is
        the run someone will want to read — so whatever is in hand goes to
        disk, marked as the partial it is.
        """
        if self.saw_result:
            return
        ts = datetime.now().isoformat()
        tail = self._unpersisted_pairs(ts)
        if self.text.strip():
            tail.append(build_assistant_text_entry(
                self.text, timestamp=ts, reasoning=self.thinking,
                reasoning_ms=self.thinking_ms, cancelled=True,
                source=self.source or "background",
            ))
        await self._append(tail)
        self._log("background.run_interrupted", {
            "reason": reason,
            "response_chars": len(self.text),
            "tool_calls": len(self.tool_calls),
            "results": len(self.results_by_id),
        })


async def record_events(events: AsyncIterator[dict], *, session_id: str,
                        turn_id: str, prompt: str = "", model: str = "",
                        source: str = "") -> AsyncIterator[dict]:
    """Persist a background run's events and re-yield every one unchanged.

    Drop-in around an existing `run_query(...)` loop:

        async for evt in record_events(run_query(messages, options),
                                       session_id=sid, turn_id=run_id,
                                       prompt=prompt, source="autonomy"):
            ...

    The session must already exist (`sessions_io.create_session`), so the
    caller decides its platform, its title and whether Inner Voice observes
    it. Recording is not observation: it is cheap, universal and silent, and
    the observer is a separate opt-in that costs primary capacity.
    """
    if not recording_enabled():
        async for evt in events:
            yield evt
        return

    rec = _RunRecorder(session_id, turn_id, model=model, source=source)
    await rec.open(prompt)
    reason = "end-of-stream"
    try:
        async for evt in events:
            try:
                await rec.handle(evt)
            except Exception as exc:  # noqa: BLE001 — recording is not the run
                logger.warning("run_recorder: dropped a %s event for %s: %s",
                               evt.get("type"), session_id, exc)
            yield evt
    except BaseException as exc:      # noqa: BLE001 — including CancelledError
        reason = f"{type(exc).__name__}"
        raise
    finally:
        # Shielded: the common way a background run ends is its own deadline
        # cancelling this task, and an unshielded await here would be
        # cancelled before it wrote the partial transcript that is the entire
        # point. The shielded task finishes even though we stop waiting on it.
        with contextlib.suppress(BaseException):
            await asyncio.shield(
                asyncio.ensure_future(rec.close_interrupted(reason)))
