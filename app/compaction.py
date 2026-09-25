"""Conversation compaction — client-side context management.

The harness is stateless per request, so we reconstruct the conversation
from the persisted session JSON each turn, fit it to the model's
context window, and return ready-to-send OpenAI-format messages.

Three defensive layers, in order of cost (cheapest first):

  1. **Microcompaction** (:mod:`app.harness.microcompact`) — clears
     stale compactable-tool results inline. Pure structural pass, no
     LLM call. Runs every load.
  2. **LLM summarization** (:mod:`app.compaction_llm`) — when
     microcompact alone doesn't get under threshold, summarize the
     dropped block via a non-streaming POST to the local vLLM. Falls
     back to truncation on any failure so the user's turn always
     completes.
  3. **Truncation** — last resort. Drops oldest turns past
     ``TURNS_TO_KEEP``, prepends a ``[compaction: N tokens omitted]``
     marker. This is what the module did historically and remains the
     fallback when ``compaction.mode`` is ``truncate`` or when
     summarization fails.

Behind everything else, the harness loop has a reactive 413 recovery
path (``app.harness.loop._truncate_largest_tool_results``) that fires
if vLLM still rejects the prompt. That's the safety net under all of
the above.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("lloyd-server")

# The rows the turn-start stack keeps from the session file. The persisted
# summary's indexes (app/compaction_state.py) are into this filtered list, so
# the two must agree; `tests/test_compaction_persisted_summary.py` pins it.
_CONVERSATION_ROLES = ("user", "assistant", "tool", "system")

# P3 (review 2026-09-24, app/memory_flush.py): every row a memory-flush turn
# writes carries a `turn_id` with this prefix (X3), because the flush turn's
# SessionTurn is minted with it. Those rows stay in the transcript and never
# re-enter the prompt or the summary: the flush is bookkeeping the model did
# for itself, not conversation.
FLUSH_TURN_PREFIX = "mflush-"


def is_flush_row(m: dict) -> bool:
    """True for a row written by a memory-flush turn."""
    tid = m.get("turn_id") if isinstance(m, dict) else None
    return isinstance(tid, str) and tid.startswith(FLUSH_TURN_PREFIX)


def is_history_row(m: dict) -> bool:
    """The one definition of a row the turn-start stack keeps.

    `app.compaction_state.conversation_rows` reads it too: the persisted
    summary's indexes are into this filtered list, and `save_record`
    re-validates against it, so the two filters must be the same function.
    """
    return (isinstance(m, dict) and m.get("role") in _CONVERSATION_ROLES
            and not is_flush_row(m))

# The roles `app.routers._messages_harness_adapter._prepare_messages_for_harness`
# forwards to the engine. `tokens_after` counts only these.
_SENT_ROLES = ("user", "assistant", "tool")


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

# Rough English-text heuristic: 1 token ≈ 4 chars. Integer arithmetic only.
TOKENS_PER_CHAR = 4

# Output headroom — reserve this many tokens for the model's response.
OUTPUT_TOKENS_RESERVED = 20_000

# Additional safety buffer so we start truncating before hitting the wall.
TRUNCATION_BUFFER_TOKENS = 32_000

# Default context window if a model has no configured `context_length`.
DEFAULT_CONTEXT_WINDOW = 128_000

# Number of most-recent turns (user+assistant pairs) to keep when
# truncation fires.
TURNS_TO_KEEP = 20


def estimate_tokens(text: str) -> int:
    """Rough token count for a string. Integer result."""
    if not text:
        return 0
    return max(1, len(text) // TOKENS_PER_CHAR)


def _message_text(message: dict) -> str:
    """Flatten a message's content to a single string for token counting.

    Handles both string content and structured content (list of blocks).
    Tool-use / tool-result / thinking blocks are serialized in full — no
    silent truncation, since they're often the informationally dense
    parts of a turn.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type", "")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "")
            inp = json.dumps(block.get("input", {}))
            parts.append(f"[tool_call: {name}({inp})]")
        elif btype == "tool_result":
            result = block.get("content", "")
            if isinstance(result, str):
                parts.append(f"[tool_result: {result}]")
            elif isinstance(result, list):
                for b in result:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(f"[tool_result: {b.get('text', '')}]")
                    elif isinstance(b, dict):
                        parts.append(f"[tool_result: {json.dumps(b)}]")
        elif btype == "thinking":
            parts.append(f"[thinking: {block.get('thinking', '')}]")
        elif btype == "image":
            parts.append("[image]")
        else:
            parts.append(json.dumps(block))
    return "\n".join(p for p in parts if p)


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for a single message dict.

    Harness assistant messages may carry a `reasoning` field (preserved
    thinking — see app/harness/loop.py::_prune_reasoning). It is rendered
    into the prompt's `<think>` block, so it costs real context and is
    counted here. Session-shape messages never carry it, so this is a
    no-op for every other caller.
    """
    if not message:
        return 0
    text = _message_text(message)
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        text = f"{text}\n[thinking: {reasoning}]"
    total = estimate_tokens(text)
    # Screenshots riding on a harness tool message (tool_images). A flat
    # per-image estimate: the ViT cost depends on pixels, not on bytes.
    refs = message.get("_image_refs")
    if refs:
        try:
            from app.harness.tool_images import image_token_estimate
            total += image_token_estimate() * len(refs)
        except Exception:
            total += 1500 * len(refs)
    return total


def estimate_conversation_tokens(messages: list[dict], system_prompt: str = "") -> int:
    """Estimate total tokens for a conversation.

    Counts the literal bytes of `system_prompt` + each message. Does NOT
    add any opaque padding constants — the caller is responsible for
    passing the real system prompt if it wants that accounted for.
    """
    total = estimate_tokens(system_prompt)
    total += sum(estimate_message_tokens(m) for m in messages)
    return total


# ---------------------------------------------------------------------------
# Per-model context window
# ---------------------------------------------------------------------------


def get_context_window(model: str) -> int:
    """Return the configured context length for a model, or the default.

    Reads `models.<name>.context_length` from config.yaml. Imports lazily
    so this module stays importable without a full app bootstrap (tests
    can monkey-patch `_get_model_cfg` if needed).
    """
    try:
        from app.config import _get_model_cfg  # lazy — avoids circular deps at import
    except Exception:
        return DEFAULT_CONTEXT_WINDOW
    cfg = _get_model_cfg(model) if model else {}
    ctx = cfg.get("context_length")
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    return DEFAULT_CONTEXT_WINDOW


def truncation_threshold(context_window: int) -> int:
    """Token budget above which truncation should fire."""
    return max(1_000, context_window - OUTPUT_TOKENS_RESERVED - TRUNCATION_BUFFER_TOKENS)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def _group_into_turns(messages: list[dict]) -> list[list[dict]]:
    """Group a message list into turns.

    A turn = one user message optionally followed by assistant/tool
    messages, up to the next user message. The very first segment
    (messages before any user msg) is treated as its own "turn" so the
    system-initial content isn't lost.
    """
    turns: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "user" and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)
    return turns


def _flatten_turns(turns: list[list[dict]]) -> list[dict]:
    """Flatten a turn-grouped message list back into a flat list."""
    out: list[dict] = []
    for t in turns:
        out.extend(t)
    return out


def _split_for_summary(
    messages: list[dict],
    keep_recent_turns: int,
) -> tuple[list[dict], list[dict]]:
    """Partition ``messages`` into (older, recent) along turn boundaries.

    The ``recent`` block — last ``keep_recent_turns`` turns — is kept
    verbatim past the summary boundary. The ``older`` block is what
    gets summarized.
    """
    turns = _group_into_turns(messages)
    if len(turns) <= keep_recent_turns:
        return [], _flatten_turns(turns)
    cutoff = len(turns) - keep_recent_turns
    return _flatten_turns(turns[:cutoff]), _flatten_turns(turns[cutoff:])


def truncate_conversation(
    messages: list[dict],
    max_tokens: int,
    turns_to_keep: int = TURNS_TO_KEEP,
    system_prompt: str = "",
) -> tuple[list[dict], int]:
    """Truncate conversation to fit within a token budget.

    Returns (truncated_messages, tokens_dropped). If no truncation is
    needed, returns (messages, 0).

    Strategy: keep the last `turns_to_keep` turns. If that's still too
    big, drop turns from the front until we fit. Always keeps at least
    the final turn. When anything is dropped, prepends one synthetic
    user message describing what was removed, so the model isn't
    confused by mid-conversation jumps.
    """
    current_tokens = estimate_conversation_tokens(messages, system_prompt)
    if current_tokens <= max_tokens:
        return messages, 0

    turns = _group_into_turns(messages)
    if not turns:
        return messages, 0

    # Start by keeping the last N turns.
    kept = turns[-turns_to_keep:] if len(turns) > turns_to_keep else turns[:]

    # Drop oldest turns until under budget, but never drop the last turn.
    while len(kept) > 1 and estimate_conversation_tokens(_flatten_turns(kept), system_prompt) > max_tokens:
        kept.pop(0)

    truncated = _flatten_turns(kept)
    dropped_tokens = current_tokens - estimate_conversation_tokens(truncated, system_prompt)

    if dropped_tokens > 0:
        note = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"[compaction: {dropped_tokens} tokens of earlier conversation "
                        f"omitted to fit the context window. {len(turns) - len(kept)} "
                        f"turns were dropped.]"
                    ),
                }
            ],
        }
        truncated = [note] + truncated

    return truncated, dropped_tokens


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _compaction_cfg() -> dict[str, Any]:
    """Read the ``compaction:`` block from config.yaml with defaults.

    Called per invocation so config edits take effect on the next
    turn (no restart needed for tuning).
    """
    try:
        from app.config import CONFIG  # type: ignore
    except Exception:
        return {}
    cfg = dict(CONFIG.get("compaction") or {})
    cfg.setdefault("mode", "summarize")          # summarize | truncate
    cfg.setdefault("summary_model", None)        # None → falls back to default model
    cfg.setdefault("keep_recent_turns", 5)
    # D2 (review 2026-09-24): the persisted, incremental summary
    # (app/compaction_state.py). Off = a stored record is ignored and the
    # summarize layer regenerates from scratch as it always did.
    cfg.setdefault("persist_summary", False)
    cfg.setdefault("summary_input_budget_tokens", 48_000)
    cfg.setdefault("max_folds_per_turn", 3)
    micro =dict(cfg.get("microcompact") or {})
    micro.setdefault("enabled", True)
    micro.setdefault("keep_recent_tools", 15)
    micro.setdefault("count_threshold", 20)
    micro.setdefault("compactable_tools", None)  # None → use module default
    # Deny-list mode (D10). Set → every tool's result may be cleared except
    # these, and `compactable_tools` is ignored. None → the allow-list above.
    micro.setdefault("non_compactable_tools", None)
    # Budget gating. Microcompaction gets first refusal above
    # `trigger_fraction` of the truncation threshold — it is cheaper than
    # summarization and preserves more — and clears down to
    # `target_fraction` so it does not have to run again next turn.
    # Below the trigger it does nothing at all.
    micro.setdefault("trigger_fraction", 0.8)
    micro.setdefault("target_fraction", 0.6)
    micro.setdefault("min_chars_to_clear", 2_000)
    cfg["microcompact"] = micro
    restore = dict(cfg.get("restore") or {})
    restore.setdefault("enabled", True)
    restore.setdefault("budget_tokens", 50_000)
    restore.setdefault("max_per_file", 5_000)
    restore.setdefault("max_files", 5)
    cfg["restore"] = restore
    manual = dict(cfg.get("manual") or {})
    manual.setdefault("buffer_tokens", 3_000)
    # D11: folds one `/compact` may run (app/compaction_state.manual_compact).
    manual.setdefault("max_folds", 12)
    cfg["manual"] = manual
    return cfg


# ---------------------------------------------------------------------------
# Session-level helper — read JSON, truncate, return ready-to-send history
# ---------------------------------------------------------------------------


def _record_turn_start(session_id: str, result: dict[str, Any]) -> None:
    """Emit one event per turn-start REWRITE (#1078).

    Emitted exactly when the caller's `[compaction]` log line is emitted — same
    condition, mechanisms non-empty — so the two can be checked against each
    other and a reader who finds one and not the other has found a bug. Events
    are for rewrites only; the decision to leave a history alone is recorded on
    the turn's usage row instead, because "did this path run at all" is a
    question answered by counting rows, and every chat turn writes one.

    The session id is the session file's stem, which is why this is emitted here
    rather than by the caller: `app/routers/voice.py` calls this function on a
    turn that books no usage row at all, and the event log is the one store where
    a voice turn's rewrite still lands with an owner. The manual `/compact` turn
    does not reach here either; it records itself as `compaction.manual`
    (`app/routers/messages.py::_run_compact_turn`, D11).
    """
    from app.compaction_record import turn_start_record

    try:
        record = turn_start_record(result)
        if not record["mechanisms"]:
            return
        from app.event_log import log_event
        log_event(session_id, "compaction.turn_start", record)
    except Exception as e:  # noqa: BLE001
        logger.warning("compaction: turn_start event write failed: %s", e)


async def _persisted_summary_layer(
    convo: list[dict],
    data: dict,
    *,
    path: Path,
    cfg: dict[str, Any],
    model: str,
    threshold: int,
    system_prompt: str,
    cur_tokens: int,
    allow_fold: bool = True,
) -> dict[str, Any]:
    """Layer A under ``compaction.persist_summary`` (D2), or for a session a
    person compacted by hand (D11, ``compaction_state.is_manual``).

    A stored record that validates is applied whatever the size — the same
    summary text every turn is what keeps the prefix cached. The threshold
    decides only whether the rows past its boundary are folded in, and only
    when ``allow_fold`` (summarize mode): a manual record is honoured in
    truncate mode too, but that mode never calls a summariser. See
    `app/compaction_state.py` for the record and the fold.
    """
    out: dict[str, Any] = {
        "convo": convo, "summarized": False, "attempted": False,
        "outcome": "under_threshold", "restored": 0, "reused": False,
        "folds": 0, "covered_rows": 0, "head": 0,
    }
    session_id = path.stem
    try:
        from app import compaction_state as CS
        from app.harness.telemetry import log_harness_event
    except Exception as e:  # noqa: BLE001
        logger.warning("compaction_state import failed: %s", e)
        if cur_tokens > threshold:
            out.update(attempted=True, outcome="import_failed")
        return out

    record = CS.load_record(data)
    covered: list[dict] = []
    remaining = convo
    if record is not None:
        applied = CS.apply(convo, record)
        if applied is None:
            # The past was rewritten under the record (a legacy `/compact`, a
            # hand edit). Its summary may describe rows that are gone, so it is
            # discarded and rebuilt from the rows as they are now.
            log_harness_event(session_id, "compaction.record_invalidated", {
                "covers_through_index": record.get("covers_through_index"),
                "covered_rows": record.get("covered_rows"),
                "rows_now": len(convo),
            })
            record = None
        else:
            summary_msg, covered, remaining = applied
            out["reused"] = True
            cur_tokens = estimate_conversation_tokens(
                [summary_msg] + remaining, system_prompt)

    if cur_tokens > threshold and not allow_fold:
        out["outcome"] = "mode_truncate"
    elif cur_tokens > threshold:
        out["attempted"] = True
        older, _recent = _split_for_summary(remaining, int(cfg.get("keep_recent_turns", 5)))
        if not older:
            out["outcome"] = "no_older_block"
        else:
            start = len(covered)
            try:
                res = await CS.fold(
                    convo, start=start, end=start + len(older), record=record,
                    session_id=session_id, path=path,
                    model=cfg.get("summary_model") or model or "", cfg=cfg,
                    max_folds=int(cfg.get("max_folds_per_turn", 3)),
                )
            except ImportError as e:
                logger.warning("compaction_llm import failed: %s", e)
                res = {"folds": 0, "record": record, "import_failed": True}
            except Exception as e:  # noqa: BLE001 — a failed fold keeps the prior record
                logger.warning("compaction: fold failed for %s: %s", session_id, e)
                res = {"folds": 0, "record": record}
            if res.get("folds"):
                record = res["record"]
                out["folds"] = int(res["folds"])
                out["outcome"] = "summarized"
            else:
                out["outcome"] = "import_failed" if res.get("import_failed") else "empty_summary"
    elif out["reused"]:
        out["outcome"] = "reused"

    if record is None:
        return out
    boundary = int(record["covered_rows"])
    covered, remaining = convo[:boundary], convo[boundary:]
    new_convo: list[dict] = [CS.summary_message(record)]
    # Layer C, over what the summary covers. Re-read from disk, so it is as
    # stable as the files are.
    if cfg["restore"].get("enabled", True):
        try:
            from app.compaction_llm import restore_recent_files, restored_file_count
            restored = restore_recent_files(
                covered,
                budget_tokens=int(cfg["restore"].get("budget_tokens", 50_000)),
                max_per_file=int(cfg["restore"].get("max_per_file", 5_000)),
                max_files=int(cfg["restore"].get("max_files", 5)),
            )
            new_convo.extend(restored)
            out["restored"] = restored_file_count(restored)
        except Exception as e:  # noqa: BLE001
            logger.warning("restore_recent_files failed: %s", e)
    head = len(new_convo)
    new_convo.extend(remaining)
    out.update(convo=new_convo, summarized=True, covered_rows=boundary, head=head)
    return out


def _flushed_this_cycle(data: dict) -> bool:
    try:
        from app import memory_flush
        return memory_flush.flushed_this_cycle(data)
    except Exception:  # noqa: BLE001 — accounting never breaks a turn
        return False


async def load_and_compact_session(
    session_path: Path | str,
    model: str = "",
    system_prompt: str = "",
    *,
    mode_override: str | None = None,
    disallowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Read a persisted session, apply the compaction stack, and return
    ready-to-send history plus metadata.

    Layer order (cheapest first):

      1. Load + filter to conversation roles.
      2. Microcompact pre-pass (clears stale tool results).
         ``disallowed_tools`` is the reading turn's deny list, threaded to the
         markers this pass rewrites: a worker's second turn reads its own first
         turn's markers out of the session file, and a marker that says ``Read
         that path`` to a turn whose policy refuses ``Read`` is an instruction
         the history itself is issuing (#1066). Unset reads as
         everything-allowed, which is the chat case and the safe direction.
      3. If still over threshold and ``mode == "summarize"``:
         try LLM summarization; on success, replace dropped block with
         the summary and re-inject recent files (Layer C).
      4. If still over threshold (or summary failed, or mode is
         ``"truncate"``): fall through to the historical drop-oldest
         truncation path.

    Returns a dict with:
      - history:        list[dict], possibly compacted
      - tokens_before:  int, estimated tokens in the original history
      - tokens_after:   int, estimated tokens after compaction
      - truncated:      bool, True if any turns were dropped (compat-name)
      - summarized:     bool, True if LLM summarization replaced a block
      - microcompacted: int, count of tool results cleared by the pre-pass
      - restored_files: int, count of files re-injected post-summary
      - context_window: int, the model's configured window
      - threshold:      int, the compaction threshold in use
      - summarize_attempted: bool, True if the summarize layer was reached
        (history over threshold and ``mode == "summarize"``), regardless of
        whether it summarized
      - summarize_outcome: str, one of ``no_history``, ``under_threshold``,
        ``mode_truncate``, ``import_failed``, ``no_older_block``,
        ``empty_summary``, ``summarized`` — which of those paths this turn took;
        with ``compaction.persist_summary`` also ``reused`` (a stored summary
        was applied and nothing needed folding)
      - summary_reused: bool, a stored summary record was applied (D2)
      - summary_folds:  int, folds this turn added to the record
      - summary_covered_rows: int, rows the applied record covers

    On any error (missing file, malformed JSON), returns an empty result
    with ``history=[]`` and logs a warning. Summarization failures are
    handled transparently — the user's turn always completes.
    """
    cfg = _compaction_cfg()
    context_window = get_context_window(model)
    threshold = truncation_threshold(context_window)

    # `summarize_attempted` / `summarize_outcome` are the fields that make the
    # dead-configuration question decidable (#1078). Until now the only trace of
    # this stack was one log line emitted on a rewrite, so "the summarize layer
    # ran and declined" and "the turn-start path was never reached" printed
    # exactly the same nothing. `no_history` is the value for a stack that ran
    # against an unreadable or empty session: reached, and nothing to do.
    empty: dict[str, Any] = {
        "history": [],
        "tokens_before": 0,
        "tokens_after": 0,
        "truncated": False,
        "summarized": False,
        "microcompacted": 0,
        "restored_files": 0,
        "context_window": context_window,
        "threshold": threshold,
        "summarize_attempted": False,
        "summarize_outcome": "no_history",
    }

    try:
        path = Path(session_path) if not isinstance(session_path, Path) else session_path
        if not path.exists():
            return empty
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning("compaction: failed to read session %s: %s", session_path, e)
        return empty

    messages = data.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return empty

    # Drop non-conversation entries (subliminal, tombstones, ambient markers)
    # since they're for UI display, not for the model.
    # P3: and the rows a memory-flush turn wrote (`is_history_row`).
    convo = [m for m in messages if is_history_row(m)]
    # P1, off by default: replay each past user turn as what was sent (its
    # subliminal prefix and tail re-joined), not the bare text.
    from app import prompt_layout as _prompt_layout
    if _prompt_layout.replay_injected_context():
        convo = _prompt_layout.replay_injected(convo, messages)

    tokens_before = estimate_conversation_tokens(convo, system_prompt)

    # ---- Layer B: microcompaction pre-pass ----------------------------
    micro_cleared = 0
    if cfg["microcompact"].get("enabled", True):
        # Lazy import — keeps app.compaction importable without the
        # harness module tree (e.g. for unit tests of truncation alone).
        try:
            from app.harness.microcompact import (
                DEFAULT_COMPACTABLE_TOOLS,
                microcompact,
            )
            mc_cfg = cfg["microcompact"]
            tools: Iterable[str] = (
                mc_cfg.get("compactable_tools") or DEFAULT_COMPACTABLE_TOOLS
            )
            # Budget the pass against the same threshold Layer A uses.
            # Trigger high enough that a conversation with room to spare
            # is left completely alone; clear down to a lower target so
            # the next turn does not immediately re-trigger.
            trigger = int(threshold * float(mc_cfg.get("trigger_fraction", 0.8)))
            target = int(threshold * float(mc_cfg.get("target_fraction", 0.6)))
            # `session_id` is the session file's stem — the same key the
            # spill module names its directory after. Without it,
            # spill-before-clear is skipped and nothing is cleared that
            # was not already on disk.
            convo, micro_cleared = microcompact(
                convo,
                keep_recent_tools=int(mc_cfg.get("keep_recent_tools", 15)),
                count_threshold=int(mc_cfg.get("count_threshold", 20)),
                compactable_tools=tools,
                token_budget=target if tokens_before > trigger else None,
                # Never the count rule: below the trigger this pass must
                # do nothing beyond dropping previews already on disk.
                legacy_count_rule=False,
                estimate_fn=(
                    lambda msgs: estimate_conversation_tokens(msgs, system_prompt)
                ),
                min_chars_to_clear=int(mc_cfg.get("min_chars_to_clear", 2_000)),
                session_id=path.stem,
                disallowed_tools=disallowed_tools,
                # Deny mode wins when configured; None keeps `tools`.
                non_compactable_tools=mc_cfg.get("non_compactable_tools"),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("microcompact pre-pass failed: %s", e)

    # If microcompact alone got us under threshold, we're done.
    cur_tokens = estimate_conversation_tokens(convo, system_prompt)
    summarized = False
    restored_count = 0

    # ---- Layer A: LLM summarization -----------------------------------
    mode = (mode_override or cfg.get("mode") or "summarize").lower()
    # Four outcomes are possible for the whole layer, and each is stored rather
    # than inferred: `under_threshold` (never entered), `mode_truncate` (never
    # entered, because configuration says not to), `import_failed` /
    # `no_older_block` / `empty_summary` / `summarized` (entered). Only the last
    # one has ever left a trace. `summarize_attempted` is the entered-set, so a
    # reader can count "turns where the summarize layer had its chance" without
    # knowing this function's branch names.
    summarize_attempted = False
    summarize_outcome = "under_threshold"
    # D2: the persisted summary. Summarize mode only — `mode: truncate` (and
    # voice's override) says drop-oldest, and a stored summary is a summary.
    # D11: except a record a person asked for with `/compact`, which is applied
    # whatever the flag or the mode says. `/compact` no longer rewrites the
    # messages, so that record is all it leaves; ignoring it would make the
    # command a no-op. It is never folded in truncate mode.
    persist = bool(cfg.get("persist_summary")) and mode == "summarize"
    if not persist:
        try:
            from app.compaction_state import is_manual, load_record
            persist = is_manual(load_record(data))
        except Exception as e:  # noqa: BLE001
            logger.warning("compaction_state import failed: %s", e)
    summary_reused = False
    summary_folds = 0
    summary_covered_rows = 0
    summary_head = 0
    if persist:
        layer = await _persisted_summary_layer(
            convo, data, path=path, cfg=cfg, model=model,
            threshold=threshold, system_prompt=system_prompt, cur_tokens=cur_tokens,
            allow_fold=(mode == "summarize"),
        )
        convo = layer["convo"]
        summarized = layer["summarized"]
        summarize_attempted = layer["attempted"]
        summarize_outcome = layer["outcome"]
        restored_count = layer["restored"]
        summary_reused = layer["reused"]
        summary_folds = layer["folds"]
        summary_covered_rows = layer["covered_rows"]
        summary_head = layer["head"]
        cur_tokens = estimate_conversation_tokens(convo, system_prompt)
    elif cur_tokens > threshold and mode == "summarize":
        summarize_attempted = True
        summarize_outcome = "import_failed"
        try:
            from app.compaction_llm import (
                restore_recent_files,
                restored_file_count,
                summarize_history,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("compaction_llm import failed, falling back to truncate: %s", e)
            summarize_history = None  # type: ignore[assignment]
            restore_recent_files = None  # type: ignore[assignment]

        if summarize_history is not None:
            keep_recent_turns = int(cfg.get("keep_recent_turns", 5))
            older, recent = _split_for_summary(convo, keep_recent_turns)
            # Pre-seeded for the branch about to be taken, overwritten only on
            # the way out that actually summarized: a falsy summary and an
            # unsplittable history are two different reasons for
            # `summarized: False`, and both used to be silent.
            summarize_outcome = "no_older_block" if not older else "empty_summary"
            if older:
                summary = await summarize_history(
                    older,
                    model=cfg.get("summary_model") or model or None,
                )
                if summary:
                    summarized = True
                    summarize_outcome = "summarized"
                    summary_msg = {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[compaction summary — earlier conversation "
                                    "summarized to fit context window]\n\n"
                                    + summary
                                ),
                            }
                        ],
                    }
                    new_convo: list[dict] = [summary_msg]
                    # ---- Layer C: post-compact restore ----------------
                    if cfg["restore"].get("enabled", True) and restore_recent_files is not None:
                        try:
                            restored = restore_recent_files(
                                older,
                                budget_tokens=int(cfg["restore"].get("budget_tokens", 50_000)),
                                max_per_file=int(cfg["restore"].get("max_per_file", 5_000)),
                                max_files=int(cfg["restore"].get("max_files", 5)),
                            )
                            # One `user` row (D3): a `system` row is dropped by
                            # the harness adapter and never reached the engine.
                            new_convo.extend(restored)
                            restored_count = restored_file_count(restored)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("restore_recent_files failed: %s", e)
                    new_convo.extend(recent)
                    convo = new_convo
                    cur_tokens = estimate_conversation_tokens(convo, system_prompt)
    elif cur_tokens > threshold:
        # Over the wall and the configuration says drop-oldest: the summarize
        # layer was not reached because of a setting, which is a different
        # finding from not reaching it because there was nothing to do.
        summarize_outcome = "mode_truncate"

    # ---- Layer (fallback): truncation ---------------------------------
    if summary_head:
        # D2: the persisted summary (and its restored files) is the head of
        # the history and is never what drop-oldest drops — a fold capped by
        # `max_folds_per_turn` leaves verbatim rows the next turn keeps folding,
        # and truncating those first keeps the summary that covers the rest.
        head = convo[:summary_head]
        tail, dropped = truncate_conversation(
            convo[summary_head:],
            max_tokens=max(1_000, threshold - estimate_conversation_tokens(head)),
            turns_to_keep=TURNS_TO_KEEP,
            system_prompt=system_prompt,
        )
        truncated_msgs = head + tail
    else:
        truncated_msgs, dropped = truncate_conversation(
            convo,
            max_tokens=threshold,
            turns_to_keep=TURNS_TO_KEEP,
            system_prompt=system_prompt,
        )
    # What is SENT (D3): `_prepare_messages_for_harness` keeps only these
    # roles, so a legacy `system` row in the session file (old `/compact`
    # restores) is dropped on the way to the engine and must not be counted.
    tokens_after = estimate_conversation_tokens(
        [m for m in truncated_msgs if m.get("role") in _SENT_ROLES], system_prompt,
    )

    if dropped > 0 and mode == "summarize" and not summarized:
        # Summarization was supposed to handle this but didn't (import
        # failure, wedged vLLM, empty summary) — turns were dropped
        # unsummarized. The fallback itself is correct design; losing
        # memory silently is not, so leave a visible trail.
        session_id = Path(session_path).stem
        logger.warning(
            "compaction: summarize fell back to truncation for %s (%d tokens dropped)",
            session_id, dropped,
        )
        try:
            from app.event_log import log_event
            log_event(session_id, "compaction.summarize_fallback", {
                "tokens_dropped": dropped,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning("compaction: event_log write failed: %s", e)

    result: dict[str, Any] = {
        "history": truncated_msgs,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "truncated": dropped > 0,
        "summarized": summarized,
        "microcompacted": micro_cleared,
        "restored_files": restored_count,
        "context_window": context_window,
        "threshold": threshold,
        "summarize_attempted": summarize_attempted,
        "summarize_outcome": summarize_outcome,
        "summary_reused": summary_reused,
        "summary_folds": summary_folds,
        "summary_covered_rows": summary_covered_rows,
        # P3: a flush finished in this cycle before this rewrite. False when
        # nothing was rewritten, or with `compaction.memory_flush` off.
        "flushed_before_summary": bool(
            (summarized or dropped > 0) and _flushed_this_cycle(data)),
    }
    _record_turn_start(path.stem, result)
    return result


__all__ = [
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_conversation_tokens",
    "get_context_window",
    "truncation_threshold",
    "truncate_conversation",
    "load_and_compact_session",
    "TOKENS_PER_CHAR",
    "OUTPUT_TOKENS_RESERVED",
    "TRUNCATION_BUFFER_TOKENS",
    "DEFAULT_CONTEXT_WINDOW",
    "TURNS_TO_KEEP",
]
