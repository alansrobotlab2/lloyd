"""The persisted compaction summary: one record per session, folded forward.

Review 2026-09-24, D2. Until this module the between-turn LLM summary was
never stored. Once a session crossed the compaction wall, every turn
re-summarised the whole older block from scratch: up to 120 s before the first
token, a *different* summary each time (so the prefix cache was invalidated on
every turn), and an older block with no bound on its size — a big one 400'd the
summariser and the turn silently fell back to drop-oldest, every turn.

Now the summary lives in the session file as a top-level
``data["compaction"]`` record (X6), and the history each turn is rebuilt as
``[summary] + rows past the boundary``:

* **Reused as is while it validates.** The same summary text every turn is the
  prefix-cache win; the summariser is not called at all.
* **Folded forward, never regenerated.** When the history past the boundary is
  over the wall, the older part of that delta is folded into the existing
  summary chunk by chunk (``chunk_by_turns``, bounded by
  ``compaction.summary_input_budget_tokens``), the record persisted after EACH
  fold so the boundary advances even if a later fold fails, at most
  ``compaction.max_folds_per_turn`` per turn. What is left over stays verbatim
  (the truncation fallback still runs) and the next turn keeps folding.
* **Invalidated by a changed past, not by a rewritten marker.** ``covered_sha``
  hashes ``id:role`` lines of the covered rows, so microcompact rewriting a
  result's content does not invalidate it, while a legacy ``/compact`` that
  replaced ``data["messages"]`` does (``compaction.record_invalidated``; the
  record is rebuilt from the rows).

Why a top-level key and not a message row (X6): a row would have to be
excluded from ``_group_into_turns``, ``_split_for_summary``, nine transcript
producers and ``/api/messages``; a key beside ``plan``/``goal`` is invisible to
all of them by construction. It is written AFTER ``messages`` — the retention
sweep (``scripts/groundskeeper/retention-sweep.py::_session_age_days``) reads a
4 KB prefix for ``last_active``, and a summary written ahead of that field
would push it out of the window.

``Files touched`` is not the model's to write: it is rendered from the per-turn
change ledger (``agent_mcp/_change_ledger``) over the turns the summary covers,
which every row has named since X3. Best effort — the ledger keeps 7 days, so
the record carries the list forward rather than re-deriving it.

Off switch: ``compaction.persist_summary`` (ships false until the recall eval
says otherwise). Off, a record on disk is ignored entirely and the turn-start
stack behaves exactly as it did before this module existed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Iterable

from app.compaction import _CONVERSATION_ROLES

logger = logging.getLogger("lloyd-compaction-state")

RECORD_KEY = "compaction"
RECORD_VERSION = 1

#: The roles the turn-start stack keeps from `data["messages"]`. Indexes in the
#: record are into THIS filtered list, so the filter is the one definition in
#: `app.compaction`, not a copy of it.
CONVERSATION_ROLES = _CONVERSATION_ROLES

#: The header on the summary row. Unchanged from the regenerate-every-turn
#: path, so a reader (and the model) sees the same thing either way.
SUMMARY_HEADER = (
    "[compaction summary — earlier conversation summarized to fit context window]"
)

#: Cap on the rendered Files touched list; the tail is counted, not dropped
#: silently.
FILES_TOUCHED_MAX = 60

#: Defaults, restated by `app.compaction._compaction_cfg`.
DEFAULT_INPUT_BUDGET_TOKENS = 48_000
DEFAULT_MAX_FOLDS_PER_TURN = 3
SUMMARY_MAX_OUTPUT_TOKENS = 8_000
#: Prompt scaffolding (system prompt + wrapper) the budget leaves room for.
_PROMPT_OVERHEAD_TOKENS = 2_000
#: Smallest input budget a fold is given, whatever the arithmetic says.
_MIN_INPUT_BUDGET_TOKENS = 2_000


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def conversation_rows(messages: Iterable[dict]) -> list[dict]:
    """The rows the turn-start stack works on, in session order."""
    return [m for m in messages or [] if isinstance(m, dict)
            and m.get("role") in CONVERSATION_ROLES]


def covered_sha(rows: Iterable[dict]) -> str:
    """sha256 over one ``id:role`` line per row.

    Content is deliberately left out: microcompact rewrites old results into
    markers and pointers on every load, and that must not read as a changed
    past. What does change the past — rows removed, reordered or replaced by a
    whole-history rewrite — changes ids or roles.
    """
    h = hashlib.sha256()
    for row in rows:
        h.update(f"{row.get('id') or ''}:{row.get('role') or ''}\n".encode())
    return h.hexdigest()


def turn_ids_of(rows: Iterable[dict]) -> list[str]:
    """Distinct `turn_id`s the rows name (X3), in first-seen order."""
    seen: list[str] = []
    for row in rows:
        tid = row.get("turn_id")
        if isinstance(tid, str) and tid and tid not in seen:
            seen.append(tid)
    return seen


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def load_record(data: dict | None) -> dict | None:
    """The stored record, or None when there is none worth reading."""
    if not isinstance(data, dict):
        return None
    rec = data.get(RECORD_KEY)
    if not isinstance(rec, dict):
        return None
    if rec.get("version") != RECORD_VERSION or not isinstance(rec.get("summary"), str):
        return None
    if not rec["summary"].strip():
        return None
    return rec


def validate(record: dict | None, convo: list[dict]) -> int | None:
    """Rows the record covers (the boundary), or None when it does not hold.

    Holds when the covered prefix still exists, its last row is the one the
    record names, and the ``id:role`` hash over it matches. Anything else means
    the past was rewritten under the record and the summary may describe rows
    that are no longer there.
    """
    if not record:
        return None
    try:
        index = int(record.get("covers_through_index"))
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= len(convo):
        return None
    want_id = record.get("covers_through_entry_id") or ""
    if want_id and (convo[index].get("id") or "") != want_id:
        return None
    if covered_sha(convo[: index + 1]) != record.get("covered_sha"):
        return None
    return index + 1


def render_summary(record: dict) -> str:
    """The text the summary row carries: the model's sections, then ours."""
    text = f"{SUMMARY_HEADER}\n\n{record.get('summary', '').strip()}"
    files = render_files_touched(record.get("files_touched") or [])
    if files:
        text += "\n\n" + files
    return text


def summary_message(record: dict) -> dict:
    """The row that replaces the covered history. Same role and header as the
    regenerate-every-turn path always used."""
    return {"role": "assistant",
            "content": [{"type": "text", "text": render_summary(record)}]}


def apply(convo: list[dict], record: dict) -> tuple[dict, list[dict], list[dict]] | None:
    """``(summary_msg, covered, remaining)``, or None when the record does not
    validate against ``convo``."""
    boundary = validate(record, convo)
    if boundary is None:
        return None
    return summary_message(record), convo[:boundary], convo[boundary:]


def build_record(
    *,
    summary: str,
    convo: list[dict],
    boundary: int,
    files_touched: list[dict],
    covered_turn_ids: list[str],
    model: str,
    prior: dict | None = None,
    folds_added: int = 1,
    source: str = "auto",
    instructions: str = "",
) -> dict:
    """A record covering ``convo[:boundary]`` (``boundary`` > 0)."""
    now = time.time()
    covered = convo[:boundary]
    return {
        "version": RECORD_VERSION,
        "summary": summary.strip(),
        "covers_through_entry_id": covered[-1].get("id") or "",
        "covers_through_index": boundary - 1,
        "covered_sha": covered_sha(covered),
        "covered_rows": boundary,
        "covered_turn_ids": list(covered_turn_ids),
        "files_touched": list(files_touched),
        "created_at": (prior or {}).get("created_at") or now,
        "updated_at": now,
        "model": model or "",
        "folds": int((prior or {}).get("folds") or 0) + int(folds_added),
        "source": source,
        "instructions": instructions or "",
    }


async def save_record(session_id: str, record: dict, *,
                      path: Path | str | None = None) -> bool:
    """Write ``data["compaction"]`` and nothing else, under the session lock.

    The record goes in LAST (popped and re-set), so it always sits after
    ``messages`` and the sweep's 4 KB prefix read still finds ``last_active``.
    Re-validated against the rows on disk inside the lock: rows appended since
    the caller read the file are fine (the boundary is a prefix), a rewritten
    past is not, and the record is then not written at all.
    """
    from app import sessions_io

    def _write(data: dict) -> None:
        convo = conversation_rows(data.get("messages") or [])
        if validate(record, convo) is None:
            raise _Stale()
        data.pop(RECORD_KEY, None)
        data[RECORD_KEY] = record

    try:
        return await sessions_io.mutate_session(
            session_id, _write, path=Path(path) if path is not None else None)
    except _Stale:
        logger.warning("compaction_state: %s changed under the fold; record not saved",
                       session_id)
        return False
    except Exception as e:  # noqa: BLE001 — a lost record costs a re-fold, not a turn
        logger.warning("compaction_state: save_record failed for %s: %s", session_id, e)
        return False


class _Stale(Exception):
    """The on-disk past no longer matches the record being saved."""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _estimate(rows: list[dict]) -> int:
    from app.compaction import estimate_conversation_tokens
    return estimate_conversation_tokens(rows)


def _row_text(row: dict) -> str:
    from app.harness.microcompact import _tool_result_text
    return _tool_result_text(row)


def _with_text(row: dict, text: str) -> dict:
    from app.harness.microcompact import _replace_tool_content
    return _replace_tool_content(row, text)


def _reduce_turn(turn: list[dict], budget_tokens: int) -> list[dict]:
    """Shrink one over-budget turn for the summariser's eyes only.

    First every spilled tool result drops to its ``<persisted-output>`` header
    (the path is what matters to a summary); then, if the turn is still over,
    every row's text is clipped to an equal share of the budget. The rows the
    record covers are the originals — this is the summariser's copy.
    """
    from app.harness.microcompact import _persisted_block_only

    out = []
    for row in turn:
        if row.get("role") == "tool":
            text = _row_text(row)
            reduced = _persisted_block_only(text)
            out.append(_with_text(row, reduced) if reduced != text else row)
        else:
            out.append(row)
    if _estimate(out) <= budget_tokens:
        return out
    from app.compaction import TOKENS_PER_CHAR
    share = max(200, (budget_tokens * TOKENS_PER_CHAR) // max(1, len(out)))
    clipped = []
    for row in out:
        text = _row_text(row) if row.get("role") == "tool" else _content_text(row)
        if len(text) > share:
            clipped.append(_with_text(row, text[:share] + "\n…[clipped for the summary]"))
        else:
            clipped.append(row)
    return clipped


def _content_text(row: dict) -> str:
    content = row.get("content")
    if isinstance(content, str):
        return content
    return _row_text(row)


def chunk_by_turns(rows: list[dict], budget_tokens: int) -> list[tuple[int, list[dict]]]:
    """Split ``rows`` into summariser inputs of at most ~``budget_tokens``.

    Returns ``[(end, rows_for_summary), ...]`` where ``end`` is the index in
    ``rows`` one past the chunk's last row. Chunks break only on
    ``_group_into_turns`` boundaries, so a fold never leaves a tool result
    verbatim whose call went into the summary. A single turn bigger than the
    budget is its own chunk, reduced by ``_reduce_turn``.
    """
    from app.compaction import _group_into_turns

    budget_tokens = max(1, int(budget_tokens))
    chunks: list[tuple[int, list[dict]]] = []
    cur: list[dict] = []
    cur_tokens = 0
    pos = 0
    for turn in _group_into_turns(rows):
        t_tokens = _estimate(turn)
        if cur and cur_tokens + t_tokens > budget_tokens:
            chunks.append((pos, cur))
            cur, cur_tokens = [], 0
        if t_tokens > budget_tokens:
            pos += len(turn)
            chunks.append((pos, _reduce_turn(turn, budget_tokens)))
            continue
        cur.extend(turn)
        cur_tokens += t_tokens
        pos += len(turn)
    if cur:
        chunks.append((pos, cur))
    return chunks


# ---------------------------------------------------------------------------
# Files touched
# ---------------------------------------------------------------------------


def files_touched_for(session_id: str, turn_ids: Iterable[str],
                      prior: list[dict] | None = None) -> list[dict]:
    """``[{path, op}]`` from the change ledger over ``turn_ids``, merged onto
    ``prior`` (a later write of the same path moves it to the end)."""
    merged: dict[str, dict] = {}
    for entry in prior or []:
        if isinstance(entry, dict) and entry.get("path"):
            merged[entry["path"]] = {"path": entry["path"], "op": entry.get("op") or ""}
    try:
        from agent_mcp import _change_ledger as ledger
    except Exception as e:  # noqa: BLE001 — best effort by design
        logger.debug("compaction_state: change ledger unavailable: %s", e)
        return list(merged.values())
    for tid in turn_ids:
        try:
            entries = ledger.list_changes(session_id, tid)
        except Exception:  # noqa: BLE001
            continue
        for e in entries:
            path = e.get("path") or e.get("real")
            if not path:
                continue
            op = e.get("op") or ""
            if e.get("reverted_at"):
                op = f"{op}, reverted"
            merged.pop(path, None)
            merged[path] = {"path": path, "op": op}
    return list(merged.values())


def render_files_touched(files: list[dict]) -> str:
    if not files:
        return ""
    shown = files[-FILES_TOUCHED_MAX:]
    lines = ["## Files touched"]
    if len(files) > len(shown):
        lines.append(f"({len(files) - len(shown)} earlier files not listed)")
    lines += [f"- {f['path']} ({f.get('op') or 'changed'})" for f in shown]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


def input_budget(cfg: dict, summary_model: str, prior_summary: str = "") -> int:
    """Tokens of delta one fold may hand the summariser.

    ``min(summary_input_budget_tokens, window - max_output - overhead)``, less
    the prior summary it is sent alongside.
    """
    from app.compaction import estimate_tokens, get_context_window

    configured = int(cfg.get("summary_input_budget_tokens") or DEFAULT_INPUT_BUDGET_TOKENS)
    window = get_context_window(summary_model)
    room = window - SUMMARY_MAX_OUTPUT_TOKENS - _PROMPT_OVERHEAD_TOKENS
    budget = min(configured, room) - estimate_tokens(prior_summary or "")
    return max(_MIN_INPUT_BUDGET_TOKENS, budget)


async def fold(
    convo: list[dict],
    *,
    start: int,
    end: int,
    record: dict | None,
    session_id: str,
    path: Path | str | None,
    model: str,
    cfg: dict,
    max_folds: int,
    instructions: str = "",
    source: str = "auto",
) -> dict[str, Any]:
    """Fold ``convo[start:end]`` into ``record`` (None = no summary yet).

    ``start`` is the record's boundary (0 without one). Chunk by chunk, each
    fold persisted before the next is attempted. Returns ``{"record",
    "folds", "failed", "remaining_chunks", "duration_ms"}``; ``record`` is the
    newest one saved (or the one passed in when no fold succeeded).
    """
    from app import compaction_llm
    from app.harness.telemetry import log_harness_event

    t0 = time.monotonic()
    prior_summary = (record or {}).get("summary") or ""
    folds = 0
    failed = False
    pos = start
    remaining_chunks = 0
    while True:
        # Re-chunked after every fold: the prior summary the next request
        # carries has grown, so the room left for delta has shrunk.
        budget = input_budget(cfg, model, prior_summary)
        chunks = chunk_by_turns(convo[pos:end], budget)
        remaining_chunks = len(chunks)
        if not chunks or folds >= max(0, int(max_folds)):
            break
        chunk_end, rows = chunks[0]
        boundary = pos + chunk_end
        new_summary = await compaction_llm.summarize_incremental(
            prior_summary or None, rows, model=model or None,
            instructions=instructions or None,
            max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
            input_budget_tokens=budget,
        )
        if not new_summary:
            failed = True
            break
        # The turns this fold newly covers; the ledger is read for those only
        # and the earlier list rides forward on the record (7-day retention).
        prior_turns = list((record or {}).get("covered_turn_ids") or [])
        new_turns = [t for t in turn_ids_of(convo[:boundary]) if t not in prior_turns]
        files = files_touched_for(session_id, new_turns,
                                  (record or {}).get("files_touched") or [])
        record = build_record(
            summary=new_summary, convo=convo, boundary=boundary,
            files_touched=files, covered_turn_ids=prior_turns + new_turns,
            model=model, prior=record, source=source, instructions=instructions,
        )
        await save_record(session_id, record, path=path)
        folds += 1
        pos = boundary
        prior_summary = record["summary"]
        # One event per fold: the boundary it advanced to is the datum.
        log_harness_event(session_id, "compaction.summary_updated", {
            "covers_through": record["covers_through_index"],
            "covered_rows": record["covered_rows"],
            "folds": record["folds"],
            "chars": len(record["summary"]),
            "model": record["model"],
            "source": source,
            "duration_ms": int((time.monotonic() - t0) * 1000),
        })
    return {
        "record": record,
        "folds": folds,
        "failed": failed,
        "remaining_chunks": remaining_chunks,
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }


__all__ = [
    "RECORD_KEY",
    "RECORD_VERSION",
    "SUMMARY_HEADER",
    "conversation_rows",
    "covered_sha",
    "turn_ids_of",
    "load_record",
    "validate",
    "apply",
    "summary_message",
    "render_summary",
    "build_record",
    "save_record",
    "chunk_by_turns",
    "files_touched_for",
    "render_files_touched",
    "input_budget",
    "fold",
]
