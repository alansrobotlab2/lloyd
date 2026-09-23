#!/usr/bin/env python3
"""Three tools over djev, the structured-decision engine on GPU 2.

djev is DiffusionGemma 26B-A4B NVFP4 (`github.com/mmastrac/djev-spark`)
answering typed questions — yes/no, one-of-N, ordered scale — off one
diffusion canvas in ~40 ms plus ~0.25 ms per prompt token. The primary runs at
100% utilization; this card sits at 0% with 23.7 GiB of weights loaded, so
every decision moved here is pure throughput. `architecture/djev.md` is the
long version and `app/djev.py` is the client all three tools go through.

WHAT THESE TOOLS ARE FOR, AND THE LINE THEY DO NOT CROSS
--------------------------------------------------------
Measured over four framings of one question on 2026-09-20: AUC held at
0.72-0.83 in every arm, so the ORDER djev produces is a real signal. But
reversing the option order moved P by 0.324 on average, and each framing had
a different optimal threshold (0.39, 0.03, 0.30, 0.01). **Ranking and
ordering are safe; a fixed probability cutoff is meaningless.** Every
description below says so, because a tool description is the only thing a
model reads before calling it.

WHY A MODULE AND NOT A SECOND MCP SERVER
----------------------------------------
Same five reasons as `code_graph`: bare-name collisions in
`build_tool_list`, Task subagents pinning `DEFAULT_LLOYD_MCP_SERVERS`,
`tests/test_mcp_layer.py` needing every configured server discoverable inside
an automod worktree where no second daemon runs, `agent-services/supervisor/**`
being a protected automod path, and the tools Lloyd actually needs not being
the ones an upstream server would ship.

`list_tools()` IS OFFLINE AND MUST STAY THAT WAY
------------------------------------------------
No reachability probe, no config read that can raise, no import that touches
a socket. A module that degrades makes the aggregator answer `/health` with a
503, and `agent-services/guardian/detect.py::mcp_degraded_is_fatal` reads that
as a rollback trigger — so a djev engine that is merely stopped would revert
whatever landed last. Reachability is a per-CALL concern and `djev_status` is
where it is reported.

There is no module `enabled` flag for the same reason `code_graph` has none:
an `enabled: false` that emptied `list_tools()` breaks the
annotation-staleness test. The kill switch is
`mcp_servers.lloyd-mcp.disabled_tools`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.types import Tool

from agent_mcp._shared import text_result
from app import djev

logger = logging.getLogger("lloyd-djev-tools")

#: Longest candidate text carried into the canvas, per candidate. A rerank
#: pool of 12 at 1,200 chars is ~4k tokens of state, which is ~1 s cold and
#: ~70 ms warm.
CANDIDATE_CHARS = 1200

_TRUST_NOTE = (
    "Scores are self-consistent within one call, NOT calibrated: the ordering "
    "is meaningful and the absolute numbers are not comparable to a fixed "
    "cutoff."
)


def _err(message: str, **extra) -> str:
    return json.dumps({"error": message, **extra})


# ---------------------------------------------------------------------------
# djev_rank
# ---------------------------------------------------------------------------

async def _rank(args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return _err("query is required")
    raw = args.get("candidates")
    if not isinstance(raw, list) or not raw:
        return _err("candidates must be a non-empty array of strings")
    candidates = [str(c) for c in raw]
    n = len(candidates)
    if n > djev.RANK_MAX_N:
        # Refused rather than truncated. A caller handing over 40 rows means
        # to rank 40, and silently scoring the first 16 returns a confident
        # ordering of a slice nobody chose.
        return _err(
            f"djev ranks at most {djev.RANK_MAX_N} candidates in one request; "
            f"got {n}. Shortlist with a cheap first stage (vault_search, "
            f"backlog_similar) and rank the top {djev.RANK_DEFAULT_N} here.",
            max_candidates=djev.RANK_MAX_N, got=n)
    levels = args.get("levels")
    levels = [str(x) for x in levels] if isinstance(levels, list) and len(levels) >= 2 \
        else list(djev.RANK_LEVELS)

    rows = await asyncio.to_thread(
        djev.rank, query, [c[:CANDIDATE_CHARS] for c in candidates],
        seam="tool:rank", levels=levels,
        timeout=float(args.get("timeout_seconds") or djev.DEFAULT_TIMEOUT_S))
    if rows is None:
        return _err("djev did not answer (engine unreachable, disabled, or the "
                    "server split the canvas). Nothing was ranked; use the "
                    "order you already had.")
    out = []
    for r in rows:
        out.append({"rank": len(out) + 1, "index": r["index"],
                    "score": round(r["score"], 4), "level": r["label"],
                    "label_mass": round(r["label_mass"], 4),
                    "argmax_is_label": r["argmax_is_label"],
                    "candidate": candidates[r["index"]][:160]})
    low = [r["index"] for r in rows if r["label_mass"] < 0.5]
    return json.dumps({
        "query": query, "n": n, "levels": levels, "ranked": out,
        "note": _TRUST_NOTE,
        # Surfaced on every answer, not only when it is bad: the returned
        # probabilities are renormalized over the label set regardless of how
        # little mass landed there, so they always look confident.
        "min_label_mass": round(min(r["label_mass"] for r in rows), 4),
        "low_label_mass_indexes": low,
    }, default=str)


# ---------------------------------------------------------------------------
# djev_decide
# ---------------------------------------------------------------------------

_VALID_TYPES = ("noul", "choice", "score")


def _validate(questions: Any) -> str | None:
    if not isinstance(questions, dict) or not questions:
        return "questions must be a non-empty object of id -> question"
    if len(questions) > djev.CANVAS_CHUNK_QUESTIONS:
        return (f"more than {djev.CANVAS_CHUNK_QUESTIONS} questions splits the "
                f"canvas into separate shared contexts, and answers from "
                f"different chunks are not comparable with each other. Split "
                f"the call instead.")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            return f"question {qid!r} must be an object"
        kind = q.get("type")
        if kind not in _VALID_TYPES:
            return f"question {qid!r}: type must be one of {', '.join(_VALID_TYPES)}"
        crit = q.get("criteria")
        if kind == "choice" and (not isinstance(crit, dict) or len(crit) < 2):
            return (f"question {qid!r}: a choice needs a criteria object mapping "
                    f"at least two option names to descriptions")
        if kind == "score" and (not isinstance(crit, list) or len(crit) < 2):
            return (f"question {qid!r}: a score needs criteria as an ORDERED "
                    f"list of at least two level names, worst first")
    return None


async def _decide(args: dict) -> str:
    state = args.get("state")
    if state is None or (isinstance(state, str) and not state.strip()):
        return _err("state is required: the text the questions are asked about")
    questions = args.get("questions")
    if isinstance(questions, str):
        try:
            questions = json.loads(questions)
        except ValueError:
            return _err("questions was a string that is not JSON")
    bad = _validate(questions)
    if bad:
        return _err(bad)

    out = await djev.ask(
        state, questions, seam="tool:decide",
        instructions=str(args.get("instructions") or "") or None,
        samples=int(args["samples"]) if args.get("samples") else None,
        timeout=float(args.get("timeout_seconds") or djev.DEFAULT_TIMEOUT_S))
    if out is None:
        return _err("djev did not answer (engine unreachable, disabled, or the "
                    "response was malformed). No decision was made.")
    body = out.as_dict()
    body["note"] = _TRUST_NOTE
    if out.cross_chunk:
        body["warning"] = (
            "the server split these questions across "
            f"{len(out.chunks)} canvas chunks; answers from different chunks "
            "were produced in separate shared contexts and MUST NOT be sorted "
            "against each other")
    if out.uninformative:
        body["warning_uninformative"] = (
            "every answer carried the same value — a degenerate read, which a "
            "label_mass floor cannot catch because the mass is legal and the "
            "answer is empty")
    return json.dumps(body, default=str)


# ---------------------------------------------------------------------------
# djev_status
# ---------------------------------------------------------------------------

async def _status(args: dict) -> str:
    body: dict[str, Any] = {"client": djev.stats()}
    body["reachable"] = await asyncio.to_thread(djev.reachable)
    body["rank_limits"] = {"default": djev.RANK_DEFAULT_N,
                           "max": djev.RANK_MAX_N,
                           "canvas_chunk_questions": djev.CANVAS_CHUNK_QUESTIONS}
    try:
        from app import djev_shadow
        body["shadow"] = djev_shadow.stats()
        # `seams` is the per-seam SWITCH. Read alone it is exactly the sentence
        # that hid #1372: three `true`s and a whole-file `log_rows`, while the
        # rerank hook had been unreachable since 2026-09-21. `seam_log` is the
        # traffic, per seam; `structurally_dark` below is the reachability.
        body["shadow"]["seams"] = {s: djev_shadow.seam_enabled(s)
                                   for s in djev_shadow.SEAMS}
    except Exception as exc:  # noqa: BLE001 — status never fails on a part
        body["shadow"] = {"error": f"{type(exc).__name__}: {exc}"}
    if "error" not in body["shadow"]:
        try:
            # Owned by the module that owns the dispatch, not by the recorder:
            # `app/djev_shadow.py` cannot know what `vault_recall` reaches. A
            # verdict that cannot be re-measured belongs in neither file, so a
            # read that fails says it cannot tell instead of answering `{}` —
            # which would read as "no seam is dark", the false verdict this
            # whole report exists to remove.
            from agent_mcp import vault
            body["shadow"]["structurally_dark"] = vault.shadow_seams_dark_by_dispatch()
        except Exception as exc:  # noqa: BLE001 — status never fails on a part
            body["shadow"]["structurally_dark"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        # `eval/` is not a package the aggregator normally imports, so this is
        # best-effort by design: a status route that 500s because a sibling
        # tree moved is worse than one that says the floors are unreadable.
        from eval.djev import schemas
        body["schemas"] = {n: s.as_dict() for n, s in schemas.SCHEMAS.items()}
    except Exception as exc:  # noqa: BLE001
        body["schemas"] = {"error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(body, default=str)


_HANDLERS = {"djev_rank": _rank, "djev_decide": _decide, "djev_status": _status}


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------

async def list_tools():
    """Offline by construction — see the module docstring. No socket, no
    config read, nothing that can raise."""
    return [
        Tool(
            name="djev_rank",
            description=(
                "Re-rank a SHORTLIST of candidates against a query on the idle "
                "GPU-2 decision engine, in ~0.5 s for 12. Use as a final stage "
                "after a cheap retrieval pass, never as the retrieval itself. "
                "Ordering is the trustworthy output; the scores are not "
                "calibrated probabilities. Refuses more than "
                f"{djev.RANK_MAX_N} candidates rather than truncating."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description":
                              "What the candidates are being ranked against"},
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description":
                            f"Candidate texts, best {djev.RANK_DEFAULT_N} or "
                            f"fewer; {djev.RANK_MAX_N} is a hard ceiling "
                            f"because label_mass degrades past it. Each is "
                            f"truncated to {CANDIDATE_CHARS} characters.",
                    },
                    "levels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description":
                            "Ordered relevance levels, WORST first (default: "
                            f"{', '.join(djev.RANK_LEVELS)}). Reversing this "
                            "list reverses the ranking.",
                    },
                    "timeout_seconds": {"type": "number", "description":
                                        "Client-side bound (default "
                                        f"{djev.DEFAULT_TIMEOUT_S})"},
                },
                "required": ["query", "candidates"],
            },
        ),
        Tool(
            name="djev_decide",
            description=(
                "Ask typed questions — yes/no, one-of-N, ordered scale — about "
                "one piece of text and get probabilities back in ~40 ms, on "
                "the otherwise-idle GPU 2 rather than the busy primary. Good "
                "for classifying, shortlisting and ordering. The scores are "
                "self-consistent, NOT calibrated: do not compare them to a "
                "fixed cutoff such as 0.5."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description":
                              "The text the questions are asked about"},
                    "questions": {
                        "type": "object",
                        "description":
                            "id -> {type, instructions, criteria}. type is "
                            "'noul' (yes/no, criteria optional), 'choice' "
                            "(criteria maps option name -> description) or "
                            "'score' (criteria is an ORDERED list of level "
                            "names, worst first). Option order changes the "
                            "answer, so keep it fixed across calls you intend "
                            f"to compare. At most "
                            f"{djev.CANVAS_CHUNK_QUESTIONS} questions.",
                    },
                    "instructions": {"type": "string", "description":
                                     "Optional context prepended to every question"},
                    "samples": {"type": "integer", "description":
                                "Noise draws to average (default 1; more is "
                                "steadier and proportionally slower)"},
                    "timeout_seconds": {"type": "number", "description":
                                        "Client-side bound (default "
                                        f"{djev.DEFAULT_TIMEOUT_S})"},
                },
                "required": ["state", "questions"],
            },
        ),
        Tool(
            name="djev_status",
            description=(
                "Is the djev decision engine switched on and answering, how "
                "fast has it been lately per call site, how deep is the shadow "
                "queue and what has it dropped, and which question schemas "
                "have a measured threshold and label_mass floor yet."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "verbose": {"type": "boolean", "description":
                                "Accepted for symmetry with the other status "
                                "tools; this route already returns everything "
                                "it knows."},
                },
            },
        ),
    ]


async def call_tool(name: str, arguments: dict):
    fn = _HANDLERS.get(name)
    if fn is None:
        return text_result(json.dumps({"error": f"Unknown tool: {name}"}))
    try:
        text = await fn(arguments or {})
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("djev: %s failed", name)
        text = json.dumps({"error": f"{name} failed: {exc}"})
    return text_result(text)


#: How long a shutting-down aggregator waits for the shadow queue. Short on
#: purpose: a landing restart is drained and idle-gated already, and a
#: recorder that delayed a restart would be costing production time for an
#: observation. What does not make it out is counted into
#: `dropped_at_shutdown` on the next process's first row.
SHUTDOWN_FLUSH_S = 5.0


async def shutdown() -> None:
    """Drain the shadow queue on the way down.

    Through the MODULES shutdown hook `main.lifespan` already runs, rather
    than a special case inside it: the worker is a daemon thread and its queue
    goes with the process otherwise, and the stack restarts several times a
    night, so a silent loss would read exactly like a quiet seam.
    """
    try:
        from app import djev_shadow
        left = await asyncio.to_thread(djev_shadow.flush, SHUTDOWN_FLUSH_S)
        if left:
            logger.info("djev shadow: %d rows unsent at shutdown", left)
    except Exception:  # noqa: BLE001
        logger.debug("djev shadow: flush at shutdown failed", exc_info=True)
