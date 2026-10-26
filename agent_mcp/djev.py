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

#: What `djev_rank` annotates against when its schema has no measured floor
#: (#3115), and the vocabulary the payload uses to say so.
#:
#: The floor a ranking is supposed to be judged by lives in
#: `eval/djev/schemas.py`, and the `rerank` schema's `label_mass_floor` is
#: deliberately `None`: listwise `label_mass` at n=12-16 measured 0.446, 0.807,
#: 0.965 and 1.000 across four corpora, so a cutoff read off any one of those
#: means nothing for the others — and 0.5 sits inside the band that spread
#: calls healthy. The tool compared against a bare 0.5 anyway and reported the
#: result with no sign of where it came from. The number stays, as the pointer
#: it is; what it could not do before is name itself as uncalibrated.
UNCALIBRATED_RANK_CUTOFF = 0.5

#: The schema a ranking answer is judged against, and the two values
#: `low_label_mass_cutoff_source` may hold. Resolution goes by schema NAME, not
#: by `schemas.floor_for(seam)`: `BY_SEAM` holds only the seams the registry
#: declares (`rerank`, `dedupe`, `entity`, `action_review`) and this tool's seam
#: is `tool:rank`, which is absent from it — so `floor_for("tool:rank")` answers
#: None for a MISSING KEY, exactly the value an unmeasured schema gives, and the
#: two states would be indistinguishable in the one place the difference
#: matters. `agent_mcp/vault.py`'s recall ranker has the same key problem with
#: `recall_rank`; registering those seams is a separate ruling.
RANK_SCHEMA_NAME = "rerank"
CUTOFF_FROM_SCHEMA = "rerank_schema"
CUTOFF_UNCALIBRATED_FALLBACK = "uncalibrated_fallback"


def _rank_floor() -> float | None:
    """The measured `label_mass` floor for a ranking, or None if none exists.

    Read at call time, so the registry stays the only place a floor is set:
    whenever `eval/djev/replay.py --floors` does give `rerank` one, this tool
    follows without a second edit. `eval/` is not a package the aggregator
    normally imports — the same best-effort reasoning as `_status`'s import
    further down — and an annotation that cannot resolve must not cost the
    caller the ordering it asked for (§8.1: advisory, fail open), so a failed
    import is the honest None, which the payload then labels as a fallback.
    """
    try:
        from eval.djev import schemas
    except Exception as exc:  # noqa: BLE001 — an annotation never loses a ranking
        logger.debug("djev rank: schema registry unreadable: %s", exc)
        return None
    schema = schemas.get(RANK_SCHEMA_NAME)
    return schema.label_mass_floor if schema is not None else None


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

    # One read, pinned, because this path was measured and not copied across
    # the seam: over the 21 real `djev_rank` calls on 2026-10-26 `auto` cost
    # 4 reads on 21/21 at a 323.9 ms median against 1 read at 133.9 ms, top
    # pick identical on 20/21. §8.1 found the same penalty on the recall
    # shape and pinned one read there, so both rank paths hold one budget
    # (#3377). tests/test_djev_tools.py pins the kwarg and the posted body.
    # The registry's floor for a ranking, or None. Handed to the client as well
    # as used for the annotation below, because `app/djev.py`'s per-row
    # `low_trust` is `floor is not None and label_mass < floor` (`:355`) — with
    # no floor threaded it was structurally False on every row, so the tool
    # threw away the calibrated signal and shipped its own unowned 0.5 in its
    # place (#3115). `floor` never reaches the engine: `_body` does not carry
    # it, so no request and no read count changes here.
    floor = _rank_floor()
    rows = await asyncio.to_thread(
        djev.rank, query, [c[:CANDIDATE_CHARS] for c in candidates],
        seam="tool:rank", levels=levels, floor=floor, samples=1,
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
                    # `.get`, not `[...]`: this is a pointer on an ordering, not
                    # a precondition of it. A row that somehow lacked the field
                    # reads as unflagged rather than raising away an answer the
                    # caller asked for (§8.1's fail-open contract).
                    "low_trust": r.get("low_trust", False),
                    "candidate": candidates[r["index"]][:160]})
    cutoff = floor if floor is not None else UNCALIBRATED_RANK_CUTOFF
    low = [r["index"] for r in rows if r["label_mass"] < cutoff]
    return json.dumps({
        "query": query, "n": n, "levels": levels, "ranked": out,
        "note": _TRUST_NOTE,
        # Surfaced on every answer, not only when it is bad: the returned
        # probabilities are renormalized over the label set regardless of how
        # little mass landed there, so they always look confident.
        "min_label_mass": round(min(r["label_mass"] for r in rows), 4),
        # Three keys, answering the one question a reader of
        # `low_label_mass_indexes` has to be able to ask: is this cutoff
        # something the calibration work measured? `label_mass_floor` is the
        # registry's number or null; `low_label_mass_cutoff` is what was
        # actually compared against; the source names which of the two it was.
        # Today the answer is `null` / 0.5 / `uncalibrated_fallback`, and that
        # is the honest state of the `rerank` schema, not an oversight.
        "label_mass_floor": floor,
        "low_label_mass_cutoff": cutoff,
        "low_label_mass_cutoff_source": (
            CUTOFF_FROM_SCHEMA if floor is not None
            else CUTOFF_UNCALIBRATED_FALLBACK),
        "low_label_mass_indexes": low,
    }, default=str)


# ---------------------------------------------------------------------------
# djev_decide
# ---------------------------------------------------------------------------

_VALID_TYPES = ("noul", "choice", "score")

#: Upper bound of a meaningful fixed `samples` count. It belongs to the engine,
#: not to this client: `agent-services/llm/djev-spark/server/structured_server.py`
#: answers a fixed count with `{"mode": "fixed", "n": min(samples, 32)}`, so a
#: larger number is accepted and clamped rather than refused — which is why the
#: handler refuses only a count below 1. Named here because the schema
#: description and the refusal message both have to state the same range; the
#: client deliberately does not enforce it.
SAMPLES_MAX = 32


def _samples_error(raw: Any) -> str:
    return (f"samples must be a positive count of noise draws between 1 and "
            f"{SAMPLES_MAX} (got {raw!r}). Omit the field to let the engine "
            f"choose: its auto policy is one read, or up to four when a first "
            f"read is uncertain.")


def _samples_arg(args: dict) -> "tuple[int | None, str | None]":
    """The `samples` argument as the engine should receive it, or an error.

    An absent field stays `None`, and `app/djev.py::_body` writes the key only
    when it is not None, so the request reaches djev with no `samples` and the
    engine applies its own default — `value.get("samples", "auto")`, mode auto
    with max 4 — which is one read, or four when a first read's entropy is
    above its threshold. That is the honest reading of an omitted field, and
    the schema description now says so.

    What must not stay is the falsy collapse. The handler used to write
    `int(args["samples"]) if args.get("samples") else None`, so an explicit `0`
    became the very auto policy the caller had just tried to override. The
    engine refuses a 0 outright (`SchemaError: schema: samples must be a
    positive count or "auto"`), but the client turns any engine failure into
    "engine unreachable, disabled, or the response was malformed", so the
    argument can never be named from there. This is the one place it can be.
    """
    raw = args.get("samples")
    if raw is None:
        return None, None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None, _samples_error(raw)
    if n < 1:
        return None, _samples_error(raw)
    return n, None


#: The keys a `noul` rubric must carry. The engine renders its prompt from
#: `crit.get("true")` / `crit.get("false")` and checks only `isinstance(crit,
#: dict)` (`structured_server.py:742`, `:135`), so a `{"yes": ..., "no": ...}`
#: object passes its door and answers with two empty label descriptions — a
#: wrong call that reads as an answer. Checking the keys here refuses both the
#: prose and the wrong keys, before a socket is opened.
_NOUL_CRITERIA_KEYS = ("true", "false")


def _noul_criteria_error(qid: str, crit: Any) -> str | None:
    """Refuse a `noul` criteria the engine cannot read, without inventing one.

    Absent is fine and is the shape most calls use: a noul with no rubric is
    the plain "is this true of the state?" question, and the engine answers it —
    all 6 `djev_decide` calls that ever got answers had criteria absent or typed
    for `choice`/`score`. Present-but-unreadable is the case that cost 9 calls:
    the schema invited prose by naming the shape for `choice` and `score` and
    not for `noul`, every criteria-bearing call wrote prose, the engine answered
    422, and the fold in `_decide` called that an outage.

    The fix is a refusal, never an expansion. Turning `"needs a reply now"` into
    `{"true": "needs a reply now", "false": ???}` would put words in the
    caller's mouth that the engine then scores against as if the caller had
    written the rubric — the same invention on the missing key as on the string.
    """
    if crit is None:
        return None
    if (isinstance(crit, dict)
            and all(isinstance(crit.get(k), str) and crit[k].strip()
                    for k in _NOUL_CRITERIA_KEYS)):
        return None
    return (f"question {qid!r}: a noul's criteria, if given, must be an object "
            f"with a non-empty 'true' and a non-empty 'false' description — "
            f"{{\"true\": \"what makes it yes\", \"false\": \"what makes it no\"}}. "
            f"A prose string is refused rather than invented into a rubric, and "
            f"the shape can be left out entirely for a plain yes/no. Nothing was "
            f"decided; correct the shape and call again.")


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
        if kind == "noul":
            bad = _noul_criteria_error(qid, crit)
            if bad:
                return bad
    return None


def _compact(out: "djev.Answers") -> dict:
    """What a caller acts on, and nothing it reads past.

    The full `as_dict()` is ~300 characters an answer — `label_mass`,
    `argmax_is_label`, unrounded probabilities — and every character of it
    comes back into the primary's context. The point of the tool is that the
    judgement is cheaper here than in the primary's reasoning, so the read-back
    must not eat the saving. A trust problem is still said: `low_trust` and
    `uninformative` appear on an answer only when true, `min_label_mass` stays
    on the body, and the cross-chunk and degenerate-read warnings are added by
    the caller exactly as on the verbose path. `verbose: true` returns it all.
    """
    answers = {}
    for qid, a in out.answers.items():
        row: dict[str, Any] = {"label": a.label}
        if a.type != "choice":
            # A choice's value IS its label; a noul's is P(yes) and a score's
            # the expected level index, both worth the number.
            row["value"] = round(float(a.value), 3)
        # A score's probabilities are keyed by level index; the verbose form
        # carries a `legend` to read them by, so compact keys them by name.
        legend = a.legend or {}
        row["p"] = {legend.get(str(k), k): round(float(v), 3)
                    for k, v in a.probabilities.items()}
        if a.low_trust:
            row["low_trust"] = True
        if a.uninformative:
            row["uninformative"] = True
        answers[qid] = row
    return {"answers": answers, "min_label_mass": round(out.min_label_mass, 3)}


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
    samples, samples_err = _samples_arg(args)
    if samples_err:
        return _err(samples_err)

    # Filled only by a 4xx (`app/djev.py::_fill_reject`), which is the one
    # failure the engine explains. A dict rather than an exception because the
    # engine's own answer for every other failure is a bare None, and the
    # outage sentence below is the honest reading of those.
    reject: dict = {}
    out = await djev.ask(
        state, questions, seam="tool:decide",
        instructions=str(args.get("instructions") or "") or None,
        samples=samples, reject=reject,
        timeout=float(args.get("timeout_seconds") or djev.DEFAULT_TIMEOUT_S))
    if out is None:
        if reject.get("status"):
            return _err(
                f"djev rejected the arguments (HTTP {reject['status']}, a "
                f"schema/argument error, not an outage — the engine answered): "
                f"{reject.get('message') or 'it gave no message'}. Nothing was "
                f"decided; fix the question's shape and call again.",
                http_status=reject["status"])
        return _err("djev did not answer (engine unreachable, disabled, or the "
                    "response was malformed). No decision was made.")
    body = out.as_dict() if args.get("verbose") else _compact(out)
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
                "Order a SHORTLIST against a query instead of comparing the "
                "candidates in your reasoning, in ~0.5 s for 12 on the idle "
                "GPU 2. Use as a final stage after a cheap retrieval pass, "
                "never as the retrieval itself. "
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
                "Classify or triage instead of deliberating: typed questions "
                "about one text, answered in under a second. Up to 32 yes/no, "
                "one-of-N or ordered-scale questions per call, on the idle "
                "GPU 2. Trust the label and the order; the scores are NOT "
                "calibrated, so do not compare them to a fixed cutoff such "
                "as 0.5."
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
                            "'noul' (yes/no; criteria, if given, "
                            "{true: desc, false: desc}), 'choice' (criteria "
                            "maps option name -> description) or 'score' "
                            "(criteria is an ORDERED list of level names, "
                            "worst first). Option order changes the answer, so "
                            "keep it fixed across calls you intend to compare.",
                    },
                    "instructions": {"type": "string", "description":
                                     "Optional context prepended to every question"},
                    "samples": {"type": "integer", "description":
                                "Noise draws to average per question. Omitting "
                                "it is not one read: the engine's auto policy "
                                "is one read, or four when a first read is "
                                f"uncertain. A positive count fixes it (above "
                                f"{SAMPLES_MAX} the engine clamps to "
                                f"{SAMPLES_MAX})"},
                    "verbose": {"type": "boolean", "description":
                                "Every diagnostic instead of the compact "
                                "answer (default false)"},
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
                                "Accepted for symmetry; this route already "
                                "returns everything it knows."},
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
