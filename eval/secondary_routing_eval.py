#!/usr/bin/env python3
"""Paired engine comparison on the work the *secondary* is routed to — item #551.

    cd ~/lloyd && .venvs/lloyd/bin/python eval/secondary_routing_eval.py

Five jobs run on the weaker engine (Qwen3.6-35B-A3B GGUF Q3 on llama.cpp,
one slot, ~9 tok/s) because the routing decision was made on throughput.
Nothing in `eval/` measured whether that engine is adequate for them; the
nightly quality eval scores retrieval, not generation. This does the
generation half: the same recorded input through both engines, interleaved
repeats, mechanical scores, wall seconds, output tokens, and a routing
decision per job against a margin written down before the run.

Why the two engines are compared through the production functions and not
by re-issuing prompts: a model is only ever measured inside its harness
(that is the one transferable idea from the Artificial Analysis walkthrough
this item came from), and the payload the routed jobs actually send —
temperature, `max_tokens`, `enable_thinking: False`, the exact system
prompt — is the thing under test. So the secondary arm calls the real
`_sync_secondary_*` functions, and the primary arm calls the *same*
functions with the one thing that differs swapped: `resolve_model_alias`,
which is exactly the knob `secondary_enabled: false` already turns. A gap
measured any other way would be a gap between two prompt writers.

Token counts come from the engine's own `usage` block, not from a local
tokenizer approximation, by teeing the response body of the production
call (`_recording`) — the production functions discard `usage`, and the
number the engine reports for its own output is the one the cost axis
should use.

Scoring is mechanical on purpose. A bad scorer is worse than no scorer, and
the judge model here is the primary engine, which is one of the two arms
being compared — so the judge runs only under `--judge`, is reported as its
own column, and is deliberately excluded from `decide()`.

Outputs land in `eval/secondary-routing/`: `results-<label>.json` (every
trial) and `REPORT-<label>.md` (the table and the per-job decision).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The five production entry points this eval exists to measure. Importing
# them (rather than re-implementing the HTTP call) is what keeps the two
# arms on the same harness; `grep -rln _sync_secondary_capture_call eval/`
# is the acceptance check that this file still does that.
import app.secondary_models as secondary_models  # noqa: E402  — the router itself
from app.secondary_models import (  # noqa: E402
    _sync_secondary_capture_call,
    _sync_secondary_fact_extraction,
    _sync_secondary_focus_extraction,
    _sync_secondary_title,
    _sync_secondary_voice_summary,
)

EVAL_DIR = Path(__file__).resolve().parent
ITEMS_PATH = EVAL_DIR / "secondary_generation_items.yaml"
OUT_DIR = EVAL_DIR / "secondary-routing"

#: Every job `app/secondary_models.py` routes to the secondary slot. The
#: triage correction to #551 counted these five explicitly; the original
#: item listed three and omitted the two that write into memory.
JOBS = ("title", "capture", "facts", "focus", "voice")

JOB_LABELS = {
    "title": "session title",
    "capture": "post-session capture",
    "facts": "durable fact extraction",
    "focus": "focus/topic extraction",
    "voice": "voice summary",
}

# The routed function per job, and where production calls it from. Kept as
# data so the report can name the call site a decision would change.
JOB_CALLS: dict[str, Callable[[Any], Any]] = {
    "title": _sync_secondary_title,            # app/session_titles.py:250
    "capture": _sync_secondary_capture_call,   # app/post_capture.py:356
    "facts": _sync_secondary_fact_extraction,  # app/post_capture.py:370
    "focus": _sync_secondary_focus_extraction,  # app/post_capture.py:169
    "voice": _sync_secondary_voice_summary,    # app/routers/voice.py:215
}

# ── Decision policy, stated before the run ───────────────────────────────
#
# The secondary costs roughly 40x the primary in generation latency (9.01
# vs 357.5 tok/s in eval/measurements.json). Paying that is only worth it
# if the cheaper engine's output is *as good*, so the secondary keeps a job
# only while it stays inside both margins. These two numbers are the
# tolerance the acceptance clause asks for; changing one is a policy change
# and belongs in a commit message, not in a result.
KEEP_MARGIN_POINTS = 5.0      # composite mean: secondary within 5 pts of primary
KEEP_DEFECT_MARGIN = 0.05     # over-long-or-duplicated trials: within +5 points

#: The floor the acceptance clause names ("spread over >=3 interleaved
#: repeats"). Below it `decide` declines to route anything.
MIN_REPEATS = 3

# Composite weights, identical for every job so a per-job score means the
# same thing across the table.
W_FORMAT = 30.0     # does the output have the fields the job's prompt asked for
W_ANCHORS = 30.0    # does it name subjects that are actually in the input
W_LENGTH = 25.0     # within the job's length budget (the over-long defect)
W_UNIQUE = 15.0     # no near-duplicate lines inside one output (the duplicate defect)

# Character budgets per job. `capture`'s is the one tied to standing
# problem #4: its prompt asks for 2-4 sentences, and daily-learnings
# entries being over-long is the defect the item names.
LENGTH_BUDGETS = {"title": 60, "capture": 900, "facts": 1000, "focus": 170, "voice": 600}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")
_MD_MARKERS = ("```", "`", "\n- ", "\n* ", "| ", "](", "## ", " #")


# ── Text signals ─────────────────────────────────────────────────────────


def _as_text(result: Any) -> str:
    """A job's output as text, whatever the production function returned.

    Facts return a list of dicts, focus a list of phrases, the others a
    string or None. A None is an empty output, which scores zero rather
    than raising — a job that returns nothing failed, and that has to be
    countable.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, list):
        if result and isinstance(result[0], dict):
            return "\n".join(f"[{d.get('entity', '')}] {d.get('fact', '')}".strip() for d in result)
        return "\n".join(str(x) for x in result)
    return str(result)


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split((text or "").strip())]
    return [p for p in parts if p]


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _shingles(text: str) -> set[str]:
    words = _norm(text).split()
    if len(words) <= 2:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + 2]) for i in range(len(words) - 1)}


def duplicate_rate(text: str) -> float:
    """Fraction of a unit's lines that repeat another unit's line.

    Comparison is over word bigrams with Jaccard >= 0.7. The over-long and
    duplicate defects in standing problem #4 are both "the entry says it
    twice", and a substring match would miss a paraphrased restatement
    while an embedding call would make the scorer the thing under review.
    """
    units = _lines(text) or _sentences(text)
    if len(units) < 2:
        return 0.0
    sets = [_shingles(u) for u in units]
    dup = 0
    pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            pairs += 1
            a, b = sets[i], sets[j]
            if not a or not b:
                continue
            overlap = len(a & b)
            # Jaccard alone missed a line repeated with two words appended
            # (0.57 against a 0.7 bar): a restatement is not two equal sets,
            # it is one line contained in another, so containment is the
            # measure that matches the defect. The 3-shingle floor on the
            # smaller line keeps the focus job's short topics — two 3-word
            # phrases sharing their first two words are not a repetition —
            # from reading as duplicating itself.
            if overlap / len(a | b) >= 0.7 or (min(len(a), len(b)) >= 3
                                               and overlap / min(len(a), len(b)) >= 0.75):
                dup += 1
    return dup / pairs if pairs else 0.0


def anchor_recall(text: str, anchors: list[str]) -> Optional[float]:
    """Share of the input's identifying tokens the output actually names.

    Anchors are paths, identifiers, numbers and quoted names lifted from
    the recorded input, so this is a floor on groundedness: an output that
    drifts into "the conversation discussed several topics" scores 0
    without needing a model to notice. Case-insensitive, and a path anchor
    also counts when only its basename survives.
    """
    if not anchors:
        return None
    return _anchor_hits(text, anchors) / len(anchors)


# ── Per-job format checks ────────────────────────────────────────────────
#
# Each returns the list of ways the output missed what that job's own
# prompt asked for. Everything here is legible off the raw text, and the
# raw text comes from the engine's response body — because the production
# parsers *repair* the shape they were asked for (a fact line with no
# [Entity] prefix is silently filed under "Lloyd", which would otherwise
# hide the exact format failure worth measuring).


def _check_title(text: str) -> list[str]:
    fails = []
    words = _norm(text).split()
    if not 3 <= len(words) <= 6:
        fails.append(f"{len(words)} words, prompt asks 3-6")
    if "\n" in text:
        fails.append("more than one line")
    if text.endswith((".", "!", "?", ":", ";")):
        fails.append("ends in punctuation")
    if '"' in text or text.startswith("'"):
        fails.append("contains quotes")
    low = _norm(text)
    if low.startswith(("title:", "this conversation", "summary:")):
        fails.append("preamble instead of a title")
    return fails


def _check_capture(text: str) -> list[str]:
    fails = []
    count = len(_sentences(text))
    if not 2 <= count <= 4:
        fails.append(f"{count} sentences, prompt asks 2-4")
    if not text.strip():
        fails.append("empty")
    if text.strip().upper() == "TRIVIAL":
        fails.append("answered TRIVIAL on a substantive session")
    return fails


def _check_facts(text: str) -> list[str]:
    fails = []
    lines = _lines(text)
    if not 3 <= len(lines) <= 5:
        fails.append(f"{len(lines)} fact lines, prompt asks 3-5")
    bracketed = [ln for ln in lines if re.match(r"^\[[^\]]+\]\s*\S", ln)]
    if lines and len(bracketed) < len(lines):
        fails.append(f"{len(lines) - len(bracketed)} lines lack an [Entity] prefix")
    if lines and len(bracketed) == len(lines):
        long = [ln for ln in lines if len(ln) > 200]
        if long:
            fails.append(f"{len(long)} lines over 200 chars")
    return fails


def _check_focus(text: str) -> list[str]:
    fails = []
    lines = _lines(text)
    if not 3 <= len(lines) <= 5:
        fails.append(f"{len(lines)} topics, prompt asks 3-5")
    for ln in lines:
        words = len(_norm(ln).split())
        if not 2 <= words <= 6:
            fails.append(f"topic {ln!r} is {words} words")
            break
    numbered = [ln for ln in lines if re.match(r"^\s*(\d+[.)-]|[-*]\s)", ln)]
    if numbered:
        fails.append(f"{len(numbered)} lines numbered or bulleted")
    return fails


def _check_voice(text: str) -> list[str]:
    fails = []
    count = len(_sentences(text))
    if not 1 <= count <= 3:
        fails.append(f"{count} sentences, prompt asks 1-3")
    left = [m for m in _MD_MARKERS if m in text]
    if left:
        fails.append(f"unflattened markup: {left[:3]}")
    if not text.strip():
        fails.append("empty")
    return fails


FORMAT_CHECKS: dict[str, Callable[[str], list[str]]] = {
    "title": _check_title,
    "capture": _check_capture,
    "facts": _check_facts,
    "focus": _check_focus,
    "voice": _check_voice,
}


def score(job: str, raw: str, anchors: list[str]) -> dict[str, Any]:
    """Composite 0-100 plus the two defect flags the decision reads."""
    text = (raw or "").strip()
    if not text:
        # An output that is not there cannot earn length or uniqueness
        # credit. Without this branch the arithmetic below hands an empty
        # response 40 points for being short and non-repetitive, which
        # turns a dead call into a middling result — the exact way a
        # zero-pinned metric ends up on a trend line.
        return {"composite": 0.0, "format_ok": False, "format_fails": ["empty output"],
                "anchor_recall": 0.0 if anchors else None, "duplicate_rate": 0.0,
                "over_long": False, "chars": 0, "defect": True}
    fails = FORMAT_CHECKS[job](text)
    recall = anchor_recall(text, anchors)
    dup = duplicate_rate(text)
    over = len(text) > LENGTH_BUDGETS[job]

    composite = 0.0
    composite += W_FORMAT if not fails else 0.0
    composite += W_ANCHORS * (recall or 0.0)
    composite += 0.0 if over else W_LENGTH
    composite += W_UNIQUE * (1.0 - min(dup, 1.0))
    return {
        "composite": round(composite, 2),
        "format_ok": not fails,
        "format_fails": fails,
        "anchor_recall": None if recall is None else round(recall, 3),
        "duplicate_rate": round(dup, 3),
        "over_long": over,
        "chars": len(text),
        "defect": bool(over or dup > 0.0),
    }


# ── The two arms, through one production code path ──────────────────────


class _BodyOnce:
    """A response whose single `read()` was already taken by the recorder."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, *_args: Any) -> bytes:
        return self._body

    def __enter__(self) -> "_BodyOnce":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


@contextlib.contextmanager
def _recording():
    """Tee the production call's response so `usage` survives.

    `app.secondary_models` returns only `choices[0].message.content` and
    throws the rest of the response away, including the token counts. The
    patch is on the module attribute for the duration of one call, and the
    eval is single-threaded, so exactly one request is in flight inside it.
    """
    rec: dict[str, Any] = {"output_tokens": None, "prompt_tokens": None, "raw": None,
                           "url": None}
    real = urllib.request.urlopen

    def spy(req: Any, *args: Any, **kwargs: Any) -> _BodyOnce:
        # Which engine answered is the one claim the whole comparison rests
        # on, and an alias override that silently failed would leave both
        # arms identical. Record the URL the production call actually used.
        rec["url"] = getattr(req, "full_url", None) or (req if isinstance(req, str) else None)
        resp = real(req, *args, **kwargs)
        body = resp.read()
        try:
            data = json.loads(body.decode("utf-8"))
            usage = data.get("usage") or {}
            rec["output_tokens"] = usage.get("completion_tokens")
            rec["prompt_tokens"] = usage.get("prompt_tokens")
            rec["raw"] = (((data.get("choices") or [{}])[0]
                           .get("message") or {}).get("content") or "")
        except (ValueError, KeyError, IndexError, AttributeError):
            rec["raw"] = None
        return _BodyOnce(body)

    urllib.request.urlopen = spy
    try:
        yield rec
    finally:
        urllib.request.urlopen = real


@contextlib.contextmanager
def _engine(job: str, alias: str):
    """Put `job` on `alias` for the duration of one trial, however it is routed now.

    The knob is the one production actually has: `JOBS_ON_PRIMARY` in
    `app/secondary_models.py`, the per-job flip this eval exists to recommend.
    The secondary arm lifts the job out of it, the primary arm puts it in, and
    the production function is then called completely unmodified — so both
    arms run the real router and the real prompt builder, and the only
    difference between them is the engine that answers.

    That is why this does not patch `resolve_model_alias` any more. Once a job
    could be pinned, an override at the alias resolver would have measured the
    *unpinned* router in the secondary arm and something the code never does in
    the primary one, and the two arms would have stopped being the same
    harness. `secondary_enabled: false` still outranks this in both arms
    (`resolve_model_alias` has the last word), which is what
    `arms_are_separate` then reports as both arms answering from one endpoint.

    The pair therefore means the same thing whichever jobs are pinned when the
    run starts, so a run taken to confirm a flip is comparable with the run
    that recommended it.
    """
    import app.secondary_models as sm

    original = sm.JOBS_ON_PRIMARY
    pinned = set(original)
    if alias == "primary":
        pinned.add(job)
    else:
        pinned.discard(job)
    sm.JOBS_ON_PRIMARY = frozenset(pinned)
    try:
        yield
    finally:
        sm.JOBS_ON_PRIMARY = original


def run_trial(job: str, item: dict[str, Any], alias: str) -> dict[str, Any]:
    """One output of one item on one engine, with wall seconds and tokens."""
    call = JOB_CALLS[job]
    with _recording() as rec, _engine(job, alias):
        started = time.perf_counter()
        error = None
        try:
            result = call(item["input"])
        except Exception as exc:  # a dead engine is a data point, not a crash
            result, error = None, f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - started

    raw = rec["raw"] if rec["raw"] else _as_text(result)
    row = {"job": job, "item": item["id"], "source_session": item.get("source_session"),
           "alias": alias, "endpoint": rec["url"], "wall_s": round(wall, 3),
           "output_tokens": rec["output_tokens"], "prompt_tokens": rec["prompt_tokens"],
           "error": error}
    row.update(score(job, raw, item.get("anchors") or []))
    row["raw"] = raw
    return row


def plan(repeats: int, jobs: list[str], items: list[dict[str, Any]]) -> list[tuple[str, dict, str]]:
    """Interleaved trial order: arm A then arm B for each item, every repeat.

    Ordering matters for interpretation. Back-to-back same-arm runs let one
    engine's warm prefix cache flatter it against the other's cold start,
    and the video this item came from shows exactly that kind of pairing
    mistaken for a model difference.
    """
    return [(job, item, alias)
            for _repeat in range(repeats)
            for job in jobs
            for item in items
            for alias in ("secondary", "primary")]


# ── Judge pass (reported, never decisive) ────────────────────────────────

_JUDGE_SYSTEM = (
    "You grade one machine output against the job it was asked to do and "
    "the source it was given. Reply with a single digit and nothing else. "
    "2 = does the job, nothing invented. 1 = does part of it, or adds "
    "something not in the source. 0 = wrong shape, empty, or invented."
)


def judge(job: str, item: dict[str, Any], output: str) -> Optional[int]:
    """One 0-2 pass by the primary engine. Off unless `--judge`.

    The grader is one of the arms, so it can only be read as corroboration
    of the mechanical score, never as the decision input.
    """
    if not output:
        return 0
    user = (f"JOB: {JOB_LABELS[job]}\n"
            f"SOURCE (excerpt): {str(item['input'])[:1500]}\n"
            f"OUTPUT:\n{output[:1500]}\n\nGrade 0, 1 or 2:")
    try:
        body = json.dumps({
            "model": _primary_model_name(),
            "messages": [{"role": "system", "content": _JUDGE_SYSTEM},
                         {"role": "user", "content": user}],
            "temperature": 0.0, "max_tokens": 4,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(_primary_url(), data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
    except Exception:
        return None
    found = re.search(r"[012]", text or "")
    return int(found.group()) if found else None


def _primary_endpoint() -> tuple[str, str]:
    """(chat_completions_url, served model name) for the primary alias.

    The served name, not the alias: vLLM answers for the model it loaded,
    and a request naming `primary` would be answered by nothing.
    """
    from app.config import _get_model_cfg, _resolve_model_name

    cfg = _get_model_cfg("primary") or {}
    base = cfg.get("base_url") or "http://127.0.0.1:8096"
    return f"{base.rstrip('/')}/v1/chat/completions", _resolve_model_name("primary")


def _primary_url() -> str:
    return _primary_endpoint()[0]


def _primary_model_name() -> str:
    return _primary_endpoint()[1]


# ── Aggregation and decision ─────────────────────────────────────────────


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per job, per arm: mean composite, spread, wall seconds, tokens."""
    out: dict[str, Any] = {}
    for job in sorted({r["job"] for r in rows}):
        per_arm: dict[str, Any] = {}
        for alias in ("secondary", "primary"):
            mine = [r for r in rows if r["job"] == job and r["alias"] == alias]
            if not mine:
                per_arm[alias] = None
                continue
            scores = [r["composite"] for r in mine]
            walls = [r["wall_s"] for r in mine]
            tokens = [r["output_tokens"] for r in mine if r["output_tokens"] is not None]
            defects = [r for r in mine if r["defect"]]
            graded = [r.get("judge") for r in mine if r.get("judge") is not None]
            per_item = Counter(r["item"] for r in mine)
            per_arm[alias] = {
                "n": len(mine),
                # `n` is a trial count and grows with the item set; `repeats_min`
                # is how many times any single input was replayed on this arm,
                # which is the only number that says anything about whether a
                # gap is spread or one unlucky output. A run over 20 items once
                # each has n=20 and no spread at all.
                "items": len(per_item),
                "repeats_min": min(per_item.values()),
                "repeats_max": max(per_item.values()),
                "score_mean": round(statistics.fmean(scores), 2),
                "score_min": round(min(scores), 2),
                "score_max": round(max(scores), 2),
                "score_spread": round(max(scores) - min(scores), 2),
                "score_stdev": round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0.0,
                "format_ok_rate": round(sum(r["format_ok"] for r in mine) / len(mine), 3),
                "defect_rate": round(len(defects) / len(mine), 3),
                "anchor_recall_mean": _mean_or_none([r["anchor_recall"] for r in mine]),
                "wall_s_mean": round(statistics.fmean(walls), 2),
                "wall_s_max": round(max(walls), 2),
                "output_tokens_mean": round(statistics.fmean(tokens), 1) if tokens else None,
                # Measured here, not from eval/measurements.json: that file
                # has the secondary at 9.01 tok/s, and this sweep saw 100+
                # (400 tokens in 2.98 s on a direct probe). A routing
                # decision quoted against a 40x cost that is not there is
                # not a decision.
                "output_tokens_per_s": (round(sum(tokens) / sum(
                    r["wall_s"] for r in mine if r["output_tokens"] is not None), 1)
                    if tokens and sum(r["wall_s"] for r in mine
                                       if r["output_tokens"] is not None) > 0 else None),
                "endpoints": sorted({r["endpoint"] for r in mine if r.get("endpoint")}),
                "errors": sum(1 for r in mine if r["error"]),
                "judge_mean": _mean_or_none(graded),
            }
        out[job] = per_arm
    return out


def arms_are_separate(summary: dict[str, Any]) -> tuple[bool, str]:
    """Did the two arms actually reach two different engines?

    The whole comparison is one alias override, and an override that did
    not take would produce two identical arms and a plausible-looking
    table. Checked from the URLs the recorded calls used, not from the
    label the run asked for.
    """
    for job, arms in summary.items():
        sec, pri = arms.get("secondary"), arms.get("primary")
        if not sec or not pri or not sec["endpoints"] or not pri["endpoints"]:
            continue
        overlap = set(sec["endpoints"]) & set(pri["endpoints"])
        if overlap:
            return False, f"{job}: both arms reached {sorted(overlap)}"
    return True, "every arm reached a distinct engine endpoint"


def _mean_or_none(values: list[Any]) -> Optional[float]:
    real = [v for v in values if v is not None]
    return round(statistics.fmean(real), 3) if real else None


def decide(job: str, arms: dict[str, Any]) -> dict[str, Any]:
    """Keep or flip one job, against the margins at the top of this file."""
    sec, pri = arms.get("secondary"), arms.get("primary")
    if not sec or not pri or not sec["n"] or not pri["n"]:
        return {"job": job, "decision": "insufficient_data",
                "reason": "both arms need trials before a call can be made"}
    # The spread is the whole point of the repeats: a same-prompt replay of
    # Opus 5 Max under Claude Code scored below plain Opus 5 under a
    # different harness in the source walkthrough, with no error bars to
    # show that it might have been noise. One sample of an input cannot tell
    # a flaky output from a worse engine, so it does not get to route anything.
    #
    # The floor is on `repeats_min` — times one input was replayed on an arm —
    # and never on `n`. The first version of this gate read `n`, so a pilot of
    # 4 items at 1 repeat each came in at n=4, walked straight past it, and
    # produced a routing recommendation off a single sample per input per arm.
    # That is the exact confusion the clause exists to forbid, and `n` still
    # grows with the item set, which is a different axis of evidence.
    if sec["repeats_min"] < MIN_REPEATS or pri["repeats_min"] < MIN_REPEATS:
        return {"job": job, "decision": "insufficient_data",
                "reason": (f"needs {MIN_REPEATS} interleaved repeats of the same input "
                           f"per arm, got secondary repeats_min={sec['repeats_min']} "
                           f"over {sec['items']} item(s), "
                           f"primary repeats_min={pri['repeats_min']} "
                           f"over {pri['items']} item(s)")}

    gap = round(pri["score_mean"] - sec["score_mean"], 2)
    defect_gap = round(sec["defect_rate"] - pri["defect_rate"], 3)
    latency_ratio = (round(sec["wall_s_mean"] / pri["wall_s_mean"], 1)
                     if pri["wall_s_mean"] else None)

    if gap <= KEEP_MARGIN_POINTS and defect_gap <= KEEP_DEFECT_MARGIN:
        decision = "keep_secondary"
        reason = (f"within tolerance: {gap:+.1f} pts vs a {KEEP_MARGIN_POINTS}-pt "
                  f"margin, defect {defect_gap:+.3f} vs a {KEEP_DEFECT_MARGIN:+.3f} margin, "
                  f"at {latency_ratio}x the primary's wall time")
    else:
        decision = "flip_to_primary"
        why = [f"{gap:+.1f} pts (margin {KEEP_MARGIN_POINTS})"] if gap > KEEP_MARGIN_POINTS else []
        if defect_gap > KEEP_DEFECT_MARGIN:
            why.append(f"defect rate {defect_gap:+.3f} (margin {KEEP_DEFECT_MARGIN:+.3f})")
        reason = ("outside tolerance: " + "; ".join(why)
                  + f" — the {latency_ratio}x latency cost is not buying parity")
    return {"job": job, "decision": decision, "reason": reason,
            "score_gap_primary_minus_secondary": gap,
            "defect_gap_secondary_minus_primary": defect_gap,
            "latency_ratio_secondary_over_primary": latency_ratio,
            "margin_points": KEEP_MARGIN_POINTS, "defect_margin": KEEP_DEFECT_MARGIN,
            "repeats_per_input": {"secondary": sec["repeats_min"],
                                  "primary": pri["repeats_min"]},
            "items_per_job": {"secondary": sec["items"], "primary": pri["items"]}}


# ── The routing decision, as a file the router is checked against ────────
#
# `decide()` returns a verdict and the previous round stopped there: the report
# said `flip_to_primary` for `title`, nothing in the tree changed, and the
# item's second verb — *then re-route* — never happened. So the verdict is now
# also written to a tracked YAML, and `app.secondary_models.JOBS_ON_PRIMARY` is
# asserted equal to its flips list by
# `tests/test_secondary_routing_eval.py`. A recommendation nobody executed and
# a router that contradicts the measurement both fail that test, and the report
# can no longer disagree with the code silently.


DECISIONS_NAME = "decisions.yaml"


def decisions_path(out_dir: Path = OUT_DIR) -> Path:
    return out_dir / DECISIONS_NAME


def write_decisions(decisions: list[dict[str, Any]], meta: dict[str, Any],
                    out_dir: Path = OUT_DIR) -> Path:
    """Record the per-job verdicts and the router they oblige.

    YAML, not JSON: `.gitignore:33` ignores `*.json` across the whole tree
    (personal data), so a JSON artifact of this could not be checked in, and
    the acceptance clause asks for a checked-in artifact.
    """
    import yaml

    flips = sorted(d["job"] for d in decisions if d["decision"] == "flip_to_primary")
    payload = {
        "generated_at": meta["started_at"],
        "repeats": meta["repeats"],
        "repeats_at_floor": meta["repeats_at_floor"],
        "arms_separate": meta["arms_separate"]["ok"],
        "tolerance": meta["tolerance"],
        # The router file must contain exactly this list. Empty is a real
        # answer: it means every job measured inside tolerance, so the
        # secondary keeps all five.
        "on_primary": flips,
        "jobs": {d["job"]: d for d in decisions},
    }
    path = decisions_path(out_dir)
    header = ("# The routing decision this measurement obliges (item #551).\n"
              "#\n"
              "# `on_primary` must equal `JOBS_ON_PRIMARY` in\n"
              "# `app/secondary_models.py`; a test asserts it. Written by\n"
              "# `eval/secondary_routing_eval.py`, edited by nobody — to change\n"
              "# a routing decision you re-run the eval, not this file.\n")
    path.write_text(header + yaml.safe_dump(payload, sort_keys=False, width=100),
                    encoding="utf-8")
    return path


def load_decisions(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """The recorded verdicts, or None when no decision-grade run has been recorded."""
    import yaml

    path = path or decisions_path()
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def router_matches_decisions(recorded: Optional[dict[str, Any]],
                            on_primary: frozenset) -> tuple[bool, str]:
    """Does the live router say what the last decision-grade run concluded?

    A missing record is a pass, not a failure: the file only exists once a
    sweep at `MIN_REPEATS` has actually been recorded, and the eval is usable
    before then. A *stale* record is not a pass — if the run was not at the
    repeat floor or the two arms were not separate, its flips are not a
    verdict, and a router built from it would be a guess with a citation.
    """
    if recorded is None:
        return True, "no decision recorded yet (eval has not been run at the repeat floor)"
    if not recorded.get("repeats_at_floor") or not recorded.get("arms_separate"):
        return False, ("the recorded run was not decision-grade "
                       f"(repeats_at_floor={recorded.get('repeats_at_floor')}, "
                       f"arms_separate={recorded.get('arms_separate')}), "
                       "so it cannot license a routing change")
    want = frozenset(recorded.get("on_primary") or [])
    if want == set(on_primary):
        return True, f"router and measurement agree: on_primary={sorted(want) or '[]'}"
    return False, (f"router says on_primary={sorted(on_primary) or '[]'} but the "
                   f"measurement says {sorted(want) or '[]'} — "
                   "either execute the flip or re-run the eval")


#: The router line this writes. Deliberately a whole assignment on one line so
#: the edit is one regex against one line, not an AST rewrite of production.
ROUTER_ATTR = "JOBS_ON_PRIMARY"
#: The file that holds the routing decision. Read and written as source text,
#: never by import: the checked-in file is the authority on what is pinned, and
#: a pin that exists only in a running process is not a routing decision.
ROUTER_PATH = ROOT / "app" / "secondary_models.py"


def router_source(path: Path) -> frozenset:
    """The pinned jobs as the router file states them.

    Reads the source, not the imported module, because the thing being checked
    is what a reader of the file sees: a pin that only exists in memory after
    some other code set it is not a routing decision.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = ([t.id for t in node.targets if isinstance(t, ast.Name)]
                     if isinstance(node, ast.Assign)
                     else ([node.target.id] if isinstance(node.target, ast.Name) else []))
            if ROUTER_ATTR in names:
                value = node.value
                # `literal_eval` accepts a bare `frozenset()` but not
                # `frozenset({'a'})`: a call with an argument is not a literal.
                # Unwrap it rather than drop to eval.
                if isinstance(value, ast.Call) and getattr(value.func, "id", "") in {
                        "frozenset", "set"} and len(value.args) <= 1:
                    # `frozenset()` alone is legal and means the empty set;
                    # `frozenset({'a'}) is not a literal, so unwrap its one arg.
                    value = value.args[0] if value.args else ast.Set(elts=[])
                return frozenset(ast.literal_eval(value))
    raise LookupError(f"{ROUTER_ATTR} is not a module-level assignment in {path}")


def set_router_pins(jobs, path: Path) -> Path:
    """Write the pinned-job set into the router source, and return the path.

    A flip this file did not measure is refused: the check is against
    `decisions.yaml`, which only a decision-grade run may write. Without the
    refusal the loop would carry a hand-edit that looked like a decision, which
    is the thing #551 was opened to stop.

    The loop lands this by running it, committing the one-line change to
    `app/secondary_models.py`, and re-running the eval — the next run's primary
    arm and this run's primary arm pin the same job the same way, so the pair
    stays comparable and `--confirm` can check the gain survived.
    """
    import re

    wanted = frozenset(jobs)
    unknown = sorted(set(wanted) - set(JOBS))
    if unknown:
        raise ValueError(f"not routable jobs: {unknown} (known: {list(JOBS)})")
    text = path.read_text(encoding="utf-8")
    inner = ", ".join(f"{j!r}" for j in sorted(wanted))
    line = (f"{ROUTER_ATTR}: frozenset = frozenset({{{inner}}})" if wanted
            else f"{ROUTER_ATTR}: frozenset = frozenset()")
    pattern = re.compile(rf"^{ROUTER_ATTR}: frozenset = frozenset\(.*\)$", re.MULTILINE)
    if not pattern.search(text):
        raise LookupError(f"{path} no longer states {ROUTER_ATTR} on one line; "
                          "update set_router_pins rather than hand-editing the router")
    path.write_text(pattern.sub(line, text), encoding="utf-8")
    assert set(router_source(path)) == wanted, "the router line did not take"
    return path


# ── Downstream check: the defect this item exists to fix ─────────────────


#: The heading `_append_daily_note` writes. Auto-captured entries are the
#: secondary's own output, so they are the only part of a daily note this
#: measurement may read — a note also holds the user's own prose and other
#: agents' entries, and scoring those would attribute a person's writing to an
#: engine.
DAILY_ENTRY_HEADING = "### Session "


def daily_entries(text: str) -> list[str]:
    """Each auto-captured entry in one daily note, body included.

    Deliberately not `app.post_capture.entry_window`, which stops at the next
    `### ` of any kind: a summary the secondary emits *contains* a `### ` when
    it disobeys its own no-markdown instruction, and truncating there would
    delete the exact over-long tail this counts.
    """
    marks = [i for i in range(len(text)) if text.startswith(DAILY_ENTRY_HEADING, i)]
    out = []
    for n, start in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(text)
        out.append(text[start:end].strip())
    return out


def capture_defects(text: str) -> dict[str, Any]:
    """Per-entry defect signals over a chunk of daily-note text.

    Same two signals as the offline scorer — over length and a repeated line
    inside one entry — because the clause compares like with like: the same
    `duplicate_rate` and the same character budget the eval scores `capture`
    against, applied to what actually shipped to the daily note.
    """
    entries = daily_entries(text)
    budget = LENGTH_BUDGETS["capture"]
    flagged = 0
    over = 0
    duplicated = 0
    for e in entries:
        body = e.split("\n", 1)[1] if "\n" in e else ""
        dups = duplicate_rate(body)
        long_ = len(body) > budget
        if dups > 0.0 or long_:
            flagged += 1
        over += bool(long_)
        duplicated += bool(dups > 0.0)
    return {"entries": len(entries),
            "flagged_rate": round(flagged / len(entries), 3) if entries else None,
            "over_long_rate": round(over / len(entries), 3) if entries else None,
            "duplicate_rate": round(duplicated / len(entries), 3) if entries else None}


def measure_capture_window(memory_dir: Path, days: list[str]) -> dict[str, Any]:
    """Defect rate over the named daily notes (ISO dates), read-only.

    `days` is chosen by the caller so the two 7-day windows the clause names
    are explicit and re-runnable rather than implied by 'now'. A missing note
    is counted in `missing_days`: a week with no captured sessions has no
    denominator, and reporting its rate as 0.0 would be the zero-denominator
    mistake this item's own history keeps cataloguing.
    """
    per_day = {}
    missing = []
    for day in days:
        path = memory_dir / f"{day}.md"
        per_day[day] = capture_defects(path.read_text(encoding="utf-8")) if path.exists() else None
        if not path.exists():
            missing.append(day)
    present = [d for d in per_day.values() if d]
    entries = sum(d["entries"] for d in present)
    return {"days": len(days), "missing_days": missing, "notes_read": len(present),
            "entries": entries,
            "flagged": sum(round((d["flagged_rate"] or 0) * d["entries"]) for d in present),
            "flagged_rate": (round(sum((d["flagged_rate"] or 0) * d["entries"] for d in present)
                                   / entries, 3) if entries else None),
            "over_long_rate": (round(sum((d["over_long_rate"] or 0) * d["entries"] for d in present)
                                     / entries, 3) if entries else None),
            "duplicate_rate": (round(sum((d["duplicate_rate"] or 0) * d["entries"] for d in present)
                                     / entries, 3) if entries else None),
            "per_day": per_day}


def window_dates(end_day: str, length: int = 7) -> list[str]:
    """`length` dates ending at, and including, `end_day`."""
    from datetime import date, timedelta

    last = date.fromisoformat(end_day)
    return [(last - timedelta(days=n)).isoformat()
            for n in range(length)][::-1]


def preceding_window(end_day: str, length: int = 7) -> list[str]:
    """The `length` dates immediately before `window_dates(end_day, length)`.

    Non-overlapping and adjacent by construction — the clause's check is 7
    days against the prior 7, and a window that quietly shared a day with its
    own baseline would compare a week with itself.
    """
    from datetime import date, timedelta

    first = date.fromisoformat(window_dates(end_day, length)[0])
    return window_dates((first - timedelta(days=1)).isoformat(), length)


# ── Item set ─────────────────────────────────────────────────────────────




#: Words every session contains. Frequency without this filter just returns
#: the transcript's grammar.
_STOPWORDS = frozenset("""the and for with that this from they you are was were have has had
their about which what when where while because these those into over under not but all any
can will would could should there here then than once some more most other your ours their
""".split())


def anchors_from(text: str, limit: int = 6) -> list[str]:
    """The subject words a correct output must still be talking about.

    This matched paths, symbols and flags first — the obvious groundedness
    signal — and the first sweep falsified it. Anchors came out as
    `/home/…/bg-20260908-192331-4acd31.log` and `task_id`, because the
    harness injects background-task log paths into a transcript, and both
    arms then scored 0.0 recall on summaries that named the session's actual
    subjects correctly. A signal that is zero on output a human calls good
    does not discriminate, and 30 composite points riding on it turns the
    comparison into noise. Anchors are therefore the most repeated
    non-grammatical words of the recorded input: not a proof of correctness,
    a floor under drift, and the only candidate that came back non-zero on
    both arms' good outputs.

    A word that appears once is not an anchor: making the model quote back a
    hapax is a transcription test, not a groundedness one.
    """
    counts: dict[str, int] = {}
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text or ""):
        low = word.lower()
        if low not in _STOPWORDS:
            counts[low] = counts.get(low, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, count in ranked if count >= 2][:limit]

#: `build_items` refuses an input with fewer than this, rather than pinning
#: an item 30 points of its score cannot be computed for. The alternative —
#: padding with whatever long words the text had — would score noise.
MIN_ANCHORS = 2


def _anchor_variants(anchor: str) -> list[str]:
    """The forms in which an output may legitimately name an anchor.

    `app/post_capture.py` is named by the path, by its basename, and by
    `post_capture` — an output that says "post_capture kept timing out"
    grounded, and scoring it as a miss would punish paraphrase, which is
    exactly what a good summary does with a path.
    """
    low = _norm(anchor)
    variants = [low, low.rsplit("/", 1)[-1]]
    for part in re.split(r"[/._-]", low):
        if len(part) >= 6 and part not in _STOPWORDS:
            variants.append(part)
    return [v for v in dict.fromkeys(variants) if v]


def _anchor_hits(text: str, anchors: list[str]) -> int:
    hay = _norm(text)
    return sum(1 for anchor in anchors
               if any(variant in hay for variant in _anchor_variants(anchor)))


def focus_transcript(messages: list[dict], window: int = 10, cap: int = 200) -> str:
    """The transcript `app/post_capture.py:129` `_maybe_extract_focus` builds.

    Replicated rather than imported because that builder is inline in an
    async function that also reads a session off disk and writes focus
    state; extracting it is a production refactor this item does not
    authorise. The shape it must keep — last `window` turns, `USER:` /
    `ASSISTANT:` labels, `cap` chars per turn, context blocks dropped — is
    pinned by tests/test_secondary_routing_eval.py.
    """
    recent = [m for m in messages[-window:] if m.get("role") in ("user", "assistant")]
    lines: list[str] = []
    for msg in recent:
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text")
        elif isinstance(content, str):
            text = content
        else:
            continue
        stripped = text.strip()
        if not stripped or any(stripped.startswith(p) for p in
                               ("<context>", "<system-reminder>", "<memory>", "<daily_notes>")):
            continue
        label = "USER" if msg.get("role") == "user" else "ASSISTANT"
        lines.append(f"{label}: {stripped[:cap]}")
    return "\n".join(lines)


#: Sessions older than this are finished conversations, which is what makes
#: a pinned input hash mean something: an item whose session is still taking
#: turns would rebuild to a different string every run.
MIN_SESSION_AGE_HOURS = 48

LIVE_SESSIONS_DIR = Path.home() / "lloyd" / "sessions"


def job_inputs(job: str, data: dict[str, Any]) -> str:
    """The exact string production would have sent for `job`.

    Each builder is the one the routed call site uses; `focus` is the
    replica documented on `focus_transcript`. This is the single place that
    knows how an item's input is made, so `--build-items` (which pins the
    hash) and the run (which verifies it) cannot drift apart.
    """
    from app.post_capture import _build_capture_transcript
    from app.session_titles import build_transcript

    messages = data.get("messages") or []
    if job == "title":
        return build_transcript(data).strip()
    if job in ("capture", "facts"):
        return _build_capture_transcript(messages).strip()
    if job == "focus":
        return focus_transcript(messages)
    if job == "voice":
        return _first_unspeakable_reply(messages)
    raise KeyError(f"no input builder for job {job!r}")


def _first_unspeakable_reply(messages: list[dict]) -> str:
    """A real reply the voice route would actually rewrite.

    `app/routers/voice.py` skips the secondary for text
    `_is_trivially_speakable` accepts, so an item drawn from the plain-prose
    replies would never have been this job's work.
    """
    from app.secondary_models import _is_trivially_speakable

    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        text = content if isinstance(content, str) else " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text")
        text = text.strip()
        if text and not _is_trivially_speakable(text) and len(text) <= 1500:
            return text
    return ""


def build_items(per_job: int = 4, sessions_dir: Optional[Path] = None,
                min_age_hours: float = MIN_SESSION_AGE_HOURS) -> dict[str, list[dict]]:
    """Pin the item set from finished user sessions. `--build-items`.

    The pinned record names the session and the hash of its rebuilt input
    rather than carrying the text: this repo has a remote, and a routed
    transcript is a record of a person's day. The input is rebuilt from the
    local session store at run time and compared against the hash, which
    keeps "checked into eval/" reproducible while making a changed input a
    hard error instead of a silently different measurement.
    """
    import hashlib
    import time

    from app.sessions_io import is_user_session

    sessions_dir = sessions_dir or LIVE_SESSIONS_DIR
    cutoff = time.time() - min_age_hours * 3600
    paths = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime,
                   reverse=True)

    items: dict[str, list[dict]] = {job: [] for job in JOBS}
    used: set[str] = set()

    for path in paths:
        if all(len(items[j]) >= per_job for j in JOBS):
            break
        if path.stat().st_mtime > cutoff:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not is_user_session(data):
            continue
        session_id = data.get("session_id") or path.stem
        if session_id in used:
            continue
        if len([m for m in (data.get("messages") or [])
                if m.get("role") == "user"]) < 3:
            continue
        used.add(session_id)

        for job in JOBS:
            if len(items[job]) >= per_job:
                continue
            text = job_inputs(job, data)
            if len(text) < (200 if job in ("capture", "facts") else 50):
                continue
            anchors = anchors_from(text)
            if len(anchors) < MIN_ANCHORS:
                # Nothing to check an output against: 30 of the composite
                # points would be uncomputable for every one of its trials.
                continue
            items[job].append({
                "id": f"{job}-{len(items[job]) + 1}", "job": job,
                "source_session": session_id,
                "input_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                "input_chars": len(text),
                "anchors": anchors_from(text),
            })
    return items


def resolve_input(item: dict[str, Any], sessions_dir: Path) -> str:
    """Rebuild one item's input and prove it is still the pinned string."""
    import hashlib

    path = sessions_dir / f"{item['source_session']}.json"
    if not path.exists():
        raise SystemExit(
            f"item {item['id']}: source session {item['source_session']} is not in "
            f"{sessions_dir}; the item set is real recorded input, so there is nothing "
            f"to substitute — re-pin with --build-items")
    text = job_inputs(item["job"], json.loads(path.read_text(encoding="utf-8")))
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    if digest != item["input_sha256"]:
        raise SystemExit(
            f"item {item['id']}: rebuilt input is {digest}, pinned "
            f"{item['input_sha256']} ({item['input_chars']} chars). The session moved, "
            f"or a transcript builder changed. Re-run --build-items and say which.")
    return text


def load_items(path: Path) -> dict[str, list[dict]]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = {job: list((raw.get(job) or {}).get("items") or []) for job in JOBS}
    if not any(items.values()):
        raise SystemExit(f"no items in {path}; run --build-items first")
    return items


def write_items(items: dict[str, list[dict]], path: Path) -> None:
    import yaml

    header = (
        "# Item set for the secondary-engine routing eval (item #551).\n"
        "#\n"
        "# Each item names a finished user session and the sha256 of the input\n"
        "# that the routed job would have sent it. The input itself is not\n"
        "# checked in: it is a record of a person's day, and this repo has a\n"
        "# remote. `eval/secondary_routing_eval.py` rebuilds it from the local\n"
        "# session store with the production transcript builder and refuses to\n"
        "# run when the hash moves, so a stale item is an error rather than a\n"
        "# different measurement. Re-pin with `--build-items N`.\n"
        "#\n"
        "# `anchors` are identifying tokens (paths, identifiers, numbers) lifted\n"
        "# from that input; the scorer's groundedness floor is how many of them\n"
        "# the output names.\n"
    )
    body = {job: {"label": JOB_LABELS[job], "items": items[job]} for job in JOBS}
    path.write_text(header + yaml.safe_dump(body, sort_keys=False, width=100),
                    encoding="utf-8")


# ── Report ───────────────────────────────────────────────────────────────


def render_report(summary: dict[str, Any], decisions: list[dict[str, Any]],
                  meta: dict[str, Any]) -> str:
    lines = [
        "# Secondary engine: quality on the work it is routed to (item #551)",
        "",
        f"- run: `{meta['started_at']}` · repeats: {meta['repeats']} · "
        f"items per job: {meta['items_per_job']} · arms: secondary vs primary",
        f"- tolerance: secondary keeps a job within **{KEEP_MARGIN_POINTS}** composite "
        f"points and **{KEEP_DEFECT_MARGIN:+.2f}** defect rate of the primary",
        f"- judge pass: {'on (reported only, never decisive)' if meta['judge'] else 'off'}",
        "",
        "Composite = format compliance (30) + anchor recall (30) + "
        "length budget (25) + uniqueness (15). Defect = over-long or a "
        "near-duplicate line inside one output.",
        "",
        f"- arms separate: {meta['arms_separate']['ok']} ({meta['arms_separate']['detail']})",
        f"- router agrees with this measurement: {meta['router_agreement']['ok']} "
        f"({meta['router_agreement']['detail']})",
        "- repeats are warm: repeat 2 and 3 replay a prompt the engine has already "
        "cached, so wall seconds favour the later repeats equally in both arms and "
        "`max` is the cold one. The primary's hit rate is on the dashboard "
        "(`prefix_cache_hit_rate`); the secondary's is reported `null` because "
        "llama.cpp's cached-token counter cannot be turned into a rate (f7028ce).",
        "- `n` is trials, `items × repeats` is inputs replayed: the decision floor is "
        f"on repeats (`{MIN_REPEATS}`), not on `n`, so a wide run of one repeat each "
        "still cannot route anything.",
        "",
    ]
    for job in [j for j in JOBS if j in summary]:
        arms = summary[job]
        decision = next((d for d in decisions if d["job"] == job), {})
        lines += [f"## {JOB_LABELS[job]}  (`{job}`)", "",
                  "| arm | n | items × repeats | score mean ± spread | format ok "
                  "| defect rate | wall s | out tok | tok/s |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for alias in ("secondary", "primary"):
            arm = arms.get(alias)
            if not arm:
                lines.append(f"| {alias} | 0 | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {alias} | {arm['n']} | {arm['items']} × {arm['repeats_min']} "
                f"| {arm['score_mean']} ± {arm['score_spread']} "
                f"(σ {arm['score_stdev']}) | {arm['format_ok_rate']:.0%} "
                f"| {arm['defect_rate']:.0%} | {arm['wall_s_mean']} (max {arm['wall_s_max']}) "
                f"| {arm['output_tokens_mean']} | {arm.get('output_tokens_per_s')} |")
        lines += ["", f"**Decision: `{decision.get('decision', 'insufficient_data')}`** — "
                      f"{decision.get('reason', '')}", ""]
        for alias in ("secondary", "primary"):
            arm = arms.get(alias)
            if arm and arm.get("judge_mean") is not None:
                lines.append(f"- judge (primary engine, corroboration only): "
                             f"{alias} mean {arm['judge_mean']}/2")
        lines.append("")

    flips = [d["job"] for d in decisions if d["decision"] == "flip_to_primary"]
    lines += ["## Routing", "",
              f"- on primary now (`app.secondary_models.JOBS_ON_PRIMARY`): "
              f"`{sorted(meta['on_primary_now']) or '[]'}`",
              f"- this run recommends on primary: `{sorted(flips) or '[]'}`",
              f"- router agrees with the measurement: **{meta['router_agreement']['ok']}** — "
              f"{meta['router_agreement']['detail']}",
              "- a flip is executed by adding the job to `JOBS_ON_PRIMARY` in "
              "`app/secondary_models.py` and re-running this eval: the secondary arm "
              "lifts the pin and the primary arm sets it, so both arms keep measuring "
              "the same router and a confirmation run stays comparable with the one "
              "that recommended the flip.",
              "- after a flip, the downstream check is one command: "
              "`python3 eval/secondary_routing_eval.py --downstream <YYYY-MM-DD>` — "
              "the over-long/duplicate rate over the 7 days ending that date against "
              "the 7 before it, read from the auto-captured entries in the daily notes.",
              "",
              "---", "",
              "Interpretation guard: the composite is mechanical, so it measures shape "
              "and grounding, not truth. Use the judge lines as corroboration only — "
              "the video this came from showed a harness pairing beating a bigger "
              "model, so a single number is not a winner. The latency ratio in each "
              "decision is the price being paid for whatever parity is claimed.",
              ""]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3,
                        help="interleaved repeats per item per arm (>=3 for a decision)")
    parser.add_argument("--jobs", default=",".join(JOBS), help="comma list to restrict the run")
    parser.add_argument("--items", default=str(ITEMS_PATH))
    parser.add_argument("--sessions-dir", default=str(LIVE_SESSIONS_DIR),
                        help="where the pinned source sessions live (items are rebuilt "
                             "from these, and the eval refuses a hash mismatch)")
    parser.add_argument("--build-items", type=int, default=0, metavar="N",
                        help="re-pin N items per job from live sessions, then exit")
    parser.add_argument("--downstream", metavar="END_DATE", default=None,
                        help="skip the sweep and measure the daily-note defect rate over "
                             "the 7 days ending END_DATE against the 7 before it. The "
                             "clause's downstream check, runnable by one command once a "
                             "job has flipped; prints both windows and exits.")
    parser.add_argument("--memory-dir", default=str(Path.home() / "obsidian" / "memory"),
                        help="daily notes for --downstream")
    parser.add_argument("--pin", metavar="JOB[,JOB]", default=None,
                        help="write the named job(s) into app.secondary_models.JOBS_ON_PRIMARY. "
                             "Refused until a checked-in decision artifact exists that "
                             "recommends exactly them, which is what keeps a routing change "
                             "from being hand-edited. The loop's own route for landing this.")
    parser.add_argument("--print-pins", action="store_true",
                        help="print the router's current per-job pins and exit")
    parser.add_argument("--write-decisions", action="store_true",
                        help="load the last decision-grade artifact and write it to "
                             "eval/secondary-routing/decisions.yaml without re-running any engine")
    parser.add_argument("--judge", action="store_true",
                        help="add the 0-2 primary-engine pass (reported, not decisive)")
    parser.add_argument("--confirm", action="store_true",
                        help="after a pin, re-read the named run's own decisions against the "
                             "router's current pins: the clause's 're-run the pair to "
                             "confirm the gain survives on the same items', checked.")
    parser.add_argument("--label", default="latest")
    parser.add_argument("--budget-seconds", type=float, default=2400.0,
                        help="stop the sweep and report what completed; the secondary "
                             "also serves voice summaries, so a runaway eval is a outage")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    sessions_dir = Path(args.sessions_dir)

    if args.build_items:
        items = build_items(per_job=args.build_items, sessions_dir=sessions_dir)
        write_items(items, Path(args.items))
        print(f"pinned { {j: len(items[j]) for j in JOBS} } -> {args.items}")
        return 0

    if args.print_pins:
        print(json.dumps({"on_primary": sorted(secondary_models.JOBS_ON_PRIMARY)}))
        return 0

    if args.pin is not None:
        wanted = sorted({j.strip() for j in args.pin.split(",") if j.strip()})
        bad = [j for j in wanted if j not in JOBS]
        if bad:
            print(f"not routable jobs: {bad} (known: {list(JOBS)})")
            return 2
        recorded = load_decisions()
        if not recorded:
            print("refused: no checked-in decision artifact. Run the sweep so it writes "
                  f"{decisions_path()} first — a pin has to be a measured one, and an "
                  "empty decision file is a decision, so an unmeasured pin is refused.")
            return 3
        recommended = sorted(recorded["on_primary"])
        if wanted != recommended:
            print(f"refused: the measured decisions recommend {recommended}, not {wanted}. "
                  "A pin this eval did not earn is the defect #551 exists to remove.")
            return 3
        path = set_router_pins(wanted, ROUTER_PATH)
        print(f"wrote {path} — app.secondary_models.JOBS_ON_PRIMARY is now {wanted}. "
              "Commit it, then confirm the flip with --confirm --label <the new run>.")
        return 0

    if args.write_decisions:
        latest = OUT_DIR / "results-latest.json"
        if not latest.exists():
            print(f"no {latest} to read decisions from")
            return 3
        payload = json.loads(latest.read_text(encoding="utf-8"))
        if not payload["meta"].get("decision_grade"):
            print("refused: the last run was not decision-grade, so it recommends nothing")
            return 3
        path = write_decisions(payload["decisions"], payload.get("meta", {}), OUT_DIR)
        print(f"wrote {path} — app.secondary_models.JOBS_ON_PRIMARY must now equal "
              f"{sorted(d['job'] for d in payload['decisions'] if d['decision'] == 'flip_to_primary')}")
        return 0

    if args.confirm:
        run = OUT_DIR / f"results-{args.label}.json"
        if not run.exists():
            print(f"nothing to confirm: {run} does not exist. Run the sweep with "
                  f"--label {args.label} first, or point --label at the confirming run.")
            return 3
        payload = json.loads(run.read_text(encoding="utf-8"))
        agree, why = router_matches_decisions(
            {"on_primary": sorted(d["job"] for d in payload["decisions"]
                                  if d["decision"] == "flip_to_primary"),
             "repeats_at_floor": payload["meta"]["repeats_at_floor"],
             "arms_separate": payload["meta"]["arms_separate"]["ok"]},
            router_source(ROUTER_PATH))
        print(json.dumps({"confirmed": agree,
                          "on_primary": sorted(secondary_models.JOBS_ON_PRIMARY),
                          "that_run_recommends": sorted(
                              d["job"] for d in payload["decisions"]
                              if d["decision"] == "flip_to_primary"),
                          "detail": why, "run": str(run)}, indent=2))
        return 0 if agree else 3

    if args.downstream:
        memory_dir = Path(args.memory_dir)
        after = window_dates(args.downstream)
        before = preceding_window(args.downstream)
        result = {"after": measure_capture_window(memory_dir, after),
                  "before": measure_capture_window(memory_dir, before),
                  "window_days": len(after), "memory_dir": str(memory_dir)}
        print(json.dumps(result, indent=2))
        for arm, days in (("after", after), ("before", before)):
            w = result[arm]
            print(f"{arm:6s} {days[0]}..{days[-1]}: {w['entries']} entries over "
                  f"{w['notes_read']}/{w['days']} notes, flagged "
                  f"{w['flagged_rate']}, over-long {w['over_long_rate']}, "
                  f"duplicate {w['duplicate_rate']}")
        a, b = result["after"], result["before"]
        if a["entries"] == 0 or b["entries"] == 0:
            # The seventh instance of the class this item keeps hitting: a
            # rate over no denominator reads as a result.
            print("NO DOWNSTREAM VERDICT: a window has 0 captured entries")
            return 5
        return 0

    items = load_items(Path(args.items))
    jobs = [j for j in args.jobs.split(",") if j in JOBS]
    missing = [j for j in jobs if not items[j]]
    if missing:
        print(f"no items pinned for {missing} — run --build-items or drop them from --jobs")
        return 2

    # Rebuild every input from its source session and prove the hash before
    # a single engine call, so a moved session costs nothing.
    for job in jobs:
        for item in items[job]:
            item["input"] = resolve_input(item, sessions_dir)

    trials = plan(args.repeats, jobs, [i for j in jobs for i in items[j]])
    # `plan` interleaves across items of every job; keep only this job's items.
    wanted = {j: {i["id"] for i in items[j]} for j in jobs}
    trials = [t for t in trials if t[1]["id"] in wanted[t[0]]]

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.monotonic()
    rows: list[dict[str, Any]] = []
    for index, (job, item, alias) in enumerate(trials, 1):
        row = run_trial(job, item, alias)
        if args.judge:
            row["judge"] = judge(job, item, row["raw"])
        rows.append(row)
        if not args.quiet:
            print(f"[{index}/{len(trials)}] {job:8s} {item['id']:12s} {alias:9s} "
                  f"score={row['composite']:6.2f} wall={row['wall_s']:7.2f}s "
                  f"tok={row['output_tokens']}"
                  + (f" ERR {row['error']}" if row["error"] else ""), flush=True)
        row.pop("raw", None)
        if time.monotonic() - t0 > args.budget_seconds:
            print(f"budget of {args.budget_seconds}s reached; reporting partial")
            break

    summary = summarise(rows)
    separate, why = arms_are_separate(summary)
    decisions = [decide(job, summary[job]) for job in jobs if job in summary]
    if not separate:
        # An alias override that silently did not take would leave two
        # identical arms and a table that still reads like a result.
        for row in decisions:
            row["decision"] = "insufficient_data"
            row["reason"] = f"arms not separate — {why}"
    # Decision-grade is what the arms actually contain, not what was asked for:
    # a run that hit the budget mid-sweep asked for 3 repeats and may have
    # landed 1 on some items, and `decide` reads `repeats_min` for the same
    # reason. Deriving it from the request is how the previous round's 1-repeat
    # pilot came out looking like it had earned a routing recommendation.
    arms_repeats = [a["repeats_min"] for job in summary for a in summary[job].values() if a]
    at_floor = bool(arms_repeats) and min(arms_repeats) >= MIN_REPEATS
    router_ok, router_why = router_matches_decisions(
        load_decisions(), router_source(ROUTER_PATH))
    meta = {"started_at": started, "repeats": args.repeats,
            "items_per_job": {j: len(items[j]) for j in jobs},
            "item_provenance": {j: [{"id": i["id"], "source_session": i["source_session"],
                                     "input_sha256": i["input_sha256"]} for i in items[j]]
                                for j in jobs},
            "sessions_dir": str(sessions_dir),
            "arms_separate": {"ok": separate, "detail": why},
            "router_agreement": {"ok": router_ok, "detail": router_why},
            "on_primary_now": sorted(router_source(ROUTER_PATH)),
            "repeats_at_floor": at_floor,
            # Named separately from `repeats_at_floor` because it also carries
            # arm separation: `--write-decisions` trusts this flag alone.
            "decision_grade": bool(separate and at_floor),
            "judge": args.judge, "elapsed_s": round(time.monotonic() - t0, 1),
            "trials": len(rows), "tolerance": {"margin_points": KEEP_MARGIN_POINTS,
                                               "defect_margin": KEEP_DEFECT_MARGIN}}

    OUT_DIR.mkdir(exist_ok=True)
    payload = {"meta": meta, "summary": summary, "decisions": decisions, "trials": rows}
    json_path = OUT_DIR / f"results-{args.label}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = render_report(summary, decisions, meta)
    md_path = OUT_DIR / f"REPORT-{args.label}.md"
    md_path.write_text(report, encoding="utf-8")
    decision_paths = []
    if separate and at_floor:
        # Only a decision-grade run is allowed to move the router. A pilot
        # writes nothing, so a partial sweep cannot leave a routing claim
        # behind it.
        decision_paths.append(write_decisions(decisions, meta))

    print()
    print(report)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for p in decision_paths:
        print(f"wrote {p} — app.secondary_models.JOBS_ON_PRIMARY must now equal "
              f"{sorted(frozenset(d['job'] for d in decisions if d['decision'] == 'flip_to_primary'))}")
    if not router_ok:
        print(f"ROUTER DISAGREES WITH THE MEASUREMENT: {router_why}")
    if not separate:
        print(f"NOT DECISION GRADE: {why}")
        return 3
    if not at_floor:
        print(f"NOT DECISION GRADE: {min(arms_repeats) if arms_repeats else 0} repeat(s) "
              f"per input per arm, {MIN_REPEATS} required")
        return 4
    if not router_ok:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
