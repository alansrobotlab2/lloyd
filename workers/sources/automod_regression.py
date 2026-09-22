"""Nightly retrieval-quality regression check for a landed self-modification.

**The axis, said first.** Retrieval quality, not behaviour. Every metric this
file arms scores a retrieval query, and none of them observes a tool call, a
turn count or a model decision — see the limit beside the edge-set limit under
`ARMED_METRICS`, and `architecture/automod.md` §13.

**What this deliberately does not do.** It does not compare the autoresearch
composite score. Three identical baseline runs of that metric scored
0.719 / 0.542 / 0.624 — a spread of 0.177 against a promotion threshold of
0.05 — and 61 of 83 historical promotions were decided inside that noise
(see the docstring of `tests/test_autoresearch_promotion.py`). A detector
built on it would fire on sampling noise and be switched off within a week.

**What it does instead.** `eval/run_eval.py` has no LLM in it — it calls
`agent_mcp.vault._vault_recall` directly and scores against a fixed YAML.
Measured on this machine, five consecutive runs against an unchanged vault
produced *identical* values for every quality metric (stdev 0.0000 for
entity_hit_rate, entity_recall_avg, fact_entity_recall_avg, ndcg10, mrr_doc,
doc_hit_rate, doc_recall_avg); only latency varied. So the eval itself
contributes no noise at all, and any movement in those numbers is signal.

**The real confound is vault drift, not measurement noise.** Cross-day
baselines from 09-03/04/05 differed by up to 0.044 in ndcg10 — but the vault
changed underneath them. Comparing today's number against one recorded at the
last promotion would therefore measure how much the vault moved, not what the
code change did.

So the comparison is a **paired A/B on identical data**: check the
last-known-good commit out into a scratch worktree, point it at the *live*
fact tree and knowledge graph via `LLOYD_FACTS_ROOT` / `LLOYD_KG_DB` (which
exist for exactly this purpose), and run both arms in the same window. Drift
cancels; what is left is the code.

A missing noise file means "cannot evaluate", never "no regression".
`eval/baselines/` is gitignored, so it is untracked runtime state that can
simply be absent.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.djev import REPLAY_FAILURES, replay_env, replay_stats
from scripts.automod.evalpin import PinError, PinnedCorpus
from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.automod-regression")

NAME = "automod-regression"
DEFAULT_PRIORITY = 70
DEDUP_KEY = "automod:regression"

# The retriever's qmd client gives up at 15 s, which is production's latency
# budget and the wrong number for a measurement: an eval arm that loses one
# answer to the clock is "cannot evaluate", and one that loses all of them used
# to score zero. `agent_mcp.vault._qmd_post` reads this at call time; an arm
# whose code predates the override simply keeps the 15 s it always had.
QMD_TIMEOUT_ENV = "LLOYD_QMD_TIMEOUT_S"
EVAL_QMD_TIMEOUT_S = 60

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
# Both arms score THESE questions, whichever commit's code is running.
LIVE_QUERIES = LIVE_ROOT / "eval" / "vault_recall_queries.yaml"

# Deliberately NOT under eval/baselines/. That directory holds eval RUN
# RECORDS, and `tests/test_eval_scorer.py` globs `*.json` there and asserts
# every file carries a run record's fields — so parking a noise summary in it
# fails an unrelated test. It also belongs with the loop's other runtime state,
# which lives outside the repo so it survives a rollback.
NOISE_PATH = Path(os.environ.get(
    "LLOYD_AUTOMOD_STATE",
    Path.home() / ".local" / "state" / "lloyd-automod")) / "eval-noise.json"

# ARMED = the metrics this comparison actually CONTROLS.
#
# This set was wrong twice, in opposite directions, and both corrections were
# measured rather than reasoned.
#
# First it held all seven, chosen because five consecutive runs gave stdev
# 0.0000 for every one. Real measurement, wrong experiment: it describes
# repeatability inside one short window, and the paired A/B runs its arms
# minutes apart with a worktree checkout between them. The first real run
# demanded a rollback of a promotion whose entire diff was text inside an
# inject string, on ndcg10 -0.0060 and mrr_doc -0.0060. The four document
# metrics were disarmed, which left every graph-to-document change
# unfalsifiable.
#
# The cause was not drift, and pinning the qmd corpus alone did not fix it:
# the arms still differed by exactly -0.0060. **This retriever greps the
# repository it ships in.** `_grep_lloyd_code` searches `agent_mcp/`, `app/`,
# `scripts/` and `workers/` under its own checkout, so the code under test is
# also part of the corpus, and each arm was searching its own source. Measured
# 2026-09-07, the query `lloyd-vllm-rel` returned six different files per arm.
#
# With BOTH halves pinned — a frozen qmd snapshot and one shared
# `LLOYD_CODE_ROOT` — the two arms agree to 0.0000 on all seven, and four
# repeat runs under the same pin move 0.0000 on all seven. So all seven are
# armed, and the pin is a precondition rather than an optimisation:
# `execute` refuses to compare without it instead of falling back to the live
# daemon, which would be the old broken comparison wearing the new name.
#
# `latency_ms_avg` is the one thing that still moves, and it moves for a reason
# that makes a *relative* comparison impossible: the qmd daemon caches query
# embeddings, so the same text re-runs ~20x faster than an unseen one. It is
# therefore the one metric compared ABSOLUTELY, against a per-context budget
# (`LATENCY_BUDGET_MS` below), and never relatively against another arm.
ARMED_METRICS = ("entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg",
                 "ndcg10", "mrr_doc", "doc_hit_rate", "doc_recall_avg")
# Reported metrics: never scored against another run's value. `latency_ms_avg`
# is still in this tuple — it has no tolerance and cannot make a comparison
# regress — but it is no longer merely recorded: it is compared against
# `LATENCY_BUDGET_MS`, which yields `over_budget` (see `evaluate`).
REPORT_ONLY = ("latency_ms_avg", "n_queries")

# What the armed set can and cannot see, MEASURED rather than assumed.
#
# Measured 2026-09-06 against an empty LLOYD_FACTS_ROOT/LLOYD_KG_DB: with the
# graph deleted entirely, mrr_doc (0.468), ndcg10 (0.563), doc_hit_rate (0.90)
# and every per-category mrr_doc came back IDENTICAL to the real run. Two
# reasons — `RECALL_GRAPH_RERANK` is False, so the graph never reorders
# documents, and the document leg queries the qmd daemon over an absolute URL
# that keeps working regardless. The doc-side numbers are real measurements of
# the document retriever and say nothing whatsoever about graph quality.
#
# Two degradation drills on COPIES of the live store, 2026-09-06:
#
#   70% of fact_idx rows deleted (205,689 -> 61,707)
#       entity_hit_rate    0.6000 -> 0.5500   FIRES
#       entity_recall_avg  0.4430 -> 0.4330   FIRES
#       every doc-side metric                 unchanged
#
#   70% of ACTIVE EDGES expired (4,029 -> 1,209)
#       every metric, armed and reported      unchanged  <-- blind
#
# So the name matters. These three read the fact tree and the fact index; they
# do NOT read graph edges, and a change that expires most of the edge set walks
# past this check in silence. Calling them "graph sensitive" would be the exact
# overclaim this whole detector is meant to avoid — a check that reports
# healthy because it was never looking. Edge quality has no armed metric and is
# a stated limit in architecture/automod.md §13, not a covered case.
#
# THE SAME STATEMENT, ONE AXIS OVER — the agent loop. No armed metric observes a
# tool call, a turn count or a model decision: `eval/run_eval.py` issues no model
# request at all (it calls `agent_mcp.vault._vault_recall` directly and scores
# against a fixed YAML), so a landed change to the agent loop — dropping a tool
# from the set, changing turn accounting, compaction, or skill injection — moves
# none of the seven, and this check reports green by construction. Name the
# disproof: dropping `Grep` from the baseline tool set changes no metric here.
#
# This is NOT "nothing observes the loop". The gate has a scored loop-side rung,
# `prompt_surface` (`scripts/automod/gate.py`, `rung_prompt_surface`), which runs
# `eval/run_tool_choice_eval.py` then `eval/compare_tool_choice.py` — but ONLY
# pre-landing, and ONLY when the diff touches one of five path names:
# `prompt_builder.py`, `prefetch.py`, and the vault `SOUL.md`, `MEMORY.md`,
# `USER.md`. So a loop-side change that is not one of those five has no
# pre-landing check and no post-landing one. `AXIS_*` below carries that into the
# measurement itself, because this file's output lands in the LKG record's `eval`
# slot, where every later "observed healthy in production" claim reads it.
#
# `test_automod_doc_claims` pins the split. The four metrics outside this
# subset are armed too, and the pin is what makes that honest: `PinnedCorpus`
# (`scripts/automod/evalpin.py`) freezes the qmd index and fixes one shared
# `LLOYD_CODE_ROOT`, so both arms score the same documents against the same
# code, and `execute` refuses to compare without it. A drop on any of the four
# past 3σ is a rollback reason, not corpus noise. This tuple therefore names
# which armed metrics read the FACT layer — not a smaller armed set, and not
# the whole armed set.
FACT_LAYER_METRICS = ("entity_hit_rate", "entity_recall_avg",
                      "fact_entity_recall_avg")

# The coverage of this measurement, written INTO the measurement (#829). The
# `eval` slot of `last_known_good.json` is the slot every later "observed healthy
# in production" claim reads, and until now it stated a set of numbers and no
# axis — so a retrieval-quality pass over seven query scores could be read as
# evidence about behaviour. A reader of the green result should not have to know
# what an eval is to know what it did not look at, and `app/routers/automod.py`
# serves this record straight to Mission Control.
AXIS_FIELD = "axis"
AXIS_MEASURES = ("retrieval quality, paired A/B: the promotion's parent and the "
                 "landed commit score the same pinned questions on the same "
                 "pinned corpus in the same window")
AXIS_DOES_NOT_MEASURE = (
    "agent-loop behaviour: no armed metric observes a tool call, a turn count or a "
    "model decision — the only loop-side check is the PRE-landing gate rung "
    "`prompt_surface`, and only for prompt_builder.py / prefetch.py / SOUL.md / "
    "MEMORY.md / USER.md",
    "graph edge quality: expiring 70% of the active edge set moved no metric, "
    "armed or reported",
)


def axis_coverage() -> dict:
    """What this check measured and what it cannot see, as one artifact field."""
    return {"measures": AXIS_MEASURES, "does_not_measure": list(AXIS_DOES_NOT_MEASURE)}


# The floor applied to an armed metric is `max(SIGMA_MULTIPLIER × σ, one
# question's quantum)` — see `effective_floor`. `MIN_SIGMA` is what the σ term
# falls back to when the published noise artifact carries no measured stdev for
# the metric, and the resolution term is what stops a floor from being narrower
# than the number it grades. #1352: on 2026-09-21 a floor of exactly
# `3 × MIN_SIGMA = 0.0030` rolled back two promotions whose every delta was one
# question flipping, because 0.0030 is a third of this eval's quantum.
MIN_SIGMA = 0.001
SIGMA_MULTIPLIER = 3.0

# The reported score is `round(mean, 3)` (`eval/run_eval.py:601`, inside
# `summarize.avg`), so a metric cannot move by less than this and have it show.
SCORE_ROUND_STEP = 0.001
# Which term won, recorded per metric so a reader never has to re-derive it.
SIGMA_SOURCE_MEASURED = "measured"
SIGMA_SOURCE_MIN_FLOOR = "min_sigma_floor"
FLOOR_BY_SIGMA = "sigma"
FLOOR_BY_RESOLUTION = "resolution"

# ── the latency budget ──────────────────────────────────────────────────────
#
# Why latency gets a ceiling and not a sigma. Every other metric in this file is
# compared against the other arm of the SAME run, which is what cancels corpus
# drift. Latency cannot be compared that way, because the qmd daemon caches
# query embeddings: re-sending identical text is 20-34x cheaper than sending
# text the daemon has never seen, so a paired A/B of latency measures arm order.
# Priced on this box 2026-09-18 by the two rounds that wrote and re-measured
# this constant: 3 and 5 samples per cell, live daemon, named segments, a fresh
# never-seen query text per sample and then the identical repeat, rerank ON at
# pool 240 = `RECALL_DOC_POOL` — the shape `_vault_recall`'s document leg sends:
#
#   3,672-4,027 ms FRESH at pool 240  vs  119-183 ms CACHED (the same text again)
#   2,161-2,291 ms FRESH at pool 40   and 150-210 ms FRESH at 240 rerank-OFF
#
# So the cache alone is a 3.5-3.9 s (20-34x) swing on arm order, wider than any
# step worth catching, and no ratio, band or sigma on latency can be graded — only an
# absolute number. The other two cells are the two levers a widening moves, and
# the rerank-OFF figure is what says the cross-encoder is the whole regression:
# the fetch leg alone is 150-210 ms at 240 and 109-139 ms at 40, so widening the
# pool buys 1,500-1,700 ms of cross-encoding, not 150 ms of fetching. Re-measure the
# first row with:
#
#   .venvs/lloyd/bin/python -c "import time,random,string,agent_mcp.vault as V; \
#     q='aurora basalt cobalt delta ember index maintenance cadence '+''.join(random.choices(string.ascii_lowercase,k=10)); \
#     t=time.perf_counter(); V._qmd_daemon_search(q,240,V.VAULT_SEGMENTS); \
#     print(round((time.perf_counter()-t)*1000))"
#
# (run from the repo root; a query seen before returns 119-183 ms and is the
# cached arm, not this one)
#
# Two contexts, because the two runs are not the same experiment. The nightly
# eval hits the LIVE daemon; the pinned paired check runs two arms through a
# frozen snapshot with one shared `LLOYD_CODE_ROOT`, so it is the same queries at
# a different absolute cost (measured across the artifacts in `eval/baselines/`:
# the nightly nights after #504 average 4,226-4,379 ms, and the fourteen newest
# `automod-check-*.json` run 12,086-12,863 ms). One ceiling for both would be
# wrong in whichever direction it was set.
#
# Each value is AT OR ABOVE the worst average that context has EVER recorded, so
# no already-recorded run reads over budget: #504's accepted 6x cost is
# grandfathered by construction, and the budget exists to make the NEXT step of
# that class report itself. Widening a pool is exactly the change that moves this
# number and nothing else — #504 landed at 708 -> 4,230 ms nightly with every
# rung green, because latency was recorded and read by nobody.
#
# The nightly ceiling sits 8.9% over 4,408.0 ms, and that worst run is
# `nightly-20260904-20260904-060219.json` — nine days BEFORE #504, on the narrow
# pool. So the nightly series has one unexplained outlier the widening does not
# account for (it is a finding on #1129, not something this constant settles);
# the ceiling is set against the worst the context has actually produced rather
# than against the post-widening 4,378.8, because "grandfather everything already
# recorded" is the only rule here that cannot be argued with later.
#
# Correction, 2026-09-18. "The same queries at a different absolute cost" was an
# observation, not an explanation, and the explanation was a defect: the pin
# served published qmd on default settings while production's daemon ran the
# fork with a 4-way reranker and a 1200-char window (`evalpin.QMD_PROGRAM_CONF`).
# The ledger dates it to the hour — the paired check averaged 0.49-0.69 s a
# question through 2026-09-14, 11.0-12.9 s from the check after #504 landed, and
# 15.05 s (every recall at the client's timeout, nothing retrieved) from
# 2026-09-18 18:03Z. A 12 s average under a 15 s timeout is also why single
# questions went missing from one arm or the other, and one of those was the
# false "regression" of 2026-09-17 05:34Z (doc_hit_rate 1.00 -> 0.95). On
# production's settings the same recall is 4.4-5.5 s. The paired ceiling below
# still clears the old readings, because "no recorded run reads over budget" is
# the rule this constant was set by; re-derive it from a week of readings taken
# on the corrected pin rather than from one day's.
LATENCY_BUDGET_MS = {
    "nightly": 4800.0,          # worst ever 4,408.0 (nightly-20260904-060219)
    "paired_check": 14000.0,    # worst ever 12,863.4 (automod-check-20260917-024810)
}
# The two contexts this module knows, named so a caller passes one rather than a
# free-text string that silently falls through to a default.
CONTEXT_NIGHTLY = "nightly"
CONTEXT_PAIRED_CHECK = "paired_check"
# The field the verdict lands in. A separate key from `regressed`/`reasons`
# because the whole point is that this verdict must never be the thing that
# requests a rollback — the exemption it replaces existed to keep a
# cache-order artefact from reverting healthy code, and an absolute ceiling on a
# metric with a 20x cache spread does not make that property any less true.
OVER_BUDGET_FIELD = "latency_over_budget"


def latency_budget(context: str) -> float:
    """The ceiling for `context`; 0.0 for one this file does not know.

    Zero rather than a guess: an unknown context must read as "no verdict
    possible", never as a number that silently grades a run it was never priced
    for — which is how the 165 ms cached figure ended up justifying a 6x step.
    """
    return float(LATENCY_BUDGET_MS.get(context, 0.0))


def over_budget(current: dict, context: str) -> dict | None:
    """The absolute-latency verdict for one comparison, or None if undecidable.

    None — not a pass — when there is no average to read or no budget priced
    for the context. Same rule as the missing noise floor above: a measurement
    that did not happen is never reported as a clean bill of health.
    """
    value = current.get("latency_ms_avg")
    budget = latency_budget(context)
    if value is None or budget <= 0:
        return None
    now = float(value)
    return {"context": context, "budget_ms": budget, "latency_ms_avg": now,
            "over": now > budget,
            "ratio": (now / budget if budget else None)}


def _runner_needed() -> bool:
    """A promotion is owed a measurement and no runner is working on the queue."""
    return bool(pending_promotions()) and not runner_alive()


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    """Offer the job only when it would do something.

    This source is the safety net under the promoter's own spawn, polled every
    fifteen minutes, and its job is a spawn: offered unconditionally it writes
    ninety-six run rows a day that say "nothing to measure", over the handful
    that say a runner had to be started from here — which is the row worth
    reading. The ledger read is ~150 ms on 8 MB, so it leaves the event loop.
    Fails open: a ledger nobody can read is `execute`'s to report.
    """
    import asyncio
    try:
        if not await asyncio.to_thread(_runner_needed):
            return
    except Exception:  # noqa: BLE001
        logger.debug("could not tell whether a runner is needed; offering the job", exc_info=True)
    new_id = queue.enqueue(
        source=NAME, kind="check", payload={},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued automod regression check id=%d", new_id)


def _load_run(baselines: Path, label: str) -> dict | None:
    """Newest run record for `label` as {overall, corpus_ok, corpus, fact_coverage}.

    `corpus_ok` is what `eval/run_eval.py` records about the DATA it scored.
    An arm that measured an empty graph is not a low score, it is a
    measurement that did not happen — and because the doc-side metrics read
    identically with the graph deleted, it looks like a perfectly ordinary
    result. Older records predate the field; absent is treated as unknown
    rather than as False, so a stale baseline cannot fabricate a regression.

    `corpus_ok` cannot see the FACT tree, so `fact_coverage` is derived here
    from the run's own records (#1250): the six arms of 2026-09-18 recorded
    `fact_entity_recall_avg: 0.0` with `corpus_ok: true` against a corpus
    naming 315,462 facts, and nothing else in the artifact says the fact leg
    read nothing. It is coverage evidence about the read path, never a restated
    metric — no metric in `overall` is reinterpreted here, no null is filled,
    no missing metric defaults to 0. See `evaluate`.
    """
    runs = sorted(Path(baselines).glob(f"*{label}*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return None
    try:
        blob = json.loads(runs[-1].read_text())
        records = blob.get("records") or []
        return {"overall": blob["summary"]["overall"],
                "corpus_ok": blob.get("corpus_ok"),
                "corpus": blob.get("corpus") or {},
                "fact_coverage": fact_read_coverage(records),
                **_doc_coverage(records)}
    except (OSError, ValueError, KeyError):
        return None


def _doc_coverage(records: list) -> dict:
    """Which queries the document leg answered with NOTHING, and how many ran.

    On 2026-09-17 05:34Z the pinned qmd daemon returned zero documents for
    one query of twenty (`backlog-363`, no error recorded) while the fact leg
    resolved it fine; the same code answered it with twenty documents four
    hours later. One empty query is exactly -0.05 `doc_hit_rate`, and it drags
    ndcg, mrr and recall with it — a 16σ "regression" that reverted a change
    to a memory-maintenance script. A query the daemon did not answer is a
    measurement that did not happen, not a score of zero. Records older than
    `result_summary` report nothing here, so a stale baseline cannot refuse.
    """
    empty = [str(r.get("id") or r.get("query"))
             for r in records
             if isinstance(r.get("result_summary"), dict)
             and r["result_summary"].get("n_docs") == 0
             and not r.get("error")]
    return {"n_records": len(records), "empty_doc_queries": empty}


def unanswered_doc_queries(arm: dict) -> list[str]:
    """The queries that make `arm` a non-measurement, or [] if it is one.

    A strict subset of the questions coming back with no documents is the
    daemon dropping requests. ALL of them coming back empty is the retriever
    itself, which is the one shape a code change under test can produce, and
    that stays a score (of zero, which `evaluate` will refuse).
    """
    empty = list(arm.get("empty_doc_queries") or [])
    total = int(arm.get("n_records") or 0)
    if empty and len(empty) < total:
        return empty
    return []


def fact_read_coverage(records: list) -> dict:
    """How much of the fact tree a run's fact leg actually READ, per run.

    The ONE implementation: `eval/run_eval.py` imports this rather than keeping
    its own copy, and `workers` reads it off the loaded artifact. Two copies of
    a guard is how a guard ends up disagreeing with itself across a process
    boundary, which is the defect class #1250 is filed under.

    `result_summary.n_facts` is the only per-query record of the fact leg's
    yield, so it is the only evidence separating "the fact leg read nothing"
    from "the fact leg read everything and matched nothing". Records predating
    `result_summary` report nothing, so a stale artifact cannot refuse a
    promotion it never measured.

    The failed-read counter is aggregated defensively and never required:
    artifacts written before #1250 lack `n_fact_reads_failed` entirely, and a
    coverage block that demanded the key would read a healthy old baseline as an
    outage — the inverse of the bug this exists to catch.
    """
    reported = [r for r in records
                if isinstance(r.get("result_summary"), dict)
                and isinstance(r["result_summary"].get("n_facts"), int)]
    empty = [str(r.get("id") or r.get("query")) for r in reported
             if r["result_summary"]["n_facts"] == 0 and not r.get("error")]
    failed = 0
    first_error: str | None = None
    for r in reported:
        rs = r["result_summary"]
        try:
            failed += int(rs.get("n_fact_reads_failed") or 0)
        except (TypeError, ValueError):
            pass
        first_error = first_error or rs.get("fact_read_first_error")
    return {
        "n_records": len(records),
        "n_facts_reports": len(reported),
        "n_facts_total": sum(int(r["result_summary"]["n_facts"]) for r in reported),
        "empty_fact_queries": empty,
        "n_fact_reads_failed_total": failed,
        "fact_read_first_error": first_error,
    }


def fact_leg_empty(cov: dict, n_corpus_facts: int) -> bool:
    """True when this run's fact leg read NOTHING on a corpus that has facts.

    `corpus_ok` cannot see this and never could: it is
    `bool(corpus["edges_active"]) and bool(corpus["entities"])`, the graph half
    only, and `corpus["facts"]` is the store's index count — what the index
    holds, not what recall could read. So from 2026-09-18T18:03Z six paired
    check arms recorded `fact_entity_recall_avg: 0.0` with `corpus_ok: true` and
    `errors: 0` against `corpus.facts: 315462`, because every per-entity fact
    read was being discarded by a bare `except Exception: continue`.

    Total-across-queries, never per query. A per-query fact count of 0 is a
    legitimate score — the last healthy arm's distribution is nineteen 10s and
    one 1, with no query at 0 — so only the whole run reading nothing is a
    non-measurement. Every query must also have REPORTED its count: a run whose
    records predate `result_summary` is silent, and silence is not evidence.
    """
    reported = int(cov.get("n_facts_reports") or 0)
    if reported <= 0 or reported != int(cov.get("n_records") or 0):
        return False
    return int(cov.get("n_facts_total") or 0) == 0 and int(n_corpus_facts or 0) > 0


def empty_fact_leg(arm: dict) -> bool:
    """Whether `arm`'s fact leg read nothing while the fact tree is not empty.

    The fact-leg mirror of `unanswered_doc_queries`, sitting beside it for a
    reason: the document guard refuses an arm the DAEMON failed (a strict
    subset of queries), while this one refuses an arm whose EVERY query read
    nothing. That is deliberately the opposite strictness, because the two
    failures have opposite shapes — the daemon drops some requests, a broken
    fact path drops all of them and records them as zeros.

    Unlike `unanswered_doc_queries` this returns a verdict and not a list:
    there is no per-query subset to name, only the run total. What the leg
    actually hit, when the artifact carries it, is
    `arm["fact_coverage"]["fact_read_first_error"]`.
    """
    cov = arm.get("fact_coverage")
    if not isinstance(cov, dict):
        return False
    return fact_leg_empty(cov, int(((arm.get("corpus") or {}).get("facts")) or 0))


@contextmanager
def _baseline_worktree(commit: str):
    """Check `commit` out into a scratch worktree for the duration of the block.

    Kept open across BOTH arms, not just the baseline one: it is also the
    shared grep corpus. See `_run_arm`.
    """
    scratch = Path(tempfile.mkdtemp(prefix="automod-eval-"))
    wt = scratch / "lloyd"
    try:
        r = subprocess.run(["git", "-C", str(LIVE_ROOT), "worktree", "add",
                            "--detach", "-q", str(wt), commit],
                           capture_output=True, text=True, check=False)
        if r.returncode != 0:
            logger.error("paired eval worktree failed: %s", r.stderr[-300:])
            yield None
            return
        yield wt
    finally:
        subprocess.run(["git", "-C", str(LIVE_ROOT), "worktree", "remove",
                        "--force", str(wt)], capture_output=True, check=False)
        subprocess.run(["git", "-C", str(LIVE_ROOT), "worktree", "prune"],
                       capture_output=True, check=False)
        shutil.rmtree(scratch, ignore_errors=True)


# The arms of one comparison, as the djev replay file names them.
ARM_BASELINE, ARM_CURRENT, ARM_CONFIRM = "baseline", "current", "confirm"


def _replayed(env: dict, db: Path, arm: str, anchor: str = ARM_BASELINE) -> dict:
    """`env` with one arm put under the comparison's djev replay file."""
    return {**env, **replay_env(db, arm, anchor)}


def ranker_reading(stats: dict, arm: str) -> str:
    """Whether djev's own noise is in this arm's comparison with its anchor.

    `replayed`: every rank request the arm made was one the anchor had already
    asked, so djev answered it identically and cannot be the difference. `fresh`:
    the change moved what djev was asked for at least one recall, and that
    answer is one draw of a ranker that does not repeat itself. `unknown`: the
    arm left no replay record, because its code predates replay or it never
    asked djev at all.
    """
    row = stats.get(arm) or {}
    if not row:
        return "unknown"
    return "fresh" if row.get("fresh", 0) > 0 else "replayed"


def _floor_for(noise: dict, reading: str) -> dict:
    """The noise record `evaluate` should read for a ranker reading.

    Replayed arms compare deterministically, so the pinned floor (`metrics`) is
    the right one. An arm that drew djev fresh is compared against the spread
    djev produces on its own (`metrics_fresh_ranker`, measured by
    `measure_noise` without replay); with no such measurement the pinned floor
    is used and the record says so.
    """
    noise = dict(noise or {})
    if reading == "replayed":
        return {**noise, "floor": "replayed"}
    fresh = noise.get("metrics_fresh_ranker")
    if fresh:
        return {**noise, "metrics": fresh, "floor": "fresh_ranker"}
    return {**noise, "floor": "replayed_no_fresh_floor"}


def query_count(*arms: dict | None) -> int | None:
    """The denominator BOTH arms agree on, or None when they do not share one.

    It is the run's own `n_queries`, not a count of the query file: `summarize`
    averages over `records` (`eval/run_eval.py:617`), so an arm that scored fewer
    records has a correspondingly coarser score.

    Arms that scored a different number of questions do not have one quantum
    between them, so the term contributes nothing beyond the rounding step and
    the comparison keeps the threshold it always had. That is deliberately not
    treated as a measurement failure here: an arm that dropped a question is
    already refused upstream as a non-measurement (`unanswered_doc_queries`,
    `empty_fact_leg`), and the differing counts are reported in `detail` either
    way.
    """
    counts: set[int] = set()
    for arm in arms:
        try:
            n = int((arm or {}).get("n_queries"))
        except (TypeError, ValueError):
            continue
        if n > 0:
            counts.add(n)
    if len(counts) != 1:
        return None
    return counts.pop()


def score_resolution(n: int | None) -> float:
    """The smallest reported move that cannot be one question flipping.

    Two terms, both derived rather than chosen:

    * `1/n` — every armed metric is an `avg` over the arm's scored records
      (`eval/run_eval.py:617`), so ONE question's entire contribution is `1/n`.
      For the two hit-rates the per-query score IS 0/1, which makes that exactly
      the smallest non-zero move the metric has at all: the three `doc_hit_rate`
      values across the evening of 2026-09-21 — 0.529, 0.517, 0.506 — are 46/87,
      45/87 and 44/87, each one question flipping, and each was reported as a
      regression "beyond 3σ=0.0030". For the graded averages (`ndcg10`, `mrr_doc`,
      the two recall averages) one question can move LESS than `1/n`, so this is
      the ceiling of its effect; taking the ceiling is the side of the trade that
      keeps a single question from reverting a commit, and it is the trade the
      loop's own history argues for — every rollback this check has performed was
      a false positive.
    * `+ SCORE_ROUND_STEP` — the reported number is `round(mean, 3)`, so each
      side carries up to 0.0005 of representation error and the difference of two
      reported values can overstate one question by up to one full step. At n=87 a
      single flip is truly 0.011494 and was REPORTED as -0.012 (0.529 → 0.517), so
      a floor of exactly `1/n` would still fire on the very flip it is meant to
      absorb. `1/n + 0.001` = 0.012494 cannot.

    With no denominator the term degrades to the rounding step alone: a value
    rounded to 3 dp cannot move by less than 0.001 whatever `n` is.
    """
    return (1.0 / n if n else 0.0) + SCORE_ROUND_STEP


def metric_sigma(noise: dict | None, key: str) -> tuple[float, str]:
    """The published paired σ for one metric, and whether it was measured.

    `min_sigma_floor` is reported whenever the artifact has no positive stdev
    for the metric: that value is a floor chosen by this file, and the record has
    to say so, because `stdev: 0.0` is also what an arm that could not vary
    produces — the two are not the same claim.
    """
    entry = ((noise or {}).get("metrics") or {}).get(key) or {}
    raw = entry.get("stdev")
    sigma = float(raw) if isinstance(raw, (int, float)) else 0.0
    if sigma > 0:
        return sigma, SIGMA_SOURCE_MEASURED
    return MIN_SIGMA, SIGMA_SOURCE_MIN_FLOOR


def effective_floor(noise: dict | None, key: str, n: int | None) -> dict:
    """The one number that decides whether `key` moved: `max(k·σ, 1/n)`.

    Both terms are measurements of a different thing. σ is the spread of this
    instrument on one settled commit (published per metric by `measure_noise`);
    `1/n` is the granularity of the number being graded. Neither alone is
    sufficient: a σ of 0.0 with no resolution floor re-creates #1352 (floor
    0.0030 on a metric that cannot move by less than 0.0115), and a resolution
    floor with no σ would treat a genuinely noisy metric as deterministic.
    """
    sigma, source = metric_sigma(noise, key)
    k_sigma = SIGMA_MULTIPLIER * sigma
    resolution = score_resolution(n)
    by_sigma = k_sigma >= resolution
    return {"sigma": sigma, "sigma_source": source, "k_sigma": k_sigma,
            "resolution": resolution, "n_queries": n,
            "floor": k_sigma if by_sigma else resolution,
            "floor_governed_by": FLOOR_BY_SIGMA if by_sigma else FLOOR_BY_RESOLUTION}


def paired_sigma(noise: dict | None, key: str) -> tuple[float, str]:
    """The σ that governs two INDEPENDENT draws of the instrument.

    The confirm arm runs under its own ranker arm name (`ARM_CONFIRM`, see
    `check_promotion`), so its djev answers are fresh draws rather than replays
    of the anchor's: the spread between a first pass and a confirm pass is the
    live-ranker spread, which `measure_noise` publishes separately as
    `metrics_fresh_ranker`. Measured there on 2026-09-22 over 5 fresh-ranker
    trials of the 81-question set: `ndcg10` σ 0.010271, `mrr_doc` 0.010183,
    `doc_recall_avg` 0.005505, `doc_hit_rate` 0.005367 — three to ten times the
    0.0030 the check rolled back on. The replayed bucket is the fallback, and its
    source label is what tells a reader no fresh-ranker σ was published.
    """
    fresh = ((noise or {}).get("metrics_fresh_ranker") or {}).get(key) or {}
    raw = fresh.get("stdev")
    sigma = float(raw) if isinstance(raw, (int, float)) else 0.0
    if sigma > 0:
        return sigma, f"{SIGMA_SOURCE_MEASURED}_fresh_ranker"
    return metric_sigma(noise, key)


def magnitude_confirmed(first: dict, second: dict, noise: dict | None,
                        metrics: list[str]) -> tuple[list[str], list[str]]:
    """Decide the second look on MAGNITUDE against the paired σ, not on sign.

    `evaluate` answers "is this metric past the floor again?", and for a
    zero-mean ±0.01 variable two draws of that question is near-guaranteed: on
    2026-09-21 21:44Z the first pass said `ndcg10` Δ-0.0060 and the confirm pass
    said Δ-0.0200 — the second draw three times deeper than the first, which is
    what an independent draw of noise looks like — and the pair was booked as
    `confirmed_by` and became a rollback. An observation confirms an effect only
    when BOTH draws clear k·σ of zero; a pair that does not is reported in
    `unconfirmed_reasons` as a finding about the instrument.

    Returns (confirmed, rejected) metric names.
    """
    confirmed: list[str] = []
    rejected: list[str] = []
    for key in metrics:
        sigma, _source = paired_sigma(noise, key)
        limit = SIGMA_MULTIPLIER * sigma
        d1 = (first.get(key) or {}).get("delta")
        d2 = (second.get(key) or {}).get("delta")
        if d1 is None or d2 is None:
            rejected.append(key)
            continue
        if abs(float(d1)) >= limit and abs(float(d2)) >= limit:
            confirmed.append(key)
        else:
            rejected.append(key)
    return confirmed, rejected


def metric_evidence(detail: dict, reasons: list[str],
                    noise_floor_stale: bool) -> dict:
    """Per triggering metric: floor, σ and whether the floor was stale.

    This is what lets the needs-human routing answer the only question that
    matters about a halt — did any metric actually move past a floor the
    instrument could justify, or did the instrument grade itself with a ruler it
    had not measured? Before #1352 the answer was unfindable: `noise_floor_stale`
    was written on every check and read by no branch, and every armed metric
    carried the same unmeasured `tolerance: 0.003`.
    """
    out: dict[str, dict] = {}
    for reason in reasons:
        key = reason.split()[0] if reason.split() else ""
        entry = detail.get(key)
        if not isinstance(entry, dict):
            continue
        out[key] = {"floor": entry.get("floor"), "sigma": entry.get("sigma"),
                    "sigma_source": entry.get("sigma_source"),
                    "floor_governed_by": entry.get("floor_governed_by"),
                    "resolution": entry.get("resolution"),
                    "n_queries": entry.get("n_queries"),
                    "delta": entry.get("delta"),
                    "noise_floor_stale": bool(noise_floor_stale)}
    return out


def floors_line(evidence: dict, noise_floor_stale: bool) -> str:
    """One clause per triggering metric: Δ, floor, σ and whether it was stale.

    This string is what makes a rollback the instrument could not justify
    distinguishable from one it could, on the two surfaces a person actually
    reads: the check's `summary`, and the guardian's rollback alert, whose body is
    this request's `reason` verbatim (`Guardian._rollback_once` → `f"Trigger:
    {trigger}\\n{reason}"`). Before #1352 both said only "beyond 3σ=0.0030" — a
    number that was identical whether or not the σ behind it had ever been
    measured, which is how two noise-sized rollbacks came to look like two
    regressions in the ledger.
    """
    parts = [f"{name}: Δ{(ev.get('delta') or 0.0):+.4f} vs floor={ev.get('floor'):.4f}"
             f" (σ={ev.get('sigma'):.6f} {ev.get('sigma_source') or '?'}, resolution="
             f"{ev.get('resolution'):.4f} at n={ev.get('n_queries')})"
             for name, ev in evidence.items() if isinstance(ev, dict)]
    parts.append(f"noise_floor_stale={bool(noise_floor_stale)}")
    return "; ".join(parts)


def _run_arm(tree: Path, label: str, env: dict, timeout: float = 900.0) -> dict | None:
    """Run the eval from `tree`, scoring the LIVE queries against a pinned corpus.

    Three things are held identical across the two arms so that the only
    difference left is the code being compared:

    * the fact tree and knowledge graph, via `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB`;
    * the document corpus, via a frozen qmd snapshot the caller is serving and
      a pinned `LLOYD_CODE_ROOT` for the grep leg;
    * the QUESTIONS. The baseline arm runs the OLD `run_eval.py` out of a
      worktree, which carries the OLD `vault_recall_queries.yaml`. Editing the
      eval set would otherwise ask the two arms different questions and score
      the difference as a code regression, which is how a change to the
      MEASUREMENT gets attributed to the thing being measured.
    """
    from app.paths import VAULT_FACTS_ROOT, VAULT_KG_DB

    env = {
        **env,
        "PYTHONPATH": str(tree),
        "LLOYD_FACTS_ROOT": str(VAULT_FACTS_ROOT),   # live data, whichever code
        "LLOYD_KG_DB": str(VAULT_KG_DB),
    }
    r = subprocess.run(
        [str(LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"),
         str(tree / "eval" / "run_eval.py"),
         "--label", label,
         "--queries", str(LIVE_QUERIES)],
        cwd=str(tree), env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    if r.returncode != 0:
        logger.error("eval arm %s failed: %s", label, (r.stdout + r.stderr)[-500:])
        return None
    return _load_run(tree / "eval" / "baselines", label)


def measure_noise(trials: int = 5, fresh_trials: int = 5) -> dict:
    """Record mean/stdev per metric on an unchanged tree. Run once, by hand.

    Runs inside the pinned corpus, because that is the condition the armed
    metrics are compared under. A noise floor measured against the live daemon
    describes a different experiment from the one it is used to judge — which
    is exactly how the doc-side metrics came to be armed on a "stdev 0.0000"
    that did not hold when it mattered.

    Two floors, because the check has two conditions (`ranker_reading`):

    * `metrics` — trial 0 plus `trials - 1` arms replaying trial 0's djev
      answers: the condition of a comparison whose change left the ranker's
      input alone. Deterministic when the pin is, and that is what it checks.
    * `metrics_fresh_ranker` — trial 0 plus `fresh_trials - 1` arms that each
      draw djev afresh: the spread djev's own answers put on the metrics, used
      when the change moved what djev is asked.

    A trial that the daemon or djev did not fully answer is dropped, not
    scored, exactly as `check_promotion` refuses to score one.
    """
    import tempfile as _tf
    samples: dict[str, list[float]] = {}
    fresh_samples: dict[str, list[float]] = {}
    dropped: list[str] = []
    work = Path(_tf.mkdtemp(prefix="automod-noise-pin-"))
    db = work / "djev-replay.sqlite"
    anchor = "trial-0"
    ranker: dict = {}
    try:
        with PinnedCorpus(work) as pin:
            env = pin.env_for(code_root=LIVE_ROOT)
            plan = [(f"trial-{i}", anchor, i == 0, True) for i in range(trials)]
            plan += [(f"fresh-{j}", f"fresh-{j}", False, False) for j in range(1, fresh_trials)]
            for arm, arm_anchor, is_anchor, replayed in plan:
                run = _run_arm(LIVE_ROOT, f"automod-noise-{arm}",
                               _replayed(env, db, arm, arm_anchor))
                overall = (run or {}).get("overall")
                if not overall:
                    continue
                # A trial the daemon did not fully answer is not a sample of
                # the eval's noise, exactly as `execute` refuses to score it.
                unanswered = unanswered_doc_queries(run)
                if unanswered:
                    dropped.append(f"{arm}: {', '.join(unanswered)}")
                    continue
                failed = {k: v for k, v in (replay_stats(db).get(arm) or {}).items()
                          if k in REPLAY_FAILURES and v}
                if failed:
                    dropped.append(f"{arm}: djev {failed}")
                    continue
                if replayed:
                    _accumulate(samples, overall)
                if is_anchor or not replayed:
                    _accumulate(fresh_samples, overall)
            ranker = replay_stats(db)
            pin.discard()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return _summarise_noise(samples, trials, dropped=dropped,
                            fresh_samples=fresh_samples, fresh_trials=fresh_trials,
                            ranker=ranker)


def _accumulate(samples: dict, overall: dict) -> None:
    for key, value in overall.items():
        if isinstance(value, (int, float)):
            samples.setdefault(key, []).append(float(value))


def queries_fingerprint() -> str:
    """Hash of the question set a measurement was taken against.

    A noise floor describes one experiment. Retargeting a query changes the
    experiment, and a floor carried over from the old one is provenance the
    reader cannot see is stale. Recorded so `execute` can say so.
    """
    import hashlib
    try:
        return hashlib.sha1(LIVE_QUERIES.read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def _summarise_noise(samples: dict, trials: int, *, dropped: list[str] | None = None,
                     fresh_samples: dict | None = None, fresh_trials: int | None = None,
                     ranker: dict | None = None) -> dict:
    def summary(bucket: dict) -> dict:
        return {k: {"mean": statistics.fmean(v),
                    "stdev": (statistics.stdev(v) if len(v) > 1 else 0.0),
                    "min": min(v), "max": max(v), "n": len(v)}
                for k, v in bucket.items() if v}

    noise = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trials": trials,
        "dropped_trials": list(dropped or []),
        "queries_fingerprint": queries_fingerprint(),
        "pinned": True,
        "metrics": summary(samples),
    }
    if fresh_samples:
        noise["fresh_trials"] = fresh_trials
        noise["metrics_fresh_ranker"] = summary(fresh_samples)
    if ranker:
        noise["ranker"] = ranker
    NOISE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOISE_PATH.write_text(json.dumps(noise, indent=2), encoding="utf-8")
    return noise


def evaluate(current: dict, baseline: dict, noise: dict,
             context: str = CONTEXT_PAIRED_CHECK,
             floor_stale: bool = False) -> tuple[bool, list[str], dict]:
    """Pure comparison. Returns (regressed, reasons, per-metric detail).

    `floor_stale` is provenance travelling into the reason text and the
    per-metric detail, so the string a human reads names the floor, the σ behind
    it and whether that σ was measured against this question set. It changes what
    is SAID here and not what is `regressed`: deciding that an unjustified floor
    is no verdict is `check_promotion`'s job, because that is where the rollback
    request is made. Each armed metric's threshold is `effective_floor` —
    `max(k·σ, 1/n_queries)` — so a floor can never be narrower than the number it
    grades (#1352).

    `context` selects the absolute latency ceiling (`LATENCY_BUDGET_MS`) that
    `latency_ms_avg` is read against. It changes what is REPORTED and never what
    is `regressed`: the verdict lands in
    `detail["latency_ms_avg"]["budget"]["over"]`, and `reasons` stays the armed
    metrics plus new eval errors. That separation is the property the
    report-only exemption was created to protect — latency movement, largely
    cache order, must not be able to revert a healthy commit.
    """
    reasons: list[str] = []
    detail: dict[str, Any] = {}
    n = query_count(current, baseline)

    for key in ARMED_METRICS:
        if key not in current or key not in baseline:
            continue
        if current[key] is None or baseline[key] is None:
            # A null armed metric is a non-measurement, and #1250 is what makes
            # one exist: `run_eval` now records `fact_entity_recall_avg: null`
            # instead of a scored 0.0 when every query's fact leg read nothing on
            # a corpus that indexes facts. Comparing it is not possible — the
            # `float()` below raised `TypeError: float() argument must be a string
            # or a real number, not 'NoneType'`, which `_would_regress`'s blanket
            # handler turned into `return False` and the caller's into a skip that
            # named a crash rather than the dead leg. Skipping it here is the same
            # answer as a key that is absent, and the entry says WHY, so the
            # detail written to the ledger can never be read as "this metric was
            # compared and moved nothing".
            detail[key] = {"before": baseline[key], "after": current[key],
                           "armed": True, "not_measured": True}
            continue
        now, was = float(current[key]), float(baseline[key])
        floor = effective_floor(noise, key, n)
        delta = now - was
        detail[key] = {"before": was, "after": now, "delta": delta,
                       # `tolerance` is the historical key and stays: it is the
                       # same number under the name the ledger already carries.
                       "tolerance": floor["floor"], "armed": True,
                       "noise_floor_stale": bool(floor_stale), **floor}
        if delta < -floor["floor"]:
            reasons.append(
                f"{key} {was:.4f} → {now:.4f} (Δ{delta:+.4f}, beyond "
                f"floor={floor['floor']:.4f} "
                f"[{SIGMA_MULTIPLIER:g}σ={floor['k_sigma']:.4f} "
                f"{floor['sigma_source']}, resolution={floor['resolution']:.4f} "
                f"n={floor['n_queries']}], noise_floor_stale={bool(floor_stale)})")

    for key in REPORT_ONLY:
        if key in current and key in baseline:
            detail[key] = {"before": baseline[key], "after": current[key],
                           "delta": float(current[key]) - float(baseline[key]),
                           "armed": False}

    # The one reported metric that is still COMPARED, and the only comparison in
    # this function that is absolute rather than paired. Attached to the latency
    # entry rather than appended to `reasons`: `regressed` must stay False for a
    # run that merely reads slow, and `reasons` is the channel the guardian reads
    # as a rollback request.
    reading = over_budget(current, context)
    if reading is not None:
        entry = detail.setdefault("latency_ms_avg", {"armed": False})
        entry["budget"] = reading
        # The named verdict exists ONLY past the ceiling. An in-budget
        # comparison gets the reading (`budget`, so a reader can tell "inside"
        # from "never measured") and no verdict to act on.
        if reading["over"]:
            entry[OVER_BUDGET_FIELD] = reading

    if current.get("errors", 0) and not baseline.get("errors", 0):
        reasons.append(f"eval errors appeared: 0 → {current['errors']}")

    return bool(reasons), reasons, detail


async def execute(item: QueueItem) -> dict[str, Any]:
    """Start the detached runner if a promotion is waiting to be measured.

    **The comparison no longer runs inside the backend**, because nothing that
    runs there survives a landing, and at a landing every twenty minutes that
    is every check. What 2026-09-18's ledger showed once landings sped up —
    8 of 17 promotions measured, 0 of the last 4 — came from four things this
    job being a pool job caused:

    * the landing's idle gate counts agent turns, and this was two blocking
      eval subprocesses on a thread, so a landing restarted the backend under
      it;
    * supervisord's group kill took the check and left its pinned qmd daemon
      (own process group) holding :8182, where every later check started a
      second one on top of it — the port probe asked the wrong loopback — and
      recorded `pinned qmd exited immediately`;
    * the round hold keeps this source unclaimed while a round is in flight,
      which is always, so a check could only start in the seconds after a
      restart — the one moment guaranteed to be followed by another restart;
    * and it measured "the latest promotion", so one that landed while the
      previous check ran was never measured by anyone.

    `run_pending` is the same comparison in its own session (`spawn_detached`,
    like the gate and the landing): one check per promotion, each with its own
    commit and parent, oldest first, behind `regression.lock`. This job is now
    a few milliseconds and needs no engine, so it is exempt from the round
    hold; the promoter starts the runner too, the moment a landing is verified.
    """
    import asyncio
    return await asyncio.to_thread(start_runner)


def start_runner() -> dict[str, Any]:
    """Spawn `run_pending` detached, unless nothing is pending or one is running."""
    from scripts.automod import state as S
    try:
        pending = pending_promotions()
    except Exception as exc:  # noqa: BLE001 — an unreadable ledger is a skip, said so
        return _skipped(f"could not read the ledger for pending promotions: {exc}")
    if not pending:
        return _skipped("no promotion is waiting to be measured")
    if runner_alive():
        return {"status": "success", "summary":
                f"a regression runner is already working; {len(pending)} promotion(s) pending"}
    log = S.STATE_DIR / "regression.log"
    pid = S.spawn_detached(
        [LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python", "-m", RUNNER_MODULE, "run"],
        log, cwd=LIVE_ROOT)
    return {"status": "success", "pid": pid,
            "summary": (f"started the detached regression runner (pid {pid}) for "
                        f"{len(pending)} promotion(s): "
                        + ", ".join(str(p["commit"])[:8] for p in pending[:6]))}


def _skipped(reason: str) -> dict[str, Any]:
    """A measurement that did not happen, said so that a reader can see it.

    `{"skipped": reason}` alone was the old shape, and `pool.normalize_result`
    is now able to read it — but the run record still deserves a real status
    and a summary, because all 22 of this source's runs to date are rows
    marked `success` with an empty summary. For a detector whose entire
    premise is "a missing noise floor means cannot evaluate, never no
    regression", recording a skip as a success is the one failure mode it
    cannot afford.
    """
    return {"status": "skipped", "skipped": reason, "summary": reason[:500]}


def _execute_blocking() -> dict[str, Any]:
    """Compare quality against the promotion's PARENT, on live data, paired.

    **Nothing in here may run on the event loop.** Both arms are
    `subprocess.run` with a 900-second timeout, on either side of a `git
    worktree add`, and this used to be awaited directly from `pool.py` — so a
    single check froze the backend for as long as it took. That is the shared
    loop which serves every HTTP request and streams every chat turn; the
    109-second run in the history is 109 seconds during which Lloyd answered
    nothing. Every other source that touches the disk hard already hops onto a
    thread and says why (`gap_fill`, `bench_mine`); this one, the heaviest of
    them by an order of magnitude, was the one that did not.

    Two things had to change before this could ever run.

    It keyed on `current.json`, which the guardian DELETES the moment a
    promotion settles — fifteen minutes after landing. A once-a-day job
    therefore found "no promotion under observation" essentially always, which
    is why this source has zero runs in its history. It now also accepts
    `last_settled.json`, written at settle for exactly this reason.

    And it compared against the LKG commit, which by then IS the promoted
    commit — the checked-out baseline arm and the live arm would have been the
    same code, so the comparison was guaranteed to find nothing. The baseline
    is the promotion's own parent.
    """
    from scripts.automod import state as S

    promo = S.read_current()
    if promo and promo.get("state") == "observing":
        subject, stage = promo, "observing"
    else:
        subject, stage = S.read_last_settled(), "settled"
    if not subject or not subject.get("commit"):
        return _skipped("no recent promotion to check")

    commit = subject["commit"]
    landed = float(subject.get("landed_ts") or 0)
    if not landed or time.time() - landed > 24 * 3600:
        return _skipped("last promotion is older than 24h")

    # One measurement per promotion. Both arms are full evals plus a worktree,
    # and the answer cannot change while the commit does not.
    for ev in S.read_events(limit=200):
        if ev.get("event") == "regression_check" and ev.get("commit") == commit:
            return _skipped(f"{commit[:8]} already checked")

    return check_promotion(subject, stage)


# How many times one promotion may come back "cannot evaluate" before the
# runner stops offering it. A skip is a measurement that did not happen, so it
# is retried — but a promotion that can never be measured (its parent is gone,
# the corpus will not pin) must not hold the queue for ever.
MAX_SKIPS_PER_COMMIT = 2
# The entry point the runner is started through — its own tiny module, because
# `-m` on THIS one executes it twice (see `scripts/automod/regression_runner.py`).
RUNNER_MODULE = "scripts.automod.regression_runner"
PENDING_MAX_AGE_S = 24 * 3600.0
REGRESSION_LOCK = "regression.lock"


def pending_promotions(now: float | None = None) -> list[dict]:
    """Promotions nobody has measured yet, oldest first.

    Read off the ledger, which is what makes "one check per promotion" true at
    any landing rate: every `promoted` row of the last day that was not rolled
    back, has no `regression_check` row that measured anything
    (`state.regression_measured` — zero against zero never did), and has not
    already come back "cannot evaluate" `MAX_SKIPS_PER_COMMIT` times. Each carries its own `commit` and
    `parent` — the check used to measure "whatever landed last", so a
    promotion that landed while the previous check ran was measured by nobody.
    """
    from scripts.automod import state as S
    now = now or time.time()
    events = S.read_events(limit=4000)
    checked = {str(e.get("commit") or "") for e in events
               if e.get("event") == "regression_check" and S.regression_measured(e)}
    reverted = S.reverted_commits(events)
    skips: dict[str, int] = {}
    for e in events:
        if e.get("event") == "regression_skipped" and e.get("commit"):
            skips[str(e["commit"])] = skips.get(str(e["commit"]), 0) + 1
    out: list[dict] = []
    for e in events:
        if e.get("event") != "promoted" or not e.get("commit"):
            continue
        commit = str(e["commit"])
        landed = float(e.get("ts") or 0)
        if now - landed > PENDING_MAX_AGE_S or commit in checked or commit in reverted:
            continue
        if skips.get(commit, 0) >= MAX_SKIPS_PER_COMMIT:
            continue
        out.append({"commit": commit, "parent": e.get("parent"), "landed_ts": landed,
                    "round_id": e.get("round_id"), "changed_paths": e.get("changed_paths") or []})
    out.sort(key=lambda p: p["landed_ts"])
    return out


def runner_alive() -> bool:
    """Whether a runner holds `regression.lock` right now (flock: a dead
    holder's lock is already gone, so this cannot go stale)."""
    from scripts.automod import state as S
    try:
        S.Lock(S.STATE_DIR / REGRESSION_LOCK, owner="probe").acquire().release()
        return False
    except S.LockHeld:
        return True


# Older than any check can be: two arms at `_run_arm`'s 900 s each, the pin's
# start-up and its warm-up. Scratch this old has no owner.
STALE_SCRATCH_AGE_S = 2 * 3600.0
SCRATCH_PREFIXES = ("automod-eval-", "automod-pin-", "automod-noise-pin-")


def sweep_stale_scratch(now: float | None = None) -> list[str]:
    """Remove what a killed check left in the temp dir; the paths removed.

    `_baseline_worktree` and `check_promotion` clean up in a `finally`, which a
    SIGKILL never runs — and until 2026-09-18 every check ran inside the backend
    a landing restarts. Each one left a whole checkout registered as a worktree
    of the LIVE repo (three were, that day) plus the pin's work dir. Called by
    the runner under its lock, so the only scratch a live check could own is
    younger than `STALE_SCRATCH_AGE_S`.
    """
    now = now or time.time()
    removed: list[str] = []
    try:
        entries = [e for e in Path(tempfile.gettempdir()).iterdir()
                   if e.is_dir() and e.name.startswith(SCRATCH_PREFIXES)]
    except OSError:
        return removed
    for scratch in entries:
        try:
            if now - scratch.stat().st_mtime < STALE_SCRATCH_AGE_S:
                continue
        except OSError:
            continue
        wt = scratch / "lloyd"
        if wt.exists():
            subprocess.run(["git", "-C", str(LIVE_ROOT), "worktree", "remove", "--force", str(wt)],
                           capture_output=True, check=False)
        shutil.rmtree(scratch, ignore_errors=True)
        if not scratch.exists():
            removed.append(str(scratch))
    if removed:
        subprocess.run(["git", "-C", str(LIVE_ROOT), "worktree", "prune"],
                       capture_output=True, check=False)
        logger.info("removed %d stale scratch dir(s) a killed check left: %s",
                    len(removed), ", ".join(removed))
    return removed


# Longer than any check that is still making progress: the pin's start-up and
# four warm-up tries, then three arms at `_run_arm`'s 900 s each, is 59 minutes.
CHECK_WATCHDOG_S = 75 * 60


def _arm_watchdog(commit: str, seconds: int | None = None) -> None:
    """End the runner if one check outlives `seconds`, saying which.

    Every long step inside a check has its own timeout except `git` and the
    snapshot, and a runner wedged in one of those would hold `regression.lock`
    for ever: no later runner could start, nothing would be measured, and
    nothing would say so — the failure this whole module was rebuilt around. A
    dead runner is recoverable by construction (the kernel drops the flock and
    ends the pin, `sweep_stale_scratch` takes the rest, the next landing starts
    another), so the watchdog's whole job is to turn a wedged one into a dead
    one, after recording a skip so the same promotion cannot wedge the queue
    more than `MAX_SKIPS_PER_COMMIT` times. Main thread only; a no-op elsewhere.
    """
    import signal
    import threading
    if threading.current_thread() is not threading.main_thread():
        return
    seconds = int(seconds if seconds is not None else CHECK_WATCHDOG_S)

    def _expired(signum, frame):
        try:
            from scripts.automod import state as S
            S.append_event({"event": "regression_skipped", "commit": commit,
                            "reason": f"the check did not finish in {seconds} s; the runner ended "
                                      "itself so the queue is not held — cannot evaluate"})
        finally:
            # Whatever the check was wedged IN goes too — a hung `git`, an arm
            # mid-run — or it outlives the runner holding its pipes and its
            # worktree. Only a group of our own making: `spawn_detached` makes
            # the runner a session leader, and a runner started any other way
            # shares its group with whoever started it. The pin has its own
            # group and is ended by the kernel when this process is.
            try:
                if os.getpgrp() == os.getpid():
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    os.killpg(os.getpgrp(), signal.SIGTERM)
            finally:
                os._exit(3)
    signal.signal(signal.SIGALRM, _expired)
    signal.alarm(max(1, seconds))


def _disarm_watchdog() -> None:
    import signal
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.alarm(0)


def run_pending(max_checks: int | None = None) -> list[dict]:
    """Measure every pending promotion, oldest first. One runner at a time.

    Re-reads the queue after each check, so a promotion that lands while this
    runs is picked up by the same runner rather than waiting for the next
    start. Each check is the paired comparison it always was; what changed is
    who runs it and what it is asked about.
    """
    from scripts.automod import state as S
    try:
        lock = S.Lock(S.STATE_DIR / REGRESSION_LOCK, owner=f"regression-runner-{os.getpid()}").acquire()
    except S.LockHeld:
        logger.info("another regression runner holds the lock; nothing to do")
        return []
    done: list[dict] = []
    try:
        try:
            sweep_stale_scratch()
        except Exception:  # noqa: BLE001 — housekeeping never costs a measurement
            logger.exception("stale scratch sweep failed")
        while max_checks is None or len(done) < max_checks:
            pending = pending_promotions()
            if not pending:
                break
            subject = pending[0]
            logger.info("measuring %s against %s (%d pending)", subject["commit"][:8],
                        str(subject.get("parent"))[:8], len(pending))
            _arm_watchdog(subject["commit"])
            try:
                result = check_promotion(subject, "detached")
            except Exception as exc:  # noqa: BLE001 — one bad check never stops the queue
                logger.exception("regression check of %s crashed", subject["commit"][:8])
                S.append_event({"event": "regression_skipped", "commit": subject["commit"],
                                "reason": f"the check crashed: {exc!r}"[:400]})
                result = _skipped(f"the check crashed: {exc!r}")
            finally:
                _disarm_watchdog()
            done.append({"commit": subject["commit"], **{k: result.get(k) for k in
                                                         ("status", "summary", "regressed")}})
            # Said as it happens. The rows `main` prints arrive when the whole
            # queue is done — seventeen checks and two hours later, the first
            # time this ran — and until then this log is the only live trace.
            logger.info("%s: %s — %s", subject["commit"][:8], result.get("status"),
                        str(result.get("summary") or result.get("skipped") or "")[:300])
            if result.get("regressed"):
                break       # a rollback has been requested; let the guardian act first
    finally:
        lock.release()
    return done


def _skip(commit: str, reason: str, *, level: int = logging.ERROR) -> dict[str, Any]:
    """Record "cannot evaluate" FOR A COMMIT, and return the skipped result.

    The row used to carry a reason and no commit, so nothing could tell which
    promotion went unmeasured, or how many times."""
    from scripts.automod import state as S
    logger.log(level, "%s: %s", commit[:8], reason)
    S.append_event({"event": "regression_skipped", "commit": commit, "reason": reason})
    _announce_if_given_up(commit, reason)
    return _skipped(reason)


def _announce_if_given_up(commit: str, reason: str) -> None:
    """One toast when a promotion has come back "cannot evaluate" for the last
    time. The runner's log is a file nobody tails — this check's errors used to
    land in `server.err` — and a promotion the queue has stopped offering is
    exactly the silence the coverage gauge exists to break. News, not an
    incident (`promote.announce`): the skip rows are already the record."""
    try:
        from scripts.automod import promote as P, state as S
        skips = sum(1 for e in S.read_events(limit=4000)
                    if e.get("event") == "regression_skipped" and e.get("commit") == commit)
        if skips == MAX_SKIPS_PER_COMMIT:
            P.announce(f"Regression check gave up on {commit[:8]}",
                       f"It could not be evaluated {skips} times and will not be tried again: "
                       f"{reason[:240]}")
    except Exception:  # noqa: BLE001 — an announcement never costs the queue
        logger.debug("give-up announcement failed", exc_info=True)


def _would_regress(current: dict | None, baseline: dict | None, noise: dict,
                   floor_stale: bool = False) -> bool:
    """Whether these two arms, as they stand, would be reported as a regression:
    both measurable, and `evaluate` says so. The question `check_promotion` asks
    while the pinned corpus is still up, to decide whether to look twice.

    A stale floor answers this `False` on purpose (#1352): the second look exists
    to decide whether a rollback request is justified, and `check_promotion`
    cannot justify one against a floor measured against a different question set,
    so the extra arm would cost one eval run to confirm a verdict that is already
    going to be withheld.
    """
    if floor_stale:
        return False
    if not baseline or not current or all_queries_empty(baseline):
        return False
    for blob in (baseline, current):
        # `empty_fact_leg` (#1250) belongs in this pre-check for the same reason
        # the other two guards do: an arm whose fact leg read nothing is refused
        # below as a non-measurement, so spending a second arm to "confirm" the
        # 0.0 it produced would buy nothing.
        if (blob.get("corpus_ok") is False or unanswered_doc_queries(blob)
                or empty_fact_leg(blob)):
            return False
    try:
        return bool(evaluate(current["overall"], baseline["overall"], noise,
                             CONTEXT_PAIRED_CHECK)[0])
    except Exception:  # noqa: BLE001 — the real evaluation below will say why
        return False


def all_queries_empty(arm: dict) -> bool:
    """Every question of the arm came back with no document at all."""
    total = int(arm.get("n_records") or 0)
    return total > 0 and len(arm.get("empty_doc_queries") or []) >= total


def eval_last_payload(*, commit: str, measured_at: str, baseline_commit: str,
                      stage: str, current: dict, pin: dict, regressed: bool,
                      reasons: list, latency_reading: dict | None,
                      latency_verdict: dict | None) -> dict:
    """The record the guardian folds into `last_known_good.json`'s `eval` slot.

    Built here, as one function with one call site, so the field list is
    assertable without running an eval arm. Two fields exist because of what a
    reader of the green result cannot see in it: `AXIS_FIELD` names what this
    measurement covered and what it did not (#829), and `commit` is the commit it
    was measured on — which is NOT necessarily the commit the guardian ends up
    recording, and `gstate.set_lkg` stamps that distinction into the slot rather
    than leaving a three-day-old measurement to read as this promotion's
    baseline.
    """
    return {
        "commit": commit, "measured_at": measured_at,
        "baseline_commit": baseline_commit, "stage": stage,
        "overall": current["overall"], "corpus": current.get("corpus") or {},
        "pin": pin,
        # What this number is a measurement OF, and the two axes it cannot see
        # (agent-loop behaviour, graph edge quality). See `axis_coverage`.
        AXIS_FIELD: axis_coverage(),
        "regressed": regressed, "reasons": reasons,
        # Two keys, two meanings: `latency_budget` is the reading (present for
        # every measurable run, inside or out, so a reader can tell "fast enough"
        # from "never measured"), and `latency_over_budget` is the verdict — null
        # unless this run actually went past the ceiling.
        "latency_budget": latency_reading,
        OVER_BUDGET_FIELD: latency_verdict,
    }


def check_promotion(subject: dict, stage: str) -> dict[str, Any]:
    """The paired comparison for ONE promotion: `subject["commit"]` against
    `subject["parent"]`, both checked out, both on live data, one pinned corpus.

    Both arms run from a scratch worktree now. The current arm used to run the
    live tree, which is the promoted commit only until the next landing —
    exact for "the latest promotion", wrong for a queue.
    """
    from scripts.automod import state as S

    commit = str(subject["commit"])
    baseline_commit = subject.get("parent") or subject.get("rollback_target")
    if not baseline_commit:
        return _skip(commit, "promotion record carries no parent to compare against")

    noise = None
    if NOISE_PATH.exists():
        try:
            noise = json.loads(NOISE_PATH.read_text())
        except ValueError:
            noise = None
    if not noise:
        # Explicitly "cannot evaluate" — never "no regression".
        return _skip(commit, f"no measured noise floor at {NOISE_PATH}; run "
                             f"`automod_regression.measure_noise()` once on an unchanged tree",
                     level=logging.WARNING)

    # Decided BEFORE an arm is spent on a second look. A floor whose σ was
    # measured against a different question set describes a different experiment,
    # and #1352 made that a verdict-ending fact rather than a note in the record:
    # every check on 2026-09-21 carried `noise_floor_stale: True` and still asked
    # for two rollbacks, because nothing read the flag.
    live_fp = queries_fingerprint()
    artifact_fp = str(noise.get("queries_fingerprint") or "")
    stale_floor = artifact_fp != live_fp

    # Both arms run inside ONE pinned corpus: a frozen qmd snapshot plus a
    # single grep root. Without that the comparison measures the corpus as
    # much as the code — this retriever searches the repository it ships in,
    # so each arm was grepping its own source.
    pin_provenance: dict = {}
    work = Path(tempfile.mkdtemp(prefix="automod-pin-"))
    baseline = current = confirm = None
    confirm_ran = False
    ranker: dict = {}
    try:
        with PinnedCorpus(work) as pin:
            # Before anything is timed against it: the first query loads the
            # models, and the retriever's client gives up at 15 s while qmd
            # keeps working — one slow query and every later one queues behind
            # it (2026-09-18: a whole arm at 116-160 s a query, all empty).
            warm = getattr(pin, "warm_up", None)
            if callable(warm):
                warm()
            pin_provenance = dict(pin.provenance)
            with _baseline_worktree(baseline_commit) as wt:
                if wt is None:
                    return _skip(commit, "baseline worktree failed — cannot evaluate")
                with _baseline_worktree(commit) as cur:
                    if cur is None:
                        return _skip(commit, "worktree of the promoted commit failed — cannot evaluate")
                    # The baseline tree is the shared grep corpus for BOTH arms.
                    env = {**pin.env_for(code_root=wt), QMD_TIMEOUT_ENV: str(EVAL_QMD_TIMEOUT_S)}
                    # djev answers the same request differently each time, so
                    # every arm runs under one replay file anchored on the
                    # baseline: an identical rank request gets the baseline's
                    # answer, a changed one a fresh draw (`ranker_reading`).
                    replay_db = work / "djev-replay.sqlite"
                    baseline = _run_arm(wt, "automod-paired-lkg",
                                        _replayed(env, replay_db, ARM_BASELINE))
                    current = _run_arm(cur, "automod-check",
                                       _replayed(env, replay_db, ARM_CURRENT))
                    # A regression has to reproduce before anyone acts on it.
                    # Under a pinned corpus the armed metrics are deterministic,
                    # so a real one comes back the same; what does not is the
                    # instrument — a recall lost to load, a rank flipped by the
                    # GPU. Every rollback this loop has performed has been a
                    # false positive, this check's own two among them
                    # (2026-09-07: ndcg -0.006; 2026-09-17: one question lost to
                    # the client's timeout), and a second look costs one arm.
                    if _would_regress(current, baseline, _floor_for(
                            noise, ranker_reading(replay_stats(replay_db), ARM_CURRENT)),
                            floor_stale=stale_floor):
                        confirm_ran = True
                        # Its own arm name: a request the change moved is drawn
                        # again rather than replayed from the first current run,
                        # so the second look is an independent draw of the ranker.
                        confirm = _run_arm(cur, "automod-check-confirm",
                                           _replayed(env, replay_db, ARM_CONFIRM))
                    ranker = replay_stats(replay_db)
            pin.discard()
    except PinError as exc:
        # A comparison that quietly fell back to the live daemon would be the
        # unpinned comparison this replaced, wearing its name.
        return _skip(commit, f"pinned corpus unavailable: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not baseline:
        return _skip(commit, "paired baseline run failed — cannot evaluate")
    if not current:
        return _skip(commit, "eval run failed")

    # The BASELINE answering nothing is never the change: that code was live
    # and answering when it landed. It is the pinned daemon or the corpus, and
    # zero against zero is not "no regression" — five checks in a row said so
    # on 2026-09-18 (every document metric 0.0 in both arms, every query at the
    # client's 15 s timeout). Only the CURRENT arm answering nothing is a score:
    # that is the one shape a change under test can produce, and `evaluate`
    # refuses it below.
    if all_queries_empty(baseline):
        return _skip(commit, f"baseline arm answered none of its {baseline.get('n_records')} "
                             f"queries — the pinned daemon or the corpus, not the change; "
                             f"cannot evaluate")

    # An arm that scored an EMPTY corpus is not a low score, it is a
    # measurement that did not happen — and the doc-side metrics read
    # identically with the graph deleted, so it looks like an ordinary result.
    for arm, blob in (("baseline", baseline), ("current", current)):
        if blob.get("corpus_ok") is False:
            return _skip(commit, f"{arm} arm scored an empty corpus "
                                 f"({blob.get('corpus')}) — cannot evaluate")
        unanswered = unanswered_doc_queries(blob)
        if unanswered:
            return _skip(commit, f"{arm} arm got zero documents for {len(unanswered)} of "
                                 f"{blob.get('n_records')} queries ({', '.join(unanswered)}) — "
                                 "the pinned daemon did not answer, cannot evaluate")
        # The fact-leg mirror of the two guards above (#1250), and the one that
        # was missing while six arms on 2026-09-18 recorded
        # `fact_entity_recall_avg: 0.0` against `corpus.facts: 315462`.
        # `fact_entity_recall_avg` is ARMED, and under a pinned corpus the
        # measured stdev is 0.0 so the tolerance is the MIN_SIGMA floor: 0.375 →
        # 0.0 is therefore reported as a regression and becomes a ROLLBACK
        # REASON for a commit that touched nothing in the fact path. The doc
        # guards above cannot catch it — a zeroed fact leg answers every query
        # with documents as normal, so `corpus_ok` stays true,
        # `unanswered_doc_queries` is empty and `all_queries_empty` is False.
        # The failed-read count and the first failure's repr go into the reason:
        # the swallow that hid this discarded the traceback, so whatever the
        # gate arm hits is only nameable from what the arm now records.
        if empty_fact_leg(blob):
            cov = blob.get("fact_coverage") or {}
            first = cov.get("fact_read_first_error")
            return _skip(commit,
                         f"{arm} arm's fact leg read nothing on any of its "
                         f"{cov.get('n_records')} queries while the corpus names "
                         f"{((blob.get('corpus') or {}).get('facts'))} facts "
                         f"({cov.get('n_fact_reads_failed_total')} failed fact reads"
                         f"{f', first: {first}' if first else ', no failure reported'}) — "
                         "cannot evaluate")

    # djev not answering is the instrument, whichever arm it happened in: the
    # recall falls back to the cross-encoder, so that arm ranked with a
    # different ranker. On 2026-09-21 a check ran while djev was being
    # restarted for an experiment (recall latency 2.0 s against 0.63 s) and
    # rolled back a scheduler change for the difference.
    for arm_label, arm_name in (("baseline", ARM_BASELINE), ("current", ARM_CURRENT)):
        failed = {k: v for k, v in (ranker.get(arm_name) or {}).items()
                  if k in REPLAY_FAILURES and v}
        if failed:
            return _skip(commit, f"djev did not answer {sum(failed.values())} rank request(s) "
                                 f"in the {arm_label} arm ({failed}) — those recalls fell back to "
                                 "the cross-encoder, so the arms ranked differently for a reason "
                                 "that is not the change; cannot evaluate")

    # A GATE, not provenance (#1352). A floor whose σ was measured against a
    # different question set describes a different experiment, so it cannot
    # license reverting a commit: the deltas get recorded and reported, and the
    # rollback channel stays shut. This reverses the previous ruling — "provenance,
    # not a gate", on the reasoning that a pinned corpus is deterministic so the
    # floor falls back to MIN_SIGMA either way. That premise failed on
    # 2026-09-21: the doc leg's ranker became djev (`RECALL_RERANKER`, landed
    # 19:45Z), one settled commit measured itself at Δ-0.0090 and Δ+0.0010 on
    # `ndcg10` inside one check, and two promotions were reverted on a floor that
    # was flagged stale on every check that day. What is NOT deterministic is
    # whether the daemon answers every query, and that stays refused above as a
    # non-measurement rather than budgeted for here: a dropped answer is not a
    # low score, and a floor wide enough to absorb it would absorb a real
    # one-query regression too.
    # Which floor: replayed (the ranker a pure function of its input, so the
    # pinned floor holds) or fresh (the change moved what djev was asked, so its
    # own noise is in the comparison). `ranker_reading` says which.
    reading = ranker_reading(ranker, ARM_CURRENT)
    floor_noise = _floor_for(noise, reading)

    regressed, reasons, detail = evaluate(current["overall"], baseline["overall"], floor_noise,
                                          CONTEXT_PAIRED_CHECK, floor_stale=stale_floor)
    unconfirmed: list[str] = []
    confirmed_by: list[str] = []
    if regressed and confirm_ran:
        if not confirm:
            return _skip(commit, "the run that would have confirmed a regression failed — "
                                 "cannot evaluate: " + "; ".join(reasons)[:300])
        # #1250, same hole the paired arms got a guard for and this arm did not:
        # `unanswered_doc_queries` keys on `n_docs` alone, so a confirm arm whose
        # FACT leg read nothing — `n_docs` populated, `n_facts` 0, `corpus_ok`
        # true because the index answers — reached `evaluate` and was accepted as
        # the second independent observation. That is the arm that decides
        # "rollback confirmed" for the commit under review, and with the pinned
        # corpus's stdev of 0.0 its `fact_entity_recall_avg` delta (0.375 → 0.0
        # was measured, 6 arms on 2026-09-18) was the reason. `corpus_ok is
        # False` cannot cover it either: it is the graph half only, and
        # `empty_fact_leg` recomputes from records, so an older `run_eval`
        # artifact with no `fact_leg` block is still caught here.
        if (confirm.get("corpus_ok") is False or unanswered_doc_queries(confirm)
                or empty_fact_leg(confirm)):
            fact_cov = confirm.get("fact_coverage") or {}
            return _skip(
                commit,
                "the run that would have confirmed a regression did not get an answer to "
                "every query — cannot evaluate: " + "; ".join(reasons)[:300]
                + (" [fact leg]" if empty_fact_leg(confirm) else "")
                + (f" {fact_cov.get('n_facts_reports', 0)}/{fact_cov.get('n_records', 0)} "
                   f"queries returned no facts at all; {fact_cov.get('empty_fact_queries') or []}"
                   if empty_fact_leg(confirm) else "")
                + (f", {fact_cov['n_fact_reads_failed_total']} entity fact read(s) failed"
                   if fact_cov.get("n_fact_reads_failed_total") else "no fact read failed"))
        confirm_regressed, confirm_reasons, detail_again = evaluate(
            confirm["overall"], baseline["overall"],
            _floor_for(noise, ranker_reading(ranker, ARM_CONFIRM)), CONTEXT_PAIRED_CHECK,
            floor_stale=stale_floor)
        # #1352 clause 3: the second look decides on MAGNITUDE against the
        # published paired σ, not on whether the metric is still past the floor.
        # `evaluate` answers the sign question, and for two independent draws of a
        # zero-mean ±0.01 variable that question passes often: at 2026-09-21 21:44Z
        # the first pass said `ndcg10` Δ-0.0060 and the confirm pass Δ-0.0200
        # (`mrr_doc` -0.0050 → -0.0340) — the second draw landing three times
        # deeper than the first, which is what a fresh draw of noise does — and the
        # pair was booked `confirmed_by` and reverted a commit that had touched no
        # ranking path. Reasons, detail and the reported arm all come from the
        # second pass from here, so a reason string and its `detail` entry can
        # never disagree about which draw they describe.
        confirm_by = {r.split()[0]: r for r in confirm_reasons if r.split()}
        first_by = {r.split()[0]: r for r in reasons if r.split()}
        past_size, size_rejected = magnitude_confirmed(
            detail, detail_again, floor_noise,
            [m for m in confirm_by if m in ARMED_METRICS])
        # A reason that names no armed metric (`eval errors appeared: ...`) is not
        # a measurement subject to a σ, so it survives the screen unchanged.
        kept = set(past_size) | {m for m in confirm_by if m not in ARMED_METRICS}
        reasons = [confirm_by[m] for m in confirm_by if m in kept]
        confirmed_by = [m for m in confirm_by if m in kept]
        unconfirmed = [first_by[m] for m in first_by if m not in kept]
        if size_rejected:
            # Named in the record, not only in the log: "the drop did not
            # reproduce" and "it reproduced, and BOTH passes are inside the
            # instrument's own spread" are different findings about the
            # instrument, and the second is the one that cost two promotions.
            unconfirmed = [
                f"refused on size, not on direction: {', '.join(size_rejected)} "
                f"cleared the floor on both passes and neither passed "
                f"{SIGMA_MULTIPLIER:g}σ of the paired σ measured against these "
                f"questions, so the pair is the instrument's own swing"] + unconfirmed
        regressed = bool(reasons)
        current, detail = confirm, detail_again
        if not regressed:
            # Same code, same corpus, same questions, a different answer — or an
            # answer that never left the instrument's own spread: either way this
            # is a finding about the instrument, recorded as one.
            logger.warning("regression after %s did NOT reproduce as an effect on a second run "
                           "of the same arm (second pass regressed=%s, past 3σ of the paired σ=%s, "
                           "rejected on size=%s) — not a regression: %s",
                           commit[:8], confirm_regressed, past_size, size_rejected,
                           "; ".join(first_by.values()))
    no_verdict = ""
    if regressed and stale_floor:
        # Clause 1 of #1352. The floor this verdict rests on was measured against
        # a different question set, so the verdict is not this check's to make:
        # record what was seen, report it loudly, and leave the rollback channel
        # alone. `unconfirmed_reasons` keeps the metric strings so the deltas are
        # still greppable in the ledger.
        unconfirmed = reasons + unconfirmed
        no_verdict = (f"stale floor — no verdict: the published noise floor's queries_fingerprint "
                      f"{artifact_fp or '(absent)'} is not the live set {live_fp}, so no floor here "
                      f"is justified for these questions and no rollback is requested; "
                      f"{len(unconfirmed)} metric delta(s) past it are report-only, re-measure with "
                      f"`python -m scripts.automod.regression_runner noise`. Were: "
                      + "; ".join(unconfirmed)[:900])
        reasons = [no_verdict]
        regressed = False
    fact_side_src = reasons + unconfirmed if stale_floor else reasons
    fact_side = [r for r in fact_side_src if r.split() and r.split()[0] in FACT_LAYER_METRICS]

    # The latency verdict, on the same absolute ceiling `evaluate` reports inside
    # its detail. Written into both records a later reader consults — the ledger
    # event and `eval_last.json`, which the guardian folds into the LKG record —
    # because before this the field was written into 20 nightly artifacts and read
    # by nothing. Deliberately NOT in `reasons`: `regressed` is the rollback
    # channel and this is a report.
    latency_reading = over_budget(current["overall"], CONTEXT_PAIRED_CHECK)
    latency_verdict = latency_reading if (latency_reading or {}).get("over") else None
    if latency_verdict:
        logger.warning("latency over budget after %s: %.0f ms > %.0f ms (%s) — "
                       "reported, not a rollback reason", commit[:8],
                       latency_verdict["latency_ms_avg"],
                       latency_verdict["budget_ms"], latency_verdict["context"])

    # Hand the measurement to the guardian, which folds it into the LKG record
    # when this promotion settles. Written here rather than into
    # last_known_good.json directly: that file has exactly one writer, and
    # that is what makes "last known good" mean observed-healthy. The field list
    # lives in `eval_last_payload`, which carries the axis this measured.
    S.write_eval_last(eval_last_payload(
        commit=commit, measured_at=S.now_iso(), baseline_commit=baseline_commit,
        stage=stage, current=current, pin=pin_provenance, regressed=regressed,
        reasons=reasons, latency_reading=latency_reading,
        latency_verdict=latency_verdict))
    S.append_event({"event": "regression_check", "regressed": regressed,
                    "reasons": reasons, "fact_side_reasons": fact_side,
                    "stage": stage, "baseline_commit": baseline_commit,
                    "pin": pin_provenance, "noise_floor_stale": stale_floor,
                    "ranker": {"arms": ranker, "reading": reading,
                               "floor": floor_noise.get("floor")},
                    "latency_budget": latency_reading,
                    OVER_BUDGET_FIELD: latency_verdict,
                    # What a second run of the same arm said about a regression
                    # the first one reported: reproduced, or not.
                    "confirmed_by": confirmed_by if regressed else [],
                    "unconfirmed_reasons": unconfirmed,
                    "detail": detail, "commit": commit})
    if not regressed:
        # A withheld verdict must not read as a clean bill. #1352's shape was an
        # instrument that reported "no regression" on every check while each of
        # those checks carried a flag saying its own floor did not apply, because
        # the no-verdict return was byte-identical to the nothing-moved return.
        if no_verdict:
            summary = (f"STALE FLOOR after {commit[:8]}: no verdict — "
                       f"{len(unconfirmed)} delta(s) report-only, no rollback requested")
        elif unconfirmed:
            summary = (f"no regression after {commit[:8]}: reported delta(s) refused "
                       f"on the second pass")
        else:
            summary = (f"no regression after {commit[:8]} vs "
                       f"{baseline_commit[:8]} ({len(ARMED_METRICS)} armed metrics)")
        # Deliberately no `reason` key: `_execute_blocking` writes that field with
        # the QUEUE's routing decision, and a check-level string here would be
        # silently overwritten before anything read it.
        return {"status": "success", "regressed": False, "stage": stage,
                "summary": summary,
                "unconfirmed_reasons": unconfirmed,
                "detail": detail, "noise_floor_stale": stale_floor}

    logger.error("retrieval-quality regression after %s: %s", commit[:8],
                 "; ".join(reasons))

    # Requested, never performed here. A rollback stops the backend — this
    # process — so an inline `_rollback_inline` call issued the stop that kills
    # its own caller and then never reached `git reset`. The guardian is
    # outside that blast radius, and it also brings evidence preservation,
    # retries, the denylist and flap protection that an inline revert skipped.
    # Clause 5 of #1352. The floors clause goes FIRST because everything that
    # carries this string truncates it (`state.request_rollback` at 2000 chars,
    # the ledger event at 1000) and the per-metric reasons are the long part: the
    # sentence that says whether any floor was justified must survive the cut that
    # the list of metric deltas does not. It lands in the rollback ledger event's
    # `reason`, in the revert commit message, and in this item's summary — the
    # three places a person reading a halt actually looks.
    evidence = metric_evidence(detail, reasons, stale_floor)
    reason = (f"retrieval-quality regression after {commit[:8]} [floors] "
              f"{floors_line(evidence, stale_floor)} | " + "; ".join(reasons))
    S.request_rollback(
        reason=reason,
        trigger="regression", target=baseline_commit, commit=commit,
        changed_paths=subject.get("changed_paths") or [],
        metric_floors=evidence, noise_floor_stale=stale_floor)
    return {"status": "failed", "regressed": True, "reasons": reasons,
            "summary": f"REGRESSION after {commit[:8]}: " + reason,
            "fact_side_reasons": fact_side, "metric_floors": evidence,
            "noise_floor_stale": stale_floor,
            "rollback_requested_to": baseline_commit}


def main(argv: list[str] | None = None) -> int:
    """`python -m scripts.automod.regression_runner run|pending|latest|noise`.

    `run` is what `start_runner` and the promoter spawn, detached. It logs to
    stderr (the caller points that at `regression.log`) and prints one JSON
    line per promotion it measured. `noise` re-measures both noise floors
    (`measure_noise`) under the same lock, so no check reads a half-written
    floor or shares the pinned daemon with the measurement."""
    import argparse
    ap = argparse.ArgumentParser(description="Paired retrieval-quality regression check")
    ap.add_argument("command", choices=("run", "pending", "latest", "noise"))
    ap.add_argument("--max", type=int, default=None, help="stop after this many checks")
    ap.add_argument("--trials", type=int, default=5, help="noise: replayed trials")
    ap.add_argument("--fresh-trials", type=int, default=5, help="noise: fresh-ranker trials")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if args.command == "noise":
        from scripts.automod import state as S
        try:
            lock = S.Lock(S.STATE_DIR / REGRESSION_LOCK,
                          owner=f"regression-noise-{os.getpid()}").acquire()
        except S.LockHeld:
            print("a regression runner holds regression.lock; try again when it is done")
            return 3
        try:
            print(json.dumps(measure_noise(args.trials, args.fresh_trials), indent=2))
        finally:
            lock.release()
        return 0
    if args.command == "pending":
        print(json.dumps(pending_promotions(), indent=2))
        return 0
    if args.command == "latest":
        print(json.dumps(_execute_blocking(), indent=2, default=str))
        return 0
    for row in run_pending(args.max):
        print(json.dumps(row, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
