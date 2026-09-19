"""Nightly behavioural-regression check for a landed self-modification.

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
# A floor, not a measurement: the eval contributes zero variance, so this only
# absorbs float representation wobble. Real tolerance comes from the paired
# comparison, which removes vault drift rather than budgeting for it.
MIN_SIGMA = 0.001
SIGMA_MULTIPLIER = 3.0

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


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    new_id = queue.enqueue(
        source=NAME, kind="check", payload={},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued automod regression check id=%d", new_id)


def _load_run(baselines: Path, label: str) -> dict | None:
    """Newest run record for `label` as {overall, corpus_ok, corpus}.

    `corpus_ok` is what `eval/run_eval.py` records about the DATA it scored.
    An arm that measured an empty graph is not a low score, it is a
    measurement that did not happen — and because the doc-side metrics read
    identically with the graph deleted, it looks like a perfectly ordinary
    result. Older records predate the field; absent is treated as unknown
    rather than as False, so a stale baseline cannot fabricate a regression.
    """
    runs = sorted(Path(baselines).glob(f"*{label}*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return None
    try:
        blob = json.loads(runs[-1].read_text())
        return {"overall": blob["summary"]["overall"],
                "corpus_ok": blob.get("corpus_ok"),
                "corpus": blob.get("corpus") or {},
                **_doc_coverage(blob.get("records") or [])}
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


def measure_noise(trials: int = 5) -> dict:
    """Record mean/stdev per metric on an unchanged tree. Run once, by hand.

    Runs inside the pinned corpus, because that is the condition the armed
    metrics are compared under. A noise floor measured against the live daemon
    describes a different experiment from the one it is used to judge — which
    is exactly how the doc-side metrics came to be armed on a "stdev 0.0000"
    that did not hold when it mattered.
    """
    import tempfile as _tf
    samples: dict[str, list[float]] = {}
    dropped: list[str] = []
    work = Path(_tf.mkdtemp(prefix="automod-noise-pin-"))
    try:
        with PinnedCorpus(work) as pin:
            env = pin.env_for(code_root=LIVE_ROOT)
            for i in range(trials):
                run = _run_arm(LIVE_ROOT, f"automod-noise-{i}", env)
                overall = (run or {}).get("overall")
                if not overall:
                    continue
                # A trial the daemon did not fully answer is not a sample of
                # the eval's noise, exactly as `execute` refuses to score it.
                unanswered = unanswered_doc_queries(run)
                if unanswered:
                    dropped.append(f"trial {i}: {', '.join(unanswered)}")
                    continue
                _accumulate(samples, overall)
            pin.discard()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return _summarise_noise(samples, trials, dropped=dropped)


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


def _summarise_noise(samples: dict, trials: int, *, dropped: list[str] | None = None) -> dict:
    noise = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trials": trials,
        "dropped_trials": list(dropped or []),
        "queries_fingerprint": queries_fingerprint(),
        "pinned": True,
        "metrics": {
            k: {"mean": statistics.fmean(v),
                "stdev": (statistics.stdev(v) if len(v) > 1 else 0.0),
                "min": min(v), "max": max(v), "n": len(v)}
            for k, v in samples.items() if v
        },
    }
    NOISE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOISE_PATH.write_text(json.dumps(noise, indent=2), encoding="utf-8")
    return noise


def evaluate(current: dict, baseline: dict, noise: dict,
             context: str = CONTEXT_PAIRED_CHECK) -> tuple[bool, list[str], dict]:
    """Pure comparison. Returns (regressed, reasons, per-metric detail).

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
    metrics = (noise or {}).get("metrics") or {}

    for key in ARMED_METRICS:
        if key not in current or key not in baseline:
            continue
        now, was = float(current[key]), float(baseline[key])
        sigma = max(float(metrics.get(key, {}).get("stdev", 0.0)), MIN_SIGMA)
        tolerance = SIGMA_MULTIPLIER * sigma
        delta = now - was
        detail[key] = {"before": was, "after": now, "delta": delta,
                       "tolerance": tolerance, "armed": True}
        if delta < -tolerance:
            reasons.append(f"{key} {was:.4f} → {now:.4f} "
                           f"(Δ{delta:+.4f}, beyond {SIGMA_MULTIPLIER:g}σ={tolerance:.4f})")

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
    reverted = {str(e.get("commit") or "") for e in events if e.get("event") == "rollback_succeeded"}
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
SCRATCH_PREFIXES = ("automod-eval-", "automod-pin-")


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
            try:
                result = check_promotion(subject, "detached")
            except Exception as exc:  # noqa: BLE001 — one bad check never stops the queue
                logger.exception("regression check of %s crashed", subject["commit"][:8])
                S.append_event({"event": "regression_skipped", "commit": subject["commit"],
                                "reason": f"the check crashed: {exc!r}"[:400]})
                result = _skipped(f"the check crashed: {exc!r}")
            done.append({"commit": subject["commit"], **{k: result.get(k) for k in
                                                         ("status", "summary", "regressed")}})
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


def _would_regress(current: dict | None, baseline: dict | None, noise: dict) -> bool:
    """Whether these two arms, as they stand, would be reported as a regression:
    both measurable, and `evaluate` says so. The question `check_promotion` asks
    while the pinned corpus is still up, to decide whether to look twice."""
    if not baseline or not current or all_queries_empty(baseline):
        return False
    for blob in (baseline, current):
        if blob.get("corpus_ok") is False or unanswered_doc_queries(blob):
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

    # Both arms run inside ONE pinned corpus: a frozen qmd snapshot plus a
    # single grep root. Without that the comparison measures the corpus as
    # much as the code — this retriever searches the repository it ships in,
    # so each arm was grepping its own source.
    pin_provenance: dict = {}
    work = Path(tempfile.mkdtemp(prefix="automod-pin-"))
    baseline = current = confirm = None
    confirm_ran = False
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
                    baseline = _run_arm(wt, "automod-paired-lkg", env)
                    current = _run_arm(cur, "automod-check", env)
                    # A regression has to reproduce before anyone acts on it.
                    # Under a pinned corpus the armed metrics are deterministic,
                    # so a real one comes back the same; what does not is the
                    # instrument — a recall lost to load, a rank flipped by the
                    # GPU. Every rollback this loop has performed has been a
                    # false positive, this check's own two among them
                    # (2026-09-07: ndcg -0.006; 2026-09-17: one question lost to
                    # the client's timeout), and a second look costs one arm.
                    if _would_regress(current, baseline, noise):
                        confirm_ran = True
                        confirm = _run_arm(cur, "automod-check-confirm", env)
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

    # Provenance, not a gate. Under a pinned corpus every armed metric is
    # deterministic (re-measured 2026-09-17: five trials agree to 0.0000), so
    # the measured stdev is 0.0 and the tolerance falls back to the MIN_SIGMA
    # floor either way — a floor taken against an older question set cannot
    # make the comparison wrong, only its record misleading. Say so rather
    # than silently carrying it. What is NOT deterministic is whether the
    # daemon answers every query, and that is refused above as a
    # non-measurement rather than budgeted for here: one dropped answer is
    # -0.05 doc_hit_rate, and a floor wide enough to absorb it would also
    # absorb a real regression of one query.
    stale_floor = (noise.get("queries_fingerprint") or "") != queries_fingerprint()

    regressed, reasons, detail = evaluate(current["overall"], baseline["overall"], noise,
                                          CONTEXT_PAIRED_CHECK)
    unconfirmed: list[str] = []
    confirmed_by: list[str] = []
    if regressed and confirm_ran:
        if not confirm:
            return _skip(commit, "the run that would have confirmed a regression failed — "
                                 "cannot evaluate: " + "; ".join(reasons)[:300])
        if confirm.get("corpus_ok") is False or unanswered_doc_queries(confirm):
            return _skip(commit, "the run that would have confirmed a regression did not get "
                                 "an answer to every query — cannot evaluate: "
                                 + "; ".join(reasons)[:300])
        again, confirmed_by, detail_again = evaluate(confirm["overall"], baseline["overall"],
                                                     noise, CONTEXT_PAIRED_CHECK)
        if not again:
            # Same code, same corpus, same questions, a different answer: that
            # is a finding about the instrument, and it is recorded as one.
            logger.warning("regression after %s did NOT reproduce on a second run of the same "
                           "arm — not a regression: %s", commit[:8], "; ".join(reasons))
            unconfirmed, regressed, reasons = reasons, False, []
            current, detail = confirm, detail_again
    fact_side = [r for r in reasons if r.split()[0] in FACT_LAYER_METRICS]

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
    # that is what makes "last known good" mean observed-healthy.
    S.write_eval_last({
        "commit": commit, "measured_at": S.now_iso(),
        "baseline_commit": baseline_commit, "stage": stage,
        "overall": current["overall"], "corpus": current.get("corpus") or {},
        "pin": pin_provenance,
        "regressed": regressed, "reasons": reasons,
        # Two keys, two meanings: `latency_budget` is the reading (present for
        # every measurable run, inside or out, so a reader can tell "fast enough"
        # from "never measured"), and `latency_over_budget` is the verdict — null
        # unless this run actually went past the ceiling.
        "latency_budget": latency_reading,
        OVER_BUDGET_FIELD: latency_verdict,
    })
    S.append_event({"event": "regression_check", "regressed": regressed,
                    "reasons": reasons, "fact_side_reasons": fact_side,
                    "stage": stage, "baseline_commit": baseline_commit,
                    "pin": pin_provenance, "noise_floor_stale": stale_floor,
                    "latency_budget": latency_reading,
                    OVER_BUDGET_FIELD: latency_verdict,
                    # What a second run of the same arm said about a regression
                    # the first one reported: reproduced, or not.
                    "confirmed_by": confirmed_by if regressed else [],
                    "unconfirmed_reasons": unconfirmed,
                    "detail": detail, "commit": commit})
    if not regressed:
        return {"status": "success", "regressed": False, "stage": stage,
                "summary": f"no regression after {commit[:8]} vs "
                           f"{baseline_commit[:8]} ({len(ARMED_METRICS)} armed metrics)",
                "detail": detail, "noise_floor_stale": stale_floor}

    logger.error("behavioural regression after %s: %s", commit[:8], "; ".join(reasons))

    # Requested, never performed here. A rollback stops the backend — this
    # process — so an inline `_rollback_inline` call issued the stop that kills
    # its own caller and then never reached `git reset`. The guardian is
    # outside that blast radius, and it also brings evidence preservation,
    # retries, the denylist and flap protection that an inline revert skipped.
    S.request_rollback(
        reason=f"behavioural regression after {commit[:8]}: " + "; ".join(reasons),
        trigger="regression", target=baseline_commit, commit=commit,
        changed_paths=subject.get("changed_paths") or [])
    return {"status": "failed", "regressed": True, "reasons": reasons,
            "summary": f"REGRESSION after {commit[:8]}: " + "; ".join(reasons),
            "fact_side_reasons": fact_side,
            "rollback_requested_to": baseline_commit}


def main(argv: list[str] | None = None) -> int:
    """`python -m scripts.automod.regression_runner run|pending|latest`.

    `run` is what `start_runner` and the promoter spawn, detached. It logs to
    stderr (the caller points that at `regression.log`) and prints one JSON
    line per promotion it measured."""
    import argparse
    ap = argparse.ArgumentParser(description="Paired behavioural-regression check")
    ap.add_argument("command", choices=("run", "pending", "latest"))
    ap.add_argument("--max", type=int, default=None, help="stop after this many checks")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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
