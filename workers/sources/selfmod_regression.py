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

from scripts.selfmod.evalpin import PinError, PinnedCorpus
from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.selfmod-regression")

NAME = "selfmod-regression"
DEFAULT_PRIORITY = 70
DEDUP_KEY = "selfmod:regression"

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
# Both arms score THESE questions, whichever commit's code is running.
LIVE_QUERIES = LIVE_ROOT / "eval" / "vault_recall_queries.yaml"

# Deliberately NOT under eval/baselines/. That directory holds eval RUN
# RECORDS, and `tests/test_eval_scorer.py` globs `*.json` there and asserts
# every file carries a run record's fields — so parking a noise summary in it
# fails an unrelated test. It also belongs with the loop's other runtime state,
# which lives outside the repo so it survives a rollback.
NOISE_PATH = Path(os.environ.get(
    "LLOYD_SELFMOD_STATE",
    Path.home() / ".local" / "state" / "lloyd-selfmod")) / "eval-noise.json"

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
# `latency_ms_avg` is the one thing that still moves, by ~1.7s between a cold
# and warm embedding cache, and it is never compared.
ARMED_METRICS = ("entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg",
                 "ndcg10", "mrr_doc", "doc_hit_rate", "doc_recall_avg")
REPORT_ONLY = ("latency_ms_avg", "n_queries")

# What the armed set can and cannot see, MEASURED rather than assumed.
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
# a stated limit in architecture/self-modification.md §13, not a covered case.
#
# `test_selfmod_doc_claims` pins the split. Re-arming a doc-side metric means
# first making the doc corpus part of the pairing — pointing both arms at a
# pinned qmd index instead of the live daemon.
FACT_LAYER_METRICS = ("entity_hit_rate", "entity_recall_avg",
                      "fact_entity_recall_avg")
# A floor, not a measurement: the eval contributes zero variance, so this only
# absorbs float representation wobble. Real tolerance comes from the paired
# comparison, which removes vault drift rather than budgeting for it.
MIN_SIGMA = 0.001
SIGMA_MULTIPLIER = 3.0


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    new_id = queue.enqueue(
        source=NAME, kind="check", payload={},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued selfmod regression check id=%d", new_id)


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
                "corpus": blob.get("corpus") or {}}
    except (OSError, ValueError, KeyError):
        return None


@contextmanager
def _baseline_worktree(commit: str):
    """Check `commit` out into a scratch worktree for the duration of the block.

    Kept open across BOTH arms, not just the baseline one: it is also the
    shared grep corpus. See `_run_arm`.
    """
    scratch = Path(tempfile.mkdtemp(prefix="selfmod-eval-"))
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
    work = Path(_tf.mkdtemp(prefix="selfmod-noise-pin-"))
    try:
        with PinnedCorpus(work) as pin:
            env = pin.env_for(code_root=LIVE_ROOT)
            for i in range(trials):
                run = _run_arm(LIVE_ROOT, f"selfmod-noise-{i}", env)
                overall = (run or {}).get("overall")
                if not overall:
                    continue
                _accumulate(samples, overall)
            pin.discard()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return _summarise_noise(samples, trials)


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


def _summarise_noise(samples: dict, trials: int) -> dict:
    noise = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trials": trials,
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


def evaluate(current: dict, baseline: dict, noise: dict) -> tuple[bool, list[str], dict]:
    """Pure comparison. Returns (regressed, reasons, per-metric detail)."""
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

    if current.get("errors", 0) and not baseline.get("errors", 0):
        reasons.append(f"eval errors appeared: 0 → {current['errors']}")

    return bool(reasons), reasons, detail


async def execute(item: QueueItem) -> dict[str, Any]:
    """Compare quality against the promotion's PARENT, on live data, paired.

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
    from scripts.selfmod import state as S

    promo = S.read_current()
    if promo and promo.get("state") == "observing":
        subject, stage = promo, "observing"
    else:
        subject, stage = S.read_last_settled(), "settled"
    if not subject or not subject.get("commit"):
        return {"skipped": "no recent promotion to check"}

    commit = subject["commit"]
    landed = float(subject.get("landed_ts") or 0)
    if not landed or time.time() - landed > 24 * 3600:
        return {"skipped": "last promotion is older than 24h"}

    # One measurement per promotion. Both arms are full evals plus a worktree,
    # and the answer cannot change while the commit does not.
    for ev in S.read_events(limit=200):
        if ev.get("event") == "regression_check" and ev.get("commit") == commit:
            return {"skipped": f"{commit[:8]} already checked"}

    baseline_commit = subject.get("parent") or subject.get("rollback_target")
    if not baseline_commit:
        return {"skipped": "promotion record carries no parent to compare against"}

    noise = None
    if NOISE_PATH.exists():
        try:
            noise = json.loads(NOISE_PATH.read_text())
        except ValueError:
            noise = None
    if not noise:
        # Explicitly "cannot evaluate" — never "no regression".
        msg = (f"no measured noise floor at {NOISE_PATH}; run "
               f"`selfmod_regression.measure_noise()` once on an unchanged tree")
        logger.warning(msg)
        S.append_event({"event": "regression_skipped", "reason": msg})
        return {"skipped": msg}

    # Both arms run inside ONE pinned corpus: a frozen qmd snapshot plus a
    # single grep root. Without that the comparison measures the corpus as
    # much as the code — this retriever searches the repository it ships in,
    # so each arm was grepping its own source.
    pin_provenance: dict = {}
    work = Path(tempfile.mkdtemp(prefix="selfmod-pin-"))
    try:
        with PinnedCorpus(work) as pin:
            pin_provenance = dict(pin.provenance)
            with _baseline_worktree(baseline_commit) as wt:
                if wt is None:
                    S.append_event({"event": "regression_skipped",
                                    "reason": "baseline worktree failed — cannot evaluate"})
                    return {"skipped": "baseline worktree failed"}
                # The baseline tree is the shared grep corpus for BOTH arms.
                env = pin.env_for(code_root=wt)
                baseline = _run_arm(wt, "selfmod-paired-lkg", env)
                current = _run_arm(LIVE_ROOT, "selfmod-check", env)
            pin.discard()
    except PinError as exc:
        # A comparison that quietly fell back to the live daemon would be the
        # unpinned comparison this replaced, wearing its name.
        msg = f"pinned corpus unavailable: {exc}"
        logger.error(msg)
        S.append_event({"event": "regression_skipped", "reason": msg})
        return {"skipped": msg}
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not baseline:
        S.append_event({"event": "regression_skipped",
                        "reason": "paired baseline run failed — cannot evaluate"})
        return {"skipped": "paired baseline run failed"}
    if not current:
        S.append_event({"event": "regression_skipped", "reason": "eval run failed"})
        return {"skipped": "eval run failed"}

    # An arm that scored an EMPTY corpus is not a low score, it is a
    # measurement that did not happen — and the doc-side metrics read
    # identically with the graph deleted, so it looks like an ordinary result.
    for arm, blob in (("baseline", baseline), ("current", current)):
        if blob.get("corpus_ok") is False:
            msg = (f"{arm} arm scored an empty corpus "
                   f"({blob.get('corpus')}) — cannot evaluate")
            logger.error(msg)
            S.append_event({"event": "regression_skipped", "reason": msg})
            return {"skipped": msg}

    # Provenance, not a gate. Under a pinned corpus every armed metric is
    # deterministic, so the measured stdev is 0.0 and the tolerance falls back
    # to the MIN_SIGMA floor either way — a floor taken against an older
    # question set cannot make the comparison wrong, only its record
    # misleading. Say so rather than silently carrying it.
    stale_floor = (noise.get("queries_fingerprint") or "") != queries_fingerprint()

    regressed, reasons, detail = evaluate(current["overall"], baseline["overall"], noise)
    fact_side = [r for r in reasons if r.split()[0] in FACT_LAYER_METRICS]

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
    })
    S.append_event({"event": "regression_check", "regressed": regressed,
                    "reasons": reasons, "fact_side_reasons": fact_side,
                    "stage": stage, "baseline_commit": baseline_commit,
                    "pin": pin_provenance, "noise_floor_stale": stale_floor,
                    "detail": detail, "commit": commit})
    if not regressed:
        return {"regressed": False, "stage": stage, "detail": detail,
                "noise_floor_stale": stale_floor}

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
    return {"regressed": True, "reasons": reasons, "fact_side_reasons": fact_side,
            "rollback_requested_to": baseline_commit}
