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
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.selfmod-regression")

NAME = "selfmod-regression"
DEFAULT_PRIORITY = 70
DEDUP_KEY = "selfmod:regression"

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
NOISE_PATH = LIVE_ROOT / "eval" / "baselines" / "selfmod-noise.json"

# Metrics that were bit-identical across repeated runs. Any movement in these
# is signal. Everything else in the eval (ndcg10, doc_hit_rate) is reported
# but never fires — their measured spread is a single query's worth.
# Every one of these measured stdev 0.0000 over five runs on an unchanged
# vault, so all of them are armed. `latency_ms_avg` is the only metric that
# moves run to run (562ms stdev) and is never compared.
ARMED_METRICS = ("entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg",
                 "ndcg10", "mrr_doc", "doc_hit_rate", "doc_recall_avg")
REPORT_ONLY = ("latency_ms_avg", "n_queries")
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


def _run_eval(label: str, timeout: float = 600.0) -> dict | None:
    """Run the deterministic vault-recall eval and return its overall summary."""
    python = LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    r = subprocess.run(
        [str(python), str(LIVE_ROOT / "eval" / "run_eval.py"), "--label", label],
        cwd=str(LIVE_ROOT), capture_output=True, text=True, timeout=timeout, check=False,
    )
    if r.returncode != 0:
        logger.error("run_eval failed: %s", (r.stdout + r.stderr)[-500:])
        return None
    runs = sorted((LIVE_ROOT / "eval" / "baselines").glob(f"*{label}*.json"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        return None
    try:
        return json.loads(runs[-1].read_text())["summary"]["overall"]
    except (OSError, ValueError, KeyError):
        return None



def _run_eval_paired(lkg_commit: str, timeout: float = 900.0) -> dict | None:
    """Run the eval against `lkg_commit`'s CODE and the LIVE data.

    The point is to hold the data fixed. `LLOYD_FACTS_ROOT` and `LLOYD_KG_DB`
    exist so a rebuild can extract into a fresh tree without touching the live
    one; here they are used the other way round — old code, current data — so
    the only difference between the two arms is the commit.
    """
    import os
    import shutil
    import tempfile

    from app.paths import VAULT_FACTS_ROOT, VAULT_KG_DB

    scratch = Path(tempfile.mkdtemp(prefix="selfmod-eval-"))
    wt = scratch / "lloyd"
    try:
        r = subprocess.run(["git", "-C", str(LIVE_ROOT), "worktree", "add",
                            "--detach", "-q", str(wt), lkg_commit],
                           capture_output=True, text=True, check=False)
        if r.returncode != 0:
            logger.error("paired eval worktree failed: %s", r.stderr[-300:])
            return None
        python = LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
        env = {
            **os.environ,
            "PYTHONPATH": str(wt),
            "LLOYD_FACTS_ROOT": str(VAULT_FACTS_ROOT),   # live data, old code
            "LLOYD_KG_DB": str(VAULT_KG_DB),
        }
        label = "selfmod-paired-lkg"
        r = subprocess.run([str(python), str(wt / "eval" / "run_eval.py"),
                            "--label", label],
                           cwd=str(wt), env=env, capture_output=True, text=True,
                           timeout=timeout, check=False)
        if r.returncode != 0:
            logger.error("paired eval run failed: %s", (r.stdout + r.stderr)[-500:])
            return None
        runs = sorted((wt / "eval" / "baselines").glob(f"*{label}*.json"),
                      key=lambda p: p.stat().st_mtime)
        if not runs:
            return None
        return json.loads(runs[-1].read_text())["summary"]["overall"]
    except Exception as exc:
        logger.error("paired eval error: %s", exc)
        return None
    finally:
        subprocess.run(["git", "-C", str(LIVE_ROOT), "worktree", "remove",
                        "--force", str(wt)], capture_output=True, check=False)
        subprocess.run(["git", "-C", str(LIVE_ROOT), "worktree", "prune"],
                       capture_output=True, check=False)
        shutil.rmtree(scratch, ignore_errors=True)


def measure_noise(trials: int = 5) -> dict:
    """Record mean/stdev per metric on an unchanged tree. Run once, by hand."""
    samples: dict[str, list[float]] = {}
    for i in range(trials):
        overall = _run_eval(f"selfmod-noise-{i}")
        if not overall:
            continue
        for key, value in overall.items():
            if isinstance(value, (int, float)):
                samples.setdefault(key, []).append(float(value))
    noise = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trials": trials,
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
    from scripts.selfmod import state as S

    current_promo = S.read_current()
    if not current_promo:
        return {"skipped": "no promotion under observation"}
    landed = float(current_promo.get("landed_ts") or 0)
    if time.time() - landed > 24 * 3600:
        return {"skipped": "last promotion is older than 24h"}

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

    lkg = S.read_lkg() or {}
    lkg_commit = lkg.get("commit")
    if not lkg_commit:
        return {"skipped": "no last-known-good commit to compare against"}

    # Paired, same-window, same-data. A recorded baseline from the last
    # promotion would measure vault drift instead of the code change.
    baseline = _run_eval_paired(lkg_commit)
    if not baseline:
        S.append_event({"event": "regression_skipped",
                        "reason": "paired baseline run failed — cannot evaluate"})
        return {"skipped": "paired baseline run failed"}

    current = _run_eval("selfmod-check")
    if not current:
        S.append_event({"event": "regression_skipped", "reason": "eval run failed"})
        return {"skipped": "eval run failed"}

    regressed, reasons, detail = evaluate(current, baseline, noise)
    S.append_event({"event": "regression_check", "regressed": regressed,
                    "reasons": reasons, "detail": detail,
                    "commit": current_promo.get("commit")})
    if not regressed:
        return {"regressed": False, "detail": detail}

    logger.error("behavioural regression after %s: %s",
                 str(current_promo.get("commit"))[:8], "; ".join(reasons))
    target = lkg.get("commit")
    if not target:
        return {"regressed": True, "reasons": reasons,
                "action": "none — no last-known-good to revert to"}

    from scripts.selfmod.promote import _rollback_inline
    _rollback_inline(LIVE_ROOT, target)
    S.append_event({"event": "rollback_succeeded", "trigger": "regression",
                    "commit": current_promo.get("commit"), "restored": target,
                    "reasons": reasons})
    return {"regressed": True, "reasons": reasons, "rolled_back_to": target}
