"""The held-out bench slice — the dev set AutoDesign keeps away from its optimizer.

AutoDesign (arXiv:2608.13560) accepts a harness change on two conditions, not one:
`J_train(H') > J_train(H)` AND `J_dev(H') >= J_dev(H)`, with the dev slice
deliberately hidden from the optimizer. Lloyd's promotion gate had no split at
all: `evaluate_promotion()` averaged a win fraction over the same 11 tasks that
`_recent_baseline_failures()` printed, by name and with their scores, into the
proposer's prompt. The proposer aimed at the tasks it was then scored on.

The partition is by category, because that is the axis a hypothesis can name:

* **targeted** — `replay` + `synthetic` (8 tasks): the pool a variant is allowed
  to aim at, and the only pool whose failures may reach the proposer.
* **held-out** — `adversarial` + `safety` (3 tasks): always in the veto. These
  are the tasks that must refuse a contract that buys shape-benchmark points by
  making an untargeted probe worse — the 2026-09-08 case in `promote.py`.

Rotating whole categories is not possible with this bench: the categories are
4/4/2/1 tasks, so a category-level rotation would swing the slices between 3/8
and 9/2 and re-measure the gate on a different denominator every round. Instead
the two fixed categories stay held out and `ROTATION_SIZE` tasks rotate out of
the targeted pool into the veto, selected by `sha256(round_id || task_id)`.
Slice sizes are constant (6 targeted / 5 held-out) and no replay or synthetic
task is permanently exempt from veto duty, which is what "no slice is
permanently privileged" has to mean at 11 tasks.

The split is written to `bench_split.json` BEFORE the proposer runs, with a
`split_hash` over the pools, so a split cannot be re-picked after seeing scores:
`load_split()` recomputes the hash and returns nothing if it disagrees.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import AutoresearchConfig, now_iso

# Categories a hypothesis may be told about, and the categories that veto it.
TARGETED_CATEGORIES = ("replay", "synthetic")
HELDOUT_CATEGORIES = ("adversarial", "safety")

# Tasks pulled from the targeted pool into the veto each round. With the live
# bench (4 replay + 4 synthetic + 2 adversarial + 1 safety) this makes every
# round a 6-vote targeted slice against a 5-task held-out slice.
ROTATION_SIZE = 2

SPLIT_FILENAME = "bench_split.json"


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ids_by_category(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    """category → sorted task ids, from loaded bench tasks."""
    out: dict[str, list[str]] = {}
    for t in tasks:
        tid = t.get("id") or t.get("_path")
        if not tid:
            continue
        out.setdefault(str(t.get("category", "unknown")), []).append(str(tid))
    return {cat: sorted(ids) for cat, ids in sorted(out.items())}


def _rotated(targeted_pool: list[str], round_id: str, k: int) -> list[str]:
    """Deterministically pick `k` targeted ids to serve in the veto this round.

    Ordered by `sha256(round_id || task_id)` rather than `random.Random`, so the
    choice does not depend on the Python build and can be re-derived by anyone
    holding the round id.
    """
    if k <= 0 or len(targeted_pool) <= k:
        return []
    ranked = sorted(targeted_pool, key=lambda tid: _digest(str(round_id), tid))
    return ranked[:k]


def _payload(round_id: str, targeted: list[str], heldout: list[str], rotated: list[str]) -> dict[str, Any]:
    return {
        "round_id": round_id,
        "targeted_categories": list(TARGETED_CATEGORIES),
        "heldout_categories": list(HELDOUT_CATEGORIES),
        "rotation_size": ROTATION_SIZE,
        "rotated_into_heldout": sorted(rotated),
        "targeted": sorted(targeted),
        "heldout": sorted(heldout),
    }


def compute_split(tasks: list[dict[str, Any]], round_id: str) -> dict[str, Any]:
    """Partition `tasks` (loaded bench tasks) for `round_id`. Pure."""
    by_cat = _ids_by_category(tasks)
    pool = [tid for cat in TARGETED_CATEGORIES for tid in by_cat.get(cat, [])]
    fixed = [tid for cat in HELDOUT_CATEGORIES for tid in by_cat.get(cat, [])]
    if not pool or not fixed:
        raise RuntimeError(
            f"bench produces no {'targeted' if not pool else 'held-out'} tasks "
            f"(categories seen: {sorted(by_cat)}) — refusing to gate with a one-sided split"
        )
    rotated = _rotated(pool, round_id, ROTATION_SIZE)
    targeted = [tid for tid in pool if tid not in set(rotated)]
    heldout = sorted(fixed + rotated)
    split = _payload(round_id, targeted, heldout, rotated)
    split["split_hash"] = _digest(json.dumps(_payload(round_id, targeted, heldout, rotated),
                                            sort_keys=True))
    split["created_at"] = now_iso()
    return split


def verify(split: dict[str, Any]) -> bool:
    """Does `split_hash` still describe these pools? Recomputed, never trusted."""
    recorded = split.get("split_hash")
    if not recorded:
        return False
    body = _payload(str(split.get("round_id", "")),
                    list(split.get("targeted") or []),
                    list(split.get("heldout") or []),
                    list(split.get("rotated_into_heldout") or []))
    return _digest(json.dumps(body, sort_keys=True)) == recorded


def split_path(cfg: AutoresearchConfig) -> Path:
    return cfg.paths.research_root / SPLIT_FILENAME


def write_split(cfg: AutoresearchConfig, tasks: list[dict[str, Any]], round_id: str) -> dict[str, Any]:
    """Write this round's split before any variant is proposed. Returns it."""
    split = compute_split(tasks, round_id)
    path = split_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    return split


def load_split(cfg: AutoresearchConfig) -> dict[str, Any] | None:
    """The current round's split, or None if it is absent or its hash disagrees.

    A tampered split is reported as absent rather than repaired: callers fall
    back to the category default, which is stricter about what reaches the
    proposer, not looser.
    """
    path = split_path(cfg)
    try:
        split = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(split, dict):
        return None
    return split if verify(split) else None


def heldout_ids(cfg: AutoresearchConfig, tasks: list[dict[str, Any]] | None = None) -> set[str]:
    """Task ids that must not be named to the proposer.

    Uses the round's written split when there is one. With no split on disk —
    a hand-run generator, a unit test, a crashed round's leftovers — it still
    holds out the fixed categories, because the leak this guards is worst
    exactly when nobody remembered to set up.
    """
    split = load_split(cfg)
    if split:
        return set(split.get("heldout") or [])
    if tasks is None:
        from .common import load_bench_tasks
        tasks = load_bench_tasks(cfg.paths.bench_dir)
    by_cat = _ids_by_category(tasks)
    return {tid for cat in HELDOUT_CATEGORIES for tid in by_cat.get(cat, [])}
