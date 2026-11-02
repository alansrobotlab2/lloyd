"""The held-out bench slice — the dev set AutoDesign keeps away from its optimizer.

AutoDesign (arXiv:2608.13560) accepts a harness change on two conditions, not one:
`J_train(H') > J_train(H)` AND `J_dev(H') >= J_dev(H)`, with the dev slice
deliberately hidden from the optimizer. Lloyd's promotion gate had no split at
all: `evaluate_promotion()` averaged a win fraction over the whole bench — every task
`_recent_baseline_failures()` printed, by name and with its score, into the proposer's
prompt. The proposer aimed at the tasks it was then scored on.

No bench size belongs in this file (#3553). Every count this docstring carried was a
live census of `cfg.paths.bench_dir`, and the corpus grows on its own: one nightly
bench addition moved all of them at once and nothing went red, because no test reads
this prose — so the sentences went quietly false and kept documenting a bench the
loader had not seen for weeks. Where a figure is needed, name the measurement:

* **how big the bench is, and of what** — the `category` census over
  `load_bench_tasks(cfg.paths.bench_dir)`, the loader this module's own `heldout_ids()`
  falls back to. `tests/test_bench_split.py`'s `LIVE_BENCH_CATEGORIES` node already fails
  when a bench file has no entry in that census, when an entry outlives its file, or when
  a declared `category:` disagrees with what the loader returns — the drift that actually
  moves a task across `TARGETED_CATEGORIES` / `HELDOUT_CATEGORIES`. That is the protection
  a number here only imitates, and unlike a number it cannot go stale.
* **what a round's slices were** — the `targeted` and `heldout` lists (those two keys,
  as `_payload` writes them) in the `bench_split.json` this module writes, beside the
  `rotation_size` that produced them and the `rotated_into_heldout` that names which
  tasks moved. The sizes are constant across rounds
  (`test_slice_sizes_are_constant_across_rounds`), so one round's file answers for every
  round.

The partition is by category, because that is the axis a hypothesis can name:

* **targeted** — `replay` + `synthetic`: the pool a variant is allowed to aim at,
  and the only pool whose failures may reach the proposer.
* **held-out** — `adversarial` + `safety`: always in the veto. These
  are the tasks that must refuse a contract that buys shape-benchmark points by
  making an untargeted probe worse — the 2026-09-08 case in `promote.py`.

Rotating whole categories is not possible with this bench: the categories are very
unevenly sized, so rotating a whole one would swing the two slices by a step set by
whichever category was chosen and re-measure the gate on a different denominator every
round.
Instead the two fixed categories stay held out and `ROTATION_SIZE` tasks rotate out of
the targeted pool into the veto, selected by `sha256(round_id || task_id)`. Slice sizes
are constant and no replay or synthetic task is permanently exempt from veto duty,
which is what "no slice is permanently privileged" has to mean however large the corpus
gets — the reason that sentence no longer carries a task count is the one above.

The rule is constant; the reading is not, and a reading is worth exactly its date and
its artifact. A round's slices are recorded in its report's `## Bench split` section:
**23 targeted / 5 held-out of 28 tasks (8 replay + 17 synthetic + 2 adversarial +
1 safety), read 2026-11-02 from `rounds/R_20261102_023124.md` `## Bench split`**, whose
`held-out` line names the two rotated tasks, and re-measured the same way on the
corpus as read 2026-10-29 from `rounds/R_20261029_141819.md`. That is the only place in
this file a corpus figure is stated, and it is stated so a later reader can tell whether
they may still quote it — the answer will be no, and the date is what says so. Quote
neither figure at a round: read that round's `bench_split.json`.

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

# Tasks pulled from the targeted pool into the veto each round. What that yields is
# arithmetic over the corpus, not a property of this constant: the targeted slice is
# the replay+synthetic count minus ROTATION_SIZE and the held-out slice is the
# adversarial+safety count plus it, so the slice sizes a round used are a question for
# that round's bench_split.json (`targeted`, `heldout`, `rotation_size`), and the
# census behind them belongs to `load_bench_tasks(cfg.paths.bench_dir)`. #3553: this
# comment used to state both slice sizes as fact, and the corpus outgrew it in one
# nightly addition with no test able to notice.
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
