"""Promotion pipeline — snapshot, atomic swap, fact write.

A promotion executes when the targeted slice strictly improves, the held-out
slice does not decline, the variant strictly beats baseline on at least N% of the
targeted tasks (a tie is NOT a win — the tie rule and its purpose are written at
the counting line in `slice_metrics`), and every safety probe passes.
Before swapping, we snapshot the current state
(SOUL.md, MEMORY.md, USER.md) into `_pipeline/research/snapshots/<ts>/`.
After swap, we write the winning experiment as a fact under
`cfg.paths.facts_experiments_dir/<variant_id>/` (configured in config.yaml,
currently `~/lloyd/_pipeline/vault-derived/facts/experiments/`) so it's
queryable via the normal memory pipeline.

`rollback(snapshot_ts)` reverses a promotion by restoring files from the
named snapshot. Both directions reach the live vault through
`scripts.automod.vault_round.land()`, so whatever the contract files say at any
moment is validated, committed, and in the automod ledger — nothing lands by a
bare file copy.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import bench_split
from .common import LLOYD_HOME, AutoresearchConfig, now_iso

logger = logging.getLogger("autoresearch.promote")


def _mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0

# Canonical targets that can be overwritten by promotion
CANONICAL_PROMPTS = {
    "SOUL.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "SOUL.md",
    "MEMORY.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "MEMORY.md",
    "USER.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "USER.md",
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _per_task(summary: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in summary.get("per_task", []) or []:
        tid = p.get("task_id")
        score = p.get("composite_score")
        if tid is not None and isinstance(score, (int, float)):
            out[str(tid)] = float(score)
    return out


def derive_split(*summaries: dict[str, Any]) -> dict[str, Any]:
    """A category split taken from the summaries themselves.

    `judge.aggregate_variant` already stamps `category` onto every per-task row,
    so the two fixed halves of the split are recoverable from any real summary.
    This is the fallback for a caller that has no written round split — it
    carries the categories, never the per-round rotation.
    """
    cats: dict[str, str] = {}
    for summ in summaries:
        for p in summ.get("per_task", []) or []:
            tid, cat = p.get("task_id"), p.get("category")
            if tid and cat:
                cats.setdefault(str(tid), str(cat))
    return {
        "round_id": None,
        "targeted": sorted(t for t, c in cats.items() if c in bench_split.TARGETED_CATEGORIES),
        "heldout": sorted(t for t, c in cats.items() if c in bench_split.HELDOUT_CATEGORIES),
        "rotated_into_heldout": [],
        "split_hash": None,
        "derived_from": "task_category",
    }


def slice_metrics(
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    """Per-slice numbers the gate decides on. One implementation, shared with
    `replay_promotion_gate.py` so a replay cannot drift from the predicate it is
    replaying — a re-derived arithmetic would be measuring the replay, not the gate.
    """
    base_per = _per_task(baseline_summary)
    var_per = _per_task(variant_summary)
    targeted = [str(t) for t in (split.get("targeted") or [])]
    heldout = [str(t) for t in (split.get("heldout") or [])]
    known = set(targeted) | set(heldout)
    unsplit = sorted((set(base_per) | set(var_per)) - known)

    def pair(ids: list[str]) -> tuple[list[str], float, float]:
        shared = [t for t in ids if t in base_per and t in var_per]
        return shared, _mean([base_per[t] for t in shared]), _mean([var_per[t] for t in shared])

    t_ids, t_base, t_var = pair(targeted)
    h_ids, h_base, h_var = pair(heldout)
    # ── The win leg's tie semantics, in one place, deliberately STRICT ─────────
    # A win is a per-task STRICT increase (`variant > baseline`); a tie is not a
    # win, a decrease is a loss. The leg exists to catch a variant that gains on a
    # few tasks while REGRESSING on others, and strictness is what makes a
    # regression a non-win — that is the whole purpose, so it is decided here
    # rather than left to whichever spelling of a comment is nearest.
    # Counting ties as wins would not weaken that purpose — a regression is still a
    # non-win — but it would empty the leg on this bench:
    # `bench_mine.calibrate_candidate` records four of the eleven live tasks sitting
    # at exactly 0.00, so a variant that moved ONE targeted task and tied every other
    # one would score `compared/compared` = 1.00 and clear the threshold outright.
    # And 0.5 is a measured operating point: `promotion_fp_rate.py` derived its
    # false-positive rate over the live promoted-round corpus under exactly these
    # strict semantics, and #549's strict held-out legs below refuse a tie for the
    # same reason.
    # The ledger shape this refuses, reproduced by a named test
    # (`test_the_strict_tie_rule_is_a_deliberate_decision_on_the_ledger_shape`): a
    # 2026-09-01 variant better on 4 of 11 tasks, tied on 7, WORSE ON NONE, whole-bench
    # mean +0.1227, refused at the 0.36 the ledger recorded for it. A ledger census
    # taken 2026-09-16 found all 552 `insufficient_win_fraction` refusals carrying a
    # positive mean delta, and 24 refused variants that strictly dominated their
    # baseline (better on some tasks, worse on none). Refusing those is the decision;
    # printing the split beside it, instead of only the fraction, is what stops the
    # next reader recomputing 28k ledger rows to discover that `losses=0`.
    wins = sum(1 for t in t_ids if var_per[t] > base_per[t])
    losses = sum(1 for t in t_ids if var_per[t] < base_per[t])
    ties = len(t_ids) - wins - losses
    return {
        "compared": len(t_ids),
        "wins": wins, "ties": ties, "losses": losses,
        "unsplit": unsplit,
        "targeted_ids": t_ids, "targeted_baseline": t_base, "targeted_variant": t_var,
        "targeted_delta": t_var - t_base,
        "heldout_ids": h_ids, "heldout_baseline": h_base, "heldout_variant": h_var,
        "heldout_delta": h_var - h_base,
        "win_fraction": (wins / len(t_ids)) if t_ids else 0.0,
        # HarnessOpt-Bench's normalized gain: the fraction of the seed's *remaining
        # headroom* the change captured. A raw +0.05 over a 0.50 baseline is 10% of
        # what was available; the same +0.05 over 0.85 is 33%. Printed beside the
        # delta so a decision says how much it bought, not only which way it moved.
        "headroom": 1.0 - t_base,
        "normalized_gain": ((t_var - t_base) / (1.0 - t_base)) if t_base < 1.0 else None,
    }


def evaluate_promotion(
    cfg: AutoresearchConfig,
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    split: dict[str, Any] | None = None,
    require_full_slice: bool = True,
) -> tuple[bool, str]:
    """Decide whether a variant should be promoted over baseline.

    Two conditions on two slices, per AutoDesign (arXiv:2608.13560): the targeted
    slice must improve AND the held-out slice must not decline. `safety_passed`
    and `contract_refusals()` stay as vetoes on top — neither is a substitute for
    the split, because `safety_critical` is set on exactly one of the 11 tasks and
    the contract guard reads the prompt, not the scores.

    The absolute `promotion_min_composite_delta` is gone. It compared a raw delta
    against a fixed 0.05 while the rounds' own baselines were seeded from
    `composite < 0.5` tasks, so the gate's strictness moved with the baseline mix:
    +0.05 off a 0.50 seed captured 10% of the available headroom, +0.05 off 0.85
    captured 33%. `targeted_delta > 0` asks only for a real gain on the pool the
    variant aimed at; the no-decline half is what stops a gain there from paying
    for a loss anywhere else.

    Returns (should_promote, reason).
    """
    if not variant_summary.get("safety_passed", False) and cfg.promotion_require_safety_pass:
        return False, "safety_regression"

    split = split or derive_split(baseline_summary, variant_summary)
    if not split.get("heldout"):
        # No veto slice means this is the pre-#549 gate wearing new clothes.
        return False, "no_heldout_slice"

    m = slice_metrics(baseline_summary, variant_summary, split)
    if m["unsplit"]:
        # A scored task outside both pools is a task no condition looks at —
        # usually a bench file added between the split write and the run.
        return False, f"unsplit_tasks ({len(m['unsplit'])}: {', '.join(m['unsplit'][:3])})"
    if not m["targeted_ids"]:
        return False, "no_targeted_overlap"
    if not m["heldout_ids"]:
        return False, "no_heldout_overlap"
    if require_full_slice:
        # Every task the split declared has to have been scored. A round truncated
        # by `--bench-limit` normally scores SOME of the veto slice — one task of
        # four — and then "the held-out mean did not decline" is a verdict on a
        # quarter of the veto, the same averaging defect the split exists to
        # remove, moved one layer down. Fail closed and name what went unread.
        for label, pool, scored in (("held-out", split["heldout"], m["heldout_ids"]),
                                    ("targeted", split["targeted"], m["targeted_ids"])):
            missing = sorted(set(pool) - set(scored))
            if missing:
                return False, (
                    f"partial_{'heldout' if label == 'held-out' else 'targeted'}_coverage "
                    f"({len(scored)} of {len(pool)} {label} tasks scored; "
                    f"unscored: {', '.join(missing)})"
                )

    if m["targeted_delta"] <= 0:
        return False, (
            f"targeted_no_gain (targeted {m['targeted_baseline']:.4f} → "
            f"{m['targeted_variant']:.4f}, {m['targeted_delta']:+.4f} on "
            f"{len(m['targeted_ids'])} tasks)"
        )
    # Strict: a tie is a refuse. The item's "held-out must not decline, strict —
    # a tie on held-out is a refuse" is the difference between an unchanged slice
    # and an unmeasured one at 5 tasks, and ties are what a rubric judge with a
    # coarse objective check actually returns.
    if m["heldout_delta"] < 0:
        return False, (
            f"heldout_decline (held-out {m['heldout_baseline']:.4f} → "
            f"{m['heldout_variant']:.4f}, {m['heldout_delta']:+.4f} on "
            f"{len(m['heldout_ids'])} veto tasks)"
        )
    if m["heldout_delta"] == 0:
        return False, (
            f"heldout_tie (held-out flat at {m['heldout_variant']:.4f} across "
            f"{len(m['heldout_ids'])} veto tasks — strict no-decline refuses a tie)"
        )
    if m["win_fraction"] < cfg.promotion_min_win_fraction:
        # The `insufficient_win_fraction (X.XX < Y.YY` prefix is load-bearing: the
        # ledger's 552-row census keys on it. Everything after it is the tie/loss
        # split, so a variant that beat baseline with ZERO regressions — the shape
        # the strict leg refuses by design — is recognisable from the round report or
        # the ledger row alone instead of only from a recomputation of the ledger.
        return False, (
            f"insufficient_win_fraction ({m['win_fraction']:.2f} < "
            f"{cfg.promotion_min_win_fraction}; wins={m['wins']} "
            f"ties={m['ties']} losses={m['losses']} of {m['compared']} targeted tasks)"
        )

    gain = m["normalized_gain"]
    gain_str = "n/a" if gain is None else f"{gain:.2%}"
    return True, (
        f"promote (targeted_delta={m['targeted_delta']:+.4f}, "
        f"heldout_delta={m['heldout_delta']:+.4f}, normalized_gain={gain_str}, "
        f"win_frac={m['win_fraction']:.2f})"
    )


def snapshot_current_prompts(cfg: AutoresearchConfig) -> Path:
    """Copy current SOUL/MEMORY/USER into a timestamped snapshot dir. Returns the dir."""
    snap_dir = cfg.paths.snapshots_dir / _ts()
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, src in CANONICAL_PROMPTS.items():
        if src.exists():
            shutil.copy2(src, snap_dir / name)
    saved = sorted(p.name for p in snap_dir.iterdir() if p.is_file())
    # A snapshot is the only rollback point a promotion has, and `mkdir` had
    # been the whole guarantee: a copy that raised still left a directory, so
    # `promote` went on to overwrite the live contract with nothing to restore.
    # 26 of the ledger's 83 promotions have no matching snapshot. Found while
    # writing `test_promote_refuses_when_the_snapshot_cannot_be_written` on
    # 2026-09-06 and carried as an xfail until now.
    if not any(name in CANONICAL_PROMPTS for name in saved):
        raise RuntimeError(
            f"snapshot {snap_dir} holds no prompt file — refusing to promote "
            f"with no rollback point (expected any of {sorted(CANONICAL_PROMPTS)})"
        )
    (snap_dir / "snapshot.json").write_text(
        json.dumps({"created_at": now_iso(), "files": saved}, indent=2),
        encoding="utf-8",
    )
    logger.info("snapshotted canonical prompts into %s", snap_dir)
    return snap_dir


def apply_overlay(overlay_dir: Path) -> list[str]:
    """Copy variant overlay files onto canonical prompts. Returns list of applied files."""
    applied: list[str] = []
    for name, dest in CANONICAL_PROMPTS.items():
        src = overlay_dir / name
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            applied.append(name)
    return applied


def _prospective(overlay_dir: Path, name: str) -> str | None:
    """What `name` would contain after this overlay is applied."""
    src = overlay_dir / name
    if src.exists():
        return src.read_text(encoding="utf-8")
    dest = CANONICAL_PROMPTS.get(name)
    if dest and dest.exists():
        return dest.read_text(encoding="utf-8")
    return None


#: The two surfaces a candidate's own shape can be measured on, in the order the
#: ceilings govern them. SOUL.md first: `check_contract` puts both ceilings on that
#: file, so when an overlay carries both files the candidate ratio that means
#: something to the ratchet is SOUL.md's, and a MEMORY.md number recorded beside it
#: would enter a MEMORY ratio into a SOUL series — the cross-entity fragmentation
#: failure in miniature, one measurement filed under the wrong system.
SHAPE_SURFACES: tuple[str, ...] = ("SOUL.md", "MEMORY.md")


def candidate_shape(overlay_dir: Path) -> dict[str, Any]:
    """The shape a candidate's *own* file has, and which file that was.

    `{surface, gate_share, prohibition_ratio}`, all-`None` when the overlay carries
    neither measured surface. Deliberately the overlay's text rather than the
    prospective merged contract: the live ratios answer "what is the contract now",
    these answer "what is this candidate proposing", and the ratchet needs the
    second one or a no-op overlay would inherit the live value as a plateau.

    `surface` is recorded so the history can be filtered by it. A MEMORY.md-only
    candidate is the case #789 names as the blind spot — `check_contract` puts both
    ceilings on SOUL.md, so a MEMORY-only overlay that doubles its own gate stack
    trips nothing. Recording it is the half this item can close; ceiling-ing
    MEMORY.md is a scope call a person has to make.
    """
    try:
        import prompt_surface
    except ImportError:  # pragma: no cover - repo is always importable
        return {"surface": None, "gate_share": None, "prohibition_ratio": None}
    for name in SHAPE_SURFACES:
        src = overlay_dir / name
        if src.exists():
            shape = prompt_surface.contract_shape(
                src.read_text(encoding="utf-8")
            )
            return {
                "surface": name,
                "gate_share": shape["gate_share"],
                "prohibition_ratio": shape["prohibition_ratio"],
            }
    return {"surface": None, "gate_share": None, "prohibition_ratio": None}


def contract_shape_fields(overlay_dir: Path | None) -> dict[str, Any]:
    """The #789 block one round records: the live contract, and this candidate.

    `contract_*` is the SOUL.md the loop is running against right now — the file whose
    two ratios the ceilings police. `candidate_*` is the overlay's own file, per
    `candidate_shape`. A missing measurement is `None`, never `0.0`: a round with no
    candidate and a contract with no gate stack are different facts, and a zero inside
    a series reads as a fall and masks a real climb.

    Here rather than in `post_promotion`, which writes the block to the ledger: reading
    prompt text needs this module's `CANONICAL_PROMPTS`, and #429's separation test
    pins the recorder's import list so it cannot reach this module.
    """
    fields: dict[str, Any] = {
        "contract_surface": None, "contract_gate_share": None,
        "contract_prohibition_ratio": None,
    }
    soul = CANONICAL_PROMPTS.get("SOUL.md")
    if soul is not None:
        try:
            import prompt_surface

            live = prompt_surface.contract_shape(
                soul.read_text(encoding="utf-8", errors="replace")
            )
            fields = {
                "contract_surface": "SOUL.md",
                "contract_gate_share": live["gate_share"],
                "contract_prohibition_ratio": live["prohibition_ratio"],
            }
        except (ImportError, OSError):  # pragma: no cover - always importable here
            pass
    cand = candidate_shape(overlay_dir) if overlay_dir is not None else {}
    fields.update({
        "candidate_surface": cand.get("surface"),
        "candidate_gate_share": cand.get("gate_share"),
        "candidate_prohibition_ratio": cand.get("prohibition_ratio"),
    })
    return fields


def shape_ratchet_refusals(
    shape: dict[str, Any],
    history: list[dict[str, Any]],
    run: int | None = None,
) -> list[str]:
    """Refuse a candidate that ratchets the contract's shape upward, #789.

    The absolute ceilings in `prompt_surface.check_contract` are per-candidate and
    cannot see this class: a climb that stays under 50% gate stack and under 25%
    prohibitions passes every ceiling forever, one promotion at a time, and each
    round's overlay only has to add a few hundred bytes to gain the ratchet. The
    live record is exactly that shape — gate share over the 65 promotion snapshots
    ran 39.0% (08-23) → 23.7% (09-02) → 39.4 → 52.9 → 63.0 → 63.6% (09-04/05), 7
    rises over 28 changed values. By the third consecutive rise the absolute check
    caught the 09-04 case, so this rule earns its keep only on climbs that stay
    *under* the ceiling, which is where the loop has headroom today (live SOUL.md
    measures 45.3% gate share and 19.3% prohibitions against ceilings of 50%/25%).

    Refuses when the candidate's own value is higher than the last `run - 1`
    recorded candidate values **in order** — three rises counting the candidate.
    The refusal fires only while **both** ratios sit under their ceilings: once one
    is past a ceiling, the absolute check has already refused this candidate and
    named its bytes and line counts, and a second refusal on a trend adds nothing a
    reader could act on.

    A metric with too few recorded values to form a run is skipped rather than
    guessed at: the first rounds after this ships have one row each, and a rule
    that fired on one data point would refuse every candidate for a reason nobody
    could check.
    """
    try:
        import prompt_surface
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"prompt_surface unavailable, refusing to promote blind: {exc}"]

    run = run or prompt_surface.CONTRACT_RISE_RUN
    metrics = (
        ("gate_share", "gate stack", prompt_surface.GATE_STACK_CEILING),
        ("prohibition_ratio", "prohibition lines", prompt_surface.PROHIBITION_RATIO_CEILING),
    )
    values = {key: shape.get(key) for key, _, _ in metrics}
    if any(values[key] is None for key, _, _ in metrics):
        return []
    if any(float(values[key]) > ceiling for key, _, ceiling in metrics):
        return []

    # Oldest first by the round's own timestamp, with file position as the tiebreak,
    # so a ledger whose rows were replayed out of order — a restore, a backfill — still
    # describes the series in the order it happened. A row whose `created_at` is missing
    # or unparseable has no comparable timestamp and is dropped rather than defaulted:
    # defaulting to "oldest" or "newest" would let an untrusted field put a value inside
    # the window and forge a climb, or move one out and hide one. Dropping costs the run
    # a value, and a short run refuses nothing — the safe direction. `post_promotion`
    # stamps `created_at` on every row it writes, so in production this set is empty.
    stamped: list[tuple[str, int, dict]] = []
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            continue
        stamp = row.get("created_at")
        if isinstance(stamp, str) and len(stamp) >= 10 and stamp[4:5] == "-":
            stamped.append((stamp, index, row))
    ordered = [row for _, _, row in sorted(stamped, key=lambda item: (item[0], item[1]))]

    errors: list[str] = []
    for key, label, ceiling in metrics:
        # Recorded under the `candidate_` prefix, which is `SHAPE_FIELDS` and therefore
        # also the row's own column name; a bare key is accepted too so a hand-built
        # history in a test is one dict rather than a simulated ledger row. The window is
        # taken over *usable* values, not over rows: a row that is None for this metric
        # must not consume a window slot, or a series recorded 0.10, None, 0.11, 0.12
        # against a candidate of 0.13 — three real rises — would be read as its last two
        # rows, 0.12 then 0.13, and refuse nothing.
        #
        # The whole ordered series, not the last `run - 1` values, and no cheaper
        # window will do: the run is walked back from the end through *changed* values,
        # so a plateau inside the look-back is part of the climb rather than a break in
        # it. A series recorded g-0.03, g-0.02, g-0.02 against a candidate of g is three
        # rises, which is what the 2026-09-04→05 snapshot run actually looked like
        # (52.9 % → 63.0 % → 63.6 % → 63.6 %); taking two rows positionally would see
        # one rise and refuse nothing. Extra history cannot manufacture a run either —
        # `rising_run` stops at the first non-rise, so a fall anywhere in the window
        # ends the walk wherever it stands.
        field = f"candidate_{key}"
        prior = [
            float(r.get(field, r.get(key))) for r in ordered
            if r.get(field, r.get(key)) is not None
        ]
        series = prompt_surface.rising_run([*prior, float(values[key])], run)
        if len(series) < run:
            continue
        errors.append(
            f"{label} has risen across {len(series)} recorded shapes without crossing "
            f"the {ceiling:.0%} ceiling — "
            f"{' → '.join(f'{v:.1%}' for v in series)} — which no per-candidate "
            f"ceiling can see (backlog #789). Refused as a ratchet; the series is in "
            f"the ledger `round_summary` rows."
        )
    return errors


def contract_refusals(overlay_dir: Path, shape_history: list[dict[str, Any]] | None = None) -> list[str]:
    """Why this variant must not be written over the live contract. [] = fine.

    This path is what produced #464 and #465. It writes Lloyd's identity files
    in the live vault on an hourly cadence with no gate, no test, no review and
    no revert, and on 2026-09-08 it promoted a variant whose own hypothesis was
    "add negative constraints to fix the shape benchmarks" — 64% gate stack,
    28% prohibition lines — over a contract that had been trimmed to 48%/19%
    ninety minutes earlier. A search over prompts is legitimate; landing the
    result unchecked is not, and the search cannot be trusted to police itself
    because bench score is exactly what it is optimising.

    Checked against the *prospective* text — the overlay's file where it has
    one, the live file where it does not — so a variant that only rewrites
    MEMORY.md is still judged against the SOUL.md it will sit beside.

    `shape_history` is the recorded candidate shapes from earlier rounds, oldest
    first (see `post_promotion.contract_shape_history`). It defaults to `None`,
    which means "no series to consult" and skips only the cross-round ratchet — the
    absolute ceilings always run. The two callers that pass it (`promote()` reading
    its own ledger, and the tests) are the only place the ratchet can be evaluated
    at all, because it is the one check here that needs a fact about *other*
    rounds; every other check in this function is answerable from the overlay.
    """
    try:
        import prompt_surface
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"prompt_surface unavailable, refusing to promote blind: {exc}"]
    soul = _prospective(overlay_dir, "SOUL.md")
    if soul is None:
        return ["no SOUL.md to check, in the overlay or on disk"]
    errors = prompt_surface.check_contract(soul, _prospective(overlay_dir, "MEMORY.md"))
    # The ratchet compares against a series of SOUL.md candidates (`surface="SOUL.md"`
    # at the call site), so it may only be applied to an overlay that carries one. A
    # MEMORY.md-only overlay has no SOUL.md of its own, so `soul` here is the live
    # contract unchanged: the comparison would ask "is the live file higher than the last
    # three candidates were?" — and since a refused promotion never changes the live
    # file, that answer stays yes every hour thereafter, locking the loop into refusing
    # on a series that is not the candidate's (the review of SM_20260919_074839 called
    # exactly this: live 45 % against candidates 20 %/22 %, "a series that is not the
    # candidate's and that repeats every round"). Clause 5 is the answer for that
    # candidate — its own ratios are recorded — and the absolute ceilings above still
    # judge the text that would actually be landed, since `apply_overlay` lands SOUL.md.
    if shape_history is not None and (overlay_dir / "SOUL.md").is_file():
        errors += shape_ratchet_refusals(prompt_surface.contract_shape(soul), shape_history)
    return errors


def write_experiment_fact(
    cfg: AutoresearchConfig,
    variant: dict[str, Any],
    variant_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    snapshot_dir: Path,
) -> Path | None:
    """Write the promoted experiment as a fact file under cfg.paths.facts_experiments_dir/<id>/."""
    ex_dir = cfg.paths.facts_experiments_dir / variant["variant_id"]
    ex_dir.mkdir(parents=True, exist_ok=True)
    fact_file = ex_dir / f"{variant['variant_id']}-experiment.md"

    baseline_mean = baseline_summary.get("mean_composite", 0.0)
    variant_mean = variant_summary.get("mean_composite", 0.0)
    delta = variant_mean - baseline_mean

    frontmatter = {
        "type": "facts",
        "entity": variant["variant_id"],
        "category": "experiment",
        "last_updated": now_iso(),
        "facts": [
            {
                "fact": f"Autoresearch variant {variant['variant_id']} promoted ({variant.get('description', '')}). "
                        f"mean_composite {baseline_mean:.3f} → {variant_mean:.3f} (Δ {delta:+.3f}) over "
                        f"{variant_summary.get('task_count', 0)} bench tasks.",
                "confidence": 0.95,
                "category": "experiment",
                "id": f"exp-{variant['variant_id']}",
                "created_at": now_iso(),
                "valid_at": now_iso(),
                "invalid_at": None,
                "expired_at": None,
                "provenance": "EXTRACTED",
                "source_doc": str(snapshot_dir),
            }
        ],
    }
    import yaml

    body = (
        f"\n# {variant['variant_id']} - experiment\n\n"
        f"**Target surface:** {variant.get('target_surface', 'prompts')}\n"
        f"**Hypothesis:** {variant.get('hypothesis', '')}\n"
        f"**Snapshot:** `{snapshot_dir}`\n"
        f"**Baseline mean composite:** {baseline_mean:.3f}\n"
        f"**Variant mean composite:** {variant_mean:.3f}\n"
        f"**Delta:** {delta:+.3f}\n"
        f"**Task count:** {variant_summary.get('task_count', 0)}\n"
    )
    fact_file.write_text(f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n{body}", encoding="utf-8")
    logger.info("wrote experiment fact to %s", fact_file)
    return fact_file


def _vault_relative_paths(names: list[str]) -> tuple[list[str], list[str], Path]:
    """Resolve prompt names to (the ones inside the vault, their vault-relative
    paths, the vault root) — one derivation shared by `promote()` and `rollback()`.

    Both write `CANONICAL_PROMPTS`, and both have to answer the same question
    before they reach for a commit: *are these files actually in the vault?* If
    any one target resolves outside it the answer is "no" for the whole set, so
    nothing is committed rather than half a contract.

    The `vault_round` import is deliberately **not** wrapped: a caller that
    cannot reach the landing route must see that as a failure, not be handed the
    `([], …)` answer that means "outside the vault, commit nothing".
    """
    from scripts.automod import vault_round as VR

    vault_root = VR.VAULT.resolve()
    kept: list[str] = []
    rel: list[str] = []
    for name in names:
        try:
            rel.append(str(CANONICAL_PROMPTS[name].resolve().relative_to(vault_root)))
        except ValueError:
            return [], [], vault_root
        kept.append(name)
    return kept, rel, vault_root


def promote(
    cfg: AutoresearchConfig,
    variant: dict[str, Any],
    variant_overlay_dir: Path,
    variant_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full promotion pipeline. Returns a result dict."""
    result: dict[str, Any] = {
        "variant_id": variant["variant_id"],
        "dry_run": dry_run,
        "snapshot_dir": None,
        "applied_files": [],
        "experiment_fact": None,
    }
    # The ratchet is the only check here that needs a fact about *other* rounds, so
    # `promote()` is the one caller that can supply it. Read from the same ledger
    # the rounds write to — `record_round_summary` appends the shape row whether or
    # not anything promoted, so the series exists by the second round of a loop that
    # promotes nothing at all. The import is function-local because `post_promotion`
    # reads `CANONICAL_PROMPTS` from this module at import time; the cycle only ever
    # closes at call time, when both modules are already loaded.
    from .post_promotion import contract_shape_history

    # `cfg is None` is not a test convenience to accommodate: it is the shape of a
    # caller that has a candidate and no config, and such a caller must still get the
    # absolute ceilings rather than an unguarded write. What it cannot get is the
    # ratchet, whose only input is a ledger no config was named to point at.
    history = contract_shape_history(cfg.paths.ledger_path) if cfg else None
    refusals = contract_refusals(variant_overlay_dir, shape_history=history)
    if refusals:
        # Not an exception: a refused promotion is a normal outcome of a search
        # that proposed something out of bounds, and the round must carry on
        # evaluating the rest. It is logged at ERROR because a variant that wins
        # on bench score and still cannot be written is the signal that the
        # score and the constraint disagree — which is the whole finding.
        logger.error("REFUSED promotion of %s — %s",
                     variant["variant_id"], "; ".join(refusals))
        result["refused"] = refusals
        return result

    if dry_run:
        logger.info("[dry-run] would promote %s", variant["variant_id"])
        return result

    try:
        snap = snapshot_current_prompts(cfg)
    except (OSError, RuntimeError) as exc:
        logger.error("REFUSED promotion of %s — no rollback point: %s",
                     variant["variant_id"], exc)
        result["refused"] = [f"snapshot failed: {exc}"]
        return result
    applied = apply_overlay(variant_overlay_dir)
    result["snapshot_dir"] = str(snap)
    result["applied_files"] = applied

    # Commit through the vault route, which runs the real loaders and reverts
    # the paths if any of them fails. Before this, a promotion was an
    # uncommitted overwrite of a live tracked file: nothing recorded that it
    # had happened, `automod_vault_revert` had no sha to undo, and the only
    # trace was a snapshot directory nobody was told about. The sha goes in the
    # automod ledger as a `vault_land` event like every other vault change.
    try:
        from scripts.automod import vault_round as VR

        # Derive the vault-relative paths rather than assuming `lloyd/<name>`,
        # and commit nothing when the canonical prompts are not in the vault at
        # all. `CANONICAL_PROMPTS` is redirected by tests and by the variant
        # sandbox, and a hardcoded prefix would have made this function run
        # `git add` against the real `~/obsidian` from inside a unit test.
        _applied_in_vault, rel, vault_root = _vault_relative_paths(applied)
        if rel:
            landed = VR.land(
                rel,
                f"autoresearch: promote {variant['variant_id']}\n\n"
                f"{variant.get('description', '')}\n\n"
                f"Snapshot of the previous contract: {snap}",
            )
            result["vault_commit"] = landed.get("commit")
        else:
            # Not an error: the overlay sandbox and the tests both point the
            # canonical prompts somewhere that is not a git repo, and there is
            # nothing to commit there. The fact below is still written.
            logger.info("canonical prompts are outside %s — applied without a "
                        "vault commit", vault_root)
    except Exception as exc:  # VaultRoundError, or git refusing for any reason
        # `land` reverts the paths it validated before raising, so the live
        # contract is already back. Say so loudly and leave the snapshot.
        logger.error("promotion of %s did not land, contract restored: %s",
                     variant["variant_id"], exc)
        result["refused"] = [str(exc)]
        result["applied_files"] = []
        return result

    fact = write_experiment_fact(cfg, variant, variant_summary, baseline_summary, snap)
    result["experiment_fact"] = str(fact) if fact else None
    logger.info("promoted %s: applied=%s vault_commit=%s snapshot=%s",
                variant["variant_id"], applied, result.get("vault_commit"), snap)
    return result


def rollback(cfg: AutoresearchConfig, snapshot_ts: str) -> dict[str, Any]:
    """Restore canonical prompts from the named snapshot, through the vault route.

    This is the one undo path for a bad prompt promotion, and it used to be three
    `shutil.copy2` calls. Its targets are `CANONICAL_PROMPTS` — tracked files in
    the live vault — so a raw copy left the contract modified in the working tree
    while HEAD still pointed at the promotion commit, and nothing anywhere
    recorded that a restore had happened. `scripts/util/vault-commit.sh:53` runs
    `git add -A` over `~/obsidian` from ten call sites in seven nightly skills, so
    the next job to commit a dirty vault landed the restore under *its* message:
    the blame-masking that let the 2026-09-10 MEMORY.md truncation sit undiscovered
    for 19 hours (see the clobber note in lloyd/MEMORY.md).

    The copy stays — a validated commit is not a revert, and the bytes have to be
    in the tree before anything can validate them — but
    `scripts.automod.vault_round.land` finishes the job the way every other vault
    write is finished: it runs the prompt-surface validators and the real loaders,
    puts the paths back if any of them fails, commits exactly the restored files
    on the vault's `main` with the snapshot ts in the message, and appends a
    `vault_land` ledger event carrying the sha. The sha is returned.
    """
    snap = cfg.paths.snapshots_dir / snapshot_ts
    result: dict[str, Any] = {"snapshot": str(snap), "restored_files": [],
                              "vault_commit": None, "no_change": False}
    if not snap.exists():
        result["error"] = f"snapshot {snapshot_ts} not found"
        return result

    present = [name for name in CANONICAL_PROMPTS if (snap / name).exists()]
    try:
        _present_in_vault, rel, vault_root = _vault_relative_paths(present)
    except Exception as exc:  # noqa: BLE001 — no landing route reachable: refuse, do not raw-copy
        result["refused"] = [f"cannot reach the vault landing route: {exc}"]
        logger.error("rollback to %s refused before touching anything: %s", snapshot_ts, exc)
        return result

    for name in present:
        shutil.copy2(snap / name, CANONICAL_PROMPTS[name])
        result["restored_files"].append(name)

    if not present:
        # A snapshot directory that holds no prompt file restores nothing and has
        # nothing to commit. `land` would refuse an empty path list; that refusal
        # is about a malformed call, not about this, so say the true thing here.
        logger.warning("snapshot %s holds none of %s — nothing restored",
                       snap, sorted(CANONICAL_PROMPTS))
        return result

    if not rel:
        # Not an error: the variant sandbox and the unit tests point the canonical
        # prompts at a directory that is not the vault, where a raw copy is all
        # that can be done — and no nightly sweep commits that tree, so there is
        # nothing to mask. `promote()` has the same branch.
        logger.info("rolled back %s files from %s with no vault commit — those "
                    "paths are outside %s", len(result["restored_files"]), snap, vault_root)
        return result

    try:
        from scripts.automod import vault_round as VR

        landed = VR.land(
            rel,
            f"autoresearch: rollback to snapshot {snapshot_ts}\n\n"
            f"Restored {', '.join(result['restored_files']) or 'no files'} from {snap}. "
            "The vault route validated the restored contract before committing it, so a "
            "bad promotion is undone by one revert.",
        )
    except Exception as exc:  # VaultRoundError, or git refusing for any reason
        if "nothing to commit" in str(exc):
            # Content that already matches HEAD: a repeated rollback, or a
            # snapshot that *is* the live contract. A no-op is not a failed undo,
            # and letting this raise would answer the tool's caller with an error
            # for having asked the same question twice.
            result["no_change"] = True
            logger.info("rollback to %s restored content that already matches "
                        "HEAD — nothing to commit", snapshot_ts)
            return result
        # `land` reverted the paths it validated before raising, so the tree is
        # back at HEAD and the live contract is the pre-rollback one. Report it as
        # a refusal rather than a restore that did not happen: an old snapshot
        # predating a structural change must be refused, not silently applied.
        logger.error("rollback to %s refused; contract restored: %s", snapshot_ts, exc)
        result["refused"] = [str(exc)]
        result["restored_files"] = []
        return result

    result["vault_commit"] = landed.get("commit")
    logger.info("rolled back %s files to %s: vault_commit=%s",
                len(result["restored_files"]), snapshot_ts, result["vault_commit"])
    return result
