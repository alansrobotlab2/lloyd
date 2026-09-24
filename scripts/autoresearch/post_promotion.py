"""Post-promotion comparison — this module records and surfaces, never restores.

Acting on what is recorded here is a separate module: see
:mod:`scripts.autoresearch.auto_restore` (#1099) for the automated restore.

The defect this was filed on
----------------------------
`promote()` decides, writes and logs, and then nothing ever looks again.
`rollback()` (:mod:`scripts.autoresearch.promote`) had one entry point, the manual
`autoresearch_rollback` MCP handler, when this was written. Measured on
2026-09-10 over the live ledger (`_pipeline/research/ledger.jsonl`, 30,953 rows
then), the string `rollback`/`revert` appeared **0** times while
`"promoted": true` appeared 65 — no promotion had ever been detected as a
false positive, because nothing re-measured anything after a promotion landed.
`run_round.py`'s only reference to prior rounds was
:func:`~scripts.autoresearch.common.find_last_promoted_variant`, used for parent
*lineage*, never for a score. So a variant promoted on a lucky bench draw —
exactly the thing backlog #428 measured the rate of — stays in the contract with
no signal.

What this module does
---------------------
Option (b) from the item: **record** and **surface**.

1. *Record* — one machine-readable `event: "round_summary"` ledger row per
   round, carrying `baseline_mean`, the `promoted_variant_id` that landed and
   that variant's `promoted_variant_mean` as recorded in its own round, plus the
   `snapshot_dir` the promotion left behind. This is the row whose absence was
   the item's acceptance check.
2. *Surface* — the next round's own fresh baseline is compared against the
   previously promoted variant's recorded mean, and when the fall exceeds the
   configured noise floor the round report **names the decline**, attributes it
   to the promoting round and variant, and prints the snapshot directory the
   contract could be restored from.

What it deliberately does not do
--------------------------------
It never touches a prompt file. Acting on a decline it detects is real work and it
lives in :mod:`scripts.autoresearch.auto_restore` (backlog #1099, Alan's sign-off on
the out-of-scope clause this module originally reserved for a human); this module
still only records and surfaces, and #429's separation test — which walks this
module's import list looking for the promotion and file-copy machinery — is the
reason the restore cannot simply move in here. What this module owns of #1099 is the
*reading* side of the same record: :func:`restore_rows` and
:func:`restored_promotion_rounds`, so a promotion somebody already undied is neither
restored again nor measured as the live contract. The statistical backstop remains
the false-positive sweep in :mod:`scripts.autoresearch.promotion_fp_rate` (backlog
#428), which measures the *rate*; this measures *this round*.

The floor
---------
`noise_floor` is `autoresearch.promotion.noise_floor`, falling back to
:data:`DEFAULT_NOISE_FLOOR` — backlog #324's cross-round baseline-mean standard
deviation, 0.1389 (`knowledge/evaluation/autoresearch-baseline-stability.md`,
84 rounds). Anything smaller than that is ordinary round-to-round movement of a
freshly generated bench, not a regression the promotion caused. `config.yaml` is
on the self-modification loop's never-touch list, so until a human adds the key
the default is the live value.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_NOISE_FLOOR,  # noqa: F401 — re-exported: the config layer owns the
    #                      # number, this module is its only consumer.
    AutoresearchConfig,
    ledger_append,
    now_iso,
)
from .promotion_fp_rate import BASELINE_RE

#: Ledger event name for the per-round machine-readable comparison row.
ROUND_SUMMARY_EVENT = "round_summary"

#: Ledger event name for a restore decision, written by
#: :mod:`scripts.autoresearch.auto_restore` (#1099). The schema constant lives here
#: because this module is the reader — `restored_promotion_rounds` below is what keeps
#: a undone promotion from being compared as the live contract again — and a schema
#: constant must not be imported from the module that writes it, or the reader cannot
#: be tested without the restore machinery.
RESTORE_EVENT = "promotion_restored"

#: Which `promotion_restored` statuses mean the promotion is undone. `no_change`
#: counts: content that already matches the snapshot is the same end state, reached
#: without a second commit to attribute, and treating it as "not restored" would have
#: a later round look for a vault sha that does not exist.
RESTORED_STATUSES = frozenset({"restored", "no_change"})

#: `- BASELINE_x` and `- V_...` lines in the "Variant summaries" block.
_VARIANT_MEAN_RE = re.compile(
    r"^-\s*`(?P<vid>[^`]+)`(?:\s*\(baseline\))?:\s*mean=(?P<mean>[0-9.]+)", re.M
)
#: The landing record: `- variant: `V_...`` under "## Promoted".
_PROMOTED_VID_RE = re.compile(r"^-\s*variant:\s*`(?P<vid>[^`]+)`", re.M)
#: `- snapshot_dir: `/abs/path`` — promote.py resolves it into `result["snapshot_dir"]`.
_SNAPSHOT_DIR_RE = re.compile(r"^-\s*snapshot_dir:\s*`?(?P<dir>[^`\n]+?)`?\s*$", re.M)


def _rows(ledger_path: Path):
    if not ledger_path.exists():
        return
    for line in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def round_summary_rows(ledger_path: Path) -> list[dict[str, Any]]:
    """Every `event == "round_summary"` row, oldest first."""
    return [r for r in _rows(ledger_path) if r.get("event") == ROUND_SUMMARY_EVENT]


def restore_rows(ledger_path: Path) -> list[dict[str, Any]]:
    """Every `event == "promotion_restored"` row, oldest first (#1099)."""
    return [r for r in _rows(ledger_path)
            if isinstance(r, dict) and r.get("event") == RESTORE_EVENT]


def restored_promotion_rounds(ledger_path: Path) -> set[str]:
    """Round ids whose promotion a later round already undid.

    The reason this is subtracted out of :func:`last_promotion` is not bookkeeping,
    it is that the comparison has to describe the contract that is actually live:
    once a round restores a promotion's snapshot, the files on disk are the ones that
    promotion's predecessor produced, so measuring a fresh baseline against the
    *restored* promotion's mean would call the restored contract a regression and
    send the loop looking for a second promotion to undo.
    """
    return {str(r.get("restored_round_id")) for r in restore_rows(ledger_path)
            if r.get("status") in RESTORED_STATUSES and r.get("restored_round_id")}


#: How many recorded shape pairs the round report prints. One more promotion than the
#: ratchet needs to fire (`prompt_surface.CONTRACT_RISE_RUN`), so a person reading a
#: round sees the trend a refusal just claimed.
SHAPE_SERIES_LEN = 5

#: The #789 shape fields a `round_summary` row carries, in report order. Computed by
#: `promote.contract_shape_fields` and passed in: this module measures no prompt text
#: itself, so it stays free of the promotion machinery it must not call (#429's
#: separation test walks its import list for exactly that).
SHAPE_FIELDS: tuple[str, ...] = (
    "contract_surface", "contract_gate_share", "contract_prohibition_ratio",
    "candidate_surface", "candidate_gate_share", "candidate_prohibition_ratio",
)


def contract_shape_history(
    ledger_path: Path, surface: str = "SOUL.md", limit: int | None = None,
) -> list[dict[str, Any]]:
    """Recorded candidate shapes, oldest first, filtered to one surface (#789).

    This is the series `promote.shape_ratchet_refusals` compares a candidate against.
    Filtering by surface is what keeps a MEMORY.md candidate's ratios out of the
    SOUL.md series: the two numbers are not comparable, and pooling them would let a
    round refuse a SOUL candidate because some MEMORY-only variant measured long.
    Rounds with no measured candidate contribute nothing rather than a zero, which
    would read as a fall and mask a real run.

    `limit` is the caller's cap on how far back to look; the default is the whole
    recorded series, because the run check has to see every value that could belong
    to it. Clamping to `CONTRACT_RISE_RUN - 1` here instead would cut a plateau out
    of the middle of a climb — a series recorded 0.09 → 0.10 → 0.10 followed by a
    candidate at 0.11 is three rises over changed values, but its last two rows alone
    are one rise, and the refusal would go missing on the round that earned it. See
    `prompt_surface.rising_run`.
    """
    out = [
        row for row in round_summary_rows(ledger_path)
        if row.get("candidate_surface") == surface
        and row.get("candidate_gate_share") is not None
    ]
    return out[-limit:] if limit and limit > 0 else out


def contract_shape_series(
    ledger_path: Path, n: int = SHAPE_SERIES_LEN,
) -> list[dict[str, Any]]:
    """The last `n` rounds that recorded a contract shape, oldest first, for the report.

    Unlike `contract_shape_history` this is not filtered by surface: the report shows
    a human what the loop measured, including rounds whose candidate aimed at
    MEMORY.md — the ones no ceiling currently covers.
    """
    out = [
        row for row in round_summary_rows(ledger_path)
        if row.get("contract_gate_share") is not None
    ]
    return out[-n:] if n > 0 else out


def shape_report_lines(
    series: list[dict[str, Any]],
    gate_ceiling: float,
    prohibition_ceiling: float,
    rise_run: int,
    series_len: int = SHAPE_SERIES_LEN,
) -> list[str]:
    """The contract-shape drift block for a round report (#789).

    The three numbers in the header are arguments rather than imports on purpose: the
    ceilings live in `prompt_surface`, and #429's separation test pins this module's
    import list to the ones a recorder is allowed to have. The caller already resolves
    both modules and passes the figures through.
    """
    lines = [
        f"## Contract shape drift (last {series_len} recorded rounds)",
        f"- ceilings: gate stack {gate_ceiling:.0%}, prohibitions "
        f"{prohibition_ceiling:.0%}; a candidate whose ratio rises across "
        f"{rise_run} recorded shapes is refused even with both still under their ceiling",
    ]
    if not series:
        # Printed rather than omitted: a reader has to be able to tell "the record is
        # new" from "the record is missing", and a section that vanishes cannot say
        # which of the two it is.
        lines.append(
            "- no shape recorded yet — this is the first round that measures it"
        )
        return lines
    for row in series:
        cand = "no candidate measured"
        if row.get("candidate_gate_share") is not None:
            cand = (
                f"{row.get('candidate_surface')} "
                f"{float(row['candidate_gate_share']):.1%}/"
                f"{float(row['candidate_prohibition_ratio']):.1%}"
            )
        lines.append(
            f"- {row.get('round_id')}: contract "
            f"{float(row['contract_gate_share']):.1%}/"
            f"{float(row['contract_prohibition_ratio']):.1%} · candidate {cand}"
        )
    return lines


def promoted_variant_mean(ledger_path: Path, variant_id: str) -> float | None:
    """The mean recorded for `variant_id` in the round that promoted it.

    Read off the `round_summary` row rather than re-measured: the comparison has
    to be against the number the promotion itself was justified by, or a bench
    that has drifted since reads as a regression.
    """
    for row in reversed(round_summary_rows(ledger_path)):
        if row.get("promoted_variant_id") == variant_id:
            mean = row.get("promoted_variant_mean")
            if mean is not None:
                return float(mean)
    return None


def promotion_record_from_report(path: Path) -> dict[str, Any] | None:
    """The landing record inside one `rounds/R_<id>.md`, or None if it promoted nothing.

    Exists so the check is not stranded behind its own introduction date: the 65
    promotions that landed before this module are readable from their reports,
    which is also what lets a test replay a real round instead of running one.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    promoted = _PROMOTED_VID_RE.search(text)
    if not promoted:
        return None
    vid = promoted.group("vid")
    baseline = BASELINE_RE.search(text)
    snapshot = _SNAPSHOT_DIR_RE.search(text)
    snap_dir = snapshot.group("dir").strip() if snapshot else ""
    if snap_dir in {"None", "null"}:
        snap_dir = ""
    means = {m.group("vid"): float(m.group("mean")) for m in _VARIANT_MEAN_RE.finditer(text)}
    return {
        "source": "report",
        "round_id": path.stem,
        "baseline_mean": float(baseline.group(1)) if baseline else None,
        "promoted_variant_id": vid,
        "promoted_variant_mean": means.get(vid),
        "snapshot_dir": snap_dir,
        # A report never carried the sha it landed with — the round report names the
        # snapshot, and the sha lives in the landing ledger — so a promotion read from
        # its prose has no reference for "unchanged since the promotion". Explicit
        # None, so a reader never mistakes an absent key for an unmeasured one.
        "vault_commit": None,
    }


def promotion_records(ledger_path: Path, rounds_dir: Path) -> list[dict[str, Any]]:
    """All known promotions, oldest first, deduped per round id.

    A `round_summary` row wins over a report of the same round — the row is the
    machine-readable record; the report is prose for humans.
    """
    by_round: dict[str, dict[str, Any]] = {}
    if rounds_dir.exists():
        for path in sorted(rounds_dir.glob("R_*.md")):
            record = promotion_record_from_report(path)
            if record:
                by_round[record["round_id"]] = record
    for row in round_summary_rows(ledger_path):
        rid = row.get("round_id")
        vid = row.get("promoted_variant_id")
        if not rid or not vid:
            continue
        by_round[rid] = {
            "source": "ledger",
            "round_id": rid,
            "baseline_mean": row.get("baseline_mean"),
            "promoted_variant_id": vid,
            "promoted_variant_mean": row.get("promoted_variant_mean"),
            "snapshot_dir": row.get("snapshot_dir") or "",
            # The sha the promotion was committed with (#1099). Present on every
            # ledger-sourced record, None on every row written before that key
            # existed and on every record recovered from a round report — a
            # missing sha downstream is a refusal, never a guess.
            "vault_commit": row.get("vault_commit"),
        }
    return [by_round[rid] for rid in sorted(by_round)]


def last_promotion(
    ledger_path: Path,
    rounds_dir: Path,
    *,
    exclude_round: str | None = None,
) -> dict[str, Any] | None:
    """The most recent promotion on record, excluding the round asking.

    Also excludes any promotion a later round has already restored (#1099 clause 4),
    which is what stops a restored promotion from being both restored twice and
    measured as the live contract: after a restore the files on disk are the ones its
    predecessor produced, so the next round compares against that predecessor — the
    promotion whose mean actually describes the contract now in force.

    Round ids are `R_%Y%m%d_%H%M%S`, so lexicographic order is chronological.
    """
    restored = restored_promotion_rounds(ledger_path)
    records = [
        r for r in promotion_records(ledger_path, rounds_dir)
        if r["round_id"] != exclude_round and r["round_id"] not in restored
    ]
    return records[-1] if records else None


def compare(
    baseline_mean: float,
    prior: dict[str, Any] | None,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
) -> dict[str, Any] | None:
    """How far this round's fresh baseline sits below the last promotion's mean.

    Positive `decline` means performance later fell under the level the promoted
    variant recorded in its own round. `regression` is true only past the floor.
    """
    if not prior or prior.get("promoted_variant_mean") is None:
        return None
    recorded = float(prior["promoted_variant_mean"])
    decline = recorded - float(baseline_mean)
    snapshot_dir = prior.get("snapshot_dir") or ""
    return {
        "prior_round_id": prior.get("round_id"),
        "prior_source": prior.get("source"),
        "promoted_variant_id": prior.get("promoted_variant_id"),
        "promoted_variant_mean": round(recorded, 4),
        "baseline_mean": round(float(baseline_mean), 4),
        "decline": round(decline, 4),
        "noise_floor": round(float(noise_floor), 4),
        "regression": decline > float(noise_floor),
        "snapshot_dir": snapshot_dir,
        "snapshot_ts": Path(snapshot_dir).name if snapshot_dir else "",
    }


def report_section(comparison: dict[str, Any] | None,
                   restore: dict[str, Any] | None = None) -> list[str]:
    """The `## Post-promotion check` lines for a round report.

    A decline past the floor is named, attributed to the promoted variant and the
    round that landed it, and carries the snapshot directory it can be restored from.

    `restore` is the outcome of
    :func:`scripts.autoresearch.auto_restore.restore_for_decline` for this round —
    passed in as already-rendered lines by the caller, because #429's separation test
    keeps the recorder from importing the restore machinery. Three states are
    deliberately distinguishable: `restore is None` means no restore was attempted
    (the pre-#1099 report shape, still what a direct caller gets), an empty
    `restore["report_lines"]` means the loop ran and found nothing to undo, and a
    non-empty one names the vault sha it restored or the reason it refused. A reader
    must never be able to confuse "the loop undid it" with "the loop declined" with
    "no loop exists".
    """
    lines = ["", "## Post-promotion check"]
    if comparison is None:
        lines.append(
            "- no promotion on record to compare against — nothing to check "
            "(ledger `round_summary` rows plus `rounds/R_*.md` are both searched)."
        )
        return lines
    if comparison["regression"]:
        snap = comparison["snapshot_dir"]
        restore_src = (
            f"rollback point `{comparison['snapshot_ts']}` (`{snap}`), i.e. "
            f"`autoresearch_rollback(snapshot_ts=\"{comparison['snapshot_ts']}\")`"
            if snap
            else "**no snapshot directory on record — no rollback point exists for this promotion**"
        )
        lines.append(
            f"- **BASELINE DECLINE PAST NOISE FLOOR**: baseline mean "
            f"{comparison['baseline_mean']:.4f} is {comparison['decline']:.4f} below the "
            f"{comparison['promoted_variant_mean']:.4f} recorded for promoted variant "
            f"`{comparison['promoted_variant_id']}` (round `{comparison['prior_round_id']}`), "
            f"which exceeds the noise floor {comparison['noise_floor']:.4f}. "
            f"Restore source: {restore_src}."
        )
        if restore is None:
            # No restore outcome was supplied: the report says so rather than
            # implying the loop acted, and names the statistical backstop.
            lines.append(
                "- Nothing was restored by this section: no restore outcome was supplied, "
                "and the FP sweep in backlog #428 (`scripts/autoresearch/promotion_fp_rate.py`) "
                "is the statistical backstop for promotions like this one."
            )
        else:
            lines.extend(restore.get("report_lines") or [])
    else:
        lines.append(
            f"- baseline mean {comparison['baseline_mean']:.4f} vs "
            f"{comparison['promoted_variant_mean']:.4f} recorded for promoted variant "
            f"`{comparison['promoted_variant_id']}` (round `{comparison['prior_round_id']}`): "
            f"decline {comparison['decline']:+.4f} is within the noise floor "
            f"{comparison['noise_floor']:.4f}."
        )
        if restore is not None:
            lines.extend(restore.get("report_lines") or [])
    return lines


def record_round_summary(
    cfg: AutoresearchConfig,
    round_id: str,
    baseline_mean: float,
    promotion_result: dict[str, Any] | None,
    variant_summary: dict[str, Any] | None = None,
    contract_shape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append this round's machine-readable comparison row and return it.

    Written every round, promotion or no: clause 1 is a *per-round* row, and a
    round that promoted nothing is the evidence that the check ran anyway.

    The row also carries the contract shape (#789): the live contract's gate-stack
    and prohibition ratios, and the winning candidate's own two with the surface they
    were measured on. `candidate_overlay` is that candidate's overlay directory, or
    `None` when the round had no candidate that reached the contract check — the
    ratios are then `None`, not zero, because a round with nothing to measure and a
    contract with no gate stack are different facts. This is the only append-only
    series the cross-round ratchet reads; a second file would be a second series that
    can silently diverge from this one.
    """
    landed = bool(
        promotion_result
        and not promotion_result.get("refused")
        and promotion_result.get("snapshot_dir")
    )
    variant_id = promotion_result["variant_id"] if landed else None
    mean = None
    if landed and variant_summary is not None:
        mean = variant_summary.get("mean_composite")
    measured = contract_shape or {}
    row: dict[str, Any] = {
        "round_id": round_id,
        "event": ROUND_SUMMARY_EVENT,
        "baseline_mean": round(float(baseline_mean), 4),
        "promoted_variant_id": variant_id,
        "promoted_variant_mean": round(float(mean), 4) if mean is not None else None,
        "snapshot_dir": promotion_result.get("snapshot_dir") if landed else None,
        # The sha the promotion was committed with (#1099). A later restore reads this
        # to answer "is this file still what the promotion wrote", and the answer has
        # to come from the commit that landed, not from whatever HEAD said at some
        # earlier point — so `promote()`'s returned sha is recorded here, in the row
        # that says the promotion happened. None whenever nothing landed.
        "vault_commit": promotion_result.get("vault_commit") if landed else None,
        "noise_floor": round(float(cfg.promotion_noise_floor), 4),
        # The variant whose shape is recorded below, named even when it was refused:
        # `promoted_variant_id` above is None in exactly the rounds the shape series
        # most needs to explain, and a ratio with no variant attached is a number a
        # reader cannot go and look up.
        "candidate_variant_id": (promotion_result or {}).get("variant_id"),
        # Every shape key is present on every row, absent measurements included, so a
        # reader of the series never has to distinguish "not measured" from "not
        # recorded". Only the keys in SHAPE_FIELDS are taken from the caller; a stray
        # key in `contract_shape` must not widen the row.
        **{k: measured.get(k) for k in SHAPE_FIELDS},
        "created_at": now_iso(),
    }
    ledger_append(cfg.paths.ledger_path, row)
    return row
