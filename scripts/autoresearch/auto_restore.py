"""Restoring a promotion automatically when a later baseline declines past the floor.

Backlog #1099, the clause #429 reserved for a human: #429 landed *record and
surface* — :func:`scripts.autoresearch.post_promotion.compare` names a
beyond-noise decline in the round report and prints the snapshot it could be
restored from — and Alan signed off on the loop using that verdict itself. This is
the half that acts on it, and it is deliberately a *separate module* from the
recorder: #429's separation test pins `post_promotion`'s import list to the ones a
recorder is allowed to have, so the code that writes prompt files must live
somewhere else or that tripwire becomes a lie.

What it does
------------
`restore_for_decline(cfg, round_id, comparison, ...)` is called by
:func:`scripts.autoresearch.run_round.run` once, after this round's fresh baseline
has been compared against the previous promotion's recorded mean:

  * no beyond-noise decline → nothing happens and no row is written;
  * a decline → `promote.rollback(..., promotion=...)` restores the promotion's
    snapshot through `scripts.automod.vault_round.land`, which validates the
    restored contract with the real loaders, commits exactly those paths on the
    vault's `main`, and reverts them if any validator fails. There is no copy
    path here that bypasses that route.
  * whatever the outcome, one `event: "promotion_restored"` ledger row joins the
    round that acted, the promotion's variant id, its round, the snapshot and the
    vault sha — and a promotion with such a row is never restored again and never
    compared as the last promotion again
    (:func:`scripts.autoresearch.post_promotion.restored_promotion_rounds`).

What it refuses
---------------
*No snapshot on record* and *no vault commit for the promotion* are refusals, not
restores: without the first there is nothing to write, and without the second
nothing can tell "this file is still what the promotion wrote" from "someone
edited this file afterwards", which is the difference between undoing a promotion
and silently reverting a person's work. `promote.rollback` refuses both, by name
per file, and this module records the refusal and says it in the report. A
`--dry-run` round never restores either — measuring a hypothesis is not a licence
to rewrite the live contract.

The floor it acts on is #324's measured cross-round baseline noise
(`DEFAULT_NOISE_FLOOR`, 0.1389), the same number the recorder compares against, so
"declined past noise" means one thing across the whole chain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from scripts.automod import state as automod_state

from . import post_promotion, promote
from .common import AutoresearchConfig, ledger_append, now_iso

logger = logging.getLogger("autoresearch.auto_restore")

#: Statuses that write no ledger row: a round that found nothing to undo took no
#: decision worth recording, and a row per round for "not attempted" would bury
#: the rows that matter.
SILENT_STATUSES = frozenset({"not_regression"})

#: Statuses that mean "this promotion has been undone", and so must not be
#: undone again or compared as the live contract's last promotion. `no_change`
#: counts: the content already matching HEAD is the same end state, reached
#: without a second commit to attribute.
RESTORED_STATUSES = post_promotion.RESTORED_STATUSES


restore_rows = post_promotion.restore_rows


def already_restored(ledger_path: Path, promoted_round_id: str | None) -> dict[str, Any] | None:
    """The row saying this promotion was already undone, or None if it never was.

    Keyed on the *promoting* round rather than the variant id: two rounds can
    promote the same variant id string, and the thing being protected from a
    second restore is one specific landed commit.
    """
    if not promoted_round_id:
        return None
    for row in reversed(restore_rows(ledger_path)):
        if row.get("restored_round_id") == promoted_round_id and (
                row.get("status") in RESTORED_STATUSES):
            return row
    return None


def promotion_vault_sha(variant_id: str | None,
                        ledger_path: Path | None = None) -> str | None:
    """The vault commit `promote()` used to land `variant_id`, from the landing ledger.

    Back-fill for promotions recorded before the round summary row carried a
    `vault_commit`: every successful `promote()` ends in
    `scripts.automod.vault_round.land`, whose `vault_land` row carries both the sha
    and the first 200 characters of the message it committed, and `promote()`'s
    subject is `autoresearch: promote <variant id>`. That coupling is real and is
    pinned by a test that drives a real `promote()` and reads the sha back — if the
    subject ever changes, the test names the lookup rather than this quietly
    answering None forever.

    None is a refusal downstream, never a guess.
    """
    if not variant_id:
        return None
    subject = f"autoresearch: promote {variant_id}"
    rows = automod_state.read_events(ledger_path, limit=10 ** 9)
    for row in reversed(rows):
        if row.get("event") != "vault_land" or not row.get("ok") or not row.get("commit"):
            continue
        first = str(row.get("message") or "").splitlines()[0].strip() if row.get("message") else ""
        if first == subject:
            return str(row["commit"])
    return None


def _row(cfg: AutoresearchConfig, restoring_round: str, prior: dict[str, Any], *,
         status: str, comparison: dict[str, Any], reason: str = "",
         vault_commit: str | None = None, refused: list[dict] | None = None,
         restored_files: list[str] | None = None) -> dict[str, Any]:
    """Append the ledger row for one restore decision and return it.

    Every field is present on every row, refusals included: a reader asking "was
    this promotion undone, and where is the sha" must get `None` rather than a
    missing key, the same rule `record_round_summary` follows for its shape fields.
    """
    row: dict[str, Any] = {
        "round_id": restoring_round,          # the round that acted
        "event": post_promotion.RESTORE_EVENT,
        "restored_round_id": prior.get("round_id"),   # the promotion being undone
        "promoted_variant_id": prior.get("promoted_variant_id"),
        "snapshot": prior.get("snapshot_dir") or "",
        "snapshot_ts": comparison.get("snapshot_ts") or "",
        "vault_commit": vault_commit,
        "status": status,
        "reason": reason,
        "refused_files": refused or [],
        "restored_files": restored_files or [],
        "decline": comparison.get("decline"),
        "noise_floor": comparison.get("noise_floor"),
        "created_at": now_iso(),
    }
    ledger_append(cfg.paths.ledger_path, row)
    return row


def _with_lines(outcome: dict[str, Any]) -> dict[str, Any]:
    """Attach this outcome's report lines to itself.

    `post_promotion.report_section` renders the restore verdict from
    `outcome["report_lines"]` rather than by calling back in here because #429's
    separation test keeps the recorder free of this module, and a caller that has to
    remember to render before it renders the section is a caller that will forget.
    """
    return {**outcome, "report_lines": report_lines(outcome)}


def restore_for_decline(cfg: AutoresearchConfig, restoring_round: str,
                        comparison: dict[str, Any] | None,
                        prior: dict[str, Any] | None = None, *,
                        dry_run: bool = False,
                        automod_ledger_path: Path | None = None) -> dict[str, Any] | None:
    """Act on one post-promotion comparison. Returns the outcome, or None if there
    was no comparison to act on at all.

    Never raises: a restore that cannot happen is reported in the round report and
    the ledger, and the round continues. The round's own verdict on the bench is
    not this function's to fail.
    """
    if comparison is None:
        return None
    if not comparison.get("regression"):
        return {"status": "not_regression", "restored": False, "row": None}

    prior = dict(prior or {})
    prior.setdefault("round_id", comparison.get("prior_round_id"))
    prior.setdefault("promoted_variant_id", comparison.get("promoted_variant_id"))
    prior.setdefault("snapshot_dir", comparison.get("snapshot_dir") or "")
    variant_id = prior.get("promoted_variant_id")
    snapshot_dir = str(prior.get("snapshot_dir") or "")
    ts = comparison.get("snapshot_ts") or Path(snapshot_dir).name

    def say(status: str, **kw: Any) -> dict[str, Any]:
        row = None if status in SILENT_STATUSES else _row(
            cfg, restoring_round, prior, status=status, comparison=comparison, **kw)
        return _with_lines({"status": status, "restored": status == "restored",
                            "row": row, "vault_commit": (row or {}).get("vault_commit")})

    if dry_run:
        logger.info("round %s: dry run, so the decline past the noise floor is "
                    "reported but promotion %s is not restored", restoring_round, variant_id)
        return say("dry_run", reason="the round was a dry run: measuring a hypothesis "
                                     "is not a licence to rewrite the live contract")
    if not ts:
        logger.error("round %s: decline past the noise floor against %s, but that "
                     "promotion has no snapshot on record — nothing to restore",
                     restoring_round, variant_id)
        return say("no_snapshot", reason="the promotion has no snapshot directory on "
                                         "record, so no rollback point exists")
    if not (cfg.paths.snapshots_dir / ts).exists():
        logger.error("round %s: snapshot %s named by promotion %s is not on disk under %s",
                     restoring_round, ts, variant_id, cfg.paths.snapshots_dir)
        return say("no_snapshot", reason=f"snapshot {ts} is not on disk under "
                                         f"{cfg.paths.snapshots_dir}")
    done = already_restored(cfg.paths.ledger_path, prior.get("round_id"))
    if done:
        logger.info("round %s: promotion %s (round %s) was already restored in round "
                    "%s — not restoring it twice", restoring_round, variant_id,
                    prior.get("round_id"), done.get("round_id"))
        return _with_lines({
            "status": "already_restored", "restored": False, "row": None,
            "vault_commit": done.get("vault_commit"),
            "restored_by": done.get("round_id"),
            "promoted_variant_id": variant_id,
            "restored_round_id": prior.get("round_id"),
            "snapshot_ts": ts})

    vault_commit = str(prior.get("vault_commit") or "") or promotion_vault_sha(
        variant_id, automod_ledger_path)
    if not vault_commit:
        logger.error("round %s: no vault commit is on record for promotion %s, so the "
                     "restore cannot prove which files still hold the promotion's bytes",
                     restoring_round, variant_id)
        return say("no_vault_commit",
                   reason="no vault commit is on record for this promotion: the restore "
                          "cannot tell a file the promotion wrote from one edited since")

    try:
        out = promote.rollback(cfg, ts, promotion={
            "variant_id": variant_id,
            "promoted_round_id": prior.get("round_id"),
            "restoring_round_id": restoring_round,
            "vault_commit": vault_commit,
            "decline": comparison.get("decline"),
            "noise_floor": comparison.get("noise_floor"),
        })
    except Exception as exc:  # noqa: BLE001 — a round must not die on its own undo
        logger.exception("round %s: restoring %s raised", restoring_round, variant_id)
        return say("error", reason=f"the restore raised: {exc}")

    refused = out.get("refused_files") or []
    if out.get("vault_commit"):
        logger.info("round %s: restored promotion %s from %s — vault commit %s%s",
                    restoring_round, variant_id, ts, out["vault_commit"],
                    f" (refused: {', '.join(r['file'] for r in refused)})" if refused else "")
        return say("restored", vault_commit=out["vault_commit"], refused=refused,
                   restored_files=out.get("restored_files") or [])
    if out.get("no_change"):
        logger.info("round %s: promotion %s is already undone — %s matches the "
                    "snapshot and HEAD, nothing to commit", restoring_round, variant_id, ts)
        return say("no_change", refused=refused,
                   reason="the live files already match the snapshot, so there was "
                          "nothing to commit")
    reason = "; ".join(out.get("refused") or []) or str(out.get("error") or "no vault commit")
    logger.error("round %s: decline past the noise floor against %s, restore refused: %s",
                 restoring_round, variant_id, reason)
    return say("refused", reason=reason, refused=refused)


def report_lines(outcome: dict[str, Any] | None) -> list[str]:
    """The restore verdict for the round report's post-promotion section.

    Names the variant, the snapshot and the vault sha when it restored, or the
    refusal reason when it did not — a reader of a round report has to be able to
    tell "the loop undid it" from "the loop looked and declined", and both are
    answers a human needs to see without opening the ledger.
    """
    if not outcome:
        return []
    status = outcome.get("status")
    row = outcome.get("row") or {}

    def field(key: str) -> Any:
        return row.get(key) or outcome.get(key)

    vid = field("promoted_variant_id") or "(unknown)"
    snap = field("snapshot_ts") or ""
    promoted_round = field("restored_round_id") or "(unknown round)"
    if status == "restored":
        refused = row.get("refused_files") or []
        return [
            f"- **PROMOTION RESTORED automatically (backlog #1099)**: variant `{vid}` "
            f"(round `{promoted_round}`) restored from snapshot `{snap}` — files "
            f"{', '.join(row.get('restored_files') or []) or '(none)'}; vault commit "
            f"`{row.get('vault_commit')}`, one revert to redo it. Ledger row "
            f"`{post_promotion.RESTORE_EVENT}`.",
        ] + (["- refused by name, not overwritten: "
              + "; ".join(f"`{r['file']}` ({r['reason']})" for r in refused) + "."]
             if refused else [])
    if status == "no_change":
        return [f"- already at the snapshot's content: `{vid}` (round `{promoted_round}`) "
                f"needed no restore commit — the live files match snapshot `{snap}`, so "
                f"the promotion counts as undone and will not be compared as the last "
                f"promotion again."]
    if status == "already_restored":
        return [f"- `{vid}` (round `{promoted_round}`) was already restored (vault commit "
                f"`{outcome.get('vault_commit')}`) and is not restored twice; it is no "
                f"longer compared as the last promotion either."]
    if status in ("no_snapshot", "no_vault_commit", "refused", "error", "dry_run"):
        head = ("- **NOT RESTORED (dry run)**: " if status == "dry_run"
                else "- **DECLINE PAST NOISE FLOOR, NOT RESTORED**: ")
        return [f"{head}variant `{vid}` (round `{promoted_round}`), snapshot "
                f"`{snap or '(none)'}` — {row.get('reason') or outcome.get('reason') or status}."
                ] + (["- refused by name, not overwritten: "
                      + "; ".join(f"`{r['file']}` ({r['reason']})"
                                  for r in row.get("refused_files") or []) + "."]
                     if row.get("refused_files") else [])
    return []
