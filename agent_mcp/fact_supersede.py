"""Expire the facts a note's earlier bytes produced, once the note is re-extracted.

Nightly extraction re-reads a changed note with `existing_facts=""` and appends
what it finds; nothing ever expired what the note USED to say. Measured
2026-11-10 on the live store: 333 notes carry active facts under more than one
`source_hash`, 62,115 active rows (31%) not from their note's latest bytes. Some
of that is legitimate — a feed that appends keeps its old entries true — which
is why feeds are exempt by glob and the whole thing ships `off`.

Gate: `knowledge_graph.supersede_on_reextract.mode` = off | dry_run | apply.

The decision (`plan`) is pure over index rows, so it is testable without a
store. What it will not judge:

- a pass that did not read the whole note from offset 0 (a resumed or truncated
  pass saw only part of the bytes, so silence about a fact means nothing);
- a note matching an `exempt_globs` pattern;
- a pass whose own facts are too few beside what it would expire
  (`min_restate_ratio`): an extraction that came back thin is a bad sample of
  the note, not evidence the note stopped saying everything else.

A fact the new pass restated verbatim (the store's `text_hash`) stays, because
the extractor's duplicate refusal (#1144) keeps the OLD row for a restated
claim — its `source_hash` is the earlier one even though the current bytes still
say it.

The expiry is a mark (`expired_at` + `expired_reason`) through
`agent_mcp.facts._apply_fact_marks`, the one route every fact-file writer uses,
then `facts_idx.reindex`. Never a delete.
"""
from __future__ import annotations

import datetime
import fnmatch
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("agent_mcp.fact_supersede")

MODES = ("off", "dry_run", "apply")
REASON = "superseded: note re-extracted"


@dataclass(frozen=True)
class SupersedeConfig:
    mode: str = "off"
    min_restate_ratio: float = 0.5
    exempt_globs: tuple[str, ...] = ()


def load_config() -> SupersedeConfig:
    try:
        from app.config import CONFIG
        block = (CONFIG.get("knowledge_graph", {}) or {}).get("supersede_on_reextract", {}) or {}
    except Exception:  # noqa: BLE001 — no config means off
        return SupersedeConfig()
    mode = str(block.get("mode") or "off")
    if mode not in MODES:
        logger.warning("supersede_on_reextract.mode %r is not one of %s; treated as off",
                       mode, MODES)
        mode = "off"
    try:
        ratio = float(block.get("min_restate_ratio", 0.5))
    except (TypeError, ValueError):
        ratio = 0.5
    globs = tuple(str(g) for g in (block.get("exempt_globs") or []) if g)
    return SupersedeConfig(mode=mode, min_restate_ratio=ratio, exempt_globs=globs)


def is_exempt(source_doc: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(source_doc, g) for g in globs)


@dataclass
class Plan:
    source_doc: str
    new_hash: str
    verdict: str                      # expire | nothing | skipped
    reason: str = ""
    expire: list[dict] = field(default_factory=list)
    kept_restated: int = 0
    old_active: int = 0


def plan(source_doc: str, new_hash: str, new_fact_texts: Iterable[str],
         active_rows: list[dict], cfg: SupersedeConfig, *,
         read_whole_note: bool) -> Plan:
    """Which of a note's active rows its re-extraction supersedes. Pure."""
    from app.kg_store import text_hash

    p = Plan(source_doc=source_doc, new_hash=new_hash, verdict="nothing")
    if not new_hash:
        p.verdict, p.reason = "skipped", "no source hash"
        return p
    if is_exempt(source_doc, cfg.exempt_globs):
        p.verdict, p.reason = "skipped", "exempt feed"
        return p
    if not read_whole_note:
        p.verdict, p.reason = "skipped", "pass did not read the whole note"
        return p
    restated = {text_hash(t) for t in new_fact_texts if (t or "").strip()}
    old = [r for r in active_rows
           if r.get("source_hash") and r.get("source_hash") != new_hash]
    p.old_active = len(old)
    if not old:
        return p
    candidates = []
    for r in old:
        if r.get("text_hash") in restated:
            p.kept_restated += 1
        else:
            candidates.append(r)
    if not candidates:
        return p
    # How much the new pass said, beside what it would retire. The new pass's
    # facts are `restated` (deduplicated); a pass that came back with a handful
    # against hundreds of old rows is a thin sample, not a verdict.
    if len(restated) < cfg.min_restate_ratio * len(old):
        p.verdict = "skipped"
        p.reason = (f"thin pass: {len(restated)} facts against {len(old)} "
                    f"from earlier bytes (min_restate_ratio {cfg.min_restate_ratio})")
        return p
    unaddressable = [r for r in candidates if not r.get("fact_id") or not r.get("file_path")]
    p.expire = [r for r in candidates if r.get("fact_id") and r.get("file_path")]
    p.verdict = "expire" if p.expire else "nothing"
    if unaddressable:
        p.reason = f"{len(unaddressable)} row(s) without an id or file left alone"
    return p


def _journal(rows: list[dict], path: Optional[Path] = None) -> None:
    from app.paths import FACT_SUPERSEDE_JOURNAL_PATH
    path = path or FACT_SUPERSEDE_JOURNAL_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
    except OSError as exc:  # a witness, never a reason to fail the pass
        logger.warning("fact_supersede: journal write failed: %s", exc)


def supersede_after_reextract(source_doc: str, new_hash: str,
                              new_fact_texts: Iterable[str], *,
                              read_whole_note: bool,
                              cfg: Optional[SupersedeConfig] = None,
                              journal_path: Optional[Path] = None) -> dict:
    """Run the gate for one note after its facts were written. Never raises.

    Returns a small summary for the caller's log line.
    """
    cfg = cfg or load_config()
    if cfg.mode == "off":
        return {"mode": "off"}
    try:
        from app.kg_store import store
        rows = store().facts_idx.active_for_source(source_doc)
        pl = plan(source_doc, new_hash, list(new_fact_texts), rows, cfg,
                  read_whole_note=read_whole_note)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary = {"mode": cfg.mode, "verdict": pl.verdict, "reason": pl.reason,
                   "would_expire": len(pl.expire), "kept_restated": pl.kept_restated,
                   "old_active": pl.old_active}
        if pl.verdict != "expire":
            if pl.verdict == "skipped" and pl.old_active:
                _journal([{"ts": now, "mode": cfg.mode, "source_doc": source_doc,
                           "new_hash": new_hash, "verdict": "skipped",
                           "reason": pl.reason, "old_active": pl.old_active}],
                         journal_path)
            return summary
        journal_rows = [{"ts": now, "mode": cfg.mode, "source_doc": source_doc,
                         "new_hash": new_hash, "old_hash": r.get("source_hash"),
                         "entity": r.get("entity"), "file": r.get("file_path"),
                         "fact_id": r.get("fact_id"),
                         "fact": str(r.get("fact") or "")[:200]}
                        for r in pl.expire]
        if cfg.mode == "dry_run":
            _journal(journal_rows, journal_path)
            return summary
        from agent_mcp import facts as facts_mod
        root = facts_mod.FACTS_ROOT
        marks = {(r["file_path"], str(r["fact_id"])): REASON for r in pl.expire}
        files = sorted({root / r["file_path"] for r in pl.expire})
        applied = facts_mod._apply_fact_marks(
            marks, files, field="expired_at", stamp=now, reason_field="expired_reason")
        marked = {(m["file"], m["id"]) for m in applied["matched_facts"]}
        _journal([dict(j, applied=(j["file"], str(j["fact_id"])) in marked)
                  for j in journal_rows], journal_path)
        summary["expired"] = applied["marked"]
        summary["unapplied"] = len(applied["unapplied"])
        if applied.get("index_warning"):
            summary["warning"] = applied["index_warning"]
        return summary
    except Exception as exc:  # noqa: BLE001 — extraction already succeeded
        logger.warning("fact_supersede: %s: %s", source_doc, exc)
        return {"mode": cfg.mode, "error": str(exc)}
