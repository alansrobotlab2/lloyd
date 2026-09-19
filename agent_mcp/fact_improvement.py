#!/usr/bin/env python3
"""
Lloyd MCP Server: fact improvement — the feedback loop over the knowledge graph.

Backlog #376, Priority 1. Cognee ships an `improve` operation that learns from
feedback to refine memory quality; until now Lloyd had the *writers*
(`fact_invalidate`, `fact_resolve`) and no caller that knew when to use them.
`fact_resolve(auto_resolve=…)` had zero automated callers, and the one place
that claimed to learn from feedback — dream consolidation — is dead text (#463).

Why this is a module and not just a cron line over `fact_resolve`
----------------------------------------------------------------
The detector is not a verdict. `_detect_contradictions_sync` fires when two
facts of the same entity share >0.6 token overlap, which is *two facts phrased
alike*, not necessarily two facts that disagree — measured on `Lloyd`: 32,857
"contradictions" from 5,489 facts. That heuristic is exactly why
`fact_resolve`'s `auto_resolve` stopped defaulting to true. So a loop that
auto-resolved whatever the detector returned would be a fact-deletion machine
with extra steps.

Every action here therefore needs an independent reason *on top of* the
detector's pairing:

  confidence    the two sides disagree on confidence → `fact_resolve`'s case
  created_at    one was written ≥ MIN_STALE_GAP_DAYS after the other, so the
                older one is the one a later write superseded → expire it
  same day, same confidence → no basis. Reported, never acted on.

Writers are the existing tools, called as functions. This module never edits a
fact file: three writers of `expired_at` would be the same drift bug
`fact_profile` and the router already had (agent_mcp/facts.py:71-82), and
`fact_*` are the documented owners.

Dry-run by default. `apply=True` is opt-in, and each run writes a record under
`_pipeline/improvement/` with before/after active-fact counts so a run can be
audited or its reasoning re-read afterwards.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path

from app.paths import LLOYD_HOME, VAULT_FACTS_ROOT, VAULT_ROOT
from agent_mcp._shared import _find_entity_dir
from agent_mcp.facts import _apply_fact_marks, _detect_contradictions_sync
from agent_mcp.retrieval import get_facts_sync as _get_facts_sync
from app.kg_store import store as _store

logger = logging.getLogger("lloyd.improvement")

# Overridable module-level names: tests point these at a temp tree.
FACTS_ROOT = VAULT_FACTS_ROOT
# Two places a correction gets recorded, in two shapes. The compiled-in single
# path could see only the first: `memory/corrections.md`, whose newest heading
# was 2026-05-08 and whose file mtime was Aug 23, while the operator's live
# corrections — `## corrections_log` bullets in `lloyd/USER.md`, dated
# 09-07/08/09 — were invisible to the loop (no `.py` in the repo named
# `corrections_log` at all). Both shapes now get a parser and both files get
# read, because pointing the heading regex at a bullet list returns a silent
# zero, which is the old blindness with a new filename in the record.
CORRECTIONS_PATH = VAULT_ROOT / "memory" / "corrections.md"          # `## <date> — <text>`
CORRECTIONS_BULLETS_PATH = VAULT_ROOT / "lloyd" / "USER.md"          # `- 2026-09-11: <text>`
CORRECTIONS_BULLETS_SECTION = "corrections_log"
# Overridable for tests, like FACTS_ROOT. None means "the two constants above",
# read at CALL time rather than import time: a caller that repoints
# `CORRECTIONS_PATH` — every test of this reader does — has to change what the
# read sees, and a list assembled here at import would have frozen the real
# paths before the first patch landed.
CORRECTIONS_SOURCES: list[Path] | None = None


def corrections_paths() -> list[Path]:
    """The correction logs this reader will consult, in order."""
    if CORRECTIONS_SOURCES is not None:
        return list(CORRECTIONS_SOURCES)
    # Read the module globals, so repointing either constant repoints the read.
    return [CORRECTIONS_PATH, CORRECTIONS_BULLETS_PATH]
# A correction is evidence about a *recent* mistake. Undated, a March heading fed
# the nightly loop forever and its two entities outranked every fresh drift write
# in every run, permanently. Entries outside the window are counted in the record
# and contribute no entity.
CORRECTIONS_WINDOW_DAYS = 30
RECORD_DIR = LLOYD_HOME / "_pipeline" / "improvement"

# Entity dirs whose mtime is inside this window count as "a writer has been
# here recently, and that fresh claim is the one most likely to already be
# stale". Three days: the nightly extractor runs daily, so one cycle of slack.
DRIFT_WINDOW_DAYS = 3
# The minimum age gap before "written later" is evidence of "superseded".
# Same-day pairs are re-phrasings, not corrections — that is the class that
# produced the 32,857 false positives.
MIN_STALE_GAP_DAYS = 1.0
# Act only on pairs the detector flagged for *opposing terms* (working/broken,
# enabled/disabled, true/false…). The detector's other trigger is
# `_token_overlap > 0.6` alone, which labels two facts that say nearly the same
# thing a contradiction — on this tree that is the dominant mode, and picking
# one of a near-duplicate pair to expire is a dedupe decision made by whichever
# fact happened to carry a lower confidence number. Measured: acting on those
# pairs is what moved `fact_entity_recall` 0.35 -> 0.30, i.e. it deleted useful
# facts. Near-duplicate pairs are reported, never acted on.
REQUIRE_OPPOSING_TERMS = True
# Cap per entity: a god-node's contradiction list is noise, and expiring 20
# facts because a heuristic shrugged is not an improvement.
MAX_ACTIONS_PER_ENTITY = 5
# Cap per run across all entities.
MAX_ACTIONS_PER_RUN = 25
# Substring length used to aim `fact_invalidate` at one fact. The tool matches
# on a case-insensitive substring, so a 60-char prefix is specific in practice.
_SUBSTRING_LEN = 60

# Entity-shaped token in a corrections heading: capitalized word-run that is
# also a known entity, checked against the store — a heading alone is a guess.
_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9][A-Za-z0-9._-]{1,}\b")
_DATE_IN_HEAD_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b|\b[A-Z][a-z]{2,8}\s+\d{1,2}\b")


# ── signal sources ───────────────────────────────────────────────────────────

def _known_entities() -> dict[str, str]:
    """lowercased entity name → canonical, from the entity registry.

    An empty map means the store is unreadable; callers must then decline to
    act rather than fall back to guessing from the heading's own capitalisation
    — that is how you end up expiring facts about a word.
    """
    try:
        return {name.lower(): name for name in _store().entities.all()}
    except Exception:  # noqa: BLE001 - a store that cannot answer must not become a guess
        return {}


_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_BULLET_DATE_RE = re.compile(
    r"^\s*[-*]\s*(?:\*\*)?\s*(\d{4}-\d{2}-\d{2})\s*(?:\*\*)?\s*[:.\-—]?\s*(.*)$")
_SECTION_HEAD_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
# `2026-09-11 02:41 PDT — ` between a heading's date and its text. Stripped so a
# heading's entity is not competing with its own timestamp for the token scan.
_TIME_IN_HEAD_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*[A-Za-z]{1,4}?\s*[—–-]?\s*")

# What the last corrections read saw, per file. `read_correction_signals` is a
# pure list in, list out today, and a `[]` it returns means four different
# things: no log, an empty log, a log whose format it cannot parse, a log whose
# entries are all too old. The record has to say which, or a silent zero keeps
# reading as "no corrections tonight" — see the run that reported exactly that
# while reading a four-month-old file. Overwritten on every read; tests and the
# run record read it through `last_corrections_read()`.
# What a run record says when the corrections read never happened — a
# `--sources drift` pass consults no log, and `corrections_status: null` would
# read as "the log was empty", which is the same false verdict in a new costume.
_NOT_READ: dict = {"status": "not_read", "sources": {}, "paths_yielding_signals": []}
_LAST_CORRECTIONS_READ: dict = {}

# Every status the corrections read can report, best first. One vocabulary, so a
# zero in the record is never ambiguous: `empty` means the log is empty,
# `undated`/`no_section`/`unreadable`/`missing` mean the loop could not see the
# log, and the two middle ones say whether there was anything to see inside the
# window or inside the entity registry. `read_correction_signals` is where the
# file-level status (`_corrections_entries`) is combined with the window and the
# registry into one of these.
CORRECTIONS_STATUSES = ("signals", "empty", "entries_no_entity", "no_entries_in_window",
                        "undated", "no_section", "registry_unreadable", "unreadable",
                        "missing", "not_read")


def _entry_date(line: str) -> tuple[str | None, str]:
    """(iso date or None, entry text) for a corrections line."""
    m = _DATE_RE.match(line.strip().lstrip("#").strip())
    if m:
        rest = line.strip().lstrip("#").strip()[len(m.group(0)):].lstrip(" —–-\t")
        return m.group(1), rest
    m = _BULLET_DATE_RE.match(line)
    if m:
        return m.group(1), m.group(2)
    return None, ""


def _corrections_entries(path: Path) -> tuple[list[tuple[str | None, str]], str]:
    """Parse one corrections log into (dated entries, status).

    Two shapes, because the two live logs use two shapes:
      * `memory/corrections.md` — one heading per entry, `## 2026-05-08 14:37 PDT
        — Tool calls without ToolSearch schema loading`;
      * `lloyd/USER.md` `## corrections_log` — bullets, `- 2026-09-11: …`, with
        standing prose between them that is not an entry.
    A heading-only parser pointed at the second returns zero, and zero from an
    unparseable format is the exact failure this module is here to make visible.

    Shape, not path, picks the parser: a file that carries a `## corrections_log`
    section is read as a bullet log, everything else as dated headings. Keying on
    the filename is the mistake one reversion made — it parses whatever the file
    is called instead of what it contains.

    Status meanings, so a zero in the record can be read:
      `missing`     the file is not there
      `unreadable`  it is there and cannot be read
      `no_section`  a bullet log whose `## corrections_log` section is absent
      `undated`     entries are present and none carries a date
      `empty`       it parsed cleanly and holds no correction at all
    Window-filtering happens in the caller, which knows the window; this reports
    what the file *is*, not what the run chose to use.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return [], "missing"
    except OSError as exc:
        logger.debug("corrections log %s unreadable: %s", path, exc)
        return [], "unreadable"

    start = None
    for m in _SECTION_HEAD_RE.finditer(text):
        if m.group(2).strip().strip(":").strip("`").lower() == CORRECTIONS_BULLETS_SECTION:
            start = m.end()
            break
    if start is not None:
        # Bullet log: entries are `- 2026-09-11: …` lines; anything else in the
        # section (the `## corrections_log` header's own standing prose) is not an
        # entry, and a run that counted those as corrections would be crediting
        # the operator with a mistake they never made.
        nxt = _SECTION_HEAD_RE.search(text, start)
        body = text[start:nxt.start()] if nxt else text[start:]
        entries, bullets = [], 0
        for line in body.splitlines():
            if not line.strip().startswith(("-", "*")):
                continue
            bullets += 1
            date, rest = _entry_date(line)
            entries.append((date, rest))
        if not bullets:
            return [], "empty"
        return entries, ("undated" if not any(d for d, _ in entries) else "entries")

    heads = list(_SECTION_HEAD_RE.finditer(text))
    entries = []
    for m in heads:
        date, rest = _entry_date(m.group(2))
        if date:
            entries.append((date, _TIME_IN_HEAD_RE.sub("", rest) or m.group(2)))
    if not entries:
        # A file carrying only its own `# Title` is an empty log; one with real
        # sections that name no date is a log the reader cannot parse. The two
        # look identical to a caller that only sees `[]`, and they mean "nothing
        # to correct" and "I could not read you" — which is the whole
        # distinction this reader exists to make.
        return [], ("empty" if len(heads) <= 1 else "undated")
    return entries, "entries"


def read_correction_signals(limit: int = 25, window_days: int = CORRECTIONS_WINDOW_DAYS,
                            now: datetime.datetime | None = None) -> list[dict]:
    """Entities named in the operator's own corrections logs.

    Reads every path in `CORRECTIONS_SOURCES` — the compiled-in
    `memory/corrections.md` *and* `lloyd/USER.md`'s `## corrections_log`, which
    is where corrections have actually been written since the compiled-in file
    stopped being. An entry is credited to an entity only when a token in it is
    a registered entity name; dates and prose are stripped first, so
    "2026-09-08 — TTS service status" yields `TTS` and nothing else does.

    Entries older than `window_days` contribute nothing. Undated lines contribute
    nothing either, and are counted: a correction that cannot say when it
    happened cannot say whether it still applies.

    The `[]` case is where the honesty lives — see `last_corrections_read()`.
    """
    known = _known_entities()
    now = now or datetime.datetime.now(datetime.timezone.utc)
    read: dict[str, dict] = {}
    out: list[dict] = []
    seen: set[str] = set()
    for path in corrections_paths():
        entries, status = _corrections_entries(Path(path))
        in_window = 0
        outside = 0
        undated = 0
        for date, text in entries:
            if date is None:
                undated += 1
                continue
            try:
                when = datetime.datetime.fromisoformat(date).replace(
                    tzinfo=datetime.timezone.utc)
            except ValueError:
                undated += 1
                continue
            if abs((now - when).days) > window_days:
                outside += 1
                continue
            in_window += 1
            if not known or len(out) >= limit:
                continue
            head = _DATE_IN_HEAD_RE.sub(" ", text)
            for token in _TOKEN_RE.findall(head):
                canonical = known.get(token.lower())
                if not canonical or canonical in seen:
                    continue
                seen.add(canonical)
                out.append({"entity": canonical, "source": "corrections",
                            "evidence": f"{date} {text}".strip()[:200],
                            "corrections_path": str(path)})
                break
        signals_here = sum(1 for s in out if s.get("corrections_path") == str(path))
        if status == "entries" and not in_window:
            status = "no_entries_in_window"      # the log is fine; the window is not
        elif status == "entries" and not signals_here:
            status = "entries_no_entity"          # in-window entries name no entity
        read[str(path)] = {"status": status, "entries": len(entries),
                           "in_window": in_window, "outside_window": outside,
                           "undated": undated, "signals": signals_here}
    global _LAST_CORRECTIONS_READ
    if not known and not out:
        # No registry means every token is a guess. Say the store is why the read
        # produced nothing, rather than reporting a log that is fine as empty.
        for info in read.values():
            if info["status"] in ("entries", "entries_no_entity", "empty",
                                  "no_entries_in_window"):
                info["status"] = "registry_unreadable"
                info["why"] = "entity registry unreadable; no token can be resolved"
    _LAST_CORRECTIONS_READ = {"sources": read,
                              "paths_yielding_signals": sorted(
                                  {s["corrections_path"] for s in out}),
                              "status": _roll_up(read, bool(out), known)}
    return out


def _roll_up(read: dict[str, dict], got_signals: bool, known: dict) -> str:
    """One status for the whole read: signals beat everything, else the worst file."""
    if got_signals:
        return "signals"
    if not known:
        return "registry_unreadable"
    worst = "empty"
    for info in read.values():
        st = info.get("status", "missing")
        if st in CORRECTIONS_STATUSES and \
                CORRECTIONS_STATUSES.index(st) > CORRECTIONS_STATUSES.index(worst):
            worst = st
    return worst


def last_corrections_read() -> dict:
    """What the most recent `read_correction_signals()` call actually saw.

    `{status, sources: {path: {status, entries, in_window, outside_window,
    undated, signals}}, paths_yielding_signals}`. The run record stores this and
    stores `corrections_path` from `paths_yielding_signals`, so a zero in the
    record can be told apart from an unread file — the distinction the compiled-in
    constant destroyed by naming itself whether or not it had been read.

    Before any read has happened it reports `not_read`: a pass that consulted no
    log has to be able to say so, and it must not inherit the shape of a pass
    that read one and found nothing.
    """
    return dict(_LAST_CORRECTIONS_READ) if _LAST_CORRECTIONS_READ else dict(_NOT_READ)


def _drift_candidates(days: int = DRIFT_WINDOW_DAYS) -> list[dict]:
    """Every entity with a fact file written inside `days`, newest write first.

    This is the whole ranked candidate pool, untruncated, and it exists as a
    function of its own so a run can name the denominator it selected from:
    `read_drift_signals` returns a `limit`-sized prefix of this list, and
    #699's complaint was that nothing in the record said what the prefix was a
    prefix *of*.

    Fact-file mtimes, not `facts_idx.created_at`: the index records when a
    fact was *written into the graph*, and the whole tree was rebuilt on
    09-03, so every backfilled row carries a rebuild date. Directory mtimes
    would be wrong too — they only move when a file is added or removed, and
    the nightly extractor rewrites in place. The tree walk is 0.4 s over
    23,625 entity dirs on the live box.

    Order is newest write first, ties broken by name so the ranking is
    reproducible across runs. Sorting by name and truncating — which is what
    this did until #699 — made the nightly `--limit 40` the ASCII-earliest 40 of
    the drifted pool (1,594 entities measured 2026-09-15), the same 40 on three
    consecutive nights (records 20260912-210116, 20260913-210013 and
    20260914-210026 are set-identical), and that drift slice shared 0 of 38
    entities with the 38 most-recently-written ones.
    """
    cutoff = datetime.datetime.now().timestamp() - days * 86400
    try:
        entries = list(FACTS_ROOT.iterdir())
    except OSError:
        return []
    ranked: list[tuple[float, str]] = []
    for child in entries:
        try:
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            newest = max((p.stat().st_mtime for p in child.glob("*.md")), default=0)
            if newest < cutoff:
                continue
        except OSError:
            continue
        ranked.append((newest, child.name))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    return [{"entity": name, "source": "drift",
             "evidence": f"fact file written {datetime.datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}"}
            for newest, name in ranked]


def read_drift_signals(days: int = DRIFT_WINDOW_DAYS, limit: int = 50) -> list[dict]:
    """The `limit` entities whose fact files were written most recently.

    This is the only automatic quality signal the store itself emits: a fact
    written this week is a claim about a world that has moved since. It is not
    a verdict either — it selects *where to look*, and the contradiction
    pairing plus the `created_at` ordering decides whether anything is stale.

    Ranked newest-first because the point of the signal is recency: the
    entities a writer touched today are the ones whose newest claim is most
    likely already stale, and an alphabetical slice could not see them (see
    `_drift_candidates`). Callers that report coverage need the size of the
    pool this slice came from: use `_drift_candidates(days)` for that, or read
    `drift_candidates_total` off a run record.
    """
    return _drift_candidates(days)[:limit]


def _collect_signals(sources=("corrections", "drift"), days: int = DRIFT_WINDOW_DAYS,
                     limit: int = 40) -> tuple[list[dict], dict]:
    """`collect_signals` plus what it selected from.

    The tally is read off the same tree walk that produced the slice, so the
    denominator cannot disagree with the entities the run actually scanned.
    `drift_candidates_total` is None when drift was not consulted at all — a
    run over `--sources corrections` did not measure a drift population, and 0
    would be a number it never measured.
    """
    wanted = set(sources)
    out: list[dict] = []
    seen: set[str] = set()
    drift_candidates_total: int | None = None
    if "corrections" in wanted:
        out.extend(read_correction_signals(limit=limit))
    if "drift" in wanted:
        candidates = _drift_candidates(days)
        drift_candidates_total = len(candidates)
        out.extend(candidates[:limit])
    deduped = []
    for sig in out:
        if sig["entity"] in seen:
            continue
        seen.add(sig["entity"])
        deduped.append(sig)
        if len(deduped) >= limit:
            break
    return deduped, {"drift_candidates_total": drift_candidates_total}


def collect_signals(sources=("corrections", "drift"), days: int = DRIFT_WINDOW_DAYS,
                    limit: int = 40) -> list[dict]:
    """Union of the enabled sources, deduped by entity, corrections first.

    Corrections outrank drift because a user saying "that was wrong" is a
    better reason to look than a file being new — and drift is ranked
    newest-write-first within its own block (see `read_drift_signals`).
    """
    return _collect_signals(sources=sources, days=days, limit=limit)[0]


# ── the plan ─────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Whitespace/case/punctuation-collapsed fact text, for uniqueness counts."""
    return " ".join(str(text or "").lower().split())


def _iso(value) -> datetime.datetime | None:
    """Parse a fact timestamp. Values arrive as ISO strings or dates, and the
    tree holds both naive and offset-bearing stamps."""
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    try:
        parsed = datetime.datetime.fromisoformat(str(value).strip().rstrip("Z") + (
            "+00:00" if str(value).strip().endswith("Z") else ""))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def _loser_by_age(f1: dict, f2: dict) -> tuple[dict, dict, str] | None:
    """Return (loser, winner, reason) when write order is evidence."""
    t1, t2 = _iso(f1.get("created_at")), _iso(f2.get("created_at"))
    if not t1 or not t2:
        return None
    gap = abs((t2 - t1).total_seconds()) / 86400
    if gap < MIN_STALE_GAP_DAYS:
        return None
    loser, winner = (f1, f2) if t1 < t2 else (f2, f1)
    return loser, winner, (
        f"contradiction detector paired it with {winner.get('fact', '')[:70]!r}, "
        f"written {gap:.1f} days later; created_at order makes the older claim the superseded one")


def plan_entity(entity: str, max_actions: int = MAX_ACTIONS_PER_ENTITY) -> dict:
    """Decide what — if anything — to change about one entity's facts.

    Reads through the same `_get_facts_sync` the recall path uses, so the plan
    is made on the facts a query would actually have been answered with — and
    read WITH a file attribution, because a plan that cannot name the file its
    loser lives in cannot be executed safely: fact ids are a per-file counter
    (`app.fact_ids.next_fact_id`), so one id is one fact in this view and
    several facts on disk. The detector is handed that same list instead of
    re-reading, so what gets judged and what gets written come from one read.
    """
    facts_view = _get_facts_sync(entity, with_source_file=True).get("facts", [])
    detection = _detect_contradictions_sync(entity, facts=facts_view)
    if detection.get("refused"):
        return {"entity": entity, "refused": True, "checked": detection.get("checked", 0),
                "contradictions": 0, "actions": [],
                "before_active": _active_count(entity),
                "skipped_reason": detection.get("hint", "entity too large to scan")}

    contradictions = detection.get("contradictions", [])
    if REQUIRE_OPPOSING_TERMS:
        actable = [c for c in contradictions
                   if str(c.get("reason", "")).startswith("opposing_terms")]
    else:
        actable = list(contradictions)
    actions: list[dict] = []
    planned_against: set[str] = set()
    for item in actable:
        if len(actions) >= max_actions:
            break
        f1 = item.get("fact1") or {}
        f2 = item.get("fact2") or {}
        if not f1.get("fact") or not f2.get("fact"):
            continue
        c1 = float(f1.get("confidence") or 0.5)
        c2 = float(f2.get("confidence") or 0.5)
        if c1 != c2:
            loser, winner = (f2, f1) if c1 > c2 else (f1, f2)
            kind = "confidence"
            reason = (f"contradiction detector paired it with "
                      f"{winner.get('fact', '')[:70]!r}; confidence "
                      f"{loser.get('confidence')} < {winner.get('confidence')}")
        else:
            ordered = _loser_by_age(f1, f2)
            if ordered is None:
                continue          # equal confidence, no age basis → leave it
            loser, winner, reason = ordered
            kind = "superseded"
        action = {"kind": kind, "entity": entity,
                  "category": loser.get("category"),
                  "loser_fact": loser.get("fact", ""), "loser_id": loser.get("id"),
                  # Where the loser is, not only what it is called. The writer
                  # marks inside this file, which is what turns "one action, one
                  # fact" from an aim the scan may overshoot into a scope. An
                  # action planned against a view that carried no attribution
                  # gets "" and the writer says so rather than guessing.
                  "loser_source_file": loser.get("source_file") or "",
                  "reason": reason}
        # Dedupe on the fact text, not the id: ids are per-file counters, so
        # `fact-001` in three category files is three different facts and one
        # condemned claim must not silence the other two.
        key = (action["category"], _normalise(action["loser_fact"]))
        if key in planned_against:
            continue
        planned_against.add(key)
        actions.append(action)

    return {"entity": entity, "refused": False, "checked": detection.get("checked", 0),
            "contradictions": len(contradictions),
            # The number this mechanism can move, which `fact_entity_recall` is
            # not. That metric scores whether an entity an eval query names wins
            # one of ten ranked fact slots, so retiring one side of a pair leaves
            # the claim retrievable through its twin and the number is
            # bit-identical after a correction — while expiring an entity's
            # best-scoring fact ejects it from the pool and the number drops for
            # a reason that is blast radius, not quality. The pair count over the
            # entities this pass scanned is the loop's own denominator: retiring
            # a superseded claim takes a pair with it, and nothing else moves it.
            #
            # Exactly `len(contradictions)`: one pair, one count. An earlier
            # revision of this line added the near-duplicate count again
            # (`C + (C - actable)`), so a pair the loop declines to touch was
            # counted twice — once as a pair and once as a near-duplicate — and
            # the number named in the field's own comment was not the number in
            # the field. `pairs_after` is computed from this same expression, so
            # the inflation was invisible in the delta and visible only in the
            # absolute count, which is the half anyone reads.
            "pairs_before": len(contradictions),
            # Reported so the near-duplicate class stays visible: it is the
            # auto-capture noise this loop declines to delete, and the reason
            # the metric does not move. A subset of `pairs_before`, never added
            # to it.
            "near_duplicates": len(contradictions) - len(actable),
            "actions": actions, "before_active": _active_count(entity)}


# ── applying a plan (through the existing writers) ───────────────────────────

def _active_count(entity: str | None = None) -> int:
    """Active facts in the index. -1 when the store cannot answer, so a
    missing count is never mistaken for a zero."""
    try:
        return _store().facts_idx.count(entity=entity, active_only=True)
    except Exception:  # noqa: BLE001 - StoreUnavailable and everything the driver raises
        return -1


def _aim_substring(entity: str, action: dict) -> str | None:
    """A substring that identifies exactly one current fact, or None.

    `fact_invalidate` matches case-insensitively over a category file, so a
    prefix is only safe once counted. If the full fact text still matches more
    than one stored fact, the two are indistinguishable to the writer and the
    action is dropped — a plan may not take out a fact it did not condemn, and
    on this tree that is not hypothetical: fact ids are per-file counters
    (`Assistant` has 19 facts sharing id `fact-001`), which is the same
    collateral-damage shape that made `fact_resolve(auto_resolve=true)`
    invalidate 25 facts to change 2.
    """
    target = _normalise(action["loser_fact"])
    if not target:
        return None
    facts = _get_facts_sync(entity, action.get("category")).get("facts") or []
    lowered = [(f.get("fact") or "", (f.get("fact") or "")) for f in facts]
    for length in (_SUBSTRING_LEN, 120, len(action["loser_fact"])):
        probe = action["loser_fact"][:length]
        if not probe.strip():
            continue
        hits = [t for t, _ in lowered if probe.lower() in t.lower()]
        if len(hits) == 1:
            return probe
    return None


def apply_action(action: dict, now_iso: str) -> dict:
    """Retire exactly one condemned fact. Returns the facts it marked.

    One action, one fact — by scope, not by aim. The mark names the FILE the
    plan read the loser in and the writer scans that file only, so a phrase that
    also appears in a sibling category file cannot reach it, and the scan stops
    at the first hit inside the file it does scan.

    What this replaced was a phrase aimed at the whole entity: `_aim_substring`
    picked a prefix unique across the entity and `fact_invalidate` marked
    whatever matched it, so the loop's idea of its own blast radius was the plan
    (`expired_count` came back from a scan of every category file), and the
    guard against over-reach was to halt the entity when `expired_count > 1` —
    which stopped the damage while still reporting the wrong number for it.
    `fact_resolve` selecting losers by bare id is what made that necessary: ids
    count within one file, so 2 planned actions became 25 invalidated facts on
    the live `Assistant` entity (29 active → 4). That selection is now scoped to
    (file, id) too, so this loop's writer and that tool's share one loop again.

    The field carries the reason, per the semantics `facts.py` documents: a
    contradiction pair decided on stored confidence condemns a claim that should
    never have been recorded, so its loser gets `invalid_at`; an ordering pair
    retires a claim that was true and has since been replaced, so its loser gets
    `expired_at`. The first round of this item wrote `expired_at` for both and
    recorded the mismatch as a finding — which is the kind of thing a record can
    say about itself while the code keeps doing the opposite.
    """
    aim = _aim_substring(action["entity"], action)
    if aim is None:
        return {"expired_count": 0, "skipped": "no unique match for the condemned fact"}
    entity_dir = _find_entity_dir(action["entity"])
    if entity_dir is None:
        return {"expired_count": 0, "skipped": "entity directory not found"}
    if action["kind"] == "confidence":
        field, reason_field = "invalid_at", "invalid_reason"
    else:
        field, reason_field = "expired_at", "expire_reason"
    # The attribution is relative to FACTS_ROOT (`Entity/Entity-usage.md`), so
    # it resolves against the root — joining it to the entity dir would name a
    # path one level too deep and match nothing.
    scope = ([FACTS_ROOT / action["loser_source_file"]]
             if action.get("loser_source_file")
             else list(entity_dir.glob("*.md")))
    applied = _apply_fact_marks(
        {}, scope, field=field, stamp=now_iso, reason_field=reason_field,
        text_matches={aim.lower(): ("phrase",
                                    f"improve {action['kind']}: {action['reason']}")},
        stop_after_first=True)
    if applied["marked"] == 0 and applied["unapplied"]:
        return {"expired_count": 0, "error": str(applied["unapplied"][0]["reason"])}
    return {"expired_count": applied["marked"], "field": field,
            "marked": applied["matched_facts"]}


def _fact_entity_recall(limit: int = 20) -> float | None:
    """Score the live fact tree with the eval's own scorer, on production knobs.

    `eval/run_eval.py` is a script, not a package, so it is loaded by path. Two
    traps, both documented in that file and both hit here on the first try:

      * `run_eval()`'s **signature** defaults are production's, because they are
        the same `agent_mcp.vault.RECALL_*` constants its argparse defaults use
        (#498). They used to be restated literals, and one of them —
        `rerank_alpha=0.5` against production's 0.3 — was stale, so calling
        `run_eval(queries)` directly measured a configuration nothing serves;
        this function's first version reported such a number. The knobs are
        still passed explicitly, so the measurement never depends on the
        pinning test holding. `expand_graph=True` is the one knob deliberately
        not production's (the eval runs the graph leg expanded; #1000 owns that
        claim).
      * the corpus underneath must be readable. Measured rather than assumed:
        a failure to score returns None, not 0.0.

    Returns None when it could not be measured, so "did not run" can never be
    read as "measured zero".
    """
    try:
        import importlib.util

        import yaml
        from agent_mcp import vault

        script = LLOYD_HOME / "eval" / "run_eval.py"
        spec = importlib.util.spec_from_file_location("lloyd_run_eval", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        queries = (yaml.safe_load(
            (LLOYD_HOME / "eval" / "vault_recall_queries.yaml").read_text()) or {}).get("queries") or []
        knobs = {"expand_graph": True,
                 "graph_rerank": vault.RECALL_GRAPH_RERANK,
                 "rerank_alpha": vault.RECALL_RERANK_ALPHA,
                 "graph_top_k": vault.RECALL_GRAPH_TOP_K,
                 "graph_hops": vault.RECALL_GRAPH_HOPS}
        records = module.run_eval(queries[:limit], limit=limit, **knobs)
        summary = module.summarize(records)
        if summary["overall"]["errors"]:
            logger.warning("improve: %d eval queries errored; not reporting the metric",
                           summary["overall"]["errors"])
            return None
        return round(summary["overall"]["fact_entity_recall_avg"], 4)
    except Exception as exc:  # noqa: BLE001 - a metric that cannot run is not a zero
        logger.warning("improve: fact_entity_recall could not be measured: %s", exc)
        return None


# ── the operation ────────────────────────────────────────────────────────────

def run_improvement(apply: bool = False, sources=("corrections", "drift"),
                    entities=None, days: int = DRIFT_WINDOW_DAYS,
                    limit: int = 40, report_eval: bool = False,
                    eval_limit: int = 20, record: bool = True,
                    max_actions: int = MAX_ACTIONS_PER_RUN) -> dict:
    """One improvement pass. Dry-run unless `apply=True`.

    Reports `fact_entity_recall` when asked: #376's own acceptance bar is that
    a change which cannot move that metric is not this feature, so the number
    has to come out of the operation, not out of a human remembering to run
    the eval afterwards.
    """
    started = datetime.datetime.now(datetime.timezone.utc)
    now_iso = started.isoformat()
    before = _active_count()

    if entities:
        signals = [{"entity": e, "source": "explicit", "evidence": "named by caller"}
                   for e in entities]
        tally = {"drift_candidates_total": None}
    else:
        signals, tally = _collect_signals(sources=sources, days=days, limit=limit)

    planned = taken = 0
    per_entity: list[dict] = []
    budget = max_actions
    for signal in signals:
        entity = signal["entity"]
        try:
            plan = plan_entity(entity)
        except Exception as exc:  # noqa: BLE001 - one bad entity must not stop the run
            per_entity.append({"entity": entity, "error": str(exc)})
            continue
        entry = {"entity": entity, "signal": signal["source"],
                 "contradictions": plan["contradictions"],
                 "near_duplicates": plan.get("near_duplicates", 0),
                 "refused": plan["refused"],
                 # The loop's own denominator, before its writes. `pairs_after`
                 # is measured the same way after them.
                 "pairs_before": plan.get("pairs_before", 0),
                 "planned": len(plan["actions"]), "taken": 0, "actions": []}
        planned += len(plan["actions"])
        if apply:
            for action in plan["actions"]:
                if budget <= 0:
                    entry["stopped"] = "run action budget"
                    break
                result = apply_action(action, now_iso)
                changed = int(result.get("expired_count") or 0)
                # One action, at most one fact: `apply_action` only ever aims a
                # substring that matches a single stored fact. Anything larger
                # here means the writer over-reached, so the run reports it and
                # stops rather than continuing to delete.
                if changed > 1:
                    entry["actions"].append({
                        "kind": action["kind"], "loser_fact": action["loser_fact"][:90],
                        "reason": action["reason"], "applied": False, "changed": changed,
                        "error": f"writer expired {changed} facts for one action; run halted for this entity"})
                    entry["stopped"] = "blast radius exceeded one fact per action"
                    budget = 0
                    break
                taken += changed
                budget -= 1
                entry["actions"].append({
                    "kind": action["kind"], "loser_fact": action["loser_fact"][:90],
                    "reason": action["reason"], "applied": changed > 0,
                    "changed": changed, "error": result.get("error"),
                    "skipped": result.get("skipped"),
                })
            entry["taken"] = sum(a["changed"] for a in entry["actions"] if a.get("applied"))
            # Re-measure the pair count with the same detector that produced
            # `pairs_before`, after the writes. Only where something was applied:
            # a pass that wrote nothing changed nothing, and re-planning every
            # entity would double the pass for a number it can predict.
            entry["pairs_after"] = (plan_entity(entity).get("pairs_before", 0)
                                    if entry["taken"] else entry["pairs_before"])
        else:
            # A plan-mode pass that reports only a count is not a plan. List
            # what it would do, with the reason, so an operator can read the
            # judgement before trusting it with `apply`.
            entry["actions"] = [{"kind": a["kind"], "loser_fact": a["loser_fact"][:90],
                                 "reason": a["reason"], "planned": True, "applied": False}
                                for a in plan["actions"]]
            # Nothing was written, so nothing moved. Stated rather than absent,
            # because a missing before/after field is how a flat metric first
            # looked like an unchanged tree.
            entry["pairs_after"] = entry["pairs_before"]
        entry["before_active"] = plan.get("before_active", -1)
        entry["after_active"] = _active_count(entity) if apply else entry["before_active"]
        per_entity.append(entry)

    after = _active_count()
    record_obj = {
        "ran_at": now_iso,
        "apply": bool(apply),
        "sources": list(sources) if not entities else ["explicit"],
        "days": days,
        "signals": len(signals),
        # The size of the pool `signals` was selected from (#699). Without it a
        # record's `actions_planned: 0` reads as a verdict on the tree when it
        # is a verdict on a slice; with it, scanned-over-total is computable
        # from this JSON alone. None = drift was not consulted (explicit
        # entities, or `--sources corrections`).
        "drift_candidates_total": tally["drift_candidates_total"],
        "entities": [s["entity"] for s in signals],
        "actions_planned": planned,
        "actions_taken": taken,
        "before_active": before,
        "after_active": after,
        "delta_active": (after - before) if after >= 0 and before >= 0 else None,
        "per_entity": per_entity,
        # The pass's acceptance number: contradiction+near-duplicate pairs over
        # the entities it scanned, before and after its own writes. A retired
        # superseded claim takes a pair with it, so this moves when the loop
        # works — which is exactly what `fact_entity_recall`, kept below for
        # continuity, does not: see the comment on `pairs_before` in plan_entity.
        "pairs_before": sum(e.get("pairs_before", 0) for e in per_entity),
        "pairs_after": sum(e.get("pairs_after", 0) for e in per_entity),
        "fact_entity_recall": _fact_entity_recall(eval_limit) if report_eval else None,
        # Name the file the signals came from, not the one that was compiled in
        # first. Before this the field was `str(CORRECTIONS_PATH)` unconditionally,
        # so a record that had read USER.md still reported `memory/corrections.md`
        # and a reader could not tell a live log from a four-month-old one.
        "corrections_path": (last_corrections_read()["paths_yielding_signals"][0]
                             if last_corrections_read()["paths_yielding_signals"]
                             else None),
        "corrections_paths_read": sorted(last_corrections_read()["sources"]),
        "corrections_status": last_corrections_read()["status"],
        "corrections_sources": {p: {"status": i["status"], "entries": i["entries"],
                                    "in_window": i["in_window"],
                                    "outside_window": i["outside_window"],
                                    "undated": i["undated"], "signals": i["signals"]}
                                for p, i in last_corrections_read()["sources"].items()},
    }
    if record:
        try:
            RECORD_DIR.mkdir(parents=True, exist_ok=True)
            path = RECORD_DIR / f"{started:%Y%m%d-%H%M%S}-{'apply' if apply else 'dryrun'}.json"
            path.write_text(json.dumps(record_obj, indent=2, default=str), encoding="utf-8")
            record_obj["record_path"] = str(path)
        except OSError as exc:
            record_obj["record_error"] = str(exc)

    logger.info(
        "improve(%s): signals=%d drift_candidates=%s planned=%d taken=%d active_facts %d -> %d%s",
        "apply" if apply else "dry-run", len(signals), tally["drift_candidates_total"],
        planned, taken, before, after,
        "" if record_obj.get("fact_entity_recall") is None
        else f" fact_entity_recall={record_obj['fact_entity_recall']}")
    return record_obj
