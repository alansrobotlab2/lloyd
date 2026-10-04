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

  confidence    the two sides' confidence differs by ≥ MIN_CONFIDENCE_GAP *and*
                the higher side is attributable — carries `source_doc` or
                `created_at`. A smaller gap is extraction noise, and on
                extracted facts confidence encodes capture kind, not truth
                (#701), so the 0.9 < 0.95 pair condemns nothing; and a row
                carrying neither field beating one that carries either is the
                extractor's default `1.0` beating a sourced claim (#1348), so
                that pair is reported too, never acted on.
                *Attributable is not the same as independent*, and only one of
                the two was ever checked: a pair whose sides carry the SAME
                non-empty `source_doc`, or the same non-empty `source_hash`, is
                one document restated twice, and one document is one reason —
                not two that happen to disagree (#1941). Such a pair is flagged
                `shared_lineage` on the pair itself, counted in the record's
                `lineage_withheld_pairs`, and never acted on. This veto is on
                the confidence basis ONLY: on the `created_at` basis a later
                write inside the same document is the whole evidentiary value of
                write order, and vetoing it there is vetoing the basis.
  created_at    one was written ≥ MIN_STALE_GAP_DAYS after the other, so the
                older one is the one a later write superseded → expire it.
                *Write order alone is not a basis* (#2078): the pair also has to
                name one identifier-shaped PREDICATE TOKEN — `/rtx/dldenoiser/
                responsiveDenoising`, `sim.has_gui` — in BOTH facts, because
                otherwise the only evidence that the two rows are about the same
                thing is the keyword opposition that already paired them, which
                is #701's screen being read as an authority. A pair that clears
                the age test and fails the predicate test is reported in the
                plan's and the record's `keyword_only_flags`, never acted on.
                *And the shared token has to be a predicate, not the SUBJECT*
                (#2199): a fragment cut out of a longer identifier names nothing,
                and a token that is the entity's own name — or a prefix of it, or
                of a known alias for it — is what every pair in that entity's
                file shares anyway. The `reason` then names EVERY token that
                survives, not the alphabetically-first one.
  same day, same confidence → no basis. Reported, never acted on.

Each admitted action also names the detector's own trigger — the
`opposing_terms:<a>/<b>` the pair was classified by — so a reviewer reads the
specific pair that fired and can reject it (#701).

Writers are the existing tools, called as functions. This module never edits a
fact file: three writers of `expired_at` would be the same drift bug
`fact_profile` and the router already had (agent_mcp/facts.py:71-82), and
`fact_*` are the documented owners.

Dry-run by default. `apply=True` is opt-in, and each run writes a record under
`_pipeline/improvement/` with before/after active-fact counts so a run can be
audited or its reasoning re-read afterwards. That record also names the fact
tree, the store file and the code commit it acted on, and flags itself
`isolated` when those are not the production locations — without them, a
verification pass against a redirected copy and a real expiration are
byte-shape identical in the same directory (#700).
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from app import paths as _paths
from app.gitinfo import head_commit as _head_commit
from app.paths import LLOYD_HOME, PIPELINE_DIR, VAULT_FACTS_ROOT, VAULT_ROOT
from agent_mcp._shared import _find_entity_dir
from agent_mcp.facts import (
    _apply_fact_marks,
    _CONTRADICTION_TRACE,
    _contradiction_trace,
    _detect_contradictions_sync,
)
from agent_mcp.retrieval import get_facts_sync as _get_facts_sync
from app.kg_store import StoreUnavailable as _StoreUnavailable, store as _store

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
RECORD_DIR = PIPELINE_DIR / "improvement"

# Entity dirs holding a fact whose own `created_at` is inside this window count
# as "a writer has been here recently, and that fresh claim is the one most
# likely to already be stale". Three days: the nightly extractor runs daily, so
# one cycle of slack.
DRIFT_WINDOW_DAYS = 3
# A drift pool this large a share of the entity dirs is not a recency slice, it
# is the corpus (#1461: 11,806 of 11,807 on 2026-09-25, off file mtimes the
# nightly rebuild resets). The run still scans its newest-created `limit`, but
# it says `sweep` beside the counts rather than letting `of 11806` read as a
# selection.
DRIFT_SWEEP_FRACTION = 0.9
# The one timestamp the drift signal reads: a fact row's own `created_at`, as
# the fact markdown's front matter carries it (`  created_at: '2026-09-23T…'`,
# one per row). Read by regex over the front matter rather than through YAML:
# the walk covers every fact file in the tree each run, and a YAML parse of
# ~31k files is the difference between a second and a minute.
_CREATED_AT_RE = re.compile(
    r"""^[ \t-]*created_at:[ \t]*['"]?([0-9][^'"\n#]*?)['"]?[ \t]*$""", re.M)
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
# The smallest confidence difference that counts as evidence (#701). On this
# corpus a fact's confidence records *how it was captured*, not how true it is:
# vault-maintenance notes carry 1.0 and extracted content 0.9/0.95, so a 0.05
# gap is extraction noise — and the 2026-09-08 dry run condemned facts on
# exactly `0.9 < 0.95` and `0.85 < 0.9`, the two smallest gaps on the board.
# The pair worth acting on is the one where one side was confidently asserted
# and the other was barely believed.
#
# This floors the CONFIDENCE basis only. The equal-confidence path rests on a
# different basis — write order, via `_loser_by_age` and MIN_STALE_GAP_DAYS —
# and a pair with identical confidences reaches it exactly as it did before.
MIN_CONFIDENCE_GAP = 0.1
# Sub-trillion slack so the floor is `>=` and not `>`. Confidence values are one-
# and two-decimal, and `1.0 - 0.9` computes to 0.09999999999999998, which is
# below the constant it is compared against. No confidence on this corpus is
# specified finer than three decimals, so this cannot let a 0.0999999 pair
# through; it only undoes representation error.
_GAP_TOLERANCE = 1e-9
# The shape that makes an equal-confidence pair's age worth acting on (#2078):
# one identifier-shaped PREDICATE token that BOTH facts name. A fact that names
# a predicate — `/rtx/dldenoiser/responsiveDenoising`, `sim.has_gui` — is a
# claim about one named thing, so "the later write supersedes it" is a claim
# about that thing. A fact that names no predicate can only be paired by its
# wording, and wording is what `opposing_terms` already read: keyword opposition
# plus `created_at` order is the detector agreeing with itself, which is #701's
# ruling ("a required screen, not an authority").
#
# The shape is deliberately narrow: a SLASH PATH (`rtx/dldenoiser/
# responsiveDenoising`, `agent_mcp/fact_improvement.py`) or a DOTTED IDENTIFIER
# (`sim.has_gui`, `voice.log`), over `[A-Za-z0-9_]` segments. Hyphens are not
# joiners, on purpose — `connection-refused`, `read-only`, `state-of-the-art`
# are English compounds, and counting them is the "shared word is too loose"
# failure: both of the live false witnesses say "TTS", and a shared word would
# have paired them again. URLs are stripped before the read, so a shared
# citation cannot become a basis either — co-naming one source document is
# #1941's ONE reason, not a predicate.
#
# But "not a joiner" is not the same as "not a cut", and the difference was the
# hole (#2199): the pattern matches the part of `anthropic.claude-code` BEFORE
# the hyphen and returns it as if it were a whole identifier. `_predicate_tokens`
# drops such a truncated match — a fragment of one identifier names no predicate.
_URL_RE = re.compile(r"""\b(?:https?|ftp)://\S+""", re.I)
_PREDICATE_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:/|\.)[A-Za-z0-9_]+)+")
#: Shortest token counted. Below this a `.` or `/` in a sentence is more likely
#: punctuation than part of a name.
_MIN_PREDICATE_TOKEN_LEN = 4
# The two fields that name WHERE a row came from, checked for agreement across a
# pair (#1941). Both are per-record fields on the fact markdown itself, read
# through the same YAML parse that produced the rows the plan loop is holding —
# verified on the live tree 2026-10-01: two rows of
# `_pipeline/vault-derived/facts/LLM Inference/LLM Inference-general.md` carry
# identical `source_hash` values, so the veto needs no new read. (The hash is
# quoted as a field name, not a value, because a bare hex run in a comment reads
# as a commit sha to anything that validates citations against this repo, where
# no such object exists.)
#
# Both are asked because neither is a superset of the other: the same document
# can be cited by two paths, giving two `source_doc` strings over one
# `source_hash`, and `source_hash` is missing on rows written before a writer
# stamped one (120,572 of the 121,112 active rows carry it, against 121,103
# carrying `source_doc`). That near-universality is what makes this veto
# reachable — 19,933 (entity, `source_doc`) groups hold ≥2 rows — and it is the
# same fact that makes the *presence* of these fields useless as a reason: the
# #1348 guard asks only whether a row has one, and on 2026-10-01 every active
# row does, because `created_at` is on 121,112 of 121,112.
_LINEAGE_FIELDS = ("source_doc", "source_hash")
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

def _store_probe() -> dict:
    """Whether the knowledge-graph store answers, from ONE probe.

    #1383: `_known_entities` and `_active_count` convert every store failure
    into the `{}` / `-1` sentinels by design — a missing count must never read
    as a zero — so `run_improvement` never sees the exception, and the
    2026-09-23 nightly ran with no `kg.sqlite` at all, printed
    `active -1 -> -1 (delta None)`, wrote a record saying
    `corrections_status: registry_unreadable`, and exited 0 as if the graph
    had been read and found clean. The sentinels stay; what must not stay
    silent is the verdict. The probe performs the same read `_active_count`
    makes — open the store, count active facts — so it describes the store
    this pass actually consults, and its error text names the failing class
    and the path it could not open (`StoreUnavailable: no knowledge-graph
    database at …`). Returns both keys always; a pass can never be silent
    about whether the store was seen.
    """
    try:
        _store().facts_idx.count(entity=None, active_only=True)
    except Exception as exc:  # noqa: BLE001 - StoreUnavailable and everything the driver raises
        return {"store_ok": False, "store_error": f"{type(exc).__name__}: {exc}"}
    return {"store_ok": True, "store_error": None}


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
_NOT_READ: dict = {"status": "not_read", "sources": {}, "paths_yielding_signals": [],
                   "window_days": None, "stale_since": None, "newest_entry": None}
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

    This function **is** the corrections → fact quality route, and the only one
    that exists: `collect_signals` puts what it returns first in the signal
    union (a user saying "that was wrong" is a better reason to look than a file
    being new), `run_improvement` scans exactly those entities for superseded and
    out-ranked claims, and the `improve` tool defaults to
    `sources=("corrections", "drift")`. The prose here used to claim the
    opposite — that `memory/corrections.md` fed only the behaviour/prompt loops
    and that no path into fact quality existed — while this function underneath
    it was that path. Prose that denies its own call graph is how the channel
    stayed unnoticed while it spent two entity slots every night on headings
    last written 2026-05-08 (#802).

    Recency is what makes a correction evidence rather than history, so
    `window_days` bounds it: `CORRECTIONS_WINDOW_DAYS` by default, threaded
    through from `collect_signals(corrections_days=…)` and
    `run_improvement(corrections_days=…)`. An entry older than the window names
    no entity but is counted in `outside_window`, and so is an undated one: a
    correction that cannot say when it happened cannot say whether it still
    applies. When that filter empties a log, `last_corrections_read()` reports
    `no_entries_in_window` with the log's newest entry date, so a stale log is
    never reported as a quiet week.

    Reads every path in `corrections_paths()` — the compiled-in
    `memory/corrections.md` *and* `lloyd/USER.md`'s `## corrections_log`, which
    is where corrections have actually been written since the compiled-in file
    stopped being. An entry is credited to an entity only when a token in it is
    a registered entity name; dates and prose are stripped first, so
    "TTS service status" yields `TTS` and nothing else does.

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
        # Newest parseable entry date, window included. The stale marker is
        # "stale since <this date>", which cannot be recovered from a count:
        # `outside_window: 10` says nothing about when the log went quiet, and
        # the date is the part an operator can act on.
        newest: str | None = None
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
            if newest is None or date > newest:      # ISO dates sort as strings
                newest = date
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
                           "undated": undated, "signals": signals_here,
                           "newest_entry": newest}
    global _LAST_CORRECTIONS_READ
    if not known and not out:
        # No registry means every token is a guess. Say the store is why the read
        # produced nothing, rather than reporting a log that is fine as empty.
        for info in read.values():
            if info["status"] in ("entries", "entries_no_entity", "empty",
                                  "no_entries_in_window"):
                info["status"] = "registry_unreadable"
                info["why"] = "entity registry unreadable; no token can be resolved"
    # The stale marker is computed after the registry fix-up above: a pass that
    # could not resolve any entity name did not establish that the log is stale,
    # it established that it could not read, and reporting a date in the same
    # breath would dress the second failure up as the first.
    stale_dates = [info["newest_entry"] for info in read.values()
                   if info["status"] == "no_entries_in_window" and info["newest_entry"]]
    _LAST_CORRECTIONS_READ = {"sources": read,
                              "paths_yielding_signals": sorted(
                                  {s["corrections_path"] for s in out}),
                              "status": _roll_up(read, bool(out), known),
                              # The window THIS read used, so a run that threaded
                              # its own still reports the one that filtered.
                              "window_days": window_days,
                              # Newest entry in a log whose every entry fell
                              # outside the window, else None. This is the half
                              # that makes "0 corrections" and "the log has been
                              # stale since 2026-05-08" different records.
                              "stale_since": max(stale_dates) if stale_dates else None,
                              "newest_entry": max(
                                  (i["newest_entry"] for i in read.values()
                                   if i["newest_entry"]), default=None)}
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
    undated, signals, newest_entry}}, paths_yielding_signals, window_days,
    stale_since, newest_entry}`. The run record stores `corrections_path` from
    `paths_yielding_signals`, `corrections_window_days` from `window_days`, and
    `corrections_stale_since` from `stale_since`, so a zero in the record can be
    told apart from an unread file and from a dead log — the distinction the
    compiled-in constant destroyed by naming itself whether or not it had been
    read. `stale_since` is the newest entry of a log whose every entry fell
    outside the window, which is what makes "no corrections tonight" and "the
    corrections log last heard from the operator on 2026-05-08" two records.

    Before any read has happened it reports `not_read`: a pass that consulted no
    log has to be able to say so, and it must not inherit the shape of a pass
    that read one and found nothing.
    """
    return dict(_LAST_CORRECTIONS_READ) if _LAST_CORRECTIONS_READ else dict(_NOT_READ)


def _newest_fact_created(entity_dir: Path) -> datetime.datetime | None:
    """The newest per-fact `created_at` in one entity dir's fact files, or None.

    Front matter only: the body is rendered prose and names no timestamps, but
    a stray `created_at:` line in it must not become evidence either.
    """
    newest: datetime.datetime | None = None
    for path in entity_dir.glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        front = text[3:end] if end != -1 else text[3:]
        for m in _CREATED_AT_RE.finditer(front):
            when = _iso(m.group(1))
            if when is not None and (newest is None or when > newest):
                newest = when
    return newest


def _drift_scan(days: int = DRIFT_WINDOW_DAYS,
                now: datetime.datetime | None = None) -> tuple[list[dict], dict]:
    """(`_drift_candidates`, census of the tree it was selected from).

    The census is `{corpus, undated, newest_fact, pool, fraction, status,
    window_days}`: `corpus` is every entity dir walked, `undated` those whose
    fact rows carry no parseable `created_at` (never candidates — a row that
    cannot say when it was written cannot say it is recent), `newest_fact` the
    newest `created_at` anywhere in the tree, and `status` one of

      slice         the pool is a strict, discriminating subset of the corpus
      sweep         pool >= DRIFT_SWEEP_FRACTION of the corpus: a rebuild
                    re-stamped the tree, so the window selects everything
      stale         nothing was created inside the window — `newest_fact` says
                    since when (the drift twin of `corrections_stale_since`)
      empty_corpus  no entity dirs at all
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=days)
    census = {"corpus": 0, "undated": 0, "newest_fact": None, "pool": 0,
              "fraction": None, "status": "empty_corpus", "window_days": days}
    try:
        entries = list(FACTS_ROOT.iterdir())
    except OSError:
        return [], census
    newest_all: datetime.datetime | None = None
    ranked: list[tuple[datetime.datetime, str]] = []
    for child in entries:
        try:
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
        except OSError:
            continue
        census["corpus"] += 1
        newest = _newest_fact_created(child)
        if newest is None:
            census["undated"] += 1
            continue
        if newest_all is None or newest > newest_all:
            newest_all = newest
        if newest < cutoff:
            continue
        ranked.append((newest, child.name))
    ranked.sort(key=lambda pair: (-pair[0].timestamp(), pair[1]))
    census["pool"] = len(ranked)
    census["newest_fact"] = newest_all.isoformat() if newest_all else None
    if census["corpus"]:
        census["fraction"] = round(len(ranked) / census["corpus"], 4)
        census["status"] = ("stale" if not ranked else
                            "sweep" if len(ranked) >= DRIFT_SWEEP_FRACTION * census["corpus"]
                            else "slice")
    candidates = [{"entity": name, "source": "drift",
                   "evidence": ("fact created "
                                f"{newest.astimezone(datetime.timezone.utc):%Y-%m-%d %H:%M}Z")}
                  for newest, name in ranked]
    return candidates, census


def _drift_candidates(days: int = DRIFT_WINDOW_DAYS) -> list[dict]:
    """Every entity with a fact row created inside `days`, newest first.

    This is the whole ranked candidate pool, untruncated, and it exists as a
    function of its own so a run can name the denominator it selected from:
    `read_drift_signals` returns a `limit`-sized prefix of this list, and
    #699's complaint was that nothing in the record said what the prefix was a
    prefix *of*. `_drift_scan` returns the same list with the census that says
    what the pool is a share of.

    Per-fact `created_at`, not file or directory mtimes (#1461). The nightly
    rebuild rewrites the whole fact tree, so every file's mtime sat inside a
    3-day window by construction: on 2026-09-25 11,806 of 11,807 entity dirs
    were "drift", the pool was the corpus, and the 40 scanned were whichever
    dirs the rebuild wrote last. A row's `created_at` moves only when a writer
    adds that row. A full re-extraction still re-stamps every row — then the
    pool really is the corpus, and the census names it a `sweep` rather than
    letting it pass as a selection. The walk reads the fact files' front
    matter by regex, not the store's `facts_idx`, so the signal follows
    `FACTS_ROOT` wherever a test or a rebuild points it and needs no store.

    Order is newest `created_at` first, ties broken by name so the ranking is
    reproducible across runs. Sorting by name and truncating — which is what
    this did until #699 — made the nightly `--limit 40` the ASCII-earliest 40 of
    the drifted pool (1,594 entities measured 2026-09-15), the same 40 on three
    consecutive nights (records 20260912-210116, 20260913-210013 and
    20260914-210026 are set-identical), and that drift slice shared 0 of 38
    entities with the 38 most-recently-written ones.
    """
    return _drift_scan(days)[0]


def read_drift_signals(days: int = DRIFT_WINDOW_DAYS, limit: int = 50) -> list[dict]:
    """The `limit` entities whose newest fact row was created most recently.

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
                     limit: int = 40,
                     corrections_days: int = CORRECTIONS_WINDOW_DAYS,
                     ) -> tuple[list[dict], dict]:
    """`collect_signals` plus what it selected from.

    The tally is read off the same tree walk that produced the slice, so the
    denominator cannot disagree with the entities the run actually scanned.
    `drift_candidates_total` is None when drift was not consulted at all — a
    run over `--sources corrections` did not measure a drift population, and 0
    would be a number it never measured.

    `days` windows drift and `corrections_days` windows corrections. They were
    once one argument threaded to the drift branch alone, which is why a
    five-month-old correction outranked every fresh write: the corrections
    branch received no window at all (#802).
    """
    wanted = set(sources)
    out: list[dict] = []
    seen: set[str] = set()
    drift_candidates_total: int | None = None
    census: dict | None = None
    if "corrections" in wanted:
        out.extend(read_correction_signals(limit=limit, window_days=corrections_days))
    if "drift" in wanted:
        candidates, census = _drift_scan(days)
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
    return deduped, {"drift_candidates_total": drift_candidates_total,
                     **_drift_tally(census)}


def _drift_tally(census: dict | None) -> dict:
    """The drift census as run-record keys; every value None when drift was not
    consulted, because a run that walked no tree measured no corpus (#1461)."""
    census = census or {}
    return {"drift_corpus_total": census.get("corpus"),
            "drift_pool_fraction": census.get("fraction"),
            "drift_status": census.get("status"),
            "drift_newest_fact": census.get("newest_fact"),
            "drift_undated_dirs": census.get("undated"),
            "drift_window_days": census.get("window_days")}


def collect_signals(sources=("corrections", "drift"), days: int = DRIFT_WINDOW_DAYS,
                    limit: int = 40,
                    corrections_days: int = CORRECTIONS_WINDOW_DAYS) -> list[dict]:
    """Union of the enabled sources, deduped by entity, corrections first.

    Corrections outrank drift because a user saying "that was wrong" is a
    better reason to look than a file being new — and drift is ranked
    newest-write-first within its own block (see `read_drift_signals`).

    Two windows, one per source: `days` bounds drift writes and
    `corrections_days` bounds correction entries (default
    `CORRECTIONS_WINDOW_DAYS`) and is passed to `read_correction_signals`, which
    is the only thing that decides whether an old log contributes entities.
    """
    return _collect_signals(sources=sources, days=days, limit=limit,
                            corrections_days=corrections_days)[0]


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


def _has_attribution(fact: dict) -> bool:
    """Whether one facts-view row can say where it came from.

    Read with `.get()` and nothing else. `_read_facts_cached` returns the parsed
    YAML entries verbatim, so a fact whose file never carried `created_at` or
    `source_doc` is a dict with NO such key rather than one carrying `None`: the
    2026-09-21 `pass@k` winner row (#1348) came back with keys `['category',
    'confidence', 'entity', 'event_date', 'fact', 'id', 'provenance',
    'source_file']` while its loser's carried both. `fact["created_at"]` raises
    `KeyError` on the winner — the one row this guard exists to catch — and
    `provenance` is no substitute for the field: both rows read `'EXTRACTED'`,
    which is why the guard tests attribution and not provenance.
    """
    return bool(fact.get("source_doc")) or bool(fact.get("created_at"))


def _unattributed_winner(winner: dict, loser: dict) -> bool:
    """True when the higher-confidence row carries no evidence the lower one does.

    Asymmetric on purpose, because the unattributed side is the majority class:
    measured through `app.kg_store` on 2026-09-21, 204,706 of 321,252 active
    rows carry neither `source_doc` nor `created_at`, and 162,175 of those sit at
    confidence >= 0.95. A rule barring an unattributed row from winning at all
    would suppress most of the actions this loop can take — the pass would report
    every entity and plan nothing, which reads as a healthy night. Two rows
    nobody can source are still decided on confidence, the only signal left to
    decide on. What is refused here is the one-sided contest: a bare `1.0` with
    no date and no source document demoting a claim that can name either.
    """
    return not _has_attribution(winner) and _has_attribution(loser)


def _shared_lineage(f1: dict, f2: dict) -> str | None:
    """Which field names ONE origin for both rows of a pair, or None.

    The question `_has_attribution` never asked (#1941). That predicate is about
    whether a row can say where it came from; this one is about whether the two
    rows came from *different* places. A pair sourced to the same document on
    both sides is that document said twice, and one document is one reason — it
    cannot be the independent second reason this pass's whole thesis requires
    ("every action here therefore needs an independent reason *on top of* the
    detector's pairing", module docstring). Confidence is what makes the pair
    look decidable: the witness on the live tree is entity `LLM Inference`, two
    rows both sourced to
    `projects/lloyd/channel-eval/ai-engineer.md`, at 0.95 against 0.85 — a gap
    of exactly `MIN_CONFIDENCE_GAP`, cleared only because `_GAP_TOLERANCE` makes
    the floor `>=`.

    Returns the field name that agreed, not just a bool, so a reader of the pair
    can tell a shared document from a shared content hash. Compares stripped
    strings and refuses to call two empty values shared: `source_doc: None` on
    both sides says nothing about where either came from, which is the #1348
    shape and is already refused on its own grounds — treating "neither knows"
    as "same source" would silently swallow the unattributed-winner guard and
    every row written without attribution.
    """
    for field in _LINEAGE_FIELDS:
        v1, v2 = str(f1.get(field) or "").strip(), str(f2.get(field) or "").strip()
        if v1 and v2 and v1 == v2:
            return field
    return None


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


def _predicate_tokens(text: str) -> set[str]:
    """The identifier-shaped tokens one fact's own wording names (#2078).

    Lowercased, because the same flag reaches a fact as `/RTX/DLDenoiser/…` in
    one note and `responsiveDenoising` in another, and the basis has to survive
    the casing. URLs go first: two facts citing one page share
    `github.com/isaac-sim/…`, which is a shared source, not a shared predicate.

    A match that STOPS inside a longer identifier is not a token (#2199).
    Hyphens are not joiners on purpose, but that also cuts `anthropic.claude-code`
    at its hyphen and hands back `anthropic.claude` — a FRAGMENT of one
    identifier, which names no predicate. On the live corpus the fragment named
    everything: it is a prefix of the extension id the pair's entity is written
    by, so any two facts about that extension co-named it and cleared #2078's bar
    for free. The character the match ends on tells a fragment from a token —
    prose (a space, a comma, a sentence period, a backtick) follows a token, and
    the hyphen it was cut at follows a fragment.
    """
    stripped = _URL_RE.sub(" ", text or "")
    out: set[str] = set()
    for m in _PREDICATE_TOKEN_RE.finditer(stripped):
        tok = m.group(0)
        if len(tok) < _MIN_PREDICATE_TOKEN_LEN:
            continue
        if m.end() < len(stripped) and stripped[m.end()] == "-":
            continue                        # truncated mid-identifier (#2199)
        out.add(tok.lower())
    return out


def _name_form(name: str) -> str:
    """A name reduced to the characters a token can carry: `[a-z0-9]` only.

    `anthropic.claude` vs `anthropic.claude-code` vs `Claude Code` differ only in
    punctuation and case, and a prefix test on the raw strings would compare
    spellings that are the same name.
    """
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _store_alias_surfaces(entity: str) -> list[str]:
    """Every alias surface the store routes to `entity`; [] when there is no store.

    The alias table is what says `claude.code` and `anthropic.claude-code` are one
    entity written two ways, so it is the only place "a known alias for it" has a
    definition. `StoreUnavailable` is a real state on this box (#1236), and the
    answer there is no aliases rather than a crash: the guard loses its alias half
    and keeps the half it can read off the entity name it was handed.
    """
    try:
        return [r["surface"] for r in _store().aliases.for_canonical(entity)]
    except (_StoreUnavailable, OSError, sqlite3.Error):
        return []


def _entity_name_forms(entity: str | None,
                       aliases: Iterable[str] = ()) -> set[str]:
    """Every spelling of the pair's OWN subject, as `_name_form`s.

    Three sources: the entity name the caller is holding, the aliases the caller
    supplies, and every alias surface the store routes to that entity. The entity
    itself is in here because it is the case that matters: #2078's bar asks
    whether the two rows name one thing PREDICATED of a subject, and a token that
    IS the subject answers a different question — a fact in an entity's own file
    naming that entity's identifier is what every pair in that file does.
    """
    forms: set[str] = set()
    for name in (entity, *aliases):
        form = _name_form(name or "")
        if form:
            forms.add(form)
    if entity:
        for surface in _store_alias_surfaces(entity):
            form = _name_form(surface)
            if form:
                forms.add(form)
    return forms


def _names_the_entity(token: str, forms: set[str]) -> bool:
    """True when `token` is the pair's subject, not something predicated of it.

    Prefix, not equality, because the whole defect is a truncated name: a guard
    matching only the WHOLE alias would still let `anthropic.claude` through as a
    prefix of `anthropic.claude-code`. What it deliberately does not match is a
    token LONGER than the name — under an entity named `sim.has_gui`,
    `sim.has_gui.enabled` is a field of the subject, which is exactly the
    predicate the bar is asking for.
    """
    if not forms:
        return False
    form = _name_form(token)
    return bool(form) and any(form == f or f.startswith(form) for f in forms)


def _shared_predicate_reason(shared: str) -> str:
    """The witness half of an admitted action's `reason`, plural-safe.

    Single token: byte-identical to the sentence #2078 shipped. Several: every
    one is named, because the point of #2199's change is that the reviewer sees
    the whole shared set, not one member of it.
    """
    tokens = shared.split(", ")
    if len(tokens) == 1:
        return f"both facts name `{tokens[0]}`, so the later write is about that predicate"
    quoted = ", ".join(f"`{t}`" for t in tokens)
    return (f"both facts name {quoted}, so the later write is about "
            f"{len(tokens)} predicates both facts name")


def _shared_predicate(t1: str, t2: str, entity: str | None = None,
                      aliases: Iterable[str] = ()) -> str | None:
    """Every identifier-shaped predicate token BOTH texts name, else None.

    A property of the PAIR, which is the whole point (#2078). A basis keyed on
    ONE side's wording is unsafe: the live false witness's own loser reads
    "flipped `/rtx/dldenoiser/responsiveDenoising` from false to true", so it
    contains an explicit supersession statement and would authorise its own
    expiry. Two texts agreeing on one name cannot do that to each other.

    Two exclusions decide what the intersection counts (#2199), and both are
    about what a shared token is a NAME of. A fragment cut out of a longer
    identifier never enters the intersection (`_predicate_tokens`). And a token
    that is, or is a prefix of, the entity's own name or one of its known aliases
    is the SUBJECT the two facts share rather than a predicate about it: this
    scan is already entity-scoped, so co-naming the subject proves nothing the
    caller did not know, and on `Claude Code` it proved everything — the
    truncated extension id let the pair planned in
    `backlog/data/20261004-210010-dryrun.json` clear the bar on a token that was
    part of the entity's own identifier. Excluding it narrows the basis without
    silencing it: a genuine predicate both facts name about that same entity
    still stands, and is reported beside whatever else they share.

    ALL surviving tokens are named, sorted and comma-joined. `sorted(shared)[0]`
    reported `anthropic.claude` for a pair that also shared the real field
    `metadata.pinned`, so the witness a reviewer reads in `reason` was always the
    weakest candidate on the pair — #701's "name what admitted it" was implemented
    as "name one thing that admitted it", and the alphabet chose it. With exactly
    one token the returned string is that token, unchanged.
    """
    shared = _predicate_tokens(t1) & _predicate_tokens(t2)
    forms = _entity_name_forms(entity, aliases)
    shared = {t for t in shared if not _names_the_entity(t, forms)}
    return ", ".join(sorted(shared)) if shared else None


def _keyword_only_age_flags(entity: str, pairs: list[dict]) -> list[dict]:
    """Pairs whose ONLY basis for an expiry is keyword opposition plus write order.

    Reported, never acted on (#2078). A pair is flagged when three things hold:
    the two rows carry equal confidence (so the confidence branch, with its own
    floor and vetoes, was never the question), `_loser_by_age` finds an age
    basis (so this is a pair that WAS about to be expired), and
    `_shared_predicate` finds nothing naming both facts (so the age basis had
    nothing under it but the trigger string). A pair with no age basis is not
    flagged: nothing was ever going to happen to it, and calling it a withheld
    expiry would report a veto that was never exercised — the #1348 lesson about
    a zero that reads as a clean night.

    Computed over every actable pair, not over the action loop, so the figure
    does not depend on `max_actions`: an entity whose first five pairs filled
    the cap still reports the keyword-only pairs behind them.
    """
    flags: list[dict] = []
    for item in pairs:
        f1 = item.get("fact1") or {}
        f2 = item.get("fact2") or {}
        if not f1.get("fact") or not f2.get("fact"):
            continue
        # `!=` mirrors the branch below: same test, opposite side of it.
        if float(f1.get("confidence") or 0.5) != float(f2.get("confidence") or 0.5):
            continue
        ordered = _loser_by_age(f1, f2)
        if ordered is None:
            continue
        loser, winner, _ = ordered
        # `entity` so a token that is the subject itself cannot count as the
        # predicate (#2199). Same argument, same entity, as the planning site
        # below — the two must decline the SAME pairs or the flag list reports a
        # withheld expiry that the planner was about to take.
        if _shared_predicate(str(loser.get("fact") or ""),
                             str(winner.get("fact") or ""), entity):
            continue
        flags.append({
            "entity": entity,
            "trigger": str(item.get("reason") or "").strip(),
            "older_fact": loser.get("fact", ""),
            "older_created_at": loser.get("created_at"),
            "newer_fact": winner.get("fact", ""),
            "newer_created_at": winner.get("created_at"),
            "basis": ("opposing_terms keyword plus created_at order only; neither "
                      "fact names an identifier-shaped predicate token the other "
                      "also names"),
        })
    return flags


#: How a plan's pair counts were produced. `entity` is one whole-entity
#: pairwise scan, which is the only scope that can pair two facts living in
#: different category files. `by_category` is the retry below, reached only by an
#: entity the whole-entity scan refused: it covers every fact in every
#: category-sized piece, and it cannot cross a category boundary.
SCAN_ENTITY = "entity"
SCAN_BY_CATEGORY = "by_category"

#: The category a fact row lands in when it carries none. Same default
#: `facts.py::_get_all_facts` gives a row it groups for display, so the
#: partition the scan uses and the profile an operator reads agree.
_NO_CATEGORY = "general"


def _facts_by_category(facts: list[dict]) -> dict[str, list[dict]]:
    """One entity's fact rows, grouped by the `category` each row carries.

    There is no category listing to call. `fact_profile` gets its per-category
    view by grouping the rows it already fetched (`facts.py::_get_all_facts`), and
    a retry that instead enumerated `FACTS_ROOT/<entity>/*.md` would judge a
    different set of rows than the read that produced the refusal — the one
    situation in this module where the facts being planned against and the facts
    that were scanned could disagree. So the partition comes from the refusal's own
    input list, and a row with no category is grouped under `_NO_CATEGORY` exactly
    as the display path groups it.
    """
    groups: dict[str, list[dict]] = {}
    for fact in facts:
        groups.setdefault(str(fact.get("category") or _NO_CATEGORY), []).append(fact)
    return groups


def _rescan_by_category(entity: str, facts_view: list[dict]) -> dict:
    """Apply the remedy the refusal itself names, once per category.

    `_detect_contradictions_sync` ends its refusal with "Pass a narrower
    `category`" (#1251). No caller ever had: `plan_entity` scanned at the coarsest
    possible scope, took the refusal, and returned the entity with zero actions —
    so the entities this pass most wants were the ones it was structurally blind
    to. A big entity is big *across* its category files, not inside each one:
    measured on the live tree 2026-09-20, `Assistant` is 55 facts over 5 files
    (largest 26) and `code4AI` is 109 over 4 (largest 39), and both were dropped
    whole while every file in them sat under the bound.

    The bound still holds, because it is applied by the same check in the same
    place rather than re-derived here: each piece goes through the detector, which
    refuses a piece larger than `FACT_GODNODE_THRESHOLD`, so a category over the
    bound is skipped and named instead of scanned. The cost follows: the
    whole-entity scan is n² comparisons, the partition is Σn_c² ≤
    `FACT_GODNODE_THRESHOLD` × n — linear in the entity, not quadratic. `QMD`
    (3,321 live facts, 5 of its 13 categories over the bound) goes from 5.5 M
    comparisons to a few hundred, against the 15 M that took 113 seconds on
    `Lloyd` measured in `facts.py`.

    What the partition cannot do is pair a fact with one in a *different*
    category, so this is a coverage change and not only a cost change. That is why
    it runs only on the refusal path: an entity at or under the bound is still
    scanned whole and keeps its cross-category pairs, and an entity that reaches
    here had nothing scanned at all before — it loses no pair it ever had.

    Returns a detection-shaped dict so the pair-handling in `plan_entity` below is
    untouched, plus the two lists that make the result readable:
    `categories_scanned` (what the numbers cover) and `categories_skipped` (what
    they do not). `refused` is True only when nothing at all could be scanned.
    """
    groups = _facts_by_category(facts_view)
    scanned: list[str] = []
    skipped: list[str] = []
    contradictions: list[dict] = []
    checked = 0
    for category in sorted(groups):
        piece = _detect_contradictions_sync(entity, category, facts=groups[category])
        if piece.get("refused"):
            skipped.append(category)
            continue
        contradictions.extend(piece.get("contradictions", []))
        checked += piece.get("checked", 0)
        scanned.append(category)
    return {"entity": entity, "category": None, "contradictions": contradictions,
            "checked": checked, "refused": not scanned,
            "categories_scanned": scanned, "categories_skipped": skipped}


def plan_entity(entity: str, max_actions: int = MAX_ACTIONS_PER_ENTITY) -> dict:
    """Decide what — if anything — to change about one entity's facts.

    Reads through the same `_get_facts_sync` the recall path uses, so the plan
    is made on the facts a query would actually have been answered with — and
    read WITH a file attribution, because a plan that cannot name the file its
    loser lives in cannot be executed safely: fact ids are a per-file counter
    (`app.fact_ids.next_fact_id`), so one id is one fact in this view and
    several facts on disk. The detector is handed that same list instead of
    re-reading, so what gets judged and what gets written come from one read.

    Two rules decide whether a classified pair becomes an action (#701). Only a
    pair the detector classified as opposing terms is eligible —
    `REQUIRE_OPPOSING_TERMS` — and that classification now means whole-word
    agreement on one of the seven pairs, not bare substring containment. And a
    pair whose sides differ in confidence needs a difference of at least
    `MIN_CONFIDENCE_GAP` to be condemned; a smaller gap says the two facts were
    captured differently, not that one is weaker. Each action's `reason` names
    the trigger that admitted it, so a reviewer can reject the specific pair.

    A third rule (#2078, from #701's owed ruling) narrows the equal-confidence
    path: **a keyword-opposition match is a screen, never a sole authority to
    expire.** `created_at` order says only which row was written last, and the
    `opposing_terms:<a>/<b>` trigger that got the pair into this loop was read
    off the two sentences' wording, so keyword opposition plus write order is
    the detector agreeing with itself. A `superseded` action therefore needs a
    non-lexical, pair-level basis: both facts must name one identifier-shaped
    predicate token (`/rtx/dldenoiser/responsiveDenoising`, `sim.has_gui`).
    Without it the pair plans nothing and is reported in `keyword_only_flags`.
    The basis is keyed on the PAIR, not on one side's wording, because a
    one-sided "explicit supersession" test is self-authorising — the second
    witness below is itself a sentence saying something was "flipped … from
    false to true".

    Both actions this guard planned after #701 landed are false, and both are
    what the new basis rejects. `opposing_terms:success/failure` paired "TTS
    success JSON response reads were bounded in commit #96984." (2026-09-26)
    with "TTS failure during an outage results in a connection-refused error and
    leaves only a `voice.log` line." (2026-09-29, 3.4 days later) — a
    success-path claim and a failure-path claim about one subsystem, where the
    older row names no predicate at all. `opposing_terms:true/false` paired the
    Kit row recording that `/rtx/dldenoiser/responsiveDenoising` was flipped
    from false to true (2026-09-28) with "The sim.has_gui property is always
    False in Isaac Lab 3.0.0-beta2.patch1, leading to a missing IsaacLab GUI
    tab." (2026-09-30, 1.7 days later) — two unrelated subsystems, each naming
    its own identifier and neither naming the other's. Records:
    `_pipeline/improvement/20260930-210019-dryrun.json` and
    `20261001-210151-dryrun.json`; both rows are on disk under
    `_pipeline/vault-derived/facts/`, and both pairs are rebuilt as fixtures in
    `tests/test_memory_improvement.py`.

    An entity too big for one pairwise scan is not skipped: it is scanned in
    category-sized pieces, which is what the detector's own refusal tells the
    caller to do. See `_rescan_by_category` for why that is a coverage fix and not
    a cost fix, and for what it cannot see.
    """
    facts_view = _get_facts_sync(entity, with_source_file=True).get("facts", [])
    detection = _detect_contradictions_sync(entity, facts=facts_view)
    scan_scope = SCAN_ENTITY
    categories_scanned: list[str] = []
    categories_skipped: list[str] = []
    if detection.get("refused"):
        retry = _rescan_by_category(entity, facts_view)
        if not retry["categories_scanned"]:
            # Every category is itself above the bound: the godnode shape the
            # threshold exists for. The entity-level refusal stands exactly as it
            # was — nothing about this entity is scanable, so there is no partial
            # coverage to report and no scope to name beyond the refusal.
            return {"entity": entity, "refused": True, "checked": detection.get("checked", 0),
                    "contradictions": 0, "actions": [],
                    "scan_scope": SCAN_ENTITY,
                    "categories_scanned": [], "categories_skipped": [],
                    "before_active": _active_count(entity),
                    "skipped_reason": detection.get("hint", "entity too large to scan")}
        # Some part of the entity was scanned. `refused` stays False for it and
        # `categories_skipped` names what was not, so a partial scan reads as
        # partial — never as the whole-entity refusal it replaced, and never as a
        # complete scan whose pair count is a coverage claim.
        detection = retry
        scan_scope = SCAN_BY_CATEGORY
        categories_scanned = retry["categories_scanned"]
        categories_skipped = retry["categories_skipped"]

    contradictions = detection.get("contradictions", [])
    # The lineage flag goes on EVERY pair the detector returned, before the
    # opposing-terms filter runs (#1941): "these two rows are one document
    # restated" is a property of the pair, and a pair the filter sets aside has
    # it as plainly as one it keeps. Emitting it before the veto is what keeps
    # the two halves of clause 5 apart — `lineage_pairs` is exposure, and
    # `lineage_withheld` is what the veto cost. A withheld count of 0 means
    # nothing until the exposure beside it says whether there was anything to
    # withhold.
    for pair in contradictions:
        if _shared_lineage(pair.get("fact1") or {}, pair.get("fact2") or {}):
            pair["shared_lineage"] = True
    lineage_pairs = sum(1 for c in contradictions if c.get("shared_lineage"))
    lineage_withheld = 0
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
        # The detector's own classification of THIS pair, verbatim:
        # `opposing_terms:working/broken` (or `high_overlap_potential_update`
        # when `REQUIRE_OPPOSING_TERMS` is off). Carried into the action's
        # reason (#701): the 2026-09-15 planned actions read only
        # "contradiction detector paired it with '…'; confidence 0.9 < 1.0", so
        # nothing on the page said which of the seven opposing pairs fired, and
        # a reviewer could not reject the one pair that was wrong.
        trigger = str(item.get("reason") or "").strip()
        c1 = float(f1.get("confidence") or 0.5)
        c2 = float(f2.get("confidence") or 0.5)
        if c1 != c2:
            # `+ _GAP_TOLERANCE`: confidences here are one- and two-decimal
            # numbers, and `1.0 - 0.9` is 0.09999999999999998 in binary floating
            # point. Without it a gap of exactly the floor is declined —
            # including the 2026-09-15 working/broken hit at 0.9 against 1.0 —
            # so the constant would enforce "strictly more than 0.1" while
            # reading as "at least 0.1". The floor means what it says.
            if abs(c1 - c2) + _GAP_TOLERANCE < MIN_CONFIDENCE_GAP:
                # 0.9 vs 0.95 is two facts captured two ways, not two claims
                # of differing strength. No basis, so no action — the pair
                # stays in `contradictions` and is reported (#701).
                continue
            if item.get("shared_lineage"):
                # One document, two rows → no action, pair still reported
                # (#1941). Placed after the gap floor because the floor is what
                # the clause's "differ by >= MIN_CONFIDENCE_GAP" refers to, and
                # before the #1348 contest because the lineage question is the
                # earlier one and does not depend on which side holds the higher
                # number. The two refusals are disjoint on the `source_doc` half:
                # two rows sharing a non-empty `source_doc` makes the winner
                # attributed by that fact alone, so `_unattributed_winner` is
                # False for any pair this vetoes.
                lineage_withheld += 1
                continue
            loser, winner = (f2, f1) if c1 > c2 else (f1, f2)
            if _unattributed_winner(winner, loser):
                # #1348. The winner's only advantage is a number, and the
                # loser's is a file or a timestamp. On 2026-09-21 the pass's one
                # planned action demoted `stat-009` — conf 0.9, `created_at`
                # 2026-09-21T17:50:12Z, sourced to
                # `knowledge/youtube/AI_Engineer/20260820-your-agent-evolved-
                # your-evals-didnt-ameya-bhatawdekar-braintrust.md` — to a row at
                # conf 1.0 whose `created_at` and `source_doc` are both NULL. A
                # provenance-free row cannot win that contest on the bare
                # comparison, so the pair yields no action and stays in
                # `contradictions`, exactly as a sub-floor gap does (#701): "no
                # basis to act" is not a verdict that the pair is agreeable.
                continue
            kind = "confidence"
            reason = (f"{trigger}; contradiction detector paired it with "
                      f"{winner.get('fact', '')[:70]!r}; confidence "
                      f"{loser.get('confidence')} < {winner.get('confidence')}")
        else:
            ordered = _loser_by_age(f1, f2)
            if ordered is None:
                continue          # equal confidence, no age basis → leave it
            loser, winner, age_reason = ordered
            # #2078: a keyword-opposition match is a screen, never a sole
            # authority to expire. Everything above this line is a property of
            # the pair's WORDING (`opposing_terms:<a>/<b>`) or of the fact
            # layer's clock, and the two false witnesses this guard was filed
            # from both passed all of it: the TTS pair's older row names no
            # identifier at all, and the Kit/Isaac pair names two DIFFERENT
            # ones (`responsiveDenoising` against `sim.has_gui`, two unrelated
            # subsystems). So the age basis needs something under it that is
            # not the pair's wording: one identifier-shaped predicate token
            # both facts name. Absent it, no action — and the pair is reported
            # in `keyword_only_flags`, where `_keyword_only_age_flags` counts
            # exactly the pairs declining here.
            #
            # `entity` is what makes the token a PREDICATE rather than the
            # subject re-named (#2199): a shared token that is the entity's own
            # name, or a prefix of it or of a known alias, proves only that both
            # rows live in that entity's file.
            shared = _shared_predicate(str(loser.get("fact") or ""),
                                       str(winner.get("fact") or ""), entity)
            if shared is None:
                continue
            kind = "superseded"
            # Same rule on this basis: the trigger that admitted the pair is
            # part of the record, and `MIN_CONFIDENCE_GAP` does not apply here
            # — this pair's basis is write order, not a confidence gap. The
            # token that makes the write order about one named thing is named
            # beside it (#701's rule that an admitted action says what admitted
            # it, now applied to the non-lexical half of the basis too), so a
            # reviewer can reject the co-naming without re-reading both files.
            # EVERY token that survives the entity-name exclusion is named, not
            # one of them: a witness that quietly reported the alphabetically-
            # first member hid the real field behind the entity fragment (#2199).
            reason = f"{trigger}; {age_reason}; {_shared_predicate_reason(shared)}"
        action = {"kind": kind, "entity": entity,
                  "category": loser.get("category"),
                  "loser_fact": loser.get("fact", ""), "loser_id": loser.get("id"),
                  # Where the loser is, not only what it is called. The writer
                  # marks inside this file, which is what turns "one action, one
                  # fact" from an aim the scan may overshoot into a scope. An
                  # action planned against a view that carried no attribution
                  # gets "" and the writer says so rather than guessing.
                  "loser_source_file": loser.get("source_file") or "",
                  # And the same for the WINNER, because a contradiction has two
                  # facts and the loser's own record cannot name the one it lost
                  # to. The plan step computed this record three lines above and
                  # until #1817 let it escape only as prose inside `reason`, which
                  # is why the daily writer's marks carried no
                  # `conflicts_with` field while the nightly count of that field
                  # read `0 of 119,167 fact records`: apply-time had a text and a
                  # sentence, and a trace needs a file and an id.
                  "winner_id": winner.get("id"),
                  "winner_source_file": winner.get("source_file") or "",
                  # Whole, not the 70-character excerpt `reason` carries for the
                  # reader: a trace's `fact` is what a later reader reads instead of
                  # opening the winner's file, and it must be the claim, not a cut.
                  "winner_fact": winner.get("fact", ""),
                  "reason": reason}
        # Dedupe on the fact text, not the id: ids are per-file counters, so
        # `fact-001` in three category files is three different facts and one
        # condemned claim must not silence the other two.
        key = (action["category"], _normalise(action["loser_fact"]))
        if key in planned_against:
            continue
        planned_against.add(key)
        actions.append(action)

    # The class this round narrowed, stated as a number beside the pairs it was
    # computed over (#2078). Read off `actable` — every pair the opposing-terms
    # screen let through — and not off the action loop above, so an entity whose
    # cap filled on its first `max_actions` pairs still reports the keyword-only
    # pairs that sat behind them. With `REQUIRE_OPPOSING_TERMS` on, which is the
    # only shipped setting, every entry here carries an `opposing_terms:*`
    # trigger by construction; flipping that screen off would widen this list to
    # the near-duplicate class as well, which is one more reason it stays on.
    keyword_only_flags = _keyword_only_age_flags(entity, actable)

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
            # The pairs that reached this loop with an age basis and nothing
            # else (#2078), reported beside that denominator rather than summed
            # into it: a flagged pair is still a contradiction the detector
            # found, and `pairs_before` is the loop's own denominator, so
            # removing them would move the denominator to match the guard and
            # leave nothing to compare a future run against. Each entry names
            # both facts and both timestamps, because the thing an operator has
            # to be able to check is whether the guard's judgement that these
            # two facts are about different things is right.
            "keyword_only_flags": keyword_only_flags,
            # Reported so the near-duplicate class stays visible: it is the
            # auto-capture noise this loop declines to delete, and the reason
            # the metric does not move. A subset of `pairs_before`, never added
            # to it.
            #
            # Counted off the detector's CLASSIFICATION, not off `actable`.
            # Those two sets stopped being the same in #701: an opposing-terms
            # pair can now be declined for a confidence difference below
            # `MIN_CONFIDENCE_GAP`, and such a pair is not a near-duplicate —
            # it was classified as an opposition and left alone for a reason
            # that is a property of the threshold, not of the pair. Deriving the
            # figure from `actable` would have reported the loop's own floor as
            # a count of auto-capture noise.
            "near_duplicates": sum(
                1 for c in contradictions
                if not str(c.get("reason", "")).startswith("opposing_terms")),
            # Exposure and cost, stated separately (#1941). `lineage_pairs` is
            # how many of THIS entity's pairs share an origin — the number that
            # says whether the veto below had anything to bite on.
            # `lineage_withheld` is how many of those the confidence basis was
            # about to act on and did not. The second is the one an action-rate
            # drop is read from; the first is what stops a zero in the second
            # from being read as a clean night, which is the #1348 lesson this
            # item names. Both are per-entity, and a refused entity reports
            # neither, because it was never scanned.
            "lineage_pairs": lineage_pairs, "lineage_withheld": lineage_withheld,
            "actions": actions, "before_active": _active_count(entity),
            # Coverage, stated. `pairs_before` on a `by_category` plan counts the
            # pairs inside each scanned category and nothing across one, and
            # `categories_skipped` names the files that were too large even for a
            # piece of the scan — so an entity scanned in pieces cannot be read as
            # one scanned whole, and neither can be mistaken for the whole-entity
            # refusal, which reports `refused: True` with no scope to name.
            "scan_scope": scan_scope,
            "categories_scanned": categories_scanned,
            "categories_skipped": categories_skipped}


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
    is_confidence = action["kind"] == "confidence"
    if is_confidence:
        field, reason_field = "invalid_at", "invalid_reason"
    else:
        field, reason_field = "expired_at", "expire_reason"
    # The entity travels with the action, so this seam asks the SAME question the
    # planner asked (#2199). Without it the two could disagree on one pair: the
    # plan declines a supersession whose only co-naming is the entity's own name
    # or an alias for it, and a record carrying that same pair would be written
    # here on the strength of the token the plan had just excluded.
    if not is_confidence and not _shared_predicate(
            str(action.get("loser_fact") or ""),
            str(action.get("winner_fact") or ""),
            str(action.get("entity") or "") or None):
        # #2078 at the write seam, not only the plan seam. `plan_entity` cannot
        # produce such an action any more, so this refuses the shapes that reach
        # a writer without one: a record written by an older revision and
        # hand-applied later, an action dict assembled by hand, or a plan built
        # against a tree whose guard was not yet deployed. It reads the two
        # FACT TEXTS the action itself carries rather than re-planning, so it
        # asks the same question the planner asked and cannot be talked out of it
        # by the plan that brought the action here. Fails closed on a missing
        # `winner_fact`: no second text, no co-naming, no write.
        return {"expired_count": 0,
                "skipped": ("supersession basis is keyword opposition plus write "
                            "order only — no predicate token named by both facts"),
                "field": field, "traces_written": 0, "traces": [], "untraceable": 0}
    aim = _aim_substring(action["entity"], action)
    if aim is None:
        # Same keys as every other return, so a caller summing a run's actions reads a
        # measured zero rather than a missing one it has to guess the meaning of.
        return {"expired_count": 0, "skipped": "no unique match for the condemned fact",
                "field": field, "traces_written": 0, "traces": [], "untraceable": 0}
    entity_dir = _find_entity_dir(action["entity"])
    if entity_dir is None:
        return {"expired_count": 0, "skipped": "entity directory not found"}
    # Two writers stamp contradiction-loser semantics onto `invalid_at`, and only one
    # of them left a trace: `facts._fact_resolve_apply` (the MCP tool, no automated
    # caller) marks by (file, id) and passes `extra_fields`, while this daily loop
    # marked the same field through a text match with no extras at all, so its losers
    # were unanswerable and the nightly `conflicts_with` count stayed blind to the
    # only writer that runs. A `superseded` action is a different judgment — write
    # order, not a contradiction — so it keeps `expired_at`/`expire_reason` and gains
    # no trace. #1817 closes the gap by making the confidence route aim by identity,
    # the route that can carry one.
    # The attribution is relative to FACTS_ROOT (`Entity/Entity-usage.md`), so
    # it resolves against the root — joining it to the entity dir would name a
    # path one level too deep and match nothing.
    scope = ([FACTS_ROOT / action["loser_source_file"]]
             if action.get("loser_source_file")
             else list(entity_dir.glob("*.md")))
    # #1817: a `confidence` action whose loser carries an attribution is aimed by
    # (file, id) — the one route `_apply_fact_marks` will attach an extra field on
    # (`facts.py`, `if how == "identity" and extra:`, a documented restriction this
    # change honours rather than widens). The trace is built BEFORE the call and
    # travels inside the same atomic write as the mark: mark-then-field could leave a
    # trace naming a winner on a fact whose mark then failed, and a mark with no trace
    # is the state #1817 was filed to end. A `superseded` action, and a `confidence`
    # loser planned against a view that carried no file, keep the text route exactly
    # as before — re-aiming an id-less action at the WINNER's file could stamp the
    # winner invalid by text match, which costs more than the trace it would buy.
    key = None
    if is_confidence and action.get("loser_id") and action.get("loser_source_file"):
        key = (action["loser_source_file"], action["loser_id"])
    trace = _confidence_trace(action) if (is_confidence and key) else None
    reason = f"improve {action['kind']}: {action['reason']}"
    marks = {key: [reason]} if key else {}
    extras = {key: {_CONTRADICTION_TRACE: trace}} if trace else None
    applied = _apply_fact_marks(
        marks, scope,
        field=field, stamp=now_iso, reason_field=reason_field,
        text_matches=None if key else {aim.lower(): ("phrase", reason)},
        stop_after_first=True, extra_fields=extras)
    if applied["marked"] == 0 and applied["unapplied"]:
        # The accounting keys travel with EVERY return, including this one: a caller
        # summing `traces_written` across a run's actions must not read a missing key
        # as a zero it never measured, and an action that marked nothing has by
        # definition written no trace and condemned nothing it cannot name.
        return {"expired_count": 0, "field": field, "traces_written": 0,
                "traces": [], "untraceable": 0,
                "error": str(applied["unapplied"][0]["reason"])}
    # Counted from what the write REPORTS, not from what was asked for, which is how
    # `_fact_resolve_apply` counts its own (`traces_written = sum(... if key in
    # traces)`, over `matched_facts`): an already-invalid fact lands in
    # `already_marked` instead of `matched_facts` and gets no second trace, and an
    # action that marked nothing has an empty `matched_facts`, so a rerun over one
    # pair cannot inflate the nightly figure that reads this field.
    # The record this action aimed, that the writer says it touched: the same test
    # `_fact_resolve_apply:906-907` applies (`key in traces` over `matched_facts`), made
    # per-record because this writer has exactly one key. A rerun lands the loser in
    # `already_marked` instead, so it contributes nothing, and a report that names some
    # other record cannot raise the count either — which keeps `len(traces) ==
    # traces_written` true, the invariant a caller has to be able to rely on when it
    # sums these across a run.
    traces_written = (1 if trace is not None and key is not None and any(
        (m.get("file"), m.get("id")) == (str(key[0]), str(key[1]))
        for m in applied["matched_facts"]) else 0)
    return {"expired_count": applied["marked"], "field": field,
            # What the nightly line counts, per action. The counter sums these, so the
            # figure in the report and the figure in this run record are one
            # measurement of one store rather than two claims about it.
            "traces_written": traces_written,
            "traces": ([trace] if traces_written else []),
            # The #1817 name for the same fact as `_fact_resolve_apply`'s
            # `untraceable`: a confidence pair that marked something but cannot name
            # its winner is counted here, not silently absent from `traces_written`.
            "untraceable": (1 if (is_confidence and trace is None
                                  and applied["marked"]) else 0),
            "marked": applied["matched_facts"]}


def _confidence_trace(action: dict) -> dict | None:
    """The `conflicts_with` record for one planned `confidence` action, or None.

    Rebuilds the winner as a fact view from the three keys the plan step now carries
    (`winner_source_file`, `winner_id`, `winner_fact`) and hands it to
    `facts._contradiction_trace`, the function that produced the record before — so
    there is still exactly ONE builder of this shape in the tree, and the two writers
    cannot drift into two spellings of a resolution.

    None is the caller's `untraceable` case, and it is reached the same way
    `_fact_resolve_apply:888-890` reaches it: a winner with neither a file nor an id
    cannot be pointed at, and a trace naming nothing would be a record asserting a
    resolution nobody can check. It is NOT an empty-string file — `_aim_identity`'s
    spelling of that pair is a mark aimed at a directory, which is a bug, not a gap.
    """
    # Both halves or neither, which is `fact_identity`'s rule (`facts.py:296-300`
    # returns None unless id AND source_file are present): an id with no file names
    # every fact sharing that per-file counter, so it is not an address (#874).
    if not (action.get("winner_source_file") and action.get("winner_id")):
        return None
    return _contradiction_trace(
        {"source_file": action.get("winner_source_file"),
         "id": action.get("winner_id"),
         "fact": action.get("winner_fact")},
        action["entity"], action.get("reason", ""), "")


def _recall_detail(records: list[dict], summary: dict) -> dict:
    """The evidence behind the score, from what `run_eval` already returned.

    The record used to keep only `fact_entity_recall_avg` and discard the
    per-query rows, so two apply runs that expired 24 and 8 facts both reported
    a bare 0.375 and the artifact could not say whether the eval ran, how many
    queries answered, or which entity moved (#702). A per-query
    `fact_entity_recall` of None is a query with no expected entities — not
    run, not zero — and stays None here; `queries_answered` counts the rest.
    """
    per_query = [{"id": r.get("id"),
                  "fact_entity_recall": (r.get("scoring") or {}).get("fact_entity_recall"),
                  "fact_entities_matched": list((r.get("scoring") or {}).get("fact_entities_matched") or []),
                  "expected_entities": list((r.get("expected") or {}).get("entities") or []),
                  "error": r.get("error")}
                 for r in records]
    return {"score": round(summary["overall"]["fact_entity_recall_avg"], 4),
            "queries": len(records),
            "queries_answered": sum(1 for q in per_query if q["fact_entity_recall"] is not None),
            "errors": int(summary["overall"].get("errors") or 0),
            "per_query": per_query}


def _fact_entity_recall(limit: int = 20) -> float | None:
    """`_fact_entity_recall_detail`'s score alone — the CLI's before-value and
    the shape every reader of the metric had before the detail was kept."""
    detail = _fact_entity_recall_detail(limit)
    return None if detail is None else detail["score"]


def _fact_entity_recall_detail(limit: int = 20) -> dict | None:
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
    read as "measured zero"; otherwise `_recall_detail`'s dict — the score with
    the query count, how many answered, and the per-query hits behind it.
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
        if summary["overall"]["fact_entity_recall_avg"] is None:
            # #1250 clause 3, and the reason this is not the last consumer to
            # teach: clause 3 makes `run_eval`'s OWN WRITER record null instead of
            # 0.0, and a `round()` of that raises `TypeError: type NoneType
            # doesn't define __round__` — which this function's blanket handler
            # below reported as "fact_entity_recall could not be measured: …",
            # discarding the real reason (the leg read nothing) inside a message
            # that reads like a broken harness. Testing for null before the
            # `round()`, and asking the shared `fact_leg_read_nothing` why, names
            # the real condition in the log — the same words the eval artifact's
            # own warning uses. A null metric is a non-measurement, and this
            # function returns null for one; a null it cannot explain is raised
            # rather than rounded, so it cannot silently become a 0.0.
            if module.fact_leg_read_nothing(records, module._corpus_provenance()):
                logger.warning(
                    "improve: the fact leg read nothing while the store indexes facts; "
                    "fact_entity_recall is not a measurement here — reporting None, not 0.0")
                return None
            raise TypeError("fact_entity_recall_avg is null but the fact leg measured "
                            "facts — run_eval nulled it for a reason this reader cannot see")
        return _recall_detail(records, summary)
    except Exception as exc:  # noqa: BLE001 - a metric that cannot run is not a zero
        logger.warning("improve: fact_entity_recall could not be measured: %s", exc)
        return None


# ── what a record has to say about itself (#700) ─────────────────────────────

def run_provenance() -> dict:
    """Which fact tree, which store and which commit this pass is acting on.

    `RECORD_DIR` is code-relative, while the tree and the store are
    env-overridable (`LLOYD_FACTS_ROOT`, `LLOYD_KG_DB` in `app.paths`). So code
    aimed at a copy wrote a byte-shape-identical record into the live audit
    trail: five `--apply` records from 2026-09-09 report 32 expirations that
    never reached the live knowledge graph — all 32 condemned facts are still
    active, and `facts_idx` holds no row stamped 09-09 — and nothing in the
    JSON said which store they described. A deletion loop's audit trail must not
    be able to look like a deletion that did not happen.

    `isolated` is what separates the two readings, and it compares against the
    BUILT-IN defaults (`VAULT_FACTS_ROOT_DEFAULT`, `VAULT_KG_DB_DEFAULT`), not
    against `VAULT_FACTS_ROOT`/`VAULT_KG_DB`: those are precisely the constants
    the environment moves, so a run checked against them would report a redirected
    copy as production. The commit is read from the tree the CODE runs from
    (`LLOYD_HOME`), which is also what distinguishes a worktree run from a live
    one; `app.gitinfo.head_commit` returns None rather than raising, so the key
    is stamped `"unknown"` rather than going missing.
    """
    facts_root = Path(FACTS_ROOT)
    try:
        kg_db = Path(_store().path)
    except Exception:  # noqa: BLE001 - StoreUnavailable and anything the driver raises
        # Name the file it would have written, rather than losing the key.
        kg_db = Path(_paths.VAULT_KG_DB)
    isolated = (facts_root.resolve() != Path(_paths.VAULT_FACTS_ROOT_DEFAULT).resolve()
                or kg_db.resolve() != Path(_paths.VAULT_KG_DB_DEFAULT).resolve())
    return {
        "facts_root": str(facts_root),
        "kg_db": str(kg_db),
        "git_head": _head_commit(LLOYD_HOME) or "unknown",
        "isolated": bool(isolated),
    }


# ── the operation ────────────────────────────────────────────────────────────

def run_improvement(apply: bool = False, sources=("corrections", "drift"),
                    entities=None, days: int = DRIFT_WINDOW_DAYS,
                    corrections_days: int = CORRECTIONS_WINDOW_DAYS,
                    limit: int = 40, report_eval: bool = False,
                    eval_limit: int = 20, record: bool = True,
                    max_actions: int = MAX_ACTIONS_PER_RUN) -> dict:
    """One improvement pass. Dry-run unless `apply=True`.

    `days` is the drift window and `corrections_days` the corrections window
    (default `CORRECTIONS_WINDOW_DAYS`); both are threaded to the reader that
    consults them, and the record reports the corrections window it ran with as
    `corrections_window_days` so a zero in the record is reproducible.

    Reports `fact_entity_recall` when asked: #376's own acceptance bar is that
    a change which cannot move that metric is not this feature, so the number
    has to come out of the operation, not out of a human remembering to run
    the eval afterwards.
    """
    started = datetime.datetime.now(datetime.timezone.utc)
    now_iso = started.isoformat()
    # ONE probe sets the store verdict (#1383): the flag and the error text
    # the record carries below must come from the same read, so the record
    # cannot be silent about a store this pass consulted.
    store_verdict = _store_probe()
    before = _active_count()

    if entities:
        signals = [{"entity": e, "source": "explicit", "evidence": "named by caller"}
                   for e in entities]
        tally = {"drift_candidates_total": None, **_drift_tally(None)}
    else:
        signals, tally = _collect_signals(sources=sources, days=days, limit=limit,
                                          corrections_days=corrections_days)

    # #1654: the store verdict gates the WRITES, not the plan. Everything this
    # pass reads from a dead store comes back as a sentinel (-1, {}), and the
    # apply path never touches the store at all — `apply_action` stamps
    # `expired_at`/`invalid_at` into the MARKDOWN fact files under FACTS_ROOT and
    # aims through `_get_facts_sync`, which is the markdown read path too. So
    # before this an `--apply` run during a store outage retired facts anyway and
    # then reported `actions_taken` beside `before_active: -1`: a record that
    # cannot show what it deleted, over an index still serving those facts as
    # active until a rebuild. Only an explicit `store_ok: False` refuses, so this
    # is an outage gate and not an unconditional stop. The dry run is deliberately
    # NOT gated (#1383): a pass that cannot see the graph must still be able to
    # say what it would do, and the wrapper's exit 2 stays reachable.
    writes_refused = bool(apply) and store_verdict.get("store_ok") is False
    apply_writes = bool(apply) and not writes_refused

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
        # A refused entity was never scanned, so its pair counts are null, not
        # 0 (#702): `contradictions: 0, refused: true` was byte-identical to a
        # scanned-and-clean entity once the flag was overlooked, and the seven
        # god-nodes a night that read that way held 1,048 unscanned facts.
        # `checked` (the facts the detector saw) and `skipped_reason` (its own
        # hint) come off the plan, which had carried both since #1251 while the
        # copy here dropped them.
        entry = {"entity": entity, "signal": signal["source"],
                 "contradictions": None if plan["refused"] else plan["contradictions"],
                 "near_duplicates": None if plan["refused"] else plan.get("near_duplicates", 0),
                 # Null for a refused entity for the same #702 reason as
                 # `near_duplicates`: it was never scanned, so its lineage is
                 # unknown, not clean.
                 "lineage_pairs": None if plan["refused"] else plan.get("lineage_pairs", 0),
                 "lineage_withheld": None if plan["refused"] else plan.get("lineage_withheld", 0),
                 "refused": plan["refused"],
                 "checked": plan.get("checked", 0),
                 "skipped_reason": plan.get("skipped_reason") if plan["refused"] else None,
                 # Coverage, carried from the plan. `refused: False` stopped
                 # meaning "scanned whole" the moment an over-bound entity could
                 # be scanned in category-sized pieces, so the record has to say
                 # which it got — `scan_scope` is `entity` or `by_category`, and
                 # `categories_skipped` names the categories too large even for a
                 # piece. Without these two the record's own refusal flag becomes
                 # ambiguous, which is the one thing it cannot afford.
                 "scan_scope": plan.get("scan_scope"),
                 "categories_skipped": plan.get("categories_skipped", []),
                 # The loop's own denominator, before its writes. `pairs_after`
                 # is measured the same way after them.
                 "pairs_before": plan.get("pairs_before", 0),
                 # The #2078 class, per entity, beside that denominator. Null for
                 # a refused entity on the same #702 reasoning as
                 # `lineage_pairs`: it was never scanned, so how many of its
                 # pairs rested on keyword-plus-write-order alone is unknown, not
                 # zero. A `planned: 0` night is otherwise indistinguishable from
                 # one where the guard refused everything it saw — which is the
                 # exact shape #1348 and #1941 each had to be filed for.
                 "keyword_only_flags": (None if plan["refused"]
                                        else plan.get("keyword_only_flags", [])),
                 "planned": len(plan["actions"]), "taken": 0, "actions": []}
        planned += len(plan["actions"])
        if apply_writes:
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
            # judgement before trusting it with `apply`. An apply pass refused by
            # an unreadable store (#1654) lands here too, and the same list is
            # the honest entry for it: asked to write, wrote nothing.
            entry["actions"] = [{"kind": a["kind"], "loser_fact": a["loser_fact"][:90],
                                 "reason": a["reason"], "planned": True, "applied": False}
                                for a in plan["actions"]]
            # Nothing was written, so nothing moved. Stated rather than absent,
            # because a missing before/after field is how a flat metric first
            # looked like an unchanged tree.
            entry["pairs_after"] = entry["pairs_before"]
        entry["before_active"] = plan.get("before_active", -1)
        entry["after_active"] = (_active_count(entity) if apply_writes
                                 else entry["before_active"])
        per_entity.append(entry)

    after = _active_count()
    recall = _fact_entity_recall_detail(eval_limit) if report_eval else None
    record_obj = {
        "ran_at": now_iso,
        "apply": bool(apply),
        # `store_ok` / `store_error` (#1383): whether the store the -1 counts
        # below were supposed to come from was readable at all. The sentinels
        # are kept (a missing count is not a zero), so the record is the only
        # place the two readings part ways.
        **store_verdict,
        # `writes_refused_reason` (#1654): why an APPLY pass took no writes.
        # `actions_taken: 0` cannot say it on its own — that is also what a clean
        # tree reports — so the refused pass quotes the probe's own error, which
        # is the only text naming the store it could not open (class + path).
        # Null otherwise, including for a dry run: a pass that was never asked to
        # write has no refusal to explain.
        "writes_refused_reason": (
            f"apply refused, knowledge-graph store unreadable: "
            f"{store_verdict.get('store_error')}" if writes_refused else None),
        # `facts_root`, `kg_db`, `git_head`, `isolated` (#700): which tree and
        # which store the counts below describe, and which code produced them.
        **run_provenance(),
        "sources": list(sources) if not entities else ["explicit"],
        "days": days,
        "signals": len(signals),
        # The size of the pool `signals` was selected from (#699). Without it a
        # record's `actions_planned: 0` reads as a verdict on the tree when it
        # is a verdict on a slice; with it, scanned-over-total is computable
        # from this JSON alone. None = drift was not consulted (explicit
        # entities, or `--sources corrections`).
        "drift_candidates_total": tally["drift_candidates_total"],
        # What that pool is a share of (#1461). `drift_corpus_total` is every
        # entity dir walked and `drift_pool_fraction` the pool over it, so a
        # pool that IS the corpus (a rebuild re-stamped every row) reads as
        # `drift_status: sweep`, and an empty one as `stale` with the newest
        # fact's date — the drift twin of `corrections_stale_since`.
        "drift_corpus_total": tally["drift_corpus_total"],
        "drift_pool_fraction": tally["drift_pool_fraction"],
        "drift_status": tally["drift_status"],
        "drift_newest_fact": tally["drift_newest_fact"],
        "drift_undated_dirs": tally["drift_undated_dirs"],
        "drift_window_days": tally["drift_window_days"],
        "entities": [s["entity"] for s in signals],
        "actions_planned": planned,
        "actions_taken": taken,
        # What the shared-lineage veto cost this run (#1941), over the entities
        # it scanned. `actions_planned` alone cannot tell a quiet night from a
        # vetoed one: this pass already has two refusals that leave no mark on
        # any field (#1348's pass "reported every entity and planned nothing" and
        # read as healthy), so a third that also removed most of the actions, and
        # said nothing, would be invisible from the record. `lineage_pairs` is the
        # exposure beside it — a withheld of 0 over an exposure of 0 is "nothing
        # to withhold", a withheld of 0 over an exposure of 40 is the veto doing
        # its work somewhere the confidence basis never reached. `or 0` because a
        # refused entity carries null here and its unknowns are not counted.
        "lineage_pairs": sum(e.get("lineage_pairs") or 0 for e in per_entity),
        "lineage_withheld_pairs": sum(e.get("lineage_withheld") or 0
                                      for e in per_entity),
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
        # Every pair this pass declined to expire because its only basis was a
        # keyword opposition plus `created_at` order (#2078), flattened across
        # the entities scanned and named by entity. The pass-level `planned`
        # count cannot tell a quiet night from a guarded one, and this is the
        # third refusal the loop has made that leaves `actions_planned` at zero
        # either way (#1348's attribution guard, #1941's lineage veto, this):
        # `lineage_withheld_pairs` above is the same statement for the
        # confidence basis, and its absence here is what made a 2-action night
        # read as a 2-action night rather than as a 2-of-4 night. `len()`, not a
        # count field, because the list is what names the pairs and a second
        # number beside it could be computed from a different set than the one
        # an operator is reading.
        "keyword_only_flags": [f for e in per_entity
                               for f in (e.get("keyword_only_flags") or [])],
        "fact_entity_recall": None if recall is None else recall["score"],
        # The evidence behind that number, or None when the eval did not run —
        # null is the only spelling of "not measured" on both keys (#702).
        "fact_entity_recall_detail": recall,
        # Name the file the signals came from, not the one that was compiled in
        # first. Before this the field was `str(CORRECTIONS_PATH)` unconditionally,
        # so a record that had read USER.md still reported `memory/corrections.md`
        # and a reader could not tell a live log from a four-month-old one.
        "corrections_path": (last_corrections_read()["paths_yielding_signals"][0]
                             if last_corrections_read()["paths_yielding_signals"]
                             else None),
        "corrections_paths_read": sorted(last_corrections_read()["sources"]),
        "corrections_status": last_corrections_read()["status"],
        # `corrections_status: no_entries_in_window` says the window emptied a
        # log; these two keys say which date it stopped hearing from the
        # operator and how wide the window was that decided it. Without the
        # date, "0 corrections" from a dead channel and "0 corrections" from a
        # quiet fortnight are one record, and #125's rate-over-time metric
        # reads the dead one as a calm week. `corrections_window_days` is the
        # window the read used, not the constant it defaulted from, so a run
        # that overrode it is reproducible from its own record.
        "corrections_stale_since": last_corrections_read()["stale_since"],
        "corrections_window_days": last_corrections_read()["window_days"],
        "corrections_newest_entry": last_corrections_read()["newest_entry"],
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
        "improve(%s): signals=%d drift_candidates=%s/%s (%s) planned=%d taken=%d "
        "active_facts %d -> %d%s",
        "apply" if apply else "dry-run", len(signals), tally["drift_candidates_total"],
        tally["drift_corpus_total"], tally["drift_status"],
        planned, taken, before, after,
        "" if record_obj.get("fact_entity_recall") is None
        else f" fact_entity_recall={record_obj['fact_entity_recall']}")
    return record_obj
