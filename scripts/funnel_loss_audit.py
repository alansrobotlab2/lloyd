#!/usr/bin/env python3
"""One dated, seeded, agent-labelled audit of funnel LOSS CASES (#4238).

The funnel census (#2507, `scripts/funnel_census.py`) answers WHERE the loop loses
durable output: its `dominant_loss_stage` is a count over stages. It cannot answer
WHICH failure mode a lost row is, because a stage count is one number and a loss is
a per-case event. #2507's step 4 is the per-case audit, and #552 makes it the one
control that survives its own ruling — `backlog/552-…:202`: "Same-family grader.
The classifier is a Qwen model judging advice written by Qwen jobs; it may agree
with its own bad writing. Hand-labels in steps 1 and 4 are the control — do not
skip step 4." This module draws those cases and publishes them WITH their evidence
so a person can do step 4 without re-deriving the frame. It does not do step 4.

    python -m scripts.funnel_loss_audit --seed 4238
    python -m scripts.funnel_loss_audit --seed 4238 --cases 24 --newest 250

Frame, and the one thing the frame is not. Every case is drawn from the funnel's
widest loss population: a `prefetch.skill_match` offer row whose `landed` is not
true. The `offered`-stage stratum (cohort artifacts with no offer row at all) is
NOT drawn from, because it has no turn to read — its unit is an artifact, so "turn
index" and "deciding quote from that turn's user text" cannot exist for it; it is
published as a named count in `never_offered_cohort` instead.

WHAT THIS AUDIT CANNOT SEE, stated before its own numbers because the numbers read
as a discovery otherwise:

  * `landed` IS the not-rendered flag. `app/prefetch.py:733-734` derives `landed`
    and `injected_body` from one `mode`, so an unlanded row did not render — which
    makes `written-not-surfaced` the label the DRAW assigns, not one the transcript
    supplies. Measured over all 246 attributable unlanded rows in the 250 newest
    event files on 2026-11-26: 246 of 246 landed in that label, and 0 of 246 showed
    the offered name in that turn's rendered `<skill name=…>` block. Three of the
    four labels can therefore only measure zero unless an independent rendering
    route reaches the prompt anyway; `contradiction` is where such a case would
    show up, and it is the one field here that is not implied by the draw.
  * the frame is machine-authored. Those same 246 rows sit on 204 `worker` and 42
    `browser` sessions and ZERO chat sessions, so every "deciding quote" is a
    queue-authored job prompt, not a human utterance. `platform_split` publishes
    that rather than letting one label merge two populations.
  * the census it supplements is stale and is cited nowhere in here. The funnel
    census's newest artifact is older than its declared 7-day cadence
    (`scripts/eval_instrument_index.py` reports it `stale`), so this audit prints
    its OWN measured denominator in `frame` and quotes no stage figure from it.

Labels are assigned by mechanical predicates over transcript evidence, never by a
model call: two runs on one seed are byte-identical apart from `measured_at`, and
#552's same-family warning is a second reason not to route the judgement through a
model. The header says so in the same block as `labeled_by`, because a set of
agent-labelled cases is provisional measurement evidence and NOT the hand-label
control #552 reserves — quoting it as Alan's, or as a human-labelled set, is the
#2441/#2556 bound that `tests/test_fact_write_gate.py` pins.

Where the ordinal comes from. An offer row carries no turn id (`prefetch` emits it
while the prompt is being BUILT, so its row lands BEFORE the `brain1.user_prompt_received`
row of the very turn it served — `app/uptake.py:1630-1648`, which is why uptake's
telemetry credits `turn + 1`). A loss row is therefore attributed to the turn whose
receipt row FOLLOWS it: the published `turn_ordinal` is that receipt's 1-based index
among receipt rows in that session's event file, and a row with no following receipt
row is not attributable at all — uptake's own safe side ("a row after the last
receipt row … contributes nothing"). Two divergences are published rather than
smoothed: uptake's counter further restricts to `data.source == "user"`
(`app/uptake.py:1609-1611`), a restriction that selects nothing on this frame, and
22 of the 38 frame sessions log fewer receipt rows than the transcript holds user
messages, so a receipt index equals a transcript index only where the two counts
agree — `ordinal_corroborated_cases` counts the cases that rest on a session where
they do.

Nothing acts on these labels: no skill is written, merged or archived, no gate reads
them, and no dollar figure is emitted (the census's own cost block is not read, and
a quote that would carry a currency figure makes its case unquotable).

The artifact lands under `app.paths.DATA_ROOT/eval/funnel/` and is named
`loss-audit-<date>.json` — deliberately NOT `funnel-loss-audit-<date>.json`, because
the instrument index registers the census arm on `pattern="funnel-*.json"` with
`value_path="stages[surfaced].rate"` (`scripts/eval_instrument_index.py:386-405`)
and `_newest_dated_file` takes the newest file matching that glob: a `funnel-*`
name here would hijack the census arm and carry no rate for it. The write dir is
resolved through `funnel_census.resolve_out_dir`, which is the rail that refuses a
path inside the checkout (#2378).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import paths
from eval.stats import wilson_ci
from scripts import funnel_census as fc
from scripts import uptake_hand_audit as uha

__all__ = ["LABELS", "MIN_CASES", "MIN_SESSIONS", "QUOTE_MAX", "ARTIFACT_PREFIX",
           "CENSUS_GLOB", "LossAuditRefused", "artifact_name", "build_audit",
           "choose_quote", "main", "matches_census_glob", "render_text",
           "resolve_out_dir", "scan_loss_rows", "write_artifact"]

ARTIFACT = "funnel-loss-audit"
VERSION = 1
ITEM = "#4238"

#: The four failure modes #2507 step 4 asks for. The order is the funnel's own:
#: an artifact that was never written, one written but never reaching a prompt, one
#: that reached a prompt and changed nothing, and one that was followed and made
#: the work worse. Fixed order, because `dominant_label` breaks a tie on it and a
#: tie broken on dictionary order would move with the code that builds it.
LABELS = ("never-written", "written-not-surfaced", "surfaced-not-used",
          "used-and-harmful")

#: The published floors. MIN_CASES is #2507's own "the >=20 hand-labelled loss
#: cases" minimum, and it is a refusal rather than a warning: an artifact of 19
#: cases would be quoted beside a 20-case requirement, and a k/n over 19 is a
#: different claim wearing the same words.
MIN_CASES = 20
MIN_SESSIONS = 8
DEFAULT_CASES = 24
#: Cases per session in the draw. Without a cap a 24-case sample of a 246-row frame
#: can be four sessions' bad afternoon, which is what "stratified over >=8 distinct
#: sessions" exists to prevent.
MAX_PER_SESSION = 3
QUOTE_MAX = 200
#: Event files scanned, newest first. The frame is what the store holds NOW; 250
#: files is the window the triage probe measured the frame over (246 attributable
#: rows over 38 sessions), so the published denominator is comparable to it.
DEFAULT_NEWEST = 250

SKILL_MATCH_EVENT = "prefetch.skill_match"
RECEIPT_EVENT = "brain1.user_prompt_received"

#: The instrument index's census-arm glob. Named so a test can assert the audit's
#: own filename against the pattern it must not satisfy, rather than against a
#: comment saying so.
CENSUS_GLOB = "funnel-*.json"
ARTIFACT_PREFIX = "loss-audit-"

#: A currency figure, the shape #4238's clause 6 keeps out of the artifact. Only
#: `$` followed by a digit: `$` alone is shell prose and `$LLOYD_DATA` is a path.
_CURRENCY = re.compile(r"\$\s*\d")


class LossAuditRefused(Exception):
    """The frame cannot produce a publishable audit; nothing is written."""


# ── the frame ─────────────────────────────────────────────────────────────────

def _newest_event_files(event_root: Path, newest: int) -> list[Path]:
    """The `newest` most recently modified event files, by name for ties.

    mtime ordering with a name tiebreak, so a seed reproduces across runs of the
    same store and a test's fixture files cannot be scanned in creation order.
    """
    files = [p for p in Path(event_root).glob("*.events.jsonl") if p.is_file()]
    return sorted(files, key=lambda p: (-p.stat().st_mtime, p.name))[:newest]


def _rows_of(path: Path) -> list[dict]:
    """One event file, parsed. An unparseable line is skipped, not fatal.

    `app/uptake.py` documents that these files hold lines that do not parse, and a
    loss audit that dies on one of them would report no frame at all rather than a
    frame one line shorter.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("data"), dict):
            out.append(row)
    return out


def scan_loss_rows(event_root: Path | None = None, *, newest: int = DEFAULT_NEWEST) -> dict:
    """Every attributable unlanded offer row in the frame, with its turn ordinal.

    Attribution is positional and it is the whole design: the offer row is emitted
    during prompt build so it lands before the receipt row of the turn it served
    (`app/uptake.py:1630-1648`), so the turn is the one whose receipt row FOLLOWS
    the offer row and the ordinal is that receipt's index among receipt rows. A row
    with no following receipt row cannot be attributed and is counted, never
    guessed at.
    """
    root = Path(event_root) if event_root else paths.EVENT_LOGS_DIR
    files = _newest_event_files(root, newest)
    rows: list[dict] = []
    offer_rows = unlanded = unattributable = 0
    offered_names: set[str] = set()
    receipt_totals: dict[str, int] = {}
    for path in files:
        parsed = _rows_of(path)
        sid = path.name[: -len(".events.jsonl")]
        receipts = 0
        receipt_totals[sid] = 0
        for i, row in enumerate(parsed):
            event, data = row.get("event"), row["data"]
            if event == RECEIPT_EVENT:
                receipts += 1
                receipt_totals[sid] = receipts
            elif event == SKILL_MATCH_EVENT:
                offer_rows += 1
                name = data.get("skill")
                if isinstance(name, str) and name:
                    offered_names.add(name)
                if data.get("landed") is True:
                    continue
                unlanded += 1
                following = next(
                    (j for j in range(i + 1, len(parsed))
                     if parsed[j].get("event") == RECEIPT_EVENT), None)
                if following is None:
                    unattributable += 1
                    continue
                rows.append({
                    "session": sid,
                    "turn_ordinal": receipts + 1,
                    "skill": name if isinstance(name, str) and name else "",
                    "score": data.get("score"),
                    "landed": data.get("landed"),
                    "injected_body": data.get("injected_body"),
                    "offer_ts": row.get("ts"),
                    "receipt_ts": parsed[following].get("ts"),
                })
    return {
        "rows": rows,
        "files_scanned": len(files),
        "offer_rows": offer_rows,
        "unlanded_rows": unlanded,
        "attributable_rows": len(rows),
        "unattributable_rows": unattributable,
        "sessions": sorted({r["session"] for r in rows}),
        "receipt_rows_by_session": receipt_totals,
        "offered_names": sorted(offered_names),
    }


def transcript_path(sessions_dir: Path, session: str) -> Path | None:
    """The session's transcript, live OR archived; None when there is none.

    `.json` first, then uptake's `SESSION_ARCHIVE_SUFFIX`: the sweep gzips old
    transcripts in place, and an audit that read only `.json` would silently drop
    every loss row whose session has been archived — a frame that shrinks on the
    sweep's schedule is not a measurement.
    """
    from app import uptake
    base = Path(sessions_dir) / session
    plain = base.with_suffix(".json")
    if plain.exists():
        return plain
    # `SESSION_ARCHIVE_SUFFIX` is itself ".json.gz" (`app/uptake.py:622`), so it goes
    # onto the bare session id, not onto a `.json` name — `x.json.json.gz` matches
    # nothing and reads as a missing transcript.
    archived = Path(f"{base}{uptake.SESSION_ARCHIVE_SUFFIX}")
    return archived if archived.exists() else None


def read_transcript(path: Path) -> dict | None:
    """One transcript, through the one reader that is gzip-transparent.

    `app.uptake.read_session` is the only reader here, on purpose: it is what keeps
    `.json.gz` archives in frame, and its own exception list is the difference
    between "this session is unusable" and "the audit died on one rotten member"
    (`app/uptake.py:655-676`).
    """
    from app import uptake
    return uptake.read_session(path)


def turns_of(doc: dict) -> tuple[int, dict[int, str], dict[int, set[str]]]:
    """Non-empty user turns, and the `<skill name=…>` block merged onto each.

    The ordinal sequence is non-empty user messages in order, which is the
    transcript's own numbering. A `subliminal` message is merged into the turn it
    follows and never creates one, exactly as `app/uptake.py:820-838` does: the
    writer persists an injected block as its own message after the turn it was
    built for.
    """
    from app import uptake
    n = 0
    texts: dict[int, str] = {}
    blocks: dict[int, set[str]] = {}
    for msg in doc.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = uptake._text_of(msg.get("content"))
        if role == "user":
            if not text.strip():
                continue
            n += 1
            texts[n] = text.strip()
        elif role == "subliminal" and n:
            found = uptake._SKILL_INJECTION.findall(text)
            if found:
                blocks.setdefault(n, set()).update(found)
    return n, texts, blocks


def choose_quote(text: str) -> str | None:
    """A verbatim span of this turn's own user text, <= `QUOTE_MAX`, currency-free.

    Verbatim is the clause: the returned span is a substring of `text`, so a reader
    can grep the transcript for it. Preference order is the front of the turn
    (where the ask is), then the longest line, then a currency-free window — a
    window rather than a truncation because #4238's clause 6 forbids emitting a
    dollar figure, and a quote that had to include one would be the audit paying
    for its own quote with a number it is not allowed to publish. Returns None when
    no >=40-char currency-free window exists; that case is counted, not published
    with a doctored quote.
    """
    def ok(span: str) -> bool:
        return bool(span) and not _CURRENCY.search(span)

    def cap(span: str) -> str:
        span = span.strip()
        if len(span) <= QUOTE_MAX:
            return span
        cut = span.rfind(" ", 0, QUOTE_MAX)
        return span[:cut if cut > QUOTE_MAX // 2 else QUOTE_MAX].rstrip()

    for line in text.split("\n"):
        if ok(line) and (cand := cap(line)) and len(cand) >= min(40, QUOTE_MAX):
            return cand
    head = cap(text)
    if ok(head):
        return head
    for chunk in (s.strip() for s in re.split(r"\n+", text)):
        if ok(chunk) and len(chunk) >= 40:
            return cap(chunk)
    return None


# ── labelling ─────────────────────────────────────────────────────────────────

def cohort_names(skills_roots: list[Path] | None = None) -> set[str]:
    """The artifact names the live loader serves — #2507's cohort definition.

    Read through the census's own `_cohort_names`, so "does this artifact exist"
    has one answer on this box and the audit cannot call an artifact dead because
    it re-derived the loader's rule.
    """
    return fc._cohort_names(skills_roots)


def label_case(*, skill: str, in_cohort: bool, rendered: bool, read_after_turn: bool,
               dispute: bool) -> str | None:
    """One label from four mechanical signals, or None when nothing here is a loss.

    `never-written` is the offered name missing from the live cohort: the matcher
    reached for an artifact that is not in circulation, which is a produce-leg loss
    that the offer row alone cannot show (measured 0 of the 35 distinct offered
    names on 2026-11-26 — every name was in the cohort).

    `surfaced-not-used` and `used-and-harmful` require the artifact to have reached
    the prompt anyway despite `landed` not being true, which is the contradiction
    the audit exists to find; `used` is uptake's own proxy — a `skills_read` for
    that name at or after the attributed turn (`app.uptake.skills_read_by_session`,
    `app/uptake.py:1723-1733`) — and `harmful` is uptake's own recall-first dispute
    screen (`app/uptake.candidate_disputes`), not a ruling: a positive is a
    candidate that reached the screen #552 uses, and the screen's docstring says it
    trades recall for engine cost and leaves the decision to the classifier.

    None is a real answer: the row says the offer never rendered, yet the artifact
    reached that turn, was read, and drew no correction. There is no loss there to
    label, and inventing a fifth label for it would rename the contradiction rather
    than report it.
    """
    if not in_cohort:
        return "never-written"
    if not rendered:
        return "written-not-surfaced"
    if not read_after_turn:
        return "surfaced-not-used"
    return "used-and-harmful" if dispute else None


def build_cases(rows: list[dict], *, seed: int, want: int = DEFAULT_CASES,
                sessions_dir: Path | None = None,
                cohort: set[str] | None = None,
                skills_read: dict[str, dict[str, int]] | None = None,
                receipt_totals: dict[str, int] | None = None,
                event_root: Path | None = None) -> dict:
    """The seeded draw, labelled, with every drop counted under a name.

    Deterministic by construction, not by luck: the pool is sorted into one total
    order before the seeded shuffle, the session cap makes the spread structural,
    and the published cases are re-sorted afterwards — so `--seed 4238` twice gives
    the same cases, counts and k/n, and the seed is the only thing a reader needs.
    """
    from app import uptake
    sess_dir = Path(sessions_dir) if sessions_dir else paths.SESSIONS_DIR
    cohort = cohort_names() if cohort is None else cohort
    if skills_read is None:
        skills_read = uptake.skills_read_by_session(paths.DATA_ROOT)

    pool = sorted(rows, key=lambda r: (r["session"], r["turn_ordinal"], r["skill"],
                                       str(r.get("offer_ts"))))
    random.Random(seed).shuffle(pool)

    cases: list[dict] = []
    dropped: Counter[str] = Counter()
    # One read per session, and one drop per session: `unusable` counts transcripts,
    # not rows, so a session with thirty loss rows must not report thirty of them —
    # and re-reading a thirty-thousand-message transcript per row is not measurement,
    # it is I/O.
    frames: dict[str, dict | None] = {}
    per_session: Counter[str] = Counter()
    contradictions: Counter[str] = Counter()
    corroborated = 0
    for row in pool:
        if len(cases) >= want:
            break
        if per_session[row["session"]] >= MAX_PER_SESSION:
            dropped["session cap"] += 1
            continue
        if row["session"] not in frames:
            path = transcript_path(sess_dir, row["session"])
            doc = read_transcript(path) if path is not None else None
            frames[row["session"]] = doc
            if path is None:
                dropped["no transcript"] += 1
            elif doc is None:
                # Printed as `unusable`, never a silent drop: a frame that shrinks
                # without saying so is the #1679 shape (a probe that measures nothing
                # and reads as health).
                dropped["unusable"] += 1
        doc = frames[row["session"]]
        if doc is None:
            continue
        n_users, texts, blocks = turns_of(doc)
        ordinal = row["turn_ordinal"]
        if ordinal not in texts:
            dropped["no user text at that ordinal"] += 1
            continue
        quote = choose_quote(texts[ordinal])
        if quote is None:
            dropped["unquotable (currency figure)"] += 1
            continue
        rendered = row["skill"] in blocks.get(ordinal, set())
        read_turn = skills_read.get(row["session"], {}).get(row["skill"])
        read_after = read_turn is not None and read_turn >= ordinal
        in_cohort = row["skill"] in cohort
        dispute = bool(uptake.candidate_disputes([
            uptake.Turn(session=row["session"], ts=row.get("offer_ts"),
                        user_text=texts[ordinal], ordinal=ordinal)]))
        label = label_case(skill=row["skill"], in_cohort=in_cohort, rendered=rendered,
                           read_after_turn=read_after, dispute=dispute)
        if label is None:
            dropped["no loss to label"] += 1
            continue
        if rendered:
            contradictions["landed not true, yet rendered in that turn"] += 1
        if read_after:
            contradictions["landed not true, yet read at or after that turn"] += 1
        total_receipts = (receipt_totals or {}).get(row["session"], -1)
        if n_users == total_receipts:
            corroborated += 1
        per_session[row["session"]] += 1
        cases.append({
            "session": row["session"],
            "turn_ordinal": ordinal,
            "turn_id": f"{row['session']}#{ordinal}",
            "platform": str(doc.get("platform") or "unset"),
            "session_source": str(doc.get("source") or ""),
            "skill": row["skill"],
            "score": row.get("score"),
            "landed": row.get("landed"),
            "injected_body": row.get("injected_body"),
            "offer_ts": row.get("offer_ts"),
            "receipt_ts": row.get("receipt_ts"),
            "label": label,
            "evidence": {"in_live_cohort": in_cohort,
                         "rendered_in_that_turn": rendered,
                         "skills_read_at_or_after_turn": read_after,
                         "reached_dispute_screen": dispute,
                         "ordinal_corroborated": n_users == total_receipts},
            "deciding_quote": quote,
            "deciding_quote_chars": len(quote),
        })

    cases.sort(key=lambda c: (c["session"], c["turn_ordinal"], c["skill"]))
    return {"cases": cases, "dropped": dict(dropped),
            "contradictions": dict(contradictions),
            "ordinal_corroborated_cases": corroborated}


# ── the artifact ──────────────────────────────────────────────────────────────

def artifact_name(day: str) -> str:
    """`loss-audit-<date>.json`, and never a name the census arm's glob matches."""
    return f"{ARTIFACT_PREFIX}{day}.json"


def matches_census_glob(name: str) -> bool:
    """Would the instrument index book this file as the census arm's newest?"""
    return fnmatch.fnmatch(Path(name).name, CENSUS_GLOB)


def label_counts(cases: list[dict]) -> dict[str, int]:
    """All four labels, always all four. A label that measured nothing is `0`.

    An omitted key and a measured zero are different claims: `never-written: 0`
    says the audit looked and the offered names were all in circulation, and it is
    that sentence the funnel reader needs in order to rule the produce leg out.
    """
    seen = Counter(c["label"] for c in cases)
    return {label: seen.get(label, 0) for label in LABELS}


def dominant_label(counts: dict[str, int]) -> dict[str, Any]:
    """The label with the highest count, ties broken on `LABELS` order, with k/n.

    Wilson, not Wald, for the same reason `eval/stats.py` gives: at k = n Wald
    collapses to width zero and reports a draw's selection as a certainty — which
    is precisely the failure mode of a 24/24 dominant label.
    """
    n = sum(counts.values())
    if n <= 0:
        raise LossAuditRefused("no cases to summarise")
    best = max(LABELS, key=lambda lab: (counts[lab], -LABELS.index(lab)))
    k = counts[best]
    lo, hi = wilson_ci(k, n)
    return {"label": best, "k": k, "n": n, "rate": round(k / n, 4),
            "wilson_ci_95": [round(lo, 4), round(hi, 4)],
            "tie_break": "highest count, then LABELS order",
            "note": ("k/n over the DRAWN cases only: the denominator is unlanded "
                     "offer rows, so this is not the rate of this failure mode "
                     "among all offers")}


def human_control(*, root: Path | None = None,
                  turns: list | None = None) -> dict[str, Any]:
    """The still-owed #552 step-4 control, with the triggers it names measured.

    Read through `uptake_hand_audit`, which owns `TRIGGER_TURNS`, `MIN_CANDIDATES`
    and the projection, so this audit cannot state a floor the hand-audit has since
    moved. `ready_date: None` is that module's "no rate can be measured", passed
    through rather than filled in.
    """
    from app import uptake
    corpus = (uptake.human_turns(root=root, days=uha.CORPUS_DAYS) if turns is None
              else turns)
    candidates = uptake.candidate_disputes(corpus)
    projection = uha.projection(corpus, len(candidates))
    return {
        "owed": ("#552 reserves steps 1 and 4 for hand labels — 'do not skip step "
                 "4' — so a person must read >=5 of the cases published here and "
                 "record agreement or disagreement before any label in this file "
                 "is quoted as a control"),
        "human_turns": projection["human_turns"],
        "trigger_turns": uha.TRIGGER_TURNS,
        "human_turns_at_or_past_trigger": projection["human_turns"] >= uha.TRIGGER_TURNS,
        "candidates": projection["candidates"],
        "min_candidates": uha.MIN_CANDIDATES,
        "candidates_at_or_past_floor": projection["candidates"] >= uha.MIN_CANDIDATES,
        "corpus_days": uha.CORPUS_DAYS,
        "ready_date": projection["ready_date"],
        "target_turns": projection["target_turns"],
        "target_reason": projection["target_reason"],
    }


def build_audit(*, seed: int, cases: int = DEFAULT_CASES, newest: int = DEFAULT_NEWEST,
                event_root: Path | None = None, sessions_dir: Path | None = None,
                skills_roots: list[Path] | None = None,
                uptake_root: Path | None = None,
                now: datetime | None = None) -> dict:
    """The whole artifact, or a `LossAuditRefused` naming the shortfall.

    Every argument exists for a test or a job: the frame's two stores, the cohort's
    roots and the clock are all injectable, so no test has to touch a production
    store to pin a clause.
    """
    from app import uptake
    now = now or datetime.now(timezone.utc)
    scan = scan_loss_rows(event_root, newest=newest)
    cohort = cohort_names(skills_roots)
    drawn = build_cases(scan["rows"], seed=seed, want=cases,
                        sessions_dir=sessions_dir, cohort=cohort,
                        skills_read=uptake.skills_read_by_session(
                            uptake_root or paths.DATA_ROOT),
                        receipt_totals=scan["receipt_rows_by_session"])
    complete = drawn["cases"]
    n_cases, n_sessions = len(complete), len({c["session"] for c in complete})
    short: list[str] = []
    if n_cases < MIN_CASES:
        short.append(f"complete cases {n_cases} < {MIN_CASES}")
    if n_sessions < MIN_SESSIONS:
        short.append(f"distinct sessions {n_sessions} < {MIN_SESSIONS}")
    if short:
        raise LossAuditRefused(
            "; ".join(short) + f" (frame: {scan['attributable_rows']} attributable "
            f"unlanded rows over {len(scan['sessions'])} sessions; dropped: "
            f"{drawn['dropped'] or 'none'})")

    counts = label_counts(complete)
    day = now.astimezone(timezone.utc).date().isoformat()
    platforms = Counter(c["platform"] for c in complete)
    offered = set(scan["offered_names"])
    return {
        "artifact": ARTIFACT, "version": VERSION, "item": ITEM,
        "measured_at": now.isoformat(timespec="seconds"),
        "tree": paths.describe_tree(),
        "labeling": {
            "labeled_by": f"agent:lloyd-transcript-audit-{day}",
            "kind": "agent-labelled provisional measurement evidence",
            "not_hand_labels": (
                "These are NOT hand labels, are NOT Alan's labels, and must not be "
                "quoted as a human-labelled set. Every label here was assigned by "
                "mechanical predicates over the transcript in this repo "
                "(`label_case`), with no model call, so that a re-run on the same "
                "seed reproduces it."),
            "hand_label_control_still_owed": (
                "#552 reserves steps 1 and 4 for hand labels ('do not skip step 4') "
                "because a same-family grader may agree with its own bad writing; "
                "this file supplements step 4 and cannot replace it. Bound: "
                "#2441/#2556, pinned by tests/test_fact_write_gate.py."),
        },
        "human_control": human_control(root=uptake_root),
        "frame": {
            "event_files_scanned": scan["files_scanned"],
            "offer_rows": scan["offer_rows"],
            "unlanded_offer_rows": scan["unlanded_rows"],
            "attributable_unlanded_rows": scan["attributable_rows"],
            "unattributable_rows_no_following_receipt": scan["unattributable_rows"],
            "distinct_sessions_in_frame": len(scan["sessions"]),
            "denominator_note": (
                "Measured here, on this store, now. No stage figure from the "
                "#2507 funnel census is quoted in this artifact: that census's "
                "newest artifact is older than its declared cadence and the "
                "instrument index reports it stale."),
        },
        "never_offered_cohort": {
            "count": len(cohort - offered),
            "cohort_size": len(cohort),
            "basis": ("Cohort artifacts with no `prefetch.skill_match` row in the "
                      "scanned window. Published as a count, never drawn as a case: "
                      "this stratum's unit is an artifact, so it has no turn to read "
                      "and no user text to quote."),
        },
        "sample": {
            "seed": seed, "cases_asked": cases, "complete_cases": n_cases,
            "distinct_sessions": n_sessions, "max_cases_per_session": MAX_PER_SESSION,
            "min_cases": MIN_CASES, "min_sessions": MIN_SESSIONS,
            "dropped": drawn["dropped"],
            "attribution": ("the turn whose `brain1.user_prompt_received` row "
                            "follows the offer row; ordinal = that receipt's index "
                            "among receipt rows in the session's event file"),
            "ordinal_corroborated_cases": drawn["ordinal_corroborated_cases"],
            "ordinal_corroboration": (
                "receipt rows == non-empty user messages for that session, which is "
                "when a receipt index is also the transcript's own index"),
        },
        "platform_split": dict(sorted(platforms.items())),
        "chat_cases": platforms.get("chat", 0) + platforms.get("", 0),
        "contradiction": {
            "counts": drawn["contradictions"],
            "meaning": ("Rows whose `landed` is not true and which the transcript "
                        "nevertheless shows rendering, or being read anyway. These "
                        "are the audit's only finding the draw does not imply."),
        },
        "counts": counts,
        "dominant_label": dominant_label(counts),
        "what_this_cannot_see": [
            "`written-not-surfaced` is the draw's own condition, not a discovery: "
            "`app/prefetch.py:733-734` derives `landed` from one `mode`, so an "
            "unlanded row did not render by definition. A dominant label of "
            "n/n here restates the denominator.",
            "Three of the four labels can fire only on a contradiction (see "
            "`contradiction`); measured over all 246 attributable unlanded rows on "
            "2026-11-26 that was 0 rows, which is why three zeros are the expected "
            "shape rather than a quiet day.",
            "No human turn is in the frame: 204 of 246 attributable rows sit on "
            "`worker` sessions and 42 on `browser`, so every deciding quote is a "
            "queue-authored job prompt. `chat_cases` says how many are not.",
            "`used` is uptake's presence proxy, not an application measure: a "
            "`skills_read` proves the model asked for the skill, not that it "
            "followed it.",
            "`harmful` is uptake's recall-first dispute screen reaching a turn, not "
            "a ruling that the artifact made the work worse.",
        ],
        "cases": complete,
        "acts_on": ("nothing: no skill is written, merged or archived, no gate and "
                    "no prune job reads this file, and no dollar figure is emitted"),
    }


def write_artifact(audit: dict, out_dir: Path) -> Path:
    """Write exactly one dated artifact and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.fromisoformat(audit["measured_at"]).date().isoformat()
    path = out_dir / artifact_name(day)
    text = json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False)
    # A currency figure anywhere in the artifact is a clause-6 breach, and the
    # quote chooser is the only place one could enter from a transcript. Refuse
    # rather than redact: a doctored "verbatim" quote is worse than no quote.
    if _CURRENCY.search(text):
        raise LossAuditRefused(
            "refusing to write an artifact carrying a currency figure; clause 6 of "
            "#4238 forbids a dollar figure, and a quote with one in it cannot be "
            "both verbatim and publishable")
    audit["artifact_path"] = str(path)
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")
    return path


def render_text(audit: dict) -> str:
    """The report. Exactly one line may begin `dominant loss label: `."""
    s, f, d = audit["sample"], audit["frame"], audit["dominant_label"]
    hc = audit["human_control"]
    lo, hi = d["wilson_ci_95"]
    lines = [
        f"funnel loss audit {ITEM} — {s['complete_cases']} cases over "
        f"{s['distinct_sessions']} sessions, seed {s['seed']}",
        f"frame: {f['unlanded_offer_rows']} unlanded offer rows in "
        f"{f['offer_rows']} offers, {f['attributable_unlanded_rows']} attributable "
        f"over {f['distinct_sessions_in_frame']} sessions "
        f"({f['event_files_scanned']} event files)",
        f"never-offered cohort: {audit['never_offered_cohort']['count']} of "
        f"{audit['never_offered_cohort']['cohort_size']} artifacts",
        f"labeled_by: {audit['labeling']['labeled_by']} — provisional measurement "
        f"evidence, NOT hand labels",
        f"human control owed: human_turns {hc['human_turns']} vs trigger "
        f"{hc['trigger_turns']}; candidates {hc['candidates']} vs floor "
        f"{hc['min_candidates']}",
        "labels:",
    ]
    lines += [f"  {label}: {audit['counts'][label]}" for label in LABELS]
    lines += [
        f"chat vs machine cases: chat {audit['chat_cases']} of "
        f"{s['complete_cases']}; platforms "
        + ", ".join(f"{k}={v}" for k, v in audit["platform_split"].items()),
        f"unusable transcripts: {audit['sample']['dropped'].get('unusable', 0)}"
        f" (dropped: {', '.join(f'{k}={v}' for k, v in sorted(s['dropped'].items()))})"
        if s["dropped"] else "unusable transcripts: 0 (no case dropped)",
        f"dominant loss label: {d['label']} k/n={d['k']}/{d['n']} "
        f"wilson95=[{lo}, {hi}]",
    ]
    return "\n".join(lines)


def resolve_out_dir(cli_value: str | None = None, *, data_root: Path | None = None,
                    code_root: Path | None = None) -> Path:
    """The artifact's home, through the census's own rail against the checkout."""
    return fc.resolve_out_dir(cli_value, data_root=data_root, code_root=code_root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.funnel_loss_audit",
        description="Audit funnel loss cases by transcript (#4238).")
    ap.add_argument("--seed", type=int, default=4238,
                    help="draw seed (default 4238); printed, and the draw "
                         "reproduces on it")
    ap.add_argument("--cases", type=int, default=DEFAULT_CASES,
                    help=f"cases to draw (default {DEFAULT_CASES}; "
                         f"a run completing under {MIN_CASES} refuses to write)")
    ap.add_argument("--newest", type=int, default=DEFAULT_NEWEST,
                    help=f"newest event files to scan (default {DEFAULT_NEWEST})")
    ap.add_argument("--out-dir", default=None,
                    help="where the dated artifact goes (default "
                         "<data root>/eval/funnel; a path inside the checkout is "
                         "refused)")
    ap.add_argument("--event-root", default=None,
                    help="directory of <session>.events.jsonl (default "
                         "app.paths.EVENT_LOGS_DIR)")
    ap.add_argument("--sessions-dir", default=None,
                    help="directory of <session>.json / .json.gz (default "
                         "app.paths.SESSIONS_DIR)")
    ap.add_argument("--uptake-root", default=None,
                    help="root for the skills_read proxy and human_turns (default "
                         "app.paths.DATA_ROOT)")
    ap.add_argument("--skills-root", action="append", default=None, metavar="DIR",
                    help="skill root for the cohort; repeatable (default "
                         "agent_mcp.skills.SKILLS_DIRS)")
    args = ap.parse_args(argv)

    try:
        out_dir = resolve_out_dir(args.out_dir)
    except fc.CensusWriteRefused as exc:
        print(f"funnel loss audit REFUSED: {exc}")
        return 2
    try:
        audit = build_audit(
            seed=args.seed, cases=args.cases, newest=args.newest,
            event_root=Path(args.event_root) if args.event_root else None,
            sessions_dir=Path(args.sessions_dir) if args.sessions_dir else None,
            skills_roots=[Path(p) for p in args.skills_root] if args.skills_root
            else None,
            uptake_root=Path(args.uptake_root) if args.uptake_root else None)
    except LossAuditRefused as exc:
        print(f"funnel loss audit REFUSED: {exc}")
        return 2
    path = write_artifact(audit, out_dir)
    print(render_text(audit))
    print(f"artifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
