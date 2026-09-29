#!/usr/bin/env python3
"""Build the hand-audit packet that rebuilds the uptake label corpus (#1849).

The uptake gate (`app/uptake.py`) grades its dispute classifier against a
hand-labeled corpus and refuses below `PRECISION_FLOOR` 0.70 / `RECALL_FLOOR`
0.50. The only set this repo ever had, `eval/uptake/labels/hand-2026-09-11.json`
(46 items, 23 positives), is `spent`: every turn it names rolled off the
transcript store at the 2026-09-22 data-root cutover, so `validate_labels`
resolves 0 of 46 and the probe cannot grade anything with it. Relabelling is a
HUMAN judgement — `labeled_by` in that file is `hand:alan-turns-2026-09-11`, and
`app/uptake.py`'s `LABELER` constant is dead, so nothing downstream even checks
the string. This script therefore makes that judgement cheap to give and refuses
to fake it: it emits the turns worth reading with `label` and `reason` left NULL,
it never calls the classifier, and it never writes a non-null label in either
mode.

Two modes.

Emit (default) — read the transcript corpus, run the cue screen
(`uptake.candidate_disputes`), and write `packet-<date>.json` holding EVERY
candidate turn plus a stratified sample of non-candidates:

    scripts/uptake_hand_audit.py
    scripts/uptake_hand_audit.py --labels-dir /tmp/labels --date 2026-11-19

Below its floors it refuses, prints the two counts it measured and the date it
projected from them, and writes nothing — which is what it does today, at 60
human turns and 2 candidates:

    human_turns(days=900) = 60    cue-screen candidates = 2    yield 3.33%
    REFUSED: needs >= 40 packet items from >= 15 candidates (candidates 2 < 15)
    growth: 10.0 human turns/day over a 6.0-day corpus span
    projected ready date 2026-12-05 (500 turns: the 500-turn trigger)
    wrote nothing

Merge — read a packet a human has filled in and write the live label set:

    scripts/uptake_hand_audit.py --merge eval/uptake/labels/packet-2026-11-19.json \\
        --labeled-by hand:alan-turns-2026-11-19

WHY 20 POSITIVES AND NOT 40 LABELS IS THE FLOOR THAT BINDS. `labels_path`
(`app/uptake.py`) returns `files[-1]` of the `hand-*.json` glob, so a merged
`hand-<date>.json` SHADOWS `hand-2026-09-11.json` instead of adding to it — the
new set cannot borrow the old set's 23 positives, it has to carry its own. Two
tests pin 20: `test_hand_labeled_corpus_covers_the_item_s_minimum` and
`test_classifier_metrics_meet_the_item_s_minimums`. A merged set emitted at the
item's original floors (40 items / 15 candidates) lands near 16 positives and
turns both of them red, so `--merge` refuses below 20 and names the shortfall.

WHY THE SAMPLE IS STRATIFIED AND DETERMINISTIC. The non-candidate pool keeps
`human_turns`' own order (session name, then position within it) and the sample
takes evenly spread indices from it. Nothing is shuffled and no RNG is seeded: a
packet rebuilt from an unchanged store is byte-identical, so either copy can be
handed to a labeler, and one long session cannot stand in for a store of many.

WHY THE PACKET OMITS THE ENGINE FIELDS. A committed label item also carries
`engine_raw` / `engine_predicted`, and the old file's own `engine_replay.note`
records `prompt_tuned_on_labels: True` — those keys are how a label set becomes
in-sample. The few-shot exemplars in `app/uptake.py` were written against this
corpus, so a packet that shipped the engine's guess next to the blank label would
hand the labeler the answer and quietly re-infect the relabelled set. A packet
item carries exactly six keys and no engine field at all.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import uptake  # noqa: E402

#: The window the builder reads. 900 days because it has to match the window
#: `uptake_probe._corpus_index` resolves labels through: a packet emitted over a
#: rolling 30-day window would be a different corpus from the one the graded
#: probe indexes, and a label written from it could go unresolvable the moment
#: its turn aged out of the short window.
CORPUS_DAYS = 900

#: The packet's own floors, from the item. `MIN_ITEMS` is what makes a packet a
#: corpus rather than a list of disputes: the old set was 46 items over 23
#: positives, and a set with no negatives measures no precision at all.
MIN_ITEMS = 40
MIN_CANDIDATES = 15

#: The floor on the MERGED set, and the reason `MIN_CANDIDATES` is 15 and not 5.
#: See the module docstring: a merged set shadows the old one, so it must clear
#: the >= 20 positives `tests/test_uptake.py` pins twice, on its own.
MIN_POSITIVES = 20

#: The documented trigger, in turns. Never a calendar date: this corpus grows, so
#: a date written into prose is false the week usage changes.
TRIGGER_TURNS = 500

#: Minimum non-candidates to sample. 25 alongside 15 candidates is the old set's
#: shape — roughly half positives among the candidates, negatives to grade
#: precision against — and it is what makes `MIN_ITEMS` reachable at the floor.
NON_CANDIDATE_SAMPLE = 25

#: Used ONLY when the corpus cannot yield a rate of its own (one turn, or
#: timestamps that do not parse). 8/day is the growth measured at triage
#: (44 -> 60 human turns in two days); it is a fallback and says `fallback` in
#: the printed basis, never a stand-in for the rate the store itself shows.
FALLBACK_TURNS_PER_DAY = 8.0

#: 3 is what `uptake_probe` already means by "the measurement refused", so an
#: operator who schedules this sees one convention for refusal.
EXIT_BELOW_FLOOR = 3
EXIT_MERGE_REFUSED = 4

#: The packet item contract: exactly these keys, and no engine field. A human
#: fills `reason` and `label`; nothing else is blank.
PACKET_KEYS = ("turn_id", "ts", "prev_assistant", "user_text", "reason", "label")

#: Copied into the packet so the labeler reads the definition from the same bytes
#: the packet came in, and `--merge` copies it back out of the PACKET rather than
#: from a second copy of this string in the merge path.
LABEL_DEFINITION = (
    "DISPUTE = the user reacts against what the agent just delivered: says it was "
    "wrong/false/broken, rejects or reverses it, reports a claimed-working result "
    "does not work, or re-asks for promised work. NOT = new request, follow-up "
    "question about content, approval, directive to proceed, small talk."
)


# ------------------------------------------------------------------ corpus --

def corpus_index(turns: Sequence[uptake.Turn]) -> dict[str, uptake.Turn]:
    """Every human turn keyed by `turn_id`, the way the probe indexes them."""
    return {t.turn_id: t for t in turns}


def _systematic(pool: Sequence[Any], want: int) -> list[Any]:
    """`want` evenly spread members of `pool`, no RNG, endpoints included.

    Deterministic on purpose: a packet rebuilt from the same store must be the
    same packet, because a human may be handed either copy and a resampled one
    would silently change which turns got read. When the pool is smaller than
    `want` every member is taken.
    """
    n = len(pool)
    if want <= 0 or n <= want:
        return list(pool)
    if want == 1:
        return [pool[0]]
    return [pool[int(round(i * (n - 1) / (want - 1)))] for i in range(want)]


def _packet_item(turn: uptake.Turn) -> dict[str, Any]:
    """One blank row of the packet. No label parameter: this cannot be filled."""
    return {
        "turn_id": turn.turn_id,
        "ts": turn.ts,
        "prev_assistant": turn.prev_assistant,
        "user_text": turn.user_text,
        "reason": None,
        "label": None,
    }


def build_packet(turns: Sequence[uptake.Turn],
                 candidates: Sequence[uptake.Turn],
                 *, sample: int = NON_CANDIDATE_SAMPLE,
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every candidate, plus a stratified sample of the rest, as packet items.

    There is no parameter by which a label could be supplied here, and the
    classifier is not an argument either: the refusal to auto-label is structural
    rather than a convention a caller can ignore.
    """
    cand_ids = {t.turn_id for t in candidates}
    missing = sorted(cand_ids - {t.turn_id for t in turns})
    if missing:
        # The cue screen filters the list it was handed, so this is reachable only
        # if a caller passes two unrelated collections. Naming the ids is the
        # difference between a one-line bug report and an hour of guessing.
        raise ValueError(
            f"{len(missing)} candidate turn(s) are not in the turn list, "
            f"e.g. {missing[:3]}")
    ordered_cands = [t for t in turns if t.turn_id in cand_ids]
    non_cands = [t for t in turns if t.turn_id not in cand_ids]
    want = max(sample, MIN_ITEMS - len(ordered_cands))
    sampled = _systematic(non_cands, want)
    items = [_packet_item(t) for t in ordered_cands + sampled]
    meta = {
        "human_turns": len(turns),
        "candidates": len(ordered_cands),
        "sampled_non_candidates": len(sampled),
        "non_candidates": len(non_cands),
    }
    return items, meta


# -------------------------------------------------------------- projection --

def _iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def measured_growth(turns: Sequence[uptake.Turn]) -> tuple[float, str]:
    """Human turns per day, off the corpus's own first and last turn stamps.

    The store is the only honest clock here: the builder is triggered by VOLUME,
    and a rate hard-coded at triage would be wrong the week the corpus starts
    gzipping or usage doubles. Falls back to the triage rate — and says `fallback`
    in the basis string, so the printed projection can be traced — when there is
    no span to measure.
    """
    stamps = sorted(s for s in (_iso(t.ts) for t in turns) if s is not None)
    if len(stamps) < 2:
        return FALLBACK_TURNS_PER_DAY, (
            f"fallback {FALLBACK_TURNS_PER_DAY}/day (fewer than two turn "
            "timestamps parsed)")
    span_days = (stamps[-1] - stamps[0]).total_seconds() / 86400.0
    if span_days <= 0:
        return FALLBACK_TURNS_PER_DAY, (
            f"fallback {FALLBACK_TURNS_PER_DAY}/day (corpus spans no time)")
    rate = len(turns) / span_days
    return rate, (f"{rate:.1f} human turns/day over a {span_days:.1f}-day "
                  "corpus span")


def projection(turns: Sequence[uptake.Turn], n_candidates: int,
               *, today: datetime | None = None) -> dict[str, Any]:
    """When this corpus could produce a mergeable packet, DERIVED from the counts.

    The target is the LATER of the documented trigger and the turn count that
    yields `MIN_CANDIDATES` at the yield measured in this very corpus: at 2/60 =
    3.3% a 500-turn corpus gives ~16 candidates, but at a 25%-yield corpus 60
    turns already clears 15 — and a projection that ignored the measured yield
    would publish a date nobody has to wait for. Returns `ready_date: None` when
    no rate can be measured, rather than inventing one.
    """
    now = today or datetime.now(timezone.utc)
    n = len(turns)
    rate, basis = measured_growth(turns)
    yield_frac = (n_candidates / n) if n else 0.0
    target, why = TRIGGER_TURNS, f"{TRIGGER_TURNS}-turn trigger"
    if yield_frac > 0:
        for_cands = math.ceil(MIN_CANDIDATES / yield_frac)
        if for_cands > target:
            target, why = for_cands, (
                f"{MIN_CANDIDATES}-candidate floor at the measured "
                f"{yield_frac * 100:.2f}% yield needs {for_cands} turns")
    out: dict[str, Any] = {
        "human_turns": n, "candidates": n_candidates, "yield_frac": yield_frac,
        "rate_per_day": rate, "rate_basis": basis, "target_turns": target,
        "target_reason": why, "ready_date": None, "days": None,
    }
    if target <= n:
        out["ready_date"] = now.date().isoformat()
        out["days"] = 0
        return out
    # An empty store gets NO date. `FALLBACK_TURNS_PER_DAY` is a rate from a
    # populated corpus at triage; applying it to zero turns would publish
    # "ready in 63 days" over a tree with no transcripts at all — which is the
    # #1679 shape (a probe that measures an unmounted root and reads as health).
    # A fallback that survives an empty corpus stops being a fallback and becomes
    # an assumption, so the printed basis says `fallback` whenever it is used.
    if n == 0 or rate <= 0:
        return out
    days = math.ceil((target - n) / rate)
    out["days"] = days
    out["ready_date"] = (now.date() + timedelta(days=days)).isoformat()
    return out


# -------------------------------------------------------------------- emit --

def emit(*, labels_dir: Path, date: str | None, root: Path | None,
         out: Any, err: Any, dry_run: bool = False) -> int:
    """Write `packet-<date>.json`, or refuse with the counts it measured."""
    turns = uptake.human_turns(root=root, days=CORPUS_DAYS)
    candidates = uptake.candidate_disputes(turns)
    items, meta = build_packet(turns, candidates)
    yield_pct = (len(candidates) / len(turns) * 100) if turns else 0.0
    print(f"human_turns(days={CORPUS_DAYS}) = {len(turns)}    cue-screen "
          f"candidates = {len(candidates)}    yield {yield_pct:.2f}%", file=out)

    short: list[str] = []
    if len(candidates) < MIN_CANDIDATES:
        short.append(f"candidates {len(candidates)} < {MIN_CANDIDATES}")
    if len(items) < MIN_ITEMS:
        short.append(f"packet items {len(items)} < {MIN_ITEMS}")
    if short:
        p = projection(turns, len(candidates))
        print(f"REFUSED: needs >= {MIN_ITEMS} packet items from "
              f">= {MIN_CANDIDATES} candidates ({'; '.join(short)})", file=err)
        print(f"growth: {p['rate_basis']}", file=err)
        if p["ready_date"] is None:
            print(f"projected ready date: none — no growth rate measurable; "
                  f"target {p['target_turns']} turns ({p['target_reason']})",
                  file=err)
        else:
            print(f"projected ready date {p['ready_date']} "
                  f"({p['days']} days at {p['rate_per_day']:.1f}/day; target "
                  f"{p['target_turns']} turns: {p['target_reason']})", file=err)
        print("wrote nothing", file=err)
        return EXIT_BELOW_FLOOR

    if date is None:
        date = datetime.now(timezone.utc).date().isoformat()
    elif not _valid_date(date):
        print(f"REFUSED: --date {date!r} is not YYYY-MM-DD", file=err)
        print("wrote nothing", file=err)
        return EXIT_BELOW_FLOOR
    path = labels_dir / f"packet-{date}.json"
    if path.exists():
        # A packet is a document a human may be reading RIGHT NOW. Rewriting it
        # silently would discard the labels already filled in — the one loss this
        # whole tool exists to prevent.
        print(f"REFUSED: {path} already exists, and a packet may be under audit. "
              "Pass --date another date, or remove it deliberately.", file=err)
        print("wrote nothing", file=err)
        return EXIT_BELOW_FLOOR

    doc = {
        "schema": 1,
        "packet": True,
        "created": date,
        "definition": LABEL_DEFINITION,
        "n_items": len(items),
        "n_candidates": meta["candidates"],
        "corpus": {**meta, "days": CORPUS_DAYS},
        "merge_floors": {"min_items": MIN_ITEMS, "min_positives": MIN_POSITIVES},
        "fill_in": "set label to 1 (DISPUTE) or 0 (NOT) and write a one-line "
                   "reason for every item; --merge refuses while any item is "
                   f"unlabelled, and while fewer than {MIN_POSITIVES} are 1",
        "items": items,
    }
    if dry_run:
        print(f"DRY RUN: would write {path} ({len(items)} items, "
              f"{meta['candidates']} candidates, "
              f"{meta['sampled_non_candidates']} sampled)", file=out)
        return 0
    labels_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {path} ({len(items)} items: {meta['candidates']} candidates + "
          f"{meta['sampled_non_candidates']} sampled non-candidates)", file=out)
    return 0


# ------------------------------------------------------------------- merge --

def read_packet(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("items"), list):
        raise ValueError(f"{path} has no `items` list")
    return doc


def merge(*, packet: Path, labeled_by: str | None, labels_dir: Path,
          date: str | None, root: Path | None, out: Any, err: Any) -> int:
    """Write `hand-<date>.json` from a filled-in packet, or refuse."""
    # Provenance is checked BEFORE the packet is opened. `LABELER` in
    # `app/uptake.py` is a dead constant and nothing downstream validates the
    # string, so this flag is the only control left on who attests the corpus —
    # a default, even `hand:unknown`, would be a fabricated attestation.
    if labeled_by is None:
        print("REFUSED: --merge needs an explicit --labeled-by (for example "
              f"--labeled-by hand:alan-turns-{date or datetime.now(timezone.utc).date()}). "
              "`labeled_by` is the only record that a corpus was hand-labeled: "
              "app/uptake.py's LABELER constant is dead and no reader validates "
              "the string, so it is never defaulted and never guessed.", file=err)
        print("wrote nothing", file=err)
        return EXIT_MERGE_REFUSED
    if not labeled_by.strip():
        print("REFUSED: --labeled-by is empty; it must name who labeled the set "
              "(for example --labeled-by hand:alan-turns-<date>)", file=err)
        print("wrote nothing", file=err)
        return EXIT_MERGE_REFUSED
    if date is None:
        date = datetime.now(timezone.utc).date().isoformat()
    if not _valid_date(date):
        print(f"REFUSED: --date {date!r} is not YYYY-MM-DD", file=err)
        print("wrote nothing", file=err)
        return EXIT_MERGE_REFUSED

    try:
        doc = read_packet(packet)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSED: cannot read packet {packet}: {exc}", file=err)
        print("wrote nothing", file=err)
        return EXIT_MERGE_REFUSED

    items = [i for i in doc["items"] if isinstance(i, dict) and i.get("turn_id")]
    if not items:
        print(f"REFUSED: packet {packet} has no items", file=err)
        print("wrote nothing", file=err)
        return EXIT_MERGE_REFUSED

    incomplete = _incompleteness(items)
    if incomplete:
        print(f"REFUSED: {packet} is not a completed packet — "
              + "; ".join(incomplete), file=err)
        print("wrote nothing", file=err)
        return EXIT_MERGE_REFUSED

    positives = [i for i in items if i["label"] == 1]
    if len(positives) < MIN_POSITIVES:
        # The shortfall is printed against the floor and the tests that pin it:
        # the number the item first named (15 candidates) is NOT the number that
        # binds, and a reader told only "17 positives" cannot act on it.
        print(f"REFUSED: {len(positives)} of {len(items)} items labelled 1, but a "
              f"merged set needs >= {MIN_POSITIVES} positives — short by "
              f"{MIN_POSITIVES - len(positives)}. The floor exists because "
              f"`labels_path` takes the NEWEST `hand-*.json` and so SHADOWS "
              f"hand-2026-09-11.json instead of adding to it: the new set must "
              f"clear it alone, and tests/test_uptake.py::"
              f"test_hand_labeled_corpus_covers_the_item_s_minimum and "
              f"::test_classifier_metrics_meet_the_item_s_minimums both pin "
              f">= {MIN_POSITIVES}. Emit a larger packet; do not relabel "
              f"negatives to reach it.", file=err)
        print("wrote nothing", file=err)
        return EXIT_MERGE_REFUSED

    target = labels_dir / f"hand-{date}.json"
    if target.exists():
        print(f"REFUSED: {target} already exists. A label set is history; write a "
              "new date rather than overwrite one.", file=err)
        print("wrote nothing", file=err)
        return EXIT_MERGE_REFUSED

    out_items = [{**{k: i.get(k) for k in PACKET_KEYS}, "labeled_by": labeled_by}
                 for i in items]
    resolved = _re_resolve(out_items, root)
    doc_out = {
        "schema": 1,
        "labeled_by": labeled_by,
        "created": date,
        "definition": doc.get("definition") or LABEL_DEFINITION,
        "n_positives": len(positives),
        "n_items": len(out_items),
        "status": "live",
        # `n_resolved` is #1848's field: `labels_status` reads it, and a set that
        # stops re-resolving is how a corpus becomes spent. Reported, not gated —
        # a turn can legitimately roll off between emit and merge, and the
        # probe's own label check is where a dead corpus must be refused.
        "n_resolved": resolved,
        "source_packet": str(packet),
        "items": out_items,
    }
    labels_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc_out, indent=2) + "\n")
    tail = (f"re-resolved {resolved}/{len(out_items)} against the transcript store"
            if resolved >= 0 else "re-resolution unavailable")
    print(f"wrote {target} ({len(out_items)} items, {len(positives)} positives, "
          f"labeled_by={labeled_by}, {tail})", file=out)
    return 0


def _incompleteness(items: Sequence[Mapping[str, Any]]) -> list[str]:
    """Why a packet is not yet a completed label set, empty if it is.

    A label that is neither 0 nor 1 is reported rather than coerced: `"yes"` and
    `true` and `2` are all a human's honest attempt at "this was a dispute", and
    silently reading any of them as 1 would put a judgement the labeler did not
    make into the corpus the gate trusts.
    """
    unlabelled = [i for i in items if i.get("label") is None]
    illegal = [i for i in items
               if i.get("label") is not None and not _is_label(i["label"])]
    no_reason = [i for i in items
                 if i.get("label") == 1 and not str(i.get("reason") or "").strip()]
    out: list[str] = []
    if unlabelled:
        out.append(f"{len(unlabelled)} of {len(items)} item(s) still unlabelled "
                   "(label is null)")
    if illegal:
        out.append(f"{len(illegal)} item(s) with a label that is not 0 or 1, e.g. "
                   f"{illegal[0].get('turn_id')}={illegal[0]['label']!r}")
    if no_reason:
        out.append(f"{len(no_reason)} positive(s) with no reason")
    return out


def _is_label(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def _re_resolve(items: Sequence[Mapping[str, Any]], root: Path | None) -> int:
    """How many merged items still point at a real turn. A report, never a gate."""
    try:
        index = corpus_index(uptake.human_turns(root=root, days=CORPUS_DAYS))
        return int(uptake.validate_labels(list(items), index=index)["n_resolved"])
    except Exception as exc:  # noqa: BLE001 - a report must not fail a merge
        print(f"note: could not re-resolve the merged set ({exc})", file=sys.stderr)
        return -1


def _valid_date(value: Any) -> bool:
    return (isinstance(value, str)
            and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))
            and _iso(value) is not None)


# -------------------------------------------------------------------- main --

def main(argv: Sequence[str] | None = None, *,
         out: Any = None, err: Any = None) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    ap = argparse.ArgumentParser(
        prog="uptake_hand_audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "TRIGGER: run the emit mode when human_turns(days=900) >= about 500\n"
            "turns, NOT on a calendar date and not on the packet's own floors\n"
            f"alone. 500 is where >= {MIN_CANDIDATES} cue-screen candidates appear at\n"
            f"the measured ~3.3% yield, which is where >= {MIN_POSITIVES} labelled\n"
            "positives become reachable at all: --merge refuses below\n"
            f"{MIN_POSITIVES} positives because labels_path takes the NEWEST\n"
            "hand-*.json and so SHADOWS the spent hand-labeled set rather than\n"
            "adding to it, and tests/test_uptake.py pins >= 20 positives twice.\n"
            "Below the floors this exits 3, prints both measured counts and the\n"
            "projected date derived from them, and writes nothing.\n"
            "\n"
            "A packet never carries a label, and never carries engine_raw /\n"
            "engine_predicted either: the few-shot exemplars in app/uptake.py were\n"
            "tuned on this corpus, so a shown guess would make the relabelled set\n"
            "in-sample again. A packet is also invisible to the gate by name --\n"
            "LABEL_GLOB is 'eval/uptake/labels/hand-*.json'.\n"),
    )
    ap.add_argument("--labels-dir", default=None,
                    help="where packet-<date>.json / hand-<date>.json live "
                         "(default: eval/uptake/labels beside this repo)")
    ap.add_argument("--root", default=None,
                    help="transcript data root to read (default: uptake.lloyd_root())")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD to name the output file (default: today, UTC)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what the emit mode would write, and write nothing")
    ap.add_argument("--merge", metavar="PACKET", default=None,
                    help="read a packet a human has filled in and write "
                         "hand-<date>.json from it")
    ap.add_argument("--labeled-by", default=None,
                    help="REQUIRED with --merge: who labeled the set (for example "
                         "hand:alan-turns-<date>). Never defaulted.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root).expanduser() if args.root else None
    labels_dir = (Path(args.labels_dir).expanduser() if args.labels_dir
                  else REPO / "eval" / "uptake" / "labels")
    if args.merge:
        return merge(packet=Path(args.merge), labeled_by=args.labeled_by,
                     labels_dir=labels_dir, date=args.date, root=root,
                     out=out, err=err)
    return emit(labels_dir=labels_dir, date=args.date, root=root, out=out,
                err=err, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
