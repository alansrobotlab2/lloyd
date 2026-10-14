#!/usr/bin/env python3
"""Step 1 of #2717 — the revision-triggered candidate set, read-only.

ReHarness's backward-revision loop starts by naming the reflections a reward-bearing run
retrieved; Lloyd has never printed that list. This script prints it, and nothing else: it
edits no memory file, calls no model, and changes no stored row. What it establishes is
the denominator the whole idea has been missing — of the items retrieval actually put in
front of a run, how many appeared in runs the dispute classifier flagged, and how many of
those are more than a single flag.

The attribution is uptake's. Presence for a prefetched note comes from
`app/uptake.py:2067`, which records it only where the persisted injected-context block
names the note; presence for a MEMORY.md/USER.md bullet is `always_in_force` — every turn,
by construction, which is a bound and is reported as one. Nothing here re-derives either:
the corpus walk, the turn key, the cue screen and the classifier all belong to
`app/uptake.py` and `scripts/uptake_probe.py`, and a second join here would produce numbers
that could not be compared with the tables the nightly writer already cites.

Two ways to ask:

    .venvs/lloyd/bin/python scripts/memory_revision_candidates.py --table eval/uptake/uptake-2026-09-20.json
    .venvs/lloyd/bin/python scripts/memory_revision_candidates.py --days 30

**One input: the table file, and the verdict map the probe now persists into it.**
`uptake.build_uptake_table` writes `uptake.DISPUTE_MAP_KEY` — the `{turn_id: verdict}` map
its own run computed — so a table carries both the attribution rows and the verdicts, and
this script reads the table and nothing else (#2717 clause 2). A table written before that
key existed has no map, and that is reported as `unavailable` naming the table and the
reason, never as a window with zero disputes.

**No engine call happens here, and that is a design constraint, not a cost saving.** The
first version of `refresh_window` walked the live window the way the probe does
(`uptake.human_turns` -> `uptake.build_uptake_table`) and rebuilt the disputed-run list by
calling `uptake.classify_dispute` on the turns it wanted verdicts for. It crashed on its
first call: `app/uptake.py:1254` takes `(prev_assistant, user_text)`, not a `Turn`, so the
call raised `TypeError: missing a required argument: 'user_text'`, and no node in the test
file named the function that crashed. Beyond the crash the design was wrong twice over: a
verdict recomputed today against a model or prompt that may have moved is a DIFFERENT
measurement from the one the table reports, and printing both in one column makes the
artifact unfalsifiable; and #2269's reason stands — the engine that wrote the disputed turns
is the one that would decide they were disputed.

What lands on disk is one JSON artifact per run, named with the date it was generated, in
`eval/memory-revision/` beside the tables it reads (#1094's rule: dated files, newest wins
— never a rolling `-latest` name, which is how a skeleton once got read as a report).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import uptake  # noqa: E402

#: The two kinds this item's step 1 reads. A prefetched note/fact is the honest half —
#: it has a per-turn retrieval signal. A loaded index line is included because the item
#: names it, and because its rows are exactly the ones a reader must not trust: see
#: `attribution_upper_bound`.
RETRIEVED_KINDS = ("note", "memory_entry")

#: Kinds uptake emits that this report deliberately does not grade, with the reason
#: printed beside their count. A row that is dropped silently is a row whose absence
#: will be read as "none exist" — the same reason #552 prints the memory lines its entry
#: grammar could not identify instead of discarding them.
EXCLUDED_KINDS = {
    "skill": "uptake attributes skill rows to injection and read events, not to the "
             "prefetched retrieval step 1 reads; and presence is a floor for them "
             "(one-way), so a dispute rate over them would not mean the same thing",
}

#: The item's own risk rule, as data: "never narrow on a single dispute". Uptake's
#: dispute classifier runs at recall 0.43 over a heuristic cue, so one flag on one
#: retrieval is not evidence that a line caused a bad turn. Measured on the three
#: committed tables, every disputed note row sat at exactly 1 (#2717 triage), which is
#: why this is a field rather than a paragraph.
MIN_DISPUTED_RUNS = 2

ELIGIBLE = "eligible"
INELIGIBLE_BELOW_FLOOR = "ineligible_below_dispute_floor"

#: A table file cannot answer "which runs disagreed" — only "how many". Said as a value
#: on the artifact rather than left to the reader, because a missing column that looks
#: like an empty one is the failure this report exists to avoid.
VERDICTS_UNAVAILABLE = ("unavailable: a table file records dispute counts, not per-turn "
                        "verdicts; run with --days for the per-run verdict column")

#: `app/uptake.py:2067` truncates `presence_turn_ids` at 20 by design while
#: `present_in_turns` keeps counting. The artifact flags that disagreement per row.
PRESENCE_IDS_CAP = 20

ARTIFACT_DIR_NAME = "memory-revision"
ARTIFACT_PREFIX = "memory-revision-candidates-"


def _rate(numerator: int, denominator: int) -> dict:
    """A rate, printed with what it was divided by.

    A bare float is what let a share on #552's own report get read back as a percentage of
    something nobody had named, so the pair travels together and the walk in
    `tests/test_memory_revision_candidates.py` refuses any rate that loses it.
    """
    return {"numerator": int(numerator), "denominator": int(denominator),
            "value": (round(int(numerator) / int(denominator), 4) if denominator else None)}


def _row_by_item(table: Mapping[str, Any]) -> dict[str, dict]:
    return {row["entry"]: row for row in table.get("entries", [])}


def read_verdict_map(doc: Mapping[str, Any]) -> "tuple[dict | None, str]":
    """The per-turn verdict map this table's own probe run computed, or why it has none.

    Returns `(map, "")` when the table carries a non-empty map, else `(None, reason)`. The
    reason is what clause 2 is graded on: a reader has to be able to tell "this instrument
    never recorded WHICH of its turns were disputed" from "these turns were recorded and
    none of them was disputed" — the first is a missing measurement, the second is a clean
    bill, and before this function the artifact could print either one as an empty dispute
    column.

    The key is read through `uptake.DISPUTE_MAP_KEY` and never spelled as a literal here: a
    copied string would degrade silently into `unavailable` forever if the writer renamed
    the key, which is the same failure the map exists to prevent, in a quieter costume.

    The keys are `Turn.turn_id` (`<session>#<ordinal>`) as of that run's own turn list —
    per-run ids, which is what makes the join reconstructible, and also why they cannot be
    replayed against the transcript store months later. `corpus.verdict_keyed_on` says the
    same thing from the writer's side.
    """
    raw = doc.get(uptake.DISPUTE_MAP_KEY)
    if isinstance(raw, dict) and raw:
        return dict(raw), ""
    if raw is None:
        why = "carries no per-turn dispute-verdict map"
    elif isinstance(raw, dict):
        why = "carries an EMPTY dispute-verdict map"
    else:
        why = f"carries a {uptake.DISPUTE_MAP_KEY} that is a {type(raw).__name__}, not a map"
    return None, (f"{why}, so WHICH runs were disputed cannot be reconstructed from it and "
                  "no per-run dispute verdict exists for any item. This is a missing "
                  "measurement, not a window in which nothing was disputed.")


def _run_verdict(run: str, verdicts: Mapping[str, Any]) -> Any:
    """One run's verdict, with the two kinds of "nothing" kept apart.

    `None` means the probe screened the turn and the engine answered nothing; the string
    `"unscreened"` means the turn is not in the map at all and was never screened. Both used
    to arrive as `None`, which is how a run nobody looked at ends up reading as a cleared one.
    """
    return verdicts[run] if run in verdicts else "unscreened"


def candidates(table: Mapping[str, Any],
               verdicts: Optional[Mapping[str, Any]] = None) -> list[dict]:
    """One candidate per retrieved row: its runs, the verdicts in them, and eligibility.

    `verdicts` is uptake's own `{turn_id: bool}` map. Without it the row still carries
    everything a committed table knows — run list, counts, source — and the caller is told
    in the artifact's `verdicts` field that the per-run column is empty for a reason.

    The disputed-run count comes from uptake's `disputes` field rather than from the
    printed id list, because the list is capped at 20 and a count derived from it would
    quietly understate a widely-present line.
    """
    out = []
    for row in table.get("entries", []):
        if row.get("kind") not in RETRIEVED_KINDS:
            continue
        always = row.get("presence_source", "").startswith("always_in_force")
        runs = list(row.get("presence_turn_ids") or [])
        if verdicts is None:
            verdict_list: Any = VERDICTS_UNAVAILABLE
            flagged: list[str] = []
            unresolved: list[str] = []
            unscreened: list[str] = list(runs)
        else:
            # An always-in-force row is present in every turn by construction, so its
            # per-run column is the whole window and is labelled an upper bound; a
            # prefetched row's column is only the runs its own row names.
            scope = sorted(verdicts) if always else runs
            # The verdict list stays uptake's own, sentinel-free: a reader comparing this
            # column with the table sees the value the table holds. The three lists below
            # are where the two kinds of blank are told apart. `None` in the map is a turn
            # the cue screen reached on which the classifier answered NOTHING; absence from
            # the map is a turn nobody screened. Both are blank in the list, which is how a
            # run nobody looked at ends up reading as one that cleared the item.
            verdict_list = [{"run": run, "verdict": verdicts.get(run)} for run in scope]
            flagged = [run for run in scope if verdicts.get(run) is True]
            unresolved = [run for run in scope
                          if run in verdicts and verdicts[run] is None]
            unscreened = [run for run in scope if run not in verdicts]
        count = int(row.get("disputes") or 0)
        eligible = count >= MIN_DISPUTED_RUNS
        out.append({
            "item": row["entry"],
            "kind": row["kind"],
            "presence_source": row.get("presence_source"),
            #: True for a loaded index line: uptake cannot see WHICH runs needed it, only
            #: that it was in all of them. Every rate over these rows is an upper bound,
            #: which is the item's own reason for distrusting them.
            "attribution_upper_bound": bool(always),
            "text": row.get("text"),
            "source_doc": row.get("source_doc"),
            "present_in_turns": row.get("present_in_turns"),
            "present_in_runs": None if always else runs,
            "runs_truncated": bool(not always and runs
                                   and len(runs) < int(row.get("present_in_turns") or 0)),
            "verdicts": verdict_list,
            #: The other two per-run lists the persisted map makes possible.
            #: `disputed_run_ids` is the disputed one; these two are the difference between
            #: "screened and cleared" and "never screened at all", which the table's
            #: `disputes` count cannot express in either direction.
            "unresolved_run_ids": unresolved,
            "unscreened_run_ids": unscreened,
            "disputed_run_count": count,
            "disputed_run_ids": flagged,
            "weighted_disputes": row.get("weighted_disputes"),
            "min_disputed_runs": MIN_DISPUTED_RUNS,
            "eligible": eligible,
            "eligibility": ELIGIBLE if eligible else INELIGIBLE_BELOW_FLOOR,
            "eligibility_reason": (
                "disputed in at least two runs" if eligible else
                f"single dispute: {count} flagged run{'' if count == 1 else 's'} in the "
                f"window, below the floor of {MIN_DISPUTED_RUNS} "
                f"(classifier recall is 0.43, so one flag is not a cause)"),
        })
    return out


def default_out_dir() -> Path:
    """Where a dated artifact belongs: the data root, never the code checkout.

    `~/lloyd` is code only — a run that left its output beside the source would be a stray
    for the watchdog to find and would not survive a promote. Resolved through
    `app/paths.DATA_ROOT` rather than spelled as `~`, and imported lazily because resolving
    a data root can fail on a tree with no marker, and a module that only ever READS a table
    should still be importable there.
    """
    from app.paths import DATA_ROOT
    return Path(DATA_ROOT) / "eval" / ARTIFACT_DIR_NAME


def _eligibility_by_attribution(cands: Sequence[Mapping[str, Any]]) -> dict:
    """The two-dispute floor, counted once per attribution family and never pooled.

    `prefetched` is the honest half: uptake saw the note or fact in that turn's
    `<vault-context>` block, so a disputed run under it is a run the item was retrieved into.
    `always_in_force` is the loaded `MEMORY.md`/`USER.md` index line, present in every prompt,
    so every one of its rows is "disputed" whenever any turn is and its rate says nothing about
    the item. The item's own pilot finding is the reason this is split rather than pooled: over
    the committed window all 14 disputed prefetched rows carried a single dispute and every
    `always_in_force` row cleared the floor by construction — a pooled "112 of 117 eligible"
    would have read as a working filter over a window in which nothing honestly attributed was
    eligible at all.
    """
    out: dict = {}
    for family, upper_bound in (("prefetched", False), ("always_in_force", True)):
        rows = [c for c in cands if bool(c["attribution_upper_bound"]) is upper_bound]
        passing = [c for c in rows if c["eligible"]]
        out[family] = {
            "rows": len(rows),
            "meeting_floor": len(passing),
            "rate": _rate(len(passing), len(rows)),
            "attribution": ("upper bound — present in every prompt, so no per-run signal "
                            "can falsify it" if upper_bound else
                            "per-run — uptake recorded this item in that turn's block"),
        }
    return out


def build_artifact(table: Mapping[str, Any], *, window_days: int, as_of: str,
                   source_table: str,
                   verdicts: Optional[Mapping[str, Any]] = None) -> dict:
    """The dated artifact: candidates, denominators, and an explicit zero finding.

    `verdicts` is an override for tests. Left out, the map is read from the table itself
    (`read_verdict_map`), which is what makes it impossible for this artifact and the
    instrument it reports on to disagree about what was disputed. Nothing here recomputes a
    verdict: a second measurement, taken after the model or the prompt moved, printed in the
    same column as the first is what makes an artifact unfalsifiable — and it is the engine
    grading its own output, which is #2269's objection.

    `zero_finding` is present exactly when nothing was disputed in the window. The empty
    case is the common one on this box, and an empty table is the reading that gets a
    later agent to conclude there is nothing to look at — so the zero is written as a
    sentence with its denominators in it, including how many turns the cue screen ever
    reached.
    """
    if verdicts is None:
        verdicts, map_reason = read_verdict_map(table)
    elif not verdicts:
        # An EMPTY map is not a measurement, whoever hands it over. `read_verdict_map` calls a
        # table's own empty map unavailable, and a caller that passed `{}` through the override
        # would otherwise print "available, 0 turns, 0 disputed" over a table whose persisted
        # map records disputes — the exact mis-wire #2717's review caught when `refresh_window`
        # handed `main()` an empty dict. The shape is refused, not the value.
        verdicts = None
        map_reason = (f"an empty per-turn dispute-verdict map was handed over, which records "
                      f"no turn and is therefore no reading of "
                      f"`{uptake.DISPUTE_MAP_KEY}`")
    else:
        map_reason = ""
    rows = [row for row in table.get("entries", [])
            if row.get("kind") in RETRIEVED_KINDS]
    cands = candidates(table, verdicts)
    disputed = [c for c in cands if c["disputed_run_count"] > 0]
    eligible = [c for c in cands if c["eligible"]]
    corpus = table.get("corpus") or {}
    turns = int(corpus.get("turns") or 0)
    screened = int(corpus.get("classified_turns") or 0)

    excluded = {}
    for kind, why in EXCLUDED_KINDS.items():
        n = sum(1 for row in table.get("entries", []) if row.get("kind") == kind)
        excluded[kind] = {"rows": n, "why": why}

    artifact: dict = {
        "schema": 1,
        "item": "2717",
        "generated_on": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_table": source_table,
        "window_days": int(window_days),
        "probe_timestamp": table.get("probe_timestamp"),
        "verdicts": VERDICTS_UNAVAILABLE if verdicts is None else "per-turn",
        "eligibility_rule": {
            "min_disputed_runs": MIN_DISPUTED_RUNS,
            "source": "#2717 risk rule: never narrow on a single dispute",
            "classifier_precision": (table.get("classifier") or {}).get("precision"),
            "classifier_recall": (table.get("classifier") or {}).get("recall"),
        },
        "counts": {
            "retrieved_rows": len(rows),
            "rows_by_kind": {kind: sum(1 for r in rows if r["kind"] == kind)
                             for kind in RETRIEVED_KINDS},
            "rows_with_presence_ids": sum(1 for r in rows if r.get("presence_turn_ids")),
            "excluded_rows": excluded,
            # Two denominators the artifact printed without: the window's own
            # engine-unanswered count (a `None` verdict is not a cleared turn, and a
            # "no disputes" line is a weaker claim when a third of the window was never
            # answered for) and the number of retrieved rows no map covers at all.
            "window_engine_unclassified": corpus.get("engine_unanswered"),
            # How much of the window the map leaves unanswered: a turn the cue screen
            # reached on which the classifier returned NOTHING is neither disputed nor
            # cleared, so a two-dispute floor over a largely unanswered window is a floor
            # applied to almost nothing. `None`, not 0, when the table carries no map.
            "map_unanswered": (sum(1 for v in verdicts.values() if v is None)
                               if verdicts is not None else None),
            "rows_without_a_verdict": sum(1 for c in cands
                                          if not c["disputed_run_ids"]
                                          and not c["unresolved_run_ids"]
                                          and not c["unscreened_run_ids"]),
        },
        "rates": {
            "disputed_retrieved_rows": _rate(len(disputed), len(rows)),
            "rows_with_two_or_more_disputed_runs": _rate(len(eligible), len(rows)),
            "disputed_turns_over_window": _rate(int(corpus.get("disputes") or 0), turns),
            "screened_share_of_window": _rate(screened, turns),
        },
        "candidates": cands,
        #: Clause 4's second half: the floor reported SEPARATELY per attribution family. The
        #: two families are not interchangeable evidence — a prefetched note's disputed runs
        #: are runs the item was actually retrieved into, while an `always_in_force` loaded
        #: index line is in every prompt by construction, so its rate is an upper bound that
        #: no per-run signal can falsify. Reported only as a pooled number, "N of M rows pass
        #: the floor" hides the case that matters: a floor that clears every upper-bound row
        #: while excluding every honestly-attributed one would look like a working filter.
        "eligibility_by_attribution": _eligibility_by_attribution(cands),
        # The map's own state, so whoever reads a rate can see what produced its numerator.
        "verdict_map": ({
            "available": True, "source_key": uptake.DISPUTE_MAP_KEY,
            "turns_in_map": len(verdicts),
            "disputed_turns": sum(1 for v in verdicts.values() if v is True),
            "engine_unanswered": sum(1 for v in verdicts.values() if v is None),
            "keyed_on": (table.get("corpus") or {}).get("verdict_keyed_on"),
        } if verdicts is not None else {
            "available": False, "source_key": uptake.DISPUTE_MAP_KEY,
            "turns_in_map": 0, "disputed_turns": None, "engine_unanswered": None,
            "keyed_on": (table.get("corpus") or {}).get("verdict_keyed_on"),
        }),
        # Artifact-level, because a reader who sees 198 rows with empty dispute columns has
        # no way to find a per-row reason, and none of those rows may read as "0 disputes".
        "unavailable_finding": (None if verdicts is not None else {
            "why": "no-persisted-verdict-map", "table": source_table,
            "statement": (f"{source_table}: UNAVAILABLE — {map_reason} The "
                          f"key is `{uptake.DISPUTE_MAP_KEY}`, which the probe writes into "
                          "every table it emits now, so a table without one predates that "
                          "change and no per-run rate in this artifact is measured."),
        }),
    }
    if not disputed:
        artifact["zero_finding"] = {
            "disputed_retrieved_rows": 0,
            "retrieved_rows": len(rows),
            "window_turns": turns,
            "screened_turns": screened,
            "statement": (
                f"0 of {len(rows)} retrieved rows appeared in a dispute-flagged run over "
                f"the {int(window_days)}-day window ({turns} turns, of which "
                f"{screened} were reached by the dispute cue screen). This is a finding, "
                "not a clean bill: classifier recall is 0.43, so the flagged count is a "
                "lower bound and no narrowing may be proposed from this window."),
        }
    return artifact


def render(artifact: Mapping[str, Any]) -> str:
    """The same numbers as text, for a run that prints and changes nothing."""
    counts = artifact["counts"]
    lines = [f"memory-revision candidates [{artifact['generated_on']}] "
             f"window={artifact['window_days']}d source={artifact['source_table']}"]
    vm = artifact["verdict_map"]
    # Printed ONCE, and the unavailable case is not silent: an artifact whose dispute columns
    # are empty because the measurement is missing has to say so on the same line a reader
    # would otherwise skim as "no disputes".
    lines.append("  per-run verdicts: " + (
        f"from the persisted map `{vm['source_key']}` in {artifact['source_table']} — "
        f"{vm['turns_in_map']} turns, {vm['disputed_turns']} disputed, "
        f"{vm['engine_unanswered']} engine-unanswered"
        if vm["available"] else
        f"UNAVAILABLE — `{vm['source_key']}` absent from {artifact['source_table']}"
        f" ({artifact['verdicts']})"))
    if artifact.get("unavailable_finding"):
        lines.append(f"  {artifact['unavailable_finding']['statement']}")
    lines.append(f"  retrieved rows: {counts['retrieved_rows']} "
                 f"({counts['rows_by_kind']})")
    for key, rate in artifact["rates"].items():
        lines.append(f"  {key}: {rate['numerator']}/{rate['denominator']} "
                     f"= {rate['value']}")
    for family, block in artifact["eligibility_by_attribution"].items():
        rate = block["rate"]
        lines.append(f"  eligible[{family}] ({block['attribution']}): "
                     f"{rate['numerator']}/{rate['denominator']} = {rate['value']}")
    if "zero_finding" in artifact:
        lines.append(f"  ZERO FINDING: {artifact['zero_finding']['statement']}")
    for cand in artifact["candidates"]:
        mark = "eligible" if cand["eligible"] else "INELIGIBLE"
        lines.append(f"  [{mark}] {cand['item']} present_in={cand['present_in_turns']} "
                     f"disputed_runs={cand['disputed_run_count']} "
                     f"upper_bound={cand['attribution_upper_bound']}")
    return "\n".join(lines)


def artifact_path(out_dir: Path, as_of: str) -> Path:
    return Path(out_dir) / f"{ARTIFACT_PREFIX}{as_of}.json"


def write_artifact(artifact: Mapping[str, Any], out_dir: Path) -> Path:
    path = artifact_path(Path(out_dir), artifact["generated_on"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def load_table(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def newest_table_name(table_dir: "Path | None" = None) -> str:
    """The newest committed uptake table, named — not parsed, not joined.

    Same newest-wins rule as `uptake.latest_table`. The directory is read off the module
    attribute at call time rather than captured at import, because that attribute is what a
    fixture repoints; a caller that imported the path itself would read the real artifacts no
    matter what the test said. Raises `FileNotFoundError` naming the directory it looked in,
    so "no table exists" cannot be mistaken for "the table is empty".
    """
    root = Path(table_dir or uptake.UPTAKE_ARTIFACT_DIR)
    names = sorted(p.name for p in root.glob("uptake-*.json"))
    if not names:
        raise FileNotFoundError(
            f"no uptake table (uptake-*.json) in {root}; the probe writes one only while its "
            "hand-labelled dispute corpus resolves, so until someone re-labels there is "
            "nothing to join")
    return names[-1]


def resolve_table_path(table: "str | Path | None" = None) -> Path:
    """The one file this script reads: the named table, else the newest committed one.

    Both CLI spellings end up here, which is the point. `--days` never walked the session
    store, so the old `main()` labelled an artifact read from a file on disk `live:30d`, and
    `unavailable_finding.table` then named a source no process had opened. One resolver, one
    name, one file.
    """
    if table:
        return Path(str(table))
    return Path(uptake.UPTAKE_ARTIFACT_DIR) / newest_table_name()


def refresh_window(days: int = 0,
                   *, table: "str | Path | None" = None) -> tuple[dict, "dict | None"]:
    """The newest (or named) uptake table, and the verdict map THAT TABLE persisted.

    `(table, map-or-None)`. The map is read out of the artifact by `read_verdict_map`; this
    function never computes one, and the distinction is #2717 clause 2:

    * the previous version returned `(table, {})` and `main()` fed that empty dict to
      `build_artifact` as if it were a map, so the default `--days` path printed "available, 0
      turns, 0 disputed" over a table whose own map recorded disputes — a wrong answer in a
      measured-looking shape, green because no node covered `main()` without `--table`;
    * a verdict recomputed here, against a model or prompt that may have moved, would be a
      DIFFERENT measurement from the one the table reports, printed in the same column as the
      first, and #2269's objection stands: the engine that wrote the disputed turns is the one
      that would decide they were disputed. The refused round called
      `uptake.classify_dispute(turn)` for exactly that purpose and died on a `TypeError` — the
      fix is not a corrected call, it is no call;
    * `None` is the honest answer for every committed table: all three predate
      `uptake.DISPUTE_MAP_KEY`, and with `labels/hand-2026-09-11.json` at `status: spent` and
      0 of 46 of its turns resolving against today's transcript store, the probe writes no new
      table at all (`scripts/uptake_probe.py:633-646`). Naming that is the deliverable.

    `days` survives only as the window LABEL stamped on the artifact; the window's real
    numbers come from the table's own `corpus` block.
    """
    del days  # label only; the window's numbers come from the table's corpus block
    doc = load_table(resolve_table_path(table))
    return doc, read_verdict_map(doc)[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only #2717 step 1: retrieved items joined to dispute verdicts.")
    parser.add_argument("--table", default=None,
                        help="one committed uptake table to read instead of the newest one")
    parser.add_argument("--days", type=int, default=30,
                        help="window LABEL to stamp on the artifact (default 30); the numbers "
                             "always come from the table's own corpus block")
    parser.add_argument("--out-dir", default=None,
                        help="defaults to eval/%s under the data root, never the checkout"
                             % ARTIFACT_DIR_NAME)
    parser.add_argument("--as-of", default=None,
                        help="date the artifact is named for (default: today, UTC)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the report and write no file")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # One path for both spellings. The named file, or the newest committed one; and the
    # verdict map is whatever THAT FILE persisted (None when it never wrote one, which is
    # what makes the artifact say `unavailable` rather than "0 disputes").
    path = resolve_table_path(args.table)
    table, flags = refresh_window(args.days, table=path)
    source = path.name
    # A committed table's own corpus is the window it measured; `--days` is a label over it,
    # not a re-measurement, so say which rather than stamping today's date over a probe that
    # ran weeks ago.
    window = args.days

    as_of = args.as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    artifact = build_artifact(table, window_days=window, as_of=as_of,
                              source_table=source, verdicts=flags)
    print(render(artifact))
    if args.dry_run:
        print("  (dry run: nothing written)")
        return 0
    path = write_artifact(artifact, Path(args.out_dir or default_out_dir()))
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
