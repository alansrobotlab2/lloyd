#!/usr/bin/env python3
"""Relabel the pre-#1710 staged bench-mine notes as uncalibrated (#1769).

A staged candidate note is two documents in one file. The FIRST front matter
block is `workers/sources/_common.py::write_staging_note`'s envelope — `source`,
`confidence`, `review_status`, `rationale`, `generated_at`, and the
`calibration` block — and the mined bench TASK, the thing a human actually
promotes, sits in the body.

Until #1710 the calibration ran against the envelope: no `prompt`, no
`objective_checks`, so the objective layer was awarded by default and the mean
of the ten trials described a document the task never was. The verdict that
measurement produced was nevertheless stamped onto the note as
`calibration.status: ok` with `in_band` true or false, and
`GET /api/workers/pending` hands the whole envelope to the promotion UI — so a
human reading the queue sees "this task's mean composite is inside the
capability edge" about a task the trials never read. #1710 fixed the writer: a
note whose `calibration` block carries a `task_id` records what its mean
describes. Every note staged before that commit lacks the key, which is what
makes the stale set closed and enumerable rather than growing.

This script relabels exactly those notes and nothing else:

  * `review_status: uncalibrated` — no band verdict is being claimed;
  * `calibration.status: stale_envelope` — what the trials were run against;
  * `calibration.in_band: null` — the verdict is void, not false. A void is not
    the same statement as `false`, which would keep a task out of the bench on
    the strength of a measurement that measured nothing;
  * `calibration.measured_against: staging envelope (pre-#1710)`.

What it deliberately does NOT do:

  * **Delete the numbers.** The composites, the mean, `runs` and `error` stay in
    the block. What was tried and how it scored is the evidence a re-measurement
    is compared against; the fix for a mislabelled measurement is the label.
  * **Invent a `task_id`.** The trials still were not a measurement of any task,
    and an id here would be the same false claim in a tidier shape. This is why
    the sweep's post-condition is "no note with no `task_id` still carrying a
    `review_status` of `pending` or `out_of_band`" rather than "every note has a
    `task_id`".
  * **Touch a note that HAS a `task_id`.** Four such notes exist, and one of
    them sits at `review_status: pending`. Whether the writer should also stamp
    `measured_against` going forward is owed ruling 2 of #1769 — a decision about
    the writer, not a licence to backfill four notes from a maintenance script.
  * **Move or delete anything.** No `_rejected/`, no re-calibration. Relabelling
    cannot cost a promotion either: `POST /api/workers/pending/promote` never
    reads the staged note's `review_status` (it stamps `promoted` on the LANDED
    copy and unlinks the staged one), and
    `tests/test_workers_router.py::test_an_uncalibrated_note_still_promotes_its_task`
    pins that.
  * **Spend a trial.** Nothing here imports the calibration engine or the model
    servers: this is a front-matter edit over files on disk. To re-measure one of
    these notes on demand, read its BODY, parse it with
    `workers/sources/bench_mine.py::_candidate_frontmatter` — which takes TEXT,
    not a path — and pass the result as the `task=` argument of the calibration
    entry point. Handing that entry point a staged NOTE re-reads the envelope and
    is the #1710 bug again; the route is stated in that module's own docstring.

Modes: `--dry-run` (the default, and the mode a run with no flag is in) writes
nothing at all; `--apply` rewrites in place. Passing both is a usage error,
because a run whose mode is decided by whichever flag came last is unauditable.
`--root` points at another staging tree; with no flag the root is the one the
writer writes, `app.paths.VAULT_PENDING_RESEARCH_DIR / "bench-mine"`, so from
the production checkout it is `~/lloyd-data/_pipeline/vault-derived/
pending-research/bench-mine` and from a worktree or a test run with `LLOYD_DATA`
set it is that isolated root — which is what keeps a forgotten `--root` from
quietly writing live data.

Exit 0: the run made a claim it can back. Exit 2: no staging root at that path
(a 0-note scan is not a clean bill of health, it is a missing denominator), or an
`--apply` after which some note STILL asserts a band verdict — the post-check
re-reads the same sweep the item states, so the script grades its own work rather
than reporting the number of writes it attempted.

That last clause is what #1873 tightened. The post-check used to parse every note
with PyYAML and `continue` past the ones that would not parse, which made "0
notes still assert a verdict" mean "0 PARSEABLE notes assert one" — a certificate
with a hole exactly the shape of the note that most needs one. An unreadable note
is now read line by line instead (`asserts_band_verdict`), and a note no reading
can clear is named and fails the run: `SKIP` is a report, never a clean bill.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

#: The values #1769 names, one source each. A drift between these and what the
#: promotion queue expects is a failing test, not a note that reads three ways.
REVIEW_STATUS = "uncalibrated"
CALIBRATION_STATUS = "stale_envelope"
MEASURED_AGAINST = "staging envelope (pre-#1710)"

#: The `review_status` values that read to a human as a band verdict. `pending`
#: says "promote it, it is at the edge"; `out_of_band` says "drop it, it is not".
#: Both are claims about a measurement, so both have to go on a note whose
#: measurement was of the envelope.
STILL_ASSERTING = ("pending", "out_of_band")

#: The leaf the bench-mine source stages into (`workers/sources/bench_mine.py`'s
#: own `NAME`), and the source directory `GET /api/workers/pending` lists.
SOURCE = "bench-mine"


# ---------------------------------------------------------------- parsing --

def split_front_matter(raw: str):
    """`(envelope_text, body)` for a front-matter note, or None.

    The same splice `bench_mine._record_calibration` performs on the same files:
    the note must open with `---`, and the body is everything after the closing
    fence. Splitting rather than round-tripping the whole file through
    `yaml.safe_load`/`yaml.dump` is the point — the body carries the candidate's
    own front matter block and prose, and a re-dump of the whole document would
    reflow exactly the text a human promotes.
    """
    if not raw.startswith("---"):
        return None
    end = raw.find("\n---\n", 3)
    if end < 0:
        return None
    return raw[3:end], raw[end + 5:]


def is_stale(fm: dict) -> bool:
    """Does this envelope claim a calibration verdict for a task it never named?

    One key decides it: `calibration.task_id` records which task the trials ran
    against, and #1710 made the writer always fill it. A missing block, an empty
    block and a block without the key all read the same way — nothing in the note
    says what the mean is a mean of.
    """
    return "task_id" not in (fm.get("calibration") or {})


class Unreadable(Exception):
    """The note is on disk but cannot be relabelled safely, and why."""


def relabelled_text(raw: str):
    """The note with its stale verdict voided; None when it is already measured.

    Raises `Unreadable` rather than returning a guess: a note whose front matter
    does not parse, or is not a mapping, cannot be relabelled without writing
    bytes nobody authored over the document that is there. Such a note is
    reported by name so a person can look at it.
    """
    split = split_front_matter(raw)
    if split is None:
        raise Unreadable("no closing front matter fence")
    fm_text, body = split
    try:
        fm = yaml.safe_load(fm_text)
    except Exception as exc:
        raise Unreadable(f"front matter does not parse ({type(exc).__name__})") from exc
    if not isinstance(fm, dict):
        raise Unreadable("front matter is not a mapping")
    if not is_stale(fm):
        return None
    cal = fm.get("calibration")
    if cal is not None and not isinstance(cal, dict):
        raise Unreadable(f"calibration block is a {type(cal).__name__}, not a mapping")
    cal = dict(cal or {})
    cal["status"] = CALIBRATION_STATUS
    cal["in_band"] = None
    cal["measured_against"] = MEASURED_AGAINST
    fm["calibration"] = cal
    fm["review_status"] = REVIEW_STATUS
    # `sort_keys=False` keeps the envelope's own key order, so the rewritten half
    # reads as the same envelope with four fields moved rather than as a file
    # someone re-serialised. The body is spliced back verbatim, byte for byte.
    return ("---\n"
            + yaml.dump(fm, default_flow_style=False, allow_unicode=True,
                        sort_keys=False)
            + "---\n" + body)


# ----------------------------------------------------------------- walking --

def iter_notes(root: Path) -> list[Path]:
    """Every staged note under `root`, recursing into each `{yyyy-mm-dd}` leaf.

    `rglob` rather than one fixed depth so a note cannot hide one directory
    deeper and read as absent, and `_`/`.`-prefixed subtrees are left out exactly
    as `GET /api/workers/pending` leaves them out — `_rejected/` is not this
    queue's surface, and `README.md` is not a note.

    This is the ONE walk over the tree, and `tests/test_relabel_stale_bench_calibration.py`
    sweeps through it rather than through its own `rglob`: with two walkers, the
    check that grades the run and the run itself can disagree on which files exist
    — a `README.md` carrying valid front matter, `review_status: pending` and no
    `calibration.task_id` is a note to a bare `rglob` and not a note here, so the
    apply exits 0 while the check that is supposed to certify it reports one
    (#1769 blind spot 2, #1873 clause 3). The name match is `p.name == "README.md"`
    on purpose, the same test `app/routers/workers.py` applies to the same tree.
    """
    out = []
    for p in sorted(root.rglob("*.md")):
        if p.name == "README.md":
            continue
        if any(part.startswith(("_", ".")) for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return out


def envelope_of(raw: str) -> str:
    """The FIRST front matter block's text — the only place a band verdict lives.

    A note with no closing fence has no body either, and all of its bytes are the
    envelope half: that truncated shape is one of the two #1873 refuses to
    certify. A file that never opened a fence (`README.md`) has no envelope at
    all, so nothing in its prose can read as a verdict.
    """
    split = split_front_matter(raw)
    if split is not None:
        return split[0]
    return raw if raw.startswith("---") else ""


def asserts_band_verdict(raw: str) -> bool:
    """Would a reader of these bytes see a band verdict no task stands behind?

    One predicate, read two ways in this order, and it is the ONLY reading of the
    question in this file — `still_asserting` and `relabel_tree` both call it, so
    a note cannot be a SKIP on one side and clean on the other:

      1. PyYAML's reading of the envelope, when it yields a mapping: no
         `calibration.task_id` (#1710's key) and a `review_status` in
         `STILL_ASSERTING`. This is the reading the item's own independent sweep
         makes, so a nested key or a multi-line scalar cannot invent a verdict.
      2. When PyYAML gives no mapping — a front matter block that does not parse,
         a truncated note with no closing fence, an envelope whose top level is a
         list — a scan of the envelope's own lines: a `review_status:` naming one
         of `STILL_ASSERTING`, with no `task_id:` line anywhere in the envelope.

    Half 2 is #1769's blind spot. The old `still_asserting` wrapped half 1 in
    `except Exception: continue`, so every note half 2 exists for was skipped,
    never entered `before` or `after`, and the exit-0 post-check counted the file
    it could not read as a note that had been fixed.
    """
    env = envelope_of(raw)
    if not env.strip():
        return False
    try:
        fm = yaml.safe_load(env)
    except Exception:
        fm = None
    if isinstance(fm, dict):
        return is_stale(fm) and fm.get("review_status") in STILL_ASSERTING
    seen_task_id = False
    found_verdict = False
    for line in env.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.split(" #")[0].strip()
        if key == "task_id":
            seen_task_id = True
        elif key == "review_status" and value in STILL_ASSERTING:
            found_verdict = True
    return found_verdict and not seen_task_id


def still_asserting(root: Path) -> list[Path]:
    """Notes with no `calibration.task_id` whose `review_status` still claims a
    band verdict — #1769's check, read from disk, and the ONLY denominator this
    script reports.

    Both the dry-run's "what would change" line and the post-`--apply` check call
    THIS function, so the number the script reports as fixed is the number it
    then verifies, rather than a count of writes and a count of files that happen
    to agree. Both now call it through `asserts_band_verdict`, which is the
    reading that survives a note PyYAML cannot parse; the `except Exception:
    continue` this function used to carry is what let an unreadable note be
    counted as clean (#1769 blind spot 1, #1873 clause 1).
    """
    out = []
    for p in iter_notes(root):
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if asserts_band_verdict(raw):
            out.append(p)
    return out


def relabel_tree(root: Path, *, apply: bool) -> list[dict]:
    """Relabel every stale note under `root`. Returns one record per note.

    A record's `action` is `relabelled`, `already`, `measured` or `skipped`;
    `already` means the note carries the label already and its bytes came back
    identical, which is what makes a re-run open nothing for writing. Nothing is
    written unless `apply`, and with `apply` a note is written only when the
    rewrite actually differs from what is on disk.

    A note that cannot be relabelled is `skipped` and stays byte-identical —
    inventing front matter over a document nobody wrote is worse than an
    unlabelled note. What #1873 clause 1 changes is what a skip may PROVE: when
    the note's own envelope still reads as a band verdict with no task behind it,
    the record carries `asserting=True`, and `main` turns that into exit 2. Before
    that, an unreadable note was a `SKIP` line under a `post-check OK` line and
    `return 0`, so the certificate the item describes covered every note except
    the one it could not check.
    """
    records: list[dict] = []
    for p in iter_notes(root):
        rec = {"path": p, "action": "skipped", "why": "", "asserting": False}
        records.append(rec)
        # The same bytes, read through the same predicate, as `still_asserting` —
        # one note, one denominator, so a skip here can never be a clean reading
        # there. Held in a local, not on the record: a note that started out
        # asserting and was successfully relabelled has nothing left to assert.
        asserting = False
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
            asserting = asserts_band_verdict(raw)
        except OSError as exc:
            rec["why"] = f"unreadable ({type(exc).__name__})"
            continue
        try:
            new = relabelled_text(raw)
        except Unreadable as exc:
            rec["why"] = f"unreadable ({type(exc).__name__}): {exc}"
        except Exception as exc:                                    # noqa: BLE001
            rec["why"] = f"relabel failed ({type(exc).__name__})"
        else:
            if new is None:
                rec.update(action="measured",
                           why="calibration.task_id is present")
                continue
            if new == raw:
                rec.update(action="already",
                           why="labelled already; bytes identical")
                continue
            if not apply:
                rec.update(action="relabelled", why="dry-run: nothing written")
                continue
            try:
                p.write_text(new, encoding="utf-8")
            except OSError as exc:
                rec["why"] = f"write failed ({type(exc).__name__})"
            else:
                rec.update(action="relabelled", why="")
                continue
        if asserting:
            # Left on disk still claiming a verdict it cannot name: a write
            # failure or an envelope no reader can parse. `main` fails the run.
            rec["asserting"] = True
            rec["why"] += "; still asserting a band verdict it cannot name"
    return records


# ------------------------------------------------------------------ main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="relabel_stale_bench_calibration.py",
        description="Relabel staged bench-mine notes whose calibration block "
                    "records no task_id (#1769). Edits front matter only; never "
                    "runs a trial.")
    ap.add_argument("--root", default="",
                    help=f"staging root to relabel (default: the writer's own "
                         f"pending-research/{SOURCE} root)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="report what would change; write nothing (default)")
    mode.add_argument("--apply", action="store_true",
                      help="rewrite the stale notes in place")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser() if args.root else _default_root()
    if not root.is_dir():
        print(f"relabel: no staging root at {root} — 0 notes scanned, so nothing "
              "here is being claimed about the queue. Pass --root.")
        return 2

    applying = args.apply
    print(f"relabel: {'APPLY' if applying else 'DRY-RUN (no writes)'} on {root}")
    before = still_asserting(root)
    records = relabel_tree(root, apply=applying)

    for rec in records:
        if rec["action"] == "skipped":
            print(f"  SKIP    {rec['path'].name}: {rec['why']}")
        elif rec["action"] == "relabelled" and rec["why"]:
            print(f"  DRY     {rec['path'].name}: {rec['why']}")
        elif rec["action"] == "relabelled":
            print(f"  RELABEL {rec['path'].name}")

    counted = {"relabelled": 0, "already": 0, "measured": 0, "skipped": 0}
    for rec in records:
        counted[rec["action"]] += 1
    print(f"relabel: scanned {len(records)} note(s); relabelled {counted['relabelled']}, "
          f"already uncalibrated {counted['already']}, already measured "
          f"(has calibration.task_id) {counted['measured']}, unreadable/skipped "
          f"{counted['skipped']}")

    # A note this run could not relabel and could not clear: named whether or
    # not the sweep can read it either, so the SKIP line is never the last word.
    left = [rec for rec in records if rec["asserting"]]
    if left:
        print(f"relabel: {len(left)} note(s) could not be relabelled and STILL "
              f"assert a band verdict (no calibration.task_id, review_status in "
              f"{list(STILL_ASSERTING)}) "
              f"{'after --apply' if applying else 'in this dry run'}: "
              f"{sorted(str(r['path'].relative_to(root)) for r in left)[:20]}")

    after = still_asserting(root)
    if not applying:
        print(f"relabel: dry-run check — {len(before)} note(s) with no "
              f"calibration.task_id still assert a band verdict "
              f"(review_status in {list(STILL_ASSERTING)}); nothing written")
        return 0
    if after or left:
        unresolved = {str(p.relative_to(root)) for p in after}
        unresolved |= {str(r["path"].relative_to(root)) for r in left}
        print(f"relabel: POST-CHECK FAILED — {len(unresolved)} note(s) still "
              f"assert a band verdict after --apply: {sorted(unresolved)[:20]}")
        return 2
    print(f"relabel: post-check OK — 0 notes with no calibration.task_id carry "
          f"review_status in {list(STILL_ASSERTING)} (was {len(before)})")
    return 0


def _default_root() -> Path:
    """The tree the writer writes: `app.paths.VAULT_PENDING_RESEARCH_DIR/bench-mine`.

    Resolved lazily and only when `--root` is absent, the way
    `scripts/maintenance/corpus_shape.py` resolves its own roots, so importing
    this module — which its test file does — cannot touch any live data root.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.paths import VAULT_PENDING_RESEARCH_DIR
    return VAULT_PENDING_RESEARCH_DIR / SOURCE


if __name__ == "__main__":
    raise SystemExit(main())
