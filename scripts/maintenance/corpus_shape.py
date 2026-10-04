#!/usr/bin/env python3
"""Cross-run aggregate shape of four nightly-written corpora (#761, split from #543).

Per-write scoring (#717) looks at one artifact at a time, so it structurally
cannot see an artifact that is fine alone and wrong because it recurs. The fact
store has its own dated snapshots (`kg-health-*.json`); these four did not:

  daily       `## ` sections of the flat daily notes `memory/YYYY-MM-DD.md`
  user_memory top-level `- ` entries of `lloyd/USER.md` and `lloyd/MEMORY.md`
  skills      `skills/*/SKILL.md` bodies (front matter stripped)
  trajectories rows of `_pipeline/trajectories/YYYY-MM-DD.jsonl`

(Tool-call `summary` captions, #543's fifth corpus, are not persisted anywhere a
reader can reach, so they are not measured.)

Per corpus: n, length mean and p95, distinct-value ratio of the key column,
exact-duplicate rate on a content hash, and self-reference rate — the share of
items that name the artifact they live in (a daily-note section talking about
daily notes, a USER.md entry about USER.md, a trajectory row from a session that
was reading trajectories). The prose corpora are also scanned for a sentence that
recurs across items, which is the planted-regression shape #543 asked for.

THE SERIES HOLDS ONE ROW PER UTC DATE, AND THIS SCRIPT IS ITS ONLY WRITER. The
row key is the UTC date of the run's resolved `--now`, not its second: each new
row is opened with O_EXCL (a stamp already taken gets a numeric suffix rather
than an overwrite), and a run whose day already has a row writes nothing. A retry
inside a day therefore cannot append a second row and become the next night's diff
base (#1576) — before that guard, three rows written 39 s apart on 2026-09-24 sat
in the series and the 09-25 run diffed against one of them. The row a run diffs
against is always the newest row from an EARLIER date, and it is read before this
run creates anything, so no run can diff against — or gate on — a file it wrote
itself. That is the `graph-baseline.json` failure #543 named: a baseline the run
it judges rewrites. Nothing is derived from git HEAD either, for the same reason.
One row per day bounds duplicates, not gaps: a day that did not run leaves the
next run's base older than one day.

THE BOUNDS BELOW ARE MEASURED, NOT GUESSED (#2200). Each is the largest
day-over-day delta its metric actually showed over the first 8 clean UTC dates of
this series — 2026-09-27 to 2026-10-04, one row per date now that #1576 keys the row
to the date — rounded up to the next 0.1 (relative) or 0.05 (absolute) step, and
never below the bound already in force. A night inside that noise stays silent; a
move past a bound is still a prompt to look rather than a verdict, because it says
the shape changed, not that the change is bad.

A FIFTH COUNT, KEPT WITH SOMEBODY ELSE'S CLASSIFIER (#2039). The four corpora
above are measured with this file's own metrics, and for the ten YouTube digests
under `knowledge/*/youtube-digest.md` that is the wrong tool: the defect worth
catching there is an entry body that rates an item against the reader's interest
profile instead of describing it, and the thing that knows what such a sentence
looks like is `intel_pipeline.body.is_interest_profile_prose` — the shipped guard
`vault_writer` applies on the write path. #2011 found 27 published instances of it
after a hand-written acceptance grep had reported the corpus clean, catching 10 of
the 27, so the count below calls the guard and never re-spells its phrasing: a
phrase pattern here would be the instrument that already lied. It is reported per
digest file, and it is a finding in its own right rather than a threshold — one
flagged line exits 2, because "0 flagged" is the whole claim and a number that
cannot fail is not a check.

Exit 0: nothing to report. Exit 2: a metric moved past its threshold, a sentence
recurs that did not recur in the earlier day's row, or a digest line is flagged by
that guard. `--quiet` prints nothing on exit 0, so a scheduled run speaks only when
it has something to say; in quiet mode a flagged line is still printed, named with
its file and its line number, and the per-digest counts are not.
Read-only apart from the day's one new JSON — and apart from nothing at all on a
same-day retry, which writes no row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

CORPORA = ("daily", "user_memory", "skills", "trajectories")
PROSE = ("daily", "user_memory", "skills")
FILE_PREFIX = "corpus-shape-"
STAMP_FMT = "%Y%m%dT%H%M%SZ"

# metric -> (kind, bound). "rel": |new-old|/old; "abs": |new-old|.
#
# Recalibrated 2026-10-04 (#2200) from the first 8 clean UTC dates of the live
# series — 2026-09-27 to 2026-10-04, one row per date since #1576 keyed the row to
# the date, so 7 adjacent day-over-day pairs across all four corpora. Each bound is
# the largest one-night move its metric actually made, rounded UP to the next 0.1
# (rel) or 0.05 (abs) step, and never below the bound already in force:
#   metric                max delta    corpus / pair that produced it
#   n                      0.5366 rel  daily 2026-10-01->10-02 (41 -> 19)
#   len_mean               0.1977 rel  skills 2026-10-01->10-02 (11977.9 -> 14345.7)
#   len_p95                0.2623 rel  trajectories 2026-10-01->10-02 (51473 -> 64972)
#   distinct_key_ratio     0.0885 abs  daily 2026-10-01->10-02 (0.122 -> 0.2105)
#   duplicate_rate         0.1117 abs  daily 2026-10-01->10-02 (0.5854 -> 0.4737)
#   self_reference_rate    0.0361 abs  skills 2026-09-30->10-01 (0.0833 -> 0.1194)
# user_memory's deltas are zero except n (<= 0.0216) and len_mean (<= 0.038), so the
# whole floor here is daily, skills and trajectories — which is also why one global
# table ends up wide where the frozen corpus needs nothing at all; whether the table
# should split per corpus is owed, and is not settled by this constant.
# Under the table this replaced, 6 of these 8 nights printed a MOVED line, which is
# what `duplicate_rate` leaving abs 0.02 and `distinct_key_ratio` leaving abs 0.05
# is for: both sat below an ordinary night. Every bound is asserted against these
# same numbers as OBSERVED_MAX_DELTA in tests/test_corpus_shape.py, so the constants
# and their provenance cannot drift apart.
THRESHOLDS = {
    "n": ("rel", 0.6),
    "len_mean": ("rel", 0.25),
    "len_p95": ("rel", 0.5),
    "distinct_key_ratio": ("abs", 0.10),
    "duplicate_rate": ("abs", 0.15),
    "self_reference_rate": ("abs", 0.05),
}
# A sentence in at least this many distinct items of one corpus is reported.
MIN_REPEATS = 3
MIN_SENTENCE_CHARS = 30

_DAILY_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(md|jsonl)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_BOLD_LEAD = re.compile(r"^\*\*(.+?)\*\*")


def _item(corpus, path, key, text, self_ref):
    return {"corpus": corpus, "file": str(path), "key": key, "text": text,
            "self_ref": bool(self_ref)}


def _dated_files(directory: Path, suffix: str, since, until):
    if not directory.is_dir():
        return []
    out = []
    for p in sorted(directory.iterdir()):
        m = _DAILY_NAME.match(p.name)
        if not m or not p.name.endswith(suffix):
            continue
        day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if since <= day <= until:
            out.append(p)
    return out


def _strip_front_matter(text: str) -> str:
    return _FRONT_MATTER.sub("", text, count=1)


def collect_daily(vault: Path, since, until):
    items = []
    for p in _dated_files(vault / "memory", ".md", since, until):
        body = _strip_front_matter(p.read_text(errors="replace"))
        for chunk in re.split(r"(?m)^## ", body)[1:]:
            heading, _, text = chunk.partition("\n")
            items.append(_item("daily", p, heading.strip(), text.strip(),
                               re.search(r"daily[- ]note|memory/\d{4}-\d{2}-\d{2}", text, re.I)))
    return items


def collect_user_memory(vault: Path):
    items = []
    for name in ("USER.md", "MEMORY.md"):
        p = vault / "lloyd" / name
        if not p.is_file():
            continue
        body = _strip_front_matter(p.read_text(errors="replace"))
        entry = None
        for line in body.splitlines():
            if line.startswith("- "):
                if entry is not None:
                    items.append(entry)
                text = line[2:].strip()
                lead = _BOLD_LEAD.match(text)
                entry = _item("user_memory", p, lead.group(1) if lead else text[:60], text,
                              name.lower() in text.lower())
            elif entry is not None and line.startswith((" ", "\t")) and line.strip():
                entry["text"] += " " + line.strip()
                entry["self_ref"] = entry["self_ref"] or name.lower() in line.lower()
            else:
                if entry is not None:
                    items.append(entry)
                entry = None
        if entry is not None:
            items.append(entry)
    return items


def collect_skills(vault: Path, since_ts: float):
    items = []
    root = vault / "skills"
    if not root.is_dir():
        return items
    for p in sorted(root.glob("*/SKILL.md")):
        if p.stat().st_mtime < since_ts:
            continue
        body = _strip_front_matter(p.read_text(errors="replace")).strip()
        items.append(_item("skills", p, p.parent.name, body,
                           f"skills/{p.parent.name}/" in body))
    return items


def collect_trajectories(pipeline: Path, since, until):
    items = []
    for p in _dated_files(pipeline / "trajectories", ".jsonl", since, until):
        for line in p.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = {}
            items.append(_item("trajectories", p, str(row.get("session_key")), line,
                               "trajectories" in line))
    return items


#: The digest corpus (#2039): one entry body per published item, ten category
#: digests, written by the intel pipeline's `vault_writer` on every run.
DIGEST_GLOB = "knowledge/*/youtube-digest.md"

#: The lines an entry is built from AROUND its body, so not prose: a heading, the
#: `---` rule between entries, the source/relevance line, the item's link. Skipping
#: these is what keeps the count about bodies — a `**Source:**` line or a heading
#: carrying a rubric sentence is structure the writer does not choose, and counting
#: it would train the reader to ignore the alert.
DIGEST_STRUCTURAL_PREFIXES = ("#", "---", "**Source:**", "[Link]")

#: `scripts/intel-pipeline`, the package that owns the guard. This script is
#: stdlib-only and lives one directory over, so reaching it is a `sys.path` insert
#: rather than an import — the same seam `main` already uses for `app.paths`.
_INTEL_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "intel-pipeline"
_interest_profile_guard = None


def _classify_interest_profile(text: str) -> bool:
    """Ask the SHIPPED guard, do not re-implement it (#2039).

    `intel_pipeline.body.is_interest_profile_prose` is the same callable
    `vault_writer` consults on the write path, so a line that reaches a digest and a
    line counted here answer to one definition. Re-spelling that definition as a
    phrase list here is the mistake #2011 records: its acceptance grep caught 10 of
    the 27 published sentences and reported the corpus clean.
    """
    global _interest_profile_guard
    if _interest_profile_guard is None:
        if str(_INTEL_PIPELINE_DIR) not in sys.path:
            sys.path.insert(0, str(_INTEL_PIPELINE_DIR))
        from intel_pipeline.body import is_interest_profile_prose
        _interest_profile_guard = is_interest_profile_prose
    return bool(_interest_profile_guard(text))


def interest_profile_sweep(vault: Path, classify=None) -> dict:
    """Per digest: how many lines the guard was asked about, and how many it flagged.

    `lines_checked` is the denominator the guard actually saw — every non-blank line
    that does not start with one of `DIGEST_STRUCTURAL_PREFIXES`, the file's own front
    matter included, since a `tags:` key is neither structure nor prose and can never
    flag. Front matter is not skipped because the sweep is the count #2039 asked for,
    stated line by line, and an undocumented extra skip is how a sweep starts reading
    clean for the wrong reason. Every flag carries the line number it sits on,
    because the reader of a finding has to open that line. `classify` is a seam for
    tests to show the count follows the guard they hand it and not some pattern of
    their own; production calls it with none.
    """
    classify = classify or _classify_interest_profile
    files, total = [], 0
    for p in sorted(vault.glob(DIGEST_GLOB)):
        hits, checked = [], 0
        for lineno, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            text = line.strip()
            if not text or text.startswith(DIGEST_STRUCTURAL_PREFIXES):
                continue
            checked += 1
            if classify(text):
                hits.append({"line": lineno, "text": text})
        total += len(hits)
        files.append({"file": str(p.relative_to(vault)), "lines_checked": checked,
                      "flagged": len(hits), "hits": hits})
    return {"glob": DIGEST_GLOB, "skips": list(DIGEST_STRUCTURAL_PREFIXES),
            "files": files, "files_count": len(files), "total_flagged": total}


def interest_profile_lines(report: dict) -> tuple[list, list, bool]:
    """`(per_digest, flagged, any_flagged)` for one measured report.

    Split in two because the two halves have different audiences: the per-digest
    counts are the series a person trends and belong to the non-quiet print, while a
    flag is the thing the task exists to catch and survives `--quiet`. The flagged
    line names the file AND its line number — "1 flagged somewhere in ten digests"
    is not actionable.
    """
    sweep = report.get("interest_profile") or {}
    per_digest, flagged = [], []
    for f in sweep.get("files", []):
        per_digest.append(f"interest_profile [{f['file']}]: "
                          f"lines_checked={f['lines_checked']} flagged={f['flagged']}")
        for h in f["hits"]:
            flagged.append(f"interest_profile: FLAGGED {f['file']}:{h['line']} "
                           f"{h['text']!r}")
    return per_digest, flagged, bool(sweep.get("total_flagged"))


def _vault_default() -> Path:
    """The vault to measure: `LLOYD_OBSIDIAN_VAULT` if set, else `~/obsidian`.

    The same precedence as `tests/board_presence.py::vault_root`, which is how the rest
    of this repo reaches a vault — including a test's redirected one. Before #2039 the
    fallback here was `app.paths.VAULT_ROOT` and nothing else, a plain
    `Path.home() / "obsidian"`, so the variable other modules honour was silently
    ignored at this entry point. The first real injection of a rubric sentence ran with
    that variable pointing at a dirty vault, read an undeclared clean one, printed
    nothing and exited 0 — a check reporting "clean" about a corpus it never opened,
    which is the exact failure #2011 was about.

    The name is `LLOYD_…VAULT`, not a new `LLOYD_…` spelling, because it already existed
    here as an option default and in `board_presence`; a check gets its corpus from one
    place, and a test that redirects it is redirecting the corpus, not the machine.
    """
    raw = os.environ.get("LLOYD_OBSIDIAN_VAULT")
    return Path(raw).expanduser() if raw else Path.home() / "obsidian"


def _pipeline_default() -> Path:
    """`LLOYD_DATA`, else the checkout's `.lloyd-data/_pipeline`, else `~/lloyd-data`.

    Same precedence as `app.paths.resolve_data_root` (which resolves the data root and
    not the `_pipeline` under it, hence the re-spelling): environment first, then the
    git tree this script sits in, then the machine's data root. Consulted only when
    `--pipeline` is absent, which is never in a test.

    It is not `app.paths.PIPELINE_DIR` any more because a round's worktree is a git
    tree too, and `app.paths` deliberately anchors to it — so importing that constant
    made a read-only probe of this script from a worktree point at the worktree's
    empty `.lloyd-data/` and report `n=0` for all four corpora, which reads exactly
    like a corpus that emptied. Falling back to the machine's data root when the
    anchored one holds nothing measures the same box a scheduled run measures; the
    scheduled run's own tree has the directory, so its path does not move.
    """
    data = os.environ.get("LLOYD_DATA")
    if data:
        return Path(data).expanduser() / "_pipeline"
    anchored = Path(__file__).resolve().parents[2] / ".lloyd-data" / "_pipeline"
    return anchored if anchored.exists() else Path.home() / "lloyd-data" / "_pipeline"


def _p95(values):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def shape(items):
    n = len(items)
    if not n:
        return {"n": 0, "len_mean": 0.0, "len_p95": 0, "distinct_key_ratio": None,
                "duplicate_rate": None, "self_reference_rate": None}
    lengths = [len(i["text"]) for i in items]
    hashes = [hashlib.sha1(i["text"].encode()).hexdigest() for i in items]
    return {
        "n": n,
        "len_mean": round(sum(lengths) / n, 1),
        "len_p95": _p95(lengths),
        "distinct_key_ratio": round(len({i["key"] for i in items}) / n, 4),
        "duplicate_rate": round(1 - len(set(hashes)) / n, 4),
        "self_reference_rate": round(sum(i["self_ref"] for i in items) / n, 4),
    }


def recurring_sentences(items, min_repeats=MIN_REPEATS):
    """Sentences appearing in at least `min_repeats` distinct items of one corpus."""
    seen = defaultdict(set)          # sentence -> {item index}
    files = defaultdict(set)
    for idx, item in enumerate(items):
        for raw in _SENTENCE_SPLIT.split(item["text"]):
            s = raw.strip()
            if len(s) < MIN_SENTENCE_CHARS or s.startswith("#"):
                continue
            seen[s].add(idx)
            files[s].add(item["file"])
    found = [{"sentence": s, "items": len(ix), "files": sorted(files[s])}
             for s, ix in seen.items() if len(ix) >= min_repeats]
    return sorted(found, key=lambda r: (-r["items"], r["sentence"]))


def measure(vault: Path, pipeline: Path, now: datetime, days: int):
    until = now.date()
    since = until - timedelta(days=days - 1)
    since_ts = (now - timedelta(days=days)).timestamp()
    by_corpus = {
        "daily": collect_daily(vault, since, until),
        "user_memory": collect_user_memory(vault),
        "skills": collect_skills(vault, since_ts),
        "trajectories": collect_trajectories(pipeline, since, until),
    }
    windows = {
        "daily": f"memory/YYYY-MM-DD.md dated {since}..{until}",
        "user_memory": "current USER.md + MEMORY.md (entries carry no date)",
        "skills": f"SKILL.md modified since {since}",
        "trajectories": f"trajectories/YYYY-MM-DD.jsonl dated {since}..{until}",
    }
    report = {"generated": now.strftime(STAMP_FMT), "days": days,
              "vault": str(vault), "pipeline": str(pipeline), "corpora": {}}
    for name in CORPORA:
        entry = {"window": windows[name], **shape(by_corpus[name])}
        if name in PROSE:
            entry["recurring"] = recurring_sentences(by_corpus[name])
        report["corpora"][name] = entry
    # Outside `corpora` on purpose: its unit is a prose line and its verdict is the
    # shipped guard's, so it shares none of the four shape metrics, and folding it in
    # would put a fifth row through the threshold loop it has no thresholds for.
    report["interest_profile"] = interest_profile_sweep(vault)
    return report


_RUN_NAME = re.compile(re.escape(FILE_PREFIX) + r"(\d{8}T\d{6}Z)(?:-(\d+))?\.json$")


def _run_order(path: Path):
    # By name, `…Z-1.json` sorts BEFORE `…Z.json` ('-' < '.'), which would make
    # a same-stamp rerun's diff read the older file as the newer one.
    m = _RUN_NAME.match(path.name)
    return (m.group(1), int(m.group(2) or 0)) if m else ("", -1)


#: The row key: the date half of `STAMP_FMT`, the UTC date of the `--now` that
#: measured the row.
_ROW_DATE_FMT = "%Y%m%d"


def _stamp_date(stamp: str):
    """The UTC date half of a `STAMP_FMT` stamp."""
    return datetime.strptime(stamp[:8], _ROW_DATE_FMT).date()


def _row_date(path: Path):
    """The UTC date a row is keyed to — the date half of its stamp name."""
    m = _RUN_NAME.match(path.name)
    return _stamp_date(m.group(1)) if m else None


def _rows(out_dir: Path) -> list[Path]:
    """Every well-named row in the out dir, oldest first."""
    if not out_dir.is_dir():
        return []
    return sorted((p for p in out_dir.glob(FILE_PREFIX + "*.json")
                   if _RUN_NAME.match(p.name)), key=_run_order)


def row_for(out_dir: Path, day):
    """The row already written for UTC date `day`, or None.

    A series written before #1576 can hold several rows for one date; the newest
    is that date's row for ordering purposes, and none of them is ever rewritten.
    """
    rows = [p for p in _rows(out_dir) if _row_date(p) == day]
    return rows[-1] if rows else None


def previous_run(out_dir: Path, before=None):
    """The newest row to diff against, or None.

    `before` is the UTC date the calling run is keyed to. A row carrying that date
    belongs to the run's own day — that day's first row, or a retry of it — and is
    never a valid base: it was measured at an earlier hour, whose skills window
    and memory files can differ, and a run diffed against it reports a delta of
    zero whatever the series did overnight (#1576). With `before` unset the newest
    readable row is the base.
    """
    files = _rows(out_dir)
    if before is not None:
        files = [p for p in files if _row_date(p) < before]
    for p in reversed(files):
        try:
            return p, json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    return None


def write_new(out_dir: Path, report: dict) -> tuple[Path, bool]:
    """Create the row for the report's UTC date; never open a row for writing.

    Returns `(row, written)`. When that date already has a row this run is a retry
    inside its own day, and the existing row comes back untouched: the series keeps
    a day's first measurement, so a re-run hours later — whose skills mtime window
    has shifted and whose `USER.md`/`MEMORY.md` may have changed — cannot replace
    the row the following night diffs against, and nothing is opened for writing at
    all (#1576). Skipping rather than replacing is also what lets a row kept
    read-only stay unwritten.
    """
    existing = row_for(out_dir, _stamp_date(report["generated"]))
    if existing is not None:
        return existing, False
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(report, indent=1, sort_keys=True) + "\n"
    base = FILE_PREFIX + report["generated"]
    for n in range(1000):
        path = out_dir / (base + (f"-{n}" if n else "") + ".json")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(data)
        return path, True
    raise RuntimeError(f"no free name for {base} in {out_dir}")


def _moved(metric, old, new):
    if old is None or new is None:
        return False, ""
    kind, bound = THRESHOLDS[metric]
    if kind == "rel":
        if old == 0:
            delta = 0.0 if new == 0 else float("inf")
        else:
            delta = abs(new - old) / abs(old)
        return delta > bound, f"{metric} {old}->{new} ({delta:+.0%} vs {bound:.0%})"
    delta = new - old
    return abs(delta) > bound, f"{metric} {old}->{new} ({delta:+.4f} vs {bound})"


def diff_lines(prev, report):
    """One line per corpus: the threshold verdict, or `no prior run`."""
    lines, moved_any = [], False
    for name in CORPORA:
        cur = report["corpora"][name]
        old = (prev or {}).get("corpora", {}).get(name) if prev else None
        if not old:
            lines.append(f"{name}: no prior run")
            continue
        moves = [desc for m in THRESHOLDS for hit, desc in [_moved(m, old.get(m), cur.get(m))] if hit]
        moved_any = moved_any or bool(moves)
        verdict = "MOVED " + "; ".join(moves) if moves else "within thresholds"
        lines.append(f"{name}: {verdict}")
    return lines, moved_any


def recurring_lines(prev, report):
    """Every recurring sentence, and which of them are new since `prev`.

    Only a NEW recurrence is a finding. Skills share boilerplate by design and a
    standing alert repeats until it is fixed; re-announcing either every night
    teaches the reader to skip the output. The first run is the baseline and
    announces nothing, as #543 specified.
    """
    lines, new_any = [], False
    for name in PROSE:
        before = None
        if prev:
            before = {r["sentence"] for r in
                      prev.get("corpora", {}).get(name, {}).get("recurring", [])}
        for r in report["corpora"][name].get("recurring", []):
            is_new = before is not None and r["sentence"] not in before
            new_any = new_any or is_new
            lines.append(f"{name}: {'NEW ' if is_new else ''}recurring sentence in "
                         f"{r['items']} items [{', '.join(r['files'])}]: {r['sentence']!r}")
    return lines, new_any


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vault", type=Path)
    ap.add_argument("--pipeline", type=Path)
    ap.add_argument("--out-dir", type=Path, help="default <pipeline>/metrics")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--now", help="ISO timestamp to pin the run (UTC if naive)")
    ap.add_argument("--quiet", action="store_true", help="print nothing when there is no finding")
    args = ap.parse_args(argv)

    # An explicit option always wins; the fallbacks are the two the rest of the repo
    # uses, so a redirected vault actually redirects this check. Resolution moved out
    # of `app.paths` entirely (#2039), which also drops the import that anchored a
    # worktree run to the worktree's own empty data dir.
    args.vault = args.vault or _vault_default()
    args.pipeline = args.pipeline or _pipeline_default()
    out_dir = args.out_dir or args.pipeline / "metrics"
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    report = measure(args.vault, args.pipeline, now, args.days)
    day = _stamp_date(report["generated"])      # the row key: resolved --now, in UTC
    prior = previous_run(out_dir, before=day)   # read BEFORE this run writes anything
    lines, moved = diff_lines(prior[1] if prior else None, report)
    recurring, new_recurring = recurring_lines(prior[1] if prior else None, report)
    per_digest, flags, flagged = interest_profile_lines(report)
    row, wrote = write_new(out_dir, report)

    finding = moved or new_recurring or flagged
    if finding or not args.quiet:
        kept = "" if wrote else f" [row for {day} already existed; nothing written]"
        print(f"corpus shape over {args.days} d -> {row}"
              + (f" (vs {prior[0].name})" if prior else "") + kept)
        for name in CORPORA:
            c = report["corpora"][name]
            print(f"  {name} [{c['window']}]: n={c['n']} len_mean={c['len_mean']} "
                  f"len_p95={c['len_p95']} distinct_key_ratio={c['distinct_key_ratio']} "
                  f"duplicate_rate={c['duplicate_rate']} "
                  f"self_reference_rate={c['self_reference_rate']}")
        if not args.quiet:
            for line in per_digest:
                print(line)
        for line in lines + recurring + flags:
            print(line)
    return 2 if finding else 0


if __name__ == "__main__":
    sys.exit(main())
