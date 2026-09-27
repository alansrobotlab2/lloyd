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

The thresholds below are provisional. Real ones need a week of series (a human
clause on #761); until then a move past them is a prompt to look, not a verdict.

Exit 0: nothing to report. Exit 2: a metric moved past its threshold, or a
sentence recurs that did not recur in the earlier day's row. `--quiet` prints
nothing on exit 0, so a scheduled run speaks only when it has something to say.
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
THRESHOLDS = {
    "n": ("rel", 0.5),
    "len_mean": ("rel", 0.25),
    "len_p95": ("rel", 0.5),
    "distinct_key_ratio": ("abs", 0.05),
    "duplicate_rate": ("abs", 0.02),
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

    if args.vault is None or args.pipeline is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from app import paths
        args.vault = args.vault or paths.VAULT_ROOT
        args.pipeline = args.pipeline or paths.PIPELINE_DIR
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
    row, wrote = write_new(out_dir, report)

    finding = moved or new_recurring
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
        for line in lines + recurring:
            print(line)
    return 2 if finding else 0


if __name__ == "__main__":
    sys.exit(main())
