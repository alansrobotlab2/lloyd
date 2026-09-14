#!/usr/bin/env python3
"""Build the labelled durable-write corpus for backlog #580.

A *bad* sample is a real Lloyd durable write (a video-digest note under
``knowledge/youtube/``) that Lloyd later repaired, recovered as it stood
**before** that repair: ``git show <repair>^:<path>``. Its defect class is a
rule whose signature is present in the repair diff and whose name is the word
the repairer itself used in the auto-captured session record (citations in
``DEFECT_CLASSES``). A *good* sample is a matched digest note — same channel,
written in the same window — for which no repair diff ever fired a named class.

Nothing here writes to the vault or to any live write path: the only git
invocation is ``git -C <vault> show/log``, read-only.

Usage
-----
    python eval/durable_write_judge/build_corpus.py --out eval/durable_write_judge/corpus.jsonl
    python eval/durable_write_judge/build_corpus.py --counts eval/durable_write_judge/corpus.jsonl

The second form prints the counts without touching git and is what the tests
run; the first also writes the artifact.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

DEFAULT_VAULT = Path.home() / "obsidian"
NOTE_PREFIX = "knowledge/youtube/"
MAX_TEXT_CHARS = 4000

# Every class name below is the repairer's own word for what it found, taken
# from the auto-captured session records in the daily notes (the vault commit
# messages carry none of them — repairs land inside multi-hundred-file batch
# commits). ``detect`` is the deterministic signature of that defect in the
# repair diff: the class is only assigned when the signature is present.
DEFECT_CLASSES: dict[str, str] = {
    "invented_url": "removed http(s) URL that the repaired version does not "
                    "carry (memory/2026-09-09.md:204 'removing three fabricated "
                    "URLs'; memory/2026-09-08.md:412 'invented non-existent URLs')",
    "false_cutoff_claim": "a 'the transcript cut off' assertion present before "
                          "the repair and absent after it (memory/2026-09-08.md:574 "
                          "'falsely claimed the transcript cut off')",
    "omitted_numerical_data": "no metric-bearing line before the repair, three or "
                              "more after it (memory/2026-09-09.md:342 'missing "
                              "numerical data')",
    "missing_section": "a '## ' section the repaired version adds and the original "
                       "never had (memory/2026-09-09.md:1175 'missing Q&A content')",
    "wrong_metadata": "a title/source/published/video_id/speaker front-matter field "
                      "the repair changed (memory/2026-09-09.md:1709 'incorrect "
                      "metadata')",
}

URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
FRONT_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
FM_KEYS = ("title", "source", "published", "video_id", "speaker")
CUTOFF_RE = re.compile(
    r"cut\s?off|cuts off|transcript (is |was )?(incomplete|truncated)|ends? abruptly",
    re.I,
)
METRIC_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|x\b|ms\b|s\b|GB|MB|tokens|params|k\b|M\b|B\b|\$)", re.I
)
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


def _git(vault: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """One read-only git call. Never a write: show/log/diff only."""
    return subprocess.run(
        ["git", "-C", str(vault), *args], capture_output=True, text=True, timeout=timeout
    )


def show(vault: Path, rev: str, path: str) -> str | None:
    """``git show <rev>:<path>`` or None when the object does not exist."""
    r = _git(vault, "show", f"{rev}:{path}")
    return r.stdout if r.returncode == 0 else None


def touched_commits(vault: Path, lo: str, hi: str) -> list[tuple[str, str]]:
    """Newest-first (sha, ISO author date) commits under NOTE_PREFIX in [lo, hi]."""
    out = _git(vault, "log", "--format=%H|%aI", f"--since={lo}", f"--until={hi}",
               "--", NOTE_PREFIX)
    rows = []
    for line in out.stdout.strip().splitlines():
        if "|" in line:
            sha, when = line.split("|", 1)
            rows.append((sha, when))
    return rows


def changed_paths(vault: Path, sha: str) -> list[tuple[str, str]]:
    """(status, path) for one commit under NOTE_PREFIX; renames count as Modified."""
    out = _git(vault, "show", "--name-status", "--format=", sha, "--", NOTE_PREFIX)
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        status = parts[0][0]
        path = parts[2] if status in ("R", "C") and len(parts) > 2 else parts[1]
        rows.append((status, path))
    return rows


def front_matter(text: str) -> dict[str, str]:
    m = FRONT_RE.match(text or "")
    if not m:
        return {}
    fields = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t", "-")):
            key, val = line.split(":", 1)
            fields[key.strip()] = val.strip()
    return fields


def detect_defect_classes(pre: str, post: str) -> dict[str, list[str]]:
    """Name every defect whose signature is visible in the repair diff pre -> post."""
    found: dict[str, list[str]] = {}

    removed_urls = sorted(set(URL_RE.findall(pre)) - set(URL_RE.findall(post)))
    if removed_urls:
        found["invented_url"] = removed_urls[:5]

    if CUTOFF_RE.search(pre) and not CUTOFF_RE.search(post):
        found["false_cutoff_claim"] = [
            line.strip() for line in pre.splitlines() if CUTOFF_RE.search(line)
        ][:2]

    pre_metrics = [ln.strip() for ln in pre.splitlines() if METRIC_RE.search(ln)]
    post_metrics = [ln.strip() for ln in post.splitlines() if METRIC_RE.search(ln)]
    if not pre_metrics and len(post_metrics) >= 3:
        found["omitted_numerical_data"] = post_metrics[:3]

    added_headings = sorted(set(HEADING_RE.findall(post)) - set(HEADING_RE.findall(pre)))
    if added_headings:
        found["missing_section"] = added_headings[:3]

    pre_fm, post_fm = front_matter(pre), front_matter(post)
    wrong = [k for k in FM_KEYS
             if k in pre_fm and k in post_fm and pre_fm[k] != post_fm[k]]
    if wrong:
        found["wrong_metadata"] = [
            f"{k}: {pre_fm[k]!r} -> {post_fm[k]!r}" for k in wrong
        ]
    return found


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    return text[:MAX_TEXT_CHARS] + "\n[… truncated at 4000 chars …]", True


def build(vault: Path, lo: str, hi: str, bad_cap: int, good_cap: int,
          class_quota: int = 4) -> list[dict]:
    """Scan [lo, hi] and return the labelled corpus, bad-first, ordered by path."""
    commits = touched_commits(vault, lo, hi)
    # Oldest -> newest, so the *earliest* repair of a note is the one sampled.
    history = list(reversed(commits))

    bad: dict[str, dict] = {}
    added: dict[str, tuple[str, str]] = {}
    dirty: set[str] = set()

    for sha, when in history:
        for status, path in changed_paths(vault, sha):
            if status == "A" and path not in added:
                added[path] = (sha, when)
            if status != "M":
                continue
            pre = show(vault, f"{sha}^", path)
            post = show(vault, sha, path)
            if pre is None or post is None or pre == post:
                continue
            classes = detect_defect_classes(pre, post)
            if not classes:
                continue
            dirty.add(path)
            if path in bad:
                bad[path]["defect_classes"] = sorted(
                    set(bad[path]["defect_classes"]) | set(classes)
                )
                continue
            text, truncated = _truncate(pre)
            bad[path] = {
                "id": f"bad-{len(bad) + 1:03d}",
                "label": "bad",
                "vault_path": path,
                "defect_classes": sorted(classes),
                "correction_evidence": {k: v for k, v in classes.items()},
                "recovered_from": f"git show {sha}^:{path}",
                "repair_commit": sha,
                "repair_date": when,
                "pre_repair_text": text,
                "text_truncated": truncated,
            }

    # A note written in the window and never touched by a class-firing repair is
    # the matched `good` sample; scored at the text it was accepted with.
    good: list[dict] = []
    for path, (sha, when) in sorted(added.items()):
        if path in dirty or path in bad:
            continue
        text_now = show(vault, "HEAD", path)
        if text_now is None or len(text_now.strip()) < 400:
            continue
        text, truncated = _truncate(text_now)
        good.append({
            "id": f"good-{len(good) + 1:03d}",
            "label": "good",
            "vault_path": path,
            "defect_classes": [],
            "correction_evidence": {},
            "recovered_from": f"git show {sha}:{path}",
            "created_commit": sha,
            "created_date": when,
            "accepted_text": text,
            "text_truncated": truncated,
        })

    # Stratified, deterministic selection: every class that has examples gets
    # `class_quota` of them (rarest class first), then the budget is filled in
    # path order. Without this the corpus is 90/110 `invented_url` and per-class
    # recall for the rarer classes is reported on n=0 because they were never
    # sampled.
    by_path = sorted(bad.values(), key=lambda s: s["vault_path"])
    chosen: dict[str, dict] = {}
    counts = class_counts(by_path)
    for name in sorted(counts, key=lambda c: (counts[c], c)):
        for sample in by_path:
            if len([s for s in chosen.values() if name in s["defect_classes"]]) >= min(
                    class_quota, counts[name]):
                break
            if name in sample["defect_classes"]:
                chosen[sample["vault_path"]] = sample
    for sample in by_path:
        if len(chosen) >= bad_cap:
            break
        chosen.setdefault(sample["vault_path"], sample)

    ordered = sorted(chosen.values(), key=lambda s: s["vault_path"])[:bad_cap]
    ordered += sorted(good, key=lambda s: s["vault_path"])[:good_cap]
    # Ids are positional in the *shipped* corpus, not in the scan that found it:
    # a capped run must still read bad-001..bad-0NN with no gaps.
    for kind, n in (("bad", bad_cap), ("good", good_cap)):
        i = 0
        for sample in ordered:
            if sample["label"] == kind:
                i += 1
                sample["id"] = f"{kind}-{i:03d}"
    for i, sample in enumerate(ordered, 1):
        sample["seq"] = i
    return ordered


def sample_text(sample: dict) -> str:
    """The text the judge reads, whichever label the sample carries."""
    return sample.get("pre_repair_text") or sample.get("accepted_text") or ""


def class_counts(samples: list[dict]) -> Counter:
    counts: Counter = Counter()
    for s in samples:
        for c in s.get("defect_classes", []):
            counts[c] += 1
    return counts


def format_counts(samples: list[dict]) -> str:
    """The count block clause 1 requires a shipped script to print."""
    bad = sum(1 for s in samples if s["label"] == "bad")
    good = sum(1 for s in samples if s["label"] == "good")
    lines = [f"corpus samples={len(samples)} bad={bad} good={good}"]
    counts = class_counts(samples)
    for name in sorted(DEFECT_CLASSES):
        lines.append(f"class {name} n={counts.get(name, 0)}")
    for name in sorted(set(counts) - set(DEFECT_CLASSES)):
        lines.append(f"class {name} n={counts[name]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vault", default=str(DEFAULT_VAULT))
    ap.add_argument("--since", default="2026-09-05")
    ap.add_argument("--until", default="2026-09-12")
    ap.add_argument("--bad-cap", type=int, default=20)
    ap.add_argument("--good-cap", type=int, default=30)
    ap.add_argument("--out", default=str(Path(__file__).parent / "corpus.jsonl"))
    ap.add_argument("--counts", metavar="ARTIFACT",
                    help="print the counts of an existing artifact and exit")
    ap.add_argument("--class-quota", type=int, default=4,
                    help="max samples guaranteed per defect class")
    args = ap.parse_args(argv)

    if args.counts:
        samples = [json.loads(ln) for ln in Path(args.counts).read_text().splitlines()
                   if ln.strip()]
        print(format_counts(samples))
        return 0

    samples = build(Path(args.vault), args.since, args.until,
                    args.bad_cap, args.good_cap, args.class_quota)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n")
    print(format_counts(samples))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
