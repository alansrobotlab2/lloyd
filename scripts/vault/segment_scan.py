#!/usr/bin/env python3
"""segment_scan.py — which vault notes lack the `segment:` or `tags:` key (#1167, #1804).

Neither key is an OKF requirement, so this is a separate scan and
`validate_okf.py` stays out of it: OKF v0.1 requires only a `type`, and folding
a convention gap into that gate would turn it into a false OKF violation.

Before this, the count came from a heredoc rewritten on every maintenance run,
and it drifted 201 -> 177 -> 152 -> 128 across four runs partly because each one
measured something slightly different (`grep -L '^segment: backlog'` tests a
value, not the key; a bounded read window calls a note with a 200-line front
matter block an offender). So the rules live here once:

* the directories are the extractor allow-list
  (`scripts/memory/next-gen-memory/pipeline_config.yaml` `sources.paths`, with
  its `exclude_patterns`) plus `backlog/`, which that list leaves out on purpose
  but whose every file carries `segment: backlog`;
* files are walked with `validate_okf.iter_md`, so reserved and utility files
  are skipped by the same rule the OKF gate uses;
* front matter is the strict block from offset 0 to its closing fence, wherever
  that fence is; a note with no block at all lacks both keys and is counted.

`segment:` is tested on the block's text — the key is there, whatever its value
— exactly as it has been since #1167. `tags:` is tested on the *parsed* block
(#1804), because there the shape is the requirement and YAML expresses one list
four ways: a note is conformant when `yaml.safe_load` of that same strict block
yields a mapping whose `tags` is a list holding at least one item, so no key,
`tags: []` and a bare scalar are all missing. Flagging the scalar is deliberate
and is *not* what the board's readers do — `app/backlog_tags.normalize_tags`
forgives `tags: '[a, b]'` so one bad row cannot blank the Mission Control board
— but a reader's tolerance is not a writer's licence, and both backlog create
paths now emit a real list. Tag *vocabulary* stays out of scope: #868 retired
that maintenance on 2026-09-22 because no query-time consumer reads these
strings, so this scan checks presence and never content.

Prints a per-directory count line for each key and exits 1 when either count is
non-zero.

Usage:
    python scripts/vault/segment_scan.py [--root PATH] [--list]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from app.paths import VAULT_ROOT  # noqa: E402
from scripts.vault.validate_okf import STRICT_FM_RE, iter_md  # noqa: E402

PIPELINE_CONFIG = REPO_ROOT / "scripts" / "memory" / "next-gen-memory" / "pipeline_config.yaml"
EXTRA_DIRS = ("backlog",)
_SEGMENT_RE = re.compile(r"^segment:", re.M)


def scan_dirs(config_path: Path = PIPELINE_CONFIG) -> tuple[list[str], list[str]]:
    """(vault-relative directories, exclude patterns) the scan covers."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sources = cfg.get("sources") or {}
    dirs = []
    for raw in sources.get("paths") or []:
        name = str(raw).rstrip("/").split("/obsidian/", 1)[-1].strip("/")
        if name and name not in dirs:
            dirs.append(name)
    dirs += [d for d in EXTRA_DIRS if d not in dirs]
    excludes = [str(p) for p in sources.get("exclude_patterns") or [] if "/" in str(p)]
    return dirs, excludes


def missing_keys(path: Path) -> tuple[bool, bool]:
    """(lacks `segment:`, lacks a usable `tags` list) for one file, one read.

    Unreadable file or no strict block at all: both keys are missing, which is
    what #1167 already decided for `segment:` and what the tags half inherits —
    nothing that cannot be read can prove it carries a list.

    `segment:` is a text test on the block, unchanged since #1167, so a note
    whose block will not parse as YAML still counts as declaring its segment.
    `tags:` is decided on the parsed mapping: a list with >= 1 item, anything
    else missing. An unparsable block is therefore missing tags but not missing
    segment — the two halves have different denominators on purpose, and the
    only corpus file in that state (0 of them today, across all scanned dirs)
    is a broken note that has to be fixed either way.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return True, True
    m = STRICT_FM_RE.match(text)
    if not m:
        return True, True
    block = m.group(1)
    no_segment = not _SEGMENT_RE.search(block)
    try:
        fm = yaml.safe_load(block)
    except yaml.YAMLError:
        return no_segment, True
    tags = fm.get("tags") if isinstance(fm, dict) else None
    return no_segment, not (isinstance(tags, list) and len(tags) >= 1)


def missing_segment(path: Path) -> bool:
    return missing_keys(path)[0]


def missing_tags(path: Path) -> bool:
    return missing_keys(path)[1]


def scan(root: Path, dirs: list[str], excludes: list[str]) -> dict[str, dict[str, list[str]]]:
    """{directory: {"segment": [...], "tags": [...]}} of offending rel paths.

    Only directories that exist are keys; a file short enough to lack both keys
    appears in both lists, so the two counts are not a count of files.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for d in dirs:
        if not (root / d).is_dir():
            continue
        seg: list[str] = []
        tag: list[str] = []
        for p in iter_md(root, d):
            rel = p.relative_to(root).as_posix()
            if any(pat in f"/{rel}" for pat in excludes):
                continue
            no_segment, no_tags = missing_keys(p)
            if no_segment:
                seg.append(rel)
            if no_tags:
                tag.append(rel)
        out[d] = {"segment": seg, "tags": tag}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", default=None, help="vault root (default: the live vault)")
    ap.add_argument("--list", action="store_true", help="name every offending file")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else VAULT_ROOT

    dirs, excludes = scan_dirs()
    result = scan(root, dirs, excludes)
    seg_total = 0
    tag_total = 0
    print(f"[segment_scan] {root}")
    for d in dirs:
        if d not in result:
            print(f"  {d + '/':<12} absent")
            continue
        seg = result[d]["segment"]
        tag = result[d]["tags"]
        seg_total += len(seg)
        tag_total += len(tag)
        print(f"  {d + '/':<12} missing segment: {len(seg)}")
        if args.list or len(seg) <= 10:
            for rel in seg[:200]:
                print(f"    {rel}")
        print(f"  {d + '/':<12} missing tags: {len(tag)}")
        if args.list or len(tag) <= 10:
            for rel in tag[:200]:
                print(f"    {rel}")
    print(f"  total missing segment: {seg_total}")
    print(f"  total missing tags: {tag_total}")
    print(f"  total missing: {seg_total + tag_total}")
    return 1 if (seg_total or tag_total) else 0


if __name__ == "__main__":
    sys.exit(main())
