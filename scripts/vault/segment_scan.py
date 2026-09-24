#!/usr/bin/env python3
"""segment_scan.py — which vault notes lack the `segment:` convention key (#1167).

`segment:` is the store's own convention, not an OKF requirement, so this is a
separate scan and `validate_okf.py` stays out of it: OKF v0.1 requires only a
`type`, and folding a convention gap into that gate would turn it into a false
OKF violation.

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
  that fence is; a note with no block at all lacks the key and is counted.

Prints one line per directory and exits 1 when any note lacks the key.

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


def missing_segment(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return True
    m = STRICT_FM_RE.match(text)
    return not (m and _SEGMENT_RE.search(m.group(1)))


def scan(root: Path, dirs: list[str], excludes: list[str]) -> dict[str, list[str]]:
    """{directory: [vault-relative paths lacking `segment:`]} for dirs that exist."""
    out: dict[str, list[str]] = {}
    for d in dirs:
        if not (root / d).is_dir():
            continue
        missing = []
        for p in iter_md(root, d):
            rel = p.relative_to(root).as_posix()
            if any(pat in f"/{rel}" for pat in excludes):
                continue
            if missing_segment(p):
                missing.append(rel)
        out[d] = missing
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", default=None, help="vault root (default: the live vault)")
    ap.add_argument("--list", action="store_true", help="name every offending file")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else VAULT_ROOT

    dirs, excludes = scan_dirs()
    result = scan(root, dirs, excludes)
    total = 0
    print(f"[segment_scan] {root}")
    for d in dirs:
        if d not in result:
            print(f"  {d + '/':<12} absent")
            continue
        n = len(result[d])
        total += n
        print(f"  {d + '/':<12} missing segment: {n}")
        if args.list or n <= 10:
            for rel in result[d][:200]:
                print(f"    {rel}")
    print(f"  total missing: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
