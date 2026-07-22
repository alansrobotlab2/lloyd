#!/usr/bin/env python3
"""validate_okf.py — fail-loud OKF v0.1 conformance gate for ~/obsidian.

OKF requires exactly one thing of every concept document: parseable YAML
frontmatter with a non-empty `type`. It deliberately does NOT fix a taxonomy —
"consumers must tolerate unknown types." So this gate:

    * FAILS (exit 1) on: a concept `.md` with no frontmatter, unparseable
      frontmatter (STRICT `^---\\n(.*?)\\n---\\n` — the form `relations_index.py`
      needs), or a missing / empty `type`. These are true OKF violations.
    * WARNS (exit 2 with --strict) on: a `type` value outside the known
      vocabulary — informational only, not an OKF violation.

Reserved / utility files are skipped, matching the migrator: `templates/`,
`images/`, `.git/`, `tags.md`, and any `_*.md` (logs/indexes like `_log.md`).

Run in CI / a healthcheck / the nightly conformance task, and before any bulk
vault edit.

Usage:
    python scripts/vault/validate_okf.py [--dir NAME] [--strict]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.paths import VAULT_ROOT  # noqa: E402

EXCLUDE_DIRS = {"templates", "images", ".git", ".obsidian", ".trash"}
EXCLUDE_FILES = {"tags.md"}
STRICT_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

# Known vocabulary — union of code-branching values and dominant real values.
# Out-of-set values only WARN (OKF tolerates unknown types).
KNOWN_TYPES = {
    "autonomy", "facts", "overview", "compiled_wiki",
    "entity-overview", "concept-synthesis", "how-to", "comparison",
    "source-summary", "quick-research", "medium-research", "deep-research",
    "reference", "note", "notes", "video-note", "research", "knowledge-note",
    "hub", "skill", "person", "daily-note", "reflection", "stack-update",
    "knowledge-note", "talk", "project-notes", "work-notes",
}


def iter_md(root: Path, only_dir: str | None):
    base = root / only_dir if only_dir else root
    for p in sorted(base.rglob("*.md")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.name in EXCLUDE_FILES or p.name.startswith("_"):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="treat unknown-type warnings as failure (exit 2)")
    args = ap.parse_args()

    violations: list[str] = []
    warnings: list[str] = []
    type_hist = Counter()
    n = 0

    for p in iter_md(VAULT_ROOT, args.dir):
        n += 1
        rel = str(p.relative_to(VAULT_ROOT))
        content = p.read_text(encoding="utf-8")
        m = STRICT_FM_RE.match(content)
        if not m:
            violations.append(f"{rel}: no parseable frontmatter block")
            continue
        try:
            fm = yaml.safe_load(m.group(1))
        except yaml.YAMLError as e:
            violations.append(f"{rel}: yaml error: {str(e).splitlines()[0]}")
            continue
        if not isinstance(fm, dict):
            violations.append(f"{rel}: frontmatter is not a mapping")
            continue
        t = str(fm.get("type", "") or "").strip()
        if not t:
            violations.append(f"{rel}: missing/empty `type`")
            continue
        type_hist[t] += 1
        if t not in KNOWN_TYPES:
            warnings.append(f"{rel}: unknown type '{t}'")

    print(f"[validate_okf] scanned {n} concept files"
          + (f" in {args.dir}" if args.dir else ""))
    print(f"  conformant : {n - len(violations)}")
    print(f"  VIOLATIONS : {len(violations)}")
    print(f"  warnings   : {len(warnings)}")
    if type_hist:
        top = ", ".join(f"{t}={c}" for t, c in type_hist.most_common(12))
        print(f"  types: {top}")

    if violations:
        print("\n🔴 OKF violations (exit 1):")
        for v in violations[:100]:
            print(f"   {v}")
        if len(violations) > 100:
            print(f"   ... and {len(violations) - 100} more")

    if warnings and args.strict:
        print(f"\n⚠️  {len(warnings)} unknown-type warning(s):")
        for w in warnings[:50]:
            print(f"   {w}")

    if violations:
        return 1
    if warnings and args.strict:
        return 2
    print("\n✅ OKF v0.1 conformant" +
          (f" ({len(warnings)} unknown-type notes)" if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
