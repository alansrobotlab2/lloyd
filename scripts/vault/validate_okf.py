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
`images/`, `.git/`, `tags.md`, any `_*.md` (Lloyd's own index convention is
`_index.md`), and OKF's two RESERVED names — `index.md` and `log.md` at any
level (§3.1). A reserved file is the opposite of a concept document: §8 says an
index file "contains no frontmatter" (only a bundle-root `index.md` may carry
any, and only `okf_version`). Before #450 they were not in EXCLUDE_FILES, so the
gate FAILED a spec-conformant index — "no parseable frontmatter block" — while
PASSING `projects/inner-voice-paper/index.md`, which is conformant-looking to
the gate only because it disobeys §8 by carrying frontmatter.

Run in CI / a healthcheck / the nightly conformance task, and before any bulk
vault edit.

Usage:
    python scripts/vault/validate_okf.py [--dir NAME] [--strict] [--root PATH]
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
# `index.md` / `log.md` are OKF-reserved at any depth (§3.1) and §8 forbids
# frontmatter in them, so they can never be concept documents — matching on
# filename is what "at any depth" means here, the same rule as `tags.md`. #450.
EXCLUDE_FILES = {"tags.md", "index.md", "log.md"}
STRICT_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

# The vocabulary is #370's 12 canonical values plus what already exists on disk
# outside knowledge/, imported from one module — see scripts/vault/okf_taxonomy.py.
# This used to be 28 literals maintained here by hand, and it drifted from the
# vault it was checking: after #370 renamed knowledge/ onto the canonical set
# (vault 6eed1445, c2b0962), `--dir knowledge --strict` reported 194
# unknown-type warnings because `research-quick`, `research-deep`, `synthesis`,
# `book-note`, `agent-pattern` and `gap-analysis` were not in this set while
# `quick-research`, `medium-research`, `deep-research`, `knowledge-note` and
# `hub` — the values they were renamed FROM — still were. A gate whose vocabulary
# is a second copy of the thing it gates measures its own stale notes.
# Out-of-set values only WARN (OKF tolerates unknown types).
from scripts.vault.okf_taxonomy import KNOWN_TYPES  # noqa: E402


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
    ap.add_argument("--root", default=None,
                    help="tree to scan instead of the live vault root "
                         "(app.paths.VAULT_ROOT) — so a fixture can be graded on "
                         "the same command line the weekly gate uses")
    args = ap.parse_args()
    root = Path(args.root) if args.root else VAULT_ROOT

    violations: list[str] = []
    warnings: list[str] = []
    type_hist = Counter()
    n = 0

    for p in iter_md(root, args.dir):
        n += 1
        rel = str(p.relative_to(root))
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
