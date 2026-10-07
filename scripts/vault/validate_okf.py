#!/usr/bin/env python3
"""validate_okf.py — fail-loud OKF v0.1 conformance gate for ~/obsidian.

OKF requires exactly one thing of every concept document: parseable YAML
frontmatter with a non-empty `type`. It deliberately does NOT fix a taxonomy —
"consumers must tolerate unknown types." So this gate:

    * FAILS (exit 1) on: a concept `.md` with no frontmatter, unparseable
      frontmatter (STRICT `^---\\n(.*?)\\n---\\n` — the form `relations_index.py`
      needs), a missing / empty `type`, or a SECOND frontmatter block stranded in
      the BODY that is not on `scripts/vault/okf_stranded.py`'s checked-in legacy
      allow-list. These are true OKF violations.
    * COUNTS separately without failing: `known-stranded` — the legacy body-block
      set from the July 2026 wrapper pass that the allow-list names (#478's data
      half). Folding those files into VIOLATIONS would turn the weekly
      conformance job red every week and bury the only number that matters, which
      is a NEW stranded file. See #960.
    * WARNS (exit 2 with --strict) on: a `type` value outside the known
      vocabulary, or a `domain` under `knowledge/` that is neither canonical nor
      an alias (#949) — informational only, not an OKF violation. ~118 notes
      carried an out-of-set domain when the vocabulary was closed; failing on
      them would make re-tagging them a precondition of every run.

Before #960 the body was read by nothing: STRICT_FM_RE is anchored at offset 0
and applied with `.match()`, so the violations above were the whole check and a
file whose real frontmatter sat 5 lines below the stamped one was reported
conformant — 2574/2574 clean on 2026-09-12 while 220 files carried a stranded
block. The strict block still decides `type`; the detector only reports what the
strict block cannot see.

Reserved / utility files are skipped, matching the migrator: `templates/`,
`images/`, `.git/`, `tags.md`, any `_*.md` (Lloyd's own index convention is
`_index.md`), and OKF's two RESERVED names — `index.md` and `log.md` at any
level (§3.1). A reserved file is the opposite of a concept document: §8 says an
index file "contains no frontmatter" (only a bundle-root `index.md` may carry
any, and only `okf_version`). Before #450 they were not in EXCLUDE_FILES, so the
gate FAILED a spec-conformant index — "no parseable frontmatter block" — while
PASSING `projects/inner-voice-paper/index.md`, which is conformant-looking to
the gate only because it disobeys §8 by carrying frontmatter.

A subtree can be out of scope the same way, and `EXCLUDE_PATHS` is where every such
ruling lives, by vault-relative path prefix: `backlog/data/`, the witness-extract
directory an item's owed-witness clause writes into (#1934); `lloyd/memory/`, the
loaded-memory topic files a nightly writes whole (#2340); and `plans/`, the
`ExitPlanMode` handoffs (#2340). That list is the complete one — a ruling kept only in a
comment below, or in a test, reads as scope the gate has not agreed to, so a new one goes
in the tuple AND on this line. `.pytest_cache` is the same kind of ruling in
`EXCLUDE_DIRS`, where a directory NAME is right for a tool artifact and wrong for a
subtree an OKF segment shares a name with.

`segment_scan.py` imports `iter_md`, so one exemption settles both this gate and the
`segment:`/`tags:` scan; `okf_migrate.py` imports `EXCLUDE_PATHS` for the same reason,
because that one WRITES.

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

#: The domain warning as the loop below appends it — `"{rel}: unknown domain '{d}'"`.
#:
#: The split the summary prints (#2204) is READ BACK out of the warnings list rather
#: than recomputed from the tree, and that is the point: the file count is then a
#: subset of the total printed beside it by construction, so the two can never be two
#: different measurements of one pass. Anchored at the end because a vault-relative
#: path may itself contain a colon; the prefix before it is the file.
UNKNOWN_DOMAIN_WARNING = re.compile(r": unknown domain '(?P<value>[^']*)'$")

# `.pytest_cache` joins the tool directories because a pytest run rooted in the vault
# leaves a `README.md` with no frontmatter INSIDE the vault root, and the vault's copy is
# git-IGNORED (`~/obsidian/.pytest_cache/.gitignore` is `*`): deleting it reports a clean
# vault until the next run recreates it (#2340). A cleanup pass cannot fix an artifact
# that a tool owns, so the gate declines to treat it as a document.
EXCLUDE_DIRS = {"templates", "images", ".git", ".obsidian", ".trash", ".pytest_cache"}
# `index.md` / `log.md` are OKF-reserved at any depth (§3.1) and §8 forbids
# frontmatter in them, so they can never be concept documents — matching on
# filename is what "at any depth" means here, the same rule as `tags.md`. #450.
EXCLUDE_FILES = {"tags.md", "index.md", "log.md"}
# Whole subtrees that are out of scope for every gate, by VAULT-RELATIVE PATH
# PREFIX (#1934). `backlog/data/` holds what an item's owed-witness clause
# commits: the frozen extract itself (`voice.log`, `usage.db`, `*.ndjson`) and
# the `.md` sidecar that names its commit and md5. The sidecars are minted by
# `scripts/automod/backlog.py`'s auto-witness clause, so their population is
# open-set — the offender count grew 4 → 5 between #1934's filing and its
# triage — and no writer-side check can pin a frontmatter convention on files
# whose count is decided by whoever files a measurement item next. The ruling
# is therefore a scope call, the same verdict as a reserved filename: these are
# not concept documents. Not a defect to be repaired after the fact by a
# nightly that edits bytes whose provenance is their checksum.
#
# A prefix, deliberately NOT an entry in EXCLUDE_DIRS: that set is matched against each
# path COMPONENT in `iter_md` below, so a bare `"data"` there would also take a future
# `knowledge/x/data/` out of the gate. `iter_md` applies this one to the path relative to
# the scan root, which is the vault root, so it is anchored at `backlog/data/` and
# nowhere else. (Cited by function name rather than `file:line`: adding a ruling to this
# tuple moves the walk below it, and a number written here would be a claim nobody
# re-checks.)
#
# One skip, both gates: `segment_scan.py:54` imports `iter_md` from here, so
# the OKF gate and the `segment:`/`tags:` scan clear on the same exemption.
# `okf_migrate.py` imports THIS tuple rather than copying it, so the gate's leniency
# and the migrator's reach cannot diverge — for the memory topic files that
# divergence was not cosmetic: `--apply` would have backfilled a `type:` fence into
# the 39 files the gate had just been told to leave alone, editing the bytes the
# memory system reads (#2339's ruling was made only on the gate; #2340 closed the
# second surface with it).
#
# `lloyd/memory/` is the loaded-memory detail the index lines in `lloyd/MEMORY.md`
# point at — the nightly reflection job writes these whole, and 39 of the 40 files
# there had no fence when #2340 measured the vault (the 40th was written fenced, so
# "39 of 40" is the shape of the class, not its count). A topic file is not an OKF
# concept document: §3 asks a `type` of a CONCEPT, and this is the detail half of a
# memory index, pulled on demand by `memory_read(file="topics/<slug>")`. Not, as
# #2340 first argued, a prompt-budget saving — only the index line is spliced every
# turn, so the token claim is refutable from the loaded prompt and the class does not
# need it. The prefix rather than an edit for two reasons: hand-adding a fence is
# reverted by the reflection job's next write of the same file (#1488/#1500/#1789 own
# that surface), and a bare `"memory"` in EXCLUDE_DIRS would take the vault's
# `memory/` daily-note segment — which IS in the OKF taxonomy, and which
# `segment_scan.py` counts through this same `iter_md` — out of both gates with it.
#
# `plans/` is `ExitPlanMode`'s output: plan bodies and session-named handoffs, an
# open-set population only that tool writes, so a per-file fix has no stopping rule —
# the argument #1934 used for the sidecar directory, and the vault's copy holds four
# `.md` files of which two came out fenced and two did not.
#
# What is NOT here, so the next reader can tell a narrowed gate from a satisfied one:
# #2340 closed 42 of the live vault's 48 §3 violations and left 6 in scope BY RULING —
# the 4 `autonomy/referential-integrity*.md` outputs await #2326's generator fix, and
# the 2 `lloyd/reviews/` archives are a named two-file vault write, not a class.
EXCLUDE_PATHS = ("backlog/data/", "lloyd/memory/", "plans/")
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
from scripts.vault.okf_taxonomy import is_known_domain  # noqa: E402
# One detector, shared with the migrator, so the two scripts cannot disagree
# about what a stranded body block is — and one allow-list, so neither keeps its
# own count of how big the legacy set is (#960).
from scripts.vault.okf_stranded import load_allowlist  # noqa: E402
from scripts.vault.okf_stranded import find_stranded_frontmatter  # noqa: E402


def iter_md(root: Path, only_dir: str | None):
    base = root / only_dir if only_dir else root
    for p in sorted(base.rglob("*.md")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        rel_posix = rel.as_posix()
        if any(rel_posix.startswith(prefix) for prefix in EXCLUDE_PATHS):
            continue
        if p.name in EXCLUDE_FILES or p.name.startswith("_"):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="treat unknown type/domain warnings as failure (exit 2)")
    ap.add_argument("--root", default=None,
                    help="tree to scan instead of the live vault root "
                         "(app.paths.VAULT_ROOT) — so a fixture can be graded on "
                         "the same command line the weekly gate uses")
    args = ap.parse_args()
    root = Path(args.root) if args.root else VAULT_ROOT

    violations: list[str] = []
    warnings: list[str] = []
    type_hist = Counter()
    # The legacy body-block set, and what this scan found. `known_hits` counts
    # allow-listed files; anything stranded that is NOT on the list is a
    # violation. `stale` is listed-but-no-longer-stranded, i.e. #478's data fix
    # reached it and the list needs regenerating — an observation, not a fault.
    known_stranded = load_allowlist()
    known_hits = 0
    stranded_seen: set[str] = set()
    n = 0

    for p in iter_md(root, args.dir):
        n += 1
        rel = p.relative_to(root).as_posix()
        content = p.read_text(encoding="utf-8")
        stranded = find_stranded_frontmatter(content)
        if stranded:
            stranded_seen.add(rel)
            if rel in known_stranded:
                known_hits += 1
            else:
                # `rel:line: message`, so the line is greppable and every stranded
                # line the file carries is named — not just the first.
                rest = ("" if len(stranded) == 1 else
                        " (+ also "
                        + ", ".join(f"line {h.line}" for h in stranded[1:6])
                        + (f" and {len(stranded) - 6} more"
                           if len(stranded) > 6 else "") + ")")
                violations.append(
                    f"{rel}:{stranded[0].line}: stranded frontmatter block "
                    f"[{', '.join(stranded[0].keys[:4])}]{rest}")
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
        # `domain` is governed only where it names a directory: knowledge/.
        d = str(fm.get("domain", "") or "").strip()
        if d and rel.startswith("knowledge/") and not is_known_domain(d):
            warnings.append(f"{rel}: unknown domain '{d}'")

    print(f"[validate_okf] scanned {n} concept files"
          + (f" in {args.dir}" if args.dir else ""))
    print(f"  conformant : {n - len(violations)}")
    print(f"  VIOLATIONS : {len(violations)}")
    print(f"  known-stranded : {known_hits}  (legacy allow-list; "
          "scripts/vault/okf_stranded_known.txt)")
    print(f"  warnings   : {len(warnings)}")
    # The domain axis, split out of the list above — see UNKNOWN_DOMAIN_WARNING. Both
    # numbers come from `warnings` and nothing else, so `… file(s)` can never be more
    # than the total one line up. Printed here, in the summary block, because the
    # weekly job runs WITHOUT `--strict` (only `--strict` ever named a domain row, and
    # its block is capped at 50 rows that domain rows never reach), and because this
    # tree usually has violations, which exits 1 before the closing line below.
    domain_pairs = {(w[:m.start()], m.group("value"))
                    for w in warnings if (m := UNKNOWN_DOMAIN_WARNING.search(w))}
    domain_hist = Counter(v for _, v in domain_pairs)
    print(f"  unknown-domain : {len({r for r, _ in domain_pairs})} file(s), "
          f"{len(domain_hist)} value(s)")
    if type_hist:
        top = ", ".join(f"{t}={c}" for t, c in type_hist.most_common(12))
        print(f"  types: {top}")
    # The promotion census (#949's ruling promotes a spelling once files carry it), so
    # only values clearing the bar are named: on 2026-10-05 the off-set set was 76
    # values and just 11 cleared it, and a line naming all 76 would bury the 11. No
    # line at all when nothing clears it — an empty `domains:` reads as a measured
    # zero, which is how a number gets believed without being measured.
    clearing = [(v, c) for v, c in domain_hist.most_common() if c >= 2]
    if clearing:
        print("  domains (>=2 files): "
              + ", ".join(f"{v}={c}" for v, c in clearing))

    if violations:
        print("\n🔴 OKF violations (exit 1):")
        for v in violations[:100]:
            print(f"   {v}")
        if len(violations) > 100:
            print(f"   ... and {len(violations) - 100} more")

    # The stale set is only meaningful over the tree the list was generated from.
    # Under `--dir`, or over a `--root` that is not the vault, every listed path
    # outside the scan would read as "resolved" without anything fixing it — a
    # check whose denominator is the scan, which is the defect shape this file
    # exists to avoid. So: report it only for a full-vault scan.
    if args.dir is None and root == VAULT_ROOT:
        stale = sorted(known_stranded - stranded_seen)
        if stale:
            print(f"\nℹ️  {len(stale)} allow-listed file(s) no longer stranded — "
                  "regenerate the list (#478 progress):")
            for r in stale[:20]:
                print(f"   {r}")
            if len(stale) > 20:
                print(f"   ... and {len(stale) - 20} more")

    if warnings and args.strict:
        print(f"\n⚠️  {len(warnings)} unknown-type/domain warning(s):")
        for w in warnings[:50]:
            print(f"   {w}")

    if violations:
        return 1
    if warnings and args.strict:
        return 2
    print("\n✅ OKF v0.1 conformant" +
          (f" ({len(warnings)} unknown-type/domain notes)" if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
