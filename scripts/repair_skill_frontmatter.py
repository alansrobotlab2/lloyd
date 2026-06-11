#!/usr/bin/env python3
"""Repair corrupted SKILL.md frontmatter across the skills library.

The 2026-06 corruption family (flagged DEAD by skill_lint.py because the
YAML fails to parse, leaving description+tags unreadable):

  A. indented `  status: active` after a flow-style tags list
  B. orphan col-0 `- foo` list items not attached to any key
  C. duplicate top-level keys (second `tags:` etc.)
  D. a SECOND `---` frontmatter block stacked under the first one,
     holding the real name/description/metadata

Repair = mechanical line fixes on the first block, merge keys from a
stacked second block (first block wins on conflicts), re-emit ONE valid
frontmatter block, and verify with yaml.safe_load before writing.
Also strips stale `SIGNAL:TASK_COMPLETE` sentinel lines from bodies.

Files whose repaired frontmatter still fails YAML are reported and left
untouched.

Usage:
    repair_skill_frontmatter.py            # dry run
    repair_skill_frontmatter.py --apply
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

SKILLS_DIR = Path.home() / "obsidian" / "skills"

KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):(\s|$)")
INDENTED_KEY_RE = re.compile(
    r"^\s+(status|category|segment|name|type|metadata|version|author|license):\s"
)


def split_frontmatter(text: str):
    """Return (fm_lines, rest) for a leading --- block, or (None, text)."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], "\n".join(lines[i + 1:])
    return None, text


def fix_block_lines(fm_lines: list[str]) -> list[str]:
    """Apply mechanical fixes: de-indent stray keys, drop orphan list
    items and duplicate top-level keys."""
    out: list[str] = []
    seen_keys: set[str] = set()
    dropping_dup = False
    for line in fm_lines:
        stripped = line.strip()
        # de-indent stray `  status: active`-style keys (continuations of a
        # real multiline value never look like key: value)
        m = INDENTED_KEY_RE.match(line)
        if m and not (out and out[-1].rstrip().endswith(":")):
            line = line.strip()
        km = KEY_RE.match(line)
        if km:
            key = km.group(1)
            if key in seen_keys:
                dropping_dup = True  # drop this line + its hangers-on
                continue
            seen_keys.add(key)
            dropping_dup = False
            out.append(line)
            continue
        # orphan col-0 list item: previous emitted line doesn't end with ':'
        if stripped.startswith("- ") and not line.startswith(" "):
            if dropping_dup or not (out and out[-1].rstrip().endswith(":")):
                continue
        if dropping_dup and (line.startswith(" ") or stripped.startswith("- ")):
            continue  # continuation of a dropped duplicate key
        if stripped:
            dropping_dup = False
        out.append(line)
    return out


def repair_file(path: Path) -> tuple[str, str | None]:
    """Return (action, new_text). action ∈ ok | repaired | failed."""
    text = path.read_text(encoding="utf-8")
    fm_lines, body = split_frontmatter(text)
    if fm_lines is None:
        return "failed", None

    fm: dict | None = None
    try:
        parsed = yaml.safe_load("\n".join(fm_lines))
        fm = parsed if isinstance(parsed, dict) else None
    except yaml.YAMLError:
        fm = None

    repaired_lines = fix_block_lines(fm_lines)
    try:
        repaired = yaml.safe_load("\n".join(repaired_lines))
        if not isinstance(repaired, dict):
            repaired = None
    except yaml.YAMLError:
        repaired = None
    if repaired is None:
        return "failed", None

    # stacked second frontmatter block directly under the first?
    body_stripped = body.lstrip("\n")
    if body_stripped.startswith("---"):
        second_lines, rest = split_frontmatter(body_stripped)
        if second_lines is not None:
            try:
                second = yaml.safe_load("\n".join(fix_block_lines(second_lines)))
                if isinstance(second, dict):
                    for k, v in second.items():
                        repaired.setdefault(k, v)
                    body = rest
            except yaml.YAMLError:
                pass  # keep second block as body text

    name = repaired.get("name") or path.parent.name
    repaired["name"] = name
    if not repaired.get("description"):
        # no description anywhere — tags still make it non-DEAD; leave it
        pass

    body = "\n".join(
        ln for ln in body.split("\n") if "SIGNAL:TASK_COMPLETE" not in ln
    )
    new_text = (
        "---\n"
        + yaml.safe_dump(repaired, sort_keys=False, allow_unicode=True,
                         default_flow_style=None).rstrip()
        + "\n---\n"
        + body.lstrip("\n")
    )
    if fm is not None and fm.get("description") and fm.get("tags") \
            and "SIGNAL:TASK_COMPLETE" not in text \
            and not body_stripped.startswith("---"):
        return "ok", None
    return "repaired", new_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    n_ok = n_rep = 0
    failed: list[str] = []
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        action, new_text = repair_file(skill_md)
        if action == "ok":
            n_ok += 1
        elif action == "repaired":
            n_rep += 1
            print(f"  repair: {skill_md.parent.name}")
            if args.apply and new_text is not None:
                skill_md.write_text(new_text, encoding="utf-8")
        else:
            failed.append(skill_md.parent.name)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[repair-skill-frontmatter] {mode}: "
          f"{n_ok} ok, {n_rep} repaired, {len(failed)} need manual fix")
    for name in failed:
        print(f"  ! manual: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
