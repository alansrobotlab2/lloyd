#!/usr/bin/env python3
"""Build the indexed MEMORY.md + topic files into an overlay — never into the vault.

Review 2026-09-24, P4. `lloyd/MEMORY.md` is 73 KB against a 73,728 B ceiling: one
entry from refusal, and ~18k tokens on every user turn. The proposal is an index —
one typed line per entry, the entry itself in a topic file pulled on demand — and
this is the one-off that builds it, deterministically and offline, so the same
input always yields the same index and the eval's `indexed` arm is reproducible.

What it does:

* splits MEMORY.md into units (each `### ` subsection, or a `## ` section with no
  subsections) and writes each unit **verbatim** to `<out>/memory/<slug>.md`, so
  nothing is lost — every source line is in exactly one topic file (a unit over
  `TOPIC_FILE_CEILING_BYTES` is split at entry boundaries into `<slug>-2`, …);
* writes `<out>/MEMORY.md`: the same front matter, title and headings, and one
  line per entry, `- [type] <hook> → topics/<slug>`. The hook is the entry's bold
  lead, else its first sentence, clipped at a word boundary; the clip length is the
  largest that keeps the index within `--target` bytes (80% of
  `prompt_surface.MEMORY_MD_INDEX_CEILING_BYTES` by default). A `feedback` entry —
  Alan's rulings and the house style rules — is kept whole when it fits a line,
  because a ruling cut to its first clause is a different ruling;
* copies SOUL.md and USER.md unchanged, so `<out>` is a complete prompt overlay
  (`prompt_builder.build_system_prompt(overlay_dir=<out>)`);
* runs `validate_memory_index.py --mode full` over `<out>` and exits 1 if the index
  it just built would not pass, and writes `<out>/consolidation.json`.

It only ever writes under `--out`, refuses an `--out` inside the vault (resolved,
so a symlink cannot smuggle one in), and requires `--dry-run`: applying the index
to the live vault is the deploy step, gated on `eval/run_memory_index_ab.py`, and
is done by a person with the vault skill patch
(`scripts/maintenance/vault-memory-index-skills.patch`) in the same change.

Usage:
  .venvs/lloyd/bin/python scripts/memory/consolidate_memory_index.py \\
      --dry-run --out /tmp/memory-index-overlay
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prompt_surface as ps  # noqa: E402
from app import memory_ceiling as mc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_memory_index as vmi  # noqa: E402

VAULT_ROOT = Path.home() / "obsidian"
DEFAULT_TARGET = int(ps.MEMORY_MD_INDEX_CEILING_BYTES * vmi.DEFAULT_TIGHTNESS)
PROMPT_FILES = ("SOUL.md", "USER.md")

#: Units whose every entry is a ruling or a house rule — `feedback`, kept whole.
FEEDBACK_HEADINGS = re.compile(r"retired decisions|meta-instructions", re.I)
#: An entry anywhere that records a ruling by Alan is `feedback` too.
RULING_RE = re.compile(r"Alan'?s (ruling|call|rule)|\bAlan ruled\b", re.I)
#: A short bold-led correction (`**Scope Preservation**: If the user …`) in the
#: corrections section is a house rule, not a lesson with a proof attached.
CORRECTIONS_HEADINGS = re.compile(r"corrections|anti-patterns", re.I)
SHORT_RULE_CHARS = 300

INDEX_PREAMBLE = (
    "Index: one line per entry. The line is the hook; the entry in full — its "
    "numbers, proving commands and caveats — is in the topic file it ends with: "
    '`memory_read(file="topics/<slug>")` before acting on a detail the line does '
    "not carry. `[feedback]` lines are rulings and are complete as written."
)


class RefusedOutput(ValueError):
    """The overlay path would write into the vault."""


@dataclass
class Entry:
    text: str                 # verbatim source text (one bullet or one paragraph)
    kind: str = "project"


@dataclass
class Unit:
    heading: str              # the unit's own heading text, without the #s
    level: int                # 2 or 3
    parent: str = ""          # the `## ` heading a `### ` unit sits under
    lines: list[str] = field(default_factory=list)  # verbatim, heading excluded
    entries: list[Entry] = field(default_factory=list)
    slugs: list[str] = field(default_factory=list)  # one per topic part
    entry_part: list[int] = field(default_factory=list)


def slugify(heading: str, taken: set[str]) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "topic"
    if len(s) > 40:
        s = s[:40].rsplit("-", 1)[0] or s[:40]
    base, n = s, 2
    while s in taken:
        s = f"{base}-{n}"
        n += 1
    taken.add(s)
    return s


def split_front_matter(text: str) -> tuple[str, str]:
    body = ps.body(text)
    return text[: len(text) - len(body)], body


def parse_units(body: str) -> tuple[list[str], list[Unit]]:
    """(lines before the first `## `, units in file order)."""
    head: list[str] = []
    units: list[Unit] = []
    parent = ""
    cur: Unit | None = None
    for line in body.split("\n"):
        if line.startswith("## "):
            parent = line[3:].strip()
            cur = Unit(heading=parent, level=2)
            units.append(cur)
        elif line.startswith("### "):
            cur = Unit(heading=line[4:].strip(), level=3, parent=parent)
            units.append(cur)
        elif cur is None:
            head.append(line)
        else:
            cur.lines.append(line)
    for u in units:
        u.entries = split_entries(u.lines)
    return head, units


def split_entries(lines: list[str]) -> list[Entry]:
    """Top-level bullets and paragraphs, each verbatim."""
    out: list[Entry] = []
    buf: list[str] = []

    def flush() -> None:
        if buf and any(b.strip() for b in buf):
            out.append(Entry("\n".join(buf).strip("\n")))
        buf.clear()

    for line in lines:
        if line.startswith(("- ", "* ")):
            flush()
            buf.append(line)
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    return out


def classify(unit: Unit, entry: Entry) -> str:
    if FEEDBACK_HEADINGS.search(unit.heading) or FEEDBACK_HEADINGS.search(unit.parent):
        return "feedback"
    if RULING_RE.search(entry.text):
        return "feedback"
    if (CORRECTIONS_HEADINGS.search(unit.heading) and entry.text.startswith("- **")
            and len(entry.text) <= SHORT_RULE_CHARS):
        return "feedback"
    return "project"


def _plain(text: str) -> str:
    """One line, list marker dropped, markdown emphasis kept (it is cheap and it is
    how the source marks what matters)."""
    t = re.sub(r"^[-*]\s+", "", text.strip())
    return re.sub(r"\s+", " ", t)


def hook(entry: Entry, cap: int) -> str:
    """The index text for one entry, at most `cap` characters."""
    text = _plain(entry.text)
    if entry.kind == "feedback" and len(text) <= cap:
        return text
    m = re.match(r"\*\*(.+?)\*\*", text)
    lead = m.group(0) if m else re.split(r"(?<=[.;:])\s", text, maxsplit=1)[0]
    if len(lead) <= cap:
        # Room left after the lead is spent on the text that follows it: a bold
        # lead alone is often a label ("**Scope Preservation**:"), not the rule.
        rest = text[len(lead):]
        room = cap - len(lead)
        if room > 20 and rest.strip():
            tail = rest if len(rest) <= room else rest[: room - 1].rsplit(" ", 1)[0] + "…"
            return (lead + tail).rstrip()
        return lead
    return lead[: cap - 1].rsplit(" ", 1)[0] + "…"


def assign_topics(units: list[Unit], ceiling: int = mc.TOPIC_FILE_CEILING_BYTES
                  ) -> dict[str, str]:
    """{slug: topic file text}. Each unit verbatim, split at entries if too big."""
    taken: set[str] = set()
    files: dict[str, str] = {}
    for u in units:
        if not u.entries:
            continue  # a `## ` that only holds `### ` units has nothing of its own
        base = slugify(u.heading, taken)
        title = f"{u.parent} — {u.heading}" if u.parent and u.level == 3 else u.heading
        parts: list[list[int]] = [[]]
        header = f"# {title}\n\n"
        size = len(header.encode())
        for i, e in enumerate(u.entries):
            eb = len(e.text.encode()) + 2
            if parts[-1] and size + eb > ceiling:
                parts.append([])
                size = len(header.encode())
            parts[-1].append(i)
            size += eb
        u.entry_part = [0] * len(u.entries)
        for n, idxs in enumerate(parts):
            slug = base if n == 0 else slugify(f"{base}-{n + 1}", taken)
            u.slugs.append(slug)
            for i in idxs:
                u.entry_part[i] = n
            if len(parts) == 1:
                # One part: the unit verbatim, prose between entries included.
                body = "\n".join(u.lines).strip("\n")
            else:
                body = "\n\n".join(u.entries[i].text for i in idxs)
            files[slug] = header + body + "\n"
    return files


def render_index(front: str, head: list[str], units: list[Unit], cap: int) -> str:
    out = [front.rstrip("\n")] if front else []
    title = [ln for ln in head if ln.strip()]
    out.extend(title or ["# Lloyd Long-Term Memory"])
    out += ["", INDEX_PREAMBLE]
    for u in units:
        out += ["", ("## " if u.level == 2 else "### ") + u.heading]
        for i, e in enumerate(u.entries):
            slug = u.slugs[u.entry_part[i]]
            link = f" → topics/{slug}"
            prefix = f"- [{e.kind}] "
            line_room = vmi.INDEX_LINE_MAX_CHARS - len(prefix) - len(link)
            # A ruling gets the whole line whatever the budget: the budget is
            # balanced on the project lines, never on the ones the index may not cut.
            room = line_room if e.kind == "feedback" else min(cap, line_room)
            out.append(prefix + hook(e, room) + link)
    return "\n".join(out).rstrip("\n") + "\n"


def build(src: Path, target: int = DEFAULT_TARGET) -> tuple[str, dict[str, str], dict]:
    """(index text, {slug: topic text}, report) for the MEMORY.md under `src`."""
    text = (src / "MEMORY.md").read_text(encoding="utf-8")
    front, body = split_front_matter(text)
    head, units = parse_units(body)
    for u in units:
        for e in u.entries:
            e.kind = classify(u, e)
    topics = assign_topics(units)
    lo, hi, best = 24, vmi.INDEX_LINE_MAX_CHARS, None
    while lo <= hi:
        mid = (lo + hi) // 2
        idx = render_index(front, head, units, mid)
        if len(idx.encode()) <= target:
            best, lo = (mid, idx), mid + 1
        else:
            hi = mid - 1
    if best is None:
        best = (24, render_index(front, head, units, 24))
    cap, index = best
    kinds: dict[str, int] = {}
    for u in units:
        for e in u.entries:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
    report = {
        "source": str(src / "MEMORY.md"),
        "source_bytes": len(text.encode()),
        "index_bytes": len(index.encode()),
        "target_bytes": target,
        "hook_cap_chars": cap,
        "entries": sum(len(u.entries) for u in units),
        "entry_types": kinds,
        "topics": {slug: len(t.encode()) for slug, t in topics.items()},
    }
    return index, topics, report


def _refuse_vault(out: Path) -> None:
    real = Path(os.path.realpath(out))
    vault = Path(os.path.realpath(VAULT_ROOT))
    if real == vault or vault in real.parents:
        raise RefusedOutput(f"--out {out} resolves inside the vault ({vault}); this "
                            "script writes overlays only")


def write_overlay(src: Path, out: Path, target: int = DEFAULT_TARGET) -> dict:
    _refuse_vault(out)
    index, topics, report = build(src, target)
    out.mkdir(parents=True, exist_ok=True)
    for name in PROMPT_FILES:
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    (out / "MEMORY.md").write_text(index, encoding="utf-8")
    tdir = out / mc.TOPICS_SUBDIR
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir()
    for slug, t in topics.items():
        (tdir / f"{slug}.md").write_text(t, encoding="utf-8")
    report["validation"] = vmi.check(out, ceiling=ps.MEMORY_MD_INDEX_CEILING_BYTES,
                                     mode="full")
    (out / "consolidation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="required: build into --out only (applying is the deploy step)")
    ap.add_argument("--out", type=Path, required=True, help="overlay directory to write")
    ap.add_argument("--src", type=Path, default=mc.MEMORIES_DIR,
                    help="memory root to read (default: live, read-only)")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET,
                    help=f"index byte budget (default {DEFAULT_TARGET})")
    args = ap.parse_args(argv)
    if not args.dry_run:
        print("refusing: only --dry-run exists; the live index is written by hand "
              "after the eval promotes it", file=sys.stderr)
        return 2
    try:
        report = write_overlay(args.src, args.out, args.target)
    except RefusedOutput as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    v = report["validation"]
    print(f"{report['source_bytes']:,} B → index {report['index_bytes']:,} B "
          f"(target {report['target_bytes']:,}, hook cap {report['hook_cap_chars']}), "
          f"{report['entries']} entries {report['entry_types']}, "
          f"{len(report['topics'])} topic files → {args.out}")
    print(f"validation: {'OK' if v['ok'] else 'FAIL'}")
    for e in v["errors"]:
        print(f"  - {e}")
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
