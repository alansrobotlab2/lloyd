"""Treat a backlog item as a hypothesis, not an instruction.

The Lloyd backlog holds 50-odd open items going back to February. Many were
written against a system that has since changed: the bug was fixed, the module
was rewritten, the upstream issue closed, the approach was superseded. Acting
on those blind produces the worst possible outcome — a confident, tested,
gated change that solves a problem nobody has.

So nothing here implements anything until the item's *premise* has been
re-checked against the system as it is today. Four verdicts:

  confirmed     the premise still holds; there is real work here
  already_done  the premise held once and something has since fixed it
  stale         the premise no longer describes this system
  unverifiable  the item states no claim that can be checked

**`stale` and `already_done` are successes.** For a backlog this age, retiring
items with evidence is worth more than implementing them, and it is the
outcome most items should reach. A pipeline that "succeeds" only by writing
code will quietly convert a stale backlog into a pile of unnecessary changes.

The premise check does double duty. When it confirms a problem, that same
check becomes the acceptance test: after the fix lands it must now fail to
reproduce. An item whose premise cannot be turned into a check is not ready to
be implemented automatically — that is what `unverifiable` means, and it is a
request for a human to sharpen the item, not a failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

BACKLOG_DIR = Path.home() / "obsidian" / "backlog"
OPEN_STATUSES = {"up_next", "draft", "in_progress"}

# Only Lloyd's own board. The backlog is shared: of 52 open items, 3 are Alfie
# (robot firmware) and 1 is on an Architecture board. Those are legitimately
# out of scope for a self-modification pass, and the `board` field says so for
# free — filtering here rather than spending an LLM turn per item to rediscover
# it. Verified against #38 "Alfie — Fix mecanum wheels behavior": triage burned
# a full turn to correctly conclude `not_code`, which the board already knew.
DEFAULT_BOARDS = ("lloyd",)

VERDICTS = ("confirmed", "already_done", "stale", "unverifiable", "not_code")

# Verdicts that retire an item rather than producing work. Both are wins.
RETIRING = {"already_done", "stale"}


@dataclass
class Item:
    path: Path
    id: int
    name: str
    status: str
    priority: str
    created: str
    body: str
    board: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def age_days(self) -> int:
        try:
            created = datetime.fromisoformat(self.created.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - created).days
        except (ValueError, AttributeError):
            return 0


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), parts[2]


def load_item(path: Path) -> Item | None:
    try:
        fm, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    m = re.match(r"^(\d+)[-_]", path.name)
    if not m:
        return None
    name = ""
    for line in body.splitlines():
        if line.startswith("# "):
            name = line[2:].strip()
            break
    return Item(
        path=path, id=int(m.group(1)), name=name or path.stem,
        status=str(fm.get("status", "draft")), priority=str(fm.get("priority", "medium")),
        created=str(fm.get("created", "")), body=body,
        board=str(fm.get("board", "") or ""),
        tags=[str(t) for t in (fm.get("tags") or [])],
    )


def open_items(boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> list[Item]:
    """Open items, restricted to `boards` unless it is None."""
    wanted = {b.lower() for b in boards} if boards else None
    out = []
    for path in sorted(BACKLOG_DIR.glob("*.md")):
        item = load_item(path)
        if not item or item.status not in OPEN_STATUSES:
            continue
        if wanted is not None and item.board.lower() not in wanted:
            continue
        out.append(item)
    return out


def triaged_ids(ledger: Path) -> dict[int, str]:
    """{item_id: verdict} from prior triage runs, so nothing is re-checked."""
    seen: dict[int, str] = {}
    if not ledger.exists():
        return seen
    try:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") == "backlog_triage" and d.get("item_id") is not None:
                seen[int(d["item_id"])] = d.get("verdict", "")
    except OSError:
        pass
    return seen


def select_candidate(ledger: Path,
                     boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> Item | None:
    """Oldest untriaged open item first.

    Oldest-first deliberately: age is the best available proxy for staleness,
    and the point of this pipeline is to find out which old items are still
    real. Priority ordering would front-load the items most likely to be
    genuine, which is exactly backwards for a first pass over a stale backlog.
    """
    seen = triaged_ids(ledger)
    candidates = [i for i in open_items(boards) if i.id not in seen]
    if not candidates:
        return None
    return sorted(candidates, key=lambda i: (i.created or "9999", i.id))[0]


def record_verdict(item: Item, verdict: str, evidence: str, *,
                   check: str = "", close: bool = False) -> Path:
    """Append the verdict to the item's activity log, optionally closing it.

    Always writes the evidence, never just the conclusion. An item closed as
    stale with no stated reason is indistinguishable from one closed by
    mistake, and the whole value of this pass is that a human can audit it
    later.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    text = item.path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    entry = f"**{stamp}** — selfmod triage: **{verdict}**. {evidence.strip()}"
    if check:
        entry += f" Check: `{check}`"
    log = list(fm.get("activity_log") or [])
    log.append(entry)
    fm["activity_log"] = log
    fm["updated"] = stamp

    if close and verdict in RETIRING:
        fm["status"] = "done"
        fm["selfmod_retired"] = verdict

    section = (f"\n\n## Selfmod triage — {stamp[:10]}\n\n"
               f"**Verdict:** {verdict}\n\n{evidence.strip()}\n")
    if check:
        section += f"\n**Premise check:**\n```\n{check.strip()}\n```\n"

    item.path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body.rstrip()}{section}",
        encoding="utf-8")
    return item.path


def summarize(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> dict:
    seen = triaged_ids(ledger)
    counts: dict[str, int] = {}
    for verdict in seen.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    items = open_items(boards)
    total_open = len(items)
    return {
        "boards": list(boards) if boards else "all",
        "open_items": total_open,
        "triaged": len(seen),
        "untriaged": max(0, total_open - sum(1 for i in items if i.id in seen)),
        "verdicts": counts,
        "retired": sum(counts.get(v, 0) for v in RETIRING),
        "confirmed": counts.get("confirmed", 0),
    }
