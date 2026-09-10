#!/usr/bin/env python3
"""skill_verdicts.py — a keyed ledger of skill-candidate verdicts (backlog #530).

The nightly skill pipeline adjudicates candidates and then forgets it. Every
disposition is written down — `REVIEW-LOG.md`, and a `status:` line in each
candidate's frontmatter — and none of it is read back before the next run.
`mine-trajectories.py` regenerates a date-stamped candidate file per pattern each
night, so yesterday's `reviewed_no_skill` never reaches today's input, and the
consolidator's own skip list (`nightly-skill-consolidation/SKILL.md` 1.3) named only
`consolidated` and `noise` while the statuses actually in use were
`rejected_false_positive` (12 files) and `reviewed_no_skill` (22). The result is on
disk: all seven error patterns rejected on 2026-09-06 came back as fresh files on
2026-09-09 and were hand-rejected again, one `review_reason` at a time. Run #57
named this in its own words on 2026-09-01: "the miner regenerated both files as
`pending_review`, silently wiping the previous review — that's the real process
bug here."

WikiSkill's Skill Proposer solves the same problem with one read: its step 2 is
"read the impact log to see what was tried before, including the full context of
rejected proposals", and its step 5 permits "no action at all" as a legal answer.
Its load-bearing ablation — the wiki helps only when the *proposer* reads it, and
handing it to the executor makes things worse — says the value is in accumulated,
already-adjudicated knowledge, not in rewriting skills more often. So this ledger is
a development-side instrument. It must never be injected into a runtime prompt.

Two invariants:

* **Append-only, latest-wins per key.** A verdict is never edited or deleted; a
  reopen is a new line. Rollback is asymmetric the way WikiSkill's is: a skill can
  be reverted, the record of why it was rejected never is.
* **`evidence_cmd` is required.** A prose verdict is an assertion the next run can
  only inherit. A re-executable check is what lets a later run *falsify* it, which is
  the weakness #525 records on the worker-evidence path.

Stickiness is capped, because a verdict that outlives its evidence becomes a veto on
real work: a terminal verdict reopens itself `REOPEN_AFTER_DAYS` after it was
decided, or immediately once the pattern's live `occurrences` exceed
`REOPEN_OCCURRENCE_GROWTH` times the count at the time of the decision.

Field names deliberately match the target-keyed decision ledger proposed in #588
(`pattern_key` there is its `target_key`), which explicitly defers to this item for
the skill-candidate domain and is meant to absorb this store, not compete with it.
Do not build a second one.

Usage:
    skill_verdicts.py check   --candidates DIR    # what must not be proposed again
    skill_verdicts.py record  --pattern K --verdict V --reason R --evidence-cmd C
    skill_verdicts.py seed    --candidates DIR    # harvest existing dispositions once
    skill_verdicts.py list                        # latest-wins view
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Append-only. `~/lloyd/_pipeline` is gitignored, same as `REVIEW-LOG.md` beside it.
DEFAULT_STORE = Path.home() / "lloyd" / "_pipeline" / "skills" / "reviews" / "verdicts.jsonl"
DEFAULT_CANDIDATES = Path.home() / "lloyd" / "_pipeline" / "skills" / "candidates"

# Verdicts that mean "this pattern has been adjudicated; do not propose it again".
# `reviewed_no_skill` covers "real signal, an installed skill already owns it", which
# is the most common one on this board and the one that kept coming back.
TERMINAL_VERDICTS = frozenset({
    "rejected_false_positive",
    "reviewed_no_skill",
    "rejected_unverifiable",
    "rejected_artifact_class",
    "archived_content",
})

# A terminal verdict that has aged out or whose evidence has grown >10x is reopened
# rather than trusted. 60 days is roughly two full re-emergence cycles on this board.
REOPEN_AFTER_DAYS = 60
REOPEN_OCCURRENCE_GROWTH = 10.0

# The status the miner writes instead of `pending_review` when a key is terminal.
SUPERSEDED_STATUS = "superseded_by_verdict"

# Frontmatter keys, in order. `pattern_key` is the join key everywhere.
FIELDS = (
    "pattern_key",
    "verdict",
    "reason",
    "evidence_cmd",
    "occurrences_at_decision",
    "decided_at",
    "decided_by",
    "source_candidate",
)

_STATUS_RE = re.compile(r"^status:\s*(.+?)\s*$", re.MULTILINE)
_PATTERN_RE = re.compile(r"^pattern:\s*(\S+)", re.MULTILINE)
_OCCURRENCES_RE = re.compile(r"^occurrences:\s*(\d+)", re.MULTILINE)


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(value: str) -> datetime | None:
    """Parse a decided_at stamp. Returns None if unparseable — a verdict whose date
    cannot be read is treated as un-expiring rather than silently reopened, because
    the reopen path is the one that can re-mint a rejected skill."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def store_path(explicit: str | None = None) -> Path:
    """Explicit path, else $SKILL_VERDICTS_STORE, else the default. The env var exists
    so the nightly job (and a test) can point the ledger somewhere else without a
    code change — `mine-trajectories.py` calls this with no argument."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("SKILL_VERDICTS_STORE")
    return Path(env).expanduser() if env else DEFAULT_STORE



def load_verdicts(store: Path) -> dict[str, dict]:
    """Latest line per `pattern_key`. Missing store is an empty ledger, not an error;
    an unparseable line is skipped with a warning rather than dropping the ledger."""
    verdicts: dict[str, dict] = {}
    path = Path(store)
    if not path.exists():
        return verdicts
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"  warn: {path}:{lineno} is not JSON, skipped", file=sys.stderr)
            continue
        key = (row.get("pattern_key") or "").strip()
        if not key:
            print(f"  warn: {path}:{lineno} has no pattern_key, skipped", file=sys.stderr)
            continue
        verdicts[key] = row  # append-only file, so last write is the live verdict
    return verdicts


def is_terminal(row: dict | None) -> bool:
    return bool(row) and (row.get("verdict") or "").strip() in TERMINAL_VERDICTS


def reopen_reason(row: dict, occurrences: int = 0, now: datetime | None = None) -> str | None:
    """Why a terminal verdict no longer binds, or None while it still does."""
    now = now or datetime.now(tz=timezone.utc)
    decided = _parse_ts(row.get("decided_at") or "")
    if decided is None:
        return None
    if now - decided > timedelta(days=REOPEN_AFTER_DAYS):
        return f"expired after {REOPEN_AFTER_DAYS}d"
    baseline = row.get("occurrences_at_decision") or 0
    try:
        baseline = int(baseline)
    except (TypeError, ValueError):
        baseline = 0
    if baseline > 0 and occurrences > baseline * REOPEN_OCCURRENCE_GROWTH:
        return (
            f"occurrences grew {occurrences}>{int(baseline * REOPEN_OCCURRENCE_GROWTH)}"
            f" (>10x the {baseline} at decision)"
        )
    return None


def terminal_verdict(
    pattern_key: str,
    store: Path | str | None = None,
    occurrences: int = 0,
    verdicts: dict[str, dict] | None = None,
    now: datetime | None = None,
) -> dict | None:
    """The binding terminal verdict for `pattern_key`, or None.

    None means "propose freely": no line, a non-terminal line, or a line the reopen
    rules have lifted. A missing ledger returns None for every key, so the worst this
    mechanism can do is fail open exactly as far as the status quo already does.
    """
    table = verdicts if verdicts is not None else load_verdicts(store_path(store))
    row = table.get((pattern_key or "").strip())
    if not is_terminal(row):
        return None
    if reopen_reason(row, occurrences=occurrences, now=now) is not None:
        return None
    return row


def record_verdict(
    store: Path | str | None = None,
    pattern_key: str = "",
    verdict: str = "",
    reason: str = "",
    evidence_cmd: str = "",
    occurrences: int = 0,
    decided_by: str = "agent",
    source_candidate: str = "",
    decided_at: str | None = None,
) -> dict:
    """Append one decision. Never mutates an earlier line.

    `reason` and `evidence_cmd` are required: without the second, a verdict is an
    assertion a later run can only inherit.
    """
    if not (pattern_key := pattern_key.strip()):
        raise ValueError("pattern_key is required")
    if not (verdict := verdict.strip()):
        raise ValueError("verdict is required")
    if not (reason := reason.strip()):
        raise ValueError(
            "reason is required — a verdict with no stated reason becomes a bare veto"
        )
    if not (evidence_cmd := evidence_cmd.strip()):
        raise ValueError(
            "evidence_cmd is required — a verdict with no re-executable check cannot "
            "be falsified by a later run, only inherited (#525)"
        )
    row = {
        "pattern_key": pattern_key,
        "verdict": verdict,
        "reason": " ".join(reason.split()),
        "evidence_cmd": evidence_cmd,
        "occurrences_at_decision": int(occurrences or 0),
        "decided_at": decided_at or now_iso(),
        "decided_by": decided_by,
        "source_candidate": source_candidate,
    }
    path = store_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_candidate(file: Path) -> tuple[str, str, int]:
    """(pattern_key, status_token, occurrences) from one candidate's frontmatter."""
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    # Frontmatter only: `status:` also appears in mined body examples.
    head = text.split("---", 2)[1] if text.startswith("---") else text[:2000]
    pattern = _PATTERN_RE.search(head)
    status = _STATUS_RE.search(head)
    occ = _OCCURRENCES_RE.search(head)
    status_token = status.group(1).split()[0].strip("_*` ") if status else ""
    return (
        pattern.group(1) if pattern else "",
        status_token,
        int(occ.group(1)) if occ else 0,
    )


def scan_candidates(candidates_dir: Path, store: Path | str | None = None) -> list[dict]:
    """One row per candidate file: its key, whether a verdict blocks it, and why."""
    table = load_verdicts(store_path(store))
    rows = []
    for file in sorted(Path(candidates_dir).glob("candidate-*.md")):
        key, status, occurrences = read_candidate(file)
        if not key:
            continue
        row = table.get(key)
        terminal = is_terminal(row)
        lift = reopen_reason(row, occurrences=occurrences) if terminal else None
        rows.append({
            "file": file.name,
            "pattern_key": key,
            "status": status,
            "occurrences": occurrences,
            "verdict": (row or {}).get("verdict", ""),
            "reason": (row or {}).get("reason", ""),
            "decided_at": (row or {}).get("decided_at", ""),
            "blocked": bool(terminal and lift is None),
            "reopened": lift or "",
        })
    return rows


def cmd_check(args: argparse.Namespace) -> int:
    rows = scan_candidates(Path(args.candidates).expanduser(), args.store)
    skipped = [r for r in rows if r["blocked"]]
    for row in rows:
        if row["blocked"]:
            print(
                f"SKIP {row['pattern_key']} :: {row['verdict']} :: {row['reason']} "
                f"(decided {row['decided_at']}, file {row['file']})"
            )
        elif row["reopened"]:
            print(f"REOPEN {row['pattern_key']} :: {row['reopened']}")
        else:
            print(f"PROCEED {row['pattern_key']}")
    print(f"checked: {len(rows)}  skipped_by_verdict: {len(skipped)}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    try:
        row = record_verdict(
            store=args.store,
            pattern_key=args.pattern,
            verdict=args.verdict,
            reason=args.reason,
            evidence_cmd=args.evidence_cmd,
            occurrences=args.occurrences,
            decided_by=args.decided_by,
            source_candidate=args.source_candidate,
        )
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"recorded {row['pattern_key']} -> {row['verdict']} at {row['decided_at']}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Harvest dispositions already on disk into the ledger, once.

    Reads the `status:` line of each candidate — the same hand-written verdicts the
    runbook used to re-make every night — and gives each one its re-executable check:
    the grep that re-derives that status from that file. Duplicate keys keep the
    newest file's disposition. Skips keys already in the ledger, so re-seeding after
    the key derivation changes is a re-run, not a cleanup.
    """
    table = load_verdicts(store_path(args.store))
    newest: dict[str, tuple[Path, str, int]] = {}
    for file in sorted(Path(args.candidates).expanduser().glob("candidate-*.md")):
        key, status, occurrences = read_candidate(file)
        if key and is_terminal({"verdict": status}):
            prev = newest.get(key)
            if prev is None or file.name > prev[0].name:
                newest[key] = (file, status, occurrences)
    seeded = 0
    for key, (file, status, occurrences) in sorted(newest.items()):
        if key in table:
            print(f"  keep: {key} already in the ledger ({table[key]['verdict']})")
            continue
        text = file.read_text(encoding="utf-8", errors="replace")
        head = text.split("---", 2)[1] if text.startswith("---") else ""
        raw = _STATUS_RE.search(head)
        reason = raw.group(1).strip() if raw else status
        reason = reason.split("—", 1)[1].strip() if "—" in reason else reason
        record_verdict(
            store=args.store,
            pattern_key=key,
            verdict=status,
            reason=reason or status,
            evidence_cmd=f"grep -m1 '^status:' {file}",
            occurrences=occurrences,
            decided_by="seed-from-candidates",
            source_candidate=file.name,
        )
        seeded += 1
        print(f"  seed: {key} -> {status}")
    print(f"seeded: {seeded}  ledger now: {len(load_verdicts(store_path(args.store)))} keys")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    table = load_verdicts(store_path(args.store))
    for key in sorted(table):
        row = table[key]
        print(f"{key}\t{row.get('verdict')}\t{row.get('decided_at')}\t{row.get('reason')}")
    print(f"keys: {len(table)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def with_store(subparser):
        """`--store` per subcommand rather than on the root parser: the runbooks write
        `skill_verdicts.py check --candidates DIR --store PATH`, and a root-level option
        is only accepted *before* the subcommand, which is not how anybody types it."""
        subparser.add_argument(
            "--store", default=None, help=f"verdict ledger path (default {DEFAULT_STORE})")
        return subparser

    p_check = with_store(sub.add_parser("check", help="which candidate patterns a verdict blocks"))
    p_check.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    p_check.set_defaults(func=cmd_check)

    p_record = with_store(sub.add_parser("record", help="append one decision, including 'no action'"))
    p_record.add_argument("--pattern", required=True, help="pattern key, e.g. Edit/not_found")
    p_record.add_argument("--verdict", required=True)
    p_record.add_argument("--reason", required=True)
    p_record.add_argument("--evidence-cmd", dest="evidence_cmd", required=True)
    p_record.add_argument("--occurrences", type=int, default=0)
    p_record.add_argument("--decided-by", dest="decided_by", default="agent")
    p_record.add_argument("--source-candidate", dest="source_candidate", default="")
    p_record.set_defaults(func=cmd_record)

    p_seed = with_store(sub.add_parser("seed", help="harvest existing candidate dispositions once"))
    p_seed.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    p_seed.set_defaults(func=cmd_seed)

    p_list = with_store(sub.add_parser("list", help="latest-wins view of the ledger"))
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
