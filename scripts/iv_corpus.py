#!/usr/bin/env python3
"""The tracked Inner Voice intervention corpus: `eval/iv/` (IV plan R3).

Everything Inner Voice ever learned about its own interventions lived in
gitignored runtime data — `usage.db`, the event logs, the metrics series — and
the 2026-09-22 wipe took all of it before that day (#1229 is the same finding
for the baselines). This keeps the part a person paid attention for in the
tree, where a wipe cannot reach it.

Two sources, two files, one row shape:

  seed      `eval/iv/recovered-2026-08-22..09-21.jsonl` — every `[INNER VOICE]`
            line in the transcripts recovered after the wipe, with the text on
            either side, classified by who wrote it (repetition guard, model,
            cancel). The model-written ones carry the hand reading from the IV
            review of 2026-09-24 as `label` (helped / unactionable / obsolete /
            harmful / low_value / failed); everything else is unlabelled.
            Run once; the recovered set does not grow.
  export    `eval/iv/labelled.jsonl` — rows a human thumbed in the Inner Voice
            tab (`inner_voice_observations.verdict`), appended, deduplicated on
            the row's (session, turn, sequence, id). usage.db opened read-only.

Row: {source, session_id, turn_id?, date, kind, trigger?, safeguard?, text,
      request?, before?, after?, label, label_source}

    python scripts/iv_corpus.py seed ~/lloyd-data/recovered/qmd-sessions-pre-wipe/gemma-20260921
    python scripts/iv_corpus.py export
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

_LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LLOYD_HOME))

CORPUS_DIR = _LLOYD_HOME / "eval" / "iv"
SEED_FILE = CORPUS_DIR / "recovered-2026-08-22..09-21.jsonl"
LABELLED_FILE = CORPUS_DIR / "labelled.jsonl"

LABELS = ("helped", "unactionable", "obsolete", "harmful", "low_value", "failed",
          "up", "down")

_IV_LINE = re.compile(r"^user: \[INNER VOICE\]\s*(.*)$")
_TURN_LINE = re.compile(r"^(user|lloyd|assistant):\s?")

# The 2026-09-24 hand reading of the model-written injects, keyed on how each
# one opens. Order matters only where two prefixes could both match; none do.
HAND_READING: tuple[tuple[str, str], ...] = (
    ("You found the extension but the user's actual question", "helped"),
    ("You're at iteration 50 of 60", "helped"),
    ("You have one iteration left before the harness kills this turn", "helped"),
    ("The round is not complete", "helped"),
    ("You're at iteration 57 of 60", "failed"),
    ("You printed the sqlite inspection command as raw JSON text", "unactionable"),
    ("You pasted the Bash call as text", "unactionable"),
    ("You stopped with an empty message", "unactionable"),
    ("You stopped with no output and no tool call. #540", "unactionable"),
    ("You're about to end the turn with no output at all", "unactionable"),
    ("Your final block is missing the VERDICT", "obsolete"),
    ("You ended with no text at all. The vault note is written — now emit", "obsolete"),
    ("Your final line says \"SPAWNED: pending\"", "obsolete"),
    ("Your message is complete on substance but is missing the mandatory closing block", "obsolete"),
    ("You stopped without emitting the final block", "obsolete"),
    ("The goal card explicitly says \"Do not pull the YouTube transcript", "harmful"),
    ("You're still working on the YouTube transcript", "harmful"),
    ("Stop reading the transcript — it's explicitly out of scope", "harmful"),
    ("The report is delivered and complete — call TodoWrite", "low_value"),
    ("The review is delivered and complete — call TodoWrite", "low_value"),
)


def hand_label(text: str) -> str | None:
    for prefix, label in HAND_READING:
        if text.startswith(prefix):
            return label
    return None


def classify(text: str) -> str:
    if text.startswith("Stop: you have"):
        return "guard_repetition"
    return "model_inject"


def _blocks(lines: list[str]) -> list[tuple[str, str]]:
    """(speaker, text) blocks from an exported `role: text` transcript."""
    out: list[tuple[str, list[str]]] = []
    for line in lines:
        m = _TURN_LINE.match(line)
        if m:
            out.append((m.group(1), [line[m.end():]]))
        elif out:
            out[-1][1].append(line)
    return [(who, "\n".join(body).strip()) for who, body in out]


def seed_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.md")):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        blocks = _blocks(lines)
        for i, (who, body) in enumerate(blocks):
            if who != "user" or not body.startswith("[INNER VOICE]"):
                continue
            text = body[len("[INNER VOICE]"):].strip()
            # The request the turn was serving: the last user block before
            # this one that is not itself an [INNER VOICE] line.
            request = next((b for w, b in reversed(blocks[:i])
                            if w == "user" and not b.startswith("[INNER VOICE]")), "")
            before = next((b for w, b in reversed(blocks[:i]) if w != "user"), "")
            after = next((b for w, b in blocks[i + 1:] if w != "user"), "")
            kind = classify(text)
            label = hand_label(text) if kind == "model_inject" else None
            rows.append({
                "source": "recovered", "session_id": path.stem,
                "date": path.parent.name, "kind": kind, "text": text[:1200],
                "request": request[:1200],
                "before": before[-1500:], "after": after[:400],
                "label": label,
                "label_source": "iv-review-2026-09-24" if label else None,
            })
        # Cancels are written as an assistant line, not an [INNER VOICE] user one.
        for who, body in blocks:
            if who != "user" and body.startswith("*(Inner Voice stopped this turn"):
                rows.append({
                    "source": "recovered", "session_id": path.stem,
                    "date": path.parent.name, "kind": "cancel", "text": body[:600],
                    "before": "", "after": "", "label": None, "label_source": None,
                })
    return rows


def export_rows(db: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(inner_voice_observations)")}
        if "verdict" not in cols:
            return []
        rows = conn.execute(
            """SELECT id, session_id, turn_id, sequence_in_turn, trigger, action,
                      reason, content, safeguard, verdict, verdict_at, created_at
                 FROM inner_voice_observations
                WHERE verdict IS NOT NULL ORDER BY id""").fetchall()
    finally:
        conn.close()
    return [{
        "source": "usage.db", "id": r["id"], "session_id": r["session_id"],
        "turn_id": r["turn_id"], "sequence_in_turn": r["sequence_in_turn"],
        "date": (r["created_at"] or "")[:10], "kind": r["action"],
        "trigger": r["trigger"], "safeguard": r["safeguard"],
        "text": (r["content"] or "")[:1200], "reason": (r["reason"] or "")[:600],
        "label": r["verdict"], "label_source": f"thumbs {r['verdict_at'] or ''}".strip(),
    } for r in rows]


def _key(row: dict) -> tuple:
    return (row.get("session_id"), row.get("turn_id"), row.get("sequence_in_turn"),
            row.get("id"))


def append_new(path: Path, rows: list[dict]) -> int:
    """Append rows not already in `path`; a relabel replaces the old line."""
    existing: dict[tuple, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[_key(r)] = r
    changed = 0
    for r in rows:
        k = _key(r)
        if existing.get(k) != r:
            existing[k] = r
            changed += 1
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                                for r in existing.values()))
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("root", type=Path)
    s.add_argument("--out", type=Path, default=SEED_FILE)
    e = sub.add_parser("export")
    e.add_argument("--db", type=Path, default=None)
    e.add_argument("--out", type=Path, default=LABELLED_FILE)
    args = ap.parse_args(argv)
    if args.cmd == "seed":
        rows = seed_rows(args.root.expanduser())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        by: dict[str, int] = {}
        for r in rows:
            by[r["kind"]] = by.get(r["kind"], 0) + 1
        labelled = sum(1 for r in rows if r["label"])
        print(f"seeded {len(rows)} rows into {args.out}: {by}; {labelled} hand-labelled")
        return 0
    db = args.db
    if db is None:
        from app.paths import USAGE_DB
        db = USAGE_DB
    if not Path(db).exists():
        print(f"no usage db at {db}", file=sys.stderr)
        return 1
    n = append_new(args.out, export_rows(Path(db)))
    print(f"{n} labelled row(s) written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
