#!/usr/bin/env python3
"""
conversation_relations.py — Mine session trajectories for document co-access
patterns and propose typed relationships between vault documents.

Stage 1 (deterministic): Extract co-access pairs from trajectory JSONL,
  compute weighted scores, aggregate across sessions.
Stage 2 (LLM-assisted): For high-confidence pairs, extract conversation
  context from raw session JSON and classify relationship type via primary model.

Usage:
    python3 conversation_relations.py --incremental    # Stage 1 only (watermark-gated)
    python3 conversation_relations.py --classify       # Stage 2 only (needs Stage 1 output)
    python3 conversation_relations.py --full           # Both stages, ignore watermark
    python3 conversation_relations.py --stats          # Print statistics
    python3 conversation_relations.py --approve-strong # Auto-approve confidence >= 0.85
"""

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional


# ── Paths ────────────────────────────────────────────────────────────────────

VAULT = Path.home() / "obsidian"
FACTS_ROOT = VAULT / "facts"
TRAJECTORY_DIR = Path.home() / "lloyd" / "_pipeline" / "trajectories"
PROPOSALS_FILE = Path.home() / "lloyd" / "_pipeline" / "conversation-relation-proposals.json"
RELATIONS_INDEX = Path.home() / "lloyd" / "_pipeline" / "relations-index.json"
LLOYD_SESSIONS = Path.home() / "lloyd" / "sessions"

LLM_ENDPOINT = "http://127.0.0.1:8096/v1/chat/completions"
LLM_MODEL = "primary"

# Vault segments that are valid relationship endpoints
VAULT_SEGMENTS = {
    "memory", "knowledge", "projects", "agents", "personal",
    "work", "skills", "backlog", "autonomy", "procedures", "facts",
}

# ── Weighting ────────────────────────────────────────────────────────────────

WEIGHT_ADJACENT = 0.8        # sequence_distance <= 2
WEIGHT_SAME_SESSION = 0.4    # sequence_distance > 2
MIN_AGGREGATE_WEIGHT = 0.5   # minimum to propose
LLM_CLASSIFY_THRESHOLD = 0.8 # minimum to run Stage 2

# Autonomy sessions are maintenance, not meaningful user work
AUTONOMY_WEIGHT_FACTOR = 0.3

# ── Valid relation types ─────────────────────────────────────────────────────

VALID_RELATION_TYPES = {
    "implements", "designed-by",
    "supersedes", "superseded-by",
    "depends-on", "required-by",
    "derived-from", "produces",
    "related-to", "conflicts-with",
}

# ── Tool → vault path extraction rules ───────────────────────────────────────

OBSIDIAN_PREFIXES = [
    str(Path.home() / "obsidian") + "/",
    "~/obsidian/",
]

LLOYD_PREFIXES = [
    str(Path.home()) + "/",
    "~/",
]


def normalize_vault_path(raw_path: str) -> Optional[str]:
    """Normalize a tool param path to vault-relative form.

    Returns vault-relative path like 'memory/2026-03-23.md' or None if not in vault.
    """
    if not raw_path or not isinstance(raw_path, str):
        return None

    path = raw_path.strip().strip("'\"")

    # Strip vault root prefixes
    for prefix in OBSIDIAN_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    else:
        # Not an obsidian path
        return None

    # Must start with a known vault segment
    first_segment = path.split("/")[0] if "/" in path else path
    if first_segment not in VAULT_SEGMENTS:
        return None

    # Must be a markdown file
    if not path.endswith(".md"):
        return None

    # Verify file actually exists
    full = VAULT / path
    if not full.exists():
        return None

    return path


def _extract_path_from_param(value: str) -> Optional[str]:
    """Try to extract a vault path from a parameter value string."""
    if not value or not isinstance(value, str):
        return None
    # Handle truncated params
    if value.startswith("[truncated:"):
        return None
    return normalize_vault_path(value)


def extract_vault_docs_from_trajectory(entry: dict) -> list[dict]:
    """Extract (doc_path, tool_name, sequence) from a trajectory entry's tools.

    Returns list of dicts: {"path": str, "tool": str, "sequence": int}
    """
    docs = []
    tools = entry.get("tools", [])

    for tool in tools:
        name = tool.get("name", "")
        params = tool.get("params_summary", {})
        seq = tool.get("sequence", 0)

        if not isinstance(params, dict):
            continue

        paths_found = []

        # Direct file access tools
        if name in ("vault_read", "vault_write", "file_read", "read_file",
                     "file_write", "write_file", "file_edit", "patch",
                     "mem_get", "mem_write"):
            p = _extract_path_from_param(params.get("path", ""))
            if p:
                paths_found.append(p)

        # Skill access
        elif name in ("skills_get", "skill_view", "skills_read"):
            skill_name = params.get("name", "")
            if skill_name:
                candidate = f"skills/{skill_name}/SKILL.md"
                if (VAULT / candidate).exists():
                    paths_found.append(candidate)

        # Search result parsing — check result_summary for paths
        elif name in ("vault_search", "vault_recall", "file_grep"):
            result = tool.get("result_summary", "")
            if isinstance(result, str):
                # Extract paths from result text
                for match in re.finditer(r'(?:^|[\s"\'(])(' + '|'.join(VAULT_SEGMENTS) + r')/[\w/.-]+\.md', result):
                    p = normalize_vault_path(match.group(0).strip().strip("\"'("))
                    if p:
                        paths_found.append(p)

        for p in paths_found:
            docs.append({"path": p, "tool": name, "sequence": seq})

    return docs


def extract_co_access_pairs(
    trajectory_dir: Path,
    since_date: Optional[str] = None,
) -> list[dict]:
    """Stage 1: Extract document co-access pairs from trajectory JSONL files.

    Args:
        trajectory_dir: Path to directory containing YYYY-MM-DD.jsonl files
        since_date: Only process files from this date onward (YYYY-MM-DD)

    Returns:
        List of co-access pair dicts with weights
    """
    pairs = []

    jsonl_files = sorted(trajectory_dir.glob("*.jsonl"))
    if since_date:
        jsonl_files = [f for f in jsonl_files if f.stem >= since_date]

    for jsonl_file in jsonl_files:
        file_date = jsonl_file.stem
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                session_key = entry.get("session_key", "")
                timestamp = entry.get("timestamp", "")
                is_autonomy = session_key.startswith("autonomy_")

                docs = extract_vault_docs_from_trajectory(entry)
                if len(docs) < 2:
                    continue

                # Generate pairs from all doc combinations in this session
                for i, j in combinations(range(len(docs)), 2):
                    doc_a = docs[i]
                    doc_b = docs[j]

                    if doc_a["path"] == doc_b["path"]:
                        continue

                    # Compute weight based on sequence distance
                    distance = abs(doc_a["sequence"] - doc_b["sequence"])
                    if distance <= 2:
                        weight = WEIGHT_ADJACENT
                    else:
                        weight = WEIGHT_SAME_SESSION

                    # Down-weight autonomy sessions
                    if is_autonomy:
                        weight *= AUTONOMY_WEIGHT_FACTOR

                    # Canonical ordering for dedup
                    path_a, path_b = sorted([doc_a["path"], doc_b["path"]])

                    pairs.append({
                        "doc_a": path_a,
                        "doc_b": path_b,
                        "session_key": session_key,
                        "timestamp": timestamp,
                        "file_date": file_date,
                        "sequence_distance": distance,
                        "tool_a": doc_a["tool"],
                        "tool_b": doc_b["tool"],
                        "weight": weight,
                    })

    return pairs


def aggregate_pairs(pairs: list[dict]) -> dict[tuple, dict]:
    """Aggregate co-access pairs across sessions.

    Uses dampened sum: aggregate_weight = sum(w_i) * (1 - 0.3^n)
    where n = number of distinct sessions.

    Returns dict keyed by (doc_a, doc_b) with aggregated data.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        key = (pair["doc_a"], pair["doc_b"])
        grouped[key].append(pair)

    aggregated = {}
    for key, group in grouped.items():
        sessions = set(p["session_key"] for p in group)
        n = len(sessions)
        raw_weight = sum(p["weight"] for p in group)
        dampened = raw_weight * (1 - 0.3 ** n)

        if dampened < MIN_AGGREGATE_WEIGHT:
            continue

        # Filter: skip daily-note-to-daily-note pairs
        if _is_daily_note(key[0]) and _is_daily_note(key[1]):
            continue

        aggregated[key] = {
            "doc_a": key[0],
            "doc_b": key[1],
            "aggregate_weight": round(dampened, 3),
            "co_access_count": len(group),
            "session_count": n,
            "sessions": sorted(sessions),
            "min_distance": min(p["sequence_distance"] for p in group),
            "tools": list(set(p["tool_a"] for p in group) | set(p["tool_b"] for p in group)),
        }

    return aggregated


def _is_daily_note(path: str) -> bool:
    """Check if a path is a daily note (memory/YYYY-MM-DD.md)."""
    return bool(re.match(r"^memory/\d{4}-\d{2}-\d{2}\.md$", path))


# ── Stage 2: LLM Classification ─────────────────────────────────────────────

CLASSIFY_PROMPT = """You are classifying the relationship between two vault documents based on how they were used together in a conversation.

Document A: {doc_a}
Document B: {doc_b}

Conversation context where both documents were accessed:
---
{context}
---

Based on this context, classify the relationship. Choose exactly one type:
- "depends-on": A requires B to function or be understood
- "derived-from": A was created based on B
- "implements": A is an implementation of what B describes
- "supersedes": A replaces or updates B
- "related-to": A and B are topically related but no stronger relation applies
- "conflicts-with": A and B contain contradictory information

Also extract a one-sentence reason explaining WHY these documents are related, grounded in the conversation context.

Respond ONLY with a JSON object, no other text:
{{"type": "...", "reason": "...", "confidence": 0.0-1.0}}"""


def find_session_file(session_key: str) -> Optional[Path]:
    """Resolve session_key to raw session JSON path.

    Lloyd sessions: YYYYMMDD_HHMMSS_XXXXXX.json
    Autonomy sessions: autonomy_NN_YYYYMMDD.json
    """
    candidate = LLOYD_SESSIONS / f"{session_key}.json"
    if candidate.exists():
        return candidate

    return None


def extract_conversation_context(
    session_path: Path,
    doc_a: str,
    doc_b: str,
    max_chars: int = 4000,
) -> Optional[str]:
    """Extract user/assistant text around co-access of doc_a and doc_b.

    Scans session messages for tool calls that accessed doc_a/doc_b,
    then extracts surrounding user and assistant text.
    """
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    messages = data.get("messages", [])
    if not messages:
        return None

    # Find message indices that reference either doc
    hit_indices = set()
    for i, msg in enumerate(messages):
        msg_text = json.dumps(msg).lower()
        if doc_a.lower() in msg_text or doc_b.lower() in msg_text:
            hit_indices.add(i)

    if not hit_indices:
        return None

    # Extract window: 2 messages before first hit to 2 after last hit
    min_idx = max(0, min(hit_indices) - 2)
    max_idx = min(len(messages), max(hit_indices) + 3)

    context_parts = []
    for i in range(min_idx, max_idx):
        msg = messages[i]
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "\n".join(text_parts)

        if not content or not isinstance(content, str):
            continue

        # Skip tool result noise
        if role == "assistant" and not content.strip():
            continue

        context_parts.append(f"[{role}]: {content[:1000]}")

    context = "\n\n".join(context_parts)
    return context[:max_chars] if context else None


def classify_relationship(doc_a: str, doc_b: str, context: str) -> Optional[dict]:
    """Use LLM to classify relationship type and extract reason."""
    prompt = CLASSIFY_PROMPT.format(doc_a=doc_a, doc_b=doc_b, context=context)

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    }).encode()

    req = urllib.request.Request(
        LLM_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            text = result["choices"][0]["message"].get("content") or ""
            text = text.strip()
            if not text:
                return None
    except Exception as e:
        print(f"  LLM error for ({doc_a}, {doc_b}): {e}", file=sys.stderr)
        return None

    # Parse JSON from response (handle markdown fences)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{[^}]+\}', text)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    rel_type = parsed.get("type", "related-to")
    if rel_type not in VALID_RELATION_TYPES:
        rel_type = "related-to"

    return {
        "type": rel_type,
        "reason": str(parsed.get("reason", ""))[:200],
        "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
    }


# ── Proposals I/O ────────────────────────────────────────────────────────────

def load_proposals() -> dict:
    """Load existing proposals file."""
    if PROPOSALS_FILE.exists():
        try:
            return json.loads(PROPOSALS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "watermark": {"last_trajectory_date": None, "sessions_processed": 0},
        "proposals": [],
        "stats": {},
    }


def save_proposals(data: dict) -> None:
    """Write proposals file."""
    PROPOSALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROPOSALS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def deduplicate_against_index(proposals: list[dict]) -> list[dict]:
    """Remove proposals that duplicate existing relations-index entries."""
    if not RELATIONS_INDEX.exists():
        return proposals

    try:
        index = json.loads(RELATIONS_INDEX.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return proposals

    existing = set()
    for edge in index.get("edges", []):
        key = (edge.get("source", ""), edge.get("target", ""))
        existing.add(key)
        existing.add((key[1], key[0]))  # bidirectional check

    return [
        p for p in proposals
        if (p["source"], p["target"]) not in existing
    ]


def auto_approve_strong(proposals: list[dict], threshold: float = 0.85) -> int:
    """Auto-approve proposals above confidence threshold that are >48h old."""
    now = datetime.now(timezone.utc)
    approved = 0
    for p in proposals:
        if p.get("status") != "pending":
            continue
        if p.get("confidence", 0) < threshold:
            continue
        proposed_at = p.get("proposed_at", "")
        if proposed_at:
            try:
                dt = datetime.fromisoformat(proposed_at.replace("Z", "+00:00"))
                if (now - dt) < timedelta(hours=48):
                    continue
            except ValueError:
                continue
        p["status"] = "approved"
        approved += 1
    return approved


# ── CLI Commands ─────────────────────────────────────────────────────────────

def cmd_incremental():
    """Stage 1: Extract co-access pairs from new trajectory data."""
    data = load_proposals()
    watermark = data.get("watermark", {})
    since = watermark.get("last_trajectory_date")

    print(f"Stage 1: Extracting co-access pairs (since={since or 'all'})")
    pairs = extract_co_access_pairs(TRAJECTORY_DIR, since_date=since)
    print(f"  Raw pairs extracted: {len(pairs)}")

    aggregated = aggregate_pairs(pairs)
    print(f"  Aggregated pairs above threshold: {len(aggregated)}")

    # Merge with existing proposals
    existing_keys = {(p["source"], p["target"]) for p in data["proposals"]}

    new_count = 0
    for key, agg in aggregated.items():
        if (key[0], key[1]) in existing_keys or (key[1], key[0]) in existing_keys:
            # Update existing proposal's evidence
            for p in data["proposals"]:
                if (p["source"] == key[0] and p["target"] == key[1]) or \
                   (p["source"] == key[1] and p["target"] == key[0]):
                    p["evidence"] = {
                        "co_access_count": agg["co_access_count"],
                        "session_count": agg["session_count"],
                        "aggregate_weight": agg["aggregate_weight"],
                        "sessions": agg["sessions"][:10],
                    }
                    break
            continue

        signal_strength = "strong" if agg["aggregate_weight"] >= 1.0 else "moderate"

        data["proposals"].append({
            "source": key[0],
            "target": key[1],
            "type": "related-to",  # default until Stage 2 classifies
            "reason": "",
            "confidence": min(0.7, agg["aggregate_weight"] / 2),  # capped pre-classification
            "signal_strength": signal_strength,
            "evidence": {
                "co_access_count": agg["co_access_count"],
                "session_count": agg["session_count"],
                "aggregate_weight": agg["aggregate_weight"],
                "sessions": agg["sessions"][:10],
            },
            "status": "pending",
            "proposed_at": datetime.now(timezone.utc).isoformat(),
            "classification_source": "co-access",
        })
        new_count += 1

    # Update watermark
    jsonl_files = sorted(TRAJECTORY_DIR.glob("*.jsonl"))
    if jsonl_files:
        data["watermark"]["last_trajectory_date"] = jsonl_files[-1].stem
    data["watermark"]["sessions_processed"] = watermark.get("sessions_processed", 0) + len(pairs)

    data["stats"] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "raw_pairs": len(pairs),
        "aggregated_pairs": len(aggregated),
        "new_proposals": new_count,
        "total_proposals": len(data["proposals"]),
    }

    save_proposals(data)
    print(f"  New proposals: {new_count}")
    print(f"  Total proposals: {len(data['proposals'])}")


def cmd_classify():
    """Stage 2: LLM-classify unclassified high-weight proposals."""
    data = load_proposals()
    candidates = [
        p for p in data["proposals"]
        if p.get("classification_source") == "co-access"
        and p.get("evidence", {}).get("aggregate_weight", 0) >= LLM_CLASSIFY_THRESHOLD
        and p.get("status") == "pending"
    ]

    if not candidates:
        print("Stage 2: No candidates above LLM threshold. Nothing to classify.")
        return

    print(f"Stage 2: Classifying {len(candidates)} proposals via LLM")
    classified = 0

    for p in candidates:
        sessions = p.get("evidence", {}).get("sessions", [])
        context = None

        # Try to find conversation context from session files
        for session_key in sessions[:3]:
            session_file = find_session_file(session_key)
            if session_file:
                context = extract_conversation_context(
                    session_file, p["source"], p["target"]
                )
                if context:
                    break

        if not context:
            print(f"  Skip ({p['source']}, {p['target']}): no session context found")
            continue

        result = classify_relationship(p["source"], p["target"], context)
        if not result:
            print(f"  Skip ({p['source']}, {p['target']}): LLM classification failed")
            continue

        p["type"] = result["type"]
        p["reason"] = result["reason"]
        p["confidence"] = round(
            result["confidence"] * min(1.0, p["evidence"]["aggregate_weight"]),
            3
        )
        p["classification_source"] = "llm"

        if p["confidence"] >= 0.7:
            p["signal_strength"] = "strong"

        classified += 1
        print(f"  Classified: {p['source']} --[{p['type']}]--> {p['target']} "
              f"(conf={p['confidence']})")

    save_proposals(data)
    print(f"  Classified: {classified}/{len(candidates)}")


def cmd_full():
    """Run both stages, ignore watermark."""
    # Reset watermark
    data = load_proposals()
    data["watermark"]["last_trajectory_date"] = None
    save_proposals(data)

    cmd_incremental()
    cmd_classify()


def cmd_stats():
    """Print statistics about current proposals."""
    data = load_proposals()
    proposals = data.get("proposals", [])

    print(f"Proposals file: {PROPOSALS_FILE}")
    print(f"Total proposals: {len(proposals)}")
    print(f"Watermark: {data.get('watermark', {})}")
    print()

    by_status = defaultdict(int)
    by_type = defaultdict(int)
    by_strength = defaultdict(int)
    by_source = defaultdict(int)

    for p in proposals:
        by_status[p.get("status", "unknown")] += 1
        by_type[p.get("type", "unknown")] += 1
        by_strength[p.get("signal_strength", "unknown")] += 1
        by_source[p.get("classification_source", "unknown")] += 1

    print("By status:", dict(by_status))
    print("By type:", dict(by_type))
    print("By strength:", dict(by_strength))
    print("By classification:", dict(by_source))

    if proposals:
        print("\nTop 10 by confidence:")
        top = sorted(proposals, key=lambda p: p.get("confidence", 0), reverse=True)[:10]
        for p in top:
            print(f"  {p['confidence']:.3f}  {p['type']:15s}  {p['source']} <-> {p['target']}")
            if p.get("reason"):
                print(f"         {p['reason']}")


def cmd_approve():
    """Auto-approve strong proposals older than 48h."""
    data = load_proposals()
    # Also deduplicate against index
    data["proposals"] = deduplicate_against_index(data["proposals"])
    count = auto_approve_strong(data["proposals"])
    save_proposals(data)
    print(f"Auto-approved: {count}")


def main():
    parser = argparse.ArgumentParser(description="Conversation-derived relation linking")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--incremental", action="store_true", help="Stage 1: co-access extraction")
    group.add_argument("--classify", action="store_true", help="Stage 2: LLM classification")
    group.add_argument("--full", action="store_true", help="Both stages, ignore watermark")
    group.add_argument("--stats", action="store_true", help="Print statistics")
    group.add_argument("--approve-strong", action="store_true", help="Auto-approve confidence >= 0.85")

    args = parser.parse_args()

    if args.incremental:
        cmd_incremental()
    elif args.classify:
        cmd_classify()
    elif args.full:
        cmd_full()
    elif args.stats:
        cmd_stats()
    elif args.approve_strong:
        cmd_approve()


if __name__ == "__main__":
    main()
