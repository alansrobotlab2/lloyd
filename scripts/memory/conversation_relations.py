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
# Fact graph lives under ~/lloyd, NOT the vault (migrated 2026-06-03).
FACTS_ROOT = Path.home() / "lloyd" / "_pipeline" / "vault-derived" / "facts"
TRAJECTORY_DIR = Path.home() / "lloyd" / "_pipeline" / "trajectories"
PROPOSALS_FILE = Path.home() / "lloyd" / "_pipeline" / "conversation-relation-proposals.json"
RELATIONS_INDEX = Path.home() / "lloyd" / "_pipeline" / "relations-index.json"
LLOYD_SESSIONS = Path.home() / "lloyd" / "sessions"
AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"

# The skill_name of the autonomy task that runs this script. Its frontmatter is
# where the Stage-2 model decision lives — see resolve_llm_target().
TASK_SKILL_NAME = "conversation-relation-linking"

# Fallback only. These used to be THE endpoint and THE model: a hardcoded
# :8096/primary that no config could move, so Stage 2 always spent its
# MAX_CLASSIFY_PER_RUN batch on the contended RTX PRO 6000 even though task #51
# was deliberately moved to the secondary slot on the idle 3090 (#420).
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
# Stage-2 classify gate. Pre-classification confidence is capped at min(0.7, weight/2),
# so co-access weights rarely reach 0.8 and proposals piled up unclassified (911 stuck
# as of 2026-06-03). Lowered to match MIN_AGGREGATE_WEIGHT so every proposed pair gets
# LLM-typed. Classification is non-destructive; auto-approve stays gated at 0.85 below.
LLM_CLASSIFY_THRESHOLD = 0.5 # minimum to run Stage 2
# Per-run cap on Stage-2 LLM classification so the every-15min #51 task drains a large
# pending backlog (911 as of 2026-06-03) in bounded batches instead of attempting all
# at once and timing out. Progress is saved incrementally so a kill never loses work.
MAX_CLASSIFY_PER_RUN = 40

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

# Direct file access, as the current harness names it. The legacy lowercase
# names below left the tool pool when vault access moved to Read/Grep with
# absolute paths; the extractor still listed only those, so Stage 1 saw about
# 3.5 % of the vault accesses it exists to mine and #51 exited 0 on 4 MB of
# unreadable input while looking healthy (#420). The legacy names stay: they
# still appear in older trajectory files and in a few live calls.
SDK_DIRECT_TOOLS = {"Read", "Write", "Edit"}
LEGACY_DIRECT_TOOLS = {"vault_read", "vault_write", "file_read", "read_file",
                       "file_write", "write_file", "file_edit", "patch",
                       "mem_get", "mem_write"}
DIRECT_FILE_TOOLS = SDK_DIRECT_TOOLS | LEGACY_DIRECT_TOOLS

# Param keys that can carry the path, in precedence order: a call contributing
# both keys still counts as one document. The SDK tools put the path under
# `file_path`; recognising the names alone yielded 0 pairs, so both are read.
PATH_PARAM_KEYS = ("path", "file_path")

SKILL_ACCESS_TOOLS = {"skills_get", "skill_view", "skills_read"}

# Tools whose *result* text carries paths. Unchanged by #420 on purpose: this
# branch cannot currently yield a single document whatever its membership — its
# regex emits bare `segment/….md` strings, which normalize_vault_path rejects at
# its `for/else` because they carry no vault prefix — so admitting `Grep` here
# would be a no-op dressed as a fix. Finding recorded on #420; the direct-file
# tools above are what Stage 1 actually mines.
SEARCH_RESULT_TOOLS = {"vault_search", "vault_recall", "file_grep"}


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

        # Direct file access tools — the SDK names (Read/Write/Edit) carry the
        # path under file_path, the legacy names under path. normalize_vault_path
        # still rejects anything outside the vault, so widening the names is safe.
        if name in DIRECT_FILE_TOOLS:
            for key in PATH_PARAM_KEYS:
                p = _extract_path_from_param(params.get(key, ""))
                if p:
                    paths_found.append(p)
                    break

        # Skill access
        elif name in SKILL_ACCESS_TOOLS:
            skill_name = params.get("name", "")
            if skill_name:
                candidate = f"skills/{skill_name}/SKILL.md"
                if (VAULT / candidate).exists():
                    paths_found.append(candidate)

        # Search result parsing — check result_summary for paths
        elif name in SEARCH_RESULT_TOOLS:
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


def extract_stage1(
    trajectory_dir: Path,
    since_date: Optional[str] = None,
) -> tuple[list[dict], set[str]]:
    """Stage 1 in one pass: co-access pairs, plus which dates produced docs.

    The second return value is the set of file dates that yielded at least one
    vault doc. The watermark may only advance past those: setting it to
    `sorted(glob)[-1]` unconditionally marked every blind day as processed and
    the extractor never got another chance at it (#420).

    Returns:
        (pairs, dates_with_docs) — pairs carry weights, see aggregate_pairs().
    """
    pairs = []
    dates_with_docs: set[str] = set()

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
                if not docs:
                    continue
                dates_with_docs.add(file_date)
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

    return pairs, dates_with_docs


def extract_co_access_pairs(
    trajectory_dir: Path,
    since_date: Optional[str] = None,
) -> list[dict]:
    """Stage 1's co-access pairs only — the measurement #420 is graded on.

    Args:
        trajectory_dir: Path to directory containing YYYY-MM-DD.jsonl files
        since_date: Only process files from this date onward (YYYY-MM-DD)
    """
    return extract_stage1(trajectory_dir, since_date)[0]


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


def _task_model_alias() -> Optional[str]:
    """The `model:` declared by the autonomy task that runs this script.

    Task #51 was moved primary → secondary on 2026-09-03 so Stage 2 would run on
    the idle RTX 3090 instead of queueing behind the primary's two worker slots
    on the contended RTX PRO 6000 — and that frontmatter change never reached
    this script, which asked :8096 by constant. The decision belongs to the task
    file; this reads it (#420). #937 owns the disagreement between #51's
    frontmatter and its activity log; whichever way that lands, this follows it.
    """
    try:
        task_files = sorted(AUTONOMY_DIR.glob("*.md"))
    except OSError:
        return None
    for f in task_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        skill = model = None
        for line in parts[1].splitlines():
            if line.startswith("skill_name:"):
                skill = line.split(":", 1)[1].strip().strip("'\"")
            elif line.startswith("model:"):
                model = line.split(":", 1)[1].strip().strip("'\"")
        if skill == TASK_SKILL_NAME:
            return model or None
    return None


def resolve_llm_target(model_alias: Optional[str] = None) -> tuple[str, str]:
    """(chat-completions endpoint, model name), resolved from config.yaml.

    The alias comes from the task frontmatter (or an explicit argument) and the
    base_url comes from `models:` via app.config, so moving the task between the
    primary and secondary slots moves Stage 2 with it instead of leaving a port
    baked into the script. Only when that resolution finds nothing do these fall
    back to the module constants.
    """
    alias = model_alias or _task_model_alias()
    if alias:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        try:
            from app.config import MODEL_CONFIGS, resolve_model_alias
        except Exception:
            return LLM_ENDPOINT, LLM_MODEL
        alias = resolve_model_alias(alias)  # honours secondary_enabled=false
        cfg = MODEL_CONFIGS.get(alias) or {}
        base = (cfg.get("base_url")
                or (cfg.get("env") or {}).get("ANTHROPIC_BASE_URL") or "")
        if base:
            return base.rstrip("/") + "/v1/chat/completions", alias
    return LLM_ENDPOINT, LLM_MODEL


def classify_relationship(doc_a: str, doc_b: str, context: str,
                          endpoint: Optional[str] = None,
                          model: Optional[str] = None) -> Optional[dict]:
    """Use LLM to classify relationship type and extract reason."""
    if endpoint is None:
        endpoint, model = resolve_llm_target(model)
    prompt = CLASSIFY_PROMPT.format(doc_a=doc_a, doc_b=doc_b, context=context)

    payload = json.dumps({
        "model": model or LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
        # primary is a reasoning model; without this it spends the whole token budget
        # "thinking" and returns empty content (finish_reason=length). Classification
        # needs no reasoning — force a direct JSON answer.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        endpoint,
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
    """Remove proposals that duplicate existing relations-index entries.

    The index writes its rows under `relationships`; this read `edges`, so
    it matched nothing and the dedupe was a no-op against 486,961 existing
    relations (2026-09-03 review). Both keys are accepted now so an older
    index file still dedupes.
    """
    if not RELATIONS_INDEX.exists():
        return proposals

    try:
        index = json.loads(RELATIONS_INDEX.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return proposals

    rows = index.get("relationships") or index.get("edges") or []
    existing = set()
    for edge in rows:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        existing.add((src, tgt))
        existing.add((tgt, src))  # bidirectional check

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


def provenance_pointer(p: dict) -> Optional[str]:
    """Resolve a proposal back to the trajectory evidence behind it.

    Proposals carry their sessions under `evidence.sessions` (written by
    cmd_incremental). The keys this used to read — `evidence_trajectory` and
    `source_doc` — are written by nothing, so all 33 conversation edges landed
    with source_doc NULL and the claim that approved links are auditable did
    not hold (#420). `evidence_trajectory` is still honoured for proposals that
    carry it.
    """
    sessions = [s for s in (p.get("evidence") or {}).get("sessions") or []
                if isinstance(s, str) and s]
    if sessions:
        first = sessions[0]
        m = re.search(r"(\d{8})", first)
        if m:
            d = m.group(1)
            ptr = f"_pipeline/trajectories/{d[:4]}-{d[4:6]}-{d[6:]}.jsonl#{first}"
        else:
            ptr = f"sessions/{first}"
        if len(sessions) > 1:
            ptr += f" (+{len(sessions) - 1} more)"
        return ptr
    return p.get("evidence_trajectory") or p.get("source_doc") or None


def land_approved_edges(proposals: list[dict]) -> int:
    """Write approved links into the edge store, typed and attributable.

    Approving a proposal used to change a status field in a JSON file that
    nothing downstream read, so this task produced no graph. Each approved pair
    becomes an edge of the type Stage 2 classified — co_accessed only when
    nothing classified it — with provenance INFERRED and a pointer to the
    trajectory session that evidenced it, which is what makes it visible to
    retrieval and auditable afterwards.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from app.kg_store import StoreUnavailable, store

    landed = 0
    unattributed = 0
    try:
        st = store()
    except StoreUnavailable as exc:
        print(f"  [edges] store unavailable ({exc}); nothing landed")
        return 0
    with st.transaction():
        for p in proposals:
            if p.get("status") != "approved" or p.get("edge_id"):
                continue
            src, tgt = p.get("source"), p.get("target")
            if not src or not tgt or src == tgt:
                continue
            # An edge with no provenance pointer is unauditable, which is the
            # one thing the 2026-09-04 landing path promised. Leave it approved
            # and unscored instead of writing another NULL-source_doc row.
            source_doc = provenance_pointer(p)
            if not source_doc:
                unattributed += 1
                continue
            edge_id = st.edges.add({
                "source": src, "target": tgt,
                # Proposals carry the Stage-2 verdict under `type`; the key read
                # here before (`relation_type`) is written by nothing, so the
                # classification was discarded and every edge landed as
                # co_accessed (116 proposals carried a type, 33 edges landed,
                # all co_accessed — #420).
                "type": p.get("type") or p.get("relation_type") or "co_accessed",
                "confidence": float(p.get("confidence", 0.85)),
                "provenance": "INFERRED",
                "source_doc": source_doc,
                "evidence": (p.get("reason") or "")[:500] or None,
            }, origin="conversation")
            p["edge_id"] = edge_id
            landed += 1
    if unattributed:
        print(f"  [edges] {unattributed} approved proposal(s) carry no evidence "
              f"sessions or trajectory pointer — not landed")
    return landed


# ── CLI Commands ─────────────────────────────────────────────────────────────

def cmd_incremental():
    """Stage 1: Extract co-access pairs from new trajectory data."""
    data = load_proposals()
    watermark = data.get("watermark", {})
    since = watermark.get("last_trajectory_date")

    print(f"Stage 1: Extracting co-access pairs (since={since or 'all'})")
    pairs, dates_with_docs = extract_stage1(TRAJECTORY_DIR, since_date=since)
    print(f"  Raw pairs extracted: {len(pairs)}")
    print(f"  Trajectory days that yielded vault docs: {len(dates_with_docs)}")

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

    # Update the watermark only as far as the newest day that actually yielded
    # documents. It used to jump to sorted(glob)[-1] unconditionally, so a day
    # the extractor was blind to got marked as processed and never revisited —
    # 2026-09-09 → 09-12 burned that way while Stage 1 saw nothing (#420).
    prev_wm = watermark.get("last_trajectory_date")
    if dates_with_docs:
        newest = max(dates_with_docs)
        if not prev_wm or newest > prev_wm:
            data["watermark"]["last_trajectory_date"] = newest
    else:
        print(f"  Watermark held at {prev_wm or 'none'} — no day under "
              f"{TRAJECTORY_DIR} since then yielded a vault document")

    # The old key was named sessions_processed but always held a pair count.
    carried = watermark.get("pairs_processed",
                            watermark.get("sessions_processed", 0))
    data["watermark"]["pairs_processed"] = int(carried) + len(pairs)
    data["watermark"].pop("sessions_processed", None)

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

    # Prune proposals whose source sessions no longer exist (old UUID-named sessions
    # that predate the current timestamp naming). They can never be classified and
    # would otherwise clog the front of every bounded batch with skips. Mark them so
    # they leave the pending pool. Spend each run's batch only on resolvable proposals.
    orphaned = 0
    resolvable = []
    for p in candidates:
        if any(find_session_file(s) for s in p.get("evidence", {}).get("sessions", [])[:3]):
            resolvable.append(p)
        else:
            p["status"] = "skipped-no-session"
            orphaned += 1
    if orphaned:
        print(f"Stage 2: pruned {orphaned} proposals with no resolvable session "
              f"(status=skipped-no-session)")
        save_proposals(data)
    candidates = resolvable

    if not candidates:
        print("Stage 2: No classifiable candidates with resolvable sessions.")
        return

    total_eligible = len(candidates)
    if total_eligible > MAX_CLASSIFY_PER_RUN:
        print(f"Stage 2: {total_eligible} eligible; capping this run to "
              f"{MAX_CLASSIFY_PER_RUN} (remaining drain on subsequent runs)")
        candidates = candidates[:MAX_CLASSIFY_PER_RUN]

    endpoint, model = resolve_llm_target()
    print(f"Stage 2: Classifying {len(candidates)} proposals via {model} @ {endpoint}")
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

        result = classify_relationship(p["source"], p["target"], context,
                                       endpoint=endpoint, model=model)
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

        # Incremental save so a timeout/kill mid-batch never loses classified work
        if classified % 10 == 0:
            save_proposals(data)

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
    """Auto-approve strong proposals older than 48h, and land them as edges."""
    data = load_proposals()
    # Also deduplicate against index
    before = len(data["proposals"])
    data["proposals"] = deduplicate_against_index(data["proposals"])
    print(f"Deduplicated against relations-index: {before - len(data['proposals'])} dropped")
    count = auto_approve_strong(data["proposals"])
    landed = land_approved_edges(data["proposals"])
    save_proposals(data)
    print(f"Auto-approved: {count}")
    print(f"Edges landed in the store: {landed}")


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
