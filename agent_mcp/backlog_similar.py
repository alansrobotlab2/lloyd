"""Write-time similarity for `backlog_write_task`: is this finding already on the board?

The triage prompt used to say "first run `backlog_tasks` to be sure no item
already covers it". That tool has no text search — it returns ~800 title-only
rows — so the check was a ritual, and the loop filed the same finding again
every time a round re-ran: #549 ran four times in 110 minutes and filed
#788, #795 and #799 for one dead config floor; #570 ran six times and filed
three "Further findings" items carrying the same two findings.

Two legs, and the second is mandatory:

* **lexical** — token Jaccard over the title and the first few hundred chars
  of every item on disk (the `research_store._similar` rule: accept on
  jaccard OR shared-token count, small stoplist). Reads disk, so it sees an
  item written one second ago.
* **semantic** — the qmd daemon's reranked vector search over the `backlog`
  collection it already embeds. Its score is graded (a verbatim title scores
  its own item 1.0, the nearest *distinct* neighbour 0.55-0.60) but an
  unrelated query still put 0.75 on its top hit, so the reranker score alone
  never merges: rule A needs the lexical leg to agree.

Rule B exists for the qmd watcher's debounce: the daemon cannot have embedded
a file written seconds ago, and the only thing that matches a seconds-old
item that strongly is the same session filing the same finding twice.

Everything here fails open. A daemon that is down, a malformed row, a
missing config block — each costs the advisory list, never the write.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from app.backlog_status import OPEN_STATUSES

DEDUPE_LOG = Path.home() / ".local" / "state" / "lloyd-automod" / "dedupe.jsonl"

DEFAULTS: dict = {
    "enabled": True,
    # False = observation mode: compute and return `similar`, never merge.
    "merge": True,
    "threshold": 0.78,          # reranker score floor for rule A
    "lexical_min": 0.4,         # jaccard floor (either this or shared_min)
    "shared_min": 4,            # shared-token floor
    "recent_lexical_min": 0.6,  # rule B: strong lexical match on a very new item
    "recent_window_seconds": 600,
    "timeout_seconds": 5.0,
    "limit": 5,
    "head_bytes": 2048,
    "body_chars": 600,
}

# research_store's stoplist plus the words every backlog handoff repeats.
# Kept small on purpose: a big stoplist merges items that differ only in a
# word it dropped.
_STOPWORDS = frozenset("""
a an and are as at be by for from in into is of on or the to via vs with
using based real time model models learning agent agents robot robotic robots
policy policies system systems approach approaches new current latest
item items backlog lloyd automod found while implementing split triage
file files test tests run runs should still after before when that this
""".split())

_ID_RE = re.compile(r"^(\d+)[-_]")
_QMD_ID_RE = re.compile(r"(?:^|/)(\d+)[-_][^/]*\.md$")


def dedupe_config() -> dict:
    """`backlog.dedupe` from config.yaml over the defaults above. Lazy, so a
    broken config import costs the block, not the tool."""
    cfg = dict(DEFAULTS)
    try:
        from app.config import CONFIG
        block = ((CONFIG or {}).get("backlog") or {}).get("dedupe") or {}
        if isinstance(block, dict):
            cfg.update({k: v for k, v in block.items() if k in DEFAULTS})
    except Exception:  # noqa: BLE001 — fail open
        pass
    return cfg


def _tokens(text: str) -> set[str]:
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return {w for w in t.split() if len(w) > 2 and w not in _STOPWORDS}


def candidate_text(name: str, description: str, *, body_chars: int = 500) -> str:
    return f"{name or ''}\n{(description or '')[:body_chars]}"


def _parse_created(value) -> float | None:
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value or "").strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _head(path: Path, *, head_bytes: int, body_chars: int) -> dict | None:
    """Id, title, status, created and the first `body_chars` of the body —
    from the first `head_bytes` of the file, because this runs on every
    create against ~800 files."""
    m = _ID_RE.match(path.name)
    if not m:
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(head_bytes)
    except OSError:
        return None
    status, created = "", None
    body = head
    if head.startswith("---"):
        parts = head.split("---", 2)
        if len(parts) >= 3:
            fm, body = parts[1], parts[2]
            sm = re.search(r"^status:\s*['\"]?([A-Za-z_]+)", fm, re.M)
            cm = re.search(r"^created:\s*['\"]?([^'\"\n]+)", fm, re.M)
            status = sm.group(1) if sm else ""
            created = _parse_created(cm.group(1).strip()) if cm else None
    tm = re.search(r"^#\s+(.+)$", body, re.M)
    title = tm.group(1).strip() if tm else path.stem
    text = body[tm.end():] if tm else body
    return {"id": int(m.group(1)), "title": title, "status": status or "draft",
            "created": created, "text": text.strip()[:body_chars]}


def lexical_candidates(text: str, *, backlog_dir: Path, cfg: dict | None = None,
                       exclude: frozenset[int] | set[int] = frozenset()) -> list[dict]:
    """Token-overlap neighbours over every item on disk. Sub-100 ms for 800
    files; deliberately Python over the heads rather than an index that has
    to be kept in sync."""
    cfg = cfg or dedupe_config()
    want = _tokens(text)
    if not want or not backlog_dir.exists():
        return []
    out: list[dict] = []
    for path in backlog_dir.glob("*.md"):
        row = _head(path, head_bytes=int(cfg["head_bytes"]), body_chars=int(cfg["body_chars"]))
        if not row or row["id"] in exclude:
            continue
        have = _tokens(f"{row['title']}\n{row['text']}")
        shared = want & have
        if not shared:
            continue
        jaccard = len(shared) / len(want | have)
        if jaccard < float(cfg["lexical_min"]) and len(shared) < int(cfg["shared_min"]):
            continue
        out.append({"id": row["id"], "title": row["title"], "status": row["status"],
                    "created": row["created"], "lexical": round(jaccard, 3),
                    "shared": len(shared)})
    out.sort(key=lambda r: (r["lexical"], r["shared"]), reverse=True)
    return out


def semantic_candidates(text: str, *, limit: int = 6, timeout: float = 5.0) -> list[dict] | None:
    """The qmd daemon's reranked vec search over the `backlog` collection.
    `None` on ANY failure — the caller must tell "no neighbours" from "no
    daemon", because only the first is evidence."""
    try:
        import urllib.request
        from agent_mcp.vault import QMD_DAEMON_URL, _qmd_sanitize
        q = _qmd_sanitize(text)[:1000]
        if not q:
            return []
        payload = {"searches": [{"type": "vec", "query": q}],
                   "collections": ["backlog"], "limit": int(limit), "rerank": True}
        req = urllib.request.Request(QMD_DAEMON_URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        out: list[dict] = []
        for r in data.get("results", []) or []:
            m = _QMD_ID_RE.search(str(r.get("file") or ""))
            if not m:
                continue
            try:
                score = float(r.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            out.append({"id": int(m.group(1)), "score": round(score, 3)})
        return out
    except Exception:  # noqa: BLE001 — fail open
        return None


def similar_items(name: str, description: str, board: str | None = None, *,
                  backlog_dir: Path, limit: int | None = None,
                  exclude: frozenset[int] | set[int] = frozenset(),
                  cfg: dict | None = None,
                  semantic=None) -> list[dict]:
    """Open-or-closed items that look like this one, best first. Each row:
    `{id, title, status, score, lexical, shared, created, source}` where
    `source` is `both`, `lex` or `vec`."""
    cfg = cfg or dedupe_config()
    semantic = semantic or semantic_candidates
    text = candidate_text(name, description)
    lex = {r["id"]: r for r in lexical_candidates(text, backlog_dir=backlog_dir, cfg=cfg,
                                                   exclude=exclude)}
    vec = semantic(text, limit=max(int(cfg["limit"]), 6), timeout=float(cfg["timeout_seconds"]))
    rows: dict[int, dict] = {}
    for iid, r in lex.items():
        rows[iid] = {**r, "score": 0.0, "source": "lex"}
    for r in (vec or []):
        iid = r["id"]
        if iid in exclude:
            continue
        if iid in rows:
            rows[iid]["score"] = r["score"]
            rows[iid]["source"] = "both"
            continue
        head = None
        for path in backlog_dir.glob(f"{iid}-*.md"):
            head = _head(path, head_bytes=int(cfg["head_bytes"]), body_chars=int(cfg["body_chars"]))
            break
        if not head:
            continue
        rows[iid] = {"id": iid, "title": head["title"], "status": head["status"],
                     "created": head["created"], "lexical": 0.0, "shared": 0,
                     "score": r["score"], "source": "vec"}
    out = sorted(rows.values(), key=lambda r: (r["score"], r["lexical"], r["shared"]), reverse=True)
    n = int(limit if limit is not None else cfg["limit"])
    return out[:n]


def merge_target(similar: list[dict], *, cfg: dict | None = None,
                 now: float | None = None) -> dict | None:
    """The one open item a spawn-tagged write should be appended to, or None.

    Rule A: reranker score over the threshold AND the lexical leg agrees.
    Rule B: a strong lexical match on an item created inside the recent
    window — the same session filing the same finding twice, before qmd has
    seen the first copy. The reranker score alone never merges.
    """
    cfg = cfg or dedupe_config()
    now = time.time() if now is None else now
    best: dict | None = None
    for r in similar:
        if r.get("status") not in OPEN_STATUSES:
            continue
        lex_ok = (float(r.get("lexical") or 0) >= float(cfg["lexical_min"])
                  or int(r.get("shared") or 0) >= int(cfg["shared_min"]))
        rule_a = float(r.get("score") or 0) >= float(cfg["threshold"]) and lex_ok
        created = r.get("created")
        rule_b = (float(r.get("lexical") or 0) >= float(cfg["recent_lexical_min"])
                  and created is not None
                  and 0 <= now - float(created) <= float(cfg["recent_window_seconds"]))
        if not (rule_a or rule_b):
            continue
        cand = {**r, "rule": "A" if rule_a else "B"}
        if best is None or (cand["score"], cand["lexical"]) > (best["score"], best["lexical"]):
            best = cand
    return best


def log_decision(rec: dict, path: Path | None = None) -> None:
    """Every decision, for tuning the threshold from data rather than from
    the three points measured when this landed. Best-effort."""
    try:
        p = path or DEDUPE_LOG
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **rec}, sort_keys=True) + "\n")
    except Exception:  # noqa: BLE001
        pass
