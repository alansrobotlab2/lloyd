"""Write-time similarity for `backlog_write_task`: is this finding already on the board?

The triage prompt used to say "first run `backlog_tasks` to be sure no item
already covers it". That tool has no text search — it returns ~800 title-only
rows — so the check was a ritual, and the loop filed the same finding again
every time a round re-ran: #549 ran four times in 110 minutes and filed
#788, #795 and #799 for one dead config floor; #570 ran six times and filed
three "Further findings" items carrying the same two findings.

Two legs, and the second is mandatory:

* **lexical** — token Jaccard over the title and the first few hundred chars
  of every item on disk (small stoplist; a neighbour is *listed* on jaccard OR
  shared-token count, the `research_store._similar` rule, but only the
  jaccard can carry a merge). Reads disk, so it sees an item written one
  second ago.
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

import yaml

from app.backlog_status import OPEN_STATUSES

DEDUPE_LOG = Path.home() / ".local" / "state" / "lloyd-automod" / "dedupe.jsonl"

DEFAULTS: dict = {
    "enabled": True,
    # False = observation mode: compute and return `similar`, never merge.
    "merge": True,
    "threshold": 0.78,          # reranker score floor for rule A
    "lexical_min": 0.4,         # jaccard floor: the lexical leg of rule A
    # Shared-token floor for LISTING a lexical neighbour in `similar`. Never a
    # merge leg: until #934 it was OR'd into rule A, and 178 of the first 179
    # merges fired below `lexical_min` on 4-12 tokens of shared vocabulary
    # (the fleet's own tags and handoff boilerplate) against a reranker 1.0.
    "shared_min": 4,
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
# The closing delimiter of a front-matter block: `---` alone on its own line.
# Anchored, because `str.split("---")` also fires on a `---` in prose or
# inside a quoted activity-log entry (the dashboard's `_frontmatter` rule).
_FM_END_RE = re.compile(r"^---[ \t]*$", re.M)
# Bounds the FRONT MATTER, not the prefix read: `head_bytes` is the chunk
# size. A flat 2048-byte read skipped the whole parse for the 587 of 1184
# items whose closing `---` sat past it (#934): every one of them read as
# `status: draft` with the file slug for a title and no `created`, so 284
# closed items were merge targets, rule B was unreachable for them, and the
# lexical leg compared body prose to their raw YAML.
_FM_LIMIT_BYTES = 65536
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
    """Id, title, status, created and the first `body_chars` of the body.
    Read `head_bytes` at a time — the ordinary case is still one small read,
    because this runs on every create against ~1200 files — and on past the
    closing `---` when the front matter is longer, up to `_FM_LIMIT_BYTES`.
    Anything past that with no closing line is malformed rather than large
    and falls through to the no-front-matter reading."""
    m = _ID_RE.match(path.name)
    if not m:
        return None
    status, created = "", None
    end = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(head_bytes)
            if head.startswith("---"):
                while (end := _FM_END_RE.search(head, 3)) is None and len(head) < _FM_LIMIT_BYTES:
                    chunk = fh.read(head_bytes)
                    if not chunk:
                        break
                    head += chunk
                # The body has to hold the H1 and `body_chars` of prose too.
                while end is not None and len(head) < end.end() + body_chars + head_bytes:
                    chunk = fh.read(head_bytes)
                    if not chunk:
                        break
                    head += chunk
    except OSError:
        return None
    body = head
    if end is not None:
        fm, body = head[:end.start()], head[end.end():]
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
    `source` is `both`, `lex` or `vec`; a closed row adds `verdict` and
    `closure_reason` (`closure_of`)."""
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
    out = out[:n]
    # A closed neighbour with nothing beside its status reads as "someone
    # already did this" whatever actually closed it — #420 was closed `done`
    # and its defect re-found by two nightlies (#1051). Read only for the
    # rows returned, so the full front-matter parse is ≤ `limit` files.
    for r in out:
        if r.get("status") not in OPEN_STATUSES:
            r.update(closure_of(backlog_dir, r["id"]))
    return out


_VERDICT_LINE_RE = re.compile(r"autotriage: \*\*([a-z_]+)\*\*\.?\s*(.*)", re.S)
_CLOSURE_REASON_CHARS = 240
NO_VERDICT = "unrecorded"


def closure_of(backlog_dir: Path, item_id: int) -> dict:
    """`{verdict, closure_reason}` for a closed item, from what its closers
    wrote: `duplicate_of` (group triage and the sweep), `autotriage_retired`
    and the `autotriage: **<verdict>**. <evidence>` activity line
    (`scripts/automod/backlog.py::record_verdict`). Most closed items carry
    none of them — a landing, expiry or a human close — and say so as
    `NO_VERDICT` rather than have a verdict inferred for them."""
    fm: dict = {}
    try:
        for path in backlog_dir.glob(f"{int(item_id)}[-_]*.md"):
            text = path.read_text(encoding="utf-8", errors="replace")[:_FM_LIMIT_BYTES]
            end = _FM_END_RE.search(text, 3) if text.startswith("---") else None
            if end is not None:
                loaded = yaml.safe_load(text[3:end.start()])
                fm = loaded if isinstance(loaded, dict) else {}
            break
    except Exception:  # noqa: BLE001 — fail open: the row stays, unexplained
        fm = {}
    line_verdict, line_reason = "", ""
    for entry in reversed(list(fm.get("activity_log") or [])):
        m = _VERDICT_LINE_RE.search(str(entry))
        if m:
            line_verdict = m.group(1)
            line_reason = m.group(2).split(" Check: ")[0].strip()
            break
    dup = fm.get("duplicate_of")
    if dup:
        verdict, reason = "duplicate_of", f"duplicate of #{dup}"
        if line_reason:
            reason += f": {line_reason}"
    elif fm.get("autotriage_retired") or line_verdict:
        verdict = str(fm.get("autotriage_retired") or line_verdict)
        reason = line_reason or f"retired by autotriage as {verdict}"
    else:
        verdict, reason = NO_VERDICT, "closed, no recorded verdict"
    return {"verdict": verdict, "closure_reason": " ".join(reason.split())[:_CLOSURE_REASON_CHARS]}


def merge_target(similar: list[dict], *, cfg: dict | None = None,
                 now: float | None = None) -> dict | None:
    """The one open item a spawn-tagged write should be appended to, or None.

    Rule A: reranker score over the threshold AND the lexical leg agrees —
    the jaccard, not the shared-token count. `shared` is reported on the row
    and is not a merge leg: the fleet's tags and handoff boilerplate put 4-12
    tokens in common between items about different files (#934).
    Rule B: a strong lexical match on an item created inside the recent
    window — the same session filing the same finding twice, before qmd has
    seen the first copy. The reranker score alone never merges.
    """
    return _best_match(similar, cfg=cfg, now=now, open_rows=True)


def closed_match(similar: list[dict], *, cfg: dict | None = None,
                 now: float | None = None) -> dict | None:
    """The closed item a spawn-tagged write would have merged into had it been
    open — the same two rules. The caller refuses that create rather than
    append to a finished item or file a silent twin of it (#1051)."""
    return _best_match(similar, cfg=cfg, now=now, open_rows=False)


def _best_match(similar: list[dict], *, cfg: dict | None, now: float | None,
                open_rows: bool) -> dict | None:
    cfg = cfg or dedupe_config()
    now = time.time() if now is None else now
    best: dict | None = None
    for r in similar:
        if (r.get("status") in OPEN_STATUSES) != open_rows:
            continue
        lex_ok = float(r.get("lexical") or 0) >= float(cfg["lexical_min"])
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
