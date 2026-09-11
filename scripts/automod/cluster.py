"""Group the open backlog by what it is about, so triage can judge a cluster.

The board grows by splitting: triage files one item per surviving claim,
implement rounds file what they notice, and a re-offered round re-derives
and re-files. By 2026-09-11, 84 parent items had produced 282 children, and
#788, #795 and #799 were one finding filed by three re-runs of #549. Nothing
in the system could put them back together — the only similarity check was
a 30-character title-prefix histogram in a weekly hygiene skill.

This pass is deterministic and cheap. Three signals, all already on disk:

* **cosine** over the chunk-0 vectors qmd keeps for the `backlog` collection
  it already embeds (`~/.cache/qmd/index.sqlite`, read through qmd's own
  bundled `vec0.so` — no daemon, no re-embedding, 0.15 s for the board).
  Chunk 0 only: mean-pooling long items pulls them toward the corpus
  centroid and produced obvious false pairs.
* **shared file paths** — 75% of open items name repo files in backticks,
  normalised to basenames so `vault.py` and `agent_mcp/vault.py` agree.
* **a common parent**, parsed from the prose first line ("Split from #N",
  "Found while implementing #N") and persisted to frontmatter as `parent`.

An optional LLM pair-judge (the secondary, priority 2, cached by body hash)
adjudicates only the ambiguous edges. `distinct` drops an edge; an error
keeps it — the judge is a filter, not a requirement.

Quarantine is deliberately NOT applied here. `is_quarantined` answers "is
this old claim still true?", and a self-filed item cannot be stale by
construction; this pass asks "is this the same work as that?", which a
one-day-old item can answer. The output is `clusters.json` in the automod
state dir, which the group-triage mode of `autotriage` consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.automod import backlog as B
from scripts.automod import state as S

LLOYD_HOME = Path(__file__).resolve().parents[2]
CLUSTERS_PATH = S.STATE_DIR / "clusters.json"
JUDGMENTS_PATH = S.STATE_DIR / "cluster_judgments.jsonl"
QMD_DB = Path.home() / ".cache" / "qmd" / "index.sqlite"
# qmd's bundled sqlite-vec extension. `node_modules` is not checked in, so a
# sandbox clone or an automod worktree has none of its own; the live tree's
# copy is the fallback, because the index it reads is the live one anyway.
_VEC0_REL = Path("qmd") / "node_modules" / "sqlite-vec-linux-x64" / "vec0.so"
VEC0_SO = next((p for p in (LLOYD_HOME / _VEC0_REL, Path.home() / "lloyd" / _VEC0_REL)
                if p.exists()), LLOYD_HOME / _VEC0_REL)

# Measured on the 2026-09-11 board (429 drafts): >=0.80 gave 5 clusters over
# 10 items, >=0.75 16 over 42, >=0.70 33 over 114 with one of 20. 0.75 is
# the precision knob; the path and parent rules add recall below it.
DEFAULT_THRESHOLD = 0.75
# Two, not three: a pair of duplicates is the most valuable cluster there is
# — one group-triage turn closes one of them with certainty — and #795/#799
# were exactly that pair, at cosine 0.79 with no third sibling near them.
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 12
SCHEMA = 1

_PATH_RE = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:py|tsx?|jsx?|md|ya?ml|sh|json|toml))(?::\d+(?:-\d+)?)?`")
# Every item names these; sharing one says nothing.
_PATH_STOPLIST = frozenset({"config.yaml", "__init__.py", "README.md", "SKILL.md",
                            "CLAUDE.md", "pytest.ini", "requirements.lock"})
# Three spellings on disk: "Split from #N", "Split out of #N" (#400/#401),
# "Found while implementing #N".
_PARENT_RE = re.compile(r"(?:split (?:from|out of)|found while implementing)\s+#(\d+)", re.I)


# ── signals ────────────────────────────────────────────────────────────────

def named_paths(body: str) -> set[str]:
    """Basenames of the repo files an item names in backticks."""
    out: set[str] = set()
    for p in _PATH_RE.findall(body or ""):
        base = os.path.basename(p)
        if base and base not in _PATH_STOPLIST:
            out.add(base)
    return out


def parse_parent(body: str) -> int | None:
    """The parent id from the FIRST non-heading, non-blank line only. Items
    cite other items throughout their bodies; the provenance line is at the
    top, where the prompts put it."""
    for line in (body or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _PARENT_RE.search(s)
        return int(m.group(1)) if m else None
    return None


def persist_parents(items: list[B.Item]) -> int:
    """Write `parent:` onto each item whose prose names one and whose
    frontmatter does not. Once: the link becomes machine-readable and the
    prompts' first-line convention stops being the only record of it."""
    n = 0
    for it in items:
        if it.parent is not None:
            continue
        pid = parse_parent(it.body)
        if pid is None or pid == it.id:
            continue
        if B.update_frontmatter(it.path, {"parent": pid}):
            it.parent = pid
            n += 1
    return n


def load_vectors(items: list[B.Item], *, db: Path | None = None, so: Path | None = None) -> dict:
    """`{item_id: unit vector}` from qmd's stored chunk-0 embeddings. Empty
    when numpy, the extension or the index is unavailable — the pass then
    runs on paths and parents alone, and says so."""
    # Resolved at call time, not bound at import: tests point these at a
    # scratch dir, and a default bound at def time would read the real index.
    db = db or QMD_DB
    so = so or VEC0_SO
    try:
        import numpy as np
    except ImportError:
        return {}
    if not db.exists() or not so.exists():
        return {}
    wanted = {it.id for it in items}
    out: dict = {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            con.enable_load_extension(True)
            con.load_extension(str(so))
            con.enable_load_extension(False)
            # The join key is hash || '_' || seq, not ':'; getting it wrong
            # silently returns zero rows.
            rows = con.execute(
                "SELECT d.path, v.embedding FROM documents d "
                "JOIN content_vectors cv ON cv.hash = d.hash "
                "JOIN vectors_vec v ON v.hash_seq = cv.hash||'_'||cv.seq "
                "WHERE d.collection='backlog' AND d.active=1 AND cv.seq=0").fetchall()
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — degrade to the other two signals
        return {}
    for path, blob in rows:
        m = re.match(r"^(\d+)[-_]", os.path.basename(str(path)))
        if not m:
            continue
        iid = int(m.group(1))
        if iid not in wanted:
            continue
        v = np.frombuffer(blob, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            out[iid] = v / norm
    return out


def candidate_pairs(items: list[B.Item], vecs: dict, *, threshold: float = DEFAULT_THRESHOLD
                    ) -> list[dict]:
    """Edges, each with the rules that fired. Rules:
    1. cosine >= threshold
    2. two shared paths, or one shared path and cosine >= threshold - 0.10
    3. a common parent

    Rule 3 needs no cosine on purpose. The twelve children of #549 were
    filed by four re-runs of one round in 110 minutes and are one
    consolidation job whatever their pairwise similarity — the first cut
    demanded cosine >= threshold - 0.15 and dropped that family entirely. A
    sibling that really is distinct work is what the group triage's `keep`
    verdict is for.
    """
    paths = {it.id: named_paths(it.body) for it in items}
    parents = {it.id: (it.parent if it.parent is not None else parse_parent(it.body))
               for it in items}
    ids = sorted(it.id for it in items)
    cos: dict[tuple[int, int], float] = {}
    if vecs:
        try:
            import numpy as np
            have = [i for i in ids if i in vecs]
            if len(have) >= 2:
                X = np.stack([vecs[i] for i in have])
                M = X @ X.T
                for a_i, a in enumerate(have):
                    for b_i in range(a_i + 1, len(have)):
                        cos[(a, have[b_i])] = float(M[a_i, b_i])
        except Exception:  # noqa: BLE001
            cos = {}
    out: list[dict] = []
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            c = cos.get((a, b), 0.0)
            shared = paths[a] & paths[b]
            same_parent = parents[a] is not None and parents[a] == parents[b]
            reasons = []
            if c >= threshold:
                reasons.append("cosine")
            if len(shared) >= 2 or (len(shared) >= 1 and c >= threshold - 0.10):
                reasons.append("paths")
            if same_parent:
                reasons.append("parent")
            if not reasons:
                continue
            out.append({"a": a, "b": b, "cosine": round(c, 3), "shared_paths": sorted(shared),
                        "parent": parents[a] if same_parent else None, "reasons": reasons})
    return out


# ── the judge ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = ("You are a precise triage judge for a software backlog. Given two backlog "
                "items, decide whether they describe the SAME piece of work, RELATED work "
                "that would be implemented together, or DISTINCT work. Respond with JSON only.")

JUDGE_USER = """Two items from the same backlog.

ITEM A (#{a_id}): {a_title}
{a_body}

ITEM B (#{b_id}): {b_title}
{b_body}

Signals: cosine {cosine}; shared files {shared}; common parent {parent}.

- SAME: the same finding or fix, written twice (one would close the other)
- RELATED: distinct findings that one change would address together
- DISTINCT: different work that happens to share words or files

Respond with strict JSON:
{{"verdict": "same" | "related" | "distinct", "confidence": 0.0-1.0, "reason": "one sentence"}}
"""


def _judgment_key(a: B.Item, b: B.Item) -> str:
    lo, hi = sorted((a, b), key=lambda i: i.id)
    ha = hashlib.sha1(lo.body[:3000].encode("utf-8", "replace")).hexdigest()[:12]
    hb = hashlib.sha1(hi.body[:3000].encode("utf-8", "replace")).hexdigest()[:12]
    return hashlib.sha256(f"{lo.id}\x00{hi.id}\x00{ha}\x00{hb}".encode()).hexdigest()[:32]


def load_judgments(path: Path | None = None) -> dict[str, dict]:
    path = path or JUDGMENTS_PATH
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("key"):
            out[rec["key"]] = rec
    return out


def append_judgment(rec: dict, path: Path | None = None) -> None:
    path = path or JUDGMENTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def judge_pair(a: B.Item, b: B.Item, pair: dict, *, endpoint: str, model: str,
               timeout: float = 30.0, urlopen=None) -> dict:
    """One JSON verdict from the local model. `{"error": …}` on any failure;
    the caller keeps the edge then. `urlopen` is injectable for tests."""
    import urllib.request
    urlopen = urlopen or urllib.request.urlopen
    prompt = JUDGE_USER.format(
        a_id=a.id, a_title=a.name, a_body=a.body[:1500],
        b_id=b.id, b_title=b.name, b_body=b.body[:1500],
        cosine=pair.get("cosine"), shared=", ".join(pair.get("shared_paths") or []) or "none",
        parent=f"#{pair['parent']}" if pair.get("parent") else "none")
    payload = {"model": model,
               "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": prompt}],
               "temperature": 0.1, "max_tokens": 300,
               "response_format": {"type": "json_object"},
               "chat_template_kwargs": {"enable_thinking": False},
               # vLLM scheduling priority: chat 0, autonomy 1, batch 2. This
               # must yield to both or a long batch starves the fleet.
               "priority": 2}
    try:
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parsed = json.loads(data["choices"][0]["message"]["content"])
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ("same", "related", "distinct"):
        verdict = "distinct"
    try:
        conf = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"verdict": verdict, "confidence": conf, "reason": str(parsed.get("reason", ""))[:300]}


def _judge_endpoint() -> tuple[str, str]:
    from app.secondary_models import _endpoint
    return _endpoint()


# ── components ─────────────────────────────────────────────────────────────

def components(edges: list[tuple[int, int]]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, set[int]] = {}
    for x in list(parent):
        groups.setdefault(find(x), set()).add(x)
    return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def _weight(p: dict) -> float:
    return float(p.get("cosine") or 0.0) + 0.5 * len(p.get("reasons") or [])


def peel(pairs: list[dict], *, max_size: int = MAX_CLUSTER_SIZE) -> list[set[int]]:
    """Connected components, with any component over `max_size` peeled into
    hub-centred groups rather than trimmed.

    On the live board every family the three signals touch joins one giant
    component (cosine edges bridge #549's children to the OKF items to the
    trajectory miners), and the first cut kept the twelve highest-degree
    nodes and listed the rest as overflow — which dropped most of the board
    from the night's output. Peeling takes the highest-degree node still
    unassigned, its strongest neighbours up to `max_size`, removes them, and
    repeats, so every item that has an edge ends up in some group.
    """
    adj: dict[int, dict[int, float]] = {}
    for p in pairs:
        a, b, w = p["a"], p["b"], _weight(p)
        adj.setdefault(a, {})[b] = max(w, adj.get(a, {}).get(b, 0.0))
        adj.setdefault(b, {})[a] = max(w, adj.get(b, {}).get(a, 0.0))
    out: list[set[int]] = []
    for comp in components([(p["a"], p["b"]) for p in pairs]):
        if len(comp) <= max_size:
            out.append(comp)
            continue
        left = set(comp)
        while left:
            seed = max(left, key=lambda i: (sum(1 for j in adj[i] if j in left), -i))
            group = {seed}
            # Grow by the strongest edge into the group, so a hub's tight
            # neighbourhood comes out together rather than its twelve
            # loosest acquaintances.
            while len(group) < max_size:
                best, best_w = None, -1.0
                for i in group:
                    for j, w in adj[i].items():
                        if j in left and j not in group and w > best_w:
                            best, best_w = j, w
                if best is None:
                    break
                group.add(best)
            out.append(group)
            left -= group
    return sorted(out, key=lambda g: (-len(g), min(g)))


def _describe(group: set[int], pairs: list[dict], same_pairs: list[list[int]]) -> dict:
    inner = [p for p in pairs if p["a"] in group and p["b"] in group]
    anchor_paths: dict[str, int] = {}
    for p in inner:
        for sp in p["shared_paths"]:
            anchor_paths[sp] = anchor_paths.get(sp, 0) + 1
    parents = [p["parent"] for p in inner if p.get("parent")]
    parent = max(set(parents), key=parents.count) if parents else None
    reason_counts: dict[str, int] = {}
    for p in inner:
        for r in p["reasons"]:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    return {
        "id": cluster_id(group), "item_ids": sorted(group),
        "reason": "; ".join(f"{r} on {n} pair(s)" for r, n in sorted(reason_counts.items())),
        "anchor_paths": [k for k, _ in sorted(anchor_paths.items(), key=lambda kv: -kv[1])][:6],
        "parent": parent,
        "duplicates": [d for d in same_pairs if d[0] in group and d[1] in group],
    }


def cluster_id(item_ids) -> str:
    return "c-" + hashlib.sha1(",".join(str(i) for i in sorted(int(x) for x in item_ids))
                               .encode()).hexdigest()[:8]


def clusterable_items(ledger: Path, boards: tuple[str, ...] | None = B.DEFAULT_BOARDS
                      ) -> list[B.Item]:
    """Open drafts nobody has judged: not triaged, not folded into an
    umbrella, not an umbrella, not parked for a human, not expired.
    Quarantine does not apply — see the module docstring."""
    seen = B.triaged_ids(ledger)
    skip_tags = {"umbrella", B.NEEDS_HUMAN_TAG, B.EXPIRED_TAG}
    return [i for i in B.open_items(boards)
            if i.status == B.TRIAGE_POOL_STATUS and i.id not in seen
            and i.group is None and not (set(i.tags) & skip_tags)]


# ── the pass ───────────────────────────────────────────────────────────────

def build_clusters(*, ledger: Path | None = None, threshold: float = DEFAULT_THRESHOLD,
                   judge: bool = True, max_pairs_judged: int = 200,
                   persist_parents_: bool = True, boards=B.DEFAULT_BOARDS,
                   items: list[B.Item] | None = None, vecs: dict | None = None,
                   judge_fn=None, judgments_path: Path | None = None) -> dict:
    ledger = ledger or S.LEDGER_PATH
    judgments_path = judgments_path or JUDGMENTS_PATH
    items = clusterable_items(ledger, boards) if items is None else items
    parents_written = persist_parents(items) if persist_parents_ else 0
    vecs = load_vectors(items) if vecs is None else vecs
    pairs = candidate_pairs(items, vecs, threshold=threshold)
    by_id = {it.id: it for it in items}

    judged = cached = errors = 0
    kept_pairs: list[dict] = []
    same_pairs: list[list[int]] = []
    if judge and pairs:
        cache = load_judgments(judgments_path)
        endpoint = model = ""
        # Only the ambiguous edges: cosine-only ones just over the line, and
        # parent-only ones. A path-supported edge or a strong cosine stands.
        def ambiguous(p: dict) -> bool:
            r = set(p["reasons"])
            return (r == {"cosine"} and p["cosine"] < threshold + 0.05) or r == {"parent"}
        order = sorted(pairs, key=lambda p: (not ambiguous(p), -p["cosine"]))
        for p in order:
            if not ambiguous(p):
                kept_pairs.append(p)
                continue
            if judged >= max_pairs_judged:
                kept_pairs.append(p)  # unjudged: keep, the rules fired
                continue
            a, b = by_id[p["a"]], by_id[p["b"]]
            key = _judgment_key(a, b)
            rec = cache.get(key)
            if rec is None:
                if judge_fn is None and not endpoint:
                    try:
                        endpoint, model = _judge_endpoint()
                    except Exception:  # noqa: BLE001 — no endpoint: keep the edge
                        errors += 1
                        kept_pairs.append(p)
                        continue
                fn = judge_fn or (lambda x, y, pr: judge_pair(x, y, pr, endpoint=endpoint, model=model))
                verdict = fn(a, b, p)
                judged += 1
                if verdict.get("error"):
                    errors += 1
                    kept_pairs.append(p)
                    continue
                rec = {"key": key, "a": a.id, "b": b.id, "ts": time.time(), **verdict}
                cache[key] = rec
                append_judgment(rec, judgments_path)
            else:
                cached += 1
            if rec.get("verdict") == "distinct":
                continue
            kept_pairs.append({**p, "judge": rec.get("verdict")})
            if rec.get("verdict") == "same":
                same_pairs.append([a.id, b.id])
    else:
        kept_pairs = pairs

    clusters = [_describe(group, kept_pairs, same_pairs)
                for group in peel(kept_pairs, max_size=MAX_CLUSTER_SIZE)
                if len(group) >= MIN_CLUSTER_SIZE]
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "schema": SCHEMA, "threshold": threshold,
            "items_considered": len(items), "vectors_found": len(vecs),
            "pairs_candidate": len(pairs), "pairs_kept": len(kept_pairs),
            "pairs_judged": judged, "judge_cached": cached, "judge_errors": errors,
            "parents_persisted": parents_written,
            "clusters": clusters}


def write_clusters(data: dict, path: Path | None = None) -> Path:
    path = path or CLUSTERS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_clusters(path: Path | None = None) -> dict:
    path = path or CLUSTERS_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cluster the open backlog by what it is about")
    ap.add_argument("--write", action="store_true", help="write clusters.json (default: print only)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM pair-judge")
    ap.add_argument("--max-judged", type=int, default=200)
    ap.add_argument("--no-persist-parents", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    data = build_clusters(threshold=args.threshold, judge=not args.no_judge,
                          max_pairs_judged=args.max_judged,
                          persist_parents_=not args.no_persist_parents)
    if args.write:
        data["path"] = str(write_clusters(data))
    if args.json:
        print(json.dumps(data, indent=1, sort_keys=True))
    else:
        print(f"{len(data['clusters'])} cluster(s) over "
              f"{sum(len(c['item_ids']) for c in data['clusters'])} items "
              f"({data['items_considered']} considered, {data['vectors_found']} with vectors, "
              f"{data['pairs_candidate']} candidate pairs, {data['pairs_judged']} judged, "
              f"{data['judge_cached']} cached, {data['judge_errors']} judge errors, "
              f"{data['parents_persisted']} parents persisted)")
        for c in data["clusters"]:
            print(f"  {c['id']}  {len(c['item_ids'])} items  {c['item_ids']}"
                  f"{'  parent #' + str(c['parent']) if c['parent'] else ''}"
                  f"{'  paths ' + ','.join(c['anchor_paths']) if c['anchor_paths'] else ''}")
            if c["duplicates"]:
                print(f"      same: {c['duplicates']}")
        if args.write:
            print(f"wrote {data['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
