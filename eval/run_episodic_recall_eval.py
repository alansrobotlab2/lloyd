#!/usr/bin/env python3
"""#1485 — do chat transcripts and autonomy runs earn a place in the recall?

Two proposals, each measured against today's behaviour, arms interleaved per
query so daemon and djev drift land on both sides:

1. ``vault_recall`` with episodic floors (`RECALL_EPISODIC_FLOORS`: the module's
   `RECALL_EPISODIC_FLOOR` collections join the doc leg, the djev head shrinks to
   `RECALL_EPISODIC_DJEV_HEAD` so the pool still fits one canvas; `--floor` and
   `--head` override both for a variant arm) against the
   production recall. An A/A arm (production twice) prices djev's own noise.
2. ``session_recall`` as a qmd query (`SESSION_RECALL_BACKEND = "qmd"`) against
   the token scorer.

Two query sets:

- **gold** — `vault_recall_queries.yaml` rows with `expect_docs`; the doc set
  must not regress (paired).
- **episodic** — a known-item set built from what is on disk, because P0's
  conversational questions (#1480) do not exist yet. Every `### Session HH:MM —
  Auto-captured` summary in a daily note (`~/obsidian/memory/YYYY-MM-DD.md`) was
  written by the secondary model from ONE chat moments after its first turn; the
  chat is the transcript in qmd's `sessions` collection with the latest id at or
  before the capture minute on that date, within `--window-min`. The query is the
  summary's first sentence — a paraphrase, never the transcript's text, so this is
  not the #1511 echo. A proxy for "what did we discuss about X", and a biased
  one: the daily note carrying the same sentence is in the index too, so the
  measure is whether the transcript ALSO reaches the top K.

Everything reads; nothing is written but the artifact. Queries go to the live qmd
daemon and djev (both allowed to be queried; neither is restarted or loaded).

Usage:
    .venvs/lloyd/bin/python eval/run_episodic_recall_eval.py \\
        --sessions-json-dir ~/lloyd-data/sessions --out ~/lloyd-data/eval/1485/run.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from run_prefetch_eval import _quiet_logging  # noqa: E402

with _quiet_logging():
    from agent_mcp import session as session_mod  # noqa: E402
    from agent_mcp import transcript_self_hit as tsh  # noqa: E402
    from agent_mcp import vault as vault_mod  # noqa: E402
    from stats import paired_bootstrap_ci  # noqa: E402

_HEAD = re.compile(r"^### Session (\d{2}):(\d{2}) \w+ — Auto-captured\s*$", re.M)
_ID = re.compile(r"^(\d{8})_(\d{6})_")


_STOP = {"assistant", "user", "requested", "provided", "asked", "about", "which",
          "their", "there", "these", "those", "that", "this", "with", "from", "into",
          "were", "been", "have", "having", "would", "could", "should", "while",
          "after", "before", "also", "then", "than", "them", "they", "what", "when"}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]{4,}", text.lower()) if w not in _STOP}


def _overlap(words: set[str], note: Path) -> float:
    try:
        body = note.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return 0.0
    have = set(re.findall(r"[a-z0-9_]{4,}", body))
    return len(words & have) / len(words) if words else 0.0


def episodic_items(memory_dir: Path, sessions_root: Path, window_min: int) -> list[dict]:
    items = []
    for note in sorted(memory_dir.glob("20[0-9][0-9]-[0-9][0-9]-[0-9][0-9].md")):
        date = note.stem
        text = note.read_text(encoding="utf-8", errors="replace")
        day_dir = sessions_root / date
        if not day_dir.is_dir():
            continue
        ids = []
        for f in day_dir.glob("*.md"):
            m = _ID.match(f.stem)
            # Chats only (three-part ids): before 2026-09-10 worker sessions were
            # exported here too, dozens an hour, and a capture minute cannot tell
            # which of them a summary came from.
            if m and len(f.stem.split("_")) == 3:
                ids.append((m.group(2), f))
        for m in _HEAD.finditer(text):
            cap = int(m.group(1)) * 60 + int(m.group(2))
            body = text[m.end():].lstrip("\n")
            para = body.split("\n\n", 1)[0].strip()
            if not para or para.startswith(("#", "---")):
                continue
            first = re.split(r"(?<=[.!?])\s", para, maxsplit=1)[0][:300]
            cands = [(int(h[:2]) * 60 + int(h[2:4]), f) for h, f in ids]
            cands = [c for c in cands if cap - window_min <= c[0] <= cap + 1]
            # The capture minute alone mislabelled most summaries (the chat that
            # opened last is often not the one captured), so the label is the
            # candidate whose transcript holds most of the summary's content
            # words — kept only when it holds at least half of them and clearly
            # more than the runner-up. That favours lexically findable items, a
            # bias stated in the write-up; a wrong label would be worse.
            words = _content_words(para)
            scored = sorted(((_overlap(words, f), t, f) for t, f in cands), reverse=True)
            if not scored or not words or scored[0][0] < 0.5:
                continue
            if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.15:
                continue
            best = (scored[0][1], scored[0][2])
            items.append({"id": f"{date}@{m.group(1)}{m.group(2)}", "query": first,
                          "expect": f"sessions/{date}/{best[1].name}",
                          "session_id": best[1].stem, "date": date})
    # One item per transcript: several summaries can resolve to one busy chat.
    seen, out = set(), []
    for it in items:
        if it["expect"] not in seen:
            seen.add(it["expect"])
            out.append(it)
    return out


_ASK = (
    "Below is a transcript of a past conversation between Alan and his assistant "
    "Lloyd. Weeks later Alan wants to find THIS conversation again. Write the one "
    "question he would ask Lloyd to recall it, in his own casual words. Do NOT copy "
    "any phrase of four or more words from the transcript, name no file path, and "
    "do not start with 'Remember'. Reply with the question only.\n\n<transcript>\n{t}\n</transcript>"
)


def synth_items(sessions_root: Path, n: int, base_url: str, model: str,
                seed: int = 25, since: str = "", min_turns: int = 2) -> list[dict]:
    """One paraphrased recall question per sampled chat transcript (primary model).

    Chats only (three-part ids), at least two user turns, the E2E probe excluded.
    A question that repeats the transcript near-verbatim (#1511's echo test) is
    discarded rather than kept, so the set measures recall, not string match.
    """
    import random
    import httpx

    notes = []
    for f in sorted(sessions_root.glob("*/*.md")):
        if len(f.stem.split("_")) != 3 or not _ID.match(f.stem):
            continue
        if since and f.parent.name < since:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        turns = tsh.user_turns(text)
        if len(turns) < min_turns or "E2E harness check" in text:
            continue
        notes.append((f, text))
    random.Random(seed).shuffle(notes)
    items = []
    with httpx.Client(timeout=120) as client:
        for f, text in notes:
            if len(items) >= n:
                break
            body = {"model": model, "temperature": 0.7, "max_tokens": 120, "priority": 3,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": _ASK.format(t=text[:6000])}]}
            try:
                r = client.post(f"{base_url}/v1/chat/completions", json=body)
                q = r.json()["choices"][0]["message"]["content"].strip().strip('"')
            except Exception:  # noqa: BLE001
                continue
            if not q or any(tsh.echoes(q, u) for u in tsh.user_turns(text)):
                continue
            rel = f.relative_to(sessions_root).as_posix()
            items.append({"id": f.stem, "query": q[:300], "expect": f"sessions/{rel}",
                          "session_id": f.stem, "date": f.parent.name})
    return items


def _rank(expect: list[str], paths: list[str]) -> int | None:
    exp = [e.lower().replace("-", "_") for e in expect]
    for i, p in enumerate(paths, 1):
        pl = p.lower().replace("-", "_")
        if any(e in pl for e in exp):
            return i
    return None


def recall(query: str, episodic: bool, k: int) -> tuple[list[str], float, str | None]:
    vault_mod.RECALL_EPISODIC_FLOORS = episodic
    t0 = time.perf_counter()
    try:
        res = vault_mod._vault_recall({"query": query, "limit": k, "include_facts": False,
                                       "grep_code": True})
        err = res.get("error") if isinstance(res, dict) else None
        paths = [d.get("path", "") for d in (res.get("documents") or [])]
    except Exception as e:  # noqa: BLE001
        paths, err = [], f"{type(e).__name__}: {e}"
    finally:
        vault_mod.RECALL_EPISODIC_FLOORS = None
    return paths, (time.perf_counter() - t0) * 1000, err


def session_recall(query: str, backend: str, days: int, limit: int) -> tuple[list[str], float]:
    session_mod.SESSION_RECALL_BACKEND = backend
    t0 = time.perf_counter()
    try:
        res = session_mod._session_recall({"query": query, "days": days, "limit": limit})
    finally:
        session_mod.SESSION_RECALL_BACKEND = None
    return ([s.get("session_id", "") for s in res.get("sessions") or []],
            (time.perf_counter() - t0) * 1000)


def _pair(a: list[float], b: list[float]) -> dict:
    ci = paired_bootstrap_ci(a, b)
    return {"a": round(sum(a) / len(a), 4), "b": round(sum(b) / len(b), 4),
            "diff": round(ci["diff"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
            "n": ci["n"], "significant": ci["significant"]}


def _metrics(rows: list[dict], arm: str) -> tuple[list[float], list[float]]:
    hit = [1.0 if r[arm]["rank"] else 0.0 for r in rows]
    rr = [1.0 / r[arm]["rank"] if r[arm]["rank"] else 0.0 for r in rows]
    return hit, rr


def _p50(xs: list[float]) -> float:
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 1) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-dir", default=str(Path.home() / "obsidian" / "memory"))
    ap.add_argument("--sessions-json-dir", required=True,
                    help="production's sessions/*.json, read by the token scorer")
    ap.add_argument("--window-min", type=int, default=90)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--max-episodic", type=int, default=200)
    ap.add_argument("--synth-file", default="",
                    help="episodic set = paraphrased questions from the primary, cached "
                         "here (built on first use); omit for the daily-note set")
    ap.add_argument("--base-url", default="http://127.0.0.1:8096")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-nvfp4")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--min-turns", type=int, default=2)
    ap.add_argument("--floor", default="",
                    help="arm B's episodic floors, e.g. 'sessions=1' (default: the module's)")
    ap.add_argument("--head", type=int, default=0,
                    help="arm B's djev head (default: the module's RECALL_EPISODIC_DJEV_HEAD)")
    ap.add_argument("--skip-session-recall", action="store_true")
    ap.add_argument("--since", default="",
                    help="synth only from transcripts dated on/after YYYY-MM-DD (e.g. the "
                         "days whose session JSON still exists, so both backends can see them)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.floor:
        vault_mod.RECALL_EPISODIC_FLOOR = {k: int(v) for k, v in
                                           (kv.split("=") for kv in args.floor.split(","))}
    if args.head:
        vault_mod.RECALL_EPISODIC_DJEV_HEAD = args.head
    root = tsh._sessions_root()
    if root is None:
        print("no qmd sessions collection configured"); return 2
    # The token scorer reads the checkout's SESSIONS_DIR, which from a worktree is
    # an empty data root; point it (read-only) at production's JSON.
    session_mod.SESSIONS_DIR = Path(args.sessions_json_dir).expanduser()

    gold = [q for q in (yaml.safe_load((HERE / "vault_recall_queries.yaml").read_text())
                        .get("queries") or []) if q.get("expect_docs")]
    if args.synth_file and Path(args.synth_file).expanduser().exists():
        epi = json.loads(Path(args.synth_file).expanduser().read_text())
    elif args.synth_file:
        epi = synth_items(root, args.max_episodic, args.base_url, args.model,
                          since=args.since, min_turns=args.min_turns)
        Path(args.synth_file).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.synth_file).expanduser().write_text(json.dumps(epi, indent=1))
    else:
        epi = episodic_items(Path(args.memory_dir).expanduser(), root, args.window_min)
    epi = epi[-args.max_episodic:]
    print(f"[info] gold={len(gold)} episodic={len(epi)}")
    if args.build_only:
        return 0

    out: dict = {"ran_at": datetime.now(timezone.utc).isoformat(), "k": args.k,
                 "days": args.days, "window_min": args.window_min,
                 "arm_b": {"floor": vault_mod.RECALL_EPISODIC_FLOOR,
                           "head": vault_mod.RECALL_EPISODIC_DJEV_HEAD}}
    with _quiet_logging():
        for name, qs in (("gold", [{"id": q.get("id"), "query": q["query"],
                                    "expect": q["expect_docs"]} for q in gold]),
                         ("episodic", [{**it, "expect": [it["expect"]]} for it in epi])):
            rows = []
            for i, q in enumerate(qs):
                row = {"id": q["id"], "query": q["query"][:200], "expect": q["expect"]}
                # Arm order rotates per query: djev and qmd drift sit on every arm.
                order = [("A", False), ("A2", False), ("B", True)]
                order = order[i % 3:] + order[:i % 3]
                for arm, epis in order:
                    paths, ms, err = recall(q["query"], epis, args.k)
                    row[arm] = {"rank": _rank(q["expect"], paths), "ms": round(ms, 1),
                                "err": err, "n_sessions": sum(p.startswith("sessions/") for p in paths),
                                "n_runs": sum(p.startswith("autonomy-runs/") for p in paths)}
                rows.append(row)
            summ = {}
            for a, b in (("A", "A2"), ("A", "B")):
                ha, ra = _metrics(rows, a)
                hb, rb = _metrics(rows, b)
                summ[f"{a}_vs_{b}"] = {"doc_hit": _pair(ha, hb), "mrr": _pair(ra, rb)}
            summ["p50_ms"] = {arm: _p50([r[arm]["ms"] for r in rows]) for arm in ("A", "A2", "B")}
            summ["errors"] = {arm: sum(1 for r in rows if r[arm]["err"]) for arm in ("A", "A2", "B")}
            summ["B_rows_with_transcripts"] = sum(1 for r in rows if r["B"]["n_sessions"])
            summ["B_rows_with_runs"] = sum(1 for r in rows if r["B"]["n_runs"])
            out[name] = {"summary": summ, "records": rows}
            print(name, json.dumps(summ, indent=1))

        # session_recall: the episodic items inside the window of both backends.
        cutoff = (datetime.now().date().toordinal() - args.days)
        recent = [] if args.skip_session_recall else [
            it for it in epi
            if datetime.strptime(it["date"], "%Y-%m-%d").date().toordinal() >= cutoff]
        rows = []
        for i, it in enumerate(recent):
            row = {"id": it["id"], "query": it["query"][:200], "expect": it["session_id"]}
            for backend in (("tokens", "qmd") if i % 2 == 0 else ("qmd", "tokens")):
                ids, ms = session_recall(it["query"], backend, args.days, 5)
                rank = next((j for j, s in enumerate(ids, 1) if s == it["session_id"]), None)
                row[backend] = {"rank": rank, "ms": round(ms, 1)}
            rows.append(row)
        if rows:
            ht, rt = _metrics(rows, "tokens")
            hq, rq = _metrics(rows, "qmd")
            summ = {"n": len(rows), "hit@5": _pair(ht, hq), "mrr": _pair(rt, rq),
                    "p50_ms": {b: _p50([r[b]["ms"] for r in rows]) for b in ("tokens", "qmd")}}
        else:
            summ = {"n": 0}
        out["session_recall"] = {"summary": summ, "records": rows}
        print("session_recall", json.dumps(summ, indent=1))

    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).expanduser().write_text(json.dumps(out, indent=2, default=str))
    print(f"[info] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
