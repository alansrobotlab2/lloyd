#!/usr/bin/env python3
"""#1494, measured offline without touching the qmd fork: LLM-written situating context.

Anthropic's contextual retrieval prefixes a 1-2 sentence situating context onto
every chunk, in both the embedding text and the FTS text. The fork cannot be
edited from here (human-landed tree), but it already prefixes every chunk's
embedding text with the document's title, and qmd takes that title from the
document's first `#`/`##` heading (`extractTitle`). So on a SCRATCH copy of the
#1493 sub-corpus (`eval/embed_side_index.py prepare`) this:

  generate   asks the primary, once per document, for a <=40-word context that
             situates it in the vault (whole-document context, not per chunk —
             the documented difference from the technique);
  prepare    copies the sub-corpus index to `subctx`, rewrites each document's
             first heading as `<heading> — <context>` (or prepends one), and
             re-fires the FTS trigger so the lex leg indexes it too.

Then `embed_side_index.py embed --name subctx` re-embeds it with production's
0.6B and `embed_side_index.py eval --candidate subctx --item 1494` scores it
against `sub06` — same documents, same model, only the context differs.
`documents.title` is left alone, so djev's row keeps today's title.

The primary is shared: take primary.lock shared for `generate`.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

QMD_CACHE = Path.home() / ".cache/qmd"
QMD_CONFIG = Path.home() / ".config/qmd"
STATE = Path.home() / "lloyd-data/eval/1494"
DOC_CHARS = 6000
PROMPT = (
    "Here is a note from Alan's personal Obsidian vault, which also holds the "
    "design notes, backlog and memory of Lloyd, his local AI agent.\n\n"
    "<document path=\"{path}\">\n{doc}\n</document>\n\n"
    "Write a short succinct context (one or two sentences, at most 40 words) that "
    "situates this note within the vault — what it is, and which project, system, "
    "person or decision it concerns — for the purpose of improving search "
    "retrieval of its passages. Answer only with the succinct context."
)
_HEADING = re.compile(r"^(##?)\s+(.+)$", re.MULTILINE)


def _post(url: str, model: str, prompt: str, timeout: float = 60.0) -> str:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 90,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return " ".join((data["choices"][0]["message"]["content"] or "").split())


def generate(args) -> int:
    from app.secondary_models import _endpoint
    url, model = _endpoint("focus")
    src = QMD_CACHE / "sub06.sqlite"
    db = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = db.execute("""select d.hash, min(d.collection || '/' || d.path), c.doc
                         from documents d join content c on c.hash = d.hash
                         where d.active = 1 group by d.hash""").fetchall()
    db.close()
    out_path = STATE / "contexts.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [(h, p, doc) for h, p, doc in rows if h not in done]
    t0 = time.time()
    lat = []

    def one(row):
        h, p, doc = row
        s = time.perf_counter()
        try:
            ctx = _post(url, model, PROMPT.format(path=p, doc=(doc or "")[:DOC_CHARS]))
        except Exception as e:  # noqa: BLE001
            return h, None, str(e)[:120], 0.0
        return h, ctx, None, time.perf_counter() - s

    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (h, ctx, err, dt) in enumerate(ex.map(one, todo), start=1):
            if ctx:
                done[h] = ctx
                lat.append(dt)
            else:
                errors += 1
            if i % 100 == 0:
                out_path.write_text(json.dumps(done))
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    out_path.write_text(json.dumps(done))
    info = {"documents": len(rows), "contexts": len(done), "errors": errors,
            "wall_s": round(time.time() - t0, 1), "workers": args.workers,
            "mean_call_s": round(sum(lat) / len(lat), 2) if lat else None,
            "model": model,
            "mean_words": round(sum(len(v.split()) for v in done.values()) / max(1, len(done)), 1)}
    (STATE / "generate.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


def with_context(doc: str, path: str, ctx: str) -> str:
    m = _HEADING.search(doc or "")
    if m:
        return doc[:m.start()] + f"{m.group(1)} {m.group(2).strip()} — {ctx}" + doc[m.end():]
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return f"# {stem} — {ctx}\n{doc}"


def prepare(args) -> int:
    from eval.embed_side_index import _config_text, EMBED_06B
    ctxs = json.loads((STATE / "contexts.json").read_text())
    dst = QMD_CACHE / "subctx.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(str(dst) + suffix).unlink(missing_ok=True)
    # sub4b is the pruned snapshot before any re-embed of its own vectors matters:
    # the content and documents tables are identical in all three copies.
    shutil.copyfile(QMD_CACHE / "sub06.sqlite", dst)
    db = sqlite3.connect(dst)
    rows = db.execute("""select c.hash, min(d.collection || '/' || d.path), c.doc
                         from content c join documents d on d.hash = c.hash
                         where d.active = 1 group by c.hash""").fetchall()
    n = 0
    for h, p, doc in rows:
        ctx = ctxs.get(h)
        if ctx:
            db.execute("update content set doc = ? where hash = ?", (with_context(doc, p, ctx), h))
            n += 1
    db.execute("update documents set title = title where active = 1")   # re-fires the FTS trigger
    db.commit()
    db.close()
    (QMD_CONFIG / "subctx.yml").write_text(_config_text(EMBED_06B))
    print(json.dumps({"documents": len(rows), "with_context": n}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("generate", "prepare"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    STATE.mkdir(parents=True, exist_ok=True)
    return {"generate": generate, "prepare": prepare}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
