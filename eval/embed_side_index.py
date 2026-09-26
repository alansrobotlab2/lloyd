#!/usr/bin/env python3
"""#1493: an embedding-model A/B on a side index, never on the live one.

Qwen3-Embedding-4B against production's 0.6B. A full re-embed of the searched
collections at the 4B's speed on the 3090 that also serves production qmd, TTS
and the desktop would hold GPU 0 for hours, so this builds a SUB-CORPUS that
both arms share exactly:

  * every active document in the recall's collections whose path satisfies a
    dev-set `expect_docs` label (so the gold answers are all present), plus
  * a seeded uniform sample (`--sample`, default 0.25) of the rest of those
    collections as distractors;
  * everything else, including the collections the recall never searches, is
    deactivated (`active = 0`) in the copy.

`prepare` VACUUMs the live index INTO two copies (read-only on the source),
prunes both identically and writes a per-index config naming each arm's embed
model (`~/.config/qmd/<name>.yml`; `index.yml` and `evalpin.yml` are never
touched). `embed` re-embeds one copy from scratch (`qmd --index <name> embed -f`)
so the two arms differ in the model and nothing else. `eval` serves both on
their own ports and scores control (0.6B), candidate (4B) and control_b (0.6B
again, the noise floor) per query, interleaved, with the vec leg's own latency
read from each daemon's phases.

Holdout ids are never read here: the sub-corpus is built from the dev labels
only, and the eval scores the dev set.

The caller holds regression.lock for `embed` and `eval` (GPU 0, a daemon beside
production's).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

QMD_CACHE = Path.home() / ".cache/qmd"
QMD_CONFIG = Path.home() / ".config/qmd"
LIVE_INDEX = QMD_CACHE / "index.sqlite"
EMBED_06B = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"
EMBED_4B = "hf:Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf"
# `subctx` is #1494's arm: the 0.6B over the same sub-corpus with an LLM-written
# situating context in every document's first heading (eval/contextual_titles.py).
ARMS = {"sub06": EMBED_06B, "sub4b": EMBED_4B}
PORTS = {"sub06": 8184, "sub4b": 8185, "subctx": 8186}
EMBEDS = {**ARMS, "subctx": EMBED_06B}
NARROW_LABEL_MAX = 50


def _config_text(embed: str) -> str:
    base = (QMD_CONFIG / "evalpin.yml").read_text()
    out = []
    for line in base.splitlines():
        if line.strip().startswith("embed:"):
            line = line.split("embed:")[0] + f"embed: {embed}"
        out.append(line)
    return "\n".join(out) + "\n"


def prepare(args) -> dict:
    import yaml
    from agent_mcp.vault import VAULT_SEGMENTS
    from eval.run_eval import _norm, _doc_pair_satisfied

    specs = (yaml.safe_load((HERE / "vault_recall_queries.yaml").read_text()) or {})["queries"]
    labels = [_norm(d) for s in specs for d in (s.get("expect_docs") or [])]
    first = QMD_CACHE / f"{list(ARMS)[0]}.sqlite"
    for suffix in ("", "-wal", "-shm"):
        Path(str(first) + suffix).unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{LIVE_INDEX}?mode=ro", uri=True, timeout=120)
    src.execute(f"VACUUM INTO '{first}'")
    src.close()
    db = sqlite3.connect(first)
    rows = db.execute("select id, collection, path from documents where active = 1").fetchall()
    searched = set(VAULT_SEGMENTS)
    rng = random.Random(args.seed)
    keep, gold = set(), 0
    paths = {did: _norm(f"{coll}/{path}") for did, coll, path in rows if coll in searched}
    # A label that names a family ("backlog/", "robot") is satisfied by the sample
    # anyway; forcing it in would keep most of the corpus. Only a narrow label
    # (<= NARROW_LABEL_MAX matches) forces its documents in.
    narrow = [lbl for lbl in set(labels)
              if sum(1 for f in paths.values() if _doc_pair_satisfied(lbl, [f])) <= NARROW_LABEL_MAX]
    for did, coll, path in sorted(rows):
        if coll not in searched:
            continue
        full = paths[did]
        if any(_doc_pair_satisfied(lbl, [full]) for lbl in narrow):
            keep.add(did)
            gold += 1
        elif rng.random() < args.sample:
            keep.add(did)
    db.execute("create temp table keep(id integer primary key)")
    db.executemany("insert into keep values (?)", [(k,) for k in keep])
    db.execute("update documents set active = 0 where active = 1 and id not in (select id from keep)")
    db.commit()
    active = db.execute("select count(*) from documents where active = 1").fetchone()[0]
    db.close()
    for name in list(ARMS)[1:]:
        dst = QMD_CACHE / f"{name}.sqlite"
        shutil.copyfile(first, dst)
    for name, embed in ARMS.items():
        (QMD_CONFIG / f"{name}.yml").write_text(_config_text(embed))
    info = {"documents_active": active, "gold_matching_docs": gold,
            "narrow_labels": len(narrow), "labels": len(set(labels)),
            "sample": args.sample, "seed": args.seed,
            "searched_collections": sorted(searched),
            "taken_at": datetime.now(timezone.utc).isoformat()}
    (Path(args.state) / "prepare.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return info


def _daemon_env() -> tuple[list[str], dict]:
    from scripts.automod.evalpin import production_daemon
    argv, program_env, _src = production_daemon()
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
           "QMD_LLAMA_GPU": "cuda", **program_env}
    return argv, env


def embed(args) -> int:
    argv, env = _daemon_env()
    node = argv[0]
    cli = next(a for a in argv if a.endswith("qmd.js"))
    t0 = time.time()
    log = Path(args.state) / f"embed-{args.name}.log"
    with open(log, "wb") as fh:
        r = subprocess.run([node, cli, "--index", args.name, "embed", "-f", "--timeout", "0"],
                           env=env, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    db = sqlite3.connect(f"file:{QMD_CACHE / (args.name + '.sqlite')}?mode=ro", uri=True)
    n = db.execute("select count(*) from content_vectors").fetchone()[0]
    models = db.execute("select model, count(*) from content_vectors group by model").fetchall()
    db.close()
    info = {"name": args.name, "rc": r.returncode, "wall_s": round(wall, 1), "chunks": n,
            "chunks_per_s": round(n / wall, 2) if wall else None, "models": models}
    (Path(args.state) / f"embed-{args.name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return r.returncode


class _Daemon:
    def __init__(self, name: str, port: int, workdir: Path):
        self.name, self.port, self.workdir = name, port, workdir
        self.proc = None

    def __enter__(self):
        from scripts.automod.evalpin import _die_with_parent, _probe, pin_command
        argv, env = _daemon_env()
        self.fh = open(self.workdir / f"{self.name}.log", "wb")
        self.proc = subprocess.Popen(pin_command(argv, self.port, self.name), env=env,
                                     stdout=self.fh, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, process_group=0,
                                     preexec_fn=_die_with_parent())
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited rc={self.proc.returncode}")
            if _probe(self.port):
                return self
            time.sleep(1)
        raise RuntimeError(f"{self.name} never answered on :{self.port}")

    def __exit__(self, *exc):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        self.fh.close()


def child(args) -> int:
    """Scores the arms; each arm's recalls go to its own daemon via services.qmd."""
    import yaml
    from agent_mcp import vault
    # The topics merge (#1456) is its own arm with its own primary call; off here.
    vault.recall_topics_merge_mode = lambda: "off"
    from agent_mcp.facts import _extract_entities_from_query
    from eval.recall_topics_merge import METRICS, _num, _summ
    from eval import stats as evstats
    from eval.run_eval import _score

    specs = (yaml.safe_load((HERE / "vault_recall_queries.yaml").read_text()) or {})["queries"]
    if args.max_queries:
        specs = specs[: args.max_queries]
    arms = (("control", "sub06"), ("candidate", args.candidate), ("control_b", "sub06"))
    phases: dict = {a: [] for a, _ in arms}
    real_query = vault.qmd_query

    def seeds_for(q):
        return [e for e, _ in (_extract_entities_from_query(q) or [])[:vault.RECALL_SEED_TOP_K]]

    box = {"arm": None}

    def timed_query(payload, **kw):
        t0 = time.perf_counter()
        data = real_query(payload, **kw)
        phases[box["arm"]].append({"ms": (time.perf_counter() - t0) * 1000,
                                   **(((data or {}).get("meta") or {}).get("phases") or {})})
        return data
    vault.qmd_query = timed_query

    recs = []
    for i, spec in enumerate(specs):
        q = spec.get("query") or ""
        seeds = seeds_for(q)
        rec = {"id": spec.get("id"), "arms": {}}
        for arm, name in arms:
            vault.QMD_DAEMON_URL = f"http://localhost:{PORTS[name]}/query"
            box["arm"] = arm
            t0 = time.perf_counter()
            res = vault._vault_recall({"query": q, "limit": 20, "expand_graph": True},
                                      seed_top_k=vault.RECALL_SEED_TOP_K)
            ms = (time.perf_counter() - t0) * 1000
            if "error" in res:
                rec["arms"][arm] = {"scoring": {}, "ms": ms, "error": str(res["error"])[:200]}
                continue
            rec["arms"][arm] = {"scoring": _score(spec, res, seeds), "ms": round(ms, 1)}
        recs.append(rec)
        print(f"[{i + 1}/{len(specs)}] {rec['id']}", flush=True)
    names = [a for a, _ in arms]
    s: dict = {"n_queries": len(recs), "means": {}, "paired_vs_control": {}, "latency": {}}
    for arm in names:
        s["means"][arm] = {}
        for metric, field in METRICS.items():
            vals = [_num(r["arms"][arm]["scoring"].get(field)) for r in recs if not r["arms"][arm].get("error")]
            vals = [v for v in vals if v is not None]
            s["means"][arm][metric] = round(statistics.fmean(vals), 4) if vals else None
        s["latency"][arm] = {"recall_ms": _summ([r["arms"][arm]["ms"] for r in recs]),
                             "qmd_ms": _summ([p["ms"] for p in phases[arm]]),
                             "embed_ms": _summ([p.get("embed") for p in phases[arm]]),
                             "vec_ms": _summ([p.get("vec") for p in phases[arm]])}
    for arm in names[1:]:
        s["paired_vs_control"][arm] = {}
        for metric, field in METRICS.items():
            pairs = [(_num(r["arms"]["control"]["scoring"].get(field)),
                      _num(r["arms"][arm]["scoring"].get(field))) for r in recs]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if not pairs:
                continue
            ci = evstats.paired_bootstrap_ci([x for x, _ in pairs], [y for _, y in pairs])
            s["paired_vs_control"][arm][metric] = {
                "delta": round(ci["diff"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                "p": round(ci["p"], 4), "n": ci["n"],
                "wins": sum(1 for x, y in pairs if y > x),
                "losses": sum(1 for x, y in pairs if y < x)}
    s["errors"] = {a: sum(1 for r in recs if r["arms"][a].get("error")) for a in names}
    out = {"item": args.item, "candidate": args.candidate, "ran_at": datetime.now(timezone.utc).isoformat(),
           "prepare": json.loads((Path(args.state) / "prepare.json").read_text()),
           "summary": s, "records": recs}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    return 0


def evaluate(args) -> int:
    work = Path(tempfile.mkdtemp(prefix="emb-ab-"))
    out = Path(args.out).with_suffix(".json")
    try:
        with _Daemon("sub06", PORTS["sub06"], work), \
                _Daemon(args.candidate, PORTS[args.candidate], work):
            env = {**os.environ, "PYTHONPATH": str(ROOT), "LLOYD_DJEV_SHADOW": "0",
                   "LLOYD_CODE_ROOT": str(ROOT),
                   "LLOYD_FACTS_ROOT": str(Path(args.fact_pin) / "facts"),
                   "LLOYD_KG_DB": str(Path(args.fact_pin) / "kg.sqlite")}
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child",
                                "--candidate", args.candidate, "--item", str(args.item),
                                "--state", args.state, "--max-queries", str(args.max_queries),
                                "--out", str(out)], cwd=str(ROOT), env=env)
            for name in ("sub06", args.candidate):
                shutil.copyfile(work / f"{name}.log", Path(args.state) / f"daemon-{name}.log")
            if r.returncode:
                return r.returncode
    finally:
        shutil.rmtree(work, ignore_errors=True)
    s = json.loads(out.read_text())["summary"]
    print(json.dumps({k: s[k] for k in ("means", "latency", "errors")}, indent=1))
    for arm, rows in s["paired_vs_control"].items():
        for m, v in rows.items():
            print(f"{arm:10s} {m:24s} d={v['delta']:+.4f} ci={v['ci95']} p={v['p']} "
                  f"W/L={v['wins']}/{v['losses']} n={v['n']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("prepare", "embed", "eval", "child"))
    ap.add_argument("--state", default=str(Path.home() / "lloyd-data/eval/1493"))
    ap.add_argument("--name", choices=tuple(EMBEDS), default="sub4b")
    ap.add_argument("--candidate", choices=("sub4b", "subctx"), default="sub4b")
    ap.add_argument("--item", type=int, default=1493)
    ap.add_argument("--sample", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1493)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--fact-pin", default=str(Path.home() / "lloyd-data/eval/pin-2026-09-25"))
    ap.add_argument("--out", default=str(Path.home() / "lloyd-data/eval/1493/eval"))
    args = ap.parse_args(argv)
    Path(args.state).mkdir(parents=True, exist_ok=True)
    return {"prepare": lambda a: (prepare(a), 0)[1], "embed": embed,
            "eval": evaluate, "child": child}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
