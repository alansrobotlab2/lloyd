#!/usr/bin/env python3
"""#1494 on the FULL searched corpus: LLM-written situating context, measured offline.

`eval/contextual_titles.py` measured the document-level proxy on the #1493
sub-corpus (1,564 docs): MRR +0.054 [-0.007, +0.115], p=0.087. This runs the same
proxy on every active document of the collections `_vault_recall` searches
(`VAULT_SEGMENTS`), with the stock fork build, and never touches the live index,
`index.yml` or `evalpin.yml`:

  generate  asks the primary, once per distinct document, for a <=40-word
            situating context (the predecessor's prompt and cap, re-used by
            import). Resumable: contexts are keyed by content hash and seeded
            from the sub-corpus run. Take primary.lock SHARED around it.
  prepare   VACUUMs the live index INTO two scratch files under STATE/idx
            (read-only on the source), deactivates every collection the recall
            does not search in both, and in `ctx` rewrites each document's first
            heading as `<heading> — <context>` and re-fires the FTS trigger, so
            the context reaches every chunk's embed text (qmd prefixes the title
            it extracts from that heading) and the lex leg.
  embed     re-embeds one arm from scratch with production's 0.6B
            (`INDEX_PATH` + `QMD_CONFIG_DIR`, so the fork's CLI reads and writes
            only the scratch file and the scratch config). Both arms are
            re-embedded, so they differ in the context and nothing else.
  score     serves both arms on their own ports and runs the unmodified
            `eval/run_eval.py` against each (config overlay pointing
            `services.qmd` at the arm, the pinned fact tree, one code root),
            dev set and holdout leg, with djev under one replay file per leg
            anchored on `plain`. Dev: paired bootstrap per metric from the
            records. Holdout: aggregates only (#1412's reserve rule), and the
            holdout run records are deleted once aggregated.
  clean     deletes the scratch indexes (the contexts are kept).

`embed` and `score` hold regression.lock exclusively (GPU 0, daemons beside
production's) — the caller wraps them in flock.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
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

LIVE_INDEX = Path.home() / ".cache/qmd/index.sqlite"
EVALPIN_CONF = Path.home() / ".config/qmd/evalpin.yml"      # read, never written
BASE = Path.home() / "lloyd-data/eval/1494"
CTX_DIR = BASE / "contexts-full"
STATE = BASE / "full"
IDX = STATE / "idx"
CONF = IDX / "conf"
ARMS = ("plain", "ctx")
PORTS = {"plain": 8187, "ctx": 8188}
FACT_PIN = Path.home() / "lloyd-data/eval/pin-2026-09-25"
METRICS = {"doc_hit_rate": "doc_hit", "mrr_doc": "rr_doc", "ndcg10": "ndcg10",
           "doc_recall_avg": "doc_recall"}


def _searched() -> list[str]:
    from agent_mcp.vault import VAULT_SEGMENTS
    return list(VAULT_SEGMENTS)


def _live_docs() -> list[tuple[str, str, str]]:
    segs = _searched()
    db = sqlite3.connect(f"file:{LIVE_INDEX}?mode=ro", uri=True, timeout=120)
    q = ",".join("?" * len(segs))
    rows = db.execute(f"""select d.hash, min(d.collection || '/' || d.path), c.doc
                          from documents d join content c on c.hash = d.hash
                          where d.active = 1 and d.collection in ({q})
                          group by d.hash""", segs).fetchall()
    db.close()
    return rows


# ---------------------------------------------------------------- generate
def generate(args) -> int:
    from app.secondary_models import _endpoint
    from eval.contextual_titles import DOC_CHARS, PROMPT, _post
    url, model = _endpoint("focus")
    CTX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CTX_DIR / "contexts.json"
    done: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    seeded = 0
    prior = BASE / "contexts.json"          # the sub-corpus run, same prompt, same model
    if prior.exists() and not args.no_seed:
        for h, c in json.loads(prior.read_text()).items():
            if h not in done:
                done[h] = c
                seeded += 1
    rows = _live_docs()
    todo = [r for r in rows if r[0] not in done]
    print(f"{len(rows)} documents, {len(rows) - len(todo)} already have a context "
          f"({seeded} seeded from the sub-corpus run), {len(todo)} to generate", flush=True)
    t0 = time.time()
    lat: list[float] = []
    errors: list[str] = []

    def one(row):
        h, p, doc = row
        s = time.perf_counter()
        try:
            ctx = _post(url, model, PROMPT.format(path=p, doc=(doc or "")[:DOC_CHARS]))
        except Exception as e:  # noqa: BLE001
            return h, None, f"{p}: {str(e)[:120]}", 0.0
        return h, ctx, None, time.perf_counter() - s

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (h, ctx, err, dt) in enumerate(ex.map(one, todo), start=1):
            if ctx:
                done[h] = ctx
                lat.append(dt)
            else:
                errors.append(err or "empty")
            if i % 100 == 0:
                out_path.write_text(json.dumps(done))
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s errors={len(errors)}", flush=True)
    out_path.write_text(json.dumps(done))
    have = sum(1 for r in rows if r[0] in done)
    info = {"documents": len(rows), "with_context": have, "generated_this_run": len(lat),
            "seeded": seeded, "errors": len(errors), "error_sample": errors[:5],
            "wall_s": round(time.time() - t0, 1), "workers": args.workers,
            "mean_call_s": round(statistics.fmean(lat), 2) if lat else None,
            "model": model, "collections": _searched(),
            "mean_words": round(statistics.fmean(len(done[r[0]].split()) for r in rows
                                                 if r[0] in done), 1) if have else None,
            "finished_at": datetime.now(timezone.utc).isoformat()}
    (CTX_DIR / "generate.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0 if not errors else 1


# ---------------------------------------------------------------- prepare
def _conf_text() -> str:
    """evalpin.yml's models block with no collections: the arms' configs."""
    return EVALPIN_CONF.read_text()


def prepare(args) -> int:
    from eval.contextual_titles import with_context
    IDX.mkdir(parents=True, exist_ok=True)
    CONF.mkdir(parents=True, exist_ok=True)
    segs = _searched()
    plain = IDX / "plain.sqlite"
    for arm in ARMS:
        for suffix in ("", "-wal", "-shm"):
            Path(str(IDX / f"{arm}.sqlite") + suffix).unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{LIVE_INDEX}?mode=ro", uri=True, timeout=120)
    src.execute(f"VACUUM INTO '{plain}'")
    src.close()
    db = sqlite3.connect(plain)
    q = ",".join("?" * len(segs))
    db.execute(f"update documents set active = 0 where active = 1 and collection not in ({q})", segs)
    db.commit()
    active = db.execute("select count(*), count(distinct hash) from documents where active = 1").fetchone()
    db.close()
    shutil.copyfile(plain, IDX / "ctx.sqlite")
    ctxs = json.loads((CTX_DIR / "contexts.json").read_text())
    db = sqlite3.connect(IDX / "ctx.sqlite")
    rows = db.execute("""select c.hash, min(d.collection || '/' || d.path), c.doc
                         from content c join documents d on d.hash = c.hash
                         where d.active = 1 group by c.hash""").fetchall()
    n = 0
    for h, p, doc in rows:
        c = ctxs.get(h)
        if c:
            db.execute("update content set doc = ? where hash = ?", (with_context(doc, p, c), h))
            n += 1
    db.execute("update documents set title = title where active = 1")   # re-fires the FTS trigger
    db.commit()
    db.close()
    for arm in ARMS:
        (CONF / f"{arm}.yml").write_text(_conf_text())
    info = {"documents_active": active[0], "distinct_active": active[1],
            "with_context": n, "without_context": len(rows) - n,
            "collections": segs, "source": str(LIVE_INDEX),
            "taken_at": datetime.now(timezone.utc).isoformat()}
    (STATE / "prepare.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


# ---------------------------------------------------------------- embed / daemons
def _arm_env(arm: str) -> tuple[list[str], dict]:
    from scripts.automod.evalpin import production_daemon
    argv, program_env, _src = production_daemon()
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
           "QMD_LLAMA_GPU": "cuda", **program_env,
           "INDEX_PATH": str(IDX / f"{arm}.sqlite"), "QMD_CONFIG_DIR": str(CONF)}
    return argv, env


def embed(args) -> int:
    argv, env = _arm_env(args.arm)
    node, cli = argv[0], next(a for a in argv if a.endswith("qmd.js"))
    t0 = time.time()
    with open(STATE / f"embed-{args.arm}.log", "wb") as fh:
        r = subprocess.run([node, cli, "--index", args.arm, "embed", "-f", "--timeout", "0"],
                           env=env, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    db = sqlite3.connect(f"file:{IDX / (args.arm + '.sqlite')}?mode=ro", uri=True)
    n = db.execute("select count(*) from content_vectors").fetchone()[0]
    models = db.execute("select model, count(*) from content_vectors group by model").fetchall()
    db.close()
    info = {"arm": args.arm, "rc": r.returncode, "wall_s": round(wall, 1), "chunks": n,
            "chunks_per_s": round(n / wall, 2) if wall else None, "models": models}
    (STATE / f"embed-{args.arm}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return r.returncode


class _Daemon:
    def __init__(self, arm: str, workdir: Path):
        self.arm, self.port, self.workdir = arm, PORTS[arm], workdir

    def __enter__(self):
        from scripts.automod.evalpin import _die_with_parent, _probe, pin_command, port_free
        if not port_free(self.port):
            raise RuntimeError(f":{self.port} is in use; refusing to score against a daemon I did not start")
        argv, env = _arm_env(self.arm)
        self.fh = open(self.workdir / f"daemon-{self.arm}.log", "wb")
        self.proc = subprocess.Popen(pin_command(argv, self.port, self.arm), env=env,
                                     stdout=self.fh, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, process_group=0,
                                     preexec_fn=_die_with_parent())
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.arm} daemon exited rc={self.proc.returncode}")
            if _probe(self.port) and self.proc.poll() is None:
                self._warm()
                return self
            time.sleep(1)
        raise RuntimeError(f"{self.arm} daemon never answered on :{self.port}")

    def _warm(self) -> None:
        """The vector index loads on the first vector query, and until then
        /health reports `vectors: 0` — which `run_eval` rightly refuses as an
        empty corpus. Ask production's recall shape a few times first."""
        import urllib.request
        from scripts.automod.evalpin import production_payload
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for q in ("how does the automod gate decide a landing",
                  "what voice does Lloyd speak alerts in",
                  "which model serves the primary slot"):
            req = urllib.request.Request(f"http://localhost:{self.port}/query",
                                         data=json.dumps(production_payload(q)).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with opener.open(req, timeout=180) as r:
                r.read()
        with opener.open(f"http://localhost:{self.port}/health", timeout=30) as r:
            vec = (json.loads(r.read()).get("vecIndex") or {}).get("vectors")
        if not vec:
            raise RuntimeError(f"{self.arm} daemon reports vectors={vec!r} after warm-up")

    def __exit__(self, *exc):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        self.fh.close()


# ---------------------------------------------------------------- score
def _run_arm(arm: str, replay_arm: str, leg: str, data_root: Path, replay_db: Path,
             overlay: Path) -> dict:
    from app.djev import replay_env
    from eval import retrieval_holdout
    queries = retrieval_holdout.HOLDOUT_QUERIES if leg == "holdout" else retrieval_holdout.DEV_QUERIES
    env = {**os.environ, "PYTHONPATH": str(ROOT), "LLOYD_DATA": str(data_root),
           "LLOYD_CONFIG_OVERLAY": str(overlay), "LLOYD_DJEV_SHADOW": "0",
           "LLOYD_CODE_ROOT": str(ROOT), "LLOYD_QMD_INDEX": str(IDX / f"{arm}.sqlite"),
           "LLOYD_FACTS_ROOT": str(FACT_PIN / "facts"), "LLOYD_KG_DB": str(FACT_PIN / "kg.sqlite"),
           "LLOYD_QMD_TIMEOUT_S": "60",
           **replay_env(replay_db, replay_arm, "baseline")}
    if leg == "holdout":
        env[retrieval_holdout.HOLDOUT_LEG_ENV] = "1"
    label = f"x1494-{leg}-{replay_arm}"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "run_eval.py"), "--label", label,
                        "--queries", str(queries), "--no-counterfactual"],
                       cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=3600)
    log = STATE / f"run-{leg}-{replay_arm}.log"
    # The holdout leg logs the fact, never the per-query tail (#1412).
    log.write_text(f"rc={r.returncode} wall={time.time() - t0:.0f}s\n" +
                   ("" if leg == "holdout" else r.stdout[-4000:] + r.stderr[-4000:]))
    if r.returncode:
        raise RuntimeError(f"run_eval {leg}/{replay_arm} rc={r.returncode}: "
                           + ("(holdout: see nothing)" if leg == "holdout" else r.stderr[-400:]))
    files = sorted((data_root / "eval" / "baselines").glob(f"{label}-*.json"), key=os.path.getmtime)
    return json.loads(files[-1].read_text())


def _paired(control: dict, cand: dict) -> dict:
    from eval import stats as evstats
    a_rec = {r["id"]: r for r in control["records"] if not r.get("error")}
    b_rec = {r["id"]: r for r in cand["records"] if not r.get("error")}
    ids = [i for i in a_rec if i in b_rec]
    out = {}
    for metric, field in METRICS.items():
        pairs = []
        for i in ids:
            x, y = a_rec[i]["scoring"].get(field), b_rec[i]["scoring"].get(field)
            if x is None or y is None:
                continue
            pairs.append((float(x), float(y)))
        if not pairs:
            continue
        ci = evstats.paired_bootstrap_ci([x for x, _ in pairs], [y for _, y in pairs])
        out[metric] = {"control": round(statistics.fmean(x for x, _ in pairs), 4),
                       "candidate": round(statistics.fmean(y for _, y in pairs), 4),
                       "delta": round(ci["diff"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                       "p": round(ci["p"], 4), "n": ci["n"], "significant": ci["significant"],
                       "wins": sum(1 for x, y in pairs if y > x),
                       "losses": sum(1 for x, y in pairs if y < x)}
    return out


def score(args) -> int:
    from app.djev import replay_stats
    from scripts.automod.evalpin import write_overlay
    work = Path(tempfile.mkdtemp(prefix="ctx1494-"))
    data_root = STATE / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    out: dict = {"item": 1494, "ran_at": datetime.now(timezone.utc).isoformat(),
                 "prepare": json.loads((STATE / "prepare.json").read_text()),
                 "embed": {a: json.loads((STATE / f"embed-{a}.json").read_text()) for a in ARMS},
                 "legs": {}}
    try:
        with _Daemon("plain", work), _Daemon("ctx", work):
            overlays = {a: write_overlay(work / f"overlay-{a}.yaml", PORTS[a]) for a in ARMS}
            for leg in args.legs:
                replay_db = STATE / f"djev-replay-{leg}.sqlite"
                replay_db.unlink(missing_ok=True)
                runs = {}
                # plain first: it is the replay anchor ("baseline"); plain_b replays
                # it (the pipeline's own determinism); ctx is the candidate.
                for arm, rarm in (("plain", "baseline"), ("ctx", "current"), ("plain", "confirm")):
                    print(f"[{leg}] {arm} as {rarm}", flush=True)
                    runs[rarm] = _run_arm(arm, rarm, leg, data_root, replay_db, overlays[arm])
                leg_out = {"n_queries": {k: v["summary"]["overall"]["n_queries"] for k, v in runs.items()},
                           "errors": {k: v["summary"]["overall"].get("errors") for k, v in runs.items()},
                           "corpus_ok": {k: v.get("corpus_ok") for k, v in runs.items()},
                           "latency_ms_avg": {k: v["summary"]["overall"].get("latency_ms_avg")
                                              for k, v in runs.items()},
                           "djev_replay": replay_stats(replay_db),
                           "candidate_vs_control": _paired(runs["baseline"], runs["current"]),
                           "control_b_vs_control": _paired(runs["baseline"], runs["confirm"])}
                out["legs"][leg] = leg_out
                if leg == "holdout":
                    # Reserve rule: aggregates only. The per-query records go.
                    for f in (data_root / "eval" / "baselines").glob("x1494-holdout-*.json"):
                        f.unlink()
                (STATE / "score.json").write_text(json.dumps(out, indent=1, default=str))
    finally:
        for f in work.glob("daemon-*.log"):
            shutil.copyfile(f, STATE / f.name)
        shutil.rmtree(work, ignore_errors=True)
    out["overfit_suspected"] = _overfit(out)
    (STATE / "score.json").write_text(json.dumps(out, indent=1, default=str))
    for leg, lo in out["legs"].items():
        for cmp_name in ("candidate_vs_control", "control_b_vs_control"):
            for m, v in lo[cmp_name].items():
                print(f"{leg:8s} {cmp_name:22s} {m:16s} {v['control']:.4f}->{v['candidate']:.4f} "
                      f"d={v['delta']:+.4f} ci={v['ci95']} p={v['p']} W/L={v['wins']}/{v['losses']} n={v['n']}")
        print(leg, "djev", lo["djev_replay"], "errors", lo["errors"], "latency", lo["latency_ms_avg"])
    print("overfit_suspected:", out["overfit_suspected"])
    return 0


def _overfit(out: dict) -> dict:
    """#549's strict form, read off the paired intervals: a metric that GAINED on
    dev clear of its interval while LOSING on holdout clear of its own."""
    dev = (out["legs"].get("dev") or {}).get("candidate_vs_control") or {}
    hold = (out["legs"].get("holdout") or {}).get("candidate_vs_control") or {}
    flagged = [m for m in dev if m in hold and dev[m]["ci95"][0] > 0 and hold[m]["ci95"][1] < 0]
    soft = [m for m in dev if m in hold and dev[m]["delta"] > 0 and hold[m]["delta"] < 0]
    return {"flagged": flagged, "sign_disagrees": soft}


def clean(args) -> int:
    freed = 0
    if IDX.exists():
        freed = sum(p.stat().st_size for p in IDX.rglob("*") if p.is_file())
        shutil.rmtree(IDX)
    print(json.dumps({"removed": str(IDX), "bytes": freed}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("generate", "prepare", "embed", "score", "clean"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--arm", choices=ARMS, default="plain")
    ap.add_argument("--legs", nargs="+", default=["dev", "holdout"], choices=("dev", "holdout"))
    args = ap.parse_args(argv)
    STATE.mkdir(parents=True, exist_ok=True)
    return {"generate": generate, "prepare": prepare, "embed": embed,
            "score": score, "clean": clean}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
