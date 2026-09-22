#!/usr/bin/env python3
"""YaRN x2 quality/decode A/B against the primary on :8096.

  corpus            build the three token corpora once (code, docs, agent)
  dense   ARM       teacher-forced prompt logprobs on short windows (<=512 tok)
  sparse  ARM       next-token top-20 at sampled positions in 2k/20k/150k buckets
  decode  ARM       greedy 512-token generations: tok/s, MTP acceptance, text
  compare A B [C]   metrics of B (and C) against A

Every request carries priority 5 (lower runs sooner), so any real turn
outranks it. A request is only sent while the engine reports nothing running.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8096"
MODEL = "primary"
import os
# Raw results are large (~10 MB per dense arm) and belong outside the tree.
HERE = Path(os.environ.get("YARN_AB_OUT", "/tmp/yarn_ab"))
HERE.mkdir(parents=True, exist_ok=True)
LLOYD = Path("/home/alansrobotlab/lloyd")
CORPUS_LEN = 152_000
PRIO = 5
client = httpx.Client(timeout=httpx.Timeout(900.0, connect=10.0))


# ── engine helpers ─────────────────────────────────────────────────────────
def metrics() -> dict[str, float]:
    out: dict[str, float] = {}
    for line in client.get(f"{BASE}/metrics").text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = re.search(r'position="(\d+)"', labels)
            name = f"{name}[{pos.group(1) if pos else '?'}]"
        out[name] = out.get(name, 0.0) + val
    return out


def wait_quiet(max_s: float = 600.0) -> None:
    t0 = time.time()
    while True:
        m = metrics()
        if m.get("vllm:num_requests_running", 0) == 0 and m.get("vllm:num_requests_waiting", 0) == 0:
            return
        if time.time() - t0 > max_s:
            raise SystemExit("engine never went quiet — someone else is using it")
        time.sleep(1.0)


def busy() -> float:
    m = metrics()
    return m.get("vllm:num_requests_running", 0) + m.get("vllm:num_requests_waiting", 0)


def tokenize(**body) -> list[int]:
    r = client.post(f"{BASE}/tokenize", json={"model": MODEL, **body})
    r.raise_for_status()
    return r.json()["tokens"]


def complete(prompt_ids: list[int], **extra) -> dict:
    body = {"model": MODEL, "prompt": prompt_ids, "max_tokens": 1, "temperature": 0.0,
            "priority": PRIO, **extra}
    r = client.post(f"{BASE}/v1/completions", json=body)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
    return r.json()


# ── corpus ─────────────────────────────────────────────────────────────────
def _text_corpus(paths: list[Path], header: str) -> list[int]:
    chunks, ids = [], []
    for p in paths:
        try:
            t = p.read_text(errors="replace")
        except OSError:
            continue
        chunks.append(f"{header} {p.relative_to(LLOYD)}\n{t}\n")
        if sum(len(c) for c in chunks) > 900_000:
            break
    ids = tokenize(prompt="".join(chunks), add_special_tokens=False)
    return ids[:CORPUS_LEN]


def _session_messages(path: Path) -> list[dict]:
    import ast
    d = json.loads(path.read_text())

    def text_of(content) -> str:
        if isinstance(content, str):
            try:
                content = ast.literal_eval(content)
            except Exception:
                return content
        if isinstance(content, list):
            return "".join(b.get("text", "") for b in content if isinstance(b, dict))
        return str(content or "")

    msgs = []
    for m in d["messages"]:
        role = m.get("role")
        if role == "user":
            msgs.append({"role": "user", "content": text_of(m.get("content"))})
        elif role == "assistant":
            out = {"role": "assistant", "content": text_of(m.get("content"))}
            tc = m.get("tool_calls")
            if isinstance(tc, str):
                try:
                    tc = ast.literal_eval(tc)
                except Exception:
                    tc = None
            if tc:
                out["tool_calls"] = [{"id": c.get("id"), "type": "function",
                                      "function": {"name": c["function"]["name"],
                                                   "arguments": c["function"].get("arguments", "{}")}}
                                     for c in tc if isinstance(c, dict) and c.get("function")]
            msgs.append(out)
        elif role == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("tool_call_id"),
                         "content": text_of(m.get("content"))})
    return msgs


def cmd_corpus() -> None:
    py = sorted(p for d in ("app", "agent_mcp", "scripts/automod", "workers")
                for p in (LLOYD / d).rglob("*.py") if "__pycache__" not in p.parts)
    md = sorted((LLOYD / "architecture").glob("*.md")) + [LLOYD / "SETUP.md", LLOYD / "CLAUDE.md"]
    corpus = {"code": _text_corpus(py, "# file:"), "docs": _text_corpus(md, "<!-- file:")}
    agent: list[int] = []
    for s in sorted((LLOYD / "sessions").glob("*autocode*.json"), key=lambda p: -p.stat().st_size)[:6]:
        ids = tokenize(messages=_session_messages(s), add_generation_prompt=False)
        agent.extend(ids)
        print(f"  agent: {s.name} -> {len(ids)} tokens")
        if len(agent) >= CORPUS_LEN:
            break
    corpus["agent"] = agent[:CORPUS_LEN]
    for k, v in corpus.items():
        print(f"{k}: {len(v)} tokens")
        assert len(v) >= CORPUS_LEN, f"{k} corpus too short"
    (HERE / "corpus.json").write_text(json.dumps(corpus))


def load_corpus() -> dict[str, list[int]]:
    return json.loads((HERE / "corpus.json").read_text())


# ── dense: short-window prompt logprobs ─────────────────────────────────────
def cmd_dense(arm: str, window: int = 512, per_type: int = 150) -> None:
    corpus = load_corpus()
    out = []
    for kind, ids in corpus.items():
        starts = [int(i * (CORPUS_LEN - 512) / (per_type - 1)) for i in range(per_type)]  # fixed grid: arms must align
        for s in starts:
            w = ids[s:s + window]
            wait_quiet()
            r = complete(w, prompt_logprobs=1)
            pl = r["choices"][0]["prompt_logprobs"]
            rows = []
            for i in range(1, len(w)):
                d = pl[i] or {}
                act = d.get(str(w[i]))
                top = min(d.items(), key=lambda kv: kv[1]["rank"]) if d else None
                rows.append([act["logprob"] if act else None, act["rank"] if act else None,
                             int(top[0]) if top else None, top[1]["logprob"] if top else None])
            out.append({"kind": kind, "start": s, "rows": rows})
        print(f"  dense {arm} {kind}: {per_type} windows", flush=True)
    (HERE / f"dense_{arm}.json").write_text(json.dumps(out))


# ── sparse: next-token top-20 at sampled positions ──────────────────────────
BUCKETS = {"2k": (1500, 2000, 5), "20k": (18000, 20000, 20), "150k": (135000, 150000, 100)}


def cmd_sparse(arm: str) -> None:
    corpus = load_corpus()
    out = []
    for kind, ids in corpus.items():
        for bucket, (a, b, step) in BUCKETS.items():
            t0 = time.time()
            for p in range(a, b, step):
                wait_quiet()
                r = complete(ids[:p], logprobs=20, return_tokens_as_token_ids=True)
                top = r["choices"][0]["logprobs"]["top_logprobs"][0]
                tops = {int(k.split(":", 1)[1]): v for k, v in top.items()}
                out.append({"kind": kind, "bucket": bucket, "pos": p, "actual": ids[p], "top": tops})
            print(f"  sparse {arm} {kind} {bucket}: {time.time() - t0:.0f}s", flush=True)
    (HERE / f"sparse_{arm}.json").write_text(json.dumps(out))


# ── decode: greedy generations ──────────────────────────────────────────────
def _decode_prompts() -> list[tuple[str, list[dict], bool]]:
    corpus = load_corpus()
    doc32k = client.post(f"{BASE}/detokenize",
                         json={"model": MODEL, "tokens": corpus["docs"][40000:72000]}).json()["prompt"]
    return [
        ("code", [{"role": "user", "content":
            "Write a complete Python module implementing a thread-safe LRU cache with per-entry TTL, "
            "size-based eviction, hit/miss statistics and an async wrapper. Include type hints, "
            "docstrings, and a pytest test suite covering eviction order, expiry and concurrency."}], False),
        ("json", [{"role": "user", "content":
            "Output only a JSON array (no prose) of 14 tool-call objects for an agent auditing a Linux "
            "server. Each object has keys: id (string), name (one of Bash, Read, Grep, Glob, http_fetch), "
            "arguments (object with realistic, fully specified parameters), and rationale (one sentence)."}], False),
        ("prose", [{"role": "user", "content":
            "Write a long, carefully argued essay on why distributed systems fail in practice: partial "
            "failure, clock skew, retries and idempotency, backpressure, and the human side of incident "
            "response. Use concrete examples and flowing paragraphs, no bullet points."}], False),
        ("thinking", [{"role": "user", "content":
            "A warehouse has 7 robots on a 12x12 grid. Each robot moves one cell per tick, cannot share a "
            "cell, and must visit two pickup cells before a dock. Devise a collision-free scheduling "
            "strategy, prove it terminates, and estimate the makespan for the worst case."}], True),
        ("summary32k", [{"role": "user", "content":
            doc32k + "\n\nSummarize the key design decisions in the documentation above, grouped by "
            "subsystem, with the reason given for each."}], False),
    ]


def _stream(messages: list[dict], thinking: bool, max_tokens: int = 512) -> dict:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "priority": PRIO, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    text, t_first, t_last, usage, peak_busy = [], None, None, {}, 0.0
    with client.stream("POST", f"{BASE}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("reasoning") or d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    now = time.time()
                    t_first = t_first or now
                    t_last = now
                    text.append(piece)
    return {"text": "".join(text), "usage": usage, "t_first": t_first, "t_last": t_last}


def cmd_decode(arm: str, reps: int = 2) -> None:
    out = []
    for name, msgs, thinking in _decode_prompts():
        for rep in range(reps):
            wait_quiet()
            m0 = metrics()
            res = _stream(msgs, thinking)
            m1 = metrics()
            dk = lambda k: m1.get(k, 0) - m0.get(k, 0)
            n = res["usage"].get("completion_tokens", 0)
            span = (res["t_last"] - res["t_first"]) if res["t_first"] else float("nan")
            drafts = dk("vllm:spec_decode_num_drafts_total")
            acc = dk("vllm:spec_decode_num_accepted_tokens_total")
            per_pos = [dk(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]") / drafts if drafts else None
                       for i in range(3)]
            row = {"prompt": name, "rep": rep, "tokens": n, "tok_s": (n - 1) / span if span else None,
                   "accept_len": 1 + acc / drafts if drafts else None, "accept_per_pos": per_pos,
                   "prompt_tokens": res["usage"].get("prompt_tokens"), "text": res["text"],
                   "busy_after": busy()}
            out.append(row)
            print(f"  decode {arm} {name} rep{rep}: {n} tok {row['tok_s']:.1f} tok/s "
                  f"accept {row['accept_len']:.2f}", flush=True)
    (HERE / f"decode_{arm}.json").write_text(json.dumps(out))


# ── compare ─────────────────────────────────────────────────────────────────
def _boot_ci(pairs_by_block: list[list[float]], n: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    rng = random.Random(seed)
    blocks = [b for b in pairs_by_block if b]
    flat_mean = lambda bs: sum(sum(b) for b in bs) / max(1, sum(len(b) for b in bs))
    means = sorted(flat_mean([rng.choice(blocks) for _ in blocks]) for _ in range(n))
    return flat_mean(blocks), means[int(0.025 * n)], means[int(0.975 * n)]


def _dense_report(a: str, b: str) -> None:
    A = json.loads((HERE / f"dense_{a}.json").read_text())
    B = json.loads((HERE / f"dense_{b}.json").read_text())
    for kind in ("code", "docs", "agent", "ALL"):
        blocks, diffs, agree, nll_a, nll_b, big = [], [], [], [], [], 0
        for wa, wb in zip(A, B):
            if kind != "ALL" and wa["kind"] != kind:
                continue
            blk = []
            for ra, rb in zip(wa["rows"], wb["rows"]):
                if ra[0] is None or rb[0] is None:
                    continue
                nll_a.append(-ra[0]); nll_b.append(-rb[0])
                d = (-rb[0]) - (-ra[0])
                blk.append(d); diffs.append(d)
                big += abs(d) > 1.0
                agree.append(ra[2] == rb[2])
            blocks.append(blk)
        mean, lo, hi = _boot_ci(blocks)
        sd = st.pstdev(diffs)
        print(f"  dense {kind:5s} n={len(diffs):6d}  NLL {st.mean(nll_a):.4f} -> {st.mean(nll_b):.4f}  "
              f"dNLL {mean:+.4f} [{lo:+.4f},{hi:+.4f}]  sd(d) {sd:.3f}  >1nat {big / len(diffs):.2%}  "
              f"top1-agree {sum(agree) / len(agree):.2%}")


def _sparse_report(a: str, b: str) -> None:
    A = json.loads((HERE / f"sparse_{a}.json").read_text())
    B = json.loads((HERE / f"sparse_{b}.json").read_text())
    for bucket in BUCKETS:
        for kind in ("code", "docs", "agent", "ALL"):
            ra = [r for r in A if r["bucket"] == bucket and (kind == "ALL" or r["kind"] == kind)]
            rb = [r for r in B if r["bucket"] == bucket and (kind == "ALL" or r["kind"] == kind)]
            agree = diffs = 0
            dl, blocks, acc_a, acc_b, cov = [], {}, 0, 0, 0
            for x, y in zip(ra, rb):
                assert x["pos"] == y["pos"]
                ta = {int(k): v for k, v in x["top"].items()}
                tb = {int(k): v for k, v in y["top"].items()}
                top_a, top_b = max(ta, key=ta.get), max(tb, key=tb.get)
                agree += top_a == top_b
                acc_a += top_a == x["actual"]; acc_b += top_b == x["actual"]
                if x["actual"] in ta and x["actual"] in tb:
                    cov += 1
                    d = -tb[x["actual"]] - (-ta[x["actual"]])
                    dl.append(d)
                    blocks.setdefault((x["kind"], x["pos"]), []).append(d)  # per-position resampling
            n = len(ra)
            mean, lo, hi = _boot_ci(list(blocks.values())) if dl else (float("nan"),) * 3
            print(f"  sparse {bucket:4s} {kind:5s} n={n:4d}  top1-agree {agree / n:.1%}  "
                  f"acc {acc_a / n:.1%}->{acc_b / n:.1%}  dNLL(actual, n={cov}) {mean:+.4f} [{lo:+.4f},{hi:+.4f}]")


def _decode_report(a: str, b: str) -> None:
    A = json.loads((HERE / f"decode_{a}.json").read_text())
    B = json.loads((HERE / f"decode_{b}.json").read_text())
    for name in dict.fromkeys(r["prompt"] for r in A):
        xa = [r for r in A if r["prompt"] == name]
        xb = [r for r in B if r["prompt"] == name]
        f = lambda rs, k: st.mean(r[k] for r in rs if r[k] is not None)
        ta, tb = xa[0]["text"], xb[0]["text"]
        common = next((i for i, (p, q) in enumerate(zip(ta, tb)) if p != q), min(len(ta), len(tb)))
        print(f"  decode {name:10s} tok/s {f(xa, 'tok_s'):6.1f} -> {f(xb, 'tok_s'):6.1f}  "
              f"accept {f(xa, 'accept_len'):.2f} -> {f(xb, 'accept_len'):.2f}  "
              f"same-text-prefix {common}/{min(len(ta), len(tb))} chars")


def cmd_compare(a: str, *others: str) -> None:
    for b in others:
        print(f"=== {b} vs {a}")
        for fn in (_dense_report, _sparse_report, _decode_report):
            try:
                fn(a, b)
            except FileNotFoundError as e:
                print(f"  (missing {e.filename})")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"corpus": cmd_corpus, "dense": cmd_dense, "sparse": cmd_sparse,
     "decode": cmd_decode, "compare": cmd_compare}[cmd](*args)
