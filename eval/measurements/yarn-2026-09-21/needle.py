#!/usr/bin/env python3
"""Long-range retrieval, graded exactly, identical prompts on every arm.

  build                    haystack text from the repo (~540k tokens)
  run ARM L1,L2,... [T]    T trials per length (default 8); each trial hides 4
                           access codes at spread depths and asks for each one
                           in a separate request over the same prefix
  report A [B]             accuracy by length and depth band

Every depth slot in [0.02, 0.98] is covered once per length (32 slots = 8
trials x 4 needles). Priority 5; a request is only sent while the engine is
idle, and VRAM is sampled across each prefill.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.argv, _argv = sys.argv[:1], sys.argv
import yarn_ab as Y  # noqa: E402  (engine helpers, same client and priority)
sys.argv = _argv

HERE = Y.HERE
LLOYD = Y.LLOYD
NAMES = ("Aldebaran Brisket Cobalt Dunmore Everglade Foxglove Garnet Halyard Ironwood Juniper "
         "Kestrel Lanyard Marlowe Nettle Obsidian Pemberton Quillon Rosewood Saltmarsh Tamarind "
         "Umber Vesper Wickham Xenon Yarrow Zephyr Arbor Bramble Cinder Delphine Ember Fennick").split()


def cmd_build() -> None:
    dirs = ("app", "agent_mcp", "scripts", "workers", "tests", "eval", "architecture", "web/src")
    files = sorted(p for d in dirs for p in (LLOYD / d).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".md", ".ts", ".tsx", ".yaml")
                   and "__pycache__" not in p.parts and "node_modules" not in p.parts)
    parts, total = [], 0
    for p in files:
        t = p.read_text(errors="replace")
        parts.append(f"\n\n# ==== {p.relative_to(LLOYD)} ====\n{t}")
        total += len(t)
        if total > 2_600_000:
            break
    text = "".join(parts)
    ids = Y.tokenize(prompt=text, add_special_tokens=False)
    print(f"haystack: {len(ids)} tokens from {len(parts)} files")
    assert len(ids) >= 520_000
    (HERE / "haystack.json").write_text(json.dumps(ids[:520_000]))


def _vram() -> int:
    return int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"]).decode())


def _ask(messages: list[dict]) -> tuple[dict, float, int]:
    body = {"model": Y.MODEL, "messages": messages, "max_tokens": 24, "temperature": 0.0,
            "priority": Y.PRIO, "chat_template_kwargs": {"enable_thinking": False}}
    peak, stop = [_vram()], [False]

    def watch():
        while not stop[0]:
            peak.append(_vram())
            time.sleep(0.25)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t0 = time.time()
    try:
        r = Y.client.post(f"{Y.BASE}/v1/chat/completions", json=body)
    finally:
        stop[0] = True
        th.join()
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json(), time.time() - t0, max(peak)


def cmd_run(arm: str, lengths: str, trials: str = "8") -> None:
    hay = json.loads((HERE / "haystack.json").read_text())
    out_path = HERE / f"needle_{arm}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else []
    T = int(trials)
    for L in (int(x) for x in lengths.split(",")):
        text = Y.client.post(f"{Y.BASE}/detokenize",
                             json={"model": Y.MODEL, "tokens": hay[:L - 400]}).json()["prompt"]
        # cut on line boundaries so a needle never lands inside a token run
        lines = text.split("\n")
        for t in range(T):
            rng = random.Random(L * 1000 + t)
            slots = [t + k * T for k in range(4)]                     # 4 of the 4*T depth slots
            depths = [0.02 + 0.96 * s / (4 * T - 1) for s in slots]
            names = rng.sample(NAMES, 4)
            codes = [f"{rng.randrange(10**6, 10**7)}" for _ in range(4)]
            doc = list(lines)
            for d, n, c in sorted(zip(depths, names, codes), key=lambda z: -z[0]):
                at = int(d * len(doc))
                doc.insert(at, f"# NOTE: the access code for vault {n} is {c}.")
            hay_text = "\n".join(doc)
            for d, n, c in zip(depths, names, codes):
                q = (f"{hay_text}\n\n---\nQuestion: somewhere above is a note giving the access code "
                     f"for vault {n}. What is that access code? Reply with the 7-digit code only.")
                Y.wait_quiet()
                r, dt, peak = _ask([{"role": "user", "content": q}])
                ans = r["choices"][0]["message"].get("content") or ""
                got = re.findall(r"\d{7}", ans)
                out.append({"L": L, "trial": t, "depth": round(d, 3), "name": n, "code": c,
                            "answer": ans.strip()[:60], "correct": bool(got) and got[0] == c,
                            "other_needle": bool(got) and got[0] in codes and got[0] != c,
                            "prompt_tokens": r["usage"]["prompt_tokens"], "seconds": round(dt, 2),
                            "vram_peak_mib": peak})
                out_path.write_text(json.dumps(out))
            last = [o for o in out if o["L"] == L and o["trial"] == t]
            print(f"  needle {arm} L={L} trial {t}: {sum(o['correct'] for o in last)}/4  "
                  f"prefill {last[0]['seconds']:.1f}s  vram peak {max(o['vram_peak_mib'] for o in last)} MiB",
                  flush=True)


def cmd_hard(arm: str, lengths: str, trials: str = "4") -> None:
    """24 codes per prompt under confusable names (Kestrel-17 / Kestrel-71 /
    Kestral-17 ...), spread over the whole depth; 4 of them asked per trial.
    Same prompts on every arm (seeded by length and trial)."""
    hay = json.loads((HERE / "haystack.json").read_text())
    out_path = HERE / f"hard_{arm}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else []
    T = int(trials)
    for L in (int(x) for x in lengths.split(",")):
        text = Y.client.post(f"{Y.BASE}/detokenize",
                             json={"model": Y.MODEL, "tokens": hay[:L - 1200]}).json()["prompt"]
        lines = text.split("\n")
        for t in range(T):
            rng = random.Random(L * 7919 + t)
            stems = rng.sample(NAMES, 4)
            names = []
            for s in stems:                       # 6 confusable names per stem
                a, b = rng.sample(range(10, 99), 2)
                typo = s[:-2] + s[-1] + s[-2]
                names += [f"{s}-{a}", f"{s}-{str(a)[::-1]}", f"{typo}-{a}", f"{s}-{b}", f"{typo}-{b}", f"{s}-{a}{b % 10}"]
            names = list(dict.fromkeys(names))[:24]
            codes = [f"{rng.randrange(10**6, 10**7)}" for _ in names]
            depths = sorted(rng.uniform(0.02, 0.98) for _ in names)
            order = list(range(len(names)))
            rng.shuffle(order)
            doc = list(lines)
            for i in sorted(order, key=lambda i: -depths[i]):
                doc.insert(int(depths[i] * len(doc)), f"# NOTE: the access code for vault {names[i]} is {codes[i]}.")
            hay_text = "\n".join(doc)
            for i in rng.sample(range(len(names)), 4):
                q = (f"{hay_text}\n\n---\nQuestion: several notes above give access codes for vaults with "
                     f"similar names. What is the access code for vault {names[i]} exactly? "
                     f"Reply with the 7-digit code only.")
                Y.wait_quiet()
                r, dt, peak = _ask([{"role": "user", "content": q}])
                ans = r["choices"][0]["message"].get("content") or ""
                got = re.findall(r"\d{7}", ans)
                out.append({"L": L, "trial": t, "depth": round(depths[i], 3), "name": names[i], "code": codes[i],
                            "answer": ans.strip()[:60], "correct": bool(got) and got[0] == codes[i],
                            "other_needle": bool(got) and got[0] in codes and got[0] != codes[i],
                            "prompt_tokens": r["usage"]["prompt_tokens"], "seconds": round(dt, 2),
                            "vram_peak_mib": peak})
                out_path.write_text(json.dumps(out))
            last = [o for o in out if o["L"] == L and o["trial"] == t]
            print(f"  hard {arm} L={L} trial {t}: {sum(o['correct'] for o in last)}/4  "
                  f"prefill {max(o['seconds'] for o in last):.1f}s", flush=True)


def cmd_sparse250(arm: str) -> None:
    """Next-token top-20 at 200 positions in [240k, 250k) of the haystack —
    the same measure as yarn_ab's sparse buckets, near the native limit."""
    hay = json.loads((HERE / "haystack.json").read_text())
    out, t0 = [], time.time()
    for p in range(240_000, 250_000, 50):
        Y.wait_quiet()
        r = Y.complete(hay[:p], logprobs=20, return_tokens_as_token_ids=True)
        top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        out.append({"kind": "hay", "bucket": "250k", "pos": p, "actual": hay[p],
                    "top": {int(k.split(":", 1)[1]): v for k, v in top.items()}})
    (HERE / f"sparse250_{arm}.json").write_text(json.dumps(out))
    print(f"  sparse250 {arm}: {len(out)} positions in {time.time() - t0:.0f}s", flush=True)


def cmd_stall(arm: str, prefill_len: str) -> None:
    """Decode a 1024-token chat; 2 s in, submit a cold `prefill_len` prompt at
    the same priority as the chat (0). Record the chat's inter-token gaps."""
    hay = json.loads((HERE / "haystack.json").read_text())
    L = int(prefill_len)
    salt = Y.tokenize(prompt=f"stall probe {time.time()}\n", add_special_tokens=False)
    big = salt + hay[:L - len(salt)]   # the salted first block makes the whole prompt a cache miss
    gaps, stamps = [], []
    fired = {}

    def fire():
        time.sleep(2.0)
        t = time.time()
        Y.client.post(f"{Y.BASE}/v1/completions", json={"model": Y.MODEL, "prompt": big, "max_tokens": 1,
                                                         "temperature": 0, "priority": 0})
        fired["span"] = (t, time.time())
    Y.wait_quiet()
    th = threading.Thread(target=fire, daemon=True)
    body = {"model": Y.MODEL, "messages": [{"role": "user", "content":
            "Write a long, detailed history of the printing press, chapter by chapter."}],
            "max_tokens": 1024, "temperature": 0, "stream": True, "priority": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    with Y.client.stream("POST", f"{Y.BASE}/v1/chat/completions", json=body) as r:
        th.start()
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                ev = json.loads(line[6:])
                if any((c.get("delta") or {}).get("content") for c in ev.get("choices", [])):
                    stamps.append(time.time())
    th.join()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    a, b = fired.get("span", (0, 0))
    during = [g for s, g in zip(stamps[1:], gaps) if a <= s <= b + 1]
    res = {"arm": arm, "prefill_len": L, "prefill_s": round(b - a, 1), "chunks": len(stamps),
           "max_gap_s": round(max(gaps), 2), "gaps_over_1s": sum(g > 1 for g in gaps),
           "stalled_s": round(sum(g for g in during if g > 0.2), 1),
           "chat_tok_s_during": round(len(during) / max(1e-9, b - a), 1)}
    print("  stall", json.dumps(res), flush=True)
    p = HERE / "stall.json"
    allres = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(allres + [res]))


def cmd_report(*arms: str, prefix: str = "needle") -> None:
    if arms and arms[0] in ("needle", "hard"):
        prefix, arms = arms[0], arms[1:]
    rows = {a: json.loads((HERE / f"{prefix}_{a}.json").read_text()) for a in arms}
    Ls = sorted({o["L"] for rs in rows.values() for o in rs})
    print(f"{'L':>7} " + "  ".join(f"{a:>22}" for a in arms))
    for L in Ls:
        cells = []
        for a in arms:
            rs = [o for o in rows[a] if o["L"] == L]
            if not rs:
                cells.append(f"{'—':>22}")
                continue
            ok = sum(o["correct"] for o in rs)
            oth = sum(o["other_needle"] for o in rs)
            sec = max(o["seconds"] for o in rs)
            cells.append(f"{ok:>3}/{len(rs):<3} wrong-needle {oth:<2} {sec:5.0f}s")
        print(f"{L:>7} " + "  ".join(cells))
    for a in arms:
        miss = [o for o in rows[a] if not o["correct"]]
        if miss:
            print(f"  {a} misses: " + ", ".join(f"L={o['L']//1000}k d={o['depth']} -> {o['answer']!r}"
                                                 for o in miss[:12]))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"build": cmd_build, "run": cmd_run, "report": cmd_report, "sparse250": cmd_sparse250,
     "stall": cmd_stall, "hard": cmd_hard}[cmd](*args)
