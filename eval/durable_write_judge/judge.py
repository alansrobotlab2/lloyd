#!/usr/bin/env python3
"""Two offline judges over the #580 corpus, differing only in what they are shown.

Judge A (``--judge a``) sees five labelled examples drawn at random from the
corpus. Judge B (``--judge b``) sees five **nearest-retrieved** labelled examples
(k=5, :mod:`eval.durable_write_judge.retrieve`). Same model, same rubric
preamble, same number of examples, same sample block — Fox's ablation, run
locally: only the example *selection* changes.

Both are run over a fixed commit of the corpus; nothing here is wired into a
live write path. ``--dry-run`` builds and reports every prompt without calling a
model, and the tests use that path — the judging pass itself is a separate,
deliberate command (see README).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:  # imported as eval.durable_write_judge.judge
    from .build_corpus import DEFECT_CLASSES, sample_text
    from .retrieve import build_index, excluded_ids, top_k
except ImportError:  # run as a script from this directory
    from build_corpus import DEFECT_CLASSES, sample_text
    from retrieve import build_index, excluded_ids, top_k

ENGINE_URL = "http://127.0.0.1:8091/v1/chat/completions"  # secondary, llama.cpp
ENGINE_MODEL = "secondary"  # Qwen3.6-35B-A3B GGUF; the primary stays untouched
EXAMPLE_K = 5
SEED = 580
EXAMPLE_TEXT_CHARS = 700

ROLE = (
    "You are auditing one durable knowledge note that an autonomous agent wrote "
    "into its own long-term memory and later may have repaired. Your job is to "
    "decide whether THIS NOTE, as it stands, is safe to keep as a record of what "
    "its source said. Flag it if it plausibly contains any defect below, "
    "including an omission; accept it only if you would stake the record on it."
)

RUBRIC = """Defect classes, each with a worked fail case and a worked pass case.
(These eight lines are the fixed rubric; they are illustrative, not corpus samples.)

1. invented_url — FAIL: "See the implementation at https://github.com/acme/quantum-router"
   appears in the note but the source transcript never names that repository.
   PASS: the note cites only the video URL in its own front matter.
2. false_cutoff_claim — FAIL: "the transcript cuts off before the solutions" when the
   source in fact goes on for another twenty minutes of solutions.
   PASS: "the Q&A runs to the end of the recording" in a note whose source does end.
3. omitted_numerical_data — FAIL: a note about a production system that reports no
   figure at all where the speaker gave the load-bearing numbers (throughput, cost, size).
   PASS: the note quotes the speaker's own numbers with their units.
4. missing_section — FAIL: a talk whose richest material is the audience Q&A, digested
   with no Q&A section at all.
   PASS: a short note covering a five-minute lightning talk end to end.
5. wrong_metadata — FAIL: front matter `speaker:`/`title:`/`published:` that does not
   match the source the note links to.
   PASS: front matter that matches the linked source exactly."""

EXAMPLES_OPEN = "\n\nLabelled examples from past writes. Each shows what was found and, where it was a defect, what the repairer wrote down:\n"
SAMPLE_OPEN = "\n\nThe write to judge now. Judge only this note:\n"

# JSON first, not JSON last: llama.cpp serves this Qwen build with its chain of
# thought in `reasoning_content`, and at 1 500 tokens a measured reply reached the
# object only mid-way (`parse_verdict` returned `unparsed` on both smoke prompts).
# Asking for the object first keeps it inside the budget.
OUTPUT_CONTRACT = """

Your entire answer must be this one JSON object, with no prose before it:
{"verdict": "flag" or "accept", "defect_class": one of {classes} or "none", "reason": one short sentence}""".replace(
    "{classes}", ", ".join(sorted(DEFECT_CLASSES))
)


@dataclass
class Prompt:
    """The three parts, kept separate so a test can prove A and B differ in one
    block only: `head` and `tail` must be byte-identical across judges."""

    head: str
    examples_block: str
    tail: str

    @property
    def text(self) -> str:
        return self.head + self.examples_block + self.tail

    @property
    def example_count(self) -> int:
        return self.examples_block.count("\n--- labelled example ")


def render_example(sample: dict, index: int) -> str:
    label = sample["label"]
    classes = ", ".join(sample.get("defect_classes") or []) or "none"
    evidence = []
    for cls, lines in (sample.get("correction_evidence") or {}).items():
        evidence.append(f"{cls}: " + " | ".join(str(x) for x in lines[:3]))
    return (
        f"\n--- labelled example {index} ---\n"
        f"label: {label}\n"
        f"defect_class: {classes}\n"
        + (f"correction record: {'; '.join(evidence)}\n" if evidence else "")
        + f"write (excerpt):\n{sample_text(sample)[:EXAMPLE_TEXT_CHARS]}\n"
    )


def build_prompt(sample: dict, examples: list[dict]) -> Prompt:
    head = ROLE + "\n\n" + RUBRIC + EXAMPLES_OPEN
    block = "".join(render_example(ex, i) for i, ex in enumerate(examples, 1))
    tail = (
        SAMPLE_OPEN
        + f"vault_path: {sample['vault_path']}\n"
        + f"write:\n{sample_text(sample)}\n"
        + OUTPUT_CONTRACT
    )
    return Prompt(head=head, examples_block=block, tail=tail)


def random_examples(samples: list[dict], sample: dict, k: int = EXAMPLE_K,
                    seed: int = SEED) -> list[dict]:
    """Judge A's five: a seeded random draw from the labelled corpus, excluding
    the sample under test and its same-file twin (the same leak rule as B)."""
    banned = excluded_ids(samples, sample)
    pool = sorted(s["id"] for s in samples if s["id"] not in banned)
    rng = random.Random(f"{seed}:{sample['id']}")
    picked = rng.sample(pool, min(k, len(pool)))
    by_id = {s["id"]: s for s in samples}
    return [by_id[i] for i in sorted(picked)]


def retrieved_examples(samples: list[dict], sample: dict,
                       k: int = EXAMPLE_K) -> list[dict]:
    """Judge B's five: nearest by BM25 over the corpus, same exclusion rule."""
    return top_k(build_index(samples), samples, sample, k=k)


_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_verdict(reply) -> dict:
    """Take the last JSON object in the reply's answer channel.

    Accepts the `call_engine` dict or a bare string (a test may hand it text).
    `unparsed` is a real state and is scored as *unjudged*, never as an accept: a
    reply that never named a verdict must not be counted as a clean pass, because
    that is exactly the silent nod this whole item exists to measure.
    """
    if isinstance(reply, dict):
        text = reply.get("content") or reply.get("reasoning") or ""
    else:
        text = reply or ""
    found = _JSON_RE.findall(text)
    for candidate in reversed(found):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict in ("flag", "accept"):
                return {"verdict": verdict,
                        "defect_class": obj.get("defect_class"),
                        "reason": str(obj.get("reason", ""))[:400]}
    return {"verdict": "unparsed", "defect_class": None, "reason": (text or "")[:400]}


def call_engine(prompt_text: str, url: str = ENGINE_URL, model: str = ENGINE_MODEL,
                timeout: int = 240, max_tokens: int = 400,
                enable_thinking: bool = False) -> dict:
    """One judging call. Returns the reply with the channel it arrived on.

    `enable_thinking=False` is load-bearing and measured, not cosmetic: llama.cpp
    serves this Qwen build with its chain of thought in `reasoning_content`, and a
    first real pass left **35 of 50 replies truncated inside that chain of thought
    with no verdict at all** (`finish_reason: length`) — the same self-reporting
    failure this item is about, one level up. `chat_template_kwargs` is the knob
    this llama.cpp build answers (measured: a trivial prompt, 1.4 s with reasoning
    versus 0.1 s without, and no `reasoning_content`). It applies identically to
    both judges, so the A/B comparison is unaffected; what it changes is that a
    verdict arrives.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    choice = data["choices"][0]
    msg = choice["message"]
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason")}


def judge_one(judge_name: str, samples: list[dict], sample: dict,
              url: str, model: str) -> dict:
    examples = (random_examples(samples, sample) if judge_name == "a"
                else retrieved_examples(samples, sample))
    prompt = build_prompt(sample, examples)
    try:
        raw = call_engine(prompt.text, url=url, model=model)
    except Exception as exc:  # engine unreachable: an error row, never a silent accept
        return {"sample_id": sample["id"], "judge": judge_name, "verdict": "error",
                "defect_class": None, "reason": f"engine call failed: {exc}"[:300],
                "example_ids": [e["id"] for e in examples],
                "n_examples": prompt.example_count, "prompt_chars": len(prompt.text)}
    row = parse_verdict(raw)
    row.update({"sample_id": sample["id"], "judge": judge_name,
                "example_ids": [e["id"] for e in examples],
                "n_examples": prompt.example_count, "model": model,
                "prompt_chars": len(prompt.text),
                "finish_reason": raw.get("finish_reason")})
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(Path(__file__).parent / "corpus.jsonl"))
    ap.add_argument("--judge", choices=("a", "b", "both"), default="both")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    ap.add_argument("--limit", type=int, default=0, help="score only the first N samples")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--url", default=ENGINE_URL)
    ap.add_argument("--model", default=ENGINE_MODEL)
    ap.add_argument("--dry-run", action="store_true",
                    help="build every prompt and report sizes; call no model")
    args = ap.parse_args(argv)

    samples = [json.loads(ln) for ln in Path(args.corpus).read_text().splitlines()
               if ln.strip()]
    # `--limit` narrows what gets judged, never the pool examples are drawn from:
    # a smaller pool would change Judge B's retrieval, not just its workload.
    targets = samples[:args.limit] if args.limit else samples
    names = ("a", "b") if args.judge == "both" else (args.judge,)

    for name in names:
        out_path = Path(args.out_dir) / f"judge_raw_{name}.jsonl"
        rows: list[dict] = []
        if args.dry_run:
            for s in targets:
                examples = (random_examples(samples, s) if name == "a"
                            else retrieved_examples(samples, s))
                p = build_prompt(s, examples)
                assert p.example_count == EXAMPLE_K, (s["id"], p.example_count)
                rows.append({"sample_id": s["id"], "label": s["label"], "judge": name,
                             "verdict": "dry_run", "defect_class": None, "reason": "",
                             "n_examples": p.example_count,
                             "prompt_chars": len(p.text),
                             "example_ids": [e["id"] for e in examples]})
            print(f"[dry-run] judge {name}: {len(rows)} prompts, "
                  f"mean {sum(r['prompt_chars'] for r in rows) // max(len(rows), 1)} chars, "
                  f"examples per prompt {EXAMPLE_K}")
        else:
            with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = [pool.submit(judge_one, name, samples, s, args.url, args.model)
                           for s in targets]
                for fut in cf.as_completed(futures):
                    rows.append(fut.result())
            rows.sort(key=lambda r: r["sample_id"])
            unparsed = [r["sample_id"] for r in rows if r["verdict"] == "unparsed"]
            errored = [r["sample_id"] for r in rows if r["verdict"] == "error"]
            if unparsed:
                print(f"[warn] unparsed verdicts: {unparsed}")
            if errored:
                print(f"[warn] engine errors: {errored}")
        rows.sort(key=lambda r: r["sample_id"])
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"judge {name}: wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
