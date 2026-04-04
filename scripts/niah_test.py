#!/usr/bin/env python3
"""
Needle-in-a-Haystack (NIAH) quality test for TurboQuant KV cache.

Tests whether the model can retrieve a specific fact ("needle") hidden at
varying depths inside long context windows ("haystack").

Usage:
  # Baseline (current fp8_e4m3 endpoint):
  python scripts/niah_test.py --url http://127.0.0.1:8096

  # After restarting with --kv-cache-dtype turboquant_4bit:
  python scripts/niah_test.py --url http://127.0.0.1:8096 --tag turboquant_4bit

  # Quick single test:
  python scripts/niah_test.py --context-len 8192 --depth 50 --verbose
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Haystack generation
# ---------------------------------------------------------------------------

FILLER_SENTENCES = [
    "The development of artificial intelligence has been marked by several key breakthroughs over the past decades.",
    "Modern neural networks consist of multiple layers that transform input data through learned weight matrices.",
    "The attention mechanism allows transformers to weigh the relevance of different parts of the input sequence.",
    "Gradient descent optimization iteratively updates model parameters to minimise the training loss.",
    "Large language models have demonstrated emergent capabilities that were not explicitly trained.",
    "Distributed computing systems enable training on datasets too large to fit on a single machine.",
    "Tokenisation converts raw text into discrete tokens that neural networks can process efficiently.",
    "The softmax function normalises a vector of real numbers into a probability distribution.",
    "Regularisation techniques like dropout prevent overfitting by randomly zeroing activations during training.",
    "Benchmark datasets provide standardised evaluation criteria to compare different model architectures.",
    "Transfer learning allows models pre-trained on large corpora to be fine-tuned on smaller task-specific datasets.",
    "The transformer architecture replaced recurrent networks as the dominant paradigm for sequence modelling.",
    "Positional encodings inject information about token order into the otherwise position-invariant attention mechanism.",
    "Beam search explores multiple candidate sequences simultaneously during autoregressive text generation.",
    "Temperature scaling controls the sharpness of the output probability distribution at inference time.",
    "Flash attention reduces memory usage by computing attention in tiled blocks without materialising the full matrix.",
    "Mixture-of-experts models route each token to a subset of specialist feed-forward networks.",
    "Quantisation reduces model size by representing weights and activations in lower-precision formats.",
    "Speculative decoding accelerates autoregressive inference by drafting multiple tokens in parallel.",
    "KV cache stores key and value projections across decoder layers to avoid redundant recomputation.",
]

NEEDLE_TEMPLATES = [
    "The secret passcode for the underground vault is {value}.",
    "According to the confidential memo dated 14 March, the activation word is {value}.",
    "Professor Hartley's hidden research note states: the critical constant equals {value}.",
    "The answer to the mystery question posed by the committee is {value}.",
    "The unique identifier assigned to project Nighthawk is {value}.",
]

QUESTION_TEMPLATES = {
    0: "What is the secret passcode for the underground vault?",
    1: "What is the activation word mentioned in the confidential memo dated 14 March?",
    2: "What critical constant did Professor Hartley record in his hidden research note?",
    3: "What is the answer to the mystery question posed by the committee?",
    4: "What unique identifier was assigned to project Nighthawk?",
}


def generate_needle_value() -> str:
    """Generate a distinctive, hard-to-guess value for the needle."""
    adjectives = ["crimson", "phantom", "cobalt", "verdant", "silver"]
    nouns = ["aurora", "cascade", "vortex", "nexus", "labyrinth"]
    numbers = [str(random.randint(1000, 9999))]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{random.choice(numbers[0])}"


def build_haystack(
    context_len_tokens: int,
    depth_pct: float,  # 0.0 = beginning, 1.0 = end
    needle_template_idx: int = 0,
    seed: int = 42,
) -> tuple[str, str, str]:
    """
    Build a haystack string with the needle inserted at the given depth.

    Returns (haystack_text, needle_value, question)
    """
    rng = random.Random(seed)
    needle_value = generate_needle_value()

    # Rough chars-per-token estimate for English
    chars_per_token = 4.2
    target_chars = int(context_len_tokens * chars_per_token)

    # Build filler pool
    filler_pool = FILLER_SENTENCES * ((target_chars // (sum(len(s) for s in FILLER_SENTENCES))) + 2)
    rng.shuffle(filler_pool)
    filler_text = " ".join(filler_pool)[:target_chars]

    # Build needle sentence
    needle = NEEDLE_TEMPLATES[needle_template_idx].format(value=needle_value)
    question = QUESTION_TEMPLATES[needle_template_idx]

    # Insert needle at depth
    insert_pos = max(0, min(int(len(filler_text) * depth_pct), len(filler_text) - len(needle) - 1))
    # Find a sentence boundary near insert_pos
    for offset in range(0, 200, 1):
        candidate = insert_pos + offset
        if candidate < len(filler_text) and filler_text[candidate] == ".":
            insert_pos = candidate + 1
            break

    haystack = filler_text[:insert_pos] + " " + needle + " " + filler_text[insert_pos:]
    return haystack, needle_value, question


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def call_model(
    url: str,
    model_name: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = 120,
) -> Optional[str]:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Disable Qwen3 thinking mode — otherwise content=null until </think>
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"].get("content")
        if content is None:
            # Reasoning model used all budget on thinking — increase max_tokens
            reasoning = data["choices"][0]["message"].get("reasoning", "")
            print(f"  WARN: content=null (reasoning={len(reasoning)} chars)", file=sys.stderr)
            return None
        return content.strip()
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None


def get_model_name(url: str) -> str:
    try:
        r = requests.get(f"{url}/v1/models", timeout=10)
        return r.json()["data"][0]["id"]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# NIAH test runner
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using only the information "
    "provided in the context. Give ONLY the exact answer value, nothing else."
)


def run_niah_test(
    url: str,
    context_len: int,
    depth_pct: float,
    needle_template_idx: int = 0,
    seed: int = 42,
    verbose: bool = False,
    timeout: int = 180,
) -> dict:
    model_name = get_model_name(url)

    haystack, needle_value, question = build_haystack(
        context_len_tokens=context_len,
        depth_pct=depth_pct,
        needle_template_idx=needle_template_idx,
        seed=seed,
    )

    user_prompt = f"Context:\n{haystack}\n\nQuestion: {question}"

    if verbose:
        print(f"  Needle: {needle_value!r}")
        print(f"  Context len: ~{context_len} tokens, depth: {depth_pct*100:.0f}%")

    t0 = time.time()
    response = call_model(
        url=url,
        model_name=model_name,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=32,
        temperature=0.0,
        timeout=timeout,
    )
    elapsed = time.time() - t0

    if response is None:
        match = False
        exact = False
    else:
        response_lower = response.lower().strip()
        needle_lower = needle_value.lower().strip()
        exact = response_lower == needle_lower
        match = needle_lower in response_lower

    result = {
        "context_len": context_len,
        "depth_pct": depth_pct,
        "needle_value": needle_value,
        "response": response,
        "exact_match": exact,
        "contains_needle": match,
        "elapsed_s": round(elapsed, 2),
        "model": model_name,
    }

    if verbose:
        status = "✓ EXACT" if exact else ("~ CONTAINS" if match else "✗ MISS")
        print(f"  Response: {response!r}")
        print(f"  Result: {status}  ({elapsed:.1f}s)")

    return result


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_LENS = [4096, 8192, 16384, 32768]
DEFAULT_DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]


def run_matrix(
    url: str,
    context_lens: list[int],
    depths: list[float],
    tag: str = "baseline",
    verbose: bool = False,
    output_file: Optional[str] = None,
) -> list[dict]:
    results = []
    total = len(context_lens) * len(depths)
    idx = 0

    print(f"\n{'='*60}")
    print(f"NIAH Test — {tag}")
    print(f"  URL:    {url}")
    print(f"  Tests:  {total} ({len(context_lens)} context lengths × {len(depths)} depths)")
    print(f"  Start:  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    for ctx_len in context_lens:
        for depth in depths:
            idx += 1
            print(f"\n[{idx}/{total}] ctx={ctx_len} depth={depth*100:.0f}%")
            r = run_niah_test(
                url=url,
                context_len=ctx_len,
                depth_pct=depth,
                verbose=verbose,
            )
            r["tag"] = tag
            results.append(r)

            status = "✓" if r["contains_needle"] else "✗"
            print(f"  {status}  response={r['response']!r}  ({r['elapsed_s']}s)")

    # Summary
    total_tests = len(results)
    hits = sum(1 for r in results if r["contains_needle"])
    exact = sum(1 for r in results if r["exact_match"])
    print(f"\n{'='*60}")
    print(f"SUMMARY — {tag}")
    print(f"  Contains needle: {hits}/{total_tests} ({100*hits/total_tests:.1f}%)")
    print(f"  Exact match:     {exact}/{total_tests} ({100*exact/total_tests:.1f}%)")

    # Per-context-length breakdown
    print("\n  Context length breakdown:")
    for ctx_len in context_lens:
        subset = [r for r in results if r["context_len"] == ctx_len]
        n = len(subset)
        h = sum(1 for r in subset if r["contains_needle"])
        print(f"    {ctx_len:>6} tokens: {h}/{n} ({100*h/n:.0f}%)")

    print(f"{'='*60}\n")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIAH quality test for TurboQuant KV cache")
    parser.add_argument("--url", default="http://127.0.0.1:8096",
                        help="vLLM OpenAI-compatible endpoint URL")
    parser.add_argument("--tag", default="baseline",
                        help="Label for this run (e.g. fp8_e4m3, turboquant_4bit)")
    parser.add_argument("--context-len", type=int, nargs="+",
                        default=DEFAULT_CONTEXT_LENS,
                        help="Context lengths to test (tokens)")
    parser.add_argument("--depth", type=float, nargs="+",
                        default=DEFAULT_DEPTHS,
                        help="Needle depths as fractions (0.0-1.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full prompts and responses")
    parser.add_argument("--single", action="store_true",
                        help="Run a single quick test (8k context, 50pct depth)")
    args = parser.parse_args()

    if args.single:
        print("Running single quick test...")
        r = run_niah_test(
            url=args.url,
            context_len=8192,
            depth_pct=0.5,
            verbose=True,
        )
        status = "PASS" if r["contains_needle"] else "FAIL"
        print(f"\n{status}: needle={r['needle_value']!r}  response={r['response']!r}")
        sys.exit(0 if r["contains_needle"] else 1)

    results = run_matrix(
        url=args.url,
        context_lens=args.context_len,
        depths=args.depth,
        tag=args.tag,
        verbose=args.verbose,
        output_file=args.output or f"niah_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    # Exit code: 0 if ≥90% pass, 1 otherwise
    hits = sum(1 for r in results if r["contains_needle"])
    sys.exit(0 if hits / len(results) >= 0.9 else 1)


if __name__ == "__main__":
    main()
