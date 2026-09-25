#!/usr/bin/env python3
"""Replay a fixed, taskless turn N times at an engine and count fabricated traces (#1510).

    # the primary, 8 identical requests
    .venvs/lloyd/bin/python -m scripts.thinking_replay_probe --turns 8

    # a sampling A/B, no engine restart
    ... --turns 8 --temperature 0
    ... --turns 8 --extra-body '{"top_k": 1}'

    # the secondary slot, as a control on a different engine
    ... --base-url http://127.0.0.1:8091 --model secondary

Every request is the SAME prompt: one `role="system"` message and nothing else,
which is the degenerate shape the flagged traces describe ("the last message is
just the system instructions block… There's no actual task"). The number it
prints is how many of those N responses came back with reasoning that
`app.thinking_fidelity` flags, over N — the denominator is printed even when the
count is 0.

What it settles, and what it does not. The item's two mechanism candidates are
engine-side state (MTP speculative decoding, `MTP_ENABLED=0` behind an
`agent-llm-primary` restart) and a decode-side prior fired on a taskless turn.
This probe cannot separate those on its own — it is the half that needs no
restart, and it makes the restart informative: run it, record the rate, then run
it again after the operator restarts the primary with MTP off. A rate that
collapses to 0 points at engine state; a rate that survives points at the prior,
and the sampling flags above are then the cheap lever to test the same
distinction without touching the engine at all. N identical requests is also the
only way to see the shape at all: a single request that comes back clean says
nothing, because the flagged traces are a small minority of what the engine
returns, and in every session read so far the answer that followed was correct.
`python -m scripts.thinking_fidelity_scan` prints the current prevalence with its
denominator, and that is the figure to compare a probe run against — this file
deliberately does not carry one.

The request goes out through `app.harness.client.stream_chat`, the same send
site every agent-loop iteration uses, so what this measures is our parse path
plus the engine, not a second implementation of either.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.harness.client import stream_chat  # noqa: E402
from app.thinking_fidelity import flag_fabricated_reasoning, matched_marker  # noqa: E402

#: The fixed prompt. Deliberately taskless and deliberately identical on every
#: request: any variation would confound the one question the probe asks, which
#: is whether the SAME input sometimes yields a fabricated trace.
FIXED_SYSTEM_PROMPT = (
    "You are an expert software engineer. Helps user to solve problems."
)


@dataclass
class ProbeResult:
    """One probe run: N identical requests, and how many came back flagged."""
    base_url: str
    model: str
    sent: int = 0
    answered: int = 0
    flagged: int = 0
    reasoning_chars: int = 0
    markers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def report(self) -> str:
        """Flagged count over the denominator, printed whether or not it is 0."""
        return (
            f"thinking-replay probe: {self.flagged} of {self.sent} responses "
            f"carried a flagged reasoning block "
            f"(denominator: {self.sent} identical requests sent to "
            f"{self.base_url.rstrip('/')}/v1/chat/completions as model "
            f"{self.model!r}; {self.answered} answered, "
            f"{self.reasoning_chars} reasoning chars total)"
        )


def collect_reasoning(chunks: list[dict[str, Any]]) -> str:
    """Concatenate the reasoning deltas of one streamed response.

    Both field names, because the two engines behind this harness disagree
    about which is real — the same tolerance `app/harness/loop.py` applies when
    it reads its own stream (`reasoning_content` through vLLM ~0.22, `reasoning`
    from 0.23 on, and llama.cpp's template only ever reads the first).
    """
    parts: list[str] = []
    for chunk in chunks:
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(piece, str):
                parts.append(piece)
    return "".join(parts)


async def probe(
    *, base_url: str, model: str, turns: int = 8,
    user_text: str = "", temperature: float | None = None,
    extra_body: dict[str, Any] | None = None, timeout_s: float = 120.0,
) -> ProbeResult:
    """Fire `turns` copies of the fixed prompt and score each response's reasoning."""
    messages: list[dict[str, Any]] = [{"role": "system",
                                       "content": FIXED_SYSTEM_PROMPT}]
    if user_text:
        messages.append({"role": "user", "content": user_text})
    body = dict(extra_body or {})
    if temperature is not None:
        body["temperature"] = temperature

    out = ProbeResult(base_url=base_url, model=model)
    for _ in range(turns):
        out.sent += 1
        chunks: list[dict[str, Any]] = []
        try:
            async for chunk in stream_chat(
                base_url=base_url, model=model, messages=messages,
                tools=None, extra_body=body, cancel_event=None,
                timeout_s=timeout_s, session_id="thinking-replay-probe",
            ):
                chunks.append(chunk)
        except Exception as exc:                       # noqa: BLE001
            out.errors.append(f"{type(exc).__name__}: {exc}")
            continue
        out.answered += 1
        reasoning = collect_reasoning(chunks)
        out.reasoning_chars += len(reasoning)
        if flag_fabricated_reasoning(reasoning):
            out.flagged += 1
            out.markers.append(matched_marker(reasoning))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="",
                    help="engine base URL (default: config models.primary)")
    ap.add_argument("--model", default="primary")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--user-text", default="",
                    help="also send this user turn, for the "
                         "system-only-versus-taskful A/B")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--extra-body", default="",
                    help="JSON object merged into the request body")
    args = ap.parse_args(argv)

    base_url = args.base_url
    if not base_url:
        from app.config import CONFIG
        base_url = ((CONFIG.get("models") or {}).get(args.model) or {}) \
            .get("base_url", "")
    if not base_url:
        print(f"thinking-replay probe: no base URL for model {args.model!r} "
              "and none passed with --base-url — no request was sent, which is "
              "NOT a clean verdict", file=sys.stderr)
        return 2

    extra = json.loads(args.extra_body) if args.extra_body else None
    result = asyncio.run(probe(
        base_url=base_url, model=args.model, turns=args.turns,
        user_text=args.user_text, temperature=args.temperature,
        extra_body=extra,
    ))
    print(result.report())
    for marker in result.markers:
        print(f"  flagged reasoning matched: {marker!r}")
    for err in result.errors:
        # A failed request is a diagnostic, not a measurement: it goes to stderr
        # so a caller piping the report cannot read "0 of 2 flagged" as a rate
        # over requests that never got a reply.
        print(f"  request failed: {err}", file=sys.stderr)
    if result.answered == 0:
        print("  no response arrived, so the flag rate above is 0 over a "
              "denominator of requests that were never answered",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
