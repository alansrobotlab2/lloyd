#!/usr/bin/env python
"""Grade the conversation-mode addressee judge (agent-services/voice/addressee.py).

The gate asks djev "was this said to Lloyd?" only inside an open conversation
and past the short follow-up window, so the cases here are that situation:
Lloyd said something a while ago, the room is still open, somebody speaks. Each
case carries a label. What matters is precision — a false accept injects
somebody's phone call as a turn — so the report leads with the false accepts at
the configured threshold and sweeps the threshold beside it.

    python scripts/voice/addressee_eval.py            # live djev, config threshold
    python scripts/voice/addressee_eval.py --json out.json

Needs djev up (GPU 2). One question per case, ~40 ms each.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-services"))
sys.path.append(str(ROOT))

from voice.addressee import AddresseeClassifier  # noqa: E402

# (label, lloyd's last sentence, seconds since, utterance, mentions name)
CASES: list[tuple[bool, str, float, str, bool]] = [
    # ── addressed: follow-ups, answers, new requests with no name ──────────
    (True, "The nightly reflection chain is stalled because the handoff file is missing.", 20, "Why is the handoff file missing?", False),
    (True, "The build finished with two warnings.", 35, "What were the warnings?", False),
    (True, "It's sixty-eight degrees and sunny.", 40, "And what about tomorrow?", False),
    (True, "I've added that to your list.", 25, "Can you also add milk?", False),
    (True, "Your next meeting is at three with the design team.", 50, "Move it to four.", False),
    (True, "There are three open items on the backlog.", 30, "Which one is the oldest?", False),
    (True, "Want me to restart it?", 15, "Yes, go ahead.", False),
    (True, "Should I file that as a new item?", 20, "No, add it to the existing one.", False),
    (True, "The primary engine is healthy.", 45, "How much memory is it using?", False),
    (True, "I found two sessions from yesterday.", 30, "Open the second one.", False),
    (True, "The weather service is back up.", 60, "Set a timer for ten minutes.", False),
    (True, "Done, the reminder is set for seven.", 40, "Actually make that seven thirty.", False),
    (True, "The vault has four thousand notes.", 25, "How many of them are about robots?", False),
    (True, "I can't reach the printer right now.", 30, "Try again in a minute.", False),
    (True, "The guardian raised an alert forty minutes ago.", 20, "What did it say?", False),
    (True, "That task runs every night at two.", 55, "Can you run it now instead?", False),
    (True, "I've paused the worker pool.", 35, "Okay, now restart the backend.", False),
    (True, "Here's what I found about the kernel bug.", 45, "Read me the first part again.", False),
    (True, "The package arrives Thursday.", 50, "Remind me Thursday morning.", False),
    (True, "Sure.", 20, "What time is it in Tokyo?", False),
    (True, "The test suite passed.", 30, "Great, what's next on the list?", False),
    (True, "I'm not sure which file you mean.", 15, "The config file for the worker.", False),
    (True, "Lloyd here, ready when you are.", 40, "Tell me about today's calendar.", False),
    (True, "It took four minutes.", 25, "Why so slow?", False),
    # ── not addressed: other people, phones, self-talk, TV, acknowledgements ─
    (False, "The build finished with two warnings.", 35, "Honey, can you grab the milk from the car?", False),
    (False, "It's sixty-eight degrees and sunny.", 40, "Yeah I'll be there in like five minutes, save me a seat.", False),
    (False, "Your next meeting is at three.", 50, "Where did I put my keys?", False),
    (False, "I've added that to your list.", 30, "Did you feed the dog?", False),
    (False, "The primary engine is healthy.", 45, "Hi mom, yeah, we're good, how's dad?", False),
    (False, "There are three open items.", 60, "Tonight on the news, storms across the valley.", False),
    (False, "The test suite passed.", 30, "Ugh, this stupid cable.", False),
    (False, "I found two sessions from yesterday.", 25, "Kids, dinner's ready!", False),
    (False, "The weather service is back up.", 55, "No, I told him Tuesday, not Wednesday.", False),
    (False, "Done, the reminder is set.", 40, "Yeah.", False),
    (False, "The vault has four thousand notes.", 20, "Okay.", False),
    (False, "Here's what I found.", 30, "Mm-hmm.", False),
    (False, "That task runs every night.", 50, "Crazy.", False),
    (False, "I can't reach the printer.", 35, "Did Lloyd finish the report yet?", True),
    (False, "The package arrives Thursday.", 45, "I told Lloyd about it earlier.", True),
    (False, "It took four minutes.", 25, "Let me think about that for a sec.", False),
    (False, "The guardian raised an alert.", 30, "Can you pass me the remote?", False),
    (False, "Sure.", 20, "What are you guys watching?", False),
    (False, "I've paused the worker pool.", 40, "Hold on, someone's at the door.", False),
    (False, "Your next meeting is at three.", 55, "Sorry, I was talking to my computer. What did you say?", False),
    (False, "It's sunny.", 45, "Touchdown! Did you see that?", False),
    (False, "I've set the timer.", 30, "Thank you.", False),
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    import yaml
    cfg = (yaml.safe_load((ROOT / "config.yaml").read_text()) or {})
    conv = ((cfg.get("livekit") or {}).get("conversation") or {})
    judge = AddresseeClassifier({**conv, "addressee": "shadow",
                                 "addressee_timeout_ms": 5000})
    thr = args.threshold if args.threshold is not None else judge.threshold
    rows = []
    for label, last, since, utt, name in CASES:
        v = await judge.judge(utt, last, since, None, name)
        rows.append({"label": label, "utterance": utt,
                     "p": None if v is None else round(v.probability, 4),
                     "ms": None if v is None else round(v.latency_ms, 1),
                     "mass": None if v is None else round(v.label_mass, 3)})
    answered = [r for r in rows if r["p"] is not None]
    if not answered:
        print("djev answered nothing — is it up? (GPU 2, :8011)")
        return 2
    for r in rows:
        mark = "?" if r["p"] is None else ("+" if r["p"] >= thr else "-")
        ok = r["p"] is not None and ((r["p"] >= thr) == r["label"])
        print(f"{'ok ' if ok else 'BAD'} {mark} p={r['p']} label={'A' if r['label'] else 'n'}  {r['utterance']}")
    print()
    print(f"{'thr':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  precision recall")
    for t in sorted({0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, thr}):
        tp = sum(1 for r in answered if r["label"] and r["p"] >= t)
        fp = sum(1 for r in answered if not r["label"] and r["p"] >= t)
        fn = sum(1 for r in answered if r["label"] and r["p"] < t)
        tn = sum(1 for r in answered if not r["label"] and r["p"] < t)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        star = " <- config" if t == thr else ""
        print(f"{t:5.2f} {tp:3d} {fp:3d} {fn:3d} {tn:3d}  {prec:9.3f} {rec:6.3f}{star}")
    ms = sorted(r["ms"] for r in answered)
    print(f"\nanswered {len(answered)}/{len(rows)}, latency p50 {ms[len(ms)//2]} ms, max {ms[-1]} ms")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
