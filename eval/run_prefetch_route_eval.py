#!/usr/bin/env python3
"""#1483 — can one djev read per turn gate or route the subliminal `<context>`?

For every labelled turn the script asks djev one `choice` question (the frozen
`QUESTION` below, negative first) and records the probability of each store,
the label mass and the round-trip latency. It then prices a SKIP gate —
P(none) >= threshold means nothing search-derived is injected — against:

- ``gold``        — `vault_recall_queries.yaml` rows with `expect_docs`: every
                    one needs the vault, so any skip is a lost document;
- ``toolchoice``  — `tool_choice_queries.yaml`, the prompts the tool-choice eval
                    runs: a skip changes that eval's prompts only if one fires;
- ``skill_pos``   — `skill_match_queries.yaml` turns (verbatim user text) with an
                    expected skill: a skip loses that skill;
- ``skill_empty`` — the same file's expected-empty turns: a skip here is the win;
- ``trivial``     — ten hand-written acknowledgements, the case the gate is for.

and the characters it would save, from the block `prefetch_context` renders for
each turn (`split_injected`). The store-routing half is judged by whether the
argmax label names the store the gold row's documents live in.

djev is queried, never loaded or restarted. Its latency depends on who else is
reading it (MAX_SEQS=1): run it with nothing else on djev and report the p50.

Usage:
    .venvs/lloyd/bin/python eval/run_prefetch_route_eval.py --out ~/lloyd-data/eval/1483/route.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from run_prefetch_eval import _quiet_logging  # noqa: E402

QUESTION = {
    "route": {
        "type": "choice",
        "instructions": (
            "A message a user just sent to their personal AI assistant is shown. "
            "What stored context would the assistant need to answer it well?"),
        "criteria": {
            "none": "nothing stored: it is small talk, an acknowledgement, or fully specified in the message",
            "web": "fresh information from the public internet",
            "vault": "notes, documents or design docs from the user's own vault",
            "facts": "stored facts about a named person, project or system",
            "sessions": "an earlier conversation with the user",
            "skill": "a written procedure for a task the assistant repeats",
        },
    },
}
TRIVIAL = ["ok thanks", "sounds good, go ahead", "hi lloyd", "yes please do that",
           "great work", "thank you so much", "cool", "lol nice", "good morning!",
           "no, that's fine"]
THRESHOLDS = (0.5, 0.8, 0.9, 0.99)


def ask(text: str, timeout: float) -> dict | None:
    from app import djev
    t0 = time.perf_counter()
    out = djev.ask_sync(f"User message:\n{text[:1200]}", QUESTION, timeout=timeout,
                        seam="eval:prefetch_route")
    ms = (time.perf_counter() - t0) * 1000
    a = out.get("route") if out is not None else None
    if a is None:
        return None
    return {"p": {k: float(v) for k, v in a.probabilities.items()}, "label": a.label,
            "label_mass": round(float(a.label_mass), 4), "ms": round(ms, 1)}


def sets() -> dict[str, list[str]]:
    g = yaml.safe_load((HERE / "vault_recall_queries.yaml").read_text())["queries"]
    t = yaml.safe_load((HERE / "tool_choice_queries.yaml").read_text())
    t = t.get("queries") if isinstance(t, dict) else t
    s = yaml.safe_load((HERE / "skill_match_queries.yaml").read_text())
    recs = s.get("records") or s.get("queries") or []
    return {"gold": [q["query"] for q in g if q.get("expect_docs")],
            "toolchoice": [q["prompt"] for q in t],
            "skill_pos": [r["turn"] for r in recs if r.get("expected_skills")],
            "skill_empty": [r["turn"] for r in recs if not r.get("expected_skills")],
            "trivial": TRIVIAL}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--no-chars", action="store_true", help="skip rendering prefetch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with _quiet_logging():
        import prefetch
        out: dict = {}
        for name, turns in sets().items():
            rows = []
            for q in turns:
                r = ask(q, args.timeout)
                chars = None
                if not args.no_chars:
                    rendered = prefetch.prefetch_context(q, session_id=None, plan_mode=False)
                    chars = prefetch.split_injected(rendered, q)["injected_chars"]
                rows.append({"q": q[:200], "r": r, "chars": chars})
            ok = [x for x in rows if x["r"]]
            ms = sorted(x["r"]["ms"] for x in ok)
            summ = {"n": len(rows), "answered": len(ok),
                    "ms_p50": round(statistics.median(ms), 1) if ms else None,
                    "ms_p90": ms[int(0.9 * (len(ms) - 1))] if ms else None,
                    "labels": {}}
            for x in ok:
                summ["labels"][x["r"]["label"]] = summ["labels"].get(x["r"]["label"], 0) + 1
            total = sum(x["chars"] or 0 for x in rows)
            summ["chars_total"] = total
            for thr in THRESHOLDS:
                sk = [x for x in ok if x["r"]["p"].get("none", 0.0) >= thr]
                summ[f"skip@{thr}"] = {"k": len(sk), "chars_saved": sum(x["chars"] or 0 for x in sk)}
            out[name] = {"summary": summ, "rows": rows}
            print(name, json.dumps(summ))
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).expanduser().write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
