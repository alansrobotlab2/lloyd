#!/usr/bin/env python3
"""Replay past triage decisions through the CURRENT triage prompt.

`cases.json` (built by `build.py`) holds single-triage decisions whose outcome
is known from what happened afterwards. This runner renders each case through
`workers.sources.autotriage.render_prompt` — the production single-item prompt,
unchanged — sends it to a model, parses the answer with the production
`parse_verdict`, and reports how often the verdict agrees with the outcome
label.

**It is a plain completion with no tools, on purpose.** A tooled replay against
today's tree cannot work for this set: a `good_confirm` case is by definition
one whose fix has since landed, so a triage turn that greps the live tree would
correctly answer `already_done` for every one of them. Running the checks
honestly needs the tree and the vault as they were — each case carries
`tree_as_of` and `vault_sha` for whoever builds that — and until then the
replay asks the narrower question the text alone can answer: shown the item as
it read at triage time, does the current prompt's judgment still come out
`confirmed` for the work that landed and stayed landed? The prompt gets one
appended paragraph (`REPLAY_NOTE`) saying so, because a triage prompt sent with
no `tools` array otherwise narrates tool calls as prose.

Nothing here starts an agent turn, opens a session or touches the backend: the
model is a callable `prompt -> text`. `endpoint_model` is the one that talks to
an engine, over `/v1/chat/completions`, and only when a `--base-url` is given.

    python eval/triage_replay/run.py plan                       # sizes, no model
    python eval/triage_replay/run.py run --base-url http://127.0.0.1:8096
    python eval/triage_replay/run.py grade                      # re-grade recorded rows

Rows are appended to eval/measurements/triage-replay/rows.jsonl, one per case,
so a re-grade never needs the engine.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

LLOYD_HOME = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LLOYD_HOME))

CASES_PATH = Path(__file__).resolve().parent / "cases.json"
ROWS_PATH = LLOYD_HOME / "eval" / "measurements" / "triage-replay" / "rows.jsonl"

#: Below this many graded cases a label's agreement is `unmeasured`.
MIN_N = 20

EXPECTED = ("confirmed", "not_confirmed")

REPLAY_NOTE = """

REPLAY — read this before anything else. This is a replay of a past triage, as \
a single completion: **no tools are available**, and the tree has moved since \
the item was written, so nothing you could run today would describe the system \
the item describes. Do not write tool calls and do not narrate checks you did \
not run. Skip steps 3, 4 and 6. Judge the item from its own text, as of the day \
it was triaged: is the premise a specific, checkable claim about a real \
problem in this system that a round should fix (`confirmed`), or not (the \
verdict that fits)? For CHECK write `replay: text only`. Finish with the \
VERDICT block exactly as specified above."""


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = doc["cases"] if isinstance(doc, dict) else doc
    for c in cases:
        if c.get("expected") not in EXPECTED:
            raise ValueError(f"case {c.get('key')!r}: expected must be one of {EXPECTED}")
        if not str((c.get("item") or {}).get("body") or "").strip():
            raise ValueError(f"case {c.get('key')!r} has no item text")
    return cases


def candidate_for(case: dict) -> SimpleNamespace:
    """The item as triage saw it: a draft, at the age it had then. `parent` is
    dropped — the prompt would look the parent up on the LIVE board."""
    item = case["item"]
    return SimpleNamespace(
        id=int(case["item_id"]), name=item["name"], status="draft",
        priority=item.get("priority") or "low", created=item.get("created") or "",
        body=item["body"], board="lloyd", tags=list(item.get("tags") or []), parent=None,
        group=None, members=[], worth=item.get("worth") or "", size=item.get("size") or "",
        clause_count=0, age_days=int(case.get("age_days") or 0),
        path=Path(f"/nonexistent/{case['item_id']}.md"))


def build_prompt(case: dict, *, ledger: Path) -> str:
    """The production prompt for this case plus `REPLAY_NOTE`. `ledger` must be
    an empty file: the origin block reads it, and the live ledger knows how
    every one of these items ended."""
    from workers.sources import autotriage
    return autotriage.render_prompt(candidate_for(case), ledger=ledger) + REPLAY_NOTE


def grade(case: dict, text: str) -> dict:
    """`{verdict, got, agree}`. No parseable verdict is `no_verdict`, which
    agrees with nothing — a model that answers nothing has not agreed."""
    from workers.sources import autotriage
    parsed = autotriage.parse_verdict(text or "")
    verdict = (parsed or {}).get("verdict") or ""
    got = "no_verdict" if not verdict else ("confirmed" if verdict == "confirmed"
                                           else "not_confirmed")
    return {"verdict": verdict or None, "got": got, "agree": got == case["expected"]}


def replay(cases: list[dict], model: Callable[[str], str], *,
           on_row: Callable[[dict], None] | None = None) -> list[dict]:
    """Run every case through `model`. A model that raises is recorded as an
    `error` row and the replay goes on."""
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="triage-replay-") as tmp:
        ledger = Path(tmp) / "empty-ledger.jsonl"
        ledger.write_text("", encoding="utf-8")
        for case in cases:
            prompt = build_prompt(case, ledger=ledger)
            started = time.monotonic()
            try:
                text, error = model(prompt), ""
            except Exception as e:  # noqa: BLE001
                text, error = "", f"{type(e).__name__}: {e}"[:300]
            row = {"key": case["key"], "item_id": case["item_id"], "label": case["label"],
                   "expected": case["expected"], "prompt_chars": len(prompt),
                   "seconds": round(time.monotonic() - started, 2), "error": error,
                   "text_tail": (text or "")[-2500:], **grade(case, text)}
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows


def _wilson(k: int, n: int) -> tuple[float, float]:
    from eval.stats import wilson_ci
    lo, hi = wilson_ci(k, n)
    return (round(lo, 4), round(hi, 4))


def summarize(rows: list[dict], *, min_n: int = MIN_N) -> dict[str, Any]:
    """Agreement overall and per label, latest row per case. A label under
    `min_n` carries its counts and `rate: None`."""
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["key"]] = r
    graded = [r for r in latest.values() if not r.get("error")]

    def cell(rs: list[dict]) -> dict:
        k, n = sum(1 for r in rs if r["agree"]), len(rs)
        measured = n >= min_n
        return {"agree": k, "n": n, "rate": round(k / n, 4) if measured else None,
                "ci": _wilson(k, n) if measured else None,
                "verdicts": _count(r.get("verdict") or "no_verdict" for r in rs)}

    labels = sorted({r["label"] for r in graded})
    return {"cases": len(latest), "errors": len(latest) - len(graded), "min_n": min_n,
            "overall": cell(graded),
            "by_label": {name: cell([r for r in graded if r["label"] == name]) for name in labels}}


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def render(summary: dict) -> str:
    def line(name: str, c: dict) -> str:
        rate = (f"unmeasured (n < {summary['min_n']})" if c["rate"] is None
                else f"{c['rate'] * 100:.0f}% [{c['ci'][0] * 100:.0f}–{c['ci'][1] * 100:.0f}]")
        return f"| {name} | {c['agree']}/{c['n']} | {rate} | {c['verdicts']} |"
    out = [f"Triage replay — {summary['cases']} cases, {summary['errors']} errored", "",
           "| label | agree | agreement | verdicts |", "|---|---|---|---|"]
    out += [line(name, c) for name, c in summary["by_label"].items()]
    out.append(line("all", summary["overall"]))
    return "\n".join(out)


#: The budget a reasoning replay needs to reach its verdict (measured: ~8,000).
THINKING_MAX_TOKENS = 16000


def endpoint_model(base_url: str, *, model: str = "primary", max_tokens: int = 3000,
                   timeout: float = 600.0, thinking: bool = False) -> Callable[[str], str]:
    """A `prompt -> text` over `/v1/chat/completions`: one user message, no
    `tools`, temperature 0, and reasoning OFF unless `thinking` is set.

    Reasoning is off because the first recorded run (2026-11-22) was lost to it:
    with reasoning on, 36 of 42 cases spent the whole 3,000-token budget before
    writing a word of content and were scored `no_verdict` (about 21 s each).
    Probed on one case: reasoning on needs about 8,000 tokens and 60 s to reach a
    verdict; reasoning off answers in 489 tokens and 4 s. A replay is already a
    text-only reading of a turn that had tools, so the cheap, repeatable form is
    the one a gate rung can afford. `--thinking` restores the production shape
    and raises the budget to `THINKING_MAX_TOKENS`; rows from the two modes are
    not comparable with each other."""
    if thinking:
        max_tokens = max(max_tokens, THINKING_MAX_TOKENS)
    extra = {} if thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    url = base_url.rstrip("/") + "/v1/chat/completions"

    def call(prompt: str) -> str:
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}], **extra}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"].get("content") or ""
    return call


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay past triage decisions through the current prompt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "grade"):
        p = sub.add_parser(name)
        p.add_argument("--cases", type=Path, default=CASES_PATH)
        p.add_argument("--rows", type=Path, default=ROWS_PATH)
        if name != "grade":
            p.add_argument("--only", default="", help="one case key")
            p.add_argument("--limit", type=int, default=0)
        if name == "run":
            p.add_argument("--base-url", required=True,
                           help="the engine, e.g. http://127.0.0.1:8096 — there is no default")
            p.add_argument("--model", default="primary")
            p.add_argument("--max-tokens", type=int, default=3000)
            p.add_argument("--thinking", action="store_true",
                           help="reason before answering (about 60 s a case; not comparable with the default)")
    args = ap.parse_args(argv)

    if args.cmd == "grade":
        cases = {c["key"]: c for c in load_cases(args.cases)}
        rows = [{**r, **grade(cases[r["key"]], r.get("text_tail", ""))}
                for r in _read_rows(args.rows) if r.get("key") in cases]
        print(render(summarize(rows)))
        return 0

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c["key"] == args.only]
    if args.limit:
        cases = cases[:args.limit]
    if args.cmd == "plan":
        sizes: list[int] = []
        replay(cases, lambda prompt: sizes.append(len(prompt)) or "")
        tokens = sum(sizes) / 3.6
        print(f"{len(cases)} cases; prompt chars median {sorted(sizes)[len(sizes) // 2]}, "
              f"max {max(sizes)}, total {sum(sizes)} (≈{tokens:,.0f} prompt tokens at 3.6 "
              f"chars/token); labels {_count(c['label'] for c in cases)}")
        return 0

    args.rows.parent.mkdir(parents=True, exist_ok=True)
    model = endpoint_model(args.base_url, model=args.model, max_tokens=args.max_tokens,
                           thinking=getattr(args, "thinking", False))

    def record(row: dict) -> None:
        with args.rows.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**row, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                 "model": args.model}) + "\n")
        print(f"{row['key']}: {row['verdict'] or row['error'] or 'no verdict'} "
              f"({'agree' if row['agree'] else 'DISAGREE'}, {row['seconds']} s)", flush=True)

    rows = replay(cases, model, on_row=record)
    print()
    print(render(summarize(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
