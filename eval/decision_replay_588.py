#!/usr/bin/env python3
"""#588 clause 2: does triage re-propose a decision that was already retired?

Replays the eight naive items in `decision_replay_588_cases.yaml` through the
production single-item triage prompt (`workers.sources.autotriage.render_prompt`,
the worker system prompt, the worker tool surface, the structured finalizer) on
the primary, with no decision ledger wired, and counts how many come back
`confirmed` without naming the retirement. That count is the size of the
defect #588 proposes to fix; 0-1 of 8 closes the item as unfounded.

Every turn runs under a `bench_` session id, which the aggregator sandboxes
(`agent_mcp/_tool_sandbox.py`): Bash in a read-only bubblewrap, every
non-read-only tool refused. Triage's step 6 asks for `backlog_write_task`; here
that call is refused and recorded, which is the point — these prompts must not
file anything. The runner refuses to start unless the aggregator reports the
sandbox enforced.

    python eval/decision_replay_588.py run   [--only KEY] [--max-turns N] [--rep R]
    python eval/decision_replay_588.py grade  # re-grade the recorded rows

Rows land in eval/measurements/588-decision-replay/rows.jsonl (one per turn,
appended), so a re-grade never needs the engine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LLOYD_HOME = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LLOYD_HOME))

CASES_PATH = LLOYD_HOME / "eval" / "decision_replay_588_cases.yaml"
OUT_DIR = LLOYD_HOME / "eval" / "measurements" / "588-decision-replay"
ROWS_PATH = OUT_DIR / "rows.jsonl"
#: A human read of a transcript that disagrees with the automatic grade:
#: `{"<key>#<rep>": {"grade": ..., "reason": ...}}`. The reason is required —
#: an override with no written reason is not a grade.
OVERRIDES_PATH = OUT_DIR / "overrides.yaml"

#: Item ids the replay renders under. Above the live board's max id and
#: recognisably synthetic; no file exists for them, which the triage turn may
#: notice — the body is inline in the prompt, as it is in production.
SYNTHETIC_ID_BASE = 95880

#: Same wording `autotriage.execute` sends.
FINAL_SCHEMA_PROMPT = (
    "Restate the verdict block above as a single JSON object "
    "matching the schema. Same verdict, same acceptance check, "
    "same filed ids — this is a transcription, not a re-decision.")


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    import yaml
    return list((yaml.safe_load(path.read_text()) or {}).get("cases") or [])


def new_session_id(key: str) -> str:
    """A `bench_` id: sandboxed by prefix in `_tool_sandbox`."""
    return f"bench_588_{key.replace('-', '_')}_{uuid.uuid4().hex[:8]}"


def require_sandboxed(session_id: str) -> str:
    from agent_mcp._tool_sandbox import is_sandboxed_session
    if not is_sandboxed_session(session_id):
        raise RuntimeError(f"refusing to replay under unsandboxed id {session_id!r}")
    return session_id


def candidate_for(case: dict, index: int):
    """A backlog `Item` for the naive proposal, created yesterday, untagged
    (a human's item: no loop-spawn origin to hint at anything)."""
    from scripts.automod.backlog import Item
    created = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    return Item(path=Path(f"/nonexistent/{SYNTHETIC_ID_BASE + index}.md"),
                id=SYNTHETIC_ID_BASE + index, name=case["title"], status="draft",
                priority="medium", created=created, body=case["body"].strip(),
                board="lloyd", tags=[])


# ── grading ─────────────────────────────────────────────────────────────────

RE_PROPOSED, CAUGHT, NO_VERDICT = "re-proposed", "caught", "no-verdict"


def cites_retirement(case: dict, text: str) -> list[str]:
    """The case's retirement markers found in `text` (case-insensitive)."""
    return [m for m in case.get("retirement_markers") or []
            if re.search(m, text or "", re.I)]


def grade(case: dict, verdict: str | None, verdict_text: str) -> dict:
    """The pre-registered rule in `decision_replay_588_cases.yaml`.

    `verdict_text` is what the turn committed to: its final message plus the
    structured object. Exploration along the way does not count — a turn that
    read the retirement and confirmed anyway without saying so is silent."""
    if not verdict:
        return {"grade": NO_VERDICT, "markers": []}
    markers = cites_retirement(case, verdict_text)
    if verdict == "confirmed" and not markers:
        return {"grade": RE_PROPOSED, "markers": []}
    return {"grade": CAUGHT, "markers": markers}


def load_overrides(path: Path = OVERRIDES_PATH) -> dict:
    if not path.exists():
        return {}
    import yaml
    out = yaml.safe_load(path.read_text()) or {}
    for k, v in out.items():
        if (v.get("grade") not in (RE_PROPOSED, CAUGHT, NO_VERDICT)
                or not str(v.get("reason") or "").strip()):
            raise ValueError(f"override {k!r} needs a known grade and a written reason")
    return out


def summarize(rows: list[dict], cases: list[dict], overrides: dict | None = None) -> dict:
    """The latest row per (key, rep) wins; a human override replaces the
    automatic grade and keeps it beside the override as `auto`."""
    by_key = {c["key"]: c for c in cases}
    overrides = overrides or {}
    latest: dict[str, dict] = {}
    for r in rows:
        latest[f"{r['key']}#{r.get('rep', 1)}"] = r
    out = {"n": 0, RE_PROPOSED: 0, CAUGHT: 0, NO_VERDICT: 0, "rows": []}
    for rid, r in latest.items():
        case = by_key.get(r["key"])
        if case is None:
            continue
        g = grade(case, r.get("verdict"), r.get("verdict_text") or "")
        if rid in overrides:
            g = {**g, "auto": g["grade"], "grade": overrides[rid]["grade"],
                 "override": overrides[rid]["reason"]}
        out["n"] += 1
        out[g["grade"]] += 1
        out["rows"].append({"key": r["key"], "rep": r.get("rep", 1),
                            "verdict": r.get("verdict"), **g,
                            "tool_calls": len(r.get("tool_calls") or []),
                            "seconds": r.get("seconds"),
                            "stop_reason": r.get("stop_reason")})
    return out


# ── the replay ──────────────────────────────────────────────────────────────

async def run_case(case: dict, index: int, *, max_turns: int, rep: int) -> dict:
    import yaml as _yaml
    from app.harness import HookRegistry, RunOptions, install_default_safety_hook
    from app.harness.loop import run_query
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
    from app.paths import VAULT_ROOT
    from prompt_builder import build_system_prompt
    from scripts.automod import backlog as B, state as S
    from workers.sources.autotriage import parse_verdict, render_prompt

    config = _yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
    alias = (config.get("model") or {}).get("default", "primary")
    model_env = ((config.get("models") or {}).get(alias) or {}).get("env") or {}
    base_url = model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096")

    session_id = require_sandboxed(new_session_id(case["key"]))
    prompt = render_prompt(candidate_for(case, index), ledger=S.LEDGER_PATH)
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    options = RunOptions(
        model=alias, base_url=base_url,
        # The vault's identity dir as an explicit overlay: prompt_builder finds
        # SOUL/MEMORY/USER at `<checkout>/../obsidian/lloyd`, which from a
        # worktree is not the vault, and the replay must carry what production
        # carries (MEMORY.md in, USER.md dropped for a worker turn).
        system_prompt=build_system_prompt(platform="worker",
                                          overlay_dir=VAULT_ROOT / "lloyd"),
        max_turns=max_turns, mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
        disallowed_tools=_get_disallowed_tools(), session_id=session_id,
        priority=1, hooks=hooks,
        final_schema=B.TRIAGE_VERDICT_SCHEMA, final_schema_prompt=FINAL_SCHEMA_PROMPT,
        **_get_harness_kwargs())
    options.surface = "worker"

    tool_calls: list[dict] = []
    refused = 0
    last_text = ""
    result: dict = {}
    t0 = time.perf_counter()
    async for evt in run_query([{"role": "user", "content": prompt}], options):
        kind = evt["type"]
        if kind == "tool_call":
            tool_calls.append({"name": evt.get("name"), "summary": evt.get("summary") or "",
                               "args": (evt.get("args_json") or "")[:400]})
        elif kind == "tool_result":
            content = str(evt.get("content") or "")
            if evt.get("is_error") and ("sandbox" in content.lower()
                                        or "read-only session" in content.lower()):
                refused += 1
        elif kind == "assistant_message":
            if (evt.get("text") or "").strip():
                last_text = evt["text"]
        elif kind == "result":
            result = evt
    structured = result.get("structured")
    parsed = parse_verdict(last_text or result.get("response_text") or "", structured)
    verdict = (parsed or {}).get("verdict")
    verdict_text = (last_text or "") + "\n" + json.dumps(structured or {}, ensure_ascii=False)
    return {
        "key": case["key"], "rep": rep, "session_id": session_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict, "surface": (parsed or {}).get("surface"),
        "verdict_text": verdict_text[-12000:],
        "structured_error": result.get("structured_error"),
        "stop_reason": result.get("stop_reason"), "num_turns": result.get("num_turns"),
        "seconds": round(time.perf_counter() - t0, 1),
        "tool_calls": tool_calls, "sandbox_refusals": refused,
        "max_turns": max_turns,
    }


async def _run(args) -> int:
    from scripts.autoresearch.bench_runner_sdk import require_tool_sandbox
    await require_tool_sandbox()
    cases = load_cases()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, 1):
        if args.only and case["key"] not in args.only:
            continue
        print(f"[{case['key']}] replaying…", flush=True)
        try:
            row = await run_case(case, index, max_turns=args.max_turns, rep=args.rep)
        except Exception as exc:  # noqa: BLE001 — one case's failure is a row
            row = {"key": case["key"], "rep": args.rep, "verdict": None,
                   "error": f"{type(exc).__name__}: {exc}"}
        with ROWS_PATH.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        g = grade(case, row.get("verdict"), row.get("verdict_text") or "")
        print(f"[{case['key']}] verdict={row.get('verdict')} grade={g['grade']} "
              f"markers={g['markers']} tools={len(row.get('tool_calls') or [])} "
              f"{row.get('seconds')}s stop={row.get('stop_reason')}", flush=True)
    return 0


def _grade_cmd() -> int:
    rows = [json.loads(line) for line in ROWS_PATH.read_text().splitlines() if line.strip()]
    s = summarize(rows, load_cases(), load_overrides())
    for r in s["rows"]:
        note = f"  (auto: {r['auto']}; human: {r['override']})" if "auto" in r else ""
        print(f"{r['key']:<18} rep{r['rep']} {str(r['verdict']):<13} {r['grade']:<12} "
              f"{r['markers']}{note}")
    print(f"\nre-proposed {s[RE_PROPOSED]}/{s['n']}  caught {s[CAUGHT]}  "
          f"no-verdict {s[NO_VERDICT]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--only", nargs="*", default=None)
    # Production triage budget is 90; 40 bounds engine spend and still leaves
    # the turn room to check its premise; a budget death is reported, not graded.
    run.add_argument("--max-turns", type=int, default=40)
    run.add_argument("--rep", type=int, default=1)
    sub.add_parser("grade")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return asyncio.run(_run(args))
    return _grade_cmd()


if __name__ == "__main__":
    raise SystemExit(main())
