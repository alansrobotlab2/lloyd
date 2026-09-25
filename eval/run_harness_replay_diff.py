"""Replay persisted sessions through two trees' `run_query` and diff them (P13).

The proof of "unchanged behaviour" for a refactor of the agent loop
(`app/harness/loop.py`). A unit test pins one property at a time; this pins
all of them at once, on the turns Lloyd actually ran.

    python eval/run_harness_replay_diff.py sample --n 200 --out DIR
    python eval/run_harness_replay_diff.py all --base BASE_TREE --head HEAD_TREE \
        --fixtures DIR/fixtures.json --out DIR/run

`sample` reads `~/lloyd-data/sessions/*.json` READ-ONLY, picks a stratified
sample (platform/source x batch shape x size), and writes one fixture per user
turn: the user message, one scripted engine step per persisted iteration (its
reasoning, text, tool calls with their raw argument strings, and the usage the
engine reported), and every tool result keyed by call id. Nothing is ever
written under the sessions directory.

`all` runs every fixture through `run_query` in a child process per tree
(`sys.path[0]` = that tree, `LLOYD_DATA` = a scratch dir so a spill or an event
log never reaches production), with the loop's two process seams replaced —
`stream_chat` by a scripted engine and `_build_pool` by a scripted MCP pool —
exactly as `app/harness/tests/_replay.py` does. The base tree runs twice (an
A/A: anything nondeterministic in the harness or this script shows up there
first), then the head. It diffs, per turn:

  - events, as `(type, call_id, sha1(content))`, timing fields dropped;
  - the final `chat_messages`, as `(role, tool_call_id, sha1(content),
    has_reasoning, has_reasoning_content)`;
  - the session event log (`harness.*` and friends), as `(event, sha1(data))`;
  - a hook trace: what every PreToolUse / OnEvent callback saw, including the
    length and tail of the message list at that moment — which is where "the
    pre-dispatch hook for call 2 sees result 1" is actually observable.

Every fixture runs under two arms: `seq` (parallel dispatch off — production's
setting) and `par` (on, concurrency 3, so batches wider than the semaphore are
exercised). A third of the sessions add perturbations drawn deterministically
from the session name: a context-overflow rejection, a stalled stream with one
retry, a state anchor every fourth iteration, a notification drain, a
disallowed tool, a deny and an inject from PreToolUse hooks, an observer inject
on a terminal iteration, and a `max_turns` budget with the toolless wrap-up.

Whitelist: in the `par` arm only, an events or hook-trace diff whose multiset
of `(type, call_id)` is unchanged is classed `interleave` — the one difference
a single dispatch path is allowed to make (a batch wider than the semaphore now
announces a call when it is admitted). History and the event log are never
whitelisted.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
SESSIONS_DEFAULT = Path.home() / "lloyd-data" / "sessions"
VOLATILE_KEYS = {"duration_ms", "ttft_ms", "request_ms", "handshake_ms",
                 "reasoning_ms"}
REPLAY_BASE_URL = "http://replay.invalid:1"
END_TEXT = "(replay: end of scripted steps)"


def _sha(obj: Any) -> str:
    if not isinstance(obj, str):
        obj = json.dumps(obj, sort_keys=True, default=repr)
    return hashlib.sha1(obj.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


# ── sample: sessions → fixtures ────────────────────────────────────────────

def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _turns_of(session: dict[str, Any]) -> list[dict[str, Any]]:
    """One fixture per user turn, rebuilt from the persisted rows."""
    turns: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    step: dict[str, Any] | None = None
    step_iter: Any = object()

    def new_step(it: Any) -> dict[str, Any]:
        nonlocal step, step_iter
        step = {"reasoning": "", "text": "", "tool_calls": [], "usage": None}
        step_iter = it
        cur["steps"].append(step)
        return step

    for m in session.get("messages") or []:
        role = m.get("role")
        if role == "user":
            cur = {"user": _text_of(m.get("content")), "steps": [],
                   "answers": {}}
            turns.append(cur)
            step, step_iter = None, object()
            continue
        if cur is None:
            continue
        if role == "thinking":
            it = (m.get("thinking") or {}).get("iteration")
            s = step if (step is not None and it == step_iter
                         and not step["text"] and not step["tool_calls"]) \
                else new_step(it)
            s["reasoning"] += m.get("reasoning") or ""
        elif role == "assistant":
            if m.get("synthetic_empty_terminal"):
                continue
            stats = m.get("stats") or {}
            it = stats.get("iteration", step_iter)
            s = step if (step is not None and it == step_iter) else new_step(it)
            s["text"] += _text_of(m.get("content"))
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                s["tool_calls"].append({
                    "id": tc.get("id") or tc.get("call_id"),
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments") or "{}",
                })
            if s["usage"] is None and stats.get("input_tokens") is not None:
                s["usage"] = {
                    "prompt_tokens": int(stats.get("input_tokens") or 0),
                    "completion_tokens": int(stats.get("output_tokens") or 0),
                    "prompt_tokens_details": {
                        "cached_tokens": int(stats.get("cache_read") or 0)},
                }
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid:
                cur["answers"][cid] = {
                    "content": _text_of(m.get("content")),
                    "is_error": bool((m.get("stats") or {}).get("is_error")),
                }
    return [t for t in turns if t["steps"]]


def _stratum(name: str, session: dict[str, Any]) -> tuple[str, str, str]:
    parts = name.split("_")
    kind = parts[2] if len(parts) >= 4 else (session.get("platform") or "chat")
    turns = _turns_of(session)
    widths = [len(s["tool_calls"]) for t in turns for s in t["steps"]]
    shape = ("wide" if any(w > 3 for w in widths)
             else "multi" if any(w > 1 for w in widths) else "single")
    n = sum(len(t["steps"]) for t in turns)
    size = "s" if n <= 15 else "m" if n <= 60 else "l"
    return kind, shape, size


def cmd_sample(args: argparse.Namespace) -> None:
    src = Path(args.sessions)
    out = Path(args.out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    strata: dict[tuple, list[str]] = collections.defaultdict(list)
    for p in sorted(src.glob("*.json")):
        try:
            session = json.loads(p.read_text())
        except Exception:
            continue
        if not _turns_of(session):
            continue
        strata[_stratum(p.stem, session)].append(p.name)
    # Round-robin over strata, deterministic within each (hash order), so a
    # rare shape is represented before a common one is exhausted.
    for names in strata.values():
        names.sort(key=_h)
    picked: list[str] = []
    keys = sorted(strata)
    while len(picked) < args.n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < args.n:
                picked.append(strata[k].pop(0))
    fixtures = []
    for name in picked:
        shutil.copyfile(src / name, out / "sessions" / name)   # read from src only
        session = json.loads((out / "sessions" / name).read_text())
        for ti, turn in enumerate(_turns_of(session)):
            fixtures.append({"session": Path(name).stem, "turn": ti,
                             "stratum": list(_stratum(Path(name).stem, session)),
                             **turn})
    (out / "fixtures.json").write_text(json.dumps(fixtures))
    counts = collections.Counter(tuple(f["stratum"]) for f in fixtures
                                 if f["turn"] == 0)
    print(json.dumps({"sessions": len(picked), "turns": len(fixtures),
                      "strata": {"/".join(k): v for k, v in sorted(counts.items())}},
                     indent=1))


# ── child: run fixtures through one tree ───────────────────────────────────

def _child(args: argparse.Namespace) -> None:
    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(tree))
    os.chdir(tree)
    import logging
    logging.disable(logging.CRITICAL)

    from app import event_log
    from app.harness import loop as L
    from app.harness.errors import ContextOverflowError, StreamStalledError
    from app.harness.hooks import HookRegistry
    from app.harness.options import RunOptions
    try:
        from agent_mcp.annotations import READ_ONLY
    except Exception:
        READ_ONLY = frozenset({"Read", "Grep", "Glob"})

    fixtures = json.loads(Path(args.fixtures).read_text())
    if args.limit:
        fixtures = fixtures[: args.limit]
    elog: list[tuple[str, str]] = []

    def _log_event(session_id, event, data, turn_id=None, **_kw):
        d = {k: v for k, v in (data or {}).items()
             if k not in VOLATILE_KEYS and k not in ("ts", "timestamp")}
        elog.append((event, _sha(d)))

    event_log.log_event = _log_event

    class Engine:
        def __init__(self, steps, chaos):
            self.steps = steps
            self.pos = 0
            self.requests = 0
            self.chaos = chaos

        def __call__(self, **kw):
            self.requests += 1
            return self._gen(self.requests)

        async def _gen(self, req):
            if req == self.chaos.get("overflow_at_request"):
                raise ContextOverflowError(
                    "replay overflow", requested_input_tokens=270_000)
            if self.pos < len(self.steps):
                step = self.steps[self.pos]
            else:
                step = {"reasoning": "", "text": END_TEXT, "tool_calls": [],
                        "usage": None}
            stall = req == self.chaos.get("stall_at_request")
            if step["reasoning"]:
                half = max(1, len(step["reasoning"]) // 2)
                for part in (step["reasoning"][:half], step["reasoning"][half:]):
                    if part:
                        yield {"choices": [{"delta": {"reasoning": part}}]}
            if step["text"]:
                yield {"choices": [{"delta": {"content": step["text"]}}]}
            if stall:
                raise StreamStalledError(5.0, lines_seen=2)
            self.pos += 1
            for i, tc in enumerate(step["tool_calls"]):
                raw = tc["arguments"]
                yield {"choices": [{"delta": {"tool_calls": [{
                    "index": i, "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": ""}}]}}]}
                cut = len(raw) // 2
                for frag in (raw[:cut], raw[cut:]):
                    if frag:
                        yield {"choices": [{"delta": {"tool_calls": [{
                            "index": i, "function": {"arguments": frag}}]}}]}
            fin = "tool_calls" if step["tool_calls"] else "stop"
            yield {"choices": [{"delta": {}, "finish_reason": fin}]}
            yield {"choices": [], "usage": step["usage"] or {
                "prompt_tokens": 1000, "completion_tokens": 10}}

    class Pool:
        def __init__(self, names, answers, delays):
            self._discovered = [("lloyd-mcp", [
                {"name": n, "description": "",
                 "inputSchema": {"type": "object", "properties": {}},
                 "annotations": {"readOnlyHint": n in READ_ONLY}}
                for n in sorted(names)])]
            self.answers = answers
            self.delays = delays

        @property
        def discovered(self):
            return self._discovered

        async def ensure_fresh(self):
            return None

        async def call_tool(self, name, args, **kw):
            cid = kw.get("call_id") or ""
            # Delays are counted in event-loop passes, not seconds: completion
            # order is then a function of the fixture alone, where a timer's
            # would move with the machine's load and fail the A/A.
            for _ in range(self.delays.get(cid, 0)):
                await asyncio.sleep(0)
            a = self.answers.get(cid)
            if a is None:
                return {"content": f"RESULT[{name}]", "is_error": False}
            return {"content": a["content"], "is_error": a["is_error"]}

    def _msg_row(m):
        return [m.get("role"), m.get("tool_call_id"),
                _sha(m.get("content")) if m.get("content") is not None else None,
                bool(m.get("tool_calls")) and _sha(m.get("tool_calls")),
                "reasoning" in m, "reasoning_content" in m]

    def _evt_row(e):
        d = {k: v for k, v in e.items() if k not in VOLATILE_KEYS}
        if e.get("type") == "system":
            d.pop("session_id", None)
        return [e.get("type"), e.get("call_id"), _sha(d)]

    async def run_one(fx, arm):
        chaos_on = _h(fx["session"]) % 3 == 0
        seed = _h(f'{fx["session"]}:{fx["turn"]}')
        steps = fx["steps"]
        n_steps = len(steps)
        chaos: dict[str, Any] = {}
        names = {tc["name"] for s in steps for tc in s["tool_calls"] if tc["name"]}
        names |= {"Read", "Grep", "Glob", "Bash"}
        hooks = HookRegistry()
        trace: list[list[Any]] = []
        handle: list[dict[str, Any]] = []

        def tail():
            return [len(handle), _sha([_msg_row(m) for m in handle[-3:]])]

        async def pre(inp, tool_use_id, _ctx):
            trace.append(["pre", tool_use_id, inp["tool_name"], *tail()])
            if chaos_on and _h(str(tool_use_id)) % 23 == 0:
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "replay deny"}}
            if chaos_on and _h(str(tool_use_id)) % 29 == 1:
                handle.append({"role": "user",
                               "content": f"[INNER VOICE] replay inject {tool_use_id}"})
            return {}

        async def post(inp, tool_use_id, _ctx):
            trace.append(["post", tool_use_id, inp["tool_name"], *tail()])
            return {}

        iv_state = {"injected": False}

        async def on_event(evt):
            trace.append(["evt", evt.get("type"), evt.get("call_id"), *tail()])
            if (chaos_on and evt.get("type") == "assistant_message"
                    and not evt.get("tool_calls") and not iv_state["injected"]
                    and seed % 2 == 0):
                iv_state["injected"] = True
                handle.append({"role": "user",
                               "content": "[INNER VOICE] replay terminal inject"})

        hooks.add_pre_tool_use(None, pre)
        hooks.add_post_tool_use(post)
        hooks.add_on_event(on_event)

        kw: dict[str, Any] = {}
        max_turns = n_steps + 3
        if chaos_on:
            if n_steps >= 2 and seed % 4 == 0:
                chaos["overflow_at_request"] = 2
            if n_steps >= 3 and seed % 5 == 1:
                chaos["stall_at_request"] = 3
                kw["stream_retry_max"] = 1
            if seed % 3 == 0 and n_steps >= 4:
                max_turns = max(2, n_steps // 2)
                kw["max_turns_wrapup"] = True
                kw["max_turns_wrapup_base_urls"] = (REPLAY_BASE_URL,)

            async def anchor(n):
                if n % 4 == 0:
                    return [{"role": "user", "content": f"<budget>iteration {n}</budget>"}]
                return []

            drained = {"done": False}

            async def drain():
                if not drained["done"] and seed % 2 == 1:
                    drained["done"] = True
                    return [{"role": "user", "content": "<task_notification>replay</task_notification>"}]
                return []

            kw["state_anchor"] = anchor
            kw["notification_drain"] = drain
            if seed % 7 == 0:
                kw["disallowed_tools_refresh"] = lambda: ["Glob"]
        delays = {}
        if arm == "par":
            for s in steps:
                k = len(s["tool_calls"])
                for i, tc in enumerate(s["tool_calls"]):
                    delays[tc["id"]] = 3 * ((k - i) % 4)
        engine = Engine(steps, chaos)
        pool = Pool(names, fx["answers"], delays)

        async def _ready(_o):
            return pool

        L._build_pool = _ready
        L.stream_chat = engine
        opts = RunOptions(
            model="primary", base_url=REPLAY_BASE_URL,
            system_prompt="You are Lloyd (replay).",
            max_turns=max_turns, hooks=hooks, chat_messages_handle=handle,
            session_id=f'replay_{fx["session"]}_{fx["turn"]}_{arm}',
            tool_search_enabled=False, tool_call_summaries=True,
            preserve_thinking_iterations=6,
            parallel_tool_calls_enabled=(arm == "par"),
            parallel_tool_calls_max_concurrency=3,
            **kw,
        )
        del elog[:]
        out_events: list[list[Any]] = []
        error = None
        try:
            async for evt in L.run_query(
                    [{"role": "user", "content": fx["user"] or "(empty)"}], opts):
                out_events.append(_evt_row(evt))
        except Exception as exc:   # recorded, and diffed like anything else
            error = f"{type(exc).__name__}: {exc}"[:300]
        return {
            "key": f'{fx["session"]}#{fx["turn"]}#{arm}',
            "arm": arm,
            "events": out_events,
            "history": [_msg_row(m) for m in handle],
            "elog": [list(x) for x in elog],
            "trace": trace,
            "error": error,
            "requests": engine.requests,
        }

    async def main():
        t0 = time.time()
        with open(args.out, "w") as fh:
            for fx in fixtures:
                for arm in ("seq", "par"):
                    rec = await run_one(fx, arm)
                    fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"tree": str(tree), "fixtures": len(fixtures),
                          "seconds": round(time.time() - t0, 1)}))

    asyncio.run(main())


def cmd_run(args: argparse.Namespace) -> None:
    data = Path(args.scratch_data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LLOYD_DATA"] = str(data)          # this child only; never production
    env["LLOYD_VOICE_ALERTS"] = "0"
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, str(HERE), "_child", "--tree", args.tree,
           "--fixtures", str(Path(args.fixtures).resolve()),
           "--out", str(Path(args.out).resolve())]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    subprocess.run(cmd, check=True, env=env, cwd=args.tree)


# ── diff ───────────────────────────────────────────────────────────────────

def _load(path: str) -> dict[str, dict[str, Any]]:
    out = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["key"]] = rec
    return out


def _multiset(rows, width: int = 2):
    return collections.Counter(tuple(r[:width]) for r in rows)


def diff_runs(a_path: str, b_path: str) -> dict[str, Any]:
    a, b = _load(a_path), _load(b_path)
    report: dict[str, Any] = {"turns": 0, "clean": 0, "whitelisted": [],
                              "failures": [], "by_channel": collections.Counter()}
    for key in sorted(set(a) | set(b)):
        report["turns"] += 1
        ra, rb = a.get(key), b.get(key)
        if ra is None or rb is None:
            report["failures"].append({"key": key, "channel": "missing"})
            continue
        bad, white = [], []
        for ch in ("history", "elog", "error", "requests"):
            if ra[ch] != rb[ch]:
                bad.append(ch)
        for ch, width in (("events", 2), ("trace", 3)):
            if ra[ch] == rb[ch]:
                continue
            if ra["arm"] == "par" and _multiset(ra[ch], width) == _multiset(rb[ch], width):
                white.append(ch)
            else:
                bad.append(ch)
        for ch in bad:
            report["by_channel"][ch] += 1
        if bad:
            first = {}
            for ch in bad:
                xa, xb = ra[ch], rb[ch]
                if isinstance(xa, list) and isinstance(xb, list):
                    i = next((i for i, (p, q) in enumerate(zip(xa, xb)) if p != q),
                             min(len(xa), len(xb)))
                    first[ch] = {"at": i, "a": xa[i:i + 3], "b": xb[i:i + 3],
                                 "len": [len(xa), len(xb)]}
                else:
                    first[ch] = {"a": xa, "b": xb}
            report["failures"].append({"key": key, "channels": bad, "first": first})
        elif white:
            report["whitelisted"].append({"key": key, "channels": white})
        else:
            report["clean"] += 1
    report["by_channel"] = dict(report["by_channel"])
    return report


def cmd_diff(args: argparse.Namespace) -> None:
    rep = diff_runs(args.a, args.b)
    print(json.dumps({k: (v if k != "failures" else v[:20]) for k, v in rep.items()},
                     indent=1))
    sys.exit(1 if rep["failures"] else 0)


def cmd_all(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {"base": args.base, "base_aa": args.base, "head": args.head}
    for label, tree in runs.items():
        # ONE data dir, emptied per run: a spilled result names its path in
        # the text the model reads, so two dirs would differ on every spill.
        data = out / "data"
        shutil.rmtree(data, ignore_errors=True)
        ns = argparse.Namespace(tree=tree, fixtures=args.fixtures,
                                out=str(out / f"{label}.jsonl"),
                                scratch_data=str(data), limit=args.limit)
        cmd_run(ns)
    aa = diff_runs(str(out / "base.jsonl"), str(out / "base_aa.jsonl"))
    ab = diff_runs(str(out / "base.jsonl"), str(out / "head.jsonl"))
    summary = {
        "aa": {k: aa[k] for k in ("turns", "clean", "by_channel")}
        | {"whitelisted": len(aa["whitelisted"]), "failures": len(aa["failures"])},
        "ab": {k: ab[k] for k in ("turns", "clean", "by_channel")}
        | {"whitelisted": len(ab["whitelisted"]), "failures": len(ab["failures"])},
        "ab_failures": ab["failures"][:20],
        "ab_whitelisted": ab["whitelisted"][:50],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:6000])
    sys.exit(1 if (aa["failures"] or ab["failures"]) else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--sessions", default=str(SESSIONS_DEFAULT))
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--out", required=True)
    c = sub.add_parser("_child")
    c.add_argument("--tree", required=True)
    c.add_argument("--fixtures", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True)
    r.add_argument("--fixtures", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--scratch-data", required=True)
    r.add_argument("--limit", type=int, default=0)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    al = sub.add_parser("all")
    al.add_argument("--base", required=True)
    al.add_argument("--head", required=True)
    al.add_argument("--fixtures", required=True)
    al.add_argument("--out", required=True)
    al.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"sample": cmd_sample, "_child": _child, "run": cmd_run, "diff": cmd_diff,
     "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
