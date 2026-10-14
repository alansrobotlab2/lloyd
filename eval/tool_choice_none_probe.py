#!/usr/bin/env python3
"""Does the live primary still render `tools` at `tool_choice: "none"`? (#2678)

    <venv>/bin/python eval/tool_choice_none_probe.py                # report, rc 0 or 1
    <venv>/bin/python eval/tool_choice_none_probe.py --json         # one artifact on stdout
    <venv>/bin/python eval/tool_choice_none_probe.py --json-out F    # ...and written there

Why this exists
---------------
Two shipped things rest on one engine property. The finalizer
(`app/harness/finalizer.py`) sends the turn's identical `tools` array with
`tool_choice: "none"`, which is what keeps its extra completion a byte-identical
continuation of the turn's prefix — 1.19 s against 25.00 s for dropping the
array (`architecture/harness.md` §Structured verdicts,
`eval/measurements/finalizer-2026-09-08.md`). P6(b)'s max-turns wrap-up
(`harness.max_turns_wrapup`, shipped `models: [primary]`) sends the same shape
and is enabled per engine for the same reason.

The property was verified on vLLM 0.28.1 on 2026-09-08. The build serving the
primary has moved since, and the last look at it was a grep of the serve args
for `--exclude-tools-when-tool-choice-none` — an inference about a flag, not the
measured behaviour. A build that dropped `tools` at `"none"`, or attached the
tool parser there, is silent: the answer still arrives and the only trace is a
full prefill in `usage.db`.

Why `prompt_tokens` and not `cached_tokens`
-------------------------------------------
`cached_tokens` is what 09-08 measured, and it needs a warm prefix — it reads 0
on the first request of a fresh one (re-measured 2026-10-14: 0 on the first
request of a prefix, 3200 on every later request that kept it), so one cold
moment reads exactly like the dropped-tools failure. `usage.prompt_tokens`
counts the RENDERED template instead. While tools are still rendered at
`"none"`, the `none` request costs exactly what the `auto` request costs; once a
build drops them, it costs what the tools-dropped control costs. That comparison
needs no warm cache, and it also sees template drift between builds, which the
static 41-of-1536-token figures in the 09-08 artifact (measured through the
template with no engine involved) cannot.

The control is not a nice-to-have. With a template that rendered no tools at
all, `none` and `auto` would be equal and the equality would witness nothing;
`no_tools` < `auto` is what makes a PASS mean the schema is in the prompt. That
same request is the 25.00 s failure shape, so it doubles as the proof that the
instrument can see the thing it is watching for.

The decision is a pure function over the readings (`decide`), so it is decided
offline in `tests/test_tool_choice_none_probe.py` — the live-script-plus-
offline-test split of `eval/engine_output_probe.py`. A request the engine
refuses is a FAIL, never an exception: an engine that will not take the shape
the wrap-up sends is exactly as disqualifying as one that takes it wrongly, and
a crash would let a refused request look like a broken script rather than a
verdict.

Exit codes
----------
0  PASS — `none` renders the same prompt size as `auto` with tools attached, and
   the control shows a non-zero tool schema in that prompt.
1  FAIL — the two sizes differ, a graded request was refused or answered without
   `usage.prompt_tokens`, `none` emitted a tool call, or the engine would not
   name its own build: a measurement that cannot say which build it measured
   must not be quotable, which is the whole point of recording it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

LLOYD_HOME = Path(__file__).resolve().parent.parent
if str(LLOYD_HOME) not in sys.path:
    sys.path.insert(0, str(LLOYD_HOME))

SCHEMA = 1
DEFAULT_BASE_URL = "http://127.0.0.1:8096"
DEFAULT_MODEL = "primary"
DEFAULT_MAX_TOKENS = 64
DEFAULT_WARM_REQUESTS = 1
#: ~8k prompt tokens: long enough that a divergence is a real number rather than
#: a rounding artefact, short enough to run beside a live turn.
DEFAULT_FILLER_REPEATS = 60
REQUEST_TIMEOUT_S = 300.0

#: The graded shapes. `auto` is what a turn sends and the reference for the
#: rendered size; `none` is what the finalizer and the wrap-up send; `no_tools`
#: is the control that prices the tool schema — and the failure shape itself.
AUTO, NONE, NO_TOOLS = "auto", "none", "no_tools"
SHAPES = (AUTO, NONE, NO_TOOLS)

_PREFIX_BLOCK = (
    "The machine runs a supervisor of services: the backend on port 8000, the "
    "MCP server, a local vLLM primary on 8096 serving a quantised Qwen3.8 Flash "
    "Next with an n-gram speculative decoder, a llama.cpp secondary, a djev "
    "structured-decision engine on the third GPU, and a voice pipeline that "
    "binds a UDP media port. Knowledge lives in an Obsidian vault that is "
    "git-backed, and runtime state lives in a separate data root outside the "
    "checkout, so a scan that walks the code tree never sees a store the "
    "retention sweep is meant to bound. "
)


def make_messages(filler_repeats: int = DEFAULT_FILLER_REPEATS) -> list[dict[str, str]]:
    """The one conversation every graded request shares, byte-identical.

    The whole probe hangs on the three requests differing ONLY in the tool
    fields, so the messages are built once per run and never mutated.
    """
    return [
        {"role": "system",
         "content": "You are Lloyd, a pragmatic autonomous agent. " + _PREFIX_BLOCK * filler_repeats},
        {"role": "user",
         "content": "Without calling any tool, tell me in two sentences what services run on this box."},
    ]


#: Two function tools. The property is about a non-empty `tools` array being
#: rendered, so two is enough to price it; production sends more, which only
#: moves the control gap, never the equality.
TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "Bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "cwd": {"type": "string"},
            "timeout": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "Read", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "offset": {"type": "integer"},
            "limit": {"type": "integer"}}, "required": ["file_path"]}}},
]


class EngineClient:
    """One tiny client, injectable so the tests never touch an engine."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = REQUEST_TIMEOUT_S):
        import httpx

        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    def get_text(self, path: str) -> str:
        resp = self._http.get(self.base_url + path)
        resp.raise_for_status()
        return resp.text

    def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post(self.base_url + path, json=body)
        resp.raise_for_status()
        return resp.json()


# --------------------------------------------------------------------------
# The three request bodies
# --------------------------------------------------------------------------

def completion_body(*, shape: str, model: str = DEFAULT_MODEL,
                    messages: list[dict[str, str]],
                    tools: list[dict[str, Any]] | None = None,
                    max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    """The request for one graded shape, identical apart from the tool fields.

    `no_tools` deliberately omits `tool_choice` as well: that is the shape the
    doc measures at 25.00 s, and sending `"none"` with no tools would be a third
    thing nobody ships.
    """
    body: dict[str, Any] = {"model": model, "messages": list(messages),
                            "max_tokens": max_tokens, "temperature": 0.0}
    if shape != NO_TOOLS:
        body["tools"] = list(tools if tools is not None else TOOLS)
        body["tool_choice"] = shape
    return body


def engine_version(client: Any) -> str | None:
    """The engine's own build string from `/version`; None when it will not say.

    Never raise: an engine that answers completions but not `/version` still
    gets a verdict, and `decide` is the place that decides whether a measurement
    with no build attached can be reported at all.
    """
    try:
        return json.loads(client.get_text("/version")).get("version")
    except Exception:  # noqa: BLE001 — any transport failure is "did not name itself"
        return None


def parse_response(resp: dict[str, Any]) -> dict[str, Any]:
    """The three readings this instrument can make off one answer.

    A missing `usage.prompt_tokens` stays None rather than becoming 0: 0 is a
    measurement, a missing field is the engine not answering the question.
    """
    usage = resp.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {"prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "tool_calls": len(message.get("tool_calls") or []),
            "finish_reason": choice.get("finish_reason")}


def _sha(*parts: Any) -> str:
    blob = "\x1e".join(json.dumps(p, sort_keys=True) for p in parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------

def run_probe(client: Any, *, model: str = DEFAULT_MODEL,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              warm_requests: int = DEFAULT_WARM_REQUESTS,
              filler_repeats: int = DEFAULT_FILLER_REPEATS,
              tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """One pass -> one readings record. Never raises: every request that fails
    becomes a row with `ok: false`, which `decide` reads as a FAIL.

    Order matters. The warm-up requests and the graded `auto`/`none` pair share
    one prefix so `cached_tokens` has something to report; the tools-dropped
    control goes LAST because it diverges that prefix, and an alternating A/B
    ends up caching both shapes — which is not what production does (§Structured
    verdicts).
    """
    tools = TOOLS if tools is None else tools
    messages = make_messages(filler_repeats)
    version = engine_version(client)
    rows: dict[str, Any] = {}

    def _send(label: str, shape: str) -> None:
        body = completion_body(shape=shape, model=model, messages=messages,
                               tools=tools, max_tokens=max_tokens)
        t0 = time.monotonic()
        try:
            parsed = parse_response(client.post_json("/v1/chat/completions", body))
            row = {"ok": True, "error": None, **parsed}
        except Exception as exc:  # noqa: BLE001 — refused == FAIL, decided below
            row = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "prompt_tokens": None, "cached_tokens": None,
                   "tool_calls": 0, "finish_reason": None}
        row.update({"label": label, "shape": shape, "engine_version": version,
                    "tools_attached": "tools" in body,
                    "tool_choice": body.get("tool_choice"),
                    "elapsed_s": round(time.monotonic() - t0, 2)})
        rows[label] = row

    for i in range(max(0, warm_requests)):
        _send(f"warm-{i + 1}", AUTO)
    for shape in SHAPES:
        _send(shape, shape)

    return {
        "schema": SCHEMA,
        "kind": "tool_choice_none_probe",
        "measured": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": {"base_url": getattr(client, "base_url", None), "version": version},
        "params": {"model": model, "max_tokens": max_tokens,
                   "warm_requests": max(0, warm_requests),
                   "filler_repeats": filler_repeats, "n_tools": len(tools),
                   "prefix_sha256": _sha(messages, tools)},
        "rows": rows,
    }


# --------------------------------------------------------------------------
# The decision — pure over the readings, so it runs offline
# --------------------------------------------------------------------------

def decide(readings: dict[str, Any]) -> dict[str, Any]:
    """readings -> verdict. No I/O, no clock: a dict in, a dict out.

    Decides on `prompt_tokens` equality and nothing else. `cached_tokens` is
    carried into the artifact and the report because it is the number the 25.00 s
    figure came from, but it never moves the verdict: it needs a warm prefix, and
    a cold moment is indistinguishable from the failure it would be watching for.
    """
    rows = readings.get("rows") or {}
    reasons: list[str] = []

    for shape in SHAPES:
        row = rows.get(shape)
        if row is None:
            reasons.append(f"the {shape!r} request was never sent")
        elif not row.get("ok"):
            reasons.append(f"the engine refused the {shape!r} request: {row.get('error')}")
        elif row.get("prompt_tokens") is None:
            reasons.append(f"the {shape!r} answer carried no usage.prompt_tokens")

    version = (readings.get("engine") or {}).get("version")
    if not version:
        reasons.append("the engine did not name its build on /version, so this "
                       "measurement could not be recorded beside one")

    numbers = {
        "auto_prompt_tokens": (rows.get(AUTO) or {}).get("prompt_tokens"),
        "none_prompt_tokens": (rows.get(NONE) or {}).get("prompt_tokens"),
        "no_tools_prompt_tokens": (rows.get(NO_TOOLS) or {}).get("prompt_tokens"),
        "auto_cached_tokens": (rows.get(AUTO) or {}).get("cached_tokens"),
        "none_cached_tokens": (rows.get(NONE) or {}).get("cached_tokens"),
        "no_tools_cached_tokens": (rows.get(NO_TOOLS) or {}).get("cached_tokens"),
        "none_tool_calls": (rows.get(NONE) or {}).get("tool_calls"),
        "schema_cost_tokens": None,
    }
    a, n, c = numbers["auto_prompt_tokens"], numbers["none_prompt_tokens"], numbers["no_tools_prompt_tokens"]
    if a is not None and c is not None:
        numbers["schema_cost_tokens"] = a - c

    if not reasons:
        # The control first: equality only witnesses "tools are rendered" if the
        # schema is worth something in the prompt to begin with.
        if numbers["schema_cost_tokens"] <= 0:
            reasons.append(f"the tools-dropped control cost as much as tools attached "
                           f"(auto={a}, no_tools={c}), so this template renders no tool "
                           f"schema and the equality would witness nothing")
        if n != a:
            gap = "the same" if n == c else f"not the control's {c}"
            reasons.append(f"tool_choice 'none' renders a different prompt than 'auto': "
                           f"none={n}, auto={a}, control(no tools)={gap} — "
                           f"{abs(a - n)} prompt_tokens of tool schema are not being "
                           f"rendered at 'none', so the wrap-up loses the prefix cache")
        if numbers["none_tool_calls"]:
            reasons.append(f"tool_choice 'none' emitted {numbers['none_tool_calls']} tool "
                           f"call(s): the wrap-up would have to drop a dispatched call")

    return {"passed": not reasons, "reasons": reasons,
            "engine_version": version, "numbers": numbers}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render(readings: dict[str, Any], verdict: dict[str, Any]) -> str:
    """The human report. Always names the build, PASS or FAIL — a verdict that
    cannot be traced to a build is how 0.28.1 came to be quoted on 0.31.1."""
    eng = readings.get("engine") or {}
    num = verdict["numbers"]
    lines = [f"engine   {eng.get('base_url')}  version {eng.get('version')}",
            f"measured {readings.get('measured')}  prefix {readings['params']['prefix_sha256']} "
            f"({readings['params']['n_tools']} tools, {readings['params']['filler_repeats']} filler blocks)"]
    for label, row in readings["rows"].items():
        if not row.get("ok"):
            lines.append(f"  {label:9s} REFUSED  {row.get('error')}")
            continue
        lines.append(f"  {label:9s} prompt={row['prompt_tokens']} cached={row['cached_tokens']} "
                     f"tool_calls={row['tool_calls']} finish={row['finish_reason']} "
                     f"({row['elapsed_s']}s)")
    lines.append(f"tool-schema cost: {num['schema_cost_tokens']} prompt_tokens "
                 f"(auto with tools minus the no-tools control)")
    lines.append(f"cached_tokens is reported, not decided: auto={num['auto_cached_tokens']} "
                 f"none={num['none_cached_tokens']} control={num['no_tools_cached_tokens']}")
    if verdict["passed"]:
        lines.append(f"PASS — on {verdict['engine_version']} tool_choice 'none' renders the same "
                     f"{num['none_prompt_tokens']}-token prompt as 'auto' with tools attached, "
                     f"and emitted no tool call")
    else:
        lines.append(f"FAIL — on {verdict['engine_version']}")
        lines.extend(f"  * {r}" for r in verdict["reasons"])
        lines.append("  The max-turns wrap-up must not ship on this engine: clear "
                     "harness.max_turns_wrapup.models until the flag situation changes.")
    return "\n".join(lines)


def build_artifact(readings: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    """The `--json` artifact: the verdict, and every reading with the build it
    came from sitting in the same row as it.

    `engine_version` is repeated inside each graded row on purpose. A top-level
    field can be stripped by whoever pastes the numbers into a doc; a row that
    carries `{engine_version, prompt_tokens, cached_tokens}` together cannot be
    quoted without its build.
    """
    rows = readings["rows"]
    graded = {}
    for shape in SHAPES:
        row = rows.get(shape) or {}
        graded[shape] = {"engine_version": row.get("engine_version"),
                         "prompt_tokens": row.get("prompt_tokens"),
                         "cached_tokens": row.get("cached_tokens"),
                         "tool_calls": row.get("tool_calls"),
                         "ok": row.get("ok"), "error": row.get("error")}
    return {"schema": SCHEMA, "kind": "tool_choice_none_probe",
            "measured": readings.get("measured"),
            "engine": readings.get("engine"),
            "params": readings.get("params"),
            "verdict": "PASS" if verdict["passed"] else "FAIL",
            "reasons": verdict["reasons"],
            "numbers": verdict["numbers"],
            "readings": graded,
            "warm_rows": [rows[k] for k in rows if k.startswith("warm-")]}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None, *,
         client_factory: Callable[[str], Any] = EngineClient) -> int:
    ap = argparse.ArgumentParser(
        description="Measure whether the live engine still renders `tools` at "
                    "tool_choice 'none' (the finalizer / max-turns wrap-up prefix property).")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--warm", type=int, default=DEFAULT_WARM_REQUESTS,
                    help="extra 'auto' requests first, so cached_tokens has reuse to report")
    ap.add_argument("--filler-repeats", type=int, default=DEFAULT_FILLER_REPEATS,
                    help="prefix size in ~130-token blocks")
    ap.add_argument("--json", action="store_true", help="print the JSON artifact instead of the report")
    ap.add_argument("--json-out", type=Path, help="also write the artifact here")
    args = ap.parse_args(argv)

    client = client_factory(args.base_url)
    readings = run_probe(client, model=args.model, max_tokens=args.max_tokens,
                         warm_requests=args.warm, filler_repeats=args.filler_repeats)
    verdict = decide(readings)
    artifact = build_artifact(readings, verdict)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(json.dumps(artifact, indent=1) if args.json else render(readings, verdict))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
