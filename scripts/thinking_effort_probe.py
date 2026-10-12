#!/usr/bin/env python3
"""#2576 — does the primary honour a thinking-effort spelling, and what does each cost?

    cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.thinking_effort_probe
    ... --base-url http://127.0.0.1:8096 --model <id>   # point it elsewhere
    ... --json                      # the rows, machine-readable
    ... --markdown                  # the vault report's table form
    ... --json-out <path>           # and the same rows committed to a file
    ... --witness-out <path>        # the tracked .md witness the report cites
    ... --max-tokens 120            # the matched ceiling (default 120)

Eight POSTs of one prompt, so this costs engine time on the box that has least of
it: run it deliberately, and note that `--max-tokens` is part of the experiment —
under a ceiling a thinking arm cannot finish inside, every thinking arm is
indistinguishable by construction (see the report's ceiling-confound section).

One fixed prompt, posted once per spelling, on the primary only. #580 measured
`thinking_budget`, `reasoning_effort` and a top-level `enable_thinking` as
**no-ops on the llama.cpp secondary**, so the secondary is out of scope and this
script never looks at it; #1431 proved the primary's BINARY switch
(`completion 431 -> 125`, `reasoning_tokens 252 -> 0` on a JSON re-state) but
nobody has measured whether the primary honours a GRADED level — which is the
only question that decides whether a per-class effort knob is even expressible.
If the answer is binary, the arm set is on/off and the report says so rather than
inventing a ladder.

What each arm records, and why those three numbers
--------------------------------------------------
`reasoning_tokens` (from `usage.completion_tokens_details.reasoning_tokens`,
which is where vLLM puts it) is the token bill; `content_len` is what the tokens
bought; `finish_reason` says whether the arm was cut off by the ceiling before it
finished, which matters because a ceiling-punished arm shows up as
`length`, not as a lower score. A response that carries NO reasoning counter is
recorded as None and never as 0: "the engine did not report" and "thinking was
free" are opposite findings, and this repo has already catalogued the difference
once (#1431's whole reason for keeping `reasoning_tokens` in `_usage`).

One prompt, provably
--------------------
Every arm sends byte-identical `messages` and the same `max_tokens`; only the
knob differs. `prompt_sha` — the first 12 hex digits of the sha256 of the prompt
string — is recorded on every row, so a table where two arms disagree about
their prompt is visible from the table itself and not only from the source. This
is the "change any part and it's a different experiment" rule from the source
talk, applied to the one knob we are actually asking about.

No arm is ever omitted
----------------------
An engine that is down, or that refuses a spelling it does not implement, must
still produce a row for that arm: a table missing the row for
`reasoning_effort: "none"` reads exactly like a table where that arm was tried
and cost nothing. So a connection failure or timeout is `unreachable`, an HTTP
4xx/5xx is `error` carrying its status code and the first line of the engine's
complaint, and both are printed. Which is also the honest split — a refused
connection and a refused argument are different facts, and collapsing them would
lose the difference between "cannot measure" and "measured: not implemented".

The knob this script measures is deliberately NOT a knob anyone may set from
here: the engine's default level is a launch flag in
`agent-services/bin/start-qwen38-flash-next.sh`, outside this repo and outside
this loop, and moving it is a person's decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: One prompt for every arm. Short and stateless on purpose: the item's own risk
#: note says displacement needs turns to bite, and a single call can only price
#: the token side of the mechanism — the multi-turn view is the join in
#: `scripts/reasoning_by_job_class.py`, not this probe.
FIXED_PROMPT = ("A train leaves Kings Cross at 09:17 travelling at 84 km/h. "
                "A second leaves at 09:29 on the same line at 96 km/h, and the "
                "first is 12 km ahead when the second starts. When does the "
                "second overtake the first? Give the time and the arithmetic.")

#: The arms, in the order they are always printed. `extra` is merged into the
#: payload; the first row is the control — what `app/harness/client.py` actually
#: sends today, which is no thinking key at all.
#:
#: The graded level names are not guesses. The first run of this probe sent
#: `reasoning_effort: "high"` and the primary answered 400 with its supported set
#: spelled out: "Unexpected reasoning effort high. Supported types are xhigh
#: (default), medium, and low." That is the engine itself saying which graded
#: levels this box accepts, and it says two things at once — the knob IS graded
#: (so a per-class level is expressible), and the level name this repo's stamped
#: prompt sentence uses for the thing it removes (`app/harness/finalizer.py:28-37`
#: strips a sentence reading "Reasoning effort is set to xhigh…") is a level the
#: engine accepts and reports as its DEFAULT.
#:
#: `high` stays in the arm list although the engine refuses it: it is the
#: non-default spelling someone will try next, and its measured answer — an
#: error row carrying the 400 and the engine's own words — is worth more in the
#: table than its absence. An arm that is refused is a result; an arm that is
#: missing reads as a zero.
SPELLINGS: list[dict[str, Any]] = [
    {"spelling": "default (no knob sent)", "extra": {}},
    {"spelling": "chat_template_kwargs.enable_thinking=true",
     "extra": {"chat_template_kwargs": {"enable_thinking": True}}},
    {"spelling": "chat_template_kwargs.enable_thinking=false",
     "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    {"spelling": "reasoning_effort=xhigh", "extra": {"reasoning_effort": "xhigh"}},
    {"spelling": "reasoning_effort=medium", "extra": {"reasoning_effort": "medium"}},
    {"spelling": "reasoning_effort=low", "extra": {"reasoning_effort": "low"}},
    {"spelling": "reasoning_effort=high", "extra": {"reasoning_effort": "high"}},
    {"spelling": "reasoning_effort=none", "extra": {"reasoning_effort": "none"}},
]


def payload_for(arm: dict[str, Any], *, model: str, prompt: str,
                max_tokens: int) -> dict[str, Any]:
    """The request body for one arm: the shared shape plus that arm's knob only."""
    body: dict[str, Any] = {"model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.0}
    body.update(json.loads(json.dumps(arm["extra"])))   # deep copy, no aliasing
    return body


def _row_from_response(arm: dict[str, Any], resp: Any, *, prompt_sha: str) -> dict[str, Any]:
    """One measured row out of a 2xx response."""
    body = resp.json()
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens", usage.get("reasoning_tokens"))
    content = (choice.get("message") or {}).get("content")
    return {"spelling": arm["spelling"], "status": "ok", "http_status": resp.status_code,
            "prompt_sha": prompt_sha,
            "reasoning_tokens": reasoning if isinstance(reasoning, int) else None,
            "content_len": len(content) if isinstance(content, str) else None,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "error": None}


def probe(*, post: Callable[..., Any], url: str, model: str,
          headers: dict[str, str], arms: list[dict[str, Any]] | None = None,
          prompt: str = FIXED_PROMPT, max_tokens: int = 120,
          timeout_s: float = 120.0) -> list[dict[str, Any]]:
    """Post the fixed prompt once per arm and return one row per arm. Always.

    `post` is the transport seam (`httpx.post` in production, a stub in tests):
    this is the process boundary — a script process talking to an engine process
    that can be down, busy, or opinionated about a field it does not implement.
    """
    import httpx

    sha = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    rows: list[dict[str, Any]] = []
    for arm in (arms if arms is not None else SPELLINGS):
        body = payload_for(arm, model=model, prompt=prompt, max_tokens=max_tokens)
        base = {"spelling": arm["spelling"], "prompt_sha": sha, "http_status": None,
                "reasoning_tokens": None, "content_len": None,
                "finish_reason": None, "completion_tokens": None,
                "extra": arm["extra"]}
        try:
            resp = post(url, json=body, headers=headers, timeout=timeout_s)
        except httpx.HTTPError as exc:                    # refused, timed out, no route
            rows.append({**base, "status": "unreachable", "error": repr(exc)[:300]})
            continue
        except Exception as exc:                          # noqa: BLE001 — never drop an arm
            rows.append({**base, "status": "error", "error": repr(exc)[:300]})
            continue
        if resp.status_code >= 400:
            try:
                detail = json.dumps(resp.json())[:300]
            except Exception:                             # noqa: BLE001
                detail = str(getattr(resp, "text", ""))[:300]
            rows.append({**base, "status": "error", "http_status": resp.status_code,
                         "error": detail})
            continue
        row = _row_from_response(arm, resp, prompt_sha=sha)
        rows.append({**base, **{k: v for k, v in row.items() if k != "spelling"}})
    return rows


#: Column order for the markdown rendering, which the vault report reprints. As in
#: `reasoning_by_job_class.py`, the rows have exactly one formatter and the
#: report's copy is pinned to its output, because a hand-typed table drifts.
MD_COLUMNS = ("spelling", "status", "reasoning_tokens", "content chars",
              "completion", "finish")

#: Em-dash-free en dash for "no such counter in the response", matching the report.
NO_VALUE = "–"


def markdown_rows(rows: list[dict[str, Any]]) -> str:
    """The probe table as the vault report prints it, in the probe's arm order.

    `reasoning_tokens` and `content chars` render `–` for None and never `0`: a
    refused or unreachable arm printed as `0` would read as the cheapest setting in
    the table, which is the one false result #2576 must not produce. Content is
    CHARACTERS — the engine reports a string, not a count — and that is what the
    column is named.
    """
    def shown(value: Any) -> str:
        return NO_VALUE if not isinstance(value, int) else f"{value:,}"

    lines = ["| " + " | ".join(MD_COLUMNS) + " |",
             "|" + "|".join(["---"] * len(MD_COLUMNS)) + "|"]
    for r in rows:
        status = r["status"] if r["status"] == "ok" else f"{r['status']} {r['http_status']}"
        lines.append("| " + " | ".join([
            f"`{r['spelling']}`" if " " not in r["spelling"] else r["spelling"],
            status, shown(r["reasoning_tokens"]), shown(r["content_len"]),
            shown(r["completion_tokens"]), r["finish_reason"] or NO_VALUE]) + " |")
    return "\n".join(lines)


WITNESS_HEAD = """# #2576 thinking-effort spelling probe

Committed witness for #2576 (`knowledge/thinking-effort-per-job-class-2026-10-11.md`). Produced by `scripts/thinking_effort_probe.py {flags}` — one fixed prompt, posted to the `url` below once per arm of `SPELLINGS`, on the primary only.

`prompt_sha` is the sha256 prefix of that prompt: every arm sent the same bytes, so a table whose arms disagree about their prompt is visible here. `model` is what the engine's `/v1/models` reported, not what was typed. Numbers only — no prompt text, no completion text.
"""


def write_witness(payload: dict[str, Any], path: Path, *, flags: str) -> None:
    """Commit the probe as the tracked `.md` witness (`*.json` is ignored repo-wide)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(WITNESS_HEAD.format(flags=flags)
                    + "\n```json\n"
                    + json.dumps(payload, indent=2, sort_keys=True)
                    + "\n```\n")


def chat_completions_url(base_url: str) -> str:
    """`http://h:9` and `http://h:9/` both address `/v1/chat/completions`.

    Trailing-slash handling is the whole reason this is a function: the config's
    `base_url` and a hand-passed `--base-url` differ by one character, and a probe
    that 404s on the slash prints eight `error` rows that look exactly like an
    engine refusing every spelling.
    """
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def resolve_model(base_url: str, *, get: Callable[..., Any],
                  headers: dict[str, str], timeout_s: float = 10.0) -> str:
    """The engine's own model id, or "" when it will not say.

    Asked of the engine rather than read from config because the question is what
    THIS engine will accept: a config id the running process never loaded answers
    with a 404 on every arm, and the probe would report the engine as refusing the
    knob. An empty id is not a failure — vLLM serves a single model and accepts an
    unknown or empty one — so the probe still runs all eight arms and says which
    id it used.
    """
    try:
        got = get(f"{base_url.rstrip('/')}/v1/models", headers=headers,
                  timeout=timeout_s)
        got.raise_for_status()
        data = got.json().get("data") or []
        return str(data[0]["id"]) if data else ""
    except Exception as exc:                            # noqa: BLE001 — probe reports it
        print(f"/v1/models on {base_url} failed: {exc!r}; probing with model=''",
              file=sys.stderr)
        return ""


def render(rows: list[dict[str, Any]], *, url: str, model: str) -> str:
    out = [f"#2576 thinking-effort probe  url={url}  model={model or '(unresolved)'}  "
           f"prompt_sha={rows[0]['prompt_sha'] if rows else '-'}", ""]
    header = (f"{'spelling':<42}{'status':<13}{'http':>5}{'reason':>8}"
              f"{'content':>8}{'compl':>7}{'finish':>9}")
    out += [header, "-" * len(header)]
    for r in rows:
        def shown(value: Any, width: int) -> str:
            return str(value).rjust(width) if isinstance(value, int) else "-".rjust(width)
        status_code = r["http_status"]
        out.append(f"{r['spelling']:<42}{r['status']:<13}"
                   f"{(str(status_code) if status_code else '-'):>5}"
                   f"{shown(r['reasoning_tokens'], 8)}{shown(r['content_len'], 8)}"
                   f"{shown(r['completion_tokens'], 7)}"
                   f"{(r['finish_reason'] or '-'):>9}")
        if r["error"]:
            out.append(f"{'':<42}error: {r['error']}")
    out += ["", "`reason`/`compl` are the engine's reported counts; `-` means the "
                "engine returned no such counter or the arm never got a 2xx — it is "
                "NEVER zero. status: ok | unreachable (connection refused or timed "
                "out) | error (the engine answered and refused the argument)."]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=None,
                    help="default: the config `models.primary.base_url`")
    ap.add_argument("--model", default=None,
                    help="default: the first id the engine's /v1/models reports")
    ap.add_argument("--api-key", default="no-key-required")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--markdown", action="store_true",
                    help="print the vault report's table form instead of the aligned one")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--witness-out", type=Path, default=None,
                    help="write the tracked .md witness the report cites to this path")
    args = ap.parse_args(argv)

    import httpx

    base_url = args.base_url
    if not base_url:
        from app.config import MODEL_CONFIGS
        cfg = MODEL_CONFIGS.get("primary") or {}
        base_url = str(cfg.get("base_url") or "")
        if not base_url:
            raise SystemExit("no models.primary.base_url in config and no --base-url")
    url = chat_completions_url(base_url)
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}

    model = args.model or resolve_model(base_url, get=httpx.get, headers=headers)

    rows = probe(post=httpx.post, url=url, model=model, headers=headers,
                 max_tokens=args.max_tokens, timeout_s=args.timeout_s)
    from datetime import datetime, timezone
    payload = {"url": url, "model": model, "max_tokens": args.max_tokens,
               "prompt_sha": rows[0]["prompt_sha"] if rows else None,
               # Recorded because the same command minutes apart produces different
               # graded-level counts (two runs of this probe, ~3 h apart, put `low`
               # above and then below `medium`). A witness of a non-reproducible
               # measurement that does not say when it ran cannot be re-run to check.
               "run_ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "rows": rows}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.witness_out:
        write_witness(payload, args.witness_out,
                      flags=f"--max-tokens {args.max_tokens}")
    if args.as_json:
        print(json.dumps({"url": url, "model": model, "rows": rows}, indent=2))
    elif args.markdown:
        print(markdown_rows(rows))
    else:
        print(render(rows, url=url, model=model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
