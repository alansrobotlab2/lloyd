#!/usr/bin/env python3
"""Per-tool failure-signature baseline, rebuilt from the transcripts on disk.

Backlog #851. Everything needed to rank tool-surface failure modes by cost is
already persisted per call — every `role="tool"` row in `sessions/*.json`
carries `stats.is_error` — and nothing persisted the aggregate, so every
analysis re-derived it and no run could tell whether a fix worked.

Why not just count `is_error`: it fires on any non-zero shell exit, so Bash's
flagged rows are mostly `grep` no-match runs and deliberate failing tests, not
tool defects (#500 measures the over-count at ~4x). The split that makes the
number usable is the one production already makes at the tool boundary — see
`agent_mcp/_shared.py::text_result`, which calls a body an error when it is a
top-level JSON object with an `"error"` key, and `agent_mcp/builtin_bash.py`,
which for a non-zero exit returns *raw command output* plus an
`[exit code: N]` marker precisely because "the payload is raw output, so the
JSON sniffer in `text_result` has nothing to go on". This script recognises
those two shapes and nothing else, so:

  * a structured `{"error": ...}` body becomes a signature (its normalised
    first line), which is what ranks causes;
  * a bare non-zero shell exit becomes the `shell_exit` count and contributes
    no signature, so a Bash rate is never read as a tool defect;
  * a flagged row that is neither — nor one of the three harness-emitted
    error prefixes — is totalled in `guess_class`, outside every signature
    count, so an unclassified row can never inflate a ranked cause.

The window anchors on the corpus (newest session mtime minus `--days`), not on
the clock, so re-running minutes later over the same sessions reproduces the
same numbers; `generated_at` is the only field that may differ between two
runs.

Output lands in `eval/baselines/tool-failures/<date>.json` — a SUBDIRECTORY of
the baselines root. `tests/test_eval_scorer.py::test_a_run_record_declares_
whether_it_matched_production` takes the newest file matching
`eval/baselines/*.json` by mtime and asserts it carries the retrieval-config
knobs; a sibling file there would turn that test red. `Path.glob("*.json")` is
non-recursive, so the subdirectory (sibling of the existing `tool-choice/`) is
invisible to it.

This is an offline read of logs: nothing under `app/` or `agent_mcp/` imports
it, and no field written here reaches a per-run prompt.

    .venvs/lloyd/bin/python -m eval.tool_failure_baseline --days 21
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

DEFAULT_DAYS = 21
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.paths import EVAL_BASELINES_DIR, production_data_root  # noqa: E402

# The live box's transcripts, not this process's data root: a worktree's has
# none, and the aggregate describes production traffic whichever tree the script
# is invoked from. Precedent: `eval/run_skill_dispatch_probe.py:40`,
# `eval/secondary_routing_eval.py:1176`.
DEFAULT_SESSIONS_DIR = production_data_root() / "sessions"
DEFAULT_OUT_DIR = EVAL_BASELINES_DIR / "tool-failures"

# Longest signature kept. Two bodies that differ only past this point are the
# same cause, and an unbounded signature turns a ranked list into a log tail.
SIGNATURE_MAX_CHARS = 200

# How many entries the human-facing global ranking shows. Per-tool signature
# lists are NOT capped: `flagged == sum(signature counts) + shell_exit +
# guess_class` has to hold as read, and a cap would break that arithmetic for
# whoever checks it.
TOP_RANKING = 25

# Failure texts the harness itself writes: plain strings, never JSON, so the
# `text_result` sniffer cannot see them and they are not shell exits either.
# Each is matched on its own line and each names its writer, because a
# signature that turns out to be a prose drift is cheaper to fix than one
# invented here. The `(?:^|: )` forms exist because `mcp_pool` prefixes the
# tool's own name: `Bash: transport error: …`.
HARNESS_MARKERS = (
    (r"^Tool call arguments could not be parsed as JSON", "app/harness/loop.py:1983"),
    (r"^Tool call denied:", "app/harness/loop.py:2129"),
    (r"^Tool dispatch failed:", "app/harness/loop.py:714, :2252"),
    (r"^Tool \S+ is disabled by configuration\.", "app/harness/loop.py:2057"),
    (r"^Tool \S+ cancelled by user\.", "app/harness/loop.py:2225"),
    (r"(?:^|: )no server claims tool ", "app/harness/mcp_pool.py:478"),
    (r"(?:^|: )transport error: ", "app/harness/mcp_pool.py:544, :579"),
)

# The class names, exported so a reader of the JSON knows the whole vocabulary.
CLASS_ERROR_JSON = "error_json"
CLASS_SHELL_EXIT = "shell_exit"
CLASS_HARNESS_ERROR = "harness_error"
CLASS_GUESS = "guess_class"
CLASSES = (CLASS_ERROR_JSON, CLASS_SHELL_EXIT, CLASS_HARNESS_ERROR, CLASS_GUESS)

# Bare non-zero exit marker, `agent_mcp/builtin_bash.py:225`: every non-zero
# exit ends with `\n[exit code: N]`, whether or not `[truncated]` was put in
# front of it and whatever the command printed above it. Matched at the END of
# the body, because the payload above it is raw output — a `grep` whose own
# output happened to contain the string `[exit code: 3]` and which then exited
# 0 is not flagged at all, so the suffix is the only reliable signal.
_SHELL_EXIT_MARKER = re.compile(r"\n\[exit code: -?\d+\]\s*$")

# Volatile spans inside an otherwise identical error, masked so the same cause
# in two files collapses to one signature. Ordered: a URL has to go before the
# path rule, which would otherwise eat `https://host/part`.
_MASK_URL = re.compile(r"https?://[^\s\"'`,)\]}]+")
# Two or more `/segment` runs: matches `/home/alansrobotlab/lloyd/x.py` without
# matching the slash inside "and/or".
_MASK_PATH = re.compile(r"(?:/[^\s'\"`,;:()\[\]]+){2,}/?")
_MASK_HEXID = re.compile(r"\b[0-9a-fA-F]{12,}\b")
_MASK_TS = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?"
                      r"(?:Z|[+-]\d{2}:?\d{2})?)?")
# Positional numbers only — `nearest lines 251, 827`, `at line 42`,
# `column 5234`. Deliberately NOT every digit run, because an HTTP status has
# to stay visible: `HTTP 404` and `HTTP 403` are different causes (#850).
_MASK_POS = re.compile(r"\b(lines?|columns?|offsets?|chars?|rows?)\b\s*"
                       r"(\d+)(?:\s*,\s*\d+)*", re.I)
_MASK_MS = re.compile(r"\b\d+(?:\.\d+)?ms\b")
_MASK_EPOCH = re.compile(r"\b17\d{8,11}(?:\.\d+)?\b")
# Payloads a tool echoes back verbatim: the fetched page in an http_fetch
# error, the command in a Bash timeout. Masked BY KEY, never by length — an
# error's own text is routinely longer than 24 characters (
# `{"error": "old_string not found in file (must match exactly)"}` is 48), and
# a length-based rule would collapse the very signature this file exists to
# rank into `{"error": "<long>"}`.
_MASK_ECHOED = re.compile(r'"(body|command|content|output|preview|text)":\s*"(?:[^"\\]|\\.)*"')
# Single-quoted candidate fragments from an edit or memory-matching error, and
# the blocked pattern in a safety denial. Four characters, not eight: `'sudo'`
# is the corpus's most frequent denial and a threshold of 8 left `blocked
# 'sudo' on 'sudo'` and `blocked <q> on <q>` as two signatures for one cause.
_MASK_SQUOTE = re.compile(r"'[^'\n]{4,}'")


def _signature_line(text: str) -> str:
    """The first line of `text` that carries a letter.

    Pretty-printed JSON error bodies open with a bare `{`, and a signature of
    `{` groups every such error together and ranks nothing — which is what the
    first cut of this script produced for the multi-line `automod_vault_land`
    and `automod_land` error bodies. Structural lines carry no cause, so they
    are skipped.
    """
    for line in (text or "").splitlines():
        stripped = line.strip()
        if any(ch.isalpha() for ch in stripped):
            return stripped
    return ""


def normalise_signature(text: str) -> str:
    """The signature of one tool-result body: its normalised first line.

    Pure function of the text — same body, same signature, in any run — so a
    baseline written today is comparable with one written after #849/#850 land.
    """
    first = _signature_line(text)
    if not first:
        return ""
    out = _MASK_ECHOED.sub(lambda m: f'"{m.group(1)}": "<long>"', first)
    out = _MASK_URL.sub("<url>", out)
    out = _MASK_PATH.sub("<path>", out)
    out = _MASK_HEXID.sub("<id>", out)
    out = _MASK_TS.sub("<ts>", out)
    out = _MASK_POS.sub(lambda m: f"{m.group(1)} <n>", out)
    out = _MASK_MS.sub("<n>ms", out)
    out = _MASK_EPOCH.sub("<id>", out)
    out = _MASK_SQUOTE.sub(" <q> ", out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > SIGNATURE_MAX_CHARS:
        out = out[:SIGNATURE_MAX_CHARS - 3].rstrip() + "..."
    return out


def _structured_error(body: str) -> bool:
    """True if the body is what `agent_mcp/_shared.py::_looks_like_error_json`
    calls an error: a JSON object with a top-level `"error"` key.

    The transcripts store a *truncated* body, so a long error payload may no
    longer parse even though production flagged it by parsing it whole. The
    leading-`{` plus `"error"` gate is the same cheap test production runs
    first, and a body whose head still carries `"error"` after the truncation
    is a structured error however unreadable the tail became.
    """
    s = (body or "").lstrip()
    if not s.startswith("{") or '"error"' not in s:
        return False
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        return '"error"' in s[:400]
    return isinstance(parsed, dict) and "error" in parsed


def _bare_shell_exit(body: str) -> bool:
    """True for a non-zero exit whose payload is raw command output."""
    return bool(_SHELL_EXIT_MARKER.search(body or ""))


def classify(body: str) -> tuple[str, str]:
    """One flagged result body → (class, signature).

    `signature` is "" for a class that must not contribute one — `shell_exit`
    and `guess_class` — so a caller can add the signature to a counter and the
    class counts stay honest.

    The shell-exit marker is tested BEFORE the JSON shape on purpose. A command
    that exits non-zero and prints a JSON document on its way out — a test
    runner's failing-case dump, an eval record — ends with `[exit code: N]` and
    merely looks structured, and its cost is an exit code, not a tool defect.
    A real tool error never carries that suffix: Bash returns its JSON error
    payloads before the exit code exists (`builtin_bash.py:197` returns one,
    `:219` is where a non-zero exit is even detected).
    """
    if _bare_shell_exit(body):
        return CLASS_SHELL_EXIT, ""
    if _structured_error(body):
        return CLASS_ERROR_JSON, normalise_signature(body)
    first = _signature_line(body)
    if any(re.search(pattern, first) for pattern, _writer in HARNESS_MARKERS):
        return CLASS_HARNESS_ERROR, normalise_signature(body)
    return CLASS_GUESS, ""


def _body_text(entry: dict) -> str:
    content = entry.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return ""


def _tool_names(session: dict) -> dict[str, str]:
    """call_id → tool name, from the assistant rows that carry `tool_calls`."""
    names: dict[str, str] = {}
    for e in session.get("messages") or []:
        if not isinstance(e, dict):
            continue
        for tc in e.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = (tc.get("function") or {}).get("name") or tc.get("name") or ""
            for key in (tc.get("call_id"), tc.get("id")):
                if key:
                    names[key] = name
    return names


def build_record(sessions_dir: pathlib.Path, days: int,
                 now: float | None = None) -> dict:
    """Scan `sessions_dir` and return the aggregate record (no file I/O)."""
    # NON-recursive on purpose. Beside the transcripts, `sessions/` holds one
    # `<session-id>.tool-results/` directory per session whose truncated tool
    # results are written back as JSON — ~2,200 files as this shipped, 2.2× a
    # single window's call count, and none of them a transcript. `rglob` would
    # not slow this down, it would inflate it.
    paths = [p for p in sorted(sessions_dir.glob("*.json")) if p.is_file()]
    mtimes = {p: p.stat().st_mtime for p in paths}
    window_end = max(mtimes.values()) if mtimes else (now or 0.0)
    cutoff = window_end - days * 86400.0
    in_window = [p for p in paths if mtimes[p] >= cutoff]

    calls: Counter = Counter()
    flagged: Counter = Counter()
    sigs: dict[str, Counter] = defaultdict(Counter)
    classed: dict[str, Counter] = defaultdict(Counter)
    unjoined = 0

    for path in in_window:
        try:
            session = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(session, dict):
            continue
        names = _tool_names(session)
        for e in session.get("messages") or []:
            if not isinstance(e, dict) or e.get("role") != "tool":
                continue
            name = names.get(e.get("tool_call_id")) or ""
            if not name:
                unjoined += 1
                name = "<unjoined>"
            calls[name] += 1
            if not (e.get("stats") or {}).get("is_error"):
                continue
            flagged[name] += 1
            cls, sig = classify(_body_text(e))
            classed[name][cls] += 1
            if sig:
                sigs[name][sig] += 1

    def _iso(ts: float) -> str:
        return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat(
            timespec="seconds")

    tools: dict[str, dict] = {}
    for name in sorted(calls, key=lambda n: (-flagged[n], n)):
        counter = sigs.get(name) or Counter()
        tools[name] = {
            "calls": calls[name],
            "flagged": flagged[name],
            "flagged_rate": round(flagged[name] / calls[name], 4) if calls[name] else 0.0,
            "shell_exit": classed[name][CLASS_SHELL_EXIT],
            "guess_class": classed[name][CLASS_GUESS],
            "signatures": [{"signature": s, "count": c}
                           for s, c in counter.most_common()],
        }

    ranked = sorted(((name, s, c) for name, ctr in sigs.items()
                     for s, c in ctr.items()), key=lambda t: (-t[2], t[0], t[1]))
    date = _dt.datetime.fromtimestamp(window_end, _dt.timezone.utc).strftime("%Y-%m-%d")

    return {
        "schema": 1,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "days": days,
        "sessions_dir": str(sessions_dir),
        "window_start": _iso(cutoff) if in_window else "",
        "window_end": _iso(window_end) if in_window else "",
        "date": date,
        "sessions_scanned": len(in_window),
        "tool_calls_total": sum(calls.values()),
        "flagged_total": sum(flagged.values()),
        "guess_class_total": sum(c[CLASS_GUESS] for c in classed.values()),
        "shell_exit_total": sum(c[CLASS_SHELL_EXIT] for c in classed.values()),
        "tool_rows_without_a_call_row": unjoined,
        "classes": list(CLASSES),
        "signature_max_chars": SIGNATURE_MAX_CHARS,
        "tools": tools,
        "top_signatures": [{"tool": t, "signature": s, "count": c}
                           for t, s, c in ranked[:TOP_RANKING]],
    }


def default_out_path(record: dict, out_dir: pathlib.Path | None = None) -> pathlib.Path:
    """`eval/baselines/tool-failures/<date>.json`, keyed on the window's date.

    Subdirectory, not sibling: the scorer test globs `eval/baselines/*.json`
    and demands its newest file be a retrieval run record.
    """
    return (out_dir or DEFAULT_OUT_DIR) / f"{record['date']}.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Aggregate per-tool failure signatures from sessions/*.json "
                    "into eval/baselines/tool-failures/<date>.json.")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"Look-back window in days, anchored on the newest "
                         f"session mtime (default {DEFAULT_DAYS})")
    ap.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS_DIR),
                    help="Directory of session JSON transcripts")
    ap.add_argument("--out", default=None,
                    help="Output path (default: eval/baselines/tool-failures/"
                         "<date>.json)")
    ap.add_argument("--stdout", action="store_true",
                    help="Print the record instead of writing a file")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.days <= 0:
        print(f"--days must be positive, got {args.days}", file=sys.stderr)
        return 2
    sessions_dir = pathlib.Path(args.sessions_dir).expanduser()
    if not sessions_dir.is_dir():
        print(f"sessions dir not found: {sessions_dir}", file=sys.stderr)
        return 2
    record = build_record(sessions_dir, args.days)
    if args.stdout:
        json.dump(record, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
        return 0
    out = pathlib.Path(args.out).expanduser() if args.out else default_out_path(record)
    if out.resolve().parent == EVAL_BASELINES_DIR.resolve():
        # Writing here would make this record the newest `eval/baselines/*.json`
        # that `tests/test_eval_scorer.py::test_a_run_record_declares_whether_it_matched_production`
        # reads by mtime, and this record carries none of the retrieval knobs it
        # demands. The default path sidesteps the glob by being one level deeper;
        # an explicit `--out` has to be refused, not warned about, because the
        # test that goes red is in another file.
        print(f"refusing to write {out}: a top-level eval/baselines/*.json becomes "
              f"the newest file tests/test_eval_scorer.py reads by mtime. Use the "
              f"default (a subdirectory) or --stdout.", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{out} — {record['tool_calls_total']} calls / "
          f"{record['flagged_total']} flagged across "
          f"{record['sessions_scanned']} sessions "
          f"(shell_exit {record['shell_exit_total']}, "
          f"guess_class {record['guess_class_total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
