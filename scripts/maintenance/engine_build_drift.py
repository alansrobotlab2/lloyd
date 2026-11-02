#!/usr/bin/env python3
"""Is the engine still the build the `tool_choice:"none"` property was measured on?

    python scripts/maintenance/engine_build_drift.py
    python scripts/maintenance/engine_build_drift.py --base-url http://127.0.0.1:8096

Why this exists. `eval/tool_choice_none_probe.py` (#2678) measured on 2026-10-14 that
the primary engine renders a `tool_choice:"none"` request with the tools array attached
as a byte-identical prefix of the `"auto"` request — 7,982 prompt tokens against 7,639
with `tools` dropped, 343 tokens of schema still rendered, and no tool call emitted.
P6(b) of `architecture/harness.md` rests the max-turns wrap-up on that: the wrap-up
request must continue the turn's prefix, not fork it.

The measurement is per-BUILD, and the same document says so — "the property is per-build,
so re-run it after every engine rebuild and record the build beside the result". Nothing
enforced the "after every engine rebuild". `tool_choice_none_probe` is named by prose and
by its own tests and by NO runner: no file under `workers/`, no job in the autonomy
tasks, no `config.yaml` entry. So the re-run was a sentence in a document, and the drift
that makes it false is silent — an engine that starts rendering a `"none"` request
differently does not fail, error, or log. The next turn still completes; the only
symptom is the prompt-token count on a wrap-up request, which is visible in `usage.db` if
someone happens to look at the one turn in which the budget ran out, and invisible
otherwise. The wrap-up could keep forking the prefix for weeks while every document
describing it stayed confidently up to date.

That is the class: a measurement carrying a build string, and no code comparing that
string to the build the box now serves. This is that comparison. It is not a probe — it
makes ONE `GET /version`, the same read the probe's own build reader makes — and it never
issues a completion request, so it costs no GPU and cannot perturb the prefix-cache
numbers it exists to protect.

The expected build is PARSED out of `app/harness/finalizer.py`'s module docstring, which
already carries it on its `Re-verified <date> on <build>` line, and is never re-spelled
here as a second literal. Two copies of a build string are the failure this script exists
to catch, told from the inside: `finalizer.py`'s docstring once said "the installed vLLM
0.28.1" and became false the day the box moved, with no test to notice
(`app/harness/tests/test_finalizer.py:879`). Editing the docstring moves the expectation
with it — and editing that docstring is exactly what a re-measurement does, so the check
can never disagree with the record it grades. The install-side pin (`VLLM_PIN=` in
`agent-services/setup/`) is deliberately NOT read: that is the string a rebuild changes,
and comparing a record against a record would drift precisely when the docstring does
not, which is the day this matters.

Exit codes, so a job can gate on it and Mission Control can show the state:

    0  PASS      the live build is the recorded one
    1  DRIFT     the engine answers, and its build is not the recorded one
    2  UNKNOWN   the engine would not name its build, or the record names none

UNKNOWN exits non-zero on purpose. A check that cannot see cannot certify a property, and
"the engine is not answering" reads exactly like "the engine is unchanged" if it is
allowed to exit 0.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The record. One docstring line is the whole expectation, and the date on it is the
#: day the probe was last run.
FINALIZER = ROOT / "app/harness/finalizer.py"
#: The instrument a mismatch sends a human back to. Named in the DRIFT line because a
#: mismatch is not a repair: the property may well still hold on the new build, and only
#: re-running this can say so.
PROBE = "eval/tool_choice_none_probe.py"
#: Same default the probe's own build reader uses, which is `agent_llm.base_url`.
DEFAULT_BASE_URL = "http://127.0.0.1:8096"
#: `<date> on <build>` after the words `Re-verified`. The build is captured as `(\S+)`
#: rather than matched against a version grammar because the job is to copy whatever the
#: record says, byte for byte: a setuptools-scm version's local segment IS the commit, so
#: whole-string equality IS the test — and a parse-then-normalise comparison would make a
#: rebuild that changed only the commit hash look like a match.
RECORDED_RE = re.compile(r"Re-verified\s+(\d{4}-\d{2}-\d{2})\s+on\s+(\S+)")

PASS, DRIFT, UNKNOWN = 0, 1, 2


def module_docstring(path: Path) -> str:
    """The module docstring of `path`, or "" when it has none or will not parse.

    Read with `ast`, not grepped, so a line inside a comment or a string literal can
    never become the expectation: the thing under record is a docstring, and reading it
    as syntax is what keeps "parse the record" from meaning "match any line that looks
    like the record".
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return ""
    return ast.get_docstring(tree) or ""


def parse_recorded(text: str) -> tuple[str, str] | None:
    """`(<measured-on date>, <build>)` out of a docstring's text, or None if it names no build."""
    m = RECORDED_RE.search(text)
    return (m.group(1), m.group(2)) if m else None


def recorded(path: Path = FINALIZER) -> tuple[str, str] | None:
    """The build the record says the property was last measured on."""
    return parse_recorded(module_docstring(path))


def live_build(base_url: str, *, timeout: float = 5.0) -> str | None:
    """The build the engine names, or None when it will not say.

    The same one request the probe's own build reader makes — `GET <base>/version`, read
    `["version"]` — so this check sees what the probe saw, from the same direction. A
    refused socket, a timeout, a body that is not JSON, a JSON object with no `version`
    or an empty one: all None, all UNKNOWN, never a mismatch. A build we could not read
    cannot be diffed against anything, and reporting a phantom drift would send a human
    off to re-measure a property that is fine.
    """
    req = urllib.request.Request(base_url.rstrip("/") + "/version",
                                 headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    value = body.get("version") if isinstance(body, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def verdict(got_recorded: tuple[str, str] | None, live: str | None
            ) -> tuple[int, str]:
    """(exit code, one-line report) over the two build strings and nothing else.

    The module's ONLY decision point, and pure by construction: it takes the recorded
    `(date, build)` pair and the live build, does no I/O, reads no clock, no environment,
    no file, no socket. That is what lets every case be decided offline over synthetic
    pairs, which is what lets the PASS case be tested at all — a test that pinned the
    build serving this box today would go red the moment the box is rebuilt, which is the
    precise moment the check is supposed to be right about something else.

    Absence on either side is UNKNOWN, not a difference. `got_recorded is None` means
    `finalizer.py` stopped carrying a build on its `Re-verified` line — a re-measurement
    that recorded the date and forgot the build, the exact defect this script exists to
    catch, caught here on the record side — so there is nothing to compare against and no
    pass to claim.
    """
    if got_recorded is None:
        return (UNKNOWN,
                "engine-build-drift UNKNOWN: no `Re-verified <date> on <build>` line in "
                f"{FINALIZER.relative_to(ROOT)}'s module docstring, so no recorded build "
                f"to compare. Not a pass: re-run {PROBE} and record the build beside the "
                "result.")
    date, want = got_recorded
    if live is None or not live.strip():
        return (UNKNOWN,
                "engine-build-drift UNKNOWN: the engine would not name its build (no "
                f"`GET /version`, or a reply with no `version`), so the recorded {want} "
                f"(measured {date}) is neither confirmed nor contradicted. Not a pass.")
    if live == want:
        return (PASS,
                f"engine-build-drift PASS: live build {live} is the build {PROBE} was "
                f"measured on ({date}) — the tool_choice:\"none\" prefix-equality "
                "measurement still describes this engine.")
    return (DRIFT,
            f"engine-build-drift DRIFT: recorded build {want} (measured {date}), live "
            f"build {live}. The tool_choice:\"none\" prefix-equality property is "
            f"per-build, so re-run {PROBE} against {live} and record the new build "
            "beside the result: P6(b)'s max-turns wrap-up is resting on a measurement "
            "of an engine that no longer answers.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help=f"engine to ask for its build (default {DEFAULT_BASE_URL})")
    ap.add_argument("--finalizer", type=Path, default=FINALIZER,
                    help="file whose module docstring carries the recorded build")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="seconds for the one /version request (default 5)")
    a = ap.parse_args(argv)
    code, line = verdict(recorded(a.finalizer), live_build(a.base_url,
                                                           timeout=a.timeout))
    print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
