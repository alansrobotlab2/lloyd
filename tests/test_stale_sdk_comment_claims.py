"""#932: `claude_agent_sdk` is gone from the tree, and so are the comments that named it.

The harness replaced the SDK (`app/harness/`), and its module docstrings say
what each piece replaces — those are the only legitimate mentions left. Three
deferred-import comments kept giving "avoid loading claude_agent_sdk" as their
reason long after nothing imported it, so the reason they stated was false
while the deferral itself was still right.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "claude_agent_sdk"


def _tracked_py_hits() -> dict[str, list[str]]:
    out = subprocess.run(["git", "-C", str(ROOT), "grep", "-n", NAME, "--", "*.py"],
                         capture_output=True, text=True)
    hits: dict[str, list[str]] = {"harness": [], "other": []}
    for line in out.stdout.splitlines():
        path = line.split(":", 1)[0]
        if path.startswith(".venvs/"):
            continue
        hits["harness" if path.startswith("app/harness/") else "other"].append(line)
    return hits


def test_no_tracked_python_outside_the_harness_names_the_sdk():
    hits = _tracked_py_hits()
    # The corpus is real: the harness docstrings still name what they replace,
    # so a zero below cannot come from a grep that searched nothing.
    assert len(hits["harness"]) >= 4, hits["harness"]
    # This file names it too, as the thing it searches for.
    other = [h for h in hits["other"] if not h.startswith("tests/test_stale_sdk_comment_claims.py:")]
    assert other == []


def test_the_deferred_imports_still_defer_and_still_say_why():
    ar = (ROOT / "agent_mcp/autoresearch.py").read_text()
    assert "Keep imports lazy inside handlers" in ar
    src = (ROOT / "workers/sources/autoresearch.py").read_text()
    body = src.split("async def execute", 1)[1]
    assert "# Lazy import" in body
    assert "from scripts.autoresearch.run_round import run as run_round" in body
    test = (ROOT / "tests/test_autoresearch_variant_pairs.py").read_text()
    assert "lazily inside its body" in test and "module attribute it re-reads" in test
