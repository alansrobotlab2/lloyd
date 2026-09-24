"""#1125: the tool layer's comments must describe the argument contract the
code actually has.

Several comments said the aggregator (or "the SDK") validates a call's
`arguments` against the tool's inputSchema before the handler runs, so an
unknown key is a dispatch error. Nothing does: the low-level MCP server
validates the JSON-RPC params model only, `mcp_pool._coerce_args` coerces
top-level primitives and passes everything else through, and an unknown key
reaches the handler (pinned behaviourally by
`tests/test_mcp_layer.py::test_undeclared_argument_reaches_handler_and_is_ignored`).
A change designed on the old claim relies on a rejection that never happens.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATHS = [
    "app/harness/tool_schema.py",
    "agent_mcp/main.py",
    "app/harness/mcp_pool.py",
    "app/harness/loop.py",
    "CLAUDE.md",
]

# The retired claim, in every spelling it took: "<X> validates `args`/arguments
# against the/each tool's (real) inputSchema", or "is validated against each
# tool's inputSchema", or an unknown key being "a dispatch error".
_CLAIM = re.compile(
    r"(?<!nothing )(?<!Nothing )validates\s+`?(?:args|arguments)`?\s+against"
    r"|`?(?:args|arguments)`?\s+is\s+validated\s+against"
    r"|is\s+a\s+dispatch\s+error",
    re.IGNORECASE,
)


def _joined(path: str) -> str:
    """Source with comment markers and line breaks folded into single spaces,
    so a sentence wrapped across comment lines still matches as one."""
    text = (ROOT / path).read_text()
    text = re.sub(r"\n\s*(?:#\s?)?", " ", text)
    return re.sub(r"\s+", " ", text)


def test_no_path_claims_arguments_are_schema_validated():
    offenders = {}
    for p in PATHS:
        hits = [m.group(0) for m in _CLAIM.finditer(_joined(p))]
        if hits:
            offenders[p] = hits
    assert offenders == {}


def test_the_real_contract_is_stated_where_the_claim_was():
    # The sentence that replaced the claim, in each place it stood.
    for p in PATHS:
        text = _joined(p)
        assert re.search(r"[Nn]othing\s+validates\s+`?(?:args|arguments)`?", text), p
        assert "unknown keys" in text or "unknown key" in text, p
    # `_coerce_args` names itself as the only shape defense.
    pool = _joined("app/harness/mcp_pool.py")
    assert "only shape defense" in pool


def test_the_scan_catches_the_retired_sentence():
    # Positive control: the pattern matches the claim as it used to read, so
    # a green first test is not a pattern that matches nothing.
    old = ("the aggregator validates arguments against each tool's real "
           "inputSchema, so an unknown key there is a dispatch error")
    assert len(_CLAIM.findall(old)) == 2
    old2 = "the SDK validates `arguments` against the tool's inputSchema"
    assert _CLAIM.search(old2)
    assert not _CLAIM.search("Nothing validates `arguments` against the inputSchema")
