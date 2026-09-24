"""#1421: `eval/desktop_grounding/` was cited as the authority that chose
`desktop.coordinate_space`, and it does not exist. The code comment now says
the value is an untested default; the architecture doc keeps saying the
instrument is unbuilt, so neither can rot back into a promise.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "agent_mcp" / "desktop" / "__init__.py"
DOC = ROOT / "architecture" / "desktop.md"


def _section(text: str, heading_prefix: str) -> str:
    m = re.search(rf"^## {re.escape(heading_prefix)}.*?$(.*?)(?=^## |\Z)",
                  text, re.MULTILINE | re.DOTALL)
    assert m, heading_prefix
    return m.group(1)


def test_module_cites_no_grounding_eval_and_calls_the_default_untested():
    src = MODULE.read_text()
    doc = DOC.read_text()
    # Positive control: the same needle is found in the doc, so an absent
    # hit in the module is not a wrong path or a wrong spelling.
    assert "eval/desktop_grounding" in doc
    assert "eval/desktop_grounding" not in src
    lines = src.splitlines()
    i = next(n for n, ln in enumerate(lines) if '"coordinate_space":' in ln)
    comment = " ".join(ln.strip() for ln in lines[max(0, i - 6):i]
                       if ln.strip().startswith("#"))
    assert "untested default" in comment


def test_doc_still_says_the_instrument_is_unbuilt():
    doc = DOC.read_text()
    s2 = re.sub(r"\s+", " ", _section(doc, "2."))
    assert re.search(r"nothing has measured which space", s2)
    s7 = _section(doc, "7. Not built yet")
    assert "eval/desktop_grounding/" in s7 and "#1421" in s7
