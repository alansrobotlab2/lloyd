"""`architecture/background-runs.md` §8's claims about the chat summarizer (#1228).

The paragraph said `_summarize_chat` ran on every `POST /api/mc/state`, opened
every file with no ceiling, and cost 238 ms at 1,420 files. The first two were
false and the third was a carried-over figure from a directory five months
younger. What it may say is pinned here as behaviour the doc must describe.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "background-runs.md"


def _paragraph() -> str:
    text = DOC.read_text(encoding="utf-8")
    assert text.count("_summarize_chat") >= 1, "the doc no longer names the chat summarizer"
    for para in re.split(r"\n\s*\n", text):
        if "_summarize_chat" in para:
            return " ".join(para.split())
    raise AssertionError("unreachable")


def test_the_falsified_claims_are_gone():
    text = DOC.read_text(encoding="utf-8")
    for stale in ("238 ms", "1,420", "#1017", "POST /api/mc/state"):
        assert stale not in text, stale
    para = _paragraph()
    for stale in ("no ceiling", "opens every file"):
        assert stale not in para, stale


def test_the_paragraph_states_its_call_path_scaling_and_a_dated_figure():
    para = _paragraph()
    assert "only from `mc_navigate`" in para
    assert "scales with the background share" in para
    ms = re.search(r"~?\d+(?:\.\d+)? ms", para)
    assert ms, para
    assert re.search(r"\b20\d\d-\d\d-\d\d\b", para), "a timing needs the date it was measured on"
