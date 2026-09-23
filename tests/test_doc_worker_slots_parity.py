"""`architecture/workers.md` may not state a `workers.slots` that contradicts config.yaml.

#1338: the doc documented `slots: 2` while config.yaml had been 6 since
`e009808` (2026-09-18, "autocode went to four rounds, and the rule is rounds +
triages + 1"). Nothing joined the two, so the stale number sat in the docs the
whole time — and a measured invariant that only lives in prose is exactly the
thing that drifts, which is why the fix is a check and not a one-off edit.

config.yaml is authoritative. The doc is a copy of the shape; if it disagrees,
the doc is wrong, so this test reads the value out of the §6 sample block and
against the live config and refuses a mismatch. It matches on the fenced block
under "## 6. Configuration" specifically — a bare grep for "slots" would also
catch the prose in §1 and the queue-worker prose in §3.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "workers.md"
CONFIG = ROOT / "config.yaml"


def _config_slots() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    slots = cfg["workers"]["slots"]
    assert isinstance(slots, int), f"config workers.slots is not an int: {slots!r}"
    return slots


def _doc_slots() -> int | None:
    """The `slots:` value in the fenced YAML block under §6 Configuration."""
    text = DOC.read_text(encoding="utf-8")
    m = re.search(r"^##\s+6\.\s+Configuration\s*$", text, re.M)
    assert m, "§6 Configuration heading disappeared from workers.md"
    tail = text[m.end():]
    fence = re.search(r"```(?:yaml|yml)?\n(.*?)\n```", tail, re.S)
    assert fence, "no fenced config block under §6 Configuration"
    found = re.search(r"^\s*slots:\s*(\d+)", fence.group(1), re.M)
    return int(found.group(1)) if found else None


def test_doc_config_block_declares_a_slots_value():
    assert _doc_slots() is not None, "workers.slots vanished from the §6 config sample"


def test_doc_worker_slots_matches_config():
    doc = _doc_slots()
    cfg = _config_slots()
    assert doc == cfg, (
        f"architecture/workers.md §6 shows workers.slots: {doc} but config.yaml "
        f"has {cfg}. config.yaml is authoritative (#1338) — fix the doc, and see "
        f"tests/test_loop_depth.py for why config's value is what it is "
        f"(rounds + triages + 1)."
    )
