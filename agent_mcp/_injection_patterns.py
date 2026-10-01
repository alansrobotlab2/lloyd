"""The one table of instruction-shape regex families (#1959).

Two readers test text against these, and they make **different judgements on
purpose** — so each declares its own subset where it lives, by family id:

- `agent_mcp/session.py::INJECTION_GATE_FAMILIES` **refuses**: a `memory_add` /
  `memory_replace` entry matching one is returned as `ErrorCode.INJECTION`.
- `agent_mcp/_injection_probe.py::PROBE_FAMILIES` **records** (and in `warn`
  mode appends a warning): fetched text in a background session.

They were two independent regex literals whose only stated relation — the
probe's docstring saying it "mirrors" the gate — was false in both directions.
The danger of that sentence is the edit it invites: "syncing" the lists widens a
refusal. So the relation is declared here instead and pinned by
`tests/test_injection_pattern_relation.py`:

- The gate's shapes are the narrow ones, kept **verbatim** (the `*_strict` and
  `*_unanchored` families). Changing one changes what a chat turn can write to
  loaded memory, which is a ruling, not a tidy-up.
- The probe's broader families cover each of those shapes' canonical form, and
  add the ones only a recorder can afford (`role_header`, `you_must_now`,
  `run_the_following`, `conceal_from_user`, previous-class-free
  `ignore_instructions`).
- `invisible_chars` is the one family both read. It includes NUL: the gate
  always refused it, and until #1959 the probe's copy did not, so the gate's
  strictest shape was invisible in fetched text.

Stdlib only; imported by both readers and by nothing else.
"""

from __future__ import annotations

import re

FAMILIES: dict[str, re.Pattern[str]] = {
    # ── read by the probe ────────────────────────────────────────────────────
    "role_header": re.compile(r"^[ \t]*(?:system|assistant)[ \t]*:", re.I | re.M),
    "ignore_instructions": re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:of\s+)?(?:the\s+|your\s+)?"
        r"(?:previous\s+|prior\s+|above\s+|earlier\s+)?instructions\b", re.I),
    "you_must_now": re.compile(r"\byou\s+must\s+now\b", re.I),
    "run_the_following": re.compile(r"\brun\s+the\s+following\b", re.I),
    "conceal_from_user": re.compile(
        r"\b(?:do\s+not|don'?t|never)\s+(?:tell|inform|mention\s+(?:this\s+)?to)\s+the\s+user\b",
        re.I),
    "persona_swap": re.compile(
        r"\byou\s+are\s+now\s+a\b|\bpretend\s+you\s+are\b", re.I),
    "new_system_prompt": re.compile(r"\bnew\s+system\s+prompt\b", re.I),
    # ── read by both ─────────────────────────────────────────────────────────
    "invisible_chars": re.compile("[\x00​‌‍⁠﻿]"),
    # ── read by the memory gate: its five other shapes, verbatim ─────────────
    "ignore_previous_strict": re.compile(
        r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    "disregard_strict": re.compile(
        r"disregard\s+(your\s+)?(previous\s+)?instructions", re.I),
    "you_are_now_a_unanchored": re.compile(r"you\s+are\s+now\s+a", re.I),
    "pretend_you_are_unanchored": re.compile(r"pretend\s+you\s+are", re.I),
    "new_system_prompt_unanchored": re.compile(r"new\s+system\s+prompt", re.I),
}


def select(ids: tuple[str, ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """`(id, regex)` for each named family, in the order named. A reader that
    names a family the table does not hold fails at import, not at first use."""
    return tuple((fid, FAMILIES[fid]) for fid in ids)
