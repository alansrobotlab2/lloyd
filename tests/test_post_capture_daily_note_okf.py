"""Item #519 — auto-capture must not create an OKF violation at birth.

`_append_daily_note` creates today's daily note when none exists yet. Until
item #519 it wrote a bare `segment: agents` header with no `type`, so every
calendar day whose first captured session started a fresh file also started a
fresh OKF violation — `scripts/vault/validate_okf.py --dir memory` reports
those files as "missing/empty `type`", and 2026-09-04 … 2026-09-09 are the
six this writer produced before the fix.

These tests pin the emitted header against the same strict frontmatter form the
conformance gate uses (`^---\\n(.*?)\\n---\\n`, DOTALL), so a regression fails in
the suite rather than at the next midnight.
"""
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from app import post_capture

# Byte-identical to STRICT_FM_RE in scripts/vault/validate_okf.py — the check a
# newly created daily note has to survive.
STRICT_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    """An empty vault segment, via a temp `~` (the writer resolves Path.home())."""
    mem = tmp_path / "obsidian" / "memory"
    mem.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("USERPROFILE", raising=False)
    return mem


def _today_note(memory_dir: Path) -> Path:
    today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    return memory_dir / f"{today}.md"


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = STRICT_FM_RE.match(text)
    assert m, f"daily note has no strict-parseable frontmatter:\n{text[:200]}"
    fm = yaml.safe_load(m.group(1))
    assert isinstance(fm, dict), f"frontmatter is not a mapping: {fm!r}"
    return fm


def test_created_daily_note_frontmatter_declares_type(memory_dir):
    """The OKF requirement itself: parseable frontmatter with a non-empty type."""
    post_capture._append_daily_note("sess-a", "Summarised what happened.")
    note = _today_note(memory_dir)
    assert note.exists(), "first capture did not create the daily note"

    fm = _frontmatter(note)
    assert fm.get("type") == "note"


def test_created_daily_note_matches_established_convention(memory_dir):
    """Segment/tags/timestamp match the conformant reference memory/2026-06-14.md.

    The file lives under `memory/`, so `segment: agents` was wrong as well as
    incomplete; timestamp is what makes the note sortable by the KG index.
    """
    post_capture._append_daily_note("sess-a", "Summarised what happened.")
    fm = _frontmatter(_today_note(memory_dir))

    assert fm.get("segment") == "memory"
    assert fm.get("tags") == ["memory", "daily-notes"]
    assert isinstance(fm.get("timestamp"), str) and fm["timestamp"], (
        f"timestamp missing or not a string: {fm.get('timestamp')!r}"
    )
    # Strict YAML timestamp, parseable as an instant, not free text.
    assert datetime.fromisoformat(fm["timestamp"])


def test_created_daily_note_still_contains_the_captured_section(memory_dir):
    """Fixing the header must not lose the entry or the heading consumers grep."""
    post_capture._append_daily_note("sess-a", "First session body.")
    note = _today_note(memory_dir)
    text = note.read_text(encoding="utf-8")

    assert "Daily Notes" in text
    assert "## Sessions" in text
    assert "Auto-captured" in text
    assert "First session body." in text


def test_second_capture_appends_and_keeps_one_header(memory_dir):
    post_capture._append_daily_note("sess-a", "First session body.")
    post_capture._append_daily_note("sess-b", "Second session body.")
    note = _today_note(memory_dir)
    text = note.read_text(encoding="utf-8")

    assert text.count("---\nsegment: memory\n") == 1, "header rewritten on append"
    fm = _frontmatter(note)
    assert fm.get("type") == "note"
    assert "First session body." in text and "Second session body." in text


def test_existing_note_header_is_left_alone(memory_dir):
    """A pre-existing daily note (any header) is appended to, never rewritten."""
    note = _today_note(memory_dir)
    note.write_text(
        "---\nsegment: memory\ntype: note\n---\n\n# Today Daily Notes\n\n"
        "## Sessions\n\n### Session 08:00 PDT — hand written\n\nManual.\n",
        encoding="utf-8",
    )

    post_capture._append_daily_note("sess-a", "Captured later.")
    text = note.read_text(encoding="utf-8")

    assert "Manual." in text
    assert "Captured later." in text
    assert _frontmatter(note) == {"segment": "memory", "type": "note"}
