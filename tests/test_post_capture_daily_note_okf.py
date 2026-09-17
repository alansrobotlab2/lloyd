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


# --- Item #601 (umbrella #1189): the heading's zone word comes from the clock -----
# Until now the suffix was the literal `PDT` in one f-string, so the whole winter
# archive — first Sunday in November to second Sunday in March, when
# America/Los_Angeles is PST (UTC−8) — mislabelled every auto-captured section.
# A live `datetime.now` cannot show that in summer, so these pins freeze instants
# and read the zone word off the same `strftime('%Z')` that produced the clock.
# The frozen instants below are LA-local, exactly what `datetime.now(pst)`
# returns: each carries its own fold, so `strftime('%Z')` answers for the zone
# and nothing in the test supplies the expected abbreviation to the writer.

_JANUARY = datetime(2026, 1, 15, 9, 5, tzinfo=ZoneInfo("America/Los_Angeles"))
_AUGUST = datetime(2026, 8, 15, 9, 5, tzinfo=ZoneInfo("America/Los_Angeles"))


def test_frozen_january_instant_heading_never_emits_PDT(memory_dir):
    """The bug itself, frozen: a January 09:05 LA instant is PST, not PDT.

    The reproduction command from #601 formatted a fixed winter date and got
    `PST | ### Session 09:05 PDT — Auto-captured` — the label contradicting the
    zone it came from. This asserts the contradiction is impossible: the
    heading carries the instant's own `%Z`, so it can bear no zone token that
    disagrees with its own clock reading.
    """
    assert _JANUARY.strftime("%Z") == "PST", (
        "the frozen instant is not what it claims to be; the tz database must "
        "still resolve January in America/Los_Angeles to PST"
    )
    post_capture._append_daily_note("sess-winter", "January body.", now=_JANUARY)
    text = (memory_dir / "2026-01-15.md").read_text(encoding="utf-8")
    assert "### Session 09:05 PST — Auto-captured" in text, (
        f"the winter heading did not take the instant's own zone: {text!r}"
    )
    assert "PDT" not in text, (
        f"a January capture still emitted the summer abbreviation: {text!r}"
    )


def test_frozen_august_instant_heading_keeps_the_summer_label(memory_dir):
    """Summer output is unchanged, byte for byte, by the fix.

    `PDT` is not banned from the heading — it is banned from being *typed*. A
    mid-August LA instant must still print `PDT`, or this fix would have
    traded one wrong season for the other.
    """
    assert _AUGUST.strftime("%Z") == "PDT"
    post_capture._append_daily_note("sess-summer", "August body.", now=_AUGUST)
    text = (memory_dir / "2026-08-15.md").read_text(encoding="utf-8")
    assert "### Session 09:05 PDT — Auto-captured" in text


def test_live_heading_shape_and_single_clock_reading(memory_dir):
    """The two invariants the clause carries for the *live* path.

    1. Shape: the heading still matches `^### Session \\d{2}:\\d{2} ` and ends
       ` — Auto-captured` — the grep shape of every daily-note section (74 in
       memory/2026-09-09.md at member #601's measurement) — and its zone word
       equals `strftime('%Z')` of the reading that set the filename's date, so
       the two can never disagree.
    2. One clock: the LA-date filename and a brand-new file's `timestamp:`
       frontmatter come from the same reading passed down as `now`. They
       previously came from three separate `datetime.now` calls that could
       straddle midnight; here both are derived from `live` below, so a
       date disagreement between filename and timestamp fails this at the
       straddle instead of silently misfiling the note.
    """
    live = datetime.now(ZoneInfo("America/Los_Angeles"))
    post_capture._append_daily_note("sess-live", "Live-shaped body.")
    note = memory_dir / f"{live.strftime('%Y-%m-%d')}.md"
    assert note.exists(), "live-shaped capture did not create the LA-dated note"
    text = note.read_text(encoding="utf-8")

    heading = re.search(r"^### Session .*$", text, re.MULTILINE)
    assert heading, f"no auto-captured heading in:\n{text[:400]}"
    line = heading.group(0)
    m = re.match(r"^### Session (\d{2}:\d{2}) (\S+) — Auto-captured$", line)
    assert m, f"heading lost the daily-note section shape: {line!r}"
    hhmm, zone = m.groups()
    assert hhmm == live.strftime("%H:%M"), (
        f"heading clock {hhmm} disagrees with the reading {live:%H:%M}"
    )
    assert zone == live.strftime("%Z"), (
        f"heading zone {zone!r} disagrees with the instant's own zone "
        f"{live.strftime('%Z')!r} — the zone word must be strftime('%Z'), "
        "never a typed literal"
    )
    assert _frontmatter(note)["timestamp"].startswith(live.strftime("%Y-%m-%d")), (
        "fresh-file timestamp and LA-date filename came from different readings"
    )
