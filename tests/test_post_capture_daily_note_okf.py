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
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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


# --- item #1159: re-arming fact extraction must not re-open the daily note -----
#
# The daily-note half of post_capture is once-per-session; the fact half now
# re-arms on a per-session watermark. Both live in `_post_session_capture`, and
# the whole point of splitting their gates is that extraction runs again on a
# session whose summary is already written. So the regression this pins is the
# obvious one: a second extraction pass that also re-summarises would put a
# second `### Session HH:MM ZONE — Auto-captured` section in the user's daily
# note — an OKF-shape change to the file daily notes are grepped by, and a
# duplicate recap of the same conversation. `tests/test_post_capture_fact_rearm.py`
# pins the extraction half; this pins that it costs the note nothing.

from app import sessions_io as _sio

_OKF_SID = "20260920_120000_bbbbbb"


def _okf_user_turn(n: int) -> dict:
    return {
        "id": f"u{n}",
        "role": "user",
        "content": [{"type": "text", "text": f"Turn {n}: the tts service listens on port 8090."}],
        "timestamp": "2026-09-20T12:00:00",
    }


def _okf_assistant_turn(n: int) -> dict:
    return {
        "id": f"a{n}",
        "role": "assistant",
        "content": [{"type": "text", "text": f"Noted turn {n}, port 8090 recorded."}],
        "timestamp": "2026-09-20T12:00:01",
    }


@pytest.fixture
def session_dir(memory_dir, tmp_path, monkeypatch):
    """A session file plus temp `~`, with both secondary engines and fact_add replaced.

    The real ones are HTTP to a single-slot llama.cpp, and `_fact_add` appends to
    the live fact store, so neither belongs in a suite run.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(post_capture, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(_sio, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(post_capture, "VAULT_SESSIONS_DIR", tmp_path / "vault-sessions")
    monkeypatch.setattr(post_capture, "VAULT_BACKGROUND_SESSIONS_DIR",
                        tmp_path / "vault-background")

    calls: dict = {"summary": [], "facts": []}
    monkeypatch.setattr(
        post_capture, "_sync_secondary_capture_call",
        lambda t: calls["summary"].append(t) or "Talked through the tts port.",
    )
    monkeypatch.setattr(
        post_capture, "_sync_secondary_fact_extraction",
        lambda t: calls["facts"].append(t) or [],
    )

    import agent_mcp.facts as facts_mod
    monkeypatch.setattr(facts_mod, "_fact_add", lambda payload: {"success": True})

    def write(messages: list[dict]) -> None:
        path = sessions / f"{_OKF_SID}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.update({
            "session_id": _OKF_SID,
            "created_at": "2026-09-20T12:00:00",
            "model": "primary",
            "messages": messages,
        })
        path.write_text(json.dumps(data), encoding="utf-8")

    def read() -> dict:
        return json.loads((sessions / f"{_OKF_SID}.json").read_text(encoding="utf-8"))

    return SimpleNamespace(calls=calls, write=write, read=read)


async def test_rearming_extraction_adds_no_second_auto_captured_section(
        memory_dir, session_dir):
    """Clause 4: one `Auto-captured` section per session, however often facts re-arm."""
    session_dir.write([_okf_user_turn(1), _okf_assistant_turn(1)])
    await post_capture._post_session_capture(_OKF_SID)

    note = _today_note(memory_dir)
    assert note.exists(), "the first non-trivial pass wrote no daily note"
    first = note.read_text(encoding="utf-8")
    assert first.count("— Auto-captured") == 1, f"first pass wrote more than one section:\n{first[:600]}"
    assert session_dir.read()["captured"] is True, "the summary half must latch `captured`"
    assert len(session_dir.calls["summary"]) == 1
    assert session_dir.calls["facts"] == [], "two user turns is below the ≥3 gate"

    # Now grow the session past the extraction gate and run the pass again: this
    # is the state #1159 is about — `captured` already latched by turn 1.
    grown = ([_okf_user_turn(1), _okf_assistant_turn(1)]
             + [_okf_user_turn(n) for n in (2, 3)]
             + [_okf_assistant_turn(2), _okf_assistant_turn(3)])
    session_dir.write(grown)
    await post_capture._post_session_capture(_OKF_SID)

    after = note.read_text(encoding="utf-8")
    assert after.count("— Auto-captured") == 1, (
        "re-arming fact extraction appended a second summary section to the daily note"
    )
    assert len(session_dir.calls["summary"]) == 1, "the session was summarised twice"
    assert len(session_dir.calls["facts"]) == 1, (
        "three new user messages past a zero watermark must spend exactly one "
        "extraction call"
    )
