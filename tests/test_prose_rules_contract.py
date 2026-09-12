"""Backlog #431 — one prose ban-list, and every knowledge-note writer can reach it.

Volume decided this was worth doing, not taste: the triage measured 2,205 notes
under ``knowledge/`` newer than 2026-09-01, 807 of them since 09-11, and 434 of
those 807 (54 %) written by the ``youtube-digest`` source alone. None of the
writers carried a style contract. ``knowledge/KNOWLEDGE_SCHEMA.md`` had seven
sections — Page Types, Required Frontmatter, Naming Conventions, Page
Structure, Cross-Reference Conventions, Operations Log, Compounding Rule — and
a keyword grep for ``prose|voice|tone|style|hedg|filler|slop`` over it and over
``skills/*/SKILL.md`` returned exactly one file, ``ml-paper-writing``. So
machine-generated notes opened with *"In this video, the speaker dives into the
evolving landscape of …"* at scale, and nothing downstream noticed, because
there was no rule to violate.

The fix is one ban-list in the schema **plus a copy inlined in each writer**.
Inlined, not pointed at: ``grep -c KNOWLEDGE_SCHEMA`` is 0 in
``intelligence-pipeline``, ``ai-engineer-monitor`` and ``documentation-digester``,
so a one-line pointer to the schema is a reference to a file the writer never
opens — which is the whole reason clause 3 is written the way it is.

Pinned here, one test per acceptance clause:

1. the schema carries exactly one ``## Prose Rules`` section and the
   pre-existing sections are all still present, in order, unaltered.
2. that section is a ban-list with at least two bad→good example pairs and
   stays ≤40 lines — the writers are already long prompts, so the size is a
   requirement, and pinning it is what keeps this a ban-list instead of an
   essay that grows.
3. each of the four writer skills carries the string ``Prose Rules`` **and**
   concrete banned phrases, so the contract is executable from the skill alone.
4. the ``youtube-digest`` writing session is reached. Asserted on the prompt
   ``execute()`` hands to ``run_prompt_in_session`` — the seam the note is
   actually written behind — not merely on the ``PROMPT`` constant, since a
   constant the dispatcher does not forward would pass the weaker check.

Two clauses #431 leaves to a person are deliberately **not** asserted: reading
5 post-change notes for whether filler openings actually stopped (needs
post-landing traffic and an ear), and the final choice of which phrases are
banned (that is Alan's style preference, not a mechanical property). This file
pins that the contract exists and reaches its readers, which is the part a test
can decide.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from workers.queue import QueueItem
from workers.sources import youtube_digest as Y

VAULT = Path.home() / "obsidian"
SCHEMA = VAULT / "knowledge" / "KNOWLEDGE_SCHEMA.md"
SKILLS = VAULT / "skills"

#: The four writers #431 names. Every one of them generates knowledge prose on
#: a schedule (#30 daily, #53 daily, the youtube-digest worker source, and the
#: ai-engineer-monitor operator fallback).
WRITERS = (
    "intelligence-pipeline",
    "nightly-reflection-knowledge-write",
    "ai-engineer-monitor",
    "documentation-digester",
)

# The ban-list, as classes. The schema is the source of truth for wording; these
# are the spellings the contract is checked against, split so a writer cannot
# satisfy clause 3 by copying one throwaway phrase out of the list.
FILLER_OPENINGS = [
    "In this video",
    "In today's video",
    "This video provides a comprehensive overview of",
    "It's worth noting",
    "It is important to note",
    "Let's dive in",
    "In conclusion",
    "All in all",
    "At the end of the day",
]
BANNED_VOCABULARY = [
    "delve",
    "the evolving landscape",
    "a testament to",
    "seamless",
    "cutting-edge",
    "game-changer",
    "revolutionize",
]
HEDGE_STACKS = ["may potentially", "could possibly", "might perhaps"]

#: Sections that existed before #431. Clause 1 says the new section sits *inside*
#: the schema's existing list, so these must survive in order and untouched.
PREEXISTING_SECTIONS = [
    "## Page Types",
    "## Required Frontmatter",
    "## Naming Conventions",
    "## Page Structure",
    "## Cross-Reference Conventions",
    "## Operations Log",
    "## Compounding Rule",
]

MAX_SECTION_LINES = 40  # clause 2, verbatim


def _section(text: str, heading: str) -> str:
    """Body of one ``## `` section, up to the next one (same reader as the
    OKF taxonomy test's ``_schema_section``)."""
    start = text.index(heading) + len(heading)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _banned_present(text: str, phrases: list[str]) -> list[str]:
    low = text.lower()
    return [p for p in phrases if p.lower() in low]


def _require(path: Path) -> str:
    if not path.is_file():
        pytest.fail(f"{path} missing — this check cannot see its input")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Clause 1: one section, existing sections intact
# ---------------------------------------------------------------------------


def test_schema_has_exactly_one_prose_rules_section():
    """`grep -n "^## Prose Rules"` hits exactly once — the item's own check."""
    text = _require(SCHEMA)
    assert text.count("\n## Prose Rules\n") == 1, (
        "knowledge/KNOWLEDGE_SCHEMA.md must carry exactly one `## Prose Rules` "
        f"section; found {text.count(chr(10) + '## Prose Rules' + chr(10))}")


def test_prose_rules_is_added_without_disturbing_the_existing_sections():
    text = _require(SCHEMA)
    positions = []
    for heading in PREEXISTING_SECTIONS:
        assert f"\n{heading}\n" in text, (
            f"#431 adds a section; it does not remove one — {heading} is gone")
        positions.append(text.index(f"\n{heading}\n"))
    assert positions == sorted(positions), (
        "the pre-existing sections must stay in their original order:\n"
        + "\n".join(f"{p}: {h}" for p, h in zip(positions, PREEXISTING_SECTIONS)))
    # Page Types is the type vocabulary and Required Frontmatter the OKF block;
    # neither may have absorbed prose advice, which is clause 1's "without
    # altering" half read for content rather than for headings.
    for heading in ("## Page Types", "## Required Frontmatter", "## Naming Conventions"):
        body = _section(text, heading)
        assert not _banned_present(body, ["filler", "hedg", "banned vocabulary"]), (
            f"{heading} should hold its own subject; style rules belong to Prose Rules")


# ---------------------------------------------------------------------------
# Clause 2: a ban-list with examples, kept short
# ---------------------------------------------------------------------------


def test_prose_rules_section_is_a_ban_list_with_examples_and_stays_short():
    text = _require(SCHEMA)
    body = _section(text, "## Prose Rules")
    lines = body.strip("\n").splitlines()
    assert len(lines) <= MAX_SECTION_LINES, (
        f"Prose Rules is {len(lines)} lines; clause 2 caps it at "
        f"{MAX_SECTION_LINES} so it stays a ban-list and not an essay — the "
        "writers are already long prompts")

    filler = _banned_present(body, FILLER_OPENINGS)
    vocab = _banned_present(body, BANNED_VOCABULARY)
    hedge = _banned_present(body, HEDGE_STACKS)
    assert len(filler) >= 4, f"a ban-list needs concrete openings, got {filler}"
    assert len(vocab) >= 4, f"a ban-list needs concrete words, got {vocab}"
    assert hedge, f"a ban-list needs the hedge-stack spellings, got none of {HEDGE_STACKS}"

    # At least two bad→good example pairs, read from the table: rows under a
    # header naming Bad and Good, each row two backticked cells.
    header = re.search(r"^\|\s*Bad\s*\|\s*Good\s*\|$", body, re.M | re.I)
    assert header, "Prose Rules needs a | Bad | Good | example table"
    rows = [ln for ln in body[header.end():].splitlines()
            if re.match(r"^\|[^-|].*\|\s*`", ln)]
    assert len(rows) >= 2, (
        f"clause 2 requires >=2 bad→good example sentence pairs; found {len(rows)}")
    for row in rows:
        cells = [c for c in row.split("|")[1:-1]]
        assert len(cells) == 2 and "`" in cells[0] and "`" in cells[1], (
            f"example rows need both sides: {row[:80]}")


# ---------------------------------------------------------------------------
# Clause 3: self-contained in each writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skill", WRITERS)
def test_each_writer_skill_carries_the_contract_itself(skill):
    text = _require(SKILLS / skill / "SKILL.md")
    assert "Prose Rules" in text, (
        f"skills/{skill}/SKILL.md must name the contract it is obeying")
    filler = _banned_present(text, FILLER_OPENINGS)
    vocab = _banned_present(text, BANNED_VOCABULARY)
    hedge = _banned_present(text, HEDGE_STACKS)
    assert len(filler) + len(vocab) + len(hedge) >= 3, (
        f"skills/{skill}/SKILL.md must be executable without opening the schema: "
        "inline concrete banned phrases, not just a reference")
    assert vocab and hedge, (
        f"skills/{skill}/SKILL.md should carry banned vocabulary AND hedge "
        f"spellings (vocab={vocab}, hedge={hedge}); a single copied phrase is not "
        "a contract")


def test_the_three_writers_that_never_load_the_schema_are_why_inlining_is_required():
    """Guards the premise clause 3 rests on.

    If someone later teaches these skills to read the schema, inlining stops
    being necessary and this test says so out loud instead of the reasoning
    quietly rotting. Today the pointer does not exist, so the inline copy is
    the only route the rules have.
    """
    for skill in ("intelligence-pipeline", "ai-engineer-monitor", "documentation-digester"):
        text = _require(SKILLS / skill / "SKILL.md")
        if "KNOWLEDGE_SCHEMA" not in text:
            continue  # the premise: no pointer, so the copy must be inline
        assert re.search(r"Prose Rules", text), (
            f"skills/{skill}/SKILL.md now references the schema; if it really "
            "loads it, clause 3's inline copy may be revisited — until then the "
            "section must stay nameable from the skill alone")


# ---------------------------------------------------------------------------
# Clause 4: the youtube-digest session reaches the rules (the seam)
# ---------------------------------------------------------------------------


def _meta(tmp_path: Path) -> dict:
    bdir = tmp_path / "bundles" / "abc123"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "transcript.txt").write_text("words\n" * 50)
    return {
        "channel_key": "discover-ai", "channel_handle": "code4AI",
        "channel_name": "Discover AI", "video_id": "abc123",
        "url": "https://www.youtube.com/watch?v=abc123",
        "title": "Some Harness Talk", "published": "20260911",
        "bundle_dir": str(bdir), "meta_path": str(bdir / "meta.json"),
        "transcript_path": str(bdir / "transcript.txt"), "transcript_words": 6000,
        "transcript_lines": 420, "entities": {}, "existing_note": None,
        "target_note": str(tmp_path / "vault" / "20260911-some-harness-talk.md"),
        "enrichment": {"github": [], "papers": []},
        "measurements_path": str(bdir / "measurements.json"),
        "measurements_summary": "prefix-cache hit rate since boot 68.7%",
    }


def _assert_rules_reach_prose(prompt: str, where: str) -> None:
    """The item's own clause-4 grep, applied to rendered prompt text."""
    assert re.search(r"prose rule|hedg|filler|slop", prompt, re.I), (
        f"{where}: the acceptance grep for "
        "`prose rule|hedg|filler|slop` finds nothing — the digest session has no "
        "prose contract to follow")
    assert "Prose Rules" in prompt, f"{where}: must name the shared contract"
    filler = _banned_present(prompt, FILLER_OPENINGS)
    vocab = _banned_present(prompt, BANNED_VOCABULARY)
    hedge = _banned_present(prompt, HEDGE_STACKS)
    assert filler and vocab and hedge, (
        f"{where}: needs concrete phrases in all three classes, got "
        f"filler={filler} vocab={vocab} hedge={hedge}")


def test_the_rendered_digest_prompt_carries_the_prose_rules(tmp_path):
    prompt = Y.build_prompt(_meta(tmp_path), [])
    _assert_rules_reach_prose(prompt, "build_prompt()")
    # The rules have to land where the note is written — step 2 of the job,
    # not appended after the RESULT block a model is already formatting.
    step2 = prompt.index("Write the vault note")
    step3 = prompt.index("Evaluate it for Lloyd")
    assert step2 < prompt.index("Prose Rules") < step3, (
        "the ban-list belongs between writing the note and evaluating it; pasted "
        "anywhere else it is instruction the model has already stopped reading")


async def test_the_rules_reach_the_session_that_writes_the_note(tmp_path, monkeypatch):
    """The seam: ``execute()`` → ``run_prompt_in_session`` is a separate SDK
    session, and the note is written there, not here. Reading the ``PROMPT``
    constant would not catch a prompt that was built and then dropped, so this
    captures what the dispatcher is actually handed."""
    meta = _meta(tmp_path)
    calls: list[dict] = []

    async def fake_script(channel, *args, timeout):
        if args[0].split("=", 1)[0] == "--fetch":
            return {"ok": True, "meta": meta}
        return {"ok": True}

    async def fake_session(prompt, **kwargs):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        return {"text": "the note is written", "session_id": "s1",
                "stop_reason": "stop", "num_turns": 4, "errors": []}

    monkeypatch.setattr(Y, "_script", fake_script)
    monkeypatch.setattr(Y, "run_prompt_in_session", fake_session)
    monkeypatch.setattr(Y, "_vault_dirty_paths", lambda: set())
    monkeypatch.setattr(Y, "BACKLOG_DIR", tmp_path / "backlog")

    item = QueueItem(id=1, source=Y.NAME, kind="video", priority=60,
                     payload={"channel": "discover-ai", "video_id": "abc123"},
                     dedup_key=None, state="running", attempts=1, enqueued_at="",
                     claimed_at=None, claimed_by=None, completed_at=None, error=None)
    await Y.execute(item)

    assert calls, "execute() never dispatched a session — nothing to check"
    _assert_rules_reach_prose(calls[0]["prompt"], "the prompt dispatched by execute()")
