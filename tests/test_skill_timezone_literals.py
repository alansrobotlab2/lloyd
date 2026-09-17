"""Umbrella #1189 clause 3 — a governed skill template must not re-type a
season's timezone abbreviation beside a displayed-time token, and the landing
path must refuse it when one does.

The class and its five sightings
--------------------------------
A zone abbreviation typed into a template instead of read from the clock was
filed five times: ``app/post_capture.py``'s auto-captured heading asserted `PDT`
year-round (#601), ``morning-brief-and-triage`` typed `HH:MM PST` with no clock
step at all (#1079), ``nightly-reflection-signals`` typed `generated: … PST`
(#1080), and the three capture skills' daily-note heredocs typed
`### Session HH:MM PDT` while dating the note "PST" (#1081). #1112 named the
reason the class kept coming back: the machine check for exactly this was
gate-passed on branches ``9c09e9e``/``6620216`` and never merged, so every fix
was prose plus a hand-run grep, and the next edit to the same heredoc could
silently restore the literal. This file is that check, merged — the clause-4
twin is ``tests/test_skill_reflection_archive.py``.

What the class costs concretely: the abbreviation flips with the season —
America/Los_Angeles is PST from the first Sunday in November to the second
Sunday in March — so a typed abbreviation is wrong for roughly four and a half
months of every year, in the one field a reader uses to order a day's sessions
(the daily-note heading) or to tell which cycle a surviving report is (the
`generated:` header).

Two consumers, one definition
-----------------------------
The rule is ``scripts/skill_timezone.py``, and like ``reflection_archive`` it
exists so the definition has two consumers rather than an asserted copy:

* ``scripts/automod/vault_round.py::skill_timezone_errors`` calls it on every
  ``skills/**/SKILL.md`` a vault round touches. **That is the enforcement
  point** — regaining a literal is refused at ``automod_vault_land`` before the
  commit and the round's paths revert. The mutation below drives ``validate()``
  end to end, not the helper: "a correct helper that nothing calls is the exact
  shape of 'a guard whose input nothing wired up'"
  (``test_skill_reflection_archive.py``), and #1112 says the class died of
  exactly that — the check existed and nothing ran it.
* This file: unmarked mutations prove the rule can fail on the strings history
  typed (those run on every gate rung, and are why an inert rule cannot ship),
  and the live-vault group — the reporting copy ``pytest.ini`` describes —
  asserts the real skills, per file, both directions.

The live group is marked ``live_vault`` for the reason ``reflection_archive``
states: an assertion about a tree no round under test controls (an hourly
writer can re-word a skill between rounds) belongs on the writer, not on a hard
gate rung that would fail the next author for the previous writer's wording. The
mutations, which read no vault, are not marked and cannot be skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.skill_timezone import template_clock_violations

SKILLS_DIRS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]


def _iter_governed():
    """(path-name, body) for every governed SKILL.md found in either tree."""
    for root in SKILLS_DIRS:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            rel = entry.name + "/SKILL.md"
            if rel in _GOVERNED:
                f = entry / "SKILL.md"
                if f.exists():
                    yield rel, f.read_text(encoding="utf-8", errors="replace")


def _active_skills() -> dict[str, str]:
    """Every SKILL.md the prompt actually advertises → its body.

    The loader mirrors ``prompt_builder._load_skills_index`` exactly as
    ``tests/test_skill_reflection_archive.py`` does: dot-prefixed directories
    are the archive, quarantined skills are out of circulation. The rule judges
    the corpus that actually gets loaded.
    """
    from prompt_builder import _is_quarantined_skill

    out: dict[str, str] = {}
    for root in SKILLS_DIRS:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in out:
                continue
            skill_file = entry / "SKILL.md"
            if skill_file.exists() and not _is_quarantined_skill(skill_file):
                out[entry.name] = skill_file.read_text(
                    encoding="utf-8", errors="replace")
    return out


# --- The corpus of the fix ---------------------------------------------------------
# One entry per skill #1189's clause 2 governs (the three capture heredocs, the
# Phase 5 header, and the three more `generated:` sightings its triage grep
# surfaced: nightly-behavior-test ×3, nightly-morning-briefing ×1,
# nightly-prompt-audit ×2). Value = (string #1189 measured at triage, string that
# replaced it on 2026-09-17). Restoring the first in the real body must make the
# rule fire; the second must be clean. That pairing is the rule proved against the
# real text of the real fix, per file — the vault edit and this check cannot then
# drift apart silently, because a nightly rewrite that changes a fix shape fails
# this and points at the pair to update.
_GOVERNED: dict[str, tuple[str, str]] = {
    # Heading: the season's abbreviation deleted; the clock named instead (#1081).
    "memory-capture/SKILL.md": (
        "### Session HH:MM PDT — Title",
        "### Session HH:MM — Title",
    ),
    "periodic-memory-capture-lloyd/SKILL.md": (
        "### Session HH:MM PDT — Title",
        "### Session HH:MM — Title",
    ),
    "periodic-memory-capture-dee/SKILL.md": (
        "### Session HH:MM PDT — Title",
        "### Session HH:MM — Title",
    ),
    # Report headers: hand-typed local stamp → ISO-8601 UTC from `date -u` (#1080).
    "nightly-reflection-signals/SKILL.md": (
        "generated: YYYY-MM-DD HH:MM PST",
        "generated: YYYY-MM-DDTHH:MM:SSZ",
    ),
    "nightly-behavior-test/SKILL.md": (
        "generated: YYYY-MM-DD HH:MM PST",
        "generated: YYYY-MM-DDTHH:MM:SSZ",
    ),
    "nightly-morning-briefing/SKILL.md": (
        "generated: YYYY-MM-DD HH:MM PST",
        "generated: YYYY-MM-DDTHH:MM:SSZ",
    ),
    "nightly-prompt-audit/SKILL.md": (
        "generated: YYYY-MM-DD HH:MM PST",
        "generated: YYYY-MM-DDTHH:MM:SSZ",
    ),
}

#: #1189 clause 2's acceptance check, restated as a pattern so the live scan
#: asserts the literal contract the item was written with and not only this
#: module's generalisation of it. `generated` is case-folded because skill prose
#: capitalises the field at least once; check before "fixing" that.
_ACCEPTANCE_GREP = re.compile(r"HH:MM PST|HH:MM PDT|(?i:generated): .* P[SD]T")


# --- The live vault: what the clause required, and what keeps it true --------------

@pytest.mark.live_vault
def test_the_acceptance_grep_returns_zero_on_the_governed_skills():
    """#1189's own acceptance check, as written, over the live tree.

    ``grep -rn 'HH:MM PST\\|HH:MM PDT\\|generated: .* PST' ~/obsidian/skills/*/SKILL.md``
    → 0 hits is the clause; this is that grep over the same files, plus the repo
    copies at ``~/lloyd/skills`` (empty today, and the rule judges what is
    loaded, so an unswept shadow copy must not be able to hide). A hit here is
    one of the exact strings the item names, re-typed.
    """
    hits = [
        f"{rel}:{lineno}: {line.strip()[:100]}"
        for rel, body in _iter_governed()
        for lineno, line in enumerate(body.splitlines(), 1)
        if _ACCEPTANCE_GREP.search(line)
    ]
    assert not hits, "the acceptance grep reappeared:\n" + "\n".join(hits)


@pytest.mark.live_vault
def test_every_governed_skill_is_clean_and_its_fix_is_what_the_rule_blesses():
    """Per governed file, both directions against the real bytes.

    (a) as landed the file yields no violation — clause 2's outcome; (b) the fix
    string is actually present (so the pairing below is describing this file and
    not a memory of it); (c) restoring the triage-measured string *in place, at
    its own line* makes exactly one violation appear, and it names the skill and
    the line — so the rule's reason is the real defect rather than an artefact
    of a synthetic fixture, and a rule that catches the class while
    over-flagging the rest of the file is caught too (exactly one, not four).
    """
    seen = set()
    for rel, body in _iter_governed():
        seen.add(rel)
        assert not template_clock_violations(rel, body), (
            f"{rel}: a governed skill regains a hand-typed zone beside a "
            "displayed-time token"
        )
        before, after = _GOVERNED[rel]
        assert after in body, (
            f"{rel}: the fix string {after!r} is not in the file — the template "
            "changed shape and _GOVERNED must be updated with it, or this "
            "pairing is describing a file that no longer exists"
        )
        assert before not in body, f"{rel}: the triage string {before!r} is back"
        vs = template_clock_violations(rel, body.replace(after, before, 1))
        assert len(vs) == 1, (
            f"{rel}: restoring {before!r} produced {len(vs)} violations, not 1 — "
            "the rule no longer sees the exact string the class was measured "
            f"with: {vs}"
        )
    assert seen == set(_GOVERNED), (
        f"governed skills found in neither tree: {sorted(set(_GOVERNED) - seen)}"
    )


@pytest.mark.live_vault
def test_the_live_skill_tree_is_clean_beyond_the_governed_seven():
    """The whole active corpus under the rule, with its denominator printed.

    Seven files were the measured sightings; the rule governs every template, so
    a fresh sighting in an unwatched skill fails *here* — the reporting copy of
    the class recurring. The assertion on the count is the point of the message:
    a green run that read no skills would be the exact vacuity #1112 found the
    class in (a check that existed and never ran), and the corpus size is
    recorded in the failure text so a future pass can see what was covered
    rather than re-deriving it.
    """
    bodies = _active_skills()
    assert len(bodies) >= 150, (
        f"only {len(bodies)} active skills discovered under {SKILLS_DIRS}; the "
        "live corpus measured 194 on 2026-09-17, so this scan is reading far "
        "less than it claims to cover"
    )
    dirty = {
        name: template_clock_violations(name, body)
        for name, body in bodies.items()
        if template_clock_violations(name, body)
    }
    assert not dirty, (
        f"{len(dirty)} of {len(bodies)} active skills hand-type a zone beside a "
        "displayed time:\n"
        + "\n".join(f"{n}: {v}" for n, v in sorted(dirty.items()))
    )


# --- Mutations: the rule can fail, on the strings the class was measured with ------
# Unmarked on purpose: no vault is read, and every gate rung must run these. A
# mutation proving nothing is the failure mode the archived branch version could
# not be trusted for (#1112), and these are what make an inert rule uninstallable.

_TYPED_SHAPES = {
    # capture-skill heredoc heading, both seasons (#1081)
    "capture heading PDT": "### Session HH:MM PDT — Title\n",
    "capture heading PST": "### Session HH:MM PST — Title\n",
    # morning-brief header template and its inject summary, as typed (#1079)
    "brief header": "Brief + Triage — YYYY-MM-DD HH:MM PST\n",
    "inject summary": 'summary="Brief + Triage — YYYY-MM-DD HH:MM PDT"\n',
    # report headers: the `HH:MM` placeholder form and a measured-time form (#1080)
    "report header placeholder": "generated: YYYY-MM-DD HH:MM PST\n",
    "report header measured": "generated: 2026-09-03 22:35 PST\n",
    "report header as a bullet": "- generated: YYYY-MM-DD HH:MM PDT\n",
}


@pytest.mark.parametrize("shape", sorted(_TYPED_SHAPES))
def test_the_rule_fails_on_every_shape_the_class_was_measured_with(shape):
    """Each typed string, restored into a minimal template, is refused by name."""
    body = "# Skill\n\nWrite it like this:\n\n```\n" + _TYPED_SHAPES[shape] + "```\n"
    vs = template_clock_violations("mutant-skill", body)
    assert vs, f"the rule missed the measured shape {shape!r}: {body!r}"
    assert vs[0].startswith("mutant-skill: line ") and "P" in vs[0], (
        f"a violation must name the skill and the line: {vs[0]}"
    )


def test_the_rule_blesses_every_shape_the_fix_uses():
    """The replacements are clean under the same rule.

    If they were not, the fix would be unlandable — which is how a branch-only
    check stays on a branch: a lint too broad to pass is a lint nobody runs. The
    clean set is every form the 2026-09-17 edits actually introduced, plus the
    two sanctioned ways of *naming* the abbreviation (inside inline code, or as
    a rule sentence with no displayed-time token).
    """
    fixed = (
        "### Session HH:MM — Title\n"                       # capture heading, zone gone
        "Brief + Triage — <YYYY-MM-DD> <HH:MM> <OFFSET>\n"  # header, zone from %Z
        "generated: YYYY-MM-DDTHH:MM:SSZ\n"                 # ISO Z header (#1080's fix)
        "- generated: YYYY-MM-DDTHH:MM:SSZ\n"
        "Take the date from `TZ=America/Los_Angeles date +%Y-%m-%d` and paste it\n"
        "abbreviation flips with the season (PST November–March, PDT April–October)\n"
        "all times in PST/PDT (local) — the abbreviation comes from `%Z`\n"
    )
    assert template_clock_violations("fixed-skill", fixed) == []


def test_the_rule_ignores_prose_about_the_zone_and_escaped_time_tokens():
    """The false-positive half of the boundary, pinned line by line.

    Every line below appears in this tree's live skills (or is the shape a fix
    would want): scheduling prose, conventions, quoted history, path templates,
    and a rule sentence that names the abbreviations to forbid them. None is a
    displayed-time template. They must stay clean or the rule alarms on
    *vocabulary* — the same timezone vocabulary that fooled the dedupe twice
    (#1079 auto-merged into #601 at lexical 0.105, #1112 into #1079 at 0.125),
    which is why a check in this area must be able to say what a template is.
    """
    prose = (
        "Runs at 6:00 AM PST daily.\n"
        "- Time (PST)\n"
        "Use yesterday's date (PST) to read `~/obsidian/memory/YYYY-MM-DD.md`.\n"
        "the brief injected `Brief + Triage — 2026-09-17 02:53 PDT` and\n"
        "a hand-typed `PST` anywhere is a defect\n"
        "Use PST timezone for the date\n"
        "| generated | 2026-09-13 22:35 PST |\n"   # a table row is prose too
    )
    assert template_clock_violations("prose-skill", prose) == []


def test_the_carve_outs_are_narrow_enough_that_a_template_inside_them_still_fires():
    """Fenced blocks and table rows are judged; only the backtick *span* escapes.

    ```markdown fences and `| … |` table rows carry templates as routinely as
    bullets do, so both are judged — the fenced case is exactly what a capture
    heredoc looks like, and a rule that exempted fenced text would exempt the
    class.

    The recorded residual, stated rather than hidden: text between backticks is
    exempt at landing (that is what makes the fix documentation landable), so an
    author who wrote a template purely as inline code would duck
    ``skill_timezone_errors``. The un-stripped layer covers it —
    ``test_the_acceptance_grep_returns_zero_on_the_governed_skills`` runs #1189's
    literal grep over live text with no code-span stripping, so a quoted
    `HH:MM PST` in a real skill fails there even though the land-time rule
    ignores it. Two consumers, one definition, different strictness on purpose;
    the two live scans disagree by design and both are asserted here.
    """
    for body in (
        "```markdown\ngenerated: YYYY-MM-DD HH:MM PST\n```\n",
        "| generated: YYYY-MM-DD HH:MM PST | x |\n",
    ):
        assert template_clock_violations("smuggler", body), (
            f"a template line stayed invisible to the rule: {body!r}"
        )


# --- The enforcement point: the vault writer, not a test rung ----------------------

def _scratch_skill(vault: Path, name: str, body: str) -> str:
    rel = f"skills/{name}/SKILL.md"
    target = vault / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return rel


@pytest.fixture
def scratch_vault(tmp_path, monkeypatch):
    """A vault tree this test owns, with the loader subprocess stubbed for the
    same reason ``test_skill_reflection_archive.scratch_vault`` gives: skill
    loadability is a real check of a different question, owned by
    tests/test_automod_vault_round.py, and a subprocess per mutation is waste.
    ``frontmatter_error``/``contract_errors`` need no stub — a skill body with
    front matter neither names a prompt surface nor a knowledge vocabulary, so
    any error surface left here is this round's own."""
    from scripts.automod import vault_round

    monkeypatch.setattr(vault_round, "VAULT", tmp_path)
    monkeypatch.setattr(vault_round, "loader_errors", lambda paths: [])
    return tmp_path


def test_the_vault_writer_refuses_a_skill_that_retyped_the_zone_abbreviation(scratch_vault):
    """The wiring, not just the helper: ``validate()`` — what ``land()`` and so
    ``automod_vault_land`` call — must reach the timezone branch, refuse the real
    fix reverted, name the file and the line, and then let the fixed shape land.

    Both template shapes are driven through the writer, because they are matched
    by two different patterns in the rule and only the call graph is shared.
    """
    from scripts.automod import vault_round

    heading = _scratch_skill(
        scratch_vault, "memory-capture",
        "# Capture\n\n```\n### Session HH:MM PDT — Title\n```\n",
    )
    errors, _ = vault_round.validate([heading])
    assert len(errors) == 1, f"the lander let a retyped heading through: {errors}"
    assert errors[0].startswith(f"{heading}: "), (
        f"the refusal must name the file: {errors[0]}"
    )
    assert "HH:MM PDT" in errors[0], (
        f"the refusal must quote the offending template line: {errors[0]}"
    )

    _scratch_skill(scratch_vault, "memory-capture",
                   "# Capture\n\n```\n### Session HH:MM — Title\n```\n")
    assert vault_round.validate([heading])[0] == [], (
        "the fixed heading must be landable, or the rule cannot be satisfied"
    )

    header = _scratch_skill(
        scratch_vault, "nightly-reflection-signals",
        "# Signals\n\n```markdown\n---\ngenerated: YYYY-MM-DD HH:MM PST\n---\n```\n",
    )
    errors, _ = vault_round.validate([header])
    assert len(errors) == 1 and "generated: YYYY-MM-DD HH:MM PST" in errors[0], (
        f"a report header regaining the hand-typed PST stamp must be refused: {errors}"
    )


def test_the_vault_writer_only_judges_the_skills_a_round_touched(scratch_vault):
    """Scoping, as every other vault validator has it: an untouched dirty skill
    must not hold an unrelated round hostage. Coverage of the untouched one is
    ``test_the_live_skill_tree_is_clean_beyond_the_governed_seven``'s job."""
    from scripts.automod import vault_round

    _scratch_skill(scratch_vault, "sinner", "### Session HH:MM PDT — Title\n")
    unrelated = _scratch_skill(scratch_vault, "innocent",
                               "# Notes\n\nNothing about clocks here.\n")
    assert vault_round.validate([unrelated])[0] == []
    assert vault_round.skill_timezone_errors([unrelated]) == []


def test_the_writer_and_the_test_share_one_definition():
    """Static pin: the lander must call the rule this file asserts.

    ``test_skill_reflection_archive.py`` carries the twin of this assertion, for
    the same reason — a helper and its test can both remain while the call site
    quietly disappears, and then the rule is documentation again. That is the
    precise state #1112 found this class in: written, gate-passed, unmerged, and
    therefore unable to stop anything.
    """
    src = (ROOT / "scripts" / "automod" / "vault_round.py").read_text(encoding="utf-8")
    assert "import skill_timezone" in src, (
        "vault_round no longer imports the clock-literal rule: the writer "
        "stopped enforcing it and only the live_vault reporting copy remains"
    )
    assert "skill_timezone_errors(paths)" in src, (
        "skill_timezone_errors is imported but not wired into validate(): a "
        "guard whose input nothing wired up"
    )
