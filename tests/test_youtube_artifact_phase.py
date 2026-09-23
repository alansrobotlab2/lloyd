"""Backlog #448: the YouTube extraction flow must persist its digest to the vault.

Research skills had a write-and-report rule everywhere except one path. ``medium-research``,
``quick-research``, ``research-agent`` and ``deep-research`` all required it; the two skills
that actually run transcript extraction did not. ``skills/youtube-transcript/SKILL.md``
stopped at ``yt-dlp -o "%(id)s"`` and ``skills/youtube-content/SKILL.md`` documented
stdout-only ``fetch_transcript.py`` modes — so nothing said where the *derived* summary must
land. The cost, measured 2026-09-09: 71 of 154 transcript-class sessions since 09-01 wrote no
vault ``.md``, and the freshest (``20260908_170732_iv214c``, "YouTube transcript highlights
for T2v2pf_uypE") delivered a 7,299-char digest whose only persistence pointer was
``~/lloyd/_pipeline/tmp/T2v2pf_uypE.txt`` — a gitignored path — with
``grep -rl "T2v2pf_uypE" ~/obsidian/`` returning 0 hits. The digest was gone by the next day.

Alan stated the failure twice on 2026-09-07 (17:39, 21:02); the second statement arrived
*after* he asked for the results back, so the loss was demonstrated, not hypothesized.
Transcript-only is doubly unreliable because auto-capture compresses whole days — 09-05
produced 3 sections for 14+ session files.

Backlog #566 widened the pin: #448 had fixed the *derived* half and left the *input* half
half-true. The raw transcript lands in scratch that is gitignored and, since 2026-09-11 in
some sessions, under ``/tmp`` where ``systemd-tmpfiles-clean.timer`` reaps it daily; no skill
named a directory (a bare ``-o "%(id)s"`` writes wherever the shell happened to be) and no
note recorded a hash, so a lost transcript was undetectable from the note. So: one named
scratch dir, a save step that writes into it, and ``transcript_path`` + ``transcript_md5`` in
the video-note front matter, with ``scripts/groundskeeper/retention-sweep.py`` owning the age
of that directory.

Both skills now carry a named ``## Artifact Phase`` section. This test pins it, so rewriting
either skill cannot silently drop the requirement again. The check is expressed as a pure
function over skill text and exercised hermetically on fixtures below — those cases RUN under
the automod gate. The assertions against the live skill files read ``~/obsidian``, which no
round under test controls, so they carry ``live_vault``: they run in the ordinary suite and in
CI, and are excluded from the gate's ``-m "not live_vault"`` rung for the same reason
``test_prompt_surface_budget.py`` is.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.data_root as dr  # noqa: E402  (needs ROOT on the path first)

# Vault skills first, then the repo checkout — same precedence as
# tests/test_skill_tool_names.py and prompt_builder._load_skills_index.
SKILLS_DIRS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]

# The two skills the item names. Deliberately not every research skill: the others already
# had the rule before #448 existed, and asserting on them here would fold their history into
# this item's pin.
SKILLS_UNDER_TEST = ["youtube-content", "youtube-transcript"]

# These live cases carry no skip of any kind. The gate runs `-m "not live_vault"`, so they
# execute in the ordinary suite and in CI, on a machine that has the vault. A vault checkout
# missing one of these SKILL.md files is the regression under test, not a reason to pass, so
# `_require_skill_text` asserts rather than skipping. Round SM_20260914_184202 was refused over
# exactly one line: a `skipif(not SKILLS_DIRS[0].is_dir())` marker on these cases. A pin that
# can disappear by its own subject going missing is not a pin, whatever the reason string says.

def _fenced_blocks(text: str) -> list[str]:
    """The bodies of ```-fenced blocks, fence lines dropped, in document order.

    Structural rather than a regex trick: a pattern can only say "a fence appears somewhere
    before the key", and that is still satisfied by a key sitting in prose immediately AFTER a
    closing fence — the exact rewrite this has to refuse. Toggling on a line that opens with
    three backticks is what markdown itself does with them, so these are the real boundaries.
    """
    blocks: list[str] = []
    current: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if inside:
                blocks.append("\n".join(current))
                current = []
            inside = not inside
            continue
        if inside:
            current.append(line)
    return blocks


# Requirements that must hold INSIDE a fenced block, not merely somewhere in the file. Round
# SM_20260914_184202 was refused for grading these two position-blind: `^transcript_path:` with
# MULTILINE matches the key anywhere, so a rewrite that moved it out of the front-matter template
# the skill ships and into a sentence kept the pin green while making the key unqueryable — which
# is the only reason #566 asks for front matter rather than prose. The scope lives here beside the
# labels, so loosening it is one visible deletion instead of a quiet edit to a regex.
TEMPLATE_SCOPED = frozenset({
    "transcript_path as a front-matter key",
    "transcript_md5 as a front-matter key",
})


# What the Artifact Phase must state, as (requirement, pattern). Each one is a distinct
# failure mode from the item, not a stylistic preference. A label in TEMPLATE_SCOPED is graded
# per fenced block rather than over the whole file.
ARTIFACT_PHASE_REQUIREMENTS: dict[str, re.Pattern[str]] = {
    # The phase must be a NAMED phase, so it reads as a step and not as prose.
    "an '## Artifact Phase' heading": re.compile(r"^##\s+Artifact Phase", re.MULTILINE),
    # Marked non-optional — the wording every other research skill needed to get conformance.
    "a HARD / mandatory marker": re.compile(r"\bHARD\b|[Mm]andatory"),
    # A concrete vault destination, not "save it somewhere". Convention per
    # skills/ai-engineer-monitor/SKILL.md:84 (`knowledge/youtube/<Channel>/`).
    "the knowledge/youtube/<Channel>/ path convention":
        re.compile(r"obsidian/knowledge/youtube/[<\w]"),
    "a dated, slug-bearing filename": re.compile(r"<YYYYMMDD>-<slug>\.md|<YYYYMMDD>-<"),
    # The note type the vault already uses for video notes, so the artifact is findable
    # alongside the ones ai-engineer-monitor writes.
    "video-note front matter": re.compile(r"type:\s*video-note"),
    # The video id must be greppable from the note body — that is what makes
    # `grep -rl "<videoId>" ~/obsidian/` the verification command it is in the acceptance.
    "the video id greppable in the note (video_id)": re.compile(r"video_id"),
    "`grep -rl <videoId> ~/obsidian/` as the verification step":
        re.compile(r"grep -rl"),
    # Report the absolute path in the run's own final message — "where did this land?"
    # must be answerable from the transcript.
    "report the absolute path in the final message":
        re.compile(r"(final message|Saved:)"),
    # The 08-24 rule: confirm on disk, never from a commit message or from memory.
    "on-disk verification, not a claimed write": re.compile(r"[Dd]isk"),
    # Why the rule exists, in the place it will be reread: _pipeline/tmp is gitignored and
    # the transcript compresses. Without this line the next rewrite treats it as noise.
    "why transcript/tmp is not persistence": re.compile(r"_pipeline/tmp|gitignored"),

    # --- backlog #566: the INPUT half. The scratch file has to have one owner and the note
    # has to be able to prove the input is still there.
    # One named directory, stated as the variable both save forms use. Before #566 neither
    # skill named a location, so sessions invented one: `~/lloyd/_pipeline/tmp/` in some,
    # `/tmp/yt/` in others (measured 2026-09-13: both trees live the same week).
    "the one named transcript scratch dir":
        re.compile(r'TRANSCRIPT_DIR\s*=\s*"?\$HOME/lloyd-data/_pipeline/tmp'),
    # Naming a directory is not saving to it: yt-dlp needs `-o "$TRANSCRIPT_DIR/%(id)s"`
    # rather than the bare cwd-relative `%(id)s`, and the stdout-only helper script needs a
    # redirect. This is the half that actually moves the bytes.
    "a save step that writes into that dir":
        re.compile(r'(?:-o\s+"|>\s*")\$TRANSCRIPT_DIR/'),
    # The scratch file is swept (see sweep_transcript_scratch), so the note must carry the
    # pointer. Both keys are TEMPLATE_SCOPED: they have to sit inside the front-matter template
    # the skill ships, because both skills ALSO mention the keys in prose and a whole-file match
    # would grade that prose as if it were the template.
    "transcript_path as a front-matter key": re.compile(r"^transcript_path:\s+\S", re.MULTILINE),
    # ...and the digest, so a swept or edited transcript is detectable from the note alone.
    "transcript_md5 as a front-matter key": re.compile(r"^transcript_md5:\s+\S", re.MULTILINE),
    # Which of the two files is the artifact. Stated, because "save the transcript too" is
    # what made #448 read as satisfied while the input rotting in gitignore stayed invisible.
    "the note is durable and the scratch file disposable":
        re.compile(r"durable artifact[\s\S]{0,200}disposable|disposable[\s\S]{0,200}durable artifact"),
}


# Which save form each skill's OWN prescribed route has to carry. Both skills satisfy the
# shared save-step requirement on either form, and that latitude is wrong here: this box has
# no Node runtime, so youtube-transcript's Environment Guardrail (2026-08-19) forbids yt-dlp
# outright and the youtube-transcript-api block is the route that actually runs. Review on
# round SM_20260914_184202 deleted that block and the skill still passed — the `-o` line was
# holding the requirement up. So the route that runs is pinned by name, per skill.
SKILL_ROUTE_REQUIREMENTS: dict[str, dict[str, re.Pattern[str]]] = {
    "youtube-transcript": {
        "the yt-dlp step writes into the scratch dir":
            re.compile(r'-o\s+"?\$TRANSCRIPT_DIR/'),
        # One fenced block only (the `(?!```)` guards): the path is built from `_pipeline`/`tmp`
        # AND written, in the same snippet. Naming it without writing it does not save a file.
        "the youtube-transcript-api route builds the scratch path and writes it":
            re.compile(r'_pipeline(?:(?!```)[\s\S]){0,300}?tmp(?:(?!```)[\s\S]){0,300}?write_text'),
    },
    "youtube-content": {
        # Every mode of its helper script is stdout-only, so a redirect IS the save step.
        "the helper script's stdout is redirected into the scratch dir":
            re.compile(r'>\s*"\$TRANSCRIPT_DIR/'),
    },
}


def missing_artifact_phase(text: str, skill: str | None = None) -> list[str]:
    """Return every Artifact-Phase requirement ``text`` fails to state.

    Pure over skill text, so the gate can run it on fixtures and the live_vault cases can run
    it on the real files with no duplicated logic. Pass ``skill`` to also grade the save form
    that skill's own prescribed extraction route must carry.

    A label in ``TEMPLATE_SCOPED`` is satisfied only by a match inside some fenced block; every
    other label is graded over the whole file.
    """
    required = dict(ARTIFACT_PHASE_REQUIREMENTS)
    if skill is not None:
        required.update(SKILL_ROUTE_REQUIREMENTS.get(skill, {}))
    blocks = _fenced_blocks(text) if TEMPLATE_SCOPED else []
    missing: list[str] = []
    for label, pattern in required.items():
        if label in TEMPLATE_SCOPED:
            if not any(pattern.search(block) for block in blocks):
                missing.append(label)
        elif not pattern.search(text):
            missing.append(label)
    return missing


def _skill_file(name: str) -> Path | None:
    for root in SKILLS_DIRS:
        candidate = root / name / "SKILL.md"
        if candidate.exists():
            return candidate
    return None


def _require_skill_text(name: str) -> str:
    """Read a live skill, or fail outright. These live cases take no skip at all: a vault
    checkout that is missing a SKILL.md is the regression under test, not a reason to pass."""
    path = _skill_file(name)
    assert path is not None, (
        f"skills/{name}/SKILL.md is not installed under {SKILLS_DIRS} — the skill that "
        "carries this rule has gone missing, which is how a pinned requirement disappears "
        "without a diff to it")
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# hermetic — these RUN under the automod gate (`-m "not live_vault"`)
# ---------------------------------------------------------------------------

_CANONICAL = """
## Artifact Phase — Persist the digest (HARD — backlog #448)

A digest that exists only in the chat transcript does not exist. Auto-capture compresses
whole days and `~/lloyd-data/_pipeline/tmp/` is gitignored, so neither one is persistence.
Write the derived output to `~/obsidian/knowledge/youtube/<Channel>/<YYYYMMDD>-<slug>.md`
**before** reporting anything.

The raw transcript goes to the one scratch dir — never a bare `-o "%(id)s"`, never `/tmp`:

```bash
TRANSCRIPT_DIR="$HOME/lloyd-data/_pipeline/tmp"
mkdir -p "$TRANSCRIPT_DIR"
yt-dlp --write-auto-sub --skip-download --sub-lang en -o "$TRANSCRIPT_DIR/%(id)s" "$URL"
```

The vault note is the durable artifact; the scratch file is disposable.

```yaml
type: video-note
video_id: <videoId>
transcript_path: /home/alansrobotlab/lloyd-data/_pipeline/tmp/<videoId>.txt
transcript_md5: <md5sum computed at write time>
---
```

Then verify on disk: `grep -rl "<videoId>" ~/obsidian/` returns the note. A path named in
the message but absent from disk is a failed run.

Final message ends with the path, e.g. `Saved: /home/alansrobotlab/obsidian/...md`.
"""


def test_canonical_phase_satisfies_every_requirement():
    """The checker accepts a phase that states all of them — otherwise the assertions
    below are unfalsifiable and pin nothing."""
    assert missing_artifact_phase(_CANONICAL) == []


def test_checker_is_not_vacuous():
    """Fifteen requirements is the pin's resolution: the ten #448 clauses plus the five
    #566 ones. A checker with one loose regex would pass a stripped skill file, which is the
    failure this whole test exists to catch."""
    assert len(ARTIFACT_PHASE_REQUIREMENTS) >= 15


@pytest.mark.parametrize("strip", [
    "## Artifact Phase — Persist the digest (HARD — backlog #448)\n",
    "~/obsidian/knowledge/youtube/<Channel>/<YYYYMMDD>-<slug>.md",
    "type: video-note\nvideo_id: <videoId>\n",
    "grep -rl \"<videoId>\" ~/obsidian/",
    "Final message ends with the path, e.g. `Saved: /home/alansrobotlab/obsidian/...md`.",
    'TRANSCRIPT_DIR="$HOME/lloyd-data/_pipeline/tmp"\n',
    '-o "$TRANSCRIPT_DIR/%(id)s" ',
    "transcript_path: /home/alansrobotlab/lloyd-data/_pipeline/tmp/<videoId>.txt\n",
    "transcript_md5: <md5sum computed at write time>\n",
    "The vault note is the durable artifact; the scratch file is disposable.",
])
def test_removing_one_requirement_is_caught(strip: str):
    """Each requirement is individually load-bearing: delete exactly one line group and
    the checker must name it. Guards against a pattern so loose it matches anything."""
    # A strip that isn't an exact substring is a no-op, and a no-op silently passes for the
    # wrong reason — which is what happened to the first draft of this case.
    assert strip in _CANONICAL, f"strip {strip!r} is not in the fixture; the case proves nothing"
    text = _CANONICAL.replace(strip, "")
    missing = missing_artifact_phase(text)
    assert missing, f"stripping {strip!r} should have been caught, but the checker passed"


_KEY_LINES = ("transcript_path: /home/alansrobotlab/lloyd-data/_pipeline/tmp/<videoId>.txt\n"
              "transcript_md5: <md5sum computed at write time>\n")


def test_provenance_keys_belong_in_the_fenced_template_not_in_prose():
    """SM_20260914_184202's second finding, made permanent.

    Move the two key lines out of the front-matter template the skill ships and say the same
    thing in prose — placed immediately after the closing fence, where a whole-file
    `^transcript_path:` match still lands. Nothing is deleted; the keys have only stopped being
    queryable front matter, which is the failure #566 asked for keys rather than sentences about
    keys in the first place. Grade these labels over the whole file again and this case goes red.
    """
    assert any("transcript_path:" in b for b in _fenced_blocks(_CANONICAL)), \
        "the fixture below is meaningless unless the template is a fenced block"
    moved = _CANONICAL.replace(_KEY_LINES, "").replace(
        "Then verify on disk:", _KEY_LINES + "\nThen verify on disk:")
    assert not any("transcript_path:" in b for b in _fenced_blocks(moved)), \
        "the relocation must actually have happened to prove anything"
    assert "transcript_path:" in moved, "this is relocation, not deletion — the old check passed"
    missing = missing_artifact_phase(moved)
    assert "transcript_path as a front-matter key" in missing, missing
    assert "transcript_md5 as a front-matter key" in missing, missing


# The shape the youtube-transcript-api route uses to save (its skill section, condensed): the
# path built from `_pipeline`/`tmp`, and the transcript written to it in the same snippet.
_YTA_SAVE_BLOCK = (
    "```python\n"
    'scratch = Path.home() / "lloyd" / "_pipeline" / "tmp" / f"{video_id}.txt"\n'
    'scratch.parent.mkdir(parents=True, exist_ok=True)\n'
    'scratch.write_text("\\n".join(s.text for s in transcript), encoding="utf-8")\n'
    "```\n"
)


def test_each_prescribed_route_save_form_is_load_bearing():
    """`_CANONICAL` carries only the yt-dlp save step, so grading it AS each skill must
    report exactly the save form that skill still owes — and handing it the real
    youtube-transcript-api block closes that one gap and no other.

    This is the round-SM_20260914_184202 review's finding as a permanent test: the shared
    save-step requirement accepts either form, so deleting the Python block from
    youtube-transcript — the route this box must use, since the skill's Environment Guardrail
    (2026-08-19) forbids yt-dlp on a box with no Node — left the skill passing."""
    assert missing_artifact_phase(_CANONICAL, skill="youtube-transcript") == [
        "the youtube-transcript-api route builds the scratch path and writes it"]
    assert missing_artifact_phase(_CANONICAL, skill="youtube-content") == [
        "the helper script's stdout is redirected into the scratch dir"]
    assert missing_artifact_phase(_CANONICAL + _YTA_SAVE_BLOCK, skill="youtube-transcript") == []


def test_a_pre_448_skill_file_fails_the_checker():
    """The failure this test was written against: an extraction skill that documents
    fetching and formatting but never says where the digest lands."""
    pre_448 = """
## Helper script

```bash
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID" --text-only
```

## Workflow

1. Fetch the transcript using the helper script
2. Transform into the requested output format using your own reasoning
"""
    missing = missing_artifact_phase(pre_448)
    assert "an '## Artifact Phase' heading" in missing
    assert "the knowledge/youtube/<Channel>/ path convention" in missing
    assert len(missing) >= 6, f"expected most requirements missing, got {missing}"


# ---------------------------------------------------------------------------
# live vault — excluded from the gate rung, run in the ordinary suite
# ---------------------------------------------------------------------------

@pytest.mark.live_vault
@pytest.mark.parametrize("skill", SKILLS_UNDER_TEST)
def test_extraction_skill_states_the_artifact_phase(skill: str):
    """#448's acceptance, on the real files: both extraction skills must carry a named
    Artifact Phase that says where the digest lands, how to verify it on disk, and to
    report the absolute path."""
    missing = missing_artifact_phase(_require_skill_text(skill), skill=skill)
    assert missing == [], (
        f"skills/{skill}/SKILL.md lost its Artifact Phase requirements: {missing}. "
        "A transcript digest written only to the chat transcript is lost when the session "
        "ends (backlog #448); the phase must name a vault path, a verification step, and "
        "the path report — and per #566 the scratch dir, the save form its own extraction "
        "route uses, and the two transcript provenance keys."
    )


# ---------------------------------------------------------------------------
# the seam between the rule and the job that enforces it (backlog #566)
#
# The skill text lives in ~/obsidian/skills (a live tree no round under test controls — the
# same reason the live-file assertions above carry `live_vault`); the retention policy lives
# in scripts/groundskeeper/retention-sweep.py in THIS repo. Those two files are read by
# different actors a day apart — the extraction session follows the skill, the weekly sweep
# obeys the script — so renaming the directory on one side leaves a store that the session
# believes is bounded and the sweep has never heard of. Two cases, split by what they can see:
# the hermetic one runs under the gate's `-m "not live_vault"` rung, so a round that moves
# the store and not the rule is caught without the vault; the live one pins the shipped text.
# ---------------------------------------------------------------------------

_SWEEP_SCRIPT = ROOT / "scripts" / "groundskeeper" / "retention-sweep.py"


def _load_sweep_module():
    """Load the groundskeeper script by path — a hyphenated filename with no package, so
    plain ``import`` cannot reach it. Same loader tests/test_retention_sweep.py uses.

    Loaded the way PRODUCTION runs it, because production is the pair the skill text
    names: ``TRANSCRIPT_DIR="$HOME/lloyd-data/_pipeline/tmp"`` is the marked live root,
    which the sweep reaches there by rule 2. #1415 changed what loading this file does:
    it resolved ``${LLOYD_DATA:-$HOME/lloyd-data}`` and so reported that path from every
    tree — including a worktree, whose sweep must never reach the live root — while it
    now follows ``app.paths``' three rules, so from this checkout its own answer is
    ``<tree>/.lloyd-data/_pipeline/tmp``. That answer is right for a sweep run here and
    wrong to compare against a production literal, so the two inputs rule 2 reads are
    pinned rather than inherited: the tree is the live checkout, and it is not a linked
    worktree. ``LLOYD_DATA`` stays popped — cron has no such variable, and the suite
    points it at scratch.
    """
    import importlib.util
    from unittest import mock

    spec = importlib.util.spec_from_file_location(
        "retention_sweep_under_artifact_test", _SWEEP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    saved = os.environ.pop("LLOYD_DATA", None)
    try:
        with mock.patch.object(dr, "live_checkout", lambda: ROOT), \
                mock.patch.object(dr, "tree_is_worktree", lambda tree: False):
            spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            os.environ["LLOYD_DATA"] = saved
    return mod


def test_the_sweep_constant_still_satisfies_the_pinned_requirement():
    """What the sweep actually bounds must be the directory the skill rule pins, checked
    from the repo side alone so it RUNS under the gate."""
    mod = _load_sweep_module()
    # The account home from passwd, not `Path.home()`: the sweep's root is
    # passwd-anchored for the reason #1415 is about, and under a gate's
    # `HOME=<round>/home` the two differ — where `$HOME` says the round, passwd says
    # the machine this skill text describes.
    literal = str(mod.TRANSCRIPT_SCRATCH_DIR).replace(str(dr.ACCOUNT_HOME), "$HOME")
    named = ARTIFACT_PHASE_REQUIREMENTS["the one named transcript scratch dir"]
    assert named.search(f'TRANSCRIPT_DIR="{literal}"'), (
        f"retention-sweep.py bounds {mod.TRANSCRIPT_SCRATCH_DIR} but the skill requirement "
        "pins a different directory — one of the two is renamed and the scratch store now "
        "has no owner")


def test_the_scratch_store_is_neither_tmp_nor_the_vault():
    """#566's location clause, from the constant alone. `/tmp` is reaped daily by
    ``systemd-tmpfiles-clean.timer`` — that is how ``/tmp/yt/6IFVTcM28KA.txt`` was set to
    vanish — and the vault is Obsidian-synced, where 40 KB raw transcripts at ~48 videos a
    day is not what the sync quota is for. A durable artifact belongs in the note; the input
    belongs somewhere Lloyd owns and sweeps."""
    mod = _load_sweep_module()
    scratch: Path = mod.TRANSCRIPT_SCRATCH_DIR
    assert not str(scratch).startswith("/tmp"), f"{scratch} is under the tmpfiles reaper"
    assert Path.home() / "obsidian" not in scratch.parents, f"{scratch} is inside the vault"


@pytest.mark.live_vault
@pytest.mark.parametrize("skill", SKILLS_UNDER_TEST)
def test_the_skill_and_the_sweep_name_the_same_scratch_dir(skill: str):
    """The shipped seam, read off both files: the directory a session writes to and the
    directory the sweep bounds must be one directory."""
    m = re.search(r'TRANSCRIPT_DIR\s*=\s*"?\$HOME/(?P<rel>[^"\n]+)', _require_skill_text(skill))
    assert m, f"skills/{skill}/SKILL.md no longer declares TRANSCRIPT_DIR"
    mod = _load_sweep_module()
    assert mod.TRANSCRIPT_SCRATCH_DIR == dr.ACCOUNT_HOME / m.group("rel"), (
        f"skills/{skill}/SKILL.md saves to ~/{m.group('rel')} but "
        f"retention-sweep.py bounds {mod.TRANSCRIPT_SCRATCH_DIR}")


@pytest.mark.live_vault
def test_the_retention_skill_table_states_the_sweep_age():
    """The policy table in skills/retention-sweep/SKILL.md is what the weekly operator reads
    before running --apply. If its number for the scratch store drifts from
    ``TRANSCRIPT_MAX_AGE_DAYS``, the operator is approving an age the script does not
    implement. Every row naming the store is graded, not the first one: a second row with a
    different age is the drift, and reading only rows[0] would let it sit there unread."""
    mod = _load_sweep_module()
    # passwd's home again (`_load_sweep_module`'s note): the table row is written for
    # the machine, and under a gate's HOME the two homes are different directories.
    needle = "~/" + str(mod.TRANSCRIPT_SCRATCH_DIR.relative_to(dr.ACCOUNT_HOME))
    rows = [ln for ln in _require_skill_text("retention-sweep").splitlines() if needle in ln]
    assert rows, f"no retention-sweep table row names {needle}"
    for row in rows:
        stated = re.search(r">\s*(\d+)\s*d", row)
        assert stated, f"row for {needle} states no age: {row!r}"
        assert int(stated.group(1)) == mod.TRANSCRIPT_MAX_AGE_DAYS, (
            f"skills/retention-sweep/SKILL.md says >{stated.group(1)}d but "
            f"retention-sweep.py uses TRANSCRIPT_MAX_AGE_DAYS={mod.TRANSCRIPT_MAX_AGE_DAYS}")


@pytest.mark.live_vault
@pytest.mark.parametrize("skill", SKILLS_UNDER_TEST)
def test_extraction_skill_still_loads(skill: str):
    """The pin must not be satisfiable by a file the loader can no longer parse —
    front matter that stops parsing means the skill is not advertised at all."""
    from prompt_builder import _is_quarantined_skill

    path = _skill_file(skill)
    assert path is not None, f"skills/{skill}/SKILL.md is not installed"
    body = path.read_text(encoding="utf-8", errors="replace")
    assert body.startswith("---"), f"{skill}: front matter must parse"
    assert not _is_quarantined_skill(path), f"{skill} is quarantined and would not load"
