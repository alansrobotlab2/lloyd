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

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Vault skills first, then the repo checkout — same precedence as
# tests/test_skill_tool_names.py and prompt_builder._load_skills_index.
SKILLS_DIRS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]

# The two skills the item names. Deliberately not every research skill: the others already
# had the rule before #448 existed, and asserting on them here would fold their history into
# this item's pin.
SKILLS_UNDER_TEST = ["youtube-content", "youtube-transcript"]

# What the Artifact Phase must state, as (requirement, pattern). Each one is a distinct
# failure mode from the item, not a stylistic preference.
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
        re.compile(r'TRANSCRIPT_DIR\s*=\s*"?\$HOME/lloyd/_pipeline/tmp'),
    # Naming a directory is not saving to it: yt-dlp needs `-o "$TRANSCRIPT_DIR/%(id)s"`
    # rather than the bare cwd-relative `%(id)s`, and the stdout-only helper script needs a
    # redirect. This is the half that actually moves the bytes.
    "a save step that writes into that dir":
        re.compile(r'(?:-o\s+"|>\s*")\$TRANSCRIPT_DIR/'),
    # The scratch file is swept (see sweep_transcript_scratch), so the note must carry the
    # pointer. Key-line form on purpose: the prose names these keys in backticks, and a note
    # that only ever said it in prose would leave the input unprovable.
    "transcript_path as a front-matter key": re.compile(r"^transcript_path:\s+\S", re.MULTILINE),
    # ...and the digest, so a swept or edited transcript is detectable from the note alone.
    "transcript_md5 as a front-matter key": re.compile(r"^transcript_md5:\s+\S", re.MULTILINE),
    # Which of the two files is the artifact. Stated, because "save the transcript too" is
    # what made #448 read as satisfied while the input rotting in gitignore stayed invisible.
    "the note is durable and the scratch file disposable":
        re.compile(r"durable artifact[\s\S]{0,200}disposable|disposable[\s\S]{0,200}durable artifact"),
}


def missing_artifact_phase(text: str) -> list[str]:
    """Return every Artifact-Phase requirement ``text`` fails to state.

    Pure over skill text, so the gate can run it on fixtures and the live_vault cases can run
    it on the real files with no duplicated logic.
    """
    return [
        label
        for label, pattern in ARTIFACT_PHASE_REQUIREMENTS.items()
        if not pattern.search(text)
    ]


def _skill_file(name: str) -> Path | None:
    for root in SKILLS_DIRS:
        candidate = root / name / "SKILL.md"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# hermetic — these RUN under the automod gate (`-m "not live_vault"`)
# ---------------------------------------------------------------------------

_CANONICAL = """
## Artifact Phase — Persist the digest (HARD — backlog #448)

A digest that exists only in the chat transcript does not exist. Auto-capture compresses
whole days and `~/lloyd/_pipeline/tmp/` is gitignored, so neither one is persistence.
Write the derived output to `~/obsidian/knowledge/youtube/<Channel>/<YYYYMMDD>-<slug>.md`
**before** reporting anything.

The raw transcript goes to the one scratch dir — never a bare `-o "%(id)s"`, never `/tmp`:

```bash
TRANSCRIPT_DIR="$HOME/lloyd/_pipeline/tmp"
mkdir -p "$TRANSCRIPT_DIR"
yt-dlp --write-auto-sub --skip-download --sub-lang en -o "$TRANSCRIPT_DIR/%(id)s" "$URL"
```

The vault note is the durable artifact; the scratch file is disposable.

```yaml
type: video-note
video_id: <videoId>
transcript_path: /home/alansrobotlab/lloyd/_pipeline/tmp/<videoId>.txt
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
    'TRANSCRIPT_DIR="$HOME/lloyd/_pipeline/tmp"\n',
    '-o "$TRANSCRIPT_DIR/%(id)s" ',
    "transcript_path: /home/alansrobotlab/lloyd/_pipeline/tmp/<videoId>.txt\n",
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
    path = _skill_file(skill)
    if path is None:
        pytest.skip(f"{skill} not installed on this machine")
    missing = missing_artifact_phase(path.read_text(encoding="utf-8", errors="replace"))
    assert missing == [], (
        f"skills/{skill}/SKILL.md lost its Artifact Phase requirements: {missing}. "
        "A transcript digest written only to the chat transcript is lost when the session "
        "ends (backlog #448); the phase must name a vault path, a verification step, and "
        "the path report."
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
    plain ``import`` cannot reach it. Same loader tests/test_retention_sweep.py uses."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "retention_sweep_under_artifact_test", _SWEEP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_sweep_constant_still_satisfies_the_pinned_requirement():
    """What the sweep actually bounds must be the directory the skill rule pins, checked
    from the repo side alone so it RUNS under the gate."""
    mod = _load_sweep_module()
    literal = str(mod.TRANSCRIPT_SCRATCH_DIR).replace(str(Path.home()), "$HOME")
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
    path = _skill_file(skill)
    if path is None:
        pytest.skip(f"{skill} not installed on this machine")
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'TRANSCRIPT_DIR\s*=\s*"?\$HOME/(?P<rel>[^"\n]+)', text)
    assert m, f"skills/{skill}/SKILL.md no longer declares TRANSCRIPT_DIR"
    mod = _load_sweep_module()
    assert mod.TRANSCRIPT_SCRATCH_DIR == Path.home() / m.group("rel"), (
        f"skills/{skill}/SKILL.md saves to ~/{m.group('rel')} but "
        f"retention-sweep.py bounds {mod.TRANSCRIPT_SCRATCH_DIR}")


@pytest.mark.live_vault
def test_the_retention_skill_table_states_the_sweep_age():
    """The policy table in skills/retention-sweep/SKILL.md is what the weekly operator reads
    before running --apply. If its number for the scratch store drifts from
    ``TRANSCRIPT_MAX_AGE_DAYS``, the operator is approving an age the script does not
    implement."""
    path = _skill_file("retention-sweep")
    if path is None:
        pytest.skip("retention-sweep skill not installed on this machine")
    mod = _load_sweep_module()
    needle = "~/" + str(mod.TRANSCRIPT_SCRATCH_DIR.relative_to(Path.home()))
    rows = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if needle in ln]
    assert rows, f"no retention-sweep table row names {needle}"
    stated = re.search(r">\s*(\d+)\s*d", rows[0])
    assert stated, f"row for {needle} states no age: {rows[0]!r}"
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
    if path is None:
        pytest.skip(f"{skill} not installed on this machine")
    body = path.read_text(encoding="utf-8", errors="replace")
    assert body.startswith("---"), f"{skill}: front matter must parse"
    assert not _is_quarantined_skill(path), f"{skill} is quarantined and would not load"
