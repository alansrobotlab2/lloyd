"""The commit-time witness warning (#2381, the recurrence tripwire for #2284).

A `live_vault` node can pin a vault file WHOLE: vault `e45a6b6c` appended one bullet
to `skills/nightly-reflection-knowledge-write/step-2a-ter-curation.md` on 2026-10-06,
its committed witness `tests/fixtures/step_2a_ter_curation_witness_2212.md` was never
re-frozen, and `tests/test_memory_ledger_bound.py` went red only at the next
morning's #83 pre-flight. The committing job had no signal that the file it appended
to was frozen anywhere.

`scripts/util/witness_fixture_findings.py` is that signal, run by
`scripts/util/vault-commit.sh` between staging and `git commit`. These nodes pin the
four things that make it worth having: the matcher answers the item's own path
correctly, the warning carries both sides' figures, the warning never blocks the
commit, and an ordinary commit that matches nothing stays silent. The last one is
not a throwaway: a nightly commit almost never stages a witnessed file, so a rung
that always printed would be an alarm nobody reads.

The last two nodes read the live vault, not a fixture, because clauses 4 and 5 are
a sentence in a skill the nightly job actually follows — the precedent for pinning
that surface is `tests/test_vault_round_skill_gate.py::test_the_knowledge_write_skill_carries_a_memory_index_pre_flight`.
They are marked `live_vault` for the reason that precedent gives: the skill body is
owned by the nightly skills pass, and an unmarked node reddens the next author's
gate for the previous writer's edit. Run them with:

    .venvs/lloyd/bin/python -m pytest -m live_vault \\
        tests/test_vault_commit_witness_warning.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import skill_lint as SL
from scripts.util import witness_fixture_findings as rung

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "util" / "vault-commit.sh"
RUNG = REPO_ROOT / "scripts" / "util" / "witness_fixture_findings.py"
LIVE_VAULT = Path.home() / "obsidian"
KNOWLEDGE_WRITE = "skills/nightly-reflection-knowledge-write/SKILL.md"

#: The one witnessed vault path on this box today, and the fixture frozen for it.
#: Both are named literally because the normalization being pinned is exactly the
#: relation between the two spellings: `-` -> `_`, extension dropped.
WITNESSED = "skills/nightly-reflection-knowledge-write/step-2a-ter-curation.md"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "step_2a_ter_curation_witness_2212.md"

#: A stem no fixture can contain: the literal is this item's number plus a word no
#: fixture name uses, so the zero-hit answer cannot be a glob that silently matched
#: nothing because the corpus failed to load. Spelled without the word "witness" so
#: the silent-commit node below can grep the wrapper's whole output for the rung and
#: not have the file's own name answer it.
UNMATCHED = "knowledge/zz-unmatched-note-2381.md"

#: The staged skill's content for the commit-time nodes. Deliberately NOT the
#: fixture's text: the warning has to report the two sides' counts, and the way to
#: show it is not reporting one number twice is to stage a file of a different size.
STAGED_TEXT = ("# §2a-ter curation (staged by a test)\n\n"
               "- the nightly appended this bullet\n"
               "- and its witness was not re-frozen\n")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"git {' '.join(args)}: {proc.stderr}"
    return proc.stdout


def _env(repo: Path) -> dict:
    """Minimal env, as `tests/test_vault_commit_attribution.py` does it: no HOME so
    no `~/.gitconfig` leaks an identity, and `LLOYD_PYTHON` names this interpreter
    for the rungs the wrapper runs."""
    return {"VAULT_DIR": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin",
            "LLOYD_PYTHON": sys.executable}


@pytest.fixture
def vault(tmp_path):
    """A real git repo on `main` with one committed seed file."""
    repo = tmp_path / "vault"
    (repo / "skills" / "nightly-reflection-knowledge-write").mkdir(parents=True)
    (repo / "knowledge").mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    seed = repo / "seed.md"
    seed.write_text("committed before the run\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _write(repo: Path, rel: str, text: str = STAGED_TEXT) -> Path:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _commit(repo: Path, msg: str, *paths: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(WRAPPER), msg, "--", *paths],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(repo), env=_env(repo))


def _run_rung(repo: Path, *paths: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUNG), "--repo", str(repo),
                           *sum((("--path", p) for p in paths), ())],
                          capture_output=True, text=True, timeout=60,
                          env=_env(repo))


def _committed_paths(repo: Path) -> list[str]:
    out = _git(repo, "show", "--format=", "--name-only", "HEAD")
    return [line for line in out.splitlines() if line.strip()]


def _figures(data: bytes) -> tuple[int, int]:
    """(bytes, newlines) measured here rather than by the rung's own helper, so the
    warning's numbers are checked against an independent measurement."""
    return len(data), data.count(b"\n")


# ---------------------------------------------------------------------------
# Clause 1: the matcher over a staged-path list.
# ---------------------------------------------------------------------------

def test_the_matcher_reports_the_one_fixture_frozen_for_the_curation_skill():
    """`skills/…/step-2a-ter-curation.md` -> `step_2a_ter_curation_witness_2212.md`.

    The fixture corpus is this checkout's real `tests/fixtures`, so the node pins
    the naming convention as it is actually used and not a mock of it. Exactly one
    hit: a matcher that returned the fixture twice, or it plus a sibling, would have
    the committing job triage a list instead of one file.
    """
    assert FIXTURE.is_file(), (
        f"{FIXTURE} is the committed witness the §2a-ter drift (#2284) was caught "
        "by; without it the normalization below has nothing to normalize against")
    hits = rung.witness_matches([WITNESSED])
    assert [fx.name for _rel, fx in hits] == [FIXTURE.name], (
        f"the matcher answered {hits} for the one staged path that has a witness")
    assert hits[0][0] == WITNESSED, "the hit must name the staged path it matched on"


def test_the_matcher_reports_no_hit_for_a_stem_no_fixture_name_contains():
    """The zero side of clause 1, proven against a corpus the node itself lists.

    `witness_fixtures` answers `[]` for a missing directory exactly as it does for an
    unmatched stem, so a zero-hit assertion alone would pass on an unreadable corpus.
    The listing is therefore taken inside this node and the known fixture is required
    to be in it before the zero answer means anything.
    """
    listing = sorted(p.name for p in rung.FIXTURES_DIR.glob("*.md"))
    assert FIXTURE.name in listing, (
        f"{FIXTURE.name} is not in the corpus this node reads ({listing}), so a "
        "zero-hit answer below would prove only that the directory is unreadable")
    assert rung.witness_fixtures(rung.stem_of(UNMATCHED)) == [], (
        "a stem containing this item's number matched a fixture, so the corpus "
        "listing is answering something other than the name")
    assert rung.witness_matches([UNMATCHED]) == []


def test_the_stem_is_the_basename_with_underscores_and_no_extension():
    """The normalization rule itself: `-` -> `_`, extension dropped, basename only.

    Pinned directly because the two ends of the item's example differ in three ways
    at once (directory, dashes, extension), and a rule that only survives one of
    them would still match the example by accident.
    """
    assert rung.stem_of(WITNESSED) == "step_2a_ter_curation"
    assert rung.stem_of("knowledge/a-b-c.md") == "a_b_c"
    assert rung.stem_of("knowledge/no-extension") == "no-extension".replace("-", "_")


def test_the_matcher_looks_only_under_skills_and_knowledge():
    """The rung's own blast radius: a `memory/` file sharing a witnessed basename is
    not the file any witness was frozen from, so it must stay silent. `#2284` froze
    a SKILL-tree file; nothing in the convention freezes a memory note by stem."""
    same_stem = f"memory/{Path(WITNESSED).name}"
    assert rung.witness_matches([same_stem]) == [], (
        f"{same_stem} matched a witness, so the segment guard is not applied")


# ---------------------------------------------------------------------------
# Clause 2: a commit that stages a witnessed path warns, with both counts, and
# still commits.
# ---------------------------------------------------------------------------

def test_a_commit_staging_a_witnessed_skill_warns_with_both_counts_and_completes(vault):
    """The #2284 scenario, run end to end through the real wrapper.

    The staged file here is a different size than the fixture on purpose: the point
    of the line is that the reader can see the two sides apart without running `wc`,
    so the counts are asserted as measured from disk here, not by the rung's own
    helper. Exit status 0 and a commit carrying the path are clause 2's other half —
    a warning that refused the commit would be a new hard gate, which the item
    rules out.
    """
    path = _write(vault, WITNESSED)
    staged_bytes, staged_newlines = _figures(path.read_bytes())
    fixture_bytes, fixture_newlines = _figures(FIXTURE.read_bytes())

    proc = _commit(vault, "nightly-reflection-knowledge-write: appended a bullet",
                   "skills/")
    assert proc.returncode == 0, f"the warning blocked a commit: {proc.stderr}"
    assert WITNESSED in _committed_paths(vault), (
        f"the commit did not carry the witnessed path: {proc.stdout}{proc.stderr}")

    warnings = [line for line in proc.stdout.splitlines()
                if line.startswith("witness WARNING")]
    assert len(warnings) == 1, (
        f"expected exactly one witness warning, got {warnings} on {proc.stdout!r}")
    line = warnings[0]
    assert FIXTURE.name in line, line
    assert WITNESSED in line, line
    assert f"fixture {fixture_bytes} B / {fixture_newlines} newlines" in line, line
    assert f"staged {staged_bytes} B / {staged_newlines} newlines" in line, line
    # The re-freeze recipe and the disclosure duty, so a job that reads only its own
    # commit output can file the draft without opening the suite.
    for fragment in ("backlog draft", "`wc -c`", "tr -cd '\\n' | wc -c",
                     "pinning node"):
        assert fragment in proc.stdout, f"{fragment!r} missing from {proc.stdout!r}"


def test_the_witness_warning_goes_to_stdout_and_never_to_stderr(vault):
    """The #1070 seam: the `unattributed dirty state:` path list is the last thing
    the WRAPPER writes to stderr, so a job can copy its tail, and
    `test_vault_commit_attribution.py` pins that. A warning appended to stderr after
    it would sit inside the copyable block, so the rung prints to stdout like the
    two rungs before it.

    The commit here is unscoped, which is what produces the unattributed block in
    the first place. A rung's own import noise can trail the list (in a round tree
    `app.paths` warns that it is anchored to a worktree), so what is pinned is that
    no line of the warning is on stderr at all and that the indented entries after
    the header are exactly the staged paths, in the index's own order.
    """
    _write(vault, "skills/nightly-reflection-knowledge-write/step-2a-ter-curation.md")
    (vault / "memory").mkdir(exist_ok=True)
    (vault / "memory" / "someone-elses-note.md").write_text("not this job's\n",
                                                            encoding="utf-8")
    proc = subprocess.run(["bash", str(WRAPPER), "job: snapshot"],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(vault), env=_env(vault))
    assert proc.returncode == 0, proc.stderr
    # The rung's two line prefixes, checked at line start because the emitted
    # post-flight command quotes this test's own tmp path, which is named after the
    # node and so contains the word.
    assert [line for line in proc.stdout.splitlines()
            if line.startswith("witness WARNING")], (
        f"the rung did not fire at all, so the stderr half proved nothing: {proc.stdout!r}")
    assert not [line for line in proc.stderr.splitlines()
                if line.startswith(("witness WARNING", "witness: CHECK"))], (
        f"the warning reached stderr and now sits under the #1070 tail: {proc.stderr!r}")
    assert "unattributed dirty state:" in proc.stderr
    # Of the lines the WRAPPER itself printed (a prefixed line — the rungs' import
    # noise is not the wrapper's voice, and in a round tree one of them emits an
    # `app.paths` worktree warning after the list), the last is the unattributed
    # header, whose indented entries are what the job copies.
    wrapper_lines = [line for line in proc.stderr.splitlines()
                     if line.startswith("vault-commit.sh:")]
    assert wrapper_lines[-1].startswith("vault-commit.sh: unattributed dirty state:"), (
        f"the wrapper's last stderr line is not the copyable list: {wrapper_lines!r}")
    _header, listed_block = proc.stderr.split(
        "copy this exact list into the job's run record:\n", 1)
    assert [line.strip() for line in listed_block.splitlines()
            if line.startswith("    ")] == ["memory/someone-elses-note.md", WITNESSED], (
        f"the copyable list is not the block's tail: {proc.stderr!r}")


# ---------------------------------------------------------------------------
# Clause 3: nothing matches, nothing is printed, the commit is unchanged.
# ---------------------------------------------------------------------------

def test_a_commit_staging_no_witnessed_path_prints_no_witness_line_and_completes(vault):
    """The ordinary nightly commit: skills/ and knowledge/ both staged, neither
    witnessed, and the wrapper's output carries no witness line at all.

    This is the case that decides whether the rung gets read. Eleven knowledge-write
    commits in the last fourteen days touch `skills/`, and exactly one path in the
    whole tree has a fixture, so a rung that announced itself every night would be
    background noise inside a week.
    """
    _write(vault, "skills/nightly-reflection-knowledge-write/SKILL.md", "# skill\n")
    _write(vault, UNMATCHED, "# a note nobody froze\n")
    proc = _commit(vault, "nightly-reflection-knowledge-write: routine write",
                   "skills/", "knowledge/")
    assert proc.returncode == 0, proc.stderr
    assert "witness" not in proc.stdout.lower(), proc.stdout
    assert "witness" not in proc.stderr.lower(), proc.stderr
    assert sorted(_committed_paths(vault)) == sorted(
        [UNMATCHED, "skills/nightly-reflection-knowledge-write/SKILL.md"])


def test_the_rung_dry_run_over_an_explicit_path_list_warns_then_stays_silent(vault):
    """Clause 5's check as a node: the matcher is reachable on its own, given a path
    list, without staging anything or running a commit.

    Same two answers as the commit-time route — the fixture and both counts for the
    witnessed path, empty output for the unmatched one — because the check the item
    asks for is of the matcher, not of the wrapper.
    """
    _write(vault, WITNESSED)
    _write(vault, UNMATCHED, "# a note nobody froze\n")
    fixture_bytes, fixture_newlines = _figures(FIXTURE.read_bytes())
    staged_bytes, staged_newlines = _figures((vault / WITNESSED).read_bytes())

    hit = _run_rung(vault, WITNESSED)
    assert hit.returncode == 0, hit.stderr
    assert FIXTURE.name in hit.stdout
    assert f"fixture {fixture_bytes} B / {fixture_newlines} newlines" in hit.stdout
    assert f"staged {staged_bytes} B / {staged_newlines} newlines" in hit.stdout

    miss = _run_rung(vault, UNMATCHED)
    assert miss.returncode == 0, miss.stderr
    assert miss.stdout == "", f"an unmatched path printed: {miss.stdout!r}"


# ---------------------------------------------------------------------------
# Clauses 4 and 5: the guardrail the committing job has to follow, read off the
# live skill page.
# ---------------------------------------------------------------------------

def _witness_guardrail() -> str:
    """The `### Whole-File Witness Guardrail` section, from its heading to the next
    `##` — the text the knowledge-write job reads between Published-Path
    Verification and its Completion Report."""
    text = (LIVE_VAULT / KNOWLEDGE_WRITE).read_text(encoding="utf-8")
    heading = "### Whole-File Witness Guardrail"
    assert heading in text, (
        "the knowledge-write skill has no Whole-File Witness Guardrail at all")
    start = text.index(heading)
    rest = text[start + len(heading):]
    end = rest.index("\n## ") if "\n## " in rest else len(rest)
    return heading + rest[:end]


@pytest.mark.live_vault
def test_the_guardrail_sits_beside_published_path_verification():
    """Clause 4's placement, and the body-length ceiling the landing rail enforces.

    Placement is what makes it readable: the guardrail belongs where the job reads
    it, between the Published-Path rule that is already a guardrail on this same
    commit and the Completion Report that closes the run. The ceiling is measured
    with `skill_lint`'s own ruler, the one `vault_round.skill_body_findings` refuses
    on, because the guardrail's landing is what that ruler is for here: two landed
    lines took this skill to 92 body lines, which the 100-line landing rail passed
    while the repo's stricter ceiling on this same skill — 90, owned by
    `tests/test_skill_lint_size.py::test_the_skill_that_sat_at_the_cap_now_has_headroom`,
    whose number this node deliberately does not copy — went red. §Step 1 was
    reflowed to pay for the guardrail rather than the ceiling moved.
    """
    text = (LIVE_VAULT / KNOWLEDGE_WRITE).read_text(encoding="utf-8")
    heading = "### Whole-File Witness Guardrail"
    assert heading in text, (
        "the knowledge-write skill carries no Whole-File Witness Guardrail, so a "
        "nightly append to a witnessed skill has nothing telling it what to file")
    published = text.index("### Published-Path Verification Guardrail")
    completion = text.index("## Completion Report")
    guardrail = text.index(heading)
    assert published < guardrail < completion, (
        f"guardrail at {guardrail} is not between Published-Path Verification at "
        f"{published} and the Completion Report at {completion}")
    _, body = SL.parse_frontmatter(text)[:2]
    lines = SL.skill_size("nightly-reflection-knowledge-write",
                          LIVE_VAULT / KNOWLEDGE_WRITE, text, body)["body_lines"]
    assert lines <= SL.MAX_BODY_LINES, (
        f"{lines} body lines against the {SL.MAX_BODY_LINES}-line landing rail: the "
        "guardrail could not land, and #2381's vault half has eaten the headroom "
        "the repo's own ceiling on this skill is there to protect")


@pytest.mark.live_vault
def test_the_guardrail_says_a_witnessed_file_may_be_frozen_and_the_writer_must_not_refreeze():
    """Clause 4: the guardrail tells the job a file it edits may be frozen whole,
    that re-freezing is NOT its call, and what to file instead.

    The four things pinned are the four the job can act on with no other reading:
    the witness convention (`*_witness_*.md`, and the `-` -> `_` stem rule that
    decides what matches), the byte-compare by a `live_vault` node, the prohibition,
    and the draft it must file — which has to name the fixture and BOTH sides'
    counts, because those are the figures a re-freeze is measured against.
    """
    section = _witness_guardrail()
    assert "_witness_" in section, "the fixture naming convention is not stated"
    assert "byte-compar" in section and "live_vault" in section, (
        "the guardrail does not say a live_vault node byte-compares the pair")
    assert "does NOT re-freeze" in section, "the prohibition on re-freezing is gone"
    assert "one-line backlog draft" in section, "no filing duty is stated"
    for fragment in ("byte and newline counts", "`wc -c`",
                     "tr -cd '\\n' | wc -c",
                     "test_the_curation_skill_witness_carries_no_ledger_cap_at_the_topic_ceiling"):
        assert fragment in section, f"{fragment!r} missing from the recipe: {section}"


@pytest.mark.live_vault
def test_the_guardrail_records_2284_and_refuses_an_automatic_refreeze_path():
    """Clause 5: the paragraph keeps its incident and its ruling.

    The incident is the reason the paragraph exists, so its identifiers are pinned:
    item #2284, vault commit `e45a6b6c`, the line-61 bullet it appended, that the
    witness was never re-frozen, and that discovery waited for the next morning's #83
    pre-flight. The ruling is the half that keeps the next round from "fixing" this
    one by generating the fixture: no automatic re-generation path, because the byte
    pin is what makes a re-freeze a reviewed act.
    """
    section = _witness_guardrail()
    for fragment in ("#2284", "e45a6b6c", "line-61 bullet", "never re-frozen",
                     "next morning"):
        assert fragment in section, f"{fragment!r} missing from {section}"
    assert "out of the question" in section or "never be added" in section, (
        f"the no-auto-regeneration ruling is gone: {section}")
    assert "reviewed act" in section, (
        "the guardrail must state WHY there is no auto path, not just that there "
        f"is none: {section}")
