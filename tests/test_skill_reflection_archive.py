"""Backlog #436 — nightly reflection reports must be archived before they are
overwritten, and the archive name must not come from a hand-written stamp.

Why this is a code problem and not a skill problem
--------------------------------------------------
The nightly chain writes three live reports in place under
``~/lloyd/_pipeline/reflection/``: ``signals-latest.md``,
``tool-patterns-latest.md`` and ``conversation-patterns-latest.md``. That
directory is gitignored (``~/lloyd/.gitignore:25`` ``/_pipeline/``;
``git ls-files _pipeline | wc -l`` → 0), so an overwrite with no prior copy
destroys the only version that has ever existed — the same irreversibility that
made the 2026-08-22 entity-graph destruction unrecoverable.

Measured at triage (2026-09-12T10:15Z), against ~22 nightly cycles since 08-22:
``signals-latest*`` → 7 files (6 dated copies), and **zero** dated copies for
either pattern file. So the retention rule was enforced by nothing.

The first fix (vault commit ``3514dac1``, 2026-09-12) added an archive ``cp`` to
one skill and only for ``signals-latest.md``. Prose one model reads once per run
is what the remaining cycles were still relying on, and #436's own merged finding
says so in the skill's own words: "A Read-before-Write discipline that preserves
a file only while the reader remembers to copy it is not a retention policy."
This file is the instrument: the archive step is now pinned, and deleting it is
refused at the writer that would land the change.

Where the rule lives, and where it runs
--------------------------------------
The rule itself is ``scripts/reflection_archive.py`` — one definition, two consumers.
``scripts/automod/vault_round.py`` calls it on every ``skills/**/SKILL.md`` a
vault round touches, so dropping an archive step is refused at
``automod_vault_land`` before the commit and the round's paths are reverted.
**That is the automated rung this rule is enforced at**, and
``test_the_vault_writer_refuses_a_skill_that_dropped_its_archive_step`` below
drives ``validate`` end to end to prove the wiring, not just the helper.

The assertions in this file that read ``~/obsidian/skills`` carry
``live_vault`` (``pytest.ini:6-12``): skill prose is state no round under test
controls, and the gate's hard ``tests`` rung runs ``-m "not live_vault"`` so an
autoresearch promotion or a nightly rewrite of some unrelated skill cannot fail
the next author's round for the previous writer's wording. Being honest about the
rest: the full suite *command* runs the marked group, and no job in the fleet runs
the full suite command — they name individual files — so those assertions are
executed by hand and by the gate's review rung, and backlog #979 owns giving
``live_vault`` a rung of its own. That is why the enforcement is at the writer and
not here; it is the architecture ``pytest.ini`` already states for the identity
surface ("Enforcement lives at the writers … this mark is the reporting copy"),
and ``reflection_archive`` exists so this rule can follow it. The unmarked tests
here — every mutation that proves the rule can fail, the writer wiring, and the
untracked-reports pin — run on every rung.

What each clause is pinned by
-----------------------------
1. *Every live report gets a dated copy before its overwrite, on every cycle* →
   ``test_every_skill_that_writes_a_reflection_report_archives_it_first``, over
   the writer set discovered by ``classify_all``. Clause 1's specific
   requirement — §2e of ``nightly-reflection-knowledge-write`` for both pattern
   files — is asserted separately in
   ``test_knowledge_write_section_2e_archives_both_pattern_files``.
2. *A test in the ``tests/test_skill_*.py`` family, failing when such a line is
   removed* → the same test, plus ``test_the_archive_check_can_actually_fail``,
   which mutates synthetic skill bodies and asserts the check reports each
   removal class, and the two writer-wiring tests that show a removal is
   *refused* and not merely reported. A pin that cannot fail is not a pin.
3. *The stamp must be UTC-derived, never a `generated:` field read out of the
   previous report* → ``test_archive_stamp_is_derived_from_a_utc_command`` (the
   destination is ``$STAMP``/``$(date …)`` and that variable is assigned from
   ``date -u … +%Y-%m-%d-%H%M`` on the same logical line) and
   ``test_no_skill_builds_an_archive_name_from_a_generated_field``.
4. *No gitignored file under ``~/lloyd`` is touched* → the diff is this file, one
   module and one validator hunk, all tracked paths, and
   ``test_retention_is_dated_copies_not_reports_tracked_in_the_repo`` fails if a
   generated report ever enters the index — the change that would mean retention
   was achieved by tracking ``_pipeline/`` instead of by dated copies. It asserts
   that and nothing more: whether ``_pipeline/`` is still *ignored* is repo state
   this item does not own and preflight will not let a round edit, and pinning it
   here made every later round that touches a comment in this file answer for
   somebody else's decision (review finding, round ``SM_20260912_163822``).
   ``_pipeline/`` staying ignored is what makes an unarchived overwrite
   unrecoverable — stated throughout, and pinned as a loop boundary by
   ``tests/test_automod_doc_claims.py``'s ``.gitignore`` denied-path row, not by
   this file. Relocating the reports into the vault is #436's other route and is
   Alan's decision, not a side effect.

Two more assertions hold up claims that are not clauses.
``test_the_exempt_tier_is_a_call_that_cannot_reach_the_lloyd_tree`` crosses the one
process boundary the rule's single exemption sits on — ``vault_write`` rejecting a
``~/lloyd/`` target is why a skill prescribing that call cannot destroy a report, and
until round ``SM_20260912_163822`` that rested on a note in ``lloyd/MEMORY.md``, which
is not a test. ``test_the_section_2e_guard_fails_on_every_way_it_could_be_satisfied_wrongly``
breaks the §2e text as landed, three ways, and shows the per-file guard names the broken
file each time while the two section-wide greps the previous round used stay silent —
that is the review's finding about clause 1 turned into an assertion.

How writers are recognised, and why the carve-outs are honest
-------------------------------------------------------------
A *writer* is a skill whose text instructs a write to the live report. Three
tiers, discovered from the same classifier, and asserted to equal the sets
recorded below — a new writer in **any** tier fails the suite rather than being
silently uncovered, and any mention that is neither a write nor a read lands in
``unclassified``, which must be empty, so an under-inclusive rule cannot hide a
writer.

The archive rule binds **two** tiers: a real ``Write``/``Edit`` call on the path,
and a prose instruction to overwrite it ("Merge all ``signals-*.md`` into …").
The distinction between them is only how the archive step has to be written, not
whether one is owed — #436's quantifier is about *skills that write a report*,
not about calls a run executes, so a rule binding only ``tool-write`` would let a
skill destroy all three reports in prose and still read as covered. The prose tier
is bound in the one way a prose writer can be bound: the step must be named in the
same skill text, and the vault writer refuses to land the skill without it.
``test_the_bound_tiers_are_the_two_that_can_lose_a_report`` pins ``BOUND_TIERS`` by
value, because narrowing that tuple narrows the clause.

``vault-write`` is the single exemption, and its reason is not a loophole: those
calls reject any ``~/lloyd/`` target with ``PATH_ESCAPE`` (recorded in
``nightly-reflection-signals`` Phase 0, ``nightly-reflection-knowledge-analysis``
step 1, and MEMORY.md), so nothing they prescribe can overwrite anything. Those
five (skill, report) pairs are re-verified every run and the discovered set must
equal the recorded one; if a skill there ever gains a real ``Write`` call it moves
into a bound tier and owes an archive step. The dead instructions themselves —
prescriptions that read as pipeline steps and cannot execute — are recorded on
#436 as findings, not widened into this diff.

The attribution boundary, stated plainly: a mention counts when it names the path
(``reflection/<name>-latest.md``). A skill that refers to ``test-results-latest.md``
by bare filename somewhere else is not attributed, because a bare filename cannot
be told apart from one in a different directory. That is the reason
``PROSE_WRITE`` carries verbs of instruction and not nouns like "output" — the
nouns are what descriptive annotations use, and three of the read-list items in
``autonomy-reflection-pipeline`` Step 6.1 are annotated that way.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reflection_archive as ra
from scripts.reflection_archive import archive_problems, classify, classify_all

SKILLS_DIRS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]


def _active_skills() -> dict[str, str]:
    """Every SKILL.md the prompt actually advertises → its body.

    Mirrors prompt_builder._load_skills_index: dot-prefixed directories are the
    archive, quarantined skills are out of circulation. The quarantine filter
    lives here and not in ``reflection_archive`` because the vault writer is
    handed the one path a round touched and must judge that file whether or not
    the index advertises it — a skill being un-listed is not a licence to lose a
    report.
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
                out[entry.name] = skill_file.read_text(encoding="utf-8", errors="replace")
    return out


@pytest.fixture(scope="module")
def skills() -> dict[str, str]:
    """Active skill bodies. Every test using this reads state no round controls, so
    those tests carry `live_vault`; an unreadable skill tree is a failure, not a skip
    — a green run that read nothing is exactly the vacuity this file argues against."""
    bodies = _active_skills()
    assert bodies, (
        f"no SKILL.md found under {SKILLS_DIRS} — the skills tree is what this check "
        "reads, so a run that found none proves nothing and must not pass"
    )
    return bodies


# --- The writers the archive rule binds, with the tier that binds them. ---------
# Discovered and asserted as a set, so a new writer of any reflection report has to
# be looked at rather than quietly inheriting "no archive step". Two of these
# overwrite with a `Write` call a run executes; `historical-knowledge-refresh`
# instructs the overwrite in prose and is bound anyway, because #436's quantifier is
# about skills that write reports rather than calls a run executes.
EXPECTED_BOUND_WRITERS: dict[str, tuple[str, set[str]]] = {
    "nightly-reflection-signals": ("tool-write", {"signals-latest"}),
    "nightly-reflection-knowledge-write": (
        "tool-write",
        {"tool-patterns-latest", "conversation-patterns-latest"},
    ),
    "historical-knowledge-refresh": (
        "prose-write",
        {"signals-latest", "tool-patterns-latest", "conversation-patterns-latest"},
    ),
}

# --- Tier 2: write prescriptions that are not machine-bindable, and why. --------
# Each entry is re-verified to still be true by
# test_declared_write_prescriptions_are_still_accurate, and the discovered tier-2
# set must equal this set. If a skill here gains a real Write call it moves to
# tier 1 and owes an archive step; both families are recorded as findings on #436.
#: Write prescriptions routed through vault_write / mem_write, which reject any
#: ~/lloyd/ target with PATH_ESCAPE — the instruction cannot destroy a report, so
#: the absence of an archive step costs nothing today.
#: (`autonomy-reflection-pipeline` names prompt-audit-latest and test-results-latest
#: only inside Step 6.1's *read* list, so they classify as reads; it prescribes
#: producing them by bare filename, which no path-shaped check can attribute.)
#: Two entries were dropped on 2026-09-20 (backlog #1270): `nightly-prompt-audit`
#: and `nightly-behavior-test` are `status: archived` since vault commit `978aa34b`
#: (backlog #900), so `_load_skill` returns None for them, the discovery loop can
#: never produce them, and the assert below — an equality, not a subset — was red at
#: the checkout while the gate's `-m "not live_vault"` deselected it. A retired
#: skill cannot hold an exemption: there is no instruction left to exempt.
#: tests/test_archived_skill_artifacts.py pins that rule for both tiers, so the
#: recurrence guard no longer depends on anyone remembering to prune this dict.
EXPECTED_VAULT_WRITE_WRITERS: dict[str, set[str]] = {
    "autonomy-reflection-pipeline": {"signals-latest", "day-end-synthesis-latest"},
    "nightly-day-end-synthesis": {"day-end-synthesis-latest"},
}

#: The recorded debt, **measured by archive presence rather than by tier**: bound
#: writers that still lack a named archive step. Computed from `archive_problems`
#: by test_the_recorded_debt_is_measured_by_archive_presence, which asserts it
#: equals this. Empty as of round SM_20260912_163822: `historical-knowledge-refresh`
#: was the last entry, and it now reads and dated-copies all three reports before
#: merging into them, in the same shape the nightly writers use.
#:
#: Measuring rather than listing is the point (review finding, round
#: SM_20260912_163822): keyed on classification tier, an expectation would keep
#: demanding a paid-off skill's name on #436 forever, so the debt could never
#: shrink by being paid. Keyed on presence, paying it removes the entry from the
#: measurement, and the only way to grow the set is to break an archive step.
EXPECTED_UNARCHIVED_DEBT: dict[str, set[str]] = {}


# --- Clause 1 + 2 ---------------------------------------------------------------
@pytest.mark.live_vault
def test_every_skill_that_writes_a_reflection_report_archives_it_first(skills):
    """The acceptance clause: every live nightly reflection report gets a dated
    copy before it is overwritten, on every cycle, pinned here rather than by the
    skill prose alone."""
    offenders = {
        f"{name}:{stem}": problems
        for (name, stem), tier in classify_all(skills).items()
        if tier in ra.BOUND_TIERS
        for problems in [archive_problems(skills[name], stem)]
        if problems
    }
    assert offenders == {}, (
        "skills write a nightly reflection report in place without an archive "
        f"step for that same path: {offenders}. `_pipeline/` is gitignored, so "
        "the overwritten report is unrecoverable (#436)."
    )


@pytest.mark.live_vault
def test_discovered_tool_writers_are_exactly_the_expected_set(skills):
    """Anti-drift for the classifier itself: a new bound writer, or one whose tier
    changed, is a decision someone has to record here.

    Carrying the tier in the expectation is what makes this a drift guard for the
    *rule* and not just for the file list: had the set been names-and-stems only,
    narrowing `BOUND_TIERS` back to `tool-write` would have dropped three entries
    and been answered by editing the expectation down to match, which is the
    regression `test_the_bound_tiers_are_the_two_that_can_lose_a_report` exists to
    prevent."""
    found: dict[str, tuple[str, set[str]]] = {}
    for (name, stem), tier in classify_all(skills).items():
        if tier in ra.BOUND_TIERS:
            prev_tier, stems = found.get(name, (tier, set()))
            assert prev_tier == tier, f"{name} carries two tiers: {prev_tier}, {tier}"
            stems.add(stem)
            found[name] = (tier, stems)
    assert found == EXPECTED_BOUND_WRITERS, (
        f"reflection-report writers changed: found {found}, expected "
        f"{EXPECTED_BOUND_WRITERS}. Every entry must archive before overwriting."
    )


@pytest.mark.live_vault
def test_knowledge_write_section_2e_archives_both_pattern_files(skills):
    """Clause 1 by name: §2e of nightly-reflection-knowledge-write, for each of
    the two pattern files — Read the existing file, copy it to a dated path, then
    write. This is the half that had never landed: zero dated copies of either
    pattern file across ~22 nightly cycles."""
    body = skills["nightly-reflection-knowledge-write"]
    section = body.split("### 2e.", 1)
    assert len(section) == 2, "nightly-reflection-knowledge-write lost its §2e"
    section = "### 2e." + section[1].split("\n### ", 1)[0]
    # One predicate per file, from the same function the mutation test breaks —
    # `section_2e_problems`, not `archive_problems`. The section covers two files,
    # so a text-wide search lets one file's Read/cp block discharge the other's:
    # clause 1 says "for each of", and the previous round's evidence for it was
    # refused precisely because deleting one file's `cp` and retargeting its `Read`
    # left both of its assertions green. See
    # `test_the_section_2e_guard_fails_on_every_way_it_could_be_satisfied_wrongly`.
    for stem in ("tool-patterns-latest", "conversation-patterns-latest"):
        assert f"_pipeline/reflection/{stem}.md" in section, f"§2e no longer writes {stem}.md"
        assert not section_2e_problems(section, stem), (
            f"§2e must Read and dated-copy {stem}.md before overwriting it: "
            f"{section_2e_problems(section, stem)}"
        )


# --- Clause 3 ------------------------------------------------------------------
@pytest.mark.live_vault
def test_archive_stamp_is_derived_from_a_utc_command(skills):
    """The archive name must come from `date -u … +%Y-%m-%d-%H%M` (or the source's
    mtime) — never a hand-typed date."""
    problems: dict[str, list[str]] = {}
    for (name, stem), tier in classify_all(skills).items():
        if tier not in ra.BOUND_TIERS:
            continue
        bad = [p for p in archive_problems(skills[name], stem) if "UTC" in p]
        if bad:
            problems[f"{name}:{stem}"] = bad
    assert problems == {}, f"archive stamps are not UTC-derived: {problems}"


def _sentences_mentioning(text: str, token: str) -> list[str]:
    """Sentences containing `token`, on text with blockquote markers stripped.

    Sentence boundaries are rough by design (`.md` contains a period), which is
    safe here: the rule only ever applies to a sentence that also names a
    filename, and a fragment that loses the negation loses the `.md` with it.
    """
    flat = re.sub(r"^\s*> ?", "", text, flags=re.MULTILINE)
    flat = re.sub(r"\s+", " ", flat)
    return [s for s in re.split(r"(?<=[.!?:])\s+", flat) if token in s]


def _generated_field_offenders(text: str) -> list[str]:
    """Sentences that instruct naming a file from a `generated:` header.

    A negation is the difference between a rule and a prohibition on the rule: the
    skills now spend several lines explaining why the old wording was wrong, and
    those sentences name both the field and a filename.
    """
    return [
        s for s in _sentences_mentioning(text, "generated:")
        if ".md" in s and not re.search(r"\b(?:never|not|no longer|isn't|cannot)\b", s, re.I)
    ]


@pytest.mark.live_vault
def test_no_skill_builds_an_archive_name_from_a_generated_field(skills):
    """Clause 3's negative half. The report's own `generated:` header is a local
    (PST) clock reading, so a copy named from it sorts 8 hours out from the
    UTC-named siblings in the same directory and cannot be re-derived when the
    field is missing. A skill may *say* this — with a negation — but must not
    instruct it."""
    offenders: dict[str, list[str]] = {}
    for (name, stem), tier in classify_all(skills).items():
        if tier not in ra.BOUND_TIERS:
            continue
        bad = _generated_field_offenders(skills[name])
        if bad:
            offenders[name] = bad
    assert offenders == {}, (
        f"skills derive an archive filename from a report's generated: field: "
        f"{offenders}. Use `date -u +%Y-%m-%d-%H%M` (#436 clause 3)."
    )


# --- Clause 4 ------------------------------------------------------------------
def test_retention_is_dated_copies_not_reports_tracked_in_the_repo():
    """Clause 4: retention is achieved by dated copies (or, later, by #436's option
    2 — relocating the canonical reports into the vault), never by committing
    generated reports into `~/lloyd`.

    Scoped to the exclusion the clause actually states — the reports are not
    *tracked* — and deliberately not to the repo-wide ignore rule, which is the
    finding this round answers: an unmarked assertion that `_pipeline/` is still
    ignored made a file this item does not own, and which preflight refuses to
    let any round edit anyway, a precondition of every future round that touches a
    comment in this file. That is a second, unrelated use of the same mark the
    module docstring above argues against. `git ls-files _pipeline` asks only what
    #436 decided, and it fails on exactly the one change that would contradict the
    clause: reports added to the index.

    `.gitignore:25` `/_pipeline/` is still what makes an unarchived overwrite
    unrecoverable — that is stated in the archive rule and in this file's opening,
    where it is the *reason* for the change, and pinned as a loop boundary by
    `tests/test_automod_doc_claims.py::test_path_policy_matches_the_doc[.gitignore-denied]`
    rather than by an assertion about the working tree.

    No skip: this reads the index of the repo the test lives in, and a run that
    cannot ask git is a failure worth seeing, not a green line.
    """
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "_pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"`git ls-files _pipeline` failed: {proc.stderr.strip()}"
    assert proc.stdout.splitlines() == [], (
        f"generated reflection reports are tracked in the repo: "
        f"{proc.stdout.splitlines()[:3]}. #436 keeps retention in dated copies — "
        "putting _pipeline/ under version control is a decision recorded on the "
        "item, not a side effect of a test file."
    )


# --- The seam the one exemption rests on ---------------------------------------
def test_the_exempt_tier_is_a_call_that_cannot_reach_the_lloyd_tree(tmp_path, monkeypatch):
    """The process boundary this rule's single exemption sits on.

    `scripts/reflection_archive.py` refuses to land a skill that overwrites a
    reflection report without a dated copy, with one exempt tier: a write
    prescribed through `vault_write`. The reason is a claim about a *different
    process* — the MCP server resolves that call against `~/obsidian` and rejects a
    `~/lloyd/` target — and until now nothing pinned it (review finding, round
    SM_20260912_163822: the exemption rested on a note in `lloyd/MEMORY.md`, and a
    memory file is not a test). If that resolver ever starts accepting a `~/lloyd/`
    path, the exempt tier stops being a no-op and silently re-opens the loss this
    item is about, so the refusal is asserted against the real handler.

    Three shapes, because the skills write the path three ways and the guard's
    three branches are separate code: a `~/`-prefixed home path, an absolute path
    outside the vault, and a `..` traversal that reaches the same file. All three
    must say `PATH_ESCAPE`, for a read as well as a write — the skills' read lists
    name the same paths.

    Then the positive control, which is what makes the six refusations above mean
    anything: with `VAULT` pointed at a scratch tree, the same handler really does
    write a file that is inside the vault. Without it, a handler broken to fail
    every call would satisfy the exemption for the wrong reason forever.

    `VAULT` is pointed at the scratch tree and `_audit_write` stubbed before *any*
    call, the refusals included. Both are about the red run, not the green one, and
    this file is unmarked so a gate rung runs it. If the guard ever stops refusing, an
    unpatched `VAULT` means the write half-happens into the live vault: a regressed
    `~/lloyd/…` target resolves under it as a literal `~/obsidian/~/lloyd/…` tree,
    which is the stray-root incident `_normalize_vault_path` documents, and this test
    would create one per rung. And the hook's target is `AUDIT_LOG_FILE`, derived from
    the real vault at import rather than from the patched `VAULT`, so an unstubbed hook
    would append the control write to the real `~/obsidian/memory/audit/writes.jsonl`.

    The root identity is asserted unpatched, before the patch: the exemption is a
    claim about where the writer resolves paths, and "outside the vault" only implies
    "cannot reach ~/lloyd" while the writer's root really is ~/obsidian."""
    from agent_mcp import vault as vault_tool

    assert Path(vault_tool.VAULT).expanduser().resolve() == (Path.home() / "obsidian").resolve(), (
        f"the vault writer's root is {vault_tool.VAULT!r}, not the vault: the "
        "`vault-write` exemption claims a `~/lloyd/` target is outside everything "
        "this handler can write to, and that is false the moment its root moves"
    )

    monkeypatch.setattr(vault_tool, "VAULT", tmp_path)
    monkeypatch.setattr(vault_tool, "_audit_write", lambda *a, **k: None)

    targets = (
        "~/lloyd/_pipeline/reflection/signals-latest.md",
        "/home/alansrobotlab/lloyd/_pipeline/reflection/signals-latest.md",
        "../../../home/alansrobotlab/lloyd/_pipeline/reflection/signals-latest.md",
    )
    for target in targets:
        for handler in (vault_tool._vault_write, vault_tool._vault_read):
            result = handler(
                {"path": target, "content": "# would destroy the report\n"}
            )
            # Checked before the code, because the two are not exclusive:
            # `Path(VAULT) / "~/lloyd/…"` is how the 2026-07 incident wrote — `~`
            # becomes a real directory *inside* the vault root — so a guard that
            # both refuses and half-writes is exactly what a code-only check misses.
            # In the loop rather than after the control write, because on a
            # regression the code assertion below would abort the test first.
            assert not (tmp_path / "~").exists(), (
                f"{handler.__name__}({target!r}) created a stray `~` directory inside "
                "the vault root while refusing it, so the exemption's premise — the "
                "write does not happen — is false even where the code says it is"
            )
            assert result.get("code") == "PATH_ESCAPE", (
                f"{handler.__name__}({target!r}) returned {result!r}. The "
                "`vault-write` exemption in scripts/reflection_archive.py is only "
                "honest while a `~/lloyd/` target is refused here"
            )
    written = vault_tool._vault_write(
        {"path": "reflection-archive-probe.md", "content": "inside the vault\n"}
    )
    assert written.get("success") is True, (
        f"the handler no longer writes a path inside the vault, so the six "
        f"refusations above prove nothing: {written!r}"
    )
    assert (tmp_path / "reflection-archive-probe.md").read_text(
        encoding="utf-8"
    ) == "inside the vault\n"
    # And the scratch root holds exactly the control file: the same half-write check
    # the loop makes per call, re-run here over the whole tree so a target that
    # landed somewhere else inside the vault — not just under `~` — is visible too.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["reflection-archive-probe.md"], (
        "a refused target left something behind in the vault root besides the control "
        f"write: {sorted(str(p.relative_to(tmp_path)) for p in tmp_path.iterdir())}"
    )


# --- The enforcement point: the vault writer, not a test rung -------------------
_SKILL_FRONT = "---\nname: test-skill\ndescription: A skill that overwrites a report\n---\n\n"
_STEM = "tool-patterns-latest"
_LIVE = f"/home/alansrobotlab/lloyd/_pipeline/reflection/{_STEM}.md"
# Written the way the skills write it: one instruction wrapped over three lines
# with backslash continuations, so this fixture also exercises logical_lines.
_READ_LINE = f'Read("{_LIVE}")'
_COPY_LINE = (
    'Bash("STAMP=$(date -u +%Y-%m-%d-%H%M) && \\\n'
    f"      cp {_LIVE} \\\n"
    f"         {_LIVE[:-3]}-$STAMP.md\")"
)
_WRITE_LINE = f'- `Write(file_path="{_LIVE}", content=…)` — new content'
_GOOD_BODY = _SKILL_FRONT + "\n".join(
    ["### 2e. Pattern Output Files\n", _READ_LINE, _COPY_LINE, "", _WRITE_LINE, ""]
)
_BAD_BODY = _SKILL_FRONT + "\n".join(
    ["### 2e. Pattern Output Files\n", _READ_LINE, "", _WRITE_LINE, ""]
)


@pytest.fixture
def scratch_vault(tmp_path, monkeypatch):
    """A vault tree the test owns, plus the loader subprocess stubbed out.

    `skills/**` is a *validated* vault path, so `validate` would otherwise shell
    out to `agent_mcp.skills._load_skill` against this scratch tree — which is a
    real check of a different question (does the skill load) and belongs to
    tests/test_automod_vault_round.py. Stubbing it keeps this assertion about the
    retention branch and nothing else, the same way every case in that file
    stubs it.
    """
    from scripts.automod import vault_round

    monkeypatch.setattr(vault_round, "VAULT", tmp_path)
    monkeypatch.setattr(vault_round, "loader_errors", lambda paths: [])
    return tmp_path


def _write_skill(vault: Path, name: str, body: str) -> str:
    rel = f"skills/{name}/SKILL.md"
    target = vault / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return rel


def test_the_vault_writer_refuses_a_skill_that_dropped_its_archive_step(scratch_vault):
    """The wiring, not the helper: `validate()` — the function `land()` and
    therefore `automod_vault_land` actually call — must reach the retention branch.

    Calling `reflection_archive.skill_rule_violations` directly pins the rule; it
    does not pin the call site, and a correct helper that nothing calls is the
    exact shape of "a guard whose input nothing wired up" that this item exists to
    close. So this drives the lander: a skill edit that loses the archive `cp`
    comes back refused, naming the report and the missing step, while the same
    skill with its `cp` intact lands.
    """
    from scripts.automod import vault_round

    bad = _write_skill(scratch_vault, "nightly-reflection-knowledge-write", _BAD_BODY)
    errors, _buckets = vault_round.validate([bad])
    assert len(errors) == 1, f"the lander let an unarchived overwrite through: {errors}"
    assert _STEM in errors[0], f"the refusal must name the report: {errors[0]}"
    assert "dated-copy archive step" in errors[0], (
        f"the refusal must name what is missing: {errors[0]}"
    )
    assert errors[0].startswith(f"{bad}: "), f"the error must name the file: {errors[0]}"

    good = _write_skill(scratch_vault, "nightly-reflection-knowledge-write", _GOOD_BODY)
    assert vault_round.validate([good])[0] == [], (
        "a skill that archives before overwriting must be landable: "
        f"{vault_round.validate([good])[0]}"
    )


def test_the_vault_writer_only_judges_the_skills_a_round_touched(scratch_vault):
    """Scoping, which is what keeps this off an unrelated round's critical path.

    A governed skill sitting unarchived in the vault must not block a round that
    lands a different file — the same reason `contract_errors` reads only the two
    identity files it is named for. Without this, adding the check to `validate`
    would make every skill edit in the tree hostage to the worst skill in it.
    """
    from scripts.automod import vault_round

    _write_skill(scratch_vault, "nightly-reflection-knowledge-write", _BAD_BODY)
    unrelated = _write_skill(
        scratch_vault, "some-other-skill",
        _SKILL_FRONT + "Nothing about reflection reports here.\n",
    )
    assert vault_round.validate([unrelated])[0] == [], (
        "an untouched skill was judged for a condition it did not change"
    )
    # And a touched skill that never mentions a report is judged and clean.
    assert vault_round.reflection_archive_errors([unrelated]) == []


def test_the_writer_and_the_test_share_one_definition():
    """Static pin: the lander must call the same rule this file asserts.

    Mirrors `tests/test_prompt_surface_guard.py::test_both_writers_call_the_shared_invariants`
    for the same reason — a mark and a helper can both exist while the call site
    quietly disappears, and then the rule is documentation again.
    """
    src = (ROOT / "scripts" / "automod" / "vault_round.py").read_text(encoding="utf-8")
    assert "import reflection_archive" in src, (
        "vault_round no longer imports the retention rule: the writer stopped "
        "enforcing it and only the live_vault reporting copy remains"
    )
    assert "reflection_archive.skill_rule_violations" in src, (
        "vault_round no longer calls the retention rule"
    )
    validate_src = src.split("def validate(", 1)[1].split("\ndef ", 1)[0]
    assert "reflection_archive_errors(paths)" in validate_src, (
        "reflection_archive_errors is defined but validate() no longer calls it — "
        "the lander would compute nothing and land everything"
    )


_P = "/home/alansrobotlab/lloyd/_pipeline/reflection/"

#: §2e of `nightly-reflection-knowledge-write`, transcribed from the live vault
#: (heading at line 151; the copy blocks landed in vault commit 56e71846, whose
#: prose has since been reworded around them). Transcribed rather than re-worded
#: because the mutations below are mutations of the shape the run actually reads, and
#: the guard has to be shown to reject that shape, not a paraphrase of it.
#:
#: The transcription is checked against the vault by the live-vault test for this
#: section, not from here: this fixture only has to be shaped like §2e, and
#: `test_knowledge_write_section_2e_archives_both_pattern_files` is what fails when
#: the real section stops looking like this.
_REAL_2E_ARCHIVE_TOOL = (
    f'Read("{_P}tool-patterns-latest.md")\n'
    f'Bash("test -f {_P}tool-patterns-latest.md && \\\n'
    "STAMP=$(date -u +%Y-%m-%d-%H%M) && \\\n"
    f"cp {_P}tool-patterns-latest.md \\\n"
    f'{_P}tool-patterns-latest-$STAMP.md")\n'
)
_REAL_2E_ARCHIVE_CONV = (
    f'Read("{_P}conversation-patterns-latest.md")\n'
    f'Bash("test -f {_P}conversation-patterns-latest.md && \\\n'
    "STAMP=$(date -u +%Y-%m-%d-%H%M) && \\\n"
    f"cp {_P}conversation-patterns-latest.md \\\n"
    f'{_P}conversation-patterns-latest-$STAMP.md")\n'
)
_REAL_2E_WRITE_TOOL = (
    f"- `Write(file_path=\"{_P}tool-patterns-latest.md\", content=…)` — source: "
    "`## Tool Patterns — Failures` and `## Tool Patterns — Successes` sections\n"
)
_REAL_2E_WRITE_CONV = (
    f"- `Write(file_path=\"{_P}conversation-patterns-latest.md\", content=…)` — "
    "source: `## Conversation Patterns` section\n"
)
_REAL_2E_RULES = (
    "\nFour rules that are not stylistic:\n"
    "\n- **`STAMP` is `date -u +%Y-%m-%d-%H%M` evaluated by the shell**, never "
    "typed by hand and never read out of a field inside the previous report.\n"
    "- **`Read` before `Write`.** `Write` refuses to overwrite a file it has not "
    "read in the current session, and both files exist from the previous cycle.\n"
    "- **A missing source is checked, not ignored.** `test -f` gates the `cp`.\n"
)


def _section_2e(archive: str, writes: str) -> str:
    """Assemble §2e the way the live vault writes it: archive fence, then writes."""
    return (
        "### 2e. Pattern Output Files\n"
        "\n"
        "Both pattern files are overwritten **in place**, `_pipeline/` is gitignored.\n"
        "Archive first, one `Read` + one `cp` per file:\n"
        "\n"
        "```\n"
        + archive
        + "\n```\n"
        "\n"
        "Then write the new content from the artifact:\n"
        + writes
        + _REAL_2E_RULES
    )


_REAL_2E = _section_2e(_REAL_2E_ARCHIVE_TOOL + _REAL_2E_ARCHIVE_CONV,
                       _REAL_2E_WRITE_TOOL + _REAL_2E_WRITE_CONV)

#: Two `Write` calls, one `cp` instruction. That is all §2e contains, so dropping
#: either pattern file's archive block from the fence leaves every string a
#: section-wide search looks for still present.
_ARCHIVE_FENCE = _REAL_2E_ARCHIVE_TOOL + _REAL_2E_ARCHIVE_CONV

#: The mutations the review rung performed on the real §2e, plus the one it did not
#: need to: each breaks a distinct obligation of clause 1 while §2e still names both
#: files and still contains the string `Read(`. Key is what broke, value is
#: (mutated section, the file whose clause is now unmet).
_REAL_2E_MUTATIONS = {
    # "For each of the two pattern files" — conversation-patterns loses its copy
    # step; the tool-patterns block above it still satisfies any search that is not
    # per file, and archive_problems asked per stem on the whole section catches it
    # only because the stem is paired with the text.
    "conversation-patterns loses its cp block": (
        _section_2e(_REAL_2E_ARCHIVE_TOOL, _REAL_2E_WRITE_TOOL + _REAL_2E_WRITE_CONV),
        "conversation-patterns-latest",
    ),
    # "the existing file is Read" — the Read is retargeted at the other governed
    # report. `Read(` is still in the section and both paths are still named, so
    # only a Read paired with *this* stem fails.
    "tool-patterns's Read retargeted at the other report": (
        _section_2e(
            _REAL_2E_ARCHIVE_TOOL.replace(
                f'Read("{_P}tool-patterns-latest.md")',
                f'Read("{_P}conversation-patterns-latest.md")',
            )
            + _REAL_2E_ARCHIVE_CONV,
            _REAL_2E_WRITE_TOOL + _REAL_2E_WRITE_CONV,
        ),
        "tool-patterns-latest",
    ),
    # "before the overwrite write" — the copy block for conversation-patterns moves
    # below the Write call that replaces it, so it archives the new content and the
    # previous cycle is still lost. A section-wide min-index comparison over all
    # copies and all writes passes this: tool-patterns' copy is still first.
    "conversation-patterns's copy moved below its Write": (
        "### 2e. Pattern Output Files\n"
        "\n"
        "Both pattern files are overwritten **in place**, `_pipeline/` is gitignored.\n"
        "\n"
        "```\n"
        + _REAL_2E_ARCHIVE_TOOL
        + "\n```\n"
        "\n"
        "Then write the new content from the artifact:\n"
        + _REAL_2E_WRITE_TOOL
        + _REAL_2E_WRITE_CONV
        + "\n"
        "```\n"
        + _REAL_2E_ARCHIVE_CONV
        + "```\n"
        + _REAL_2E_RULES,
        "conversation-patterns-latest",
    ),
}


def section_2e_problems(body: str, stem: str) -> list[str]:
    """Every obligation clause 1 places on §2e, for one named file.

    Shared by the live-vault assertion and its mutations, so the mutation test
    proves the same predicates that gate the clause and not a copy of them.

    `archive_problems` alone is not enough here, and the gap is exactly the one the
    review found. It answers "does the text it is handed archive `<stem>`", and its
    ordering checks compare the *minimum* copy index against the *minimum* write
    index across the whole text — so in a section that covers two files, a block
    belonging to file A discharges file B's obligation for free, and A's healthy
    copy masks B's late one. Pairing the stem with its own Read, and comparing its
    own copy against its own `Write`, is what makes "for each of" mean something.
    """
    problems = archive_problems(body, stem)
    if not re.search(
        r"(?<![A-Za-z_])\bRead\s*\(\s*[\"'`][^\"'`]*" + re.escape(stem) + r"\.md", body
    ):
        problems = problems + [
            f"§2e never Reads {stem}.md itself: Write refuses a file it has not read "
            "in the session, so the instruction has to name this file"
        ]
    copy_at = [
        i for i, ln in enumerate(ra.logical_lines(body))
        if ra.COPY.search(ln) and ra.archive_dest(stem).search(ln)
    ]
    write_at = [
        i for i, ln in enumerate(ra.logical_lines(body))
        if re.search(r"(?<![A-Za-z_])\bWrite\s*\(.*" + re.escape(stem) + r"\.md", ln)
    ]
    if copy_at and write_at and min(copy_at) > min(write_at):
        problems = problems + [
            f"{stem}'s dated copy appears below the Write that overwrites it, so it "
            "archives the new report and the previous cycle is still lost"
        ]
    return problems


#: The two greps the prior round's evidence for clause 1 actually used, lifted out
#: so its finding can be restated as an assertion instead of argued in prose.
_LOOSE_PATH = re.compile(r"_pipeline/reflection/[a-z-]+-latest\.md")
_LOOSE_READ = re.compile(r"(?<![A-Za-z_])\bRead\s*\(")


def test_the_section_2e_guard_fails_on_every_way_it_could_be_satisfied_wrongly():
    """The guard clause 1 rests on, proved by breaking §2e three ways.

    Round SM_20260912_163822 was refused on clause 1 (review_retry) because its
    evidence was two section-wide greps — the file named somewhere in the section,
    and the string `Read(` somewhere in the section — and the grader broke the
    clause while both still passed: it deleted the `cp` for one pattern file and
    retargeted its `Read` at the other. So this test does not assert that a guard
    reports a problem. It takes the §2e text as landed in the vault, applies the
    review's own mutations to it, and asserts three things per mutation: the
    per-file guard names *the file that was broken*, a guard written the way the
    previous round wrote it stays silent, and the unmutated section is clean (so
    none of this is vacuous).
    """
    for stem in ("tool-patterns-latest", "conversation-patterns-latest"):
        assert section_2e_problems(_REAL_2E, stem) == [], (
            f"the §2e as landed in the vault was reported as broken for {stem}: "
            f"{section_2e_problems(_REAL_2E, stem)} — every mutation below would be "
            "vacuous"
        )

    for label, (section, victim) in _REAL_2E_MUTATIONS.items():
        assert section_2e_problems(section, victim), (
            f"§2e mutated ({label}) and the guard for {victim} stayed silent"
        )
        # It must name the file it is refusing, or the next reader cannot act.
        assert victim in " ".join(section_2e_problems(section, victim)), (
            f"§2e mutated ({label}) but the refusal for {victim} did not name it: "
            f"{section_2e_problems(section, victim)}"
        )
        # The sibling file, untouched by this mutation, must still be clean — the
        # guard is per file, so it cannot answer a broken clause by complaining
        # about the wrong one.
        sibling = (
            "conversation-patterns-latest" if victim == "tool-patterns-latest"
            else "tool-patterns-latest"
        )
        assert section_2e_problems(section, sibling) == [], (
            f"§2e mutated ({label}) reported {sibling} as broken, which the mutation "
            f"did not touch: {section_2e_problems(section, sibling)}"
        )
        # The previous round's two greps, run on the broken section: both still
        # true, which is the review's finding restated as a fact about this tree
        # rather than argued. If this assertion ever fails, the loose form has
        # become able to catch the case and the pairing above is no longer the only
        # thing standing between a mutation and a green suite.
        assert _LOOSE_PATH.search(section) and _LOOSE_READ.search(section), (
            f"{label}: the loose greps would now fail too, so this mutation no longer "
            "demonstrates the gap the pairing above closes"
        )


# --- Non-vacuity: the pin must be able to fail ---------------------------------
def _mutate(body: str, old: str, new: str) -> str:
    assert old in body, "mutation target absent — the fixture drifted, fix the fixture"
    return body.replace(old, new)


def test_the_archive_check_can_actually_fail():
    """Every way this guard was asked to fail, on synthetic bodies. Without this
    the suite could be green on a check that matches nothing."""
    assert classify(_GOOD_BODY, _STEM) == "tool-write", (
        "the well-formed skill body must classify as a writer — otherwise every "
        "archive assertion below is vacuous"
    )
    assert archive_problems(_GOOD_BODY, _STEM) == [], "the well-formed case must be clean"
    assert ra.skill_rule_violations("s", _GOOD_BODY) == [], (
        "the writer's entry point reported a problem on a skill that archives"
    )

    bad = _BAD_BODY
    assert ra.skill_rule_violations("s", bad), (
        "the writer's entry point passed a skill with no archive step"
    )
    assert classify(bad, _STEM) == "tool-write", (
        "removing the cp must not have removed the writer — the skill still "
        "overwrites the report, which is the whole problem"
    )

    good = _GOOD_BODY
    cases = {
        "the cp instruction deleted": good.replace(_COPY_LINE, ""),
        "the cp replaced by a no-op": _mutate(good, "      cp ", "      echo "),
        "the stamp taken from the local clock": _mutate(
            good, "STAMP=$(date -u +%Y-%m-%d-%H%M)", "STAMP=$(date +%Y-%m-%d-%H%M)"
        ),
        "the stamp a `<cycle>` placeholder": _mutate(
            good, f"{_LIVE[:-3]}-$STAMP.md", f"{_LIVE[:-3]}-<cycle>.md"
        ),
        "the stamp a hand-typed date": _mutate(
            good, f"{_LIVE[:-3]}-$STAMP.md", f"{_LIVE[:-3]}-2026-09-10-2230.md"
        ),
        "the copy pointed at a different file": _mutate(
            good,
            f"{_LIVE[:-3]}-$STAMP.md",
            "/home/alansrobotlab/lloyd/_pipeline/reflection/other-latest-$STAMP.md",
        ),
        # "before it is overwritten" is part of the clause, not a suggestion: the
        # same two instructions in the other order copy the new report and lose the
        # old one, so a check blind to order would pass a body that saves nothing.
        "the archive step written below the overwrite": "\n".join(
            ["### 2e. Pattern Output Files\n", _READ_LINE, "", _WRITE_LINE, _COPY_LINE, ""]
        ),
        "the read written below the overwrite": "\n".join(
            ["### 2e. Pattern Output Files\n", _COPY_LINE, "", _WRITE_LINE, _READ_LINE, ""]
        ),
    }
    for label, body in cases.items():
        problems = archive_problems(body, _STEM)
        assert problems, f"the check passed a body whose archive step is broken ({label})"
        assert ra.skill_rule_violations("s", body), (
            f"the writer would have landed a body with a broken archive step ({label})"
        )

    # A body that reads the file but never writes it is not a writer, so it must
    # not be demanded an archive step.
    assert classify(_READ_LINE + "\n", _STEM) == "read"
    # A write prescribed through vault_write is a broken instruction, not a writer.
    vault = f'Write to `vault_write(path="~/lloyd/_pipeline/reflection/{_STEM}.md")`:\n'
    assert classify(vault, _STEM) == "vault-write"

    # A line that reads *and* destroys the file is a write. If `read` won here, a
    # skill could say "Read the previous report and merge everything into <path>",
    # classify as a reader, and escape the archive rule, the tier-2 set equality and
    # the unclassified alarm all at once — so this case is the leak, pinned.
    read_merge = (
        f'Read the previous report and merge everything into '
        f'`~/lloyd/_pipeline/reflection/{_STEM}.md`\n'
    )
    assert classify(read_merge, _STEM) == "prose-write", (
        "a read-and-merge instruction was classified as a harmless read"
    )
    # Its inverse: a genuine read annotated with a write that belongs to another
    # file further down the line stays a reader. Without this pair the rule would
    # simply be "any line mentioning a write verb is a writer", which is wrong.
    annotated_read = (
        f'1. Signal report: `Read(file_path="/home/alansrobotlab/lloyd/_pipeline/'
        f'reflection/{_STEM}.md")` — absolute path, not `vault_read`. Same rule for '
        f"the handoff write below and for reading it back.\n"
    )
    assert classify(annotated_read, _STEM) == "read", (
        "a reader was recruited as a writer by a noun belonging to another file"
    )


def test_the_bound_tiers_are_the_two_that_can_lose_a_report():
    """The quantifier of clause 2, pinned by value.

    `BOUND_TIERS` is what decides whether a skill's overwrite is checked at all, so
    a edit that quietly narrowed it to `tool-write` would leave every prose merge —
    including the one `historical-knowledge-refresh` used to perform on all three
    reports — looking covered. The exemption is asserted too: `vault-write` is out
    because `PATH_ESCAPE` makes the write impossible, not because it is prose.
    """
    assert ra.BOUND_TIERS == ("tool-write", "prose-write"), (
        f"BOUND_TIERS is {ra.BOUND_TIERS}; the clause covers every skill that "
        "writes a report, which is both the call and the instruction"
    )
    assert "vault-write" not in ra.BOUND_TIERS, (
        "vault-write writes are refused by PATH_ESCAPE, so refusing to land a "
        "skill over them would block on an instruction that cannot execute"
    )
    # And the tier the clause is really about stays bound end to end: a prose
    # merge with no archive step is a problem, is refused by the writer, and is
    # not excused by its tier.
    prose_only = (
        "1. **Consolidate signals:** Merge everything into "
        "`~/lloyd/_pipeline/reflection/signals-latest.md`\n"
    )
    assert classify(prose_only, "signals-latest") == "prose-write"
    assert archive_problems(prose_only, "signals-latest"), (
        "a prose merge with no archive step passed, which is the exact shape "
        "historical-knowledge-refresh had before round SM_20260912_163822"
    )
    assert ra.skill_rule_violations("s", prose_only), (
        "the vault writer would land a skill that destroys a report in prose"
    )
    # "Before it is overwritten" binds the prose tier with the same force: an
    # archive step written *below* the merge instruction copies the merged report
    # and loses the old one, so a sequence-blind rule passes a body that saves
    # nothing. This is the shape `historical-knowledge-refresh` now avoids by
    # putting its archive block first in Phase 3.
    prose_after = (
        "1. **Consolidate signals:** Merge everything into "
        "`~/lloyd/_pipeline/reflection/signals-latest.md`\n"
        "   STAMP=$(date -u +%Y-%m-%d-%H%M) && "
        "cp signals-latest.md signals-latest-$STAMP.md\n"
    )
    assert archive_problems(prose_after, "signals-latest"), (
        "a prose merge followed by its archive step passed the order check"
    )
    assert not archive_problems(
        prose_after.splitlines()[1] + "\n" + prose_after.splitlines()[0] + "\n",
        "signals-latest",
    ), "an archive step written before the prose merge was still flagged"


_BARE_REPORT = r"(?<![A-Za-z0-9_./-])"  # a filename not preceded by `reflection/`


@pytest.mark.live_vault
def test_a_bound_writer_names_every_report_it_owes_by_full_path(skills):
    """Attribution is textual, so the shape of the mention is part of the pin.

    `LATEST_PATH` needs the `_pipeline/reflection/` prefix to see a report at all —
    that is what tells a report in that directory apart from a file that merely
    shares its name, and the price is that a *write instruction* naming only a bare
    filename is invisible to the classifier: no tier, no archive rule, no writer-set
    entry. So every bound writer's mentions must be path-shaped. A copy step is
    allowed to name its source relatively (the skills `cd` into the directory for
    the archive block), and a sentence *about* the file is allowed too — what must
    not happen is an instruction with a write verb in front of a bare name, because
    that is a writer this check would never find.
    """
    offenders: dict[str, list[str]] = {}
    for (name, stem), tier in classify_all(skills).items():
        if tier not in ra.BOUND_TIERS:
            continue
        bare = re.compile(_BARE_REPORT + re.escape(stem) + r"\.md")
        for ln in ra.logical_lines(skills[name]):
            at = ln.find(f"{stem}.md")
            if at < 0 or not bare.search(ln):
                continue
            if ra.COPY.search(ln) and ra.archive_dest(stem).search(ln):
                continue  # an archive copy may name its source relatively
            head = ln[:at]
            if ra.WRITE_TO.search(head) or ra.PROSE_WRITE.search(head) or ra.TOOL_WRITE.search(head):
                offenders.setdefault(f"{name}:{stem}", []).append(ln.strip()[:100])
    assert offenders == {}, (
        f"bound writers name a report by bare filename in a write instruction, "
        f"where this check cannot see them: {offenders}. Spell the path as "
        "`_pipeline/reflection/<name>-latest.md`."
    )


def test_the_generated_field_check_can_actually_fail():
    """The pre-#436 wording, verbatim from vault commit 3514dac1, must be caught."""
    old_wording = (
        "> **Archive before the skeleton write, every run.** `<cycle>` is the "
        "previous report's own `generated:` stamp, so "
        "`signals-latest-2026-09-10-2230.md`.\n"
    )
    assert _generated_field_offenders(old_wording), (
        "the check accepted an instruction to build the name from generated:"
    )

    allowed = (
        "> `STAMP` is never read out of the previous report's `generated:` field. "
        "The `generated:` stamp is not usable here: the copy named "
        "`signals-latest-2026-09-10-2230.md` sorts 8 hours out.\n"
    )
    flagged = _generated_field_offenders(allowed)
    assert flagged == [], f"a permitted, negated mention was flagged: {flagged}"


def test_the_rule_survives_a_wrapped_shell_instruction():
    """The seam between a wrapped shell instruction and a line-matching rule.

    The archive step is one `Bash("cp …` written over three markdown lines with
    backslash continuations. If `logical_lines` stopped splicing them, the `cp`
    and the dated destination would be on different lines and every real archive
    step in the vault would read as missing — so the splice is asserted, in both
    directions, on a body that is wrapped exactly as §2e's is.
    """
    wrapped = ra.logical_lines(_COPY_LINE)
    assert len(wrapped) == 1, f"the continuation was not spliced: {wrapped}"
    assert ra.COPY.search(wrapped[0]) and ra.archive_dest(_STEM).search(wrapped[0]), (
        "the spliced line no longer matches as a copy to a dated destination"
    )
    assert ra.utc_stamp(wrapped[0]), "the spliced line no longer carries the UTC stamp"
    # The negative control: the same instruction un-spliced matches nothing, which
    # is what the splice is for.
    assert not any(
        ra.COPY.search(ln) and ra.archive_dest(_STEM).search(ln)
        for ln in _COPY_LINE.splitlines()
    ), "an un-spliced body matched, so the splice is not what makes the check work"


# --- Live-vault reporting copy: drift guards on the classifier itself -----------
@pytest.mark.live_vault
def test_classification_survives_an_unclassifiable_mention(skills):
    """Every mention must be attributed. A path that is neither written nor read
    nor prose-written lands in `unclassified`, which is how an under-inclusive
    rule would leak a writer — so it is a failure, not a shrug."""
    unclassified = {
        f"{name}:{stem}" for (name, stem), tier in classify_all(skills).items()
        if tier == "unclassified"
    }
    assert unclassified == set(), (
        f"reflection-report mentions the classifier could not attribute: "
        f"{unclassified}. Classify them (and if one writes the file, it owes an "
        "archive step) rather than letting the writer scan go narrow."
    )


@pytest.mark.live_vault
def test_the_only_exemption_is_a_call_that_cannot_reach_the_path(skills):
    """One exemption is allowed, and it is the one that cannot destroy a report.

    Anything routed through `vault_write`/`mem_write` returns `PATH_ESCAPE` on a
    `~/lloyd/` target, so no archive step can be owed by it — but that reason has
    an expiry date: the moment such a skill gains a real `Write`/`Edit` call it is
    a writer and owes the step. So the exempt set is asserted equal to the
    recorded one (it cannot grow unlisted) and each entry is re-checked to still
    classify as `vault-write` (an entry whose reason lapsed fails rather than
    grandfathering in)."""
    classified = classify_all(skills)
    exempt: dict[str, set[str]] = {}
    for (name, stem), tier in classified.items():
        if tier == "vault-write":
            exempt.setdefault(name, set()).add(stem)
    assert exempt == EXPECTED_VAULT_WRITE_WRITERS, (
        f"PATH_ESCAPE-shaped prescriptions changed: found {exempt}, expected "
        f"{EXPECTED_VAULT_WRITE_WRITERS}. A bound writer must archive; the only "
        "acceptable exemption is a call that cannot reach the path."
    )
    for name, stems in EXPECTED_VAULT_WRITE_WRITERS.items():
        assert name in skills, f"{name} is no longer an active skill — prune its entry"
        for stem in stems:
            assert classified.get((name, stem)) == "vault-write", (
                f"{name}:{stem} changed classification to "
                f"{classified.get((name, stem))}; if it now writes with Write/Edit, "
                "or merges in prose, it must archive before overwriting"
            )


@pytest.mark.live_vault
def test_the_recorded_debt_is_measured_by_archive_presence(skills):
    """The last skill that overwrote a report without naming a copy is now gone
    from the debt, and the debt itself is a measurement, not a list.

    Round SM_20260912_155333 recorded `historical-knowledge-refresh` as the
    outstanding case: it merges into all three live reports in prose, and prose is
    not a call a run executes. Round SM_20260912_163822 bound the prose tier and
    gave that skill its own Read + `date -u` `cp` step, so the expectation is empty
    — and it is checked as `archive_problems` output, so the only way to add an
    entry is to break an archive step, while paying one removes it. Two half-states
    are pinned against that: a bound writer with no archive step appears in the
    measurement (mutation below), and a skill cannot sit in the debt by virtue of
    its tier alone.
    """
    classified = classify_all(skills)
    debt: dict[str, set[str]] = {}
    for (name, stem), tier in classified.items():
        if tier in ra.BOUND_TIERS:
            problems = archive_problems(skills[name], stem)
            if problems:
                debt.setdefault(name, set()).add(stem)
    assert debt == EXPECTED_UNARCHIVED_DEBT, (
        f"skills overwrite a nightly report with no named archive step: {debt}. "
        "`_pipeline/` is gitignored, so those reports are unrecoverable (#436)."
    )
    # The payment, pinned: same skill, prose tier, no problems, absent from debt.
    paid = "historical-knowledge-refresh"
    for stem in ("signals-latest", "tool-patterns-latest", "conversation-patterns-latest"):
        assert classified.get((paid, stem)) == "prose-write", (
            f"{paid} no longer merges {stem} in prose, so the pinned payment above "
            "is checking a condition that no longer exists — update the expectation"
        )
        assert archive_problems(skills[paid], stem) == [], (
            f"{paid} lost the archive step round SM_20260912_163822 gave it for "
            f"{stem}: {archive_problems(skills[paid], stem)}"
        )
    assert paid not in debt, f"{paid} is back in the debt the round cleared"
    # Anything still owed has to be named on the item, where the next reader of
    # #436 will see it. Asserted, not guarded: a renamed or pruned item file stops
    # this test dead rather than leave its last assertion silently unrun (review
    # finding, round SM_20260912_155333).
    if debt:
        items = sorted((Path.home() / "obsidian" / "backlog").glob("436-*.md"))
        assert items, (
            "backlog/436-*.md is gone, so unarchived writers have nowhere to be "
            "recorded: re-record them on the item that owns #436's remaining scope"
        )
        body = items[0].read_text(encoding="utf-8", errors="replace")
        for skill in debt:
            assert skill in body, (
                f"{skill} is unarchived debt on #436 but is not named in the item — "
                "invisible to whoever closes it"
            )
    else:
        assert any((Path.home() / "obsidian" / "backlog").glob("436-*.md")), (
            "the backlog item this file's scope is written against is gone"
        )


# --- #1189 clause 4: the archive cp in the real `nightly-reflection-signals` text --
# #436's guard, merged here from the aborted-round branches rather than rewritten.
# What the branches actually held, re-measured 2026-09-17 rather than taken from
# #1112's body: `git cat-file -t <sha>:tests/test_skill_reflection_archive.py` gives
# a blob at `6620216` (828 lines), `9c09e9e` (738), `78a8037` (699) and `bb26ed4`
# (585); `main`'s version is longer than all four (1174 lines), and the one node
# name present there and absent here —
# `test_pipeline_stays_gitignored_so_retention_is_copies_not_tracking` — was
# *superseded*, not lost: `main` replaced its `git check-ignore` probe with the
# stronger `git ls-files _pipeline` pin above. What no branch ever held is the
# mutation clause 4 names — deleting the `signals-latest` archive `cp` from the
# REAL skill text — so that is what lands here, built on this file's existing
# transcription pattern (§2e's fixture states its own rule: transcribe the real
# shape, prove it against the vault from a marked test).
#
# One correction to carry forward: the archive step is `nightly-reflection-signals`'
# **Phase 0** — the code block under "## Phase 0: Claim the output file FIRST" — not
# Phase 5 as #1080/#1189 describe it. Phase 5 is where the report body is written;
# the claim→archive→overwrite triple that #436 protects is Phase 0, which is also
# where clause 2's protected `date -u` `cp` lives. Same instruction, same guard;
# the phase number in the item text is simply wrong.

# One more correction, from #1285 (2026-09-21): the skeleton no longer lives in this
# triple at all. Phase 0 used to claim, archive, then overwrite the canonical pointer
# with a `status: in-progress` stub, so a run that died mid-investigation left the stub
# sitting at the path every downstream job reads as current — which is what happened on
# 2026-09-20. Phase 0 now claims and archives and writes the stub to
# `signals-inflight.md`, and the pointer is written exactly once per run, by Phase 3,
# with the complete report. So the shape the retention mutation has to be applied to is
# no longer one section: it is Phase 0's `cp` plus the Phase 3 publish the `cp` exists to
# protect, and a fixture holding only the in-flight write would still pass with the `cp`
# deleted because nothing in it overwrites the pointer any more.

_SIGNALS_LIVE = "/home/alansrobotlab/lloyd/_pipeline/reflection/signals-latest.md"
_SIGNALS_INFLIGHT = "/home/alansrobotlab/lloyd/_pipeline/reflection/signals-inflight.md"
_REAL_SIGNALS_READ = f'Read("{_SIGNALS_LIVE}")   # MUST come first'
_REAL_SIGNALS_INFLIGHT_READ = (
    f'Read("{_SIGNALS_INFLIGHT}")   # exists from last night too; '
    "Write refuses an unread file"
)
_REAL_SIGNALS_ARCHIVE = (
    'Bash("STAMP=$(date -u +%Y-%m-%d-%H%M) && \\\n'
    f"     cp {_SIGNALS_LIVE} \\\n"
    f"        {_SIGNALS_LIVE[:-3]}-$STAMP.md\")   # archive, new path"
)
_REAL_SIGNALS_SKELETON = (
    f'Write(file_path="{_SIGNALS_INFLIGHT}",\n'
    '      content="# Signal Report <today>\\n\\nstatus: in-progress\\n\\n(Investigating.)\\n")'
)
#: The single write to the canonical pointer (Phase 3, complete report). The archive
#: step is there to protect *this* overwrite, so the transcription the retention
#: mutation is applied to must carry it — otherwise the mutation bites on a fixture that
#: has no canonical overwrite left to archive, which is a pass for the wrong reason.
_REAL_SIGNALS_PUBLISH = (
    'Write the structured signal report to '
    f'`Write(file_path="{_SIGNALS_LIVE}")`'
)


def _real_signals_archive_shape(with_archive: bool = True) -> str:
    """Transcription of the shipped instructions the archive rule binds for
    `signals-latest`: Phase 0's claim, archive `cp` and in-flight skeleton, then the
    Phase 3 publish that overwrites the pointer — in that order.

    Copied byte-for-byte from `~/obsidian/skills/nightly-reflection-signals/SKILL.md`
    including the backslash continuations, because the wrapping is what
    `logical_lines` exists for — a fixture written on one line would prove the
    guard against a shape the shipped skill does not have.

    Both sections, not Phase 0 alone. Since #1285 the skeleton no longer names the
    canonical pointer, so a Phase 0-only fixture would report `no writes to
    signals-latest` with the `cp` deleted and nothing at all with it present: the
    mutation would bite on a report nobody was about to overwrite, which is a pass for
    the wrong reason.
    """
    phase0 = [_REAL_SIGNALS_READ, _REAL_SIGNALS_INFLIGHT_READ]
    if with_archive:
        phase0.append(_REAL_SIGNALS_ARCHIVE)
    phase0.append(_REAL_SIGNALS_SKELETON)
    return ("## Phase 0: Claim the output file FIRST\n\n```\n" + "\n".join(phase0) + "\n```\n"
            "## Phase 3: Finalise the Signal Report\n\n" + _REAL_SIGNALS_PUBLISH + "\n")


def test_the_rule_fails_when_the_signals_archive_copy_is_deleted():
    """#1189 clause 4: with the archive `cp` deleted from the signals skill's own
    shipped shape, the rule must refuse it and name the missing step; with it
    present, `signals-latest` must be clean.

    Unmarked, so every gate rung runs it, and it reads no vault — the transcription
    is pinned to the shipped text by
    ``test_the_live_signals_archive_step_is_the_one_this_mutation_deletes``, which is
    how a fixture stays honest about what it claims to mutate.

    Deletion is the failure #436 was filed for: instructions that keep the `Read` and
    the publish `Write` and lose the `cp` between them overwrite the live report with no
    copy of the previous cycle, and `_pipeline/` is gitignored (`.gitignore:25`), so
    nothing else can recover it. The expected message is asserted exactly, because
    a looser assertion would pass on a rule that fired for the wrong reason — and after
    #1285 moved the skeleton off the pointer, exactness is also what stops the node
    passing on a fixture with no canonical write in it at all.
    """
    body = _real_signals_archive_shape()
    assert ra.archive_problems(body, "signals-latest") == [], (
        "the transcription is not clean as written, so the deletion below proves "
        f"nothing about the real step: {ra.archive_problems(body, 'signals-latest')}"
    )
    assert ra.utc_stamp(_REAL_SIGNALS_ARCHIVE), (
        "the transcribed step no longer carries the `date -u` stamp, so the copy it "
        "survives would be named from a local clock (#436 clause 2)"
    )
    problems = ra.archive_problems(_real_signals_archive_shape(with_archive=False),
                                   "signals-latest")
    assert problems == [
        "no dated-copy archive step for signals-latest: no cp to "
        "`signals-latest-<stamp>.md`"
    ], (
        f"deleting the archive cp changed the verdict: {problems}. The rule must "
        "refuse exactly this shape — claim, overwrite, no copy — and no other"
    )


@pytest.mark.live_vault
def test_the_live_signals_archive_step_is_the_one_this_mutation_deletes():
    """The transcription drift guard, on the real skill text.

    Three things at once, in the direction that matters: every transcribed
    instruction — Phase 0's claim, in-flight skeleton read, archive `cp` and
    skeleton write, and the Phase 3 publish — appears verbatim in
    `~/obsidian/skills/nightly-reflection-signals/SKILL.md`; the
    real body is clean under the rule; and deleting *its own* archive `cp` — located
    by the rule's predicates (`COPY` + `archive_dest` over `logical_lines`), never by
    a remembered line number — yields the same verdict as the transcription. So
    clause 4's mutation is a mutation of the text that ships, and a rewrite of that
    prose fails here naming the transcription to update instead of quietly narrowing
    the guard to a fixture.

    Marked ``live_vault`` for the reason this file's opening states: skill prose is
    state no round under test controls, and the writer
    (``scripts/automod/vault_round.py::reflection_archive_errors``) is where dropping
    the `cp` is refused before the commit. The unmarked deletion test above is what
    runs on the hard rung; this is the reporting copy that keeps it aimed at the real
    text, which is what makes it more than the prose-plus-hand-grep arrangement that
    let the class recur five times (#1189).
    """
    path = next(
        (d / "nightly-reflection-signals" / "SKILL.md" for d in SKILLS_DIRS
         if (d / "nightly-reflection-signals" / "SKILL.md").exists()),
        None,
    )
    assert path is not None, f"nightly-reflection-signals/SKILL.md absent under {SKILLS_DIRS}"
    body = path.read_text(encoding="utf-8", errors="replace")
    for instruction in (_REAL_SIGNALS_READ, _REAL_SIGNALS_INFLIGHT_READ,
                        _REAL_SIGNALS_ARCHIVE, _REAL_SIGNALS_SKELETON,
                        _REAL_SIGNALS_PUBLISH):
        assert instruction in body, (
            f"the real Phase 0 no longer contains {instruction!r} — the transcription "
            "this file mutates has drifted from the shipped text; update "
            "_REAL_SIGNALS_* with the new shape"
        )
    assert ra.archive_problems(body, "signals-latest") == [], (
        f"the live signals skill lost its archive step: "
        f"{ra.archive_problems(body, 'signals-latest')}"
    )
    lines = ra.logical_lines(body)
    cp_idx = [
        i for i, ln in enumerate(lines)
        if ra.COPY.search(ln) and ra.archive_dest("signals-latest").search(ln)
    ]
    assert len(cp_idx) == 1, (
        f"expected exactly one archive cp for signals-latest in the real Phase 0, "
        f"found {len(cp_idx)} — the mutation target is ambiguous, so this guard needs "
        "rewriting with whatever replaced it"
    )
    mutated = "\n".join(ln for i, ln in enumerate(lines) if i != cp_idx[0])
    assert ra.archive_problems(mutated, "signals-latest") == [
        "no dated-copy archive step for signals-latest: no cp to "
        "`signals-latest-<stamp>.md`"
    ], "deleting the real archive cp did not produce the guard's verdict"



# --- #1285: the canonical pointer only ever carries a complete report ------------
# Phase 0 used to claim the canonical slot by overwriting it with a `status:
# in-progress` skeleton, so a run that died mid-investigation left a stub at the path
# every downstream job reads as this cycle's report. The damage is on the record:
# `autonomy-runs/38/run_38_20260920_050102.md` is `status: failed`,
# `failure_kind: infra`, and `_pipeline/reflection/knowledge-handoff-2026-09-20.md`
# says "this morning's #42 consumed it" about the 46-line stub it left behind. The
# skeleton now goes to `signals-inflight.md`; the canonical slot is written once per
# run, by Phase 3, with `status: complete`.
_SIGNALS_LATEST_NAME = "signals-latest.md"
_SIGNALS_INFLIGHT_NAME = "signals-inflight.md"
#: The call form both publish instructions use for the canonical slot — the prose
#: before the template ("Write the structured signal report to …") and the line after
#: it ("**Write this report to:** …"). The skills prescribe the same call twice, so a
#: rule about what writes the pointer has to count call instructions, not sections.
_PUBLISH_CALL = f'Write(file_path="{_SIGNALS_LIVE}")'
#: The consumers that act on the pointer's contents: **#40** (config) and **#42**
#: (knowledge-analysis). `nightly-reflection-knowledge-write` (**#39**) is deliberately
#: absent — it takes the day's handoff as its only source of truth and never opens this
#: path (#1285's triage re-derived that from its Data Load), so guarding it would pin
#: prose no reader needs.
SIGNALS_POINTER_READERS = ("nightly-reflection-config",
                           "nightly-reflection-knowledge-analysis")


def _pointer_write_instructions(body: str) -> list[str]:
    """Every instruction in `body` that writes the canonical signal pointer.

    An *instruction* here is the call plus its continuation lines up to a blank line or
    a closed fence, not one physical line. That is the whole design decision: the
    shape #1285 removed broke after the `file_path` argument, so the `content` carrying
    `status: in-progress` sat on the line below the path. `logical_lines` splices
    backslash continuations only — it exists for the wrapped `cp` — so scanning it a
    line at a time reports the forbidden shape as clean, which is a pin that cannot
    fail. A run reading the skill reads both lines as one instruction, so this does too.

    The archive `cp` is excluded because it names the path to copy it, never to write
    it, and is discharged by `archive_problems` instead.
    """
    lines = body.split("\n")
    out: list[str] = []
    for i, ln in enumerate(lines):
        if (_SIGNALS_LATEST_NAME not in ln or not ra.TOOL_WRITE.search(ln)
                or ra.COPY.search(ln)):
            continue
        parts = [ln]
        j = i + 1
        while j < len(lines) and lines[j].strip() and lines[j].strip() != "```":
            parts.append(lines[j])
            j += 1
        out.append("\n".join(parts))
    return out


def test_the_pointer_write_scan_catches_the_shape_1285_removed():
    """The unmarked control for the two assertions below: the scan must catch the
    instruction that shipped until #1285, and must not catch the one that replaced it.

    Transcribed from `5f746695:skills/nightly-reflection-signals/SKILL.md` (the last
    vault commit that carried it): the `Write` names the canonical path on one line and
    carries `status: in-progress` on the next, which is the split a per-line scan
    reports as clean. Reading no vault, so every gate rung runs it — the failure this
    file's own docstring calls a pin that cannot fail.
    """
    forbidden = (
        '## Phase 0: Claim the output file FIRST\n\n```\n'
        + _REAL_SIGNALS_READ + "\n" + _REAL_SIGNALS_ARCHIVE + "\n"
        + f'Write(file_path="{_SIGNALS_LIVE}",\n'
          '      content="# Signal Report <today>\\n\\nstatus: in-progress\\n\\n(Investigating.)\\n")'
        + "\n```\n"
    )
    caught = _pointer_write_instructions(forbidden)
    assert len(caught) == 1, (
        f"the scan found {len(caught)} write instructions in the pre-#1285 Phase 0, "
        "which is the shape this rule exists to refuse — a scan that misses it cannot "
        "guard the replacement either"
    )
    assert "in-progress" in caught[0], (
        f"the write instruction found in the pre-#1285 Phase 0 does not carry its "
        f"skeleton content, so the split across lines is what hid it: {caught[0]!r}"
    )
    shipped = _pointer_write_instructions(_real_signals_archive_shape())
    assert shipped and not any("in-progress" in ins for ins in shipped), (
        f"the transcription of the shipped shape is not what #1285 replaced: {shipped}"
    )


@pytest.mark.live_vault
def test_no_skill_writes_an_in_progress_report_to_the_canonical_pointer(skills):
    """#1285 clause 1: nowhere in the advertised skills does an instruction write
    `status: in-progress` content to `signals-latest.md`.

    Asserted across the whole corpus rather than one skill: the defect is a *slot*
    holding a stub, and any second writer could leave one as effectively as the
    producer did. The producer now writes that skeleton only to
    `signals-inflight.md`, and the sentence of **#40**'s guard which says both
    `status: complete` and `in-progress` about the canonical path stays out of this set
    because it carries no write call — which is why the unit is an instruction with a
    `Write`/`Edit` target, not a mention.
    """
    offenders: dict[str, list[str]] = {}
    for name, body in skills.items():
        bad = [ins for ins in _pointer_write_instructions(body) if "in-progress" in ins]
        if bad:
            offenders[name] = bad
    assert offenders == {}, (
        "skills instruct an in-progress write to the canonical signal report slot, "
        "which a run that dies before Phase 3 leaves as the file every downstream job "
        f"reads as this cycle's report (#1285): {offenders}"
    )


@pytest.mark.live_vault
def test_the_signals_skill_publishes_the_pointer_only_with_a_complete_report(skills):
    """#1285 clause 1's other half: the writes the producer keeps for the canonical
    slot are the Phase 3 publish calls, and the template they fill carries
    `status: complete`.

    Checking the call form rather than counting sections is the point — a later edit
    that moved the publish elsewhere, or quietly re-added a claim-time overwrite in
    some other shape, changes either the set of instructions or the call form, and this
    names which. The `status: complete` line is what clauses 5's consumer guards read,
    so a publish that dropped it would leave the guard checking for a token no report
    contains.
    """
    body = skills["nightly-reflection-signals"]
    instructions = _pointer_write_instructions(body)
    assert instructions, "the signals skill no longer writes signals-latest.md at all"
    wrong = [ins for ins in instructions if _PUBLISH_CALL not in ins]
    assert wrong == [], (
        f"an instruction writes the canonical signal slot in a form other than the "
        f"Phase 3 publish call {_PUBLISH_CALL!r} (#1285): {wrong}"
    )
    assert "status: complete" in body, (
        "the published report template no longer carries `status: complete`, so the "
        "consumers' guard (#1285 clause 5) has nothing to check"
    )


def test_the_in_flight_skeleton_is_not_an_archive_bound_report():
    """#1285: moving the skeleton to a new path must not give it an archive obligation.

    The archive rule binds a *canonical* slot — the `-latest` in the name is what marks
    a file as the one thing readers believe, and `LATEST_PATH` is what discovers the
    set. `signals-inflight.md` is deliberately named otherwise: it is the working file
    of a run that may have died, overwriting it destroys nothing anyone reads, so a
    `cp` for it would be an archive nobody consults. Renaming it into the `-latest`
    shape would silently hand it an unmet obligation and make the nightly writer dirty
    on a path nobody overwrote unreadably.

    Unmarked: it reads the shipped constant, no path.
    """
    assert not ra.LATEST_PATH.search(_SIGNALS_INFLIGHT), (
        f"{_SIGNALS_INFLIGHT} now matches LATEST_PATH, so it classifies as a bound "
        "report and the producer owes a dated copy before every skeleton write"
    )


def _signals_reader_item(body: str) -> str:
    """The block of a consumer skill that covers reading the canonical signal report.

    A guard belongs to the read it qualifies. The two consumers attach it differently —
    **#40** as an indented continuation of its numbered item, **#42** as a following
    paragraph — so this takes the item and everything up to the next heading or sibling
    list item. Asserting inside that block is what stops a `status: complete` sentence
    parked in some unrelated section of a 300-line skill from discharging the clause.
    """
    lines = body.split("\n")
    start = next(
        (i for i, ln in enumerate(lines)
         if f"_pipeline/reflection/{_SIGNALS_LATEST_NAME}" in ln),
        None,
    )
    assert start is not None, "this skill no longer reads the canonical signal report"
    out = [lines[start]]
    for ln in lines[start + 1:]:
        if ln.lstrip().startswith("#") or re.match(r"^\s*(?:\d+\.|[-*+])\s", ln):
            break
        out.append(ln)
    return "\n".join(out).strip()


@pytest.mark.live_vault
def test_signal_pointer_readers_check_the_report_is_this_cycle_complete(skills):
    """#1285 clause 5: every consumer of the pointer carries the freshness guard.

    Each reader's data-load item must state the two checks **#40** already had —
    `status:` is `complete`, and `generated:` names this cycle — and the fallback they
    trigger: read the newest dated `signals-latest-<stamp>.md` instead. The producer fix
    (clause 1) makes a stub impossible from *this* producer; the guard is what makes it
    harmless from any other, and it also covers the failure a producer change cannot: a
    run that dies after archiving but before publishing leaves a *complete* report from
    the previous cycle under this name, which is stale rather than incomplete.
    """
    offenders: dict[str, list[str]] = {}
    for name in SIGNALS_POINTER_READERS:
        item = _signals_reader_item(skills[name])
        missing = [token for token in ("status:", "complete", "generated:",
                                       "newest", "signals-latest-")
                   if token not in item]
        if missing:
            offenders[name] = missing
    assert offenders == {}, (
        "consumers read the canonical signal report without checking it is this cycle's "
        "complete report, so a stub or a stale report is consumed as current (#1285): "
        f"{offenders}"
    )
