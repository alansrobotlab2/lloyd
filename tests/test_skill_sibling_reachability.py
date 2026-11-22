"""Every `.md` beside a `SKILL.md` is reachable from it, and one skill proves why that
matters (#4307).

A skill is a folder: `agent_mcp/skills.py` loads `SKILL.md` alone, so a sibling exists in a
prompt only if the body points at it. #624 established the spill ("your skill is really a
folder") and `tests/test_skill_spill_624.py` polices the five sampled skills; this module is
the same invariant across the whole live library, which is what a run that mints a new
sibling per cycle needs — `kg-mention-classifier` writes `run-log-<date>.md` every cycle, and
the day its hand-maintained list stops covering the newest file, that run-log is invisible to
the next run and to every reader, with nothing red anywhere.

What counts as reachable, in the order checked:

* the sibling's **stem** (its name without `.md`) appears in the body — the basename itself
  qualifies, and so does the other spelling in use today: `skills/queue-health-check/SKILL.md`
  lists `step1-fleet-line, step2-run-records, … (each `.md`)`, seven stems on one line with
  the suffix stated once;
* a **stated range** covers it: two backticked `.md` names sharing a stem and separated by an
  ellipsis or a dash (`run-log-2026-10-27.md` … `run-log-2026-11-09.md`), the sibling's stem
  sitting between the two endpoints at the first position where they differ. This is how a
  dated run log is made reachable without editing the body every cycle — the literal-naming
  rule alone would either fail the fleet or force a `SKILL.md` commit per run.

Anchored and not a substring soup: both endpoints must be backticked, the same length, and
share the sibling's stem, so a range cannot cover a name it was not written over.

The corpus leg reads the live vault and stays unmarked, the ruling
`tests/test_yaml_fix_skill_claims.py`'s module docstring records: the gate runs
`-m "not live_vault and not fault_injection"`, so a `live_vault` marker here would deselect
the one node that tells a fold round its sibling is unreachable. The synthetic leg carries
the failure half, so the invariant can be seen to fail without editing a live skill —
including the two directions a naive matcher gets wrong: an unnamed sibling, and a dated log
that falls OUTSIDE the range its body states.

Three of these nodes are also #4307's own contract pins, one per clause they name in their
docstring: `test_the_fleet_leg_reports_the_sibling_a_run_would_lose` (clause 4 — the fleet
guard's own walk, run over a tree where the invariant is false, so its green on the live
library is known to be a result and not an absence), `test_the_fold_left_every_spilled_section_heading_in_skill_md`
(clause 2 — the fold's headings graded against `SKILL.md`, never against the spliced union,
which tail-appends a stranded section whole and so could not report being stranded), and
`test_the_twelve_thousand_ceiling_and_the_uncapped_splice_are_both_intact` (clause 5 — the
12,000 ceiling in both its modules and the uncapped #624 dispatch route, none of which this
fold was authorized to change).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Never app.paths.VAULT_ROOT: that re-anchors to a round's worktree, and the skills tree
# lives in the account's vault, which no round has.
VAULT = Path.home() / "obsidian"
SKILLS_DIR = VAULT / "skills"
PIPELINE_SLUG = "autonomy-data-pipeline"

#: `_skill_desc_and_body`'s target for the folded `skills/autonomy-data-pipeline/SKILL.md`,
#: and the number a ~1,500-char probe edit has to fit under:
#: `scripts/automod/vault_round.SKILL_BODY_CHAR_CAP` is 12_000 and its row fires on
#: `new_body > CAP and new_body > old_body`, so a body at 11,000 would still be refused the
#: moment someone added 1,500 chars to it. 10,500 is what leaves the probe room.
PIPELINE_BODY_TARGET = 10_500
PROBE_CHARS = 1_500

#: Two denominators, so an empty or narrowed walk can never read as a clean library.
#: Measured 2026-11-21 over `skills/*/` (dot-dirs excluded, which is `.archived`'s 142
#: archived `SKILL.md` files out of scope: an archived skill is never dispatched into a
#: prompt, so nothing a run of it writes needs to be reachable): 75 sibling `.md` files
#: across 20 skills.
MIN_SIBLINGS = 60
MIN_SKILLS_WITH_SIBLINGS = 15

#: The stems the fold moved out of `autonomy-data-pipeline/SKILL.md` on 2026-11-22 UTC. Each has
#: to stay reachable by name, which is what makes the fold's index the thing under test
#: rather than the sibling count.
PIPELINE_SPILL_STEMS = (
    "step0-kg-health-snapshot",
    "step1-fact-extraction",
    "step2-index-rebuild",
    "step2-5-denominators",
    "step3-kg-health-readback",
    "step4-run-log",
    "step4-guardrails",
    "step5-vault-tasks",
    "post-flight",
)

#: Siblings the rule does not apply to, and the reason, in full.
#:
#: `README.md` and `DESCRIPTION.md` are the companion documents that skill importer leaves
#: beside an imported skill: `skills/ascii-video/` and `skills/ocr-and-documents/` arrived from
#: `NousResearch/hermes-agent` with theirs already in the directory. No run of either skill
#: writes them, and `agent_mcp/skills.py` loads `SKILL.md` alone, so there is no section in
#: them for a run or a reader to reach. The class is what the rule is about, so the exemption
#: covers both names rather than the one the ceiling refuses.
#:
#: Naming them is refused by the ceiling this round must not touch. `skills/ascii-video/SKILL.md`
#: measures 12,266 body chars against `vault_round.SKILL_BODY_CHAR_CAP = 12_000`, and
#: `skill_contract_findings` refuses a landing that grows an over-cap body — measured directly:
#: `automod_vault_land` on 2026-11-22 answered `skill body size: the body is 12,266 chars, past
#: 12,000, and this landing grows it` and reverted the whole commit. The only way to name
#: `README.md` is to fold that skill, which #4307 does not authorize.
#:
#: The exemption is pinned to these two basenames, counted on the live library by
#: `test_the_provenance_exemption_stays_two_vendored_files`, and can never cover a dated run
#: log or a fold sibling — the node asserts that.
UPSTREAM_PROVENANCE = ("README.md", "DESCRIPTION.md")

_STEM_SUFFIX = ".md"
#: An ellipsis or a dash between two backticked names. `-{1,2}` so a hyphenated stem inside
#: one pair of backticks (`run-log-2026-11-11.md`) is never read as a range separator.
_RANGE = re.compile(r"`([^`\n]*?\.md)`\s*(?:…|\.\.\.|–|—|–|-{1,2}|\bto\b)\s*`([^`\n]*?\.md)`")

#: The shape a cycle's own output has, used to prove the provenance exemption below can never
#: cover a file a run writes.
_DATED = re.compile(r"run-log-\d{4}-\d{2}-\d{2}")
_H2 = re.compile(r"^## ", re.MULTILINE)


def _covered_by_range(main: str, stem: str) -> str | None:
    """The range line whose endpoints bracket `stem`, or None.

    Endpoints must be the same length and share the stem's prefix up to the first position
    where they differ, so `run-log-2026-10-27.md … run-log-2026-11-09.md` brackets
    `run-log-2026-11-03` and brackets nothing else: a comparison is only taken over the
    suffix after the endpoints' common prefix, which is the only part of a range a reader
    understood as a range.
    """
    for line in main.splitlines():
        for low, high in _RANGE.findall(line):
            low_stem, high_stem = low[: -len(_STEM_SUFFIX)], high[: -len(_STEM_SUFFIX)]
            if len(low_stem) != len(high_stem) or len(stem) != len(low_stem):
                continue
            shared = 0
            while (shared < len(low_stem)
                   and low_stem[shared] == high_stem[shared]):
                shared += 1
            if low_stem[:shared] != stem[:shared]:
                continue
            if low_stem[shared:] <= stem[shared:] <= high_stem[shared:]:
                return line.strip()
    return None


def unreachable_reason(main: str, sibling: str) -> str | None:
    """Why `sibling` is not reachable from `main`, or None when it is.

    Returned as a sentence rather than a bool because the corpus assertion's job is to tell a
    round which line to write, not just that one is missing.
    """
    if sibling in UPSTREAM_PROVENANCE:
        return None                 # vendored import artifact — see the comment on the tuple
    stem = sibling[: -len(_STEM_SUFFIX)] if sibling.endswith(_STEM_SUFFIX) else sibling
    if _names_stem(main, stem):
        return None
    if _covered_by_range(main, stem):
        return None
    return (f"no line of SKILL.md names `{sibling}` (or its stem `{stem}`) and no stated "
            f"range covers it")


def _names_stem(main: str, stem: str) -> bool:
    """Does some line name this stem AS A NAME, rather than containing its letters?

    The arm was `stem in main` — the substring soup this module's docstring warns about: a
    short sibling (`run.md`, `notes.md`) reads as reachable from any prose holding those
    letters, `run` inside "the run completes" and `notes` inside "release notes below", and
    the guard's job is to report an unreachable sibling. A name is a token, so the stem must
    be delimited by something that is neither a word character nor a hyphen:
    `run-log-2026-11-16.md`, a backticked `step2-5-denominators`, "watermark-write-history:"
    all name it; "running" does not. The range arm is unchanged, which keeps the advisory's
    finding true as written: it was the range arm alone that the "not a substring soup" claim
    was made of, and now the stem arm is too.
    """
    return re.search(rf"(?<![\w-]){re.escape(stem)}(?![\w-])", main) is not None


def folders_with_siblings(skills: Path) -> list[tuple[Path, list[str]]]:
    """`(folder, sibling names)` for every live skill that has siblings.

    Top-level directories only, and dot-directories are skipped: `skills/.archived/` holds
    142 archived `SKILL.md` files whose siblings nothing dispatches.
    """
    out: list[tuple[Path, list[str]]] = []
    for folder in sorted(p for p in skills.iterdir()
                         if p.is_dir() and not p.name.startswith(".")):
        if not (folder / "SKILL.md").is_file():
            continue
        siblings = sorted(p.name for p in folder.glob("*.md")
                          if p.name != "SKILL.md")
        if siblings:
            out.append((folder, siblings))
    return out


# ── the synthetic leg: the rule has to be able to fail ─────────────────────────


def _folder(root: Path, slug: str, body: str, *siblings: str) -> Path:
    folder = root / slug
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(body, encoding="utf-8")
    for name in siblings:
        (folder / name).write_text(f"# {name}\n\ndetail that a run reads.\n",
                                   encoding="utf-8")
    return folder


def test_a_sibling_the_body_never_points_at_is_reported(tmp_path):
    """The rule can fail. A folder whose body says nothing about its one sibling is exactly
    the state #4307 was filed about, and a matcher that only ever returns None would let it
    through green."""
    folder = _folder(tmp_path, "quiet", "## Step 1\n\nDo the thing.\n", "detail.md")
    main = (folder / "SKILL.md").read_text(encoding="utf-8")
    reason = unreachable_reason(main, "detail.md")
    assert reason is not None and "detail.md" in reason, reason
    assert folders_with_siblings(tmp_path) == [(folder, ["detail.md"])]


def test_a_dated_log_outside_the_stated_range_is_reported(tmp_path):
    """The range arm is a bracket, not a licence.

    The body names `run-log-2026-10-27.md` … `run-log-2026-11-09.md`; a `run-log-2026-11-20.md`
    sits past the high endpoint, and that is the state a per-cycle writer falls into the day
    after its last body edit. A matcher that called any line containing two `.md` names a
    range would pass it.
    """
    body = ("- `run-log-2026-10-27.md` … `run-log-2026-11-09.md` — one file per cycle\n")
    folder = _folder(tmp_path, "late", body, "run-log-2026-11-20.md")
    assert unreachable_reason(body, "run-log-2026-11-20.md") is not None
    assert _covered_by_range(body, "run-log-2026-11-20") is None
    assert folders_with_siblings(tmp_path) == [(folder, ["run-log-2026-11-20.md"])]


def test_a_dated_log_inside_the_stated_range_is_reachable(tmp_path):
    """The other half of the same line: the file the range was written over is reachable
    without naming it, which is the whole reason a per-cycle skill can hold this invariant
    at all."""
    body = ("- `run-log-2026-10-27.md` … `run-log-2026-11-09.md` — one file per cycle\n")
    _folder(tmp_path, "on-time", body, "run-log-2026-11-03.md")
    assert unreachable_reason(body, "run-log-2026-11-03.md") is None


def test_a_stem_listed_with_the_suffix_stated_once_is_reachable(tmp_path):
    """`skills/queue-health-check/SKILL.md` states seven stems and `(each `.md`)` once, and
    matches today; the basename is not the only spelling of a name."""
    body = ("detail: step1-fleet-line, step2-run-records (each `.md`),\n"
            "plus fleet_health.py.\n")
    _folder(tmp_path, "stems", body, "step1-fleet-line.md")
    assert unreachable_reason(body, "step1-fleet-line.md") is None


def test_a_hyphenated_name_is_not_read_as_a_range(tmp_path):
    """`run-log-2026-11-11.md` alone on a line must not become a range whose endpoints are
    `run` and `2026-11-11.md`: `-{1,2}` is surrounded by the backtick requirement, so a
    hyphen inside one name is part of the name."""
    body = "- `run-log-2026-11-11.md` — the newest cycle\n"
    _folder(tmp_path, "single", body, "run-log-2099-01-01.md")
    assert _covered_by_range(body, "run-log-2099-01-01") is None
    assert unreachable_reason(body, "run-log-2099-01-01.md") is not None


# ── the corpus: the live library, and the fold that this round shipped ─────────


def _folder_texts(folder: Path) -> tuple[str, str]:
    """`(SKILL.md, the section-spliced union)` — the reader every spilled section depends on.

    `tests/skill_folder.py` is what #4192 built for exactly this: the body keeps a heading
    and a pointer, the text moves to a sibling, and a module that grades a *section* reads
    the union so its scoping keeps its teeth.
    """
    from tests.skill_folder import disk_skill_folder

    return disk_skill_folder(folder)


def _no_tree(path: Path) -> str:
    """The fail-loud message, after `NO_VAULT` in `test_yaml_fix_skill_claims.py:96`.

    Every live-corpus node here used to `pytest.skip` when its tree was absent. A skip inside a
    library-wide invariant is the rule stopping while the run still reads green, and the gate
    reports a skip beside passes — which is why `tests/test_skill_step25_label_pin.py`'s
    `test_nothing_in_this_file_is_allowed_to_skip` exists and why the two other modules that
    read this same vault (`test_skill_step25_label_pin.py`, `test_skill_stranded_ruled_sets.py`)
    name the path and fail instead. This module now reads as loudly as they do, and
    `test_nothing_in_this_file_is_allowed_to_skip` at the bottom holds it there.
    """
    return (f"no skills tree at {path}: this file polices that tree, so it reports rather "
            "than skips")


def corpus_offenders(skills: Path) -> list[tuple[str, str, str]]:
    """`(slug, sibling, why)` for every sibling on `skills` unreachable from its own `SKILL.md`.

    One walk, two callers: the live-library node below asserts this is empty, and
    `test_the_fleet_leg_reports_the_sibling_a_run_would_lose` asserts it is NOT empty on a
    synthetic tree. The two nodes read the same function, so the failure half cannot be a
    different rule from the green half.
    """
    return [(folder.name, name, why)
            for folder, names in folders_with_siblings(skills)
            for name in names
            if (why := unreachable_reason(
                (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace"), name))]


def test_the_fleet_leg_reports_the_sibling_a_run_would_lose(tmp_path):
    """Clause 4's failure half at the level of the node that polices the fleet.

    Clause 4 is not "the helper can return a reason" — the synthetic nodes above pin that — it
    is that the fleet guard itself goes red on a sibling `SKILL.md` neither names nor brackets,
    and that a dated `run-log-<date>.md` stays green through a stated range. The only honest way
    to pin that is to hand the same function the fleet guard asserts over a tree where the
    invariant is false: two folders, one whose body brackets `run-log-2026-10-27.md` …
    `run-log-2026-11-09.md` and so reaches its `run-log-2026-11-03.md`, one whose body brackets
    the same range but has just written `run-log-2026-12-19.md` outside it. Exactly one offender
    comes back, and it names the slug, the file and the reason; on the live tree the same call
    returns nothing. `test_the_whole_live_library_points_at_every_sibling` asserts
    `not corpus_offenders(SKILLS_DIR)`, which is the assertion this node shows can fail.
    """
    late = _folder(tmp_path, "late-cycle",
                   "- `run-log-2026-10-27.md` … `run-log-2026-11-09.md` — one file per cycle\n",
                   "run-log-2026-12-19.md")
    _folder(tmp_path, "on-schedule",
            "- `run-log-2026-10-27.md` … `run-log-2026-11-09.md` — one file per cycle\n",
            "run-log-2026-11-03.md")
    offenders = corpus_offenders(tmp_path)
    # Pinned as a 2-tuple with the reason checked separately: embedding `offenders[0][2]`
    # in the expected value compares that element to itself, so it could not fail.
    assert [(slug, name) for slug, name, _reason in offenders] == [
        ("late-cycle", "run-log-2026-12-19.md")], offenders
    assert "run-log-2026-12-19.md" in offenders[0][2], offenders[0][2]
    assert (late / "SKILL.md").is_file()

    assert SKILLS_DIR.is_dir(), _no_tree(SKILLS_DIR)
    assert corpus_offenders(SKILLS_DIR) == [], "the live tree the fleet node guards is red"


def test_the_whole_live_library_points_at_every_sibling():
    """Clause 4's fleet half: the invariant, fleet-wide, on the live vault.

    84 sibling `.md` files across 20 live skills as measured 2026-11-22, and every one of
    them reachable — including `kg-mention-classifier`'s 25, of which the body names 7
    literally and brackets the 18 dated run logs with two stated ranges. The count is a
    head, not a floor: a skill that writes a dated run log grows one sibling a day, which is
    why the assertion below is `>= MIN_SIBLINGS` (60) and not an equality. The denominators
    above are asserted first: a walk that quietly narrowed to nothing would otherwise be
    the greenest run of this node's life.
    """
    assert SKILLS_DIR.is_dir(), _no_tree(SKILLS_DIR)
    rows = folders_with_siblings(SKILLS_DIR)
    siblings = sum(len(names) for _folder, names in rows)
    assert siblings >= MIN_SIBLINGS, (
        f"only {siblings} siblings across {len(rows)} skills: the walk narrowed, and an "
        f"empty denominator makes this node vacuously true")
    assert len(rows) >= MIN_SKILLS_WITH_SIBLINGS, (
        f"only {len(rows)} skills carry siblings")

    offenders = corpus_offenders(SKILLS_DIR)
    assert not offenders, "unreachable sibling(s): " + "; ".join(
        f"{slug}/{name}: {why}" for slug, name, why in offenders)


def test_the_provenance_exemption_stays_two_vendored_files():
    """The exemption above is two basenames wide and cannot widen itself.

    Pinned three ways, because an exemption to an invariant is where an invariant goes to
    die: the set is exactly those two names; nothing in it could ever be a run's output or a
    fold's spill target; and on the live library exactly two siblings lean on it, in the two
    skills that were imported rather than written here. If a third skill acquires an unnamed
    `README.md`, this node names it — the exemption keeps a vendored file reachable-by-name
    from the importer's convention, it does not license a skill to stop pointing at its own
    material.
    """
    from scripts.automod.vault_round import SKILL_BODY_CHAR_CAP, _skill_desc_and_body

    assert UPSTREAM_PROVENANCE == ("README.md", "DESCRIPTION.md"), UPSTREAM_PROVENANCE
    for name in UPSTREAM_PROVENANCE:
        assert _DATED.fullmatch(Path(name).stem) is None, (
            f"{name} matches the dated-run-log shape, so a run could write it and the "
            "exemption would hide it")
        assert name not in PIPELINE_SPILL_STEMS, (
            f"{name} is a fold spill target, so a section of a live skill would be exempt "
            "from ever being pointed at")

    assert SKILLS_DIR.is_dir(), _no_tree(SKILLS_DIR)
    lean = [(folder.name, name)
            for folder, names in folders_with_siblings(SKILLS_DIR)
            for name in names
            if name in UPSTREAM_PROVENANCE]
    assert lean == [("ascii-video", "README.md"), ("ocr-and-documents", "DESCRIPTION.md")], (
        f"the exemption is load-bearing for {lean}, not the two vendored imports it documents")
    assert _skill_desc_and_body(
        (SKILLS_DIR / "ascii-video" / "SKILL.md").read_text(encoding="utf-8"))[1] \
        > SKILL_BODY_CHAR_CAP, (
        "ascii-video's body is no longer over the cap, so naming README.md there is possible "
        "again — fold it or name it, and retire the exemption")


def test_the_data_pipeline_body_is_folded_under_the_probe_headroom(tmp_path):
    """Clause 1, pinned: the spliced body fits the cap with a probe's worth to spare.

    `_skill_desc_and_body` is the same measurement `scripts/automod/vault_round.py`'s
    `skill_contract_findings` row is computed from, and that row fires on
    `new_body > SKILL_BODY_CHAR_CAP and new_body > old_body`. So the number that matters is
    not 12,000 but 12,000 minus whatever a normal edit costs: a ~1,500-char clause added to
    a 10,500-char body lands at 12,000 and is refused, and a skill that cannot take an edit
    is one every future round has to fold first.
    """
    from scripts.automod.vault_round import (SKILL_BODY_CHAR_CAP, _skill_desc_and_body,
                                             skill_contract_findings)

    skill = SKILLS_DIR / PIPELINE_SLUG / "SKILL.md"
    assert skill.is_file(), f"{skill} is the surface this item folded"
    body_chars = _skill_desc_and_body(skill.read_text(encoding="utf-8"))[1]
    assert body_chars <= PIPELINE_BODY_TARGET, (
        f"{PIPELINE_SLUG}'s body is {body_chars} chars; the fold's whole point is that a "
        f"probe edit of {PROBE_CHARS} still stays under the {SKILL_BODY_CHAR_CAP} refuse")
    assert body_chars + PROBE_CHARS <= SKILL_BODY_CHAR_CAP

    # The refusal itself, replayed. `skill_contract_findings(paths)` reads the new text off
    # disk and the old text off the vault's `HEAD:`, so the replay is a throwaway git vault
    # with `vr.VAULT` pointed at it — the same call the promotion gate makes, with nothing
    # stubbed. The baseline body is not needed: the row's condition is `new_body > cap AND
    # new_body > old_body`, so every arm below holds for whatever HEAD holds, and pinning a
    # 124,732-char copy of the pre-fold file to a path on disk would only add a file that
    # /tmp, which is tmpfs here, is free to take away.
    folded = skill.read_text(encoding="utf-8")
    rel = f"skills/{PIPELINE_SLUG}/SKILL.md"
    probe = folded + "\n## Probe\n\n" + "x" * (PROBE_CHARS - 20) + "\n"
    grown = folded + "\n## Grown\n\n" + "x" * 40_000 + "\n"

    shrink = _rows_in_scratch_vault(skill_contract_findings, [rel], tmp_path,
                                    "shrink",
                                    head=grown, worktree=folded)
    assert not [r for r in shrink if r["gate"] == "skill body size"], (
        f"a landing that SHRINKS a body from over the cap to {body_chars} chars was refused "
        f"on body size — the fold itself could not land: {shrink}")

    head_folded = _rows_in_scratch_vault(skill_contract_findings, [rel], tmp_path,
                                         "head-folded",
                                         head=folded, worktree=probe)
    assert not [r for r in head_folded if r["gate"] == "skill body size"], (
        f"a {PROBE_CHARS}-char probe edit is refused on body size at {body_chars} chars, so "
        f"the fold bought no room: {head_folded}")

    head_grown = _rows_in_scratch_vault(skill_contract_findings, [rel], tmp_path,
                                        "head-grown",
                                        head=folded, worktree=grown)
    assert [r for r in head_grown if r["gate"] == "skill body size"], (
        "skill_contract_findings refused nothing even for a body grown 40,000 chars past the "
        f"fold, so the two clean results above say nothing: {head_grown}")


def _rows_in_scratch_vault(findings, paths: list[str], root, name: str, *,
                           head: str, worktree: str) -> list[dict]:
    """`findings(paths)` against a throwaway vault whose `HEAD:` holds `head` and whose
    working tree holds `worktree`, for both files under the one path.

    `scripts.automod.vault_round` takes the new text from `VAULT / rel` and the old text from
    `git -C VAULT show HEAD:rel`, and `_git` reads the same module global, so repointing
    `VAULT` redirects both. Nothing else about the call is faked.
    """
    import subprocess

    import scripts.automod.vault_round as vr

    # One directory per arm: the arms differ only in which text is HEAD, and two arms that
    # share a directory would leave the second one with nothing to commit.
    vault = Path(root) / f"vault-{name}"
    target = vault / "skills" / PIPELINE_SLUG / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    run = subprocess.run

    def _commit(text: str) -> None:
        target.write_text(text, encoding="utf-8")
        run(["git", "-C", str(vault), "add", "-A"], check=True)
        run(["git", "-C", str(vault), "-c", "user.email=t@lloyd.local",
             "-c", "user.name=lloyd-test", "commit", "-qm", "head"], check=True)

    subprocess.run(["git", "init", "-q", str(vault)], check=True)
    _commit(head)
    target.write_text(worktree, encoding="utf-8")
    old_vault = vr.VAULT
    vr.VAULT = vault
    try:
        return findings(paths)
    finally:
        vr.VAULT = old_vault


def _body_chars(text: str) -> int:
    """The body length `scripts/automod/vault_round.py` measures against its ceiling."""
    from scripts.automod.vault_round import _skill_desc_and_body

    return _skill_desc_and_body(text)[1]


def test_a_probe_edit_on_the_folded_body_stays_under_the_gate_ceiling(tmp_path, monkeypatch):
    """Clause 1's headroom, granted by the rung that withholds it and not by subtraction.

    The item's proof is a ~1,500-char probe edit that must emit no `skill body size` row, and
    the reason it is stated that way is the rung's second term: the row in
    `skill_contract_findings` fires on `new_body > SKILL_BODY_CHAR_CAP and new_body > old_body`,
    so an 11,000-char body passes a plain "is it under 12,000" check today and is still refused
    by a 1,500-char edit tomorrow. Hence 10,500. This node therefore does not subtract 1,500
    from 12,000 in its own arithmetic: it copies the folded `SKILL.md` into a scratch vault,
    commits it, patches `vault_round.VAULT` — the module global the rung reads for both the
    file and its `git show HEAD:` — applies the probe to the working tree, and calls
    `skill_contract_findings` for real.

    Two arms, because a probe arm alone cannot tell absorbed headroom from a rung that never
    ran. The probe arm: the ~1,500-char edit produces no body-size row. The control arm: the
    same body grown past 12,000 produces exactly one, with `would_refuse` set. The control is
    what makes the probe arm's silence a measurement; without it this node is the inert-check
    complaint `scripts/skill_lint.py` records against its own time rule.

    Measured on the shipped bytes 2026-11-22, which is what the two arms are built from and
    what the assertion window on the delta is for: the folded body is 10,322 chars, the probe
    edit adds 1,493 and lands at 11,815 — 185 under the ceiling — and the control body reaches
    12,026, past it.
    """
    import subprocess

    from scripts.automod import vault_round as vr

    folded = (SKILLS_DIR / PIPELINE_SLUG / "SKILL.md").read_text(
        encoding="utf-8", errors="replace")
    body = _body_chars(folded)
    assert body <= PIPELINE_BODY_TARGET, "clause 1's own number, re-asserted before the probe"

    repo = tmp_path / "vault"
    (repo / "skills" / PIPELINE_SLUG).mkdir(parents=True)
    target = repo / "skills" / PIPELINE_SLUG / "SKILL.md"
    target.write_text(folded, encoding="utf-8")
    for cmd in (["init", "-q", "-b", "main", "."], ["add", "-A"],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "folded"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True,
                       capture_output=True, text=True)
    monkeypatch.setattr(vr, "VAULT", repo)
    rel = f"skills/{PIPELINE_SLUG}/SKILL.md"

    target.write_text(folded + "\n\n## Probe\n\n" + "p" * 1480, encoding="utf-8")
    delta = _body_chars(target.read_text(encoding="utf-8")) - body
    assert 1400 <= delta <= 1600, (
        f"the probe edit added {delta} chars, not the ~1,500 clause 1's proof specifies")
    rows = vr.skill_contract_findings([rel])
    assert isinstance(rows, list), rows
    assert not [r for r in rows if r.get("gate") == "skill body size"], (
        f"a ~1,500-char probe edit on the folded body is refused by the ceiling: {rows}")

    target.write_text(
        folded + "\n\n## Overflow\n\n" + "x" * (vr.SKILL_BODY_CHAR_CAP - body + 10),
        encoding="utf-8")
    grown = [r for r in vr.skill_contract_findings([rel])
             if r.get("gate") == "skill body size"]
    assert len(grown) == 1 and grown[0]["would_refuse"], (
        f"the rung stayed silent on a body grown past its ceiling ({rows}); the probe arm "
        "above then proves nothing about headroom")


def test_the_twelve_thousand_ceiling_and_the_uncapped_splice_are_both_intact():
    """Clause 5, pinned: this fold changed neither the ceiling nor the #624 route.

    Three things, none of them this round's to change, all of them load-bearing for the number
    clause 1 pins:

    * `scripts/skill_lint.SKILL_BODY_CHAR_CAP` and
      `scripts/automod/vault_round.SKILL_BODY_CHAR_CAP` are both 12_000. That equality is
      already shipped as
      `tests/test_skill_lint_size.py::test_the_char_cap_is_the_gate_s_own_twelve_thousand`
      (#4191 clause 1) and still passes; it is restated here because clause 5's evidence has to
      sit in a file this diff changes, and because `PIPELINE_BODY_TARGET` is 10,500 only
      because the refuse is at 12,000. Loaded by path for the reason that module gives:
      `vault_round` imports `skill_lint` lazily inside its body-lines check, so a
      module-level import back is a cycle.
    * the #624 route in `app/autonomy.py` still hands a run the WHOLE `SKILL.md`: the comment
      block that says so, `prompt = _build_task_prompt(task, skill_content)`, and both
      `embedded_chars=len(skill_content)` and `source_chars=len(skill_content)` on the
      `record_skill_embed` call that measures it.
    * that module never mentions the ceiling at all, which is the whole contradiction: the
      gate refuses a landing that grows a body past 12,000 while the dispatch route splices
      the full text in uncapped. #4307 leaves the ruling to owed-check, so a round that
      resolved it by editing either side turns this node red.

    The fold this pin is against moved the bulk of a 124,732-char body into named siblings and
    left both the ceiling and the uncapped splice byte-for-byte where they were.
    """
    import importlib.util

    from scripts.automod import vault_round as vr

    spec = importlib.util.spec_from_file_location(
        "sibling_reachability_skill_lint", ROOT / "scripts" / "skill_lint.py")
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    assert lint.SKILL_BODY_CHAR_CAP == vr.SKILL_BODY_CHAR_CAP == 12_000, (
        f"the ceiling moved (skill_lint {lint.SKILL_BODY_CHAR_CAP}, vault_round "
        f"{vr.SKILL_BODY_CHAR_CAP}); #4186 ruled 12,000 must NOT be raised, and "
        "PIPELINE_BODY_TARGET's probe headroom is computed against it")

    src = (ROOT / "app" / "autonomy.py").read_text(encoding="utf-8", errors="replace")
    assert "# #624:" in src, (
        "the #624 comment block is gone from app/autonomy.py: the uncapped autonomy splice "
        "is either deleted or re-worded, and this fold did not authorize that")
    at = src.index("# #624:")
    splice = src[at:at + 800]
    assert "prompt = _build_task_prompt(task, skill_content)" in src[:at], (
        "the autonomy route no longer builds its prompt from the whole skill content")
    assert "embedded_chars=len(skill_content)" in splice, (
        "record_skill_embed no longer reports the full spliced length, so the 94,075 "
        "per-dispatch baseline the item measures against is no longer comparable")
    assert "source_chars=len(skill_content)" in splice, (
        "the embedded/source pair came apart, which is what a cap on this route would look like")
    assert "SKILL_BODY_CHAR_CAP" not in src, (
        "app/autonomy.py now reads the gate's ceiling: the cap-vs-splice contradiction owed-"
        "check rules on has been resolved on the splice side by a round that was not asked to")


#: The headings the pre-fold `autonomy-data-pipeline/SKILL.md` carried, matched on their
#: stable prefix and not their full title: `## Step 2.5:` rather than `## Step 2.5:
#: Stranded-entity linking — DRY RUN ONLY, record the denominators`, because a run is free to
#: re-title a section and a node that reddens on that is a node a later round deletes.
PIPELINE_MAP_HEADINGS = (
    "## Step 0:", "## Step 1:", "## Step 2:", "## Step 2.5:", "## Step 3:",
    "## Step 4:", "## Post-Flight: Git Snapshot", "## Post-Flight: Watermark Update",
    "## Error Handling", "## Consolidation Awareness", "## Guardrails",
    "### Run log", "### Guardrails (read before", "### Task 5.3:", "### Task 5.4:",
    "### GATE — if `status=locked`", "### GATE — if `status=noop`",
    "### If `status=ran`", "### NEVER pass `--apply`", "### Run it",
    "### Record the denominators",
)


def test_the_fold_left_every_spilled_section_heading_in_skill_md():
    """Clause 2's heading half, graded against `SKILL.md` and not against the folder union.

    The first cut of this node matched the heading list against the union, which cannot fail
    for the reason it claims to guard: `skill_folder.splice_union` appends a sibling section
    whole — heading line included — when the map file does not carry that heading, so a heading
    that had LEFT `SKILL.md` still appeared in the union and the node stayed green on exactly
    the state it was written to catch. Graded against `SKILL.md`, the node is red on that
    state, which is also the state that breaks every section-scoped reader of this folder:
    `sections()` keys the splice off the map file, so a stranded sibling section reaches the
    union's tail and no span contains it — the asymmetry `splice_union`'s own docstring calls
    "clause 3's asymmetry, not an oversight".

    Three assertions, in the order a later fold breaks them. Measured here on the shipped
    bytes 2026-11-22: all 21 prefixes below start a line of `SKILL.md`, which carries 23
    headings; the nine spill siblings' 17 section titles are a subset of those 23; and every
    one of their bodies sits inside its own heading span of the union, which is the span
    `tests/test_data_pipeline_watermark.py` grades. `watermark-write-history.md` keeps 3
    headings the map lacks and is deliberately outside the second assertion: it is a whole-file
    reference a pointer sends a run to, not one of the 17 spilled sections, and it is not in
    `PIPELINE_SPILL_STEMS`.
    """
    from tests.skill_folder import sections

    folder = SKILLS_DIR / PIPELINE_SLUG
    assert folder.is_dir(), _no_tree(folder)
    main = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")

    missing = [head for head in PIPELINE_MAP_HEADINGS
               if not any(ln.startswith(head) for ln in main.splitlines())]
    assert not missing, (
        f"`SKILL.md` no longer starts a line with {missing}: a fold that moves a heading out "
        "of the map file strands its sibling section in the union's tail")

    map_titles = {title for title, _line, _body, _end in sections(main)}
    _main, union = _folder_texts(folder)
    spans = {title: (body, end)
             for title, _line, body, end in sections(union)}
    for stem in PIPELINE_SPILL_STEMS:
        text = (folder / f"{stem}.md").read_text(encoding="utf-8", errors="replace")
        sections_of = list(sections(text))
        orphans = [title for title, _l, _b, _e in sections_of if title not in map_titles]
        assert not orphans, (
            f"{stem}.md carries heading(s) {orphans} that `SKILL.md` does not, so "
            "`splice_union` tail-appends that section and no section span reaches it")
        for title, _line, body, end in sections_of:
            chunk = text[body:end].strip()
            if not chunk:
                continue
            span = spans.get(title)
            assert span and chunk in union[span[0]:span[1]], (
                f"`{title}` from {stem}.md is not inside its own heading span of the union, "
                "so a reader that section-scopes the union grades the one-line pointer where "
                "the procedure used to be")


def test_the_folded_body_still_names_every_sibling_it_spilled_to():
    """The fold's own index is what the run reads, so it is what has to be complete.

    Each of the nine spill siblings has to be named by stem on a line that also says when to
    read it — `tests/test_skill_spill_624.py`'s `\b[Rr]ead\b.*\b(when|before|after|only|at)\b`
    condition, restated here because that module samples five other skills and this one is the
    largest dispatcher in the fleet.
    """
    folder = SKILLS_DIR / PIPELINE_SLUG
    assert folder.is_dir(), _no_tree(folder)
    main = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    # Scoped to the index section, because clause 2 says "named under '## Files in this
    # skill'" and a scan of the whole body would accept a stem mentioned anywhere — in a
    # worked example, in a Step's prose — and still call the index complete. The index is
    # the thing a run reads to find the sibling, so the index is what is graded: a stem
    # mentioned only in a Step body but missing from the index is exactly the state a
    # later round would trip over, and this node must be the one that says so.
    from tests.skill_folder import sections

    # `sections` yields (title, heading-line start, body start, body end) — CHARACTER
    # offsets into `main`, not line numbers, and the body ends at the NEXT heading of
    # either level, which is the same cut `splice_union`'s readers make.
    idx = {title.strip(): (line_start, body_end)
           for title, line_start, _body_start, body_end in sections(main)}
    span = idx.get("Files in this skill")
    assert span is not None, "`SKILL.md` lost its '## Files in this skill' section"
    index_lines = main[span[0]:span[1]].splitlines()
    lines = [ln for ln in index_lines
             if any(stem in ln for stem in PIPELINE_SPILL_STEMS)]
    absent = [stem for stem in PIPELINE_SPILL_STEMS if not any(stem in ln for ln in lines)]
    assert not absent, f"{PIPELINE_SLUG}'s body never points at {absent}"
    unguided = [stem for stem in PIPELINE_SPILL_STEMS
                if not any(stem in ln
                           and re.search(r"\b[Rr]ead\b.*\b(when|before|after|only|at)\b", ln)
                           for ln in lines)]
    assert not unguided, (
        f"{PIPELINE_SLUG} names {unguided} but no line says when to read them")


# ------------------------------------------------------------- the no-skip rule


def test_nothing_in_this_file_is_allowed_to_skip():
    """Every node here runs, or this node reddens and names what was silenced.

    #4307's review rung read a `pytest.skip` in this file as an advisory against clause 4: a
    guard that exits by skipping is an invariant that stops enforcing while the fleet still
    reads green. The four live-corpus guards are now `assert …, _no_tree(…)`, and this node is
    what keeps them that way — it parses the file rather than grepping it, so the prose above
    that WORDS the word "skip" is not itself an offence, only a call is.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8", errors="replace"))
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert funcs, "no functions parsed out of this file: the rule became vacuous"

    def skip_calls(fn):
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in {"skip", "skip_if", "xfail"}:
                    yield name

    offenders = sorted(fn.name for fn in funcs if any(skip_calls(fn)))
    assert not offenders, (
        f"{offenders} skip: every corpus walk in this file is meant to fail loudly, naming "
        "the path it looked in — see `_no_tree`")
    marked = [f"{fn.name}:{ast.dump(d)[:60]}"
              for fn in funcs for d in fn.decorator_list
              if "skip" in ast.dump(d).lower()]
    assert not marked, f"skip decorator(s) on a corpus node: {marked}"


# ── the reverse direction: main's #4496 nodes, restored ────────────────────────
#
# These six nodes were in `tests/test_skill_sibling_reachability.py` at `ade16dfd` and this
# round's FIRST commit replaced the file wholesale, dropping them — the review rung's
# advisory `:1`, and one of the two halves of the fold this item exists to do. Restored as
# written, because the two directions ask different questions and only one of them is asked
# above: the nodes here ask "is every sibling `SKILL.md` POINTS AT actually on disk, with
# text under its heading?", the nodes above ask "is every file in the folder pointed at?".
# Losing these also orphaned `tests/skill_sibling_reachability.py`, the shipped detector —
# these were its only consumer — and with `scripts/skill_lint.py`'s `SPILL_SAMPLE` excluding
# this skill, the reverse direction was policed by nothing at all. `unrecovered_spans` is
# #4496's answer to skill_lint's 40-line tail heuristic: the only checker that can tell a
# section the fold MOVED from one it TRUNCATED, graded over main's body and the spliced
# folder rather than a sampled tail. They stay UNMARKED (not `live_vault`), as #4496 wrote
# them and as `test_skill_tool_names.py` argues the trade: a guard the gate deselects
# enforces nothing, and a fold is exactly the vault write they exist to catch.

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.skill_sibling_reachability import (  # noqa: E402
    INDEX_BULLET_RE, folder_violations, missing_index_files, pointer_rows,
    publishes_reachability_claim, unreachable)
from tests.skill_folder import splice_union  # noqa: E402
SKILL_DIR = SKILLS_DIR / "autonomy-data-pipeline"
SKILL = SKILL_DIR / "SKILL.md"

#: A two-section skill with one section folded out, used as the shape the negative
#: controls break. `## Kept` stays in `SKILL.md`; `## Folded` lives in `sibling.md`.
MAIN = """# Index

## Kept

The rule that stayed behind.

## Folded

Full text: `sibling.md` — read before running this step.
"""

SIBLING = """## Kept

Not here.

## Folded

The moved rule: print the denominator or the step did not happen.
"""


def _siblings(main: str, *pairs: tuple[str, str]) -> dict[str, str]:
    return dict(pairs)


def test_the_folded_skill_keeps_every_pointered_section_reachable():
    """The live claim: every `Full text:` pointer resolves to text under its heading.

    Three shapes are one assertion because all three make the same reader blind — a
    sibling that was never written, one that carries the section under a reworded
    heading (`splice_union` joins by title, so the prose lands in a tail append and a
    section-scoped reader never sees it), and a heading left empty. The skill carries 17
    `Full text:` pointers across 9 files, measured 2026-11-22, so the count assertion
    below is what stops a fold that deleted every pointer reading as a pass.
    """
    assert SKILL.is_file(), f"no {SKILL}: this module polices that file, so it reports"
    violations = folder_violations(SKILL_DIR)
    assert violations == [], (
        "the fold published in the skill's last line is not true of the folder:\n  "
        + "\n  ".join(violations))
    rows = pointer_rows(SKILL.read_text(encoding="utf-8"))
    assert len(rows) == 17, (
        f"the skill carries {len(rows)} `Full text:` pointers, expected 17 as measured "
        "2026-11-22 — a different count means sections moved again, and this "
        "node's silence about a *missing* pointer is the hole to keep shut")


def test_a_pointer_to_a_sibling_that_is_not_in_the_folder_is_named(tmp_path):
    """Non-vacuity, shape one: the file the pointer names does not exist.

    This is the #85 failure class in prose form — a doc naming a file the checkout
    lacks — and without this control the clean live result above would only prove the
    checker never found anything to look at."""
    folder = tmp_path / "folded"
    folder.mkdir()
    (folder / "SKILL.md").write_text(MAIN, encoding="utf-8")
    assert pointer_rows(MAIN) == [("Folded", "sibling.md")], (
        "the fixture stopped being the shape the checker reads")
    problems = unreachable(MAIN, _siblings(MAIN))
    assert problems == ["`sibling.md` named under `Folded` is not in the folder"], problems
    assert folder_violations(folder) == problems, (
        "the disk route disagrees with the pure checker, so the live node above is "
        "grading a different rule than the controls here")


def test_a_relabelled_heading_in_the_sibling_is_named_naming_both_titles():
    """Non-vacuity, shape two: the sibling exists but retitled the section.

    #4192's finding is the reason this is the important control: `splice_union` joins a
    sibling's body to `SKILL.md` only on an exact stripped-title match, so
    `## Folded, rewritten` against `## Folded` puts the prose in the WHOLE-FILE union
    (appended, where a keyword scan still finds it) but not in the section span — every
    section-scoped assertion loses it while every keyword assertion passes. The message
    from `unreachable` must name both titles, because the fix is to re-copy the heading
    line verbatim."""
    moved = SIBLING.replace("## Folded\n", "## Folded, rewritten\n")
    problems = unreachable(MAIN, _siblings(MAIN, ("sibling.md", moved)))
    assert len(problems) == 1, problems
    assert "sibling.md" in problems[0] and "Folded" in problems[0], problems[0]
    assert "Folded, rewritten" in problems[0], (
        f"the message must name the heading the sibling actually carries: {problems[0]}")
    union = splice_union(MAIN, [("sibling.md", moved)])
    assert "The moved rule" in union, (
        "the retitled section's text should survive somewhere in the whole-file union — "
        "that is what makes a keyword assertion pass on a broken fold")
    span = union[union.index("## Folded\n"):]
    span = span[:span.index("\n## ")] if "\n## " in span else span
    assert "The moved rule" not in span, (
        f"a section-scoped reader is supposed to lose a retitled sibling's text; it "
        f"didn't, so this control no longer describes the defect: {span!r}")


def test_a_heading_the_sibling_carries_with_no_text_under_it_is_named():
    """Non-vacuity, shape three: an empty section is a landmark, not the promised text."""
    empty = SIBLING.replace(
        "The moved rule: print the denominator or the step did not happen.", "")
    problems = unreachable(MAIN, _siblings(MAIN, ("sibling.md", empty)))
    assert problems == ["`sibling.md` carries `Folded` with no text under it"], problems


def test_an_index_naming_a_sibling_the_folder_lacks_is_named(tmp_path):
    """The index half: the `Files in this skill` list is an instruction to go read a file.

    A run that obeys the index and finds nothing improvises, which is the behaviour
    #1240 was filed for. The live folder's 11 index bullets and its 11 sibling files
    agree (measured 2026-11-22), so this control is what proves the direction is
    actually enforced rather than trivially satisfied by a folder in agreement."""
    folder = tmp_path / "indexed"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        MAIN + "\n## Files in this skill\n\n- `sibling.md` — the folded rule\n"
               "- `never-written.md` — a rule nobody moved\n", encoding="utf-8")
    assert missing_index_files(MAIN, _siblings(MAIN, ("sibling.md", SIBLING))) == [], (
        "the control's positive arm must hold, or the negative arm proves nothing")
    problems = folder_violations(folder)
    assert any("never-written.md" in p for p in problems), (
        f"an index bullet naming an absent file went unreported: {problems}")


def test_the_fleet_of_skills_that_publish_the_promise_are_all_reachable():
    """Every skill on this box that left a `Full text:` pointer is clean.

    Scoped by the promise, not by the filesystem: `publishes_reachability_claim` is what
    puts a folder inside this loop, so a skill that keeps all its prose in `SKILL.md` is
    not judged by a contract it never signed, and a skill that folds a section tomorrow
    is judged the moment it says so. A sibling the `Files in this skill` index never
    named is PRINTED rather than asserted — the census printed 20 unnamed
    `run-log-*.md` files in `kg-mention-classifier` on 2026-11-22, and those are daily run
    logs, whose growing count is the point of having them, so a rail there is a rail that
    gets disabled.
    """
    claiming, census = _skill_folders()
    assert claiming, (
        "no skill folder carries a `Full text:` pointer line, which means the fold route "
        "this check polices has been un-done everywhere and the check is dead")
    bad = {name: rows for name, rows in
           ((name, folder_violations(path)) for name, path in sorted(claiming.items()))
           if rows}
    assert bad == {}, ("folded skills whose published pointer text is not reachable:\n"
                       + "\n".join(f"  {name}: {rows}" for name, rows in sorted(bad.items())))
    print(f"skills publishing the promise: {len(claiming)}; siblings the index does "
          f"not name (reported, not asserted): {census or 'none'}")


def _skill_folders() -> tuple[dict[str, Path], dict[str, list[str]]]:
    """`(name -> folder)` for the skills that publish the promise, plus the unnamed census."""
    claiming: dict[str, Path] = {}
    census: dict[str, list[str]] = {}
    for folder in sorted(p for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file()):
        main = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        named = set(INDEX_BULLET_RE.findall(main)) | {
            name for _title, name in pointer_rows(main)}
        unlisted = sorted(p.name for p in folder.glob("*.md")
                          if p.name != "SKILL.md" and p.name not in named)
        if unlisted:
            census[folder.name] = unlisted
        if publishes_reachability_claim(main):
            claiming[folder.name] = folder
    return claiming, census
