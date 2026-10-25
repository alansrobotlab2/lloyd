"""#3299: what a run-record citation is, and whether a record still answers to it.

The property this module's existence rests on: a skill cites `run_<task>_<ts>` as the
evidence for a rule, and the corpus those ids point into is deleted on a horizon by
design — `RUN_RECORD_MAX_AGE_DAYS = 30` (`scripts/groundskeeper/retention-sweep.py:590`,
consumed by `sweep_autonomy_runs` at :869/:881), run by autonomy task #79 `frequency:
daily`. Nothing told the writer the citation had an expiry, so 15 ids in 10 vault skill
files had silently decayed by 2026-10-25 while every rail stayed green.

Why these tests are not marked `live_vault`. The two things that must not regress are
(a) which tokens the extractor sees and (b) which names the corpus answers to, and both
are pure functions of text and a directory listing. The LIVE corpus is what the report
grades, and it is graded by `scripts/skill_run_citation.py` itself — a test that read the
live records would flip red the morning task #79 sweeps another day's worth, which is the
same trap `tests/test_skill_tool_names.py:417` put `autonomy-runs/` in `RUNTIME_FIRST` to
avoid. Every node here plants its own corpus.
"""
import importlib.util
import re
import pytest
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "skill_run_citation", _ROOT / "scripts" / "skill_run_citation.py")
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)

# A day no real record can occupy, so a planted id can never resolve against a corpus
# that happens to be the live one.
DEAD = "run_99_20991231_235959"
LIVE = "run_7_20260925_010203"


def _corpus(tmp_path, names):
    """A fixture `$LLOYD_DATA/autonomy-runs` holding exactly `names`.

    `names` are spelled as citations are (no `.md`), because that is the form a test
    reasons about; the on-disk form gains `.md` unless the caller asked for `.gz`, since
    `record_names` reads only `*/*.md` and `*/*.md.gz` and a suffix-less file is a corpus
    that holds nothing — which would make every "resolves" assertion below a pass on an
    empty list.
    """
    for name in names:
        fname = name if name.endswith((".md", ".md.gz")) else f"{name}.md"
        p = tmp_path / "autonomy-runs" / name.split("_")[1] / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# run\n", encoding="utf-8")
    return tmp_path / "autonomy-runs"


# ── the extractor ──────────────────────────────────────────────────────────

def test_every_spelling_of_a_citation_is_extracted_once():
    """Clause 1's front half: bare, path-prefixed, `.md` or not, one token each.

    The four spellings are the ones the live corpus actually contains, taken from
    `~/obsidian/skills/*/SKILL.md` on 2026-10-25. Deduplication matters because the
    land door reports an added set, and a token repeated in one file would otherwise be
    reported as many refusals as the writer wrote — pushing them to shorten prose.
    """
    text = (
        "the run found it (`run_58_20260920_104452.md`)\n"
        "see `autonomy-runs/58/run_58_20260920_104452.md`\n"
        "and bare: run_58_20260920_104452, again ~$LLOYD_DATA/autonomy-runs/58/"
        "run_58_20260920_104452.md\n")
    assert src.citations_in_prose(text) == ["run_58_20260920_104452"]


def test_a_citation_inside_a_fenced_block_is_not_read_as_one():
    """Fenced commands are shown, not cited: `ls`ing a record is not evidence of a rule.

    Measured 2026-10-25, because this is one of the two rules that make the module's
    count differ from the item's: 3 occurrences of 3 distinct ids sit inside fences in 2
    files (`autonomy-data-pipeline` twice, `nightly-skill-consolidation` once), against
    the whole-text reading's 51 occurrences / 43 ids and this extractor's 42 / 40. The
    `~~~` arm is CommonMark's, and an untracked fence would silently turn every line
    after it into a command — which is how a fence rule becomes an invisibility rule.
    """
    text = ("prose cites `run_1_20260101_000000`\n"
            "```bash\nls ~/lloyd-data/autonomy-runs/2/run_2_20260102_000000.md\n```\n"
            "~~~\nrun_3_20260103_000000\n~~~\n"
            "prose again cites run_4_20260104_000000\n")
    assert src.citations_in_prose(text) == ["run_1_20260101_000000",
                                            "run_4_20260104_000000"]


def test_a_timestamp_is_not_a_citation():
    """`run_` is the anchor so an 8-digit-plus-6-digit timestamp never matches.

    A skill page full of `generated: 2026-10-25T05:05:25Z` stamps and ISO windows is
    what `nightly-reflection-signals` actually reads, and the shape `20261025_050525`
    satisfies a bare `[0-9]{8}_[0-9]{6}` exactly. A rail that refused those would be
    loosened by whoever met it first, which is why the prefix is load-bearing and
    pinned here rather than assumed.
    """
    body = ("generated: 2026-10-25T05:05:25Z, window 20261025_050525 to 20261024_050525, "
            "and the run itself is run_7_20261025_050525\n")
    assert src.citations_in_prose(body) == ["run_7_20261025_050525"]


# ── resolution against the corpus (clauses 1 and 2) ────────────────────────

def test_a_planted_id_naming_no_record_comes_back_gone(tmp_path):
    """Clause 2, first half: the module can say `gone` at all.

    Without this leg the instrument would be a counter that always answers
    `resolves`, which is exactly the failure `tests/test_skill_tool_names.py`'s own
    vacuity guards exist for: a green run and an inert check look identical.
    """
    runs = _corpus(tmp_path, [LIVE])
    assert src.verdict(LIVE, src.record_names(runs)) == src.RESOLVES
    assert src.verdict(DEAD, src.record_names(runs)) == src.GONE


def test_a_citation_missing_the_writer_appended_suffix_still_resolves(tmp_path):
    """Clause 2, second half, and the case that decided the resolution rule.

    #3299's triage found `run_scheduled-task_20261016_233600` cited while the record on
    disk is `run_scheduled-task_20261016_233600_69ae2b.md`: the writer mints a 6-hex
    suffix the citation omits. Exact-name matching would report that as decay the corpus
    does not have, and a report full of false deaths is a report nobody reads — so a
    citation resolves as a NAME PREFIX, and this node is the reason, not an optimisation.
    """
    runs = _corpus(tmp_path, ["run_scheduled-task_20261016_233600_69ae2b.md"])
    assert src.verdict("run_scheduled-task_20261016_233600",
                       src.record_names(runs)) == src.RESOLVES


def test_a_record_compressed_by_the_sweeper_still_counts(tmp_path):
    """`sessions/*.json` are gzipped rather than deleted; a run record could be too.

    `retention-sweep.py`'s rule 10 gzip-compresses session transcripts instead of
    unlinking them, so `.md.gz` is a state this corpus can legitimately be in. Ignoring
    it would call a compressed citation dead and hand the writer a refusal they cannot
    act on except by deleting true prose.
    """
    runs = _corpus(tmp_path, [LIVE])
    (runs / LIVE.split("_")[1] / f"{LIVE}.md").rename(
        runs / LIVE.split("_")[1] / f"{LIVE}.md.gz")
    assert src.verdict(LIVE, src.record_names(runs)) == src.RESOLVES


def test_a_stamp_written_on_the_local_clock_is_gone_not_resolves(tmp_path):
    """One of the two ids #3299 found that was never alive, kept as a test.

    `run_40_20261023_020003` is cited in `nightly-reflection-config` while the record is
    `run_40_20261023_090003.md` — the same instant spelled on PDT and UTC. Prefix matching
    must NOT let the wrong spelling resolve: the mismatch is seven hours and the citation
    names a record that does not exist, so it is `gone`, and this node pins which way the
    prefix rule points. (Whether to re-point the citation is #3299's owed ruling.)
    """
    runs = _corpus(tmp_path, ["run_40_20261023_090003"])
    assert src.verdict("run_40_20261023_020003", src.record_names(runs)) == src.GONE


def test_the_horizon_is_read_from_the_sweeper_not_repeated_here():
    """The 30 in the writer-facing prose has to be the 30 the deleter uses.

    Two literals that agree today are two literals that disagree the day someone moves
    the sweep — and the moving is the point, so the module reads the constant's own line.
    The sweeper is not importable from here (it is a script with a data-root import at
    module scope), hence one regex on its source, which is the same shape
    `tests/test_retention_sweep.py` is left with.
    """
    assert src.horizon_days() == 30
    body = src._SWEEP.read_text(encoding="utf-8")
    assert re.search(r"^RUN_RECORD_MAX_AGE_DAYS\s*=\s*30$", body, re.MULTILINE)


def test_added_citations_is_a_set_difference_not_a_line_diff():
    """The land door's added-only rule, at the level of the extractor.

    A re-wrapped paragraph must not be reported as a new citation: refusing a writer for
    re-flowing a sentence is how the rail gets bypassed by re-flowing on purpose. And a
    NEW file (`old_text=""`) adds everything it cites, which is what makes a fresh skill
    graded rather than exempt.
    """
    old = "the finding is `run_58_20260920_104452` and it is\nstill one sentence here\n"
    new = "the finding is `run_58_20260920_104452` and it is still one sentence here\n"
    assert src.added_citations(new, old) == []
    assert src.added_citations(new + "\nalso run_7_20260101_010101\n", old) == \
        ["run_7_20260101_010101"]
    assert src.added_citations(old, "") == ["run_58_20260920_104452"]


# ── the report (ask 2) ─────────────────────────────────────────────────────

def _vault(tmp_path, files):
    for rel, body in files.items():
        p = tmp_path / "skills" / rel / "SKILL.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return tmp_path


def test_the_report_counts_citations_and_names_the_dead_ones(tmp_path):
    """The 15 pre-existing ids are visible somewhere, or they can never be closed.

    The refusal only ever sees what a diff adds, so without this the dangling population
    is known to nobody. Counts are per-file as well as fleet-wide because the repair is
    per-file work.
    """
    vault = _vault(tmp_path, {
        "alpha": f"cites `run_2_20260101_000000` and `{DEAD}`\n",
        "beta": f"cites `{DEAD}` twice {DEAD}\n",
        "gamma": f"cites `{LIVE}`\n"})
    runs = _corpus(tmp_path, [LIVE, "run_2_20260101_000000"])
    rep = src.report(vault, runs)
    assert rep["gone_ids"] == [DEAD]
    assert rep["gone_distinct"] == 1 and rep["gone_citations"] == 2
    assert sorted(rep["by_file"]) == ["skills/alpha/SKILL.md", "skills/beta/SKILL.md"]
    # `citations` counts (file, id) pairs, not occurrences: beta names the dead id twice
    # on one line and counts once. Per-file dedupe is the extractor's, so a writer who
    # repeats an id in one file is not refused twice for it.
    assert rep["citations"] == 4 and rep["distinct"] == 3
    assert rep["files_scanned"] == 3


def test_the_report_counts_a_fenced_citation_nowhere(tmp_path):
    """The report's denominator is prose, matching the extractor the refusal uses.

    `report()` and the land door must disagree on nothing, or the number a human reads
    and the number that refuses a land are two measurements of different corpora and the
    first is not evidence about the second.
    """
    vault = _vault(tmp_path, {"alpha": f"cites `{LIVE}`\n```\n{DEAD}\n```\n"})
    rep = src.report(vault, _corpus(tmp_path, [LIVE]))
    assert rep["citations"] == 1 and rep["gone_distinct"] == 0


# ── clause 5: the horizon is stated where citations are written ────────────

def test_the_writer_facing_run_id_clause_carries_the_horizon():
    """`nightly-reflection-config` tells writers to name a dated run id; it must say for
    how long that id means anything.

    Read from the LIVE vault, because that file is the corpus — `skills/` is not in this
    checkout (`git ls-files skills/` is empty), so the thing under test is the vault
    itself, exactly as #3292's architecture-doc claim nodes read their corpus from the
    tree that holds it. Skips rather than passes when the vault is absent: a checkout on
    a machine with no vault has no opinion about vault prose. The numbers are the ones a
    writer needs (30, the constant's name, the task that deletes), and the point is that
    the sentence telling a model to name an id is the sentence that must also tell it the
    id expires — 15 dangling ids were in this corpus the day #3299 counted.
    """
    skill = Path.home() / "obsidian" / "skills" / "nightly-reflection-config" / "SKILL.md"
    if not skill.is_file():
        pytest.skip("no vault on this box")
    body = skill.read_text(encoding="utf-8", errors="replace")
    i = body.find("a dated run id")
    assert i >= 0, "the clause that tells a writer to name a dated run id has moved"
    window = body[max(0, i - 700): i + 1400]
    assert "RUN_RECORD_MAX_AGE_DAYS" in window, window[:200]
    assert "30 days" in window, window[:200]
    assert "Retention Sweep" in window, "the deleter has to be named: " + window[:200]
