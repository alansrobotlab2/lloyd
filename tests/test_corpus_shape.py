"""#761: scripts/maintenance/corpus_shape.py — cross-run shape of four corpora.

Every test builds its corpora under tmp_path; nothing reads the real vault or
the live `_pipeline/`.
"""
import importlib.util
import json
import os
import shutil
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "maintenance" / "corpus_shape.py"
NOW = "2026-09-20T03:00:00+00:00"          # reference day: 2026-09-20 UTC
PREV_DAY = "2026-09-19T03:00:00+00:00"     # the day before it, same hour
NEXT_DAY = "2026-09-21T03:00:00+00:00"     # the day after it, same hour
SAME_DAY_LATER = "2026-09-20T23:30:00+00:00"   # a retry: same UTC date, 20.5 h on
METRICS = ("n", "len_mean", "len_p95", "distinct_key_ratio", "duplicate_rate",
           "self_reference_rate")


def _load():
    spec = importlib.util.spec_from_file_location("corpus_shape", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load()


def _build(root: Path) -> tuple[Path, Path]:
    vault, pipeline = root / "vault", root / "pipeline"
    mem = vault / "memory"
    mem.mkdir(parents=True)
    for day, topic in (("2026-09-18", "backups"), ("2026-09-19", "voice"),
                       ("2026-09-20", "the guardian"), ("2026-08-01", "old")):
        (mem / f"{day}.md").write_text(
            f"---\ntype: note\n---\n## Morning on {topic}\n\nWe looked at {topic} for a while today.\n\n"
            f"## Evening on {topic}\n\nThe daily note for {topic} was written late.\n")
    lloyd = vault / "lloyd"
    lloyd.mkdir()
    (lloyd / "USER.md").write_text(
        "---\ntype: note\n---\n# User\n\n## operational\n"
        "- **Scope**: agent memory lives in the vault.\n"
        "- **Ceiling**: USER.md is capped at sixteen kilobytes,\n  enforced by nobody.\n")
    (lloyd / "MEMORY.md").write_text("# Memory\n\n- **Restarts** go through the round CLI.\n")
    for name in ("alpha", "beta"):
        d = vault / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}\n\nDo the {name} thing carefully.\n")
    traj = pipeline / "trajectories"
    traj.mkdir(parents=True)
    rows = [{"session_key": "s1", "tools": []}, {"session_key": "s2", "tools": []},
            {"session_key": "s2", "tools": []}]
    (traj / "2026-09-19.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (traj / "2026-08-01.jsonl").write_text(json.dumps({"session_key": "old"}) + "\n")
    _add_digests(vault)
    return vault, pipeline


#: A rubric sentence #2011's hand-written acceptance grep MISSES entirely — no
#: `the user's`, no `interest profile`, no bare `tangential`, no `lacks relevance` —
#: while the shipped guard flags it. Every injection below uses it, because a
#: sentence the old grep would have caught cannot pin the blind spot closed.
RUBRIC_SENTENCE = ("Directly intersects robotics and AI/LLMs, aligning perfectly "
                   "with top-weighted interests.")


def _add_digests(vault: Path) -> None:
    """Three category digests: two with a descriptive body, one whose only
    rubric-looking lines are structure.

    The third (`local-llm`) is the skip list made arithmetic. Its heading, its
    `**Source:**` line and its `[Link]` line each hold a sentence the guard flags,
    and none of them is an entry body, so a correct sweep reads `prose_lines=1
    flagged=0` there. Drop `DIGEST_STRUCTURAL_PREFIXES` and that same digest reads
    `flagged=3`, which is the difference between a count of bodies and a count of
    whatever a heading happened to say.
    """
    for cat, title, body in (
            ("ai-llms", "A video about caching",
             "Muse adds a KV cache preset per model; the note walks through the config."),
            ("robotics", "ROS2 image_to_3d package demo",
             "A ROS2 package turns one image into a point cloud and publishes it as a topic."),
    ):
        d = vault / "knowledge" / cat
        d.mkdir(parents=True)
        (d / "youtube-digest.md").write_text(
            "---\ntype: note\n---\n# YouTube digest\n\n"
            "## 2026-09-19\n\n"
            f"### {title}\n\n"
            "**Source:** youtube | **Relevance:** 8/10\n\n"
            f"{body}\n\n"
            "[Link](https://example.invalid/a)\n\n"
            "---\n", encoding="utf-8")
    structural = vault / "knowledge" / "local-llm"
    structural.mkdir(parents=True)
    (structural / "youtube-digest.md").write_text(
        "# YouTube digest\n\n"
        "## Aligning perfectly with top-weighted interests\n\n"
        "### One short\n\n"
        "**Source:** youtube | aligns with the ai-llms interest, though it lacks "
        "specific relevance to robotics\n\n"
        "The channel published one short video this week.\n\n"
        "[Link](https://example.invalid/b) — matching the reader's core interests\n\n"
        "---\n", encoding="utf-8")


def _digest(vault: Path, cat: str) -> Path:
    return vault / "knowledge" / cat / "youtube-digest.md"


def _run(vault, pipeline, out, *extra, now=NOW):
    return cs.main(["--vault", str(vault), "--pipeline", str(pipeline),
                    "--out-dir", str(out), "--now", now, "--days", "7", *extra])


def test_prints_every_metric_for_each_of_the_four_corpora(tmp_path, capsys):
    vault, pipeline = _build(tmp_path)
    assert _run(vault, pipeline, tmp_path / "out") == 0
    out = capsys.readouterr().out
    for corpus in cs.CORPORA:
        line = next(l for l in out.splitlines() if l.strip().startswith(corpus + " ["))
        for metric in METRICS:
            assert f"{metric}=" in line, (corpus, metric, line)
    assert "dated 2026-09-14..2026-09-20" in out

    report = json.loads(next((tmp_path / "out").glob("corpus-shape-*.json")).read_text())
    c = report["corpora"]
    assert c["daily"]["n"] == 6                        # the August note is outside the window
    assert c["daily"]["self_reference_rate"] == 0.5    # each "Evening" names the daily note
    assert c["user_memory"]["n"] == 3
    assert c["user_memory"]["self_reference_rate"] == round(1 / 3, 4)
    assert c["skills"]["n"] == 2
    assert c["trajectories"]["n"] == 3
    assert c["trajectories"]["distinct_key_ratio"] == round(2 / 3, 4)
    assert c["trajectories"]["duplicate_rate"] == round(1 / 3, 4)


def test_two_utc_dates_append_both_rows_and_never_rewrite_the_prior_day(tmp_path):
    """Clause 2 of #1576, and the re-scoped form of this file's old
    `test_a_run_never_rewrites_a_prior_file`.

    That test pinned the invariant with two runs sharing ONE pinned `--now`, so it
    asserted a second same-stamp row exists — the exact behaviour #1576 removes. The
    acceptance re-scopes "never rewritten" to prior *days* ("Running it with --now on
    two different UTC dates still appends both rows, the row written for the earlier
    date is not rewritten by the later run, and previous_run() returns the later row"),
    so the second run here is the next day's.
    """
    vault, pipeline = _build(tmp_path)
    out = tmp_path / "out"
    _run(vault, pipeline, out)
    (first,) = list(out.glob("corpus-shape-*.json"))
    before = first.read_bytes()
    os.chmod(first, 0o444)          # an overwrite attempt would raise, not pass quietly
    assert _run(vault, pipeline, out, now=NEXT_DAY) == 0
    (second,) = set(out.glob("corpus-shape-*.json")) - {first}
    assert first.read_bytes() == before
    assert (cs._row_date(first), cs._row_date(second)) == (date(2026, 9, 20), date(2026, 9, 21))
    # The later row is the one a third run diffs against.
    assert cs.previous_run(out)[0] == second
    assert cs.previous_run(out, before=date(2026, 9, 21))[0] == first


def test_a_retry_inside_one_utc_date_writes_no_second_row(tmp_path):
    """Clause 1 of #1576: the row key is the UTC DATE of the resolved `--now`.

    Three runs, one date — 03:00Z, the same UTC day at 23:30Z, and 02:00 at +05:00,
    whose *local* date is already 2026-09-21 but which resolves to 2026-09-20 UTC.
    Keying on wall clock would append on the retry (no row carries "today"), and
    keying on the unconverted local date would append on the third run; either way
    this goes red. So does a replace-based guard, which would fail on the chmod.
    """
    vault, pipeline = _build(tmp_path)
    out = tmp_path / "out"
    assert _run(vault, pipeline, out, "--quiet") == 0
    (first,) = list(out.glob("corpus-shape-*.json"))
    before = first.read_bytes()
    os.chmod(first, 0o444)          # a replace would raise here rather than pass quietly
    assert _run(vault, pipeline, out, "--quiet", now=SAME_DAY_LATER) == 0
    assert _run(vault, pipeline, out, "--quiet", now="2026-09-21T02:00:00+05:00") == 0
    assert list(out.glob("corpus-shape-*.json")) == [first]
    assert first.read_bytes() == before
    assert cs.row_for(out, date(2026, 9, 20)) == first
    assert cs.row_for(out, date(2026, 9, 21)) is None


def test_a_same_day_retry_diffes_against_the_earlier_day_and_still_reports_moved(
        tmp_path, capsys):
    """Clause 3 of #1576: the day's own row is never the diff base.

    Trajectories grow past the ±50% `n` bound between the 2026-09-19 row and the
    2026-09-20 row, so `n 3->12` is MOVED against 09-19. The retry runs on 2026-09-20
    as well; without the guard its base is the 09-20 row it is re-measuring, its
    header names that row, and every metric reads unchanged — exit 0 and no finding.
    With the guard the header names the 09-19 row and the MOVED line survives.
    """
    vault, pipeline = _build(tmp_path)
    out = tmp_path / "out"
    assert _run(vault, pipeline, out, "--quiet", now=PREV_DAY) == 0
    base = cs.previous_run(out)[0]
    extra = "".join(json.dumps({"session_key": f"n{i}"}) + "\n" for i in range(9))
    with open(pipeline / "trajectories" / "2026-09-19.jsonl", "a") as fh:
        fh.write(extra)
    assert _run(vault, pipeline, out, now=NOW) == 2
    today = cs.previous_run(out)[0]
    assert today != base
    capsys.readouterr()

    assert _run(vault, pipeline, out, now=SAME_DAY_LATER) == 2
    printed = capsys.readouterr().out
    header = printed.splitlines()[0]
    # The `-> path` half may name the day's row (it is the row that stands for the
    # day); the clause is about the base, so read the `(vs …)` part of the header.
    assert "(vs " in header, header
    base_named = header.split("(vs ", 1)[1].split(")", 1)[0]
    assert base_named == base.name, header
    assert base_named != today.name, header
    assert "nothing written" in header, header
    assert next(l for l in printed.splitlines()
                if l.startswith("trajectories: MOVED")).count("n 3->12")
    assert len(list(out.glob("corpus-shape-*.json"))) == 2


def test_diff_says_no_prior_run_then_one_verdict_line_per_corpus(tmp_path, capsys):
    vault, pipeline = _build(tmp_path)
    out = tmp_path / "out"
    _run(vault, pipeline, out)
    first = capsys.readouterr().out.splitlines()
    for corpus in cs.CORPORA:
        assert f"{corpus}: no prior run" in first

    # Grow one corpus past its threshold, then diff against the FIRST DAY's row.
    # #1576 clause 3 makes the base the newest row from an earlier UTC date, so the
    # run that carries the delta is the next day's: this file previously ran it with
    # the same pinned --now as the first run and diffed against that run's row, the
    # same-day base the clause now forbids.
    extra = "".join(json.dumps({"session_key": f"n{i}"}) + "\n" for i in range(9))
    with open(pipeline / "trajectories" / "2026-09-19.jsonl", "a") as fh:
        fh.write(extra)
    assert _run(vault, pipeline, out, now=NEXT_DAY) == 2
    second = capsys.readouterr().out.splitlines()
    assert "no prior run" not in "\n".join(second)
    for corpus in cs.CORPORA:
        verdicts = [l for l in second if l.startswith(corpus + ": ")
                    and ("within thresholds" in l or "MOVED" in l)]
        assert len(verdicts) == 1, (corpus, second)
    assert next(l for l in second if l.startswith("trajectories: MOVED")).count("n 3->12")
    assert any(l == "skills: within thresholds" for l in second)


def test_an_injected_repeated_sentence_is_named_with_its_file(tmp_path, capsys):
    clean_vault, pipeline = _build(tmp_path / "clean")
    dirty_root = tmp_path / "dirty"
    shutil.copytree(tmp_path / "clean", dirty_root)
    injected = "Remember to verify the watermark before trusting any count."
    target = dirty_root / "vault" / "memory" / "2026-09-19.md"
    target.write_text(target.read_text().replace(
        "for a while today.", f"for a while today. {injected}").replace(
        "was written late.", f"was written late. {injected}")
        + f"\n## Night\n\n{injected}\n")

    # Baseline both copies identically, then measure: only the dirty copy speaks.
    # The three runs sit on three UTC dates because #1576 clause 3 makes a run's
    # base the newest row from an EARLIER date: two runs pinned to one date would
    # leave the dirty run with no base at all, since its own day's row is the row
    # the clause excludes.
    assert _run(clean_vault, pipeline, tmp_path / "out-clean", now=PREV_DAY) == 0
    clean_out = capsys.readouterr().out
    assert injected not in clean_out and "recurring" not in clean_out

    dirty_out_dir = tmp_path / "out-dirty"
    shutil.copytree(tmp_path / "out-clean", dirty_out_dir)
    code = _run(dirty_root / "vault", dirty_root / "pipeline", dirty_out_dir, "--quiet")
    dirty_out = capsys.readouterr().out
    assert code == 2
    line = next(l for l in dirty_out.splitlines() if injected in l)
    assert str(target) in line and "NEW recurring" in line

    # The untouched copy, measured the day after its own baseline in quiet mode,
    # says nothing: nothing in it moved, and its base is the row above, not a row
    # it wrote itself.
    assert _run(clean_vault, pipeline, tmp_path / "out-clean", "--quiet", now=NEXT_DAY) == 0
    assert capsys.readouterr().out == ""


def test_the_first_run_is_a_baseline_even_with_a_recurrence(tmp_path, capsys):
    vault, pipeline = _build(tmp_path)
    note = vault / "memory" / "2026-09-20.md"
    rep = "The watchdog can no longer perform one of its own preconditions."
    note.write_text(note.read_text() + "".join(f"\n## Alert {i}\n\n{rep}\n" for i in range(3)))
    assert _run(vault, pipeline, tmp_path / "out", "--quiet") == 0
    assert capsys.readouterr().out == ""
    report = json.loads(next((tmp_path / "out").glob("*.json")).read_text())
    assert report["corpora"]["daily"]["recurring"][0]["sentence"] == rep


# --- the recalibrated bounds (#2200): a noise floor that is measured, not guessed ---

#: The loudest day-over-day move each bound metric actually made in the live series:
#: the 8 rows written one per UTC date from 2026-09-27 to 2026-10-04, the first
#: stretch in which #1576's date key yields exactly one row per date — so 7 adjacent
#: pairs. Reproduce it by grouping `~/lloyd-data/_pipeline/metrics/corpus-shape-*.json`
#: by the 8-digit prefix of the filename, taking one row per date, and differencing
#: `corpora[<corpus>][<metric>]` between adjacent dates across all four corpora. Each
#: entry is the maximum of those differences, in the kind `THRESHOLDS` declares for
#: that metric, and the corpus and pair the maximum came from.
#:
#: A bound has to sit ABOVE the number here or the guard alerts on ordinary noise,
#: which is the defect #2200 fixes: under the table it replaced, 6 of these 8 nights
#: printed a MOVED line. Pinned by
#: `test_every_bound_sits_above_the_noise_the_series_actually_showed`.
OBSERVED_MAX_DELTA = {
    "n": ("rel", 0.5366, "daily, 2026-10-01->2026-10-02, n 41 -> 19"),
    "len_mean": ("rel", 0.1977, "skills, 2026-10-01->2026-10-02, 11977.9 -> 14345.7"),
    "len_p95": ("rel", 0.2623, "trajectories, 2026-10-01->2026-10-02, 51473 -> 64972"),
    "distinct_key_ratio": ("abs", 0.0885, "daily, 2026-10-01->2026-10-02, 0.122 -> 0.2105"),
    "duplicate_rate": ("abs", 0.1117, "daily, 2026-10-01->2026-10-02, 0.5854 -> 0.4737"),
    "self_reference_rate": ("abs", 0.0361, "skills, 2026-09-30->2026-10-01, 0.0833 -> 0.1194"),
}

#: The three rows those maxima come from, transcribed metric-for-metric out of
#: `corpus-shape-20260930T100045Z.json`, `corpus-shape-20261001T100034Z.json` and
#: `corpus-shape-20261002T001850Z.json`, plus the two adjacent pairs to replay them
#: in. Transcribed rather than read at test time: a node that re-read the live series
#: would re-pin itself whenever a night moved, and what this block asserts is a fixed,
#: dated measurement. 2026-10-02 is the row the 00:18Z run wrote — the 10:00 run that
#: night wrote nothing, #1576 — so it is the row that stands for that date.
OBSERVED_ROWS = {
    "2026-09-30": {
        "daily": {"n": 44, "len_mean": 578.7, "len_p95": 1117,
                  "distinct_key_ratio": 0.1136, "duplicate_rate": 0.5227,
                  "self_reference_rate": 0.0227},
        "user_memory": {"n": 142, "len_mean": 225.5, "len_p95": 407,
                        "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                        "self_reference_rate": 0.0141},
        "skills": {"n": 72, "len_mean": 12150.2, "len_p95": 40745,
                   "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                   "self_reference_rate": 0.0833},
        "trajectories": {"n": 2575, "len_mean": 15239.7, "len_p95": 53037,
                         "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                         "self_reference_rate": 0.0272},
    },
    "2026-10-01": {
        "daily": {"n": 41, "len_mean": 620.1, "len_p95": 1172,
                  "distinct_key_ratio": 0.122, "duplicate_rate": 0.5854,
                  "self_reference_rate": 0.0244},
        "user_memory": {"n": 142, "len_mean": 224.7, "len_p95": 407,
                        "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                        "self_reference_rate": 0.0141},
        "skills": {"n": 67, "len_mean": 11977.9, "len_p95": 43129,
                   "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                   "self_reference_rate": 0.1194},
        "trajectories": {"n": 2768, "len_mean": 15381.1, "len_p95": 51473,
                         "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                         "self_reference_rate": 0.0246},
    },
    "2026-10-02": {
        "daily": {"n": 19, "len_mean": 730.9, "len_p95": 1172,
                  "distinct_key_ratio": 0.2105, "duplicate_rate": 0.4737,
                  "self_reference_rate": 0.0},
        "user_memory": {"n": 143, "len_mean": 223.6, "len_p95": 407,
                        "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                        "self_reference_rate": 0.014},
        "skills": {"n": 52, "len_mean": 14345.7, "len_p95": 43241,
                   "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                   "self_reference_rate": 0.1346},
        "trajectories": {"n": 2274, "len_mean": 16818.7, "len_p95": 64972,
                         "distinct_key_ratio": 1.0, "duplicate_rate": 0.0,
                         "self_reference_rate": 0.0273},
    },
}
OBSERVED_PAIRS = (("2026-09-30", "2026-10-01"), ("2026-10-01", "2026-10-02"))

#: How far past the record a "beyond anything the series did" jump has to be: three
#: times the loudest night on file is comfortably past every bound while staying
#: inside what each metric can physically read (the three rates top out at 1.0).
JUMP_MULTIPLE = 3


def _observed_row(day):
    """One transcribed day, in the dict shape `diff_lines()` takes."""
    return {"corpora": {name: dict(vals) for name, vals in OBSERVED_ROWS[day].items()}}


def _delta(kind, old, new):
    """The same difference `_moved()` compares against the bound."""
    return abs(new - old) / abs(old) if kind == "rel" else abs(new - old)


def _comment_above_thresholds() -> str:
    """The `#` block sitting immediately above `THRESHOLDS`, hard wraps collapsed."""
    lines = SCRIPT.read_text().split("THRESHOLDS = {")[0].splitlines()
    block = []
    for line in reversed(lines):
        if line.strip().startswith("#"):
            block.append(line.strip().lstrip("#").strip())
        elif not line.strip():
            continue
        else:
            break
    return " ".join(reversed(block))


@pytest.mark.parametrize("metric", sorted(OBSERVED_MAX_DELTA))
def test_every_bound_sits_above_the_noise_the_series_actually_showed(metric):
    """#2200 clause 1: no bound may sit at or below a move an ordinary night made.

    `duplicate_rate` was bound at abs 0.02 and `distinct_key_ratio` at abs 0.05 while
    the daily corpus was moving 0.1117 and 0.0885 in a single night, which is why 6
    of the 8 nights printed a MOVED line about numbers nobody would call a finding.
    Each bound now clears its measured maximum: `n` 0.6 over 0.5366, `len_mean` 0.25
    over 0.1977, `len_p95` 0.5 over 0.2623, `distinct_key_ratio` 0.10 over 0.0885,
    `duplicate_rate` 0.15 over 0.1117, `self_reference_rate` 0.05 over 0.0361.
    """
    kind, observed_max, source = OBSERVED_MAX_DELTA[metric]
    assert metric in cs.THRESHOLDS, f"{metric} has a measured maximum and no bound"
    bound_kind, bound = cs.THRESHOLDS[metric]
    assert bound_kind == kind, (
        f"{metric}: the table measures {kind} deltas, the code bounds {bound_kind}")
    assert bound > observed_max, (
        f"{metric}'s bound {bound} does not clear the {observed_max} one-night move "
        f"the live series recorded ({source}); a bound under the noise fires nightly")


def test_the_comment_beside_the_bounds_and_the_measured_table_agree():
    """#2200 clause 1's other half: the comment beside `THRESHOLDS` names the sample
    the table above asserts, in the same numbers.

    The item asks for the sample dates beside the constant, and prose nobody re-checks
    is exactly how "provisional … needs a week of series" outlived the week by five
    months. So the comment is graded against `OBSERVED_MAX_DELTA`: re-derive the bounds
    from a longer series and leave the comment behind, or edit the table and leave the
    comment, and this goes red.
    """
    assert set(cs.THRESHOLDS) == set(OBSERVED_MAX_DELTA), (
        "THRESHOLDS and the measured table cover different metrics, so one of them is "
        "silently not being checked against the live series")
    comment = _comment_above_thresholds()
    for day in ("2026-09-27", "2026-10-04"):
        assert day in comment, (
            f"the comment beside THRESHOLDS never names {day}, so a reader cannot tell "
            f"which dates the bounds were measured over")
    assert "8 clean UTC dates" in comment, (
        "the comment beside THRESHOLDS does not say how many dates the bounds were "
        "derived from, which is the claim #2200 replaced 'provisional' with")
    for metric, (_, observed_max, _) in OBSERVED_MAX_DELTA.items():
        assert f"{observed_max:.4f}" in comment, (
            f"the comment beside THRESHOLDS does not record {metric}'s measured "
            f"maximum {observed_max:.4f}, so the constant and its provenance can drift")


@pytest.mark.parametrize("metric", sorted(OBSERVED_MAX_DELTA))
def test_the_transcribed_pairs_are_where_those_maxima_happened(metric):
    """Keeps the two tables above tied to each other: the maxima asserted against the
    bounds must be the maxima of the pairs replayed below, or the within-thresholds
    replay would be quiet about a night that was never the loudest one.
    """
    kind, observed_max, _ = OBSERVED_MAX_DELTA[metric]
    got = max(_delta(kind, OBSERVED_ROWS[first][corp][metric],
                     OBSERVED_ROWS[second][corp][metric])
              for first, second in OBSERVED_PAIRS for corp in cs.CORPORA)
    assert got == pytest.approx(observed_max, abs=1e-4), (
        f"{metric}: the pairs transcribed into this file peak at {got}, the table the "
        f"bounds are graded against says {observed_max} — one of the two is stale")


@pytest.mark.parametrize("first,second", OBSERVED_PAIRS)
def test_an_ordinary_night_from_the_live_series_prints_within_thresholds(first, second):
    """#2200 clause 2, first half: the two noisiest nights of the 09-27..10-04 series,
    replayed through `diff_lines()`, say nothing.

    2026-10-01 -> 2026-10-02 is the pair the guard fired on under the old table: the
    daily corpus moved `n` 41->19 (53.7% against a 50% bound), `distinct_key_ratio`
    0.122->0.2105 (0.0885 against abs 0.05) and `duplicate_rate` 0.5854->0.4737
    (0.1117 against abs 0.02), and `~/lloyd-data/autonomy-runs/90/run_90_20261002_100019.md`
    printed MOVED for numbers of exactly that size. Every one of those moves now sits
    inside its bound, so all four corpora read `within thresholds` and `moved_any` is
    False — the silent-unless-fired contract #90 ships with, back in force on nights
    like this one.
    """
    lines, moved_any = cs.diff_lines(_observed_row(first), _observed_row(second))
    assert moved_any is False, lines
    for name in cs.CORPORA:
        assert (next(l for l in lines if l.startswith(f"{name}: "))
                == f"{name}: within thresholds"), lines


@pytest.mark.parametrize("metric", sorted(OBSERVED_MAX_DELTA))
def test_a_jump_past_the_recorded_maximum_prints_moved_naming_metric_and_bound(metric):
    """#2200 clause 2, second half: the widened bounds still bite, and bite visibly.

    One metric at a time is pushed to three times its measured maximum above the
    2026-10-01 daily value the day is diffed from, leaving the other three corpora at
    their observed values so only `daily` can speak. `_moved()` prints the bound itself as the last field of
    the line — `(delta vs bound)`, the delta a signed percent for a relative metric
    or a signed 4-decimal number for an absolute one, the bound `{bound:.0%}` and
    `{bound}` respectively — so the line is checked against `THRESHOLDS` rather than a
    copy of it: the printed number is how a scheduled run tells a reader which constant
    it was judged by.
    """
    kind, observed_max, _ = OBSERVED_MAX_DELTA[metric]
    prev, cur = _observed_row("2026-10-01"), _observed_row("2026-10-02")
    base = prev["corpora"]["daily"][metric]          # what the delta is measured from
    target = JUMP_MULTIPLE * observed_max
    cur["corpora"]["daily"][metric] = (round(base + target, 4) if kind == "abs"
                                       else round(base * (1 + target)))
    bound_kind, bound = cs.THRESHOLDS[metric]
    lines, moved_any = cs.diff_lines(prev, cur)
    assert moved_any is True, lines
    line = next(l for l in lines if l.startswith("daily: "))
    assert line.startswith("daily: MOVED "), line
    assert f"{metric} " in line, line
    assert sum(1 for l in lines if "MOVED" in l) == 1, lines
    shown = f"{bound:.0%}" if bound_kind == "rel" else f"{bound}"
    assert line.endswith(f"vs {shown})"), line


def test_a_corpus_of_only_duplicates_still_exits_2_under_the_recalibrated_bounds(
        tmp_path, capsys):
    """#2200 clause 3: widening the bounds must not cost the guard its purpose.

    The defect this script exists for (#543 -> #761) is an artifact that is fine alone
    and wrong because it recurs, so a planted regression has to fire AT the new number.
    Every trajectory row is rewritten to the same text: `duplicate_rate` is
    `1 - distinct/number of items`, so three identical items read 0.6667 rather than
    1.0 — that is the metric's ceiling, not a weak injection — and the move is +0.3334,
    three times the 0.1117 the live series ever showed and twice the recalibrated 0.15
    bound. The run exits 2 and the MOVED line names the metric and the bound it crossed.
    """
    vault, pipeline = _build(tmp_path)
    out = tmp_path / "out"
    assert _run(vault, pipeline, out, "--quiet", now=PREV_DAY) == 0
    capsys.readouterr()
    one = json.dumps({"session_key": "s2", "tools": []})
    (pipeline / "trajectories" / "2026-09-19.jsonl").write_text(
        "".join(one + "\n" for _ in range(3)))

    assert _run(vault, pipeline, out, now=NEXT_DAY) == 2
    printed = capsys.readouterr().out
    moved = next(l for l in printed.splitlines()
                 if l.startswith("trajectories: MOVED"))
    assert "duplicate_rate 0.3333->0.6667 (+0.3334 vs 0.15)" in moved, moved


def test_an_n_multiplied_tenfold_still_exits_2_under_the_recalibrated_bounds(
        tmp_path, capsys):
    """#2200 clause 3's other half: the shape the trend check most often gets called
    for — a corpus that grew by an order of magnitude overnight — still fires at `n`'s
    new 60% bound, and the header still names the row it judged against.

    27 rows go into the file the baseline measured, so `n` reads 3->30: +900%, fifteen
    times the 0.5366 the live series peaked at and far past the 0.6 that replaced it.
    The other three corpora are untouched, so nothing else can make this exit 2.
    """
    vault, pipeline = _build(tmp_path)
    out = tmp_path / "out"
    assert _run(vault, pipeline, out, "--quiet", now=PREV_DAY) == 0
    capsys.readouterr()
    traj = pipeline / "trajectories" / "2026-09-19.jsonl"
    with open(traj, "a") as fh:
        fh.write("".join(json.dumps({"session_key": f"x{i}"}) + "\n" for i in range(27)))

    assert _run(vault, pipeline, out, now=NEXT_DAY) == 2
    printed = capsys.readouterr().out
    assert "(vs corpus-shape-20260919T030000Z.json)" in printed.splitlines()[0], printed
    moved = next(l for l in printed.splitlines()
                 if l.startswith("trajectories: MOVED"))
    assert "n 3->30 (+900% vs 60%)" in moved, moved


@pytest.mark.parametrize("old,new,hit", [
    (10, 15.5, False),     # +55%: under the 0.6 bound, and above the 0.5366 noise peak
    (10, 16.5, True),      # +65%: over it
    (0, 0, False),
])
def test_relative_threshold_on_n(old, new, hit):
    """#2200 clause 4: the pin brackets the constant rather than sitting on it.

    This node pinned 0.5 with `(10, 16) -> True`, and 16 is exactly +60% — the new
    bound — so it would have read False and gone red the moment the bound moved,
    and at the old bound it sat on the boundary where a rounding change flips the
    answer. The daily corpus's 41->19 move (0.5366) is what forced `n` from 0.5 to
    0.6, so the fixture moved with it: +55% stays quiet, +65% fires.
    """
    assert cs._moved("n", old, new)[0] is hit


def _added_files(patch: Path) -> dict:
    """{path: text} for each file a `git diff` patch creates."""
    files, cur = {}, None
    for line in patch.read_text().splitlines():
        if line.startswith("+++ b/"):
            cur = line[len("+++ b/"):]
            files[cur] = []
        elif cur and line.startswith("+"):
            files[cur].append(line[1:])
    return {k: "\n".join(v) + "\n" for k, v in files.items()}


def test_the_scheduled_task_runs_quiet_off_hours_and_names_its_skill():
    """The task lives in the vault, so it is held as a patch until this lands
    (the `vault-automod-skill-blast-radius.patch` precedent): applied earlier, it
    would schedule a script production does not have yet."""
    import yaml

    files = _added_files(ROOT / "scripts" / "maintenance" / "vault-corpus-shape-task.patch")
    task_path = next(p for p in files if p.startswith("autonomy/"))
    front = yaml.safe_load(files[task_path].split("---\n")[1])
    assert front["type"] == "autonomy" and front["status"] == "up_next"
    assert set(front["preferred_hours"]) <= {23, 0, 1, 2, 3, 4, 5}
    assert "scripts/maintenance/corpus_shape.py --quiet" in front["description"]
    assert "exit code 2" in front["description"]
    skill = files[f"skills/{front['skill_name']}/SKILL.md"]
    assert "corpus_shape.py --quiet" in skill and "exit code 2" in skill


# --- the digest sweep (#2039): a count kept with the shipped guard, not a grep ---

def _digest_rows(report: dict) -> dict:
    """{digest path relative to the vault: the file's own sweep row}."""
    return {f["file"]: f for f in report["interest_profile"]["files"]}


def test_the_digest_count_prints_one_line_per_digest_and_reads_zero_when_clean(
        tmp_path, capsys):
    """Clause 3 of #2039: one per-digest count per digest file found, each reading 0.

    Three digests exist in the fixture and exactly three lines are printed — a digest
    silently missing from the print is a digest nobody is watching, which is how the
    robotics entry stayed published: nothing was counting it at all. The clean bodies
    read `flagged=0`, and `local-llm` reads it while three flagged-looking sentences
    sit on its heading, its `**Source:**` line and its `[Link]` line, so the 0 is the
    skip list holding and not the sweep looking away from the file.
    """
    vault, pipeline = _build(tmp_path)
    assert _run(vault, pipeline, tmp_path / "out") == 0
    out = capsys.readouterr().out

    printed = [l for l in out.splitlines() if l.startswith("interest_profile [")]
    assert len(printed) == 3, printed
    assert all("flagged=0" in l for l in printed), printed
    assert "interest_profile [knowledge/robotics/youtube-digest.md]: lines_checked=2 " \
           "flagged=0" in out, out
    assert "interest_profile [knowledge/local-llm/youtube-digest.md]: lines_checked=1 " \
           "flagged=0" in out, out

    report = json.loads(next((tmp_path / "out").glob("corpus-shape-*.json")).read_text())
    assert report["interest_profile"]["files_count"] == 3
    assert report["interest_profile"]["total_flagged"] == 0
    assert _digest_rows(report)["knowledge/local-llm/youtube-digest.md"] == {
        "file": "knowledge/local-llm/youtube-digest.md", "lines_checked": 1,
        "flagged": 0, "hits": []}


def test_an_injected_rubric_sentence_survives_quiet_and_names_its_file_and_line(
        tmp_path, capsys):
    """Clause 4 of #2039: the leak reports ITSELF, by path and line number.

    The injected sentence is the shape #2011's blind spot was made of — the shipped
    guard flags it, `#2011`'s hand-written acceptance grep matches none of its four
    phrases (that half is pinned in `tests/test_intel_pipeline_body.py`, where the
    grep is transcribed). It goes into a copy of the vault, so the clean fixture is
    still the control: the untouched copy runs `--quiet`, exits 0 and says nothing,
    while the dirty copy exits 2 and prints the digest's vault-relative path with the
    line number the sentence actually sits on — the derived `lineno` is read back out
    of the file, not pasted in, so an off-by-one in the sweep's enumeration fails here.
    """
    clean_vault, pipeline = _build(tmp_path / "clean")
    dirty_root = tmp_path / "dirty"
    shutil.copytree(tmp_path / "clean", dirty_root)
    target = _digest(dirty_root / "vault", "robotics")
    target.write_text(target.read_text() + f"\n{RUBRIC_SENTENCE}\n", encoding="utf-8")
    lineno = next(i for i, line in enumerate(target.read_text().splitlines(), 1)
                  if RUBRIC_SENTENCE in line)

    assert _run(clean_vault, pipeline, tmp_path / "out-clean", "--quiet") == 0
    assert capsys.readouterr().out == ""

    dirty_out_dir = tmp_path / "out-dirty"
    shutil.copytree(tmp_path / "out-clean", dirty_out_dir)
    # The dirty run is the NEXT UTC date's, not a second run on 2026-09-20: #1576's
    # one-row-per-day rule means a same-date re-run writes no row at all, and reading
    # the directory back would then hand this node the clean row above. Its base is
    # that clean row, and no shape metric differs between the two copies, so the only
    # thing that can make it exit non-zero is the digest.
    code = _run(dirty_root / "vault", dirty_root / "pipeline", dirty_out_dir, "--quiet",
                now=NEXT_DAY)
    out = capsys.readouterr().out
    assert code == 2, out
    line = next(l for l in out.splitlines() if "FLAGGED" in l)
    assert "knowledge/robotics/youtube-digest.md" in line, line
    assert f":{lineno} " in line, (lineno, line)
    assert RUBRIC_SENTENCE in line, line
    # Quiet mode prints the finding and nothing else: the ten per-digest lines are the
    # series a person trends, not something to copy into a report every night.
    assert not [l for l in out.splitlines() if l.startswith("interest_profile [")], out

    row = next(p for p in dirty_out_dir.glob("corpus-shape-*.json")
               if cs._row_date(p) == date(2026, 9, 21))
    report = json.loads(row.read_text())
    assert report["interest_profile"]["total_flagged"] == 1
    assert _digest_rows(report)["knowledge/robotics/youtube-digest.md"]["flagged"] == 1
    # The other two digests still read 0, so the finding names one file rather than
    # turning the whole corpus red and sending the reader hunting.
    assert _digest_rows(report)["knowledge/ai-llms/youtube-digest.md"]["flagged"] == 0


def test_the_digest_count_is_the_guard_s_verdict_and_not_a_phrase_pattern(
        tmp_path, monkeypatch, capsys):
    """Clause 5's other half: no phrase list of this script's own is in the loop.

    Both halves are needed, because either one alone survives a hand-written pattern
    quietly added alongside the guard. Flipping the guard to a stub that flags on
    `"point cloud"` — a description of a video, which the shipped guard would never
    flag — moves the count to exactly the digest carrying those words, so the number
    printed is whatever the callable says. Flipping it to a stub that flags nothing
    takes the injected rubric sentence back to 0 while the sentence is still on the
    page, so nothing else is counting phrases.
    """
    vault, pipeline = _build(tmp_path)
    shipped = cs._classify_interest_profile        # the real guard, before any patch
    monkeypatch.setattr(cs, "_classify_interest_profile",
                        lambda text: "point cloud" in text)
    assert _run(vault, pipeline, tmp_path / "out-a") == 2
    out = capsys.readouterr().out
    report = json.loads(next((tmp_path / "out-a").glob("*.json")).read_text())
    rows = _digest_rows(report)
    assert rows["knowledge/robotics/youtube-digest.md"]["flagged"] == 1, out
    assert rows["knowledge/ai-llms/youtube-digest.md"]["flagged"] == 0
    assert rows["knowledge/local-llm/youtube-digest.md"]["flagged"] == 0

    dirty = _digest(vault, "ai-llms")
    dirty.write_text(dirty.read_text() + f"\n{RUBRIC_SENTENCE}\n", encoding="utf-8")
    monkeypatch.setattr(cs, "_classify_interest_profile", lambda text: False)
    assert _run(vault, pipeline, tmp_path / "out-b", "--quiet") == 0
    assert capsys.readouterr().out == ""
    assert RUBRIC_SENTENCE in dirty.read_text()      # the sentence is still published
    # …and the guard the script really ships does flag it, so the 0 above is the stub
    # and not a corpus that happened to be clean. `shipped` is the function object
    # from before the patch, which is the one that reaches intel_pipeline.body.
    assert shipped(RUBRIC_SENTENCE) is True


# --- the entry point's own fallbacks, which is where a real leak hid (#2039) -----

def _inject_into_a_digest(vault: Path) -> int:
    """Append the rubric sentence to a copy of the live robotics digest; give back its line."""
    src = Path.home() / "obsidian" / "knowledge" / "robotics" / "youtube-digest.md"
    dst = vault / "knowledge" / "robotics" / "youtube-digest.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8").rstrip("\n")
    dst.write_text(text + "\n\n" + RUBRIC_SENTENCE + "\n", encoding="utf-8")
    return len(dst.read_text().splitlines())


def test_the_run_finds_the_vault_and_the_data_root_from_the_environment(
        tmp_path, monkeypatch, capsys):
    """A redirected vault must redirect the real command, not just one with `--vault`.

    Measured, not imagined: the first time the shipped CLI was run against a digest
    with a rubric sentence appended, it printed nothing and exited 0. `--vault` and
    `--pipeline` fall back to `app.paths`, whose `VAULT_ROOT` is a plain
    `Path.home() / "obsidian"` — so `LLOYD_OBSIDIAN_VAULT`, the variable
    `tests/board_presence.py` and the rest of the repo honour, was ignored at this
    entry point, and the check reported "clean" about a corpus it had not opened. The
    option in the other tests here hid that; this node runs the command a person
    actually types — no roots named — and so is the only node that exercises
    `_vault_default`/`_pipeline_default`.

    `LLOYD_DATA` points at an empty pipeline root rather than the machine's, so the
    four shape corpora are empty and the only thing that can move the exit code is the
    digest. Same argument in reverse for the clean half: exit 0 with the sentence
    absent proves the 0 is the corpus and not a broken reader.
    """
    empty_pipeline = tmp_path / "empty-pipeline"
    empty_pipeline.mkdir()
    monkeypatch.setenv("LLOYD_DATA", str(empty_pipeline))

    clean = tmp_path / "vault-clean"
    src = Path.home() / "obsidian" / "knowledge" / "robotics" / "youtube-digest.md"
    (clean / "knowledge" / "robotics").mkdir(parents=True)
    (clean / "knowledge" / "robotics" / "youtube-digest.md").write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("LLOYD_OBSIDIAN_VAULT", str(clean))
    assert cs.main(["--quiet", "--out-dir", str(tmp_path / "out-clean")]) == 0
    assert capsys.readouterr().out == ""

    dirty = tmp_path / "vault-dirty"
    lineno = _inject_into_a_digest(dirty)
    monkeypatch.setenv("LLOYD_OBSIDIAN_VAULT", str(dirty))
    assert cs.main(["--quiet", "--out-dir", str(tmp_path / "out-dirty")]) == 2, (
        "the command a person types reads a vault it was not told about: "
        "`_vault_default` is ignoring LLOYD_OBSIDIAN_VAULT again")
    out = capsys.readouterr().out
    assert "knowledge/robotics/youtube-digest.md" in out, out
    assert f":{lineno} " in out, (lineno, out)
