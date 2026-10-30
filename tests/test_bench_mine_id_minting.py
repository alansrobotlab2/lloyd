"""Who mints a bench task's number (#3531).

The mining prompt used to show `id: bench_XXX_<slug>` and leave the number to the
turn. The number a turn can read lives in `~/obsidian/lloyd/bench/` and advances only
when a human PROMOTES a staged candidate — and
`grep -l 'review_status: promoted' ~/obsidian/lloyd/bench/*.md` returns no file, so it
has never advanced once. Every turn since `bench_028` therefore answered `bench_029`
correctly and uselessly: as of 2026-10-29 the pile held 542 notes and 324 of them
declared `bench_029_*` (next `bench_018` ×59, `bench_024` ×42, `bench_020` ×30), against
a live corpus of 28 files topping out at `bench_028`. 39 ids were each repeated by more
than one note, covering 134 notes.

These nodes pin the mint: the number comes from code, over the live corpus AND the
staging pile together, and the same string lands in the note's task block and in its
recorded `calibration.task_id`.

Read the redirect in `_redir` before changing anything here. The mint reads the corpus
through `_BENCH_DIR_OVERRIDE` and the pile through `_common.STAGING_ROOT`; a node that
redirected only the pile would mint against the live 324-`bench_029` pile while
asserting against an empty one, and every number below would be wrong by 300-odd.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import workers.sources.bench_mine as bench_mine
import workers.sources._common as common
from app.harness import bench_authoring
from workers.queue import QueueItem

#: The number the fixture's live corpus tops out at. 28 is not arbitrary: it is the
#: highest id the real corpus holds, so `bench_029` here is the same number 324 real
#: notes claimed, and a mint that produced 30 would be visibly wrong to anyone who has
#: read the item.
TOP_LIVE = 28


def _item() -> QueueItem:
    """A claimed queue item, built the way every other workers test builds one."""
    return QueueItem(id=1, source="bench-mine", kind="k", priority=50,
                     payload={"run_id": "run_24_20261029_000000"},
                     dedup_key=None, state="running", attempts=1, enqueued_at="",
                     claimed_at=None, claimed_by=None, completed_at=None, error=None)


class _Trials:
    """`_bench_composites` replaced, recording the id of every task it was handed.

    `calibrate_candidate` binds its `trials` default from this module at CALL time
    (`trials = trials or _bench_composites`), so patching the module attribute IS the
    calibration's own seam — nothing is being measured through a fake beside the real
    one. The scores sit mid-band, so a calibration that runs reports `ok`.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def __call__(self, tasks, *, model: str = "") -> list[float]:
        self.seen.extend(str(t.get("id")) for t in tasks)
        return [0.4] * bench_mine.CALIBRATION_RUNS


def _redir(tmp_path, monkeypatch, *, live_top: int = TOP_LIVE, pile_ids=()) -> _Trials:
    """Point the mint at a tmp corpus and a tmp pile; stub the GPU and the watermark.

    Two redirections, both required: `_BENCH_DIR_OVERRIDE` for the corpus the number is
    counted against, `_common.STAGING_ROOT` for the pile it is counted against AND
    written into. The corpus files carry a real front matter `id:` because
    `load_bench_tasks` keys on the declared id, not the file name.
    """
    staging = tmp_path / "pending-research"
    monkeypatch.setattr(common, "STAGING_ROOT", staging)
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir(exist_ok=True)
    for n in range(1, live_top + 1):
        (bench_dir / f"bench_{n:03d}_seeded{n}.md").write_text(
            f"---\nid: bench_{n:03d}_seeded{n}\n---\n\nseeded live task\n",
            encoding="utf-8")
    pile = staging / bench_mine.NAME
    pile.mkdir(parents=True, exist_ok=True)
    for pid in pile_ids:
        (pile / f"pre-staged-{pid}.md").write_text(
            "---\nsource: bench-mine\nreview_status: pending\n---\n\n"
            f"---\nid: {pid}\nprompt: x\n---\n\nalready staged\n", encoding="utf-8")
    monkeypatch.setattr(bench_mine, "_BENCH_DIR_OVERRIDE", bench_dir)
    monkeypatch.setattr(bench_mine, "_mark_done", lambda *a, **k: None)
    trials = _Trials()
    monkeypatch.setattr(bench_mine, "_bench_composites", trials)
    return trials


def _task_text(task_id: str | None) -> str:
    """A candidate as a mining turn writes one: fenceless front matter plus body.

    `objective_checks` and `prompt` are present because `_rejection_reason` refuses a
    candidate without them, and a node that staged a rejected candidate would assert
    about a note that was never written.
    """
    fm = f"id: {task_id}\n" if task_id is not None else ""
    return ("---\n" + fm
            + "category: synthetic\n"
              "objective: the check\n"
              "max_tool_calls: 4\n"
              "edge_direction: add\n"
              "prompt: do the thing\n"
              "objective_checks:\n"
              "  - type: contains\n"
              "    value: the thing\n"
              "safety_critical: false\n"
            + "---\n\nSuccess is the thing happening.\n")


async def _stage(tmp_path, monkeypatch, task_id, *, slug="mined-from-run-24",
                 live_top: int = TOP_LIVE, pile_ids=(), text: str | None = None):
    """Stage one candidate end-to-end; returns (note, trials-that-ran)."""
    trials = _redir(tmp_path, monkeypatch, live_top=live_top, pile_ids=pile_ids)
    turn = SimpleNamespace(text=text if text is not None else _task_text(task_id),
                           ok=True, model="m", stop_reason="end_turn", num_turns=2)
    result = await bench_mine._stage_and_calibrate(
        _item(), turn, slug=slug, rationale="mined from a failed run",
        source_refs=["run_24_20261029_000000"], source_label="run 24")
    assert result["status"] == "success", (
        "staging did not happen at all, so an id assertion below would be reading an "
        f"empty pile rather than a minted id: {result['summary'][:220]}")
    return trials


def _notes(tmp_path) -> list[Path]:
    """Notes THIS test staged, excluding notes it pre-seeded into the pile.

    `_redir` plants pile entries named `pre-staged-*.md`; a staged note is named
    `{HHMMSS}-{slug}.md`, so the leading digit is what separates a freshly mined note
    from the pile the mint was told to count.
    """
    pile = tmp_path / "pending-research" / bench_mine.NAME
    return sorted(p for p in pile.rglob("*.md") if not p.name.startswith("pre-staged-"))


def _only_note(tmp_path) -> Path:
    notes = _notes(tmp_path)
    assert len(notes) == 1, f"expected exactly one staged note, found {len(notes)}"
    return notes[0]


def _task_block_id(note: Path) -> str:
    """The id the note's OWN TASK BLOCK declares — the second front matter in the file.

    A staged note is two documents, and the envelope is written from the minted id too,
    so reading the file's first `id:` would pass on a note whose task block still
    carried the turn's stale number. `bench_authoring.frontmatter_id`, the write rail's
    own reader, deliberately reads that FIRST block, which is why this helper exists
    separately rather than being the same call.
    """
    blocks = re.findall(r"^---\n(.*?)^---\n", note.read_text(encoding="utf-8"),
                        re.MULTILINE | re.DOTALL)
    for b in blocks:
        try:
            fm = yaml.safe_load(b) or {}
        except Exception:
            continue
        if isinstance(fm, dict) and fm.get("id") and "calibration" not in fm:
            return str(fm["id"])
    raise AssertionError(f"no task block declaring an id in {note.name}")


def _calibration_task_id(note: Path) -> str | None:
    """The `calibration.task_id` in the note's envelope front matter."""
    blocks = re.findall(r"^---\n(.*?)^---\n", note.read_text(encoding="utf-8"),
                        re.MULTILINE | re.DOTALL)
    for b in blocks:
        try:
            fm = yaml.safe_load(b) or {}
        except Exception:
            continue
        if isinstance(fm, dict) and isinstance(fm.get("calibration"), dict):
            return fm["calibration"].get("task_id")
    raise AssertionError(f"no calibration block in {note.name}")


# --- clause 2: the number comes from the corpus plus the pile, not from the turn -----

async def test_a_stale_number_is_replaced_by_one_past_the_live_top(tmp_path, monkeypatch):
    """Clause 2: `bench_001_old_name` over a corpus topping at 28 stages `bench_029_…`.

    This is the exact substitution the item names. The turn's number is not consulted
    and not merged — one past the highest live number, which is the arithmetic the
    prompt used to hand to a model that could only ever see 28.
    """
    await _stage(tmp_path, monkeypatch, "bench_001_old_name")
    note = _only_note(tmp_path)
    assert _task_block_id(note) == "bench_029_old_name", (
        "a turn that copies a number out of the corpus it was told to read — the "
        "normal thing to do, and what 118 of the 2026-09-28 candidates did — must land "
        f"on a free number, not on a live task's: {_task_block_id(note)!r}")
    assert "bench_001_old_name" not in note.read_text(encoding="utf-8"), (
        "the stale id survives somewhere in the note, so the file declares two ids and "
        "which one a reader gets depends on which block they open first")


async def test_the_number_advances_when_the_pile_already_holds_it(tmp_path, monkeypatch):
    """Clause 2: a corpus topping at 28 plus a pile holding 29 stages `bench_030_…`.

    Counting only the live corpus reproduces the defect inside the staging directory:
    the corpus advanced 2026-09-28 and the pile has taken 324 numbers since, so the
    pile is where the sequence actually lives.
    """
    await _stage(tmp_path, monkeypatch, "bench_001_old_name",
                 pile_ids=("bench_029_someone_else",))
    note = _only_note(tmp_path)
    assert _task_block_id(note) == "bench_030_old_name", (
        "the pile was not counted, which is the half of #3531 that produced 324 notes "
        f"all claiming bench_029: {_task_block_id(note)!r}")


async def test_a_letter_suffixed_id_holds_its_number_so_the_mint_steps_over_it(
        tmp_path, monkeypatch):
    """`bench_029b_zz` in the pile DOES hold 29, so the mint takes 30.

    Measured, not assumed: the underscore after the digits is optional in
    `_BENCH_ID_RE`, so `_bench_id_number("bench_029b_zz")` is 29 and the suffixed id
    counts as holding the number. The asymmetry against clause 4 is deliberate — an id
    of that shape must never be MINTED, but if one is already in the corpus or the pile
    it reserves its integer, because reserving can only cost a skipped number while
    failing to reserve mints a number an existing note already carries. It is stems
    rather than numbers that the promote route 409s on (`_bench_default_filename`
    returns the full `<id>.md`), so `bench_029b_zz` and a minted `bench_029_x` can
    coexist harmlessly.

    Asserted twice, because the unit and the staging path can disagree: what the number
    parser returns, and what the mint therefore hands out with that id in the pile.
    """
    assert bench_mine._bench_id_number("bench_029b_zz") == 29, (
        "a letter-suffixed id stopped holding its number, so the mint would hand out 29 "
        "to a pile that already has a bench_029b_* note in it")
    await _stage(tmp_path, monkeypatch, "old_name", pile_ids=("bench_029b_zz",))
    assert _task_block_id(_only_note(tmp_path)) == "bench_030_old_name", (
        "the mint did not step over a number a suffixed id holds: "
        f"{_task_block_id(_only_note(tmp_path))!r}")


# --- clause 3: two stagings, two ids, and each calibration names its own -------------

async def test_two_candidates_in_sequence_get_two_different_ids(tmp_path, monkeypatch):
    """Clause 3: the pile's ids stop repeating — the observable the item asked for.

    Run twice through the same tmp pile, so the second mint sees the first note. Under
    the old prompt both turns were asked "what number comes next?" with the same corpus
    in view and both answered 29; that answer was correct and useless.
    """
    staging = tmp_path / "pending-research"
    monkeypatch.setattr(common, "STAGING_ROOT", staging)
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    for n in range(1, TOP_LIVE + 1):
        (bench_dir / f"bench_{n:03d}_seeded{n}.md").write_text(
            f"---\nid: bench_{n:03d}_seeded{n}\n---\n\nseeded\n", encoding="utf-8")
    (staging / bench_mine.NAME).mkdir(parents=True)
    monkeypatch.setattr(bench_mine, "_BENCH_DIR_OVERRIDE", bench_dir)
    monkeypatch.setattr(bench_mine, "_mark_done", lambda *a, **k: None)
    trials = _Trials()
    monkeypatch.setattr(bench_mine, "_bench_composites", trials)

    for idea in ("first_idea", "second_idea"):
        # Both turns propose 29, because both read the same corpus and 29 is the right
        # answer to the question the old prompt asked. The ideas differ, which is the
        # realistic case: 303 distinct ideas, one number.
        turn = SimpleNamespace(text=_task_text(f"bench_029_{idea}"), ok=True, model="m",
                               stop_reason="end_turn", num_turns=2)
        out = await bench_mine._stage_and_calibrate(
            _item(), turn, slug=idea, rationale="r", source_refs=["run_24"],
            source_label=idea)
        assert out["status"] == "success", out["summary"][:200]

    notes = _notes(tmp_path)
    assert len(notes) == 2, f"two stagings must leave two notes, found {len(notes)}"
    ids = [_task_block_id(n) for n in notes]
    assert ids == ["bench_029_first_idea", "bench_030_second_idea"], (
        f"the second staging took a number the pile already held, which is #3531: {ids}")
    for note in notes:
        assert _calibration_task_id(note) == _task_block_id(note), (
            f"{note.name} records calibration.task_id "
            f"{_calibration_task_id(note)!r} but declares {_task_block_id(note)!r}: the "
            "mint moved the body and not the dict handed to calibrate_candidate, which "
            "is #1710 reopened — a mean attributed to a task that is not this one")
    # One staging spends CALIBRATION_RUNS trials on the SAME task dict, so what has to
    # hold is the sequence of DISTINCT ids, in order, and that every trial of a note ran
    # against that note's own id.
    distinct = list(dict.fromkeys(trials.seen))
    assert distinct == ["bench_029_first_idea", "bench_030_second_idea"], (
        "the trials ran against ids other than the ones the notes declare, so each "
        f"recorded mean measures a document that was never staged: {distinct}")
    assert len(trials.seen) == 2 * bench_mine.CALIBRATION_RUNS, (
        f"expected {bench_mine.CALIBRATION_RUNS} trials per staged note: "
        f"{len(trials.seen)}")
    assert set(trials.seen[:bench_mine.CALIBRATION_RUNS]) == {"bench_029_first_idea"}, (
        "note one's trials were not all spent on note one's id")


async def test_a_candidate_written_with_the_previous_note_s_id_still_gets_its_own(
        tmp_path, monkeypatch):
    """Two turns both answering `bench_029_repeated` cannot both keep it.

    The realistic version of the defect: every turn reads the same corpus and proposes
    the same next number. The ids differ here because the mint re-numbers the second
    one; the slugs are the same on purpose, since a slug is not the identity that was
    colliding.
    """
    staging = tmp_path / "pending-research"
    monkeypatch.setattr(common, "STAGING_ROOT", staging)
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    for n in range(1, TOP_LIVE + 1):
        (bench_dir / f"bench_{n:03d}_seeded{n}.md").write_text(
            f"---\nid: bench_{n:03d}_seeded{n}\n---\n\nseeded\n", encoding="utf-8")
    (staging / bench_mine.NAME).mkdir(parents=True)
    monkeypatch.setattr(bench_mine, "_BENCH_DIR_OVERRIDE", bench_dir)
    monkeypatch.setattr(bench_mine, "_mark_done", lambda *a, **k: None)
    monkeypatch.setattr(bench_mine, "_bench_composites", _Trials())
    # Two different mined runs, so two different slugs and two different note paths:
    # `write_staging_note` names a note `{HHMMSS}-{slug}.md`, so identical slugs in the
    # same second would overwrite each other and this node would find one file. What is
    # held IDENTICAL here is what actually collided — the id each turn proposed.
    for n in (1, 2):
        turn = SimpleNamespace(text=_task_text("bench_029_repeated"), ok=True, model="m",
                               stop_reason="end_turn", num_turns=2)
        assert (await bench_mine._stage_and_calibrate(
            _item(), turn, slug=f"repeated_run_{n}", rationale="r",
            source_refs=["run_24"], source_label=f"run {n}"))["status"] == "success"
    ids = sorted(_task_block_id(n) for n in _notes(tmp_path))
    assert len(set(ids)) == 2, (
        f"one id, two notes — the pile shape that made 134 notes share 39 ids: {ids}")
    assert all(i.startswith("bench_0") for i in ids), ids


# --- clause 4: whatever the turn wrote, the note ends up a bench task file name ------

#: The junk shapes, each with the slug the caller passes as a fallback. `None` is the
#: front matter with no `id:` line at all, which `_candidate_frontmatter` hands over as a
#: missing key rather than an empty one — a different code path from `id:` with a blank
#: value, so both are listed.
JUNK_IDS = {
    "stale-number": "bench_001_old_name",
    "letter-suffixed": "bench_006b_recall_chain",
    "non-numeric-string": "not-a-bench-id-at-all",
    "number-with-empty-slug": "bench_007",
    "spaces-and-capitals": "Two Tasks With Spaces",
    "blank-value": "",
    "no-id-line-at-all": None,
}


@pytest.mark.parametrize("junk", sorted(JUNK_IDS))
async def test_any_id_the_turn_writes_leaves_a_bench_task_file_name(junk, tmp_path,
                                                                    monkeypatch):
    """Clause 4: the note's id satisfies the rail's own name test, whatever came in.

    Two consumers key on this shape and neither is asked to tolerate anything else:
    `app/routers/workers.py::_bench_default_filename` returns `<id>.md` only for an id
    matching its own `_ID_SAFE_RE`, and `app/harness/bench_authoring.BENCH_TASK_FILE_RE`
    is what the four-lane write rail matches a file on before it will grade or protect
    it. An id of `Two Tasks With Spaces` therefore stops being a bench task the moment a
    human clicks Promote: the destination name is rejected, and a file the rail cannot
    name is a file the rail cannot guard.
    """
    await _stage(tmp_path, monkeypatch, JUNK_IDS[junk])
    declared = _task_block_id(_only_note(tmp_path))
    assert bench_authoring.BENCH_TASK_FILE_RE.match(declared + ".md"), (
        f"{junk}: the staged note declares {declared!r}, which "
        "BENCH_TASK_FILE_RE does not match, so the write rail cannot name it and the "
        "promote route's default filename will not be <id>.md")
    assert declared.startswith("bench_029_"), (
        f"{junk}: the number is not one past the fixture corpus's top of 28: {declared!r}")
    assert _calibration_task_id(_only_note(tmp_path)) == declared, (
        f"{junk}: calibration and the note disagree about which task this is")


async def test_a_junk_id_keeps_the_turn_s_words_rather_than_the_fallback_slug(
        tmp_path, monkeypatch):
    """The normaliser fixes the NUMBER and keeps the WORDS.

    Falling back to the caller's slug (`mined-from-run-24`, `mined-from-ledger-…`) for
    every malformed id is the collision all over again in the slug half, and it is what
    the 2026-09-28 shape measured: 118 candidates, 113 of them
    `mined-from-bench-007-skill-invocation` (`architecture/workers-jobs.md` §Mining). A
    junk id is a formatting failure, not a loss of the idea.
    """
    await _stage(tmp_path, monkeypatch, "bench_007b_Two Tasks!")
    declared = _task_block_id(_only_note(tmp_path))
    assert "mined-from-run-24" not in declared, (
        "the turn's words were thrown away for the caller's fallback, so this note lands "
        f"beside every other `mined-from-…`: {declared!r}")
    # The NUMBER the turn wrote is deliberately gone — it is the thing being re-minted —
    # and `b` is the residue of the letter suffix. What must survive is the words.
    assert "two" in declared.lower() and "tasks" in declared.lower(), (
        f"the turn's words did not survive the normaliser: {declared!r}")
    assert bench_authoring.BENCH_TASK_FILE_RE.match(declared + ".md"), declared


async def test_the_fallback_slug_is_used_only_when_the_turn_gave_no_words_at_all(
        tmp_path, monkeypatch):
    """The caller's slug is the last resort, not the first.

    `bench_007` declares a number and no slug, and no id at all declares nothing; both
    have to land somewhere, and the name the caller derived from the mined run or ledger
    row is the only information left. It is prefixed `bench_<NNN>_` like every other id,
    so the note is still a bench task file.
    """
    for junk in ("number-with-empty-slug", "no-id-line-at-all", "blank-value"):
        staging = tmp_path / f"case-{junk}" / "pending-research"
        monkeypatch.setattr(common, "STAGING_ROOT", staging)
        bench_dir = tmp_path / f"case-{junk}" / "bench"
        bench_dir.mkdir(parents=True)
        for n in range(1, TOP_LIVE + 1):
            (bench_dir / f"bench_{n:03d}_seeded{n}.md").write_text(
                f"---\nid: bench_{n:03d}_seeded{n}\n---\n\nseeded\n", encoding="utf-8")
        (staging / bench_mine.NAME).mkdir(parents=True)
        monkeypatch.setattr(bench_mine, "_BENCH_DIR_OVERRIDE", bench_dir)
        monkeypatch.setattr(bench_mine, "_mark_done", lambda *a, **k: None)
        monkeypatch.setattr(bench_mine, "_bench_composites", _Trials())
        turn = SimpleNamespace(text=_task_text(JUNK_IDS[junk]), ok=True, model="m",
                               stop_reason="end_turn", num_turns=2)
        out = await bench_mine._stage_and_calibrate(
            _item(), turn, slug="mined-from-run-24", rationale="r",
            source_refs=["run_24"], source_label=junk)
        assert out["status"] == "success", out["summary"][:200]
        pile = staging / bench_mine.NAME
        note = next(p for p in pile.rglob("*.md") if not p.name.startswith("pre-staged-"))
        declared = _task_block_id(note)
        assert declared == "bench_029_mined-from-run-24", (
            f"{junk} produced {declared!r}; with no slug from the turn the caller's "
            "derived slug is the only name available")
        assert bench_authoring.BENCH_TASK_FILE_RE.match(declared + ".md"), declared


# --- what the mint refuses -----------------------------------------------------------

async def test_a_corpus_scan_that_fails_refuses_the_staging_rather_than_numbering_from_zero(
        tmp_path, monkeypatch):
    """An unreadable corpus is a refusal, not an empty corpus.

    `_bench_ids_in` in the promote route raises `RuntimeError` on exactly this condition,
    and the docstring there says why: a check that could not run has no verdict, and
    "no ids are in use" is the reading that lets a duplicate land. The same policy is
    needed here for a sharper reason — a mint that fell back to an empty corpus would
    hand out `bench_001_<slug>`, the number of a LIVE graded task, and staging is
    supposed to be the recoverable step.
    """
    _redir(tmp_path, monkeypatch)
    monkeypatch.setattr(bench_mine, "_live_bench_numbers",
                        lambda: (_ for _ in ()).throw(OSError("vault unreadable")))
    turn = SimpleNamespace(text=_task_text("bench_001_old_name"), ok=True, model="m",
                           stop_reason="end_turn", num_turns=2)
    # Raised, not returned: a raised exception is the only route out of a worker source
    # that reaches `WorkQueue.mark_failed`, and `mark_failed` REQUEUES under the queue's
    # attempts cap. An in-band `{"status": "failed"}` would be followed by `q.complete()`
    # at the pool's call site and would retire the candidate on a transient I/O fault; an
    # invented status would be recorded as SUCCESS (`workers/pool.py::_execute`
    # recognises success/skipped/failed and warns on anything else).
    with pytest.raises(OSError):
        await bench_mine._stage_and_calibrate(
            _item(), turn, slug="mined-from-run-24", rationale="r",
            source_refs=["run_24"], source_label="unreadable corpus")
    assert _notes(tmp_path) == [], (
        "a note was written despite the failed scan, and its id would have been minted "
        "against a corpus the code had just failed to read")


async def test_a_rejected_candidate_never_reaches_the_mint(tmp_path, monkeypatch):
    """The gate runs BEFORE the number is asked for.

    With the number derived from what already exists rather than from a counter, a
    rejected candidate cannot in fact consume one — nothing advances unless a note is
    written — so the interesting claim is not "no number was burned" but "the mint was
    never consulted", which is what keeps a source whose turns keep getting rejected
    from tripping the mint's corpus scan on every rejected turn, and what makes the
    refusal path above unreachable for a candidate nobody would promote anyway.

    Asserted by counting calls on the mint, because the ordering is the whole clause and
    a node that only looked at the pile would pass on either order.
    """
    _redir(tmp_path, monkeypatch)
    calls: list = []
    real_mint = bench_mine._mint_bench_id

    def spying(candidate_id, fallback_slug):
        calls.append(candidate_id)
        return real_mint(candidate_id, fallback_slug)

    monkeypatch.setattr(bench_mine, "_mint_bench_id", spying)
    turn = SimpleNamespace(text="REJECT: this needs judgement, not a check", ok=True,
                           model="m", stop_reason="end_turn", num_turns=2)
    out = await bench_mine._stage_and_calibrate(
        _item(), turn, slug="mined-from-run-24", rationale="r", source_refs=["run_24"],
        source_label="a judgement")
    assert out["status"] != "success", out["summary"][:200]
    assert _notes(tmp_path) == [], "a rejected candidate was staged"
    assert calls == [], (
        "the mint ran for a candidate the reject gate was about to refuse, so every "
        f"rejected turn pays a corpus and pile scan: {calls}")


def test_the_three_spellings_of_the_bench_directory_agree():
    """The mint must count the corpus the promoted tasks actually land in.

    The path is spelled three times by three owners, and #3531 made all three load-
    bearing for one number: `BENCH_CORPUS_REL` is what the mint enumerates,
    `_DEFAULT_DEST["bench-mine"]` is where `/api/workers/pending/promote` writes the file
    (resolved against the same `VAULT_ROOT` by `_safe_vault_dest`), and
    `autoresearch.bench_dir` in config.yaml is what `load_bench_tasks`' caller passes for
    every eval and autoresearch read. They agree today; nothing stopped one moving.

    A moved mint-side path is the silent kind of breakage: the number is still produced,
    just from the wrong corpus, so 303 notes-worth of collision returns while every test
    that pins an id keeps passing. Asserted against the literal config file rather than
    the parsed config object so a loader defaulting a missing key cannot hide the drift.
    """
    import app.paths
    from app.routers import workers as workers_router
    import workers.sources.bench_mine as bm

    minted_against = (app.paths.VAULT_ROOT / bm.BENCH_CORPUS_REL)
    promoted_into = (app.paths.VAULT_ROOT
                     / workers_router._DEFAULT_DEST["bench-mine"])
    assert minted_against == promoted_into, (
        f"the mint numbers candidates against {minted_against} but a promoted candidate "
        f"lands in {promoted_into}: the mint would be counting a corpus the graded tasks "
        "never enter, which is #3531 reopened by a path change rather than by a prompt")

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml")
                         .read_text(encoding="utf-8"))
    configured = Path(cfg["autoresearch"]["bench_dir"]).expanduser()
    assert configured == minted_against.resolve() or configured == minted_against, (
        f"autoresearch.bench_dir points at {configured} while the mint counts "
        f"{minted_against}: the eval harness and the mint would disagree about which ids "
        "are in use, so a minted number could collide with a task the eval already grades")
