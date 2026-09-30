"""#1906 clause 4 — a stray left in the live checkout is named at the round that wrote it.

On 2026-09-30 the live tree held three sets of untracked files no round admitted
writing: the review grader's `.t/05431072c3/r1873/` fixture (nine files), a
`agent-services/services/tts/.prior-1842b8cf.patch` a draft test left behind, and the
nightly probe's `eval/uptake/classifier-report.json`. The only notice was datawatch's
hourly "Runtime data is being written into the code tree" alert — eleven rows that
day, each naming the tree and none naming a round.

The shape is the one `workers/sources/arch_review.py:1722` and
`workers/sources/arch_review.py:1827-1829` already use for a worker turn: read
`git status --porcelain --untracked-files=all` before, read it after, and diff.
`app/live_strays.py` is that diff and nothing else. Its four call sites are the reason
it exists, named here fully qualified so no number in this file can be read against the
wrong file:

- `scripts/automod/round.py:191` — `round.start` writes the baseline into the round's
  own directory, and the `round_start` row at `scripts/automod/round.py:194-205` carries
  that set and its uncapped `live_untracked_count`;
- `scripts/automod/gate.py:1079` / `scripts/automod/gate.py:1101` — the gate's own
  `start_live_strays` / `end_live_strays`, bracketing the ladder at
  `scripts/automod/gate.py:1159` and `scripts/automod/gate.py:1219`, the second inside
  the `finally`, so a round that stops mid-ladder still records.

Two windows, because a round has two halves and only one of them is the gate: the
baseline the round recorded at open (`scripts/automod/round.py:191`) reaches back over
the implement turn that ran before the gate process existed, while the gate's own
snapshot (`scripts/automod/gate.py:1159`) covers only the ladder. `end_live_strays`
subtracts the round's baseline when it has one and falls back to its own snapshot when
it does not. A row that measured just the ladder says so
(`stray_window: gate_ladder_only`, `implement_turn_measured: false`) rather than reading
clean for a window it never looked at.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest  # noqa: E402

from app import live_strays  # noqa: E402
from scripts.automod import gate as G  # noqa: E402
from scripts.automod import state as S  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True,
                   capture_output=True, text=True)


@pytest.fixture
def live_repo(tmp_path):
    """A working tree with one tracked file, so porcelain has something to ignore."""
    repo = tmp_path / "live"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "promotions.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", path)
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── the diff itself ───────────────────────────────────────────────────
def test_a_path_that_appears_between_two_reads_is_named(live_repo):
    """The unit the whole clause rests on: `appeared` names the new path and nothing
    else, and `--untracked-files=all` is what makes a fixture directory readable as
    the nine files it is rather than as one line nobody can go and look at."""
    (live_repo / "already-there.txt").write_text("pre-existing\n", encoding="utf-8")
    baseline = live_strays.untracked(live_repo)
    assert baseline == {"already-there.txt"}

    (live_repo / ".t").mkdir()
    (live_repo / ".t" / "r1873").mkdir()
    for i in range(3):
        (live_repo / ".t" / "r1873" / f"f{i}.py").write_text("x\n", encoding="utf-8")
    now = live_strays.untracked(live_repo)

    appeared = live_strays.appeared(baseline, now)
    assert appeared == [".t/r1873/f0.py", ".t/r1873/f1.py", ".t/r1873/f2.py"], (
        f"the pre-existing stray must not be credited to this round, and the fixture "
        f"must appear file by file: {appeared}")


def test_an_unreadable_read_is_unknown_and_never_clean(live_repo):
    """`None` in, `None` out, from either side.

    This is the branch that decides whether the clause is a check or a rumour. A
    `git` that fails and returns an empty set would make every round look clean and
    every alert look unattributable forever, and the ledger row would read
    `stray_count: 0` with the same confidence as a real zero.
    """
    assert live_strays.appeared(None, {"a.py"}) is None
    assert live_strays.appeared({"a.py"}, None) is None
    assert live_strays.appeared(None, None) is None
    # …and a genuine empty diff is the empty list, a different value from None:
    assert live_strays.appeared({"a.py"}, {"a.py"}) == []


def test_untracked_parsing_survives_a_quoted_and_a_nested_path():
    """Porcelain quoting is a parser question with a text answer, so it is pinned
    here rather than by laying a working tree with a quote in a filename."""
    text = ("?? .t/05431072c3/r1873/old_relabel.py\n"
            "?? \"weird name.py\"\n"
            " M tracked.py\n"
            "A  staged-new.py\n")
    assert live_strays.parse_porcelain(text) == {
        ".t/05431072c3/r1873/old_relabel.py", "weird name.py"}


# ── the round's own record ────────────────────────────────────────────
def test_the_gate_records_what_appeared_during_the_round(live_repo, tmp_path, ledger,
                                                         monkeypatch):
    """A round that leaves a new untracked path has that path named in its own row."""
    (live_repo / "stale-left-over.patch").write_text("old\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    g = G.Gate("SM_STRAY_1", worktree, "deadbeef", live_root=live_repo)

    g.start_live_strays()
    (live_repo / ".t").mkdir()
    (live_repo / ".t" / "fixture.py").write_text("x\n", encoding="utf-8")
    g.end_live_strays()

    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["round_id"] == "SM_STRAY_1"
    assert row["stray_status"] == "recorded"
    assert row["strays"] == [".t/fixture.py"], (
        f"the row must name the path that appeared, not the pre-existing one: {row}")
    assert row["stray_count"] == 1
    # Called directly, with no `round_start` record behind it, this measures the ladder
    # and nothing else — and the row has to say so, or it reads as a clean round.
    assert row["stray_window"] == "gate_ladder_only", row
    assert row["implement_turn_measured"] is False, row


def test_a_stray_written_by_the_implement_turn_is_named_by_its_round(live_repo,
                                                                    tmp_path,
                                                                    ledger,
                                                                    monkeypatch):
    """The window the gate's own snapshot cannot see, and the one the 09-30 writer
    actually wrote in.

    `round.start` runs, the implement turn leaves `.t/r1873/` in the live tree, and only
    then does a gate process exist to look at it. Subtracting the gate's snapshot leaves
    that path inside BOTH of its reads, so it is unnamed and the round is reported clean
    — which is the review's residual on the first attempt here, and the difference
    between the round admitting its own stray and a datawatch alert naming the tree five
    hours later and nobody. The baseline that closes it is the set the round recorded
    when it opened, which is what `round.start` writes and `read_baseline` returns.
    """
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state")
    (live_repo / "pre-existing.txt").write_text("was here at open\n", encoding="utf-8")
    # The same call `round.start` makes, off the same read it makes.
    assert live_strays.write_baseline(
        "SM_STRAY_6", live_strays.untracked(live_repo)) is not None

    # The implement turn, before the gate exists.
    (live_repo / ".t").mkdir()
    (live_repo / ".t" / "r1873").mkdir()
    (live_repo / ".t" / "r1873" / "old_relabel.py").write_text("x\n", encoding="utf-8")

    worktree = tmp_path / "wt"
    worktree.mkdir()
    g = G.Gate("SM_STRAY_6", worktree, "deadbeef", live_root=live_repo)

    def _rung(name, fn):
        return True

    monkeypatch.setattr(g, "_serialized", lambda name, fn: fn)
    monkeypatch.setattr(g, "_rung", _rung)
    monkeypatch.setattr(g, "_start_review_prefetch", lambda: None)
    monkeypatch.setattr(g, "_discard_review_prefetch", lambda why: None)
    g.run()

    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["stray_status"] == "recorded", row
    assert row["strays"] == [".t/r1873/old_relabel.py"], (
        f"the path written before the gate opened must still be this round's: {row}")
    assert row["stray_window"] == "since_round_start", row
    assert row["implement_turn_measured"] is True, row
    assert "pre-existing.txt" not in row["strays"], row


def test_a_round_that_added_none_records_none(live_repo, tmp_path, ledger):
    """The good outcome is still a row. A clean run that writes nothing is
    indistinguishable from a check that never ran, and the owed measure is a count of
    per-round rows — silence would read as a broken instrument."""
    (live_repo / "pre-existing.txt").write_text("was already here\n", encoding="utf-8")
    g = G.Gate("SM_STRAY_2", tmp_path / "wt", "deadbeef", live_root=live_repo)
    g.start_live_strays()
    g.end_live_strays()

    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, rows
    assert rows[0]["stray_status"] == "recorded" and rows[0]["stray_count"] == 0, rows
    # Clean is only a clean ladder here: with no round record behind it the node's
    # second read cannot say anything about the implement turn, and the row must not
    # imply otherwise.
    assert rows[0]["stray_window"] == "gate_ladder_only", rows
    assert rows[0]["implement_turn_measured"] is False, rows


def test_a_baseline_written_at_open_says_the_round_added_nothing(live_repo, tmp_path,
                                                                ledger, monkeypatch):
    """The good outcome measured over the whole round, not just the ladder.

    Same tree, same two paths, one recorded baseline: nothing this row can call new.
    Without the baseline the identical run reports `unknown`, and the difference between
    those two rows is the difference between an instrument and a rumour.
    """
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state")
    (live_repo / "pre-existing.txt").write_text("was already here\n", encoding="utf-8")
    (live_repo / ".t").mkdir()
    (live_repo / ".t" / "fixture.py").write_text("x\n", encoding="utf-8")
    assert live_strays.write_baseline(
        "SM_STRAY_7", live_strays.untracked(live_repo)) is not None

    worktree = tmp_path / "wt"
    worktree.mkdir()
    g = G.Gate("SM_STRAY_7", worktree, "deadbeef", live_root=live_repo)

    def _rung(name, fn):
        return True

    monkeypatch.setattr(g, "_serialized", lambda name, fn: fn)
    monkeypatch.setattr(g, "_rung", _rung)
    monkeypatch.setattr(g, "_start_review_prefetch", lambda: None)
    monkeypatch.setattr(g, "_discard_review_prefetch", lambda why: None)
    g.run()

    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["stray_status"] == "recorded" and row["stray_count"] == 0, row
    assert row["stray_window"] == "since_round_start", row
    assert row["implement_turn_measured"] is True, row


def test_twenty_five_pre_existing_paths_are_not_twenty_of_this_rounds(live_repo,
                                                                     tmp_path,
                                                                     ledger,
                                                                     monkeypatch):
    """The cap on the ledger sample must not leak into the subtraction.

    `round_start` prints at most twenty untracked paths so a human can read the row; the
    baseline the gate subtracts is the whole set, kept in the round's own record. A gate
    working from the row's sample instead would name twenty paths as this round's writing
    on a tree that already held twenty-five, and every one of them would be a false
    accusation against the round that merely looked at the tree first.
    """
    for i in range(25):
        (live_repo / f"stray{i:02d}.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state")
    baseline = live_strays.untracked(live_repo)
    assert baseline is not None and len(baseline) == 25, baseline
    assert live_strays.write_baseline("SM_STRAY_8", baseline) is not None

    worktree = tmp_path / "wt"
    worktree.mkdir()
    g = G.Gate("SM_STRAY_8", worktree, "deadbeef", live_root=live_repo)

    def _rung(name, fn):
        if name == "static":
            (live_repo / ".t").mkdir()
            (live_repo / ".t" / "fixture.py").write_text("x\n", encoding="utf-8")
        return True

    monkeypatch.setattr(g, "_serialized", lambda name, fn: fn)
    monkeypatch.setattr(g, "_rung", _rung)
    monkeypatch.setattr(g, "_start_review_prefetch", lambda: None)
    monkeypatch.setattr(g, "_discard_review_prefetch", lambda why: None)
    g.run()

    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["strays"] == [".t/fixture.py"], (
        f"the twenty-five pre-existing paths must not be credited to this round: {row}")
    assert row["stray_count"] == 1, row


def test_a_missing_baseline_records_unknown_not_zero(tmp_path, live_repo, ledger):
    """A round opened before this shipped has no baseline in its row. Diffing that
    against "nothing" would credit it with every stray already in the tree; reporting
    it as clean would do the same thing more quietly and more convincingly."""
    g = G.Gate("SM_STRAY_3", tmp_path / "wt", "deadbeef", live_root=live_repo)
    # No start_live_strays(): the attribute does not exist, exactly as for a round
    # whose `round_start` row predates the key.
    (live_repo / "someone-elses-stray.txt").write_text("x\n", encoding="utf-8")
    g.end_live_strays()

    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, rows
    assert rows[0]["stray_status"] == "unknown", rows
    assert "strays" not in rows[0] and "stray_count" not in rows[0], (
        f"an unknown must not carry a number: {rows[0]}")


def test_the_gate_brackets_its_own_ladder_with_the_two_reads(live_repo, tmp_path,
                                                             ledger, monkeypatch):
    """`Gate.run` is what puts the reads around the ladder — not a caller that might
    remember to call them.

    Every rung is stubbed and the real `run()` drives them, because the failure this
    pins is not a wrong diff (the two nodes above cover that) but a correct pair of
    methods that is never called: a baseline taken after the first rung, or a record
    that is never written, both leave the round silent exactly as an absent check
    would. Two writes settle both ends. A path already in the tree when `run()` is
    called must NOT be named, which is what proves a baseline was taken at all; a path
    written by the FIRST rung must BE named, which proves the baseline predates even
    the first rung; and the row exists only because the `finally` ran after the LAST.
    """
    (live_repo / "pre-existing.txt").write_text("was here before run()\n",
                                                encoding="utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    g = G.Gate("SM_STRAY_4", worktree, "deadbeef", live_root=live_repo)

    seen: list[str] = []

    def _rung(name, fn):
        seen.append(name)
        if name == "preflight":
            (live_repo / "written-by-the-first-rung.txt").write_text(
                "the baseline must predate this\n", encoding="utf-8")
        elif name == "static":
            (live_repo / ".t").mkdir()
            (live_repo / ".t" / "fixture.py").write_text("x\n", encoding="utf-8")
        return True

    monkeypatch.setattr(g, "_serialized", lambda name, fn: fn)
    monkeypatch.setattr(g, "_rung", _rung)
    monkeypatch.setattr(g, "_start_review_prefetch", lambda: None)
    monkeypatch.setattr(g, "_discard_review_prefetch", lambda why: None)

    report = g.run()

    assert report.ok, report.rungs
    assert seen[:1] == ["preflight"] and seen[-1:] == ["drill"], seen
    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, f"run() must leave exactly one row: {rows}"
    row = rows[0]
    assert row["round_id"] == "SM_STRAY_4" and row["stray_status"] == "recorded", row
    assert row["strays"] == [".t/fixture.py", "written-by-the-first-rung.txt"], row
    assert "pre-existing.txt" not in row["strays"], (
        f"a path that predates the baseline is not this round's: {row}")
    # No `round_start` record exists for this round, so what it measured is the ladder.
    assert row["stray_window"] == "gate_ladder_only", row
    assert row["implement_turn_measured"] is False, row


def test_a_baseline_that_cannot_be_read_records_unknown(tmp_path, live_repo, ledger,
                                                       monkeypatch):
    """A baseline file that exists but does not parse is not a baseline of nothing.

    The other half of "a missing or unreadable baseline records unknown": the round's
    record was truncated by a crash, or written by a version whose schema this one does
    not recognise. A reader that fell back to an empty set there would name every stray
    in the tree as this round's writing — and `stray_count` would be the loudest wrong
    number in the ledger, because a real incident would finally have a count to cite.
    """
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state")
    (live_repo / "pre-existing.txt").write_text("x\n", encoding="utf-8")
    p = live_strays.baseline_path("SM_STRAY_9", tmp_path / "state")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"round_id": "SM_STRAY_9", "untracked": "not a list"}',
                 encoding="utf-8")
    assert live_strays.read_baseline("SM_STRAY_9", tmp_path / "state") is None

    worktree = tmp_path / "wt"
    worktree.mkdir()
    g = G.Gate("SM_STRAY_9", worktree, "deadbeef", live_root=live_repo)
    g.start_live_strays()
    g.end_live_strays()

    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, rows
    # The gate's own snapshot IS readable, so this row is a measured ladder — but it
    # says which window it measured, and the round's half of the answer stays unknown
    # rather than becoming a clean one.
    assert rows[0]["stray_window"] == "gate_ladder_only", rows
    assert rows[0]["implement_turn_measured"] is False, rows
    assert live_strays.read_baseline("SM_STRAY_9", tmp_path / "no-such-dir") is None


def test_the_round_start_row_carries_the_baseline(live_repo, tmp_path, monkeypatch):
    """`round_start` records the untracked set it opened with, so a later reader can
    tell a path this round wrote from one that was already there. This node is the READ
    SUCCEEDING; the read failing, and the empty baseline that would falsely credit the
    round, is `test_a_round_opened_against_an_unreadable_tree_lays_no_empty_baseline`."""
    from scripts.automod import round as R

    monkeypatch.setattr(R, "LIVE_ROOT", live_repo)
    monkeypatch.setattr(R.S, "require_enabled", lambda action, repo=None: None)
    for name in ("STATE_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(R.S, name, tmp_path / "state")
    monkeypatch.setattr(R.S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(R.W, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(R.W, "LIVE_ROOT", live_repo)
    (live_repo / "already.txt").write_text("x\n", encoding="utf-8")

    R.start("stray baseline", opened_by="cli")

    rows = _rows(tmp_path / "promotions.jsonl")
    start = [r for r in rows if r["event"] == "round_start"][0]
    assert start["live_untracked_recorded"] is True, start
    assert start["live_untracked"] == ["already.txt"], start
    # The row is the human-facing sample; the thing the gate subtracts is the file in the
    # round's own record, and the row has to say whether that write happened.
    assert start["live_baseline_recorded"] is True, start
    assert live_strays.read_baseline(start["round_id"],
                                     tmp_path / "state") == {"already.txt"}, start


def test_a_round_whose_baseline_cannot_be_written_says_so(live_repo, tmp_path,
                                                         monkeypatch):
    """`live_baseline_recorded` is a recorded FALSE, not an absent key.

    The tree read can succeed while the write into the round's directory fails, and a
    row that simply omitted the key would let a later reader infer "no baseline needed"
    — the same impersonation an empty list performs on an unreadable tree, one layer up.
    """
    from scripts.automod import round as R

    monkeypatch.setattr(R, "LIVE_ROOT", live_repo)
    monkeypatch.setattr(R.S, "require_enabled", lambda action, repo=None: None)
    for name in ("STATE_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(R.S, name, tmp_path / "state")
    monkeypatch.setattr(R.S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(R.W, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(R.W, "LIVE_ROOT", live_repo)
    (live_repo / "already.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(live_strays, "write_baseline",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    R.start("baseline write fails", opened_by="cli")

    rows = _rows(tmp_path / "promotions.jsonl")
    start = [r for r in rows if r["event"] == "round_start"][0]
    assert start["live_untracked_recorded"] is True, start
    assert start["live_baseline_recorded"] is False, start


def test_a_round_opened_against_an_unreadable_tree_lays_no_empty_baseline(
        live_repo, tmp_path, ledger, monkeypatch):
    """A failed tree READ at round open must produce no baseline, not an empty one.

    This is the branch the first review measured directly and no node reached: the only
    failure injected at `round.start` was a failed WRITE. `live_strays.untracked` returns
    None on a failed read and never raises, so the shipped
    `found = live_strays.untracked(LIVE_ROOT) or set()` turned the documented None into an
    empty set, recorded `live_untracked_recorded: true`, and wrote that empty set as this
    round's baseline — which the gate then subtracted, printing every path already in the
    tree as this round's writing under `stray_window: since_round_start`. A `stray_count`
    of "everything in the tree" on a round that wrote nothing is the loudest wrong number
    this ledger can carry, and it is what a `git status` that fails for one second (a
    locked index, a momentary fork failure) would have produced.

    Injected through the same fake the module documents: `untracked` returns
    `live_strays.UNREADABLE`, which is what it returns for a non-zero exit, and is what
    `test_an_unreadable_read_is_unknown_and_never_clean` pins for the reader itself. The
    fake fails only while the round opens, so the gate's own read below is real.

    Both halves of the seam are asserted, because the bug was a handoff: the row the round
    prints, and the row the GATE prints after reading (or not reading) that file.
    """
    from scripts.automod import round as R

    monkeypatch.setattr(R, "LIVE_ROOT", live_repo)
    monkeypatch.setattr(R.S, "require_enabled", lambda action, repo=None: None)
    for name in ("STATE_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(R.S, name, tmp_path / "state")
    monkeypatch.setattr(R.W, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(R.W, "LIVE_ROOT", live_repo)
    (live_repo / "already.txt").write_text("pre-existing\n", encoding="utf-8")

    real_untracked = live_strays.untracked
    unreadable = {"now": True}

    def _flaky_untracked(root, *a, **k):
        return live_strays.UNREADABLE if unreadable["now"] else real_untracked(root)

    monkeypatch.setattr(live_strays, "untracked", _flaky_untracked)
    unreadable["now"] = True
    R.start("tree unreadable at open", opened_by="cli")
    unreadable["now"] = False

    start = [r for r in _rows(ledger) if r["event"] == "round_start"][0]
    assert start["live_untracked_recorded"] is False, (
        f"a failed read cannot record a successful one: {start}")
    assert start["live_untracked"] == [], start
    assert start["live_untracked_count"] == -1, start
    assert start["live_baseline_recorded"] is False, (
        f"no baseline may be claimed when the tree could not be read: {start}")
    rid = start["round_id"]
    assert live_strays.read_baseline(rid, tmp_path / "state") is None, (
        "an empty baseline file is the false accusation: the gate subtracts it and "
        "credits this round with every path already in the tree")

    worktree = tmp_path / "wt"
    worktree.mkdir()
    g = G.Gate(rid, worktree, "deadbeef", live_root=live_repo)
    g.start_live_strays()
    g.end_live_strays()

    rows = [r for r in _rows(ledger) if r["event"] == "round_live_strays"]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["strays"] == [] and row["stray_count"] == 0, (
        f"a round that wrote nothing must not be credited with the pre-existing stray: "
        f"{row}")
    assert "already.txt" not in row.get("strays", []), row
    # And the row cannot read as though it had covered the implement turn, because the
    # round's own baseline does not exist for it to subtract.
    assert row["stray_window"] == "gate_ladder_only", row
    assert row["implement_turn_measured"] is False, row
