"""The weekly whole-tree census (#2810): what it measures, writes, files, and exits.

`scripts/maintenance/code_rules_census.py` is the caller `code_rules.run` has had a
parameter for since #2574 and never had: `whole_tree=True` over every tracked `.py`,
on a schedule, with a durable row. Everything the diff-scoped rung cannot do is what
is pinned here — that a shape predating every round still gets counted, that the row
says what its zero was measured over, that a day gets exactly one row, that a finding
becomes exactly one item, and that none of it can fail a round.

The trees are scratch git repos, not this one: a census of the live tree takes seconds
and its numbers belong to whoever edited it last, so a node that pinned them would be
a test that changes colour with unrelated work. The real rule file is used throughout,
because the thing under test is this program's reading of it.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.automod import backlog as B  # noqa: E402
from scripts.automod import code_rules as CR  # noqa: E402
from scripts.maintenance import code_rules_census as C  # noqa: E402

RULES_PATH = ROOT / "scripts" / "automod" / "code_rules.toml"

#: The shape the naive rule exists for, planted where the widened scope reaches. Two
#: details are load-bearing: the stamp is built by a `/` join (`_is_path_expr` only
#: counts an expression that *builds* a path), and the `now()` call sits inside it —
#: the variable hop (`ts = now()` then a name built from `ts`) is invisible to the
#: matcher and is #2810's other owed ruling. A fixture planted in either invisible
#: shape would pass this file while proving nothing about the whole-tree pass.
NAIVE_SITE = "scripts/memory/planted_stamp.py"
NAIVE_SRC = (
    "import datetime as dt\n"
    "from pathlib import Path\n"
    "\n"
    "RUN_ROOT = Path('/tmp')\n"
    'out = RUN_ROOT / f"rebuild-{dt.datetime.now():%Y%m%dT%H%M%SZ}"\n'
)
#: The same shape one directory outside the rule's scope. The census's denominator is
#: every tracked file, so this one must be counted in `of` and not in `covered`.
OUT_OF_SCOPE = "docs_tooling/not_in_scope.py"


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(tree), *args],
                          capture_output=True, text=True, check=True).stdout


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """A real git repo with `files` committed — `ls-files` needs a real one."""
    tree = tmp_path / "wt"
    tree.mkdir()
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "config", "user.email", "census@example.invalid")
    _git(tree, "config", "user.name", "census")
    for rel, text in files.items():
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "the tree under census")
    return tree


@pytest.fixture
def dirty(tmp_path) -> Path:
    """A tree with one naive site in scope, one out of scope, one suppression marker."""
    return _repo(tmp_path, {
        NAIVE_SITE: NAIVE_SRC,
        OUT_OF_SCOPE: NAIVE_SRC,
        "README.md": "census fixture\n",
        "app/unrelated.py": "import json  # noqa: BLE001 — the fixture's marker\n\n"
                            "X = json\n",
    })


@pytest.fixture
def clean(tmp_path) -> Path:
    return _repo(tmp_path, {"app/calm.py": "X = 1\n", "README.md": "clean\n"})


@pytest.fixture
def out_dir(tmp_path) -> Path:
    return tmp_path / "rows"


@pytest.fixture
def backlog_dir(tmp_path) -> Path:
    return tmp_path / "backlog"


def _argv(tree: Path, out_dir: Path, backlog_dir: Path, *extra: str) -> list[str]:
    return ["--tree", str(tree), "--out", str(out_dir),
            "--backlog-dir", str(backlog_dir), "--rules-path", str(RULES_PATH),
            "--quiet", *extra]


# ---------------------------------------------------------------------------
# The entrypoint's own standing: what it may import, and what it may exit.
# ---------------------------------------------------------------------------

def test_the_census_imports_nothing_the_weekly_driver_cannot_supply():
    """Stdlib and in-repo only, checked the way the runner's own import table is.

    This runs unattended on a schedule, and the lesson of the guardian's import rule
    one file over is that a dependency a scheduled job cannot import is a job that
    stops running and looks like a tree with nothing wrong with it. The backlog writer
    is the one deliberate exception and it is reached lazily, inside the filing call —
    which is why this node checks the roots and not the import sites.
    """
    tree = ast.parse((ROOT / "scripts" / "maintenance" / "code_rules_census.py")
                     .read_text(encoding="utf-8"))
    stdlib = set(sys.stdlib_module_names)
    local = {"scripts", "agent_mcp", "app", "workers"}
    roots: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            roots += [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                continue
            roots.append((n.module or "").split(".")[0])
    for r in roots:
        assert r in stdlib or r in local, (
            f"the census imports {r!r}, which is neither stdlib nor in-repo — a weekly "
            "job that cannot import is a week with no row and no error")


def test_a_tree_the_rules_cannot_be_read_from_is_not_reported_as_clean(clean, tmp_path,
                                                                       out_dir,
                                                                       backlog_dir):
    """Exit 2 and no row: `unevaluated` is the runner's own verdict, passed through.

    This is the whole #679 clause applied to a schedule. A weekly job that prints
    "0 hits" because its rule file went missing is worse than no job, and a row would
    make that fiction durable — next week's diff would be measuring a lie.
    """
    missing = tmp_path / "no-such-rules.toml"
    rc = C.main(_argv(clean, out_dir, backlog_dir, "--rules-path", str(missing)))
    assert rc == 2, f"an unreadable rule file returned {rc}, which reads as a clean census"
    assert not list(out_dir.glob("*.json")), (
        "a census that could not read the rules still wrote a row, so the series now "
        "carries a day that never happened")


# ---------------------------------------------------------------------------
# What the row measures.
# ---------------------------------------------------------------------------

def test_a_shape_that_predates_every_round_is_counted_and_named(dirty, out_dir,
                                                                backlog_dir):
    """The census's reason to exist: the diff rung cannot see a site no round touched.

    `code_rules.run` over a whole tree is the only way a naive stamp that landed
    before the rule was written ever surfaces, so the row has to name it by rule, path
    and line — and the same shape one directory outside the scope must stay out of the
    hit list, or the widened scope is not the boundary the rule file says it is.
    """
    row = C.census(dirty, RULES_PATH)
    assert row["status"] == CR.EVALUATED, row
    hits = {(h["rule"], h["path"], h["line"]) for h in row["hits"]}
    naive = [h for h in hits if h[0] == "naive-now-in-artifact-name"]
    assert len(naive) == 1, (
        f"the planted naive site was not counted exactly once: {sorted(hits)}")
    assert naive[0][1] == NAIVE_SITE, naive
    assert not any(h[1] == OUT_OF_SCOPE for h in hits), (
        "the out-of-scope copy was counted, so the rule's scope line is not being "
        "honoured by the whole-tree pass")


def test_the_row_says_what_each_rules_zero_was_measured_over(dirty, out_dir):
    """Per rule: covered, of, hits, allowlisted — plus the tree's suppression markers.

    A rule covering 1 file and a rule covering 1377 both report zero hits, and the row
    is the only place that difference survives into next month. This is the same
    reason `run` carries `totals.coverage` on the record at all; the census is where a
    reader actually goes looking for it.
    """
    row = C.census(dirty, RULES_PATH)
    shipped = {r.id for r in CR.load_rules(RULES_PATH)}
    assert {r["id"] for r in row["rules"]} == shipped, (
        "the row does not carry every rule the file ships, so a rule missing from the "
        "census is indistinguishable from a rule with no hits")
    tracked = len(C.tracked_python(dirty))
    for r in row["rules"]:
        assert r["of"] == tracked, (
            f"{r['id']}'s denominator is {r['of']} but the tree tracks {tracked} .py "
            "— the row would understate what the rule did not look at")
        assert 0 <= r["covered"] <= r["of"], r
    broad = [r for r in row["rules"] if r["id"] in
             {"opens-kg-store-outside-kg-store", "broad-except-swallow-same-line",
              "exports-lloyd-data-to-process-environ"}]
    assert len(broad) == 3 and all(r["covered"] == tracked for r in broad), (
        f"the three `**/*.py`-scoped rules must see every tracked file: {row['rules']}")
    assert {r["id"] for r in row["rules"] if r["covered"] == 0} == {"guardian-import-hygiene"}, (
        "the fixture tree has no `agent-services/guardian/` file, so exactly one rule "
        f"should report a zero coverage row: {row['rules']}")
    naive = {r["id"]: r for r in row["rules"]}["naive-now-in-artifact-name"]
    assert naive["covered"] == 2, (
        f"the planted tree tracks 3 .py and the widened scope reaches "
        f"scripts/memory/ and app/ (2), got {naive}")
    assert naive["hits"] == 1, naive
    assert row["totals"]["suppression_markers"] == 1, (
        "the fixture plants one `# noqa` and the row must show it counted — markers are "
        "reported, never honoured, and a count of none means the pass is not reading")


# ---------------------------------------------------------------------------
# One row per UTC day.
# ---------------------------------------------------------------------------

def test_a_clean_tree_writes_one_row_files_nothing_and_exits_zero(clean, out_dir,
                                                                  backlog_dir,
                                                                  monkeypatch):
    monkeypatch.setattr(C, "utc_now",
                        lambda: datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc))
    rc = C.main(_argv(clean, out_dir, backlog_dir))
    assert rc == 0, "a clean census must exit 0 — it files findings, it does not fail runs"
    rows = sorted(out_dir.glob("*.json"))
    assert [p.name for p in rows] == ["code-rules-census-2026-10-16.json"], rows
    row = json.loads(rows[0].read_text(encoding="utf-8"))
    assert row["totals"]["hits"] == 0
    assert row["filing"]["item"] is None and row["filing"]["skipped"] == "no hits", (
        "a census with nothing to report filed something anyway")
    assert not list(backlog_dir.glob("*.md")), "a clean tree filed a backlog item"


def test_a_second_run_inside_the_same_utc_day_adds_no_second_row(clean, out_dir,
                                                                 backlog_dir,
                                                                 monkeypatch):
    """The #1576 rule in a new series: a day's first row is the row that series keeps."""
    monkeypatch.setattr(C, "utc_now",
                        lambda: datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc))
    first = C.main(_argv(clean, out_dir, backlog_dir))
    body = out_dir.joinpath("code-rules-census-2026-10-16.json").read_text(encoding="utf-8")

    monkeypatch.setattr(C, "utc_now",
                        lambda: datetime(2026, 10, 16, 18, 30, tzinfo=timezone.utc))
    second = C.main(_argv(clean, out_dir, backlog_dir))
    assert (first, second) == (0, 0), (first, second)
    rows = sorted(out_dir.glob("*.json"))
    assert len(rows) == 1, f"a same-day re-run wrote {len(rows)} rows: {[p.name for p in rows]}"
    assert rows[0].read_text(encoding="utf-8") == body, (
        "the re-run rewrote the day's row, so the week that diffs against it is "
        "diffing a measurement taken at a different hour than the one scheduled")


# ---------------------------------------------------------------------------
# Filing: one item per finding, and a finding is never a failure.
# ---------------------------------------------------------------------------

def test_a_hit_files_exactly_one_item_naming_the_site_and_still_exits_zero(
        dirty, out_dir, backlog_dir, monkeypatch):
    """The whole point of the pass, across the one seam it crosses: the backlog writer.

    Asserted by reading the item back with the loader the board itself uses, because
    filing into a directory with a markdown blob that no reader can open would be a
    green test and a useless filing.
    """
    monkeypatch.setattr(C, "utc_now",
                        lambda: datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc))
    rc = C.main(_argv(dirty, out_dir, backlog_dir))
    assert rc == 0, (
        "a census that found something exited non-zero. A hit is a finding for a "
        "person or a future round; failing on it would make a weekly job red forever "
        "over code that predates the rule (#2574's observe-only split)")

    filed = sorted(backlog_dir.glob("*.md"))
    assert len(filed) == 1, f"expected exactly one filed item, got {[p.name for p in filed]}"
    item = B.load_item(filed[0])
    assert item is not None, f"the filed item cannot be loaded by the board's reader: {filed[0]}"
    assert item.board == C.BOARD and item.status == "draft", (
        f"#{item.id} is on board {item.board!r} with status {item.status!r}")
    assert "naive-now-in-artifact-name" in item.body, item.body
    assert NAIVE_SITE in item.body, item.body
    assert set(C.TAGS) <= set(item.tags), f"filed with tags {item.tags}"

    row = json.loads((out_dir / "code-rules-census-2026-10-16.json").read_text())
    assert row["filing"]["item"] == item.id, (
        "the row does not record the item it filed, so the next census cannot tell a "
        f"repeat from a new finding (row says {row['filing']})")


def test_the_same_sites_are_not_filed_again_by_next_weeks_census(dirty, out_dir,
                                                                 backlog_dir,
                                                                 monkeypatch):
    """Dedupe by signature, because a weekly job over an unfixed site is a standing spam risk.

    The second week still writes its row — the series is the point — but it files
    nothing, and the row says which item already carries the finding.
    """
    monkeypatch.setattr(C, "utc_now",
                        lambda: datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc))
    assert C.main(_argv(dirty, out_dir, backlog_dir)) == 0
    assert len(list(backlog_dir.glob("*.md"))) == 1

    monkeypatch.setattr(C, "utc_now",
                        lambda: datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc))
    assert C.main(_argv(dirty, out_dir, backlog_dir)) == 0
    assert len(list(backlog_dir.glob("*.md"))) == 1, (
        "a site nobody fixed filed a second item a week later; the signature check is "
        "the only thing standing between a weekly pass and one item per week forever")
    rows = sorted(out_dir.glob("*.json"))
    assert len(rows) == 2, f"the second week wrote no row of its own: {[p.name for p in rows]}"
    later = json.loads(rows[-1].read_text(encoding="utf-8"))
    first = json.loads(rows[0].read_text(encoding="utf-8"))
    assert later["signature"] == first["signature"]
    assert later["filing"]["item"] is None
    assert str(first["filing"]["item"]) in str(later["filing"]["skipped"]), later["filing"]


# ──────────────────────────────────────────────────────────────────────────────
# #3257 — a same-day re-run files nothing, and says out loud what the day filed.
#
# Filing used to run BEFORE `write_row`, so the O_EXCL guard that keeps one row per UTC
# day was reached only once the backlog item already existed. `prior_rows` skips today by
# design, so `_already_filed` could not see the filing the day had already made, and the
# duplicate was invisible to the very dedupe meant to stop it: a re-run on a hit tree left
# ONE row (recording only `{'hits': 1, 'item': 1}` — neither its own filing nor its own
# measurement) and TWO items for one signature. The board kept the orphan while next week
# read the row, saw the signature already filed, and skipped.
#
# Nothing above covered it. The same-day node uses the `clean` fixture, whose filing is
# skipped with "no hits" so it passes while filing twice would fail, and the dedupe node
# spans two dates, which `prior_rows` does see. Same day × hit tree is the gap.
#
# Splitting the claim from the record has one bad half, and the last node here holds it
# open: between `write_row` and `finalize_row` an item is on the board and no row knows
# it. That window is reported (`filing NOT RECORDED`) rather than papered over, because
# the consequence is next week filing a second item — the same defect, arrived at by
# failure instead of by ordering.
# ──────────────────────────────────────────────────────────────────────────────

MORNING = datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc)
#: Nine hours after the run above. Same UTC day, so the same filename: far enough apart
#: to be a retry after a red tick or a human checking whether the cleanup worked, which is
#: the pair of runs that filed twice.
AFTERNOON = datetime(2026, 10, 16, 18, 30, tzinfo=timezone.utc)
NEXT_WEEK = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)

#: The day's row, spelled the way the module spells it.
DAY_ROW = "code-rules-census-2026-10-16.json"


def _argv_loud(tree: Path, out_dir: Path, backlog_dir: Path, *extra: str) -> list[str]:
    """`_argv` without `--quiet`: these nodes read the printed report, which is half of
    what #3257 promises (a dropped filing is reported, never silent)."""
    return [a for a in _argv(tree, out_dir, backlog_dir, *extra) if a != "--quiet"]


def _rows(out_dir: Path) -> list[str]:
    return sorted(p.name for p in out_dir.glob("*.json"))


def _items(backlog_dir: Path) -> list[str]:
    return sorted(p.name for p in backlog_dir.glob("*.md"))


def _item_id(name: str) -> str:
    return name.split("-", 1)[0]


def test_a_same_day_rerun_on_a_hit_tree_files_no_second_item(dirty, out_dir, backlog_dir,
                                                            monkeypatch, capsys):
    """One UTC day, two runs, one signature: exactly one row AND exactly one item.

    The clause this file existed to acquire. At the item's base the day ended with one row
    and two items — `1-content-rule-census-found-1-site-s-the-diff-rung-c.md` and `2-…`,
    both from a single planted hit — while the row recorded only the first. Both counts are
    asserted because the row count already passed while the filing doubled: the same-day
    guard was working perfectly, and `file_hits` ran in front of it.
    """
    monkeypatch.setattr(C, "utc_now", lambda: MORNING)
    assert C.main(_argv_loud(dirty, out_dir, backlog_dir)) == 0
    monkeypatch.setattr(C, "utc_now", lambda: AFTERNOON)
    assert C.main(_argv_loud(dirty, out_dir, backlog_dir)) == 0, (
        "a re-run that could not claim the day exited non-zero. Its measurement is sound "
        "and its item is already filed, so the census ran: a second exit code would make a "
        "routine retry look like a broken instrument, which #679 forbids")

    assert _rows(out_dir) == [DAY_ROW], (
        f"the day holds {len(_rows(out_dir))} rows: {_rows(out_dir)}. The row count is not "
        f"what stopped the second item, and a second row would mean the first run never "
        f"claimed the day at all")
    items = _items(backlog_dir)
    assert len(items) == 1, (
        f"a same-day re-run filed {len(items)} items for one signature: {items}. The orphan "
        f"is the expensive half — next week reads the day's row, sees this signature "
        f"already filed, skips, and the board keeps the duplicate indefinitely")

    report = capsys.readouterr().out
    parts = report.split("census 2026-10-16: 1 hit(s)")
    assert len(parts) == 3, (
        f"expected a header from each of the two runs, read {len(parts) - 1}: {report!r}")
    afternoon = parts[-1]
    item_id = _item_id(items[0])
    assert f"filed backlog item #{item_id}" in afternoon, (
        "the re-run's report does not name the filing the day already carries, so whoever "
        "reads the tick sees a run that found a hit and filed nothing, and cannot tell a "
        f"deferral from a lost filing: {afternoon!r}")
    assert "filing deferred" in afternoon, (
        f"the re-run reported the hit without a deferral line: {afternoon!r}")
    assert "already existed" in afternoon, (
        "the re-run did not say it declined to write the row, which is the reason nothing "
        "was filed")

    # A deferral, not a loss: the signature is in the series (the run that wrote nothing
    # still left the day's row carrying it), and next week files nothing new.
    monkeypatch.setattr(C, "utc_now", lambda: NEXT_WEEK)
    assert C.main(_argv_loud(dirty, out_dir, backlog_dir)) == 0
    assert len(_rows(out_dir)) == 2, (
        "the same-day refusal left the signature appearing in NO row of the series, so the "
        "deferral is not up to a week of latency but a finding lost until something else "
        f"happens to notice: {_rows(out_dir)}")
    assert _items(backlog_dir) == items, (
        f"next week re-filed a signature the day had already filed: {_items(backlog_dir)}")


def test_the_day_is_claimed_before_filing_and_the_item_is_recorded_after(dirty, out_dir,
                                                                        backlog_dir,
                                                                        monkeypatch):
    """The order itself, pinned in the one place it is observable from inside the process.

    Every other node here reads the FILES the two writes leave behind, which is the right
    level for everything except this: a refactor that moved `write_row` back below
    `file_hits` would still pass them all, because a first run of a day files exactly one
    item either way. What the fix promises is that the claim exists before the backlog is
    touched — that is the only reason the guard can stop a filing at all — so this node
    stands in the seam and looks at the row directory while `file_hits` runs.

    It pins the half that the ordering made necessary: the run holding the claim writes the
    id it obtained into its own row, because a row that does not know about its filing is
    what makes next week file a second item.
    """
    seen: list[bool] = []
    real_file_hits = C.file_hits

    def spy(row, **kwargs):
        seen.append((out_dir / DAY_ROW).exists())
        return real_file_hits(row, **kwargs)

    monkeypatch.setattr(C, "utc_now", lambda: MORNING)
    monkeypatch.setattr(C, "file_hits", spy)
    assert C.main(_argv(dirty, out_dir, backlog_dir)) == 0

    assert seen == [True], (
        f"`file_hits` ran with the day's claim present={seen}: the claim must be taken "
        f"BEFORE the backlog is touched, or a same-day re-run files a second item again — "
        f"and this node is the only one in the suite that can see the difference")
    items = _items(backlog_dir)
    assert len(items) == 1, items

    stored = (out_dir / DAY_ROW).read_text(encoding="utf-8")
    row = json.loads(stored)
    assert str(row["filing"]["item"]) == _item_id(items[0]), (
        f"the row records filing.item={row['filing'].get('item')!r} while the board has "
        f"item {_item_id(items[0])}: an unrecorded filing is a duplicate waiting for next "
        f"week's dedupe to skip")
    assert "record_error" not in row["filing"], (
        f"the run could not write its own filing into the row it had just claimed: "
        f"{row['filing'].get('record_error')}")
    assert row["totals"]["hits"] == 1, (
        "recording the filing altered the measurement the row carries. The day's first "
        "number is the one next week diffs against (#1576)")
    assert row["signature"] == C.hit_signature(row["hits"]), (
        "the rewrite did not preserve the signature the row started with, so the dedupe "
        "keys on something other than what was measured")


def test_a_recorded_filing_that_could_not_be_written_is_reported(dirty, out_dir,
                                                                backlog_dir, monkeypatch,
                                                                capsys):
    """The window the fix opens, held open on purpose.

    Claiming the day before filing means there is a moment when an item exists and no row
    knows it. The cheap fix swallows that failure and looks clean; the honest one reports
    it, because the consequence of an unrecorded filing is next week filing a second item —
    this item's own defect, reached by an I/O failure instead of by ordering. This node
    makes the rename fail and asserts the run says so, keeps exit 0 (the census ran), and
    leaves the row readable with no filing in it.
    """
    real_replace = C.os.replace
    monkeypatch.setattr(C, "utc_now", lambda: MORNING)
    monkeypatch.setattr(C.os, "replace", lambda *a, **k: (_ for _ in ()).throw(
        OSError(28, "No space left on device")))
    try:
        assert C.main(_argv_loud(dirty, out_dir, backlog_dir)) == 0, (
            "a census that measured the tree and filed its item should not exit non-zero "
            "because annotating the row failed — exit 2 is reserved for a broken instrument")
    finally:
        monkeypatch.setattr(C.os, "replace", real_replace)

    report = capsys.readouterr().out
    assert "filing NOT RECORDED" in report, (
        f"the run could not record its filing and said nothing: {report!r}")
    assert "No space left on device" in report, (
        "the record failure was reported without its reason, which is the only thing a "
        "watcher needs to act on")
    assert len(_items(backlog_dir)) == 1, _items(backlog_dir)
    row = json.loads((out_dir / DAY_ROW).read_text(encoding="utf-8"))
    assert not (row.get("filing") or {}).get("item"), (
        "the row claims a filing that the failed rename could not have written")
    assert not list(out_dir.glob("*.tmp")), (
        f"a failed record left a temporary file in the series: {_rows(out_dir)}")


def test_a_day_row_recording_no_filing_defers_without_inventing_an_item(
        dirty, out_dir, backlog_dir, monkeypatch, capsys):
    """The deferral line never credits the board with an item the row does not name.

    Naming the day's filing is only half a rule, because the day's row can legitimately
    carry no item while this run has hits — a future path that writes a row before filing,
    or a hand-edited row like the one this node plants. Printing `#None` would put a
    fabricated citation into a report a human reads before deciding whether to act, so the
    empty branch has to say it found nothing on record.

    The planted row is a fixture, not the defect: what makes this node evidence rather than
    theatre is that the deferred signature still lands in the series, which is the property
    the fix had to keep while it stopped the filing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DAY_ROW).write_text(json.dumps({
        "date": "2026-10-16",
        "signature": "deadbeefdead",
        "filing": {"item": None, "hits": 0, "skipped": "filing was skipped by that run"},
    }) + "\n", encoding="utf-8")

    monkeypatch.setattr(C, "utc_now", lambda: AFTERNOON)
    assert C.main(_argv_loud(dirty, out_dir, backlog_dir)) == 0, (
        "a census that scanned a hit tree and deferred its filing exited non-zero. The "
        "instrument worked, so it exits 0 like every other run that ran")

    report = capsys.readouterr().out
    assert "filed backlog item #" not in report, (
        f"the report cites an item the day's row does not record: {report!r}")
    assert "records no filing" in report, (
        "the re-run found no filing to name and stayed silent about it, which is the exact "
        f"silence this item is about: {report!r}")
    assert not backlog_dir.exists() or _items(backlog_dir) == [], (
        f"a re-run invented an item the day had never recorded: {_items(backlog_dir)}")


def test_a_no_file_run_still_files_nothing_and_says_only_that(dirty, out_dir, backlog_dir,
                                                             monkeypatch, capsys):
    """`--no-file` is unchanged by the reordering, and its report line is the same words.

    The item's fifth check is that the flag's contract survived: the census prints its
    count line and exits 0. The skip reason matters too — `--no-file` means the operator
    asked not to file, which is a different statement from a guard refusing, and both
    appear in `filing skipped:` lines that a watcher learns to read apart.
    """
    monkeypatch.setattr(C, "utc_now", lambda: MORNING)
    assert C.main(_argv_loud(dirty, out_dir, backlog_dir, "--no-file")) == 0
    report = capsys.readouterr().out

    assert report.startswith("census 2026-10-16: 1 hit(s) over "), report
    assert "filing skipped: --no-file" in report, report
    assert not backlog_dir.exists() or _items(backlog_dir) == [], _items(backlog_dir)
    row = json.loads((out_dir / DAY_ROW).read_text(encoding="utf-8"))
    assert row["filing"] == {"item": None, "hits": 1, "skipped": "--no-file"}, row["filing"]


def test_the_module_says_a_same_day_rerun_files_nothing_and_what_it_prints():
    """The docstring used to state the opposite of the code, and that sentence is why the
    duplicate went unfixed: "this run writes nothing and files nothing" sat directly above
    a `file_hits` call that ran first, so the claim a reader trusted was the reason nobody
    went looking.

    The ban list is assembled from fragments so this file never carries the phrase it
    refuses; the second half of the node is what makes that safe, because a ban list whose
    strings can be edited without consequence stops banning — so the paragraph must also
    say the rule in the open.

    Three things it has to say, each for a reason: that a same-day re-run files nothing (the
    rule), that it names what the day already filed (the non-silent half), and that a
    signature arriving after the day's first row waits for the next pass (the cost, stated
    where the rule is so the next reader does not meet it as a surprise).
    """
    doc = C.__doc__ or ""
    for phrase in ["writes nothing and " + "files nothing",
                   "files nothing and " + "writes nothing"]:
        assert phrase not in doc, (
            f"the module docstring is back to claiming {phrase!r} while filing is gated on "
            f"the row claim")

    lower = doc.lower()
    assert "files nothing" in lower, "the replacement lost the rule it replaced"
    assert "already carries" in lower or "already filed" in lower, (
        "the paragraph no longer says what a rejected run prints instead — the non-silent "
        "half, which is what a watcher actually reads")
    assert "seven days" in lower or "next pass" in lower, (
        "the paragraph no longer states the cost of deferring: a signature that arrives "
        "after the day's row waits up to a week for the next pass, and that belongs beside "
        "the rule causing it rather than in an item that will be closed")


def test_the_exit_code_still_means_only_a_broken_instrument(tmp_path, monkeypatch,
                                                            capsys):
    """Exit 2 stays reserved for a broken instrument, and a real census still exits 0.

    #679 is why this file treats the exit code as load-bearing: a green nobody can
    interrogate is worse than no green, so 2 has meant "the instrument is broken" and never
    "the tree is dirty". Moving a filing behind a file write is the kind of edit that leaks
    a third 2 into the path, so both refusal sites are exercised — the unreadable-rules one
    already has a node above; the unevaluated one has never had one, and it is the site a
    future refactor is most likely to reorder past.

    Then the real tree: exit 0, and the count line's SHAPE, with its numbers read back out
    of the line itself. `scanned == tracked` is the property that makes `0 hit(s)` mean "the
    tree is clean" instead of "nothing was read" (#2810); an absolute count here would rot
    the moment a file lands, and the shape is what the item asked to keep.
    """
    empty = tmp_path / "no-rules"
    empty.mkdir()
    assert C.main(["--tree", str(ROOT), "--rules-path", str(empty / "absent.toml"),
                   "--out", str(tmp_path / "s1"), "--no-file"]) == 2, (
        "a tree whose rules file cannot be read stopped exiting 2, so a week with no rules "
        "would report a clean board")

    monkeypatch.setattr(C, "census", lambda tree, rules_path=None: {
        "status": "unevaluated",
        "reason": "the stub stands in for a rules load that refused"})
    assert C.main(["--tree", str(ROOT), "--out", str(tmp_path / "s2"),
                   "--no-file"]) == 2, (
        "an unevaluated census no longer exits 2: a week that measured nothing would "
        "report green, which is the failure the exit contract exists to prevent")
    monkeypatch.undo()

    out = tmp_path / "s3"
    assert C.main(["--tree", str(ROOT), "--out", str(out), "--no-file"]) == 0, (
        "a census that ran and reported now exits non-zero, so every tick would read red "
        "even with the count line intact")

    line = capsys.readouterr().out.splitlines()[0]
    head, _, tail = line.partition(": ")
    assert head == f"census {C.utc_now():%Y-%m-%d}", (
        f"the census line no longer opens with today's date: {line!r}")
    scanned = tail.split(" hit(s) over ")[1].split(" scanned file(s)")[0]
    tracked = tail.split(" scanned file(s) of ")[1].split(" tracked .py")[0]
    assert scanned == tracked and int(scanned) > 0, (
        f"`{scanned} scanned` of `{tracked} tracked` — a census whose denominator is not "
        f"what it read reports a clean tree it never looked at, which is #2810's whole "
        f"complaint: {line!r}")
    assert "allowlisted site(s)" in tail and "parse error(s)" in tail, line
    stored = json.loads((out / f"code-rules-census-{C.utc_now():%Y-%m-%d}.json").read_text(
        encoding="utf-8"))
    assert stored["status"] == C.code_rules.EVALUATED, stored.get("status")
    assert stored["filing"] == {"item": None, "hits": stored["totals"]["hits"],
                               "skipped": "--no-file"}, stored["filing"]
