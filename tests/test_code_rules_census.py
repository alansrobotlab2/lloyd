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
