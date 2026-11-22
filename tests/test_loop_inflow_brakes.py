"""The loop's own inflow is bounded in code, and its growth is on the report.

2026-11-22 review: 112 of 160 open items had been filed by owed-check, a source
that stamped no tag and so was missing from every gauge; the share of new items
that only witness, record or pin an earlier item's claim had risen from ~4 % to
~27 %; every worker's spawn cap was a sentence in its prompt; and
`scorecard.jsonl` held two rows because nothing called `--record`. Pinned here:

* an owed-check follow-up carries `spawned-by-owed`, so expiry and the gauge see it;
* a follow-up that only witnesses the parent's claim is not filed — the debt
  stays on the parent, where `MAX_RECHECKS` bounds it;
* `backlog_write_task` refuses an unattended run's create past its cap, and
  never a person's;
* the scorecard measures tree growth, derived work and the tests rung, and
  records itself once a day.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as BL
from agent_mcp import backlog_similar as SIM
from scripts.automod import backlog as B, owed as O, scorecard as SC, state as S

NOW = 1_800_000_000.0
DAY = 86400.0


# ── the derived-name rule ────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "Commit #3820's premise witness into backlog/data",
    "Publish the 2026-11-19 promotion-halt witness extract of promotions.jsonl",
    "Record the #4242 carrier-guard widening ruling and its reopen trigger",
    "Pin that a prose citation never settles a human_clause (#4368 owed 3)",
    "Settle #3956's surviving 'still owed' mover line in the vault",
])
def test_a_name_that_only_witnesses_an_earlier_item_is_derived(name):
    assert B.is_derived_name(name)


@pytest.mark.parametrize("name", [
    "Desktop 1280px geometry pin still flakes under xdist-8",
    "Measure frame resolution cap and scene-vs-uniform sampling on real decks (#3983 owed)",
    "Record the deliberate eval/baselines hold in the retention sweep",   # cites no item
    "Run the cc11 ask-rate baseline on the live primary (single arm)",
    "",
])
def test_engineering_work_is_not_derived_even_when_it_cites_an_item(name):
    assert not B.is_derived_name(name)


# ── owed-check follow-ups ────────────────────────────────────────────────

@pytest.fixture
def owed_board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(O, "CITE_ROOTS", {"vault": tmp_path / "no-vault", "repo": tmp_path / "no-repo"})
    monkeypatch.setattr(O, "_cite_revision", lambda: "")
    return d


def _owing(d: Path, item_id: int, whats: list[str]):
    fm = {"status": "done", "priority": "medium", "board": "lloyd", "tags": ["backlog"],
          "created": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
    path = d / f"{item_id}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nBody.\n",
                    encoding="utf-8")
    O.add_owed(path, whats)
    return path, O.entries_of(B._split_frontmatter(path.read_text(encoding="utf-8"))[0])


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text(encoding="utf-8"))[0]


def test_a_filed_follow_up_carries_the_owed_spawn_tag(owed_board):
    p, entries = _owing(owed_board, 21, ["the retry path has no test"])
    out = O.apply_verdict(p, entries, [{
        "n": 1, "outcome": "work", "evidence": "x",
        "follow_up": {"name": "Add a test for the retry path",
                      "body": "Write the test that drives the retry path twice."}}], item_id=21)
    assert len(out["filed"]) == 1
    child = B.load_item(next(owed_board.glob(f"{out['filed'][0]}-*.md")))
    assert B.OWED_SPAWN_TAG in child.tags
    assert B.is_loop_spawned(child)            # so expiry and the gauge count it
    assert not B.is_quarantined(child)         # triage still judges it


def test_a_witness_follow_up_is_not_filed_and_the_debt_stays_on_the_parent(owed_board):
    p, entries = _owing(owed_board, 22, ["the quoted bytes have no history"])
    before = sorted(x.name for x in owed_board.glob("*.md"))
    out = O.apply_verdict(p, entries, [{
        "n": 1, "outcome": "work", "evidence": "x",
        "follow_up": {"name": "Commit #22's witness bytes to backlog/data",
                      "body": "Copy the store to backlog/data and commit it."}}], item_id=22)
    assert out["filed"] == []
    assert sorted(x.name for x in owed_board.glob("*.md")) == before
    still = O.entries_of(_fm(p))
    assert [e["what"] for e in still] == ["the quoted bytes have no history"]
    assert still[0].get("rechecks", 0) == 1     # counted, so MAX_RECHECKS bounds it


# ── the spawn cap at the write ───────────────────────────────────────────

@pytest.fixture
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    monkeypatch.setattr(SIM, "DEDUPE_LOG", tmp_path / "dedupe.jsonl")
    monkeypatch.setattr(SIM, "dedupe_config", lambda: dict(SIM.DEFAULTS))
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: [])
    monkeypatch.setattr(BL, "_spawn_counts", {})
    monkeypatch.setattr(BL, "_spawn_cap_for", lambda tag: 1)
    journal = tmp_path / "denials.jsonl"
    monkeypatch.setenv("LLOYD_DENIAL_JOURNAL", str(journal))
    return d, journal


def _as(monkeypatch, cls: str, sid: str):
    monkeypatch.setattr(BL, "_session_class", lambda: (cls, sid))


def _create(name: str, tags) -> dict:
    return json.loads(BL._handle_write({
        "name": name, "board": "lloyd", "tags": list(tags),
        "description": f"{name}: a finding with enough words to stand as its own item."}))


def test_an_unattended_runs_second_spawn_is_refused_and_journaled(board, monkeypatch):
    d, journal = board
    _as(monkeypatch, "background", "20261122_010101_worker_aaaa")
    first = _create("The retry path drops the last frame", ["spawned-by-triage"])
    second = _create("The upload queue never drains on a 503", ["spawned-by-triage"])
    assert first["success"] and first["created"]
    assert second["success"] is False
    assert second["error"].startswith("Tool call denied:")
    assert len(list(d.glob("*.md"))) == 1
    rows = [json.loads(x) for x in journal.read_text().splitlines()]
    assert [(r["guard"], r["tool"], r["label"]) for r in rows] == [
        ("spawn_cap", "backlog_write_task", "spawned-by-triage")]


def test_the_cap_is_per_session(board, monkeypatch):
    _as(monkeypatch, "background", "20261122_010101_worker_aaaa")
    assert _create("The retry path drops the last frame", ["spawned-by-triage"])["success"]
    _as(monkeypatch, "background", "20261122_020202_worker_bbbb")
    assert _create("The upload queue never drains on a 503", ["spawned-by-triage"])["success"]


def test_a_person_is_never_capped(board, monkeypatch):
    d, journal = board
    _as(monkeypatch, "chat", "20261122_010101_cccc")
    for name in ("The retry path drops the last frame",
                 "The upload queue never drains on a 503",
                 "Voice wake word misfires on a television"):
        assert _create(name, ["spawned-by-triage"])["created"]
    assert len(list(d.glob("*.md"))) == 3 and not journal.exists()


@pytest.mark.parametrize("extra", ["blocker", "umbrella"])
def test_a_blocker_and_an_umbrella_are_exempt(board, monkeypatch, extra):
    _as(monkeypatch, "background", "20261122_010101_worker_aaaa")
    assert _create("The retry path drops the last frame", ["spawned-by-autocode"])["success"]
    assert _create("The gate cannot start: canary port held",
                   ["spawned-by-autocode", extra])["created"]


def test_an_untagged_create_from_a_scheduled_task_is_not_capped(board, monkeypatch):
    _as(monkeypatch, "background", "20261122_010101_autonomy_dddd")
    for name in ("The retry path drops the last frame", "The upload queue never drains on a 503"):
        assert _create(name, ["backlog"])["created"]


def test_a_cap_check_that_raises_is_the_old_write(board, monkeypatch):
    def boom():
        raise RuntimeError("registry down")
    monkeypatch.setattr(BL, "_session_class", boom)
    assert _create("The retry path drops the last frame", ["spawned-by-triage"])["success"]


# ── scorecard rows 18-20 and the daily record ────────────────────────────

def git(repo, *args, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=False, env=env)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "app").mkdir(parents=True)
    (r / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "l@l")
    git(r, "config", "user.name", "lloyd")
    (r / "app" / "m.py").write_text("A = 1\nB = 2\nC = 3\n")
    git(r, "add", "-A")
    date = str(int(NOW - 30 * DAY))
    git(r, "commit", "-q", "-m", "base",
        env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "PATH": "/usr/bin:/bin"})
    return r


def _commit(repo, files: dict, days_ago: float):
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    date = str(int(NOW - days_ago * DAY))
    git(repo, "commit", "-q", "-m", "change",
        env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "PATH": "/usr/bin:/bin"})


def test_tree_growth_splits_tests_prose_and_code_and_counts_deletions(repo):
    _commit(repo, {"app/m.py": "A = 1\n",                       # 2 deleted
                   "tests/test_m.py": "def test_m():\n    assert True\n    # x\n    # y\n",
                   "architecture/x.md": "one\ntwo\n",
                   "app/n.py": "N = 1\n"}, 2)
    g = SC._tree_growth(repo, NOW - 7 * DAY)
    assert g["measured"] and g["commits"] == 1
    assert (g["added"], g["deleted"], g["net"]) == (7, 2, 5)
    assert (g["added_tests"], g["added_prose"], g["added_other"]) == (4, 2, 1)
    assert g["test_share"] == round(4 / 7, 3)


def test_tree_growth_on_an_unreadable_repo_is_unmeasured_not_zero(tmp_path):
    assert SC._tree_growth(tmp_path / "nowhere", NOW - DAY)["measured"] is False


def _item(d: Path, item_id: int, name: str, *, tags=("backlog",), created: float):
    stamp = datetime.fromtimestamp(created, tz=timezone.utc).replace(tzinfo=None).isoformat()
    fm = {"status": "draft", "priority": "low", "board": "lloyd", "tags": list(tags),
          "created": stamp}
    (d / f"{item_id}-x.md").write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\nBody.\n", encoding="utf-8")


def test_derived_share_counts_new_items_only(tmp_path):
    d = tmp_path / "backlog"
    d.mkdir()
    _item(d, 1, "Commit #7's witness bytes", created=NOW - DAY, tags=("backlog", B.OWED_SPAWN_TAG))
    _item(d, 2, "The retry path drops the last frame", created=NOW - DAY)
    _item(d, 3, "Commit #5's witness bytes", created=NOW - 20 * DAY)      # outside the window
    out = SC._derived_share(d, NOW - 7 * DAY)
    assert (out["created"], out["derived"], out["owed_filed"]) == (2, 1, 1)
    assert out["derived_share"] == 0.5


def test_tests_rung_reads_only_the_tests_rung():
    rows = [{"rung": "tests", "seconds": 100.0}, {"rung": "tests", "seconds": 2000.0},
            {"rung": "tests", "seconds": 300.0}, {"rung": "static", "seconds": 9999.0},
            {"rung": "tests"}]
    assert SC._tests_rung(rows) == {"runs": 3, "median_s": 300.0, "over_1800": 1}


def test_a_row_recorded_before_these_rows_existed_still_renders():
    lines = SC._render_growth_rows({})
    assert [x.split("|")[1].strip() for x in lines] == ["18", "19", "20", "21"]
    assert all("—" in x for x in lines)


def test_the_scorecard_records_once_a_day(tmp_path, monkeypatch):
    path = tmp_path / "scorecard.jsonl"
    calls = []

    def fake_compute(*, now, **kw):
        calls.append(now)
        return {"computed_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds")}

    monkeypatch.setattr(SC, "compute", fake_compute)
    assert SC.record_if_due(now=NOW, path=path) == path             # no file: due
    assert SC.record_if_due(now=NOW + 3600, path=path) is None      # an hour on: not due
    assert SC.record_if_due(now=NOW + 22 * 3600, path=path) is None
    assert SC.record_if_due(now=NOW + 23 * 3600 + 1, path=path) == path
    assert len(path.read_text().splitlines()) == 2 and len(calls) == 2
    assert SC.last_recorded_at(path) == NOW + 23 * 3600 + 1


def test_a_torn_last_line_does_not_stop_the_record(tmp_path):
    path = tmp_path / "scorecard.jsonl"
    stamp = datetime.fromtimestamp(NOW, tz=timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps({"computed_at": stamp}) + "\n" + '{"computed_at": "2026-', encoding="utf-8")
    assert SC.last_recorded_at(path) == NOW


# ── row 21: what each source's items turned into ─────────────────────────

def _yield_item(d: Path, item_id: int, name: str, *, tags=("backlog",), status="draft",
                body="Body.", days_ago=1.0, **extra):
    stamp = datetime.fromtimestamp(NOW - days_ago * DAY, tz=timezone.utc).replace(tzinfo=None).isoformat()
    fm = {"status": status, "priority": "low", "board": "lloyd", "tags": list(tags),
          "created": stamp, **extra}
    (d / f"{item_id}-x.md").write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n", encoding="utf-8")


def test_source_yield_attributes_and_classifies(tmp_path):
    from scripts.automod import source_yield as SY
    d = tmp_path / "backlog"
    d.mkdir()
    # owed-check: by tag, and by the body marker its older follow-ups carry
    _yield_item(d, 1, "Commit #9's witness bytes", tags=("backlog", B.OWED_SPAWN_TAG),
                status="done", automod_landed="abc")
    _yield_item(d, 2, "Add the retry test", body=f"x\n\n{SY.OWED_BODY_MARKER}9's owed entry: y",
                status="done", automod_landed="def")
    _yield_item(d, 3, "Another follow-up", tags=("backlog", B.OWED_SPAWN_TAG))
    _yield_item(d, 4, "A stale finding", tags=("spawned-by-triage",), status="done",
                autotriage_retired="stale")
    _yield_item(d, 5, "Nobody picked this up", tags=("spawned-by-review", B.EXPIRED_TAG), status="done")
    _yield_item(d, 6, "A census signature", tags=("spawned-by-autonomy-92",), status="done")
    _yield_item(d, 7, "Too old to count", tags=("spawned-by-triage",), days_ago=90)
    out = SY.compute(days=30, backlog_dir=d, now=NOW)
    s = out["sources"]
    assert set(s) == {"owed-check", "triage", "arch-review", "scheduled-task"}
    assert (s["owed-check"]["created"], s["owed-check"]["landed"], s["owed-check"]["open"],
            s["owed-check"]["derived"]) == (3, 2, 1, 1)
    assert s["triage"]["retired"] == 1 and s["arch-review"]["expired"] == 1
    assert s["scheduled-task"]["closed"] == 1
    assert out["total"]["created"] == 6
    assert s["owed-check"]["yield"] is None            # two closed: not a rate
    assert list(s)[0] == "owed-check"                  # largest first


def test_a_yield_is_a_rate_only_over_enough_closed_items(tmp_path):
    from scripts.automod import source_yield as SY
    d = tmp_path / "backlog"
    d.mkdir()
    for n in range(SY.MIN_CLOSED):
        _yield_item(d, n + 1, f"Finding {n}", tags=("spawned-by-triage",), status="done",
                    **({"automod_landed": "abc"} if n < 2 else {"autotriage_retired": "stale"}))
    assert SY.compute(days=30, backlog_dir=d, now=NOW)["sources"]["triage"]["yield"] == 0.4
