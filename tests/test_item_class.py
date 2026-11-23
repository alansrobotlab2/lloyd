"""Item class: what kind of work an item is, and what selection does with it.

`scripts/automod/item_class.py`. Triage states `class` (and, for an
`improvement`, the `metric` it should move); the two keys are written to the
item, survive every later rewrite, and — behind two switches that ship off —
order the implement pool inside one human `priority` band.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, cluster as CL, item_class as IC, scorecard as SC, state as S
from workers.sources import autocode as AC, autotriage as W


def write_item(d: Path, item_id, *, status="up_next", days_old=10, priority="medium",
               **fm_extra) -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "created": created, "board": "lloyd", "tags": ["backlog"],
          "priority": priority, **fm_extra}
    p = d / f"{item_id}-item-{item_id}.md"
    p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# Item {item_id}\n\nDo it.\n",
                 encoding="utf-8")
    return p


@pytest.fixture
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(CL, "CLUSTERS_PATH", tmp_path / "clusters.json")
    return d


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text())[0]


BLOCK = (
    "VERDICT: confirmed\n"
    "SURFACE: code\n"
    "{extra}"
    "CHECK: pytest tests/test_x.py fails today\n"
    "EVIDENCE: app/foo.py:12 does the wrong thing\n"
    "ACCEPTANCE: the test passes\n"
    "ACCEPTANCE_CLAUSES:\n1. it works — tests/test_x.py\n"
    "HUMAN_CLAUSES: none\n"
    "SPAWNED: none\n")

STRUCTURED = {"verdict": "confirmed", "surface": "code", "check": "c", "evidence": "e",
              "acceptance": "a", "acceptance_clauses": ["it works — tests/test_x.py"],
              "human_clauses": [], "spawned": []}


# ── the schema ──────────────────────────────────────────────────────────────

def test_the_schema_offers_the_four_classes_and_a_way_to_state_none():
    props = B.TRIAGE_VERDICT_SCHEMA["properties"]
    assert props["class"]["enum"] == [*IC.CLASSES, IC.NO_CLASS]
    assert IC.CLASSES == ("improvement", "fix", "maintenance", "simplify")
    assert props["metric"]["maxLength"] == IC.METRIC_MAX > IC.METRIC_KEEP
    # The rail in test_structured_verdict.py: every declared property required.
    assert set(B.TRIAGE_VERDICT_SCHEMA["required"]) == set(props)


def test_extending_the_schema_twice_changes_nothing():
    once = copy.deepcopy(B.TRIAGE_VERDICT_SCHEMA)
    assert IC.extend_triage_schema(copy.deepcopy(once)) == once


# ── both parse paths, and the verdicts that name no class ───────────────────

def test_a_text_verdict_with_no_class_parses_to_exactly_the_old_keys():
    parsed = W.parse_verdict(BLOCK.format(extra=""))
    assert parsed["source"] == "regex"
    assert "class" not in parsed and "metric" not in parsed
    assert set(parsed) == {"verdict", "surface", "check", "evidence", "acceptance",
                           "acceptance_clauses", "clauses_dropped", "clauses_dropped_text",
                           "human_clauses", "spawned", "source"}


def test_a_structured_verdict_with_no_class_parses_to_exactly_the_old_keys():
    """An object written before the fields existed, one that says `none`, and
    one with a word outside the vocabulary are the same verdict."""
    old = W.parse_verdict("", structured=dict(STRUCTURED))
    assert old["source"] == "structured" and "class" not in old and "metric" not in old
    for cls in ("none", "", "refactor", None):
        again = W.parse_verdict("", structured={**STRUCTURED, "class": cls, "metric": "x"})
        assert again == old


def test_the_text_block_carries_class_and_metric_without_disturbing_its_neighbours():
    plain = W.parse_verdict(BLOCK.format(extra=""))
    parsed = W.parse_verdict(BLOCK.format(
        extra="CLASS: improvement\nMETRIC: `tool_choice`\n"))
    assert parsed["class"] == "improvement" and parsed["metric"] == "tool_choice"
    assert {k: v for k, v in parsed.items() if k not in ("class", "metric")} == plain
    # After SPAWNED too: a METRIC line there must not be read as a filed id.
    tail = W.parse_verdict(BLOCK.format(extra="") + "CLASS: improvement\nMETRIC: recall@5\n")
    assert tail["spawned"] == [] and tail["metric"] == "recall@5"


def test_prose_that_merely_starts_with_class_is_still_evidence():
    """`_FIELD` is case-insensitive for the old fields; the two new names match
    in capitals only, so an evidence line beginning `Class:` stays evidence."""
    text = BLOCK.format(extra="").replace(
        "EVIDENCE: app/foo.py:12 does the wrong thing\n",
        "EVIDENCE: app/foo.py:12 does the wrong thing\nClass: Foo is the culprit\n")
    parsed = W.parse_verdict(text)
    assert "Class: Foo is the culprit" in parsed["evidence"] and "class" not in parsed


def test_the_structured_path_carries_class_and_drops_a_metric_that_means_nothing():
    imp = W.parse_verdict("", structured={**STRUCTURED, "class": "improvement",
                                          "metric": "  tool_choice  "})
    assert (imp["class"], imp["metric"]) == ("improvement", "tool_choice")
    fix = W.parse_verdict("", structured={**STRUCTURED, "class": "fix", "metric": "tool_choice"})
    assert fix["class"] == "fix" and "metric" not in fix, "metric is an improvement's field"
    bare = W.parse_verdict("", structured={**STRUCTURED, "class": "improvement", "metric": "none"})
    assert bare["class"] == "improvement" and "metric" not in bare
    long = W.parse_verdict("", structured={**STRUCTURED, "class": "improvement",
                                           "metric": "m" * 500})
    assert len(long["metric"]) == IC.METRIC_KEEP


# ── the prompt ──────────────────────────────────────────────────────────────

def test_the_triage_prompt_asks_for_the_class_once_and_still_formats(board):
    write_item(board, 7, status="draft")
    text = W.render_prompt(B.item_by_id(7), ledger=S.LEDGER_PATH)
    assert text.count("CLASS: <if confirmed: one of improvement|fix|maintenance|simplify") == 1
    assert text.count("METRIC: <if improvement:") == 1
    assert "`simplify` = same behaviour, fewer lines" in text
    assert text.index("Say what kind of work it is") < text.index("Finish with exactly this block")
    assert text.index("SURFACE: <one of") < text.index("CLASS: <if") < text.index("CHECK: <the command")
    assert IC.splice_triage_prompt(W.PROMPT) == W.PROMPT, "splicing is idempotent"


def test_the_prompt_addition_stays_small():
    added = len(IC.TRIAGE_PARAGRAPH % "") + len(IC.TRIAGE_BLOCK_LINES)
    assert added < 800, f"{added} chars added to every single-item triage prompt"
    assert "{" not in IC.TRIAGE_PARAGRAPH + IC.TRIAGE_BLOCK_LINES, "PROMPT goes through str.format"


def test_a_reworded_prompt_fails_loudly_rather_than_never_asking():
    with pytest.raises(ValueError, match="anchor"):
        IC.splice_triage_prompt("a prompt with neither anchor")


def test_bench_names_come_from_the_manifest_when_there_is_one(tmp_path):
    assert IC.bench_names(tmp_path / "absent.yaml") == []
    m = tmp_path / "manifest.yaml"
    m.write_text("benches:\n  tool_choice: {floor: 0.9}\n  recall: {}\n")
    assert IC.bench_names(m) == ["tool_choice", "recall"]
    m.write_text("metrics:\n  - name: tool_choice\n  - id: recall\n  - plain\n")
    assert IC.bench_names(m) == ["tool_choice", "recall", "plain"]
    m.write_text("- not\n- a mapping\n")
    assert IC.bench_names(m) == []
    hinted = IC.splice_triage_prompt(
        "x\nFinish with exactly this block and nothing after it:\n"
        "CHECK: <the command or method you ran, one line>\n", benches=["tool_choice", "recall"])
    assert "(one of: tool_choice, recall)" in hinted


# ── the item: written once, kept by every writer ────────────────────────────

def test_a_confirmed_verdicts_class_is_written_to_the_item(board):
    p = write_item(board, 1, status="draft")
    item = B.item_by_id(1)
    parsed = {"verdict": "confirmed", "class": "improvement", "metric": "tool_choice"}
    assert IC.write_to_item(item, parsed) == {"class": "improvement", "metric": "tool_choice"}
    assert (_fm(p)["class"], _fm(p)["metric"]) == ("improvement", "tool_choice")
    assert IC.class_of(B.item_by_id(1)) == ("improvement", "tool_choice")
    # A re-triage that calls it a fix removes the metric it no longer means.
    assert IC.write_to_item(item, {"verdict": "confirmed", "class": "fix"}) == {"class": "fix"}
    assert _fm(p)["class"] == "fix" and "metric" not in _fm(p)


def test_no_class_no_confirmation_or_a_closed_item_writes_nothing(board):
    p = write_item(board, 1, status="draft")
    before = p.read_text()
    item = B.item_by_id(1)
    assert IC.write_to_item(item, {"verdict": "confirmed"}) == {}
    assert IC.write_to_item(item, {"verdict": "stale", "class": "fix"}) == {}
    assert IC.write_to_item(item, {"verdict": "confirmed", "class": "fix"}, closed=True) == {}
    assert p.read_text() == before
    assert IC.class_of(item) == ("", "")


def test_a_write_that_raises_costs_the_keys_and_never_the_triage(board, monkeypatch):
    write_item(board, 1, status="draft")
    monkeypatch.setattr(B, "update_frontmatter",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    assert IC.write_to_item(B.item_by_id(1), {"verdict": "confirmed", "class": "fix"}) == {"class": "fix"}


def test_the_class_survives_every_writer_that_rewrites_an_item(board):
    """`update_frontmatter`, a status move, `record_verdict`, the reconciler's
    writer (`_apply_status`) and the board API's writer each re-dump the dict
    they parsed, so a key they do not know is carried."""
    from app.routers import backlog as BR
    p = write_item(board, 1, status="draft", **{"class": "improvement", "metric": "tool_choice"})
    kept = lambda: (_fm(p).get("class"), _fm(p).get("metric"))  # noqa: E731
    B.update_frontmatter(p, {"worth": "high"}, activity="ranked", add_tags=("swept",))
    assert kept() == ("improvement", "tool_choice")
    B.record_verdict(B.item_by_id(1), "confirmed", "Reproduced.", acceptance="x",
                     acceptance_clauses=["a — tests/test_x.py"])
    assert kept() == ("improvement", "tool_choice")
    assert _fm(p)["status"] == "up_next"
    assert B._apply_status(p, "draft", "parked by hand") and _fm(p)["status"] == "draft"
    assert kept() == ("improvement", "tool_choice")
    fm, body = BR._backlog_parse_fm(p)
    fm["priority"] = "high"
    BR._write_task_file(p, fm, body)
    assert _fm(p)["priority"] == "high" and kept() == ("improvement", "tool_choice")


# ── selection: the pure rule ────────────────────────────────────────────────

OFF = IC.Prefs()
IMPROVE = IC.Prefs(prefer_improvement=True)
OVER = IC.Prefs(net_lines_budget=1000, growth=5000)
BOTH = IC.Prefs(prefer_improvement=True, net_lines_budget=1000, growth=5000)


def test_with_nothing_preferred_every_item_ranks_the_same():
    for cls, metric in (("improvement", "m"), ("simplify", ""), ("fix", ""), ("", "")):
        assert IC.selection_rank(cls, metric, OFF) == 0
    under = IC.Prefs(net_lines_budget=1000, growth=999)
    unknown = IC.Prefs(net_lines_budget=1000, growth=None)
    assert not under.active and not unknown.active, "an unknown growth is not over budget"
    assert IC.selection_rank("simplify", "", under) == IC.selection_rank("fix", "", under) == 0


def test_prefer_improvement_puts_an_improvement_that_names_a_metric_first():
    assert IC.selection_rank("improvement", "tool_choice", IMPROVE) \
        < IC.selection_rank("improvement", "", IMPROVE) \
        == IC.selection_rank("fix", "", IMPROVE) == IC.selection_rank("", "", IMPROVE) \
        == IC.selection_rank("simplify", "", IMPROVE)
    # Where a manifest lists the benches, the name has to be one of them.
    assert IC.selection_rank("improvement", "Tool_Choice", IMPROVE, ["tool_choice"]) == 1
    assert IC.selection_rank("improvement", "vibes", IMPROVE, ["tool_choice"]) == 2


def test_over_the_net_lines_budget_simplify_goes_first():
    assert IC.selection_rank("simplify", "", OVER) < IC.selection_rank("fix", "", OVER)
    assert IC.selection_rank("improvement", "m", OVER) == IC.selection_rank("fix", "", OVER), \
        "prefer_improvement is its own switch"
    assert [IC.selection_rank(c, m, BOTH) for c, m in
            (("simplify", ""), ("improvement", "m"), ("fix", ""))] == [0, 1, 2]


def test_prefs_come_from_config_and_ship_off():
    assert IC.prefs_from({}) == OFF and IC.prefs_from(None) == OFF
    assert IC.prefs_from({"automod": {"selection": {}}}) == OFF
    shipped = yaml.safe_load((Path(IC.__file__).resolve().parents[2] / "config.yaml").read_text())
    assert shipped["automod"]["selection"] == {"prefer_improvement": False,
                                               "net_lines_budget": None}
    on = IC.prefs_from({"automod": {"selection": {"prefer_improvement": True}}})
    assert on.prefer_improvement and not on.over_budget
    assert not IC.prefs_from({"automod": {"selection": {"prefer_improvement": "yes"}}}).active
    assert IC.prefs_from({"automod": {"selection": {"net_lines_budget": True}}}).net_lines_budget is None


# ── the growth figure: recorded, fresh, and never guessed ───────────────────

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc).timestamp()


def _row(net_non_test=5000, *, age_h=2.0, days=7.0, measured=True, **growth) -> dict:
    at = datetime.fromtimestamp(NOW - age_h * 3600, tz=timezone.utc).isoformat()
    g = {"measured": measured, "net": 60000, **growth}
    if net_non_test is not None:
        g["net_non_test"] = net_non_test
    return {"computed_at": at, "since_days": days, "tree_growth": g}


def test_growth_is_read_from_a_fresh_recorded_row():
    assert IC.growth_from_row(_row(5000), now=NOW) == 5000
    assert IC.growth_from_row(_row(-300), now=NOW) == -300
    assert IC.growth_from_row(_row(4000, days=14), now=NOW) == 2000, "per 7 days"
    cfg = {"automod": {"selection": {"net_lines_budget": 1000}}}
    assert IC.prefs_from(cfg, _row(5000), now=NOW).over_budget
    assert not IC.prefs_from(cfg, _row(900), now=NOW).over_budget


def test_a_missing_stale_unmeasured_or_old_shaped_row_is_unknown_not_zero():
    cfg = {"automod": {"selection": {"net_lines_budget": -10**9}}}   # any number is over this
    for row in (None, {}, _row(5000, age_h=24 * 4), _row(5000, measured=False),
                _row(None), {"computed_at": "junk", "tree_growth": {"measured": True, "net_non_test": 1}}):
        assert IC.growth_from_row(row, now=NOW) is None
        assert not IC.prefs_from(cfg, row, now=NOW).active


def test_the_newest_row_is_read_off_the_file_tail(tmp_path):
    p = tmp_path / "scorecard.jsonl"
    assert IC.newest_scorecard_row(p) is None
    p.write_text(json.dumps(_row(1)) + "\n" + json.dumps(_row(2)) + "\nnot json\n")
    assert IC.newest_scorecard_row(p)["tree_growth"]["net_non_test"] == 2


def test_selection_never_shells_out_for_the_growth_figure(board, monkeypatch, tmp_path):
    """The figure is the recorded row's; `git log` in a path that runs every
    worker tick is what the brief for this rule ruled out."""
    import subprocess
    path = tmp_path / "scorecard.jsonl"
    path.write_text(json.dumps({**_row(5000), "computed_at":
                                datetime.now(timezone.utc).isoformat()}) + "\n")
    monkeypatch.setattr(SC, "scorecard_path", lambda: path)
    from app import config as APPCFG
    monkeypatch.setitem(APPCFG.CONFIG, "automod",
                        {"selection": {"net_lines_budget": 1000}})
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: pytest.fail("selection ran a subprocess"))
    prefs = IC.current_prefs()
    assert prefs.over_budget and prefs.growth == 5000
    write_item(board, 1, **{"class": "simplify"})
    assert IC.selection_rank_of(B.item_by_id(1)) == 0


def test_the_scorecard_row_records_net_non_test_growth(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    def git(*a):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                        *a], check=True, capture_output=True)
    git("init", "-q")
    (repo / "a.py").write_text("x\n" * 10)
    (repo / "tests" / "test_a.py").write_text("t\n" * 20)
    git("add", "-A"); git("commit", "-qm", "one")
    (repo / "a.py").write_text("x\n" * 4)                       # −6 code
    (repo / "tests" / "test_a.py").write_text("t\n" * 15)       # −5 tests
    (repo / "b.py").write_text("y\n" * 3)                       # +3 code
    git("add", "-A"); git("commit", "-qm", "two")
    g = SC._tree_growth(repo, 0.0)
    assert g["measured"] and (g["added"], g["deleted"]) == (33, 11)
    assert g["added_tests"] == 20 and g["deleted_tests"] == 5
    assert g["net"] == 22
    assert g["net_non_test"] == (13 - 6), "non-test added minus non-test deleted"


# ── selection: inside the real ordering ─────────────────────────────────────

def _confirmed(board, i, *, priority="medium", days_old=10, **fm):
    write_item(board, i, priority=priority, days_old=days_old,
               acceptance_clauses=["clause a"], **fm)
    S.append_event({"event": "backlog_triage", "item_id": i, "verdict": "confirmed",
                    "acceptance": "x", "acceptance_clauses": ["clause a"]}, path=S.LEDGER_PATH)


@pytest.fixture
def pool(board):
    _confirmed(board, 1, days_old=40)                                         # oldest, unclassified
    _confirmed(board, 2, days_old=30, **{"class": "fix"})
    _confirmed(board, 3, days_old=20, **{"class": "improvement", "metric": "tool_choice"})
    _confirmed(board, 4, days_old=10, **{"class": "simplify"})
    _confirmed(board, 5, days_old=25, **{"class": "improvement"})             # names no metric
    return board


def _pick(monkeypatch, prefs) -> int:
    monkeypatch.setattr(IC, "current_prefs", lambda: prefs)
    return B.select_confirmed(S.LEDGER_PATH)[0].id


def test_switched_off_the_order_is_what_it_was_and_no_item_is_opened(pool, monkeypatch):
    monkeypatch.setattr(IC, "class_of", lambda item: pytest.fail("read an item while off"))
    assert _pick(monkeypatch, OFF) == 1, "oldest first, as before"


def test_prefer_improvement_picks_the_improvement_with_a_metric(pool, monkeypatch):
    assert _pick(monkeypatch, IMPROVE) == 3


def test_over_budget_picks_the_simplify_item(pool, monkeypatch):
    assert _pick(monkeypatch, OVER) == 4
    assert _pick(monkeypatch, BOTH) == 4
    assert _pick(monkeypatch, IC.Prefs(net_lines_budget=1000, growth=10)) == 1


def test_the_humans_priority_still_dominates_the_class(pool, monkeypatch):
    """A `low` improvement never passes a `medium` anything, and a `high`
    item is next whatever its class — within `high`, recency decides."""
    B.update_frontmatter(next(pool.glob("3-*.md")), {"priority": "low"})
    assert _pick(monkeypatch, IMPROVE) == 1
    B.update_frontmatter(next(pool.glob("3-*.md")), {"priority": "medium"})
    _confirmed(pool, 8, priority="high", days_old=50, **{"class": "improvement", "metric": "m"})
    _confirmed(pool, 9, priority="high", days_old=1, **{"class": "maintenance"})
    assert _pick(monkeypatch, IMPROVE) == 9, "newest high first; class does not reorder highs"
    assert _pick(monkeypatch, OFF) == 9


# ── simplify: deletion is creditable ────────────────────────────────────────

def test_net_lines_counts_non_test_python_only():
    numstat = ("10\t40\tapp/foo.py\n"
               "0\t200\ttests/test_foo.py\n"
               "5\t0\tapp/harness/tests/test_bar.py\n"
               "0\t30\ttests/conftest.py\n"
               "300\t0\tarchitecture/backlog.md\n"
               "7\t1\tconfig.yaml\n"
               "-\t-\tweb/logo.png\n"
               "3\t9\tscripts/automod/x.py\n"
               "garbage\n")
    assert IC.net_lines(numstat) == {"added": 13, "deleted": 49, "files": 2, "net_removed": 36}
    assert IC.net_lines("")["net_removed"] == 0
    assert IC.net_lines("50\t10\tapp/foo.py\n0\t500\ttests/test_foo.py\n")["net_removed"] == -40, \
        "deleting tests does not make the code smaller"


def test_only_a_simplify_items_contract_gains_the_implicit_check(board):
    write_item(board, 1, **{"class": "simplify"})
    write_item(board, 2, **{"class": "fix"})
    write_item(board, 3)
    note = IC.contract_note(B.item_by_id(1))
    assert "net non-test Python lines removed must be positive" in note
    assert "Not a numbered clause" in note
    assert IC.contract_note(B.item_by_id(2)) == IC.contract_note(B.item_by_id(3)) == ""
    assert IC.contract_note(object()) == "", "a candidate with no path is not a simplify item"


def test_the_implement_prompt_carries_the_note_after_the_clauses(board):
    triage = {"acceptance": "x", "acceptance_clauses": ["it shrinks — tests/test_x.py"],
              "surface": "code"}
    write_item(board, 1, **{"class": "simplify"})
    write_item(board, 2, **{"class": "fix"})
    simplify, _ = AC._build_prompt(B.item_by_id(1), triage, 100, "", None)
    plain, _ = AC._build_prompt(B.item_by_id(2), triage, 100, "", None)
    assert IC.SIMPLIFY_NOTE in simplify and IC.SIMPLIFY_NOTE not in plain
    assert simplify.index("1. it shrinks") < simplify.index("Implicit check")
    assert "Implicit check" not in plain


def test_the_gate_records_the_number_for_a_simplify_round_and_nothing_otherwise(board, tmp_path):
    import subprocess
    repo = tmp_path / "wt"
    (repo / "tests").mkdir(parents=True)
    def git(*a):
        r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                            *a], check=True, capture_output=True, text=True)
        return r.stdout.strip()
    git("init", "-q")
    (repo / "a.py").write_text("x\n" * 30)
    git("add", "-A"); git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (repo / "a.py").write_text("x\n" * 12)
    (repo / "tests" / "test_a.py").write_text("t\n" * 50)
    git("add", "-A"); git("commit", "-qm", "smaller")
    write_item(board, 1, **{"class": "simplify"})
    write_item(board, 2, **{"class": "fix"})
    data = IC.gate_data(1, base, repo)
    assert data == {"simplify": {"added": 0, "deleted": 18, "files": 1, "net_removed": 18,
                                 "ok": True, "observe_only": True}}
    assert IC.gate_data(2, base, repo) == {} and IC.gate_data(None, base, repo) == {}
    assert IC.gate_data(404, base, repo) == {}
    assert IC.gate_data(1, "no-such-rev", repo) == {"simplify": {"status": "unevaluated",
                                                                 "observe_only": True}}
    (repo / "a.py").write_text("x\n" * 40)
    git("add", "-A"); git("commit", "-qm", "bigger")
    assert IC.gate_data(1, base, repo)["simplify"]["ok"] is False
