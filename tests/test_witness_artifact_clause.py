"""#1869: a measurement witness that lives in no history gets an archive clause.

A measurement item's claim is a number read off a live store — rows of a jsonl,
a count out of a db — and those paths sit under the data root, where
`git rev-parse --show-toplevel` exits 128 (`fatal: not a git repository`) while
`~/lloyd` and `~/obsidian` each report a toplevel. The bytes a claim was read
off therefore have no history anywhere: #1621's 44-row report sat on one disk
until #1756 committed it to `backlog/data/…jsonl` at `5c5e6826`. #1756's owed
ruling (2026-09-30T02:14:09, run `20260929_190733_owedcheck_16ba`) settled the
policy — a standing rule on the item-authoring side, not the review prompt — so
the rule is a function on the path that writes a confirmed contract
(`add_witness_artifact_clause`, called from `record_verdict` after
`cap_new_clauses`), not a prompt bullet: four of this item's five checks are
behaviour of the emitted clause list.

Magnitude at filing: 6 open items under `~/obsidian/backlog/*.md` cite an
out-of-tree bytes path and 0 of them name a vault artifact, so the rule is
prospective. Its stated limit holds too: the trigger is a probe on a path the
item NAMES, so an item quoting a store's numbers with no path at all is
unreachable by it.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, cluster as CL, state as S


def write_item(d: Path, item_id, *, status="draft", days_old=100, tags=("backlog",),
               name=None, body="Do the thing.", extra=None) -> Path:
    name = name or f"Item {item_id}"
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": "lloyd", "tags": list(tags), **(extra or {})}
    p = d / f"{item_id}-{name.lower().replace(' ', '-')[:30]}.md"
    p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
                 encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(CL, "CLUSTERS_PATH", tmp_path / "clusters.json")
    return d


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text())[0]


def _witness(tmp_path: Path, name: str = "iv-metrics.jsonl") -> Path:
    """A witness where a witness lives on the live box: under the data root,
    in a directory no git tree covers."""
    d = tmp_path / "lloyd-data" / "_pipeline" / "reflection"
    d.mkdir(parents=True)
    p = d / name
    p.write_text('{"llm_calls": 164, "miss_rate": null}\n' * 17)
    return p


def _git_repo(tmp_path: Path) -> Path:
    """A real repository, so 'inside a tree' is the probe's own answer and not
    this file's assertion about a path string."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / "data").mkdir()
    (repo / "data" / "report.jsonl").write_text('{"rows": 17}\n')
    return repo


# The item as triage writes it: the witness path is in the body, and no clause
# says where its bytes go.
_BODY_1869 = ("Live run 2026-09-30 over `{witness}` printed 17 rows with "
              "`miss_rate` null in 9. The change is display-only.")
_AUTHORED = ["the printed count is pinned by a test — tests/test_x.py"]


def test_a_witness_in_no_git_tree_gains_one_clause_with_a_path_and_a_command(tmp_path):
    """Clause 1: the added clause carries both halves — where the bytes go, and
    how to recompute the quoted report from the committed bytes.

    The default probe is the production one, run against a real directory that
    no repository covers, so the trigger is measured rather than injected.
    """
    witness = _witness(tmp_path)
    assert B.in_git_tree(witness) is False, "fixture must sit outside every tree"
    assert B.in_git_tree(Path(__file__)) is True, "the probe works at all"

    before = list(_AUTHORED)
    out = B.add_witness_artifact_clause(_AUTHORED, _BODY_1869.format(witness=witness))

    assert len(out) == 2, f"exactly one clause is added, got {out}"
    assert out[0] == before[0], "the authored clause is kept first, untouched"
    assert _AUTHORED == before, "the caller's list is never mutated in place"
    added = out[1]
    artifact = f"{B.WITNESS_ARTIFACT_DIR}/{witness.name}"
    assert artifact == "backlog/data/iv-metrics.jsonl", artifact
    assert artifact in added, added
    assert f"`wc -l < {artifact}`" in added, (
        f"the clause must carry a shell command over the COMMITTED bytes, not a "
        f"promise to re-measure: {added}")
    assert str(witness) in added, "and it names the witness it is archiving"
    assert len(added) <= B.CLAUSE_MAX_CHARS, "it survives clean_clauses intact"


def test_evidence_that_all_sits_in_a_git_tree_returns_the_list_unchanged(tmp_path):
    """Clause 2: nothing is invented when the bytes already have a history.

    The paths are inside a real `git init`'d repo and the real probe answers,
    so an unchanged list is the rule declining to fire, not the probe failing
    to find anything.
    """
    repo = _git_repo(tmp_path)
    in_tree = [repo / "data" / "report.jsonl", repo / "notes.txt", Path(__file__)]
    for p in in_tree:
        assert B.in_git_tree(p) is True, f"{p} is inside a tree"

    text = "Measured over " + ", ".join(f"`{p}`" for p in in_tree) + ": 17 rows."
    out = B.add_witness_artifact_clause(_AUTHORED, text)

    assert out == _AUTHORED, out
    assert not any("backlog/data/" in c for c in out), out


def test_the_trigger_is_the_probe_on_the_witness_not_the_data_root_string(tmp_path):
    """Clause 3: the same path decides differently when the probe does.

    A `~/lloyd-data` path with an injected probe that reports "in a tree" adds
    nothing, and a `tmp_path` witness — which really has no tree — adds the
    clause only when the probe says so. Both directions are needed: the first
    says the rule is not a substring match on the data root, the second says
    it is not keyed to one machine's layout either.
    """
    data_root_witness = "~/lloyd-data/_pipeline/reflection/iv-metrics.jsonl"
    tmp_witness = _witness(tmp_path, "usage-report.jsonl")

    said_in_a_tree = B.add_witness_artifact_clause(
        _AUTHORED, _BODY_1869.format(witness=data_root_witness),
        probe=lambda path: True)
    assert said_in_a_tree == _AUTHORED, (
        f"a path under the data root alone must not fire the rule: {said_in_a_tree}")

    said_no_tree = B.add_witness_artifact_clause(
        _AUTHORED, _BODY_1869.format(witness=tmp_witness), probe=lambda path: False)
    assert len(said_no_tree) == 2, said_no_tree
    assert "backlog/data/usage-report.jsonl" in said_no_tree[1]

    but_in_one = B.add_witness_artifact_clause(
        _AUTHORED, _BODY_1869.format(witness=tmp_witness), probe=lambda path: True)
    assert but_in_one == _AUTHORED, (
        "the probe is the whole trigger; the path's shape decides nothing")


def test_the_real_probe_follows_a_tilde_witness_to_whatever_tree_home_is(tmp_path,
                                                                         monkeypatch):
    """The same clause-3 property with the production probe instead of an
    injected one, so nothing about the trigger is only true of a fake.

    `~` is expanded when the witness is classified, so pointing `HOME` at a
    throwaway directory decides both ways with the shipped `in_git_tree`: an
    out-of-tree home gains the clause, a home that is itself a git repository
    does not. A rule keyed on the literal string `~/lloyd-data` cannot make
    those two differ, since the text is identical.
    """
    tilde = "~/lloyd-data/_pipeline/reflection/iv-metrics.jsonl"
    body = _BODY_1869.format(witness=tilde)
    home = tmp_path / "home"
    (home / "_pipeline" / "reflection").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    out = B.add_witness_artifact_clause(_AUTHORED, body)
    assert len(out) == 2 and "backlog/data/iv-metrics.jsonl" in out[1], out

    (tmp_path / "vh").mkdir()
    versioned_home = _git_repo(tmp_path / "vh")
    monkeypatch.setenv("HOME", str(versioned_home))
    assert B.add_witness_artifact_clause(_AUTHORED, body) == _AUTHORED, (
        "the same path text under a home that is a tree must not fire")


def test_a_clause_that_already_names_an_archive_adds_no_second_one(tmp_path):
    """The de-dup half of the trigger: the rule is one clause per item, and an
    item whose contract already routes the bytes (the `backlog/data/`
    convention, or an explicit vault path) is done asking."""
    witness = _witness(tmp_path)
    body = _BODY_1869.format(witness=witness)
    for clause in (f"commit the extract to `backlog/data/{witness.name}` in the vault",
                   "the mirror at `~/obsidian/knowledge/inner-voice/iv-metrics.jsonl` "
                   "carries the same bytes"):
        out = B.add_witness_artifact_clause([_AUTHORED[0], clause], body)
        assert out == [_AUTHORED[0], clause], out


def test_the_generated_clause_reaches_the_graded_contract(tmp_path, isolated):
    """Clause 5: `record_verdict` writes it into `acceptance_clauses`, and the
    post-landing backstop leaves it there.

    End to end through the real writer: the clause is generated between
    `cap_new_clauses` and `split_post_landing_clauses`, so the same regex that
    parks an authored "needs a day of traffic" clause gets its chance at this
    one — and does not move it, because it is worded over committed bytes with
    no time shape.
    """
    witness = _witness(tmp_path)
    assert B.in_git_tree(witness) is False, "fixture must sit outside every tree"
    p = write_item(isolated, 11, body=_BODY_1869.format(witness=witness))

    B.record_verdict(B.item_by_id(11), "confirmed", "the counts re-measure exactly",
                     acceptance="the report re-derives from committed bytes",
                     acceptance_clauses=list(_AUTHORED))

    fm = _fm(p)
    graded = fm["acceptance_clauses"]
    assert len(graded) == 2, graded
    generated = graded[1]
    assert "backlog/data/iv-metrics.jsonl" in generated and "`wc -l < " in generated
    assert not fm.get("human_clauses"), fm.get("human_clauses")
    assert "Moved to human clauses" not in p.read_text(), p.read_text()[-400:]
    assert B.split_post_landing_clauses([generated])[0] == [generated], (
        "the wording itself must survive the backstop, not its position in the list")
