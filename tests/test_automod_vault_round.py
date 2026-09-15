"""The vault route: validate → commit only these paths → revert on failure.

The vault is a live, shared, always-dirty tree with no worktree, so "nothing
lands unverified" has to be enforced after the edit, not before it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.automod import state as S, vault_round as V


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    r = tmp_path / "obsidian"
    (r / "autonomy").mkdir(parents=True)
    (r / "skills" / "foo").mkdir(parents=True)
    (r / "backlog").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "autonomy" / "1-task.md").write_text("---\nid: 1\nstatus: up_next\n---\n# task\n")
    (r / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo\n")
    (r / "backlog" / "9-item.md").write_text("---\nstatus: draft\n---\n# item\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    monkeypatch.setattr(V, "VAULT", r)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    return r


def _events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


def test_the_apps_own_config_is_denied(vault):
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "app.json").write_text("{}")
    with pytest.raises(V.VaultRoundError, match="denied"):
        V.land([".obsidian/app.json"], "touch app config")
    assert (vault / ".obsidian" / "app.json").exists(), "a scope refusal does not revert anything"


def test_broken_front_matter_is_refused_and_the_file_put_back(vault):
    f = vault / "autonomy" / "1-task.md"
    f.write_text("---\nid: [unclosed\n---\n# task\n")
    with pytest.raises(V.VaultRoundError, match="reverted"):
        V.land(["autonomy/1-task.md"], "break a task", item_id=1)
    assert f.read_text() == "---\nid: 1\nstatus: up_next\n---\n# task\n"
    ev = _events("vault_land")[-1]
    assert ev["ok"] is False and ev["reverted"] == ["autonomy/1-task.md"] and ev["item_id"] == 1


def test_a_new_file_that_fails_validation_is_deleted(vault):
    f = vault / "autonomy" / "2-new.md"
    f.write_text("---\nnot: [valid\n---\n")
    with pytest.raises(V.VaultRoundError):
        V.land(["autonomy/2-new.md"], "add a broken task")
    assert not f.exists()


def test_success_commits_exactly_the_given_paths_and_leaves_the_rest_dirty(vault):
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: up_next\n---\n# item\n")
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo, edited elsewhere\n")
    out = V.land(["backlog/9-item.md"], "backlog: promote #9", item_id=9)
    assert out["ok"] and out["paths"] == ["backlog/9-item.md"]
    shown = git(vault, "show", "--stat", "--format=", out["commit"]).stdout
    assert "backlog/9-item.md" in shown and "SKILL.md" not in shown
    assert "skills/foo/SKILL.md" in git(vault, "status", "--short").stdout, "someone else's edit stays theirs"
    ev = _events("vault_land")[-1]
    assert ev["ok"] and ev["commit"] == out["commit"] and ev["item_id"] == 9


def test_loaders_run_only_for_paths_that_feed_a_prompt_or_the_scheduler(vault, monkeypatch):
    seen = []
    monkeypatch.setattr(V, "loader_errors", lambda paths: seen.append(list(paths)) or [])
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")
    V.land(["backlog/9-item.md"], "close #9")
    assert seen == [], "a backlog edit cannot break a prompt; do not spend a loader on it"
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v2\n")
    V.land(["skills/foo/SKILL.md"], "skill: foo v2")
    assert seen == [["skills/foo/SKILL.md"]]


def test_a_loader_failure_reverts_too(vault, monkeypatch):
    monkeypatch.setattr(V, "loader_errors", lambda paths: ["skills/foo: does not load"])
    f = vault / "skills" / "foo" / "SKILL.md"
    f.write_text("---\nname: foo\n---\n# broken in a way only the loader sees\n")
    with pytest.raises(V.VaultRoundError, match="does not load"):
        V.land(["skills/foo/SKILL.md"], "skill: foo")
    assert f.read_text() == "---\nname: foo\n---\n# foo\n"


def test_revert_is_a_plain_git_revert_and_is_recorded(vault):
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")
    sha = V.land(["backlog/9-item.md"], "close #9")["commit"]
    out = V.revert(sha, reason="wrong item")
    assert out["reverted"] == sha and out["commit"] != sha
    assert (vault / "backlog" / "9-item.md").read_text() == "---\nstatus: draft\n---\n# item\n"
    assert _events("vault_revert")[-1]["reason"] == "wrong item"


def test_commits_land_on_main_even_from_a_stranded_branch(vault):
    git(vault, "checkout", "-q", "-b", "experiment-7")
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")
    V.land(["backlog/9-item.md"], "close #9")
    assert git(vault, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


def test_nothing_to_commit_is_an_error_not_a_silent_success(vault):
    with pytest.raises(V.VaultRoundError, match="nothing to commit"):
        V.land(["backlog/9-item.md"], "no-op")


# ── #955: the landing clause, and what an abstention leaves behind ──────────


def write_item(d: Path, item_id: int, clauses: list[str]) -> Path:
    """An item in the backlog tree with front-matter acceptance clauses — what
    `review.item_contract` reads. Empty `clauses` means the file has none."""
    import yaml
    body = ("---\nstatus: up_next\nboard: lloyd\nacceptance_clauses:\n"
            f"{yaml.dump(list(clauses), default_flow_style=False, indent=2)}"
            "---\n\n# A thing\n\nDo it.\n") if clauses else (
        "---\nstatus: up_next\nboard: lloyd\n---\n\n# A thing\n\nDo it.\n")
    p = d / f"{item_id}-a-thing.md"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def items(tmp_path, monkeypatch):
    """A backlog tree of one, so `item_contract` is readable without the vault."""
    import scripts.automod.backlog as B
    d = tmp_path / "backlog-items"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    return d


GRADER_SAYS_CONTENT_MET_LANDING_UNMET = {
    "premise": "sound", "summary": "content is fine, no commit yet", "test_honesty": [],
    "seams_unverified": [],
    "clauses": [
        {"clause": 1, "verdict": "met", "evidence_path": "skills/foo/SKILL.md",
         "evidence_line": 1, "test_node_id": "", "how_verified": "read", "note": "named"},
        {"clause": 2, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
         "test_node_id": "", "how_verified": "read",
         "note": "Never landed — uncommitted working-tree state, no revertable sha."},
    ]}


def test_a_vault_round_whose_contract_demands_a_sha_can_reach_a_commit(vault, items, monkeypatch):
    """#425 and #502, end to end. The item's clause 2 IS the landing; `land()`
    grades before it commits, so the reviewer reported "no revertable sha" and
    refused — twice on each item, clause 1 satisfied both times — and the second
    refusal reverted the work. Only the network seam is replaced: the grader is
    the real `review.grade_vault`, as on the MCP path."""
    import scripts.automod.backlog as B
    import scripts.automod.review as RV
    write_item(items, 42, ["the skill names the retry rule",
                           "The change lands via automod_vault_land as one revertable sha on vault main"])
    monkeypatch.setattr(V, "GRADER", RV.grade_vault)
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": True, "structured": GRADER_SAYS_CONTENT_MET_LANDING_UNMET})
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo retry rule\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo names the retry rule (#42)", item_id=42)
    assert out["review"] == "pass", "a clause about the landing must not refuse the round"
    sha = out["commit"]
    assert out["landing_clauses"] == [{"clause": 2, "verdict": "met", "commit": sha}]
    ev = _events("vault_land")[-1]
    assert ev["landing_clauses"] == out["landing_clauses"]
    # A land the reviewer passed carries no skip reason: `review: pass` beside a
    # reason naming a reviewer that was never consulted would be a false field.
    assert out["review_reason"] is None and ev["review_reason"] is None
    assert [c["clause"] for c in ev["review_clauses"]] == [1, 2]
    assert ev["review_clauses"][1]["commit"] == sha
    # The consumer that closes a vault item on its review: with clause 2 graded
    # from the sha the contract is complete; before #955 it stayed
    # `post_landing` forever and this answered None, which left the item open.
    outcome = B.vault_review_outcome(S.LEDGER_PATH, [sha])
    assert outcome and outcome["acceptance"] == "met", outcome
    assert sorted(c["clause"] for c in outcome["clause_outcomes"]) == [1, 2]


def test_a_landing_clause_is_the_only_clause_graded_after_the_commit(vault, items, monkeypatch):
    """The indices come from the contract, so the caller can grade them even when
    the grader abstained and returned no rows at all."""
    write_item(items, 43, ["the skill names the retry rule",
                           "The change is submitted through `automod_vault_land` as one call "
                           "naming exactly the six SKILL.md paths above, and no path under "
                           "~/lloyd changes.",
                           "No field list names `board_id` (0 occurrences across 37 real keys)."])
    assert V._landing_clause_indices(43) == [2]
    # The reviewer grades 1 and 3, marks 2 `post_landing` for the caller, and says
    # nothing that would refuse the round; `land()` then fills 2 in from the sha.
    monkeypatch.setattr(V, "GRADER", lambda **kw: (
        "pass", "all content clauses met",
        [{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "post_landing",
          "subject": "landing"}, {"clause": 3, "verdict": "met"}]))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo retry rule\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo (#43)", item_id=43)
    assert out["landing_clauses"] == [{"clause": 2, "verdict": "met", "commit": out["commit"]}]
    # Clauses 1 and 3 stay the reviewer's: the exclusion must not widen until it
    # swallows the whole contract. Clause 2's `post_landing` placeholder is gone,
    # replaced by the verdict the sha carries.
    recorded = _events("vault_land")[-1]["review_clauses"]
    assert [(c["clause"], c["verdict"]) for c in recorded] == [(1, "met"), (2, "met"), (3, "met")]
    assert "commit" not in recorded[0] and recorded[1]["commit"] == out["commit"]
    # A land the reviewer passed is not explained by a reviewer that was never
    # consulted: the reason field is a skip's, and a passing land has none.
    assert out["review_reason"] is None
    assert _events("vault_land")[-1]["review_reason"] is None


def test_a_landing_clause_is_graded_unmet_when_a_named_path_missed_the_commit(vault, items, monkeypatch):
    """The after-the-fact verdict is derived from the commit's own file list, not
    asserted. Here the round named two paths and only one was changed, so the sha
    does not contain the other and the clause that names both is false — recorded
    as false rather than as a pass that `git show` contradicts."""
    (vault / "skills" / "foo" / "SECOND.md").write_text("---\nname: second\n---\n# second\n")
    git(vault, "add", "-A", "--", "skills/foo/SECOND.md")
    git(vault, "commit", "-q", "-m", "second file exists and will not change")
    write_item(items, 44, ["The change lands through automod_vault_land naming exactly "
                           "skills/foo/SKILL.md and skills/foo/SECOND.md"])
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("pass", "content met"))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo retry rule\n")
    out = V.land(["skills/foo/SKILL.md", "skills/foo/SECOND.md"], "skill: foo (#44)", item_id=44)
    row = out["landing_clauses"][0]
    assert row["verdict"] == "unmet" and row["commit"] == out["commit"]
    assert "skills/foo/SECOND.md" in row["note"]
    assert _events("vault_land")[-1]["landing_clauses"] == out["landing_clauses"]


async def test_the_landing_fields_survive_the_mcp_tool_boundary(vault, items, monkeypatch):
    """The one process boundary this change crosses.

    `agent_mcp/automod.py` is the ONLY caller that wires a grader — it sets
    `VR.GRADER = RV.grade_vault` itself and hands the agent `json.dumps(out)` —
    so it is where `landing_clauses` and `review_reason` are either kept or
    dropped on the floor. A test on the in-process `land()` return value cannot
    see that. Only the network seam is replaced, so the handler runs the real
    `grade_vault` over the real contract. The IV gate is stubbed: it reads
    session state, and what is under test here is the payload, not who may call.
    """
    import json

    import agent_mcp.automod as AM
    import scripts.automod.review as RV
    monkeypatch.setattr(AM, "_inner_voice_gate", lambda action: None)
    write_item(items, 45, ["the skill names the retry rule",
                           "The change lands through automod_vault_land as one "
                           "revertable sha on vault main"])
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": True, "structured": GRADER_SAYS_CONTENT_MET_LANDING_UNMET})
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo retry rule\n")
    payload = json.loads((await AM.call_tool("automod_vault_land", {
        "paths": ["skills/foo/SKILL.md"], "message": "skill: foo names the retry rule (#45)",
        "item_id": 45})).content[0].text)
    # The reviewer said clause 2 was unmet because no sha existed; the handler
    # still reports a pass, and clause 2's real verdict comes back from the commit.
    assert payload["review"] == "pass" and payload["commit"]
    assert payload["landing_clauses"] == [{"clause": 2, "verdict": "met",
                                           "commit": payload["commit"]}]
    assert payload["review_reason"] is None


def test_a_skipped_vault_review_records_which_abstention_it_was(vault, items, monkeypatch):
    """All six skip causes, verbatim. #955's merged finding: 34 of 54 successful
    item-bound landings carried one unlabelled word, so a refusal a later attempt
    ignored was indistinguishable from a grader outage."""
    import scripts.automod.review as RV
    write_item(items, 90, ["a clause"])
    write_item(items, 91, [])
    reasons = []
    monkeypatch.setattr(V, "GRADER", None)
    reasons.append(V._vault_review(["a.md"], 1)[1])
    monkeypatch.setattr(V, "GRADER", lambda **kw: (_ for _ in ()).throw(RuntimeError("engine gone")))
    reasons.append(V._vault_review(["a.md"], 1)[1])
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": False, "error": "backend 503"})
    reasons.append(RV.grade_vault(item_id=90, paths=["a.md"], diff="+x", vault=vault)[1])
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": True, "structured": {"premise": "?"}})
    reasons.append(RV.grade_vault(item_id=90, paths=["a.md"], diff="+x", vault=vault)[1])
    S.append_event({"event": "backlog_triage", "item_id": 90, "verdict": "confirmed",
                    "surface": "code", "acceptance": "a", "acceptance_clauses": ["a"]},
                   path=S.LEDGER_PATH)
    reasons.append(RV.grade_vault(item_id=90, paths=["a.md"], diff="+x", vault=vault)[1])
    reasons.append(RV.grade_vault(item_id=91, paths=["a.md"], diff="+x", vault=vault)[1])
    assert len(set(reasons)) == 6, reasons
    assert [r.split(":")[0].split(" not vault")[0] for r in reasons[:4]] == [
        "no grader configured", "grader raised RuntimeError",
        "grader did not answer", "grader returned an unusable object"], reasons
    assert reasons[4].startswith("surface is code, not vault")
    assert "item #91 has no acceptance clauses" == reasons[5]
    # And one of them, carried all the way onto the ledger by a landing.
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("skipped", "grader did not answer: 503", []))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v6\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v6", item_id=9)
    assert out["review"] == "skipped" and out["review_reason"] == "grader did not answer: 503"
    ev = _events("vault_review")[-1]
    assert ev["blocking"] is False and ev["kind"] == "skipped"
    assert ev["review_reason"] == "grader did not answer: 503"
    assert _events("vault_land")[-1]["review_reason"] == "grader did not answer: 503"


def test_a_land_bound_to_no_item_is_not_recorded_as_a_grader_outage(vault):
    """`scripts/autoresearch/promote.py` and this module's CLI never wire a
    grader, so every prompt promotion shared one label with a grader being down."""
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v7\n")
    out = V.land(["skills/foo/SKILL.md"], "autoresearch: promote v-x")
    assert out["review"] == "skipped"
    assert "no item bound" in out["review_reason"]
    assert "autoresearch" in out["review_reason"] and "not consulted" in out["review_reason"]
    assert out["review_reason"] != "grader did not answer: backend 503"
    assert _events("vault_land")[-1]["review_reason"] == out["review_reason"]
    assert _events("vault_review") == [], "there is no item to file a review against"


def test_front_matter_checker():
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "a.md").write_text("no front matter\n")
    (d / "b.md").write_text("---\nk: v\n---\nbody\n")
    (d / "c.md").write_text("---\nk: v\nbody without close\n")
    (d / "d.md").write_text("---\n- a list\n---\n")
    assert V.frontmatter_error(d / "a.md") is None
    assert V.frontmatter_error(d / "b.md") is None
    assert "never closes" in V.frontmatter_error(d / "c.md")
    assert "mapping" in V.frontmatter_error(d / "d.md")
