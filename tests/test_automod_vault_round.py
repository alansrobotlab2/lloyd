"""The vault route: validate → commit only these paths → revert on failure.

The vault is a live, shared, always-dirty tree with no worktree, so "nothing
lands unverified" has to be enforced after the edit, not before it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_mcp.skills import _QUARANTINE_STATUSES
from scripts.automod import state as S, vault_round as V


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _scratch_vault(tmp_path, monkeypatch, *, mock_loaders: bool):
    """A git-init'd vault with one task, one skill, one backlog item.

    `mock_loaders=False` leaves the lander's REAL loader subprocess in place.
    Every fixture in this module until #777 mocked it, which is exactly how the
    skills predicate inside that subprocess could invent the slug `.archived`
    out of a dot-directory and refuse every archive rename while the suite
    stayed green — the refusal lives on the other side of a process boundary,
    so a mocked seam pinned everything except the one line that mattered.
    """
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
    if mock_loaders:
        monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    return r


@pytest.fixture
def vault(tmp_path, monkeypatch):
    return _scratch_vault(tmp_path, monkeypatch, mock_loaders=True)


@pytest.fixture
def livevalidatorvault(tmp_path, monkeypatch):
    """Same tree, real loader subprocess: the seam `automod_vault_land` uses."""
    return _scratch_vault(tmp_path, monkeypatch, mock_loaders=False)


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


# ── #777: the land check must be able to express retiring a skill ───────────
#
# `agent_mcp/skills.py:90` skips any dot-prefixed directory, which is what makes
# `skills/.archived/` an archive; `agent_mcp/skills.py` also returns its "do not
# retrieve this" sentinel for a skill whose front matter sets `status: archived`.
# The lander read BOTH as "the skill is broken", so retiring a skill — by moving
# it or by status — was impossible through `automod_vault_land`, and the failed
# validation reverted every other path in the batch with it. Vault commit
# `60776c12` archives `ingest` "by hand because automod_vault_land refuses a move
# into skills/.archived/": the cost of this was a human doing the landing.
#
# These four tests call the real `loader_errors` (and, for the last, the real
# `land`) against `livevalidatorvault`, because both halves of the defect live
# inside the subprocess `loader_errors` spawns — a mocked seam cannot see them.


def _retire_by_rename(vault) -> None:
    """`skills/foo/` → `skills/.archived/foo/`, quarantined at the destination."""
    d = vault / "skills" / ".archived" / "foo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: foo\nstatus: archived\n---\n# foo retired\n")
    (vault / "skills" / "foo" / "SKILL.md").unlink()


def test_an_archive_directory_is_never_a_skill_slug_to_the_load_check(livevalidatorvault):
    """Clause 1. The path set's only `skills/` entries sit under a dot-directory.

    Before the fix this returned `['skills/.archived: does not load']`: the slug
    was `p.split("/")[1]`, so a two-segment archive path yielded the directory
    name `.archived`, the loader abstained on it — correctly, it is not a skill —
    and the abstention was reported as damage.
    """
    _retire_by_rename(livevalidatorvault)
    errors = V.loader_errors(["skills/foo/SKILL.md", "skills/.archived/foo/SKILL.md"])
    assert not [e for e in errors if ".archived" in e], (
        f"the load check invented a skill slug out of a dot-directory: {errors}")


@pytest.mark.parametrize("status", sorted(_QUARANTINE_STATUSES))
def test_a_quarantine_status_is_not_reported_as_a_broken_skill(livevalidatorvault, status):
    """Clause 2. One test per status in the loader's own set, not a hand-picked one.

    `_QUARANTINE_STATUSES` is imported rather than repeated, so a sixth status
    added to the loader is covered here the same day it is added there; a
    hard-coded list would silently stop covering the set it is meant to pin.
    """
    (livevalidatorvault / "skills" / "foo" / "SKILL.md").write_text(
        f"---\nname: foo\nstatus: {status}\n---\n# foo\n", encoding="utf-8")
    errors = V.loader_errors(["skills/foo/SKILL.md"])
    assert not [e for e in errors if "skills/foo" in e], (
        f"`status: {status}` is a deliberate quarantine — present on disk, pulled "
        f"from retrieval — not a load failure: {errors}")


def test_the_loosened_check_still_refuses_a_skill_directory_with_no_body(livevalidatorvault):
    """Clause 3. Loosening must not empty the check: one call, both verdicts.

    The archived rename from clause 1 and a live `skills/bar/` with no `SKILL.md`
    are judged together, so a fix that made the predicate vacuous would have to
    fail here rather than in the clause-1 test alone.
    """
    _retire_by_rename(livevalidatorvault)
    (livevalidatorvault / "skills" / "bar").mkdir()
    errors = V.loader_errors([
        "skills/foo/SKILL.md", "skills/.archived/foo/SKILL.md", "skills/bar/SKILL.md"])
    assert [e for e in errors if e.startswith("skills/bar:")] == ["skills/bar: no SKILL.md"], (
        f"a live skill directory with no SKILL.md must still be refused: {errors}")
    assert not [e for e in errors if ".archived" in e], errors


def test_a_retirement_by_rename_lands_as_one_commit(livevalidatorvault):
    """Clause 4. `land()`, end to end: one sha, destination present, source gone.

    `loader_errors` unmocked means this exercises the same subprocess the MCP
    tool does — a pass here is `automod_vault_land` accepting a skill retirement,
    which is the acceptance check itself. No `item_id`: the second reader is not
    what is under test, and a land bound to no item is the route the module CLI
    and `scripts/autoresearch/promote.py` take.
    """
    before = int(git(livevalidatorvault, "rev-list", "--count", "HEAD").stdout.strip())
    _retire_by_rename(livevalidatorvault)
    out = V.land(["skills/foo/SKILL.md", "skills/.archived/foo/SKILL.md"],
                 "archive the foo skill")
    assert out["ok"], out
    assert int(git(livevalidatorvault, "rev-list", "--count", "HEAD").stdout.strip()) == before + 1, \
        "the retirement must be one commit, so one revert undoes it"
    assert (livevalidatorvault / "skills" / ".archived" / "foo" / "SKILL.md").is_file()
    assert not (livevalidatorvault / "skills" / "foo" / "SKILL.md").exists()
    shown = git(livevalidatorvault, "show", "--name-only", "--format=", out["commit"]).stdout
    assert {"skills/foo/SKILL.md", "skills/.archived/foo/SKILL.md"} <= set(shown.splitlines()), shown


# ── #1562: the loader child judges the checkout named by `LLOYD_HOME` ────────

def _fake_checkout(root: Path, name: str, *, prompt_len: int, real_layout: bool) -> Path:
    """A directory shaped like a Lloyd checkout for the one import the loader makes.

    Its `app/prompt_builder.py` returns a prompt `prompt_len` characters long: the
    loader's own threshold is 500, so below it is that child's "the system prompt
    failed to build" verdict and above it is its clean verdict.

    `real_layout=True` also writes `app/__init__.py`, which is what makes `app` a
    REGULAR package — the layout every real checkout has. A checkout without it is
    only a PEP 420 namespace portion, and a regular package anywhere on `sys.path`
    beats a namespace portion outright however early the portion sits. That
    asymmetry is the entire defect: the stub checkout the two red tests aim the
    loader at is a bare `app/` directory, while the gate's `PYTHONPATH` names a
    real checkout, so the child answered for the gate's tree, not the one it was
    pointed at.
    """
    checkout = root / name
    (checkout / "app").mkdir(parents=True)
    if real_layout:
        (checkout / "app" / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "app" / "prompt_builder.py").write_text(
        "def build_system_prompt(*a, **k):\n"
        f"    return {'x' * prompt_len!r}\n", encoding="utf-8")
    return checkout


def test_the_loader_subprocess_judges_the_checkout_named_by_loyd_home(
        livevalidatorvault, tmp_path, monkeypatch):
    """Clause: the fresh-interpreter check runs the loaders of `LLOYD_HOME`.

    Both halves run the REAL subprocess (the `livevalidatorvault` fixture leaves it
    unmocked, which is the only way to see what happens across that boundary) and
    aim it at one checkout while the caller's inherited `PYTHONPATH` names the
    other — the shape `scripts/automod/gate.py::_child_env` puts the suite in,
    where `PYTHONPATH` is the round's worktree.

    Half 1 is the one the defect fails: `LLOYD_HOME` is a stub checkout holding a
    prompt builder that cannot clear the 500-character threshold, `PYTHONPATH`
    names a real-layout checkout that can, and the verdict must be the refusal.
    Half 2 points the same two checkouts the other way round, so a loader child
    that refused unconditionally could not pass this test either.
    """
    stub = _fake_checkout(tmp_path, "stub-checkout", prompt_len=20, real_layout=False)
    builds = _fake_checkout(tmp_path, "real-layout-checkout", prompt_len=600, real_layout=True)
    assert not (stub / "app" / "__init__.py").exists() and (builds / "app" / "__init__.py").is_file()

    monkeypatch.setenv("PYTHONPATH", str(builds))
    monkeypatch.setattr(V, "LLOYD_HOME", stub)
    errors = V.loader_errors(["lloyd/SOUL.md"])
    assert [e for e in errors if "failed to build" in e], (
        f"the loader child reported for the checkout on PYTHONPATH ({builds}) instead "
        f"of the one named by LLOYD_HOME ({stub}), whose prompt builder returns 20 "
        f"characters against the loader's 500-character floor: {errors}")

    monkeypatch.setenv("PYTHONPATH", str(stub))
    monkeypatch.setattr(V, "LLOYD_HOME", builds)
    assert V.loader_errors(["lloyd/SOUL.md"]) == [], (
        "LLOYD_HOME holds a builder clearing the 500-character floor, so the verdict "
        "must be clean whatever the caller's PYTHONPATH names")


# ── #1360: an archive staged with `git mv` lands, and is graded as a move ────

ARCHIVE_PATHS = ["skills/foo/SKILL.md", "skills/.archived/foo/SKILL.md"]
LANDING_CLAUSE = ("The change is submitted through `automod_vault_land` as one call "
                  "naming exactly skills/foo/SKILL.md and skills/.archived/foo/SKILL.md.")


def _retire_by_git_mv(vault) -> None:
    (vault / "skills" / ".archived").mkdir(parents=True)
    assert git(vault, "mv", "skills/foo", "skills/.archived/foo").returncode == 0
    (vault / "skills" / ".archived" / "foo" / "SKILL.md").write_text(
        "---\nname: foo\nstatus: archived\n---\n# foo\n")


@pytest.mark.parametrize("shape", ["git_mv", "plain_mv"])
def test_both_archive_shapes_land_as_one_met_commit(livevalidatorvault, items, monkeypatch, shape):
    """Clauses 1, 2 and 4. `git mv` made `git add -A -- <source>` fatal after a
    passing review; plain `mv` survived. Both land now, the reviewer sees a
    rename for the staged shape, and the landing clause derives `met` because
    the source path counts as covered by the rename."""
    vault = livevalidatorvault
    write_item(items, 409, [LANDING_CLAUSE])
    seen = {}

    def _grader(**kw):
        seen["diff"] = kw["diff"]
        return ("pass", "moved", [{"clause": 1, "verdict": "post_landing",
                                   "subject": "landing"}])

    monkeypatch.setattr(V, "GRADER", _grader)
    before = int(git(vault, "rev-list", "--count", "HEAD").stdout.strip())
    (_retire_by_git_mv if shape == "git_mv" else _retire_by_rename)(vault)
    out = V.land(ARCHIVE_PATHS, "archive foo (#409)", item_id=409)
    assert out["ok"], out
    assert int(git(vault, "rev-list", "--count", "HEAD").stdout.strip()) == before + 1
    status = git(vault, "show", "--name-status", "--format=", out["commit"]).stdout
    assert "skills/.archived/foo/SKILL.md" in status, status
    assert git(vault, "cat-file", "-e", f"{out['commit']}:skills/foo/SKILL.md").returncode != 0
    assert not (vault / "skills" / "foo" / "SKILL.md").exists()
    if shape == "git_mv":
        assert "rename from skills/foo/SKILL.md" in seen["diff"], seen["diff"]
        assert "rename to skills/.archived/foo/SKILL.md" in seen["diff"], seen["diff"]
        assert "+++ new file" not in seen["diff"], "a staged destination shown twice"
    assert out["landing_clauses"] == [{"clause": 1, "verdict": "met",
                                       "commit": out["commit"]}]


def test_a_named_path_in_neither_head_nor_the_commit_is_still_unmet(livevalidatorvault, items,
                                                                     monkeypatch):
    """Clause 3. Tolerating a vanished pathspec must not hide a dropped path."""
    vault = livevalidatorvault
    write_item(items, 410, [LANDING_CLAUSE])
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("pass", "moved", []))
    _retire_by_git_mv(vault)
    ghost = "skills/ghost/SKILL.md"
    out = V.land(ARCHIVE_PATHS + [ghost], "archive foo (#410)", item_id=410)
    row = out["landing_clauses"][0]
    assert row["verdict"] == "unmet" and ghost in row["note"], row
    assert "skills/foo/SKILL.md" not in row["note"], "the renamed source is covered"


# Two real gate-role sections with nothing but gate bytes beside them: the gate stack is
# ~100% of the contract, so `check_contract` refuses it on the ceiling. The headings carry
# their trigger text because `check_contract` demands the named tokens live under them —
# a fixture that tripped only the ratio would be a fixture the ratio check could delete.
CONTRACT_HEAVY_SOUL = """---
type: note
---
# Heavy

## ZERO PREAMBLE (never opens with a filler token)
no prose before the first tool call or answer
## BLOCK SIGNAL
the whole response is the block signal with nothing after it
""" + "\n".join(f"- never do this thing number {i}" for i in range(6))


def _contract_errors_in_a_fresh_interpreter(paths, vault_dir):
    """Run the vault route's contract check the way its sibling check is run: a separate
    interpreter, cwd at the repo root, `LLOYD_VAULT` pointing at `vault_dir`."""
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path.cwd()))\n"
        "from scripts.automod import vault_round as V\n"
        "print(json.dumps(V.contract_errors(json.loads(sys.argv[1]))))\n"
    )
    import json as _json
    import os
    import sys as _sys
    env = dict(os.environ, LLOYD_VAULT=str(vault_dir))
    return subprocess.run(
        [_sys.executable, "-c", script, _json.dumps(paths), str(vault_dir)],
        cwd=str(V.LLOYD_HOME), capture_output=True, text=True, timeout=180, env=env)


def test_the_contract_refusal_survives_the_vault_routes_fresh_interpreter(vault, monkeypatch):
    """#789 widened `prompt_surface`, and the vault route is the one caller that would
    swallow the widening going wrong.

    The finding this answers said the route runs `check_contract` in a fresh interpreter.
    Half true, and the false half matters: `contract_errors` imports `prompt_surface`
    **in-process** (`vault_round.py:226-232`); the fresh interpreter in this module is
    `loader_errors` (`:199-202`), whose script imports `prompt_builder`, the skills
    loader and the autonomy parser and never reaches `prompt_surface`. So the coverage
    the finding wanted is real but had no existing shape: `contract_errors` is wrapped in
    `try/except Exception` that reports ANY failure as `prompt_surface unavailable`, so
    an import error, a circular import, or a bug in the new `contract_shape`/`rising_run`
    path all arrive as the same generic refusal — and a refusal whose text names the
    wrong cause is the failure mode this loop keeps hitting. Run the route's own function
    in a clean interpreter (nothing imported by pytest, nothing on `sys.path` but the repo
    root, exactly how `loader_errors` spawns one) and require the *ceiling* refusal to come
    back, with the swallow absent and the verdict identical to this process's.
    """
    import json

    # The `vault` fixture mkdirs the trees `check_scope` lets a round write into; the
    # contract's own file lives beside them, so its directory is made here.
    soul = vault / "lloyd" / "SOUL.md"
    soul.parent.mkdir(parents=True, exist_ok=True)
    soul.write_text(CONTRACT_HEAVY_SOUL, encoding="utf-8")

    fresh = _contract_errors_in_a_fresh_interpreter(["lloyd/SOUL.md"], vault)
    assert fresh.returncode == 0, (fresh.stdout[-500:], fresh.stderr[-1500:])
    assert "Traceback" not in fresh.stderr, fresh.stderr[-800:]
    errs = json.loads(fresh.stdout.strip().splitlines()[-1])
    assert any("over the 50% ceiling" in e for e in errs), (
        f"a gate-stack-heavy SOUL.md was not refused across the boundary: {errs}")
    assert not [e for e in errs if "prompt_surface unavailable" in e], (
        f"the widened module failed to import in a clean interpreter and the route "
        f"reported it as the generic swallow: {errs}")

    # Same bytes, same verdict in this process: the two sides must not be running two
    # copies of the arithmetic, which is what made the guardian's two file counts
    # disagree by 2,000 and false-tripped a rollback.
    monkeypatch.setattr(V, "VAULT", vault)
    assert V.contract_errors(["lloyd/SOUL.md"]) == errs

    # The ceiling refusal is a verdict about bytes, not this fixture refusing everything:
    # the same route over a short SOUL.md whose gate stack is ~0 % comes back with a
    # different complaint — the gate-role sections it is missing (`check_contract` also
    # requires every role to survive, and this stub has none). Asserted as the absence of
    # the ceiling refusal rather than an empty list, because writing a fully compliant
    # contract here would be a second copy of the guard suite's `GOOD_CONTRACT`, and a
    # copy that drifts is a control that quietly stops controlling.
    soul.write_text("# Light\n\nBe useful, and say so plainly.\n", encoding="utf-8")
    light = json.loads(_contract_errors_in_a_fresh_interpreter(["lloyd/SOUL.md"], vault)
                       .stdout.strip().splitlines()[-1])
    assert not [e for e in light if "over the 50% ceiling" in e], light


# ── #1868: the vault round's seam decision reads the configured policy ────────
#
# `grade_vault` used to call `decide(parsed, [])`, so `attempt` and `policy`
# fell to their defaults (`1` and `first`) whatever config.yaml said. With the
# shipped `seams_block: never` that refused rounds the operator had ruled a
# seam could never refuse: 4 `vault_review` ledger rows carry the blocking
# spelling after that setting shipped, and #1621's pair (attempt 1 AND attempt
# 2) is the proof both arguments were defaulted, because `seams_block` returns
# False for any policy at attempt 2. These tests run the REAL `grade_vault` with
# only `run_grader` replaced, so the policy has to be read through
# `seams_policy()` → the shared CONFIG, exactly as the code rung reads it.

#: The one grader object these tests need: both clauses met, and the only
#: defect a testable, actionable, never-before-seen seam. The default readings of
#: `parse_review` are what makes it blocking-shaped (#84 in
#: test_review_grader_policy.py pins them), so this is a real grader answer and
#: not a fixture invented to be refused.
GRADER_SAYS_MET_WITH_ONE_SEAM = {
    "premise": "sound", "summary": "content is fine", "test_honesty": [],
    "seams_unverified": [{"seam": "the dashboard reads the ledger the job writes",
                          "testable_before_landing": True,
                          "actionable_in_round": True, "same_as_prior": False}],
    "clauses": [
        {"clause": 1, "verdict": "met", "evidence_path": "skills/foo/SKILL.md",
         "evidence_line": 1, "test_node_id": "", "how_verified": "read", "note": "named"},
    ]}


def _seams_policy_set(monkeypatch, policy: str) -> None:
    """Set `automod.review.seams_block` through the config `seams_policy()` reads,
    not by patching the function: the bug was a caller bypassing that reader, so a
    test that patched the reader could not fail it."""
    from app.config import CONFIG
    automod = dict(CONFIG.get("automod") or {})
    review = dict(automod.get("review") or {})
    review["seams_block"] = policy
    automod["review"] = review
    monkeypatch.setitem(CONFIG, "automod", automod)


def _grade_once(items, monkeypatch, *, attempt: int):
    """One real `grade_vault` over a one-clause item, grader answer fixed."""
    import scripts.automod.review as RV
    write_item(items, 411, ["the skill names the retry rule"])
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": True, "structured": GRADER_SAYS_MET_WITH_ONE_SEAM})
    return RV.grade_vault(item_id=411, paths=["skills/foo/SKILL.md"], diff="x",
                          attempt=attempt)


def test_the_shipped_seams_policy_means_a_vault_round_cannot_refuse_on_a_seam(vault, items,
                                                                              monkeypatch):
    """Clause 1, on the shipped setting: config.yaml says `never`, so a vault
    round whose clauses are met and whose only defect is a testable, actionable
    seam must not be a `retry` — and the seam must still be in the findings, in
    the advisory spelling, because advisory is a report and not a deletion.
    """
    _seams_policy_set(monkeypatch, "never")
    kind, findings, clauses = _grade_once(items, monkeypatch, attempt=1)
    assert kind == "pass", f"`seams_block: never` refused anyway: {kind} — {findings}"
    assert "seam unverified (advisory under seams_block=never)" in findings, findings
    assert "the dashboard reads the ledger the job writes" in findings, findings
    assert [c["verdict"] for c in clauses] == ["met"], clauses


def test_a_vault_round_decides_a_seam_by_the_attempt_it_is_given(vault, items, monkeypatch):
    """Clause 2: `first` blocks attempt 1 and advises on attempt 2, so threading
    is demonstrably real and not a hard-wired `never`. Same grader answer, same
    policy, only the attempt differs — which is the only shape that distinguishes
    `attempt=attempt` from no argument at all."""
    _seams_policy_set(monkeypatch, "first")
    first_kind, first_findings, _ = _grade_once(items, monkeypatch, attempt=1)
    assert first_kind == "retry", first_findings
    assert first_findings.startswith("seam unverified: "), first_findings

    second_kind, second_findings, _ = _grade_once(items, monkeypatch, attempt=2)
    assert second_kind == "pass", (
        f"attempt 2 refused on a policy that permits refusal on attempt 1 only: "
        f"{second_findings}")
    assert "seam unverified (attempt 2, not refusing again)" in second_findings, second_findings


def test_the_round_tells_the_grader_which_attempt_this_is(vault, items, monkeypatch):
    """Clause 3, across the `land()` → `_vault_review` → `GRADER` boundary: the
    attempt count `land()` already computes for the ledger row has to reach the
    grader, or a round graded after one refusal is decided as if it were the
    first. A prior blocking row for the item is what makes this attempt 2, which
    is how the count is produced in production."""
    seen: list[int] = []

    def _grader(**kw):
        seen.append(kw["attempt"])
        return ("pass", "content met, no advisories", [])

    monkeypatch.setattr(V, "GRADER", _grader)
    write_item(items, 412, ["the skill names the retry rule"])
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v2\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v2 (#412)", item_id=412)
    assert out["ok"], out
    assert seen == [1], seen

    S.append_event({"event": "vault_review", "item_id": 412, "kind": "retry",
                    "blocking": True, "attempt": 1, "findings": "seam unverified: x"})
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v3\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v3 (#412)", item_id=412)
    assert out["ok"], out
    assert seen == [1, 2], (
        f"the second grading was decided as attempt {seen[-1]}; the grader needs the "
        "same count the ledger row would carry")


def test_an_advisory_surviving_a_pass_is_recorded_on_the_landing(vault, items, monkeypatch):
    """Clause 4: making a seam advisory is only a report if the report is kept, in
    both places a reader of this item has. A refused round writes `findings` on its
    own row and then reverts, so a PASSING round's findings had nowhere to go at all
    — they were computed and dropped at the call site. This lands a real
    `grade_vault` pass carrying a real advisory seam and reads it back off the
    `vault_land` event AND off `backlog.vault_review_outcome`, the record that
    closes a vault item: an advisory on an event nobody queries is the same as a
    deleted rail."""
    import scripts.automod.backlog as B
    import scripts.automod.review as RV
    _seams_policy_set(monkeypatch, "never")
    write_item(items, 413, ["the skill names the retry rule"])
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": True, "structured": GRADER_SAYS_MET_WITH_ONE_SEAM})
    monkeypatch.setattr(V, "GRADER", RV.grade_vault)
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo retry rule\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo names the retry rule (#413)", item_id=413)
    assert out["ok"] and out["review"] == "pass", out
    assert "seam unverified (advisory under seams_block=never)" in out["review_findings"], out
    ev = _events("vault_land")[-1]
    assert "the dashboard reads the ledger the job writes" in ev["review_findings"], ev
    outcome = B.vault_review_outcome(S.LEDGER_PATH, [out["commit"]])
    assert outcome and outcome["acceptance"] == "met", outcome
    assert "the dashboard reads the ledger the job writes" in outcome["review_findings"], (
        f"the record that closes the item dropped the advisory the review passed: {outcome}")


def test_a_pass_with_nothing_to_report_records_no_findings_field(vault, items, monkeypatch):
    """The positive control for the clause-4 key's absence rule: a pass whose
    findings string is EMPTY — the reviewer said nothing beyond its verdict — gets
    no `review_findings` on the event or on the outcome, so a reader can tell "the
    reviewer had no finding" from "the field was written empty", and the landing
    rows this key predates stay exactly what they were. A pass that DID say
    something writes it verbatim, summary or not: the key is "the reviewer's report
    of a round it passed", not "the advisories only".
    """
    import scripts.automod.backlog as B
    monkeypatch.setattr(V, "GRADER", lambda **kw: (
        "pass", "", [{"clause": 1, "verdict": "met"}]))
    write_item(items, 414, ["the skill names the retry rule"])
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v9\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v9 (#414)", item_id=414)
    assert out["ok"] and "review_findings" not in out, out
    ev = _events("vault_land")[-1]
    assert "review_findings" not in ev, ev
    outcome = B.vault_review_outcome(S.LEDGER_PATH, [out["commit"]])
    assert outcome and "review_findings" not in outcome, outcome
