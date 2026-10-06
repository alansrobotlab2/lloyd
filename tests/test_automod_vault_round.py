"""The vault route: validate → commit only these paths → revert on failure.

The vault is a live, shared, always-dirty tree with no worktree, so "nothing
lands unverified" has to be enforced after the edit, not before it.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from agent_mcp.skills import _QUARANTINE_STATUSES
from app.harness.policy import SCHEDULE_STATE_FIELDS
from scripts.automod import state as S, vault_guards as VG, vault_round as V


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


# ── #2175: a refusal that says WHO ALREADY TOOK THE FILE, and leaves a row ───
#
# A concurrent job's pre-flight snapshot (`vault-commit.sh` with no pathspec, its
# documented whole-tree mode) commits whatever is dirty at its instant, including
# another job's in-flight file, under the OTHER job's --author and Job: trailer. The
# author's own `automod_vault_land` then said only "nothing to commit on those paths"
# — true of the index, silent about the sha, and it wrote no ledger row at all, so of
# the 390 `vault_land` rows in promotions.jsonl none records a refusal of this kind.
# The fix is a pointer, not a claim about which of the two readings applies.

def _sweep_commit(vault, rel: str, text: str, author: str) -> str:
    """Simulate the sweep: a second job's commit takes this job's in-flight file."""
    (vault / rel).write_text(text)
    git(vault, "add", "-A", "--", rel)
    git(vault, "commit", "-q", "--author", author, "-m",
        "autonomy-data-pipeline: pre-flight 2026-10-04 (unattributed dirty state)")
    return git(vault, "rev-parse", "HEAD").stdout.strip()


def test_the_refusal_names_the_commit_that_took_the_path(vault):
    """Clause 1, in the shape the item was filed for: the job writes a skill, a
    concurrent snapshot commits it, and the job's own land must answer with a sha a
    reader can go and read, not with a bare "nothing to commit"."""
    rel = "skills/runtime-store-schema-probe/SKILL.md"
    (vault / "skills" / "runtime-store-schema-probe").mkdir(parents=True)
    (vault / rel).write_text("---\nname: runtime-store-schema-probe\n---\n# probe\n")
    swept = _sweep_commit(vault, rel, "---\nname: runtime-store-schema-probe\n---\n"
                                   "# probe\n",
                          "lloyd-autonomy-data-pipeline <adb@jobs.lloyd.local>")
    with pytest.raises(V.VaultRoundError) as got:
        V.land([rel], "consolidation: runtime-store-schema-probe")
    text = str(got.value)
    short = swept[:7]
    assert short in text, (short, text)
    assert rel in text, text
    # The named sha is the commit that last committed THIS path, from the same
    # command the docstring names — not the vault's first commit, not HEAD by luck.
    assert git(vault, "log", "-1", "--format=%h", "--", rel).stdout.strip() == short
    assert git(vault, "cat-file", "-e", swept).returncode == 0
    # And the sentence says what the sha IS without asserting which reading applies:
    # either somebody else's snapshot took this job's content, or there was no change.
    assert "a commit not its own took it" in text, text
    assert "unattributed dirty state" in text, text


def test_the_refusal_names_each_path_with_its_own_commit(vault):
    """Two named paths, two different taking commits: the answer pairs them, so a
    reader can tell which file went into which sweep."""
    git(vault, "add", "-A", "--", "backlog/9-item.md")
    git(vault, "commit", "-q", "--author", "job-a <a@jobs.lloyd.local>",
        "-m", "job a took the item")
    idx = "memory/learnings/skills-index.md"
    (vault / "memory" / "learnings").mkdir(parents=True, exist_ok=True)
    (vault / idx).write_text("# index\n")
    item_sha = git(vault, "log", "-1", "--format=%h", "--", "backlog/9-item.md").stdout.strip()
    _sweep_commit(vault, idx, "# index\n", "job b <b@jobs.lloyd.local>")
    idx_sha = git(vault, "log", "-1", "--format=%h", "--", idx).stdout.strip()
    with pytest.raises(V.VaultRoundError) as got:
        V.land(["backlog/9-item.md", idx], "two paths")
    text = str(got.value)
    assert f"backlog/9-item.md -> {item_sha}" in text, text
    assert f"{idx} -> {idx_sha}" in text, text


def test_the_refusal_still_carries_the_head_substring_the_promoter_matches(vault):
    """Clause 1's other half: the phrases below are additions. `promote.py` classifies
    a repeated rollback as `no_change` by `"nothing to commit" in str(exc)`, so the
    extended text has to still open with what it matched before."""
    rel = "skills/foo/SKILL.md"
    swept = _sweep_commit(vault, rel, "---\nname: foo\n---\n# foo changed\n",
                          "lloyd-autonomy-data-pipeline <adb@jobs.lloyd.local>")
    with pytest.raises(V.VaultRoundError, match="nothing to commit") as got:
        V.land([rel], "no-op again")
    assert str(got.value).startswith(V.NOTHING_TO_COMMIT), str(got.value)
    assert swept[:7] in str(got.value)


def test_a_refused_land_writes_exactly_one_ok_false_row_with_the_shas(vault):
    """Clause 3: every other refusal on this path already wrote a row; this one wrote
    nothing. Now the ledger can answer "did a land commit nothing today, and what took
    its content?" without the caller's run record."""
    before = len(_events("vault_land"))
    rel = "memory/learnings/skills-usage.jsonl"
    (vault / "memory" / "learnings").mkdir(parents=True, exist_ok=True)
    (vault / rel).write_text('{"skill": "x"}\n')
    swept = _sweep_commit(vault, rel, '{"skill": "x"}\n',
                          "lloyd-autonomy-data-pipeline <adb@jobs.lloyd.local>")
    with pytest.raises(V.VaultRoundError):
        V.land([rel], "usage row", session_id="sess-83")
    rows = _events("vault_land")
    assert len(rows) == before + 1, f"wrote {len(rows) - before} rows, expected exactly one"
    row = rows[-1]
    assert row["ok"] is False, row
    assert row["paths"] == [rel], row
    assert "nothing to commit on those paths" in row["errors"][0], row
    short = git(vault, "log", "-1", "--format=%h", "--", rel).stdout.strip()
    assert row["held_by"] == {rel: short}, row
    assert short in row["errors"][0], row
    assert row["session_id"] == "sess-83", row
    # A land that made no commit must not appear to have made one: the passing row's
    # `commit` key is absent here, so no reader mistakes the sweep's sha for this land.
    assert "commit" not in row, row
    # The row points at a commit whose author is the OTHER job — which is the whole
    # defect this refusal now makes legible from the ledger alone.
    assert git(vault, "log", "-1", "--format=%an", swept).stdout.strip() == \
        "lloyd-autonomy-data-pipeline", row


def test_a_path_with_no_commit_in_history_is_named_without_a_sha(vault):
    """Clause 4: git has never seen this path, so there is no sha to name and none is
    invented. The call still refuses — an unlandable path is not a success."""
    ghost = "backlog/never-committed-item.md"
    with pytest.raises(V.VaultRoundError) as got:
        V.land([ghost], "land a path git has never seen")
    text = str(got.value)
    assert text.startswith(V.NOTHING_TO_COMMIT), text
    assert ghost in text, text
    assert "none of them has a commit in HEAD's history" in text, text
    # No fabricated pointer: every hex token of sha length in the message must be a
    # real object in this repo. An empty answer names none, and none may appear.
    for tok in _hex_tokens(text):
        assert git(vault, "cat-file", "-e", tok).returncode == 0, tok


def test_a_git_error_is_absence_and_never_a_guess(vault, monkeypatch):
    """The one survivor of a mutant sweep: swap the lookup's `continue` for a placeholder
    and every other node stays green, because on a real tree `_git` never raises. So the
    docstring's claim that a git error is absence too is graded here — git answers a
    `log` query with an exception, and the refusal must name NO sha for that path, name
    the path in its no-sha set, and still refuse."""
    rel = "backlog/9-item.md"
    real = V._git

    def die_on_log(*args):
        if args and args[0] == "log":
            raise RuntimeError("git: unable to read commit graph")
        return real(*args)

    monkeypatch.setattr(V, "_git", die_on_log)
    with pytest.raises(V.VaultRoundError) as got:
        V.land([rel], "lookup cannot run")
    text = str(got.value)
    assert text.startswith(V.NOTHING_TO_COMMIT), text
    assert not _hex_tokens(text), f"a sha was named from a failed lookup: {text}"
    assert f"{rel} -> (none)" not in text, text
    assert "none of them has a commit in HEAD's history" in text, text
    assert _events("vault_land")[-1]["held_by"] == {}, _events("vault_land")[-1]


# ── #2175 clause 5: the bytes behind "the ledger has no refusal of this kind" ──
#
# The report this item quotes — every `vault_land` row of 2026-10-04 is `ok: true`, so
# task-83's three refusals are nowhere in the ledger — was read out of the LIVE ledger,
# which another job appends to between one reading and the next. #2042's own finding is
# why a dated copy sits in `backlog/data/`: the live file is not a witness. This extract
# is every `vault_land` row dated 2026-10-04 whose `created_at` is at or before
# 08:31:40Z — twelve rows, copied out byte for byte at 10:50Z.
#
# The path is dated (#2178). #2175's clause 5 named `backlog/data/promotions.jsonl`, which
# is the path the retention sweep retired: #2054's para records that "the promotions ledger
# is no longer mirrored into the vault", and #2064 deleted the 33,113,707-byte copy at
# vault `ffc04ce5`. `tests/test_failure_ledger_witness.py` fails its first node while ANY
# file stands on that path, so the land that followed clause 5 literally is what left main
# red at `4e6bae21` — the witness and the retire were written by two items that never read
# each other. Same twelve bytes, at a name in the shape every other witness in that
# directory has, and no retired path touched.

WITNESS_2175_VAULT_PATH = "backlog/data/2026-10-04.2175-vault-land-witness.jsonl"


@pytest.mark.live_vault
def test_the_vault_copy_shows_a_day_of_lands_with_no_refusal_of_this_kind():
    """Re-derived from committed bytes, not from a live file that keeps growing.

    `wc -l < backlog/data/2026-10-04.2175-vault-land-witness.jsonl` is 12 — the figure
    this item's triage quotes — and it is the twelve `vault_land` rows the live ledger held
    for 2026-10-04
    up to 08:31:40Z, when that count was taken. Every one is `ok: true` and none carries
    a `held_by` key, which is the whole report in bytes: three task-83 lands refused with
    "nothing to commit on those paths" and not one of them is in the ledger.

    The cut at 08:31:40Z is the count's own timestamp, not a filter that flatters the
    claim: the live file has gained three more rows since (items 2172, 2177, 2176, all
    `ok: true`), and a copy that grew with it would stop being the twelve the triage
    counted. What this extract is evidence of is the ABSENCE of an `ok: false` row, and
    the node asserts exactly that, so the next reader re-derives the finding rather than
    trusting a line count that a landing job moves every few minutes.

    Read from disk if it is there, else from the last commit that touched the path: the
    sweep this item describes deletes vault files and commits the deletion, and a check
    that skipped once the file vanished would be a witness the defect could silence.
    """
    import json
    text = _witness_2175_bytes()
    if text is None:
        pytest.skip(f"{WITNESS_2175_VAULT_PATH} is neither on disk nor in the vault's "
                    "history, so the copy is unreadable here")
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    assert len(text.splitlines()) == 12, "the file's line count is what wc -l reports"
    assert len(rows) == 12, f"the extract holds {len(rows)} rows, not the 12 triage counted"
    assert all(r.get("event") == "vault_land" for r in rows), "not the extract asked for"
    dates = [str(r.get("created_at", "")) for r in rows]
    assert all(d.startswith("2026-10-04") for d in dates), dates[0]
    assert min(dates) == "2026-10-04T00:51:17Z" and max(dates) == "2026-10-04T08:31:40Z", \
        "the extract is not the twelve rows up to the count's own timestamp"
    assert all(r.get("ok") is True for r in rows), \
        "a refusal is in the extract, so it no longer shows a day with none"
    assert not [r for r in rows if "held_by" in r], \
        "the extract postdates #2175 and cannot show the pre-fix absence"


def _witness_2175_bytes() -> str | None:
    """The committed extract, from disk or from the last sha that wrote the path."""
    rel = WITNESS_2175_VAULT_PATH
    root = V.VAULT
    if not (root / ".git").exists():
        return None
    disk = root / rel
    if disk.is_file():
        return disk.read_text(encoding="utf-8", errors="replace")
    seen = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%H",
                           "--", rel], capture_output=True, text=True, check=False)
    sha = (seen.stdout or "").strip()
    if seen.returncode != 0 or not sha:
        return None
    got = subprocess.run(["git", "-C", str(root), "show", f"{sha}:{rel}"],
                         capture_output=True, text=True, check=False)
    return None if got.returncode != 0 else got.stdout



def test_a_mixed_batch_names_the_sha_it_has_and_none_for_the_path_it_lacks(vault):
    """The same rule inside a batch: one path HEAD holds (named with its sha), one git
    never committed (named with nothing after it), and the no-sha set is stated."""
    git(vault, "add", "-A", "--", "backlog/9-item.md")
    git(vault, "commit", "-q", "--author", "job-a <a@jobs.lloyd.local>",
        "-m", "job a took the item")
    item_sha = git(vault, "log", "-1", "--format=%h", "--", "backlog/9-item.md").stdout.strip()
    (vault / "memory" / "learnings").mkdir(parents=True, exist_ok=True)
    ghost = "memory/learnings/never-indexed.md"
    with pytest.raises(V.VaultRoundError) as got:
        V.land(["backlog/9-item.md", ghost], "mixed batch")
    text = str(got.value)
    assert f"backlog/9-item.md -> {item_sha}" in text, text
    assert f"{ghost} -> (none)" in text, text
    assert "With no commit named for it, and none invented" in text, text
    for tok in _hex_tokens(text):
        assert git(vault, "cat-file", "-e", tok).returncode == 0, tok

def _hex_tokens(text: str) -> list[str]:
    """Every run of 7+ hex characters in a refusal — the shape a named sha has, so the
    no-fabrication check has something to check."""
    import re
    return [t for t in re.findall(r"\b[0-9a-f]{7,40}\b", text)]



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


# ── #1987: a refused vault clause is on the row, and the amend tool is honest ──

def test_a_blocking_vault_review_row_carries_the_per_clause_verdicts(vault, items, monkeypatch):
    """0 of 35 refusal rows carried `clauses`, so `met` was the only verdict any
    vault row ever held and the refused clause existed in prose alone."""
    write_item(items, 1987, ["the skill names the retry rule", "the skill names the cap"])
    rows = [{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "unmet"}]
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("retry", "clause 2 unmet", rows))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo retry rule\n")
    with pytest.raises(V.VaultRoundError, match="review sent it back"):
        V.land(["skills/foo/SKILL.md"], "skill: foo (#1987)", item_id=1987)
    refusal = _events("vault_review")[-1]
    assert refusal["blocking"] is True and refusal["kind"] == "retry"
    assert [(c["clause"], c["verdict"]) for c in refusal["clauses"]] == [(1, "met"), (2, "unmet")]
    assert "round_id" not in refusal, "a vault landing has no round; none is invented"
    # The same key, in the same shape, as the non-blocking row writes.
    monkeypatch.setattr(V, "GRADER", lambda **kw: (
        "pass", "", [{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "met"}]))
    V.land(["skills/foo/SKILL.md"], "skill: foo (#1987)", item_id=1987)
    passed = _events("vault_review")[-1]
    assert passed["blocking"] is False
    assert {type(c) for c in passed["clauses"]} == {type(c) for c in refusal["clauses"]} == {dict}
    assert set(passed["clauses"][0]) >= {"clause", "verdict"} <= set(refusal["clauses"][0])
    # The attempt counter reads blocking rows only and is unmoved by the new key.
    assert V._vault_review_attempts(1987) >= 1


def test_the_amend_tool_names_the_working_route_for_a_vault_landing(tmp_path, monkeypatch):
    import agent_mcp.automod as T
    import scripts.automod.backlog as B
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    out = T._amend_clause("vault-land-1886", 4, "narrowed", "the clause cannot be met")
    assert B.VAULT_NO_AMENDMENT_ROUTE in out["error"]
    assert "edit the clause on the backlog item" in out["error"] and "blocker" in out["error"]
    assert out["error"] != "no run spec for vault-land-1886"
    assert not out["error"].startswith("no run spec for")


# --------------------------------------------------------------------------- #
# #2036: a vault land cannot commit prose that the code tree's own guards
# contradict.
#
# Both trees here are throwaways. `guard_tree` is a code checkout holding one
# vault-reading guard in its default selection; `vault` (the module's own
# fixture) is a git repo that behaves like ~/obsidian. The guard counts the store
# lines a script prints against the count the prose states — the same shape as
# the two `tests/test_retention_sweep.py` nodes #1975 left red — and it names the
# vault through `LLOYD_VAULT_ROOT`, the knob `app/data_root.py:183` honours and
# the one `vault_guards._run_selection` sets. Neither ~/lloyd nor ~/obsidian is
# touched: `agreement` is handed explicit roots and runs the same subprocess
# production runs.
# --------------------------------------------------------------------------- #

#: The guard's fixture source. It reads the vault through the environment
#: (`vault_root()` in the real guards, `LLOYD_VAULT_ROOT` here — the same
#: resolution, `app/data_root.py:183`), and its assert message carries both
#: counts, so a refusal that quotes the guard's output names the stated count
#: without this module re-deriving a number from prose.
GUARD_SRC = '''"""A vault-reading guard, shaped like the two #1975 left red."""
import os
from pathlib import Path

import pytest


def _store_lines():
    return [l for l in Path("store_lines.py").read_text().splitlines() if l.strip()]


@pytest.mark.skipif(os.environ.get("GUARD_FORCE_SKIP") == "1",
                    reason="fixture switch: the denominator can be driven to zero")
def test_the_prose_states_the_count_the_script_prints():
    text = (Path(os.environ["LLOYD_VAULT_ROOT"]) / "skills" / "foo" / "SKILL.md").read_text()
    stated = [int(l.split()[-1]) for l in text.splitlines() if l.startswith("stores: ")]
    assert len(stated) == 1, f"the skill states {len(stated)} counts"
    assert stated[0] == len(_store_lines()), f"prose {stated[0]}, script {len(_store_lines())}"
'''

#: A guard that WRITES into the vault it is shown, then fails on the count. The
#: probe runs a selection only to make a judgement, but `land()` calls it with
#: its own edit already sitting in the vault working tree and commits that tree
#: afterwards — so a selection handed `~/obsidian` itself would let one mutating
#: guard add unreviewed bytes to the commit the route came only to judge. It
#: fails against the proposed vault and passes against the pre-land one, so both
#: runs happen and both are held to this.
GUARD_SRC_WRITES = '''"""A vault-reading guard that writes into the vault it is shown."""
import os
from pathlib import Path


def test_a_guard_that_mutates_the_vault_it_judges():
    v = Path(os.environ["LLOYD_VAULT_ROOT"])
    (v / "PROBED").write_text("written by the probe")
    text = (v / "skills" / "foo" / "SKILL.md").read_text()
    stated = [int(l.split()[-1]) for l in text.splitlines() if l.startswith("stores: ")]
    assert stated == [4], f"prose {stated}, script 4"
'''

#: The two nodes clause 5 is about, named exactly as they exist in the repo.
RETENTION_NODES = (
    # The name #1975's own diff gives this guard: shipping the thirteenth store line
    # renames the node, and a citation of the twelve-store name would fail on a file
    # guarding the count harder than before, not less. The fourteenth store line
    # (#2225, landed by #2242 because its absence was the two failures this round is
    # named for) renamed it again — `…says_thirteen_stores…` → `…says_fourteen_stores…`
    # — so the citation moves with it, exactly as the sentence above rules. What this
    # tripwire holds is that each guard is PRESENT, undecorated and RUN by the gate's
    # own mark expr — not that its spelling is frozen. The second name needs no edit
    # for the same change, which is the difference between the two entries.
    "test_the_skill_says_fourteen_stores_and_its_table_has_a_row_per_report_line",
    "test_the_task_description_names_every_store_the_sweep_prints",
)

SKILL_AT_FOUR = "---\nname: foo\n---\nstores: 4\n"


def repo() -> Path:
    return Path(__file__).resolve().parent.parent


def _head(repo) -> str:
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def _prose(vault, body: str, *, commit: bool = False) -> None:
    """Write the skill's prose; `commit=True` makes it the vault's HEAD state.

    Default `False` because the land being staged is exactly that: a working-tree
    edit the route has not committed yet.
    """
    (vault / "skills" / "foo" / "SKILL.md").write_text(body)
    if commit:
        git(vault, "add", "-A")
        git(vault, "commit", "-q", "-m", "prose at vault HEAD")


def make_guard_tree(root: Path, *, src: str = GUARD_SRC,
                    store_lines: str = "draft\nup_next\nin_progress\ndone\n") -> Path:
    """A code checkout at HEAD: the store lines, and `src` as its one guard.

    A `pytest.ini` so the probe child's rootdir is this tree and nothing above
    it — the child runs with this tree as its cwd, exactly as the gate's base
    probe runs with the candidate checkout as its own.
    """
    (root / "tests").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\naddopts =\n")
    (root / "tests/test_guard.py").write_text(src)
    (root / "store_lines.py").write_text(store_lines)
    git(root.parent, "init", "-q", "-b", "main", str(root))
    git(root, "config", "user.email", "t@e.com")
    git(root, "config", "user.name", "t")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


@pytest.fixture
def guard_tree(tmp_path):
    """A code checkout at HEAD: four store lines, one vault-reading guard."""
    return make_guard_tree(tmp_path / "tree")


@pytest.fixture
def probed(tmp_path, monkeypatch):
    """Arm `land()` to probe given roots for real.

    `tests/conftest.py` defaults `LLOYD_VAULT_GUARD_PROBE` for the suite — the
    nesting rule that keeps ~25 `land()` calls from each starting a ~70 s pytest
    subprocess — so a test of the probe has to clear it as well as point the
    roots. Production never sets it, so every real landing is probed.
    """
    def arm(tree, vault, *, python=None, timeout=None):
        """Point the real probe at a fixture tree and vault.

        `python` and `timeout` are passed through to `agreement` itself, so the
        #2042 nodes can watch when the probe's child is spawned and hand it a
        budget it cannot spend, instead of standing in for the probe they are
        meant to be measuring.
        """
        monkeypatch.delenv(VG.NESTING_ENV, raising=False)
        real = VG.agreement
        extra = {} if timeout is None else {"timeout": timeout}
        monkeypatch.setattr(
            V.VG, "agreement",
            lambda *, paths, **ignored: real(paths=list(paths), live_root=tree,
                                             live_vault=vault,
                                             python=python or Path(sys.executable),
                                             scratch_parent=tmp_path / "probe-scratch",
                                             **extra))
    return arm


def test_a_land_whose_stated_count_the_tree_disagrees_with_is_refused_before_commit(
        vault, guard_tree, probed):
    """Clause 1. The prose says five, the tree prints four store lines.

    Asserted on the guard's own output inside the refusal, not on a count this
    module re-derived: `prose 5, script 4` is what the guard measured, and the
    node id plus the tree root and sha are what clause 1 asks the refusal to
    name.
    """
    _prose(vault, SKILL_AT_FOUR, commit=True)         # vault HEAD agrees with the tree
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n")
    head = _head(vault)
    with pytest.raises(V.VaultRoundError) as ei:
        V.land(["skills/foo/SKILL.md"], "#2036 five stores", item_id=None)
    msg = str(ei.value)
    assert "code agreement failed" in msg
    assert "tests/test_guard.py::test_the_prose_states_the_count_the_script_prints" in msg
    assert "prose 5, script 4" in msg
    assert str(guard_tree) in msg and _head(guard_tree)[:8] in msg
    assert _head(vault) == head, "refused before any vault commit"


def test_a_refusal_leaves_the_vault_at_its_previous_head_and_is_ledged(
        vault, guard_tree, probed, monkeypatch):
    """Clause 3. Nothing committed, and the ledger row names the item and why."""
    _two_workers(monkeypatch)      # so the `workers` pinned below is 2, not config
    _prose(vault, SKILL_AT_FOUR, commit=True)
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n")
    head = _head(vault)
    with pytest.raises(V.VaultRoundError):
        V.land(["skills/foo/SKILL.md"], "#2036 five stores", item_id=42)
    assert _head(vault) == head
    rows = _events("vault_land")
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["ok"] is False and "commit" not in row
    assert row["item_id"] == 42
    assert row["reverted"] == ["skills/foo/SKILL.md"]
    g = row["guards"]
    # The decision keys, unchanged in value by #2042.
    assert {k: g[k] for k in ("state", "refuse", "nodes")} == {
        "state": "checked", "refuse": True,
        "nodes": ["tests/test_guard.py::test_the_prose_states_the_count_the_script_prints"]}
    assert g["candidate"]["ran"] == 1 and g["candidate"]["failed"] == 1
    assert g["baseline"]["ran"] == 1 and g["baseline"]["failed"] == 0
    # The row's shape is still closed: a key appears only because a clause put it
    # there. `lock_wait_s` is absent because nothing was queued behind here, and
    # `_guards_row` writes detail keys only when there is something to say. #2046
    # put the last two keys here, and the count is `_two_workers`' 2 rather than the
    # box's `automod.gate.test_workers`, so the pin says which value the projection
    # owed the row. A `checked` row cannot exist at `workers` 1 with a failing node,
    # because `vault_guards.py:718` returns `skipped` before launching a child, and
    # the failing node was re-asked serially before it could refuse — which the row
    # now says as data. `parallel_only_failures` being ABSENT is the other half of
    # the pin: the serial re-ask reproduced the failure, so nothing was dismissed.
    # `reason` joins the set at #2049: clause 2 requires a refusal to name the node id in
    # the prose a `promotions.jsonl` reader actually reads, not only in `nodes`, and
    # `_guards_row` writes `reason` only when it is non-empty — which on a plain refusal it
    # was not before. The pin moves because the row legitimately grew, not loosely.
    assert set(g) == {"state", "refuse", "seconds", "candidate", "baseline",
                      "nodes", "excerpt", "workers", "parallel_retry", "reason"}, g
    assert "tests/test_guard.py::" in g["reason"], g["reason"]
    assert g["workers"] == 2, g
    assert g["parallel_retry"] == {"ran": 1, "failed": 1, "workers": 1}, g
    # #2042: a refusal row now carries its own cost beside its verdict — the seconds
    # each run took and how many vault-reading files it ran over — and the guard
    # output that states the counts. Before this, only the verdict was ledgered.
    assert g["candidate"]["files"] == 1 and g["baseline"]["files"] == 1
    assert 0 < g["candidate"]["seconds"] < 60 and 0 < g["baseline"]["seconds"] < 60
    assert g["seconds"] >= g["candidate"]["seconds"] + g["baseline"]["seconds"]
    # The row's own excerpt is pytest's tail, so it names the node that failed;
    # the guard's stated counts reach the ledger through `errors`, via
    # `refusal_text`, which is where the refusal a human reads gets them.
    assert "tests/test_guard.py::test_the_prose_states_the_count" in g["excerpt"], g["excerpt"]
    assert "prose 5, script 4" in row["errors"][0]
    assert row["review_reason"] == "code agreement refused the land"
    # The prose is back at the vault's HEAD state: "nothing lands" has to mean
    # "nothing stays", the #599 rule this route already enforces for validation.
    assert (vault / "skills/foo/SKILL.md").read_text() == SKILL_AT_FOUR


def test_a_land_whose_stated_count_agrees_with_the_tree_still_commits(
        vault, guard_tree, probed):
    """Clause 2, first half: a land that only rewords prose is never refused."""
    _prose(vault, SKILL_AT_FOUR, commit=True)
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 4\n\nBounded by the weekly sweep.\n")
    head = _head(vault)
    out = V.land(["skills/foo/SKILL.md"], "#2036 reworded", item_id=None)
    assert _head(vault) != head
    assert out["guards"]["state"] == "checked" and out["guards"]["refuse"] is False
    cand = out["guards"]["candidate"]
    assert cand["ran"] == 1 and cand["failed"] == 0
    # #2042: the pass row states what it spent and over how much of the selection.
    assert cand["files"] == 1 and 0 < cand["seconds"] < 60, cand
    assert _events("vault_land")[0]["guards"]["refuse"] is False


def test_a_count_land_whose_code_half_already_landed_proceeds(vault, guard_tree, probed):
    """Clause 2, second half: five store lines on the tree, so prose may say five.

    This is #1975's repair direction — the code half landed, the prose follows it
    — and it must not be refused, which is what makes the check consistency-scoped
    rather than a ban on a `mixed` item landing its vault half.
    """
    (guard_tree / "store_lines.py").write_text("draft\nup_next\nin_progress\ndone\nquarantined\n")
    git(guard_tree, "add", "-A")
    git(guard_tree, "commit", "-q", "-m", "the thirteenth store line lands")
    _prose(vault, SKILL_AT_FOUR, commit=True)
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n")
    out = V.land(["skills/foo/SKILL.md"], "#2036 five stores, code first", item_id=None)
    g = out["guards"]
    assert {k: g[k] for k in ("state", "refuse", "reason")} == {
        "state": "checked", "refuse": False,
        "reason": "1 vault-reading node passes against the vault as proposed"}
    # The one-run pass path never reaches the baseline, and it now states its cost.
    assert "baseline" not in g, g
    assert g["candidate"]["ran"] == 1 and g["candidate"]["failed"] == 0
    assert g["candidate"]["files"] == 1 and 0 < g["candidate"]["seconds"] < 60, g


def test_a_guard_already_red_before_the_land_does_not_refuse_it(vault, guard_tree, probed):
    """The delta rule: a red tree is not this land's fault, and is not a block.

    `main` is red right now (#1975's unlanded code half). Refusing on any red
    guard would hold every landing on the route hostage to a failure this land did
    not cause, which is why the check compares the proposed vault against the same
    tree with only this land's paths put back. The route is busy enough that this
    is not a hypothetical: #1975 landed three prose commits in forty-one minutes, all
    three `review: skipped`, and those rows are the witness this file reads.
    """
    _prose(vault, "---\nname: foo\n---\nstores: 6\n", commit=True)   # red at vault HEAD
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 6\n\nReworded only.\n")
    out = V.land(["skills/foo/SKILL.md"], "#2036 reword while main is red", item_id=None)
    assert out["guards"]["state"] == "checked" and out["guards"]["refuse"] is False
    # Both runs red, the same node: that is what "pre-existing" measured.
    cand, base = out["guards"]["candidate"], out["guards"]["baseline"]
    assert cand["ran"] == 1 and cand["failed"] == 1
    assert base["ran"] == 1 and base["failed"] == 1
    # Both runs red on the same node, and both say how long and over what.
    assert cand["files"] == 1 and base["files"] == 1
    assert cand["seconds"] > 0 and base["seconds"] > 0


# --------------------------------------------------------------------------- #
# #2265: the delta rule only works when a node id names the thing a land changes.
#
# `agreement` decides by differencing NODE IDS against the same selection run
# against the pre-land vault (`vault_guards.py:953`). A guard that walks a whole
# corpus inside one function therefore has ONE id for the whole corpus: one
# already-bad file makes that id fail in both runs, `new` comes back empty, and
# every later land that ADDS an offender commits under the reason "pre-existing,
# not this land". The two fixture guards below differ in exactly the two
# properties #2265 changed in `tests/test_research_doc_claims.py` — one node per
# file instead of one per corpus, and a corpus read that follows the
# `LLOYD_VAULT_ROOT` mirror the probe hands its child (`vault_guards.py:508`)
# instead of a root baked in at write time — so the pair is the before and the
# after of the same land, and the verdict flips with them.
#
# The offender rule is deliberately simpler than the real one: the probe child runs
# with the fixture tree as its PYTHONPATH, so `app.autonomy._build_task_prompt` is
# not importable inside it. The front-matter `description:` stands in for the
# delivered prompt and the body before `## Activity` for the instruction region —
# the same two channels `_prompt_and_region` compares. What is under test is the
# node id and the root resolution, and both of those are the real thing.
# --------------------------------------------------------------------------- #

#: The corpus guard in the shape `tests/test_research_doc_claims.py` had until
#: #2265: one node for every task file, and `__LIVE_VAULT__` replaced with the
#: vault's own path, so neither of the probe's two mirrors can make it read a
#: different corpus. Naming `LLOYD_VAULT_ROOT` here is also what keeps this file in
#: the probe's own selection (`VAULT_ROOT_TOKENS`, `vault_guards.py:90`).
GUARD_SRC_CORPUS_ONE_NODE = '''"""One node for a whole corpus, blind to LLOYD_VAULT_ROOT (#2265's before)."""
from pathlib import Path

_TOOLS = ("vault_write", "fact_relate")
VAULT = Path("__LIVE_VAULT__")


def _offenders(vault):
    out = []
    for path in sorted((vault / "autonomy").glob("[0-9]*-*.md")):
        parts = path.read_text().split("---\\n", 2)
        fm, body = (parts[1], parts[2]) if len(parts) > 2 else ("", "")
        region = body.split("## Activity")[0]
        bad = sorted(t for t in _TOOLS
                     if "`" + t + "`" in region and "`" + t + "`" not in fm)
        if bad:
            out.append(f"{path.name}: {bad}")
    return out


def test_no_task_body_orders_a_tool_its_description_never_names():
    bad = _offenders(VAULT)
    assert not bad, "task bodies ordering an unnamed tool: " + "; ".join(bad)
'''

#: The same guard in the shape #2265 gave it: one node per task file, the node id
#: being that file's name, and the corpus read through the knob — the resolution
#: `app/data_root.py::vault_root` performs, restated here because the child cannot
#: import `app` from the throwaway tree it runs in.
GUARD_SRC_ONE_NODE_PER_FILE = '''"""One node per task file, corpus read through LLOYD_VAULT_ROOT (#2265's after)."""
import os
from pathlib import Path

import pytest

_TOOLS = ("vault_write", "fact_relate")


def _vault():
    return Path(os.environ.get("LLOYD_VAULT_ROOT") or (Path.home() / "obsidian"))


def _offenders(path):
    parts = path.read_text().split("---\\n", 2)
    fm, body = (parts[1], parts[2]) if len(parts) > 2 else ("", "")
    region = body.split("## Activity")[0]
    return sorted(t for t in _TOOLS
                  if "`" + t + "`" in region and "`" + t + "`" not in fm)


@pytest.mark.parametrize("task_file",
                         sorted((_vault() / "autonomy").glob("[0-9]*-*.md")),
                         ids=lambda p: p.name)
def test_no_task_body_orders_a_tool_its_description_never_names(task_file):
    bad = _offenders(Path(task_file))
    assert not bad, f"{Path(task_file).name}: {bad}"
'''

#: The third shape, which exists to isolate one variable: per-file nodes, but the
#: vault root baked in at write time. Parametrizing alone does not close the hole.
#: File B's node is collected in BOTH runs, because the pre-land run still reads
#: the live vault with file B sitting in its working tree; `new` is empty, and the
#: land that the shape above refuses commits here. Both properties are needed, and
#: this is the run that shows the root half is not decoration.
GUARD_SRC_ONE_NODE_BAKED_ROOT = GUARD_SRC_ONE_NODE_PER_FILE.replace(
    '    return Path(os.environ.get("LLOYD_VAULT_ROOT") or (Path.home() / "obsidian"))',
    '    return Path("__LIVE_VAULT__")').replace(
    '"""One node per task file, corpus read through LLOYD_VAULT_ROOT (#2265\'s after)."""',
    '"""One node per task file, vault root baked in (#2265: granularity, no knob)."""')
assert "__LIVE_VAULT__" in GUARD_SRC_ONE_NODE_BAKED_ROOT, \
    "the replace missed its target, so the baked-root guard is the knob-reading one"

TASK_A_OFFENDER = ("---\nid: 10\nname: Pre-existing offender\n"
                   "description: Summarise the notes.\n---\n\n"
                   "# Offender\n\nThe run calls `vault_write` on every note.\n")
TASK_B_OFFENDER = ("---\nid: 20\nname: Fresh offender\n"
                   "description: Reconcile the facts.\n---\n\n"
                   "# Fresh\n\nThe run calls `fact_relate` on every pair.\n")
TASK_CLEAN = ("---\nid: 30\nname: Clean\n"
              "description: Write the notes with `vault_write`.\n---\n\n"
              "# Clean\n\nWrite the notes with `vault_write`.\n")

TASK_B_PATH = "autonomy/20-b-fresh.md"
#: The guard's own node id with no param attached: in the corpus-shaped fixture that
#: ONE id is the whole corpus, and in the other two it is the prefix every per-file
#: id hangs off. `agreement` compares strings like these, so this is the surface the
#: fix turns on.
TASK_CORPUS_NODE = ("tests/test_guard.py::"
                    "test_no_task_body_orders_a_tool_its_description_never_names")
TASK_A_NODE = TASK_CORPUS_NODE + "[10-a-offender.md]"
#: The node #2265 is about: the guard's own name, suffixed with file B's name.
TASK_B_NODE = TASK_CORPUS_NODE + "[20-b-fresh.md]"


def _corpus_vault(vault) -> None:
    """The vault as it stands before the land: one offender already at HEAD, one
    clean file, both committed. The pre-existing offender is what makes the corpus
    red today, which is the condition the excuse hides behind.
    """
    (vault / "autonomy" / "10-a-offender.md").write_text(TASK_A_OFFENDER)
    (vault / "autonomy" / "30-clean.md").write_text(TASK_CLEAN)
    git(vault, "add", "-A")
    git(vault, "commit", "-q", "-m", "a corpus already red at vault HEAD")


def _corpus_tree(tmp_path, vault, src: str):
    """A code checkout whose one vault-reading guard is `src`, with the live vault
    path baked into it (the `__LIVE_VAULT__` seam is only meaningful there).
    """
    return make_guard_tree(tmp_path / "corpus-tree",
                           src=src.replace("__LIVE_VAULT__", str(vault)),
                           store_lines="")


def _add_fresh_offender(vault):
    """File B: a NEW task file, offending in a way file A does not offend."""
    f = vault / TASK_B_PATH
    f.write_text(TASK_B_OFFENDER)
    return f


def test_a_land_that_adds_an_offender_to_a_red_corpus_is_refused_and_names_it(
        vault, tmp_path, monkeypatch, probed):
    """#2265 clause 4, the half the route used to get wrong.

    File A is red at vault HEAD, so the corpus guard is red whatever this land
    does. File B is the land: a new file, offending with a tool name A never
    mentions. With per-file node ids and a corpus read through the probe's mirror,
    file B's node does not exist in the pre-land run — it is `new`, the land is
    refused, and the refusal names B. `test_a_whole_corpus_guard_excuses_the_same_land`
    is the same land against the same red corpus with the guard in its pre-#2265
    shape, and it commits.
    """
    _two_workers(monkeypatch)
    _corpus_vault(vault)
    probed(_corpus_tree(tmp_path, vault, GUARD_SRC_ONE_NODE_PER_FILE), vault)
    _add_fresh_offender(vault)
    head = _head(vault)
    with pytest.raises(V.VaultRoundError) as ei:
        V.land([TASK_B_PATH], "#2265 a fresh offender joins a red corpus", item_id=None)
    msg = str(ei.value)
    assert "code agreement failed" in msg
    assert TASK_B_NODE in msg, msg
    assert "fact_relate" in msg, "the refusal should carry the guard's own output"
    assert _head(vault) == head, "refused before any vault commit"
    assert not (vault / TASK_B_PATH).exists(), "the new file was not reverted"

    row = _events("vault_land")[-1]
    assert row["ok"] is False and row["item_id"] is None, row
    assert row["reverted"] == [TASK_B_PATH], row
    g = row["guards"]
    assert g["state"] == "checked" and g["refuse"] is True
    # The sharp half: A's node is red in BOTH runs and is not refused; only B's is
    # new, because the pre-land mirror did not hold file B to collect a node from.
    assert g["nodes"] == [TASK_B_NODE], g["nodes"]
    assert TASK_A_NODE not in g["nodes"], \
        "file A is red in both runs and must not be the refused one"
    assert TASK_B_PATH.split("/")[-1] in g["reason"], g["reason"]
    assert g["candidate"]["failed"] == 2 and g["candidate"]["ran"] == 4, g["candidate"]
    assert g["baseline"]["failed"] == 1 and g["baseline"]["ran"] == 3, g["baseline"]


def test_a_whole_corpus_guard_excuses_the_same_land_as_pre_existing(
        vault, tmp_path, monkeypatch, probed):
    """#2265's before: the identical land, the identical red corpus, one node.

    One id for the whole corpus, and a vault root baked in, so the candidate run
    and the pre-land run read byte-identically the same corpus: both fail that one
    node, `new` is empty, and the route commits file B with "pre-existing, not this
    land" as its reason. This node is the measurement the parametrization exists to
    change, so it asserts the excuse's exact sentence — a route that stopped
    producing it while the guard was still corpus-shaped would be a second,
    different fix, and this node says which one it was.
    """
    _two_workers(monkeypatch)
    _corpus_vault(vault)
    probed(_corpus_tree(tmp_path, vault, GUARD_SRC_CORPUS_ONE_NODE), vault)
    _add_fresh_offender(vault)
    head = _head(vault)
    out = V.land([TASK_B_PATH], "#2265 the same land, corpus-shaped guard",
                 item_id=None)
    assert _head(vault) != head, "this land is the one #2265 says used to commit"
    assert (vault / TASK_B_PATH).exists()
    g = out["guards"]
    assert g["state"] == "checked" and g["refuse"] is False, g
    assert g["reason"] == ("1 failing node(s) fail against the pre-land vault too — "
                           "pre-existing, not this land"), g["reason"]
    # `_guards_row` only records node ids on the refuse path, which is why the real
    # `vault_land` rows that carry this reason name no node at all — the ledger
    # cannot say what was excused, only how many.
    assert "nodes" not in g, g
    assert g["candidate"]["failed"] == 1 and g["candidate"]["ran"] == 1, g["candidate"]
    assert g["baseline"]["failed"] == 1 and g["baseline"]["ran"] == 1, g["baseline"]
    assert g["baseline"]["files"] == 1, g["baseline"]
    # The one id the excuse covered is the corpus, not a file: nothing in this
    # ledger row could ever have named file B.
    assert TASK_CORPUS_NODE in g["excerpt"], g["excerpt"]


def test_per_file_nodes_with_a_baked_vault_root_are_still_excused(
        vault, tmp_path, monkeypatch, probed):
    """#2265: node granularity alone is not the fix; the corpus has to be the mirror.

    Same per-file guard, same red corpus, same new file B — only the root
    resolution reverts to a path baked in at write time. File B's node now exists
    in the pre-land run as well, because the run that is supposed to be reading
    "before" is reading the live vault with B sitting in its working tree. Two
    failing ids in, two failing ids out, `new` empty, and the land the parametrized
    shape refuses goes through. The count of failing nodes is the tell: the corpus
    shape above excuses ONE id and this one excuses TWO, because per-file ids have
    now given the excuse one id per offender instead of one for the lot.
    """
    _two_workers(monkeypatch)
    _corpus_vault(vault)
    probed(_corpus_tree(tmp_path, vault, GUARD_SRC_ONE_NODE_BAKED_ROOT), vault)
    _add_fresh_offender(vault)
    head = _head(vault)
    out = V.land([TASK_B_PATH], "#2265 per-file nodes, root still baked in",
                 item_id=None)
    assert _head(vault) != head, "granularity without the knob still commits"
    g = out["guards"]
    assert g["state"] == "checked" and g["refuse"] is False, g
    assert g["reason"] == ("2 failing node(s) fail against the pre-land vault too — "
                           "pre-existing, not this land"), g["reason"]
    assert g["candidate"]["failed"] == 2 and g["candidate"]["ran"] == 4, g["candidate"]
    assert g["baseline"]["failed"] == 2 and g["baseline"]["ran"] == 4, g["baseline"]


def test_the_rewritten_corpus_guard_is_still_in_the_land_time_selection():
    """#2265's string-key boundary: the rewritten module still gets RUN at land time.

    `agreement` only ever sees what `guard_selection` picks, and the picker is a
    substring scan for `VAULT_ROOT_TOKENS` (`vault_guards.py:90`), not an import and
    not an AST. Rewriting how a module resolves its vault is exactly the edit that
    could drop a token and silently remove the guard from the probe while leaving
    every suite node green — a guard nobody runs reports no failures. So the
    selection is asked directly, over this checkout, for the real file.
    """
    root = Path(__file__).resolve().parent.parent
    sel = VG.guard_selection(root)
    assert "tests/test_research_doc_claims.py" in sel, \
        "the body-tool guard left the land-time selection: the probe cannot refuse on " \
        "a file it never runs"
    src = (root / "tests/test_research_doc_claims.py").read_text(encoding="utf-8")
    token = next((t for t in VG.VAULT_ROOT_TOKENS if t in src), None)
    assert token, "no selection token survives in the module at all"


#: #2265 clause 5's witness: every `vault_land` row of the promotions ledger,
#: committed at a dated name because the path the clause named is the retired
#: mirror (#2054's retire, #2064's `ffc04ce5`, enforced by
#: `tests/test_failure_ledger_witness.py::test_the_witness_is_a_dated_file_not_the_retired_mirror_path`).
#: The route #2178 took for #2175 clause 5, for the same reason.
WITNESS_2265_NAME = "2026-10-06.2265-vault-land-witness.jsonl"


def test_the_2265_vault_land_witness_re_derives_the_quoted_counts():
    """Clause 5 asked for the figures to come off committed bytes, so they are read here.

    The item's report is "45 of 433 `vault_land` rows mention 'pre-existing'" and
    three rows naming the corpus node beside that excuse. On the committed bytes the
    same two commands say 46 of 435, and four such rows — the ledger is append-only
    and the item counted on a window that closed on 2026-10-06, so the denominator
    moved by two rows and the count by one after the triage. What has NOT moved is
    the set the claim is about: the four commits below, each `refuse: false` with
    that reason, each carrying this file's corpus node in `guards.excerpt`.

    The extract is a frozen file, not the live ledger: `wc -l` on it is a figure a
    later `vault_land` cannot change, which is why it is pinned at an exact number
    rather than a floor. The four node-naming rows are asserted by commit so a
    re-extract that loses one cannot pass.
    """
    witness = Path.home() / "obsidian" / "backlog" / "data" / WITNESS_2265_NAME
    assert witness.is_file(), (
        f"{WITNESS_2265_NAME} must be committed: clause 5 asks that the quoted report "
        "be re-derivable from bytes, and the automod state dir is not a git tree")
    text = witness.read_text(encoding="utf-8")
    assert text.count("\n") == 435, "wc -l on the witness moved off the 435 the report says"
    rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    assert len(rows) == 435, f"{len(rows)} rows parse, not the 435 `wc -l` counted"
    assert all(r["event"] == "vault_land" for r in rows), \
        "rows that are not vault_land are not the extract #2265 asked for"
    pre = [r for r in rows if "pre-existing" in json.dumps(r)]
    assert len(pre) == 46, (
        f"{len(pre)} of the 435 rows mention pre-existing; the item quotes 45 of 433, "
        "and the gap is what this witness exists to make checkable")
    named = [r for r in rows if "test_no_task_body_orders_a_tool_its_prompt_never_names"
             in str((r.get("guards") or {}).get("excerpt") or "")]
    assert [str(r["commit"])[:9] for r in named] == \
        ["9d07359df", "6be4ef94d", "d698d38ae", "d31bbbb7d"], \
        "the rows that name the corpus node beside the excuse are the evidence"
    for r in named:
        assert r["guards"]["refuse"] is False and r["ok"] is True, r["commit"]
        assert "pre-existing, not this land" in r["guards"]["reason"], r["commit"]
    assert all("baseline" in r["guards"] for r in named), \
        "each of these is a pre-land run that could not have changed the verdict"


#: `scripts/automod/review.py:1433`'s own abstention for a surface that is not
#: `vault` — #2036's blindness site, and the sentence three `vault_land` rows for
#: #1975 carry verbatim. `test_a_mixed_surface_items_vault_land_is_checked_too`
#: gets it by calling `review.grade_vault` for real rather than reciting it, and the
#: ledger-witness test below compares the recorded rows against it.
MIXED_SKIP = "surface is mixed, not vault: the clauses are graded at the code gate"
#: review.py builds that sentence from an f-string, so only this tail of it is a
#: literal in the source. The test below asserts the tail is there before it asserts
#: the real grader produced the whole sentence: if the wording moves, that ordering
#: says which of the two went stale.
MIXED_SKIP_TAIL = "not vault: the clauses are graded at the code gate"


@pytest.mark.parametrize("surface,skip", [
    ("mixed", MIXED_SKIP),
    ("code", "surface is code, not vault: the clauses are graded at the code gate"),
], ids=["mixed", "code"])
def test_a_mixed_surface_items_vault_land_is_checked_too(vault, guard_tree, probed,
                                                         monkeypatch, surface, skip):
    """Clause 4. The review rung abstained on this surface and that exempted nothing.

    `review.grade_vault` runs FOR REAL here: a confirmed triage row names the
    surface, the branch at review.py:1433 executes, and the abstention quoted in
    the ledger is that function's own sentence. Stubbing the grader would stay
    green through any change to the condition that produces it — which is exactly
    how a future widening of that skip would hide. The land is refused regardless,
    by the code-agreement check: the rung that owns the clauses abstained, and the
    land was still checked against the tree.
    """
    import scripts.automod.review as RV
    assert MIXED_SKIP_TAIL in (repo() / "scripts/automod/review.py").read_text(), \
        "review.py no longer abstains in these words; this node's expect is stale"
    S.append_event({"event": "backlog_triage", "item_id": 42, "verdict": "confirmed",
                    "surface": surface, "acceptance": "a",
                    "acceptance_clauses": ["a"]}, path=S.LEDGER_PATH)
    monkeypatch.setattr(V, "GRADER", RV.grade_vault)
    _prose(vault, SKILL_AT_FOUR, commit=True)
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n")
    with pytest.raises(V.VaultRoundError) as ei:
        V.land(["skills/foo/SKILL.md"], "#2036 mixed item prose", item_id=42)
    assert "test_the_prose_states_the_count_the_script_prints" in str(ei.value)
    reviews = _events("vault_review")
    assert reviews and reviews[0]["kind"] == "skipped", reviews
    assert reviews[0]["findings"] == skip, "the row is not the skip review.py wrote"
    assert _events("vault_land")[0]["guards"]["refuse"] is True


@pytest.mark.parametrize("surface", ["vault", "mixed", "code", None],
                         ids=["vault", "mixed", "code", "no-surface-field"])
def test_the_code_agreement_call_carries_no_surface_and_no_item(vault, monkeypatch,
                                                                surface):
    """Clause 4, mechanism half: asked once with `paths` and nothing else, whatever
    the item's surface is — which is what makes an exemption impossible to write.

    A route that decides per surface is a route that eventually exempts one: the
    #551 history inside review.py:1426-1430 is what a surface-scoped refusal did
    last time, and #2036's own triage refuses to re-break it. So this walks every
    surface a triage row can carry — including the absent field, which is most of
    the ledger's history — and refuses a call that names one. A probe told which
    item is landing is one branch away from a probe that skips some of them, so
    `item_id` is refused by the same assert rather than by a name nobody would
    think to grep for.
    """
    calls = []

    def fake(**kw):
        calls.append(kw)
        return {"state": "checked", "refuse": False}

    monkeypatch.setattr(V.VG, "agreement", fake)
    monkeypatch.setattr(V, "GRADER", lambda *a, **k: ("skipped", "not consulted here", []))
    row = {"event": "backlog_triage", "item_id": 42, "verdict": "confirmed",
           "acceptance": "a", "acceptance_clauses": ["a"]}
    if surface is not None:
        row["surface"] = surface
    S.append_event(row, path=S.LEDGER_PATH)
    (vault / "skills/foo/SKILL.md").write_text("---\nname: foo\n---\nstores: 4\n\n#2036.\n")
    out = V.land(["skills/foo/SKILL.md"], "#2036 shape", item_id=42)
    assert len(calls) == 1, f"surface {surface!r}: {calls}"
    assert set(calls[0]) == {"paths"}, f"surface {surface!r} reached the probe call"
    assert calls[0]["paths"] == ["skills/foo/SKILL.md"]
    assert out["guards"]["refuse"] is False


def test_a_probe_that_cannot_judge_proceeds_and_says_why_on_the_row(vault, tmp_path,
                                                                   monkeypatch):
    """A non-answer is never a refusal, and never silent either.

    Holding the vault route hostage to a subprocess it does not own would be a
    worse failure than the hole being closed — the same shape as the reviewer's
    abstention one block earlier in `land()`. What is not allowed is a row that
    cannot tell a clean check from a check that never ran.
    """
    monkeypatch.delenv(VG.NESTING_ENV, raising=False)
    real = VG.agreement   # bound before the attribute is replaced: `V.VG` is `VG`
    monkeypatch.setattr(V.VG, "agreement", lambda **kw: real(
        paths=kw["paths"], live_root=tmp_path / "no-such-checkout", live_vault=vault,
        python=Path(sys.executable), scratch_parent=tmp_path / "scratch"))
    _prose(vault, "---\nname: foo\n---\n# foo\n\nReworded while unprobeable.\n")
    out = V.land(["skills/foo/SKILL.md"], "#2036 unjudgeable", item_id=None)
    assert out["commit"]
    assert out["guards"]["state"] == "skipped" and out["guards"]["refuse"] is False
    # The named root is unreadable as a git tree, which is the first thing the
    # probe can fail on and the reason it has to say rather than assume agreement.
    assert "cannot read the code tree" in out["guards"]["reason"]
    assert "no-such-checkout" in out["guards"]["reason"]
    assert _events("vault_land")[0]["guards"]["state"] == "skipped"


def test_a_selection_whose_only_guard_is_skipped_answers_nothing_and_still_lands(
        vault, guard_tree, probed, monkeypatch):
    """The denominator counts nodes that RAN, not nodes that were collected.

    A selection whose one vault-reading guard is `skipif`-skipped collects fine
    and asserts nothing — the zero-denominator failure the tree already names
    seven instances of. It has to read as "not judged", never as "found nothing",
    and it must not be the thing that stops a landing.
    """
    _prose(vault, "---\nname: foo\n---\nstores: 6\n", commit=True)
    monkeypatch.setenv("GUARD_FORCE_SKIP", "1")
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 7\n")
    out = V.land(["skills/foo/SKILL.md"], "#2036 skipped selection", item_id=None)
    assert out["commit"]
    assert out["guards"]["state"] == "skipped" and out["guards"]["refuse"] is False
    assert "answered nothing" in out["guards"]["reason"]


def test_the_selection_is_named_by_token_not_by_a_hand_kept_list(tmp_path):
    """The selection is derived from the tree, so no list here can go stale.

    #1734 and #1835 are both on record with a *guard's own* count going stale, and
    a registry of prose claims to verify rots the same way. What is enumerable is
    the set of guards a vault land can break — which is what `guard_selection`
    returns, from the tree it is handed.
    """
    tree = tmp_path / "t"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests/test_reads_a_vault.py").write_text("def a(): vault_root()\n")
    (tree / "tests/test_reads_env.py").write_text("VAULT_ROOT = 1\n")
    (tree / "tests/test_reads_nothing.py").write_text("def a(): return 1\n")
    (tree / "tests/helper_not_a_test.py").write_text("vault_root()\n")
    assert VG.guard_selection(tree) == ["tests/test_reads_a_vault.py",
                                        "tests/test_reads_env.py"]
    assert VG.guard_selection(tmp_path / "absent") == []


# --------------------------------------------------------------------------- #
# Clause 5 lives in this file because it pins the OTHER file: the two
# `tests/test_retention_sweep.py` count guards have to stay in the default
# selection, or a red main can be silenced by excluding the nodes that read the
# vault — which is the wrong fix #2036's own triage names and rejects.
# --------------------------------------------------------------------------- #

def test_the_two_retention_count_guards_stay_in_the_default_selection(tmp_path):
    """Clause 5, measured — not asserted about a marker.

    Three ways a red main gets silenced are all refused here: a node vanishing
    from the file, a node growing a decorator (a `live_vault` or `skip` mark), and
    a node that survives the mark expr but is skipped or not run by the child
    pytest that the gate would actually spawn. The child runs with `LLOYD_DATA` in
    this test's own `tmp_path`, so it cannot write into a data root another test
    is reading — `tests/test_retention_sweep.py`'s sweep nodes create directories
    under theirs — and the tree it runs in is the round's own checkout, never
    `~/lloyd`, which the suite refuses to run in at all (`tests/conftest.py:205`).
    """
    import ast

    src = (repo() / "tests/test_retention_sweep.py").read_text()
    fns = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    for name in RETENTION_NODES:
        assert name in fns, f"{name} is gone from the file that has guarded it"
        assert fns[name].decorator_list == [], f"{name} grew a decorator"
    assert "live_vault" not in src, "a live_vault mark entered the file that reads the vault"

    mark, _failed_ids, summary = VG._gate_tools()
    assert mark == "not live_vault and not fault_injection", "the gate's own constant"
    env = {**os.environ, "LLOYD_DATA": str(tmp_path / "retention-data")}
    # The child is the gate's own command line: the mark expr is that constant, the
    # file is named, and `-p no:cacheprovider` keeps a probe run from writing a
    # cache the suite would inherit. Whatever way the two nodes fail is decided by
    # the summary counts below, and deselection shows up there too: a node the mark
    # expr excludes never reaches passed or failed, so the sum is not 2.
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-m", mark, "-k", " or ".join(RETENTION_NODES),
                        "tests/test_retention_sweep.py"],
                       cwd=str(repo()), capture_output=True, text=True, timeout=300,
                       env=env)
    line = r.stdout.strip().splitlines()[-1]
    counts = summary(r.stdout)
    # Two nodes RUN. Which way they fail is #1975's business — the diff that ships
    # the thirteenth store line is the diff that makes those two counts agree, so
    # which way they agreed is that item's, never a fact this node can hold over a
    # moving tree. That they are neither deselected nor skipped is clause 5, and is
    # what a `live_vault` mark or a `pytest.skip` on either node would turn red.
    # Counted from the summary line rather than grepped, because the file's other 83
    # nodes are legitimately deselected by `-k` and the word alone says nothing.
    assert counts["passed"] + counts["failed"] == 2, line
    assert counts["tests_skipped"] == 0, line


def test_the_probe_judges_a_copy_so_a_writing_guard_cannot_reach_the_live_vault(
        vault, tmp_path, probed):
    """The check reads a mirror of the vault, never the vault it is about to land.

    `land()` runs the probe with its own edit already written into the vault's
    working tree, and commits that tree afterwards. A selection handed the live
    vault would therefore let any guard in it add bytes to the commit the route
    came only to judge — and a hardlink or symlink farm fails identically, because
    both write THROUGH. The fixture guard writes a file into whichever vault it is
    shown, then fails on the count, so a probe pointed at the live tree leaves
    `PROBED` behind there while still refusing the land. The refusal is asserted
    too: a mirror that did not carry the proposed edit would wave the bad land
    through, which is the other way this node can go red.
    """
    tree = make_guard_tree(tmp_path / "writing-tree", src=GUARD_SRC_WRITES)
    _prose(vault, SKILL_AT_FOUR, commit=True)
    probed(tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n")
    with pytest.raises(V.VaultRoundError) as ei:
        V.land(["skills/foo/SKILL.md"], "#2036 a guard that writes", item_id=None)
    assert "test_a_guard_that_mutates_the_vault_it_judges" in str(ei.value)
    assert not (vault / "PROBED").exists(), "a probe guard wrote into the vault being landed"


# --------------------------------------------------------------------------- #
# #2036's own evidence, kept where a later reader can open it. The previous round
# of this item was refused on clause 6 because its report quoted ledger rows the
# tree could not reproduce: `tail -3` of the live promotions ledger has moved on
# (sibling items land every hour), and the vault's committed copy of the whole
# ledger came from #1975's land, not from this item. So the rows the report quotes
# are committed here verbatim, and this node re-reads them. A claim nobody can
# re-check is the defect, not a footnote to it.
# --------------------------------------------------------------------------- #

#: The three `vault_land` rows of #1975 — the `mixed` item whose vault half landed
#: "thirteen" while its code half sat on an unmerged branch, which is #2036's
#: premise. Cut verbatim from ~/.local/state/lloyd-automod/promotions.jsonl on
#: 2026-10-01, and the family the root `.gitignore`'s `*.json` rule does not reach.
WITNESS = repo() / "tests/fixtures/promotions_vault_land_rows_2026-10-01-item1975.jsonl"


def test_the_promotions_ledger_witness_still_shows_one_mixed_item_landed_thrice():
    """The refusal this item is built on is re-checkable from bytes in the tree.

    #2036's claim about the world is one sentence: a `mixed` item's vault half can
    land — prose and all, unreviewed — while its code half sits unmerged. These
    three rows are that sentence. Three lands of item #1975 inside forty-one minutes,
    every one `review: skipped` with review.py's own mixed-surface abstention, the
    first carrying the two prose paths that took the count from twelve to thirteen
    and the two after it validating nothing, and no row carrying a round at all
    (that gap is #1987's, not this item's). The guard the first row's prose reddened
    is the one clause 5 keeps in the default selection.
    """
    import json
    rows = [json.loads(l) for l in WITNESS.read_text().splitlines() if l.strip()]
    assert len(rows) == 3, [r.get("ts") for r in rows]
    assert all(r["event"] == "vault_land" and r["item_id"] == 1975 for r in rows)
    assert all(r["ok"] is True for r in rows), "these are landed rows, not refusals"
    assert [len(r["validated"]) for r in rows] == [2, 0, 0], [r["validated"] for r in rows]
    assert rows[0]["validated"] == ["skills/retention-sweep/SKILL.md",
                                    "autonomy/79-retention-sweep.md"]
    assert [r["commit"][:8] for r in rows] == ["ceea3904", "bf91f9e5", "4bc93177"]
    assert [r["review"] for r in rows] == ["skipped"] * 3
    assert all(r["review_reason"] == MIXED_SKIP for r in rows)
    # The blindness in one key: no row names a round, so no rung had one to read.
    assert all("round" not in r and "round_id" not in r for r in rows)
    span = float(rows[-1]["ts"]) - float(rows[0]["ts"])
    # Cut within one sitting on 2026-10-01: first land 19:37:44Z, last 20:17:56Z.
    assert 0 < span <= 45 * 60, f"three lands {span:.0f}s apart is not one sitting"


def test_a_vault_surface_items_land_is_checked_by_the_same_call(vault, guard_tree,
                                                               probed, items,
                                                               monkeypatch):
    """Clause 4, other half: a plain `vault`-surface land is refused by that same call.

    The `mixed` item is the case that was blind, but the clause's words are that a
    plain vault-surface land is checked the same way — so here is one, with a triage
    row that says `surface: vault` and real acceptance clauses, so the reviewer runs
    the full `grade_vault` path rather than skipping early. `run_grader` is stubbed at
    the transport (a reviewer that cannot reach a backend abstains, and an abstention
    is not what is being tested here); the land is refused by the code-agreement check
    either way, which is the point: the probe sits on the route whatever the reviewer
    made of the clauses.
    """
    import scripts.automod.review as RV
    S.append_event({"event": "backlog_triage", "item_id": 43, "verdict": "confirmed",
                    "surface": "vault", "acceptance": "a",
                    "acceptance_clauses": ["a"]}, path=S.LEDGER_PATH)
    write_item(items, 43, ["a"])
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": False, "error": "backend 503"})
    monkeypatch.setattr(V, "GRADER", RV.grade_vault)
    _prose(vault, SKILL_AT_FOUR, commit=True)
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n")
    with pytest.raises(V.VaultRoundError):
        V.land(["skills/foo/SKILL.md"], "#2036 vault-surface prose", item_id=43)
    reviews = _events("vault_review")
    assert reviews[0]["kind"] == "skipped" and "backend 503" in reviews[0]["findings"]
    lands = _events("vault_land")
    assert lands[0]["guards"]["refuse"] is True and lands[0]["ok"] is False
    assert lands[0]["item_id"] == 43


# ── #2040: a vault round cannot be refused for citing evidence its own edit removed ──
#
# `grade_vault` called `parse_review` without `changed_paths`, so `evidence_of_absence`'s
# "the first path token is one the diff touched and is not on disk" arm was unreachable
# on this surface while `gate.py:2274` and `review_tools.py:159` both pass it. #2038 spent
# both attempts on it: clauses [1 partial, 2 partial, 3 met, 4 met, 5 met] twice, the
# refusal text naming the rail and not the diff, over a staged deletion whose witness is
# the file's absence. Those two rows are the witness committed at
# tests/fixtures/promotions_vault_review_rows_2026-10-01-item2038.jsonl.

#: The leaf `automod_vault_land` for #2038 deleted and the grader cited, verbatim from the
#: ledger row: `paths` carries it, the disk does not.
DELETED_CITATION = ("_pipeline/tmp/pt1018/test_run_records_age_by_frontm0"
                    "/autonomy-runs/24/run_24_20260903_120000.md")

#: The two rows, in the tree. The review rung grades from a candidate checkout with a
#: stub HOME, so a claim that lives only in a live state file cannot be opened by the
#: reader who has to judge it; these bytes travel with the diff.
WITNESS_2038_REVIEW = repo() / "tests/fixtures/promotions_vault_review_rows_2026-10-01-item2038.jsonl"

#: The same grading's second refusal, whose evidence was the directory alone. `_pipeline`
#: is in NO path list, so no `changed_paths` argument could have saved it — the only shape
#: that answers a removal at directory level is the `(absent)` marker, which the vault
#: prompt never named. That is why half of this fix is prose to the grader.
DIRECTORY_CITATION = "_pipeline"


def _deleted_paths_grader(path: str) -> dict:
    """A grader object whose one clause is `met` on a removal, citing `path`.

    The real answer #2038 got twice, not one invented to be refused: `met`, `how_verified:
    read`, no test node (the vault prompt says leave it empty), and evidence that is a
    path the lander is in the middle of deleting.
    """
    return {"premise": "sound", "summary": "the scratch tree is gone", "test_honesty": [],
            "seams_unverified": [],
            "clauses": [{"clause": 1, "verdict": "met", "evidence_path": path,
                         "evidence_line": 0, "test_node_id": "",
                         "how_verified": "read", "note": "witness is the path's absence"}]}


def _grade_a_deletion(vault, items, monkeypatch, *, evidence: str,
                      paths: list[str]) -> tuple[str, str, list[dict]]:
    """One real `grade_vault` over a one-clause deletion item, grader answer fixed.

    The scratch paths are written and then removed wholesale, so the tree the rails read is
    the tree a deletion land stands in: the leaf listed by the lander and gone from disk,
    and no `_pipeline` directory left behind either — which is the state #2038's own
    findings record (`test -e ~/obsidian/_pipeline -> ABSENT`). Leaving the emptied
    directory standing would quietly change the question, because a directory on disk
    resolves as an evidence path and the far-wall node below would pass for the wrong
    reason.
    """
    import scripts.automod.review as RV
    # The rails must read ONLY the tree under review. `REVIEW_EVIDENCE_ROOTS` names the
    # live vault, where `_pipeline/` still stands after #2038's attempt 2 was reverted, so
    # unisolated these nodes would grade against the machine and not against the deletion.
    monkeypatch.setattr(RV, "REVIEW_EVIDENCE_ROOTS", ())
    write_item(items, 2038, ["the tracked `_pipeline/*` scratch paths are gone from the vault"])
    import shutil
    target = vault / DELETED_CITATION
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("run record\n", encoding="utf-8")
    shutil.rmtree(vault / "_pipeline")
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {
        "ok": True, "structured": _deleted_paths_grader(evidence)})
    return RV.grade_vault(item_id=2038, paths=paths, diff="D " + DELETED_CITATION, attempt=1)


def test_a_deletion_clause_citing_a_removed_but_listed_path_stays_met(vault, items,
                                                                     monkeypatch):
    """Clause 1: the lander's own path list is what makes a removal citable.

    Across the boundary the round actually crosses — `land()` hands `_vault_review` its
    normalised `paths`, which reaches `grade_vault` and had stopped there. Before the fix
    this returns `retry` with clause 1 `partial`; after it, the clause keeps `met` and the
    waiver is recorded rather than silent, because a waived rail that leaves no trace is
    indistinguishable from a rail that was never there.
    """
    kind, findings, clauses = _grade_a_deletion(
        vault, items, monkeypatch, evidence=DELETED_CITATION,
        paths=[".gitignore", DELETED_CITATION])
    assert kind == "pass", f"a removal was refused for its own witness: {findings}"
    assert [c["verdict"] for c in clauses] == ["met"], clauses
    assert DELETED_CITATION in clauses[0]["accepted"][0], clauses
    assert "evidence of absence" in clauses[0]["accepted"][0], clauses


def test_a_deletion_clause_citing_an_unlisted_unmarked_path_is_still_refused(vault, items,
                                                                            monkeypatch):
    """The narrowing has a far wall: a removal whose evidence is a bare directory the
    lander never listed and never marks is still refused.

    This is #2038's attempt-1 clause 2 verbatim — `_pipeline`, a prefix of the listed
    leaves but not itself one of them. Same fixture, same real `grade_vault`, one word
    changed in the citation, and the round comes back: absence is admissible when the
    change is named, not whenever a path looks like one.
    """
    kind, findings, clauses = _grade_a_deletion(
        vault, items, monkeypatch, evidence=DIRECTORY_CITATION,
        paths=[".gitignore", DELETED_CITATION])
    assert kind == "retry", f"the absence rail stopped refusing anything: {findings}"
    assert [c["verdict"] for c in clauses] == ["partial"], clauses
    assert "evidence_path missing or not on disk" in findings, findings
    assert "'_pipeline'" in findings, "the refusal must name what the grader wrote"


#: Where the durable copy lives, beside the vault's own dated extracts. NOT
#: `backlog/data/promotions.jsonl`, which the item named: that file is another job's
#: rolling mirror, 28678 lines against a live ledger of 28824, and it carries ZERO
#: `vault_review` rows for item 2038 — quoting it could not have reproduced these
#: figures, which is what the last round was refused for asserting.
WITNESS_VAULT_PATH = "backlog/data/2026-10-01.2040-vault-review-witness.jsonl"


def _witness_rows(text: str) -> list[dict]:
    """The two `vault_review` rows for item 2038, and only those, as graded."""
    import json
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    assert [r["event"] for r in rows] == ["vault_review"] * len(rows)
    assert [r["item_id"] for r in rows] == [2038] * len(rows)
    return rows


def _assert_the_rail_refused_them(rows: list[dict]) -> None:
    """What the rows say, in the words the refusal was written in.

    Both attempts carry clauses [1,2] `partial` and [3,4,5] `met`, both block, and the
    findings name the rail rather than the work. The leaf attempt 1's clause 1 cited is
    IN that row's `paths` array — `vault_round` had already handed it to `grade_vault`,
    which is the whole of half (a). The bare `_pipeline` that clause 2 cited is NOT: a
    directory is in no path list, so no argument could admit it and only the prompt's
    marker answers it, which is the whole of half (b). Attempt 2 reverted all 64 staged
    paths, so the rail refused a tree that had already done the work twice over.
    """
    assert [r["attempt"] for r in rows] == [1, 2]
    assert [r["kind"] for r in rows] == ["retry", "retry"]
    assert [r["blocking"] for r in rows] == [True, True]
    for r in rows:
        assert [(c["clause"], c["verdict"]) for c in r["clauses"]] == [
            (1, "partial"), (2, "partial"), (3, "met"), (4, "met"), (5, "met")]
        assert DELETED_CITATION in r["paths"], "the cited leaf is in the lander's own list"
        assert DIRECTORY_CITATION not in r["paths"], "a directory is never a listed path"
        assert "evidence_path missing or not on disk" in r["findings"]
    assert "'_pipeline'" in rows[0]["findings"], "attempt 1 cited the bare directory"
    assert "_pipeline/tmp/pt1018/usagecurrent" in rows[1]["findings"]
    assert "_pipeline/tmp/pt1018/usagecurrent" in rows[1]["paths"]
    assert len(rows[0]["reverted"]) == 0 and len(rows[1]["reverted"]) == 64
    assert len(rows[0]["paths"]) == 64, "63 `_pipeline/*` leaves plus .gitignore"


def test_the_witness_rows_are_in_the_tree_and_the_vault_copy_is_the_same_bytes():
    """#2040's evidence, readable from the bytes the review can open.

    A `vault_review` citation is resolved against the vault, so the durable copy goes
    there and the assertion below resolves it the way the rail will — through
    `review._resolve_in_worktree` at the vault root the guards themselves use. What this
    node refuses is the last round's failure: prose quoting a ledger that had already
    moved. Every figure #2040 quotes is re-derived here from the bytes — two rows,
    16506 of them — and nothing else is claimed.

    No digest is written down as a literal anywhere in this file, and that is a mechanism,
    not tidiness: the review rung validates a bare hex token in a grader note as a commit
    of the tree under review, so quoting a file checksum in prose made the rung report
    itself unreliable — "note cites commit <the checksum>, which `git cat-file -t` does
    not resolve" — over a round whose code was graded sound. Identity is pinned instead as
    bytes a reader can re-measure with `wc`, and as a digest COMPARED between the two
    copies, which catches them diverging without naming either with a number a citation
    validator can mistake for the repo's history.
    """
    import hashlib
    import scripts.automod.review as RV

    raw = WITNESS_2038_REVIEW.read_bytes()
    assert len(raw) == 16506, f"the witness bytes changed: {len(raw)}"
    rows = _witness_rows(WITNESS_2038_REVIEW.read_text(encoding="utf-8"))
    assert len(rows) == 2
    _assert_the_rail_refused_them(rows)

    # Where a vault is reachable, its committed copy must be the same bytes; where one
    # is not (`REVIEW_EVIDENCE_ROOTS` is `~/obsidian`, and a candidate checkout under a
    # stub HOME has none), the node reports the same green for the claims that hold
    # everywhere. No `pytest.skip`: a skipped node is a claim the review rung cannot
    # read, and #2040's clause 4 was refused once for exactly that. The path is spelled
    # out rather than read back from `WITNESS_VAULT_PATH`, so a constant that moved
    # cannot take its own assertion with it.
    for vault in RV.REVIEW_EVIDENCE_ROOTS:
        if not vault.is_dir():
            continue
        durable = vault / "backlog/data/2026-10-01.2040-vault-review-witness.jsonl"
        assert durable.is_file(), f"the durable copy is not on the vault's main: {durable}"
        assert hashlib.sha256(durable.read_bytes()).hexdigest() == \
            hashlib.sha256(raw).hexdigest(), "the two copies diverged"
        assert len(_witness_rows(durable.read_text(encoding="utf-8"))) == 2
        assert RV._resolve_in_worktree(WITNESS_VAULT_PATH, vault) == WITNESS_VAULT_PATH


def test_the_waiver_reaches_the_promotions_row_beside_the_verdict(vault, items,
                                                                 monkeypatch):
    """The seam #2040 crossed on paper and did not: waiver -> ledger, end to end.

    `review.grade_vault` records an admitted absence into the clause, and the row the
    promotions ledger keeps is built by a projection in `vault_round` that carried
    `clause`, `verdict` and `subject` and nothing else — so the record #2040's own fix
    produces died one function before the only place a vault landing is written down. A
    `met` earned from a deleted file then looked exactly like a `met` earned from a file
    on disk, and the round is unreadable to the next reader except by re-running it.
    This drives the real `land()` with the real `grade_vault` — lander, grader, rails,
    projection, ledger — over a working tree whose staged change is a deletion, which is
    the shape #2038 was in.
    """
    import scripts.automod.review as RV
    monkeypatch.setattr(RV, "REVIEW_EVIDENCE_ROOTS", ())
    write_item(items, 2038, ["the tracked `_pipeline/*` scratch paths are gone from the vault"])
    S.append_event({"event": "backlog_triage", "item_id": 2038, "verdict": "confirmed",
                    "surface": "vault", "acceptance": "a",
                    "acceptance_clauses": ["the tracked `_pipeline/*` scratch paths are "
                                           "gone from the vault"]}, path=S.LEDGER_PATH)
    target = vault / DELETED_CITATION
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("run record\n", encoding="utf-8")
    git(vault, "add", "-A")
    git(vault, "commit", "-q", "-m", "the scratch file, tracked at HEAD")
    target.unlink()                      # the land's change IS the deletion
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {
        "ok": True, "structured": _deleted_paths_grader(DELETED_CITATION)})
    monkeypatch.setattr(V, "GRADER", RV.grade_vault)
    out = V.land([DELETED_CITATION], "#2038 delete the tracked scratch paths", item_id=2038)
    assert out["review"] == "pass", out
    rows = _events("vault_review")
    assert len(rows) == 1, rows
    clause = rows[0]["clauses"][0]
    assert clause["verdict"] == "met", clause
    assert DELETED_CITATION in clause["accepted"][0], clause
    assert "evidence of absence" in clause["accepted"][0], clause
    # And on the row the landing itself writes: `vault_land.review_clauses` is what
    # `backlog.vault_review_outcome` reads back, so the waiver has to survive that
    # projection too — a reader of the landing alone must be able to see which rail
    # the deletion clause stood on.
    lands = _events("vault_land")
    assert len(lands) == 1 and lands[0]["ok"] is True, lands
    landed = [c for c in lands[0]["review_clauses"] if c["clause"] == 1]
    assert len(landed) == 1 and landed[0]["verdict"] == "met", landed
    assert "evidence of absence" in landed[0]["accepted"][0], landed


# --------------------------------------------------------------------------- #
#  #2042: the probe has to reach a verdict inside the land budget, and a probe
#  that cannot reach one has to say so in numbers.
#
#  Three real item-bound lands armed the #2036 probe — #2040 at
#  2026-10-01T23:26:32Z, #2027 at 23:36:50Z, #2038 at 2026-10-02T01:03:30Z — and
#  every one recorded `guards.state = "skipped"`, `refuse = False`,
#  `candidate.ran = 0` and the reason "the proposed-vault run answered nothing:
#  timed out after 300s". `checked` has never appeared on a `vault_land` row.
#  Two causes, one fix: the probe launched its child pytest while a gate `tests`
#  rung was holding the box on eight xdist workers, and the budget was quoted per
#  run so the queue for one was charged to the other.
# --------------------------------------------------------------------------- #

#: A vault-reading guard that says one line at import and then hangs. The line is
#: written at import, not inside the test, because pytest redirects the child's
#: descriptors at session start and a hung test never gets to flush — the bytes
#: that survive in the pipe the probe is reading are the ones written before the
#: kill, which is exactly what clause 2 asks the ledger to carry.
GUARD_SRC_HANGS = '''"""A vault-reading guard that says one line and then never answers again."""
import os
import sys
import time
from pathlib import Path

import pytest

sys.stderr.write("GUARD-IS-HANGING prose 4, script 4\\n")
sys.stderr.flush()


@pytest.fixture
def _unused():
    yield None


def test_the_prose_states_the_count_the_script_prints():
    time.sleep(120)
    text = (Path(os.environ["LLOYD_VAULT_ROOT"]) / "skills" / "foo" / "SKILL.md").read_text()
    stated = [int(l.split()[-1]) for l in text.splitlines() if l.startswith("stores: ")]
    assert stated == [4], f"prose {stated}, script 4"
'''

#: A vault-reading file the gate's mark expression deselects and nothing else can:
#: it names `obsidian`, so `guard_selection` must pick it, and it fails if the
#: child is ever given a mark expression that does not mention its own marker.
#: Clause 4's witness that the child is handed the gate's CURRENT object and not a
#: copy of its text.
DECOY_SRC = '''"""A vault-reading guard the gate deselects: it names ~/obsidian and is marked."""
import pytest


@pytest.mark.probed_decoy
def test_a_decoy_that_must_never_run_under_the_gates_marks():
    raise AssertionError("the decoy ran: the child was not given the gate's mark expr")
'''

#: The three post-ship `vault_land` rows, re-read out of the promotions ledger on
#: 2026-10-02 and committed here so every number in the section below is a fact
#: about bytes in the tree rather than about a live file that moves.
WITNESS_2042 = repo() / "tests/fixtures/promotions_vault_land_rows_2026-10-02-item2042.jsonl"
WITNESS_2042_VAULT_PATH = "backlog/data/2026-10-02.2042-probe-timeout-witness.jsonl"
#: `git show 4bc93177:backlog/data/promotions.jsonl | wc -l`, the row-count figure #2042
#: quotes. The figure is unchanged from the one this item measured off the working-tree
#: copy at that path; what #2054 changed is where a reader gets it, because that copy is
#: retired — a refresh cost ~7.7 MB of vault-git history and #2043's 14-day window means
#: the sweep rewrites the ledger it was cut from, so the file could never be re-compared
#: again. The commit is the dated extract now, and it is pinned HERE as well as in
#: `tests/test_retention_sweep.py` for the reason that file names: a figure held in two
#: items moves both when it moves, and neither gets to be a sentence about the other.
LEDGER_WITNESS_COMMIT = "4bc93177"
LEDGER_WITNESS_PATH = "backlog/data/promotions.jsonl"
WITNESS_LEDGER_LINES = 28678
#: The `vault_land` rows those same bytes hold. None of them can carry a `guards` key:
#: 4bc93177 is #1975's third land, and #2036's guard — the thing that writes the key —
#: shipped on 2026-10-01T22:41:51Z, after every row in this ledger. That is the anti-marking
#: fact the node below asserts. The three rows #2042 quotes are in their own extract because
#: they post-date this copy entirely: the copy is cut from the ledger before the land
#: appends its row, so no refresh of a copy can ever hold the row of the land that made it.
WITNESS_LEDGER_VAULT_LAND_ROWS = 453


def test_a_killed_run_ledges_the_seconds_the_selection_and_its_own_tail(
        tmp_path, monkeypatch):
    """Clause 2 at the child boundary: the numbers a kill leaves behind, and which
    channel the tail came from.

    A deterministic stand-in for the kill — a `Popen` whose first `communicate`
    raises the `TimeoutExpired` the real one raises and whose second is the drain
    after the process-group kill. The #2042 node this one replaces built its
    `held` bytes into the exception it raised and then asserted those bytes came
    out: a literal the test itself made, over a channel a killed child may not
    even have. So the fake child here is parameterised over every source the kill
    can leave behind, and the cases below pin the ORDER they are consulted in —
    the seam a reviewer of this diff called unverified.

    What `TimeoutExpired.output` actually holds for a killed child is not
    asserted anywhere here, deliberately: the two measurements of it taken on this
    box disagreed (`_tail`'s docstring names both), and a test that predicted it
    would be a claim about CPython rather than about this module. `_run_selection`
    is written to work whichever it turns out to be, and that is what the case
    table proves.

    The point beside it is not that a timeout is reported (the old code did that)
    but what it reports: the three real rows said only `timed out after 300s`,
    which cannot be told apart from a hang by anyone reading the ledger.
    """
    import signal
    import subprocess as SP

    held = "collected 7065 items\n.... GUARD-IS-HANGING prose 4, script 4\n"

    class KilledController:
        """The run's controller, still alive when the budget gives up on it.

        `carries` is what the exception hands back, `drain` what the post-reap
        `communicate` does — the two things a killed child can and must not be
        assumed to agree about.
        """

        pid = 4242          # nothing owns this pid; `killpg` is patched below
        returncode = None

        def __init__(self, carries=(None, None), drain=(None, None)) -> None:
            self.carries, self.drain = carries, drain
            self.communicates = 0

        def communicate(self, timeout=None):
            self.communicates += 1
            if self.communicates == 1:
                raise SP.TimeoutExpired(["pytest"], timeout,
                                        output=self.carries[0], stderr=self.carries[1])
            return self.drain          # what the drain after the reap takes back

        def wait(self, timeout=None) -> int:
            return -9

        def kill(self) -> None:
            pass

    signalled: list[int] = []
    monkeypatch.setattr(VG.os, "killpg", lambda pgid, sig: signalled.append(sig))

    def kill(carries=(None, None), drain=(None, None)) -> dict:
        child = KilledController(carries, drain)
        monkeypatch.setattr(VG.subprocess, "Popen", lambda cmd, **kw: child)
        out = VG._run_selection(Path(sys.executable), tmp_path,
                                ["tests/test_a.py", "tests/test_b.py"],
                                tmp_path / "vault", tmp_path / "data", None, 42.0)
        out["communicates"] = child.communicates
        return out

    r = kill(drain=(held, ""))
    assert r["ran"] == 0
    # The denominator and the cost, in the note itself: "over 2 vault-reading
    # file(s)" and the seconds it was given are what a future reader needs to tell
    # a hang from a selection that simply does not fit the budget.
    assert r["files"] == 2 and isinstance(r["seconds"], float)
    assert "over 2 vault-reading file" in r["note"], r["note"]
    assert "the 42s it was given" in r["note"], r["note"]
    assert "timed out after" in r["note"], r["note"]
    # The tail is the child's bytes, not a placeholder: this is the half the old
    # code failed, whose timeout branch returned `"excerpt": ""` unconditionally.
    assert "GUARD-IS-HANGING" in r["excerpt"], r
    assert "GUARD-IS-HANGING" in r["note"], r["note"]
    # The run was given up on by GROUP, and reaped: SIGTERM for the grace, SIGKILL
    # for the certainty, and a second `communicate` that ends the pipes and takes
    # the controller out of the process table. `test_a_killed_parallel_run_leaves_
    # no_worker_of_its_scratch_tree_alive` is the real-children witness of the same
    # rule; this half is the deterministic order.
    assert signalled == [signal.SIGTERM, signal.SIGKILL], signalled
    assert r["communicates"] == 2, "the killed child was left unreaped"
    # A clean reap adds nothing to the note: only a survivor is worth reporting.
    assert "; " not in r["note"], r["note"]

    # The seam above only shows the LAST channel winning, because that case had
    # nothing anywhere else. Which of the four the code consults first is the other
    # half of "where did this tail come from", and each row below is a killed child
    # that could have supplied the note from more than one place. The order is
    # `_run_selection`'s, and the bytes row is why `_tail` decodes: an exception
    # built by a bytes-mode pipe hands over `bytes` even when the reader wanted text.
    for carries, drain, want in (
            (("FROM-EXC-OUTPUT", "FROM-EXC-STDERR"),
             ("FROM-DRAIN-OUT", "FROM-DRAIN-ERR"), "FROM-EXC-OUTPUT"),
            ((None, "FROM-EXC-STDERR"),
             ("FROM-DRAIN-OUT", "FROM-DRAIN-ERR"), "FROM-EXC-STDERR"),
            ((None, None), ("FROM-DRAIN-OUT", "FROM-DRAIN-ERR"), "FROM-DRAIN-OUT"),
            ((None, None), (None, "FROM-DRAIN-ERR"), "FROM-DRAIN-ERR"),
            ((b"FROM-EXC-BYTES", None), (None, None), "FROM-EXC-BYTES"),
    ):
        got = kill(carries, drain)
        assert want in got["note"], (carries, drain, got["note"])
        assert want in got["excerpt"], (carries, drain, got["excerpt"])
    # And a killed child that left nothing anywhere says so, rather than being
    # reported as if a tail had been read and found empty.
    empty = kill()
    assert empty["excerpt"] == "" and "no output captured" in empty["note"], empty["note"]


def test_a_budget_that_cannot_hold_a_run_reports_it_before_starting_one(
        vault, guard_tree, probed):
    """Clause 1's other half: a wait is never charged to a run that cannot finish.

    One second of total budget, which is less than the five a pytest child needs to
    import and report. The probe must say that up front instead of starting a run
    whose kill is certain — and say it in a sentence that names the budget, the
    seconds left and the size of the selection, which is the same self-explaining
    rule clause 2 asks of a timeout.
    """
    _prose(vault, "---\nname: foo\n---\nstores: 4\n", commit=True)
    probed(guard_tree, vault, timeout=1.0)
    _prose(vault, "---\nname: foo\n---\nstores: 4\n\nReworded under a tight budget.\n")
    head = _head(vault)
    out = V.land(["skills/foo/SKILL.md"], "#2042 no room for a run", item_id=None)
    assert out["commit"], "a probe with no room must not stop the land"
    assert _head(vault) != head
    row = _events("vault_land")[0]["guards"]
    assert row["state"] == "skipped" and row["refuse"] is False
    assert "no run fits" in row["reason"], row
    assert "1 vault-reading file" in row["reason"], row
    assert row["seconds"] < VG.MIN_RUN_SECONDS, f"the rail spoke after {row['seconds']}s"
    assert "candidate" not in row, "no run started, so no run may be reported"


def test_a_probe_that_hangs_states_its_seconds_and_still_lands(vault, tmp_path, probed):
    """Clauses 2 and 3 together, on the path the three real rows died on.

    End to end this time, with a guard that really hangs and a budget short enough
    to kill it. What the ledger gets: a `skipped` (clause 3 — a probe that answers
    nothing is never a silent pass), the seconds the child was alive, the size of
    the selection it was running, and the whole-probe seconds beside the queue for
    the gate's tests slot. The land still goes through: the #2036 rule is that a
    non-answer does not hold a landing hostage.
    """
    tree = make_guard_tree(tmp_path / "hanging-tree", src=GUARD_SRC_HANGS)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n", commit=True)   # red at vault HEAD
    probed(tree, vault, timeout=12.0)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n\nReworded while the probe hangs.\n")
    head = _head(vault)
    out = V.land(["skills/foo/SKILL.md"], "#2042 the probe hangs", item_id=None)
    assert out["commit"] and _head(vault) != head
    row = _events("vault_land")[0]["guards"]
    assert row["state"] == "skipped" and row["refuse"] is False
    assert row["candidate"] == {"ran": 0, "failed": 0,
                                "seconds": row["candidate"]["seconds"], "files": 1}
    # #2046: the new keys are ROW-level, so the per-run block #2042 shaped keeps
    # exactly its four keys — this is the pin that says so rather than leaving it to
    # the dict equality above, which a stray fifth key inside `candidate` would not
    # catch only if it happened to equal the placeholder. And the row above now
    # states the worker count the hung run was launched with, which is what made
    # this row unreadable before: `ran: 0` with no witness of the parallelism.
    assert set(row["candidate"]) == {"ran", "failed", "seconds", "files"}, row["candidate"]
    # `> 1` and not an equality: this box's count is `automod.gate.test_workers`, a
    # key this node does not own, and reading it through the gate's accessor here
    # perturbs the module state its neighbours in the same xdist worker monkeypatch.
    # What the row must prove is that a hung run's row states a PARALLEL count; the
    # exact value is pinned by the node that patches the worker decision.
    assert row["workers"] > 1, row
    assert "parallel_retry" not in row, "no re-ask ran, so the row may not imply one"
    assert "answered nothing" in row["reason"] and "timed out after" in row["reason"]
    assert "over 1 vault-reading file" in row["reason"], row["reason"]
    # The child spent real seconds and was then killed for spending them: that is
    # the difference between "the run was slow" and "the budget was already gone",
    # which is the ruling #2042 leaves to the owed-check job and which the old row
    # could not support either way. The floor is the arithmetic one, not a guess at
    # how much of the 12 s is left: `run_budget()` refuses to start a run under
    # `MIN_RUN_SECONDS`, so a run that started had at least that much and a run that
    # timed out spent all of it. The first cut of this line asked for >= 8.0 and
    # failed under the gate's own `-n 8` (`seconds: 7.9`): the worktree, the mirror
    # and — since #2044 — the interpreter boot of the `import xdist` probe all come
    # out of the same deadline before the child is handed what is left. Asserting a
    # fixed share of the budget would make this node measure the box's load; the
    # rail's own floor measures the code.
    assert row["candidate"]["seconds"] >= VG.MIN_RUN_SECONDS, row["candidate"]
    assert row["seconds"] >= row["candidate"]["seconds"], row
    # What the rail promises about the queue is that it is REPORTED and that the
    # probe stays inside its own deadline (plus the bounded reap), not that the wait
    # is short. Two reasons not to bound the length here. The measured one: the `-n
    # 8` run that broke the previous line lost ~4 s of this node's 12 s BEFORE the
    # child was spawned (`seconds: 7.9`), and a bound on what the child gets is a
    # bound on what the box let the probe prepare. The structural one: the tests lock
    # is one real path (`S.GATE_TESTS_LOCK_PATH`, printed from `scripts.automod.state`
    # on 2026-10-02, not a per-test file), and a sibling node in this very file —
    # `test_the_probe_waits_for_the_gate_tests_slot_before_it_starts_a_child` — holds
    # it for eight seconds from a timer thread, so under `-n 8` how long this probe
    # may queue is decided by which worker got there first. The queue's cost is on
    # the row beside the run's, which is the half #2042 actually asked for.
    assert row["seconds"] <= 12.0 + VG.GROUP_TERM_GRACE_S + VG.REAP_DRAIN_S, (
        f"the probe overran its own 12s deadline by more than the bounded reap: "
        f"{row['seconds']}s, lock_wait_s={row.get('lock_wait_s')}")


def test_the_probe_waits_for_the_gate_tests_slot_before_it_starts_a_child(
        vault, guard_tree, probed, tmp_path):
    """Clause 1: the probe's pytest never runs alongside a gate `tests` rung.

    Timed from outside the probe, on the child's own spawn time: the interpreter it
    is handed is a one-line wrapper that touches a file and execs. Uncontended the
    child is spawned at once — the queue must not cost anything when the lock is
    free, or this fix would have traded a timeout for a stall. Contended, with the
    gate's own tests lock held here and released from a timer thread, the child may
    not exist before the release: the 300 s starts counting after the slot, not
    during the wait, which is the arithmetic that made all three real probes
    non-answers.
    """
    import threading
    started = tmp_path / "child-spawned"
    wrapper = tmp_path / "python-shim"
    wrapper.write_text(f'#!/bin/sh\n: > "{started}"\nexec {sys.executable} "$@"\n')
    wrapper.chmod(0o755)
    probed(guard_tree, vault, python=wrapper)

    _prose(vault, SKILL_AT_FOUR, commit=True)
    _prose(vault, "---\nname: foo\n---\nstores: 4\n\n#2042 uncontended.\n")
    t0 = time.time()
    out = V.land(["skills/foo/SKILL.md"], "#2042 the slot is free", item_id=None)
    assert out["guards"]["state"] == "checked" and out["guards"]["refuse"] is False
    assert out["guards"]["candidate"]["ran"] == 1, out["guards"]
    free_spawn = started.stat().st_mtime - t0
    assert "lock_wait_s" not in out["guards"], "a probe that did not queue said it did"

    started.unlink()
    held = S.Lock(S.GATE_TESTS_LOCK_PATH, owner="test-holding-the-gate-tests-rung")
    held.acquire()
    released: list[float] = []
    timer = threading.Timer(8.0, lambda: (released.append(time.time()), held.release()))
    timer.daemon = True
    timer.start()
    try:
        _prose(vault, "---\nname: foo\n---\nstores: 4\n\n#2042 contended.\n")
        t1 = time.time()
        out2 = V.land(["skills/foo/SKILL.md"], "#2042 the slot is busy", item_id=None)
    finally:
        timer.cancel()
        held.release()
    assert out2["guards"]["state"] == "checked" and out2["guards"]["refuse"] is False
    assert out2["guards"]["candidate"]["ran"] == 1, out2["guards"]
    assert released, "the timer never fired, so the contention this node names never existed"
    contended_spawn = started.stat().st_mtime - t1
    assert started.stat().st_mtime >= released[0] - 0.05, (
        f"the child started {released[0] - started.stat().st_mtime:+.1f}s around the "
        f"release: the probe ran its selection over a gate tests rung")
    # The load-robust half of the claim, and the one the clause is actually about:
    # the same land that spawned its child after `free_spawn` s of validators and
    # mirror took `contended_spawn` s to get there, because the queue was in front
    # of it. A loaded box inflates both terms equally; it cannot inflate one alone.
    assert contended_spawn - free_spawn >= 3.0, (
        f"uncontended child at {free_spawn:.1f}s, contended at "
        f"{contended_spawn:.1f}s: the hold changed nothing")
    assert out2["guards"]["lock_wait_s"] >= 1.0, out2["guards"]


def test_the_child_is_given_the_gate_mark_object_not_a_copy_of_its_text(
        vault, guard_tree, probed, monkeypatch):
    """Clause 4: the selection invariant survives, and it survives by identity.

    Two halves, both behavioural. First the mark expression: a decoy file that
    `guard_selection()` must pick (it names `obsidian`) and that fails if pytest
    runs it is added to the tree, and the gate's own constant is then changed — not
    this file's copy of it, the attribute on `scripts.automod.gate` that
    `_gate_tools` imports. The child obeys the change, so the string in its argv
    came through the object; a retyped literal would hand the child the old text,
    run the decoy and refuse a land that agrees with the tree.

    Second the mirror, which the node below it already pins
    (`test_the_probe_judges_a_copy_so_a_writing_guard_cannot_reach_the_live_vault`):
    `LLOYD_VAULT_ROOT` names a throwaway copy, never `~/obsidian`.
    """
    import scripts.automod.gate as GATE
    (guard_tree / "tests/test_decoy.py").write_text(DECOY_SRC)
    git(guard_tree, "add", "-A")
    git(guard_tree, "commit", "-q", "-m", "a decoy the gate's marks deselect")
    assert VG.guard_selection(guard_tree) == ["tests/test_decoy.py", "tests/test_guard.py"], \
        "the decoy is not in the file set the probe runs"
    monkeypatch.setattr(GATE, "TESTS_MARK_EXPR",
                        f"{GATE.TESTS_MARK_EXPR} and not probed_decoy")
    _prose(vault, SKILL_AT_FOUR, commit=True)         # the prose agrees: nothing to refuse
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 4\n\n#2042 the marks are the gate's.\n")
    out = V.land(["skills/foo/SKILL.md"], "#2042 the mark expr is the gate's object",
                 item_id=None)
    cand = out["guards"]["candidate"]
    assert cand["ran"] == 1 and cand["failed"] == 0, (
        f"the decoy ran, so the child was not handed the gate's mark object: {cand}")
    assert cand["files"] == 2, f"the selection is not guard_selection()'s file set: {cand}"
    assert out["guards"]["refuse"] is False


def test_the_probe_timeout_witness_is_the_three_rows_the_item_quotes():
    """#2042's premise, re-derived from bytes a reader can hold in their hands.

    Every figure the item states about its own witness is recomputed here: three
    `vault_land` rows after #2036 shipped, each `guards.state == "skipped"`,
    `refuse` false and `candidate.ran == 0`, each reason the same 300-second
    non-answer, every one a real item-bound land that went through unjudged. The
    rows deliberately carry no `seconds` and no `excerpt`: that absence is the
    blindness the item filed, and a witness that grew the new fields would stop
    being evidence of it.

    The row-count figure the item asks to be reproduced comes from the same committed
    bytes that command read, and is pinned to the 28678 the item quotes — but read out of
    the vault's history at `4bc93177`, not out of `backlog/data/promotions.jsonl` on disk.
    #2050 owed 5 retired that working-tree copy (a refresh costs ~7.7 MB of permanent
    vault-git history on a 110 MB `.git` with no remote, and #2043's 14-day window means
    the sweep rewrites the very ledger the copy was cut from, so a prefix-compare against it
    was going to stop being askable whatever this node did), and #2054 re-aimed this node at
    the commit instead. Those bytes hold 453 `vault_land` rows and none of them
    post-dates the ship — which is why the three rows this report quotes are committed as
    their own extract, the way #2040 committed its review rows, instead of being looked for
    inside the ledger.

    Two things about that ordering are asserted rather than recited, because the prose
    above is the rotting kind of claim if anything is: the row count and the `vault_land`
    count are taken from ONE read of ONE blob, so they cannot describe two ledgers; and the
    anti-marking assert runs over both generations of committed bytes this item has — the
    453 `vault_land` rows of the frozen ledger, and the three rows of #1975's extract,
    whose newest is the very commit being read here. Both sets are pre-#2036 and so carry
    no `guards` key at all. The three rows this node is about are the opposite: they carry
    `guards`, and `state == "skipped"` inside it, and that is the blindness, which is why
    they are an extract of their own and are asserted above rather than below.

    As #2040's witness node records, no digest is written down as a literal here: a
    bare hex token in a grader note is validated as a commit of the tree under
    review, so identity is pinned as bytes to re-measure and a digest COMPARED
    between the two copies.
    """
    import hashlib
    import json
    import scripts.automod.review as RV
    import test_retention_sweep as RS

    raw = WITNESS_2042.read_bytes()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    assert [r["event"] for r in rows] == ["vault_land"] * 3
    assert [r["item_id"] for r in rows] == [2040, 2027, 2038]
    assert [r["created_at"] for r in rows] == ["2026-10-01T23:26:32Z",
                                               "2026-10-01T23:36:50Z",
                                               "2026-10-02T01:03:30Z"]
    assert all(r["created_at"] >= "2026-10-01T22:41:51Z" for r in rows), \
        "a row predates #2036's ship, so the probe could not have been armed"
    assert [r["ok"] for r in rows] == [True, True, True], "these lands went through"
    for r in rows:
        g = r["guards"]
        assert g["state"] == "skipped" and g["refuse"] is False
        assert g["candidate"] == {"ran": 0, "failed": 0}, r["item_id"]
        assert g["reason"] == ("the proposed-vault run answered nothing: "
                               "timed out after 300s"), r["item_id"]
        assert "seconds" not in g and "excerpt" not in g, (
            f"#2040's row grew the fields #2042 is filing the absence of: item "
            f"{r['item_id']} is no longer the witness")

    for vault in RV.REVIEW_EVIDENCE_ROOTS:
        if not vault.is_dir():
            continue
        durable = vault / WITNESS_2042_VAULT_PATH
        assert durable.is_file(), f"the durable copy is not on the vault's main: {durable}"
        assert len([l for l in durable.read_text().splitlines() if l.strip()]) == 3
        assert hashlib.sha256(durable.read_bytes()).hexdigest() == \
            hashlib.sha256(raw).hexdigest(), "the two copies diverged"
        # One read of one blob, and both figures come off it: a line count taken from the
        # commit and a `vault_land` count taken from somewhere else would be two numbers
        # about two ledgers, which is the defect this node's own history is full of.
        raw_ledger = RS._blob_at(vault, LEDGER_WITNESS_COMMIT, LEDGER_WITNESS_PATH)
        lines = raw_ledger.splitlines()
        assert len(lines) == WITNESS_LEDGER_LINES, (
            f"`git show {LEDGER_WITNESS_COMMIT}:{LEDGER_WITNESS_PATH} | wc -l` is "
            f"{len(lines)}, not the {WITNESS_LEDGER_LINES} the item quotes. The copy this "
            "figure was measured on is retired, so the commit is where it lives: a move "
            "here means both this node and test_retention_sweep's pin of it need the new "
            "number, and the named commit needs re-picking with them")
        vl = [json.loads(ln) for ln in lines if b'"vault_land"' in ln]
        assert len(vl) == WITNESS_LEDGER_VAULT_LAND_ROWS, (
            f"the frozen ledger holds {len(vl)} `vault_land` rows, not "
            f"{WITNESS_LEDGER_VAULT_LAND_ROWS}: the bytes at "
            f"{LEDGER_WITNESS_COMMIT} are not the copy the extract below was cut from")
        assert all("guards" not in r for r in vl), (
            f"a `vault_land` row at {LEDGER_WITNESS_COMMIT} carries a `guards` key, so the "
            "frozen ledger now holds post-#2036 rows and the extract above is no longer the "
            "only witness of the blindness #2042 filed")
        # The same anti-marking fact over the other committed extract this item has: #1975's
        # three `vault_land` rows. The property is that a land before #2036's guard existed
        # wrote no marking AT ALL — #2036's premise, and the reason this node's own three rows
        # carry `guards` and these do not.
        markless = [json.loads(l) for l in WITNESS.read_text().splitlines() if l.strip()]
        assert len(markless) == 3, f"#1975's extract is {len(markless)} rows, not 3"
        assert all("guards" not in r for r in markless), (
            "#1975's extract grew a `guards` key, so it is no longer the pre-#2036 "
            "generation this node's anti-marking claim rests on")

        # The relation between the two committed witnesses, and it is NOT containment. The
        # copy is refreshed from the live ledger BEFORE the land appends its own row, so a
        # copy committed by a land can never contain the row for that land: #1975's first land
        # is in these bytes, its own final row is not. That lag is the exact mechanism #2042
        # is the report of — a `vault_land` row invisible to the guard that was supposed to
        # see it — and it is why the three rows this node is about are committed as an extract
        # rather than looked for in the copy. A reader who assumed containment would run the
        # lookup, find nothing, and conclude the land never happened.
        frozen_commits = {r.get("commit") for r in vl}
        assert markless[0]["commit"] in frozen_commits, (
            f"the first of #1975's lands ({markless[0]['commit'][:8]}) is not a `vault_land` "
            "row of the copy that land refreshed, so the copy and the extract are not the "
            "same ledger and one of the two figures above is about a file nobody is citing")
        assert markless[-1]["commit"] not in frozen_commits, (
            f"the copy committed at {LEDGER_WITNESS_COMMIT} now contains the row for that same "
            "land, which the refresh order cannot produce — the copy is cut from the ledger "
            "before the land appends its row. If this ever passes, the lag described above has "
            "stopped being real and committing an extract was not necessary")


# --------------------------------------------------------------------------- #
#  #2054 clause 5: after the retire, nothing under tests/ or scripts/ opens the
#  promotions mirror as a file; the only surviving mentions are the tmp-vault
#  stand-ins, the two history-path constants, and prose that names a commit.
#
#  Why this is a node and not the proving command: `git grep -n
#  'backlog/data/promotions.jsonl' -- tests scripts` is a LIST, and a list is only
#  an acceptance test if something says which entries are allowed and why. Every
#  one of the four surviving code lines names the path to a `git show` argument or
#  to a tmp vault the node itself built; the moment anyone writes a line that
#  joins it onto `vault_root()` or opens it, the string's neighbours change, and
#  this node is what reddens. The mirror is a 33 MB file whose working-tree copy
#  #2050 owed 5 retired, so a fresh reader is not untidy — it is the cost paid
#  again, plus a figure that stops being reproducible the day task 79 rewrites the
#  live ledger (#2043's window).
# --------------------------------------------------------------------------- #

#: The ways a mention of a path stops being a citation and becomes a READ: the literal
#: joined onto a resolved root, or the file opened. Each is checked on the line that holds
#: the path, which is where a reader has to spell its root or its call.
_MIRROR_READER_MARKS = ("read_text", "read_bytes", "open(", "vault_root(", "unlink(",
                        "write_text", "write_bytes", '"backlog" / "data"',
                        "'backlog' / 'data'", ".readlines", ".readline(")

#: The files permitted to name the retired mirror at all, and what each one's mention is.
#: A NEW file appearing here is the event this node is for: it means somebody started
#: talking about the retired copy again. The list is deliberately about WHERE a mention may
#: live and not about line counts, because the counts move with every comment edit while the
#: set of readers is the property the clause is stated over.
_MIRROR_MENTION_FILES = {
    "scripts/automod/vault_guards.py":
        "the agreement probe's docstring naming the copy as the pinned-witness shape an "
        "acknowledgement was written for (#2049's mechanism text). #2064 re-timed it: it now "
        "says the copy is RETIRED and the excuse stands for any pinned witness whose test file "
        "names the path, which "
        "test_the_agreement_excuse_text_says_the_copy_is_retired_not_current pins",
    "scripts/groundskeeper/retention-sweep.py":
        "the store-13 paragraph, whose figures #2054 re-aimed at vault commit "
        f"{LEDGER_WITNESS_COMMIT} and whose retire that paragraph records",
    "tests/fixtures/promotions_vault_land_rows_2026-10-01-item1975.jsonl":
        "a `vault_land` row recording the land that wrote the copy: the ledger's own history, "
        "which is the reason the path cannot be erased from the repository",
    "tests/test_failure_ledger_witness.py":
        "prose in #2178's node naming the path #2175 clause 5 sent its extract to, which is "
        "the fact the node is about. No reader: the file's first node is the one that keeps "
        "the working-tree path empty, and the bytes it checks are at a dated name",
    "tests/test_automod_hardening.py":
        "ACK_MIRROR, a string the ack-rail nodes pass to a lander they have all replaced",
    "tests/test_automod_vault_round.py":
        "this file: the history-path constant, the #2049 MIRROR_PATH stand-in, and prose "
        "about the retire",
    "tests/test_retention_sweep.py":
        "test_retention_sweep: its history-path constant and the prose of the nodes #2054 "
        "re-cut",
    "tests/test_vault_sync_refusals.py":
        "the #1903 generation's citations — 10,917,874 bytes at vault 0d96fdb0 — and the "
        "transport file-type claim about that file",
}


def test_no_reader_under_tests_or_scripts_opens_the_retired_mirror():
    """#2054 clause 5, pinned as four questions instead of a list someone reviewed.

    The clause's own instrument is `git grep -n 'backlog/data/promotions.jsonl' -- tests
    scripts`, and this node runs the same walk over `git ls-files` (tracked files, both
    trees, so a file no test imports cannot hide in it). A grep is only an acceptance test
    once something decides which lines are admissible, so every hit is asked:

    1. Does the file of the hit belong to the set permitted to mention the path at all? A
       new file is somebody resuming the copy's career, and it is refused by name.
    2. Does the line READ it — the literal joined onto a resolved vault root, or the file
       opened? That is the clause, and it is refused outright.
    3. Are there hits at all? A scan that finds none has stopped checking, which is the
       failure mode of every absence test this loop has written: the pattern was wrong, or
       the citations it protects were deleted, and the node reports green either way.
    4. Do the tmp-vault stand-ins survive, in the files that own them? Getting a clean grep
       by deleting `MIRROR_PATH` and `ACK_MIRROR` would produce a green clause 5 by removing
       the only tests of the agreement probe's refusal of an unacknowledged pinned witness —
       the rail #2049 built and #2054's owed delete may have to be `--ack`'d through.

    The surviving mentions are citations by construction: `git show <sha>:<path>` arguments
    (which read the object store, not the working tree), two string constants handed to
    `git show`, two stand-ins rooted in a tmp vault each node builds itself, and prose that
    names a commit or records the retire.
    """
    import subprocess as SP

    root = repo()
    listed = SP.run(["git", "-C", str(root), "ls-files", "--", "tests", "scripts"],
                    capture_output=True, text=True, check=True).stdout.splitlines()
    hits: list[tuple[str, int, str]] = []
    for rel in listed:
        p = root / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue                       # binary: a path inside it is data, not a reader
        for n, line in enumerate(text.splitlines(), 1):
            if "backlog/data/promotions.jsonl" in line:
                hits.append((rel, n, line.strip()))

    assert hits, (
        "no tracked file under tests/ or scripts/ names `backlog/data/promotions.jsonl`, so "
        "this node is checking nothing. The retired-history citations are what it protects "
        "(#1903's 10,917,874 bytes at vault 0d96fdb0, the two `git show` path constants, the "
        "tmp-vault stand-ins); if they have been deleted the clause-5 grep is green for the "
        "wrong reason")

    from_new_files = sorted({rel for rel, _, _ in hits} - set(_MIRROR_MENTION_FILES))
    assert not from_new_files, (
        f"these files name the retired promotions mirror and are not in the permitted set: "
        f"{from_new_files}. Every file that may mention "
        "`backlog/data/promotions.jsonl` is listed with its reason in `_MIRROR_MENTION_FILES` "
        "— if this is a legitimate new citation, add the file and say what its mention is; if "
        "it is a reader, the figures belong at vault commit "
        f"{LEDGER_WITNESS_COMMIT}, read with `git show`")

    readers = [(rel, n, line) for rel, n, line in hits
               if any(mark in line for mark in _MIRROR_READER_MARKS)]
    assert not readers, (
        "these lines open or root the retired promotions mirror instead of citing it: "
        f"{readers}. The working-tree copy is retired (#2050 owed 5) and the figures live in "
        "the vault's history; a file-level reader re-buys the 33 MB copy and pins a number "
        "the sweep rewrites, which is what #2043's window guarantees")

    hardening = (root / "tests/test_automod_hardening.py").read_text(encoding="utf-8")
    this_file = Path(__file__).read_text(encoding="utf-8")
    assert 'ACK_MIRROR = "backlog/data/promotions.jsonl"' in hardening, (
        "test_automod_hardening's ACK_MIRROR stand-in is gone, so the ack rail's "
        "pinned-witness nodes no longer exist to be run")
    assert 'MIRROR_PATH = "backlog/data/promotions.jsonl"' in this_file, (
        "this file's MIRROR_PATH stand-in is gone, so the agreement probe's refusal of an "
        "unacknowledged pinned witness — #2049's subject — is no longer tested")
    print(f"clause 5: {len(hits)} mentions in {len({h[0] for h in hits})} files, "
          f"{len(readers)} of them readers, none outside the permitted set")


def test_the_agreement_excuse_text_says_the_copy_is_retired_not_current():
    """#2064 clause 4: the excuse's own text must not still describe the mirror as live.

    `agreement()`'s docstring is the only description of when an ack is legitimate, and it said
    the 2026-10 arrangement in the present tense: "`backlog/data/promotions.jsonl` is a copy of
    the live promotions ledger, `tests/test_retention_sweep.py` pins figures measured FROM that
    copy, and no ordering lets both move together". #2054 moved every reader to the blob in the
    vault's history and #2064 deletes the file, so the paragraph is read AFTER the thing it
    describes in the present tense has gone — and a false mechanism note in the module that
    IMPLEMENTS the excuse is worse than no note, because it teaches the next round that
    re-committing a 33,113,707-byte mirror is what an ack is for. The price of re-timing it is
    gate rung 6's live rollback drill, since `scripts/automod/**` is in `PROTECTED_GLOBS`
    (`scripts/automod/spec.py`); the item rules that worth paying rather than leaving the lie.

    Pinned on the docstring of the IMPORTED function, not a re-read of the file: `agreement` is
    the probe that enforces the ack rule, so certifying a second copy of the text would let the
    shipped module say something different from what this node passes.

    Non-vacuity runs both ways. The paragraph must still NAME the path, or every absence assert
    below would be satisfied by deleting the example wholesale — which is the shape of green
    this loop has been burned by before. And the stale wording is checked against a control
    string that still carries it, so a marker that had silently stopped matching anything is
    caught here rather than passing forever.
    """
    import inspect

    flat = " ".join((inspect.getdoc(VG.agreement) or "").split())

    assert "backlog/data/promotions.jsonl" in flat, (
        "the agreement docstring no longer names the pinned-witness path at all, so each "
        "assert below would pass on a paragraph that says nothing about the excuse — the "
        "#2049 shape has to stay the worked example, in past tense")

    # The present-tense claims, exactly as #2054 left them, and a control that still says them.
    STALE = ("`backlog/data/promotions.jsonl` is a copy of the live promotions ledger, "
             "`tests/test_retention_sweep.py` pins figures measured FROM that copy, "
             "and no ordering lets both move together")
    STALE_IS = "is a copy of the live promotions"
    STALE_NOW = "no ordering lets both move together"
    assert STALE_IS in STALE and STALE_NOW in STALE, (
        "the control paragraph no longer carries the present-tense claims this node refuses, "
        f"so `{STALE_IS!r}` / `{STALE_NOW!r}` match nothing and the absence asserts below are "
        "green by construction")
    assert STALE_IS not in flat, (
        f"the agreement docstring still says the promotions mirror `{STALE_IS}` ledger. The "
        "working-tree copy is retired (#2054's readers, #2064's delete); a note claiming it is "
        "there invites a round to re-commit it, which is the cost clause 5 of #2054 exists to "
        "keep paid")
    assert STALE_NOW not in flat, (
        f"the docstring still claims `{STALE_NOW}` in the present tense. That deadlock was the "
        "#2047 arrangement; with the copy gone the ordering problem is historical, and prose "
        "that says otherwise is a reason to reach for an ack that is no longer needed")

    # The re-timing, positively: retired, attributed, and generalised over any pinned witness.
    assert "was then a rolling copy of the live promotions ledger" in flat, (
        "the paragraph no longer tells the #2047 story in the past tense, so a reader cannot "
        "tell what the excuse was written for — the history is the point of naming it")
    assert "RETIRED" in flat, (
        "the docstring does not say the copy is retired, which is the sentence a round "
        "considering an ack actually needs")
    assert "#2054" in flat and "#2064" in flat, (
        "the retirement is unattributed: the reader has no way to find the round that moved "
        "the readers and the round that deleted the file")
    assert ("any vault path whose pinned witness the same change is re-deriving" in flat
            and "nodes whose own test file names that file" in flat), (
        "the excuse is no longer stated as standing for ANY pinned witness whose own test file "
        "names the path. The mechanism text is what makes the rail reusable after this "
        "particular path is gone; narrowed to one dead path it becomes an artefact")


# --------------------------------------------------------------------------- #
#  #2044: the proposed-vault run goes out on pytest-xdist, and a failure that
#  only parallelism caused is re-asked serially before it can refuse anything.
#
#  Why: `PROBE_TIMEOUT_SECONDS` is 300 s and the production selection is 196
#  files / ~7,000 nodes, which #2044 puts at ~735 s run serially (its `[34%]`-at
#  250 s progress line — the figure `SERIAL_SELECTION_COST_S` carries and labels
#  there as a claim nobody re-ran, corroborated in order by the gate's own ~600 s
#  for a ~5,900-node suite at `gate.py:1792-1796`) — so every one of the
#  five real probes the promotion ledger ever recorded came back
#  `state=skipped ran=0 "timed out after 300s"`, and the #2036 guard has never
#  once judged a land. The gate stopped paying that cost on its own `tests` rung
#  at `gate.py:1842` (~600 s serial → ~76 s on 8 xdist workers). What that fix
#  also had to bring is the re-ask: eight workers make load, and a node that
#  loses under load is a fact about the box during the run, not about the prose
#  being landed (`gate.py:1798-1806` records the gate learning this the hard
#  way). These nodes pin both halves.
# --------------------------------------------------------------------------- #

#: A vault-reading guard that loses ONLY under a parallel run, asked two ways
#: because the first cut of this file was refused at review for asking it one.
#: `request.config.workerinput` is pytest-xdist's own documented "am I a worker"
#: handle and exists in no other run; `PYTEST_XDIST_WORKER` is what xdist writes
#: into each worker (`xdist/remote.py:417` at 3.8.0). The env var is only a
#: faithful signal once `_run_selection` strips the OUTER run's identity
#: (`PYTEST_CHILD_ENV_DROP`) — and it was not when this guard was first written:
#: the suite runs on 8 xdist workers, so the worker executing the probe leaked its
#: `gw3` into a child the probe had asked to run serially, the re-ask inherited the
#: flake it exists to clear, and the node below went red under `-n` for a reason
#: that had nothing to do with the code. Both signals, so the guard says what it
#: means whichever of the two moves, with the same shape as the gate's recorded
#: flaker: green serially, red on any worker.
GUARD_SRC_FLAKES = '''"""A vault-reading guard that loses only when the selection runs parallel."""
import os
from pathlib import Path


def test_a_guard_that_flakes_only_under_parallelism(request):
    v = Path(os.environ["LLOYD_VAULT_ROOT"])
    assert (v / "skills" / "foo" / "SKILL.md").exists()
    assert not hasattr(request.config, "workerinput"), (
        "lost in an xdist worker process, not under this land")
    assert not os.environ.get("PYTEST_XDIST_WORKER"), (
        "lost under a parallel run, not under this land")
'''

FLAKER_NODE = "tests/test_flaker.py::test_a_guard_that_flakes_only_under_parallelism"
COUNT_NODE = "tests/test_guard.py::test_the_prose_states_the_count_the_script_prints"


def _two_workers(monkeypatch, n: int = 2) -> list:
    """Pin the worker count the probe reads, without touching `config.yaml`.

    `config.yaml` is a path the loop may never write, and the clause is that the
    probe reads the count through the gate's own accessor — so the patch goes on
    the accessor, which is also the only way to hold a test to two workers instead
    of eight.

    The literal `("test_workers", 1)` is the pin, and it is inside the patched
    accessor rather than beside it because of the failure this guards: a probe that
    asked `_gate_cfg` for any other spelling — `test_worker_count`, or with a
    default of 8 — gets the REAL config's answer for a key nobody reads, which for
    a missing key is its default, which is 1, which serialises the selection and
    skips every land with the entire suite green. Handing `n` to whoever asked for
    whatever they asked for is precisely the lie that would let that pass. So this
    answers only the exact key-and-default #2044 names, and delegates anything else
    to the real accessor, where a wrong key lands on the box's real 8 and the
    caller's `workers == 2` assertion goes red at once. The returned list records
    every ask, which one node reads to name the spelling in its failure output.
    """
    import scripts.automod.gate as GATE

    asks: list[tuple] = []
    real = GATE._gate_cfg

    def accessor(key, default):
        asks.append((key, default))
        if key == "test_workers" and default == 1:
            return n
        return real(key, default)

    monkeypatch.setattr(GATE, "_gate_cfg", accessor)
    return asks


def _add_flaker(tree) -> None:
    """Add `GUARD_SRC_FLAKES` to a guard tree as its second vault-reading file."""
    (tree / "tests/test_flaker.py").write_text(GUARD_SRC_FLAKES)
    git(tree, "add", "-A")
    git(tree, "commit", "-q", "-m", "a guard that loses only under parallelism")


def _probe(tree, vault, tmp_path, *, paths=("skills/foo/SKILL.md",), **kw) -> dict:
    """One real `agreement` call, the same one `land()` makes, report in hand.

    `land()` hands the projection in `_guards_row` to the ledger and keeps only
    some keys; the clauses below are about the report itself, so they are read
    from the source rather than from what the row projection chose to carry.
    """
    return VG.agreement(paths=list(paths), live_root=tree, live_vault=vault,
                        python=Path(sys.executable),
                        scratch_parent=tmp_path / "probe-scratch", **kw)


def test_the_proposed_vault_run_gets_the_gates_worker_count_on_its_argv(monkeypatch, tmp_path):
    """Clause 1 and clause 3's env half, at the boundary the child is spawned over.

    `subprocess.Popen` is recorded, not written out of the picture, so what is
    asserted is the command line, the environment and the spawn flags the probe
    actually hands pytest — the ones `grep -n '"-n"'` in the item's check is a
    proxy for. The mark expression is asserted beside the flags because both live
    in the same list and a reordering that pushed the files ahead of `-m` would
    silently change the selection. `start_new_session` is asserted here for the
    same reason it exists: without it the group kill in `_reap_group` has no group
    of its own to kill, and it is a kwarg, so nothing on the argv would show it.
    """
    import scripts.automod.gate as GATE

    cmds: list[list[str]] = []
    envs: list[dict] = []
    flags: list[dict] = []

    class OnePassRun:
        """A controller that answers immediately with a one-node summary."""

        pid = 4243
        returncode = 0

        def communicate(self, timeout=None):
            return "collected 1 item\n1 passed in 0.1s\n", ""

    def record(cmd, **kw):
        cmds.append([str(c) for c in cmd])
        envs.append(dict(kw.get("env") or {}))
        flags.append(kw)
        return OnePassRun()

    monkeypatch.setattr(VG.subprocess, "Popen", record)
    r = VG._run_selection(Path(sys.executable), tmp_path, ["tests/test_guard.py"],
                          tmp_path / "vault", tmp_path / "data", None, 30.0, 3)

    pytest_cmds = [c for c in cmds if "pytest" in c]
    assert len(pytest_cmds) == 1, f"the probe launched {len(pytest_cmds)} children: {cmds}"
    argv = pytest_cmds[0]
    i = argv.index("-n")
    assert argv[i:i + 4] == ["-n", "3", "--dist", "loadfile"], argv
    marks = [i for i, t in enumerate(argv) if t == "-m"]
    assert len(marks) == 2 and argv[marks[1] + 1] == GATE.TESTS_MARK_EXPR, argv
    assert argv[-1] == "tests/test_guard.py", f"the file set moved: {argv}"
    assert argv.index("-n") < argv.index("tests/test_guard.py")
    assert r["ran"] == 1 and r["workers"] == 3, r
    assert flags[0].get("start_new_session") is True, (
        "the child has no process group of its own to kill")
    # The child inherits the process that launched the probe, which the suite runs
    # inside of, on 8 xdist workers. Its own identity has to go, or the serial
    # re-ask runs inside a leaked `gw3` and a node can be red for the outer run's
    # parallelism instead of its own — which is the bug that made this node fail
    # its first gate. EVERY name the drop list claims to strip is put into this
    # process's environment first and proven present right after the call: a name
    # the outer run never held comes back absent from the child's env whatever the
    # code does, so injecting two of four and asserting the absence of four pins two
    # of four and leaves the other two assertions vacuous.
    injected = {"PYTEST_XDIST_WORKER": "gw7",
                "PYTEST_XDIST_WORKER_COUNT": "8",
                "PYTEST_XDIST_TESTRUNUID": "c0ffee00cafe1234",
                "PYTEST_CURRENT_TEST": "outer::test"}
    for _name, _value in injected.items():
        monkeypatch.setenv(_name, _value)
    cmds.clear()
    envs.clear()
    flags.clear()
    VG._run_selection(Path(sys.executable), tmp_path, ["tests/test_guard.py"],
                      tmp_path / "vault", tmp_path / "data", None, 30.0, 3)
    assert set(injected) == set(VG.PYTEST_CHILD_ENV_DROP), (
        f"the injected set {sorted(injected)} no longer covers the drop list "
        f"{sorted(VG.PYTEST_CHILD_ENV_DROP)}, so the loop below is asserting the "
        f"absence of a name nothing ever put there")
    for _name, _value in injected.items():
        assert os.environ.get(_name) == _value, (
            f"{_name} was never in the environment the child inherits, so its "
            f"absence below would pass on an absent input")
    assert len(envs) == 1, f"the probe launched {len(envs)} children: {cmds}"
    child_env = envs[0]
    for name in VG.PYTEST_CHILD_ENV_DROP:
        assert name not in child_env, f"the outer run's {name} reached the child"
    for name, want in (("LLOYD_VAULT_ROOT", str(tmp_path / "vault")),
                       ("LLOYD_DATA", str(tmp_path / "data")),
                       (VG.NESTING_ENV, "1")):
        assert child_env[name] == want, child_env
    # The other half of clause 3 at the same boundary: `workers=1` is a serial
    # run, and a serial run never grew these two tokens. `-n 1` is not how you
    # ask for one, because passing `-n` at all hands the run to xdist.
    cmds.clear()
    envs.clear()
    flags.clear()
    VG._run_selection(Path(sys.executable), tmp_path, ["tests/test_guard.py"],
                      tmp_path / "vault", tmp_path / "data", None, 30.0)
    assert "-n" not in cmds[0] and "--dist" not in cmds[0], cmds[0]


def test_the_two_runs_choose_their_parallelism_independently_over_real_children(
        vault, guard_tree, probed, tmp_path, monkeypatch):
    """Clauses 1 and 3, with real children: the big run parallel, the re-ask serial.

    Not a recorded argv — an actual `python -m pytest` that either accepts
    `-n 2 --dist loadfile` and reports a denominator or the node fails. What each
    run's `workers` field is is the count `_run_selection` ECHOES back from its own
    parameter, so it witnesses the count the caller asked for and the fact that a
    real child ran (`ran`) beside it — it is not by itself proof that the `-n`
    tokens reached the command line. Those two things live in the nodes beside this
    one, which is where a reviewer of this diff looked for them and was right to:
    `test_the_proposed_vault_run_gets_the_gates_worker_count_on_its_argv` pins the
    tokens on the argv (and goes red when they are removed), and
    `test_a_flaker_cannot_refuse_and_a_real_failure_still_does` is the behavioural
    witness that the proposed run really ran in parallel — its guard loses ONLY
    under an xdist scheduler, and it is red in `candidate` here and green in a
    serial run. What this node adds over both is that the SELECTION, the mark object
    and the real children all still work together end to end at `-n 2`, and that the
    pre-land run stays serial where a parallel base would corrupt the delta.

    The tree is red at both ends (prose says six stores, the script prints four,
    before and after the reword), which is the case that reaches the pre-land run:
    that run must be serial, because a parallel base could cancel a parallel
    proposed run's flake and take a real refusal down with it.
    """
    asks = _two_workers(monkeypatch)
    _prose(vault, "---\nname: foo\n---\nstores: 6\n", commit=True)   # red at vault HEAD
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 6\n\n#2044 reworded on a red tree.\n")
    rep = _probe(guard_tree, vault, tmp_path)

    # The spelling, named rather than inferred from a number that happened to match:
    # `_two_workers` answers ONLY this key-and-default pair, so a probe reading any
    # other one would have got the box's real 8 and failed below.
    assert ("test_workers", 1) in asks, (
        f"the probe never asked automod.gate._gate_cfg('test_workers', 1); it asked "
        f"{sorted(set(asks))}")
    assert rep["workers"] == 2, rep
    assert rep["candidate"]["workers"] == 2 and rep["candidate"]["ran"] == 1, rep["candidate"]
    assert rep["candidate"]["failed"] == [COUNT_NODE], rep["candidate"]
    # The serial re-ask ran, serially, over the proposed vault.
    assert rep["parallel_retry"]["workers"] == 1, rep["parallel_retry"]
    assert rep["parallel_retry"]["ran"] == 1, rep["parallel_retry"]
    assert rep["parallel_retry"]["files"] == 1, rep["parallel_retry"]
    # And the pre-land run over the failing files, also serially.
    assert rep["baseline"]["workers"] == 1, rep["baseline"]
    assert rep["baseline"]["failed"] == [COUNT_NODE], rep["baseline"]
    assert "parallel_only_failures" not in rep, rep
    assert rep["state"] == "checked" and rep["refuse"] is False, rep
    assert "pre-existing" in rep["reason"], rep["reason"]
    # #2042's arithmetic is untouched: the whole probe still costs at least the
    # runs it launched, and the queue for the gate's slot is still reported.
    assert rep["seconds"] >= (rep["candidate"]["seconds"]
                              + rep["parallel_retry"]["seconds"]
                              + rep["baseline"]["seconds"]), rep


def test_a_flaker_cannot_refuse_and_a_real_failure_still_does(
        vault, guard_tree, probed, tmp_path, monkeypatch):
    """Clauses 4 and 5 in one selection, with both kinds of failure in it.

    Two guards, one tree. The count guard is red against the vault as proposed and
    green against the vault before the land — prose said five, the tree prints four
    — which is the real case this check exists for. The flaker is red only under
    xdist, which is the load case, and the parallel run cannot tell them apart: its
    failure set names both. The serial re-ask can. So the refusal must name exactly
    the count guard, the flaker must be named as a parallel-only failure the way
    `gate.py:1861` names them, and the land must still be refused — clause 5 is
    that none of this softened the answer for a failure that is real.
    """
    _two_workers(monkeypatch)
    _add_flaker(guard_tree)
    assert len(VG.guard_selection(guard_tree)) == 2, "the flaker is not in the selection"
    _prose(vault, SKILL_AT_FOUR, commit=True)     # vault HEAD agrees with the tree
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n")
    rep = _probe(guard_tree, vault, tmp_path)

    cand = rep["candidate"]
    assert cand["workers"] == 2 and cand["ran"] == 2, cand
    assert sorted(cand["failed"]) == sorted([COUNT_NODE, FLAKER_NODE]), cand
    assert rep["parallel_retry"]["workers"] == 1, rep["parallel_retry"]
    assert rep["parallel_retry"]["failed"] == [COUNT_NODE], rep["parallel_retry"]
    assert rep["parallel_only_failures"] == [FLAKER_NODE], rep
    assert rep["state"] == "checked" and rep["refuse"] is True, rep
    assert rep["nodes"] == [COUNT_NODE], rep["nodes"]
    # Read the other way, the same report says the re-ask is what separated them:
    # the flaker is nowhere in `nodes`, so it bought no refusal, and `excerpt` —
    # what `refusal_text` prints as the guard output — is the serial run's tail.
    assert FLAKER_NODE not in str(rep["excerpt"]), rep["excerpt"][-400:]
    assert rep["baseline"] == {"ran": 1, "failed": [], "note": rep["baseline"]["note"],
                              "seconds": rep["baseline"]["seconds"], "files": 1,
                              "workers": 1}, rep["baseline"]
    # The seam to the route itself: a real failure still blocks the commit, with
    # the flaker nowhere in the refusal a human reads.
    head = _head(vault)
    with pytest.raises(V.VaultRoundError) as ei:
        V.land(["skills/foo/SKILL.md"], "#2044 a flaker and a real one", item_id=None)
    msg = str(ei.value)
    assert COUNT_NODE in msg and "prose 5, script 4" in msg, msg
    assert FLAKER_NODE not in msg, f"a load flaker reached the refusal: {msg}"
    assert _head(vault) == head, "refused before any vault commit"


@pytest.mark.parametrize("mode", ["too-many-files", "names-no-file"], ids=["41-files", "no-file"])
def test_an_unattributable_parallel_failure_set_never_buys_a_serial_whole_selection_run(
        vault, guard_tree, probed, tmp_path, monkeypatch, mode):
    """Clause 4's second half: past the ceiling, the answer is the non-answer.

    `gate.py:1851` answers "more than `PARALLEL_RETRY_MAX_FILES` files failed" by
    re-running its whole suite serially, and for this probe that IS the ~735 s run
    the budget cannot hold — so copying the fallback would trade an honest
    `skipped` for a timeout that reports nothing, which is the exact state of all
    five real ledger rows. The child runner is stood in for with a return dict
    shaped exactly like `_run_selection`'s own, because the condition needs a
    failure set naming 41 files (or none), and the thing under test is what
    `agreement` does with it: the assertion that carries the clause is that the
    child runner was called ONCE, with the parallel count, and never again with
    the whole selection at 1 worker.
    """
    calls: list[dict] = []
    failed = ([f"tests/test_synth{i}.py::test_x" for i in range(41)]
              if mode == "too-many-files" else ["::a_failure_that_names_no_file"])

    def fake_run(python, tree, files, vroot, data, mark, budget, workers=1):
        calls.append({"files": list(files), "workers": workers, "budget": budget})
        return {"ran": 41, "failed": list(failed), "seconds": 1.0,
                "files": len(files), "workers": workers, "note": "canned run",
                "excerpt": ""}

    _two_workers(monkeypatch)
    monkeypatch.setattr(VG, "_run_selection", fake_run)
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 4\n\n#2044 an unattributable set.\n")
    rep = _probe(guard_tree, vault, tmp_path)

    assert rep["state"] == "skipped" and rep["refuse"] is False, rep
    assert rep["nodes"] == [], rep
    assert rep["parallel_failures"] == failed[:50], rep
    # Literals, not the same expressions that build the string: an assertion that
    # re-evaluates `_parallel_retry_max_files()` or `SERIAL_SELECTION_COST_S` cannot
    # fail however the reason is worded, so it would pin nothing about clause 4's
    # ceiling or about the cost the non-answer has to name.
    assert "40-file ceiling" in rep["reason"], rep["reason"]
    assert "~735s" in rep["reason"], rep["reason"]
    assert len(calls) == 1, f"a serial re-run happened after the ceiling: {calls}"
    assert calls[0]["workers"] == 2, calls[0]
    assert calls[0]["files"] == VG.guard_selection(guard_tree), calls[0]
    # And the same shape out through the route: an unattributable probe is a
    # landed land with a `skipped` row, not a refused one.
    out = V.land(["skills/foo/SKILL.md"], "#2044 an unattributable set", item_id=None)
    assert out["commit"], out
    row = out["guards"]
    assert row["state"] == "skipped" and row["refuse"] is False, row


def test_a_serial_re_ask_that_answers_buys_neither_a_refusal_nor_a_second_run(
        vault, guard_tree, probed, tmp_path, monkeypatch):
    """Clause 4's other unanswered branch: the re-ask itself says nothing.

    The node above covers a failure set that cannot BE re-asked (41 files, or no
    file at all). This is the branch where the re-ask launches and answers nothing:
    the parallel run reports a failing node, the serial re-ask of its file times
    out. Both easy ways out are wrong. Refusing would blame the land for a node
    nothing has confirmed red at the proposed vault; falling through to the pre-land
    run would report a `checked` whose delta was computed against a failure that may
    have been load. So the answer is `skipped`, and the witness that this branch is
    not the one above is the number and shape of the child runs: two — the parallel
    proposed run and the serial re-ask — and no third over the pre-land mirror.
    """
    calls: list[dict] = []

    def fake_run(python, tree, files, vroot, data, mark, budget, workers=1):
        calls.append({"files": list(files), "workers": workers})
        if workers > 1:
            return {"ran": 3, "failed": [COUNT_NODE], "seconds": 1.0,
                    "files": len(files), "workers": workers,
                    "note": "canned parallel run", "excerpt": ""}
        return {"ran": 0, "failed": [], "seconds": 1.0, "files": len(files),
                "workers": 1, "note": "canned: timed out after 1.0s", "excerpt": ""}

    _two_workers(monkeypatch)
    monkeypatch.setattr(VG, "_run_selection", fake_run)
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 4\n\n#2044 the re-ask answers nothing.\n")
    rep = _probe(guard_tree, vault, tmp_path)

    assert rep["state"] == "skipped" and rep["refuse"] is False, rep
    assert rep["nodes"] == [], rep
    assert "parallel_only_failures" not in rep, rep
    assert rep["parallel_retry"] == {"ran": 0, "failed": [], "seconds": 1.0,
                                     "files": 1, "workers": 1,
                                     "note": "canned: timed out after 1.0s"}, \
        rep["parallel_retry"]
    assert "re-ask" in rep["reason"] and "answered nothing" in rep["reason"], rep["reason"]
    assert rep["candidate"]["failed"] == [COUNT_NODE], rep["candidate"]
    assert [c["workers"] for c in calls] == [2, 1], (
        f"expected the parallel proposed run and the serial re-ask and nothing else: "
        f"{calls}")
    assert "baseline" not in rep, (
        "a pre-land run was launched off a failure nothing confirmed")


#: Every live process whose working directory is `tree`, however it was started.
#: Read out of `/proc` and not out of anything Python is tracking, because the whole
#: question is which processes the probe LOST track of. A zombie has no `cmdline`,
#: so a reaped corpse is not counted as a survivor.
def _procs_of(tree: Path) -> list[int]:
    root, procs = str(tree), []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.readlink(f"/proc/{entry.name}/cwd") != root:
                continue
            cmd = Path(f"/proc/{entry.name}/cmdline").read_bytes() \
                .replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "pytest" in cmd or "exec(" in cmd:
            procs.append(int(entry.name))
    return procs


def test_a_killed_parallel_run_leaves_no_worker_of_its_scratch_tree_alive(
        tmp_path):
    """The boundary #2044's own `-n` opens, tested across it with real processes.

    A serial run is one process and a timeout kills it. Adding the worker count
    turns a killed run into a controller plus `workers` workers, and killing only
    the process the probe started abandons the rest — which is not hypothetical for
    this probe: measured on this box before the fix, a `pytest -n 4` against a guard
    that sleeps, killed by the timeout at 10 s, left a worker alive THIRTY seconds
    later with its cwd still inside the tree, while the same command launched in its
    own session and killed by process group left nothing alive one second later.
    That survivor is a guard still reading a vault mirror `agreement` deletes one
    `finally` later, on cores the next gate `tests` rung needs.

    Nothing here is faked: a real hanging guard, a real `-n 2`, a real timeout, and
    the assertion is the absence of processes afterwards. The positive control is
    the same scan taken DURING the run — without it an empty result would only show
    that nothing ever started.
    """
    import threading

    tree = make_guard_tree(tmp_path / "killed-parallel", src=GUARD_SRC_HANGS)
    files = VG.guard_selection(tree)
    assert files, "the hanging guard is not in the selection this probe would run"

    seen: list[int] = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            seen.append(len(_procs_of(tree)))
            time.sleep(0.25)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        r = VG._run_selection(Path(sys.executable), tree, files, tmp_path / "mirror",
                              tmp_path / "data", None, 12.0, 2)
    finally:
        stop.set()
        watcher.join(timeout=5)

    assert r["ran"] == 0 and r["workers"] == 2, r
    assert "timed out" in r["note"], r["note"]
    assert max(seen) >= 2, (
        f"the run was never parallel, so an empty survivor list would prove "
        f"nothing; the scan saw {seen}")
    survivors = _procs_of(tree)
    for pid in survivors:          # a red node must not leave the box dirty either
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    assert survivors == [], (
        f"{len(survivors)} process(es) outlived the run the probe gave up on: "
        f"{survivors}, still rooted at {tree}")


@pytest.mark.parametrize("mode", ["workers-1", "no-xdist"], ids=["workers-1", "no-xdist"])
def test_a_probe_that_cannot_get_pytest_xdist_says_so_rather_than_running_serially(
        vault, guard_tree, probed, tmp_path, monkeypatch, mode):
    """Clause 2: no xdist is a named non-answer, never a 735 s serial timeout.

    Both ways the gate's `_test_workers` reaches serial: the config says one
    worker, or `import xdist` fails in the interpreter that would launch the child
    (`gate.py:1781`'s probe, and the reason the gate's candidate venv still
    gates). Here the degraded answer is worse than the gate's, because a serial
    probe of a 196-file selection does not degrade, it times out — so the clause
    is that nothing is run at all, and `calls` being empty is the witness of
    "nothing", while the `skipped` state is what keeps the outage from reading as
    agreement. The land still goes through: #2036's rule is that a probe outage
    never blocks, and it never passes either.
    """
    import subprocess as SP

    calls: list[list] = []
    monkeypatch.setattr(VG, "_run_selection",
                        lambda *a, **kw: (calls.append(list(map(str, a))), {})[1])
    _two_workers(monkeypatch, 1 if mode == "workers-1" else 8)
    if mode == "no-xdist":
        real = subprocess.run

        def no_xdist(cmd, **kw):
            if str(cmd[-1]) == "import xdist":
                return SP.CompletedProcess(cmd, 1, "",
                                           "ModuleNotFoundError: No module named 'xdist'")
            return real(cmd, **kw)

        monkeypatch.setattr(VG.subprocess, "run", no_xdist)

    _prose(vault, SKILL_AT_FOUR, commit=True)
    probed(guard_tree, vault)
    _prose(vault, "---\nname: foo\n---\nstores: 4\n\n#2044 xdist is not available.\n")
    rep = _probe(guard_tree, vault, tmp_path)
    assert rep["state"] == "skipped" and rep["refuse"] is False, rep
    assert "pytest-xdist" in rep["reason"], rep["reason"]
    assert "~735s" in rep["reason"], rep["reason"]
    assert rep["workers"] == 1, rep
    assert calls == [], f"a child run was launched without xdist: {calls}"
    # Absence, not falsiness: a pre-seeded `{"ran": 0, "failed": []}` would satisfy
    # `not rep["candidate"]` while telling the reader a run reported zero, which is
    # the placeholder-as-denominator shape #1691 is about. `_guards_row` already
    # writes the key only `if cand:`; the report now agrees with the ledger.
    assert "candidate" not in rep, rep
    assert "baseline" not in rep, rep
    assert "parallel_retry" not in rep, rep

    out = V.land(["skills/foo/SKILL.md"], "#2044 the probe cannot get xdist", item_id=None)
    assert out["commit"], "a probe outage must not stop the land"
    row = _events("vault_land")[0]["guards"]
    assert row["state"] == "skipped" and row["refuse"] is False, row
    assert "pytest-xdist" in row["reason"] and "735" in row["reason"], row
    assert "candidate" not in row, row
    # #2046 clause 1: a serial probe STATES its count instead of dropping the key.
    # Omitting it would make this row read as a probe that never reached its worker
    # decision — `vault_guards.py:708-714`'s unreadable HEAD or empty selection —
    # which is a different non-answer from the one this reason sentence gives.
    assert row["workers"] == 1, row
    assert "parallel_retry" not in row, row


# --------------------------------------------------------------------------- #
#  #2046: the xdist fields the parallel probe reports reach the LEDGER ROW.
#
#  Why: #2044 put `workers`, `parallel_retry` and `parallel_only_failures` on the
#  guard REPORT (`vault_guards.py:717`, `:831`, `:844`), and every node above reads
#  that report — while `_guards_row`, the one projection between it and
#  `promotions.jsonl`, dropped all three. Promotion-ledger row 29142 (`ts`
#  2026-10-02T06:33:44Z, commit 88bab697) is the witness: `candidate.failed=1`,
#  `refuse=false`, and the sentence "once the 1 that failed only under parallelism
#  were re-asked serially and pass" ONLY in `reason`. A reader holding the
#  structured keys could not tell that row's parallel probe from a serial one, nor
#  a dismissed flaker from a real refusal. These nodes read the row `land()` writes
#  (`vault_round.py:866`, `:913`), which is the half no #2044 node covered.
# --------------------------------------------------------------------------- #

def _flaky_report(**over) -> dict:
    """A guard report shaped exactly as `agreement` builds one, for projection only.

    Four of these numbers are ledger row 29142's own as committed — 7,070 nodes over
    196 files, 155.2 s of proposed-vault run, 1 failed node — and the row carries NO
    worker count at all, which is the defect itself: its committed `guards` keys are
    candidate/excerpt/reason/refuse/seconds/state. So the `workers: 8` this fixture
    supplies is not read off that row; it is what the probe takes from
    `automod.gate.test_workers` (`config.yaml:1346`) on the box that wrote it — the
    value the row should have carried and did not. Only the keys `_guards_row` reads
    are here; where a field is produced is pinned where it is produced, in the nodes
    above.
    """
    base = {"state": "checked", "refuse": False, "seconds": 157.8, "workers": 8,
            "candidate": {"ran": 7070, "failed": [], "seconds": 155.2, "files": 196,
                          "workers": 8},
            "reason": "7070 vault-reading nodes pass against the vault as proposed",
            "excerpt": ""}
    base.update(over)
    return base


def _no_reason(row: dict) -> dict:
    """The row minus its prose — the comparison clause 4 asks for, enforced."""
    return {k: v for k, v in row.items() if k != "reason"}


def test_a_parallel_probe_states_its_workers_and_its_re_ask_on_the_row_it_lands_with(
        vault, monkeypatch):
    """Clauses 1, 2 and 3 across the one seam this change owns.

    The boundary is report → projection → JSONL: `land()` takes whatever
    `VG.agreement` returns and projects it once (`vault_round.py:866` on the refusal
    path) into the file a later reader has. So the guard is replaced with a report
    that has the shape `agreement` builds when it launches on workers and re-asks
    serially — the same fields #2044's nodes pin AT THE PRODUCER, with ledger row
    29142's own numbers — and what is asserted below is on
    `_events("vault_land")[0]["guards"]`, never on the report.

    No probe child is launched here on purpose. The probe lock
    (`vault_guards.py:722`) is a box-wide file, so a node that ran real children in
    parallel with `test_the_probe_waits_for_the_gate_tests_slot_before_it_starts_a_child`
    made that node measure a 2 s queue and lose its `lock_wait_s`-absent assertion;
    the producer's real children are covered by the nodes above, and this one covers
    the projection without borrowing the box's lock.
    """
    report = _flaky_report(
        refuse=True, nodes=[COUNT_NODE],
        candidate={"ran": 7070, "failed": [FLAKER_NODE, COUNT_NODE],
                   "seconds": 155.2, "files": 196, "workers": 8},
        parallel_retry={"ran": 2, "failed": [COUNT_NODE], "seconds": 4.4, "files": 2,
                        "workers": 1, "note": "1 failed in 4.4s"},
        parallel_only_failures=[FLAKER_NODE],
        reason=("7070 vault-reading nodes, 2 failed under parallelism; 1 of them "
                "passed serially, 1 is real"),
        excerpt="FAILED tests/test_guard.py::test_the_prose_states_the_count")
    monkeypatch.setattr(VG, "agreement", lambda *a, **k: dict(report))
    # The path has to exist in the vault's working tree, as every other node here
    # leaves it: validation runs before the guard, so a missing path would refuse
    # for the wrong reason and no guards row would be written at all.
    _prose(vault, SKILL_AT_FOUR, commit=True)
    _prose(vault, "---\nname: foo\n---\nstores: 5\n")
    head = _head(vault)
    with pytest.raises(V.VaultRoundError):
        V.land(["skills/foo/SKILL.md"], "#2046 the flaker is on the row", item_id=None)
    assert _head(vault) == head, "refused before any vault commit"
    g = _events("vault_land")[0]["guards"]

    # Clause 1: the count the proposed-vault run was launched with, as data.
    assert g["workers"] == 8, g
    # Clause 2: the serial re-ask as its own block, `failed` a node COUNT like the
    # two run blocks beside it, `workers` 1 because the re-ask is serial by
    # construction (`vault_guards.py:829`) — which is the whole reason it exists.
    assert g["parallel_retry"] == {"ran": 2, "failed": 1, "workers": 1}, g
    # Clause 3: the node parallelism alone caused, named rather than implied.
    assert g["parallel_only_failures"] == [FLAKER_NODE], g
    assert "parallel_only_failures_count" not in g, "one node is under the 10 cap"
    # The decision the row already recorded is untouched by the projection: the
    # flaker bought no refusal and the real disagreement still did.
    assert {k: g[k] for k in ("state", "refuse", "nodes")} == {
        "state": "checked", "refuse": True, "nodes": [COUNT_NODE]}, g
    assert FLAKER_NODE not in str(g.get("excerpt", "")), g
    assert g["candidate"] == {"ran": 7070, "failed": 2, "seconds": 155.2,
                             "files": 196}, g


def test_a_row_from_a_probe_that_re_asked_is_not_the_row_from_one_that_never_re_asked():
    """Clause 4: structurally, with `reason` out of the comparison on both sides.

    The defect is that row 29142's dismissal lives in its prose and nowhere else, so
    the two rows below are given the SAME `reason` sentence: if the only place a
    re-ask could ever be recorded were that string, stripping it would make the rows
    equal and this node goes red. It is the strictest form of the claim, and the
    `set(...) - set(...)` line names which keys carry the difference rather than
    leaving it to a bare `!=`.
    """
    cand = {"ran": 7070, "failed": [FLAKER_NODE], "seconds": 155.2, "files": 196,
            "workers": 8}
    shared_reason = ("7070 vault-reading nodes pass against the vault as proposed, "
                     "once the 1 that failed only under parallelism were re-asked "
                     "serially and pass")
    reasked = V._guards_row(_flaky_report(
        candidate=cand, parallel_only_failures=[FLAKER_NODE],
        parallel_retry={"ran": 2, "failed": [], "seconds": 3.1, "files": 1,
                        "workers": 1, "note": "2 passed in 3.1s"},
        reason=shared_reason))
    never = V._guards_row(_flaky_report(candidate=cand, reason=shared_reason))

    assert _no_reason(reasked) != _no_reason(never), (
        "the two rows are structurally identical once the prose is removed, which is "
        "exactly ledger row 29142's shape")
    assert set(reasked) - set(never) == {"parallel_retry",
                                        "parallel_only_failures"}, (
        sorted(reasked), sorted(never))
    assert "parallel_retry" not in never, (
        "a probe that never re-asked must omit the key, never carry it present-empty")
    assert _no_reason(never)["workers"] == 8, "the count both rows share stays put"


def test_a_dismissed_flaker_list_is_capped_where_the_nodes_list_is_already_capped():
    """Clause 3's cap, both sides of it: 10 named plus a total, or 3 and no total.

    The producer puts an unbounded list on the report — `flinched` is a subset of the
    parallel run's `failed` (`vault_guards.py:840-844`) and nothing bounds it — and
    the ledger's own precedent for a node list is the `[:10]` on `nodes`. The count
    appears only past the cap, because a count equal to the list's own length says
    nothing.
    """
    many = [f"tests/test_f{ i }.py::test_node" for i in range(13)]
    wide = V._guards_row(_flaky_report(
        candidate={"ran": 7070, "failed": many, "seconds": 155.2, "files": 196,
                   "workers": 8},
        parallel_retry={"ran": 13, "failed": [], "seconds": 9.9, "files": 13,
                        "workers": 1, "note": "13 passed"},
        parallel_only_failures=many))
    assert wide["parallel_only_failures"] == many[:10], wide["parallel_only_failures"]
    assert wide["parallel_only_failures_count"] == 13, wide

    few = V._guards_row(_flaky_report(parallel_only_failures=many[:3]))
    assert few["parallel_only_failures"] == many[:3], few
    assert "parallel_only_failures_count" not in few, few


def test_a_report_that_reported_nothing_adds_no_key_to_the_row():
    """Clauses 1-3's absence halves, on the rows the probe never fills in.

    `agreement` returns before it decides a worker count when it cannot read the
    tree's HEAD or the selection is empty (`vault_guards.py:708-714`), and it reaches
    a re-ask only when a parallel run failed something — so all three keys can be
    legitimately absent at once, and a projection that wrote `workers: 0`,
    `parallel_retry: {}` or `parallel_only_failures: []` would turn "nothing
    reported" into "reported as nothing", the #1691 shape this function's own
    docstring names.
    """
    row = V._guards_row({"state": "skipped", "refuse": False, "seconds": 0.4,
                         "reason": "no test file under tests/ names a vault root"})
    assert set(row) == {"state", "refuse", "seconds", "reason"}, row
    for key in ("workers", "parallel_retry", "parallel_only_failures",
                "parallel_only_failures_count"):
        assert key not in row, f"{key} was written for a probe that never reported it"
    assert V._guards_row(_flaky_report(workers=1))["workers"] == 1, (
        "a serial probe states 1; it does not drop the key")
    empty_retry = V._guards_row(_flaky_report(parallel_retry={}))
    assert "parallel_retry" not in empty_retry, empty_retry
    no_flakers = V._guards_row(_flaky_report(
        parallel_retry={"ran": 0, "failed": [], "seconds": 1.0, "files": 1,
                        "workers": 1, "note": "timed out after 1.0s"}))
    assert no_flakers["parallel_retry"] == {"ran": 0, "failed": 0, "workers": 1}, (
        "the re-ask that answered nothing is still a re-ask, and it explains itself "
        "in `reason` — `parallel_only_failures` is absent beside it by design")
    assert "parallel_only_failures" not in no_flakers, no_flakers


#: #2046's witness, verbatim: ledger row 29142 of
#: `~/.local/state/lloyd-automod/promotions.jsonl`, re-read 2026-10-02. Inline rather
#: than in a fixture file because clause 5 allows exactly two paths in `git diff --stat`
#: and a third file there would break it, whatever its purpose — so the bytes are
#: committed in this file, and a byte-identical copy sits on the vault's main at
#: WITNESS_2046_VAULT_PATH below, which is not this diff.
WITNESS_2046_ROW = (
    '{"ts": 1790922824.546555, "created_at": "2026-10-02T06:33:44Z", "event": "vault_land", "ok": tru'
    'e, "item_id": null, "commit": "88bab69793af05d8670f8914c26b44170698573f", "paths": ["skills/kg-m'
    'ention-classifier/SKILL.md"], "validated": ["skills/kg-mention-classifier/SKILL.md"], "guards": '
    '{"state": "checked", "refuse": false, "seconds": 157.8, "candidate": {"ran": 7070, "failed": 1, '
    '"seconds": 155.2, "files": 196}, "reason": "7070 vault-reading nodes pass against the vault as p'
    'roposed, once the 1 that failed only under parallelism were re-asked serially and pass", "excerp'
    't": "..............                                                           [100%]\\n14 passed '
    'in 0.82s\\n"}, "review": "skipped", "review_reason": "no item bound: the second reader has no con'
    'tract to grade, so it was not consulted (module CLI, autoresearch promote)", "review_clauses": ['
    '], "landing_clauses": [], "session_id": "20261001_230425_autonomy_e695", "skill_gate": [{"skill"'
    ': "kg-mention-classifier", "has_eval": false, "would_refuse": false, "reason": "no activation ev'
    'al for this skill"}], "message": "kg-mention-classifier: record where mid-run backlog numbers mi'
    'slead (task #74 run 2026-10-02)\\n\\nThree traps a run hits and can only be warned about from this'
    ' file:\\n- the runner\'s queue line agrees with"}'
)

#: The durable copy, landed through `automod_vault_land`. Named as a repo-relative
#: vault path so the citation resolver finds it, the way #2042's witness does.
WITNESS_2046_VAULT_PATH = "backlog/data/2026-10-02.2046-parallel-flaker-witness.jsonl"
LIVE_PROMOTIONS_LEDGER = Path("~/.local/state/lloyd-automod/promotions.jsonl").expanduser()
WITNESS_2046_ROW_NUMBER = 29142


def test_the_witness_row_the_item_quotes_carries_no_worker_count_at_all():
    """Clause 6: the bytes the defect is quoted from, re-derived and pinned.

    Every figure in the item's own premise is recomputed here from these bytes: the
    row is a `vault_land` at 2026-10-02T06:33:44Z against commit 88bab697, its
    `guards` say `checked`/`refuse: false` over 7,070 nodes on 196 files in 155.2 s
    with ONE failure, and the dismissal of that failure is ONLY the prose sentence
    naming a serial re-ask. The three keys this change projects are absent, which is
    the absence the round exists to end — so this node goes red the moment anyone
    rewrites the witness to look like the fixed shape, the way #2042's node refuses
    a witness that grew the fields it is filing the absence of.

    Against the live ledger the row is looked up by its own `ts`, not by its line
    number (29142): that file gains a row with every promotion and the groundskeeper
    folds it, which is the #1193 failure this file's own witness nodes warn about. A
    row the live file still holds must equal these bytes byte for byte; one it has
    archived away is not a disagreement. With no evidence root present the durable leg
    is skipped, as #2042's node does — the inline bytes and the live leg always run.
    """
    import hashlib
    import json

    import scripts.automod.review as RV

    row = json.loads(WITNESS_2046_ROW)
    assert row["event"] == "vault_land" and row["ok"] is True
    assert row["created_at"] == "2026-10-02T06:33:44Z", row["created_at"]
    assert str(row["commit"]).startswith("88bab697"), str(row["commit"])[:8]
    g = row["guards"]
    assert g["state"] == "checked" and g["refuse"] is False, sorted(g)
    assert g["seconds"] == 157.8, g["seconds"]
    assert g["candidate"] == {"ran": 7070, "failed": 1, "seconds": 155.2,
                               "files": 196}, g["candidate"]
    for key in ("workers", "parallel_retry", "parallel_only_failures"):
        assert key not in g, (
            f"{key} is on the witness now, so this row is no longer the non-answer "
            f"#2046 cites: re-cut it from a pre-landing row before rerunning the item")
    assert "re-asked serially and pass" in g["reason"], g["reason"]
    assert "baseline" not in g, "a pruned baseline is a different land"

    if LIVE_PROMOTIONS_LEDGER.is_file():
        # Looked up by the row's own `ts`, never by line number: #1193 is on the
        # board precisely because a whole-file live-ledger figure pinned in a test
        # reddens every later promotion, and this file grows by a row per land and
        # is folded by the groundskeeper. A row the ledger still holds must match
        # these bytes exactly; one it has archived away says so without lying.
        lines = LIVE_PROMOTIONS_LEDGER.read_text(errors="replace").splitlines()
        live = [l for l in lines if f'"ts": {row["ts"]}' in l]
        assert len(live) <= 1, f"the live ledger holds {len(live)} copies of one ts"
        if live:
            assert live[0] == WITNESS_2046_ROW, (
                "row 29142 of the live ledger is not the bytes pinned here — the "
                "witness was rewritten, not merely aged")

    for root in RV.REVIEW_EVIDENCE_ROOTS:
        durable = root / WITNESS_2046_VAULT_PATH
        if not root.is_dir():
            continue
        assert durable.is_file(), f"the durable copy is not on the vault's main: {durable}"
        assert hashlib.sha256(durable.read_bytes()).hexdigest() == \
            hashlib.sha256((WITNESS_2046_ROW + "\n").encode()).hexdigest(), \
            "the vault's durable copy and these committed bytes diverged"




# ── #2049: the probe's ONE acknowledged excuse, and its ledger row ────────────
#
# The shape is #2047's: `backlog/data/promotions.jsonl` is a prefix-extract of the live
# promotions ledger and `tests/test_retention_sweep.py` pins figures measured FROM that copy,
# so no ordering lets both move together. The probe always judges HEAD's code (it takes no
# caller-supplied tree), so bytes-first shows HEAD's pins disagreeing with the new bytes; and
# the gate's `tests` rung runs a candidate against the real vault, so pins-first shows the
# candidate disagreeing with the bytes still on disk. The escape is one ack naming the landed
# path, scoped to nodes whose own file names that path, with every excused id on the row.
#
# Every node here drives the real `agreement`; only `_run_selection` is stood in for pytest —
# the same stand-in #2042 and #2044 use — and it decides pass/fail from the VAULT BYTES it is
# handed, which is the only way to reproduce "passed with these paths put back, failed with
# them in place" without a 32 MB fixture and a 7,000-node run. Not stood in: the worktree off
# HEAD, the vault copy, `baseline_vault` putting the path back, the ack split, the node-id
# scoping and the report.

MIRROR_PATH = "backlog/data/promotions.jsonl"
PINNED_NODE = "tests/test_guard.py::test_the_mirror_row_count_the_item_quotes"
UNRELATED_NODE = "tests/test_unrelated_guard.py::test_a_guard_this_land_broke"

# `make_guard_tree` writes `src` as the tree's one guard at `tests/test_guard.py`, which is
# the file `PINNED_NODE` names. Both stand-in files carry `vault_root(` because that token is
# what puts a file in `guard_selection`; only `tests/test_guard.py` mentions
# `promotions.jsonl`, and that difference IS the scoping question — an ack on a ledger speaks
# for the guard that reads the ledger, not for one the land genuinely broke.
PINNED_TEST_SRC = '''"""Stand-in for tests/test_retention_sweep.py: a node that pins a figure in the mirror."""


def vault_root():
    return None


def test_the_mirror_row_count_the_item_quotes():
    """Pins the mirror at one row, the way `_WITNESS_ROWS` pins 28,678."""
    rows = len((vault_root() / "backlog" / "data" / "promotions.jsonl").read_text().splitlines())
    assert rows == 1, f"{rows} rows, not the 1 the pinned figure was measured on"
'''

UNRELATED_TEST_SRC = '''"""A guard this land breaks for an unrelated reason; it never reads the mirror."""


def vault_root():
    return None


def test_a_guard_this_land_broke():
    assert False, "broken by this land, and nothing to do with the mirror"
'''


def _witness_tree(tmp_path):
    """A committed checkout: `tests/test_guard.py` pins the mirror, `tests/test_unrelated_guard.py` does not."""
    tree = make_guard_tree(tmp_path / "witness-tree", src=PINNED_TEST_SRC)
    (tree / "tests" / "test_unrelated_guard.py").write_text(UNRELATED_TEST_SRC,
                                                            encoding="utf-8")
    git(tree, "add", "-A")
    git(tree, "commit", "-q", "-m", "a second vault-reading guard")
    return tree


def _pin_runs(monkeypatch, *, breaks_sibling: bool = False):
    """`_run_selection` that answers from the VAULT BYTES, the way pytest would.

    `PINNED_NODE` passes over a one-row mirror — the vault as it stands, which is what the
    pinned figure was measured on — and fails over any other row count, which is exactly a
    stale pin meeting refreshed bytes. `breaks_sibling` adds the other shape: a guard that
    passes on the mirror as it stands and fails on the proposed bytes, i.e. one this land
    genuinely broke rather than one whose figure is stale. Both are NEW failures under the
    probe's own attribution rule, which is why the scoping has to be by file and not by
    freshness. The pre-land run is the same call over the mirror put back, so it reproduces
    the passes: the rule the ack rides past is the thing being tested, not a formality.
    """
    calls: list = []

    def fake(python, tree, files, vault_root, data_root, mark, timeout, workers):
        mirror = Path(vault_root) / MIRROR_PATH
        rows = len(mirror.read_text(errors="replace").splitlines()) if mirror.is_file() else 0
        calls.append(rows)
        failed = [PINNED_NODE] if rows != 1 else []
        if rows != 1 and breaks_sibling:
            failed.append(UNRELATED_NODE)
        return {"ran": 2, "failed": failed, "note": "", "seconds": 0.01,
                "files": list(files), "workers": workers, "excerpt": "",
                "argv_tail": "", "parallel_only_failures": []}

    # `tests/conftest.py` sets the probe's nesting flag for the whole suite, which is the
    # rule that keeps a suite-run `land()` from launching a second ~70 s probe. These nodes
    # ARE the probe, so the flag has to come off here — #2042's node does the same at
    # `tests/test_automod_vault_round.py:2343` for the same reason.
    monkeypatch.delenv(VG.NESTING_ENV, raising=False)
    monkeypatch.setattr(VG, "_run_selection", fake)
    return calls


def _stage_mirror_refresh(vault, *, stands: int = 1, proposed: int = 2) -> None:
    """Commit the mirror as it stands, then put the refresh on disk UNCOMMITTED.

    That split is the real shape of a vault round: `land()` probes before it commits, so the
    proposal is working-tree bytes and HEAD still holds the copy the pins were measured on.
    `baseline_vault` builds the pre-land mirror with `git show HEAD:<path>`, so an
    already-committed refresh would make the "before" side identical to the "after" side, both
    runs would fail, and every disagreement would read as pre-existing rather than the land's.
    """
    ledger = vault / MIRROR_PATH
    ledger.parent.mkdir(parents=True, exist_ok=True)

    def write(rows: int) -> None:
        ledger.write_text("".join('{"event": "landed", "n": %d}\n' % i
                                  for i in range(rows)), encoding="utf-8")

    write(stands)
    git(vault, "add", "-A")
    git(vault, "commit", "-q", "-m", f"mirror as it stands ({stands} rows)")
    write(proposed)


def test_an_acknowledged_pinned_witness_leaves_the_probe_checked_and_not_refused(
        monkeypatch, tmp_path, vault):
    """Clause 1: with the ack, a node that passes on the old bytes and fails on the new ones
    leaves `state="checked"` with `refuse` false — and says which node it excused.

    The node is not hypothetical: it passes with the landed path put back (one row, what the
    pin was measured on) and fails against the proposed bytes (two rows), so `refuse` would be
    true on the probe's own attribution rule. The ack is the statement that the pin moves in
    the same change.
    """
    tree = _witness_tree(tmp_path)
    _stage_mirror_refresh(vault)
    _pin_runs(monkeypatch)

    report = _probe(tree, vault, tmp_path, paths=[MIRROR_PATH], ack=[MIRROR_PATH])

    assert report["ack"] == {"requested": [MIRROR_PATH], "accepted": [MIRROR_PATH],
                             "unmatched": []}, report["ack"]
    assert report["state"] == "checked", report["reason"]
    assert report["refuse"] is False, (
        f"the ack did not speak for the node and refuse stayed set: {report['reason']}")
    assert report["nodes"] == [], (
        f"an excused node must not also be an unanswered one: {report['nodes']}")
    assert report["excused"] == [PINNED_NODE], report.get("excused")
    assert "excused" in report["reason"] and MIRROR_PATH in report["reason"], report["reason"]
    assert report["candidate"]["ran"] > 0 and report["baseline"]["ran"] > 0, (
        "an excuse must not arrive with a zero denominator beside it")


def test_the_probe_still_refuses_without_the_ack_and_with_one_for_another_path(
        monkeypatch, tmp_path, vault):
    """Clause 2: the same disagreement refuses when nothing was acknowledged, and when the ack
    names a path this land does not declare.

    Both halves are load-bearing. Without the first, the excuse is the default and the probe
    stops catching a vault change that breaks a code-side guard reading the vault. Without the
    second, a caller can pre-authorise a future disagreement by naming a file it is not
    landing — the same hole by another route. `unmatched` records the attempt either way.
    """
    tree = _witness_tree(tmp_path)
    _stage_mirror_refresh(vault)
    _pin_runs(monkeypatch)

    plain = _probe(tree, vault, tmp_path, paths=[MIRROR_PATH])
    assert plain["refuse"] is True, plain["reason"]
    assert plain["nodes"] == [PINNED_NODE], plain["nodes"]
    assert PINNED_NODE in plain["reason"], plain["reason"]
    assert plain["ack"] == {"requested": [], "accepted": [], "unmatched": []}, plain["ack"]
    assert "excused" not in plain, plain

    wrong = _probe(tree, vault, tmp_path, paths=[MIRROR_PATH],
                   ack=["skills/foo/SKILL.md"])
    assert wrong["refuse"] is True, wrong["reason"]
    assert PINNED_NODE in wrong["nodes"], wrong["nodes"]
    assert PINNED_NODE in wrong["reason"], wrong["reason"]
    assert wrong["ack"] == {"requested": ["skills/foo/SKILL.md"], "accepted": [],
                            "unmatched": ["skills/foo/SKILL.md"]}, wrong["ack"]
    assert "excused" not in wrong, (
        "an ack for a path outside the land excused nothing, yet something was excused")


def test_an_ack_never_excuses_a_node_whose_file_does_not_read_the_landed_path(
        monkeypatch, tmp_path, vault):
    """Clause 2's scope: a land that refreshes a pinned witness AND breaks an unrelated guard
    is excused for the first and still refused for the second.

    Scoping is by what the failing node's own file NAMES, because that is the only difference
    between "this figure is stale until the same commit re-derives it" and "this land broke
    something". `tests/test_unrelated_guard.py` reads a vault root and never names
    `promotions.jsonl`, so it stays
    in `nodes` while its sibling is excused — and the refusal counts what was excused, so a
    partial excuse cannot read as a clean pass.
    """
    tree = _witness_tree(tmp_path)
    _stage_mirror_refresh(vault)
    _pin_runs(monkeypatch, breaks_sibling=True)

    report = _probe(tree, vault, tmp_path, paths=[MIRROR_PATH], ack=[MIRROR_PATH])
    assert report["refuse"] is True, report["reason"]
    assert report["excused"] == [PINNED_NODE], report.get("excused")
    assert report["nodes"] == [UNRELATED_NODE], report["nodes"]
    assert str(len(report["excused"])) in report["reason"], report["reason"]


def test_a_land_that_proceeds_on_an_ack_carries_it_and_the_excused_ids_on_its_row(
        monkeypatch, vault):
    """Clause 3: an excused failure is never silent — the row carries the ack, the node ids it
    excused, and the probe's denominator beside them.

    `land()` runs for real over a `VG.agreement` stub shaped like a successful excuse, so what
    is graded is the projection into the ledger row: `guards.ack.accepted`, `guards.excused`,
    `guards.candidate` `ran`/`failed` and `guards.baseline.ran` on ONE row. The second half
    drives the mixed report — one id excused, one still unanswered — and asserts the land is
    refused with both still on the row, because an excuse that swallowed a real failure is
    exactly the silence this clause is written against.
    """
    (vault / "backlog" / "data").mkdir(parents=True, exist_ok=True)
    (vault / MIRROR_PATH).write_text('{"event": "landed", "n": 0}\n', encoding="utf-8")
    checked = {"state": "checked", "refuse": False, "reason": "1 excused", "nodes": [],
               "excused": [PINNED_NODE],
               "ack": {"requested": [MIRROR_PATH], "accepted": [MIRROR_PATH],
                       "unmatched": []},
               "candidate": {"ran": 3774, "failed": [PINNED_NODE, UNRELATED_NODE],
                             "seconds": 156.0, "files": ["tests/test_guard.py"]},
               "baseline": {"ran": 3774, "failed": [UNRELATED_NODE], "seconds": 12.0},
               "seconds": 211.8}
    mixed = {**checked, "refuse": True, "nodes": [UNRELATED_NODE],
             "reason": "1 new failure after 1 were excused"}
    seen: dict = {}

    def agree(*, paths, **kw):
        seen.clear()
        seen.update(kw, paths=list(paths))
        return dict(mixed if agree.refuse else checked)

    agree.refuse = False
    monkeypatch.setattr(VG, "agreement", agree)
    out = V.land([MIRROR_PATH], "refresh the pinned witness", item_id=9,
                 ack=[MIRROR_PATH])
    assert out["ok"] is True, out
    row = _events("vault_land")[-1]
    assert row["ok"] is True and row["item_id"] == 9, row
    guards = row["guards"]
    assert guards["ack"] == {"requested": [MIRROR_PATH], "accepted": [MIRROR_PATH],
                             "unmatched": []}, guards
    assert guards["excused"] == [PINNED_NODE], guards
    assert (guards["candidate"]["ran"], guards["candidate"]["failed"]) == (3774, 2), guards
    assert guards["candidate"]["seconds"] == 156.0, guards
    assert guards["baseline"]["ran"] == 3774, guards
    assert guards["refuse"] is False and guards["state"] == "checked", guards

    agree.refuse = True
    with pytest.raises(V.VaultRoundError):
        V.land([MIRROR_PATH], "refresh the pinned witness, refused", item_id=9)
    refused = _events("vault_land")[-1]
    assert refused["ok"] is False, refused
    assert refused["guards"]["excused"] == [PINNED_NODE], refused["guards"]
    assert refused["guards"]["nodes"] == [UNRELATED_NODE], refused["guards"]


def test_land_asks_the_probe_about_its_paths_and_the_ack_and_nothing_that_moves_it(
        monkeypatch, vault):
    """Clause 4: `land()` hands the probe its paths and the ack and NOTHING that re-points the
    probe at a tree, so the probe keeps judging HEAD's code.

    Two halves, both about the hole #2036 closed. The call `land()` really makes carries no
    `live_root`, so no caller can aim the probe at a candidate checkout and have it agree with
    itself; and `land` takes no such parameter, so there is no keyword that gets there. The ack
    is the only new door and it buys a narrower thing.
    """
    import inspect

    (vault / "backlog" / "data").mkdir(parents=True, exist_ok=True)
    (vault / MIRROR_PATH).write_text('{"event": "landed", "n": 0}\n', encoding="utf-8")
    seen: dict = {}

    def agree(*, paths, **kw):
        seen.clear()
        seen.update(kw, paths=list(paths))
        return {"state": "checked", "refuse": False, "reason": "", "nodes": []}

    monkeypatch.setattr(VG, "agreement", agree)
    V.land([MIRROR_PATH], "a land with an ack", item_id=9, ack=[MIRROR_PATH])

    assert "live_root" not in seen, (
        f"land() handed the probe a tree to judge: {sorted(seen)} — it must judge HEAD, "
        "which is the hole #2036 closed")
    assert seen["paths"] == [MIRROR_PATH], seen
    assert seen["ack"] == [MIRROR_PATH], seen
    assert "live_root" not in inspect.signature(V.land).parameters, (
        "land() must not accept a tree for the probe to judge")

    # And a land WITHOUT an ack makes byte-identically the call it made before this
    # parameter existed, so the door is narrower than "always pass something".
    seen.clear()
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# 9\n")
    V.land(["backlog/9-item.md"], "a land with no ack", item_id=10)
    assert set(seen) == {"paths"}, (
        f"an unacknowledged land changed the probe call: {sorted(seen)}")


def test_the_cli_land_forwards_its_ack_flag_and_invents_none_without_it(monkeypatch):
    """Clause 4: `--ack` is the human-driven half of the same door.

    The MCP handler is the surface an autonomous round drives; the CLI is the one a
    person at a terminal drives, and the land that needed the acknowledgement in the
    first place — refreshing the rolling promotions mirror, whose row count the code
    agreement probe pins — is the kind of one-off a human runs. A flag argparse never
    declared makes the acknowledgement chat-only, so `land()`'s parameter stays
    unreachable off the MCP path.

    Both halves are here because the interesting failure is the second one: a
    subparser that declares `--ack` and then forwards it unconditionally, so an
    ordinary `land -m MSG PATH` sends `ack=None` or `ack=[]` and the row reads as
    though somebody excused a change they did not name. Nothing reaches a real vault:
    the lander is replaced, and `main()` only parses and calls it.
    """
    import inspect

    bind = inspect.signature(V.land).bind
    seen: list = []

    def rec(*a, **k):
        seen.append(bind(*a, **k).arguments)
        return {"ok": True, "commit": "not-landed", "paths": list(a[0])}

    monkeypatch.setattr(V, "land", rec)

    rc = V.main(["land", "-m", "refresh the witness", "--ack", MIRROR_PATH, MIRROR_PATH])
    assert rc == 0, "the acked land exited non-zero"
    assert len(seen) == 1, seen
    assert seen[0]["paths"] == [MIRROR_PATH], dict(seen[0])
    assert seen[0]["message"] == "refresh the witness", dict(seen[0])
    assert seen[0].get("ack") == [MIRROR_PATH], (
        f"--ack never reached land(): {dict(seen[0])}")

    seen.clear()
    rc = V.main(["land", "-m", "an ordinary land", "backlog/9-item.md"])
    assert rc == 0, "the ack-less land exited non-zero"
    assert len(seen) == 1, seen
    assert "ack" not in seen[0], (
        f"an ack-less CLI land handed land() an ack of {seen[0]['ack']!r}")


# ─────────────────────────────────────────────────────────────────────────────
# #2190 — the #724 dispatch-affecting-field rail has two doors.
#
# `policy.py` refuses an unattended `autonomy_write_task` call that moves a
# field the scheduler reads to decide whether to run a task, and until this
# section the vault route had no equivalent check: `validate()` re-parsed a
# touched `autonomy/*.md` and nothing else. The nodes below drive the real
# `validate()` (and `land()` for the end-to-end one) over a fixture git repo,
# never `~/obsidian`.
#
# Every dispatch field at HEAD, so a refusal is a changed VALUE and nothing
# else; the fixture carries all seven because the guard reads the frozenset,
# not this list.
# ─────────────────────────────────────────────────────────────────────────────

TASK_HEAD_FM: dict = {
    "id": 1,
    "name": "one",
    "status": "up_next",
    "frequency": "daily",
    "skill_name": "foo",
    "scheduled_at": "2026-10-06T04:00:00+00:00",
    "depends_on": 24,
    "auto_advance": False,
    "preferred_hours": {"start": "04:00", "end": "05:00"},
    "description": "original description",
    "agent_id": "primary",
    "priority": "medium",
}

TASK_BODY = "\n# task\n\nOriginal body paragraph.\n"


def fm_task(fm: dict, body: str = TASK_BODY) -> str:
    """A task file: `fm` dumped as front matter, then `body`."""
    return "---\n" + yaml.safe_dump(fm, sort_keys=False) + "---\n" + body


def _task_at(repo, rel: str, text: str) -> None:
    (repo / rel).write_text(text)


@pytest.fixture
def armed_task_vault(tmp_path, monkeypatch):
    """The scratch vault, its HEAD moved to a task carrying every dispatch field.

    `status: up_next` at HEAD is the shape the 2026-10-01 route reached for: a
    file the scheduler will dispatch. The one added commit is this fixture's
    own, so `HEAD:<path>` is the baseline every refusal below compares against.
    """
    r = _scratch_vault(tmp_path, monkeypatch, mock_loaders=True)
    _task_at(r, "autonomy/1-task.md", fm_task(TASK_HEAD_FM))
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "task 1 carries every dispatch field")
    return r


def test_a_round_that_moves_an_existing_task_s_status_is_refused_by_name(armed_task_vault):
    """Clause 1: the error names the path, the field, and the #724 rail."""
    f = armed_task_vault / "autonomy" / "1-task.md"
    f.write_text(fm_task({**TASK_HEAD_FM, "status": "draft"}))
    errors, _buckets = V.validate(["autonomy/1-task.md"])
    assert len(errors) == 1, errors
    why = errors[0]
    assert "autonomy/1-task.md" in why, why
    assert "status" in why, why
    assert "#724" in why, why
    assert "draft" in why and "up_next" in why, f"the refusal must quote both values: {why}"


def test_the_refusal_reverts_the_task_file_and_names_the_rail_on_the_ledger_row(armed_task_vault):
    """The same refusal through `land()`: nothing lands, the row says why.

    The rail is only legible after the fact if the refusal writes it into the
    ledger's `errors`, which is what #2148 established for the skill gate.
    """
    f = armed_task_vault / "autonomy" / "1-task.md"
    head_text = f.read_text()
    f.write_text(fm_task({**TASK_HEAD_FM, "status": "draft"}))
    with pytest.raises(V.VaultRoundError, match="reverted"):
        V.land(["autonomy/1-task.md"], "park task 1", item_id=2190)
    assert f.read_text() == head_text, "the disarmed file must be back at HEAD"
    assert git(armed_task_vault, "status", "--short").stdout.strip() == "", "a refusal leaves no diff"
    ev = _events("vault_land")[-1]
    assert ev["ok"] is False and ev["item_id"] == 2190
    assert any("status" in e and "#724" in e for e in ev["errors"]), ev["errors"]


@pytest.mark.parametrize("field", sorted(SCHEDULE_STATE_FIELDS))
def test_every_field_of_the_policy_frozenset_is_refused_when_its_value_moves(armed_task_vault,
                                                                            field):
    """Clause 4: one definition — the refused set IS `policy.SCHEDULE_STATE_FIELDS`.

    Parametrised over the frozenset itself, so a field added there in `policy.py`
    is refused by this route with no edit to `vault_round.py` and no edit here:
    `fm_task` puts any key it is handed into the front matter, and moving a field
    that HEAD does not carry is a value change like any other. The fixture's HEAD
    copy spells every field it knows, which is why a NEW field still reads as
    absent-at-HEAD -> present rather than as a pass.
    """
    f = armed_task_vault / "autonomy" / "1-task.md"
    f.write_text(fm_task({**TASK_HEAD_FM, field: "moved-by-this-round"}))
    errors, _buckets = V.validate(["autonomy/1-task.md"])
    assert any(field in e and "#724" in e for e in errors), (
        f"{field} is in SCHEDULE_STATE_FIELDS but was not refused: {errors}")


def test_a_creating_round_that_arms_the_task_is_refused(armed_task_vault):
    """Clause 2, first half: the 2026-10-01 reproduction, refused.

    `autonomy/92-vllm-prefix-miss-daily.md` landed through this route at
    2026-10-01T18:51:05Z, eighteen seconds after the same round's
    `autonomy_write_task` was denied by the rail. It came in `status: draft`, so
    nothing was armed; this is the same call with the arming kept.
    """
    f = armed_task_vault / "autonomy" / "3-new.md"
    f.write_text(fm_task({"id": 3, "name": "three", "status": "up_next",
                          "skill_name": "foo", "frequency": "daily"}))
    errors, _buckets = V.validate(["autonomy/3-new.md"])
    assert any("autonomy/3-new.md" in e and "status" in e and "#724" in e for e in errors), errors


def test_the_armed_create_refusal_reaches_the_tool_route_end_to_end(armed_task_vault):
    """The same create through `land()`: the new file is deleted, not committed.

    `validate()` is the tool's gate — `automod_vault_land` is a thin wrapper over
    `land()` — so the reproduction has to come back refused there, not only when
    the validator is driven directly.
    """
    f = armed_task_vault / "autonomy" / "3-new.md"
    f.write_text(fm_task({"id": 3, "name": "three", "status": "up_next"}))
    with pytest.raises(V.VaultRoundError, match="reverted"):
        V.land(["autonomy/3-new.md"], "arm task 3", item_id=2190)
    assert not f.exists(), "a refused create must be deleted, as any new file is"
    assert "3-new" not in git(armed_task_vault, "log", "--oneline", "-1").stdout


def test_a_creating_round_that_leaves_the_task_draft_passes_with_its_skill_and_window(
        armed_task_vault):
    """Clause 2, second half: a draft create carrying `skill_name` and `frequency`.

    This is the shape that actually landed as task 92 (vault commit `1ec6da8b`),
    and the shape `skills/pipeline-dispatch` uses to hand work down the nightly
    chain: a new task that is inert until a human or a granted call arms it. A
    guard that refused it would deny the documented dispatch route to buy no
    safety — the same asymmetry `policy.schedule_fields_changed` draws.
    """
    f = armed_task_vault / "autonomy" / "3-new.md"
    f.write_text(fm_task({"id": 3, "name": "three", "status": "draft",
                          "skill_name": "nightly-skills-management",
                          "frequency": "daily"}))
    errors, _buckets = V.validate(["autonomy/3-new.md"])
    assert errors == [], errors
    out = V.land(["autonomy/3-new.md"], "add task 3 draft")
    assert out["ok"] and (armed_task_vault / "autonomy" / "3-new.md").exists()


def test_a_round_that_rewords_the_body_and_description_passes(armed_task_vault):
    """Clause 3: a value-equal front matter over changed prose is not a refusal.

    Refusing a touched file that merely *carries* a dispatch field would refuse
    every autonomy round ever, because every real task file already has a
    `status:` key. The guard compares values, so this round — new description,
    new body, dispatch values untouched — lands.
    """
    f = armed_task_vault / "autonomy" / "1-task.md"
    f.write_text(fm_task({**TASK_HEAD_FM, "description": "reworded, same schedule"},
                         body="\n# task\n\nRewritten body, more precise.\n"))
    errors, _buckets = V.validate(["autonomy/1-task.md"])
    assert errors == [], errors
    # No item_id: an item's contract would route this land into the vault
    # REVIEWER, which is a different gate from the schedule guard under test, and
    # the suite's conftest refuses a grader call that was never stubbed.
    assert V.land(["autonomy/1-task.md"], "task 1: reword the description")["ok"]


def test_a_round_that_requotes_the_yaml_but_moves_no_value_passes(armed_task_vault):
    """Clause 3's other half: values, not serialized YAML — and not bytes.

    Same parsed values quoted differently, keys in a different order, and a
    field the guard does not care about (`priority`) moved. The owed-after-landing
    measurement says a false refusal on a benign diff means this check is
    comparing serialized YAML rather than field values; quoting is how that
    mistake shows up in a fixture.
    """
    f = armed_task_vault / "autonomy" / "1-task.md"
    f.write_text(
        "---\n"
        "priority: high\n"
        "name: 'one'\n"
        'status: "up_next"\n'
        "skill_name: foo\n"
        "frequency: daily\n"
        "auto_advance: false\n"
        "depends_on: 24\n"
        "scheduled_at: '2026-10-06T04:00:00+00:00'\n"
        "preferred_hours:\n  end: '05:00'\n  start: '04:00'\n"
        "id: 1\n"
        "description: original description\n"
        "agent_id: primary\n"
        "---\n" + TASK_BODY)
    errors, _buckets = V.validate(["autonomy/1-task.md"])
    assert errors == [], errors


def test_a_round_that_drops_a_dispatch_field_entirely_is_refused(armed_task_vault):
    """Deleting a key is a value change: `frequency: daily` -> absent disarms a run.

    The tool gate cannot see the disk and so refuses a *proposal*; this route can
    read HEAD, so it refuses on the diff — which cuts both ways, and this is the
    direction that matters most: a file rewritten without a key the scheduler
    reads is the quiet one.
    """
    f = armed_task_vault / "autonomy" / "1-task.md"
    dropped = {k: v for k, v in TASK_HEAD_FM.items() if k != "frequency"}
    f.write_text(fm_task(dropped))
    errors, _buckets = V.validate(["autonomy/1-task.md"])
    assert any("frequency" in e and "#724" in e for e in errors), errors


def test_a_round_that_resends_the_same_armed_status_is_accepted(armed_task_vault):
    """The asymmetry the vault route must NOT copy from the tool gate.

    `policy.schedule_fields_changed`'s docstring is explicit that the tool gate
    reads the call and not the disk, so resending `up_next` to an already-armed
    task is denied there. This route reads `HEAD:<path>`, so a round that lands
    the file with `status` still `up_next` moves nothing and must pass — refusing
    it would refuse the nightly writer that rewrites a task file every time it
    records an activity note.
    """
    f = armed_task_vault / "autonomy" / "1-task.md"
    f.write_text(fm_task(dict(TASK_HEAD_FM), body="\n# task\n\nActivity note appended.\n"))
    errors, _buckets = V.validate(["autonomy/1-task.md"])
    assert errors == [], errors


def test_the_guard_holds_only_task_files_and_leaves_other_vault_paths_alone(armed_task_vault):
    """A guard on the dispatch state of tasks, not a guard on the vault.

    The same frozenset name (`status`) is a backlog item's field too; #9's
    promotion is nobody's dispatch decision, and refusing it here would spend
    the rail's credibility on a file the scheduler never reads.
    """
    (armed_task_vault / "backlog" / "9-item.md").write_text(
        "---\nstatus: up_next\nfrequency: daily\n---\n# item\n")
    errors, _buckets = V.validate(["backlog/9-item.md"])
    assert errors == [], errors


# ── #1850: a skill that reads the uptake verdict cannot drop a citability rule ──
#
# `nightly-reflection-knowledge-write` §2a is the only consumer of the uptake
# instrument's verdict, and its rules are what keep a nightly from citing a table
# that was never a measurement. The prose lives in the vault, so the gate's
# `tests` rung cannot read it (`live_vault` is excluded, `pytest.ini:6-12`) and a
# dropped sentence would land silently. Enforcement therefore sits on this route,
# beside `reflection_archive_errors` and `skill_timezone_errors`, and the rule is
# one definition in `scripts/uptake_citability.py` shared with the `live_vault`
# reporting node in `tests/test_uptake.py`.

#: Compliant: names the reroute key (the trigger), states a citability verdict
#: beside each of the three keys, and states the re-base facts (#1850: primary,
#: decided 2026-10-04, slot retired 2026-09-20).
UPTAKE_SKILL_COMPLIANT = """---
name: foo
---
# foo

1. Read the newest uptake table. **The scoring engine is the primary, by deliberate
   decision since 2026-10-04**; the secondary slot has been retired since 2026-09-20
   and is not coming back. A table with `engine_rerouted: true` is not a measurement,
   because its `measured` is forced false: its numbers are not citable, and it is not
   a comparable of later tables. A table carrying `audit_only: true` is not citable
   either, whatever its `measured` says: cite nothing, never archive on it.
"""

#: Only the `audit_only` verdict removed — the key is still named, so a reader is
#: told the marker exists and not what it forbids. This is the exact drift #1850's
#: clause 5 must not permit, because the nightly then archives on a stamped table.
UPTAKE_SKILL_AUDIT_RULE_DROPPED = UPTAKE_SKILL_COMPLIANT.replace(
    "`audit_only: true` is not citable\n   either, whatever its `measured` says: "
    "cite nothing, never archive on it.",
    "`audit_only: true` is written past a floor.")

#: Only the dated re-base removed: the rules survive, the decision that re-based
#: them does not, and gen-1 tables start being read as comparables again.
UPTAKE_SKILL_REBASE_UNDATED = UPTAKE_SKILL_COMPLIANT.replace(
    "since 2026-10-04", "by decision").replace(
    "since 2026-09-20", "long ago")


def test_the_vault_writer_refuses_a_skill_that_drops_an_uptake_citability_rule(vault):
    """The route refuses a §2a rewrite that removes a citability rule, and accepts
    the rule as written. #1850 clause 5's enforcement half.

    Driven through `validate` — what `land` calls before it commits — so this pins
    the wiring, not just the predicate. Three things have to be true for that to
    mean anything: the compliant text passes (the guard is not a ban on the skill),
    each of the two mutations is refused naming what went, and a skill that never
    names the reroute key is never asked at all. That last control is why the rule
    can be always-on across 202 live skills: obligations are discovered from the
    candidate's own bytes, not from a list of slugs.
    """
    skill = vault / "skills" / "foo" / "SKILL.md"
    path = "skills/foo/SKILL.md"
    assert UPTAKE_SKILL_AUDIT_RULE_DROPPED != UPTAKE_SKILL_COMPLIANT, \
        "the audit_only fixture's replace is a no-op — its anchor drifted"
    assert UPTAKE_SKILL_REBASE_UNDATED != UPTAKE_SKILL_COMPLIANT, \
        "the re-base fixture's replace is a no-op — its anchor drifted"

    skill.write_text(UPTAKE_SKILL_COMPLIANT, encoding="utf-8")
    assert V.uptake_citability_errors([path]) == [], "the shipped rule was refused"
    errors, _buckets = V.validate([path])
    assert not [e for e in errors if "citability" in e], errors

    skill.write_text(UPTAKE_SKILL_AUDIT_RULE_DROPPED, encoding="utf-8")
    errs = V.uptake_citability_errors([path])
    assert any("audit_only" in e for e in errs), errs
    errors, _buckets = V.validate([path])
    assert any("uptake citability rule" in e and "audit_only" in e for e in errors), errors

    skill.write_text(UPTAKE_SKILL_REBASE_UNDATED, encoding="utf-8")
    errs = V.uptake_citability_errors([path])
    assert any("re-base" in e for e in errs), errs

    skill.write_text("---\nname: foo\n---\n# foo\n\nSay the thing.\n", encoding="utf-8")
    assert V.uptake_citability_errors([path]) == [], \
        "a skill that never names the reroute key must not be asked"


def test_the_uptake_citability_guard_survives_a_skill_that_moves_its_detail(vault):
    """A rule spilled into a sibling counts, and the trigger stays in the body.

    #624's spill route is the tree's sanctioned way to shorten a skill body, so a
    guard that read only `SKILL.md` would refuse a compliant skill and push its
    author to duplicate prose — and the reverse matters too: the body must still
    name the trigger key, or discovery would silently stop asking of a skill that
    kept reading the verdict. Both halves are pinned here.
    """
    from scripts import uptake_citability

    slug_dir = vault / "skills" / "foo"
    skill = slug_dir / "SKILL.md"
    sibling = slug_dir / "steps-citability.md"
    # The body keeps the trigger key (so the skill is still asked) and every key
    # name, and carries NO verdict sentence: discovery is the body, the content may
    # live in the sibling the body points at.
    body_only_names = """---
name: foo
---
# foo

1. Read the newest uptake table; `engine_rerouted`, `measured` and `audit_only`
   are explained in `steps-citability.md`.
"""
    skill.write_text(body_only_names, encoding="utf-8")
    sibling.write_text(
        "`engine_rerouted: true` is not a measurement and not citable; `audit_only: "
        "true` is not citable; `measured: false` means cite nothing. Decided "
        "2026-10-04: the scorer is the primary, the slot was retired 2026-09-20.\n",
        encoding="utf-8")
    assert uptake_citability.skill_rule_violations(
        "foo", body_only_names, detail=sibling.read_text()) == [], \
        "the rule lives in the sibling the body points at, and that is allowed"
    # The sibling is what saves it, not concatenation: with the sibling gone the
    # same body is refused, and a sibling that names the keys without a verdict
    # saves nothing either.
    assert uptake_citability.skill_rule_violations("foo", body_only_names) != [], (
        "the body alone passes, so the sibling was never what this node measured")
    assert uptake_citability.skill_rule_violations(
        "foo", body_only_names,
        detail="`engine_rerouted`, `measured` and `audit_only` are keys.\n") != [], (
        "a sibling that names the keys without saying what they permit must not "
        "clear the rule")
    assert V.uptake_citability_errors(["skills/foo/SKILL.md"]) == [], (
        "the landing route must agree with the rule about a spilled skill")
