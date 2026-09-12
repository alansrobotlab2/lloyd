"""arch-review — a production doc edits itself, and what keeps that safe.

Every review in this source writes to `~/lloyd`, which is the running tree.
The tests that matter are therefore not about the verdict; they are about the
four rails that decide what survives a turn:

  * the picklist never offers a retired doc, and a group resolves to its own
    lines through the decorated headings the two jobs docs actually use;
  * the scheduler rests a reviewed unit and parks a failing one, and counts a
    failure only when the failure was the unit's;
  * every write but the one doc is reverted, in this repo and in the vault;
  * the doc's own diff is thrown away when it breaks a bound — too large, too
    much deleted, front matter gone, or (for a group) a hunk in somebody
    else's section — and only what survives all of that is committed.

The fixtures build a real git repo and a real vault, because every one of
those rails is `git` behaviour and a mock of it would pin the mock.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from workers.queue import QueueItem
from workers.sources import arch_review as A
from workers.sources._common import DrainActive, TurnTimeout, WORKER_AUTOMOD_BAN


# ── fixtures ─────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True).stdout


def _init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")


DOC = """\
---
segment: architecture
type: reference
date: 2026-01-01
---

# The worker jobs

Intro prose that belongs to no section.

## 3. Dispatch — one door onto the autonomy fleet

Dispatch body line one.
Dispatch body line two.

### A subheading inside dispatch

---

Still dispatch, after a horizontal rule.

## 4. Self-modification — the loop that changes Lloyd's own code

Self-mod body.

## 6. Mining — Lloyd's own exhaust back out

Mining body one.
Mining body two.
"""

SOLO = """\
---
segment: architecture
type: reference
date: 2026-01-01
---

# Memory

Body line 1.
Body line 2.
Body line 3.
Body line 4.
Body line 5.
"""


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A tmp `~/lloyd` and a tmp vault, both git repos, both wired into the
    module and into every collaborator it reaches through."""
    from scripts.automod import backlog as B, state as S

    repo = tmp_path / "lloyd"
    _init(repo)
    arch = repo / "architecture"
    arch.mkdir()
    (arch / "workers-jobs.md").write_text(DOC)
    (arch / "memory.md").write_text(SOLO)
    (arch / "voice.md").write_text(SOLO.replace("# Memory", "# Voice"))
    archive = arch / ".archive"
    archive.mkdir()
    (archive / "groundskeeper.md").write_text("# retired\n")
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")

    vault = tmp_path / "obsidian"
    _init(vault)
    for sub in ("backlog", "skills", "autonomy"):
        (vault / sub).mkdir()
        (vault / sub / ".keep").write_text("")
    _git(vault, "add", "-A")
    _git(vault, "commit", "-qm", "seed")

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(A, "LLOYD_HOME", repo)
    monkeypatch.setattr(A, "VAULT_ROOT", vault)
    monkeypatch.setattr(S, "STATE_DIR", state)
    monkeypatch.setattr(S, "LEDGER_PATH", state / "promotions.jsonl")
    monkeypatch.setattr(S, "LOCK_PATH", state / "lock")
    monkeypatch.setattr(B, "BACKLOG_DIR", vault / "backlog")
    return {"repo": repo, "vault": vault, "state": state, "arch": arch,
            "backlog": vault / "backlog", "ledger": state / "promotions.jsonl"}


def await_sync(coro):
    """Drive one coroutine from a sync test."""
    return asyncio.run(coro)


def _item(payload: dict, item_id: int = 1) -> QueueItem:
    return QueueItem(id=item_id, source=A.NAME, kind="unit", priority=62, payload=payload,
                     dedup_key=None, state="running", attempts=1, enqueued_at="",
                     claimed_at=None, claimed_by=None, completed_at=None, error=None)


def _payload(unit: str, kind: str, doc: str, name: str = "", **over) -> dict:
    p = {"unit": unit, "kind": kind, "doc": doc, "name": name, "max_turns": 5,
         "spawn_cap": 5, "max_delta_lines": 400, "max_shrink_pct": 30}
    p.update(over)
    return p


BLOCK = """\
Reviewed it.

DOC_STATUS: {status}
DOC_UPDATED: {updated}
GROUPING: {grouping}
SUMMARY: Checked every path and the two counts.
FILED: {filed}
APPENDED_TO: {appended}
"""


def _block(status="current", updated="yes", grouping="none", filed="none", appended="none"):
    return BLOCK.format(status=status, updated=updated, grouping=grouping,
                        filed=filed, appended=appended)


def _turn(text: str, *, edits=None, raises: Exception | None = None,
          stop_reason: str = "stop", structured=None):
    """A stand-in session that really writes, so the git rails see real files."""
    async def run(prompt, **kwargs):
        run.prompt = prompt
        run.kwargs = kwargs
        if edits is not None:
            edits()
        if raises is not None:
            raise raises
        return {"text": text, "session_id": "20260911_archrevi_ab12",
                "stop_reason": stop_reason, "num_turns": 7, "errors": [],
                "structured": structured, "structured_error": "",
                "finalizer_tokens": 120}
    run.prompt = ""
    run.kwargs = {}
    return run


def _write_item(backlog: Path, item_id: int, *, slug: str, name: str = "",
                tags=("arch-review", "spawned-by-review"), status="draft",
                title="A finding", provenance: str | None = None) -> Path:
    line = A.provenance_line(slug, name) if provenance is None else provenance
    p = backlog / f"{item_id}-{title.lower().replace(' ', '-')}.md"
    p.write_text(
        "---\n"
        f"id: {item_id}\nboard: lloyd\nstatus: {status}\n"
        "tags:\n" + "".join(f"  - {t}\n" for t in tags) +
        "---\n\n"
        f"# {title}\n\n{line}\n\nDetail.\n")
    return p


# ── 1. the interface ─────────────────────────────────────────────────────────


def test_the_source_has_the_interface_the_pool_calls():
    assert A.NAME == "arch-review"
    assert isinstance(A.DEFAULT_PRIORITY, int)
    assert A.LONG_LIVED is True, "an hour-long doc review must be held by the KV gate"
    assert inspect.iscoroutinefunction(A.execute)
    assert inspect.iscoroutinefunction(A.enqueue_if_due)


def test_neither_coroutine_blocks_the_event_loop():
    """The same rule `tests/test_workers_pool.py` applies to every source, and
    applied here to `enqueue_if_due` too — this one shells out to git on the
    tick as well as in the turn."""
    blocking = ("subprocess.run", "subprocess.check_output", "subprocess.call",
                "time.sleep(", "urlopen(")
    for fn in (A.execute, A.enqueue_if_due):
        src = inspect.getsource(fn)
        for pattern in blocking:
            assert pattern not in src, f"{fn.__name__} calls {pattern} on the event loop"


def test_the_toolbox_denies_what_a_doc_review_must_not_do():
    """`Write` is denied because the doc exists: a new file is either a finding
    to file or a stray write. `Task` is denied because a subagent's writes land
    on the parent's turn and would arrive after the diff was measured."""
    for name in WORKER_AUTOMOD_BAN:
        assert name in A.DISALLOWED
    for name in ("Task", "Write", "vault_write", "autonomy_write_task",
                 "research_propose", "graph_refresh", "http_request"):
        assert name in A.DISALLOWED
    for name in ("Bash", "Edit", "Read", "Grep", "Glob", "backlog_write_task",
                 "graph_explain", "graph_affected"):
        assert name not in A.DISALLOWED, f"{name} is how the review does its job"


async def test_inner_voice_is_never_passed_as_a_literal(tree, monkeypatch):
    """`run_prompt_in_session` resolves it from config. A caller that passes a
    literal makes the per-source switch read as broken the one time somebody
    uses it — which is what `deep-research` did."""
    run = _turn(_block())
    monkeypatch.setattr(A, "run_prompt_in_session", run)
    await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert "inner_voice" not in run.kwargs
    assert run.kwargs["source"] == A.NAME
    assert set(A.DISALLOWED) <= set(run.kwargs["extra_disallowed"])
    assert run.kwargs["final_schema"] is A.ARCH_REVIEW_SCHEMA


# ── 2. the picklist ──────────────────────────────────────────────────────────


def test_only_top_level_docs_are_offered(tree):
    slugs = A.doc_slugs(tree["repo"])
    assert slugs == ["memory", "voice", "workers-jobs"]
    assert "groundskeeper" not in slugs, "a doc retired for being wrong is never reviewed"


def test_a_tracked_archive_doc_is_still_not_offered(tree):
    """Five archived docs were tracked when this landed, so "it is gitignored"
    was never the reason they are excluded — the path is."""
    tracked = _git(tree["repo"], "ls-files", "architecture/.archive/")
    assert "groundskeeper.md" in tracked, "the fixture tracks it, as the live tree did"
    assert "groundskeeper" not in A.doc_slugs(tree["repo"])


def test_a_group_names_a_doc_and_a_section(tree):
    cfg = {"groups": ["workers-jobs:Dispatch", "memory:Nope", "bad-entry", "gone:Thing"]}
    units = A.all_units(cfg, tree["repo"])
    ids = [u["unit"] for u in units]
    assert ids[:3] == ["doc:memory", "doc:voice", "doc:workers-jobs"], "docs first, then groups"
    assert "group:workers-jobs:Dispatch" in ids
    assert "group:memory:Nope" in ids, "a bad section is offered and fails at execute time"
    assert "group:gone:Thing" not in ids, "a group whose doc is gone is dropped on the picklist"
    assert not any("bad-entry" in i for i in ids)


@pytest.mark.parametrize("heading,want", [
    ("## Distil: #38 #42 #39", "distil"),
    ("## 6. Mining — Lloyd's own exhaust back out", "mining"),
    ("## 3. Dispatch — one door onto the autonomy fleet", "dispatch"),
    ("## Build the graph: #24, #51", "build the graph"),
    ("## Plain", "plain"),
])
def test_a_decorated_heading_still_matches_its_name(heading, want):
    """Both jobs docs list their members in the heading, and that list changes
    whenever a member does. Matching the whole line would make every regroup a
    `section_missing`."""
    assert A.heading_key(heading.removeprefix("## ")) == want


def test_a_section_runs_to_the_next_level_two_heading(tree):
    text = (tree["arch"] / "workers-jobs.md").read_text()
    start, end = A.find_section(text, "Dispatch")
    lines = text.splitlines()
    assert lines[start - 1].startswith("## 3. Dispatch")
    assert lines[end].startswith("## 4."), "it ends on the line before the next `## `"
    body = "\n".join(lines[start:end])
    assert body.rstrip().endswith("Still dispatch, after a horizontal rule.")
    assert "### A subheading inside dispatch" in body, "a ### belongs to its section"
    assert "---" in body, "a horizontal rule does not end a section"
    assert "Self-modification" not in body


def test_the_last_section_runs_to_end_of_file(tree):
    text = (tree["arch"] / "workers-jobs.md").read_text()
    start, end = A.find_section(text, "Mining")
    assert end == len(text.splitlines())


def test_a_missing_heading_is_none(tree):
    assert A.find_section((tree["arch"] / "workers-jobs.md").read_text(), "Nope") is None


# ── 3. the scheduler ─────────────────────────────────────────────────────────


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(timespec="seconds")


def test_never_reviewed_comes_first_then_oldest(tree):
    units = [{"unit": f"doc:{s}", "kind": "doc", "doc": s, "name": ""}
             for s in ("a", "b", "c")]
    state = {"doc:a": {"last_reviewed_at": _iso(40)}, "doc:b": {"last_reviewed_at": _iso(90)}}
    got = A.pending_units(units, state, now=time.time(), interval_days=30,
                          retry_spacing=1, max_attempts=3)
    assert [u["unit"] for u in got] == ["doc:c", "doc:b", "doc:a"]


def test_a_recently_reviewed_unit_rests(tree):
    row = {"last_reviewed_at": _iso(5)}
    assert not A.is_pending(row, now=time.time(), interval_days=30,
                            retry_spacing=1, max_attempts=3)
    assert A.is_pending(row, now=time.time(), interval_days=1,
                        retry_spacing=1, max_attempts=3)


def test_three_failures_park_a_unit_until_the_spacing_elapses():
    now = time.time()
    row = {"attempts": 3, "last_attempt_at": _iso(0.01)}
    assert not A.is_pending(row, now=now, interval_days=30, retry_spacing=21600, max_attempts=3)
    old = {"attempts": 3, "last_attempt_at": _iso(1)}
    assert A.is_pending(old, now=now, interval_days=30, retry_spacing=21600, max_attempts=3)


def test_daily_max_is_counted_from_the_ledger(tree):
    """From the ledger, not a counter in the state file: the ledger is what
    survives the state file being deleted, and `daily_max` bounds how much of
    the board one day may rewrite."""
    from scripts.automod import state as S
    for age in (0.1, 0.5, 2.0):  # hours ago... the third is 2 days
        S.append_event({"event": "arch_review", "unit": "doc:x",
                        "ts": time.time() - age * 86400})
    S.append_event({"event": "gate", "round_id": "SM_x"})
    assert A.reviews_today(tree["ledger"]) == 2


# ── 4. the prompt ────────────────────────────────────────────────────────────


def _prompt_for(tree, unit, kind, name="", **kw):
    text = (tree["arch"] / f"{unit}.md").read_text()
    section = A.find_section(text, name) if kind == "group" else None
    heading = (text.splitlines()[section[0] - 1].removeprefix("## ") if section else "")
    return A.build_prompt(
        {"kind": kind, "doc": unit, "name": name}, head="abc1234",
        last_reviewed=kw.get("last_reviewed", ""), doc_lines=len(text.splitlines()),
        section=section, heading=heading, already_filed=kw.get("already_filed", []),
        groups_config=kw.get("groups_config"), spawn_cap=5, max_delta_lines=400,
        max_shrink_pct=30, today="2026-09-11", root=tree["repo"])


def test_a_doc_prompt_carries_its_slug_tags_and_provenance_rule(tree):
    p = _prompt_for(tree, "memory", "doc",
                    already_filed=[{"id": 41, "title": "Fix the memory doc"}])
    assert "architecture/memory.md" in p
    assert "`arch-review`, `spawned-by-review`, `memory`" in p
    assert "Found reviewing architecture/memory.md" in p
    assert "#41 Fix the memory doc" in p
    assert "never" in p and "git commit" in p
    assert "DOC_STATUS: <current|stale|superseded|aspirational>" in p
    assert "GROUPING: <none>" in p, "a whole doc has no grouping question"
    assert "Review log" in p


def test_a_group_prompt_carries_the_three_lenses_and_its_own_lines(tree):
    p = _prompt_for(tree, "workers-jobs", "group", "Dispatch")
    start, end = A.find_section((tree["arch"] / "workers-jobs.md").read_text(), "Dispatch")
    assert f"lines {start}-{end}" in p
    assert f"only inside lines {start + 1}-{end}" in p, "the heading line is never editable"
    assert "(a) **Architecture**" in p and "(b) **Code**" in p and "(c) **Drift**" in p
    assert "GROUPING: <holds|split|merge|move>" in p
    assert "Found reviewing architecture/workers-jobs.md §Dispatch" in p
    assert "Review log" not in p, "a group must not add a log section to a shared doc"


def test_provenance_keys_on_the_configured_name_not_the_decorated_heading(tree):
    """The heading lists the group's members and changes whenever one does —
    which is exactly what `heading_key` exists to ignore. Keying the
    provenance line on it would make every prior finding invisible to the next
    review of that group and re-file the lot, which is the one failure
    `<already_filed>` exists to prevent."""
    assert A.provenance_line("workers-jobs", "Dispatch") == \
        "Found reviewing architecture/workers-jobs.md §Dispatch"
    doc = tree["arch"] / "workers-jobs.md"
    doc.write_text(doc.read_text().replace(
        "## 3. Dispatch — one door onto the autonomy fleet",
        "## 3. Dispatch: #12, #13 — one door onto the autonomy fleet"))
    p = _prompt_for(tree, "workers-jobs", "group", "Dispatch")
    assert "Found reviewing architecture/workers-jobs.md §Dispatch" in p, \
        "the same line survives a heading the doc has re-decorated"


async def test_a_prior_finding_survives_a_redecorated_heading(tree, monkeypatch):
    _write_item(tree["backlog"], 900, slug="workers-jobs", name="Dispatch", title="Old find")
    doc = tree["arch"] / "workers-jobs.md"
    doc.write_text(doc.read_text().replace("## 3. Dispatch — one door",
                                           "## 3. Dispatch: #12, #13 — one door"))
    _git(tree["repo"], "commit", "-qam", "regroup")
    run = _turn(_block(grouping="holds"))
    monkeypatch.setattr(A, "run_prompt_in_session", run)
    await A.execute(_item(_payload("group:workers-jobs:Dispatch", "group",
                                   "workers-jobs", "Dispatch")))
    assert "#900 Old find" in run.prompt


def test_a_jobs_doc_review_is_shown_the_configured_group_list(tree):
    p = _prompt_for(tree, "workers-jobs", "doc",
                    groups_config=["workers-jobs:Dispatch", "workers-jobs:Mining"])
    assert "<groups_config>" in p
    assert "workers-jobs:Dispatch" in p
    assert "that is a finding: file it" in p


def test_a_big_doc_is_told_to_read_in_sections(tree):
    big = tree["arch"] / "memory.md"
    big.write_text(SOLO + "filler\n" * 800)
    p = _prompt_for(tree, "memory", "doc")
    assert "offset/limit" in p


# ── 5. the verdict ───────────────────────────────────────────────────────────


def test_the_structured_object_wins_when_it_is_usable():
    parsed = A.parse_result(_block(status="stale"),
                            {"doc_status": "current", "doc_updated": True,
                             "grouping": "none", "summary": "s", "filed": [7],
                             "appended_to": []}, "doc")
    assert parsed["doc_status"] == "current" and parsed["source"] == "structured"
    assert parsed["filed"] == [7]


def test_an_unusable_object_falls_back_to_the_block():
    parsed = A.parse_result(_block(status="stale", filed="#12, #13"),
                            {"doc_status": "nonsense"}, "doc")
    assert parsed["doc_status"] == "stale" and parsed["source"] == "regex"
    assert parsed["filed"] == [12, 13]


def test_the_last_block_wins():
    """A model that states an outcome, reconsiders and restates would otherwise
    have its first verdict paired with its last evidence."""
    text = _block(status="current") + "\nOn reflection:\n\n" + _block(status="stale", filed="#9")
    parsed = A.parse_result(text, None, "doc")
    assert parsed["doc_status"] == "stale" and parsed["filed"] == [9]


def test_a_status_outside_the_vocabulary_is_not_a_verdict():
    assert A.parse_result(_block(status="mostly-fine"), None, "doc") is None
    assert A.parse_result("no block here at all", None, "doc") is None


def test_a_group_clamps_a_whole_doc_verdict_and_defaults_its_grouping():
    """A section cannot be superseded on its own — the doc it lives in holds
    that verdict — and a section describing something gone is `stale`."""
    parsed = A.parse_result(_block(status="superseded", grouping="split"), None, "group")
    assert parsed["doc_status"] == "stale" and parsed["grouping"] == "split"
    assert A.parse_result(_block(status="current", grouping="none"), None,
                          "group")["grouping"] == "holds"


def test_a_doc_never_reports_a_grouping():
    assert A.parse_result(_block(grouping="split"), None, "doc")["grouping"] is None


def test_the_schema_is_built_from_the_vocabularies():
    props = A.ARCH_REVIEW_SCHEMA["properties"]
    assert props["doc_status"]["enum"] == list(A.DOC_STATUSES)
    assert props["grouping"]["enum"] == [*A.GROUPINGS, A.GROUPING_NONE]
    assert A.ARCH_REVIEW_SCHEMA["additionalProperties"] is False
    for prop in props.values():
        assert "maxLength" not in prop, (
            "a guided decoder stops AT the limit rather than writing something "
            "shorter; clamps belong in Python, after the fact")


# ── 6. execute: the rails ────────────────────────────────────────────────────


def _events(tree) -> list[dict]:
    if not tree["ledger"].exists():
        return []
    return [json.loads(l) for l in tree["ledger"].read_text().splitlines() if l.strip()]


def _arch_events(tree) -> list[dict]:
    return [e for e in _events(tree) if e.get("event") == "arch_review"]


def _doc(tree, slug="memory") -> Path:
    return tree["arch"] / f"{slug}.md"


def _edit(path: Path, old: str, new: str):
    def go():
        path.write_text(path.read_text().replace(old, new))
    return go


async def test_a_clean_doc_edit_is_committed_and_nothing_else_is(tree, monkeypatch):
    doc = _doc(tree)
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn(_block(), edits=_edit(doc, "Body line 1.", "Body line one.")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["status"] == "success"
    assert _git(tree["repo"], "status", "--porcelain") == "", "the tree is left clean"
    subject = _git(tree["repo"], "log", "-1", "--format=%s").strip()
    assert subject == "arch-review: doc:memory — current"
    files = _git(tree["repo"], "show", "--name-only", "--format=", "HEAD").split()
    assert files == ["architecture/memory.md"], "exactly one file in the commit"
    ev = _arch_events(tree)[0]
    assert ev["doc_updated"] is True and ev["commit"] and not ev["doc_update_rejected"]
    state = A.load_state()
    assert state["doc:memory"]["attempts"] == 0
    assert state["doc:memory"]["last_reviewed_at"]
    assert state["doc:memory"]["verdict"] == "current"


async def test_an_unchanged_doc_is_not_committed(tree, monkeypatch):
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(updated="no")))
    before = _git(tree["repo"], "rev-parse", "HEAD")
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["status"] == "success"
    assert _git(tree["repo"], "rev-parse", "HEAD") == before
    assert _arch_events(tree)[0]["doc_updated"] is False


async def test_a_stray_write_in_the_repo_is_reverted(tree, monkeypatch):
    doc, other = _doc(tree), tree["repo"] / "README.md"

    def edits():
        doc.write_text(doc.read_text().replace("Body line 1.", "Body line one."))
        other.write_text("the model rewrote production\n")
        (tree["repo"] / "sneaky.py").write_text("print('new file')\n")
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(), edits=edits))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert other.read_text() == "hello\n", "a tracked stray write goes back to HEAD"
    assert not (tree["repo"] / "sneaky.py").exists(), "an untracked stray write is unlinked"
    assert "architecture/memory.md" in _git(tree["repo"], "show", "--name-only",
                                            "--format=", "HEAD")
    actions = {s["path"]: s["action"] for s in out["meta"]["stray_writes"]}
    assert actions == {"README.md": "checkout", "sneaky.py": "unlink"}
    assert _arch_events(tree)[0]["stray_writes"], "the strays are on the ledger event"


async def test_a_stray_write_in_the_vault_is_reverted_too(tree, monkeypatch):
    """`skills/` and `autonomy/` only: filing into `backlog/` is the job, and a
    sweep that reverted it would delete the review's findings."""
    skill = tree["vault"] / "skills" / "some-skill.md"
    kept = tree["backlog"] / "900-a-finding.md"

    def edits():
        skill.write_text("the model fixed a skill instead of filing it\n")
        kept.write_text("---\nid: 900\n---\n# filed\n")
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(), edits=edits))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert not skill.exists(), "a skill fix is FILED, never made"
    assert kept.exists(), "the backlog item it filed survives"
    assert [s["path"] for s in out["meta"]["stray_writes"]] == ["skills/some-skill.md"]


async def test_a_pre_existing_dirty_path_is_not_reverted(tree, monkeypatch):
    """A diff against a baseline, never a snapshot: a human with an editor open
    in `~/lloyd` must not have their work thrown away by a doc review."""
    human = tree["repo"] / "README.md"
    human.write_text("a human is mid-edit\n")
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(updated="no")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert human.read_text() == "a human is mid-edit\n"
    assert out["meta"]["stray_writes"] == []


async def test_a_doc_a_human_is_editing_is_skipped_without_an_attempt(tree, monkeypatch):
    doc = _doc(tree)
    doc.write_text(doc.read_text() + "\nsomeone is mid-edit\n")
    called = {"n": 0}

    async def never(*a, **k):
        called["n"] += 1
        return {}
    monkeypatch.setattr(A, "run_prompt_in_session", never)
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["status"] == "skipped" and out["meta"]["doc_dirty"] is True
    assert called["n"] == 0, "no session is spent"
    assert A.load_state() == {}, "and no attempt is counted against the unit"


# ── 6b. the doc bound ────────────────────────────────────────────────────────


async def test_an_oversized_diff_is_thrown_away_but_the_filings_survive(tree, monkeypatch):
    doc = _doc(tree)
    original = doc.read_text()

    def edits():
        _write_item(tree["backlog"], 900, slug="memory")
        doc.write_text("---\nsegment: architecture\n---\n\n# Memory\n" + "rewritten\n" * 60)
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn(_block(filed="#900"), edits=edits))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory", max_delta_lines=20)))
    assert doc.read_text() == original, "the doc is back to HEAD"
    assert out["status"] == "success"
    assert "cap 20" in out["meta"]["doc_update_rejected"]
    assert out["meta"]["filed"] == [900], "a rejected doc edit does not unfile the findings"
    assert _arch_events(tree)[0]["doc_update_rejected"]


async def test_gutting_a_doc_is_refused_unless_it_is_being_retired(tree, monkeypatch):
    doc = _doc(tree)
    keep = "---\nsegment: architecture\ntype: reference\ndate: 2026-09-11\n---\n\n# Memory\n"
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn(_block(status="current"), edits=lambda: doc.write_text(keep)))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert "cap 30%" in out["meta"]["doc_update_rejected"]
    assert "Body line 1." in doc.read_text()

    # The same deletion, with a status that says the body should not be trusted.
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn(_block(status="superseded"), edits=lambda: doc.write_text(keep)))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["meta"]["doc_update_rejected"] == ""
    assert doc.read_text() == keep


async def test_a_destroyed_front_matter_block_is_refused(tree, monkeypatch):
    doc = _doc(tree)
    monkeypatch.setattr(
        A, "run_prompt_in_session",
        # One line changed, so the size bounds pass and only this rule can fire.
        _turn(_block(), edits=lambda: doc.write_text(
            "+++" + doc.read_text().removeprefix("---"))))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert "front matter" in out["meta"]["doc_update_rejected"]
    assert doc.read_text().startswith("---")


# ── 6c. a group edits only its own section ───────────────────────────────────


def _section(tree, name="Dispatch"):
    return A.find_section((tree["arch"] / "workers-jobs.md").read_text(), name)


async def _run_group(tree, monkeypatch, edits, name="Dispatch", **over):
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn(_block(grouping="holds"), edits=edits))
    return await A.execute(_item(_payload(
        f"group:workers-jobs:{name}", "group", "workers-jobs", name, **over)))


async def test_a_group_edit_inside_its_section_lands(tree, monkeypatch):
    doc = tree["arch"] / "workers-jobs.md"
    out = await _run_group(tree, monkeypatch,
                           _edit(doc, "Dispatch body line one.", "Dispatch body line ONE."))
    assert out["meta"]["doc_update_rejected"] == ""
    assert out["meta"]["commit"], "and is committed"
    assert "Dispatch body line ONE." in doc.read_text()


async def test_an_edit_in_another_groups_section_is_thrown_away(tree, monkeypatch):
    """Seven groups share `autonomy-jobs.md` and four share `workers-jobs.md`.
    Without this they re-open each other's text a month apart."""
    doc = tree["arch"] / "workers-jobs.md"
    before = doc.read_text()
    out = await _run_group(tree, monkeypatch, _edit(doc, "Mining body one.", "Mining body ONE."))
    assert "outside section" in out["meta"]["doc_update_rejected"]
    assert doc.read_text() == before


async def test_editing_the_heading_line_itself_is_thrown_away(tree, monkeypatch):
    doc = tree["arch"] / "workers-jobs.md"
    out = await _run_group(tree, monkeypatch,
                           _edit(doc, "## 3. Dispatch — one door", "## 3. Dispatch — THE door"))
    assert "outside section" in out["meta"]["doc_update_rejected"]
    assert "one door" in doc.read_text()


async def test_an_insertion_at_either_end_of_the_section_is_inside_it(tree, monkeypatch):
    doc = tree["arch"] / "workers-jobs.md"
    start, end = _section(tree)

    # Immediately after the heading (old-side line == start).
    out = await _run_group(tree, monkeypatch,
                           _edit(doc, "\nDispatch body line one.", "\nA new opening line.\n\nDispatch body line one."))
    assert out["meta"]["doc_update_rejected"] == "", "an insertion at the section head is inside"
    # And at the tail (old-side line == end).
    out = await _run_group(
        tree, monkeypatch,
        _edit(doc, "Still dispatch, after a horizontal rule.",
              "Still dispatch, after a horizontal rule.\nA new closing line."))
    assert out["meta"]["doc_update_rejected"] == "", "an insertion at the section tail is inside"


def test_the_hunk_arithmetic_directly():
    """`@@ -L,0` means 'after old line L' and `@@ -L,n` covers `L..L+n-1`;
    indexing them the same way is how a legal edit gets rejected."""
    s, e = 10, 20
    assert A.hunk_inside((10, 0), s, e) and A.hunk_inside((20, 0), s, e)
    assert not A.hunk_inside((9, 0), s, e) and not A.hunk_inside((21, 0), s, e)
    assert A.hunk_inside((11, 1), s, e) and A.hunk_inside((20, 1), s, e)
    assert not A.hunk_inside((10, 1), s, e), "the heading line is never editable"
    assert not A.hunk_inside((20, 2), s, e), "a change may not run past the section"


# ── 6d. what it claims it filed ──────────────────────────────────────────────


async def test_a_filed_id_is_verified_on_disk(tree, monkeypatch):
    def edits():
        _write_item(tree["backlog"], 900, slug="memory")   # really filed, for this unit
        _write_item(tree["backlog"], 901, slug="voice")    # on disk, but another unit's
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn(_block(filed="#900, #901, #902"), edits=edits))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["meta"]["filed"] == [900]
    assert out["meta"]["filed_unverified"] == [901, 902]


async def test_an_id_below_the_floor_is_a_merge_not_a_spawn(tree, monkeypatch):
    """`backlog_write_task` answers `merged_into: #n` for a finding an open
    item already covers. That is a filing, and it is not a new item."""
    _write_item(tree["backlog"], 500, slug="memory", title="Pre existing")
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(filed="#500")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["meta"]["merged"] == [500] and out["meta"]["filed"] == []
    assert _arch_events(tree)[0]["id_floor"] == 500


async def test_appended_ids_are_recorded(tree, monkeypatch):
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(appended="#41, #42")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["meta"]["appended_to"] == [41, 42]


# ── 6e. how a turn can end ───────────────────────────────────────────────────


async def test_a_drain_skips_without_touching_anything(tree, monkeypatch):
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn("", raises=DrainActive("landing in progress")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["status"] == "skipped" and out["meta"]["drain_active"] is True
    assert A.load_state() == {}, "a drain is not the unit's failure"


async def test_a_timeout_still_runs_the_cleanup_and_counts_the_attempt(tree, monkeypatch):
    """The turn is gone; its writes are not. This is the path that most needs
    the sweep, and returning early from the exception handler would skip it."""
    doc = _doc(tree)

    def edits():
        doc.write_text(doc.read_text().replace("Body line 1.", "half an edit"))
        (tree["repo"] / "leftover.py").write_text("x = 1\n")
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn("", edits=edits, raises=TurnTimeout("exceeded 3540s")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["status"] == "failed" and out["meta"]["turn_timeout"] is True
    assert not (tree["repo"] / "leftover.py").exists(), "the stray write is still reverted"
    assert _git(tree["repo"], "status", "--porcelain") == "", "and the half edit is not committed"
    assert A.load_state()["doc:memory"]["attempts"] == 1
    assert _arch_events(tree)[0]["timed_out"] is True


async def test_an_infra_shaped_turn_costs_no_attempt(tree, monkeypatch):
    """No text and no stop reason means the harness never got a completion —
    engine unreachable, aggregator restarting. Not the unit's fault, so the
    unit stays due rather than being parked after three of them."""
    monkeypatch.setattr(A, "run_prompt_in_session", _turn("", stop_reason=None))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["status"] == "failed" and out["meta"]["infra"] is True
    assert A.load_state() == {}
    assert _arch_events(tree) == [], "an infra failure is not a review"


async def test_a_missing_section_fails_and_counts_an_attempt(tree, monkeypatch):
    """A hand-kept group list is the cost of not parsing the docs' tables, and
    this is the tell that Alan regrouped and config did not follow."""
    async def never(*a, **k):
        raise AssertionError("no session should be spent")
    monkeypatch.setattr(A, "run_prompt_in_session", never)
    out = await A.execute(_item(_payload(
        "group:workers-jobs:Gone", "group", "workers-jobs", "Gone")))
    assert out["status"] == "failed" and "section_missing" in out["summary"]
    assert A.load_state()["group:workers-jobs:Gone"]["attempts"] == 1


async def test_a_turn_with_no_parseable_verdict_still_reverts_and_records(tree, monkeypatch):
    doc = _doc(tree)
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn("I ran out of room.", stop_reason="max_turns",
                              edits=_edit(doc, "Body line 1.", "Body line one.")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["status"] == "success" and out["meta"]["parsed"] is False
    ev = _arch_events(tree)[0]
    assert ev["verdict"] is None and ev["stop_reason"] == "max_turns"


# ── 6f. committing under the loop's own locks ────────────────────────────────


async def test_a_held_automod_lock_defers_the_commit_to_the_next_tick(tree, monkeypatch):
    from scripts.automod import state as S
    doc = _doc(tree)
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn(_block(), edits=_edit(doc, "Body line 1.", "Body line one.")))
    with S.Lock(owner="a round"):
        out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["meta"]["commit"] == "" and out["meta"]["commit_deferred"]
    assert "Body line one." in doc.read_text(), "the edit is kept, dirty"
    assert A.load_state()["doc:memory"]["pending_commit"]["rel"] == "architecture/memory.md"

    # The lock is gone; the next tick owes the commit and pays it.
    state = A.load_state()
    done = A.commit_pending(state, tree["repo"])
    A.save_state(state)
    assert [d["unit"] for d in done] == ["doc:memory"]
    assert _git(tree["repo"], "log", "-1", "--format=%s").strip().startswith("arch-review:")
    assert "pending_commit" not in A.load_state()["doc:memory"]


async def test_a_landing_drain_defers_the_commit(tree, monkeypatch):
    """A commit into the promoter's idle window moves HEAD under a round that
    is mid-merge."""
    import app.routers.automod as automod_router
    doc = _doc(tree)
    monkeypatch.setattr(automod_router, "drain_active", lambda: True)
    monkeypatch.setattr(A, "run_prompt_in_session",
                        _turn(_block(), edits=_edit(doc, "Body line 1.", "Body line one.")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert "draining" in out["meta"]["commit_deferred"]
    assert A.load_state()["doc:memory"]["pending_commit"]


def test_a_deferred_commit_someone_else_resolved_is_dropped(tree):
    state = {"doc:memory": {"pending_commit": {"rel": "architecture/memory.md",
                                               "message": "arch-review: doc:memory — current"}}}
    done = A.commit_pending(state, tree["repo"])
    assert done and done[0]["note"] == "nothing to commit"
    assert "pending_commit" not in state["doc:memory"]


# ── 7. the tick ──────────────────────────────────────────────────────────────


class _Queue:
    def __init__(self, depth=None):
        self.rows: list[dict] = []
        self._depth = depth or {}

    def depth_by_source(self):
        return self._depth

    def enqueue(self, *, source, kind, payload, priority, dedup_key):
        self.rows.append({"source": source, "kind": kind, "payload": payload,
                          "priority": priority, "dedup_key": dedup_key})
        return len(self.rows)


def _cfg(**over):
    cfg = {"groups": ["workers-jobs:Dispatch", "workers-jobs:Mining"], "batch": 2,
           "daily_max": 4, "max_open_items": 25, "review_interval_days": 30,
           "retry_spacing_seconds": 21600, "max_attempts": 3, "priority": 62}
    cfg.update(over)
    return cfg


async def test_the_tick_enqueues_the_oldest_units_up_to_batch(tree):
    q = _Queue()
    await A.enqueue_if_due(q, _cfg())
    assert len(q.rows) == 2, "`batch` bounds what sits in the queue at once"
    assert q.rows[0]["dedup_key"] == "arch-review:doc:memory"
    assert q.rows[0]["payload"]["kind"] == "doc"
    assert all(r["priority"] == 62 for r in q.rows)


async def test_a_group_carries_its_name_in_the_dedup_key(tree):
    A.save_state({f"doc:{s}": {"last_reviewed_at": _iso(0)} for s in
                  ("memory", "voice", "workers-jobs")})
    q = _Queue()
    await A.enqueue_if_due(q, _cfg())
    keys = [r["dedup_key"] for r in q.rows]
    assert keys == ["arch-review:group:workers-jobs:Dispatch",
                    "arch-review:group:workers-jobs:Mining"]
    assert q.rows[0]["payload"]["name"] == "Dispatch"


async def test_a_full_queue_enqueues_nothing(tree):
    q = _Queue(depth={"arch-review": {"queued": 2}})
    await A.enqueue_if_due(q, _cfg())
    assert q.rows == []


async def test_backpressure_stops_the_pass_when_its_own_findings_pile_up(tree):
    """The R > 1 lesson, applied before it can happen: a pass that files faster
    than the board closes does not need better verdicts, it needs an edge cut."""
    for i in range(4):
        _write_item(tree["backlog"], 900 + i, slug="memory")
    _write_item(tree["backlog"], 950, slug="workers-jobs", name="Dispatch")
    _write_item(tree["backlog"], 951, slug="memory", status="done")
    q = _Queue()
    await A.enqueue_if_due(q, _cfg(max_open_items=5))
    assert q.rows == [], "five open items of both kinds reach the bound"
    q2 = _Queue()
    await A.enqueue_if_due(q2, _cfg(max_open_items=6))
    assert q2.rows, "a closed item does not count against it"


async def test_daily_max_bounds_the_day(tree):
    from scripts.automod import state as S
    for _ in range(4):
        S.append_event({"event": "arch_review", "unit": "doc:x"})
    q = _Queue()
    await A.enqueue_if_due(q, _cfg())
    assert q.rows == []


async def test_the_tick_pays_a_deferred_commit_first(tree):
    doc = _doc(tree)
    doc.write_text(doc.read_text().replace("Body line 1.", "Body line one."))
    A.save_state({"doc:memory": {"last_reviewed_at": _iso(1), "pending_commit": {
        "rel": "architecture/memory.md", "message": "arch-review: doc:memory — current"}}})
    await A.enqueue_if_due(_Queue(), _cfg())
    assert _git(tree["repo"], "log", "-1", "--format=%s").strip() == \
        "arch-review: doc:memory — current"
    assert "pending_commit" not in A.load_state()["doc:memory"]


async def test_only_open_review_items_for_this_unit_reach_the_prompt(tree, monkeypatch):
    _write_item(tree["backlog"], 900, slug="memory", title="Mine")
    _write_item(tree["backlog"], 901, slug="voice", title="Someone elses")
    _write_item(tree["backlog"], 902, slug="memory", status="done", title="Closed one")
    run = _turn(_block())
    monkeypatch.setattr(A, "run_prompt_in_session", run)
    await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert "#900 Mine" in run.prompt
    assert "Someone elses" not in run.prompt and "Closed one" not in run.prompt


# ── the two rails the pass found wrong in its own first self-review ──────────


async def test_a_group_may_not_delete_its_own_section_wholesale(tree, monkeypatch):
    """#913. The shrink denominator is the UNIT, not the file.

    A group's section is a small fraction of a jobs doc — 43 lines of 824 — so
    measuring its deletions against the whole file let it delete itself
    entirely and score 5%. The section rail does not catch it either: a hunk
    that removes the whole section is, by construction, inside the section.
    """
    doc = tree["arch"] / "workers-jobs.md"
    start, end = _section(tree)
    lines = doc.read_text().splitlines(keepends=True)
    gutted = "".join(lines[:start] + lines[end:])   # heading kept, body removed
    out = await _run_group(tree, monkeypatch, lambda: doc.write_text(gutted))
    assert "its section" in out["meta"]["doc_update_rejected"]
    assert "cap 30%" in out["meta"]["doc_update_rejected"]
    assert "Dispatch body line one." in doc.read_text(), "the section is back"


async def test_a_small_edit_inside_a_section_still_passes_the_shrink_rail(tree, monkeypatch):
    """The other side of #913: narrowing the denominator must not make an
    ordinary one-line correction unlandable."""
    doc = tree["arch"] / "workers-jobs.md"
    out = await _run_group(tree, monkeypatch,
                           _edit(doc, "Dispatch body line two.\n", ""))
    assert out["meta"]["doc_update_rejected"] == ""


def test_a_doc_units_provenance_does_not_match_its_groups_findings(tree):
    """#914. `provenance_line(slug, "")` is a strict PREFIX of every group's
    line for the same doc, and both readers matched it as a substring — so the
    doc unit saw every one of its groups' findings as its own."""
    _write_item(tree["backlog"], 900, slug="workers-jobs", name="Dispatch", title="Group find")
    _write_item(tree["backlog"], 901, slug="workers-jobs", title="Doc find")

    doc_items = [i["id"] for i in A.open_review_items("workers-jobs", "")]
    grp_items = [i["id"] for i in A.open_review_items("workers-jobs", "Dispatch")]
    assert doc_items == [901], "the doc unit sees only its own"
    assert grp_items == [900], "and the group only its own"

    assert A._filed_item_exists(900, "workers-jobs", "Dispatch")
    assert not A._filed_item_exists(900, "workers-jobs", ""), (
        "a group's item is not proof of a doc unit's claim")
    assert A._filed_item_exists(901, "workers-jobs", "")
    assert not A._filed_item_exists(901, "workers-jobs", "Dispatch")


def test_provenance_matches_a_whole_line_only(tree):
    """Prose that merely quotes the line is not a filing."""
    _write_item(tree["backlog"], 902, slug="memory",
                provenance="See also: Found reviewing architecture/memory.md in the log")
    assert not A._filed_item_exists(902, "memory")
    _write_item(tree["backlog"], 903, slug="memory")
    assert A._filed_item_exists(903, "memory")


# ── #709 / #915: what the sweep can and cannot see ──────────────────────────


def test_the_toolbox_denies_the_memory_and_fact_writers(tree):
    """#709. `DISALLOWED` subtracts from the whole chat toolbox rather than
    granting from nothing, so everything not named here is live — and the
    memory and fact surfaces were."""
    for name in ("memory_add", "memory_remove", "memory_replace",
                 "fact_add", "fact_relate", "fact_invalidate", "fact_resolve"):
        assert name in A.DISALLOWED, f"{name} can write outside the one doc"
    for name in ("memory_read", "fact_get", "fact_profile", "vault_read",
                 "vault_search", "session_recall"):
        assert name not in A.DISALLOWED, f"{name} is a reader; a review legitimately reads"


def test_denying_the_fact_writers_is_the_only_defence_not_a_second_one(tree):
    """The fact tree lives under `_pipeline/`, which is gitignored — and
    `git status` does not report ignored paths at all, with or without
    `-uall`. So `revert_strays` is structurally incapable of seeing a fact
    write: the deny list is not belt-and-braces there, it is the belt.

    Pinned as behaviour rather than stated in a comment, because the day
    someone narrows the deny list on the theory that the sweep will catch it
    is the day this matters.
    """
    repo = tree["repo"]
    (repo / ".gitignore").write_text("/_pipeline/\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "ignore the pipeline")
    before = A._porcelain(repo)

    facts = repo / "_pipeline" / "vault-derived" / "facts" / "Lloyd"
    facts.mkdir(parents=True)
    (facts / "Lloyd-state.md").write_text("- Lloyd believes something new.\n")

    after = A._porcelain(repo)
    assert after == before, "an ignored path is invisible to the sweep, by design"
    assert A.revert_strays(repo, before, after) == []
    assert (facts / "Lloyd-state.md").exists(), "so nothing reverts it"


def test_the_sweep_covers_the_whole_vault_not_two_directories(tree, monkeypatch):
    """#709. It was an allowlist of the two directories the prompt mentions,
    which left `lloyd/MEMORY.md`, `knowledge/` and the rest unwatched."""
    vault = tree["vault"]
    (vault / "lloyd").mkdir()
    (vault / "knowledge").mkdir()

    def edits():
        (vault / "lloyd" / "MEMORY.md").write_text("- a belief the review invented\n")
        (vault / "knowledge" / "note.md").write_text("# invented\n")
        (tree["backlog"] / "900-a-real-finding.md").write_text("---\nid: 900\n---\n# filed\n")
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(), edits=edits))
    out = await_sync(A.execute(_item(_payload("doc:memory", "doc", "memory"))))

    assert not (vault / "lloyd" / "MEMORY.md").exists()
    assert not (vault / "knowledge" / "note.md").exists()
    assert (tree["backlog"] / "900-a-real-finding.md").exists(), "backlog/ is the job"
    swept = {s["path"] for s in out["meta"]["stray_writes"]}
    assert swept == {"lloyd/MEMORY.md", "knowledge/note.md"}


def test_the_exempt_prefix_names_what_is_skipped_not_what_is_guarded(tree):
    """A directory added to the vault next month is swept by default rather
    than silently not being."""
    assert A.VAULT_UNSWEPT_PREFIXES == ("backlog/",)
    vault = tree["vault"]
    (vault / "brand-new-area").mkdir()
    (vault / "brand-new-area" / "x.md").write_text("new\n")
    assert "brand-new-area/x.md" in A.vault_dirty(vault)
    (tree["backlog"] / "1-x.md").write_text("filed\n")
    assert not any(p.startswith("backlog/") for p in A.vault_dirty(vault))


async def test_a_rewrite_of_an_already_dirty_path_is_reported_not_reverted(tree, monkeypatch):
    """#915. A path already dirty cannot appear in `after - before` however
    much the turn rewrote it, and the vault's `autonomy/*.md` files are dirty
    on nearly every run because the scheduler rewrites one each time.

    Reported, never reverted: somebody else is mid-edit on that path, and
    restoring it would destroy their uncommitted work to undo ours.
    """
    task = tree["vault"] / "autonomy" / "39-nightly.md"
    task.write_text("---\nstatus: up_next\n---\n# scheduled\n")   # dirty BEFORE the turn
    human = tree["repo"] / "README.md"
    human.write_text("a human is mid-edit\n")

    def edits():
        task.write_text("---\nstatus: paused\n---\n# the review edited this\n")
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(updated="no"), edits=edits))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))

    reported = {s["path"]: s["action"] for s in out["meta"]["stray_writes"]}
    assert "autonomy/39-nightly.md" in reported
    assert "already dirty" in reported["autonomy/39-nightly.md"]
    assert "the review edited this" in task.read_text(), "not reverted"
    assert human.read_text() == "a human is mid-edit\n", "and untouched paths are silent"
    assert "README.md" not in reported


async def test_an_untouched_dirty_path_is_not_reported(tree, monkeypatch):
    """The vault is never clean. Reporting every pre-existing dirty path would
    make the field noise and nobody would read it."""
    (tree["vault"] / "autonomy" / "39-nightly.md").write_text("---\nstatus: up_next\n---\n")
    (tree["repo"] / "README.md").write_text("mid-edit\n")
    monkeypatch.setattr(A, "run_prompt_in_session", _turn(_block(updated="no")))
    out = await A.execute(_item(_payload("doc:memory", "doc", "memory")))
    assert out["meta"]["stray_writes"] == []


def test_fingerprints_skips_what_it_cannot_read(tree):
    fp = A.fingerprints(tree["repo"], ["README.md", "does-not-exist.md", "../escape.md"])
    assert set(fp) == {"README.md"}
