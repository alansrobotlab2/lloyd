"""The #724 dispatch-field rail on the `vault_write` lane (#2362).

The rail was written for the tool that asks (#724, `app/harness/policy.py`: an
unattended `autonomy_write_task` may not move a field the scheduler dispatches
on), and #2190 extended it to the vault round's landing route
(`scripts/automod/vault_round.py`). The third lane onto the same files was left
open, and it is the one with nothing in front of it: `vault_write` is tier 1 and
consults only the protected-path deny-set, whose five entries
(`app/harness/protected_paths.py`) do not include `~/obsidian/autonomy/` — the
directory `app.autonomy.AUTONOMY_DIR` points the scheduler at.

Not hypothetical. On 2026-10-05 a turn created `autonomy/96-djev-name-prior-probe.md`
with `status: up_next` through `vault_write` at 06:27:38.866Z
(`~/obsidian/memory/audit/writes.jsonl`, `pre_sha256: ""` = a create);
`run_scheduled-task_20261005_062825_b3ceb1` started 47 seconds later and
succeeded. The same turn's `autonomy_write_task` against the same target was
refused at 06:38:16Z (`~/lloyd-data/safety/denials.jsonl`: "would change
dispatch-affecting field(s) [status] on target #96"). Armed through the open
door, refused through the closed one. 17 of the 2514 audit rows are
`vault_write` onto `autonomy/`, across 7 sessions.

So these tests drive the real handler against a scratch vault, and they check
both halves of a guard worth having: the move is refused with its evidence, and
the benign write still lands. A rail that refuses everything is not a rail, and
a rail that refuses in only one direction is a surprise its users discover.
"""
from __future__ import annotations

import ast
import json
import re
import types
from pathlib import Path

import pytest

from app.harness import policy as P

#: The front matter of a real task file, in the shape the files under
#: `~/obsidian/autonomy/` carry it. `description` and the Activity Log body are
#: here because clause 3 is about them staying editable when the fields the
#: scheduler dispatches on are not.
BASE_FM = {
    "name": "djev name-prior probe",
    "description": "run the weekly probe",
    "status": "draft",
    "priority": "medium",
    "frequency": "weekly",
    "skill_name": "djev-name-prior-probe",
    "agent_id": "worker",
}

TASK_PATH = "autonomy/96-djev-name-prior-probe.md"

#: The genuine journal writer, captured at import time because the `scratch`
#: fixture replaces `denial_journal.record` with a list-appending spy for every
#: test. The node that reads the journal FILE back needs the real writer, and an
#: import inside that node would import the spy.
from app.harness.denial_journal import record as _REAL_DENIAL_RECORDER  # noqa: E402

#: The module under test, parsed by the placement test. `tests/test_memory_writer_
#: lane.py` owns the same root constant; importing it from there would make this
#: file fail to collect for a reason that has nothing to do with vault_write.
_VAULT_PY = Path(__file__).resolve().parents[1] / "agent_mcp" / "vault.py"


def _task(**overrides) -> str:
    """One autonomy task file, with front matter overridden field by field."""
    fm = dict(BASE_FM)
    fm.update({k: v for k, v in overrides.items() if v is not None})
    head = "\n".join(f"{k}: {v}" for k, v in fm.items())
    return ("---\n" + head + "\n---\n\n## Activity Log\n\n"
            "- 2026-10-05T06:27Z created by an autocode round\n")


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """`vault_write` aimed at a scratch tree, with its audit log and journal here.

    `VAULT` is imported into `agent_mcp.vault`'s namespace, so that is the name
    to patch — patching `agent_mcp._shared` would leave the handler writing into
    ~/obsidian, which is exactly the tree this item exists to protect.
    """
    import agent_mcp.vault as V
    from app.harness import denial_journal

    monkeypatch.setattr(V, "VAULT", tmp_path)
    monkeypatch.setattr(V, "AUDIT_LOG_DIR", tmp_path / "audit")
    monkeypatch.setattr(V, "AUDIT_LOG_FILE", tmp_path / "audit" / "writes.jsonl")
    journal: list[dict] = []
    # Plain append, no `or True`: a spy that returns a truthy value reads like an
    # assertion that cannot fail, and this one's whole job is to be believed.
    monkeypatch.setattr(denial_journal, "record", lambda **kw: journal.append(kw))
    return types.SimpleNamespace(root=tmp_path, mod=V, journal=journal)


def _write(scratch, rel: str, content: str) -> dict:
    return scratch.mod._vault_write({"path": rel, "content": content})


def _audit_rows(scratch) -> list[dict]:
    log = scratch.root / "audit" / "writes.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _seed(scratch, content: str, rel: str = TASK_PATH) -> str:
    """Land a task file through the lane itself, so the test starts from real bytes."""
    assert _write(scratch, rel, content).get("success") is True, "seed write refused"
    return (scratch.root / rel).read_text(encoding="utf-8")


def _seed_armed(scratch, content: str, rel: str = TASK_PATH) -> str:
    """Put an ARMED task file on disk, bypassing the lane under test.

    Necessary, not convenient: this lane refuses an `up_next` create by design,
    so the file clause 3 is about — one whose `status` already reads `up_next` —
    can only arrive the way an armed task really arrives, through a granted call
    or a human edit. A test that seeded it through the guarded door would be
    asserting the refusal it is meant to test around.
    """
    target = scratch.root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return content


def test_moving_a_dispatch_field_on_an_existing_task_is_refused(scratch):
    """Clause 1: the refusal names the path, the field, `old -> new` and the rail.

    The 2026-10-05 write was a create, so it also proves the lane reads the
    on-disk pre-image: `draft -> up_next` on a file that already exists is the
    same move the #724 rail refused for the tool call, spelled one tool later.
    """
    before = _seed(scratch, _task(status="draft"))
    rows = len(_audit_rows(scratch))

    result = _write(scratch, TASK_PATH, _task(status="up_next"))

    assert "error" in result and result.get("success") is not True, (
        f"a dispatch-affecting move was accepted: {result}")
    msg = result["error"]
    assert TASK_PATH in msg, msg
    assert "`status`" in msg and "draft" in msg and "up_next" in msg, msg
    assert "#724" in msg, f"a refusal must name the rail it defers to: {msg}"
    assert (scratch.root / TASK_PATH).read_text(encoding="utf-8") == before, (
        "the refusal wrote bytes anyway")
    assert len(_audit_rows(scratch)) == rows, "a refused write appended an audit row"
    assert scratch.journal and scratch.journal[-1]["excerpt"] == TASK_PATH, (
        "a refusal nobody can find after the fact is a refusal nobody learns from")


def test_creating_a_task_up_next_is_refused_while_the_same_draft_create_lands(scratch):
    """Clause 2: a create is judged only on whether it arms.

    The left half is the exact 2026-10-05 call, reproduced from the audit row's
    shape: a new task file whose `status` is `up_next` dispatches the moment it
    lands, so the rail has to answer at create time — there is no pre-image to
    diff. The right half is why the answer is not "refuse creates": the nightly
    chain's documented hand-down writes a task with `skill_name` and `frequency`
    set and `status: draft`, and a guard that stopped that would stop the chain.
    """
    armed = _task(status="up_next")
    result = _write(scratch, TASK_PATH, armed)

    assert "error" in result and result.get("success") is not True, (
        f"creating `status: up_next` was accepted: {result}")
    assert "up_next" in result["error"] and "#724" in result["error"], result["error"]
    assert not (scratch.root / TASK_PATH).exists(), "refused, but the file is there"
    assert _audit_rows(scratch) == [], "a refused create appended an audit row"

    accepted = _write(scratch, TASK_PATH,
                      _task(status="draft", frequency="weekly",
                            skill_name="djev-name-prior-probe"))
    assert accepted.get("success") is True, (
        f"the sanctioned draft create was refused: {accepted}")
    assert (scratch.root / TASK_PATH).is_file()
    assert [r["path"] for r in _audit_rows(scratch)] == [TASK_PATH]


def test_an_edit_that_moves_no_dispatch_field_lands_on_an_armed_task(scratch):
    """Clause 3: `status: up_next` already on disk is not a reason to refuse.

    Three benign writes against an armed task, plus the one the #2190
    owed-after-landing measurement exists to catch: PyYAML resolves an unquoted
    `scheduled_at` to a `datetime`, so re-quoting it is a reformat and a rail
    that byte-diffs would refuse the nightly writer's own rewrites. The control
    at the end is what makes the four accepts above mean something: the same
    file, same bytes, one moved field, refused.
    """
    armed = _task(status="up_next", scheduled_at="2026-10-09T00:00:00+00:00")
    _seed_armed(scratch, armed)

    body = armed.replace("- 2026-10-05T06:27Z created by an autocode round",
                         "- 2026-10-05T06:27Z created by an autocode round\n"
                         "- 2026-10-07T16:40Z probed; 3 divergences in 24 h\n")
    assert _write(scratch, TASK_PATH, body).get("success") is True, "body edit refused"

    # `description=`, not a new key: the clause is about the field the file
    # already carries being editable, so the write has to edit THAT key.
    described = _task(status="up_next",
                      description="run the weekly probe and log the count",
                      scheduled_at="2026-10-09T00:00:00+00:00")
    assert "desc:" not in described, "the fixture wrote a new key, not the field"
    assert described.count("description:") == 1
    assert _write(scratch, TASK_PATH, described).get("success") is True, (
        "a description edit refused")
    assert "log the count" in (scratch.root / TASK_PATH).read_text(encoding="utf-8"), (
        "the accept was reported without the new description reaching the file")

    requoted = described.replace("scheduled_at: 2026-10-09T00:00:00+00:00",
                                 "scheduled_at: '2026-10-09T00:00:00+00:00'")
    assert requoted != described, "the re-quote changed nothing, so it proves nothing"
    assert _write(scratch, TASK_PATH, requoted).get("success") is True, (
        "a re-quoted scheduled_at was treated as a schedule change")
    old = P.front_matter_map(described)["scheduled_at"]
    assert not isinstance(old, str), (
        "the mechanism this protects is gone: the loader no longer resolves an "
        "unquoted timestamp to a date, so this case cannot recur")

    # The control that makes the four accepts above mean something: a field the
    # file already carries, rewritten unchanged and accepted, then moved by one
    # digit and refused. Adding the field where there was none would also be
    # refused — absent on one side is a move, the rule #2190 shipped — so the
    # accept half has to be a value that is genuinely already there.
    with_dependency = requoted.replace("agent_id: worker",
                                       "agent_id: worker\ndepends_on: 12")
    _seed_armed(scratch, with_dependency)
    assert _write(scratch, TASK_PATH, with_dependency.replace(
        "run the weekly probe", "run the weekly probe now")).get("success") is True, (
        "a description edit next to an unchanged depends_on was refused")

    moved = with_dependency.replace("depends_on: 12", "depends_on: 34")
    refused = _write(scratch, TASK_PATH, moved)
    assert "error" in refused, f"a moved depends_on was accepted: {refused}"
    assert "`depends_on`" in refused["error"] and "12 -> 34" in refused["error"], refused


#: Realistic old -> new pairs for the fields the rail names today, so a refusal
#: is quoted with values a reader recognises. A field added to
#: `policy.SCHEDULE_STATE_FIELDS` later falls through to the generic pair and is
#: still refused — which is the whole point of parametrising over the frozenset
#: instead of over a list this file maintains.
PAIRS = {
    "status": ("draft", "up_next"),
    "frequency": ("weekly", "daily"),
    "skill_name": ("djev-name-prior-probe", "queue-health-check"),
    "scheduled_at": ("2026-10-09", "2026-10-12"),
    "depends_on": ("12", "34"),
    "auto_advance": ("true", "false"),
    "preferred_hours": ("09-17", "00-23"),
}


@pytest.mark.parametrize("field", sorted(P.SCHEDULE_STATE_FIELDS))
def test_every_field_in_the_rail_is_guarded_by_this_lane(scratch, field):
    """Clause 4: the refused set is `policy.SCHEDULE_STATE_FIELDS`, not a copy.

    Parametrised over the frozenset itself, so the day a field is added to the
    rail this test gains a case and `agent_mcp/vault.py` does not need an edit.
    `test_a_field_added_to_the_rail_is_guarded_without_editing_this_lane` is the
    same claim executed rather than asserted, and the two together are what make
    a hand-copied list in this lane impossible to reintroduce quietly.
    """
    old, new = PAIRS.get(field, ("alpha-value", "beta-value"))
    _seed(scratch, _task(**{field: old}))

    result = _write(scratch, TASK_PATH, _task(**{field: new}))

    assert "error" in result, (
        f"`{field}` moved and this lane did not refuse it: {result}")
    msg = result["error"]
    assert f"`{field}`" in msg, f"the refusal does not name the field it refused: {msg}"
    assert f"{old} -> {new}" in msg or (old in msg and new in msg), msg
    assert "#724" in msg, msg


def test_a_field_added_to_the_rail_is_guarded_without_editing_this_lane(scratch,
                                                                       monkeypatch):
    """Clause 4 executed: widen the rail, and the lane widens with it.

    The parametrised test above can only show the fields that exist today. This
    one adds a field that exists nowhere, moves it through the real handler, and
    is refused — which is only possible if the lane reads the frozenset at call
    time, the way `autonomy_write_task` and the vault round do. A lane that kept
    its own tuple would pass the parametrised test and fail this one.
    """
    _seed(scratch, _task(review_gate="off"))

    monkeypatch.setattr(P, "SCHEDULE_STATE_FIELDS",
                        set(P.SCHEDULE_STATE_FIELDS) | {"review_gate"})
    result = _write(scratch, TASK_PATH, _task(review_gate="on"))

    assert "error" in result, (
        "the lane refused a known field only because it knew it — it is not "
        f"reading the rail: {result}")
    assert "`review_gate`" in result["error"], result["error"]


def test_front_matter_this_lane_cannot_read_is_a_refusal_not_a_pass(scratch):
    """Clause 4's fail-closed half: no mapping, no write.

    A guard that skips what it cannot parse is a guard that can be switched off
    with a missing `---`: the recovery parser is the scheduler's own, so a file
    this lane cannot read may still be a file the scheduler dispatches. Three
    shapes, all refused — a create with no front matter at all, a create whose
    front matter never closes, and a benign-looking body edit onto a task file
    whose ON-DISK front matter is broken.
    """
    refused = _write(scratch, TASK_PATH, "# just prose, no front matter\n")
    assert "error" in refused, f"a create with no front matter was accepted: {refused}"
    assert "front matter" in refused["error"], refused["error"]
    assert not (scratch.root / TASK_PATH).exists()

    broken = "---\nstatus: draft\nfrequency: weekly\n"
    assert P.front_matter_map(broken) is None, "the fixture is not unparseable"
    refused = _write(scratch, "autonomy/97-open-front-matter.md", broken)
    assert "error" in refused, f"unclosed front matter was accepted: {refused}"

    (scratch.root / "autonomy").mkdir(parents=True, exist_ok=True)
    (scratch.root / "autonomy" / "98-corrupt.md").write_text(broken, encoding="utf-8")
    refused = _write(scratch, "autonomy/98-corrupt.md", broken + "\n# more prose\n")
    assert "error" in refused, (
        "an edit to an unreadable task file passed a rail that cannot read it")
    assert "#724" not in refused["error"], (
        "this refusal is a fail-closed, not a field move: " + refused["error"])


def test_an_unloadable_rail_stops_the_write(scratch, monkeypatch):
    """Clause 4's other fail-closed half: no rail, no write.

    `sys.modules[...] = None` is the import machinery's way of making an import
    raise, which is what a broken install or a circular import does to a
    function-local import. The refusal here must come from the guard, not from a
    traceback that the caller's `except Exception` would report as an internal
    error with no instruction.
    """
    import sys

    _seed(scratch, _task(status="draft"))
    monkeypatch.setitem(sys.modules, "app.harness.policy", None)

    result = _write(scratch, TASK_PATH, _task(status="up_next"))

    assert "error" in result, "a write went through with no rail to check it"
    assert "rail" in result["error"], result["error"]
    assert (scratch.root / TASK_PATH).read_text(encoding="utf-8").count("draft") == 1


def test_a_backlog_status_move_is_not_this_lane_s_business(scratch):
    """The prefix test is a scope, not a shield: `status` is a backlog field too.

    `status` means something entirely different on a backlog item, and this
    board's own writers move it constantly through `vault_write`. A lane that
    refused on the field name alone would stop legitimate work and would not be
    the #724 rule, which is about the directory the scheduler reads.
    """
    item = "backlog/96-something.md"
    _seed(scratch, "---\ntype: note\nstatus: draft\n---\n# Item\n", rel=item)

    assert _write(scratch, item,
                  "---\ntype: note\nstatus: up_next\n---\n# Item\n").get(
        "success") is True, "the rail reached outside autonomy/ and refused a board move"


def test_the_lane_does_not_refuse_what_the_tool_would_allow(scratch):
    """A create carrying a dispatch field at its CURRENT value is not a move.

    #724's tool-side rule denies an `autonomy_write_task` even when it resends
    the value a task already has, because that call never reads the disk; the
    file-side lanes can read it, and #2190 chose "does the value MOVE" for
    exactly that reason. This is the difference between the two rules, pinned so
    a later hand "harmonises" them the wrong way: rewriting an armed file with
    its own `status` is the nightly writer's ordinary reformat and lands.
    """
    armed = _task(status="up_next", frequency="daily")
    _seed_armed(scratch, armed)

    assert _write(scratch, TASK_PATH, armed).get("success") is True, (
        "a resend of an armed task's own front matter was refused, which is the "
        "tool-side rule, not the file-side one")


def _stmts_with_call(node, name: str) -> list[tuple[int, ast.stmt]]:
    """(index, statement) for each statement of `node`'s body holding `name()`.

    Position, not just presence: this is about the ORDER of three calls in one
    block, and a walk that returns nodes loses exactly that.
    """
    out = []
    for i, stmt in enumerate(node.body):
        for c in ast.walk(stmt):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) \
                    and c.func.id == name:
                out.append((i, stmt))
                break
    return out


def test_the_diff_rule_runs_inside_the_writer_s_own_critical_section():
    """The guard's placement is part of the guard, so it is parsed, not eyeballed.

    `test_memory_writer_lane.py::test_commit_site_writes_atomically_under_the_shared_
    lock` forbids reading the target outside `commit_lock`, and `_vault_write` takes
    its pre-image only inside that lock — which is why this lane is split in two:
    the create rule, which has no pre-image to read, is a pre-lock guard; the diff
    rule is not. A diff moved back out of the block would restore the race the
    writer-lane guard exists to stop — two writers each comparing against bytes
    neither of them is still about to replace — and a diff placed after the write
    would only ever describe bytes that are already gone.

    The failure this node caught while #2362 was being written: an insertion that
    dedented the commit block left `write_text_durable` stranded under the refusal's
    `return`, so every `vault_write` in the process reported `success` having written
    nothing. Every refusal test still passed — a write that never happened certainly
    did not move a field — and so did the no-audit-row assertion. Only the shape of
    the block tells a guard from a sabotage, so the shape is asserted.
    """
    tree = ast.parse((_VAULT_PY).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_vault_write")

    def named(node, name: str) -> list[ast.Call]:
        return [c for c in ast.walk(node) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name) and c.func.id == name]

    locked = [w for w in ast.walk(fn) if isinstance(w, ast.With)
              and named(w.items[0].context_expr, "commit_lock")]
    assert locked, "_vault_write no longer commits under commit_lock()"
    guarded = [w for w in locked if named(w, "_autonomy_schedule_move_refusal")]
    assert guarded, (
        "the dispatch-field diff left the critical section, so it would compare "
        "against bytes another writer can replace before this one does")
    block = guarded[0]

    diff = _stmts_with_call(block, "_autonomy_schedule_move_refusal")
    write = _stmts_with_call(block, "write_text_durable")
    assert diff and write, "the guarded block must hold both the diff and the write"
    assert diff[0][0] < write[0][0], (
        "the diff runs after the write, so it can only describe bytes already gone")

    between = block.body[diff[0][0] + 1:write[0][0]]
    assert _returns_on_refusal(between), (
        "the refusal is not returned before the write, so a refused move would "
        "write anyway — or the write is unreachable")

    # The create rule stays OUT of the lock: it reads no pre-image, so locking
    # around it buys nothing and lengthens the critical section for everyone else.
    assert named(fn, "_autonomy_schedule_refusal"), (
        "the create rule vanished, so an `up_next` create dispatches again")
    assert not any(named(w, "_autonomy_schedule_refusal") for w in locked), (
        "the create rule moved inside the lock, where it guards nothing")


def _returns_on_refusal(stmts) -> bool:
    """True when one of `stmts` is `if refusal is not None: return refusal`."""
    return any(isinstance(s, ast.If)
               and isinstance(s.test, ast.Compare)
               and isinstance(s.test.left, ast.Name) and s.test.left.id == "refusal"
               and any(isinstance(n, ast.Return) for n in ast.walk(s))
               for s in stmts)


def test_a_refused_write_records_no_change_ledger_entry(scratch, monkeypatch):
    """No audit row and no ledger entry: a refusal must leave the vault untouched.

    The ledger is what makes a vault change undoable, so an entry written for a
    write that did not happen would be an undo record pointing at nothing — and
    the next reader of `vault_changes` would count a change that never occurred.
    Spied rather than read back, because the ledger's own storage is off-tree and
    a scratch vault has no git history to inspect.
    """
    recorded: list[str] = []
    monkeypatch.setattr(scratch.mod, "_ledger_record",
                        lambda *a, **k: recorded.append(str(k.get("path") or a[:1])))

    _seed(scratch, _task(status="draft"))
    recorded.clear()
    refused = _write(scratch, TASK_PATH, _task(status="up_next"))

    assert "error" in refused, refused
    assert recorded == [], (
        f"a refused write still opened a ledger entry: {recorded}")

    assert _write(scratch, TASK_PATH,
                  _task(status="draft",
                        description="run the weekly probe now")).get("success") is True
    assert len(recorded) == 1, (
        "the spy saw nothing even on an accepted write, so the assertion above "
        "would pass on a broken spy: " + str(recorded))


def test_an_up_next_create_is_refused_wherever_the_directory_already_sits(scratch,
                                                                          tmp_path):
    """A create is refused in all three shapes a create arrives in.

    Two of them are easy to miss and one of them was the review's own probe: a
    write whose `autonomy/` directory does NOT exist yet, and a create under a
    subdirectory of the task dir. Read that half honestly: the scheduler's own scan
    is `AUTONOMY_DIR.glob("*.md")`, non-recursive (`app/autonomy.py:101`, which also
    skips a name not matching `\\d+-`), so a file in a subdir is not dispatched
    today — `architecture/autonomy.md:92` records that the walk is never `rglob`,
    and that IS how retirement works: 14 retired task files sit in
    `autonomy/_archived/` invisible to the walk while still carrying
    `status: up_next` on disk (`_archived/25-memory-capture.md:27`). So this refusal
    makes no claim about what the scheduler reads. It is the other half of that
    mechanism: restoring one of those tasks is a `vault_write` of the same bytes one
    level up, retirement changed no status field, and an `up_next` create at
    `autonomy/25-*.md` is therefore exactly how a retired armed task comes back — a
    dispatch decision. That, not a directory listing, is why the rule is a prefix
    over the subtree rather than a name list over the live directory.

    Both must be refused, and `_vault_write`'s `mkdir` is what makes the first one
    look suspicious: the guard runs before it, so on that path the refusal returns
    first and the directory is never created — which is also asserted, because the
    docstring claims it.
    """
    fresh = tmp_path / "no-autonomy-yet"
    fresh.mkdir()
    scratch.mod.VAULT = fresh

    refused = _write(scratch, "autonomy/99-never-existed.md", _task(status="up_next"))
    assert "error" in refused, f"an armed create slipped past an absent directory: {refused}"
    assert "up_next" in refused["error"] and "#724" in refused["error"], refused["error"]
    assert not (fresh / "autonomy/99-never-existed.md").exists()
    assert not (fresh / "autonomy").exists(), (
        "the guard runs before `_vault_write`'s mkdir, so a refusal on this path "
        "must not leave a directory behind — the docstring says so")

    scratch.mod.VAULT = tmp_path
    sub = "autonomy/dream/99-armed-in-a-category.md"
    refused = _write(scratch, sub, _task(status="up_next"))
    assert "error" in refused, f"an armed create under a category dir was accepted: {refused}"
    assert not (tmp_path / sub).exists()

    benign = _write(scratch, sub, _task(status="draft"))
    assert benign.get("success") is True, (
        f"the same create as a draft was refused, so the rule is not about arming: {benign}")
    assert (tmp_path / sub).is_file(), "reported success without writing the file"



# ── the review's finding: a path gate is a rail only over the spellings it reads ─
#
# The first round gated both phases on `path.startswith("autonomy/")`, a plain
# string prefix over whatever `_normalize_vault_path` returned — and that function
# tested `Path(p).parts` for traversal while returning the caller's STRING, dot
# segments and all. `Path("./autonomy/x.md").parts` is `('autonomy', 'x.md')`, so
# the traversal check passed, `"./autonomy/x.md".startswith("autonomy/")` is False,
# both phases returned None, and `VAULT / "./autonomy/x.md"` is the very same file.
# One leading `./` reached the live task file with the rail skipped: the exact
# failure this item exists to close, reproduced inside the guard written to close
# it. The same spelling also hid the row from the grep the owed-after-landing check
# on this item re-runs (`'"path": "autonomy/'`).
#
# The fix is at the root, not at the gate: the normalizer now reduces the path it
# hands back, so every consumer — both phases, the two `knowledge/` OKF guards, the
# audit row, the success echo — reads one canonical spelling. These nodes pin the
# property, the shape, and the two seams the review listed as unverified.

#: The six spellings of one task file that reach it through `vault_write` today.
#: The `$VAULT` form is expanded per test: `VAULT` is patched per-scratch, so this
#: list cannot hold the absolute string itself.
TASK_FILE_SPELLINGS = (
    pytest.param("autonomy/96-djev-name-prior-probe.md", id="vault-relative"),
    pytest.param("./autonomy/96-djev-name-prior-probe.md", id="one-dot-segment"),
    pytest.param("././autonomy/96-djev-name-prior-probe.md", id="two-dot-segments"),
    pytest.param(".//autonomy//96-djev-name-prior-probe.md", id="dot-and-empty-segment"),
    pytest.param("~/obsidian/autonomy/96-djev-name-prior-probe.md", id="home-prefixed"),
    pytest.param("$VAULT/autonomy/96-djev-name-prior-probe.md", id="absolute-in-vault"),
)


def _spelled(scratch, spelling: str) -> str:
    return spelling.replace("$VAULT", str(scratch.mod.VAULT))


@pytest.mark.parametrize("spelling", TASK_FILE_SPELLINGS)
def test_every_spelling_of_the_task_file_is_refused(scratch, spelling):
    """clause 1 across the path seam: the gate is on the file, not on the string.

    The grader that sent the first round back named the bypass in one line:
    `VAULT / './autonomy/x.md'` and `VAULT / 'autonomy/x.md'` are one file, so a
    rail that tests the spelling refused one and wrote the other. Each spelling
    here moves `status` on the SAME existing file, and every one must answer with
    the #724 refusal naming path, field and `old -> new`, leave the bytes alone and
    append no audit row.
    """
    before = _seed_armed(scratch, _task(status="up_next"))

    out = _write(scratch, _spelled(scratch, spelling), _task(status="draft"))
    assert "error" in out, (
        f"{spelling!r} is the same file as {TASK_PATH!r} and it moved `status`: "
        f"the rail is a string prefix again — {out}")
    msg = out["error"]
    assert TASK_PATH in msg, f"the refusal must name the canonical path: {msg}"
    assert "`status`" in msg, msg
    assert "up_next" in msg and "draft" in msg, msg
    assert "#724" in msg, msg
    assert (scratch.root / TASK_PATH).read_text(encoding="utf-8") == before, (
        f"a refusal reached through {spelling!r} wrote bytes anyway")
    assert _audit_rows(scratch) == [], (
        f"a refusal reached through {spelling!r} appended an audit row")


@pytest.mark.parametrize("spelling", TASK_FILE_SPELLINGS)
def test_an_armed_create_is_refused_under_every_spelling(scratch, spelling):
    """clause 2 across the same seam, on the shape that actually walked the door.

    The 2026-10-05 event was a create, and a create has no pre-image to diff, so
    phase A is the whole guard for it — which makes it the phase a spelling could
    most easily skip. The refusal must also precede `_vault_write`'s `mkdir`, or a
    refusal still creates the tree it refused.
    """
    out = _write(scratch, _spelled(scratch, spelling), _task(status="up_next"))
    assert "error" in out, f"an armed create slipped through as {spelling!r}: {out}"
    assert "up_next" in out["error"] and "#724" in out["error"], out["error"]
    assert not (scratch.root / TASK_PATH).exists(), (
        f"a refused armed create still wrote {TASK_PATH!r} reached as {spelling!r}")
    assert not (scratch.root / "autonomy").exists(), (
        f"the refusal for {spelling!r} left the directory mkdir would have made")
    assert _audit_rows(scratch) == []


def test_the_normalizer_hands_every_guard_one_spelling_of_a_path(scratch):
    """The root of the finding: the reducer reduced nothing.

    `_normalize_vault_path`'s docstring says it reduces a caller-supplied path to a
    vault-relative POSIX path, and its own escape test reads `Path(p).parts` — where
    a `./` is already gone — while the value it returned kept the caller's dots.
    Six spellings of one task file come back as one string, with a positive control
    on ordinary paths (so this is a reduction, not a rewrite of the resolver), and
    the shapes it already refused still refuse for the same reason.
    """
    ok = scratch.mod._normalize_vault_path
    for given, want in (
        ("autonomy/96-x.md", "autonomy/96-x.md"),
        ("./autonomy/96-x.md", "autonomy/96-x.md"),
        ("././autonomy/96-x.md", "autonomy/96-x.md"),
        (".//autonomy//96-x.md", "autonomy/96-x.md"),
        ("~/obsidian/autonomy/96-x.md", "autonomy/96-x.md"),
        (f"{scratch.mod.VAULT}/autonomy/96-x.md", "autonomy/96-x.md"),
        ("knowledge/agents/foo.md", "knowledge/agents/foo.md"),
        ("memory/learnings/DAILY_NOTES.md", "memory/learnings/DAILY_NOTES.md"),
        ("./knowledge/agents/foo.md", "knowledge/agents/foo.md"),
    ):
        path, err = ok(given)
        assert err is None, f"{given!r} was refused: {err}"
        assert path == want, f"{given!r} reduced to {path!r}, not {want!r}"

    for given, why in (
        ("~/lloyd/notes/x.md", "an absolute path under another root"),
        ("../etc/passwd", "a traversal out of the vault"),
        ("autonomy/../../etc/passwd", "a traversal dressed as a task file"),
        (".", "the vault root itself"),
        ("./.", "the vault root dressed in a dot segment"),
        ("", "nothing at all"),
    ):
        path, err = ok(given)
        assert path is None and err, f"{why} was accepted as {given!r}: {path!r}"


def test_a_benign_write_through_a_dot_segment_lands_and_is_ledgered_canonically(scratch):
    """The half that keeps the fix honest: reducing a path is not refusing it.

    A `./autonomy/x.md` create that arms nothing is an ordinary write. It must
    succeed, land on the canonical file, and be audited under that canonical path:
    the owed-after-landing check on this item counts `'"path": "autonomy/'` rows in
    `~/obsidian/memory/audit/writes.jsonl`, so a benign write recorded as
    `"./autonomy/…"` is a row that check cannot see, and a reduction that only
    served the guard would leave the audit half of the hole open.
    """
    out = _write(scratch, "./autonomy/97-benign-through-a-dot.md",
                 _task(status="draft"))
    assert out.get("success") is True, out
    assert out["path"] == "autonomy/97-benign-through-a-dot.md", out
    assert (scratch.root / "autonomy/97-benign-through-a-dot.md").is_file(), (
        "reported success without writing the canonical file")
    assert [r["path"] for r in _audit_rows(scratch)] == [
        "autonomy/97-benign-through-a-dot.md"], _audit_rows(scratch)


def test_the_refusal_reaches_the_denial_journal_file_itself(scratch, monkeypatch,
                                                            tmp_path):
    """The seam `_record_schedule_refusal` → `denial_journal.record` → the file.

    Every other node here spies on `record`, which proves the call was made and
    nothing about the row. The journal is the only surface that counts this rail's
    refusals — `~/lloyd-data/safety/denials.jsonl`, the file holding 2026-10-05's
    row 75, which refused through `autonomy_write_task` the move this door let
    through a minute earlier — so a row that never reaches it is a rail that
    reports zero refusals while firing. `LLOYD_DENIAL_JOURNAL` is the writer's own
    override, and the fixture's spy is put back for the duration: the real writer,
    pointed at a scratch file.
    """
    import json as _json

    from app.harness import denial_journal

    journal = tmp_path / "denials.jsonl"
    monkeypatch.setenv(denial_journal.JOURNAL_ENV, str(journal))
    monkeypatch.setattr(denial_journal, "record", _REAL_DENIAL_RECORDER)

    _seed_armed(scratch, _task(status="up_next"))
    out = _write(scratch, TASK_PATH, _task(status="draft"))
    assert "error" in out, out

    rows = [_json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["guard"] == "autonomy_schedule_rail", row
    assert row["where"] == "dispatch", row
    assert row["tool"] == "vault_write", row
    # The journal's own shape: `reason` is the short why, `excerpt` is the target.
    # Read from `denial_journal.record`'s signature rather than assumed, because a
    # row whose path lives in a field nobody greps is a row nobody finds.
    assert row["excerpt"] == TASK_PATH, row
    assert "status" in row["reason"], row
    assert "up_next" in row["reason"] and "draft" in row["reason"], row
    assert "autonomy_schedule_rail" in row["label"] or row["label"], row


async def _dispatch_vault_write(monkeypatch, sessions_dir, session_id):
    """Send `vault_write` the way a turn sends it: through the aggregator.

    `agent_mcp.main.call_tool` is the boundary every caller crosses — the tool
    sandbox, the no-session rule, the effect ledger and the `asyncio.to_thread`
    hop into the module handler all sit between a turn and `_vault_write`. A
    refusal proved only on the handler can still fail to reach the caller: `_wrap`
    is what sets `isError`, and the thread hop is where a handler exception would
    surface as a transport error instead of a refusal. The effect ledger is stubbed
    for the reason the automod seam test stubs it: it is not this seam, and its
    duplicate guard plus its live-DB write would judge the wrong thing.
    """
    import agent_mcp.main as M
    import app.paths

    (sessions_dir / f"{session_id}.json").write_text(
        json.dumps({"id": session_id, "inner_voice": False,
                    "platform": "mission-control"}),
        encoding="utf-8")
    monkeypatch.setattr(app.paths, "SESSIONS_DIR", sessions_dir)

    async def _claim(*_a, **_k):
        return M._tool_effects.Claim()

    monkeypatch.setattr(M._tool_effects, "claim", _claim)

    async def _call(path: str, content: str):
        res = await M.call_tool("vault_write", {"path": path, "content": content},
                                {M.META_SESSION_ID: session_id})
        return json.loads(res.content[0].text), res.is_error

    return _call


def test_the_rail_survives_real_mcp_dispatch_in_both_directions(scratch, monkeypatch,
                                                                tmp_path):
    """clause 1 and clause 3 where a turn actually stands: `agent_mcp.main.call_tool`.

    Four crossings of one boundary, in the order the rail's own rule requires. The
    draft create lands as a success — the positive control, because without it the
    three refusals below are satisfiable by a handler that never runs, the
    silent-no-op shape the finding appended to this item warns about. Then the file
    is armed off-lane (`_seed_armed`), since arming is exactly what this lane
    refuses and a test that armed it through the door under test would be seeding
    the refusal it means to measure. The dispatch move then comes back `isError`
    naming `status` and #724, the same move spelled `./autonomy/…` comes back
    refused too — that one is the finding closing here, on the wire and not only in
    the handler — and a benign Activity-Log write comes back a success that really
    landed.
    """
    import asyncio

    call = asyncio.run(_dispatch_vault_write(monkeypatch, tmp_path, "s-vault-rail"))
    armed = _task(status="up_next")

    seeded, is_error = asyncio.run(call(TASK_PATH, _task(status="draft")))
    assert not is_error and seeded.get("success") is True, seeded

    _seed_armed(scratch, armed)

    moved, is_error = asyncio.run(call(TASK_PATH, _task(status="draft")))
    assert is_error, f"a dispatch move came back a success across dispatch: {moved}"
    assert "`status`" in moved["error"] and "#724" in moved["error"], moved
    assert "up_next" in moved["error"] and "draft" in moved["error"], moved

    dotted, is_error = asyncio.run(call(f"./{TASK_PATH}", _task(status="draft")))
    assert is_error, f"a dot-segment dispatch move came back a success: {dotted}"
    assert "`status`" in dotted["error"] and TASK_PATH in dotted["error"], dotted

    text = (scratch.root / TASK_PATH).read_text(encoding="utf-8")
    assert text.count("status: up_next") == 1, "one of the two refusals wrote bytes"

    benign, is_error = asyncio.run(call(
        TASK_PATH,
        armed.replace("- 2026-10-05T06:27Z created by an autocode round",
                      "- 2026-10-05T06:27Z created by an autocode round\n"
                      "- 2026-10-07T18:20Z appended through real dispatch\n")))
    assert not is_error and benign.get("success") is True, benign
    assert "appended through real dispatch" in (
        (scratch.root / TASK_PATH).read_text(encoding="utf-8")), (
        "the success result lied: the file on disk never changed")


# ── the fourth lane: `Write` and `Edit` (#2516) ─────────────────────────────
#
# Everything above is `vault_write`. The two handlers a turn reaches for most —
# `agent_mcp/builtin_fs._write` and `_edit` — carried no autonomy guard at all:
# `PROTECTED_WRITE_ROOTS` has five entries and the directory the scheduler reads
# is none of them, so an `Edit` of `status: draft` -> `status: up_next` wrote
# bytes. Triage counted 0 such calls across all 17,097 files in
# `~/lloyd-data/sessions`, which makes this #2070's shape rather than #2362's: the
# rule existed, the second surface it belongs on did not carry it, and closing the
# detour is the whole of it. The 2026-10-05 incident quoted at the top of this file
# is a `vault_write` incident, and that lane has refused it since #2362.
#
# So these nodes drive the real `Write`/`Edit` handlers, not a re-derived
# predicate, against a scratch vault, and they pin the two things that make two
# lanes holding one rule safe: the field set is read from
# `policy.SCHEDULE_STATE_FIELDS` at call time here too, and the two lanes' path
# predicates agree on extension even though one sees a vault-relative string and
# the other a realpath.

#: The scratch-vault seam. `_autonomy_root()` reads `app.paths.VAULT_ROOT` per call
#: — the same seam `tests/test_okf_type_taxonomy.py::_fs_scratch` uses for the
#: vocabulary guard, and the reason this fixture patches the module attribute and
#: not a name inside `builtin_fs`.
@pytest.fixture
def fs(tmp_path, monkeypatch):
    """`Write`/`Edit` aimed at a scratch tree, with the denial journal spied on.

    `_Mutation`'s defaults do the rest: `gate_on` is False with no aggregator
    context, which is the shape a unit caller takes straight into the handler, and
    the write deny-set is `$HOME`-relative, so a temp path sits outside it and the
    only thing that can refuse here is a guard under test.
    """
    import agent_mcp.builtin_fs as bfs
    import app.paths as paths
    from app.harness import denial_journal

    monkeypatch.setattr(paths, "VAULT_ROOT", tmp_path)
    journal: list[dict] = []
    # Plain append, no `or True` — the same reason the `scratch` fixture gives.
    monkeypatch.setattr(denial_journal, "record", lambda **kw: journal.append(kw))
    return types.SimpleNamespace(root=tmp_path, mod=bfs, journal=journal)


def _abs(fs, rel: str) -> str:
    return str(fs.root / rel)


def _refusal(result: str) -> dict:
    """The refusal payload when the lane refused, `{}` when it wrote.

    This lane answers success with a sentence (`File written: …`, `Edited …`) and
    failure with a JSON object — the shape
    `tests/test_okf_type_taxonomy.py::_fs_refusal` documents, and the one
    `text_result` sniffs to set `isError`.
    """
    stripped = result.lstrip()
    return json.loads(stripped) if stripped.startswith("{") else {}


def _fs_write(fs, rel: str, content: str) -> dict:
    return _refusal(fs.mod._write({"file_path": _abs(fs, rel), "content": content}))


def _fs_edit(fs, rel: str, old_string: str, new_string: str) -> dict:
    return _refusal(fs.mod._edit({"file_path": _abs(fs, rel),
                                  "old_string": old_string,
                                  "new_string": new_string}))


def _fs_seed(fs, content: str, rel: str = TASK_PATH) -> str:
    """Put a task file on disk WITHOUT the lane under test.

    Necessary, not convenient, for the same reason `_seed_armed` is: this lane
    refuses an armed create by design, so a file whose `status` reads `up_next` can
    only arrive the way a real one arrives — a granted call, a human editor, or the
    scheduler's own writer, none of which is the door under test.
    """
    target = fs.root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return content


def test_an_edit_moving_status_on_an_existing_task_is_refused_and_writes_no_bytes(fs):
    """Clause 1 on the `Edit` lane: refused, fully named, and not one byte moved.

    The delta is why the guard sits under the commit lock rather than in
    `_gate_check`: an `Edit` arrives as `old_string`/`new_string`, so the move this
    rail exists to catch is visible only in the file the lane has already assembled.
    The refusal has to carry everything the `vault_write` refusal carries — path,
    field, `old -> new`, the rail it defers to — because a writer that is told "no"
    without being told which field cannot do the half of the job that is left to it.
    """
    before = _fs_seed(fs, _task(status="draft"))

    out = _fs_edit(fs, TASK_PATH, "status: draft", "status: up_next")

    assert out, "a draft -> up_next Edit reached a task file through the open lane"
    msg = out["error"]
    assert _abs(fs, TASK_PATH) in msg, f"the refusal must name the path: {msg}"
    assert "`status`" in msg and "draft -> up_next" in msg, msg
    assert "#724" in msg, f"a refusal must name the rail it defers to: {msg}"
    assert (fs.root / TASK_PATH).read_text(encoding="utf-8") == before, (
        "the refusal wrote bytes anyway")
    assert fs.journal and len(fs.journal) == 1, f"one refusal, one row: {fs.journal}"
    assert fs.journal[0]["tool"] == "Edit", fs.journal
    assert fs.journal[0]["guard"] == "autonomy_schedule_rail", fs.journal


def test_a_write_creating_an_armed_task_is_refused_while_the_draft_create_lands(fs):
    """Clause 2: the create answer is `policy.create_arms_a_task`, not "no creates".

    The left half is the 2026-10-05 call spelled with `Write` instead of
    `vault_write`: a new file whose `status` is `up_next` dispatches the moment it
    lands, and there is no pre-image to diff, so the lane has to answer at create
    time. The right half is why the answer is not a blanket refusal: the nightly
    chain hands a task down as `status: draft` with `skill_name` and `frequency`
    already set, and a guard that stopped that would stop the chain. The directory
    assertion is the third shape of "nothing was written" — a refusal that ran its
    own `mkdir` first leaves the outline of the write it refused.
    """
    refused = _fs_write(fs, TASK_PATH, _task(status="up_next"))

    assert refused, "creating `status: up_next` was accepted"
    assert "up_next" in refused["error"] and "#724" in refused["error"], refused
    assert not (fs.root / TASK_PATH).exists(), "refused, but the file is there"
    assert not (fs.root / "autonomy").exists(), "refused, but the mkdir ran"

    landed = _fs_write(fs, TASK_PATH, _task(status="draft", frequency="weekly",
                                            skill_name="djev-name-prior-probe"))
    assert landed == {}, f"the sanctioned draft create was refused: {landed}"
    text = (fs.root / TASK_PATH).read_text(encoding="utf-8")
    assert "status: draft" in text and "skill_name: djev-name-prior-probe" in text, text
    assert len(fs.journal) == 1, f"only the refusal journals: {fs.journal}"


def test_writes_that_move_no_dispatch_field_still_land_on_the_write_edit_lane(fs):
    """Clause 3 on this lane: an armed task file is still a writable file.

    Four benign writes — an Activity Log append, a `description` edit, a whole-file
    body append, and a `scheduled_at` re-quote whose PARSED value is identical —
    then the control that makes them mean something: the same file, one field
    moved, refused. The re-quote is the case the #2190 owed-after-landing
    measurement exists to catch, and its mechanism is asserted rather than assumed:
    if the loader stopped resolving an unquoted timestamp to a date, the accept
    below would be proving nothing, and this file wants to know that.
    """
    _fs_seed(fs, _task(status="up_next", scheduled_at="2026-10-09T00:00:00+00:00"))

    logged = _fs_edit(fs, TASK_PATH,
                      "- 2026-10-05T06:27Z created by an autocode round",
                      "- 2026-10-05T06:27Z created by an autocode round\n"
                      "- 2026-10-11T09:00Z probed; 3 divergences in 24 h")
    assert logged == {}, f"an Activity Log append refused: {logged}"
    assert "3 divergences in 24 h" in (fs.root / TASK_PATH).read_text(encoding="utf-8")

    described = _fs_edit(fs, TASK_PATH, "description: run the weekly probe",
                         "description: run the weekly probe and log the count")
    assert described == {}, f"a description edit refused: {described}"

    on_disk = (fs.root / TASK_PATH).read_text(encoding="utf-8")
    parsed = P.front_matter_map(on_disk)["scheduled_at"]
    assert not isinstance(parsed, str), (
        "the loader no longer resolves an unquoted timestamp to a date, so the "
        "re-quote below cannot be a reformat and this node proves nothing")

    requoted = _fs_edit(fs, TASK_PATH, "scheduled_at: 2026-10-09T00:00:00+00:00",
                        "scheduled_at: '2026-10-09T00:00:00+00:00'")
    assert requoted == {}, f"a re-quoted scheduled_at was a schedule change: {requoted}"
    after = (fs.root / TASK_PATH).read_text(encoding="utf-8")
    assert after.count("scheduled_at:") == 1, "the re-quote duplicated the key"

    appended = _fs_write(fs, TASK_PATH, after + "\n## Findings\n\n- 3 in 24 h\n")
    assert appended == {}, f"a whole-file body append refused: {appended}"

    parked = _fs_edit(fs, TASK_PATH, "status: up_next", "status: draft")
    assert parked, "the control failed: parking an armed task is a dispatch move"
    assert "`status`" in parked["error"] and "up_next -> draft" in parked["error"], parked


@pytest.mark.parametrize("field", sorted(P.SCHEDULE_STATE_FIELDS))
def test_every_field_in_the_rail_is_guarded_on_this_lane_too(fs, field):
    """Clause 4: the set this lane refuses IS `policy.SCHEDULE_STATE_FIELDS`.

    Parametrised over the frozenset itself — not a list this file maintains — and
    driven as an `Edit`, the shape whose move is invisible to anything but the
    assembled bytes. `PAIRS` is reused from the `vault_write` nodes above so a field
    added to the rail later falls through to the generic pair here as well and is
    still refused; the node after this one executes that claim rather than
    asserting it.
    """
    old, new = PAIRS.get(field, ("alpha-value", "beta-value"))
    _fs_seed(fs, _task(**{field: old}))

    out = _fs_edit(fs, TASK_PATH, f"{field}: {old}", f"{field}: {new}")

    assert out, f"`{field}` moved through `Edit` and nothing asked: {out}"
    assert f"`{field}`" in out["error"], f"the refusal omits the field: {out['error']}"
    # The pair, in the form the two lanes share: an unquoted `scheduled_at` parses
    # to a `datetime.date`, which `shown_schedule_value` renders as JSON, so that
    # case arrives as `"2026-10-09" -> "2026-10-12"` and the values are still both
    # in the refusal. Same tolerance the `vault_write` node above spells out.
    assert f"{old} -> {new}" in out["error"] or (
        old in out["error"] and new in out["error"]), out["error"]
    assert "#724" in out["error"], out["error"]


def test_a_field_added_to_the_rail_widens_this_lane_without_editing_it(fs,
                                                                      monkeypatch):
    """Clause 4 executed on the fourth lane: widen the rail, the lane widens.

    A lane that kept its own tuple would pass the parametrised node above and fail
    this one — which is exactly why each lane carries both, and why
    `app/harness/policy.py`'s claim that "a field added above is refused by every
    lane" needed this node to be true of three lanes and not two.
    """
    _fs_seed(fs, _task(review_gate="off"))

    monkeypatch.setattr(P, "SCHEDULE_STATE_FIELDS",
                        set(P.SCHEDULE_STATE_FIELDS) | {"review_gate"})
    out = _fs_edit(fs, TASK_PATH, "review_gate: off", "review_gate: on")

    assert out, (
        "the lane refused known fields only because it knew them — it is not "
        f"reading the rail: {out}")
    assert "`review_gate`" in out["error"], out["error"]


def test_the_write_edit_lane_fails_closed_on_front_matter_it_cannot_read(fs):
    """The same two fail-closed shapes the `vault_write` lane pins, on this lane.

    Skipping what it cannot parse is a rail a caller switches off by deleting one
    `---` line, and the recovery parser is the scheduler's own, so a file this lane
    cannot read may still be a file the scheduler dispatches. A create with no
    front matter, and a body-level `Edit` onto a file whose ON-DISK front matter
    never closes, are both refusals — and neither is the #724 refusal, which is why
    the second one asserts the absence of that wording: a fail-closed that reads
    like a field move teaches the writer the wrong lesson.
    """
    refused = _fs_write(fs, TASK_PATH, "# just prose, no front matter\n")
    assert refused, f"a create with no front matter was accepted: {refused}"
    assert "front matter" in refused["error"], refused["error"]
    assert not (fs.root / TASK_PATH).exists()

    broken = "---\nstatus: draft\nfrequency: weekly\n"
    assert P.front_matter_map(broken) is None, "the fixture is not unparseable"
    _fs_seed(fs, broken, rel="autonomy/98-corrupt.md")

    out = _fs_edit(fs, "autonomy/98-corrupt.md", "status: draft", "status: up_next")
    assert out, "an Edit past unreadable front matter passed a rail that cannot read it"
    assert "front matter" in out["error"], out["error"]
    assert "#724" not in out["error"], "not a field move: " + out["error"]
    assert (fs.root / "autonomy" / "98-corrupt.md").read_text(encoding="utf-8") == broken


def test_an_unloadable_rail_stops_the_write_edit_lane_too(fs, monkeypatch):
    """The other fail-closed half: `sys.modules[...] = None`, the import trap.

    The rail is imported inside the guard, which is what makes the field set live
    at call time and also what makes a broken install or a circular import a
    caller's silent problem. `sys.modules[name] = None` is the import machinery's
    own way of making that import raise. `builtin_fs` imports `policy` nowhere
    else, so the only handler on this path that could produce the word below is
    this guard — a refusal from the deny-set would say `protected`, not `rail`.
    """
    import sys

    before = _fs_seed(fs, _task(status="draft"))
    monkeypatch.setitem(sys.modules, "app.harness.policy", None)

    out = _fs_edit(fs, TASK_PATH, "status: draft", "status: up_next")

    assert out, "a write went through with no rail to check it"
    assert "rail" in out["error"], out["error"]
    assert (fs.root / TASK_PATH).read_text(encoding="utf-8") == before


def test_a_task_path_the_lane_cannot_read_as_bytes_is_a_refusal_not_a_create(fs):
    """The third fail-closed shape: no pre-image, yet something on disk.

    `_write` sets `mut.pre_bytes` only when its own read succeeds, so
    `pre_bytes is None` beside a path that exists is either a read failure or a
    file that appeared after `mut.existed` was computed — and the lane cannot tell
    either of those from a create without the read that just failed. Assuming
    "create" is the direction that arms something nobody has read. A directory
    named `96-x.md` reaches that branch without needing root: `read_bytes()` on it
    raises `IsADirectoryError`, the same `OSError` a permission or I/O failure
    raises at the same site.
    """
    (fs.root / "autonomy" / "96-x.md").mkdir(parents=True)

    out = _fs_write(fs, "autonomy/96-x.md", _task(status="up_next"))

    assert out, f"an unreadable task path was treated as a create: {out}"
    assert "could not be read" in out["error"], out["error"]
    assert "#724" not in out["error"], out["error"]
    assert (fs.root / "autonomy" / "96-x.md").is_dir(), (
        "the refusal replaced a directory with a file")


#: Shapes whose realpath is dispatch state, or emphatically is not, and whether the
#: rail is meant to reach them. The `_archived/` case is True on purpose: the vault
#: lane's predicate counts the whole subtree because retirement moves a file into
#: `_archived/` without touching its `status`, so a restore is a write of those same
#: bytes one level up. This lane must count the same set, or the two lanes disagree
#: about which files are dispatch state — a disagreement no test would notice until
#: one of them refused a write the other had just allowed.
PATH_SHAPES = (
    pytest.param("autonomy/96-djev-name-prior-probe.md", True, id="task-file"),
    pytest.param("autonomy/_archived/25-memory-capture.md", True, id="archived-subtree"),
    pytest.param("autonomy/referential-integrity-latest.md", True, id="report-in-task-dir"),
    pytest.param("backlog/96-something.md", False, id="backlog-status-is-not-dispatch"),
    pytest.param("notes/autonomy/96-x.md", False, id="deeper-dir-same-name"),
    pytest.param("autonomy/96-notes.txt", False, id="not-markdown"),
)


def _lane_reaches(fs, rel: str) -> bool:
    """Does this lane refuse a `status` move at `rel`? Behaviour, not a predicate.

    Asked of the handler rather than of a helper on purpose: the claim worth pinning
    is what a caller experiences, and a private predicate that nothing calls cannot
    drift into disagreement with a guard that is.
    """
    _fs_seed(fs, _task(status="draft"), rel=rel)
    return bool(_fs_edit(fs, rel, "status: draft", "status: up_next"))


@pytest.mark.parametrize("rel,expected", PATH_SHAPES)
def test_the_two_lanes_agree_on_which_paths_are_dispatch_state(fs, rel, expected):
    """One rule, two path shapes, one extension: the lane-drift clause.

    `agent_mcp/vault.py` decides on the vault-relative string its normalizer hands
    it; this lane decides on `mut.real`. Those are different values for the same
    file, so what can be checked is that they answer the same question the same way
    — which is what this node does, shape by shape, against the real refusal. The
    first assertion is the reader's control: if the vault lane's predicate moves,
    this node says the comparison went stale instead of quietly agreeing.
    """
    import agent_mcp.vault as V

    assert V._is_autonomy_task_path(rel) is expected, (
        f"`_is_autonomy_task_path` moved for {rel!r}, so this comparison is stale")
    assert _lane_reaches(fs, rel) is expected, (
        f"{rel!r} is dispatch state to one lane and not to the other")


def test_a_spelling_only_a_realpath_test_catches_is_refused(fs):
    """The realpath half: a dot segment, a doubled separator, a symlinked parent.

    Every one of these is the SAME file as `autonomy/96-x.md`, and every one is a
    string the `startswith("autonomy/")` guard #2362's first round shipped would
    have waved through — that bypass is documented at the top of the `vault_write`
    section of this file and pinned by its spelling-parametrised nodes. This lane
    keys on `mut.real`, so all four get the same refusal and the bytes still read
    `draft`. The precondition is asserted too: spellings that are not one file would
    make the loop below a tautology.
    """
    before = _fs_seed(fs, _task(status="draft"))
    name = "96-djev-name-prior-probe.md"
    (fs.root / "taskdir").symlink_to(fs.root / "autonomy", target_is_directory=True)
    spellings = [
        _abs(fs, TASK_PATH),
        f"{fs.root}/autonomy/./{name}",
        f"{fs.root}/autonomy//{name}",
        _abs(fs, f"taskdir/{name}"),
    ]
    assert len({str(Path(s).resolve()) for s in spellings}) == 1, (
        "these spellings are not one file, so the loop below proves nothing")

    for spelling in spellings:
        out = _refusal(fs.mod._edit({"file_path": spelling,
                                     "old_string": "status: draft",
                                     "new_string": "status: up_next"}))
        assert out, f"{spelling!r} moved `status` through the open lane"
        assert "`status`" in out["error"] and "#724" in out["error"], out["error"]
        assert (fs.root / TASK_PATH).read_text(encoding="utf-8") == before, (
            f"{spelling!r} was refused after the bytes were replaced")


def test_a_refusal_here_lands_the_same_denial_row_the_other_lanes_write(fs, monkeypatch,
                                                                       tmp_path):
    """Clause 1's journal half, read back off the file, under the shared guard name.

    The spy above shows a row was produced; this shows a person can find it, which
    is the only thing the owed-after-landing traffic count can use — and on this
    lane it is the ONLY trace, since `memory/audit/writes.jsonl` belongs to
    `agent_mcp/vault.py` and `builtin_fs.py` keeps no vault audit row. The guard
    name is deliberately the `vault_write` lane's: the question is "did anyone reach
    a task file through a lane that used to be open", one grep on `guard`, with
    `tool` telling the lanes apart. A per-lane guard name would make that two greps
    nobody runs.
    """
    from app.harness import denial_journal

    journal = tmp_path / "denials.jsonl"
    monkeypatch.setenv(denial_journal.JOURNAL_ENV, str(journal))
    monkeypatch.setattr(denial_journal, "record", _REAL_DENIAL_RECORDER)
    _fs_seed(fs, _task(status="draft"))

    assert _fs_edit(fs, TASK_PATH, "status: draft", "status: up_next")
    assert _fs_write(fs, "autonomy/97-new.md", _task(status="up_next"))

    rows = [json.loads(line)
            for line in journal.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2, rows
    assert {r["guard"] for r in rows} == {"autonomy_schedule_rail"}, rows
    assert {r["tool"] for r in rows} == {"Edit", "Write"}, rows
    assert {r["where"] for r in rows} == {"dispatch"}, rows
    assert all(r["excerpt"].endswith(".md") for r in rows), rows
    assert all("status" in r["reason"] and "up_next" in r["reason"] for r in rows), rows


def test_the_refusal_reaches_a_turn_through_the_tool_dispatcher(fs):
    """The seam the code graph cannot see: `call_tool`'s name-keyed Write/Edit branch.

    `_write` and `_edit` are reached from `agent_mcp/builtin_fs.call_tool` by a
    string comparison on the tool name, with no symbol reference for a grep or the
    AST walk in `test_the_guard_is_wired_into_both_commit_sites` to find. A guard
    wired into a helper the dispatcher skips would pass every direct-handler node
    above and protect nothing, and the success it reported is the one the caller
    acts on. Both directions again: the move comes back `isError`, the Activity Log
    append comes back a success that really landed.
    """
    import asyncio

    _fs_seed(fs, _task(status="up_next"))

    moved = asyncio.run(fs.mod.call_tool("Edit", {
        "file_path": _abs(fs, TASK_PATH), "old_string": "status: up_next",
        "new_string": "status: draft"}))
    assert moved.is_error is True, (
        "the tool result said success; the caller will move on, not fix it")
    assert moved.content, "isError with no payload: the caller has nothing to act on"
    payload = json.loads(moved.content[0].text)
    assert "`status`" in payload["error"] and "#724" in payload["error"], payload

    benign = asyncio.run(fs.mod.call_tool("Edit", {
        "file_path": _abs(fs, TASK_PATH),
        "old_string": "- 2026-10-05T06:27Z created by an autocode round",
        "new_string": "- 2026-10-05T06:27Z created by an autocode round\n"
                      "- 2026-10-11T09:15Z appended across dispatch"}))
    assert benign.is_error is False, benign.content[0].text
    assert "appended across dispatch" in (fs.root / TASK_PATH).read_text(encoding="utf-8"), (
        "the success result lied: the file on disk never changed")


def test_the_guard_is_wired_into_both_commit_sites():
    """Both lanes, pinned structurally: `_write` and `_edit` each call the guard.

    Two call sites, one rule, and the failure mode is one of them quietly losing its
    call in a refactor — a handler that no longer asks is invisible to every node
    above, each of which drives exactly one of the two. So this parses the module
    and counts what it finds: `_gate_check` could not hold this rule (the post bytes
    do not exist there), and a call outside the `commit_lock` block would compare
    against a pre-image somebody else has since replaced, which is the invariant
    `tests/test_memory_writer_lane.py` exists to keep.
    """
    src = (Path(__file__).resolve().parents[1] / "agent_mcp" / "builtin_fs.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    guard = "_autonomy_schedule_refusal"

    found = {name: 0 for name in ("_write", "_edit")}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in found:
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == guard):
                    found[fn.name] += 1
                    assert fn.lineno < node.lineno
                    inside_lock = any(
                        isinstance(w, ast.With) and w.lineno < node.lineno
                        <= max(getattr(c, "lineno", 0) for c in ast.walk(w))
                        for w in ast.walk(fn)
                        if isinstance(w, ast.With)
                        and getattr(getattr(w.items[0].context_expr, "func", None),
                                    "id", "") == "commit_lock")
                    assert inside_lock, (
                        f"{fn.name} calls {guard} outside its commit_lock block, so "
                        "the pre-image it judges is not the one it will replace")
    assert found == {"_write": 1, "_edit": 1}, (
        f"one rule, two commit sites, and this is what the parse found: {found}")

    # And the rule is not restated here: the field set comes from the rail, so a
    # second copy in this module is impossible to miss from the outside.
    text = src.read_text(encoding="utf-8")
    assert "SCHEDULE_STATE_FIELDS" in text, "the lane no longer reads the rail at all"
    assert '"status", "scheduled_at", "depends_on"' not in text, (
        "a hand-copied field list has appeared in the lane")


# ── #2611: the two vault-path refusals are the whole contract a rejected caller gets ─
#
# Two refusals stand between a caller and the vault, and before this section
# neither was pinned by any test. `_normalize_vault_path` answers a wrong-SPELLING
# path with `path must be vault-relative or under <root>, got '<yours>'` — the one
# message that lets the caller repair its own call — and `git grep -n
# 'vault-relative or under' -- tests/` returned nothing, so deleting the
# `{vault_str}` half of that f-string failed nothing and left a refusal that says
# only "not that, and I won't say what". The two `is_relative_to` sites in
# `_vault_read` and `_vault_write` answer a DIFFERENT failure — a vault entry that
# is a symlink leaving the tree — with the bare constant `"path escapes vault
# root"`, naming neither the target nor the root it resolved through.
#
# The nodes below pin the two messages apart, which is the property: the reducer's
# must name the root and the word `vault-relative`, and the escape message must
# name the rejected target and must NOT contain that word. For a path that reaches
# the escape check, being vault-relative is already true — the reducer rejects a
# leading `/`, a leading `~` and every `~`/`..` segment long before
# `is_relative_to` runs — so advice to re-spell the path would be false advice
# about a path that already is one.
#
# Everything here asserts against the `scratch` fixture's patched root, never
# `/home/alansrobotlab/obsidian`. A node that read the real vault would pass
# against any message naming that path, including one produced by a reducer that
# had stopped honouring the patched `VAULT` — the exact thing being pinned.

#: A benign body for the write lane: none of these paths is an autonomy task file,
#: so the schedule rail, the OKF guards and the bench rail must all stay out of
#: the way of what is being measured here — the path checks in front of them.
BENIGN_NOTE = "# outside the vault\n\nNothing here is an autonomy task.\n"


def _outside_vault_shapes(scratch) -> tuple[tuple[str, str], ...]:
    """The two outside-vault spellings, built from the fixture rather than the machine.

    A `~/lloyd/...` home path (the shape a caller sends when it means the code
    checkout) and an absolute path under no vault at all (the generic case). The
    absolute one is a sibling of the scratch root, so this file hardcodes no home
    directory and keeps meaning the same thing wherever pytest put the tmp dir.
    """
    return (
        ("~/lloyd/agent_mcp/vault.py", "a home path under the lloyd checkout"),
        (str(Path(scratch.root).parent / "outside-the-vault" / "CLAUDE.md"),
         "an absolute path under no vault"),
    )


def _outside_vault_refusals(scratch) -> list[tuple[str, str, str, dict]]:
    """Four refusals: both tools against both outside-vault shapes.

    Driven through the handlers, not the reducer, because the reducer's string and
    the handler's error CODE are joined at the call site — `_vault_read` maps
    `norm_err` to `MISSING_PARAM` when the text contains "required" — and a node
    that called `_normalize_vault_path` directly could not see that join break.
    """
    rows: list[tuple[str, str, str, dict]] = []
    for tool in ("_vault_read", "_vault_write"):
        handler = getattr(scratch.mod, tool)
        for given, why in _outside_vault_shapes(scratch):
            params = {"path": given, "content": BENIGN_NOTE}
            rows.append((tool, given, why, handler(params)))
    return rows


def test_an_outside_vault_refusal_names_the_scratch_vault_root(scratch):
    """clause 1: the refusal that works still says which root it wants.

    The message pinned here is `path must be vault-relative or under <root>, got
    '<the caller's string>'`, and the half a later edit can most easily lose is
    `<root>`, leaving a refusal that says only "not that, and I won't say what".

    Four refusals — `_vault_read` and `_vault_write` each against a `~/lloyd/...`
    target and an absolute outside-vault target — and every one must carry both
    halves of the answer: the root (`str(scratch.root)`, the patched `VAULT`) and
    the required form (`vault-relative`). Pinned apart from the echo-and-code node
    below because they are halves of one f-string: a rewrite that kept the root and
    dropped the echo would leave this node green, and vice versa.
    """
    root = str(scratch.root)
    rows = _outside_vault_refusals(scratch)
    assert len(rows) == 4, f"two tools by two shapes is four refusals, not {len(rows)}"
    for tool, given, why, out in rows:
        assert "error" in out, f"{tool} accepted {why} {given!r}: {out}"
        msg = out["error"]
        assert root in msg, (
            f"{tool} refused {given!r} without naming the vault root {root!r} — "
            f"a caller cannot tell a vault from a home directory: {msg!r}")
        assert "vault-relative" in msg, (
            f"{tool} refused {given!r} without naming the required form: {msg!r}")


def test_an_outside_vault_refusal_still_echoes_the_callers_own_string(scratch):
    """clause 2: the same four refusals keep the half that makes the message actionable.

    `got '<the caller's own string>'` plus `code: PATH_ESCAPE` is what distinguishes
    "this spelling is wrong" from a refusal that names a root the caller can only
    guess about, and the code is what every caller's error branch keys on. Pinned in
    its own node because it is a different half of one f-string: a rewrite that kept
    the root and dropped the echo would leave clause 1 green.
    """
    for tool, given, why, out in _outside_vault_refusals(scratch):
        assert "error" in out, f"{tool} accepted {why} {given!r}: {out}"
        msg = out["error"]
        assert f"got {given!r}" in msg, (
            f"{tool}'s refusal of {given!r} stopped echoing the caller's string: "
            f"{msg!r}")
        assert out["code"] == "PATH_ESCAPE", (
            f"{tool} refused {given!r} with {out['code']!r}, not PATH_ESCAPE — the "
            "root is named but every caller's error branch keys on the code")


def _plant_escape_link(scratch, rel: str = "dream-link.md") -> tuple[Path, Path]:
    """Put a symlink INSIDE the scratch vault that points outside it.

    The only input that can reach `_vault_read`'s and `_vault_write`'s
    `is_relative_to` check: the reducer has already rejected a leading `/`, a
    leading `~` and every `~`/`..` segment, so a `..` traversal never gets there.
    Returns `(link, target)` with `target` living one directory OUT of the vault.
    """
    outside = Path(scratch.root).parent / "outside-the-vault"
    outside.mkdir(parents=True, exist_ok=True)
    target = outside / "dream-target.md"
    target.write_text("bytes that live outside the vault\n", encoding="utf-8")
    link = Path(scratch.root) / rel
    link.symlink_to(target)
    return link, target


def test_a_symlink_leaving_the_scratch_vault_is_refused_naming_its_target(scratch):
    """clause 3: the escape refusal names the rejected target, and never re-spells it.

    Both tools, one vault-relative path that resolves out of the tree. The message
    must carry the path as the caller named it AND what it resolves to — the only
    two facts that make a symlink escape findable — and must NOT contain
    `vault-relative`, because telling this caller to send a vault-relative path is
    false advice about a path that already is one. The refusals also have to refuse:
    `_vault_read` must not hand back the outside bytes, `_vault_write` must not
    replace them, and neither may leave an audit row.
    """
    link, target = _plant_escape_link(scratch)
    resolved = str(target.resolve())

    read = scratch.mod._vault_read({"path": link.name})
    assert "error" in read, f"a symlink out of the vault was read: {read}"
    assert "bytes that live outside the vault" not in read.get("text", ""), (
        "the escape guard let the outside file's bytes through anyway")
    for msg, where in ((read["error"], "_vault_read"),):
        assert link.name in msg, (
            f"{where} refused an escape without naming the rejected path "
            f"{link.name!r}: {msg!r}")
        assert resolved in msg, (
            f"{where} refused an escape without naming what it resolves to "
            f"({resolved}): {msg!r}")
        assert "vault-relative" not in msg, (
            f"{where} told a caller whose path already IS vault-relative to "
            f"re-spell it: {msg!r}")
        assert read["code"] == "PATH_ESCAPE", read

    wrote = _write(scratch, link.name, "rewritten through a symlink\n")
    assert "error" in wrote, (
        f"a symlink out of the vault was written through: {wrote}")
    msg = wrote["error"]
    assert link.name in msg, (
        f"_vault_write refused an escape without naming the rejected path "
        f"{link.name!r}: {msg!r}")
    assert resolved in msg, (
        f"_vault_write refused an escape without naming what it resolves to "
        f"({resolved}): {msg!r}")
    assert "vault-relative" not in msg, (
        "_vault_write told a caller whose path already IS vault-relative to "
        f"re-spell it: {msg!r}")
    assert wrote["code"] == "PATH_ESCAPE", wrote
    assert target.read_text(encoding="utf-8") == "bytes that live outside the vault\n", (
        "the refused write reached the file the symlink points at, outside the vault")
    assert _audit_rows(scratch) == [], (
        "the escape refusal returned after the audit row, so a refused write is "
        "recorded as if it landed")


def test_a_symlink_that_stays_inside_the_vault_is_not_the_refusal_above(scratch):
    """The control the node above needs: the guard refuses LEAVING, not symlinks.

    Without this, `test_a_symlink_leaving_the_scratch_vault_is_refused_naming_its_target`
    would still pass if the containment check were replaced by `return _err(...)`
    unconditionally, and a vault whose links all resolve in-tree is the normal
    shape of the tree — the real vault currently holds none at all
    (`find ~/obsidian -maxdepth 4 -type l` → 0 on 2026-10-12), which is why
    item #2611 files this as defence in depth rather than a hot path.
    """
    inside = Path(scratch.root) / "notes"
    inside.mkdir(parents=True, exist_ok=True)
    (inside / "target.md").write_text("inside bytes\n", encoding="utf-8")
    (Path(scratch.root) / "alias.md").symlink_to(inside / "target.md")

    out = scratch.mod._vault_read({"path": "alias.md"})
    assert out.get("text") == "inside bytes\n", (
        f"a link that stays in the vault was refused: {out}")





# ── #2589: the body rule, on both free-form lanes ────────────────────────────────
#
# The #2265 corpus guard asks one question of a task file: does its instruction region
# order the run to use a tool that the prompt the scheduler actually builds never names?
# Until #2589 only the automod land graded the answer, because the predicate lived in
# `tests/test_research_doc_claims.py` and a writer's rail cannot import a test helper. So
# both free-form lanes read front matter only, a body could join the graded corpus
# unrefused, and the item's incident file surfaced hours later as a red base probe on
# somebody else's round.
#
# The predicate is shipped code now — `agent_mcp/_shared.py::body_undelivered_tools`,
# beside `with_body_contract` — and every node below drives a rail that calls it.
#
# Fixture geometry, which is what makes these verdicts mean anything: the seed's
# instruction region names no tool, and its backticked names sit exactly where the region
# builder cuts — under `## Activity Log`, and inside a `- 2026-…T…:` dated bullet. So an
# untouched seed is clean, and a refusal on it can only mean the cut broke. The status is
# always `draft`, because `_task`'s default is `up_next` and an `up_next` create is
# refused by the #2362 arming rule: such a node would pass with the body rule deleted.
#
# `vault_write` is the offender name, and the reason is measured: `email_send` looked like
# the better choice (it is in `_GUARD_MUST_SEE`) but it is NOT in the universe on this box —
# the Thunderbird bridge is absent from a worktree, so the advertised surface has 107 names
# and none of them is a mail tool, while the annotated tables have 132 and none either. A
# fixture naming a tool outside the universe is refused by neither lane and the node passes
# because the guard is blind, not because the rule held. `vault_write` is in
# `_GUARD_MUST_SEE`, so the universe floor that both the guard and the rails run is what
# keeps it present; the three `_GUARD_MUST_SEE` names that are not mail tools were checked
# in universe here.
BODY_OFFENDER = "vault_write"
BODY_ORDER = f"Send the digest with `{BODY_OFFENDER}`.\n"


def _body_task(**over) -> str:
    """A draft task file with an empty-by-design instruction region and NO skill.

    Dropping `skill_name` (`_task` filters a None value out) is load-bearing, not tidiness.
    `_task`'s default names a real skill, `agent_mcp._shared.task_prompt_of` delivers that
    SKILL.md into the prompt exactly as a run receives it, and the skill this fixture would
    inherit backticks `email_send` — so the body sentence below would be ordering a tool the
    prompt DOES name and the correct verdict would be to let it through. Measured: with the
    inherited skill, `body_undelivered_tools` on the offending document returns []. An empty
    `skill_name` is what removes it: `_task` filters a None override away, so the key has to
    arrive as `""` to render as `skill_name: ` and resolve to no skill at all — the smallest
    prompt there is, which is where an undelivered name is unambiguous.
    """
    over.setdefault("status", "draft")
    over.setdefault("skill_name", "")
    head, log = _task(**over).split("## Activity Log", 1)
    return (head + "Read the probe report, then record the outcome.\n\n"
            "## Activity Log\n" + log)


def _ordered(doc: str) -> str:
    """The same document plus one instruction sentence, ABOVE the Activity Log heading —
    the one position that is unambiguously an order and not a record."""
    head, log = doc.split("## Activity Log", 1)
    return head.rstrip("\n") + "\n\n" + BODY_ORDER + "\n## Activity Log" + log


def _body_rows(rows: list[dict]) -> list[dict]:
    """This rule's own denial rows out of a journal spy. The owed check over the live
    journal is one grep on the guard name, so the tests read the spy the same way."""
    return [r for r in rows if r.get("guard") == "autonomy_body_rail"]


def _ledger_spy(scratch, monkeypatch) -> list[str]:
    """The vault lane's own ledger call, spied on the lane.

    The same patch site
    `test_a_refused_write_records_no_change_ledger_entry` uses, for the same reason:
    `_ledger_record` is this module's one way into the change ledger, so a list that
    stays empty on a refusal and grows on an accept is the whole claim — a refusal
    opens no undo record, because an entry for a write that never happened is an undo
    record pointing at nothing. The benign node below is that spy's positive control.
    """
    recorded: list[str] = []
    monkeypatch.setattr(scratch.mod, "_ledger_record",
                        lambda *a, **k: recorded.append(str(k.get("path") or a[:1])))
    return recorded


# ── clause 1: vault_write refuses on the create phase, naming tool and path ───────

def test_vault_write_refuses_a_body_ordering_a_tool_its_prompt_never_names(scratch, monkeypatch):
    """clause 1: the body rule refuses on this lane, says both the offending tool and the
    path, and leaves nothing behind.

    The file is the shape the incident had — a task whose body gained a sentence ordering
    a tool its prompt never delivers. Before #2589 this call returned success.
    """
    ledger = _ledger_spy(scratch, monkeypatch)

    res = _write(scratch, TASK_PATH, _ordered(_body_task()))

    text = str(res)
    assert "refused" in text.lower(), f"the write was accepted: {text[:300]}"
    assert BODY_OFFENDER in text, f"the refusal does not name the offending tool: {text[:300]}"
    assert TASK_PATH in text, f"the refusal does not name the path: {text[:300]}"
    assert not (scratch.root / TASK_PATH).exists(), "a refused create wrote the file"
    assert ledger == [], f"a refused write opened a ledger entry: {ledger}"
    assert _audit_rows(scratch) == [], "a refused write opened an audit row"
    rows = _body_rows(scratch.journal)
    assert len(rows) == 1, f"expected exactly one body-denial row, got {rows}"
    assert rows[0]["tool"] == "vault_write", rows[0]


# ── clause 3: the locked-diff phase, the only one an edit can reach ──────────────

def test_the_vault_lanes_locked_diff_phase_refuses_the_body_too(scratch, monkeypatch):
    """clause 3: phase A only ever sees a CREATE, so rewriting the body of a file that
    already exists — which is how #97's red body arrived — moved no dispatch field and
    reached the disk untouched. A body rule on phase A alone was a create check.

    Driven through the handler AND by calling the locked phase directly. The handler alone
    cannot tell "phase B refused" from "phase A refused", since A runs first and would have
    said the same words; the direct call is what pins which one.
    """
    ledger = _ledger_spy(scratch, monkeypatch)
    seed = _seed(scratch, _body_task())
    ledger.clear()      # the seed write above is an accepted write and records one entry
    before = (scratch.root / TASK_PATH).read_bytes()
    new = _ordered(seed)

    direct = scratch.mod._autonomy_schedule_move_refusal(TASK_PATH, new, before)
    assert direct is not None and "refused" in str(direct).lower(), (
        f"the locked-diff phase passed a body ordering {BODY_OFFENDER}: {direct}")
    assert BODY_OFFENDER in str(direct) and TASK_PATH in str(direct), (
        f"the phase-B refusal names neither the tool nor the path: {str(direct)[:300]}")

    res = _write(scratch, TASK_PATH, new)
    assert "refused" in str(res).lower(), f"the lane accepted the body rewrite: {str(res)[:300]}"
    assert (scratch.root / TASK_PATH).read_bytes() == before, "a refused write changed the file"
    assert ledger == [], f"a refused write opened a ledger entry: {ledger}"

    # The other half of "as well as": the dispatch rule still fires on this phase. A body
    # rule that displaced it would still read as a working rail everywhere else.
    moved = new.replace("status: draft", "status: up_next")
    dispatch = scratch.mod._autonomy_schedule_move_refusal(TASK_PATH, moved, before)
    assert dispatch is not None and "status" in str(dispatch), (
        f"the dispatch rule stopped firing on the locked phase: {dispatch}")
    assert (scratch.root / TASK_PATH).read_bytes() == before


# ── clause 2: Write and Edit, same bytes, same verdict, same site ────────────────

def test_write_and_edit_refuse_the_same_body_shape(fs, monkeypatch):
    """clause 2: the second free-form lane, which is the one the item's own evidence
    points at — the incident file has zero `vault_write` audit rows, and the fs lane is
    the door with neither a rail nor a journal row. A rule on one of two write surfaces
    is not a rule (#2362's lesson, learned twice).

    `Edit` is not decoration: it is the same reducer with a different post-text, and the
    corpus guard catches two-word body edits, so a lane that refused only whole-file
    writes would refuse the rare shape and pass the common one.
    """
    led: list[dict] = []
    monkeypatch.setattr(fs.mod, "_ledger_begin", lambda mut, **kw: led.append(kw))
    _fs_seed(fs, _body_task())
    before = (fs.root / TASK_PATH).read_bytes()

    w = _refusal(fs.mod._write({"file_path": _abs(fs, TASK_PATH),
                                "content": _ordered(_body_task())}))
    assert w, "Write accepted the offending body and returned prose instead of a refusal"
    assert BODY_OFFENDER in w["error"], f"Write did not name the tool: {w['error'][:300]}"
    assert TASK_PATH in w["error"], f"Write did not name the path: {w['error'][:300]}"

    e = _refusal(fs.mod._edit({"file_path": _abs(fs, TASK_PATH),
                               "old_string": "Read the probe report, then record the outcome.",
                               "new_string": BODY_ORDER.rstrip("\n")}))
    assert e, "Edit accepted the same shape"
    assert BODY_OFFENDER in e["error"], f"Edit did not name the tool: {e['error'][:300]}"
    assert TASK_PATH in e["error"], f"Edit did not name the path: {e['error'][:300]}"

    assert (fs.root / TASK_PATH).read_bytes() == before, "a refused Write/Edit changed the file"
    assert led == [], f"a refused Write/Edit opened the undo/ledger entry: {led}"
    assert [r.get("tool") for r in _body_rows(fs.journal)] == ["Write", "Edit"], (
        f"expected one body-denial row per lane, got {[r.get('tool') for r in fs.journal]}")


def test_the_body_rule_is_called_by_the_same_rail_at_the_same_site(fs):
    """clause 2's placement half — "at the same pre-commit site as the #2362 dispatch-field
    rail" is a claim about WHERE, and every behavioural node above passes whether the check
    runs before the write or after it. Only a post-write refusal would leave bytes behind,
    and that would fail somewhere else for the wrong reason. So this reads the tree.

    Three things: the dispatch rail itself reaches the body rule (one decision point, not
    two that can drift — drift is how #2589 happened); both commit sites still ask; and in
    line order, the ask comes before `_ledger_begin`, so a refusal cannot leave an undo
    entry pointing at a write that never happened.
    """
    tree = ast.parse(Path(fs.mod.__file__).read_text(encoding="utf-8"))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def calls_by_line(fn):
        return {n.func.id: n.lineno for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    asked = calls_by_line(fns["_autonomy_schedule_refusal"])
    assert "_autonomy_body_denial" in asked, (
        "the fs lane's dispatch rail no longer reaches the body rule, so the two checks "
        "are two places again and one of them will be forgotten")
    for site in ("_write", "_edit"):
        lines = calls_by_line(fns[site])
        assert "_autonomy_schedule_refusal" in lines, f"{site} stopped asking the rail"
        assert lines["_autonomy_schedule_refusal"] < lines["_ledger_begin"], (
            f"{site} opens the ledger before asking the rail, so a refusal there leaves an "
            "undo entry for a write that never happened")


# ── clause 4: what must keep landing, on both lanes ──────────────────────────────

def test_a_benign_body_still_lands_and_is_ledgered_on_both_lanes(scratch, fs):
    """clause 4: three shapes that are not orders, each carrying a backticked tool name and
    so each one this rule could refuse by accident.

      * the name in the front-matter `description`, which `_build_task_prompt` DOES deliver
        to the run — ordering it in prose is then a true instruction, and refusing it would
        make the guard refuse the fix it tells the caller to make;
      * the name only under `## Activity Log`, the scheduler's own history;
      * the name only inside a dated `- 2026-…T…:` bullet, which is what
        `append_activity_line` writes and what a run's honest report looks like.

    Each must land AND be recorded: a guard that refuses nothing but stops writing the undo
    ledger and the audit row would satisfy this clause's first sentence and break the run's
    ability to undo the write.
    """
    named = _body_task(description=f"Report the digest with {BODY_OFFENDER}.")
    assert BODY_OFFENDER in named, "fixture lost its prompt-named tool"
    log_only = _body_task()
    dated = _body_task().replace(
        "## Activity Log",
        f"- 2026-10-11T07:00:00Z: used `{BODY_OFFENDER}` for this.\n\n## Activity Log", 1)
    for label, doc in (("prompt-named", named), ("log-only", log_only), ("dated", dated)):
        res = _write(scratch, TASK_PATH, doc)
        assert res.get("success") is True, f"{label}: a benign write was refused: {res}"
        assert doc in (scratch.root / TASK_PATH).read_text(encoding="utf-8"), \
            f"{label}: reported success without writing the bytes"
    assert not _body_rows(scratch.journal), f"a benign write was denied: {_body_rows(scratch.journal)}"
    assert len(_audit_rows(scratch)) == 3, (
        f"expected one audit row per benign write, got {_audit_rows(scratch)}")

    _fs_seed(fs, _body_task())
    w = fs.mod._write({"file_path": _abs(fs, TASK_PATH), "content": named})
    assert not _refusal(w), f"Write refused a prompt-named body: {w}"
    assert BODY_OFFENDER in (fs.root / TASK_PATH).read_text(encoding="utf-8")
    e = fs.mod._edit({"file_path": _abs(fs, TASK_PATH),
                      "old_string": "- 2026-10-05T06:27Z created by an autocode round",
                      "new_string": "- 2026-10-05T06:27Z created by an autocode round\n"
                                    f"- 2026-10-12T09:00:00Z: used `{BODY_OFFENDER}`"})
    assert not _refusal(e), f"Edit refused a dated log bullet: {e}"
    assert not _body_rows(fs.journal), f"the fs lane denied a benign body: {_body_rows(fs.journal)}"


# ---------------------------------------------------------------------------
# The rail's prose, held to the set it introduces (#3578)
# ---------------------------------------------------------------------------
# Every node above this line drives a lane that REFUSES a write, and each of
# them reads `policy.SCHEDULE_STATE_FIELDS` at call time — so the enforcement
# was right about `preferred_hours` while the comment describing it was wrong.
# `app/harness/policy.py` used to say, eight lines above the constant and twelve
# above the frozenset: "`model`, `preferred_hours`, `next_run`/`last_run`,
# `max_retries`, `failure_count` and `runs_per_day` are deliberately NOT in the
# set", while the frozenset below contained `preferred_hours` and the comment
# attached to it said the field was added on purpose, because the nightly
# chain's order lives nowhere else. Two opposite rules inside one block, and the
# false one was the sentence telling a future agent the window is writable — the
# write that #3359 had to carry through a human authorisation, and whose absence
# from the rail #3578's triage measured as a clean grep for the bound field and
# a refusal for this one.
#
# Rewriting the sentence alone leaves the drift one edit away, and no runtime
# test can see a comment. So these three nodes parse it. `_membership_claims`
# and `_set_exclusion_offenders` take TEXT rather than the module, which is what
# lets `test_the_drift_pin_fires_on_the_sentence_that_started_it` feed the
# detector the exact sentence quoted above and show it bites — the failure this
# file was written to prevent, reproduced rather than asserted.
_POLICY_PY = Path(__file__).resolve().parents[1] / "app" / "harness" / "policy.py"

#: The two markers `policy.py` uses to enumerate its own membership, and the
#: only spelling of a field name these parsers accept: backticks, lowercase,
#: underscores — how every field is written in that file.
_IN_CLAIM = "IN the set:"
_OUT_CLAIM = "NOT in the set"
_BACKTICKED = re.compile(r"`([a-z_]+)`")

#: Phrases that deny membership of the set. Deliberately phrases about THE SET,
#: not any negation: `policy.py` also says of `preferred_hours` "Leaving it out
#: would keep the hazard this item was filed about standing", which is a reason
#: it is in, and an exclusion detector with a looser pattern would report the
#: file's best rationale as its bug.
_SET_EXCLUSION_PHRASES = (
    "not in the set",
    "not in that set",
    "not in `schedule_state_fields`",
    "outside `schedule_state_fields`",
    "outside the set",
    "are not members",
    "left out of the set",
    "removed from the set",
)


def _comment_paragraphs(src: str) -> list[tuple[int, str]]:
    """Contiguous runs of `#` lines, each one broken at a blank comment line.

    Returns `(1-based first row, joined text)`. Breaking at a bare `#` is what
    keeps the exclusion scan honest: `policy.py` separates a block's paragraphs
    that way, so a member named in one paragraph cannot be indicted by an
    exclusion phrase belonging to its neighbour.
    """
    out: list[tuple[int, str]] = []
    start: int | None = None
    buf: list[str] = []
    for row, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        text = stripped[1:].strip() if stripped.startswith("#") else ""
        if text:
            if start is None:
                start, buf = row, [text]
            else:
                buf.append(text)
        elif start is not None:
            out.append((start, " ".join(buf)))
            start, buf = None, []
    if start is not None:
        out.append((start, " ".join(buf)))
    return out


def _membership_claims(src: str) -> dict[str, set[str]]:
    """Field names the comment claims IN and OUT, keyed "in"/"out".

    A claim is one paragraph opening with its marker and continuing to the next
    blank comment line, so a wrapped list still reads as one claim.
    """
    claims = {"in": set(), "out": set()}
    for _, para in _comment_paragraphs(src):
        if para.startswith(_IN_CLAIM):
            claims["in"].update(_BACKTICKED.findall(para))
        elif para.startswith(_OUT_CLAIM):
            claims["out"].update(_BACKTICKED.findall(para))
    return claims


def _set_exclusion_offenders(src: str, members: set[str]) -> list[tuple[int, list[str]]]:
    """Paragraphs that name a member of `members` while denying it is one.

    Each hit is the paragraph's first row and the members it indicts, so the
    report points at a line rather than at a file.
    """
    hits: list[tuple[int, list[str]]] = []
    for row, para in _comment_paragraphs(src):
        low = para.lower()
        if not any(p in low for p in _SET_EXCLUSION_PHRASES):
            continue
        named = sorted({t for t in _BACKTICKED.findall(para)} & members)
        if named:
            hits.append((row, named))
    return hits


#: The sentence as it stood at `bb333b3b`, kept verbatim as the thing the
#: detectors must catch. `git -C ~/lloyd show bb333b3b:app/harness/policy.py`
#: lines 298-306; the trailing constant is here so the paragraph reads as it did
#: in context.
_PRE_3578_SENTENCE = (
    "# `model`, `preferred_hours`, `next_run`/`last_run`, `max_retries`,\n"
    "# `failure_count` and `runs_per_day` are deliberately NOT in the set. The first\n"
    "# two change how a run behaves once dispatched, not whether it dispatches — a\n"
    "# parked task stays parked with a new model, and a narrowed hour window cannot\n"
    "# start a run its status has not authorised. The rest are scheduler OUTPUT; the\n"
    "# scheduler writes them itself, and gating them would deny every completion\n"
    "# stamp. There is no interactive fallback on the worker path, so an over-broad\n"
    "# predicate here ends every nightly parked behind a human.\n"
    'SCHEDULE_STATE_TOOL = "autonomy_write_task"\n'
)


def test_no_comment_in_the_rail_puts_a_member_outside_the_set():
    """Clause 1 of #3578: no comment denies a field the set contains.

    The whole file is scanned, not just the block beside the constant, because
    the claim was never scoped to one place — a sentence in the middle of the
    module is what an agent skimming for the rule actually reads.
    """
    members = set(P.SCHEDULE_STATE_FIELDS)
    offenders = _set_exclusion_offenders(
        _POLICY_PY.read_text(encoding="utf-8"), members)
    assert not offenders, (
        "app/harness/policy.py comments deny these fields are members of "
        "`SCHEDULE_STATE_FIELDS` while the set contains them "
        f"(line, fields): {offenders}. The prose is what tells a turn which "
        "writes are refused; a denial here is a false statement that the "
        "scheduler's ordering field — or any other member — is free to edit.")


def test_the_rail_comment_and_the_set_cannot_drift_apart():
    """Clause 2 of #3578: the enumerated lists and the set are the same facts.

    Both directions are pinned. `IN` must equal the frozenset, so a field added
    to the set with no line written about it fails here instead of quietly
    becoming undocumented; `OUT` must contain no member at all, which is the
    #3578 defect itself.
    """
    members = set(P.SCHEDULE_STATE_FIELDS)
    claims = _membership_claims(_POLICY_PY.read_text(encoding="utf-8"))
    assert claims["in"] and claims["out"], (
        "`policy.py` no longer carries a paragraph opening "
        f"{_IN_CLAIM!r} and one opening {_OUT_CLAIM!r}, so this pin has nothing "
        f"to read (in={sorted(claims['in'])}, out={sorted(claims['out'])}). "
        "Deleting the enumeration is a decision to unpin the prose from the "
        "set; say so in the commit and update this node with it.")
    assert claims["in"] == members, (
        f"the comment and the set disagree: enumerated {_IN_CLAIM[:-1]} "
        f"{sorted(claims['in'])} but `SCHEDULE_STATE_FIELDS` is "
        f"{sorted(members)} — in the comment only "
        f"{sorted(claims['in'] - members)}, in the set only "
        f"{sorted(members - claims['in'])}. One of them is wrong about which "
        "writes the rail refuses.")
    assert not (claims["out"] & members), (
        f"`{_OUT_CLAIM}` names {sorted(claims['out'] & members)}, which the set "
        "does contain — the #3578 contradiction, back again.")


def test_the_drift_pin_fires_on_the_sentence_that_started_it():
    """The two detectors fail on the defect they were written for.

    Not a restatement of the nodes above: each feeds a detector text that is
    wrong and requires a finding. Without this node a parser that silently
    returned empty sets would pass both pins forever — an empty `claims` and an
    empty `offenders` are exactly what a broken detector produces.
    """
    members = set(P.SCHEDULE_STATE_FIELDS)
    hits = _set_exclusion_offenders(_PRE_3578_SENTENCE, members)
    assert hits and hits[0][1] == ["preferred_hours"], (
        "the pre-#3578 sentence, which denies membership to a field the set "
        f"contains, produced {hits}. The exclusion scan no longer sees the "
        "sentence that filed this item, so clause 1 is unpinned.")

    src = _POLICY_PY.read_text(encoding="utf-8")
    trimmed = src.replace("# `frequency`, `skill_name`, `preferred_hours`.",
                          "# `frequency`, `preferred_hours`.")
    assert trimmed != src, (
        "the IN line no longer wraps through `skill_name`, so this mutation has "
        "nothing to trim and the equality pin below is untested — re-point it at "
        "whatever the line says now.")
    assert _membership_claims(trimmed)["in"] != members, (
        "dropping `skill_name` from the enumeration left the IN line equal to "
        "the set anyway, so a field that enters `SCHEDULE_STATE_FIELDS` without "
        "a line in the comment would pass the pin above. That is the drift "
        "clause 2 exists to catch.")
