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
