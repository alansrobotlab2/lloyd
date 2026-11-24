"""Five lists name a backlog task's `status`, and none had a case for a
value outside them.

The four Python readers and the React board all agree on the four words, so
drift between them was never the bug. The bug is the *gap*: an item whose
status is none of the four is counted **open** by
`dashboard._BACKLOG_CLOSED` and is invisible to `backlog.OPEN_STATUSES`, so a
human sees work in progress and every machine that would move it — including
`reconcile_statuses`, which forms opinions only about items already in
`open_items` — cannot reach it. #287 (`review`) and #304 (`closed`) sat in
that gap from April 2026 until 2026-09-09.

These pin the shared definition, the rescue, and the direction of the
mapping: a word wrongly called terminal buries a live item in `done` forever,
while a word wrongly called live costs one triage run that closes it.

The last section is the same vocabulary at the *call*: what one of the four
words' neighbours — `open`, `todo`, `backlog` — costs a caller of
`backlog_write_task`, and whether the board's own two writers answer it the way
the MCP tool does.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.backlog_status import (  # noqa: E402
    CLOSED_ALIASES,
    CLOSED_STATUSES,
    PIPELINE_STATUSES,
    STATUS_GUESS_WORDS,
    canonical_status,
    is_off_vocabulary,
)

WEB = ROOT / "web" / "src"


def _board_page_statuses() -> set[str]:
    """The statuses the React board renders — it cannot import Python."""
    src = (WEB / "components" / "pages" / "BacklogPage.tsx").read_text(encoding="utf-8")
    m = re.search(r"const STATUSES\s*=\s*\[(.*?)\]\s*as const", src, re.DOTALL)
    assert m, "no `const STATUSES = [...] as const` in BacklogPage.tsx"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


# ── the five lists ─────────────────────────────────────────────────────────

def test_every_python_reader_shares_one_vocabulary():
    from agent_mcp import backlog as mcp_backlog
    from app.routers import backlog as board_router
    from scripts.automod import backlog as loop

    assert set(mcp_backlog.VALID_STATUSES) == set(PIPELINE_STATUSES)
    assert set(board_router._VALID_STATUSES) == set(PIPELINE_STATUSES)
    assert set(loop.PIPELINE_STATUSES) == set(PIPELINE_STATUSES)
    assert set(loop.OPEN_STATUSES) == set(PIPELINE_STATUSES) - {"done"}


def test_react_board_renders_every_status_the_writers_accept():
    missing = set(PIPELINE_STATUSES) - _board_page_statuses()
    assert not missing, (
        f"statuses the API accepts that the board cannot render: {sorted(missing)}. "
        "An item written with one lands in no column."
    )


def test_dashboard_closed_set_is_the_shared_one():
    from app.routers import dashboard

    assert dashboard._BACKLOG_CLOSED is CLOSED_STATUSES
    # The legacy half is what makes the gap asymmetric: the dashboard knows
    # these words take a task off the board and the pipeline has never heard
    # of them.
    assert CLOSED_STATUSES - {"done"} == CLOSED_ALIASES
    assert "done" in CLOSED_STATUSES


# ── the mapping ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("review", "draft"),      # #287 — the case that started this
    ("closed", "done"),       # #304
    ("cancelled", "done"),
    ("wontfix", "done"),
    ("draft", "draft"),
    ("up_next", "up_next"),
    ("Done", "done"),         # case is a human writing the same word
    ("  In_Progress ", "in_progress"),
    ("blocked", "draft"),     # unknown → the recoverable direction
    ("", "draft"),
    (None, "draft"),
])
def test_canonical_status(raw, want):
    assert canonical_status(raw) == want


def test_unknown_words_go_to_draft_not_done():
    """The lopsided direction, stated as a property.

    `done` is terminal for every writer here, so a live item sent there is
    never read again. A finished item sent to `draft` costs one triage run,
    which closes it. Only words already known to be terminal may map to
    `done`.
    """
    for word in ("review", "blocked", "waiting", "needs-review", "parked"):
        assert canonical_status(word) == "draft", word


def test_is_off_vocabulary_is_case_sensitive_and_ignores_absent():
    # `Done` strands exactly as `review` does — neither reader lowercases.
    assert is_off_vocabulary("Done")
    assert is_off_vocabulary("review")
    assert is_off_vocabulary("closed")
    # Absent: `load_item` and the dashboard both already default it to draft,
    # so the halves agree and there is nothing on disk to correct.
    assert not is_off_vocabulary("")
    assert not is_off_vocabulary(None)
    for good in PIPELINE_STATUSES:
        assert not is_off_vocabulary(good)


# ── the rescue ─────────────────────────────────────────────────────────────

def _write_item(d: Path, num: int, status: str, board: str = "lloyd") -> Path:
    p = d / f"{num}-item.md"
    p.write_text(
        f"---\nstatus: {status}\npriority: medium\nboard: {board}\n"
        f"created: '2026-04-13T07:09:17'\n---\n\n# Item {num}\n\nbody\n",
        encoding="utf-8")
    return p


@pytest.fixture()
def board(tmp_path, monkeypatch):
    from scripts.automod import backlog as loop
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(loop, "BACKLOG_DIR", d)
    return d


def test_an_off_vocabulary_item_is_stranded_between_the_two_halves(board):
    """The regression, stated end to end — this is what the rescue undoes."""
    from app.routers import dashboard
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")

    # The loop cannot see it...
    assert [i.id for i in loop.open_items()] == []
    # ...but the walk that ignores status can.
    assert [(i.id, i.status) for i in loop.all_items()] == [(287, "review")]
    # ...and the dashboard counts it as open work.
    assert "review" not in dashboard._BACKLOG_CLOSED


def test_rescue_moves_review_to_draft_and_closed_to_done(board, tmp_path):
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    _write_item(board, 304, "closed")
    ledger = tmp_path / "ledger.jsonl"

    moved = loop.rescue_off_vocabulary(ledger)

    assert {m["item_id"]: m["to"] for m in moved} == {287: "draft", 304: "done"}
    assert {i.id: i.status for i in loop.all_items()} == {287: "draft", 304: "done"}
    # #287 is now reachable by the pass that could never see it.
    assert [i.id for i in loop.open_items()] == [287]
    # #304 is off the board, not back in the triage pool.
    assert 304 not in {i.id for i in loop.open_items()}


def test_rescue_records_each_move_on_the_ledger(board, tmp_path):
    import json

    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    ledger = tmp_path / "ledger.jsonl"
    loop.rescue_off_vocabulary(ledger)

    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["event"] == "status_moved"
    assert (rows[0]["item_id"], rows[0]["from"], rows[0]["to"]) == (287, "review", "draft")
    assert rows[0]["off_vocabulary"] is True


def test_rescue_writes_the_reason_into_the_activity_log(board, tmp_path):
    from scripts.automod import backlog as loop

    p = _write_item(board, 287, "review")
    loop.rescue_off_vocabulary(tmp_path / "ledger.jsonl")

    text = p.read_text(encoding="utf-8")
    assert "review" in text and "→ draft" in text


def test_rescue_is_idempotent(board, tmp_path):
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    ledger = tmp_path / "ledger.jsonl"

    assert len(loop.rescue_off_vocabulary(ledger)) == 1
    assert loop.rescue_off_vocabulary(ledger) == []
    assert loop.rescue_off_vocabulary(ledger) == []


def test_rescue_respects_the_board_filter(board, tmp_path):
    """The backlog is shared. An Alfie item with an unusual status is not
    this loop's to rewrite."""
    from scripts.automod import backlog as loop

    _write_item(board, 500, "review", board="alfie")
    ledger = tmp_path / "ledger.jsonl"

    assert loop.rescue_off_vocabulary(ledger, ("lloyd",)) == []
    assert {i.id: i.status for i in loop.all_items(None)} == {500: "review"}
    # ...but None means every board, as it does everywhere else here.
    assert len(loop.rescue_off_vocabulary(ledger, None)) == 1


def test_rescue_leaves_the_four_good_statuses_alone(board, tmp_path):
    from scripts.automod import backlog as loop

    for n, st in enumerate(PIPELINE_STATUSES, start=100):
        _write_item(board, n, st)
    before = {i.id: i.path.read_text(encoding="utf-8") for i in loop.all_items()}

    assert loop.rescue_off_vocabulary(tmp_path / "ledger.jsonl") == []
    assert {i.id: i.path.read_text(encoding="utf-8") for i in loop.all_items()} == before


def test_reconcile_runs_the_rescue_first(board, tmp_path):
    """The rescued item must be judged in the same pass, not the next one."""
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    ledger = tmp_path / "ledger.jsonl"

    moved = loop.reconcile_statuses(ledger, ("lloyd",))

    assert {m["item_id"]: m["to"] for m in moved} == {287: "draft"}
    assert [i.status for i in loop.all_items()] == ["draft"]


def test_reconcile_kill_switch_covers_the_rescue(board, tmp_path):
    from scripts.automod import backlog as loop

    _write_item(board, 287, "review")
    assert loop.reconcile_statuses(tmp_path / "ledger.jsonl", ("lloyd",), enabled=False) == []
    assert [i.status for i in loop.all_items()] == ["review"]


# ── the write surfaces: what a guessed status word costs (#4525) ────────────
#
# The gap above is on disk. This half is at the call, and it is one field: in the
# seven-day window #4525 was filed from, `Invalid board`, `Invalid priority` and
# `Invalid description_mode` occurred **zero** times while `Invalid status`
# occurred ten times — and `priority`, `board` and `description_mode` all declare
# an `enum` in the same `backlog_write_task` inputSchema, while `status` was a bare
# `{"type": "string", "description": "Task status"}`. The one guessed field was the
# one whose vocabulary was never declared where the caller reads it, and no line
# number for that schema is quoted here, because this very diff moves it.
#
# Measured 2026-11-24, anchored on the tool payload so that a queue-health run
# quoting the error text back at itself is not counted as a rejected call:
#
#   find ~/lloyd-data/autonomy-runs -name 'run_*.md' -newermt '-7 days' -print0 \
#     | xargs -0 grep -o 'success": false, "error": "Invalid status' | wc -l
#
# → 10 calls in 9 run records across 9 tasks (24, 60, 67, 76 ×2, 78, 80, 83, 84,
# 92): `open` ×6, `todo` ×3, `backlog` ×1. #4525's own 11-calls-in-10-tasks figure
# is that count one day older: the records behind its tasks 53 and 79 have since
# aged out of the window and one new hit has arrived (`run_83_20261124_014832`).
# The anchor is what makes it a count of calls: `run_76_20261124_011244.md` — the
# queue-health pass that filed this item — quotes the message twice in prose and
# made neither call, and a bare `grep -cE "Invalid status '" matches both.
#
# The rule these nodes pin: a guessed word costs nothing where it cannot do harm,
# and costs exactly what it costs today everywhere else. It is never allowed to
# *arm* an item — `up_next` is the state `app/autonomy.py` dispatches from, so a
# wrong `up_next` is not self-correcting the way a wrong `draft` is — hence
# `draft` as the target, and only where the item is already at `draft` or carries
# no status at all.

#: The three words the run records actually show, in the shapes they arrive in:
#: `' OPEN '` and `'Backlog'` are the same guesses with the casing and padding a
#: hand-written call produces, which `canonical_status` already tolerates for the
#: four legal words.
GUESS_WORDS = ("open", "todo", "backlog", " open ", "Open", "Todo", "BACKLOG")

#: A guessed word on an item that has already moved is a demotion, and an
#: `up_next → draft` flip is how #68 went dark for ~50 cycles
#: (`app/autonomy.py`, the comment at `:3393`).
MOVED_STATUSES = ("up_next", "in_progress", "done")

#: Words outside the enum that nobody in the measured window guessed. No alias is
#: invented for them: `closed` is a retired *terminal* word (see CLOSED_ALIASES),
#: and a guess-words list grown to cover every English synonym is the enum widened
#: by another door.
UNGUESSED_WORDS = ("review", "closed", "WIP", "pending")

#: The refusal byte for byte, as a literal rather than rebuilt from the sorted
#: tuple the writers use — a pin that recomputed the message from the same
#: expression could not catch the message itself being changed.
REFUSAL_TAIL = "Must be one of: done, draft, in_progress, up_next"


def _refusal(word) -> str:
    return f"Invalid status '{word}'. {REFUSAL_TAIL}"


def _status_on_disk(board: Path, item_id: int) -> str:
    """The raw `status` field, not `canonical_status`'s reading of it.

    `load_item`'s own default is `draft`, so a node that means to prove a word
    was *written* also checks the line; see
    `test_a_guessed_status_word_on_a_create_lands_as_draft_and_says_so`.
    """
    from scripts.automod import backlog as loop

    hits = list(board.glob(f"{item_id}-*.md"))
    assert len(hits) == 1, f"expected one file for #{item_id}, got {[h.name for h in hits]}"
    item = loop.load_item(hits[0])
    assert item is not None, f"{hits[0].name} did not parse as a backlog item"
    return item.status


class _Post:
    """The `Request` the two board routes read: only `.json()` is ever called."""

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def write_board(tmp_path, monkeypatch):
    """One isolated backlog dir, wired into every surface these nodes write through.

    The MCP handler and the board's routes keep *separate* module-level bindings
    (`agent_mcp.backlog.BACKLOG_DIR`, `app.routers.backlog._BACKLOG_DIR`), so a node
    that means to prove the surfaces agree has to point both at one dir. `_FM_CACHE`
    goes too: the update route re-reads through that stat-validated cache, and a
    stale entry would read as a write that did not happen.

    A create through the MCP handler also appends its dedupe decision and its spawn
    count under the real `~/.local/state/lloyd-automod/` state dir — the same side
    effect `test_backlog_blocker_evidence.py` accepts when it files through this
    handler, and not what these nodes are about.
    """
    from agent_mcp import backlog as mcp_backlog
    from app.routers import backlog as board_router

    board = tmp_path / "backlog"
    board.mkdir()
    monkeypatch.setattr(mcp_backlog, "BACKLOG_DIR", board)
    monkeypatch.setattr(board_router, "_BACKLOG_DIR", board)
    monkeypatch.setattr(board_router, "_FM_CACHE", {})
    return board


def _mcp_result(board: Path, item_id, word) -> dict:
    """One `backlog_write_task` call carrying `status=word`; `item_id=None` creates."""
    from agent_mcp import backlog as mcp_backlog

    args: dict = {"status": word}
    if item_id is not None:
        args["task_id"] = item_id
    else:
        args.update({"name": f"Guess-word create probe {word!r}",
                     "description": "Filed by the #4525 status-vocabulary pin.",
                     "board": "lloyd"})
    return json.loads(mcp_backlog._handle_write(args))


def _mcp_write(board: Path, verdict_of, **args):
    from agent_mcp import backlog as mcp_backlog

    return verdict_of(json.loads(mcp_backlog._handle_write(args)), board, args)


def _route_update(board: Path, verdict_of, item_id: int, word):
    from app.routers import backlog as board_router

    try:
        resp = asyncio.run(board_router.backlog_task_update(
            _Post({"id": item_id, "status": word})))
    except board_router.HTTPException as exc:
        return verdict_of({"success": False, "error": exc.detail}, board,
                          {"id": item_id, "status": word})
    return verdict_of(json.loads(resp.body), board, {"id": item_id, "status": word})


def _route_create(board: Path, verdict_of, word):
    from app.routers import backlog as board_router

    body = {"name": f"Guess-word create probe {word!r}", "status": word}
    try:
        resp = asyncio.run(board_router.backlog_task_create(_Post(body)))
    except board_router.HTTPException as exc:
        return verdict_of({"success": False, "error": exc.detail}, board, body)
    out = json.loads(resp.body)
    return verdict_of({**out, "success": True}, board, body)


def _verdict(result: dict, board: Path, args: dict):
    """(accepted, status_now_on_disk, refusal_message, reported_coercion).

    `status_now_on_disk` is read from the file rather than from the response, so a
    writer that reports a status it did not write cannot pass. On an update the id
    is the probe file the node just created (#700, #701, #702); on a create it is
    the id the surface assigned to the item it made.
    """
    accepted = result.get("success") is True
    # `task-update` answers with no id at all, so the update surfaces' item comes
    # from the request: the tool calls it `task_id`, the route calls it `id`.
    item_id = (result.get("task_id") or result.get("id")
               or args.get("task_id") or args.get("id"))
    status = _status_on_disk(board, int(item_id)) if accepted and item_id else None
    return (accepted, status, result.get("error"), result.get("status_coerced"))


# ── clause 1: the vocabulary is declared where the caller reads it ──────────

@pytest.mark.parametrize("tool", ["backlog_write_task", "backlog_tasks"])
def test_every_backlog_status_property_in_the_tool_schemas_declares_the_four_words(tool):
    """A guessed word has to be visible as illegal *before* the call.

    Both tools take a `status`; until #4525 neither declared its values, so
    `backlog_write_task(status="open")` was invisible until it was refused and
    `backlog_tasks(status="open")` answered with an empty list, silently.
    `priority`, `board` and `description_mode` in the same schema already do this.
    """
    # `by_alias=True` because the claim is about what a client *reads*: `Tool` is
    # a pydantic model whose attribute is `input_schema` and whose wire name is
    # `inputSchema`, and the vocabulary only helps a caller once it is in the
    # serialised schema the `tools/list` response carries.
    from agent_mcp import backlog as mcp_backlog

    schemas = {t.name: t.model_dump(by_alias=True)["inputSchema"]
               for t in asyncio.run(mcp_backlog.list_tools())}
    declared = schemas[tool]["properties"]["status"].get("enum")

    assert declared is not None, f"{tool}'s `status` still declares no enum"
    assert set(declared) == set(PIPELINE_STATUSES), (
        f"{tool}'s `status` enum is not the four shared statuses: {declared}")
    # The enum is the four words and nothing else: `open` is coerced, not added,
    # because two names for one state is how a status field starts lying.
    assert not ({"open", "todo", "backlog"} & set(declared)), (
        f"{tool}'s `status` enum widened to a guess-word: {declared}")


# ── clause 2: the measured guesses cost nothing, everything else still does ─

@pytest.mark.parametrize("word", GUESS_WORDS)
def test_a_guessed_status_word_lands_as_draft_and_says_so(write_board, word):
    """The 10 measured rejected calls, replayed as writes that succeed.

    `draft` is the target, not `up_next`, for the two reasons the tree already
    states: `app/backlog_status.py` maps every non-terminal word to `draft`
    because a wrong draft costs one triage run and self-corrects, and
    `up_next` is the armed pool — a write of it is answered by the promotion
    ledger rather than applied, so the alias would have changed the error's
    shape instead of removing it.
    """
    _write_item(write_board, 700, "draft")

    out = _mcp_result(write_board, 700, word)

    assert out["success"] is True, out
    assert out["status_coerced"]["from"] == word, out
    assert out["status_coerced"]["to"] == "draft", out
    assert _status_on_disk(write_board, 700) == "draft"


@pytest.mark.parametrize("word", GUESS_WORDS[:3])
def test_a_guessed_status_word_on_a_create_lands_as_draft_and_says_so(write_board, word):
    """A create has no status to protect, and its own default is already `draft`.

    The written line is checked as text as well as through `load_item`, because
    `load_item` defaults a missing `status` to `draft` and would report success
    for a create that wrote no status field at all.
    """
    out = _mcp_result(write_board, None, word)
    task_id = out["task_id"]
    text = next(write_board.glob(f"{task_id}-*.md")).read_text(encoding="utf-8")

    assert out["success"] is True and out["created"] is True, out
    assert set(out["status_coerced"]) == {"from", "to", "why"}, out
    assert out["status_coerced"]["from"] == word, out
    assert out["status_coerced"]["to"] == "draft", out
    assert "status: draft\n" in text, f"no written status line in the created file:\n{text}"
    assert _status_on_disk(write_board, task_id) == "draft"


@pytest.mark.parametrize("word", UNGUESSED_WORDS)
def test_a_word_nobody_guessed_still_costs_the_call_with_the_same_refusal(write_board, word):
    """No alias is invented beyond the three words the records show.

    `closed` earns its place in the list of refusals most sharply: it is a retired
    *terminal* word, so mapping it anywhere from a write would be the one
    mistake `canonical_status` refuses to make — sending a live item off the board.
    An omitted or empty status is not a write at all on this path, so it is not
    probed here; `app/routers/backlog.py`'s create default covers that case.
    """
    _write_item(write_board, 701, "draft")

    out = _mcp_result(write_board, 701, word)

    assert out["success"] is False, out
    assert out["error"] == _refusal(word), out
    assert _status_on_disk(write_board, 701) == "draft"


# ── clause 3: a guess cannot move an item backwards ─────────────────────────

@pytest.mark.parametrize("live", MOVED_STATUSES)
@pytest.mark.parametrize("word", ("open", "todo", "backlog"))
def test_a_guessed_word_never_demotes_an_item_the_board_has_already_moved(write_board, live, word):
    """The coercion is allowed only where it cannot be the #68 flip.

    `app/autonomy.py:3393` records that an `up_next → draft` flip is how task #68
    went dark for ~50 cycles. A guessed word must therefore meet today's refusal
    on an armed or closed item, and the file must not be rewritten at all — a
    rejected call that re-stamped `updated:` would leave a write with no verdict.
    """
    _write_item(write_board, 702, live)
    path = next(write_board.glob("702-*.md"))
    before = path.read_bytes()

    out = _mcp_result(write_board, 702, word)

    assert out["success"] is False, out
    assert out["error"] == _refusal(word), out
    assert _status_on_disk(write_board, 702) == live
    assert path.read_bytes() == before, "a refused write still touched the file"


# ── clause 4: the two board writers cannot drift from the tool ──────────────

#: Words on which all three surfaces must answer identically. `up_next` is
#: deliberately absent: the MCP tool additionally rules on a promotion
#: (`_promotion` answers a create with `created_draft` and consults the ledger on
#: an update) and the Mission Control route does not, by #1023's design — that is
#: a ruling about arming, not the vocabulary question this clause is about.
AGREEMENT_WORDS = GUESS_WORDS + UNGUESSED_WORDS + ("draft", "in_progress", "done")


@pytest.mark.parametrize("word", AGREEMENT_WORDS)
def test_the_two_board_writers_accept_and_refuse_exactly_what_the_mcp_tool_does(
        write_board, word):
    """One rule, three doors: the tool, `task-update`, and `task-create`.

    #518's lesson is written into both files already — fixing one of the two
    writers left the other emitting non-conformant files — and an `Invalid
    status` refusal that differs between them is the same class of bug: the
    Mission Control board is the surface where a person types the word, so a
    rule that lives only in the MCP tool would leave every human guess refused.
    """
    _write_item(write_board, 700, "draft")

    tool = _mcp_write(write_board, _verdict, task_id=700, status=word)
    route = _route_update(write_board, _verdict, 700, word)
    created = _route_create(write_board, _verdict, word)

    assert tool == route == created, (
        f"the three write surfaces answered {word!r} differently:\n"
        f"  backlog_write_task  {tool}\n"
        f"  task-update         {route}\n"
        f"  task-create         {created}")
    # And the answer they agree on is the right one, not merely a shared one.
    if word in PIPELINE_STATUSES:
        assert tool == (True, word, None, None), tool
    elif re.sub(r"[^a-z0-9]+", "", word.strip().lower()) in STATUS_GUESS_WORDS:
        assert tool[0] is True and tool[1] == "draft" and tool[2] is None, tool
        assert tool[3]["from"] == word and tool[3]["to"] == "draft", tool
    else:
        assert tool == (False, None, _refusal(word), None), tool


@pytest.fixture
def board_client(write_board):
    """The board's routes behind a real ASGI request, not a hand-built `Request`.

    `_route_update` and `_route_create` call the handlers directly, which is how the
    three surfaces get compared field by field. This client is what crosses the
    seam: body decoding, the 400 a browser actually sees, and the JSON the response
    really serialises.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import backlog as board_router

    app = FastAPI()
    app.include_router(board_router.router)
    with TestClient(app) as c:
        yield c


def test_the_board_route_answers_a_guess_over_http_the_way_the_tool_does(
        write_board, board_client):
    """200 + `status_coerced` for a measured guess, 400 + the identical message for a stranger.

    Clause 4 is that the two write surfaces cannot drift, and the seam between them
    is HTTP: a node that only awaits the coroutine leaves a route which decoded the
    body differently, or answered a refusal with a 200 and an `error` field, green.
    """
    _write_item(write_board, 705, "draft")
    _write_item(write_board, 706, "draft")

    guess = board_client.post("/api/backlog/task-update",
                              json={"id": 705, "status": "open"})
    stranger = board_client.post("/api/backlog/task-update",
                                 json={"id": 706, "status": "WIP"})
    created = board_client.post("/api/backlog/task-create",
                               json={"name": "Over-the-wire guess probe",
                                     "status": "todo"})

    assert guess.status_code == 200, guess.text
    assert guess.json()["status_coerced"]["to"] == "draft", guess.text
    assert _status_on_disk(write_board, 705) == "draft"
    assert stranger.status_code == 400, stranger.text
    assert stranger.json()["detail"] == _refusal("WIP"), stranger.text
    assert _status_on_disk(write_board, 706) == "draft", "a refusal still moved the item"
    assert created.status_code == 200, created.text
    assert created.json()["status_coerced"]["to"] == "draft", created.text
    assert _status_on_disk(write_board, created.json()["id"]) == "draft"


# ── the read side must not hand out a word the writers refuse ───────────────

def test_a_file_with_no_status_reads_back_as_draft_not_as_todo(write_board):
    """`backlog_tasks` used to render a missing status as `todo`.

    `todo` is not in `PIPELINE_STATUSES`, is refused by every writer here, and is
    one of the three words the run records show being guessed (`todo` ×3) — so the
    reader was handing out the very word every writer here refuses.
    Honest scope: none of the board's 4,477 numbered item files was missing its
    `status` line when that was re-measured on 2026-11-24 (`for f in
    ~/obsidian/backlog/[0-9]*.md; do grep -q '^status:' "$f" || echo "$f"; done`
    prints nothing), so this is a latent vocabulary violation, not a proven source
    of the `todo` ×3. The default now matches `canonical_status`'s empty-status
    rule.
    """
    from agent_mcp import backlog as mcp_backlog

    _write_item(write_board, 703, "draft")
    path = next(write_board.glob("703-*.md"))
    path.write_text(
        path.read_text(encoding="utf-8").replace("status: draft\n", ""), encoding="utf-8")
    # A word from the old vocabulary, still on disk, to show the rule covers it:
    # the reader renders the canonical reading, not the stranded one.
    _write_item(write_board, 704, "review")

    rows = json.loads(mcp_backlog._handle_tasks({}))["tasks"]
    row = next(r for r in rows if r["id"] == 703)

    assert row["status"] == "draft", row
    assert next(r for r in rows if r["id"] == 704)["status"] == "draft", rows
    # The invariant the `"todo"` default broke, stated where it can actually be
    # measured: whatever is on disk, the list a caller reads back carries only the
    # four words every writer here accepts.
    assert all(r["status"] in PIPELINE_STATUSES for r in rows), rows


def test_filtering_by_a_declared_status_returns_what_the_reader_labels_with_it(write_board):
    """The enum on the filter argument is only honest if the filter reads it back.

    `backlog_tasks`'s `status` argument compared raw front matter while the row it
    emitted carried `canonical_status`'s reading, so an item stranded on an
    off-vocabulary word was labelled `draft` in the listing and then absent when the
    caller asked for `draft` — the pair of answers that teaches a caller its own
    vocabulary does not work. Both sides go through `canonical_status` now, which is
    what makes the four-value `enum` clause 1 puts on the argument answerable.
    """
    from agent_mcp import backlog as mcp_backlog

    _write_item(write_board, 707, "review")   # stranded: readers call it draft
    _write_item(write_board, 708, "draft")

    rows = json.loads(mcp_backlog._handle_tasks({"status": "draft"}))["tasks"]

    assert {707, 708} <= {r["id"] for r in rows}, rows
    assert all(r["status"] == "draft" for r in rows), rows
