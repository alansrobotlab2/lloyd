"""#1906 clause 3 — every session kind that gets the Bash tool records a start cwd.

Kinds reach the tool through separate mints, and each is tested at its own mint rather
than at a shared helper:

1. an automod review grader — `scripts/automod/review.py::write_session`, the one mint
   both grading routes pass through (a code round's grade and the vault-review grade);
2. a worker turn — `workers.sources._common.new_worker_session`, together with
   `ensure_session_start_cwd` for a turn handed a session that already existed;
3. a turn dispatched straight to the primary model —
   `workers.sources._common.run_prompt_on_primary`, which mints its own
   `platform="worker"` record and reaches NO other mint. This is the mint the first
   review of this clause missed, and it is what `workers/sources/bench_mine.py:934`,
   `bench_mine.py:955` and `workers/sources/session_distill.py:300` run their turns on;
4. an autocode round turn — `scripts/automod/round.py::start`, which restamps the
   worker mint's scratch with that round's worktree;
5. an autonomy run — `app/autonomy.py::run_task`, which mints its OWN session
   (`new_background_session_id("autonomy")` then `create_session(platform="autonomy")`)
   and is what `workers/sources/scheduled_task.py:407` delegates to, so a scheduled
   autonomy turn passes through none of the others. The incident writer
   `20260930_010013_autonomy_80fe` is one of those records: `platform: autonomy`,
   `source: autonomy-task:39`;
6. the two measurement harnesses whose turns carry a Bash tool —
   `eval/run_rpc_eval.py::run_one` (whose hook explicitly allows Bash) and
   `scripts/autoresearch/bench_runner_sdk.py::run_trial`. Both mint
   `platform="worker"`, so both fall under the AST walk below, which is where they are
   pinned rather than by a node each.

That list is kept honest by `test_every_session_mint_with_an_unattended_platform_stamps_its_record`,
which walks the tree for `create_session(platform="worker"|"autonomy")` call sites and
fails on one with no stamp behind it — so a seventh mint cannot be added unstamped the
way the sixth was. What the walk does NOT cover is the one mint deliberately left
unstamped, `scripts/automod/canary_smoke.py::run`: its turn is served by the canary's own
stack, whose `directory=` already puts that server's cwd in the round's worktree, so the
clause is true there without a stamp — see
`test_the_canary_smoke_turn_starts_outside_the_live_checkout_without_a_stamp`, and
`test_a_stamp_written_to_a_non_default_root_is_read_by_that_root_and_no_other` for what a
record in a non-default root can and cannot be read by.

Each writes `start_cwd` into its own session record — the one place
`agent_mcp/builtin_bash.py` reads — and each writes a directory OUTSIDE the live
checkout. A session handed the tree is moved to a scratch outside it rather than
falling back to the tree, which is the bug being closed: a storer that fell back would
reproduce the incident it exists to prevent.

The tests are the mints, not copies of one helper, because the mints are where a future
session kind gets forgotten: a helper that passes here tells you nothing about whether
the next `create_session` call site calls it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import session_cwd  # noqa: E402


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Redirect the convention's scratch root, and resolve the live root to a tree that
    exists in the test — so a recorded path is refused or accepted for the same reason
    production would refuse or accept it."""
    root = tmp_path / "session-cwd"
    monkeypatch.setattr(session_cwd, "SCRATCH_ROOT", root)
    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setattr(session_cwd, "live_root", lambda: live.resolve())
    return root


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── kind 1: the automod review grader ─────────────────────────────────
def test_the_review_graders_mint_records_a_start_cwd_outside_the_tree(tmp_path, scratch,
                                                                     monkeypatch):
    """`write_session` is the single mint both grading routes pass through — the code
    grade (`grade` -> `run_grader`) and the vault-review grade (`grade_vault` ->
    `run_grader`, which passes `round_id="vault"`) — so a test on the mint is a test on
    both call sites, including the vault one, which is the site that takes no worktree
    and would otherwise inherit the checkout."""
    from scripts.automod import review as RV

    sessions = tmp_path / "sessions"
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", sessions)
    session_id = RV.write_session(sessions, item_id=1906, round_id="SM_X", model="primary")

    record = sessions / f"{session_id}.json"
    assert record.is_file(), "the mint must still mint the session itself"
    # The import inside `write_session` resolves `app.session_cwd` afresh per call, so
    # the `SESSIONS_DIR` patch above is the one it sees.
    import app.session_cwd as SC
    start = SC.read(session_id)
    assert start is not None, (
        f"the grader's record names no start directory: {_read(record)}")
    start_path = Path(start)
    assert start_path.is_dir(), start
    assert session_cwd.outside_live(start_path), (
        f"a grader's Bash would start inside the live checkout: {start}")
    assert start_path.is_relative_to(scratch), (
        f"the grader should land on the convention's scratch root: {start}")
    assert record == sessions / f"{session_id}.json"


def test_the_review_graders_stamp_survives_when_the_route_is_a_vault_one(tmp_path, scratch,
                                                                        monkeypatch):
    """`round_id="vault"` is not a code round and has no worktree; it must still get a
    directory outside the tree rather than nothing."""
    from scripts.automod import review as RV

    sessions = tmp_path / "sessions"
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", sessions)
    session_id = RV.write_session(sessions, item_id=7, round_id="vault", model="primary")
    import app.session_cwd as SC
    start = SC.read(session_id)
    assert start and session_cwd.outside_live(Path(start)), (
        f"the vault grade has no worktree, so its scratch is the only answer: {start}")


# ── kind 2: the worker turn ───────────────────────────────────────────
def test_the_worker_mint_records_a_start_cwd_outside_the_tree(tmp_path, scratch,
                                                             monkeypatch):
    """`workers.sources._common.new_worker_session` is one function and the worker
    SOURCES that use it are many, so stamping here is what makes the rule hold for
    every `run_prompt_in_session` turn — the review grader, autocode and the vault
    reviewer included.

    Scope of this node, stated plainly: it drives the worker mint and nothing else. A
    scheduled autonomy run does NOT come through it — `app/autonomy.py::run_task` mints
    its own session with `platform="autonomy"`, which is why the record below asserts
    `platform == "worker"`, and why
    `test_an_autonomy_run_mints_its_session_with_a_start_cwd_outside_the_tree` drives
    `run_task` itself."""
    from workers.sources import _common as C

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    import app.sessions_io as SIO
    monkeypatch.setattr(SIO, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", sessions)
    session_id = C.new_worker_session(title="nightly probe", source="autonomy")

    import app.session_cwd as SC
    start = SC.read(session_id)
    assert start is not None, _read(sessions / f"{session_id}.json")
    assert session_cwd.outside_live(Path(start)) and Path(start).is_dir(), start
    record = _read(sessions / f"{session_id}.json")
    assert record["platform"] == "worker", (
        "this node pins the WORKER mint; an autonomy run's session is minted "
        f"elsewhere: {record['platform']!r}")


def test_an_autonomy_run_mints_its_session_with_a_start_cwd_outside_the_tree(
        tmp_path, scratch, monkeypatch):
    """Clause 3's third kind, driven through the mint an autonomy run ACTUALLY gets.

    A scheduled autonomy turn does not pass through `new_worker_session`:
    `workers/sources/scheduled_task.py` hands the job to `app.autonomy.py::run_task`,
    which mints its own session (`new_background_session_id("autonomy")` then
    `create_session(platform="autonomy", source="autonomy-task:<id>")`). On live,
    221 session records carry `platform: autonomy`, and the writer of the unmeasured
    `eval/uptake/classifier-report.json` this item opens with —
    `20260930_010013_autonomy_80fe` — is one of them, with no start directory. So this
    node drives the real `run_task` over a scripted `run_query` (the seam
    `tests/test_autonomy_run_summary_closing.py` uses) and reads back the record it
    minted through the resolver the Bash tool reads.

    The assertion is the two things the clause names: the record is the autonomy run's
    own (`platform`, `source`), and the directory the resolver hands back for it is a
    real directory outside the live checkout.
    """
    import asyncio

    import app.harness as harness_mod
    import app.harness.mcp_pool as mcp_pool
    from app import autonomy

    sessions = tmp_path / "sessions"
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", sessions)
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", sessions)

    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "autonomy-runs")
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_write_run_record", lambda *a, **k: {"run_id": "r"})

    def _task() -> dict:
        return {"id": 39, "name": "Uptake probe", "skill_name": "retrieval-eval",
                "status": "up_next", "timeout_seconds": 300}

    async def _run_query(messages, options):
        yield {"type": "text_delta", "text": "probe done\n"}
        yield {"type": "assistant_message", "text": "probe done\n"}

    class _Opts:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "t.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: _task())
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
    monkeypatch.setattr(harness_mod, "run_query", _run_query)
    monkeypatch.setattr(harness_mod, "RunOptions", _Opts)
    monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
    monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "SYS")
    monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False)

    asyncio.run(autonomy.run_task(39))

    created = [f for f in sessions.glob("*.json") if f.is_file()]
    assert len(created) == 1, (
        f"run_task must mint exactly one session for its run: "
        f"{[f.name for f in created]}")
    record = _read(created[0])
    assert record["platform"] == "autonomy", record
    assert record["source"] == "autonomy-task:39", record

    start = session_cwd.read(record["session_id"])
    assert start is not None, (
        f"an autonomy run's Bash inherits the server's cwd — the live checkout — "
        f"because its record names no start directory: {record}")
    start_path = Path(start)
    assert start_path.is_dir(), f"the recorded start cwd is not usable: {start}"
    assert session_cwd.outside_live(start_path), (
        f"an autonomy run's relative write would land in the live checkout: {start}")
    assert start_path.is_relative_to(scratch), (
        f"the autonomy mint should land on the convention's scratch root: {start}")


def test_a_turn_on_an_existing_session_gets_a_start_cwd_too(tmp_path, scratch,
                                                           monkeypatch):
    """A warmed autocode continuation arrives with `session_id=` already set, and a
    session that predates this change has no key. `run_prompt_in_session` stamps those,
    or the resumed turn's relative writes go back to the checkout.

    Two halves, because the first review was right that a helper passing proves nothing
    about the call site. What is pinned here: (a) the helper gives an existing session a
    start cwd outside the tree — driven directly, since running `run_prompt_in_session`
    for real would POST to a live backend at 127.0.0.1:8090, which is worse than the
    coverage it buys; and (b) that call site actually reaches the helper, read off the
    function's own AST rather than off a grep, so deleting the line is a red node and not
    a silently-uneventful one.
    """
    import ast
    import inspect
    import textwrap

    from workers.sources import _common as C

    body = ast.parse(textwrap.dedent(inspect.getsource(C.run_prompt_in_session)))
    called = {n.func.id for n in ast.walk(body)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "ensure_session_start_cwd" in called, (
        "`run_prompt_in_session` no longer stamps a session it was handed, so a warm "
        f"continuation's Bash inherits the server's cwd again: called {sorted(called)}")

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "sess-warm.json").write_text(
        json.dumps({"session_id": "sess-warm", "platform": "worker"}), encoding="utf-8")
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", sessions)

    written = C.ensure_session_start_cwd("sess-warm")
    assert written is not None
    assert Path(written).is_dir() and session_cwd.outside_live(Path(written)), written


def test_an_existing_start_cwd_is_not_overwritten_with_a_scratch(tmp_path, scratch,
                                                                monkeypatch):
    """A round that opened a round has its WORKTREE recorded, which is a better answer
    than a bare scratch: overwriting it would send the resumed turn's relative writes
    to a directory holding no diff."""
    from workers.sources import _common as C

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "sess-r.json").write_text(
        json.dumps({"session_id": "sess-r"}), encoding="utf-8")
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", sessions)
    worktree = tmp_path / "work" / "SM_R" / "home" / "lloyd"
    worktree.mkdir(parents=True)
    import app.session_cwd as SC
    SC.stamp("sess-r", worktree)

    assert C.ensure_session_start_cwd("sess-r") == str(worktree)


# ── kind 3b: the turn dispatched straight to the primary model ────────
def test_the_primary_model_mint_records_a_start_cwd_outside_the_tree(tmp_path, scratch,
                                                                    monkeypatch):
    """`run_prompt_on_primary` mints its own `platform="worker"` session and reaches no
    other mint, so the stamp that gives a worker its scratch never touched it — the gap
    the second review named. This node drives the real function over a scripted
    `run_query` (the seam `tests/test_run_recorder.py::test_run_prompt_on_primary_leaves_a_session_and_a_transcript`
    uses) and reads the minted record back through the resolver the Bash tool reads.

    The two things asserted are the mint's identity and the directory it now names: the
    record is `platform="worker"` with the source the caller passed, and
    `resolved_for`/`read` hand back a real directory outside the live checkout on the
    convention's scratch root. Before this commit the record came out with no
    `start_cwd` at all, which is the sentence in the failure message.
    """
    import asyncio

    import app.harness as harness
    from workers.sources import _common as C

    sessions = tmp_path / "sessions"
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", sessions)
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", sessions)
    monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False)

    async def _fake_run_query(messages, options):
        yield {"type": "text_delta", "text": "mined"}
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1}

    monkeypatch.setattr(harness, "run_query", _fake_run_query)
    monkeypatch.setattr(C, "_worker_run_options",
                        lambda *a, **k: type("O", (), {"session_id": "",
                                                       "turn_id": ""})())

    turn = asyncio.run(C.run_prompt_on_primary(
        "mine this failed run", max_turns=5, source="bench-mine",
        title="mine failed run 1234"))

    created = [f for f in sessions.glob("*.json") if f.is_file()]
    assert len(created) == 1, (
        f"the primary path must mint exactly one session for its turn: "
        f"{[f.name for f in created]}")
    record = _read(created[0])
    assert record["session_id"] == turn.session_id, record
    assert record["platform"] == "worker", record
    assert record["source"] == "bench-mine", record

    start = session_cwd.read(turn.session_id)
    assert start is not None, (
        "a bench-mine / session-distill turn's Bash inherits the server's cwd — the "
        f"live checkout — because its record names no start directory: {record}")
    start_path = Path(start)
    assert start_path.is_dir(), f"the recorded start cwd is not usable: {start}"
    assert session_cwd.outside_live(start_path), (
        f"a primary-path turn's relative write would land in the live checkout: {start}")
    assert start_path.is_relative_to(scratch), (
        f"the primary mint should land on the convention's scratch root: {start}")


def test_the_sources_named_in_the_finding_still_run_on_that_mint(tmp_path):
    """The mint above is only worth stamping if the turns the review named actually run
    on it. Read off the AST rather than a grep, for the two sources and the three call
    sites the finding lists: `workers/sources/bench_mine.py` twice (its failed-run and
    its ledger-loser mining turns) and `workers/sources/session_distill.py` once."""
    import ast

    def _calls_to(path: Path, name: str) -> list[int]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and ((isinstance(n.func, ast.Name) and n.func.id == name)
                     or (isinstance(n.func, ast.Attribute)
                         and n.func.attr == name))]

    mine = _calls_to(REPO / "workers/sources/bench_mine.py", "run_prompt_on_primary")
    distill = _calls_to(REPO / "workers/sources/session_distill.py",
                        "run_prompt_on_primary")
    assert len(mine) >= 2, (
        f"bench-mine is expected to mine through `run_prompt_on_primary` at least "
        f"twice; it now calls it {len(mine)} time(s) at {mine}, so the mint stamped "
        "above is no longer the path its turns take")
    assert len(distill) >= 1, (
        f"session-distill is expected to run on `run_prompt_on_primary`; found "
        f"{len(distill)} call site(s) at {distill}")


# ── the rail: no unattended mint is added unstamped again ─────────────
def test_every_session_mint_with_an_unattended_platform_stamps_its_record(tmp_path):
    """Every `create_session(platform="worker"|"autonomy")` call site in the tree has a
    start-cwd stamp after it, found by walking the AST rather than by keeping a list of
    mints by hand.

    This is the rail this clause needed: the first two reviews each caught a mint the
    previous one's list had missed, and the item's own text says why a list cannot work
    (a hand-maintained allowlist cannot close a property over an open set). The
    discriminator is the field that actually decides the risk — a session whose
    `platform` is a string literal of `worker` or `autonomy` is an unattended turn, the
    kind nobody is watching when it writes a relative path — so the walk says nothing
    about a chat mint, whose `platform` is a variable the caller supplies
    (`app/routers/sessions.py:795`), and changing an interactive turn's `pwd` is a
    different decision with a human in it.

    The accepted stamps are the helper (`stamp_new_session`), the backfill that also
    covers an already-minted session (`ensure_session_start_cwd`), and the autonomy
    module's own wrapper (`_stamp_session_start_cwd`). A mint whose stamp call sits
    *before* its `create_session` fails too: `stamp` refuses to mint a record, so an
    earlier stamp writes nothing.

    The denominator is asserted before the node returns. A walk that found nothing would
    pass, and an empty walk is exactly the failure mode this file keeps being refactored
    away from.
    """
    import ast

    unattended = {"worker", "autonomy"}
    accepted = ("stamp_new_session", "ensure_session_start_cwd",
                "_stamp_session_start_cwd")
    skip_parts = {".git", "__pycache__", ".venvs", "node_modules", "tests"}

    mints: dict[tuple[str, int], str] = {}
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        if skip_parts & set(rel.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        def _stamp_lines(node) -> list[int]:
            out = []
            for n in ast.walk(node):
                if not isinstance(n, ast.Call):
                    continue
                name = (n.func.id if isinstance(n.func, ast.Name)
                        else n.func.attr if isinstance(n.func, ast.Attribute) else "")
                if name and any(name.endswith(a) for a in accepted):
                    out.append(n.lineno)
            return out

        # Only the function that OWNS each `create_session` call can carry its stamp,
        # so mints are collected per enclosing def; module level counts as the module.
        owners: list = [tree]
        owners.extend(n for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        for owner in owners:
            creates = []
            for n in ast.walk(owner):
                if not isinstance(n, ast.Call):
                    continue
                fname = (n.func.id if isinstance(n.func, ast.Name)
                         else n.func.attr if isinstance(n.func, ast.Attribute) else "")
                if fname != "create_session":
                    continue
                platform = next((kw.value for kw in n.keywords
                                 if kw.arg == "platform"), None)
                if isinstance(platform, ast.Constant) and platform.value in unattended:
                    creates.append((n.lineno, str(platform.value)))
            if not creates:
                continue
            stamps = _stamp_lines(owner)
            for lineno, platform in creates:
                assert any(s > lineno for s in stamps), (
                    f"{rel}:{lineno} mints a `platform={platform!r}` session and never "
                    f"records a start directory after it (stamp lines in this scope: "
                    f"{stamps}); its Bash inherits the server's cwd, which is the live "
                    "checkout — this is the hole #1906 was opened for")
                mints[f"{rel}:{lineno}"] = platform

    assert len(mints) >= 5, (
        "the walk is only a guard if it is looking at the mints that exist; found "
        f"{len(mints)}: {sorted(mints)}")


# ── the non-default root: a decision, pinned ─────────────────────────
def test_a_stamp_written_to_a_non_default_root_is_read_by_that_root_and_no_other(
        tmp_path, scratch, monkeypatch):
    """`stamp`/`stamp_new_session` take a `sessions_dir`; `resolved_for` does not. That
    asymmetry is the decision the review asked for, and it is pinned in both directions
    so it cannot drift into an accident.

    `scripts/automod/canary.py:224` mints its smoke turn into
    `canary_config.canary_data_root(round_dir) / "sessions"`, and `review.write_session`
    would stamp into whatever root it was handed — so a grader pointed at a canary root
    writes where the live resolver never looks. Threading a root through the resolver was
    rejected: `agent_mcp/builtin_bash.py:144` deliberately calls `resolved_for` with no
    root so it reads the store of the process serving the turn, and guessing by scanning
    other roots would make a session id's start directory depend on which directory a
    `glob` met first. The canary boots its own MCP against its own root, and reads its
    own record there; the live aggregator never serves a turn from that stack.

    So: the stamp lands in the root it was given and NOWHERE else (no copy reaches the
    default store), the stack whose `SESSIONS_DIR` is that root resolves it, and the
    default root's resolver answers `None` — the inherited default, which is what
    clause 2 pins as correct behaviour for a session it has no record of.
    """
    default_store = tmp_path / "default-sessions"
    canary_store = tmp_path / "canary-round" / "sessions"
    canary_store.mkdir(parents=True)
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", default_store)

    session_id = "20260930_000000_canary_smoke_deadbeef"
    (canary_store / f"{session_id}.json").write_text(
        json.dumps({"session_id": session_id, "platform": "canary"}), encoding="utf-8")

    written = session_cwd.stamp_new_session(session_id, sessions_dir=canary_store)
    assert written is not None, "a mint into a non-default root still gets its directory"
    assert Path(written).is_dir() and session_cwd.outside_live(Path(written)), written
    record = _read(canary_store / f"{session_id}.json")
    assert record.get("start_cwd") == str(Path(written).resolve()), record
    assert not (default_store / f"{session_id}.json").exists(), (
        "stamping a non-default root must not mint a record in the default one")

    # The stack that reads that root resolves it.
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", canary_store)
    assert session_cwd.resolved_for(session_id) == str(Path(written).resolve())

    # The default root's resolver — the one the Bash tool actually calls — does not.
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", default_store)
    assert session_cwd.resolved_for(session_id) is None, (
        "the live resolver found a record outside the store it serves: it must answer "
        "for its own process's SESSIONS_DIR and no other")


def test_the_canary_smoke_turn_starts_outside_the_live_checkout_without_a_stamp(
        tmp_path, monkeypatch):
    """The canary is the one mint this change leaves unstamped, and it is not a hole:
    its stack never starts in the live checkout, so there is nothing for a stamp to fix.

    `scripts/automod/canary.py:224` mints the smoke turn into
    `canary_data_root(round_dir) / "sessions"` — the canary's own data root — and this
    node renders that stack's real supervisord config to show where its processes run:
    both `[program:lloyd-mcp]` and `[program:lloyd-backend]` carry
    `directory={worktree}` (`scripts/automod/canary_config.py:211` and `:225`), so
    supervisor chdirs into the round's worktree before exec'ing the MCP server. An
    unstamped Bash call inherits its server's cwd, so the canary's relative write already
    lands in the round's scratch checkout, which is the property clause 3 asks for.

    Stamping it would buy nothing and cost the isolation the canary data root exists to
    provide: `stamp_new_session` creates its scratch under `SCRATCH_ROOT`, and the process
    that would call it is the gate, whose `SCRATCH_ROOT` is production's
    `~/lloyd-data/session-cwd`. So the assertion cuts both ways — the config must keep
    `directory=`, AND `canary_smoke.run` must keep NOT stamping. If either half changes,
    the reason to revisit is in this docstring.
    """
    import ast
    import inspect

    from scripts.automod import canary_config as cc
    from scripts.automod import canary_smoke as CS

    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setattr(session_cwd, "live_root", lambda: live.resolve())
    round_dir = tmp_path / "round"
    worktree = tmp_path / "wt" / "home" / "lloyd"
    worktree.mkdir(parents=True)

    conf, _sock = cc.write_supervisord_conf(round_dir, worktree,
                                            {"LLOYD_DATA_ROOT": "x"},
                                            tmp_path / "bin" / "python")
    text = conf.read_text(encoding="utf-8")

    def _key(section: str, key: str) -> str | None:
        """supervisord's own reading: a `key=value` line inside one `[section]`. Done
        line by line rather than with a section regex because a blank line is legal
        between two keys, and a regex that stops at one reports a missing key."""
        inside = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                inside = stripped == f"[{section}]"
            elif inside and stripped.startswith(f"{key}="):
                return stripped.split("=", 1)[1].strip()
        return None

    for program in ("lloyd-mcp", "lloyd-backend"):
        started_raw = _key(f"program:{program}", "directory")
        assert started_raw is not None, (
            f"[program:{program}] has no `directory=`, so its server inherits whoever "
            "spawned it — and the canary's smoke turn would then inherit a cwd this "
            "change never set")
        started = Path(started_raw)
        assert started == worktree, (
            f"[program:{program}] starts at {started}, not the round's {worktree}")
        assert session_cwd.outside_live(started), (
            f"the canary's {program} would start in the live checkout: {started}")

    body = ast.parse(inspect.getsource(CS.run))
    stamped = [n.lineno for n in ast.walk(body) if isinstance(n, ast.Call)
               and (n.func.id if isinstance(n.func, ast.Name)
                    else n.func.attr if isinstance(n.func, ast.Attribute) else "")
               .endswith(("stamp_new_session", "ensure_session_start_cwd"))]
    assert not stamped, (
        f"`canary_smoke.run` now stamps at {stamped}: that call runs in the gate, whose "
        "SCRATCH_ROOT is production's ~/lloyd-data/session-cwd, which is the isolation "
        "the canary data root exists to deny — see this node's docstring")


# ── kind 3: the autocode round turn ───────────────────────────────────
def test_opening_a_round_records_its_worktree_as_the_start_cwd(tmp_path, monkeypatch):
    """The item's own proposal for the round kind: a round turn's relative writes
    belong where its diff lives, and its `round_start` row says so too.

    Driven the way an autocode round actually happens: the worker mint stamps a scratch
    on the session first (`workers.sources._common.new_worker_session`), and then
    `round.start` restamps that record with the worktree. The restamp IS the clause, so
    the node asserts the record names the WORKTREE: a `round.start` that never stamped
    leaves the mint's scratch there and fails, and a `stamp` that refused to overwrite
    fails the same way.

    The session is created through the real mint, and the store is redirected rather
    than dodged. The earlier shape of this node hid its assertion behind
    `if record.is_file():`, which was false on every run because `session_cwd.stamp`
    refuses to mint a record it was not given — so the clause rested on a branch that
    never executed.
    """
    from scripts.automod import round as R
    from workers.sources import _common as C
    import app.sessions_io as SIO

    live = tmp_path / "live"
    live.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(live), check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(live), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(live), check=True)
    (live / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=str(live), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=str(live), check=True)

    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(R.S, "require_enabled", lambda action, repo=None: None)
    for name in ("STATE_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(R.S, name, tmp_path / "state")
    monkeypatch.setattr(R.S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(R.W, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(R.W, "LIVE_ROOT", live)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(SIO, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(session_cwd, "SCRATCH_ROOT", tmp_path / "session-cwd")

    session_id = C.new_worker_session(title="implement", source="autocode")
    scratch = Path(session_cwd.read(session_id))
    assert scratch.is_dir(), "the mint gives the session a scratch to start in"

    out = R.start("the round turn starts in its worktree", opened_by="cli",
                  session_id=session_id)

    rows = [json.loads(l) for l in (tmp_path / "promotions.jsonl").read_text()
            .splitlines() if l.strip()]
    start = [r for r in rows if r["event"] == "round_start"][0]
    assert start["start_cwd"] == out["worktree"], start
    assert start["live_untracked_recorded"] is True, start

    record = sessions / f"{session_id}.json"
    assert record.is_file(), "the minted record is where the round must stamp"
    stored = json.loads(record.read_text()).get("start_cwd")
    assert stored == out["worktree"], (
        f"the round turn's Bash would start in {stored!r}, not in its worktree "
        f"{out['worktree']!r} — and the start directory is where a relative write goes")
    assert stored != str(scratch), "the round's stamp did not overwrite the mint's scratch"
    assert session_cwd.outside_live(Path(stored)), stored


def test_a_round_opened_without_a_session_still_records_its_baseline(tmp_path,
                                                                     monkeypatch):
    """`round.start` with no `session_id` has no record to stamp, and must still write
    the untracked set it opened against — clause 4's subtraction needs a baseline even
    on the rounds clause 3 cannot cover."""
    from scripts.automod import round as R

    live = tmp_path / "live"
    live.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(live), check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(live), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(live), check=True)
    (live / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=str(live), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=str(live), check=True)
    (live / "already-there.txt").write_text("pre-existing\n", encoding="utf-8")

    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(R.S, "require_enabled", lambda action, repo=None: None)
    for name in ("STATE_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(R.S, name, tmp_path / "state")
    monkeypatch.setattr(R.S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(R.W, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(R.W, "LIVE_ROOT", live)

    out = R.start("baseline with no session to stamp", opened_by="cli")

    rows = [json.loads(l) for l in (tmp_path / "promotions.jsonl").read_text()
            .splitlines() if l.strip()]
    start = [r for r in rows if r["event"] == "round_start"][0]
    assert "session_id" not in start, start
    assert start["live_untracked"] == ["already-there.txt"], start
    assert start["live_untracked_count"] == 1, start
    assert session_cwd.read(out.get("session_id") or "") is None, (
        "a round with no session must not invent a start directory for one")
