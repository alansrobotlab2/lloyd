"""#4163: a round boots ONE fixture-rooted child aggregator for its writes card.

Every test here is about the seam a scored writes-trial row does not exist without: the
trial's write executes in the SERVER process, so the two halves of the round's claim —
the `/state` URL the runner reads and the `mcp_servers` route the harness dials its tools
through — must name ONE child booted over ONE copied fixture. A round that points one
half and not the other is a round that reads the child's answer and writes somewhere
else, which is the failure `require_copied_state` (#3403) exists to refuse.

No model traffic and no real spawner: `boot_writes_trial_child` is exercised against
faked `build_fixture`/`spawn` whose signatures are the REAL ones (`build_fixture` takes
`fixtures_parent`, `spawn` takes `fixture_root` and `scopes` and nothing else, and
`BenchAggregator.url` is an origin with no path on it — a double with a different shape
would pass a caller that could never run). What these DO prove is the round's own wiring:
one boot, one admitted card, both halves naming the same child, the harness actually
being handed that route, and the child plus its fixture gone on every exit path.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.autoresearch import bench_runner_sdk as sdk  # noqa: E402
from scripts.autoresearch import run_round  # noqa: E402
from tests.test_autoresearch_round_report import round_env as _report_round_env

# The report file's scratch-root fixture, aliased here rather than re-derived: pytest
# collects a fixture by scanning module attributes, so a second name for it works, and it
# builds the tmp `AutoresearchConfig` and replaces `_run_trials`, so nothing in this file
# can reach the live ledger or a real model.
round_env = _report_round_env

REPO = Path(__file__).resolve().parent.parent
RUN_ROUND = REPO / "scripts" / "autoresearch" / "run_round.py"
SDK = REPO / "scripts" / "autoresearch" / "bench_runner_sdk.py"

#: The child's ORIGIN. `bench_spawner.BenchAggregator.url` is `http://127.0.0.1:PORT`
#: with no path — its own docstring says the caller appends `/mcp` — so a double that
#: carried a path would not be the handle this code gets.
CHILD_ORIGIN = "http://127.0.0.1:45999"
CHILD_URL = CHILD_ORIGIN + "/mcp"
CHILD_STATE = CHILD_ORIGIN + "/state"
#: What the supervised server resolves to on this box when nothing is repointed:
#: `config.yaml mcp_servers.lloyd-mcp.url`, which is what `resolve_state_url()` and
#: `effective_mcp_route()` both read through `app.mcp_discovery._get_mcp_servers`.
SUPERVISED = "http://127.0.0.1:8500/mcp"
OTHER_SERVER = "http://127.0.0.1:9101/mcp"
CARD = "bench_029_durable_write_vault_note"
NEIGHBOUR = "bench_030_durable_write_kg_row"


@pytest.fixture
def supervised_route(monkeypatch):
    """The box's configured MCP mapping: `lloyd-mcp` on 8500 plus a neighbour server."""
    # The shape `app.mcp_discovery._get_mcp_servers` yields on this box, and the same
    # one `tests/test_autoresearch_bench_runner_sdk.py` stubs it with: a dict entry with
    # a url, whose transport `mcp_pool._apply_default_transports` fills in. No inline
    # `{"type": …}` here — `tests/test_mcp_layer.py` keeps those out of the tree.
    servers = {"lloyd-mcp": {"url": SUPERVISED}, "other": {"url": OTHER_SERVER}}
    import app.mcp_discovery as mcp_disc
    monkeypatch.setattr(mcp_disc, "_get_mcp_servers", lambda: dict(servers))
    monkeypatch.delenv(sdk.STATE_URL_ENV, raising=False)
    return servers


class _FakeFixture:
    def __init__(self, root: Path):
        self.root = root
        self.cleanups = 0

    def cleanup(self) -> Path:
        self.cleanups += 1
        return self.root


class _FakeHandle:
    """Shaped like `bench_spawner.BenchAggregator`: `url` is an origin with no path, and
    `env` is the child's own environment — the only place its credential is named."""

    def __init__(self, url: str = CHILD_ORIGIN, env: dict[str, str] | None = None):
        self.url = url
        self.env = env if env is not None else {}
        self.stops = 0

    def stop(self):
        self.stops += 1
        return 0, 0.16


class _BootCall:
    """Records what a faked build/spawn pair was handed, and how often."""

    def __init__(self, fixture: _FakeFixture, handle: _FakeHandle):
        self.fixture = fixture
        self.handle = handle
        self.builds: list[dict] = []
        self.spawns: list[dict] = []

    def install(self, monkeypatch):
        # **kwargs only, with the REAL parameter names: a fake that accepted a `data_root`
        # or a `vault` the builder does not take would hide a caller that could not run.
        def _build(**kw):
            self.builds.append(kw)
            return self.fixture

        def _spawn(**kw):
            self.spawns.append(kw)
            return self.handle

        monkeypatch.setattr("scripts.autoresearch.bench_fixture.build_fixture", _build)
        monkeypatch.setattr("scripts.autoresearch.bench_spawner.spawn", _spawn)


#: What the child's env names as its credential file, and the token inside it. The child
#: mints its OWN token under the fixture data root (`bench_spawner.py:151-164` is explicit
#: that the process which minted it is the fixture child), so this is not the supervised
#: process's secret and cannot be resolved from its environment.
CHILD_TOKEN = "childtokenabcdef0123456789"


@pytest.fixture
def boot_env(monkeypatch, tmp_path, supervised_route):
    from agent_mcp.aggregator_auth import AUTH_HEADER
    token_file = tmp_path / "child-aggregator-token"
    token_file.write_text(CHILD_TOKEN + "\n", encoding="utf-8")
    fixture = _FakeFixture(tmp_path / "fixtures" / "round-R_T_1_ab12cd34ef56")
    handle = _FakeHandle(env={"LLOYD_MCP_TOKEN_FILE": str(token_file)})
    handle.auth_header_name = AUTH_HEADER
    call = _BootCall(fixture, handle)
    call.install(monkeypatch)
    cfg = run_round.load_config()
    return cfg, call, fixture, handle


# ── clause 1: one boot per round, over a fixture scoped to the card's writes ──────
def _handle_auth_name(handle) -> str:
    from agent_mcp.aggregator_auth import AUTH_HEADER
    return handle.auth_header_name or AUTH_HEADER


def test_boot_builds_one_fixture_for_the_card_scopes_and_boots_it(boot_env):
    cfg, call, fixture, _handle = boot_env
    boot = run_round.boot_writes_trial_child(cfg, card=CARD, scopes=["vault"],
                                             round_id="R_T_1")
    assert len(call.builds) == 1 and len(call.spawns) == 1, (
        "one round means one copy and one child: a second of either is a second copy of "
        "the user's memory (or a second process holding its roots) per trial")
    assert call.builds[0]["writes"] == ["vault"], (
        "the copy is scoped to what the admitted card declares, not to every writes "
        "scope the box has")
    assert call.builds[0]["fixtures_parent"].name == "fixtures", (
        "the copy goes under <data_root>/eval/fixtures, the one area `cleanup()` and "
        "`leftover_fixtures` and the orphan sweep all agree to look at")
    # The tool route carries the CHILD's credential, read from the child's own env. The
    # runner's usual resolver (`app/harness/mcp_pool.py:655` -> `headers_for_url(url)`)
    # reads THIS process's environment, so a route built without it carries the supervised
    # token, which the child rejects: 401, and `bench_spawner.py:439-447` records a first
    # real boot being read as "the aggregator never came up" for exactly that. A repointed
    # route that cannot authenticate is a repoint that never dials, so clause 2's "same
    # child" would be true of the URL and false of the connection.
    assert boot["mcp_servers"]["lloyd-mcp"]["headers"] == {
        _handle_auth_name(_handle): CHILD_TOKEN}, (
        "the repointed route must carry the credential the child actually accepts")
    # The spawner's real signature is `(fixture_root, scopes)` and nothing else: the
    # roots travel inside the fixture's own env, so a `data_root=` here would be a
    # second, ignored answer to which roots the child gets.
    assert set(call.spawns[0]) == {"fixture_root", "scopes"}, call.spawns[0]
    assert call.spawns[0]["fixture_root"] is fixture
    assert call.spawns[0]["scopes"] == ["vault"]
    # Both halves of the round's claim name this one child, derived from the origin the
    # handle carries — which carries no path of its own.
    assert boot["url"] == CHILD_URL
    assert boot["state_url"] == CHILD_STATE
    assert boot["mcp_servers"]["lloyd-mcp"]["url"] == CHILD_URL
    assert sdk.url_origin(boot["state_url"]) == sdk.url_origin(boot["url"]), (
        "same origin: one process answers both halves, so the state that was asked about "
        "is the state the write lands in")


def test_the_repointed_route_moves_one_entry_and_keeps_the_rest(boot_env):
    cfg, call, _fixture, _handle = boot_env
    boot = run_round.boot_writes_trial_child(cfg, card=CARD, scopes=["vault"],
                                             round_id="R_T_1b")
    servers = boot["mcp_servers"]
    assert servers["other"]["url"] == OTHER_SERVER, (
        "a writes trial's tool surface must differ from an ordinary round's in exactly "
        "one entry, or the comparison is about the servers the harness lost")
    assert set(servers["lloyd-mcp"]) == {"url", "headers"}, (
        "only the origin and the credential move: every other key the box declared "
        "survives as declared, and no transport is written inline here — the pool's "
        "`_apply_default_transports` fills that in, and `tests/test_mcp_layer.py` keeps "
        "inline `{\"type\": ...}` config out of the tree")


def test_the_child_and_the_runner_share_the_aggregator_credential(tmp_path):
    """Route repointed at a child on another port, and NO header.

    `boot_env` hands the builder a fresh port each call, so a fixture carrying its own
    token would make this node fail — which is the point: the header-less entry below is
    only correct while the credential is shared.

    The `BenchFixture.env()` claim goes through the real builder, not a transcription: an
    earlier cut of this node hand-built a two-key dict, which is not what `env()` returns
    and asserted nothing about the builder's actual output. `env()` is a no-argument
    method, so the fixture is built and asked. The last assertion is the sensitivity check
    the review asked for — add a private token file to that same dict and the two resolved
    paths MUST differ. An assertion that cannot fail is worse than none: the review caught a
    vacuous one here, and that shape is what this node now pins.
    """
    from agent_mcp import aggregator_auth
    from agent_mcp.aggregator_auth import TOKEN_FILE_ENV
    from scripts.autoresearch.bench_fixture import build_fixture, fixture_parent

    shared = aggregator_auth.token_path({})
    src_data = tmp_path / "src-data"
    src_data.mkdir()
    fixture = build_fixture(trial_id="credential-4163", writes=["vault"],
                            vault=_seeded_source_vault(tmp_path),
                            data_root=src_data,
                            fixtures_parent=fixture_parent(src_data))
    try:
        real_env = fixture.env()
        assert real_env["LLOYD_DATA"] == str(fixture.data_root), (
            "env() must return the copy's own real paths, not a caller's transcription")
        assert not any(k in real_env for k in (aggregator_auth.TOKEN_FILE_ENV,
                                               aggregator_auth.TOKEN_ENV)), (
            "the fixture env no longer hands the child a private credential, so the shared "
            "file is the only answer and the route needs no header")
        assert aggregator_auth.token_path(real_env) == shared, (
            "both processes resolve the same token file: the child accepts what the pool "
            "will send for `child_url`")

        # Sensitivity: a fixture that DID get its own token must break that equality, or
        # nothing above was actually observed.
        with_own_token = dict(real_env)
        with_own_token[TOKEN_FILE_ENV] = str(tmp_path / "child-token")
        assert aggregator_auth.token_path(with_own_token) != shared, (
            "a fixture given its own token file must resolve somewhere else; if this "
            "assertion cannot fail then the one above never compared anything")
    finally:
        fixture.cleanup()


def test_a_child_that_never_boots_leaves_no_fixture_behind(boot_env, monkeypatch):
    cfg, call, fixture, _handle = boot_env

    def _refuse(**_kw):
        raise RuntimeError("SpawnRefused: the child reported production roots")

    monkeypatch.setattr("scripts.autoresearch.bench_spawner.spawn", _refuse)
    with pytest.raises(RuntimeError, match="production roots"):
        run_round.boot_writes_trial_child(cfg, card=CARD, scopes=["vault"],
                                          round_id="R_T_2")
    assert call.builds and fixture.cleanups == 1, (
        "a child that never answered is still a copy on disk, and the copy is the thing "
        "a later round must not find")


def test_run_boots_once_outside_any_loop():
    tree = ast.parse(RUN_ROUND.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "run")
    boots = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") in
             ("boot_writes_trial_child", "teardown_writes_trial_child")]
    assert len(boots) == 2, (
        f"run() must boot once and tear down once, not per trial; found {len(boots)}")
    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.AsyncFor, ast.While))]
    inside = [n for n in boots for lp in loops if lp.lineno <= n.lineno <= lp.end_lineno]
    assert not inside, "the boot sits outside every loop: one child per round, not per arm"


def test_the_admitted_trial_carries_the_bundle_and_only_that_trial_gets_it(
        monkeypatch, supervised_route):
    seen: list[tuple[str, dict | None]] = []

    async def _fake_run_trial(task, variant_id, *_a, **kw):
        seen.append((task["id"], kw.get("writes_child")))
        return {"variant_id": variant_id, "task_id": task["id"], "harness": "sdk",
                "status": "ok", "duration_seconds": 148.2, "messages": [],
                "tool_calls": []}

    monkeypatch.setattr(sdk, "run_trial", _fake_run_trial)
    child = {"task_id": CARD, "state_url": CHILD_STATE, "fixture_dir": "/tmp/fix-1",
             "mcp_servers": {"lloyd-mcp": {"url": CHILD_URL}}}
    traces = asyncio.run(sdk.run_bench_sdk(
        None, [("v0", Path("/overlay"))],
        [{"id": CARD, "prompt": "x"}, {"id": NEIGHBOUR, "prompt": "y"}],
        model="m", writes_trial=child))
    assert len(traces) == 2 and len(seen) == 2
    by_task = dict(seen)
    assert by_task[CARD] == child, (
        "the admitted card's trial gets all three pointers as one bundle")
    assert by_task[NEIGHBOUR] is None, (
        "at most ONE writes trial per round sees the child: the second card keeps the "
        "pre-#4163 path and refuses, which is what keeps its row honest")
    # The pointer is a per-call argument, never the process-wide env: an env pointer is
    # how a second writes card would read the CHILD's /state (green) while its harness
    # dialled the supervised server — the half-repoint clause 2 refuses.
    assert sdk.STATE_URL_ENV not in os.environ, (
        "the round must not publish a process-wide child pointer at all")

    row = sdk.ledger_row_for(traces[0], {"objective_score": 0.8, "rubric_overall": None,
                                         "composite_score": 0.8}, "R_T_3")
    assert row["variant_id"] == "v0", "a scored row names the variant it scored"
    assert row.get("duration_seconds") == 148.2, (
        "and a non-zero duration: a 0-second trial there is the fabricated-trial shape")


# ── clause 2: the two halves must name ONE process ───────────────────────────────
@pytest.mark.asyncio
async def test_a_state_url_without_the_route_refuses_and_reads_nothing(monkeypatch,
                                                                       supervised_route):
    class _NoHttpx:
        def __getattr__(self, _name):
            raise AssertionError("httpx must not be reached: the refusal is local")

    monkeypatch.setitem(sys.modules, "httpx", _NoHttpx())
    with pytest.raises(sdk.TrialRefused) as exc:
        await sdk.require_copied_state(["vault"], state_url=CHILD_STATE,
                                       mcp_route=SUPERVISED,
                                       measured={"is_production_data": False,
                                                 "is_production_vault": False})
    msg = str(exc.value)
    assert CHILD_STATE in msg and SUPERVISED in msg, (
        "the refusal names BOTH halves, or a reader cannot tell which one moved")
    assert "4163" in msg
    assert exc.value.state["state_is_copied"] is False


def test_a_trial_handed_only_the_child_state_refuses_with_no_scored_row(
        monkeypatch, supervised_route, tmp_path):
    """The whole clause, at the level a real trial runs.

    `run_trial` is called the way `run_bench_sdk` calls it for a writes card the round
    published a `/state` for but did NOT repoint the tool route of — the exact shape of
    the bug: `mcp_servers=None` means `build_options` hands the harness
    `_get_mcp_servers()`, i.e. the supervised server on 8500. The trial must come back
    `refused_unsandboxed`, must never reach the harness, and must write no scored row.
    """
    async def _boom(*_a, **_kw):
        raise AssertionError("the harness must not be reached by a refused trial")

    monkeypatch.setattr(sdk, "require_tool_sandbox", _boom)
    # The round's own copy, borrowed: the fixture it points at is the child's, and it is
    # the round's teardown that removes it, not a refused trial's.
    borrowed = _FakeFixture(tmp_path / "fixtures" / "round-R_T_4_ff00ee11aa22")
    monkeypatch.setattr(
        sdk, "isolation_refusal",
        lambda task, env, trial_id, reuse_dir=None: {
            "env": dict(env), "fixture": borrowed, "flags": {
                "is_production_data": False, "is_production_vault": False},
            "borrowed": reuse_dir is not None})

    async def _half_bundle():
        return await sdk.run_trial(
            {"id": CARD, "prompt": "x", "writes": ["vault"]}, "v0",
            Path("/nonexistent-overlay"), "test-model", 30,
            writes_child={"state_url": CHILD_STATE,
                          "fixture_dir": str(borrowed.root)})

    trace = asyncio.run(_half_bundle())
    assert trace["status"] == "refused_unsandboxed", trace
    assert trace["variant_id"] == "v0"
    assert "4163" in trace["error"] and SUPERVISED in trace["error"], trace["error"]
    assert borrowed.cleanups == 0, (
        "a refused trial must not delete the copy the round's child is running on")
    row = sdk.ledger_row_for(trace, {"objective_score": 0.0, "rubric_overall": None,
                                     "composite_score": None}, "R_T_4")
    assert row["trace_status"] == "refused_unsandboxed", (
        "the row says refusal, and is not a scored per-trial row for the card")
    assert row["composite_score"] is None and row["variant_id"] == "v0"


@pytest.mark.asyncio
async def test_a_config_resolved_state_url_beside_an_unrepointed_route_is_no_claim(
        monkeypatch, supervised_route):
    """#3403's hand-run workflow survives the pairing check, which is why it is conditional.

    An operator pointing one trial at a fixture-rooted aggregator publishes
    `LLOYD_BENCH_STATE_URL` and leaves discovery alone, so config still names the
    supervised origin — two different processes, and no claim was made about them being
    one. Refusing that would refuse the workflow #3403 shipped (and the ten scoped-trial
    nodes in `tests/test_autoresearch_bench_runner_sdk.py` drive exactly it), so the
    comparison fires only when a caller handed over a half of the pair.
    """
    monkeypatch.setenv(sdk.STATE_URL_ENV, CHILD_STATE)
    body = {"tool_sandbox": {
        "enforced": True, "bwrap": True,
        "is_production_data": False, "is_production_vault": False,
        "state_is_copied": True,
        "write_scope": sorted(sdk.WRITE_SCOPE_TOOLS),
        "allowed_writes": sorted(sdk.WRITE_SCOPE_TOOLS)}}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return body

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def get(self, _url, headers=None):
            return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _Client())
    state = await sdk.require_copied_state(
        sdk.write_scope_of({"writes": ["vault"]}),
        measured={"is_production_data": False, "is_production_vault": False})
    assert state["state_is_copied"] is True, (
        "the child's own answer stands on its own: config naming another origin is not a "
        "half anyone repointed, so there is no pair to disagree")


def test_build_options_hands_the_harness_the_route_it_was_given(supervised_route,
                                                                tmp_path):
    """The seam the review named: the mapping reaches `RunOptions`, or nothing moved.

    `build_options` is where the harness gets `mcp_servers` (#4163's repoint is a code
    change here, not a `config.yaml` edit), and every other caller must keep resolving
    the supervised mapping unchanged.
    """
    plain = sdk.build_options(model="primary", overlay_dir=tmp_path,
                             session_id="bench_4163_plain")
    assert plain.mcp_servers["lloyd-mcp"]["url"] == SUPERVISED, (
        "a trial nobody repointed still dials the supervised server, exactly as before")
    moved = sdk.build_options(model="primary", overlay_dir=tmp_path,
                             session_id="bench_4163_child",
                             mcp_servers={"lloyd-mcp": {"url": CHILD_URL},
                                          "other": {"url": OTHER_SERVER}})
    assert moved.mcp_servers["lloyd-mcp"]["url"] == CHILD_URL
    assert moved.mcp_servers["other"]["url"] == OTHER_SERVER


def test_an_admitted_trial_reuses_the_rounds_copy_and_builds_no_second_one(tmp_path):
    """The second copy is the bug the boot has to remove, not merely pay for.

    The write executes in the SERVER process against the roots THAT process resolved, so
    a runner-side copy is a bystander: the assertion at the end of the trial would read a
    tree the write never touched. Measured here with the real builder over two throwaway
    source trees, so "no second copy" is a directory listing and not a claim.
    """
    from scripts.autoresearch import bench_fixture as _fx

    src_vault = tmp_path / "vault"
    src_data = tmp_path / "data"
    src_data.mkdir(parents=True)
    (src_vault / "memory").mkdir(parents=True)
    (src_vault / "memory" / "MEMORY.md").write_text("# index\n", encoding="utf-8")
    (src_vault / "USER.md").write_text("- a fact\n", encoding="utf-8")
    (src_data / ".lloyd-data").write_text("", encoding="utf-8")
    parent = src_data / "eval" / "fixtures"
    round_copy = _fx.build_fixture(trial_id="round-R_T_5_ab12cd34ef56",
                                   writes=["vault"], vault=src_vault,
                                   data_root=src_data, fixtures_parent=parent)
    before = sorted(p.name for p in parent.iterdir())
    assert before == ["round-R_T_5_ab12cd34ef56"], before

    task = {"id": CARD, "writes": ["vault"]}
    try:
        borrowed = sdk.build_task_fixture(task, "bench_v0_bench_029_x",
                                          str(round_copy.root))
        assert Path(borrowed.root) == Path(round_copy.root), (
            "the trial must be handed the copy its own child is rooted in")
        assert sorted(p.name for p in parent.iterdir()) == before, (
            "and must not build a second copy of the memory alongside it: ~5.2 s of the "
            "trial's cost and a bystander where the assertion should be")
        assert borrowed.env()["LLOYD_DATA"] == round_copy.env()["LLOYD_DATA"]
    finally:
        round_copy.cleanup()


# ── clause 5: torn down on every path, and a leftover fails the round ────────────
def test_teardown_stops_the_child_and_removes_the_fixture(boot_env):
    cfg, call, fixture, handle = boot_env
    out = run_round.teardown_writes_trial_child(
        {"handle": handle, "fixture": fixture})
    assert handle.stops == 1 and fixture.cleanups == 1
    assert out["stop"]["code"] == 0 and out["fixture_removed"].endswith(
        "round-R_T_1_ab12cd34ef56"), "the teardown reports both ends it closed"


def test_a_raised_trial_still_stops_the_child_and_removes_the_fixture(
        round_env, monkeypatch):
    """Clause 5 on the production `run()` path: the teardown sits in a `finally`.

    The double raises through the matrix rather than returning a refusal, because a
    raise is the exit path a `finally` is for. Both halves must still close: a child left
    alive holds a port and the copy's roots, and a copy left behind is a second copy of
    the memory under the fixtures area.
    """
    fixture = _FakeFixture(Path(str(round_env.paths.ledger_path)) / "stranded")
    handle = _FakeHandle()
    boot = {"handle": handle, "fixture": fixture, "card": CARD, "scopes": ["vault"],
            "url": CHILD_URL, "state_url": CHILD_STATE,
            "mcp_servers": {"lloyd-mcp": {"url": CHILD_URL}}}
    monkeypatch.setattr(run_round, "pick_writes_card",
                        lambda task_ids, *, round_index: CARD)
    monkeypatch.setattr(run_round, "writes_declared", lambda task: ["vault"])
    monkeypatch.setattr(run_round, "boot_precondition", lambda **_kw: None)
    monkeypatch.setattr(run_round, "boot_writes_trial_child",
                        lambda cfg, *, card, scopes, round_id: boot)

    async def _raise(*_a, **_kw):
        raise RuntimeError("variant overlay missing from disk")

    monkeypatch.setattr(run_round, "_run_trials", _raise)
    with pytest.raises(RuntimeError, match="overlay missing"):
        asyncio.run(run_round.run(targets=["memory"], budget_minutes=5,
                                  max_variants=1, model="m-test"))
    assert handle.stops == 1 and fixture.cleanups == 1, (
        f"child stops={handle.stops}, fixture cleanups={fixture.cleanups}")


def test_a_fixture_left_under_the_fixtures_area_is_visible_to_the_round(tmp_path):
    (tmp_path / "eval" / "fixtures" / "round-R_T_9_cc00dd11ee22").mkdir(parents=True)
    (tmp_path / "eval" / "fixtures" / "dw-20261116-050000-bench_029-x").mkdir(
        parents=True, exist_ok=True)
    assert run_round.leftover_fixtures(tmp_path, before=set()) == [
        "dw-20261116-050000-bench_029-x", "round-R_T_9_cc00dd11ee22"], (
        "a copy the round's own teardown lost track of is a second copy of the user's "
        "memory; the round has to be able to SEE it to fail on it — every directory in "
        "the area, whatever name its builder gave it")
    assert run_round.leftover_fixtures(tmp_path / "nothing", before=set()) == []


def test_main_exits_nonzero_on_a_leftover_fixture(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_round"])

    async def _round(**_kw):
        return {"round_id": "R_T_9",
                "leftover_fixtures": ["round-R_T_9_cc00dd11ee22"]}

    monkeypatch.setattr(run_round, "run", _round)
    with pytest.raises(SystemExit) as exc:
        run_round.main()
    assert exc.value.code == 1
    assert "round-R_T_9_cc00dd11ee22" in capsys.readouterr().err, (
        "the verdict has to be in the exit status and the message, not only in the "
        "printed payload a worker never reads")


# ── clauses 1 and 2 on the live rails: nothing in these two nodes is stubbed ──────
#: The card whose scopes these nodes build and boot: `vault` (the note the oracle reads)
#: and `backlog` (the card it updates) — the declaration
#: `bench_029_durable_write_vault_note` carries, and no data-store scope, so the copy never
#: reads a live store while the process that owns it is writing to it.
CARD029_SCOPES = ["vault", "backlog"]
CARD029_TOOLS = {"vault_write", "memory_add", "memory_replace", "memory_remove",
                 "backlog_write_task"}


def _seeded_source_vault(tmp: Path) -> Path:
    """A source vault with something in each declared scope.

    `build_fixture` refuses a declared vault scope whose source directory is absent, so the
    seed is also the check that the copy is scoped to what the card asked for.
    """
    vault = tmp / "src-vault"
    (vault / "lloyd" / "topics").mkdir(parents=True)
    (vault / "backlog").mkdir()
    (vault / "lloyd" / "MEMORY.md").write_text("# index\n- one line\n", encoding="utf-8")
    (vault / "lloyd" / "topics" / "seed.md").write_text("# seed\n", encoding="utf-8")
    (vault / "backlog" / "4163-card.md").write_text("# card\n", encoding="utf-8")
    return vault


def test_the_old_builder_arguments_are_refused_by_the_rails_as_they_stand(tmp_path):
    """The negative control node A depends on: the rails are live, not decoration.

    These are the arguments the previous round's boot passed — `data_root=DATA_ROOT`,
    `fixtures_parent=fixture_parent(DATA_ROOT)` — verbatim, with `DATA_ROOT` set to the
    production data root, which is what `app.paths` resolves on a real round. `build_fixture`
    rule (2) refuses any destination inside `PRODUCTION_DATA_ROOT`, so the boot raised on
    every call and never produced a child; the test that would have caught it patched the
    production roots away, which is why the clause came back unmet twice instead of red. If
    these rails are ever relaxed, node A below stops proving anything — that is what this
    node is here to say out loud.
    """
    from app.data_root import PRODUCTION_DATA_ROOT
    from scripts.autoresearch.bench_fixture import FixtureRefused, build_fixture, fixture_parent

    with pytest.raises(FixtureRefused) as refused:
        build_fixture(trial_id="negctl-4163", writes=CARD029_SCOPES,
                      vault=_seeded_source_vault(tmp_path),
                      data_root=PRODUCTION_DATA_ROOT,
                      fixtures_parent=fixture_parent(PRODUCTION_DATA_ROOT))
    assert "PRODUCTION_DATA_ROOT" in str(refused.value), (
        f"refused for a reason other than the production root: {refused.value}")


def test_a_production_rooted_round_boots_a_real_child_it_can_actually_write_to(tmp_path,
                                                                              monkeypatch):
    """Clauses 1 and 2 together, over two real process/filesystem boundaries.

    The real `build_fixture` and the real `bench_spawner.spawn`: a child process
    (`python -m agent_mcp.main`) booted over the copy, its own `GET /state` answered over
    loopback. Nothing here is faked at that seam — `state_is_copied: true` is the child
    reporting what IT resolved, and the armed grant is read back as TOOL NAMES because
    `require_copied_state` refuses a trial whose server honours none of them.

    The case is a round whose data root IS the production one (`data_root=PRODUCTION_DATA_ROOT`,
    which is what `app.paths.DATA_ROOT` answers on this box for a real round). That is the
    case the previous boot always raised in, and no node here reached it before: leaving the
    argument unset makes the boot answer for a scratch-rooted round, which is the easy case.
    """
    from app.data_root import PRODUCTION_DATA_ROOT, PRODUCTION_VAULT

    monkeypatch.setattr("app.paths.VAULT_ROOT", _seeded_source_vault(tmp_path))
    area = tmp_path / "area"
    monkeypatch.setattr(run_round, "WRITES_FIXTURE_AREA", area)

    boot = run_round.boot_writes_trial_child(
        run_round.load_config(), card=CARD, scopes=list(CARD029_SCOPES),
        round_id="R_4163_1", data_root=PRODUCTION_DATA_ROOT)
    try:
        root = Path(boot["fixture"].root)
        # Clause 1: a handle, and a fixture root outside BOTH live roots. Not inside
        # PRODUCTION_DATA_ROOT is what the rails guard; not inside PRODUCTION_VAULT is the
        # same claim about the other half of the user's memory.
        assert boot["handle"] is not None
        assert not root.is_relative_to(PRODUCTION_DATA_ROOT), root
        assert not root.is_relative_to(PRODUCTION_VAULT), root
        assert root.is_relative_to(area / "eval" / "fixtures"), (
            f"the copy is not in the area the round censuses for leftovers: {root}")
        assert (boot["fixture"].vault / "backlog" / "4163-card.md").exists(), (
            "the declared `backlog` scope must be a real copy of <vault>/backlog inside the "
            "copy's own vault, not a directory the trial's backlog_write then invents")

        # Clause 2: the child's own answer about itself.
        state = boot["handle"].child_isolation()
        block = state["sandbox"]
        assert block["state_is_copied"] is True, block
        assert block["is_production_data"] is False and block["is_production_vault"] is False, block
        assert set(block["allowed_writes"]) == CARD029_TOOLS, block

        # Clause 3's credential, read the way the harness will: the header the repointed
        # route carries is the one this socket accepts. `/state` is not an open path, so a
        # wrong credential here answers 401 — the same 401 `bench_spawner.py:439-447`
        # records a first real boot being misread as "never came up".
        headers = boot["mcp_servers"]["lloyd-mcp"]["headers"]
        request = urllib.request.Request(boot["state_url"], headers=dict(headers))
        with urllib.request.urlopen(request, timeout=10) as response:
            published = json.loads(response.read().decode("utf-8", "replace"))
        assert published["tool_sandbox"]["state_is_copied"] is True, published
        assert boot["url"] == boot["handle"].url + "/mcp", (
            "the tool route and the state URL name one origin, and that origin is the "
            "handle's — neither half is resolved a second time from anywhere else")
    finally:
        code, _up = boot["handle"].stop()
        boot["fixture"].cleanup()
    assert boot["handle"].proc.poll() is not None, "stop() left the child running"
    assert not Path(boot["fixture"].root).exists(), (
        "cleanup() must remove the copy: it is the user's memory in a second location")
