"""#3688 clause 1: the spawner boots lloyd-mcp on ONE trial's fixture roots, scope armed.

The process boundary is the whole subject: a real `python -m agent_mcp.main` child, its own
`GET /state` answered over loopback with the credential. Nothing here is faked at that seam —
the flags come from the child's own `_tool_sandbox.status()`, so an answer that reads
`state_is_copied: true` is the child reporting what IT resolved, not what this test hoped.
"""
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.autoresearch import bench_spawner as spawner  # noqa: E402
from scripts.autoresearch.bench_fixture import build_fixture  # noqa: E402

# `vault`, not `memory`: `build_fixture` copies the `memory` scope from `<vault>/memory`, while
# `memory_add` writes `<vault>/lloyd/` (`agent_mcp/session.py:58`) — so a card scoped `memory`
# alone gets a copy of a directory the tool never touches. Recorded as a finding on #3688.
SCOPES = ("vault", "backlog")


@pytest.fixture
def fixture(tmp_path):
    src_vault = tmp_path / "src-vault"
    (src_vault / "lloyd").mkdir(parents=True)
    (src_vault / "memory").mkdir()
    (src_vault / "lloyd" / "MEMORY.md").write_text("# seed\n", encoding="utf-8")
    (src_vault / "backlog").mkdir()
    src_data = tmp_path / "src-data"
    src_data.mkdir()
    fx = build_fixture(trial_id="spawn-probe", writes=list(SCOPES),
                       vault=src_vault, data_root=src_data)
    yield fx
    fx.cleanup()


def test_the_spawned_child_reports_a_copy_with_the_declared_scope_armed(fixture):
    """`GET /state` on the booted child: state_is_copied true, both flags false, scope armed.

    This is the witness the item asks for as `curl <url>/state | jq .tool_sandbox.state_is_copied`,
    asserted from inside a test. The armed grant is asserted as TOOL NAMES — `vault_write`,
    `memory_add`, … — because `_tool_sandbox.write_scope_tools()` parses `LLOYD_BENCH_WRITE_SCOPE`
    as tool names and `require_copied_state` refuses a trial whose server honours none of them. The
    first cut of this node only compared `write_scope` to the declared scope words, which the child
    echoes back verbatim whatever the env carried, so it stayed green while the grant was
    unintelligible to the gate and every durable-write trial was dead on arrival. Asserting the
    exact gated set is also the no-widening check: a name beyond the two declared scopes would let
    an agent write a store its card never asked for.

    The two raw reads below are the same endpoint WITHOUT and WITH the credential, because the
    acceptance reading is a `curl` and `/state` is not an open path: `OPEN_PATHS` is
    `{"/health"}`, so the unauthenticated read is a 401 and only the credentialed one is the
    witness. Asserting the 401 is what keeps the published `witness_command` honest — a probe
    that treated 401 as "not ready yet" would report a boot that never happened.
    """
    from scripts.autoresearch.bench_runner_sdk import WRITE_SCOPE_TOOLS, write_scope_of

    expected = set(write_scope_of({"writes": list(SCOPES)}))
    assert expected == {"vault_write", "memory_add", "memory_replace", "memory_remove",
                        "backlog_write_task"}, expected
    handle = spawner.spawn(fixture_root=fixture, scopes=SCOPES, repo_root=REPO)
    try:
        state = handle.child_isolation()
        assert "sandbox" in state, state
        block = state["sandbox"]
        assert block["state_is_copied"] is True, block
        assert block["is_production_data"] is False and block["is_production_vault"] is False
        assert set(block["allowed_writes"]) == expected, block
        assert set(block["write_scope"]) <= set(WRITE_SCOPE_TOOLS), block
        assert not set(block["allowed_writes"]) & set(SCOPES), (
            f"the child was armed with scope words, not tool names: {block['allowed_writes']}")

        with pytest.raises(urllib.error.HTTPError) as uncredentialed:
            urllib.request.urlopen(handle.url + "/state", timeout=5)
        assert uncredentialed.value.code == 401, uncredentialed.value

        request = urllib.request.Request(
            handle.url + "/state",
            headers={"X-Lloyd-Aggregator-Token":
                     spawner.auth_headers(handle.env)["X-Lloyd-Aggregator-Token"]})
        with urllib.request.urlopen(request, timeout=5) as response:
            published = json.loads(response.read().decode("utf-8", "replace"))
        assert published["tool_sandbox"]["state_is_copied"] is True, published
    finally:
        code, up = handle.stop()
    assert handle.alive is False
    assert code is not None and up > 0.0, (code, up)
    # Clause 2's other half: the child is gone AND nothing still holds the port it bound, so a
    # retried trial gets a clean bind instead of a server nobody owns. `SO_REUSEADDR` is set on
    # the probe socket because uvicorn binds with it, and a socket lingering in TIME_WAIT is a
    # closed connection, not a live listener — the bind below fails only if something is still
    # listening.
    assert handle.proc.poll() is not None, "stop() returned with the child still running"
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", handle.port))


def test_the_spawned_child_admits_a_writes_scoped_trial_through_the_runners_own_gate(
        fixture):
    """The seam a stub cannot fake: `require_copied_state`, the runner's admission gate, run
    against this child over its real socket, with the tool names the declared scopes grant.

    The node above reads the `/state` block; this one asks the question the trial actually asks —
    `is the process that will EXECUTE my writes sitting on a copy, and does it honour the granted
    names` — because that gate is a separate process's judgement and nothing else in this file
    crosses it. It is also where the arming bug lived: with `LLOYD_BENCH_WRITE_SCOPE` set to the
    scope words, this call raised `TrialRefused: the aggregator … honours none of [memory_add,
    memory_remove, memory_replace, vault_write]` and no durable-write trial could ever run, while
    every boot probe stayed green. A stub `/state` that merely lists tool names passes the same
    gate with the spawner uninvolved, which is why this node boots the child.
    """
    import asyncio

    from scripts.autoresearch.bench_runner_sdk import require_copied_state, write_scope_of

    declared = write_scope_of({"writes": list(SCOPES)})
    handle = spawner.spawn(fixture_root=fixture, scopes=SCOPES, repo_root=REPO)
    try:
        reported = asyncio.run(require_copied_state(
            declared, state_url=handle.url + "/state"))
        assert reported["state_is_copied"] is True, reported
        assert (reported["is_production_data"] is False
                and reported["is_production_vault"] is False), reported
        assert set(declared) <= set(reported["allowed_writes"]), reported
    finally:
        handle.stop()


def test_a_fixture_inside_either_live_root_is_refused_before_any_child_starts(tmp_path,
                                                                              monkeypatch):
    """Clause 2's guard runs BEFORE the boot, and refuses the two shapes that would write the house.

    `Popen` is replaced by a node that raises `AssertionError`, so a guard that ran after the
    spawn would fail here as "the guard did not run" rather than as a green test that left a
    scope-armed server listening on the account's own roots. The marker check that
    `fixture_from_dir` performs is deliberately placed AFTER these guards, so matching on the
    live-tree wording — rather than on any RuntimeError — is what pins the order.
    """
    def never_popen(*args, **kwargs):
        raise AssertionError("spawn() reached Popen: the live-root guard ran after the boot")

    monkeypatch.setattr(spawner.subprocess, "Popen", never_popen)
    house = spawner.production_roots()

    # (a) underneath a live root: `<live data root>/eval/fixtures/<name>` is exactly where a
    # fixture built with the REAL data root lands, which is the shape the guard is for.
    inside = house["data"] / "eval" / "fixtures" / "clause2-probe"
    with pytest.raises(RuntimeError, match="resolves inside the live tree"):
        spawner.spawn(fixture_root=inside, scopes=SCOPES)

    # (b) the live root itself, which is not "inside" anything and needs its own arm.
    with pytest.raises(RuntimeError, match="would be the live tree itself"):
        spawner.spawn(fixture_root=house["data"], scopes=SCOPES)

    # (c) a directory that CONTAINS a live root — the parent of the live data root — has to be
    # refused too, because its `cleanup()` would be one rmtree away from the tree the fixture
    # exists to protect. Named off `house`, never off `Path.home()`: the gate runs a round with
    # `$HOME` redirected into the round home, and `PRODUCTION_DATA_ROOT` is an absolute literal,
    # so under a gate `$HOME` is an ordinary sibling directory and this arm would silently reach
    # `fixture_from_dir` and raise the marker error instead.
    with pytest.raises(RuntimeError, match="CONTAINS the live"):
        spawner.spawn(fixture_root=house["data"].parent, scopes=SCOPES)


def test_no_scope_means_no_boot(tmp_path):
    """An unscoped boot is refused: an armed-with-nothing server cannot be told from a broken one."""
    with pytest.raises(RuntimeError, match="no declared write scopes"):
        spawner.spawn(fixture_root=tmp_path, scopes=())


def test_a_directory_that_is_not_a_fixture_is_refused(tmp_path):
    """The `--fixture-root` CLI path refuses a hand-made directory by its missing marker."""
    (tmp_path / "data").mkdir()
    with pytest.raises(RuntimeError, match="not a fixture directory"):
        spawner.spawn(fixture_root=tmp_path, scopes=SCOPES)


def test_an_env_that_lost_its_roots_never_reaches_a_boot(tmp_path):
    """The pre-flight refusal fires on an environment that would route the child to the house."""
    house = spawner.production_roots()
    with pytest.raises(RuntimeError, match="names no vault root"):
        spawner.assert_not_production({"LLOYD_DATA": str(tmp_path / "d")})
    with pytest.raises(RuntimeError, match="IS this account's"):
        spawner.assert_not_production({"LLOYD_DATA": str(house["data"]),
                                       "LLOYD_VAULT_ROOT": str(tmp_path / "v")})
    spawner.assert_not_production({"LLOYD_DATA": str(tmp_path / "d"),
                                   "LLOYD_VAULT_ROOT": str(tmp_path / "v")})
