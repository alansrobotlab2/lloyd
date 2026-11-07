"""The five legs of the planted durable-write proof (#3403 owed 3, item #3689).

One node per acceptance clause. Every one exercises the routine through stand-in trees
under `tmp_path`, because this machine is a running box: manifesting the real
`~/lloyd-data` (709,991 files measured 2026-11-03) inside a unit test would take minutes
and would assert on state another process is rewriting.

Four nodes run the whole pipeline rather than a piece of it: `build_fixture` builds a real
copy, a real subprocess HTTP server stands in for the aggregator, and `run_proof` runs with
the real `fetch_state_json`, the real `mcp_tool_call` and real one-second windows. The
stand-in is stood in for as little as possible: it resolves its stores through the
`app.paths` of the process it was booted into (so pointing it at the live roots reproduces
the routing bug #3403 was filed for), it reports `GET /state` with the real
`_tool_sandbox.status()` nested under `tool_sandbox` exactly as `agent_mcp/main.py:909`
nests it, it checks the credential through the same `aggregator_auth` the server does, and
it refuses an out-of-scope tool the same gate refuses it. One of those nodes is the
leak-negative — every flag PASS-shaped, the writes routed into the live trees — and its
verdict must be FAIL or no PASS this instrument prints would mean anything; another boots the
server on the account's real vault and shows the guard refusing before a byte arrives.

What stays OWED (named, not faked) is the run against the REAL `agent_mcp` under a
fixture's roots: the stand-in server stands in for the TOOL, not for `agent_mcp/main.py`'s
handler, and #3688's spawner does not exist, so `main()` refuses without both urls.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from scripts.autoresearch.bench_fixture import build_fixture

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.autoresearch.bench_durable_write_proof as proof  # noqa: E402

#: The three tool names this trial declares, as the server reports them in `allowed_writes`.
WRITE_SCOPE = ("memory_add", "fact_add", "backlog_write_task")


def state_body(*, is_production_data: bool = False, is_production_vault: bool = False,
               state_is_copied: bool = True, allowed_writes=WRITE_SCOPE, block: bool = True,
               drop: tuple[str, ...] = ()) -> dict:
    """A `GET /state` body shaped like the one the real aggregator sends.

    The flags live INSIDE `tool_sandbox`, which is `_tool_sandbox.status()`
    (`agent_mcp/main.py:909`): a reader that looked for them at the top level would be
    reading a shape no aggregator has ever emitted, which is the mistake the review of the
    previous commit caught. `block=False` is an aggregator that predates #3403 and answers
    with no isolation block at all; `drop=` is one that reports the block but not a required
    key. Both have to refuse, as refusals rather than as clean results.

    The composition nodes never call this: they take their body from the real `status()`
    inside a child process, so the flags there are the real predicate's answer.
    """
    sandbox = {"enforced": True, "bwrap": True, "error": None,
               "write_scope": sorted(allowed_writes),
               "state_is_copied": bool(state_is_copied),
               "allowed_writes": sorted(allowed_writes),
               "is_production_data": bool(is_production_data),
               "is_production_vault": bool(is_production_vault)}
    for key in drop:
        sandbox.pop(key, None)
    body = {"tools": 3, "changes": {"count": 0}}
    if block:
        body["tool_sandbox"] = sandbox
    return body


#: A server that reports copied state: the only state the guard lets through.
COPIED_STATE = state_body()


def _tree(root: Path, name: str, files: dict[str, str]) -> Path:
    """A stand-in tree. An empty `files` still makes the DIRECTORY exist.

    That distinction is the clause-2 case: `~/obsidian/facts` exists and holds 0 files,
    which is a vacuous tree, whereas a root that is not there at all is a different fault.
    """
    tree = root / name
    tree.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tree


# ── clause 1: the manifest reports exactly the paths that changed ───────────

def test_changed_paths_reports_the_planted_pair_and_nothing_else(tmp_path):
    """Two stand-in trees, two planted changes, and the reported set is those two.

    One NEW file and one byte-EDITED file, one in each pair of the four background files.
    The clause is the `and no others` half: an implementation that reported a whole tree
    whenever anything in it moved, or that keyed on size alone, would still leave the
    end-to-end verdict looking reasonable. So the edit is deliberately the SAME byte count
    as the original (`aaaa` -> `baaa`) — a size-keyed manifest reports nothing for it, and
    that miss would otherwise surface only as a missing witness on the one run that was
    supposed to be the artifact.
    """
    memory = _tree(tmp_path, "memory", {"MEMORY.md": "aaaa\n", "topics/a.md": "same\n"})
    backlog = _tree(tmp_path, "backlog", {"1-item.md": "keep\n", "2-item.md": "keep2\n"})
    roots = {"memory": memory, "backlog": backlog}

    before = {name: proof.snapshot_tree(name, root) for name, root in roots.items()}
    (memory / "topics" / "new.md").write_text("brand new\n", encoding="utf-8")
    (memory / "MEMORY.md").write_text("baaa\n", encoding="utf-8")   # same byte count
    after = {name: proof.snapshot_tree(name, root) for name, root in roots.items()}

    for name in roots:
        assert before[name]["refused"] is None and after[name]["refused"] is None
        assert before[name]["unreadable"] == []

    changed = {name: proof.changed_paths(before[name], after[name]) for name in roots}
    reported = {str(path) for paths in changed.values() for path in paths}
    assert reported == {str(memory / "topics" / "new.md"), str(memory / "MEMORY.md")}, (
        "the reported set must be exactly the planted new file and the planted byte-edit: "
        "the two untouched memory files and the whole backlog tree must not appear")
    assert changed["backlog"] == set(), "a tree nobody touched must report no paths"
    untouched = str(memory / "topics" / "a.md")
    assert after["memory"]["manifest"][untouched] == before["memory"]["manifest"][untouched]


# ── clause 2: trees come from app.paths, and a zero-file tree is a refusal ──

def test_trees_resolve_through_app_paths_and_an_empty_tree_is_a_refusal(tmp_path, monkeypatch):
    """The four live roots are `app.paths`-resolved, and `facts` is NOT `~/obsidian/facts`.

    #3689 as filed names `~/obsidian/facts`, which exists on this box holding **0 files**
    while the real root — `app.paths.VAULT_FACTS_ROOT`, `app/paths.py:309-311`, redirectable
    by `LLOYD_FACTS_ROOT` — held 75,562 files on 2026-11-03. Manifesting the literal would have "proved" the
    facts tree clean by reading a directory nothing writes to, which is the exact shape of a
    vacuous pass. Hence the clause's second half: a selected tree resolving to zero files is
    a REFUSAL, not a clean diff.

    `memory` is clause 4 of #3863, and it is the same trap wearing different clothes. It used
    to be `VAULT_ROOT/'memory'`, which is not empty — it holds this vault's dated daily notes
    — so the vacuity refusal above could never fire on it, and it is also not a directory
    `memory_add` can write: the shipped writer resolves `agent_mcp/session.py:58`
    `MEMORIES_ROOT = vault_root() / 'lloyd'`. A leak into `~/obsidian/lloyd/MEMORY.md`, the
    file this instance is loaded from, would therefore have read CLEAN. Hence the three forms
    of the assertion below: the tree equals the patched symbol, the tree is NOT
    `VAULT_ROOT/'memory'`, and on the live tree it equals the shipped attribute itself.

    Resolution happens per call, not at import, so patching the module's attributes proves
    the routine reads them rather than carrying a copy: the stand-in roots below are nowhere
    under `~/obsidian`.
    """
    stand_in = types.SimpleNamespace(VAULT_ROOT=Path("/stand-in/vault"),
                                     VAULT_FACTS_ROOT=Path("/stand-in/facts-tree"),
                                     DATA_ROOT=Path("/stand-in/data"))
    # The writer's own root, re-aimed at the stand-in vault. Patching it and then asserting
    # the tree lands there is what makes this node read the SYMBOL: the pre-#3863 entry was
    # `resolved.VAULT_ROOT / 'memory'`, which under this stub is `/stand-in/vault/memory` —
    # a path that looks exactly as plausible as the right one, which is why the old assertion
    # passed while the instrument watched daily notes.
    monkeypatch.setattr("agent_mcp.session.MEMORIES_ROOT", Path("/stand-in/vault/lloyd"))
    assert proof.durable_write_trees(resolved=stand_in) == {
        "memory": Path("/stand-in/vault/lloyd"),
        "facts": Path("/stand-in/facts-tree"),
        "backlog": Path("/stand-in/vault/backlog"),
        "data": Path("/stand-in/data")}, (
        "memory must be the writer's MEMORIES_ROOT, backlog VAULT_ROOT/'backlog', facts "
        "VAULT_FACTS_ROOT and data DATA_ROOT, all read at call time")
    # The wrong tree, named: equality against the stub vault is not enough, since
    # `/stand-in/vault/memory` and `/stand-in/vault/lloyd` are both vault children.
    assert (proof.durable_write_trees(resolved=stand_in)["memory"]
            != Path("/stand-in/vault/memory")), (
        "memory must NOT be VAULT_ROOT/'memory': that is the daily-notes directory "
        "memory_add cannot write (#3863 clause 4)")

    live = proof.durable_write_trees()
    assert set(live) == {"memory", "facts", "backlog", "data"}
    assert Path(live["facts"]) != Path.home() / "obsidian" / "facts", (
        "facts must be VAULT_FACTS_ROOT: the literal ~/obsidian/facts holds 0 files and "
        "manifesting it proves nothing")
    # And on the real tree, against the real symbol — the equality the clause asks for,
    # checked with the patch lifted so it is the shipped value being compared.
    monkeypatch.undo()
    from agent_mcp import session as agent_session
    assert proof.memories_root() == Path(agent_session.MEMORIES_ROOT) == (
        Path(agent_session.vault_root()) / "lloyd"), (
        "the watched memory tree must be the directory memory_add resolves "
        "(agent_mcp/session.py MEMORIES_ROOT = vault_root()/'lloyd')")
    assert proof.store_dirname("memory") == "lloyd", (
        "the copy-side witness matches directories by this name; 'memory' would match "
        "nothing the tool can write")
    monkeypatch.setattr(proof.paths, "VAULT_FACTS_ROOT", tmp_path / "moved-facts",
                        raising=False)
    assert proof.durable_write_trees()["facts"] == tmp_path / "moved-facts"

    vacuous = proof.snapshot_tree("facts", _tree(tmp_path, "vacuous", {}))
    assert vacuous["refused"] and "zero files" in vacuous["refused"], (
        "a tree that resolves to zero files must be refused, not reported clean")
    assert vacuous["manifest"] == {}
    missing = proof.snapshot_tree("memory", tmp_path / "not-here")
    assert missing["refused"] and "not a directory" in missing["refused"]

    # The refusal is load-bearing: a vacuous tree cannot produce a PASS.
    result = _run(tmp_path, trees={"memory": _tree(tmp_path, "empty", {}),
                                   "backlog": _tree(tmp_path, "bkl", {"1-x.md": "y\n"})})
    assert result["verdict"] == "REFUSED", result.get("reason")
    assert "zero files" in result["reason"]


# ── clause 3: the server is asked BEFORE any write ─────────────────────────

@pytest.mark.parametrize("flags", [
    pytest.param(state_body(is_production_data=True), id="production-data"),
    pytest.param(state_body(is_production_vault=True), id="production-vault"),
    pytest.param(state_body(state_is_copied=False), id="state-not-copied"),
    # The two bodies a live aggregator can actually hand you, and the two ways a copy can
    # be unproven without anyone saying so: no isolation block at all (an aggregator older
    # than #3403), and a block that omits a key. MEMBERSHIP is the check, never truthiness —
    # the rule `bench_runner_sdk._SERVER_ISOLATION_KEYS` (:509) applies for the same reason.
    pytest.param({"tools": 3}, id="no-isolation-block"),
    pytest.param(state_body(drop=("state_is_copied",)), id="block-missing-a-key"),
    # An empty dict is the same case as `no-isolation-block` viewed from a client that sent
    # nothing: every default has to fall on the production side.
    pytest.param({}, id="empty-body"),
    # The scope never reached the process that will execute the tools, so all three calls
    # would come back refused and the row would blame the tools for a plumbing fault.
    pytest.param(state_body(allowed_writes=()), id="declared-scope-unhonoured"),
])
def test_production_state_or_uncopied_flag_refuses_before_any_tool_call(tmp_path, flags,
                                                                        monkeypatch):
    """Production data, production vault, an unproven copy, or a dead scope: refusal, zero calls.

    Every case defaults toward production and toward not-copied — `isolation_flags` coerces a
    missing flag that way, and `prewrite_guard` refuses on a missing BLOCK or a missing key
    outright — so an unreadable answer can never become a licence to write into Alan's actual
    memory. The mirror default is the one that eventually corrupts the house.

    All three names must be in the refusal because they are three different faults with
    three different fixes: production data means the fixture's data root did not travel,
    production vault means `LLOYD_VAULT_ROOT` did not, and `state_is_copied` False means the
    server's own gate (`agent_mcp/_tool_sandbox.py`) has no evidence it is on a copy.
    """
    calls: list[str] = []
    result = _run(tmp_path, state=flags, recorded=calls)

    assert result["verdict"] == "REFUSED"
    assert calls == [], "not one write may be attempted once the server reports live state"
    for name in ("is_production_data", "is_production_vault", "state_is_copied"):
        assert name in result["reason"], f"the refusal must name {name}: {result['reason']}"
    assert result["isolation_flags"] == proof.isolation_flags(flags)

    # The same refusal at the exit code the module docstring promises (3 = refused).
    monkeypatch.setattr(proof, "fetch_state_json", lambda url, **kw: dict(flags))
    assert proof.main(["--aggregator-url", "http://127.0.0.1:9/mcp",
                       "--state-url", "http://127.0.0.1:9/state"]) == 3


def test_the_cli_refuses_to_boot_a_fixture_that_was_never_built(capsys, monkeypatch):
    """No urls AND an empty `--fixture-root`: refusal (3) naming the builder, spawner untouched.

    Why this is a refusal changed when clause 3 landed, and the node has to say the new
    reason. It used to be "no urls means nowhere to write"; now no urls means THIS script
    boots an aggregator, so the missing piece is a BUILT FIXTURE, and a message that only
    named #3688 would tell a reader who to blame without telling them what to run. The three
    strings asserted here are the three the caller acts on: the flag to pass
    (`--fixture-root`), the module that builds what to pass it (`bench_fixture`), and the
    item that owns the builder (#3688).

    Exit 3, not argparse's 2, because the trial is refusing rather than the parser
    complaining — and the boot is spied, because a half-built invocation that reached
    `bench_spawner` would be a process launch on a directory nobody copied. The
    half-named-url case, which refuses for a different reason, is
    `test_the_cli_refuses_a_half_named_aggregator_without_booting_anything`.
    """
    def _no_boot(*args, **kwargs):
        raise AssertionError(
            "an empty --fixture-root must refuse, not launch a child on nothing")

    monkeypatch.setattr(proof, "boot_fixture_aggregator", _no_boot)
    assert proof.main([]) == 3
    err = capsys.readouterr().err
    for want in ("--fixture-root", "bench_fixture", "#3688"):
        assert want in err, f"{want!r} is missing from the refusal: {err}"


# ── clause 4: the verdict is ATTRIBUTED against an equal-length control ────

def test_a_change_in_only_the_trial_window_fails_and_one_in_both_passes(tmp_path):
    """Background churn is forgiven; a path that moved ONLY during the trial is FAIL.

    42 files under `~/lloyd-data` rewrote themselves in one 120 s window with no trial
    running at all (triage measurement, 2026-11-01), so `changed == 0` is unreachable on a
    live box and a check that can only read RED is not a check. PASS is a SUBSET relation
    against a control window of the SAME length — and equality is structural here, not
    hoped for: `run_proof` pads the trial out to `window_seconds` after the calls, because a
    shorter trial window than the control is a control that saw more background than the
    trial could have, which is the one error direction that forgives a real durable write.

    Both sides of the boundary in one node, since the clause IS a boundary: a sleeper that
    rewrites one log file in every window is the box, and a file planted by the write itself
    is the fault. Same path both windows -> PASS; one extra path in the trial window -> FAIL.
    """
    memory = _tree(tmp_path, "memory", {"MEMORY.md": "one\n"})
    backlog = _tree(tmp_path, "backlog", {"1-item.md": "keep\n"})
    trees = {"memory": memory, "backlog": backlog}
    leak = memory / "LEAKED.md"

    def churn(seconds: float) -> None:
        """What the box does in any window: rewrite one log file."""
        (memory / "log.md").write_text(f"tick {seconds}\n", encoding="utf-8")

    passed = _run(tmp_path, trees=dict(trees), sleep=churn,
                  diff_paths=["backlog/9-bench-durable.md"])
    assert passed["verdict"] == "PASS", passed["reason"]
    assert passed["windows"]["unexplained_count"] == 0
    assert passed["windows"]["per_tree"]["memory"]["counts"] == {
        "trial": 1, "control": 1, "unexplained": 0}, passed["windows"]["per_tree"]["memory"]
    assert not leak.exists(), "the control run must not have planted the leak"

    # A write that reaches the live tree: planted by the call itself, so the control
    # window (which runs first, with no trial in it) cannot possibly have seen it.
    def leaking_call(name: str, arguments: dict, *, session_id: str = "") -> dict:
        leak.write_text("a durable write reached the live tree\n", encoding="utf-8")
        return {"tool": name, "ok": True, "session_id": session_id}

    leaked = _run(tmp_path, trees=dict(trees), sleep=churn, call=leaking_call,
                  diff_paths=["backlog/9-bench-durable.md"])
    assert leaked["verdict"] == "FAIL", leaked["reason"]
    assert leaked["windows"]["unexplained"] == [str(leak)]
    assert "LEAKED.md" in leaked["reason"]
    assert leaked["checks"]["copy_witness_landed"] is True, (
        "the copy leg was green: the FAIL must come from the live diff alone")
    assert leaked["windows"]["per_tree"]["memory"]["counts"] == {
        "trial": 2, "control": 1, "unexplained": 1}


# ── clause 5: no copy-side landing is FAIL, never PASS ─────────────────────

@pytest.mark.parametrize("diff_paths,expect_pass", [
    ([], False),                                    # wrote nothing anywhere
    (["random/unrelated.md"], False),               # wrote something, not the stores
    (["lloyd/MEMORY.md"], True),                    # one intended store, writer's dir name
    (["backlog/7-bench-durable.md"], True),         # another
    (["lloyd/MEMORY.md", "unrelated/x.md"], True),  # intended target plus noise
    # The directory `memory_add` cannot write, offered as if it were the landing: the
    # witness must not be fooled by a path in the vault that names a store-shaped word.
    (["memory/MEMORY.md"], False),
])
def test_an_empty_or_unrelated_copy_diff_is_fail_never_pass(diff_paths, expect_pass,
                                                            tmp_path):
    """A trial that wrote NOTHING anywhere looks exactly like isolation working.

    That is the trap clause 5 closes. All three calls return `ok`, the live trees are
    pristine because nothing was actually attempted, the attributed diff is empty — and a
    naive verdict prints PASS for a trial that proved only that it did nothing. So the copy
    side is a POSITIVE requirement: `git diff --name-only` must name at least one of the
    stores the three tools were asked to write.

    The empty diff is not the only degenerate case, hence the unrelated-path row: a diff
    naming a file in some other directory proves the fixture is writable, not that the
    durable writes landed.
    """
    result = _run(tmp_path, diff_paths=diff_paths)
    assert (result["verdict"] == "PASS") is expect_pass, result["reason"]
    if not expect_pass:
        assert "copy-side witness" in result["reason"]
        assert result["checks"]["copy_witness_landed"] is False
        assert result["checks"]["unexplained_count"] == 0, (
            "the live side was clean: the FAIL has to be the copy witness doing its job")
    assert result["copy_side"]["intended_stores"] == {
        "memory_add": "memory", "fact_add": "facts", "backlog_write_task": "backlog"}


def test_the_record_is_json_and_carries_the_three_flags_in_every_verdict(tmp_path):
    """Whatever else it says, the artifact has to be re-readable by the owed-check job.

    That job reads this record afterwards, so the flags are top level in every row —
    including refusal rows, where they are the whole finding. `is_production_vault` and
    `state_is_copied` are named exactly as #3403's owed entry names them, at the top level,
    rather than nested under an object a grep for the flag would miss.
    """
    result = _run(tmp_path, diff_paths=["lloyd/MEMORY.md"])
    assert json.loads(json.dumps(result)) == result
    for key in ("isolation_flags", "verdict", "windows", "copy_side", "live_trees",
                "tool_calls", "trees"):
        assert key in result, key
    assert result["verdict"] == "PASS"
    assert [call["tool"] for call in result["tool_calls"]] == [
        "memory_add", "fact_add", "backlog_write_task"]
    assert all(call["ok"] for call in result["tool_calls"])
    assert result["isolation_flags"] == {"is_production_data": False,
                                         "is_production_vault": False,
                                         "state_is_copied": True}


# ── the shared driver ──────────────────────────────────────────────────────

def _run(tmp_path, *, trees=None, state=None, sleep=None, call=None,
         recorded=None, diff_paths=None):
    """One trial against stand-in trees, with the calls recorded instead of sent.

    `sleep` is the window: `run_proof` calls it once for the control window and once for the
    trial's padding, with the same duration, which is how "same-length" is a fact of the
    driver instead of an assertion about it.
    """
    subject = trees if trees is not None else {
        "memory": _tree(tmp_path, "mem", {"MEMORY.md": "index line\n"}),
        "backlog": _tree(tmp_path, "bkl", {"1-item.md": "keep\n"})}
    seen = recorded if recorded is not None else []

    def call_tool(name: str, arguments: dict, *, session_id: str = "") -> dict:
        seen.append(name)
        return call(name, arguments) if call else {"tool": name, "ok": True}

    return proof.run_proof(state=dict(COPIED_STATE if state is None else state),
                           trees=dict(subject), call_tool=call_tool,
                           window_seconds=1.0, sleep=sleep or (lambda _s: None),
                           diff_paths=list(diff_paths or []), trial_id="t_proof")




# ── the composition, run for real against a copy and a real subprocess server ─
#
# These nodes exist because unit-pinning each routine does not show the ORDER of the
# assembled thing, which is what the review rung of `SM_20261103_061435` refused the first
# commit for (clauses 3 and 4: "no test exercises the composition of `run_proof`", "no
# demonstration of the negative direction … as a whole-pipeline case"). They are subprocess
# tests on purpose: `app.paths` resolves its roots at IMPORT, so "the server was booted under
# the copy's roots" is a fact about a PROCESS, and no in-process monkeypatch can create it.

_STANDIN_AGGREGATOR = '''
"""Stand-in aggregator: the real roots, the real flags, the real credential check.

Not a mock of the target directory. It resolves its stores through the `app.paths` OF THE
PROCESS IT WAS BOOTED INTO — memory and backlog under `VAULT_ROOT`, facts through
`VAULT_FACTS_ROOT` (`app/paths.py:310-311`, the `LLOYD_FACTS_ROOT`-redirectable root) — and
it answers `GET /state` with `_tool_sandbox.status()`, the same function
`agent_mcp/main.py:909` nests under `tool_sandbox` in the real body. Its isolation flags are
therefore computed by the same predicate the real aggregator computes, from the roots this
process was launched with: boot it with a copy's env and it reports a copy; boot it with no
override and it reports `is_production_vault: true`, because that is what `app.paths` says.

It refuses a durable write the way the real gate does, too: a name outside
`_tool_sandbox.allowed_bench_writes()` (`_tool_sandbox.py:108` — empty unless
`LLOYD_BENCH_WRITE_SCOPE` names it AND the state resolves to a copy) gets 403 and no write,
so a client that skipped its own guard cannot be made to write here either. And it refuses an
unauthenticated request with 401 before it looks at the tool, the way
`aggregator_auth.py` does, because `OPEN_PATHS` (`:95`) holds only `/health`.

What is stood in for is the TOOL BODY: one identifiable file per call, not `memory_add`'s
markdown-plus-index-row transaction. What is not stood in for is the routing, the flag
computation, the flag SHAPE, or the credential check — the four things a stand-in can get
wrong in a way that makes the proof worth nothing.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.environ["STANDIN_REPO"])
import app.paths as P                                            # noqa: E402
from agent_mcp import _tool_sandbox                              # noqa: E402
from agent_mcp.aggregator_auth import AUTH_HEADER, read_token    # noqa: E402
from agent_mcp.main import _refused_call                         # noqa: E402
from agent_mcp.session import MEMORIES_ROOT                      # noqa: E402

#: Store directory per tool, resolved by THIS process's roots and never handed in by the
#: test — which is the whole point of booting a child instead of patching a dictionary.
#: `memory_add` lands in `MEMORIES_ROOT`, the shipped writer's own symbol
#: (`agent_mcp/session.py:58`, `vault_root()/'lloyd'`), NOT in a directory called `memory`:
#: standing that up wrong here would send the trial's copy-side witness to a place the tool
#: cannot write, and would keep the LIVE-side node green as well, because a leak node only
#: fails when the watched tree is the one the writer reaches (#3863 clause 4, both sides).
STORES = {"memory_add": MEMORIES_ROOT,
          "fact_add": P.VAULT_FACTS_ROOT,
          "backlog_write_task": P.VAULT_ROOT / "backlog"}
#: Every POST, in arrival order, including the ones refused before they reached a tool. The
#: witness for "the guard refused, so nothing arrived": a refusal that had reached the tool
#: would leave a line here.
POST_LOG = os.environ.get("STANDIN_POST_LOG", "")
#: One line per ACCEPTED call: `<tool> <session_id>`. Kept apart from `POST_LOG` so the
#: arrival-order witness keeps its exact shape, and because it answers a different question:
#: not "did a call reach the tool" but "which session did the server charge it to".
SESSION_LOG = os.environ.get("STANDIN_SESSION_LOG", "")
#: An override of `SANDBOXED_ID_PREFIXES` for the node that wants every call refused: set it
#: to a prefix no trial id can carry and the child answers each one with the shipped
#: `_refused_call` body, which is how the all-refused trial gets refusals that are the real
#: shape rather than a dict this file invented.
REQUIRE_PREFIX = os.environ.get("STANDIN_REQUIRE_PREFIX", "")
TOKEN = read_token(publish=False)


def _note(text):
    if POST_LOG:
        with open(POST_LOG, "a", encoding="utf-8") as handle:
            handle.write(text + "\\n")


def _note_session(line):
    if SESSION_LOG:
        with open(SESSION_LOG, "a", encoding="utf-8") as handle:
            handle.write(line + "\\n")


def _denied(name, reason, guard, sid):
    """The refusal body the SHIPPED handler sends, built by the shipped function.

    `_refused_call` is what `agent_mcp/main.py:768` returns for every guard, serialised the
    way the MCP layer serialises it — `by_alias`, because the wire is camelCase (`isError`,
    not `is_error`), which is the split this file's composition node exists to keep honest.
    Calling the function instead of typing a dict is the difference between a pair that
    agrees with itself and a test that reads the answer the shipped server gives: what comes
    back is a JSON-RPC *result* with no `error` member, so a client that only looks for
    `error` scores the refusal as a success.
    """
    result = _refused_call(name, reason, guard=guard, session_id=sid)
    return result.model_dump(mode="json", by_alias=True, exclude_none=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):   # noqa: A002 — matches the base's name
        pass

    def _authorised(self):
        return TOKEN is not None and self.headers.get(AUTH_HEADER) == TOKEN

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/state":
            self._send(404, {"detail": "no such path"})
            return
        if not self._authorised():
            self._send(401, {"error": "AGGREGATOR_UNAUTHENTICATED"})
            return
        self._send(200, {"tools": len(STORES), "tool_sandbox": _tool_sandbox.status()})

    def do_POST(self):
        _note("posted")
        if not self._authorised():
            _note("unauthorised")
            self._send(401, {"error": "AGGREGATOR_UNAUTHENTICATED"})
            return
        # Path-sensitive on purpose. The real aggregator mounts `tools/call` on `/mcp` and
        # Starlette answers anything else with a 404, and `bench_spawner` hands back a BARE
        # origin (`:362`), so appending the route is load-bearing work this script has to do.
        # Answering every path would let `handle.url`, without the suffix, sail through: the
        # first attempt of this round did exactly that, and the grader measured it.
        if self.path != "/mcp":
            _note("bad-path")
            self.send_error(404, "not found")
            return
        rpc = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
        params = rpc.get("params") or {}
        name, args = params.get("name"), params.get("arguments") or {}
        # The session id, read the way the shipped handler reads it: `_meta` first
        # (`_bound_session_id` + `META_SESSION_ID`), the only place a JSON-RPC caller can put
        # it. Every tool this stand-in serves is state-changing (`state_changing_tool` is
        # `name not in READ_ONLY`, `_tool_sandbox.py:366-378`), so a call with no id is
        # refused the way `main.py:594-602` refuses it, with the same body — which is the part
        # the trial's reader is graded on, since the old reader called that body a success.
        meta = params.get("_meta") or {}
        sid = str(meta.get("lloyd/session_id") or "")
        if not sid:
            _note("refused-sessionless")
            self._send(200, {"jsonrpc": "2.0", "id": rpc.get("id"), "result": _denied(
                name,
                "a call that can change state arrived with no session id — refusing rather "
                "than treating a missing session id as 'not sandboxed' (#1053); pass the "
                "session id in `lloyd/session_id` `_meta`", "sessionless", sid)})
            return
        # A tightening of the STAND-IN, not a claim about the shipped server: an ordinary
        # non-sandboxed session would be allowed to write by `main.py`'s sandbox guard. This
        # child refuses one anyway, because the armed scope is the grant that lets a SANDBOXED
        # trial write into its own copy, and accepting an id the sandbox does not class as a
        # bench/eval trial would let a node read a granted write as proof the trial was
        # sandbox-classed when it never was.
        prefixes = ((REQUIRE_PREFIX,) if REQUIRE_PREFIX
                    else _tool_sandbox.SANDBOXED_ID_PREFIXES)
        if not sid.startswith(tuple(prefixes)):
            _note("refused-unsandboxed")
            self._send(200, {"jsonrpc": "2.0", "id": rpc.get("id"), "result": _denied(
                name, f"session {sid!r} is not classed as a bench or eval trial "
                      f"(SANDBOXED_ID_PREFIXES)", "tool_sandbox", sid)})
            return
        if name not in STORES:
            self._send(400, {"ok": False, "error": "unknown tool"})
            return
        allowed = _tool_sandbox.allowed_bench_writes()
        if name not in allowed:
            _note("refused-scope")
            self._send(403, {"ok": False, "error": "write_scope_closed",
                             "allowed_writes": sorted(allowed)})
            return
        # A house-routed child can never reach a write: `allowed_bench_writes()` is empty
        # once `state_is_copied()` is false, so the check above is what keeps a test child out
        # of /home/alansrobotlab/obsidian, and the house node asserts that emptiness directly
        # rather than probing it with a POST that a regressed gate would have WRITTEN.
        marker = str(args.get("fact") or args.get("description") or args.get("entry")
                     or args.get("name") or "")
        trial = marker.split()[-1] if marker else "unknown"
        store = STORES[name]
        store.mkdir(parents=True, exist_ok=True)
        target = store / f"proof-{trial}.md"
        target.write_text(f"# {name}\\n\\n{marker}\\n", encoding="utf-8")
        _note("wrote")
        _note_session(f"{name} {sid}")
        self._send(200, {"jsonrpc": "2.0", "id": rpc.get("id"), "result": {"content": [
            {"type": "text", "text": json.dumps({"ok": True, "record_path": str(target)})}]}})


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_address[1], flush=True)
server.serve_forever()
'''

#: The env names that decide which tree a process calls home. Stripped from the child before
#: the wanted ones are set, so "booted on the house" means *no override present* and not
#: whatever the gate's own environment happened to export; and stripped from the parent for
#: the duration of a node, so the credential `proof.auth_headers` resolves is the probe file
#: and never an ambient token this test process inherited — with one deliberate exception:
#: while the boot node runs `main` it re-points the parent's `LLOYD_MCP_TOKEN_FILE` at
#: `DECOY_TOKEN`, because a parent and a child holding the SAME credential file is what made
#: the boot's choice of credential unobservable there (see `DECOY_TOKEN`).
_ROUTED_ENV = ("LLOYD_DATA", "LLOYD_VAULT_ROOT", "LLOYD_VAULT", "LLOYD_FACTS_ROOT",
               "LLOYD_KG_DB", "LLOYD_MCP_TOKEN")

#: 26 characters of the alphabet `aggregator_auth._looks_valid` (:112-114) accepts — at
#: least 16 of alphanumerics, `-` and `_` — so `read_token(publish=False)` returns this
#: instead of rejecting it as malformed and answering every request with 401.
PROBE_TOKEN = "probe-token-0123456789abcdef"

#: A second, DIFFERENT credential for the process that runs this file to hold while the boot
#: node calls `main`. `read_token`/`auth_headers` (`agent_mcp/aggregator_auth.py:306-322`) cache
#: nothing, so which file the parent points at decides what a fall-back to
#: `auth_headers(handle.url + '/state')` actually puts on the wire.
#:
#: It exists because the first version of the boot node set the child's and the parent's
#: `LLOYD_MCP_TOKEN_FILE` to the SAME file, so the two candidate sources of the trial
#: credential carried the same bytes and the node could not tell them apart: measured on
#: SM_20261106_223337, replacing `spawner.auth_headers(handle.env)` with
#: `auth_headers(handle.url + '/state')` left
#: `test_the_cli_boots_its_own_aggregator_from_a_built_fixture_and_stops_it` GREEN, and the
#: review rung scored clause 3's credential half "not gradeable, because the mutation is
#: equivalent under the test's own credential-sharing". Under this token the two differ, so
#: that fall-back gets its 401 on the state read and the node's `code == 0` is the assertion
#: that reddens.
DECOY_TOKEN = "decoy-token-fed-to-the-parent-process-only"


def _live_stub(tmp_path: Path):
    """The four watched trees as small stand-ins, same shapes, same resolution rules."""
    vault = tmp_path / "live-vault"
    (vault / "lloyd").mkdir(parents=True)
    (vault / "memory").mkdir(parents=True)
    (vault / "backlog").mkdir(parents=True)
    # The memory stand-in is `lloyd/`, the directory the shipped writer resolves
    # (`agent_mcp/session.py:58 MEMORIES_ROOT = vault_root()/'lloyd'`), and `memory/` is
    # present too as the daily-notes directory the writer does NOT use — the two--directory
    # shape is what lets the leak node tell a watched tree from a merely adjacent one
    # (#3863 clause 4).
    (vault / "lloyd" / "MEMORY.md").write_text("# index\n", encoding="utf-8")
    (vault / "memory" / "2026-11-07.md").write_text("# daily\n", encoding="utf-8")
    # A real backlog holds files; an empty stand-in would be refused as vacuous, which is
    # the clause-2 rule working, not the state a trial runs in.
    (vault / "backlog" / "1-something.md").write_text("# 1\n", encoding="utf-8")
    data = tmp_path / "live-data"
    facts = data / "_pipeline" / "vault-derived" / "facts"
    facts.mkdir(parents=True)
    (facts / "entity-aliases.json").write_text("{}", encoding="utf-8")
    (data / "workers.db").write_text("stub\n", encoding="utf-8")
    return vault, data


class _Server:
    """A booted stand-in: its base url, and the log of what reached its tool endpoint."""

    def __init__(self, url: str, post_log: Path, session_log: Path | None = None,
                 token_file: Path | None = None):
        self.url = url
        self.post_log = post_log
        self.session_log = session_log
        #: The credential file the child was started with, so a node can hand it to a process
        #: that has to authenticate and does not inherit this one's environment.
        self.token_file = token_file

    def posts(self) -> list[str]:
        return ([] if not self.post_log.exists()
                else self.post_log.read_text(encoding="utf-8").split())

    def sessions(self) -> list[str]:
        """`<tool> <id>` per ACCEPTED call — the server's own account of who it wrote for."""
        return ([] if self.session_log is None or not self.session_log.exists()
                else [line.strip() for line in
                      self.session_log.read_text(encoding="utf-8").splitlines() if line.strip()])


@contextlib.contextmanager
def _aggregator(tmp_path: Path, roots: dict, *, write_scope=("memory_add", "fact_add",
                                                             "backlog_write_task"),
                require_prefix: str = ""):
    """Boot the stand-in with `roots` as its `app.paths`, and the credential on file.

    A `None` value means "let the child resolve that root itself", which is how a node gets a
    child that really is production-routed rather than one that has been told to pretend.
    """
    token_file = tmp_path / "aggregator-token"
    token_file.write_text(PROBE_TOKEN + "\n", encoding="utf-8")
    script = tmp_path / "standin_aggregator.py"
    script.write_text(_STANDIN_AGGREGATOR, encoding="utf-8")
    post_log = tmp_path / "post-log.txt"
    session_log = tmp_path / "session-log.txt"

    child_env = {key: value for key, value in os.environ.items() if key not in _ROUTED_ENV}
    child_env["STANDIN_REPO"] = str(REPO)
    child_env["STANDIN_POST_LOG"] = str(post_log)
    child_env["STANDIN_SESSION_LOG"] = str(session_log)
    if require_prefix:
        child_env["STANDIN_REQUIRE_PREFIX"] = require_prefix
    child_env["LLOYD_MCP_TOKEN_FILE"] = str(token_file)
    child_env["LLOYD_BENCH_WRITE_SCOPE"] = ",".join(write_scope)
    for key, var in (("vault", "LLOYD_VAULT_ROOT"), ("data", "LLOYD_DATA"),
                     ("facts", "LLOYD_FACTS_ROOT")):
        if roots.get(key) is not None:
            child_env[var] = str(roots[key])

    saved = {key: os.environ.pop(key, None) for key in _ROUTED_ENV}
    os.environ["LLOYD_MCP_TOKEN_FILE"] = str(token_file)
    child = subprocess.Popen([sys.executable, str(script)], env=child_env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        url = ""
        for _ in range(40):
            line = (child.stdout.readline() or "").strip()
            if not line:
                break
            if line.isdigit():
                url = f"http://127.0.0.1:{line}"
                break
        if not url:
            raise RuntimeError("stand-in aggregator did not report a port: "
                               + (child.stderr.read() or "")[:3000])
        yield _Server(url, post_log, session_log, token_file)
    finally:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:      # pragma: no cover — never hang the suite
            child.kill()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        os.environ.pop("LLOYD_MCP_TOKEN_FILE", None)


def _copy_roots(fixture, tmp_path: Path) -> dict:
    """Boot the server on the COPY's roots. One deliberate choice inside the choice: the
    copy's facts root is pulled inside the copied vault, because `init_git` roots the
    copy's git repo at `<fixture>/obsidian` (`bench_fixture.py:418`) while the default
    `VAULT_FACTS_ROOT` sits under the fixture's DATA root and would therefore be outside
    every path `git diff --name-only` can name. `LLOYD_FACTS_ROOT` is exactly the override
    for that (`app/paths.py:310-311`), and pulling it in is what lets clause 5 witness all
    three stores rather than two.
    """
    return {"vault": fixture.vault, "data": fixture.data_root,
            "facts": fixture.vault / "facts"}


def _live_tree_roots(fixture, tmp_path: Path) -> dict:   # noqa: ARG001 - ignoring the copy IS the point
    """Boot the server on the watched LIVE trees. The `0fa26e0b` bug in one line: its roots
    are tmp dirs, so `_tool_sandbox.state_is_copied()` is true for real and every guard
    passes, while its writes land outside the copy."""
    vault, data = tmp_path / "live-vault", tmp_path / "live-data"
    return {"vault": vault, "data": data,
            "facts": data / "_pipeline" / "vault-derived" / "facts"}


def _house_roots(fixture, tmp_path: Path) -> dict:      # noqa: ARG001 - the house ignores the copy
    """Boot the server with NO override, the way a supervised `lloyd-mcp` comes up. Its
    `VAULT_ROOT` is then `/home/alansrobotlab/obsidian` and its
    `IS_PRODUCTION_VAULT` is True because it genuinely is. (Its data root resolves to the
    worktree's own `.lloyd-data`, `app/paths.py:83`, so `is_production_data` is False here —
    one of the two flags is enough to refuse, and this node reports which one it saw.)"""
    return {"vault": None, "data": None, "facts": None}


class _TrialRun:
    def __init__(self, server: _Server, state: dict, trees: dict, fixture):
        self.server = server
        self.state = state
        self.trees = trees
        self.fixture = fixture
        self.record = None

    def run(self, *, window_seconds=1.0):
        """`main()`'s own wiring: real `fetch_state_json`, real `mcp_tool_call`, real clock."""
        self.record = proof.run_proof(
            state=self.state, trees=self.trees,
            call_tool=lambda name, args, session_id="": proof.mcp_tool_call(
                self.server.url + "/mcp", name, args, session_id=session_id,
                timeout=20.0),
            fixture_root=self.fixture.root, sleep=time.sleep, clock=time.monotonic,
            window_seconds=window_seconds, trial_id="abc")
        return self.record


@contextlib.contextmanager
def _trial(tmp_path: Path, roots_for, *, write_scope=("memory_add", "fact_add",
                                                      "backlog_write_task"),
           require_prefix: str = ""):
    """One assembled trial: real builder, real subprocess server, state read over the wire.

    Yields a `_TrialRun` with the state already fetched (that read is the guard's input, so
    a node that wants to inspect what the server really said can) and the watched stand-in
    trees. Call `.run()` to drive `run_proof` exactly as the CLI does.
    """
    vault, data = _live_stub(tmp_path)
    trees = {"memory": vault / "lloyd", "backlog": vault / "backlog",
             "facts": data / "_pipeline" / "vault-derived" / "facts", "data": data}
    # The builder requires the copy inside the data root's own fixtures area
    # (`bench_fixture.py:268-270`), which puts it inside the watched `data` tree — and that
    # is exactly why `run_proof` excludes `fixture_root` from every snapshot: without the
    # exclusion the copy's own files would be the largest "live change" in the trial window.
    fixture = build_fixture(trial_id="t_durable_proof_0001", writes=["vault"],
                            vault=vault, data_root=data,
                            fixtures_parent=data / "eval" / "fixtures")
    fixture.init_git()
    try:
        with _aggregator(tmp_path, roots_for(fixture, tmp_path),
                         write_scope=write_scope, require_prefix=require_prefix) as server:
            run = _TrialRun(server, proof.fetch_state_json(server.url + "/state"),
                            trees, fixture)
            yield run
    finally:
        fixture.cleanup()


def test_one_whole_trial_against_a_copy_records_pass_and_writes_only_in_the_copy(tmp_path):
    """The composition itself: guard, control window, three transport calls, verdict.

    The review refused this item's first attempt because no node ran the pipeline — "a
    guard is not a guard if every caller could pass the arguments it trusts in some other
    order", and the trust that matters here is WHICH roots the writer resolves. So this
    builds a real fixture with `build_fixture`, boots a real subprocess aggregator on the
    copy's roots, and calls `run_proof` with the real `fetch_state_json`, the real
    `mcp_tool_call` and a real 1-second window on each side.

    The server's `/state` block is `_tool_sandbox.status()` computed in the child, so the
    flags this row records are the flags the real predicate returns for a process pointed at
    a copy — not three booleans this file chose.
    """
    with _trial(tmp_path, _copy_roots) as run:
        res = run.run()
        assert res["verdict"] == "PASS", res["reason"]
        assert res["isolation_flags"] == {"is_production_data": False,
                                          "is_production_vault": False,
                                          "state_is_copied": True}, res["isolation_flags"]
        assert res["write_scope_gap"] == [] and res["missing_isolation_keys"] == []
        assert [row["ok"] for row in res["tool_calls"]] == [True] * 3, res["tool_calls"]
        assert res["copy_side"]["landed_by_git_diff"] == ["backlog_write_task", "fact_add",
                                                          "memory_add"], res["copy_side"]
        assert res["copy_side"]["not_landed_by_git_diff"] == [], res["copy_side"]
        assert res["windows"]["unexplained_count"] == 0, res["windows"]
        assert run.server.posts() == ["posted", "wrote"] * 3, (
            "the tool endpoint must have seen exactly three authenticated calls, each of "
            "which wrote — no refused-scope row, no fourth request")
    paths = []
    for row in res["tool_calls"]:
        path = json.loads(row["result_text"])["record_path"]
        paths.append(path)
        assert "/eval/fixtures/t_durable_proof_0001/" in path, (
            f"a durable write did not land inside the copy: {path}")
    assert len(set(paths)) == 3, f"the three tools reported one path: {paths}"
    # The disjointness the item is actually about. `data` gets a different sentence because
    # the builder MANDATES the copy inside `<data>/eval/fixtures` (bench_fixture.py:268-270),
    # so the fixtures area is under the watched data root by design — which is exactly what
    # `run_proof`'s `exclude=fixture_root` handles in the diff. The claim that matters is
    # that the write is inside the trial's own copy directory, not merely under data.
    fixtures_area = Path(res["live_trees"]["data"]["root"]) / "eval" / "fixtures"
    for name, tree in res["live_trees"].items():
        for path in paths:
            if name == "data":
                trial_dir = fixtures_area / "t_durable_proof_0001"
                assert path.startswith(str(trial_dir) + os.sep), (
                    f"data: a write landed under the data root but outside the trial's own "
                    f"copy directory: {path}")
                continue
            assert not path.startswith(str(tree["root"]) + os.sep), (
                f"{name}: a durable write reached the live tree at {path}")


def test_a_write_that_reaches_the_live_tree_cannot_be_recorded_pass(tmp_path):
    """The leak-negative, whole-pipeline: every flag true, only the ROUTING differs.

    The server answers `/state` with the real `status()` of a process whose roots are tmp
    dirs, so `state_is_copied` is true because it genuinely is true for that process and
    every guard passes — it was simply booted writing into the watched live trees, which is
    what 0fa26e0b did in production. If this verdict were PASS, every PASS this instrument
    ever prints would be consistent with a trial that wrote into the house. The copy-side
    diff is empty here, so clause 5 refuses from the other side of the same event.
    """
    with _trial(tmp_path, _live_tree_roots) as run:
        res = run.run()
    assert res["verdict"] == "FAIL", f"a live-tree write was recorded as PASS: {res}"
    assert res["isolation_flags"]["state_is_copied"] is True
    assert [row["ok"] for row in res["tool_calls"]] == [True] * 3
    assert res["windows"]["unexplained_count"] > 0, res["windows"]
    leaked = res["windows"]["unexplained"]
    assert any(p.endswith("proof-abc.md") for p in leaked), leaked
    assert (run.trees["memory"] / "proof-abc.md").exists(), (
        "the leak has to be a real file in the watched tree, or this node proves nothing")
    assert res["copy_side"]["diff_paths"] == [], res["copy_side"]
    assert res["checks"]["copy_witness_landed"] is False


def test_a_server_booted_on_the_house_vault_is_refused_before_any_call_reaches_it(tmp_path):
    """Clause 3 across a real process boundary, in the direction a mock cannot get wrong.

    The child is booted with no vault override, so its `app.paths` resolves
    `/home/alansrobotlab/obsidian` and `status()` reports `is_production_vault: true`
    because that IS the case, `state_is_copied: false`, and `allowed_writes: []`. The proof
    must refuse on the server's own answer before snapshotting, before either window and
    before any POST — and the POST log is the witness for that: a refusal that had reached
    the tool would leave a line in it. The child's own gate would also 403 the call, so this
    node shows the two guards agree about a production-routed process.
    """
    with _trial(tmp_path, _house_roots) as run:
        block = run.state["tool_sandbox"]
        assert block["is_production_vault"] is True, block
        assert block["state_is_copied"] is False, block
        assert block["allowed_writes"] == [], block
        res = run.run()
        assert res["verdict"] == "REFUSED", res
        for flag in ("is_production_data", "is_production_vault", "state_is_copied"):
            assert flag in res["reason"], f"the refusal must name {flag}: {res['reason']}"
        assert res["tool_calls"] == [], res["tool_calls"]
        assert run.server.posts() == [], (
            "the guard refused, so nothing may have reached the tool endpoint")


def test_the_bodies_this_script_trusts_are_the_ones_the_harness_reader_trusts(tmp_path):
    """The two readers of the flags are checked against the same bytes, in both directions.

    `prewrite_guard` here and `bench_runner_sdk.require_copied_state` there are one rule
    written twice — once for a synchronous script, once for an async runner — and the review
    of the previous commit rightly refused to believe a stand-in whose flag shape was
    invented for the test. So this node runs the REAL `require_copied_state` (httpx, the real
    auth header, the real `tool_sandbox` nesting) against the stand-in's real body and pins
    that it accepts the copy-routed answer and refuses the house-routed one, and that the
    required-key lists have not drifted apart.
    """
    import asyncio

    from scripts.autoresearch import bench_runner_sdk as sdk
    assert tuple(proof.SERVER_ISOLATION_KEYS) == tuple(sdk._SERVER_ISOLATION_KEYS)

    declared = ["memory_add", "fact_add", "backlog_write_task"]
    with _aggregator(tmp_path, {"vault": tmp_path / "copy-vault",
                                "data": tmp_path / "copy-data",
                                "facts": tmp_path / "copy-vault" / "facts"}) as copied:
        block = asyncio.run(sdk.require_copied_state(declared=declared,
                                                     state_url=copied.url + "/state"))
        assert block["state_is_copied"] is True and block["is_production_vault"] is False
        assert sorted(block["allowed_writes"]) == sorted(declared)

    with _aggregator(tmp_path, _house_roots(None, tmp_path)) as house:
        with pytest.raises(sdk.TrialRefused) as refused:
            asyncio.run(sdk.require_copied_state(declared=declared,
                                                 state_url=house.url + "/state"))
        assert "is_production_vault" in str(refused.value)


def test_the_wire_refuses_a_wrong_or_absent_credential_before_it_refuses_the_tool(tmp_path):
    """`mcp_tool_call`'s credential path in both directions, over a real socket.

    The previous commit shipped a 401 branch no node entered, and its trial handed
    `token=` in directly, which skipped the file read that the default path takes. Here
    nothing is handed in: the working call's credential comes from
    `LLOYD_MCP_TOKEN_FILE` through `agent_mcp.aggregator_auth.headers_for_url`, a wrong one
    and an absent one are answered 401 and land in the row as `ok: false` with the status in
    the detail, and the store ends up holding only the file the authenticated call wrote.
    The state leg is covered too, because `OPEN_PATHS` (`aggregator_auth.py:95`) holds only
    `/health` — a credential-less `GET /state` is a 401, and parsing one as a state answer
    would be reading a body the real server never sends.
    """
    vault = tmp_path / "copy-vault"
    with _aggregator(tmp_path, {"vault": vault, "data": tmp_path / "copy-data",
                                "facts": vault / "facts"}) as server:
        wrong = proof.mcp_tool_call(server.url + "/mcp", "memory_add",
                                    {"entry": "wrong credential abc"}, timeout=20.0,
                                    headers={"X-Lloyd-Aggregator-Token":
                                             "wrong-token-0123456789abcdef"})
        assert wrong["ok"] is False and "401" in wrong["detail"], wrong
        absent = proof.mcp_tool_call(server.url + "/mcp", "memory_add",
                                     {"entry": "absent credential abc"}, timeout=20.0,
                                     headers={})
        assert absent["ok"] is False and "401" in absent["detail"], absent
        # A bench-prefixed id, because this node is about the credential and the stand-in,
        # like the shipped server, will not let a state-changing call through without a
        # session id at all (#1053). The two refused calls above deliberately carry none:
        # they die at the credential gate, which is upstream of the session gate, and that
        # ordering is what the `posts()` shape below is asserting.
        good = proof.mcp_tool_call(server.url + "/mcp", "memory_add",
                                   {"entry": "carried by the file read abc"}, timeout=20.0,
                                   session_id="bench_credential-probe")
        assert good["ok"] is True and json.loads(good["text"])["record_path"].endswith(
            "proof-abc.md"), good

        assert server.posts() == ["posted", "unauthorised"] * 2 + ["posted", "wrote"]
        assert "refused-scope" not in server.posts(), (
            "an unauthenticated POST must be refused before the scope check runs")
        written = sorted(p.name for p in vault.glob("lloyd/proof-*.md"))
        assert written == ["proof-abc.md"], (
            "the two refused calls must have written nothing to the store")

        # The state leg: a credential that does not match the server's is a refusal, not a
        # body — `fetch_state_json` raises rather than handing back a 401 page to parse.
        other = tmp_path / "other-token"
        other.write_text("another-token-0123456789abcdef\n", encoding="utf-8")
        os.environ["LLOYD_MCP_TOKEN_FILE"] = str(other)
        with pytest.raises(proof.ProofRefused):
            proof.fetch_state_json(server.url + "/state")


# ─────────────────────────────────────────────────────────────────────────────
# #3863 — the session-id carrier, the self-booted aggregator, the writer's own
# memory directory, and the refusal that is not a success.
#
# Four clauses, one hazard each, and every one of them is a FALSE GREEN:
#   * a `tools/call` with no `_meta` is refused by the shipped handler as a BODY, not as a
#     JSON-RPC error, so the pre-round reader scored all three refusals as successes;
#   * an id without a sandboxed prefix is refused the same way, by the next guard;
#   * an aggregator this script never boots is an aggregator whose flags were never read;
#   * the memory tree used to be a directory full of files `memory_add` cannot write, so it
#     was neither vacuous (never refused) nor watchable (never changed).
# A test that only agrees with its own stub proves none of that, so each node below either
# reads a value the shipped code computes (`_bound_session_id`, `SANDBOXED_ID_PREFIXES`,
# `_refused_call`, `MEMORIES_ROOT`) or drives a subprocess that does.
# ─────────────────────────────────────────────────────────────────────────────


def _wire_body(rpc_body, session_id):
    """`rpc_body`'s params as the JSON layer will actually put them on the wire."""
    return json.loads(json.dumps(rpc_body("memory_add", {"entry": "e"},
                                          session_id=session_id)))["params"]


def test_the_session_id_rides_in_params_meta_and_the_wire_refuses_a_call_without_one(tmp_path):
    """Clause 1: the carrier is on the wire, and the wire says so by refusing without it.

    Two halves, because the unit half and the wire half fail for different reasons. The unit
    half pins the KEY — and pins it to the shipped constant rather than to this file's
    spelling, so renaming `META_SESSION_ID` on the server moves the test and a proof that
    quietly stopped identifying itself does not. The wire half is the boundary crossing: the
    same body, sent, with the id present and absent.

    The absent half is what #3689's proof could never have caught. A sessionless
    state-changing call is refused by `agent_mcp/main.py:594-602` (`guard="sessionless"`,
    body built by `_refused_call` at `:768`), and that body arrives as a normal JSON-RPC
    *result* whose only content text is `{"error": "Tool call denied: …"}` — there is no
    `error` member in the envelope, so `ok = "error" not in payload` returned True. Three
    refusals write nothing, so all four live trees stay pristine and the equal-length control
    explains every path: the trial that proved nothing was the trial that printed PASS.
    """
    import agent_mcp.main as server_main

    params = _wire_body(proof.rpc_body, "bench_carry-probe")
    assert params["_meta"] == {proof.META_SESSION_ID: "bench_carry-probe"}, params
    # And read back by it: `_bound_session_id` is the function the shipped dispatch hands
    # `ctx.meta` to, so feeding it the body this script builds proves the two ends agree on
    # the key and on which field wins. It does NOT prove the SDK copies `params._meta` into
    # `ctx.meta` for a single-shot POST — that seam stays open here, and only
    # tests/test_autoresearch_bench_spawner.py boots a real child, which never POSTs.
    assert server_main._bound_session_id({"entry": "e"}, params["_meta"]) == \
        "bench_carry-probe", params
    assert proof.META_SESSION_ID == server_main.META_SESSION_ID, (
        "the key this script writes must be the key the handler reads; if these drift, the "
        "trial is sessionless again and every refusal it took will have read as a write")
    assert "lloyd/session_id" in params["_meta"], params

    # The id is per call, never left over in the envelope: `rpc_body` builds each body fresh,
    # and an id that leaked from one call into the next would attribute a write to the wrong
    # session in the one record whose job is to say who did what.
    assert "_meta" not in _wire_body(proof.rpc_body, ""), (
        "an empty id must not put an empty _meta on the wire — the handler treats "
        "a present-but-empty id exactly like an absent one")

    with _trial(tmp_path, _copy_roots) as run:
        url = run.server.url + "/mcp"

        bare = proof.mcp_tool_call(url, "memory_add", {"entry": "no id at all"}, timeout=20.0)
        assert bare["ok"] is False, (
            f"a sessionless state-changing call was recorded as a success: {bare}")
        assert "Tool call denied" in bare["denial"], bare
        assert "error" not in bare["response"], (
            "the refusal is a JSON-RPC result, not an error member — which is precisely why "
            f"reading only `error` was the bug: {bare['response']}")

        carried = proof.mcp_tool_call(url, "memory_add", {"entry": "carried"},
                                      session_id="bench_carry-probe", timeout=20.0)
        assert carried["ok"] is True, carried
        assert run.server.sessions() == ["memory_add bench_carry-probe"], (
            "the server's own log must name the session it charged the write to")
        assert "refused-sessionless" in run.server.posts(), (
            "the refused call must be visible in the arrival order, before the accepted one")


def test_every_trial_call_carries_a_bench_prefixed_id_the_record_and_the_server_both_name(tmp_path):
    """Clause 2: the prefix comes from the sandbox's rule, and the record shows what was sent.

    `SANDBOXED_ID_PREFIXES` (`_tool_sandbox.py:60`, matched at `:170`) is what classes a
    session as a bench/eval trial, so this reads the tuple rather than typing `bench_`: a
    future prefix is honoured here on the day it is added, and — the part that matters —
    `trial_session_id` cannot disagree with the server about which ids are trials, which is
    the disagreement that would make every call refuse while looking like a passing trial.

    The record carries the id that was SENT, not the one that was asked for, and the node
    checks the server's own accepted-session log against it. An artefact naming a different
    session from the one the handler charged the writes to is an artefact that cannot answer
    the only question a leak review asks.
    """
    from agent_mcp import _tool_sandbox

    sandboxed = _tool_sandbox.SANDBOXED_ID_PREFIXES
    assert "bench_" in sandboxed, sandboxed
    for raw in ("abc", "durable-trial-7"):
        out = proof.trial_session_id(raw)
        assert out.startswith("bench_"), out
        assert out.startswith(tuple(sandboxed)), (
            f"{raw!r} became {out!r}, which the server would class as an ordinary chat "
            "session — and an ordinary session is not sandboxed, so it would have been "
            "allowed to write to whatever its roots name")
    for already in ("bench_x", "pt-eval-x"):
        assert proof.trial_session_id(already) == already, (
            f"an id already classed as a trial must arrive unchanged, not re-prefixed into "
            f"{proof.trial_session_id(already)!r}")
    assert proof.trial_session_id(proof.trial_session_id("abc")) == proof.trial_session_id("abc")

    with _trial(tmp_path, _copy_roots) as run:
        res = run.run()
        sid = res["session_id"]
        assert sid.startswith("bench_"), sid
        assert sid.startswith(tuple(sandboxed)), sid
        assert run.server.sessions() == [f"{name} {sid}" for name in
                                        ("memory_add", "fact_add", "backlog_write_task")], (
            f"the server charged the writes to another session: {run.server.sessions()} vs "
            f"recorded {sid}")
        assert [c["session_id"] for c in res["tool_calls"]] == [sid] * 3, (
            "every recorded call must name the session it was sent with, so a reader of a "
            "leak can attribute each write to one session")


def test_a_trial_in_which_every_write_was_refused_can_never_be_recorded_pass(tmp_path):
    """Clause 5: refusals are recorded as not-ok, so a clean live side cannot carry a PASS.

    The child answers every call with the body the SHIPPED `_refused_call` builds — the node
    asks it to require a prefix no trial id can carry, so the refusal comes from the real
    function on the real path rather than from a dict this file typed.

    This is the shape that would have made #3689's proof certify itself: three refusals write
    nothing, so all four live trees are pristine, the control is the same length and explains
    zero paths, and every leg of the old verdict was satisfied. The only thing that can
    condemn it is knowing that a body saying `Tool call denied` is not a write — and, second,
    that the copy received nothing it was supposed to receive.
    """
    with _trial(tmp_path, _copy_roots, require_prefix="task:") as run:
        res = run.run()
        assert res["verdict"] == "FAIL", (
            f"an all-refused trial was recorded PASS: {res['reason']}")
        assert [c["ok"] for c in res["tool_calls"]] == [False] * 3, res["tool_calls"]
        for call in res["tool_calls"]:
            assert "Tool call denied" in call["denial"], call
            assert call["session_id"] == res["session_id"], call
        assert "refused by" in res["reason"], (
            f"the reason must name the refusals, not just the clean trees: {res['reason']}")

        # The trap itself: the live side really IS clean, and that cleanliness is worthless.
        assert res["windows"]["unexplained_count"] == 0, res["windows"]
        assert res["copy_side"]["landed_by_git_diff"] == [], res["copy_side"]
        # A bare count would let a later reader file this as "inconclusive, nothing happened",
        # which is the misfiling clause 5 exists to prevent, so the leg has to name the calls
        # AND carry the server's own words for each. `verdict` builds one entry per refused
        # call as "<tool>: <body>", the body being `_refused_call`'s JSON — so the tool name
        # and the marker sit in the same entry but never adjacent. Cutting the leg to
        # "3 tool call(s) were refused" reddens the first assertion; dropping the quoted body
        # from it reddens the second.
        for tool in ("memory_add", "fact_add", "backlog_write_task"):
            assert f"{tool}: " in res["reason"], (
                f"the refusal leg must name {tool}, not just count the refusals: "
                f"{res['reason']}")
        assert res["reason"].count(proof.DENIAL_MARKER) == 3, (
            "one quoted denial per refused call, so the artefact says WHY each was refused: "
            f"{res['reason']}")
        assert run.server.posts().count("refused-unsandboxed") == 3, run.server.posts()
        assert run.server.sessions() == [], (
            "a refused call must not appear in the accepted-session log")
        written = list((run.trees["memory"]).glob("proof-*.md"))
        assert written == [], f"a refused call wrote: {written}"


def test_a_tool_call_denied_body_is_recorded_not_ok_across_a_socket(monkeypatch):
    """Clause 5 at the reader itself: no subprocess, no fixture, and nothing to boot.

    `_refused_call` (`agent_mcp/main.py:768-788`) answers a guard refusal as a normal JSON-RPC
    **result**, so the decoded body has no `error` member at all and its only content text is
    `{"error": "Tool call denied: …", "tool": …}`. That is the premise this node pins first: on
    the pre-#3863 reader (`ok = "error" not in payload`) these three calls were three
    successes, and the trial PASSed on trees that were clean precisely because nothing was
    written.

    The body is built by the shipped function and serialized the way the shipped model
    serializes it (`model_dump()`), not typed here, and served over a real socket so
    `mcp_tool_call`'s own `ok` expression is what is under test. Both spellings of the error
    flag go over that socket — `is_error` is what `CallToolResult.model_dump()` yields on this
    mcp version, `isError` is the MCP wire name, and `agent_mcp/main.py:_result_is_error`
    (:369-380) exists because both occur. A third response, an ordinary successful write, is
    served too: a reader that condemned every body would keep this trial honest but useless,
    so the arm that must stay green is the one that is not a refusal.

    `denial_journal.record` is stilled for the duration because `_refused_call` writes a row
    through it, and this node is in the product's own process: the body is what is being
    measured, and a test row in the live journal is not.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from agent_mcp.main import _refused_call

    monkeypatch.setattr("app.harness.denial_journal.record", lambda **_kw: None)
    refused = _refused_call(
        "memory_add",
        "a call that can change state arrived with no session id", guard="sessionless",
        session_id=None)
    snake = refused.model_dump()
    camel = dict(snake, isError=snake["is_error"])
    del camel["is_error"]
    accepted = dict(snake, is_error=False)
    accepted["content"] = [{"type": "text",
                            "text": '{"record_path": "/copy/lloyd/proof-x.md"}'}]

    def _url_for(result_body: dict) -> str:
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):   # noqa: A002 — base's own name
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                out = json.dumps({"jsonrpc": "2.0", "id": 1,
                                  "result": result_body}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.handle_request, daemon=True).start()
        return f"http://127.0.0.1:{server.server_port}/mcp"

    # The premise, first, asserted on KEYS and not on the text: a refusal really is a
    # `result`, the JSON-RPC envelope carries no `error` member and neither does the result
    # object, so the pre-#3863 test `ok = "error" not in payload` was TRUE of it. The word
    # `error` does appear in the body — inside the content text, as the key of the JSON the
    # guard stringifies (`{"error": "Tool call denied: …", "tool": …}`), which is exactly why a
    # reader has to look at the marker and the flag rather than at the envelope.
    assert "error" not in snake, (
        f"the shipped refusal must not carry an `error` field of its own: {snake}")
    assert snake["content"][0]["text"].startswith('{"error": "Tool call denied'), (
        f"the marker lives in the content text, and this node's reader arm depends on it: "
        f"{snake['content'][0]['text']}")

    # A third form: the marker with NO error flag at all — what a proxy that re-wraps the
    # content and drops the flag sends. Only the marker arm of `denied_call` can see it, so
    # without this row the flag arms could stay green while the marker arm was dead code.
    rewrapped = {"content": [{"type": "text", "text": snake["content"][0]["text"]}]}
    for label, body in (("is_error", snake), ("isError", camel), ("marker-only", rewrapped)):
        call = proof.mcp_tool_call(_url_for(body), "memory_add", {},
                                   session_id="bench-clause5")
        # The envelope as actually received: a refusal arrives as a `result`, so the JSON-RPC
        # `error` member the old reader looked for is absent on the wire and not just in
        # principle.
        assert "error" not in call["response"], (
            f"the {label} response really did carry a JSON-RPC `error` member, which would "
            f"mean the pre-#3863 reader was fine after all: {call['response']}")
        assert call["ok"] is False, f"a {label} refusal was recorded as a write: {call}"
        assert "Tool call denied" in call["denial"], (
            f"the {label} refusal is not in the record's `denial`: {call}")
        assert proof.denied_call({"result": body}) is not None, label

    ok_call = proof.mcp_tool_call(_url_for(accepted), "memory_add", {},
                                  session_id="bench-clause5")
    assert ok_call["ok"] is True, f"a clean write was read as a refusal: {ok_call}"
    assert proof.denied_call({"result": accepted}) is None
    assert proof.denied_call({"result": {"content": [{"type": "text",
                                                     "text": "fine"}]}}) is None


def test_the_copy_side_matcher_routes_memory_to_the_writers_own_directory():
    """Clause 4's other side, in-process: the copy's git-diff witness matches `lloyd/`.

    `copy_side_witness` answers "did the write land in the COPY?" by matching each
    `git diff --name-only` path against the directory of the store each tool was asked to write,
    through `_under_store` → `store_dirname`. For `memory` that directory is the basename of the
    shipped writer's `MEMORIES_ROOT` — `lloyd/` — and not the word `memory`, which in this vault
    names the dated daily-notes directory `memory_add` cannot write.

    Keyed on the label, a probe that landed in the copy's `lloyd/MEMORY.md` matched NO store, and
    `landed_by_git_diff` — the field the owed on-box record is read for, and the only leg that
    can tell a write-in-the-copy from a trial that wrote nothing at all — came back without
    `memory_add` on a trial that had just written it. That was the same bug clause 4 names on the
    live side, present on both sides of the boundary at once (item #3863,
    `## Findings (round SM_20261106_213639)`).

    The live-tree half is `test_the_memory_tree_is_the_directory_the_shipped_writer_resolves` and
    the end-to-end half is the trial node above it; this one is here so the matcher alone, with
    no git repo and no child process, is what a change to it has to answer to.
    """
    from agent_mcp.session import MEMORIES_ROOT

    intended = {"memory_add": "memory", "fact_add": "facts",
                "backlog_write_task": "backlog"}
    written = proof.copy_side_witness(None, intended=intended,
                                      diff_paths=["lloyd/MEMORY.md",
                                                  "backlog/9-proof.md"])
    assert "memory_add" in written["landed_by_git_diff"], (
        f"a write in the copy's {Path(MEMORIES_ROOT).name}/ directory was not credited to "
        f"memory_add: {written}")
    assert "backlog_write_task" in written["landed_by_git_diff"], written
    assert written["not_landed_by_git_diff"] == ["fact_add"], (
        "facts lives under the fixture's DATA root, outside the copied vault's git repo, so it "
        f"is the one store this witness cannot see: {written}")

    daily = proof.copy_side_witness(None, intended=intended,
                                    diff_paths=["memory/2026-11-07.md"])
    assert daily["landed_by_git_diff"] == [], (
        "a change in the daily-notes directory credited memory_add: that is the directory the "
        f"writer does NOT use, so crediting it is a witness that passes on a non-write: {daily}")
    assert daily["landed"] is False and daily["not_landed_by_git_diff"] == [
        "backlog_write_task", "fact_add", "memory_add"], daily
    assert proof.store_dirname("memory") == Path(MEMORIES_ROOT).name != "memory", (
        "the matcher and the watched tree must agree on one name, and it is the writer's")


def test_the_memory_tree_is_the_directory_the_shipped_writer_resolves(tmp_path):
    """Clause 4 itself: the watched memory tree IS `MEMORIES_ROOT`, and not `VAULT_ROOT/'memory'`.

    The clause names the live side, so the live side is asserted here first, in this process,
    against the shipped attribute and against the wrong path spelled out — nothing is booted
    and nothing is patched, so the assertion is readable without running the file.
    `test_trees_resolve_through_app_paths_and_an_empty_tree_is_a_refusal` covers the other
    shape: all four roots through a stand-in `app.paths`, which is what proves resolution
    happens per call rather than at import.

    Then the copy-side half, which is where the same bug wore different clothes: the stand-in
    aggregator's memory store is `agent_mcp.session.MEMORIES_ROOT` as resolved BY THE CHILD on
    its own roots, so a trial's copy-side witness is watching the directory the tool can
    actually reach. If the two ever disagree, the disagreement has to fail here rather than
    show up as a `copy_side` verdict that silently omits `memory` — which is the failure mode
    the live-side bug had, and it would have been invisible on both sides at once.
    """
    from agent_mcp.session import MEMORIES_ROOT as shipped_memories_root
    from app import paths as app_paths

    wrong = Path(app_paths.VAULT_ROOT) / "memory"
    live = proof.durable_write_trees()["memory"]
    assert live == Path(shipped_memories_root), (
        f"the memory tree watched for a leaked write is {live}, not the directory the shipped "
        f"writer resolves ({shipped_memories_root})")
    assert live != wrong, (
        f"the pre-#3863 entry named {wrong}, the daily-notes directory `memory_add` cannot "
        "write and which is not empty, so the vacuity refusal could never fire on it — a leak "
        f"into {wrong / 'MEMORY.md'} would have read CLEAN")

    with _trial(tmp_path, _copy_roots) as run:
        res = run.run()
        memory_paths = [json.loads(c["result_text"])["record_path"]
                        for c in res["tool_calls"] if c["tool"] == "memory_add"]
        assert len(memory_paths) == 1, res["tool_calls"]
        rel = Path(memory_paths[0]).relative_to(run.fixture.vault.resolve())
        assert rel.parts[0] == proof.store_dirname("memory") == "lloyd", (
            f"memory_add landed in {rel}, not in the directory the shipped writer names")
        assert "memory_add" in res["copy_side"]["landed_by_git_diff"], res["copy_side"]


_MISSING = object()


@contextlib.contextmanager
def _patched(obj, **named):
    """Set attributes on `obj` for the duration of the block and restore them after."""
    undo = [(name, getattr(obj, name, _MISSING)) for name in named]
    for name, value in named.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, old in reversed(undo):
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)


class _FakeHandle:
    """What `bench_spawner.spawn` returns, minus the subprocess.

    Only the parts `boot_fixture_aggregator` and `main` use: `url`, `env`, `stop()`. The url is
    the REAL stand-in server's, and it is a bare origin (`bench_spawner.py:362` builds
    `f"http://{host}:{chosen}"`), because the point of this seam is that the script must append
    `/mcp` itself — a handle whose url already carried the route would let a script that forgot
    the suffix pass.
    """

    def __init__(self, url: str, env: dict):
        self.url = url
        self.env = env
        self.stop_calls = 0

    def stop(self, **_kw):
        self.stop_calls += 1
        return (0, None)


class _FakeFixture:
    """The `BenchFixture` shape as far as the boot reads it: a directory and its scopes."""

    def __init__(self, root: Path, scopes):
        self.root = root
        self.writes = list(scopes)


@contextlib.contextmanager
def _fake_spawner(handle: _FakeHandle):
    """Stand in for the two spawner entry points the boot uses; leave `auth_headers` the real one.

    `auth_headers` stays real on purpose. It is the function that turns the CHILD's env into a
    credential, and faking it would hide the mismatch this seam has to get right: the token
    `spawn` injects into the child is not the one this process reads off its own
    `LLOYD_MCP_TOKEN_FILE`. What is stood in for is the process launch, whose own behaviour —
    the live-root guards, the fixture marker test, the credential injection, the wait for
    `state_is_copied` — is `tests/test_autoresearch_bench_spawner.py`'s subject and not this
    file's. What is tested HERE is that this script goes THROUGH that module, in that order,
    appends `/mcp` to a bare origin, and stops what it started.
    """
    from scripts.autoresearch import bench_spawner as real

    calls: list[tuple] = []

    def fixture_from_dir(root, scopes):
        calls.append(("fixture_from_dir", Path(root), tuple(scopes)))
        return _FakeFixture(Path(root), scopes)

    def spawn(*, fixture_root, scopes):
        calls.append(("spawn", fixture_root, tuple(scopes)))
        return handle

    with _patched(real, fixture_from_dir=fixture_from_dir, spawn=spawn):
        yield calls


def test_the_cli_boots_its_own_aggregator_from_a_built_fixture_and_stops_it(tmp_path, capsys,
                                                                            monkeypatch):
    """Clause 3, the boot half: given no urls, `main()` brings the server up and takes it down.

    The spawner is stood in for at its own function boundary while everything the boot drives
    is real: the state read and the three tool calls cross a socket to a subprocess rooted on
    the copy, the record is written by `main`, and the copy-side witness is a real `git diff`
    over a fixture `build_fixture` built. This is the composition on the other side of the
    guard — the row the first attempt of this item could not produce because `main` had no
    boot path and every prior node entered through `run_proof`.

    While `main` runs, THIS process holds a different credential from the child
    (`DECOY_TOKEN`, written to a file and pointed at through `LLOYD_MCP_TOKEN_FILE`, restored
    by the context before the `/mcp` probe below). Without that split the two sources of the
    trial credential are the same bytes and the node cannot see which one the boot used — see
    `DECOY_TOKEN`. It is the only thing that makes the credential half of clause 3 falsifiable:
    `boot_fixture_aggregator` may ask the child with `spawner.auth_headers(handle.env)` or fall
    back to `auth_headers(handle.url + '/state')`, and only one of those is answered.
    """
    with _trial(tmp_path, _copy_roots) as run:
        handle = _FakeHandle(run.server.url,
                             {"LLOYD_MCP_TOKEN_FILE": str(run.server.token_file)})
        out = tmp_path / "record.json"
        decoy = tmp_path / "parent-token"
        decoy.write_text(DECOY_TOKEN + "\n", encoding="utf-8")
        with _fake_spawner(handle) as calls:
            with _patched(proof, durable_write_trees=lambda *a, **k: dict(run.trees)):
                with monkeypatch.context() as gated:
                    gated.setenv("LLOYD_MCP_TOKEN_FILE", str(decoy))
                    code = proof.main(["--fixture-root", str(run.fixture.root),
                                       "--window-seconds", "0",
                                       "--trial-id", "cli-boot", "--out", str(out)])
        assert code == 0, (
            f"exit {code}, expected 0: with the parent process holding a different credential "
            f"than the child, a boot that asks the child with its OWN credential is answered "
            f"401 by the state read and refuses. stderr was: {capsys.readouterr().err}")
        # Snapshot the endpoint's log NOW, before this node sends anything of its own. The
        # stand-in notes `posted` on every POST it receives, including the ones it 404s, so a
        # suffix probe sent later would land in the same log. What is being pinned is that
        # `main`'s three trial calls each reached `/mcp` and each wrote: three `posted` paired
        # with three `wrote`, and no `bad-path` or `refused-*` token anywhere in between. Had
        # `main` POSTed the bare origin, Starlette's 404 would have made this
        # ['posted', 'bad-path'] * 3 and `code` 1, not 0.
        trial_posts = run.server.posts()
        assert trial_posts == ["posted", "wrote"] * 3, trial_posts
        assert [(c[0], c[2]) for c in calls] == [
            ("fixture_from_dir", proof.SPAWN_SCOPES),
            ("spawn", proof.SPAWN_SCOPES)], (
            "the boot must re-derive the fixture from the directory with the scopes it will "
            f"ask to be armed, then spawn on that fixture, in that order: {calls}")
        assert str(calls[0][1]) == str(run.fixture.root), calls[0]
        # `spawn` takes the FIXTURE, not the path — the `BenchFixture` the previous call
        # returned, which is what carries the roots the child is booted on.
        assert calls[1][1] is not None and Path(calls[1][1].root) == run.fixture.root, calls[1]

        # The scopes have to be ones the tools under test can actually be armed with, or the
        # trial's three calls all die at the scope check and the record reads as a
        # fully-refused run. Read from the shipped table, not from this file's opinion.
        from scripts.autoresearch.bench_runner_sdk import WRITE_SCOPE_TOOLS
        for row in proof.PROOF_TOOLS:
            assert WRITE_SCOPE_TOOLS[row["tool"]] in proof.SPAWN_SCOPES, (
                f"{row['tool']} could never be armed by {proof.SPAWN_SCOPES}: the table names "
                f"{WRITE_SCOPE_TOOLS[row['tool']]!r} as its scope word")

        # The handle is a bare origin (`bench_spawner.py:362`), so `/mcp` is the script's own
        # work — and the stand-in 404s any other path the way Starlette does, which is what
        # makes the suffix falsifiable rather than assumed. `trial_posts` above is the
        # falsifier: POSTing `handle.url` alone gives three 404s, so `trial_posts` reads
        # ['posted', 'bad-path'] * 3 and `code` is 1. The probe below then proves the OTHER
        # half — that the stand-in really does refuse a non-`/mcp` path, so the token
        # `bad-path` in `trial_posts` would have meant a missing suffix and not a server that
        # answers everything.
        assert not handle.url.endswith("/mcp"), handle.url
        bare_origin = proof.mcp_tool_call(handle.url, "memory_add", {"entry": "e"},
                                         session_id="bench_suffix-probe", timeout=20.0)
        assert bare_origin["ok"] is False and "404" in bare_origin["detail"], bare_origin
        assert "bad-path" in run.server.posts(), (
            "the stand-in answered a request outside /mcp, so the suffix proves nothing")
        assert handle.stop_calls == 1, (
            "a script that boots a server has to stop it: a leaked fixture-routed aggregator "
            f"is a second writer on this box ({handle.stop_calls} stops)")

        assert run.server.posts() == trial_posts + ["posted", "bad-path"], run.server.posts()
        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["verdict"] == "PASS", record["reason"]
        assert record["session_id"].startswith("bench_"), record["session_id"]
        assert run.server.sessions() == [f"{name} {record['session_id']}" for name in
                                        ("memory_add", "fact_add", "backlog_write_task")], (
            "the server's own account of which session it wrote for must match the record")
        assert record["copy_side"]["landed_by_git_diff"] == ["backlog_write_task", "fact_add",
                                                            "memory_add"], record["copy_side"]


def test_the_cli_still_refuses_a_production_server_without_letting_a_write_through(tmp_path,
                                                                                  capsys):
    """Clause 3, the refusal half: a production-routed server exits 3 with no tool call at all.

    The server is the house-routed stand-in — a real subprocess running real
    `_tool_sandbox.status()`, `IS_PRODUCTION_VAULT` true because its `VAULT_ROOT` really is this
    box's vault. "Having booted nothing" has two readings and only one of them is literally
    available to this script, so both are pinned:

    * given its own urls, `main` never consults the spawner — `calls == []` below, nothing was
      booted, in the literal sense;
    * booting one's own is the only way to LEARN the flags, so on that path the guarantee that
      exists is the one checked after: the refusal came before a single `tools/call` reached the
      endpoint, and the handle was stopped again. Nothing was attempted, and the thing that
      could have attempted it is gone.
    """
    with _trial(tmp_path, _house_roots) as run:
        handle = _FakeHandle(run.server.url,
                             {"LLOYD_MCP_TOKEN_FILE": str(run.server.token_file)})
        with _fake_spawner(handle) as calls:
            with _patched(proof, durable_write_trees=lambda *a, **k: dict(run.trees)):
                code = proof.main(["--aggregator-url", run.server.url + "/mcp",
                                   "--state-url", run.server.url + "/state",
                                   "--fixture-root", str(run.fixture.root),
                                   "--window-seconds", "0", "--trial-id", "cli-house",
                                   "--out", str(tmp_path / "house-record.json")])
        record_path = tmp_path / "house-record.json"
        assert code == 3, capsys.readouterr().out
        assert calls == [], (
            "the caller supplied the urls, so the spawner must not be consulted even though "
            f"the outcome is a refusal either way: {calls}")
        assert run.server.posts() == [], (
            "a production-reporting server must never see a tools/call from this script: the "
            f"guard runs before the first write, not after it. Saw {run.server.posts()}")
        # The record is the readable form of the refusal: a `verdict: REFUSED` whose reason
        # names the server's own flag, so the artefact says why there is no measurement in it
        # rather than leaving a missing file to be read as a forgotten run.
        refusal = json.loads(record_path.read_text(encoding="utf-8"))
        assert refusal["verdict"] == "REFUSED", refusal
        assert "is_production_vault" in refusal["reason"], refusal["reason"]
        assert refusal["tool_calls"] == [], refusal["tool_calls"]
        assert refusal["isolation_flags"]["is_production_vault"] is True, refusal

        handle2 = _FakeHandle(run.server.url,
                              {"LLOYD_MCP_TOKEN_FILE": str(run.server.token_file)})
        with _fake_spawner(handle2) as calls2:
            with _patched(proof, durable_write_trees=lambda *a, **k: dict(run.trees)):
                code2 = proof.main(["--fixture-root", str(run.fixture.root),
                                    "--window-seconds", "0", "--trial-id", "cli-house-self"])
        assert code2 == 3, capsys.readouterr().out
        assert [c[0] for c in calls2] == ["fixture_from_dir", "spawn"], calls2
        assert handle2.stop_calls == 1, (
            "the aggregator this script booted has to be stopped by the same call that "
            "decided not to use it")
        assert run.server.posts() == [], (
            f"the self-booted production server received tool calls: {run.server.posts()}")


def test_the_cli_refuses_a_half_named_aggregator_without_booting_anything(tmp_path):
    """One url without the other is refused before the spawner is consulted at all.

    `--aggregator-url` names the tool endpoint, `--state-url` the guard's endpoint; a run with
    only the first would have somewhere to write and nowhere to ask whether writing is
    permitted — the order that has to be impossible. Both shapes exit 3 with the spawner
    untouched, which is also what keeps `--plan`'s no-boot behaviour a special case of this
    rule rather than a fourth rule.
    """
    with _trial(tmp_path, _copy_roots) as run:
        handle = _FakeHandle(run.server.url,
                             {"LLOYD_MCP_TOKEN_FILE": str(run.server.token_file)})
        for argv in (["--aggregator-url", run.server.url + "/mcp"],
                     ["--state-url", run.server.url + "/state"]):
            with _fake_spawner(handle) as calls:
                code = proof.main(argv + ["--window-seconds", "0"])
            assert code == 3, argv
            assert calls == [], f"{argv} reached the spawner: {calls}"
        assert run.server.posts() == []
