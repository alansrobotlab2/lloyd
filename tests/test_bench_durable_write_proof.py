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

    Resolution happens per call, not at import, so patching the module's attributes proves
    the routine reads them rather than carrying a copy: the stand-in roots below are nowhere
    under `~/obsidian`.
    """
    stand_in = types.SimpleNamespace(VAULT_ROOT=Path("/stand-in/vault"),
                                     VAULT_FACTS_ROOT=Path("/stand-in/facts-tree"),
                                     DATA_ROOT=Path("/stand-in/data"))
    assert proof.durable_write_trees(resolved=stand_in) == {
        "memory": Path("/stand-in/vault/memory"),
        "facts": Path("/stand-in/facts-tree"),
        "backlog": Path("/stand-in/vault/backlog"),
        "data": Path("/stand-in/data")}, (
        "memory and backlog must come from VAULT_ROOT, facts from VAULT_FACTS_ROOT and "
        "data from DATA_ROOT, read at call time")

    live = proof.durable_write_trees()
    assert set(live) == {"memory", "facts", "backlog", "data"}
    assert Path(live["facts"]) != Path.home() / "obsidian" / "facts", (
        "facts must be VAULT_FACTS_ROOT: the literal ~/obsidian/facts holds 0 files and "
        "manifesting it proves nothing")
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


def test_the_cli_refuses_without_an_aggregator_rather_than_inventing_one(capsys):
    """No urls means no isolated server to write through, and that is a refusal (3).

    Not a usage error (2): the message names who owns the missing piece — #3688's spawner,
    or the scope ruling on #3689 that says this script boots `agent_mcp` itself. A bare
    `argparse` error would read as "someone forgot a flag" when the truth is that the trial
    has nowhere to run.
    """
    assert proof.main([]) == 3
    assert "#3688" in capsys.readouterr().err


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
    def leaking_call(name: str, arguments: dict) -> dict:
        leak.write_text("a durable write reached the live tree\n", encoding="utf-8")
        return {"tool": name, "ok": True}

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
    (["memory/MEMORY.md"], True),                   # one intended store
    (["backlog/7-bench-durable.md"], True),         # another
    (["memory/MEMORY.md", "unrelated/x.md"], True)  # intended target plus noise
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
    result = _run(tmp_path, diff_paths=["memory/MEMORY.md"])
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

    def call_tool(name: str, arguments: dict) -> dict:
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

#: Store directory per tool, resolved by THIS process's roots and never handed in by the
#: test — which is the whole point of booting a child instead of patching a dictionary.
STORES = {"memory_add": P.VAULT_ROOT / "memory",
          "fact_add": P.VAULT_FACTS_ROOT,
          "backlog_write_task": P.VAULT_ROOT / "backlog"}
#: Every POST, in arrival order, including the ones refused before they reached a tool. The
#: witness for "the guard refused, so nothing arrived": a refusal that had reached the tool
#: would leave a line here.
POST_LOG = os.environ.get("STANDIN_POST_LOG", "")
TOKEN = read_token(publish=False)


def _note(text):
    if POST_LOG:
        with open(POST_LOG, "a", encoding="utf-8") as handle:
            handle.write(text + "\\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
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
        rpc = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0))
        name, args = rpc["params"]["name"], rpc["params"]["arguments"]
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
#: and never an ambient token this test process inherited.
_ROUTED_ENV = ("LLOYD_DATA", "LLOYD_VAULT_ROOT", "LLOYD_VAULT", "LLOYD_FACTS_ROOT",
               "LLOYD_KG_DB", "LLOYD_MCP_TOKEN")

#: 26 characters of the alphabet `aggregator_auth._looks_valid` (:112-114) accepts — at
#: least 16 of alphanumerics, `-` and `_` — so `read_token(publish=False)` returns this
#: instead of rejecting it as malformed and answering every request with 401.
PROBE_TOKEN = "probe-token-0123456789abcdef"


def _live_stub(tmp_path: Path):
    """The four watched trees as small stand-ins, same shapes, same resolution rules."""
    vault = tmp_path / "live-vault"
    (vault / "memory").mkdir(parents=True)
    (vault / "backlog").mkdir(parents=True)
    (vault / "memory" / "MEMORY.md").write_text("# index\n", encoding="utf-8")
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

    def __init__(self, url: str, post_log: Path):
        self.url = url
        self.post_log = post_log

    def posts(self) -> list[str]:
        return ([] if not self.post_log.exists()
                else self.post_log.read_text(encoding="utf-8").split())


@contextlib.contextmanager
def _aggregator(tmp_path: Path, roots: dict, *, write_scope=("memory_add", "fact_add",
                                                             "backlog_write_task")):
    """Boot the stand-in with `roots` as its `app.paths`, and the credential on file.

    A `None` value means "let the child resolve that root itself", which is how a node gets a
    child that really is production-routed rather than one that has been told to pretend.
    """
    token_file = tmp_path / "aggregator-token"
    token_file.write_text(PROBE_TOKEN + "\n", encoding="utf-8")
    script = tmp_path / "standin_aggregator.py"
    script.write_text(_STANDIN_AGGREGATOR, encoding="utf-8")
    post_log = tmp_path / "post-log.txt"

    child_env = {key: value for key, value in os.environ.items() if key not in _ROUTED_ENV}
    child_env["STANDIN_REPO"] = str(REPO)
    child_env["STANDIN_POST_LOG"] = str(post_log)
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
        yield _Server(url, post_log)
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
            call_tool=lambda name, args: proof.mcp_tool_call(
                self.server.url + "/mcp", name, args, timeout=20.0),
            fixture_root=self.fixture.root, sleep=time.sleep, clock=time.monotonic,
            window_seconds=window_seconds, trial_id="abc")
        return self.record


@contextlib.contextmanager
def _trial(tmp_path: Path, roots_for, *, write_scope=("memory_add", "fact_add",
                                                      "backlog_write_task")):
    """One assembled trial: real builder, real subprocess server, state read over the wire.

    Yields a `_TrialRun` with the state already fetched (that read is the guard's input, so
    a node that wants to inspect what the server really said can) and the watched stand-in
    trees. Call `.run()` to drive `run_proof` exactly as the CLI does.
    """
    vault, data = _live_stub(tmp_path)
    trees = {"memory": vault / "memory", "backlog": vault / "backlog",
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
                         write_scope=write_scope) as server:
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
        good = proof.mcp_tool_call(server.url + "/mcp", "memory_add",
                                   {"entry": "carried by the file read abc"}, timeout=20.0)
        assert good["ok"] is True and json.loads(good["text"])["record_path"].endswith(
            "proof-abc.md"), good

        assert server.posts() == ["posted", "unauthorised"] * 2 + ["posted", "wrote"]
        assert "refused-scope" not in server.posts(), (
            "an unauthenticated POST must be refused before the scope check runs")
        written = sorted(p.name for p in vault.glob("memory/proof-*.md"))
        assert written == ["proof-abc.md"], (
            "the two refused calls must have written nothing to the store")

        # The state leg: a credential that does not match the server's is a refusal, not a
        # body — `fetch_state_json` raises rather than handing back a 401 page to parse.
        other = tmp_path / "other-token"
        other.write_text("another-token-0123456789abcdef\n", encoding="utf-8")
        os.environ["LLOYD_MCP_TOKEN_FILE"] = str(other)
        with pytest.raises(proof.ProofRefused):
            proof.fetch_state_json(server.url + "/state")
