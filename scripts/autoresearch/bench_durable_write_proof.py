#!/usr/bin/env python3
"""The planted durable-write proof: did a fixture-routed write touch the house? (#3403 owed 3)

#3403's safety claim is "a fixture-routed write never touches the house": the trial runs
against a disposable copy (`scripts/autoresearch/bench_fixture.py`), and the aggregator's
own gate (`agent_mcp/_tool_sandbox.py`) refuses a durable write it cannot prove is running
on copied state. Both halves are code. Neither has ever been WITNESSED: `state_is_copied`
appears only in that module and in the runner, `~/lloyd-data/eval/fixtures/` does not
exist so `build_fixture` has never run on this box, and 0 of the 28 cards in
`~/obsidian/lloyd/bench/` declare a `side_effect_checks:` block. A claim with no produced
artifact is a claim a future round can quietly widen. This script produces the artifact.

Copy-side proof already exists and is NOT rebuilt here: rule 3 of `bench_fixture.py` gives
the copy its own `git init` + baseline commit so "what did this run write" is one
`git diff --name-only`. What was missing is the LIVE side — an assertion about the four
trees the three tools could reach if isolation were broken. Three decisions make that
assertion mean something on a machine that is running while the trial runs.

**The trees come from `app.paths`, never from a literal.** #3689 as filed names
`~/obsidian/facts`, and that directory exists and holds **0 files**: the real facts root is
`app.paths.VAULT_FACTS_ROOT` (`app/paths.py:310-311`), which is
`~/lloyd-data/_pipeline/vault-derived/facts` — 75,562 files on 2026-11-03 — and is
redirectable by `LLOYD_FACTS_ROOT`. Manifesting the literal would have "proved" the
facts tree clean by reading a directory nothing writes to. A tree that resolves to zero
files is therefore a REFUSAL (`TREE_EMPTY`), not a clean diff, for the same reason a
refused read is not an empty result anywhere else in this repo: the pass and the
vacuity look identical otherwise.

The `memory` tree is the same trap wearing different clothes, and #3863 clause 4 closed it: it
used to be `VAULT_ROOT/'memory'`, which is not nothing — it is full of dated daily notes, so it
is never refused as vacuous — but it is a directory `memory_add` cannot write. The shipped
writer resolves `agent_mcp/session.py:58` `MEMORIES_ROOT = vault_root() / 'lloyd'`, so
`~/obsidian/lloyd/MEMORY.md`, the file this instance is loaded from, was the one leak the
instrument was built to catch and was not being watched. `memories_root()` reads that symbol
instead of re-spelling a path, and `store_dirname()` is the same fact on the copy side.

**A zero-diff verdict is unreachable, so the verdict is ATTRIBUTED.** Measured on
2026-11-01 over one 120 s window with no trial running at all: 42 files under
`~/lloyd-data` changed (logs, sessions, event logs, `_pipeline`, the `workers.db` and
`usage.db` WAL/SHM sidecars) and 2 under `~/obsidian/backlog` — the box writes to itself
continuously. So `changed == 0` could only ever read RED, and a check that can only fail
is not a check. PASS instead requires the trial window's changed-path set to be a SUBSET
of a same-length control window taken with no trial running: a path that moved during the
trial and also moves by itself is background, and a path that moved ONLY during the trial
is what this proof is looking for. Same-length, not merely "a window": a longer control
sees more background and forgives a real write, which is how the item's "equal-length"
wording earned its place.

**The flags are read where the real server puts them, and the question is asked with the
credential on.** `GET /state` answers with `tool_sandbox: _tool_sandbox.status()`
(`agent_mcp/main.py:909`), whose `is_production_data` / `is_production_vault` /
`state_is_copied` / `allowed_writes` sit FLAT *inside that block* — the harness's own reader
`bench_runner_sdk.require_copied_state` reads `body["tool_sandbox"]` and refuses on a missing
block or a missing key (`_SERVER_ISOLATION_KEYS`, membership and never `.get(...) or False`,
because an aggregator older than #3403 reports no isolation block at all). Reading the three
flags off the top level of the body instead would default them to the production side on
every answer a real aggregator ever sends: fail-closed, so never unsafe, but an instrument
that can only ever refuse is an instrument that never produces the artifact it exists to
produce. So this script reads the same block the same way, and carries
`X-Lloyd-Aggregator-Token` on BOTH requests, because only `/health` is in `OPEN_PATHS`
(`agent_mcp/aggregator_auth.py:95`) — a credential-less probe of `/state` is a 401, and
a test that passed against such a probe would be proving a request the real server rejects.
The header comes from `agent_mcp.aggregator_auth.headers_for_url`, the one definition of the
header name and of where the token lives, shared with the process that checks it.

**Every call names the session it belongs to, and a refusal is not an answer that passed.**
Two more seams had to be wired before a call could land anywhere (`#3863`, this round). This
server is stateless (`agent_mcp/main.py:9-13`), so a `tools/call` with no session id has no
session; all three trial tools are state-changing
(`_tool_sandbox.state_changing_tool` is `name not in READ_ONLY`, `:366-378`), so the handler
refused them all (`main.py:595-602`) before this script put an id at
`params._meta['lloyd/session_id']` — the key `META_SESSION_ID` (`main.py:225`) reads, and
`SANDBOXED_ID_PREFIXES` (`_tool_sandbox.py:60`, matched at `:170`) decides which ids are
bench/eval trials, so the prefix is READ from it rather than typed. The refusal that used to
be the whole story arrives as an ordinary JSON-RPC *result* whose text is
`{"error": "Tool call denied: …"}` (`main.py:_refused_call`, `:784-790`), which the old
`ok = "error" not in payload` line scored as success. That pairing is the reason this
instrument would have printed PASS on a fully-refused trial: three refusals write nothing, so
the four live trees stay pristine, the equal-length control explains every path, and the only
leg left standing was the one that could not tell a refusal from a write. `denied_call` is now
the reader, and `copy_side_witness` still forces FAIL when the copy received nothing.

**The manifest keys on `(size, mtime_ns)` and walks with `lstat`.** `~/lloyd-data` holds
about 710,000 files — `find` returned 709,991, 709,992 and 710,103 within one session on
2026-11-03, so the figure is an order of magnitude because the tree is being written under
it — and hashing that twice per trial is not a check anyone would run. One full
`(size, mtime_ns)` walk of all four trees measured 790,516 paths in 12 seconds on this box,
which is a price a trial pays twice. Two consequences are accepted out loud: a rewrite that preserves both size and mtime is
not observed (every writer reachable from these three tools rewrites content and so moves
`mtime_ns`), and `lstat` means a symlinked directory is recorded as a link and NOT
traversed — which is the point, since following links out of a tree would let the manifest
report a clean tree while a write landed through a symlink.

Exit: `0` PASS · `1` FAIL (the diff, the copy witness or the tool results say the claim
did not hold) · `3` refused (production state, an empty tree, a fixture destination
`build_fixture` will not build, or no aggregator to ask — no write attempted) · `2` argparse
usage.

    # ONE command: build the copy, boot the aggregator on it, run the trial, stop it (#3915).
    # `build_fixture` refuses a destination inside either PRODUCTION tree, and
    # `bench_spawner.spawn` refuses to boot on one either, so on a box whose data root IS
    # ~/lloyd-data the scratch root has to be named rather than defaulted:
    LLOYD_DATA=/scratch/durable-write-proof \
    .venvs/lloyd/bin/python -m scripts.autoresearch.bench_durable_write_proof \
        --build-fixture durable-write-<ts> --out <record.json>
    # boot the aggregator from a fixture that is already built, and stop it on the way out:
    scripts/autoresearch/bench_durable_write_proof.py \
        --fixture-root <data-root>/eval/fixtures/<trial> --out <record.json>
    # or point it at one that is already running (nothing is booted on that path):
    scripts/autoresearch/bench_durable_write_proof.py --aggregator-url http://127.0.0.1:8819/mcp \
        --state-url http://127.0.0.1:8819/state \
        --fixture-root <data-root>/eval/fixtures/<trial> --out <record.json>

What is still owed after this lands (#3689's owed list) is the RUN itself, not the wiring: a
built fixture on this box (`ls ~/lloyd-data/eval/fixtures` answers `No such file or directory`
as of 2026-11-07, so `build_fixture` has still never run here) and one trial against it, both
of which `--build-fixture` now reaches in one command. `--plan` prints what a trial WOULD
check against the live trees without a server or a write, which is the mode a reader can run
today.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import paths  # noqa: E402
from scripts.autoresearch.bench_fixture import (  # noqa: E402
    FixtureRefused, build_fixture)

logger = logging.getLogger(__name__)

#: The three durable-write tools, each with the store it was asked to write.
#:
#: `memory_add` writes the entity markdown file AND the `facts_idx` row inside
#: `kg.sqlite` (a transaction that can leave a markdown file and no index row), and
#: `backlog_write_task` writes under `backlog/`; those are the shapes the copy-side check
#: looks for. Keys are stable because they go in the record.
PROOF_TOOLS: tuple[dict[str, str], ...] = (
    {"tool": "memory_add", "store": "memory"},
    {"tool": "fact_add", "store": "facts"},
    {"tool": "backlog_write_task", "store": "backlog"},
)

#: How long each window is when nobody overrides it. 120 s is the interval the triage
#: measurement was taken over, so the number a first PASS is judged against is the number
#: that established the background-write rate.
DEFAULT_WINDOW_SECONDS = 120.0

#: Where the aggregator's state endpoint comes from when the flag is absent. The same
#: variable `scripts/autoresearch/bench_runner_sdk.py:485` reads, named here as a constant
#: so a caller (and a test) can clear it by name instead of spelling the literal twice.
STATE_URL_ENV = "LLOYD_BENCH_STATE_URL"

#: JSON-RPC method for a tool call. `agent_mcp/main.py:794` is the handler, mounted at
#: `/mcp` (`main.py:1119`); `aggregator_auth.py:73` names the one accepted credential
#: header, and the credential is the token file's stripped first line
#: (`aggregator_auth.py:117-121`), never an `Authorization` bearer.
RPC_METHOD = "tools/call"

#: The keys of the server's `tool_sandbox` block that must be PRESENT, not merely truthy.
#: The same four names `bench_runner_sdk._SERVER_ISOLATION_KEYS` (`:509`) lists, and for the
#: reason the comment there gives: MEMBERSHIP is the check, because an aggregator that
#: predates #3403 answers `GET /state` with no isolation block and a `.get(...) or False`
#: reads that silence as "not production". Spelled out rather than imported because
#: `bench_runner_sdk` is an async module a synchronous script should not pull in to ask one
#: question; the test asserts the two lists still agree, so the copy cannot drift.
SERVER_ISOLATION_KEYS: tuple[str, ...] = ("is_production_data", "is_production_vault",
                                          "state_is_copied", "allowed_writes")

#: The `_meta` key carrying the caller's session id — the literal
#: `agent_mcp/main.py:225` (`META_SESSION_ID`) matches on, restated here rather than
#: imported because reading it means importing the whole MCP-SDK aggregator for one
#: string. The test asserts the two spellings are still the same string. This server is
#: STATELESS (`main.py:9-13`: no initialize, no `Mcp-Session-Id`, "every request carries
#: its own context in `_meta`"), so `_meta` is not a nicety — it is the only carrier a
#: single-shot POST has for the session the server keys its gates on.
META_SESSION_ID = "lloyd/session_id"

#: The marker every pre-dispatch refusal carries. `agent_mcp/main.py:_refused_call`
#: (:784-790) answers a guard refusal with a `CallToolResult` whose one content text is
#: `{"error": "Tool call denied: <reason>", "tool": <name>}` — a 200 response with a
#: JSON-RPC `result`, no `error` member. `"error" not in payload` is therefore TRUE for a
#: refused write, which is the false green #3863 clause 5 exists to close. The test feeds
#: this reader the REAL `_refused_call` object's own wire form rather than a paraphrase.
DENIAL_MARKER = "Tool call denied"


def rpc_body(tool: str, arguments: dict[str, Any], *,
             session_id: str = "") -> dict[str, Any]:
    """One `tools/call` envelope, built fresh so no call can see another's arguments.

    `session_id` goes at `params._meta['lloyd/session_id']`, which is where
    `_bound_session_id` (`agent_mcp/main.py:362`) reads it and where the sessionless
    refusal says to put it (`main.py:595-602`). Omitting it is not a neutral choice:
    all three trial tools are state-changing (`_tool_sandbox.state_changing_tool` is
    `name not in READ_ONLY`, `:366-378`), so a call that arrives with no session id is
    refused before dispatch — and, until clause 5 landed, refused in a shell this script
    read as success. An empty id is left off the wire entirely rather than sent as `""`,
    because `main.py:595` tests falsiness: sending an empty string and sending nothing are
    the same request to that guard, and a record that claimed a session id it never sent
    would be the same lie in a different column.
    """
    params: dict[str, Any] = {"name": tool, "arguments": arguments}
    if session_id:
        params["_meta"] = {META_SESSION_ID: session_id}
    return {"jsonrpc": "2.0", "id": 1, "method": RPC_METHOD, "params": params}


def trial_session_id(trial_id: str) -> str:
    """The session id every trial call carries: the trial id under the bench prefix.

    The prefix is READ from `_tool_sandbox.SANDBOXED_ID_PREFIXES` (`:60`, matched at
    `:170`) instead of typed, for the reason the item gives: the sandbox, not taste,
    decides which ids are bench/eval trials. `is_sandboxed_session` is what turns the
    write surface on or off for the session the handler resolves, so an id that merely
    *looks* like a bench id but is not in that tuple gets the read-only refusal — and an
    un-prefixed id gets the sessionless one. A non-importable `agent_mcp` is a
    `ProofRefused`, never a bare trial id: the closed direction, same as `auth_headers`.

    Idempotent on an id that already carries a prefix, so a caller that supplies its own
    bench id gets it back unchanged rather than `bench_bench_…`.
    """
    try:
        from agent_mcp._tool_sandbox import SANDBOXED_ID_PREFIXES
    except Exception as exc:  # noqa: BLE001 — closed: no prefix, no permission to write
        raise ProofRefused(
            f"cannot read `_tool_sandbox.SANDBOXED_ID_PREFIXES` ({exc.__class__.__name__}: "
            f"{exc}); refusing to guess a session id prefix, because a trial that guesses "
            "wrong gets a refusal on every write and would read it as one that worked"
        ) from exc
    sid = (trial_id or "").strip()
    if sid.startswith(tuple(SANDBOXED_ID_PREFIXES)):
        return sid
    return f"{SANDBOXED_ID_PREFIXES[0]}{sid or 'proof'}"


def denied_call(payload: dict[str, Any]) -> str | None:
    """The refusal a `tools/call` response hides inside an ok-shaped body, else None.

    Two arms, because the two halves of `_refused_call`'s output live at different depths
    and a reader that checked only one would go wrong the day the server moved the other:

    * the result's error flag, read under BOTH spellings — `isError` is the wire name and
      `is_error` is the Python attribute name in mcp 2.x, the same split
      `agent_mcp/main.py:_result_is_error` (:369-380) exists for. Reading only the camel
      form is a check that silently stops firing on a serialization change.
    * the marker inside the content text, which is the part a human reads and the part a
      proxy that re-wraps the body preserves.

    Returns the refusal text (truncated) so the RECORD carries why a call did not happen;
    returns None for a body that reports success.
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    errored = any(bool(result.get(key)) for key in ("isError", "is_error"))
    text = "".join(str(part.get("text", "")) for part in (result.get("content") or [])
                   if isinstance(part, dict))
    if errored and text:
        return text[:400]
    if DENIAL_MARKER in text:
        return text[:400]
    return None


class ProofRefused(RuntimeError):
    """The trial must not proceed, and nothing was written. Exit 3, never a FAIL row."""


def memories_root() -> Path:
    """Where the shipped `memory_add` writer lands, read from the writer itself.

    Clause 4 of #3863. `durable_write_trees()` used to name `VAULT_ROOT/'memory'`, which on
    this box is the directory of DATED DAILY NOTES (`~/obsidian/memory/2026-02-22.md`,
    `~/obsidian/memory/2026/`) — a tree `memory_add` cannot write. The tool resolves its
    destination through `agent_mcp/session.py:58` `MEMORIES_ROOT = vault_root() / "lloyd"`
    (`_resolve_file` at `:269,271,275` all go through it), so until now the one tree that
    was supposed to catch a leaked memory write was watching a directory the trial's own
    tool never touches, and a leak into `~/obsidian/lloyd/MEMORY.md` — the file this
    instance is LOADED from — would have read CLEAN and counted as proof of isolation.

    Read as the symbol, not re-spelled as `VAULT_ROOT/'lloyd'`: a move of the memory root
    then moves the watched tree with it, and the test pins the equality against the shipped
    attribute rather than against a second guess of the same path.
    """
    try:
        from agent_mcp.session import MEMORIES_ROOT
    except Exception as exc:  # noqa: BLE001 — closed: an unreadable root is not a clean tree
        raise ProofRefused(
            f"cannot read `agent_mcp.session.MEMORIES_ROOT` ({exc.__class__.__name__}: "
            f"{exc}); refusing to fall back to `VAULT_ROOT/'memory'`, which is the "
            "directory the writer does NOT use and the vacuous tree this instrument was "
            "reading until #3863"
        ) from exc
    return Path(MEMORIES_ROOT)


def store_dirname(store: str) -> str:
    """The one directory a store label names inside a vault root.

    `memory` is not a directory called `memory`: the shipped writer lands in `lloyd/`
    (`memories_root`), so a copy-side match on the label would send the git-diff witness to
    the same wrong place clause 4 just moved the live tree from. `facts` and `backlog` keep
    their label names (`VAULT_FACTS_ROOT`'s own directory, and `backlog/`).
    """
    return Path(memories_root()).name if store == "memory" else store


def durable_write_trees(resolved: Any = None) -> dict[str, Path]:
    """The four live trees a durable write could reach, resolved at call time.

    Read from `app.paths` per call, not at import: the roots are env-overridable
    (`LLOYD_DATA`, `LLOYD_VAULT_ROOT`, `LLOYD_FACTS_ROOT`), and a proof that bound them at
    import would report the machine it was imported on. `facts` is
    `VAULT_FACTS_ROOT` and NOT the empty `~/obsidian/facts` the item names — see the
    module docstring. `data` is the whole data root, which subsumes the facts root; the
    overlap is deliberate (a changed file appears under both names, and `paths` are
    absolute so the two key spaces never collide.

    `memory` is the one entry NOT derived from `resolved`: a watched tree has to be the
    directory the tool under test can actually reach, and that is a fact about
    `agent_mcp/session.py`, not about this module's root table. It comes from
    `memories_root()` — the shipped writer's own `MEMORIES_ROOT` — for the reason given
    there: `VAULT_ROOT/'memory'` is daily notes, so the entry that clause 2 of #3689
    built the vacuous-tree rule to protect was vacuous in a way no file count would show,
    since `~/obsidian/memory` is full of files `memory_add` never writes.
    """
    p = resolved if resolved is not None else paths
    return {
        "memory": memories_root(),
        "facts": Path(p.VAULT_FACTS_ROOT),
        "backlog": Path(p.VAULT_ROOT) / "backlog",
        "data": Path(p.DATA_ROOT),
    }


def snapshot_tree(tree: str, root: Path, *, exclude: tuple[Path, ...] = ()) -> dict[str, Any]:
    """One tree's state: `{abs_path: (size, mtime_ns)}` plus the walk's own account.

    Returns a row rather than raising for an empty or unreadable tree, because the caller
    must be able to put the reason in the RECORD; a raised exception here would be a
    stack trace where a verdict belongs. `refused` is set for a missing root and for a
    directory that resolves to zero files — both would otherwise look like a clean diff.
    `unreadable` lists directories the walk could not open: a tree with an unreadable
    corner is not a clean tree, and naming it is the difference between a witness and an
    accident.
    """
    row: dict[str, Any] = {"tree": tree, "root": str(root), "exists": False,
                           "files": 0, "refused": None, "unreadable": [],
                           "excluded": [str(p) for p in exclude], "manifest": {}}
    if not root.is_dir():
        row["refused"] = f"tree {tree}: {root} is not a directory"
        return row
    row["exists"] = True
    skip = {Path(os.path.realpath(p)) for p in exclude}
    manifest = row["manifest"]

    def walk(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                entries = list(entries)
        except OSError as exc:  # unreadable is a finding, never a clean pass
            row["unreadable"].append(f"{directory}: {exc.__class__.__name__}")
            return
        for entry in entries:
            path = Path(entry.path)
            if not entry.is_symlink() and entry.is_dir(follow_symlinks=False):
                if Path(os.path.realpath(path)) in skip:
                    continue
                walk(path)
                continue
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                row["unreadable"].append(f"{path}: {exc.__class__.__name__}")
                continue
            manifest[str(path)] = (stat.st_size, stat.st_mtime_ns)

    walk(root)
    row["files"] = len(manifest)
    if not row["files"]:
        row["refused"] = (f"tree {tree}: {root} resolves to zero files, so a diff of it "
                          "is vacuous — refusing to call that clean")
    return row


def changed_paths(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """Exactly the paths that moved between two snapshots of ONE tree: new, gone, changed.

    Nothing else is reported. A path present in both with the same `(size, mtime_ns)` is
    unchanged and must not appear, because the verdict in `attribute_windows` is a set
    difference and every spurious member of it is a false alarm that trains the reader to
    ignore the artifact.
    """
    b, a = before["manifest"], after["manifest"]
    moved = {p for p, v in a.items() if b.get(p) != v}
    moved |= {p for p in b if p not in a}
    return moved


def attribute_windows(control: dict[str, set[str]], trial: dict[str, set[str]]) -> dict[str, Any]:
    """The attributed diff: which trial-window changes are NOT explained by background.

    `control[tree]` is what changed in a window of the SAME length with no trial running
    (`_measure_window`); `trial[tree]` is what changed during the trial. A change is
    background when the control window saw the same path; `unexplained` is the trial set
    minus that, and `unexplained` being empty is the live half of PASS. See the module
    docstring for why the subset rule replaced `changed == 0`.
    """
    per_tree: dict[str, Any] = {}
    unexplained: set[str] = set()
    for tree, trial_changed in trial.items():
        background = control.get(tree, set())
        gap = sorted(trial_changed - background)
        unexplained |= set(gap)
        per_tree[tree] = {
            "trial_changed": sorted(trial_changed), "control_changed": sorted(background),
            "unexplained": gap,
            "counts": {"trial": len(trial_changed), "control": len(background),
                       "unexplained": len(gap)},
        }
    return {"per_tree": per_tree, "unexplained": sorted(unexplained),
            "unexplained_count": len(unexplained)}


def server_isolation_block(body: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """The `tool_sandbox` block of a `/state` body, and which required keys it lacks.

    The same rule `bench_runner_sdk.require_copied_state` applies: a block that is absent or
    not an object becomes the empty dict, and what decides the question is MEMBERSHIP, never
    a truthiness test — an aggregator that predates #3403 answers with no isolation block at
    all, and a `.get(...) or False` would read that silence as "not production".
    """
    block = (body or {}).get("tool_sandbox")
    block = block if isinstance(block, dict) else {}
    return block, [key for key in SERVER_ISOLATION_KEYS if key not in block]


def isolation_flags(state: dict[str, Any] | None) -> dict[str, bool]:
    """The three server-reported flags, coerced to bool, missing read as True/not-copied.

    Read out of the `tool_sandbox` block, which is where `agent_mcp/main.py:909` puts them —
    see the module docstring for why the top level is the wrong place to look. A missing key
    is the unsafe direction on purpose: `is_production_data` defaulting to `True` and
    `state_is_copied` defaulting to `False` means an aggregator that answers with a body this
    code cannot parse is treated as PRODUCTION, which refuses. The mirror default — absent
    means "fine, carry on" — is the one that eventually writes into Alan's actual memory.
    """
    block, _missing = server_isolation_block(state)
    return {"is_production_data": bool(block.get("is_production_data", True)),
            "is_production_vault": bool(block.get("is_production_vault", True)),
            "state_is_copied": bool(block.get("state_is_copied", False))}


def prewrite_guard(state: dict[str, Any] | None,
                   declared: tuple[str, ...] = tuple(r["tool"] for r in PROOF_TOOLS)
                   ) -> dict[str, Any]:
    """Ask the server BEFORE any write; refuse unless it reports copied state.

    The flags are the AGGREGATOR's, read from the server's own state endpoint
    (`agent_mcp/_tool_sandbox.py:411`), not this script's: the process deciding whether to
    write is not the process whose roots are at risk of being production. All three names
    are in the refusal because a reader of the record has to be able to tell WHICH gate
    shut — production data, production vault and unproven-copy are three different faults,
    and the old single flag conflated them (#3403).

    Two further faults are checked here rather than left to the trial, both copied from
    `require_copied_state`: a body with no isolation block or a missing key (the server
    cannot answer the question), and an `allowed_writes` that does not cover the declared
    tools (the scope never reached the process that will execute them, so all three calls
    would come back refused and the row would read FAIL for a reason the record would have
    to be inferred from). `declared` is a parameter so a caller can ask about a subset.
    """
    block, missing = server_isolation_block(state)
    flags = isolation_flags(state)
    faults = []
    if "tool_sandbox" not in (state or {}):
        faults.append("the server reported no `tool_sandbox` block, so it cannot say "
                      "whether its own state is a copy (an aggregator older than #3403)")
    elif missing:
        faults.append(f"the server's `tool_sandbox` block does not report {missing}")
    if flags["is_production_data"]:
        faults.append("is_production_data is True")
    if flags["is_production_vault"]:
        faults.append("is_production_vault is True")
    if not flags["state_is_copied"]:
        faults.append("state_is_copied is False")
    gap = sorted(set(declared) - set(block.get("allowed_writes") or []))
    if gap:
        faults.append(f"the aggregator honours none of {gap}, though the trial declares "
                      "them: the write scope never reached the process that will execute "
                      "the tools, so every call would come back refused")
    return {"flags": flags, "allowed": not faults, "write_scope_gap": gap,
            "missing_keys": missing,
            "refusal": (None if not faults else
                        "refusing to attempt a durable write: " + "; ".join(faults)
                        + " (is_production_data=%(is_production_data)s "
                        "is_production_vault=%(is_production_vault)s "
                        "state_is_copied=%(state_is_copied)s)" % flags)}


def auth_headers(url: str) -> dict[str, str]:
    """The credential this request must carry, from the ONE reader the server also uses.

    `agent_mcp/aggregator_auth.py:95` keeps only `/health` credential-free, so BOTH of
    this script's requests — the state read and the tool call — arrive without it only to be
    answered `401 AGGREGATOR_UNAUTHENTICATED`. That is why the header is not optional here:
    a proof that GETs `/state` bare and reads a body back is validating a request the real
    server rejects, which is the same mistake as reading its flags from the wrong nesting.

    `headers_for_url` (not a local `readline()`) so the header name, the
    `LLOYD_MCP_TOKEN`/`LLOYD_MCP_TOKEN_FILE` precedence and the loopback-only rule are
    defined once and shared with the process that checks the value. It answers `{}` when
    there is no credential at all rather than raising — deliberately, because the refusal
    belongs on the server — and the 401 that comes back is what this script then reports.
    Importing `agent_mcp` costs ~25 ms and pulls no MCP dependency for this module
    (stdlib-only imports), so the earlier "too heavy to import" reason for a private copy
    did not survive the measurement.
    """
    try:
        from agent_mcp.aggregator_auth import headers_for_url
    except Exception as exc:  # noqa: BLE001 — closed: no credential, no unauthenticated probe
        raise ProofRefused(f"cannot read the loopback credential ({exc.__class__.__name__}: "
                           f"{exc}); refusing an unauthenticated probe of {url} rather than "
                           "reading its 401 as a state answer") from exc
    return headers_for_url(url)


#: The scopes a booted aggregator is armed with for this trial (#3863 clause 3).
#:
#: `vault`, not `memory`, for the memory leg — the same reason
#: `tests/test_autoresearch_bench_spawner.py:24-27` gives: `build_fixture` copies a `memory`
#: scope from `<vault>/memory` (`bench_fixture.py:360-364`), which is where the DATED DAILY
#: NOTES live and what `memory_add` never touches, so a card scoped `memory` alone boots a
#: server whose copy of the memory store is a copy of the wrong directory. `vault` is the
#: whole slice, so `<fixture>/obsidian/lloyd/` — the directory `MEMORIES_ROOT` names — is
#: present whatever the memory root is later renamed to. `kg` is the KG database the
#: `fact_add` row goes into, `backlog` the card.
#:
#: One store this cannot arm, and it is recorded rather than hidden: `fact_add`'s MARKDOWN
#: lands under `VAULT_FACTS_ROOT`, which is `DATA_ROOT/_pipeline/vault-derived/facts`
#: (`app/paths.py:307-311`) — inside the fixture's copy, so still isolated, but copied by no
#: scope word and outside the fixture vault's git repo, so it can never appear in
#: `copy_side.landed_by_git_diff`. See `copy_side_witness`'s note on the same asymmetry.
SPAWN_SCOPES: tuple[str, ...] = ("vault", "kg", "backlog")


def build_trial_fixture(trial_id: str, *, resolved: Any = None) -> Path:
    """Build the copy this trial runs on and return its directory (#3915).

    The one route from an empty fixtures area to a running trial. Until now the CLI's own
    refusal told a caller to invoke the builder as a module, and `bench_fixture.py` has no
    `__main__` block and no argparse — that command imported the file, printed nothing and
    exited 0 having copied nothing — so a box with no fixture had no way to get one and
    every path through this instrument dead-ended at that sentence. The dead invocation is
    not spelled out here: #3915 leaves behind a line-wise grep of this module for a
    module-run command naming the builder, and a docstring that recites it would keep that
    check red for the rest of its life.

    Three things are deliberately NOT done here:

    * **No destination rule of its own.** The destination is `build_fixture`'s:
      `fixture_parent(data_root)/<trial_id>` = `<data_root>/eval/fixtures/<trial_id>`
      (`bench_fixture.py:51,97-98,264`), and the refusal that guards it
      (`:282-290`, plus the containment rails at `:288-299`) propagates as
      `FixtureRefused` for `main` to exit 3 on. A second spelling of "is this inside the
      fixtures area" is a second rail that can disagree with the first, which is exactly
      what `bench_fixture._delete_refusal`'s docstring says the ONE guard is for.
    * **No second scope list.** `writes` is `SPAWN_SCOPES`, the same tuple
      `boot_fixture_aggregator` arms the child with. A fixture built with a wider or
      narrower scope than the server it is booted on is a trial whose copy holds
      directories its own aggregator will not let it write, or none of the ones it will —
      and `copy_side_witness` would then read that mismatch as the trial's failure. The
      `vault` leg (not `memory`, and not the item's `memory/facts/backlog` vocabulary) is
      the reason given on `SPAWN_SCOPES` above: `memory` copies the dated-note directory the
      writer never touches, and `facts` arms nothing at all.
    * **No root literals.** `paths.VAULT_ROOT` and `paths.DATA_ROOT` are read per call off
      the module, the way `durable_write_trees` reads them, so the copy is of the trees THIS
      process resolves — and so a test can stand a small vault in for the real one without
      patching a string.

    `build_fixture`'s rule (2) refuses a destination inside either production tree, so on
    this box the build needs `LLOYD_DATA` pointed at a scratch root outside
    `~/lloyd-data`; the same rule is what `bench_spawner.spawn` refuses to boot on, so no
    route exists that puts a fixture inside the house and uses it.
    """
    p = resolved if resolved is not None else paths
    fixture = build_fixture(trial_id=trial_id, writes=list(SPAWN_SCOPES),
                            vault=p.VAULT_ROOT, data_root=p.DATA_ROOT)
    return Path(fixture.root)


def boot_fixture_aggregator(fixture_root: Path, *,
                            scopes: tuple[str, ...] | None = None) -> tuple[Any,
                                                                            dict[str, str],
                                                                            dict[str, str]]:
    """Boot a fixture-rooted aggregator and return `(handle, urls, credential)`.

    The caller is responsible for `handle.stop()`; `main` does it in its `finally`, because a
    scope-armed server outliving the trial is a server some later process finds listening.

    Two spawner calls, in the order the spawner requires them (clause 3 of #3863):
    `fixture_from_dir` re-derives the `BenchFixture` from the directory's own layout, which is
    the step that refuses a hand-made directory, and `spawn` then runs the live-root guards
    and blocks until the child's `GET /state` reports `state_is_copied` true. This script
    never copies a slice of the house itself: building the fixture is `bench_fixture`'s job
    (#3688 is `done`), and a second builder here would be a second answer to "what got
    copied".

    `urls` carries BOTH endpoints built off `handle.url`, and the `/mcp` suffix is not
    optional: `bench_spawner.py:362` constructs `url=f"http://{host}:{chosen}"`, a bare
    origin, so a POST to it is answered by Starlette's 404 — which `mcp_tool_call` then wraps
    as `ok: False` and the trial reads as three failed writes rather than as a missing route.

    `credential` is the CHILD's (`bench_spawner.auth_headers(handle.env)`), and it is returned
    because the token the child demands is the one `spawn` injected into its env
    (`bench_spawner.py:349`), which is not the token this process would read off its own
    `LLOYD_MCP_TOKEN_FILE`. `fetch_state_json`'s `headers` and `mcp_tool_call`'s take it.
    """
    try:
        from scripts.autoresearch import bench_spawner as spawner
    except Exception as exc:  # noqa: BLE001 — closed: no spawner, no sandboxed server
        raise ProofRefused(
            f"cannot import `scripts.autoresearch.bench_spawner` ({exc.__class__.__name__}: "
            f"{exc}); refusing to boot an aggregator by hand, because the guards that refuse "
            "a fixture inside a live root live in that module") from exc

    chosen = tuple(scopes) if scopes else SPAWN_SCOPES
    root = Path(fixture_root)
    try:
        fixture = spawner.fixture_from_dir(root, chosen)
    except Exception as exc:  # FixtureError/RuntimeError: the marker test refuses for us
        raise ProofRefused(f"{root} is not a bootable fixture: {exc}") from exc
    try:
        handle = spawner.spawn(fixture_root=fixture, scopes=chosen)
    except Exception as exc:  # noqa: BLE001 — a boot that failed is a refusal, not a traceback
        raise ProofRefused(
            f"the spawner refused to boot an aggregator on {root}: "
            f"{exc.__class__.__name__}: {exc}") from exc
    try:
        credential = spawner.auth_headers(handle.env)
    except Exception as exc:  # noqa: BLE001 — no credential, so every request would 401
        handle.stop()
        raise ProofRefused(
            f"the booted aggregator's environment carries no usable credential "
            f"({exc.__class__.__name__}: {exc}); asking it for its isolation state without "
            "one is a 401, which this script can only read as an unknown state") from exc
    return handle, {"mcp": handle.url + "/mcp", "state": handle.url + "/state"}, credential


def fetch_state_json(url: str, *, timeout: float = 10.0,
                     headers: dict[str, str] | None = None) -> dict[str, Any]:
    """GET the aggregator's state endpoint. Raises `ProofRefused` if it will not answer.

    A server that cannot be asked is a refusal and not a FAIL: the claim under test is
    about where writes land, and no evidence either way was gathered. A 401 lands in the
    same branch, which is the correct reading — an unreadable state is an unknown state.

    `headers` overrides the credential set wholesale, which exists for one reason: the
    spawner's `auth_headers` (`bench_spawner.py:151-164`) reads `LLOYD_MCP_TOKEN_FILE` out of
    the CHILD's env, and if that names a path, the token the child will demand is in that
    file and not in this process's. Asking the child for its isolation state with the
    wrong credential is a 401, which this function — correctly — turns into a refusal; so a
    boot passes the child's own pair in. On the default path (`headers=None`) the reader is
    the same one the server uses, unchanged.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   **(auth_headers(url) if headers is None
                                                      else headers)})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ProofRefused(f"state endpoint {url} did not answer with JSON: "
                           f"{exc.__class__.__name__}: {exc}") from exc


def mcp_tool_call(url: str, tool: str, arguments: dict[str, Any], *,
                  headers: dict[str, str] | None = None,
                  session_id: str = "",
                  timeout: float = 120.0) -> dict[str, Any]:
    """POST one JSON-RPC `tools/call` and return the decoded body.

    Deliberately thin: no retry, no session handshake, no fallback, because a trial that
    half-worked must read as a failed call in the record rather than a success with a
    caveat. The credential comes from `auth_headers(url)` — the real file read, on the
    default path every trial takes; `headers` overrides the whole set so a test can post a
    wrong one or none.

    `session_id` rides at `params._meta['lloyd/session_id']` (see `rpc_body`). It is not
    decoration: this server is stateless, so a call that arrives with no session id has no
    session at all, and all three trial tools are state-changing, so the handler refuses
    them all before dispatch (`agent_mcp/main.py:595-602`).

    A refusal is reported as `ok: False` — clause 5 of #3863, and the reason this function
    cannot simply ask `"error" not in payload`. `_refused_call` answers a guard refusal as a
    normal JSON-RPC **result** whose payload therefore has no `error` member, so the old test
    was true for a refused write; the four live trees stay pristine precisely BECAUSE nothing
    was written, and the trial would have printed PASS on the strength of the cleanest
    evidence it never had. `denied_call` is what tells the two apart, and whatever it returns
    rides out in `denial` so the record names the guard rather than the reader having to
    infer it from `result_text`.

    The wire IS exercised, in both directions: `tests/test_bench_durable_write_proof.py`
    POSTs through this function to a real subprocess server that resolves its stores through
    `app.paths`, reports the real `tool_sandbox` block, answers a wrong or absent credential
    with the same 401 the aggregator's own guard answers (`aggregator_auth.py`), and — since
    #3863 — refuses a sessionless state-changing call with the REAL `_refused_call` body built
    by the shipped handler code, then re-sent in the aggregator's own wire form. That stands
    in for `agent_mcp/main.py`'s dispatch shape; the real on-box trial (#3689's owed list) is
    what pins its behaviour.
    """
    body = rpc_body(tool, arguments, session_id=session_id)
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 **(auth_headers(url) if headers is None else headers)},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"tool": tool, "ok": False,
                "detail": f"{exc.__class__.__name__}: {exc}"}
    text = "".join(str(part.get("text", "")) for part in
                   (payload.get("result") or {}).get("content", []) if isinstance(part, dict))
    denial = denied_call(payload)
    out: dict[str, Any] = {"tool": tool, "ok": ("error" not in payload) and denial is None,
                           "text": text, "response": payload}
    if denial is not None:
        out["denial"] = denial
    return out


def trial_arguments(store: str, trial_id: str) -> dict[str, Any]:
    """The smallest write that is durable and identifiable, per store.

    One row per store, with the trial id inside it, so a leak into the live tree is
    attributable to this trial by grep rather than by inference. Category and entity are
    the ones the fact store already accepts for bench material.
    """
    marker = f"bench-durable-write-proof {trial_id}"
    if store == "memory":
        return {"entity": "bench-durable-write-proof", "category": "reference",
                "fact": marker, "provenance": "STATED"}
    if store == "facts":
        return {"entity": "bench-durable-write-proof", "category": "reference",
                "fact": marker, "provenance": "STATED"}
    return {"name": f"bench durable write proof {trial_id}",
            "description": marker, "board": "lloyd"}


def copy_side_witness(fixture_root: Path | None, *, intended: dict[str, str],
                      diff_paths: list[str] | None = None) -> dict[str, Any]:
    """Did the write land in the COPY? A live-side diff alone proves nothing.

    `git diff --name-only` is read from the fixture when a root is given and not supplied
    (the runner's own `BenchFixture.changed_paths()` is the same `git -C <vault> diff
    --name-only`, run here so this script also works against a fixture left on disk).
    `intended` maps tool → store, and a store counts as landed when a diff path lies under
    that store's directory inside the copy.

    The empty case is the one that matters. A fixture whose diff lists nothing means the
    trial wrote NOTHING ANYWHERE: all three calls could then report `ok` while the live
    trees stayed pristine, and a diff-of-nothing would look identical to the isolation
    working. So no-diff-is-the-intended-target forces FAIL, which is clause 5.

    One asymmetry is recorded rather than papered over: the copy's git repo is rooted at
    the fixture's VAULT (`bench_fixture.py:410`), and `VAULT_FACTS_ROOT` lives under the
    fixture's DATA root, so a fact_add's markdown is outside `git diff`'s reach. The row
    therefore says which store matched, and the facts store's witness is its file count,
    which `run_proof` supplies.
    """
    if diff_paths is None:
        diff_paths = []
        if fixture_root is not None:
            # Which directory holds the repo is not settled by the caller's flag:
            # `BenchFixture.init_git` runs `git init` inside the copied vault slice, which
            # `build_fixture` lays down as `<fixture>/obsidian` (bench_fixture.py:307, :418),
            # while `--fixture-root` is documented as the fixture DIRECTORY. So try the given
            # root, then any one-level child that holds a repo. A root with no repo anywhere
            # stays UNVERIFIED below rather than becoming an empty diff — falling through to
            # `[]` is what made a written-to fixture look unwritten.
            root_dir = Path(fixture_root)
            candidates = [root_dir] + sorted(
                child for child in root_dir.iterdir()
                if child.is_dir() and (child / ".git").exists()) if root_dir.is_dir() else []
            for candidate in candidates:
                if not (candidate / ".git").exists():
                    continue
                # `git add -N` (intent-to-add) first, for the reason bench_fixture.py:439-450
                # gives for the same call: a file this trial CREATED is untracked, and plain
                # `diff --name-only` says nothing about it — which would read as "the copy
                # received no write" for the most ordinary kind of durable write.
                subprocess.run(["git", "-C", str(candidate), "add", "-N", "-A"],
                               capture_output=True, text=True, timeout=120)
                proc = subprocess.run(["git", "-C", str(candidate), "diff", "--name-only"],
                                      capture_output=True, text=True, timeout=60)
                diff_paths = [line for line in proc.stdout.splitlines() if line.strip()]
                break
    landed = sorted({tool for tool, store in intended.items()
                     for path in diff_paths if _under_store(path, store)})
    return {"fixture_root": str(fixture_root) if fixture_root else None,
            "diff_paths": sorted(diff_paths),
            "intended_stores": dict(intended),
            "landed_by_git_diff": landed,
            "not_landed_by_git_diff": sorted(set(intended) - set(landed)),
            "landed": bool(landed)}


def _under_store(rel_path: str, store: str) -> bool:
    """Is a copy-side path inside the store that a named tool was asked to write?

    Matched on `store_dirname(store)`, not on the label: `memory_add` lands in the copy's
    `lloyd/` directory (`agent_mcp/session.py:58` via `memories_root`), so a matcher keyed to
    the literal word `memory` could only ever name a path the tool cannot produce — and
    `landed_by_git_diff`, the field the owed run is read for, would come back missing
    `memory` on a trial that had just written it.
    """
    name = store_dirname(store)
    return rel_path.startswith(f"{name}/") or f"/{name}/" in f"/{rel_path}"


def verdict(*, guard: dict[str, Any], windows: dict[str, Any],
            witness: dict[str, Any], calls: list[dict[str, Any]],
            snapshots: dict[str, Any]) -> dict[str, Any]:
    """PASS only with every leg green. Any one of them is enough to make the row FAIL.

    Four legs, in the order a reader should have to check them: the server said copied
    state BEFORE the first write; the trees were non-empty so the diff was not vacuous; no
    live path changed during the trial that background does not also explain; the copy
    actually received the writes. The first two are refusals (nothing was attempted); the
    last two are FAILs (something was attempted and the claim did not hold).
    """
    reasons: list[str] = []
    refused: list[str] = []
    if not guard["allowed"]:
        refused.append(guard["refusal"])
    empty = sorted(name for name, row in snapshots.items() if row.get("refused"))
    if empty:
        refused.append(f"tree(s) refused as vacuous: {empty} — "
                       + "; ".join(str(snapshots[n].get("refused")) for n in empty))
    unreadable = {n: r["unreadable"] for n, r in snapshots.items() if r.get("unreadable")}
    if unreadable:
        reasons.append(f"unreadable directories inside the manifested trees: {unreadable}")
    if windows["unexplained_count"]:
        reasons.append(f"{windows['unexplained_count']} live path(s) changed during the "
                       f"trial and not during the equal-length control window: "
                       f"{windows['unexplained'][:10]}")
    failed = sorted(str(c.get("tool")) for c in calls if not c.get("ok"))
    if failed:
        reasons.append(f"tool call(s) did not return success: {failed}")
    # Named separately from `failed`, because the two readings of the same trial are not
    # interchangeable: "the live trees are pristine and nothing landed" sounds to a later
    # reader like a boring no-op, while "the server refused every write before dispatch" is
    # the wiring fault it actually is. This is the sentence that stops an all-refused trial
    # being filed as inconclusive (#3863 clause 5).
    denied = sorted(str(c.get("tool")) + ": " + str(c.get("denial", "")).strip()
                    for c in calls if str(c.get("denial", "")).strip())
    if denied:
        reasons.append(f"{len(denied)} tool call(s) were refused by a guard before "
                       f"dispatch, so nothing was written anywhere: {denied}")
    if not witness["landed"]:
        reasons.append("copy-side witness empty or naming none of the intended stores: "
                       f"git diff listed {len(witness['diff_paths'])} path(s), none under "
                       f"{sorted(set(witness['intended_stores'].values()))} — a trial that "
                       "wrote nothing anywhere would also leave the live trees clean")
    if refused:
        state, why = "REFUSED", "; ".join(refused)
    elif reasons:
        state, why = "FAIL", "; ".join(reasons)
    else:
        state, why = "PASS", ("no live path changed during the trial that the "
                              "equal-length control window does not also explain, and the "
                              "copy received the writes")
    return {"verdict": state, "reason": why,
            "checks": {"guard_allowed": bool(guard["allowed"]),
                       "trees_non_empty": not empty,
                       "unexplained_count": windows["unexplained_count"],
                       "all_calls_ok": not failed,
                       "copy_witness_landed": bool(witness["landed"])}}


def _measure_window(trees: dict[str, Path], *, sleep: Callable[[float], None],
                    clock: Callable[[], float], window_seconds: float,
                    exclude: tuple[Path, ...] = ()) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Snapshot a set of trees, wait `window_seconds`, snapshot again, diff the two.

    One function for the control window and the trial window so the two cannot drift apart
    in what they count — two spellings of one measurement is how a control starts proving
    something different from the trial it is supposed to excuse.
    """
    before = {name: snapshot_tree(name, root, exclude=exclude) for name, root in trees.items()}
    start = clock()
    sleep(window_seconds)
    after = {name: snapshot_tree(name, root, exclude=exclude) for name, root in trees.items()}
    changed = {name: changed_paths(before[name], after[name]) for name in trees}
    elapsed = clock() - start
    for name, row in after.items():
        row["window_seconds"] = elapsed
    return after, changed


class CallTool(Protocol):
    """The write seam: a tool name, its arguments, and the session id that must ride with them.

    The id is in the TYPE because it is in the call: `run_proof` invokes this as
    `call_tool(tool, arguments, session_id=…)`, and the annotation this replaces —
    `Callable[[str, dict], dict]` — declared a seam with no id in it while the line calling it
    passed one. pyright caught the contradiction as a new error in the gated run
    (`bench_durable_write_proof.py:942: No parameter named "session_id"`), and it is the
    contradiction that matters, not the diagnostic: an annotation that omits the carrier is a
    contract that lets a new caller pass `(tool, arguments)` alone and silently drop the id,
    which is the failure #3863 clause 1 exists to prevent.
    """

    def __call__(self, tool: str, arguments: dict[str, Any], /, *,
                 session_id: str = "") -> dict[str, Any]: ...


def run_proof(*, state: dict[str, Any], trees: dict[str, Path],
              call_tool: CallTool,
              fixture_root: Path | None = None, diff_paths: list[str] | None = None,
              window_seconds: float = DEFAULT_WINDOW_SECONDS,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic,
              trial_id: str = "proof",
              session_id: str = "") -> dict[str, Any]:
    """One trial: guard, control window, live snapshots, the three calls, the verdict.

    `call_tool`, `sleep` and `clock` are injected because the two things this routine must
    prove are (a) that it asks the server BEFORE the first write and (b) that a real
    background change does not condemn it — neither of which is observable through a real
    aggregator and a wall clock. The order below is the contract: guard → live baseline
    check → control window → calls padded to the same length → live after → verdict.

    `call_tool` is called as `call_tool(tool, arguments, session_id=…)`: the id is part of the
    call seam rather than something the injected closure happens to have captured, so a stub
    can assert what went out and a caller cannot forget to thread it. Every caller in this
    file and its test passes the keyword.

    Both windows are `window_seconds` long BY CONSTRUCTION, not approximately: after the
    calls the trial waits out the remainder of the window. A shorter trial window than the
    control would be a control that saw more background than the trial could have, which
    forgives a real durable write — the one error direction this instrument must not have.
    Three writes over the box's own 42-changes-per-120-seconds rate is a boundary the
    padding is there to close rather than to reason about after the fact.

    `session_id` is the caller's preferred id, and it is not taken as-is: it goes through
    `trial_session_id`, which is the sandbox's own rule (`_tool_sandbox.SANDBOXED_ID_PREFIXES`),
    so a bare `abc` reaches the wire as `bench_abc` and a `bench_x`/`pt-eval-x` arrives
    unchanged. There is deliberately no input to this function that puts an un-prefixed id on
    the wire, because an un-prefixed id is a session the sandbox does not class as a trial and
    the refusal for that is indistinguishable, at the old `ok` line, from a write that worked.
    The id that was actually sent is recorded, not the one that was asked for: the owed run
    has to be able to prove from the artefact which session the handler charged the writes to.
    """
    guard = prewrite_guard(state)
    sid = trial_session_id(session_id or trial_id)
    intended = {row["tool"]: row["store"] for row in PROOF_TOOLS}
    record: dict[str, Any] = {
        "kind": "durable_write_proof", "trial_id": trial_id,
        # The id on the wire, in `params._meta['lloyd/session_id']` of every call below. A
        # record that named a different id from the one it sent would leave the artefact
        # unable to say which session the server charged the writes to, which is the fact a
        # reader of a leak needs.
        "session_id": sid,
        "tools": [row["tool"] for row in PROOF_TOOLS],
        "isolation_flags": guard["flags"], "window_seconds": window_seconds,
        # Kept even on an allowed guard: the record is the artifact, and "the aggregator
        # honoured all three declared tools" is a fact a later reader should not have to
        # re-derive from a server that has since restarted.
        "write_scope_gap": guard["write_scope_gap"],
        "missing_isolation_keys": guard["missing_keys"],
        "trees": {n: {"root": str(r)} for n, r in trees.items()},
        "tool_calls": [], "verdict": "REFUSED",
    }
    if not guard["allowed"]:
        # No write is attempted, and the record says which of the three flags refused.
        record["refusal"] = guard["refusal"]
        record["reason"] = guard["refusal"]
        return record

    exclude = tuple([Path(fixture_root)] if fixture_root is not None else [])
    baseline = {name: snapshot_tree(name, root, exclude=exclude)
                for name, root in trees.items()}
    empty = sorted(n for n, row in baseline.items() if row.get("refused"))
    if empty:
        # Asked before the window runs: a vacuous tree cannot be made non-vacuous by
        # waiting, and 120 seconds is not a price to pay for finding that out.
        record["refusal"] = ("; ".join(str(baseline[n]["refused"]) for n in empty))
        record["reason"] = record["refusal"]
        record["trees"] = {n: {k: v for k, v in baseline[n].items() if k != "manifest"}
                           for n in baseline}
        return record

    control_after, control_changed = _measure_window(
        trees, sleep=sleep, clock=clock, window_seconds=window_seconds, exclude=exclude)

    live_before = baseline
    start = clock()
    for row in PROOF_TOOLS:
        call = call_tool(row["tool"], trial_arguments(row["store"], trial_id),
                         session_id=sid)
        # `result_text` is kept because the artifact must show WHERE the server says it
        # wrote: a row that records only `ok: true` cannot be checked against the live trees
        # by a later reader, and the record_path is the one field that can be.
        record["tool_calls"].append({"tool": row["tool"], "store": row["store"],
                                     "ok": bool(call.get("ok")),
                                     # Per call, because the artefact has to survive one call
                                     # being refused and the next not: a single record-level
                                     # id would attribute a write to the wrong session in the
                                     # one record whose job is to say who did what.
                                     "session_id": sid,
                                     "result_text": str(call.get("text", ""))[:600],
                                     "detail": str(call.get("detail", ""))[:400],
                                     # The guard's own words. `verdict` reads them to name the
                                     # refusals in the FAIL reason, and an artefact carrying
                                     # only `ok: false` could not tell a refusal — a wiring
                                     # fault — from a transport failure.
                                     "denial": str(call.get("denial", ""))[:400]})
    calls_seconds = clock() - start
    sleep(max(0.0, window_seconds - calls_seconds))   # pad: windows are equal length
    trial_seconds = clock() - start

    live_after = {name: snapshot_tree(name, root, exclude=exclude)
                  for name, root in trees.items()}
    trial_changed = {name: changed_paths(live_before[name], live_after[name])
                     for name in trees}
    windows = attribute_windows(control_changed, trial_changed)
    witness = copy_side_witness(fixture_root, intended=intended, diff_paths=diff_paths)
    after_rows = {n: {k: v for k, v in row.items() if k != "manifest"}
                  for n, row in live_after.items()}
    for name, row in after_rows.items():
        row["baseline_files"] = live_before[name]["files"]
        row["trial_seconds"] = trial_seconds
    outcome = verdict(guard=guard, windows=windows, witness=witness,
                      calls=record["tool_calls"], snapshots=after_rows)
    record.update(
        verdict=outcome["verdict"], reason=outcome["reason"], checks=outcome["checks"],
        control_files={n: r["files"] for n, r in control_after.items()},
        windows=windows, copy_side=witness, live_trees=after_rows,
    )
    return record


def main(argv: list[str] | None = None) -> int:
    """CLI. Exit `0` PASS · `1` FAIL · `3` refused · `2` argparse usage.

    `--plan` resolves the four trees, reports their file counts and refuses if any is
    vacuous, without contacting a server or attempting a write. It prints no verdict, because
    a proof of nothing has no verdict to print.

    Three ways to get a server. Pass `--aggregator-url` plus a state url to try one that is
    already running — nothing is booted on that path, before or after the guard. Pass
    NEITHER and the script boots its own from `--fixture-root` through
    `boot_fixture_aggregator` and stops it on the way out (#3863 clause 3). Pass
    `--build-fixture <trial_id>` and it first BUILDS that fixture with
    `bench_fixture.build_fixture`, then boots on the directory that build created — the step
    the empty-`--fixture-root` refusal used only to describe, naming a module whose
    invocation built nothing (#3915). What has not changed is that no path reaches a write
    before `prewrite_guard` has read the SERVER's flags: a production-reporting aggregator
    exits 3 with no trial, and a caller who supplied its own urls exits 3 without the
    spawner being consulted at all.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aggregator-url", default="",
                        help="an already-running fixture-routed aggregator's /mcp "
                             "(JSON-RPC tools/call); omit WITH --state-url to boot one")
    parser.add_argument("--state-url", default="",
                        help="its state endpoint; defaults to $LLOYD_BENCH_STATE_URL")
    parser.add_argument("--fixture-root", default="",
                        help="the BUILT fixture directory (the one holding `lloyd-data/` and "
                             "`obsidian/`): the copy-side git diff reads it, and with no "
                             "aggregator urls it is also what the aggregator is booted on")
    parser.add_argument("--build-fixture", default=None, metavar="TRIAL_ID",
                        help="build the fixture for TRIAL_ID first, into "
                             "<data-root>/eval/fixtures/<TRIAL_ID> with the scopes this "
                             "script boots on, and run the trial against it. Pass it alone: "
                             "--plan, --fixture-root and the aggregator urls each name a "
                             "different copy (or none), so combining them is refused rather "
                             "than guessed. Needs LLOYD_DATA outside the production data "
                             "root, which `build_fixture` refuses to build inside")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--out", default="", help="write the record here as JSON")
    parser.add_argument("--plan", action="store_true",
                        help="resolve and size the live trees, ask nothing, write nothing")
    parser.add_argument("--trial-id", default=None,
                        help="the id written into every planted marker and into the record; "
                             "defaults to --out's stem, or 'proof'")
    args = parser.parse_args(argv)

    # The url pair is read before anything branches, because `--build-fixture` has to refuse
    # a caller who ALSO named a server to talk to: a freshly built copy is not the one that
    # aggregator is routed on, and grading its git diff would be grading a directory no
    # process was ever pointed at.
    state_url = (args.state_url or os.environ.get(STATE_URL_ENV, "")).strip()
    given_url = bool(args.aggregator_url.strip())
    given_state = bool(state_url)
    # `None` is the absent flag and `""` is a caller who typed one with no id, so the two are
    # told apart by identity and not by truthiness — argparse's own default is the empty
    # string's twin, and a check that read them alike would build nothing and say nothing.
    build_id = (args.build_fixture or "").strip()
    if args.build_fixture is not None:
        # An empty trial id would resolve the destination to the fixtures AREA itself, where
        # `_delete_refusal` (`bench_fixture.py:156`) will not delete — a copy that outlives its
        # trial by construction. It is reported in the same breath as the conflicts because
        # both are "this flag cannot mean what it was given with".
        conflicts = ([] if build_id else ["an empty TRIAL_ID"]) + [
            named for named, taken in (
                ("--plan", args.plan),
                ("--fixture-root", bool(args.fixture_root.strip())),
                ("--aggregator-url", given_url),
                (f"--state-url (or ${STATE_URL_ENV})", given_state)) if taken]
        if conflicts:
            print("refused: --build-fixture builds the copy this trial runs on, so it is "
                  f"passed alone; it was given with: {', '.join(conflicts)}. Pass "
                  "--build-fixture <trial_id> with nothing else to build and boot in one "
                  "run, or --fixture-root <dir> for a fixture that is already built.",
                  file=sys.stderr)
            return 3

    trees = durable_write_trees()
    if args.plan:
        rows = {n: snapshot_tree(n, root) for n, root in trees.items()}
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "manifest"}
                          for k, v in rows.items()}, indent=2))
        refused = sorted(n for n, row in rows.items() if row.get("refused"))
        if refused:
            print(f"refused: vacuous tree(s) {refused}", file=sys.stderr)
            return 3
        return 0

    # Three shapes, and they are all the shapes there are. BOTH urls given: use them, boot
    # nothing — this is the path a caller who already has a fixture-routed aggregator takes,
    # and it is the path that must still exit 3 without reaching for a spawner when the guard
    # reports production. NEITHER given: boot one here (#3863 clause 3), which needs a built
    # fixture to boot from. Exactly ONE given: a half-named pair is a typo, not an
    # instruction to boot, and guessing which half the caller meant is how a trial ends up
    # POSTing to a stranger's port.
    if given_url != given_state:
        print("refused: --aggregator-url and a state url (flag or $" + STATE_URL_ENV
              + ") go together; got only one of them. Pass both to use an aggregator that is "
                "already running, or neither to have this script boot one from "
                "--fixture-root.", file=sys.stderr)
        return 3

    handle: Any = None
    credential: dict[str, str] = {}
    # The copy this run grades: the flag's directory, and the build's directory once a build
    # has happened. One variable on purpose — `run_proof`'s copy-side witness and the
    # aggregator's roots must be the same directory, and the flag that could have named a
    # second one was refused above.
    fixture_root = args.fixture_root.strip()
    try:
        if build_id:
            # Build first, then boot on exactly what was built. The directory comes back from
            # `build_trial_fixture` rather than being re-derived from the trial id, so the
            # path the spawner is handed is the path the builder created — and if
            # `build_fixture` ever moves the destination, this line moves with it instead of
            # quietly booting on a directory that is no longer where copies go.
            fixture_root = str(build_trial_fixture(build_id))
            print(f"fixture built at {fixture_root}", file=sys.stderr)
        if not given_url:
            if not fixture_root:
                print("refused: no aggregator urls given, so this script would boot one — "
                      "and --fixture-root is empty, so there is no built fixture to boot it "
                      "on. Pass --build-fixture <trial_id> to build one with "
                      "`bench_fixture.build_fixture` and boot on it in this same run, or "
                      "--fixture-root <dir> for a fixture that is already built.",
                      file=sys.stderr)
                return 3
            handle, urls, credential = boot_fixture_aggregator(Path(fixture_root))
            mcp_url, state_url = urls["mcp"], urls["state"]
        else:
            mcp_url = args.aggregator_url
        record = run_proof(
            state=fetch_state_json(state_url, headers=credential or None), trees=trees,
            call_tool=lambda name, arguments, session_id="": mcp_tool_call(
                mcp_url, name, arguments, session_id=session_id,
                headers=(credential or None)),
            fixture_root=Path(fixture_root) if fixture_root else None,
            window_seconds=args.window_seconds,
            trial_id=(args.trial_id or build_id
                      or (Path(args.out).stem if args.out else "proof")))
    except (ProofRefused, FixtureRefused) as exc:
        # `FixtureRefused` is `build_fixture`'s own refusal class, caught BY NAME rather than
        # re-derived into a `ProofRefused`: the builder has the only answer to "may this
        # destination be built", and a destination it refuses is exit 3 with nothing copied
        # (`bench_fixture.py:286` raises before any `mkdir`), not a traceback and not a FAIL
        # row. Wrapping it here would put a second copy of that rule in this file, which is
        # the duplication `_delete_refusal`'s docstring exists to warn against.
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    finally:
        # The child is stopped whether the trial passed, failed or refused. A scope-armed
        # aggregator on the account's own loopback outliving the process that booted it is a
        # server nobody owns, and `stop()` is terminate-before-kill precisely so it does not
        # die holding a SQLite lock. `stop()` is already idempotent (`self.stopped` is
        # cached), so this is safe even if the boot itself raised after Popen.
        if handle is not None:
            handle.stop()

    payload = json.dumps(record, indent=2)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    state = record["verdict"]
    return {"PASS": 0, "FAIL": 1}.get(state, 3)


if __name__ == "__main__":
    sys.exit(main())
