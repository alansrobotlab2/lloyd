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
did not hold) · `3` refused (production state, an empty tree, or no aggregator to ask —
no write attempted) · `2` argparse usage.

    scripts/autoresearch/bench_durable_write_proof.py --aggregator-url http://127.0.0.1:8819/mcp \
        --fixture-root /home/alansrobotlab/lloyd-data/eval/fixtures/<trial> --out <record.json>

The real on-box run is owed after this lands (#3689's owed list): it needs an aggregator
booted under the fixture's roots, which is #3688's spawner or a boot this script does not
own. `--plan` prints what a trial WOULD check against the live trees without a server or a
write, which is the mode a reader can run today.
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
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import paths  # noqa: E402

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


def rpc_body(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """One `tools/call` envelope, built fresh so no call can see another's arguments."""
    return {"jsonrpc": "2.0", "id": 1, "method": RPC_METHOD,
            "params": {"name": tool, "arguments": arguments}}


class ProofRefused(RuntimeError):
    """The trial must not proceed, and nothing was written. Exit 3, never a FAIL row."""


def durable_write_trees(resolved: Any = None) -> dict[str, Path]:
    """The four live trees a durable write could reach, resolved at call time.

    Read from `app.paths` per call, not at import: the roots are env-overridable
    (`LLOYD_DATA`, `LLOYD_VAULT_ROOT`, `LLOYD_FACTS_ROOT`), and a proof that bound them at
    import would report the machine it was imported on. `facts` is
    `VAULT_FACTS_ROOT` and NOT the empty `~/obsidian/facts` the item names — see the
    module docstring. `data` is the whole data root, which subsumes the facts root; the
    overlap is deliberate (a changed file appears under both names, and `paths` are
    absolute so the two key spaces never collide).
    """
    p = resolved if resolved is not None else paths
    return {
        "memory": Path(p.VAULT_ROOT) / "memory",
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


def fetch_state_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """GET the aggregator's state endpoint. Raises `ProofRefused` if it will not answer.

    A server that cannot be asked is a refusal and not a FAIL: the claim under test is
    about where writes land, and no evidence either way was gathered. A 401 lands in the
    same branch, which is the correct reading — an unreadable state is an unknown state.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                   **auth_headers(url)})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ProofRefused(f"state endpoint {url} did not answer with JSON: "
                           f"{exc.__class__.__name__}: {exc}") from exc


def mcp_tool_call(url: str, tool: str, arguments: dict[str, Any], *,
                  headers: dict[str, str] | None = None,
                  timeout: float = 120.0) -> dict[str, Any]:
    """POST one JSON-RPC `tools/call` and return the decoded body.

    Deliberately thin: no retry, no session handshake, no fallback, because a trial that
    half-worked must read as a failed call in the record rather than a success with a
    caveat. The credential comes from `auth_headers(url)` — the real file read, on the
    default path every trial takes; `headers` overrides the whole set so a test can post a
    wrong one or none.

    The wire IS exercised, in both directions: `tests/test_bench_durable_write_proof.py`
    POSTs through this function to a real subprocess server that resolves its stores through
    `app.paths`, reports the real `tool_sandbox` block, and answers a wrong or absent
    credential with the same 401 the aggregator's own guard answers (`aggregator_auth.py`).
    That server stands in for the TOOL body, not for `agent_mcp/main.py`'s handler — the real
    on-box trial (#3689's owed list) is what pins the handler side.
    """
    body = rpc_body(tool, arguments)
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
    return {"tool": tool, "ok": "error" not in payload, "text": text, "response": payload}


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
    """Is a copy-side path inside the store that a named tool was asked to write?"""
    return rel_path.startswith(f"{store}/") or f"/{store}/" in f"/{rel_path}"


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


def run_proof(*, state: dict[str, Any], trees: dict[str, Path],
              call_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
              fixture_root: Path | None = None, diff_paths: list[str] | None = None,
              window_seconds: float = DEFAULT_WINDOW_SECONDS,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic,
              trial_id: str = "proof") -> dict[str, Any]:
    """One trial: guard, control window, live snapshots, the three calls, the verdict.

    `call_tool`, `sleep` and `clock` are injected because the two things this routine must
    prove are (a) that it asks the server BEFORE the first write and (b) that a real
    background change does not condemn it — neither of which is observable through a real
    aggregator and a wall clock. The order below is the contract: guard → live baseline
    check → control window → calls padded to the same length → live after → verdict.

    Both windows are `window_seconds` long BY CONSTRUCTION, not approximately: after the
    calls the trial waits out the remainder of the window. A shorter trial window than the
    control would be a control that saw more background than the trial could have, which
    forgives a real durable write — the one error direction this instrument must not have.
    Three writes over the box's own 42-changes-per-120-seconds rate is a boundary the
    padding is there to close rather than to reason about after the fact.
    """
    guard = prewrite_guard(state)
    intended = {row["tool"]: row["store"] for row in PROOF_TOOLS}
    record: dict[str, Any] = {
        "kind": "durable_write_proof", "trial_id": trial_id,
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
        call = call_tool(row["tool"], trial_arguments(row["store"], trial_id))
        # `result_text` is kept because the artifact must show WHERE the server says it
        # wrote: a row that records only `ok: true` cannot be checked against the live trees
        # by a later reader, and the record_path is the one field that can be.
        record["tool_calls"].append({"tool": row["tool"], "store": row["store"],
                                     "ok": bool(call.get("ok")),
                                     "result_text": str(call.get("text", ""))[:600],
                                     "detail": str(call.get("detail", ""))[:400]})
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

    `--plan` is the mode that runs today: it resolves the four trees, reports their file
    counts and refuses if any is vacuous, without contacting a server or attempting a
    write. It prints no verdict, because a proof of nothing has no verdict to print.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aggregator-url", default="",
                        help="the fixture-routed aggregator's /mcp (JSON-RPC tools/call)")
    parser.add_argument("--state-url", default="",
                        help="its state endpoint; defaults to $LLOYD_BENCH_STATE_URL")
    parser.add_argument("--fixture-root", default="",
                        help="the built fixture's vault, for the copy-side git diff")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--out", default="", help="write the record here as JSON")
    parser.add_argument("--plan", action="store_true",
                        help="resolve and size the live trees, ask nothing, write nothing")
    parser.add_argument("--trial-id", default=None,
                        help="the id written into every planted marker and into the record; "
                             "defaults to --out's stem, or 'proof'")
    args = parser.parse_args(argv)

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

    state_url = (args.state_url or os.environ.get(STATE_URL_ENV, "")).strip()
    if not args.aggregator_url or not state_url:
        print("refused: --aggregator-url and a state url (flag or $"
              + STATE_URL_ENV + ") are both required: this script does not boot an "
              "aggregator under the fixture's roots (that is #3688's spawner, or the "
              "scope ruling on #3689 that says this script owns it)", file=sys.stderr)
        return 3

    try:
        record = run_proof(
            state=fetch_state_json(state_url), trees=trees,
            call_tool=lambda name, arguments: mcp_tool_call(
                args.aggregator_url, name, arguments),
            fixture_root=Path(args.fixture_root) if args.fixture_root else None,
            window_seconds=args.window_seconds,
            trial_id=args.trial_id or (Path(args.out).stem if args.out else "proof"))
    except ProofRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3

    payload = json.dumps(record, indent=2)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    state = record["verdict"]
    return {"PASS": 0, "FAIL": 1}.get(state, 3)


if __name__ == "__main__":
    sys.exit(main())
