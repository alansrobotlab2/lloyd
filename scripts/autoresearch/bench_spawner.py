"""Boot one lloyd-mcp aggregator on ONE trial's fixture roots, armed for that trial's writes.

#3403 landed the sandbox that refuses a bench write to production state, and
`require_copied_state` in `bench_runner_sdk.py` is written for a server somebody else boots —
but nobody ever booted one. Every `writes:`-scoped trial was therefore refused before it ran
(`tests/test_autoresearch_bench_runner_sdk.py` pins that refusal), and the three dead routines
(`declared_scopes`, `write_scope_of`, `score_side_effects`) had no caller. This is the missing
boot: the same entry point supervisor uses (`agent_mcp.main.main`), one subprocess per trial,
with `fixture.env()` in its environment, `LLOYD_BENCH_WRITE_SCOPE` armed from the task's
declared scopes, and its URL published for `LLOYD_BENCH_STATE_URL`.

**The armed scope comes from the task, and nothing wider.** `agent_mcp._tool_sandbox` gates
every mutating tool on two facts (its `writes` decorator): the tool is named in
`LLOYD_BENCH_WRITE_SCOPE`, and `state_is_copied()` — which asks whether BOTH roots the process
routes to are non-production. Arming a scope wider than the task declares would let an agent
write a store its card never asked for and still grade green, so `spawn` takes the scope as a
required argument and passes it through verbatim.

**The boot is refused, loudly, if the child would be production.** Before spawning: the
resolved data root and vault root must not be the account's, and the fixture must not be
inside either live root. `bench_fixture.py` already refuses to BUILD a fixture from those
arguments; this is the second half of that promise, because a fixture path is not enough — the
ENVIRONMENT is what routes the child, and an environment that lost `LLOYD_VAULT_ROOT` would
point a scope-armed aggregator at Alan's real vault.

**Who owns the process.** `spawn()` returns a handle and does NOT clean up; the caller stops
it. `cli()` is the standalone mode for a human (`--seconds 120`, then stop), which is how the
`curl <url>/state | jq .tool_sandbox.state_is_copied` witness in #3688 is produced. A child
left running would hold a port and a fixture directory, so `stop()` terminates, waits, and
kills; `cli()` also installs a SIGTERM/SIGINT handler so an interrupted run reaps its child
rather than orphaning an aggregator that answers `/state` forever.

Run as a module: `python -m scripts.autoresearch.bench_spawner --fixture-root DIR --scope
vault,backlog [--seconds 120]`. The fixture directory must come from `build_fixture`, which
mandates `<data_root>/eval/fixtures`.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # only for annotations; the module imports lazily at use
    from scripts.autoresearch.bench_fixture import BenchFixture

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_mcp._tool_sandbox import BENCH_WRITE_SCOPE_ENV  # noqa: E402
from app.data_root import (  # noqa: E402
    DATA_ROOT_MARKER, PRODUCTION_DATA_ROOT, production_vault)

#: The only host the aggregator binds: `agent_mcp/main.py:1151` passes `host="127.0.0.1"` to
#: `uvicorn.run` with no flag for another, and `aggregator_auth` only releases the credential
#: to a loopback host anyway.
LOOPBACK_HOST = "127.0.0.1"

#: How long `spawn` waits for the child to answer `/state` with a complete isolation block.
#: The aggregator imports the SDK, builds its tool list and opens a kg connection, which
#: measured ~4 s warm on this box; a cold cache and 8-way test parallelism make it worse, and
#: a too-short ceiling turns a working boot into a "did not become ready" refusal that reads
#: like a code fault. Generous because the readiness loop exits the moment the answer is good.
READY_TIMEOUT_S = 90.0

#: `GET /state` keys this spawner requires before calling a child ready. Membership, not
#: truthiness, for the reason `bench_runner_sdk._SERVER_ISOLATION_KEYS` gives: an aggregator
#: that predates #3403 answers with no isolation block at all, and a truthy-check would read
#: that silence as "fine, just no flags".
REQUIRED_STATE_KEYS = ("is_production_data", "is_production_vault", "state_is_copied",
                       "allowed_writes")


def free_port(host: str = LOOPBACK_HOST) -> int:
    """A port free to bind RIGHT NOW — and closed again before returning.

    The classic bug lives in the gap: pick 0, keep the socket, hand the number out, and the
    child's bind fails because the parent is still holding it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def production_roots() -> dict[str, Path]:
    """The two roots whose ownership by a bench child would be the incident #3403 exists to stop.

    Read from the `app.data_root` PREDICATES, not from `app.paths`: `paths` reports
    `IS_PRODUCTION_*=False` whenever the gate redirects the env for a trial, so asking it
    "is this production?" inside a trial would always answer no. `production_vault()` is the
    account's vault computed from the passwd entry, so it survives that redirect too.
    """
    return {"data": PRODUCTION_DATA_ROOT, "vault": production_vault()}


def assert_not_production(env: dict[str, str]) -> None:
    """Refuse to boot an armed aggregator on the account's own data or vault.

    Read from the ENVIRONMENT the child will get, never from `app.paths`: `app.paths` resolved
    its roots when THIS process imported it, so inside a trial it reports the wrong tree, and
    `IS_PRODUCTION_*` there reports False whenever the gate redirects the env — asking it "is
    this production?" would always answer no. The child's routing is decided by the three
    variables below, so those are what gets checked, and a MISSING one is the hazard: a
    fixture env that lost `LLOYD_VAULT_ROOT` leaves the child on `/home/<user>/obsidian`.

    This is the parent's cheap pre-flight. The authority is the readiness loop below, which
    asks the booted child what it resolved and refuses if `state_is_copied` comes back False
    — a check of the bytes on the wire, not of the intent in a dict.
    """
    from app.data_root import VAULT_ROOT_ENV, VAULT_ROOT_FALLBACK_ENV

    house = production_roots()
    routing = {
        "data root": (env.get("LLOYD_DATA"), house["data"]),
        "vault root": (env.get(VAULT_ROOT_ENV) or env.get(VAULT_ROOT_FALLBACK_ENV),
                       house["vault"]),
    }
    for label, (declared, live) in routing.items():
        if not declared:
            raise RuntimeError(
                f"refusing to boot a write-scope-armed aggregator: the child's environment "
                f"names no {label} (neither LLOYD_DATA nor "
                f"{VAULT_ROOT_ENV}/{VAULT_ROOT_FALLBACK_ENV}, as applicable), so it would "
                f"resolve {live} — this account's own. `BenchFixture.env()` sets all three "
                f"names precisely so that cannot happen; an env that reached here without them "
                f"did not come from `fixture.env()`.")
        if _same(Path(declared), live):
            raise RuntimeError(
                f"refusing to boot a write-scope-armed aggregator: the child's {label} is "
                f"{live}, which IS this account's. A scope-armed server on the house vault is "
                f"the accident #3403 was written to prevent, so this is a refusal and not a "
                f"warning.")


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a) == str(b)


def auth_headers(env: dict[str, str]) -> dict[str, str]:
    """The credential header for a child booted with `env` — read with THAT environment.

    `aggregator_auth.headers_for_url` reads this process's environment, and the process that
    MINTED the token is the fixture child, which writes it under the fixture's own data root
    (`token_path` is `DATA_ROOT`-derived). So asking the shared helper for a header here would
    hand back this process's credential or none, and every probe would be a 401 that reads as
    "the boot never happened". The name of the header and the rule for the file both still come
    from `aggregator_auth` — one definition, read through the environment that owns the secret.
    """
    from agent_mcp.aggregator_auth import AUTH_HEADER, read_token

    token = read_token(publish=False, env=env)
    return {AUTH_HEADER: token} if token else {}


def isolation_state(url: str, *, env: dict[str, str] | None = None,
                    timeout: float = 5.0) -> dict[str, Any]:
    """The child's `tool_sandbox` block, or a dict carrying `error`.

    Carries the credential because `OPEN_PATHS` is `{"/health"}` and an unauthenticated
    `/state` is a 401 — a probe that treated 401 as "not ready yet" would spin to the timeout
    and report a boot that never happened. Loopback only, which is the rule
    `aggregator_auth.headers_for_url` exists to enforce; the spawner binds 127.0.0.1.
    """
    try:
        request = urllib.request.Request(url.rstrip("/") + "/state",
                                         headers=auth_headers(env or dict(os.environ)))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code} on GET /state"}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}
    block = body.get("tool_sandbox")
    if not isinstance(block, dict):
        return {"error": "GET /state answered with no tool_sandbox block"}
    missing = [k for k in REQUIRED_STATE_KEYS if k not in block]
    if missing:
        return {"error": f"tool_sandbox block missing key(s): {missing}"}
    return {"sandbox": block}


class BenchAggregator:
    """One running fixture-rooted aggregator: its url, its environment, its stop button."""

    def __init__(self, proc: subprocess.Popen, url: str, env: dict[str, str],
                 port: int, log_path: Path) -> None:
        self.proc = proc
        self.url = url
        self.env = env
        self.port = port
        self.log_path = log_path
        self.stopped: tuple[int, float] | None = None

    @property
    def alive(self) -> bool:
        return self.stopped is None and self.proc.poll() is None

    def stop(self, *, terminate_s: float = 8.0, kill_s: float = 5.0) -> tuple[int, float]:
        """Terminate, then kill; return (returncode, uptime_seconds).

        terminate-before-kill because the aggregator closes its kg connection and MCP sessions
        on SIGTERM, and a child that dies holding a SQLite lock leaves a `-wal` the next
        reader trips over. The clock starts at the terminate signal, so the figure that comes
        out is how long the process ran.
        """
        if self.stopped is not None:
            return self.stopped
        started = time.monotonic()
        self.proc.terminate()
        try:
            code = self.proc.wait(timeout=terminate_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                code = self.proc.wait(timeout=kill_s)
            except subprocess.TimeoutExpired:
                code = self.proc.poll()
        self.stopped = (code if code is not None else -1,
                        started - self.started_monotonic)
        return self.stopped

    started_monotonic: float = 0.0

    def child_isolation(self) -> dict[str, Any]:
        return isolation_state(self.url, env=self.env)

    def __enter__(self) -> "BenchAggregator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


def fixture_from_dir(root: Path, scopes: Sequence[str]) -> "BenchFixture":
    """Rebuild the `BenchFixture` a directory was built as, for a caller holding only a path.

    The standalone CLI receives `--fixture-root` as a string and still needs the fixture's
    `env()`. Every field is derivable from the layout `build_fixture` writes —
    `<root>/lloyd-data` and `<root>/obsidian`, the copy's data root named for the tree it
    stands in for (`fixture_data = dest / "lloyd-data"`, `bench_fixture.py:305`) — so nothing
    is stored and nothing is guessed. But the directory is checked against that layout FIRST,
    with `build_fixture`'s own data-root marker as the witness that this tree was built by the
    builder and not assembled by hand next to the live vault.

    The names are read from that builder, not from a plausible spelling: this function asked
    for `<root>/data` first, and a real fixture has no `data/` directory, so the marker test
    below failed on every genuine fixture and `--fixture-root` could only ever refuse. A
    layout guess that fails SHUT is the lucky shape — the same guess applied to a copy whose
    `obsidian/` slice and data slice are named differently would have handed `env()` the live
    roots instead, which is what the marker test is here to make impossible.
    """
    from scripts.autoresearch.bench_fixture import BenchFixture

    data_root, vault = root / "lloyd-data", root / "obsidian"
    if not (data_root / DATA_ROOT_MARKER).is_file():
        raise RuntimeError(
            f"{root} is not a fixture directory: it has no `{DATA_ROOT_MARKER}` marker in "
            f"`lloyd-data/`, which `build_fixture` writes at bench_fixture.py:310. A hand-made "
            f"directory has never been checked against the live roots by that builder, and its "
            f"cleanup is not the one that deletes ONLY inside the fixtures area.")
    if not vault.is_dir():
        raise RuntimeError(f"{root} has no `obsidian/` vault slice; it is not a fixture")
    return BenchFixture(id=root.name, root=root, vault=vault, data_root=data_root,
                        scopes=list(scopes))


def spawn(*, fixture_root: "Path | str | BenchFixture", scopes: Sequence[str],
          repo_root: Path | str | None = None, host: str = LOOPBACK_HOST,
          port: int | None = None, ready_timeout: float = READY_TIMEOUT_S,
          token: str | None = None) -> BenchAggregator:
    """Boot the aggregator on this fixture's roots with `scopes` armed; block until ready.

    `fixture_root` is the `BenchFixture` `build_fixture` returned, or its directory. `scopes`
    is what the task declares (`declared_scopes`), passed through verbatim: the spawner never
    widens it. Returns after `GET /state` reports a complete isolation block with
    `state_is_copied` true, so a caller holding the handle has a server the runner's own gate
    will accept. Raises on any condition that would make the boot unsafe or useless, with the
    offending values in the message.
    """
    if not scopes:
        raise RuntimeError(
            "refusing to spawn: no declared write scopes. An aggregator booted with no "
            "LLOYD_BENCH_WRITE_SCOPE answers state_is_copied true and refuses every mutating "
            "tool — a trial against it cannot tell 'isolated' from 'broken', which is the "
            "distinction #3403 exists to make.")
    root = (Path(fixture_root).resolve() if not hasattr(fixture_root, "env")
            else Path(fixture_root.root).resolve())
    house = production_roots()
    # The inside test is `build_fixture`'s OWN refusal predicate, not a second spelling of it:
    # `_refuse_if_inside` tests the literal and the resolved path against both the tree itself
    # and every ancestor of it, so a fixture reached through a symlink into `~/obsidian` is
    # refused here for the same reason the builder refuses to build it, and the two rails can
    # never disagree about what "inside" means. The first cut of this line was
    # `root == live or root in live.parents`, which is the CONTAINMENT test pointing the wrong
    # way: it refused a fixture root that held the live tree and waved through
    # `/home/<user>/lloyd-data/somewhere/fixture` — a scope-armed server sitting inside the
    # account's own data root, which is the one shape the message claimed to refuse.
    from scripts.autoresearch.bench_fixture import FixtureRefused, _refuse_if_inside

    for label, live in (("data root", house["data"]), ("vault root", house["vault"])):
        try:
            _refuse_if_inside(root, live, f"the fixture root {root}")
        except FixtureRefused as exc:
            raise RuntimeError(
                f"refusing to spawn: {exc} — and the {label} {live} is this account's live "
                f"tree. A fixture belongs under <data_root>/eval/fixtures, which "
                f"`build_fixture` already enforces; reaching here means it did not come from "
                f"`build_fixture`.") from None
        if live.is_relative_to(root):
            raise RuntimeError(
                f"refusing to spawn: the fixture root {root} CONTAINS the live {label} {live}. "
                f"A directory with a copy's name over the house is not a fixture, and its "
                f"`cleanup()` would be one `rmtree` away from the tree it is meant to protect.")

    fixture = (fixture_root if hasattr(fixture_root, "env")
               else fixture_from_dir(root, scopes))
    env = fixture.env()
    # TOOL NAMES, never the scope words the card declares. `_tool_sandbox.write_scope_tools()`
    # reads this variable as a list of tool names, and the runner arms it the same way
    # (`bench_runner_sdk.py:1081`, `",".join(write_scope_of(task))`), so arming the words here
    # made a `writes: [vault]` trial's server report `allowed_writes: ["vault"]` — a name that is
    # not a tool — and `require_copied_state` then refused the trial with "honours none of
    # [memory_add, memory_remove, memory_replace, vault_write]": every durable-write trial dead
    # on arrival, and the boot probe green, because the probe was checking the words round-tripped.
    # Expanded through the runner's own `write_scope_of`, not a second mapping: the card's `writes:`
    # entry, the runner's grant and this server's gate are then one lookup, so a new write class
    # cannot be reachable from one and invisible to the other.
    from scripts.autoresearch.bench_runner_sdk import write_scope_of

    granted = write_scope_of({"writes": list(scopes)})
    env[BENCH_WRITE_SCOPE_ENV] = ",".join(granted)
    env["LLOYD_MCP_PORT"] = str(port or free_port(host))
    if token:
        # A fixture's token file is the copy's own secret, already in the env as
        # `LLOYD_MCP_TOKEN_FILE`; `--token` exists for a caller that would rather pass it by
        # env, and the aggregator reads this variable before the file (`read_token`).
        env["LLOYD_MCP_TOKEN"] = token
    assert_not_production(env)

    chosen = int(env["LLOYD_MCP_PORT"])
    log_path = root / "aggregator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # `agent_mcp/main.py` has no argparse: it binds `127.0.0.1:PORT` where PORT comes from
    # `LLOYD_MCP_PORT` (`:119`), the same env the canary uses to take a second port. So the
    # port travels in the environment and the host does not travel at all.
    argv = [sys.executable, "-m", "agent_mcp.main"]
    with log_path.open("ab") as log:
        proc = subprocess.Popen(argv, cwd=str(repo_root or REPO_ROOT), env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    handle = BenchAggregator(proc=proc, url=f"http://{host}:{chosen}", env=env,
                             port=chosen, log_path=log_path)
    handle.started_monotonic = time.monotonic()
    deadline = time.monotonic() + ready_timeout
    last = "never answered"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"the fixture-rooted aggregator exited rc={proc.returncode} before becoming "
                f"ready on {handle.url}; its own log is {log_path} (last lines: "
                f"{_tail(log_path)})")
        state = isolation_state(handle.url, env=env)
        if "sandbox" in state:
            block = state["sandbox"]
            if not block["state_is_copied"]:
                raise RuntimeError(
                    f"the spawned aggregator reports state_is_copied False at {handle.url} "
                    f"with is_production_data={block['is_production_data']} "
                    f"is_production_vault={block['is_production_vault']}: the roots it resolved "
                    f"are not a copy, so no trial may run against it. Environment: "
                    f"LLOYD_DATA={env.get('LLOYD_DATA')!r} "
                    f"LLOYD_VAULT_ROOT={env.get('LLOYD_VAULT_ROOT')!r}")
            armed = sorted(set(env[BENCH_WRITE_SCOPE_ENV].split(",")) & set(block["allowed_writes"]))
            if not armed:
                raise RuntimeError(
                    f"the spawned aggregator accepted the boot but its `allowed_writes` "
                    f"{block['allowed_writes']} share no tool with what the declared scopes "
                    f"{sorted(set(env[BENCH_WRITE_SCOPE_ENV].split(',')))} grant: the tools the "
                    f"task names are not the tools the server gates, so every write in the "
                    f"trial would be refused for the wrong reason.")
            return handle
        last = str(state.get("error"))
        time.sleep(0.25)
    handle.stop()
    raise RuntimeError(
        f"fixture-rooted aggregator did not become ready within {ready_timeout:.0f}s on "
        f"{handle.url} (last probe: {last}); log: {log_path} (last lines: {_tail(log_path)})")


def _tail(path: Path, lines: int = 4) -> str:
    try:
        return " | ".join(path.read_text(encoding="utf-8", errors="replace")
                          .splitlines()[-lines:])[:400]
    except OSError as exc:
        return f"<unreadable: {exc.__class__.__name__}>"


def cli(argv: Sequence[str] | None = None) -> int:
    """Standalone mode: boot, print the witness line a human can `curl`, then stop cleanly."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture-root", required=True,
                    type=Path, help="A directory built by build_fixture")
    ap.add_argument("--scope", required=True,
                    help="Comma-separated declared scopes (vault, memory, facts, backlog, kg, ...)")
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="How long to keep it up before stopping it (default 120)")
    ap.add_argument("--token", default=None,
                    help="Pass the aggregator credential by env instead of the fixture file")
    args = ap.parse_args(list(argv) if argv is not None else None)

    scopes = tuple(s.strip() for s in args.scope.split(",") if s.strip())
    try:
        handle = spawn(fixture_root=args.fixture_root, scopes=scopes, token=args.token)
    except (RuntimeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def _reap(signum, _frame):
        print(f"signal {signum}: stopping the fixture aggregator before exiting",
              file=sys.stderr)
        handle.stop()
        sys.exit(143 if signum == signal.SIGTERM else 130)

    for sig in previous:
        signal.signal(sig, _reap)
    try:
        state = handle.child_isolation()
        block = state.get("sandbox") or {}
        # The header is part of the command, not decoration: `OPEN_PATHS` is `{"/health"}`
        # (`agent_mcp/aggregator_auth.py:73` names the header, `:95` the open paths, `:106` the
        # file), so a
        # bare `curl <url>/state` is answered 401 — measured on the first real boot, where the
        # line printed here returned nothing while `isolation_state`, which sends the
        # credential, got the block. A published witness a person cannot run is how an
        # unverified reading of `state_is_copied` gets quoted later, so the command names the
        # file this child's credential actually came from instead of a token in argv or stdout.
        from agent_mcp.aggregator_auth import token_path as _token_path

        print(json.dumps({
            "url": handle.url,
            "state_url_env": {"LLOYD_BENCH_STATE_URL": handle.url},
            "write_scope_armed": sorted(
                set(handle.env[BENCH_WRITE_SCOPE_ENV].split(",")) & set(block.get("allowed_writes", []))),
            "state_is_copied": block.get("state_is_copied"),
            "is_production_data": block.get("is_production_data"),
            "is_production_vault": block.get("is_production_vault"),
            "witness_command": (
                f"curl -s -H 'X-Lloyd-Aggregator-Token: $(head -1 {_token_path(handle.env)})' "
                f"{handle.url}/state | jq .tool_sandbox.state_is_copied"),
        }, indent=2, sort_keys=True))
        time.sleep(max(0.0, args.seconds))
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        code, up = handle.stop()
        print(f"aggregator stopped rc={code} after {up:.1f}s uptime; log: {handle.log_path}",
              file=sys.stderr)


if __name__ == "__main__":
    sys.exit(cli())
