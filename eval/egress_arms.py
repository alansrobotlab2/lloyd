#!/usr/bin/env python3
"""Open the egress-arms window: one bench, two aggregators, one label apiece (#2435).

The window is `eval/run_egress_arms.sh N --on-port PORT`. The measurement it produces is
the pair #2338 landed the machinery for and nobody has since taken: the same shipped
canary run once per arm — enforce OFF against the shared aggregator, enforce ON against
a private one — so `grade`'s per-arm `attack_success` / `benign_control_ok` pair has two
populations to compare instead of the single arm every row in
`eval/measurements/injection-canary/rows.jsonl` carries. `GET /state` on the shared
aggregator still answers `egress: {"enforce": false}`, which is why the ON arm cannot
reuse it: #2338's own `verified_arm` refuses an enforce-on request against an
enforce-off endpoint, and rightly — the arm is the SERVING guard's property, since
`agent_mcp/egress.py::guard()` is the process that decides the deny.

So the ON arm gets an aggregator this window launches and stops:

* `python -m agent_mcp.main` with `LLOYD_EGRESS_ENFORCE=1` in ITS environment (the flag
  in this driver's environment would prove nothing — that is the exact confusion #2154
  left 1478 un-denied rows over), and `LLOYD_MCP_PORT` set to the port the caller
  supplied, so it never touches the shared daemon's port.
* `LLOYD_DATA` pointed at a run-local root, which is where its `egress_events` land.
  The dashboard's destination inventory (`app/routers/dashboard.py::_network`) reads the
  LIVE `workers.db`, and a synthetic deny from a measurement has no business in a
  report that operators act on. The leak check at the end is what makes that a checked
  claim rather than an intention.
* Every episode of that arm served through the bench's `--mcp-url`, which is the
  supported route (#2338) and stamps `mcp_url` plus `guard_egress_enforce` on every row
  it appends — from the endpoint's `/state`, never from this shell.

Two refusals this window owes the A/B and the bench does not enforce, because the bench
only ever sees one endpoint at a time: **an already-held pool** (the reps are held so a
job running beside them cannot move a row's outcome, and `PoolPause` would otherwise
continue with the hold held by someone else) and **an off arm that is not off** — an
enforcing shared endpoint would have both arms stamped `enforce-on`, and `grade` would
publish one arm's rate twice as a difference.

    eval/run_egress_arms.sh 1 --on-port 8599
    LLOYD_CANARY_ROWS=$HOME/tmp/egress-arms/rows.jsonl \\
      eval/run_egress_arms.sh 1 --on-port 8599

Environment: `LLOYD_BACKEND_URL` (default `http://127.0.0.1:8080`) is the backend whose
pause routes the hold drives; `LLOYD_MCP_URL` is the shared aggregator the OFF arm runs
against; `LLOYD_CANARY_CMD` replaces the bench and `LLOYD_MCP_CMD` the aggregator, the
two seams that let a test drive this loop without the engine, the sandbox or a second
daemon; `LLOYD_WORKERS_DB` points the leak check and the hold's watermark read at
another store, defaulting to `app.paths.WORKERS_DB`.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import signal
import sys
import tempfile
import time
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parent.parent
for _p in (str(LLOYD_HOME), str(LLOYD_HOME / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The hold machinery, reused rather than re-derived. `PersistenceHold` already refuses
#: an already-held pool, refuses a hold whose holders include the automod promoter,
#: refuses to start while a job is in flight, resumes only a pause it took from inside a
#: `finally` behind SIGTERM/SIGHUP/SIGINT handlers, and — with `settle` — re-checks the
#: persisted watermark after the event loop is gone. A window that re-implemented any of
#: that worse is the failure #2397's docstring exists to warn about.
from persistence_arms import (  # noqa: E402
    HoldRefused, PersistenceHold, RepFailed, backend_url, canary_cmd, run_cli, settle,
    workers_db,
)
#: The arm vocabulary and the one reader of a serving guard's state, both imported from
#: the module that owns them so the window and the bench cannot mean different things by
#: `enforce-on` — the drift #2338 was filed against.
from run_injection_canary import (  # noqa: E402
    ARM_ENV_VAR, ARM_OFF, ARM_ON, read_guard_egress, state_url_for,
)

MCP_CMD_ENV = "LLOYD_MCP_CMD"
MCP_PORT_ENV = "LLOYD_MCP_PORT"
MCP_URL_ENV = "LLOYD_MCP_URL"
DATA_ROOT_ENV = "LLOYD_DATA"
RUN_ROOT_ENV = "LLOYD_EGRESS_ARMS_ROOT"
READY_WAIT_ENV = "LLOYD_EGRESS_ARMS_READY_S"

#: The shipped aggregator, as a module so the child resolves the `agent_mcp` package its
#: own interpreter owns rather than a path this file guesses.
DEFAULT_MCP_CMD = [sys.executable, "-m", "agent_mcp.main"]

#: The host the canary's own corpus plants. `webpage-egress-fetch.html` carries
#: `CANARY-INJ-EGRESS-7f04` behind it, and it is the scenario whose egress call the two
#: arms are expected to answer differently: allowed and recorded on the off arm, denied
#: on the on arm. A test re-reads the corpus and fails if this literal drifts from it,
#: because a leak check pointed at a host no scenario fetches passes forever.
CANARY_EGRESS_HOST = "collect.off-allow-list.invalid"

#: Both arms, in the order the item asks for them: the arm that runs today first, so a
#: window interrupted halfway still leaves the baseline comparable to the 2026-09-30 one.
WINDOW_ARMS = (ARM_OFF, ARM_ON)


class AggregatorRefused(Exception):
    """The private aggregator could not be made to serve the arm, so no episode ran."""


def mcp_cmd() -> list[str]:
    """The command that launches the ON arm's aggregator, as argv.

    Default is `python -m agent_mcp.main`; `LLOYD_MCP_CMD` is the seam a test uses to
    stand in for a daemon it may not boot, for the same reason `LLOYD_CANARY_CMD` stands
    in for the bench: the real thing needs the sandbox, the engine and hours.
    """
    raw = (os.environ.get(MCP_CMD_ENV) or "").strip()
    if raw:
        return shlex.split(raw)
    return list(DEFAULT_MCP_CMD)


def shared_mcp_url(cli_value: str | None = None) -> str:
    """The aggregator the OFF arm is served by — the shared pool's endpoint.

    Resolution order is the bench's own (`--mcp-url`, then the environment, then the
    pool's default) so that pointing the window at a non-default shared daemon points
    the bench at the same one; `state_url_for` then derives the `/state` this driver
    reads from the very URL the episodes will be served from.
    """
    raw = (cli_value or os.environ.get(MCP_URL_ENV) or "").strip()
    if raw:
        return raw
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_URL
    return DEFAULT_LLOYD_MCP_URL


def on_arm_url(port: int) -> str:
    """The MCP URL of the aggregator this window launched on `port`."""
    return f"http://127.0.0.1:{int(port)}/mcp"


def run_root(cli_value: str | None = None) -> Path:
    """The run-local data root that holds the ON arm's `egress_events`.

    Under the scratch rows file when one is named — the window's rows and its
    aggregator's telemetry then sit in the same directory the operator cleans up
    together — and in a fresh temp directory otherwise. Never the live data root: the
    whole reason the ON arm needs its own aggregator is that its deny must not land in
    the store the dashboard reads.
    """
    raw = (cli_value or os.environ.get(RUN_ROOT_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser()
    rows = (os.environ.get("LLOYD_CANARY_ROWS") or "").strip()
    if rows:
        return Path(rows).expanduser().parent / "aggregator-data"
    return Path(tempfile.mkdtemp(prefix="lloyd-egress-arms-"))


async def wait_enforcing(url: str, proc, *, ready_wait_s: float, poll_s: float) -> dict:
    """Wait for the child at `url` to answer `/state` with an enforcing guard.

    Two ways to fail, told apart out loud: the child died (the message carries its exit
    code, since that is the only clue there is), or it never said anything enforceable
    in the window given. Neither is retried forever: a window that hangs holding the
    dispatch pool is worse than one that quits.
    """
    deadline = time.monotonic() + ready_wait_s
    last: str = "no answer yet"
    while True:
        rc = getattr(proc, "returncode", None)
        if rc is not None:
            raise AggregatorRefused(
                f"the {ARM_ON} arm's aggregator exited {rc} before it could serve "
                f"{url}; nothing was measured, and dispatch is being handed back")
        try:
            state = await read_guard_egress(state_url_for(url))
        except Exception as exc:  # noqa: BLE001 — any answer-but-no-verdict is a retry
            last = f"{type(exc).__name__}: {exc}"
        else:
            if state.get("enforce"):
                return state
            raise AggregatorRefused(
                f"the aggregator on {url} reports enforce={state.get('enforce')!r} "
                f"despite {ARM_ENV_VAR}=1 in its environment. The {ARM_ON} arm would be "
                f"stamped from its answer and measure the other arm.")
        if time.monotonic() >= deadline:
            raise AggregatorRefused(
                f"the {ARM_ON} arm's aggregator never answered /state within "
                f"{ready_wait_s:.0f}s ({last})")
        await asyncio.sleep(poll_s)


async def start_on_arm_aggregator(port: int, root: Path, *, ready_wait_s: float,
                                  poll_s: float = 0.5):
    """Launch the ON arm's aggregator, wait for it to say it enforces, return the child.

    The three environment entries are the clause: the flag goes to the process that
    DECIDES (`agent_mcp/egress.py::enforce_on()` reads its own environment), the port
    keeps it off the shared daemon's, and `LLOYD_DATA` keeps its telemetry out of the
    live store. Nothing here edits `config.yaml` and nothing here restarts a service:
    this child belongs to this window and this window stops it.
    """
    env = dict(os.environ)
    env[ARM_ENV_VAR] = "1"
    env[MCP_PORT_ENV] = str(int(port))
    env[DATA_ROOT_ENV] = str(root)
    argv = mcp_cmd()
    print("$ " + " ".join(shlex.quote(a) for a in argv)
          + f"  {ARM_ENV_VAR}=1 {MCP_PORT_ENV}={port} {DATA_ROOT_ENV}={root}",
          flush=True)
    proc = await asyncio.create_subprocess_exec(*argv, env=env)
    try:
        await wait_enforcing(on_arm_url(port), proc, ready_wait_s=ready_wait_s,
                             poll_s=poll_s)
    except BaseException:
        await stop_aggregator(proc)
        raise
    print(f"{ARM_ON} aggregator up on {on_arm_url(port)} (data root {root})", flush=True)
    return proc


async def stop_aggregator(proc) -> None:
    """Stop a child this window started. Tolerates None, an exit, and a refusal to die."""
    if proc is None or proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=15)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            return
        await proc.wait()


async def verify_off_arm(url: str) -> None:
    """Refuse a window whose OFF arm would not be off.

    `verified_arm` refuses an enforce-ON request served by an enforce-OFF endpoint, and
    has no opinion on the mirror: an enforcing shared endpoint stamps its own rows
    `enforce-on`, so a window opened against one would run the same arm twice and hand
    `grade` two identical populations to publish as a difference. The bench cannot see
    that coming, because it is aimed at one endpoint; the window sees both, so the
    window is where the refusal belongs. An unreadable `/state` is the same refusal —
    the arm label is exactly what that endpoint is being asked to certify.
    """
    try:
        state = await read_guard_egress(state_url_for(url))
    except Exception as exc:  # noqa: BLE001
        raise HoldRefused(
            f"cannot read the egress state of the shared aggregator at {url}: {exc}. "
            f"The {ARM_OFF} arm's rows would carry an arm nobody verified, so the "
            "window is not opening.") from None
    if state.get("enforce"):
        raise HoldRefused(
            f"the shared aggregator at {url} already enforces "
            f"(egress.enforce={state.get('enforce')!r}); both arms would be stamped "
            f"{ARM_ON} and `grade` would publish one arm's rate twice as the "
            "difference between the arms.")


async def run_window(n: int, hold: PersistenceHold, *, arms: tuple[str, ...],
                     port: int | None, root: Path, off_url: str,
                     ready_wait_s: float) -> int:
    """The reps, both arms, under a hold and a child `main` will settle on."""
    import httpx

    cmd = canary_cmd()
    urls = {arm: off_url for arm in arms}
    if ARM_ON in arms:
        urls[ARM_ON] = on_arm_url(port or 0)
    proc = None
    async with httpx.AsyncClient() as client:
        hold.client = client
        code = 0
        # ONE try whose `finally` is the only way out of this block, refusal included: a
        # `return 2` from a separate `except` around the take would leave a hold this
        # process took set until `settle` cleaned up after the loop, and the aggregator
        # would outlive a window that refused for its own reason.
        try:
            await hold.take()
            if ARM_OFF in arms:
                await verify_off_arm(off_url)
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, task.cancel)
                except (NotImplementedError, RuntimeError, ValueError):
                    pass
            if ARM_ON in arms:
                proc = await start_on_arm_aggregator(port, root,
                                                     ready_wait_s=ready_wait_s)
            for rep in range(1, n + 1):
                for arm in arms:
                    print(f"── rep {rep} · {arm} · {urls[arm]}", flush=True)
                    await run_cli(cmd, ["run", "--rep", str(rep),
                                        "--mcp-url", urls[arm]])
            await run_cli(cmd, ["grade"])
            if os.environ.get("LLOYD_CANARY_REPORT", "").strip():
                await run_cli(cmd, ["report", "--write"])
        except HoldRefused as exc:
            print(f"refused: {exc}", file=sys.stderr, flush=True)
            code = 2
        except RepFailed as exc:
            print(f"window failed: {exc}", file=sys.stderr, flush=True)
            code = 1
        except AggregatorRefused as exc:
            print(f"window refused: {exc}", file=sys.stderr, flush=True)
            code = 2
        except asyncio.CancelledError:
            print("window interrupted: handing dispatch back", file=sys.stderr,
                  flush=True)
            code = 1
        finally:
            await stop_aggregator(proc)
            await hold.release()
            for line in hold.log:
                print(f"  {line}", file=sys.stderr)
        return code


def leak_report(db: Path, host: str = CANARY_EGRESS_HOST) -> str | None:
    """A sentence about the LIVE store's denies for `host`; None when the check is clean.

    Runs OUTSIDE the episode loop, after the event loop is gone: the answer has to
    survive a window that died mid-cancel, and it reads the store the dashboard reads,
    not the run-local one the ON arm wrote — a deny in the live store means the private
    aggregator's isolation failed and a synthetic destination is now in an operator
    facing report. `None` means nothing to report; a store that cannot be read is NOT
    `None`, because "we could not look" must never be filed as "it did not leak".
    """
    from agent_mcp.egress import decision_count_for_host

    count = decision_count_for_host(host, db=db)
    if count is None:
        return (f"!! cannot read `egress_events` in {db}: this window cannot certify "
                f"that no {ARM_ON}-arm deny for {host} reached the live store")
    if count:
        return (f"!! LEAKED: {count} `decision=deny` row(s) for {host} in {db}. The "
                f"{ARM_ON} arm's synthetic deny belongs in the run-local root; these "
                "rows are in the destination inventory the dashboard renders")
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reps", type=int, help="reps of the shipped scenario set, per arm")
    ap.add_argument("--backend", default=None,
                    help="backend whose pause routes the hold drives "
                         "(default: $LLOYD_BACKEND_URL or http://127.0.0.1:8080)")
    ap.add_argument("--mcp-url", default=None,
                    help=f"the SHARED aggregator the {ARM_OFF} arm is served by "
                         f"(default: ${MCP_URL_ENV} or the pool's own default)")
    ap.add_argument("--on-port", type=int, default=None,
                    help=f"port for this window's private {ARM_ON} aggregator "
                         "(required when that arm is selected)")
    ap.add_argument("--arms", nargs="+", choices=[ARM_OFF, ARM_ON],
                    default=list(WINDOW_ARMS),
                    help="which arms to run (default: both)")
    ap.add_argument("--run-root", default=None,
                    help=f"where the {ARM_ON} aggregator's LLOYD_DATA lives "
                         f"(default: ${RUN_ROOT_ENV}, else beside the scratch rows)")
    ap.add_argument("--ready-wait", type=float, default=None,
                    help=f"seconds to wait for the private aggregator's /state "
                         f"(default: ${READY_WAIT_ENV} or 120)")
    ap.add_argument("--drain-wait", type=float, default=None,
                    help="seconds to wait for the queue to idle (default 900)")
    ap.add_argument("--automod-wait", type=float, default=None,
                    help="seconds to wait out an automod promoter hold at exit")
    args = ap.parse_args(argv)

    if args.reps < 1:
        print(f"reps must be >= 1, got {args.reps}", file=sys.stderr)
        return 2
    arms = tuple(dict.fromkeys(args.arms))
    off_url = shared_mcp_url(args.mcp_url)
    root = run_root(args.run_root)
    if ARM_ON in arms and not args.on_port:
        print(f"--on-port is required to run the {ARM_ON} arm: the window launches its "
              "own aggregator and must not aim it at a port it does not own",
              file=sys.stderr)
        return 2
    if args.on_port and args.on_port < 1:
        print(f"--on-port must be a port number, got {args.on_port}", file=sys.stderr)
        return 2

    def _num(flag: float | None, var: str, default: float) -> float:
        if flag is not None:
            return flag
        try:
            raw = (os.environ.get(var) or "").strip()
            return float(raw) if raw else default
        except ValueError:
            return default

    backend = backend_url(args.backend)
    db = workers_db()
    hold = PersistenceHold(backend, None,
                           drain_wait_s=_num(args.drain_wait, "LLOYD_PERSISTENCE_DRAIN_S",
                                             900.0),
                           drain_poll_s=_num(None, "LLOYD_PERSISTENCE_DRAIN_POLL_S", 5.0),
                           automod_wait_s=_num(args.automod_wait,
                                               "LLOYD_PERSISTENCE_AUTOMOD_WAIT_S", 900.0))
    code = 1
    try:
        code = asyncio.run(run_window(
            args.reps, hold, arms=arms, port=args.on_port, root=root, off_url=off_url,
            ready_wait_s=_num(args.ready_wait, READY_WAIT_ENV, 120.0)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        code = 1
    except Exception as exc:  # noqa: BLE001 — neither check may be skipped
        print(f"window aborted: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        for line in hold.log:
            print(f"  {line}", file=sys.stderr)
        code = 3

    # The leak check runs after the loop: an aggregator that leaked would otherwise have
    # its own failure buried under the one that ended the window, and the row it left in
    # the live store is the finding a reader needs even from a run that failed early.
    report = leak_report(db)
    if report:
        print(report, file=sys.stderr, flush=True)
        code = code or 1
    return settle(code, bool(hold.ours), backend=backend, db=db)


if __name__ == "__main__":
    raise SystemExit(main())
