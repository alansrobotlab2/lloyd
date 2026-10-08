#!/usr/bin/env python3
"""Open the persistence-arm window with a dispatch hold that cannot be leaked (#2397).

The window is `eval/run_persistence_arms.sh N`, and the measurement it protects is the
one #2194 is still owed: N reps of the three persistence arms with worker dispatch
HELD, so a row's survival can be attributed to the summariser rather than to a job
that happened to run beside it. Every one of those rows is worthless if the hold is
still set when the runner finishes, because the next window — and the deploy gate the
leak rate feeds — inherits a paused queue.

Why this is a driver and not a shell script with a `trap`: the hold/resume it needs
already exists, hardened, as `eval/run_context_rot_eval.py::PoolPause` — it resumes
only a pause it took, runs the resume in a `finally` behind SIGTERM/SIGHUP handlers,
and refuses to lift a hold the automod promoter still owns (an operator resume lifts
BOTH holders — `workers/pool.py:716-728` — so a hand-rolled `curl -d '{"paused":false}'`
in an EXIT trap un-pauses a landing's drain mid-flight). This module subclasses it and
adds the two refusals a measurement window needs on top:

* **an already-held pool is a refusal.** `PoolPause.__aenter__` continues with
  `ours=False` when someone else holds the pool — right for an eval that only wants
  the engine to itself, wrong for a window whose denominator is "reps with dispatch
  held", where continuing means measuring something the label on the rows denies.
* **nothing is in flight before rep 1.** No pause route answers that:
  `GET /api/workers/pause` returns `paused`, `paused_by` and `paused_since`
  (`app/routers/workers.py:406-433`) and says nothing about the queue, so depth comes
  from `GET /api/workers/status` (`:124`).

And the leak check runs OUTSIDE the event loop, in `main`'s `finally`: a rep loop that
dies mid-cancel can take the async resume down with it, and the persisted row in
`workers.db` is the only witness that says whether dispatch came back.

    eval/run_persistence_arms.sh 10
    LLOYD_CANARY_ROWS=/tmp/w/rows.jsonl LLOYD_CANARY_REPORT=/tmp/w/run.md \\
      eval/run_persistence_arms.sh 2

Environment: `LLOYD_BACKEND_URL` (default `http://127.0.0.1:8080`) is the backend
whose pause routes this takes — a test points it at a stub; `LLOYD_CANARY_CMD`
replaces the bench command for the same reason; `LLOYD_WORKERS_DB` points the final
watermark read at another store, defaulting to `app.paths.WORKERS_DB`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LLOYD_HOME = Path(__file__).resolve().parent.parent
for _p in (str(LLOYD_HOME), str(LLOYD_HOME / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The hold machinery this window reuses rather than re-implements. `PoolPause` is the
#: version that has been running under real landings since #1550.
from run_context_rot_eval import PoolPause  # noqa: E402

BACKEND_ENV = "LLOYD_BACKEND_URL"
DEFAULT_BACKEND = "http://127.0.0.1:8080"
CLI_ENV = "LLOYD_CANARY_CMD"
WORKERS_DB_ENV = "LLOYD_WORKERS_DB"
DRAIN_ENV = "LLOYD_PERSISTENCE_DRAIN_S"
DRAIN_POLL_ENV = "LLOYD_PERSISTENCE_DRAIN_POLL_S"
AUTOMOD_WAIT_ENV = "LLOYD_PERSISTENCE_AUTOMOD_WAIT_S"

#: The three arms #2041 ran on 2026-10-04, in that order: two attacks and the benign
#: control. Spelled once, because the rows and the report are only comparable to that
#: window if the selection is the same one.
ARMS = ("persistence-web-digest", "persistence-relay-email", "persistence-control-handover")

#: The persisted operator hold's (source, key) pair — imported from the module that
#: WRITES it, not restated here. `tests/test_workers_pause_pair_stays_one_source.py`
#: keeps that import real. Two columns, not one: a checker that looks for a single
#: `key` equal to `_pool/operator_paused` reads no row and concludes "no hold" whether
#: or not the pool is paused.
from workers.pool import PAUSE_WM_KEY, PAUSE_WM_SOURCE  # noqa: E402

#: Queue states that mean a job has left the queue and is inside a worker
#: (`workers/queue.py:679` `state='claimed'`, `:711` `state='running'`). A `queued`
#: row is WAITING, which is precisely the state a held dispatch produces on purpose:
#: counting it as in flight would make this window impossible to open.
IN_FLIGHT_STATES = ("claimed", "running")


class HoldRefused(Exception):
    """The window may not open — and nothing was taken that this process must return."""


class RepFailed(Exception):
    """A shipped-CLI invocation exited non-zero, so the window's rows are incomplete."""


def backend_url(cli_value: str | None = None) -> str:
    """The backend whose pause routes this runner drives, from `LLOYD_BACKEND_URL`."""
    return (cli_value or os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND).rstrip("/")


def canary_cmd() -> list[str]:
    """The bench command this window drives, as argv.

    Default is the shipped CLI; `LLOYD_CANARY_CMD` is the seam that lets a test drive
    the same rep loop against a stub, because the real bench needs the aggregator, the
    engine and thirteen minutes, none of which a gate may spend.
    """
    raw = (os.environ.get(CLI_ENV) or "").strip()
    if raw:
        return shlex.split(raw)
    return [sys.executable, str(LLOYD_HOME / "eval" / "run_injection_canary.py")]


def workers_db() -> Path:
    """The store the persisted hold lives in.

    `LLOYD_WORKERS_DB` for a stub; otherwise `app.paths.WORKERS_DB`, imported lazily
    and with NO fallback of its own — a `${LLOYD_DATA:-~/lloyd-data}` guess is the
    second-data-root bug `app/data_root.py` exists to prevent, and a guessed root here
    would read someone else's watermarks and certify a live hold as cleared.
    """
    raw = (os.environ.get(WORKERS_DB_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser()
    from app.paths import WORKERS_DB
    return Path(WORKERS_DB)


def operator_pause_set(db: Path) -> bool | None:
    """The persisted operator hold as the store records it: True, False, or None.

    Read through the queue's OWN accessor (`WorkQueue.wm_get`, `workers/queue.py:1402`),
    not a hand-rolled `sqlite3` query: `tests/test_counterfactual_eval.py::
    test_no_script_under_eval_opens_the_knowledge_store_file` is the standing rule that
    a script under `eval/` reaches a store through its class and never raw, and the
    same property is what makes this read survive a schema change instead of silently
    returning no row. A missing file is `None`, not False — the runner is created before
    the store is, and an absent witness is a gap in coverage, not a clean bill.
    """
    if not db.is_file():
        return None
    try:
        from workers.queue import WorkQueue
        return WorkQueue(db).wm_get(PAUSE_WM_SOURCE, PAUSE_WM_KEY) == "1"
    except Exception:      # noqa: BLE001 — unreadable is not "not held"
        return None


def _get_json(url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — the caller decides how loud to be
        return None
    return body if isinstance(body, dict) else None


def _post_json(url: str, payload: dict) -> dict | None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return body if isinstance(body, dict) else None


def sync_holders(backend: str) -> list[str] | None:
    """`paused_by` off the read-only pause route, no event loop required."""
    st = _get_json(f"{backend}/api/workers/pause")
    if st is None:
        return None
    holders = st.get("paused_by")
    if isinstance(holders, str):
        return [holders]
    return list(holders or [])


def sync_resume(backend: str) -> None:
    """An operator resume — the same POST `PoolPause._post(False)` makes."""
    _post_json(f"{backend}/api/workers/pause", {"paused": False})


def in_flight(status: dict[str, Any]) -> dict[str, int]:
    """What the status route says is mid-flight, empty when the pool is idle.

    Both halves matter: `pool.in_flight_count` is a job inside a worker right now,
    and a queue row in `claimed`/`running` is one the pool has taken and may not have
    published its exit for yet.
    """
    out: dict[str, int] = {}
    pool = status.get("pool") or {}
    count = int(pool.get("in_flight_count") or 0)
    if count:
        out["in_flight"] = count
    for source, states in (status.get("depth") or {}).items():
        for state, n in (states or {}).items():
            if state in IN_FLIGHT_STATES and int(n or 0) > 0:
                out[f"{source}:{state}"] = int(n)
    return out


class PersistenceHold(PoolPause):
    """`PoolPause` plus the refusals that keep a measurement window honest."""

    def __init__(self, backend: str, client, *, drain_wait_s: float = 900.0,
                 drain_poll_s: float = 5.0, **kw):
        super().__init__(backend, client, **kw)
        self.drain_wait_s = drain_wait_s
        self.drain_poll_s = drain_poll_s

    async def full_status(self) -> dict[str, Any] | None:
        """The whole `/api/workers/status` body — `status()` returns only its `pool`."""
        try:
            r = await self.client.get(f"{self.backend}/api/workers/status", timeout=10)
            body = r.json()
        except Exception:  # noqa: BLE001
            return None
        return body if isinstance(body, dict) else None

    async def pause_state(self) -> dict[str, Any]:
        """`GET /api/workers/pause` — the read-only hold route, taken nowhere."""
        r = await self.client.get(f"{self.backend}/api/workers/pause", timeout=10)
        r.raise_for_status()
        body = r.json()
        return body if isinstance(body, dict) else {}

    async def take(self) -> None:
        """Hold dispatch, and prove the hold is ours and the queue is quiet.

        Ordered so that every path out either took nothing or has already given back
        what it took: the two refusals fire before the POST, and a refusal raised after
        it releases first.
        """
        pool = await self.status()
        if pool is None or "paused" not in pool:
            raise HoldRefused("cannot read /api/workers/status — refusing to open a "
                              "window whose hold it cannot verify")
        if pool.get("paused"):
            raise HoldRefused(f"dispatch is already held by "
                              f"+{'+'.join(pool.get('paused_by') or ['unknown'])}; this "
                              "window measures reps with the hold OFF nobody else's, so "
                              "it neither takes nor lifts a hold it did not take")
        try:
            await self.__aenter__()      # PoolPause: POST, verify, set `ours`
        except SystemExit as exc:        # its own "the take did not land" refusal
            raise HoldRefused(str(exc)) from None
        try:
            got = await self.pause_state()
            holders = got.get("paused_by") or []
            if not got.get("paused"):
                raise HoldRefused(f"the hold did not land: GET /api/workers/pause "
                                  f"reports {got}")
            if "automod" in holders:
                raise HoldRefused("the automod promoter took a hold during the take; a "
                                  "window run now would straddle a landing's restart")
            if "operator" not in holders:
                raise HoldRefused(f"the hold that landed is not an operator hold: "
                                  f"{holders}")
            await self.wait_idle()
        except BaseException:
            await self.release()          # never refuse and hold
            raise

    async def wait_idle(self) -> None:
        """Wait for the queue to quiet down, then refuse if it never did."""
        deadline = time.monotonic() + self.drain_wait_s
        while True:
            status = await self.full_status()
            if status is None:
                raise HoldRefused("cannot read /api/workers/status to check the queue "
                                  "depth — refusing to start reps blind")
            busy = in_flight(status)
            if not busy:
                self.log.append("queue idle: nothing in flight")
                return
            if time.monotonic() >= deadline:
                raise HoldRefused(f"queue still busy after {self.drain_wait_s:.0f}s: "
                                  f"{busy} — a rep run now would measure those jobs too")
            self.log.append(f"waiting for {busy} to finish")
            await asyncio.sleep(self.drain_poll_s)

    async def release(self) -> None:
        """Resume dispatch if it is ours to resume. Never raises."""
        await self.__aexit__(None, None, None)


async def run_cli(cmd: list[str], args: list[str]) -> None:
    """One shipped-CLI invocation. A non-zero exit is a failed window, not a warning.

    A cancellation here KILLS the child before it propagates. The window is held; the
    interrupt path resumes dispatch on the way out; a rep left running after that would
    keep measuring episodes with dispatch unheld — which is the leak this runner exists to
    prevent, arriving from the other side. Killing is best-effort and never masks the
    original cancellation.
    """
    argv = [*cmd, *args]
    print("$ " + " ".join(shlex.quote(a) for a in argv), flush=True)
    proc = await asyncio.create_subprocess_exec(*argv)
    try:
        rc = await proc.wait()
    except BaseException:
        try:
            proc.kill()
        except ProcessLookupError:      # already gone; nothing to stop
            pass
        raise
    if rc != 0:
        raise RepFailed(f"{' '.join(args)} exited {rc}")


async def run_window(n: int, hold: PersistenceHold) -> int:
    """The reps and the grades, under a hold `main` built and will settle on.

    The hold comes in rather than being made here because `main`'s `finally` has to be
    able to ask it whether the pool is its to resume: a run interrupted before
    `take()` ever ran must not reach `settle` claiming a hold it never took, which is
    how a runner ends up lifting somebody else's pause.
    """
    import httpx

    cmd = canary_cmd()
    async with httpx.AsyncClient() as client:
        hold.client = client
        try:
            await hold.take()
        except HoldRefused as exc:
            print(f"refused: {exc}", file=sys.stderr, flush=True)
            for line in hold.log:
                print(f"  {line}", file=sys.stderr)
            return 2
        # SIGTERM/SIGHUP/INT cancel the task so the `finally` still resumes — the same
        # shape `eval/run_context_rot_eval.py:1085-1090` uses for the same reason.
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, task.cancel)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        code = 0
        try:
            for rep in range(1, n + 1):
                await run_cli(cmd, ["run", "--only", *ARMS, "--rep", str(rep)])
            await run_cli(cmd, ["grade"])
            if os.environ.get("LLOYD_CANARY_REPORT", "").strip():
                # A report is written only when asked for, and the ask is that variable:
                # the default `report` invocation prints and leaves the tree alone.
                await run_cli(cmd, ["report", "--write"])
        except RepFailed as exc:
            print(f"window failed: {exc}", file=sys.stderr, flush=True)
            code = 1
        except asyncio.CancelledError:
            print("window interrupted: resuming dispatch", file=sys.stderr, flush=True)
            code = 1
        finally:
            await hold.release()
            for line in hold.log:
                print(f"  {line}", file=sys.stderr)
        return code


def settle(code: int, ours: bool, *, backend: str, db: Path) -> int:
    """The backstop that runs after the event loop is gone (#2397 clause 4).

    Only ever touches a hold this process took: a refusal path that never held the pool
    must not resume someone else's pause on its way out, and a hold the automod promoter
    owns is not ours to lift either — the same rule `PoolPause.__aexit__` applies, which
    is why a `trap`-based runner could not be trusted with this line.
    """
    if not ours:
        return code
    held = operator_pause_set(db)
    if held is False:
        return code
    if held is None:
        print(f"!! cannot read the operator hold in {db}: the window cannot certify "
              "that dispatch came back", file=sys.stderr, flush=True)
        return code or 1
    holders = sync_holders(backend)
    if holders is not None and "automod" in holders:
        print("!! dispatch is STILL held and the automod promoter is among the holders:"
              " NOT lifting it — an operator resume would lift its landing's drain too."
              " Resume by hand once the landing is done.", file=sys.stderr, flush=True)
        return code or 1
    print("!! dispatch is STILL held after the window closed; attempting the "
          "last-resort resume", file=sys.stderr, flush=True)
    sync_resume(backend)
    if operator_pause_set(db) is not False:
        print(f"!! LEAKED: {db} still records ({PAUSE_WM_SOURCE},{PAUSE_WM_KEY})='1'. "
              "Dispatch is paused; nothing else on this box will resume it.",
              file=sys.stderr, flush=True)
    else:
        print("!! the last-resort resume cleared it, but the async resume failed and "
              "that is the finding", file=sys.stderr, flush=True)
    return code or 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reps", type=int, help="number of reps (1..N) of the three arms")
    ap.add_argument("--backend", default=None,
                    help=f"backend URL (default: ${BACKEND_ENV} or {DEFAULT_BACKEND})")
    ap.add_argument("--drain-poll", type=float, default=None,
                    help=f"seconds between depth reads while draining (default: "
                         f"${DRAIN_POLL_ENV} or 5)")
    ap.add_argument("--drain-wait", type=float, default=None,
                    help=f"seconds to wait for the queue to idle (default: ${DRAIN_ENV}"
                         " or 900)")
    ap.add_argument("--automod-wait", type=float, default=None,
                    help=f"seconds to wait out an automod promoter hold at exit "
                         f"(default: ${AUTOMOD_WAIT_ENV} or 900)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reps < 1:
        print(f"reps must be >= 1, got {args.reps}", file=sys.stderr)
        return 2
    backend = backend_url(args.backend)
    db = workers_db()
    env = os.environ.get

    def _num(flag: float | None, var: str, default: float) -> float:
        if flag is not None:
            return flag
        raw = (env(var) or "").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    # Built here, before the loop, so `settle` can ask it afterwards whether the pool is
    # THIS process's to resume. The client arrives inside `run_window`, which is the only
    # place a loop exists to make one.
    hold = PersistenceHold(backend, None,
                           drain_wait_s=_num(args.drain_wait, DRAIN_ENV, 900.0),
                           drain_poll_s=_num(args.drain_poll, DRAIN_POLL_ENV, 5.0),
                           automod_wait_s=_num(args.automod_wait, AUTOMOD_WAIT_ENV, 900.0))
    code = 1
    try:
        code = asyncio.run(run_window(args.reps, hold))
    except (KeyboardInterrupt, asyncio.CancelledError):
        code = 1
    except Exception as exc:      # noqa: BLE001 — never leave a hold without the check
        # Anything that escapes the window still has to reach `settle`: the exit code is
        # the least of it if the pool is left paused.
        print(f"window aborted: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        for line in hold.log:
            print(f"  {line}", file=sys.stderr)
        code = 3
    return settle(code, bool(hold.ours), backend=backend, db=db)


if __name__ == "__main__":
    raise SystemExit(main())
