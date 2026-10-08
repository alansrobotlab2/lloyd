"""One-shot KV cache dtype arm over the gated primary-restart leg (#2404).

`eval/run_kv_dtype_arm.sh` execs this; run that, not this.

WHAT THIS IS FOR. Every primary boot on this box so far has come up `kv_dtype=fp8`: nine
`A/B config:` lines, zero bf16, and no `flash-next-canary.jsonl` either, because the route
that would produce one has never been entered. The engine-output integrity canary can only
answer whether output drifts with the KV cache dtype after something boots the engine on
that dtype and runs the fold against it. This is the window that does that in one attended
firing: hold dispatch, stage the dtype where the boot reads it, ask the gated restart
command to bring the engine up, run the fold once, boot again with nothing staged, and
refuse to report success until the log of *that* boot says the shipped dtype is back and
the engine answers `/health`.

WHAT IT REUSES, and which refusals it adds.
* The hold is `PersistenceHold` (`eval/persistence_arms.py:209`), which is `PoolPause`
  (`eval/run_context_rot_eval.py:494`) hardened for a window: `take()` refuses an
  already-held pool instead of continuing without one, re-reads the pause route rather
  than believing its own POST, refuses if the automod promoter lands a hold mid-take,
  refuses a hold that is not an operator hold, waits for the queue to go quiet, and gives
  back what it took on every refusal path. `settle()` is the same module's post-loop
  backstop, and it is the second half of "never lifts a promoter hold": it refuses to
  resume while `automod` is among the holders. `ArmHold` below adds one thing
  `PersistenceHold` does not have: a check that the fleet is idle BEFORE the take, so the
  window refuses rather than becoming the thing #1751 recorded — four autonomy jobs
  cancelled mid-turn with their leases outliving them.
* The restart is `python -m scripts.automod.round restart --only agent-llm-primary
  --reason <r>`, the form `tests/test_service_control_guard.py` treats as legitimate,
  because that leg owns the automod lease, the host-RAM floor, the drain and the boot
  budget. A `supervisorctl restart` here would be a second owner of the same engine with
  none of those, which is the hazard that put the existing sweep launcher on the command
  guard's refused list. The guard is NOT widened for this file and does not look inside
  it, so none of this window's safety comes from the guard — it comes from refusing to
  touch an engine without a drained hold it took itself.

WHY THE RESTORE IS NOT BEHIND THE CANARY'S RESULT. The fold always exits 0 — a bad verdict
is a verdict, not a failed step — so a non-zero rc means the instrument itself did not
run. Under `&&` that is exactly where a window would stop, leaving the primary serving the
arm dtype with nobody owning it and no record that it was ever armed. So the fold is a
step whose rc is reported, the restore always runs, and the run's own exit is the worst
thing that happened.

Exit: 0 = armed, measured, restored and verified. 2 = the window did not happen, or it did
and something in it failed. A stuck arm is a primary running a KV dtype nobody chose, so a
non-zero exit always says which of the two you are holding, and prints the restore command
when the runner could not complete it.

Environment (every path is printed before it is touched):
  LLOYD_DATA             root for the one-shot env, the service log and the fold. The boot
                         derives the same path from it; absent is a refusal, not a default.
  LLOYD_BACKEND_URL      the backend whose pause routes this holds (default
                         http://127.0.0.1:8080). A test points it at a stub.
  LLOYD_WORKERS_DB       where `settle` reads the operator hold. Default app.paths.
                         WORKERS_DB.
  KV_ARM_ENGINE_HEALTH   the engine's own health URL (default http://127.0.0.1:8096/health).
                         This is the ENGINE port, not the backend's.
  KV_ARM_RESTART_CMD     the seam for the gated leg, for tests only. Empty means "build the
                         real argv", which is what an attended run does.
  ARM_ENV / KV_ARM_LOG / CANARY_LOG / KV_ARM_REPO / KV_ARM_VENV_PYTHON   the paths and the
                         interpreter, each derived from `$LLOYD_DATA` or this checkout.
  KV_ARM_FLEET_WAIT_S / KV_ARM_DRAIN_WAIT_S / KV_ARM_BOOT_WAIT_S /
  KV_ARM_STEP_WAIT_S / KV_ARM_POLL_S   the waits; the drain wait is not the boot wait.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# httpx, not aiohttp, because it is the client the hold this window reuses was
# written against: `PersistenceHold.full_status` calls `r.json()` without awaiting
# it, which is httpx's synchronous accessor and an un-awaited coroutine under
# aiohttp — a wrong client here reads as "the queue is unreadable" and the window
# refuses a fleet it could actually see.
import httpx  # noqa: E402

# The hold machinery this window reuses rather than re-implements, exactly as
# `persistence_arms` reuses the context-rot window's.
from persistence_arms import (  # noqa: E402
    HoldRefused,
    PersistenceHold,
    backend_url,
    in_flight,
    settle,
    workers_db,
)

PRIMARY = "agent-llm-primary"
HOLD_ID = "kv-dtype-arm"

#: What the engine ships with, and what the window must hand back. Pinned to the supervisor
#: conf (`KV_CACHE_DTYPE="fp8"`) rather than to the launcher's `${KV_CACHE_DTYPE:-bf16}`
#: fallback — that fallback is why a *missing* dtype line proves nothing (see `boot_dtype`).
RESTORE_DTYPE = "fp8"

#: The boot looks here, sources the file once, and deletes it
#: (`agent-services/bin/start-qwen38-flash-next.sh:190-197`).
ARM_ENV_NAME = "flash-next-arm.env"
SERVICE_LOG_NAME = "agent-llm-primary.log"
CANARY_STEP_REL = "agent-services/bin/flash-next-canary-step.sh"


class ArmError(Exception):
    """A refusal that stops the window before it touches an engine."""


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _get(url: str, timeout: float = 15.0):
    return urllib.request.urlopen(url, timeout=timeout)


def _data_root() -> Path:
    """`$LLOYD_DATA`, or a refusal. The boot derives the one-shot env's home from the same
    variable, so defaulting it here could stage one tree's file and verify another's log."""
    root = os.environ.get("LLOYD_DATA", "").strip()
    if not root:
        raise ArmError("LLOYD_DATA is not set: the one-shot arm env has no home")
    return Path(root)


def arm_env_path() -> Path:
    return Path(os.environ.get("ARM_ENV") or _data_root() / "logs/services" / ARM_ENV_NAME)


def service_log_path() -> Path:
    return Path(os.environ.get("KV_ARM_LOG")
                or _data_root() / "logs/services" / SERVICE_LOG_NAME)


def canary_log_path() -> Path:
    return Path(os.environ.get("CANARY_LOG")
                or _data_root() / "logs/services" / "flash-next-canary.jsonl")


def repo_root() -> Path:
    """The tree whose `agent-services/bin` this window runs, printed before anything uses
    it: an arm that ran half a checkout's steps is a mislabel."""
    return Path(os.environ.get("KV_ARM_REPO") or REPO)


def venv_python() -> Path:
    return Path(os.environ.get("KV_ARM_VENV_PYTHON") or REPO / ".venvs/lloyd/bin/python")


def engine_health_url() -> str:
    return (os.environ.get("KV_ARM_ENGINE_HEALTH")
            or "http://127.0.0.1:8096/health").rstrip("/")


def restart_cmd(root: Path, reason: str) -> str:
    """The one command this window may use to touch an engine.

    Built here so the argv the run executes and the argv it prints when a restore fails are
    one string: advice that differs from what would have worked is worse than no advice.
    `--only agent-llm-primary` takes its own leg inside the round command — lease, pool
    pause, drain, host-RAM floor, then a long boot budget.
    """
    return (f"{shlex.quote(str(venv_python()))} -m scripts.automod.round restart "
            f"--only {PRIMARY} --reason {shlex.quote(reason)}")


class ArmHold(PersistenceHold):
    """`PersistenceHold` plus the check an arm needs and a rep window did not.

    `take()` already refuses an already-held pool, re-reads the pause route instead of
    believing its own POST, refuses if the promoter holds the pool, refuses a non-operator
    hold, waits for the queue to go quiet, and releases on every refusal path. What it does
    not do is look at the fleet BEFORE the POST, because a rep window can afford to wait
    out a straggler once it holds the pool and an arm cannot: an engine stopped under a
    live job loses that job, and the hold makes the loss invisible to the scheduler."""

    async def wait_fleet_clear(self, wait: float, poll_s: float = 5.0) -> bool:
        """True when nothing is running and the queue is empty, before the hold is taken."""
        deadline = time.monotonic() + wait
        while True:
            status = await self.full_status()
            if status is not None:
                busy = in_flight(status)
                if not busy:
                    self.log.append("fleet clear before the take: nothing in flight")
                    return True
            else:
                # An unreadable queue is not an empty one.
                busy = {"unreadable": 1}
            if time.monotonic() >= deadline:
                self.log.append(f"fleet not clear before the take, still in flight: {busy}")
                return False
            self.log.append(f"fleet busy ({busy}); waiting before taking the hold")
            await asyncio.sleep(poll_s)


def stage_env(path: Path, dtype: str) -> None:
    """Write exactly one line, or refuse if anything is already there.

    The one-shot file is consumed by whichever boot comes first, so an arm staged beside
    this one never reaches an engine — while the record the fold writes would still read as
    though it had. That is a silent mislabel, the one failure a dtype arm cannot afford:
    its entire product is which dtype served. `O_EXCL` rather than a truncate because an
    existing file is somebody else's arm, and refusing keeps it theirs."""
    if path.exists():
        raise ArmError(
            f"one-shot arm env already staged: {path}\n"
            f"  contents: {path.read_text(encoding='utf-8')!r}\n"
            "  Two staged arms is a silent mislabel: the boot consumes one file. Remove "
            "it, or wait for the boot that will.")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"export KV_CACHE_DTYPE={dtype}\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(line)
    log(f"staged one-shot arm env {path}: {line.strip()}")


def log_offset(path: Path) -> int:
    """Bytes in the service log right now; 0 for a log that does not exist yet, which is
    what a fresh box has before the first boot of this window creates it."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def boot_dtype(text: str) -> str | None:
    """The `kv_dtype` of the LAST `A/B config:` line in `text`, or None.

    Anchored at column 0 because the launcher prints that line once per boot
    (`start-qwen38-flash-next.sh:587`) and nothing else writes it. A boot whose text has no
    such line is `None`, deliberately, and not the launcher's documented
    `${KV_CACHE_DTYPE:-bf16}` fallback: a line that fell back is a line whose dtype nobody
    chose, and reading a default as a measurement is exactly how an fp8 pin gets reported as
    bf16."""
    found = None
    for line in text.splitlines():
        if line.startswith("A/B config:"):
            for token in line.split():
                if token.startswith("kv_dtype="):
                    found = token.split("=", 1)[1]
    return found


def wait_boot_dtype(log_path: Path, offset: int, wait: float,
                    poll: float = 5.0) -> str | None:
    """The dtype of the boot that appended after `offset`, polling up to `wait`.

    The offset is what makes this a statement about *this* boot: the live service log can
    hold zero `A/B config:` lines (the rotated ones hold the last nine), so searching the
    whole file would report whatever the last rotation happened to leave."""
    deadline = time.monotonic() + wait
    while True:
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                tail = fh.read()
        except OSError:
            tail = ""
        dtype = boot_dtype(tail)
        if dtype:
            return dtype
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll)


def health_ok(timeout: float = 10.0) -> bool:
    """The engine answers its own health endpoint. A 2xx with no body is an ok: this is
    vllm on :8096, not the backend, and the launcher polls the same URL the same way."""
    try:
        with _get(engine_health_url(), timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


async def run_leg(root: Path, reason: str) -> tuple[bool, str]:
    """One turn of the gated restart leg. Returns (ok, the command for the record)."""
    seam = os.environ.get("KV_ARM_RESTART_CMD")
    argv = (shlex.split(seam) + [reason]) if seam else shlex.split(restart_cmd(root, reason))
    gated = restart_cmd(root, reason)
    log("leg: $ " + " ".join(shlex.quote(a) for a in argv))
    if seam:
        # A seam only ever STANDS IN for the leg, so the per-boot line names the command it
        # is standing in for rather than only the substitute's path. Two boots share one
        # window and one log: unless each boot prints the gated command it took, with its own
        # `--reason`, a reader cannot tell the arm boot from the restore, and the string an
        # operator needs after a half-closed window is the gated one, never the stub's.
        log(f"leg: standing in for the gated command: {gated}")
    try:
        proc = await asyncio.to_thread(
            lambda: subprocess.run(argv, cwd=str(root), capture_output=True, text=True,
                                   timeout=7200))
    except subprocess.TimeoutExpired:
        warn("leg timed out after 7200s; the engine may or may not have come back")
        return False, gated
    except OSError as exc:
        warn(f"leg could not start: {exc}")
        return False, gated
    if proc.returncode != 0:
        warn(f"leg exited {proc.returncode}")
        for line in ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines()[-25:]:
            warn(f"  {line}")
        return False, gated
    return True, gated


async def run_canary_step(root: Path, label: str, wait: float) -> tuple[int, bool]:
    """The fold, as the one executable both routes call. Returns (rc, ran).

    `shutil.which` is not consulted: the call names `bash`, because this repo tracks the
    step 755 but the arm route 644, and a mode bit is not a statement about the code."""
    step = root / CANARY_STEP_REL
    if not step.is_file():
        warn(f"canary step missing: {step}")
        return 127, False
    argv = ["bash", str(step), label]
    # `CANARY_LOG` is exported so the fold and this window name one file: the step derives
    # the same path from `$LLOYD_DATA` on its own, and the window counts the lines in it, so
    # a disagreement would read as "no verdict was written". The fold still has exactly one
    # writer — this only tells it which file the fold already chose.
    log("canary: $ " + " ".join(shlex.quote(a) for a in argv))
    try:
        proc = await asyncio.to_thread(
            lambda: subprocess.run(argv, cwd=str(root), timeout=wait,
                                   env={**os.environ,
                                        "CANARY_LOG": str(canary_log_path())}))
    except subprocess.TimeoutExpired:
        warn(f"canary step timed out after {wait}s")
        return 124, True
    except OSError as exc:
        warn(f"canary step could not start: {exc}")
        return 126, False
    return proc.returncode, True


def canary_records(path: Path) -> list[dict]:
    """The verdict lines in the fold's log, each with its byte span, so a run can tell a
    line it appended from one that was already there."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"unparseable": line})
    return out


async def run_window(cfg: dict[str, Any]) -> tuple[int, bool]:
    """The whole window. Returns (exit code, whether this process ever held the pool)."""
    root, env_file = cfg["repo"], cfg["env_file"]
    log_path, canary_file = cfg["log"], cfg["canary"]
    async with httpx.AsyncClient() as client:
        hold = ArmHold(cfg["backend"], client, drain_wait_s=cfg["drain_wait_s"],
                       drain_poll_s=cfg["poll_s"])
        # One `finally` for the hold, not one per branch: every return below is a path out
        # of the window, and a hold that survives only because the branch that added the
        # return forgot its own release is the leak this file exists not to have.
        # `PersistenceHold.release` is `PoolPause.__aexit__`, which no-ops once the hold is
        # gone, so the call is safe on a refusal that took nothing.
        try:
            # Before the take: `PoolPause` cancels running jobs to get an idle engine, and
            # an arm has no reason on earth to spend somebody's job to make a measurement.
            if not await hold.wait_fleet_clear(cfg["fleet_wait_s"], cfg["poll_s"]):
                for line in hold.log[-4:]:
                    warn(f"  {line}")
                warn("refused: the fleet is not clear, and this window does not cancel a "
                     "running job to get the engine.")
                return 2, hold.ours
            try:
                await hold.take()
            except HoldRefused as exc:
                warn(f"refused: no usable dispatch hold — {exc}")
                for line in hold.log[-4:]:
                    warn(f"  {line}")
                return 2, hold.ours
            log("hold taken and queue drained: this window owns dispatch and the primary")

            step_rc: int | None = None
            try:
                stage_env(env_file, cfg["dtype"])

                # The offset is taken BEFORE the leg that boots, not after it. The gated leg
                # waits for the new primary to answer /health, so by the time it returns the
                # boot has already written its A/B line; an offset read afterwards has already
                # stepped over the only line this check exists to read, and the window reports
                # "the boot logged nothing" about a boot that logged exactly what was asked.
                offset = log_offset(log_path)
                ok, _ = await run_leg(root, f"kv-{cfg['dtype']}-arm")
                if not ok:
                    staged = env_file.exists()
                    warn("the arm boot failed. "
                         + ("The staged file is STILL THERE, so the next boot comes up on "
                            f"{cfg['dtype']} unless you remove it: {env_file}" if staged
                            else "Nothing is staged, so the next boot is the shipped dtype.")
                         + f"\n  restore by hand: "
                           f"{restart_cmd(root, f'kv-{RESTORE_DTYPE}-restore')}")
                    return 2, hold.ours

                booted = await asyncio.to_thread(
                    wait_boot_dtype, log_path, offset, cfg["boot_wait_s"], cfg["poll_s"])
                if booted != cfg["dtype"]:
                    warn(f"the boot after staging did not report kv_dtype={cfg['dtype']} "
                         f"(it logged {booted!r}). The arm did not happen, so the canary is "
                         "not run against an engine whose dtype is unknown.")
                else:
                    log(f"arm is serving: kv_dtype={booted}")
                    # Counted BEFORE the fold runs, because the count is how this window
                    # proves the line it is about to read is its own and not an arm's from
                    # last week. Taken after the step, the count already includes the line it
                    # is looking for, and the comparison reports "nothing was appended" about
                    # a verdict the window itself just wrote.
                    before = len(canary_records(canary_file))
                    step_rc, ran = await run_canary_step(root, f"kv-{cfg['dtype']}",
                                                         cfg["step_wait_s"])
                    if not ran:
                        warn("the canary step did not run at all")
                    elif step_rc != 0:
                        warn(f"the canary step exited {step_rc}; its verdict is not on disk. "
                             "Restoring the primary anyway — an arm is not lost to its "
                             "instrument, and a missing verdict is a visible gap while an "
                             "armed engine is not.")

                # The restore is never behind the fold's result: see the module docstring.
                restore_offset = log_offset(log_path)      # before the restore, as above
                ok_restore, restore = await run_leg(root, f"kv-{RESTORE_DTYPE}-restore")
                if not ok_restore:
                    warn("THE WINDOW DID NOT CLOSE. The primary may still be serving the "
                         f"arm dtype ({cfg['dtype']}). Run the restore by hand, NOW:")
                    warn(f"  {restore}")
                    warn(f"  then confirm: grep -a '^A/B config:' {log_path} | tail -1")
                    return 2, hold.ours

                restored = await asyncio.to_thread(
                    wait_boot_dtype, log_path, restore_offset, cfg["boot_wait_s"],
                    cfg["poll_s"])
                if restored != RESTORE_DTYPE:
                    warn(f"the boot after the restore logged kv_dtype={restored!r}, not "
                         f"{RESTORE_DTYPE}. The engine is still on the arm dtype or on an "
                         "unreadable default: this is not a closed window.")
                    return 2, hold.ours
                if not await asyncio.to_thread(health_ok):
                    warn(f"the restored boot logged kv_dtype={restored}, but "
                         f"{engine_health_url()} does not answer ok. Not reporting a closed "
                         "window on an engine nobody can reach.")
                    return 2, hold.ours
                log(f"window closed: kv_dtype={restored} and {engine_health_url()} ok")

                if len(canary_records(canary_file)) <= before:
                    warn(f"no verdict line was appended to {canary_file} by this window. "
                         "That is the blind state #1625 was opened against, so it is not a "
                         f"success: the canary step exited {step_rc}.")
                    return 2, hold.ours
                log(f"canary: {len(canary_records(canary_file)) - before} verdict line(s) "
                    f"appended to {canary_file}")

                if step_rc:
                    warn(f"the window completed but the canary step exited {step_rc}")
                    return 2, hold.ours
                return 0, hold.ours
            except ArmError as exc:
                warn(f"refused: {exc}")
                return 2, hold.ours
            finally:
                for line in hold.log[-8:]:
                    log(f"  {line}")
                # `PersistenceHold.release` is `PoolPause.__aexit__`: it resumes only a hold
                # this process set, and on an operator hold it resumes with a plain POST —
                # which `settle` has already refused to do if the promoter is a holder.
                await hold.release()
        except Exception as exc:  # noqa: BLE001 — a window that dies loudly is fine; one
            # that dies holding the pool is not, and `settle` is the backstop for that.
            warn(f"window failed: {type(exc).__name__}: {exc}")
            for line in hold.log[-4:]:
                warn(f"  {line}")
            return 2, hold.ours


def config(dtype: str) -> dict[str, Any]:
    return {
        "dtype": dtype,
        "repo": repo_root(),
        "env_file": arm_env_path(),
        "log": service_log_path(),
        "canary": canary_log_path(),
        "backend": backend_url(),
        "fleet_wait_s": float(os.environ.get("KV_ARM_FLEET_WAIT_S", "0")),
        "drain_wait_s": float(os.environ.get("KV_ARM_DRAIN_WAIT_S", "900")),
        "boot_wait_s": float(os.environ.get("KV_ARM_BOOT_WAIT_S", "1500")),
        "step_wait_s": float(os.environ.get("KV_ARM_STEP_WAIT_S", "600")),
        "poll_s": float(os.environ.get("KV_ARM_POLL_S", "10")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kv_dtype_arm",
        description="One-shot KV cache dtype arm over the gated primary restart.")
    ap.add_argument("dtype", help=f"KV cache dtype to boot once, e.g. bf16 (the restore is "
                                  f"always {RESTORE_DTYPE})")
    args = ap.parse_args(argv)
    dtype = args.dtype.strip()
    if not dtype or not dtype.replace("-", "").replace(".", "").isalnum():
        warn(f"refused: {dtype!r} is not a kv-cache-dtype")
        return 2

    try:
        cfg = config(dtype)
    except ArmError as exc:
        warn(f"refused: {exc}")
        return 2

    log(f"window: repo={cfg['repo']} backend={cfg['backend']}")
    log(f"window: one-shot env={cfg['env_file']}")
    log(f"window: service log={cfg['log']}")
    log(f"window: canary step={cfg['repo'] / CANARY_STEP_REL} log={cfg['canary']}")
    log(f"window: engine health={engine_health_url()}")
    log(f"window: arm dtype={dtype}, restore dtype={RESTORE_DTYPE}, "
        f"gated leg={restart_cmd(cfg['repo'], f'kv-{dtype}-arm')}")
    log(f"window: waits fleet={cfg['fleet_wait_s']}s drain={cfg['drain_wait_s']}s "
        f"boot={cfg['boot_wait_s']}s step={cfg['step_wait_s']}s poll={cfg['poll_s']}s")

    try:
        code, ours = asyncio.run(run_window(cfg))
    except KeyboardInterrupt:
        warn("window interrupted: resuming dispatch")
        code, ours = 130, True
    # The same backstop the persistence window uses: it only ever touches a hold this
    # process took, and it will not lift one while the promoter is a holder.
    return settle(code, ours, backend=cfg["backend"], db=workers_db())


if __name__ == "__main__":
    sys.exit(main())
