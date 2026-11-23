"""Behaviour fitness: did Lloyd's *behaviour* move, outside its own noise?

Every scorecard row measures the loop's process, and the regression check
measures retrieval. Nothing said whether the agent got better or worse at the
things it does: picking a tool, ignoring a planted instruction, acting on a
remembered fact, taking a refusal for an answer, matching a skill. This module
runs the benches that already measure those (`eval/fitness/manifest.yaml`; the
inventory they were picked from is architecture/testing.md "Behaviour benches")
and appends what they said to ONE series.

It is not a second index and not a second store of composites: each run appends
one `bench` row per bench to `app.paths.FITNESS_SERIES` and one closing `summary`
row, and the daily instrument index (#2597) reads that summary row through its
`fitness_*` arms like any other instrument. This module never imports or reads
the index. Report-only: scorecard row 22 and `report` read the series; no gate,
promotion rule or rollback path does.

    python -m scripts.automod.fitness run --once [--only NAME] [--repeat N] [--force]
    python -m scripts.automod.fitness noise --repeat 5 [--only NAME] [--force]
    python -m scripts.automod.fitness report [--json]

Rules, each one a way this could have been quietly wrong:

- **One bench at a time, behind `regression.lock`.** The benches share the
  primary with each other and with the regression check; a bench that ran beside
  either would measure the contention. The lock is the regression runner's own
  (`workers/sources/automod_regression.py::REGRESSION_LOCK`), so a fitness run
  and a regression measurement can never overlap in either direction.
- **The pool must be paused.** `run` and `noise` refuse to start an engine bench
  unless `GET /api/workers/status` says the pool is paused with nothing in
  flight — a momentary idle reading is not idle, and a bench request pre-empts a
  live worker turn. `--force` overrides and is stamped on every row it produced.
  `--pool-job` is the nightly source's mode: the job itself is one of the pool's
  in-flight rows, so it waits before each bench for the pool to hold nothing
  else, and records a bench it could not get a quiet engine for as `skipped`.
- **A bench that did not measure has no value.** Timed out, crashed, wrote no
  artifact, or wrote one with nothing at the declared pointer: the row carries a
  status and `value: null`. It is never a zero, and no delta is taken from it.
- **A delta inside the noise floor is not a move.** `noise` runs each bench N
  times at one commit and records min/max/stdev; `moves` reports a bench only
  when its 7-day delta is larger than that measured spread AND the latest row's
  Wilson interval does not cover the old value. A bench with no measured floor,
  or one flagged `too-noisy`, is excluded and says so.
- **A regression is recorded, never acted on.** When `moves` reports a bench
  that got worse, `record_moves` appends one `fitness_move` ledger event naming
  the bench, the delta and the landings between the two measured commits. No
  bisect, no rollback request: every rollback this loop has performed was a
  false positive, and this is one more instrument, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 1
MANIFEST_PATH = ROOT / "eval" / "fitness" / "manifest.yaml"
POOL_STATUS_URL = "http://127.0.0.1:8080/api/workers/status"
HEALTH_URL = "http://127.0.0.1:8080/health"

#: The ledger event `record_moves` appends.
MOVE_EVENT = "fitness_move"
#: How far back the delta looks: the newest row at least this much older than the
#: latest one is the reference.
DELTA_DAYS = 7.0
#: A summary row carries forward a bench's newest value only this long, so an
#: `--only` run does not blank the other arms, and a bench that stopped being
#: measured still goes `no-value` in the index within a day and a half.
SUMMARY_CARRY_HOURS = 36.0
#: Fewer repeats than this is not a spread.
MIN_NOISE_REPEATS = 3
#: `--pool-job`: how many in-flight rows are this job's own.
OWN_INFLIGHT = 1

EXIT_OK, EXIT_SOME_FAILED, EXIT_REFUSED, EXIT_LOCK_HELD = 0, 1, 2, 3

#: Row statuses. Only `measured` carries a value.
MEASURED = "measured"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DAYS_AGO_RE = re.compile(r"\{days_ago:(\d+)\}")


class ManifestError(ValueError):
    """The manifest names something this runner will not run."""


@dataclass(frozen=True)
class Bench:
    name: str
    covers: str
    commands: tuple[tuple[str, ...], ...]
    read: Mapping[str, Any]
    higher_is_better: bool
    timeout_s: float
    engine: bool
    safety: str
    cadence_days: int = 1
    engine_minutes: float = 0.0
    noise_max_spread: float = 0.1
    env: Mapping[str, str] = field(default_factory=dict)
    ok_exit_codes: tuple[int, ...] = (0,)
    #: False when the value is a mean of fractions rather than k of n: the Wilson
    #: interval is then an approximation and the row says so.
    binomial: bool = True


# ── the manifest ──────────────────────────────────────────────────────────────

def load_manifest(path: Path | None = None) -> list[Bench]:
    """The benches, validated. Refuses rather than runs anything it cannot name."""
    import yaml

    target = Path(path or MANIFEST_PATH)
    doc = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    raw = doc.get("benches")
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{target} declares no benches")
    out: list[Bench] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ManifestError(f"{target}: a bench entry is not a mapping")
        name = str(entry.get("name") or "")
        if not _NAME_RE.match(name):
            raise ManifestError(f"{target}: bench name {name!r} must match "
                                f"{_NAME_RE.pattern} (it is a key in the summary row)")
        if name in seen:
            raise ManifestError(f"{target}: bench {name!r} is declared twice")
        seen.add(name)
        commands = entry.get("commands")
        if (not isinstance(commands, list) or not commands
                or not all(isinstance(c, list) and c and all(isinstance(a, str) for a in c)
                           for c in commands)):
            raise ManifestError(f"{name}: `commands` must be a list of argv lists")
        read = entry.get("read")
        if not isinstance(read, dict) or read.get("from") not in ("file", "stdout"):
            raise ManifestError(f"{name}: `read.from` must be `file` or `stdout`")
        if not read.get("value") or not read.get("n"):
            raise ManifestError(f"{name}: `read` must name both `value` and `n` — a "
                                f"share with no denominator is not recorded")
        if read["from"] == "file" and not read.get("path"):
            raise ManifestError(f"{name}: `read.from: file` needs `read.path`")
        env = entry.get("env") or {}
        if not isinstance(env, dict):
            raise ManifestError(f"{name}: `env` must be a mapping")
        if "LLOYD_DATA" in env:
            raise ManifestError(f"{name}: a bench may not set LLOYD_DATA — a child "
                                f"that inherits it writes into whichever root it names")
        if not str(entry.get("safety") or "").strip():
            raise ManifestError(f"{name}: `safety` must say why this bench cannot "
                                f"change the machine (sandboxed id, or no live tools)")
        for key in ("higher_is_better", "engine"):
            if not isinstance(entry.get(key), bool):
                raise ManifestError(f"{name}: `{key}` must be true or false")
        timeout = entry.get("timeout_s")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ManifestError(f"{name}: `timeout_s` must be a positive number")
        out.append(Bench(
            name=name, covers=str(entry.get("covers") or ""),
            commands=tuple(tuple(c) for c in commands), read=dict(read),
            higher_is_better=entry["higher_is_better"], timeout_s=float(timeout),
            engine=entry["engine"], safety=str(entry["safety"]),
            cadence_days=int(entry.get("cadence_days", 1)),
            engine_minutes=float(entry.get("engine_minutes", 0.0)),
            noise_max_spread=float(entry.get("noise_max_spread", 0.1)),
            env={str(k): str(v) for k, v in env.items()},
            ok_exit_codes=tuple(int(c) for c in entry.get("ok_exit_codes", (0,))),
            binomial=bool(entry.get("binomial", True))))
    budget = doc.get("budget_engine_minutes")
    if isinstance(budget, (int, float)):
        total = sum(b.engine_minutes for b in out)
        if total > budget:
            raise ManifestError(f"{target}: the benches' engine_minutes add up to "
                                f"{total:g}, over the declared budget of {budget:g}")
    return out


def select(benches: Sequence[Bench], only: Sequence[str] | None) -> list[Bench]:
    if not only:
        return list(benches)
    known = {b.name for b in benches}
    unknown = sorted(set(only) - known)
    if unknown:
        raise ManifestError(f"no bench named {unknown}; the manifest has {sorted(known)}")
    return [b for b in benches if b.name in set(only)]


# ── arithmetic ────────────────────────────────────────────────────────────────

def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for k of n, or None when n is not positive."""
    if n <= 0:
        return None
    p = min(1.0, max(0.0, k / n))
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6))


def dig(doc: Any, pointer: str) -> Any:
    """Follow `a.b.c` through nested mappings; any miss is None."""
    cur = doc
    for seg in str(pointer).split("."):
        if not isinstance(cur, Mapping) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


def _number(raw: Any) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return None if math.isnan(float(raw)) else float(raw)


def read_value(bench: Bench, doc: Any) -> dict[str, Any]:
    """`{status, value, n, k, ci_low, ci_high, ci_kind, detail}` from one artifact.

    `value` is the declared pointer, or k/n when the artifact carries only the
    counts. A value outside 0..1, a missing n, or an n of zero is `no-value`:
    a share over nothing is not a zero.
    """
    spec = bench.read
    none = {"value": None, "n": None, "k": None, "ci_low": None, "ci_high": None,
            "ci_kind": None}
    n_raw = _number(dig(doc, spec["n"]))
    n = int(n_raw) if n_raw is not None else None
    k = _number(dig(doc, spec["k"])) if spec.get("k") else None
    value = _number(dig(doc, spec["value"]))
    if value is None and k is not None and n:
        value = k / n
    if not n:
        return {**none, "status": "no-value", "n": n,
                "detail": f"n at `{spec['n']}` is {n!r}: nothing was measurable"}
    if value is None:
        return {**none, "status": "no-value", "n": n,
                "detail": f"no number at `{spec['value']}`"}
    if not 0.0 <= value <= 1.0:
        return {**none, "status": "no-value", "n": n,
                "detail": f"{value} at `{spec['value']}` is outside 0..1, not a share"}
    k_used = k if k is not None else value * n
    ci = wilson(k_used, n)
    return {"status": MEASURED, "value": round(value, 6), "n": n,
            "k": (int(k) if k is not None and float(k).is_integer() else k),
            "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
            "ci_kind": "wilson" if bench.binomial else "wilson-approx (mean of fractions)",
            "detail": ""}


def trailing_json(text: str) -> Any:
    """The last pretty-printed JSON object on a stream that also carries prose.

    `eval/run_injection_canary.py grade` prints its table and then
    `json.dumps(summary, indent=2)`, so the object opens on a line that is
    exactly `{`.
    """
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].rstrip() == "{":
            try:
                return json.loads("\n".join(lines[i:]))
            except ValueError:
                continue
    try:
        return json.loads(text)
    except ValueError:
        return None


# ── the stores ────────────────────────────────────────────────────────────────

def series_path() -> Path:
    from app import paths as p
    return p.FITNESS_SERIES


def noise_path() -> Path:
    from app import paths as p
    return p.FITNESS_NOISE_PATH


def read_rows(path: Path | None = None) -> list[dict]:
    target = Path(path or series_path())
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_rows(rows: Iterable[dict], path: Path | None = None) -> Path:
    target = Path(path or series_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return target


def read_noise(path: Path | None = None) -> dict:
    try:
        doc = json.loads(Path(path or noise_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def write_noise(doc: dict, path: Path | None = None) -> Path:
    target = Path(path or noise_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def _ts_of(row: Mapping[str, Any]) -> float | None:
    ts = row.get("ts")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return float(ts)
    try:
        return datetime.fromisoformat(str(row.get("recorded_at"))).timestamp()
    except ValueError:
        return None


def measured_rows(rows: Iterable[dict], bench: str) -> list[dict]:
    """One bench's rows that carry a value, oldest first."""
    out = [r for r in rows if r.get("kind") == "bench" and r.get("bench") == bench
           and r.get("status") == MEASURED and _number(r.get("value")) is not None
           and _ts_of(r) is not None]
    return sorted(out, key=lambda r: _ts_of(r) or 0.0)


def summary_row(rows: Sequence[dict], names: Sequence[str], *, run_id: str,
                now: datetime, commit: str | None) -> dict:
    """The closing row of a run: each bench's newest value no older than
    `SUMMARY_CARRY_HOURS`, in the rate-record shape the instrument index reads
    (`{k, n, rate, ci_low, ci_high}`). A bench with nothing that recent is absent,
    which the index reports as `no-value` — a named refusal, not a zero."""
    floor = now.timestamp() - SUMMARY_CARRY_HOURS * 3600
    benches: dict[str, dict] = {}
    for name in names:
        fresh = [r for r in measured_rows(rows, name) if (_ts_of(r) or 0.0) >= floor]
        if not fresh:
            continue
        r = fresh[-1]
        benches[name] = {"rate": r["value"], "n": r.get("n"), "k": r.get("k"),
                         "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                         "recorded_at": r.get("recorded_at"), "run_id": r.get("run_id"),
                         "commit": r.get("commit")}
    return {"schema": SCHEMA, "kind": "summary", "run_id": run_id,
            "recorded_at": now.isoformat(timespec="seconds"), "ts": now.timestamp(),
            "commit": commit, "benches": benches}


# ── the pool and the lock ─────────────────────────────────────────────────────

def _get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (loopback)
            doc = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 — unreachable is an answer, not a crash
        return None
    return doc if isinstance(doc, dict) else None


def pool_state(fetch: Callable[[str], dict | None] = _get_json) -> dict | None:
    """The `pool` block of `GET /api/workers/status`, or None when unreadable."""
    body = fetch(POOL_STATUS_URL)
    pool = body.get("pool") if isinstance(body, dict) else None
    return pool if isinstance(pool, dict) else None


def _inflight(pool: Mapping[str, Any]) -> int:
    n = pool.get("in_flight_count")
    return int(n) if isinstance(n, int) and not isinstance(n, bool) else 0


def pool_precondition(pool: Mapping[str, Any] | None, *, mode: str) -> tuple[bool, str]:
    """Whether an engine bench may start now. `mode` is `paused` (the hand run),
    `pool-job` (the nightly source, itself one in-flight row) or `force`."""
    if mode == "force":
        return True, "forced"
    if pool is None:
        return False, ("the pool state is unreadable (GET /api/workers/status); an "
                       "engine bench is not started blind")
    n = _inflight(pool)
    if mode == "pool-job":
        if n > OWN_INFLIGHT:
            return False, f"{n - OWN_INFLIGHT} other job(s) in flight"
        if (pool.get("kv_gate") or {}).get("engaged"):
            return False, "the KV gate is engaged"
        return True, "the pool holds nothing but this job"
    if not pool.get("paused"):
        return False, ("the worker pool is not paused; pause it on Mission Control "
                       "first (a bench request pre-empts a live worker turn), or "
                       "pass --force")
    if n:
        return False, f"the pool is paused but {n} job(s) are still in flight"
    return True, "the pool is paused and idle"


def lock_path() -> Path:
    """The regression runner's own lock, by its own name."""
    from scripts.automod import state as S
    from workers.sources.automod_regression import REGRESSION_LOCK
    return S.STATE_DIR / REGRESSION_LOCK


def git_head(root: Path = ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


# ── running one bench ─────────────────────────────────────────────────────────

def expand(text: str, ctx: Mapping[str, str], *, now: datetime) -> str:
    def days_ago(m: re.Match) -> str:
        return (now.date() - timedelta(days=int(m.group(1)))).isoformat()
    out = _DAYS_AGO_RE.sub(days_ago, text)
    for key, value in ctx.items():
        out = out.replace("{" + key + "}", value)
    return out


def run_command(argv: Sequence[str], *, env: Mapping[str, str], cwd: Path,
                timeout: float) -> tuple[int | None, str]:
    """One child, in its own process group; `(exit code, output)`, code None on a
    timeout. The group is killed by the pid this call started, never by a pattern."""
    proc = subprocess.Popen(list(argv), cwd=str(cwd), env=dict(env),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        out, _ = proc.communicate()
        return None, out or ""


def _tail(text: str, n: int = 4) -> str:
    return " | ".join(line.strip() for line in text.strip().splitlines()[-n:])[:600]


def _newest_since(pattern: str, started: float) -> Path | None:
    """The newest file matching a path or glob that THIS run wrote. An older file
    at the same path is some other run's number and is never read."""
    target = Path(pattern)
    if any(ch in target.name for ch in "*?["):
        candidates = list(target.parent.glob(target.name))
    else:
        candidates = [target]
    fresh = []
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= started - 1.0:
            fresh.append((mtime, path))
    return max(fresh)[1] if fresh else None


def run_bench(bench: Bench, *, scratch: Path, now: datetime, data_root: Path,
              runner: Callable[..., tuple[int | None, str]] = run_command,
              clock: Callable[[], float] = time.time,
              base_env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run one bench's commands in order and read its value. Never raises for a
    bench that failed: the result says how."""
    ctx = {"python": sys.executable, "root": str(ROOT), "scratch": str(scratch),
           "data_root": str(data_root)}
    env = dict(os.environ if base_env is None else base_env)
    env.update({k: expand(v, ctx, now=now) for k, v in bench.env.items()})
    started = clock()
    deadline = started + bench.timeout_s
    codes: list[int | None] = []
    output = ""
    for argv in bench.commands:
        remaining = deadline - clock()
        if remaining <= 0:
            codes.append(None)
            break
        try:
            code, output = runner([expand(a, ctx, now=now) for a in argv], env=env,
                                  cwd=ROOT, timeout=remaining)
        except OSError as exc:
            return {"status": "failed", "value": None, "n": None, "k": None,
                    "ci_low": None, "ci_high": None, "ci_kind": None,
                    "exit_codes": codes, "duration_s": round(clock() - started, 1),
                    "detail": f"could not start {argv[1] if len(argv) > 1 else argv[0]}: "
                              f"{type(exc).__name__}"}
        codes.append(code)
        if code is None or code not in bench.ok_exit_codes:
            break
    base = {"value": None, "n": None, "k": None, "ci_low": None, "ci_high": None,
            "ci_kind": None, "exit_codes": codes,
            "duration_s": round(clock() - started, 1)}
    if codes and codes[-1] is None:
        return {**base, "status": "timeout",
                "detail": f"exceeded {bench.timeout_s:g} s and was killed"}
    if codes[-1] not in bench.ok_exit_codes:
        return {**base, "status": "failed",
                "detail": f"exit {codes[-1]}: {_tail(output)}"}
    if bench.read["from"] == "stdout":
        doc = trailing_json(output)
        if doc is None:
            return {**base, "status": "no-artifact",
                    "detail": "the last command printed no JSON object"}
    else:
        path = _newest_since(expand(str(bench.read["path"]), ctx, now=now), started)
        if path is None:
            return {**base, "status": "no-artifact",
                    "detail": f"no artifact from this run at {bench.read['path']}"}
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {**base, "status": "no-artifact",
                    "detail": f"{path.name} unreadable: {type(exc).__name__}"}
    return {**base, **read_value(bench, doc)}


# ── run / noise ───────────────────────────────────────────────────────────────

@dataclass
class Deps:
    """Everything `execute_run` touches outside this process, injectable."""
    pool: Callable[[], dict | None] = pool_state
    health: Callable[[], dict | None] = lambda: _get_json(HEALTH_URL)
    commit: Callable[[], str | None] = git_head
    runner: Callable[..., tuple[int | None, str]] = run_command
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    lock: Callable[[float], Any] | None = None       # (wait_s) -> releasable, raises if held
    series: Path | None = None
    data_root: Path | None = None
    make_scratch: Callable[[], Path] | None = None
    record_moves: Callable[..., list[dict]] | None = None


def _acquire_lock(wait_s: float):
    from scripts.automod import state as S
    lock = S.Lock(lock_path(), owner=f"fitness-{os.getpid()}")
    return lock.acquire_wait(wait_s) if wait_s > 0 else lock.acquire()


def _wait_for_quiet(deps: Deps, *, mode: str, wait_s: float, poll_s: float = 15.0
                    ) -> tuple[bool, str, dict | None]:
    deadline = deps.clock() + max(0.0, wait_s)
    while True:
        pool = deps.pool()
        ok, why = pool_precondition(pool, mode=mode)
        if ok or deps.clock() >= deadline:
            return ok, why, pool
        deps.sleep(poll_s)


def execute_run(benches: Sequence[Bench], *, repeat: int = 1, mode: str = "paused",
                now: datetime | None = None, deps: Deps | None = None,
                lock_wait_s: float = 0.0, quiet_wait_s: float = 0.0,
                budget_s: float | None = None, record: bool = True,
                keep_scratch: bool = False) -> dict[str, Any]:
    """Run `benches` sequentially, `repeat` times each. Returns
    `{exit, run_id, rows, refused}`; appends to the series when `record`."""
    deps = deps or Deps()
    stamp = now or datetime.now(timezone.utc)
    run_id = stamp.strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    needs_engine = any(b.engine for b in benches)
    first_pool = deps.pool() if needs_engine else None
    if needs_engine and mode == "paused":
        ok, why = pool_precondition(first_pool, mode=mode)
        if not ok:
            return {"exit": EXIT_REFUSED, "run_id": run_id, "rows": [], "refused": why}
    # An offline bench touches no engine and no daemon, so a run of only those
    # neither needs the pool paused nor makes the regression runner wait.
    lock = None
    if needs_engine:
        try:
            lock = (deps.lock or _acquire_lock)(lock_wait_s)
        except Exception as exc:  # noqa: BLE001 — LockHeld, or a lock that cannot be opened
            return {"exit": EXIT_LOCK_HELD, "run_id": run_id, "rows": [],
                    "refused": f"regression.lock is held ({exc}); nothing was run"}
    from app import paths as p
    data_root = deps.data_root or p.DATA_ROOT
    scratch = (deps.make_scratch or (lambda: Path(tempfile.mkdtemp(prefix="lloyd-fitness-"))))()
    commit = deps.commit()
    health = deps.health() if needs_engine else None
    backend_commit = (health or {}).get("commit") if isinstance(health, dict) else None
    started = deps.clock()
    rows: list[dict] = []
    try:
        for bench in benches:
            for rep in range(1, max(1, repeat) + 1):
                row = {"schema": SCHEMA, "kind": "bench", "run_id": run_id,
                       "bench": bench.name, "rep": rep, "commit": commit,
                       "backend_commit": backend_commit,
                       "higher_is_better": bench.higher_is_better,
                       "engine": bench.engine, "mode": mode}
                skipped = None
                pool = None
                if bench.engine and mode != "force":
                    ok, why, pool = _wait_for_quiet(deps, mode=mode, wait_s=quiet_wait_s)
                    if not ok:
                        skipped = f"no quiet engine: {why}"
                if (skipped is None and budget_s is not None
                        and deps.clock() - started + bench.timeout_s > budget_s):
                    skipped = (f"the run's budget of {budget_s:g} s has no room for "
                               f"this bench's {bench.timeout_s:g} s timeout")
                if skipped:
                    result = {"status": "skipped", "value": None, "n": None, "k": None,
                              "ci_low": None, "ci_high": None, "ci_kind": None,
                              "exit_codes": [], "duration_s": 0.0, "detail": skipped}
                else:
                    bench_scratch = scratch / f"{bench.name}-{rep}"
                    bench_scratch.mkdir(parents=True, exist_ok=True)
                    result = run_bench(bench, scratch=bench_scratch, now=stamp,
                                       data_root=data_root, runner=deps.runner,
                                       clock=deps.clock)
                at = datetime.fromtimestamp(deps.clock(), tz=timezone.utc)
                row.update(result)
                row.update({"recorded_at": at.isoformat(timespec="seconds"),
                            "ts": at.timestamp(),
                            "pool": (None if pool is None else
                                     {"paused": bool(pool.get("paused")),
                                      "in_flight": _inflight(pool)})})
                rows.append(row)
                if record:
                    append_rows([row], deps.series)
    finally:
        try:
            if lock is not None:
                lock.release()
        except Exception:  # noqa: BLE001
            pass
        # Only the directory this call made, and only when it made one.
        if not keep_scratch and deps.make_scratch is None:
            shutil.rmtree(scratch, ignore_errors=True)
    if record:
        end = datetime.fromtimestamp(deps.clock(), tz=timezone.utc)
        series_rows = read_rows(deps.series)
        every = sorted({str(r["bench"]) for r in series_rows
                        if r.get("kind") == "bench" and r.get("bench")})
        append_rows([summary_row(series_rows, every, run_id=run_id, now=end,
                                 commit=commit)], deps.series)
        # The default recorder writes the live ledger, so it runs only for the live
        # series; a caller that redirected the series names its own recorder.
        recorder = deps.record_moves or (record_moves if deps.series is None else None)
        if recorder is not None:
            try:
                recorder(now=end, series=deps.series)
            except Exception as exc:  # noqa: BLE001 — attribution never fails a run
                print(f"fitness: could not record moves ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
    failed = [r for r in rows if r["status"] != MEASURED]
    return {"exit": EXIT_SOME_FAILED if failed else EXIT_OK, "run_id": run_id,
            "rows": rows, "refused": None, "scratch": str(scratch) if keep_scratch else None}


def noise_of(values: Sequence[float], *, max_spread: float) -> dict[str, Any]:
    """The spread of N repeats at one commit. `too-noisy` when max - min exceeds
    the bench's declared threshold; `insufficient` under `MIN_NOISE_REPEATS`."""
    vals = [float(v) for v in values]
    if len(vals) < MIN_NOISE_REPEATS:
        return {"status": "insufficient", "repeats": len(vals), "values": vals,
                "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                "mean": None, "stdev": None, "spread": None, "max_spread": max_spread}
    spread = max(vals) - min(vals)
    return {"status": "too-noisy" if spread > max_spread else "ok",
            "repeats": len(vals), "values": [round(v, 6) for v in vals],
            "min": round(min(vals), 6), "max": round(max(vals), 6),
            "mean": round(statistics.fmean(vals), 6),
            "stdev": round(statistics.stdev(vals), 6),
            "spread": round(spread, 6), "max_spread": max_spread}


def execute_noise(benches: Sequence[Bench], *, repeat: int, mode: str = "paused",
                  now: datetime | None = None, deps: Deps | None = None,
                  noise_file: Path | None = None, lock_wait_s: float = 0.0
                  ) -> dict[str, Any]:
    """N runs of each bench at one commit, written as that bench's noise floor.
    The repeats are not appended to the series: they are one commit measured N
    times, and a trend line with five points on one night would read as a trend."""
    out = execute_run(benches, repeat=repeat, mode=mode, now=now, deps=deps,
                      lock_wait_s=lock_wait_s, record=False)
    if out["refused"]:
        return out
    doc = read_noise(noise_file)
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    for bench in benches:
        mine = [r for r in out["rows"] if r["bench"] == bench.name]
        vals = [r["value"] for r in mine if r["status"] == MEASURED]
        entry = noise_of(vals, max_spread=bench.noise_max_spread)
        entry.update({"measured_at": stamp, "commit": mine[0]["commit"] if mine else None,
                      "attempted": len(mine), "mode": mode,
                      "duration_s": [r["duration_s"] for r in mine],
                      "failures": [f"{r['status']}: {r['detail']}" for r in mine
                                   if r["status"] != MEASURED]})
        doc[bench.name] = entry
    write_noise(doc, noise_file)
    out["noise"] = {b.name: doc[b.name] for b in benches}
    return out


# ── deltas, moves, attribution ────────────────────────────────────────────────

def bench_delta(rows: Sequence[dict], bench: str, noise: Mapping[str, Any], *,
                now: float, days: float = DELTA_DAYS) -> dict[str, Any]:
    """One bench's latest value, its delta against the newest row at least `days`
    older, and whether that delta is outside the measured noise floor.

    `outside_noise` is True or False only when there is a floor to compare with;
    it is None — and `why` says which — when the bench has never been measured,
    has no row old enough, has no recorded floor, or is flagged `too-noisy`.
    Outside means BOTH: the delta is larger than the measured spread, and the
    latest row's own Wilson interval does not cover the reference value.
    """
    mine = measured_rows(rows, bench)
    floor = noise.get(bench) if isinstance(noise.get(bench), Mapping) else None
    out: dict[str, Any] = {"bench": bench, "latest": None, "reference": None,
                           "delta": None, "direction": None, "outside_noise": None,
                           "noise_status": (floor or {}).get("status") or "unmeasured",
                           "noise_spread": (floor or {}).get("spread"), "why": ""}
    if not mine:
        out["why"] = "unmeasured: no row carries a value"
        return out
    latest = mine[-1]
    out["latest"] = {k: latest.get(k) for k in
                     ("value", "n", "ci_low", "ci_high", "commit", "recorded_at",
                      "run_id", "ts")}
    out["age_days"] = round((now - (_ts_of(latest) or now)) / 86400.0, 2)
    cutoff = (_ts_of(latest) or 0.0) - days * 86400.0
    older = [r for r in mine if (_ts_of(r) or 0.0) <= cutoff]
    if not older:
        out["why"] = f"no row at least {days:g} d older than the latest"
        return out
    ref = older[-1]
    out["reference"] = {k: ref.get(k) for k in
                        ("value", "n", "commit", "recorded_at", "run_id", "ts")}
    delta = round(float(latest["value"]) - float(ref["value"]), 6)
    out["delta"] = delta
    better = delta > 0 if latest.get("higher_is_better", True) else delta < 0
    out["direction"] = "unchanged" if delta == 0 else ("improved" if better else "regressed")
    if out["noise_status"] == "too-noisy":
        out["why"] = "excluded: the bench is flagged too-noisy"
    elif out["noise_status"] != "ok" or out["noise_spread"] is None:
        out["why"] = "no measured noise floor (run `fitness noise`)"
    else:
        # Two bounds, and a move has to clear both. The measured spread is what the
        # SAME corpus gives back at one commit; the Wilson interval is what a
        # different draw of the corpus could give. A bench that is deterministic on
        # a fixed set has a spread of zero, and one that scores live traffic has a
        # spread that says nothing about next week's traffic — so neither bound
        # alone is a floor (#696: a value the interval covers has not moved).
        lo, hi = _number(latest.get("ci_low")), _number(latest.get("ci_high"))
        beyond_spread = abs(delta) > float(out["noise_spread"])
        covered = lo is not None and hi is not None and lo <= float(ref["value"]) <= hi
        out["outside_noise"] = beyond_spread and not covered
        if beyond_spread and covered:
            out["why"] = "the latest row's 95% interval covers the reference value"
    return out


def deltas(now: float, *, rows: Sequence[dict] | None = None,
           noise: Mapping[str, Any] | None = None,
           names: Sequence[str] | None = None) -> list[dict]:
    rows = read_rows() if rows is None else rows
    noise = read_noise() if noise is None else noise
    if names is None:
        try:
            names = [b.name for b in load_manifest()]
        except (OSError, ManifestError):
            names = sorted({r["bench"] for r in rows if r.get("kind") == "bench"
                            and r.get("bench")})
    return [bench_delta(rows, name, noise, now=now) for name in names]


def moves(now: float, *, rows: Sequence[dict] | None = None,
          noise: Mapping[str, Any] | None = None,
          names: Sequence[str] | None = None) -> list[dict]:
    """The benches whose 7-day delta is outside their noise floor, as at `now`.

    Pure given `rows`, `noise` and `names`; with none of them it reads the two
    stores and the manifest. This is the function a later consumer calls.
    """
    return [d for d in deltas(now, rows=rows, noise=noise, names=names)
            if d["outside_noise"] is True]


def landings_between(events: Iterable[Mapping[str, Any]], start_ts: float,
                     end_ts: float) -> list[dict]:
    """The `promoted` ledger events after `start_ts` and up to `end_ts`."""
    out = []
    for e in events:
        ts = e.get("ts")
        if (e.get("event") == "promoted" and isinstance(ts, (int, float))
                and start_ts < float(ts) <= end_ts):
            out.append({"round_id": e.get("round_id"), "commit": e.get("commit"),
                        "title": e.get("title")})
    return out


def record_moves(*, now: datetime | None = None, series: Path | None = None,
                 noise: Mapping[str, Any] | None = None,
                 names: Sequence[str] | None = None,
                 events: Sequence[Mapping[str, Any]] | None = None,
                 append: Callable[[dict], None] | None = None) -> list[dict]:
    """Append one `fitness_move` ledger event per bench that REGRESSED outside its
    noise floor: the bench, the delta, and the landings between the two measured
    commits. Record only — nothing is bisected and nothing is rolled back. A move
    already recorded for the same pair of runs is not recorded twice.
    """
    from scripts.automod import state as S
    stamp = now or datetime.now(timezone.utc)
    rows = read_rows(series)
    found = [m for m in moves(stamp.timestamp(), rows=rows, noise=noise, names=names)
             if m["direction"] == "regressed"]
    if not found:
        return []
    ledger = list(S.ledger_rows()) if events is None else list(events)
    seen = {(e.get("bench"), e.get("to_run"), e.get("from_run")) for e in ledger
            if e.get("event") == MOVE_EVENT}
    written: list[dict] = []
    for m in found:
        latest, ref = m["latest"], m["reference"]
        if (m["bench"], latest["run_id"], ref["run_id"]) in seen:
            continue
        entry = {"event": MOVE_EVENT, "bench": m["bench"], "direction": m["direction"],
                 "delta": m["delta"], "noise_spread": m["noise_spread"],
                 "value": latest["value"], "n": latest["n"],
                 "reference_value": ref["value"],
                 "from_commit": ref["commit"], "to_commit": latest["commit"],
                 "from_run": ref["run_id"], "to_run": latest["run_id"],
                 "from_at": ref["recorded_at"], "to_at": latest["recorded_at"],
                 "landings": landings_between(ledger, float(ref["ts"]), float(latest["ts"])),
                 "action": "recorded only: no bisect, no rollback"}
        (append or S.append_event)(entry)
        written.append(entry)
    return written


# ── report ────────────────────────────────────────────────────────────────────

def _fmt(x: Any, spec: str = ".3f") -> str:
    return "—" if not isinstance(x, (int, float)) or isinstance(x, bool) else f"{x:{spec}}"


def render_report(ds: Sequence[dict]) -> str:
    lines = [f"{'bench':18s} {'value':>7s} {'n':>5s} {'95% CI':>15s} {'7d delta':>9s} "
             f"{'noise':>7s}  verdict",
             "-" * 92]
    for d in ds:
        latest = d["latest"] or {}
        ci = (f"[{_fmt(latest.get('ci_low'))},{_fmt(latest.get('ci_high'))}]"
              if latest else "—")
        if not latest:
            verdict = "unmeasured"
        elif d["outside_noise"] is True:
            verdict = f"MOVED ({d['direction']}), outside the noise floor"
        elif d["outside_noise"] is False:
            verdict = "inside the noise floor"
        else:
            verdict = d["why"]
        lines.append(
            f"{d['bench']:18s} {_fmt(latest.get('value')):>7s} "
            f"{str(latest.get('n') if latest else '—'):>5s} {ci:>15s} "
            f"{_fmt(d['delta'], '+.3f'):>9s} {_fmt(d['noise_spread']):>7s}  {verdict}")
    return "\n".join(lines)


def scorecard_block(now: float, *, rows: Sequence[dict] | None = None,
                    noise: Mapping[str, Any] | None = None,
                    names: Sequence[str] | None = None) -> dict[str, Any]:
    """What scorecard row 22 carries: per bench, value and 7-day delta. An empty
    series is `measured: False`, which the row renders `unmeasured` — never 0."""
    ds = deltas(now, rows=rows, noise=noise, names=names)
    benches = [{"bench": d["bench"],
                "value": (d["latest"] or {}).get("value"),
                "n": (d["latest"] or {}).get("n"),
                "delta_7d": d["delta"], "outside_noise": d["outside_noise"],
                "direction": d["direction"], "noise_status": d["noise_status"],
                "age_days": d.get("age_days")} for d in ds]
    return {"measured": any(b["value"] is not None for b in benches),
            "benches": benches,
            "moved": [b["bench"] for b in benches if b["outside_noise"] is True]}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _mode(args: argparse.Namespace) -> str:
    if getattr(args, "force", False):
        return "force"
    return "pool-job" if getattr(args, "pool_job", False) else "paused"


def _print_rows(rows: Sequence[dict]) -> None:
    for r in rows:
        print(f"  {r['bench']:18s} rep {r['rep']}  {r['status']:11s} "
              f"value {_fmt(r['value'])}  n {r['n']}  {r['duration_s']:.0f} s"
              + (f"  — {r['detail']}" if r.get("detail") else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.automod.fitness",
        description="Behaviour fitness benches: run, measure noise floors, report. "
                    "Report-only: nothing gates, promotes or rolls back on it.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run the manifest once and append to the series")
    noise_p = sub.add_parser(
        "noise", help="repeat each bench at one commit; record its noise floor")
    for sp, default_repeat in ((run_p, 1), (noise_p, 5)):
        sp.add_argument("--only", action="append", default=None, metavar="NAME")
        sp.add_argument("--repeat", type=int, default=default_repeat)
        sp.add_argument("--force", action="store_true",
                        help="run engine benches although the pool is not paused")
        sp.add_argument("--lock-wait", type=float, default=0.0, metavar="SECONDS",
                        help="how long to queue for regression.lock (default: refuse at once)")
    run_p.add_argument("--once", action="store_true",
                       help="one pass over the manifest (the only mode there is)")
    run_p.add_argument("--pool-job", action="store_true",
                       help="the nightly source's mode: wait before each engine bench "
                            "until the pool holds nothing but this job")
    run_p.add_argument("--quiet-wait", type=float, default=600.0, metavar="SECONDS")
    run_p.add_argument("--budget-seconds", type=float, default=None)
    run_p.add_argument("--keep-scratch", action="store_true")
    rp = sub.add_parser("report", help="latest value, 7-day delta and noise verdict per bench")
    rp.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "report":
        ds = deltas(time.time())
        print(json.dumps(ds, indent=2) if args.json else render_report(ds))
        return EXIT_OK

    try:
        benches = select(load_manifest(), args.only)
    except (ManifestError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    mode = _mode(args)
    if args.cmd == "run":
        out = execute_run(benches, repeat=args.repeat, mode=mode,
                          lock_wait_s=args.lock_wait,
                          quiet_wait_s=args.quiet_wait if mode == "pool-job" else 0.0,
                          budget_s=args.budget_seconds, keep_scratch=args.keep_scratch)
    else:
        if args.repeat < MIN_NOISE_REPEATS:
            print(f"refused: a noise floor needs at least {MIN_NOISE_REPEATS} repeats",
                  file=sys.stderr)
            return EXIT_REFUSED
        out = execute_noise(benches, repeat=args.repeat, mode=mode,
                            lock_wait_s=args.lock_wait)
    if out["refused"]:
        print(f"refused: {out['refused']}", file=sys.stderr)
        return out["exit"]
    print(f"fitness {args.cmd} {out['run_id']}:")
    _print_rows(out["rows"])
    for name, entry in (out.get("noise") or {}).items():
        print(f"  noise {name:18s} {entry['status']:12s} spread {_fmt(entry['spread'])} "
              f"over {entry['repeats']} of {entry['attempted']}")
    measured = sum(1 for r in out["rows"] if r["status"] == MEASURED)
    print("FITNESS_RESULT " + json.dumps(
        {"run_id": out["run_id"], "measured": measured, "rows": len(out["rows"]),
         "not_measured": [f"{r['bench']}: {r['status']}" for r in out["rows"]
                          if r["status"] != MEASURED]}))
    return out["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
