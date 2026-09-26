"""The promotion gate: eleven rungs, cheapest first, every one fails closed.

"Fails closed" is not a slogan here — it is the reason `_rung` catches every
exception and records it as a FAILED rung. With no human review tier, a rung
that errors and is read as "didn't fail" would silently remove a check.

The gate always runs **from the live tree** against the worktree, so a
candidate that weakens the gate is judged by the old gate, not by itself.

Rung order is a cost ladder, and it short-circuits: an import error should
cost 3 seconds, not a full canary boot.

  0 preflight      ~1s    lock, clean tree, ancestry, diff scope; with an item
                          bound, that it has clauses and the diff has a test
  1 vet            ~1s    OBSERVE-ONLY structural pass over the change set
                          itself (scripts/automod/vet.py): a file non-empty at
                          base and empty at HEAD, a newly-added binary blob
                          outside the binary allowlist, a diff over the changed-
                          line ceiling. Records into report + ledger; blocks
                          nothing until a soak of real landings says it may.
  2 static         ~8s    compileall, import smoke, pyflakes delta
  3 tests         ~70s    full pytest on `automod.gate.test_workers` xdist
                          workers (~600 s serial) + a collected-count floor.
                          A failure under load is re-run serially before it
                          is believed; a real one re-probes the failing files
                          at the round's base to say whose breakage it is, and
                          a failure that is not the round's (pre-existing or
                          flaky, in a file the diff did not touch) PASSES the
                          rung, recorded and filed as a `red-tree` item
  4 review      60-180s   a second reader: a fresh session on the LIVE
                          backend grades the diff against the item's
                          acceptance clauses (scripts/automod/review.py).
                          Refuses with findings; the premise verdict decides
                          whether the item is retried or handed to a human.
                          Skipped (recorded) only when no item is bound.
  5 venv        0-300s    only when requirements changed (reflink + uv)
  6 canary boot   ~30s    both /health green, tool floor, config-follows-code
  7 canary smoke  ~30s    one real turn, sentinel through a real Bash call
                          (recorded as SKIPPED when the engine is unreachable;
                           `skip_smoke` is refused while it answers)
  8 drill         ~90s    only when the diff touches the rollback path

Two rungs are not numbered above because they only exist conditionally:
`frontend` (between `static` and `tests`: tsc delta + vite build, recorded as
SKIPPED when no `web/` path changed) and `prompt_surface` (between `tests` and
`review`: the scored behavioural check, only for a prompt-surface diff). Of the
eleven, `vet` is the only rung that reads the change set rather than the
behaviour it produces, and the only one deliberately written not to block.

`review` sits after `tests` so the grader can trust a green tree and is
handed the counts, and before `venv` so a refusal saves the venv build, the
boot, the smoke and the drill — and because `canary_smoke` must immediately
precede `drill` (the drill needs the canary's ports).
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app import lint_findings
from scripts.automod import canary as C
from scripts.automod import canary_smoke as CS
from scripts.automod import spec, state as S, testpaths as TP, vet as V, worktree as W

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
PYTEST_MIN_COLLECTED = 1000
# Collected is not the same as run. `pytest -q` exits 0 having collected 1632
# items and executed none of them if a conftest import fails into a
# module-level skip, or if a round adds a broad `skipif`. Both counts have to
# hold, or the rung's own guarantee ("the suite still passes") is satisfied by
# a suite that did nothing.
PYTEST_MIN_PASSED = 1000
PYTEST_MAX_SKIPPED = 40
UV_BIN = Path.home() / ".local" / "bin" / "uv"


@dataclass
class RungResult:
    name: str
    ok: bool
    detail: str = ""
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


@dataclass
class GateReport:
    round_id: str
    base: str
    head: str
    changed_paths: list[str] = field(default_factory=list)
    rungs: list[RungResult] = field(default_factory=list)
    ok: bool = False
    venv: str | None = None

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id, "base": self.base, "head": self.head,
            "ok": self.ok, "changed_paths": self.changed_paths, "venv": self.venv,
            "rungs": [{"name": r.name, "ok": r.ok, "detail": r.detail,
                       "seconds": round(r.seconds, 2), **({"data": r.data} if r.data else {})}
                      for r in self.rungs],
        }


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
         timeout: float = 900.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env,
                          capture_output=True, text=True, timeout=timeout, check=False)


def _pyflakes(python: Path, root: Path, files: list[str]) -> set[str]:
    """Findings for `files`, normalized so line numbers do not create noise."""
    existing = [f for f in files if (root / f).exists() and f.endswith(".py")]
    if not existing:
        return set()
    r = _run([str(python), "-m", "pyflakes", *existing], cwd=root, timeout=300)
    return lint_findings.parse_pyflakes(r.stdout + r.stderr)


# The three normalisers moved to `app/lint_findings.py` so the aggregator can
# use them for post-edit diagnostics. It cannot import this module — that
# would pull the whole self-modification package into every tool call — and
# two private copies of "is this finding new?" is exactly how the gate and
# the model would come to disagree about the same edit.
_parse_tsc = lint_findings.parse_tsc
_node_env = lint_findings.node_env


def _tsc_findings(web: Path, timeout: float = 300) -> Counter:
    web = web.resolve()
    tsc = web / "node_modules" / ".bin" / "tsc"
    r = _run([str(tsc), "--noEmit", "-p", "."], cwd=web, env=_node_env(), timeout=timeout)
    return _parse_tsc(r.stdout + r.stderr)


def _canary_lock_held() -> bool:
    """Whether another gate holds the canary ports right now. Probed by
    taking and at once releasing the lock, which is what `flock` offers."""
    try:
        S.Lock(S.GATE_CANARY_LOCK_PATH, owner="preflight-probe").acquire().release()
    except S.LockHeld:
        return True
    except OSError:
        return False
    return False


def _vite_build(web: Path, out_dir: Path, timeout: float = 600) -> tuple[bool, str]:
    web = web.resolve()
    vite = web / "node_modules" / ".bin" / "vite"
    r = _run([str(vite), "build", "--outDir", str(out_dir), "--emptyOutDir",
              "--logLevel", "error"], cwd=web, env=_node_env(), timeout=timeout)
    text = (r.stdout + r.stderr).strip()
    return r.returncode == 0, text[-900:]


def _vitest_run(web: Path, timeout: float = 600) -> tuple[bool | None, str]:
    """`(ok, tail)` for the frontend unit tests; `ok` is None when they cannot run.

    Until 2026-09-17 `web/` had no test runner at all, so a frontend clause
    had no node a gate could grade (#1199's mount cascade and polling
    interval). `vitest.config.ts` plus the binary means the tests exist and
    can run; either one missing is a skip that SAYS so, never a pass that
    looks like one — `node_modules` is untracked, and a box where nobody has
    run `npm install` since vitest was added must not fail every frontend
    round for it.
    """
    web = web.resolve()
    vitest = web / "node_modules" / ".bin" / "vitest"
    if not (web / "vitest.config.ts").exists():
        return None, "no web/vitest.config.ts"
    if not vitest.exists():
        return None, "vitest is not installed — run npm install in ~/lloyd/web"
    r = _run([str(vitest), "run", "--reporter=dot"], cwd=web, env=_node_env(), timeout=timeout)
    text = (r.stdout + r.stderr).strip()
    return r.returncode == 0, text[-900:]


def engine_reachable(root: Path, timeout: float = 4.0) -> tuple[bool, str]:
    """Is the default model's endpoint answering? Decides whether `skip_smoke` holds.

    `skip_smoke` exists for a machine with no vLLM. It is not a discretionary
    "this round does not need a real turn" switch, but nothing enforced that,
    and on 2026-09-08 round SM_20260908_165950 passed it while the engine was
    up — the same turn then ran four bench tasks and a 20-query eval through
    that very endpoint. The gate recorded seven rungs and said nothing about
    the eighth, so the promotion record for a 66% rewrite of the operating
    contract showed no live turn and no gap where one should have been.

    Reads `config.yaml` raw rather than `app.config.CONFIG`, for the same
    reason `canary_config` does: the gate must not import the application it
    is judging.
    """
    try:
        import urllib.request

        import yaml

        raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
        alias = (raw.get("model") or {}).get("default") or "primary"
        base = ((raw.get("models") or {}).get(alias) or {}).get("base_url") or ""
        if not base:
            return False, f"no base_url for model {alias!r} in config.yaml"
        url = base.rstrip("/") + "/v1/models"
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 300, f"{url} answered {resp.status}"
    except Exception as exc:  # noqa: BLE001 — unreachable is the answer, not an error
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


def _parse_pytest_summary(text: str) -> dict:
    """Pull counts out of pytest's summary line.

    The skipped COUNT is `tests_skipped`, not `skipped`, and the rename is the
    whole of a real bug. `_rung` records a rung as skipped from
    `data["skipped"]`, and `rung_tests` returns these counts as its data — so
    a suite with three skipped tests recorded the entire `tests` rung as
    "skipped" on the ledger. The scorecard reads that field, which made a
    round that ran its whole suite indistinguishable from one that never ran
    it.
    """
    out: dict = {"passed": 0, "failed": 0, "errors": 0, "xfailed": 0,
                 "tests_skipped": 0, "collected": 0}
    m = re.search(r"collected (\d+) item", text)
    if m:
        out["collected"] = int(m.group(1))
    for key, pattern in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"),
                         ("errors", r"(\d+) error"), ("xfailed", r"(\d+) xfailed"),
                         ("tests_skipped", r"(\d+) skipped")):
        m = re.search(pattern, text)
        if m:
            out[key] = int(m.group(1))
    if not out["collected"]:
        out["collected"] = (out["passed"] + out["failed"] + out["xfailed"]
                            + out["errors"] + out["tests_skipped"])
    return out


# Every rollback and every abort this loop has performed has been a false
# positive, and the `tests` rung has its own version of that failure: on
# 2026-09-08 three rounds aborted on the same three pre-existing failures, and
# because `implemented_ids` counts any finished round as the item's one
# attempt, #361, #370 and #376 were consumed by breakage none of them wrote.
# The fix landed seven hours after the last of them and nothing went back.
#
# So the rung asks a second question when it fails: do these exact tests
# already fail at the base this round branched from, where the diff under test
# is absent? Until 2026-09-24 a yes still blocked the promotion ("landing onto
# a red tree would give the guardian's observation window a broken baseline").
# It does not: the guardian never runs pytest. A yes now PASSES the rung with
# the ids recorded and the breakage filed as a `red-tree` item (see
# `Gate._tests_failed`); what a red tree cost was 157 rounds a week.
EXTERNAL_PROBE_TIMEOUT = 600.0

# The `-m` expression every pytest invocation in this module carries, in one place because the
# three sites below must agree and an argv is not something a typo survives quietly.
#
# `not live_vault` is the pre-existing half: those assertions read the live ~/obsidian vault,
# which no round under test controls, so on a hard rung they would fail the next author for the
# previous writer's change (see tests/test_prompt_surface_guard.py, which pins the seam).
#
# `not fault_injection` is #644 clause 4: the degradation matrix's rows bind fixture ports and
# write temp state files, and this rung runs `pytest -q` on a box whose worker pool may be live
# serving Alan. They are excluded from the gate, never from the suite — anything that runs
# without a narrowing `-m` still reaches them. The marker is deliberately NOT registered in
# pytest.ini: that file is a path this loop may not write, and #644's acceptance check requires
# the suite to land without editing it, so this expression is the only place the exclusion could
# live, which is exactly why it is pinned (tests/test_degradation_contract.py asserts it carries
# the marker the rows apply). Selection does not depend on registration either way — `-m`
# matches marks applied to an item, not names listed in `ini` — so the unregistered name costs
# one PytestUnknownMarkWarning and buys a harmless term in a candidate tree that predates
# tests/degradation/, where it must match nothing rather than error.
#
# It has to be ONE argument rather than two `-m` flags: pytest's `-m` is `store`, so a second
# occurrence replaces the first and would silently drop the `live_vault` exclusion.
TESTS_MARK_EXPR = "not live_vault and not fault_injection"


def _failed_node_ids(text: str) -> list[str]:
    """Node ids from pytest's short test summary, in order, deduplicated.

    Both `FAILED tests/x.py::test_y - AssertionError` and the bare
    `ERROR tests/x.py` a collection failure emits are node ids pytest will
    accept back on a command line, which is what the base probe needs.
    """
    out: list[str] = []
    for raw in text.splitlines():
        m = re.match(r"^(?:FAILED|ERROR)\s+(\S+)", raw.strip())
        if not m:
            continue
        nid = m.group(1)
        # Guard against prose: pytest writes "ERROR" in other contexts, and a
        # node id always names a file.
        if ".py" not in nid or nid in out:
            continue
        out.append(nid)
    return out


def _classify_test_failure(node_ids: list[str],
                           base_failed: set[str]) -> tuple[bool, list[str]]:
    """`(external, new)` — a delta, for the same reason the pyflakes rung is one.

    A failure absent from the base is this round's. A failure present at the
    base predates it. External means *every* failure reproduces: one new
    failure alongside twenty old ones is still a round that broke something,
    and the exemption is for rounds that broke nothing at all.

    Empty `node_ids` is never external. pytest failed and the summary could not
    be parsed, which is a reason to know less, not a reason to grant an
    exemption — this whole path fails closed.
    """
    new = [n for n in node_ids if n not in base_failed]
    return (bool(node_ids) and not new), new


def _name_ids(ids: list[str], cap: int = 20) -> str:
    """Failing node ids for a rung detail: every one up to `cap`, then a count.
    Whole ids only — a clipped id reads as the name of a test that does not
    exist."""
    shown = ", ".join(ids[:cap])
    return shown + (f", +{len(ids) - cap} more (all in data.failed_node_ids)"
                    if len(ids) > cap else "")


# How many times a candidate failure that the base probe did NOT reproduce is
# re-run before it is called the round's own. Three, so a node that fails every
# repeat has been asked four times in total counting the suite run — enough to
# separate "this change broke it" from "this run flickered", and seconds of
# wall clock for the handful of node ids a round typically fails.
REPEAT_RUNS = 3
# Above this many new failures the tree is broken, not fickle, and the repeats
# would only spend minutes proving what `failed` already says.
REPEAT_MAX_NODES = 40
REPEAT_TIMEOUT = 300.0


def _reconfirm_candidate_failures(python: Path, root: Path,
                                  node_ids: list[str], *,
                                  repeats: int = REPEAT_RUNS,
                                  timeout: float = REPEAT_TIMEOUT,
                                  env: dict | None = None) -> tuple[list[str], str]:
    """`(flaky_ids, note)` — the nodes among `node_ids` that pass a repeat run.

    The `tests` rung used to decide "pre-existing vs new in this round" from one
    pytest invocation on each side: one candidate run, one base probe of the
    failing files. A node that flickers therefore belonged to whichever round
    happened to trip it, and because `backlog.implemented_ids` counts any
    finished round as the item's one attempt, the item was spent by a coin toss.
    Round SM_20260916_042752 was refused for `all 1 failure(s) are new in this
    round` on a node that failed 1 run in 5 at the base it branched from, where
    the base probe's single run came back green.

    So a node that was not reproduced at base is asked again, up to `repeats`
    times. One pass and it is flaky — an intermittent red is a fact about the
    box or about test order, not about this diff. Failing every repeat keeps the
    old verdict verbatim.

    Called only with the nodes the base probe did NOT already reproduce, and
    only by node id: a node whose file the base probe already runs is attributed
    by that probe, and re-running whole files here would spend the suite a
    second time to learn nothing new.

    Fails closed the same way the base probe does. `pytest rc == 0` is a pass,
    and `pytest rc != 0 with the node absent from the summary` is a pass only
    when pytest actually ran — a missing interpreter, a usage error or a crash
    all come back naming nothing, and granting flaky on those would hand every
    environment break an exemption. No exception escapes: an attribution that
    cannot be earned is the round's, which is the status quo.
    """
    k = int(repeats)
    if k <= 0:
        return [], f"repeats off (k={k}); attribution from one run"
    if not node_ids:
        return [], "no node ids to re-run"
    if len(node_ids) > REPEAT_MAX_NODES:
        return [], (f"{len(node_ids)} new failures (>{REPEAT_MAX_NODES}) is a broken "
                    "tree, not a flake: repeats skipped")
    flaky: list[str] = []
    pending = list(node_ids)
    try:
        for _ in range(k):
            if not pending:
                break
            r = _run([str(python), "-m", "pytest", "-q", "--no-header",
                      "-p", "no:cacheprovider", "-m", TESTS_MARK_EXPR, *pending],
                     cwd=root, env=env or None, timeout=timeout)
            text = r.stdout + r.stderr
            if not _parse_pytest_summary(text)["collected"]:
                tail = " | ".join(text.strip().splitlines()[-3:])[:200]
                return flaky, (f"reconfirm INCONCLUSIVE after {len(flaky)} flaky — "
                               f"pytest produced no summary (rc={r.returncode}): {tail}")
            named = set(_failed_node_ids(text))
            # A node "passed" only if the run named neither the node nor its
            # file: a collection failure comes back as `ERROR tests/x.py`, which
            # names the file and means nothing inside it ran.
            passed = [n for n in pending
                      if n not in named and n.split("::", 1)[0] not in named]
            flaky.extend(passed)
            # Anything that did not clearly pass stays in the batch — including a
            # node the run never named because its file failed to collect, which
            # is neither a pass nor a named failure.
            pending = [n for n in pending if n not in passed]
    except subprocess.TimeoutExpired:
        return flaky, (f"reconfirm timed out after {timeout:.0f}s with "
                       f"{len(pending)} node(s) still failing")
    except Exception as exc:
        return flaky, f"reconfirm failed: {type(exc).__name__}: {exc}"
    if not pending:
        return flaky, (f"{len(flaky)} failure(s) passed a repeat run of {k} and are "
                       f"recorded as flaky")
    return flaky, (f"{len(flaky)} of {len(node_ids)} passed a repeat run; "
                   f"{len(pending)} failed all {k}")


def _failures_at_base(python: Path, live_root: Path, base: str,
                      node_ids: list[str], scratch: Path,
                      env: dict | None = None) -> tuple[set[str], str]:
    """Which of `node_ids` already fail at `base`, in a throwaway worktree.

    Module-level and fully parameterised so it can be exercised against a real
    throwaway repo without standing up a Gate.

    Fails closed in every direction: a worktree that will not create, a probe
    that times out, a python that will not start — all return the empty set,
    which classifies the failure as the round's own. The cost of being wrong
    that way is the status quo; the cost of being wrong the other way is
    landing a change nobody checked.

    The worktree is at base; the data root has to be too. `env` arrives as the
    tests rung's own (`_child_env` with the round home), whose `LLOYD_DATA` is
    the round data root the failing run just wrote into, so a store or cache
    the CANDIDATE's tests created was present "at base" and its own defect
    reproduced there — `external_blocker`, the item excused, the real cause
    never reported (#1436: a 0-row `kg.sqlite` provisioned by the round's own
    test). The probe therefore swaps in a data root of its own under `scratch`,
    fresh per probe and removed with the worktree. `HOME` stays the round's:
    the round home is a symlink farm over the real one, so the only residue it
    can hold is `lloyd-data`, and that is the directory being replaced.
    """
    if not node_ids:
        return set(), "no node ids to probe"
    wt = scratch / "baseline"
    shutil.rmtree(wt, ignore_errors=True)
    probe_data = scratch / "baseline-data"
    shutil.rmtree(probe_data, ignore_errors=True)
    r = W.git(live_root, "worktree", "add", "--detach", "-q", str(wt), base)
    if r.returncode != 0:
        return set(), f"baseline worktree failed: {r.stderr.strip()[:200]}"
    try:
        # Probe by FILE, never by node id. Handed a node id that does not exist
        # at the base — a test the round itself added — pytest exits 4 with
        # `ERROR: not found:` and runs *nothing*, so one new test would hide
        # every pre-existing failure beside it and the probe would report a red
        # tree as green. Files the round added are skipped for the same reason
        # and are new by construction.
        files = []
        for nid in node_ids:
            f = nid.split("::", 1)[0]
            if f not in files and (wt / f).exists():
                files.append(f)
        if not files:
            return set(), "none of the failing files exist at base"
        probe_env = dict(env or {})
        if probe_env:
            probe_env["PYTHONPATH"] = str(wt)
        if "LLOYD_DATA" in probe_env:
            probe_data.mkdir(parents=True, exist_ok=True)
            probe_env["LLOYD_DATA"] = str(probe_data)
        # `--no-header -p no:cacheprovider`: the probe must not write a
        # .pytest_cache into a tree it is about to delete, and must not read
        # one written by the candidate run.
        r = _run([str(python), "-m", "pytest", "-q", "--no-header",
                  "-p", "no:cacheprovider", "--continue-on-collection-errors",
                  "-m", TESTS_MARK_EXPR, *files],
                 cwd=wt, env=probe_env or None, timeout=EXTERNAL_PROBE_TIMEOUT)
        text = r.stdout + r.stderr
        failed = set(_failed_node_ids(text))
        # "pytest ran and found nothing pre-existing" and "pytest never ran"
        # both come back as an empty set, and only one of them is an answer.
        # A missing interpreter, a broken venv or a usage error all exit
        # non-zero with no summary — and reporting that as "0 already failing"
        # blames the round for breakage it may not own, silently, which is the
        # failure this whole probe exists to stop. Same verdict either way
        # (no exemption), but the note has to say which one happened.
        if not failed and not _parse_pytest_summary(text)["collected"]:
            tail = " | ".join(text.strip().splitlines()[-3:])[:200]
            return set(), (f"baseline probe INCONCLUSIVE at base {base[:8]} — "
                           f"pytest produced no summary (rc={r.returncode}): {tail}")
        where = " in a fresh data root" if "LLOYD_DATA" in probe_env else ""
        return failed, (f"probed {len(files)} file(s) at base {base[:8]}{where}: "
                        f"{len(failed)} already failing")
    except subprocess.TimeoutExpired:
        return set(), f"baseline probe timed out after {EXTERNAL_PROBE_TIMEOUT:.0f}s"
    except Exception as exc:
        return set(), f"baseline probe failed: {type(exc).__name__}: {exc}"
    finally:
        W.git(live_root, "worktree", "remove", "--force", str(wt))
        W.git(live_root, "worktree", "prune")
        shutil.rmtree(wt, ignore_errors=True)
        shutil.rmtree(probe_data, ignore_errors=True)


def _probe_conclusive(note: str) -> bool:
    """Whether `_failures_at_base` actually answered. Every non-answer — a
    worktree that would not create, a timeout, a crash, a pytest that produced
    no summary — returns an empty set too, and only its note tells them apart.
    A non-answer grants nothing (the `tests` rung fails closed on it) and is
    never written into the red set."""
    n = str(note or "")
    return n.startswith("probed ") or n == "none of the failing files exist at base"


# How long a per-base red set (`state.read_red_set`) stands in for the base
# probe. `automod.gate.red_set_max_age_s` overrides it.
RED_SET_MAX_AGE_S = 6 * 3600


def _gate_cfg(key: str, default):
    """`automod.gate.<key>` from config, or the default. Never raises."""
    try:
        from app.config import CONFIG
        return ((CONFIG.get("automod") or {}).get("gate") or {}).get(key, default)
    except Exception:
        return default


class _ReviewPrefetch:
    """One grading turn started beside the tests rung (`Gate._start_review_prefetch`).

    The snapshot it graded is owned here until the rung joins it (the rung's
    post step drops it) or the gate throws it away (`abandon`). The lock
    closes the one race that matters: a thread still creating its checkout
    when the gate gives up must drop that checkout itself, or it leaks.
    """

    def __init__(self, key: dict):
        self.key = key
        self.done = threading.Event()
        self.bundle: dict | None = None
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._abandoned = False
        self._snapshot: Path | None = None
        self._session_id = ""

    def adopt_snapshot(self, snap: Path | None) -> bool:
        """False when the gate already gave up: the caller drops `snap`."""
        with self._lock:
            if self._abandoned:
                return False
            self._snapshot = snap
            return True

    def set_session(self, session_id: str) -> None:
        with self._lock:
            self._session_id = session_id
            late = self._abandoned
        if late:
            Gate._cancel_grader(session_id)

    def abandon(self) -> tuple[Path | None, str]:
        """Mark it thrown away; hand back the snapshot to drop and the session."""
        with self._lock:
            self._abandoned = True
            snap, self._snapshot = self._snapshot, None
            return snap, self._session_id


class Gate:
    def __init__(self, round_id: str, worktree: Path, base: str, *,
                 live_root: Path | None = None, skip_smoke: bool = False,
                 item_id: int | None = None):
        self.round_id = round_id
        self.worktree = Path(worktree)
        self.base = base
        self.live = live_root or LIVE_ROOT
        self.skip_smoke = skip_smoke
        # The backlog item this round implements, when the round said so
        # (`automod_start(item_id=…)`, recorded in run_spec.yaml). It is what
        # the review rung grades against; without it the rung records a skip.
        self.item_id: int | None = int(item_id) if item_id else None
        self._smoke_skip_reason = "not requested"
        self._smoke_skip_refused = ""
        self.python = self.live / ".venvs" / "lloyd" / "bin" / "python"
        self.home_isolation = "not requested"
        self.report = GateReport(round_id=round_id, base=base,
                                 head=W.head(self.worktree) or "")

    def _child_env(self, root: Path | None = None, *,
                   isolate_home: bool = False, live_data: bool = False) -> dict:
        """Environment for rungs that execute CANDIDATE code outside the canary.

        Only the canary redirected the self-modification state dir. The static,
        tests and venv rungs ran candidate code against the LIVE
        `~/.local/state/lloyd-automod/`, so a candidate test that forgot its
        isolation fixture could write a real `BROKEN` or `promotions-halted`
        flag, or append to the production audit trail — from inside the very
        gate that is supposed to be read-only judgment. Point them at scratch.

        `LLOYD_VOICE_ALERTS=0` for the same class of reason: `tests/conftest.py`
        sets it precisely because a test process that outlives itself will
        otherwise talk to the room.

        `isolate_home` points `HOME` at `<round>/home`, which makes
        `Path.home()/"lloyd"` the worktree — the whole reason `worktree.py` lays
        a round out that way, and a thing this function did not do until
        2026-09-22. It is ON for the rungs that execute CANDIDATE TEST CODE
        (`tests`, the base probe, the flake re-confirm) and off elsewhere: the
        other rungs run scripts this repo controls, and two of them
        (`tool_choice`) deliberately work from the live tree. Fails open to the
        real home, loudly and on the record — `self.home_isolation` rides onto
        the tests rung's data, because a silent downgrade here is
        indistinguishable from the isolation working.

        `LLOYD_DATA` is the round's own data root on every call, whatever
        `isolate_home` says: runtime data lives outside the tree, so the
        worktree no longer isolates it by being where it is. `live_data` leaves
        it unset for the live-tree scripts that write live baselines on purpose
        (`tool_choice`), which then resolve production the way the backend does.
        """
        scratch = W.round_dir(self.round_id) / "gate-state"
        (scratch / "automod").mkdir(parents=True, exist_ok=True)
        (scratch / "guardian").mkdir(parents=True, exist_ok=True)
        home = Path.home()
        if isolate_home:
            try:
                home = W.ensure_round_home(self.round_id)
                self.home_isolation = f"round home ({home})"
            except Exception as exc:
                self.home_isolation = (f"FELL BACK to the real home "
                                       f"({type(exc).__name__}: {exc})")
                print(f"[warn] {self.home_isolation}: candidate tests can reach "
                      f"{Path.home() / 'lloyd'} through Path.home()")
        env = {
            "PATH": "/usr/bin:/bin", "HOME": str(home),
            "PYTHONPATH": str(root or self.worktree),
            "LLOYD_AUTOMOD_STATE": str(scratch / "automod"),
            "LLOYD_GUARDIAN_STATE": str(scratch / "guardian"),
            "LLOYD_VOICE_ALERTS": "0",
        }
        if not live_data:
            data = W.round_data_root(self.round_id)
            data.mkdir(parents=True, exist_ok=True)
            env["LLOYD_DATA"] = str(data)
        return env

    # ── rung reuse ─────────────────────────────────────────────────────
    #
    # A gate with review runs 7-12 minutes and every re-gate replays the whole
    # ladder, so a 60-minute round gets about one fix cycle. But the ladder
    # short-circuits at the first failure, so after a review refusal only the
    # rungs BEFORE review have a cached pass — `frontend` and `tests`, about
    # 146 s. The larger win is the promoter's rebase chase, which re-runs the
    # ladder twice more against a moved base for a diff that has not changed.
    #
    # Four conditions, all of them about "is this still the same question":
    # the config allows it, the base has not moved (a rebase changes what the
    # diff MEANS), the entry is fresh, and the cached head is an ancestor of
    # this one. Per-rung rules then ask whether the delta since that head
    # could have changed the answer.
    REUSE_MAX_AGE_S = 3600

    def _reuse_path(self) -> Path:
        return W.round_dir(self.round_id) / "gate-state" / "rung_cache.json"

    def _reuse_load(self) -> dict:
        try:
            return json.loads(self._reuse_path().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _reuse_save(self, name: str, data: dict) -> None:
        """Record a genuine pass. Wholly guarded: a cache is not the gate.

        Every attribute this touches is one a caller could have left off —
        the test stubs build a Gate without a worktree — and a rung that
        passed must never be reported as failed because its bookkeeping did.
        """
        if not _gate_cfg("reuse_rungs", True):
            return
        try:
            head = W.head(self.worktree) or self.report.head or ""
            if not head:
                return
            cache = self._reuse_load()
            cache[name] = {"base": self.base, "head": head, "ts": time.time(),
                           "data": data or {}}
            path = self._reuse_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — a cache is not the gate
            print(f"[warn] could not write the rung cache: {exc}")

    def _reuse(self, name: str) -> tuple[dict, str] | None:
        """The cached pass for `name`, if it still answers the question.

        Guarded end to end for the same reason `_reuse_save` is: a failure to
        READ the cache must run the rung, never skip it.
        """
        try:
            return self._reuse_inner(name)
        except Exception as exc:  # noqa: BLE001 — fail toward running it
            print(f"[warn] rung cache unreadable for {name}: {exc}")
            return None

    def _reuse_inner(self, name: str) -> tuple[dict, str] | None:
        if not _gate_cfg("reuse_rungs", True):
            return None
        if name in ("preflight", "static", "review"):
            # `preflight` rebases and must always run; `static` is
            # milliseconds; `review` has its own patch-id reuse, which is a
            # stronger test than "no relevant file changed".
            return None
        entry = self._reuse_load().get(name)
        if not isinstance(entry, dict):
            return None
        head = W.head(self.worktree) or self.report.head or ""
        old_head = str(entry.get("head") or "")
        if not head or not old_head:
            return None
        if str(entry.get("base") or "") != self.base:
            return None      # a rebase changed what the diff means
        max_age = float(_gate_cfg("reuse_max_age_s", self.REUSE_MAX_AGE_S))
        if time.time() - float(entry.get("ts") or 0) > max_age:
            return None
        if old_head != head:
            anc = _run(["git", "merge-base", "--is-ancestor", old_head, head],
                       cwd=self.worktree, timeout=30)
            if anc.returncode != 0:
                return None
        delta = [p for p in _run(
            ["git", "diff", "--name-only", f"{old_head}..{head}"],
            cwd=self.worktree, timeout=60).stdout.splitlines() if p.strip()]
        why = self._reuse_rule(name, delta)
        if why is None:
            return None
        return entry.get("data") or {}, why

    @staticmethod
    def _reuse_rule(name: str, delta: list[str]) -> str | None:
        """Why this rung's answer cannot have changed, or None."""
        if not delta:
            return "nothing committed since"
        if name == "frontend":
            if any(p.startswith("web/") for p in delta):
                return None
            return "no web/ path in the delta"
        if name == "venv":
            if any(p in ("requirements.txt", "requirements.lock") for p in delta):
                return None
            return "no requirements path in the delta"
        if name in ("canary_boot", "canary_smoke", "drill"):
            # A canary boots the candidate: any code change invalidates it.
            if all(TP.is_test_path(p) or p.endswith(".md") for p in delta):
                return "only tests and docs in the delta"
            return None
        if name == "prompt_surface":
            if any(p in ("prompt_builder.py", "prefetch.py")
                   or p.rsplit("/", 1)[-1] in ("SOUL.md", "MEMORY.md", "USER.md")
                   for p in delta):
                return None
            return "no prompt-surface path in the delta"
        # `tests` is never reused bare — see `rung_tests`.
        return None

    # ── driver ─────────────────────────────────────────────────────────
    # How many node ids a gate ledger event carries per list.
    EVENT_ID_CAP = 50
    # Rungs whose SKIP writes no ledger row (`gate.json` still lists them).
    # A skipped `frontend`, `prompt_surface`, `venv` or `canary_smoke` is the
    # common case — ~1,900 zero-second rows a week that say only "not this
    # round" — and nothing reads them: `backlog._last_gate_per_round` keys a
    # failed run on its failing rung and a full pass on an ok `drill` row, so
    # `drill` is never on this list, skipped or not.
    QUIET_SKIP_RUNGS = frozenset({"frontend", "prompt_surface", "venv", "canary_smoke"})

    def _quiet_skip(self, name: str, data: dict, ok: bool = True) -> bool:
        return (ok and name in self.QUIET_SKIP_RUNGS
                and (data or {}).get("skipped") is True)

    def _rung(self, name: str, fn) -> bool:
        started = time.time()
        reused = self._reuse(name)
        if reused is not None:
            data, why = reused
            data = {**data, "reused": True}
            if name == "canary_smoke":
                # No turn ran: the cached trace describes an older commit, and
                # carrying it forward would read as this build's (#828).
                data = {k: v for k, v in data.items()
                        if k not in CS.TRACE_KEYS
                        and k not in ("trace_diff", "trace_baseline")}
                data.update(has_trace=False, no_trace_reason="reused")
            res = RungResult(name, True, f"REUSED ({why})", 0.0, data)
            self.report.rungs.append(res)
            if not self._quiet_skip(name, data):
                S.append_event({"event": "gate", "round_id": self.round_id, "rung": name,
                                "ok": True, "detail": res.detail, "reused": True,
                                "skipped": data.get("skipped") is True, "seconds": 0.0})
            print(f"[PASS] {name} (0.0s) {res.detail}")
            return True
        try:
            ok, detail, data = fn()
        except Exception as exc:
            # An erroring rung is a FAILED rung. Reading it as "didn't fail"
            # would silently remove a check, and there is no human tier here.
            ok, detail, data = False, f"{type(exc).__name__}: {exc}", {}
        res = RungResult(name, ok, str(detail)[:2000], time.time() - started, data or {})
        self.report.rungs.append(res)
        # `external_blocker` rides on the event, not only in the report:
        # `gate.json` lives in the round dir and is deleted with the worktree,
        # while `backlog.implemented_ids` has to answer "was this round's
        # failure its own fault?" long after the round is gone.
        event = {"event": "gate", "round_id": self.round_id, "rung": name,
                 "ok": ok, "detail": res.detail[:500],
                 # `is True`, not truthy: a rung's data is its own shape and
                 # an integer count in a key of this name is how the `tests`
                 # rung came to record itself skipped.
                 "skipped": (res.data or {}).get("skipped") is True,
                 "seconds": round(res.seconds, 2)}
        if (res.data or {}).get("external_blocker"):
            event["external_blocker"] = True
            event["external_failures"] = (res.data or {}).get("external_failures", [])
        # Whose failures a `tests` pass (or a mixed failure) carried, for the
        # same reason: the scorecard counts red-tree passes, and the next round
        # is told which ids are not its own, long after `gate.json` is gone.
        for key in ("pre_existing_failures", "flaky_node_ids"):
            ids = (res.data or {}).get(key)
            if ids:
                event[key] = list(ids)[:self.EVENT_ID_CAP]
        if (res.data or {}).get("red_tree_item"):
            event["red_tree_item"] = res.data["red_tree_item"]
        # The review rung's two verdicts ride the event for the same reason:
        # `backlog.implement_outcomes` reads them long after the round dir is
        # gone, and the findings are what the next round is told.
        for key in ("review_retry", "review_premise_unsound"):
            if (res.data or {}).get(key):
                event[key] = True
        for key in ("review_findings", "review_summary", "review_attempt"):
            if (res.data or {}).get(key):
                event[key] = (res.data or {})[key]
        # The vet's record rides the event because it is the record of a rung
        # that never fails, and #679's soak asks a question only the ledger can
        # answer months later: "did the vet run on every landing, and what did
        # it see?". `gate.json` is copied by the landing and deleted with the
        # worktree, so a non-blocking observation preserved only there has no
        # denominator. Compact by construction — status, counts, labels, the
        # totals — so an always-present rung does not bloat the ledger.
        if (res.data or {}).get("vet"):
            v = res.data["vet"]
            event["vet"] = {"status": v.get("status"),
                            "observe_only": True,
                            "counts": v.get("counts") or {},
                            "labels": v.get("labels") or [],
                            "totals": v.get("totals") or {}}
        if not self._quiet_skip(name, res.data or {}, ok):
            S.append_event(event)
        if ok:
            self._reuse_save(name, res.data)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({res.seconds:.1f}s) {res.detail[:160]}")
        return ok

    def run(self) -> GateReport:
        # `canary_smoke` is ALWAYS on the ladder. It used to be omitted when
        # `skip_smoke` was passed, so a skipped rung left no trace at all: the
        # report listed seven rungs and a reader had to know the eighth existed
        # to notice it was gone. A skip is now a recorded rung that says so.
        # `vet` is second, immediately behind the scope check that shares its
        # inputs (a resolved base, the enumerated change set) and costs about as
        # much: it is the only rung that reads the change set itself rather than
        # the behaviour it produces, so it runs before anything is executed —
        # before compileall imports the candidate, long before a canary boots
        # it. Observe-only today, so its position is about when the record is
        # written, not about what can fail.
        ladder = [("preflight", self.rung_preflight), ("vet", self.rung_vet),
                  ("static", self.rung_static),
                  ("frontend", self.rung_frontend),
                  ("tests", self.rung_tests),
                  ("prompt_surface", self.rung_prompt_surface),
                  ("review", self.rung_review),
                  ("venv", self.rung_venv),
                  ("canary_boot", self.rung_canary_boot),
                  ("canary_smoke", self.rung_canary_smoke),
                  ("drill", self.rung_drill)]

        # And the skip is refused while the engine is answering. The flag is
        # for a machine with no vLLM, not for a round that would rather not
        # spend 30 seconds on a real turn.
        if self.skip_smoke:
            up, why = engine_reachable(self.live)
            self._smoke_skip_reason = why
            if up:
                self.skip_smoke = False
                self._smoke_skip_refused = why

        # Two gates at once (`workers.sources.autocode.max_inflight` > 1, or a
        # person's CLI round beside the loop's) share two things no worktree
        # isolates: the canary's fixed ports, which the drill reuses, and the
        # machine the full suite runs on — tests that bind loopback ports and
        # a ~6 minute CPU-bound run that two copies would each stretch past
        # the other's timeouts. Each is queued for, not refused: the wait is
        # minutes, and a `port already in use` failure would spend a review
        # attempt on a round that did nothing wrong. The review rung, the
        # long one, overlaps freely.
        ladder = [(n, self._serialized(n, f)) for n, f in ladder]
        self._canary: C.Canary | None = None
        self._canary_lock: S.Lock | None = None
        self._review_prefetch = None
        try:
            for name, fn in ladder:
                if name == "tests":
                    # The review's grading turn starts now and is joined at
                    # the review rung (`_take_review_prefetch`).
                    self._start_review_prefetch()
                if not self._rung(name, fn):
                    self.report.ok = False
                    return self.report
            self.report.ok = True
        finally:
            # A gate that stopped before the review rung judged nothing: the
            # prefetch leaves no event and spends no attempt.
            self._discard_review_prefetch("the gate stopped before the review rung")
            if getattr(self, "_canary", None):
                try:
                    self._canary.stop()
                except Exception:
                    pass
            if self._canary_lock is not None:
                self._canary_lock.release()
                self._canary_lock = None
        return self.report

    # How long a rung queues for a resource another gate holds: a full suite
    # is ~6 min and a canary + drill under two, so this is several of either.
    SERIAL_MAX_WAIT = 2400.0

    def _serialized(self, name: str, fn):
        """`fn`, holding the machine resource the rung `name` needs.

        `tests` holds its lock for the rung. `canary_boot` takes the canary
        lock and KEEPS it — smoke and the drill use the same ports — until
        `run`'s `finally` releases it. A reused rung never reaches here
        (`_rung` answers it first), so a reuse never queues.
        """
        if name == "tests":
            def _tests():
                waited = time.time()
                with_lock = S.Lock(S.GATE_TESTS_LOCK_PATH, owner=f"gate-{self.round_id}")
                with_lock.acquire_wait(self.SERIAL_MAX_WAIT)
                waited = time.time() - waited
                try:
                    ok, detail, data = fn()
                finally:
                    with_lock.release()
                return ok, detail, {**(data or {}), "lock_wait_s": round(waited, 1)}
            return _tests
        if name in ("canary_boot", "canary_smoke", "drill"):
            # All three, not only the boot: a reused `canary_boot` never runs
            # its wrapper, and the drill binds the ports whoever booted.
            def _ports():
                waited = time.time()
                if self._canary_lock is None:
                    lock = S.Lock(S.GATE_CANARY_LOCK_PATH, owner=f"gate-{self.round_id}")
                    lock.acquire_wait(self.SERIAL_MAX_WAIT)
                    self._canary_lock = lock
                waited = time.time() - waited
                ok, detail, data = fn()
                return ok, detail, {**(data or {}), "lock_wait_s": round(waited, 1)}
            return _ports
        return fn

    # ── rungs ──────────────────────────────────────────────────────────
    def rung_preflight(self):
        if S.is_halted():
            return False, f"promotions halted: {S.HALTED_PATH}", {}
        if S.is_broken():
            return False, f"guardian is in a BROKEN state: {S.BROKEN_PATH}", {}

        head = W.head(self.worktree)
        if not head:
            return False, "cannot read the worktree HEAD", {}

        data: dict = {}
        live_head = _run(["git", "-C", str(self.live), "rev-parse", "HEAD"]).stdout.strip()
        if live_head != self.base:
            # The tree is shared: a human commits to `main` while a round is
            # open. Until 2026-09-09 this was a refusal — "something landed
            # under you; abort and re-cut" — which threw the round's diff away
            # to be reapplied by hand onto a base one commit newer. The rebase
            # is that reapplication and the rest of the ladder is the retest:
            # every rung below runs against the round's change ON TOP of what
            # landed, which is the only thing worth testing. Only a conflict
            # fails, and it names the files.
            # `upstream`: replay only the round's own commits, never a base
            # the guardian reset away (see `W.rebase_onto`).
            ok, why, conflicts = W.rebase_onto(self.worktree, live_head, upstream=self.base)
            if not ok and conflicts:
                return False, (f"live HEAD moved: {live_head[:8]} != base "
                               f"{self.base[:8]}, and rebasing onto it conflicts in "
                               f"{conflicts} — resolve in the worktree and gate again "
                               f"(the rebase was aborted; the worktree is as it was)"), {
                                   "external_blocker": True, "conflicts": conflicts,
                                   "moved_to": live_head}
            if not ok:
                # The worktree itself refused: uncommitted changes. That is
                # the round's own state, and no exemption.
                return False, f"live HEAD moved: {live_head[:8]} != base {self.base[:8]}, but {why}", {
                    "moved_to": live_head}
            old_base, old_head = self.base, head
            self.base = live_head
            head = W.head(self.worktree) or ""
            if not head:
                return False, "cannot read the worktree HEAD after rebase", {}
            data["rebased"] = {"from": old_base, "onto": live_head,
                               "old_head": old_head, "new_head": head}
        # `gate.json` is what `land` reads its base and head from, and after a
        # rebase both have moved.
        self.report.base = self.base
        self.report.head = head

        anc = _run(["git", "-C", str(self.live), "merge-base", "--is-ancestor",
                    self.base, head])
        if anc.returncode != 0:
            return False, "candidate is not a descendant of base — not a fast-forward", data
        if W.has_merge_commits(self.live, self.base, head):
            return False, "candidate contains merge commits", data

        changed = W.changed_paths(self.worktree, self.base)
        self.report.changed_paths = changed
        if not changed:
            return False, "no changes to promote", data

        # Uncommitted edits in production are tolerated when they are not in
        # the round's diff: a fast-forward of disjoint paths leaves them
        # exactly where they are, and refusing on any dirt at all is what cost
        # #447 its 587 lines while an unrelated file sat modified for an hour.
        # Overlap is the real hazard — two writers on one file — and the
        # refusal names it. Still not the round's fault, so still no attempt
        # spent.
        dirty = W.dirty_paths(self.live)
        overlap = sorted(set(dirty) & set(changed))
        if overlap:
            # The wording is part of this refusal's contract (#1038). It used to
            # tell the round to get that edit out of the way and retry, which in
            # practice meant the live checkout's one global LIFO working-tree
            # stack — a structure every author of this checkout shares, so a
            # round that writes there is rearranging somebody else's in-flight
            # work. That has already cost somebody their only copy: on
            # 2026-09-11 a round implementing an unrelated item popped #573's
            # recovered 136-line diff off the stack and it had to be rebuilt by
            # hand. The rule, with the command spelled out, is the read-only
            # bullet in the automod skill's Boundaries. And nothing in this rung
            # needs a clean live tree — the paragraph above tolerates disjoint
            # dirt — so there is never anything worth moving.
            return False, (f"live tree has uncommitted edits in paths this round also "
                           f"changes: {overlap} — two writers on one file. Report the "
                           f"paths and who is editing them: the live edit is not yours "
                           f"to commit and not yours to move out of the tree, and this "
                           f"gate needs no clean live tree to run again"), {
                               **data, "external_blocker": True,
                               "dirty_paths": dirty[:20], "overlap": overlap}
        if dirty:
            data["dirty_tolerated"] = dirty[:20]

        ok, reason, buckets = spec.check_scope(changed)
        if not ok:
            return False, reason, {**data, "buckets": buckets}

        if self.item_id:
            # Zero-cost fail-fasts for a round that has a contract. A round
            # with no clauses has nothing the review rung can grade; a
            # Python change with no test in the diff cannot have pinned any
            # clause (the implement prompt demands a pinning test). Both are
            # the round's own, and both are cheaper to learn here than after
            # a 77 s test run.
            from scripts.automod import review as RV
            contract = RV.item_contract(self.item_id)
            if not contract["clauses"]:
                return False, (f"item #{self.item_id} has no acceptance clauses to judge "
                               f"against — the review rung cannot grade it"), data
            py_changed = any(p.endswith(".py") and not TP.is_test_file(p, self.worktree)
                             for p in changed)
            if py_changed and not any(TP.is_test_path(p, self.worktree) for p in changed):
                return False, (f"item #{self.item_id} has {len(contract['clauses'])} acceptance "
                               f"clause(s) and this diff changes code with no test under "
                               f"a pytest testpath — nothing pins a clause"), data
            data["item_id"] = self.item_id
            data["clauses"] = len(contract["clauses"])

        # A busy port is a stale canary only when no gate holds the canary
        # lock. With two rounds at once the usual reason is the OTHER gate's
        # canary, which this gate queues for at `canary_boot`; failing here
        # refused SM_20260917_184334 two seconds into its gate, the first
        # time two gates overlapped at this rung.
        if not _canary_lock_held():
            for port in (C.cc.BACKEND_PORT, C.cc.MCP_PORT):
                if not C.port_free(port):
                    return False, f"canary port {port} is in use (stale canary?)", data

        detail = (f"{len(changed)} file(s) in scope"
                  + (f"; {len(buckets['protected'])} protected → drill required"
                     if buckets["protected"] else ""))
        if data.get("rebased"):
            detail += (f"; rebased {data['rebased']['from'][:8]}→{live_head[:8]} — "
                       "every rung below judges the change on top of what landed")
        if dirty:
            detail += f"; tolerating {len(dirty)} uncommitted live path(s) outside this diff"
        return True, detail, {"buckets": buckets, **data}

    def rung_vet(self):
        """OBSERVE-ONLY structural vet of the change set (#679).

        Every other rung on this ladder asks whether the system still behaves;
        this one asks whether the change set is structurally what the round
        claims to have done — files emptied relative to base, binary blobs
        dropped into the tree, a diff far outside anything this repo lands. It
        is stdlib, deterministic, sub-second, and it blocks nothing: the
        violation list rides onto the report and the ledger event, and a person
        decides after a soak whether any of the three checks earns a `False`.
        (Backlog #679's acceptance: >=20 landings and <5% false flags first.)

        It never reports "clean" for a pass that could not run: an
        `unevaluated` result says `VET UNEVALUATED` in the detail and carries
        `status: unevaluated` in the data, which is why the soak's denominator
        is readable from the ledger rather than assumed.
        """
        res = V.vet_change_set(self.base, self.worktree)
        data = {"vet": res.to_dict()}
        totals = res.totals or {}
        counts = {k: sum(1 for v in res.violations if v.kind == k)
                  for k in (V.EMPTY_FILE, V.BINARY_ARTIFACT, V.DIFF_TOO_LARGE)}
        data["vet"]["counts"] = {k: n for k, n in counts.items() if n}
        # Two fields the SOAK needs and a verdict never does: `observe_only` so
        # no reader infers enforcement from a green rung, and `labels` — one
        # short string per finding, no prose — so the soak can tally kinds
        # across rounds without parsing detail text. Bounded, because a round
        # that trips the size ceiling writes one entry, not forty.
        data["vet"]["observe_only"] = True
        data["vet"]["labels"] = [v.label() for v in res.violations[:20]]
        if not res.evaluated:
            return True, f"observe-only: VET UNEVALUATED ({res.reason})", data
        if not res.violations:
            return True, (f"observe-only: clean — {totals.get('files', 0)} file(s), "
                          f"{totals.get('changed_lines', 0)} changed line(s) of "
                          f"{totals.get('max_diff_lines', 0)} allowed"), data
        listed = "; ".join(v.label() for v in res.violations[:6])
        more = f" (+{len(res.violations) - 6} more)" if len(res.violations) > 6 else ""
        return True, (f"observe-only: {len(res.violations)} violation(s) — "
                      f"{listed}{more}"), data

    def rung_static(self):
        r = _run([str(self.python), "-m", "compileall", "-q", str(self.worktree)], timeout=300)
        if r.returncode != 0:
            return False, f"compileall failed: {(r.stdout + r.stderr)[-800:]}", {}

        # The single highest-value cheap check: an import-time failure is the
        # number one way a self-modification bricks the boot, and this catches
        # it in seconds without binding a port or running a startup hook.
        env = self._child_env()
        imp = _run([str(self.python), "-c", "import server, agent_mcp.main"],
                   cwd=self.worktree, env=env, timeout=180)
        if imp.returncode != 0:
            return False, f"import smoke failed: {(imp.stdout + imp.stderr)[-800:]}", {}

        changed_py = [p for p in self.report.changed_paths if p.endswith(".py")]
        if not changed_py:
            return True, "compiled; imports clean; no python changed", {}

        head_findings = _pyflakes(self.python, self.worktree, changed_py)
        with_base = Path(_run(["mktemp", "-d"]).stdout.strip())
        try:
            # Same files as they stood at the merge base, so a pre-existing
            # finding on an untouched line cannot fail the gate. The tree has
            # 69 such findings; an absolute bar would be disabled within a day.
            for rel in changed_py:
                blob = _run(["git", "-C", str(self.live), "show", f"{self.base}:{rel}"])
                if blob.returncode == 0:
                    dest = with_base / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(blob.stdout, encoding="utf-8")
            base_findings = _pyflakes(self.python, with_base,
                                      [p for p in changed_py if (with_base / p).exists()])
        finally:
            shutil.rmtree(with_base, ignore_errors=True)

        new = head_findings - base_findings
        if new:
            return False, f"{len(new)} new pyflakes finding(s): {sorted(new)[:5]}", {
                "new": sorted(new)}
        return True, f"compiled; imports clean; no new pyflakes ({len(head_findings)} pre-existing)", {}

    def rung_frontend(self):
        """Type-check and build the frontend when the diff touches it.

        Two checks, and neither alone is enough. `tsc --noEmit` sees type
        regressions that a bundler happily ships; `vite build` sees what tsc
        does not — a missing asset, an unresolvable import, a plugin that
        rejects the tree. Both run from the worktree against the LIVE tree's
        `node_modules` (636 MB, gitignored, and identical by construction
        because `package.json` and the lockfile are denied paths).

        tsc is judged as a delta, like pyflakes in rung `static`: the tree
        carried three pre-existing errors on the day this rung was written, and
        an absolute bar would have been switched off within the hour. The
        build is absolute — it passes on the live tree today and must keep
        passing.

        There is deliberately no runtime probe to pair this with. A broken
        `src` change is a browser-side error the Vite dev server serves with a
        200, so a guardian probe of :5173 would measure liveness of a process
        the change cannot kill and nothing the change can break. The build is
        where a frontend change can be verified, so the build is the gate.
        """
        changed_web = [p for p in self.report.changed_paths if p.startswith("web/")]
        if not changed_web:
            return True, "no frontend changed", {"skipped": True,
                                                "reason": "no web/ path"}
        web = self.worktree / "web"
        live_web = self.live / "web"
        nm = web / "node_modules"
        if not nm.exists():
            if not (live_web / "node_modules" / ".bin" / "vite").exists():
                return False, "live web/node_modules has no vite — run npm install in ~/lloyd/web", {}
            nm.symlink_to(live_web / "node_modules")
        head = _tsc_findings(web)
        base = _tsc_findings(live_web)
        new = head - base
        if new:
            return False, (f"{sum(new.values())} new tsc error(s): "
                           f"{sorted(new)[:4]}"), {"new": sorted(new)}
        out_dir = Path(_run(["mktemp", "-d"]).stdout.strip())
        try:
            ok, tail = _vite_build(web, out_dir)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
        if not ok:
            return False, f"vite build failed: {tail}", {}
        tested, vt_tail = _vitest_run(web)
        if tested is False:
            return False, f"vitest failed: {vt_tail}", {"changed_web": changed_web}
        unit = "vitest ok" if tested else f"vitest SKIPPED ({vt_tail})"
        return True, (f"tsc: no new errors ({sum(head.values())} pre-existing); "
                      f"vite build ok; {unit}; {len(changed_web)} frontend file(s)"), {
                          "changed_web": changed_web, "vitest": tested,
                          **({} if tested else {"vitest_skipped": vt_tail})}

    # Files whose edit changes what the model is *told*, rather than what the
    # code does. A behavioural regression here passes every other rung: the
    # tests are green, tsc is clean, the canary boots, and the model has
    # quietly stopped reaching for `http_search`.
    PROMPT_SURFACE_PATHS = (
        "prompt_builder.py", "prefetch.py",
    )
    PROMPT_SURFACE_VAULT = ("SOUL.md", "MEMORY.md", "USER.md")

    def _touches_prompt_surface(self) -> bool:
        for path in self.report.changed_paths:
            base = path.rsplit("/", 1)[-1]
            if path in self.PROMPT_SURFACE_PATHS or base in self.PROMPT_SURFACE_PATHS:
                return True
            if base in self.PROMPT_SURFACE_VAULT:
                return True
        return False

    def rung_prompt_surface(self):
        """Scored behavioural check, but only when the prompt surface moved.

        This used to be four commands in the autocode prompt, and that placed
        it exactly wrong. The model ran it from inside its own turn, against
        the live engine, while its own 150k-token round was the other tenant —
        20 primary queries beside the round's own iterations, twice, evicting
        the round's prefix both times. The 2026-09-11 sessions show 0.14-0.85M
        tokens of re-prefill each and this is part of why.

        As a rung it runs while the model is idle in `automod_gate_wait`,
        which is the one window in a round when nothing else of its own is on
        the engine.

        Exit codes come from the live script, not from a copy of its contract
        here: `compare_tool_choice.py` exits 0 pass / 1 regression / 2 nothing
        to compare against, and #875's kept branch adds a 3. Anything non-zero
        fails the rung and the message quotes what it said.
        """
        if not self._touches_prompt_surface():
            return True, "no prompt-surface path in the diff", {"skipped": True,
                                                                "reason": "not touched"}
        # Labelled by item where the round has one. `--label item377` reads
        # back as the run that judged item 377, months later, in a directory
        # of bare timestamps; the round id is the fallback.
        label = f"item{self.item_id}" if self.item_id else f"gate-{self.round_id}"
        # From the LIVE tree, not the worktree: the eval writes its baseline
        # next to the script, and a baseline written inside a worktree is
        # deleted with it (SM_20260908_165950's was).
        run = _run([str(self.python), "eval/run_tool_choice_eval.py",
                    "--label", label],
                   cwd=self.live, env=self._child_env(live_data=True), timeout=1800)
        if run.returncode != 0:
            tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-15:])
            return False, f"tool-choice eval failed to run: {tail}", {"label": label}
        cmp_ = _run([str(self.python), "eval/compare_tool_choice.py",
                     "--label", label],
                    cwd=self.live, env=self._child_env(live_data=True), timeout=600)
        text = (cmp_.stdout + cmp_.stderr).strip()
        tail = "\n".join(text.splitlines()[-15:])
        data = {"label": label, "compare_exit": cmp_.returncode}
        if cmp_.returncode != 0:
            # Name every code the live script can return. #875 added exit 3, and
            # a round handed an unlabelled "exit 3" reads it as just another
            # regression -- which is the one verdict it is trained to argue
            # past. Exit 3 is not an argument to be had: the control rows moved,
            # the comparison certified nothing, and both sides must be re-run.
            # This rung is the only surface where a round reads these codes now
            # (the copy #875's first round found in workers/sources/autocode.py
            # is gone, its prompt no longer mentions the eval at all).
            return False, (
                f"tool-choice comparison exit {cmp_.returncode} "
                f"(0=pass, 1=regression of the change, 2=nothing to compare "
                f"against which is not a pass, 3=INSTRUMENT FAILURE: the control "
                f"set moved so this comparison certified nothing; re-run both "
                f"sides, this is not a regression to argue past): {tail}"), data
        return True, f"tool-choice eval: no regression. {tail[-300:]}", data

    def _tests_delta_only(self) -> list[str] | None:
        """Test files to re-run instead of the whole suite, or None.

        `tests` is never reused bare: a cached pass says the suite was green
        at an earlier commit, and any code change since could have broken
        anything. But a delta that is ONLY test files — and no `conftest.py`
        and no `tests/_*.py` helper, either of which changes how every other
        test runs — can be answered by running those files.

        The floors (`PYTEST_MIN_COLLECTED`, `PYTEST_MIN_PASSED`) do not apply
        to a partial run and are skipped; the removed-files check still runs,
        because deleting the test that was failing is the failure mode those
        floors exist for.
        """
        entry = self._reuse_load().get("tests")
        if not isinstance(entry, dict) or not _gate_cfg("reuse_rungs", True):
            return None
        if str(entry.get("base") or "") != self.base:
            return None
        max_age = float(_gate_cfg("reuse_max_age_s", self.REUSE_MAX_AGE_S))
        if time.time() - float(entry.get("ts") or 0) > max_age:
            return None
        old_head = str(entry.get("head") or "")
        head = W.head(self.worktree) or self.report.head or ""
        if not old_head or not head or old_head == head:
            return None
        if _run(["git", "merge-base", "--is-ancestor", old_head, head],
                cwd=self.worktree, timeout=30).returncode != 0:
            return None
        delta = [p for p in _run(
            ["git", "diff", "--name-only", f"{old_head}..{head}"],
            cwd=self.worktree, timeout=60).stdout.splitlines() if p.strip()]
        if not delta:
            return None
        files = []
        for rel in delta:
            if not TP.is_test_file(rel, self.worktree):
                return None                       # any code path: full run
            base = rel.rsplit("/", 1)[-1]
            if base == "conftest.py" or base.startswith("_"):
                return None                       # changes how everything runs
            files.append(rel)
        return files or None

    # A parallel failure naming more files than this is not load, it is a
    # broken tree: re-ask the whole suite serially rather than file by file.
    PARALLEL_RETRY_MAX_FILES = 40

    def _test_workers(self) -> tuple[int, str]:
        """`(workers, why_serial)` for a FULL run of the suite.

        `automod.gate.test_workers`, and only when the venv this gate runs
        tests with can import `xdist` — a candidate venv built by the `venv`
        rung, a fresh clone before `pip install`, and a box where nobody
        installed it must all still gate, serially, and say why.
        """
        try:
            n = int(_gate_cfg("test_workers", 1) or 1)
        except (TypeError, ValueError):
            n = 1
        if n <= 1:
            return 1, ""
        probe = _run([str(self.python), "-c", "import xdist"], cwd=self.worktree,
                     env=self._child_env(), timeout=60)
        if probe.returncode != 0:
            return 1, "pytest-xdist is not importable in the gate's venv"
        return n, ""

    def _run_suite(self, only: list[str] | None):
        """Run pytest for the `tests` rung: `(completed, text, counts)`.

        **The whole suite runs in parallel; a failure is re-asked serially
        before it is believed.** The suite grew from ~4,900 tests to ~5,900 in
        the week to 2026-09-18 and its serial run from 153 s to ~600 s, on a
        32-core box, holding `gate-tests.lock` the whole time — so at depth 2
        the second gate waited another 400–470 s, and the gate outgrew the
        turn that has to wait for it (automod.md §3.2g). Eight `xdist` workers
        run it in ~76 s.

        What parallelism costs is load: a test that asserts a latency budget
        can lose it. The first trial did exactly that —
        `test_edit_diagnostics::test_a_cross_file_break_names_its_caller`
        asserts the blast-radius rail answers inside its 90 ms budget, and
        under eight workers it did not. That is a fact about the box during
        the run, not about the candidate. So every file with a failure is run
        again, serially, and THAT run is the verdict: it passes, the rung
        passes and `parallel_only_failures` names what flinched (so a flaky
        test is a number, not a mystery); it fails, and the failure is judged
        exactly as a serial run's always was — by the serial run's own node
        ids, through the same base probe. A parallel failure that names no
        file (a crashed worker, a timeout) or more than
        `PARALLEL_RETRY_MAX_FILES` re-runs the whole suite serially.

        `--dist loadfile`: one file's tests stay on one worker, in order, so
        module-scoped fixtures (a booted uvicorn, a temp repo) are built once
        per file as they always were. A partial run — the changed test files
        a re-gate gets — is seconds long and stays serial.
        """
        base_cmd = [str(self.python), "-m", "pytest", "-q", "-m", TESTS_MARK_EXPR]
        env = self._child_env(isolate_home=True)

        def serial(extra: list[str]):
            done = _run(base_cmd + list(extra), cwd=self.worktree, env=env, timeout=1800)
            out = done.stdout + done.stderr
            return done, out, _parse_pytest_summary(out)

        workers, why_serial = (1, "") if only else self._test_workers()
        if workers <= 1:
            r, text, counts = serial(list(only or []))
            counts["workers"] = 1
            if why_serial:
                counts["parallel_unavailable"] = why_serial
            return r, text, counts

        r = _run(base_cmd + ["-n", str(workers), "--dist", "loadfile"],
                 cwd=self.worktree, env=env, timeout=1800)
        text = r.stdout + r.stderr
        counts = _parse_pytest_summary(text)
        counts["workers"] = workers
        if r.returncode == 0:
            return r, text, counts
        node_ids = _failed_node_ids(text)
        files = sorted({nid.split("::", 1)[0] for nid in node_ids})
        if not files or len(files) > self.PARALLEL_RETRY_MAX_FILES:
            r2, text2, counts2 = serial([])
            counts2.update({"workers": workers, "serial_rerun": "whole suite",
                            "parallel_failures": node_ids[:50]})
            return r2, text2, counts2
        r2, text2, counts2 = serial(files)
        counts["serial_retry_files"] = files
        if r2.returncode == 0:
            counts["passed"] += counts["failed"] + counts["errors"]
            counts["failed"] = counts["errors"] = 0
            counts["parallel_only_failures"] = node_ids
            return subprocess.CompletedProcess(r.args, 0, r.stdout, r.stderr), text, counts
        # Real. The serial run names the failures; the counts stay the whole
        # suite's, with the serial run's word on how many of them failed.
        still = counts2["failed"] + counts2["errors"]
        counts["passed"] += max(0, counts["failed"] + counts["errors"] - still)
        counts["failed"], counts["errors"] = counts2["failed"], counts2["errors"]
        counts["parallel_only_failures"] = [n for n in node_ids
                                            if n not in set(_failed_node_ids(text2))]
        return r2, text2, counts

    def rung_tests(self, only: list[str] | None = None):
        # `-m "not live_vault"`: this rung judges the CANDIDATE, and a test that
        # reads the live `~/obsidian` vault judges whatever last wrote to it.
        # An hourly autoresearch promotion or a nightly reflection job can move
        # `SOUL.md` between one round and the next, so a live-vault assertion on
        # a hard rung fails the next author for the previous writer's change —
        # the `data/tool_overrides.yaml` shape that aborted three rounds in
        # fifteen hours on 2026-09-07, where the only drain the backlog had was
        # blocked by a test asserting something no diff under test controlled.
        # Those invariants are enforced where the writes happen instead:
        # `vault_round.validate` and `autoresearch.promote` both call
        # `prompt_surface.check_contract` before committing.
        only = only if only is not None else self._tests_delta_only()
        r, text, counts = self._run_suite(only)
        if r.returncode != 0:
            return self._tests_failed(text, counts, only)
        ok, detail, data = self._tests_pass(counts, only, {})
        if ok and not only:
            # A full green run: the proof the red-tree bookkeeping waits for.
            # Both halves are wholly guarded — a cache or a backlog write is
            # never the gate's verdict.
            try:
                S.write_red_set(self.base, [], by=self.round_id, merge=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] could not record the green red set: {exc}")
            healed = self._close_healed_red_tree()
            if healed:
                data["red_tree_closed"] = healed
        return ok, detail, data

    def _tests_pass(self, counts: dict, only: list[str] | None, extra: dict):
        """The pass tail of the `tests` rung, shared by a green run and a run
        whose every failure is pre-existing or flaky — so a pass on a red tree
        is held to exactly the floors a green one is.

        `extra` rides onto the data (and the detail's lead, via
        `extra["_lead"]`, which is popped): the pre-existing and flaky ids.
        """
        extra = dict(extra or {})
        lead = str(extra.pop("_lead", "") or "")
        if only:
            # A partial run answers a narrower question, so the whole-suite
            # floors below do not apply — they would fail every partial run
            # by construction. The removed-files check still runs.
            removed = [p for p in self.report.changed_paths
                       if TP.is_test_path(p, self.worktree) and not (self.worktree / p).exists()]
            if removed:
                return False, (f"test files removed by this round: {removed}"), {**counts, **extra}
            return True, (lead + f"pytest (partial, {len(only)} changed test file(s) since the "
                          f"last full run): {counts['passed']} passed, "
                          f"{counts['tests_skipped']} skipped"), {
                              **counts, **extra, "partial": True, "only": list(only)}

        # Non-negotiable under auto-landing: `pytest -q` exits 0 if the round
        # simply deleted the test that was failing.
        if counts["collected"] < PYTEST_MIN_COLLECTED:
            return False, (f"only {counts['collected']} tests collected "
                           f"(floor {PYTEST_MIN_COLLECTED}) — did the round delete tests?"), {
                               **counts, **extra}
        # ...and it exits 0 just as happily having collected everything and RUN
        # nothing. A collected-count floor alone is satisfied by a suite that
        # skipped itself wholesale, which is a one-line conftest change away.
        if counts["passed"] < PYTEST_MIN_PASSED:
            return False, (f"only {counts['passed']} tests passed "
                           f"(floor {PYTEST_MIN_PASSED}) of {counts['collected']} "
                           f"collected, {counts['tests_skipped']} skipped — the suite "
                           "was collected but not run"), {**counts, **extra}
        if counts["tests_skipped"] > PYTEST_MAX_SKIPPED:
            return False, (f"{counts['tests_skipped']} tests skipped "
                           f"(limit {PYTEST_MAX_SKIPPED}) — a round that skips its way "
                           "to green is not a round that passed"), {**counts, **extra}
        removed = [p for p in self.report.changed_paths
                   if TP.is_test_path(p, self.worktree) and not (self.worktree / p).exists()]
        if removed:
            return False, f"test files removed: {removed}", {**counts, **extra}
        flinched = counts.get("parallel_only_failures") or []
        # On the pass too, not only on the failure branches: the one reading that
        # says whether the suite ran redirected is worth nothing if it is absent
        # from every report where nothing went wrong.
        counts["home"] = self.home_isolation
        return True, (lead + f"{counts['passed']} passed, {counts['xfailed']} xfailed, "
                      f"{counts['tests_skipped']} skipped"
                      + (f" ({counts['workers']} workers)" if counts.get("workers", 1) > 1 else "")
                      + (f"; {len(flinched)} failed only under parallel load and passed "
                         f"serially: {_name_ids(flinched)}" if flinched else "")), {**counts, **extra}

    def _touched_paths(self) -> set[str]:
        """The round's own diff. `preflight` records it; a direct call of the
        rung (no preflight) reads it from git. Empty when neither can say."""
        if self.report.changed_paths:
            return set(self.report.changed_paths)
        try:
            return set(W.changed_paths(self.worktree, self.base))
        except Exception:  # noqa: BLE001
            return set()

    def _base_failures(self, probe_ids: list[str]) -> tuple[set[str], str, bool, bool]:
        """`(base_failed, note, conclusive, cached)` for `probe_ids`.

        The per-base red set answers first (`state.read_red_set`, fresh within
        `automod.gate.red_set_max_age_s`): when every id is already known to
        fail at this exact base, the throwaway-worktree probe is skipped. It
        applies only to nodes that failed at HEAD, so a stale entry can skip a
        probe, never hide a failure. A conclusive probe is merged back in; an
        INCONCLUSIVE one writes nothing.
        """
        if not probe_ids:
            return set(), "no failures outside the round's own files to probe", True, False
        try:
            entry = S.read_red_set(self.base, float(_gate_cfg("red_set_max_age_s", RED_SET_MAX_AGE_S)))
        except Exception:  # noqa: BLE001 — a cache is not the gate
            entry = None
        if entry and set(probe_ids) <= set(entry["nodes"]):
            return (set(probe_ids),
                    f"red set cached for base {self.base[:8]} (recorded by {entry.get('by') or '?'}, "
                    f"{int(time.time() - entry['ts'])}s ago): all {len(probe_ids)} already failing; "
                    f"probe skipped", True, True)
        base_failed, note = _failures_at_base(
            self.python, self.live, self.base, probe_ids,
            W.round_dir(self.round_id), self._child_env(isolate_home=True))
        conclusive = _probe_conclusive(note)
        if conclusive:
            try:
                S.write_red_set(self.base, sorted(base_failed), by=self.round_id, merge=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] could not record the red set: {exc}")
        return base_failed, note, conclusive, False

    def _files_red_tree(self) -> bool:
        """Whether this gate may write the red-tree item: the switch is on and
        the gate is judging against the production tree. A test that builds a
        Gate over a throwaway repo must never reach the real board."""
        if not _gate_cfg("file_red_tree", True):
            return False
        try:
            return Path(self.live).resolve() == LIVE_ROOT.resolve()
        except Exception:  # noqa: BLE001
            return False

    def _file_red_tree(self, pre_existing: list[str]) -> int | None:
        if not pre_existing or not self._files_red_tree():
            return None
        try:
            from scripts.automod import backlog as B
            res = B.file_red_tree_item(self.base, pre_existing, self.round_id,
                                       sorted(self._touched_paths()), live_root=self.live)
            if res:
                return int(res["item_id"])
            # Nothing written (already covered): name the open item that covers
            # them, so the round is told who owns these failures.
            items = B.open_red_tree_items()
            wanted = set(pre_existing)
            for it in items:
                covered = {str(n) for n in (B._red_tree_fm(it).get("red_tree_nodes") or [])}
                if wanted & covered:
                    return it.id
            return items[0].id if items else None
        except Exception as exc:  # noqa: BLE001 — a filing failure is never the gate
            print(f"[warn] could not file the red-tree item: {exc}")
            return None

    def _close_healed_red_tree(self) -> list[int]:
        if not self._files_red_tree():
            return []
        try:
            from scripts.automod import backlog as B
            return B.close_healed_red_tree(self.base, self.round_id, self.live,
                                           touched=sorted(self._touched_paths()))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not close a healed red-tree item: {exc}")
            return []

    def _tests_failed(self, text: str, counts: dict, only: list[str] | None):
        """Judge a red run: whose failures are these?

        Since 2026-09-24 the rung PASSES when every failure is somebody
        else's — reproduced at the round's base with the diff absent, or
        passing a repeat run (flaky, #1196) — and records them
        (`pre_existing_failures`, `flaky_node_ids`) plus the `red-tree` item
        that now owns them. It used to fail with `external_blocker`: a red tree
        "is not a tree to land onto, the guardian would judge the promotion
        against a broken baseline". The guardian never runs pytest, and 85 of
        257 landings that week restarted nothing; what the refusal actually did
        was kill 157 rounds in a week, 21 of which ever landed.

        Three rules keep that from becoming a way round the gate:
          * a failing node in a file the round's own diff touches is the
            round's, even if it also fails at base — a round that edits a red
            test file and leaves it red owns it;
          * one new failure fails the rung, however many old ones sit beside
            it (the mixed case still records which ones are not the round's);
          * an INCONCLUSIVE base probe grants nothing — fail closed.
        """
        tail = "\n".join(text.strip().splitlines()[-15:])
        node_ids = _failed_node_ids(text)
        touched = self._touched_paths()
        own = [n for n in node_ids if n.split("::", 1)[0] in touched]
        probe_ids = [n for n in node_ids if n not in own]
        base_failed, probe_note, conclusive, cached = self._base_failures(probe_ids)
        pre_existing = [n for n in probe_ids if n in base_failed]
        _external, new = _classify_test_failure(probe_ids, base_failed)
        new = own + new
        # One run is one sample. The base probe above is a single invocation,
        # so a node that flickers clears it often enough to matter and lands
        # in `new`. Ask those nodes again before the wording commits to that.
        # Nodes the base probe DID reproduce never come here: their attribution
        # already exists, and re-running them spends minutes to repeat it.
        flaky: list[str] = []
        flaky_note = ""
        # Node ids only. An id with no `::` is a file pytest could not even
        # collect: repeating it re-runs a whole file to ask one question, and
        # an import or syntax error is deterministic far more often than an
        # assertion is. Such a failure stays the round's on the first sample.
        to_reconfirm = [n for n in new if "::" in n]
        if to_reconfirm:
            flaky, flaky_note = _reconfirm_candidate_failures(
                self.python, self.worktree, to_reconfirm,
                repeats=int(_gate_cfg("test_repeat_runs", REPEAT_RUNS) or 0),
                env=self._child_env(isolate_home=True))
        data = {**counts, "failed_node_ids": node_ids,
                "base_probe": probe_note, "home": self.home_isolation,
                "red_set_cached": cached}
        if own:
            data["touched_failures"] = own
        if pre_existing:
            data["pre_existing_failures"] = pre_existing
        if flaky:
            # What was actually asked again, not everything that failed: a
            # node the base probe reproduced was never in the repeat batch.
            data["retried_node_ids"] = list(to_reconfirm)
            data["flaky_node_ids"] = list(flaky)
            new = [n for n in new if n not in flaky]
        red_item = self._file_red_tree(pre_existing) if (pre_existing and conclusive) else None
        if red_item:
            data["red_tree_item"] = red_item
            if self.item_id and int(red_item) == int(self.item_id):
                # This round IS the red-tree item's round: the headline must not
                # tell it to leave its own contract alone.
                data["red_tree_item_is_own"] = True

        if node_ids and not new and conclusive:
            parts = []
            if pre_existing:
                parts.append(f"{len(pre_existing)} failure(s) PRE-EXISTING at base "
                             f"{self.base[:8]} (reproduced with this round's diff absent; "
                             f"not this change's"
                             + (f", tracked by item #{red_item}" if red_item else "")
                             + f"): {_name_ids(pre_existing)}")
            if flaky:
                parts.append(f"{len(flaky)} FLAKY (passed a repeat run): {_name_ids(flaky)}")
            lead = ("tests pass on this diff — " + "; ".join(parts) + f". {probe_note}"
                    + (f". {flaky_note}" if flaky_note else "") + ". ")
            return self._tests_pass(counts, only, {**{k: v for k, v in data.items()
                                                      if k not in counts},
                                                   "_lead": lead})
        if not conclusive and node_ids and not new:
            return False, (f"pytest failed ({counts}): every failure outside the round's own "
                           f"files was unexplained, and the base probe could not say whose "
                           f"they are — failing closed. {probe_note}. {flaky_note}\n"
                           f"{tail[-600:]}"), data
        if flaky:
            # Some failures flickered and some did not: the round still owns
            # the ones that failed every repeat. Only the flaky ones come out of
            # the count, so the number the round reads is the number it caused.
            data["new_failures"] = new
            return False, (f"pytest failed ({counts}): {len(new)} of "
                           f"{len(node_ids)} failures are new in this round "
                           f"({_name_ids(new)}); {len(flaky)} passed a repeat run "
                           f"and are FLAKY ({_name_ids(flaky)}). {probe_note}. "
                           f"{flaky_note}\n{tail[-600:]}"), data
        if node_ids and pre_existing:
            data["new_failures"] = new
            return False, (f"pytest failed ({counts}): {len(new)} of "
                           f"{len(node_ids)} failures are new in this round "
                           f"({_name_ids(new)}); the rest predate it "
                           f"({_name_ids(pre_existing)} — not yours"
                           + (f", tracked by item #{red_item}" if red_item else "")
                           + f"). {probe_note}\n{tail[-600:]}"), data
        # Nothing reproduces at base: every failure is this round's. Say so
        # in the same field the mixed case uses, so a reader of the report
        # never has to infer the delta from its absence — and name the
        # failures in the detail itself, ahead of the pytest tail. On
        # 2026-09-13 this branch returned only `tail[-900:]`, which began
        # mid-name; round SM_20260913_165927 (#472) never saw its twelve real
        # failures, invented a test file that exists nowhere, and filed
        # blocker #1093 against the gate for it.
        data["new_failures"] = new
        if node_ids:
            return False, (f"pytest failed ({counts}): all {len(node_ids)} failure(s) are "
                           f"new in this round ({_name_ids(node_ids)}). {probe_note}\n"
                           f"{tail[-600:]}"), data
        return False, f"pytest failed ({counts}): {tail[-900:]}\n{probe_note}", data

    def rung_review(self):
        """A second reader grades the diff against the item's clauses.

        Fails closed in every direction that matters: an unreachable grader is
        `external_blocker` (the engine, not the diff — the item keeps its
        attempt), never a pass; a third refusal in one round is refused
        without asking the model again; and a clause the grader marks `met`
        without evidence is downgraded in Python. Only a round with no item
        bound records a skip, because there is no contract to grade against.

        Three rules from 2026-09-11, when every review-gated round aborted
        and none of the refusals had reached the model (see `state.py`'s
        gate marker for how one gate became two):

        - **An attempt is a graded refusal of a distinct commit.** A grader
          timeout is the engine's problem and spends nothing; the same head
          refused twice is one refusal, delivered twice — the second gate of a
          duplicated pair, or a model that re-gated without committing. That
          case is answered from the ledger without a grading turn.
        - **The grader reads a detached checkout of the commit, never the
          working tree.** The second review of #578 caught a test file
          mid-write and graded a head the author had already moved past.
        - **The event carries `head`**, so `backlog.review_disagreement` can
          tell two refusals of one commit from two refusals of two.
        """
        ctx = self._review_prepare()
        if isinstance(ctx, tuple):
            # Answered without a grading turn (no item, reuse, same head,
            # exhausted). A prefetch cannot normally exist here — its own
            # prepare would have answered the same way — but if the ledger
            # moved under it, its grade answers a question nobody is asking.
            self._discard_review_prefetch("the rung answered without grading")
            return ctx
        from scripts.automod import backlog as _B
        from scripts.automod import review as RV
        contract, head, attempt = ctx["contract"], ctx["head"], ctx["attempt"]
        changed, changed_tests, pre = ctx["changed"], ctx["changed_tests"], ctx["pre"]
        test_counts = next((r.data for r in self.report.rungs if r.name == "tests"), {}) or {}
        # Failures the tests rung passed over because they predate the round:
        # the grader is told, and a `met` that leans on one does not stand.
        pre_existing_failures = [str(n) for n in (test_counts.get("pre_existing_failures") or [])]
        rung_started = time.time()
        bundle, concurrent, discarded = self._take_review_prefetch(ctx, pre_existing_failures)
        if bundle is None:
            started = time.time()
            snapshot, snap_note = self._review_snapshot(head)
            try:
                bundle = self._review_call(ctx, snapshot, snap_note, started,
                                           test_counts=test_counts,
                                           pre_existing_failures=pre_existing_failures)
            except BaseException:
                self._drop_snapshot(snapshot)
                raise
        res, snapshot, started = bundle["res"], bundle["snapshot"], bundle["started"]
        grade_root, base_event = bundle["grade_root"], bundle["base_event"]
        # Whether the grade ran beside the tests rung, on every review row and
        # in the rung data, so the scorecard can tell the two paths apart and
        # a discarded prefetch says why.
        base_event["review_concurrent"] = concurrent
        if concurrent:
            base_event["rung_wait_s"] = round(time.time() - rung_started, 1)
        if discarded:
            base_event["review_prefetch_discarded"] = discarded
        concurrency = {"review_concurrent": concurrent,
                       **({"review_prefetch_discarded": discarded} if discarded else {})}
        # The tree the grader's citations are validated against, recorded so a
        # refusal states what it graded: the round that was refused on
        # `agent_mcp/facts.py:520-540` (a write path, not a test) and a landing
        # at `08a4f4f0` (not an object) left no way for the next reader to
        # re-run either check.
        validated = {"review_validated_head": head, "review_validated_worktree": str(grade_root),
                     **concurrency}
        parsed = None
        try:
            if not res["ok"]:
                S.append_event({**base_event, "ok": False, "blocking": False,
                                "error": str(res.get("error") or "")[:400]})
                return False, (f"review could not run: {res.get('error')} — the grader, not the "
                               f"diff; neither the item's attempt nor a review attempt is spent"), {
                                   "external_blocker": True, "external_failures": [],
                                   "external_reason": "grader unreachable",
                                   "retry_after_s": 120,
                                   "review_session": res.get("session_id"), **concurrency}
            parsed = RV.parse_review(res["structured"], worktree=grade_root,
                                     changed_tests=changed_tests, n_clauses=len(contract["clauses"]),
                                     # A suite-level or unchanged-test `met`
                                     # stands only on a green tests rung. Read
                                     # off the REAL rung even when the grade
                                     # ran beside it: that is what makes the
                                     # prefetch safe to join.
                                     tests_passed=any(r.name == "tests" and r.ok
                                                      for r in self.report.rungs),
                                     changed_paths=changed,
                                     # Every commit the grader cites is asked of
                                     # the round's own repo, and the `def test_`
                                     # delta the round really added is the
                                     # deterministic answer to "this diff adds
                                     # no test".
                                     repo=self.live,
                                     added_tests=RV.def_test_delta(grade_root, self.base, changed),
                                     pre_existing_failures=set(pre_existing_failures))
        finally:
            self._drop_snapshot(snapshot)
        tree_note = (f" (citations validated against {head[:8] or 'the working tree'} "
                     f"in {grade_root})")
        if parsed and parsed.get("unreliable"):
            # The grader's own evidence is not in the tree it was handed, so
            # this text is not a judgment of the diff: it is the same shape as
            # an unreachable grader, and it spends no attempt (#1442).
            reasons = "; ".join(parsed["unreliable"])
            S.append_event({**base_event, "ok": False, "blocking": False,
                            "unreliable": parsed["unreliable"],
                            # The clause entries carry their own
                            # `citation_unresolved` markers, so the record shows
                            # WHICH citation failed, not just that one did.
                            "clauses": parsed["clauses"],
                            "error": f"review unreliable: {reasons}"[:400],
                            "session_id": res.get("session_id"),
                            "seconds": round(time.time() - started, 1)})
            return False, ("review is unreliable"
                           f"{tree_note}: the findings cite evidence that does not exist where "
                           f"the grader said it looked — {reasons}. The item keeps its attempt and "
                           f"no review attempt is spent; gate again (the grader may answer) and "
                           f"report the citation if it repeats"), {
                               "external_blocker": True, "external_failures": [],
                               "external_reason": "grader cited evidence not in the graded tree",
                               "retry_after_s": 120,
                               "review_unreliable": parsed["unreliable"],
                               "review_session": res.get("session_id"), **validated}
        if parsed is None:
            S.append_event({**base_event, "ok": False, "blocking": False,
                            "error": "structured review unusable"})
            return False, "review returned an unusable object; the item keeps its attempt", {
                "external_blocker": True, "external_failures": [],
                "external_reason": "grader returned an unusable object",
                "retry_after_s": 120,
                "review_session": res.get("session_id"), **concurrency}
        if parsed.get("clauses_unreadable"):
            unread = parsed["clauses_unreadable"]
            keys = ", ".join(unread.get("keys") or []) or "(no readable keys on any entry)"
            S.append_event({**base_event, "ok": False, "blocking": False,
                            "error": f"review clause entries unreadable: {unread['entries']} "
                                     f"entries, no usable 1-based `clause` index; "
                                     f"keys on them: {keys[:300]}"})
            # The synthesized "not addressed by the grader" partials are a
            # statement about the grader's key names, not about the diff. On
            # SM_20260916_032218, SM_20260922_100227 and SM_20260924_104224
            # refusing on them spent an attempt of two on rounds the second
            # reader had actually approved, and told the author to change code
            # that was already graded `met`. An unreadable verdict is the same
            # kind of event as the unusable object above: the rail failed, so
            # nothing is charged and nothing is named for the author to change.
            why = (f"review verdict could not be read: {unread['entries']} clause entries "
                   f"came back and none carried a usable 1-based `clause` index (keys found "
                   f"on them: {keys[:300]}); no clause was graded, so this is the grading "
                   f"rail and not a judgment of the diff — gate again; the item keeps its "
                   f"attempt")
            return False, why, {
                "external_blocker": True, "external_failures": [],
                "external_reason": "grader's clause entries carried no usable clause index",
                "retry_after_s": 120,
                "review_session": res.get("session_id"), **concurrency}
        amendments = contract.get("amendments") or []
        kind, findings = RV.decide(parsed, pre, amendments=amendments,
                                   attempt=attempt, policy=RV.seams_policy())
        S.append_event({**base_event, "ok": True, "premise": parsed["premise"],
                        "clauses": parsed["clauses"], "test_honesty": parsed["test_honesty"],
                        "seams_unverified": [s["seam"] if isinstance(s, dict) else s
                                             for s in parsed["seams_unverified"]],
                        "seams_untestable": [s["seam"] for s in parsed["seams_unverified"]
                                             if isinstance(s, dict)
                                             and not s.get("testable_before_landing", True)],
                        # The whole judgment per seam. The two lists above
                        # lose `actionable_in_round` and `same_as_prior`, so
                        # re-deciding a recorded review had to guess them.
                        "seams": [s for s in parsed["seams_unverified"] if isinstance(s, dict)],
                        "downgraded": parsed["downgraded"], "summary": parsed["summary"],
                        "amendments_ok": parsed.get("amendments_ok", True),
                        "amendments_note": parsed.get("amendments_note", ""),
                        "blocking": kind != "pass", "kind": kind,
                        "findings": findings[:2000]})
        self._settle_amendments(amendments, parsed, kind)
        if kind == "unsound":
            return False, f"review: premise unsound{tree_note} — {findings}", {
                "review_premise_unsound": True, "review_summary": findings[:800],
                "review_session": res.get("session_id"), **validated}
        if kind == "retry":
            contract_refusal = any(c.get("verdict") == "unsatisfiable" for c in parsed["clauses"])
            if contract_refusal:
                nxt = ("this refusal spends no attempt — amend the unsatisfiable clause(s) "
                       "with automod_amend_clause, fix anything else it names, commit if "
                       "needed, and gate again")
                shown = f"{attempt - 1}/{RV.REVIEW_MAX_PER_ROUND} spent"
            else:
                nxt = ("fix what it names, commit, and gate again"
                       if attempt < RV.REVIEW_MAX_PER_ROUND else
                       "abort and report — the item comes back with these findings and your branch")
                shown = f"{attempt}/{RV.REVIEW_MAX_PER_ROUND}"
            return False, (f"review sent it back ({shown}; {nxt}){tree_note}: {findings}"), {
                "review_retry": True, "review_findings": findings[:1500],
                "review_attempt": attempt, "review_session": res.get("session_id"), **validated}
        # On a PASS, record the grader's `post_landing` clauses onto the item.
        # Written here rather than by the implementer because it is a fact the
        # grader established about a change that is about to land, not a claim
        # the author made about its own work.
        marked: list[int] = []
        for c in parsed["clauses"]:
            if c["verdict"] != "post_landing":
                continue
            try:
                if _B.mark_clause_post_landing(self.item_id, int(c["clause"]),
                                               note=c.get("note") or "",
                                               round_id=self.round_id):
                    marked.append(int(c["clause"]))
            except Exception as exc:  # noqa: BLE001 — a mark is not the gate
                print(f"[warn] could not mark clause {c['clause']} post_landing: {exc}")
        if marked:
            S.append_event({"event": "gate", "round_id": self.round_id,
                            "rung": "review", "ok": True, "skipped": False,
                            "post_landing_clauses": marked, "item_id": self.item_id,
                            "detail": "clauses marked observable only after landing"})
        # Everything a pass did not refuse on is still something the grader
        # said. It goes onto the item — the prompt promises the grader an
        # untestable seam is recorded there — and into the rung data, so
        # `gate.json` and the landing report carry it too.
        advisory_seams = [s["seam"] if isinstance(s, dict) else str(s)
                          for s in parsed["seams_unverified"]]
        advisory_findings = [f"{h['file']}:{h['line']}: {h['problem']}"
                             for h in list(pre) + list(parsed["test_honesty"])]
        try:
            _B.note_review_advisories(self.item_id, self.round_id,
                                      advisory_seams, advisory_findings)
        except Exception as exc:  # noqa: BLE001 — a note is not the gate
            print(f"[warn] could not record review advisories: {exc}")
        return True, (f"review: {RV.summarize_clauses(parsed)} of {len(contract['clauses'])} "
                      f"clause(s); {parsed['summary'][:160]}"), {
                          "review_session": res.get("session_id"),
                          "clauses": parsed["clauses"], "review_attempt": attempt,
                          "post_landing_clauses": marked,
                          "advisory_seams": advisory_seams,
                          "advisory_findings": advisory_findings,
                          "amendments_ratified": [a.get("clause") for a in amendments],
                          **validated}

    def _review_prepare(self):
        """Everything the review rung decides before it asks a grader.

        Returns the rung's finished answer as a `(ok, detail, data)` tuple
        when no grading turn is owed — no item, no clauses, the hard cap, a
        reused patch-id, the same head refused, the attempt cap — or else a
        dict of what a grade and its judgment need. Appends no ledger event
        and spends nothing, so the prefetch beside the tests rung can call it
        and the rung can call it again: the one write, orphaning another
        round's stale amendments, is idempotent.
        """
        if not self.item_id:
            return True, "SKIPPED (no backlog item bound to this round — no contract to grade)", {
                "skipped": True, "reason": "no item"}
        from scripts.automod import backlog as _B
        from scripts.automod import review as RV
        # Before the contract is read: an amendment another round left
        # pending was never ratified, so it is not part of the contract this
        # round is graded on. Best effort — the filter below is what keeps a
        # stale one from reopening the cap even if this write fails.
        try:
            _B.orphan_stale_amendments(self.item_id, self.round_id)
        except Exception as exc:  # noqa: BLE001 — bookkeeping, not the verdict
            print(f"[warn] could not orphan stale amendments: {exc}")
        contract = RV.item_contract(self.item_id)
        if not contract["clauses"]:
            return False, f"item #{self.item_id} has no acceptance clauses", {}
        contract = {**contract, "amendments": [
            a for a in (contract.get("amendments") or [])
            if str(a.get("round_id") or "") == str(self.round_id)]}
        head = W.head(self.worktree) or self.report.head or ""
        reviews = [e for e in S.read_events(limit=1000) if e.get("event") == "review"]
        prior = [e for e in reviews if e.get("round_id") == self.round_id]
        # What the GRADER is shown is the item's history, not the round's: a
        # re-offered round's first review otherwise judges `same_as_prior`
        # blind to the refusal that sent the item back. Attempts are still
        # counted on `prior` alone.
        # Since the item's last re-triage, though: that wrote a new contract,
        # and its clause 2 is not the clause 2 an earlier review refused.
        try:
            since = _B.retriage_marks(S.LEDGER_PATH).get(int(self.item_id), float("-inf"))
        except Exception:  # noqa: BLE001 — history is context, not the verdict
            since = float("-inf")
        item_history = [e for e in reviews if e.get("ok")
                        and str(e.get("item_id") or "") == str(self.item_id)
                        and float(e.get("ts") or 0) > since][-3:]
        refused_by_head: dict[str, dict] = {}
        graded_refusals: list[dict] = []
        graded_total = 0
        spent = 0
        for e in prior:
            if not e.get("ok"):
                continue
            graded_total += 1
            if not e.get("blocking"):
                continue
            graded_refusals.append(e)
            h = str(e.get("head") or "")
            if h and h in refused_by_head:
                continue
            if h:
                refused_by_head[h] = e
            # A refusal of the CONTRACT spends nothing: the diff could not
            # have passed a clause no diff can satisfy, and the author's next
            # move — amend, gate again — must not be the one the cap refuses.
            if any(c.get("verdict") == "unsatisfiable" for c in (e.get("clauses") or [])):
                continue
            spent += 1
        attempt = spent + 1
        if graded_total >= RV.REVIEW_HARD_CAP:
            last = graded_refusals[-1] if graded_refusals else {}
            return False, (f"review has graded this round {graded_total} times (ceiling "
                           f"{RV.REVIEW_HARD_CAP}); abort and report — the item comes back "
                           f"to the next round with the findings"), {
                               "review_retry": True, "review_exhausted": True,
                               "review_attempt": spent + 1,
                               "review_findings": str(last.get("findings") or "")[:1500]}
        # A pending amendment changes the contract, so the SAME head is a
        # different question and has to be graded again. 866-c amended a
        # clause, re-gated the same commit, and was answered from the ledger
        # — the amendment was never looked at, and the round aborted on a
        # refusal of the text it had just replaced. THIS round's amendments
        # only (filtered above): #860's third attempt was bought by one from
        # a round two days gone.
        pending_amendments = list(contract.get("amendments") or [])
        # A clean rebase produces a new commit with an identical diff. The
        # promoter chases a moved base by re-gating, and `_regate_after_move`
        # used to record `review: skipped` for exactly this case — so a
        # landing could be judged on a review of a commit that no longer
        # existed. `git patch-id --stable` is the identity of the CHANGE
        # rather than of the commit, so a prior PASS of the same patch is the
        # same answer to the same question.
        patch_id = self._patch_id()
        if patch_id and not pending_amendments:
            for e in reversed(prior):
                if (e.get("ok") and not e.get("blocking")
                        and str(e.get("patch_id") or "") == patch_id):
                    return True, (f"review: identical diff already passed at "
                                  f"{str(e.get('head') or '')[:8]} (patch-id "
                                  f"{patch_id[:8]}); rebased, not re-graded"), {
                                      "review_reused": True, "patch_id": patch_id,
                                      "review_attempt": attempt,
                                      "clauses": e.get("clauses") or []}

        same = refused_by_head.get(head) if (head and not pending_amendments) else None
        if same is not None:
            findings = str(same.get("findings") or "")[:1500]
            return False, (f"review already refused this exact commit {head[:8]} (attempt "
                           f"{same.get('attempt')}) and nothing has been committed since — "
                           f"no grading turn spent; fix what it names, commit, and gate "
                           f"again: {findings}"), {
                               "review_retry": True, "review_findings": findings,
                               "review_attempt": int(same.get("attempt") or spent),
                               "review_same_head": True}
        # Same reason: a round holding an unratified amendment has not had
        # its new contract judged even once. The HARD_CAP above still bounds
        # it — that one counts grading turns, not refusals, so amendments
        # cannot buy unlimited passes.
        if attempt > RV.REVIEW_MAX_PER_ROUND and not pending_amendments:
            last = graded_refusals[-1] if graded_refusals else {}
            return False, (f"review already sent this round back {RV.REVIEW_MAX_PER_ROUND} "
                           f"times; abort and report (automod_abort with the findings as "
                           f"the reason) — the item comes back to the next round with them"), {
                               "review_retry": True, "review_exhausted": True,
                               "review_attempt": attempt,
                               "review_findings": str(last.get("findings") or "")[:1500]}
        changed = list(self.report.changed_paths)
        changed_tests = TP.pick_test_files(changed, self.worktree)
        pre = RV.honesty_prechecks(self.worktree, self.base, changed,
                                   n_clauses=len(contract["clauses"]))
        return {"contract": contract, "head": head, "attempt": attempt,
                "patch_id": patch_id, "pending_amendments": pending_amendments,
                "item_history": item_history, "changed": changed,
                "changed_tests": changed_tests, "pre": pre}

    @staticmethod
    def _review_key(ctx: dict) -> dict:
        """What a prefetched grade must still agree with to be joined."""
        return {"head": ctx["head"], "patch_id": ctx["patch_id"], "attempt": ctx["attempt"],
                "clauses": list(ctx["contract"]["clauses"]),
                "amendments": [(a.get("clause"), a.get("now"))
                               for a in ctx["pending_amendments"]]}

    def _review_call(self, ctx: dict, snapshot: Path | None, snap_note: str, started: float, *,
                     test_counts: dict, pre_existing_failures: list[str],
                     scratch_dir: Path | None = None, on_session=None) -> dict:
        """The grading turn itself, and the `review` event it will be recorded
        under. Leaves `snapshot` for the caller to drop."""
        from scripts.automod import review as RV
        head, pending_amendments = ctx["head"], ctx["pending_amendments"]
        grade_root = snapshot or self.worktree
        base_event = {"event": "review", "round_id": self.round_id, "item_id": self.item_id,
                      "attempt": ctx["attempt"], "head": head, "grader_model": "primary",
                      "snapshot": bool(snapshot), "snapshot_note": snap_note,
                      "prechecks": ctx["pre"],
                      "validated_head": head, "validated_worktree": str(grade_root),
                      # A refusal shown an amendment is a judgment of a new
                      # contract; `backlog.review_disagreement` reads this so
                      # it does not count the amended pass as a repeat.
                      "patch_id": ctx["patch_id"],
                      "amendments_shown": [a.get("clause") for a in pending_amendments]}
        kwargs = dict(round_id=self.round_id, worktree=grade_root, base=self.base,
                      contract=ctx["contract"], changed_paths=ctx["changed"],
                      test_counts=test_counts,
                      python=self.python, child_env=self._child_env(grade_root),
                      scratch_dir=scratch_dir or (W.round_dir(self.round_id) / "gate-state"),
                      # The item's recent reviews across rounds, so the
                      # grader judges repeats itself (`same_as_prior`)
                      # rather than the ledger inferring them by head.
                      prior_reviews=ctx["item_history"],
                      pre_existing_failures=pre_existing_failures)
        if on_session is not None:
            kwargs["on_session"] = on_session
        res = RV.grade(**kwargs)
        base_event.update({"session_id": res.get("session_id"),
                           "seconds": round(time.time() - started, 1),
                           # Waits the grader sat out because another
                           # round was landing. On the scorecard, so a
                           # grader that is chronically unavailable is a
                           # number rather than a story.
                           "retries": int(res.get("retries") or 0),
                           "waited_s": float(res.get("waited_s") or 0.0)})
        return {"res": res, "snapshot": snapshot, "started": started,
                "grade_root": grade_root, "base_event": base_event}

    # ── the review graded beside the tests rung ───────────────────────
    # The grader needs nothing the tests rung computes before it can start:
    # the suite's counts only colour one prompt line, and the two things that
    # decide a verdict — a green rung, and which failures predate the round —
    # are applied in Python by `parse_review` AFTER the grader answers. So
    # the grading turn (p50 ~280 s) starts when the suite (p50 ~150 s) does,
    # and is joined to the real tests result when the review rung is reached.
    # `automod.gate.concurrent_review: false` is the serial ladder exactly.

    def _start_review_prefetch(self) -> None:
        """Start the review's grading turn in a daemon thread, if owed."""
        self._review_prefetch = None
        if not self.item_id or not _gate_cfg("concurrent_review", True):
            return
        try:
            ctx = self._review_prepare()
        except Exception as exc:  # noqa: BLE001 — the rung will ask again, serially
            print(f"[warn] review prefetch not started: {type(exc).__name__}: {exc}")
            return
        if isinstance(ctx, tuple):
            return  # the rung answers without a grade; nothing to start
        pf = _ReviewPrefetch(self._review_key(ctx))
        head = ctx["head"]

        def _body():
            try:
                started = time.time()
                # Its own checkout and its own scratch dir: a discarded
                # prefetch whose grader is still running must not share a
                # tree or a `run_tests.sh` with the serial grade that
                # replaces it.
                snap, note = self._review_snapshot(head, suffix="-prefetch")
                if not pf.adopt_snapshot(snap):
                    self._drop_snapshot(snap)
                    return
                pf.bundle = self._review_call(
                    ctx, snap, note, started, test_counts={}, pre_existing_failures=[],
                    scratch_dir=W.round_dir(self.round_id) / "gate-state" / "review-prefetch",
                    on_session=pf.set_session)
            except BaseException as exc:  # noqa: BLE001 — reported at the join
                pf.error = exc
            finally:
                pf.done.set()

        # Daemon, never an executor: `concurrent.futures` workers are joined
        # at interpreter exit, and a gate that stopped at `tests` must not sit
        # out a five-minute grading turn before its process can end.
        t = threading.Thread(target=_body, name=f"review-prefetch-{self.round_id}", daemon=True)
        pf.thread = t
        self._review_prefetch = pf
        t.start()
        print(f"[info] review grading started beside the tests rung ({head[:8]})")

    def _take_review_prefetch(self, ctx: dict, pre_existing_failures: list[str]):
        """`(bundle, concurrent, discard_reason)` for the rung to judge.

        `bundle` is None when there is no usable prefetch and the rung must
        grade serially; `discard_reason` then says why one was thrown away.
        """
        pf = getattr(self, "_review_prefetch", None)
        self._review_prefetch = None
        if pf is None:
            return None, False, ""
        if pre_existing_failures:
            # The grader was not told which failures predate the round, and
            # the prompt is where it is told — so grade again, as today.
            why = (f"the tests rung passed over {len(pre_existing_failures)} pre-existing "
                   f"failure(s) the prefetched grader was not told about")
        else:
            now = self._review_key(ctx)
            moved = [k for k in now if now[k] != pf.key.get(k)]
            why = f"{', '.join(moved)} changed since it started" if moved else ""
        if not why:
            # No timeout beyond the grade's own: `RV.grade` is bounded, and a
            # serial grade would wait exactly as long.
            pf.done.wait()
            if pf.error is not None:
                why = f"it raised {type(pf.error).__name__}: {str(pf.error)[:200]}"
            elif pf.bundle is None:
                why = "it produced nothing"
            else:
                return pf.bundle, True, ""
        self._abandon_prefetch(pf)
        print(f"[info] review prefetch discarded: {why}")
        return None, False, why

    def _discard_review_prefetch(self, why: str) -> None:
        """Throw a prefetch away without recording anything.

        The gate stopped before the review rung (a red `tests` or
        `prompt_surface`, or an exception), or the rung answered without a
        grade: no `review` event is written and no attempt is spent, because
        nothing was judged. The grader's backend turn is cancelled best
        effort; one that was not yet streaming may run to its own end on the
        live backend and its answer goes nowhere. That is accepted — it costs
        the primary one grading turn, never a verdict.
        """
        pf = getattr(self, "_review_prefetch", None)
        self._review_prefetch = None
        if pf is None:
            return
        self._abandon_prefetch(pf)
        print(f"[info] review prefetch discarded: {why}")

    def _abandon_prefetch(self, pf: "_ReviewPrefetch") -> None:
        snap, session_id = pf.abandon()
        self._drop_snapshot(snap)
        if session_id and not pf.done.is_set():
            self._cancel_grader(session_id)

    @staticmethod
    def _cancel_grader(session_id: str) -> None:
        try:
            from scripts.automod import review as RV
            RV.cancel_grader(session_id)
        except Exception as exc:  # noqa: BLE001 — best effort
            print(f"[warn] could not cancel the prefetched grader {session_id}: {exc}")

    def _patch_id(self) -> str:
        """`git patch-id --stable` of this round's whole diff, or "".

        The identity of the CHANGE rather than of the commit: a clean rebase
        moves the head and leaves this alone. Never raises — an unavailable
        patch-id costs the reuse, not the rung.
        """
        try:
            diff = _run(["git", "diff", f"{self.base}...HEAD"],
                        cwd=self.worktree, timeout=120).stdout
            if not diff.strip():
                return ""
            proc = subprocess.run(["git", "patch-id", "--stable"],
                                  input=diff, capture_output=True, text=True,
                                  timeout=60, check=False)
            return (proc.stdout.split() or [""])[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not compute a patch-id: {exc}")
            return ""

    def _review_snapshot(self, head: str, suffix: str = "") -> tuple[Path | None, str]:
        """A detached checkout of `head` for the grader to read and test.

        The worktree is the author's, and on 2026-09-11 the author kept
        editing while a review ran — the grader caught a test file mid-write.
        A checkout of the commit cannot move. Lives under the round's scratch
        dir, registered against the live repo like the tests rung's baseline
        probe, and removed in `_drop_snapshot`. Falls back to the working
        tree with a note when git will not cooperate; the note rides the
        event so a graded working tree is visible, not silent.
        """
        if not head:
            return None, "no head to snapshot; graded the working tree"
        wt = W.round_dir(self.round_id) / "gate-state" / f"review-{head[:12]}{suffix}"
        shutil.rmtree(wt, ignore_errors=True)
        r = W.git(self.live, "worktree", "add", "--detach", "-q", str(wt), head)
        if r.returncode != 0 or not wt.exists():
            return None, (f"snapshot failed ({(r.stderr or '').strip()[:160]}); "
                          f"graded the working tree")
        return wt, f"graded a detached checkout of {head[:8]}"

    def _drop_snapshot(self, wt: Path | None) -> None:
        if not wt:
            return
        try:
            W.git(self.live, "worktree", "remove", "--force", str(wt))
        except Exception:
            pass
        shutil.rmtree(wt, ignore_errors=True)
        try:
            W.git(self.live, "worktree", "prune")
        except Exception:
            pass

    def _settle_amendments(self, amendments: list[dict], parsed: dict, kind: str) -> None:
        """Ratify or refuse the clause amendments this review was shown.

        The author may amend only a clause the previous review called
        unsatisfiable, and the amendment holds only if the next review
        accepts it — the second reader's judgment, not the author's. A
        refusal restores the clause. Best effort: a backlog write that fails
        costs the bookkeeping, never the verdict.
        """
        if not amendments or not self.item_id:
            return
        from scripts.automod import backlog as B
        ok = bool(parsed.get("amendments_ok", True))
        try:
            B.settle_amendments(self.item_id, self.round_id, ratified=ok,
                                note=str(parsed.get("amendments_note") or "")[:600])
        except Exception as exc:  # pragma: no cover - bookkeeping only
            print(f"[warn] settle_amendments failed: {exc}")

    def rung_venv(self):
        if not spec.touches_requirements(self.report.changed_paths):
            return True, "requirements unchanged — using the live venv", {
                "skipped": True, "reason": "requirements unchanged"}
        if not UV_BIN.exists():
            return False, f"uv not found at {UV_BIN}", {}

        live_venv = self.live / ".venvs" / "lloyd"
        clone = self.worktree / ".venvs" / "lloyd"
        clone.parent.mkdir(parents=True, exist_ok=True)
        # /home is btrfs, so this is a copy-on-write clone: measured at 3.2s
        # and effectively zero allocation for the 6.2GB / 48k-file venv, and
        # writes to the clone do not touch the live one.
        #
        # `--reflink=always` deliberately, not `auto`: auto falls back to a
        # real 6GB copy *silently*, so a filesystem change would turn this rung
        # into a multi-minute mystery. Fail loudly, then retry as a real copy
        # so the gate still works — but say which happened.
        cp = _run(["cp", "--reflink=always", "-a", str(live_venv), str(clone)], timeout=900)
        reflinked = cp.returncode == 0
        if not reflinked:
            shutil.rmtree(clone, ignore_errors=True)
            cp = _run(["cp", "-a", str(live_venv), str(clone)], timeout=1800)
            if cp.returncode != 0:
                return False, f"venv clone failed: {cp.stderr[-400:]}", {}

        lock = self.worktree / "requirements.lock"
        req = lock if lock.exists() else self.worktree / "requirements.txt"
        inst = _run([str(UV_BIN), "pip", "install", "-r", str(req),
                     "--python", str(clone / "bin" / "python")], timeout=1800)
        if inst.returncode != 0:
            return False, f"uv pip install failed: {(inst.stdout + inst.stderr)[-800:]}", {}

        imp = _run([str(clone / "bin" / "python"), "-c", "import server, agent_mcp.main"],
                   cwd=self.worktree, env=self._child_env(), timeout=300)
        if imp.returncode != 0:
            return False, f"candidate venv cannot import the app: {(imp.stdout + imp.stderr)[-600:]}", {}

        self.python = clone / "bin" / "python"
        self.report.venv = str(clone)
        how = "reflink clone" if reflinked else "FULL COPY (reflink unavailable)"
        return True, f"candidate venv built ({how} + delta) and imports cleanly", {
            "reflinked": reflinked}

    def rung_canary_boot(self):
        self._canary = C.Canary(W.round_dir(self.round_id), self.worktree,
                                python=self.python)
        self._canary.start()
        rep = self._canary.probe(timeout=180)
        if not rep["ok"]:
            return False, "; ".join(rep["errors"])[:900], {"probe": rep.get("internal_tools")}
        ok, why = self._canary.assert_commit(self.report.head)
        if not ok:
            return False, why, {}
        return True, f"booted; {rep.get('internal_tools')} internal tools; commit verified", {
            "internal_tools": rep.get("internal_tools")}

    def rung_canary_smoke(self):
        if self.skip_smoke:
            # Passes, because "no engine here" is not a defect in the candidate
            # — but it is recorded as a skip so the promotion record shows the
            # check did not happen, and `promoted` events can be filtered on it.
            return True, f"SKIPPED (engine unreachable: {self._smoke_skip_reason})", {
                "skipped": True, "reason": self._smoke_skip_reason,
                "has_trace": False, "no_trace_reason": "skipped"}
        rep = self._canary.smoke(timeout=240)
        # The turn's structural trace rides in the rung data whether or not
        # the four assertions held (#828). It is observation only: nothing
        # below may turn a trace difference into a failure.
        trace = CS.trace_of(rep)
        if not rep["ok"]:
            return False, "; ".join(rep["errors"])[:900], {
                **{k: rep.get(k) for k in ("tool_called", "tool_result_ok", "done")},
                **trace, "has_trace": True}
        diff, baseline = self._canary_trace_diff(trace, rep)
        against = (f"vs {baseline.get('round_id') or '?'}" if diff["has_baseline"]
                   else "no previous trace")
        return True, (f"real turn in {rep['duration_s']}s; Bash dispatched, "
                      f"sentinel round-tripped; trace {against}: "
                      f"{CS.format_comparison(diff)}"), {
            "duration_s": rep["duration_s"],
            "sentinel_in_response": rep.get("sentinel_in_response"),
            **trace, "has_trace": True, "trace_diff": diff,
            "trace_baseline": {k: baseline.get(k) for k in
                               ("round_id", "head", "recorded_at")} if baseline else None}

    def _canary_trace_diff(self, trace: dict, rep: dict) -> tuple[dict, dict | None]:
        """Compare against the previous passing smoke's trace, then store this one.

        The baseline lives in its own state file (`S.canary_trace_path()`),
        never in `last_known_good.json` — the guardian is that file's only
        writer. Wholly guarded: a trace that cannot be read or written costs
        the comparison, never the rung."""
        previous = None
        try:
            previous = S.read_canary_trace()
        except Exception as exc:  # noqa: BLE001 — observation is not the gate
            print(f"[warn] could not read the canary trace baseline: {exc}")
        diff = CS.compare_traces(trace, (previous or {}).get("trace"))
        try:
            S.write_canary_trace({
                "round_id": self.round_id, "head": self.report.head,
                "recorded_at": S.now_iso(),
                "trace": {**trace, "duration_s": rep.get("duration_s")}})
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not write the canary trace baseline: {exc}")
        return diff, previous

    def rung_drill(self):
        if not spec.requires_drill(self.report.changed_paths):
            return True, "no protected paths touched — drill not required", {
                "skipped": True, "reason": "no protected paths"}
        from scripts.automod import rehearse
        # Stop the gate's canary first: the drill needs the ports.
        if self._canary:
            self._canary.stop()
            self._canary = None
        ok, detail = rehearse.run_drill(self.round_id, self.worktree, self.base,
                                        python=self.python)
        return ok, detail, {}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run the self-modification gate")
    ap.add_argument("--round-id", required=True)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--skip-smoke", action="store_true",
                    help="skip the live-LLM turn (CI without vLLM)")
    args = ap.parse_args(argv)

    g = Gate(args.round_id, Path(args.worktree), args.base, skip_smoke=args.skip_smoke)
    report = g.run()
    out = S.ROUNDS_DIR / args.round_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
