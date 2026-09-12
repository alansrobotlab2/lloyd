"""The promotion gate: nine rungs, cheapest first, every one fails closed.

"Fails closed" is not a slogan here — it is the reason `_rung` catches every
exception and records it as a FAILED rung. With no human review tier, a rung
that errors and is read as "didn't fail" would silently remove a check.

The gate always runs **from the live tree** against the worktree, so a
candidate that weakens the gate is judged by the old gate, not by itself.

Rung order is a cost ladder, and it short-circuits: an import error should
cost 3 seconds, not a full canary boot.

  0 preflight      ~1s    lock, clean tree, ancestry, diff scope; with an item
                          bound, that it has clauses and the diff has a test
  1 static         ~8s    compileall, import smoke, pyflakes delta
  2 tests         ~35s    full pytest + a collected-count floor; on
                          failure, re-probes the failing files at the
                          round's base to say whose breakage it is
  3 review      60-180s   a second reader: a fresh session on the LIVE
                          backend grades the diff against the item's
                          acceptance clauses (scripts/automod/review.py).
                          Refuses with findings; the premise verdict decides
                          whether the item is retried or handed to a human.
                          Skipped (recorded) only when no item is bound.
  4 venv        0-300s    only when requirements changed (reflink + uv)
  5 canary boot   ~30s    both /health green, tool floor, config-follows-code
  6 canary smoke  ~30s    one real turn, sentinel through a real Bash call
                          (recorded as SKIPPED when the engine is unreachable;
                           `skip_smoke` is refused while it answers)
  7 drill         ~90s    only when the diff touches the rollback path

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
import time
from dataclasses import dataclass, field
from pathlib import Path

from app import lint_findings
from scripts.automod import canary as C
from scripts.automod import spec, state as S, worktree as W

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


def _vite_build(web: Path, out_dir: Path, timeout: float = 600) -> tuple[bool, str]:
    web = web.resolve()
    vite = web / "node_modules" / ".bin" / "vite"
    r = _run([str(vite), "build", "--outDir", str(out_dir), "--emptyOutDir",
              "--logLevel", "error"], cwd=web, env=_node_env(), timeout=timeout)
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
    """Pull counts out of pytest's summary line."""
    out: dict = {"passed": 0, "failed": 0, "errors": 0, "xfailed": 0,
                 "skipped": 0, "collected": 0}
    m = re.search(r"collected (\d+) item", text)
    if m:
        out["collected"] = int(m.group(1))
    for key, pattern in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"),
                         ("errors", r"(\d+) error"), ("xfailed", r"(\d+) xfailed"),
                         ("skipped", r"(\d+) skipped")):
        m = re.search(pattern, text)
        if m:
            out[key] = int(m.group(1))
    if not out["collected"]:
        out["collected"] = (out["passed"] + out["failed"] + out["xfailed"]
                            + out["errors"] + out["skipped"])
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
# is absent? Blocking the promotion is still right — landing onto a red tree
# would give the guardian's observation window a broken baseline. Consuming the
# backlog item is what was wrong.
EXTERNAL_PROBE_TIMEOUT = 600.0


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
    """
    if not node_ids:
        return set(), "no node ids to probe"
    wt = scratch / "baseline"
    shutil.rmtree(wt, ignore_errors=True)
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
        # `--no-header -p no:cacheprovider`: the probe must not write a
        # .pytest_cache into a tree it is about to delete, and must not read
        # one written by the candidate run.
        r = _run([str(python), "-m", "pytest", "-q", "--no-header",
                  "-p", "no:cacheprovider", "--continue-on-collection-errors",
                  "-m", "not live_vault", *files],
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
        return failed, (f"probed {len(files)} file(s) at base {base[:8]}: "
                        f"{len(failed)} already failing")
    except subprocess.TimeoutExpired:
        return set(), f"baseline probe timed out after {EXTERNAL_PROBE_TIMEOUT:.0f}s"
    except Exception as exc:
        return set(), f"baseline probe failed: {type(exc).__name__}: {exc}"
    finally:
        W.git(live_root, "worktree", "remove", "--force", str(wt))
        W.git(live_root, "worktree", "prune")
        shutil.rmtree(wt, ignore_errors=True)


def _review_policy(key: str, default: str = "first") -> str:
    """`automod.review.<key>` from config, or the default. Never raises."""
    try:
        from app.config import CONFIG
        return str(((CONFIG.get("automod") or {}).get("review") or {})
                   .get(key, default))
    except Exception:
        return default


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
        self.report = GateReport(round_id=round_id, base=base,
                                 head=W.head(self.worktree) or "")

    def _child_env(self, root: Path | None = None) -> dict:
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
        """
        scratch = W.round_dir(self.round_id) / "gate-state"
        (scratch / "automod").mkdir(parents=True, exist_ok=True)
        (scratch / "guardian").mkdir(parents=True, exist_ok=True)
        return {
            "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
            "PYTHONPATH": str(root or self.worktree),
            "LLOYD_AUTOMOD_STATE": str(scratch / "automod"),
            "LLOYD_GUARDIAN_STATE": str(scratch / "guardian"),
            "LLOYD_VOICE_ALERTS": "0",
        }

    # ── driver ─────────────────────────────────────────────────────────
    def _rung(self, name: str, fn) -> bool:
        started = time.time()
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
                 "skipped": bool((res.data or {}).get("skipped")),
                 "seconds": round(res.seconds, 2)}
        if (res.data or {}).get("external_blocker"):
            event["external_blocker"] = True
            event["external_failures"] = (res.data or {}).get("external_failures", [])
        # The review rung's two verdicts ride the event for the same reason:
        # `backlog.implement_outcomes` reads them long after the round dir is
        # gone, and the findings are what the next round is told.
        for key in ("review_retry", "review_premise_unsound"):
            if (res.data or {}).get(key):
                event[key] = True
        for key in ("review_findings", "review_summary", "review_attempt"):
            if (res.data or {}).get(key):
                event[key] = (res.data or {})[key]
        S.append_event(event)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({res.seconds:.1f}s) {res.detail[:160]}")
        return ok

    def run(self) -> GateReport:
        # `canary_smoke` is ALWAYS on the ladder. It used to be omitted when
        # `skip_smoke` was passed, so a skipped rung left no trace at all: the
        # report listed seven rungs and a reader had to know the eighth existed
        # to notice it was gone. A skip is now a recorded rung that says so.
        ladder = [("preflight", self.rung_preflight), ("static", self.rung_static),
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

        self._canary: C.Canary | None = None
        try:
            for name, fn in ladder:
                if not self._rung(name, fn):
                    self.report.ok = False
                    return self.report
            self.report.ok = True
        finally:
            if getattr(self, "_canary", None):
                try:
                    self._canary.stop()
                except Exception:
                    pass
        return self.report

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
            ok, why, conflicts = W.rebase_onto(self.worktree, live_head)
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
            return False, (f"live tree has uncommitted edits in paths this round also "
                           f"changes: {overlap} — two writers on one file; commit or "
                           f"stash the live edit, then gate again"), {
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
            py_changed = any(p.endswith(".py") and not p.startswith("tests/") for p in changed)
            if py_changed and not any(p.startswith("tests/") for p in changed):
                return False, (f"item #{self.item_id} has {len(contract['clauses'])} acceptance "
                               f"clause(s) and this diff changes code with no test under "
                               f"tests/ — nothing pins a clause"), data
            data["item_id"] = self.item_id
            data["clauses"] = len(contract["clauses"])

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
            return True, "no frontend changed", {}
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
        return True, (f"tsc: no new errors ({sum(head.values())} pre-existing); "
                      f"vite build ok; {len(changed_web)} frontend file(s)"), {
                          "changed_web": changed_web}

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
                   cwd=self.live, env=self._child_env(), timeout=1800)
        if run.returncode != 0:
            tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-15:])
            return False, f"tool-choice eval failed to run: {tail}", {"label": label}
        cmp_ = _run([str(self.python), "eval/compare_tool_choice.py",
                     "--label", label],
                    cwd=self.live, env=self._child_env(), timeout=600)
        text = (cmp_.stdout + cmp_.stderr).strip()
        tail = "\n".join(text.splitlines()[-15:])
        data = {"label": label, "compare_exit": cmp_.returncode}
        if cmp_.returncode != 0:
            return False, (f"tool-choice comparison exit {cmp_.returncode} "
                           f"(0=pass, 1=regression, 2=nothing to compare "
                           f"against, which is not a pass): {tail}"), data
        return True, f"tool-choice eval: no regression. {tail[-300:]}", data

    def rung_tests(self):
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
        r = _run([str(self.python), "-m", "pytest", "-q", "-m", "not live_vault"],
                 cwd=self.worktree, env=self._child_env(), timeout=1800)
        text = r.stdout + r.stderr
        counts = _parse_pytest_summary(text)
        if r.returncode != 0:
            tail = "\n".join(text.strip().splitlines()[-15:])
            node_ids = _failed_node_ids(text)
            base_failed, probe_note = _failures_at_base(
                self.python, self.live, self.base, node_ids,
                W.round_dir(self.round_id), self._child_env())
            external, new = _classify_test_failure(node_ids, base_failed)
            data = {**counts, "failed_node_ids": node_ids,
                    "base_probe": probe_note}
            if external:
                # The rung still fails: a red tree is not a tree to land onto,
                # and the guardian would judge the promotion against a broken
                # baseline. What changes is that this does not spend the
                # backlog item's one attempt — see `backlog.implemented_ids`.
                data["external_blocker"] = True
                data["external_failures"] = node_ids
                return False, (f"pytest failed ({counts}), but every failure "
                               f"reproduces at base {self.base[:8]} with this "
                               f"round's diff absent — PRE-EXISTING BREAKAGE, "
                               f"not caused by this change: {node_ids}. "
                               f"{probe_note}. Blocking the promotion; the item "
                               f"keeps its attempt."), data
            if node_ids and base_failed:
                data["new_failures"] = new
                return False, (f"pytest failed ({counts}): {len(new)} of "
                               f"{len(node_ids)} failures are new in this round "
                               f"({new}); the rest predate it. {probe_note}\n"
                               f"{tail[-600:]}"), data
            # Nothing reproduces at base: every failure is this round's. Say so
            # in the same field the mixed case uses, so a reader of the report
            # never has to infer the delta from its absence.
            data["new_failures"] = new
            return False, f"pytest failed ({counts}): {tail[-900:]}\n{probe_note}", data

        # Non-negotiable under auto-landing: `pytest -q` exits 0 if the round
        # simply deleted the test that was failing.
        if counts["collected"] < PYTEST_MIN_COLLECTED:
            return False, (f"only {counts['collected']} tests collected "
                           f"(floor {PYTEST_MIN_COLLECTED}) — did the round delete tests?"), counts
        # ...and it exits 0 just as happily having collected everything and RUN
        # nothing. A collected-count floor alone is satisfied by a suite that
        # skipped itself wholesale, which is a one-line conftest change away.
        if counts["passed"] < PYTEST_MIN_PASSED:
            return False, (f"only {counts['passed']} tests passed "
                           f"(floor {PYTEST_MIN_PASSED}) of {counts['collected']} "
                           f"collected, {counts['skipped']} skipped — the suite "
                           "was collected but not run"), counts
        if counts["skipped"] > PYTEST_MAX_SKIPPED:
            return False, (f"{counts['skipped']} tests skipped "
                           f"(limit {PYTEST_MAX_SKIPPED}) — a round that skips its way "
                           "to green is not a round that passed"), counts
        removed = [p for p in self.report.changed_paths
                   if p.startswith("tests/") and not (self.worktree / p).exists()]
        if removed:
            return False, f"test files removed: {removed}", counts
        return True, (f"{counts['passed']} passed, {counts['xfailed']} xfailed, "
                      f"{counts['skipped']} skipped"), counts

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
        if not self.item_id:
            return True, "SKIPPED (no backlog item bound to this round — no contract to grade)", {
                "skipped": True, "reason": "no item"}
        from scripts.automod import review as RV
        contract = RV.item_contract(self.item_id)
        if not contract["clauses"]:
            return False, f"item #{self.item_id} has no acceptance clauses", {}
        head = W.head(self.worktree) or self.report.head or ""
        prior = [e for e in S.read_events(limit=1000)
                 if e.get("event") == "review" and e.get("round_id") == self.round_id]
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
        # refusal of the text it had just replaced.
        pending_amendments = list(contract.get("amendments") or [])
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
        attempt = spent + 1
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
        changed_tests = [p for p in changed if p.startswith("tests/") and p.endswith(".py")]
        pre = RV.honesty_prechecks(self.worktree, self.base, changed,
                                   n_clauses=len(contract["clauses"]))
        test_counts = next((r.data for r in self.report.rungs if r.name == "tests"), {}) or {}
        started = time.time()
        snapshot, snap_note = self._review_snapshot(head)
        grade_root = snapshot or self.worktree
        base_event = {"event": "review", "round_id": self.round_id, "item_id": self.item_id,
                      "attempt": attempt, "head": head, "grader_model": "primary",
                      "snapshot": bool(snapshot), "snapshot_note": snap_note, "prechecks": pre,
                      # A refusal shown an amendment is a judgment of a new
                      # contract; `backlog.review_disagreement` reads this so
                      # it does not count the amended pass as a repeat.
                      "amendments_shown": [a.get("clause") for a in pending_amendments]}
        try:
            res = RV.grade(round_id=self.round_id, worktree=grade_root, base=self.base,
                           contract=contract, changed_paths=changed, test_counts=test_counts,
                           python=self.python, child_env=self._child_env(grade_root),
                           scratch_dir=W.round_dir(self.round_id) / "gate-state")
            base_event.update({"session_id": res.get("session_id"),
                               "seconds": round(time.time() - started, 1)})
            if not res["ok"]:
                S.append_event({**base_event, "ok": False, "blocking": False,
                                "error": str(res.get("error") or "")[:400]})
                return False, (f"review could not run: {res.get('error')} — the grader, not the "
                               f"diff; neither the item's attempt nor a review attempt is spent"), {
                                   "external_blocker": True, "external_failures": [],
                                   "external_reason": "grader unreachable",
                                   "retry_after_s": 120,
                                   "review_session": res.get("session_id")}
            parsed = RV.parse_review(res["structured"], worktree=grade_root,
                                     changed_tests=changed_tests, n_clauses=len(contract["clauses"]))
        finally:
            self._drop_snapshot(snapshot)
        if parsed is None:
            S.append_event({**base_event, "ok": False, "blocking": False,
                            "error": "structured review unusable"})
            return False, "review returned an unusable object; the item keeps its attempt", {
                "external_blocker": True, "external_failures": [],
                "external_reason": "grader returned an unusable object",
                "retry_after_s": 120,
                "review_session": res.get("session_id")}
        amendments = contract.get("amendments") or []
        kind, findings = RV.decide(parsed, pre, amendments=amendments,
                                   attempt=attempt, policy=_review_policy("seams_block"))
        S.append_event({**base_event, "ok": True, "premise": parsed["premise"],
                        "clauses": parsed["clauses"], "test_honesty": parsed["test_honesty"],
                        "seams_unverified": [s["seam"] if isinstance(s, dict) else s
                                             for s in parsed["seams_unverified"]],
                        "seams_untestable": [s["seam"] for s in parsed["seams_unverified"]
                                             if isinstance(s, dict)
                                             and not s.get("testable_before_landing", True)],
                        "downgraded": parsed["downgraded"], "summary": parsed["summary"],
                        "amendments_ok": parsed.get("amendments_ok", True),
                        "amendments_note": parsed.get("amendments_note", ""),
                        "blocking": kind != "pass", "kind": kind,
                        "findings": findings[:2000]})
        self._settle_amendments(amendments, parsed, kind)
        if kind == "unsound":
            return False, f"review: premise unsound — {findings}", {
                "review_premise_unsound": True, "review_summary": findings[:800],
                "review_session": res.get("session_id")}
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
            return False, f"review sent it back ({shown}; {nxt}): {findings}", {
                "review_retry": True, "review_findings": findings[:1500],
                "review_attempt": attempt, "review_session": res.get("session_id")}
        # On a PASS, record the grader's `post_landing` clauses onto the item.
        # Written here rather than by the implementer because it is a fact the
        # grader established about a change that is about to land, not a claim
        # the author made about its own work.
        from scripts.automod import backlog as _B
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
        return True, (f"review: {RV.summarize_clauses(parsed)} of {len(contract['clauses'])} "
                      f"clause(s); {parsed['summary'][:160]}"), {
                          "review_session": res.get("session_id"),
                          "clauses": parsed["clauses"], "review_attempt": attempt,
                          "post_landing_clauses": marked,
                          "amendments_ratified": [a.get("clause") for a in amendments]}

    def _review_snapshot(self, head: str) -> tuple[Path | None, str]:
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
        wt = W.round_dir(self.round_id) / "gate-state" / f"review-{head[:12]}"
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
            return True, "requirements unchanged — using the live venv", {}
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
                "skipped": True, "reason": self._smoke_skip_reason}
        rep = self._canary.smoke(timeout=240)
        if not rep["ok"]:
            return False, "; ".join(rep["errors"])[:900], {
                k: rep.get(k) for k in ("tool_called", "tool_result_ok", "done")}
        return True, (f"real turn in {rep['duration_s']}s; Bash dispatched, "
                      f"sentinel round-tripped"), {
            "duration_s": rep["duration_s"], "turns": rep.get("turns"),
            "sentinel_in_response": rep.get("sentinel_in_response")}

    def rung_drill(self):
        if not spec.requires_drill(self.report.changed_paths):
            return True, "no protected paths touched — drill not required", {}
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
