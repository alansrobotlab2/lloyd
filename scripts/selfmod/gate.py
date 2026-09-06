"""The promotion gate: seven rungs, cheapest first, every one fails closed.

"Fails closed" is not a slogan here — it is the reason `_rung` catches every
exception and records it as a FAILED rung. With no human review tier, a rung
that errors and is read as "didn't fail" would silently remove a check.

The gate always runs **from the live tree** against the worktree, so a
candidate that weakens the gate is judged by the old gate, not by itself.

Rung order is a cost ladder, and it short-circuits: an import error should
cost 3 seconds, not a full canary boot.

  0 preflight      ~1s    lock, clean tree, ancestry, diff scope
  1 static         ~8s    compileall, import smoke, pyflakes delta
  2 tests         ~35s    full pytest + a collected-count floor
  3 venv        0-300s    only when requirements changed (reflink + uv)
  4 canary boot   ~30s    both /health green, tool floor, config-follows-code
  5 canary smoke  ~30s    one real turn, sentinel through a real Bash call
  6 drill         ~90s    only when the diff touches the rollback path
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.selfmod import canary as C
from scripts.selfmod import spec, state as S, worktree as W

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
PYTEST_MIN_COLLECTED = 1000
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
    out: set[str] = set()
    for line in (r.stdout + r.stderr).splitlines():
        # "path:LINE:COL: message" → "path: message"; a finding that merely
        # moved down the file is not a new finding.
        m = re.match(r"^(.*?):\d+:\d+:\s*(.*)$", line.strip())
        if m:
            out.add(f"{m.group(1)}: {m.group(2)}")
        elif line.strip():
            out.add(line.strip())
    return out


def _parse_pytest_summary(text: str) -> dict:
    """Pull counts out of pytest's summary line."""
    out: dict = {"passed": 0, "failed": 0, "errors": 0, "xfailed": 0, "collected": 0}
    m = re.search(r"collected (\d+) item", text)
    if m:
        out["collected"] = int(m.group(1))
    for key, pattern in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"),
                         ("errors", r"(\d+) error"), ("xfailed", r"(\d+) xfailed")):
        m = re.search(pattern, text)
        if m:
            out[key] = int(m.group(1))
    if not out["collected"]:
        out["collected"] = out["passed"] + out["failed"] + out["xfailed"] + out["errors"]
    return out


class Gate:
    def __init__(self, round_id: str, worktree: Path, base: str, *,
                 live_root: Path | None = None, skip_smoke: bool = False):
        self.round_id = round_id
        self.worktree = Path(worktree)
        self.base = base
        self.live = live_root or LIVE_ROOT
        self.skip_smoke = skip_smoke
        self.python = self.live / ".venvs" / "lloyd" / "bin" / "python"
        self.report = GateReport(round_id=round_id, base=base,
                                 head=W.head(self.worktree) or "")

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
        S.append_event({"event": "gate", "round_id": self.round_id, "rung": name,
                        "ok": ok, "detail": res.detail[:500],
                        "seconds": round(res.seconds, 2)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({res.seconds:.1f}s) {res.detail[:160]}")
        return ok

    def run(self) -> GateReport:
        ladder = [("preflight", self.rung_preflight), ("static", self.rung_static),
                  ("tests", self.rung_tests), ("venv", self.rung_venv),
                  ("canary_boot", self.rung_canary_boot)]
        if not self.skip_smoke:
            ladder.append(("canary_smoke", self.rung_canary_smoke))
        ladder.append(("drill", self.rung_drill))

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
        if not W.is_clean(self.live):
            return False, "live tree is dirty — refusing to gate against a moving base", {}

        head = W.head(self.worktree)
        if not head:
            return False, "cannot read the worktree HEAD", {}
        self.report.head = head

        live_head = _run(["git", "-C", str(self.live), "rev-parse", "HEAD"]).stdout.strip()
        if live_head != self.base:
            return False, f"live HEAD moved: {live_head[:8]} != base {self.base[:8]}", {}

        anc = _run(["git", "-C", str(self.live), "merge-base", "--is-ancestor",
                    self.base, head])
        if anc.returncode != 0:
            return False, "candidate is not a descendant of base — not a fast-forward", {}
        if W.has_merge_commits(self.live, self.base, head):
            return False, "candidate contains merge commits", {}

        changed = W.changed_paths(self.worktree, self.base)
        self.report.changed_paths = changed
        if not changed:
            return False, "no changes to promote", {}

        ok, reason, buckets = spec.check_scope(changed)
        if not ok:
            return False, reason, {"buckets": buckets}

        for port in (C.cc.BACKEND_PORT, C.cc.MCP_PORT):
            if not C.port_free(port):
                return False, f"canary port {port} is in use (stale canary?)", {}

        return True, (f"{len(changed)} file(s) in scope"
                      + (f"; {len(buckets['protected'])} protected → drill required"
                         if buckets["protected"] else "")), {"buckets": buckets}

    def rung_static(self):
        r = _run([str(self.python), "-m", "compileall", "-q", str(self.worktree)], timeout=300)
        if r.returncode != 0:
            return False, f"compileall failed: {(r.stdout + r.stderr)[-800:]}", {}

        # The single highest-value cheap check: an import-time failure is the
        # number one way a self-modification bricks the boot, and this catches
        # it in seconds without binding a port or running a startup hook.
        env = {"PYTHONPATH": str(self.worktree), "PATH": "/usr/bin:/bin",
               "HOME": str(Path.home())}
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

    def rung_tests(self):
        env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
               "PYTHONPATH": str(self.worktree)}
        r = _run([str(self.python), "-m", "pytest", "-q"], cwd=self.worktree,
                 env=env, timeout=1800)
        text = r.stdout + r.stderr
        counts = _parse_pytest_summary(text)
        if r.returncode != 0:
            tail = "\n".join(text.strip().splitlines()[-15:])
            return False, f"pytest failed ({counts}): {tail[-900:]}", counts

        # Non-negotiable under auto-landing: `pytest -q` exits 0 if the round
        # simply deleted the test that was failing.
        if counts["collected"] < PYTEST_MIN_COLLECTED:
            return False, (f"only {counts['collected']} tests collected "
                           f"(floor {PYTEST_MIN_COLLECTED}) — did the round delete tests?"), counts
        removed = [p for p in self.report.changed_paths
                   if p.startswith("tests/") and not (self.worktree / p).exists()]
        if removed:
            return False, f"test files removed: {removed}", counts
        return True, f"{counts['passed']} passed, {counts['xfailed']} xfailed", counts

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

        env = {"PYTHONPATH": str(self.worktree), "PATH": "/usr/bin:/bin",
               "HOME": str(Path.home())}
        imp = _run([str(clone / "bin" / "python"), "-c", "import server, agent_mcp.main"],
                   cwd=self.worktree, env=env, timeout=300)
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
        from scripts.selfmod import rehearse
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
