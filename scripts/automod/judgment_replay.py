"""The `judgment_replay` gate rung: a round that rewrites a judge is measured.

The loop can edit its own judges — the triage prompt, the implement prompt, the
owed-check prompt and the review grader — and until this module nothing asked
whether such an edit made the judging worse: every other rung reads the diff,
and a prompt that confirms less of the work that lands, or a grader that passes
a change with its tests taken out, is green on all of them.

**What fires it is the prompt text, not the file.** `SURFACES` names, per judge,
the module-level symbols a prompt is rendered from; a surface's fingerprint is a
hash of the AST of those symbols and of every module-level name they reach in
the same file, docstrings and comments left out. A round fires the rung only
when a fingerprint differs between its base and its head. Measured 2026-11-22
over the last 500 landings: 20 touched one of the five judging files and 3
moved a fingerprint (all three the owed-check prompt); over all 2,021 landings
on the ledger, 82 and 21.

**Two instruments exist, so two surfaces are measured and the rest are named.**

  triage   `eval/triage_replay/run.py`, run from the ROUND's worktree so the
           prompt is the round's, compared case by case with a recorded run of
           the base's prompt (`compare_triage`). Refused only when the cases
           that flipped from agreeing with their label to disagreeing outnumber
           the ones that flipped the other way by `TRIAGE_NET_FLIPS` and a
           one-sided sign test on the flips is under `TRIAGE_SIGN_ALPHA`.
           Repeat-run floor, measured 2026-11-22: 2 of 42 cases differ between
           two runs of one prompt (13 before the runner's finalizer; §4.6).
  grader   a fixed subset of the review calibration set (`GRADER_NEGATIVE`,
           the five constructed defects whose own tests stay green, then
           `GRADER_INDEPENDENT`), graded by the round's `review.py` through the
           live backend's review path. Refused on any NEW false `met`: a
           known-bad case the base's grader refused and the round's passes.

The batch triage prompts, the implement prompt, the owed-check prompt and the
grader's vault/confirm prompts have no replay set. A round that moves one of
them is recorded as `uncovered` — on `gate.json` and on the ledger row — and is
not refused for it: there is nothing to record that would let it through.

**A baseline is keyed by fingerprint, never by commit**, so one recorded run
stands for every base that renders the same prompt, and the round's own run is
written under its head's fingerprint — which is the next round's base. Only the
first baseline is recorded by hand (`record`, below). No baseline is
`unmeasured`, and in `enforce` that blocks as external with the command to
record one; it is never a pass.

**What is not a verdict.** A lock that stays held, an unreachable engine, a run
that timed out or came back incomplete: `external_blocker`, which costs the
round nothing. Never a pass and never a refusal.

Mode is `automod.gate.judgment_replay`: `off` (ships; the rung is skipped),
`advisory` (runs and records, refuses nothing), `enforce`. The loop cannot
change it — `spec.CONFIG_DENIED_KEYS`.

    python -m scripts.automod.judgment_replay status
    python -m scripts.automod.judgment_replay record triage    # ~20-40 min of primary
    python -m scripts.automod.judgment_replay record grader    # ~45-60 min of primary
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

MODES = ("off", "advisory", "enforce")
STORE_NAME = "judgment_replay.jsonl"
EVENT = "judgment_replay"

#: surface → instrument and `{path: root symbols}`. A root's reach inside its own
#: file is followed (`closure`); a name imported from elsewhere is not, so a
#: surface that splices text from a second module lists that module too.
SURFACES: dict[str, dict] = {
    "triage": {"instrument": "triage", "paths": {
        "workers/sources/autotriage.py": ("render_prompt",),
        "scripts/automod/item_class.py": ("splice_triage_prompt",)}},
    "grader": {"instrument": "grader", "paths": {
        "scripts/automod/review.py": ("build_prompt", "REVIEW_SCHEMA")}},
    "triage_batch": {"instrument": None, "paths": {
        "workers/sources/autotriage.py": ("GROUP_PROMPT", "SWEEP_PROMPT", "NO_UMBRELLA_RULE")}},
    "implement": {"instrument": None, "paths": {
        "workers/sources/autocode.py": ("_build_prompt",)}},
    "owed": {"instrument": None, "paths": {
        "workers/sources/owed_check.py": ("build_prompt", "OWED_SCHEMA")}},
    "grader_other": {"instrument": None, "paths": {
        "scripts/automod/review.py": ("build_vault_prompt", "build_confirm_prompt",
                                      "CONFIRM_SCHEMA_PROMPT")}},
}

TRIAGE_CASES = "eval/triage_replay/cases.json"
#: Net flips-to-wrong that can refuse, and the one-sided sign-test bound on the
#: flips. 0.125 is exactly three flips to wrong and none to right: the smallest
#: result that can refuse. Provisional — see the module docstring.
TRIAGE_NET_FLIPS = 3
TRIAGE_SIGN_ALPHA = 0.125
#: More head cases than this with no answer (engine error) and the run is
#: incomplete, not a verdict.
TRIAGE_MAX_ERRORS = 4

#: The constructed defects whose own tests stay green: a `pass` on one is a bad
#: change the tests rung would not have stopped. Graded first.
GRADER_NEGATIVE = ("4180-stripped-tests", "4274-stripped-tests", "4343-stripped-tests",
                   "4363-stripped-tests", "4393-dropped-doc")
#: Good landings, the smallest non-provisional six: recorded for the false
#: refusals a rewrite adds, which the rung reports and never refuses on.
GRADER_INDEPENDENT = ("3981-gh-token-fallback", "4056-arch-review-prose", "4215-ab-report-void",
                      "3991-steward-prose-fit", "4053-ambient-ttl-description",
                      "4117-llm-call-cap")

TRIAGE_TIMEOUT_S = 3600.0
GRADER_TIMEOUT_S = 3600.0
LOCK_MAX_WAIT_S = 1800.0
RETRY_AFTER_S = 600
#: Both, in this order. `regression.lock` is what a promotion's regression check
#: and every GPU experiment queue on; `primary.lock` is what the eval drivers
#: take before a sustained run on the primary.
LOCKS = ("regression.lock", "primary.lock")


class Blocked(Exception):
    """Something outside the diff stopped a measurement: not a verdict."""


# ── mode ─────────────────────────────────────────────────────────────────

def mode(value) -> str:
    """`off | advisory | enforce` from whatever the config holds.

    YAML reads a bare `off` as False and `on` as True, so both booleans are
    accepted. Anything unrecognised is `advisory`: a typo must neither switch
    the guard off silently nor start refusing rounds."""
    if value is None or value is False:
        return "off"
    if value is True:
        return "enforce"
    text = str(value).strip().lower()
    return text if text in MODES else "advisory"


def configured_mode() -> str:
    try:
        from app.config import CONFIG
        return mode(((CONFIG.get("automod") or {}).get("gate") or {}).get("judgment_replay"))
    except Exception:  # noqa: BLE001 — an unreadable config is the shipped default
        return "off"


# ── fingerprints ─────────────────────────────────────────────────────────

def _bindings(tree: ast.Module) -> dict[str, list[ast.stmt]]:
    out: dict[str, list[ast.stmt]] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        for name in names:
            out.setdefault(name, []).append(node)
    return out


def closure(source: str, roots) -> tuple[list[str], dict[str, list[ast.stmt]]]:
    """`roots` and every module-level name they reach in `source`, sorted."""
    bound = _bindings(ast.parse(source))
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        name = todo.pop()
        if name in seen or name not in bound:
            continue
        seen.add(name)
        for node in bound[name]:
            todo.extend(sub.id for sub in ast.walk(node)
                        if isinstance(sub, ast.Name) and sub.id in bound and sub.id not in seen)
    return sorted(seen), bound


def _without_docstrings(node: ast.AST) -> ast.AST:
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and sub.body:
            first = sub.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                sub.body = sub.body[1:] or [ast.Pass()]
    return node


def fingerprint_source(source: str | None, roots) -> str:
    """A hash of what `roots` render from. `absent` for no file, and a hash of
    the text itself for one that does not parse — a file that stopped parsing
    has changed."""
    if source is None:
        return "absent"
    try:
        names, bound = closure(source, roots)
    except SyntaxError:
        return "unparsed:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    h = hashlib.sha256()
    for name in names:
        for node in bound[name]:
            h.update(name.encode("utf-8") + b"\0")
            h.update(ast.dump(_without_docstrings(node)).encode("utf-8") + b"\0")
    return h.hexdigest()[:24]


def surface_fingerprint(surface: str, read: Callable[[str], str | None]) -> str:
    """One hash over every path of `surface`; `read(path)` is the file's text or None."""
    h = hashlib.sha256()
    for path, roots in sorted(SURFACES[surface]["paths"].items()):
        h.update(f"{path}={fingerprint_source(read(path), roots)}\n".encode("utf-8"))
    return h.hexdigest()[:24]


def changed_surfaces(read_base, read_head, changed_paths) -> list[str]:
    """The surfaces whose fingerprint differs. A surface none of whose paths is
    in the diff is not read at all."""
    changed = set(changed_paths or [])
    return [name for name, spec in SURFACES.items()
            if changed & set(spec["paths"])
            and surface_fingerprint(name, read_base) != surface_fingerprint(name, read_head)]


def git_reader(repo: Path, rev: str) -> Callable[[str], str | None]:
    def read(path: str) -> str | None:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"],
                           capture_output=True, text=True, timeout=60, check=False)
        return r.stdout if r.returncode == 0 else None
    return read


# ── the store ────────────────────────────────────────────────────────────

def store_path() -> Path:
    from scripts.automod import state as S
    return S.STATE_DIR / STORE_NAME


def read_runs(path: Path | None = None) -> list[dict]:
    path = path or store_path()
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("event") == EVENT and row.get("surface"):
            out.append(row)
    return out


def record_run(*, surface: str, fingerprint: str, verdicts: dict, cases_sha: str = "",
               source: str, round_id: str = "", commit: str = "", seconds: float = 0.0,
               now: float | None = None, path: Path | None = None) -> dict:
    now = time.time() if now is None else now
    row = {"event": EVENT, "surface": surface, "fingerprint": fingerprint,
           "cases_sha": cases_sha, "source": source, "round_id": round_id, "commit": commit,
           "ts": now, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
           "seconds": round(float(seconds), 1), "verdicts": verdicts}
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def text_sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:24]


# ── triage: paired, per case ─────────────────────────────────────────────

def triage_verdicts(rows: list[dict]) -> dict[str, dict]:
    """`{case key: {got, agree}}` from `run.py` rows, latest row per case. A case
    whose call errored is left out: no answer is not a disagreement."""
    out: dict[str, dict] = {}
    for r in rows:
        key = r.get("key")
        if not key:
            continue
        if r.get("error"):
            out.pop(key, None)
            continue
        out[key] = {"got": r.get("got"), "agree": bool(r.get("agree"))}
    return out


def sign_test_p(wrong: int, right: int) -> float:
    """One-sided: the chance of `wrong` or more of `wrong + right` flips going
    the bad way when a flip is a coin toss."""
    n = wrong + right
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wrong, n + 1)) / 2 ** n


def compare_triage(base: dict[str, dict], head: dict[str, dict], *, cases: int = 0) -> dict:
    """Flips between two runs of the same cases. `status` is `pass`, `refuse`,
    or `incomplete` (too few head answers to say anything)."""
    paired = sorted(set(base) & set(head))
    to_wrong = [k for k in paired if base[k]["agree"] and not head[k]["agree"]]
    to_right = [k for k in paired if not base[k]["agree"] and head[k]["agree"]]
    p = sign_test_p(len(to_wrong), len(to_right))
    total = cases or len(base)
    missing = total - len(head)
    out = {"paired": len(paired), "cases": total, "head_missing": max(0, missing),
           "base_agree": sum(base[k]["agree"] for k in paired),
           "head_agree": sum(head[k]["agree"] for k in paired),
           "flips_to_wrong": to_wrong, "flips_to_right": to_right,
           "net": len(to_wrong) - len(to_right), "sign_p": round(p, 4),
           "thresholds": {"net": TRIAGE_NET_FLIPS, "sign_alpha": TRIAGE_SIGN_ALPHA,
                          "provisional": True}}
    if missing > TRIAGE_MAX_ERRORS or not paired:
        out["status"] = "incomplete"
    elif out["net"] >= TRIAGE_NET_FLIPS and p <= TRIAGE_SIGN_ALPHA:
        out["status"] = "refuse"
    else:
        out["status"] = "pass"
    return out


# ── grader: new false met ────────────────────────────────────────────────

def grader_subset() -> list[tuple[str, str, str]]:
    """`(set, name, expected kind)`, known-bad cases first."""
    return ([("negative", n, "retry") for n in GRADER_NEGATIVE]
            + [("independent", n, "pass") for n in GRADER_INDEPENDENT])


def compare_grader(base: dict[str, str], head: dict[str, str]) -> dict:
    """`base`/`head` are `{"set/name": kind}`. Refuses on a known-bad case the
    base refused and the head passes; a head that did not grade every known-bad
    case is `incomplete`; a base that graded none of them is `unmeasured`."""
    neg = [f"{s}/{n}" for s, n, kind in grader_subset() if kind == "retry"]
    ind = [f"{s}/{n}" for s, n, kind in grader_subset() if kind == "pass"]
    graded = ("pass", "retry", "unsound")
    new_false_met = [k for k in neg if base.get(k) in ("retry", "unsound") and head.get(k) == "pass"]
    out = {"new_false_met": new_false_met,
           "standing_false_met": [k for k in neg if base.get(k) == "pass" and head.get(k) == "pass"],
           "unbaselined": [k for k in neg if base.get(k) not in graded],
           "head_ungraded": [k for k in neg + ind if head.get(k) not in graded],
           "new_false_refusal": [k for k in ind if base.get(k) == "pass"
                                 and head.get(k) in ("retry", "unsound")],
           "head": {k: head.get(k) for k in neg + ind}}
    if any(head.get(k) not in graded for k in neg):
        out["status"] = "incomplete"
    elif len(out["unbaselined"]) == len(neg):
        out["status"] = "unmeasured"
    elif new_false_met:
        out["status"] = "refuse"
    else:
        out["status"] = "pass"
    return out


def grader_baseline(fingerprint: str, *, runs: list[dict], calibration: list[dict],
                    fingerprint_at: Callable[[str], str]) -> dict[str, str]:
    """The newest recorded kind per case under this grader fingerprint: this
    module's own runs, and `review_calibration.jsonl` rows (`calibrate --record`,
    partial ones included) whose commit renders the same grader prompt."""
    stamped: list[tuple[float, dict]] = []
    for r in runs:
        if r.get("surface") == "grader" and r.get("fingerprint") == fingerprint:
            stamped.append((float(r.get("ts") or 0), r.get("verdicts") or {}))
    cache: dict[str, str] = {}
    for r in calibration:
        commit = str(r.get("commit") or "")
        if not commit:
            continue
        if commit not in cache:
            cache[commit] = fingerprint_at(commit)
        if cache[commit] == fingerprint:
            stamped.append((float(r.get("ts") or 0), r.get("verdicts") or {}))
    out: dict[str, str] = {}
    for _ts, verdicts in sorted(stamped, key=lambda pair: pair[0]):
        out.update({k: v for k, v in verdicts.items() if v})
    return out


def run_grader_subset(out: Path, *, label: str, grade=None, repo: Path | None = None,
                      root: Path | None = None) -> int:
    """Grade the subset with THIS tree's `review.py`, one JSON row per case
    appended to `out` as it finishes, so a run cut short keeps what it graded.
    Runs in the round's worktree as a child of the rung."""
    from scripts.automod import calibration_cases as CC
    from scripts.automod import review_tools as RT
    repo, root = repo or RT.LIVE_ROOT, root or RT.FIXTURE_DIR
    grade = grade or RT.grade_commit
    for set_name, name, _kind in grader_subset():
        started = time.monotonic()
        case = next((c for c in RT.load_fixtures(root / set_name) if c["name"] == name), None)
        row = {"key": f"{set_name}/{name}", "kind": None, "error": ""}
        try:
            if case is None:
                raise ValueError("no such fixture in this tree")
            CC.materialize(repo, case)
            result = grade(repo=repo, item_id=case["item_id"], parent=case["parent"],
                           commit=case["commit"], changed_paths=case["changed_paths"],
                           label=f"jr_{label}_{name}", strip_tests=case.get("strip_tests", False))
            row["kind"] = result.get("kind")
            row["error"] = str(result.get("error") or "")[:300]
            row["session_id"] = result.get("session_id")
        except Exception as exc:  # noqa: BLE001 — one case never costs the rest
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        row["seconds"] = round(time.monotonic() - started, 1)
        with Path(out).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


# ── children, locks ──────────────────────────────────────────────────────

def run_child(cmd: list[str], *, cwd: Path, env: dict | None, timeout: float) -> dict:
    """Run to completion or kill the whole process group at `timeout`.
    `{returncode, timed_out, tail}`."""
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return {"returncode": proc.returncode, "timed_out": False,
                "tail": "\n".join((out or "").strip().splitlines()[-12:])}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _ = proc.communicate()
        return {"returncode": None, "timed_out": True,
                "tail": "\n".join((out or "").strip().splitlines()[-12:])}


class _Locks:
    """Every lock in `LOCKS`, queued for, released in reverse."""

    def __init__(self, owner: str, max_wait: float = LOCK_MAX_WAIT_S):
        self.owner, self.max_wait, self._held = owner, max_wait, []

    def __enter__(self):
        from scripts.automod import state as S
        try:
            for name in LOCKS:
                self._held.append(S.Lock(S.STATE_DIR / name, owner=self.owner)
                                  .acquire_wait(self.max_wait))
        except S.LockHeld as exc:
            self.__exit__()
            raise Blocked(f"{name} still held after {self.max_wait:.0f}s ({exc})") from exc
        return self

    def __exit__(self, *exc) -> None:
        while self._held:
            self._held.pop().release()


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


# ── the rung ─────────────────────────────────────────────────────────────

class Deps:
    """Everything the rung reaches outside itself, so a test replaces each one.

    `gate` is a `gate.Gate`: `worktree`, `base`, `live`, `python`, `round_id`,
    `report.changed_paths` and `_child_env()` are what is read."""

    def __init__(self, gate):
        self.gate = gate

    def mode(self) -> str:
        return configured_mode()

    def read_base(self, path: str):
        return git_reader(self.gate.worktree, self.gate.base)(path)

    def read_head(self, path: str):
        return git_reader(self.gate.worktree, "HEAD")(path)

    def head_commit(self) -> str:
        r = subprocess.run(["git", "-C", str(self.gate.worktree), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=60, check=False)
        return r.stdout.strip()

    def runs(self) -> list[dict]:
        return read_runs()

    def calibration(self) -> list[dict]:
        from scripts.automod import calibration_cases as CC
        return CC.read_records(include_partial=True)

    def grader_fingerprint_at(self, commit: str) -> str:
        return surface_fingerprint("grader", git_reader(self.gate.live, commit))

    def record(self, **row) -> dict:
        return record_run(**row)

    def append_event(self, event: dict) -> None:
        from scripts.automod import state as S
        S.append_event(event)

    def locks(self):
        return _Locks(f"gate-{self.gate.round_id} judgment_replay")

    def scratch(self) -> Path:
        from scripts.automod import worktree as W
        d = W.round_dir(self.gate.round_id) / "gate-state" / "judgment_replay"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def engine(self) -> tuple[bool, str, str]:
        """`(up, base_url, why)` for the default model's endpoint."""
        from scripts.automod import gate as G
        import yaml
        up, why = G.engine_reachable(self.gate.live)
        raw = yaml.safe_load((self.gate.live / "config.yaml").read_text(encoding="utf-8")) or {}
        alias = (raw.get("model") or {}).get("default") or "primary"
        return up, str(((raw.get("models") or {}).get(alias) or {}).get("base_url") or ""), why

    def run_triage(self, base_url: str, out: Path) -> dict:
        g = self.gate
        return run_child([str(g.python), "eval/triage_replay/run.py", "run",
                          "--base-url", base_url, "--rows", str(out)],
                         cwd=g.worktree, env=g._child_env(), timeout=TRIAGE_TIMEOUT_S)

    def run_grader(self, out: Path) -> dict:
        g = self.gate
        return run_child([str(g.python), "-m", "scripts.automod.judgment_replay", "run-grader",
                          "--out", str(out), "--label", str(g.round_id)],
                         cwd=g.worktree, env=g._child_env(), timeout=GRADER_TIMEOUT_S)


RECORD_HINT = ("record one from the live tree, at a commit whose prompt is the base's: "
               "`.venvs/lloyd/bin/python -m scripts.automod.judgment_replay record {surface}`")


def _measure_triage(d: Deps) -> dict:
    fp_base = surface_fingerprint("triage", d.read_base)
    fp_head = surface_fingerprint("triage", d.read_head)
    cases_text = d.read_head(TRIAGE_CASES)
    sha = text_sha(cases_text)
    n_cases = len((json.loads(cases_text) or {}).get("cases") or []) if cases_text else 0
    out = {"fingerprint_base": fp_base, "fingerprint_head": fp_head, "cases_sha": sha}
    mine = [r for r in d.runs() if r.get("surface") == "triage" and r.get("cases_sha") == sha]
    base = next((r for r in reversed(mine) if r.get("fingerprint") == fp_base), None)
    if base is None:
        return {**out, "status": "unmeasured",
                "why": "no recorded triage replay of the base's prompt on these cases; "
                       + RECORD_HINT.format(surface="triage")}
    head = next((r for r in reversed(mine) if r.get("fingerprint") == fp_head), None)
    if head is not None:
        verdicts, out["head_run"] = head.get("verdicts") or {}, "recorded"
    else:
        up, base_url, why = d.engine()
        if not up or not base_url:
            raise Blocked(f"the engine is not answering ({why})")
        rows_path = d.scratch() / f"triage-{fp_head}-{len(mine)}.jsonl"
        with d.locks():
            ran = d.run_triage(base_url, rows_path)
        verdicts = triage_verdicts(_read_jsonl(rows_path))
        out["head_run"] = "ran"
        if ran.get("timed_out") or ran.get("returncode") != 0:
            raise Blocked(("the triage replay timed out" if ran.get("timed_out") else
                           f"the triage replay exited {ran.get('returncode')}")
                          + f" with {len(verdicts)}/{n_cases} cases answered: {ran.get('tail')}")
        if n_cases - len(verdicts) <= TRIAGE_MAX_ERRORS:
            d.record(surface="triage", fingerprint=fp_head, verdicts=verdicts, cases_sha=sha,
                     source="gate", round_id=str(d.gate.round_id), commit=d.head_commit())
    cmp_ = compare_triage(base.get("verdicts") or {}, verdicts, cases=n_cases)
    out.update(cmp_, baseline_at=base.get("created_at"))
    if cmp_["status"] == "incomplete":
        raise Blocked(f"the triage replay answered {len(verdicts)}/{n_cases} cases — "
                      f"the engine, not the prompt")
    return out


def _measure_grader(d: Deps) -> dict:
    fp_base = surface_fingerprint("grader", d.read_base)
    fp_head = surface_fingerprint("grader", d.read_head)
    out = {"fingerprint_base": fp_base, "fingerprint_head": fp_head}
    runs = d.runs()
    base = grader_baseline(fp_base, runs=runs, calibration=d.calibration(),
                           fingerprint_at=d.grader_fingerprint_at)
    neg = [f"negative/{n}" for n in GRADER_NEGATIVE]
    if not any(base.get(k) for k in neg):
        return {**out, "status": "unmeasured",
                "why": "no recorded grade of the known-bad cases under the base's grader "
                       "prompt; " + RECORD_HINT.format(surface="grader")}
    head = grader_baseline(fp_head, runs=runs, calibration=[], fingerprint_at=lambda c: "")
    out["head_run"] = "recorded"
    if not all(head.get(k) for k in neg):
        rows_path = d.scratch() / f"grader-{fp_head}-{len(runs)}.jsonl"
        with d.locks():
            ran = d.run_grader(rows_path)
        rows = _read_jsonl(rows_path)
        head = {r["key"]: r["kind"] for r in rows if r.get("key") and r.get("kind")}
        out.update(head_run="ran", timed_out=bool(ran.get("timed_out")),
                   errors={r["key"]: r.get("error") for r in rows if r.get("error")})
        if ran.get("returncode") not in (0, None):
            raise Blocked(f"the calibration run exited {ran.get('returncode')}: {ran.get('tail')}")
        if all(head.get(k) for k in neg):
            d.record(surface="grader", fingerprint=fp_head, verdicts=head, source="gate",
                     round_id=str(d.gate.round_id), commit=d.head_commit())
    cmp_ = compare_grader(base, head)
    out.update(cmp_)
    if cmp_["status"] == "incomplete":
        raise Blocked(f"the grader did not answer for {cmp_['head_ungraded']} — "
                      f"the engine or the backend, not the prompt")
    return out


_MEASURE = {"triage": _measure_triage, "grader": _measure_grader}


def _explain(surface: str, r: dict) -> str:
    if r["status"] == "refuse" and surface == "triage":
        return (f"triage replay: {len(r['flips_to_wrong'])} case(s) flipped to disagreeing with "
                f"their outcome ({', '.join(r['flips_to_wrong'][:6])}) against "
                f"{len(r['flips_to_right'])} the other way, sign test p={r['sign_p']} "
                f"(provisional thresholds: net ≥ {TRIAGE_NET_FLIPS}, p ≤ {TRIAGE_SIGN_ALPHA})")
    if r["status"] == "refuse":
        return (f"grader calibration: NEW false `met` on {', '.join(r['new_false_met'])} — a "
                f"known-bad change the base's grader refused and this one passes")
    if r["status"] == "pass" and surface == "triage":
        return (f"triage replay: {r['head_agree']}/{r['paired']} agree (base {r['base_agree']}), "
                f"{len(r['flips_to_wrong'])} to wrong, {len(r['flips_to_right'])} to right")
    if r["status"] == "pass":
        return (f"grader calibration: no new false met on {len(GRADER_NEGATIVE)} known-bad cases"
                + (f"; new false refusals: {', '.join(r['new_false_refusal'])}"
                   if r.get("new_false_refusal") else ""))
    return f"{surface}: {r['status']} — {r.get('why', '')}"


def rung(gate, deps: Deps | None = None) -> tuple[bool, str, dict]:
    """`Gate.rung_judgment_replay`. See the module docstring."""
    d = deps or Deps(gate)
    m = d.mode()
    if m == "off":
        return True, "automod.gate.judgment_replay is off", {"skipped": True, "reason": "off"}
    changed = changed_surfaces(d.read_base, d.read_head, gate.report.changed_paths)
    if not changed:
        return True, "no judging prompt changed", {"skipped": True, "reason": "not touched"}
    results: dict[str, dict] = {}
    for surface in changed:
        measure = _MEASURE.get(SURFACES[surface]["instrument"] or "")
        if measure is None:
            results[surface] = {"status": "uncovered",
                                "why": "no replay set exists for this prompt"}
            continue
        try:
            results[surface] = measure(d)
        except Blocked as exc:
            results[surface] = {"status": "external", "why": str(exc)}
        except Exception as exc:  # noqa: BLE001 — a broken instrument is not a verdict on the diff
            results[surface] = {"status": "external",
                                "why": f"the instrument failed: {type(exc).__name__}: {exc}"[:400]}
    statuses = {s: r["status"] for s, r in results.items()}
    overall = next((want for want in ("refuse", "external", "unmeasured", "pass", "uncovered")
                    if want in statuses.values()), "uncovered")
    detail = "; ".join(_explain(s, r) for s, r in results.items())
    data = {"mode": m, "status": overall, "surfaces": results}
    try:
        d.append_event({"event": EVENT + "_gate", "round_id": gate.round_id, "mode": m,
                        "status": overall, "statuses": statuses,
                        "detail": detail[:800]})
    except Exception:  # noqa: BLE001 — the record never costs the rung
        pass
    if m == "advisory":
        return True, f"ADVISORY ({overall}, would not refuse in this mode): {detail}", data
    if overall == "refuse":
        return False, detail, data
    if overall in ("external", "unmeasured"):
        return False, (f"judgment replay did not reach a verdict — outside the diff, nothing "
                       f"is spent: {detail}"), {
            **data, "external_blocker": True, "external_failures": [],
            "external_reason": ("no recorded baseline for a judging prompt this round changes"
                                if overall == "unmeasured" else
                                "the judgment replay could not run (lock, engine or timeout)"),
            "retry_after_s": RETRY_AFTER_S}
    return True, detail, data


# ── CLI ──────────────────────────────────────────────────────────────────

def _record(surface: str, root: Path, base_url: str) -> int:
    """Run one instrument on the tree at `root` and store it under that tree's
    fingerprint. A hand tool: it talks to the engine."""
    def read(path: str) -> str | None:
        p = root / path
        return p.read_text(encoding="utf-8") if p.exists() else None
    fp = surface_fingerprint(surface, read)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    out = store_path().parent / "judgment_replay_runs" / f"{surface}-{fp}-{os.getpid()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with _Locks(f"judgment_replay record {surface} {os.getpid()}"):
        if surface == "triage":
            ran = run_child([sys.executable, "eval/triage_replay/run.py", "run",
                             "--base-url", base_url, "--rows", str(out)],
                            cwd=root, env=None, timeout=TRIAGE_TIMEOUT_S)
            verdicts = triage_verdicts(_read_jsonl(out))
            sha = text_sha(read(TRIAGE_CASES))
        else:
            ran = run_child([sys.executable, "-m", "scripts.automod.judgment_replay", "run-grader",
                             "--out", str(out), "--label", "record"],
                            cwd=root, env=None, timeout=GRADER_TIMEOUT_S)
            verdicts = {r["key"]: r["kind"] for r in _read_jsonl(out)
                        if r.get("key") and r.get("kind")}
            sha = ""
    print(ran.get("tail") or "")
    if not verdicts:
        print(f"nothing graded (exit {ran.get('returncode')}, timed out {ran.get('timed_out')}); "
              f"nothing recorded. Rows, if any: {out}")
        return 1
    why = baseline_unusable(surface, verdicts)
    if why:
        print(f"NOT recorded: {why}. Rows: {out}")
        return 1
    row = record_run(surface=surface, fingerprint=fp, verdicts=verdicts, cases_sha=sha,
                     source="record", commit=commit, seconds=time.monotonic() - started)
    print(f"recorded {surface} baseline: {len(verdicts)} cases under fingerprint {fp} "
          f"at {commit[:8]} → {store_path()} ({row['created_at']})")
    return 0


def _status(root: Path) -> int:
    def read(path: str) -> str | None:
        p = root / path
        return p.read_text(encoding="utf-8") if p.exists() else None
    runs = read_runs()
    print(f"mode: {configured_mode()}   store: {store_path()} ({len(runs)} runs)")
    for surface, spec in SURFACES.items():
        fp = surface_fingerprint(surface, read)
        if not spec["instrument"]:
            print(f"  {surface:13} {fp}  uncovered (no replay set)")
            continue
        have = [r for r in runs if r.get("surface") == surface and r.get("fingerprint") == fp]
        if have:
            line = f"baseline {have[-1]['created_at']} ({len(have[-1].get('verdicts') or {})} cases)"
        elif surface == "grader":
            line = _grader_status_from_calibration(root, fp, runs)
        else:
            line = "UNMEASURED at this tree"
        print(f"  {surface:13} {fp}  {line}")
    return 0


def _grader_status_from_calibration(root: Path, fp: str, runs: list[dict]) -> str:
    """What the rung itself would read for the grader: `calibrate --record` rows
    count when their commit renders the same grader prompt (`grader_baseline`).
    Reporting UNMEASURED while the rung had a full baseline sent a reader off to
    spend an hour of engine time re-recording one (2026-11-23)."""
    from scripts.automod import calibration_cases as CC
    try:
        calibration = CC.read_records(include_partial=True)
    except Exception:  # noqa: BLE001
        calibration = []
    base = grader_baseline(fp, runs=runs, calibration=calibration,
                           fingerprint_at=lambda c: surface_fingerprint("grader", git_reader(root, c)))
    keys = [f"{s}/{n}" for s, n, _kind in grader_subset()]
    covered = [k for k in keys if base.get(k)]
    neg = [k for k in keys if k.startswith("negative/")]
    if not any(base.get(k) for k in neg):
        return "UNMEASURED at this tree (no calibration row renders this grader prompt)"
    return f"baseline from `calibrate --record` rows ({len(covered)}/{len(keys)} subset cases)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    rec = sub.add_parser("record", help="run one instrument on this tree and store the baseline")
    rec.add_argument("surface", choices=sorted(_MEASURE))
    rec.add_argument("--base-url", default="http://127.0.0.1:8096",
                     help="the engine the triage replay talks to")
    run = sub.add_parser("run-grader", help="internal: grade the subset with this tree's review.py")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--label", default="hand")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        return _status(LIVE_ROOT)
    if args.cmd == "record":
        try:
            return _record(args.surface, LIVE_ROOT, args.base_url)
        except Blocked as exc:
            print(f"not run, nothing recorded: {exc}")
            return 2
    return run_grader_subset(args.out, label=args.label)

# ── a baseline the instrument could not read is not a baseline ───────────────
#
# The first `record triage` (2026-11-22) stored 36 `no_verdict` answers of 42:
# the replay's completion spent its whole token budget reasoning and wrote no
# content. Paired against that, any later run would have shown 30-odd flips to
# RIGHT and passed whatever it did. `no_verdict` from a HEAD run stays a
# disagreement — a prompt that stops the model answering is a regression — but
# a baseline made mostly of them measured the harness, so `record` refuses it.
# The newest row per fingerprint is the baseline, so a good run recorded after
# a bad one supersedes it without the store being edited.
BASELINE_MAX_NO_VERDICT_SHARE = 0.25


def baseline_unusable(surface: str, verdicts: dict) -> str:
    """Why this run may not stand as a baseline, or `""`."""
    if surface != "triage" or not verdicts:
        return ""
    silent = sum(1 for v in verdicts.values()
                 if isinstance(v, dict) and v.get("got") == "no_verdict")
    if silent / len(verdicts) > BASELINE_MAX_NO_VERDICT_SHARE:
        return (f"{silent} of {len(verdicts)} cases returned no verdict — the replay "
                f"did not get answers (token budget, reasoning mode or engine), so "
                f"this measures the harness and not the prompt")
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
