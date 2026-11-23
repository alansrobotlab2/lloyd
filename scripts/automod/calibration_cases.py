"""Known-bad calibration cases for the review rung, and the lock on every label.

`review_tools calibrate` could say how often the grader refuses good work
(`independent/`, all labelled `pass`) and nothing about the costly error — a
`pass` on a change that should have been refused — because no case anywhere was
known to be bad. The ledger cannot supply one (every rollback on record was a
false positive), so this module makes them, three ways, and none of the three
asks the grader what the label is:

  constructed        a landed round with one thing a clause demands taken back
                     out: its tests (`strip_tests`), one changed source file
                     (`revert_source`), or the doc update a clause names
                     (`drop_doc`). Each is "restore these paths to the parent's
                     version", so the label is true by construction: the clause
                     that names the removed thing cannot be met by a tree that
                     does not carry it.
  refused_then_fixed a head the rung refused, in a round whose later head
                     landed, kept only when a reader confirmed from the diff
                     between the two heads that the refusal named a real defect
                     the later head repaired. These are selected ON the grader
                     having refused once, so their false-met rate is reported
                     apart and is a floor on agreement, not a rate.

A constructed case's commit does not exist in any repo until it is asked for.
`materialize` rebuilds it with git plumbing (a private index, `write-tree`,
`commit-tree` under a fixed identity and date), so the sha is a pure function of
`(source_commit, parent, transform)` and the fixture can pin it: a rebuilt
object that hashes to anything else is refused rather than graded. Nothing is
checked out and no ref is written; the object is loose in the repo it was built
in, which is where `grade_commit` then makes its detached worktree.

Labels are frozen by `eval/review_calibration/labels.lock`: one line per case
of the `independent` and `negative` sets, carrying its label and a hash of
`(name, label, parent, commit, source_commit, transform)`. A fixture that no
longer matches its line fails `tests/test_calibration_cases.py`, and
`write_lock` refuses to re-pin a changed label unless the fixture carries a
`relabel` record naming a reader who is not the grader.

A graded run is one row in `review_calibration.jsonl` in the automod state dir
(`calibrate --record`), so the two rates are a trend; `scorecard_reading` is the
reader for a scorecard row.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from eval.stats import wilson_ci
from scripts.automod import review_tools as RT

NEGATIVE_SET = "negative"
#: The sets whose labels are frozen. The four top-level cases are not among
#: them: two were re-marked toward the grader in 2026-10 and say so in the
#: README, which is the history this lock exists to stop repeating.
LOCKED_SETS = ("independent", NEGATIVE_SET)
LOCK_NAME = "labels.lock"
TRANSFORMS = ("strip_tests", "revert_source", "drop_doc")
LABEL_SOURCES = ("constructed", "refused_then_fixed")
#: Who may not be named as the reader behind a relabel. The grader's answer is
#: the thing being measured, so it is never the reason a label moves.
NOT_A_READER = ("grader", "primary", "review", "review rung", "calibrate", "the loop", "lloyd")
RECORD_NAME = "review_calibration.jsonl"

# A constructed commit's identity. Changing any of these changes every sha in
# `negative/`, and with it every line of the lock.
_IDENT = {"GIT_AUTHOR_NAME": "review-calibration", "GIT_AUTHOR_EMAIL": "calibration@lloyd.local",
          "GIT_AUTHOR_DATE": "2026-11-22T00:00:00+00:00",
          "GIT_COMMITTER_NAME": "review-calibration",
          "GIT_COMMITTER_EMAIL": "calibration@lloyd.local",
          "GIT_COMMITTER_DATE": "2026-11-22T00:00:00+00:00"}


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       timeout=180, check=False, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def has_object(repo: Path, sha: str) -> bool:
    r = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
                       capture_output=True, check=False, timeout=60)
    return r.returncode == 0


def _message(source: str, transform: dict) -> str:
    return (f"calibration: {transform['kind']} of {source}\n\n"
            + "\n".join(f"restored to parent: {p}" for p in sorted(transform["paths"])) + "\n")


def build_commit(repo: Path, *, source: str, parent: str, transform: dict,
                 write: bool = True) -> str:
    """The commit `source` becomes when `transform.paths` go back to `parent`.

    A path the parent did not have is removed. Built in a private index, so the
    repo's own index, worktree, HEAD and refs are untouched; the only write is
    loose objects. Deterministic: same inputs, same sha, in any clone.

    `write=False` computes the sha and leaves the repo byte-identical: the new
    tree and commit go to a scratch object directory that reads the repo's own
    as an alternate. That is what a test uses to check a pinned sha."""
    if transform.get("kind") not in TRANSFORMS:
        raise ValueError(f"transform kind must be one of {TRANSFORMS}")
    paths = sorted(transform.get("paths") or [])
    if not paths:
        raise ValueError("a transform restores at least one path")
    scratch = Path(tempfile.mkdtemp(prefix="calib-index-"))
    env = {"GIT_INDEX_FILE": str(scratch / "index")}
    if not write:
        common = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        (scratch / "objects").mkdir()
        env.update(GIT_OBJECT_DIRECTORY=str(scratch / "objects"),
                   GIT_ALTERNATE_OBJECT_DIRECTORIES=str(common / "objects"))
    try:
        _git(repo, "read-tree", source, env=env)
        for p in paths:
            was = _git(repo, "ls-tree", parent, "--", p)
            if was:
                mode, _type, sha = was.split("\t")[0].split()
                _git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{sha},{p}", env=env)
            else:
                _git(repo, "update-index", "--force-remove", "--", p, env=env)
        tree = _git(repo, "write-tree", env=env)
        if tree == _git(repo, "rev-parse", f"{source}^{{tree}}"):
            raise ValueError(f"restoring {paths} changes nothing: {source} does not differ "
                             f"from its parent there")
        return _git(repo, "commit-tree", tree, "-p", parent, "-m", _message(source, transform),
                    env={**env, **_IDENT})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def materialize(repo: Path, case: dict, *, write: bool = True) -> str:
    """Make sure the commit a case grades exists in `repo`; return its sha.

    A case with no `transform` names a real commit, which only has to be there.
    A constructed one is rebuilt, and must hash to the sha the fixture pins.
    `write=False` checks the same things and leaves no object behind."""
    t = case.get("transform")
    if not t:
        if not has_object(repo, case["commit"]):
            raise RuntimeError(f"{case['name']}: commit {case['commit'][:12]} is not in {repo}")
        return case["commit"]
    for sha in (case["source_commit"], case["parent"]):
        if not has_object(repo, sha):
            raise RuntimeError(f"{case['name']}: {sha[:12]} is not in {repo}")
    built = build_commit(repo, source=case["source_commit"], parent=case["parent"], transform=t,
                         write=write)
    if built != case["commit"]:
        raise RuntimeError(f"{case['name']}: rebuilt as {built[:12]}, the fixture pins "
                           f"{case['commit'][:12]} — the case is not the one that was labelled")
    return built


# ── writing a case ───────────────────────────────────────────────────────

def changed_paths(repo: Path, parent: str, commit: str) -> list[str]:
    return [p for p in _git(repo, "diff", "--name-only", parent, commit).splitlines() if p]


def constructed_case(*, repo: Path, name: str, item_id: int, round_id: str, parent: str,
                     source_commit: str, kind: str, paths: list[str], violates: list[int],
                     clause_text: dict[int, str], why: str, tests_rung: str = "not run",
                     proof: dict | None = None) -> dict:
    """A `retry` case made by restoring `paths` of a landed round to its parent.

    `violates` is the 1-based clauses the removal makes unmeetable and
    `clause_text` their words when the case was built — the item can be amended
    later, and the label was true of these words. `tests_rung` records what the
    round's own test files did on the built tree (`green`, `red`, `not run`): a
    `green` case is one only the review stands in front of. `proof` is a fixed
    string the built tree must (`present: true`) or must not carry in one file —
    the clause's own demand, checkable without a model (`verify_case`)."""
    transform = {"kind": kind, "paths": sorted(paths)}
    commit = build_commit(repo, source=source_commit, parent=parent, transform=transform)
    case = _constructed(name, item_id, round_id, parent, commit, source_commit, transform,
                        changed_paths(repo, parent, commit), violates, clause_text, tests_rung, why)
    if proof:
        case["proof"] = {"path": proof["path"], "text": proof["text"],
                         "present": bool(proof["present"])}
    problems = verify_case(repo, case)
    if problems:
        raise ValueError("; ".join(problems))
    return case


def _constructed(name, item_id, round_id, parent, commit, source_commit, transform, changed,
                 violates, clause_text, tests_rung, why) -> dict:
    return {"name": name, "item_id": int(item_id), "round_id": round_id, "parent": parent,
            "commit": commit, "source_commit": source_commit, "transform": transform,
            "changed_paths": changed, "strip_tests": False,
            "expect": {"kind": "retry", "unmet_clauses": sorted(int(i) for i in violates),
                       "honesty_finding": False},
            "label_source": "constructed",
            "violates": [{"clause": int(i), "text": clause_text[int(i)]} for i in sorted(violates)],
            "tests_rung": tests_rung, "provisional": False,
            "note": (f"Label by construction, not by any reader or grader: {why} Never re-mark "
                     "this case to match a grader's answer.")}


def _blob(repo: Path, commit: str, path: str) -> str:
    return _git(repo, "ls-tree", commit, "--", path)


def verify_case(repo: Path, case: dict) -> list[str]:
    """Re-derive what a label rests on from git alone. Empty means it still holds.

    For a constructed case, three facts, none a judgment: the pinned sha is what
    the transform builds; every restored path differed between the landed commit
    and its parent (so restoring it removed something) and is out of the case's
    `changed_paths`; and the `proof` string is where the fixture says it is. A
    `strip_tests` case additionally leaves no test path in its diff. For a
    refused head only the commit and the `proof` can be checked — the rest of
    its label is the reader's `evidence`. Writes nothing to `repo`."""
    out = []
    try:
        materialize(repo, case, write=False)
    except (RuntimeError, ValueError) as exc:
        return [str(exc)]
    t = case.get("transform") or {"paths": [], "kind": ""}
    for p in t["paths"]:
        if _blob(repo, case["source_commit"], p) == _blob(repo, case["parent"], p):
            out.append(f"{p} never differed from the parent: restoring it removes nothing")
    if set(t["paths"]) & set(case["changed_paths"]):
        out.append("a restored path is still listed in changed_paths")
    if t["kind"] == "strip_tests":
        from scripts.automod import testpaths as TP
        left = [p for p in case["changed_paths"] if TP.is_test_path(p, repo)]
        if left:
            out.append(f"strip_tests left test paths in the diff: {left}")
    proof = case.get("proof")
    if proof:
        # A restored path IS the parent's blob, so it is read there: the built
        # commit need not exist for the proof to be checked.
        at = case["parent"] if proof["path"] in t["paths"] else case["commit"]
        r = subprocess.run(["git", "-C", str(repo), "show", f"{at}:{proof['path']}"],
                           capture_output=True, text=True, timeout=60, check=False)
        has = r.returncode == 0 and proof["text"] in r.stdout
        if has != bool(proof["present"]):
            out.append(f"proof failed: {proof['text']!r} is "
                       f"{'present in' if has else 'absent from'} {proof['path']}")
    return out


def refused_head_case(*, repo: Path, name: str, item_id: int, round_id: str, parent: str,
                      refused_head: str, fixed_head: str, violates: list[int],
                      clause_text: dict[int, str], refusal: str, evidence: str,
                      read_by: str, read_on: str, honesty: bool = False,
                      proof: dict | None = None) -> dict:
    """A `retry` case that is a real refused head, with the reader's evidence.

    `refusal` is what the rung said at the time and `evidence` what the reader
    found in `git diff refused_head fixed_head` that shows the defect was real
    and was repaired. Without both there is no case."""
    if not (refusal.strip() and evidence.strip() and read_by.strip()):
        raise ValueError("a refused-head case needs the refusal, the reader and the evidence")
    if read_by.strip().lower() in NOT_A_READER:
        raise ValueError(f"{read_by!r} is the grader, not an independent reader")
    for sha in (parent, refused_head, fixed_head):
        if not has_object(repo, sha):
            raise RuntimeError(f"{name}: {sha[:12]} is not in {repo}")
    case = {"name": name, "item_id": int(item_id), "round_id": round_id, "parent": parent,
            "commit": refused_head, "fixed_by": fixed_head,
            "changed_paths": changed_paths(repo, parent, refused_head), "strip_tests": False,
            "expect": {"kind": "retry", "unmet_clauses": sorted(int(i) for i in violates),
                       "honesty_finding": bool(honesty)},
            "label_source": "refused_then_fixed",
            "violates": [{"clause": int(i), "text": clause_text[int(i)]} for i in sorted(violates)],
            "refusal": refusal, "evidence": evidence, "provisional": False,
            "note": (f"Label by a diff read on {read_on} ({read_by}; given the refusal text and "
                     "`git diff <refused head> <landed head>`). The case was SELECTED on the "
                     "grader having refused it once, so it cannot count toward an unbiased "
                     "false-met rate; it is reported apart. Never re-mark this case to match a "
                     "grader's answer.")}
    if proof:
        case["proof"] = {"path": proof["path"], "text": proof["text"],
                         "present": bool(proof["present"])}
    problems = verify_case(repo, case)
    if problems:
        raise ValueError("; ".join(problems))
    return case


def write_case(case: dict, fixture_dir: Path) -> Path:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_dir / f"{case['name']}.json"
    path.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return path


# ── the label lock ───────────────────────────────────────────────────────

def case_identity(case: dict) -> str:
    """What a label is a label OF: the name, the label, and the diff."""
    t = case.get("transform") or {}
    blob = json.dumps([case["name"], case["expect"]["kind"], case["parent"], case["commit"],
                       case.get("source_commit") or "", t.get("kind") or "",
                       sorted(t.get("paths") or [])], separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def lock_path(root: Path | None = None) -> Path:
    return (root or RT.FIXTURE_DIR) / LOCK_NAME


def current_entries(root: Path | None = None, sets=LOCKED_SETS) -> dict[str, tuple[str, str]]:
    """`{"<set>/<name>": (label, identity)}` for every case on disk."""
    root = root or RT.FIXTURE_DIR
    out = {}
    for s in sets:
        for case in RT.load_fixtures(root / s):
            out[f"{s}/{case['name']}"] = (case["expect"]["kind"], case_identity(case))
    return out


def read_lock(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and not line.startswith("#"):
            out[parts[0]] = (parts[1], parts[2])
    return out


RELABEL_RULE = (
    "A calibration label is frozen. To change one, add to the fixture a `relabel` record "
    '{"from": <old label>, "to": <new label>, "by": <who re-read the diff>, "on": <date>, '
    '"reason": <what the re-read found>} and regenerate the lock with '
    "`python -m scripts.automod.review_tools lock-labels`. The grader's answer is never a "
    "valid reason: these cases exist to measure it.")


def _relabel_ok(case: dict, old: str, new: str) -> str:
    """'' when the fixture's `relabel` record licenses `old -> new`, else why not."""
    r = case.get("relabel")
    if not isinstance(r, dict):
        return "no `relabel` record"
    if r.get("from") != old or r.get("to") != new:
        return f"`relabel` records {r.get('from')} -> {r.get('to')}, the change is {old} -> {new}"
    by = str(r.get("by") or "").strip()
    if not by or by.lower() in NOT_A_READER:
        return f"`relabel.by` is {by!r}: the reader must be named and must not be the grader"
    if len(str(r.get("reason") or "").strip()) < 20 or not str(r.get("on") or "").strip():
        return "`relabel` needs a dated `on` and a `reason` saying what the re-read found"
    return ""


def check_lock(root: Path | None = None) -> list[str]:
    """Every way the fixtures on disk differ from the lock. Empty means frozen."""
    root = root or RT.FIXTURE_DIR
    now, locked = current_entries(root), read_lock(lock_path(root))
    out = []
    for key in sorted(set(now) | set(locked)):
        if key not in locked:
            out.append(f"{key}: not in {LOCK_NAME} (a new case is locked when it is added)")
        elif key not in now:
            out.append(f"{key}: locked, and its fixture is gone or no longer loads")
        elif now[key] != locked[key]:
            what = (f"label {locked[key][0]} -> {now[key][0]}" if now[key][0] != locked[key][0]
                    else "the commit, parent or transform it labels changed")
            out.append(f"{key}: {what}")
    return out


def write_lock(root: Path | None = None) -> Path:
    """Re-pin the lock. Refuses a changed label the fixture does not account for,
    and refuses a case whose diff moved under an unchanged name."""
    root = root or RT.FIXTURE_DIR
    path = lock_path(root)
    now, locked = current_entries(root), read_lock(path)
    cases = {f"{s}/{c['name']}": c for s in LOCKED_SETS for c in RT.load_fixtures(root / s)}
    refused = []
    for key, (label, ident) in sorted(now.items()):
        if key not in locked or locked[key] == (label, ident):
            continue
        old = locked[key][0]
        if old == label:
            refused.append(f"{key}: the diff under this label changed; that is a new case, "
                           "so give it a new name")
            continue
        why = _relabel_ok(cases[key], old, label)
        if why:
            refused.append(f"{key}: label {old} -> {label} refused — {why}")
    gone = sorted(set(locked) - set(now))
    refused.extend(f"{k}: locked case removed; a case is retired by hand-editing {LOCK_NAME} "
                   "in a reviewed commit, never by regenerating" for k in gone)
    if refused:
        raise ValueError("\n".join(refused) + "\n" + RELABEL_RULE)
    lines = ["# Frozen labels of the review-calibration sets. Generated: "
             "python -m scripts.automod.review_tools lock-labels",
             "# <set>/<case> <label> <sha256[:24] of name, label, parent, commit, source, transform>"]
    lines += [f"{k} {v[0]} {v[1]}" for k, v in sorted(now.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── running several sets, and the two rates ──────────────────────────────

def calibrate_sets(sets: list[str], *, repo: Path | None = None, root: Path | None = None,
                   grader=None, only: str | None = None,
                   seams_policy: str | None = None) -> tuple[list[dict], list[dict]]:
    """`RT.calibrate` over each named set, constructed commits built first.

    Returns `(rows, cases)`; each row and case carries its `set`. A case whose
    commit cannot be produced is a row with no `kind` — ungraded, never a
    refusal the grader did not make."""
    repo, root = repo or RT.LIVE_ROOT, root or RT.FIXTURE_DIR
    rows, cases = [], []
    for s in sets:
        d = root / s
        mine = [c for c in RT.load_fixtures(d) if not only or c["name"] == only]
        missing = {}
        for c in mine:
            c["set"] = s
            try:
                materialize(repo, c)
            except (RuntimeError, ValueError) as exc:
                missing[c["name"]] = str(exc)
        cases.extend(mine)
        for name, why in missing.items():
            rows.append({"name": name, "set": s, "ok": False, "why": f"not graded: {why}",
                         "kind": None, "cause_class": "unlabelled", "findings": why,
                         "session_id": None, "provisional": False, "seconds": 0.0, "result": {}})
        for c in mine:
            if c["name"] in missing:
                continue
            for r in RT.calibrate(repo=repo, fixture_dir=d, grader=grader, only=c["name"],
                                  seams_policy=seams_policy):
                rows.append({**r, "set": s})
    return rows, cases


def _interval(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    lo, hi = wilson_ci(k, n)
    return [round(lo, 4), round(hi, 4)]


def rates(rows: list[dict], cases: list[dict]) -> dict:
    """False met and false refusal over everything graded, each with its Wilson
    interval, and false met again per `label_source` — a refused-then-fixed case
    was chosen because the grader refused it once, so it is never folded into
    the constructed rate silently. `false_met_review_only` is the same count over
    the constructed cases whose own tests stayed green on the defect: the ones a
    `pass` would actually have landed, since a red `tests` rung stops the rest."""
    by = {(c.get("set"), c["name"]): c for c in cases}
    want = [by.get((r.get("set"), r["name"])) for r in rows]
    e = RT.error_rates([{**r, "name": f"{r.get('set')}/{r['name']}"} for r in rows],
                       [{**c, "name": f"{c.get('set')}/{c['name']}"} for c in cases])
    out = {**e, "false_met_ci": _interval(e["false_met"], e["labelled_retry"]),
           "false_refusal_ci": _interval(e["false_refusal"], e["labelled_pass"]),
           "false_met_by_source": {}}
    for src in LABEL_SOURCES:
        got = [r.get("kind") for r, c in zip(rows, want)
               if c and c["expect"]["kind"] == "retry" and c.get("label_source") == src
               and r.get("kind") in ("pass", "retry")]
        if got:
            k = sum(g == "pass" for g in got)
            out["false_met_by_source"][src] = {"k": k, "n": len(got), "ci": _interval(k, len(got))}
    green = [r.get("kind") for r, c in zip(rows, want)
             if c and c["expect"]["kind"] == "retry" and c.get("tests_rung") == "green"
             and r.get("kind") in ("pass", "retry")]
    if green:
        k = sum(g == "pass" for g in green)
        out["false_met_review_only"] = {"k": k, "n": len(green), "ci": _interval(k, len(green))}
    return out


def _part(k: int, n: int, ci, what: str) -> str:
    if not n:
        return f"{what} unmeasured (no such case)"
    return f"{what} {k}/{n} (95% Wilson {ci[0]:.0%}–{ci[1]:.0%})"


def rates_line(rows: list[dict], cases: list[dict]) -> str:
    r = rates(rows, cases)
    line = (_part(r["false_met"], r["labelled_retry"], r["false_met_ci"], "false met") + "; "
            + _part(r["false_refusal"], r["labelled_pass"], r["false_refusal_ci"], "false refusal"))
    if len(r["false_met_by_source"]) > 1 or "refused_then_fixed" in r["false_met_by_source"]:
        line += " [false met by label source: " + ", ".join(
            f"{s} {v['k']}/{v['n']}" for s, v in r["false_met_by_source"].items()) + "]"
    only = r.get("false_met_review_only")
    if only:
        line += f" [tests rung green, review alone: {only['k']}/{only['n']}]"
    if r["ungraded"]:
        line += f"; {r['ungraded']} ungraded"
    return line


# ── the trend ────────────────────────────────────────────────────────────

def record_path() -> Path:
    from scripts.automod import state as S
    return S.STATE_DIR / RECORD_NAME


def record_run(rows: list[dict], cases: list[dict], sets: list[str], *, repo: Path | None = None,
               path: Path | None = None, now: float | None = None, partial: bool = False) -> dict:
    """Append one row for this run. `partial` (an `--only` run) is written and
    marked, so the scorecard reader can leave it out of the trend."""
    now = time.time() if now is None else now
    try:
        head = _git(repo or RT.LIVE_ROOT, "rev-parse", "HEAD")
    except (RuntimeError, OSError):
        head = ""
    r = rates(rows, cases)
    row = {"event": "review_calibration", "ts": now,
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
           "sets": list(sets), "partial": bool(partial), "commit": head, "cases": len(rows),
           "agree": sum(bool(x.get("ok")) for x in rows), **r,
           "seconds": round(sum(float(x.get("seconds") or 0) for x in rows), 1),
           "verdicts": {f"{x.get('set')}/{x['name']}": x.get("kind") for x in rows}}
    path = path or record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def read_records(path: Path | None = None, *, include_partial: bool = False) -> list[dict]:
    path = path or record_path()
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("event") == "review_calibration" \
                and (include_partial or not row.get("partial")):
            out.append(row)
    return out


def scorecard_reading(path: Path | None = None, *, now: float | None = None) -> dict:
    """What a scorecard row needs: the newest full run and the one before it.

    `{"measured": False}` when no full run is recorded — unmeasured, which a
    renderer must not print as 0. `age_days` is there so a stale reading is
    visible as stale; the scorecard holds no opinion on how old is too old."""
    runs = read_records(path)
    if not runs:
        return {"measured": False, "runs": 0}
    last = runs[-1]
    now = time.time() if now is None else now
    keys = ("false_met", "labelled_retry", "false_met_ci", "false_refusal", "labelled_pass",
            "false_refusal_ci", "ungraded", "false_met_by_source", "false_met_review_only",
            "sets", "commit", "created_at")
    out = {"measured": True, "runs": len(runs), "age_days": round((now - last["ts"]) / 86400, 1),
           **{k: last.get(k) for k in keys}}
    if len(runs) > 1:
        prev = runs[-2]
        out["previous"] = {k: prev.get(k) for k in ("false_met", "labelled_retry", "false_refusal",
                                                    "labelled_pass", "created_at")}
    return out


def render_scorecard_reading(reading: dict) -> str:
    """One markdown table cell pair, in the scorecard's `| value | detail |` shape."""
    if not reading.get("measured"):
        return "— | no recorded calibration run (unmeasured, not zero)"
    def frac(k, n, ci):
        return (f"{k}/{n} ({ci[0]:.0%}–{ci[1]:.0%})" if n and ci else "unmeasured")
    return (f"false met {frac(reading['false_met'], reading['labelled_retry'], reading['false_met_ci'])}"
            f" | false refusal "
            f"{frac(reading['false_refusal'], reading['labelled_pass'], reading['false_refusal_ci'])}"
            f"; {reading['age_days']:g} d old, run {reading['runs']}")


# ── the CLI half `review_tools.main` hands over ──────────────────────────

def cli(args) -> int:
    """`calibrate --set …`, `lock-labels` and `defect`, for `review_tools.main`."""
    from scripts.automod import review_cause as RC
    if args.cmd == "lock-labels":
        print(write_lock())
        return 0
    if args.cmd == "defect":
        from scripts.automod import review as RV
        from scripts.automod import state as S
        from scripts.automod import testpaths as TP
        rows = {r["round_id"]: r for r in RT.landed_rounds(S.LEDGER_PATH)}
        if args.round not in rows:
            raise SystemExit(f"{args.round} is not a settled landing with an item")
        r = rows[args.round]
        paths = args.path or (TP.pick_test_files(r["changed_paths"], RT.LIVE_ROOT)
                              if args.kind == "strip_tests" else [])
        clauses = dict(enumerate(RV.item_contract(r["item_id"])["clauses"], 1))
        case = constructed_case(repo=RT.LIVE_ROOT, name=args.name, item_id=r["item_id"],
                                round_id=args.round, parent=r["parent"], source_commit=r["commit"],
                                kind=args.kind, paths=paths, violates=args.violates,
                                clause_text=clauses, why=args.why)
        print(write_case(case, RT.FIXTURE_DIR / NEGATIVE_SET))
        print("now lock it: python -m scripts.automod.review_tools lock-labels")
        return 0
    sets = list(dict.fromkeys(args.case_set))
    rows, cases = calibrate_sets(sets, only=args.only, seams_policy=args.seams_policy)
    for r in rows:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"[{mark}] {r['set']}/{r['name']}{' (provisional)' if r['provisional'] else ''}: "
              f"{r['why']} — {r['kind']} · {str(r['findings'])[:160]} · {r['seconds']}s")
    print("\n" + RC.render_agreement(RC.agreement_by_cause(rows, rows)))
    print(rates_line(rows, cases))
    if args.record:
        row = record_run(rows, cases, sets, partial=bool(args.only))
        print(f"recorded → {record_path()} ({row['created_at']})")
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} agree")
    return 0 if rows and all(r["ok"] for r in rows) else 1
