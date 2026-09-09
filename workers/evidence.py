"""Verifier-bound evidence claims for worker runs — backlog #525.

A worker run has always ended with prose: `runs.summary` is the model's own
final text, front-sliced to 300-500 chars. The corrections log is the ledger of
what that costs, and its single most-repeated defect is a summary that
disagrees with the disk:

* 2026-08-24 — the handoff reported the entity graph "restored to 12,131
  relationships" while the same night's health report (`| Total relationships
  | 0 |`) and the missing `_relationships.json` said zero. The recovery-claim
  rule had to be re-held for seven consecutive cycles.
* 2026-09-03 — the handoff counted 96 hypothesis-failure dumps and a 21-file
  single-day peak; disk held 121 and 64.
* 2026-09-04 — the handoff called `signals-latest.md` "307KB / 383,722 chars";
  on disk it was 13,503 bytes.

Every one of those was cheaply checkable and unchecked, because the fix so far
has been procedural — a skill that says *verify on disk before claiming*, which
holds only as long as the model chooses to. This module is the structural
version: a run ends with `{claim, check}` pairs, a stdlib verifier re-runs each
check against the filesystem at the moment the ledger row is written, and
whatever failed to verify is handed to the next run of that same task.

Two rules are load-bearing and are the reason this is its own module rather
than a few lines in `pool.py`:

1. **The verifier is stdlib-only and never LLM-judged.** A check graded by a
   model is the narration we just removed, one layer up. Same discipline the
   self-mod guardian keeps.
2. **A claim that cannot be evaluated is `insufficient`, not `verified`.** A
   gate that reads its own missing input as a pass is the failure mode recorded
   three times over in MEMORY.md (the self-rebaselining `graph-baseline.json`,
   `_is_dependency_met`'s `if not dep_task: return True`, the dream lock whose
   absent file read as 56 years). So a bundle with nothing in it reports a gap
   and a rate of `None` — never a clean zero.

Schema (one claim):

    {"claim": "<one checkable sentence>",
     "check": {"kind": "file_exists" | "count_eq" | "json_key" | "regex", ...},
     "status": "verified" | "refuted" | "insufficient",   # written by the verifier
     "observed": <what the disk actually said>,
     "detail": "<why, when it is not a plain pass>"}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

# The four kinds item #525 named. Anything else is `insufficient` by name, so a
# model that invents a fifth kind shows up as an unevaluable claim instead of a
# silent pass.
CHECK_KINDS = ("file_exists", "count_eq", "json_key", "regex")
CLAIM_STATUSES = ("verified", "refuted", "insufficient")
GAP_STATUSES = ("refuted", "insufficient")

# What `count_eq` counts. `files` needs a directory plus a glob; the rest
# operate on one file.
COUNT_MEASURES = ("files", "bytes", "lines", "matches")

# Fence tag the model is asked to close its final message with. Deliberately not
# `json`: an untagged fence in a long markdown answer is ambiguous with the
# answer's own examples, and we would parse the wrong thing.
FENCE_TAG = "evidence"

# Carried-forward gap list: per-task watermark key, and the caps that keep a
# pathological run from shipping a wall of text into its successor's prompt.
GAP_KEY_PREFIX = "evidence_gaps:"
MAX_GAP_ITEMS = 12
MAX_GAP_CHARS = 400

# Cost bounds. A check reads at most this many bytes, and a bundle at most this
# many claims — a run that emits 200 claims is asserting nothing in particular.
MAX_CHECK_BYTES = 8_000_000
MAX_BUNDLE_CLAIMS = 20


def default_root() -> Path:
    """The checkout a relative `path` is resolved against.

    `Path(__file__)` and not `os.getcwd()`: the backend, the pool and the MCP
    aggregator all run from different directories, and a relative claim that
    resolves against whichever cwd happened to be set is a check that silently
    points at nothing. Inside a self-modification worktree this is the worktree,
    which is the tree under test.
    """
    return Path(__file__).resolve().parents[1]


def gaps_key(task_id: Any) -> str:
    """Watermark key holding one task's carried-forward gap list."""
    return f"{GAP_KEY_PREFIX}{task_id}"


# ── Parsing the claims out of a final response ─────────────────────────────

_BLOCK = re.compile(
    r"```[ \t]*" + FENCE_TAG + r"[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_claims_block(text: str) -> list[dict]:
    """Pull `{claim, check}` pairs out of a run's final response.

    A block that is present but unparseable becomes a claim with an unknown
    check kind — it lands in the gap set as `insufficient`. Dropping it would be
    the same silence as a run that never emitted one, which is exactly what the
    pilot is here to make visible.
    """
    found: list[dict] = []
    for m in _BLOCK.finditer(str(text or "")):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            found.append(_malformed(raw))
            continue
        items = data.get("claims") if isinstance(data, dict) else data
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list) or not items:
            found.append(_malformed(raw))
            continue
        found.extend(i for i in items if isinstance(i, dict))
    return found[:MAX_BUNDLE_CLAIMS]


def _malformed(raw: str) -> dict:
    return {"claim": f"(claims block could not be parsed) {raw[:120]}",
            "check": {"kind": "malformed"}}


# ── Path scoping ───────────────────────────────────────────────────────────

def _resolve(path_value: Any, root: Path) -> tuple[Path | None, str | None]:
    """Resolve a check path and refuse anything that leaves the tree.

    Not a security boundary — the run already has Bash — but it keeps a
    hallucinated `/root/…` or `/etc/passwd` path from being reported as a
    refutation of the model's claim about *its own* work. Off-tree is
    `insufficient`, which says what is true: we could not check that.
    """
    if not isinstance(path_value, str) or not path_value.strip():
        return None, "check has no path"
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        target = candidate.resolve()
        base = Path(root).resolve()
    except OSError as e:                                    # unresolved symlink
        return None, f"path could not be resolved: {e}"
    try:
        target.relative_to(base)
    except ValueError:
        return None, f"path escapes the evidence root {base}"
    return target, None


def _read_text(target: Path) -> str:
    raw = target.read_bytes()[:MAX_CHECK_BYTES]
    return raw.decode("utf-8", errors="replace")


# ── The four checks ────────────────────────────────────────────────────────

def _check_file_exists(check: dict, root: Path) -> tuple[str, Any, str]:
    target, err = _resolve(check.get("path"), root)
    if err:
        return "insufficient", None, err
    want = check.get("expected", True)
    if isinstance(want, str):
        want = want.strip().lower() not in ("false", "0", "no", "absent")
    present = target.exists()
    if present == bool(want):
        return "verified", present, ""
    return ("refuted", present,
            f"claimed exists={bool(want)}, disk says exists={present} "
            f"({target})")


def _check_count_eq(check: dict, root: Path) -> tuple[str, Any, str]:
    target, err = _resolve(check.get("path"), root)
    if err:
        return "insufficient", None, err
    measure = str(check.get("measure") or "files")
    if measure not in COUNT_MEASURES:
        return "insufficient", None, (f"unknown count measure {measure!r} "
                                      f"(expected one of {list(COUNT_MEASURES)})")
    expected = check.get("expected")
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return "insufficient", None, "count_eq needs a numeric `expected`"

    if measure == "files":
        if not target.is_dir():
            # A missing directory does not make "25 rows were written"
            # unevaluable — it makes it wrong, with a count of zero.
            return ("refuted", 0, f"{target} is not a directory")
        pattern = str(check.get("glob") or "*")
        observed = sum(1 for p in target.glob(pattern) if p.is_file())
    else:
        if not target.is_file():
            return ("refuted", 0, f"{target} does not exist")
        if measure == "bytes":
            observed = target.stat().st_size
        else:
            text = _read_text(target)
            if measure == "lines":
                pattern = check.get("pattern")
                lines = text.splitlines()
                observed = (sum(1 for ln in lines if re.search(pattern, ln))
                            if pattern else len(lines))
            else:  # matches
                pattern = check.get("pattern")
                if not isinstance(pattern, str) or not pattern:
                    return "insufficient", None, "matches needs a `pattern`"
                observed = len(re.findall(pattern, text, re.MULTILINE))

    tolerance = check.get("tolerance") or 0
    try:
        ok = abs(float(observed) - float(expected)) <= float(tolerance)
    except (TypeError, ValueError):
        return "insufficient", None, "expected/tolerance are not numeric"
    if ok:
        return "verified", observed, ""
    return ("refuted", observed,
            f"claimed {measure}={expected}, disk says {observed} ({target})")


def _dig(obj: Any, key: str) -> tuple[bool, Any]:
    """Follow a dotted key, with numeric segments indexing lists."""
    cur = obj
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return False, None
        else:
            return False, None
    return True, cur


def _check_json_key(check: dict, root: Path) -> tuple[str, Any, str]:
    target, err = _resolve(check.get("path"), root)
    if err:
        return "insufficient", None, err
    key = check.get("key")
    if not isinstance(key, str) or not key:
        return "insufficient", None, "json_key needs a dotted `key`"
    if not target.is_file():
        return "refuted", None, f"{target} does not exist"
    try:
        data = json.loads(_read_text(target))
    except (ValueError, TypeError) as e:
        # Unparseable is not a false claim, it is an unreadable one.
        return "insufficient", None, f"{target} is not valid JSON: {e}"
    present, value = _dig(data, key)
    if not present:
        return "refuted", None, f"key {key!r} absent from {target}"
    expected = check.get("expected")
    if expected is None or value == expected:
        return "verified", value, ""
    return ("refuted", value,
            f"claimed {key}={expected!r}, file says {value!r} ({target})")


def _check_regex(check: dict, root: Path) -> tuple[str, Any, str]:
    target, err = _resolve(check.get("path"), root)
    if err:
        return "insufficient", None, err
    pattern = check.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return "insufficient", None, "regex needs a `pattern`"
    want = str(check.get("expect") or "match")
    if want not in ("match", "absent"):
        return "insufficient", None, "regex `expect` must be match or absent"
    if not target.is_file():
        if want == "absent":
            return "verified", 0, f"{target} does not exist, so nothing matches"
        return "refuted", 0, f"{target} does not exist"
    hits = len(re.findall(pattern, _read_text(target), re.MULTILINE))
    ok = hits > 0 if want == "match" else hits == 0
    if ok:
        return "verified", hits, ""
    return ("refuted", hits,
            f"claimed {want} for /{pattern}/ on {target}, disk says {hits} hit(s)")


_EVALUATORS: dict[str, Callable[[dict, Path], tuple[str, Any, str]]] = {
    "file_exists": _check_file_exists,
    "count_eq": _check_count_eq,
    "json_key": _check_json_key,
    "regex": _check_regex,
}


# ── Verification ───────────────────────────────────────────────────────────

def verify_claim(claim: Any, *, root: Path | None = None) -> dict:
    """Run one claim's check against disk and return the claim with a status.

    Every failure mode of the *check itself* — unknown kind, bad regex, path
    outside the tree, unreadable JSON — is `insufficient`. Only a check that ran
    and disagreed with the disk is `refuted`.
    """
    base = Path(root) if root is not None else default_root()
    if not isinstance(claim, dict):
        return {"claim": str(claim)[:300], "check": {}, "status": "insufficient",
                "observed": None, "detail": "claim is not an object"}
    text = str(claim.get("claim") or "").strip()
    check = claim.get("check") if isinstance(claim.get("check"), dict) else {}
    kind = str(check.get("kind") or "")
    if kind not in CHECK_KINDS:
        return {**claim, "claim": text, "check": check, "status": "insufficient",
                "observed": None,
                "detail": (f"unknown check kind {kind!r} — no verifier ran "
                           f"(expected one of {list(CHECK_KINDS)})")}
    try:
        status, observed, detail = _EVALUATORS[kind](check, base)
    except re.error as e:
        status, observed, detail = "insufficient", None, f"invalid regex: {e}"
    except OSError as e:
        status, observed, detail = "insufficient", None, f"read failed: {e}"
    return {**claim, "claim": text, "check": check, "status": status,
            "observed": observed, "detail": detail}


def verify_bundle(claims: list[dict] | None, *, root: Path | None = None,
                  emitted: bool = True) -> dict:
    """Verify a run's claims and partition them into `verified` and `gap`.

    `verified` is the set the next iteration must preserve; `gap` is the set it
    must address, and is what gets carried into that run's prompt. The rate's
    denominator is the claims that were checked, not the runs: a `[SILENT]` run
    that asserted nothing must not be able to dilute a task's refutation rate
    toward zero (acceptance, #525). With no claims at all the rate is `None` —
    unevaluable, which is not the same number as clean.
    """
    claims = [c for c in (claims or []) if isinstance(c, dict)][:MAX_BUNDLE_CLAIMS]
    verified: list[dict] = []
    checked: list[dict] = []
    for claim in claims:
        v = verify_claim(claim, root=root)
        checked.append(v)
        if v["status"] == "verified":
            verified.append(v)

    gap: list[str] = [
        (f"[{v['status']}] {v['claim']}" + (f" — {v['detail']}" if v["detail"] else ""))
        [:MAX_GAP_CHARS]
        for v in checked if v["status"] in GAP_STATUSES
    ]
    if not claims:
        gap.insert(0, "no evidence claims were emitted, so nothing this run "
                      "asserted was checked against disk"
                      if emitted else
                      "no evidence claims were emitted (the run did not complete)")

    counts = {
        "total": len(checked),
        "verified": len(verified),
        "refuted": sum(1 for v in checked if v["status"] == "refuted"),
        "insufficient": sum(1 for v in checked if v["status"] == "insufficient"),
    }
    unverified = counts["refuted"] + counts["insufficient"]
    return {
        "claims": checked,
        "verified": [v["claim"] for v in verified],
        "gap": gap[:MAX_GAP_ITEMS],
        "counts": counts,
        "refuted_or_insufficient_rate": (
            round(unverified / counts["total"], 3) if counts["total"] else None),
    }


# ── The prompt blocks (appended payload; the system prompt is untouched) ───

CLAIMS_INSTRUCTION = f"""

## Evidence claims — machine-verified when this run's record is written

Your prose summary is not evidence. End your FINAL message with exactly one
fenced block tagged `{FENCE_TAG}` holding one JSON object that lists the checkable
claims your summary makes:

```{FENCE_TAG}
{{"claims": [
  {{"claim": "today's handoff artifact was written",
    "check": {{"kind": "file_exists", "path": "_pipeline/reflection/knowledge-handoff-2026-09-09.md"}}}},
  {{"claim": "there are 25 hypothesis-failure dumps",
    "check": {{"kind": "count_eq", "path": "_pipeline/research/_debug",
               "glob": "hypothesis_fail_*.txt", "measure": "files", "expected": 25}}}},
  {{"claim": "the state file records 25 rows written",
    "check": {{"kind": "json_key", "path": "_pipeline/state/pipeline.json",
               "key": "rows_written", "expected": 25}}}},
  {{"claim": "the health report shows zero active relationships",
    "check": {{"kind": "regex", "path": "_pipeline/reflection/knowledge-health-2026-09-09.md",
               "pattern": "^\\\\| Active relationships \\\\| 0 \\\\|$", "expect": "match"}}}}
]}}
```

Rules:
- `path` is relative to the lloyd checkout. `kind` is one of `file_exists`,
  `count_eq` (`measure`: `files` with an optional `glob`, `bytes`, `lines`,
  `matches` with a `pattern`), `json_key` (dotted `key`, optional `expected`),
  `regex` (`expect`: `match` or `absent`).
- Bind ONLY numbers and files you actually read this run. A stdlib verifier
  re-runs every check against the disk when this run's ledger row is written;
  anything that does not match is recorded as `refuted` and handed to the next
  run of this task by name.
- Do not bind a claim you cannot express as one of those four checks. Prose you
  did not bind is not evidence.
"""


def gaps_prompt(gaps: list[str] | None) -> str:
    """The carried-forward gap block for the next run of a task."""
    items = [str(g).strip() for g in (gaps or []) if str(g).strip()]
    if not items:
        return ""
    lines = "\n".join(f"- {g}" for g in items[:MAX_GAP_ITEMS])
    return ("\n\n## Evidence gaps carried from your previous run\n\n"
            "These claims were refuted against the disk, or could not be checked, "
            "the last time this task ran. Address each one — re-verify it and bind "
            "it again, or say in your summary why it no longer applies — then close "
            "with the evidence block.\n" + lines + "\n")


def parse_gap_list(raw: str | None) -> list[str]:
    """Decode a stored gap list; a corrupt watermark is no gaps, not a crash."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(g) for g in data[:MAX_GAP_ITEMS]]
