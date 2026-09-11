"""The review rung: does the diff do what the item asked?

Every other rung asks whether the change *broke* something. None asked whether
it did what the acceptance check said, and the only verdict on that was the
round's own self-report, one enum wide. #544 landed through eight green rungs
(3271 tests, canary, smoke, a flat regression eval) with one of its five
acceptance clauses skipped outright, half the worker fleet uncovered by the
ledger it added, an `Edit` replay that silently reverted files, and a test
that asserted `x == [] or True` — and then declared its acceptance
`deferred` to an empty list. A human found all four in forty minutes with
nothing but the item and the diff. That is the grader's whole input.

What grades, and where it runs
------------------------------
The primary model, in a **fresh session on the live backend**, handed only the
item, its acceptance clauses, the diff and the changed test files. Not the
triage evidence and not the round's report: the author's prose primes
agreement, and the point of a second reader is that it has not read the first
one's conclusions.

The live backend and not the canary, deliberately. The canary is candidate
code end to end (`canary_config.py` puts the worktree on `PYTHONPATH`): its
harness, finalizer, prompt builder and tool policy are the round's own, so a
round touching `app/harness/finalizer.py` would grade itself with itself —
the exact thing the gate's first rule forbids (`gate.py`: "a candidate that
weakens the gate is judged by the old gate"). The canary's sessions dir is
also deleted with the worktree, and the grader's transcript is the raw
material the scorecard reads later.

Stdlib only, like `canary_smoke`: the gate runs inside lloyd-mcp on a thread
and must not import the application it is judging. The session JSON is
hand-written with `platform: "worker"` — the one word that makes the router
honour `final_schema` (`NON_USER_PLATFORMS`) and arm the authority gate.

Three things keep the verdict worth reading
-------------------------------------------
* **Evidence is required per clause and checked in Python.** A clause may be
  `met` only with a path that exists in the worktree, a test node id in a
  changed test file, and `how_verified` of `ran` or `read`. `parse_review`
  downgrades anything else to `partial` without consulting the model. Fails
  closed on the grader's own laziness.
* **Deterministic honesty checks run before the model is asked.** `or True`,
  `assert True`, a new skip/xfail, fewer `def test_` than before while the item
  has clauses. Any hit is a finding whatever the grader says.
* **A failed review is a retry, not a verdict on the item**, unless the
  grader says the premise itself is unsound. The rung result carries the
  findings; the round fixes and re-gates once in the same turn; after two
  refusals the rung tells it to abort and the backlog re-offers the item with
  the findings and the kept branch (`backlog.implement_outcomes`,
  `review_retry`). Premise unsound falls to the existing `spent` path and a
  human decides.
"""

from __future__ import annotations

import json
import re
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

# How many times one round may be sent back before the rung tells it to abort
# without asking the model again. Two: the first refusal is the normal
# fix-and-regate move; a second means author and grader disagree.
REVIEW_MAX_PER_ROUND = 2
# The grader's wall clock, and the gate's: past this the rung is `external`
# (the engine, not the diff) and the item keeps its attempt.
REVIEW_TIMEOUT_S = 600.0
REVIEW_MAX_TURNS = 40
DIFF_CAP_CHARS = 60_000
BODY_CAP_CHARS = 12_000

PREMISES = ("sound", "unsound")
CLAUSE_VERDICTS = ("met", "partial", "unmet")
HOW_VERIFIED = ("ran", "read", "inferred")

# Built from the tuples above, not restated (the triage schema's rule: one
# list, or a value lands in the grammar and not the validator). No maxLength:
# the decoder would stop mid-sentence at it.
REVIEW_SCHEMA: dict = {
    "type": "object",
    "title": "automod_review",
    "properties": {
        "premise": {"type": "string", "enum": list(PREMISES),
                    "description": ("sound: the item describes a real problem and this "
                                    "change is the right kind of fix for it. unsound: the "
                                    "item's premise is false, already true, or the change "
                                    "cannot satisfy it by construction — no retry will help.")},
        "clauses": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "clause": {"type": "integer", "description": "1-based index into the clauses given."},
                "verdict": {"type": "string", "enum": list(CLAUSE_VERDICTS)},
                "evidence_path": {"type": "string",
                                  "description": "Worktree-relative file the evidence is in. Empty if none."},
                "evidence_line": {"type": "integer", "description": "Line in evidence_path, or 0."},
                "test_node_id": {"type": "string",
                                 "description": ("The pytest node id that exercises THIS clause's "
                                                 "breaking input, in a test file this diff changed. "
                                                 "Empty if no such test exists.")},
                "how_verified": {"type": "string", "enum": list(HOW_VERIFIED),
                                 "description": "ran: you executed it; read: you read the code and test; inferred: neither."},
                "note": {"type": "string", "description": "One or two sentences: what is missing, or what you saw."},
            },
            "required": ["clause", "verdict", "evidence_path", "evidence_line",
                         "test_node_id", "how_verified", "note"],
            "additionalProperties": False,
        }},
        "test_honesty": {"type": "array", "items": {
            "type": "object",
            "properties": {"file": {"type": "string"}, "line": {"type": "integer"},
                           "problem": {"type": "string"}},
            "required": ["file", "line", "problem"],
            "additionalProperties": False,
        }, "description": ("Tests in the diff that cannot fail, do not exercise what they "
                           "name, or were weakened. Empty when none.")},
        "seams_unverified": {"type": "array", "items": {"type": "string"},
                             "description": ("Process boundaries the change crosses (a loopback "
                                             "POST, `_meta` over MCP, a Task subagent, a restart) "
                                             "for which no test crosses the seam. Empty when none.")},
        "summary": {"type": "string", "description": "Two sentences for the round's author."},
    },
    "required": ["premise", "clauses", "test_honesty", "seams_unverified", "summary"],
    "additionalProperties": False,
}

# What the grader may not do. A superset of the worker automod ban
# (`tests/test_automod_hardening.py` asserts it): the grader reads and runs
# tests, and that is all. Defined here rather than imported from
# `workers.sources._common`, because gate.py may not import the application.
REVIEW_DENY: tuple[str, ...] = (
    "automod_start", "automod_gate", "automod_land", "automod_abort",
    "automod_status", "automod_rollback", "automod_vault_land", "automod_vault_revert",
    "grant_create",
    "Edit", "Write", "NotebookEdit", "Task",
    "backlog_write_task", "vault_write",
    "email_send", "email_reply", "email_forward", "discord_send",
)

# Patterns that make a test unable to fail, or weaker than it was. Applied
# to the changed test files as a delta against the base version, so a tree
# that already carried one is not blamed on this round.
_HONESTY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bor\s+True\b", "`or True` makes the assertion unable to fail"),
    (r"^\s*assert\s+True\b", "`assert True` asserts nothing"),
    (r"pytest\.skip\(", "a new pytest.skip"),
    (r"pytest\.mark\.skip", "a new skip marker"),
    (r"pytest\.mark\.xfail", "a new xfail marker"),
)


# ── the contract ──────────────────────────────────────────────────────────

def item_contract(item_id: int, ledger: Path | None = None) -> dict:
    """`{id, title, body, clauses}` for the item a round is implementing.

    Clauses come from the item's front matter (`acceptance_clauses`, written by
    triage since the review rung landed), else from the confirmed triage
    event, else the prose acceptance as one clause — items confirmed before
    clauses existed have prose only, and a grader that refused them would
    block the entire current pool.
    """
    from scripts.automod import backlog as B
    from scripts.automod import state as S
    ledger = ledger or S.LEDGER_PATH
    paths = sorted(B.BACKLOG_DIR.glob(f"{int(item_id)}-*.md"))
    title, body, fm = f"#{item_id}", "", {}
    if paths:
        text = paths[0].read_text(encoding="utf-8")
        fm, body = B._split_frontmatter(text)
        m = re.search(r"^#\s+(.+)$", body, re.M)
        title = m.group(1).strip() if m else paths[0].stem
    ev = B.confirmed_verdicts(ledger).get(int(item_id)) or {}
    clauses = B.acceptance_clauses_of(ev, fm)
    return {"id": int(item_id), "title": title, "body": body[:BODY_CAP_CHARS],
            "clauses": clauses, "path": str(paths[0]) if paths else ""}


# ── deterministic half ───────────────────────────────────────────────────

def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True, timeout=120, check=False)
    return r.stdout if r.returncode == 0 else ""


def diff_text(worktree: Path, base: str) -> tuple[str, bool]:
    """`git diff base...HEAD`, capped. `(text, truncated)`."""
    out = _git(worktree, "diff", f"{base}...HEAD")
    if len(out) > DIFF_CAP_CHARS:
        files = _git(worktree, "diff", "--stat", f"{base}...HEAD")
        return (out[:DIFF_CAP_CHARS] + "\n\n[diff truncated; full file list:]\n" + files), True
    return out, False


def honesty_prechecks(worktree: Path, base: str, changed_paths: list[str],
                      *, n_clauses: int = 0) -> list[dict]:
    """Findings no model is needed for, on the round's changed test files.

    Each pattern is counted in the post-image and in the base version and only
    an INCREASE is reported: a tolerated `xfail` that predates the round is not
    this round's. The `def test_` delta is checked when the item has clauses —
    a change to code under a contract that adds no test cannot have pinned it.
    """
    out: list[dict] = []
    tests = [p for p in changed_paths if p.startswith("tests/") and p.endswith(".py")]
    added_tests = 0
    for rel in tests:
        post_path = worktree / rel
        post = post_path.read_text(encoding="utf-8", errors="replace") if post_path.exists() else ""
        pre = _git(worktree, "show", f"{base}:{rel}")
        for pat, why in _HONESTY_PATTERNS:
            rx = re.compile(pat, re.M)
            n_post, n_pre = len(rx.findall(post)), len(rx.findall(pre))
            if n_post > n_pre:
                # Name the first new occurrence's line.
                line = 0
                for i, ln in enumerate(post.splitlines(), 1):
                    if rx.search(ln):
                        line = i
                        break
                out.append({"file": rel, "line": line, "problem": why})
        added_tests += max(0, len(re.findall(r"^\s*(?:async\s+)?def test_", post, re.M))
                           - len(re.findall(r"^\s*(?:async\s+)?def test_", pre, re.M)))
    code_changed = any(p.endswith(".py") and not p.startswith("tests/") for p in changed_paths)
    if n_clauses and code_changed and tests and added_tests == 0:
        out.append({"file": tests[0], "line": 0,
                    "problem": ("test files changed but no test function was added while "
                                "the item has acceptance clauses to pin")})
    return out


# ── the grader ───────────────────────────────────────────────────────────

def build_prompt(*, contract: dict, diff: str, diff_truncated: bool,
                 changed_tests: list[str], test_counts: dict,
                 worktree: Path, run_tests: Path) -> str:
    clauses = "\n".join(f"{i}. {c}" for i, c in enumerate(contract["clauses"], 1))
    counts = ", ".join(f"{k}={v}" for k, v in test_counts.items()
                       if k in ("passed", "failed", "skipped", "collected")) or "unknown"
    return f"""\
You are reviewing a change another session made to this codebase, against the \
backlog item it claims to implement. You have NOT seen that session's report, \
and you must not look for it: your value is that you read the diff cold. Work \
in the worktree at `{worktree}` — pass that absolute path to Read/Grep/Glob, \
because the default root is the live tree, not this change.

<item id="{contract['id']}">
# {contract['title']}

{contract['body']}
</item>

<acceptance_clauses>
{clauses}
</acceptance_clauses>

<diff base_truncated="{str(diff_truncated).lower()}">
{diff}
</diff>

Test files this diff changed or added: {', '.join(changed_tests) or 'none'}.
The full suite already ran on this worktree: {counts}. To run a test yourself, \
the ONLY way is:

    {run_tests} <pytest node id or file>

(it bakes in the right interpreter, cwd and scratch state; a bare `pytest` \
here would write into production state). Use it for every clause you mark \
`ran`, and paste the tail of its output in that clause's note.

Procedure, per clause, in order:
1. Write down the input or situation that would BREAK the clause if the change \
were wrong. Be concrete: a function argument, a process boundary, a file state.
2. Find the test in the changed test files that exercises that input. Quote \
its node id. If no changed test exercises it, the clause is at most `partial`.
3. Read the code path the clause names. Note the file and line that satisfies \
it, or the gap.
4. Decide: `met` only if the code does it AND a changed test pins the breaking \
input AND you ran or read both. `partial` if the code does it but nothing pins \
it, or you could not verify. `unmet` if the code does not do it. The code may \
predate this diff — a round whose diff only adds the test that pins behaviour \
an earlier landing shipped has still met the clause, if the tree satisfies it \
and the changed test pins it.

Keep every `note` to two sentences and never paste command output into it; \
paths are worktree-relative (`app/x.py`, not `~/…`). Your review is restated \
as one JSON object at the end under a fixed token budget, and a long note in \
clause 1 is how clause 4 gets cut off.

Then the two sweeps the clauses do not cover:
- **Test honesty.** For each changed test file, look for assertions that \
cannot fail (`or True`, `assert True`, asserting on fixture state), tests that \
never call the code they name, skips, xfails, weakened assertions. Report each \
with file and line.
- **Seams.** List every process boundary this change crosses — a loopback \
POST to `/api/message/stream`, `_meta` carried over MCP, a `Task` subagent, a \
contextvar read in another task, a supervisord restart — and for each, name \
the test that crosses it. Any seam with no such test goes in \
`seams_unverified`. The code graph is blind across these; a grep is not a test.

Finally judge the PREMISE: is the item describing a real problem, and is this \
the kind of change that can fix it? `unsound` is for a false premise or a fix \
that cannot work by construction — not for an incomplete one. An incomplete \
fix with a sound premise is `sound` with `unmet`/`partial` clauses; the author \
gets your findings and another go.

You cannot edit anything and must not try. Do not file backlog items. When you \
have finished, you will be asked to restate your review as one JSON object; \
every `met` needs its evidence_path, test_node_id and how_verified, or it \
will be downgraded to `partial` without asking you.
"""


def _post_stream(url: str, payload: dict, timeout: float):
    """Yield parsed SSE events from POST `url` (canary_smoke's shape)."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        event_type = None
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                blob = line[5:].strip()
                if not blob:
                    continue
                try:
                    data = json.loads(blob)
                except ValueError:
                    data = {"raw": blob}
                yield (event_type or data.get("type") or "message"), data
            elif not line:
                event_type = None


def backend_url(root: Path | None = None) -> str:
    """`services.backend` from config.yaml, read raw (no app import)."""
    try:
        import yaml
        raw = yaml.safe_load(((root or LIVE_ROOT) / "config.yaml").read_text(encoding="utf-8")) or {}
        return str((raw.get("services") or {}).get("backend") or "http://127.0.0.1:8080").rstrip("/")
    except Exception:
        return "http://127.0.0.1:8080"


def write_run_tests(scratch: Path, *, worktree: Path, python: Path, env: dict) -> Path:
    """The one way the grader may run pytest: cwd, interpreter and scratch
    state baked in, so a grader-run suite cannot write live automod state."""
    scratch.mkdir(parents=True, exist_ok=True)
    script = scratch / "run_tests.sh"
    exports = "\n".join(f"export {k}={json.dumps(str(v))}" for k, v in sorted(env.items()))
    script.write_text(
        "#!/bin/sh\n# Written by the automod review rung. Runs pytest against the round's\n"
        "# worktree with the gate's own interpreter and scratch state.\n"
        f"{exports}\ncd {json.dumps(str(worktree))} || exit 2\n"
        f"exec {json.dumps(str(python))} -m pytest -q -p no:cacheprovider -m 'not live_vault' \"$@\"\n",
        encoding="utf-8")
    script.chmod(0o755)
    return script


def write_session(sessions_dir: Path, *, item_id: int, round_id: str, model: str) -> str:
    session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_review_{secrets.token_hex(2)}"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    (sessions_dir / f"{session_id}.json").write_text(json.dumps({
        "session_id": session_id, "id": session_id,
        "title": f"review #{item_id} ({round_id})"[:80],
        "model": model,
        # `worker`: the one platform word that makes the router honour
        # `final_schema` and arm the authority gate, and keeps the transcript
        # out of the user's chat history. `canary_smoke` uses `canary`, which
        # is neither.
        "platform": "worker", "source": "automod-review",
        "inner_voice": False,
        "messages": [], "created_at": now, "last_active": now,
        "preview": "", "message_count": 0,
    }), encoding="utf-8")
    return session_id


def grade(*, round_id: str, worktree: Path, base: str, contract: dict,
          changed_paths: list[str], test_counts: dict, python: Path, child_env: dict,
          scratch_dir: Path, backend: str | None = None, sessions_dir: Path | None = None,
          timeout: float = REVIEW_TIMEOUT_S, model: str = "primary",
          max_turns: int = REVIEW_MAX_TURNS) -> dict:
    """One grading turn on the live backend, for a code round. Never raises.

    Returns `{ok, error, session_id, structured, structured_error, text,
    stop_reason, duration_s}`. `ok` is whether a structured object came back;
    what it says is `parse_review`'s business.
    """
    worktree = Path(worktree)
    changed_tests = [p for p in changed_paths if p.startswith("tests/") and p.endswith(".py")]
    diff, truncated = diff_text(worktree, base)
    run_tests = write_run_tests(scratch_dir, worktree=worktree, python=python, env=child_env)
    prompt = build_prompt(contract=contract, diff=diff, diff_truncated=truncated,
                          changed_tests=changed_tests, test_counts=test_counts,
                          worktree=worktree, run_tests=run_tests)
    return run_grader(prompt=prompt, item_id=contract["id"], round_id=round_id,
                      backend=backend, sessions_dir=sessions_dir, timeout=timeout,
                      model=model, max_turns=max_turns)


def build_vault_prompt(*, contract: dict, paths: list[str], diff: str, vault: Path) -> str:
    clauses = "\n".join(f"{i}. {c}" for i, c in enumerate(contract["clauses"], 1))
    return f"""\
You are reviewing an edit another session made to the Obsidian vault at `{vault}` \
— prompt material, skills, scheduled tasks — against the backlog item it claims \
to implement. You have NOT seen that session's report. Read the files at their \
absolute paths under `{vault}`; there is no code and no test suite here.

<item id="{contract['id']}">
# {contract['title']}

{contract['body']}
</item>

<acceptance_clauses>
{clauses}
</acceptance_clauses>

Paths changed: {', '.join(paths)}.

<diff>
{diff[:DIFF_CAP_CHARS]}
</diff>

Per clause, in order: say what in the changed text satisfies it (file and line) \
or what is missing. `met` needs an evidence_path under the vault and \
how_verified `read`; there are no tests here, so leave test_node_id empty. \
Then judge the premise: `unsound` only for a false premise or an edit that \
cannot satisfy the item by construction. You cannot edit anything. You will be \
asked to restate the review as one JSON object.
"""


def grade_vault(*, item_id: int, paths: list[str], diff: str,
                vault: Path | None = None, backend: str | None = None,
                sessions_dir: Path | None = None, timeout: float = REVIEW_TIMEOUT_S,
                model: str = "primary") -> tuple[str, str]:
    """`(kind, findings)` for a vault round's staged edit — the
    `vault_round.GRADER` contract. `skipped` when the grader cannot run."""
    from scripts.automod import backlog as B, state as S, vault_round as VR
    vault = Path(vault or VR.VAULT)
    # Only a `vault` item's clauses can be satisfied by vault paths. #551 was a
    # `code` item whose round landed a skill and a task file first; grading the
    # whole contract against that half refused it for the code it had not
    # written yet — and would have every time. A `code` or `mixed` item's
    # clauses are the code gate's to judge; the vault half is still validated
    # through the real loaders.
    surface = str((B.confirmed_verdicts(S.LEDGER_PATH).get(int(item_id)) or {}).get("surface") or "")
    if surface and surface != "vault":
        return "skipped", f"surface is {surface}: the clauses are graded at the code gate"
    contract = item_contract(int(item_id))
    if not contract["clauses"]:
        return "skipped", f"item #{item_id} has no acceptance clauses"
    prompt = build_vault_prompt(contract=contract, paths=paths, diff=diff, vault=vault)
    res = run_grader(prompt=prompt, item_id=int(item_id), round_id="vault",
                     backend=backend, sessions_dir=sessions_dir, timeout=timeout, model=model)
    if not res["ok"]:
        return "skipped", f"grader did not answer: {res.get('error')}"
    parsed = parse_review(res["structured"], worktree=vault, changed_tests=[],
                          n_clauses=len(contract["clauses"]), require_tests=False)
    if parsed is None:
        return "skipped", "grader returned an unusable object"
    kind, findings = decide(parsed, [])
    return kind, findings


def run_grader(*, prompt: str, item_id: int, round_id: str, backend: str | None = None,
               sessions_dir: Path | None = None, timeout: float = REVIEW_TIMEOUT_S,
               model: str = "primary", max_turns: int = REVIEW_MAX_TURNS) -> dict:
    """POST one grading turn to the live backend and collect its `done`."""
    backend = (backend or backend_url()).rstrip("/")
    sessions_dir = Path(sessions_dir or (LIVE_ROOT / "sessions"))
    session_id = write_session(sessions_dir, item_id=item_id, round_id=round_id, model=model)
    report: dict = {"ok": False, "error": "", "session_id": session_id, "structured": None,
                    "structured_error": "", "text": "", "stop_reason": None, "duration_s": 0.0}
    started = time.time()
    payload = {
        "session_id": session_id, "text": prompt, "model": model,
        "priority": 1, "max_turns": int(max_turns),
        "grant_scope": "worker:automod-review",
        "extra_disallowed": list(REVIEW_DENY),
        "deadline_seconds": float(timeout),
        "final_schema": REVIEW_SCHEMA,
        "final_schema_prompt": (
            "Restate your review as one JSON object matching the schema. One entry "
            "per acceptance clause, in order. This is a transcription of what you "
            "found, not a new judgment; a `met` without evidence_path, test_node_id "
            "and how_verified=ran|read will be downgraded."),
    }
    try:
        for name, data in _post_stream(f"{backend}/api/message/stream", payload, timeout):
            if name == "error":
                report["error"] = (report["error"] + " " + str(data)[:300]).strip()
            elif name == "done":
                report["text"] = str(data.get("response") or "")
                report["stop_reason"] = data.get("stop_reason")
                report["structured"] = data.get("structured")
                report["structured_error"] = str(data.get("structured_error") or "")
                break
            if time.time() - started > timeout:
                report["error"] = f"review exceeded {timeout}s"
                break
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        report["error"] = f"HTTP {e.code}: {body}"
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    report["duration_s"] = round(time.time() - started, 1)
    if report["structured"] is None and not report["error"]:
        report["error"] = report["structured_error"] or "turn ended without a structured review"
    report["ok"] = isinstance(report["structured"], dict)
    return report


# ── judging the judge ────────────────────────────────────────────────────

def parse_review(obj, *, worktree: Path, changed_tests: list[str],
                 n_clauses: int, require_tests: bool = True) -> dict | None:
    """The grader's object, validated, with `met` downgraded where the
    evidence does not hold up. None if unusable. `require_tests=False` is
    the vault shape: prose has no pytest node to point at."""
    if not isinstance(obj, dict):
        return None
    premise = str(obj.get("premise") or "").strip().lower()
    if premise not in PREMISES:
        return None
    worktree = Path(worktree)
    changed = set(changed_tests)
    clauses: list[dict] = []
    downgraded: list[int] = []
    seen: set[int] = set()
    for raw in (obj.get("clauses") or []):
        if not isinstance(raw, dict):
            continue
        try:
            idx = int(raw.get("clause") or 0)
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > max(n_clauses, 1) or idx in seen:
            continue
        seen.add(idx)
        verdict = str(raw.get("verdict") or "").strip().lower()
        if verdict not in CLAUSE_VERDICTS:
            verdict = "partial"
        path = str(raw.get("evidence_path") or "").strip().lstrip("./")
        node = str(raw.get("test_node_id") or "").strip()
        how = str(raw.get("how_verified") or "").strip().lower()
        why: list[str] = []
        if verdict == "met":
            if not path or not (worktree / path).exists():
                why.append("evidence_path missing or not in the worktree")
            node_file = node.split("::", 1)[0]
            if require_tests and (not node or node_file not in changed):
                why.append("test_node_id not in a test file this diff changed")
            if how not in ("ran", "read"):
                why.append("how_verified is not ran|read")
            if why:
                verdict = "partial"
                downgraded.append(idx)
        clauses.append({"clause": idx, "verdict": verdict, "evidence_path": path,
                        "evidence_line": int(raw.get("evidence_line") or 0)
                        if str(raw.get("evidence_line") or "0").lstrip("-").isdigit() else 0,
                        "test_node_id": node, "how_verified": how if how in HOW_VERIFIED else "inferred",
                        "note": " ".join(str(raw.get("note") or "").split())[:600],
                        **({"downgraded": why} if why else {})})
    # A clause the grader did not mention is not met — it was not graded.
    for idx in range(1, n_clauses + 1):
        if idx not in seen:
            clauses.append({"clause": idx, "verdict": "partial", "evidence_path": "",
                            "evidence_line": 0, "test_node_id": "", "how_verified": "inferred",
                            "note": "not addressed by the grader", "downgraded": ["not graded"]})
            downgraded.append(idx)
    clauses.sort(key=lambda c: c["clause"])
    honesty = []
    for raw in (obj.get("test_honesty") or []):
        if isinstance(raw, dict) and str(raw.get("problem") or "").strip():
            honesty.append({"file": str(raw.get("file") or "")[:200],
                            "line": int(raw.get("line") or 0) if str(raw.get("line") or "0").isdigit() else 0,
                            "problem": " ".join(str(raw["problem"]).split())[:300]})
    seams = [" ".join(str(s).split())[:300] for s in (obj.get("seams_unverified") or []) if str(s).strip()]
    return {"premise": premise, "clauses": clauses, "test_honesty": honesty,
            "seams_unverified": seams[:10],
            "summary": " ".join(str(obj.get("summary") or "").split())[:600],
            "downgraded": sorted(set(downgraded))}


def decide(parsed: dict, prechecks: list[dict]) -> tuple[str, str]:
    """`(kind, findings)`: kind is `pass`, `retry` or `unsound`.

    Unsound is the grader's call alone. Everything else that is not clean is
    a retry: an unmet or partial clause, any honesty finding from either
    half, an unverified seam. Seams are advisory-to-blocking on purpose —
    #544's three worst defects were all cross-process seams with no test.
    """
    if parsed["premise"] == "unsound":
        return "unsound", parsed["summary"] or "the grader judged the premise unsound"
    lines: list[str] = []
    for c in parsed["clauses"]:
        if c["verdict"] != "met":
            tag = f" (downgraded: {'; '.join(c['downgraded'])})" if c.get("downgraded") else ""
            lines.append(f"clause {c['clause']} {c['verdict']}{tag}: {c['note'] or '(no note)'}")
    for h in prechecks + parsed["test_honesty"]:
        lines.append(f"test honesty {h['file']}:{h['line']}: {h['problem']}")
    for s in parsed["seams_unverified"]:
        lines.append(f"seam unverified: {s}")
    if not lines:
        return "pass", parsed["summary"]
    return "retry", "; ".join(lines)


def summarize_clauses(parsed: dict) -> str:
    counts = {k: 0 for k in CLAUSE_VERDICTS}
    for c in parsed["clauses"]:
        counts[c["verdict"]] += 1
    return ", ".join(f"{v} {k}" for k, v in counts.items() if v)
