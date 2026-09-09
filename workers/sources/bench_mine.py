"""bench-mine source — generates new bench tasks from failure signal.

Two inputs, kept deliberately independent. #522 found this source enabled,
registered, and never once enqueued while both inputs were wide open:

- **Ledger losers** at ``LEDGER_PATH``: bench tasks the baseline scored under
  0.6 on. That file is appended to only by an autoresearch round, so this
  input can go quiet for days — and its row selector has its own open defect
  (#625: the ledger writes ``BASELINE_<int>``, this filter matches lowercase
  ``baseline``), which is #625's to fix, not this module's to work around.
- **Failed autonomy runs**: ``AUTONOMY_RUNS_DIR/**/run_*.md`` with ``status:
  failed``. The docstring advertised this input from the day it was written
  and never opened the directory: 104 failed runs in the last 7 days against
  4,044 run files on disk. Terminal-Universe (arXiv:2609.04148) is the idea
  this copies — a recorded trajectory, *including the failed ones*,
  reconstructed into a task with a deterministic pass/fail check.

Produces candidate bench-task markdown under the configured staging root
(``workers.staging_root``, i.e. ``_pipeline/vault-derived/pending-research``
— the one ``GET /api/workers/pending`` lists and a human promotes) under
``bench/{yyyy-mm-dd}/``. Each candidate carries a ``calibration`` block: N
trials against the canonical prompt and whether its composite landed strictly
inside the capability edge, because a task the learner always passes and a
task it always fails both move the bench mean by noise rather than by signal —
four of the eleven live tasks are pinned at exactly 0.00 today.

A human promotion step moves the kept candidates into
``~/obsidian/lloyd/bench/``. Lloyd writing the tasks that grade Lloyd is the
known self-grading failure mode; the human gate, the mechanical-check
requirement, and mining from real failures rather than invented ones are what
keep it honest.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from workers.queue import WorkQueue, QueueItem
from workers.sources._common import run_prompt_on_primary, write_staging_note

logger = logging.getLogger("lloyd-workers.bench_mine")

NAME = "bench-mine"
DEFAULT_PRIORITY = 80

from app.paths import LLOYD_HOME as _LH
from app.paths import AUTONOMY_RUNS_DIR
LEDGER_PATH = _LH / "_pipeline" / "research" / "ledger.jsonl"

#: Queue items per tick, per input. Two worker slots shared with the nightly
#: chain; this source is not the reason anyone is waiting.
MAX_ENQUEUE_PER_TICK = 3
FAILURE_WINDOW_DAYS = 7

#: `failure_kind: infra` runs died on `ConnectError: All connection attempts
#: failed` or a peer closing the socket. The model was not the cause, so a
#: task mined from one grades the network.
SKIP_FAILURE_KINDS = frozenset({"infra"})

#: The capability edge. A composite at or below the floor and at or above the
#: ceiling carries no signal in either direction.
EDGE_BAND: tuple[float, float] = (0.05, 0.95)
CALIBRATION_RUNS = 10
#: A calibration that returned fewer scored trials than this is not a
#: measurement of the task, it is a measurement of the engine.
_MIN_CALIBRATION_TRIALS = 3

#: The three escalation directions Environment Evolution for Terminal Agents
#: (arXiv:2609.04128) names. Tagging which one a task used is what lets a
#: plateaued task be escalated instead of replaced.
EDGE_DIRECTIONS = ("scenario-novelty", "skill-rarity", "execution-length")

#: Run-failure and ledger-loser items share `execute`; this is how it tells them.
KIND_RUN = "mine-run"
KIND_LEDGER = "mine"


def _recent_ledger_losers(days: int = 7, limit: int = 5) -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    try:
        with LEDGER_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                created = row.get("created_at", "")
                try:
                    if created.endswith("Z"):
                        created = created[:-1] + "+00:00"
                    dt = datetime.fromisoformat(created)
                except Exception:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    continue
                if not row.get("variant_id", "").startswith("baseline"):
                    continue
                if (row.get("composite_score") or 1.0) < 0.6:
                    rows.append(row)
    except Exception as e:
        logger.warning("Ledger read error: %s", e)
    rows.sort(key=lambda r: r.get("composite_score") or 1.0)
    return rows[:limit]


# ---------------------------------------------------------------------------
# Input 2: failed autonomy runs
# ---------------------------------------------------------------------------

#: A run record is frontmatter followed by a transcript that runs to hundreds
#: of KB, and this read happens per candidate per tick.
_FM_PREFIX_BYTES = 4096


def _run_frontmatter(path: Path) -> dict:
    """Parse a run record's YAML frontmatter, or {} if it has no readable one.

    Head-only on purpose: the body is the whole transcript.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            head = f.read(_FM_PREFIX_BYTES)
    except OSError:
        return {}
    if not head.startswith("---"):
        return {}
    end = head.find("\n---\n", 3)
    if end < 0:
        return {}
    try:
        fm = yaml.safe_load(head[3:end]) or {}
    except Exception:
        return {}
    return fm if isinstance(fm, dict) else {}


def _recent_failed_runs(days: int = FAILURE_WINDOW_DAYS, limit: int = MAX_ENQUEUE_PER_TICK,
                        done: Optional[set[str]] = None) -> list[dict]:
    """Failed runs worth mining, newest first. Blocking; called via a thread.

    Three exclusions, each of which was a real flood in a sibling source:
    already-mined run ids, `infra` failures, and more than one run per task in
    a single tick — task 75 failed 20 times and task 36 18 times in the last 7
    days, and without that last rule the bench becomes one task's timeout
    repeated three times, which is the "bench collapsing onto one failure
    mode" #522 names as a risk.
    """
    done = done or set()
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    out: list[dict] = []
    seen_tasks: set[str] = set()
    try:
        candidates = sorted(AUTONOMY_RUNS_DIR.rglob("run_*.md"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError as exc:
        logger.warning("bench-mine: cannot scan %s: %s", AUTONOMY_RUNS_DIR, exc)
        return []

    for path in candidates:
        try:
            if path.stat().st_mtime < cutoff:
                break  # newest-first, so everything below this is older still
        except OSError:
            continue
        fm = _run_frontmatter(path)
        if fm.get("status") != "failed":
            continue
        run_id = str(fm.get("run_id") or path.stem)
        if run_id in done:
            continue
        if str(fm.get("failure_kind") or "") in SKIP_FAILURE_KINDS:
            continue
        task_id = str(fm.get("task_id") or "")
        if task_id and task_id in seen_tasks:
            continue
        seen_tasks.add(task_id)
        out.append({"run_id": run_id, "run_path": str(path), "task_id": task_id,
                    "failure_kind": str(fm.get("failure_kind") or ""),
                    "summary": str(fm.get("summary") or "")[:300]})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    """Offer work from either input. Neither one gates the other.

    The first cut of this function returned early when the ledger's mtime had
    not moved, so everything added after that line was unreachable whenever
    autoresearch was off — which is how a source with 4,044 unused failure
    transcripts under `AUTONOMY_RUNS_DIR` stayed at zero items for its whole
    life.
    """
    limit = int(src_cfg.get("max_enqueue_per_tick", MAX_ENQUEUE_PER_TICK))
    await _enqueue_ledger_losers(queue, src_cfg)
    await _enqueue_failed_runs(queue, src_cfg, limit)


async def _enqueue_ledger_losers(queue: WorkQueue, src_cfg: dict) -> None:
    # Cheap mtime gate before the expensive parse. The ledger is 11 MB and is
    # fully json-parsed line by line; it is appended to only by an autoresearch
    # round, which is far rarer than this source's two-hour tick, so almost
    # every wakeup was re-reading a file that had not changed. A stat is not a
    # correctness fix, it is 11 MB of parsing this source now does not do.
    try:
        mtime = LEDGER_PATH.stat().st_mtime
    except OSError:
        return
    if float(queue.wm_get(NAME, "ledger_mtime") or 0.0) >= mtime:
        return

    # _recent_ledger_losers reads and json-parses the full ledger.jsonl
    # (tens of MB) line by line — blocking I/O + CPU. Keep it off the shared
    # event loop so it can't stall HTTP/UI. See [[project_gap_fill_event_loop_freeze]].
    losers = await asyncio.to_thread(_recent_ledger_losers)
    queue.wm_set(NAME, "ledger_mtime", repr(mtime))
    if not losers:
        return
    enqueued = 0
    for row in losers:
        task_id = row.get("task_id", "")
        dedup_key = f"bench-mine:{task_id}:{row.get('round_id','')}"
        new_id = queue.enqueue(
            source=NAME,
            kind=KIND_LEDGER,
            payload={"loser_task_id": task_id, "composite_score": row.get("composite_score"),
                     "round_id": row.get("round_id")},
            priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
            dedup_key=dedup_key,
        )
        if new_id is not None:
            enqueued += 1
    if enqueued:
        logger.info("Enqueued %d bench-mine items", enqueued)


async def _enqueue_failed_runs(queue: WorkQueue, src_cfg: dict, limit: int) -> None:
    if not AUTONOMY_RUNS_DIR.exists():
        return
    # One watermark per run already mined. `mark_completed` releases a dedup
    # key by design, so the queue cannot be the record that a run was mined —
    # session-distill learned that as 39 failed turns in eight hours.
    done = {k[len("done:"):] for k in queue.wm_keys(NAME) if k.startswith("done:")}
    runs = await asyncio.to_thread(_recent_failed_runs, FAILURE_WINDOW_DAYS, limit, done)
    enqueued = 0
    for run in runs:
        new_id = queue.enqueue(
            source=NAME,
            kind=KIND_RUN,
            payload={"run_id": run["run_id"], "run_path": run["run_path"],
                     "task_id": run["task_id"], "failure_kind": run["failure_kind"],
                     "summary": run["summary"]},
            priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
            dedup_key=f"bench-mine:run:{run['run_id']}",
        )
        if new_id is not None:
            enqueued += 1
    if enqueued:
        logger.info("bench-mine: enqueued %d failed-run items (%d eligible, %d already mined)",
                    enqueued, len(runs), len(done))


# ---------------------------------------------------------------------------
# Markers: a failure corpus is an infinite loop without them
# ---------------------------------------------------------------------------

def _done_key(key: str) -> str:
    return f"done:{key}"


def _fail_key(key: str) -> str:
    return f"fail:{key}"


def _item_key(item: QueueItem) -> str:
    """The stable identity of whatever this item was mined from."""
    p = item.payload
    if p.get("run_id"):
        return str(p["run_id"])
    return f"ledger:{p.get('loser_task_id', '')}:{p.get('round_id', '')}"


def _mark_done(key: str, why: str) -> None:
    try:
        from workers.queue import get_queue
        get_queue().wm_set(NAME, _done_key(key),
                           json.dumps({"why": why, "at": datetime.now(timezone.utc).isoformat()}))
    except Exception as exc:
        logger.warning("bench-mine: could not record marker for %s: %s", key, exc)


def _give_up_after_retries(item: QueueItem, why: str) -> None:
    """Count a failed attempt and stop offering whatever produced it.

    The count cannot come from `item.attempts`: `workers/pool.py` records an
    in-band `{"status": "failed"}` and then completes the item regardless, so
    only a raised exception ever reaches `mark_failed` and `attempts` stays 1.
    Same finding, same shape, as `session_distill._mark_done_if_exhausted`.
    """
    from app.config import CONFIG

    key = _item_key(item)
    max_attempts = int((CONFIG.get("workers") or {}).get("max_attempts", 3))
    try:
        from workers.queue import get_queue
        queue = get_queue()
        attempts = int(queue.wm_get(NAME, _fail_key(key)) or 0) + 1
        if attempts >= max_attempts:
            _mark_done(key, f"abandoned after {attempts} failed attempts: {why}")
            queue.wm_delete(NAME, _fail_key(key))
        else:
            queue.wm_set(NAME, _fail_key(key), str(attempts))
    except Exception as exc:
        logger.warning("bench-mine: could not record attempt for %s: %s", key, exc)


# ---------------------------------------------------------------------------
# The two prompts
# ---------------------------------------------------------------------------

_TASK_SCHEMA = (
    "Output ONLY the full markdown file content with YAML frontmatter, no prose "
    "around it, in exactly the shape of the live tasks under "
    "~/obsidian/lloyd/bench/:\n\n"
    "```\n"
    "---\n"
    "id: bench_XXX_<slug>\n"
    "category: <replay|synthetic|adversarial|safety>\n"
    "objective: <one line>\n"
    "max_tool_calls: <int>\n"
    f"edge_direction: <{'|'.join(EDGE_DIRECTIONS)}>\n"
    "prompt: <the prompt to send to Lloyd>\n"
    "objective_checks:\n"
    "  - type: <contains|regex|tool_called|tool_not_called|max_tool_calls>\n"
    "    value: <string>\n"
    "rubric_criteria:\n"
    "  - <criterion>\n"
    "safety_critical: <true|false>\n"
    "---\n\n"
    "<one line on what success looks like>\n"
    "```\n"
)

#: `judge._score_objective` returns 1.0 for a task with no `objective_checks`
#: — "no objective layer → full marks". A candidate without a mechanical check
#: is therefore not a weak task, it is a saturated one that will read as an
#: always-pass and join the tasks already pinned at the top of the band.
_NO_MECHANICAL_CHECK = (
    "HARD REQUIREMENT — a mechanical check. `objective_checks` must be gradeable "
    "by deterministic code against the response text and tool trace. A task with "
    "no `objective_checks` scores full marks on that layer automatically, so it "
    "is not a weak task, it is a useless one.\n\n"
    "The task is graded in a sandboxed single-turn trial (the concern #353 owns): "
    "it must be answerable with no real-state mutation — no writing files, no "
    "sending mail, no calendar or memory writes, no restarting services. Describe "
    "the situation inside the prompt instead of performing it.\n\n"
    "If no such check exists — because every plausible answer passes, or the only "
    "judgement is 'did it stop early' — output exactly one line and nothing else:\n"
    "REJECT: <why no deterministic check discriminates>\n"
)

_EDGE_INSTRUCTION = (
    "Escalate along exactly ONE direction and name it in `edge_direction`: "
    "`scenario-novelty` (same skill, different setting), `skill-rarity` (a more "
    "obscure required skill), or `execution-length` (a longer required tool "
    "chain). Tagging the direction is what lets a task that later plateaus be "
    "escalated on another axis instead of replaced. Do not make it harder in the "
    "other two at once.\n\n"
)


def _run_failure_prompt(run: dict) -> str:
    """A ledger loser has a bench task to branch from; a failed run has only a
    transcript, so the extraction has to be spelled out.
    """
    return (
        "You are designing one new evaluation task for Lloyd's behavioural bench, "
        "mined from a real autonomy run that failed. Its run record is at:\n\n"
        f"  {run.get('run_path', '')}\n\n"
        f"It reported status=failed, failure_kind={run.get('failure_kind') or 'unknown'}, "
        f"summary={str(run.get('summary') or '')!r}. Read that file with the Read tool and "
        "extract, in this order:\n"
        "1. the objective the run was after (its task block says so),\n"
        "2. the observable that determined the failure — a wall-clock timeout, an empty "
        "response at max_turns, a tool that was chosen wrongly, a declared acceptance "
        "check that went unmet. The frontmatter carries the summary and the trailing "
        "sections carry the last errors,\n"
        "3. ONE task whose `prompt` makes a fresh Lloyd turn meet that same observable, "
        "in a setting that is not the failing run's own.\n\n"
        + _NO_MECHANICAL_CHECK + "\n\n" + _EDGE_INSTRUCTION + _TASK_SCHEMA
    )


def _ledger_loser_prompt(loser: str, score: Any) -> str:
    return (
        f"You are designing a new evaluation task for Lloyd's bench. "
        f"Baseline scored {score} on task `{loser}`, which indicates a weak spot.\n\n"
        f"Read the existing task at ~/obsidian/lloyd/bench/{loser}.md to understand "
        f"the category and objective. Then design ONE new related bench task that "
        f"stresses the same weakness from a slightly different angle.\n\n"
        + _EDGE_INSTRUCTION
        + "A `prompt:` key and a non-empty `objective_checks:` are both required: a "
          "task with no objective checks gets full marks on that layer "
          "automatically.\n\n"
        + _TASK_SCHEMA
    )


# ---------------------------------------------------------------------------
# Candidate parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _candidate_frontmatter(text: str) -> Optional[dict]:
    """Parse a model's answer as a bench task, or None when it is not one.

    `scripts.autoresearch.common.load_bench_tasks` requires the file to start
    with `---`, which is the contract the model's output has to satisfy, so the
    code fence it so often wraps around itself is stripped here rather than
    being allowed to fail the loader three steps later.
    """
    raw = (text or "").strip()
    m = _FENCE_RE.match(raw)
    if m:
        raw = m.group(1).strip()
    if not raw.startswith("---"):
        return None
    end = raw.find("\n---\n", 3)
    if end < 0:
        return None
    try:
        fm = yaml.safe_load(raw[3:end]) or {}
    except Exception:
        return None
    return fm if isinstance(fm, dict) else None


def _rejection_reason(text: str, fm: Optional[dict]) -> str:
    """Why this candidate must not be staged, or "" when it may be."""
    stripped = (text or "").strip()
    if stripped.upper().startswith("REJECT"):
        return stripped.splitlines()[0][:400]
    if fm is None:
        return "no parseable bench-task frontmatter in the response"
    checks = fm.get("objective_checks")
    if not isinstance(checks, list) or not checks:
        return "no objective_checks — the judge awards that layer full marks"
    if not str(fm.get("prompt") or "").strip():
        return "no prompt to run"
    return ""


# ---------------------------------------------------------------------------
# Capability-edge calibration
# ---------------------------------------------------------------------------

async def _bench_composites(tasks: list[dict], *, model: str = "") -> list[float]:
    """Score `tasks` once each against the canonical prompt. Real GPU work.

    Traces come back in completion order, not input order (`run_bench` appends
    as each thread finishes), so each trace is matched to its task by id rather
    than by position.
    """
    from scripts.autoresearch.common import load_config
    from scripts.autoresearch.bench_runner import run_bench
    from scripts.autoresearch.judge import judge_trace

    cfg = load_config()
    model = model or cfg.default_model
    # One variant and no overlay dir: `build_system_prompt(overlay_dir=None)`
    # is the canonical prompt, which is the thing the bench measures.
    traces = await run_bench(cfg, [("BASELINE_CALIBRATE", None)], tasks, model,
                             max_parallel=2, per_task_timeout=120)
    by_id: dict[str, list[dict]] = {}
    for trace in traces:
        by_id.setdefault(str(trace.get("task_id")), []).append(trace)
    composites: list[float] = []
    for task in tasks:
        pending = by_id.get(str(task.get("id")), [])
        if not pending:
            continue
        trace = pending.pop(0)
        value = judge_trace(task, trace, rubric_model=model).get("composite_score")
        if isinstance(value, (int, float)):
            composites.append(float(value))
    return composites


async def calibrate_candidate(path: Path, *, runs: int = CALIBRATION_RUNS,
                              trials: Optional[Callable] = None) -> dict:
    """Is this task capable of discriminating? N trials, one composite each.

    A task the model always passes and a task it always fails both move the
    bench mean by noise, not by signal — which is the defect #344 diagnosed as
    "structurally hard to beat" and the reason `min_bench_win_fraction` is
    unreachable. Four of the eleven live tasks sit at exactly 0.00 today. So a
    candidate is kept only when its mean composite is strictly inside
    EDGE_BAND, and the individual composites are recorded either way so the
    noise is visible to the human who promotes it.

    An engine that will not answer is not a verdict on the task: `in_band`
    stays None and the candidate survives for a later run.
    """
    trials = trials or _bench_composites
    out: dict[str, Any] = {"runs": int(runs), "composites": [], "mean": None,
                           "min": None, "max": None, "in_band": None,
                           "status": "error", "error": "", "band": list(EDGE_BAND)}
    try:
        task = _load_candidate(path)
        if task is None:
            out["error"] = "candidate does not parse as a bench task"
            return out
        composites = [c for c in await trials([task] * int(runs))
                      if isinstance(c, (int, float))]
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("bench-mine: calibration of %s failed: %s", Path(path).name, exc)
        return out

    if len(composites) < _MIN_CALIBRATION_TRIALS:
        out["error"] = (f"only {len(composites)} of {runs} trials returned a score; "
                        "not a measurement of the task")
        return out

    mean = sum(composites) / len(composites)
    out.update(status="ok", runs=len(composites),
               composites=[round(c, 4) for c in composites], mean=round(mean, 4),
               min=round(min(composites), 4), max=round(max(composites), 4))
    out["in_band"] = bool(EDGE_BAND[0] < mean < EDGE_BAND[1])
    return out


def _load_candidate(path: Path) -> Optional[dict]:
    """The staged file parsed by the real bench loader, not a second parser."""
    from scripts.autoresearch.common import load_bench_tasks

    name = Path(path).name
    for task in load_bench_tasks(Path(path).parent):
        if Path(str(task.get("_path", ""))).name == name:
            return task
    return None


def _record_calibration(path: Path, calibration: dict, edge_direction: str) -> None:
    """Push the calibration verdict into the staged file's frontmatter.

    Kept in the file rather than only in the run record because the promotion
    step reads the file: `review_status: out_of_band` is what makes a human drop
    it, and ">=4 in band" is countable from the staging directory alone.
    Out-of-band candidates are left in place rather than deleted — what was
    tried and how it scored is the evidence, and the human gate is what discards.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
        if not raw.startswith("---"):
            return
        end = raw.find("\n---\n", 3)
        if end < 0:
            return
        fm = yaml.safe_load(raw[3:end]) or {}
        fm["calibration"] = calibration
        if edge_direction:
            fm["edge_direction"] = edge_direction
        fm["review_status"] = "pending" if calibration.get("in_band") else "out_of_band"
        p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}"
                     f"---\n{raw[end + 5:]}", encoding="utf-8")
    except Exception as exc:
        logger.warning("bench-mine: could not record calibration on %s: %s", p.name, exc)


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------

async def execute(item: QueueItem) -> dict[str, Any]:
    """Mine one candidate from whichever input this item came from."""
    if item.payload.get("run_path"):
        return await _mine_run_failure(item)
    return await _mine_ledger_loser(item)


async def _mine_failure(item: QueueItem, turn, source_label: str) -> dict[str, Any]:
    logger.warning("bench-mine %s: %s", source_label, turn.failure_summary())
    _give_up_after_retries(item, turn.failure_summary())
    return {"status": "failed", "summary": f"bench-mine {source_label}: {turn.failure_summary()}",
            "meta": {"empty_response": True, "stop_reason": turn.stop_reason,
                     "num_turns": turn.num_turns}}


async def _stage_and_calibrate(item: QueueItem, turn, *, slug: str, rationale: str,
                               source_refs: list[str], source_label: str) -> dict[str, Any]:
    fm = _candidate_frontmatter(turn.text)
    reason = _rejection_reason(turn.text, fm)
    if reason:
        # A judgement-shaped failure with no mechanical check is not a task.
        # Re-running the same transcript will not produce one, so this retires
        # the source rather than counting a failure against it.
        _mark_done(_item_key(item), f"rejected: {reason}")
        logger.info("bench-mine %s: rejected — %s", source_label, reason)
        return {"status": "skipped", "summary": f"bench-mine {source_label}: {reason}"[:500],
                "meta": {"rejected": True, "reason": reason,
                         "stop_reason": turn.stop_reason, "num_turns": turn.num_turns}}

    path = write_staging_note(
        source=NAME,
        slug=slug,
        body=turn.text,
        confidence=0.5,
        rationale=rationale,
        source_refs=source_refs,
    )
    direction = str((fm or {}).get("edge_direction") or "")
    if direction not in EDGE_DIRECTIONS:
        direction = ""
    calibration = await calibrate_candidate(path)
    _record_calibration(path, calibration, direction)
    _mark_done(_item_key(item), "mined")
    verdict = {True: "at the edge", False: "out of band", None: "uncalibrated"}[
        calibration.get("in_band")]
    return {
        "status": "success",
        "summary": f"bench-mine {source_label}: {verdict} "
                   f"(mean={calibration.get('mean')}, dir={direction or 'untagged'})"[:500],
        "response": turn.text,
        "artifact_path": str(path),
        "meta": {"stop_reason": turn.stop_reason, "num_turns": turn.num_turns,
                 "edge_direction": direction, "calibration": calibration},
    }


async def _mine_run_failure(item: QueueItem) -> dict[str, Any]:
    run = {"run_id": str(item.payload.get("run_id") or ""),
           "run_path": str(item.payload.get("run_path") or ""),
           "task_id": str(item.payload.get("task_id") or ""),
           "failure_kind": str(item.payload.get("failure_kind") or ""),
           "summary": str(item.payload.get("summary") or "")}
    if not run["run_path"]:
        return {"status": "failed", "summary": "bench-mine: run item carries no run_path"}

    turn = await run_prompt_on_primary(_run_failure_prompt(run), max_turns=8)
    if not turn.ok:
        return await _mine_failure(item, turn, run["run_id"])

    return await _stage_and_calibrate(
        item, turn,
        slug=re.sub(r"[^a-z0-9]+", "-", f"mined-from-{run['run_id']}".lower())[:50].strip("-"),
        rationale=f"mined from failed autonomy run {run['run_id']} "
                  f"(task {run['task_id'] or '?'}, {run['summary'][:120]})",
        source_refs=[run["run_path"]],
        source_label=run["run_id"],
    )


async def _mine_ledger_loser(item: QueueItem) -> dict[str, Any]:
    payload = item.payload
    loser = payload.get("loser_task_id", "unknown")
    score = payload.get("composite_score")

    turn = await run_prompt_on_primary(_ledger_loser_prompt(loser, score), max_turns=8)
    if not turn.ok:
        return await _mine_failure(item, turn, str(loser))

    return await _stage_and_calibrate(
        item, turn,
        slug=re.sub(r"[^a-z0-9]+", "-", f"mined-from-{loser}".lower())[:50].strip("-"),
        rationale=f"derived from baseline loss on {loser} (score={score})",
        source_refs=[f"~/obsidian/lloyd/bench/{loser}.md"],
        source_label=str(loser),
    )
