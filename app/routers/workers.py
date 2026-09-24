"""Workers router — HTTP surface for the unified work queue.

GET  /api/workers/status           — pool state + queue depth by source
GET  /api/workers/queue            — list queue items (filterable)
GET  /api/workers/runs             — list recent runs (filterable)
POST /api/workers/enqueue          — manually enqueue (for testing / agent hooks)
POST /api/workers/pause            — pause/resume worker draining
POST /api/workers/enable           — toggle workers.enabled in config.yaml
GET  /api/workers/pending          — list pending-research artifacts
GET  /api/workers/pending/read     — read one artifact's full content
POST /api/workers/pending/promote  — move artifact to a canonical vault location
     (a `bench-mine` artifact lands its bench-TASK block, validated, not its
      staging block — see `_bench_task_block`)
POST /api/workers/pending/reject   — move artifact to pending-research/_rejected/
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import CONFIG, save_tool_overrides
from app.paths import VAULT_ROOT, VAULT_PENDING_RESEARCH_DIR as PENDING_ROOT
from workers.queue import get_queue
from workers.pool import get_pool

REJECTED_ROOT = PENDING_ROOT / "_rejected"

# Default promotion destination per source (relative to vault root).
# Tuned so "just click Promote" does the right thing for the easy cases.
# Mirrored by DEFAULT_DEST in web/src/components/pages/WorkersPage.tsx. A
# retired source keeps no entry: `domain-research`'s stayed for the 142 notes
# it left staged, and those did not survive the 2026-09-22 data wipe (#1278).
_DEFAULT_DEST: dict[str, str] = {
    "bench-mine": "lloyd/bench",
}

router = APIRouter()
logger = logging.getLogger("lloyd-server")


def _meta_session_ids(meta: object) -> list[str]:
    """Union of the two session keys a run's parsed meta blob can carry."""
    ids: list[str] = []
    if not isinstance(meta, dict):
        return ids
    for key in ("session_ids", "session_id"):
        value = meta.get(key)
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                sid = candidate.strip()
                if sid not in ids:
                    ids.append(sid)
    return ids


def run_session_ids(run: object) -> list[str]:
    """The transcripts one run produced, parsed out of its `meta_json` blob.

    Two writers own the two keys. The pool binds `sessions_io.current_run_sessions`
    around each claimed job and writes the collected list on the normal, timeout
    and exception branches (`workers/pool.py`); a source stamps its own singular
    `session_id` into the same blob (`workers/sources/arch_review.py`,
    `workers/sources/autocode.py`). The field is their union because neither
    writer knows about the other, and it is de-duplicated because the two usually
    name one session: in the live table every row carrying both keys had the
    singular key naming an id the collected list already held, so a naive
    concatenation would offer the same transcript twice per row.

    Collected list first in the result — that is the order the pool recorded, and
    the singular key only ever appends an id the list never held.

    `[]` is the answer for an absent column, an unparseable blob, a JSON value
    that is not an object, and an object naming nothing. That is not a fallback:
    some sources record no transcript by design — `architecture/background-runs.md`
    §12 lists them — and a run that dies before its first `create_session` names
    none. An empty list is their normal value, and every consumer renders it as
    "no transcript".

    No row count is quoted here, on purpose. The `runs` table is pruned at 30 days
    (`workers/maintenance.py`), so any number written into this docstring is stale
    before the next person reads it — one moved twice inside the round that added
    this field. Re-measure instead, against the real file: read-only sqlite over
    the DB at `workers.db_path` — under `LLOYD_DATA`, not beside the checkout —
    grouping `run_session_ids(row)` by `source`.
    """
    raw = run.get("meta_json") if isinstance(run, dict) else None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    return _meta_session_ids(raw)


def _row_with_session_ids(run: object) -> object:
    """A copy of a `list_runs` row with the join lifted into its own field.

    A copy because `list_runs` is shared with `app/routers/dashboard.py` and
    `agent_mcp/autoresearch.py`, which never asked for the key; stamping it into
    the row they were handed would put a UI field in two unrelated readers.
    """
    if not isinstance(run, dict):
        return run
    return {**run, "session_ids": run_session_ids(run)}


@router.get("/api/workers/status")
async def workers_status():
    try:
        q = get_queue()
    except RuntimeError:
        return JSONResponse({"initialized": False})
    pool = get_pool()
    depth = q.depth_by_source()
    sources_cfg = CONFIG.get("workers", {}).get("sources", {}) or {}

    # Compose per-source health row.
    sources = []
    for name, src_cfg in sources_cfg.items():
        sources.append({
            "name": name,
            "enabled": bool(src_cfg.get("enabled", False)),
            "interval_seconds": src_cfg.get("interval_seconds"),
            "max_inflight": src_cfg.get("max_inflight"),
            "depth": depth.get(name, {}),
        })

    return JSONResponse({
        "initialized": True,
        "workers_enabled": bool(CONFIG.get("workers", {}).get("enabled", False)),
        "pool": pool.status() if pool else {"running": False},
        "depth": depth,
        "sources": sources,
    })


@router.get("/api/workers/health")
async def workers_health(days: int = 7, runs: int = 10):
    """Per-source health: config, queue depth, outcome rollup, recent runs.

    The workers' answer to `/api/autonomy/health`, and it exists because
    `/api/workers/status` reports only what a source is *allowed* to do —
    enabled, interval, max_inflight — plus how much is queued. A source that
    failed every run for a week looked identical to one that succeeded at
    every run: nothing anywhere joined a source to its outcomes.

    Degrades a section at a time rather than 503-ing the page, the same rule
    the dashboard follows: a health view is most useful when something is
    broken, so it must not be the second thing to break.
    """
    import asyncio as _asyncio
    from datetime import timedelta, timezone

    days = max(1, min(90, int(days)))
    runs = max(0, min(50, int(runs)))
    sources_cfg = CONFIG.get("workers", {}).get("sources", {}) or {}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        q = get_queue()
    except RuntimeError:
        # No queue yet is not an error: it is a box where the pool has never
        # started. Report the configured sources with empty outcomes rather
        # than an error string the page has to special-case.
        return JSONResponse({
            "initialized": False, "days": days,
            "sources": [{"name": n, "enabled": bool(c.get("enabled", False)),
                         "inner_voice": (None if c.get("inner_voice") is None
                                         else bool(c.get("inner_voice"))),
                         "interval_seconds": c.get("interval_seconds"),
                         "max_inflight": c.get("max_inflight"),
                         "priority": c.get("priority"),
                         "depth": {}, "health": None, "recent": []}
                        for n, c in sources_cfg.items()],
        })

    loop = _asyncio.get_event_loop()
    try:
        rollup = await loop.run_in_executor(None, q.run_rollup_by_source, since)
    except Exception as e:
        logger.warning("workers health rollup failed: %s", e)
        rollup = {}
    try:
        depth = await loop.run_in_executor(None, q.depth_by_source)
    except Exception as e:
        logger.warning("workers health depth failed: %s", e)
        depth = {}

    # Every source the config names AND every source the runs table knows
    # about. A source removed from config still has history worth reading,
    # and one whose runs all predate the window still has to appear.
    names = sorted(set(sources_cfg) | set(rollup) | set(depth))
    out = []
    for name in names:
        cfg = sources_cfg.get(name) or {}
        recent = []
        if runs:
            try:
                recent = await loop.run_in_executor(
                    None, partial(q.list_runs, source=name, limit=runs))
            except Exception:
                recent = []
            # The Sources panel is where a human asks "what did this run
            # actually do", so the join leaves this endpoint already parsed —
            # see `run_session_ids` for why both meta keys are needed.
            recent = [_row_with_session_ids(r) for r in recent]
        out.append({
            "name": name,
            "configured": name in sources_cfg,
            "enabled": bool(cfg.get("enabled", False)),
            # Recording is universal; observation is this switch. Surfaced
            # here because "was anyone watching?" is the first question about
            # a run that went wrong.
            #
            # Tri-state on purpose: `null` means the source does not set it,
            # which for a `run_prompt_on_primary` source is not "off, and you
            # could turn it on" — nothing there can be observed at all, since
            # the observer is wired in the chat endpoint and nowhere else.
            # Reporting a flat False would invite a knob that reads as broken.
            "inner_voice": (None if cfg.get("inner_voice") is None
                            else bool(cfg.get("inner_voice"))),
            "interval_seconds": cfg.get("interval_seconds"),
            "max_inflight": cfg.get("max_inflight"),
            "priority": cfg.get("priority"),
            "depth": depth.get(name, {}),
            "health": rollup.get(name),
            "recent": recent,
        })
    return JSONResponse({"initialized": True, "days": days, "sources": out})


@router.get("/api/workers/queue")
async def workers_queue(state: str = "", source: str = "", limit: int = 100):
    try:
        q = get_queue()
    except RuntimeError:
        return JSONResponse({"items": []})
    items = q.list_items(
        state=state or None,
        source=source or None,
        limit=min(max(1, limit), 500),
    )
    return JSONResponse({"items": [i.to_dict() for i in items]})


@router.get("/api/workers/runs")
async def workers_runs(source: str = "", task_id: str = "", limit: int = 50):
    try:
        q = get_queue()
    except RuntimeError:
        return JSONResponse({"runs": []})
    runs = q.list_runs(
        source=source or None,
        task_id=task_id or None,
        limit=min(max(1, limit), 500),
    )
    return JSONResponse({"runs": [_row_with_session_ids(r) for r in runs]})


@router.post("/api/workers/enqueue")
async def workers_enqueue(request: Request):
    try:
        q = get_queue()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="queue not initialized")
    data = await request.json()
    source = data.get("source")
    kind = data.get("kind", "manual")
    if not source:
        raise HTTPException(status_code=400, detail="source required")

    new_id = q.enqueue(
        source=source,
        kind=kind,
        payload=data.get("payload") or {},
        priority=int(data.get("priority", 50)),
        dedup_key=data.get("dedup_key"),
    )
    if new_id is None:
        return JSONResponse({"coalesced": True})
    return JSONResponse({"id": new_id})


@router.post("/api/workers/pause")
async def workers_pause(request: Request):
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="pool not running")
    data = await request.json() if (await request.body()) else {}
    paused = bool(data.get("paused", True))
    # `owner: automod` is the promoter's transient pause; anything else is an
    # operator's and survives a restart (`WorkerPool.pause`).
    owner = "automod" if data.get("owner") == "automod" else "operator"
    pool.pause(paused, owner=owner)
    return JSONResponse({"paused": pool.paused, "paused_by": getattr(pool, "paused_by", None)})


@router.post("/api/workers/enable")
async def workers_enable(request: Request):
    """Turn the pool on or off, and make it stick across a restart.

    This used to `yaml.dump(CONFIG)` over `config.yaml`, and each of the three
    things wrong with that was serious on its own. `CONFIG` is the *loaded*
    config, so the dump wrote the expanded values back: `${LIVEKIT_API_SECRET}`
    would have been replaced by the secret itself, in a tracked file. It
    flattened every comment in a 600-line file that is mostly comments. And it
    dirtied the live tree, which `scripts/automod/gate.py` and `promote.py`
    both refuse — so one click here silently stopped the self-modification
    loop until a human committed the damage. That is the identical defect the
    Tools page was moved off `config.yaml` to avoid; this endpoint kept it
    because nothing in the frontend calls it yet.

    So it takes the same route the Tools page does: the UI-mutable slice goes
    to the untracked overrides file, which `app.config` merges at boot.
    """
    data = await request.json() if (await request.body()) else {}
    enabled = bool(data.get("enabled", True))
    CONFIG.setdefault("workers", {})["enabled"] = enabled
    try:
        save_tool_overrides()
    except Exception as e:
        logger.warning("Failed to persist workers.enabled: %s", e)

    # And make it true now, not only after the next restart. A flag that says
    # "off" beside a pool that is still draining the queue is worse than no
    # flag at all.
    pool = get_pool()
    started = False
    if enabled and (pool is None or not pool.status().get("running")):
        await start_worker_pool()
        started = True
    elif not enabled and pool is not None and pool.status().get("running"):
        await pool.stop()

    return JSONResponse({"enabled": enabled, "started": started,
                         "pool": (get_pool().status() if get_pool() else {"running": False})})


# ── Pending-research review surface ──────────────────────────────────────


def _safe_pending_path(path_str: str) -> Path:
    """Resolve a claimed artifact path; 400 unless it's under pending-research/."""
    p = Path(path_str).expanduser().resolve()
    try:
        p.relative_to(PENDING_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="path must be under pending-research/")
    return p


def _safe_vault_dest(rel_or_abs: str) -> Path:
    """Resolve a destination path; 400 unless it stays under the obsidian vault."""
    p = Path(rel_or_abs).expanduser()
    if not p.is_absolute():
        p = VAULT_ROOT / p
    p = p.resolve()
    try:
        p.relative_to(VAULT_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="destination must be under obsidian vault")
    return p


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---\n"):
        return {}, content
    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return {}, content
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, parts[2]


# --- bench-mine promotion (#706) -------------------------------------------
#
# A staged bench-mine note is TWO documents in one file.
# `workers/sources/_common.py::write_staging_note` writes the mining run's
# metadata (`calibration`, `confidence`, `source_refs`, `generated_at`, …) as the
# file's FIRST frontmatter block, and `bench_mine._stage_and_calibrate` puts the
# mining turn's answer — the candidate's own bench-task block, fenced in ```
# whenever the turn pasted a fence anyway; 4 of the 14 files under
# `_pipeline/vault-derived/pending-research/bench-mine/` are fenced — into the
# BODY. Promoting the file as it stands therefore puts the STAGING block on top
# of `lloyd/bench/<file>.md`, and `load_bench_tasks`
# (`scripts/autoresearch/common.py`) reads only that first block as the task: no
# `id`, no `category`, no `prompt`, no `objective_checks`. `bench_runner` then
# posts `task.get("prompt") or task.get("_body")` — the calibration YAML dump —
# as the prompt, and `judge._score_objective` returns 1.0 for a task with no
# checks ("no objective layer → full marks"). So the documented human route
# could satisfy its own acceptance ("the bench goes 11 → >=15") while making the
# bench strictly worse: four guaranteed-pass tasks. #706 moves the judgement the
# human gate exists to make into code the gate can test.

#: Frontmatter keys the staging block owns and a bench task must not carry.
BENCH_STAGING_KEYS: tuple[str, ...] = (
    "calibration", "confidence", "source_refs", "generated_at",
)

#: The same fence-stripping rule `bench_mine._candidate_frontmatter` applies to
#: the raw mining answer, because that answer is what lands in the body verbatim.
_FENCED_TASK_RE = re.compile(
    r"^\s*```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)

#: A bench task id that is also safe as a filename.
_ID_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _bench_task_block(body: str) -> tuple[dict, str]:
    """The candidate's bench-task frontmatter and its prose, out of a staging body.

    Parsed by exactly the rule `load_bench_tasks` applies to a live bench file —
    the text starts with `---` and the first ``\n---\n`` closes the block — so
    what this returns is what the loader will see once it is on top of the
    landed file. Returns ``({}, "")`` when the body carries no task block.
    """
    raw = (body or "").strip()
    fenced = _FENCED_TASK_RE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
    if not raw.startswith("---"):
        return {}, ""
    end = raw.find("\n---\n", 3)
    if end < 0:
        return {}, ""
    try:
        fm = yaml.safe_load(raw[3:end]) or {}
    except Exception:
        return {}, ""
    if not isinstance(fm, dict):
        return {}, ""
    return fm, raw[end + 5:].strip()


def _bench_task_defect(task_fm: dict) -> str:
    """Why this candidate must not enter the graded bench, or "" when it may.

    The same pair `bench_mine._rejection_reason` refuses to *stage*, applied at
    the other end of the pipeline — a candidate can also reach here by a
    hand-`cp` of a staged file, which bypasses the source's own gate. `prompt`
    and `objective_checks` are what make a task gradeable at all; `category` is
    what the runner records on every row.
    """
    if not task_fm:
        return "no bench-task frontmatter block in the staging body"
    if not str(task_fm.get("prompt") or "").strip():
        return "no prompt to run"
    checks = task_fm.get("objective_checks")
    if not isinstance(checks, list) or not checks:
        return ("no objective_checks — judge._score_objective gives that layer "
                "full marks, so the landed task is a guaranteed pass")
    if not str(task_fm.get("category") or "").strip():
        return "no category — every bench row for it is recorded under 'unknown'"
    return ""


def _bench_default_filename(task_fm: dict, fallback: str) -> str:
    """Default the destination to the candidate's own task id, as `<id>.md`.

    The point is that the landed `id` equals the file's stem, so two candidates
    staged under one id collide on the destination instead of landing as two
    files that both declare it — `load_bench_tasks` keys a task on its
    frontmatter `id`, so a duplicate id means one task graded twice and another
    never at all, while the file count still goes up.
    """
    claimed = str(task_fm.get("id") or "").strip()
    if claimed and _ID_SAFE_RE.match(claimed):
        return f"{claimed}.md"
    return fallback


def _bench_ids_in(dest_dir: Path) -> set[str]:
    """Every id the bench directory already declares, per the real loader."""
    from scripts.autoresearch.common import load_bench_tasks

    try:
        return {str(t.get("id")) for t in load_bench_tasks(dest_dir)}
    except Exception as exc:
        # An unreadable bench must not be reported as an empty one: "no ids in
        # use" is the verdict that lets a duplicate land, and a guard that
        # cannot see its own input has no verdict to give.
        logger.warning("promote: could not scan bench ids in %s: %s", dest_dir, exc)
        raise HTTPException(
            status_code=500,
            detail=f"could not read the existing bench to check ids: {exc}",
        )


@router.get("/api/workers/pending")
async def workers_pending(source: str = "", limit: int = 200):
    """List pending-research artifacts with frontmatter + short preview."""
    if not PENDING_ROOT.exists():
        return JSONResponse({"items": [], "sources": []})

    items: list[dict] = []
    sources_seen: set[str] = set()
    for src_dir in PENDING_ROOT.iterdir():
        if not src_dir.is_dir() or src_dir.name.startswith("_") or src_dir.name.startswith("."):
            continue
        sources_seen.add(src_dir.name)
        if source and src_dir.name != source:
            continue
        for artifact in src_dir.rglob("*.md"):
            if artifact.name == "README.md":
                continue
            try:
                stat = artifact.stat()
                content = artifact.read_text(encoding="utf-8")
            except OSError:
                continue
            fm, body = _parse_frontmatter(content)
            preview = body.strip().split("\n\n", 1)[0].strip()[:220]
            items.append({
                "path": str(artifact),
                "source": src_dir.name,
                "date": artifact.parent.name,
                "filename": artifact.name,
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "frontmatter": fm,
                "preview": preview,
            })

    items.sort(key=lambda i: i["mtime"], reverse=True)
    return JSONResponse({
        "items": items[: max(1, min(limit, 1000))],
        "sources": sorted(sources_seen),
    })


@router.get("/api/workers/pending/read")
async def workers_pending_read(path: str):
    p = _safe_pending_path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    try:
        content = p.read_text(encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"read failed: {e}")
    fm, body = _parse_frontmatter(content)
    return JSONResponse({
        "path": str(p),
        "source": p.relative_to(PENDING_ROOT).parts[0] if PENDING_ROOT in p.parents else None,
        "frontmatter": fm,
        "body": body,
        "raw": content,
    })


@router.post("/api/workers/pending/promote")
async def workers_pending_promote(request: Request):
    """Move a pending artifact to a canonical vault location.

    Body: { path, destination?, filename? }
      - path: artifact to promote (must be under pending-research/)
      - destination: directory under obsidian vault (absolute or relative).
        Defaults per-source — see _DEFAULT_DEST. Required for sources without
        a default (gap-fill, session-distill).
      - filename: override destination filename. Defaults to the artifact's name,
        except for a `bench-mine` artifact, which defaults to `<task id>.md` so
        the landed id and the landed stem are the same string.

    A `bench-mine` artifact is treated as the two documents it is (#706): the
    candidate's bench-TASK block is what lands, the staging block does not, and a
    candidate whose task block cannot be graded (no prompt, no objective_checks,
    no category, no task block at all) is refused with 400 and nothing is
    written — see `_bench_task_defect`. Its `id` is rewritten to the destination
    stem and a stem the bench already declares is refused with 409, so several
    candidates staged under one id cannot all land.
    """
    data = await request.json()
    src = _safe_pending_path(data.get("path", ""))
    if not src.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    src_name = src.relative_to(PENDING_ROOT).parts[0]
    dest_dir_str = data.get("destination") or _DEFAULT_DEST.get(src_name)
    if not dest_dir_str:
        raise HTTPException(
            status_code=400,
            detail=f"no default destination for source '{src_name}' — provide 'destination'",
        )
    dest_dir = _safe_vault_dest(dest_dir_str)

    filename = data.get("filename") or src.name
    staging_only = src_name == "bench-mine"

    # Read before the destination is resolved: for a bench-mine artifact the
    # default filename comes from inside the file (its own task id).
    try:
        content = src.read_text(encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"read failed: {e}")
    fm, body = _parse_frontmatter(content)

    if staging_only:
        task_fm, task_body = _bench_task_block(body)
        defect = _bench_task_defect(task_fm)
        if defect:
            raise HTTPException(
                status_code=400,
                detail=f"bench-mine candidate is not a gradeable bench task: {defect}",
            )
        if not data.get("filename"):
            filename = _bench_default_filename(task_fm, src.name)

    dest_path = _safe_vault_dest(str(dest_dir / filename))

    if dest_path.exists():
        raise HTTPException(status_code=409, detail=f"destination exists: {dest_path}")

    if staging_only:
        # `load_bench_tasks` keys on the frontmatter `id`, so an id already
        # declared under a DIFFERENT filename (a hand-cp'd staged file, the case
        # the human route also has to survive) is still a duplicate.
        stem = dest_path.stem
        if stem in _bench_ids_in(dest_path.parent):
            raise HTTPException(
                status_code=409,
                detail=f"bench id '{stem}' is already declared under {dest_path.parent}",
            )
        # The landed file is the bench task and nothing else. Where the staging
        # verdict lives after the move: the run record's `meta.calibration` and
        # `artifact_path` (workers.db) already name it, and the source is
        # consumed by this call.
        task_fm["id"] = stem
        task_fm["review_status"] = "promoted"
        task_fm["promoted_at"] = datetime.now(timezone.utc).isoformat()
        new_content = (
            "---\n"
            + yaml.dump(task_fm, default_flow_style=False, allow_unicode=True)
            + "---\n\n"
            + task_body
            + "\n"
        )
    else:
        # Update frontmatter review_status before move.
        if fm:
            fm["review_status"] = "promoted"
            fm["promoted_at"] = datetime.now(timezone.utc).isoformat()
            new_content = (
                "---\n"
                + yaml.dump(fm, default_flow_style=False, allow_unicode=True)
                + "---\n"
                + body
            )
        else:
            new_content = content

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        dest_path.write_text(new_content, encoding="utf-8")
        src.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"promote failed: {e}")

    return JSONResponse({
        "promoted": True,
        "from": str(src),
        "to": str(dest_path),
    })


@router.post("/api/workers/pending/reject")
async def workers_pending_reject(request: Request):
    """Move a pending artifact to pending-research/_rejected/ (recoverable)."""
    data = await request.json()
    src = _safe_pending_path(data.get("path", ""))
    if not src.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    rel = src.relative_to(PENDING_ROOT)
    dest = REJECTED_ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest = dest.with_name(f"{dest.stem}-{int(datetime.now().timestamp())}{dest.suffix}")
    shutil.move(str(src), str(dest))
    return JSONResponse({"rejected": True, "from": str(src), "to": str(dest)})


# ── Startup hook — registered in server.py ───────────────────────────────


async def start_worker_pool() -> None:
    """Initialize the queue, register sources (via import), and start the pool."""
    cfg = CONFIG.get("workers", {}) or {}
    if not cfg.get("enabled", False):
        logger.info("Worker pool disabled in config — not starting")
        return

    from workers.queue import configured_db_path, get_queue

    db_path = configured_db_path()
    # Same default as `WorkerPool.__init__`. They disagreed (8 here, 4 there),
    # so "what happens with no `workers.slots` in config" had two answers
    # depending on which one you read.
    slots = int(cfg.get("slots", 4))
    max_attempts = int(cfg.get("max_attempts", 3))

    q = get_queue(db_path)

    # Importing sources registers them into SOURCE_REGISTRY.
    import workers.sources  # noqa: F401

    from workers.pool import start_pool
    await start_pool(q, slots=slots, max_attempts=max_attempts)
    logger.info("Worker pool startup complete (slots=%d, db=%s)", slots, db_path)
