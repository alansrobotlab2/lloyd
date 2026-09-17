"""Backlog task endpoints (Obsidian-vault-backed file store).

Three measured facts set the shape of this module (item #1199, 2026-09-16, live
box: 1,137 files / 11 MB in `~/obsidian/backlog`):

* reading all 11 MB costs **0.033 s**; **one** YAML frontmatter pass over the
  corpus costs **1.78 s**. The time is parsing, never I/O;
* the list route used to parse the corpus *twice* per request — once in
  `_backlog_board_map()` and once in its own loop — with `/boards` a third
  pass, and applied its filters *after* parsing. So the cost was flat: one task
  (`?board_id=1`, 2.3 KB out) cost 3.65 s, the same as all 1,137;
* both list handlers were `async def` while blocking on that parse, which took
  the event loop with it — `/health` answered in 3.1 ms idle and 3.496 s fired
  0.4 s into a backlog load, so the whole backend queued behind this page.

Hence: one scan shared by every route (`_backlog_scan`), a `(path, mtime_ns,
size)`-keyed cache over the parse (`_backlog_cached_fm`), plain `def` handlers
so FastAPI dispatches them to the threadpool, and a snippet instead of a body on
the list (`DESC_SNIPPET_CHARS`) with the full text behind
`GET /api/backlog/task/{id}`.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_mcp._shared import parse_frontmatter_text
from app.backlog_status import PIPELINE_STATUSES
from app.backlog_tags import normalize_tags


logger = logging.getLogger(__name__)

router = APIRouter()

_BACKLOG_DIR = Path.home() / "obsidian" / "backlog"
_VALID_STATUSES = frozenset(PIPELINE_STATUSES)
_BACKLOG_PATTERN = re.compile(r"^(\d+)[-_].*\.md$")
_BOARD_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
_BOARD_ICONS = ["📋", "📋", "📋", "📋", "📋"]

# The list row's `description` is a preview, not the item. `TaskCard` renders it
# through `line-clamp-2` — two visible lines — and bodies were 8,409,812 of the
# 9,202,414 bytes the route shipped (91.4 %), with a 118,494-byte maximum behind
# a card that shows two lines. The full body is the detail route's job, and
# `?q=` matches against it server-side so search did not shrink with the payload.
DESC_SNIPPET_CHARS = 300

# Parsed frontmatter, keyed by `str(path)`, validated against `stat()` on every
# lookup: `(mtime_ns, size, fm, body)`.
#
# Keyed on stat and *not* an in-process memo of results, because this directory
# has writers outside this process — `agent_mcp/backlog.py`,
# `agent-services/guardian/notify.py`, plus any editor or sync job touching the
# vault. A plain result memo would serve a stale row to the board, and worse: a
# route that *writes* from a stale read produces a stale write. Validating
# `mtime_ns` and `size` per lookup means the first request after any writer
# touches a file re-parses that one file and nothing else — which is also why
# this cache needs no invalidation hook to stay honest.
_FM_CACHE: dict[str, tuple[int, int, dict, str]] = {}

_FALLBACK_FIELDS = ("board", "status", "priority", "tags", "blocked", "assigned", "position")
# `low` since 2026-09-16, matching `agent_mcp/backlog.py::DEFAULT_PRIORITY`:
# the loop's pools honour this field, and a create that named no priority
# used to be stamped `none`, which nothing could sort.
_DEFAULT_PRIORITY = "low"

# The heading that carries the task name; everything after it is the body.
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _backlog_parse_fm(path: Path) -> tuple:
    """Read and parse one file: return (dict, body_str).

    Takes the *path*, not the text, because this is the unit the cache wraps — a
    warm row pays neither the read nor the parse, so both have to live behind one
    call.

    Uses the same graduated recovery as `agent_mcp/backlog.py` rather than a bare
    `yaml.safe_load`. Twenty items on the lloyd board (counted 2026-09-16, with
    stamps running to that day, so the set is growing; tracked as #918) have an
    activity_log entry with an unterminated quote; a strict parse raises, the
    callers here skip what they cannot parse, and those items vanish from the
    listing *and* from the board's task count with nothing logged. A degraded
    record beats an invisible one — which is the whole point of
    `parse_frontmatter_text`.

    The recovered dict carries `_yaml_broken`; `_reject_broken_fm` keeps it out of
    the writers, because the regex fallback only recovers `_FALLBACK_FIELDS` and
    rewriting a file from it would drop every key it did not extract.
    """
    content = path.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm = parse_frontmatter_text(
                parts[1], fallback_fields=_FALLBACK_FIELDS, log_label="backlog-api",
            )
            return fm, parts[2].strip()
    return {}, content


def _backlog_cached_fm(path: Path) -> tuple:
    """`_backlog_parse_fm` behind the stat-validated cache."""
    key = str(path)
    entry = _FM_CACHE.get(key)
    stat = path.stat()
    if entry is not None and entry[0] == stat.st_mtime_ns and entry[1] == stat.st_size:
        return entry[2], entry[3]
    fm, body = _backlog_parse_fm(path)
    _FM_CACHE[key] = (stat.st_mtime_ns, stat.st_size, fm, body)
    return fm, body


def _backlog_scan() -> list:
    """One stat-validated pass over the corpus: [(path, fm, body)].

    The only reader of the directory. `/boards`, the board map and `/tasks` all
    go through here, so the corpus is walked once per request instead of three
    times, and on a warm cache nothing is parsed at all. Files that cannot be
    read or parsed are logged and skipped: the old code swallowed them in
    silence, which is how an item could leave the board without a trace.
    """
    if not _BACKLOG_DIR.exists():
        return []
    found = []
    for f in _BACKLOG_DIR.glob("*.md"):
        if not _BACKLOG_PATTERN.match(f.name):
            continue
        try:
            fm, body = _backlog_cached_fm(f)
        except Exception:
            logger.debug("backlog: skipping unreadable/unparseable %s", f.name, exc_info=True)
            continue
        found.append((f, fm, body))
    _prune_fm_cache(found)
    return found


def _prune_fm_cache(found: list) -> None:
    """Drop cache entries for files that no longer exist.

    Runs after a full scan and is cheap when nothing changed: one length
    comparison. Without it the cache grew on every delete and a deleted item's
    body stayed resident forever.
    """
    if len(_FM_CACHE) <= len(found):
        return
    live = {str(f) for f, _, _ in found}
    for key in [k for k in _FM_CACHE if k not in live]:
        _FM_CACHE.pop(key, None)


def _board_index() -> tuple:
    """({board_name: id}, {board_name: task_count}) from one scan.

    Ids stay positional over `sorted(names)` — the filter and the browser's
    saved tab already depend on that shape — but they are not identities: a
    board appearing or vanishing renumbers every id after it. `board` (the name)
    is the identity; see `backlog_task_update`.
    """
    counts: dict[str, int] = {}
    for _, fm, _ in _backlog_scan():
        board = fm.get("board") or "default"
        counts[board] = counts.get(board, 0) + 1
    return {name: idx + 1 for idx, name in enumerate(sorted(counts))}, counts


def _backlog_board_map() -> dict:
    """Scan the corpus (through the cache), return {board_name: id} alphabetically."""
    return _board_index()[0]


def _write_task_file(filepath: Path, fm: dict, body: str) -> None:
    clean = {k: (v.isoformat() if isinstance(v, datetime) else v)
             for k, v in fm.items() if v is not None}
    fm_yaml = yaml.dump(clean, default_flow_style=False, allow_unicode=True, sort_keys=False)
    # Deliberately the byte-for-byte shape this route has always written, because
    # two of its consequences are not this item's to change: `yaml.dump` re-quotes
    # and re-wraps a hand-written frontmatter block, and `_backlog_parse_fm` hands
    # back a `.strip()`ed body, so a save ends the file without the trailing newline
    # that 764 of the 1,139 board files currently have. Both are why the clause
    # "a priority-only update leaves the file byte-identical apart from priority:
    # /updated:" is true of the **body and the other frontmatter lines** and not of
    # the last byte of a hand-written file — which `tests/test_backlog_body_
    # roundtrip.py::test_a_priority_only_update_survives_frontmatter_the_writer_
    # would_reflow` states as a test. Fixing the re-flow means preserving the block's
    # original text through a write; that is a write-path change, not a payload fix.
    filepath.write_text(f"---\n{fm_yaml}---\n\n{body}")
    # This write just made the cached copy wrong; drop the entry rather than
    # trust a same-second `mtime_ns` to differ from the one now on file. The
    # other entries are untouched, so a save costs one re-parse, not a rescan.
    _FM_CACHE.pop(str(filepath), None)


def _backlog_find_file(task_id: int) -> Path | None:
    """Find the markdown file for a given task ID."""
    if not _BACKLOG_DIR.exists():
        return None
    for f in _BACKLOG_DIR.glob("*.md"):
        m = _BACKLOG_PATTERN.match(f.name)
        if m and int(m.group(1)) == task_id:
            return f
    return None


def _split_body(body: str) -> tuple:
    """(name, description) — the `# heading` and everything after it."""
    heading = _HEADING_RE.search(body)
    if not heading:
        return "Untitled", ""
    return heading.group(1).strip(), body[heading.end():].strip()


def _snippet(text: str) -> str:
    """The head of a body, capped at `DESC_SNIPPET_CHARS`."""
    if len(text) <= DESC_SNIPPET_CHARS:
        return text
    return text[:DESC_SNIPPET_CHARS].rstrip() + "…"


def _row_from(path: Path, fm: dict, body: str, board_map: dict) -> dict:
    """One list row. `description` is a snippet: the whole body lives at
    `GET /api/backlog/task/{id}`, and `?q=` has already matched against it."""
    m = _BACKLOG_PATTERN.match(path.name)
    tid = int(m.group(1))
    name, description = _split_body(body)
    task_board = fm.get("board") or "default"
    stat = path.stat()
    created = fm.get("created") or fm.get("created_at") or ""
    updated = fm.get("updated") or fm.get("updated_at") or ""
    if not created:
        created = datetime.fromtimestamp(stat.st_ctime).isoformat()
    if not updated:
        updated = datetime.fromtimestamp(stat.st_mtime).isoformat()
    if isinstance(created, datetime):
        created = created.isoformat()
    if isinstance(updated, datetime):
        updated = updated.isoformat()
    snippet = _snippet(description)
    return {
        "id": tid,
        "name": name,
        # The snippet, and only the snippet, under a name that says what it is.
        # There is deliberately no `description` key on a list row: 344 KB of the
        # 1.27 MB first cut of this response was the same snippet a second time
        # under the old key, and an alias whose value differs in kind from the
        # body its name promises is how a caller ends up posting a snippet back.
        # `GET /api/backlog/task/{id}` is where `description` (the whole body)
        # lives; `?q=` above is what matches the part the row does not carry.
        "description_snippet": snippet,
        "status": fm.get("status", "draft"),
        "priority": fm.get("priority", _DEFAULT_PRIORITY),
        "blocked": fm.get("blocked", False),
        "tags": normalize_tags(fm.get("tags")),
        "completed": fm.get("status") == "done",
        "due_date": fm.get("due_date") or fm.get("due") or None,
        "position": fm.get("position", tid * 1000),
        "assigned_to_agent": fm.get("assigned", False),
        # Both, and the name is the real one. `board_id` is positional over
        # `sorted(names)`, so it renumbers whenever a board name appears or
        # disappears — and a board disappears exactly when its last task is
        # moved off it, which the task modal now does in one click. The vault,
        # `agent_mcp/backlog.py` and the frontmatter all key on the name; the id
        # stays for the existing filter/tab contract.
        "board": task_board,
        "board_id": board_map.get(task_board, 0),
        "group": fm.get("group"),
        "member_count": len(fm.get("members") or []),
        "url": "",
        "created_at": str(created),
        "updated_at": str(updated),
    }


# `def`, not `async def`, on every route here that reads the corpus: these
# handlers parse up to 11 MB of YAML, and a coroutine that blocks does not run
# slowly, it runs *exclusively* — `/health` measured 3.496 s behind a backlog
# request against a 3.1 ms idle baseline. FastAPI dispatches a non-coroutine
# endpoint with `run_in_threadpool`, which is where blocking work belongs.
# Pinned by tests/test_backlog_route_offload.py. The two write routes stay
# `async def` because they await `request.json()`; their corpus work is one file
# plus a cached board map, and they never held the loop for seconds.
@router.get("/api/backlog/boards")
def backlog_boards():
    if not _BACKLOG_DIR.exists():
        return JSONResponse([])
    board_map, counts = _board_index()
    boards = []
    for idx, name in enumerate(sorted(counts)):
        boards.append({
            "id": board_map[name],
            "name": name,
            "icon": _BOARD_ICONS[idx % len(_BOARD_ICONS)],
            "color": _BOARD_COLORS[idx % len(_BOARD_COLORS)],
            "tasks_count": counts[name],
        })
    return JSONResponse(boards)


@router.get("/api/backlog/task/{task_id}")
def backlog_task_detail(task_id: int):
    """One item with its **complete** body — the read half of the snippet split.

    The list cannot carry bodies (9 MB across 1,137 rows whose cards show two
    lines), so the editor reads the whole thing from here. Its `description` is
    the string `task-update` writes back, which is what makes the round-trip
    safe: the modal's editing source is this route, never the row.
    """
    filepath = _backlog_find_file(task_id)
    if not filepath:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    try:
        fm, body = _backlog_cached_fm(filepath)
    except Exception as exc:
        raise HTTPException(status_code=404,
                            detail=f"Task {task_id} could not be read: {exc}")
    name, description = _split_body(body)
    row = _row_from(filepath, fm, body, _board_index()[0])
    row["name"] = name
    row["description"] = description
    # `body_bytes` is the size of what is being edited, so the modal can say so
    # out loud. There is deliberately no `is_truncated` here: this route returns
    # the whole body, and a field named that way next to a 300-char snippet cap
    # would read as "this is a snippet" — the one thing it must never imply.
    row["body_bytes"] = len(description)
    return JSONResponse(row)


@router.get("/api/backlog/tasks")
def backlog_tasks(board_id: str = "", status: str = "", q: str = ""):
    if not _BACKLOG_DIR.exists():
        return JSONResponse([])
    board_map, _counts = _board_index()
    id_to_name = {v: k for k, v in board_map.items()}
    filter_board = ""
    if board_id:
        try:
            filter_board = id_to_name.get(int(board_id), board_id)
        except ValueError:
            filter_board = board_id
    # `?q=` exists because the row no longer carries the body. The client used to
    # grep `t.description` across 8.3 MB; a snippet would have stopped that
    # matching mid-body, so the grep moved here, over the scanned body. Same
    # behaviour, and nearly free once the parse is cached (item #1199, option a).
    needle = q.strip().lower()
    tasks = []
    for f, fm, body in _backlog_scan():
        try:
            task_board = fm.get("board") or "default"
            if filter_board and task_board != filter_board:
                continue
            if status and fm.get("status") != status:
                continue
            row = _row_from(f, fm, body, board_map)
            if needle:
                haystack = "\n".join(
                    [row["name"], body, " ".join(row["tags"]), task_board]
                ).lower()
                if needle not in haystack:
                    continue
            tasks.append(row)
        except Exception:
            logger.debug("backlog: skipping row for %s", f.name, exc_info=True)
            continue
    return JSONResponse(tasks)


@router.post("/api/backlog/task-update")
async def backlog_task_update(request: Request):
    data = await request.json()
    task_id = data.get("id")
    if not task_id:
        raise HTTPException(status_code=400, detail="id required")
    filepath = _backlog_find_file(task_id)
    if not filepath:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    # Read *through* the cache: this route is about to rewrite the file, so
    # building the new version from a snapshot another writer moved would turn a
    # stale read into a stale write.
    fm, body = _backlog_parse_fm(filepath)
    _reject_broken_fm(fm, filepath)
    board_map = _backlog_board_map()
    id_to_name = {v: k for k, v in board_map.items()}
    _, current_description = _split_body(body)
    if "name" in data:
        heading = _HEADING_RE.search(body)
        if heading:
            body = body[:heading.start()] + f"# {data['name']}" + body[heading.end():]
        else:
            body = f"# {data['name']}\n\n" + body
    description_ignored = False
    if "description" in data:
        new_description = data["description"]
        if not isinstance(new_description, str):
            raise HTTPException(status_code=400, detail="description must be a string")
        # The write half of the data-loss hazard in #1199's addendum. This field
        # replaces the *whole* body, and `TaskModal` seeds it from the list row
        # and posts it on every save — so the moment the row carries a snippet,
        # "open card, change priority, Save" overwrites a median-4,607-byte body
        # with 300 characters, silently, on every item. A shorter body is
        # therefore accepted only when the caller swears it means to: the modal
        # posts `force_body_replace` once the full text has arrived from
        # `GET /api/backlog/task/{id}`. Ignored-but-successful rather than a 400
        # keeps a priority click a priority click.
        if not data.get("force_body_replace") and len(new_description) < len(current_description):
            description_ignored = True
        else:
            heading = _HEADING_RE.search(body)
            if heading:
                body = body[:heading.end()].rstrip() + "\n\n" + new_description
            else:
                body = new_description
    if "status" in data:
        if data["status"] not in _VALID_STATUSES:
            raise HTTPException(status_code=400, detail=f"Invalid status '{data['status']}'. Must be one of: {', '.join(sorted(_VALID_STATUSES))}")
        fm["status"] = data["status"]
    for key in ("priority", "blocked", "position"):
        if key in data:
            fm[key] = data[key]
    if "tags" in data:
        fm["tags"] = normalize_tags(data["tags"])
    # `board` (a name) is preferred and `board_id` is the compatibility path.
    # An id is positional over the sorted board names, so a browser tab holding
    # a board list from before a board appeared or vanished sends an id that is
    # still *valid* — for a different board — and the item moves somewhere
    # nobody asked for. The name cannot drift that way. An unresolvable id used
    # to fall back to the task's current board and return success: a move the
    # user asked for that silently did not happen, which is the one outcome
    # worse than an error. It is a 400 now, like an invalid status above.
    if "board" in data:
        board = data["board"]
        if not isinstance(board, str) or not board.strip():
            raise HTTPException(status_code=400, detail="board must be a non-empty name")
        fm["board"] = board.strip()
    elif "board_id" in data:
        try:
            board_id = int(data["board_id"])
        except (TypeError, ValueError):
            board_id = None
        if board_id not in id_to_name:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown board_id {data['board_id']!r}. Known boards: "
                       + ", ".join(f"{i}={n}" for i, n in sorted(id_to_name.items())),
            )
        fm["board"] = id_to_name[board_id]
    if "assigned_to_agent" in data:
        fm["assigned"] = data["assigned_to_agent"]
    fm["updated"] = datetime.now().isoformat()
    _write_task_file(filepath, fm, body)
    return JSONResponse({"success": True, "description_ignored": description_ignored})


def _reject_broken_fm(fm: dict, filepath: Path) -> None:
    """Refuse to rewrite a file whose YAML only parsed by regex fallback."""
    if fm.get("_yaml_broken"):
        raise HTTPException(
            status_code=409,
            detail=f"{filepath.name} has malformed YAML frontmatter; "
                   "fix it by hand before editing it here (rewriting would "
                   "drop the fields the fallback parse could not read).",
        )


@router.post("/api/backlog/task-create")
async def backlog_task_create(request: Request):
    data = await request.json()
    name = data.get("name", "New Task")
    max_id = 0
    if _BACKLOG_DIR.exists():
        for f in _BACKLOG_DIR.glob("*.md"):
            m = _BACKLOG_PATTERN.match(f.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
    task_id = max_id + 1
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]
    filename = f"{task_id}-{slug}.md"
    board_map = _backlog_board_map()
    id_to_name = {v: k for k, v in board_map.items()}
    # Same preference as task-update: the name is the identity, the id is the
    # compatibility path. Create keeps the id's silent "default" fallback,
    # since a create with no resolvable board still has to land somewhere.
    board_name = data.get("board")
    if not isinstance(board_name, str) or not board_name.strip():
        board_name = id_to_name.get(data.get("board_id"), "default")
    else:
        board_name = board_name.strip()
    now = datetime.now().isoformat()
    create_status = data.get("status", "draft")
    if create_status not in _VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status '{create_status}'. Must be one of: {', '.join(sorted(_VALID_STATUSES))}")
    # A second live writer beside agent_mcp/backlog.py — the UI and
    # agent-services/guardian/notify.py both POST here — so fixing only the MCP
    # path leaves half of new tasks non-conformant. `type` is the one thing OKF
    # v0.1 requires of a concept file; without it the task is a violation at
    # birth and scripts/vault/validate_okf.py counts one more every time
    # (item #518). backlog_task_update only round-trips existing keys, so this
    # is the sole place a type can be declared, and updates never invent one.
    fm = {
        "type": "backlog",
        "segment": "backlog",
        "status": create_status,
        "priority": data.get("priority") or _DEFAULT_PRIORITY,
        "board": board_name,
        "blocked": False,
        "assigned": False,
        "position": task_id * 1000,
        "created": now,
        "updated": now,
    }
    if data.get("tags"):
        fm["tags"] = normalize_tags(data["tags"])
    body = f"# {name}"
    if data.get("description"):
        body += f"\n\n{data['description']}"
    _BACKLOG_DIR.mkdir(parents=True, exist_ok=True)
    _write_task_file(_BACKLOG_DIR / filename, fm, body)
    return JSONResponse({"success": True, "id": task_id})


@router.post("/api/backlog/task-delete")
async def backlog_task_delete(request: Request):
    data = await request.json()
    task_id = data.get("id")
    if not task_id:
        raise HTTPException(status_code=400, detail="id required")
    filepath = _backlog_find_file(task_id)
    if not filepath:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    filepath.unlink()
    _FM_CACHE.pop(str(filepath), None)
    return JSONResponse({"success": True})
