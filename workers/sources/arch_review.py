"""arch-review — one architecture review unit per real session, and the doc edits itself.

`architecture/` was hand-curated on 2026-09-11: 22 top-level docs, 17 retired
into the gitignored `.archive/`, an `index.md` that lists them. Nothing keeps
it honest from there. Three `tests/test_*_doc_claims.py` pin numbers in three
of the 22; `agent_mcp/memory_ops.py:12` cites a doc that no longer exists; the
measured tables in `autonomy-jobs.md` and `workers-jobs.md` are snapshots of
one afternoon. A doc that has drifted is worse than no doc, because it is read
as current.

This source is Alan's shape: a picklist of subjects with a last-run date, and
the oldest line gets one real session that reviews the architecture **and** the
code, edits the doc itself, and files everything else as backlog items.

Two kinds of unit, because the docs are not all the same shape:

* a **doc** — any top-level `architecture/*.md`, reviewed whole; and
* a **group** — one functional section of `autonomy-jobs.md` or
  `workers-jobs.md`. Those two docs group the unattended fleet by what its
  members are *for* (seven autonomy functions, four worker families), and a
  function is a review unit of its own: who may write, which consumer does not
  exist, which edges cross groups, which defect three members share. One
  session over an 824-line doc answers none of those; one session over
  "Distil" answers all four.

Three properties decide whether this is safe, and they are the whole module:

* **Every review edits production.** `~/lloyd` is the running tree, so a saved
  file is a deploy. `Write` is denied — the doc already exists and `Edit` is
  the only verb needed. Every other path the turn touched — anywhere in this
  repo, and anywhere in the vault except `backlog/`, where its filings belong —
  is reverted after the turn from a `git status` baseline taken before it, and
  a path that was *already* dirty is reported by content hash rather than
  reverted, because somebody else is mid-edit on it. What a `git status` sweep
  can never see is an **ignored** path, so the tools that write one
  (`fact_*` under `_pipeline/`, `memory_*`) are denied outright rather than
  swept. The doc's own diff is bounded
  (`max_delta_lines`, `max_shrink_pct`, front matter intact) and a group's diff
  must land inside its own section. Whatever survives all of that, the
  **source** commits — the model never runs `git`.
* **A finding is filed, not fixed.** The turn may change one doc and nothing
  else. A skill with a phantom tool name, an unbounded autonomy step, a dead
  consumer: those become `arch-review` drafts, tagged `spawned-by-review`, and
  their ids are verified on disk before the ledger records them.
* **The picklist is the scheduler.** Never-reviewed first, then oldest
  `last_reviewed_at`, docs before groups on a tie. A unit rests
  `review_interval_days` after a review; three failed attempts park it. There
  is no churn trigger and no backoff — the picklist is read from disk at call
  time, and at `daily_max` 4 a board this size is a first pass in about a
  week, then a month's rest per unit. No count is written down anywhere: the
  first one went stale the moment this module's own doc joined the picklist
  it describes, which is exactly the drift the pass exists to find.

The tag asymmetry in step 3 of the plan is the one thing to hold in mind when
reading `scripts/automod/backlog.py` beside this: `spawned-by-review` is merged
at write time and expired at 30 days like every other loop-filed item, and is
deliberately **not** quarantined. Quarantine asks "can this item answer the
staleness question?", and a review finding can: it describes the tree as of one
commit, and single triage is exactly the pass that should judge it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from app.paths import LLOYD_HOME, VAULT_ROOT
from workers.queue import WorkQueue, QueueItem
from workers.sources._common import (
    WORKER_AUTOMOD_BAN, DrainActive, TurnTimeout, run_prompt_in_session,
)

logger = logging.getLogger("lloyd-workers.arch-review")

NAME = "arch-review"
#: Between `youtube-digest` (60) and `backlog-cluster` (65). A review is
#: routine maintenance: it must not delay an automod round or a digest, and it
#: should still get a slot before the night's clustering.
DEFAULT_PRIORITY = 62
#: It holds a session for up to an hour reading a 1800-line doc and the code it
#: names, so the pool's KV gate must hold it while the primary is under
#: pressure. `tests/test_kv_gate.py` pins the long-lived set.
LONG_LIVED = True

#: Where the picklist lives, relative to the repo root.
ARCH_DIRNAME = "architecture"
#: Reviewed docs are the top-level ones only. `architecture/.archive/` holds
#: retired docs — five are still tracked — and a review of a doc that was
#: retired *for being wrong* is the one review with no possible value.
EXCLUDED_DIRS = (".archive",)

DOC_STATUSES = ("current", "stale", "superseded", "aspirational")
#: A doc's verdict says whether the prose is true. A group's says whether the
#: grouping still holds — `split` when one function has become two, `merge`
#: when two are really one, `move` when a member belongs elsewhere.
GROUPINGS = ("holds", "split", "merge", "move")
#: What the model answers when there is no grouping question — a doc unit. The
#: schema carries it as a string rather than a nullable, for the reason
#: `GROUP_TRIAGE_SCHEMA` uses `item_id: 0`: guided decoding is steadier over a
#: flat enum than over an `anyOf` with null, and the parser maps it back.
GROUPING_NONE = "none"

REVIEW_TAG = "arch-review"
SPAWN_TAG = "spawned-by-review"

#: What a review turn may not touch. It keeps `Bash` — there is no bundle, so
#: `git log --since` on the paths a doc names and `curl` on the health routes
#: are how it checks a claim against the live system — and `Edit`, which is the
#: job. `Write` is denied because the doc exists; a new file is either a
#: finding to file or a stray write to revert. `Task` is denied because a
#: subagent's writes land on the parent's turn and this turn's writes are
#: bounded by a diff the source inspects.
DISALLOWED: tuple[str, ...] = (
    *WORKER_AUTOMOD_BAN,
    "Task", "Write",
    "vault_write",
    # The memory surface writes `~/obsidian/lloyd/MEMORY.md` and `USER.md`, and
    # the fact surface writes `_pipeline/vault-derived/facts/**` plus
    # `kg.sqlite`. Both were reachable because `DISALLOWED` subtracts from the
    # whole chat toolbox rather than granting from nothing (#709), and neither
    # is any part of reviewing a document.
    #
    # The fact half is why this list is load-bearing rather than belt-and-
    # braces: `_pipeline/` is gitignored (`.gitignore:25`), and `git status`
    # does not report ignored paths at all. The sweep in `revert_strays` is
    # structurally incapable of seeing a fact write, so denying the tool is
    # not a second line of defence there — it is the only one.
    "memory_add", "memory_remove", "memory_replace",
    "fact_add", "fact_relate", "fact_invalidate", "fact_resolve",
    "http_request",
    "browser_evaluate", "browser_fill", "browser_type", "browser_click",
    "browser_press", "browser_cookies", "browser_drag", "browser_select",
    "autonomy_write_task", "autonomy_delete_task", "autonomy_run_task",
    "autonomy_config",
    "research_propose", "research_next", "research_complete",
    "graph_refresh",
)

#: The one vault prefix the sweep ignores, because writing there is the job.
#: Everything else in the vault is swept.
#:
#: This was `("skills", "autonomy")` — an allowlist of the two directories the
#: prompt talks about — which left `lloyd/MEMORY.md`, `knowledge/`, `projects/`
#: and the rest of the vault unwatched (#709). Naming what is *exempt* rather
#: than what is *guarded* is the difference between a sweep that covers what
#: somebody thought of and one that covers what exists: a directory added to
#: the vault next month is swept by default instead of silently not being.
VAULT_UNSWEPT_PREFIXES = ("backlog/",)

#: Defaults for every knob, so a source config that predates a key still runs.
DEFAULT_REVIEW_INTERVAL_DAYS = 30
DEFAULT_RETRY_SPACING_SECONDS = 6 * 3600
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_DAILY_MAX = 4
DEFAULT_MAX_OPEN_ITEMS = 25
DEFAULT_SPAWN_CAP = 5
DEFAULT_MAX_DELTA_LINES = 400
DEFAULT_MAX_SHRINK_PCT = 30
DEFAULT_MAX_TURNS = 90
DEFAULT_BATCH = 2
#: A doc past this many lines is read in sections rather than in one call.
BIG_DOC_LINES = 600

STATE_FILENAME = "arch_review.json"
_GIT_TIMEOUT = 60


# ── The picklist ─────────────────────────────────────────────────────────────


def repo_root() -> Path:
    """The tree under review. Read at call time, never bound at import: a
    default bound at import is what made an early test run write into the real
    state dir instead of its tmp one."""
    return LLOYD_HOME


def arch_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / ARCH_DIRNAME


def doc_slugs(root: Path | None = None) -> list[str]:
    """Every reviewable doc, by slug, sorted.

    `glob("*.md")` does not descend, so `.archive/` is already out; the
    explicit filter is here because five archived docs are still tracked and a
    future `rglob` would silently pick them up.
    """
    d = arch_dir(root)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.md")):
        rel = p.relative_to(d)
        if len(rel.parts) != 1 or any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        out.append(p.stem)
    return out


def parse_groups(raw: Iterable[Any] | None) -> list[tuple[str, str]]:
    """`["autonomy-jobs:Distil", ...]` → `[("autonomy-jobs", "Distil"), ...]`.

    Hand-kept in config, which is the cost of not parsing the docs' own tables.
    When Alan regroups, one line changes; until it does, `section_missing` and
    the jobs doc's own `groups_config` check are the two tells.
    """
    out: list[tuple[str, str]] = []
    for entry in raw or []:
        doc, sep, name = str(entry).partition(":")
        doc, name = doc.strip(), name.strip()
        if not sep or not doc or not name:
            logger.warning("arch-review: ignoring malformed group %r", entry)
            continue
        out.append((doc, name))
    return out


def unit_id(kind: str, doc: str, name: str = "") -> str:
    return f"doc:{doc}" if kind == "doc" else f"group:{doc}:{name}"


def all_units(src_cfg: dict, root: Path | None = None) -> list[dict]:
    """Every unit the picklist offers, docs first then groups, each in order."""
    units = [{"unit": unit_id("doc", s), "kind": "doc", "doc": s, "name": ""}
             for s in doc_slugs(root)]
    known = set(doc_slugs(root))
    for doc, name in parse_groups(src_cfg.get("groups")):
        if doc not in known:
            logger.warning("arch-review: group %s:%s names a doc that is not there", doc, name)
            continue
        units.append({"unit": unit_id("group", doc, name), "kind": "group",
                      "doc": doc, "name": name})
    return units


# ── Sections ─────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^##\s+(.*\S)\s*$")
_LEADING_NUMBER_RE = re.compile(r"^\d+[.)]\s+")


def heading_key(text: str) -> str:
    """The comparable half of a `## ` heading.

    Both jobs docs decorate their headings with what the section contains —
    `## Distil: #38 #42 …`, `## 6. Mining — Lloyd's own exhaust back out` — and
    those suffixes change whenever a member is added. Matching on the whole
    line would make every regroup a `section_missing`, so the key is the
    heading minus a leading `N. ` and minus everything from the first `:` or
    ` — ` on.
    """
    t = _LEADING_NUMBER_RE.sub("", str(text or "").strip())
    for sep in (":", " — ", " – ", " - "):
        i = t.find(sep)
        if i != -1:
            t = t[:i]
    return t.strip().casefold()


def find_section(text: str, name: str) -> Optional[tuple[int, int]]:
    """1-based `(start, end)` line numbers of the `## ` section called `name`.

    `start` is the heading line itself and `end` is the last line before the
    next `## ` (or EOF). `###` subheadings and `---` rules inside the section
    belong to it; only a level-2 heading ends it.
    """
    want = heading_key(name)
    if not want:
        return None
    lines = str(text or "").splitlines()
    start = None
    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if not m:
            continue
        if start is not None:
            return (start, i - 1)
        if heading_key(m.group(1)) == want:
            start = i
    return (start, len(lines)) if start is not None else None


# ── State ────────────────────────────────────────────────────────────────────


def state_path() -> Path:
    from scripts.automod import state as S
    return S.STATE_DIR / STATE_FILENAME


def load_state(path: Path | None = None) -> dict:
    p = path or state_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(data: dict, path: Path | None = None) -> Path:
    p = path or state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def _parse_ts(value: Any) -> float:
    """An ISO timestamp from state → epoch seconds; 0.0 when absent or bad."""
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def is_pending(row: dict | None, *, now: float, interval_days: float,
               retry_spacing: float, max_attempts: int) -> bool:
    """Is this unit due, and not parked?

    Parked means `max_attempts` consecutive failures with the last one inside
    `retry_spacing`. It is a spacing rule rather than a permanent stop: a unit
    whose doc is unreadable today is reviewable next week, and a unit nobody
    ever fixes still costs one run every six hours rather than one per tick.
    """
    row = row or {}
    attempts = int(row.get("attempts") or 0)
    if attempts >= int(max_attempts):
        last = _parse_ts(row.get("last_attempt_at"))
        if last and (now - last) < float(retry_spacing):
            return False
    last_reviewed = _parse_ts(row.get("last_reviewed_at"))
    if not last_reviewed:
        return True
    return (now - last_reviewed) >= float(interval_days) * 86400


def pending_units(units: list[dict], state: dict, *, now: float, interval_days: float,
                  retry_spacing: float, max_attempts: int) -> list[dict]:
    """Due units, never-reviewed first, then oldest `last_reviewed_at`.

    Docs before groups on a tie, which is `all_units`' own order — so the first
    pass over a fresh state file walks the docs alphabetically and then the
    groups in config order, and every later pass is driven purely by age.
    """
    due = [u for u in units
           if is_pending(state.get(u["unit"]), now=now, interval_days=interval_days,
                         retry_spacing=retry_spacing, max_attempts=max_attempts)]
    order = {u["unit"]: i for i, u in enumerate(units)}
    return sorted(due, key=lambda u: (_parse_ts((state.get(u["unit"]) or {}).get("last_reviewed_at")),
                                      order[u["unit"]]))


def reviews_today(ledger: Path | None = None, *, now: float | None = None) -> int:
    """`arch_review` ledger events in the last 24 h.

    Counted from the ledger rather than from a counter in the state file
    because the ledger is the record that survives a state file being deleted,
    and `daily_max` is a bound on how much of the board this rewrites in a day.
    """
    from scripts.automod import state as S
    path = ledger or S.LEDGER_PATH
    now = now or time.time()
    cutoff = now - 86400
    n = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or '"arch_review"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") == "arch_review" and float(d.get("ts") or 0) >= cutoff:
                n += 1
    except OSError:
        return 0
    return n


# ── Git ──────────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str, timeout: int = _GIT_TIMEOUT) -> str:
    """Run one git command in `repo` and return stdout ('' on any failure).

    Blocking, so every caller reaches it through `asyncio.to_thread` —
    `tests/test_workers_pool.py` greps `execute` for exactly this mistake.
    """
    try:
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("arch-review: git %s in %s failed: %s", args[0] if args else "?", repo, exc)
        return ""
    return r.stdout if r.returncode == 0 else ""


def _porcelain(repo: Path, *pathspec: str) -> set[str]:
    """Paths git reports as changed, as repo-relative strings.

    A rename is reported as `old -> new`; both halves are taken, because either
    one appearing where it was not before is a write this turn made.
    """
    args = ["status", "--porcelain", "--untracked-files=all"]
    if pathspec:
        args += ["--", *pathspec]
    out = _git(repo, *args)
    paths: set[str] = set()
    for line in out.splitlines():
        raw = line[3:].strip()
        if not raw:
            continue
        for part in raw.split(" -> "):
            part = part.strip().strip('"')
            if part:
                paths.add(part)
    return paths


def vault_dirty(root: Path | None = None) -> set[str]:
    """Dirty paths in the vault that this job is answerable for.

    The whole repo minus `backlog/`. Measured at 15 ms over a vault carrying
    254 dirty paths, 227 of them backlog items, so scoping by pathspec bought
    nothing and cost the coverage.
    """
    return {p for p in _porcelain(root or VAULT_ROOT)
            if not p.startswith(VAULT_UNSWEPT_PREFIXES)}


def fingerprints(repo: Path, paths: Iterable[str]) -> dict[str, str]:
    """`{path: sha1-of-bytes}` for paths that exist; absent ones are omitted.

    The answer to the sweep's blind spot (#915). `revert_strays` compares the
    *set* of dirty paths, so a path that was already dirty before the turn
    cannot appear in `after - before` however much the turn rewrote it — and
    the vault's `autonomy/*.md` task files are dirty on nearly every run,
    because the scheduler rewrites one each time it runs a task. Content is
    what tells those apart.
    """
    out: dict[str, str] = {}
    for rel in paths:
        p = _under(repo, rel)
        if p is None:
            continue
        try:
            out[rel] = hashlib.sha1(p.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def modified_preexisting(repo: Path, before: dict[str, str],
                         keep: Iterable[str] = ()) -> list[dict]:
    """Already-dirty paths whose CONTENT the turn changed.

    Reported, never reverted, and that asymmetry is the whole point: a path
    that was already dirty is one somebody else — a human with an editor open,
    the autonomy scheduler mid-run — is in the middle of. Restoring it would
    destroy their uncommitted work to undo ours, which is worse than the write
    being undone. Saying so is the honest half of what the sweep can do here.
    """
    kept = set(keep)
    now = fingerprints(repo, [p for p in before if p not in kept])
    out = []
    for rel, sha in sorted(before.items()):
        if rel in kept:
            continue
        after = now.get(rel)
        if after is not None and after != sha:
            out.append({"repo": repo.name, "path": rel, "action": "reported (was already dirty)"})
        elif after is None:
            out.append({"repo": repo.name, "path": rel, "action": "reported (deleted; was already dirty)"})
    return out


def _is_tracked(repo: Path, rel: str) -> bool:
    return bool(_git(repo, "ls-files", "--error-unmatch", "--", rel).strip())


def _numstat(repo: Path, rel: str) -> tuple[int, int]:
    """`(added, deleted)` for one path against HEAD. `(0, 0)` when unchanged."""
    out = _git(repo, "diff", "--numstat", "HEAD", "--", rel)
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return 0, 0
    return 0, 0


def _head_line_count(repo: Path, rel: str) -> int:
    out = _git(repo, "show", f"HEAD:{rel}")
    return len(out.splitlines())


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? ")


def changed_hunks(repo: Path, rel: str) -> list[tuple[int, int]]:
    """Old-side `(line, count)` per hunk of the working diff, `-U0`.

    Zero context is the point: with context lines a one-line edit reports a
    hunk that reaches three lines into the neighbouring section, and a group
    review that touched only its own text would be rejected for touching
    somebody else's.
    """
    out = _git(repo, "diff", "-U0", "HEAD", "--", rel)
    hunks: list[tuple[int, int]] = []
    for line in out.splitlines():
        m = _HUNK_RE.match(line)
        if m:
            hunks.append((int(m.group(1)), 1 if m.group(2) is None else int(m.group(2))))
    return hunks


def hunk_inside(hunk: tuple[int, int], start: int, end: int) -> bool:
    """Does this old-side hunk fall inside `[start, end]`?

    Two shapes, and they are indexed differently. A pure insertion is
    `@@ -L,0` and means "after old line L", so it is inside when
    `start <= L <= end` — an insertion at `L == start` lands immediately after
    the heading and an insertion at `L == end` lands at the section's tail,
    and both are the section's own text. A change or deletion is `@@ -L,n` and
    covers old lines `L .. L+n-1`; it is inside when every one of those lines
    is, and never when it includes the heading itself, so it starts at
    `start + 1`.
    """
    line, count = hunk
    if count == 0:
        return start <= line <= end
    return (start + 1) <= line and (line + count - 1) <= end


# ── The schema and the verdict block ─────────────────────────────────────────

ARCH_REVIEW_SCHEMA: dict = {
    "type": "object",
    "title": "arch_review_result",
    "properties": {
        "doc_status": {"type": "string", "enum": list(DOC_STATUSES),
                       "description": "Whether the prose still describes what runs."},
        "doc_updated": {"type": "boolean",
                        "description": "True if you edited the doc in this turn."},
        "grouping": {"type": "string", "enum": [*GROUPINGS, GROUPING_NONE],
                     "description": ("Group units only: does the grouping still hold. "
                                     f"`{GROUPING_NONE}` for a whole-doc unit.")},
        "summary": {"type": "string",
                    "description": "One or two sentences: what you checked and what you found."},
        "filed": {"type": "array", "items": {"type": "integer"},
                  "description": "Backlog ids filed during this review."},
        "appended_to": {"type": "array", "items": {"type": "integer"},
                        "description": "Existing item ids you appended a finding to."},
    },
    "required": ["doc_status", "doc_updated", "grouping", "summary", "filed", "appended_to"],
    "additionalProperties": False,
}

_FIELD_RE = re.compile(
    r"^(DOC_STATUS|DOC_UPDATED|GROUPING|SUMMARY|FILED|APPENDED_TO):\s*(.*)$", re.I)
_ID_RE = re.compile(r"#?\s*(\d+)")


def _ids(text: str) -> list[int]:
    t = (text or "").strip()
    if not t or t.lower().startswith("none"):
        return []
    seen: list[int] = []
    for m in _ID_RE.finditer(t):
        v = int(m.group(1))
        if v not in seen:
            seen.append(v)
    return seen


def _parse_block(text: str) -> Optional[dict]:
    """The trailing DOC_STATUS block, read from the LAST one onward.

    Last-block-wins, like `youtube_digest.parse_result` and
    `deep_research.parse_result`: a model that states an outcome, reconsiders
    and restates would otherwise have its first verdict paired with its last
    evidence.
    """
    lines = (text or "")[-8000:].splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("DOC_STATUS:"):
            start = i
    if start is None:
        return None
    fields: dict[str, list[str]] = {}
    current = None
    for line in lines[start:]:
        m = _FIELD_RE.match(line.strip())
        if m:
            current = m.group(1).upper()
            fields[current] = [m.group(2)]
        elif current:
            fields[current].append(line)

    def one(key: str) -> str:
        return " ".join(" ".join(fields.get(key, [])).split()).strip().strip("`'\"")

    return {
        "doc_status": one("DOC_STATUS").lower(),
        "doc_updated": one("DOC_UPDATED").lower().startswith(("y", "t", "1")),
        "grouping": one("GROUPING").lower(),
        "summary": one("SUMMARY"),
        "filed": _ids(one("FILED")),
        "appended_to": _ids(one("APPENDED_TO")),
    }


def parse_result(text: str, structured: Any, kind: str) -> Optional[dict]:
    """One verdict from the finalizer object if it is usable, else the block.

    `source` is recorded for the same reason `autotriage.parse_verdict` records
    it: a finalizer that quietly stopped working looks exactly like one that is
    working, and the only tell is the fallback rate.
    """
    raw, source = None, ""
    if isinstance(structured, dict) and str(structured.get("doc_status") or "").lower() in DOC_STATUSES:
        raw = {
            "doc_status": str(structured.get("doc_status") or "").lower(),
            "doc_updated": bool(structured.get("doc_updated")),
            "grouping": str(structured.get("grouping") or "").lower(),
            "summary": str(structured.get("summary") or ""),
            "filed": [int(i) for i in (structured.get("filed") or []) if str(i).lstrip("-").isdigit()],
            "appended_to": [int(i) for i in (structured.get("appended_to") or [])
                            if str(i).lstrip("-").isdigit()],
        }
        source = "structured"
    else:
        raw = _parse_block(text)
        source = "regex"
    if raw is None:
        return None

    status = raw["doc_status"] if raw["doc_status"] in DOC_STATUSES else None
    if status is None:
        return None
    grouping = raw["grouping"] if raw["grouping"] in GROUPINGS else None
    if kind == "group":
        # A section cannot be superseded or aspirational on its own — the doc
        # it lives in holds that verdict — so those clamp to `stale`, which is
        # what a section describing something gone actually is.
        if status in ("superseded", "aspirational"):
            status = "stale"
        if grouping is None:
            grouping = "holds"
    else:
        grouping = None
    return {"doc_status": status, "doc_updated": bool(raw["doc_updated"]),
            "grouping": grouping, "summary": raw["summary"][:600],
            "filed": raw["filed"], "appended_to": raw["appended_to"],
            "source": source}


# ── Already-filed items ──────────────────────────────────────────────────────


def _provenance_re(slug: str, name: str = "") -> "re.Pattern[str]":
    """A whole-line match for this unit's provenance, never a substring.

    `provenance_line(slug, "")` is a strict **prefix** of every group's line
    for the same doc — `…/workers-jobs.md` against
    `…/workers-jobs.md §Dispatch` — so a substring test made the doc unit
    match every one of its groups' findings: `<already_filed>` would show a
    group's items to the doc review, and `_filed_item_exists` would accept a
    group's item as proof of a doc unit's claim.
    """
    return re.compile(rf"^{re.escape(provenance_line(slug, name))}\s*$", re.M)


def _read_head(path: Path, nbytes: int = 8000) -> str:
    try:
        with path.open("r", errors="ignore") as f:
            return f.read(nbytes)
    except OSError:
        return ""


def provenance_line(slug: str, name: str = "") -> str:
    """The first body line every filed finding carries.

    It joins an item back to the unit that found it, and it is what
    `_filed_item_exists` checks — a `FILED: #n` pointing at somebody else's
    item is not a filing.

    `name` is the unit's **configured** name, never the heading text it
    resolved to. Both jobs docs list their members in the heading, so the
    heading changes whenever a member does; keying on it would make every
    prior finding invisible to the next review of that group and re-file the
    lot. That is the one failure `<already_filed>` exists to prevent, so it
    must not be reintroduced by the line that identifies the unit.
    """
    return f"Found reviewing architecture/{slug}.md" + (f" §{name}" if name else "")


def _backlog_dir() -> Path:
    return VAULT_ROOT / "backlog"


def _item_head(path: Path) -> tuple[str, str]:
    """`(front matter, body head)` for one backlog file. `("", …)` when the
    file does not open with a front matter block."""
    head = _read_head(path)
    if not head.startswith("---"):
        return "", head
    end = head.find("\n---", 3)
    return (head[:end], head[end:]) if end != -1 else (head, "")


def _item_status(fm: str) -> str:
    """An item's status, defaulting to `draft` when absent — which is what
    every other reader of the board already does."""
    m = re.search(r"^status:\s*(\S+)\s*$", fm, re.M)
    return m.group(1).strip().strip("'\"") if m else "draft"


def open_review_items(slug: str, name: str = "", backlog_dir: Path | None = None) -> list[dict]:
    """Open `arch-review` drafts already filed for this unit, oldest id first.

    Handed to the turn so a re-review appends to what the last one found
    instead of filing it again — the `prior_spawned` lesson from the implement
    loop, in the one place a 30-day cadence makes it certain to recur.
    """
    from app.backlog_status import OPEN_STATUSES
    d = backlog_dir or _backlog_dir()
    if not d.is_dir():
        return []
    want = _provenance_re(slug, name)
    out: list[dict] = []
    for f in sorted(d.glob("*.md")):
        m = re.match(r"^(\d+)[-_]", f.name)
        if not m:
            continue
        fm, body = _item_head(f)
        if not fm or REVIEW_TAG not in fm:
            continue
        if _item_status(fm) not in OPEN_STATUSES:
            continue
        if not want.search(body):
            continue
        t = re.search(r"^#\s+(.+)$", body, re.M)
        out.append({"id": int(m.group(1)), "title": (t.group(1).strip() if t else f.stem)[:120]})
    out.sort(key=lambda i: i["id"])
    return out


def open_review_item_count(backlog_dir: Path | None = None) -> int:
    """Every open `arch-review` draft, across both kinds — the backpressure
    gauge. Counted across units on purpose: every unit holding a handful of
    open findings is the board this pass could bury."""
    from app.backlog_status import OPEN_STATUSES
    d = backlog_dir or _backlog_dir()
    if not d.is_dir():
        return 0
    n = 0
    for f in sorted(d.glob("*.md")):
        if not re.match(r"^\d+[-_]", f.name):
            continue
        fm, _ = _item_head(f)
        if not fm or REVIEW_TAG not in fm:
            continue
        if _item_status(fm) in OPEN_STATUSES:
            n += 1
    return n


def _filed_item_exists(item_id: int, slug: str, name: str = "",
                       backlog_dir: Path | None = None) -> bool:
    """A `FILED: #n` claim holds only if that item is on disk, carries the
    review tag, and opens with this unit's provenance line."""
    d = backlog_dir or _backlog_dir()
    want = _provenance_re(slug, name)
    for f in d.glob(f"{int(item_id)}-*.md"):
        text = _read_head(f, 20000)
        return REVIEW_TAG in text and bool(want.search(text))
    return False


# ── The prompt ───────────────────────────────────────────────────────────────

PROMPT = """\
[SYSTEM: You are running the "arch-review" worker job. Work autonomously and \
do not ask for confirmation — nobody answers questions here.]

Review ONE unit of Lloyd's `architecture/` documentation against the tree it \
describes, then fix that one doc and file everything else.

Repository: {root} (this is PRODUCTION — a saved file is a deploy)
Head commit: {head}
Last reviewed: {last_reviewed}

{unit_block}
<already_filed>
{already_filed}
</already_filed>
{groups_config_block}{big_doc_hint}
Work in this order.

**1. Accuracy.** Check the unit's claims against the tree, not against memory. \
Every backticked path, line number, tool name, config key, count and cadence \
is a claim. Use `Read`, `Grep`, `Glob`, `graph_explain` and `graph_affected`; \
`git log --since='30 days ago' -- <paths>` for what has moved under it; and the \
live health routes for anything the doc states as a measured number:

    curl -s 'localhost:8080/api/workers/health?days=7'
    curl -s 'localhost:8080/api/autonomy/health?days=7'
    curl -s 'localhost:8080/api/autonomy/tasks'

Classify the unit as exactly one of:
  - `current` — the prose describes what runs;
  - `stale` — it described what ran, and the code has moved;
  - `superseded` — the thing it describes was replaced; name the replacement;
  - `aspirational` — it describes something that was never built.
{status_rule}

**2. Review.** Accuracy is the floor; the point is the architecture.
{review_lenses}
For each finding: the path and line, why it is wrong or risky, and what a fix \
would look like. Be specific enough that a fresh session could act on it \
alone.

**3. File what you found** with `backlog_write_task`: board `lloyd`, status \
`draft`, tags `{review_tag}`, `{spawn_tag}`, `{slug}`{group_tag_hint} plus the \
area tags. **The first line of the description must be exactly:**

    {provenance}

One finding per item, at most {spawn_cap} items. Do not write acceptance \
clauses — a later triage pass writes those. A fix that belongs in a vault \
skill, an autonomy task file or any file other than the one doc below is \
**FILED, never made**. If an `<already_filed>` item above already covers the \
finding, append to that item instead of filing a new one and report its id \
under APPENDED_TO. If `backlog_write_task` answers `merged_into: #n`, that \
counts as filed under #n. Never file "X is not built" against a doc you \
classified `aspirational` — that is what the classification is for.

**4. Fix the doc.** Edit **only** `{doc_path}`{section_rule}. Rules:
  - Keep the front matter block intact and refresh its date field \
(`date:` or `updated:`, whichever the doc uses) to {today}.
  - `current` or `stale`: correct what is wrong, in place. Prefer the smallest \
true edit — this is a correction pass, not a rewrite.
  - `superseded` or `aspirational`: do **not** rewrite the body. Add one \
paragraph under the H1 saying what replaced it (or that it was never built) \
and file a `needs-human` item titled "retire architecture/{slug}.md → \
.archive/".
{doc_log_rule}
  - Never run `git commit`, `git add`, `git checkout`, `git restore` or \
`git stash`. This job's runner commits the doc for you; a git write from here \
will be reverted and recorded as a defect.

Your diff is bounded and the bound is enforced after the turn: more than \
{max_delta_lines} changed lines, more than {max_shrink_pct}% of the doc \
deleted, a broken front matter block, or {section_bound_hint} — and the whole \
doc edit is thrown away, while your filed items survive. Stay well inside it.

Every file you write other than that one doc is reverted after the turn — \
anywhere in this repository, and anywhere in the vault except `backlog/`, \
where your filings belong. A file that was already modified before your turn \
is not reverted (somebody else is mid-edit) but any change you make to it is \
reported. You cannot write memory or facts: those tools are not available to \
you, because a document review has no business changing what Lloyd believes.

**5. End your final message with exactly this block and nothing after it:**

DOC_STATUS: <{doc_statuses}>
DOC_UPDATED: <yes|no>
GROUPING: <{grouping_values}>
SUMMARY: <one or two sentences: what you checked, what you found>
FILED: <#id, #id, ... or none>
APPENDED_TO: <#id, #id, ... or none>
"""

_DOC_UNIT = """\
<unit kind="doc">
  doc: architecture/{slug}.md ({lines} lines)
  This unit is the whole document.
</unit>"""

_GROUP_UNIT = """\
<unit kind="group">
  doc: architecture/{doc}.md
  section: "{heading}" — lines {start}-{end} of that file
  This unit is ONE functional group of the unattended fleet. Its members are
  the ids and source names named in that heading and in the table above it;
  read them from the doc. Everything outside lines {start}-{end} is another
  unit's and must not be edited.
</unit>"""

_DOC_LENSES = """\
Read the files the doc names and what `graph_affected` reaches from them. \
Look for: a described mechanism that no longer exists; a consumer the doc \
promises that nothing calls; a second definition of something the doc says has \
one; an invariant the code no longer holds; a kill switch that is not wired; \
a measured table that has drifted."""

_GROUP_LENSES = """\
Three lenses, in this order:
  (a) **Architecture** — does this grouping still hold? Answer GROUPING with \
`holds`, `split` (one function has become two), `merge` (it is really the same \
function as another group) or `move` (a member belongs in another group), and \
say why in the summary. Then: who has write authority here and is it bounded; \
which consumer the group writes for does not exist; which edges cross into \
another group; which defect several members share.
  (b) **Code** — the worker source modules, their tests and their imports. For \
an autonomy group, the SKILL.md bodies the tasks name: phantom tool names, \
unbounded steps, a step whose own timeout exceeds the task's budget, a report \
that ends in a question nobody answers.
  (c) **Drift** — where the section's stated invariants and what its members \
actually do have come apart."""

_DOC_LOG_RULE = """\
  - Append one dated line to a `## Review log` section at the end of the doc \
(create the heading if it is not there): the date, the status you assigned, \
and one clause on what changed."""
_GROUP_LOG_RULE = """\
  - Refresh the measured numbers in this section against what you just read \
from the health routes, and date them inline. Do not add a review-log section \
to a shared doc — each group owns only its own lines."""

_STATUS_RULE_DOC = ""
_STATUS_RULE_GROUP = (
    "A section is `current` or `stale` only; a whole document is what gets "
    "retired, not one of its functional groups.")

_BIG_DOC_HINT = """\
This doc is {lines} lines — read it in sections with Read's offset/limit \
rather than in one call, and spend your budget on the paths, numbers and \
config keys rather than on the prose.
"""


def build_prompt(unit: dict, *, head: str, last_reviewed: str, doc_lines: int,
                 section: tuple[int, int] | None, heading: str, already_filed: list[dict],
                 groups_config: list[str] | None, spawn_cap: int, max_delta_lines: int,
                 max_shrink_pct: int, today: str, root: Path) -> str:
    kind, slug = unit["kind"], unit["doc"]
    is_group = kind == "group"
    if is_group and section:
        unit_block = _GROUP_UNIT.format(doc=slug, heading=heading,
                                        start=section[0], end=section[1])
    else:
        unit_block = _DOC_UNIT.format(slug=slug, lines=doc_lines)
    filed_lines = ("\n".join(f"  - #{i['id']} {i['title']}" for i in already_filed)
                   or "  none yet for this unit")
    groups_block = ""
    if groups_config:
        groups_block = (
            "<groups_config>\n"
            "The functional groups this job is configured to review in this doc:\n"
            + "\n".join(f"  - {g}" for g in groups_config)
            + "\nIf that list no longer matches the doc's own tables — a group renamed, "
              "added or dropped — that is a finding: file it.\n</groups_config>\n")
    return PROMPT.format(
        root=str(root),
        head=head or "unknown",
        last_reviewed=last_reviewed or "never",
        unit_block=unit_block,
        already_filed=filed_lines,
        groups_config_block=groups_block,
        big_doc_hint=(_BIG_DOC_HINT.format(lines=doc_lines) if doc_lines > BIG_DOC_LINES else ""),
        status_rule=(_STATUS_RULE_GROUP if is_group else _STATUS_RULE_DOC),
        review_lenses=(_GROUP_LENSES if is_group else _DOC_LENSES),
        review_tag=REVIEW_TAG, spawn_tag=SPAWN_TAG, slug=slug,
        group_tag_hint=(f", `group-{unit['name'].casefold().replace(' ', '-')}`" if is_group else ""),
        provenance=provenance_line(slug, unit["name"] if is_group else ""),
        spawn_cap=spawn_cap,
        doc_path=f"{root}/{ARCH_DIRNAME}/{slug}.md",
        section_rule=(f', and only inside lines {section[0] + 1}-{section[1]} of it — '
                      f'never the "{heading}" heading line itself, never the tables above '
                      f'it, never another section' if is_group and section else ""),
        today=today,
        doc_log_rule=(_GROUP_LOG_RULE if is_group else _DOC_LOG_RULE),
        max_delta_lines=max_delta_lines, max_shrink_pct=max_shrink_pct,
        section_bound_hint=("any hunk outside your section" if is_group
                            else "a destroyed front matter block"),
        doc_statuses="|".join(DOC_STATUSES),
        grouping_values=("|".join(GROUPINGS) if is_group else GROUPING_NONE),
    )


FINAL_SCHEMA_PROMPT = (
    "Restate the block above as a single JSON object matching the schema. "
    "Same status, same ids — this is a transcription, not a re-decision. "
    f"`grouping` is `{GROUPING_NONE}` unless this unit was a group.")


# ── Stray writes and the doc bound ───────────────────────────────────────────


def _under(repo: Path, rel: str) -> Optional[Path]:
    """`repo/rel`, but only when it really is under `repo`. A path that escapes
    (`..`, an absolute spelling, a symlinked parent) is not this turn's to
    delete, and unlinking one would be the damage this function exists to
    undo."""
    try:
        p = (repo / rel).resolve()
        p.relative_to(repo.resolve())
    except (OSError, ValueError):
        return None
    return p


def revert_strays(repo: Path, before: set[str], after: set[str],
                  keep: Iterable[str] = ()) -> list[dict]:
    """Undo every path this turn changed except the ones named in `keep`.

    A diff against a baseline, never a snapshot: neither tree is ever clean —
    the autonomy scheduler rewrites a task file on every run, and a human may
    have an editor open in `~/lloyd` — so a snapshot would revert other
    people's work. A tracked path goes back to HEAD; an untracked one is
    unlinked. The baselines are taken with `-uall` so an untracked *directory*
    is never one entry to delete wholesale.
    """
    kept = set(keep)
    out: list[dict] = []
    for rel in sorted(after - before):
        if rel in kept:
            continue
        if _is_tracked(repo, rel):
            _git(repo, "checkout", "--", rel)
            action = "checkout"
        else:
            p = _under(repo, rel)
            if p is None:
                out.append({"repo": repo.name, "path": rel, "action": "skipped (outside repo)"})
                continue
            try:
                p.unlink()
                action = "unlink"
            except OSError as exc:
                action = f"unlink failed: {exc}"
        out.append({"repo": repo.name, "path": rel, "action": action})
    return out


def check_doc_bound(root: Path, rel: str, *, max_delta_lines: int, max_shrink_pct: int,
                    allow_shrink: bool, section: tuple[int, int] | None) -> tuple[bool, str]:
    """`(changed, reason)` — `reason` non-empty means throw the edit away.

    Four bounds, and each one is a different way a review could do harm:
    a diff far larger than a correction pass (a rewrite); a large deletion
    (the doc gutted rather than corrected — allowed for `superseded` and
    `aspirational`, where a banner over a body nobody should trust is the
    point); a destroyed front matter block (the vault indexes on it); and, for
    a group, a hunk outside its own section (seven groups share one file and
    must not re-open each other's text).
    """
    added, deleted = _numstat(root, rel)
    if added == 0 and deleted == 0:
        return False, ""
    total = added + deleted
    if total > int(max_delta_lines):
        return True, f"diff is {total} changed lines, cap {max_delta_lines}"
    # The denominator is the UNIT, not the file. A group's section is a small
    # fraction of a jobs doc — 43 lines of 824 — so measuring its deletions
    # against the whole file lets it delete itself entirely and score 5%. The
    # section rail does not catch that either: a hunk that removes the whole
    # section is, by construction, inside the section.
    base = (section[1] - section[0] + 1) if section else _head_line_count(root, rel)
    if base and not allow_shrink:
        pct = deleted * 100.0 / base
        if pct > float(max_shrink_pct):
            what = "its section" if section else "the doc"
            return True, (f"deleted {deleted} of {base} lines of {what} ({pct:.0f}%), "
                          f"cap {max_shrink_pct}%")
    was_fm = _git(root, "show", f"HEAD:{rel}").startswith("---")
    if was_fm:
        try:
            now_fm = (root / rel).read_text(encoding="utf-8", errors="replace").startswith("---")
        except OSError:
            now_fm = False
        if not now_fm:
            return True, "front matter block no longer opens the file"
    if section:
        for hunk in changed_hunks(root, rel):
            if not hunk_inside(hunk, section[0], section[1]):
                return True, (f"hunk @@ -{hunk[0]},{hunk[1]} falls outside section "
                              f"lines {section[0]}-{section[1]}")
    return True, ""


# ── Committing ───────────────────────────────────────────────────────────────


def _git_ok(repo: Path, *args: str) -> tuple[bool, str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True, timeout=_GIT_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return r.returncode == 0, (r.stderr or r.stdout).strip()[:300]


def commit_doc(root: Path, rel: str, message: str) -> tuple[bool, str]:
    """Commit exactly one doc to `main`, or say why not.

    Two things hold it off. A **landing drain** means the promoter is idling
    the backend to restart it, and a commit into that window moves HEAD under
    a round that is mid-merge. The **automod lock** means a round, a promotion
    or a rollback owns the tree. Neither is an error: the doc stays dirty and
    the next tick retries, and the loop tolerates dirt disjoint from a round's
    own diff, so a waiting doc stalls nothing.

    The commit carries a pathspec and no `git add`, so it commits this file's
    working-tree content and touches nothing else — not another file, and not
    a human's partially staged index.
    """
    from scripts.automod import state as S
    try:
        from app.routers.automod import drain_active
    except Exception:  # noqa: BLE001 — no backend here means no drain to respect
        drain_active = lambda: False  # noqa: E731
    if drain_active():
        return False, "a landing is draining the backend"
    try:
        with S.Lock(owner=f"{NAME} commit"):
            if rel not in _porcelain(root, rel):
                return False, "nothing to commit"
            ok, err = _git_ok(root, "commit", "-q", "-m", message, "--", rel)
            if not ok:
                return False, f"git commit failed: {err}"
            sha = _git(root, "rev-parse", "HEAD").strip()
            return True, sha
    except S.LockHeld as exc:
        return False, str(exc)


def _commit_message(unit: str, doc_status: str) -> str:
    return f"arch-review: {unit} — {doc_status}"


def commit_pending(state: dict, root: Path | None = None) -> list[dict]:
    """Retry the commits a drain or the lock held off last tick.

    Deliberately at the *top* of `enqueue_if_due` rather than on a timer: the
    tick is the only thing that runs regularly, and a doc waiting to be
    committed is the one piece of this job's output that is not yet durable.
    """
    root = root or repo_root()
    done: list[dict] = []
    for unit, row in list(state.items()):
        pending = (row or {}).get("pending_commit")
        if not isinstance(pending, dict):
            continue
        rel = str(pending.get("rel") or "")
        if not rel:
            row.pop("pending_commit", None)
            continue
        ok, detail = commit_doc(root, rel, str(pending.get("message") or
                                               _commit_message(unit, "current")))
        if ok:
            row["pending_commit"] = None
            row.pop("pending_commit", None)
            row["reviewed_commit"] = detail
            done.append({"unit": unit, "commit": detail, "doc": rel})
            logger.info("arch-review: committed deferred %s as %s", rel, detail[:12])
        elif detail == "nothing to commit":
            # Somebody else committed or reverted it; nothing is owed.
            row.pop("pending_commit", None)
            done.append({"unit": unit, "commit": "", "doc": rel, "note": detail})
    return done


# ── Scheduling ───────────────────────────────────────────────────────────────


def _queued_for_source(queue: WorkQueue) -> int:
    try:
        depth = queue.depth_by_source().get(NAME, {}) or {}
    except Exception:  # noqa: BLE001 — a depth read must never stop enqueueing
        return 0
    return sum(int(v) for k, v in depth.items() if k in ("queued", "claimed"))


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    """Top up the queue with the units that have rested longest.

    Ordered by age and bounded three ways: `batch` is what may sit in the
    queue at once, `daily_max` is what may be reviewed in 24 h, and
    `max_open_items` stops the whole pass when its own findings are piling up
    faster than triage drains them. That last one is the R > 1 lesson the
    triage loop paid for: a pass that files faster than the board closes does
    not need better verdicts, it needs an edge cut.
    """
    now = time.time()
    state = await asyncio.to_thread(load_state)
    committed = await asyncio.to_thread(commit_pending, state)
    if committed:
        await asyncio.to_thread(save_state, state)

    batch = int(src_cfg.get("batch", DEFAULT_BATCH))
    room = batch - _queued_for_source(queue)
    if room <= 0:
        return

    max_open = int(src_cfg.get("max_open_items", DEFAULT_MAX_OPEN_ITEMS))
    open_now = await asyncio.to_thread(open_review_item_count)
    if open_now >= max_open:
        logger.info("arch-review: %d open review items (bound %d) — skipping this tick",
                    open_now, max_open)
        return

    daily_max = int(src_cfg.get("daily_max", DEFAULT_DAILY_MAX))
    done_today = await asyncio.to_thread(reviews_today, None)
    room = min(room, max(0, daily_max - done_today))
    if room <= 0:
        return

    units = await asyncio.to_thread(all_units, src_cfg)
    due = pending_units(
        units, state, now=now,
        interval_days=float(src_cfg.get("review_interval_days", DEFAULT_REVIEW_INTERVAL_DAYS)),
        retry_spacing=float(src_cfg.get("retry_spacing_seconds", DEFAULT_RETRY_SPACING_SECONDS)),
        max_attempts=int(src_cfg.get("max_attempts", DEFAULT_MAX_ATTEMPTS)))

    priority = int(src_cfg.get("priority", DEFAULT_PRIORITY))
    for unit in due[:room]:
        new_id = queue.enqueue(
            source=NAME, kind="unit",
            payload={"unit": unit["unit"], "kind": unit["kind"], "doc": unit["doc"],
                     "name": unit["name"],
                     "max_turns": int(src_cfg.get("max_turns", DEFAULT_MAX_TURNS)),
                     "spawn_cap": int(src_cfg.get("spawn_cap", DEFAULT_SPAWN_CAP)),
                     "max_delta_lines": int(src_cfg.get("max_delta_lines",
                                                        DEFAULT_MAX_DELTA_LINES)),
                     "max_shrink_pct": int(src_cfg.get("max_shrink_pct",
                                                       DEFAULT_MAX_SHRINK_PCT))},
            priority=priority,
            dedup_key=f"{NAME}:{unit['unit']}")
        if new_id is not None:
            logger.info("Enqueued arch-review %s", unit["unit"])


# ── Execution ────────────────────────────────────────────────────────────────


def _touch_state(unit: str, **fields) -> dict:
    """Read-modify-write one unit's row.

    `max_inflight: 1` and one tick at a time, both on the backend's single
    event loop, so there is no writer to race — the file is small and rewritten
    whole rather than locked.
    """
    state = load_state()
    row = dict(state.get(unit) or {})
    row.update(fields)
    state[unit] = row
    save_state(state)
    return row


def _count_attempt(unit: str, error: str) -> int:
    row = load_state().get(unit) or {}
    n = int(row.get("attempts") or 0) + 1
    _touch_state(unit, attempts=n, last_attempt_at=_now_iso(), last_error=error[:400])
    return n


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _prepare(unit: dict, root: Path) -> dict:
    """Everything the prompt needs, read off disk in one thread hop."""
    slug, kind, name = unit["doc"], unit["kind"], unit.get("name") or ""
    rel = f"{ARCH_DIRNAME}/{slug}.md"
    path = root / rel
    out: dict[str, Any] = {"rel": rel, "path": path, "kind": kind, "slug": slug,
                           "name": name, "heading": "", "section": None,
                           "head": _git(root, "rev-parse", "HEAD").strip()[:12], "error": ""}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        out["error"] = f"doc_missing: {exc}"
        return out
    out["lines"] = len(text.splitlines())
    if kind == "group":
        section = find_section(text, name)
        if section is None:
            out["error"] = f"section_missing: no '## {name}' heading in {rel}"
            return out
        out["section"] = section
        out["heading"] = re.sub(r"^##\s+", "", text.splitlines()[section[0] - 1]).strip()
    return out


async def execute(item: QueueItem) -> dict[str, Any]:
    from workers.sources import get_sources_config
    from scripts.automod import backlog as B, state as S

    payload = item.payload or {}
    unit_key = str(payload.get("unit") or "")
    kind = str(payload.get("kind") or "doc")
    slug = str(payload.get("doc") or "")
    name = str(payload.get("name") or "")
    if not unit_key or not slug:
        return {"status": "failed", "summary": "queue item carries no unit"}
    src_cfg = get_sources_config().get(NAME, {}) or {}
    root = repo_root()
    spawn_cap = int(payload.get("spawn_cap") or src_cfg.get("spawn_cap", DEFAULT_SPAWN_CAP))
    max_delta = int(payload.get("max_delta_lines")
                    or src_cfg.get("max_delta_lines", DEFAULT_MAX_DELTA_LINES))
    max_shrink = int(payload.get("max_shrink_pct")
                     or src_cfg.get("max_shrink_pct", DEFAULT_MAX_SHRINK_PCT))
    base = {"unit": unit_key, "kind": kind, "doc": slug, "name": name}

    # 1. Resolve the unit. A doc that is gone or a heading that has been
    #    renamed is the unit's own problem and counts an attempt — three of
    #    them park it until a human fixes the config list or the doc.
    prep = await asyncio.to_thread(_prepare, {"doc": slug, "kind": kind, "name": name}, root)
    if prep["error"]:
        n = await asyncio.to_thread(_count_attempt, unit_key, prep["error"])
        logger.warning("arch-review %s: %s (attempt %d)", unit_key, prep["error"], n)
        return {"status": "failed", "summary": f"{unit_key}: {prep['error']}"[:500],
                "meta": {**base, "attempts": n}}
    rel, section, heading = prep["rel"], prep["section"], prep["heading"]

    # 2. Baselines, in both trees. `-uall` so an untracked directory is never
    #    one entry that would be deleted wholesale on the way out.
    before_lloyd = await asyncio.to_thread(_porcelain, root)
    before_vault = await asyncio.to_thread(vault_dirty, VAULT_ROOT)
    # Content, not just presence: a path already dirty can never show up in
    # `after - before`, so without this the turn could rewrite one freely (#915).
    fp_lloyd = await asyncio.to_thread(fingerprints, root, before_lloyd)
    fp_vault = await asyncio.to_thread(fingerprints, VAULT_ROOT, before_vault)
    if rel in before_lloyd:
        # A human is editing this doc right now. Not the unit's fault and not
        # an attempt: parking a unit because somebody had the file open would
        # make the picklist stall on exactly the docs being worked on.
        return {"status": "skipped", "summary": f"{unit_key}: {rel} has uncommitted edits",
                "meta": {**base, "doc_dirty": True}}

    # 3. The spawn floor, before the turn — an id at or below it that the turn
    #    claims is a merge, not a filing (`B.split_claimed`).
    id_floor = await asyncio.to_thread(B.max_item_id)
    already = await asyncio.to_thread(open_review_items, slug, name if kind == "group" else "")
    groups_cfg = [f"{d}:{n}" for d, n in parse_groups(src_cfg.get("groups")) if d == slug]
    row = (await asyncio.to_thread(load_state)).get(unit_key) or {}
    prompt = build_prompt(
        {"kind": kind, "doc": slug, "name": name}, head=prep["head"],
        last_reviewed=str(row.get("last_reviewed_at") or ""), doc_lines=int(prep.get("lines") or 0),
        section=section, heading=heading, already_filed=already,
        groups_config=groups_cfg if kind == "doc" else None,
        spawn_cap=spawn_cap, max_delta_lines=max_delta, max_shrink_pct=max_shrink,
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"), root=root)

    # 4. The session.
    run: dict = {}
    timed_out = ""
    try:
        run = await run_prompt_in_session(
            prompt, title=f"arch-review: {unit_key}", source=NAME,
            max_turns=int(payload.get("max_turns") or src_cfg.get("max_turns", DEFAULT_MAX_TURNS)),
            priority=1,
            extra_disallowed=list(DISALLOWED),
            final_schema=ARCH_REVIEW_SCHEMA,
            final_schema_prompt=FINAL_SCHEMA_PROMPT)
    except DrainActive as exc:
        # A landing owns the backend. Nothing was written, so nothing to undo.
        return {"status": "skipped", "summary": f"landing in progress: {exc}"[:500],
                "meta": {**base, "drain_active": True}}
    except TurnTimeout as exc:
        # The turn is gone but its writes are not: the cleanup below still has
        # to run, which is why this does not return here.
        timed_out = str(exc)

    session_id = str(run.get("session_id") or "")
    text = str(run.get("text") or "")
    stop_reason = run.get("stop_reason")

    # 5. Stray writes — everything but the one doc, in both trees.
    after_lloyd = await asyncio.to_thread(_porcelain, root)
    after_vault = await asyncio.to_thread(vault_dirty, VAULT_ROOT)
    strays = await asyncio.to_thread(revert_strays, root, before_lloyd, after_lloyd, {rel})
    strays += await asyncio.to_thread(revert_strays, VAULT_ROOT, before_vault, after_vault)
    # Reported, never reverted — see `modified_preexisting`. The doc is exempt
    # in this repo: a human editing it sends the unit down the `doc_dirty`
    # path before the turn ever starts, so any change here is the turn's own.
    strays += await asyncio.to_thread(modified_preexisting, root, fp_lloyd, {rel})
    strays += await asyncio.to_thread(modified_preexisting, VAULT_ROOT, fp_vault)

    # 6. The doc's own diff, against the bounds. Parsed first, because whether
    #    a large deletion is allowed depends on the status the turn assigned.
    parsed = parse_result(text, run.get("structured"), kind)
    status = (parsed or {}).get("doc_status") or ""
    changed, reject = await asyncio.to_thread(
        check_doc_bound, root, rel,
        max_delta_lines=max_delta, max_shrink_pct=max_shrink,
        allow_shrink=status in ("superseded", "aspirational"),
        section=section if kind == "group" else None)
    if changed and reject:
        await asyncio.to_thread(_git, root, "checkout", "--", rel)
        changed = False
        logger.warning("arch-review %s: doc edit rejected — %s", unit_key, reject)

    # 7. Infra-shaped: the harness never got a completion. Not the unit's
    #    fault, so it costs no attempt and the unit stays due.
    if not timed_out and not text.strip() and stop_reason is None:
        errs = "; ".join(str(e)[:160] for e in (run.get("errors") or [])[:2])
        why = f"turn produced nothing ({errs or 'no error reported'})"
        logger.warning("arch-review %s: %s — leaving the unit due", unit_key, why)
        return {"status": "failed", "summary": f"{unit_key}: {why}"[:500],
                "meta": {**base, "session_id": session_id, "infra": True,
                         "stray_writes": strays}}

    # 8. Verify what it claims it filed.
    filed_claim = list((parsed or {}).get("filed") or [])
    spawned, merged = await asyncio.to_thread(
        B.split_claimed, filed_claim, id_floor=id_floor, self_id=0)
    # `split_claimed` drops an id with no file on disk — correct for the
    # spawn/merge split, and wrong as the last word on a `FILED:` claim: an id
    # the model invented is the one thing `filed_unverified` most needs to say.
    missing = [i for i in filed_claim if i not in spawned and i not in merged]
    filed, unverified = [], list(missing)
    for i in spawned:
        ok = await asyncio.to_thread(_filed_item_exists, i, slug,
                                     name if kind == "group" else "")
        (filed if ok else unverified).append(i)
    unverified.sort()
    appended = list((parsed or {}).get("appended_to") or [])
    over_cap = max(0, len(filed) - spawn_cap)

    # 9. Commit the one doc.
    commit_sha, commit_note = "", ""
    if changed:
        message = _commit_message(unit_key, status or "reviewed")
        ok, detail = await asyncio.to_thread(commit_doc, root, rel, message)
        if ok:
            commit_sha = detail
        else:
            commit_note = detail
            await asyncio.to_thread(
                _touch_state, unit_key, pending_commit={"rel": rel, "message": message})
            logger.info("arch-review %s: leaving %s dirty for the next tick (%s)",
                        unit_key, rel, detail)

    # 10. The record. A timeout is a real attempt: the turn had its budget.
    now_iso = _now_iso()
    event = {"event": "arch_review", "unit": unit_key, "kind": kind, "doc": slug,
             "name": name, "verdict": status or None,
             "grouping": (parsed or {}).get("grouping"),
             "doc_updated": bool(changed), "doc_update_rejected": reject or "",
             "commit": commit_sha, "commit_deferred": commit_note,
             "filed": filed, "filed_unverified": unverified, "merged": merged,
             "appended_to": appended, "spawn_cap": spawn_cap,
             "spawned_over_cap": over_cap, "id_floor": id_floor,
             "stray_writes": strays, "session_id": session_id,
             "stop_reason": stop_reason, "num_turns": run.get("num_turns"),
             "verdict_source": (parsed or {}).get("source") or "",
             "structured_error": str(run.get("structured_error") or ""),
             "finalizer_tokens": run.get("finalizer_tokens"),
             "timed_out": bool(timed_out)}
    try:
        await asyncio.to_thread(S.append_event, event)
    except Exception as exc:  # noqa: BLE001 — the ledger is not the work
        logger.warning("arch-review %s: could not append the ledger event: %s", unit_key, exc)

    if timed_out:
        n = await asyncio.to_thread(_count_attempt, unit_key, timed_out)
        return {"status": "failed", "summary": f"{unit_key}: {timed_out}"[:500],
                "response": text,
                "meta": {**base, "session_id": session_id, "turn_timeout": True,
                         "attempts": n, "stray_writes": strays, "filed": filed}}

    await asyncio.to_thread(
        _touch_state, unit_key,
        last_reviewed_at=now_iso, reviewed_commit=(commit_sha or prep["head"]),
        verdict=status or None, grouping=(parsed or {}).get("grouping"),
        filed=filed, attempts=0, last_attempt_at=now_iso, last_error="",
        section_missing=False)

    bits = [f"{unit_key}: {status or 'no verdict'}"]
    if (parsed or {}).get("grouping"):
        bits.append(f"grouping {parsed['grouping']}")
    if commit_sha:
        bits.append(f"doc committed {commit_sha[:12]}")
    elif reject:
        bits.append(f"doc edit rejected ({reject})")
    elif commit_note:
        bits.append(f"doc left dirty ({commit_note})")
    if filed:
        bits.append("filed " + ", ".join(f"#{i}" for i in filed))
    if merged:
        bits.append("merged into " + ", ".join(f"#{i}" for i in merged))
    if appended:
        bits.append("appended to " + ", ".join(f"#{i}" for i in appended))
    if unverified:
        bits.append(f"{len(unverified)} unverified id(s)")
    if strays:
        bits.append(f"{len(strays)} stray write(s) reverted")
    return {
        "status": "success",
        "summary": ", ".join(bits)[:500],
        "artifact_path": str(root / rel),
        "response": text,
        "meta": {**base, "session_id": session_id, "verdict": status,
                 "grouping": (parsed or {}).get("grouping"),
                 "doc_updated": bool(changed), "doc_update_rejected": reject or "",
                 "commit": commit_sha, "commit_deferred": commit_note,
                 "filed": filed, "filed_unverified": unverified, "merged": merged,
                 "appended_to": appended, "stray_writes": strays,
                 "stop_reason": stop_reason, "num_turns": run.get("num_turns"),
                 "verdict_source": (parsed or {}).get("source") or "",
                 "parsed": parsed is not None},
    }
