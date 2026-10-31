#!/usr/bin/env python3
"""The membership tables of `architecture/autonomy-jobs.md`, checked against the fleet.

Why this file exists as a module rather than as a test helper: the doc claims a job's
schedule lives in its task file and its *membership* lives in two hand-typed tables — §The
functions' `| Function | Jobs | What it is for |` and the write-authority tier table
`| Tier | Jobs | |`. Four automod rounds (#1102, #2630 with #2631, #2813, #2881) have been
spent repairing the disagreement between those tables and the task-file directory, and
every one of them was opened **after** a suite went red. The reason the class recurs is a
selection detail: the only two nodes that read both sides carry `@live_vault`, and the
gate's mark expression (`scripts/automod/gate.py`, `TESTS_MARK_EXPR = "not live_vault and
not fault_injection"`) deselects them — deliberately, because a nightly that files a task
must not redden an unrelated author's round. So the first thing in the loop that ever reads
both sides is the promotion gate's own suite run, hours after the file landed.

This moves the check to the moment the fact is knowable: a writer has just put `NN-slug.md`
on disk, the id it used is either in each table or it is not, and the doc is one read away.
It reports that and appends one row so the drift has a start date. **It never refuses
anything** — a guard that withheld a task file would be a worse failure than the drift, and
the writer is not the doc's owner.

Two rules carried over from `tests/test_autonomy_jobs_doc_claims.py`, which now delegates
here rather than owning them:

* The **directory is the source of truth**, because the directory is what dispatches: the
  scheduler lists `~/obsidian/autonomy/*.md` and nothing else. A table row with no file
  behind it cannot run; a file with no row runs unseen.
* The retired exception is **read off the row**, never typed into a list. `#36`'s group row
  says `| #36 | … Retired 2026-09-23 …` and its file sits in `_archived/`. An allowlist here
  would be the same hand-maintained roster this check exists to replace, and a ghost could
  buy its way back by editing a list in code instead of writing a task file.

Run by a human or a nightly, and by the two task-file writers (`app/routers/autonomy.py`,
`agent_mcp/autonomy.py`). Nothing under `scripts/automod/` imports it for a verdict: a doc
behind by a day has never been a reason to stop a landing, and reading the drift journal
from a promotion or rollback path would make documentation gate execution.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import sys
# Run by path (`python …/scripts/autonomy/jobs_membership.py`), which puts this file's own directory on
# `sys.path[0]` and not the checkout: without the insertion below, `app.data_root`
# raises `ModuleNotFoundError` and the reader silently never runs. Same reason as
# `scripts/skill_lint.py`'s insertion. (#3403)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from app.data_root import vault_root  # noqa: E402

#: The doc whose tables are reconciled. Two levels above `scripts/autonomy/`, so it
#: resolves to the tree that is *running* — the live checkout in production, a round's
#: worktree inside one. Same rule as `validate_tasks.py`'s `DEFAULT_CONFIG`.
JOBS_DOC = Path(__file__).resolve().parents[2] / "architecture" / "autonomy-jobs.md"

#: The directory the scheduler dispatches from, mirrored from `agent_mcp.autonomy`'s
#: `AUTONOMY_DIR`. A writer passes its own module attribute instead of this constant, so a
#: caller that redirects its dir is honoured rather than a second hard-coded path.
DEFAULT_TASK_DIR = vault_root() / "autonomy"

#: The two headers the membership tables sit under, paired with the name a report calls
#: each by. Spelled exactly as the doc writes them, because `lines.index(header)` is how a
#: table is located: a reworded header raises `UnreadableDoc` loudly rather than parsing an
#: empty table and reporting a clean bill for a doc nobody can read.
FUNCTIONS_HEADER = "| Function | Jobs | What it is for |"
TIERS_HEADER = "| Tier | Jobs | |"
#: The two tables, as a report names them. The second element of each pair is the header
#: line `table_rows` locates the table by, spelled exactly as the doc writes it: a reworded
#: header raises `UnreadableDoc` loudly rather than parsing an empty table and reporting a
#: clean bill for a doc nobody can read.
FUNCTIONS_TABLE = "the functions table (`## The functions`)"
TIERS_TABLE = "the write-authority tier table"
TABLES: tuple[tuple[str, str], ...] = (
    (FUNCTIONS_TABLE, FUNCTIONS_HEADER),
    (TIERS_TABLE, TIERS_HEADER),
)

#: `NN-slug.md` — the scheduler's own convention (`validate_tasks.py` mirrors it), and the
#: only thing that makes a file a fleet member. `glob` does not recurse, so `_archived/`
#: contributes nothing, which is correct: an archived task is not a fleet member.
TASK_FILE_NAME = re.compile(r"(\d+)-.*\.md$")

#: A membership row that discharges its own id: `| #36 | … Retired 2026-09-23 …`.
#: Case-sensitive on purpose — *retired* and *archived* recur in prose narrating other
#: jobs' histories, and a row that merely mentions a retirement is not that row's own.
RETIRED_MEMBERSHIP_ROW = re.compile(r"^\| #(\d+) \|.*\b(?:Retired|Archived)\b", re.M)


class UnreadableDoc(ValueError):
    """The doc text handed in holds neither table header.

    Raised rather than returning empty lists: an empty parse is indistinguishable from a doc
    that is perfectly in sync, and every check in here would then report clean. The writers
    catch it and mark their report `unchecked`, which is never read as a pass.
    """


class TaskFilePath(Path):
    """The task file a writer just wrote, carrying that write's membership `report`.

    A `Path` subclass rather than a tuple or a dataclass because both writers already return
    a `Path` and their callers use it as one — `tests/test_autonomy_failure_alert.py` reads
    `path.read_text(...)` off the MCP writer's return and hands the same object to
    `_parse_task_file`. Changing the returned type would have withheld a capability that the
    report is supposed to travel beside. `report` is `None` on an update: the tables were not
    asked about, because the id was already on the fleet.
    """

    report: dict | None = None

    def __hash__(self) -> int:                    # Path defines __eq__, so hash follows it
        return Path.__hash__(self)


def reported(path: Path, report: dict | None) -> TaskFilePath:
    """`path` as a `TaskFilePath` with `report` attached."""
    out = TaskFilePath(path)
    out.report = report
    return out


def job_ids(text: str) -> list[str]:
    """The `#NN` job ids in a table cell or a stretch of prose, in the order written."""
    return [f"#{n}" for n in re.findall(r"#(\d+)", text)]


def as_id(task_id: object) -> str:
    """`102`, `"102"`, `102` and `"#102"` all name job 102."""
    s = str(task_id).strip()
    return s if s.startswith("#") else f"#{s}"


def table_rows(doc_text: str, header: str) -> dict[str, tuple[list[str], str | None]]:
    """Rows under `header`: `{first cell: (job ids, trailing count cell or None)}`."""
    lines = doc_text.splitlines()
    try:
        i = lines.index(header)
    except ValueError as exc:
        raise UnreadableDoc(
            f"no {header!r} header in the doc text, so its membership cannot be read"
        ) from exc
    out: dict[str, tuple[list[str], str | None]] = {}
    for line in lines[i + 2:]:                        # header, then the |---|---| rule
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        count = cells[-1] if re.fullmatch(r"\d+", cells[-1]) else None
        out[cells[0]] = (job_ids(cells[1]), count)
    return out


def named_ids(doc_text: str, header: str) -> list[str]:
    """Every id one membership table names, flattened, duplicates kept in order."""
    return [j for ids, _ in table_rows(doc_text, header).values() for j in ids]


def task_file_ids(task_dir: Path) -> set[str]:
    """The fleet as ids, read off `task_dir`'s filenames: `{#24, …, #97}`."""
    return {f"#{m.group(1)}"
            for m in (TASK_FILE_NAME.match(p.name) for p in Path(task_dir).glob("*.md"))
            if m}


def retired_named_ids(doc_text: str) -> set[str]:
    """Ids the doc discharges by its own row: `{#36}` in today's document."""
    return {f"#{n}" for n in RETIRED_MEMBERSHIP_ROW.findall(doc_text)}


def _by_num(ids) -> list[str]:
    return sorted(set(ids), key=lambda j: int(j[1:]))


def table_drift(named: list[str], live: set[str],
                retired: set[str]) -> tuple[list[str], list[str], list[str]]:
    """`(ghosts, unlisted, duplicated)` for ONE table's ids against `live`.

    Three ways a hand-typed roster lies about the fleet, kept as three lists so a message
    can say which one happened:

    - **ghost** — the table names an id with no live task file behind it, and that id's own
      membership row does not say it is retired. (#87 #88 #89, 2026-10-12.)
    - **unlisted** — a live task file whose id the table never names. (#91–#97, which ran
      for two weeks while the doc claimed its membership was complete.)
    - **duplicate** — the id appears twice in the table, so the roster claims the job holds
      the group twice.
    """
    named_set = set(named)
    return (_by_num(named_set - live - retired),
            _by_num(live - named_set),
            _by_num(j for j in named if named.count(j) > 1))


def membership_drift(doc_text: str,
                     task_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """`(ghosts, unlisted, duplicated)` for the whole doc against the dispatch directory.

    Aggregated over both membership tables by `table_drift`: an id is a ghost if EITHER
    table names it with no file behind it, unlisted if EITHER table omits a live file, and
    duplicated if it is listed twice within any one table. This is the doc-level question —
    "does `architecture/autonomy-jobs.md` describe the fleet that dispatches?" — and the
    per-table answer stays available as `table_drift`, which is what
    `tests/test_autonomy_jobs_doc_claims.py`'s `_membership_drift` delegates to so the
    nightly reconciler, the two writers and `validate_tasks.py` cannot each hold a version
    of the rule.
    """
    live = task_file_ids(task_dir)
    retired = retired_named_ids(doc_text)
    ghosts: list[str] = []
    unlisted: list[str] = []
    duplicated: list[str] = []
    for _, header in TABLES:
        g, u, d = table_drift(named_ids(doc_text, header), live, retired)
        ghosts += g
        unlisted += u
        duplicated += d
    return _by_num(ghosts), _by_num(unlisted), _by_num(duplicated)


def missing_tables(doc_text: str, task_id: object) -> list[str]:
    """Which membership tables do NOT name `task_id`, by the name a report calls them by.

    Empty means the id is in both, which is the state every filing since the class was found
    has been missing. Raises `UnreadableDoc` if the text holds no table to check.
    """
    tid = as_id(task_id)
    return [name for name, header in TABLES if tid not in named_ids(doc_text, header)]


def doc_short_sha(text: str) -> str:
    """First 12 hex of the doc's sha256 — which revision the report was made against.

    A report is a statement about a hand-typed file, and the file moves under it: #2881
    edited both tables by hand the same morning it was filed. Without the sha a later reader
    cannot tell "the table was missing the id" from "the table has named it since Tuesday".
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def journal_path() -> Path:
    """`~/lloyd-data/_pipeline/arch-review/membership-drift.jsonl`, resolved when called.

    Off `app.paths.PIPELINE_DIR`, the 2026-09-22 rule for runtime state: it hangs off
    `DATA_ROOT`, so a round's worktree writes into its own gitignored `.lloyd-data` instead of
    into a commit, and a `rm -r` aimed at a checkout cannot reach it. Same directory as
    `AUTONOMY_DESCRIPTION_LEDGER_PATH` one line above it — a guard's witness, append-only.

    The name is spelled here rather than added to `app/paths.py` for a reason specific to this
    doc's neighbourhood: `architecture/autonomy-jobs.md` cites `app/paths.py` line numbers for
    symbols near the end of that file, `tests/test_autonomy_jobs_doc_claims.py` checks those
    citations against the bytes, and this item's own bound is that the doc goes untouched. One
    constant inserted there would have moved those numbers and forced a doc edit to keep the
    suite green — exactly the coupling this round is not here to renegotiate. Resolved inside
    the call so a test can redirect the journal and so importing this module does not drag
    `app.paths`' data-root resolution along.
    """
    from app import paths

    return paths.PIPELINE_DIR / "arch-review" / "membership-drift.jsonl"


def report_at_filing(task_id: object, *, task_dir: Path | None = None,
                     doc: Path | None = None, file_name: str = "",
                     journal: Path | None = None) -> dict:
    """What the two membership tables say about a task file that was just CREATED.

    Reads the doc once, names the tables that omit `task_id`, and — when there is one, which
    is every filing that has happened since the class was found — appends exactly ONE row to
    the drift journal, so the drift has a start date. A filing that both tables already name
    reports the emptiness and writes nothing: the journal's subject is drift, and a row per
    filing would make `wc -l` count filings instead of problems.

    It cannot refuse a write, and every failure inside it is a field in the returned dict:

    * `tables_missing` — the names from `TABLES` that do not list the id.
    * `unchecked` — present when the doc could not be read or holds no table, in which case
      `tables_missing` holds BOTH names. An unchecked value is never a passing value, so it
      is spelled out rather than reported as clean.
    * `journal` / `journal_error` — where the row went, or why it did not.
    """
    task_dir = Path(task_dir) if task_dir else DEFAULT_TASK_DIR
    doc_path = Path(doc) if doc else JOBS_DOC
    out: dict = {"task_id": as_id(task_id), "file": file_name,
                 "tables_missing": [], "row_written": False}

    try:
        text = doc_path.read_text(encoding="utf-8")
    except OSError as exc:
        out["tables_missing"] = [name for name, _ in TABLES]
        out["unchecked"] = f"{doc_path} unreadable: {type(exc).__name__}: {exc}"
        return out
    try:
        out["tables_missing"] = missing_tables(text, out["task_id"])
    except UnreadableDoc as exc:
        out["tables_missing"] = [name for name, _ in TABLES]
        out["unchecked"] = str(exc)
        return out

    out["doc_short_sha"] = doc_short_sha(text)
    if not out["tables_missing"]:
        return out

    row = {"task_id": out["task_id"], "file": file_name,
           "tables_missing": out["tables_missing"], "doc_short_sha": out["doc_short_sha"],
           "ts": datetime.now(timezone.utc).isoformat()}
    try:
        path = Path(journal) if journal else journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        out["journal"] = str(path)
        out["row_written"] = True
    except Exception as exc:  # noqa: BLE001 — a journal that will not open is a field too
        out["journal_error"] = f"{type(exc).__name__}: {exc}"
    return out
