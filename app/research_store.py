"""The research topic registry: what to research, and what came of it.

This replaced `~/obsidian/lloyd/research-queue.md`, a markdown checklist that
had grown to 3,690 lines holding 2,839 checked items of which **314 were
unique** — a producer that no longer exists wrote the same 82 topics 390 times
under fabricated 2024 and 2025 dates. Reading that file "in full", as the
generator's skill instructed, cost 1.02 million tokens a night and timed the
task out until it auto-disabled itself.

Three things a checklist could not represent, and this exists to hold:

* **A lifecycle.** A checkbox cannot say researching, written, nothing found,
  or duplicate. It cannot say a turn tried twice and gave up. Everything
  downstream of "was it ticked" had to be guessed.
* **Dedup that survives a reword.** The box was ticked by a regex and deduped
  by eyeball over the whole history. `key` is UNIQUE and `similar()` answers
  the question the eyeball was for.
* **A feedback edge.** "Three searches found nothing" was an instruction to
  *log it* somewhere that did not exist. `nothing_found` is a recorded
  outcome, and `recent()` is what stops the generator proposing it again.

`architecture/research-pipeline.md` is the long version.

**Nothing outside this module opens the database.** Two processes write it —
the backend's `deep-research` worker and the aggregator's `research_*` MCP
tools — which is the same situation `workers.db` solves, so this copies
`workers/queue.py`: a short-lived connection per call under one lock, WAL, and
cross-process safety that comes from the SQL shape rather than from the lock.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from app.paths import RESEARCH_DB, VAULT_ROOT

logger = logging.getLogger("lloyd-research")


STATUSES = ("queued", "researching", "written", "nothing_found",
            "duplicate", "archived", "rejected")
#: States a topic never leaves. `finish` refuses to move one of these.
TERMINAL = frozenset({"written", "nothing_found", "duplicate", "archived", "rejected"})
#: The two outcomes that mean a turn actually ran today. Used for the daily
#: budget, which must not count the 314 legacy rows imported as `archived`.
PRODUCTIVE = ("written", "nothing_found")

#: Refuse new proposals past this depth. A generator that proposes 8 a night
#: against a worker that researches 3 a day would otherwise pile up months of
#: work nobody asked for, and the oldest of it would be answering questions
#: that stopped mattering weeks ago.
MAX_QUEUED = 40

#: A `queued` topic older than this is stale enough to report. Not deleted —
#: reported by `stats()`, so the decision stays a human's.
STALE_QUEUED_DAYS = 60


_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  topic         TEXT NOT NULL,
  key           TEXT NOT NULL UNIQUE,
  domain        TEXT,
  status        TEXT NOT NULL DEFAULT 'queued',
  priority      INTEGER NOT NULL DEFAULT 50,
  proposed_by   TEXT,
  signal        TEXT,
  proposed_at   TEXT NOT NULL,
  started_at    TEXT,
  finished_at   TEXT,
  not_before    TEXT,
  attempts      INTEGER NOT NULL DEFAULT 0,
  last_error    TEXT,
  queue_id      INTEGER,
  session_id    TEXT,
  artifact_path TEXT,
  outcome_note  TEXT,
  duplicate_of  INTEGER REFERENCES topics(id),
  extra         TEXT
);
CREATE INDEX IF NOT EXISTS idx_topics_ready
  ON topics(status, priority, proposed_at);
CREATE INDEX IF NOT EXISTS idx_topics_finished
  ON topics(status, finished_at);

CREATE TABLE IF NOT EXISTS events (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id  INTEGER NOT NULL REFERENCES topics(id),
  at        TEXT NOT NULL,
  event     TEXT NOT NULL,
  detail    TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic_id, at);
"""


class StoreUnavailable(RuntimeError):
    """The registry exists but cannot be opened or read.

    Deliberately not "return an empty list": an unreadable registry must never
    look like an empty one, because the caller downstream of an empty read is
    a generator that would cheerfully propose everything again.
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Key normalisation ────────────────────────────────────────────────────────

#: Trailing provenance the legacy file accumulated in five different shapes.
_SUFFIX_RE = re.compile(
    r"\s*(?:[—–-]+\s*researched\b.*|\(\s*researched\b.*?\)|→.*)$",
    re.IGNORECASE,
)

#: Words that carry no discriminating power in this corpus, so two topics that
#: share only these are not similar. Kept small and domain-specific on purpose:
#: a big stoplist starts merging topics that differ only in a word it dropped.
_STOPWORDS = frozenset("""
a an and are as at be by for from in into is of on or the to via vs with
using based real time model models learning agent agents robot robotic robots
policy policies system systems approach approaches new current latest
""".split())


def normalize_key(topic: str) -> str:
    """The identity of a topic, for the UNIQUE constraint.

    Case, punctuation and the legacy provenance suffixes are noise; a comma
    with no space after it is noise too, because the whole legacy corpus is
    written that way ("Isaac Gym,ROS2,MuJoCo").

    **Never truncated.** The retired `domain-research` source keyed on a
    50-character slug, and in the real corpus that collides: "Multi-agent task
    decomposition — hierarchical planning for" and "…for complex robotic
    workflows" both cut to the same string, so the second was silently dropped
    as a duplicate of the first.
    """
    text = unicodedata.normalize("NFKC", str(topic or ""))
    text = _SUFFIX_RE.sub("", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def _tokens(key: str) -> set[str]:
    return {t for t in key.split() if len(t) > 2 and t not in _STOPWORDS}


def _strip_suffix(topic: str) -> str:
    """The topic as a human would write it, without trailing provenance."""
    return _SUFFIX_RE.sub("", unicodedata.normalize("NFKC", str(topic or ""))).strip()


# ── Store ────────────────────────────────────────────────────────────────────


class ResearchStore:
    """Thread-safe SQLite registry of research topics.

    One instance per process; connections are short-lived and opened inside a
    lock for writes. Cross-process safety comes from the SQL — an
    `ON CONFLICT` insert and a guarded `UPDATE` — never from the lock, which
    only this process holds.
    """

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path).expanduser()
        self._lock = threading.RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StoreUnavailable(f"cannot create {self.path.parent}: {exc}") from exc
        self._init_db()

    # -- plumbing ------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """A fresh connection. `busy_timeout` is deliberately 5s, not the 30s
        `workers.db` uses: every statement here touches one row, so a lock held
        longer than that is a human with `sqlite3` open in a write transaction.
        That should surface as an error, not stall the backend's event loop for
        half a minute."""
        try:
            conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except sqlite3.Error as exc:
            raise StoreUnavailable(f"cannot open {self.path}: {exc}") from exc

    def _init_db(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.executescript(_SCHEMA)
                # Additive migrations go here, `workers/queue.py:_init_db`
                # style: read PRAGMA table_info and ALTER what is missing.
                conn.commit()
        except sqlite3.Error as exc:
            raise StoreUnavailable(f"cannot initialise {self.path}: {exc}") from exc

    def _event(self, conn: sqlite3.Connection, topic_id: int,
               event: str, detail: str = "") -> None:
        conn.execute(
            "INSERT INTO events (topic_id, at, event, detail) VALUES (?,?,?,?)",
            (topic_id, _now(), event, (detail or "")[:2000]),
        )

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[dict]:
        if row is None:
            return None
        out = dict(row)
        if out.get("extra"):
            try:
                out["extra"] = json.loads(out["extra"])
            except ValueError:
                out["extra"] = {"_unparsed": out["extra"]}
        else:
            out["extra"] = {}
        return out

    # -- propose -------------------------------------------------------------

    def propose(self, topic: str, *, domain: str = "", proposed_by: str = "",
                signal: str = "", priority: int = 50,
                extra: Optional[dict] = None) -> dict:
        """Register a topic. Returns `{id, created, status, topic, similar}`.

        The insert is `ON CONFLICT(key) DO NOTHING RETURNING id` rather than
        SELECT-then-INSERT, because two processes propose into this file: the
        nightly generator and whatever a chat turn asks for. The read-then-write
        shape (which `WorkQueue.enqueue` still uses) lets both see "not there"
        and the loser take an IntegrityError — which, on the generator, means a
        tool error inside a task that disables itself after three failures.
        """
        text = _strip_suffix(topic)
        if not text:
            raise ValueError("topic is empty")
        key = normalize_key(text)
        if not key:
            raise ValueError(f"topic normalises to nothing: {topic!r}")

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM topics WHERE key = ?", (key,)).fetchone()
            if existing is None:
                depth = conn.execute(
                    "SELECT COUNT(*) AS n FROM topics WHERE status='queued'"
                ).fetchone()["n"]
                if depth >= MAX_QUEUED:
                    return {
                        "id": None, "created": False, "status": None,
                        "topic": text, "similar": [],
                        "refused": (f"{depth} topics already queued (cap {MAX_QUEUED}); "
                                    f"research them before proposing more"),
                    }

            cur = conn.execute(
                """INSERT INTO topics
                   (topic, key, domain, status, priority, proposed_by, signal,
                    proposed_at, extra)
                   VALUES (?,?,?,'queued',?,?,?,?,?)
                   ON CONFLICT(key) DO NOTHING
                   RETURNING id""",
                (text, key, domain or None, int(priority), proposed_by or None,
                 signal or None, _now(),
                 json.dumps(extra, default=str) if extra else None),
            )
            row = cur.fetchone()
            if row is not None:
                topic_id = int(row["id"])
                self._event(conn, topic_id, "proposed",
                            f"by={proposed_by} signal={signal} domain={domain}")
                conn.commit()
                return {"id": topic_id, "created": True, "status": "queued",
                        "topic": text, "similar": self._similar(conn, key, exclude=topic_id)}

            current = conn.execute(
                "SELECT * FROM topics WHERE key = ?", (key,)).fetchone()
            if current is None:  # pragma: no cover - deleted between statements
                raise StoreUnavailable("topic vanished during propose")
            topic_id = int(current["id"])
            self._event(conn, topic_id, "reproposed", f"by={proposed_by} signal={signal}")
            conn.commit()
            return {"id": topic_id, "created": False, "status": current["status"],
                    "topic": current["topic"],
                    "similar": self._similar(conn, key, exclude=topic_id)}

    def similar(self, topic: str, *, limit: int = 5) -> list[dict]:
        """Topics that look like this one, most alike first."""
        with self._connect() as conn:
            return self._similar(conn, normalize_key(topic), limit=limit)

    def _similar(self, conn: sqlite3.Connection, key: str, *,
                 limit: int = 5, exclude: Optional[int] = None) -> list[dict]:
        """Token-overlap neighbours.

        Deliberately Python over every row rather than FTS5: the table holds a
        few hundred topics, the scan is sub-millisecond, and an FTS5
        external-content table needs three triggers to stay in sync — a
        migration story bought before there is a problem. `bm25` would also
        over-rank the domain words this corpus repeats endlessly ("sim-to-real",
        "VLA") without the stoplist doing the real work anyway. The signature
        is what matters; swapping the body for FTS5 later changes no caller.
        """
        want = _tokens(key)
        if not want:
            return []
        rows = conn.execute(
            "SELECT id, topic, key, status, finished_at, artifact_path FROM topics"
        ).fetchall()
        scored: list[tuple[float, int, dict]] = []
        for row in rows:
            if exclude is not None and row["id"] == exclude:
                continue
            have = _tokens(row["key"] or "")
            if not have:
                continue
            shared = want & have
            if not shared:
                continue
            jaccard = len(shared) / len(want | have)
            if jaccard < 0.4 and len(shared) < 4:
                continue
            scored.append((jaccard, len(shared), {
                "id": row["id"], "topic": row["topic"], "status": row["status"],
                "finished_at": row["finished_at"],
                "artifact_path": row["artifact_path"],
            }))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [s[2] for s in scored[:limit]]

    # -- claim / release / finish -------------------------------------------

    def next(self, n: int = 1) -> list[dict]:
        """The topics a worker should take, soonest first. Peek only."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM topics WHERE status='queued' "
                "AND (not_before IS NULL OR not_before <= ?) "
                "ORDER BY priority ASC, proposed_at ASC LIMIT ?",
                (_now(), max(1, int(n))),
            ).fetchall()
            return [self._row(r) for r in rows]

    def claim(self, topic_id: int, *, by: str, queue_id: Optional[int] = None) -> Optional[dict]:
        """Take a topic for research. Returns the row, or None if someone won.

        Guarded `UPDATE ... WHERE status='queued'` then a re-read, the shape
        `WorkQueue.claim_next` uses, because the aggregator and the worker are
        different processes and the lock above covers neither of them. Two
        turns researching one topic means two notes and two sets of facts.

        A re-claim of a topic this same queue item already holds succeeds: the
        pool can call `execute` again for one item after a raised failure, and
        the topic is legitimately still `researching` from the first attempt.
        """
        with self._lock, self._connect() as conn:
            # `RETURNING` on the UPDATE itself, so the answer to "did *I* claim
            # it" is the statement's own result. Updating and then re-reading
            # the row cannot tell a claim from someone else's: the re-read sees
            # `researching` either way, which handed the same topic to both
            # processes in the very first run of the concurrency test below.
            cur = conn.execute(
                "UPDATE topics SET status='researching', started_at=?, "
                "attempts=attempts+1, queue_id=?, not_before=NULL "
                "WHERE id=? AND (status='queued' "
                "                OR (status='researching' AND queue_id IS NOT NULL "
                "                    AND queue_id=?)) "
                "RETURNING *",
                (_now(), queue_id, int(topic_id), queue_id),
            )
            row = cur.fetchone()
            if row is None:
                conn.commit()
                return None
            self._event(conn, int(topic_id), "claimed", f"by={by} queue_id={queue_id}")
            conn.commit()
            return self._row(row)

    def release(self, topic_id: int, *, error: str = "",
                backoff_seconds: float = 0.0) -> Optional[dict]:
        """Put a topic back, optionally held off until the backoff expires.

        This is where a failed research turn goes, and the reason retries live
        here rather than in the work queue: `workers/pool.py` records an
        in-band `{"status": "failed"}` and then calls `mark_completed` on the
        item regardless. Only a *raised* exception reaches the queue's retry
        path. A source that returns `failed` and expects the queue to try again
        is simply not retried.
        """
        not_before = None
        if backoff_seconds and backoff_seconds > 0:
            not_before = (datetime.now(timezone.utc)
                          + timedelta(seconds=float(backoff_seconds))).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE topics SET status='queued', started_at=NULL, queue_id=NULL, "
                "not_before=?, last_error=? WHERE id=? AND status='researching'",
                (not_before, (error or "")[:2000], int(topic_id)),
            )
            self._event(conn, int(topic_id), "released", error[:500])
            conn.commit()
            return self.get(int(topic_id))

    def finish(self, topic_id: int, status: str, *, artifact_path: str = "",
               note: str = "", duplicate_of: Optional[int] = None,
               session_id: str = "", extra: Optional[dict] = None) -> dict:
        """Record a terminal outcome."""
        if status not in TERMINAL:
            raise ValueError(f"{status!r} is not a terminal status ({sorted(TERMINAL)})")
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM topics WHERE id=?", (int(topic_id),)).fetchone()
            if row is None:
                raise ValueError(f"no topic {topic_id}")
            if row["status"] in TERMINAL:
                raise ValueError(
                    f"topic {topic_id} is already {row['status']}; "
                    f"a settled outcome is not rewritten")
            merged = dict(self._row(row)["extra"])
            if extra:
                merged.update(extra)
            conn.execute(
                "UPDATE topics SET status=?, finished_at=?, artifact_path=?, "
                "outcome_note=?, duplicate_of=?, session_id=?, extra=? WHERE id=?",
                (status, _now(), str(artifact_path or "") or None,
                 (note or "")[:2000] or None,
                 int(duplicate_of) if duplicate_of else None,
                 session_id or row["session_id"],
                 json.dumps(merged, default=str) if merged else None,
                 int(topic_id)),
            )
            self._event(conn, int(topic_id), status, note[:500])
            conn.commit()
            return self.get(int(topic_id))

    def exhaust(self, topic_id: int, note: str) -> dict:
        """Give up after too many failed attempts. Recorded, never silent."""
        return self.finish(topic_id, "nothing_found", note=note,
                           extra={"reason": "exhausted"})

    def reclaim_stale(self, older_than_seconds: float) -> int:
        """Return topics stuck in `researching` to the queue.

        A backend killed mid-turn leaves one behind; without this it is
        invisible forever, exactly the way a `claimed` queue row was before
        `recover_claimed` learned to sweep every worker id.
        """
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(seconds=float(older_than_seconds))).isoformat()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM topics WHERE status='researching' "
                "AND (started_at IS NULL OR started_at < ?)", (cutoff,)).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE topics SET status='queued', started_at=NULL, queue_id=NULL "
                    "WHERE id=?", (row["id"],))
                self._event(conn, row["id"], "reclaimed",
                            f"researching since before {cutoff}")
            conn.commit()
            if rows:
                logger.warning("research: reclaimed %d stale topic(s)", len(rows))
            return len(rows)

    # -- inspection ----------------------------------------------------------

    def get(self, topic_id: int) -> Optional[dict]:
        with self._connect() as conn:
            return self._row(conn.execute(
                "SELECT * FROM topics WHERE id=?", (int(topic_id),)).fetchone())

    def list(self, *, status: Optional[str] = None, domain: Optional[str] = None,
             since_days: Optional[int] = None, limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM topics WHERE 1=1"
        args: list[Any] = []
        if status:
            sql += " AND status=?"
            args.append(status)
        if domain:
            sql += " AND domain=?"
            args.append(domain)
        if since_days:
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(days=int(since_days))).isoformat()
            sql += " AND COALESCE(finished_at, proposed_at) >= ?"
            args.append(cutoff)
        sql += " ORDER BY COALESCE(finished_at, proposed_at) DESC LIMIT ?"
        args.append(max(1, min(int(limit), 200)))
        with self._connect() as conn:
            return [self._row(r) for r in conn.execute(sql, args).fetchall()]

    def recent(self, days: int = 30, limit: int = 200) -> list[dict]:
        """What has been settled lately — the generator's "don't re-propose" list."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT topic, status, finished_at, artifact_path FROM topics "
                "WHERE finished_at >= ? AND status != 'archived' "
                "ORDER BY finished_at DESC LIMIT ?", (cutoff, int(limit))).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        stale = (datetime.now(timezone.utc)
                 - timedelta(days=STALE_QUEUED_DAYS)).isoformat()
        with self._connect() as conn:
            by_status = {r["status"]: r["n"] for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM topics GROUP BY status").fetchall()}
            placeholders = ",".join("?" * len(PRODUCTIVE))
            done_today = conn.execute(
                f"SELECT COUNT(*) AS n FROM topics WHERE status IN ({placeholders}) "
                f"AND finished_at >= ?", (*PRODUCTIVE, today)).fetchone()["n"]
            oldest = conn.execute(
                "SELECT MIN(proposed_at) AS t FROM topics WHERE status='queued'"
            ).fetchone()["t"]
            stale_n = conn.execute(
                "SELECT COUNT(*) AS n FROM topics WHERE status='queued' "
                "AND proposed_at < ?", (stale,)).fetchone()["n"]
            last_written = conn.execute(
                "SELECT topic, artifact_path, finished_at FROM topics "
                "WHERE status='written' ORDER BY finished_at DESC LIMIT 1").fetchone()
            top_domains = {r["domain"] or "(none)": r["n"] for r in conn.execute(
                "SELECT domain, COUNT(*) AS n FROM topics "
                "WHERE status NOT IN ('archived') GROUP BY domain "
                "ORDER BY n DESC LIMIT 10").fetchall()}
        return {
            "by_status": by_status,
            "total": sum(by_status.values()),
            "queued": by_status.get("queued", 0),
            "researching": by_status.get("researching", 0),
            "done_today": done_today,
            "oldest_queued_at": oldest,
            "stale_queued": stale_n,
            "last_written": dict(last_written) if last_written else None,
            "domains": top_domains,
        }

    def daily_budget_spent(self) -> int:
        """How many topics were settled by a real turn today."""
        return int(self.stats()["done_today"])

    # -- legacy import -------------------------------------------------------

    def import_legacy(self, md_path: str | Path, *, dry_run: bool = False) -> dict:
        """Fold the retired markdown checklist in as `archived` rows.

        Everything lands `archived`, never `written`, for two reasons. The
        checkbox only ever meant "a turn ran", and 90 of the 142 notes that
        produced were empty — so the tick is not evidence the topic was
        answered. And a daily budget that counts `written` would be saturated
        forever by 314 rows dated 2024.

        Idempotent: the UNIQUE key means a second run inserts nothing.
        """
        path = Path(md_path).expanduser()
        parsed = parse_legacy(path)
        result = {"parsed": parsed["parsed"], "unique": len(parsed["items"]),
                  "unchecked": parsed["unchecked"], "inserted": 0, "existing": 0,
                  "artifact_found": 0, "artifact_missing": 0,
                  "suspect_dates": parsed["suspect_dates"], "dry_run": dry_run}
        for item in parsed["items"].values():
            if item["artifact_path"]:
                result["artifact_found"] += 1
            elif item["artifact_raw"]:
                result["artifact_missing"] += 1
            if dry_run:
                continue
            with self._lock, self._connect() as conn:
                extra = {"legacy_occurrences": item["occurrences"]}
                if item["header_date_suspect"]:
                    extra["header_date_suspect"] = True
                cur = conn.execute(
                    """INSERT INTO topics
                       (topic, key, status, priority, proposed_by, signal,
                        proposed_at, finished_at, artifact_path, outcome_note, extra)
                       VALUES (?,?,'archived',90,'import','legacy-import',?,?,?,?,?)
                       ON CONFLICT(key) DO NOTHING RETURNING id""",
                    (item["topic"], item["key"], item["proposed_at"],
                     item["finished_at"], item["artifact_path"] or None,
                     item["artifact_raw"] or None, json.dumps(extra)),
                )
                row = cur.fetchone()
                if row is None:
                    result["existing"] += 1
                else:
                    self._event(conn, int(row["id"]), "imported",
                                f"from {path.name}; {item['occurrences']} occurrence(s)")
                    result["inserted"] += 1
                conn.commit()
        return result


# ── Legacy parsing (pure, so it can be tested without a database) ────────────

_ITEM_RE = re.compile(r"^\s*-\s+\[( |x|X)?\]\s+(.+?)\s*$")
_HEADER_RE = re.compile(r"^#{2,3}\s+Auto-queued\s+\(nightly\s+(\d{4}-\d{2}-\d{2})\)")
#: The note a legacy line claims to have produced, in every shape the file
#: actually used: a bare filename, a vault-relative path, either wrapped in
#: backticks, and the whole thing sometimes inside parentheses.
_ARTIFACT_RE = re.compile(r"→\s*[`(]?\s*([A-Za-z0-9._/-]+\.md)")
_RESEARCHED_RE = re.compile(r"researched\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)

#: Task #65 was created on this date, so any nightly header older than it is
#: fabricated. 307 of the file's 420 sections are.
_GENERATOR_BORN = "2026-04-19"


def parse_legacy(md_path: str | Path) -> dict:
    """Parse the retired checklist into deduplicated items.

    Returns `{parsed, unchecked, suspect_dates, items}` where `items` maps the
    normalised key to one record. Duplicates collapse onto the first key seen,
    keeping the newest plausible header date and any artifact any copy named.
    """
    path = Path(md_path).expanduser()
    text = path.read_text(encoding="utf-8", errors="replace")

    items: dict[str, dict] = {}
    header_date: Optional[str] = None
    parsed = 0
    unchecked = 0
    suspect = 0

    for line in text.splitlines():
        head = _HEADER_RE.match(line)
        if head:
            header_date = head.group(1)
            continue
        match = _ITEM_RE.match(line)
        if not match:
            continue
        parsed += 1
        checked = (match.group(1) or " ").lower() == "x"
        if not checked:
            unchecked += 1
        raw = match.group(2)
        topic = _strip_suffix(raw)
        key = normalize_key(raw)
        if not key:
            continue

        artifact_raw = ""
        artifact_path = ""
        art = _ARTIFACT_RE.search(raw)
        if art:
            artifact_raw = art.group(1)
            artifact_path = _resolve_artifact(artifact_raw)
        researched = _RESEARCHED_RE.search(raw)
        finished_at = researched.group(1) if researched else header_date
        suspect_date = bool(header_date and header_date < _GENERATOR_BORN)
        if suspect_date:
            suspect += 1

        record = items.get(key)
        if record is None:
            items[key] = {
                "topic": topic, "key": key, "occurrences": 1,
                "proposed_at": header_date or "",
                "finished_at": finished_at or header_date or "",
                "artifact_raw": artifact_raw, "artifact_path": artifact_path,
                "header_date_suspect": suspect_date,
            }
            continue
        record["occurrences"] += 1
        # Keep the newest plausible date: the fabricated 2024/2025 headers are
        # the ones that would otherwise win a naive "latest" comparison.
        if header_date and header_date > (record["proposed_at"] or ""):
            record["proposed_at"] = header_date
            if not suspect_date:
                record["header_date_suspect"] = False
        if finished_at and finished_at > (record["finished_at"] or ""):
            record["finished_at"] = finished_at
        if artifact_path and not record["artifact_path"]:
            record["artifact_path"] = artifact_path
            record["artifact_raw"] = artifact_raw
        elif artifact_raw and not record["artifact_raw"]:
            record["artifact_raw"] = artifact_raw

    return {"parsed": parsed, "unchecked": unchecked,
            "suspect_dates": suspect, "items": items}


def _resolve_artifact(raw: str) -> str:
    """Absolute path of a claimed note, or "" when it is not on disk.

    A path is recorded only if the file is really there — the point of the
    field is that a later reader can open it.
    """
    candidates: Iterable[Path]
    text = raw.strip().strip("`")
    if "/" in text:
        candidates = (VAULT_ROOT / text, Path(text).expanduser())
    else:
        candidates = (VAULT_ROOT / "knowledge" / "research" / text,)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return ""


# ── Process default ──────────────────────────────────────────────────────────

_default_lock = threading.Lock()
_default: Optional[ResearchStore] = None
_default_path: Path = RESEARCH_DB


def store() -> ResearchStore:
    """The process-wide registry, opened lazily on first use.

    Lazily on purpose: a module-level instance would create the live
    `research.db` at import time, which under the self-modification gate means
    a canary boot writing into the production file.
    """
    global _default
    with _default_lock:
        if _default is None or _default.path != _default_path:
            _default = ResearchStore(_default_path)
        return _default


def configure(path: Path | str) -> ResearchStore:
    """Point the process default at `path` (tests, rebuilds). Returns it."""
    global _default_path, _default
    with _default_lock:
        _default_path = Path(path)
        _default = None
    return store()


def reset() -> None:
    """Forget the default store (next `store()` reopens it)."""
    global _default
    with _default_lock:
        _default = None


def _main(argv: list[str]) -> int:  # pragma: no cover - operator entry point
    import argparse

    ap = argparse.ArgumentParser(prog="python -m app.research_store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="fold the retired checklist in")
    imp.add_argument("path")
    imp.add_argument("--dry-run", action="store_true")
    sub.add_parser("stats", help="print registry stats")
    args = ap.parse_args(argv)

    if args.cmd == "import":
        print(json.dumps(store().import_legacy(args.path, dry_run=args.dry_run), indent=2))
    else:
        print(json.dumps(store().stats(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(_main(sys.argv[1:]))
