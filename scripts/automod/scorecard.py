"""The automod scorecard: is the unattended loop earning its keep?

Every gate rung was green on everything the loop landed in its first four
days, and that told us nothing about whether it was useful: 7 items closed
against 148 new drafts, 41% of structured verdicts silently falling back to
regex, one landed round that skipped an acceptance clause outright. The
numbers that would have said so were all on disk — the ledger, the backlog's
front matter, `workers.db`, `git log` — and nobody was adding them up.

`compute()` adds them up for a window and returns one row; `render()` prints
it as a table; `record()` appends the row to `scorecard.jsonl` in the automod
state dir so the trend survives. Read-only against everything it measures.
Stdlib + yaml, no app imports: this runs from the CLI, from the dashboard's
thread pool, and eventually from a nightly task, and must not drag the
application in behind it.

The metrics, each defined where it is computed:

  1  acceptance_hit_rate     met / landed rounds with an outcome
  2  audit_delta             grader-met clauses / author-met clauses (review rung)
  3  review                  refusals, in-turn fixes, re-offers, escalations
  4  spawn_ratio             filed / closed, triage and implement separately;
                             merges, appended findings, expiries, and the
                             open self-spawned gauge against its bound
  5  human_touch             landed rounds a human commit touched within 7 d
  6  test_honesty            findings per gated round, grader + deterministic
  7  bookkeeping             nameless deferrals, stranded landings, bare aborts
  8  verdict_plumbing        regex-fallback rate, truncations
  9  throughput              items closed/day, median turns, median gate seconds
 10  rollbacks               count (precision is a human judgment; recorded null)
 11  grouping                clusters formed, group triages and what they judged
 12  arch review             units reviewed, docs edited vs rejected, what was filed
 13  board net flow          items created minus items closed, 24 h and 7 d
 14  autocode duty cycle     share of the window with an implement turn in flight
 15  human overrides         decision events that name the person who made them
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

# `app/kg_store.py:48` shape, for the reason in `app/routers/dashboard.py`:
# `compute` parses the front matter of every board file twice per call
# (`_self_spawned_gauge` and the stranded-landings loop), and on those files
# libyaml is 15.1x the pure-Python scanner `yaml.safe_load` is pinned to. The
# `import yaml` used to sit inside `_frontmatter`, which is why the swap has to
# be at the call site — see `tests/test_dashboard_yaml_loader.py`.
try:  # ~15x faster than the pure-Python loader; all input is our own files
    from yaml import CSafeLoader as _YamlLoader  # type: ignore
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _YamlLoader  # type: ignore

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
HUMAN_TOUCH_DAYS = 7
AUTOMOD_AUTHOR = "lloyd"
_HONESTY_RE = re.compile(r"^\+.*(\bor\s+True\b|^\+\s*assert\s+True\b)", re.M)

# ── who counts as a person, and where their words live ────────────────────
#
# There is no canonical person-name source in the checkout to import: `grep -rn
# "Alan" --include=*.py --include=*.yaml` outside `tests/`, `architecture/` and
# `_pipeline/` returns comments only (`app/uptake.py:7`,
# `scripts/automod/backlog.py:77,210`, `agent-services/livekit_worker.py:445`).
# `config.yaml` is not the place either — it is human-only for the loop, which is
# one of the gate's standing denials — so the list lives beside the one reader
# that uses it. `people/alan/` is the vault's own record of the name; adding a
# second human means adding them here and nowhere else.
PERSON_NAMES = ("Alan",)
# The ledger fields a decision is recorded in. Over 6,283 rows (2026-09-17
# 21:14Z), 87 events name a person in `reason` (62) or `note` (26); the rest of
# the census (`response_tail` 16, `evidence` 15, `summary` 14,
# `next_pick_reason` 13, `acceptance` 7, `name` 5, `goal` 3) carries prose that
# quotes a person while describing something else — an `evidence` block reciting
# a directive is not itself a decision, and listing it would double-count the
# decision it quotes.
OVERRIDE_FIELDS = ("reason", "note")
# Cap on the rows the section carries, not on what it counted: `count` is what is
# listed, `found` is what the window held. It is a runaway guard, not a budget: a
# 14-day window held 87 person-named events on 2026-09-17 (measured through
# `compute`, not counted by hand), which is ~6 a day, so 120 rows is ~4 weeks of
# overrides — no window anyone reports on is truncated, and `--since all` over a
# 6,283-row ledger cannot fill a response unboundedly. That the cap is not biting
# at 2 weeks is itself worth stating, because a cap set below the window's real
# volume would silently drop an override from the listing while the premise of this
# section is that an override in the window is listed. The cost of carrying the
# sentences verbatim is size, and it is measured: the row without this section is
# 2,736 bytes over the live ledger, with it 33,436 — inside a `/api/dashboard`
# response that was 19,727 bytes, whose `automod` section was 2,827 of them, and
# which `_automod()` caches for 60 s (`app/routers/dashboard.py:712`). Rows are
# newest-first and `found` is reported beside `count`, so if the cap ever does
# bite, the report says so in its own numbers.
OVERRIDE_ROW_CAP = 120
# Longest verbatim quote the text report prints. The JSON carries the whole
# string; the longest override text on the live ledger is 649 chars and a table
# row is not where a paragraph belongs.
OVERRIDE_RENDER_CHARS = 160
# How many overrides the text report names by reference after the newest one. The
# newest gets its sentence; the rest of a 87-row window would not fit a line.
OVERRIDE_REFS_SHOWN = 5


def _person_named(value: object) -> str | None:
    """The first person named in `value`, or None.

    Whole-word and case-sensitive: `memory/alan/2026-06-20.md` is the one ledger
    string that matches `alan` without being a person, and a path is not an
    override.
    """
    if not isinstance(value, str) or not value:
        return None
    for name in PERSON_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", value):
            return name
    return None


# ── inputs ───────────────────────────────────────────────────────────────

def _events(ledger: Path) -> list[dict]:
    """Every ledger row in the window's reach, from the shared decode.

    `scripts.automod.state.ledger_rows` reads and decodes the file at most once
    per change to it, so `compute` arriving here — and `backlog._ledger_events`
    arriving at the same rows from the other side — costs one read between them
    instead of one each. Same contract as before for this caller: a missing or
    unreadable ledger is `[]`, a malformed line is skipped. Rows are shared
    read-only objects; the metrics below build their own lists off them.
    """
    from scripts.automod import state as S

    return S.ledger_rows(ledger)


def _self_spawned_gauge(all_events: list[dict], backlog_dir: Path, now: float) -> dict:
    """How much of the board is the loop's own unjudged output, and whether
    any of it is older than the bound expiry maintains. `over_bound` should
    read 0; anything else means the expiry sweep is off or not running.
    All-time events, not the window: an item triaged last month is judged.
    """
    # Stdlib-only modules, imported lazily so the CLI stays light: the tag
    # set and the bound live in one place and are not restated here.
    from app.backlog_tags import normalize_tags
    from scripts.automod.backlog import (BLOCKER_TAG, EXPIRY_EXEMPT_TAGS, blocker_liveness,
                                         blocker_targets, load_item, loop_spawn_tags,
                                         spawn_expiry_days)
    bound_days = spawn_expiry_days()
    spawn_tags = loop_spawn_tags()
    targets: dict[int, int] | None = None

    def _status_in_dir(iid: int) -> str | None:
        for p in backlog_dir.glob(f"{int(iid)}-*.md"):
            return str(_frontmatter(p).get("status", "draft"))
        return None
    judged: set[int] = set()
    for e in all_events:
        if e.get("item_id") is None:
            continue
        if e.get("event") in ("backlog_triage", "backlog_implement", "backlog_expired"):
            try:
                judged.add(int(e["item_id"]))
            except (TypeError, ValueError):
                continue
    count = over = 0
    oldest = 0.0
    if backlog_dir.exists():
        for path in backlog_dir.glob("*.md"):
            m = re.match(r"^(\d+)[-_]", path.name)
            if not m:
                continue
            fm = _frontmatter(path)
            # The one coercion every tag reader shares: the eval digest has
            # written `tags` as a string that looks like a list.
            tags = set(normalize_tags(fm.get("tags")))
            if not (tags & spawn_tags):
                continue
            if fm.get("status") not in ("draft", "up_next", "in_progress"):
                continue
            count += 1
            age = 0.0
            try:
                created = datetime.fromisoformat(str(fm.get("created") or "").replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = (now - created.timestamp()) / 86400
            except ValueError:
                pass
            oldest = max(oldest, age)
            if (fm.get("status") == "draft" and int(m.group(1)) not in judged
                    and not (tags & EXPIRY_EXEMPT_TAGS) and age >= bound_days):
                if BLOCKER_TAG in tags:
                    # Expiry skips a live blocker, so the gauge must too.
                    item = load_item(path)
                    if targets is None:
                        targets = blocker_targets(events=all_events)
                    if item is not None and blocker_liveness(item, targets, _status_in_dir)[0]:
                        continue
                over += 1
    return {"count": count, "oldest_days": round(oldest, 1),
            "bound_days": bound_days, "over_bound": over}


def _frontmatter(path: Path) -> dict:
    """First YAML block of a markdown file, or {} if there isn't one.

    Parses through `_YamlLoader` (libyaml's `CSafeLoader` where available), not
    `yaml.safe_load` — see the import above. The `import yaml` that used to sit
    inside this function is the trap: a reader that "fixed" the loader by
    editing the module attribute would have left this line on the pure-Python
    scanner, and every check that only inspects an attribute would agree that
    it was fixed.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    m = re.search(r"^---[ \t]*$", text[3:], re.M)
    if not m:
        return {}
    try:
        fm = yaml.load(text[3:3 + m.start()], Loader=_YamlLoader)
    except Exception:
        return {}
    return fm if isinstance(fm, dict) else {}


def _ts(e: dict) -> float:
    try:
        return float(e.get("ts") or 0)
    except (TypeError, ValueError):
        return 0.0


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _git_log(repo: Path, since_ts: float) -> list[dict]:
    """Commits since `since_ts` with author, time and touched files."""
    since = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "log", f"--since={since}",
             "--format=%x1e%H%x1f%an%x1f%ct%x1f%s", "--name-only"],
            capture_output=True, text=True, timeout=60, check=False)
    except Exception:
        return []
    out: list[dict] = []
    for chunk in r.stdout.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        head, _, files = chunk.partition("\n")
        parts = head.split("\x1f")
        if len(parts) < 4:
            continue
        out.append({"sha": parts[0], "author": parts[1], "ct": float(parts[2] or 0),
                    "subject": parts[3], "files": [f for f in files.splitlines() if f.strip()]})
    return out


def _commit_diff(repo: Path, sha: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo), "show", "--format=", "--unified=0", sha,
                            "--", "tests/"], capture_output=True, text=True, timeout=60, check=False)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


# ── the row ──────────────────────────────────────────────────────────────

def _duty_cycle(ev: list[dict], since: float, now: float) -> dict[str, Any]:
    """Row 14: how much of the window had an autocode implement turn in
    flight, and what the idle gaps were waiting on.

    Alan's rule (2026-09-15): an autocoder round runs 100% of the time. A
    turn is `backlog_implement` `started` until its `finished`,
    `infra_failed` or `skipped` row (an open one runs to `now`). A gap of at
    least a minute between turns is classified by what sits in it: a
    `promoted` row is a landing (the restart and, before the chamber, the
    observation window), a `round_aborted` row is an abort's finalizer and
    the retry, a `restart` row is a human restart, anything else `other`.
    `rate` is None with no turn in the window: unmeasured, not perfect.
    """
    spans: list[tuple[float, float]] = []
    open_at: dict[int, float] = {}
    for e in ev:
        if e.get("event") != "backlog_implement":
            continue
        iid = int(e.get("item_id") or 0)
        ph = str(e.get("phase") or "")
        if ph == "started":
            open_at[iid] = _ts(e)
        elif ph in ("finished", "infra_failed", "skipped") and iid in open_at:
            spans.append((open_at.pop(iid), _ts(e)))
    spans.extend((t, now) for t in open_at.values())
    spans = sorted((max(a, since), min(b, now)) for a, b in spans if b > since and a < now)
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    window = max(1.0, now - since)
    busy = sum(b - a for a, b in merged)
    idle: dict[str, float] = {}
    gaps: dict[str, int] = {}
    largest = 0.0
    markers = [e for e in ev if e.get("event") in ("promoted", "round_aborted", "restart")]
    for i in range(len(merged) - 1):
        end, nxt = merged[i][1], merged[i + 1][0]
        gap = nxt - end
        if gap < 60:
            continue
        kinds = {str(e["event"]) for e in markers if end - 60 <= _ts(e) <= nxt + 5}
        cls = ("landing" if "promoted" in kinds else "abort" if "round_aborted" in kinds
               else "restart" if "restart" in kinds else "other")
        idle[cls] = idle.get(cls, 0.0) + gap
        gaps[cls] = gaps.get(cls, 0) + 1
        largest = max(largest, gap)
    return {"rate": (busy / window) if merged else None,
            "busy_hours": round(busy / 3600, 1), "window_hours": round(window / 3600, 1),
            "turns": len(merged), "gaps": sum(gaps.values()),
            "idle_minutes": {k: round(v / 60, 1) for k, v in sorted(idle.items())},
            "gap_counts": dict(sorted(gaps.items())),
            "largest_gap_minutes": round(largest / 60, 1)}


def _iso_of(event: dict) -> str:
    """The event's own stamp, or the one its `ts` would have been written with.

    `state.append_event` always sets `created_at` beside `ts`
    (`scripts/automod/state.py:149-158`), so the derived form is only the
    fallback that keeps a copied or hand-written row from surfacing with a blank
    time.
    """
    stamp = event.get("created_at")
    if isinstance(stamp, str) and stamp:
        return stamp
    return datetime.fromtimestamp(_ts(event), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _overrides(events: list[dict]) -> dict[str, Any]:
    """Row 15: the decisions a person made that changed what the loop worked on.

    The behaviour this reports on is already correct and is not touched here. On
    2026-09-15 at 22:13:54Z Alan interrupted a running implement round to
    prioritise the backlog sweep; ten `backlog_retriage` and
    `backlog_confirm_released` events followed inside 60 s and the round went
    back to the pool. What was missing was the record. The sentence that moved
    the queue survived only as a `reason` string on line 4523 of a 5 MB
    `promotions.jsonl`: the 09-15 daily note has no line mentioning it, and
    `--since 2w --json` did not carry it either (both re-measured at triage,
    2026-09-17 04:20Z). So the machine honoured the priority while the reason
    for the change stayed invisible to every surface a human or a later run
    reads — including the run that inherits the queue and has to explain a round
    that is missing.

    One row per event, newest first, carrying the text verbatim: the quoted words
    are the artifact, and a paraphrase of them is a second claim to audit.
    """
    found = 0
    rows: list[dict[str, Any]] = []
    # `reversed` first, then a stable reverse sort on `ts`: two events sharing a
    # timestamp — routine inside one loop pass, which stamps `ts` per event — come
    # out in append order, so "newest" is the row the ledger wrote last rather
    # than whichever the decode happened to see first.
    for e in sorted(reversed(events), key=_ts, reverse=True):
        for field in OVERRIDE_FIELDS:
            person = _person_named(e.get(field))
            if person is None:
                continue
            found += 1
            if len(rows) < OVERRIDE_ROW_CAP:
                rows.append({"created_at": _iso_of(e), "event": e.get("event"),
                             "round_id": e.get("round_id"), "item_id": e.get("item_id"),
                             "person": person, "field": field, "text": e[field]})
            break  # one decision per event: two fields naming is still one act
    by_event: dict[str, int] = {}
    fields: dict[str, int] = {f: 0 for f in OVERRIDE_FIELDS}
    for r in rows:
        by_event[str(r["event"])] = by_event.get(str(r["event"]), 0) + 1
        fields[str(r["field"])] += 1
    return {"count": len(rows), "found": found, "cap": OVERRIDE_ROW_CAP,
            "by_event": dict(sorted(by_event.items())), "fields": fields, "rows": rows}


def compute(*, since_days: float = 7.0, ledger: Path | None = None,
            backlog_dir: Path | None = None, repo: Path | None = None,
            now: float | None = None) -> dict[str, Any]:
    from scripts.automod import state as S
    ledger = ledger or S.LEDGER_PATH
    backlog_dir = backlog_dir or (Path.home() / "obsidian" / "backlog")
    repo = repo or LIVE_ROOT
    now = now or time.time()
    since = now - since_days * 86400
    ev = [e for e in _events(ledger) if _ts(e) >= since]
    by = lambda kind: [e for e in ev if e.get("event") == kind]  # noqa: E731

    # ── 1 acceptance hit rate ───────────────────────────────────────────
    promoted = {str(e.get("round_id") or ""): e for e in by("promoted") if e.get("round_id")}
    reverted = {str(e.get("commit") or "") for e in by("rollback_succeeded")}
    finished = [e for e in by("backlog_implement") if e.get("phase") == "finished"]
    landed = [e for e in finished
              if str(e.get("round_id") or "") in promoted
              and str(promoted[str(e["round_id"])].get("commit") or "") not in reverted]
    with_outcome = [e for e in landed if isinstance(e.get("outcome"), dict)]
    met = [e for e in with_outcome if e["outcome"].get("acceptance") == "met"]
    acceptance = {"landed": len(landed), "with_outcome": len(with_outcome), "met": len(met),
                  "hit_rate": _rate(len(met), len(with_outcome))}

    # ── 2 audit delta (needs the review rung) ───────────────────────────
    reviews = [e for e in by("review") if e.get("ok")]
    author_met = grader_met = 0
    compared = 0
    outcome_by_round = {str(e.get("round_id") or ""): e.get("outcome") for e in finished}
    for r in reviews:
        rid = str(r.get("round_id") or "")
        oc = outcome_by_round.get(rid)
        if not isinstance(oc, dict) or not oc.get("clause_outcomes"):
            continue
        compared += 1
        author_met += sum(1 for c in oc["clause_outcomes"] if c.get("outcome") == "met")
        grader_met += sum(1 for c in (r.get("clauses") or []) if c.get("verdict") == "met")
    audit = {"rounds_compared": compared, "author_met": author_met, "grader_met": grader_met,
             "delta": _rate(grader_met, author_met)}

    # ── 3 review ────────────────────────────────────────────────────────
    review_gates = [e for e in by("gate") if e.get("rung") == "review"]
    refused = [e for e in review_gates if not e.get("ok") and e.get("review_retry")]
    unsound = [e for e in review_gates if not e.get("ok") and e.get("review_premise_unsound")]
    graded_rounds = {str(e.get("round_id") or "") for e in review_gates if not e.get("skipped")}
    refused_rounds = {str(e.get("round_id") or "") for e in refused}
    # In-turn fix: a refusal followed by a passing review in the same round.
    fixed_in_turn = 0
    for rid in refused_rounds:
        rows = sorted((e for e in review_gates if e.get("round_id") == rid), key=_ts)
        seen_refusal = False
        for e in rows:
            if not e.get("ok") and e.get("review_retry"):
                seen_refusal = True
            elif e.get("ok") and seen_refusal:
                fixed_in_turn += 1
                break
    review = {"rounds_graded": len(graded_rounds), "refusals": len(refused),
              "rounds_refused": len(refused_rounds), "fixed_in_turn": fixed_in_turn,
              "premise_unsound": len(unsound), "escalated": len(by("review_escalated")),
              "refusal_rate": _rate(len(refused_rounds), len(graded_rounds)),
              "grader_unavailable": sum(1 for e in review_gates
                                        if not e.get("ok") and e.get("external_blocker")),
              # How often a grading turn was NOT spent: an identical diff
              # after a rebase, answered from the ledger.
              "review_reused": sum(1 for e in review_gates if e.get("review_reused")),
              # Clauses the grader said can only be observed live. These land
              # and leave the item open for a person, so a rising number is
              # not a failure — it is the rung declining to refuse work it
              # cannot judge yet.
              "post_landing_clauses": sum(
                  len(e.get("post_landing_clauses") or []) for e in review_gates),
              # Waits the grader sat out because another round was landing.
              "grader_retries": sum(int(e.get("retries") or 0) for e in review_gates),
              # Rungs answered from the round's own cache rather than re-run.
              "reused_rungs": sum(1 for e in by("gate") if e.get("reused"))}

    # ── 4 spawn ratio ───────────────────────────────────────────────────
    triage = [e for e in by("backlog_triage") if e.get("verdict") in
              ("confirmed", "already_done", "stale", "unverifiable", "not_code", "folded")]
    t_filed = sum(len(e.get("spawned") or []) for e in triage)
    # A fold is a consolidation: the item leaves the pool. Duplicates are
    # already `stale` + closed.
    t_closed = sum(1 for e in triage if e.get("closed")
                   or e.get("verdict") in ("already_done", "stale", "folded"))
    i_filed = sum(len(e.get("spawned") or []) for e in finished)
    i_closed = (sum(1 for e in by("item_landed") if e.get("closed"))
                + sum(1 for e in by("item_closed")))
    # Merges and appends are findings that did NOT become items — the two
    # exits the write-time dedupe and the findings-to-parent rule opened.
    t_merged = sum(len(e.get("merged") or []) for e in triage)
    i_merged = sum(len(e.get("merged") or []) for e in finished)
    appended = sum(int(e.get("findings_appended") or 0) for e in finished)
    # Triage's own appends, apart: `findings_appended` above is implement's
    # and a test pins it that way. Since 2026-09-13 a triage that keeps its
    # item appends its findings to it instead of filing them.
    t_appended = sum(int(e.get("findings_appended") or 0) for e in triage)
    expired = len(by("backlog_expired"))
    spawn = {"triage_filed": t_filed, "triage_closed": t_closed,
             "triage_ratio": _rate(t_filed, t_closed),
             "implement_filed": i_filed, "implement_closed": i_closed,
             "implement_ratio": _rate(i_filed, i_closed),
             "triage_merged": t_merged, "implement_merged": i_merged,
             "findings_appended": appended, "triage_findings_appended": t_appended,
             "expired": expired,
             "self_spawned_open": _self_spawned_gauge(_events(ledger), backlog_dir, now)}

    # ── 4b grouping ─────────────────────────────────────────────────────
    group_runs = [e for e in by("backlog_group_triage") if isinstance(e.get("judged"), dict) and e.get("judged")]
    umbrella_confirmed = {int(e["item_id"]) for e in by("backlog_triage")
                          if e.get("umbrella") and e.get("item_id") is not None}
    members_closed = [e for e in by("item_closed") if e.get("by") == "umbrella"]
    umbrellas_landed = {int(e["item_id"]) for e in by("item_landed")
                        if e.get("closed") and int(e.get("item_id") or 0) in umbrella_confirmed}
    last_cluster = (by("backlog_cluster") or [{}])[-1]
    grouping = {"cluster_runs": len(by("backlog_cluster")),
                "clusters_formed": int(last_cluster.get("clusters") or 0),
                "items_clustered": int(last_cluster.get("items") or 0),
                "group_triages": len(group_runs),
                "duplicates_closed": sum(int(e.get("duplicates") or 0) for e in group_runs),
                "retired_in_group": sum(int(e.get("retired") or 0) for e in group_runs),
                "folded": sum(int(e.get("folded") or 0) for e in group_runs),
                "kept": sum(int(e.get("kept") or 0) for e in group_runs),
                "umbrellas_formed": sum(1 for e in group_runs if e.get("umbrella_id")),
                "umbrellas_landed": len(umbrellas_landed),
                "members_closed": len(members_closed),
                "members_per_landing": _rate(len(members_closed), len(umbrellas_landed))}

    # ── 5 human touch ───────────────────────────────────────────────────
    commits = _git_log(repo, since - HUMAN_TOUCH_DAYS * 86400)
    human_commits = [c for c in commits if c["author"] != AUTOMOD_AUTHOR]
    touched = 0
    touched_rounds: list[str] = []
    for rid, p in promoted.items():
        files = set(p.get("changed_paths") or [])
        pts = _ts(p)
        hit = any(c["ct"] > pts and c["ct"] - pts <= HUMAN_TOUCH_DAYS * 86400
                  and files & set(c["files"]) for c in human_commits)
        if hit:
            touched += 1
            touched_rounds.append(rid)
    human = {"landed": len(promoted), "touched_within_7d": touched,
             "rate": _rate(touched, len(promoted)), "rounds": touched_rounds[:10]}

    # ── 6 test honesty ──────────────────────────────────────────────────
    grader_findings = sum(len(e.get("test_honesty") or []) + len(e.get("prechecks") or [])
                          for e in by("review"))
    static_hits = 0
    for rid, p in promoted.items():
        sha = str(p.get("commit") or "")
        if sha and _HONESTY_RE.search(_commit_diff(repo, sha)):
            static_hits += 1
    honesty = {"grader_findings": grader_findings, "landed_with_or_true": static_hits,
               "per_gated_round": _rate(grader_findings, len(graded_rounds))}

    # ── 7 bookkeeping ───────────────────────────────────────────────────
    nameless = sum(1 for e in finished if isinstance(e.get("outcome"), dict)
                   and e["outcome"].get("acceptance") == "deferred"
                   and not e["outcome"].get("deferred_to"))
    bare_aborts = sum(1 for e in by("round_aborted") if not str(e.get("reason") or "").strip())
    stranded = 0
    if backlog_dir.exists():
        day_ago = now - 86400
        settled_at = {str(e.get("commit") or ""): _ts(e) for e in by("settled")}
        for path in backlog_dir.glob("*.md"):
            fm = _frontmatter(path)
            marker = fm.get("automod_landed")
            if fm.get("status") == "in_progress" and marker:
                when = settled_at.get(str(marker), 0.0)
                if when and when < day_ago:
                    stranded += 1
    bookkeeping = {"nameless_deferrals": nameless, "stranded_landings": stranded,
                   "bare_aborts": bare_aborts}

    # ── 8 verdict plumbing ──────────────────────────────────────────────
    sourced = [e for e in by("backlog_triage") if e.get("verdict_source") in ("regex", "structured")]
    regex = sum(1 for e in sourced if e.get("verdict_source") == "regex")
    truncated = sum(1 for e in by("backlog_triage") + finished
                    if "truncated" in str(e.get("structured_error") or e.get("outcome_error") or ""))
    fin_tokens = [e.get("finalizer_tokens") for e in by("backlog_triage") + finished
                  if isinstance(e.get("finalizer_tokens"), (int, float))]
    plumbing = {"verdicts_with_source": len(sourced), "regex": regex,
                "regex_rate": _rate(regex, len(sourced)), "truncated": truncated,
                "finalizer_tokens_median": _median(fin_tokens)}

    # ── 9 throughput ────────────────────────────────────────────────────
    closed_items = i_closed + t_closed
    gate_seconds: dict[str, float] = {}
    for e in by("gate"):
        rid = str(e.get("round_id") or "")
        gate_seconds[rid] = gate_seconds.get(rid, 0.0) + float(e.get("seconds") or 0)
    # A round that tried the idea and rejected it on the evidence resolved
    # its item as surely as a landing did (Alan's rule, 2026-09-16); counted
    # beside the landings so the row reads "resolved", not "shipped".
    rejected = sum(1 for e in by("item_closed") if e.get("acceptance") == "rejected")
    throughput = {"items_closed": closed_items,
                  "items_closed_per_day": round(closed_items / max(since_days, 0.01), 2),
                  "rounds_finished": len(finished), "rounds_landed": len(landed),
                  "rounds_rejected": rejected,
                  "median_turns_landed": _median([float(e.get("num_turns") or 0) for e in landed
                                                  if e.get("num_turns")]),
                  "median_gate_seconds": _median(list(gate_seconds.values()))}

    # ── 10 rollbacks ────────────────────────────────────────────────────
    rollbacks = {"count": len(by("rollback_succeeded")),
                 "triggers": sorted({str(e.get("trigger") or "?") for e in by("rollback_succeeded")}),
                 # Whether a rollback was a true positive is a human judgment;
                 # every one so far has been a false positive. Recorded as
                 # unknown rather than as 0/N, which would be a claim.
                 "true_positives": None}

    # ── 12 architecture review ──────────────────────────────────────────
    # The doc-maintenance pass. `rejected` is the number that matters: it
    # counts turns whose doc edit was thrown away for breaking a bound, which
    # is the only way this job can waste a whole session, and a rate that
    # stops being near zero means a bound is wrong or a prompt is unclear.
    arch = by("arch_review")
    arch_review = {
        "reviewed": len(arch),
        "by_kind": {k: sum(1 for e in arch if e.get("kind") == k) for k in ("doc", "group")},
        "updated": sum(1 for e in arch if e.get("doc_updated")),
        "committed": sum(1 for e in arch if e.get("commit")),
        "rejected": sum(1 for e in arch if e.get("doc_update_rejected")),
        "filed": sum(len(e.get("filed") or []) for e in arch),
        "merged": sum(len(e.get("merged") or []) for e in arch),
        "appended_to": sum(len(e.get("appended_to") or []) for e in arch),
        "stray_writes": sum(len(e.get("stray_writes") or []) for e in arch),
        "by_status": {v: sum(1 for e in arch if e.get("verdict") == v)
                      for v in sorted({str(e.get("verdict")) for e in arch if e.get("verdict")})},
    }

    # ── 13 board net flow ───────────────────────────────────────────────
    # Items created minus items closed, off the board files. The ratio in
    # row 4 is per run; this is whether the board is actually shrinking.
    try:
        from scripts.automod.backlog import board_flow
        flow = board_flow(backlog_dir=backlog_dir, now=now) if backlog_dir.exists() else {}
    except Exception:
        flow = {}

    # ── 14 autocode duty cycle ──────────────────────────────────────────
    duty = _duty_cycle(ev, since, now)

    # ── 15 human overrides ──────────────────────────────────────────────
    overrides = _overrides(ev)

    return {"computed_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds"),
            "since_days": since_days, "events": len(ev), "grouping": grouping,
            "acceptance": acceptance, "audit": audit, "review": review, "spawn": spawn,
            "human_touch": human, "test_honesty": honesty, "bookkeeping": bookkeeping,
            "verdict_plumbing": plumbing, "throughput": throughput, "rollbacks": rollbacks,
            "arch_review": arch_review, "flow": flow, "duty_cycle": duty,
            "overrides": overrides}


# ── output ───────────────────────────────────────────────────────────────

def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def render(row: dict) -> str:
    a, au, r, s = row["acceptance"], row["audit"], row["review"], row["spawn"]
    h, t, b, p, th, rb = (row["human_touch"], row["test_honesty"], row["bookkeeping"],
                          row["verdict_plumbing"], row["throughput"], row["rollbacks"])
    lines = [
        f"# Automod scorecard — last {row['since_days']:g} d ({row['events']} ledger events), "
        f"{row['computed_at']}", "",
        "| # | metric | value | detail |", "|---|---|---|---|",
        f"| 1 | acceptance hit rate | {_pct(a['hit_rate'])} | {a['met']} met of {a['with_outcome']} landed with an outcome ({a['landed']} landed) |",
        f"| 2 | audit delta | {_pct(au['delta'])} | grader {au['grader_met']} / author {au['author_met']} met clauses over {au['rounds_compared']} rounds |",
        f"| 3 | review refusal rate | {_pct(r['refusal_rate'])} | {r['rounds_refused']} of {r['rounds_graded']} graded rounds; {r['fixed_in_turn']} fixed in turn, {r['premise_unsound']} unsound, {r['escalated']} escalated, {r['grader_unavailable']} grader-unavailable |",
        f"| 4 | spawn ratio | triage {s['triage_ratio'] if s['triage_ratio'] is not None else '—'} · implement {s['implement_ratio'] if s['implement_ratio'] is not None else '—'} | filed/closed: triage {s['triage_filed']}/{s['triage_closed']}, implement {s['implement_filed']}/{s['implement_closed']}; merged {s.get('triage_merged', 0)}+{s.get('implement_merged', 0)}, appended {s.get('findings_appended', 0)}, expired {s.get('expired', 0)}; open self-spawned {s.get('self_spawned_open', {}).get('count', 0)} (oldest {s.get('self_spawned_open', {}).get('oldest_days', 0)} d, bound {s.get('self_spawned_open', {}).get('bound_days', 0)}, over bound {s.get('self_spawned_open', {}).get('over_bound', 0)}) |",
        f"| 5 | human-touch cost | {_pct(h['rate'])} | {h['touched_within_7d']} of {h['landed']} landed rounds touched by a human commit within {HUMAN_TOUCH_DAYS} d |",
        f"| 6 | test honesty | {t['grader_findings']} findings | {t['per_gated_round'] if t['per_gated_round'] is not None else '—'} per graded round; {t['landed_with_or_true']} landed commits add `or True`/`assert True` |",
        f"| 7 | bookkeeping defects | {b['nameless_deferrals'] + b['stranded_landings'] + b['bare_aborts']} | {b['nameless_deferrals']} nameless deferrals, {b['stranded_landings']} stranded landings, {b['bare_aborts']} bare aborts |",
        f"| 8 | verdict plumbing | {_pct(p['regex_rate'])} regex | {p['regex']} of {p['verdicts_with_source']} verdicts fell back; {p['truncated']} truncated; median finalizer tokens {p['finalizer_tokens_median'] if p['finalizer_tokens_median'] is not None else '—'} |",
        f"| 9 | throughput | {th['items_closed_per_day']}/day | {th['items_closed']} closed; {th['rounds_landed']} of {th['rounds_finished']} rounds landed, {th.get('rounds_rejected', 0)} rejected on evidence; median turns {th['median_turns_landed'] if th['median_turns_landed'] is not None else '—'}; median gate {th['median_gate_seconds'] if th['median_gate_seconds'] is not None else '—'} s |",
        f"| 10 | rollbacks | {rb['count']} | triggers {', '.join(rb['triggers']) or '—'}; true positives: human judgment, not computed |",
        f"| 11 | grouping | {row.get('grouping', {}).get('group_triages', 0)} group triages | {row.get('grouping', {}).get('clusters_formed', 0)} clusters over {row.get('grouping', {}).get('items_clustered', 0)} items last night; {row.get('grouping', {}).get('duplicates_closed', 0)} duplicates closed, {row.get('grouping', {}).get('retired_in_group', 0)} retired, {row.get('grouping', {}).get('folded', 0)} folded, {row.get('grouping', {}).get('kept', 0)} kept; {row.get('grouping', {}).get('umbrellas_formed', 0)} umbrellas formed, {row.get('grouping', {}).get('umbrellas_landed', 0)} landed closing {row.get('grouping', {}).get('members_closed', 0)} members |",
    ]
    ar = row.get("arch_review") or {}
    kinds = ar.get("by_kind") or {}
    statuses = ar.get("by_status") or {}
    lines.append(
        f"| 12 | arch review | {ar.get('reviewed', 0)} units | "
        f"{kinds.get('doc', 0)} docs, {kinds.get('group', 0)} groups; "
        f"{ar.get('committed', 0)} doc edits committed, {ar.get('rejected', 0)} rejected; "
        f"filed {ar.get('filed', 0)}, merged {ar.get('merged', 0)}, "
        f"appended {ar.get('appended_to', 0)}; "
        f"{ar.get('stray_writes', 0)} stray writes reverted; "
        f"{', '.join(f'{k} {v}' for k, v in statuses.items()) or 'no verdicts'} |")
    flow = row.get("flow") or {}
    day, week = flow.get("24h") or {}, flow.get("7d") or {}
    headline = f"{int(day.get('net', 0)):+d} / 24 h" if day else "—"
    lines.append(
        f"| 13 | board net flow | {headline} | "
        f"24 h: {day.get('created', 0)} created, {day.get('closed', 0)} closed; "
        f"7 d: {week.get('created', 0)} created, {week.get('closed', 0)} closed, "
        f"net {week.get('net', 0):+d}; triage appended {s.get('triage_findings_appended', 0)} findings |")
    d = row.get("duty_cycle") or {}
    idle = d.get("idle_minutes") or {}
    counts = d.get("gap_counts") or {}
    by_class = ", ".join(f"{k} {idle[k]:g} min ({counts.get(k, 0)})" for k in idle) or "none"
    lines.append(
        f"| 14 | autocode duty cycle | {_pct(d.get('rate'))} | "
        f"{d.get('busy_hours', 0)} h of {d.get('window_hours', 0)} h with an implement turn in flight, "
        f"{d.get('turns', 0)} turns; {d.get('gaps', 0)} gaps: {by_class}; "
        f"largest {d.get('largest_gap_minutes', 0):g} min |")
    lines.append(_render_overrides(row))
    return "\n".join(lines)


def _override_ref(event_row: dict) -> str:
    """How one override names itself: its round id, else `#<item id>`.

    An item id carries the `#` the board writes it with, so `#1199` on a report
    line is never mistakable for a round id — which matters because most
    overrides are board decisions with no round at all.
    """
    rid = event_row.get("round_id")
    if rid:
        return str(rid)
    iid = event_row.get("item_id")
    return f"#{iid}" if iid is not None else "no round or item"


def _render_overrides(row: dict) -> str:
    """Table row 15. `—` in the count means the row predates this section.

    The count is column 3; the detail names the newest override by reference with
    its timestamp, event type and quoted sentence, then names the rest by
    reference alone. The point of the line is that a reader can ask a week later
    what was overridden, when, and in whose words.

    Two render touches, neither of which the JSON beside this row carries: a `|`
    is escaped so one override cannot break the markdown table (1 of the 87
    person-named rows on the live ledger has one), and a quote past
    `OVERRIDE_RENDER_CHARS` is cut with an ellipsis — the longest live override
    text is 649 characters and a table row is not where a paragraph belongs.
    `overrides.rows[].text` stays verbatim; this only shortens the picture of it.
    """
    ov = row.get("overrides")
    if ov is None:
        return "| 15 | human overrides | — | section not recorded for this row |"
    count = ov.get("count")
    rows = ov.get("rows") or []
    if not count:
        return f"| 15 | human overrides | {count} | no person-named decision event in the window |"
    newest = rows[0]
    quote = str(newest.get("text") or "").replace("|", "\\|")
    if len(quote) > OVERRIDE_RENDER_CHARS:
        quote = quote[:OVERRIDE_RENDER_CHARS].rstrip() + "…"
    tail = ""
    others = ", ".join(_override_ref(r) for r in rows[1:OVERRIDE_REFS_SHOWN])
    if others:
        tail += f"; also {others}"
    if (ov.get("found") or 0) > count:
        tail += f"; {ov['found']} found, {count} listed"
    return (f"| 15 | human overrides | {count} | newest {_override_ref(newest)} "
            f"({newest.get('created_at')} {newest.get('event')}): \"{quote}\"{tail} |")


def scorecard_path() -> Path:
    from scripts.automod import state as S
    return S.STATE_DIR / "scorecard.jsonl"


def record(row: dict, path: Path | None = None) -> Path:
    path = path or scorecard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def parse_since(text: str) -> float:
    """`7d`, `36h`, `2w` or a bare number of days."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([dhw]?)\s*", text or "")
    if not m:
        raise ValueError(f"cannot parse --since {text!r}")
    n, unit = float(m.group(1)), m.group(2)
    return n / 24 if unit == "h" else n * 7 if unit == "w" else n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Automod scorecard")
    ap.add_argument("--since", default="7d", help="window: 7d, 36h, 2w")
    ap.add_argument("--json", action="store_true", help="print the row as JSON instead of a table")
    ap.add_argument("--record", action="store_true", help="append the row to scorecard.jsonl")
    args = ap.parse_args(argv)
    row = compute(since_days=parse_since(args.since))
    print(json.dumps(row, indent=2) if args.json else render(row))
    if args.record:
        print(f"\nrecorded → {record(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
