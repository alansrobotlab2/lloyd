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

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
HUMAN_TOUCH_DAYS = 7
AUTOMOD_AUTHOR = "lloyd"
_HONESTY_RE = re.compile(r"^\+.*(\bor\s+True\b|^\+\s*assert\s+True\b)", re.M)


# ── inputs ───────────────────────────────────────────────────────────────

def _events(ledger: Path) -> list[dict]:
    out: list[dict] = []
    try:
        for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def _self_spawned_gauge(all_events: list[dict], backlog_dir: Path, now: float) -> dict:
    """How much of the board is the loop's own unjudged output, and whether
    any of it is older than the bound expiry maintains. `over_bound` should
    read 0; anything else means the expiry sweep is off or not running.
    All-time events, not the window: an item triaged last month is judged.
    """
    # Stdlib-only modules, imported lazily so the CLI stays light: the tag
    # set and the bound live in one place and are not restated here.
    from scripts.automod.backlog import (EXPIRY_EXEMPT_TAGS, LOOP_SPAWN_TAGS,
                                         SPAWN_TRIAGE_MIN_AGE_DAYS)
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
            tags = fm.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            tags = {str(t) for t in tags}
            if not (tags & LOOP_SPAWN_TAGS):
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
                    and not (tags & EXPIRY_EXEMPT_TAGS) and age >= SPAWN_TRIAGE_MIN_AGE_DAYS):
                over += 1
    return {"count": count, "oldest_days": round(oldest, 1),
            "bound_days": SPAWN_TRIAGE_MIN_AGE_DAYS, "over_bound": over}


def _frontmatter(path: Path) -> dict:
    try:
        import yaml
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    m = re.search(r"^---[ \t]*$", text[3:], re.M)
    if not m:
        return {}
    try:
        fm = yaml.safe_load(text[3:3 + m.start()])
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
                                        if not e.get("ok") and e.get("external_blocker"))}

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
    expired = len(by("backlog_expired"))
    spawn = {"triage_filed": t_filed, "triage_closed": t_closed,
             "triage_ratio": _rate(t_filed, t_closed),
             "implement_filed": i_filed, "implement_closed": i_closed,
             "implement_ratio": _rate(i_filed, i_closed),
             "triage_merged": t_merged, "implement_merged": i_merged,
             "findings_appended": appended, "expired": expired,
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
    throughput = {"items_closed": closed_items,
                  "items_closed_per_day": round(closed_items / max(since_days, 0.01), 2),
                  "rounds_finished": len(finished), "rounds_landed": len(landed),
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

    return {"computed_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds"),
            "since_days": since_days, "events": len(ev), "grouping": grouping,
            "acceptance": acceptance, "audit": audit, "review": review, "spawn": spawn,
            "human_touch": human, "test_honesty": honesty, "bookkeeping": bookkeeping,
            "verdict_plumbing": plumbing, "throughput": throughput, "rollbacks": rollbacks,
            "arch_review": arch_review}


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
        f"| 9 | throughput | {th['items_closed_per_day']}/day | {th['items_closed']} closed; {th['rounds_landed']} of {th['rounds_finished']} rounds landed; median turns {th['median_turns_landed'] if th['median_turns_landed'] is not None else '—'}; median gate {th['median_gate_seconds'] if th['median_gate_seconds'] is not None else '—'} s |",
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
    return "\n".join(lines)


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
