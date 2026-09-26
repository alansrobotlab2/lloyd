"""ReasoningBank for the autocode loop (#1489): strategy items distilled from
the rounds the review rung refused, retrieved by item similarity.

After ReasoningBank (arXiv 2509.25140): distil what failed and why — and what
fixed it — from past attempts, and hand the few most similar to the next
attempt. Here the attempts are automod rounds and the judge is the review rung,
whose `review` ledger rows already carry per-clause verdicts with a grader's
note and per-test honesty findings. So the distiller is deterministic: no model
call, no cache, the same ledger always yields the same bank.

- **What becomes an item.** A clause a blocking review graded `partial`,
  `unmet` or `unsatisfiable`, and a `blocking` test-honesty finding. Each is
  classified into one refusal cause (`classify_cause`, first matching rule,
  `other` when none) and carries that cause's lesson plus the grader's own words
  as the example. When a LATER review of the same round graded the refused
  clause `met`, the item is a `worked` item: the refusal and what the grader
  accepted afterwards. At most `MAX_PER_ROUND` items per round, distinct
  causes, `worked` first.
- **What does not.** A clean first-review landing produces nothing: its review
  row says every clause was met and nothing about why, so there is no
  transferable lesson in it that the prompt does not already state.
- **Retrieval** has two modes. `retrieve` is TF-IDF cosine between the new
  item's title and clauses and each bank item's source item (its round goal and
  the refused clause), top `k` with distinct causes. `prior_items` ignores the
  item and takes the `k` most frequent causes. Offline, `prior` named a later
  refusal's cause in 78% of held-out rounds against 62% for similarity (paired
  −0.16 [−0.28, −0.03], n=64) and similarity was no better than random, so
  `prior` is the default mode. Neither ever returns the query item's own rounds
  (the re-offer banner already carries those) nor anything dated at or after
  `before_ts`.
- **Dated, bounded, pruned.** Every item carries its review's timestamp;
  `prune` drops items older than `max_age_days`, and a `failed` item whose
  (item, clause) a later round turned into a `worked` one is superseded.

The bank file is a derived cache under `app.paths.REASONING_BANK_PATH`; the
ledger is the source of truth and `refresh` rebuilds it whole.

Injection into the implement prompt is `workers.sources.autocode.reasoning_bank`:
`off` (the code default), `on`, or `ab`. The offline measurement
(eval/measurements/reasoningbank-2026-09-25.md) cannot say whether a round told
these things lands more often; `ab` is the live A/B that can (arm assignment,
the markers on the ledger and `ab-report` are at the bottom of this module):

    python -m scripts.automod.reasoning_bank ab-report [--since ISO] [--json]
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

BANK_VERSION = 1
MAX_PER_ROUND = 3
DEFAULT_K = 3
DEFAULT_MAX_AGE_DAYS = 30
DETAIL_CHARS = 240
REFUSED_VERDICTS = ("partial", "unmet", "unsatisfiable")

# (cause, pattern, lesson). First match wins, so the specific shapes come
# before the broad ones (`half_missing` would swallow most of them).
CAUSES: tuple[tuple[str, str, str], ...] = (
    ("unfalsifiable_test",
     r"cannot fail|can ?not fail|can never fail|could never fail|asserts nothing|tautolog|vacuous"
     r"|always pass|passes? (?:for any|regardless|without reaching)|never reach(?:es)? the code"
     r"|over a literal|restates? (?:the|its|RunOptions|a)|byte copy|cannot distinguish|no direction",
     "A test that cannot fail is refused even when the code is right. Before gating, break "
     "the code the test names (delete the branch, flip the condition) and watch the test go "
     "red; assert on the behaviour the clause names, never on a literal the test itself built."),
    ("fixture_touches_production",
     r"real ~/|into the real|production (?:state|tree|vault|file|memory)|live (?:vault|tree) file"
     r"|never redirects?|not redirect|module-global|resolves every path from",
     "Tests must not read or write production state. Redirect every module-global path the code "
     "under test resolves (FACTS_ROOT, MEMORIES_ROOT, DATA_ROOT…) into tmp_path, and assert "
     "the fixture's file changed, not the live one."),
    ("worktree_path_assumption",
     r"named 'lloyd'|directory is named|HOME to|path arithmetic|only works in a checkout"
     r"|FileNotFoundError|in this worktree it|__file__|gate-state",
     "The gate runs your tests in a worktree under a redirected HOME. Derive paths from "
     "app.paths / tmp_path, never from a checkout named 'lloyd' or ~/lloyd, and run the "
     "tests from the worktree the way run_tests.sh does before gating."),
    ("test_bypasses_production_path",
     r"hand-built|\bfake\b|stub(?:bed)?\b|calls? .{0,60}directly|not exercised|never evaluates"
     r"|no caller produces|bypass|instead of (?:the )?production|reimplement",
     "A test that drives a hand-built or faked path production never takes pins nothing. Drive "
     "the production entry point (the real caller, the real loop) and let the fixture supply "
     "only inputs."),
    ("skip_or_deselected",
     r"live_vault|deselect|skipif|pytest\.skip|\bskipped\b|marker",
     "A clause pinned only by a live_vault / skipped test is pinned by nothing under the gate. "
     "Pin it with a fixture-backed test that runs under -m 'not live_vault'."),
    ("out_of_diff",
     r"vault commit|out-of-diff|out of diff|not by this diff|rather than by this diff|prior landing"
     r"|in a prior|diff adds no|this round's diff (?:adds|does) not|landed (?:in|by) (?:an?|another)",
     "Evidence the grader cannot see in this round's diff does not count. Put the change and "
     "its test in this round's diff (or amend the clause), and cite the file:line in the diff."),
    ("protected_path",
     r"human-only|protected path|cannot be written by any diff|unlisted|may never touch|SETUP\.md"
     r"|agent-services/|supervisor",
     "A clause that needs a protected or human-only path cannot be met from a round. Say so "
     "early (automod_amend_clause or a blocker) instead of spending the round on it."),
    ("needs_live_service",
     r"live (?:re-?run|engine|service|measurement)|STOPPED|restart(?:ed)?\b|needs a live"
     r"|only observable by running|against the live engine|not running|live runs?\b|--apply"
     r"|no diff can carry|post-landing|live (?:alias|store|table|traffic)",
     "A clause that needs a live engine or a restart cannot be graded met from a worktree. "
     "Deliver the offline half with a test, and defer the live half by id rather than claim it."),
    ("bar_not_reached",
     r"against the required|required (?:≥|>=|bar)|below the (?:bar|required|threshold)"
     r"|does not reach|short of|≥ ?0\.\d|>= ?0\.\d|the bar|neither reaches|reaches \+",
     "When a clause names a measured bar, report the measured number on the shipped "
     "configuration; a gain measured on a different setting is refused. Below the bar, the "
     "honest outcome is `rejected` with the numbers."),
    ("unpinned",
     r"nothing pins|unpinned|not pinned|no (?:changed )?test|greps? (?:the )?source|source[- ]text"
     r"|only (?:test )?(?:greps|reads the source)|pinned only by|deleted without replacement"
     r"|exercised only by",
     "Every clause needs a test in the diff that exercises it — not a grep of the source and "
     "not prose. Name the test node on the clause."),
    ("artifact_stale",
     r"self-contradict|stale|re-render|still (?:reads|says|names|prescribes)|contradict"
     r"|docstring (?:claims|says)|does not match",
     "Re-render or re-read every artifact and doc the clause names after the last code change; "
     "a number or sentence left from an earlier run is graded as the deliverable."),
    ("half_missing",
     r"\bhalf\b|second (?:part|half)|only (?:one|the first)|absent|missing|nothing (?:writes|runs)"
     r"|no .{0,40}exists|not implemented|is not delivered|does not (?:write|run|record)"
     r"|do(?:es)? not exist|not committed|no (?:run |audit )?(?:artifact|file) ",
     "Clauses with two halves are graded on both. Before gating, walk each clause and point at "
     "the file:line and the test for every half it names."),
    ("contract_mismatch",
     r"hard-coded|instead of|rather than the|not the policy|clause (?:says|asks|requires)"
     r"|the clause|each written",
     "Implement the clause as written. When the design differs, amend the clause "
     "(automod_amend_clause) before gating instead of hoping the grader accepts the substitute."),
)
_COMPILED = tuple((c, re.compile(p, re.I), l) for c, p, l in CAUSES)
LESSONS = {c: l for c, _, l in CAUSES}
LESSONS["other"] = ("The review refused this clause for a reason specific to the item; read the "
                    "grader's note below before assuming the same approach will pass.")


def classify_cause(text: str) -> str:
    """The refusal cause a grader's note describes, or `other`."""
    t = text or ""
    for cause, rx, _ in _COMPILED:
        if rx.search(t):
            return cause
    return "other"


# --- the ledger → rounds --------------------------------------------------

def _ts(d: dict) -> float:
    try:
        return float(d.get("ts") or 0)
    except (TypeError, ValueError):
        return 0.0


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def rounds_from_events(events: Iterable[dict]) -> dict[str, dict]:
    """round_id -> {item_id, start_ts, goal, clauses, reviews, landed}.

    `clauses` is the item's contract as the latest triage before the round's
    start recorded it — what the review numbered its clauses against.
    """
    triage: dict[int, list[tuple[float, list[str]]]] = {}
    rounds: dict[str, dict] = {}
    for d in events:
        ev = d.get("event")
        if ev == "backlog_triage" and d.get("item_id") is not None:
            cl = d.get("acceptance_clauses")
            if isinstance(cl, list) and cl:
                triage.setdefault(_int(d["item_id"]), []).append((_ts(d), [str(c) for c in cl]))
        elif ev == "round_start" and d.get("round_id"):
            rounds[d["round_id"]] = {"round_id": d["round_id"], "item_id": _int(d.get("item_id")),
                                     "start_ts": _ts(d), "goal": str(d.get("goal") or ""),
                                     "reviews": [], "landed": False}
        elif ev == "review" and d.get("round_id") in rounds:
            rounds[d["round_id"]]["reviews"].append(d)
            if rounds[d["round_id"]]["item_id"] is None:
                rounds[d["round_id"]]["item_id"] = _int(d.get("item_id"))
        elif ev == "promoted" and d.get("round_id") in rounds:
            rounds[d["round_id"]]["landed"] = True
    for r in rounds.values():
        best: list[str] = []
        for ts, cl in triage.get(r["item_id"], []):
            if ts <= r["start_ts"]:
                best = cl
        r["clauses"] = best
        r["reviews"].sort(key=_ts)
    return rounds


def _clean(text: str, n: int = DETAIL_CHARS) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _informative(note) -> bool:
    """A grader note that says something (not the parser's placeholder)."""
    t = str(note or "").strip()
    return len(t) >= 25 and "not addressed by the grader" not in t


def _refusals(review: dict) -> list[dict]:
    """(clause, text, ts) for everything a blocking review refused on."""
    out = []
    if not (review.get("ok") and review.get("blocking")):
        return out
    for c in review.get("clauses") or []:
        if (isinstance(c, dict) and c.get("verdict") in REFUSED_VERDICTS
                and _informative(c.get("note"))):
            out.append({"clause": _int(c.get("clause")), "text": str(c.get("note") or ""),
                        "verdict": c.get("verdict")})
    for t in review.get("test_honesty") or []:
        if isinstance(t, dict) and t.get("severity") == "blocking":
            out.append({"clause": None, "text": f"{t.get('file', '')}:{t.get('line', '')} "
                                                f"{t.get('problem', '')}", "verdict": "test"})
    return out


def distill_round(rnd: dict) -> list[dict]:
    """At most `MAX_PER_ROUND` strategy items from one round, distinct causes."""
    reviews = [r for r in rnd.get("reviews") or [] if r.get("ok")]
    cands: list[dict] = []
    for i, rev in enumerate(reviews):
        for ref in _refusals(rev):
            fixed, known = "", _ts(rev)
            if ref["clause"] is not None:
                for later in reviews[i + 1:]:
                    for c in later.get("clauses") or []:
                        if (isinstance(c, dict) and _int(c.get("clause")) == ref["clause"]
                                and c.get("verdict") == "met"):
                            fixed = str(c.get("note") or "")
                            break
                    if fixed:
                        # Dated when it became knowable: the review that accepted
                        # the fix, not the refusal — or a replay at a cutoff
                        # between the two would see a fix from its future.
                        known = _ts(later)
                        break
            cands.append({**ref, "ts": known, "fixed": fixed})
    if not cands:
        return []
    # `worked` first (a refusal and what then passed), then in review order.
    cands.sort(key=lambda c: (not c["fixed"], c["ts"]))
    # (ties keep review order: `sorted` is stable)
    items, seen = [], set()
    clauses = rnd.get("clauses") or []
    for c in cands:
        cause = classify_cause(c["text"])
        if cause in seen:
            continue
        seen.add(cause)
        cl_text = (clauses[c["clause"] - 1] if c["clause"] and 0 < c["clause"] <= len(clauses)
                   else "")
        items.append({
            "v": BANK_VERSION,
            "id": f"{rnd['round_id']}:{len(items) + 1}",
            "ts": c["ts"], "date": _date(c["ts"]),
            "round_id": rnd["round_id"], "item_id": rnd.get("item_id"),
            "kind": "worked" if c["fixed"] else "failed",
            "cause": cause, "clause": c["clause"],
            "lesson": LESSONS[cause],
            "detail": _clean(c["text"]),
            "fix": _clean(c["fixed"]) if c["fixed"] else "",
            "source": _clean(f"{rnd.get('goal', '')} {cl_text}", 900),
        })
        if len(items) >= MAX_PER_ROUND:
            break
    return items


def build_bank(events: Iterable[dict]) -> list[dict]:
    """Every strategy item the ledger yields, oldest first."""
    out: list[dict] = []
    for rnd in rounds_from_events(events).values():
        out.extend(distill_round(rnd))
    out.sort(key=lambda d: d["ts"])
    return out


def prune(bank: list[dict], *, now: float | None = None,
          max_age_days: float = DEFAULT_MAX_AGE_DAYS) -> list[dict]:
    """Drop items past `max_age_days`, and a `failed` item whose (item, clause)
    a later `worked` item of the same item supersedes."""
    now = time.time() if now is None else now
    floor = now - max_age_days * 86400
    worked = {(d["item_id"], d["clause"], d["cause"]): d["ts"] for d in bank
              if d["kind"] == "worked" and d.get("clause") is not None}
    out = []
    for d in bank:
        if d["ts"] < floor:
            continue
        w = worked.get((d["item_id"], d.get("clause"), d["cause"]))
        if d["kind"] == "failed" and w is not None and w > d["ts"]:
            continue
        out.append(d)
    return out


# --- storage ---------------------------------------------------------------

def bank_path() -> Path:
    from app import paths
    return paths.REASONING_BANK_PATH


def write_bank(bank: list[dict], path: Path | None = None) -> Path:
    path = Path(path or bank_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in bank))
    os.replace(tmp, path)
    return path


def load_bank(path: Path | None = None) -> list[dict]:
    path = Path(path or bank_path())
    out = []
    try:
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("v") == BANK_VERSION:
                out.append(d)
    except OSError:
        return []
    return out


def refresh(events: Iterable[dict], path: Path | None = None, *,
            max_age_days: float = DEFAULT_MAX_AGE_DAYS) -> list[dict]:
    """Rebuild the bank from the ledger and write it. Deterministic and whole:
    the ledger is the source of truth, the file a cache."""
    bank = prune(build_bank(events), max_age_days=max_age_days)
    write_bank(bank, path)
    return bank


def refresh_if_stale(ledger: Path, path: Path | None = None, *,
                     max_age_s: float = 3600,
                     max_age_days: float = DEFAULT_MAX_AGE_DAYS) -> list[dict]:
    path = Path(path or bank_path())
    try:
        fresh = time.time() - path.stat().st_mtime < max_age_s
    except OSError:
        fresh = False
    if fresh:
        return load_bank(path)
    events = []
    with open(ledger) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                events.append(d)
    return refresh(events, path, max_age_days=max_age_days)


# --- retrieval -------------------------------------------------------------

_TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}")
_STOP = frozenset("""the and for with that this from are was were not but its into only
each any all has have had one two its per than then when what which who will would
should must can could does did done been being also over under via off out our your
their them they there here more most less least same such very just new old item
backlog implement clause round test tests""".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def _tfidf(docs: list[list[str]]) -> tuple[list[dict], dict]:
    df = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs) or 1
    idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
    vecs = []
    for d in docs:
        tf = Counter(d)
        v = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs, idf


def retrieve(bank: list[dict], query: str, *, k: int = DEFAULT_K,
             exclude_item: int | None = None, before_ts: float | None = None,
             min_score: float = 0.0) -> list[dict]:
    """Top `k` bank items by similarity of their source item to `query`,
    distinct causes. Never the query item's own rounds, never an item dated
    at or after `before_ts`."""
    pool = [d for d in bank
            if (exclude_item is None or d.get("item_id") != exclude_item)
            and (before_ts is None or d["ts"] < before_ts)]
    if not pool or k <= 0:
        return []
    vecs, idf = _tfidf([tokens(d["source"]) for d in pool])
    qtf = Counter(tokens(query))
    qv = {t: (1 + math.log(c)) * idf[t] for t, c in qtf.items() if t in idf}
    qn = math.sqrt(sum(x * x for x in qv.values())) or 1.0
    scored = []
    for d, v in zip(pool, vecs):
        s = sum(x * v.get(t, 0.0) for t, x in qv.items()) / qn
        if s > min_score:
            scored.append((s, d["ts"], d))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    out, causes = [], set()
    for s, _, d in scored:
        if d["cause"] in causes:
            continue
        causes.add(d["cause"])
        out.append({**d, "score": round(s, 4)})
        if len(out) >= k:
            break
    return out


def prior_items(bank: list[dict], *, k: int = DEFAULT_K, exclude_item: int | None = None,
                before_ts: float | None = None) -> list[dict]:
    """The `k` most frequent refusal causes in the bank, each by its newest item.

    No similarity at all — the same paragraph for every round until the mix
    moves. In the offline replay this named a later refusal's cause more often
    than similarity retrieval did, so it is the mode a live A/B should test
    first (`reasoning_bank_mode: prior`)."""
    pool = [d for d in bank
            if d["cause"] != "other"
            and (exclude_item is None or d.get("item_id") != exclude_item)
            and (before_ts is None or d["ts"] < before_ts)]
    freq = Counter(d["cause"] for d in pool)
    return [max((d for d in pool if d["cause"] == cause), key=lambda d: d["ts"])
            for cause, _ in freq.most_common(max(k, 0))]


def render_item(d: dict) -> str:
    head = (f"- [{d['date']} {d['round_id']} #{d['item_id']}, "
            f"{'refused then fixed' if d['kind'] == 'worked' else 'refused'}: "
            f"{d['cause'].replace('_', ' ')}] {d['lesson']}")
    body = f"\n  Grader said: {d['detail']}"
    if d.get("fix"):
        body += f"\n  Then accepted: {d['fix']}"
    return head + body


def render_block(items: list[dict]) -> str:
    """The prompt block, or "" for no items. Sits beside the re-offer banner."""
    if not items:
        return ""
    return ("**Lessons from similar rounds** (the review rung's refusals on items "
            "like this one, distilled from the ledger; advice, not contract):\n"
            + "\n".join(render_item(d) for d in items) + "\n\n")


def item_query(name: str, clauses: Iterable[str], body: str = "") -> str:
    """The retrieval query for an item: title, clauses, and the body with any
    appended `## Findings` sections cut (those are a round's own output)."""
    body = re.split(r"(?m)^##\s+Findings", body or "", maxsplit=1)[0]
    return " ".join([name or "", *[str(c) for c in clauses or []], body[:4000]])


def fit_block(items: list[dict], max_chars: int) -> str:
    """`render_block` held under `max_chars`: the grader's words shrink first
    (they are the example, the lesson is the advice), then items drop from the
    end (the least frequent cause). "" when not even one item fits."""
    items = [dict(d) for d in items]
    block = render_block(items)
    detail = DETAIL_CHARS
    while block and len(block) > max_chars and detail > 60:
        detail -= 30
        for d in items:
            d["detail"] = _clean(d.get("detail") or "", detail)
            if d.get("fix"):
                d["fix"] = _clean(d["fix"], detail)
        block = render_block(items)
    while items and len(block) > max_chars:
        items.pop()
        block = render_block(items)
    return block


# --- the live A/B (#1489) --------------------------------------------------
#
# `workers.sources.autocode.reasoning_bank: ab` splits implement rounds into
# two arms. `control` gets no block; `common_causes` gets the `prior` block
# (the three most frequent refusal causes, recomputed from the ledger through
# the hourly bank cache), capped at `AB_MAX_BLOCK_CHARS`. The arm is written on
# the `backlog_implement started` row BEFORE the turn and copied onto the
# round's `round_start` row by `round.start`, so an analysis can join from
# either side. `ab_report` is the reading; `architecture/automod.md` §3.2j and
# eval/measurements/reasoningbank-2026-09-25.md carry the protocol.

SETTINGS = ("off", "on", "ab")
ARMS = ("control", "common_causes")
ARM_SALT = "rb-ab-v1"
#: A fresh assignment follows the hash unless the hashed arm is already this
#: many rounds ahead, in which case the lagging arm is taken. Deterministic
#: given the ledger; keeps the split within a few rounds of even.
BALANCE_SLACK = 3
#: ~530 tokens at the ~3.1 chars/token the primary's tokenizer gave the
#: rendered block offline (1626 chars = 521 tokens on the 09-25 ledger).
AB_MAX_BLOCK_CHARS = 1650
#: Below this many done rounds in either arm the report says `insufficient`.
AB_MIN_N_PER_ARM = 50
#: The stop rule: this many done rounds per arm (a ~0.16 landed-rate MDE at the
#: 09-19..25 pooled rate of 0.60), or `AB_MAX_DAYS`, whichever comes first.
AB_TARGET_N_PER_ARM = 150
AB_MAX_DAYS = 10
AB_MDE = 0.15
AB_RESAMPLES = 4000


def setting(value) -> str:
    """`reasoning_bank` read as off | on | ab. The key shipped as a bool, so
    `true`/`false` keep their meaning; anything unrecognised is `off`."""
    if value is True:
        return "on"
    if value is None or value is False:
        return "off"
    v = str(value).strip().lower()
    if v in ("on", "true", "yes", "1"):
        return "on"
    if v == "ab":
        return "ab"
    return "off"


def hash_arm(key: str) -> str:
    """The arm a key hashes to. Stable across processes (never `hash()`)."""
    import hashlib
    h = hashlib.sha256(f"{ARM_SALT}:{key}".encode()).digest()
    return ARMS[h[0] & 1]


def implement_rows(events: Iterable[dict]) -> list[dict]:
    return [e for e in events if isinstance(e, dict) and e.get("event") == "backlog_implement"]


def _chain_arm(rows: list[dict], session_id: str) -> tuple[str, str] | None:
    """(arm, cluster) of the turn that last ran in `session_id`, if it had one."""
    fin = next((r for r in reversed(rows)
                if r.get("session_id") == session_id and r.get("phase") == "finished"), None)
    if fin is None:
        return None
    item, ts = _int(fin.get("item_id")), _ts(fin)
    for r in reversed(rows):
        if (r.get("phase") == "started" and _int(r.get("item_id")) == item
                and _ts(r) <= ts and r.get("reasoning_bank_arm") in ARMS):
            return r["reasoning_bank_arm"], str(r.get("reasoning_bank_cluster")
                                                or r.get("reasoning_bank_arm_key") or "")
    return None


def assign_arm(events: Iterable[dict], item_id: int, *,
               continues_session: str | None = None) -> dict:
    """The arm for the next implement round of `item_id`.

    The unit is the item-round: key `<item>:<attempt>`, the attempt being how
    many `started` rows the item already has, so a re-offer is a new draw and
    the assignment can be re-derived from the ledger. A round that CONTINUES a
    warm session (`continue_session`) inherits that session's arm — the
    earlier prompt, block or no block, is in its history, so a fresh draw would
    contaminate both arms — and shares its cluster, which the report resamples
    by. A fresh draw follows `hash_arm` unless that arm leads by
    `BALANCE_SLACK`.
    """
    rows = implement_rows(events)
    attempt = sum(1 for r in rows if r.get("phase") == "started"
                  and _int(r.get("item_id")) == int(item_id))
    key = f"{int(item_id)}:{attempt}"
    if continues_session:
        inherited = _chain_arm(rows, continues_session)
        if inherited:
            return {"arm": inherited[0], "key": key, "cluster": inherited[1] or key,
                    "how": "chain"}
    counts = Counter(r["reasoning_bank_arm"] for r in rows
                     if r.get("phase") == "started" and r.get("reasoning_bank_arm") in ARMS)
    arm = hash_arm(key)
    other = ARMS[1 - ARMS.index(arm)]
    how = "hash"
    if counts[arm] - counts[other] >= BALANCE_SLACK:
        arm, how = other, "balance"
    return {"arm": arm, "key": key, "cluster": key, "how": how}


def arm_for_round(events: Iterable[dict], item_id: int | None) -> dict:
    """The fields `round.start` copies onto `round_start`: the arm of the item's
    open implement turn (its latest implement row is a `started` carrying one),
    or {} — a CLI round, an item not in the A/B, a turn already ended."""
    if item_id is None:
        return {}
    last = None
    for r in implement_rows(events):
        if _int(r.get("item_id")) == int(item_id):
            last = r
    if not last or last.get("phase") != "started" or last.get("reasoning_bank_arm") not in ARMS:
        return {}
    return {"reasoning_bank_arm": last["reasoning_bank_arm"],
            "reasoning_bank_arm_key": last.get("reasoning_bank_arm_key")}


# --- the A/B reading -------------------------------------------------------

_END_PHASES = ("finished", "infra_failed", "skipped")


def ab_units(events: list[dict], since_ts: float | None = None) -> list[dict]:
    """One unit per armed `started` row, joined to its end row and round.

    A unit is `open` (no end row yet), `excluded` (`skipped`: a landing drain
    refused it; `infra_failed`: the turn never ran — neither is an attempt; or
    a later `started` with no end between), or `done`, carrying: landed (a
    promotion of its round, or a vault landing), refusals (blocking review rows
    on its round), resolved (the item closed between this start and its next
    one — `item_landed closed` or an autocode `item_closed` — or the round
    reported `rejected` / `unnecessary`), hours (start to end of the turn)."""
    promoted, refusals, causes = set(), Counter(), {}
    closes: dict[int, list[float]] = {}
    round_arm: dict[str, str] = {}
    for e in events:
        ev = e.get("event")
        rid = e.get("round_id")
        if ev == "promoted" and rid:
            promoted.add(rid)
        elif ((ev == "item_landed" and e.get("closed"))
              or (ev == "item_closed" and e.get("by") == "autocode")):
            # Item-scoped, not round-scoped: a vault landing's `item_landed`
            # names no round. Credited to the attempt it falls inside.
            if _int(e.get("item_id")) is not None:
                closes.setdefault(_int(e.get("item_id")), []).append(_ts(e))
        elif ev == "review" and rid and e.get("ok") and e.get("blocking"):
            refusals[rid] += 1
            causes.setdefault(rid, Counter()).update(
                classify_cause(r["text"]) for r in _refusals(e))
        elif ev == "round_start" and rid and e.get("reasoning_bank_arm") in ARMS:
            round_arm[rid] = e["reasoning_bank_arm"]
    rows = implement_rows(events)
    units = []
    for i, r in enumerate(rows):
        if r.get("phase") != "started" or r.get("reasoning_bank_arm") not in ARMS:
            continue
        if since_ts is not None and _ts(r) < since_ts:
            continue
        item = _int(r.get("item_id"))
        end = next((x for x in rows[i + 1:] if _int(x.get("item_id")) == item
                    and x.get("phase") in _END_PHASES + ("started",)), None)
        nxt = next((_ts(x) for x in rows[i + 1:] if _int(x.get("item_id")) == item
                    and x.get("phase") == "started"), float("inf"))
        u = {"arm": r["reasoning_bank_arm"], "item_id": item, "ts": _ts(r),
             "key": r.get("reasoning_bank_arm_key"),
             "cluster": str(r.get("reasoning_bank_cluster") or r.get("reasoning_bank_arm_key")
                            or f"{item}@{_ts(r)}")}
        if end is None:
            u["state"] = "open"
        elif end.get("phase") != "finished":
            u["state"] = "excluded"
        else:
            rid = end.get("round_id")
            outcome = end.get("outcome") if isinstance(end.get("outcome"), dict) else {}
            acc = str(outcome.get("acceptance") or "")
            landed = bool(rid and rid in promoted) or bool(end.get("vault_commits"))
            closed = any(_ts(r) <= t < nxt for t in closes.get(item, ()))
            u.update({
                "state": "done", "round_id": rid, "landed": landed,
                "refusals": refusals.get(rid, 0) if rid else 0,
                "resolved": closed or acc in ("rejected", "unnecessary"),
                "pending_settle": landed and not closed and acc == "met",
                "hours": max(0.0, (_ts(end) - _ts(r)) / 3600.0),
                "causes": dict(causes.get(rid, {})) if rid else {},
                "round_arm_matches": round_arm.get(rid) in (None, u["arm"]) if rid else True,
            })
        units.append(u)
    return units


def _cluster_boot(groups: dict[str, list[list[dict]]], stat, *, seed: int,
                  n: int = AB_RESAMPLES) -> tuple[float | None, float | None]:
    """Percentile 95% CI of `stat({arm: units})`, resampling whole clusters
    within each arm. None when an arm has fewer than two clusters."""
    import random
    arms = [a for a in groups if groups[a]]
    if not arms or len(arms) < len(groups) or any(len(groups[a]) < 2 for a in arms):
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        draw = {a: [u for _ in range(len(groups[a]))
                    for u in groups[a][rng.randrange(len(groups[a]))]] for a in arms}
        v = stat(draw)
        if v is not None:
            vals.append(v)
    if len(vals) < n // 2:
        return None, None
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def _rate(us, k):
    return sum(1 for u in us if u[k]) / len(us) if us else None


def _mean(us, k):
    return sum(u[k] for u in us) / len(us) if us else None


def _per_hour(us):
    h = sum(u["hours"] for u in us)
    return sum(1 for u in us if u["resolved"]) / h if h > 0 else None


def n_for_mde(p: float, mde: float, *, z_a: float = 1.959964, z_b: float = 0.841621) -> int:
    """Rounds per arm to detect a proportion moving `p` → `p + mde` (two-sided
    alpha 0.05, power 0.8). Ignores clustering, so it is a floor."""
    p2 = min(max(p + mde, 0.0), 1.0)
    d = abs(p2 - p)
    if d <= 0:
        return 0
    pbar = (p + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar))
           + z_b * math.sqrt(p * (1 - p) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / d ** 2))


def ab_report(events: list[dict], *, since_ts: float | None = None, now: float | None = None,
              min_n: int = AB_MIN_N_PER_ARM, target_n: int = AB_TARGET_N_PER_ARM,
              max_days: float = AB_MAX_DAYS,
              mde: float = AB_MDE, seed: int | None = None) -> dict:
    """Per-arm outcomes of the live A/B with intervals, and a verdict line.

    Per arm over `done` units: rounds; landed rate (Wilson — optimistic when
    chains cluster); review refusals per round; items resolved per round-hour
    (the gauge CLAUDE.md names for anything that changes how rounds go), both
    with cluster-bootstrap CIs. Differences are `common_causes − control`,
    cluster-bootstrapped (a warm-session chain is one cluster). Below `min_n`
    done rounds in either arm the verdict is `insufficient`, whatever the
    intervals say."""
    from eval.stats import SEED, wilson_ci
    seed = SEED if seed is None else seed
    now = time.time() if now is None else now
    got = ab_units(events, since_ts)
    start = min((u["ts"] for u in got), default=None)
    done = [u for u in got if u["state"] == "done"]
    by_arm = {a: [u for u in done if u["arm"] == a] for a in ARMS}
    clusters: dict[str, dict[str, list[dict]]] = {a: {} for a in ARMS}
    for u in done:
        clusters[u["arm"]].setdefault(u["cluster"], []).append(u)
    groups = {a: list(clusters[a].values()) for a in ARMS}
    begin = since_ts if since_ts is not None else start
    out: dict = {"since": begin,
                 "days": round((now - begin) / 86400, 2) if begin else 0.0,
                 "min_n": min_n, "target_n": target_n, "max_days": max_days,
                 "mde": mde, "arms": {},
                 "open": sum(1 for u in got if u["state"] == "open"),
                 "excluded": sum(1 for u in got if u["state"] == "excluded"),
                 "round_arm_mismatches": sum(1 for u in done if not u["round_arm_matches"])}
    for i, a in enumerate(ARMS):
        us = by_arm[a]
        k = sum(1 for u in us if u["landed"])
        lo, hi = wilson_ci(k, len(us)) if us else (None, None)
        g = {a: groups[a]}
        out["arms"][a] = {
            "rounds": len(us), "clusters": len(groups[a]),
            "landed": k, "landed_rate": _rate(us, "landed"), "landed_ci": [lo, hi],
            "refusals_per_round": _mean(us, "refusals"),
            "refusals_ci": list(_cluster_boot(g, lambda d, a=a: _mean(d[a], "refusals"),
                                              seed=seed + i)),
            "resolved": sum(1 for u in us if u["resolved"]),
            "round_hours": round(sum(u["hours"] for u in us), 2),
            "resolved_per_round_hour": _per_hour(us),
            "resolved_per_round_hour_ci": list(_cluster_boot(
                g, lambda d, a=a: _per_hour(d[a]), seed=seed + 10 + i)),
            "pending_settle": sum(1 for u in us if u["pending_settle"]),
            "causes": dict(sum((Counter(u["causes"]) for u in us), Counter())),
        }
    c, t = ARMS

    def diff(f):
        def s(d):
            x, y = f(d[c]), f(d[t])
            return None if x is None or y is None else y - x
        pt = s(by_arm)
        lo, hi = _cluster_boot(groups, s, seed=seed + 20)
        return {"diff": pt, "lo": lo, "hi": hi,
                "excludes_zero": lo is not None and hi is not None and (lo > 0 or hi < 0)}

    out["diff"] = {"landed_rate": diff(lambda us: _rate(us, "landed")),
                   "refusals_per_round": diff(lambda us: _mean(us, "refusals")),
                   "resolved_per_round_hour": diff(_per_hour)}
    pooled = _rate(done, "landed")
    out["n_needed_per_arm"] = n_for_mde(pooled if pooled is not None else 0.4, mde)
    n_min = min(len(by_arm[a]) for a in ARMS)
    out["stop_rule_met"] = n_min >= target_n or out["days"] >= max_days
    if n_min < min_n:
        out["verdict"] = (f"insufficient: {n_min} done rounds in the smaller arm, "
                          f"need >= {min_n} per arm ({out['days']:g} of {max_days:g} days)")
    else:
        d, dl = out["diff"]["resolved_per_round_hour"], out["diff"]["landed_rate"]
        if d["excludes_zero"] and d["diff"] > 0 and not (dl["excludes_zero"] and dl["diff"] < 0):
            out["verdict"] = "common_causes wins on items resolved per round-hour"
        elif d["excludes_zero"] and d["diff"] < 0:
            out["verdict"] = "common_causes loses on items resolved per round-hour"
        else:
            out["verdict"] = "no measured difference (the resolved-per-round-hour interval includes zero)"
    return out


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _ci(pair, nd=3):
    return f"[{_fmt(pair[0], nd)}, {_fmt(pair[1], nd)}]"


def format_report(r: dict) -> str:
    since = (datetime.fromtimestamp(r["since"], timezone.utc).strftime("%Y-%m-%d %H:%MZ")
             if r.get("since") else "(no armed rounds yet)")
    lines = [f"ReasoningBank live A/B (#1489) since {since}, {r['days']:g} days; "
             f"open {r['open']}, excluded {r['excluded']} (skipped/infra), "
             f"round_start arm mismatches {r['round_arm_mismatches']}", ""]
    lines.append(f"{'arm':<14} {'rounds':>6} {'clus':>5} {'landed':>14} {'95% CI':>14} "
                 f"{'refusals/rd':>11} {'95% CI':>14} {'resolved/rd-h':>13} {'95% CI':>16}")
    for a, s in r["arms"].items():
        lr = f"{s['landed']}/{s['rounds']}={_fmt(s['landed_rate'], 2)}"
        lines.append(
            f"{a:<14} {s['rounds']:>6} {s['clusters']:>5} {lr:>14} {_ci(s['landed_ci'], 2):>14} "
            f"{_fmt(s['refusals_per_round'], 2):>11} {_ci(s['refusals_ci'], 2):>14} "
            f"{_fmt(s['resolved_per_round_hour']):>13} {_ci(s['resolved_per_round_hour_ci']):>16}")
    lines.append("")
    for k, d in r["diff"].items():
        lines.append(f"common_causes - control, {k}: {_fmt(d['diff'])} {_ci((d['lo'], d['hi']))}")
    for a, s in r["arms"].items():
        top = ", ".join(f"{c} {n}" for c, n in Counter(s["causes"]).most_common(5)) or "none"
        lines.append(f"refusal causes, {a}: {top}"
                     + (f"; {s['pending_settle']} landed, not yet settled"
                        if s["pending_settle"] else ""))
    lines += ["", f"n per arm for a {r['mde']:+.2f} landed-rate MDE (alpha .05, power .8, "
                  f"unclustered floor): {r['n_needed_per_arm']}",
              f"stop rule (>= {r['target_n']} done rounds per arm, or {r['max_days']:g} days): "
              f"{'MET' if r['stop_rule_met'] else 'not yet'}",
              f"verdict: {r['verdict']}"]
    return "\n".join(lines)


def _read_ledger(path: Path, contains: str | None = None) -> list[dict]:
    """The ledger's rows; with `contains`, only lines carrying that substring
    are parsed (the arm assignment runs on the event loop: ~0.2 s for the
    whole 19k-row ledger, a few ms for its implement rows)."""
    out = []
    with open(path) as fh:
        for line in fh:
            if contains is not None and contains not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m scripts.automod.reasoning_bank")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("ab-report", help="per-arm outcomes of the live A/B (#1489)")
    rep.add_argument("--ledger", type=Path, default=None)
    rep.add_argument("--since", default=None, help="ISO time; default: the first armed round")
    rep.add_argument("--min-n", type=int, default=AB_MIN_N_PER_ARM)
    rep.add_argument("--target-n", type=int, default=AB_TARGET_N_PER_ARM)
    rep.add_argument("--max-days", type=float, default=AB_MAX_DAYS)
    rep.add_argument("--mde", type=float, default=AB_MDE)
    rep.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.ledger is None:
        from scripts.automod import state as S
        args.ledger = S.LEDGER_PATH
    since = None
    if args.since:
        since = datetime.fromisoformat(args.since.replace("Z", "+00:00")).timestamp()
    r = ab_report(_read_ledger(args.ledger), since_ts=since, min_n=args.min_n,
                  target_n=args.target_n,
                  max_days=args.max_days, mde=args.mde)
    print(json.dumps(r, indent=2, default=str) if args.json else format_report(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
