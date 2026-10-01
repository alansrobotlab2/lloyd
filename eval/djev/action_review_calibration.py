"""Read the P10 `action_review` shadow corpus and answer the threshold question.

The seam has recorded worker tool calls since 2026-09-25T18:45:16Z — 57,325 of
them by the 2026-10-01 snapshot the ruling was read from — with
`ACTION_REVIEW.threshold is None`, so `warn` and `shadow` are behaviourally
identical and the rows have never been asked what they say. This is that
measurement: one pass over `~/.local/state/lloyd-djev/shadow.jsonl`, cross-tabbed
`on_task` label × `effective_tier` × `meta.source` × `actual.outcome`, every rate
printed beside the denominator it was divided by. It decides nothing and changes
nothing — it is the input to the branch decision recorded in
`knowledge/ai/action-review-threshold-measurement.md` and in `architecture/harness.md`
(P10). `tests/test_action_review_calibration.py` pins it.

WHY THE JOIN, AND WHY IT IS CHECKED RATHER THAN ASSUMED
-------------------------------------------------------
A row stores `meta.tool` and `meta.args_digest` and hashes the canvas into
`inputs_digest`, so the Bash command string a tier depends on is NOT in the row
(`app/harness/action_review.py` writes the meta, `app/djev_shadow.py` digests the
canvas). `effective_tier` needs that string: `app/harness/policy.py` asks
`app/harness/safety.py::bash_command_tier`, which is a pattern match over the
command. So the arguments are recovered from the session transcript named by
`meta.session_id`, matched on `meta.call_id`.

The recovery is then verified, not trusted: the transcript's
`function.arguments` are re-digested and compared to the row's `args_digest`.
They differ by exactly one key — the harness pops `summary` out of the arguments
before the turn sees them, so the transcript carries it and the row's digest does
not — and once it is stripped the digests agree. A row whose recovered arguments
do NOT digest-match is reported `digest_mismatch` and stays untiered, because that
transcript is not the call the row recorded. A row whose session file is gone, or
whose call record is gone, is likewise **unresolved**: tier is left `None` rather
than defaulted, because `effective_tier("Bash")` with no command answers 1, and an
analysis that defaults silently turns every lost transcript into evidence that the
positives are harmless.

RETENTION, WHICH IS THIS ANALYSIS'S OWN EXPIRY DATE
---------------------------------------------------
Nothing rotates `shadow.jsonl`: `app/djev_shadow.py` has no retention code and its
own comment says the size bound is "a guard against a runaway seam, not retention",
and `git grep -n "shadow.jsonl" -- scripts/` is empty. So the file is re-runnable
for as long as it is standing, at the cost of being unbounded. The join has a
shorter fuse: it needs `~/lloyd-data/sessions/*.json`, which the retention sweep
does bound. Re-run this inside both windows and say which one you are in —
`report()` prints the corpus's own first and last row timestamps, which is the
only span that means anything for a per-day rate. (`shadow.jsonl.bak-*` is a
PREFIX DUPLICATE of the live file. Never sum them; this module reads one path.)

The expiry date is not the whole story, because the ruling does not depend on the
transcripts. `~/obsidian/backlog/data/shadow.jsonl` is the committed extract the
ruling was read from: the seam's rows in file order, projected to the fields this
report reads, carrying `recovered_arguments` on the rows whose tier turns on them —
the 963 positives and the 6 rows that fail their digest. Point this module at it and
every figure in the ruling reproduces with no transcript open at all, and
`tests/test_action_review_calibration.py` runs exactly that command with an empty
transcripts directory on every pytest run. What the transcripts still decide is the
tier column for the *other* argument-dependent rows: over the extract with no
transcripts, 41,204 rows report `no_session_file` instead of a tier, so the
whole-corpus untierable count is a live-corpus figure and the positives' split is
not. Say which one you are quoting.

Run it::

    ~/lloyd/.venvs/lloyd/bin/python -m eval.djev.action_review_calibration
    ~/lloyd/.venvs/lloyd/bin/python -m eval.djev.action_review_calibration --json
    # the ruling, from committed bytes, with no transcripts:
    ~/lloyd/.venvs/lloyd/bin/python -m eval.djev.action_review_calibration \
        --shadow-log ~/obsidian/backlog/data/shadow.jsonl \
        --sessions-dir /nonexistent --quiet
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.data_root import ACCOUNT_HOME, production_data_root
from app.harness.action_review import SEAM, args_digest
from app.harness.policy import BASH_TOOL, SCHEDULE_STATE_TOOL, effective_tier

#: The one path a reader of this corpus means. `Path.home()` is what the
#: recorder writes (`app/djev_shadow.STATE_DIR`), but a reader that means
#: production on purpose uses `ACCOUNT_HOME`, which is read off passwd and so
#: survives a round's repointed `$HOME` — the reason this module prints a count
#: from inside an automod worktree at all.
SHADOW_LOG = ACCOUNT_HOME / ".local" / "state" / "lloyd-djev" / "shadow.jsonl"

#: The transcript root the join reads. `production_data_root()`, not
#: `app.paths.SESSIONS_DIR`: inside a round the latter resolves into the
#: worktree's own empty `.lloyd-data/`, and a scan of it returns zero rows that
#: read exactly like "the corpus is empty".
SESSIONS_DIR = production_data_root() / "sessions"

#: The tools whose tier is a property of the arguments rather than the name
#: (`app/harness/policy.py::effective_tier`). A row for one of these with no
#: recoverable arguments is UNRESOLVED, never tiered by default.
ARGS_DEPENDENT_TOOLS = frozenset({BASH_TOOL, SCHEDULE_STATE_TOOL})

#: Stripped before re-digesting recovered arguments. The harness pulls it out of
#: `args_dict` before the turn runs, so `meta.args_digest` excludes it while the
#: transcript keeps it — the whole of the difference between the two blobs.
SUMMARY_FIELD = "summary"

#: The field the committed extract (`~/obsidian/backlog/data/shadow.jsonl`, the
#: witness behind the ruling) carries on its `unrelated` rows: the arguments this
#: module's own join recovered, written into the file so the tier axis survives the
#: 30-day session-transcript window. When a row carries it, the row decides its own
#: tier and no transcript is opened for it — verified exactly as the join verifies,
#: and NOT repaired by a later transcript read if the verification fails: the whole
#: point of the join's order is that the one untierable positive is the row whose
#: arguments fail their digest.
RECOVERED_ARGS_FIELD = "recovered_arguments"

NO_LABEL = "none"


# ── the corpus ───────────────────────────────────────────────────────────────

def label_of(row: dict) -> str:
    """The reviewer's `on_task` answer, or `none` when the seam did not answer."""
    try:
        ans = (row.get("djev") or {}).get("answers") or {}
        value = (ans.get("on_task") or {}).get("value")
    except AttributeError:
        return NO_LABEL
    return str(value) if value else NO_LABEL


def label_mass_of(row: dict) -> float | None:
    try:
        return (row["djev"]["answers"]["on_task"] or {}).get("label_mass")
    except (KeyError, TypeError, AttributeError):
        return None


def iter_rows(path: Path, seam: str = SEAM) -> tuple[dict, int]:
    """`(rows, lines_read)`: the seam's rows, and the file's whole row count.

    The second number is a denominator this report keeps needing — the share of
    everything djev recorded that this seam accounts for — and reading the file
    once to get both is cheaper than a second pass for it.
    """
    rows: list[dict] = []
    lines = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("seam") == seam:
                rows.append(row)
    return rows, lines


# ── recovering the arguments a tier needs ─────────────────────────────────────

def call_arguments(path: Path, want: set[str]) -> dict[str, dict]:
    """`call_id -> arguments` for the wanted ids, from one session transcript.

    `tool_calls[].call_id` is what `meta.call_id` matches — the wire id the
    reviewer recorded at `tool_call` time is the id the assistant message keeps.
    A missing file or an unparseable transcript returns `{}`: the caller counts
    those rows unresolved, which is the point.
    """
    found: dict[str, dict] = {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return found
    for msg in (doc.get("messages") or []):
        for call in (msg.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            cid = str(call.get("call_id") or call.get("id") or "")
            if cid not in want:
                continue
            fn = call.get("function") or {}
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = None
            if isinstance(args, dict):
                found[cid] = args
    return found


def matches_digest(args: dict, want: str) -> bool:
    """Whether recovered arguments ARE the call the row recorded.

    One key apart, always: `summary` reaches the transcript but not the digest.
    Strip it, then compare with the recorder's own `args_digest`, imported rather
    than restated so the check cannot drift from the thing it checks. A row whose
    recovered arguments do not match is untiered by the caller rather than tiered
    on the strength of a plausible-looking command.
    """
    bare = {k: v for k, v in args.items() if k != SUMMARY_FIELD}
    return args_digest(bare) == want


@dataclass
class Row:
    """One seam row, projected to the four axes the report cross-tabs."""
    ts: float
    label: str
    source: str
    tool: str
    outcome: str
    tier: int | None
    tier_reason: str
    label_mass: float | None = None


def resolve_tier(tool: str, args: dict | None) -> tuple[int | None, str]:
    """Tier, or `None` with the reason it could not be answered.

    Name-only tools answer from `effective_tier(name)` — that is its documented
    contract for a caller that has no arguments. An argument-dependent tool gets
    `None`, never the default answer, because the default for `Bash` is tier 1
    and a lost transcript would otherwise be counted as a harmless call.
    """
    if tool in ARGS_DEPENDENT_TOOLS:
        if args is None:
            return None, "unresolved"
        return effective_tier(tool, args), "args"
    return effective_tier(tool), "name"


def load_corpus(shadow_log: Path = SHADOW_LOG, sessions_dir: Path = SESSIONS_DIR,
                *, progress=None) -> tuple[list[Row], dict]:
    """Tier every seam row, recovering arguments from transcripts where needed.

    Returns the rows and an integrity block: how many transcripts were read, how
    many rows could not be tiered and why, and the digest-match count that says
    the join is identifying real calls rather than plausible ones.
    """
    rows, lines_read = iter_rows(shadow_log)

    # Two sources for an argument-dependent row's arguments, in this order.
    # A row of the committed extract carries its own (`recovered_arguments`) and
    # is verified from the row alone, so the ruling's tier axis needs no
    # transcript and outlives the 30-day gzip. Only a row without one justifies
    # opening the transcript its meta names — and Bash is 73% of the corpus, so
    # those are grouped by session and each file read once.
    want: dict[str, set[str]] = collections.defaultdict(set)
    extract: dict[tuple[str, str], dict] = {}
    for r in rows:
        meta = r.get("meta") or {}
        if meta.get("tool") not in ARGS_DEPENDENT_TOOLS:
            continue
        key = (str(meta.get("session_id") or ""), str(meta.get("call_id") or ""))
        if isinstance(r.get(RECOVERED_ARGS_FIELD), dict):
            extract[key] = r[RECOVERED_ARGS_FIELD]
        else:
            want[key[0]].add(key[1])

    args_by_session: dict[str, dict[str, dict]] = {}
    for i, (sid, ids) in enumerate(sorted(want.items()), start=1):
        if progress and i % 250 == 0:
            progress(f"transcripts {i}/{len(want)}")
        args_by_session[sid] = call_arguments(sessions_dir / f"{sid}.json", ids)

    out: list[Row] = []
    integrity = collections.Counter()
    integrity["shadow_lines"] = lines_read
    integrity["seam_rows"] = len(rows)
    integrity["transcripts_read"] = len(args_by_session)
    for r in rows:
        meta = r.get("meta") or {}
        tool = str(meta.get("tool") or "")
        sid = str(meta.get("session_id") or "")
        cid = str(meta.get("call_id") or "")
        digest = str(meta.get("args_digest") or "")
        reason = "name"
        if (sid, cid) in extract:
            # Carried by the extract: verified against the row's own digest, and
            # a failure here is the row's answer, not a gap to go hunting a
            # second source for. Repairing it from a transcript would turn the
            # one untierable positive into a tier-1 positive and move the
            # published split (961 / 1 / 1) by silently disagreeing with itself.
            args = extract[(sid, cid)]
            if matches_digest(args, digest):
                integrity["extract_match"] += 1
            else:
                args, reason = None, "digest_mismatch"
                integrity["digest_mismatch"] += 1
        else:
            args = args_by_session.get(sid, {}).get(cid)
            if tool in ARGS_DEPENDENT_TOOLS:
                if args is None:
                    path = sessions_dir / f"{sid}.json"
                    reason = ("no_call_record" if path.exists() else "no_session_file")
                    integrity[reason] += 1
                elif not matches_digest(args, digest):
                    args, reason = None, "digest_mismatch"
                    integrity["digest_mismatch"] += 1
                else:
                    integrity["digest_match"] += 1
        tier, why = resolve_tier(tool, args)
        reason = why if tier is not None else reason
        if tier is None:
            integrity["untierable"] += 1
        out.append(Row(ts=float(r.get("ts") or 0.0), label=label_of(r),
                       source=str(meta.get("source") or ""), tool=tool,
                       outcome=str(((r.get("actual") or {}).get("outcome")) or ""),
                       tier=tier, tier_reason=reason,
                       label_mass=label_mass_of(r)))
    return out, dict(integrity)


# ── the measurement ───────────────────────────────────────────────────────────

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cross_tab(rows: list[Row]) -> list[dict]:
    """label × tier × source × outcome, most populous first. Tier `None` prints
    as `unresolved` — a bucket of its own, never folded into a tier."""
    tally: collections.Counter = collections.Counter()
    for r in rows:
        tally[(r.label, "unresolved" if r.tier is None else str(r.tier),
               r.source, r.outcome)] += 1
    return [{"label": l, "tier": t, "source": s, "outcome": o, "n": n}
            for (l, t, s, o), n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))]


def per_day(rows: list[Row], *, of_label: str | None = None) -> dict[str, int]:
    """Calendar-day counts over the SEAM'S OWN span.

    The first `action_review` row, not the first row in the file: the file leads
    with other seams from before this one existed, and dividing by that span
    understates the rate by however many days of silence it includes.
    """
    days: dict[str, int] = collections.Counter()
    for r in rows:
        if of_label is None or r.label == of_label:
            days[datetime.fromtimestamp(r.ts, timezone.utc).strftime("%Y-%m-%d")] += 1
    return dict(sorted(days.items()))


def report(rows: list[Row], integrity: dict, *, source: Path = SHADOW_LOG) -> dict:
    """Everything the branch decision is made of, with denominators attached."""
    total = len(rows)
    labels: collections.Counter = collections.Counter(r.label for r in rows)
    outcomes: collections.Counter = collections.Counter(r.outcome for r in rows)
    ts = [r.ts for r in rows if r.ts]
    span_days = (max(ts) - min(ts)) / 86400 if len(ts) > 1 else 0.0

    unrelated = [r for r in rows if r.label == "unrelated"]
    tiers: collections.Counter = collections.Counter(
        "unresolved" if r.tier is None else str(r.tier) for r in unrelated)
    tiered = [r for r in unrelated if r.tier is not None]
    hi = [r for r in tiered if r.tier >= 2]
    ran = [r for r in unrelated if r.outcome == "ran"]
    # The rate a `warn` would fire at: a call that ran AND was judged off-task.
    # A denied or errored call already cost the worker nothing.
    would_fire = [r for r in unrelated if r.outcome == "ran"]
    hook = [r for r in rows if r.outcome == "denied_by_hook"]
    hook_labels: collections.Counter = collections.Counter(r.label for r in hook)

    # The denominator that settles branch (b): the positives are only "topical
    # drift across the ordinary tools" if their rate per tool is flat. A tool
    # whose positive rate is an outlier is a tool the reviewer is failing, not a
    # topic it drifted off, and that would be a different decision.
    tool_total: collections.Counter = collections.Counter(r.tool for r in rows)
    tool_unrel: collections.Counter = collections.Counter(r.tool for r in unrelated)
    by_tool = {t: {"n": n, "of_unrelated": len(unrelated) or None,
                   "of_tool_rows": tool_total[t] or None,
                   "rate": round(n / tool_total[t], 5) if tool_total[t] else None}
               for t, n in sorted(tool_unrel.items(), key=lambda kv: (-kv[1], kv[0]))}

    daily_rows = per_day(rows)
    daily_unrel = per_day(rows, of_label="unrelated")
    return {
        "corpus": {
            "path": str(source),
            "bytes": source.stat().st_size if source.exists() else 0,
            "shadow_lines": integrity.get("shadow_lines", 0),
            "seam_rows": total,
            "seam_share": total / integrity["shadow_lines"] if integrity.get("shadow_lines") else None,
            "first_row": iso(min(ts)) if ts else None,
            "last_row": iso(max(ts)) if ts else None,
            "span_days": round(span_days, 3),
        },
        "labels": {k: {"n": v, "of_seam_rows": total or None}
                   for k, v in sorted(labels.items(), key=lambda kv: -kv[1])},
        "outcomes": {k: {"n": v, "of_seam_rows": total or None}
                     for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1])},
        "cross_tab": cross_tab(rows),
        "tier_resolution": {
            "seam_rows": total,
            "untierable": integrity.get("untierable", 0),
            "reasons": {k: integrity[k] for k in
                        ("no_session_file", "no_call_record", "digest_mismatch")
                        if integrity.get(k)},
            "integrity": {k: integrity[k] for k in
                          ("extract_match", "digest_match", "digest_mismatch",
                           "transcripts_read")
                          if integrity.get(k) is not None},
        },
        "unrelated": {
            "n": len(unrelated), "of_seam_rows": total or None,
            "per_day_over_seam_span": round(len(unrelated) / span_days, 1) if span_days else None,
            "would_fire_per_day": round(len(would_fire) / span_days, 1) if span_days else None,
            "ran": {"n": len(ran), "of_unrelated": len(unrelated) or None},
            "tiers": {k: {"n": v, "of_unrelated": len(unrelated) or None,
                          "of_tiered_unrelated": len(tiered) or None}
                      for k, v in sorted(tiers.items(), key=lambda kv: -kv[1])},
            "tier_ge_2": {"n": len(hi), "of_unrelated": len(unrelated) or None,
                          "of_seam_rows": total or None,
                          "per_day_over_seam_span": round(len(hi) / span_days, 1) if span_days else None},
            "per_day": daily_unrel,
            "by_tool": by_tool,
            "per_day_span_days": round(span_days, 3),
        },
        "per_day_rows": daily_rows,
        "hook_denials": {
            "n": len(hook), "of_seam_rows": total or None,
            "by_label": {k: {"n": v, "of_hook_denials": len(hook) or None}
                         for k, v in hook_labels.items()},
            "overlap_with_unrelated": hook_labels.get("unrelated", 0),
        },
    }


def format_report(rep: dict) -> str:
    """Text form. Every rate prints its denominator beside it, in words.

    A percentage with no denominator is how the item this replaced got its wrong
    rate: `92/day` was the file's span, not the seam's.
    """
    c = rep["corpus"]
    u = rep["unrelated"]
    lines = [
        f"corpus            {c['path']}",
        f"                  {c['bytes']:,} bytes, {c['shadow_lines']:,} rows all seams, "
        f"{c['seam_rows']:,} action_review "
        f"({100.0 * c['seam_share']:.1f}% of {c['shadow_lines']:,})"
        if c["seam_share"] is not None else "corpus            (unreadable)",
        f"seam's own span   {c['first_row']} → {c['last_row']}  "
        f"({c['span_days']} days; the file's head row is an earlier seam)",
        "",
        "on_task label     " + " · ".join(
            f"{k} {v['n']:,}/{rep['corpus']['seam_rows']:,} "
            f"({100.0 * v['n'] / v['of_seam_rows']:.2f}%)"
            for k, v in rep["labels"].items()),
        "outcome           " + " · ".join(
            f"{k} {v['n']:,}/{rep['corpus']['seam_rows']:,}"
            for k, v in rep["outcomes"].items()),
        "",
        f"tier resolution   {rep['tier_resolution']['untierable']:,} of "
        f"{rep['tier_resolution']['seam_rows']:,} seam rows untierable: "
        + (", ".join(f"{k} {v:,}" for k, v in rep["tier_resolution"]["reasons"].items())
           or "no reasons"),
        "join integrity    " + ", ".join(
            f"{k} {v:,}" for k, v in rep["tier_resolution"]["integrity"].items()),
        "",
        f"unrelated         {u['n']:,}/{u['of_seam_rows']:,} seam rows = "
        f"{u['per_day_over_seam_span']}/day over the seam's own span of "
        f"{rep['corpus']['span_days']} days",
        f"                  of those, ran {u['ran']['n']:,}/{u['n']:,} = the calls a "
        f"`warn` would have interrupted = {u['would_fire_per_day']}/day",
        f"                  tier ≥ 2 (durable-external): {u['tier_ge_2']['n']:,}/"
        f"{u['n']:,} unrelated ({u['tier_ge_2']['per_day_over_seam_span']}/day) = "
        f"{u['tier_ge_2']['n']:,}/{u['of_seam_rows']:,} of all seam rows",
        "                  tiers " + " · ".join(
            f"{k} {v['n']:,}/{u['n']:,}" for k, v in u["tiers"].items()),
        "                  by tool (rate per that tool's rows, the denominator beside it): "
        + " · ".join(f"{t} {v['n']:,}/{v['of_tool_rows']:,}={100.0 * v['n'] / v['of_tool_rows']:.2f}%"
                     for t, v in list(u["by_tool"].items())[:8]),
        "",
        "label × tier × source × outcome (top 25 of "
        f"{len(rep['cross_tab'])} cells, n over {rep['corpus']['seam_rows']:,} rows)",
    ]
    for cell in rep["cross_tab"][:25]:
        lines.append(f"  {cell['label']:<11} tier={cell['tier']:<10} "
                     f"{cell['outcome']:<18} {cell['source']:<24} {cell['n']:,}")
    lines += ["", "unrelated per calendar day"]
    lines.append("  " + " · ".join(f"{d} {n}" for d, n in u["per_day"].items()))
    h = rep["hook_denials"]
    lines += [
        "",
        f"deterministic hook denials: {h['n']:,}/{rep['corpus']['seam_rows']:,} seam rows; "
        "djev judged " + (" · ".join(
            f"{v['n']} {k}" for k, v in h["by_label"].items()) or "none")
        + f" — overlap with `unrelated` is {h['overlap_with_unrelated']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shadow-log", type=Path, default=SHADOW_LOG)
    ap.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.shadow_log.exists():
        print(f"no shadow log at {args.shadow_log}", flush=True)
        return 1
    rows, integrity = load_corpus(
        args.shadow_log, args.sessions_dir,
        progress=None if args.quiet else (lambda m: print(f"… {m}", file=__import__("sys").stderr)))
    rep = report(rows, integrity, source=args.shadow_log)
    print(json.dumps(rep, indent=1) if args.json else format_report(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
