"""Fitness-driven intake: the measurement chooses work.

`scripts/automod/fitness.py` says whether a behaviour bench moved outside its
own noise. Until this module nothing did anything with that but print it. After
a nightly fitness run, `run()` files AT MOST ONE backlog item a day, class
`improvement`, `metric: <bench>`, against

1. the bench that most REGRESSED outside its noise floor (`fitness.moves`), or
2. when nothing moved and `file_lowest` allows it, the bench furthest below its
   own trailing best — a slow drift no 7-day delta sees.

Rules, each one a way this could have become one more inflow nothing bounds:

- **One item per day**, counted off the ledger (`fitness_intake` events that
  filed, and `measured_close` events that filed a revert proposal — the same
  `spawned-by-fitness` budget).
- **One open item per bench.** While an open item names the bench — in its
  `metric:` or in the `fitness_bench:` key every item filed here carries, which
  survives a triage that re-classes the item — nothing is filed for it.
- **Never without a floor.** A bench flagged `too-noisy`, or with no measured
  noise floor, is never filed against: there is nothing to say the number moved.
- **The acceptance clause is the measured one**: "`<bench>` is above <baseline +
  noise floor> on a run after landing". `scripts/automod/measured_close.py` is
  what later reads that measurement.
- **Every decision is journaled** as one `fitness_intake` ledger event: `filed`
  with the item id, or `held` with the reason.

The item is tagged `spawned-by-fitness`. The `spawned-by-` prefix is what gives
it expiry and the scorecard gauge (`backlog.loop_spawn_tag`); like an owed-check
follow-up and a review finding it is NOT in `QUARANTINE_TAGS`, because single
triage is exactly the pass that should decide whether a measured regression is
worth a round.

Ships off: `workers.sources.fitness.file_items: false`. The hook is
`workers/sources/fitness.py::execute`, off the event loop, after a run that
measured something. `choose` and `render_item` are pure.

    python -m scripts.automod.fitness_intake report     # what it WOULD do, writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The tag on every item this module (and measured_close's revert proposal) files.
FITNESS_SPAWN_TAG = "spawned-by-fitness"
#: Front-matter key naming the bench an item was filed against. `metric:` is
#: triage's to rewrite (and it drops it for any class but `improvement`); this
#: key is what keeps "one open item per bench" true afterwards.
FM_BENCH = "fitness_bench"
INTAKE_EVENT = "fitness_intake"
#: The measured-close event; read here only to count its revert proposals
#: against the same one-a-day budget.
CLOSE_EVENT = "measured_close"

DAY_S = 86400.0
#: At most one filing per this many seconds.
FILE_EVERY_S = DAY_S
#: How far back "its own trailing best" looks.
TRAILING_DAYS = 30.0

RERUN = "python -m scripts.automod.fitness run --once --only {bench}"

#: Where a bench's failing cases can be read, per manifest bench — checked
#: against each bench's own writer on 2026-11-22. Every `{scratch}` artifact is
#: removed when the run ends, so those need `--keep-scratch` on the re-run (the
#: run prints nothing about the directory; the report's `scratch` key carries
#: it). A bench absent from this table reads `no per-case rows`, and
#: `tests/test_fitness_intake.py` fails until a new manifest bench is added here.
CASE_ROWS: dict[str, str] = {
    "refusal_detour": (
        "no per-case rows: the manifest command writes the summary only. A hand run "
        "of `eval/refusal_detour_baseline.py --rows --out <file>.json` writes one row "
        "per refusal to `<file>.rows.jsonl`"),
    "skill_match": (
        "per-turn rows under `arms.lexical.turns` of `{scratch}/skill-match.json` "
        "(re-run with `--keep-scratch`)"),
    "skill_activation": (
        "per-case rows under `rows` of `{scratch}/skill-activation.json` "
        "(re-run with `--keep-scratch`)"),
    "tool_choice": (
        "per-query rows under `records` of the newest "
        "`{data_root}/eval/baselines/tool-choice/fitness-*.json` (kept after the run)"),
    "injection_canary": (
        "one row per episode in `{scratch}/canary-rows.jsonl` "
        "(re-run with `--keep-scratch`)"),
    "stale_fact": (
        "per-trial rows under `trials` of `{scratch}/stale-fact.json` "
        "(re-run with `--keep-scratch`)"),
    "memory_history": (
        "per-question rows under `rows` of `{scratch}/memeval/lloydmemeval-fitness.json` "
        "(re-run with `--keep-scratch`)"),
}
NO_CASE_ROWS = "no per-case rows"


def case_rows(bench: str) -> str:
    return CASE_ROWS.get(bench, NO_CASE_ROWS)


# ── readers shared with measured_close ───────────────────────────────────────

def bench_of(fm: Mapping[str, Any]) -> str:
    """The bench an item names: `fitness_bench`, else its `metric`."""
    from scripts.automod.item_class import clean_metric
    return clean_metric(fm.get(FM_BENCH)) or clean_metric(fm.get("metric"))


def open_bench_items(fronts: Iterable[tuple[int, Mapping[str, Any]]]) -> dict[str, list[int]]:
    """`{bench: [ids]}` over the OPEN items' front matter handed in."""
    out: dict[str, list[int]] = {}
    for item_id, fm in fronts:
        bench = bench_of(fm or {})
        if bench:
            out.setdefault(bench, []).append(int(item_id))
    return out


def last_filing(events: Iterable[Mapping[str, Any]]) -> dict | None:
    """The newest ledger event that put a `spawned-by-fitness` item on the board."""
    newest = None
    for e in events:
        ts = e.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        filed = ((e.get("event") == INTAKE_EVENT and e.get("decision") == "filed")
                 or (e.get("event") == CLOSE_EVENT and e.get("follow_up")))
        if filed and (newest is None or float(ts) > float(newest["ts"])):
            newest = {"ts": float(ts),
                      "item_id": e.get("item_id") if e.get("event") == INTAKE_EVENT
                      else e.get("follow_up")}
    return newest


def _floor(noise: Mapping[str, Any], bench: str) -> tuple[str, float | None]:
    entry = noise.get(bench) if isinstance(noise.get(bench), Mapping) else {}
    spread = entry.get("spread")
    ok = isinstance(spread, (int, float)) and not isinstance(spread, bool)
    return str(entry.get("status") or "unmeasured"), (float(spread) if ok else None)


def _row_facts(row: Mapping[str, Any] | None) -> dict:
    row = row or {}
    return {k: row.get(k) for k in ("value", "n", "ci_low", "ci_high", "commit",
                                    "recorded_at", "run_id", "ts")}


# ── the choice ───────────────────────────────────────────────────────────────

def below_best(rows: Sequence[dict], bench: str, noise: Mapping[str, Any], *,
               days: float = TRAILING_DAYS) -> dict | None:
    """How far the bench's latest value sits below its own best of the last
    `days`, or None when that gap is not outside the noise floor (same two
    bounds as `fitness.bench_delta`: larger than the measured spread, and the
    latest row's interval does not cover the best value)."""
    from scripts.automod import fitness as F
    status, spread = _floor(noise, bench)
    mine = F.measured_rows(rows, bench)
    if status != "ok" or spread is None or len(mine) < 2:
        return None
    latest = mine[-1]
    higher = bool(latest.get("higher_is_better", True))
    cutoff = float(latest["ts"]) - days * DAY_S
    window = [r for r in mine[:-1] if float(r.get("ts") or 0.0) >= cutoff]
    if not window:
        return None
    best = (max if higher else min)(window, key=lambda r: float(r["value"]))
    gap = float(best["value"]) - float(latest["value"])
    gap = gap if higher else -gap
    lo, hi = latest.get("ci_low"), latest.get("ci_high")
    covered = (isinstance(lo, (int, float)) and isinstance(hi, (int, float))
               and lo <= float(best["value"]) <= hi)
    if gap <= spread or covered:
        return None
    return {"bench": bench, "kind": "below-best", "gap": round(gap, 6),
            "latest": _row_facts(latest), "reference": _row_facts(best),
            "higher_is_better": higher, "noise_spread": spread}


def choose(now: float, *, deltas: Sequence[Mapping[str, Any]], rows: Sequence[dict],
           noise: Mapping[str, Any], open_benches: Mapping[str, Sequence[int]],
           last_filed: Mapping[str, Any] | None = None,
           file_lowest: bool = False) -> dict:
    """What to do after one fitness run. Pure.

    Returns `{"decision": "file", "bench", "kind", "latest", "reference",
    "higher_is_better", "noise_spread", "skipped"}` or `{"decision": "held",
    "why", "skipped"}`; `skipped` names each candidate passed over and why.
    """
    skipped: list[str] = []
    if last_filed and now - float(last_filed.get("ts") or 0.0) < FILE_EVERY_S:
        return {"decision": "held", "skipped": skipped,
                "why": f"one item per day: #{last_filed.get('item_id')} was filed "
                       f"{(now - float(last_filed['ts'])) / 3600.0:.1f} h ago"}
    ref_rows = {(r.get("bench"), r.get("run_id")): r for r in rows if r.get("kind") == "bench"}

    def with_interval(bench: str, facts: Mapping[str, Any]) -> dict:
        # `bench_delta`'s reference carries no interval; the row it came from does.
        return {**_row_facts(ref_rows.get((bench, facts.get("run_id")))), **{
            k: v for k, v in facts.items() if v is not None}}

    candidates: list[dict] = []
    for d in deltas:
        bench = str(d.get("bench") or "")
        if d.get("outside_noise") is True and d.get("direction") == "regressed":
            status, spread = _floor(noise, bench)
            if status != "ok" or spread is None:   # `moves` already excludes these
                skipped.append(f"{bench}: {status}, no usable noise floor")
                continue
            latest = d.get("latest") or {}
            candidates.append({
                "bench": bench, "kind": "regressed", "gap": abs(float(d.get("delta") or 0.0)),
                "latest": with_interval(bench, latest),
                "reference": with_interval(bench, d.get("reference") or {}),
                "higher_is_better": bool((ref_rows.get((bench, latest.get("run_id"))) or {})
                                         .get("higher_is_better", True)),
                "noise_spread": spread})
        elif d.get("latest") and d.get("noise_status") != "ok":
            skipped.append(f"{bench}: {d.get('noise_status')}, never filed against")
    if not candidates and file_lowest:
        for d in deltas:
            found = below_best(rows, str(d.get("bench") or ""), noise)
            if found:
                candidates.append(found)
    # Most regressed first. Every value is a share in 0..1, so the absolute gap
    # compares across benches; dividing by the spread would let a deterministic
    # bench (spread 0) win every night.
    candidates.sort(key=lambda c: (-c["gap"], c["bench"]))
    for c in candidates:
        holders = list(open_benches.get(c["bench"]) or [])
        if holders:
            skipped.append(f"{c['bench']}: open item #{holders[0]} already names it")
            continue
        return {"decision": "file", **c, "skipped": skipped}
    if candidates:
        why = "every bench that moved already has an open item"
    elif file_lowest:
        why = "no bench regressed outside its noise floor, and none is below its trailing best"
    else:
        why = "no bench regressed outside its noise floor"
    return {"decision": "held", "why": why, "skipped": skipped}


# ── the item ─────────────────────────────────────────────────────────────────

def _num(x: Any, spec: str = ".3f") -> str:
    return "—" if not isinstance(x, (int, float)) or isinstance(x, bool) else f"{x:{spec}}"


def _measurement(facts: Mapping[str, Any]) -> str:
    return (f"{_num(facts.get('value'))} (n={facts.get('n') if facts.get('n') is not None else '—'}, "
            f"95% CI [{_num(facts.get('ci_low'))}, {_num(facts.get('ci_high'))}]) "
            f"at `{str(facts.get('commit') or 'unknown')[:8]}` on "
            f"{str(facts.get('recorded_at') or 'an unrecorded date')[:10]}")


def target_of(choice: Mapping[str, Any]) -> tuple[str, float]:
    """`("above"|"below", threshold)`: the baseline value moved by the noise floor
    in the bench's good direction. The baseline is the value that prompted the
    item — the regressed one."""
    value = float(choice["latest"]["value"])
    spread = float(choice.get("noise_spread") or 0.0)
    if choice.get("higher_is_better", True):
        return "above", round(value + spread, 6)
    return "below", round(value - spread, 6)


def acceptance_clause(choice: Mapping[str, Any]) -> str:
    word, threshold = target_of(choice)
    return (f"`{choice['bench']}` is {word} {_num(threshold)} (the {_num(choice['latest']['value'])} "
            f"measured at `{str(choice['latest'].get('commit') or 'unknown')[:8]}`, "
            f"{'plus' if word == 'above' else 'minus'} its noise floor "
            f"{_num(choice.get('noise_spread'))}) on a `fitness run` after landing")


def render_item(choice: Mapping[str, Any], landings: Sequence[Mapping[str, Any]] = ()
                ) -> tuple[str, str]:
    """`(name, body)` for the item a `file` decision asks for. Pure."""
    bench = str(choice["bench"])
    latest, ref = choice["latest"], choice["reference"]
    better = "higher" if choice.get("higher_is_better", True) else "lower"
    if choice.get("kind") == "below-best":
        name = (f"Recover `{bench}`: {_num(latest.get('value'))} against its "
                f"{TRAILING_DAYS:g}-day best {_num(ref.get('value'))}")
        lead = (f"Behaviour bench `{bench}` sits below its own trailing best, outside its "
                f"noise floor ({better} is better).")
        ref_label = f"Best of the last {TRAILING_DAYS:g} days"
    else:
        name = (f"Recover `{bench}`: {_num(ref.get('value'))} → {_num(latest.get('value'))} "
                f"outside its noise floor")
        lead = (f"Behaviour bench `{bench}` regressed outside its noise floor "
                f"({better} is better).")
        ref_label = "Before"
    shown = list(landings)[:20]
    lines = [lead, "",
             f"- Now: {_measurement(latest)}",
             f"- {ref_label}: {_measurement(ref)}",
             f"- Noise floor (max − min over repeats at one commit): "
             f"{_num(choice.get('noise_spread'))}",
             "", "## Landings between the two measurements", ""]
    if shown:
        lines += [f"- `{str(e.get('commit') or 'unknown')[:8]}` {e.get('round_id') or ''} "
                  f"{str(e.get('title') or '').strip()}".rstrip() for e in shown]
        if len(landings) > len(shown):
            lines.append(f"- … and {len(landings) - len(shown)} more")
    else:
        lines.append("None recorded in the ledger between the two measurements: the move "
                     "may come from the vault, the corpus or the engine, not from a landing.")
    lines += ["", "## Re-run", "", "    " + RERUN.format(bench=bench), "",
              "An engine bench refuses unless the worker pool is paused; never `--force` "
              "from a round.",
              "", "## Failing cases", "", case_rows(bench),
              "", "## Acceptance clauses", "", f"1. {acceptance_clause(choice)}",
              "", f"Filed by fitness intake (`scripts/automod/fitness_intake.py`) from run "
                  f"`{latest.get('run_id') or 'unknown'}`."]
    return name, "\n".join(lines)


def landings_for(choice: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> list[dict]:
    """The landings between the two measurements: the `fitness_move` event's own
    list when `record_moves` wrote one for this pair of runs, else the same
    reading of the ledger."""
    from scripts.automod import fitness as F
    latest, ref = choice["latest"], choice["reference"]
    for e in reversed(list(events)):
        if (e.get("event") == F.MOVE_EVENT and e.get("bench") == choice["bench"]
                and e.get("to_run") == latest.get("run_id")
                and e.get("from_run") == ref.get("run_id")):
            return list(e.get("landings") or [])
    try:
        return F.landings_between(events, float(ref["ts"]), float(latest["ts"]))
    except (TypeError, ValueError):
        return []


# ── the board (the only impure part) ─────────────────────────────────────────

def open_fronts() -> list[tuple[int, dict]]:
    from scripts.automod import backlog as B
    return [(i.id, B.front_matter_of(i.path) or {}) for i in B.open_items(None)]


def file_item(name: str, body: str, *, bench: str, cls: str = "improvement") -> int:
    """Create the draft and stamp its class keys. `metric` only for `improvement`
    (`item_class._fields` keeps it for no other class); `fitness_bench` always."""
    from scripts.automod import backlog as B
    item = B.new_item(name, body, status="draft", tags=[FITNESS_SPAWN_TAG])
    B.update_frontmatter(item.path, {"class": cls, FM_BENCH: bench,
                                     "metric": bench if cls == "improvement" else None})
    return int(item.id)


def run(*, now: float | None = None, file_lowest: bool = False, dry_run: bool = False,
        rows: Sequence[dict] | None = None, noise: Mapping[str, Any] | None = None,
        names: Sequence[str] | None = None,
        events: Sequence[Mapping[str, Any]] | None = None,
        fronts: Iterable[tuple[int, Mapping[str, Any]]] | None = None,
        filer: Callable[..., int] | None = None,
        append: Callable[[dict], None] | None = None) -> dict:
    """Decide, file at most one item, journal the decision. Returns the journal row.

    Every store is injectable; with none given it reads the fitness series, the
    noise file, the manifest, the ledger and the open board.
    """
    from scripts.automod import fitness as F, state as S
    now = time.time() if now is None else float(now)
    rows = F.read_rows() if rows is None else rows
    noise = F.read_noise() if noise is None else noise
    events = list(S.ledger_rows()) if events is None else list(events)
    deltas = F.deltas(now, rows=rows, noise=noise, names=names)
    choice = choose(now, deltas=deltas, rows=rows, noise=noise,
                    open_benches=open_bench_items(open_fronts() if fronts is None else fronts),
                    last_filed=last_filing(events), file_lowest=file_lowest)
    entry: dict[str, Any] = {"event": INTAKE_EVENT, "decision": choice["decision"],
                             "skipped": choice.get("skipped") or []}
    if choice["decision"] == "file":
        name, body = render_item(choice, landings_for(choice, events))
        word, threshold = target_of(choice)
        entry.update({"bench": choice["bench"], "kind": choice["kind"],
                      "value": choice["latest"].get("value"),
                      "reference_value": choice["reference"].get("value"),
                      "noise_spread": choice.get("noise_spread"),
                      "target": {"direction": word, "value": threshold},
                      "from_run": choice["reference"].get("run_id"),
                      "to_run": choice["latest"].get("run_id"), "name": name})
        if dry_run:
            entry.update({"decision": "would-file", "body": body})
            return entry
        entry["decision"] = "filed"
        entry["item_id"] = (filer or file_item)(name, body, bench=choice["bench"])
    else:
        entry["why"] = choice["why"]
        if dry_run:
            return entry
    (append or S.append_event)(entry)
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.automod.fitness_intake",
        description="What fitness intake would file after the latest run. Writes nothing.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--file-lowest", action="store_true")
    rp.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    out = run(dry_run=True, file_lowest=args.file_lowest)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if out["decision"] == "would-file":
        print(f"{stamp} would file: {out['name']}\n\n{out['body']}")
    else:
        print(f"{stamp} held: {out['why']}")
    for s in out.get("skipped") or []:
        print(f"  passed over — {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
