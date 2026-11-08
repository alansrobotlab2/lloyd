"""One-command census of the surfaces real rounds actually reach (#3930 step 1).

    .venvs/lloyd/bin/python -m scripts.automod.recovery_census --last 50

The item's method — "enumerate the non-git state real rounds touch, do not invent
an effect vocabulary" — cannot run over `changed_paths`, which are repo-relative
diff paths by construction: the non-git effect is a *consequence* of applying the
change, so it is never a path in the diff. What the durable record carries is the
declaration the gate made from the checked-in table
(`scripts/automod/recovery.py::SURFACES`), written onto every `rung=recovery`
ledger event. So the census regenerates two things from the ledger and prints
them together:

  * the surface → measurable digest → available inverse table itself, which is
    the checked-in contract, always printed even over a window with no records
    (an empty table would read as "nothing to recover" rather than "nothing seen
    yet");
  * how many rounds in the window DECLARED each surface, plus the three-way
    split #3930 step 6 publishes — empty contract / declared / unrecoverable —
    which is the number that decides whether `unrecoverable` is becoming a
    rubber stamp.

Stdlib, read-only, and it prints `none` out loud rather than omitting a column:
a surface with no inverse is the honest category the video insists on, and a
report that renders it as a blank cell is a report that hides it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from collections.abc import Iterable

from scripts.automod import recovery as R
from scripts.automod import state as S


def window_rounds(events: Iterable[dict], last: int) -> list[str]:
    """The `last` most recent round ids in `events` that carry a gate event.

    Ordered oldest → newest. A round with several gate events (a re-gate) counts
    once, at its latest one, because the census counts ROUNDS.
    """
    latest: dict[str, float] = {}
    for ev in events:
        if ev.get("event") != "gate" or not ev.get("round_id"):
            continue
        ts = float(ev.get("ts") or 0.0)
        rid = str(ev["round_id"])
        if ts >= latest.get(rid, -1.0):
            latest[rid] = ts
    ordered = sorted(latest, key=lambda rid: latest[rid])
    return ordered[-last:] if last > 0 else ordered


def recovery_records(events: Iterable[dict], round_ids: Iterable[str]) -> dict[str, dict]:
    """One recovery record per round in the window: the NEWEST gate event that has one.

    Resolved by `ts`, not by row position, because the merged ledger is
    archives-then-live (`state.ledger_rows_merged` puts the live rows last for
    last-wins readers) and a re-gate's older row therefore arrives AFTER its newer
    one. Taking the last row in list order would credit the superseded record, and
    the split this feeds is the number the acceptance publishes.

    A round whose rungs were reused, or that was gated before this rung existed,
    has no record — that absence is reported as a count (`rounds with a recovery
    record: N of M`), never folded into the empty-contract bucket.
    """
    wanted = set(round_ids)
    out: dict[str, dict] = {}
    newest: dict[str, float] = {}
    for ev in events:
        if ev.get("event") != "gate" or ev.get("rung") != "recovery":
            continue
        rid = str(ev.get("round_id") or "")
        ts = float(ev.get("ts") or 0.0)
        if rid in wanted and isinstance(ev.get("recovery"), dict):
            if ts >= newest.get(rid, -1.0):
                newest[rid] = ts
                out[rid] = ev["recovery"]
    return out


def census(events: Iterable[dict], *, last: int = 50,
           ledger: Path | str | None = None) -> dict:
    """The census as data: the table, the per-surface declaration counts, the split."""
    events = list(events)
    rounds = window_rounds(events, last)
    records = recovery_records(events, rounds)
    declared = Counter(name for rec in records.values()
                       for name in (rec.get("declared") or []))
    unrecoverable = Counter(name for rec in records.values()
                            for name in (rec.get("unrecoverable") or []))
    split = Counter(str(rec.get("contract") or R.CONTRACT_EMPTY)
                    for rec in records.values())
    verdicts = Counter(str(rec.get("verdict") or R.UNEVALUATED)
                       for rec in records.values())
    surfaces = [{"surface": s.name,
                 "digest": s.digest,
                 "inverse": s.inverse or "none",
                 "rounds": declared.get(s.name, 0),
                 "unrecoverable_rounds": unrecoverable.get(s.name, 0)}
                for s in R.SURFACES]
    return {"window": last,
            "rounds_in_window": len(rounds),
            "rounds_with_record": len(records),
            "surfaces": surfaces,
            "no_inverse": [s["surface"] for s in surfaces if s["inverse"] == "none"],
            "contracts": {"empty": split.get(R.CONTRACT_EMPTY, 0),
                          "declared": split.get(R.CONTRACT_DECLARED, 0)},
            "verdicts": {"verified": verdicts.get(R.VERIFIED, 0),
                         "failed": verdicts.get(R.FAILED, 0),
                         "unevaluated": verdicts.get(R.UNEVALUATED, 0)},
            "ledger": str(ledger or S.LEDGER_PATH)}


def render(c: dict) -> str:
    """The printed census. Every section prints, including the empty ones."""
    out = [f"recovery surface census — window: the last {c['window']} round(s) "
           f"with a gate event (ledger: {c['ledger']})",
           f"rounds in the window: {c['rounds_in_window']}, "
           f"rounds with a recovery record: {c['rounds_with_record']} "
           f"of {c['rounds_in_window']}",
           "",
           f"{'surface':<32} {'rounds':>6} {'unrec':>5}  digest -> inverse"]
    for row in c["surfaces"]:
        out.append(f"{row['surface']:<32} {row['rounds']:>6} "
                   f"{row['unrecoverable_rounds']:>5}  {row['digest']} -> "
                   f"{row['inverse']}")
    out += ["",
            f"rounds with an empty effect contract: {c['contracts']['empty']}",
            f"rounds declaring at least one surface: {c['contracts']['declared']}",
            f"verifier verdicts: verified={c['verdicts']['verified']} "
            f"failed={c['verdicts']['failed']} "
            f"unevaluated={c['verdicts']['unevaluated']}",
            f"surfaces with no inverse: "
            f"{', '.join(c['no_inverse']) if c['no_inverse'] else 'none'}"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.automod.recovery_census")
    ap.add_argument("--last", type=int, default=50,
                    help="how many rounds with a gate event to look at (default 50)")
    ap.add_argument("--ledger", default=None,
                    help="ledger path override (default: the promotions ledger)")
    args = ap.parse_args(argv)
    path = Path(args.ledger) if args.ledger else None
    print(render(census(S.ledger_rows_merged(path), last=args.last,
                         ledger=path or S.LEDGER_PATH)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
