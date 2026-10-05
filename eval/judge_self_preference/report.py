#!/usr/bin/env python3
"""#2254 — read the four attribution arms and print the bias question as numbers.

`render(samples, rows, url=..., model=...)` takes verdict rows (whatever produced
them — the real engine run, or a crafted fixture in the test file) and prints:

* the **repeat** flip rate, which is this engine's own floor at temperature 0 and the
  number every other rate here is an upper bound on;
* **self**-vs-`none` and **human**-vs-`none` flip rates;
* per-arm pass rate on the `bad`-labelled samples, which is the direction split —
  an attribution effect should show up as a changed pass rate on bad notes, not as
  noise in both directions;
* flip counts per defect class, including the classes at zero, because "which class
  is the judge soft on" is the diagnostic and a class that never moved is a finding;
* Δ silent-pass against the **same-session `none` arm**, with the engine and model
  recorded on the page.

The published figure is not the comparator. `eval/durable_write_judge/report.md:32`
records Judge A at silent-pass 32.4 % on `Qwen3.6-35B-A3B-UD-Q3_K_XL`, and this probe
runs on whatever answers today — the secondary slot that produced 32.4 % has been off
since 2026-09-20 (`config.yaml:37`), so a delta against 32.4 % would move the judge
model at the same time as the attribution. Every delta here is computed against the
`none` arm of the same run; 32.4 % is printed as context, labelled as a
different-model figure, and named as such on the same line it appears on.

A row whose verdict is neither `flag` nor `accept` raises rather than being dropped:
a sample that silently leaves the denominator is the failure mode #580's `unparsed`
rule exists to stop, and it would move every percentage on this page without a trace.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from eval.durable_write_judge.build_corpus import DEFECT_CLASSES, class_counts
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval.durable_write_judge.build_corpus import DEFECT_CLASSES, class_counts

from eval.judge_self_preference.arms import ARMS, BASELINE_ARM

#: Judge A's published number, kept as data so no prose in this file can quietly
#: re-derive it and so the "different model" caveat travels with the figure.
PUBLISHED_JUDGE_A = {
    "silent_pass_pct": 32.4,
    "recall_on_bad_pct": 45.0,
    "model": "Qwen3.6-35B-A3B-UD-Q3_K_XL",
    "engine": "secondary engine, llama.cpp on 127.0.0.1:8091 (off since 2026-09-20)",
    "source": "eval/durable_write_judge/report.md:32",
}

JUDGED = ("flag", "accept")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def verdicts_by_id(rows: list[dict], arm: str) -> dict[str, str]:
    return {r["sample_id"]: r["verdict"] for r in rows if r["arm"] == arm}


def _pct(num: int | float, den: int | float) -> float | None:
    """Percentage, or None when the denominator is empty.

    None renders as `n/a` rather than 0.0: an empty denominator that prints as zero
    reads as "no problem here", which is the opposite of what it means.
    """
    return None if not den else 100.0 * num / den


def _f(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def arm_metrics(samples: list[dict], verdicts: dict[str, str]) -> dict:
    """Pass rate on `bad`, and the silent-pass rate, for one arm's verdicts."""
    by_id = {s["id"]: s for s in samples}
    missing = [s["id"] for s in samples if s["id"] not in verdicts]
    bad = [s for s in samples if s["label"] == "bad"]
    accepted = [s for s in samples if verdicts.get(s["id"]) == "accept"]
    bad_accepted = [s for s in accepted if by_id[s["id"]]["label"] == "bad"]
    return {
        "n": len(samples), "n_bad": len(bad), "n_good": len(samples) - len(bad),
        "missing": missing,
        "bad_passed": len(bad_accepted),
        "pass_rate_on_bad_pct": _pct(len(bad_accepted), len(bad)),
        "accepted": len(accepted),
        "silent_pass_pct": _pct(len(bad_accepted), len(accepted)),
    }


def flip_counts(samples: list[dict], baseline: dict[str, str],
                arm: dict[str, str]) -> dict:
    """Samples whose verdict moved against the baseline arm, plus the per-class split.

    Direction is kept per class over the samples that carry it: #580's five classes
    are the only structure the labels have, and "the flips are all in
    `omitted_numerical_data`" is a different finding from "the flips are scattered".
    """
    flips = [s for s in samples
             if s["id"] in baseline and s["id"] in arm
             and baseline[s["id"]] != arm[s["id"]]]
    by_id = {s["id"]: s for s in samples}
    per_class = Counter()
    for s in flips:
        for cls in s.get("defect_classes") or []:
            per_class[cls] += 1
    return {
        "flips": [s["id"] for s in flips],
        "flip_pct": _pct(len(flips), len(samples)),
        # Every rubric class gets a count, including the ones at 0 and the ones no
        # sample in this corpus carries at all: an absent class is a measured
        # nothing, and printing only the non-zero ones hides which was which.
        "per_class": {cls: per_class.get(cls, 0) for cls in sorted(DEFECT_CLASSES)},
        "unclassified_flips": sum(1 for s in flips
                                  if not (s.get("defect_classes") or [])
                                  and by_id[s["id"]]["label"] == "bad"),
    }


def compute(samples: list[dict], rows: list[dict]) -> dict:
    """All four arms' numbers, every one of them against the same run's `none`."""
    unjudged = [r for r in rows if r["verdict"] not in JUDGED]
    if unjudged:
        raise ValueError(
            "refusing to render a report over unjudged rows — each one silently "
            "leaves the denominator: "
            + ", ".join(sorted({f"{r['sample_id']}/{r['arm']}/{r['verdict']}"
                                for r in unjudged})[:10]))
    verdicts = {arm: verdicts_by_id(rows, arm) for arm in ARMS}
    base = verdicts[BASELINE_ARM]
    out: dict = {"metrics": {}, "flips": {}, "classes_present": sorted(
        c for c, n in class_counts(samples).items() if n)}
    # Two passes: every arm's delta is measured against `none`, and `repeat` is first
    # in ARMS, so one loop would read a baseline that has not been computed yet.
    for arm in ARMS:
        out["metrics"][arm] = arm_metrics(samples, verdicts[arm])
        out["flips"][arm] = ({"flips": [], "flip_pct": None, "per_class": {},
                              "unclassified_flips": 0}
                             if arm == BASELINE_ARM
                             else flip_counts(samples, base, verdicts[arm]))
    base_sp = out["metrics"][BASELINE_ARM]["silent_pass_pct"]
    published = PUBLISHED_JUDGE_A["silent_pass_pct"]
    for arm in ARMS:
        m = out["metrics"][arm]
        sp = m["silent_pass_pct"]
        m["delta_silent_pass_vs_none_pts"] = (None if sp is None or base_sp is None
                                              else sp - base_sp)
        m["delta_silent_pass_vs_published_pts"] = None if sp is None else sp - published
    return out


def render(samples: list[dict], rows: list[dict], *, url: str, model: str,
           temperature: int = 0, generated_by: str = "run_arms.py",
           engine_asked: str | None = None,
           engine_rewritten: bool | None = None) -> str:
    data = compute(samples, rows)
    m, f = data["metrics"], data["flips"]
    n_rows = len(rows)

    L: list[str] = []
    add = L.append
    add("# #2254 — does the judge prefer its own family?")
    add("")
    add(f"Engine `{url}` | model `{model}` | temperature {temperature} | "
        f"{len(samples)} samples × {len(ARMS)} arms = {n_rows} judged rows "
        "(0 unjudged — `run_arms.py` refuses to write this file otherwise).")
    if engine_asked is not None:
        add(f"Resolved as `asked_engine={engine_asked}` → "
            f"`served_engine={model.rsplit('/', 1)[-1]}`, override applied="
            f"{engine_rewritten}. This job is registered in no `decisions.yaml` and "
            "`JOBS_ON_PRIMARY` is script-written, so it reaches this engine through "
            "`resolve_model_alias`'s secondary-disabled rule, not through a default "
            "naming primary — a reader must not mistake the routing for a guarantee.")
    add("")
    add("Every rate below is read against the **`repeat`** arm first: it is the same "
        "prompt as `none`, so its flip rate is this engine's own floor at temperature "
        "0, and any smaller effect in `self` or `human` is not a bias signal.")
    add("")
    add("| arm | flip vs `none` % | pass rate on `bad` % | silent-pass % | "
        "Δ silent-pass vs `none` (pts) |")
    add("|---|---|---|---|---|")
    for arm in ARMS:
        add(f"| {arm} "
            f"| {'—' if arm == BASELINE_ARM else _f(f[arm]['flip_pct'])} "
            f"| {_f(m[arm]['pass_rate_on_bad_pct'])} "
            f"| {_f(m[arm]['silent_pass_pct'])} "
            f"| {'—' if arm == BASELINE_ARM else _f(m[arm]['delta_silent_pass_vs_none_pts'], signed=True)} |")
    add("")
    add(f"`bad` = {m['none']['n_bad']} samples, `good` = {m['none']['n_good']}. "
        "silent-pass % = accepted-and-labelled-bad / all accepted, the same definition "
        "`eval/durable_write_judge/report.md` prints. `good` there means only *no class "
        "signature fired*, so every silent-pass figure here inherits that optimism "
        "(`eval/durable_write_judge/report.md:113`).")
    add("")
    add("## Flip counts by defect class")
    add("")
    counts = class_counts(samples)
    add("| defect class | samples carrying it | " +
        " | ".join(arm for arm in ARMS if arm != BASELINE_ARM) + " |")
    add("|---|---|" + "---|" * (len(ARMS) - 1))
    for cls in sorted(DEFECT_CLASSES):
        add(f"| {cls} | {counts.get(cls, 0)} | " +
            " | ".join(str(f[arm]["per_class"].get(cls, 0))
                       for arm in ARMS if arm != BASELINE_ARM) + " |")
    add("")
    add("Classes at 0 are printed, not dropped — including "
        + ", ".join(f"`{cls}` (no sample in this corpus carries it)"
                    for cls in sorted(DEFECT_CLASSES) if cls not in data["classes_present"])
        + "." if data["classes_present"] != sorted(DEFECT_CLASSES) else "")
    add("")
    add("## Direction")
    add("")
    add("The bias claim has a direction: a note *claimed* to come from the judge's own "
        "family should be let through more often, and one claimed to be human-written "
        "less often, on the **`bad`** rows. Flip rates alone cannot show that — half the "
        "flips in each direction would produce the same number.")
    add("")
    pub = PUBLISHED_JUDGE_A
    add("## The published figure is context, not the comparator")
    add("")
    add(f"`{pub['source']}` records Judge A at silent-pass "
        f"**{pub['silent_pass_pct']} %** on `{pub['model']}` ({pub['engine']}). That "
        f"engine no longer answers, and this run judged on `{model}`, so a delta "
        f"against {pub['silent_pass_pct']} % would move the judge model at the same time "
        "as the attribution. Comparators here are the same-session `none` arm "
        "(Δ column above). Against the published figure, for completeness only: "
        + ", ".join(f"`{arm}` {_f(m[arm]['delta_silent_pass_vs_published_pts'], signed=True)} pts"
                    for arm in ARMS if arm != "repeat") + ".")
    add(
        "Every silent-pass figure here inherits #580's own caveat "
        "(`eval/durable_write_judge/report.md:113`): a `good` label means no defect-class "
        "signature fired, not that anyone verified the note as clean. The rate is "
        "optimistic, so an arm that lowers it might be failing to see defects rather than "
        "preferring its own family — which is why the direction split is read against the "
        "`repeat` floor and not on its own.")
    add("")
    add("## Machine lines")
    add("")
    add(f"samples={len(samples)} n_bad={m['none']['n_bad']} "
        f"n_good={m['none']['n_good']} rows={n_rows} temperature={temperature}")
    add(f"engine={url}")
    add(f"model={model}")
    add("flip_vs_none_pct " + " ".join(
        f"{arm}={_f(f[arm]['flip_pct'])}" for arm in ARMS if arm != BASELINE_ARM))
    add("pass_rate_on_bad_pct " + " ".join(
        f"{arm}={_f(m[arm]['pass_rate_on_bad_pct'])}" for arm in ARMS))
    add("silent_pass_pct " + " ".join(
        f"{arm}={_f(m[arm]['silent_pass_pct'])}" for arm in ARMS))
    add("delta_silent_pass_vs_none_pts " + " ".join(
        f"{arm}={_f(m[arm]['delta_silent_pass_vs_none_pts'], signed=True)}"
        for arm in ARMS if arm != BASELINE_ARM))
    add(f"published_judge_a_silent_pass_pct={pub['silent_pass_pct']} "
        f"published_model={pub['model']} comparable_to_this_run=no")
    add("vs_published_silent_pass_pts " + " ".join(
        f"{arm}={_f(m[arm]['delta_silent_pass_vs_published_pts'], signed=True)}"
        for arm in ARMS))
    for arm in ARMS:
        if arm == BASELINE_ARM:
            continue
        add(f"class_flips arm={arm} " + " ".join(
            f"{cls}={f[arm]['per_class'].get(cls, 0)}" for cls in sorted(DEFECT_CLASSES)))
    add(f"generated_by={generated_by}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the #2254 arm report from raw rows.")
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parent.parent
                                            / "durable_write_judge" / "corpus.jsonl"))
    ap.add_argument("--rows", required=True, help="verdict rows JSONL from run_arms.py")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    text = render(load_jsonl(args.corpus), load_jsonl(args.rows),
                  url=args.url, model=args.model)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote report -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
