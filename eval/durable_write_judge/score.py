#!/usr/bin/env python3
"""Score the two judges and render the #580 report.

Reads the corpus plus each judge's raw verdicts and prints — and writes — the
pair the acceptance asks for: recall-on-bad against false-positive-rate on good,
one **silent-pass rate** per judge (the fraction of the samples the judge
accepted whose label is bad), per-class recall including classes at 0, and
**selection quality as a separate metric** (was a same-defect-class case among
the retrieved five — computed from the retrieval leg's own top-k and the gold
labels, never from the judge's answer).

    python eval/durable_write_judge/score.py --corpus corpus.jsonl \
        --raw-a judge_raw_a.jsonl --raw-b judge_raw_b.jsonl --out report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # imported as eval.durable_write_judge.score
    from .build_corpus import class_counts
    from .retrieve import same_class_retrieved, top_k, build_index
except ImportError:  # run as a script from this directory
    from build_corpus import class_counts
    from retrieve import same_class_retrieved, top_k, build_index

JUDGE_NAMES = {"a": "Judge A (static rubric, 5 random labelled examples)",
               "b": "Judge B (static rubric, 5 nearest-retrieved examples, k=5)"}
SUCCESS_RECALL_MARGIN = 10.0  # points absolute, at <= Judge A's false-positive rate


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def verdicts_by_id(rows: list[dict]) -> dict[str, str]:
    return {r["sample_id"]: r.get("verdict", "missing") for r in rows}


def metrics(samples: list[dict], verdicts: dict[str, str]) -> dict:
    """Confusion counts over the samples that got a usable verdict.

    ``unparsed`` / ``error`` rows are reported as ``unjudged`` and kept out of the
    rates rather than counted as accepts — calling a failed call a silent pass
    would put an engine outage into a safety number.
    """
    tp = fn = fp = tn = unjudged = 0
    for s in samples:
        v = verdicts.get(s["id"], "missing")
        if v not in ("flag", "accept"):
            unjudged += 1
            continue
        if s["label"] == "bad":
            tp, fn = (tp + (v == "flag"), fn + (v == "accept"))
        else:
            fp, tn = (fp + (v == "flag"), tn + (v == "accept"))
    judged_bad, judged_good = tp + fn, fp + tn
    accepted = fn + tn
    pct = lambda n, d: round(100.0 * n / d, 1) if d else None
    return {
        "n_bad": sum(1 for s in samples if s["label"] == "bad"),
        "n_good": sum(1 for s in samples if s["label"] == "good"),
        "judged_bad": judged_bad, "judged_good": judged_good, "unjudged": unjudged,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall_on_bad": pct(tp, judged_bad),
        "false_positive_rate": pct(fp, judged_good),
        "silent_pass_rate": pct(fn, accepted),
        "accepted": accepted,
    }


def per_class_recall(samples: list[dict], verdicts: dict[str, str]) -> list[dict]:
    """Recall for every class present in the corpus — a 0 row is the deliverable."""
    out = []
    for name, _ in sorted(class_counts(samples).items()):
        tagged = [s for s in samples if s["label"] == "bad"
                  and name in (s.get("defect_classes") or [])]
        judged = [s for s in tagged if verdicts.get(s["id"]) in ("flag", "accept")]
        caught = sum(1 for s in judged if verdicts[s["id"]] == "flag")
        out.append({"class": name, "n": len(tagged), "n_judged": len(judged),
                    "caught": caught,
                    "recall": round(100.0 * caught / len(judged), 1) if judged else None})
    return out


def selection_quality(samples: list[dict], k: int = 5) -> dict:
    """Retrieval-side truth for every bad sample: class match, same-file twin."""
    index = build_index(samples)
    per_sample, matched, twin = [], 0, 0
    for s in samples:
        if s["label"] != "bad":
            continue
        retrieved = top_k(index, samples, s, k=k)
        same = same_class_retrieved(s, retrieved)
        matched += bool(same)
        has_twin = any(r["vault_path"] == s["vault_path"] for r in retrieved)
        twin += has_twin
        per_sample.append({"sample_id": s["id"],
                           "classes": s["defect_classes"],
                           "same_class_retrieved": same,
                           "same_file_twin_retrieved": has_twin,
                           "retrieved": [r["id"] for r in retrieved]})
    n = len(per_sample)
    return {"n": n, "same_class_rate": round(100.0 * matched / n, 1) if n else None,
            "twin_leak_rate": round(100.0 * twin / n, 1) if n else None,
            "per_sample": per_sample}


def moved_cases(samples: list[dict], va: dict[str, str], vb: dict[str, str]) -> list[dict]:
    """Fox's explainability claim, made checkable: which cases changed, and did a
    same-class example arrive for the ones Judge B alone caught?"""
    index = build_index(samples)
    out = []
    for s in samples:
        a, b = va.get(s["id"]), vb.get(s["id"])
        if a == b:
            continue
        same = None
        if s["label"] == "bad":
            same = same_class_retrieved(s, top_k(index, samples, s))
        out.append({"sample_id": s["id"], "label": s["label"], "a": a, "b": b,
                    "classes": s.get("defect_classes") or [],
                    "same_class_retrieved": same,
                    "vault_path": s["vault_path"]})
    return out


def verdict_by_sample(raw_paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {name: verdicts_by_id(load_jsonl(path)) for name, path in raw_paths.items()}


def render(samples: list[dict], raw_paths: dict[str, Path]) -> str:
    vs = verdict_by_sample(raw_paths)
    ms = {name: metrics(samples, vs[name]) for name in vs}
    sq = selection_quality(samples)
    classes_present = sorted(class_counts(samples).items())

    lines = [
        "# Durable-write judge calibration — backlog #580",
        "",
        "**Status: offline experiment. Nothing in the live write path reads a judge "
        "score; `grep -n durable_write_judge workers/ app/ scripts/ autonomy/ skills/` "
        "returns nothing.** Labels are rule-derived from repair diffs and the "
        "repairer's own recorded words; **one rater (Lloyd), no inter-rater check, "
        "pending Alan's ratification** — the silent-pass number is not trusted until "
        "that exists.",
        "",
        "## Corpus",
        "",
        "Artifact: `eval/durable_write_judge/corpus.jsonl`, rebuilt with "
        "`python eval/durable_write_judge/build_corpus.py` "
        "(window 2026-09-05..2026-09-12, `--bad-cap 20 --good-cap 30 --class-quota 4`).",
        "",
        f"- bad: **{ms['a']['n_bad']}** / good: **{ms['a']['n_good']}** "
        "(targets >=15 bad, >=30 good)",
        "- every bad sample carries its defect classes, its pre-repair text, and the "
        "`git show <repair>^:<path>` object it came from",
        "",
        "| defect class | n in corpus |", "|---|---|",
    ]
    for name, n in classes_present:
        note = "  (<2 examples: not scored as a class result)" if n < 2 else ""
        lines.append(f"| `{name}` | {n}{note} |")

    lines += [
        "",
        "Each class is the repairer's own word for the defect, cited in "
        "`build_corpus.DEFECT_CLASSES`; vault commit messages carry none of them "
        "(repairs land inside multi-hundred-file batch commits).",
        "",
        "## Method",
        "",
        "Both judges run the **same model and the same rubric preamble**, over the "
        "same 50 samples at temperature 0 on the secondary engine "
        "(`http://127.0.0.1:8091/v1/chat/completions`, llama.cpp "
        "`Qwen3.6-35B-A3B-UD-Q3_K_XL`), 100 calls in total. They differ in one "
        "thing: **which five labelled examples they are shown** — Judge A a seeded "
        "random draw, Judge B the five nearest by BM25 over the corpus (k=5). "
        "`head` and `tail` of the prompt are byte-identical across judges "
        "(`test_the_two_judges_differ_only_in_example_selection`).",
        "",
        "Retrieval is BM25 over the labelled corpus alone, not qmd: qmd indexes the "
        "vault, where every bad sample's file now holds its **repaired** text, so "
        "the live index would return the answer that defines the label. Neither "
        "judge's model ever sees the repaired text.",
        "",
        "## Results",
        "",
        "| judge | judged bad/good | recall-on-bad % | false-positive-rate on good % | **silent-pass rate %** |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(ms):
        m = ms[name]
        lines.append(
            f"| {JUDGE_NAMES[name]} | {m['judged_bad']}/{m['judged_good']} "
            f"(unjudged {m['unjudged']}) | **{m['recall_on_bad']}** | "
            f"{m['false_positive_rate']} | **{m['silent_pass_rate']}** |"
        )
    lines += ["", "silent-pass rate = accepted-and-labelled-bad / all accepted."]
    for name in sorted(ms):
        m = ms[name]
        lines.append(
            f"- {JUDGE_NAMES[name]}: of {m['accepted']} accepted samples, "
            f"{m['fn']} were labelled bad — **{m['silent_pass_rate']}%** silent pass; "
            f"recall {m['recall_on_bad']}% ({m['tp']} of {m['judged_bad']} bad flagged)."
        )
    lines += [
        "",
        "Fox's reference point is one in five clean passes hiding a serious error.",
        "",
    ]

    if "a" in ms and "b" in ms:
        da = ms["a"]["recall_on_bad"]
        db = ms["b"]["recall_on_bad"]
        margin = None if da is None or db is None else round(db - da, 1)
        held_fpr = (ms["b"]["false_positive_rate"] is not None
                    and ms["a"]["false_positive_rate"] is not None
                    and ms["b"]["false_positive_rate"] <= ms["a"]["false_positive_rate"])
        success = margin is not None and margin >= SUCCESS_RECALL_MARGIN and held_fpr
        lines += [
            "## Success margin and stop rule",
            "",
            f"Acceptance: Judge B must raise recall-on-bad by >=10 points absolute at "
            f"<= Judge A's false-positive rate. Measured margin: **{margin} points** "
            f"at B FPR {ms['b']['false_positive_rate']} vs A FPR "
            f"{ms['a']['false_positive_rate']} "
            f"({'not worse' if held_fpr else 'WORSE'}). "
            f"Success criterion: **{'met' if success else 'NOT met'}**.",
            "",
            ("The item's stop rule: if B does not beat A at equal false-positive "
             "rate, close with the negative result and the per-class breakdown. No "
             "step-7 shadow run is started, and `k` and the index are not re-tuned "
             "for a second pass. The go/no-go on a shadow run stays the item's own "
             "decision, made by a person on this number."),
            "",
        ]

    lines += ["## Per-class recall (every class present, including 0)", "",
              "| defect class | n bad | A caught | A recall % | B caught | B recall % |",
              "|---|---|---|---|---|---|"]
    ca = {c["class"]: c for c in per_class_recall(samples, vs.get("a", {}))}
    cb = {c["class"]: c for c in per_class_recall(samples, vs.get("b", {}))}
    for name, _ in classes_present:
        a, b = ca.get(name, {}), cb.get(name, {})
        lines.append(f"| `{name}` | {a.get('n')} | {a.get('caught')} | {a.get('recall')} "
                     f"| {b.get('caught')} | {b.get('recall')} |")
    thin = sorted(c for c, v in ca.items() if v["n"] == 1)
    lines += ["",
              "A class at 0 is the deliverable, not a failure: it names something the "
              "judge cannot see." + (
                  " Classes with a single example — "
                  f"{', '.join(f'`{c}`' for c in thin)} — carry a percentage built on "
                  "one sample, so a hit and a miss there are each one sample."
                  if thin else ""),
              ""]

    lines += [
        "## Selection quality (reported separately from judgement quality)",
        "",
        "Fox's open question is whether the retrieved-case leg picks the wrong shape "
        "of case and makes the judge confidently wrong in a new way. That needs its "
        "own number, computed here from the retrieval leg and the gold labels only.",
        "",
        f"- bad samples where at least one same-defect-class case arrived in the "
        f"retrieved five: **{sq['same_class_rate']}%** of {sq['n']}",
        f"- bad samples whose same-file twin (the repaired version of the same note) "
        f"was excluded from retrieval: leak guard held for "
        f"**{round(100.0 - (sq['twin_leak_rate'] or 0.0), 1)}%** "
        f"({sq['twin_leak_rate']}% leaked, and a leak would be the answer, not a hint)",
        "",
        "| sample | classes | same-class retrieved |", "|---|---|---|",
    ]
    for row in sq["per_sample"]:
        lines.append(f"| `{row['sample_id']}` | {', '.join(row['classes'])} | "
                     f"{'yes' if row['same_class_retrieved'] else 'no'} |")

    if {"a", "b"} <= set(vs):
        moved = moved_cases(samples, vs["a"], vs["b"])
        lines += ["", "## Which cases moved the score",
                  "",
                  "| sample | label | A | B | classes | same-class retrieved | path |",
                  "|---|---|---|---|---|---|---|"]
        for m in moved:
            lines.append(f"| `{m['sample_id']}` | {m['label']} | {m['a']} | {m['b']} | "
                         f"{', '.join(m['classes']) or '-'} | "
                         f"{'yes' if m['same_class_retrieved'] else ('n/a' if m['same_class_retrieved'] is None else 'no')} | "
                         f"`{m['vault_path']}` |")
        if not moved:
            lines[-1] = "| — | — | — | — | — | — | no sample changed verdict between the two judges |"

    lines += [
        "",
        "## Boundaries kept",
        "",
        "- #525's discipline holds: anything a stdlib check can settle (file exists, "
        "count equals) belongs to that verifier, never to a judge. This judge's scope "
        "is the residue — significance, omission, wrong-level — and that boundary is "
        "the first paragraph of `eval/durable_write_judge/README.md`.",
        "- The judging pass is offline and one-shot. It costs secondary-engine "
        "capacity only (100 calls, `requests_running: 0` on the primary), and adds no "
        "prompt tokens to the interactive loop, so #520's prefix-cache pressure is "
        "untouched.",
        "",
        "## Caveats",
        "",
        "- **One rater.** Every label derives from Lloyd's own recorded repair; Alan "
        "is the domain expert who would have to ratify them, and inter-rater "
        "reliability is unavailable. Correction evidence travels with each sample "
        "(`correction_evidence`) so a rater can re-judge the label, not just the text.",
        "- **Rule-derived labels are conservative and partial.** A class is assigned "
        "only when its signature is visible in the repair diff; a repair that "
        "reworded prose without removing a URL or adding a heading is unlabelled "
        "here. Meaning-reversals and misattributed speakers — both recorded in "
        "`memory/2026-09-09.md` — are classes the detector has **no signature for**, "
        "so the corpus under-represents them; that is a limitation of the corpus, not "
        "evidence they are rare.",
        "- **`good` means no class signature fired**, not verified-clean. A judge "
        "flagging a `good` sample may be right, which inflates the measured "
        "false-positive rate and makes the reported silent-pass rate optimistic.",
        "- **Small n.** 20 bad across 5 classes; two classes have n<2. Percentages "
        "move by ~5 points per sample.",
        "- **The model cannot check the claim against the transcript.** It is judging "
        "internal plausibility, which is exactly the residue a rubric can reach; "
        "invented-but-plausible URLs are only detectable here against the retrieved "
        "examples, which is the hypothesis under test.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd ~/lloyd",
        "PY=.venvs/lloyd/bin/python",
        "$PY eval/durable_write_judge/build_corpus.py            # corpus.jsonl (reads vault git, writes nothing)",
        "$PY eval/durable_write_judge/judge.py --judge both      # 100 offline calls on the secondary",
        "$PY eval/durable_write_judge/score.py                   # this report",
        "$PY -m pytest tests/test_durable_write_judge.py -q      # the pinned clauses",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(here / "corpus.jsonl"))
    ap.add_argument("--raw-a", default=str(here / "judge_raw_a.jsonl"))
    ap.add_argument("--raw-b", default=str(here / "judge_raw_b.jsonl"))
    ap.add_argument("--out", default=str(here / "report.md"))
    args = ap.parse_args(argv)

    samples = load_jsonl(Path(args.corpus))
    raws = {}
    for name, path in (("a", args.raw_a), ("b", args.raw_b)):
        if Path(path).exists():
            raws[name] = Path(path)
    if not raws:
        raise SystemExit("no judge_raw_*.jsonl found; run judge.py first")

    report = render(samples, raws)
    Path(args.out).write_text(report)
    print(report.split("## Results")[0].strip()[:400])
    vs = verdict_by_sample(raws)
    for name in sorted(vs):
        m = metrics(samples, vs[name])
        print(f"{name}: recall_on_bad={m['recall_on_bad']}% "
              f"fp_rate={m['false_positive_rate']}% "
              f"silent_pass={m['silent_pass_rate']}% unjudged={m['unjudged']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
