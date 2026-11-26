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

``--raw-c`` (#4093) switches to the THREE-arm report over the held-out half of
``induce_split.json``, written to ``--out-three`` (default ``report_induced.md``);
``report.md`` is then not written. Without ``--raw-c`` nothing reads a Judge C file,
so the default output stays the committed ``report.md`` byte for byte.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # imported as eval.durable_write_judge.score
    from .build_corpus import DEFECT_CLASSES, class_counts
    from .retrieve import same_class_retrieved, top_k, build_index
except ImportError:  # run as a script from this directory
    from build_corpus import DEFECT_CLASSES, class_counts
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
    n_bad = sum(1 for s in samples if s["label"] == "bad")
    n_good = sum(1 for s in samples if s["label"] == "good")

    lines = [
        "# Durable-write judge calibration — backlog #580",
        "",
        "**Status: offline experiment. Nothing in the live write path reads a judge "
        "score; `grep -n durable_write_judge workers/ app/ scripts/ autonomy/ skills/` "
        "returns nothing.** Labels are rule-derived from repair diffs and the "
        "repairer's own recorded words, one rater (Lloyd), no inter-rater check; "
        f"**the {n_bad} `bad` labels are RATIFIED as gold by the #2639 owed-check "
        "2026-10-16 on Alan's delegation (each re-derives from a repair diff plus the "
        f"repairer's own defect word), the {n_good} `good` labels are refused as gold "
        "because their warrant is absence-of-repair — so recall-on-bad is quotable "
        "and silent-pass is not.**",
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
        "| judge | judged bad/good | recall-on-bad % | false-positive-rate on good % | **silent-pass rate %** (UNRATIFIED-DENOMINATOR) |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(ms):
        m = ms[name]
        lines.append(
            f"| {JUDGE_NAMES[name]} | {m['judged_bad']}/{m['judged_good']} "
            f"(unjudged {m['unjudged']}) | **{m['recall_on_bad']}** | "
            f"{m['false_positive_rate']} | **{m['silent_pass_rate']}** "
            "UNRATIFIED-DENOMINATOR |"
        )
    # #2465, on #2254's stop rule: the attribution confound these rates get read for was
    # measured, came back negative, and nothing in this repo said so. Placed directly
    # under the table rather than in prose lower down because the table is what a reader
    # reaches for first when the question is "is this judge just passing its own work",
    # and inserted AFTER the rows rather than before them so the Judge A row stays line 32
    # — the line the probe's own two files cite as that figure's source (`arms.py`'s
    # header, `report.py`'s `PUBLISHED_JUDGE_A`). It has to carry the finding in its own
    # words: the report it points at lives in runtime data, outside this checkout, so a
    # bare path would be a pointer a reader of a clone cannot follow.
    lines += [
        "",
        "> **Self-preference: probed 2026-10-09 on `primary`, negative — the bias this "
        "row is usually read for did NOT reproduce.** Same rubric and corpus, 50 samples "
        "x 4 attribution arms = 200 rows at temperature 0 (`asked_engine=secondary` -> "
        "`served_engine=primary`, which is the disabled-secondary rule and not a routing "
        "guarantee). The claim was that a note labelled as the judge's own output is "
        "passed more often; measured the other way — pass rate on `bad` was 65.0 % for "
        "the `self` arm against 75.0 % for the unattributed `none` arm, silent-pass 30.2 "
        "% against 34.1 %, i.e. -3.9 points, an attributed note was flagged MORE, not "
        "less. The floor it is read against is the arm that resends `none`'s prompt "
        "verbatim: +0.7 points silent-pass, 4.0 % flip. So the attribution block does "
        "move this judge's grade by a few points, but not in the self-favouring "
        "direction, and 32.4 % stands as a calibration figure rather than as a judge "
        "protecting itself. Report and per-arm figures: "
        "`~/lloyd-data/eval/owedcheck-2254/REPORT.md` (`arms-final.jsonl` beside it, "
        "200/200 rows judged — the 400-token judging budget the probe shipped with "
        "truncated 6 of 200 replies mid-JSON and the run correctly reported nothing).",
        "",
        "silent-pass rate = accepted-and-labelled-bad / all accepted. Every figure "
        "marked **UNRATIFIED-DENOMINATOR** rests on that denominator, and every "
        f"accepted `good` inside it is one of the {n_good} labels refused as gold, so "
        "the rate is printed as untrusted rather than quoted.",
    ]
    for name in sorted(ms):
        m = ms[name]
        lines.append(
            f"- {JUDGE_NAMES[name]}: of {m['accepted']} accepted samples, "
            f"{m['fn']} were labelled bad — **{m['silent_pass_rate']}%** silent pass "
            "(UNRATIFIED-DENOMINATOR); "
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
        "- **One rater.** Every label derives from Lloyd's own recorded repair, and "
        "inter-rater reliability is unavailable; the ratification that does exist is "
        "bounded and delegated — the "
        f"{n_bad} `bad` labels are RATIFIED as gold by the #2639 owed-check 2026-10-16 "
        "on Alan's delegation, while the "
        f"{n_good} `good` labels are refused as gold because their warrant is "
        "absence-of-repair, so recall-on-bad is quotable and silent-pass is not. "
        "Correction evidence travels with each sample (`correction_evidence`) so a "
        "rater can re-judge the label, not just the text.",
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
        # Computed, never written: this line used to assert a class count by hand
        # ("two classes have n<2") that the shipped corpus contradicts — its smallest
        # class has n = 2. A caveat that drifts from the corpus it describes is worse
        # than no caveat, because it is the sentence a reader trusts about n.
        f"- **Small n.** {n_bad} bad across {len(classes_present)} classes"
        + (f", smallest class at n = {min(n for _, n in classes_present)}"
           if classes_present else "")
        + ". Percentages move by ~5 points per sample.",
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


# --- #4093: the three-arm report over the held-out half -----------------------------

THREE_ARM_NAMES = {"a": "Judge A (hand-authored rubric, 5 random labelled examples)",
                   "b": "Judge B (hand-authored rubric, 5 nearest-retrieved examples, k=5)",
                   "c": "Judge C (induced rubric, Judge A's 5 random labelled examples)"}

#: The finding line's two forms; one of them is always printed (#4093 clause 5).
FINDING_NOVEL = "FINDING: {n} induced criterion(s) not expressible as any of the five corpus classes: {which}"
FINDING_NONE = "FINDING: no induced criterion falls outside the five corpus classes"
UNINTERPRETABLE = ("UNINTERPRETABLE: {n} criterion(s) failed the leakage check, so no margin "
                   "is reported")


def _count_pct(n: int, d: int) -> str:
    return f"{round(100.0 * n / d, 1)} ({n}/{d})" if d else "n/a (0/0)"


def _served(rows: list[dict]) -> str:
    names = sorted({str(r.get("served_model") or r.get("model") or "unrecorded")
                    for r in rows if r.get("verdict") not in ("dry_run",)})
    return ", ".join(names) or "unrecorded"


def render_three_arm(samples: list[dict], raw_paths: dict[str, Path], split: dict,
                     rubric: dict) -> str:
    """A, B and C on the IDENTICAL held-out subset, at equal n, with the leakage check.

    Rows a raw file carries for a training id are ignored and counted: the held-out
    half is the only population any arm is scored on. Missing held-out rows are
    `unjudged`, never accepts, exactly as in `metrics`.
    """
    try:
        from .induce_rubric import leakage_check
    except ImportError:
        from induce_rubric import leakage_check

    held = set(split["held_out_ids"])
    sub = [s for s in samples if s["id"] in held]
    raws = {k: load_jsonl(v) for k, v in raw_paths.items()}
    vs = {k: {r["sample_id"]: r.get("verdict", "missing") for r in rows
              if r["sample_id"] in held} for k, rows in raws.items()}
    outside = {k: sum(1 for r in rows if r["sample_id"] not in held)
               for k, rows in raws.items()}
    ms = {k: metrics(sub, vs[k]) for k in sorted(vs)}
    n_bad = sum(1 for s in sub if s["label"] == "bad")
    n_good = len(sub) - n_bad
    engines = {k: _served(raws[k]) for k in sorted(raws)}
    leak = leakage_check(rubric["criteria"], samples, split)
    flagged = [x for x in leak if x["flagged"]]

    lines = [
        "# Durable-write judge: the induced-rubric arm — backlog #4093",
        "",
        "**Status: offline experiment, one induction pass and one judging pass per arm. "
        "Nothing in the live write path reads a judge score.** This is PMR's rubric "
        "induction only (arXiv:2611.14017); its GRPO credit assignment is not reproduced.",
        "",
        "## Split and engines",
        "",
        f"- split: `induce_split.json` — {split['rule']}",
        f"- held-out subset every arm is scored on: **{len(sub)}** samples "
        f"({n_bad} bad / {n_good} good); training half: {len(split['train_ids'])} "
        "(induction input only)",
    ]
    for k in sorted(raws):
        lines.append(f"- {THREE_ARM_NAMES[k]}: served model `{engines[k]}`; "
                     f"{len(vs[k])} held-out rows read, {outside[k]} training-half rows ignored")
    mixed = len(set(engines.values())) > 1
    if mixed:
        lines.append("- **MIXED ENGINES: the arms were not judged by one model, so no "
                     "difference below is attributable to the rubric.**")
    lines += [
        "",
        "## Results (held-out subset, equal n)",
        "",
        "Exact counts beside every percentage: with ~10 held-out bad samples one sample "
        "is ~10 points.",
        "",
        "| judge | judged bad/good | recall-on-bad % (tp/judged bad) | "
        "false-positive-rate % (fp/judged good) | silent-pass % (fn/accepted) |",
        "|---|---|---|---|---|",
    ]
    for k in sorted(ms):
        m = ms[k]
        lines.append(
            f"| {THREE_ARM_NAMES[k]} | {m['judged_bad']}/{m['judged_good']} "
            f"(unjudged {m['unjudged']}) | **{_count_pct(m['tp'], m['judged_bad'])}** | "
            f"{_count_pct(m['fp'], m['judged_good'])} | "
            f"{_count_pct(m['fn'], m['accepted'])} UNRATIFIED-DENOMINATOR |")
    lines += ["",
              "silent-pass is printed UNRATIFIED-DENOMINATOR: its denominator holds the "
              "`good` labels refused as gold, so recall-on-bad is the only quotable "
              "headline.", ""]

    lines += ["## Per-class recall (held-out; every class, including 0)", "",
              "| defect class | n bad held out | " + " | ".join(
                  f"{k.upper()} caught/judged (recall %)" for k in sorted(ms)) + " |",
              "|---|---|" + "---|" * len(ms)]
    names = sorted(set(DEFECT_CLASSES) | set(class_counts(samples)))
    per = {k: {c["class"]: c for c in per_class_recall(sub, vs[k])} for k in sorted(ms)}
    for name in names:
        n = sum(1 for s in sub if s["label"] == "bad" and name in (s.get("defect_classes") or []))
        cells = []
        for k in sorted(ms):
            c = per[k].get(name) or {"caught": 0, "n_judged": 0}
            cells.append(f"{c['caught']}/{c['n_judged']} "
                         f"({round(100.0 * c['caught'] / c['n_judged'], 1) if c['n_judged'] else 'n/a'})")
        lines.append(f"| `{name}` | {n} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Leakage check (per induced criterion)", "",
              "`training hits`: a training-half defect word (class label, short evidence "
              "string) or 8-word sample/evidence run appearing verbatim. `held-out hits`: a "
              "held-out id, defect word or 8-word run the training half does not contain — "
              "i.e. the criterion could have been written from the held-out labels. Either "
              "flags the criterion.", "",
              "| # | criterion | training hits | held-out hits | flagged |",
              "|---|---|---|---|---|"]
    for x in leak:
        lines.append(f"| {x['n']} | {x['text']} | {'; '.join(x['train_hits']) or '-'} | "
                     f"{'; '.join(x['held_out_hits']) or '-'} | "
                     f"{'**yes**' if x['flagged'] else 'no'} |")
    lines.append("")

    novel = [i for i, c in enumerate(rubric["criteria"], 1)
             if (c.get("maps_to") or "none") == "none"]
    lines += ["## Class coverage", "",
              (FINDING_NOVEL.format(n=len(novel), which=", ".join(f"#{i}" for i in novel))
               if novel else FINDING_NONE),
              "",
              "`maps_to` is the induction pass's own reading of each criterion; read the "
              "criteria verbatim in `induced_rubric.md` before quoting this line.", ""]

    lines += ["## Success margin and stop rule", ""]
    if flagged:
        lines.append(UNINTERPRETABLE.format(n=len(flagged)))
    elif mixed:
        lines.append("UNINTERPRETABLE: mixed engines across arms, so no margin is reported")
    elif "a" in ms and "c" in ms:
        ra, rc = ms["a"]["recall_on_bad"], ms["c"]["recall_on_bad"]
        fa, fc = ms["a"]["false_positive_rate"], ms["c"]["false_positive_rate"]
        margin = None if ra is None or rc is None else round(rc - ra, 1)
        held_fpr = fa is not None and fc is not None and fc <= fa
        met = margin is not None and margin >= SUCCESS_RECALL_MARGIN and held_fpr
        lines.append(
            f"Judge C vs Judge A on the held-out subset: recall-on-bad "
            f"{_count_pct(ms['c']['tp'], ms['c']['judged_bad'])} vs "
            f"{_count_pct(ms['a']['tp'], ms['a']['judged_bad'])} = **{margin} points**, "
            f"FPR {fc} vs {fa} ({'not worse' if held_fpr else 'WORSE'}). "
            f"Bar (>= +{SUCCESS_RECALL_MARGIN} points at <= A's FPR): "
            f"**{'met' if met else 'NOT met'}**.")
    lines += ["",
              "Stop rule (#4093 step 6): one induction pass, no re-tuning, no second pass. "
              "A margin is quoted only after Judge C's reasons are checked against each "
              "sample's `correction_evidence` — #580 found judges right for the wrong URL.",
              ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(here / "corpus.jsonl"))
    ap.add_argument("--raw-a", default=str(here / "judge_raw_a.jsonl"))
    ap.add_argument("--raw-b", default=str(here / "judge_raw_b.jsonl"))
    ap.add_argument("--out", default=str(here / "report.md"))
    # #4093. No default for --raw-c, on purpose: a `judge_raw_c.jsonl` sitting in this
    # directory must not join the committed two-arm report the moment it exists.
    ap.add_argument("--raw-c", default=None,
                    help="Judge C's raws; switches to the three-arm held-out report")
    ap.add_argument("--split", default=str(here / "induce_split.json"))
    ap.add_argument("--rubric", default=str(here / "induced_rubric.md"))
    ap.add_argument("--out-three", default=str(here / "report_induced.md"))
    args = ap.parse_args(argv)

    samples = load_jsonl(Path(args.corpus))
    if args.raw_c is not None:
        try:
            from .induce_rubric import load_rubric, load_split
        except ImportError:
            from induce_rubric import load_rubric, load_split
        split = load_split(Path(args.split))
        raws3 = {"a": Path(args.raw_a), "b": Path(args.raw_b), "c": Path(args.raw_c)}
        missing = [str(p) for p in raws3.values() if not p.exists()]
        if missing:
            raise SystemExit(f"the three-arm report needs all three raws: missing {missing}")
        report3 = render_three_arm(samples, raws3, split,
                                   load_rubric(Path(args.rubric), split))
        Path(args.out_three).write_text(report3)
        for ln in report3.splitlines():
            if ln.startswith(("| Judge", "FINDING", "UNINTERPRETABLE", "Judge C vs")):
                print(ln)
        print(f"wrote {args.out_three}")
        return 0
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
              f"silent_pass={m['silent_pass_rate']}% (UNRATIFIED-DENOMINATOR) "
              f"unjudged={m['unjudged']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
