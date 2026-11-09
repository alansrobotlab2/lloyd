#!/usr/bin/env python3
"""#3923 clauses 2-4: read the artifact and the doc that reports it, and prove
every figure the doc prints is the artifact's own. The expected strings are RECOMPUTED
here from the JSON with the runner's own formatting (`render_md`,
`eval/run_context_rot_eval.py:1409`): the doc is never trusted.

    LLOYD_DATA=/path/to/lloyd-data .venvs/lloyd/bin/python eval/check_doc_matches_json.py \
        <artifact>.json eval/measurements/context-rot-ksweep-2026-11-08.md

The third argument, taken only when it is passed, is the conflict-count sample the cliff
line compares its k range against; with no third argument it is resolved under
`$LLOYD_DATA` (see `_conflicts_sample_path`). A missing one is reported as a missing
input and exits non-zero — it is never read as a doc that disagrees with its JSON.

Exit 0 = doc == JSON. This matters because the runner writes these tables itself, so a
hand-typed doc could differ from the bytes next to it and read as a measurement.
"""
import json
import os
import sys
from pathlib import Path

CORRECT_KEEP, HEDGE_FLAG = 0.9, 0.1
OUTCOMES = ("correct", "hedged", "wrong", "miss")

art_path, doc_path = Path(sys.argv[1]), Path(sys.argv[2])
doc = json.loads(art_path.read_bytes())
M = doc_path.read_text(encoding="utf-8")
meta, grid = doc["meta"], doc["meta"]["grid"]
arms, lengths, ks, shapes = (grid["conditions"], grid["lengths"],
                             grid["ks"], grid["shapes"])
depths = grid["depths"]
fails: list[str] = []


def need(needle: str, what: str) -> None:
    if needle not in M:
        fails.append(f"{what}: {needle!r} not found in the doc")


# --- clause 2: meta is the grid it ran, the engine it read, the pool it took --------
# The cell count lives in meta.grid; `complete` is the boolean the runner sets at :1665, so
# reading cells off it is a TypeError, and "the file holds what its grid implies" is
# cells vs the product of the grid's own axes.
if doc.get("complete") is not True:
    fails.append(f"artifact is not marked complete: {doc.get('complete')!r}, so the curve is "
                 "a partial run and no cliff should be read off it")
cells = grid["cells"]
product = (len(shapes) * len(lengths) * len(grid["depths"]) * len(arms)
           * len(ks) * len(grid["seeds"]))
if cells != product:
    fails.append(f"meta.grid.cells {cells} != product of meta.grid axes {product}")
if len(doc.get("rows", [])) != cells:
    fails.append(f"the artifact holds {len(doc.get('rows', []))} rows for a {cells}-cell grid")
need(json.dumps(ks).replace(" ", ""), f"the grid's ks {ks}")
need(f"{cells} cells", "the cell count the JSON records")
need(f'"{meta["engine"]["served"]}"', "the engine served model id, quoted as the JSON has it")
need(str(meta["engine"]["max_model_len"]), "the engine max_model_len")
need(str(meta["expect_model"]), "the config expect_model the run read")
for line in meta["pool"]:
    need(line, "a pool-pause log line from meta.pool")
# The clause asks for both pool READINGS rather than an asserted transition, and this
# window's transition was not clean (a first launch died and left its machine hold held).
# Two readings printed with no account of that episode read to a later reader as a clean
# run they cannot tell was not one.
need("Pool status read before", "the pool reading taken before the run")
need("Pool status read after", "the pool reading taken after the run")
need("The hold was not clean", "the account of the stale machine hold and how it was freed")
if doc.get("warm") or any(r.get("warm") for r in doc.get("rows", [])):
    fails.append("the JSON contains warm rows, so the doc's cold-only claim is wrong")

def curve_for(per_len: dict, length) -> tuple[dict | None, str | None]:
    """One arm's curve for one grid length, plus the key that hit.

    The key is returned because the doc's section heading is built by `render_md` from it
    (`int(L) // 1000`), so the only faithful way to find the section is to reproduce that
    expression on the artifact's own key. Building the heading from `meta.grid.lengths`
    instead produced `at 50000k` and reported every section missing — a checker whose own
    lookup is wrong reports the doc as broken, which is the failure mode to avoid here.
    """
    key = str(length)
    return (per_len.get(key) if key in per_len else None), key


# --- clauses 3+4: every curve the grid implies, every k, all four classes ----------
rows = 0
for shape in shapes:
    kc = (doc.get("summary", {}).get(shape) or {}).get("k_curve") or {}
    if sorted(kc) != sorted(arms):
        fails.append(f"summary[{shape}].k_curve arms are {sorted(kc)}, expected {arms}")
    for arm in arms:
        for length in lengths:
            per_len = kc.get(arm) or {}
            cd, ckey = curve_for(per_len, length)
            if cd is None:
                fails.append(f"no k_curve for {shape} / {arm} / {length}")
                continue
            rows += 1
            curve = cd["curve"]
            if sorted(curve, key=int) != sorted((str(k) for k in ks), key=int):
                fails.append(f"k_curve[{shape}/{arm}/{length}] ks are {sorted(curve)}, "
                             f"not the grid's {ks}")
            for k, v in sorted(curve.items(), key=lambda kv: int(kv[0])):
                for o in OUTCOMES:
                    if v[o] is None and v["n"]:
                        fails.append(f"{shape}/{arm}/{length} k={k}: {o} is None with n>0")
            # The appendix table itself, in BOTH directions. Checking only that each JSON
            # figure appears somewhere in the doc is a one-way test: it passes a doc that
            # also prints rows the JSON never produced, and it passes a doc whose row for a
            # DIFFERENT arm happens to carry the same numbers. So the section's rows are
            # parsed and compared as a list against what `render_md`'s own format string
            # yields from this JSON — the identical f-string, so agreement here is agreement
            # with the renderer and not with my transcription of it.
            head = f"### {shape} {arm} at {int(ckey) // 1000}k"
            i = M.find(head)
            if i < 0:
                fails.append(f"the doc has no k_curve section headed {head!r}")
                continue
            body = M[i:]
            # Bound the section at the next heading of ANY level. Bounding only at `###`
            # let the last curve run to the end of the doc and swallow the accuracy
            # comparison table below it, which the row parser then read as extra k rows —
            # a checker reporting the doc has 10 rows where the JSON has 7.
            stop = min((x for x in (body.find("\n### ", 5), body.find("\n## ", 5)) if x > 0),
                       default=-1)
            block = body[:stop] if stop > 0 else body
            # The block's table is the only pipe table under this heading, so the rows are
            # its pipe lines minus the header line and the `|---|` rule. No character-set
            # guessing: the two lines dropped are named by what they contain.
            got = [ln.strip() for ln in block.splitlines()
                   if ln.strip().startswith("|")
                   and "correct" not in ln and not set(ln) <= set("|- ")]
            want = [f"| {k} | {v['correct']:.2f} | {v['hedged']:.2f} | {v['wrong']:.2f} | "
                    f"{v['miss']:.2f} | {v['max_pairwise_jaccard']:.3f} | {v['n']} |"
                    for k, v in sorted(curve.items(), key=lambda kv: int(kv[0]))]
            got_k = [g.split("|")[1].strip() for g in got]
            want_k = [w.split("|")[1].strip() for w in want]
            if got_k != want_k:
                fails.append(f"{head}: the doc's k rows are {got_k}, the JSON's are {want_k}")
            for g, w in zip(got, want):
                if g != w:
                    fails.append(f"{head}: doc row {g!r} != the JSON rendered as {w!r}")
            if len(got) != len(want):
                fails.append(f"{head}: doc has {len(got)} k rows, the JSON implies {len(want)}")
            need(f"Largest k with correct >= {CORRECT_KEEP}: "
                 f"{cd['largest_k_correct_at_least_0.9']}; first k with hedged > "
                 f"{HEDGE_FLAG}: {cd['first_k_hedged_over_0.1']}",
                 f"the headline pair the JSON computes for {shape}/{arm}/{length}")
            # The doc's own headline TABLE row, compared whole. Checking a prefix of it
            # passed a doc whose SECOND headline cell had been rewritten — the probe's
            # mutation 4 — because `| needle | distractK | 50,000 | 32` is a prefix of both
            # the true row and the falsified one. Both figures have to be in the compared
            # string for either of them to be checkable.
            hl = (f"| {shape} | {arm} | {int(ckey):,} | "
                  f"{cd['largest_k_correct_at_least_0.9']} | "
                  f"{cd['first_k_hedged_over_0.1']} |")
            hl_lines = [ln.strip() for ln in M.splitlines()
                        if ln.strip().startswith(f"| {shape} | {arm} | {int(ckey):,} |")]
            if hl_lines != [hl]:
                fails.append(f"headline table row for {shape}/{arm}/{int(ckey):,} is "
                             f"{hl_lines}, the JSON implies exactly [{hl!r}]")

if rows != len(shapes) * len(arms) * len(lengths):
    fails.append(f"{rows} curve rows, expected {len(shapes) * len(arms) * len(lengths)}")

# --- clause 4's truncation statement + clause 5's cliff verdict --------------------
# Both are derived here from the artifact rather than trusted from the doc, and neither is
# inside an if/else on its own value: an earlier draft pinned the verdict only when a cliff
# existed, so deleting the verdict line on a no-cliff run still passed the check.
FULL_KS, FULL_DEPTH = [0, 1, 2, 4, 8, 16, 32], [0.5]
truncated = ks != FULL_KS or depths != FULL_DEPTH
if truncated:
    if "truncated" not in M.lower():
        fails.append("the doc never says the curve is a truncated tranche")
    for value in (f"{depths}", f"{ks}"):
        need(value, f"the axis the tranche actually covered ({value})")

cliff = min((v["first_k_hedged_over_0.1"]
             for shape in shapes
             for per_len in ((doc.get("summary", {}).get(shape) or {})
                             .get("k_curve") or {}).values()
             for v in per_len.values()
             if v["first_k_hedged_over_0.1"] is not None), default=None)


def conflicts_sample_path() -> Path:
    """The conflict-count sample the cliff line compares against, argv[3] if passed.

    The default used to be the absolute path of the one runtime directory this run's
    bytes happen to sit in, which made the landed checker die with a `FileNotFoundError`
    the moment that directory was cleaned up — and the doc's own re-run command passes
    only two arguments, so that was the documented path through the tool. It resolves
    under `LLOYD_DATA` now, the same lever the doc's command block already exports for
    the counter. Production's own resolution (`app.data_root`) is passwd-based and
    refuses an unmarked root; this script stays stdlib-only because its whole job is
    reading two files, so the no-environment fallback is the account's `lloyd-data`
    beside the home directory and a wrong guess there surfaces as the missing-input
    refusal below, naming the path it looked for, never as a verdict.
    """
    if len(sys.argv) > 3:
        return Path(sys.argv[3])
    root = os.environ.get("LLOYD_DATA")
    return (Path(root) if root else Path.home() / "lloyd-data") / "ksweep-2026-11-08" \
        / "conflicts-iv.json"


iv_path = conflicts_sample_path()
iv = None
if iv_path.is_file():
    iv = json.loads(iv_path.read_text())
else:
    fails.append(f"the cliff comparison needs the conflict-count sample and it is not at "
                 f"{iv_path} — pass it as argv[3] or export LLOYD_DATA; the cliff line "
                 f"names the interactive maximum read from it, so without those bytes "
                 f"that figure is unverifiable")

verdict = [ln.strip() for ln in M.splitlines()
           if ln.startswith("**Cliff verdict, one line:**")]
if len(verdict) != 1:
    fails.append(f"expected exactly one cliff verdict line, found {len(verdict)}")
else:
    v = verdict[0]
    # What is checkable in the verdict line is that it names THIS run's k range, so a
    # reader can see how far the axis goes. Requiring the literal word "truncation" inside
    # the sentence demanded a wording the doc states elsewhere ("**The curve is TRUNCATED**"
    # in the banner above) and would have forced duplicate prose to get green.
    if str(ks) not in v:
        fails.append(f"the cliff verdict does not state the k range that ran ({ks}): "
                     f"{v[:140]!r}")
    if cliff is None and "no hedged cliff" not in v.lower():
        fails.append(f"a run with no k over hedged {0.1} in its k_curve must say the cliff "
                     f"was not reached, not hedge it: {v[:140]!r}")
    # The two k values it pairs, not the sentence's wording: what is checkable is that the
    # line joins THIS artifact's cliff to THIS sample's interactive maximum. The missing
    # input is already in `fails` with the path named, so it is not reported twice here.
    if iv is not None and str(iv["max_items_seen"]) not in v:
        fails.append(f"the cliff verdict does not name the interactive max "
                     f"{iv['max_items_seen']} it is comparing against; it reads {v[:150]!r}")
    if cliff is None:
        if "no hedged cliff" not in v:
            fails.append(f"a run with no k over hedged {0.1} in its k_curve must say the "
                         f"cliff was not reached, not hedge it; it reads {v[:150]!r}")
    elif f"k = {cliff}" not in v:
        fails.append(f"the cliff verdict does not carry the cliff k = {cliff} that the JSON's "
                     f"k_curve computes; it reads {v[:150]!r}")

print(json.dumps({"artifact": str(art_path), "doc": str(doc_path),
                  "conflicts_sample": str(iv_path), "sample_read": iv is not None,
                  "cells": cells, "curve_rows": rows, "cliff": cliff,
                  "truncated": truncated,
                  "requests": sum(r.get("n", 0) for r in doc.get("rows", [])),
                  "ok": not fails, "failures": fails[:20]}, indent=2))
sys.exit(1 if fails else 0)
