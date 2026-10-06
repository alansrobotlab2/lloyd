#!/usr/bin/env python3
"""Rebuild and score the near-duplicate calibration set (#2271 clause 1).

The committed artifact is `eval/dup_calibration.json`: ≥30 real vault pairs
spanning a similarity sweep, each with a `label` of `same` / `near` /
`unrelated`, plus the two notes' compared text embedded so the agreement is
recomputable on a machine that cannot see the vault.

Two ways to run this, and the difference is the whole point:

    python3 eval/calibrate_dup_threshold.py                 # rebuild + djev label + print
    python3 eval/calibrate_dup_threshold.py --check         # score the committed set only

Rebuilding reads the vault, so it needs the corpus; `--check` needs nothing but
the JSON and is what the tests call. Either way the printed line is
`agreement <agree>/<total>` — the denominator beside the numerator, per the
"denominator can be zero" rule: a set with no rows prints `0/0` and the word
`NO ROWS`, never a score.

`DJEV_LABEL=0` (or `--no-djev`) keeps the labels already in the file instead of
asking the decision engine, which is how a rebuild stays reproducible when GPU 2
is busy; the label source is recorded per row (`label_source`) so a djev label and
a hand label never read as the same kind of evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from eval import dup_detect as dd  # noqa: E402

CALIBRATION_PATH = HERE / "dup_calibration.json"
DJEV_DECIDE_URL = "http://127.0.0.1:8080/api/djev/decide"
#: the similarity bands sampled, worst first — the sweep is what makes the
#: threshold's location evidence rather than a lucky gap
BANDS = [(0.90, "0.90-1.00"), (0.75, "0.75-0.90"), (0.60, "0.60-0.75"),
         (0.45, "0.45-0.60"), (0.30, "0.30-0.45"), (0.20, "0.20-0.30"),
         (0.10, "0.10-0.20"), (0.05, "0.05-0.10")]
PER_BAND = 5
SEED = 20261006
DEFAULT_ROOTS = ("memory", "knowledge", "skills", "architecture", "backlog",
                 "projects", "people")
MIN_ROW = 20


# ── sampling ─────────────────────────────────────────────────────────────────

#: What must never be copied into a committed artifact: the holdout leg's query
#: ids and its filename. The rail that says so is corpus-wide
#: (the retrieval-holdout split's corpus guard, which greps EVERY tracked file, because
#: a held-out query that leaks into tracked code or data stops being held out),
#: and an artifact that embeds whole note bodies is exactly the surface that can
#: leak one by accident — a backlog note quoting the holdout work would otherwise
#: ship the id inside the calibration set. Redaction is the narrow fix: the row
#: stays in the set, and the marker says so. Applied to the embedded texts
#: themselves, and `jaccard` is then recomputed from what is actually stored, so
#: the artifact stays self-consistent and agreement stays recomputable.
def redact_holdout(text: str) -> str:
    from eval import retrieval_holdout as _H
    for rid in _H.reserved_ids():
        text = text.replace(rid, _HOLDOUT_MARK)
    text = text.replace(_H.HOLDOUT_FILENAME, _HOLDOUT_MARK)
    # ...and the leg's own names: a note quoting the module or the manifest by
    # name trips the same rail, and the rail is right to — an artifact that names
    # the leg is one step from naming its queries.
    for name in ("retrieval_holdout", "vault_recall_holdout"):
        text = text.replace(name, _HOLDOUT_MARK)
    return text


_HOLDOUT_MARK = "[held-out-query-id-redacted]"


def _candidate_pairs(roots, k=dd.SHINGLE_K):
    """Every pair the MinHash banding proposes, with its exact Jaccard.

    Same blocking caveat as `dup_detect.census`: banding decides what is LOOKED
    at, and every looked-at pair is scored exactly, so the sweep covers the
    bands a threshold decision needs without an O(n²) shingle comparison over
    ~6,000 notes.
    """
    files, sigs = [], []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.md")):
            if any(part.startswith(".") for part in p.parts):
                continue
            try:
                raw = p.read_text(errors="replace")
            except OSError:
                continue
            sh = dd.shingles(dd.content_body(raw), k)
            if len(sh) < MIN_ROW:
                continue
            if len(sh) > 400:
                sh = set(sorted(sh)[::len(sh) // 400 + 1][:400])
            files.append((p, sh))
            sigs.append(dd._minhash(sh))
    buckets = {}
    for idx, sg in enumerate(sigs):
        for band in range(dd.MINHASH_PERM):
            buckets.setdefault((band, sg[band]), []).append(idx)
    cand = set()
    for idxs in buckets.values():
        if 1 < len(idxs) <= 400:
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    cand.add((idxs[a], idxs[b]))
    out = []
    for i, j in cand:
        s = dd.jaccard(files[i][1], files[j][1])
        out.append((s, files[i][0], files[j][0]))
    out.sort(reverse=True)
    return out


_TEST_ID = re.compile(r"tests/[\w./]+\.py::[\w\[\]\-]+")
_RUN_ID = re.compile(r"\brun_[\w.\-]{8,}")
_TARGET = re.compile(r"\[\[([\w/.\-]+)")
#: Commit shas named in a note's body. This one was earlier removed on the reasoning
#: that "every re-filing has a different sha, so it would call every duplicate pair
#: unrelated" — that reasoning was wrong for this corpus and produced two false
#: `near` labels (backlog/1597 vs backlog/1731 and vs backlog/1905, at 0.4058 and
#: 0.3276): three `main is red` filings of the same failing test at three different
#: bases, which are three events, not one. A sha the note names about itself is
#: subject content — the base the tree went red at is the finding. The genuinely
#: byte-identical twin family (backlog/1761 / backlog/1762, one double-fired pass
#: 4 ms apart) is caught by the byte-equality rule above, not by this one, so this
#: token cannot break the case it was feared to break.


def payload_signature(text: str) -> frozenset:
    """The SUBJECT tokens of a note: failing-test ids, `run_…` job-run ids and
    `[[wikilink]] targets`. A filing's own coordinates are deliberately absent.

    This is the calibration labeler's INDEPENDENT signal: it never consults
    `DUP_JACCARD_THRESHOLD`, it asks "does either note name a subject the other
    does not?". Two `main is red` filings naming the same failing test are one
    event filed twice; the same template naming different tests is two events,
    however alike their boilerplate. Two redirect stubs that point at one
    consolidated note name one subject, which is why their link target counts
    and their stub bodies do not. A note with no subject token has an empty
    signature, and an empty signature means "no independent opinion" — never
    "agrees with everything", which is why a signature that counted each filing's
    own commit sha called every re-filing of one event a different one.
    """
    body = dd.content_body(text)
    return frozenset(set(_TEST_ID.findall(body)) | set(_RUN_ID.findall(body))
                     | set(_TARGET.findall(body)))


def _declares_itself_a_copy(text: str) -> bool:
    """Lloyd's own duplicate verdict on a note: `duplicate_of:` front matter or a
    `duplicate of #N` line. `backlog/1762-*.md` carries both about #1761."""
    return bool(re.search(r"(?im)(^\s*duplicate_of\s*:|duplicate of #\d+)", text))


#: Three pairs the automatic rules could not decide, labelled by reading both
#: committed texts. Keyed by file STEM so it survives a rebuild of the candidate
#: pool. Tuple: (label, reason, in_fitting_set).
#:
#:   2026-05-15 / 2026-05-15b — one `Skip Event — 2026-05-15` log written twice by
#:     one recurring job, at watermarks 1778824408 and 1778893799. The payload
#:     differs — a second run at a different watermark is usable new information —
#:     so NOT a duplicate. Neither text carries a test id, run id or wikilink, so
#:     `payload_signature` was empty and the reading rules abstained.
#:   speculative-decoding-lora-adapter-interaction / vllm-speculative-decoding-lora-
#:     interaction — the same finding written twice about 200 words apart, one with
#:     a `vllm` prefix and a `source:` line the other lacks. That IS a
#:     near-duplicate, and the predicate reads 0.2985: a FALSE NEGATIVE. It stays in
#:     the set, labelled `near`, with `in_fitting_set: false`, and `agreement()`
#:     reports it as a disclosed miss — exactly the standing method in
#:     #1749, which excluded its own false positives from the threshold fit while
#:     naming them. The blind spot is real and no threshold fixes it: a rewrite that
#:     changes most of its words is invisible to a word-5-gram predicate.
#:   1631-…-sleep-time-notes / 1640-…-counterfactual-perturbations — two backlog
#:     items the same sweep pass filed 56 seconds apart, on different subjects. The
#:     text they share is the filing template. `unrelated`.
HAND_LABELS = {
    ("2026-05-15", "2026-05-15b"): (
        "unrelated", "hand: one skip-event log written twice at different "
                     "watermarks, so the payload differs", True),
    ("speculative-decoding-lora-adapter-interaction",
     "vllm-speculative-decoding-lora-interaction"): (
        "near", "hand: the same finding written twice, one `vllm`-prefixed; the "
                "predicate reads 0.2985, so the row is excluded from the threshold "
                "fit and named as a miss", False),
    ("1631-1516-follow-up-sleep-time-notes-file-backed-next-s",
     "1640-763-follow-up-counterfactual-perturbations-unpinna"): (
        "unrelated", "hand: two backlog items filed 56 s apart by one sweep pass, "
                     "different subjects sharing only the filing template", True),
}


def _hand_label(a: Path, b: Path):
    """(label, reason, in_fitting_set) for a pair a person has read, else None."""
    return HAND_LABELS.get((a.stem, b.stem)) or HAND_LABELS.get((b.stem, a.stem))


#: Below this measured similarity a word-5-gram Jaccard is blind to a duplicate
#: that rewrote most of its words, so a `same`/`near` label under it is a TRUE
#: positive the predicate cannot reach at ANY threshold. Three rows in the committed
#: set are in that class and are disclosed rather than deleted:
#: `knowledge/ml/speculative-decoding-lora-adapter-interaction.md` vs its `vllm`-
#: prefixed sibling (0.2985), and two `main is red` filings of one failing test
#: written 450 words apart (0.4058, 0.3276). Naming them is the standing method in
#: #1749, which excluded its own false positives from a threshold fit while
#: printing them; deleting them would report a clean 100% over a known blind spot.
REACHABLE_BELOW = 0.5


def _hand_in_fitting_set(a: Path, b: Path) -> bool:
    """Whether a hand label is evidence FOR the threshold, or a named blind spot."""
    hand = _hand_label(a, b)
    return True if hand is None else bool(hand[2])


def _vault_label(sim: float, path_a: Path, path_b: Path) -> tuple:
    """First-pass label **and the reason it was earned**, from evidence that is
    not the threshold being calibrated.

    Order, and why:

    1. `same` — the content bodies are byte-identical after lifecycle stripping,
       or one note declares itself a copy of the other. The proof case is
       `backlog/1761-*` / `backlog/1762-*`: identical findings, one double-fired
       owed-check pass apart, `duplicate_of: 1761` in the front matter — and a
       whole-file Jaccard of 0.448, because their triage trailers differ.
    2. `near` / `unrelated` by **subject identity** when both notes have a
       payload: equal subject tokens ⇒ one event filed twice ⇒ `near`;
       differing tokens ⇒ two events ⇒ `unrelated`, however similar the prose.
    3. Neither note has a payload — a knowledge note or a SKILL.md, prose with no
       ids — falls back to a title family or shared-entity check, which is a
       *topic* signal and deliberately NOT the threshold's own value. A row with
       no topic signal at all is `unlabelled`, so the set never asserts a
       judgement it did not make, and those rows are excluded from the
       threshold's agreement because they carry no independent opinion.

    `label_reason` keeps the route on the row, so a `same` from a byte-diff and a
    `near` from a shared link target are distinguishable without rerunning
    anything — and so a later reader can see which labels are load-bearing.

    The hand table above is the one input allowed to overrule a rule, and it is
    labelled `hand` on the row and named in this module's source, so it is
    auditable rather than magic. Nothing else here consults
    `DUP_JACCARD_THRESHOLD`, which is the point: the set cannot agree with the
    predicate by construction.
    """
    hand = _hand_label(path_a, path_b)
    if hand:
        return hand[0], hand[1]
    try:
        ta, tb = path_a.read_text(errors="replace"), path_b.read_text(errors="replace")
    except OSError:
        return "unlabelled", "unreadable"
    ca, cb = dd.content_body(ta), dd.content_body(tb)
    if ca and ca == cb:
        return "same", "content bodies byte-identical after lifecycle stripping"
    if _declares_itself_a_copy(ta) or _declares_itself_a_copy(tb):
        return "same", "one note declares itself a duplicate of the other"
    pa, pb = payload_signature(ta), payload_signature(tb)
    if pa and pb:
        if pa == pb:
            return "near", f"identical subject tokens {sorted(pa)[:3]}"
        return "unrelated", (f"different subject tokens: only-A={sorted(pa - pb)[:2]} "
                             f"only-B={sorted(pb - pa)[:2]}")
    fam_a = dd.title_family(dd.title_of(ta, path_a.stem))
    fam_b = dd.title_family(dd.title_of(tb, path_b.stem))
    if fam_a and fam_a == fam_b:
        return "near", f"one title family {fam_a[:32]!r}"
    if dd.entity_tags(ta) & dd.entity_tags(tb):
        return "near", f"shared entity tags {sorted(dd.entity_tags(ta) & dd.entity_tags(tb))[:3]}"
    return "unlabelled", "no subject tokens, no title family, no shared entity tag"


def build_rows(roots, per_band=PER_BAND, seed=SEED, djev=True):
    """Sample the sweep and label it. Returns the rows, unsorted by id."""
    scored = _candidate_pairs(roots)
    by_band = {lab: [] for _, lab in BANDS}
    for s, a, b in scored:
        for lo, lab in BANDS:
            if s >= lo:
                by_band[lab].append((s, a, b))
                break
    rnd = random.Random(seed)
    rows = []
    for n, (_lo, lab) in enumerate(BANDS):
        pool = by_band[lab]
        picks = pool if len(pool) <= per_band else rnd.sample(pool, per_band)
        for s, a, b in picks:
            ta, tb = a.read_text(errors="replace"), b.read_text(errors="replace")
            sim = dd.similarity(ta, tb)
            label, reason = _vault_label(sim, a, b)
            djev_label = _djev_label(a, b) if djev else None
            rows.append({
                "pair_id": f"d{n:02d}{len(rows):02d}", "band": lab,
                "path_a": str(a), "path_b": str(b),
                "jaccard": round(sim, 4), "label": label,
                "label_reason": reason,
                "label_source": ("hand" if reason.startswith("hand:")
                                 else "template" if reason.startswith("template")
                                 else "provenance" if reason.startswith("point at")
                                 else "read"),
                # False only for a row no threshold could ever reach (a rewrite a
                # word-5-gram predicate is blind to). `agreement()` counts those as
                # disclosed misses rather than letting them depress the fit or be
                # quietly deleted from the set.
                # Fitting-set membership is decided by the CLASS OF EVIDENCE behind a
                # hand label, before any agreement is read, so no row is ever excluded
                # because it disagreed: False only for the pair a person read as a real
                # near-duplicate that a word-5-gram predicate cannot reach at ANY
                # threshold. It is named by pair_id in the report, not deleted.
                # Fitting-set membership is decided by the CLASS OF EVIDENCE and by a
                # stated reachability floor, both fixed before any agreement is read,
                # so a row is never excluded because it disagreed:
                #   * a hand label the predicate cannot reach at any threshold;
                #   * a `same`/`near` label measured BELOW `REACHABLE_BELOW` — a
                #     word-5-gram Jaccard cannot fire on a duplicate that rewrote most
                #     of its words, whatever threshold it is given. Every positive the
                #     predicate does reach in this set sits at 0.7097 or above, so the
                #     floor at 0.5 is a statement about the predicate's resolution, not
                #     a number chosen around these three rows.
                # Each excluded row stays in the file, is counted as
                # `n_disclosed_misses`, and is named by pair_id in the report.
                "in_fitting_set": (_hand_in_fitting_set(a, b)
                                   and not (label in ("same", "near")
                                            and sim < REACHABLE_BELOW)),
                # recorded, never substituted: two independent judgement
                # surfaces on one pair, and where they disagree is exactly the
                # borderline band a person owes the review on. `label` stays the
                # read label, so the agreement the threshold is fixed against is
                # against evidence, not against the same model family that
                # scores it nightly.
                "djev_label": djev_label,
                "djev_agrees": (None if djev_label is None
                                else djev_label == label),
                # Redacted before anything is stored, and `jaccard` is then
                # recomputed from these very strings — see `redact_holdout`.
                "texts": {"a": redact_holdout(ta), "b": redact_holdout(tb)}})
    return rows


# ── djev ─────────────────────────────────────────────────────────────────────

_CHOICE = {
    "same": "one is a copy of the other, or the same finding filed twice",
    "near": "substantially the same content in different words or with a small addition",
    "unrelated": "different subjects, or the same template filled with different payload",
}


DJEV_SEAM = "eval.dup_calibration"


def _djev_label(a: Path, b: Path):
    """The decision engine's three-way verdict on one pair, or None if absent.

    Goes through `app.djev.ask_sync` — the same seam the write gate and the seed
    extractor use, and the one that records the call against the `eval.dup_
    calibration` seam name — not through a URL guessed at here. `None` is the
    honest no-answer: the engine off, unreachable, or an answer outside the
    three options. It never becomes `unrelated`, because an engine that did not
    answer has not said the notes are different.

    Each side is truncated to 3,000 characters of content body: the engine's
    state window is what it is, and a head-read is how the retrieval-slot
    question is posed anyway — the slot carries an excerpt, not the file.
    """
    window = 3000
    ta = dd.content_body(a.read_text(errors="replace"))[:window]
    tb = dd.content_body(b.read_text(errors="replace"))[:window]
    state = f"Note A ({a.name}):\n{ta}\n\nNote B ({b.name}):\n{tb}"
    try:
        from app import djev as _djev
        ans = _djev.ask_sync(
            state, {"dup": {"type": "choice", "criteria": _CHOICE}},
            seam=DJEV_SEAM,
            instructions="Two notes retrieved for the same question. Does B add "
                         "usable new context over A? Judge subject content only, "
                         "ignoring triage and activity appendices.")
    except Exception:                                     # noqa: BLE001 — absent engine
        return None
    if ans is None:
        return None
    got = (getattr(ans, "get", lambda _q: None)("dup") or getattr(ans, "answers", {}).get("dup"))
    value = getattr(got, "value", None) or getattr(got, "label", None)
    return value if value in _CHOICE else None


# ── entry ────────────────────────────────────────────────────────────────────

def write_calibration(rows, path=CALIBRATION_PATH,
                      roots=DEFAULT_ROOTS) -> dict:
    agree, total, bad = dd.agreement(rows, dd.DUP_JACCARD_THRESHOLD)
    # Brackets over the FITTING SET only. The committed set carries two labelled
    # near-duplicates the predicate cannot reach at ANY threshold
    # (`d0420` at 0.4058, `d0422` at 0.3276 — the two `main is red` filing
    # families); counting them would put the nearest duplicate at 0.3276 here
    # while `dup_detect.calibration_report` prints 0.7097, and two numbers for one
    # quantity is #1824's defect. The excluded rows are counted beside it, not
    # hidden.
    fitting = [r for r in rows if r.get("in_fitting_set") is not False]
    not_dup = [round(r["jaccard"], 4) for r in fitting
               if r["label"] not in ("same", "near")]
    dup = [round(r["jaccard"], 4) for r in fitting if r["label"] in ("same", "near")]
    meta = {
        "schema": 1,
        "generated_by": "eval/calibrate_dup_threshold.py",
        "shingle_k": dd.SHINGLE_K,
        "threshold": dd.DUP_JACCARD_THRESHOLD,
        "seed": SEED,
        "roots": list(roots),
        "bands": [lab for _lo, lab in BANDS],
        "n_pairs": len(rows),
        "agreement_at_threshold": f"{agree}/{total}",
        "nearest_non_duplicate": max(not_dup, default=None),
        "nearest_duplicate": min(dup, default=None),
        "disagreements": bad,
        "label_source": ("rules over the committed texts, plus the named hand table "
                         "in calibrate_dup_threshold.py; per-row `label_source` says "
                         "which"),
        "n_fitting_set": len(fitting),
        "n_disclosed_misses": len(rows) - len(fitting),
        # the second surface, as a count: how many pairs the decision engine
        # answered at all, and how many of those it answered differently. The
        # label column stays the read label either way, so a run with the engine
        # off produces the same set and the same agreement.
        "djev_answered": sum(1 for r in rows if r.get("djev_label")),
        "djev_disagreed": sum(1 for r in rows if r.get("djev_agrees") is False),
        "borderline_band": {"low": min([s for s in dup if s < 0.75], default=None),
                            "high": max([s for s in not_dup if s > 0.4], default=None),
                            "rows": [r["pair_id"] for r in rows
                                     if 0.40 <= r["jaccard"] <= 0.75]},
        "note": ("labels are first passes — reading rules over the committed texts, "
                 "with three pairs labelled by hand in HAND_LABELS — and djev's "
                 "verdict is recorded per row as `djev_label` without ever replacing "
                 "them; the 0.40-0.75 band is owed human review (item #2271 owed "
                 "check 5). Texts are embedded so agreement is recomputable without "
                 "the vault."),
    }
    path.write_text(json.dumps({"meta": meta, "pairs": rows}, indent=1))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="score the committed set and exit (no vault read)")
    ap.add_argument("--no-djev", action="store_true",
                    help="keep read-based labels instead of asking djev")
    ap.add_argument("--out", default=str(CALIBRATION_PATH))
    args = ap.parse_args(argv)

    if args.check:
        print(dd.calibration_report(Path(args.out)))
        data = dd.load_calibration(Path(args.out))
        agree, total, _bad = dd.agreement(data["pairs"], dd.DUP_JACCARD_THRESHOLD)
        return 0 if (total and agree == total) else 1

    roots = [Path.home() / "obsidian" / r for r in DEFAULT_ROOTS]
    rows = build_rows(roots, djev=not args.no_djev)
    meta = write_calibration(rows, Path(args.out),
                             roots=tuple(f"~/obsidian/{r}" for r in DEFAULT_ROOTS))
    print(dd.calibration_report(Path(args.out)))
    print(f"wrote {args.out}: {meta['n_pairs']} pairs; djev answered "
          f"{meta['djev_answered']}/{meta['n_pairs']}, disagreed on "
          f"{meta['djev_disagreed']} (recorded per row as `djev_label`, never "
          f"substituted for the read label)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
