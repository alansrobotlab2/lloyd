"""The ADD / UPDATE / NOOP decision in front of a fact write (#1487).

#499 refuses a byte-identical re-statement of a fact the entity already holds
(`(entity, text_hash)`, `agent_mcp/facts.py::_store_copy_of`). A paraphrase
walks straight through it, and paraphrase is how the writers duplicate: on the
live store 2026-09-25, of the 10,615 facts written after the 09-23 rebuild,
1,867 had a same-entity, same-category prior fact with token Jaccard >= 0.7 —
the nightly extractor re-reading a changed document states the same claims
again in new words, and the re-armed post-capture writes `stream_chat` four
times from four sessions. This module is the Mem0-shaped gate for that, built
on djev (`app/djev.py`), in three steps:

1. **Shortlist, lexically, with no model call.** The entity's ACTIVE facts in
   the same category, scored by token Jaccard against the new text; those at
   or above `SHORTLIST_FLOOR`, best `SHORTLIST_K`. An empty shortlist is ADD
   and djev is never asked — most writes stop here.
2. **One djev decision over the shortlist** — the lexical best candidate, one
   `choice` question (`QUESTION` below, negative option first, the ordering
   `eval/djev/schemas.py` measured better for pair questions).
3. **A calibrated cutoff per verdict.** djev's probabilities are an ordering,
   not a probability (`app/djev.py`'s module docstring: "a fixed 0.5 cutoff is
   meaningless"), so NOOP and UPDATE each carry their own threshold, measured on
   a hand-labelled replay (`eval/measurements/fact-write-gate-2026-09-25.md`).
   A write clears a threshold or it is ADD.

What the result may do is bounded by `mode()`:

* ``off`` (the default) — nothing here runs.
* ``shadow`` — decide and log, write as today.
* ``noop`` — apply NOOP (nothing written), record UPDATE as ADD. The scope
  call the item left to a person is whether a djev label may expire a fact at
  all (#499 recorded `fact_entity_recall` 0.35 → 0.30 from retiring the wrong
  copy), so NOOP — which destroys nothing — is its own step.
* ``on`` — NOOP and UPDATE both apply. UPDATE supersedes: the new fact is
  appended and the ONE prior fact the decision named is stamped `expired_at`.
  Never a delete.

**Every failure is ADD.** djev unreachable, timing out, answering malformed,
switched off, splitting the canvas, or `low_trust` on the answer that would
decide: the write proceeds exactly as it would with the gate off. A gate that
drops a real fact is worse than a duplicate.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger("lloyd-fact-write-gate")

MODES = ("off", "shadow", "noop", "on")
MODE_ENV = "LLOYD_FACT_WRITE_GATE"
SEAM = "fact_write"

#: Token Jaccard at or above which an existing fact is put to djev at all.
#: On the 09-25 labelled pairs 2 of 16 in the 0.2-0.3 band were duplicates,
#: against 14 of 36 at 0.3-0.5 and 33 of 36 at 0.5-0.7; the floor gives up
#: that thin band so djev is asked on ~40% of writes, not ~50%.
SHORTLIST_FLOOR = 0.30
#: Candidates per canvas: ONE, the lexical best. Measured on the same 400
#: held-out writes: with up to 6 on one canvas djev read "restated" onto
#: several candidates at once and 12 of 81 NOOPs dropped a detail the new fact
#: carried (0.852 right); with the top one alone, 76 of 77 were right (0.987)
#: and the NOOP count barely moved (81 -> 77). It is also the shape the
#: thresholds were calibrated on.
SHORTLIST_K = 1
#: Characters of each fact on the canvas. Facts are one sentence; this only
#: stops a pathological one from widening the prompt.
FACT_CHARS = 400
#: djev is on the write path: a stall must cost a duplicate, never a write.
TIMEOUT_S = 3.0

#: The frozen question. Option ORDER is part of it (negative first); changing
#: wording or order invalidates the thresholds below — re-label and re-run
#: `eval/run_fact_write_gate_eval.py` before trusting them.
QUESTION = {
    "type": "choice",
    "criteria": {
        "different": "they state different claims, or each says something "
                     "the other does not: keep both",
        "restated": "the existing fact already says everything the new fact "
                    "says: the new fact adds nothing",
        "superseded": "the new fact says everything the existing fact says "
                      "and more, or corrects it: the existing fact is replaced",
    },
}
#: No canvas-wide instructions: the thresholds below were measured without
#: any, and a sentence here is part of the frozen schema like the options.
INSTRUCTIONS: str | None = None

#: Calibrated cutoffs on P(label), measured 2026-09-25 on 160 hand-labelled
#: replay pairs and held out on a production-shape replay —
#: `eval/measurements/fact-write-gate-2026-09-25.md`. `None` means not
#: calibrated, and an uncalibrated verdict never applies (ADD).
#:
#: NOOP at 0.8: on the labelled pairs P(restated) separates "the new fact
#: adds nothing" at AUC 0.958, and at 0.8 it was right on 74 of 76.
NOOP_THRESHOLD: float | None = 0.8
#: UPDATE expires a fact, and djev alone is not good enough for it: at every
#: cutoff P(superseded) put 18-38% of its supersedes on a pair where the old
#: fact held something the new one lacked (a different date, a status the new
#: one contradicts). So an UPDATE also needs `contains(new, old)` — every word
#: and number of the old fact appears in the new one — which makes "nothing
#: the old fact said is lost" a property checked in Python rather than a
#: model's opinion, with djev as the second vote at a low cutoff.
UPDATE_THRESHOLD: float | None = 0.3
#: Below this `label_mass` the deciding answer is not trusted (ADD).
LABEL_MASS_FLOOR: float | None = 0.3

#: Function words a paraphrase may drop without dropping information. Short on
#: purpose, and without "not"/"no": a negation is information.
_CONTAIN_STOP = frozenset(
    "a an the and or of to in on at for with by from as is are was were be been "
    "has have had its it this that into onto".split())
_ALNUM = re.compile(r"[a-z0-9]+")


def contains(new_text: str, old_text: str) -> bool:
    """Every content word and number of `old_text` also appears in `new_text`,
    and `new_text` says more.

    Numbers count however short they are: "released 2026-05-16" against
    "released 2026-07-06" differs only in two-digit tokens, and that pair is
    exactly the wrong supersede this check exists to refuse.
    """
    old = {w for w in _ALNUM.findall((old_text or "").lower()) if w not in _CONTAIN_STOP}
    new = {w for w in _ALNUM.findall((new_text or "").lower()) if w not in _CONTAIN_STOP}
    return bool(old) and old <= new and len(new) > len(old)

_STOP = frozenset(
    "the and for with that this from are was were has have had its into onto "
    "than then but not all any can will would should could been being also "
    "uses used use via per when what which who how does did".split())
_WORD = re.compile(r"[a-z0-9_]+")


def tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall((text or "").lower())
                     if len(w) >= 3 and w not in _STOP)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mode() -> str:
    """`LLOYD_FACT_WRITE_GATE`, else `knowledge_graph.write_gate.mode`, else off.

    An unknown value reads as ``off``: a typo must not arm a gate that can
    expire facts.
    """
    raw = os.environ.get(MODE_ENV)
    if raw is None:
        try:
            from app.config import CONFIG
            raw = ((CONFIG.get("knowledge_graph") or {}).get("write_gate") or {}).get("mode")
        except Exception:  # noqa: BLE001
            raw = None
    raw = str(raw or "off").strip().lower()
    return raw if raw in MODES else "off"


@dataclass
class Decision:
    """What the gate concluded about one write, and why."""

    verdict: str = "add"                 # add | noop | update
    reason: str = ""                     # no_candidates | djev_failed | below_threshold | ...
    target: dict | None = None           # the existing fact a noop/update names
    candidates: list[dict] = field(default_factory=list)
    answers: dict[str, dict] = field(default_factory=dict)
    latency_ms: float | None = None
    asked: bool = False

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason,
                "target": self.target, "asked": self.asked,
                "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
                "candidates": [{k: c.get(k) for k in ("fact_id", "fact", "jaccard")}
                               for c in self.candidates],
                "answers": self.answers}


def shortlist(fact_text: str, existing: Iterable[Mapping], *,
              floor: float | None = None, k: int | None = None) -> list[dict]:
    """The active existing facts worth asking djev about, best first."""
    floor = SHORTLIST_FLOOR if floor is None else floor
    k = SHORTLIST_K if k is None else k
    new = tokens(fact_text)
    scored = []
    for f in existing:
        if not isinstance(f, Mapping):
            continue
        if f.get("expired_at") or f.get("invalid_at"):
            continue
        text = str(f.get("fact") or "")
        j = jaccard(new, tokens(text))
        if j >= floor:
            scored.append({**dict(f), "jaccard": round(j, 4)})
    scored.sort(key=lambda c: c["jaccard"], reverse=True)
    return scored[:k]


def state_for(entity: str, category: str, fact_text: str,
              candidates: Sequence[Mapping]) -> str:
    body = "\n".join(f"[{i}] {str(c.get('fact') or '')[:FACT_CHARS]}"
                     for i, c in enumerate(candidates))
    return (f"Entity: {entity}\nCategory: {category}\n\n"
            f"EXISTING facts:\n{body}\n\nNEW fact: {fact_text[:FACT_CHARS]}")


def questions_for(candidates: Sequence[Mapping]) -> dict[str, dict]:
    return {f"c{i}": {**QUESTION,
                      "instructions": f"How does the NEW fact relate to existing fact [{i}]?"}
            for i in range(len(candidates))}


def ask(entity: str, category: str, fact_text: str,
        candidates: Sequence[Mapping], *, timeout: float = TIMEOUT_S):
    """The raw djev answers for one write, or None on any failure."""
    from app import djev
    out = djev.ask_sync(state_for(entity, category, fact_text, candidates),
                        questions_for(candidates), timeout=timeout, seam=SEAM,
                        instructions=INSTRUCTIONS)
    if out is None or out.cross_chunk:
        return None
    return out


def verdict_from(answers: Mapping[str, Mapping], candidates: Sequence[Mapping],
                 fact_text: str, *,
                 noop_threshold: float | None = NOOP_THRESHOLD,
                 update_threshold: float | None = UPDATE_THRESHOLD,
                 label_mass_floor: float | None = LABEL_MASS_FLOOR) -> tuple[str, str, int | None]:
    """(verdict, reason, candidate index) from per-candidate answer dicts.

    Pure, so the offline replay and the write path decide identically. NOOP is
    checked first: of the two, it is the one that destroys nothing.
    """
    def prob(i: int, label: str) -> float | None:
        a = answers.get(f"c{i}")
        if not a:
            return None
        if label_mass_floor is not None and float(a.get("label_mass", 0.0)) < label_mass_floor:
            return None
        return float((a.get("probabilities") or {}).get(label, 0.0))

    if noop_threshold is None and update_threshold is None:
        return "add", "uncalibrated", None
    if noop_threshold is not None:
        scored = [(p, i) for i in range(len(candidates))
                  if (p := prob(i, "restated")) is not None]
        if scored:
            p, i = max(scored)
            if p >= noop_threshold:
                return "noop", f"restated p={p:.3f}", i
    if update_threshold is not None:
        # Only a candidate the new text contains may be expired, so the best
        # superseded one among THOSE — not the argmax, which could hide one.
        scored = [(p, i) for i in range(len(candidates))
                  if (p := prob(i, "superseded")) is not None
                  and contains(fact_text, str(candidates[i].get("fact") or ""))]
        if scored:
            p, i = max(scored)
            if p >= update_threshold:
                return "update", f"superseded p={p:.3f}, contained", i
    return "add", "below_threshold", None


def decide(entity: str, category: str, fact_text: str,
           existing: Iterable[Mapping], *, timeout: float = TIMEOUT_S) -> Decision:
    """The gate's decision for one write. Never raises."""
    try:
        cands = shortlist(fact_text, existing)
        if not cands:
            return Decision(reason="no_candidates")
        t0 = time.perf_counter()
        out = ask(entity, category, fact_text, cands, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1e3
        if out is None:
            return Decision(reason="djev_failed", candidates=cands, asked=True,
                            latency_ms=latency)
        answers = {k: a.as_dict() for k, a in out.answers.items()}
        verdict, reason, idx = verdict_from(answers, cands, fact_text)
        target = None
        if idx is not None:
            c = cands[idx]
            target = {"fact_id": c.get("id") or c.get("fact_id"),
                      "fact": c.get("fact"), "jaccard": c.get("jaccard")}
        return Decision(verdict=verdict, reason=reason, target=target,
                        candidates=cands, answers=answers, latency_ms=latency,
                        asked=True)
    except Exception as exc:  # noqa: BLE001 — every failure is ADD
        logger.debug("fact write gate failed open: %s", exc)
        return Decision(reason=f"error:{type(exc).__name__}")


def gate_write(entity: str, category: str, fact_text: str, existing: list,
               *, mode_: str | None = None, now_iso: str | None = None,
               source_doc: str | None = None) -> tuple[str, Decision | None]:
    """Decide one write against the file's own fact list and apply what `mode`
    allows. Returns (the verdict the write TOOK, the decision or None).

    Called inside the caller's file lock with the list it is about to write
    back, so an UPDATE is one more mutation of that list — the target's
    `expired_at` stamped in place, never removed — and lands in the same
    atomic write as the new fact. `existing` is the category file's list:
    one file per (entity, category) is exactly "the entity's facts in the
    same category".
    """
    m = mode_ or mode()
    if m == "off":
        return "add", None
    d = decide(entity, category, fact_text, existing)
    took = d.verdict
    if m == "shadow" or (m == "noop" and took == "update"):
        took = "add"
    if took == "update":
        target = _find(existing, d.target)
        if target is None:
            took = "add"            # the named fact moved; never guess another
        else:
            target["expired_at"] = now_iso or datetime.datetime.now(
                datetime.timezone.utc).isoformat()
    log_decision(entity, category, fact_text, d, mode_=m, applied=took,
                 source_doc=source_doc)
    return took, d


def _find(existing: list, target: Mapping | None):
    """The one active entry the decision named: same id AND same text."""
    if not target:
        return None
    hits = [f for f in existing if isinstance(f, dict)
            and f.get("id") == target.get("fact_id")
            and str(f.get("fact") or "") == str(target.get("fact") or "")
            and not (f.get("expired_at") or f.get("invalid_at"))]
    return hits[0] if len(hits) == 1 else None


def log_decision(entity: str, category: str, fact_text: str, decision: Decision,
                 *, mode_: str, applied: str, source_doc: str | None = None) -> None:
    """One jsonl row per asked decision, beside the store it judged."""
    if not decision.asked:
        return
    try:
        from app.paths import FACT_WRITE_GATE_LOG
        row = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "mode": mode_, "applied": applied, "entity": entity,
               "category": category, "fact": fact_text, "source_doc": source_doc,
               **decision.as_dict()}
        FACT_WRITE_GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(FACT_WRITE_GATE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — the log never costs the write
        pass
