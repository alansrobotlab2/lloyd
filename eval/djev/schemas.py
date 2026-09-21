"""The frozen question schemas, one per seam, with what was measured about each.

A djev schema is not configuration. Measured on 2026-09-20 over 40 labelled
pairs, four framings of one question:

    framing                              AUC    acc@0.5  best acc  optimal thr
    choice, "distinct" listed first      0.833  0.80     0.82      0.39
    choice, "related" listed first       0.795  0.50     0.75      0.03
    choice, mean of both orders          0.815  0.53     0.80      0.30
    noul (no option list at all)         0.723  0.50     0.55      0.01

Three things follow, and all three are the reason this file exists rather than
a `questions=` literal at each call site:

1. **Option ORDER is part of the schema.** Reversing it moved P by 0.324 on
   average and by 0.851 at worst, and dropped accuracy at a fixed 0.5 cutoff
   from 0.80 to 0.50. The hash below is computed over the question JSON with
   key order preserved, so re-ordering options changes the hash and
   invalidates the threshold measured against the old one.
2. **Order-averaging is not the fix.** It lands between the two arms and
   `acc@0.5` stays 0.53. A threshold measured per frozen schema is the fix.
3. **`noul` is not automatically safer than `choice`.** The yes/no form has
   no option list to order, and it measured *worst*. Every framing here was
   picked by measurement, never by reasoning about it.

`threshold` and `label_mass_floor` both start `None`, and `None` is a real
state rather than a placeholder: a schema with no threshold may be used for
RANKING and must not be used as a gate, and a schema with no floor reports
`label_mass` on every answer while flagging nothing. A fixed 0.5 floor would
trip on real requests inside the window the design calls safe — the listwise
runs at n=16 measured minimums of 0.446, 0.807 and 0.965 on three different
corpora.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


class SchemaDrift(Exception):
    """The wording or option order moved after a threshold was measured."""


@dataclass(frozen=True)
class Schema:
    """One frozen question shape and everything measured about it."""

    name: str
    #: Everything that determines the wording the model sees, INCLUDING option
    #: order. Serialized with key order preserved — see `hash`.
    spec: dict[str, Any]
    #: The yes/no cutoff, measured on `calibrated_on`. `None` = not calibrated,
    #: which means this schema ranks and does not gate.
    threshold: float | None = None
    #: Below this, `label_mass` says the model answered somewhere other than
    #: the label set. `None` = no floor measured yet.
    label_mass_floor: float | None = None
    #: What the two numbers above were measured against, named so a reader can
    #: tell a 40-pair result from a 1,040-row one.
    calibrated_on: str = ""
    #: The hash the numbers above were measured under. Empty until calibrated;
    #: a mismatch against `hash` is `SchemaDrift`.
    calibrated_hash: str = ""
    notes: str = ""
    #: Where this schema's decision is made in production, for the shadow log.
    seam: str = ""
    #: May this schema's `threshold` gate a production decision? False for
    #: every schema in this file, and it is a separate field from
    #: `threshold is not None` on purpose: a measured threshold and a
    #: threshold that may DECIDE something are different claims, and
    #: collapsing them is how "we calibrated it against another model's
    #: opinions" becomes "we calibrated it".
    gate_ready: bool = False
    #: Why not, in one sentence a reader can act on.
    gate_blocked_reason: str = ""

    @property
    def hash(self) -> str:
        """sha1 over the spec with **key order preserved**.

        `sort_keys=False` is the entire point and is not an oversight: the
        options of a `choice` are a dict whose insertion order is the order
        the model reads them in, and that order moved P by 0.324 on average.
        Sorting here would make the one change most likely to invalidate a
        threshold invisible to the check that exists to catch it.
        """
        blob = json.dumps(self.spec, sort_keys=False, ensure_ascii=False,
                          separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def calibrated(self) -> bool:
        return self.threshold is not None

    def check(self) -> None:
        """Raise if the spec moved since the threshold was measured."""
        if not self.calibrated_hash:
            return
        if self.calibrated_hash != self.hash:
            raise SchemaDrift(
                f"schema {self.name!r} was calibrated at {self.calibrated_hash} "
                f"and now hashes to {self.hash}: re-run "
                f"`eval/djev/replay.py --corpus ... --calibrate` before using "
                f"its threshold")

    def as_dict(self) -> dict:
        return {"name": self.name, "seam": self.seam, "hash": self.hash,
                "threshold": self.threshold,
                "label_mass_floor": self.label_mass_floor,
                "calibrated_on": self.calibrated_on,
                "calibrated_hash": self.calibrated_hash,
                "gate_ready": self.gate_ready,
                "gate_blocked_reason": self.gate_blocked_reason,
                "drifted": bool(self.calibrated_hash
                                and self.calibrated_hash != self.hash)}


# ---------------------------------------------------------------------------
# The schemas
# ---------------------------------------------------------------------------

#: The rank levels, worst to best. `score` returns an expected value over the
#: level INDEX, so reversing this list reverses every ranking silently.
RANK_LEVELS = ["irrelevant", "tangential", "partly answers it",
               "directly answers it"]

RERANK = Schema(
    name="rerank",
    seam="rerank",
    spec={
        "type": "score",
        "levels": RANK_LEVELS,
        "instruction_template": "How well does candidate [{i}] answer the query?",
        "state_template": "Query: {query}\n\nCandidates:\n{candidates}",
    },
    # Ranking needs no threshold, by construction: the answer is an ORDER and
    # the absolute values are never compared to a cutoff. This is why the
    # rerank seam leads — it is the one shape the 2026-09-20 calibration work
    # says is safe today.
    # No threshold, and there is nothing to measure: the answer is an ORDER
    # and no absolute value is ever compared to a cutoff. This is why the
    # rerank seam leads — it is the one shape the calibration work says is
    # safe today.
    threshold=None,
    # UNSET, deliberately, and this is the one place the plan's own rule was
    # tested rather than restated. Listwise `label_mass` at n=12-16 has been
    # measured at 0.446, 0.807, 0.965 and 1.000 across four runs on different
    # corpora — a fourfold spread inside the window the design calls safe. No
    # floor read off any one of them would mean anything for the others, so
    # the SHADOW rows, which are the seam's own requests on the seam's own
    # corpus, set this one. `eval/djev/replay.py --floors` prints the
    # distribution as it accumulates.
    label_mass_floor=None,
    gate_blocked_reason="not a gate: ranking has no cutoff to calibrate",
    notes="Ranking only. MRR 0.766 / recall@1 0.64 over 16 candidates against "
          "0.498 / 0.29 for the lexical leg. The comparison that decides "
          "adoption is against qmd's own reranker through eval/run_eval.py "
          "--djev-rerank, not against Jaccard.",
)

DEDUPE = Schema(
    name="dedupe",
    seam="dedupe",
    spec={
        "same_finding": {
            "type": "choice",
            "instructions": (
                "Two items from an engineering backlog are shown. Do they "
                "describe the SAME underlying finding, such that filing both "
                "is a duplicate?"),
            # The negative listed FIRST, because that arm measured AUC 0.833
            # against 0.795 for the reverse, and an optimal threshold of 0.39
            # against 0.03 — a usable number against one that is noise.
            "criteria": {
                "different": "different findings, even if they touch the same area",
                "same": "the same finding, filed twice",
            },
        },
    },
    # Measured 2026-09-20 on 150 balanced rows of dedupe.jsonl replayed three
    # pairs per canvas — the seam's own request shape. AUC 0.607, acc@0.5
    # 0.587, best cutoff 0.208 for 0.627. That is barely above chance and the
    # number is recorded because it is the evidence, not because it is usable.
    threshold=0.208,
    # min 0.5621, p01 0.6939, p05 0.8152, p50 0.9344 over those 150 reads.
    # 0.50 sits under everything observed healthy, which is what a trust flag
    # that never refuses should mean — "outside the distribution we measured",
    # not "in the bottom 5% of normal traffic".
    label_mass_floor=0.50,
    calibrated_on="150 balanced rows of ~/.local/state/lloyd-automod/dedupe.jsonl, "
                  "3 pairs per canvas (the seam's shape), 2026-09-20: AUC 0.607, "
                  "acc@0.5 0.587, best 0.208 -> 0.627",
    calibrated_hash="e0a817554a6ecdcc",
    gate_blocked_reason=(
        "AUC 0.607 is barely above chance, and the corpus cannot pose the "
        "question properly: dedupe.jsonl records the new item's NAME and "
        "never its description, so the replay compares a title against a "
        "body. That is precisely what the shadow seam fixes — it holds the "
        "description the log threw away — so the usable calibration is the "
        "shadow rows, not this."),
    notes="Shadow only. A merge appends the finding to another item and never "
          "creates the new one, so a wrong 'same' loses a finding's identity.",
)

ENTITY = Schema(
    name="entity",
    seam="entity",
    spec={
        "same_entity": {
            "type": "choice",
            "instructions": (
                "Two entity names from a personal knowledge graph are shown "
                "with their definitions. Do they refer to the SAME real-world "
                "thing? A product and its SDK are different. A company and a "
                "project named after it are different. A robot and its "
                "training pipeline are different. A system and the same "
                "system with 'System' or 'Pipeline' appended are the same."),
            "criteria": {
                "different": "different real-world things",
                "same": "the same real-world thing under two names",
            },
        },
    },
    # Measured 2026-09-20 on 150 balanced rows of semantic-verdicts.jsonl,
    # ONE pair per canvas — the seam's own shape. AUC 0.942, acc@0.5 0.827,
    # best cutoff 0.263 for 0.893 against the gate's own SAME/REVIEW verdict.
    threshold=0.263,
    # min 0.4542, p01 0.5083, p05 0.6200, p50 0.7962 over those 150 reads.
    #
    # And this is the number that proves the design's rule rather than
    # restating it: the SAME 200 pairs replayed EIGHT per canvas measured min
    # 0.053, p01 0.166, p50 0.995. Same schema, same corpus, same engine —
    # the request shape alone moved the distribution by an order of
    # magnitude. A floor lifted from the batch-8 run would never fire here,
    # and a "reasonable" 0.5 would flag 1% of perfectly healthy batch-1 reads
    # and 40% of batch-8 ones. Measure the floor on the shape the seam sends.
    label_mass_floor=0.40,
    calibrated_on="150 balanced rows of _pipeline/memory-graph/semantic-verdicts.jsonl, "
                  "1 pair per canvas (the seam's shape), 2026-09-20: AUC 0.942, "
                  "acc@0.5 0.827, best 0.263 -> 0.893",
    calibrated_hash="c4eb236a0016e456",
    gate_blocked_reason=(
        "the threshold above was measured on ORDINARY pairs and does not "
        "transfer to the hard ones. Replayed over the 137 definition-carrying "
        "pairs of the 151 verified-bad merges — the only true ground truth in "
        "the set — acc@0.5 was 0.562 and 50 of 137 scored above 0.9 while "
        "`label_mass` stayed healthy (p50 0.991): confidently wrong, not "
        "confused. Those 151 are a hard-negative set BY CONSTRUCTION, being "
        "exactly the cases the old rule got wrong, so 0.942 against a judge "
        "and 0.562 against the reverted set are consistent and the second one "
        "is the one a gate would meet."),
    notes="The highest-stakes seam and the only one with true ground truth: "
          "the 151 verified-bad merges in "
          "_pipeline/memory-graph/entity-merges-reverted-20260903T174108Z.json. "
          "Shadow only until that separation is measured; djev as the restored "
          "SECOND judge is the follow-on round this calibration is for.",
)

CLUSTERS = Schema(
    name="clusters",
    seam="",
    spec={
        "related_item": {
            "type": "choice",
            "instructions": (
                "Two items from an engineering backlog are shown. Do they "
                "overlap enough to belong to one piece of work, or are they "
                "distinct pieces of work?"),
            # The exact framing the 2026-09-20 calibration measured at AUC
            # 0.833: `choice`, the negative option listed FIRST.
            "criteria": {
                "distinct": "distinct pieces of work",
                "related": "overlapping enough to be consolidated into one",
            },
        },
    },
    # Measured 2026-09-20 on 200 balanced rows, 8 pairs per canvas: AUC 0.699,
    # acc@0.5 0.660, best cutoff 0.217 for 0.685, and the reliability curve is
    # plainly miscalibrated — the 0.0-0.1 bin observed 0.338 and the 0.9-1.0
    # bin observed 0.690. Order-swap over the same 200: mean |P_A - P_B| 0.245,
    # max 0.947, which reproduces the 40-pair result (0.324 / 0.851) at five
    # times the sample.
    threshold=0.217,
    label_mass_floor=0.80,   # min 0.8775, p01 0.9301, p05 0.9667 over 200 reads
    calibrated_on="200 balanced rows of ~/.local/state/lloyd-automod/cluster_judgments.jsonl, "
                  "8 pairs per canvas, 2026-09-20: AUC 0.699, acc@0.5 0.660, "
                  "best 0.217 -> 0.685; order-swap mean 0.245 / max 0.947",
    calibrated_hash="e077ebc76285150d",
    gate_blocked_reason=(
        "no seam and no gate; this schema exists to calibrate against a "
        "recorded corpus. Note the AUC fell from 0.833 on 40 pairs to 0.699 "
        "on 200 — the larger sample is the truer one."),
    notes="Replay corpus only (cluster_judgments.jsonl, 609 pair verdicts "
          "from the retired 35B pair-judge). Those are a model's judgements, "
          "not ground truth, so this measures AGREEMENT and never "
          "correctness. The question is RELATED vs DISTINCT because that is "
          "what the corpus labels: 382 distinct, 221 related, 6 same, and no "
          "`duplicate` verdict at all — a `duplicate`-vs-`distinct` framing "
          "measured 12 usable pairs out of 609 and reported AUC 1.000 off "
          "the six `same` rows.",
)

EDGES = Schema(
    name="edges",
    seam="",
    spec={
        "edge_type": {
            "type": "choice",
            "instructions": (
                "A relationship was extracted between two entities from the "
                "quoted text. Which relationship type does the quote support?"),
            "criteria": {
                "related_to": "a weak or generic association",
                "uses": "the source uses or depends on the target",
                "part_of": "the source is a component of the target",
                "created_by": "the target created or authored the source",
                "instance_of": "the source is an example of the target category",
                "located_in": "the source is situated in the target",
                "works_on": "the source works on the target",
                "produces": "the source produces or outputs the target",
                "compares_to": "the source is compared or contrasted with the target",
                "mentions": "the source merely mentions the target",
            },
        },
    },
    # A 10-way choice has no threshold to calibrate — the answer is the
    # argmax. Measured 2026-09-20 on 200 rows, 8 per canvas: agreement with
    # the v4 classifier 0.580 against a majority-class floor of 0.310, in
    # 6.8 s for all 200 (34 ms a decision) against two sequential primary
    # calls each today. Weakest class `mentions` at 0.383; best `created_by`
    # at 0.739.
    threshold=None,
    label_mass_floor=0.60,   # min 0.6801, p01 0.9084, p05 0.9646 over 200 reads
    calibrated_on="200 rows of _pipeline/memory-graph/classified-v4-batch.jsonl, "
                  "8 per canvas, 2026-09-20: agreement 0.580 vs a 0.310 "
                  "majority-class floor",
    calibrated_hash="3047f1de5d791c0a",
    gate_blocked_reason=(
        "nearly double the majority-class floor and ~200x cheaper than the "
        "two primary calls it would replace, but 0.580 agreement with the v4 "
        "prompt is agreement, not correctness, and no labelled subset of "
        "these 35,802 rows has been checked by a human."),
    notes="Replay corpus only (classified-v4-batch.jsonl, 35,802 rows). The "
          "largest prize in the tree — 1,183-3,817 decisions/day at two "
          "sequential primary calls each — and it needs no live wiring to "
          "evaluate, because the labelled rows already exist.",
)

SCHEMAS: dict[str, Schema] = {
    s.name: s for s in (RERANK, DEDUPE, ENTITY, CLUSTERS, EDGES)
}

#: Seam -> schema, for the shadow recorder and `djev_status`. A seam with no
#: schema here records nothing, which is how a new seam fails: silently
#: uninstrumented rather than instrumented against a shape nobody froze.
BY_SEAM: dict[str, Schema] = {s.seam: s for s in SCHEMAS.values() if s.seam}


def get(name: str) -> Schema | None:
    return SCHEMAS.get(name)


def floor_for(seam: str) -> float | None:
    s = BY_SEAM.get(seam)
    return s.label_mass_floor if s else None


def hash_for(seam: str) -> str:
    s = BY_SEAM.get(seam)
    return s.hash if s else ""


def pair_state(a_title: str, a_body: str, b_title: str, b_body: str,
               *, chars: int = 900) -> str:
    """The canvas state for every pair schema here, so dedupe, entity and
    clusters cannot drift apart in how they present a pair."""
    return (f"Item A: {a_title}\n{str(a_body or '')[:chars]}\n\n"
            f"Item B: {b_title}\n{str(b_body or '')[:chars]}")
