#!/usr/bin/env python3
"""The retrieval leg of the #580 judge: BM25 over the labelled corpus only.

Not qmd. qmd indexes the vault segments (``agent_mcp/vault.py`` ``VAULT_SEGMENTS``),
and the vault copy of every ``bad`` sample here is its **repaired** text — the
answer. Searching the corpus through the live index would hand Judge B the
correction that defines its own label, which is the "second silent failure that
just nods along" the item warns about. The corpus is 50 documents, so a
deterministic stdlib BM25 over exactly those documents is both sufficient and
auditable, and it keeps the pass offline.

Two hard rules, both tested:

* a sample is never its own retrieved example;
* a sample's **same-file twin** is never one either — the pre-repair text of a
  note and the accepted text of the same note are near-identical documents, so
  without the path exclusion Judge B retrieves the answer in the shape of the
  question.
"""
from __future__ import annotations

import math
import re
from collections import Counter

try:  # imported as eval.durable_write_judge.retrieve (tests, `python -m`)
    from .build_corpus import sample_text
except ImportError:  # run as a script: python eval/durable_write_judge/retrieve.py
    from build_corpus import sample_text

STOPWORDS = frozenset(
    """a an and are as at be by for from has have he she his her in is it its of on or
    that the this to was were will with we our they their you not but if then than so
    such can could would should about into over under more most other some no nor only
    own same too very just now here when which who whom what where why how""".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower())
            if len(t) > 2 and t not in STOPWORDS]


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) over an in-memory document set."""

    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [d for d, _ in docs]
        self.tokens = [tokenize(t) for _, t in docs]
        self.freqs = [Counter(toks) for toks in self.tokens]
        self.lengths = [len(toks) for toks in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if docs else 0.0
        self.df: Counter = Counter()
        for toks in self.tokens:
            for term in set(toks):
                self.df[term] += 1
        self.n = len(docs)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[tuple[str, float]]:
        q_terms = set(tokenize(query))
        scores: list[tuple[str, float]] = []
        for doc_id, freq, length in zip(self.ids, self.freqs, self.lengths):
            total = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * length / (self.avg_len or 1.0))
            for term in q_terms:
                tf = freq.get(term, 0)
                if tf:
                    total += self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm)
            scores.append((doc_id, total))
        return sorted(scores, key=lambda kv: (-kv[1], kv[0]))


def build_index(samples: list[dict]) -> BM25:
    return BM25([(s["id"], sample_text(s)) for s in samples])


def excluded_ids(samples: list[dict], sample: dict) -> set[str]:
    """The sample itself plus every other sample drawn from the same vault file."""
    return {sample["id"]} | {
        s["id"] for s in samples if s["vault_path"] == sample["vault_path"]
    }


def top_k(index: BM25, samples: list[dict], sample: dict, k: int = 5,
          ) -> list[dict]:
    """The k nearest labelled writes, never this sample and never its own file."""
    by_id = {s["id"]: s for s in samples}
    banned = excluded_ids(samples, sample)
    ranked = [did for did, sc in index.score(sample_text(sample))
              if did not in banned and sc > 0.0]
    picked = [by_id[did] for did in ranked[:k]]
    if len(picked) < k:  # a near-zero-overlap document still earns its place
        chosen_ids = {s["id"] for s in picked}
        for s in samples:
            if len(picked) >= k:
                break
            if s["id"] not in banned and s["id"] not in chosen_ids:
                picked.append(s)
                chosen_ids.add(s["id"])
    return picked


def same_class_retrieved(sample: dict, retrieved: list[dict]) -> bool | None:
    """Selection quality: did any retrieved example carry one of this sample's
    defect classes? Judged on the label alone — never on the judge's verdict."""
    classes = sample.get("defect_classes") or []
    if not classes:
        return None  # a `good` sample has no class to match
    return any(c in classes for r in retrieved for c in (r.get("defect_classes") or []))
