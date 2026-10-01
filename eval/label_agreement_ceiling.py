#!/usr/bin/env python3
"""A retrieval-blind second labelling pass, and the ceiling it measures (#654).

WHY THIS EXISTS

`entity_hit_rate` is reported as if the gold labels were ground truth. They are
one author's judgement, hand-repaired by the loop under the 2026-08-06 #380 rule
(`eval/vault_recall_queries.yaml:23-33`), and 26 of the 94 gold entity labels match
more than one entity name in the live store (`Memory`→111, `MCP`→54, `GR00T`→22)
against a scorer whose rule is substring-or-canonical
(`eval/run_eval.py:251-269`). So a night can lose five points because a label is
ambiguous rather than because retrieval got worse, and the number cannot tell the
two apart. The fix is not to argue about which it was: it is to measure the
agreement of the labels with themselves and stop reporting a raw rate without it.

THE METHOD (Ishan Anand, "Persona Engineering", transferred to this eval)

> never report a model's agreement with ground truth without first measuring
> ground truth's agreement with itself.

The talk's fallback when the subjects cannot be re-called is to split the gold in
two, score one half as if it were the model output, and average over many splits.
Here there is a better pair of halves available: a second, independent labelling
pass. This script asks one labeler — a model, or an injected callable in tests — to
pick the gold entity and gold documents for every query in the corpus, from the
ENTITY-NAME TABLE AND VAULT PATHS ONLY, shuffled, with no retrieval result anywhere
in the prompt. The two halves are then:

  half A  the corpus's own gold, as authored;
  half B  what an independent labeller chose when shown the same namespaces.

`entity_label_agreement` / `doc_label_agreement` are A's agreement with B over the
corpus's own label counts. The number that turns a rate into evidence is the
**surrogate ceiling**: score B itself as if it were a retriever's answer, through
the scorer's own `_score`, and the result is the highest score this corpus can
report while agreeing with B. `score / ceiling` is then a real position on a real
scale instead of a point estimate nobody can interpret, and the Wilson interval
`eval/run_eval.py` already emits stays a statement about sampling, not about
whether the labels meant what the number says they meant.

WHY THE SURROGATE AND NOT A RESAMPLED SPLIT

The talk resamples because it has one labelled set and nothing else. Splitting
*this* corpus in half is mostly degenerate: 47 of 66 entity-labelled queries carry
exactly one gold entity and 25 of 74 doc-labelled queries carry exactly one gold
doc, so a random half is empty half the time and the resample averages a number of
zeros into a ceiling. Two independent labelling passes are the same measurement
without the degeneracy, so this script builds the second pass instead of resampling
the first, and the `--split-half` figure it also emits is reported WITH the query
count it could actually run on rather than as a headline.

THE PART THAT MAKES THE NUMBER HONEST

- **Retrieval-blind.** A labeler shown what retrieval returned agrees with retrieval,
  not with the gold, and the ceiling becomes a mirror. `build_prompt` takes the
  query and two candidate lists and nothing else; the candidate pools come from
  `app.kg_store`'s entity table and the vault's `.md` paths, never from a baseline
  record. Pinned by `tests/test_eval_label_agreement.py`.
- **Candidate coverage is disclosed, not assumed.** A label the labeler was never
  offered cannot be agreed with, so a low agreement caused by narrowing is not
  label noise. Every query records `entity_labels_offered`, and the ceiling excludes
  a query whose gold was never in its own candidate pool, naming the exclusions.
- **Which model produced the ceiling is in the artifact.** A ceiling from
  Qwen3.6-35B-A3B is a different claim from a ceiling from the primary, and the
  item's own risk section says so. The engine is a person's decision
  (`config.yaml secondary_enabled: false`), so there is no default engine: the
  `--engine`/`--labeler-url` choice is required, and a `--engine stub` artifact is
  stamped `kind: stub` and REFUSED by the loader unless `--allow-stub-ceiling` is
  passed explicitly. A nightly must never inherit a stub as a measured ceiling.
- **The gold can move under the ceiling.** Commit `9b028e9` re-pointed 22 gold
  entity names with the query ids untouched, which `scripts/eval_trend_stats.py`'s
  id-join cannot see. The artifact records `labels_sha256` over the labels
  themselves, so a reader can tell whether the ceiling still describes the corpus.

SCOPE: reads only. Writes only the artifact under
`~/lloyd-data/eval/baselines/label-agreement/`. Never calls a retriever.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The scorer owns label matching. Importing it rather than copying the rule is the
# point: a second copy of `gold label satisfied?` would drift from the one that
# produces the number this is a ceiling FOR, and the two would disagree silently.
# (eval/run_eval.py imports this module lazily, inside its loader, so there is no
# import cycle at module load.)
import eval.run_eval as ev  # noqa: E402

CORPUS_PATH = Path(__file__).resolve().parent / "vault_recall_queries.yaml"

#: The artifact directory, a SUBDIRECTORY of the baselines dir on purpose: the
#: nightly globs `nightly-*.json` in the parent, and an artifact that pattern can
#: see is an artifact some trend tool will score.
ARTIFACT_DIR_NAME = "label-agreement"
ARTIFACT_GLOB = "label-agreement-*.json"

#: The seed is recorded in the artifact and the shuffle is derived from
#: (seed, query id, leg), so a re-run reproduces the candidate ORDER exactly.
SEED = 20260924
ENTITY_CAP = 40
DOC_CAP = 40

#: The two reasons a gold label can fail to appear among a query's candidates (#1823),
#: which need opposite owners. `outside_cap` is THIS instrument's own `ENTITY_CAP`/
#: `DOC_CAP` narrowing a namespace that DOES hold a name satisfying the gold — widening
#: a cap converts those labels back into measurements and says nothing about the store.
#: `absent_from_namespace` is no name in the namespace satisfying the gold at all, so
#: only a store-side change (a rename, a merge) or a re-label can ever make it
#: offerable. Measured against the live store on 2026-09-29, 52 of the 53 unofferable
#: entity gold NAMES were cap artefacts and exactly one (`Nightly Reflection`) had no
#: namespace name — the instrument printed the second story for all of them while
#: computing neither test.
#:
#: #1937 corrected the half of that which over-promised. "Widening a cap converts those
#: labels back" is true only in the limit cap -> namespace: replayed over the live
#: pools on 2026-10-01, entity gold offered went 0.468 at cap 40 -> 0.649 at cap 320 ->
#: 1.000 only uncapped, because 42 of 94 entity golds share no token with their query
#: and the ranking orders that zero-overlap band by spelling. `outside_cap` keeps its
#: meaning (the namespace holds a name); what it no longer implies is that a bounded
#: cap reaches it. The artifact's `offered` block records the ladder, so the share the
#: BUILDER hides is printed rather than inferred.
OUTSIDE_CAP = "outside_cap"
ABSENT_FROM_NAMESPACE = "absent_from_namespace"

#: Identity of the candidate-menu builder (#1937), recorded in every artifact beside
#: `caps`. A ceiling is a function of its menu: one produced by a different builder
#: (alias-expanded, embedding-ranked — `eval/gold_blind_menu_probe.py`) is a different
#: instrument, and must be distinguishable from label-agreement-20260929-005508.json,
#: which predates the key and was built by this one.
MENU_BUILDER = "query-token-overlap/v1"

#: The cap multiples the `offered` block reports beside the run's own cap.
OFFERED_LADDER = (1, 2, 4, 8)

#: The agreement below which a ceiling's own disagreement set becomes the headline
#: (#654's veto), and the figure `--print` evaluates it on — the all-gold rate.
AGREEMENT_VETO = 0.80

#: The kind string every normalized number names. Distinct from the seed-side
#: ceiling (`anchorless_query_count`) which bounds a different failure: a query
#: the seeds cannot anchor cannot be hit by any retriever, while a query the two
#: labelers disagree on can be hit and still be noise.
CEILING_KIND = "gold_label_surrogate"

#: The metric names, split by which leg's labels produce the ceiling.
ENTITY_METRICS = ("entity_hit_rate", "entity_recall_avg")
DOC_METRICS = ("doc_hit_rate", "doc_recall_avg", "mrr_doc", "ndcg10")
#: No gold-side ceiling exists for the fact leg: the labeler chooses entities and
#: document paths, never fact rows, so a surrogate here would carry no facts and its
#: fact_entity_recall would be 0 by construction, not by measurement.
UNMEASURED_METRICS = {"fact_entity_recall_avg":
                      "the second labeler labels entities and document paths, not "
                      "fact rows, so no gold-side surrogate exists for the fact leg"}

_STOPWORDS = {
    "the", "and", "about", "what", "which", "when", "where", "why", "how", "does",
    "for", "with", "from", "that", "this", "were", "was", "are", "you", "your",
    "tell", "me", "did", "can", "get", "into", "have", "has", "not", "its", "it",
    "of", "to", "in", "on", "is", "or", "an", "as", "at", "be", "by", "do",
}


# ── the corpus ───────────────────────────────────────────────────────────────

def load_corpus(path: Path | str = CORPUS_PATH) -> list[dict]:
    import yaml
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return list(spec.get("queries") or [])


def label_counts(queries: list[dict]) -> dict:
    """The corpus's OWN label counts, read from the file — never a constant.

    The item is explicit that the denominator is today's 94 entity labels across
    66 queries and 139 doc labels, not the 43 an older draft of itself quoted. An
    agreement rate whose denominator is a remembered number is a rate about
    nothing: the corpus grows (#1319 took it 20→81) and a hard-coded denominator
    silently turns every future agreement figure into a lie.
    """
    ent = [q for q in queries if q.get("expect_entities")]
    docs = [q for q in queries if q.get("expect_docs")]
    return {
        "queries": len(queries),
        "entity_queries": len(ent),
        "entity_labels": sum(len(q.get("expect_entities") or []) for q in ent),
        "doc_queries": len(docs),
        "doc_labels": sum(len(q.get("expect_docs") or []) for q in docs),
    }


def labels_sha256(queries: list[dict]) -> str:
    """Hash of the LABELS, not of the file. A comment edit moves the file hash and
    must not invalidate a ceiling; re-pointing a gold name must."""
    payload = [[q.get("id"), sorted(q.get("expect_entities") or []),
                sorted(q.get("expect_docs") or [])] for q in queries]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def corpus_sha256(path: Path | str = CORPUS_PATH) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


# ── candidate pools: entity names and vault paths, nothing else ──────────────

def query_tokens(query: str) -> list[str]:
    """Content tokens of the query, in first-seen order.

    Deterministic by construction: a fixed stopword set and a length floor, no
    model, no corpus statistics. The narrowing only has to be reproducible and
    inspectable — it is the labeler's shortlist, not its answer.
    """
    out: list[str] = []
    for tok in re.split(r"[^0-9a-z]+", str(query or "").lower()):
        if len(tok) >= 3 and tok not in _STOPWORDS and tok not in out:
            out.append(tok)
    return out


def _rng(seed: int, query_id: str, leg: str) -> random.Random:
    return random.Random(f"{int(seed)}:{leg}:{query_id}")


def _narrow(query: str, pool: list[str], *, cap: int, seed: int, query_id: str,
            leg: str) -> list[str]:
    """Top-`cap` of `pool` by query-token overlap, then shuffled under `seed`.

    Three properties, all load-bearing:
      * determinism — the total order is (-overlap, length, name), so a tie is
        broken before the shuffle and the shuffle is seeded per (query, leg);
      * it is a NARROWING of a fixed pool, never a generation. A candidate the
        store does not hold cannot appear, so the labeler cannot be scored
        against an answer that is not in the namespace;
      * GOLD-BLIND, as well as retrieval-blind. Neither a retrieval pass's output
        nor the primary label's answer reaches here: a labeler shown retrieved
        candidates agrees with retrieval rather than with the gold, and a labeler
        shown the answer agrees with the answer. The menu is a function of the query
        and the namespace and nothing else, which is what makes the agreement figure
        a second opinion instead of an echo. The cost of gold-blindness is recorded,
        not hidden — a gold whose own form falls outside the cap cannot be chosen, so
        that query is EXCLUDED from the ceiling and named in `ceiling.excluded`
        (leaving it in would depress the ceiling through the instrument's own cap and
        so inflate `score / ceiling`, the one error direction that could be mistaken
        for excusing a real defect). Raising the cap is the knob that converts
        exclusions back into measurements — but only SOME of them, which is the half
        this sentence used to omit: `ceiling.labels_unofferable` carries the per-leg
        count of excluded labels and `ceiling.labels_unofferable_detail` splits it, so
        `outside_cap` is the share a wider cap would recover and
        `absent_from_namespace` is the share no cap can (#1823). Both were promised
        here in words and neither was ever written.
    A query with no overlap anywhere still gets `cap` candidates, filled in name
    order, so a zero-overlap query is measured on a pool like every other rather
    than on an empty one that would report agreement 0 for a non-reason.
    """
    chosen = _ranked(query, pool)[:cap]
    _rng(seed, query_id, leg).shuffle(chosen)
    return chosen


def _ranked(query: str, pool: list[str]) -> list[str]:
    """The whole of `pool` in the builder's total order, before any cap or shuffle."""
    toks = query_tokens(query)
    scored: list[tuple[int, int, str]] = []
    for name in pool:
        hay = ev._norm(name)
        score = sum(1 for t in toks if t in hay)
        scored.append((-score, len(name), name))
    scored.sort()
    return [n for _s, _l, n in scored]


def offered_block(queries: list[dict], entity_names: list[str], vault_paths: list[str],
                  *, entity_cap: int = ENTITY_CAP, doc_cap: int = DOC_CAP) -> dict:
    """Per leg: gold labels offered at the run's cap, at its multiples, and uncapped.

    The same ranking `_narrow` cuts, measured at every depth (#1937). `at_cap` is
    what this run offered; `uncapped` is the offered fraction of the same ranking
    over the whole namespace, i.e. every label some namespace name satisfies; the
    rungs between say how much of the gap a wider cap buys. When `at_cap` and the
    8x rung are close and `uncapped` is far, the gold is hidden by the ORDER, and no
    bounded cap is the fix. No labeler, no gold reaches a menu: this only measures
    where the gold fell in an order built without it.
    """
    out: dict = {}
    for leg, pool, cap, key, satisfied in (
            ("entity", entity_names, int(entity_cap), "expect_entities",
             entity_label_satisfied),
            ("doc", vault_paths, int(doc_cap), "expect_docs", doc_label_satisfied)):
        ranks: list[int | None] = []
        for q in queries:
            golds = list(q.get(key) or [])
            if not golds:
                continue
            ordered = _ranked(str(q.get("query") or ""), pool) if pool else []
            for g in golds:
                ranks.append(next((i for i, c in enumerate(ordered)
                                   if satisfied(g, [c])), None))
        total = len(ranks)
        rungs = {str(cap * m): sum(1 for r in ranks if r is not None and r < cap * m)
                 for m in OFFERED_LADDER}
        uncapped = sum(1 for r in ranks if r is not None)
        out[leg] = {
            "labels": total, "cap": cap,
            "offered_by_cap": rungs, "offered_uncapped": uncapped,
            "unoffered_uncapped": total - uncapped,
            "at_cap": round(rungs[str(cap)] / total, 4) if total else None,
            "uncapped": round(uncapped / total, 4) if total else None,
        }
    return out


def entity_candidates(query: str, entity_names: list[str], *, cap: int = ENTITY_CAP,
                      seed: int = SEED, query_id: str = "") -> list[str]:
    return _narrow(query, entity_names, cap=cap, seed=seed, query_id=query_id,
                   leg="entity")


def doc_candidates(query: str, vault_paths: list[str], *, cap: int = DOC_CAP,
                   seed: int = SEED, query_id: str = "") -> list[str]:
    return _narrow(query, vault_paths, cap=cap, seed=seed, query_id=query_id,
                   leg="doc")


def entity_name_table() -> list[str]:
    """Every entity name in the live store. A store that will not open raises —
    it must never read as an empty namespace, which would silently produce an
    empty pool and an agreement of 0 that looks like a finding."""
    from app.kg_store import store
    return store().entities.all()


def vault_markdown_paths(root: Path | str | None = None) -> list[str]:
    """Vault-relative `.md` paths, sorted. This is the doc leg's whole namespace:
    the same namespace the gold `expect_docs` substrings live in, and the reason
    no retrieval output is needed to offer a plausible candidate list."""
    base = Path(root) if root else (Path.home() / "obsidian")
    out: list[str] = []
    if not base.exists():
        raise SystemExit(f"VAULT_ROOT_MISSING: {base} — the doc leg has no namespace "
                         f"to label from; refusing to write an agreement of 0")
    for p in base.rglob("*.md"):
        if ".git" in p.parts:
            continue
        try:
            out.append(str(p.relative_to(base)))
        except ValueError:
            continue
    return sorted(out)


# ── the payload and the reply ────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are labelling a retrieval benchmark. You are shown one query and two "
    "shuffled candidate lists: entity names from a knowledge graph, and markdown "
    "file paths from a vault. You have no idea what any search engine returned, and "
    "the order of the candidates means nothing. Pick the candidates that genuinely "
    "answer the query: the entity names that are the right subject of it, and the "
    "file or files that hold the answer. Pick nothing that does not answer it — an "
    "empty list is a valid answer. Reply with JSON only, of the form "
    '{"entities": [<candidate numbers>], "docs": [<candidate numbers>]}'
)


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def build_prompt(query: str, entity_candidates: list[str],
                 doc_candidates: list[str]) -> str:
    """The whole labeler payload. Its arguments ARE its provenance: a query and two
    candidate lists, so there is no parameter through which a retrieval result, a
    baseline record or a scoring outcome could reach the labeler. Pinned by test."""
    return (
        f"QUERY: {query}\n\n"
        f"ENTITY CANDIDATES (numbers 1-{len(entity_candidates)}; order means nothing):\n"
        f"{_numbered(entity_candidates)}\n\n"
        f"DOC CANDIDATES (numbers 1-{len(doc_candidates)}; order means nothing):\n"
        f"{_numbered(doc_candidates)}\n\n"
        "Reply with the JSON object only."
    )


_INDEX_BLOCK = re.compile(r'"(entities|docs)"\s*:\s*\[([^\]]*)\]')


def parse_reply(text: str, *, n_entities: int, n_docs: int) -> dict[str, list[int]]:
    """Candidate NUMBERS out of the reply, clamped to what was offered.

    Index rather than name, so a 35B model cannot hallucinate an entity id that the
    store does not hold and have it count as a second opinion; an out-of-range or
    unparsable selection is EMPTY, and `parse_failed` is stamped beside it so a
    reader sees a failure to read rather than a disagreement.
    """
    # An empty answer is NOT a parse failure. "Nothing here answers this query" is a
    # legitimate and informative label, and stamping it `parse_failed` would let a
    # labeler that is confidently abstaining read as a labeler that could not read —
    # and the second one is a broken instrument while the first is a finding. The
    # distinction is whether the reply has the shape at all.
    body = str(text or "")
    picks: dict[str, list[int]] = {"entities": [], "docs": []}
    for key, inner in _INDEX_BLOCK.findall(body):
        for num in re.findall(r"\d+", inner):
            idx = int(num)
            if 1 <= idx <= (n_entities if key == "entities" else n_docs):
                if idx not in picks[key]:
                    picks[key].append(idx)
    shaped = "entities" in body and "docs" in body
    return {"entities": [i - 1 for i in picks["entities"]],
            "docs": [i - 1 for i in picks["docs"]],
            "parse_failed": (not shaped) or not body.strip()}


# ── labelers ────────────────────────────────────────────────────────────────

def http_post_json(url: str, payload: dict, *, timeout: float = 90.0) -> dict:
    """POST `payload` as JSON and return the decoded response object.

    The only place in this module that touches a socket, so it is a function of its
    own: `http_labeler` accepts a `transport` with this signature, and a test can
    then exercise the real request builder and response decoder against a canned
    response instead of stubbing the whole labeler. The review rung's finding on the
    first round of this item was "seam unverified: http_labeler -> POST "
    "<base>/v1/chat/completions (urllib request/response shape, temperature 0, index
    clamping); every test injects a callable instead" — which is true of the labeler
    as a unit, because a stub labeler replaces the encoding, the URL, the decode and
    the clamp check in one stroke.
    """
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


#: Sampling-parameter names a diffusion-model server rejects outright. The djev
#: endpoint on this box answers ANY `temperature` (it is a diffusion build, not an
#: autoregressive one) with HTTP 400 and this sentence, copied from the live reply
#: on 2026-09-28: "The temperature, min_p, seed, min_tokens, logit_bias, bad_words,
#: and allowed_token_ids sampling parameters are not yet supported with diffusion
#: models." Those seven are the names listed here, and nothing else is dropped:
#: the retry only ever removes a key that is BOTH in this list and in the payload it
#: just sent, so an endpoint that accepts sampling parameters is never silently
#: sent a different request than the one `http_labeler` builds.
_DIFFUSION_REJECTED_SAMPLING = ("temperature", "min_p", "seed", "min_tokens",
                                "logit_bias", "bad_words", "allowed_token_ids")


def http_get_json(url: str, *, timeout: float = 10.0) -> dict:
    """GET `url` and decode a JSON object. The read-side twin of `http_post_json`,
    and its own function for the same reason: it is one of the two places in this
    module that touches a socket, so `resolve_engine` takes a fetcher of this
    signature and a test can exercise the real URL building and decoding against a
    canned `/v1/models` document instead of stubbing the resolver."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _rejected_sampling_params(exc: Any, payload: dict) -> list[str]:
    """Which sampling parameters THIS 400 says THIS payload may not carry.

    Three conditions, all required, because the failure this guards against (an
    engine that will not take `temperature`) is one the labeler can only survive by
    retrying, and retrying a 400 for any other reason would turn a real error into
    a second, weirder error:

      * the status is 400, not 401/404/500 — a refusal to sample is a bad request;
      * the reply body names one of `_DIFFUSION_REJECTED_SAMPLING` in words;
      * that name is a key the payload actually sent.

    A 400 for an unknown model or a malformed body names neither, returns the empty
    list here, and the original `HTTPError` propagates untouched — which is the
    other half of what makes the retry safe: it cannot mask a genuine request bug.
    """
    if getattr(exc, "code", None) != 400:
        return []
    try:
        body = str(exc.read().decode("utf-8", "replace")).lower()
    except Exception:
        # `HTTPError` is itself the response object, so reading usually works; if a
        # subclass has already drained the body there is nothing to match on, and
        # the safe answer is "not the sampling refusal" — propagate.
        return []
    return [name for name in _DIFFUSION_REJECTED_SAMPLING
            if name in payload and name in body]


def http_labeler(url: str, model: str, *, timeout: float = 90.0,
                 temperature: float = 0.0,
                 transport: Callable[[str, dict], dict] | None = None
                 ) -> Callable[[dict], dict]:
    """A chat-completions endpoint as a labeler. temperature 0: the ceiling is one
    judgement per query, and sampling the same query a thousand times measures the
    sampler, not the labels — the talk's weather-gauge point, taken literally.

    `transport` swaps the POST (see `http_post_json`) and nothing else: payload
    assembly, prompt embedding and response decoding still run, which is what makes
    a test through it a test of this module's HTTP behaviour.

    One retry, and only one, when the endpoint answers 400 naming a sampling
    parameter it will not accept (`_rejected_sampling_params`): the parameter is
    dropped and the same request sent again. Without it the ruling on #1655 — label
    with djev, which needs no pool pause — cannot run at all, because `urllib` raises
    `HTTPError` on a 400 and every one of the corpus's queries dies on the first POST
    rather than degrading: `LABELER_EMPTY_RESPONSE` never fires, because that check is
    downstream of the exception. The dropped names go to stderr rather than into the
    artifact, because a label produced without the temperature the ceiling's own
    docstring promises is a label whose provenance a reader has to be able to see —
    and an artifact that silently recorded it as `temperature 0` would be a number
    describing a request that was never sent. The retry is NOT applied to a 400 that
    does not name a parameter this payload sent.
    """
    post = transport or http_post_json

    def label(request: dict) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": request["prompt"]}],
            "temperature": temperature, "max_tokens": 300,
        }
        try:
            data = post(url, payload)
        except urllib.error.HTTPError as exc:
            rejected = _rejected_sampling_params(exc, payload)
            if not rejected:
                raise
            for name in rejected:
                payload.pop(name, None)
            print(f"  [labeler-sampling-dropped] {url} rejected "
                  f"{', '.join(rejected)} for {model}; retried once without "
                  f"them — this query's label was NOT sampled at temperature 0",
                  file=sys.stderr)
            data = post(url, payload)
        text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                or "")
        if not text.strip():
            # An engine that answers HTTP 200 with an empty completion is not
            # disagreeing with the gold; the instrument is dead. Left as "", every
            # such query becomes `parse_failed`, the artifact still writes, and the
            # ceiling reads as a measured number over 81 empty answers.
            raise RuntimeError(
                f"LABELER_EMPTY_RESPONSE: {model} @ {url} returned no content; "
                f"an empty reply is an instrument failure, not a label")
        return {"text": text}
    return label


#: The engine names `--engine` accepts, and where each one's endpoint lives.
#: `primary`/`secondary` are `models:` slots; djev's OpenAI-compatible server is
#: the top-level `djev.base_url` block (`djev.structured_url` is the
#: /v1/systemone server, which is not a chat-completions endpoint).
ENGINE_NAMES = ("primary", "secondary", "djev")


def served_model_name(base: str, *, fetch: Callable[[str], dict] | None = None
                      ) -> str:
    """Ask the endpoint at `base` what model it serves, via `{base}/v1/models`.

    Why the code asks rather than refusing: the ruling on #1655 is to label with
    djev, `config.yaml` carries `djev.base_url` but no `djev.model` (adding one is a
    human edit to a file this loop may not write), and the served name is sitting in
    the endpoint's own `/v1/models` reply — the refusal this replaces told a person to
    go and read exactly that document and paste it into a flag. Asking costs one GET
    and removes no decision, because the name comes FROM the thing that will be asked
    to label: it is a measurement of which model is up, not a choice of which model
    should produce the ceiling.

    It therefore refuses, and only refuses, in the two states where the endpoint
    cannot answer that question for itself: unreachable (or answering something that
    is not a model list), and serving MORE THAN ONE model — at which point picking one
    IS a choice about what the number means, and `resolve_engine`'s standing rule is
    that such a choice goes to a person. An empty list is the unreachable case in
    another costume: it names no model, so there is nothing to learn.
    """
    get = fetch or http_get_json
    url = f"{base.rstrip('/')}/v1/models"
    try:
        doc = get(url)
        ids = [str(entry.get("id") or "")
               for entry in (doc.get("data") or []) if isinstance(entry, dict)]
    except Exception as exc:
        raise SystemExit(
            f"LABELER_MODEL_UNKNOWN: config.yaml records no served model name under "
            f"`djev:`, and asking the endpoint failed ({type(exc).__name__}: {exc} "
            f"for GET {url}). Start it, or name the model yourself: "
            f"`--labeler-url {base.rstrip('/')}/v1/chat/completions "
            f"--labeler-model <served name>`. Naming the model that produced a "
            f"ceiling is part of what the ceiling means, so it is not a value to "
            f"invent here.")
    ids = [i for i in ids if i]
    if len(ids) > 1:
        raise SystemExit(
            f"LABELER_MODEL_AMBIGUOUS: {url} serves {len(ids)} models "
            f"({', '.join(ids)}), and config.yaml records no `djev.model`. Which one "
            f"produces the ceiling changes what the number means, so it is a person's "
            f"call: pass `--labeler-model <one of them>` or set `djev.model`.")
    if not ids:
        raise SystemExit(
            f"LABELER_MODEL_UNKNOWN: {url} answered but listed no model, so there is "
            f"no served name to learn. Pass `--labeler-url "
            f"{base.rstrip('/')}/v1/chat/completions --labeler-model <served name>`.")
    return ids[0]


def resolve_engine(name: str, *,
                   models_fetcher: Callable[[str], dict] | None = None
                   ) -> tuple[str, str]:
    """(chat_completions_url, served model name) for a named engine.

    No default engine exists: `secondary_enabled: false` and a stopped
    `agent-llm-secondary` make the item's named :8091/Qwen3.6-35B-A3B a human
    decision, and which model produced a ceiling is part of what the ceiling means.
    Naming an engine that is switched off is therefore an error to report, not a
    thing to silently fall back on — a fallback would swap the meaning of the
    number while keeping the number.

    The switch is read from `app.config.CONFIG`, the mapping `config.yaml` loads
    into (`app/config.py:288`). There is no lowercase `config` in that module, and
    an earlier draft of this function asked for one inside a `try:`, caught the
    ImportError it raised, and concluded `secondary_enabled` was false — so a
    person who flipped the switch would still have been told the engine was off.
    A refusal that is produced by the guard's own broken lookup is a false outage,
    the failure this repo's memory file catalogs under "a guard that reads its own
    missing input reports a verdict it cannot justify". Pinned by
    `tests/test_eval_label_agreement.py::test_resolve_engine_reads_the_real_config`,
    which flips the switch and asserts the refusal goes away.
    """
    from app.config import CONFIG, _get_model_cfg, _resolve_model_name
    root = dict(CONFIG or {})
    if name not in ENGINE_NAMES:
        raise SystemExit(f"LABELER_ENGINE_UNKNOWN: {name!r}; "
                         f"use one of {', '.join(ENGINE_NAMES)}")
    if name == "djev":
        block = root.get("djev") or {}
        base = str(block.get("base_url") or "")
        if not base:
            raise SystemExit("LABELER_ENGINE_NO_URL: config.yaml has no "
                             "`djev.base_url`, so there is no djev "
                             "chat-completions endpoint to label with")
        # A configured `djev.model` wins and costs no probe: it is what a person
        # chose. Without one the endpoint is asked what it serves
        # (`served_model_name`), which is the only reading of "the djev engine" that
        # cannot go stale the way a copied name in a config file goes stale — and the
        # alternative, refusing, is what has kept this item's ruling unrunnable since
        # it was written. Guessing a name is still not on the table: vLLM refuses a
        # model it is not serving, and finding that out on query 1 of 86, after a
        # person decided to start the run, is the failure this branch exists to
        # prevent.
        served = str(block.get("model") or "") or served_model_name(
            base, fetch=models_fetcher)
        return f"{base.rstrip('/')}/v1/chat/completions", served
    if name == "secondary" and not root.get("secondary_enabled", False):
        raise SystemExit(
            "LABELER_ENGINE_DISABLED: secondary_enabled is false in config.yaml "
            "(agent-llm-secondary is stopped, no listener on :8091). Which engine "
            "produces the ceiling is a person's call — enable it, or pass "
            "--engine primary or --labeler-url/--labeler-model.")
    cfg = _get_model_cfg(name) or {}
    # The same two-key lookup `app/secondary_models.py:_endpoint` uses: a slot may
    # carry its endpoint as `base_url` or only as the `ANTHROPIC_BASE_URL` env
    # override, and reading one key alone makes a real engine look unconfigured.
    base = str(cfg.get("base_url")
               or (cfg.get("env") or {}).get("ANTHROPIC_BASE_URL") or "")
    if not base:
        raise SystemExit(f"LABELER_ENGINE_NO_URL: config has no base_url for {name!r}")
    return f"{base.rstrip('/')}/v1/chat/completions", _resolve_model_name(name)


def stub_labeler(pick: Callable[[dict], list[int]] | None = None
                 ) -> Callable[[dict], dict]:
    """A deterministic no-engine labeler, for tests and for a plumbing smoke.

    Artifacts written by it are stamped `kind: stub` and `load_artifact` refuses
    them unless told otherwise, because a stub's "agreement" is a property of the
    stub and must never be read as a measured ceiling by a nightly.
    """
    def label(request: dict) -> dict:
        ents = request["entity_candidates"]
        docs = request["doc_candidates"]
        if pick is None:
            ent_idx = [0] if ents else []
            doc_idx = [0] if docs else []
        else:
            ent_idx = pick(request)
            doc_idx = []
        return {"text": json.dumps({"entities": [i + 1 for i in ent_idx],
                                    "docs": [i + 1 for i in doc_idx]})}
    return label


# ── the labelling run ────────────────────────────────────────────────────────

def _avg(xs: list[float]) -> float | None:
    """The same averaging rule `summarize` uses: skip None, and an empty
    denominator is None, not 0. A 0 would read as a measured zero."""
    known = [x for x in xs if x is not None]
    return (sum(known) / len(known)) if known else None


def surrogate_scoring(query_spec: dict, chosen_entities: list[str],
                      chosen_docs: list[str]) -> dict:
    """Score the SECOND labeler's answer with the corpus's own scorer.

    This is the split-half move done properly: half B is treated as the model
    output and scored against half A by `_score`, so the ceiling is in the same
    units as the metric it bounds rather than in some other similarity's units.

    `seeds=[]` is deliberate and is the difference between a ceiling and a leak.
    With `seeds=None` the scorer re-extracts seeds from the query text, so a gold
    entity that merely appears verbatim in the query would count as a hit the
    labeler never chose, and the ceiling would rise by an effect no labeler
    produced. Passing `[]` is the real call shape (`_entities_in_result`) and keeps
    the surrogate answer exactly what the labeler picked.
    """
    # The shape is a retrieval result, because `_score` reads a retrieval result:
    # `_entities_in_result` walks `graph_neighbors_used` with `.get("entity")`, so
    # bare strings are an AttributeError and an empty list would score a ceiling of
    # 0 by construction rather than by measurement.
    result = {
        "facts": [],
        "documents": [{"path": p} for p in chosen_docs],
        "graph_neighbors_used": [{"entity": e} for e in chosen_entities],
        "graph_expanded_facts": [],
    }
    return ev._score(query_spec, result, seeds=[])


def label_corpus(*, labeler: Callable[[dict], dict], queries: list[dict],
                 entity_names: list[str], vault_paths: list[str],
                 labeler_identity: dict, seed: int = SEED,
                 entity_cap: int = ENTITY_CAP, doc_cap: int = DOC_CAP,
                 corpus_path: Path | str = CORPUS_PATH,
                 sleep_seconds: float = 0.0) -> dict:
    """Label every query in the corpus and return the artifact (writes nothing).

    The artifact is self-contained on purpose: it carries the primary gold it was
    compared against, so `agreement()` recomputes both numbers from the stored file
    alone and a later reader can prove the figure was not edited.
    """
    rows: list[dict] = []
    for q in queries:
        qid = str(q.get("id") or "")
        text = str(q.get("query") or "")
        gold_ents = list(q.get("expect_entities") or [])
        gold_docs = list(q.get("expect_docs") or [])
        # Retrieval-BLIND and gold-BLIND narrowing: the menu is a function of the
        # query and the namespace only. The labeler is never shown the primary's
        # answer, and never shown what retrieval returned, so agreement between the
        # two labelers is a second opinion and not an echo. The consequence is stated
        # where it is read: when a gold label's own form is absent from the menu the
        # query is EXCLUDED from the ceiling and named (`ceiling.excluded`), because
        # leaving it in would depress the ceiling through the instrument's own cap and
        # so inflate `score / ceiling` — the one direction of this number that can be
        # mistaken for excusing a real retrieval defect.
        ents = entity_candidates(text, entity_names, cap=entity_cap, seed=seed,
                                 query_id=qid)
        docs = doc_candidates(text, vault_paths, cap=doc_cap, seed=seed, query_id=qid)
        request = {"id": qid, "query": text, "entity_candidates": ents,
                   "doc_candidates": docs,
                   "prompt": build_prompt(text, ents, docs)}
        reply = labeler(request) or {}
        picks = parse_reply(reply.get("text", ""), n_entities=len(ents),
                            n_docs=len(docs))
        chosen_ents = [ents[i] for i in picks["entities"]]
        chosen_docs = [docs[i] for i in picks["docs"]]
        # Coverage: was the gold even OFFERED? Without this a narrowing artifact is
        # read as label noise, which is the direction that lies in the dangerous way.
        ent_offered = sum(1 for g in gold_ents
                          if any(entity_label_satisfied(g, [c]) for c in ents))
        doc_offered = sum(1 for g in gold_docs
                          if any(doc_label_satisfied(g, [c]) for c in docs))
        # Why the menu missed each of these golds, recorded HERE rather than inferred at
        # print time (#1823): the namespace this run narrowed is the only namespace that
        # can say whether a dropped name was in it, and by print time the store may have
        # moved. `kind` stays None when there was no namespace to test against, and
        # ceiling() counts those as `unclassified` rather than as either reason.
        un_ent = [g for g in gold_ents
                  if not any(entity_label_satisfied(g, [c]) for c in ents)]
        un_docs = [g for g in gold_docs
                   if not any(doc_label_satisfied(g, [c]) for c in docs)]
        ent_kinds = (classify_unofferable(un_ent, leg="entity",
                                          namespace=entity_names)
                     if un_ent and entity_names else {})
        doc_kinds = (classify_unofferable(un_docs, leg="doc", namespace=vault_paths)
                     if un_docs and vault_paths else {})
        rows.append({
            "id": qid,
            "query": text,
            "category": q.get("category"),
            "primary_entities": gold_ents,
            "primary_docs": gold_docs,
            "entity_candidates": ents,
            "doc_candidates": docs,
            "second_entities": chosen_ents,
            "second_docs": chosen_docs,
            "entity_labels_offered": ent_offered,
            "doc_labels_offered": doc_offered,
            "entity_labels_unofferable": [{"gold": g, "kind": ent_kinds.get(g)}
                                          for g in un_ent],
            "doc_labels_unofferable": [{"gold": g, "kind": doc_kinds.get(g)}
                                       for g in un_docs],
            "parse_failed": bool(picks["parse_failed"]),
            "surrogate": surrogate_scoring({"query": text,
                                            "expect_entities": gold_ents,
                                            "expect_docs": gold_docs},
                                           chosen_ents, chosen_docs),
        })
        if sleep_seconds:
            time.sleep(sleep_seconds)

    art = {
        "schema": 1,
        "kind": CEILING_KIND,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "labeler": dict(labeler_identity),
        "seed": int(seed),
        "caps": {"entity": int(entity_cap), "doc": int(doc_cap)},
        # Which builder made the menus, and what its order hides at this cap and
        # beyond it (#1937) — beside the caps, because the cap alone was read as the
        # whole cost for a month.
        "menu_builder": MENU_BUILDER,
        "offered": offered_block(queries, entity_names, vault_paths,
                                 entity_cap=entity_cap, doc_cap=doc_cap),
        # How the candidate lists were built, recorded beside the number so a reader
        # never opens the source to learn what the ceiling licenses. Both menus are a
        # function of the QUERY AND THE NAMESPACE ONLY — never the primary's gold,
        # never a retrieval result — which is what makes the second labeler a second
        # opinion. The known cost is stated rather than hidden: a gold whose own form
        # falls outside the cap cannot be chosen, so such a query is excluded from the
        # ceiling and named in `ceiling.excluded`, and the cap itself is the reason
        # agreement is a lower bound on label reproducibility.
        "narrowing": {
            "entity": "kg_store entity names ranked by query-token overlap, capped, "
                      "shuffled under the recorded seed; no gold, no retrieval output",
            "doc": "vault .md paths ranked by query-token overlap on path "
                   "components, capped, shuffled under the recorded seed",
            "gold_aware": False,
            "unofferable_gold": "excluded from the ceiling and named",
        },
        "corpus": {"path": str(corpus_path), "sha256": corpus_sha256(corpus_path),
                   "labels_sha256": labels_sha256(queries),
                   "counts": label_counts(queries)},
        "queries": rows,
    }
    art["agreement"] = agreement(art)
    art["ceiling"] = ceiling(art)
    return art


# ── agreement ────────────────────────────────────────────────────────────────

def entity_label_satisfied(label: str, choices: list[str]) -> bool:
    """Does one of `choices` satisfy the corpus's gold-label rule for `label`?

    This is the scorer's OWN substring-or-canonical rule, imported from
    `eval/run_eval.py` rather than re-typed here. The direction of the difference
    matters and is stated in run_eval: the scorer additionally equates two
    spellings after canonicalisation, which this rule does not attempt, so every
    agreement figure here is a LOWER bound on scorer-equivalent agreement — i.e. a
    conservative one. A ceiling that under-claims is the safe direction; a ceiling
    that over-claims is an excuse.
    """
    return ev.entity_label_satisfied(label, choices)


def doc_label_satisfied(label: str, choices: list[str]) -> bool:
    return ev.doc_label_satisfied(label, choices)


def classify_unofferable(labels: list[str], *, leg: str,
                         namespace: list[str]) -> dict[str, str]:
    """Why each gold label in `labels` could not be offered: the cap, or the namespace.

    `labels` are gold labels already known NOT to be among the query's candidates; this
    asks the one question that separates the two owners of that fact — does the
    namespace hold ANY name that would satisfy this gold (`OUTSIDE_CAP`, so the
    instrument's own `ENTITY_CAP`/`DOC_CAP` is the reason and widening it is the fix),
    or does no name in it satisfy the gold (`ABSENT_FROM_NAMESPACE`, so the store cannot
    offer it at any cap and only a rename, a merge or a re-label touches it)?

    The test is the SAME predicate coverage uses — `entity_label_satisfied` /
    `doc_label_satisfied`, the scorer's own rule imported from `eval/run_eval.py` — so
    a label is never counted as present in the namespace by a looser rule than the one
    that would have counted it as offered.

    Raises on an empty namespace: with no namespace the question has no answer, and a
    caller that defaults to an empty list would silently turn every cap-dropped label
    into `absent_from_namespace` — the one misclassification that would print a
    store-side fix set the store does not need.
    """
    if leg not in ("entity", "doc"):
        raise ValueError(f"unknown leg {leg!r}; expected 'entity' or 'doc'")
    if not namespace:
        raise ValueError(f"empty {leg} namespace: an unofferable label cannot be "
                         "classified without the namespace to test it against")
    satisfied = entity_label_satisfied if leg == "entity" else doc_label_satisfied
    return {str(gold): (OUTSIDE_CAP
                        if any(satisfied(gold, [name]) for name in namespace)
                        else ABSENT_FROM_NAMESPACE)
            for gold in labels}


def _unofferable_names(row: dict, leg: str) -> list[str]:
    """Gold labels of `leg` this row did NOT offer, in gold order.

    The gold keys are the artifact's own (`primary_entities` / `primary_docs`), so this
    works on an artifact written before #1823 as well as on one written after it.
    """
    gold_key = "primary_entities" if leg == "entity" else "primary_docs"
    gold = list(row.get(gold_key) or [])
    offered = list(row.get(f"{leg}_candidates") or [])
    satisfied = entity_label_satisfied if leg == "entity" else doc_label_satisfied
    return [g for g in gold if not any(satisfied(g, [c]) for c in offered)]


def _recorded_kind(row: dict, leg: str, gold: str) -> str | None:
    """The reason the run recorded for `gold` being unofferable, or None.

    None covers two different states — the label WAS offered, or the artifact predates
    #1823 and classified nothing — and `gold_offered` beside it is what tells them
    apart. That is why this is read from the row rather than guessed from the
    candidates: an artifact has to say it, or say nothing.
    """
    for item in row.get(f"{leg}_labels_unofferable") or []:
        if item.get("gold") == gold:
            return item.get("kind")
    return None


def agreement(artifact: dict) -> dict:
    """Label agreement, recomputable from the stored artifact alone.

    Denominators are the corpus's own label counts, read out of the artifact's copy
    of the gold (`counts`), so the figure cannot silently be a rate over a different
    corpus than it claims. Disagreements are listed per label, named, with the
    coverage flag beside each: a low number reported without the labels that caused
    it is the excuse-shaped instrument the item warns about.
    """
    counts = (artifact.get("corpus") or {}).get("counts") or label_counts(
        [{"id": r["id"], "expect_entities": r["primary_entities"],
          "expect_docs": r["primary_docs"]} for r in artifact["queries"]])
    ent_disagree: list[dict] = []
    doc_disagree: list[dict] = []
    ent_agreed = doc_agreed = 0
    ent_agreed_offered = doc_agreed_offered = 0
    ent_offered = doc_offered = 0
    for row in artifact["queries"]:
        gold_e = list(row.get("primary_entities") or [])
        gold_d = list(row.get("primary_docs") or [])
        chose_e = list(row.get("second_entities") or [])
        chose_d = list(row.get("second_docs") or [])
        offered_e = set(row.get("entity_candidates") or [])
        offered_d = set(row.get("doc_candidates") or [])
        ent_offered += sum(1 for g in gold_e
                           if any(entity_label_satisfied(g, [c]) for c in offered_e))
        doc_offered += sum(1 for g in gold_d
                           if any(doc_label_satisfied(g, [c]) for c in offered_d))
        for g in gold_e:
            # Offered-only agreement (#1823): a label the menu never carried could not
            # be agreed with, so agreeing over the OFFERED labels is the reading that
            # says anything about the two labelers, and the gap between the two rates is
            # what the caps cost. The all-gold rate stays the veto figure; this is the
            # half that tells a cap artefact from a labelling disagreement.
            if entity_label_satisfied(g, chose_e):
                ent_agreed += 1
                if any(entity_label_satisfied(g, [c]) for c in offered_e):
                    ent_agreed_offered += 1
            else:
                ent_disagree.append({"id": row["id"], "query": row["query"],
                                     "gold": g, "second_labels": chose_e,
                                     "gold_offered": any(
                                         entity_label_satisfied(g, [c])
                                         for c in offered_e),
                                     "unofferable_kind": _recorded_kind(row, "entity",
                                                                        g)})
        for g in gold_d:
            if doc_label_satisfied(g, chose_d):
                doc_agreed += 1
                if any(doc_label_satisfied(g, [c]) for c in offered_d):
                    doc_agreed_offered += 1
            else:
                doc_disagree.append({"id": row["id"], "query": row["query"],
                                     "gold": g, "second_labels": chose_d,
                                     "gold_offered": any(
                                         doc_label_satisfied(g, [c])
                                         for c in offered_d),
                                     "unofferable_kind": _recorded_kind(row, "doc", g)})

    def rate(num: int, den: int) -> float | None:
        # A zero denominator is None with a reason, never 0.0: 0.0 would read as a
        # measured floor and this file's whole subject is a rate nobody can read.
        return round(num / den, 4) if den else None

    return {
        "entity_label_agreement": rate(ent_agreed, counts["entity_labels"]),
        "doc_label_agreement": rate(doc_agreed, counts["doc_labels"]),
        # #1823: the same numerator over the offered labels only. `entity_label_agreement`
        # remains the #654 veto figure — the veto asks whether the gold can be trusted at
        # all, and a gold nobody was offered cannot be trusted — but on 2026-09-29 that
        # figure was 0.3404 while the offered-only figure was 0.727 (32/44), and the gap
        # was this file's own ENTITY_CAP, not the two labelers disagreeing. Printing one
        # rate without the other made an instrument reading look like an opinion reading.
        "entity_label_agreement_when_offered": rate(ent_agreed_offered, ent_offered),
        "doc_label_agreement_when_offered": rate(doc_agreed_offered, doc_offered),
        "entity_labels": counts["entity_labels"],
        "entity_labels_agreed": ent_agreed,
        "entity_labels_agreed_when_offered": ent_agreed_offered,
        "entity_labels_offered": ent_offered,
        "doc_labels": counts["doc_labels"],
        "doc_labels_agreed": doc_agreed,
        "doc_labels_agreed_when_offered": doc_agreed_offered,
        "doc_labels_offered": doc_offered,
        "disagreements": {"entity": ent_disagree, "doc": doc_disagree},
    }


def _recorded_unofferable_detail(rows: list[dict], leg: str) -> dict | None:
    """The per-label classification the RUN recorded for `leg`, or None if it recorded none.

    None means the rows carry no `{leg}_labels_unofferable` key at all — an artifact
    written before #1823 — which is a different fact from the key being present with an
    empty list (the run offered every gold, and there was nothing to classify). Callers
    must not collapse the two: the first is an instrument that cannot explain itself, the
    second is an instrument that had nothing to explain.
    """
    key = f"{leg}_labels_unofferable"
    if not any(key in row for row in rows):
        return None
    total = outside = absent = unclassified = 0
    names: dict[str, set[str]] = {OUTSIDE_CAP: set(), ABSENT_FROM_NAMESPACE: set()}
    for row in rows:
        for item in row.get(key) or []:
            total += 1
            kind = item.get("kind")
            if kind == OUTSIDE_CAP:
                outside += 1
                names[OUTSIDE_CAP].add(str(item.get("gold")))
            elif kind == ABSENT_FROM_NAMESPACE:
                absent += 1
                names[ABSENT_FROM_NAMESPACE].add(str(item.get("gold")))
            else:
                unclassified += 1
    return {"recorded_rows": f"{len(rows)}/{len(rows)}", "total": total,
            "outside_cap": outside, "absent_from_namespace": absent,
            "unclassified": unclassified,
            "names": {"total": total, "outside_cap": len(names[OUTSIDE_CAP]),
                      "absent_from_namespace": len(names[ABSENT_FROM_NAMESPACE])},
            "absent_labels": sorted(names[ABSENT_FROM_NAMESPACE])}


def ceiling(artifact: dict, *, ids: list[str] | None = None) -> dict:
    """The gold-side ceiling per metric, and the queries it excludes and why.

    Value = the mean of `surrogate` over the measurable queries: what an answerer
    that agrees with the second labeler scores on the corpus's own gold, through
    the corpus's own scorer. A query whose gold was never offered as a candidate is
    EXCLUDED and named — leaving it in would depress the ceiling through candidate
    narrowing, and a depressed ceiling inflates `score / ceiling`, which is the one
    direction of this instrument that can excuse a real defect.

    `ids` restricts the average to a subset, and a caller must use it whenever the
    score being divided is not over the whole corpus. `run_eval.py --limit 3` scores
    three queries; dividing a three-query hit rate by an eighty-one-query ceiling is
    a ratio of two different experiments, and it is the kind of number that survives
    into a report because both halves look like rates.
    """
    rows = artifact["queries"]
    if ids is not None:
        wanted = set(ids)
        rows = [r for r in rows if r["id"] in wanted]
    ent_ok = [r for r in rows
              if not r.get("primary_entities")
              or r.get("entity_labels_offered", 0) > 0]
    doc_ok = [r for r in rows
              if not r.get("primary_docs") or r.get("doc_labels_offered", 0) > 0]
    excluded_entity = [r["id"] for r in rows if r not in ent_ok]
    excluded_doc = [r["id"] for r in rows if r not in doc_ok]
    values: dict[str, Any] = {}
    ns: dict[str, int] = {}
    for metric in ENTITY_METRICS:
        vals = [(_r.get("surrogate") or {}).get(_FIELD[metric]) for _r in ent_ok]
        known = [v for v in vals if v is not None]
        values[metric] = (round(_avg(vals), 4) if known else None)
        ns[metric] = len(known)
    for metric in DOC_METRICS:
        vals = [(r.get("surrogate") or {}).get(_FIELD[metric]) for r in doc_ok]
        known = [v for v in vals if v is not None]
        values[metric] = (round(_avg(vals), 4) if known else None)
        ns[metric] = len(known)
    for metric, reason in UNMEASURED_METRICS.items():
        values[metric] = None
        ns[metric] = 0
    labels_unofferable, unofferable_detail = _unofferable_counts(rows)
    return {
        "kind": CEILING_KIND,
        "values": values,
        "n": ns,
        "excluded": {"entity": excluded_entity, "doc": excluded_doc},
        "unmeasured": dict(UNMEASURED_METRICS),
        # The count `_narrow`'s docstring has promised since #654 and never wrote:
        # how many gold labels the CAPS (not the namespaces) took out of this ceiling.
        # Per leg, and the split beside it, because the number alone reads as "the
        # store could not answer these" when it is usually this file's own 40-name menu.
        "labels_unofferable": labels_unofferable,
        "labels_unofferable_detail": unofferable_detail,
    }


def _unofferable_counts(rows: list[dict]) -> tuple[dict, dict]:
    """Per-leg cap-caused exclusion count, and the full split, from the recorded rows.

    A leg whose rows carry no `*_labels_unofferable` key at all reports `None` counts —
    an artifact labelled before #1823 recorded no classification, and "0" would be the
    same false statement in the other direction.
    """
    counts: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    for leg in ("entity", "doc"):
        key = f"{leg}_labels_unofferable"
        recorded = [r for r in rows if isinstance(r.get(key), list)]
        if not recorded:
            # None on both fields: the run recorded nothing, which is a different fact
            # from a measured zero and must not collapse into one readable number.
            counts[leg] = None
            detail[leg] = None
            continue
        items = [it for r in recorded for it in r[key]]
        outside = [it for it in items if it.get("kind") == OUTSIDE_CAP]
        absent = [it for it in items if it.get("kind") == ABSENT_FROM_NAMESPACE]
        absent_labels = sorted({str(it.get("gold")) for it in absent})
        counts[leg] = len(outside)
        detail[leg] = {
            "recorded_rows": f"{len(recorded)}/{len(rows)}",
            "total": len(items),
            "outside_cap": len(outside),
            "absent_from_namespace": len(absent),
            "unclassified": len(items) - len(outside) - len(absent),
            "names": {
                "total": len({str(it.get("gold")) for it in items}),
                "outside_cap": len({str(it.get("gold")) for it in outside}),
                "absent_from_namespace": len(absent_labels),
            },
            "absent_labels": absent_labels,
        }
    return counts, detail


def unofferable_detail(artifact: dict, *, leg: str = "entity",
                       namespace: list[str] | None = None) -> dict:
    """Gold labels of `leg` that could not be offered, split by WHY, and say where
    that answer came from.

    Three sources, and the reader needs to see which one is speaking:

      `artifact`             the labelling run recorded the classification per label;
      `namespace-tested-now` an artifact from before #1823 records nothing, so this
                             function ran the namespace test against the namespace it
                             was handed — which is TODAY's namespace, not the one that
                             run narrowed, and the returned `namespace_size` says so;
      `unrecorded`           neither is available, so there is no reason to report and
                             the count comes back None rather than a guess.

    The reason this exists at all: with only the exclusion COUNT available, `--print`
    asserted "the fix set is the fragmenting entity ids, not the retriever" for 50
    entity golds, 52-of-53-names of which the namespace held and the cap of 40 had
    dropped (#1823). A count cannot tell those two stories apart.
    """
    if leg not in ("entity", "doc"):
        raise ValueError(f"unknown leg {leg!r}; expected 'entity' or 'doc'")
    recorded = ((artifact.get("ceiling") or {}).get("labels_unofferable_detail")
                or {}).get(leg)
    if isinstance(recorded, dict) and recorded.get("total") is not None:
        return {**recorded, "source": "artifact", "namespace_size": None}
    rows = artifact.get("queries") or []
    unclassified = [str(gold) for row in rows
                    for gold in _unofferable_names(row, leg)]
    if not unclassified:
        return {"recorded_rows": f"{len(rows)}/{len(rows)}", "total": 0,
                "outside_cap": 0, "absent_from_namespace": 0, "unclassified": 0,
                "names": {"total": 0, "outside_cap": 0, "absent_from_namespace": 0},
                "absent_labels": [], "source": "artifact",
                "namespace_size": (len(namespace) if namespace else None)}
    if not namespace:
        return {"recorded_rows": f"0/{len(rows)}", "total": len(unclassified),
                "outside_cap": None, "absent_from_namespace": None,
                "unclassified": len(unclassified), "names": None, "absent_labels": None,
                "source": "unrecorded", "namespace_size": None}
    kinds = classify_unofferable(unclassified, leg=leg, namespace=namespace)
    # OCCURRENCES counted, distinct names reported only under `names` — the same units the
    # recorded and unrecorded branches use, because this figure prints beside the
    # disagreement dump and `ceiling.excluded`, both of which count labels per query.
    # `classify_unofferable` returns a name-keyed map, so counting IT (as this branch
    # first did) made the one artifact on disk print "unofferable entity labels: 31" five
    # lines above "disagreed entity labels: 62", 50 of them with `gold_offered=False`, and
    # then 31/44 beside the header's own 44/94 — three readings of one 50-label set. The
    # reason a label repeats is that the same name is gold for several queries, which is
    # what the artifact and its reader both count.
    outside_list = [g for g in unclassified if kinds.get(g) == OUTSIDE_CAP]
    absent_list = [g for g in unclassified if kinds.get(g) == ABSENT_FROM_NAMESPACE]
    absent_labels = sorted(set(absent_list))
    return {"recorded_rows": f"0/{len(rows)}", "total": len(unclassified),
            "outside_cap": len(outside_list),
            "absent_from_namespace": len(absent_list),
            "unclassified": len(unclassified) - len(outside_list) - len(absent_list),
            "names": {"total": len(set(unclassified)),
                      "outside_cap": len(set(outside_list)),
                      "absent_from_namespace": len(absent_labels)},
            "absent_labels": absent_labels, "source": "namespace-tested-now",
            "namespace_size": len(namespace)}


#: metric name in summary.overall -> per-query scoring key
_FIELD = {
    "entity_hit_rate": "entity_hit",
    "entity_recall_avg": "entity_recall",
    "doc_hit_rate": "doc_hit",
    "doc_recall_avg": "doc_recall",
    "mrr_doc": "rr_doc",
    "ndcg10": "ndcg10",
}


# ── split-half (reported with the denominator it could actually run on) ──────

def split_half(artifact: dict, *, resamples: int = 200,
               seed: int = SEED) -> dict:
    """The talk's literal fallback, emitted but NOT the headline.

    It needs at least two gold labels on a query to split anything. Measured on the
    live corpus this pass (81 queries): 19 of its 66 entity-labelled queries carry
    two or more gold entities, and 49 of 74 doc-labelled queries carry two or more
    gold docs — so the entity leg is where a random half is often empty and the
    resample averages zeros into a number that looks like a ceiling, while the doc
    leg has plenty to split. The counts are returned beside the figure for that
    reason: `queries` says how much of the corpus a given resample actually saw, and
    a figure quoted without it is unreadable. Re-measure both numbers before quoting
    them anywhere; the corpus moves.
    """
    rng = random.Random(f"{int(seed)}:split-half")
    out: dict[str, Any] = {}
    for leg, key, field in (("entity", "primary_entities", "entity_hit"),
                            ("doc", "primary_docs", "doc_hit")):
        rows = [r for r in artifact["queries"] if len(r.get(key) or []) >= 2]
        if not rows:
            out[leg] = {"agreement": None, "resamples": resamples,
                        "queries": 0,
                        "reason": "no query carries two gold labels on this leg, so "
                                  "there is nothing to split"}
            continue
        total = hits = 0
        for _ in range(resamples):
            for r in rows:
                labels = list(r[key])
                rng.shuffle(labels)
                half = max(1, len(labels) // 2)
                gold, answer = labels[:half], labels[half:]
                sat = entity_label_satisfied if leg == "entity" else doc_label_satisfied
                for g in gold:
                    total += 1
                    hits += 1 if sat(g, answer) else 0
        out[leg] = {"agreement": (round(hits / total, 4) if total else None),
                    "resamples": resamples, "queries": len(rows),
                    "comparisons": total}
    return out


# ── the artifact on disk ─────────────────────────────────────────────────────

#: Test seam for the artifact directory ONLY. `LLOYD_DATA` cannot serve here: it
#: relocates the KG, the facts and the baselines together, and a test that wants to
#: exercise "no ceiling artifact exists" would also take the corpus away, so the run
#: would exit on the store probe instead of emitting `ceiling: null` — which is the
#: exact behaviour under test. The nightly never sets this, so the nightly path is
#: the un-overridden one.
OUT_DIR_ENV = "LLOYD_LABEL_AGREEMENT_DIR"


def default_out_dir() -> Path:
    from app.paths import EVAL_BASELINES_DIR
    override = os.environ.get(OUT_DIR_ENV)
    if override:
        return Path(override)
    return EVAL_BASELINES_DIR / ARTIFACT_DIR_NAME


def newest_artifact(directory: Path | str | None = None) -> Path | None:
    directory = Path(directory) if directory else default_out_dir()
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(ARTIFACT_GLOB), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


class ArtifactRefused(RuntimeError):
    """Raised for every reason an artifact must not become tonight's ceiling, with
    the reason as the message so the caller can put it in `ceiling_reason`."""


def load_artifact(path: Path | str | None = None, *, directory: Path | str | None = None,
                  allow_stub: bool = False, expect_labels_sha256: str | None = None
                  ) -> dict:
    """Read an artifact, refusing the ones that must not be believed.

    Four refusals, each a reason string and each one a way this file's own
    catalogue says a number goes wrong:
      * absent directory or no artifact          → the instrument has not run;
      * `kind: stub` labeler                     → a stub's agreement is a property
        of the stub;
      * `labels_sha256` differs from the corpus  → the gold moved under the ceiling
        (`9b028e9` re-pointed 22 labels with the ids untouched);
      * a stored `agreement` that does not recompute → the artifact was edited.
    """
    p = Path(path) if path else newest_artifact(directory)
    if p is None:
        raise ArtifactRefused(
            f"no label-agreement artifact under "
            f"{directory or default_out_dir()}; run "
            f"`python eval/label_agreement_ceiling.py --engine <engine>`")
    try:
        art = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactRefused(f"unreadable label-agreement artifact {p}: {exc}")
    art["_path"] = str(p)
    kind = str(((art.get("labeler") or {}).get("kind")) or "")
    if kind == "stub" and not allow_stub:
        raise ArtifactRefused(
            f"{p} was labelled by a stub ({kind}); a stub agreement is not a "
            f"measured ceiling — pass --allow-stub-ceiling to use it anyway")
    want = (art.get("corpus") or {}).get("labels_sha256")
    if expect_labels_sha256 and want and want != expect_labels_sha256:
        raise ArtifactRefused(
            f"the gold labels moved since {p} (artifact labels_sha256 {want}, "
            f"corpus {expect_labels_sha256}); the ceiling describes labels that no "
            f"longer exist — re-run the labeler")
    recomputed = agreement(art)
    stored = art.get("agreement") or {}
    for key in ("entity_label_agreement", "doc_label_agreement"):
        if stored.get(key) != recomputed[key]:
            raise ArtifactRefused(
                f"{p} stores {key}={stored.get(key)!r} but its stored labels "
                f"recompute to {recomputed[key]!r}; refusing an artifact whose "
                f"figure does not reproduce from its own contents")
    # The ceiling gets the same treatment as the agreement, for a sharper reason: it
    # is the DIVISOR. An artifact whose stored surrogate cannot be rebuilt from its
    # own rows — hand-edited, or written by a scorer of a different shape — would
    # divide tonight's score by a number nothing in the file supports, and the
    # resulting ratio would look more authoritative than the raw rate it annotates.
    stored_ceil = (art.get("ceiling") or {}).get("values") or {}
    recomputed_ceil = ceiling(art)["values"]
    for metric, value in stored_ceil.items():
        if recomputed_ceil.get(metric) != value:
            raise ArtifactRefused(
                f"{p} stores ceiling {metric}={value!r} but its rows recompute to "
                f"{recomputed_ceil.get(metric)!r}; refusing a ceiling that does not "
                f"reproduce from the artifact it is stored in")
    return art


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Second labelling pass and label-agreement ceiling (#654)")
    ap.add_argument("--engine", default=None,
                    help="configured engine to label with: primary | secondary | "
                         "djev. No default: which model produces the ceiling is "
                         "part of what the number means.")
    ap.add_argument("--labeler-url", default=None,
                    help="chat-completions URL, with --labeler-model")
    ap.add_argument("--labeler-model", default=None,
                    help="served model name to ask for")
    ap.add_argument("--stub", action="store_true",
                    help="label with the deterministic stub (plumbing smoke only; "
                         "the artifact is stamped stub and refused by the loader)")
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--entity-cap", type=int, default=ENTITY_CAP)
    ap.add_argument("--doc-cap", type=int, default=DOC_CAP)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to wait between queries against a live engine")
    # Declared here because it is USED below and advertised in this module's own
    # docstring, and it had neither until 2026-09-24: `main` read `args.split_half`,
    # so every successful labelling run wrote its artifact and then died with
    # `AttributeError: 'Namespace' object has no attribute 'split_half'` on the print
    # line — the number was on disk and the report never reached the person running
    # it. Default 0 keeps it opt-in, which is what the docstring's "reported WITH the
    # query count it could actually run on, not as a headline" asks for.
    ap.add_argument("--split-half", type=int, default=0,
                    help="also run the talk's literal split-half resample this many "
                         "times and print it with the query count it saw. Off by "
                         "default: most entity queries have one gold label, so the "
                         "resample averages zeros into a number shaped like a ceiling.")
    # `--print` is not a convenience flag: `skills/retrieval-eval/SKILL.md` Step 2b
    # tells the nightly reporter to run `eval/label_agreement_ceiling.py --print`, and
    # the review rung caught that the flag did not exist. Its whole subject is a
    # documented command that exits 2 with `unrecognized arguments` — the reporter
    # reads that as "no ceiling" and the instrument's output never reaches the
    # report. A CLI flag is a contract, so the pair
    # `tests/test_eval_label_agreement.py::test_every_flag_the_skill_documents_parses`
    # and `test_the_print_flag_is_the_documented_no_write_report` reads the skill
    # straight from the vault's committed HEAD and parses each documented flag,
    # skipping only when HEAD itself is unreadable.
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print the report from the newest artifact and exit; "
                         "labels nothing, writes nothing. This is what "
                         "skills/retrieval-eval/SKILL.md Step 2b runs nightly.")
    ap.add_argument("--artifact", default=None,
                    help="artifact to read for --print (default: newest)")
    ap.add_argument("--allow-stub-ceiling", action="store_true",
                    help="let a stub-labelled artifact serve as the ceiling. The "
                         "loader refuses one by default: a stub's agreement is a "
                         "property of the stub.")
    ap.add_argument("--no-labels-sha-check", action="store_true",
                    help="skip the gold-moved refusal. Only for reading an old "
                         "ceiling beside a corpus that has since been re-pointed.")
    return ap


def _pct(num: int, den: int) -> str:
    """`num` as a one-decimal share of `den`, or `n/a` when there is no denominator.

    A share rather than a count is what decides a remediation: 52 cap-dropped labels out
    of 53 says the cap is the problem, and 2 out of 3 in a fixture says the same thing in
    different units. A zero denominator is `n/a`, never 0 — the same rule as the rates in
    `agreement()`, where a 0.0 would read as a measured floor.
    """
    return f"{100.0 * num / den:.1f}%" if den else "n/a"


def sub_veto_advisory(art: dict, *, leg: str = "entity",
                      namespace: list[str] | None = None) -> str:
    """What a below-veto agreement figure may be read as, from what was MEASURED (#1823).

    The paragraph this replaces asserted "the fix set is the fragmenting entity ids, not
    the retriever" for every sub-veto run, computed from no namespace test anywhere in
    this file. Two of the three remediations #1823 names — merging fragmenting entities,
    re-labelling gold — are supported only by golds ABSENT from the namespace; golds that
    are merely outside the cap are this file's own narrowing, and on 2026-09-29 that was
    52 of the 53 unofferable entity gold NAMES, so the sentence sent a reader to fix a
    store that was not the problem. This states the measured split and the owner of each
    half, and it never names a remediation for labels the namespace test does not support.
    """
    # `_agreement_for_print`, not a raw `art["agreement"]`: this block prints four lines
    # below the header that has already shown the offered-only rate, and on an artifact
    # predating those keys a raw read printed `None (0/44)` where the header showed 0.7273
    # — one print giving two figures for one measurement, on the very artifact #1823 was
    # filed from and the only one this has to work on today.
    a = _agreement_for_print(art)
    key = f"{leg}_label_agreement"
    detail = unofferable_detail(art, leg=leg, namespace=namespace)
    agreed = int(a.get(f"{leg}_labels_agreed") or 0)
    total = int(a.get(f"{leg}_labels") or 0)
    offered = int(a.get(f"{leg}_labels_offered") or 0)
    agreed_offered = int(a.get(f"{leg}_labels_agreed_when_offered") or 0)
    lines = [f"BELOW {AGREEMENT_VETO:.2f} (#654): the {leg} hit rate is at least partly "
             f"label noise, and every label above is a candidate noise source. Which "
             f"fix owns that noise is measured below, not assumed."]
    lines.append(
        f"  agreement among the labels that WERE offered: "
        f"{a.get(f'{leg}_label_agreement_when_offered')} "
        f"({agreed_offered}/{offered}), against {a.get(key)} ({agreed}/{total}) over all "
        f"gold — the gap is what the caps cost, not what the labelers disagree on.")
    if detail["source"] == "unrecorded":
        lines.append(
            f"  {detail['total']} {leg} gold labels were never offered and this artifact "
            "records no reason for them (per-label classification is from #1823). Whether "
            "those are the caps or the namespace is UNMEASURED, so neither widening a cap "
            "nor merging entities is supported by this artifact.")
        return "\n".join(lines)
    where = ("reason as recorded by the run" if detail["source"] == "artifact"
             else f"reason tested against the namespace of {detail['namespace_size']} "
                  "names supplied now, which is not the namespace this run narrowed")
    total = detail["total"]
    cap, absent = detail["outside_cap"], detail["absent_from_namespace"]
    cap_name = "ENTITY_CAP" if leg == "entity" else "DOC_CAP"
    lines.append(
        # Labels counted, distinct names in brackets: one name can be gold for several
        # queries (31 names across 50 label occurrences on the 2026-09-29 artifact), and
        # both figures get compared against other counts in this same print.
        f"  unofferable {leg} labels: {total}"
        + (f" ({detail['names']['total']} distinct names)"
           if isinstance(detail.get("names"), dict) else "")
        + f" ({where}), of which:"
        + (f"\n    {cap} ({_pct(cap, total)}) outside the candidate cap \u2014 the "
           f"namespace DOES hold a name satisfying each, so this is a cap artefact, not a "
           f"labelling disagreement" if cap else "")
        + (f"\n    {absent} ({_pct(absent, total)}) absent from the namespace: "
           f"{', '.join(detail['absent_labels'])}" if absent else
           "\n    0 absent from the namespace, so nothing on this artifact supports a "
           "fragmenting-entity fix set")
        + (f"\n    {detail['unclassified']} unclassified: no namespace was loaded when "
           f"they were dropped" if detail["unclassified"] else "") + ".")
    if cap:
        lines.append(
            f"  fix owner for those {cap}: {cap_name}, this instrument's own constant, "
            "and the menu builder that orders the namespace under it. "
            f"{uncapped_note(art, leg)}. Either change "
            "re-bases the ceiling every baseline in the window divides by, so the "
            "decision and its re-run belong in one commit.")
    if absent:
        lines.append(
            "  fix owner for those absent labels: the entity store, and THIS is the "
            "fragmenting-entity fix set \u2014 these names and no others on this line. "
            "Widening the menu and fixing the entity ids are different jobs with "
            "different owners; until the split was measured, one sentence gave all of "
            "them the same answer.")
    return "\n".join(lines)


def menu_builder_line(art: dict) -> str:
    """Which builder made this artifact's menus (#1937), for `--print`."""
    caps = art.get("caps") or {}
    name = art.get("menu_builder")
    shown = name if name else (f"unrecorded — this artifact predates the key; the only "
                               f"builder that existed then is {MENU_BUILDER}")
    return (f"menu builder: {shown}   caps entity={caps.get('entity')} "
            f"doc={caps.get('doc')}")


def uncapped_note(art: dict, leg: str) -> str:
    """How many of a leg's labels stay unoffered with the ranking uncapped (#1937).

    The sentence that stops the unofferable line reading as "widen the cap": the
    count no cap recovers, and — when the run recorded the ladder — how little a
    cap eight times wider buys. An artifact predating the `offered` block has the
    uncapped count only if its rows classified each drop; otherwise it is unmeasured
    and says so.
    """
    block = (art.get("offered") or {}).get(leg)
    if isinstance(block, dict) and block.get("labels") is not None:
        cap, total = int(block["cap"]), int(block["labels"])
        by = block.get("offered_by_cap") or {}
        widest = str(cap * OFFERED_LADDER[-1])
        return (f"{block['unoffered_uncapped']} remain unoffered even with the ranking "
                f"uncapped; offered {by.get(str(cap))}/{total} at cap {cap}, "
                f"{by.get(widest)}/{total} at cap {widest}, "
                f"{block['offered_uncapped']}/{total} uncapped — what a wider cap does "
                "not reach is hidden by the builder's order, not by the cap")
    detail = ((art.get("ceiling") or {}).get("labels_unofferable_detail") or {}).get(leg)
    if isinstance(detail, dict) and detail.get("absent_from_namespace") is not None \
            and not detail.get("unclassified"):
        return (f"{detail['absent_from_namespace']} remain unoffered even with the ranking "
                "uncapped; how many a WIDER cap would reach is unrecorded by this "
                "artifact, and outside_cap does not mean a bounded cap recovers them")
    return ("how many remain unoffered with the ranking uncapped is unrecorded by this "
            "artifact, and outside_cap does not mean a bounded cap recovers them")


def _agreement_for_print(art: dict) -> dict:
    """The artifact's own agreement block, with any #1823 field filled by recomputation.

    The stored block is printed as recorded and never overwritten — a figure on disk is
    evidence, and re-deriving it here would let a printer edit a measurement. What it
    cannot do is print a key the artifact predates: the offered-only rates are new, and
    the single artifact on disk was written before them, which is the very file the split
    has to be measured on. `agreement()` is documented as recomputable from the stored
    rows alone, deterministically and with no engine call, so filling ONLY the absent
    keys is the same measurement, not a new one. Nothing is invented when the rows cannot
    answer: the rate stays None and prints as None.
    """
    a = dict(art.get("agreement") or {})
    missing = [k for leg in ("entity", "doc")
               for k in (f"{leg}_label_agreement_when_offered",
                         f"{leg}_labels_agreed_when_offered")
               if k not in a]
    if missing:
        fresh = agreement(art)
        for key in missing:
            a[key] = fresh.get(key)
    return a


def print_summary(art: dict, *, split_half_result: dict | None = None,
                  out_path: Path | str | None = None,
                  entity_namespace: list[str] | None = None) -> None:
    """Emit the agreement/ceiling/disagreement report. One implementation because
    the same numbers reach a person two ways: `main` after a labelling run, and
    `--print` (the nightly command) from a stored artifact. Two printers means the
    nightly can quietly print less than the run did — including the below-0.80
    advisory and the named disagreement set that clause 5 exists to guarantee.
    """
    a = _agreement_for_print(art)
    c = art["ceiling"]
    lab = art.get("labeler") or {}
    # Which model produced a ceiling is part of what the number means, and `--print`
    # reads an artifact somebody else made. A stub-labelled ceiling read as a measured
    # one is the failure mode the loader refuses by default; printing the identity on
    # the same page as the ratio is the second line of that defence.
    print(f"\nlabeler: {lab.get('kind')}/{lab.get('model')} @ {lab.get('endpoint')}"
          f"   seed {art.get('seed')}   ran {art.get('ran_at')}")
    print(menu_builder_line(art))
    # Both rates, per leg, every time (#1823 clause 4). The all-gold figure is what the
    # #654 veto is evaluated on; the offered-only figure is what the two labelers did on
    # the labels they could actually choose from. On the 2026-09-29 artifact the entity leg
    # reads 0.3404 and 0.7273 — 32 of 94 gold labels agreed, of which all 32 were among
    # the 44 that were offered — and a reader given only the first number cannot tell a cap
    # artefact from an opinion, which is precisely the reading that stood for a month.
    print(f"label agreement   entity {a['entity_label_agreement']} all-gold "
          f"({a['entity_labels_agreed']}/{a['entity_labels']} labels)  |  "
          f"{a['entity_label_agreement_when_offered']} among offered "
          f"({a['entity_labels_offered']} were offered as candidates)")
    print(f"                  doc    {a['doc_label_agreement']} all-gold "
          f"({a['doc_labels_agreed']}/{a['doc_labels']} labels)  |  "
          f"{a['doc_label_agreement_when_offered']} among offered "
          f"({a['doc_labels_offered']} were offered)")
    print(f"gold-side ceiling (kind {c['kind']}):")
    for metric, value in c["values"].items():
        why = c["unmeasured"].get(metric)
        print(f"  {metric:24} {value if value is not None else f'null — {why}'}"
              + (f"   n={c['n'][metric]}" if value is not None else ""))
    if c["excluded"]["entity"] or c["excluded"]["doc"]:
        print(f"  excluded (gold never offered): entity={c['excluded']['entity']} "
              f"doc={c['excluded']['doc']}")
        # The excluded total is only half the sentence: 52 of the entity exclusions on
        # the 2026-09-29 artifact are this file's caps narrowing a namespace that DOES
        # hold the name, and one name (`Nightly Reflection`) has no namespace entity at
        # all. Those two halves have opposite owners, so `labels_unofferable` is printed
        # beside the count that has always been printed (#1823 clause 2).
        lu = c.get("labels_unofferable") or {}
        detail = (c.get("labels_unofferable_detail") or {})
        for leg in ("entity", "doc"):
            if lu.get(leg) is None:
                print(f"  labels_unofferable ({leg}): unrecorded by this artifact")
            elif lu.get(leg) is not None:
                d = detail.get(leg) or {}
                print(f"  labels_unofferable ({leg}): {lu[leg]} excluded by the caps, "
                      f"of {d.get('total', 0)} unofferable in total "
                      f"({d.get('absent_from_namespace', 0)} absent from the namespace"
                      + (f", {d.get('unclassified', 0)} unclassified"
                         if d.get("unclassified") else "") + ")"
                      + f"; {uncapped_note(art, leg)}")
    if split_half_result is not None:
        print(f"split-half: {json.dumps(split_half_result)}")
    # Printed unconditionally, and with no flag to suppress it: clause 5 is "a low
    # ceiling is never reported without the labels that caused it", and a
    # `--no-disagreements` switch would be an excuse-shaped instrument with a CLI
    # handle. There is one printer for both the labelling run and the nightly
    # `--print`, so the nightly cannot quietly print less than the run did.
    dis = a["disagreements"]["entity"]
    print(f"\ndisagreed entity labels: {len(dis)}")
    for d in dis:
        # `unofferable=` is the same fact `gold_offered=` states, plus why: a False
        # `gold_offered` is 50 cap artefacts and one namespace absence on this artifact,
        # and the dump has always printed the False without the difference (#1823).
        kind = f"  unofferable={d['unofferable_kind']}" if d.get("unofferable_kind") else ""
        print(f"  {d['id']:<28} gold {d['gold']!r}  second {d['second_labels']!r}"
              f"  gold_offered={d['gold_offered']}{kind}")
    if out_path:
        print(f"artifact: {out_path}")
    ent_agree = a["entity_label_agreement"]
    if ent_agree is not None and float(ent_agree) < AGREEMENT_VETO:
        print()
        print(sub_veto_advisory(art, leg="entity", namespace=entity_namespace))


def main() -> int:
    args = build_parser().parse_args()

    if args.print_only:
        # Still read-only by construction: no engine is resolved, nothing is written, and
        # no GPU call is reachable. What #1823 adds is one READ of the entity namespace —
        # the same `entity_name_table()` call `--dry-run` makes three lines below, and the
        # only thing that can say whether a gold dropped by the caps was ever a name the
        # store held. An artifact written before #1823 recorded no reason per label, so
        # without this read the advisory could only ever say UNMEASURED on the very
        # artifact this item was filed from. It is labelled as re-derived where it prints,
        # and a store that will not open costs the reporter its split, not its numbers:
        # None falls back to the recorded detail and then to an UNMEASURED line that says
        # which is missing.
        try:
            art = load_artifact(args.artifact,
                                allow_stub=args.allow_stub_ceiling,
                                expect_labels_sha256=(
                                    None if args.no_labels_sha_check
                                    else labels_sha256(load_corpus(args.corpus))))
        except ArtifactRefused as exc:
            print(f"NO CEILING: {exc}")
            return 2
        try:
            names = entity_name_table()
        except Exception as exc:                     # noqa: BLE001
            # A reporting box with no store still gets the artifact's own numbers; only
            # the split is lost, and the advisory says so in as many words.
            names = None
            print(f"(entity namespace unavailable: {type(exc).__name__}: {exc})")
        print_summary(art, out_path=art.get("_path"), entity_namespace=names)
        return 0

    queries = load_corpus(args.corpus)
    if args.limit:
        queries = queries[:args.limit]
    counts = label_counts(queries)

    if args.stub:
        labeler, identity = stub_labeler(), {"kind": "stub", "model": "stub",
                                             "endpoint": "in-process"}
    elif args.labeler_url and args.labeler_model:
        labeler = http_labeler(args.labeler_url, args.labeler_model)
        identity = {"kind": "engine", "model": args.labeler_model,
                    "endpoint": args.labeler_url}
    elif args.engine:
        url, model = resolve_engine(args.engine)
        labeler = http_labeler(url, model)
        identity = {"kind": "engine", "model": model, "endpoint": url,
                    "engine": args.engine}
    else:
        print("LABELER_REQUIRED: pass --engine, or --labeler-url with "
              "--labeler-model, or --stub. There is no default engine on purpose: "
              "which model produced the ceiling changes what the number means.")
        return 2

    print(f"labelling {len(queries)} queries "
          f"({counts['entity_labels']} entity labels / {counts['doc_labels']} doc "
          f"labels) with {identity['model']} @ {identity['endpoint']}")
    names = entity_name_table()
    vault = vault_markdown_paths()
    print(f"  entity namespace: {len(names)} names   "
          f"doc namespace: {len(vault)} vault paths")

    art = label_corpus(labeler=labeler, labeler_identity=identity, queries=queries,
                       entity_names=names, vault_paths=vault, seed=args.seed,
                       entity_cap=args.entity_cap, doc_cap=args.doc_cap,
                       corpus_path=args.corpus, sleep_seconds=args.sleep)

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # The glob's own prefix, wildcard dropped: `ARTIFACT_GLOB[:-5]` kept the
    # `*` and the first real run wrote `label-agreement-*-<stamp>.json`.
    out_path = out_dir / f"{ARTIFACT_GLOB.split('*')[0]}{stamp}.json"
    out_path.write_text(json.dumps(art, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    sh = (split_half(art, resamples=args.split_half, seed=args.seed)
          if args.split_half else None)
    # The namespace this run narrowed is the namespace that classified every exclusion in
    # it, so it goes to the same printer `--print` uses; the artifact's own recorded detail
    # wins over it (`unofferable_detail` prefers "artifact"), and passing it keeps the two
    # call sites of the one printer equivalent by construction.
    print_summary(art, split_half_result=sh, out_path=out_path, entity_namespace=names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
