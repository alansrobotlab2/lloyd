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
        exclusions back into measurements, and `labels_unofferable` beside the ceiling
        says how many the caps caused.
    A query with no overlap anywhere still gets `cap` candidates, filled in name
    order, so a zero-overlap query is measured on a pool like every other rather
    than on an empty one that would report agreement 0 for a non-reason.
    """
    toks = query_tokens(query)
    scored: list[tuple[int, int, str]] = []
    for name in pool:
        hay = ev._norm(name)
        score = sum(1 for t in toks if t in hay)
        scored.append((-score, len(name), name))
    scored.sort()
    chosen = [n for _s, _l, n in scored][:cap]
    _rng(seed, query_id, leg).shuffle(chosen)
    return chosen


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
    """
    post = transport or http_post_json

    def label(request: dict) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": request["prompt"]}],
            "temperature": temperature, "max_tokens": 300,
        }
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


def resolve_engine(name: str) -> tuple[str, str]:
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
        served = str(block.get("model") or "")
        if not served:
            # vLLM refuses a model name it is not serving, and `djev` is not the
            # served name. Guessing one would surface as an HTTP error on query 1
            # of 81 — after the run has cost a person's decision to start it.
            raise SystemExit(
                f"LABELER_MODEL_UNKNOWN: config.yaml records no served model name "
                f"under `djev:`. Ask the endpoint what it serves "
                f"(`GET {base}/v1/models`) and pass "
                f"`--labeler-url {base.rstrip('/')}/v1/chat/completions "
                f"--labeler-model <served name>`. Naming the model that produced a "
                f"ceiling is part of what the ceiling means, so it is not a value "
                f"to invent here.")
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
            if entity_label_satisfied(g, chose_e):
                ent_agreed += 1
            else:
                ent_disagree.append({"id": row["id"], "query": row["query"],
                                     "gold": g, "second_labels": chose_e,
                                     "gold_offered": any(
                                         entity_label_satisfied(g, [c])
                                         for c in offered_e)})
        for g in gold_d:
            if doc_label_satisfied(g, chose_d):
                doc_agreed += 1
            else:
                doc_disagree.append({"id": row["id"], "query": row["query"],
                                     "gold": g, "second_labels": chose_d,
                                     "gold_offered": any(
                                         doc_label_satisfied(g, [c])
                                         for c in offered_d)})

    def rate(num: int, den: int) -> float | None:
        # A zero denominator is None with a reason, never 0.0: 0.0 would read as a
        # measured floor and this file's whole subject is a rate nobody can read.
        return round(num / den, 4) if den else None

    return {
        "entity_label_agreement": rate(ent_agreed, counts["entity_labels"]),
        "doc_label_agreement": rate(doc_agreed, counts["doc_labels"]),
        "entity_labels": counts["entity_labels"],
        "entity_labels_agreed": ent_agreed,
        "entity_labels_offered": ent_offered,
        "doc_labels": counts["doc_labels"],
        "doc_labels_agreed": doc_agreed,
        "doc_labels_offered": doc_offered,
        "disagreements": {"entity": ent_disagree, "doc": doc_disagree},
    }


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
    return {
        "kind": CEILING_KIND,
        "values": values,
        "n": ns,
        "excluded": {"entity": excluded_entity, "doc": excluded_doc},
        "unmeasured": dict(UNMEASURED_METRICS),
    }


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


def print_summary(art: dict, *, split_half_result: dict | None = None,
                  out_path: Path | str | None = None) -> None:
    """Emit the agreement/ceiling/disagreement report. One implementation because
    the same numbers reach a person two ways: `main` after a labelling run, and
    `--print` (the nightly command) from a stored artifact. Two printers means the
    nightly can quietly print less than the run did — including the below-0.80
    advisory and the named disagreement set that clause 5 exists to guarantee.
    """
    a = art["agreement"]
    c = art["ceiling"]
    lab = art.get("labeler") or {}
    # Which model produced a ceiling is part of what the number means, and `--print`
    # reads an artifact somebody else made. A stub-labelled ceiling read as a measured
    # one is the failure mode the loader refuses by default; printing the identity on
    # the same page as the ratio is the second line of that defence.
    print(f"\nlabeler: {lab.get('kind')}/{lab.get('model')} @ {lab.get('endpoint')}"
          f"   seed {art.get('seed')}   ran {art.get('ran_at')}")
    print(f"label agreement   entity {a['entity_label_agreement']} "
          f"({a['entity_labels_agreed']}/{a['entity_labels']} labels, "
          f"{a['entity_labels_offered']} were offered as candidates)")
    print(f"                  doc    {a['doc_label_agreement']} "
          f"({a['doc_labels_agreed']}/{a['doc_labels']} labels, "
          f"{a['doc_labels_offered']} were offered)")
    print(f"gold-side ceiling (kind {c['kind']}):")
    for metric, value in c["values"].items():
        why = c["unmeasured"].get(metric)
        print(f"  {metric:24} {value if value is not None else f'null — {why}'}"
              + (f"   n={c['n'][metric]}" if value is not None else ""))
    if c["excluded"]["entity"] or c["excluded"]["doc"]:
        print(f"  excluded (gold never offered): entity={c['excluded']['entity']} "
              f"doc={c['excluded']['doc']}")
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
        print(f"  {d['id']:<28} gold {d['gold']!r}  second {d['second_labels']!r}"
              f"  gold_offered={d['gold_offered']}")
    if out_path:
        print(f"artifact: {out_path}")
    ent_agree = a["entity_label_agreement"]
    if ent_agree is not None and float(ent_agree) < 0.80:
        print("\nBELOW 0.80: the entity hit rate is at least partly label noise. "
              "Every label above is a candidate label-noise source; the fix set is "
              "the fragmenting entity ids, not the retriever.")


def main() -> int:
    args = build_parser().parse_args()

    if args.print_only:
        # Read-only by construction: no engine is resolved, no namespace is loaded,
        # nothing is written. The reporter running it must not be able to cost a GPU
        # call or move a file, and a run that happens to have an engine up prints the
        # same thing it would with none.
        try:
            art = load_artifact(args.artifact,
                                allow_stub=args.allow_stub_ceiling,
                                expect_labels_sha256=(
                                    None if args.no_labels_sha_check
                                    else labels_sha256(load_corpus(args.corpus))))
        except ArtifactRefused as exc:
            print(f"NO CEILING: {exc}")
            return 2
        print_summary(art, out_path=art.get("_path"))
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
    out_path = out_dir / f"{ARTIFACT_GLOB[:-5]}-{stamp}.json"
    out_path.write_text(json.dumps(art, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    sh = (split_half(art, resamples=args.split_half, seed=args.seed)
          if args.split_half else None)
    print_summary(art, split_half_result=sh, out_path=out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
