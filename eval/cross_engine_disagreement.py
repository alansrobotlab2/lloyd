#!/usr/bin/env python3
"""Does cross-engine disagreement localize verdict error? — backlog #2639.

Offline replay over verdict surfaces whose correct answer is ALREADY KNOWN. Two
engines answer the SAME typed question over the SAME rows, the gold is joined from
artifacts that already exist, and the artifact this writes reports whether the rows
where the two engines disagree contain the errors.

    primary  Qwen3.8-Flash-Next NVFP4   http://127.0.0.1:8096/v1/chat/completions
    djev     DiffusionGemma 26B-A4B     http://127.0.0.1:8010/v1/chat/completions

This is the only cross-FAMILY pair the box can run: GPU 2 hosts djev or the retired
secondary, never both (`architecture/infrastructure.md:63`), and the secondary Qwen
GGUF slot at :8091 has answered nothing since 2026-09-20 (`551e9044`). That is the
condition the intersection method needs — different pedigrees, different priors —
and it is why the open second-reader designs (#708 side-effect judging, #1903 a
confirm pass on every automod review refusal, #2254 judge self-preference) have been
judged on faith rather than on a number: `scripts/automod/review.py:3954-3957` says
of its own second pass "Same weights as the first pass, so agreement is not
independent evidence", and `config.yaml` keeps `automod.review.confirm: off`.

Nothing here wires into a gate, a rung or a dispatch path. The item measures and
publishes; whether a POSITIVE number licenses reopening a gating design is a ruling
this module deliberately does not make.

What is measured, per surface and per pair, always conditioned on rows where BOTH
engines returned a usable argmax:

  agreement rate                  both engines landed on the same label
  error rate inside agreement     the shared label was wrong  (== both-wrong-agreeing
                                  rows / agreement rows)
  disagreement volume share       disagreement rows / non-abstaining rows
  error-localization recall       errors inside the disagreement set / ALL errors
  both-wrong-and-agreeing         its own count and its own share of all errors —
                                  never folded into "agreement", because it is the
                                  finding this whole exercise expects most
  abstention share                printed beside every rate, never inside one
  Cohen's kappa vs gold           per engine, chance-corrected, over the SAME
                                  non-abstaining pairs every other rate uses, against
                                  the gold column the run actually scores. Agreement
                                  rates are uncorrected for chance, so an engine that
                                  emits one label on every row can print a respectable
                                  number; kappa is the statistic that cannot, and it is
                                  printed as UNINFORMATIVE rather than as a number when
                                  the arm is degenerate (#3345).

Definitions are single-valued on purpose: gold is one label per row, so a row where
the engines disagree is automatically a row where at least one is wrong, and
`errors = disagreements + both_wrong_and_agreeing` holds by construction. That is
what makes recall the reportable quantity. And it is why kappa's denominator is the
pair set rather than every row an engine happened to answer: two kappa values computed
over different row sets are not comparable, and "is djev close enough to the primary
engine" is a question about two numbers measured on the same rows.

The ceiling on the method is stated in the same source as the result (Olickel's
talk): four model families once produced an identical wrong total to the cent
because they shared a sign convention the answer key did not. Agreement is worth
exactly what the independence behind it is worth, and two models trained on the same
annotation conventions can share a failure. `both_wrong_and_agreeing` is that case,
reported as a first-class column.

Surfaces (both reuse gold that exists; neither authors a label):

  retrieval-labels-offered  `eval/label_agreement_ceiling.py`'s own menu prompt over
      the gold in the shipped djev artifact. Rows are only the (query, gold label)
      pairs that artifact reports as OFFERED — a label whose candidate menu never
      carried it was never a question any engine could answer, and counting it is
      what made the raw 0.3404 entity figure read as a failure when the offered-only
      figure was 0.7273 (32/44).

  durable-write-judge       `eval/durable_write_judge/corpus.jsonl` (20 bad / 30
      good, rule-derived from repair diffs) joined by sample id to the 50 `primary`
      rows already in #2254's owed-check artifact. Gold is the corpus's own label, and
      each row carries the verdict `primary` recorded there plus that run's
      `prompt_sha256` and endpoint as provenance. The question is the narrow typed one
      below, put to both engines in the same words — deliberately not a second copy of
      #2254's rubric prompt, which would be a second prompt to keep in step with the
      first, and an engine asked a different question than the one it recorded an
      answer to is not a re-ask of anything.

  shadow-gold-sample        arm 1 (#2824): `eval/cross_engine_disagreement_arm1/gold.jsonl`,
      120 rows sampled read-only from the djev shadow store and labelled under a written
      rubric (`rubric.py` in that directory). Unlike the two surfaces above, its gold is
      NOT lifted from an artifact that already carried a label and NOT human: a sampler
      admits a row only after re-deriving its canvas to the `inputs_digest` the seam
      itself recorded, and labels it only where a rubric line yields the answer from an
      artifact. It therefore measures the one quantity a Lloyd-authored label can
      legitimately score — whether two different model families, shown the same canvas
      and the same gold, agree and where they differ — and it does NOT measure
      localization against a shipped answer, because no shipped subsystem ever answered
      these canvases. What its recall therefore means is NOT the shipped surfaces'
      recall: an arm-1 error is "an engine disagreed with the rubric gold", where a
      shipped-surface error is "the verdict the shipped subsystem recorded was wrong".
      `rows_with_shipped_answer` is 0 on this surface and non-zero on those two, which
      is the field a reader checks before carrying a number across surfaces. `--gold`
      points the surface at another file.

Each row is asked `--repeats` times per engine (minimum 5, enforced) and reported as
a per-case pass rate over those draws, never as one draw: a nondeterministic
verifier sampled once is an anecdote. A surface whose non-abstaining denominator is
below 30 prints `UNMEASURABLE surface=<name> non_abstaining=<n>` instead of a rate —
the nightly labeler's >50% abstention makes that a live outcome, not a guard rail.

Usage (writes one dated JSON + one dated markdown table under --out):

    python eval/cross_engine_disagreement.py --out ~/lloyd-data/eval/cross-engine/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.stats import wilson_ci  # noqa: E402

# ── engines ─────────────────────────────────────────────────────────────────

#: The two readers, and the only cross-FAMILY pair this box can run. They are NAMES:
#: the URL and the served model name come from `config.yaml` and each endpoint's own
#: `/v1/models` through `eval.label_agreement_ceiling.resolve_engine` — the shipped
#: resolver the 2026-09-29 djev labeling run used, which is why the pair here cannot
#: fork from the configuration that defines it. The served name of the model that
#: answered is part of what the number means, so it is read from the thing that
#: answered rather than typed in here.
ENGINE_NAMES = ("primary", "djev")

#: :8091 is the retired llama.cpp secondary. It is deliberately absent from
#: `ENGINE_NAMES`: `resolve_engine` reads `secondary_enabled` out of config.yaml and
#: refuses a switched-off engine instead of falling back on it — a fallback would swap
#: the meaning of the number while keeping the number. `ask()` refuses the port by
#: substring as well, so no path — default, flag or injected — sends a verdict question
#: to an endpoint that has answered nothing since 2026-09-20, whose silence would then
#: be recorded as disagreement.
RETIRED_PORTS = ("8091",)

DEFAULT_RETRIEVAL_ARTIFACT = Path(
    "/home/alansrobotlab/lloyd-data/eval/baselines/label-agreement"
    "/label-agreement-20260929-005508.json")
DEFAULT_ARMS_FINAL = Path("/home/alansrobotlab/lloyd-data/eval/owedcheck-2254/arms-final.jsonl")
DEFAULT_CORPUS = REPO / "eval" / "durable_write_judge" / "corpus.jsonl"
#: Arm 1's gold (#2824) is a tracked file beside the sampler that wrote it, so the
#: table's denominators stay checkable from the checkout instead of from a runtime
#: store that grows ~10k rows a day and rotates underneath a re-run.
DEFAULT_ARM1_GOLD = REPO / "eval" / "cross_engine_disagreement_arm1" / "gold.jsonl"

# ── thresholds (the item's, quoted verbatim into the artifact) ──────────────

MIN_REPEATS = 5                 # clause 4: never one draw
MIN_NON_ABSTAINING = 30         # below this, a cell prints UNMEASURABLE
POSITIVE_MIN_RECALL = 0.70      # disagreement holds >=70% of errors
POSITIVE_MAX_VOLUME_SHARE = 0.35   # ...while covering <=35% of volume
POSITIVE_MAX_BWA_SHARE = 0.10   # both-wrong-and-agreeing <10% of all errors

AGREE, DISAGREE, ABSTAIN = "agree", "disagree", "abstain"

# ── kappa against gold (#3345) ──────────────────────────────────────────────

#: The ONE column every engine-vs-gold figure in this file is scored on. `run_rows`
#: reads it (`gold = row["gold_verdict"]`), and it is published in `meta` per surface and
#: on every kappa cell so a reader cannot re-score the durable arm on its other column:
#: `durable_rows` carries `gold` = `bad`/`good` (the corpus's own label) ALONGSIDE
#: `gold_verdict` = `flag`/`accept`, and those are different vocabularies, so scoring the
#: 2026-10-16 rows on `gold` yields kappa 0.0000 for BOTH engines (po 0.0 — not one row
#: matches) while the same rows scored on `gold_verdict` give 0.1096 and 0.0. Two columns
#: of different shapes in one row is exactly how a zero gets read as a result. On
#: `shadow-gold-sample` the two columns hold the same labels (arm 1's `gold_verdict` is a
#: copy of its `gold`), so the trap is that one surface's `gold`, not both columns
#: everywhere — and it is still the trap that costs a routing call.
SCORED_GOLD_COLUMN = "gold_verdict"

#: Whose rows kappa is computed over, spelled out in every cell. Deliberately the pair
#: set, not "every row this engine answered": on the 2026-10-17 shadow artifact djev
#: answered all 120 rows and the primary answered 116, so a per-engine denominator would
#: print djev 0.7881 over 120 against primary 0.9125 over 116 — two numbers over two
#: different row sets, which is not a comparison. Over the shared 116 the same artifact
#: gives 0.7881 vs 0.9125, and `engine_error_rate` already uses this denominator.
KAPPA_BASIS = "non-abstaining pairs (both engines returned an argmax)"

#: The degeneracy this rail exists to catch, named in the table and in the record.
DEGENERATE_LABEL = "degenerate: constant-label arm"

#: Per-surface caveats printed ON the kappa cell, not merely filed in `meta` — because
#: the cell is the thing a routing call would be quoted from. Arm 1's gold is authored by
#: Lloyd under its own rubric, and its README says in as many words that it does not
#: license a promotion; a kappa of 0.79 against a key the system wrote for itself is
#: evidence about consistency, not about correctness.
KAPPA_GOLD_CAVEATS: dict[str, str] = {}  # filled after SURFACE_* are bound below

#: What a row needs to be ASKED but must not appear in the published record: the
#: prompt itself (an artifact of ~3K chars per row, and the corpus already carries it
#: under its own sha), the menu the prompt was built from, the scorer, and the menu
#: sizes the scorer needs.
TRANSPORT_ONLY = frozenset({"sat", "prompt", "entity_menu", "doc_menu",
                            "n_entities", "n_docs"})

Transport = Callable[[str, str, str], str]   # (url, model, prompt) -> reply text
Post = Callable[[str, dict], dict]            # (url, payload) -> decoded JSON

#: Parameters an endpoint may refuse, and which this module then drops and retries
#: ONCE. The seven sampling names are the shipped
#: `label_agreement_ceiling._DIFFUSION_REJECTED_SAMPLING` list — djev on this box is a
#: diffusion build that answers ANY `temperature` with HTTP 400 and, copied from the
#: live reply of 2026-09-28, "The temperature, min_p, seed, min_tokens, logit_bias,
#: bad_words, and allowed_token_ids sampling parameters are not yet supported with
#: diffusion models." Importing that tuple rather than re-typing it is the point: a
#: forked list would drift, and the drift would look like an engine that abstains on
#: every row. `chat_template_kwargs` is added because the shipped guard was written
#: for a labeler that never sent it, while #2254's primary pass had to send
#: `enable_thinking: False` to stop 6 of 200 replies arriving truncated with no
#: verdict at all.
DROPPABLE = ("temperature", "min_p", "seed", "min_tokens", "logit_bias", "bad_words",
             "allowed_token_ids", "chat_template_kwargs")

#: Provenance record: which engine had which parameter dropped this run, so the
#: artifact can name a number produced without the sampling parameter its own docstring
#: promises. An artifact that recorded `temperature 0` for a request that could not send
#: it would be describing a call that was never made.
PARAM_DROPS: list[dict] = []


def _refused_params(exc: BaseException, payload: dict) -> list[str]:
    """Which `DROPPABLE` parameters THIS 400 says THIS payload may not carry.

    The three conditions of the shipped `_rejected_sampling_params`, kept whole:
    the status must be 400 (a refusal to sample is a bad request, not an auth or a
    missing model); the body must NAME the parameter in words; and the payload must
    actually have sent it. A 400 for an unknown model or a malformed body names none
    of them and propagates untouched, so the retry can never mask a request bug.
    """
    if getattr(exc, "code", None) != 400:
        return []
    try:
        body = str(exc.read().decode("utf-8", "replace")).lower()   # type: ignore[attr-defined]
    except Exception:
        return []
    return [name for name in DROPPABLE if name in payload and name in body]


def ask(url: str, model: str, prompt: str, *, timeout: float = 180.0,
        post: Post | None = None) -> str:
    """One typed-question call. The only place in this module that opens a socket.

    `post` is the HTTP seam — the shipped `label_agreement_ceiling.http_post_json`,
    swappable so a test exercises the real payload assembly, the real decode and the
    real refusal-to-send, rather than replacing this function whole. Retired ports are
    refused here and not in a caller, because a caller-side check would protect one arm
    and leave the other open.
    """
    if any(p in url for p in RETIRED_PORTS):
        raise ValueError(f"refusing to query retired endpoint {url}")
    send = post or _http_post
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 1200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        data = send(url, payload)
    except Exception as exc:
        dropped = _refused_params(exc, payload)
        if not dropped:
            raise
        for name in dropped:
            payload.pop(name, None)
            PARAM_DROPS.append({"url": url, "model": model, "dropped": name})
        print(f"dropped {dropped} for {url} (endpoint refused them)", file=sys.stderr)
        data = send(url, payload)
    choice = (data.get("choices") or [{}])[0]
    return choice.get("message", {}).get("content") or choice.get("text") or ""


def _http_post(url: str, payload: dict, *, timeout: float = 180.0) -> dict:
    """POST `payload` and decode the reply — via the shipped helper, so the socket
    behaviour (headers, decode, timeout plumbing) is one implementation, not two."""
    from eval import label_agreement_ceiling as L

    return L.http_post_json(url, payload, timeout=timeout)


def argmax_verdict(draws: list[str | None], labels: tuple[str, ...]) -> dict:
    """Collapse `repeats` parsed draws into one verdict per engine.

    Returns `{label, pass_rate, draws, answered, abstain}`.

    `label` is the plurality over the draws that produced a usable label. Two things
    make an engine ABSTAIN rather than guess, and both are load-bearing:

      * no draw produced a usable label (every reply unparsable or off-vocabulary);
      * the draws TIED.

    A tie has no argmax, and `architecture/tools.md:475` is explicit that djev's
    scores are not calibrated — only the argmax label is admissible evidence, so a
    tie is not a weak yes, it is no answer. Counting either of these as an agreement
    would manufacture exactly the false positive this instrument exists to catch: an
    engine that never answered cannot agree with anything.

    `pass_rate` is over ALL draws (abstentions dilute it), which is what "asked 5
    times, agreed 4" means to a reader.
    """
    valid = [d for d in draws if d in labels]
    total = len(draws)
    if not valid:
        return {"label": None, "pass_rate": None, "draws": total,
                "answered": 0, "abstain": True}
    counts = Counter(valid)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return {"label": None, "pass_rate": round(top[0][1] / total, 4), "draws": total,
                "answered": len(valid), "abstain": True, "tie": True}
    label, hits = top[0]
    return {"label": label, "pass_rate": round(hits / total, 4) if total else None,
            "draws": total, "answered": len(valid), "abstain": False}


# ── typed questions, per surface ────────────────────────────────────────────

def retrieval_rows(artifact: dict) -> list[dict]:
    """Rows the shipped djev artifact reports as OFFERED, one per (query, gold label).

    Gold comes from the artifact's own copy of the corpus (`primary_entities` /
    `primary_docs`), and so does the row set: `entity_candidates` / `doc_candidates`
    are the menus the labeler was shown, so a gold label outside a menu contributes
    no row at all. No new label is authored, and `primary` is asked no question
    djev was not also asked.

    `djev_recorded` is that artifact's verdict for the same label, carried so a reader
    can see a re-ask moving against the shipped reading.
    """
    from eval import label_agreement_ceiling as L

    rows: list[dict] = []
    for q in artifact.get("queries") or []:
        ent_menu = list(q.get("entity_candidates") or [])
        doc_menu = list(q.get("doc_candidates") or [])
        for leg, gold_key, menu, second_key, sat in (
                ("entity", "primary_entities", ent_menu, "second_entities",
                 L.entity_label_satisfied),
                ("doc", "primary_docs", doc_menu, "second_docs", L.doc_label_satisfied)):
            chosen = list(q.get(second_key) or [])
            for gold in q.get(gold_key) or []:
                if not any(sat(gold, [c]) for c in menu):
                    continue          # never offered: not a question, not a row
                rows.append({
                    "surface": SURFACE_RETRIEVAL,
                    "id": f"{q['id']}:{leg}:{gold}",
                    "query_id": q["id"], "leg": leg, "gold": gold,
                    "gold_verdict": "relevant",
                    "labels": ("relevant", "not_relevant"),
                    "prompt": L.build_prompt(q["query"], ent_menu, doc_menu),
                    "n_entities": len(ent_menu), "n_docs": len(doc_menu),
                    "entity_menu": ent_menu, "doc_menu": doc_menu,
                    "sat": sat,
                    "djev_recorded": "relevant" if sat(gold, chosen) else "not_relevant",
                    "offered": True,
                })
    return rows


#: The one typed question this arm asks, in the answer shape
#: `durable_write_judge/judge.py::parse_verdict` already parses — one JSON object, so
#: the argmax is a read of that object and nothing else. The whole as-is note goes in:
#: this corpus caps a row at 4,030 characters, so nothing here is windowed, and a
#: question shown a window could not condemn the defect class the gold names.
DURABLE_QUESTION = """You are checking one note before it is saved durably to a knowledge vault. \
Read it as-is and decide whether it is safe to save.

Answer with one JSON object only, no prose:
  {{"verdict": "accept"}}   if the note is safe to save as it stands
  {{"verdict": "flag"}}     if it must be held back and reviewed first

Hold it back if any of these is in the text itself:
  - a URL or citation with no source shown for it
  - a number, quota or limit stated as fact with no basis in the note
  - a section or field the note says it contains but does not
  - metadata that contradicts the text it describes
  - a certainty about a model's knowledge cutoff the note cannot support

Note as-is:
{note}

Verdict (JSON object only):"""


def durable_rows(corpus: list[dict], recorded: list[dict]) -> list[dict]:
    """The 50 corpus rows joined by sample id to the 50 `primary` rows #2254 already has.

    Gold is the corpus's own `label`, so this arm needs no new labelling, and what the
    join brings across is `primary`'s recorded verdict plus that run's `prompt_sha256`
    and endpoint — provenance, published beside the new draws so a reader can see
    whether the re-ask agrees with the verdict #2254 recorded for the same sample.
    Nothing is requested of an endpoint to get the recorded half: it is read from a
    file, which is the second reason :8091 is unreachable from this arm (the first is
    that `ask()` refuses the port and `resolve_engines` never looks it up). #2254's own
    runner refuses its join on that endpoint for the same reason — `run_arms.py:31`.

    A row that cannot be joined stops the arm rather than being skipped: 50 corpus rows
    measured against 47 recorded verdicts is a denominator nobody chose, and the whole
    point of this surface is a table whose denominators a reader can reproduce.

    A bad row's text is `pre_repair_text` — the note carrying its defect, which is the
    thing worth asking about; a good row's only text is `accepted_text`.
    """
    by_sample = {}
    for r in recorded:
        if r.get("model") == "primary" and r.get("arm") == "none":
            by_sample.setdefault(r["sample_id"], r)
    missing = [s["id"] for s in corpus if s["id"] not in by_sample]
    if missing:
        raise RuntimeError(
            f"durable-write arm cannot join {len(missing)} of {len(corpus)} corpus rows "
            f"to a #2254 `primary` row (first: {missing[0]}); the arm reports the "
            "localization table beside that run's verdicts, so a partial join is a "
            "different measurement, not a smaller one.")

    rows: list[dict] = []
    for sample in corpus:
        sid = sample["id"]
        prior = by_sample[sid]
        note = (sample.get("pre_repair_text") or sample.get("accepted_text") or "")
        rows.append({
            "surface": SURFACE_DURABLE_WRITE,
            # Arm-native fields only. `leg` and `offered` are retrieval concepts — a
            # durable row has no menu and nothing was ever "offered" to it — and
            # publishing them here would put a retrieval shape on a row no retrieval
            # question was asked of.
            "id": sid,
            "gold": sample["label"],
            "gold_verdict": "flag" if sample["label"] == "bad" else "accept",
            "labels": ("flag", "accept"),
            "prompt": DURABLE_QUESTION.format(note=note),
            # #2254's own record of what `primary` answered, and of the question it was
            # asked, kept beside the new draws. It is provenance, not a second reading
            # of this question: #2254 asked "would you durably save this file as-is?"
            # under a rubric and a seeded example draw; this arm asks the narrower
            # question above of both engines, so a disagreement between the two columns
            # is the two questions differing, and the artifact says which is which.
            "recorded_primary_verdict": prior.get("verdict"),
            "recorded_prompt_sha256": prior.get("prompt_sha256"),
            "recorded_url": prior.get("url"),
            "defect_classes": list(sample.get("defect_classes") or []),
        })
    return rows


def parse_retrieval(row: dict, reply: str) -> str | None:
    """Did the engine's menu pick include the gold label? None = no usable answer."""
    from eval import label_agreement_ceiling as L

    picks = L.parse_reply(reply, n_entities=row["n_entities"], n_docs=row["n_docs"])
    if picks.get("parse_failed"):
        return None
    idx = picks["entities"] if row["leg"] == "entity" else picks["docs"]
    menu = row["entity_menu"] if row["leg"] == "entity" else row["doc_menu"]
    chosen = [menu[i] for i in idx]
    return "relevant" if row["sat"](row["gold"], chosen) else "not_relevant"


def parse_durable(row: dict, reply: str) -> str | None:
    from eval.durable_write_judge import judge as J

    verdict = J.parse_verdict(reply).get("verdict")
    return verdict if verdict in ("flag", "accept") else None


# ── arm 1: the labelled shadow gold sample (#2824) ──────────────────────────

#: The options are matched as whole words, and anything that is not exactly one of
#: them is an abstain. Loose matching is what turned the retrieval arm's 2000-row
#: first run into a fake 0.0095: replies that echoed the question's own words scored as
#: a pick there, and here a reply like "these are not the same, they are different"
#: names two options at once, which is a non-answer rather than a coin flip.
def parse_arm1(row: dict, reply: str) -> str | None:
    import re as _re

    text = str(reply or "").lower()
    hits = [label for label in row["labels"]
            if _re.search(rf"\b{_re.escape(str(label).lower())}\b", text)]
    return hits[0] if len(hits) == 1 else None


def load_arm1_gold(path: Path) -> list[dict]:
    """Read and validate arm 1's gold, and refuse the file if a row fails.

    Validation is the sampler's own (`rubric.validate_gold_row`), called from here as
    well as there, because this is the path that turns a row into a denominator. A
    hand-edited row with no warrant, or a row whose rubric line has since been edited,
    would otherwise produce a number that looks exactly like a valid one.
    """
    from eval.cross_engine_disagreement_arm1 import rubric

    gold = rubric.load_gold(path)
    if not gold:
        raise RuntimeError(
            f"arm 1 has no gold rows at {path}; the surface reports UNMEASURABLE "
            "rather than a rate computed over nothing")
    return gold


#: Arm 1's question, assembled from the parts the seam itself sent. It is assembled
#: rather than stored in `gold.jsonl` so the words an endpoint sees are produced by one
#: function in this repo, readable by the clause-3 tests, and so the gold file stays the
#: labelling artifact rather than also being a prompt file. `state` is the canvas text
#: the seam put to djev, verified byte-for-byte against the row's own `inputs_digest` by
#: the sampler before the row was ever written; `instructions` and `options` are the
#: frozen typed-question set from `eval/djev/schemas.py`.
def arm1_prompt(state: str, instructions: str, options: tuple[str, ...]) -> str:
    return (f"CANVAS STATE:\n{state}\n\nQUESTION:\n{instructions}\n\n"
            f"Answer with exactly one of: {', '.join(options)}\n")


def arm1_rows(gold: list[dict], *, limit: int | None = None) -> list[dict]:
    """Runner rows from validated gold rows — same shape, arm-native fields only."""
    rows: list[dict] = []
    for g in gold[:limit or len(gold)]:
        labels = tuple(g["options"])
        rows.append({
            "surface": SURFACE_ARM1,
            "id": f"{g['seam']}:{g['inputs_digest']}",
            "seam": g["seam"],
            "stratum": g["stratum"],
            "inputs_digest": g["inputs_digest"],
            "question_id": g["question_id"],
            "labels": labels,
            "gold": g["gold"],
            "gold_verdict": g["gold"],
            "warrant": g["warrant"],
            "labelled_by": g["labelled_by"],
            "rubric_sha256": g["rubric_sha256"],
            "prompt": arm1_prompt(g["state"], g["question"], labels),
        })
    return rows


SURFACE_RETRIEVAL = "retrieval-labels-offered"
SURFACE_DURABLE_WRITE = "durable-write-judge"
#: Arm 1 (#2824): the seeded, warrant-bearing gold sample of the djev shadow store.
#: Its labels are Lloyd-authored under a written rubric rather than lifted from an
#: artifact that already carried them, which is why its provenance block is the one
#: block that says `human_label: false`. What that makes the number evidence OF is
#: `eval/cross_engine_disagreement_arm1/README.md`'s question, not this file's.
SURFACE_ARM1 = "shadow-gold-sample"

PARSERS = {SURFACE_RETRIEVAL: parse_retrieval, SURFACE_DURABLE_WRITE: parse_durable,
           SURFACE_ARM1: parse_arm1}

#: Arm 1's caveat, bound where its surface name is. Quoted from
#: `eval/cross_engine_disagreement_arm1/README.md`, "What this arm does NOT measure, and
#: what it does not license": it does not license "reopening #708, flipping
#: `automod.review.confirm`, or any other promotion decision". The other two surfaces take
#: their gold from an artifact that already carried the label, so they carry no caveat:
#: this key is present only where a close kappa could be mistaken for a licence.
KAPPA_GOLD_CAVEATS[SURFACE_ARM1] = (
    "gold is Lloyd-authored under eval/cross_engine_disagreement_arm1/rubric.py, and that "
    "arm's README says its own gold 'does not license' reopening #708, flipping "
    "automod.review.confirm or any other promotion decision — a close kappa here measures "
    "two families agreeing with a key Lloyd wrote, not either engine being right")


# ── asking ──────────────────────────────────────────────────────────────────

def run_rows(rows: list[dict], transport: Transport, repeats: int,
             engines: dict[str, dict[str, str]]) -> list[dict]:
    """Ask every row `repeats` times of every engine and classify the pair.

    The pair class is computed only from usable argmaxes. `abstain` on a row means at
    least one engine gave none, and such a row is dropped from EVERY rate below — an
    abstention is never evidence of agreement, and a rate over rows where one engine
    did not answer is the "abstention masquerading as agreement" failure this item
    names first among its risks.
    """
    names = list(engines)
    if len(names) != 2:
        raise ValueError(f"a pair is exactly two engines, got {names}")
    out: list[dict] = []
    for row in rows:
        rec = {k: v for k, v in row.items() if k not in TRANSPORT_ONLY}
        rec["record_id"] = row["id"]
        rec["engines"] = {}
        parser = PARSERS[row["surface"]]
        for name in engines:
            eng = engines[name]
            draws = [parser(row, transport(eng["url"], eng["model"], row["prompt"]))
                     for _ in range(repeats)]
            rec["engines"][name] = argmax_verdict(draws, row["labels"])
        e1, e2 = rec["engines"][names[0]], rec["engines"][names[1]]
        gold = row["gold_verdict"]
        e1["wrong"] = (not e1["abstain"]) and e1["label"] != gold
        e2["wrong"] = (not e2["abstain"]) and e2["label"] != gold
        if e1["abstain"] or e2["abstain"]:
            rec["pair_class"] = ABSTAIN
            rec["row_error"] = None
            rec["both_wrong_agreeing"] = None
        else:
            same = e1["label"] == e2["label"]
            rec["pair_class"] = AGREE if same else DISAGREE
            rec["row_error"] = bool(e1["wrong"] or e2["wrong"])
            rec["both_wrong_agreeing"] = bool(same and e1["wrong"] and e2["wrong"])
        out.append(rec)
    return out


# ── the localization table ──────────────────────────────────────────────────

def _rate(k: int, n: int) -> dict:
    """One proportion with its Wilson CI, or nulls with a stated reason.

    `(0.0, 0.0)` for an empty denominator would read as a measured floor; `None` plus
    the denominator is the honest "this cell has no rate".
    """
    if n <= 0:
        return {"num": k, "den": 0, "value": None, "ci95": None,
                "reason": "denominator is 0"}
    lo, hi = wilson_ci(k, n)
    return {"num": k, "den": n, "value": round(k / n, 4),
            "ci95": [round(lo, 4), round(hi, 4)]}


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict:
    """Cohen's kappa for one set of ratings against another, over the same rows.

    `po` is observed agreement, `pe` is the agreement the two raters' OWN marginal label
    distributions predict by chance, and kappa is `(po - pe) / (1 - pe)`: 1.0 is perfect
    agreement, 0.0 is exactly what chance predicts from those marginals, and a negative
    value is worse than chance. It is the chance-corrected companion to an agreement rate,
    which is not corrected for anything — an engine that answers `flag` on every row of a
    corpus that is 40% `flag` agrees with gold on 40% of rows and prints that as a rate.

    Three shapes the caller must not have to remember:

      * `n == 0` → `kappa: None` with a reason. No ratings, no statistic.
      * `pe == 1` (both sides constant on the SAME single class) → `kappa: None`: 1 - pe
        is zero, and publishing 0.0 there would be arithmetic on a division that did not
        happen. `degenerate` is true of that row too, which is where its meaning is.
      * one side constant → kappa comes out 0.0 EXACTLY, by construction, whatever the
        data: a constant rater's `po` IS the prevalence of its class in the other side,
        which is precisely its `pe`. That identity is why a constant arm must never print
        a number — 0.000 would read as "this engine agrees with gold no better than a
        coin", when what it measured was an engine that never varied. Hence `degenerate`.

    Returns `{"n", "po", "pe", "kappa", "n_classes", "degenerate", "labels", "reason"}`,
    with `po`/`pe`/`kappa` rounded to 4 for publication and computed unrounded.
    """
    n = len(pairs)
    labels = Counter(a for a, _ in pairs)
    out: dict = {"n": n, "po": None, "pe": None, "kappa": None,
                 "n_classes": len(labels), "degenerate": n > 0 and len(labels) < 2,
                 "labels": dict(sorted(labels.items())), "reason": None}
    if n == 0:
        out["reason"] = "no scored pairs"
        return out
    po = sum(1 for a, b in pairs if a == b) / n
    other = Counter(b for _, b in pairs)
    pe = sum(labels[k] / n * other[k] / n for k in set(labels) | set(other))
    out["po"], out["pe"] = round(po, 4), round(pe, 4)
    if pe >= 1.0:
        out["reason"] = ("both sides constant on one class, so pe = 1 and 1 - pe = 0: "
                         "kappa has no denominator")
        return out
    out["kappa"] = round((po - pe) / (1.0 - pe), 4)
    if out["degenerate"]:
        (only, _), = out["labels"].items()
        out["reason"] = (f"{DEGENERATE_LABEL}: all {n} ratings were {only!r}, and a "
                         "constant arm scores kappa 0.000 by construction (its po IS its "
                         "pe), so the number measures the arm's constancy, not its "
                         "agreement with gold")
    return out


def _engine_kappa(pairs: list[dict], eng: str, surface: str,
                  measurable: bool, unmeasurable: str | None) -> dict:
    """One engine's kappa-against-gold cell, over the pairs where IT answered."""
    scored = [(r["engines"][eng]["label"], r[SCORED_GOLD_COLUMN]) for r in pairs
              if not r["engines"][eng]["abstain"]]
    k = cohen_kappa(scored)
    cell: dict = {"column": SCORED_GOLD_COLUMN, "basis": KAPPA_BASIS,
                  "surface": surface, "measurable": measurable,
                  "gold_caveat": KAPPA_GOLD_CAVEATS.get(surface)}
    if not measurable:
        # A thin cell prints the same stated reason the rates print, and kappa does not
        # get to compute a number over the rows the rest of the table refused. The counts
        # are still published — `n` is what a reader needs to see the cell is thin — but
        # po/pe/kappa are not, because `localize` already ruled this cell not a result.
        cell.update({"n": k["n"], "kappa": None, "uninformative": unmeasurable})
        return cell
    cell.update({k2: k[k2] for k2 in ("n", "po", "pe", "kappa", "n_classes",
                                      "degenerate", "labels", "reason")})
    if k["degenerate"] or k["kappa"] is None:
        cell["uninformative"] = k["reason"]
    return cell


def scored_gold_meta(cells: list[dict]) -> dict:
    """Per surface: WHICH column of the row is its gold, and what labels it holds.

    A function rather than a line inside `main`, so a test can assert the naming without
    two live engines — and so the claim is checkable against a fixture rather than only
    against a run. A reader holding an artifact and no source needs this: the durable
    rows carry `gold` = `bad`/`good` alongside the `gold_verdict` = `flag`/`accept` the run
    scores, and re-deriving a kappa against the wrong one yields 0.0000 for BOTH engines
    on the 2026-10-16 artifact (po 0.0 — no row matches), which is a figure a routing call
    could act on. Naming the column and its vocabulary here is what makes that misread
    visible instead of merely possible.
    """
    return {
        c["surface"]: {"column": c["gold_column"], "labels": c["gold_labels"],
                       "kappa_basis": KAPPA_BASIS}
        for c in cells}


def localize(rows: list[dict], surface: str) -> dict:
    """Every published figure for one surface, over non-abstaining rows only."""
    total = len(rows)
    # Which engines a cell reports is read from the rows themselves: the run resolved
    # them from config.yaml, and a hard-coded table here could disagree with the pair
    # that produced the numbers.
    engine_names = sorted({e for r in rows for e in r.get("engines", {})})
    abstained = [r for r in rows if r["pair_class"] == ABSTAIN]
    pairs = [r for r in rows if r["pair_class"] in (AGREE, DISAGREE)]
    n = len(pairs)
    agree = [r for r in pairs if r["pair_class"] == AGREE]
    disagree = [r for r in pairs if r["pair_class"] == DISAGREE]
    bwa = [r for r in agree if r["both_wrong_agreeing"]]
    errors = [r for r in pairs if r["row_error"]]
    errors_in_disagree = [r for r in disagree if r["row_error"]]

    cell = {
        "surface": surface,
        "rows": total,
        "non_abstaining": n,
        "abstained": len(abstained),
        "abstention_share": _rate(len(abstained), total),
        "agreement_rate": _rate(len(agree), n),
        "error_rate_inside_agreement": _rate(len(bwa), len(agree)),
        "disagreement_volume_share": _rate(len(disagree), n),
        "errors_total": len(errors),
        "errors_in_disagreement": len(errors_in_disagree),
        "error_localization_recall": _rate(len(errors_in_disagree), len(errors)),
        "both_wrong_and_agreeing": {
            "count": len(bwa),
            "share_of_all_errors": _rate(len(bwa), len(errors)),
            "share_of_rows": _rate(len(bwa), n),
        },
        # Each-engine-vs-gold, the second half of "is the pair worth anything".
        "engine_error_rate": {
            eng: _rate(sum(1 for r in pairs if r["engines"][eng]["wrong"]),
                       sum(1 for r in pairs if not r["engines"][eng]["abstain"]))
            for eng in engine_names
        },
        "engine_abstain_rate": {
            eng: _rate(sum(1 for r in rows if r["engines"][eng]["abstain"]), total)
            for eng in engine_names
        },
        "per_case_pass_rate": {
            eng: _pass_rate_summary(rows, eng) for eng in engine_names
        },
        "measurable": n >= MIN_NON_ABSTAINING,
        # Which column of the row this surface's gold actually is, and the vocabulary it
        # holds. Published so a kappa can never be silently re-derived against the other
        # column the durable rows also carry (`gold` = bad/good, a different shape).
        "gold_column": SCORED_GOLD_COLUMN,
        "gold_labels": sorted({r[SCORED_GOLD_COLUMN] for r in rows}),
        # How many rows in the cell carry a verdict a SHIPPED subsystem recorded. The
        # two original surfaces have one per row; arm 1 has none, because no shipped
        # reviewer ever put these canvases to a model. It does not empty the recall
        # denominator — `row_error` is "an engine disagreed with the gold" — but it
        # changes what the number means, so it is published rather than implied.
        "rows_with_shipped_answer": sum(
            1 for r in rows if r.get("primary_verdict") is not None),
    }
    if not cell["measurable"]:
        cell["unmeasurable"] = (f"UNMEASURABLE surface={surface} "
                                f"non_abstaining={n}")
        cell["note"] = (f"{n} non-abstaining pairs is below the {MIN_NON_ABSTAINING} "
                        "this item sets; the rates above are computed and published "
                        "for the record but are not a result, and this cell must not "
                        "be quoted as one.")
    if cell["measurable"] and not cell["errors_total"]:
        cell["localization_unmeasurable"] = (
            f"UNMEASURABLE surface={surface} localization: none of the "
            f"{n} non-abstaining pairs contains an error, so recall has no "
            f"denominator and no rate is printed")
    # Chance-corrected agreement with gold, per engine. Built after `measurable` and the
    # `unmeasurable` sentence exist, because a kappa a thin cell has no business printing
    # has to print the SAME reason the rates print rather than a number over n < 30.
    cell["engine_kappa"] = {
        eng: _engine_kappa(pairs, eng, surface, cell["measurable"],
                           cell.get("unmeasurable"))
        for eng in engine_names}
    cell["positive_gate"] = positive_for_surface(cell)
    return cell


def _pass_rate_summary(rows: list[dict], eng: str) -> dict:
    """The distribution of per-case pass rates, since one draw is not a measurement.

    `repeats` draws per row per engine; the mean is over rows where the engine
    answered at least once. A 1.0 mean with 5 repeats says the engine is
    self-consistent, which is NOT the same claim as being right — correctness is
    `engine_error_rate`, beside it.
    """
    vals = [r["engines"][eng]["pass_rate"] for r in rows
            if r["engines"][eng]["pass_rate"] is not None]
    if not vals:
        return {"rows_with_an_answer": 0, "mean": None, "min": None, "max": None}
    return {"rows_with_an_answer": len(vals), "mean": round(statistics.fmean(vals), 4),
            "min": min(vals), "max": max(vals),
            "all_5_agree": sum(1 for v in vals if v == 1.0)}


def positive_for_surface(cell: dict) -> dict:
    """The item's POSITIVE test, applied to one surface.

    Three conditions, all required, on a MEASURABLE surface: the disagreement set
    holds >=70% of the errors, covers <=35% of the volume, and both-wrong-and-agreeing
    is <10% of all errors. The third is the independence ceiling — a surface where the
    two engines are wrong together 40% of the time has a shared prior, not a second
    reader, and recall on the remaining errors would be a flattering number over the
    errors that happen to be visible.
    """
    reasons: list[str] = []
    if not cell["measurable"]:
        reasons.append(f"non_abstaining={cell['non_abstaining']} < {MIN_NON_ABSTAINING}")
        return {"positive": False, "reasons": reasons}
    recall = cell["error_localization_recall"]["value"]
    volume = cell["disagreement_volume_share"]["value"]
    bwa = cell["both_wrong_and_agreeing"]["share_of_all_errors"]["value"]
    if recall is None or cell["errors_total"] == 0:
        reasons.append("no errors observed, so localization is undefined")
    else:
        if recall < POSITIVE_MIN_RECALL:
            reasons.append(f"recall {recall} < {POSITIVE_MIN_RECALL}")
        if volume is None or volume > POSITIVE_MAX_VOLUME_SHARE:
            reasons.append(f"disagreement volume {volume} > {POSITIVE_MAX_VOLUME_SHARE}")
        if bwa is None or bwa >= POSITIVE_MAX_BWA_SHARE:
            reasons.append(f"both-wrong-and-agreeing {bwa} >= {POSITIVE_MAX_BWA_SHARE}")
    return {"positive": not reasons, "reasons": reasons}


def overall_verdict(cells: list[dict]) -> str:
    """POSITIVE only if SOME surface passes all three; NEGATIVE is a result too.

    The gate is re-derived from each cell's readings rather than read off the
    `positive_gate` field `localize` stored, so a cell assembled from a published
    artifact answers the same question as one just measured.
    """
    return "POSITIVE" if any(positive_for_surface(c)["positive"] for c in cells) \
        else "NEGATIVE"


# ── publication ─────────────────────────────────────────────────────────────

def markdown(cells: list[dict], meta: dict) -> str:
    lines = [
        "# Cross-engine disagreement as a verdict-error localizer — backlog #2639",
        "",
        f"Ran {meta['ran_at']} · repeats/engine = {meta['repeats']} · engines: "
        + ", ".join(f"{k} ({v['url']}, served {v['model']})"
                    for k, v in meta["engines"].items()),
        "",
        f"**Verdict: {meta['verdict']}.** POSITIVE requires a surface where the "
        f"disagreement set holds >={POSITIVE_MIN_RECALL:.0%} of errors at "
        f"<={POSITIVE_MAX_VOLUME_SHARE:.0%} of volume with both-wrong-and-agreeing "
        f"<{POSITIVE_MAX_BWA_SHARE:.0%} of errors. Every rate below is over "
        "non-abstaining pairs only; the abstention share is printed beside it, never "
        "inside it.",
        "",
        "| surface | rows | non-abst | abstained | agreement | error-in-agreement "
        "| disagree volume | localization recall | BOTH wrong+agree | errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        if not c["measurable"]:
            lines.append(f"| {c['surface']} | {c['rows']} | {c['non_abstaining']} "
                         f"| {c['abstained']} | **{c['unmeasurable']}** | — | — | — | — | "
                         f"{c['errors_total']} |")
            continue

        def f(key: str) -> str:
            r = c[key]
            return "—" if r["value"] is None else (
                f"{r['value']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
                f"({r['num']}/{r['den']})")
        bwa = c["both_wrong_and_agreeing"]
        lines.append(
            f"| {c['surface']} | {c['rows']} | {c['non_abstaining']} | {c['abstained']} "
            f"| {f('agreement_rate')} | {f('error_rate_inside_agreement')} "
            f"| {f('disagreement_volume_share')} | {f('error_localization_recall')} "
            f"| {bwa['count']} ({bwa['share_of_all_errors']['value'] if bwa['share_of_all_errors']['value'] is not None else '—'} "
            f"of errors) | {c['errors_total']} |")
    lines += ["", "## Per engine against gold, and self-consistency", ""]
    for c in cells:
        lines.append(f"- **{c['surface']}** — " + "; ".join(
            f"{eng}: error rate {_fmt(c['engine_error_rate'][eng])}, abstains "
            f"{_fmt(c['engine_abstain_rate'][eng])}, per-case pass rate mean "
            f"{(c['per_case_pass_rate'][eng] or {}).get('mean')}"
            f" (min {(c['per_case_pass_rate'][eng] or {}).get('min')}, "
            f"max {(c['per_case_pass_rate'][eng] or {}).get('max')}), "
            f"κ vs {_k['column']} {_fmt_kappa(_k)}"
            for eng, _k in ((e, c["engine_kappa"][e]) for e in meta["engines"])))
        for _k in (c["engine_kappa"][e] for e in meta["engines"]):
            if _k.get("gold_caveat"):
                lines.append(f"  - κ caveat: {_k['gold_caveat']}")
                break
        lines.append(f"  - κ basis: {KAPPA_BASIS} = {c['non_abstaining']} of "
                     f"{c['rows']} rows; scored gold column "
                     f"{c['gold_column']} = {'/'.join(c['gold_labels']) or '—'}")
        lines.append(f"  - POSITIVE here: {c['positive_gate']['positive']} — "
                     + ("; ".join(c['positive_gate']['reasons']) or "all three hold"))
        if c.get("localization_unmeasurable"):
            lines.append(f"  - {c['localization_unmeasurable']}")
    lines += ["", "Read `both_wrong_and_agreeing` before anything else. It is the "
                  "shared-convention case — four model families once produced the same "
                  "wrong total to the cent because they shared a sign convention the "
                  "answer key did not — and a high recall computed over the remaining "
                  "errors flatters a pair that agrees on being wrong.", "",
              "This artifact wires nothing: no gate, rung or dispatch path reads it. "
              "Whether a POSITIVE number licenses reopening #708 or the shipped-but-off "
              "`automod.review.confirm` switch is a ruling this run does not make. "
              "Kappa against gold is published for the same reason and with the same "
              "limit: it says how much of an engine's agreement with the key survives "
              "chance, and it says nothing at all about an arm that emitted one label — "
              "least of all does it say whether a cheaper engine may take a decision off "
              "the expensive one, which is the only question a routing bar answers.", ""]
    return "\n".join(lines)


def _fmt(r: dict) -> str:
    return "—" if r["value"] is None else (
        f"{r['value']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] ({r['num']}/{r['den']})")


def _stdout_kappa(k: dict) -> str:
    """The κ for the runner's one-line summary: a figure to 4 dp, or the refusal word.

    stdout carries no room for the table's prose, so the two refusals are shortened rather
    than dropped — and the shortening is deliberately not a number. Printing `0.0` for the
    constant-label arm here would put back, on the line a person copies out of a terminal,
    the figure clause 2 refuses to publish in the table.
    """
    if not k["measurable"]:
        return "UNMEASURABLE"
    if k.get("uninformative"):
        return "UNINFORMATIVE-degenerate"
    return f"{k['kappa']:.4f}"


def _fmt_kappa(k: dict) -> str:
    """One kappa cell as published text: a figure, or the reason there is no figure.

    Two of the three states are refusals to print a number, and each says why in its own
    words. A thin cell repeats the `UNMEASURABLE surface=… non_abstaining=…` sentence the
    rates already print, so kappa cannot quietly report a statistic over rows the rest of
    the table ruled out. A degenerate arm prints `UNINFORMATIVE` plus the reason and its
    marginals — po beside pe is what shows the collapse (0.400 against 0.400 on the
    2026-10-16 durable arm), and a reader who wants the arithmetic can still do it, but the
    slot where a routing call would read a number holds prose instead.
    """
    if not k["measurable"]:
        return k["uninformative"]
    if k.get("uninformative"):
        # The margins are printed precisely BECAUSE there is no figure: po sitting on top
        # of pe (0.400 against 0.400 on the 2026-10-16 durable arm) is the visible form of
        # the collapse, and it is the only evidence in the line that the refusal is right.
        # po is absent only when there were no ratings to take a mean over, which a
        # measurable cell cannot reach (its floor is MIN_NON_ABSTAINING pairs) but which
        # must not turn the refusal into a traceback.
        margin = ("" if k["po"] is None else
                  f" (po {k['po']:.3f}, pe {k['pe']:.3f}, n={k['n']})")
        return f"UNINFORMATIVE — {k['uninformative']}{margin}"
    return (f"{k['kappa']:.3f} (po {k['po']:.3f}, pe {k['pe']:.3f}, "
            f"n={k['n']}, {len(k['labels'])} label(s) emitted)")


def write_artifact(out_dir: Path, meta: dict, cells: list[dict],
                   rows: list[dict]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta["ran_at"][:19].replace(":", "").replace("+00:00", "").replace("T", "-")
    jp = out_dir / f"cross-engine-disagreement-{stamp}.json"
    mp = out_dir / f"cross-engine-disagreement-{stamp}.md"
    jp.write_text(json.dumps({"meta": meta, "surfaces": cells, "rows": rows},
                             indent=1, sort_keys=True))
    mp.write_text(markdown(cells, meta))
    return {"json": jp, "markdown": mp}


# ── arms ────────────────────────────────────────────────────────────────────

def resolve_engines(fetcher: Callable[[str], dict] | None = None
                    ) -> dict[str, dict[str, str]]:
    """(name -> {url, model}) for `ENGINE_NAMES`, through the shipped resolver.

    `fetcher` injects the `/v1/models` GET, so a test can assert WHICH engines this run
    looks up — including that `secondary` is never looked up, which is how the :8091
    rule holds at the resolution step and not only at the POST.
    """
    from eval import label_agreement_ceiling as L

    out: dict[str, dict[str, str]] = {}
    for name in ENGINE_NAMES:
        url, model = L.resolve_engine(name, models_fetcher=fetcher)
        if any(p in url for p in RETIRED_PORTS):
            raise SystemExit(f"{name} resolves to retired endpoint {url}; refusing")
        out[name] = {"url": url, "model": model}
    return out


def recompute(path: Path) -> dict[str, dict]:
    """Recompute every published cell from the artifact's own `rows`, engine offline.

    The acceptance check for this item is that a reader can reproduce the per-cell
    denominators from the artifact alone, so the numbers in `surfaces` are not the
    run's private state: they are a pure function of the stored per-row verdicts. Any
    drift between the two is a bug in one of them, and `tests/` asserts they agree.
    """
    art = json.loads(Path(path).read_text())
    by_surface: dict[str, list[dict]] = {}
    for row in art["rows"]:
        by_surface.setdefault(row["surface"], []).append(row)
    return {s: localize(rs, s) for s, rs in by_surface.items()}


def load_retrieval_artifact(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_arms_final(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_corpus(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def build_rows(artifact_path: Path, corpus_path: Path, arms_path: Path,
               arms: tuple[str, ...], limit: int | None,
               arm1_gold_path: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    if SURFACE_RETRIEVAL in arms:
        rows += retrieval_rows(load_retrieval_artifact(artifact_path))
    if SURFACE_DURABLE_WRITE in arms:
        rows += durable_rows(load_corpus(corpus_path), load_arms_final(arms_path))
    if SURFACE_ARM1 in arms:
        gold_path = Path(arm1_gold_path) if arm1_gold_path else DEFAULT_ARM1_GOLD
        rows += arm1_rows(load_arm1_gold(gold_path))
    if limit:
        rows = rows[:limit]
    return rows


def main(argv: list[str] | None = None, transport: Transport | None = None,
         engines: dict[str, dict[str, str]] | None = None) -> int:
    """Run the measurement. `transport` and `engines` are the two seams.

    They are parameters rather than monkeypatch targets for the same reason
    #2254's runner takes `call=engine_call` (`run_arms.py:214`): a test that passes its
    own transport and its own pair has exercised the real routing, the real prompt
    assembly and the real arithmetic, while a patched-over function proves only that the
    patch took. With `engines` unset the pair is resolved from config.yaml and the live
    `/v1/models`, which is what the shipped run does and what a test must not do.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="directory for the dated JSON + markdown table")
    ap.add_argument("--repeats", type=int, default=MIN_REPEATS,
                    help=f"draws per row per engine (min {MIN_REPEATS})")
    ap.add_argument("--arms", default=",".join((SURFACE_RETRIEVAL, SURFACE_DURABLE_WRITE,
                                                SURFACE_ARM1)))
    ap.add_argument("--gold", default=str(DEFAULT_ARM1_GOLD),
                    help="arm 1's gold JSONL (eval/cross_engine_disagreement_arm1/"
                         "gold.jsonl by default, the file the sampler writes)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows, for a smoke run; the published number is unlimited")
    ap.add_argument("--label-artifact", default=str(DEFAULT_RETRIEVAL_ARTIFACT))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--arms-final", default=str(DEFAULT_ARMS_FINAL))
    args = ap.parse_args(argv)

    if args.repeats < MIN_REPEATS:
        ap.error(f"--repeats must be >= {MIN_REPEATS}: one draw per row is an "
                 "anecdote, and the item's acceptance names 5 explicitly")
    ask_ = transport or ask
    chosen = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = [a for a in chosen if a not in PARSERS]
    if unknown:
        ap.error(f"unknown arm(s) {unknown}; known: {sorted(PARSERS)}")

    engines = engines or resolve_engines()
    rows = build_rows(Path(args.label_artifact), Path(args.corpus),
                      Path(args.arms_final), chosen, args.limit,
                      arm1_gold_path=Path(args.gold))
    asked = run_rows(rows, ask_, args.repeats, engines)

    meta = {
        "schema": 1,
        "kind": "cross_engine_disagreement",
        "ran_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repeats": args.repeats,
        "engines": engines,
        "params_dropped": list(PARAM_DROPS),
        "retired_ports_refused": list(RETIRED_PORTS),
        "thresholds": {"min_repeats": MIN_REPEATS,
                       "min_non_abstaining": MIN_NON_ABSTAINING,
                       "positive_min_recall": POSITIVE_MIN_RECALL,
                       "positive_max_volume_share": POSITIVE_MAX_VOLUME_SHARE,
                       "positive_max_bwa_share": POSITIVE_MAX_BWA_SHARE},
        "gold_provenance": {
            SURFACE_RETRIEVAL: f"{args.label_artifact} (offered labels only)",
            SURFACE_DURABLE_WRITE: f"{args.corpus} joined to {args.arms_final} "
                                   "(primary, arm=none, prompt sha256 verified)",
            SURFACE_ARM1: f"{args.gold} — labels authored by Lloyd under the written "
                          "rubric in eval/cross_engine_disagreement_arm1/rubric.py; "
                          "each row names its rubric line, the evidence read, and that "
                          "line's sha256. NOT a human label. No shipped subsystem "
                          "answered these canvases, so an error on this surface is "
                          "'an engine disagreed with the rubric gold', not 'the shipped "
                          "verdict was wrong' — do not carry its recall across "
                          "surfaces",
        },
        "wires_nothing": True,
    }
    cells = [localize([r for r in asked if r["surface"] == s], s) for s in chosen]
    meta["verdict"] = overall_verdict(cells)
    # Named per surface, derived from the cells rather than typed: the column the run
    # scored is a property of the rows it read, and `meta` is where a reader of the
    # artifact looks for what a number means. `labels` is the vocabulary, so re-scoring the
    # durable arm on its `bad`/`good` column cannot be done by accident or by omission.
    meta["scored_gold_column"] = scored_gold_meta(cells)
    paths = write_artifact(Path(args.out), meta, cells, asked)
    for cell in cells:
        # The table's refusal rule, applied to the line a re-run prints: a degenerate or
        # thin arm's κ reaches stdout as the word, never as `0.0`. stdout is what somebody
        # pastes into a message when they ask "what did that run measure", and a bare 0.0
        # there is exactly the number clause 2 refuses to publish in the table.
        kappa_txt = ", ".join(f"{e}: {_stdout_kappa(k)}"
                              for e, k in cell["engine_kappa"].items())
        print(cell.get("unmeasurable") or cell.get("localization_unmeasurable")
              or (f"{cell['surface']}: non_abstaining={cell['non_abstaining']} "
                  f"agreement={_rate(cell['agreement_rate']['num'], cell['agreement_rate']['den'])['value']} "
                  f"recall={cell['error_localization_recall']['value']} "
                  f"both_wrong_and_agreeing={cell['both_wrong_and_agreeing']['count']} "
                  f"kappa_vs_{cell['gold_column']}={{{kappa_txt}}}"))
    print(f"VERDICT {meta['verdict']}")
    print(f"json {paths['json']}\nmarkdown {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
