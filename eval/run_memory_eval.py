#!/usr/bin/env python3
"""LloydMemEval: does Lloyd remember what Alan told him — and act on it? (#1480)

A frozen, versioned set of conversational-memory questions in LongMemEval's
five categories (single_session, multi_session, knowledge_update, temporal,
preference), each grounded in a real fact record and carried by synthetic,
dated past sessions (`eval/memory_eval_build.py` makes the set). This runner
asks the primary each question under one or more MEMORY ARMS and scores the
final answer:

  closed_book   the question alone — the floor, and the leak check (a question
                answerable with no memory measures nothing about memory)
  history       the question's evidence sessions plus distractor sessions from
                other episodes, dated, in the system prompt — long-context
                memory: can the model read, order and reconcile what it was told
  prefetch      Lloyd's real per-turn injection (`prefetch.prefetch_context`)
                over a snapshot of the live fact tree and the live qmd daemon,
                no history — can Lloyd's memory STACK surface the fact
  prefetch_rel  as `prefetch`, with the `<facts>` block ordered by query
                relevance instead of confidence (#1482 rider 1); every other
                byte of the block is the `prefetch` arm's
  prefetch_rawspan
                the `prefetch_rel` arm's facts, rendered as the source text they
                were extracted from instead of the distilled fact line (#1556).
                Selection is not a second lookup: the arm takes the relevance
                arm's own records off one `_search_fact_records` call, so the
                entities, their order and the count are the same by
                construction, and only the rendering differs. Each fact's span
                comes from its own `source_doc` under one windowing rule
                (`rawspan_window`), and every facts block here — this arm's and
                both distilled ones — is bounded by the one char budget
                `FACTS_RENDER_CHAR_BUDGET`, because with selection held fixed
                the equalised quantity has to be bytes and not facts. Facts
                whose `source_doc` does not resolve, and facts whose windowing
                rule finds no span, are counted and reported beside the arm's
                scores (`prefetch_rawspan.counts`), never dropped quietly. What
                the pair measures is representation, and neither reading is
                wired anywhere in this file: raw above distilled says the
                extractor is where the loss is, distilled at or above raw says
                the abstraction is buying its keep. The paired number is the
                bootstrap CI against `prefetch_rel`, and the retrieval half of
                it costs no model call (`prefetch-retrieval --arms
                prefetch_rel,prefetch_rawspan`)
  recall        the documents `vault_recall` returns for the question (top 10,
                path + snippet), no history — the tool Lloyd calls to look
                something up (#1485)
  recall_episodic  as `recall` with the episodic floor on (chat transcripts,
                qmd `sessions`, floor 1). Both need a qmd index whose
                `sessions` collection holds the set's dev sessions —
                `export-sessions --out DIR` writes them in the chat-export
                shape; against the live index they do not exist
  sleep_notes     #1516's next-session channel: the same confidence-ordered facts
                the `prefetch` arm renders, written through
                `app.next_session_notes` and drained back the way a morning turn
                drains it, so they arrive inside `<next-session-notes>` instead of
                `<facts>`. Same retrieval, different transport, position and cap —
                which is what makes it a measurement of the channel rather than of
                retrieval. The run gives the arm a store of its own in `--out-dir`;
                it never writes the live one. Whether the channel is kept is the
                paired-bootstrap CI against `prefetch` in `multi_session` and
                `knowledge_update`, over a full v1 run, and no line in this file
                presumes which way that comes out.

What is scored, per answer, and why three numbers rather than one:

  evidence_in_context  RETRIEVAL: were the gold values anywhere the model could
                       read them (history: by construction; prefetch: the block)
  mentioned            the answer names every gold value (rules, deterministic)
  correct              the answer ACTS on the gold value — the "use" judgement.
                       A mentioned value the answer then does not act on (it
                       restates the current port and curls the old one) is
                       `mixed` for the rules judge, settled by djev, and never
                       correct unless djev says it acted on the gold value.
                       `use_rate` = correct / mentioned is the conditional.

Judging is two-stage, deterministic first: the rules judge settles every answer
that names the gold values and no superseded value (`correct`), names only the
superseded value (`stale`), or abstains. Only `mixed` answers (both values) and
answers that name no gold value (a paraphrase?) go to djev (typed `choice`
decide on :8011). `correct_strict` never takes djev's word for a paraphrase;
`correct` does. Both are in the artifact, with the djev/rules agreement on the
answers the rules could settle. **`correct_strict` is the headline**: on the
2026-09-25 baseline djev's reading of a MIXED answer was right 12 of 12 in a
hand audit, and its paraphrase upgrades (`none` -> `gold`) 11 of 20 — so the
lenient number over-counts by roughly as much as strict under-counts
(eval/measurements/lloydmemeval-2026-09-25.md).

The generating model is recorded in the frozen set's manifest, and a judge
whose model is the generator is refused (`SelfJudgeRefused`) — #1471's rule in
code. The holdout slice (`<version>/holdout/`) is separated at load: the
tuning view never opens its files, and a holdout run reports aggregates only.

Usage (quality run: shared primary lock; the primary answers every question):
    flock -s ~/.local/state/lloyd-automod/primary.lock \
      .venvs/lloyd/bin/python eval/run_memory_eval.py run --arms closed_book,history,prefetch \
      --label baseline-2026-09-25 [--holdout]
    # #1516's deployment decision: the channel against today's prefetch arm
    flock -s ~/.local/state/lloyd-automod/primary.lock \\
      .venvs/lloyd/bin/python eval/run_memory_eval.py run --arms prefetch,sleep_notes \\
      --label sleep-notes-2026-09-26
    .venvs/lloyd/bin/python eval/run_memory_eval.py verify
    .venvs/lloyd/bin/python eval/run_memory_eval.py prefetch-retrieval   # no model: rider-1 retrieval A/B
    # #1556: representation with retrieval held fixed — the same facts, distilled
    # line vs their source text. Retrieval half, no model call:
    .venvs/lloyd/bin/python eval/run_memory_eval.py prefetch-retrieval \
      --arms prefetch_rel,prefetch_rawspan
    # modelled half, the way sleep_notes was judged: a paired-bootstrap CI
    flock -s ~/.local/state/lloyd-automod/primary.lock \\
      .venvs/lloyd/bin/python eval/run_memory_eval.py run --arms prefetch_rel,prefetch_rawspan \\
      --label rawspan-2026-09-27
    .venvs/lloyd/bin/python eval/run_memory_eval.py export-sessions --out DIR
    .venvs/lloyd/bin/python eval/run_memory_eval.py recall-retrieval [--out F]  # no model: #1485 A/B
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(HERE))

SET_ROOT = HERE / "memory_eval"
DEFAULT_VERSION = "v1"
CATEGORIES = ("single_session", "multi_session", "knowledge_update", "temporal", "preference")
REQUIRED_FIELDS = ("id", "category", "prompt", "asked_on", "probe", "accept", "source")
#: #1516: the next-session channel, measured against the per-turn prefetch arm.
#: One name, because the arm id appears in the arm list, the comparison pairs and
#: the store redirect, and a copy that drifts would compare two different arms.
SLEEP_NOTES_ARM = "sleep_notes"
#: #1556: the representation arm. One name for the same reason: it is the arm
#: list, both `--arms` validators, the comparison pair and the counts block.
PREFETCH_RAWSPAN_ARM = "prefetch_rawspan"
ARMS = ("closed_book", "history", "prefetch", "prefetch_rel", PREFETCH_RAWSPAN_ARM,
        "recall", "recall_episodic", SLEEP_NOTES_ARM)
#: The facts block's one char budget (#1556), applied to every arm that renders
#: `<facts>` in this runner — `prefetch`, `prefetch_rel` and `prefetch_rawspan`.
#:
#: The production block is bounded by COUNT, not chars (`FACT_MAX_ENTITIES` ×
#: `FACT_MAX_PER_ENTITY` facts, no ceiling on a line), so six distilled lines and
#: six source spans are wildly different volumes of text. Held-fixed selection
#: means the only thing the pair is allowed to vary is the rendering, so the
#: thing that gets equalised has to be stated and equal: chars, this many, both
#: arms. A span is cut to `budget // (FACT_MAX_ENTITIES × FACT_MAX_PER_ENTITY)`
#: — derived from this number, not a second tuned one — and a block never exceeds
#: it, which is the same cap on the distilled arm whether or not it binds there.
FACTS_RENDER_CHAR_BUDGET = 1200
#: The counts the rawspan arm reports, in this order, always all present —
#: including the run where nothing renders and the one where everything
#: resolves. A denominator of zero is still a denominator (#1209's rule).
RAWSPAN_COUNT_KEYS = ("n_selected", "n_unresolved_source", "n_no_span",
                      "n_budget_cut", "n_rendered")
#: What the runner keeps per question off `blocks_fn`, so the artifact can say
#: what each arm was actually shown. Read with `.get`: an injected block builder
#: that returns fewer keys leaves them present-and-empty rather than making the
#: artifact's shape depend on the injection.
PREFETCH_META_KEYS = ("facts_conf", "facts_rel", "facts_rawspan", "prefetch_ms",
                      "rawspan_counts")
#: The `vault_recall` arms (#1485): the tool's documents for the question, with
#: the episodic floor off / on. They need a qmd index whose `sessions`
#: collection holds the set's sessions (`export-sessions`, then a prepared pin);
#: against the live index the synthetic sessions do not exist.
RECALL_ARMS = ("recall", "recall_episodic")
RECALL_LIMIT = 10
#: A category with fewer answered questions than this is reported
#: `insufficient` rather than as a rate: at n=20 one question is five points
#: and the Wilson interval is ~40 points wide.
MIN_CATEGORY_N = 20
DISTRACTOR_EPISODES = 4
DEFAULT_OUT = Path(os.path.expanduser("~/lloyd-data/eval/1480/runs"))
DEFAULT_CORPUS = Path(os.path.expanduser("~/lloyd-data/eval/1480/corpus"))
Z_80 = 0.8416212335729143  # one-sided power 0.8
RESERVE_RULE = ("only a run with --holdout reads the holdout leg; it reports aggregates, "
                "never per-question rows or ids; no tuning comparison reads it")


def parse_arms(spec: str) -> list[str]:
    """Split and validate `--arms`, the one rule every command that takes it uses.

    Both `run` and the no-model `prefetch-retrieval` command take the flag, and an
    arm valid to one and refused by the other would make a comparison runnable in
    the expensive half only — which is backwards for an arm whose retrieval
    reading is meant to cost no model call.
    """
    arms = [a for a in spec.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}")
    return arms


class SetLoadError(ValueError):
    """The frozen set is malformed, edited after its manifest, or unresolvable."""


class SelfJudgeRefused(RuntimeError):
    """The judge is the model that generated the set (#1471)."""


# ─────────────────────────────────────────────────────────────── text ──

def fold(text: str) -> str:
    """Lowercase, diacritics off, digit-group commas out, quotes/backticks
    normalised, whitespace squeezed. The build's grounding check and this
    judge share it, so "gold is in the fact" and "gold is in the answer" are
    one definition."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = text.replace("’", "'").replace("`", "").replace("*", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def contains(hay: str, needle: str) -> bool:
    """Whole-token containment of `needle` in `hay`, both folded."""
    n = fold(needle)
    if not n:
        return False
    return re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", fold(hay)) is not None


def _any(hay: str, forms: list[str]) -> bool:
    return any(contains(hay, f) for f in forms if str(f).strip())


def strip_reasoning(answer: str) -> str:
    """The final answer only: a `<think>` block that leaked into content is
    not what Lloyd said to Alan."""
    return re.sub(r"(?s)<think>.*?</think>", "", answer or "").strip()


# ─────────────────────────────────────────────────────────────── set ──

def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _leg_files(root: Path, leg: str) -> list[Path]:
    base = root / leg
    return sorted(p for p in base.rglob("*") if p.is_file()) if base.exists() else []


def _load_yaml(p: Path) -> dict:
    import yaml
    try:
        return yaml.safe_load(p.read_text()) or {}
    except (OSError, ValueError) as exc:
        raise SetLoadError(f"{p}: unreadable ({exc})") from exc


def write_manifest(root: Path, *, version: str, generator_model: str, notes: str = "") -> dict:
    """Freeze `root`: a sha256 per file, per leg; the reserved (holdout) ids and a
    split hash over them; the generating model. Written once, by the build."""
    legs: dict[str, dict] = {}
    reserved: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for leg in ("dev", "holdout"):
        files = _leg_files(root, leg)
        legs[leg] = {str(p.relative_to(root)): _sha256(p) for p in files}
        qs = (_load_yaml(root / leg / "questions.yaml").get("questions") or []) \
            if (root / leg / "questions.yaml").exists() else []
        counts[leg] = {c: sum(1 for q in qs if q.get("category") == c) for c in CATEGORIES}
        if leg == "holdout":
            reserved = sorted(str(q.get("id")) for q in qs)
    body = {"version": version, "generator_model": generator_model,
            "reserved_ids": reserved, "files": legs}
    manifest = {
        **body, "counts": counts, "notes": notes,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "reserve_rule": RESERVE_RULE,
        "split_hash": _digest({"reserved_ids": reserved, "holdout": legs["holdout"]}),
        "set_sha": _digest(body),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


@dataclass
class Question:
    id: str
    category: str
    prompt: str
    asked_on: str
    probe: str
    gold: list[list[str]]
    anti: list[str]
    evidence: list[str]
    source: dict
    grounding: list[dict] = field(default_factory=list)
    temporal_kind: str | None = None
    leg: str = "dev"
    episode: dict = field(default_factory=dict)

    @property
    def entities(self) -> set[str]:
        return {str(g.get("entity", "")).lower() for g in self.grounding}


@dataclass
class MemSet:
    root: Path
    version: str
    set_sha: str
    generator_model: str
    dev: list[Question]
    holdout: list[Question] | None
    reserved_ids: frozenset[str]

    def by_id(self) -> dict[str, Question]:
        return {q.id: q for q in self.dev + (self.holdout or [])}


def _validate_question(raw: dict, leg_dir: Path, leg: str) -> Question:
    qid = str(raw.get("id") or "?")
    for f in REQUIRED_FIELDS:
        if raw.get(f) in (None, "", [], {}):
            raise SetLoadError(f"{qid}: missing field {f!r}")
    if raw["category"] not in CATEGORIES:
        raise SetLoadError(f"{qid}: category {raw['category']!r} not in {CATEGORIES}")
    if raw["probe"] not in ("recall", "action"):
        raise SetLoadError(f"{qid}: probe must be recall|action")
    acc = raw["accept"]
    all_of = acc.get("all_of") if isinstance(acc, dict) else None
    if (not isinstance(all_of, list) or not all_of
            or not all(isinstance(g, list) and g and all(str(x).strip() for x in g) for g in all_of)):
        raise SetLoadError(f"{qid}: accept.all_of must be a non-empty list of non-empty alias lists")
    anti = acc.get("none_of") or []
    if not isinstance(anti, list):
        raise SetLoadError(f"{qid}: accept.none_of must be a list")
    src = raw["source"]
    sess_rel = src.get("session_file") if isinstance(src, dict) else None
    if not sess_rel:
        raise SetLoadError(f"{qid}: source.session_file missing")
    sess_path = (leg_dir / sess_rel).resolve()
    if leg_dir.resolve() not in sess_path.parents or not sess_path.is_file():
        raise SetLoadError(f"{qid}: source {sess_rel!r} does not resolve under {leg_dir}")
    try:
        episode = json.loads(sess_path.read_text())
    except ValueError as exc:
        raise SetLoadError(f"{qid}: source {sess_rel!r} is not JSON") from exc
    sids = {s.get("sid") for s in episode.get("sessions") or []}
    if not sids:
        raise SetLoadError(f"{qid}: source {sess_rel!r} holds no sessions")
    if not src.get("facts"):
        raise SetLoadError(f"{qid}: source.facts (the fact records it was built from) missing")
    evidence = [str(e) for e in raw.get("evidence") or []]
    missing = [e for e in evidence if e not in sids]
    if missing:
        raise SetLoadError(f"{qid}: evidence {missing} not in {sess_rel}")
    return Question(id=qid, category=raw["category"], prompt=str(raw["prompt"]),
                    asked_on=str(raw["asked_on"]), probe=raw["probe"],
                    gold=[[str(x) for x in g] for g in all_of],
                    anti=[str(a) for a in anti if str(a).strip()],
                    evidence=evidence, source=src, grounding=list(raw.get("grounding") or []),
                    temporal_kind=raw.get("temporal_kind"), leg=leg, episode=episode)


def _load_leg(root: Path, leg: str, manifest: dict) -> list[Question]:
    recorded = (manifest.get("files") or {}).get(leg) or {}
    on_disk = {str(p.relative_to(root)): p for p in _leg_files(root, leg)}
    if set(recorded) != set(on_disk):
        extra = sorted(set(on_disk) - set(recorded))[:3]
        gone = sorted(set(recorded) - set(on_disk))[:3]
        raise SetLoadError(f"{leg}: files differ from the manifest (added {extra}, missing {gone})")
    for rel, p in on_disk.items():
        if _sha256(p) != recorded[rel]:
            raise SetLoadError(f"{rel}: edited after the set was frozen (sha mismatch)")
    doc = _load_yaml(root / leg / "questions.yaml")
    qs = [_validate_question(r, root / leg, leg) for r in doc.get("questions") or []]
    ids = [q.id for q in qs]
    if len(set(ids)) != len(ids):
        raise SetLoadError(f"{leg}: duplicate question ids")
    return qs


def load_set(root: Path | None = None, *, view: str = "tuning") -> MemSet:
    """Load and verify a frozen set.

    `view="tuning"` — what any tuning or A/B comparison reads — never opens a
    file under `holdout/`: it verifies the dev leg's hashes and the manifest's
    own digest, and refuses a dev id the manifest reserved. `view="all"` also
    verifies and loads the holdout leg, for the reporting run only.
    """
    root = Path(root or SET_ROOT / DEFAULT_VERSION)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise SetLoadError(f"{root}: no readable manifest.json ({exc})") from exc
    body = {k: manifest.get(k) for k in ("version", "generator_model", "reserved_ids", "files")}
    if _digest(body) != manifest.get("set_sha"):
        raise SetLoadError(f"{root}: manifest edited after freezing (set_sha mismatch)")
    if not manifest.get("generator_model"):
        raise SetLoadError(f"{root}: manifest does not record the generating model")
    reserved = frozenset(manifest.get("reserved_ids") or [])
    dev = _load_leg(root, "dev", manifest)
    leaked = reserved & {q.id for q in dev}
    if leaked:
        raise SetLoadError(f"{len(leaked)} reserved holdout id(s) appear in the dev leg")
    holdout = None
    if view == "all":
        holdout = _load_leg(root, "holdout", manifest)
        if {q.id for q in holdout} != set(reserved):
            raise SetLoadError("holdout leg ids differ from the manifest's reserved ids")
    elif view != "tuning":
        raise ValueError(f"view must be tuning|all, got {view!r}")
    return MemSet(root=root, version=str(manifest["version"]), set_sha=str(manifest["set_sha"]),
                  generator_model=str(manifest["generator_model"]), dev=dev, holdout=holdout,
                  reserved_ids=reserved)


# ─────────────────────────────────────────────────────────────── judges ──

def _model_key(name: str) -> str:
    k = re.sub(r"[^a-z0-9]", "", str(name).lower())
    for suffix in ("nvfp4", "fp8", "bf16", "awq", "gguf", "instruct"):
        k = k.replace(suffix, "")
    return k


def check_judge(generator_model: str, judge_model: str) -> None:
    """Refuse a judge that is the generator (#1471: djev never judges a set it
    helped build; here the primary built it and never judges it)."""
    if judge_model in ("rules", ""):
        return
    g, j = _model_key(generator_model), _model_key(judge_model)
    for gm in filter(None, (_model_key(x) for x in str(generator_model).split(","))):
        if gm and (gm == j or gm in j or j in gm):
            raise SelfJudgeRefused(
                f"judge {judge_model!r} is the model that generated the set ({generator_model!r})")
    if g and (g == j):
        raise SelfJudgeRefused(f"judge {judge_model!r} generated this set")


_ABSTAIN = re.compile(
    r"\b(i (?:do not|don't|dont) (?:have|know|recall|remember|see)|i'm not sure|i am not sure|"
    r"no record|not (?:in|from) (?:my|our) (?:notes|records|memory|conversations?)|"
    r"(?:can't|cannot|couldn't) (?:find|recall|remember|see)|don't have (?:any )?(?:information|context|record)|"
    r"you (?:haven't|have not|never) (?:told|mentioned|said))", re.I)


def judge_rules(q: Question, answer: str | None) -> dict:
    """Deterministic first pass.

    `correct`: every gold value named and no superseded value; `stale`: a
    superseded value and no gold; `mixed`: both — who knows which it acted on;
    `partial`: some golds of a multi-value answer; `abstain`; `none`: nothing
    recognisable (maybe a paraphrase); `error`: no answer came back.
    """
    if answer is None:
        return {"verdict": "error", "mentioned": False, "golds_hit": 0}
    text = strip_reasoning(answer)
    hits = [_any(text, forms) for forms in q.gold]
    anti = bool(q.anti) and _any(text, q.anti)
    n_hit = sum(hits)
    if n_hit == len(hits):
        verdict = "mixed" if anti else "correct"
    elif n_hit:
        verdict = "mixed" if anti else "partial"
    elif anti:
        verdict = "stale"
    elif _ABSTAIN.search(text):
        verdict = "abstain"
    else:
        verdict = "none"
    return {"verdict": verdict, "mentioned": n_hit == len(hits), "golds_hit": n_hit}


#: What djev is asked, by rules verdict. Negative options first: djev's
#: measured order sensitivity (architecture/djev.md §1) rewards listing the
#: "no" side first, and the label is read by argmax, never by a threshold.
DJEV_CRITERIA = {
    "other": "the answer commits to a different value, or to no value at all",
    "superseded": "the answer commits to the SUPERSEDED / other-option value",
    "hedged": "the answer gives both values or refuses to choose between them",
    "gold": "the answer commits to the GOLD value (same meaning, any wording)",
}
DJEV_TO_SETTLED = {"gold": "correct", "superseded": "stale", "hedged": "hedged", "other": "none"}
DJEV_JUDGE_MODEL = "djev:DiffusionGemma-26B-A4B-NVFP4"


def djev_state(q: Question, answer: str) -> str:
    gold = " AND ".join(g[0] for g in q.gold)
    anti = " / ".join(q.anti) if q.anti else "(none)"
    return (f"Alan asked his assistant:\n{q.prompt}\n\n"
            f"GOLD value(s) the right answer uses: {gold}\n"
            f"SUPERSEDED or other-option value(s): {anti}\n\n"
            f"The assistant's answer:\n{strip_reasoning(answer)[:1800]}\n\n"
            "Which value did the answer commit to — the value in the command it gave, the "
            "choice it made, the fact it stated as current? Mentioning a value as history "
            "or as a caveat does not count.")


def judge_djev(q: Question, answer: str, ask: Callable | None = None) -> dict | None:
    """djev's typed reading of one answer, or None when djev did not answer."""
    if ask is None:
        from app import djev
        ask = djev.ask_sync
    out = ask(djev_state(q, answer),
              {"acted": {"type": "choice",
                         "instructions": "Judge which value the assistant's answer acted on.",
                         "criteria": dict(DJEV_CRITERIA)}},
              seam="eval:memory", timeout=30.0)
    if out is None:
        return None
    a = out.get("acted") if hasattr(out, "get") else None
    if a is None:
        return None
    return {"label": a.label, "p": {k: round(v, 3) for k, v in a.probabilities.items()},
            "label_mass": round(a.label_mass, 3)}


def settle(rules: dict, djev: dict | None) -> dict:
    """Final verdict per answer, and who decided it.

    `correct` takes djev's `gold` for a `none`/`partial` answer (a paraphrase),
    `correct_strict` does not: a paraphrase the rules could not see counts only
    in the lenient number.
    """
    v = rules["verdict"]
    if v in ("correct", "stale", "abstain", "error"):
        return {"final": v, "strict": v, "by": "rules"}
    lab = (djev or {}).get("label")
    if v == "mixed":
        final = DJEV_TO_SETTLED.get(lab, "unresolved")
        return {"final": final, "strict": final, "by": "djev" if lab else "unresolved"}
    # none / partial: the rules saw no (complete) gold value
    if lab == "gold":
        return {"final": "correct", "strict": v, "by": "djev"}
    if lab == "superseded" and v == "none":
        return {"final": "stale", "strict": v, "by": "djev"}
    return {"final": v, "strict": v, "by": "rules"}


# ─────────────────────────────────────────────────────────────── arms ──

BASE_SYSTEM = (
    "You are Lloyd, Alan's personal AI assistant, running on his home server. Today is "
    "{today}. Alan is messaging you. Answer directly and briefly. When he asks you to do "
    "something — a command, a config line, a choice, a plan, a recommendation — do it "
    "concretely, using what you know about him and his projects. If you genuinely do not "
    "know something he expects you to remember, say so rather than inventing it.")


def _day(d: str) -> str:
    x = _dt.date.fromisoformat(d)
    return x.strftime("%A, %B ") + f"{x.day}, {x.year}"


def render_session(s: dict) -> str:
    who = {"user": "Alan", "assistant": "Lloyd"}
    lines = [f"=== Conversation on {_day(s['date'])} ==="]
    lines += [f"{who.get(t['role'], t['role'])}: {t['text']}" for t in s.get("turns") or []]
    return "\n".join(lines)


def distractors(q: Question, pool: list[Question], k: int = DISTRACTOR_EPISODES) -> list[dict]:
    """Sessions from `k` other episodes of the SAME leg: no shared entity, no
    session dated after the question, no session on a date the question's own
    sessions use. Deterministic in the question id."""
    own_dates = {s["date"] for s in q.episode.get("sessions") or []}
    cands = [o for o in pool if o.id != q.id and not (o.entities & q.entities)
             and all(s["date"] <= q.asked_on and s["date"] not in own_dates
                     for s in o.episode.get("sessions") or [])]
    rng = random.Random(f"lme-distract:{q.id}")
    picked = rng.sample(cands, min(k, len(cands)))
    return [s for o in picked for s in o.episode.get("sessions") or []]


def history_block(q: Question, pool: list[Question]) -> str:
    sessions = list(q.episode.get("sessions") or []) + distractors(q, pool)
    sessions.sort(key=lambda s: (s["date"], s["sid"]))
    return "\n\n".join(render_session(s) for s in sessions)


def evidence_text(q: Question) -> str:
    return " ".join(t["text"] for s in q.episode.get("sessions") or [] for t in s["turns"])


_FACTS_RE = re.compile(r"<facts>\n.*?\n</facts>\n?", re.S)


def splice_facts(rendered: str, question: str, fact_lines: list[str]) -> str:
    """The prefetch-rendered user message with its `<facts>` section replaced
    by `fact_lines` (inserted where production renders it when absent)."""
    ctx, sep, tail = rendered.partition("\n\n" + question)
    if not sep:  # prefetch found nothing and returned the bare text
        ctx, tail = "", ""
    body = _FACTS_RE.sub("", ctx)
    block = ("<facts>\n" + "\n".join(fact_lines) + "\n</facts>\n") if fact_lines else ""
    if not body.strip():
        return (f"<context>\n{block}</context>\n\n{question}" if block else question)
    if block:
        if "<vault-context>" in body:
            body = body.replace("<vault-context>", block + "<vault-context>", 1)
        else:
            body = body.replace("</context>", block + "</context>", 1)
    return body + "\n\n" + question + tail


#: The pair #1556 is about: one selection, two renderings of it. Named once
#: because it is the comparison the arm exists to make, and a pair written twice
#: is a pair that can disagree with itself.
RENDER_PAIR = ("prefetch_rel", PREFETCH_RAWSPAN_ARM)

#: Words too common to locate a fact inside its own source document. This is the
#: windowing rule's noise list, scoped to that job; it is not the retrieval
#: stopword list and is not used anywhere else.
_WINDOW_NOISE = frozenset("""
a an the and or but of to in for on at by with from as is are was were be been being
it its this that these those there here he she they them we you i not no so if then
than what when where who which how why do does did doing have has had will would can
could should must just also very more most other some such into over under about
""".split())


def _fact_terms(fact_text: str) -> list[str]:
    """Content terms of a distilled fact, for finding it in its own source:
    folded, word-bounded, three characters or more, noise dropped, deduped,
    order kept."""
    out: list[str] = []
    seen: set[str] = set()
    for w in re.findall(r"\w+", fold(fact_text)):
        if len(w) < 3 or w in _WINDOW_NOISE or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def source_roots() -> list[Path]:
    """Roots a fact's `source_doc` may be relative to, most specific first.

    `source_doc` is stored vault-relative, but it was written by a pipeline that
    has outlived tree moves, so the checkout and the data root are candidates
    too. A root that cannot be resolved is not a candidate, and nothing
    compensates for it: the facts it would have resolved are exactly what
    `n_unresolved_source` counts, so a missing root shows up as a number beside
    the arm's scores rather than as a silent shrinkage of the block.
    """
    roots: list[Path] = []
    try:
        from app.data_root import vault_root
        roots.append(vault_root())
    except Exception:  # noqa: BLE001 — no vault module, no vault root candidate
        pass
    roots.append(LLOYD_HOME)
    try:
        from app.paths import DATA_ROOT
        roots.append(DATA_ROOT)
    except Exception:  # noqa: BLE001 — unresolved data root: not a candidate
        roots.append(Path.home() / "lloyd-data")
    return roots


def source_doc_path(source_doc: str, roots: list[Path] | None = None) -> Path | None:
    """The file a fact's `source_doc` names, or None when nothing holds it.

    Absolute paths and anything climbing with `..` come back unresolved rather
    than opening a file: this arm reads the corpus the store names, not the disk
    at large. `roots` is injectable so a test can plant a corpus; the default is
    `source_roots()`.
    """
    doc = str(source_doc or "").strip()
    if not doc or doc.startswith(("/", "\\")) or ".." in Path(doc).parts:
        return None
    for root in (roots if roots is not None else source_roots()):
        try:
            cand = Path(root) / doc
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def rawspan_window(fact_text: str, source_text: str, window_chars: int) -> str:
    """The source text to render for one fact, or "" when the rule finds none.

    The one windowing rule, kept in a single place because it is a RENDERING
    decision and a reader has to be able to tell it from retrieval: cut the
    document at blank lines into blocks, score each block by how many of the
    fact's own content terms appear in it as whole words, take the best-scoring
    block (ties to the earliest), and return its first `window_chars` characters
    with whitespace squeezed. "" when the budget leaves no room, the document
    holds no text, or no block contains one single term.

    What it is allowed to see is the fact and the document that fact was
    extracted from — never the question. That is the whole reason the arm can
    claim retrieval is held fixed: a rule that read the question would be a
    second retriever standing inside the arm whose one retriever is under test.
    """
    text = str(source_text or "")
    if window_chars <= 0 or not text.strip():
        return ""
    terms = _fact_terms(fact_text)
    if not terms:
        return ""
    best_score, best_block = 0, ""
    for block in re.split(r"\n\s*\n", text):
        if not block.strip():
            continue
        folded = fold(block)
        score = sum(1 for t in terms
                    if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", folded))
        if score > best_score:
            best_score, best_block = score, block
            if score == len(terms):
                break  # nothing later can beat it, only tie it, and ties go earliest
    if best_score == 0:
        return ""
    return " ".join(best_block.split())[:window_chars]


def apply_char_budget(lines: list[str],
                      budget: int = FACTS_RENDER_CHAR_BUDGET) -> tuple[list[str], int]:
    """Cap a facts block at `budget` chars: keep lines in order while the block
    fits, stop at the first that does not, and report how many did not fit.

    Stopping rather than skipping is deliberate — the block is ordered, and a
    shorter later fact does not get to jump the earlier one it lost to. Every arm
    in this runner that renders `<facts>` goes through here, the distilled ones
    included, so the number is equal across the pair by construction: budget
    equalised, selection equalised, rendering the only difference.
    """
    kept: list[str] = []
    used = 0
    for line in lines:
        add = len(line) + (1 if kept else 0)
        if used + add > budget:
            return kept, len(lines) - len(kept)
        kept.append(line)
        used += add
    return kept, 0


def rawspan_window_chars(budget: int = FACTS_RENDER_CHAR_BUDGET) -> int:
    """The per-fact slice: the one budget over the most facts production can
    select (`FACT_MAX_ENTITIES × FACT_MAX_PER_ENTITY`). One place, because the
    artifact reports this number and the arm renders with it — two copies is how
    a reported window stops being the window that was used."""
    from app import prefetch as pf
    return budget // max(1, pf.FACT_MAX_ENTITIES * pf.FACT_MAX_PER_ENTITY)


def render_rawspan_lines(records: list[dict], *, budget: int = FACTS_RENDER_CHAR_BUDGET,
                         window_chars: int = 0,
                         roots: list[Path] | None = None) -> tuple[list[str], dict]:
    """The rawspan arm's facts lines for one selection, plus the counts behind them.

    One line per selected record, in the arm's selection order, carrying the
    source text (`rawspan_window`) in place of the distilled fact line and naming
    the entity and the document the words came from. Then the same
    `apply_char_budget` the distilled arms go through.

    `window_chars` is the per-fact slice; 0 derives it as
    `budget // (FACT_MAX_ENTITIES × FACT_MAX_PER_ENTITY)` — the one budget shared
    across the most facts production can select, so no second tuned number
    sneaks in.

    Returns `(lines, counts)` with every key of `RAWSPAN_COUNT_KEYS` present
    whatever happened, because "the arm rendered nothing" and "the arm rendered
    nothing, and here is why" are different results — a fact dropped quietly
    would make an empty block read as a retrieval failure rather than a
    provenance gap.
    """
    if window_chars <= 0:
        window_chars = rawspan_window_chars(budget)
    counts: dict = dict.fromkeys(RAWSPAN_COUNT_KEYS, 0)
    counts["n_selected"] = len(records)
    lines: list[str] = []
    for rec in records:
        doc = str(rec.get("source_doc") or "").strip()
        path = source_doc_path(doc, roots)
        if path is None:
            counts["n_unresolved_source"] += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            counts["n_unresolved_source"] += 1
            continue
        span = rawspan_window(str(rec.get("fact") or ""), text, window_chars)
        if not span:
            counts["n_no_span"] += 1
            continue
        lines.append(f"- [{rec.get('entity', '')}] {span} (source: {doc})")
    kept, cut = apply_char_budget(lines, budget)
    counts["n_budget_cut"] = cut
    counts["n_rendered"] = len(kept)
    return kept, counts


def prefetch_blocks(q: Question) -> dict:
    """The prefetch renderings for one question, off one selection each.

    `prefetch` and `prefetch_rel` are the same facts in two orders — confidence
    order (today's shipped behaviour) and query-relevance order (#1482 rider 1) —
    with every other byte of the block shared. `prefetch_rawspan` is the
    relevance arm's records rendered as the source text behind them (#1556), and
    it takes them from the SAME `_search_fact_records` call that produced the
    relevance lines: identical selection there is a property of this code path,
    not a coincidence of two lookups agreeing. All three blocks go through one
    char budget, so what the pair prices is representation at a fixed budget and
    not volume.

    Which rendering comes out ahead is decided by a run, not here: no branch,
    default or comment in this function treats either direction as expected.
    """
    from app import prefetch as pf
    t0 = time.monotonic()
    rendered = pf.prefetch_context(q.prompt)
    ms = (time.monotonic() - t0) * 1000
    conf_records = pf._search_fact_records(q.prompt, rank="confidence")
    rel_records = pf._search_fact_records(q.prompt, rank="relevance")
    conf, conf_cut = apply_char_budget([r["line"] for r in conf_records], FACTS_RENDER_CHAR_BUDGET)
    rel, rel_cut = apply_char_budget([r["line"] for r in rel_records], FACTS_RENDER_CHAR_BUDGET)
    rawspan, rawspan_counts = render_rawspan_lines(
        rel_records, budget=FACTS_RENDER_CHAR_BUDGET,
        window_chars=rawspan_window_chars(FACTS_RENDER_CHAR_BUDGET))
    return {"prefetch": splice_facts(rendered, q.prompt, conf),
            "prefetch_rel": splice_facts(rendered, q.prompt, rel),
            PREFETCH_RAWSPAN_ARM: splice_facts(rendered, q.prompt, rawspan),
            "facts_conf": conf, "facts_rel": rel, "facts_rawspan": rawspan,
            "prefetch_ms": round(ms, 1),
            "rawspan_counts": rawspan_counts,
            "facts_rel_counts": {"n_selected": len(rel_records), "n_budget_cut": rel_cut,
                                 "n_rendered": len(rel)},
            "facts_conf_counts": {"n_selected": len(conf_records), "n_budget_cut": conf_cut,
                                  "n_rendered": len(conf)}}


def sleep_notes_block(prompt: str, fact_lines: list[str]) -> str:
    """`fact_lines` carried by the #1516 channel, ready to be the arm's user turn.

    The pair is a delivery comparison, not a retrieval one: `prefetch` shows the
    confidence-ordered facts inside the per-turn `<context>` block, and this arm
    shows the same lines inside `<next-session-notes>`, off the same one prefetch
    call. Everything else about the question — the seed, the model, the judge — is
    the same byte, so whatever the pair measures is the channel.

    It measures a real cost, because the channel is a real channel. The note is
    written through `app.next_session_notes` and drained back the way a morning
    turn drains it, so the renderer's `NEXT_SESSION_CONTENT_MAX` applies: material
    the prefetch block would have carried in full can be truncated here, and the
    truncation lands in `evidence_in_context` exactly where a reader would feel it.
    A gain for this arm is not assumed anywhere in this file — the ship/no-ship
    call is the paired-bootstrap CI over a run, which is why the arm exists.
    """
    from app import prefetch as pf
    from app import next_session_notes as nsn

    nsn.write_next_session_note(
        source="eval:lloydmemeval",
        summary="LloydMemEval sleep-note: facts retrieved for this question",
        content="\n".join(fact_lines))
    notes = nsn.drain_next_session_notes()
    if not notes:
        return prompt
    block = pf._format_context([], [], notes=notes, show_skill_hint=False)
    return f"{block}\n\n{prompt}" if block else prompt


def session_export_id(sid: str, date: str) -> str:
    """A chat-shaped id (`YYYYMMDD_HHMMSS_<tag>`, three parts like a real chat)
    for one synthetic session, deterministic in its sid."""
    h = int(hashlib.sha256(f"lme-export:{sid}".encode()).hexdigest(), 16)
    hhmmss = f"{9 + h % 11:02d}{(h >> 8) % 60:02d}{(h >> 16) % 60:02d}"
    return f"{date.replace('-', '')}_{hhmmss}_me{format((h >> 24) % 65536, '04x')}"


def render_session_export(s: dict) -> tuple[str, str]:
    """(relative path, markdown) of one synthetic session in the shape
    `app/post_capture._export_session_markdown` writes a chat into the qmd
    `sessions` collection: `<date>/<id>.md`, `# id`, `# iso`, `user:`/`lloyd:`."""
    sid_x = session_export_id(s["sid"], s["date"])
    t = sid_x.split("_")[1]
    iso = f"{s['date']}T{t[:2]}:{t[2:4]}:{t[4:]}-07:00"
    lines = [f"# {sid_x}", f"# {iso}", "# model: primary", ""]
    who = {"user": "user", "assistant": "lloyd"}
    for turn in s.get("turns") or []:
        lines.append(f"{who.get(turn['role'], turn['role'])}: {turn['text']}")
    return f"{s['date']}/{sid_x}.md", "\n".join(lines) + "\n"


def export_sessions(qs: list[Question], out: Path) -> dict[str, str]:
    """Write every session of `qs` under `out` as a chat export. Returns
    {sid: relative path}. Tuning view only: the holdout leg is never exported."""
    out.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for q in qs:
        for s in q.episode.get("sessions") or []:
            rel, text = render_session_export(s)
            p = out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            mapping[s["sid"]] = rel
    return mapping


def render_recall(question: str, docs: list[dict]) -> str:
    """The question under the documents `vault_recall` returned for it."""
    if not docs:
        return question
    body = "\n".join(f"- {d.get('path', '')}: {' '.join(str(d.get('snippet', '')).split())}"
                     for d in docs)
    return f"<vault_recall>\n{body}\n</vault_recall>\n\n{question}"


def recall_blocks(q: Question) -> dict:
    """`vault_recall` for the question with the episodic floor off and on
    (the module override, so both arms run in one process on one index)."""
    from agent_mcp import vault as V
    out: dict = {"recall_docs": {}, "recall_ms": {}}
    prev = V.RECALL_EPISODIC_FLOORS
    try:
        for arm, on in (("recall", False), ("recall_episodic", True)):
            V.RECALL_EPISODIC_FLOORS = on
            t0 = time.monotonic()
            res = V._vault_recall({"query": q.prompt, "limit": RECALL_LIMIT, "include_facts": False})
            out["recall_ms"][arm] = round((time.monotonic() - t0) * 1000, 1)
            docs = (res.get("documents") or []) if isinstance(res, dict) else []
            out["recall_docs"][arm] = [d.get("path", "") for d in docs]
            out[arm] = render_recall(q.prompt, docs)
    finally:
        V.RECALL_EPISODIC_FLOORS = prev
    return out


def gold_in(q: Question, text: str) -> bool:
    """RETRIEVAL: every gold value (or the grounding fact's text) is in `text`."""
    if all(_any(text, forms) for forms in q.gold):
        return True
    facts = [g.get("fact", "") for g in q.grounding]
    return bool(facts) and all(fact and contains(text, fact[:80]) for fact in facts)


def build_messages(q: Question, arm: str, pool: list[Question], blocks: dict | None) -> list[dict]:
    system = BASE_SYSTEM.format(today=_day(q.asked_on))
    user = q.prompt
    if arm == "history":
        system += ("\n\nYour earlier conversations with Alan, oldest first:\n\n"
                   + history_block(q, pool))
    elif arm in ("prefetch", "prefetch_rel", PREFETCH_RAWSPAN_ARM, SLEEP_NOTES_ARM) or arm in RECALL_ARMS:
        user = blocks[arm]
    elif arm != "closed_book":
        raise ValueError(f"unknown arm {arm!r}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def evidence_in_context(q: Question, arm: str, messages: list[dict]) -> bool:
    if arm == "closed_book":
        return False
    if arm == "history":
        return True
    return gold_in(q, messages[-1]["content"])


# ─────────────────────────────────────────────────────────────── engine ──

async def _answer(client, base_url: str, model: str, messages: list[dict], seed: int,
                  max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "priority": 1,
            "temperature": 0.6, "top_p": 0.95, "seed": seed}
    t0 = time.monotonic()
    try:
        r = await client.post(f"{base_url}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        return {"answer": msg.get("content") or "", "finish": data["choices"][0].get("finish_reason"),
                "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
                "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
                "latency_s": round(time.monotonic() - t0, 2)}
    except Exception as exc:  # recorded as one `error` row, never raised
        return {"answer": None, "error": f"{type(exc).__name__}: {exc}"[:300],
                "latency_s": round(time.monotonic() - t0, 2)}


async def answer_all(jobs: list[dict], *, base_url: str, model: str, concurrency: int,
                     max_tokens: int, complete=None) -> None:
    import httpx
    complete = complete or _answer
    sem = asyncio.Semaphore(concurrency)
    done = [0]
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=10)) as client:
        async def one(j):
            async with sem:
                j.update(await complete(client, base_url, model, j["messages"], j["seed"], max_tokens))
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  answered {done[0]}/{len(jobs)}", flush=True)
        await asyncio.gather(*(one(j) for j in jobs))


# ─────────────────────────────────────────────────────────────── stats ──

def mde_paired(n: int, p_disc: float, p_diff: float = 0.0) -> float | None:
    """Minimum detectable difference (two-sided α=0.05, power 0.8) for a paired
    binary comparison with discordance `p_disc`: (z.975 + z.8)·sd(d)/√n, with
    sd(d)² = p_disc − diff². None when n is 0."""
    if n <= 0:
        return None
    var = max(p_disc - p_diff * p_diff, 1e-9)
    from stats import Z_975
    return (Z_975 + Z_80) * math.sqrt(var) / math.sqrt(n)


def mde_planning(n: int, p: float) -> float | None:
    """The same MDE before any second arm exists: discordance taken as
    2p(1−p) (the arms independent given the question — the conservative case),
    p clipped to [0.1, 0.9] so a floor or ceiling arm does not promise a
    precision it has not earned."""
    if n <= 0:
        return None
    p = min(0.9, max(0.1, p))
    return mde_paired(n, 2 * p * (1 - p))


def _rate(k: int, n: int) -> dict:
    from stats import wilson_ci
    if n < MIN_CATEGORY_N:
        return {"k": k, "n": n, "rate": None, "ci": None, "insufficient": True,
                "note": f"n={n} < {MIN_CATEGORY_N}: reported insufficient, not as a number"}
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4), "ci": [round(lo, 4), round(hi, 4)],
            "mde_80": round(mde_planning(n, k / n), 4)}


METRICS = ("correct", "correct_strict", "mentioned", "stale", "abstain", "evidence_in_context")


def summarize_arm(rows: list[dict]) -> dict:
    out: dict = {}
    for cat in CATEGORIES + ("all",):
        rs = [r for r in rows if cat == "all" or r["category"] == cat]
        block = {m: _rate(sum(1 for r in rs if r[m]), len(rs)) for m in METRICS}
        ment = [r for r in rs if r["mentioned"]]
        block["use_rate"] = _rate(sum(1 for r in ment if r["correct"]), len(ment))
        block["use_rate"]["definition"] = "correct / mentioned: of answers naming the gold value, how many acted on it"
        out[cat] = block
    return out


def compare(rows: list[dict], a: str, b: str, metric: str = "correct") -> dict:
    """Paired arm-vs-arm on shared question ids, per category and overall."""
    from stats import paired_bootstrap_ci
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["arm"]] = r
    out = {}
    for cat in CATEGORIES + ("all",):
        ids = sorted(i for i, d in by.items() if a in d and b in d
                     and (cat == "all" or d[a]["category"] == cat))
        va = [float(by[i][a][metric]) for i in ids]
        vb = [float(by[i][b][metric]) for i in ids]
        if len(ids) < MIN_CATEGORY_N:
            out[cat] = {"n": len(ids), "insufficient": True}
            continue
        ci = paired_bootstrap_ci(va, vb)
        disc = sum(1 for x, y in zip(va, vb) if x != y) / len(ids)
        out[cat] = {"n": len(ids), "a": round(sum(va) / len(ids), 4), "b": round(sum(vb) / len(ids), 4),
                    "diff": round(ci["diff"], 4), "ci": [round(ci["lo"], 4), round(ci["hi"], 4)],
                    "p": round(ci["p"], 4), "significant": ci["significant"],
                    "discordant": round(disc, 4),
                    "mde_80": round(mde_paired(len(ids), disc, ci["diff"]) or 0.0, 4)}
    return {"a": a, "b": b, "metric": metric, "by_category": out}


# ─────────────────────────────────────────────────────────────── run ──

def score_rows(jobs: list[dict], qs: dict[str, Question], *, use_djev: bool,
               djev_ask: Callable | None = None) -> list[dict]:
    rows = []
    for j in jobs:
        q = qs[j["id"]]
        rules = judge_rules(q, j.get("answer"))
        dj = None
        if use_djev and j.get("answer") and rules["verdict"] in ("mixed", "none", "partial", "correct", "stale"):
            # every settle-able answer is also read by djev, so the artifact can
            # say how often djev agrees with the rules where the rules are sure
            dj = judge_djev(q, j["answer"], ask=djev_ask)
        st = settle(rules, dj)
        rows.append({
            "id": q.id, "arm": j["arm"], "category": q.category, "probe": q.probe,
            "temporal_kind": q.temporal_kind,
            "rules": rules["verdict"], "djev": (dj or {}).get("label"), "djev_p": (dj or {}).get("p"),
            "final": st["final"], "decided_by": st["by"],
            "correct": st["final"] == "correct", "correct_strict": st["strict"] == "correct",
            "mentioned": rules["mentioned"], "stale": st["final"] == "stale",
            "abstain": st["final"] == "abstain",
            "evidence_in_context": j.get("evidence_in_context", False),
            "answer": j.get("answer"), "error": j.get("error"), "finish": j.get("finish"),
            "prompt_tokens": j.get("prompt_tokens"), "completion_tokens": j.get("completion_tokens"),
            "latency_s": j.get("latency_s"), "reused_from": j.get("reused_from"),
        })
    return rows


def judge_agreement(rows: list[dict]) -> dict:
    """djev vs the rules on answers the rules settled on their own."""
    pairs = [(r["rules"], r["djev"]) for r in rows if r["djev"] and r["rules"] in ("correct", "stale")]
    agree = sum(1 for rv, dv in pairs if (rv == "correct" and dv == "gold") or (rv == "stale" and dv == "superseded"))
    return {"n": len(pairs), "agree": agree,
            "rate": round(agree / len(pairs), 4) if pairs else None,
            "by_rules": {rv: {lab: sum(1 for a, b in pairs if a == rv and b == lab)
                              for lab in DJEV_CRITERIA} for rv in ("correct", "stale")}}


def run(argv: list[str] | None = None, *, complete=None, djev_ask=None, primary=None,
        blocks_fn=None, recall_fn=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(SET_ROOT / DEFAULT_VERSION))
    ap.add_argument("--arms", default="closed_book,history,prefetch")
    ap.add_argument("--label", default="run")
    ap.add_argument("--holdout", action="store_true", help="also run the holdout leg (aggregates only)")
    ap.add_argument("--judge", default="djev", choices=("djev", "rules"))
    ap.add_argument("--limit", type=int, default=0, help="first N dev questions per category (smoke)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--base-url", default="http://127.0.0.1:8096")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                    help="facts snapshot the prefetch arms read (LLOYD_FACTS_ROOT/LLOYD_KG_DB)")
    args = ap.parse_args(argv)
    arms = parse_arms(args.arms)
    ms = load_set(Path(args.set), view="all" if args.holdout else "tuning")
    judge_model = DJEV_JUDGE_MODEL if args.judge == "djev" else "rules"
    check_judge(ms.generator_model, judge_model)
    want_facts = any(a.startswith("prefetch") for a in arms) or SLEEP_NOTES_ARM in arms
    if want_facts and blocks_fn is None:
        os.environ.setdefault("LLOYD_FACTS_ROOT", str(Path(args.corpus) / "facts"))
        os.environ.setdefault("LLOYD_KG_DB", str(Path(args.corpus) / "kg.sqlite"))
    blocks_fn = blocks_fn or prefetch_blocks
    recall_fn = recall_fn or recall_blocks
    use_recall = any(a in RECALL_ARMS for a in arms)
    # The #1516 arm writes through the live channel module, whose default path is
    # the running tree's data root. A benchmark that left a "what to know today"
    # note standing there would hand it to the next real chat turn if the run died
    # between writing and draining, so the run gets a store of its own in its own
    # out-dir unless the caller named one first.
    sleep_store = ""
    if SLEEP_NOTES_ARM in arms:
        from app import next_session_notes as nsn
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sleep_store = str(out_dir / f"sleep-notes-store-{os.getpid()}.json")
        os.environ.setdefault(nsn.STORE_PATH_ENV, sleep_store)
        sleep_store = os.environ[nsn.STORE_PATH_ENV]

    legs = [("dev", ms.dev)] + ([("holdout", ms.holdout)] if args.holdout else [])
    if args.limit:
        legs = [(n, [q for c in CATEGORIES for q in [x for x in qs if x.category == c][:args.limit]])
                for n, qs in legs]
    if primary is None:
        import httpx
        model = httpx.get(f"{args.base_url}/v1/models", timeout=10).json()["data"][0]["id"]
        base_url = args.base_url
    else:
        base_url, model = primary

    jobs: list[dict] = []
    prefetch_meta: dict[str, dict] = {}
    recall_meta: dict[str, dict] = {}
    for leg, qs in legs:
        for q in qs:
            blocks = None
            if want_facts:
                blocks = blocks_fn(q)
                prefetch_meta[q.id] = {k: blocks.get(k) for k in PREFETCH_META_KEYS}
                if SLEEP_NOTES_ARM in arms:
                    # Off the SAME retrieval, so the pair prices the channel and
                    # nothing else: the arm's only advantage would be that a note
                    # is already there when the turn opens, and its only cost is
                    # the renderer's cap on what a note may carry.
                    blocks = {**blocks, SLEEP_NOTES_ARM: sleep_notes_block(
                        q.prompt, list(blocks.get("facts_conf") or []))}
            if use_recall:
                rb = recall_fn(q)
                blocks = {**(blocks or {}), **rb}
                recall_meta[q.id] = {"docs": rb["recall_docs"], "ms": rb["recall_ms"]}
            seed = int(hashlib.sha256(q.id.encode()).hexdigest()[:8], 16)
            for arm in arms:
                msgs = build_messages(q, arm, qs, blocks)
                jobs.append({"id": q.id, "leg": leg, "arm": arm, "messages": msgs, "seed": seed,
                             "evidence_in_context": evidence_in_context(q, arm, msgs)})
    # identical input → one call: the rider-1 arm differs from `prefetch` only
    # where the facts block does, and re-sampling an identical prompt would add
    # noise the change did not cause.
    first: dict[tuple, dict] = {}
    to_call = []
    for j in jobs:
        key = (j["id"], json.dumps(j["messages"], sort_keys=True))
        if key in first:
            j["_twin"] = first[key]
        else:
            first[key] = j
            to_call.append(j)
    print(f"{len(jobs)} answers over {len(arms)} arm(s); {len(to_call)} distinct prompts", flush=True)
    t0 = time.monotonic()
    asyncio.run(answer_all(to_call, base_url=base_url, model=model, concurrency=args.concurrency,
                           max_tokens=args.max_tokens, complete=complete))
    for j in jobs:
        tw = j.pop("_twin", None)
        if tw is not None:
            for k in ("answer", "error", "finish", "completion_tokens", "prompt_tokens", "latency_s"):
                j[k] = tw.get(k)
            j["reused_from"] = tw["arm"]
    wall = round(time.monotonic() - t0, 1)
    qmap = ms.by_id()
    use_djev = args.judge == "djev"
    rows = score_rows(jobs, qmap, use_djev=use_djev, djev_ask=djev_ask)

    report: dict = {
        "eval": "LloydMemEval", "item": "#1480", "label": args.label,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "set": {"version": ms.version, "set_sha": ms.set_sha, "root": str(ms.root),
                "generator_model": ms.generator_model,
                "n_dev": len(ms.dev), "n_holdout": len(ms.holdout or []) if args.holdout else None},
        "answerer": {"model": model, "base_url": base_url, "thinking": "on (engine default)",
                     "temperature": 0.6, "top_p": 0.95, "seed": "sha256(question id)",
                     "max_tokens": args.max_tokens},
        "judge": {"model": judge_model, "first_pass": "rules (deterministic alias match)",
                  "settles": "mixed / unmatched answers" if use_djev else "nothing (rules only)"},
        "arms": arms, "min_category_n": MIN_CATEGORY_N, "wall_s": wall,
        "prefetch_corpus": os.environ.get("LLOYD_FACTS_ROOT"),
        # Which file the #1516 arm's notes went through, so a reader can tell a
        # run that used the channel from one that only named it, and can see for
        # itself that it was never the live one.
        "sleep_notes_store": sleep_store or None,
    }
    dev_rows = [r for r, j in zip(rows, jobs) if j["leg"] == "dev"]
    report["dev"] = {arm: summarize_arm([r for r in dev_rows if r["arm"] == arm]) for arm in arms}
    comps = []
    for a, b in (("closed_book", "history"), ("closed_book", "prefetch"), ("prefetch", "prefetch_rel"),
                 RENDER_PAIR, ("recall", "recall_episodic"), (SLEEP_NOTES_ARM, "prefetch")):
        if a in arms and b in arms:
            for metric in ("correct_strict", "correct", "evidence_in_context"):
                comps.append(compare(dev_rows, a, b, metric))
    report["dev_comparisons"] = comps
    report["judge_agreement"] = judge_agreement(dev_rows) if use_djev else None
    if args.holdout:
        ho = [r for r, j in zip(rows, jobs) if j["leg"] == "holdout"]
        # aggregates only — the reserve rule
        report["holdout"] = {arm: summarize_arm([r for r in ho if r["arm"] == arm]) for arm in arms}
        report["holdout_reserve_rule"] = RESERVE_RULE
    report["rows"] = dev_rows
    report["prefetch"] = {i: m for i, m in prefetch_meta.items() if qmap[i].leg == "dev"}
    # #1556: what the rawspan arm's scores were computed OVER. The block can be
    # short for two very different reasons — the facts' provenance did not
    # resolve, or the char budget had no room — and a reader told only that
    # `evidence_in_context` is low cannot tell which, so the denominators travel
    # with the rates. Emitted whenever the arm ran, whatever the counts are: an
    # all-zero set is a result about the corpus, and its absence would be
    # indistinguishable from an arm that rendered nothing at all. A run of other
    # arms leaves it absent, because that arm did not run.
    if PREFETCH_RAWSPAN_ARM in arms:
        dev_pf_ids = [i for i in prefetch_meta if qmap[i].leg == "dev"]
        report["prefetch_rawspan"] = {
            "char_budget": FACTS_RENDER_CHAR_BUDGET,
            "window_chars": rawspan_window_chars(),
            "n_questions": len(dev_pf_ids),
            "counts": {k: sum(int(((prefetch_meta[i].get("rawspan_counts") or {}).get(k)) or 0)
                              for i in dev_pf_ids) for k in RAWSPAN_COUNT_KEYS},
            "render_pair": list(RENDER_PAIR),
        }
    if use_recall:
        report["recall"] = {i: m for i, m in recall_meta.items() if qmap[i].leg == "dev"}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lloydmemeval-{args.label}.json"
    path.write_text(json.dumps(report, indent=1, default=str) + "\n")
    report["_path"] = str(path)
    _print_summary(report)
    print(f"wrote {path}")
    return report


def _fmt(b: dict) -> str:
    if b.get("insufficient"):
        return f"insufficient (n={b['n']})"
    return f"{b['rate']:.3f} [{b['ci'][0]:.3f},{b['ci'][1]:.3f}] n={b['n']}"


def _print_summary(report: dict) -> None:
    for arm, blk in report["dev"].items():
        print(f"\n[{arm}]")
        for cat in CATEGORIES + ("all",):
            c = blk[cat]
            print(f"  {cat:17s} strict {_fmt(c['correct_strict'])}  evid {_fmt(c['evidence_in_context'])}  "
                  f"use {_fmt(c['use_rate'])}")
    for comp in report.get("dev_comparisons") or []:
        a = comp["by_category"]["all"]
        if not a.get("insufficient"):
            print(f"  {comp['b']} - {comp['a']} ({comp['metric']}): {a['diff']:+.3f} "
                  f"[{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}] n={a['n']}")
    if (sp := report.get("prefetch_rawspan")):
        # The block's denominators beside the scores: which arm was rendered from
        # what, and how much of the corpus the rendering could not reach.
        print(f"\nrawspan rendering: {sp['char_budget']}-char facts block at "
              f"{sp['window_chars']} chars/fact over {sp['n_questions']} dev questions: "
              + "  ".join(f"{k}={v}" for k, v in sp["counts"].items()))


# ─────────────────────────────────────────────────────────────── no-model A/B ──

def prefetch_retrieval(argv: list[str] | None = None, *, select=None) -> dict:
    """The prefetch arms' retrieval half with no model call: for every question,
    is the gold value in the facts block this arm would render? Same snapshot,
    same query, paired.

    Two readings cost no GPU minute here. Rider 1's is confidence order against
    relevance order over the same facts (`--arms prefetch,prefetch_rel`), and
    #1556's is the distilled line against the source text behind exactly those
    facts (`--arms prefetch_rel,prefetch_rawspan`) — the second one is only cheap
    because retrieval is held fixed, so no answer is needed from anyone but the
    fact store. It reports the signed difference and its bootstrap interval for
    the requested pair and takes no position on the sign; the counts of facts
    whose source did not resolve, whose windowing rule found no span, and that
    the char budget cut travel beside the rates, so a low rawspan number can be
    read as a rendering result rather than mistaken for a provenance gap.

    `select` is the selection call, injectable for tests; by default it is
    `app.prefetch._search_fact_records` — the same function the modelled arms
    render from, so the two arms in this comparison are sharing a selection here
    exactly as they do in a real run, not because two fixtures were written to
    agree.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(SET_ROOT / DEFAULT_VERSION))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--arms", default=",".join(("prefetch", "prefetch_rel", PREFETCH_RAWSPAN_ARM)))
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--label", default=None,
                    help="artifact suffix; the arm pair by default, so a run of the pair "
                         "lands beside the modelled run's artifact and is found by name")
    args = ap.parse_args(argv)
    arms = parse_arms(args.arms)
    label = args.label or "-".join(arms)
    os.environ.setdefault("LLOYD_FACTS_ROOT", str(Path(args.corpus) / "facts"))
    os.environ.setdefault("LLOYD_KG_DB", str(Path(args.corpus) / "kg.sqlite"))
    if select is None:
        from app import prefetch as pf
        select = pf._search_fact_records
    from stats import paired_bootstrap_ci
    ms = load_set(Path(args.set), view="all" if args.holdout else "tuning")
    qs = ms.holdout if args.holdout else ms.dev
    want_conf = "prefetch" in arms
    want_rel = ("prefetch_rel" in arms) or (PREFETCH_RAWSPAN_ARM in arms)
    want_span = PREFETCH_RAWSPAN_ARM in arms
    rows = []
    for q in qs:
        conf_records = rel_records = None
        t0 = time.perf_counter()
        if want_conf:
            conf_records = select(q.prompt, "confidence")
        t1 = time.perf_counter()
        if want_rel:
            rel_records = select(q.prompt, "relevance")
        t2 = time.perf_counter()
        conf_lines, _ = apply_char_budget([r["line"] for r in (conf_records or [])])
        rel_lines, rel_cut = apply_char_budget([r["line"] for r in (rel_records or [])])
        span_lines, span_counts = render_rawspan_lines(rel_records or [])
        row = {"id": q.id, "category": q.category,
               "ms_conf": (t1 - t0) * 1e3, "ms_rel": (t2 - t1) * 1e3,
               "conf": gold_in(q, "\n".join(conf_lines)) if want_conf else None,
               "rel": gold_in(q, "\n".join(rel_lines)) if want_rel else None,
               "changed": (conf_lines != rel_lines) if (want_conf and want_rel) else None,
               "rawspan": gold_in(q, "\n".join(span_lines)) if want_span else None,
               "rawspan_counts": span_counts,
               "rel_counts": {"n_selected": len(rel_records or []), "n_budget_cut": rel_cut,
                              "n_rendered": len(rel_lines)}}
        rows.append(row)
    out: dict = {"n": len(rows), "leg": "holdout" if args.holdout else "dev", "arms": arms,
                 "char_budget": FACTS_RENDER_CHAR_BUDGET, "by_category": {}}
    for cat in CATEGORIES + ("all",):
        rs = [r for r in rows if cat == "all" or r["category"] == cat]
        if not rs:
            continue
        cell: dict = {"n": len(rs)}
        if want_conf and want_rel:
            a = [float(r["conf"]) for r in rs]
            b = [float(r["rel"]) for r in rs]
            ci = paired_bootstrap_ci(a, b) if len(rs) >= 2 else None
            cell.update({
                "conf": round(sum(a) / len(rs), 4), "rel": round(sum(b) / len(rs), 4),
                "block_changed": sum(r["changed"] for r in rs),
                "diff": round(ci["diff"], 4) if ci else None,
                "ci": [round(ci["lo"], 4), round(ci["hi"], 4)] if ci else None})
        elif want_rel:
            cell["rel"] = round(sum(float(r["rel"]) for r in rs) / len(rs), 4)
        if want_span:
            cell["rawspan"] = round(sum(float(r["rawspan"]) for r in rs) / len(rs), 4)
            cell["rawspan_counts"] = {k: sum(r["rawspan_counts"][k] for r in rs)
                                      for k in RAWSPAN_COUNT_KEYS}
        out["by_category"][cat] = cell
    if want_span:
        # The denominators for the whole leg, always all five keys, whatever the
        # arm rendered: an all-zero set is a result about the corpus, and an
        # absent block would read as "the arm did not run".
        out["rawspan_counts"] = {k: sum(r["rawspan_counts"][k] for r in rows)
                                 for k in RAWSPAN_COUNT_KEYS}
        out["window_chars"] = rawspan_window_chars()
    a_arm, b_arm = RENDER_PAIR
    if a_arm in arms and b_arm in arms:
        comp: dict = {"a": a_arm, "b": b_arm, "metric": "gold_in_block",
                      "reading": f"diff is {b_arm} - {a_arm}: rate of gold values present in "
                                 "the block, per question, paired on the question id. Both "
                                 "signs are readable and neither is expected",
                      "by_category": {}}
        # the per-question row keys the arms' gold-in-block flags were written to
        row_key = {"prefetch": "conf", "prefetch_rel": "rel", PREFETCH_RAWSPAN_ARM: "rawspan"}
        for cat in CATEGORIES + ("all",):
            rs = [r for r in rows if cat == "all" or r["category"] == cat]
            if not rs:
                continue
            va = [float(r[row_key[a_arm]]) for r in rs]
            vb = [float(r[row_key[b_arm]]) for r in rs]
            ci = paired_bootstrap_ci(va, vb) if len(rs) >= 2 else None
            comp["by_category"][cat] = {
                "n": len(rs), "a": round(sum(va) / len(rs), 4), "b": round(sum(vb) / len(rs), 4),
                "diff": round(ci["diff"], 4) if ci else None,
                "ci": [round(ci["lo"], 4), round(ci["hi"], 4)] if ci else None}
        out["comparison"] = comp
    if want_conf and want_rel:
        lat = sorted(r["ms_rel"] - r["ms_conf"] for r in rows)
        out["extra_ms_p50"] = round(lat[len(lat) // 2], 3) if lat else None
    # The reading is written, not just printed: this command's whole purpose is a
    # comparison someone quotes later, and a delta that exists only in a terminal
    # scrollback cannot be checked against the counts that qualify it. Same
    # directory and naming scheme as the modelled runs, so a paired retrieval
    # reading and its modelled counterpart sit side by side.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"prefetch-retrieval-{label}.json"
    written = {"argv": list(argv), "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "arms": arms, "facts_root": os.environ.get("LLOYD_FACTS_ROOT"), **out}
    path.write_text(json.dumps(written, indent=1, default=str) + "\n")
    out["_path"] = str(path)
    print(json.dumps(out, indent=1))
    print(f"wrote {path}", file=sys.stderr)
    return out


def evidence_hits(q: Question, paths: list[str]) -> tuple[bool, bool]:
    """(any, all) of the question's evidence sessions among `paths`."""
    by_sid = {s["sid"]: session_export_id(s["sid"], s["date"])
              for s in q.episode.get("sessions") or []}
    want = [by_sid[e] for e in q.evidence if e in by_sid]
    got = [any(w in p for p in paths) for w in want]
    return (any(got), bool(got) and all(got))


def recall_retrieval(argv: list[str] | None = None) -> dict:
    """#1485's retrieval half with no model: per dev question, `vault_recall`
    with the episodic floor off vs on — evidence session(s) in the top
    `RECALL_LIMIT` documents, and every gold value in their snippets. Run
    against a qmd index holding the exported sessions (`export-sessions`)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(SET_ROOT / DEFAULT_VERSION))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    from stats import paired_bootstrap_ci
    ms = load_set(Path(args.set), view="tuning")
    rows = []
    for q in ms.dev:
        rb = recall_blocks(q)
        r = {"id": q.id, "category": q.category, "ms": rb["recall_ms"], "docs": rb["recall_docs"]}
        for arm in RECALL_ARMS:
            anyh, allh = evidence_hits(q, rb["recall_docs"][arm])
            r[f"{arm}_any"], r[f"{arm}_all"] = anyh, allh
            r[f"{arm}_gold"] = gold_in(q, rb[arm])
        rows.append(r)
    out = {"n": len(rows), "leg": "dev", "limit": RECALL_LIMIT, "by_category": {}}
    for cat in CATEGORIES + ("all",):
        rs = [r for r in rows if cat == "all" or r["category"] == cat]
        if not rs:
            continue
        row = {"n": len(rs)}
        for m in ("any", "all", "gold"):
            a = [float(r[f"recall_{m}"]) for r in rs]
            b = [float(r[f"recall_episodic_{m}"]) for r in rs]
            ci = paired_bootstrap_ci(a, b) if len(rs) >= 2 else None
            row[m] = {"off": round(sum(a) / len(rs), 4), "on": round(sum(b) / len(rs), 4),
                      "diff": round(ci["diff"], 4) if ci else None,
                      "ci": [round(ci["lo"], 4), round(ci["hi"], 4)] if ci else None}
        out["by_category"][cat] = row
    for arm in RECALL_ARMS:
        lat = sorted(r["ms"][arm] for r in rows)
        out[f"{arm}_ms_p50"] = lat[len(lat) // 2] if lat else None
    out["rows"] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv and not argv[0].startswith("-") else "run"
    if cmd == "run":
        run(argv)
        return 0
    if cmd == "verify":
        ap = argparse.ArgumentParser()
        ap.add_argument("--set", default=str(SET_ROOT / DEFAULT_VERSION))
        a = ap.parse_args(argv)
        ms = load_set(Path(a.set), view="all")
        print(f"{ms.root}: verified v={ms.version} set_sha={ms.set_sha[:16]} dev={len(ms.dev)} "
              f"holdout={len(ms.holdout or [])} generator={ms.generator_model}")
        return 0
    if cmd == "prefetch-retrieval":
        prefetch_retrieval(argv)
        return 0
    if cmd == "recall-retrieval":
        recall_retrieval(argv)
        return 0
    if cmd == "export-sessions":
        ap = argparse.ArgumentParser()
        ap.add_argument("--set", default=str(SET_ROOT / DEFAULT_VERSION))
        ap.add_argument("--out", required=True)
        a = ap.parse_args(argv)
        ms = load_set(Path(a.set), view="tuning")
        mapping = export_sessions(ms.dev, Path(a.out))
        print(f"exported {len(mapping)} dev sessions to {a.out}")
        return 0
    raise SystemExit(f"unknown command {cmd!r} (run | verify | prefetch-retrieval | "
                     "recall-retrieval | export-sessions)")


if __name__ == "__main__":
    sys.exit(main())
