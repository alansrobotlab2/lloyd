#!/usr/bin/env python3
"""Induce the durable-write rubric from the training half of the #580 corpus (#4093).

PMR's non-training half (arXiv:2611.14017) on Lloyd's existing judge harness: one
reflection pass reads the TRAINING half's bad writes (with the repairer's own
recorded defect words and the pre-repair text), matched good writes, and the
current hand-authored ``judge.RUBRIC``, and writes at most eight diagnostic
criteria, each a check on an edit. Judge C (``judge.py --judge c``) is Judge A
with only that rubric span swapped. The GRPO credit-assignment half of PMR is not
reproduced here and cannot be on this box; nothing in a live write path changes.

    # 1. the split (deterministic, no model; also written by step 2)
    python eval/durable_write_judge/induce_rubric.py --split-only
    # 2. the one induction pass (one engine call), then the three arms and the score:
    #    see README.md "Induced-rubric arm (#4093)" for the exact commands in order

Outputs, beside this file unless ``--out-dir`` says otherwise:

- ``induce_split.json``  — the rule and both id lists (training -> induction input,
  held-out -> the only samples Judges A, B and C are scored on)
- ``induce_raw.json``    — the induction call's raw reply, verbatim, for audit
- ``induced_rubric.md``  — the criteria, rendered from that reply by
  ``render_rubric`` and read back by ``load_rubric``; never hand-edited

The validity condition of the whole experiment is that the induction call never
receives a held-out sample's text, label or defect word: ``induction_inputs`` drops
every held-out id and ``build_induction_prompt`` refuses one outright.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:  # imported as eval.durable_write_judge.induce_rubric
    from .build_corpus import DEFECT_CLASSES, sample_text
    from . import judge as jd
except ImportError:  # run as a script from this directory
    from build_corpus import DEFECT_CLASSES, sample_text
    import judge as jd

HERE = Path(__file__).parent

#: The split rule, stated once and recorded in the artifact. NOT repair_date: 17 of the
#: 20 bad samples share 2026-09-09 and the 30 good rows carry no repair_date at all
#: (#4093 triage), so a date cutoff leaves 1-2 bad samples on one side. `seq` parity
#: halves both labels exactly (10 bad + 15 good each) and puts every one of the five
#: classes on both sides of the shipped corpus.
SPLIT_RULE = "seq parity: odd seq -> train (induction input); even seq -> held_out (judged)"

#: The engine the induction pass and all three arms run on. The secondary (:8091) is
#: off since 2026-09-20 (GPU 2 is djev's), so the run goes to the primary, as #2465's
#: four-arm probe did; the reply's own `model` field is recorded as `served_model`.
DEFAULT_URL = "http://127.0.0.1:8096/v1/chat/completions"
DEFAULT_MODEL = "Qwen3.8-Flash-Next-nvfp4"

MAX_CRITERIA = 8
INDUCE_MAX_TOKENS = 3000
EXCERPT_BAD_CHARS = 1800
EXCERPT_GOOD_CHARS = 900

#: A write surface, from the note's vault path. PMR's second layer is per USER; Lloyd
#: has one user, so the layer binds to the surface a write lands on instead.
SURFACES = (("knowledge/youtube/", "digest_note"), ("memory/", "daily_memory"),
            ("USER.md", "identity"), ("MEMORY.md", "identity"), ("skills/", "skill_body"))

RUBRIC_BEGIN = "<!-- rubric:begin -->"
RUBRIC_END = "<!-- rubric:end -->"

#: Judge C's rubric span opens with this line, so the span is self-describing in a raw
#: prompt the way Judge A's "Defect classes, each with..." line is.
INDUCED_RUBRIC_OPEN = ("Diagnostic checks induced from past repairs of this agent's own "
                       "writes. Each is a check on an edit:")

_EDIT_SHAPE = re.compile(r"^when editing\b.+\bdistinguish\b", re.I | re.S)
#: A criterion that OPENS by describing a person ("The user prefers short answers") is
#: PMR's memory element, not its rubric criterion. Only the opening is tested: an edit
#: check may well mention what a user wants ("distinguish what the user wants today
#: from a standing preference" is the paper's own example of a good criterion).
_USER_SHAPE = re.compile(
    r"^(the\s+)?(user|alan|lloyd|writer|agent|author|speaker)\b.*?"
    r"\b(prefers|wants|likes|is|tends|always|never)\b", re.I | re.S)


class RubricInvalid(ValueError):
    """`induced_rubric.md` breaks the artifact contract; Judge C refuses to run on it."""


class HeldOutLeak(ValueError):
    """A held-out sample reached the induction input — the experiment's one fatal bug."""


# --- the split ----------------------------------------------------------------------

def surface_of(vault_path: str) -> str:
    for prefix, name in SURFACES:
        if vault_path.startswith(prefix) or vault_path.endswith(prefix):
            return name
    return "other"


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_split(samples: list[dict], *, corpus_digest: str = "") -> dict:
    """Partition every corpus id by `SPLIT_RULE`. Deterministic: ids and seq only."""
    train = sorted(s["id"] for s in samples if int(s["seq"]) % 2 == 1)
    held = sorted(s["id"] for s in samples if int(s["seq"]) % 2 == 0)

    def census(ids: list[str]) -> dict:
        by_id = {s["id"]: s for s in samples}
        classes: dict[str, int] = {}
        for i in ids:
            for c in by_id[i].get("defect_classes") or []:
                classes[c] = classes.get(c, 0) + 1
        return {"bad": sum(by_id[i]["label"] == "bad" for i in ids),
                "good": sum(by_id[i]["label"] == "good" for i in ids),
                "classes": dict(sorted(classes.items())),
                "surfaces": sorted({surface_of(by_id[i]["vault_path"]) for i in ids})}

    return {"item": "#4093", "rule": SPLIT_RULE, "corpus_sha256": corpus_digest,
            "n": len(samples), "train_feeds": "the one induction pass",
            "held_out_feeds": "Judges A, B and C, scored at equal n",
            "train_ids": train, "held_out_ids": held,
            "train_census": census(train), "held_out_census": census(held)}


def write_split(split: dict, path: Path) -> Path:
    path.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n")
    return path


def load_split(path: Path) -> dict:
    split = json.loads(Path(path).read_text())
    if set(split["train_ids"]) & set(split["held_out_ids"]):
        raise ValueError(f"{path}: the two halves share ids")
    return split


# --- the induction input ------------------------------------------------------------

def induction_inputs(samples: list[dict], split: dict) -> list[dict]:
    """The training half only, in id order. Every held-out sample is dropped here."""
    held = set(split["held_out_ids"])
    train = set(split["train_ids"])
    return sorted((s for s in samples if s["id"] in train and s["id"] not in held),
                  key=lambda s: s["id"])


def _render_bad(s: dict) -> str:
    words = []
    for cls, lines in (s.get("correction_evidence") or {}).items():
        words.append(f"  - {cls}: " + " | ".join(str(x)[:240] for x in lines[:3]))
    return (f"\n--- {s['id']} (bad; surface {surface_of(s['vault_path'])}) ---\n"
            f"defect classes the repairer recorded: {', '.join(s.get('defect_classes') or [])}\n"
            "what the repair changed, in the repairer's record:\n" + "\n".join(words) + "\n"
            f"the write as it stood before the repair (excerpt):\n"
            f"{sample_text(s)[:EXCERPT_BAD_CHARS]}\n")


def _render_good(s: dict) -> str:
    return (f"\n--- {s['id']} (good: never repaired; surface {surface_of(s['vault_path'])}) ---\n"
            f"{sample_text(s)[:EXCERPT_GOOD_CHARS]}\n")


INDUCE_ROLE = """You are improving the rubric a judge uses to audit durable knowledge notes \
that an autonomous agent wrote into its own long-term memory. Below are past writes from \
the TRAINING half of a labelled corpus: bad writes, each with the repairer's own record of \
what was wrong and the text as it stood before the repair, then good writes that were never \
repaired. The judge's current hand-authored rubric comes first, so you can see what it \
already covers."""

INDUCE_TASK = """Write AT MOST {max_criteria} diagnostic criteria that would have caught the \
bad writes without flagging the good ones. Rules:
- Each criterion is a CHECK ON AN EDIT, never a description of the writer or the user. \
Phrase each one exactly as "When editing <surface>, distinguish <A> from <B>."
- Write in your own words: do not quote the samples, the repairer's record, URLs or \
headings, and do not reuse the defect-class identifiers. A criterion that restates a label \
is worthless.
- Cite at least one sample id from this prompt that the criterion is induced from.
- `layer` is "shared" for a check that holds on every surface, or "surface:<name>" for one \
specific to a surface ({surfaces}).
- `maps_to` is the ONE existing defect class the criterion is a restatement of \
({classes}), or "none" when it names a defect none of them covers.
- Also list the defect modes you see in these samples that none of the five classes covers \
(an empty list is a valid answer). Look in particular for a dropped scope or time qualifier \
that turns a conditional or temporary statement into a standing one.

Your entire answer must be this one JSON object, with no prose before or after it:
{{"criteria": [{{"text": "When editing ..., distinguish ... from ....", "cites": ["bad-001"], \
"layer": "shared", "maps_to": "none"}}], \
"novel_modes": [{{"name": "...", "description": "...", "cites": ["bad-001"]}}]}}"""


def build_induction_prompt(train_samples: list[dict], split: dict) -> str:
    """The one reflection prompt. Refuses — never filters — a held-out sample."""
    held = set(split["held_out_ids"])
    leaked = sorted(s["id"] for s in train_samples if s["id"] in held)
    if leaked:
        raise HeldOutLeak(f"held-out sample(s) in the induction input: {leaked}")
    bad = [s for s in train_samples if s["label"] == "bad"]
    good = [s for s in train_samples if s["label"] == "good"]
    surfaces = sorted({surface_of(s["vault_path"]) for s in train_samples})
    return (INDUCE_ROLE
            + "\n\nThe current rubric:\n" + jd.RUBRIC
            + "\n\nBad writes from the training half:\n" + "".join(map(_render_bad, bad))
            + "\n\nGood writes from the training half:\n" + "".join(map(_render_good, good))
            + "\n\n" + INDUCE_TASK.format(
                max_criteria=MAX_CRITERIA,
                surfaces=", ".join(f"surface:{x}" for x in surfaces),
                classes=", ".join(sorted(DEFECT_CLASSES))))


# --- the reply -> the artifact ------------------------------------------------------

def _reply_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RubricInvalid("the induction reply carries no JSON object")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RubricInvalid(f"the induction reply's JSON does not parse: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("criteria"), list):
        raise RubricInvalid("the induction reply has no `criteria` list")
    return obj


def criterion_problems(text: str, cites: list[str], train_ids: set[str]) -> list[str]:
    """Why a criterion breaks the artifact contract; [] when it holds."""
    out = []
    if _USER_SHAPE.search(text.strip()):
        out.append("phrased as a description of the user, not of an edit")
    if not _EDIT_SHAPE.search(text.strip()):
        out.append("not phrased as 'When editing <surface>, distinguish A from B'")
    if not any(c in train_ids for c in cites):
        out.append("cites no training-half sample")
    return out


def curate(reply_text: str, samples: list[dict], split: dict) -> dict:
    """Validate the reply deterministically, apply the promotion rule, record every drop.

    Nothing here re-words a criterion. A cite that is not a training id is removed and
    recorded; a criterion left with no training cite, or not phrased as an edit check,
    is dropped and recorded; criteria past the eighth are dropped and recorded. The
    promotion rule: a `surface:` criterion whose cited samples span >= 2 surfaces moves
    to `shared`.
    """
    obj = _reply_object(reply_text)
    train_ids = set(split["train_ids"])
    by_id = {s["id"]: s for s in samples}
    kept, dropped = [], []
    for i, raw in enumerate(obj["criteria"], 1):
        text = " ".join(str(raw.get("text", "")).split())
        cites = [str(c) for c in (raw.get("cites") or [])]
        bad_cites = [c for c in cites if c not in train_ids]
        cites = [c for c in cites if c in train_ids]
        problems = criterion_problems(text, cites, train_ids)
        if problems or len(kept) >= MAX_CRITERIA:
            dropped.append({"index": i, "text": text, "why": problems or
                            [f"past the {MAX_CRITERIA}-criterion cap"]})
            continue
        layer = str(raw.get("layer") or "shared").strip()
        if layer != "shared" and not layer.startswith("surface:"):
            layer = "shared"
        spans = sorted({surface_of(by_id[c]["vault_path"]) for c in cites})
        if layer.startswith("surface:") and len(spans) >= 2:
            decision = f"promoted {layer} -> shared (cited samples span {len(spans)} surfaces)"
            final = "shared"
        else:
            decision = (f"kept {layer} (cited samples span {len(spans)} surface"
                        f"{'s' if len(spans) != 1 else ''}: {', '.join(spans)})")
            final = layer
        maps_to = str(raw.get("maps_to") or "none").strip()
        if maps_to not in DEFECT_CLASSES:
            maps_to = "none"
        kept.append({"text": text, "cites": cites, "layer": final, "induced_layer": layer,
                     "promotion": decision, "maps_to": maps_to,
                     "removed_cites": bad_cites})
    novel = []
    for m in obj.get("novel_modes") or []:
        novel.append({"name": " ".join(str(m.get("name", "")).split()),
                      "description": " ".join(str(m.get("description", "")).split()),
                      "cites": [str(c) for c in (m.get("cites") or []) if str(c) in train_ids]})
    if not kept:
        raise RubricInvalid(f"no criterion survived validation: {dropped}")
    return {"criteria": kept, "dropped": dropped, "novel_modes": novel}


def render_rubric(curated: dict, *, split: dict, model: str, served_model: str,
                  url: str) -> str:
    lines = [
        "# Induced durable-write rubric — #4093",
        "",
        "<!-- Generated by eval/durable_write_judge/induce_rubric.py from induce_raw.json."
        " Do not hand-edit: Judge C reads the span between the rubric markers. -->",
        "",
        f"- split: `induce_split.json` — {split['rule']}; induced from "
        f"{len(split['train_ids'])} training ids only",
        f"- engine: `{url}`, model asked `{model}`, served `{served_model or 'not reported'}`",
        "- one pass, temperature 0, no re-tuning (#4093 step 6)",
        "",
        "## Criteria",
        "",
        RUBRIC_BEGIN,
    ]
    for i, c in enumerate(curated["criteria"], 1):
        lines += [f"{i}. [{c['layer']}] {c['text']}",
                  f"   - cites: {', '.join(c['cites'])}",
                  f"   - maps_to: {c['maps_to']}"]
    lines += [RUBRIC_END, "", "## Promotion decisions", ""]
    for i, c in enumerate(curated["criteria"], 1):
        lines.append(f"- {i}: {c['promotion']}")
    lines += ["", "## Defect modes absent from the five classes", ""]
    if curated["novel_modes"]:
        for m in curated["novel_modes"]:
            lines.append(f"- {m['name']}: {m['description']} (cites: "
                         f"{', '.join(m['cites']) or 'none'})")
    else:
        lines.append("- none named by the induction pass")
    lines += ["", "## Dropped at validation", ""]
    for d in curated["dropped"]:
        lines.append(f"- reply criterion {d['index']}: {'; '.join(d['why'])} — {d['text']}")
    removed = [(i, c["removed_cites"]) for i, c in enumerate(curated["criteria"], 1)
               if c["removed_cites"]]
    for i, ids in removed:
        lines.append(f"- criterion {i}: removed non-training cite(s) {', '.join(ids)}")
    if not curated["dropped"] and not removed:
        lines.append("- none")
    return "\n".join(lines) + "\n"


_CRIT_RE = re.compile(r"^(\d+)\. \[([^\]]+)\] (.+)$")


def parse_rubric(text: str) -> dict:
    """Read `induced_rubric.md` back. Structure only; `validate_rubric` judges content."""
    if RUBRIC_BEGIN not in text or RUBRIC_END not in text:
        raise RubricInvalid("no rubric markers")
    span = text.split(RUBRIC_BEGIN, 1)[1].split(RUBRIC_END, 1)[0]
    criteria: list[dict] = []
    for line in span.splitlines():
        m = _CRIT_RE.match(line)
        if m:
            criteria.append({"n": int(m.group(1)), "layer": m.group(2).strip(),
                             "text": m.group(3).strip(), "cites": [], "maps_to": None})
        elif line.strip().startswith("- cites:") and criteria:
            criteria[-1]["cites"] = [c.strip() for c in
                                     line.split(":", 1)[1].split(",") if c.strip()]
        elif line.strip().startswith("- maps_to:") and criteria:
            criteria[-1]["maps_to"] = line.split(":", 1)[1].strip()
    promotions: dict[int, str] = {}
    novel: list[str] = []
    section = ""
    for line in text.split(RUBRIC_END, 1)[1].splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = re.match(r"^- (\d+): (.+)$", line)
        if section == "Promotion decisions" and m:
            promotions[int(m.group(1))] = m.group(2)
        elif section.startswith("Defect modes absent") and line.startswith("- "):
            novel.append(line[2:])
    return {"criteria": criteria, "promotions": promotions, "novel_modes": novel,
            "has_promotion_section": "## Promotion decisions" in text,
            "has_novel_section": "## Defect modes absent from the five classes" in text}


def validate_rubric(parsed: dict, split: dict) -> None:
    """The artifact contract (#4093 clause 2). Raises `RubricInvalid` on the first break."""
    crits = parsed["criteria"]
    if not crits:
        raise RubricInvalid("no criteria")
    if len(crits) > MAX_CRITERIA:
        raise RubricInvalid(f"{len(crits)} criteria; at most {MAX_CRITERIA}")
    train_ids = set(split["train_ids"])
    for c in crits:
        problems = criterion_problems(c["text"], c["cites"], train_ids)
        if problems:
            raise RubricInvalid(f"criterion {c['n']}: {'; '.join(problems)}")
        if c["layer"] != "shared" and not c["layer"].startswith("surface:"):
            raise RubricInvalid(f"criterion {c['n']}: layer {c['layer']!r} is neither "
                                "shared nor surface:<name>")
        if c["n"] not in parsed["promotions"]:
            raise RubricInvalid(f"criterion {c['n']}: no promotion decision")
    if not parsed["has_novel_section"]:
        raise RubricInvalid("no 'Defect modes absent from the five classes' section")


def load_rubric(path: Path, split: dict) -> dict:
    parsed = parse_rubric(Path(path).read_text())
    validate_rubric(parsed, split)
    return parsed


def rubric_span(parsed: dict) -> str:
    """Judge C's rubric text: the criteria alone, in place of `judge.RUBRIC`."""
    return INDUCED_RUBRIC_OPEN + "\n\n" + "\n".join(
        f"{i}. {c['text']}" for i, c in enumerate(parsed["criteria"], 1))


# --- the leakage check (printed by score.py's three-arm report) ----------------------

_WORD = re.compile(r"[a-z0-9]+")
NGRAM = 8


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _grams(words: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _defect_words(samples: list[dict]) -> tuple[set[str], set[tuple[str, ...]]]:
    """Short evidence strings verbatim, long ones by 8-gram, plus the class labels."""
    short, grams = set(), set()
    for s in samples:
        for cls, lines in (s.get("correction_evidence") or {}).items():
            short.add(cls.lower())
            short.add(cls.replace("_", " ").lower())
            for ln in lines:
                w = _words(str(ln))
                if len(w) >= NGRAM:
                    grams |= _grams(w)
                elif w:
                    short.add(" ".join(w))
    return short, grams


def leakage_check(criteria: list[dict], samples: list[dict], split: dict) -> list[dict]:
    """Per criterion: does a training defect word or sample phrase appear verbatim in it,
    and could it have been written from the held-out half's labels?

    `train_hits`: a training sample's recorded defect word (a class label, a short
    evidence string such as a heading or URL) or any 8-word run of a training sample's
    text or evidence, verbatim (case- and punctuation-folded). `held_out_hits`: a
    held-out id, or a held-out defect word / 8-word run that the training half does not
    also contain — something the induction pass had no way to see. Either flags the
    criterion, and one flagged criterion makes the run's margin uninterpretable.
    """
    by_id = {s["id"]: s for s in samples}
    train = [by_id[i] for i in split["train_ids"]]
    held = [by_id[i] for i in split["held_out_ids"]]
    t_short, t_grams = _defect_words(train)
    h_short, h_grams = _defect_words(held)
    for s in train:
        t_grams |= _grams(_words(sample_text(s)))
    for s in held:
        h_grams |= _grams(_words(sample_text(s)))
    h_short -= t_short
    h_grams -= t_grams
    out = []
    for i, c in enumerate(criteria, 1):
        words = _words(c["text"])
        joined = " " + " ".join(words) + " "
        grams = _grams(words)
        train_hits = sorted({w for w in t_short if f" {w} " in joined}
                            | {" ".join(g) for g in grams & t_grams})
        held_hits = sorted({h for h in split["held_out_ids"] if h in c["text"]}
                           | {w for w in h_short if f" {w} " in joined}
                           | {" ".join(g) for g in grams & h_grams})
        out.append({"n": i, "text": c["text"], "train_hits": train_hits,
                    "held_out_hits": held_hits,
                    "flagged": bool(train_hits or held_hits)})
    return out


# --- the CLI --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(HERE / "corpus.jsonl"))
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split-only", action="store_true",
                    help="write induce_split.json and stop; no model")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the split and print the induction prompt's size; no model")
    ap.add_argument("--from-reply", default="",
                    help="re-render induced_rubric.md from a saved induce_raw.json; no model")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)
    samples = [json.loads(ln) for ln in corpus.read_text().splitlines() if ln.strip()]
    split = make_split(samples, corpus_digest=corpus_sha256(corpus))
    print(f"split: {len(split['train_ids'])} train / {len(split['held_out_ids'])} held out "
          f"-> {write_split(split, out_dir / 'induce_split.json')}")
    if args.split_only:
        return 0

    prompt = build_induction_prompt(induction_inputs(samples, split), split)
    if args.dry_run:
        print(f"[dry-run] induction prompt: {len(prompt)} chars, 1 call, "
              f"max_tokens {INDUCE_MAX_TOKENS}")
        return 0

    raw_path = out_dir / "induce_raw.json"
    if args.from_reply:
        raw = json.loads(Path(args.from_reply).read_text())
    else:
        reply = jd.call_engine(prompt, url=args.url, model=args.model,
                               timeout=900, max_tokens=INDUCE_MAX_TOKENS)
        raw = {"url": args.url, "model": args.model, "prompt_chars": len(prompt),
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **reply}
        raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
        print(f"raw reply -> {raw_path} (finish_reason {raw.get('finish_reason')})")
    try:
        curated = curate(raw.get("content") or "", samples, split)
    except RubricInvalid as exc:
        print(f"REFUSED: {exc} — the one pass produced no usable rubric; "
              "per the stop rule this is the result, not a reason to re-prompt",
              file=sys.stderr)
        return 2
    text = render_rubric(curated, split=split, model=raw.get("model", args.model),
                         served_model=raw.get("served_model", ""), url=raw.get("url", args.url))
    validate_rubric(parse_rubric(text), split)
    out = out_dir / "induced_rubric.md"
    out.write_text(text)
    print(f"{len(curated['criteria'])} criteria ({len(curated['dropped'])} dropped) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
