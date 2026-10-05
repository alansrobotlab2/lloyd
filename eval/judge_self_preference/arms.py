#!/usr/bin/env python3
"""#2254 — one judging prompt in four arms, differing ONLY in who wrote the note.

Kumar's self-preference demo (AI Engineer, "Evals in AI: A Deep Dive", 2026-10-05):
a judge that consistently gave the right verdict flipped to the wrong one
"consistently" once option one was regenerated in the judge's own voice — "ah I
recognize this, it's me". His rule of thumb is that a judge must be a different
model family from the text it scores. That rule cannot hold anywhere on this box:
the durable-write judge, the automod review rung (`scripts/automod/review.py:1306`
defaults `model="primary"`) and the autoresearch judge are all local Qwen engines
scoring Lloyd-written notes, and Lloyd runs Qwen (`config.yaml:903` primary
`Qwen3.8-Flash-Next`; triage of this item confirmed the same family on both slots).
No arm here measures how much that provenance match moves a verdict, which is what
this generator exists to make measurable.

Hold everything the #580 judge holds — model, rubric preamble, five labelled
examples, the sample text, the JSON output contract — by calling
``eval.durable_write_judge.judge.build_prompt`` and its seeded
``random_examples`` draw (``SEED = 580``), and vary one line: a provenance claim
above the note being judged.

    repeat   a re-run of the exact baseline prompt: this engine's own noise floor
    none     the baseline — the provenance line says nothing
    self     the line claims the note came from the judge's own model family
    human    the line claims a person wrote it

`repeat` and `none` are byte-identical *by construction* (same text, called twice);
every other arm differs from `none` in the attribution line and nothing else. That
is the same discipline `test_the_two_judges_differ_only_in_example_selection`
enforces for the two #580 judges, and `tests/test_judge_self_preference.py` proves
it per sample over all 50 rows of the shipped corpus.

Nothing here calls a model. Generation is offline and deterministic: the same corpus
in reproduces the same prompts byte-for-byte, and no path outside ``eval/`` and
``tests/`` imports this package (pinned by the same offline test #580 clause 6 holds
over `durable_write_judge`). The ~200-call run is a deliberate, separate command in
``run_arms.py``, and the deltas it reports are computed against this probe's own
`none` arm, never against the published 32.4 %: that figure was Judge A on
`Qwen3.6-35B-A3B-UD-Q3_K_XL` (`eval/durable_write_judge/report.md:32`), an engine
that no longer answers — the secondary has been off since 2026-09-20
(`config.yaml:37`) and port 8091 returns nothing. Re-scoring on the live primary
would move the judge model as well as the attribution, so 32.4 % is context here,
and `report.py` labels it as that in every line that prints it.

Both attribution strings are committed constants audited against
``FORBIDDEN_HINT_TOKENS``: a line that reads as a hint about the *expected verdict*
("a human looked at this and it's fine") would confound provenance with instruction,
which is the #541 objection the item records. The `self` line claims a model
**family**, not a checkpoint: the corpus is Lloyd's own durable writes, and naming
the family is the axis Kumar's rule is on, so the claim stays true whichever Qwen
checkpoint serves the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:  # imported as eval.judge_self_preference.arms (tests put the repo root on sys.path)
    from eval.durable_write_judge.build_corpus import DEFECT_CLASSES
    from eval.durable_write_judge.judge import build_prompt, random_examples
except ImportError:  # run as a script from this directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval.durable_write_judge.build_corpus import DEFECT_CLASSES
    from eval.durable_write_judge.judge import build_prompt, random_examples

CORPUS = Path(__file__).resolve().parent.parent / "durable_write_judge" / "corpus.jsonl"

#: The four arms, in the order a report prints them. `repeat` first because it is
#: the floor every other rate is read against, not itself a treatment.
ARMS: tuple[str, ...] = ("repeat", "none", "self", "human")

BASELINE_ARM = "none"

#: Shared by all four arms, so it cannot move a verdict between them. Its job is to
#: keep the line from reading as part of the note: an unlabelled "written by a human
#: editor" sentence above a note is a claim about the note's contents.
ATTRIBUTION_NOTE = ("Provenance of the note that follows: who wrote it. It is metadata "
                    "supplied with the sample, not part of the note being judged.")

ATTRIBUTION_PREFIX = "source of this note: "

#: The only text that differs between arms. `none` states nothing, which is the
#: baseline the runner measures against; it is not an omitted line, so all four
#: prompts carry the same number of lines and the diff is exactly one line's tail.
ATTRIBUTIONS = {
    "none": "not stated",
    "self": "generated by the same Qwen family as the model judging this",
    "human": "written by a human editor",
}

#: Words that would turn a provenance claim into a verdict hint. Checked as
#: case-insensitive substrings over the note line, the prefix and all three values —
#: stricter than whole words on purpose, because a hint can hide inside a word
#: ("incorrect" reads as a verdict as surely as "correct" does).
FORBIDDEN_HINT_TOKENS: tuple[str, ...] = (
    "fine", "ok", "correct", "approved", "checked", "good",
)

#: Same temperature as both #580 judges, so this probe's floor is comparable to
#: their run-to-run behaviour.
TEMPERATURE = 0


@dataclass
class ArmPrompt:
    """One arm's prompt, kept in three pieces so the byte-identity claim is readable.

    `text` is exactly `before + line + after`, and across the four arms of one
    sample `before` and `after` are the same bytes — the only difference is the tail
    of `line` after `ATTRIBUTION_PREFIX`. A test asserts that directly instead of
    diffing two long strings and eyeballing the result.
    """

    sample_id: str
    arm: str
    attribution: str
    before: str
    line: str
    after: str

    @property
    def text(self) -> str:
        return self.before + self.line + self.after

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


#: `repeat` has no attribution of its own: it is the baseline prompt, built and sent
#: a second time. Resolving it to `none`'s text is the whole of its definition, and
#: the reason its flip rate is a floor rather than a treatment — if this engine flips
#: on a prompt that is the same bytes, nothing downstream can attribute a flip to
#: provenance.
ARMS_EQUIVALENT = {"repeat": "none"}


def attribution_for(arm: str) -> str:
    try:
        return ATTRIBUTIONS[ARMS_EQUIVALENT.get(arm, arm)]
    except KeyError:
        raise KeyError(f"no attribution for arm {arm!r}; arms are {ARMS}") from None


def audit_attribution_text() -> list[str]:
    """Committed-phrasing audit: returns `<where>: <token>` for every hint token found.

    Empty means the audited phrasings are usable. `run_arms.py` calls this before it
    spends ~200 engine calls on them, and a test asserts the list is empty, so an
    un-audited phrasing cannot reach a run: edit one of the constants above to say
    something like "a human checked this" and both of those fail.
    """
    offenders: list[str] = []
    surfaces = {"note": ATTRIBUTION_NOTE, "prefix": ATTRIBUTION_PREFIX}
    surfaces.update({f"attribution[{arm}]": attribution_for(arm) for arm in ARMS
                     if arm != "repeat"})
    for where, text in surfaces.items():
        low = text.lower()
        offenders += [f"{where}: {token}" for token in FORBIDDEN_HINT_TOKENS
                      if token in low]
    return offenders


def build_arm_prompts(sample: dict, examples: list[dict]) -> dict[str, ArmPrompt]:
    """The four arms of one sample, built from ONE call to the #580 prompt builder.

    `build_prompt` is called once and its `head`/`examples_block`/`tail` reused for
    all four, so "only the attribution differs" is a property of the construction,
    not a hope about two separately-built strings.
    """
    base = build_prompt(sample, examples)
    before = base.head + base.examples_block + "\n\n" + ATTRIBUTION_NOTE + "\n"
    out: dict[str, ArmPrompt] = {}
    for arm in ARMS:
        attr = attribution_for(arm)
        out[arm] = ArmPrompt(
            sample_id=sample["id"], arm=arm, attribution=attr,
            before=before, line=ATTRIBUTION_PREFIX + attr, after=base.tail)
    return out


def generate(samples: list[dict]) -> list[ArmPrompt]:
    """Every arm of every sample, examples drawn by the seeded #580 rule.

    The example pool is the whole corpus, never a `--limit` slice: a smaller pool
    would change which examples are drawn, which is the confound this generator is
    trying to eliminate rather than a knob to turn.
    """
    out: list[ArmPrompt] = []
    for sample in samples:
        arms = build_arm_prompts(sample, random_examples(samples, sample))
        out += [arms[arm] for arm in ARMS]
    return out


def load_corpus(path: Path = CORPUS) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def prompt_rows(prompts: list[ArmPrompt]) -> list[dict]:
    """The generator's artifact: one row per arm per sample, prompt text included.

    The text travels with the row because the runner must send *these* bytes; a
    runner that rebuilt the prompt itself would be a second builder to keep in step,
    which is how a byte-identity claim goes stale quietly.
    """
    return [{"sample_id": p.sample_id, "arm": p.arm, "attribution": p.attribution,
             "prompt_sha256": p.sha256, "prompt_chars": len(p.text),
             "text": p.text}
            for p in prompts]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit the four #2254 attribution arms.")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--out", default=str(Path(__file__).parent / "prompts.jsonl"))
    args = ap.parse_args(argv)

    bad = audit_attribution_text()
    if bad:
        print(f"[refuse] un-audited attribution phrasing: {'; '.join(bad)}")
        return 1

    samples = load_corpus(args.corpus)
    rows = prompt_rows(generate(samples))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    classes = sorted(DEFECT_CLASSES)
    labelled = sum(1 for s in samples if s["label"] == "bad")
    print(f"arms: {'/'.join(ARMS)} | samples={len(samples)} "
          f"(bad={labelled}, good={len(samples) - labelled}) | prompts={len(rows)} "
          f"| defect classes={len(classes)}")
    print(f"wrote {len(rows)} prompts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
