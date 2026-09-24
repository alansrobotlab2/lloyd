#!/usr/bin/env python3
"""IFScale-style instruction-density compliance sweep for Lloyd's engine slots.

Backlog #630. IFScale (2025) asks a model to write a document that must contain
N exact required words and scores how many appear verbatim; Laurie Voss's 2026
replication found frontier models breaking between ~2,000 and ~5,000
instructions, and failing in four distinguishable shapes: monotone forgetting,
an API safety refusal, thinking-budget exhaustion, and a polite stop that reads
as a finished answer. Lloyd had no arm that measures any of this — the eval
inventory is retrieval, tool-choice, prefetch, preserved-thinking A/B,
parallel-dispatch, counterfactual retrieval and dispatch/review-grader
calibration, and the nearest-named one disclaims the claim outright
(`eval/run_skill_dispatch_probe.py:13-16`: a rule that fires proves the protocol
reached the model, *not that the model obeyed*).

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
Per (engine, N): the fraction of the N required words that appear in the
model's output verbatim, on a word boundary, case-insensitively. No LLM judge
anywhere in the scoring path — an instruction-compliance arm graded by a model
would measure the grader.

It is a proxy. Prose rules with conditionals may not track word-inclusion;
that is what #630's follow-on (real SKILL.md constraints through the same
scorer) is for. And N* is a capacity ceiling, not a licence: Voss's own
conclusion is that capacity up != reliability up, so a high N* says length caps
are not the binding constraint, not that the prompt surface may grow.

TWO METRICS, BECAUSE ONE OF THEM IS GAMED BY THE MODEL
------------------------------------------------------
The first sweep run on this engine found that at every density from N=400 up,
the report ends with the required-word list pasted back as a trailing block.
Verbatim inclusion scores that as compliance — IFScale counts the words, and
the words are there. It is not the same measurement: one run holds N rules
while writing, the other can type. So every record carries `compliance`
(verbatim inclusion, the metric IFScale and #630's acceptance name, and the one
comparable to the published cloud numbers) and `prose_compliance` (the same
denominator with any line that IS the dumped list removed), and the baseline
records two ceilings beside each other: `n_star_95` and `n_star_95_prose`. For
"may the prompt/skill surface grow" and "was the rule honoured", cite the prose
one and keep the other for comparison with the paper. `echo_runs` says how many
runs were affected, so the size of the gap is visible without reading records.

A SECOND ENGINE ROW THAT IS THE SAME ENGINE
------------------------------------------
`resolve_model_alias("secondary")` rewrites the alias to `primary` whenever
`secondary_enabled` is false (`app/config.py:361`), which is why an eval that
resolves through it and records rows per *alias* would produce a baseline with
two engines in it and one engine on the machine — the exact defect that
function's own docstring documents for Inner Voice ("ran on primary the whole
time... nothing in any log said so"). So `resolve_slot` reads the SLOT's
configured `base_url` and asks the engine what it is serving, and a slot
switched off in config.yaml is reported `available: false` with the flag that
switched it off. It never silently measures a different engine under the
slot's label.

THINKING IS A KNOB THIS EVAL HAS TO SET
---------------------------------------
With the reasoning channel open, the SMALLEST density in this sweep — N=50 —
spent the entire 6,000-token budget inside `reasoning` and returned
`content: ""`, `finish_reason: "length"`, in 58.75 s. That is Voss's
thinking-budget exhaustion shape, appearing locally at the easy end of the
axis, and a curve measured that way is a curve of the reasoning budget rather
than of instruction density. So the default closes thinking on every request
(`--thinking on` reopens it to measure the exhaustion shape deliberately), the
value is recorded in the baseline and on every record, and the score reads
`message.content` and never `message.reasoning` — scoring the reasoning field
would credit the model for deliberating about the constraint.

OUTPUT — AND WHY IT IS NOT `eval/baselines/`
--------------------------------------------
The run's own artifact goes to `app.paths.EVAL_BASELINES_DIR`, which resolves
under the data root (`~/lloyd-data/eval/baselines/`), the destination
`eval/run_eval.py:1135` and `run_prefetch_eval.py:212` use. It cannot go to a
path in the repo called `eval/baselines/`: `.gitignore:100` ignores exactly
that directory (`git check-ignore -v eval/baselines/x.json` names the rule), so
the "committed baseline JSON" #630's acceptance asks for would be an
uncommittable file — and `.gitignore` is in the self-modification loop's
DENIED_GLOBS (`scripts/automod/spec.py:97`), so the rule is not this round's to
move. The committed copy is therefore a JSONL projection of the same records
under `eval/instruction-compliance/`, written with `--committed-copy`, holding
meta/config/engine/summary/records rows. Same numbers, one record per line,
greppable and diffable.

REPRO — one line, from the repo root (the secondary slot last, one request at a
time, which is what #630 asks for and also keeps a measurement sweep from being
a load test on the engine every live turn is using):

    .venvs/lloyd/bin/python eval/instruction_compliance_eval.py --label ifscale --seed 630 --sweep 50,100,200,400,800,1600 --repeats 2 --max-tokens 6000 --temperature 0.0 --thinking off --engines primary,secondary

That is the shape of the `repro_command` recorded inside every baseline this
script writes, and each per-(engine, N) point carries the `reproduction_band` a
re-run must land inside — 2x the spread between that run's own repeats at that
density, floored at `MIN_REPRODUCTION_BAND` (0.14, the drift three identical
sweeps of this command showed between runs), so the band is measured rather than
assumed.
Temperature 0 is the default because a repro contract that allows resampling
cannot be re-checked by a reader; it does not make the engine deterministic
(vLLM's continuous batching does not), which is exactly why the band exists.

WORDS
-----
`eval/instruction_compliance_vocab.txt` is the fixed vocabulary, committed, and
its sha256 goes into every baseline record, so "same seed, same word list" is
checkable against the artifact rather than argued. A run samples from it with
`random.Random(seed)`; the same (vocab, N, seed) is byte-identical on any later
day. No required word is a word of the prompt frame — the frame is built from
`PROMPT_FRAME`, and `tests/test_instruction_compliance_eval.py` asserts the two
token sets are disjoint, so the scorer cannot be handed a word the instruction
itself already supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOCAB_PATH = HERE / "instruction_compliance_vocab.txt"

#: The six density points #630 specifies. 50 is comfortably inside anything a
#: model can do and 1600 is where the 2026 frontier cloud numbers start to
#: bend; the interesting part of the curve is between them.
SWEEP_N: tuple[int, ...] = (50, 100, 200, 400, 800, 1600)


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated density list, tolerating whitespace and blanks."""
    out: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except ValueError as e:
            raise ValueError(f"bad sweep value {token!r} (expected an integer)") from e
    return out

#: The engine slots, in the order the sweep runs them. #630 asks for secondary
#: last and serially; `SLOT_PROGRAMS` says which supervisord program owns each
#: slot so a disabled slot can name its flag instead of reading as a timeout.
SLOT_ALIASES: tuple[str, ...] = ("primary", "secondary")
SLOT_PROGRAMS: dict[str, str | None] = {
    "primary": "agent-llm-primary",      # no switch in config.yaml; always on
    "secondary": "agent-llm-secondary",  # governed by `secondary_enabled`
}

DEFAULT_SEED = 630
DEFAULT_LABEL = "ifscale"
TARGET_REPORT_WORDS = 800

#: Output budget. A thinking model spends tokens in `reasoning` before a single
#: character of `content` exists (measured on this engine: an 8-token request
#: came back with `content: null` and `finish_reason: "length"`), so a budget
#: sized to the report alone truncates inside the thinking phase and a
#: truncation scored as forgetting is the failure this eval must not have.
#: Every request carries this value explicitly and records it; the number is
#: not read from config — `config.yaml`'s `max_tokens: 8192` is
#: `finalizer.max_tokens`, the structured final-answer restatement, not a cap
#: a direct engine call inherits.
DEFAULT_MAX_TOKENS = 6000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 600.0

#: Whether the reasoning channel is open, and why the default is the closed
#: one. Measured on this box before the sweep existed, 2026-09-24: with thinking
#: open, the SMALLEST density point in this sweep — N=50 — spent the entire
#: 6,000-token budget inside `reasoning` and returned `content: ""` with
#: `finish_reason: "length"` in 58.75 s. That is one of the four shapes Voss
#: reported (thinking-budget exhaustion) appearing locally at the easy end of
#: the axis, and it is a finding about the knob, not about instruction density:
#: a curve measured that way is a curve of the reasoning budget, flat at zero
#: compliance. So the default closes thinking and asks the model to obey N rules;
#: `--thinking on` reopens it to measure the exhaustion shape instead. The value
#: is recorded in the baseline and in every record, and is not inferable from
#: the config file.
DEFAULT_THINKING = "off"

#: N* is the largest density at which compliance still clears this.
N_STAR_THRESHOLD = 0.95

#: Floor for the reproduction band, used when the observed repeat spread is
#: smaller than it (`band_from_spread`). This is a DECLARED floor, not a quantity
#: the tree can re-measure: three sweeps of the identical command on 2026-09-24
#: (same seed, same vocabulary, temperature 0, `--repeats 2`) moved N=50's mean
#: compliance by up to 0.0625 and N=1600's by 0.0469, so 2x the largest is 0.125
#: and the floor rounds that to 0.14. Two of those three sweeps were overwritten
#: by the next run of the same command, so a reader holding only this tree can
#: re-derive the OTHER half — each point's within-run spread, committed in the
#: artifact — and re-running the repro command re-measures this constant rather
#: than checking it. Why a floor exists at all: at N=50 one missing word is 0.02
#: of the score and the engine is a continuously-batched vLLM server, so a band
#: built only from the two repeats inside one run would be a band its own re-runs
#: violate, which files a real regression as noise. Per-point bands stay tighter,
#: and that is the one to test a dense point with.
MIN_REPRODUCTION_BAND = 0.14

# ── Prompt ──────────────────────────────────────────────────────────────

#: The fixed frame around the required-word list. `build_prompt` is the only
#: place it is assembled, and the vocabulary must be disjoint from its tokens
#: — see the module docstring's WORDS section.
PROMPT_FRAME = (
    'TASK: write a business report of about {target_words} words titled '
    '"{title}".\n\n'
    'CONSTRAINT: the report must contain each of these {n} words verbatim, '
    'at least once each:\n{word_block}\n\n'
    'RULES:\n'
    '- Output only the report itself: a title line, then prose paragraphs.\n'
    '- Do not reproduce the constraint list, and do not add a section that '
    'just enumerates the words.\n'
    '- Use a required word inside a sentence where it fits naturally.\n'
)
REPORT_TITLE = "Quarterly Operations Review"

#: How the list is rendered. Two words per line keeps the frame's own token
#: count flat in N (the list is what grows), and the fixed width means the
#: prompt's shape does not depend on the words sampled.
WORDS_PER_LINE = 2


def load_vocab(path: Path = VOCAB_PATH) -> tuple[str, ...]:
    """The fixed required-word vocabulary, in file order.

    `#` lines are provenance comments, not words. Raises rather than sampling
    from a missing or empty list: a sweep that silently ran on an empty
    vocabulary would report 100 % compliance at every density.
    """
    if not path.exists():
        raise FileNotFoundError(f"required-word vocabulary missing: {path}")
    words = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    if not words:
        raise ValueError(f"required-word vocabulary is empty: {path}")
    return tuple(words)


def vocab_sha256(path: Path = VOCAB_PATH) -> str:
    """sha256 of the vocabulary file's bytes — the artifact the sample came from."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_words(vocab: tuple[str, ...], n: int, seed: int) -> list[str]:
    """N required words, deterministically. Same (vocab, n, seed) => same bytes.

    Sampling without replacement: a repeated required word would be one less
    constraint than N claims, and N is the x-axis of the whole measurement.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n > len(vocab):
        raise ValueError(f"n={n} exceeds the fixed vocabulary ({len(vocab)} words); "
                         "the sweep cannot sample without replacement")
    return random.Random(seed).sample(list(vocab), n)


def build_prompt(words: list[str], target_words: int = TARGET_REPORT_WORDS) -> str:
    """One prompt: a ~`target_words`-word report that must contain every word."""
    rows = [", ".join(words[i:i + WORDS_PER_LINE])
            for i in range(0, len(words), WORDS_PER_LINE)]
    return PROMPT_FRAME.format(target_words=target_words, title=REPORT_TITLE,
                               n=len(words), word_block="\n".join(rows))


def frame_tokens() -> frozenset[str]:
    """Tokens the rendered prompt frame supplies, before any required word.

    Rendered with its slots filled (the `{n}` and `{target_words}` slots come
    out as digits, and the title is fixed prose) rather than read off the raw
    template, so the set is exactly what the model is being *told* and not the
    placeholder names. A required word inside it would be free, and the sweep
    would report a compliance point it never asked for.
    `tests/test_instruction_compliance_eval.py` holds the disjointness
    assertion that keeps the vocabulary clear of it.
    """
    rendered = PROMPT_FRAME.format(target_words=TARGET_REPORT_WORDS,
                                   title=REPORT_TITLE, n=0, word_block="")
    return frozenset(re.findall(r"[a-z]+", rendered.lower()))


# ── Scoring ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=8192)
def _boundary(word: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(word) + r"\b")


def present_words(text: str, words: list[str]) -> list[str]:
    """Required words found verbatim on a word boundary, case-insensitively.

    Word-boundary, not substring: `praised` must not be credited for `praise`,
    and a required word embedded in a longer word is not the word.
    """
    low = text.lower()
    return [w for w in words if _boundary(w).search(low)]


def score_compliance(text: str, words: list[str]) -> dict[str, Any]:
    """The mechanical score: hits, misses, fraction. No judge, no threshold."""
    hits = set(present_words(text, words))
    n = len(words)
    return {"required": n, "present": len(hits),
            "missing": [w for w in words if w not in hits],
            "compliance": (len(hits) / n) if n else 0.0}


REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(i cannot|i can't|i'm unable|i am unable|i'm not able|"
               r"i won't|i will not|i refuse|i decline|i must decline|"
               r"i'm sorry, but i cannot|i'm sorry, but i can't)\b", re.I),
    re.compile(r"\b(as an ai|i'm an ai|as a language model|i cannot comply|"
               r"i can't comply|i won't comply|i cannot fulfill)\b", re.I),
    re.compile(r"\b(unable to (fulfill|comply|complete|continue|provide)|"
               r"cannot (fulfill|comply|complete|continue))\b", re.I),
)

#: A run is `refused` only when the refusal IS the output: a report that opens
#: with "I can't include all 1,600 of those, but here is the review:" and then
#: delivers 700 words is a compliance datapoint, not a refusal. The ceiling is
#: a fifth of the asked-for length — past that the model produced substance,
#: whatever it called it.
REFUSED_MAX_WORDS = TARGET_REPORT_WORDS // 5

#: A voluntary stop this far short of the asked-for length, with no refusal and
#: no length cutoff, is the model answering a different question than the one
#: asked. Truncation by the token budget is caught earlier, off
#: `finish_reason`, so this cannot double-count it.
OFFTASK_MAX_FRACTION = 0.25

#: The one spelling of every failure label. `classify_run` returns these names,
#: the records carry them, and the ceiling filter in `COMPARABLE_CLASSES` matches
#: on them, so every one of them has to come from this block. The review rung of
#: gate SM_20260924_040055 is why: that commit's filter carried `off_task` and
#: `late_refusal` while `classify_run` emitted `off-task` and `late-refusal`, so
#: the two names matched nothing, a density whose only failing run was a 0.10
#: late-refusal still qualified as a 95 % ceiling, and that same run was then
#: counted in `n_star_excluded_runs` — reported to the reader of #624 as an
#: output-budget truncation. A filter in a second spelling is not a filter.
OK = "ok"
TRUNCATED = "truncated"
REFUSED = "refused"
LATE_REFUSAL = "late-refusal"
OFF_TASK = "off-task"
#: The one class `classify_run` never returns: the endpoint did not answer, so
#: there is no head and no tail to read a class off. It lives here for the same
#: reason as the rest — the partition below has to partition the labels the code
#: actually produces, or it partitions a set nobody emits.
ENGINE_ERROR = "engine-error"

FAILURE_CLASSES: tuple[str, ...] = (
    OK, TRUNCATED, REFUSED, LATE_REFUSAL, OFF_TASK,
)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z'’-]*", text or ""))


def classify_run(text: str, finish_reason: str | None,
                 target_words: int = TARGET_REPORT_WORDS) -> str:
    """Why an incomplete run is incomplete — read off the head and the tail.

    #630's taxonomy is truncated / refused / off-task / late-refusal, and the
    point of naming them is that "the words are not there" has four different
    causes with four different fixes. `ok` means "no failure of these four was
    detected"; it does not mean the compliance score is high — a model that
    quietly forgets half the list fails by *forgetting*, which has no shape in
    the output, so it is the absence of these classes and the compliance
    fraction together that says it.

    Order matters. An empty completion whose `finish_reason` is `length` is
    thinking-budget exhaustion — the shape Voss found on a cloud model — and it
    is truncation, not a refusal or an off-topic answer. A refusal is read
    before `finish_reason` because a model that declines outright often does it
    in 30 tokens and then stops cleanly.
    """
    stripped = (text or "").strip()
    if not stripped:
        return TRUNCATED if finish_reason == "length" else OFF_TASK
    low = stripped.lower()
    head = low[:400]
    tail = low[-400:]
    head_refuses = any(p.search(head) for p in REFUSAL_PATTERNS)
    tail_refuses = any(p.search(tail) for p in REFUSAL_PATTERNS)
    if tail_refuses and not head_refuses:
        return LATE_REFUSAL
    if head_refuses and word_count(stripped) < REFUSED_MAX_WORDS:
        return REFUSED
    if finish_reason == "length":
        return TRUNCATED
    if word_count(stripped) < OFFTASK_MAX_FRACTION * target_words:
        return OFF_TASK
    return OK


#: One line holding this share of the required words is the list pasted back,
#: not prose. Measured on this engine before the artifact was committed: every
#: run at N>=400 ends in a dumped constraint list, so raw inclusion alone
#: reports compliance the model earned by copying rather than by writing. The
#: scorer therefore reports both — see `prose_compliance` in the module docstring.
ECHO_LINE_FRACTION = 0.5

#: A dump that WRAPPED is the same dump. Measured on the first committed
#: baseline: its N=200 repeat-0 run ends in a multi-line run of bare lower-case
#: vocabulary words (`output_tail` is nothing but them), no single line of it
#: held 50 % of the 200 required words, `echo_line_count` recorded 0, and the run
#: was scored `prose_compliance 0.98` — the one metric the paste had not touched
#: was the one #624 and #552 are told to cite. A line is therefore *list-like*
#: when this share of its own tokens are required words — these are rare
#: vocabulary words, so a sentence carrying that many of them is not a sentence —
#: and a run of consecutive list-like lines is an echo once it covers
#: `ECHO_BLOCK_FRACTION` of the list or `ECHO_BLOCK_MIN_WORDS` distinct required
#: words, whichever needs fewer (so a 1,600-word dump does not have to reach 800
#: words to be called what it is).
ECHO_LINE_TOKEN_FRACTION = 0.6

#: A line this short carries too few tokens for the density to mean anything:
#: "kudzu bebop" is a two-word coincidence, not a dump.
ECHO_MIN_LIST_TOKENS = 4
ECHO_BLOCK_FRACTION = 0.5
ECHO_BLOCK_MIN_WORDS = 25


def _tokens(line: str) -> list[str]:
    return re.findall(r"[a-z]+", line.lower())


def _is_list_like(tokens: list[str], present: set[str]) -> bool:
    """True when a line's own words are mostly required words: a dump line."""
    if len(tokens) < ECHO_MIN_LIST_TOKENS:
        return False
    hits = sum(1 for t in tokens if t in present)
    return hits >= ECHO_LINE_TOKEN_FRACTION * len(tokens)


def echo_line_indexes(text: str, words: list[str],
                      min_present: int = 8) -> list[int]:
    """Which lines ARE the constraint list, pasted back instead of used.

    Two shapes, because the engine has been measured doing both. One line
    carrying `ECHO_LINE_FRACTION` of the list is the paste the constant was
    tuned on; a run of consecutive list-like lines whose union covers the list
    is the same paste after it wrapped, and reading only the first silently
    credits the wrapped one as prose (see `ECHO_LINE_TOKEN_FRACTION`).

    Detection is a token-set intersection, not N regex scans: at N=1,600 a
    per-word scan of every output line costs more than the request being
    annotated, and a dumped list does not need that precision to be seen. Only
    meaningful once N is large enough that a report line could not plausibly
    carry this many required words by accident; below `min_present` present
    words the answer is always empty.
    """
    lines = (text or "").splitlines()
    if len(words) < min_present:
        return []
    present = set(present_words(text, words))
    if len(present) < min_present:
        return []
    line_tokens = [_tokens(line) for line in lines]
    flagged: set[int] = set()

    one_line = ECHO_LINE_FRACTION * len(words)
    for index, tokens in enumerate(line_tokens):
        if tokens and len(present & set(tokens)) >= one_line:
            flagged.add(index)

    block_needed = min(ECHO_BLOCK_FRACTION * len(words),
                       float(ECHO_BLOCK_MIN_WORDS))
    index = 0
    while index < len(line_tokens):
        if not _is_list_like(line_tokens[index], present):
            index += 1
            continue
        end = index
        block: set[str] = set()
        while end < len(line_tokens) and _is_list_like(line_tokens[end], present):
            block |= present & set(line_tokens[end])
            end += 1
        if len(block) >= block_needed:
            flagged.update(range(index, end))
        index = end
    return sorted(flagged)


def echo_lines(text: str, words: list[str], min_present: int = 8) -> list[str]:
    """The dumped-list lines themselves, for eyeballing a suspicious run."""
    lines = (text or "").splitlines()
    return [lines[i] for i in echo_line_indexes(text, words, min_present)]


def list_echo(text: str, words: list[str], min_present: int = 8) -> bool:
    """True when the output contains the constraint list pasted back as itself.

    Either shape: one line that is the list, or a run of consecutive list-like
    lines that is the list after it wrapped. The second shape is why this is not
    just the one-line rule — the first committed baseline holds a run whose
    wrapped dump went undetected and whose `prose_compliance` was therefore
    reporting a copy as an original.

    Recorded as a diagnostic beside the score, never as a failure class: the
    acceptance names four classes and inventing a fifth would move the headline
    number on a judgement the scorer cannot make. What it does move is
    `prose_compliance`, which is the number the length-cap decision should cite.
    """
    return bool(echo_line_indexes(text, words, min_present))


def prose_only(text: str, words: list[str]) -> tuple[str, int]:
    """The output with its dumped-list lines removed.

    Returns (prose_text, n_echo_lines). This is the text `prose_compliance` is
    computed on: words the model placed in sentences it wrote, with the
    copy-the-list channel closed.
    """
    dropped = set(echo_line_indexes(text, words))
    kept = [line for i, line in enumerate((text or "").splitlines())
            if i not in dropped]
    return "\n".join(kept), len(dropped)


# ── Engine slots ────────────────────────────────────────────────────────

class SlotUnavailable(Exception):
    """A requested slot cannot be measured, with the reason a human can act on."""


def resolve_slot(alias: str) -> dict[str, Any]:
    """Where one engine SLOT lives and what it says it is serving.

    Deliberately not `resolve_model_alias` — see the module docstring. The
    base URL comes from the slot's own `models:` entry (the pattern
    `eval/run_tool_choice_eval.py:221-225` uses), the model name comes from the
    engine's `/v1/models`, because vLLM answers for the id it loaded and a
    request naming the alias is refused outright (`HTTPStatusError: The model
    `eco` does not exist` is what an unvalidated alias already costs an
    autonomy task). `expect_model` from config is compared against what the
    engine reports so a launcher that quietly changed occupant cannot produce a
    baseline that labels one model as another.
    """
    if alias not in SLOT_ALIASES:
        raise ValueError(f"unknown engine slot {alias!r}; known: {SLOT_ALIASES}")
    from app.config import _get_model_cfg

    cfg = _get_model_cfg(alias) or {}
    base = (cfg.get("base_url")
            or (cfg.get("env") or {}).get("ANTHROPIC_BASE_URL") or "")
    program = SLOT_PROGRAMS[alias]
    enabled = True
    flag = None
    if program:
        from app.llm_slots import is_enabled, slot_flag
        enabled = is_enabled(program)
        flag = slot_flag(program)
    if not enabled:
        raise SlotUnavailable(
            f"slot {alias!r} is switched off ({flag} is false in config.yaml); "
            "refusing to measure a different engine under this slot's label")
    if not base:
        raise SlotUnavailable(f"slot {alias!r} has no base_url in config.yaml")
    base = base.rstrip("/")
    endpoint = f"{base}/v1/chat/completions"
    served = _served_models(base)
    if not served:
        raise SlotUnavailable(
            f"slot {alias!r} is switched on but nothing is answering "
            f"{base}/v1/models; the sweep measures an engine it can name, "
            "not a port it hopes is up")
    expect = str(cfg.get("expect_model") or "")
    return {
        "slot": alias,
        "program": program,
        "enabled_flag": flag,
        "configured_base_url": base,
        "endpoint": endpoint,
        "served_models": served,
        "model": served[0],
        "expect_model": expect,
        # Substring match, same convention as app/model_identity.py: the config
        # says "Qwen3.8-Flash-Next", the engine says "Qwen3.8-Flash-Next-nvfp4".
        # None means the slot declares no expectation, so nothing was checked —
        # False is reserved for an expectation the engine contradicted.
        "identity_match": (None if not expect else
                           any(expect.lower() in s.lower() for s in served)),
    }


def _http_json(url: str, payload: dict | None, timeout: float) -> dict:
    """One HTTP round trip. `payload=None` is a GET; anything else is a POST."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


@lru_cache(maxsize=8)
def _served_models_cached(base: str) -> tuple[str, ...]:
    try:
        data = _http_json(f"{base}/v1/models", None, 20.0)
    except Exception:
        return ()
    return tuple(d.get("id", "") for d in (data.get("data") or []) if d.get("id"))


def _served_models(base: str) -> list[str]:
    return list(_served_models_cached(base))


#: How many missing words a record carries. The count is always exact; the
#: list is a sample, because at N=1,600 a forgotten two-thirds of the list is
#: 1,000 words of prose inside the artifact that holds the number.
MISSING_SAMPLE = 25


def run_point(slot: dict[str, Any], words: list[str], *, n: int, seed: int,
              vocab_sha: str, max_tokens: int, temperature: float,
              target_words: int, timeout: float,
              thinking: str = DEFAULT_THINKING) -> dict[str, Any]:
    """One (engine, N) measurement: build, send, score, classify."""
    prompt = build_prompt(words, target_words)
    payload: dict[str, Any] = {
        "model": slot["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # Sent only when the caller closes thinking, so an engine that does not
    # understand `chat_template_kwargs` is not handed a key it might refuse for
    # a run that never asked for one. `eval/secondary_routing_eval.py:514` uses
    # the same key for the same reason.
    if thinking == "off":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.monotonic()
    error = None
    text = ""
    finish_reason = None
    usage: dict[str, Any] = {}
    try:
        data = _http_json(slot["endpoint"], payload, timeout)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        # `content` and `reasoning` are separate fields on a thinking engine;
        # only `content` is the answer. Reading `reasoning` would score the
        # model's private deliberation as if it had obeyed the constraint.
        text = msg.get("content") or ""
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
    except urllib.error.HTTPError as e:
        error = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
    except Exception as e:  # noqa: BLE001 — a dead engine is a datapoint, not a crash
        error = f"{type(e).__name__}: {e}"
    elapsed = round(time.monotonic() - started, 2)

    if error:
        score: dict[str, Any] = {"required": n, "present": 0,
                                 "missing": list(words), "compliance": 0.0}
        prose_score = score
        echo_count = 0
        failure = ENGINE_ERROR
    else:
        score = score_compliance(text, words)
        prose_text, echo_count = prose_only(text, words)
        prose_score = score_compliance(prose_text, words)
        failure = classify_run(text, finish_reason, target_words)
    missing = score["missing"]

    return {
        "engine": slot["slot"],
        "model": slot["model"],
        "n": n,
        "seed": seed,
        "vocab_sha256": vocab_sha,
        "required_words_sha256": hashlib.sha256(
            "\n".join(words).encode()).hexdigest(),
        "prompt_chars": len(prompt),
        "prompt_tokens": usage.get("prompt_tokens"),
        # Recorded on the record, not assumed from config: a truncation scored
        # as forgetting is this eval's own failure mode, and the only way a
        # later reader can tell the two apart is to see the budget the run was
        # actually given.
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": (None if thinking == "on" else False),
        "target_report_words": target_words,
        "seconds": elapsed,
        "finish_reason": finish_reason,
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_kept_out_of_score": True,
        "output_words": word_count(text),
        "output_head": text[:240],
        "output_tail": text[-240:] if len(text) > 240 else "",
        "compliance": score["compliance"],
        "words_present": score["present"],
        # The same denominator, counted with any line that IS the dumped list
        # removed. `compliance` is IFScale's metric (verbatim inclusion, which a
        # pasted list satisfies); `prose_compliance` is the one that answers
        # "could the engine still hold N rules while writing", and on this
        # engine the two diverge hard above N=400.
        "prose_compliance": prose_score["compliance"],
        "prose_words_present": prose_score["present"],
        "echo_line_count": echo_count,
        "missing_count": len(missing),
        "missing_sample": missing[:MISSING_SAMPLE],
        "failure_class": failure,
        "list_echo": echo_count > 0,
        "error": error,
    }


#: Failure classes whose output is still a comparable attempt at the task, so its
#: score says something about the engine.
#:
#: `ok`, `off-task`, `refused` and `late-refusal` are all the engine having its
#: turn and producing an answer: an off-topic answer or a refusal IS a compliance
#: failure, and Voss's taxonomy counts them beside forgetting. `truncated` (the
#: harness's own output budget ran out) and `engine-error` (the endpoint never
#: answered) are not — those measure the harness and the machine, and counting
#: them against the ceiling would report the box's memory as the model's capacity.
#:
#: Both sides are spelled from the constants `classify_run` returns, and
#: `NON_COMPARABLE_CLASSES` exists so the partition is checkable: the two sets
#: must cover every label and overlap nowhere. Tests
#: `tests/test_instruction_compliance_eval.py::test_the_ceiling_filter_is_spelled_in_the_labels_the_classifier_emits`
#: and `::test_a_density_with_no_comparable_run_cannot_qualify` hold that.
COMPARABLE_CLASSES = (OK, REFUSED, LATE_REFUSAL, OFF_TASK)

#: The complement: labels a ceiling may not be built from, and — because
#: `pointer_lines` describes the exclusion count in these words — the ONLY labels
#: that count can ever carry. A run that refused is not an exclusion, it is a
#: failure, and the last version of this file reported it as one to #624.
NON_COMPARABLE_CLASSES = (TRUNCATED, ENGINE_ERROR)


def n_star(points: list[dict[str, Any]],
           threshold: float = N_STAR_THRESHOLD,
           metric: str = "compliance",
           classes: tuple[str, ...] = COMPARABLE_CLASSES) -> int | None:
    """Largest swept N at which every comparable repeat cleared `threshold`.

    Every repeat, not the mean: a density that clears 95 % once and 80 % twice
    does not have a 95 % ceiling at that density — the sweep would be reporting
    the luckiest draw as a capacity. None means no swept density cleared, i.e.
    N* is below the sweep's first point.

    `classes` is the difference between a ceiling and an accident. Clause 3
    forbids scoring a run that never finished as forgetting, so such a run is
    recorded with compliance 0.0 and its class — and a rule that then folds that
    0.0 into the density's verdict is reporting the ceiling of the harness's
    `max_tokens`, or of whoever was holding the GPU, as the engine's capacity. The
    first shipped artifact is the case in point: at N=1600 the two repeats scored
    0.9781 and 0.7656, the second a `finish_reason=length` cutoff, and the
    `truncated` label it carried was enough to drop N* from 1600 to 800 while
    leaving the reader no way to see that the number came from an output budget.
    A density whose runs are ALL non-comparable still cannot qualify (there is
    nothing to compare), so this fails closed rather than wide: excluded runs are
    counted in `n_star_excluded_runs` beside the ceiling, so a ceiling resting on
    one comparable run of two says so on its face.

    `metric` is what makes this file's own finding legible: `compliance` is
    verbatim inclusion (IFScale's metric, which a model that pasted the list
    back passes), `prose_compliance` is inclusion inside prose that is not the
    list. Report both; deciding on the first alone would let a copy the
    constraint list count as holding the constraint.
    """
    best: int | None = None
    for n in sorted({int(p["n"]) for p in points}):
        mine = [p for p in points if int(p["n"]) == n
                and p.get("failure_class") in classes]
        if mine and all(float(p.get(metric, 0.0)) >= threshold for p in mine):
            best = n
    return best


def n_star_excluded(points: list[dict[str, Any]],
                    classes: tuple[str, ...] = COMPARABLE_CLASSES) -> int:
    """Runs the ceiling's verdict left out — disclosed, never silent."""
    return sum(1 for p in points if p.get("failure_class") not in classes)


def band_from_spread(spreads: list[float]) -> float:
    """The reproduction band this run can honestly claim."""
    return round(max(MIN_REPRODUCTION_BAND, 2.0 * max(spreads or [0.0])), 4)


def summarise(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Per engine: the per-N mean of both metrics, spreads, failure classes, N*s."""
    out: dict[str, Any] = {}
    for engine in sorted({p["engine"] for p in points}):
        mine = [p for p in points if p["engine"] == engine]
        per_n: dict[str, Any] = {}
        spreads: list[float] = []
        for n in sorted({int(p["n"]) for p in mine}):
            rows = [p for p in mine if int(p["n"]) == n]
            vals = [float(p["compliance"]) for p in rows]
            prose_vals = [float(p.get("prose_compliance", 0.0)) for p in rows]
            spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
            spreads.append(spread)
            per_n[str(n)] = {
                "runs": len(vals),
                "mean_compliance": round(sum(vals) / len(vals), 4),
                "min_compliance": round(min(vals), 4),
                "max_compliance": round(max(vals), 4),
                "spread": round(spread, 4),
                # The band a re-run of THIS point is judged against: 2x this
                # point's own observed spread, floored. A single global band
                # would let N=50's 0.10 spread excuse a 0.15 move at N=1600,
                # where the same run reproduced to 0.008.
                "reproduction_band": band_from_spread([spread]),
                "mean_prose_compliance": round(sum(prose_vals) / len(prose_vals), 4),
                "min_prose_compliance": round(min(prose_vals), 4),
                "echo_runs": sum(1 for p in rows if p.get("list_echo")),
                "repeats": [
                    {"compliance": round(float(p["compliance"]), 4),
                     "prose_compliance": round(float(p.get("prose_compliance", 0.0)), 4),
                     "list_echo": bool(p.get("list_echo")),
                     "failure_class": p["failure_class"],
                     "finish_reason": p.get("finish_reason"),
                     "seconds": p.get("seconds")}
                    for p in rows],
            }
        classes: dict[str, int] = {}
        for p in mine:
            classes[p["failure_class"]] = classes.get(p["failure_class"], 0) + 1
        out[engine] = {
            "points": per_n,
            "failure_classes": classes,
            "echo_runs": sum(1 for p in mine if p.get("list_echo")),
            "max_repeat_spread": round(max(spreads or [0.0]), 4),
            # Two ceilings, deliberately not collapsed into one number.
            # `n_star_95` is #630's clause verbatim: verbatim inclusion, the
            # IFScale metric, which a run that pasted the list back can clear.
            # `n_star_95_prose` is the same sweep scored with those lines
            # removed — the number to cite when the question is whether the
            # engine can HOLD N rules, not whether it can repeat them.
            "n_star_95": n_star(mine),
            "n_star_95_prose": n_star(mine, metric="prose_compliance"),
            # Beside the ceiling: how many runs it declined to compare, so a
            # ceiling resting on one run of two says so on its face.
            "n_star_excluded_runs": n_star_excluded(mine),
            "n_star_threshold": N_STAR_THRESHOLD,
        }
    return out


#: The two decisions #630 exists to inform, and the only two this function
#: names. #624 caps SKILL.md at 100 lines on a guess; #552 wants to know whether
#: each skill/entry was honoured. Both currently cite no number, so the sweep
#: emits the sentence they should cite rather than leaving a reader to translate
#: a baseline document into a claim. What #624 then *does* with it is Alan's
#: scope call — #630 records a measurement, it does not move a cap.
POINTER_TARGETS: tuple[str, ...] = ("#624", "#552")


def pointer_lines(baseline: dict[str, Any]) -> list[str]:
    """The per-engine lines to append to #624 and #552 (acceptance step 6).

    Generated from the baseline rather than typed, because a pointer whose
    number was copied by hand is a number nobody can re-derive, and the whole
    point of the clause is that those two decisions stop being guesses. Both
    ceilings go in, because the two decisions want different ones: the line
    a skill may occupy is bounded by what the engine can be *told*, and the
    rule it states is only honoured if the engine can use it in prose.
    """
    head = (f"#630 IFScale density sweep `{baseline['label']}` "
            f"({str(baseline['ran_at'])[:10]}) measured:")
    out = [head]
    engines_meta = baseline.get("engines") or {}
    for engine, summ in sorted(baseline["summary"].items()):
        # Identity lives on the engine row, not the summary row: the summary is
        # numbers, and a number attributed to an engine nobody verified is the
        # failure mode this whole file is built around.
        slot = engines_meta.get(engine) or {}
        served = ", ".join(slot.get("served_models") or []) or "?"
        def _ceiling(key: str) -> str:
            # "N* = below the floor" is not a sentence; the null case states the
            # answer in its own words, because a reader of #624 must not have to
            # learn that `None` here means "no swept density cleared the bar".
            ns = summ.get(key)
            return (f"N* = {ns}" if ns is not None
                    else f"N* below the swept floor (no point cleared "
                         f"{summ['n_star_threshold']})")
        out.append(
            f"- engine `{engine}` at `{slot.get('configured_base_url', '?')}` "
            f"serving `{served}`: "
            f"{_ceiling('n_star_95')} by verbatim inclusion (the metric "
            f"IFScale and this item's acceptance name), "
            f"{_ceiling('n_star_95_prose')} with the model's own "
            f"constraint-list dumps excluded — cite the second for a "
            f"length/honouring decision; "
            f"{summ.get('echo_runs', 0)} of "
            f"{sum(p['runs'] for p in summ['points'].values())} runs ended in a "
            f"dumped list; failure classes {summ['failure_classes']}; "
            f"{summ.get('n_star_excluded_runs', 0)} run(s) the ceiling excluded "
            f"as not comparable to the task (output-budget truncation or engine "
            f"error), so read the ceiling as the widest density with clean runs, "
            f"not as a density where every attempt cleared; repro "
            f"`{baseline['repro_command']}`")
    for engine, slot in sorted(baseline["engines"].items()):
        if not slot.get("available"):
            out.append(f"- engine `{engine}`: NOT MEASURED — {slot.get('reason')}")
    out.append("- Read N* as a capacity ceiling, not a licence: a high N* says "
               "length caps are not the binding constraint, it does not say the "
               "prompt surface may grow (Voss: capacity up != reliability up).")
    return out


def build_baseline(args: argparse.Namespace, points: list[dict[str, Any]],
                   slots: dict[str, Any], repro: str) -> dict[str, Any]:
    """The baseline document. One shape, written to one or two paths.

    `reproduction_band_abs` is deliberately the loosest band in the run (the
    max over points of 2x that point's own repeat spread), because
    `reproduction_band_meaning` promises it bounds EVERY (engine, N) mean. A
    smaller headline figure that only some points obey would be a contract
    nobody can honour. The tight checks are the per-point
    `summary[engine]["points"][N]["reproduction_band"]` values, which are
    usually an order of magnitude narrower.
    """
    summary = summarise(points)
    spreads = [float(s["max_repeat_spread"]) for s in summary.values()]
    band = band_from_spread(spreads)
    return {
        "schema": "instruction-compliance/1",
        "item": "#630",
        "label": args.label,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "repro_command": repro,
        "reproduction_band_abs": band,
        "reproduction_band_meaning": (
            "a re-run of repro_command must land every (engine, N) mean "
            f"compliance within +/-{band} of the figure here; the band is "
            "2x the largest spread between this run's own repeats at any "
            f"point, floored at {MIN_REPRODUCTION_BAND}, so it is measured, "
            "not assumed. Each per-(engine, N) point also carries its own "
            "tighter `reproduction_band`, and that is the one to test a "
            "specific density against"),
        "noise_source": "spread between repeats within this run",
        "scoring": "mechanical: regex word-boundary presence, no LLM judge",
        "failure_classes": list(FAILURE_CLASSES) + [ENGINE_ERROR],
        "config": {
            "sweep_n": list(args.sweep),
            "seed": args.seed,
            "repeats": args.repeats,
            "target_report_words": args.target_words,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            # None = the engine's own default was left alone; False = this run
            # closed the reasoning channel on every request. Which of the two a
            # baseline is changes what its compliance numbers mean.
            "enable_thinking": (None if args.thinking == "on" else False),
            "timeout_seconds": args.timeout,
            "vocab_path": str(VOCAB_PATH.relative_to(ROOT)),
            "vocab_sha256": args.vocab_sha,
            "n_star_threshold": N_STAR_THRESHOLD,
        },
        "engines": slots,
        "summary": summary,
        "records": points,
    }


def to_jsonl(baseline: dict[str, Any]) -> str:
    """The same document as one JSON object per line, for the committed copy."""
    lines = [json.dumps({"record": "meta", **{
        k: baseline[k] for k in ("schema", "item", "label", "ran_at",
                                 "repro_command", "reproduction_band_abs",
                                 "reproduction_band_meaning", "noise_source",
                                 "scoring", "failure_classes")}}, sort_keys=True),
        json.dumps({"record": "config", **baseline["config"]}, sort_keys=True)]
    for name, eng in sorted(baseline["engines"].items()):
        lines.append(json.dumps({"record": "engine", "engine": name, **eng},
                                sort_keys=True))
    for name, summ in sorted(baseline["summary"].items()):
        lines.append(json.dumps({"record": "summary", "engine": name, **summ},
                                sort_keys=True))
    lines.append(json.dumps({"record": "records", "records": baseline["records"]},
                            sort_keys=True))
    return "\n".join(lines) + "\n"


def from_jsonl(text: str) -> dict[str, Any]:
    """Read a committed JSONL copy back into the shape this script writes.

    The inverse of :func:`to_jsonl`, and it exists because the artifact is the
    durable form (`.json` is ignored repo-wide) while every consumer of the
    numbers — the pointer lines #624 and #552 cite, a nightly job that wants to
    compare two baselines — wants the document. A reader that has to re-parse
    the record types by hand is a reader who gets it wrong once and then quotes
    a hand-written number.

    Every line must carry `record`; unknown kinds are kept under their own name
    rather than dropped, so a future record type is visible to an old reader
    instead of silently vanishing from the figures.
    """
    out: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        kind = row.get("record")
        if not kind:
            raise ValueError(f"baseline line without a record kind: {line[:80]}")
        body = {k: v for k, v in row.items() if k != "record"}
        if kind == "records":
            records = body["records"]
        elif kind == "engine":
            out.setdefault("engines", {})[body.pop("engine")] = body
        elif kind == "summary":
            out.setdefault("summary", {})[body.pop("engine")] = body
        elif kind == "config":
            out["config"] = body
        elif kind == "meta":
            out.update(body)
        else:
            out.setdefault(kind, []).append(body)
    out["records"] = records
    return out


def repro_command(args: argparse.Namespace) -> str:
    parts = [f".venvs/lloyd/bin/python eval/{Path(__file__).name}",
             f"--label {args.label}", f"--seed {args.seed}",
             f"--sweep {','.join(str(n) for n in args.sweep)}",
             f"--repeats {args.repeats}",
             f"--max-tokens {args.max_tokens}",
             f"--temperature {args.temperature}",
             f"--thinking {args.thinking}"]
    if args.engines:
        parts.append(f"--engines {','.join(args.engines)}")
    return " ".join(parts)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--sweep", default=",".join(str(n) for n in SWEEP_N),
                    help="comma-separated densities; default is the #630 sweep")
    ap.add_argument("--repeats", type=int, default=1,
                    help="requests per (engine, N); >=2 is what measures the "
                         "reproduction band instead of asserting one")
    ap.add_argument("--engines", default=",".join(SLOT_ALIASES),
                    help=f"slots to sweep, default {','.join(SLOT_ALIASES)} — "
                         "which is #630's order: the secondary slot last, and "
                         "the whole sweep issues one request at a time so a "
                         "measurement is never also a load test on the engine "
                         "every live turn is using")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--thinking", choices=("off", "on"), default=DEFAULT_THINKING,
                    help="`off` (default) closes the reasoning channel with "
                         "chat_template_kwargs.enable_thinking=false; `on` "
                         "leaves the engine default alone, which on this engine "
                         "spends the whole budget inside reasoning at N=50 — "
                         "see DEFAULT_THINKING")
    ap.add_argument("--target-words", type=int, default=TARGET_REPORT_WORDS)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    ap.add_argument("--committed-copy", default="",
                    help="also write a JSONL projection of the same records to "
                         "this path (see the module docstring's OUTPUT section)")
    ap.add_argument("--make-vocab", default="", metavar="SRC",
                    help="build the fixed vocabulary from a wordlist file "
                         "(one word per line) and exit; provenance of "
                         "instruction_compliance_vocab.txt")
    ap.add_argument("--vocab-words", type=int, default=2400)
    ap.set_defaults(vocab_sha="")
    args = ap.parse_args(argv)
    # Normalised here, not in `main`, because `repro_command` and
    # `build_baseline` are callable on their own (and are tested that way): a
    # repro built from a namespace whose `--sweep` is still the raw string
    # renders `--sweep 5,0,,,1,0,0,...`, which parses back to garbage — a repro
    # line that cannot reproduce is the worst thing this file could ship.
    args.sweep = _parse_int_list(args.sweep) or list(SWEEP_N)
    args.engines = tuple(
        e.strip() for e in str(args.engines).split(",") if e.strip()) or SLOT_ALIASES
    return args


def make_vocab(src: Path, out: Path, want: int) -> int:
    """Deterministic stride sample of a wordlist, minus the prompt frame's words.

    Run once; the product is committed and its sha goes into every baseline.
    A stride (not a prefix) so the list is not all one letter of the alphabet,
    and the frame filter so no required word is free (see `frame_tokens`).
    """
    frame = frame_tokens()
    seen: set[str] = set()
    words: list[str] = []
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        w = line.strip()
        if (5 <= len(w) <= 12 and w.isalpha() and w.isascii() and w.islower()
                and w not in frame and w not in seen):
            seen.add(w)
            words.append(w)
    words.sort()
    stride = max(1, len(words) // want)
    picked = words[::stride][:want]
    header = (
        f"# Fixed required-word vocabulary for instruction_compliance_eval.py "
        f"(backlog #630).\n"
        f"# Built by: --make-vocab {src}\n"
        f"# {len(words)} candidate words of 5-12 lowercase ascii letters, "
        f"excluding every word the prompt frame supplies "
        f"({len(frame_tokens())} tokens), sorted, one stride-{stride} sample.\n"
        f"# {len(picked)} words. Regenerating this file invalidates every "
        f"baseline: the sweep compares against the sha recorded in each.\n"
    )
    out.write_text(header + "\n".join(picked) + "\n", encoding="utf-8")
    return len(picked)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.make_vocab:
        n = make_vocab(Path(args.make_vocab), VOCAB_PATH, args.vocab_words)
        print(f"[vocab] wrote {n} words to {VOCAB_PATH}")
        return 0

    engines = list(args.engines)
    vocab = load_vocab()
    args.vocab_sha = vocab_sha256()
    repro = repro_command(args)

    from app.paths import EVAL_BASELINES_DIR

    print(f"[info] vocab {len(vocab)} words sha={args.vocab_sha[:12]} "
          f"| sweep={list(args.sweep)} | seed={args.seed} "
          f"| repeats={args.repeats} | max_tokens={args.max_tokens}")
    print(f"[info] repro: {repro}")

    points: list[dict[str, Any]] = []
    slots: dict[str, Any] = {}
    for alias in engines:
        # Cleared per slot so a sweep that follows an engine boot sees the
        # engine that is up now rather than the empty answer it cached a
        # minute ago, and so the model name in the baseline is this run's.
        _served_models_cached.cache_clear()
        try:
            slot = resolve_slot(alias)
        except (SlotUnavailable, ValueError) as e:
            print(f"[warn] engine {alias}: {e}", file=sys.stderr)
            slots[alias] = {"available": False, "reason": str(e)}
            continue
        slot["available"] = True
        slots[alias] = slot
        print(f"[slot {alias}] {slot['endpoint']} serving {slot['served_models']} "
              f"(expect {slot['expect_model']!r}: match={slot['identity_match']})")

        # Serial on purpose: #630 asks for the secondary slot to run alone and
        # last, and one request at a time keeps a measurement sweep from being
        # also a load test on the engine every live turn is using.
        for n in args.sweep:
            for rep in range(max(1, args.repeats)):
                words = sample_words(vocab, n, args.seed)
                rec = run_point(slot, words, n=n, seed=args.seed,
                                vocab_sha=args.vocab_sha,
                                max_tokens=args.max_tokens,
                                temperature=args.temperature,
                                target_words=args.target_words,
                                timeout=args.timeout)
                rec["repeat"] = rep
                points.append(rec)
                print(f"[{alias} n={n} rep={rep}] compliance={rec['compliance']:.3f} "
                      f"({rec['words_present']}/{n}) class={rec['failure_class']} "
                      f"finish={rec['finish_reason']} words={rec['output_words']} "
                      f"{rec['seconds']}s")

    baseline = build_baseline(args, points, slots, repro)
    EVAL_BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = EVAL_BASELINES_DIR / f"{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(baseline, indent=2, default=str))
    print(f"[info] wrote {out_path}")
    if args.committed_copy:
        copy = Path(args.committed_copy)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(to_jsonl(baseline), encoding="utf-8")
        print(f"[info] wrote {copy}")

    for engine, summ in sorted(baseline["summary"].items()):
        ns = summ["n_star_95"]
        print(f"[N*] {engine}: {ns if ns is not None else 'below the sweep floor'}"
              f" at threshold {summ['n_star_threshold']}")
    for alias, slot in sorted(slots.items()):
        if not slot.get("available"):
            print(f"[unavailable] {alias}: {slot['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
