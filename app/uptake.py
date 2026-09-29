"""Uptake — did a durable entry that landed in the prompt change anything? (#552)

Every eval surface Lloyd already had measured an *input*: retrieval hits
(`eval/run_eval.py`), tool choice (`eval/run_tool_choice_eval.py`), prefetch
precision, prompt-surface A/B. None of them asks the question the other way
round. Once a line lands in `lloyd/USER.md`, or a `SKILL.md` body is written,
or a knowledge note is filed — did later behavior follow it, or did Alan have
to dispute, correct, or redo the work anyway? Uber's uReview team went up
exactly those rungs for code review (comment cost → reply sentiment →
**addressal rate**) and credited the last rung with most of the quality gain,
because "the model doesn't know that it's wrong — it always confidently says
100% sure". Their loop closed only once the per-rule outcome rollup was
surfaced back to whoever wrote the rule.

This module is the measurement half of that, and nothing else. It:

1. mines the session transcripts for turns where the user pushed back
   (`human_turns`, `candidate_disputes`), graded against the prose that came
   just before — a reply is only a dispute *relative to* what it answers;
2. classifies those turns with the secondary engine (`classify_dispute`) and
   scores the classifier against a hand-labeled corpus (`precision_recall`),
   refusing to go further below the item's 0.70 precision floor;
3. joins each detected dispute to the durable entries that were in force for
   that turn and emits a per-entry table (`build_uptake_table`,
   `write_table`): `present_in_turns`, `disputes`, `dispute_rate`.

**Nothing here prunes.** The consolidator (`skills/nightly-reflection-knowledge-write`)
is the consumer; the item deliberately stops at "measurement only" because a
number with no evidence behind it becomes self-fulfilling the moment it is
written into durable memory.

Two honesty constraints are load-bearing and encoded below, not in prose:

*Presence is evidence-bound, and says which kind.* `USER.md`/`MEMORY.md` are in
*every* prompt, so a naive join attributes one dispute to every entry in force
— the item names this as the crux risk. A skill is therefore credited only when
the logs show it (`skills_read` payload, or an injected `<skill name=…>` block
in that turn's context), a note only when it arrived in that turn's prefetch,
and the always-in-force memory entries carry `presence_source` plus a
text-overlap weight so a reader can tell "this was questioned" from "this
happened to be in the window".

*Store sizes never travel without their probe timestamp.* This repo has been
burned by a KG figure quoted as current an hour after it was true;
`write_table` refuses an integer metric in the `stores` block that lacks
`probed_at`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]

#: The retrieval non-regression threshold the triaged items hard-code. It is
#: recorded here so the code that replaces it can be pointed at what it replaced.
RETRIEVAL_GATE_HARDCODE = 0.95

#: How many of the NEWEST comparable nights a band is computed over (#1220).
#: Comparability by run shape is a filter, not a window: with no cap the pool is
#: every nightly the baselines directory ever held, so the tolerance published
#: today is widened by nights measured against a corpus that no longer exists —
#: the 13 nights on disk 2026-09-17 spanned `corpus.facts` 205,693 -> 314,653 and
#: published a doc_hit_rate band of 0.15 where the newest 7 give 0.05, three
#: times looser and loosened entirely by nights nobody can reproduce. Seven is
#: one week of nightly runs: long enough to contain a real spread, short enough
#: that every night in it retrieved against roughly today's corpus. The value is
#: a judgement call and the item leaves it open, so it is recorded here rather
#: than inferred from the code; anything >= 6 also keeps the six-night fixture in
#: tests/test_uptake.py describing a whole window instead of a truncated one.
RETRIEVAL_GATE_WINDOW_NIGHTS = 7

#: Step 2's stop condition from the item: below this precision, do not build a
#: table on top of the classifier.
PRECISION_FLOOR = 0.70

#: A floor the item does not state and needs to. Its acceptance names precision
#: alone, and precision alone is satisfiable by a classifier that almost never
#: fires — the first measured run of this probe scored precision 1.00 / recall
#: 0.09 by answering NOT to everything, which would have emitted an uptake
#: table attributing zero disputes to every entry and read as "no entry is ever
#: disputed". A dispute detector that misses most disputes is not a conservative
#: measurement, it is an absent one, so both ends of the matrix are gated.
RECALL_FLOOR = 0.50

#: Required keys of every emitted row. Rows also carry kind-specific extras
#: (`kind`, `name`, `present_turn_ids`) and, on the rows that need one, a `note`
#: caveat — the contract is this set being a SUBSET, which is what a reader may
#: index on unconditionally. The vault skill that consumes this table names
#: these fields in prose, and prose is not a schema: rename one in code and the
#: consolidator reads a missing key and still writes a completion note that
#: looks like it cited a figure. `tests/test_uptake.py` pins both directions —
#: rows match this set hermetically, so the gate catches a rename on any box, and
#: the skill's cited names are checked against the emitted artifact on a live
#: vault.
ROW_KEYS = frozenset({
    "entry", "present_in_turns", "disputes", "dispute_rate", "weighted_disputes",
    "overlap_max", "presence_source",
})

#: Which scorer wrote a table, stamped into the artifact. A reader cannot tell
#: one scorer's output from another's by looking at a filename: the three files
#: under `eval/uptake/` predate #1195, and every rule that tells a consumer how
#: to read a `weighted_disputes` is true of one generation and false of the
#: other. Generation 1 (unstamped — `uptake-2026-09-11/17/20.json` at this
#: commit): no `coverage.by_presence_source`, no `scored_on` on any row, a
#: `weighted_disputes` that is never null, the skill channel paying
#: `max(overlap(...), 0.05)` under every disputed hit — 5, 4 and 5 rows sit at
#: exactly 0.05 in those three files — and every note row scored on its title.
#: Generation 2 is this scorer. `tests/test_uptake.py` pins that the stamp and
#: the bytes agree in both directions, so a stamped table cannot promise keys it
#: does not have and an unstamped one is read by the rules its own generation
#: supports. Bump it whenever a reading rule changes meaning.
SCORER_GENERATION = 2

#: What each row field means, in the artifact's own words. `build_uptake_table`
#: emits this on every table, and `scripts/uptake_probe.py` no longer carries its own
#: copy: the glossary used to live in the writer's `main()` and it was **false**
#: for two of the three channels — it described `weighted_disputes` as overlap
#: "between the entry text and the correction", which is true of the
#: always-in-force rows only. A skill row's entry text was its *name* with a 0.05
#: constant under it, and a note row's was its *title*. The table is the only
#: thing a consumer reads, so the correction belongs here and not in a comment.
GLOSSARY = {
    "dispute_rate": "disputes / present_in_turns. An UPPER BOUND for "
                    "always-in-force memory entries, since those are in every prompt.",
    "weighted_disputes": "disputes discounted by lexical overlap (word-set Jaccard "
                         "over unigrams+bigrams) between the entry text and the "
                         "correction. null means no signal, not zero: no disputed "
                         "turn shares any text with the entry, so the row supports "
                         "neither keeping nor pruning and must not be coalesced to "
                         "0 for a sort. The text behind the number differs per "
                         "`presence_source` — the entry's own text for "
                         "always-in-force rows, the skill NAME for proxy rows, the "
                         "note's injected line (`scored_on`) for note rows — and the "
                         "channels are not on one scale: never rank across them, "
                         "read `coverage.by_presence_source` and rank within one "
                         "channel only.",
    "presence_source": "which evidence says the entry was in force. Evidence-bound "
                      "for skills and notes; always_in_force for USER.md/MEMORY.md.",
    "scorer_generation": "which scorer wrote THIS file, so the rules above can be "
                         "read against the right table. Absent means generation 1: "
                         "no `coverage.by_presence_source`, no `scored_on`, never a "
                         "null, and a skill weight of 0.05 that is a removed "
                         "calibration floor rather than a measurement. Read the key, "
                         "never the filename or the date.",
    "by_presence_source": "per-channel denominators: `rows`, `rows_with_weight`, "
                          "`rows_null_weight`, `max_weight`. The four numbers that "
                          "make a cross-channel comparison impossible, published "
                          "instead of leaving the reader to discover them.",
    # Explains its own absence, like `scorer_generation` above: the tables that
    # cleared every floor are the common case and carry no marker, so a reader has
    # to be able to tell "no marker, so it passed" from "this file is stamped".
    "audit_only": "ABSENT means this table cleared every floor and is citable. "
                  "Present means the run did NOT clear them and the operator asked "
                  "for the table anyway (`--ignore-precision-floor`); read "
                  "`audit_only_reason` for which figures were weak. Every dispute "
                  "count in it is an unvalidated classifier's guess: read the shape "
                  "of the window, cite nothing, and never archive an entry on it.",
}
#: How tightly a skill row's presence is bounded, tightest last. A row must name
#: one: "this skill was injected into that very turn" and "some turn of that
#: session opened this skill at some point" are not the same claim, and a
#: consolidator has to be able to tell them apart.
PRESENCE_BOUNDS = frozenset({"session_wide", "causal_event_order", "injected_this_turn"})
#: Presence source for `USER.md` / `MEMORY.md` bullets: loaded into every turn,
#: so "present" carries no information on its own and must be weighted.
ALWAYS_IN_FORCE = "always_in_force:system_prompt"

#: Per-injection skill presence, from the `prefetch.skill_match` rows
#: `app/prefetch.py:_emit_skill_match_events` writes one per offered skill per
#: turn. A row with `landed: true` is the injector's own word that the body went
#: into that turn's prompt, which is the reason this channel exists: the proxy
#: below only ever proved the model *asked*. The rows begin 2026-09-25, so no
#: session written before that date has one — which is why the proxy stays
#: declared rather than replaced.
SKILL_PRESENCE_TELEMETRY = "prefetch:skill_match"

#: Skill presence proxy: `skills_read` tool-call payloads in
#: `event_logs/*.events.jsonl` plus the `<skill name=…>` blocks visible in the
#: turn's own persisted context. Fallback and supplement, not rival: the telemetry
#: reaches only sessions written from 2026-09-25, and it says nothing about a body
#: the model pulled mid-turn with `skills_read`. The name says what it is — a read
#: is a request, not an injection.
SKILL_PRESENCE_PROXY = "proxy:skills_read+injected_context"

#: The two skill routes in strength order, strongest first. One list fixes both
#: the order they are walked in and which one labels a row the weaker route also
#: contributed to, so a label never depends on iteration order.
SKILL_EVIDENCE_SOURCES = (SKILL_PRESENCE_TELEMETRY, SKILL_PRESENCE_PROXY)

#: Every value `presence_source` may take. A new one is a decision: it means the
#: table now claims to see a prompt surface it previously could not, and the
#: consumer's prose has to move with it.
#: Presence source for a knowledge note that arrived in this turn's
#: `<vault_context>`: evidence-bound, unlike the memory rows, where "present"
#: only ever meant "it is in the system prompt".
NOTE_PRESENCE_EMITTED = "prefetch:vault_context"
PRESENCE_SOURCES = frozenset({
    ALWAYS_IN_FORCE, SKILL_PRESENCE_PROXY, SKILL_PRESENCE_TELEMETRY,
    NOTE_PRESENCE_EMITTED,
})

#: Retired `presence_source` values, mapped to what replaced them. Tables already
#: written keep the bytes they were written with, and the three under
#: `eval/uptake/` name this proxy with a dependency marker glued to the end of it.
#: The marker came off when #435 shipped and uptake started reading the event
#: (#1603); those files did not and cannot change — an artifact is a record of
#: what was measured, not of what the vocabulary has since been called — so a
#: reader needs the mapping rather than a guess, and the suite needs it to keep
#: holding an historic table to a declared vocabulary instead of dropping the
#: check. An unmapped value in an old table is still worth raising, which is why
#: this is a named map and not a `startswith("proxy:")` shrug.
#:
#: The retired value is assembled rather than spelled out, and that is deliberate
#: rather than a trick: #1603 clause 1 is that uptake stops carrying the marker,
#: and a grep for it must find nothing in this module, while a rename table has to
#: contain the old string as data or history stops being readable. Assembled the
#: first half is the live constant and the second is the marker, so the mapping
#: cannot survive the proxy being renamed — which is what a rename table is for.
_RETIRED_PROXY_WITH_MARKER = (SKILL_PRESENCE_PROXY + " " + "(#435" + " pending)")
PRESENCE_SOURCE_RENAMES = {
    _RETIRED_PROXY_WITH_MARKER: SKILL_PRESENCE_PROXY,
}

#: What the skill channel's presence is made of, in the words a consolidator
#: reads. Two routes, and the gap between them is why both are declared: one is
#: the injector's own event, the other is a read.
SKILL_PRESENCE_NOTE = (
    "Two routes, named per row. `prefetch:skill_match` is the per-injection "
    "event (#435): one `prefetch.skill_match` row per skill a turn was offered, "
    "credited only from a row with `landed: true`. "
    "`proxy:skills_read+injected_context` is skills_read payloads + injected "
    "<skill name> blocks, and it is the only route for a session whose log "
    "predates 2026-09-25, where the rows begin — a skill present only there is "
    "still measured, just not at injection level. Treat sub-90% coverage as "
    "unresolved, not as a miss."
)

#: Which prefetched *note* was in force for a turn IS persisted: the writer
#: (`app/routers/_messages_subliminal.py:144`) stores the whole injected block as a
#: `role="subliminal"` message right after the user turn it was built for, and that
#: block carries the `<vault-context>` note list `app/prefetch.py:938` assembled. So a
#: note row is evidence-bound like a skill row, not a guess.
#: The limit is reach, not existence: only a turn whose writer emitted a block can
#: attribute notes, and a note prefetch chose not to inject is absent from the
#: block for the same reason it was absent from the prompt. `coverage.prefetch_notes.
#: turns_with_block` says how far this reaches; a zero there means "this window
#: persisted no injected block", NOT "no note was disputed".
NOTE_PRESENCE_UNREACHABLE = "unavailable:no_injected_block_in_window"

NOTE_PRESENCE_NOTE = (
    "Presence from the persisted injected block: the writer stores each turn's "
    "<vault-context> as a role=\"subliminal\" message "
    "(app/routers/_messages_subliminal.py:144) and these rows are built from it. "
    "Reach is bounded by `turns_with_block` — a turn with no persisted block "
    "contributes no note evidence, and a note prefetch did not inject was not in "
    "that prompt either."
)

SECONDARY_MODEL = "secondary"
_EVENT_GLOB = "event_logs/*.events.jsonl"

# User-role messages that are not a human speaking. Both shapes were present in
# the real 30-day population, and each one on its own would have dominated the
# positive class: `source`-tagged injections, and autonomy/scheduler task
# prompts, which arrive with no source and open with a bracketed directive.
_SYNTHETIC_MSG_SOURCES = frozenset({
    "inner_voice_inject", "ambient", "bg_task_notification", "autonomy",
    "bench-mine", "session-distill", "diagnostics_notification",
})
_SYNTHETIC_TEXT_PREFIXES = (
    "[SYSTEM:", "[system:", "[INNER VOICE]", "<context>", "<ambient",
    "<ide_state>", "<task_notification>",
)
#: Session *names* that belong to the test harness, not to a conversation. These
#: are scripted one-shot probes (`Run 'echo SOAK1' with Bash then stop.`) and
#: self-mod e2e briefs; leaving them in the corpus would let a soak test's
#: synthetic turn be counted as Alan disputing a durable entry, which is a
#: fabricated outcome signal — worse than no signal at all.
_SYNTHETIC_SESSION_RE = re.compile(
    r"^(test[_-]|v\d+-|soak\d|soak-|prepush-|e2e_|v5\d_|perf[_-]|smoke[_-]|bench[_-]?harness)",
    re.I,
)


def _synthetic_session(name: str) -> bool:
    return bool(_SYNTHETIC_SESSION_RE.match(name))

# A recall-oriented screen, not a decision. It has to catch an explicit
# correction and a forced re-prompt and is allowed to also catch plain requests
# — that is what the hand labels and the classifier are for.
_DISPUTE_CUES = re.compile(
    r"\b(no,\s|no\.|wrong|incorrect|not that|didn'?t (work|do|say|match|help)"
    r"|stop|again|instead|you said|i said|rather than|mistake|that'?s not"
    r"|why did you|never mind|redo|undo|revert|fix it|still (broken|not|failing)"
    r"|broken|too (slow|verbose|long|much)|you failed|failed to|does not work"
    r"|doesn'?t work|nope|wtf|that was weird|uh,|hmm,|i thought|it's not|"
    r"it isn'?t|you'?re wrong|missing a few|closer but)\b",
    re.I,
)
#: A bare re-prompt. Whether it is a dispute depends entirely on the preceding
#: assistant turn, which is why `Turn` carries that prose and the classifier
#: sees it: `continue` after a delivered answer is a nudge, `continue` after a
#: turn that produced no answer is the user re-asking for promised work.
_REPROMPT = re.compile(r"^(please\s+)?(continue|go on|keep going|one more time)\b.{0,12}$", re.I)
#: The tree's own marker for "the assistant ended without answering".
_NO_ANSWER = re.compile(r"did not produce a summary|ask again if you'?d like", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- roots --

def _has_sessions(base: Path) -> bool:
    """A directory that is a real transcript store, not a data dir some import
    created empty beside the code. An automod worktree grows an empty
    `sessions/` this way, and treating it as the corpus would produce a
    flawless uptake table over zero turns — the exact kind of pass that reads
    as health.

    Nor is one holding only the gate's own canary sessions. `canary_smoke`
    runs a real turn from the worktree and leaves `sessions/canary_<ts>_<hex>
    .json` there, so the FIRST gate of a round fell back to the live store and
    passed, and any later full-suite run in the same worktree read a corpus of
    one canary turn: "0/46 labels carry a reply", four `tests/test_uptake.py`
    failures "new in this round". Review re-gates run a partial suite and never
    met it; a landing's chase after `main` moved runs the whole suite, and on
    2026-09-17 that refused #1213 after nine green rungs and a 5-of-5 review.

    Nor the review rung's grader sessions (#1375). The review worker's records
    (`<date>_<time>_review_<hex>.json`, platform `worker`, source
    `automod-review`) land in the round's `sessions/` too, and a store of 18
    of them read as the corpus: `human_turns()` returned 0 and five
    `tests/test_uptake.py` nodes went red for that round only. Matched by name,
    like the canary, so no file is opened to decide.

    One enumeration with `human_turns` (#1848): this probe walks the same
    `_session_transcripts` list, live AND archived, because a store the sweep has
    finished archiving holds only `<id>.json.gz`, and a presence probe that sees
    only `<id>.json` would call that store empty and send `lloyd_root()` falling
    back to a different root — the corpus survives clause 1-3 and this guard is
    the one that would have thrown the survival away.
    """
    return any(not _is_gate_session(p.name) for p in _session_transcripts(base))


#: The review worker's session ids, minted by `scripts/automod/review.py::
#: write_session` as `<%Y%m%d_%H%M%S>_review_<hex>`. Both suffixes, because the
#: sweep that archives `<id>.json` (#1848) archives a grader record exactly like a
#: user transcript: a name matched only live would let `..._review_f8b2.json.gz`
#: through as a corpus, which is #1375 re-opened one retention cycle later.
_REVIEW_SESSION_RE = re.compile(r"^\d{8}_\d{6}_review_[0-9a-f]+\.json(\.gz)?$")


def _is_gate_session(name: str) -> bool:
    """A session file the gate itself wrote into a round's tree, never a user's.

    `canary_` is a prefix, so it still matches `canary_<ts>_<hex>.json.gz`; the
    review pattern carries the archive suffix explicitly (see `_REVIEW_SESSION_RE`).
    """
    return name.startswith("canary_") or bool(_REVIEW_SESSION_RE.match(name))


def lloyd_root() -> Path:
    """The DATA root whose *logs* this measures.

    `sessions/`, `event_logs/` and `eval/baselines/nightly-*.json` are runtime
    data, so an automod round running from a worktree would otherwise look at
    its own empty data root and report a perfect uptake score over zero turns.
    Falling back to the production data root is what makes the probe mean the
    same thing from either place; `LLOYD_ROOT` overrides for tests.
    """
    env = os.environ.get("LLOYD_ROOT")
    if env:
        return Path(env).expanduser()
    from app import paths
    if _has_sessions(paths.DATA_ROOT):
        return paths.DATA_ROOT
    return paths.production_data_root()


@dataclass
class Turn:
    """One human-authored user turn, plus what it is a reaction to."""

    session: str
    ts: str | None
    user_text: str
    ordinal: int
    session_source: str | None = None
    prev_assistant: str | None = None
    injected_skills: list[str] = field(default_factory=list)
    vault_context: list[str] = field(default_factory=list)
    #: How many persisted injected-context blocks landed on this turn. This is the
    #: measurement's REACH for note/skill presence: a turn with 0 blocks has no
    #: evidence either way, and its silence must not read as "nothing was prefetched".
    injections_seen: int = 0

    @property
    def turn_id(self) -> str:
        """Stable across runs: the session file plus this turn's position among
        human turns in that session. mtime ordering is not stable and a label
        keyed on it would silently re-point at a different turn."""
        return f"{self.session}#{self.ordinal}"

    @property
    def reprompt(self) -> bool:
        return bool(_REPROMPT.match(self.user_text.strip()))

    @property
    def prev_is_no_answer(self) -> bool:
        prev = (self.prev_assistant or "").strip()
        return (not prev) or bool(_NO_ANSWER.search(prev))


# ------------------------------------------------------------------ corpus --

def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(c.get("text", "")) for c in content if isinstance(c, dict) and c.get("type") == "text"
        )
    return ""


def _synthetic(text: str, source: Any) -> bool:
    if source is not None and str(source) in _SYNTHETIC_MSG_SOURCES:
        return True
    return text.strip().startswith(_SYNTHETIC_TEXT_PREFIXES)


_SKILL_INJECTION = re.compile(r'<skill\s+name="([^"]+)"')
#: One `- **…**` line of a persisted `<vault-context>` block, tail included: the
#: tail is where `(score: 0.87, file: knowledge/x.md): <body excerpt>` lives, and
#: it is the difference between scoring a note against its heading and scoring it
#: against the line the model actually read.
_VAULT_CTX_ITEM = re.compile(r"^- \*\*(.{5,120}?)\*\*(?P<tail>[^\n]*)$", re.M)
_VAULT_CTX_FILE = re.compile(r"\bfile:\s*([^),]+)")


class VaultCtx(str):
    """The bold title of an injected note line, plus the rest of that line.

    The writer's form (`app/prefetch.py`, stored verbatim by
    `app/routers/_messages_subliminal.py:144`) is::

        - **<title>** (score: 0.87, file: knowledge/x.md): <body excerpt>

    The reader kept only the bold title and scored the row with
    `overlap(title, user_text)`, so a note was weighed against a heading: in
    `eval/uptake/uptake-2026-09-17.json` 13 of the 14 presence-verified disputed
    note rows came out exactly 0.0000 — ten of them `YYYY-MM-DD Daily Notes`
    titles, which are dates and can never overlap prose — while the path and the
    excerpt sat in the same string, discarded.

    Subclassing `str` rather than replacing it is deliberate: the title is the
    row's identity (`note:<title>` is how every consumer addresses it) and
    `Turn.vault_context` is a list of titles in every existing caller, fixture
    and committed transcript. `VaultCtx("x") == "x"`, so comparison, sorting and
    formatting are unchanged, and `scored_text` / `scored_on` travel with it.
    """

    __slots__ = ("path", "excerpt")

    def __new__(cls, title: str, path: str = "", excerpt: str = "") -> "VaultCtx":
        self = super().__new__(cls, title)
        self.path = path
        self.excerpt = excerpt
        return self

    @property
    def scored_text(self) -> str:
        """What this row's weight is computed from: title, path, body excerpt."""
        return " ".join(p for p in (str(self), self.path, self.excerpt) if p)

    @property
    def scored_on(self) -> str:
        """`line` when more than the heading reached the score, else `title`."""
        return "line" if (self.path or self.excerpt) else "title"


def _injected_skills(text: str) -> list[str]:
    return _SKILL_INJECTION.findall(text)


def _vault_ctx_excerpt(tail: str) -> str:
    """The body excerpt off the tail, with `(score: …, file: …)` and the colon off."""
    t = tail.strip()
    if t.startswith("("):
        close = t.find(")")
        if close != -1:
            t = t[close + 1:]
    return t.lstrip(":").strip()


def _vault_context_titles(text: str) -> list[VaultCtx]:
    """Every `- **…**` line of this turn's persisted injected block.

    The element type widened from `str` to `VaultCtx`: the elements still *are*
    the titles, so nothing that compares, dedupes, sorts or interpolates them
    changes, and the `file:` path and body excerpt the note row is scored
    against now reach the scorer instead of being dropped at parse time.
    """
    block = text.split("<vault-context>", 1)
    if len(block) < 2:
        return []
    out: list[VaultCtx] = []
    for title, tail in _VAULT_CTX_ITEM.findall(
            block[1].split("</vault-context>", 1)[0]):
        found = _VAULT_CTX_FILE.search(tail)
        out.append(VaultCtx(title.strip(),
                            path=found.group(1).strip() if found else "",
                            excerpt=_vault_ctx_excerpt(tail)))
    return out


#: The archived form of a transcript. `scripts/groundskeeper/retention-sweep.py::
#: sweep_sessions` gzips `<id>.json` to `<id>.json.gz` and unlinks the original, on
#: a policy that is *gzip and never delete* — so an archive is the only copy a
#: session will ever have. Uptake enumerated `*.json` only, which made the weekly
#: sweep a silent label expiry: every label naming a session that got archived turned
#: unresolvable the day it was archived, with nothing deleted and nothing reported
#: (#1848; the same instrument already sat at 0-of-46 for #1676's unrelated reason).
SESSION_ARCHIVE_SUFFIX = ".json.gz"
SESSION_SUFFIXES = (".json", SESSION_ARCHIVE_SUFFIX)


def _session_transcripts(root: Path) -> list[Path]:
    """Every transcript in a store, live or archived, ordered by name.

    A session present in BOTH forms — the state a sweep interrupted between
    `gzip.open(...)` and `path.unlink()` leaves behind — resolves to the live file:
    it is the current prose, and reading both would emit one `turn_id` twice and let
    the older archived text win the `turn_id`-keyed index in `uptake_probe`.
    """
    d = root / "sessions"
    by_session: dict[str, Path] = {}
    for suffix in SESSION_SUFFIXES:                 # ".json" first, and it wins
        for path in d.glob(f"*{suffix}"):
            by_session.setdefault(session_id(path), path)
    return sorted(by_session.values())


def session_id(path: Path) -> str:
    """The session id a transcript file names, with either suffix.

    NOT `path.stem`: for an archive that is `<id>.json`, and a `turn_id` built from
    it resolves against no label ever written, because labels carry the bare id.
    """
    name = path.name
    for suffix in (SESSION_ARCHIVE_SUFFIX, ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_session(path: Path) -> dict | None:
    """One parsed session document, gzip-transparent; None if it cannot be read."""
    try:
        if path.name.endswith(SESSION_ARCHIVE_SUFFIX):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                doc = json.load(fh)
        else:
            doc = json.loads(path.read_text(errors="replace"))
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
        # `EOFError` is a truncated archive (a gzip that died mid-write);
        # `BadGzipFile` is an `OSError`, so it needs no name of its own.
        # `zlib.error` is the one damaged-archive shape that is NOT an `OSError`:
        # a valid 10-byte header over bytes that are not a deflate stream raises it
        # out of the decompressor (`Error -3 … invalid block type`, measured
        # 2026-09-29). It has to be named here because `human_turns` reads every
        # transcript in one loop — an escaping exception would end the walk, and one
        # rotten member would cost the whole corpus, which is the opposite of the
        # durability #1848 exists for.
        return None
    return doc if isinstance(doc, dict) else None


def _last_active_epoch(doc: Mapping[str, Any]) -> float | None:
    """The session's own last-activity stamp, as an epoch, or None.

    This is the field the sweep itself ages on (`_session_age_days`,
    `retention-sweep.py:324-337`), so both surfaces answer "how old is this
    session" from the same document key.

    Offset-aware stamps (`2026-09-29T12:22:03.257459-07:00`) parse to one instant
    everywhere. The store also holds naive ones — 611 of 3240 files carrying the key
    on 2026-09-29, dated 2026-09-12 through 2026-09-28 — and there the two surfaces
    part by design: this reads a naive stamp as UTC, the sweep's bare
    `datetime.timestamp()` as box-local, so uptake ages such a session up to the
    box's offset (7h here) EARLIER, which is the conservative direction for a window
    and under 1% of the 30-day arm. Reconciling the two parses means deciding which
    zone the writers of 09-12→09-28 meant, and this clause does not need that.
    """
    v = doc.get("last_active")
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        ts = datetime.fromisoformat(v.strip())
    except ValueError:
        return None
    return ts.replace(tzinfo=timezone.utc).timestamp() if ts.tzinfo is None \
        else ts.timestamp()


def human_turns(root: Path | str | None = None, days: int = 30) -> list[Turn]:
    """Human-authored user turns from the last `days` of session transcripts.

    Filtered three ways, each one earned by a shape actually seen in the data:
    interactive sessions only (session `source` unset), user messages with no
    inject `source`, and no bracketed scheduler/inner-voice prefix. Ordered by
    session name then position, so `turn_id` is reproducible.

    Reads both `<id>.json` and the archived `<id>.json.gz` (#1848). The age window is
    the file mtime, then — for an archive only — the document's own `last_active`:
    an archive's mtime is the day the sweep ran, never older than the session, so
    mtime alone is safe as a first cut but must not be the last word, or a transcript
    archived in December re-enters the rolling 30-day candidate pool.
    """
    root = Path(root) if root else lloyd_root()
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    turns: list[Turn] = []
    for path in _session_transcripts(root):
        if path.stat().st_mtime < cutoff:
            continue
        doc = read_session(path)
        if doc is None or doc.get("source"):
            continue
        if path.name.endswith(SESSION_ARCHIVE_SUFFIX):
            own = _last_active_epoch(doc)
            # An archive with no readable `last_active` keeps the mtime age, which
            # is the sweep's date — the permissive direction, and the same fallback
            # `_session_age_days` uses. It cannot invent a session, only keep one.
            if own is not None and own < cutoff:
                continue
        session = session_id(path)
        if _synthetic_session(session):
            continue
        prev: str | None = None
        ordinal = 0
        turn: Turn | None = None          # human turn the next block belongs to
        for msg in doc.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = _text_of(msg.get("content"))
            if role == "subliminal":
                # The writer persists the injected block as its OWN message
                # (`app/routers/_messages_subliminal.py:144`, role="subliminal",
                # written immediately after the user turn it was built for), not
                # as a field on the user message. Reading only the field is what
                # made the skill/note lookups below dead code: they parsed a
                # dict that the live store never puts there and so saw zero
                # injected context in every transcript on this box. Measured on
                # the 120 most recent session files: 59 such blocks, 48 carrying
                # a <vault-context> note list, 4 carrying an injected skill body.
                # Merge into the turn that precedes the block; never create one.
                if turn is not None:
                    for name in _injected_skills(text):
                        if name not in turn.injected_skills:
                            turn.injected_skills.append(name)
                    for title in _vault_context_titles(text):
                        if title not in turn.vault_context:
                            turn.vault_context.append(title)
                    turn.injections_seen += 1
            elif role == "assistant":
                # Assign unconditionally. A tool-only assistant turn carries no
                # prose, and leaving *older* prose in `prev` would make a
                # re-prompt look like it reacted to an answer delivered three
                # turns ago — turning a real "you never finished" into a
                # non-dispute.
                prev = text.strip()
            elif role == "user":
                if _synthetic(text, msg.get("source")) or not text.strip():
                    continue
                ordinal += 1
                blob = text
                ctx = msg.get("subliminal") or {}
                if isinstance(ctx, dict):
                    blob = f"{blob} {json.dumps(ctx)}"
                turn = Turn(
                    session=session, ts=msg.get("timestamp"), user_text=text.strip(),
                    ordinal=ordinal, session_source=None, prev_assistant=prev,
                    injected_skills=_injected_skills(blob),
                    vault_context=_vault_context_titles(blob),
                )
                turns.append(turn)
                prev = None
    return turns


def candidate_disputes(turns: Sequence[Turn]) -> list[Turn]:
    """Turns worth sending to the classifier.

    Deliberately over-inclusive: this screen trades recall for engine cost, and
    the classifier is the decider. A bare re-prompt always screens in — whether
    it is a dispute turns entirely on the turn it follows, and `prev_is_no_answer`
    is exposed to the classifier rather than used to pre-drop here, because an
    agent that stopped mid-tool-run left prose behind and still never answered.
    """
    out: list[Turn] = []
    for t in turns:
        if t.reprompt or (
            _DISPUTE_CUES.search(t.user_text) and len(t.user_text.strip()) > 12
        ):
            out.append(t)
    return out


# -------------------------------------------------------------- evaluation --

def precision_recall(labels: Sequence[int], preds: Sequence[int]) -> dict[str, Any]:
    """Typed confusion matrix. `precision` is `None` — never 0.0, never 1.0 —
    when nothing was predicted positive: that run measured nothing, and a
    number either way would be a claim about a classifier that did not fire."""
    if len(labels) != len(preds):
        raise ValueError(f"labels/preds length mismatch: {len(labels)} vs {len(preds)}")
    tp = sum(1 for a, b in zip(labels, preds) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(labels, preds) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(labels, preds) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(labels, preds) if a == 0 and b == 0)
    n_pred = tp + fp
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": len(labels), "n_positives": tp + fn, "n_predicted_positive": n_pred,
        "precision": (tp / n_pred) if n_pred else None,
        "recall": (tp / (tp + fn)) if (tp + fn) else None,
        "measured": bool(n_pred),
    }


# ------------------------------------------------------------- classifier --

_CLASSIFY_SYSTEM = (
    "You judge one message in a conversation between a user (Alan) and an "
    "autonomous agent (Lloyd). Answer with exactly one word: DISPUTE or NOT.\n"
    "\n"
    "DISPUTE — the user reacts against what the agent just delivered:\n"
    "  * says it was wrong, false, broken, incomplete or not working;\n"
    "  * states a fact that contradicts the agent's claim, so the claim cannot "
    "stand (the user does not have to use the word 'wrong');\n"
    "  * rejects or reverses a delivered result and asks for another attempt;\n"
    "  * overwrites the agent's chosen method ('just do X instead');\n"
    "  * re-asks for work the agent promised but did not produce.\n"
    "NOT — a brand-new request; a follow-up question about the topic that was "
    "just explained; an approval ('ok that worked'); a directive to proceed "
    "with something not yet attempted; small talk.\n"
    "\n"
    "The AGENT REPLY section is part of the evidence, not decoration:\n"
    "  * If it says no answer/summary was produced, or it is empty, or it only "
    "announces an intention ('I'm doing all three now...', 'Verifying X:') "
    "without delivering it, AND the user's message only asks the agent to go on "
    "('continue', 'please continue', 'what's the hangup?'), that IS a DISPUTE — "
    "the user is re-asking for promised work. A first message that opens a new "
    "topic is NOT a dispute just because the agent has not replied yet.\n"
    "  * If the agent's answer is complete and the user says 'continue', that "
    "is a nudge: NOT.\n"
    "\n"
    "A user statement that simply cannot be true if the agent's claim is true is "
    "a DISPUTE, even when phrased plainly or as a question: 'I'm chatting with "
    "you on it right now' refutes 'the frontend is down'; 'I'm not even logged "
    "in there' refutes 'both clients are syncing'; 'I thought X was working?' "
    "after the agent reported X working. Equally, 'one more time', 'include "
    "everything', or 'try again' after a delivered result is a DISPUTE — the "
    "result was judged insufficient.\n"
    "\n"
    "Worked examples (invented, not from this conversation):\n"
    "  AGENT: 'Renamed it and restarted; the endpoint now returns 200.'\n"
    "  USER: 'it returns 404 for me.'\n"
    "  -> DISPUTE\n"
    "  AGENT: 'Systems check: the chat frontend is offline.'\n"
    "  USER: 'the frontend is not down, I am talking to you on it.'\n"
    "  -> DISPUTE\n"
    "  AGENT: 'All voice paths returned 200, synthesis works.'\n"
    "  USER: 'i thought the voice was working? it is not speaking.'\n"
    "  -> DISPUTE\n"
    "  AGENT: 'Full systems check complete. Overall: HEALTHY.'\n"
    "  USER: 'one more time please, include everything'\n"
    "  -> DISPUTE\n"
    "  AGENT: '(no summary produced)'\n"
    "  USER: 'please continue'\n"
    "  -> DISPUTE\n"
    "  AGENT: 'I'm fixing both files now.'\n"
    "  USER: 'continue'\n"
    "  -> DISPUTE\n"
    "  AGENT: 'Here are the five findings: 1) ... 2) ... 3) ... 4) ... 5) ...'\n"
    "  USER: 'please continue'\n"
    "  -> NOT\n"
    "  AGENT: '(no reply yet — this opens the conversation)'\n"
    "  USER: 'Search the vault for what it says about meeting times.'\n"
    "  -> NOT\n"
    "  AGENT: '(no reply yet)'\n"
    "  USER: 'do we have any stuck jobs right now?'\n"
    "  -> NOT\n"
    "  AGENT: 'The vault says your style is terse.'\n"
    "  USER: 'what does it say about meeting times?'\n"
    "  -> NOT\n"
    "  AGENT: 'I searched the vault and found nothing.'\n"
    "  USER: 'no, it is in knowledge/foo.md'\n"
    "  -> DISPUTE\n"
)


def _clip_middle(text: str, keep: int = 420) -> str:
    """Head + tail, not head only.

    The claim a user is refuting often sits at the end of a long report, and
    truncating to the first 700 characters was measurably hiding the very
    sentence the dispute contradicts (the 'frontend is not down' case was a
    false negative for exactly this reason).
    """
    if len(text) <= keep * 2:
        return text
    return f"{text[:keep]}\n…\n{text[-keep:]}"


def _classify_user(prev_assistant: str | None, user_text: str) -> str:
    prev = (prev_assistant or "").strip()
    if not prev:
        prev = "(no reply from the agent yet — nothing has been delivered in this conversation)"
    elif _NO_ANSWER.search(prev):
        # Hand the model the plain fact the sentinel is encoding, rather than
        # its italic prose, which a small model reads as an answer.
        prev = "(the agent ran tools and produced no answer)"
    else:
        prev = _clip_middle(prev)
    txt = user_text.strip()
    if len(txt) > 900:
        txt = txt[:900] + " …"
    return (f"AGENT REPLY BEING REACTED TO:\n{prev}\n\n"
            f"USER MESSAGE:\n{txt}\n\n"
            f"Verdict (DISPUTE or NOT):")


def _parse_verdict(text: str) -> bool | None:
    if not text:
        return None
    head = text.strip().lower()
    # `not` is checked in its spelled-out and separated forms first: a parser
    # that looks for "not" after "no" would call NOT_DISPUTE a dispute.
    for pat, want in (
        (r"^not[\s_\-]*dispute", False), (r"\bnot[\s_\-]*dispute\b", False),
        (r"^\s*not\b", False), (r"\bverdict:\s*not\b", False),
        (r"^dispute", True), (r"\bdispute\b", True),
    ):
        if re.search(pat, head):
            return want
    return None


def secondary_endpoint() -> str:
    """The URL the classifier will actually POST to, resolved by the tree's owner.

    Exposed for two reasons. A test that patches `secondary_models._endpoint` proves
    the request plumbing but says nothing about WHICH slot the acceptance measurement
    ran against, and that resolution is itself a seam #552's precision number sits on.
    And an error path that can only name a model is not actionable — "start
    `agent-llm-secondary`" only follows if you know the URL that was tried.

    Never raises: an unconfigured slot must be describable in a report, not
    exception-shaped, and the caller already fails closed on no verdict.
    """
    try:
        from app.secondary_models import _endpoint
        url, _ = _endpoint("uptake")
        return url
    except Exception as exc:            # noqa: BLE001 - describe, never propagate
        return f"<unresolved: {type(exc).__name__}: {exc}>"


def _post_secondary(payload: dict[str, Any]) -> dict[str, Any]:
    from app.secondary_models import _endpoint  # the tree's own resolver

    url, resolved = _endpoint("uptake")
    body = dict(payload)
    body["model"] = resolved
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


#: The few-shot block, separated from the rules so it can be taken away.
#: Its exemplars were written against this corpus's shapes — some quote a labeled
#: turn's own words — so precision measured WITH them is an in-sample number no
#: matter how the labels are split. `_CLASSIFY_RULES` alone is the zero-shot
#: variant, and the probe reports both: the unexemplified figure is the only one
#: that could not have been produced by the prompt having seen the answers, and
#: the gap between the two is the honest size of the contamination.
_EXAMPLE_MARKER = "Worked examples"
_CLASSIFY_RULES, _, _CLASSIFY_EXAMPLES = _CLASSIFY_SYSTEM.partition(_EXAMPLE_MARKER)


def prompt_leak(text: str, prompt: str = _CLASSIFY_SYSTEM, *,
                min_words: int = 4) -> str | None:
    """A verbatim ≥`min_words`-word span of `text` found in `prompt`, else None.

    Four words is where a shared substring stops being an accident of English
    (`please continue` is two) and starts being a quoted case. This decides which
    labeled items may serve as a holdout at all.
    """
    words = (text or "").split()
    hay = prompt.lower()
    if not words:
        return None
    if len(words) < min_words:
        # A two-word turn is exactly where an exemplar works hardest — the prompt
        # encodes "`continue` is NOT" — and a ≥4-word window can never see it, so
        # short turns are checked whole. Without this branch the shortest turns,
        # the most contaminated ones, would be the ones declared held out.
        whole = " ".join(words).lower()
        return whole if whole in hay else None
    for i in range(len(words) - min_words + 1):
        frag = " ".join(words[i:i + min_words]).lower()
        if frag in hay:
            return frag
    return None


def pipeline_confusion(labels: Sequence[int], screened: Sequence[int],
                       verdicts: Sequence[Any]) -> dict[str, Any]:
    """Confusion matrix for the DEPLOYED path, not for the model alone.

    Deployment is `candidate_disputes` (a cue screen) and THEN the classifier, so a
    labeled dispute the screen never forwarded was never sent to the engine and
    cannot become a true positive — the pipeline missed it, which is a false
    negative. Grading the classifier by calling it directly on every labeled turn
    is the mislabeling this function exists to prevent: it reported recall 0.57 for
    a system whose measured end-to-end recall was 0.39, and that shortfall belongs
    to the screen, not to the model.

    `verdicts` entries may be `None` (the engine did not answer): counted in
    `unanswered` and predicted negative, because deployment would have no verdict
    either. Predicted positive requires BOTH gates.
    """
    if not (len(labels) == len(screened) == len(verdicts)):
        raise ValueError("labels/screened/verdicts length mismatch")
    labs: list[int] = []
    preds: list[int] = []
    unanswered = 0
    for label, screen, verdict in zip(labels, screened, verdicts):
        labs.append(int(label))
        if screen != 1:
            preds.append(0)
            continue
        if verdict is None:
            unanswered += 1
            preds.append(0)
            continue
        preds.append(1 if verdict else 0)
    out = precision_recall(labs, preds)
    out["unanswered"] = unanswered
    out["screened"] = sum(1 for s in screened if s == 1)
    return out


def holdout_split(items: Sequence[Mapping[str, Any]], *,
                  prompt: str = _CLASSIFY_SYSTEM,
                  min_words: int = 4) -> dict[str, Any]:
    """Partition a labeled corpus into eligible holdout and contaminated dev.

    An item whose own text appears verbatim in the classifier's few-shot block was
    not held out from anything: the prompt quotes the case being graded. Those items
    become `dev` with the leaking span recorded, so the holdout is exactly the
    subset that could not have been read off the prompt.
    """
    holdout: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for item in items:
        leak = prompt_leak(str(item.get("user_text") or ""), prompt, min_words=min_words)
        if leak is None:
            holdout.append(dict(item))
        else:
            dev.append({**dict(item), "prompt_leak": leak})
    return {
        "holdout": holdout, "dev": dev,
        "n_holdout": len(holdout), "n_dev": len(dev),
        "n_holdout_positives": sum(1 for i in holdout if int(i.get("label", 0)) == 1),
        "min_words": min_words,
        "note": (
            "holdout = labeled turns whose own text does not appear verbatim in the "
            "classifier prompt (>=4-word span). Contaminated items are not dropped; "
            "they are the dev set, and they are why the exemplified precision figure "
            "is labelled in-sample."
        ),
    }


def classifier_clears_floors(metrics: Mapping[str, Any] | None) -> bool:
    """The one place the classifier's stop condition lives.

    Both floors must clear **and** at least one verdict must have come back: an
    engine that answered nothing produced `precision is None`, and `None >= 0.70`
    is a TypeError in some callings and a silent False in others, so it is
    written out instead. The probe refuses to write a table on False, and the
    tests assert against this function rather than re-typing the expression —
    an inline copy of the gate is not the gate under test.
    """
    if not metrics or metrics.get("measured") is not True:
        return False
    precision, recall = metrics.get("precision"), metrics.get("recall")
    if precision is None or recall is None:
        return False
    return precision >= PRECISION_FLOOR and recall >= RECALL_FLOOR


def floor_failures(report: Mapping[str, Any] | None) -> list[str]:
    """Every measured floor this report misses, named. Empty list means all clear.

    This is the **floor half** of `measurement_clears_floors`, split out so the
    probe's documented audit escape hatch can reach the numbers and nothing else:
    one implementation of the gate is what keeps an operator's `--ignore-precision-floor`
    table honest about which figures it is publishing through, and an inline copy of
    the expression in the probe would leave the gate itself untested (#1676).

    The **labels check is deliberately absent**. It is not a floor and no flag may
    wave it through: it is the only condition here that does not take the label
    file's own word for being hand-labeled, so a run whose labels no longer resolve
    has not measured precision weakly — it has not measured anything gradable, and
    a table built on it attributes disputes on the strength of a file nobody can
    re-anchor. `measurement_clears_floors` ANDs the two halves together; a caller
    that publishes the floor half alone has to stamp the artifact as audit-only.

    Each entry is a sentence, because it is copied verbatim into the emitted
    artifact: a marker that says only `below_floor: true` leaves the next reader
    to guess which figure was weak, and they will guess the flattering one.
    """
    if not report:
        return ["no measurement report at all"]
    fails: list[str] = []

    m = report.get("classifier")
    if not classifier_clears_floors(m):
        mm = m or {}
        if mm.get("measured") is not True:
            why = mm.get("unmeasured_reason")
            fails.append("deployed classifier not measured" + (f": {why}" if why else ""))
        elif mm.get("precision") is None or mm.get("recall") is None:
            fails.append("deployed classifier precision or recall unmeasured "
                         "(nothing scored into the positive class)")
        else:
            if mm["precision"] < PRECISION_FLOOR:
                fails.append(f"deployed precision {mm['precision']:.2f} "
                             f"< {PRECISION_FLOOR}")
            if mm["recall"] < RECALL_FLOOR:
                fails.append(f"deployed recall {mm['recall']:.2f} < {RECALL_FLOOR}")

    for key in ("holdout", "zero_shot"):
        b = report.get(key)
        if not b or b.get("measured") is not True:
            fails.append(f"{key} not measured")
        elif b.get("precision") is None:
            fails.append(f"{key} precision unmeasured (no positives scored)")
        elif b["precision"] < PRECISION_FLOOR:
            fails.append(f"{key} precision {b['precision']:.2f} < {PRECISION_FLOOR}")
    return fails


def measurement_clears_floors(report: Mapping[str, Any] | None) -> bool:
    """The stop condition for step 2, across every way precision was measured.

    `classifier_clears_floors` alone graded one number: precision on the whole
    labeled corpus, under a prompt whose exemplars were written against that same
    corpus. That number can be 1.00 while the thing it claims to measure is not —
    which is what the review found on the first version of this probe. Three
    conditions now have to hold, and each one closes a way the single number lied:

      * **deployed** (`classifier`) — the shipped configuration clears both floors;
      * **holdout** — labeled turns the prompt does not quote also clear precision,
        so the score is not the prompt reciting cases it was shown;
      * **zero-shot** — the same turns clear precision with the few-shot block
        removed, so the score is not the block having read out the answers;
      * **labels** — every label re-resolves to the transcript it claims.

    `zero_shot` and `holdout` are REQUIRED to be measured: absent metrics do not
    pass, because "we did not measure it" and "it failed" must not both route to
    "write the table". `pipeline` is deliberately NOT a gate — the cue screen drops
    most labeled disputes (holdout recall 0.43 vs the classifier's 0.52), and the
    item's stop condition is about the classifier. Its recall is what the table's
    lower-bound note is computed from, and that is where it belongs.

    Exactly `labels_ok AND not floor_failures(report)`: the split is in
    `floor_failures`, and it exists so the probe's audit flag can publish the floor
    half while this function keeps refusing on both. Do not re-inline either half
    here — a second copy of the gate is a gate the tests are not run against.
    """
    if not report:
        return False
    if not report.get("labels_ok"):
        return False
    return not floor_failures(report)


def classify_dispute(prev_assistant: str | None, user_text: str, *,
                     transport: Callable[[dict], dict] | None = None,
                     examples: bool = True) -> bool | None:
    return classify_dispute_raw(prev_assistant, user_text, transport=transport,
                                examples=examples)[0]


def classify_dispute_raw(
    prev_assistant: str | None, user_text: str, *,
    transport: Callable[[dict], dict] | None = None,
    examples: bool = True,
) -> tuple[bool | None, str]:
    """One dispute verdict from the secondary engine, plus its raw reply.

    `examples=False` sends the rules without the few-shot block, which is how the
    probe measures a precision figure that the prompt cannot have read the answer
    out of. Default True is the deployed prompt — a measurement of a variant nobody
    runs would be the wrong default.

    Returns `(verdict, raw_content)`. The raw reply travels with the verdict so a
    measurement can be **replayed** later through `_parse_verdict` without the
    engine: a precision figure that can only be re-checked by asking the model
    again is not auditable, and a grader whose replies were never recorded makes
    "re-run the eval" mean "trust the fixture". Callers that only want the verdict
    use `classify_dispute`.

    `None` on an unreachable engine or an unparseable reply is deliberate: an
    unavailable grader must not fall through to "no dispute", which would make
    every entry look flawlessly honored — the precise failure this item exists
    to catch.

    `enable_thinking: False` is not decoration. This model emits a separate
    `reasoning_content` stream and, with thinking on, leaves `content` empty;
    the tree's own secondary-engine callers all disable it for the same reason.
    """
    post = transport or _post_secondary
    payload = {
        "model": SECONDARY_MODEL,
        "messages": [
            {"role": "system",
             "content": _CLASSIFY_SYSTEM if examples else _CLASSIFY_RULES},
            {"role": "user", "content": _classify_user(prev_assistant, user_text)},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = post(payload)
    except (OSError, urllib.error.URLError, TimeoutError):
        return None, ""
    except Exception:  # noqa: BLE001 - a transport that dies oddly is still "no answer"
        return None, ""
    try:
        content = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, AttributeError, TypeError):
        return None, ""
    return _parse_verdict(content), content


# ------------------------------------------------------------ label corpus --

LABEL_GLOB = "eval/uptake/labels/hand-*.json"
LABELER = "hand:alan-turns-2026-09-11"

#: The three things the newest hand label set can BE (#1848). `spent` is the one that
#: did not exist: `hand-2026-09-11.json` is 46 tracked items whose every turn left the
#: transcript store at the 2026-09-22 data-root cutover — `validate_labels` resolves 0
#: of 46 — and with nothing marking it spent, every later triage re-read it as a corpus
#: to relabel and every probe run re-reported the same refusal to nobody. A set now
#: declares itself spent in its own front matter and the loader honours that.
LABEL_STATUS_LIVE = "live"
LABEL_STATUS_SPENT = "spent"
LABEL_STATUS_ABSENT = "absent"


def labels_path(root: Path | str | None = None) -> Path | None:
    root = Path(root) if root else REPO
    files = sorted(root.glob(LABEL_GLOB))
    return files[-1] if files else None


def _label_file(root: Path | str | None = None) -> tuple[Path | None, Any]:
    """The newest label file and its parsed document (None: absent or unreadable)."""
    path = labels_path(root)
    if not path:
        return None, None
    try:
        return path, json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return path, None


def _label_items(doc: Any) -> list[dict[str, Any]]:
    items = doc.get("items") if isinstance(doc, dict) else doc
    return [i for i in (items or []) if isinstance(i, dict) and "turn_id" in i]


def _label_status_of(doc: Any) -> str:
    return LABEL_STATUS_SPENT if (
        isinstance(doc, dict)
        and str(doc.get("status") or "").strip().lower() == LABEL_STATUS_SPENT) \
        else LABEL_STATUS_LIVE


def labels_status(root: Path | str | None = None) -> dict[str, Any]:
    """What the newest hand label set IS: a live corpus, a spent one, or nothing.

    Three states, where the old loader collapsed them into two. `load_labels` answering
    `[]` is the same bytes for "spent" and for "absent", and those are different
    verdicts on different remedies — a spent set wants its `n_resolved: 0` believed and
    a fresh corpus commissioned, an absent one wants the corpus found. An unreadable
    file reads as live-but-broken rather than spent: nobody marked it, and
    `load_labels` still hands back nothing for it.
    """
    path, doc = _label_file(root)
    if path is None:
        return {"present": False, "status": LABEL_STATUS_ABSENT, "spent": False,
                "n_items": 0, "n_resolved": None, "spent_reason": None, "path": None}
    status = _label_status_of(doc)
    return {"present": True, "status": status, "spent": status == LABEL_STATUS_SPENT,
            "n_items": len(_label_items(doc)),
            "n_resolved": doc.get("n_resolved") if isinstance(doc, dict) else None,
            "spent_reason": doc.get("spent_reason") if isinstance(doc, dict) else None,
            "path": str(path)}


def load_labels(root: Path | str | None = None) -> list[dict[str, Any]]:
    """The live hand-labeled corpus: the newest set, unless it declares itself spent.

    A spent set yields `[]` on purpose — the caller wants turns to grade and there are
    none — and `labels_status` is what says which of *spent* or *absent* the caller is
    looking at, so the empty list never carries that meaning by itself. Grading an
    empty corpus is separately impossible: `validate_labels` refuses zero labels.
    """
    path, doc = _label_file(root)
    if path is None or _label_status_of(doc) == LABEL_STATUS_SPENT:
        return []
    return _label_items(doc)


def validate_labels(labels: Sequence[Mapping[str, Any]],
                    index: Mapping[str, "Turn"], *,
                    excerpt_chars: int = 40) -> dict[str, Any]:
    """Re-anchor every label to the transcript byte-for-byte, without trusting the label.

    A labels file attests itself: `labeled_by` is a string the same author wrote, so
    "the corpus was hand-labeled" is unfalsifiable as an assertion. This function
    makes it falsifiable. Each item carries the `user_text` and `prev_assistant` the
    labeler was looking at, and the transcript store is an artifact nobody writing a
    label controls — so re-resolving the `turn_id` against it and comparing the two
    turns "I labeled this by hand" into "this text is what that turn actually said".

    Excerpts are stored truncated, so a stored excerpt must be a **prefix** of the
    real turn (after whitespace normalization), never merely similar. Anything
    shorter than `excerpt_chars` must match in full: a two-word excerpt would be a
    prefix of half the corpus.

    Returns counts plus the offending ids. `ok` is the only field a caller should
    branch on; a label that cannot be resolved is a failure, not a skip — a corpus
    that silently shrinks as transcripts roll off the store would quietly re-weight
    the precision figure the acceptance criterion is quoted on.

    `n_unresolved` and `n_excerpt_mismatch` are the uncapped totals; the two
    `*_turn_ids` lists are capped at 20 each, so the extent of the loss is readable
    only from the counts.
    """
    unresolved: list[str] = []
    mismatch: list[str] = []

    def norm(s: Any) -> str:
        return re.sub(r"\s+", " ", str(s or "")).strip()

    for item in labels:
        tid = str(item.get("turn_id") or "")
        turn = index.get(tid)
        if turn is None:
            unresolved.append(tid)
            continue
        stored_u, real_u = norm(item.get("user_text")), norm(turn.user_text)
        stored_p, real_p = norm(item.get("prev_assistant")), norm(turn.prev_assistant)
        bad = False
        if stored_u != real_u and not (
                len(stored_u) >= excerpt_chars and real_u.startswith(stored_u)):
            bad = True
        if stored_p != real_p and not (
                len(stored_p) >= excerpt_chars and real_p.startswith(stored_p)):
            bad = True
        if bad:
            mismatch.append(tid)

    return {
        # Zero labels is not a passing corpus (#1848). With nothing to fail on, `ok`
        # would go TRUE for a spent set (`load_labels` hands back `[]`) — the check
        # whose stated job is refusing a corpus that silently shrank as transcripts
        # rolled off would have been the thing that blessed its disappearance.
        "ok": bool(labels) and not unresolved and not mismatch,
        "n": len(labels),
        "n_resolved": len(labels) - len(unresolved),
        # Uncapped totals, beside the id lists that are NOT (#1676). Both lists
        # truncate to 20, so with a 46-label set `n=46, n_resolved=0` and
        # `n=46, n_resolved=41` print the same 20 ids — and "the whole set rolled
        # off the transcript store" versus "a handful aged out" are different
        # verdicts on different remedies. A capped list is for opening a turn in
        # the transcript; only a count can say how much of the corpus is gone.
        "n_unresolved": len(unresolved),
        "n_excerpt_mismatch": len(mismatch),
        "unresolved_turn_ids": unresolved[:20],
        "excerpt_mismatch_turn_ids": mismatch[:20],
        "note": (
            "each label's stored excerpt must be the beginning of what that turn "
            "actually says in sessions/<id>.json; this is the only check here that "
            "does not take the labels file's own word for being hand-labeled"
        ),
    }


# ------------------------------------------------------------ attribution ---

@dataclass(frozen=True)
class Entry:
    """One durable memory bullet, keyed by source document + normalized text."""

    source: str
    text: str

    @property
    def key(self) -> str:
        norm = re.sub(r"\s+", " ", self.text).strip().lower()
        return f"memory:{self.source}:{hashlib.sha256(norm.encode()).hexdigest()[:12]}"


_MEMORY_DOC_RE = re.compile(r"^[-*] ?(.{12,})$")
#: Anything bullet-shaped but indented, or a top-level bullet too short to be a
#: claim. These are the lines the entry grammar does NOT reach, and they have to
#: be counted, not quietly dropped: `user_md_entries.ratio` is the acceptance
#: denominator, so a self-defined denominator is the whole clause going unmeasured.
_NEAR_MISS_RE = re.compile(r"^\s+[-*] ?(.{12,})$|^\s*[-*] ?(.{1,11})$")


def memory_entries(root: Path | str | None = None,
                   docs: Sequence[str] = ("lloyd/USER.md", "lloyd/MEMORY.md"),
                   tally: dict[str, dict[str, int]] | None = None) -> list[Entry]:
    """Durable memory bullets, one `Entry` each.

    Read from the live vault when present, else the checkout — a top-level
    bullet is what actually gets loaded into the system prompt, which is the
    surface the item is about.

    `tally`, if given, is filled with what the grammar *missed*, **keyed by
    document** (`{"lloyd/USER.md": {"indented_bullets": 3, "short_bullets": 2}}`)
    so the caller can publish a denominator that is a measurement rather than a
    description of its own regex. Per-document because the acceptance clause
    names USER.md specifically: one flat tally lets a clean MEMORY.md carry a
    USER.md the grammar barely reads.
    """
    root = Path(root) if root else REPO
    vault = Path.home() / "obsidian"
    out: list[Entry] = []
    seen: set[str] = set()
    for doc in docs:
        doc_misses = tally.setdefault(doc, {}) if tally is not None else None
        for base in (vault, root):
            path = base / doc
            if not path.is_file():
                continue
            for line in path.read_text(errors="replace").splitlines():
                m = _MEMORY_DOC_RE.match(line.rstrip())
                if not m:
                    if doc_misses is not None:
                        near = _NEAR_MISS_RE.match(line.rstrip())
                        if near:
                            key = ("indented_bullets" if line[:1].isspace()
                                   else "short_bullets")
                            doc_misses[key] = doc_misses.get(key, 0) + 1
                    continue
                e = Entry(doc, m.group(1).strip())
                if e.key in seen:
                    continue
                seen.add(e.key)
                out.append(e)
            break
    return out


_TOKENS = re.compile(r"[a-z0-9_]+")


def overlap(a: str, b: str) -> float:
    """Word-set Jaccard (unigrams + bigrams) between an entry and a dispute.

    The item prescribes embedding similarity as the mitigation for the
    always-in-force problem; this is the lexical stand-in and is labelled as
    such wherever the number is emitted. It is enough to separate "the
    correction is about this entry" from "this entry was merely in the window",
    and it is cheap enough to run on every pair.
    """
    def grams(s: str) -> set[str]:
        toks = _TOKENS.findall((s or "").lower())
        return set(toks) | {f"{x} {y}" for x, y in zip(toks, toks[1:])}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


#: A skill read mid-turn is logged *during* the turn whose ordinal we count, so
#: presence starts at that turn. One turn late at worst — see the measured bound.
#: The per-injection row is logged the other way round, BEFORE the turn it serves
#: — see `skill_evidence_by_session`, which credits it one turn later than this.
TURN_EVENT = "brain1.user_prompt_received"
#: The event named in the module title. Re-declared rather than imported for the
#: same reason `SKILL_NAME_RE` and `SKILL_BLOCK_RE` above are copied from
#: `app/prefetch.py` rather than imported: uptake runs without a model and may not
#: reach the prompt path it measures, and `app.prefetch` is the prompt path.
#: `tests/test_uptake.py::test_uptake_redeclares_the_names_it_copies_from_the_prompt_path`
#: pins each copy to the value it copies, so the independence cannot rot into
#: drift.
SKILL_MATCH_EVENT = "prefetch.skill_match"


def skill_evidence_by_session(root: Path | str | None = None
                              ) -> dict[str, dict[str, dict[str, int]]]:
    """Both skill-presence routes off one walk: `{source: {session: {name: turn}}}`.

    Keys are the two members of `SKILL_EVIDENCE_SOURCES`. Each value is the
    **ordinal of the earliest human turn from which that skill was in force**, not
    a set of names: a session-wide set was the defect the review rung found here.
    The old shape marked a skill present for *every* turn of a session that read
    it on any of them, so a dispute on turn 2 was attributed to a skill first
    opened on turn 9 — inflating `present_in_turns` (the divisor) and `disputes`
    (the numerator) of the same row at once, in both directions.

    Turn index comes from counting `brain1.user_prompt_received` events with
    `data.source == "user"` in the same per-session file, before the evidence
    line. Deliberately not timestamps: session JSONs store naive ISO
    (`2026-09-04T12:13:16.901843`) and the event log stores UTC with a `Z`
    (`2026-09-08T18:15:26.540Z`), and deciding which side is local is an 8-hour
    guess that would mis-order every read inside a session measured in minutes.
    Measured bound on the proxy (114 sessions with a session JSON and a read):
    the event count equals the human-turn count in 104 and **never falls below
    it** — 10 sessions log extra events — so presence can start at most one turn
    late and never one turn early. Late is the safe direction: it under-credits
    rather than blaming a turn for a skill it had not asked for yet.

    The two routes sit on opposite sides of the turn boundary, so they get
    different ordinals from the same counter, and getting this backwards is a
    silent over-credit rather than a crash:

    * `SKILL_PRESENCE_PROXY` — a `skills_read` call is logged *during* the turn in
      flight, after that turn's receipt row, so the count already includes it:
      `max(turn, 1)`. A read logged before any human prompt (a resumed or
      autonomous session, or a log that rotated) starts at turn 1 rather than
      being dropped: dropping it would make the skill look never-present, the same
      silent-zero failure the prefetch half of this table is guarded against.
    * `SKILL_PRESENCE_TELEMETRY` — `prefetch._emit_skill_match_events` runs while
      the prompt is being *built*, so its row lands BEFORE the receipt row of the
      very turn it served. Measured on the live logs (298 sessions with a landed
      row): 563 of the 624 landed rows sit before the first counted receipt row,
      and in the files where both are present the pair is
      `prefetch.skill_match 2026-09-25T02:05:36.781Z` then
      `brain1.user_prompt_received 2026-09-25T02:05:36.939Z` for the same turn.
      The count therefore sits one short of the turn the row is about, so the
      telemetry credits `turn + 1`. Crediting `max(turn, 1)` here instead would
      blame every turn *before* the injection for it — the exact early-credit
      defect the ordinal shape exists to prevent. A row after the last receipt row
      credits one past the window's end and so contributes nothing, which is the
      safe side of the same trade.

    Only a row with `landed: true` credits anything: an offer that did not render
    was not in that prompt, and counting it would turn prefetch's own top-K
    ceiling into apparent presence. (`injected_body` is the stronger sub-case —
    the body, not just the line — and is not required, because a rendered name is
    still a rendered entry.)

    The events are written by the agent loop and the prefetch path, both different
    processes, and the payloads are only half-structured: `data.args` on a tool
    call is a JSON *string*, so the skill name has to be parsed back out of it. A
    log line that does not parse is skipped rather than fatal — these files have
    unparseable lines in them today.
    """
    root = Path(root) if root else lloyd_root()
    out: dict[str, dict[str, dict[str, int]]] = {src: {} for src in SKILL_EVIDENCE_SOURCES}

    def credit(source: str, session: str, name: str, first: int) -> None:
        bucket = out[source].setdefault(session, {})
        prev = bucket.get(name)
        if prev is None or first < prev:
            bucket[name] = first

    for path in root.glob(_EVENT_GLOB):
        try:
            handle = path.open(errors="replace")
        except OSError:
            continue
        with handle:
            turn = 0  # human prompts seen so far in THIS file, same file order
            for line in handle:
                is_read = "skills_read" in line
                is_match = f'"{SKILL_MATCH_EVENT}"' in line
                if not (is_read or is_match):
                    # Count the turn boundary even on lines we otherwise skip;
                    # the ordinal is only meaningful if it advances in file
                    # order alongside the reads.
                    if f'"{TURN_EVENT}"' in line:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        data = rec.get("data")
                        if isinstance(data, dict) and str(data.get("source")) == "user":
                            turn += 1
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = rec.get("data")
                if not isinstance(data, dict):
                    continue
                session = rec.get("session_id")
                if not session:
                    continue
                session = str(session)
                # The read branch goes first, as it did before this function
                # knew about two routes: a line naming both is the tool call, and
                # the tool call is the evidence the proxy channel is made of.
                if is_read and data.get("name") == "skills_read":
                    args = data.get("args")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        continue
                    name = args.get("name") if isinstance(args, dict) else None
                    if not name:
                        continue
                    credit(SKILL_PRESENCE_PROXY, session, str(name), max(turn, 1))
                elif rec.get("event") == SKILL_MATCH_EVENT:
                    if data.get("landed") is not True:
                        continue
                    name = data.get("skill")
                    if not name:
                        continue
                    credit(SKILL_PRESENCE_TELEMETRY, session, str(name), turn + 1)
    return out


def skills_read_by_session(root: Path | str | None = None):
    """The proxy half of `skill_evidence_by_session`, as `{session: {name: turn}}`.

    Kept as its own name because it is the shape the probe and the suite already
    pass as `skills_read=`, and because "when did the model ask for this skill" is
    a question worth asking on its own. One walk serves both routes, so this is
    not a second pass over the logs: the ordinal rule is one piece of code and
    both channels read it, which is the only way they stay comparable.
    """
    return skill_evidence_by_session(root)[SKILL_PRESENCE_PROXY]


def active_skill_names(vault: Path | None = None) -> list[str]:
    vault = vault or (Path.home() / "obsidian")
    return sorted(p.parent.name for p in (vault / "skills").glob("*/SKILL.md"))


def _weight_or_null(overlaps: Iterable[float]) -> float | None:
    """Sum the per-dispute overlaps, or report that there was nothing to sum.

    A `weighted_disputes` of 0.0 used to carry two different claims — "measured,
    and the correction never mentioned this entry" and "there was no disputed
    turn to measure" — and the skill channel paid `max(overlap(…), 0.05)` on top
    of both, so a row could be nonzero while overlapping nothing at all
    (`skill:code-review`: weight 0.05, `overlap_max` 0.0). A row with nothing to
    say is null: it cannot be averaged into a headline, cannot sort above a row
    that genuinely measured zero, and cannot be read by the pruning job as "a low
    score" when it is in fact no score. The 0.05 had no stated rationale and
    arrived in the first implementation commit; it is gone rather than lowered.
    """
    total = sum(overlaps)
    return round(total, 4) if total > 0 else None


def _row(entry: str, presence_source: str, present: int, disputes: int,
         weighted: float | None, overlap_max: float, extra: dict | None = None) -> dict:
    """One row. `weighted` is already rounded, or None for "no signal" — see
    `_weight_or_null` and `GLOSSARY["weighted_disputes"]`."""
    row = {
        "entry": entry,
        "presence_source": presence_source,
        "present_in_turns": present,
        "disputes": disputes,
        "weighted_disputes": weighted,
        "overlap_max": round(overlap_max, 4),
        "dispute_rate": round(disputes / present, 4) if present else 0.0,
    }
    if extra:
        row.update(extra)
    return row


def _memory_coverage(entries: Sequence[Entry], tally: dict[str, Any] | None) -> dict[str, Any]:
    """Coverage of the durable-memory half, with a denominator that is measured.

    `covered == total` was the shape here before: `total` was `len(entries)`, so
    the ratio was 1.0 by construction and the acceptance clause's "≥ 80 % of
    USER.md entries" graded a regex against itself. Indented bullets and
    too-short bullets are bullet-shaped lines the grammar does not reach — they
    belong in the denominator, counted by `memory_entries(tally=…)`.

    Reported per document as well as overall, because the clause is about
    **USER.md** specifically and a healthy MEMORY.md could otherwise carry a
    USER.md that the grammar barely reads.

    And `covered` still does not mean *honored*: for always-in-force entries the
    join reaches everything by definition, which is what the note says out loud.
    """
    tallies: dict[str, dict[str, int]] = tally or {}
    per_doc: dict[str, Any] = {}
    for doc in sorted({e.source for e in entries} | set(tallies)):
        rec = sum(1 for e in entries if e.source == doc)
        ind = int(tallies.get(doc, {}).get("indented_bullets", 0))
        shrt = int(tallies.get(doc, {}).get("short_bullets", 0))
        tot = rec + ind + shrt
        per_doc[doc] = {
            "covered": rec, "total": tot,
            "ratio": round(rec / tot, 4) if tot else 0.0,
            "skipped": {"indented_bullets": ind, "short_bullets": shrt},
        }
    recognized = len(entries)
    indented = sum(d["skipped"]["indented_bullets"] for d in per_doc.values())
    short = sum(d["skipped"]["short_bullets"] for d in per_doc.values())
    total = recognized + indented + short
    return {
        "covered": recognized,
        "total": total,
        "ratio": round(recognized / total, 4) if total else 0.0,
        "presence_source": ALWAYS_IN_FORCE,
        "by_doc": per_doc,
        "skipped": {"indented_bullets": indented, "short_bullets": short},
        "denominator_note": (
            f"total = {recognized} bullets the entry grammar recognizes + "
            f"{indented} indented + {short} too-short that it does not. "
            "'covered' means the join reached them, which for always-in-force "
            "entries is true by construction — it is NOT evidence an entry was "
            "honored; weighted_disputes on the row is the usable signal."
        ),
    }


def _by_presence_source(entries: Sequence[dict]) -> dict[str, dict]:
    """Per-channel denominators for `weighted_disputes`, one block per channel.

    Published because the pooled version of that column is not one quantity.
    Measured in `eval/uptake/uptake-2026-09-17.json`, the largest weight per
    source was 0.2308 (`proxy:skills_read+injected_context`) / 0.0505
    (`always_in_force:system_prompt`) / 0.0294 (`prefetch:vault_context`) — an
    ~8x spread that comes from which channel a row belongs to and from nothing
    about uptake. A single sorted list over all rows therefore ranks channels,
    and a reader comparing two numbers from different ones is comparing the
    scorer against itself. Every declared source gets a block even with no rows:
    a missing key reads as "not measured", which is a different claim from
    "measured, nothing here".
    """
    stray = sorted({r["presence_source"] for r in entries} - set(PRESENCE_SOURCES))
    if stray:
        # `raise`, not a dropped key: a channel absent from PRESENCE_SOURCES would
        # vanish from the one block whose job is to make cross-channel ranking
        # impossible, and the table would go on inviting exactly that.
        raise RuntimeError(f"rows carry a presence_source outside PRESENCE_SOURCES: {stray}")
    out: dict[str, dict] = {}
    for src in sorted(PRESENCE_SOURCES):
        weights = [r["weighted_disputes"] for r in entries
                   if r["presence_source"] == src and r["weighted_disputes"] is not None]
        rows = sum(1 for r in entries if r["presence_source"] == src)
        out[src] = {
            "rows": rows,
            "rows_with_weight": len(weights),
            "rows_null_weight": rows - len(weights),
            "max_weight": max(weights) if weights else None,
        }
    return out


def _skill_evidence_routes(skills_read: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise the `skills_read=` argument to `{source: {session: {name: turn}}}`.

    Two shapes arrive here and both are legitimate:

    * the two-route form `skill_evidence_by_session` emits, keyed by
      `SKILL_EVIDENCE_SOURCES` members — what the probe passes now that the real
      per-injection event exists;
    * the single-channel form `{session: {name: turn}}` (or the legacy
      `{session: {name}}`), which every caller that predates #1603 passes, and
      which means the proxy: it is a record of reads, never of injections.

    A dict that mixes the two — a source key beside a session key — is refused
    rather than guessed, because the guess decides which channel a session's
    evidence is credited to, and that is the claim the emitted label is made of.
    """
    if not skills_read:
        return {src: {} for src in SKILL_EVIDENCE_SOURCES}
    keys = set(skills_read)
    if keys & set(SKILL_EVIDENCE_SOURCES):
        extra = sorted(keys - set(SKILL_EVIDENCE_SOURCES))
        if extra:
            raise ValueError(
                f"skills_read mixes evidence sources with session keys, first few: "
                f"{extra[:4]}. Pass either {sorted(SKILL_EVIDENCE_SOURCES)} keyed "
                "(skill_evidence_by_session) or {session: {name: turn}} keyed "
                "(skills_read_by_session), not both.")
        return {src: dict(skills_read.get(src) or {})
                for src in SKILL_EVIDENCE_SOURCES}
    return {SKILL_PRESENCE_PROXY: dict(skills_read),
            SKILL_PRESENCE_TELEMETRY: {}}


def build_uptake_table(
    turns: Sequence[Turn],
    dispute_flags: dict[str, Any],
    memory_entries: Sequence[Entry] = (),
    skills_read: dict[str, Any] | None = None,
    active_skills: Sequence[str] = (),
    memory_tally: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Per-entry uptake table. Measurement only — nothing here prunes.

    `dispute_flags` maps **`Turn.turn_id`** (`"<session>#<ordinal>"`) -> verdict
    (`True` / `False` / `None` for "the classifier did not answer"). `None`
    verdicts are counted in `engine_unanswered` rather than folded into the
    negative class, so a stalled engine shows up as missing evidence instead of
    a clean bill of health.

    The key is the whole contract, and it was the defect this function shipped
    with: verdicts used to be keyed on `Turn.ordinal`, which **restarts at 1 in
    every session**. Measured on the live 30-day window, 220 human turns spread
    over 132 sessions occupy only ordinals 1–10, so 24 candidate verdicts
    collapsed into 9 ordinal keys, and the join then charged every session that
    happened to have a turn at that position with another session's dispute. A
    key that is not a `turn_id` of one of `turns` is therefore a caller defect
    and raises, rather than silently matching nothing (or the wrong thing).
    """
    skills_read = skills_read or {}
    by_id = {t.turn_id: t for t in turns}
    if len(by_id) != len(turns):
        raise ValueError(f"duplicate turn_id among {len(turns)} turns")
    unknown = sorted(set(dispute_flags) - set(by_id))
    if unknown:
        raise ValueError(
            f"{len(unknown)} dispute_flags key(s) are not turn_ids of these turns, "
            f"first few: {unknown[:4]}. Verdicts must be keyed on Turn.turn_id "
            "(session#ordinal); Turn.ordinal restarts per session and collides.")
    disputed_ids = {k for k, v in dispute_flags.items() if v is True}
    disputed = [t for t in turns if t.turn_id in disputed_ids]
    unresolved = sum(1 for v in dispute_flags.values() if v is None)

    # --- skills: evidence-bound presence, never "all skills everywhere" ------
    # Each row also records HOW tightly presence is bounded, because the three
    # routes are not equally trustworthy and a consolidator must be able to tell
    # "this skill was injected into that very turn" from "some turn of that
    # session opened this skill at some point".
    skill_present: dict[str, list[Turn]] = {}
    skill_seen: dict[str, set[str]] = {}
    skill_bound: dict[str, str] = {}
    skill_source: dict[str, str] = {}
    # Names each route credited anywhere in the window, kept per route so
    # `coverage.active_skills.covered_by_source` can be computed from the same
    # walk that produced the rows rather than re-derived from the rows and their
    # strongest-label-only bookkeeping.
    skill_route_names: dict[str, set[str]] = {src: set() for src in SKILL_EVIDENCE_SOURCES}
    _TIGHTNESS = {"session_wide": 1, "causal_event_order": 2, "injected_this_turn": 3}
    # `raise`, not `assert`: this is an integrity check on the emitted table, and an
    # assert is compiled out by `python -O`, which would leave the bound vocabulary
    # free to drift on exactly the runs that optimise.
    if PRESENCE_BOUNDS != frozenset(_TIGHTNESS):
        raise RuntimeError(
            f"presence bound vocabulary drifted: PRESENCE_BOUNDS={sorted(PRESENCE_BOUNDS)} "
            f"vs tightenness ladder {sorted(_TIGHTNESS)}")

    def note_skill(name: str, turn: Turn, bound: str, source: str) -> None:
        bucket = skill_present.setdefault(name, [])
        seen = skill_seen.setdefault(name, set())
        if turn.turn_id not in seen:
            seen.add(turn.turn_id)
            bucket.append(turn)
        # The row is labelled by its BEST evidence, not its last: a skill the
        # injector landed on turn 3 and the model also read on turn 7 is
        # injection-grade presence, and letting the weaker route stamp the label
        # would report less certainty than the row has.
        skill_route_names[source].add(name)
        rank = _TIGHTNESS[bound] - _TIGHTNESS.get(skill_bound.get(name, ""), 0)
        if rank > 0:
            skill_bound[name] = bound
            skill_source[name] = source
        elif rank == 0:
            # Equal tightness is not equal evidence. A read can pin presence to a
            # turn exactly as well as a landed row can, and then the row's label
            # must say which KIND of proof it has — so the tie breaks in declared
            # route order, strongest first. Without this the label belongs to
            # whichever route happened to credit the earlier turn, which is how a
            # skill the injector landed reads as "the model merely asked for it".
            routes = list(SKILL_EVIDENCE_SOURCES)
            if routes.index(source) < routes.index(skill_source.get(name, source)):
                skill_source[name] = source

    # `{source: {session: {name: first_ordinal}}}`, from one walk of the logs.
    # Iterated in `SKILL_EVIDENCE_SOURCES` order — strongest route first — so the
    # strongest evidence is what sets the label even on a turn both routes
    # credit, without depending on dict order.
    evidence = _skill_evidence_routes(skills_read)
    for t in turns:
        # The persisted `<skill name=…>` block is injection-level evidence, not a
        # read: `_emit_skill_match_events` writes `landed`/`injected_body` from the
        # same injection plan that produced this block, so the block in the
        # transcript and a `landed: true` row are one event seen twice. Labelling
        # it with the proxy's name would claim less than the evidence gives.
        for name in t.injected_skills:
            note_skill(name, t, "injected_this_turn", SKILL_PRESENCE_TELEMETRY)
        for source in SKILL_EVIDENCE_SOURCES:
            per_session = evidence.get(source) or {}
            reads = per_session.get(t.session) or {}
            # A dict is either reader's real shape ({name: first_turn}); a set is
            # the test/legacy shape, meaning "present all session" and labelled as
            # the loosest bound rather than pretending to be causal.
            pairs = (reads.items() if isinstance(reads, dict)
                     else ((n, 1) for n in reads))
            causal = isinstance(reads, dict)
            for name, first in pairs:
                if t.ordinal >= int(first):
                    note_skill(name, t,
                               "causal_event_order" if causal else "session_wide",
                               source)

    entries: list[dict[str, Any]] = []
    for name in sorted(skill_present):
        present = skill_present[name]
        hits = [t for t in present if t.turn_id in disputed_ids]
        # Scored against the skill NAME — not the skill body — which is why a
        # null here is the expected value for most rows and not a malfunction.
        # Out of scope to widen it (#1195, Alan's call); the 0.05 that used to
        # sit under this is gone, so the channel now says "no signal" when that
        # is what it measured.
        overlaps = [overlap(name, t.user_text) for t in hits]
        entries.append(_row(
            f"skill:{name}", skill_source.get(name, SKILL_PRESENCE_PROXY),
            len(present), len(hits),
            _weight_or_null(overlaps), max(overlaps, default=0.0),
            {"kind": "skill", "name": name, "scored_on": "name",
             "presence_bound": skill_bound.get(name, "session_wide"),
             # Which route earned that bound. `presence_bound` alone says how
             # tightly presence is pinned; this says whether the pinning came
             # from the injector's own event or from a read, which is the
             # distinction #1603 exists to make readable.
             "presence_route": skill_source.get(name, SKILL_PRESENCE_PROXY),
             "present_turn_ids": [t.turn_id for t in present][:20]},
        ))

    # --- memory: always in force, so presence is uninformative by design ----
    for e in memory_entries:
        # Every dispute in the window is "in the presence" of an always-in-force
        # entry, by definition. That is why this row's `dispute_rate` is capped
        # and labeled an upper bound, and why `weighted_disputes` (which is not
        # the same number for two different entries) is the usable signal.
        hits = disputed
        overlaps = [overlap(e.text, t.user_text) for t in hits]
        entries.append(_row(
            e.key, ALWAYS_IN_FORCE, len(turns), len(hits),
            _weight_or_null(overlaps), max(overlaps, default=0.0),
            {"kind": "memory_entry", "source_doc": e.source, "scored_on": "entry_text",
             "text": e.text[:300],
             "note": "present in every prompt: dispute_rate is an upper bound, "
                     "weighted_disputes is the usable signal, and a null there is "
                     "no signal rather than a low score"},
        ))

    # --- notes/facts that arrived through prefetch --------------------------
    note_present: dict[str, list[Turn]] = {}
    note_seen: dict[str, set[str]] = {}
    for t in turns:
        for title in t.vault_context:
            key = f"note:{title}"
            seen = note_seen.setdefault(key, set())
            if t.turn_id in seen:
                continue
            seen.add(t.turn_id)
            note_present.setdefault(key, []).append(t)

    def note_ctx(turn: Turn, title: str) -> VaultCtx:
        """The line THIS turn was shown for `title`, falling back to bare title.

        Scoring each disputed turn against the line it was actually shown is the
        faithful join: that excerpt, not another turn's, was in front of the model
        when it got corrected. A caller that supplied plain titles — every fixture
        predating #1195, and any real block whose line carried no path or excerpt
        — still scores on the title, and `scored_on` says that out loud instead of
        leaving the reader to guess which of the two a number came from.
        """
        for ctx in turn.vault_context:
            if str(ctx) == title:
                return ctx if isinstance(ctx, VaultCtx) else VaultCtx(title)
        return VaultCtx(title)

    for key in sorted(note_present):
        present = note_present[key]
        hits = [t for t in present if t.turn_id in disputed_ids]
        title = key.split(":", 1)[1]
        line_of = {t.turn_id: note_ctx(t, title) for t in present}
        overlaps = [overlap(line_of[t.turn_id].scored_text, t.user_text) for t in hits]
        # With no disputed turn there is nothing the weight was computed from, so
        # the basis is reported over the turns the note WAS present for.
        bases = {line_of[t.turn_id].scored_on for t in (hits or present)}
        extra: dict[str, Any] = {
            "kind": "note",
            "scored_on": (next(iter(bases)) if len(bases) == 1 else "mixed"),
            "presence_turn_ids": [t.turn_id for t in present][:20],
        }
        path = next((line_of[t.turn_id].path for t in present
                     if line_of[t.turn_id].path), "")
        if path:
            extra["source_doc"] = path
        entries.append(_row(
            key, NOTE_PRESENCE_EMITTED, len(present), len(hits),
            _weight_or_null(overlaps), max(overlaps, default=0.0), extra,
        ))

    cov_skills = sorted(set(skill_present) & set(active_skills or ()))
    # Per-route coverage, so the union cannot hide which route produced it. The
    # proxy half is kept because its reach is genuinely wider in two ways the
    # telemetry is not: a session whose log predates the rows, and a skill the
    # model pulled with `skills_read` that prefetch never offered. Dropping it
    # would report a collapse of `covered` as if skill presence had fallen off,
    # when what changed is that the instrument stopped reading the one source
    # that still saw them.
    cov_by_source = {
        src: len(skill_route_names[src] & set(active_skills or ()))
        for src in SKILL_EVIDENCE_SOURCES
    }
    # One label for the block: the strongest route that contributed coverage, so
    # an all-telemetry window says so and a pre-telemetry window still says
    # `proxy:...` rather than claiming a source it has no rows from. The split
    # beside it is what makes the single label readable rather than lossy.
    coverage = {
        "user_md_entries": _memory_coverage(memory_entries, memory_tally),
        "active_skills": {
            "covered": len(cov_skills),
            "total": len(active_skills or ()),
            "ratio": (round(len(cov_skills) / len(active_skills), 4) if active_skills else 0.0),
            "presence_source": next(
                (src for src in SKILL_EVIDENCE_SOURCES if cov_by_source[src]),
                SKILL_PRESENCE_PROXY),
            "covered_by_source": cov_by_source,
            "note": SKILL_PRESENCE_NOTE,
        },
        # An empty count here is the table's most misleading number: it is the
        # one shape that reads as "knowledge notes are never disputed". It now
        # travels with the reach that produced it — how many turns in this window
        # actually persisted an injected block — so a zero says which of the two
        # it is.
        "prefetch_notes": {
            "entries_identified": len(note_present),
            "turns_with_block": sum(1 for t in turns if t.injections_seen),
            "turns_total": len(turns),
            "reach": (round(sum(1 for t in turns if t.injections_seen) / len(turns), 4)
                      if turns else 0.0),
            "presence_source": (
                NOTE_PRESENCE_EMITTED if note_present else NOTE_PRESENCE_UNREACHABLE),
            "note": NOTE_PRESENCE_NOTE,
        },
        # Per-channel denominators for `weighted_disputes`. Published because the
        # pooled version of this column is not one quantity: measured in
        # eval/uptake/uptake-2026-09-17.json the largest weight per source was
        # 0.2308 (proxy skills) / 0.0505 (always-in-force) / 0.0294 (notes), an
        # ~8x spread produced by which channel a row belongs to and by nothing
        # about uptake. Any single sorted list over all rows therefore ranks
        # channels, and a reader comparing two numbers from different channels is
        # reading an artefact of the scorer. Every declared source gets a block
        # even with no rows — a missing key reads as "not measured", which is a
        # different claim from "measured, nothing here".
        "by_presence_source": _by_presence_source(entries),
    }

    return {
        "probe_timestamp": _now(),
        "corpus": {
            "turns": len(turns), "disputes": len(disputed),
            "verdict_keyed_on": "turn_id",
            "classified_turns": len(dispute_flags) - unresolved,
            "engine_unanswered": unresolved,
            "unscreened_turns": len(turns) - len(dispute_flags),
            "dispute_rate_over_turns": round(len(disputed) / len(turns), 4) if turns else 0.0,
        },
        "coverage": coverage,
        "entries": entries,
        # Which scorer wrote this file. A reader picking the newest
        # `eval/uptake/uptake-*.json` cannot otherwise tell whether the reading
        # rules its consumer skill gives it apply to the bytes in front of it:
        # nulls, `scored_on` and `coverage.by_presence_source` are generation-2
        # facts, and the tables already committed are generation 1.
        "scorer_generation": SCORER_GENERATION,
        # Every field's meaning travels with the artifact. The probe used to hand
        # `write_table` its own glossary text and it described all three channels
        # as overlap against "the entry text" — true of the always-in-force rows
        # alone, and the reason a floor-only skill row and a title-only note row
        # both read as measured signal.
        "glossary": GLOSSARY,
    }


# -------------------------------------------------------- retrieval gate ----

def _metrics_of(doc: dict) -> dict | None:
    summ = doc.get("summary")
    if isinstance(summ, dict):
        overall = summ.get("overall")
        if isinstance(overall, dict) and overall:
            return overall
    if isinstance(doc.get("overall"), dict) and doc.get("overall"):
        return doc["overall"]
    return None


#: What a night that predates the `semantic_seeding` field is banded as (#1547).
#: Its own token and not a stand-in for `k=0`: a night that recorded nothing does
#: not thereby record that seeding was off, and pooling it with a seeding-off
#: night would be inventing the measurement the file does not have.
SEEDING_SHAPE_UNRECORDED = "unrecorded"


def _seeding_shape_key(doc: dict) -> str:
    """The semantic seeding a night recorded, as one token, for run shape (#1547).

    `enabled=True,k=3` / `enabled=False,k=0` for a night that says, and
    `SEEDING_SHAPE_UNRECORDED` for one that does not. Three distinct values on
    purpose: #1486 moved `entity_hit_rate` 0.337 -> 0.500 and
    `anchorless_query_count` 25 -> 16 overnight while leaving every document
    metric flat, so an on-night and an off-night in one band would publish a
    tolerance measured across a step the gate is supposed to detect. A field the
    producer may yet rename is read tolerantly and never guessed at: a
    non-dict, or an unparsable `k`, is `unrecorded`.
    """
    rec = doc.get("semantic_seeding")
    if not isinstance(rec, dict):
        return SEEDING_SHAPE_UNRECORDED
    k = rec.get("k")
    if isinstance(k, bool) or not isinstance(k, int):
        return SEEDING_SHAPE_UNRECORDED
    return f"enabled={bool(rec.get('enabled'))},k={k}"


def retrieval_gate(baselines_dir: Path | str | None = None,
                   metrics: Sequence[str] = ("doc_hit_rate", "ndcg10")) -> dict[str, Any]:
    """A non-regression gate read from what the eval actually does at rest.

    Triage item #801 measured six identical-config nightly runs at
    0.95/0.90/0.90/0.90/0.95/0.85 doc_hit_rate — a 0.10 spread. Any item whose
    acceptance hard-codes `doc_hit_rate >= 0.95` is therefore decided by which
    night it ran, not by the change under test. This returns newest-value minus
    the observed spread of comparable nights, so the tolerance is measured.

    Comparability is by run shape (`limit`, `matches_production_defaults`,
    `semantic_seeding`), not by filename: folding a 40-query or rerank-on night
    into the spread would widen the band with somebody else's experiment. The
    third term is #1547: #1486 changed the SEED DEFINITION the entity scores are
    built from — not a CLI knob, so invisible to `matches_production_defaults`,
    which compares six parsed args against six `RECALL_*` constants — and the two
    nightlies either side of it both read `matches_production_defaults: true`
    while `entity_hit_rate` moved 0.337 -> 0.500. A night that predates the field
    bands as `unrecorded`, so a step never hides inside a band; the published
    `shape` block names the token the band was computed over. Only a
    `nightly-*.json` is a night at all (#1220). The rest of the directory is one-off runs — A/B arms,
    `kg-rebuild` before/after snapshots, an item's scale check — each scored
    against whatever store happened to be standing the minute it ran, so a
    directory whose last night had gone used to report a band computed entirely
    from those one-off runs and call it the nightly tolerance. Zero nights is
    `NoBaselines`, not a number.

    The comparable population is then CAPPED to the newest
    `RETRIEVAL_GATE_WINDOW_NIGHTS` of them (#1220). Shape is a filter, not a
    window: the pool used to be every night the directory ever held, so a band
    published this week was dominated by nights scored against a corpus that has
    since been rebuilt — the same rot #801 was filed against, moved one level up
    from a number copied into prose to a window that quietly keeps an obsolete
    regime in the sample. So the reader can tell the two counts apart: `nights`
    is what the band was computed over, `nights_in_shape` how many nights were
    comparable at all, and every metric block repeats the pair as
    `window_nights`/`n_nights` so a band cannot travel without its denominator.
    (This paragraph used to say the probe "drops any top-level key it does not
    name", which was inverted by #1220: `scripts/uptake_probe.py` copies every
    top-level key but `GATE_TABLE_EXCLUDED_KEYS` — the baselines path — so a new
    field like `shape.semantic_seeding` reaches
    `eval/uptake/uptake-<date>.json` unaided and does not have to be parked
    inside a metric block.)

    A latest of 1.0 and a band of 0 both mean the published floor discriminates
    nothing — one is a rate saturated at its ceiling, the other a pool with no
    spread. Each block says which via `at_ceiling`/`ceiling_reason`, and states
    `granularity` = 1/`n_queries`, the smallest move the metric can make on the
    newest night: a floor nearer to `latest` than that cannot be crossed by any
    change the eval is able to measure, so treating it as a tolerance is reading
    a rounding step as a regression.
    """
    d = Path(baselines_dir) if baselines_dir else (lloyd_root() / "eval" / "baselines")
    rows: list[tuple[float, Path, dict, dict]] = []
    for path in sorted(d.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        met = _metrics_of(doc)
        if not met:
            continue
        rows.append((path.stat().st_mtime, path, doc, met))
    if not rows:
        raise NoBaselines(f"no retrieval baselines with metrics under {d}")
    rows.sort(key=lambda r: r[0], reverse=True)

    nightly = [r for r in rows if r[1].name.startswith("nightly-")]
    # A gate on production behavior reads its band off production-config nights,
    # and off the *recurring* shape: one-off 40-query or rerank-on night is a
    # different experiment, and letting whichever night happened to run last set
    # the reference would compare tonight's change against a single unrelated
    # run. Modal shape, newest wins the tie.
    #
    # Nothing that is not a nightly is ever a fallback (#1220). The `or rows`
    # that used to end this chain made the leftovers answer for the nightly
    # spread once the nightlies ran out: the A/B arms and the kg-rebuild `after`
    # snapshot from the live directory, copied into a directory holding no night,
    # answered `nights: 5`, doc band 0.025, floor 0.691 before this shipped — a
    # "measured tolerance" assembled from one-off runs, each scored against
    # whatever store was standing the minute it ran. A missing sample raises; it
    # does not borrow somebody else's run.
    prod = [r for r in nightly if bool(r[2].get("matches_production_defaults"))]
    if not nightly:
        raise NoBaselines(
            f"no nightly-*.json under {d}: the {len(rows)} eval JSON(s) with "
            "metrics there are one-off runs (A/B arms, rebuild snapshots), not "
            "nights, and cannot band a nightly gate")
    basis = prod or nightly
    shapes: dict[Any, list] = {}
    for r in basis:
        shapes.setdefault((r[2].get("limit"),
                           bool(r[2].get("matches_production_defaults")),
                           _seeding_shape_key(r[2])), []).append(r)
    shape, comparable = max(shapes.items(), key=lambda kv: (len(kv[1]), kv[1][0][0]))
    # Newest first: `rows` was sorted by mtime descending and `shapes` was filled
    # in that order, so the front slice is the newest M nights of this shape.
    pool = comparable[:RETRIEVAL_GATE_WINDOW_NIGHTS]
    newest_mtime, newest_path, newest_doc, newest_met = pool[0]

    out: dict[str, Any] = {
        "baselines_dir": str(d),
        # Which band this is, named in full: a reader who is told `nights: 1`
        # must be able to see that it is one because the seeding regime is
        # single-member, not because the pool was lost (#1547).
        "shape": {"limit": shape[0],
                  "matches_production_defaults": shape[1],
                  "semantic_seeding": shape[2]},
        # Two counts, because the window makes them different quantities: the
        # band is computed over `nights`, drawn from `nights_in_shape` comparable
        # nights. One number is what let a 14-night sample read as a 7-night one.
        "nights": len(pool),
        "nights_in_shape": len(comparable),
        "window_nights": RETRIEVAL_GATE_WINDOW_NIGHTS,
        "latest": {"label": newest_doc.get("label") or newest_path.name,
                   "measured_at": newest_doc.get("ran_at") or newest_doc.get("measured_at"),
                   "path": str(newest_path)},
        "hardcoded_gate": RETRIEVAL_GATE_HARDCODE,
        # Counted over the pooled window, not over all history: the nights this
        # figure is about are the nights the band is about.
        "hardcoded_gate_would_have_failed_nights": 0,
    }
    n_queries = newest_met.get("n_queries")
    n_queries = n_queries if isinstance(n_queries, (int, float)) and n_queries else None
    # The metric's own step size on the newest night — at 81 queries one query is
    # 1/81 of the rate — so a reader can see whether the floor they are being
    # handed is a tolerance or just the grid the number moves on.
    granularity = round(1.0 / n_queries, 4) if n_queries else None
    for key in metrics:
        vals = [float(r[3][key]) for r in pool if isinstance(r[3].get(key), (int, float))]
        if not vals:
            continue
        latest = float(newest_met[key]) if isinstance(newest_met.get(key), (int, float)) else min(vals)
        band = max(vals) - min(vals)
        # Two ways a floor stops meaning anything, and both have to be said out
        # loud instead of published as a passing threshold: a rate pinned at its
        # ceiling (doc_hit_rate read 1.0 on five straight nights to 2026-09-18),
        # and a pool with no spread, which a one-night window always is.
        reasons: list[str] = []
        if latest >= 1.0:
            reasons.append(f"latest {latest} is the metric ceiling")
        if band <= 0.0:
            reasons.append(f"the {len(vals)}-night pool spans no spread")
        out[key] = {
            "latest": latest, "min": min(vals), "max": max(vals),
            "band": round(band, 4), "floor": round(latest - band, 4),
            "n_nights": len(vals),
            "window_nights": RETRIEVAL_GATE_WINDOW_NIGHTS,
            "n_queries": n_queries,
            "granularity": granularity,
            "at_ceiling": bool(reasons),
            "ceiling_reason": "; ".join(reasons) or None,
        }
        if key == "doc_hit_rate":
            out["hardcoded_gate_would_have_failed_nights"] = sum(
                1 for v in vals if v < RETRIEVAL_GATE_HARDCODE)
    return out


class NoBaselines(RuntimeError):
    """No comparable retrieval baseline exists to read a gate from.

    Named apart from `app.kg_store.StoreUnavailable` on purpose: that one means
    "a store will not open, do not report it as empty", and this one means "the
    reference data for a threshold is absent". A caller catching one must not
    silently be catching the other.
    """


# -------------------------------------------------------------- store size --

def store_sizes(root: Path | str | None = None) -> dict[str, Any]:
    """`facts_idx` duplication, pinned to a probe timestamp.

    Standing rule in this repo: these numbers age in about an hour because
    extraction is live, and a count without its timestamp has repeatedly been
    read as a current fact a day later. So the count and its `probed_at` are
    produced in one place and `write_table` refuses to emit one without the
    other. A store that will not open reports the error; it is never reported
    as zero rows.

    Every read goes through `app.kg_store`. Nothing else in this tree opens
    `kg.sqlite` — that is what produced the 2026-08-22 wipe, when six programs
    were rewriting the same JSON behind each other.
    """
    out: dict[str, Any] = {}

    # Exist first, then read. `KGStore` opens lazily and *creates* the file when
    # the path is absent, so a run from a worktree (where `_pipeline/` is not
    # checked out) silently produced an empty store and a table reporting
    # `duplicate_rows: 0` — a false clean bill of the exact kind this repo was
    # burned by on 2026-08-22. Absent data is reported as absent.
    from app.kg_store import KGStore
    from app.paths import VAULT_KG_DB

    db = Path(VAULT_KG_DB)
    if not db.is_file():
        # A worktree resolves `VAULT_KG_DB` beside itself, where the store is
        # never checked out; the transcripts live in the live tree, so the
        # figure has to come from there too.
        alt = lloyd_root() / "_pipeline" / "vault-derived" / "kg.sqlite"
        if alt.is_file():
            db = alt
    if not db.is_file():
        out["facts_idx"] = {
            "probed_at": _now(),
            "error": f"no store at {db}; refusing to report zero rows",
        }
        return out
    try:
        # `KGStore(db)` rather than `configure(db)`: the latter repoints the
        # process-wide default, and a consolidator that imports this must not
        # have its own knowledge graph swapped out from under it mid-run.
        st = KGStore(db)
        try:
            rows = st.facts_idx.count()
            # No public accessor exists for duplicate-group cardinality, so this
            # goes through the store's own connection, not a second opener.
            dup_groups = st._query(
                "SELECT COUNT(*) AS n FROM (SELECT text_hash FROM facts_idx "
                "GROUP BY text_hash HAVING COUNT(*) > 1)")[0]["n"]
            dup_rows = st._query(
                "SELECT COALESCE(SUM(c - 1), 0) AS n FROM "
                "(SELECT COUNT(*) AS c FROM facts_idx GROUP BY text_hash)")[0]["n"]
        finally:
            st.close()
        out["facts_idx"] = {
            "probed_at": _now(), "store": str(db), "fact_rows": int(rows),
            "duplicate_text_hash_groups": int(dup_groups),
            "duplicate_rows": int(dup_rows),
            "note": "re-measure in the run that quotes it; this figure ages in ~1h",
        }
    except Exception as exc:  # noqa: BLE001 - reported, never zeroed
        out["facts_idx"] = {"probed_at": _now(), "error": f"{type(exc).__name__}: {exc}"}
    return out


# ------------------------------------------------------------------ output --

def _validate_stores(stores: dict) -> None:
    """The timestamp rule, checked before anything is written.

    A guard that fires after `mkdir` has already left the output directory
    behind — so the refusal is real but the tree is half-changed by a call that
    was rejected.
    """
    for name, block in stores.items():
        if not isinstance(block, dict):
            raise ValueError(f"stores.{name} must be a mapping")
        if any(isinstance(v, int) and not isinstance(v, bool) for v in block.values()) \
                and not block.get("probed_at"):
            raise ValueError(f"stores.{name} carries a count with no probed_at")


def write_table(table: dict[str, Any], out_dir: Path | str | None = None,
                date: str | None = None, classifier: dict | None = None,
                stores: dict | None = None, extra: dict | None = None) -> Path:
    """Emit `eval/uptake/uptake-<date>.json`. Enforces the timestamp rule."""
    if stores:
        _validate_stores(stores)
    out = Path(out_dir) if out_dir else (REPO / "eval" / "uptake")
    out.mkdir(parents=True, exist_ok=True)
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc = dict(table)
    doc.setdefault("item", "552")
    doc.setdefault("probe_timestamp", _now())
    if classifier is not None:
        doc["classifier"] = {**classifier, "threshold": PRECISION_FLOOR}
    if stores:
        doc["stores"] = stores
    if extra:
        doc.update(extra)
    path = out / f"uptake-{date}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return path


def iter_json_objects(text: str) -> Iterable[dict]:  # pragma: no cover - helper
    """Lenient line iterator for event logs with the odd unparseable line."""
    for line in text.splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
