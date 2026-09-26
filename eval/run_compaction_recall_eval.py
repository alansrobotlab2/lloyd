#!/usr/bin/env python3
"""Planted-fact recall under the context policies (backlog #600).

Lloyd rewrites long histories in two places and nothing had ever priced
what either costs in recall:

  * the turn-start stack, `app.compaction.load_and_compact_session`
    (microcompact pre-pass above `compaction.microcompact.trigger_fraction`
    of the truncation threshold, then LLM summarize, then truncate), which
    budgets the HISTORY ONLY — the system prompt and the tool schemas are
    not in its estimate;
  * the in-turn pass, `loop._intra_turn_microcompact` through the
    `harness.context_relief` ladder, which triggers on the engine's REPORTED
    prompt size — history plus that fixed overhead — once a turn has made a
    tool call and carries >= 15 tool results.

WHAT ONE RUN IS
---------------
A synthetic session (`build_session`) is written as session JSON: filler
turns of real Lloyd-shaped traffic (Read/Grep/Bash call-result pairs whose
bodies are slices of this tree's own files, plus short prose), with ONE
planted turn at a known depth whose Read result is a deploy-notes file
stating two facts:

  distinctive  a release codename (`QUARTZ-HERON-2291`) — one of a kind;
  ambiguous    the port the `billing-east` relay listens on NOW — the notes
               also give its old port, and the filler is salted with the
               ports of sibling relays, all permutations of the same digits.

Every probe turn Reads an open-items file before it answers, which is what
arms the in-turn pass — the mechanism that actually fires in production
(#1078: 802 in-turn clearings against 0 summarizations in the retained
logs). Two shapes, `--probe`:

  early  the user message asks for both facts (`CODENAME:` / `PORT:`)
         and says to check the open items first. The model knows the
         question at iteration 1, answers it in that iteration's
         reasoning, and preserved thinking carries the answer past the
         clearing — the pilot's production arm lost the planted result off
         the wire in 5 of 6 runs and still answered all 6.
  late   (default) the question is in the open-items file, so the need for
         an old fact arises only after the in-turn pass has cleared it.
         The user message authorises answering it; a first cut had the
         file *order* the answer, and Lloyd, correctly, refused to take
         instructions from a file (and to repeat a "passphrase" — the
         distinctive fact is a release codename for that reason).

The run goes through the real code: `load_and_compact_session` under the
arm's `compaction:` block, `_prepare_messages_for_harness`, and
`app.harness.run_query` with the production `RunOptions` kwargs
(`app.mcp_discovery._get_harness_kwargs`) plus the arm's overrides,
against the live primary. Only the tool pool is local (`EvalPool`): the
real tool schemas, snapshotted once from the aggregator, served by
in-process stubs — Read serves the synthetic files, this tree's files and
the probe session's OWN record (its session JSON and the spill files
microcompaction wrote); Grep searches the same and honours `path`;
`recall_observation` is the served tool's own resolver; everything else
answers "not available in this evaluation". Until 2026-09-25 Read and Grep
saw every session under the scratch root — the other arms' and seeds'
records, each planting a different codename and port — so a Grep for
`billing-east` could answer with a stranger's facts; runs before that date
carry that leak, and it fell on the arms that searched most (`tool_clear`
Grepped in every failing row of the 2026-09-24 baseline). No tool can change the
machine, so no sandboxed session is needed, and the ids are `pt-eval-*`
anyway.

Before the probe's first request the runner sends the UNCOMPACTED prompt
with `max_tokens: 1` (the warm-up): that is the prefix cache a session in
production arrives at this turn with — its previous turn sent everything
but the new user message. The probe's TTFT and cached-token numbers are
therefore what a rewrite costs against a warm session, not against a cold
engine. A warm-up that cannot fit the window (sessions past ~200k) is
clipped from the END to the longest head of whole turns that fits — the
head is what a prefix cache is made of, so a front clip would warm nothing
the probe shares — and is skipped when not even one turn fits. It is sent
at most once, never retried, and a failure is recorded on the row
(`warmup.status`: sent / clipped / skipped / error), never raised into the
probe: the first live summary run died 24/24 because an oversized warm-up
400ed inside the loop's own stream and was re-sent on every recovery attempt.

ARMS
----
  none        no compaction beyond the window: turn-start microcompact off,
              `mode: truncate` (fires only past the truncation threshold),
              in-turn relief off. The 413 overflow recovery stays — it is
              the window limit.
  production  config.yaml as it is, both passes.
  tool_clear  tool-output clearing only, aggressively: microcompact
              trigger/target 0.2/0.1 at turn start and in turn, no
              summarize (`mode: truncate`), the rest of the relief ladder
              off. The video's blunt preset.
  self_record #1514's FREE route: `tool_clear` plus one clause in every
              cleared-result marker naming the session's own record
              (`sessions/<sid>.json`, `sessions/<sid>.tool-results/`). No new
              tool. The baseline `observation` must beat.
  observation #1481: `tool_clear` with observation stubs (id + verbatim head)
              and `recall_observation(id)` advertised.
  production_self_record / production_observation
              the same two switches at production's thresholds.
  rung4 / rung4_lossy / rung4_self_record
              #1499: relief rung 4 (truncation) is the rung that drops the
              planted result — today's spill-first rung, the pre-2026-09-11
              lossy one, and today's with #1514's clause.
  raised      trigger/target 0.9/0.7 at both passes: fire later, keep more.
  trigger90   trigger 0.9, target 0.52: fire later, clear as far as today.
  summary_legacy     summarize layer, regenerate-every-turn 9-section summary.
  summary_persisted  summarize layer with `persist_summary: true` (D2): the
              incremental Goal/Constraints/Progress/Decisions/Next steps
              record. See SUMMARY ARMS below.
  memory_flush  P3: `summary_legacy` preceded by a memory-flush turn
              (`app/memory_flush.py`, allow-listed to the memory tools;
              `memory_add`/`fact_add` are recorded, not written). The flush
              runs over the conversation as it stood when production's flush
              trigger would have fired (`flush_history`: the head of whole
              turns before the first whose turn-start-compacted prompt
              crosses `trigger_fraction` x the threshold), never over the whole
              uncompacted session, which at these sizes is past the window.
              The saved entries are rendered into the probe's system prompt,
              as the next turn's memory block would carry them. `flush_saved`
              says which planted facts the flush wrote down; recall against
              `summary_legacy` says whether that helped. A summary arm too:
              same shape, sizes and gate (below).

SUMMARY ARMS
------------
The question these two answer: does the persisted incremental 5-section
summary keep the planted facts at least as well as the regenerate-every-turn
9-section one, and what does it cost (summariser calls and wall time at turn
start). Three things had to be true for either to measure it, and the first
live run (24/24 rows `error`) had none of them:

  * The summary has to FIRE. On the default `tool` session shape the
    turn-start microcompact pass clears a 240k session to ~116k and the
    summarize layer answers `under_threshold` — both arms identical.
    Switching microcompact off instead is not a fix: `summarize_history` has
    no input bound, the older block of an uncleared 240k session is ~230k
    estimator tokens, and tool output tokenizes at ~1.22x the estimator on
    the primary (measured with its tokenizer), so the legacy request is a
    guaranteed 400 and falls back to truncation — a configuration production
    never runs, rigged against one arm. So the pair runs on the
    `conversation` shape (`--shape conversation`, the default when every
    requested arm is a summary arm) with the production microcompact left on:
    each filler turn is one Read plus a long assistant discussion drawn from
    `architecture/*.md`, so prose (~0.87x the estimator) dominates, the
    clearing frees the Read results and the history stays over the threshold.
    That is the shape the summarize layer exists for in production: a long
    conversation, not a long tool log.
  * The fact has to be IN what the summariser reads. Microcompact clears the
    planted Read result like any other, so in this shape the assistant's
    reply to it restates both facts (codename; billing-east moved from the old
    port to the new one), and salted turns mention sibling relays' ports in
    prose. The summariser sees the facts only as conversation, which is where
    a summary has to carry them from.
  * The fact has to be SUMMARISED, not kept verbatim. The persisted arm folds
    at most `max_folds_per_turn` x `summary_input_budget_tokens` (3 x 48k) of
    older history per turn, starting from the oldest, and keeps the rest
    verbatim; the legacy arm summarises everything but the last
    `keep_recent_turns`. A cold session is the persisted arm's state on the
    turn it first crosses the threshold, which is what a real session looks
    like then. Run it at depths inside the first ~140k (`--depths 0.1,0.3,0.5`
    at the recommended sizes) so both arms summarise the fact.

Validity: a summary-arm row is kept only when its summarize layer produced a
summary on this turn (`summarize_outcome == summarized`), the summary row
survived the truncation fallback (`summary_truncated_away` otherwise), and
the planted fact is not still verbatim in the history (`fact_not_summarized`).
Both are known at turn start, so the probe is not spent on a dropped row.
Every row carries `summary_has` (which facts the summary text itself names
— the direct fidelity reading, independent of the model's answer),
`summarizer` (calls, wall seconds, input/output chars) and
`turn_start_wall_s`. `--dry` stubs the summarisers so a dry run shows
whether the layer would fire without any engine traffic.

Sizes. On this shape `--sizes` counts the non-tool rows (the raw session is
~1.6x that). The window for a fair comparison is narrow, and it is the
legacy summariser that sets it: the layer fires only once the history after
clearing passes the threshold (210,144 estimator tokens; ~conversation + 26k),
and `summarize_history` sends that whole older block unbounded, which fits
the 262,144 window only up to ~253k real tokens (8,000 output + a 631-token
prompt). Measured with the primary's tokenizer on the 2026-09-24 tree:
conversation 180k does not fire; 186k and 192k fire on all 12 seeds below
with a legacy request of 228k-246k tokens; 195k reaches 249k and 210k is
259k-263k, a 400 that falls back to truncation (`summary_not_fired:
empty_summary`). That band is itself a finding about production: legacy can
summarise only in the first ~40k of growth past the threshold, and re-sends
all of it every turn. The persisted arm's folds are ~55k real tokens each
at any size. Re-run with `--dry` on the tree the run will use (the filler
is this tree's own files): every row should read `dry`, `summarized`.

    LLOYD_DATA=<scratch> .venvs/lloyd/bin/python eval/run_compaction_recall_eval.py \\
        --data-root <scratch> --tools-snapshot <scratch>/tools.json \\
        --arms summary_legacy,summary_persisted \\
        --sizes 186000,192000 --depths 0.1,0.3,0.5 --sessions 6 --probe late \\
        --out <scratch>/d2.json

The in-turn trigger is a fraction of the truncation threshold (210,144)
compared against the REPORTED prompt, which carries ~55-75k of system
prompt and tool schemas; the turn-start trigger is the same fraction of an
estimate of the history alone. So at 0.72 the in-turn pass fires on a
session whose history the turn-start pass calls ~90k, and the turn-start
microcompact pass never gets the chance: every production row in the
2026-09-24 baseline was cleared in-turn, none at turn start.

VALIDITY GATE
-------------
Which mechanism fired is read off the #1078 record
(`app.compaction_record`: the turn-start projection plus every relief
pass the loop booked on the open turn), never off the log. A run is kept
only when its arm exercised its own feature: `none` must have freed
nothing, `production` and `tool_clear` must have freed something. A
dropped run is counted per arm (`dropped`) and excluded from recall.
`production` is gated before the engine is spent when the turn-start pass
declined AND the estimated prompt cannot reach the in-turn trigger.

GRADING
-------
`grade_answer` is a code check on the labelled lines (and, failing a
label, on the whole answer). It decides every field it can; only an
`undecided` field is offered to a judge, and the default judge is none
(undecided is reported, and counts as a miss). Distinctive and ambiguous
recall are always reported as two columns, never blended.

    LLOYD_DATA=<scratch> .venvs/lloyd/bin/python eval/run_compaction_recall_eval.py \\
        --data-root <scratch> --tools-snapshot <scratch>/tools.json \\
        --sessions 3 --sizes 90000,130000 --out <scratch>/pilot.json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Every tool a probe session uses. Naming them all non-compactable holds
#: relief rung 1 off, so a later rung is the one that acts (#1499's arms).
_ALL_FILLER_TOOLS = ("Read", "Grep", "Bash", "Glob", "Edit", "Write",
                     "recall_observation")

ARMS: dict[str, dict[str, Any]] = {
    "none": {
        "compaction": {"mode": "truncate", "microcompact": {"enabled": False}},
        "options": {"intra_turn_microcompact_enabled": False,
                    "context_relief_enabled": False},
        "expects_fire": False,
    },
    "production": {
        "compaction": {},
        "options": {},
        "expects_fire": True,
    },
    "tool_clear": {
        "compaction": {"mode": "truncate",
                       "microcompact": {"trigger_fraction": 0.2,
                                        "target_fraction": 0.1}},
        # The ladder's rung 1 is the in-turn clearing, so the ladder stays
        # on; its other rungs (reasoning, arguments) have nothing to act on
        # in a probe turn of plain Reads.
        "options": {"intra_turn_microcompact_trigger_fraction": 0.2,
                    "intra_turn_microcompact_target_fraction": 0.1},
        "expects_fire": True,
    },
    # #1514: the FREE route. `tool_clear` exactly, plus one clause in every
    # cleared-result marker naming the session's own record
    # (`sessions/<sid>.json` + `<sid>.tool-results/`,
    # `tool_result_spill.session_record_route`). No new tool, no new
    # permission — the baseline `observation` has to beat.
    "self_record": {
        "compaction": {"mode": "truncate",
                       "microcompact": {"trigger_fraction": 0.2,
                                        "target_fraction": 0.1,
                                        "name_session_record": True}},
        "options": {"intra_turn_microcompact_trigger_fraction": 0.2,
                    "intra_turn_microcompact_target_fraction": 0.1,
                    "intra_turn_microcompact_name_session_record": True},
        "expects_fire": True,
    },
    # #1481: `tool_clear` exactly, with observation stubs (id + bounded
    # verbatim head) and `recall_observation(id)` advertised to resolve them.
    "observation": {
        "compaction": {"mode": "truncate",
                       "microcompact": {"trigger_fraction": 0.2,
                                        "target_fraction": 0.1,
                                        "observation_stubs": True}},
        "options": {"intra_turn_microcompact_trigger_fraction": 0.2,
                    "intra_turn_microcompact_target_fraction": 0.1,
                    "intra_turn_microcompact_observation_stubs": True},
        "expects_fire": True,
    },
    # The same two switches at production's thresholds: what flipping one
    # on would actually change. Run beside `production`.
    "production_self_record": {
        "compaction": {"microcompact": {"name_session_record": True}},
        "options": {"intra_turn_microcompact_name_session_record": True},
        "expects_fire": True,
    },
    "production_observation": {
        "compaction": {"microcompact": {"observation_stubs": True}},
        "options": {"intra_turn_microcompact_observation_stubs": True},
        "expects_fire": True,
    },
    # #1499: relief rung 4 (`_truncate_largest_tool_results`) as the rung
    # that drops the planted result. Rung 1 is held off by naming every tool
    # the session uses non-compactable, the turn-start pass is off, and the
    # target is low enough (0.3) that rung 4 reaches results the planted
    # one's size. `rung4` is today's rung (it spills first and the notice
    # names the file, since 2026-09-11); `rung4_lossy` is the rung before
    # that — no spill, "re-run narrower" — which is what #1499 describes;
    # `rung4_self_record` adds #1514's clause to the notice.
    "rung4": {
        "compaction": {"mode": "truncate", "microcompact": {"enabled": False}},
        "options": {"intra_turn_microcompact_trigger_fraction": 0.72,
                    "intra_turn_microcompact_target_fraction": 0.3,
                    "intra_turn_microcompact_non_compactable": _ALL_FILLER_TOOLS},
        "expects_fire": True,
        "expects_rung": "truncate",
    },
    "rung4_lossy": {
        "compaction": {"mode": "truncate", "microcompact": {"enabled": False}},
        "options": {"intra_turn_microcompact_trigger_fraction": 0.72,
                    "intra_turn_microcompact_target_fraction": 0.3,
                    "intra_turn_microcompact_non_compactable": _ALL_FILLER_TOOLS},
        "expects_fire": True,
        "expects_rung": "truncate",
        "rung4_lossy": True,
    },
    "rung4_self_record": {
        "compaction": {"mode": "truncate", "microcompact": {"enabled": False}},
        "options": {"intra_turn_microcompact_trigger_fraction": 0.72,
                    "intra_turn_microcompact_target_fraction": 0.3,
                    "intra_turn_microcompact_non_compactable": _ALL_FILLER_TOOLS,
                    "intra_turn_microcompact_name_session_record": True},
        "expects_fire": True,
        "expects_rung": "truncate",
    },
    # The candidate the first three arms pointed at: production's mechanism
    # with the wall moved toward the window. The in-turn trigger is measured
    # on the REPORTED prompt (history plus the system prompt and tool
    # schemas), so 0.72 fires at ~151k reported — a session holding only
    # ~90k of history by the turn-start estimate. 0.9/0.7 is 189k -> 147k.
    "raised": {
        "compaction": {"microcompact": {"trigger_fraction": 0.9,
                                        "target_fraction": 0.7}},
        "options": {"intra_turn_microcompact_trigger_fraction": 0.9,
                    "intra_turn_microcompact_target_fraction": 0.7},
        "expects_fire": True,
    },
    # Moving the target with the trigger (above) re-prefills more after a
    # clearing, so this one moves the trigger only: 189k -> 109k.
    "trigger90": {
        "compaction": {"microcompact": {"trigger_fraction": 0.9,
                                        "target_fraction": 0.52}},
        "options": {"intra_turn_microcompact_trigger_fraction": 0.9,
                    "intra_turn_microcompact_target_fraction": 0.52},
        "expects_fire": True,
    },
    # D2 (review 2026-09-24): the summary formats, head to head. Both force
    # the summarize layer; only the persisted arm writes and folds the
    # `data["compaction"]` record (Goal / Constraints / Progress / Decisions /
    # Next steps + ledger-rendered Files touched) through
    # `compaction_llm.summarize_incremental`. The legacy arm is the 9-section
    # regenerate-every-turn summary. This pair gates flipping
    # `compaction.persist_summary` on; see SUMMARY ARMS in the header for why
    # they run on the `conversation` session shape with microcompact left ON,
    # and for the sizes and depths to run them at.
    "summary_legacy": {
        "compaction": {"mode": "summarize", "persist_summary": False},
        "options": {},
        "expects_fire": True,
        "expects_summary": True,
    },
    "summary_persisted": {
        "compaction": {"mode": "summarize", "persist_summary": True},
        "options": {},
        "expects_fire": True,
        "expects_summary": True,
    },
    # P3 (review 2026-09-24): the flush before the wall, against
    # `summary_legacy` at the same sizes. This pair gates flipping
    # `compaction.memory_flush.enabled` on. A summary arm, so it runs on the
    # conversation shape at the SUMMARY ARMS sizes:
    #   --arms summary_legacy,memory_flush --sizes 186000,192000
    # The flush sees the first ~40% of rows there (`flush_history`), so a
    # fact at depth 0.5 arrives after the flush — a control, not a miss;
    # `flush.planted_in_history` says which rows are which.
    "memory_flush": {
        "compaction": {"mode": "summarize", "persist_summary": False,
                       "memory_flush": {"enabled": True}},
        "options": {},
        "expects_fire": True,
        "expects_summary": True,
        "flush": True,
    },
}

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_WORDS_A = ["QUARTZ", "COBALT", "TUNDRA", "SAFFRON", "GRANITE", "MAGPIE",
            "LANTERN", "OSPREY", "VELVET", "CINDER", "HARBOR", "JUNIPER"]
_WORDS_B = ["HERON", "FALCON", "OTTER", "BADGER", "WREN", "MARTEN",
            "PLOVER", "LYNX", "IBIS", "STOAT", "KESTREL", "VOLE"]
_SIBLINGS = ["billing-west", "search-east", "billing-central", "ledger-east",
             "billing-south", "search-west", "ledger-west", "auth-east"]
_REQUESTS = [
    "Can you look at how {stem} handles its config?",
    "Check {stem} for anything that would break on a restart.",
    "What does {stem} do with errors — skim it for me.",
    "Pull up {stem}, I want to see the main entry point.",
    "Grep around {stem} for where the timeout is set.",
    "Keep going through {stem}; flag anything odd.",
]
_PROSE = [
    "Read through {stem}. The structure is straightforward: a small public "
    "surface, the real work in two private helpers, and the config read once "
    "at call time. Nothing here looks like it would misbehave on a restart.",
    "{stem} is mostly plumbing. The error paths log and continue rather than "
    "raise, which is deliberate given the comments. I'd leave it alone.",
    "Checked {stem}. The timeout comes from config with a sane default; the "
    "only thing worth noting is that it is re-read per call, so edits apply "
    "without a restart.",
    "Nothing blocking in {stem}. A couple of long functions, but each has a "
    "docstring explaining why, and the tests next to it pin the edge cases.",
]

NOTES_PATH = "notes/deploy-halcyon-{key}.md"
OPEN_ITEMS_PATH = "ops/open-items-{key}.md"


@dataclass
class Planted:
    # A release codename, not a credential: the first cut asked for a
    # "rollback passphrase" and the model, correctly, declined to repeat a
    # secret into the chat — a refusal that graded as a recall miss.
    passphrase: str
    port: str
    old_port: str
    distractor_ports: dict[str, str]

    def all_ports(self) -> set[str]:
        return {self.port, self.old_port, *self.distractor_ports.values()}


@dataclass
class Session:
    key: str
    seed: int
    target_tokens: int
    depth: float
    messages: list[dict[str, Any]]
    files: dict[str, str]
    planted: Planted
    planted_index: int          # message index of the planted tool result
    probe: str
    est_tokens: int = 0
    sha256: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _corpus_files(root: Path = ROOT) -> list[Path]:
    """This tree's own text, deterministic order. Filler that tokenizes like
    Lloyd's real tool results, because it is what those results read."""
    # A file that quotes the spill marker would be cleared by microcompact's
    # spill-aware pass whatever the budget, which is a fire this eval did not
    # plant (tool_result_spill.py and its readers are the ones that do).
    from app.harness.tool_result_spill import PERSISTED_OUTPUT_TAG
    out: list[Path] = []
    for sub, pat in (("app", "*.py"), ("agent_mcp", "*.py"),
                     ("scripts", "*.py"), ("architecture", "*.md")):
        base = root / sub
        if base.is_dir():
            out.extend(p for p in sorted(base.rglob(pat))
                       if p.is_file() and "tests" not in p.parts
                       and p.stat().st_size > 6_000
                       and PERSISTED_OUTPUT_TAG not in p.read_text(errors="replace"))
    return out


def make_planted(rng: random.Random) -> Planted:
    passphrase = f"{rng.choice(_WORDS_A)}-{rng.choice(_WORDS_B)}-{rng.randint(1000, 9999)}"
    while True:
        digits = [str(d) for d in rng.sample(range(1, 10), 3)]
        perms = sorted({"7" + "".join(p) for p in _perms(digits)})
        if len(perms) >= 2 + len(_SIBLINGS[:4]):
            break
    rng.shuffle(perms)
    port, old_port, *rest = perms
    distractors = dict(zip(_SIBLINGS, rest))
    return Planted(passphrase=passphrase, port=port, old_port=old_port,
                   distractor_ports=distractors)


def _perms(items: list[str]) -> Iterable[tuple[str, ...]]:
    if len(items) <= 1:
        yield tuple(items)
        return
    for i, x in enumerate(items):
        for rest in _perms(items[:i] + items[i + 1:]):
            yield (x, *rest)


def _notes_body(p: Planted, rng: random.Random) -> str:
    sib = list(p.distractor_ports.items())
    lines = [
        "# Halcyon deploy notes",
        "",
        "Owner: platform. Last touched during the relay migration.",
        "",
        "## Release",
        f"- This release ships under the codename `{p.passphrase}`; the "
        "changelog and the dashboards are labelled with it.",
        "- Rollback takes ~4 minutes; do not restart the relays during it.",
        "",
        "## Relays",
        f"- billing-east relay was on port {p.old_port} until the migration; "
        f"it listens on port {p.port} now.",
        f"- {sib[0][0]} relay: port {sib[0][1]} (unchanged).",
        f"- {sib[1][0]} relay: port {sib[1][1]}.",
        "",
        "## Misc",
    ]
    for i in range(40):
        lines.append(
            f"- item {i}: checked health endpoint, latency p50 "
            f"{rng.randint(8, 60)} ms, error budget {rng.randint(90, 100)}% left, "
            "no action needed this week; revisit after the quarter closes.")
    return "\n".join(lines)


def _open_items_body(late: bool = False) -> str:
    body = ("# Open items\n\n- open items: none outstanding as of the last "
            "review; the release manager signed off.\n")
    if late:
        body += ("\n## Open question (Alan)\n\nFrom the Halcyon deploy notes we "
                 "went through earlier: what is the release codename, and which "
                 "port does the billing-east relay listen on now? Answer with "
                 "exactly two lines:\nCODENAME: <value>\nPORT: <value>\n")
    return body


PROBE = (
    "Before we stop: read {open_items} to confirm nothing is outstanding. "
    "Then, from the deploy notes we read earlier in this session, tell me "
    "the Halcyon release codename and the port the billing-east relay "
    "listens on now. Finish with exactly two lines:\n"
    "CODENAME: <value>\nPORT: <value>"
)

# The late probe: the question is in the file, so the turn cannot know it at
# iteration 1. With the early probe the model reads the question before the
# first tool call, answers it in that iteration's reasoning, and preserved
# thinking carries the answer past an in-turn clearing (pilot: 5 of 6
# production runs had the planted result cleared off the wire and still
# answered from the reasoning). The late probe is the case where the need
# for an old fact arises only after the clearing. The user authorises the
# file's question explicitly — an instruction the file issues on its own is
# one Lloyd rightly refuses.
PROBE_LATE = (
    "Before we stop: read {open_items} and answer the open question logged "
    "there, using what we saw earlier in this session."
)


def _read_render(text: str, start_line: int = 1) -> str:
    """The builtin Read's `cat -n` shape."""
    return "\n".join(f"{i:6d}\t{ln}" for i, ln in
                     enumerate(text.splitlines(), start=start_line))


def _tc(cid: str, name: str, args: dict) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _msg(role: str, text: str, **kw) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}], **kw}


def _filler_turn(i: int, rng: random.Random, corpus: list[Path], root: Path,
                 planted: Planted, salt: bool) -> list[dict]:
    path = rng.choice(corpus)
    rel = str(path.relative_to(root))
    text = path.read_text(errors="replace")
    msgs = [_msg("user", rng.choice(_REQUESTS).format(stem=rel))]
    calls, results = [], []
    for j in range(rng.randint(1, 3)):
        cid = f"call_{i}_{j}"
        kind = rng.random()
        lines = text.splitlines()
        if kind < 0.6:
            a = rng.randint(0, max(0, len(lines) - 60))
            b = min(len(lines), a + rng.randint(60, 260))
            body = _read_render("\n".join(lines[a:b]), a + 1)
            calls.append(_tc(cid, "Read", {"summary": f"Reading {path.name}",
                                           "file_path": str(root / rel),
                                           "offset": a + 1, "limit": b - a}))
        elif kind < 0.85:
            word = rng.choice([w for w in re.findall(r"[a-z_]{6,}", text)] or ["return"])
            hits = [f"{rel}:{n + 1}:{ln}" for n, ln in enumerate(lines) if word in ln][:80]
            body = "\n".join(hits) or "No matches found"
            calls.append(_tc(cid, "Grep", {"summary": f"Searching for {word}",
                                           "pattern": word, "path": str(root / rel),
                                           "output_mode": "content", "-n": True}))
        else:
            a = rng.randint(0, max(0, len(lines) - 80))
            body = "\n".join(lines[a:a + rng.randint(40, 160)])
            calls.append(_tc(cid, "Bash", {"summary": f"Printing part of {path.name}",
                                           "command": f"sed -n '{a + 1},{a + 160}p' {rel}"}))
        if salt and j == 0:
            svc, port = rng.choice(list(planted.distractor_ports.items()))
            body += f"\n# ops: {svc} relay listens on port {port}"
        results.append(_msg("tool", body, tool_call_id=cid))
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": ""}],
                 "tool_calls": calls})
    msgs.extend(results)
    msgs.append(_msg("assistant", rng.choice(_PROSE).format(stem=rel)))
    return msgs


_DISCUSS = [
    "Walk me through how {stem} fits with the design notes.",
    "Read {stem} and tell me whether it still matches what we documented.",
    "What's the reasoning behind {stem}? Talk it through.",
    "Explain {stem} to me like I'm reviewing it cold.",
]


def _prose_passage(rng: random.Random, doc: Path, lo: int = 3_000,
                   hi: int = 9_000) -> str:
    """A run of whole paragraphs from `doc`, `lo`..`hi` characters long."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc.read_text(errors="replace"))
             if p.strip()]
    if not paras:
        return ""
    want = rng.randint(lo, hi)
    start = rng.randrange(len(paras))
    out: list[str] = []
    n = 0
    for p in paras[start:] + paras[:start]:
        out.append(p)
        n += len(p)
        if n >= want:
            break
    return "\n\n".join(out)[:hi]


def _conversation_turn(i: int, rng: random.Random, corpus: list[Path],
                       prose: list[Path], root: Path, planted: Planted,
                       salt: bool) -> list[dict]:
    """The `conversation` shape: one Read, then a long assistant discussion.

    The Read result is what microcompaction clears; the discussion — whole
    paragraphs of this tree's own design docs — is what it cannot, so a
    session built of these stays over the truncation threshold after the
    clearing and reaches the summarize layer (see SUMMARY ARMS)."""
    path = rng.choice(corpus)
    rel = str(path.relative_to(root))
    lines = path.read_text(errors="replace").splitlines()
    cid = f"call_{i}_0"
    a = rng.randint(0, max(0, len(lines) - 40))
    b = min(len(lines), a + rng.randint(40, 120))
    doc = rng.choice(prose)
    reply = (f"Read through {rel}. How it lines up with "
             f"{doc.relative_to(root)}:\n\n" + _prose_passage(rng, doc))
    if salt:
        svc, port = rng.choice(list(planted.distractor_ports.items()))
        reply += (f"\n\nSide note from the ops channel while I was in there: "
                  f"the {svc} relay listens on port {port}.")
    return [
        _msg("user", rng.choice(_DISCUSS).format(stem=rel)),
        {"role": "assistant", "content": [{"type": "text", "text": ""}],
         "tool_calls": [_tc(cid, "Read", {"summary": f"Reading {path.name}",
                                          "file_path": str(root / rel),
                                          "offset": a + 1, "limit": b - a})]},
        _msg("tool", _read_render("\n".join(lines[a:b]), a + 1), tool_call_id=cid),
        _msg("assistant", reply),
    ]


def _planted_turn(key: str, planted: Planted, rng: random.Random,
                  shape: str = "tool") -> tuple[list[dict], str]:
    path = NOTES_PATH.format(key=key)
    body = _notes_body(planted, rng)
    cid = "call_planted"
    if shape == "conversation":
        # The Read result is cleared at turn start like every other one, so the
        # facts have to be in the conversation for a summary to carry them.
        sib, sib_port = next(iter(planted.distractor_ports.items()))
        reply = (f"Got the deploy notes open. This release ships under the "
                 f"codename `{planted.passphrase}`. The billing-east relay was "
                 f"on port {planted.old_port} until the migration and listens "
                 f"on port {planted.port} now; the {sib} relay stays on "
                 f"{sib_port}. Nothing in them blocks what we're doing; "
                 "carrying on with the review.")
    else:
        reply = ("Got the deploy notes open. Nothing in them blocks "
                 "what we're doing; carrying on with the review.")
    return [
        _msg("user", f"Pull up the deploy notes at {path} — I want them in "
                     "front of us before we touch the relays."),
        {"role": "assistant", "content": [{"type": "text", "text": ""}],
         "tool_calls": [_tc(cid, "Read", {"summary": "Reading the deploy notes",
                                          "file_path": path})]},
        _msg("tool", _read_render(body), tool_call_id=cid),
        _msg("assistant", reply),
    ], body


SHAPES = ("tool", "conversation")


def build_session(seed: int, target_tokens: int, depth: float,
                  *, root: Path = ROOT, corpus: list[Path] | None = None,
                  probe: str = "early", shape: str = "tool") -> Session:
    """One synthetic session: filler to `target_tokens` (estimator units, the
    ones every compaction trigger is written in) with the planted turn at
    `depth` of the way through it. Deterministic in (seed, tree, shape).

    `shape="tool"` is the #600 shape (mostly tool results), and
    `target_tokens` is the whole history. `"conversation"` is the summary
    arms' (see SUMMARY ARMS in the module docstring), and `target_tokens`
    counts every row BUT the tool results — the part microcompaction leaves —
    so the raw session (`est_tokens`) is ~1.6x larger."""
    from app.compaction import estimate_conversation_tokens

    if shape not in SHAPES:
        raise ValueError(f"unknown session shape {shape!r}")
    rng = random.Random(seed)
    corpus = corpus if corpus is not None else _corpus_files(root)
    if not corpus:
        raise RuntimeError("no filler corpus under the tree")
    prose = [p for p in corpus if p.suffix == ".md"] or corpus
    # The tool shape keeps its #600 key, so an existing results file resumes.
    key = f"s{seed}-{target_tokens // 1000}k-d{int(depth * 100)}" \
        + ("-conv" if shape == "conversation" else "")
    planted = make_planted(rng)
    planted_msgs, notes = _planted_turn(key, planted, rng, shape)

    def counted(msgs: list[dict]) -> int:
        # The conversation shape budgets what microcompaction cannot clear —
        # everything but the tool results — because that, not the raw size,
        # is what decides whether the summarize layer fires: raw-size
        # targets put seeds of one size on both sides of the threshold.
        if shape == "conversation":
            msgs = [m for m in msgs if m.get("role") != "tool"]
        return estimate_conversation_tokens(msgs)

    budget = target_tokens - counted(planted_msgs)
    turns: list[list[dict]] = []
    total = 0
    i = 0
    while total < budget:
        if shape == "conversation":
            t = _conversation_turn(i, rng, corpus, prose, root, planted,
                                   salt=(i % 4 == 0))
        else:
            t = _filler_turn(i, rng, corpus, root, planted, salt=(i % 4 == 0))
        total += counted(t)
        turns.append(t)
        i += 1
    at = min(len(turns), max(0, round(depth * len(turns))))
    turns.insert(at, planted_msgs)
    messages = [m for t in turns for m in t]
    planted_index = next(n for n, m in enumerate(messages)
                         if m.get("tool_call_id") == "call_planted")
    late = probe == "late"
    probe_text = (PROBE_LATE if late else PROBE).format(
        open_items=OPEN_ITEMS_PATH.format(key=key))
    files = {NOTES_PATH.format(key=key): notes,
             OPEN_ITEMS_PATH.format(key=key): _open_items_body(late)}
    s = Session(key=key, seed=seed, target_tokens=target_tokens, depth=depth,
                messages=messages, files=files, planted=planted,
                planted_index=planted_index, probe=probe_text,
                est_tokens=estimate_conversation_tokens(messages),
                meta={"shape": shape})
    s.sha256 = hashlib.sha256(json.dumps(messages, sort_keys=True)
                              .encode()).hexdigest()
    return s


# ---------------------------------------------------------------------------
# Grading — code first, judge only for the residue
# ---------------------------------------------------------------------------

_LINE = {
    "distinctive": re.compile(r"CODENAME\s*[:=]\s*(.+)", re.I),
    "ambiguous": re.compile(r"PORT\s*[:=]\s*(.+)", re.I),
}


def _field_verdict(text: str, want: str, wrong: set[str], *, digits: bool,
                   labelled: bool) -> str:
    def has(v: str) -> bool:
        if digits:
            return re.search(rf"(?<!\d){re.escape(v)}(?!\d)", text) is not None
        return v.lower() in text.lower()
    if has(want):
        # An answer naming the right value AND a wrong one has not answered.
        return "undecided" if any(has(w) for w in wrong) else "hit"
    if any(has(w) for w in wrong):
        return "wrong"
    # A labelled line with something else on it is a wrong answer; an
    # unlabelled answer that names nothing at all is the judge's residue.
    return "wrong" if labelled and text.strip() else "undecided"


def grade_answer(text: str, planted: Planted) -> dict[str, str]:
    """`hit` / `wrong` / `undecided` per fact, from the text alone.

    A labelled line is authoritative. With no label, the whole answer is
    checked the same way, and an answer that mentions neither the value nor
    any distractor is `undecided` — that is the residue a judge may see.
    """
    text = text or ""
    wrong_ports = planted.all_ports() - {planted.port}
    out: dict[str, str] = {}
    for fact, pat in _LINE.items():
        m = None
        for m in pat.finditer(text):
            pass                      # the LAST labelled line is the answer
        line = m.group(1) if m else ""
        if fact == "distinctive":
            want, wrong, digits = planted.passphrase, set(), False
        else:
            want, wrong, digits = planted.port, wrong_ports, True
        out[fact] = _field_verdict(line if m else text, want, wrong,
                                   digits=digits, labelled=bool(m))
    return out


def decide(text: str, planted: Planted,
           judge: Callable[[str, str, Planted], str] | None = None) -> dict[str, Any]:
    """The run's verdict. The code check decides; `judge(fact, text, planted)`
    is consulted only for a field the code left `undecided`."""
    code = grade_answer(text, planted)
    final = dict(code)
    judged: list[str] = []
    if judge is not None:
        for fact, v in code.items():
            if v == "undecided":
                final[fact] = judge(fact, text, planted)
                judged.append(fact)
    return {"code": code, "final": final, "judged": judged}


# ---------------------------------------------------------------------------
# The local tool pool
# ---------------------------------------------------------------------------

class EvalPool:
    """Real tool schemas, stub handlers. Nothing here can change the machine.

    `session_id` (the probe's own `pt-eval-*` id) scopes what Read and Grep
    see under `spill_root`: only that session's own record —
    `sessions/<sid>.json` and `sessions/<sid>.tool-results/` — never the other
    arms' and sessions' records that share the scratch root, each of which
    plants a DIFFERENT codename and port (#1514: a Grep across them answered
    with a stranger's facts). Without it the pool behaves as it always did.
    Grep honours a `path` argument (file or directory), as the real one does.
    """

    def __init__(self, discovered: list, files: dict[str, str],
                 spill_root: Path, tree_root: Path = ROOT,
                 planted: Planted | None = None, memory: bool = False,
                 session_id: str = "") -> None:
        self._discovered = discovered
        # P3 arm only: the memory tools answer (writes are recorded in
        # `saved`, never written). Every other arm keeps refusing them.
        self.memory = memory
        self.saved: list[dict[str, Any]] = []
        self.files = files
        self.spill_root = spill_root
        self.tree_root = tree_root
        self.planted = planted
        self.session_id = session_id
        self.calls: list[tuple[str, dict]] = []
        # Which planted facts a tool result handed back to the model: the
        # re-retrieval the video says compaction forces.
        self.recovered: set[str] = set()
        # Which facts came back through which route (#1514/#1481): the
        # session's own record (Read/Grep naming `<sid>`), `recall_observation`,
        # or a plain tool call.
        self.recovered_via: dict[str, set[str]] = {}

    async def call_tool(self, name: str, args: dict, **kw) -> dict:
        out = await self._call(name, args, **kw)
        p = self.planted
        if p is not None and not out.get("is_error"):
            text = str(out.get("content") or "")
            got = set()
            if p.passphrase in text:
                got.add("distinctive")
            if f"listens on port {p.port} now" in text:
                got.add("ambiguous")
            self.recovered |= got
            if got:
                self.recovered_via.setdefault(self._route(name, args), set()).update(got)
        return out

    def _route(self, name: str, args: dict) -> str:
        bare = name.rsplit("__", 1)[-1]
        if bare == "recall_observation":
            return "recall_observation"
        target = str((args or {}).get("file_path") or (args or {}).get("path") or "")
        sid = self.session_id
        if sid and f"{sid}.json" in target:
            return "session_record"          # #1514's route: the transcript
        if sid and f"{sid}.tool-results" in target:
            # One named file is what every clear marker already offers; a
            # search of the directory is the record route.
            return "spill_file" if bare == "Read" else "session_record"
        return bare

    @property
    def discovered(self):
        return self._discovered

    def _own(self, rp: Path) -> bool:
        """Under `spill_root`, only this session's own record is visible."""
        if not self.session_id:
            return True
        sess = (self.spill_root / "sessions").resolve()
        return rp == sess / f"{self.session_id}.json" or \
            rp.parent == sess / f"{self.session_id}.tool-results"

    def _allowed(self, rp: Path) -> bool:
        spill = self.spill_root.resolve()
        if rp.is_relative_to(spill):
            return self._own(rp)
        return rp.is_relative_to(self.tree_root.resolve())

    def _lookup(self, path: str) -> str | None:
        if path in self.files:
            return self.files[path]
        for k, v in self.files.items():
            if path.endswith(k):
                return v
        try:
            rp = Path(path).expanduser().resolve()
            if rp.is_file() and self._allowed(rp):
                return rp.read_text(errors="replace")
        except (OSError, ValueError):
            pass
        return None

    def _grep_pool(self, path: str) -> dict[str, str]:
        """What a Grep searches: `path` when given (a file, or a directory this
        pool may read), else the session's files plus its own record."""
        if path:
            text = self._lookup(path)
            if text is not None:
                return {path: text}
            try:
                rp = Path(path).expanduser().resolve()
            except (OSError, ValueError):
                return {}
            if not rp.is_dir():
                return {}
            out: dict[str, str] = {}
            for fp in sorted(rp.rglob("*"))[:2000]:
                if fp.is_file() and self._allowed(fp.resolve()):
                    out[str(fp)] = fp.read_text(errors="replace")
            return out
        pool = dict(self.files)
        if self.spill_root.exists():
            for sp in sorted(self.spill_root.rglob("*")):
                if sp.is_file() and self._own(sp.resolve()):
                    pool[str(sp)] = sp.read_text(errors="replace")
        return pool

    async def _call(self, name: str, args: dict, **_kw) -> dict:
        bare = name.rsplit("__", 1)[-1]
        self.calls.append((bare, dict(args or {})))
        if bare == "Read":
            text = self._lookup(str(args.get("file_path") or ""))
            if text is None:
                return {"content": f"File does not exist: {args.get('file_path')}",
                        "is_error": True}
            off = max(1, int(args.get("offset") or 1))
            lim = int(args.get("limit") or 2000)
            lines = text.splitlines()[off - 1: off - 1 + lim]
            return {"content": _read_render("\n".join(lines), off), "is_error": False}
        if bare == "Grep":
            pat = str(args.get("pattern") or "")
            try:
                rx = re.compile(pat)
            except re.error:
                rx = re.compile(re.escape(pat))
            hits: list[str] = []
            for k, v in self._grep_pool(str(args.get("path") or "")).items():
                hits += [f"{k}:{n + 1}:{ln}" for n, ln in enumerate(v.splitlines())
                         if rx.search(ln)]
            return {"content": "\n".join(hits[:200]) or "No matches found",
                    "is_error": False}
        if bare == "recall_observation":
            # The served tool's own resolver and refusal (#1481), bound to this
            # probe's session exactly as the aggregator binds it.
            from agent_mcp.recall_observation import recall, resolve
            oid = str(args.get("id") or "")
            return {"content": recall(oid, self.session_id, int(args.get("offset") or 0)),
                    "is_error": resolve(oid, self.session_id) is None}
        if self.memory and bare in ("memory_add", "fact_add"):
            self.saved.append({"tool": bare, **dict(args or {})})
            return {"content": json.dumps({"ok": True}), "is_error": False}
        if self.memory and bare == "memory_read":
            return {"content": "(empty)", "is_error": False}
        if self.memory and bare == "fact_get":
            return {"content": json.dumps({"facts": []}), "is_error": False}
        return {"content": f"{bare} is not available in this evaluation; "
                           "use Read or Grep.", "is_error": True}


def flush_saved(saved: list[dict[str, Any]], planted: Planted) -> dict[str, bool]:
    """Which planted facts the flush turn wrote down (P3)."""
    text = "\n".join(json.dumps(e) for e in saved)
    return {"distinctive": planted.passphrase in text,
            "ambiguous": planted.port in text}


def render_saved_memory(saved: list[dict[str, Any]]) -> str:
    """The flush's entries as the next turn's system prompt would carry them."""
    lines = []
    for e in saved:
        if e.get("fact"):
            body = f"{e.get('entity') or '?'}: {e['fact']}"
        else:
            body = e.get("entry") or json.dumps(
                {k: v for k, v in e.items() if k != "tool"})
        lines.append(f"- {body}")
    return "<memory>\n" + "\n".join(lines) + "\n</memory>" if lines else ""


def _flush_tools(discovered: list, names: Iterable[str]) -> list:
    """The discovered schemas a flush turn is advertised (its allow-list)."""
    want = set(names)
    out = []
    for entry in discovered or []:
        tools = entry[1] if isinstance(entry, (list, tuple)) and len(entry) == 2 else []
        out += [t for t in tools if isinstance(t, dict) and t.get("name") in want]
    return out


async def flush_history(session: "Session", *, sid: str, data_root: Path,
                        system_prompt: str, discovered: list
                        ) -> tuple[list[dict] | None, dict[str, Any]]:
    """What the P3 flush turn is sent: the conversation as it stood when
    production's flush would have fired.

    Production flushes at the END of the turn whose engine-reported prompt
    first reached `trigger_fraction` x the compaction threshold, and the flush
    turn is sent that session through the ordinary turn-start stack. So this
    is the head of whole turns just before the first one whose compacted
    prompt — system prompt, the flush's allow-listed tool schemas, history,
    JSON-sized like the warm-up — crosses that bound. The whole uncompacted session is what the
    first cut sent, and at the summary arms' sizes that is past the window.
    The turn-start probes run with `mode_override="truncate"`: they size a
    head and must never call a summariser; a head the stack had to truncate
    is refused. No engine traffic, so `--dry` runs it too.
    """
    from app import memory_flush as MF
    from app.compaction import (get_context_window, load_and_compact_session,
                                truncation_threshold)
    from app.routers._messages_harness_adapter import _prepare_messages_for_harness

    cfg = MF.flush_cfg()
    window = get_context_window("primary")
    bound = min(int(float(cfg.get("trigger_fraction") or 0.85)
                    * truncation_threshold(window)),
                window - WARMUP_RESERVE_TOKENS)
    overhead = _json_tokens({"role": "system", "content": system_prompt}) \
        + _json_tokens(_flush_tools(discovered, cfg.get("tools") or MF.DEFAULT_TOOLS)) \
        + _json_tokens({"role": "user", "content": MF.FLUSH_PROMPT})
    msgs_all = session.messages
    bounds = [n for n, m in enumerate(msgs_all) if m.get("role") == "user" and n > 0] \
        + [len(msgs_all)]
    head_path = data_root / "sessions" / f"{sid}-flushhead.json"
    cache: dict[int, tuple[list[dict] | None, int]] = {}

    async def probe(end: int) -> tuple[list[dict] | None, int]:
        if end not in cache:
            head_path.write_text(json.dumps({"session_id": head_path.stem,
                                             "platform": "mission-control",
                                             "messages": msgs_all[:end]}))
            comp = await load_and_compact_session(head_path, model="primary",
                                                  mode_override="truncate")
            prepared = await _prepare_messages_for_harness(comp["history"], "primary")
            cost = overhead + sum(_json_tokens(m) for m in prepared)
            cache[end] = (None if comp.get("truncated") or cost > bound
                          else prepared, cost)
        return cache[end]

    # Forward, stopping at the first head that does not fit: the prompt
    # grows turn by turn until the trigger fires, and that crossing is the
    # flush. Not a bisection — the predicate is not monotone, because a head
    # past the microcompact trigger is cleared back under the bound, and the
    # largest such head is a session production would have flushed long
    # before it got there.
    best = None
    for n, end in enumerate(bounds):
        prepared, _cost = await probe(end)
        if prepared is None:
            break
        best = n
    info: dict[str, Any] = {"bound_tokens": bound, "rows_total": len(msgs_all)}
    if best is None:
        info.update(history_rows=0)
        return None, info
    prepared, cost = await probe(bounds[best])
    text = json.dumps(prepared)
    info.update(history_rows=bounds[best], est_prompt_tokens=cost,
                planted_in_history=session.planted.passphrase in text)
    return prepared, info


async def run_flush(session: "Session", *, sid: str, path: Path, discovered: list,
                    system_prompt: str, data_root: Path, base_url: str,
                    hk: dict[str, Any], messages: list[dict]) -> dict[str, Any]:
    """The P3 flush turn over `messages` (see `flush_history`). Needs an engine."""
    from app import memory_flush as MF
    from app.harness import HookRegistry, install_default_safety_hook
    from app.harness import loop as L
    from app.harness.options import RunOptions
    from app.mcp_discovery import _get_disallowed_tools

    cfg = MF.flush_cfg()
    pool = EvalPool(discovered, session.files, data_root,
                    planted=session.planted, memory=True)
    msgs = list(messages)
    msgs.append({"role": "user", "content": MF.FLUSH_PROMPT})
    turn_id = MF.new_turn_id()
    # Armed like the probe turn and production's flush turn (GATE_ARM_POINTS).
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    options = RunOptions(
        model="primary", base_url=base_url, system_prompt=system_prompt,
        max_turns=int(cfg.get("max_turns") or 6),
        disallowed_tools=_get_disallowed_tools(),
        allowed_tools=list(cfg.get("tools") or MF.DEFAULT_TOOLS),
        session_id=sid, turn_id=turn_id, surface="chat", priority=0,
        hooks=hooks, **{**hk, "tool_search_enabled": False})

    async def _pool(_o):
        return pool
    real_build = L._build_pool
    L._build_pool = _pool
    t0 = time.monotonic()
    stop, err = "", ""
    try:
        async for ev in L.run_query(msgs, options):
            if ev.get("type") == "result":
                stop = ev.get("stop_reason") or ""
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        L._build_pool = real_build
    wall = time.monotonic() - t0
    # The bookkeeping a finished flush leaves, so the turn-start record's
    # `flushed_before_summary` reads what production would.
    data = json.loads(path.read_text())
    data["compaction"] = {"flush": {
        "turn_id": turn_id, "at": time.time(), "status": "done" if not err else "failed",
        **MF.count_saves(e["tool"] for e in pool.saved)}}
    path.write_text(json.dumps(data))
    return {"turn_id": turn_id, "stop_reason": stop, "error": err,
            "wall_s": round(wall, 2), "saved": pool.saved,
            "tool_names": [n for n, _ in pool.calls],
            "planted_saved": flush_saved(pool.saved, session.planted)}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@contextmanager
def compaction_overlay(overlay: dict[str, Any]):
    """Swap the `compaction:` block in the live CONFIG dict for one arm."""
    from app.config import CONFIG
    saved = copy.deepcopy(CONFIG.get("compaction"))
    merged = copy.deepcopy(saved or {})
    for k, v in overlay.items():
        if isinstance(v, dict):
            merged[k] = {**(merged.get(k) or {}), **v}
        else:
            merged[k] = v
    CONFIG["compaction"] = merged
    try:
        yield merged
    finally:
        CONFIG["compaction"] = saved


def fired(record: dict[str, Any] | None) -> dict[str, Any]:
    """Tokens each mechanism freed, from the #1078 record."""
    record = record or {}
    ts = record.get("turn_start") or {}
    relief = record.get("relief") or []
    return {
        "mechanisms": list(record.get("mechanisms") or []),
        "turn_start_freed": int(ts.get("tokens_freed") or 0),
        "turn_start_mechanisms": list(ts.get("mechanisms") or []),
        "summarize_outcome": str(ts.get("summarize_outcome") or ""),
        "relief_freed": int(sum(int(r.get("freed_tokens") or 0) for r in relief)),
        "relief_passes": len(relief),
        "relief_rungs": sorted({str(g).split(":")[0] for r in relief
                                for g in (r.get("rungs") or [])}),
        "truncated_chars_freed": int(sum(int(r.get("truncated_chars_freed") or 0)
                                         for r in relief)),
    }


def summary_fired(f: dict[str, Any]) -> bool:
    """The summarize layer produced a summary on THIS turn.

    `summarized` alone is not enough: under `persist_summary` it is also true
    when a stored record was merely re-applied (`reused`), and the outcome is
    what says a summariser actually ran. A record without an outcome (older
    rows) falls back to the mechanism list."""
    if "summarize" not in f.get("turn_start_mechanisms", []):
        return False
    outcome = f.get("summarize_outcome")
    return outcome == "summarized" if outcome else True


def valid_for_arm(arm: str, f: dict[str, Any]) -> bool:
    freed = f["turn_start_freed"] + f["relief_freed"]
    if ARMS[arm].get("expects_summary") and not summary_fired(f):
        # A summary-format arm whose summarize layer did not replace a block
        # (under threshold, a summariser that failed and fell back to
        # truncation, or a re-applied record) measured neither format.
        return False
    rung = ARMS[arm].get("expects_rung")
    if rung and rung not in (f.get("relief_rungs") or []):
        # #1499's arms measure what one rung drops; a run where that rung
        # never fired measured something else.
        return False
    return freed > 0 if ARMS[arm]["expects_fire"] else freed == 0


_SUMMARY_PREFIX = "[compaction summary"


def summary_text(history: list[dict]) -> str:
    """The summary row's text in a compacted history, or ''."""
    from app.compaction import _message_text
    for m in history:
        if m.get("role") == "assistant":
            t = _message_text(m)
            if t.lstrip().startswith(_SUMMARY_PREFIX):
                return t
    return ""


def summary_has(text: str, planted: Planted) -> dict[str, bool]:
    """Which planted facts the summary text itself names — fidelity read off
    the summary, independent of what the model then answers. The old port is
    reported beside the new one: a summary that kept only the old port has
    kept the wrong answer."""
    def port(v: str) -> bool:
        return re.search(rf"(?<!\d){re.escape(v)}(?!\d)", text) is not None
    return {"codename": planted.passphrase.lower() in text.lower(),
            "port_now": port(planted.port), "port_old": port(planted.old_port)}


def fact_verbatim(history: list[dict], planted: Planted) -> bool:
    """Is the planted codename still in a history row other than the summary?
    (Then the row does not test the summary.)"""
    from app.compaction import _message_text
    for m in history:
        t = _message_text(m)
        if planted.passphrase in t and not t.lstrip().startswith(_SUMMARY_PREFIX):
            return True
    return False


@contextmanager
def summarizer_probe(dry: bool):
    """Count and time every summariser call the turn-start stack makes.

    Live, the real `summarize_history` / `summarize_incremental` run and are
    timed; with `dry`, they are replaced by a stub that returns a placeholder
    (which names no planted fact) so a dry run shows the layer firing without
    any engine traffic."""
    from app import compaction_llm as CL
    from app.compaction_llm import _format_history_for_summary
    calls: list[dict[str, Any]] = []
    real = {"summarize_history": CL.summarize_history,
            "summarize_incremental": CL.summarize_incremental}

    def wrap(name: str):
        async def fn(*args, **kw):
            rows = args[1] if name == "summarize_incremental" else args[0]
            rec: dict[str, Any] = {
                "fn": name, "rows": len(rows or []),
                "input_chars": len(_format_history_for_summary(rows or []))}
            t = time.monotonic()
            if dry:
                out = f"(dry-run stub summary of {len(rows or [])} rows)"
            else:
                out = await real[name](*args, **kw)
            rec.update(wall_s=round(time.monotonic() - t, 2),
                       output_chars=len(out or ""), failed=not out)
            calls.append(rec)
            return out
        return fn

    CL.summarize_history = wrap("summarize_history")
    CL.summarize_incremental = wrap("summarize_incremental")
    try:
        yield calls
    finally:
        CL.summarize_history = real["summarize_history"]
        CL.summarize_incremental = real["summarize_incremental"]


def summarizer_cost(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"calls": len(calls),
            "failed": sum(1 for c in calls if c.get("failed")),
            "wall_s": round(sum(float(c.get("wall_s") or 0) for c in calls), 2),
            "input_chars": sum(int(c.get("input_chars") or 0) for c in calls),
            "output_chars": sum(int(c.get("output_chars") or 0) for c in calls),
            "per_call": calls}


def _metrics(base_url: str) -> dict[str, float]:
    import httpx
    out: dict[str, float] = {}
    try:
        text = httpx.get(f"{base_url}/metrics", timeout=5).text
    except Exception:  # noqa: BLE001
        return out
    for name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                 "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total"):
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                out[name] = float(line.rsplit(" ", 1)[-1])
                break
    return out


class _Sampler:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.peak = 0.0
        self._task: asyncio.Task | None = None

    async def _run(self):
        while True:
            m = await asyncio.to_thread(_metrics, self.base_url)
            self.peak = max(self.peak, m.get("vllm:kv_cache_usage_perc", 0.0))
            await asyncio.sleep(0.5)

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


#: Characters of JSON per real token assumed when deciding whether a warm-up
#: fits. Conservative on purpose: tool output tokenizes at ~3.27 characters of
#: TEXT a token on the primary, and JSON escaping only adds characters, so
#: over-estimating here costs a clipped warm-up and under-estimating costs a
#: 400 — the failure this bound exists for.
WARMUP_JSON_CHARS_PER_TOKEN = 3.0
#: Room kept free under the window (the warm-up's one output token, template
#: tokens the JSON length does not see).
WARMUP_RESERVE_TOKENS = 4_096


def _json_tokens(obj: Any) -> int:
    return int(len(json.dumps(obj, default=str)) / WARMUP_JSON_CHARS_PER_TOKEN) + 1


def fit_warmup(system: list[dict], history: list[dict], tools: Any,
               window: int) -> tuple[list[dict] | None, dict[str, Any]]:
    """The warm-up prompt that fits `window`, and what was decided.

    Whole prompt if it fits; else the longest HEAD of whole turns (cut before a
    user message) that does — a prefix cache is built from the front, so the
    head is the part of the uncompacted prompt the probe can share, and a clip
    from the front would warm nothing. None when not even one turn fits.
    """
    budget = int(window) - WARMUP_RESERVE_TOKENS - _json_tokens(tools or []) \
        - sum(_json_tokens(m) for m in system)
    info: dict[str, Any] = {"messages_total": len(history), "window": int(window)}
    costs = [_json_tokens(m) for m in history]
    if sum(costs) <= budget:
        info.update(status="sent", messages_kept=len(history),
                    est_prompt_tokens=int(window) - budget + sum(costs)
                    - WARMUP_RESERVE_TOKENS)
        return system + history, info
    # `cut` is the last turn boundary reached within budget: a head that stops
    # mid-turn would end on an unanswered tool call.
    spent, cut = 0, 0
    for n, (m, c) in enumerate(zip(history, costs)):
        if m.get("role") == "user":
            cut = n                    # history[:n] is whole turns, and fits
        if spent + c > budget:
            break
        spent += c
    if cut <= 0:
        info.update(status="skipped", reason="no_whole_turn_fits", messages_kept=0)
        return None, info
    info.update(status="clipped", messages_kept=cut,
                est_prompt_tokens=int(window) - budget + sum(costs[:cut])
                - WARMUP_RESERVE_TOKENS)
    return system + history[:cut], info


def _make_stream_wrapper(real, warm_messages: list[dict] | None, log: list[dict],
                         on_warm_done: Callable[[], None], needle: str = "",
                         window: int = 262_144):
    """Time every request, and send the warm-up before iteration 1.

    The warm-up is attempted at most ONCE per run, whatever happens to it: it
    is decided before the first request (`state["warm"]`), so a loop that
    retries iteration 1 — stream retry, overflow recovery — never re-sends it,
    and its failure is recorded in the log instead of raised into the probe."""
    state = {"warm": warm_messages is None}

    def wrapper(**kwargs):
        async def gen():
            if kwargs.get("iteration") in (1, None) and not state["warm"]:
                state["warm"] = True
                msgs = kwargs["messages"]
                system = [msgs[0]] if msgs and msgs[0].get("role") == "system" else []
                warm, info = fit_warmup(system, warm_messages or [],
                                        kwargs.get("tools"), window)
                rec: dict[str, Any] = {"warmup": True, **info}
                if warm is not None:
                    wk = dict(kwargs, messages=warm,
                              extra_body={**(kwargs.get("extra_body") or {}),
                                          "max_tokens": 1})
                    t = time.monotonic()
                    wu: dict = {}
                    try:
                        async for ch in real(**wk):
                            if ch.get("usage"):
                                wu = ch["usage"]
                    except Exception as e:  # noqa: BLE001 — a warm-up never kills the probe
                        rec.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
                    rec.update(wall_s=time.monotonic() - t,
                               prompt_tokens=wu.get("prompt_tokens"),
                               cached_tokens=(wu.get("prompt_tokens_details") or {})
                               .get("cached_tokens"))
                log.append(rec)
                on_warm_done()
            rec: dict[str, Any] = {"iteration": kwargs.get("iteration"), "ttft_s": None}
            if needle:
                # Is the ORIGINAL planted result still intact on the wire —
                # not whether the fact is anywhere, since a re-read puts it
                # back in a fresh result.
                rec["planted_on_wire"] = any(
                    m.get("tool_call_id") == "call_planted"
                    and needle in json.dumps(m.get("content"))
                    for m in (kwargs.get("messages") or []))
            t0 = time.monotonic()
            async for ch in real(**kwargs):
                if rec["ttft_s"] is None:
                    for c in ch.get("choices") or []:
                        d = c.get("delta") or {}
                        if d.get("content") or d.get("reasoning") or \
                                d.get("reasoning_content") or d.get("tool_calls"):
                            rec["ttft_s"] = time.monotonic() - t0
                            break
                if ch.get("usage"):
                    u = ch["usage"]
                    rec["prompt_tokens"] = u.get("prompt_tokens")
                    rec["completion_tokens"] = u.get("completion_tokens")
                    rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}) \
                        .get("cached_tokens")
                yield ch
            rec["wall_s"] = time.monotonic() - t0
            log.append(rec)
        return gen()
    return wrapper


async def run_one(session: Session, arm: str, *, discovered: list, system_prompt: str,
                  data_root: Path, base_url: str, max_turns: int = 12,
                  judge=None, dry: bool = False) -> dict[str, Any]:
    """One (session, arm) row."""
    from app import compaction_record
    from app.compaction import estimate_conversation_tokens, load_and_compact_session
    from app.harness import loop as L
    from app.harness.context_meter import ContextMeter, context_window_for
    from app.harness.options import RunOptions
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
    from app.routers._messages_harness_adapter import _prepare_messages_for_harness

    spec = ARMS[arm]
    sid = f"pt-eval-c600-{session.key}-{arm}"
    turn_id = hashlib.sha1(sid.encode()).hexdigest()[:12]
    sessions_dir = data_root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{sid}.json"
    # indent=2 is how `sessions_io` writes a session: one row field per line,
    # so a Grep of the record (#1514's route) returns a message, not the file.
    path.write_text(json.dumps({"session_id": sid, "platform": "mission-control",
                                "messages": session.messages}, indent=2))
    row: dict[str, Any] = {"session": session.key, "arm": arm, "seed": session.seed,
                           "depth": session.depth, "target_tokens": session.target_tokens,
                           "est_tokens": session.est_tokens, "session_sha256": session.sha256}

    row["shape"] = session.meta.get("shape", "tool")
    with compaction_overlay(spec["compaction"]):
        if spec.get("flush"):
            fmsgs, finfo = await flush_history(
                session, sid=sid, data_root=data_root, system_prompt=system_prompt,
                discovered=discovered)
            row["flush"] = dict(finfo)
            if fmsgs is None:
                row.update(status="dropped", reason="flush_history_empty")
                return row
            if not dry:
                flush = await run_flush(
                    session, sid=sid, path=path, discovered=discovered,
                    system_prompt=system_prompt, data_root=data_root,
                    base_url=base_url, hk=_get_harness_kwargs(), messages=fmsgs)
                row["flush"].update({k: v for k, v in flush.items() if k != "saved"})
                row["flush"]["saved_count"] = len(flush["saved"])
                memory_block = render_saved_memory(flush["saved"])
                if memory_block:
                    system_prompt = f"{system_prompt}\n\n{memory_block}"
        t_ts = time.monotonic()
        with summarizer_probe(dry) as sum_calls:
            comp = await load_and_compact_session(path, model="primary")
        row["turn_start_wall_s"] = round(time.monotonic() - t_ts, 2)
        row["summarizer"] = summarizer_cost(sum_calls)
        stext = summary_text(comp.get("history") or [])
        row["summary_chars"] = len(stext)
        row["summary_has"] = summary_has(stext, session.planted) if stext else None
        row["fact_verbatim_at_start"] = fact_verbatim(comp.get("history") or [],
                                                      session.planted)
        turn = compaction_record.start_turn(sid, turn_id)
        turn.note_turn_start(comp)
        history = await _prepare_messages_for_harness(comp["history"], "primary")
        warm_hist = await _prepare_messages_for_harness(
            [m for m in session.messages], "primary")
        history.append({"role": "user", "content": session.probe})
        hk = _get_harness_kwargs()
        hk.update(spec["options"])
        row["planted_in_prompt_at_start"] = any(
            m.get("tool_call_id") == "call_planted" and session.planted.passphrase in
            (m.get("content") or "") for m in history)
        row["history_est_after_turn_start"] = estimate_conversation_tokens(history)
        pre = fired(turn.to_record())
        row["turn_start"] = {k: comp.get(k) for k in (
            "tokens_before", "tokens_after", "microcompacted", "summarized",
            "truncated", "summarize_outcome", "summary_reused", "summary_folds",
            "summary_covered_rows", "flushed_before_summary")}

        # Pre-gate: production can only fire in-turn if the prompt can reach
        # the in-turn trigger; the fixed overhead is measured on the first
        # probe we run, so this is only used when the turn-start pass
        # declined AND even a generous 80k overhead cannot reach it.
        from app.compaction import get_context_window, truncation_threshold
        thr = truncation_threshold(get_context_window("primary"))
        trig = int(thr * float(hk.get("intra_turn_microcompact_trigger_fraction", 0.8)))
        if spec["expects_fire"] and not spec.get("expects_summary") and \
                pre["turn_start_freed"] == 0 and session.est_tokens + 80_000 < trig:
            row.update(status="dropped", reason="cannot_fire", fired=pre)
            return row
        # The summary arms' gate is fully known here, so a row that cannot
        # measure a summary never spends the probe.
        if spec.get("expects_summary"):
            if not summary_fired(pre):
                row.update(status="dropped", fired=pre,
                           reason=f"summary_not_fired:{pre['summarize_outcome'] or 'none'}")
                return row
            if not stext:
                # Summarised, then the truncation fallback dropped the summary
                # row itself (restored files pushed the result back over).
                row.update(status="dropped", reason="summary_truncated_away", fired=pre)
                return row
            if row["fact_verbatim_at_start"]:
                # Folded past, or kept among the recent turns: the answer would
                # come from the verbatim row, not from the summary.
                row.update(status="dropped", reason="fact_not_summarized", fired=pre)
                return row
        window = context_window_for("primary")
        if dry:
            # What the warm-up would do, against the same tool schemas the
            # loop would send (the snapshot's, JSON-sized).
            _w, winfo = fit_warmup([{"role": "system", "content": system_prompt}],
                                   warm_hist, discovered, window)
            row.update(status="dry", fired=pre, warmup={"warmup": True, **winfo})
            return row

        pool = EvalPool(with_recall_observation(discovered), session.files,
                        data_root, planted=session.planted, session_id=sid)
        log: list[dict] = []
        before: dict[str, float] = {}
        sampler = _Sampler(base_url)

        def warm_done():
            before.update(_metrics(base_url))
            sampler.start()

        # The floor every production turn installs (Bash safety, and the #1136
        # outbound-content gate inside it). The stub pool can reach no sender,
        # but the turn is built from the production kwargs, so it is armed the
        # way a production turn is — `outbound_content.GATE_ARM_POINTS`.
        from app.harness import HookRegistry, install_default_safety_hook
        hooks = HookRegistry()
        install_default_safety_hook(hooks)
        options = RunOptions(
            model="primary", base_url=base_url, system_prompt=system_prompt,
            max_turns=max_turns, disallowed_tools=_get_disallowed_tools(),
            session_id=sid, turn_id=turn_id, surface="chat", priority=0,
            hooks=hooks, **hk)
        options.context_meter = ContextMeter(window)

        async def _pool(_o):
            return pool
        real_build, real_stream = L._build_pool, L.stream_chat
        real_truncate = L._truncate_largest_tool_results
        if spec.get("rung4_lossy"):
            # #1499's counterfactual: rung 4 as it was before 2026-09-11 — no
            # session id reaches it, so nothing is spilled and the notice can
            # only say "re-run the call with a narrower query".
            def _lossy(msgs, **kw):
                return real_truncate(msgs, **{**kw, "session_id": ""})
            L._truncate_largest_tool_results = _lossy
        L._build_pool = _pool
        L.stream_chat = _make_stream_wrapper(real_stream, warm_hist, log, warm_done,
                                             needle=session.planted.passphrase,
                                             window=window)
        t0 = time.monotonic()
        answer, stop, err = "", "", ""
        try:
            async for ev in L.run_query(history, options):
                if ev.get("type") == "result":
                    answer = ev.get("response_text") or ""
                    stop = ev.get("stop_reason") or ""
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        finally:
            L._build_pool, L.stream_chat = real_build, real_stream
            L._truncate_largest_tool_results = real_truncate
            await sampler.stop()
        wall = time.monotonic() - t0
        after = _metrics(base_url)

    rec = turn.to_record()
    f = fired(rec)
    iters = [x for x in log if not x.get("warmup")]
    warm = next((x for x in log if x.get("warmup")), {})
    hits = after.get("vllm:prefix_cache_hits_total", 0) - before.get("vllm:prefix_cache_hits_total", 0)
    qs = after.get("vllm:prefix_cache_queries_total", 0) - before.get("vllm:prefix_cache_queries_total", 0)
    pt = sum(int(x.get("prompt_tokens") or 0) for x in iters)
    ct = sum(int(x.get("cached_tokens") or 0) for x in iters)
    verdict = decide(answer, session.planted, judge)
    row.update(
        status="ok" if not err else "error", error=err, stop_reason=stop,
        answer=answer[-400:], verdict=verdict["final"], code_verdict=verdict["code"],
        judged=verdict["judged"], fired=f, compaction_record=rec,
        valid=(not err) and valid_for_arm(arm, f),
        tool_calls=len(pool.calls), tool_names=[n for n, _ in pool.calls],
        recovered_via_tool=sorted(pool.recovered), wall_s=round(wall, 2),
        recovered_via={k: sorted(v) for k, v in pool.recovered_via.items()},
        recall_calls=sum(1 for n, _ in pool.calls if n == "recall_observation"),
        record_calls=sum(1 for n, a in pool.calls
                         if pool._route(n, a) == "session_record"),
        spill_reads=sum(1 for n, a in pool.calls if pool._route(n, a) == "spill_file"),
        ttft_first_s=iters[0]["ttft_s"] if iters else None,
        ttft_s=[x.get("ttft_s") for x in iters],
        prompt_tokens=[x.get("prompt_tokens") for x in iters],
        cached_tokens=[x.get("cached_tokens") for x in iters],
        planted_on_wire=[x.get("planted_on_wire") for x in iters],
        run_cache_hit=(ct / pt) if pt else None,
        metrics_prefix_hit_delta=(hits / qs) if qs else None,
        peak_kv_usage=sampler.peak,
        preemptions_delta=after.get("vllm:num_preemptions_total", 0)
        - before.get("vllm:num_preemptions_total", 0),
        warmup=warm,
    )
    if not row["valid"] and not err:
        row["status"] = "dropped"
        row["reason"] = "arm_feature_not_exercised"
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per arm (and per arm x depth): recall for each fact separately."""
    from statistics import median
    try:
        from eval.stats import wilson_ci  # type: ignore
    except Exception:  # noqa: BLE001
        sys.path.insert(0, str(ROOT / "eval"))
        from stats import wilson_ci  # type: ignore

    def rate(rs: list[dict], fact: str) -> dict[str, Any]:
        k = sum(1 for r in rs if (r.get("verdict") or {}).get(fact) == "hit")
        n = len(rs)
        lo, hi = wilson_ci(k, n) if n else (None, None)
        und = sum(1 for r in rs if (r.get("verdict") or {}).get(fact) == "undecided")
        return {"k": k, "n": n, "rate": (k / n) if n else None,
                "ci95": [lo, hi], "undecided": und}

    def med(rs, key):
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return median(vals) if vals else None

    out: dict[str, Any] = {}
    for arm in ARMS:
        all_rows = [r for r in rows if r["arm"] == arm]
        kept = [r for r in all_rows if r.get("status") == "ok"]
        by_depth = {}
        for d in sorted({r["depth"] for r in kept}):
            rs = [r for r in kept if r["depth"] == d]
            by_depth[str(d)] = {"distinctive": rate(rs, "distinctive"),
                                "ambiguous": rate(rs, "ambiguous")}
        out[arm] = {
            "rows": len(all_rows), "kept": len(kept),
            "dropped": sum(1 for r in all_rows if r.get("status") == "dropped"),
            "errors": sum(1 for r in all_rows if r.get("status") == "error"),
            "distinctive": rate(kept, "distinctive"),
            "ambiguous": rate(kept, "ambiguous"),
            "by_depth": by_depth,
            "median_ttft_first_s": med(kept, "ttft_first_s"),
            "median_wall_s": med(kept, "wall_s"),
            "median_run_cache_hit": med(kept, "run_cache_hit"),
            "median_peak_kv": med(kept, "peak_kv_usage"),
            "mean_tool_calls": (sum(r["tool_calls"] for r in kept) / len(kept)) if kept else None,
            "recovered_via_tool": {f: sum(1 for r in kept if f in (r.get("recovered_via_tool") or []))
                                   for f in ("distinctive", "ambiguous")},
            "planted_in_prompt_at_start": sum(1 for r in kept if r.get("planted_in_prompt_at_start")),
            # P3: which planted facts the flush turn wrote down (flush arm only).
            "flush_saw_planted": sum(1 for r in kept if (r.get("flush") or {})
                                     .get("planted_in_history")),
            "flush_saved": {f: sum(1 for r in kept if ((r.get("flush") or {})
                                                       .get("planted_saved") or {}).get(f))
                            for f in ("distinctive", "ambiguous")},
            "preemptions_delta": sum(r.get("preemptions_delta") or 0 for r in kept),
            # #1514 / #1481: which route the facts came back through, and how
            # often each route was used, per turn.
            "mean_recall_calls": (sum(r.get("recall_calls") or 0 for r in kept)
                                  / len(kept)) if kept else None,
            "mean_record_calls": (sum(r.get("record_calls") or 0 for r in kept)
                                  / len(kept)) if kept else None,
            "mean_spill_reads": (sum(r.get("spill_reads") or 0 for r in kept)
                                 / len(kept)) if kept else None,
            "recovered_via": {
                route: {f: sum(1 for r in kept
                               if f in ((r.get("recovered_via") or {}).get(route) or []))
                        for f in ("distinctive", "ambiguous")}
                for route in sorted({k for r in kept for k in (r.get("recovered_via") or {})})},
            "planted_on_wire_at_answer": sum(
                1 for r in kept if (r.get("planted_on_wire") or [None])[-1]),
            "drop_reasons": _count(r.get("reason") for r in all_rows
                                   if r.get("status") == "dropped"),
            "warmup": _count((r.get("warmup") or {}).get("status") for r in all_rows
                             if r.get("warmup")),
            # What the turn-start stack cost, and what the summary itself kept
            # (read off its text, before the model answers anything).
            "median_turn_start_wall_s": med(kept, "turn_start_wall_s"),
            "median_summarizer_calls": med(
                [{"v": (r.get("summarizer") or {}).get("calls")} for r in kept], "v"),
            "median_summarizer_wall_s": med(
                [{"v": (r.get("summarizer") or {}).get("wall_s")} for r in kept], "v"),
            "median_summary_chars": med(kept, "summary_chars"),
            "summary_has": {f: sum(1 for r in kept if (r.get("summary_has") or {}).get(f))
                            for f in ("codename", "port_now", "port_old")},
        }
    return out


def _count(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        if v is not None:
            out[str(v)] = out.get(str(v), 0) + 1
    return out


def paired(rows: list[dict[str, Any]], base: str = "production") -> dict[str, Any]:
    """Every other arm against `base`, over the sessions both kept.

    Same sessions, same engine state, so the paired bootstrap applies; a
    positive `diff` means the arm is higher than `base`. Recall is compared
    per fact as a 0/1 hit, never blended.
    """
    try:
        from eval.stats import paired_bootstrap_ci  # type: ignore
    except Exception:  # noqa: BLE001
        sys.path.insert(0, str(ROOT / "eval"))
        from stats import paired_bootstrap_ci  # type: ignore

    ok = {(r["session"], r["arm"]): r for r in rows if r.get("status") == "ok"}
    metrics: dict[str, Callable[[dict], float]] = {
        "distinctive_hit": lambda r: float((r.get("verdict") or {}).get("distinctive") == "hit"),
        "ambiguous_hit": lambda r: float((r.get("verdict") or {}).get("ambiguous") == "hit"),
        "wall_s": lambda r: float(r.get("wall_s") or 0.0),
        "ttft_total_s": lambda r: float(sum(x for x in (r.get("ttft_s") or []) if x)),
        "tool_calls": lambda r: float(r.get("tool_calls") or 0),
        "ttft_first_s": lambda r: float(r.get("ttft_first_s") or 0.0),
        "turn_start_wall_s": lambda r: float(r.get("turn_start_wall_s") or 0.0),
        "summary_codename": lambda r: float(bool((r.get("summary_has") or {}).get("codename"))),
        "summary_port_now": lambda r: float(bool((r.get("summary_has") or {}).get("port_now"))),
    }
    out: dict[str, Any] = {}
    for arm in ARMS:
        if arm == base:
            continue
        keys = sorted(s for (s, a) in ok if a == arm and (s, base) in ok)
        if len(keys) < 2:
            continue
        res: dict[str, Any] = {"n": len(keys)}
        for name, fn in metrics.items():
            a = [fn(ok[(s, base)]) for s in keys]
            b = [fn(ok[(s, arm)]) for s in keys]
            if a == b:
                res[name] = {"diff": 0.0, "identical": True}
                continue
            ci = paired_bootstrap_ci(a, b)
            res[name] = {k: ci[k] for k in ("diff", "lo", "hi", "significant")}
        out[arm] = res
    return out


def resolve_shape(shape: str, arms: list[str]) -> str:
    """`auto` is `conversation` when every arm is a summary arm (the only
    shape on which their summary fires), else the #600 `tool` shape."""
    if shape != "auto":
        return shape
    return "conversation" if arms and all(ARMS[x].get("expects_summary")
                                          for x in arms) else "tool"


def with_recall_observation(discovered: list) -> list:
    """`discovered` plus `recall_observation` (#1481) on the lloyd-mcp server.

    A snapshot taken from an aggregator older than the tool lacks it. Added
    to every arm: the harness itself hides it from a turn whose relief writes
    no observation stubs (`loop._open_turn`), so an arm with the switch off
    advertises exactly what it did before — which exercises the real hiding.
    """
    from agent_mcp.recall_observation import tool
    from app.harness.tool_result_spill import RECALL_OBSERVATION_TOOL

    out = []
    added = False
    for srv, tools in discovered:
        tools = list(tools)
        if not added and any(t.get("name") == "Read" for t in tools):
            if not any(t.get("name") == RECALL_OBSERVATION_TOOL for t in tools):
                t = tool()
                tools.append({"name": t.name, "description": t.description,
                              "inputSchema": t.model_dump(by_alias=True)["inputSchema"],
                              "annotations": {"readOnlyHint": True,
                                              "idempotentHint": True}})
            added = True
        out.append([srv, tools])
    return out


def _load_discovered(snapshot: Path) -> list:
    if snapshot.exists():
        return json.loads(snapshot.read_text())
    from app.harness.loop import DEFAULT_LLOYD_MCP_SERVERS
    from app.harness.mcp_pool import MCPPool

    async def disc():
        p = MCPPool(DEFAULT_LLOYD_MCP_SERVERS)
        await p.open()
        d = p.discovered
        await p.aclose()
        return d
    d = asyncio.run(disc())
    snapshot.write_text(json.dumps(d))
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=6,
                    help="sessions per (size) stratum; depths cycle")
    ap.add_argument("--sizes", default="90000,130000",
                    help="history sizes in estimator tokens, comma-separated")
    ap.add_argument("--depths", default="0.1,0.5,0.85")
    ap.add_argument("--arms", default="none,production,tool_clear,raised,trigger90")
    ap.add_argument("--seed", type=int, default=600)
    ap.add_argument("--probe", choices=("early", "late"), default="late",
                    help="late: the question arrives in a tool result, after "
                         "the in-turn pass has had its chance to clear")
    ap.add_argument("--shape", choices=("auto",) + SHAPES, default="auto",
                    help="session shape; auto = conversation when every arm is a "
                         "summary arm, else tool (see SUMMARY ARMS)")
    ap.add_argument("--data-root", required=True,
                    help="scratch LLOYD_DATA root (sessions, spills, event logs)")
    ap.add_argument("--tools-snapshot", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8096")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true",
                    help="generate and compact only; no engine")
    a = ap.parse_args(argv)

    data_root = Path(a.data_root)
    if os.environ.get("LLOYD_DATA") != str(data_root):
        print("refusing: run with LLOYD_DATA set to --data-root (scratch), "
              "so spills and event logs stay out of production", file=sys.stderr)
        return 2
    from prompt_builder import build_system_prompt
    system_prompt = build_system_prompt()
    discovered = _load_discovered(Path(a.tools_snapshot))
    sizes = [int(x) for x in a.sizes.split(",")]
    depths = [float(x) for x in a.depths.split(",")]
    arms = [x for x in a.arms.split(",") if x]
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        print(f"unknown arm(s): {unknown}", file=sys.stderr)
        return 2
    shape = resolve_shape(a.shape, arms)
    corpus = _corpus_files()

    sessions = [build_session(a.seed + 1000 * si + i, size, depths[i % len(depths)],
                              corpus=corpus, probe=a.probe, shape=shape)
                for si, size in enumerate(sizes) for i in range(a.sessions)]
    out_path = Path(a.out)
    rows: list[dict] = []
    if out_path.exists():   # resume: keep finished rows
        rows = json.loads(out_path.read_text()).get("rows", [])
    done = {(r["session"], r["arm"]) for r in rows}

    async def go():
        for n, s in enumerate(sessions):
            order = arms[n % len(arms):] + arms[:n % len(arms)]   # rotate
            for arm in order:
                if (s.key, arm) in done:
                    continue
                r = await run_one(s, arm, discovered=discovered,
                                  system_prompt=system_prompt, data_root=data_root,
                                  base_url=a.base_url, dry=a.dry)
                rows.append(r)
                print(json.dumps({k: r.get(k) for k in (
                    "session", "arm", "status", "reason", "verdict", "tool_calls",
                    "ttft_first_s", "wall_s", "run_cache_hit", "summary_has")}
                    | {"summarize_outcome": (r.get("fired") or {}).get("summarize_outcome"),
                       "summarizer_calls": (r.get("summarizer") or {}).get("calls"),
                       "warmup": (r.get("warmup") or {}).get("status")},
                    default=str), flush=True)
                _write(out_path, a, sessions, system_prompt, rows)
    asyncio.run(go())
    _write(out_path, a, sessions, system_prompt, rows)
    print(json.dumps(summarize(rows), indent=1, default=str))
    return 0


def _write(out_path: Path, a, sessions, system_prompt, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "item": 600,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": {k: v for k, v in vars(a).items() if k not in ("data_root",)},
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "sessions": [{"key": s.key, "seed": s.seed, "target_tokens": s.target_tokens,
                      "shape": s.meta.get("shape"),
                      "depth": s.depth, "est_tokens": s.est_tokens, "sha256": s.sha256,
                      "planted": {"passphrase": s.planted.passphrase,
                                  "port": s.planted.port,
                                  "old_port": s.planted.old_port}} for s in sessions],
        "summary": summarize(rows),
        # The in-turn trigger is a prompt size, so the arms separate by size:
        # below it some arms are identical by construction.
        "summary_by_size": {
            str(t): summarize([r for r in rows if r.get("target_tokens") == t])
            for t in sorted({r.get("target_tokens") for r in rows if r.get("target_tokens")})},
        "paired_vs_production": paired(rows),
        "paired_vs_production_by_size": {
            str(t): paired([r for r in rows if r.get("target_tokens") == t])
            for t in sorted({r.get("target_tokens") for r in rows if r.get("target_tokens")})},
        # The D2 pair: persisted against legacy, over the sessions both kept.
        "paired_vs_summary_legacy": paired(rows, base="summary_legacy"),
        # #1514 / #1481: the free route against the arm it amends, and the new
        # tool against the free route it has to beat.
        "paired_vs_tool_clear": paired(rows, base="tool_clear"),
        "paired_vs_self_record": paired(rows, base="self_record"),
        "rows": rows,
    }
    out_path.write_text(json.dumps(doc, indent=1, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
