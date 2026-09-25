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
the spill files microcompaction wrote; Grep searches the same; everything
else answers "not available in this evaluation". No tool can change the
machine, so no sandboxed session is needed, and the ids are `pt-eval-*`
anyway.

Before the probe's first request the runner sends the UNCOMPACTED prompt
with `max_tokens: 1` (the warm-up): that is the prefix cache a session in
production arrives at this turn with — its previous turn sent everything
but the new user message. The probe's TTFT and cached-token numbers are
therefore what a rewrite costs against a warm session, not against a cold
engine.

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
  raised      trigger/target 0.9/0.7 at both passes: fire later, keep more.
  trigger90   trigger 0.9, target 0.52: fire later, clear as far as today.
  summary_legacy     summarize layer, regenerate-every-turn 9-section summary.
  summary_persisted  summarize layer with `persist_summary: true` (D2): the
              incremental Goal/Constraints/Progress/Decisions/Next steps
              record. Both need sizes past the truncation threshold; valid
              only when the summarize layer replaced a block.

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
    # regenerate-every-turn summary. The layer runs only past the truncation
    # threshold (~210k on the primary), so run these at sizes above it:
    #   --arms summary_legacy,summary_persisted --sizes 240000,280000
    # This pair gates flipping `compaction.persist_summary` on.
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


def _planted_turn(key: str, planted: Planted, rng: random.Random) -> tuple[list[dict], str]:
    path = NOTES_PATH.format(key=key)
    body = _notes_body(planted, rng)
    cid = "call_planted"
    return [
        _msg("user", f"Pull up the deploy notes at {path} — I want them in "
                     "front of us before we touch the relays."),
        {"role": "assistant", "content": [{"type": "text", "text": ""}],
         "tool_calls": [_tc(cid, "Read", {"summary": "Reading the deploy notes",
                                          "file_path": path})]},
        _msg("tool", _read_render(body), tool_call_id=cid),
        _msg("assistant", "Got the deploy notes open. Nothing in them blocks "
                          "what we're doing; carrying on with the review."),
    ], body


def build_session(seed: int, target_tokens: int, depth: float,
                  *, root: Path = ROOT, corpus: list[Path] | None = None,
                  probe: str = "early") -> Session:
    """One synthetic session: filler to `target_tokens` (estimator units, the
    ones every compaction trigger is written in) with the planted turn at
    `depth` of the way through it. Deterministic in (seed, tree)."""
    from app.compaction import estimate_conversation_tokens

    rng = random.Random(seed)
    corpus = corpus if corpus is not None else _corpus_files(root)
    if not corpus:
        raise RuntimeError("no filler corpus under the tree")
    key = f"s{seed}-{target_tokens // 1000}k-d{int(depth * 100)}"
    planted = make_planted(rng)
    planted_msgs, notes = _planted_turn(key, planted, rng)
    budget = target_tokens - estimate_conversation_tokens(planted_msgs)
    turns: list[list[dict]] = []
    total = 0
    i = 0
    while total < budget:
        t = _filler_turn(i, rng, corpus, root, planted, salt=(i % 4 == 0))
        total += estimate_conversation_tokens(t)
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
                est_tokens=estimate_conversation_tokens(messages))
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
    """Real tool schemas, stub handlers. Nothing here can change the machine."""

    def __init__(self, discovered: list, files: dict[str, str],
                 spill_root: Path, tree_root: Path = ROOT,
                 planted: Planted | None = None) -> None:
        self._discovered = discovered
        self.files = files
        self.spill_root = spill_root
        self.tree_root = tree_root
        self.planted = planted
        self.calls: list[tuple[str, dict]] = []
        # Which planted facts a tool result handed back to the model: the
        # re-retrieval the video says compaction forces.
        self.recovered: set[str] = set()

    async def call_tool(self, name: str, args: dict, **kw) -> dict:
        out = await self._call(name, args, **kw)
        p = self.planted
        if p is not None and not out.get("is_error"):
            text = str(out.get("content") or "")
            if p.passphrase in text:
                self.recovered.add("distinctive")
            if f"listens on port {p.port} now" in text:
                self.recovered.add("ambiguous")
        return out

    @property
    def discovered(self):
        return self._discovered

    def _lookup(self, path: str) -> str | None:
        if path in self.files:
            return self.files[path]
        for k, v in self.files.items():
            if path.endswith(k):
                return v
        p = Path(path).expanduser()
        for base in (self.spill_root, self.tree_root):
            try:
                rp = p.resolve()
                if rp.is_file() and rp.is_relative_to(base.resolve()):
                    return rp.read_text(errors="replace")
            except (OSError, ValueError):
                continue
        return None

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
            pool = dict(self.files)
            for sp in sorted(self.spill_root.rglob("*")) if self.spill_root.exists() else []:
                if sp.is_file():
                    pool[str(sp)] = sp.read_text(errors="replace")
            for k, v in pool.items():
                hits += [f"{k}:{n + 1}:{ln}" for n, ln in enumerate(v.splitlines())
                         if rx.search(ln)]
            return {"content": "\n".join(hits[:200]) or "No matches found",
                    "is_error": False}
        return {"content": f"{bare} is not available in this evaluation; "
                           "use Read or Grep.", "is_error": True}


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
        "relief_freed": int(sum(int(r.get("freed_tokens") or 0) for r in relief)),
        "relief_passes": len(relief),
    }


def valid_for_arm(arm: str, f: dict[str, Any]) -> bool:
    freed = f["turn_start_freed"] + f["relief_freed"]
    if ARMS[arm].get("expects_summary") and \
            "summarize" not in f.get("turn_start_mechanisms", []):
        # A summary-format arm whose summarize layer did not replace a block
        # measured the truncation fallback, not the format.
        return False
    return freed > 0 if ARMS[arm]["expects_fire"] else freed == 0


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


def _make_stream_wrapper(real, warm_messages: list[dict] | None, log: list[dict],
                         on_warm_done: Callable[[], None], needle: str = ""):
    """Time every request, and send the warm-up before iteration 1."""
    def wrapper(**kwargs):
        async def gen():
            if kwargs.get("iteration") in (1, None) and warm_messages is not None and not log:
                msgs = kwargs["messages"]
                warm = ([msgs[0]] if msgs and msgs[0].get("role") == "system" else []) \
                    + warm_messages
                wk = dict(kwargs, messages=warm,
                          extra_body={**(kwargs.get("extra_body") or {}), "max_tokens": 1})
                t = time.monotonic()
                wu = {}
                async for ch in real(**wk):
                    if ch.get("usage"):
                        wu = ch["usage"]
                log.append({"warmup": True, "wall_s": time.monotonic() - t,
                            "prompt_tokens": wu.get("prompt_tokens"),
                            "cached_tokens": (wu.get("prompt_tokens_details") or {})
                            .get("cached_tokens")})
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
    path.write_text(json.dumps({"session_id": sid, "platform": "mission-control",
                                "messages": session.messages}))
    row: dict[str, Any] = {"session": session.key, "arm": arm, "seed": session.seed,
                           "depth": session.depth, "target_tokens": session.target_tokens,
                           "est_tokens": session.est_tokens, "session_sha256": session.sha256}

    with compaction_overlay(spec["compaction"]):
        comp = await load_and_compact_session(path, model="primary")
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
            "summary_covered_rows")}

        # Pre-gate: production can only fire in-turn if the prompt can reach
        # the in-turn trigger; the fixed overhead is measured on the first
        # probe we run, so this is only used when the turn-start pass
        # declined AND even a generous 80k overhead cannot reach it.
        from app.compaction import get_context_window, truncation_threshold
        thr = truncation_threshold(get_context_window("primary"))
        trig = int(thr * float(hk.get("intra_turn_microcompact_trigger_fraction", 0.8)))
        if spec["expects_fire"] and pre["turn_start_freed"] == 0 and \
                session.est_tokens + 80_000 < trig:
            row.update(status="dropped", reason="cannot_fire", fired=pre)
            return row
        if dry:
            row.update(status="dry", fired=pre)
            return row

        pool = EvalPool(discovered, session.files, data_root,
                        planted=session.planted)
        log: list[dict] = []
        before: dict[str, float] = {}
        sampler = _Sampler(base_url)

        def warm_done():
            before.update(_metrics(base_url))
            sampler.start()

        options = RunOptions(
            model="primary", base_url=base_url, system_prompt=system_prompt,
            max_turns=max_turns, disallowed_tools=_get_disallowed_tools(),
            session_id=sid, turn_id=turn_id, surface="chat", priority=0, **hk)
        options.context_meter = ContextMeter(context_window_for("primary"))

        async def _pool(_o):
            return pool
        real_build, real_stream = L._build_pool, L.stream_chat
        L._build_pool = _pool
        L.stream_chat = _make_stream_wrapper(real_stream, warm_hist, log, warm_done,
                                             needle=session.planted.passphrase)
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
            "preemptions_delta": sum(r.get("preemptions_delta") or 0 for r in kept),
        }
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
    corpus = _corpus_files()

    sessions = [build_session(a.seed + 1000 * si + i, size, depths[i % len(depths)],
                              corpus=corpus, probe=a.probe)
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
                    "session", "arm", "status", "verdict", "tool_calls",
                    "ttft_first_s", "wall_s", "run_cache_hit")}, default=str), flush=True)
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
        "rows": rows,
    }
    out_path.write_text(json.dumps(doc, indent=1, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
