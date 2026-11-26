#!/usr/bin/env python3
"""Score a tool-observation compressor by the decision it still permits (#3850).

    # harvest the pinned corpus (needs the real sessions dir; the frontier run's job)
    .venvs/lloyd/bin/python eval/run_observation_compression_eval.py harvest \
        --min-bytes 8000 --out eval/measurements/observation-compression-corpus.yaml

    # score one arm of every row against a live engine
    .venvs/lloyd/bin/python eval/run_observation_compression_eval.py score \
        --corpus eval/measurements/observation-compression-corpus.yaml \
        --compressor rung4_truncate --budget 0.25

    # publish the note from that run record (#4508: the frontier verdict, the row
    # roster, the span tally and the harvest skips, as tracked repo content)
    .venvs/lloyd/bin/python eval/run_observation_compression_eval.py report \
        --run <run.jsonl> --corpus <corpus.yaml> --skips <skips.json> --floor <floor>

`--out` names a path under `eval/measurements/`, not the `eval/baselines/` the first
drafts used: root `.gitignore:141` ignores `eval/baselines/*` with only two named
negations beside it, so a corpus written there can never be committed — and an
unwitnessed corpus is the defect family this scorer keeps tripping over.
`eval/measurements/` is not ignored: `git check-ignore -v` exits 1 for a path there and
0 for one under `eval/baselines/`.

Why this exists
---------------
Coca-Cola (arXiv:2611.06500) trains its observation compressor with
Action-Preserving Training: run the frozen agent on the full report and on the
compressed one, and minimise the KL between the two NEXT-ACTION distributions
plus cross-entropy on the action the trajectory actually recorded. The
observation is graded by the decision it still permits, never by similarity to
the prose it replaced. The latent channel is out of scope here — it needs an
embedding-input serving path and back-prop through the frozen agent — but the
SCORING FUNCTION is computable on the engine already running, with teacher-forced
`prompt_logprobs` (the readout #540 established and #638's guidance pins: ONE
logprob client, not a forked second one).

What it prices that nothing else does
-------------------------------------
Lloyd has a compaction path that discards tool content and a way to price
compression by recall, and has never joined them. Standing problem #2 in
`eval/lloyd_profile.md` is that compaction is truncation-shaped, and the rung it
runs is live: 11,053 `truncate:*` firings in 1,463 usage rows of
`usage.compaction`, last on 2026-11-06, written by `_record_relief_pass`. #600
(`eval/run_compaction_recall_eval.py`) measures whether a planted STRING survives
compaction — 35-40 recall points for blunt `tool_clear`, nothing measurable for
spill-and-clear — but recall of a planted string is not the thing Coca-Cola
optimises. Nothing here measured whether the agent, handed the compressed
observation, still reaches for the SAME next action. That number is the decision
procedure for every compression policy the relief ladder will ever run: today the
choice between truncating, summarising and spilling is intuition plus a 20-row
probe.

Reads, and what each one is for
-------------------------------
Two independent reads per (row, compressor, budget), never merged into one score:

* `delta_ce_nats` and `kl_first_position` — the paper's objective. Both prompts
  (full observation, candidate replacement) are teacher-forced THROUGH the
  recorded action, and the comparison is between the two distributions at the
  first ACTION token position: that position is each arm's first generated token,
  so its top-k is that arm's next-action distribution. ΔCE is the mean-nll
  difference over the action span in nats (positive = the action is harder to
  produce given the compression), taken from a `prompt_logprobs: 0` readout in
  which the observed token is the single entry its position carries (#4499), so the
  nll is the RECORDED ACTION's own and not its position's argmax candidate's. That
  shape carries no candidates, which is why `kl_first_position` reports None on
  every row of it rather than borrowing a distribution from another request.
* `tier` — `eval/action_equivalence.py`'s tiers over the two ARMS' generated
  calls (one step at temperature 0 under each arm), which is the version a human
  can audit. `agrees` is true only at `canonically_equal`.

Both readings are taken over the ACTION's positions in the prompt, not over a window
sized by the action's character count. `prompt_logprobs` has one entry per token and
the action is a string, at roughly four characters a token; a character-sized window
is about four times too wide and opens inside the tool observation, which is the one
region the two arms deliberately differ in — the ΔCE would then be a similarity
between the two texts, and the KL would be read at a position inside the observation
while keeping the name "first generated position". Where the readout labels its own
tokens the span is aligned exactly (`span_alignment: token_text`); where it does not,
the row is scored on the estimate and says so (`char_estimate`), because the two
numbers are not equally trustworthy and a report that mixes them silently is the same
category of error as scoring a missing logprob as zero cost.

The budget axis is matched by CHARS (`int(len(observation) * budget)`), because the
baseline arm is production's rung 4, which decides in chars, and the extractive arm is
charged for its joiners so that all three arms stand at the same point on that axis.
`tokens_retained` is a CHARACTER count — each row carries `retained_unit: chars`, and
the report prints the unit — reported per arm as the measured value, not the asked-for
one: rung 4 replaces the whole result with one fixed-size notice, so at any budget
above the notice's size arm (a) retains the notice and nothing more. That is a property
of the baseline, not a bug here, and the report reads it off the table.

What is adjacent, and is not this
---------------------------------
* `eval/engine_output_probe.py` scores prose-token agreement between two runs of
  the SAME prompt (`parse_completion` :293 reads SAMPLED-token logprobs, and
  `compare_prompt` :403 compares `agreement` / `first_divergence`). Different
  quantity: this module reads PROMPT-token logprobs, over a span, of two
  different prompts.
* `eval/run_compaction_recall_eval.py` (#600) owns the planted-fact probe and its
  corpus, and wraps `_truncate_largest_tool_results` at :2047 to build a LOSSY
  arm. That is a recall instrument; it is also the regression gate this one
  informs.
* `eval/tool_call_equivalence_probe.py` and `eval/decision_replay_588.py` replay
  recorded calls to ask whether the engine still issues them; neither pairs a
  full observation with a compressed one.
* `eval/run_harness_replay_diff.py` diffs recorded transcripts by hashing
  `tool_calls` — a hash has no canonicalisation, which is why #2843 grew
  `action_equivalence` instead.

Refusals, not zeros
-------------------
An answer that carries no `prompt_logprobs` on `choices[0]`, or scores zero
positions, makes the row INVALID and EXCLUDES it from every aggregate. It can
never read as `delta_ce_nats = 0.0` or as agreement: an absent logprob and a
logprob of 0.0 are different claims, and reading one as the other is how a silent
empty parse becomes a clean bill of health.

An arm that generates no tool call at all is recorded as a DECLINE
(`decline = compressed_arm_declined`) beside the tier, with `tier = None`. It is
never folded into `protocol_invalid`, because a compressor that suppresses the
call entirely must not be counted as agreeing with a full arm that called — and
`protocol_invalid` means "both sides are unreadable", which is a different
finding with a different fix.

LongLLMLingua, arm (d) of the item, is NOT implemented: it is a pip dependency
this box does not have, and the item says to skip it and say so rather than
block the run. `summariser` (arm c) is an injected callable for the same reason —
the frontier run wires it to the llama.cpp 35B-A3B; a unit test injects a stub,
and this module never opens a socket by itself.

"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
for _p in (str(_REPO_ROOT), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from action_equivalence import (  # noqa: E402
    CANONICALLY_EQUAL,
    PROTOCOL_INVALID,
    ToolCall,
    classify,
)

# ---------------------------------------------------------------------------
# Corpus harvesting
# ---------------------------------------------------------------------------

#: The item's size band, in bytes. ">=2k tokens" is ~8,000 chars at this box's
#: own 4-chars-per-token heuristic (`app.compaction.TOKENS_PER_CHAR`), and the
#: same number is what the triage counted sidecars over, so the band and the
#: availability evidence are on one unit.
MIN_OBSERVATION_BYTES = 8_000

#: Budgets are a fraction of the observation, matched per arm by chars.
BUDGETS: tuple[float, ...] = (0.10, 0.25, 0.50)

CORPUS_VERSION = 1


def _default_sessions_dir() -> Path:
    """The live sessions root, resolved at CALL time.

    Not a module-level constant: a test that passes a fixture directory must not
    be one attribute-assignment away from writing into the real one, and
    `app.paths.SESSIONS_DIR` is itself overridable per-process.
    """
    from app.paths import SESSIONS_DIR

    return Path(SESSIONS_DIR)


def sidecar_path(spill_dir: Path, tool_call_id: str) -> Path | None:
    """The sidecar holding this call's observation VERBATIM, or None.

    `<sid>.tool-results/<tool_call_id>.{txt,json}` — the capture-time spill route
    (`app/harness/tool_result_spill._spill_path`), which writes the original
    string and picks the extension by `_looks_like_json`. `.txt` is tried first
    because it is the more common of the two on this box.
    """
    for ext in ("txt", "json"):
        p = spill_dir / f"{tool_call_id}.{ext}"
        if p.is_file():
            return p
    return None


def _resolve_sidecar(spill_dir: Path, row: dict) -> Path | None:
    """The sidecar for one recorded tool row.

    The clause shape above is the only route used: a row whose observation was
    never spilled has no verbatim file, and re-reading the session record's
    `content` instead would silently swap a 2 KB `<persisted-output>` pointer in
    for the observation we are supposed to compress.
    """
    cid = row.get("tool_call_id")
    if not isinstance(cid, str) or not cid:
        return None
    return sidecar_path(spill_dir, _safe_spill_id(cid))


def _safe_spill_id(tool_call_id: str) -> str:
    """The same sanitising `_spill_path` applies when it wrote the file."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_call_id)[:128]
    return safe or "unknown"


def _tool_name_for(messages: Sequence[dict], index: int, tool_call_id: str) -> str | None:
    """The name of the call this tool row answers: the nearest PRECEDING assistant
    row whose `tool_calls` carries this id.

    Preceding, not following: the assistant turn that asks for the observation is
    the one naming the tool, and a same-step batch of several calls would
    otherwise get the first name in the list rather than the one that made this
    result.
    """
    for j in range(index - 1, -1, -1):
        m = messages[j]
        if m.get("role") != "assistant":
            continue
        for call in (m.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            if call.get("id") == tool_call_id or call.get("call_id") == tool_call_id:
                fn = call.get("function") or {}
                name = fn.get("name")
                return name if isinstance(name, str) and name.strip() else None
        # A closer assistant row with no such id means the call was written by a
        # path that did not record it; keep walking and say so by returning None.
        return None
    return None


def _is_history_row(m: Any) -> bool:
    """A row the turn-start stack keeps — production's own filter.

    `app.compaction.is_history_row` drops `subliminal` and `thinking` rows and
    every memory-flush row, which is exactly the projection that decides what the
    model actually saw. Import it rather than restating the role list, or this
    eval's prefix drifts from production's history on the first new row type.
    Import failure is a hard error, never a permissive fallback: a filter that
    fails open here silently puts UI-only rows into the scored prefix.
    """
    from app.compaction import is_history_row

    return bool(is_history_row(m))


def _chat_row(m: dict) -> dict:
    """One stored row as the chat message the engine is served.

    Stripped to the fields the render uses: `timestamp`, `stats` and `turn_id`
    are bookkeeping the model never sees, and carrying them into a pinned corpus
    file would make a re-render depend on when the session was written.
    """
    role = m.get("role")
    content = m.get("content")
    if content is None:
        content = ""
    out: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        calls = [c for c in (m.get("tool_calls") or []) if isinstance(c, dict)]
        if calls:
            out["tool_calls"] = [
                {
                    "id": c.get("id") or c.get("call_id") or "",
                    "type": c.get("type") or "function",
                    "function": {
                        "name": (c.get("function") or {}).get("name"),
                        "arguments": (c.get("function") or {}).get("arguments"),
                    },
                }
                for c in calls
            ]
    elif role == "tool":
        out["tool_call_id"] = m.get("tool_call_id") or ""
    return out


def _next_action(messages: Sequence[dict], index: int) -> dict | None:
    """The recorded decision that follows this observation.

    The first history row after `index` that carries a decision: an assistant row
    with a tool call (`{"kind": "tool_call", "calls": [...]}`), else an assistant
    row with non-empty prose (`{"kind": "answer", ...}`) — the item's other
    decision-bearing arm, and the reason a compressor cannot win by deleting the
    tool menu. An assistant row with neither is a step that produced nothing; it
    is not evidence of what the model did next, so the walk continues past it.
    A `user`/`tool` row ends the walk: the step we are scoring is over.
    `content` on either arm is text, never the block list a real record may store.
    """
    for j in range(index + 1, len(messages)):
        m = messages[j]
        if not _is_history_row(m) or m.get("role") != "assistant":
            if m.get("role") in ("user", "tool", "system"):
                return None
            continue
        calls = [c for c in (m.get("tool_calls") or []) if isinstance(c, dict)]
        # `content` is flattened through `_render_content` on BOTH arms, because a
        # real record's assistant row carries it as a list of blocks as often as as
        # a string. `.strip()` on that list raised AttributeError and took a
        # corpus-wide harvest down with it, and pinning the raw list would hand
        # `render_action` a Python repr to score as the model's own prose.
        content = _render_content(m.get("content"))
        if calls:
            return {"kind": "tool_call", "calls": calls, "content": content}
        if content.strip():
            return {"kind": "answer", "calls": [], "content": content}
    return None


@dataclass(frozen=True)
class CorpusRow:
    """One pinned (context, observation, recorded action) triple.

    `observation` is the VERBATIM sidecar text, so scoring a row never opens the
    sessions directory again: the frontier and this eval's re-runs then agree on
    the bytes, which is the nightly pinned-corpus discipline. `observation_bytes`
    is the size of those bytes as harvested — the denominator every budget is
    taken from, and the reason a re-harvest that found a bigger sidecar has to
    change the pinned file rather than the row's arithmetic.
    """
    row_id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    observation: str
    observation_bytes: int
    prefix: tuple[dict, ...]
    next_action: dict

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "session_id": self.session_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "observation": self.observation,
            "observation_bytes": self.observation_bytes,
            "prefix": [dict(m) for m in self.prefix],
            "next_action": self.next_action,
        }

    @property
    def messages(self) -> list[dict]:
        """The message list the arms are built on: prefix, then this observation."""
        return [dict(m) for m in self.prefix] + [
            {"role": "tool", "tool_call_id": self.tool_call_id,
             "content": self.observation}
        ]


#: The four named reasons a scanned tool row did not become a pinned row, in the
#: order `harvest` TESTS them. A row is counted under the FIRST of these it fails,
#: and under exactly one of them. The size band is tested before the two
#: record-quality checks on purpose: a row that is both under-band and nameless is a
#: band casualty, and charging it to `no_tool_name` would report a record-quality
#: problem that the band alone explains.
HARVEST_SKIP_REASONS = ("no_sidecar", "below_min_bytes", "no_tool_name",
                        "no_next_action")


@dataclass
class HarvestSkips:
    """How many scanned tool rows each of the four requirements cost.

    `no_sidecar` covers both a missing sidecar and one whose bytes could not be
    read — either way the observation was never obtained. A row that is short AND
    nameless is counted once, under whichever of the four it failed first, so the
    four counts never double-count a row and never lose one.
    """
    no_sidecar: int = 0
    below_min_bytes: int = 0
    no_tool_name: int = 0
    no_next_action: int = 0

    @property
    def total(self) -> int:
        return sum(getattr(self, k) for k in HARVEST_SKIP_REASONS)

    def as_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in HARVEST_SKIP_REASONS}


@dataclass
class Harvest:
    """One pass over the corpus: the rows it pinned, and what it skipped and why.

    The identity a caller can check without re-reading the store:
    `scanned == len(rows) + skips.total`, over the tool rows in records that had a
    readable message list AND a spill directory beside the record. A session with no
    spill directory is not scanned at all — no observation bytes could exist for it —
    and neither is a record whose JSON did not parse; `sessions_unscanned` names those
    two together, so a harvest that reads almost nothing stays visible instead of
    reading as an empty corpus. `max_rows` stops the scan where the bound lands, so
    the identity holds over the rows examined up to that point.
    """
    rows: list[CorpusRow]
    skips: HarvestSkips
    scanned: int
    sessions_unscanned: int = 0


def harvest(
    sessions_dir: Path | None = None,
    *,
    min_bytes: int = MIN_OBSERVATION_BYTES,
    max_rows: int = 0,
    session_files: Iterable[Path] | None = None,
) -> Harvest:
    """Harvest pinned rows from real session records and their spill sidecars.

    A row needs all four, or it is skipped and counted here under one of
    `HARVEST_SKIP_REASONS`: a sidecar holding the observation verbatim, a tool name
    the record can name, a next action that decides something, and an observation at
    or over `min_bytes`. `max_rows` bounds the output, 0 = all.
    """
    root = Path(sessions_dir) if sessions_dir is not None else _default_sessions_dir()
    files = (sorted(Path(p) for p in session_files) if session_files is not None
             else sorted(root.glob("*.json")))
    rows: list[CorpusRow] = []
    skips = HarvestSkips()
    scanned = 0
    unscanned_sessions = 0

    def result() -> Harvest:
        return Harvest(rows=rows, skips=skips, scanned=scanned,
                       sessions_unscanned=unscanned_sessions)

    for path in files:
        sid = path.name[:-len(".json")]
        # The sidecar lives beside ITS OWN record. Resolving it against `root` let a
        # `session_files` list aimed at one tree read a same-named session's spill
        # out of another, and made the documented `session_files=` call on a tmp dir
        # find no spill directory at all.
        spill = path.parent / f"{sid}.tool-results"
        if not spill.is_dir():
            unscanned_sessions += 1
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unscanned_sessions += 1
            continue
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            unscanned_sessions += 1
            continue
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            scanned += 1
            sc = _resolve_sidecar(spill, m)
            if sc is None:
                skips.no_sidecar += 1
                continue
            try:
                body = sc.read_text(encoding="utf-8")
            except OSError:
                skips.no_sidecar += 1
                continue
            nbytes = len(body.encode("utf-8"))
            if nbytes < min_bytes:
                skips.below_min_bytes += 1
                continue
            cid = m.get("tool_call_id") or ""
            name = _tool_name_for(messages, i, cid)
            if name is None:
                skips.no_tool_name += 1
                continue
            nxt = _next_action(messages, i)
            if nxt is None:
                skips.no_next_action += 1
                continue
            rows.append(CorpusRow(
                row_id=f"{sid}#{cid}",
                session_id=sid,
                tool_call_id=cid,
                tool_name=name,
                observation=body,
                observation_bytes=nbytes,
                prefix=tuple(_chat_row(x) for x in messages[:i] if _is_history_row(x)),
                next_action=nxt,
            ))
            if max_rows and len(rows) >= max_rows:
                return result()
    return result()


def save_corpus(rows: Sequence[CorpusRow], path: Path) -> Path:
    """Write the pinned corpus as YAML, and read it back before claiming it."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CORPUS_VERSION,
        "min_observation_bytes": MIN_OBSERVATION_BYTES,
        "budgets": list(BUDGETS),
        "row_count": len(rows),
        "rows": [r.to_dict() for r in rows],
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=10_000),
        encoding="utf-8",
    )
    loaded = load_corpus(path)
    if len(loaded) != len(rows):
        raise ValueError(f"corpus round-trip lost rows: wrote {len(rows)}, read {len(loaded)}")
    for wrote, read in zip(rows, loaded):
        if read.observation != wrote.observation or read.next_action != wrote.next_action:
            raise ValueError(f"corpus round-trip changed row {wrote.row_id}")
    return path


def load_corpus(path: Path) -> list[CorpusRow]:
    """Read a pinned corpus file. The scoring path consumes THIS, not a session dir."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw_rows = (data or {}).get("rows") or []
    out: list[CorpusRow] = []
    for r in raw_rows:
        out.append(CorpusRow(
            row_id=str(r["row_id"]),
            session_id=str(r.get("session_id", "")),
            tool_call_id=str(r.get("tool_call_id", "")),
            tool_name=str(r.get("tool_name", "")),
            observation=r["observation"],
            observation_bytes=int(r["observation_bytes"]),
            prefix=tuple(r.get("prefix") or ()),
            next_action=dict(r.get("next_action") or {}),
        ))
    return out


# ---------------------------------------------------------------------------
# Rendering: one template, both arms, and the generation it stands in for
# ---------------------------------------------------------------------------

#: Built by concatenation so no reader's tooling (or a later sed) treats this
#: file as containing a real ChatML delimiter of its own.
IM_START = "<|" + "im_start|>"
IM_END = "<|" + "im_end|>" + "\n"

TOOL_CALL_OPEN = "<|" + "tool_call" + "|>"
TOOL_CALL_CLOSE = "<|" + "end_tool_call" + "|>" + "\n"

#: What `render_chatml(..., add_generation_prompt=True)` appends. The
#: teacher-forced prompt is that render plus the recorded action, so this opener
#: sits immediately BEFORE the action and is the first thing a window taken off the
#: end of the readout reaches into when it is wider than the action — which is what
#: the estimated span (`SPAN_CHAR_ESTIMATE`) is at risk of. Named rather than
#: spelled twice so the two ends of the render cannot drift apart.
GENERATION_TRAILER = f"{IM_START}assistant\n"


def render_chatml(messages: Sequence[dict], *, add_generation_prompt: bool = True) -> str:
    """Render chat messages to the ChatML the `/v1/completions` route needs.

    `/v1/completions` applies no chat template, so the template is applied HERE,
    explicitly, and the same bytes are what both arms are scored on — the item's
    "record which route scored each number" requirement starts at the render. An
    assistant turn's `tool_calls` are rendered into content, which is what the
    `qwen3_xml` parser on this box puts in content; the reasoning parser has to be
    switched off at request time (`enable_thinking: false`) or it occupies the
    first generated positions and every number below would be scoring it.
    """
    out: list[str] = []
    for m in messages:
        role = m.get("role") or "user"
        body = _render_content(m.get("content"))
        if role == "assistant":
            body += render_calls(m.get("tool_calls") or ())
        out.append(f"{IM_START}{role}\n{body}{IM_END}")
    if add_generation_prompt:
        out.append(GENERATION_TRAILER)
    return "".join(out)


def _render_content(content: Any) -> str:
    """Message content as text, including the list-of-blocks shape rung 4 writes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return "" if content is None else str(content)


def render_calls(calls: Iterable[dict]) -> str:
    """Tool calls as the assistant text this engine emits for them."""
    out: list[str] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") or {}
        args = fn.get("arguments")
        body = args if isinstance(args, str) else json.dumps(args or {}, sort_keys=True)
        name = fn.get("name") or ""
        out.append(f"{TOOL_CALL_OPEN}{name}\n{body}{TOOL_CALL_CLOSE}")
    return "".join(out)


def render_action(next_action: dict) -> str:
    """The recorded next action, in the same form a generated step takes.

    Teacher-forcing appends this after the assistant header, so the string being
    scored is one the engine itself would have produced. The trailing stop marker
    is part of it: the first scored position is then the first action token, and
    the last is the stop, which is the whole decision and nothing after it.

    `content` goes through `_render_content` even though `harvest` now pins it as
    text, because a corpus YAML written before that fix carries the block list: an
    arm rebuilt from one of those rows must still score the prose, not the repr.
    """
    content = _render_content(next_action.get("content"))
    calls = next_action.get("calls") or ()
    return f"{content}{render_calls(calls)}{IM_END}"


_INVOKE_RE = re.compile(
    r"<\|tool_call\|>\s*([A-Za-z0-9_.:-]+)\s*\n(.*?)<\|end_tool_call\|>",
    re.DOTALL,
)
_XML_RE = re.compile(
    r"<tool_call\s+name=\"([^\"]+)\">\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def parse_calls(text: str) -> list[ToolCall]:
    """Tool calls an arm actually emitted, in order.

    Two spellings, because the answer depends on which grammar the engine was
    served with and this module must read either without pretending to know: the
    ChatML-XML form (`<tool_call name="x">{...}</tool_call>`) and the delimited
    form `render_calls` writes. An empty result is a real finding, not a parse
    failure — it is what `decline` exists to record (clause 5).
    """
    out: list[ToolCall] = []
    for rx in (_XML_RE, _INVOKE_RE):
        for name, args in rx.findall(text or ""):
            out.append(ToolCall(name=name.strip(), arguments=args.strip(),
                                origin="generated"))
        if out:
            break
    return out


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

#: Production's rung 4, named as what it is: the arm every compressor must beat.
RUNG4 = "rung4_truncate"
EXTRACTIVE = "extractive"
SUMMARY = "summary"
COMPRESSORS: tuple[str, ...] = (RUNG4, EXTRACTIVE, SUMMARY)

#: Production's own `min_chars`, restated as this module's default so the
#: baseline arm's call is the call the relief ladder makes.
DEFAULT_MIN_CHARS = 4096


def rung4_replacement(
    messages: Sequence[dict],
    *,
    target_chars: int,
    tool_call_id: str = "",
    session_id: str = "",
    min_chars: int = DEFAULT_MIN_CHARS,
    disallowed_tools: list[str] | None = None,
    name_session_record: bool = False,
) -> str:
    """What production's relief rung 4 writes in place of the largest tool result.

    This function's whole body is a call to `_truncate_largest_tool_results`, on a
    deep copy, and a read-back. It does not reimplement the notice, the ranking, or
    the spill-or-not branch: an arm that paraphrased production would compare a
    guess against a guess, and the #600 lesson is that the arm has to BE the rung.
    `app.harness.loop` is imported lazily for the same reason `microcompact` is
    imported lazily inside `app.compaction` — so this module stays importable
    without the harness tree.

    `session_id` defaults to "" on purpose. With no session id the rung takes its
    no-spill branch: purely lossy, writes nothing, and reproducible from the
    corpus alone. Pass one to score the SPILL-BACKED arm as it exists on this box
    after #1499, which is the honest baseline for a decision cost, at the price of
    writing a file and making the notice depend on that session's deny list.
    """
    from app.harness.loop import _truncate_largest_tool_results

    work = copy.deepcopy(list(messages))
    _truncate_largest_tool_results(
        work,
        target_chars=target_chars,
        session_id=session_id,
        min_chars=min_chars,
        disallowed_tools=disallowed_tools,
        name_session_record=name_session_record,
    )
    return _render_content(_chosen_tool_message(work, tool_call_id).get("content"))


def _chosen_tool_message(work: Sequence[dict], tool_call_id: str) -> dict:
    """The message rung 4 acts on, read back out of the list it just rewrote.

    Ranked the way the rung ranks — largest tool result first — so the arm names
    the same message production overwrote whether or not the caller supplied an
    id. An empty content on no-tool-message input is the honest answer: there was
    no observation to replace, and inventing one here would put a fabricated body
    into a scored prompt.
    """
    best: dict | None = None
    best_size = -1
    for m in work:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        if tool_call_id and m.get("tool_call_id") == tool_call_id:
            return m
        size = len(_render_content(m.get("content")))
        if size > best_size:
            best, best_size = m, size
    return best if best is not None else {"content": ""}


def _split_units(observation: str) -> list[str]:
    """Observation -> scoreable spans, preferring structure over lines.

    Paragraphs when the observation has them (a fetched page, a report), else
    lines. Neither is a claim about English sentences: an extractive arm that
    reassembled prose with a joined-string regex would report spans this box's
    tool output never has.
    """
    parts = [p for p in observation.split("\n\n") if p.strip()]
    if len(parts) >= 3:
        return parts
    return [ln for ln in observation.splitlines() if ln.strip()]


_WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")


def _query_terms(prefix: Sequence[dict]) -> set[str]:
    """What the compression is conditioned on: the live question and the recent
    reasoning, taken from the tail of the prefix — not the whole transcript,
    which would make every span score on the system prompt and select nothing.
    """
    tail: list[str] = []
    chars = 0
    for m in reversed(list(prefix)):
        if m.get("role") == "system":
            continue
        text = _render_content(m.get("content"))
        tail.append(text)
        chars += len(text)
        if chars >= 12_000:
            break
    return {w.lower() for w in _WORD_RE.findall(" ".join(reversed(tail)))}


def extractive_spans(observation: str, terms: set[str], budget_chars: int) -> str:
    """Budgeted extractive selection, greedy by query overlap, in source order.

    Relevance = distinct query terms present, ties broken by original position, so
    the choice is a deterministic function of (observation, prefix) and a re-run
    scores the same spans. Spans are joined by ONE newline, and the newline is
    CHARGED to the budget: an arm that spent `budget_chars` of pieces and then
    joined them with k-1 more characters came back over budget, and the whole
    comparison is at MATCHED budgets — an extractive arm holding 3,625 chars
    against a truncation arm holding 3,600 is not the same x-axis point. The
    byte-level rule is written here rather than left to `"\\n".join(...)` and then
    quoted in a report as if it were the original bytes. Joining with the source
    separators is impossible — the units are what the split removed — so a
    reassembled body is NOT byte-identical to the source, and no aggregate over
    this arm may claim it is. That is why the byte-identity test belongs to the
    rung-4 arm alone.
    """
    units = _split_units(observation)
    if budget_chars <= 0:
        return ""
    scored: list[tuple[int, int, str]] = []
    for pos, unit in enumerate(units):
        low = unit.lower()
        hits = sum(1 for t in terms if t in low)
        scored.append((-hits, pos, unit))
    scored.sort()
    chosen: list[tuple[int, str]] = []
    used = 0
    for _neg_hits, pos, unit in scored:
        sep = 1 if chosen else 0  # the newline this piece will be joined on
        room = budget_chars - used - sep
        if room <= 0:
            break
        piece = unit if len(unit) <= room else unit[:room]
        chosen.append((pos, piece))
        used += sep + len(piece)
    chosen.sort()
    return "\n".join(piece for _pos, piece in chosen)


def compress(
    row: CorpusRow,
    budget: float,
    *,
    compressor: str = RUNG4,
    summariser: Callable[[str, int], str] | None = None,
    session_id: str = "",
) -> tuple[str, dict]:
    """The candidate replacement for this row's observation, with its accounting.

    Returns `(replacement, info)`; `info` records the arm's own numbers so a
    report can tell the budget that was ASKED from the size that was RETAINED.
    """
    if compressor not in COMPRESSORS:
        raise ValueError(f"unknown compressor {compressor!r}; one of {COMPRESSORS}")
    observation = row.observation
    budget_chars = max(0, int(round(len(observation) * budget)))
    target_chars = max(0, len(observation) - budget_chars)
    if compressor == RUNG4:
        replacement = rung4_replacement(
            row.messages, target_chars=target_chars,
            tool_call_id=row.tool_call_id, session_id=session_id,
        )
        return replacement, {
            "compressor": compressor, "budget": budget,
            "budget_chars": budget_chars, "target_chars": target_chars,
            "arm": "production_relief_rung4",
        }
    if compressor == EXTRACTIVE:
        replacement = extractive_spans(observation, _query_terms(row.prefix), budget_chars)
        return replacement, {
            "compressor": compressor, "budget": budget,
            "budget_chars": budget_chars, "target_chars": target_chars,
            "arm": "query_conditioned_extractive",
        }
    if summariser is None:
        raise ValueError(
            "the summary arm needs a summariser callable (observation, budget_chars) "
            "-> str; the frontier run wires it to the llama.cpp 35B-A3B and a unit "
            "test injects a stub. Refusing rather than substituting head-truncation "
            "keeps a missing arm from reporting as a scored one."
        )
    replacement = summariser(observation, budget_chars)
    return replacement, {
        "compressor": compressor, "budget": budget,
        "budget_chars": budget_chars, "target_chars": target_chars,
        "arm": "secondary_model_summary",
    }


# ---------------------------------------------------------------------------
# The logprob readout — one client, and a refusal where there is no readout
# ---------------------------------------------------------------------------

#: The teacher-forced readout asks for the OBSERVED TOKEN ONLY: `prompt_logprobs: 0`
#: makes this engine return exactly one entry per scored position, and that entry IS
#: the prompt's own token at that position. The observed token is therefore
#: identified by CARDINALITY — one entry, whatever its `rank` says — and never by a
#: rank. This is #4499, and it replaced a `TOP_K = 8` request.
#:
#: Why a top-k map (`prompt_logprobs: K`, K >= 1) cannot be read here. vLLM's map is
#: the top-K candidates PLUS the observed token appended at its true rank when that
#: rank exceeds K, so with K >= 1 the observed token is always PRESENT but is
#: identifiable only in the K+1-entry case: when its rank is <= K it is one candidate
#: among K and nothing in the entry says which. Reading rank 1 there scores the argmax
#: CANDIDATE, not the recorded action's token — measured on this engine, the observed
#: token's rank passed 100,000 on two positions of the capture committed under
#: `tests/fixtures/`, and #4499's own typed-prompt probe found the observed
#: `' Berlin'` at rank 14 under `' Paris'` at rank 1. So `_sole_entry` refuses a
#: multi-entry map and no reader here guesses.
#:
#: What the K=0 shape cannot carry is `kl_first_position`: it has no candidates, so
#: that column reports None on every row of this route and the report prints an empty
#: cell. Buying it back costs a second, K>=1 readout per arm — twice the requests —
#: and that KL must never enter a ΔCE mean, because the two shapes are different
#: quantities taken from different requests. Whether to pay for it is ruled on in
#: #4499's owed checks, not here.
#:
#: `-1` is not a route to "all of them": this engine refuses it (`Requested prompt
#: logprobs of 248320, which is greater than max allowed: 20`). `0` is inside that
#: bound and means the observed token only, not none.
OBSERVED_TOKENS_ONLY = 0

#: One generated step, budgeted. Long enough to hold a tool call's arguments,
#: short enough that the arm is graded on the decision and not on an essay.
GENERATION_MAX_TOKENS = 64

INVALID_NO_LOGPROBS = "no_prompt_logprobs"
INVALID_ZERO_SCORED = "zero_scored_positions"
INVALID_SPAN_SHORT = "scored_positions_shorter_than_action"

#: The readout is per TOKEN and the action's length is in CHARACTERS; this is the
#: codebase's own heuristic for crossing that gap — 1 token ≈ 4 chars, the
#: `TOKENS_PER_CHAR = 4` of app/compaction.py:76-77, named locally rather than
#: imported from a compaction module that has no reason to export it.
#:
#: Conflating the two units is not a rounding error. Sized by the action's
#: CHARACTER count, the window asks for roughly four times as many positions as the
#: action occupies, so it opens before the action does — inside the tool observation,
#: which is the one region of the prompt the two arms deliberately differ in. ΔCE
#: would then average arm-dependent observation positions and report them as the
#: cost of the decision, which is the similarity-to-the-replaced-prose this
#: instrument was written to refuse; and `kl_first_position` would describe a
#: position inside the observation while keeping a column called "first generated
#: position". Where the readout labels its own tokens the span is aligned exactly
#: (SPAN_TOKEN_TEXT); where it does not, this constant is the fallback, and the row
#: records which of the two routes scored it.
CHARS_PER_TOKEN = 4

#: `span_alignment` values: which of the two alignments produced the row's numbers.
SPAN_TOKEN_TEXT = "token_text"        # exact: the readout said what each position is
SPAN_CHAR_ESTIMATE = "char_estimate"  # estimated at CHARS_PER_TOKEN


@dataclass(frozen=True)
class _Span:
    """The teacher-forced read of one arm's action span."""
    nll: float
    positions: int
    first_topk: dict
    alignment: str = ""


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _prompt_positions(choice: Any) -> list | None:
    """Per-position PROMPT-token logprobs out of one `/v1/completions` answer.

    None means there is no readout to score, which is the #3850 invalid case, not
    a zero: an engine that answers with `prompt_logprobs: null` (the flag never
    reached it, the route was the chat one, the parser ate the request) has said
    nothing, and `delta_ce_nats = 0.0` would report that silence as perfect
    preservation. Accepts both the vLLM placement (`choices[0].prompt_logprobs`)
    and the nested one (`choices[0].logprobs.prompt_logprobs`), because that is
    where the two builds put it; both are recorded per number by `engine_route`.
    """
    if not isinstance(choice, dict):
        return None
    direct = choice.get("prompt_logprobs")
    if isinstance(direct, list) and direct:
        return direct
    nested = choice.get("logprobs")
    if isinstance(nested, dict):
        inner = nested.get("prompt_logprobs")
        if isinstance(inner, list) and inner:
            return inner
    return None


def _sole_entry(pos: Any) -> dict | None:
    """The one entry of a `prompt_logprobs: 0` position, or None if it is not one.

    vLLM's `/v1/completions` labels a `prompt_logprobs` position with a map keyed by
    token id (the OpenAI chat-shaped form puts the logprob at the top level instead).
    At `prompt_logprobs: 0` that map holds EXACTLY ONE entry, and it is the prompt's
    own token at that position — the only shape this engine serves whose observed
    token is identifiable without inference, because cardinality settles it (#4499).

    A map with more than one entry is a top-k readout, and this returns None for it:
    the observed token is somewhere in that map at a rank nothing in the entries
    discloses, so the choice is to refuse the position or to score rank 1, which is
    the argmax candidate and a different quantity. The token-id key test is what
    keeps a chat-shaped position's `top_logprobs` list from reading as a one-entry
    map.
    """
    if not isinstance(pos, dict) or len(pos) != 1:
        return None
    (token_id, entry), = pos.items()
    try:
        int(token_id)
    except (TypeError, ValueError):
        return None
    return entry if isinstance(entry, dict) else None


def _position_nll(pos: Any) -> float | None:
    """The teacher-forced -log P of one position, or None when unscored.

    None is also the answer for a MULTI-entry map: an unidentified observed token is
    an unscorable position, and a rank-1 logprob would be reported in its place as
    though it were the recorded action's own (#4499).
    """
    if not isinstance(pos, dict):
        return None        # vLLM sends null at position 0, and for masked tokens
    for key in ("token_logprob", "logprob"):
        v = _num(pos.get(key))
        if v is not None:
            return -v      # served as a logprob (<= 0); CE is its negation
    v = _num(pos.get("nll"))
    if v is not None:
        return v
    entry = _sole_entry(pos)   # token-id-keyed map: only the one-entry shape scores
    lp = None if entry is None else _num(entry.get("logprob"))
    return None if lp is None else -lp


def _position_topk(pos: Any) -> dict:
    """This position's top-k as {token_text: logprob}.

    Token TEXT is the key on purpose: the two arms' tokenisers are the same model,
    but the observation lengths differ, so an id is only comparable inside one
    response and a text key is comparable across the pair.
    """
    out: dict[str, float] = {}
    if not isinstance(pos, dict):
        return out
    raw = pos.get("top_logprobs")
    rows: Iterable[Any]
    if isinstance(raw, dict):
        rows = [{"token": k, "logprob": v} for k, v in raw.items()]
    elif isinstance(raw, list):
        rows = raw
    else:
        return out
    for item in rows:
        if isinstance(item, dict):
            tok = item.get("token")
            lp = _num(item.get("logprob"))
            if tok is not None and lp is not None:
                out[str(tok)] = lp
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            lp = _num(item[1])
            if lp is not None:
                out[str(item[0])] = lp
    return out


def _token_surface(pos: Any) -> str | None:
    """The text one readout position stands for, or None when it does not say.

    A served readout labels a position one of four ways: a decoded string (`token`
    or `text`, the shape #540's route returns), a raw byte list (`bytes`, what
    vLLM's OpenAI-compat `prompt_logprobs` carries), a token-id-keyed map holding the
    observed entry alone (`prompt_logprobs: 0`, whose `decoded_token` is the prompt's
    own text at that position), or nothing but an id. The last case is the only reason
    an estimated span needs to exist — and the reason that span has to be labelled as
    an estimate wherever it is quoted.
    """
    if not isinstance(pos, dict):
        return None
    for key in ("token", "text"):
        value = pos.get(key)
        if isinstance(value, str):
            return value
    raw = pos.get("bytes")
    if isinstance(raw, list) and raw and all(isinstance(b, int) for b in raw):
        try:
            return bytes(raw).decode("utf-8", errors="replace")
        except (ValueError, OverflowError):
            return None
    if isinstance(raw, str):
        return raw
    # A token-id-keyed map: the surface is the SOLE entry's `decoded_token`. A
    # multi-entry map gets None rather than rank 1's text, because rank 1 is some
    # other token's surface and walking those against the prompt is exactly how a
    # rank-1 reader labels a span `token_text` — the exact-alignment label — on text
    # that is not the prompt's (#4499).
    entry = _sole_entry(pos)
    if entry is not None:
        value = entry.get("decoded_token")
        return value if isinstance(value, str) else None
    return None


def _align_action_span(positions: list, action_text: str) -> tuple[int, str, str | None]:
    """(index of the action's first position, alignment route, invalid reason).

    `reason` is None on a usable span and one of the `INVALID_*` values when there
    is not one; `start` is only meaningful in the first case.

    The teacher-forced prompt is `render_chatml(messages) + action_text` (see
    `score_row`), so the action is the LAST thing in it and its span is measured
    back from the end of the readout. Two routes:

    * `SPAN_TOKEN_TEXT` — every position carries its own surface, so the span is
      exactly the positions whose text covers the action's character count. This is
      alignment by the engine's own token boundaries: the action is the same bytes in
      both arms, so both windows open on the same token, and nothing before the
      action (not `GENERATION_TRAILER`, not the observation) is scored.
    * `SPAN_CHAR_ESTIMATE` — some position has no surface, so the width is estimated
      at CHARS_PER_TOKEN. That estimate is wrong in BOTH directions and the failure
      is asymmetric: too wide reaches past the assistant opener into the
      observation, which is the one region the two arms deliberately differ in, and
      the mean then reports arm-dependent prompt positions as the cost of the
      decision; too narrow drops the action's own opening, which is where the tool
      name lives and where `kl_first_position` is supposed to be read. It is kept
      because a readout with no token labels otherwise scores nothing at all, and it
      is recorded per row so no report can quote it as an aligned number.

    A readout that cannot hold the action it is claimed to have scored is an
    invalid row rather than an under-scored one.
    """
    total = len(positions)
    need = len(action_text)
    if need <= 0:
        return -1, "", INVALID_ZERO_SCORED
    estimate = max(1, -(-need // CHARS_PER_TOKEN))  # ceil, integer only
    if total < estimate:
        return -1, "", INVALID_SPAN_SHORT
    # A masked entry (vLLM sends `null` for position 0, the BOS) is treated as
    # covering no characters rather than as "unlabelled", because on this engine it
    # is the only masked one and it sits at the far front — outside any action span.
    # Treating it as unlabelled instead would drop every real readout onto the
    # estimate route, which is the weaker measurement, for a position that cannot be
    # in the window. If a masked entry ever IS inside the window, it contributes no
    # nll to the mean and, if it is the window's first, `first_topk` comes back empty
    # and the KL reports itself as unavailable — the two refusals, not a guess.
    surfaces: list[str] = []
    for pos in positions:
        if pos is None:
            surfaces.append("")
            continue
        surface = _token_surface(pos)
        if surface is None:
            return total - estimate, SPAN_CHAR_ESTIMATE, None
        surfaces.append(surface)
    covered = 0
    for i in range(total - 1, -1, -1):
        covered += len(surfaces[i])
        if covered >= need:
            return i, SPAN_TOKEN_TEXT, None
    # The whole readout, and the action still is not covered: this readout does not
    # reach over the text it is supposed to have scored.
    return -1, "", INVALID_SPAN_SHORT


def _read_span(positions: list, action_text: str) -> tuple[_Span | None, str | None]:
    """Score the action's own positions, or say why the row is invalid.

    The window is `_align_action_span`'s, NOT the last `len(action_text)` positions:
    `prompt_logprobs` has one entry per TOKEN while `action_text` is measured in
    CHARACTERS, about four of them per token, so a character-sized window is roughly
    four times as wide as the action and opens long before it. What it then averages
    is the tool observation's own positions — arm-dependent by construction — and the
    ΔCE stops being the cost of the decision and becomes a similarity between the two
    texts, the exact quantity this module was written to refuse.
    """
    # Checked before any alignment arithmetic: a readout that carries no usable
    # logprob anywhere SCORES ZERO POSITIONS whatever its length, and labelling that
    # "span shorter than the action" would send a reader to look for a short readout
    # when the real problem is that nothing in it can be read.
    if not any(_position_nll(p) is not None for p in positions):
        return None, INVALID_ZERO_SCORED
    start, alignment, reason = _align_action_span(positions, action_text)
    if reason is not None:
        return None, reason
    tail = positions[start:]
    total = 0.0
    count = 0
    # THE first generated position: the action's first position, positionally and
    # not the first SCORED one. If the engine masked that position `first_topk` stays
    # empty and `kl_first_position` comes back None; substituting a later position
    # would report a divergence about a different decision point while keeping the
    # column's name, which is the failure this module exists to refuse.
    first = _position_topk(tail[0])
    for pos in tail:
        nll = _position_nll(pos)
        if nll is None:
            continue
        total += nll
        count += 1
    if count == 0:
        return None, INVALID_ZERO_SCORED
    return _Span(nll=total / count, positions=count, first_topk=first,
                 alignment=alignment), None


def kl_first_position(full_topk: dict, cand_topk: dict) -> float | None:
    """KL(P_full || P_cand) at the first generated position, in nats, or None.

    `p` is the FULL arm's distribution — the reference the compression is measured
    against, matching the paper's KL(full vs compressed). A token outside an arm's
    served top-k is charged that list's lowest logprob: a floor, so the term is
    bounded from above rather than dropped. Dropping it would understate the
    divergence exactly when the arms disagree most, which is the case the frontier
    exists to catch. Mass outside the FULL arm's support is not credited either
    way, so this is a truncation of the paper's objective, not the objective — the
    `top_k` column travels with every number so a reader can see how narrow the
    window was. Renormalising the two truncated lists against each other is the
    alternative, and it is worse: it changes the quantity and keeps the name. None
    when either side has no top-k at all: no distribution, no divergence, and a
    0.0 here would read as "the compression changed nothing".

    On the route this module ships (`OBSERVED_TOKENS_ONLY`) it is always None: a
    `prompt_logprobs: 0` readout carries the observed token and no candidates, so
    `first_topk` is empty on both arms and there is no distribution to diverge. That
    is the price of a ΔCE whose token identity is not a guess, and the empty cell is
    the honest report — a KL borrowed from a second, K>=1 readout would be a
    different request's distribution in a column about this one.
    """
    if not full_topk or not cand_topk:
        return None
    floor_cand = min(cand_topk.values())
    kl = 0.0
    for tok, lp_full in full_topk.items():
        p = math.exp(lp_full)
        lp_cand = cand_topk.get(tok, floor_cand)
        kl += p * (lp_full - lp_cand)
    return kl


#: A generated step that contains no tool call, recorded beside the tier. Not
#: `protocol_invalid`: that tier means "a call is present and unreadable", and
#: folding suppression into it would let a compressor that deletes the call sit in
#: the same bucket as one that mangles its arguments — with a different fix.
DECLINE_COMPRESSED = "compressed_arm_declined"
DECLINE_FULL = "full_arm_declined"
DECLINE_BOTH = "both_arms_declined"


def _step_decision(generated: str, recorded: dict) -> tuple[list[ToolCall], bool]:
    """The calls this arm's generated step emitted, and whether it DECLINED.

    Decline is decided on the CALL, not on the text being empty: an arm that
    answers in prose where the trajectory recorded a tool call has suppressed the
    decision, and that is the failure a next-action score must not reward.
    """
    calls = parse_calls(generated)
    recorded_called = (recorded.get("kind") == "tool_call") and bool(recorded.get("calls"))
    return calls, recorded_called


def decide_tier(
    full_generated: str,
    cand_generated: str,
    recorded: dict,
) -> dict:
    """`{tier, decline, why}` — the audit-grade read, decided before the KL.

    The tier compares the two ARMS' generated calls (the full arm's own output vs
    the compressed arm's), which is the pair a human can check by reading two
    strings; the recorded action is the referee that says whether a call was owed
    at all. Precedence is deliberate: suppression is settled first, because a tier
    computed against a missing call is a comparison that cannot be supported —
    the same reason `action_equivalence` ranks `protocol_invalid` above
    `different_tool`.
    """
    cand_calls, recorded_called = _step_decision(cand_generated, recorded)
    full_calls, _ = _step_decision(full_generated, recorded)
    if not cand_calls and not full_calls:
        if recorded_called:
            return {"tier": None, "decline": DECLINE_BOTH,
                    "why": "the recorded step called; neither arm did"}
        return {"tier": None, "decline": None,
                "why": "no tool call was recorded and neither arm emitted one"}
    if not cand_calls:
        return {"tier": None, "decline": DECLINE_COMPRESSED,
                "why": "the full arm emitted a tool call; the compressed arm "
                       "emitted none — suppression, not agreement"}
    if not full_calls:
        return {"tier": None, "decline": DECLINE_FULL,
                "why": "the compressed arm emitted a tool call; the full arm did not"}
    return {"tier": classify(cand_calls[0], full_calls[0]), "decline": None,
            "why": "both arms emitted a call"}


def score_row(
    row: CorpusRow,
    engine: Any,
    *,
    compressor: str = RUNG4,
    budget: float = 0.25,
    top_k: int = OBSERVED_TOKENS_ONLY,
    summariser: Callable[[str, int], str] | None = None,
    session_id: str = "",
) -> dict:
    """One (row, compressor, budget) record. THE SCORING ENTRY POINT.

    `engine` is the seam this whole module exists to make testable: anything with
    `teacher_forced(prompt, *, top_k)` and `step(prompt, *, max_tokens)`, both
    returning one `choices[0]` object. `LogprobClient` is the real one; a unit
    test injects a stub and no socket is opened. Two requests per arm — the
    teacher-forced span cannot also be the generated step, since the prompt that
    contains the recorded action has already consumed the step being graded — and
    both reads of one arm come from the same engine, so no number in the record is
    comparable across different servers.

    The returned record always carries `valid`. When false, the measurement fields
    are None and `invalid_reason` names which read failed on which arm; the row is
    EXCLUDED from every aggregate, and `agrees` is false rather than absent — the
    row preserves nothing on this evidence, and it is not evidence of anything.
    """
    replacement, info = compress(
        row, budget, compressor=compressor, summariser=summariser, session_id=session_id,
    )
    full_messages = row.messages
    cand_messages = [dict(m) for m in full_messages[:-1]] + [
        {"role": "tool", "tool_call_id": row.tool_call_id, "content": replacement}
    ]
    action_text = render_action(row.next_action)

    full_tf = engine.teacher_forced(
        render_chatml(full_messages) + action_text, top_k=top_k)
    cand_tf = engine.teacher_forced(
        render_chatml(cand_messages) + action_text, top_k=top_k)
    full_step = engine.step(
        render_chatml(full_messages), max_tokens=GENERATION_MAX_TOKENS)
    cand_step = engine.step(
        render_chatml(cand_messages), max_tokens=GENERATION_MAX_TOKENS)

    record: dict[str, Any] = {
        "row_id": row.row_id,
        "tool_name": row.tool_name,
        "compressor": compressor,
        "budget": budget,
        "arm": info.get("arm", ""),
        "observation_bytes": row.observation_bytes,
        "budget_chars": info.get("budget_chars", 0),
        "target_chars": info.get("target_chars", 0),
        "tokens_retained": len(replacement),
        # Named so the column states its own unit: this is a CHARACTER count, the
        # same currency the budget axis and production's rung 4 decide in. No
        # tokenizer was consulted to produce it, so a report that calls it tokens
        # is reading a number the instrument never measured.
        "retained_unit": "chars",
        "tokens_retained_pct": (round(100.0 * len(replacement) / len(row.observation), 2)
                                if row.observation else 0.0),
        "action_kind": row.next_action.get("kind", ""),
        "valid": False,
        "invalid_reason": None,
        "delta_ce_nats": None,
        "kl_first_position": None,
        "full_arm_nll": None,
        "candidate_arm_nll": None,
        "action_positions_scored": 0,
        # Present on invalid rows too, so the JSONL schema does not depend on which
        # rows happened to score.
        "span_alignment": "",
        # The readout SHAPE that produced every number on this row, not a width to
        # interpret: 0 is the observed-token-only request, so the row's ΔCE is a mean
        # of sole-entry logprobs and its KL has no candidates to be computed from.
        # A published ΔCE names this field, because a K>=1 mean is a different
        # quantity and mixing the two would import their disagreement into the delta.
        "top_k": top_k,
        "tier": None,
        "decline": None,
        "tier_why": "",
        "agrees": False,
        "engine_route": getattr(engine, "route_label", "injected"),
    }

    full_pos = _prompt_positions(full_tf)
    cand_pos = _prompt_positions(cand_tf)
    if full_pos is None or cand_pos is None:
        side = "full" if full_pos is None else "candidate"
        record["invalid_reason"] = f"{INVALID_NO_LOGPROBS}:{side}_arm"
        record["tier_why"] = "no teacher-forced readout, so nothing was scored"
        return record

    span_full, why_full = _read_span(full_pos, action_text)
    span_cand, why_cand = _read_span(cand_pos, action_text)
    if span_full is None or span_cand is None:
        side = "full" if span_full is None else "candidate"
        reason = why_full if span_full is None else why_cand
        record["invalid_reason"] = f"{reason}:{side}_arm"
        return record
    # Which route sized the window that produced every number on this row. The two
    # arms read the same action, so a disagreement means one arm's readout labelled
    # its positions and the other's did not, and the row's ΔCE is then a comparison
    # between an aligned span and an estimated one — a mixed label, never a bare
    # `char_estimate` that reads as the weaker route's usual case.
    routes = {span_full.alignment, span_cand.alignment}
    record["span_alignment"] = next(iter(routes)) if len(routes) == 1 else "+".join(sorted(routes))

    decision = decide_tier(
        _choice_text(full_step), _choice_text(cand_step), row.next_action)
    record.update(
        valid=True,
        invalid_reason=None,
        full_arm_nll=span_full.nll,
        candidate_arm_nll=span_cand.nll,
        delta_ce_nats=span_cand.nll - span_full.nll,
        kl_first_position=kl_first_position(span_full.first_topk, span_cand.first_topk),
        action_positions_scored=min(span_full.positions, span_cand.positions),
        tier=decision["tier"],
        decline=decision["decline"],
        tier_why=decision["why"],
        agrees=(decision["tier"] == CANONICALLY_EQUAL) if decision["decline"] is None else False,
    )
    return record


def _choice_text(choice: Any) -> str:
    if isinstance(choice, dict):
        return choice.get("text") or ""
    return ""


# ---------------------------------------------------------------------------
# The engine client — the frontier run's seam; the tests never construct this
# ---------------------------------------------------------------------------

#: The primary engine's completions port. `enable_thinking` defaults to False
#: because the reasoning parser otherwise occupies the first generated positions
#: and every number above would be scoring it instead of the action (#540).
DEFAULT_BASE_URL = "http://127.0.0.1:8096/v1"


class LogprobClient:
    """One POST to `/v1/completions` per read, both reads from one server.

    #638's guidance is that there is ONE logprob client, so this is the only place
    in the repo that asks this question of this engine over HTTP: the
    teacher-forced span and the generated step are separate methods on ONE client
    hitting ONE base_url, and a record's `engine_route` says which route scored it.
    `chat_template_kwargs` is sent on both so the thinking-parser fact is recorded
    per number rather than assumed; `build flags` is exactly what the item asked to
    re-verify before the frontier run, and `route_label` is where that lands.
    """

    route_label = "completions:prompt_logprobs:enable_thinking=false"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        model: str = "",
        timeout: float = 180.0,
        enable_thinking: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.enable_thinking = enable_thinking

    def _post(self, body: dict) -> dict:
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _body(self, prompt: str, *, max_tokens: int, logprobs: int | None) -> dict:
        body: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if self.model:
            body["model"] = self.model
        if logprobs is not None:
            body["logprobs"] = logprobs
            body["prompt_logprobs"] = logprobs
        return body

    def teacher_forced(self, prompt: str, *,
                       top_k: int = OBSERVED_TOKENS_ONLY) -> dict:
        """`choices[0]` for a prompt that already CONTAINS the action to score.

        `max_tokens: 1` because the answer is in `prompt_logprobs`; the generated
        token is discarded, and asking for 0 is not a thing this API serves.

        `top_k` defaults to `OBSERVED_TOKENS_ONLY`, so the request asks for
        `prompt_logprobs: 0` and every scored position comes back as a one-entry map
        holding the prompt's own token. Passing a K >= 1 is allowed and gets you a
        candidate distribution the readers refuse to score, which is the correct
        outcome: rank 1 of that map is the argmax candidate, not the served token
        (#4499).
        """
        return self._post(self._body(prompt, max_tokens=1, logprobs=top_k))["choices"][0]

    def step(self, prompt: str, *, max_tokens: int = GENERATION_MAX_TOKENS) -> dict:
        """`choices[0]` for one generated step at temperature 0."""
        return self._post(self._body(prompt, max_tokens=max_tokens, logprobs=None))["choices"][0]


# ---------------------------------------------------------------------------
# Aggregates: an invalid row is excluded, never averaged as a zero
# ---------------------------------------------------------------------------

def aggregate(records: Sequence[dict]) -> dict:
    """Per (compressor, budget) rollup over VALID rows only.

    `agreement_rate`'s denominator is valid rows whose comparison was DECIDABLE: a
    row with a tier, or one where the COMPRESSED arm alone failed to call. A
    suppressed call is that arm's own answer, so it stays in the denominator and
    counts against the rate — a compressor that deletes the tool call on every row
    posts 0.0, not an empty cell. `full_arm_declined` and `both_arms_declined` are
    the genuinely undecidable ones: the reference arm did not call either, so there
    is no full-arm decision for the compression to have preserved, and holding such
    a row against the compressor would be holding it for the engine. A row whose
    recorded step was an answer with no call in either arm decides nothing either.
    The invalid population is reported beside the rate, never folded into it: a
    compressor whose every row came back without logprobs would otherwise post a
    perfect 0-nat delta over an empty set.
    """
    out: dict[tuple[str, float], dict] = {}
    for rec in records:
        key = (str(rec.get("compressor")), float(rec.get("budget", 0.0)))
        a = out.setdefault(key, {
            "compressor": key[0], "budget": key[1], "rows": 0,
            "valid_rows": 0, "invalid_rows": 0, "invalid_reasons": {},
            "undecidable": 0, "declines": 0, "tiers": {},
            "delta_ce_sum": 0.0, "kl_sum": 0.0, "kl_rows": 0,
            "tokens_retained_sum": 0, "span_alignments": {},
        })
        a["rows"] += 1
        if not rec.get("valid"):
            a["invalid_rows"] += 1
            reason = str(rec.get("invalid_reason"))
            a["invalid_reasons"][reason] = a["invalid_reasons"].get(reason, 0) + 1
            continue
        a["valid_rows"] += 1
        # How this cell's ΔCE and KL were anchored. A cell with any estimated span is
        # the weaker measurement, and the report says so rather than letting the number
        # arrive with no trace of the route that produced it.
        route = str(rec.get("span_alignment") or "unknown")
        a["span_alignments"][route] = a["span_alignments"].get(route, 0) + 1
        a["tokens_retained_sum"] += int(rec.get("tokens_retained") or 0)
        a["delta_ce_sum"] += float(rec.get("delta_ce_nats") or 0.0)
        kl = rec.get("kl_first_position")
        if kl is not None:
            a["kl_sum"] += float(kl)
            a["kl_rows"] += 1
        if rec.get("decline"):
            a["declines"] += 1
        if rec.get("tier"):
            a["tiers"][str(rec["tier"])] = a["tiers"].get(str(rec["tier"]), 0) + 1
        elif rec.get("decline") == DECLINE_COMPRESSED:
            # Kept out of `undecidable` on purpose: the compressed arm answered,
            # and its answer was "no call". The rate must fall for it.
            pass
        else:
            a["undecidable"] += 1
    for a in out.values():
        decidable = a["valid_rows"] - a["undecidable"]
        a["decidable_rows"] = decidable
        a["agreement_rate"] = (
            round(a["tiers"].get(CANONICALLY_EQUAL, 0) / decidable, 4)
            if decidable else None
        )
        a["mean_delta_ce_nats"] = (
            round(a["delta_ce_sum"] / a["valid_rows"], 6) if a["valid_rows"] else None
        )
        a["mean_kl_first_position"] = (
            round(a["kl_sum"] / a["kl_rows"], 6) if a["kl_rows"] else None
        )
        a["mean_tokens_retained"] = (
            round(a["tokens_retained_sum"] / a["valid_rows"], 1) if a["valid_rows"] else None
        )
        a["protocol_invalid_rows"] = a["tiers"].get(PROTOCOL_INVALID, 0)
    return out


def frontier(agg: dict, *, floor: float, baseline: str = RUNG4) -> dict:
    """Per compressor: the smallest budget whose agreement clears `floor`.

    `floor` is the full-arm self-consistency spread measured by the frontier run's
    same-arm repeats — the item's step 5, and the reason `preserves the action` is
    a comparison rather than an adjective. `None` means no budget cleared it, which
    the report must SAY instead of leaving as a blank cell.
    """
    per: dict[str, dict] = {}
    for (compressor, budget), a in sorted(agg.items()):
        rate = a.get("agreement_rate")
        entry = per.setdefault(compressor, {"clearing_budget": None, "by_budget": {}})
        entry["by_budget"][f"{budget:.2f}"] = a
        if rate is not None and rate >= floor and entry["clearing_budget"] is None:
            entry["clearing_budget"] = budget
    for compressor, entry in per.items():
        if compressor == baseline:
            continue
        entry["beats_baseline"] = _dominates(entry, per.get(baseline, {}))
    return {"floor": floor, "compressors": per}


def _dominates(entry: dict, baseline_entry: dict) -> bool:
    """True only if this compressor matches or beats rung 4 at EVERY shared budget.

    Agreement is the ranking axis and ΔCE the tiebreak; a compressor that ties on
    agreement while retaining more of the observation has not bought anything.
    """
    base = baseline_entry.get("by_budget") or {}
    if not base:
        return False
    shared = 0
    for label, arm in (entry.get("by_budget") or {}).items():
        b = base.get(label)
        if not b:
            continue
        shared += 1
        rate, base_rate = arm.get("agreement_rate"), b.get("agreement_rate")
        if rate is None or base_rate is None:
            return False
        if rate < base_rate:
            return False
        if rate == base_rate:
            d, bd = arm.get("mean_delta_ce_nats"), b.get("mean_delta_ce_nats")
            t, bt = arm.get("mean_tokens_retained"), b.get("mean_tokens_retained")
            if d is None or bd is None or d > bd:
                return False
            if d == bd and t is not None and bt is not None and t > bt:
                return False
    return shared > 0


def _cell(value: Any, spec: str) -> str:
    """One numeric cell, or the literal `n/a` a None must print as.

    `n/a` and not `0`: an aggregate over no valid rows has no mean, and printing 0
    there is the same lie as scoring a missing logprob as a 0-nat delta.
    """
    return "n/a" if value is None else spec.format(value)


def span_route_line(agg: dict) -> str:
    """The `span routes over valid rows: exact=N, estimated=M` line, over the whole run.

    One function holds the sentence because two publishers read it: `render_report` prints
    it to the terminal at score time, and `render_note` puts the identical string in the
    published file. The route decided which window was scored, so a reader quoting a ΔCE
    from the note needs to know how many of those rows were measured on the estimated
    window — and note and stdout must not be able to drift into two accounts of one run.
    Returns "" when the run has no valid rows, which is no tally to publish.
    """
    routes: dict[str, int] = {}
    for a in agg.values():
        for route, n in (a.get("span_alignments") or {}).items():
            routes[route] = routes.get(route, 0) + n
    if not routes:
        return ""
    return ("span routes over valid rows: "
            + ", ".join(f"{r}={n}" for r, n in sorted(routes.items())))


def render_report(agg: dict, *, floor: float) -> str:
    """The budget x compressor table, with the invalid population in it."""
    fr = frontier(agg, floor=floor)
    lines = [
        f"next-action agreement floor (full-arm self-consistency): {floor}",
        "",
        f"{'compressor':<18}{'budget':>7}{'rows':>6}{'valid':>7}{'invalid':>9}"
        f"{'agree':>8}{'declines':>10}{'ΔCE nats':>11}{'KL@1':>10}{'tokens':>9}",
    ]
    for (compressor, budget), a in sorted(agg.items()):
        lines.append(
            f"{compressor:<18}{budget:>7.2f}{a['rows']:>6}{a['valid_rows']:>7}"
            f"{a['invalid_rows']:>9}"
            f"{_cell(a.get('agreement_rate'), '{:.3f}'):>8}"
            f"{a['declines']:>10}"
            f"{_cell(a.get('mean_delta_ce_nats'), '{:+.4f}'):>11}"
            f"{_cell(a.get('mean_kl_first_position'), '{:.4f}'):>10}"
            f"{_cell(a.get('mean_tokens_retained'), '{:.0f}'):>9}"
        )
    # The route and the unit, above the verdicts a reader will quote. Both are in every
    # row; neither belongs only in the JSONL, which is where a footnote would go unread.
    route_line = span_route_line(agg)
    if route_line:
        lines.append(route_line)
    lines.append("unit of the `tokens` column: characters (row field `retained_unit`);"
                 " the budget axis is characters too — no tokenizer produced either number.")
    lines.append("")
    for compressor, entry in sorted(fr["compressors"].items()):
        clear = entry["clearing_budget"]
        verdict = (f"smallest preserving budget: {clear}" if clear is not None
                   else "no budget cleared the floor")
        extra = "" if compressor == RUNG4 else f" | beats {RUNG4} at matched budget: {entry['beats_baseline']}"
        lines.append(f"{compressor}: {verdict}{extra}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _http_summariser(url: str) -> Callable[[str, int], str]:
    """A secondary-model summariser over one POST, for the frontier run only.

    Returned rather than constructed at import so importing this module never
    implies a second engine is up.
    """
    import urllib.request

    def _summarise(observation: str, budget_chars: int) -> str:
        body = json.dumps({
            "prompt": (
                "Summarise the tool result below for an agent that has to decide "
                f"its next tool call from your summary alone. At most {budget_chars} "
                f"characters.\n\n{observation[:60000]}\n\nSummary:"
            ),
            "max_tokens": max(16, budget_chars // 3),
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choice = (payload.get("choices") or [{}])[0]
        return (choice.get("text") or "").strip()

    return _summarise


# ---------------------------------------------------------------------------
# Publishing the note (#4508): a run record in, tracked repo content out
# ---------------------------------------------------------------------------
#
# The six fields the owed entry has asked for since #3967 all existed already, but only
# inside `score`'s stdout and its `--out` JSONL. A terminal scroll is not a publication:
# the smallest-preserving-budget question has been carried for five items because nothing
# turned a run into a file anyone could cite. This section is that turn, and it is a
# renderer rather than a transcription so the note cannot be hand-adjusted into saying
# something the run did not measure.

#: The note's home. Derived from THIS file so `report` cannot publish into a checkout
#: other than the one it runs from, and because `eval/measurements/` is the tracked
#: measurements directory the corpus already landed in — root `.gitignore` ignores
#: `eval/baselines/*` but not this path, which is what the harvest usage above is about.
NOTE_DIR = Path(__file__).resolve().parent / "measurements"

#: The name the owed entry has asked for, dated at publish time.
NOTE_STEM = "observation-compression"

#: The readout SHAPE behind every published figure, spelled on each delta line. The shape
#: is what licenses the number: `prompt_logprobs: 0` asks for ONE entry per scored
#: position and that entry is the prompt's own token, so the KL column is null by
#: construction (`score_row` records `kl_first_position = None` for exactly that reason)
#: and the direction of the effect lives in ΔCE alone.
READOUT_SHAPE = "prompt_logprobs: 0, top_k=0, kl_first_position null"

#: ...and what that shape does NOT license. vLLM returns the entry for the token it
#: actually served, whose `rank` is the CANDIDATE's rank and not the served token's, so a
#: numeric read off one request is a property of that request. Publishing one as a stable
#: per-request quantity is the mistake #4395's clause 4 exists to keep out of a note.
NO_STABLE_LOGPROB_LINE = (
    "no per-request numeric logprob is stable on this route: `prompt_logprobs: 0` "
    "returns the served token's own entry, whose `rank` is the candidate's, so "
    "`kl_first_position` is null on every row and only the ΔCE averages and the action "
    "agreement rates are published as figures")

#: The explicit negative the item demands when nothing clears the floor, in the words the
#: item asked for. An empty frontier table reads as "not measured"; this reads as a
#: finding, which is what it is.
NO_PRESERVING_BUDGET_LINE = "no budget beats production's rung-4 at a matched budget"

#: A bucket the run never reported. Never rendered as `0`: a 0 claims the harvest skipped
#: nothing, which is the fiction these four buckets exist to prevent — and a 0-byte
#: `harvest-2026-11-16.log` is this repo's own example of a tally that does not exist.
#: The label's exact text is what the tests count, so keep it out of prose.
SKIP_GAP_LABEL = "GAP (no count reported by the harvest run)"

#: A K=0 cell's KL, printed as the null it is rather than as `0.0000`, which would read
#: as a perfect match between the arms' distributions.
NO_KL_CELL = "n/a"

#: Text for a quantity this run did not record at all, distinct from the skip gap above.
NOT_RECORDED = "not recorded"


class ReportRefused(ValueError):
    """A figure the note would publish cannot be measured, so no note is written.

    The CLI turns this into a printed refusal and exit 2. A note is a durable artefact
    that later items will cite, and this module's own defect family is an absent
    measurement arriving as a clean number — invalid rows excluded from every aggregate,
    a null KL printed as null, a skip bucket never printed as 0. A refusal that still left
    a file on disk would be that defect with a date in its name.
    """


def _fig(value, spec: str = "{}") -> str:
    """One aggregate value, or the null marker — never a bare `None` in prose."""
    return NO_KL_CELL if value is None else spec.format(value)


def load_run_records(path) -> list[dict]:
    """The records `score --out` wrote: one scored arm per JSON line."""
    src = Path(path)
    if not src.is_file():
        raise ReportRefused(f"no run record at {src}; the note publishes a scored run")
    out = [json.loads(line) for line in
           src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not out:
        raise ReportRefused(f"{src} holds no records, so there is no run to publish")
    return out


def load_skips(spec) -> dict:
    """`HarvestSkips.as_dict()` read back from a file; {} when no file was passed.

    `harvest` prints its four buckets and writes none of them anywhere, so the only route
    for a tally into a note is a JSON object the operator saved. Omitting it is legal and
    prints four named gaps; NAMING a file that is not there is a typo, and a typo that
    quietly published gaps would hide the tally the note is asking for.
    """
    if not spec:
        return {}
    src = Path(spec)
    if not src.is_file():
        raise ReportRefused(
            f"no harvest tally at {src}: pass --skips with a HarvestSkips.as_dict() "
            "object, or omit the flag and let the note print the gap")
    got = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(got, dict):
        raise ReportRefused(f"{src} is not a JSON object of bucket -> count")
    return got


def no_op_counts(records: Sequence[dict]) -> dict:
    """Per (compressor, budget): rows whose compressed arm kept every single byte.

    `tokens_retained_pct >= 100.0` is the whole test, and it matters because a no-op row's
    ΔCE and agreement are the FULL observation scored twice: a cell with many of them is a
    budget nobody could tell apart from production. #4395 found exactly one such row among
    the 48, and its `budget < min_chars` reading is why the count is per budget — rung 4
    with production's `min_chars=4096` truncates at 0.10 where 0.50 need not.
    """
    out: dict[tuple[str, float], dict] = {}
    for rec in records:
        key = (str(rec.get("compressor")), float(rec.get("budget") or 0.0))
        cell = out.setdefault(key, {"rows": 0, "no_op": 0, "unknown": 0})
        cell["rows"] += 1
        pct = rec.get("tokens_retained_pct")
        if pct is None:
            cell["unknown"] += 1
        elif float(pct) >= 100.0:
            cell["no_op"] += 1
    return out


def render_note(records: Sequence[dict], *, corpus, skips: dict | None = None,
                floor: float, date: str, run_path=None) -> str:
    """The note's full text, from the records a run produced and nothing else.

    `floor` is the full-arm self-consistency spread from the frontier run's same-arm
    repeats — the reason `preserves the action` is a comparison and not an adjective — and
    it is printed wherever it decides a line. Nothing here measures: the numbers come from
    `aggregate()` and `frontier()`, the two functions `score`'s stdout already uses, so
    the published note and the terminal output cannot become two accounts of one run.
    """
    src = Path(corpus)
    if not src.is_file():
        raise ReportRefused(
            f"no corpus at {src}: a published row id has to name the file it came from, "
            "and two corpora sit in the eval output directory")
    size = src.stat().st_size
    agg = aggregate(records)
    front = frontier(agg, floor=floor).get("compressors", {})
    noop = no_op_counts(records)

    # Clause 5's refusal, checked before a single line is built: the rung-4 mean is the
    # baseline every other arm is ranked against, so a cell that cannot say how many of
    # its rows were untouched is a baseline over an unknown population — and the
    # ambiguity cannot be resolved downstream, because downstream is this note.
    for (compressor, budget), cell in sorted(noop.items()):
        if compressor == RUNG4 and cell["unknown"]:
            raise ReportRefused(
                f"{RUNG4} at budget {budget:.2f} has {cell['unknown']} row(s) with no "
                "`tokens_retained_pct`, so its no-op count is unknown and its mean cannot "
                "be published")

    row_ids = sorted({str(r["row_id"]) for r in records if r.get("row_id")})
    routes = sorted({str(r["engine_route"]) for r in records if r.get("engine_route")})
    counts = skips or {}

    lines: list[str] = [
        f"# Observation compression: the budget frontier — {date}",
        "",
        f"Published from `{run_path or 'a run record given in memory'}`: {len(records)} "
        f"scored records over {len(row_ids)} pinned row(s), "
        f"{len(COMPRESSORS)} compressor(s) x {len(BUDGETS)} budget(s), at an agreement "
        f"floor of {floor}. Every figure below is `aggregate()`'s reading of those "
        "records — the same call `score` prints from — so re-running `report` on the same "
        "JSONL reproduces this file byte for byte.",
        "",
        f"- Corpus: `{src}` — {size:,} bytes",
        f"- Engine route(s): {', '.join(f'`{r}`' for r in routes) or NOT_RECORDED}",
        f"- Readout shape: {READOUT_SHAPE}",
        f"- {NO_STABLE_LOGPROB_LINE}",
        "",
        f"## Pinned rows ({len(row_ids)})",
        "",
        "Each id the run scored, against the corpus file it was cut from and that file's",
        "byte size. An id with no file named beside it is not re-checkable, and both",
        "numbers together are what disambiguate the corpora that share a directory.",
        "",
        "| row id | corpus | corpus bytes |",
        "| --- | --- | --- |",
    ]
    for rid in row_ids:
        lines.append(f"| `{rid}` | `{src}` | {size:,} |")

    lines += ["", "## Harvest skips", "",
              "The four buckets `HarvestSkips.as_dict()` prints, re-derived by running",
              "`harvest` against the sessions directory on the publish date. A bucket with",
              "no tally prints as a named gap and never as 0, because 0 is the claim that",
              "the harvest skipped nothing.", ""]
    for reason in HARVEST_SKIP_REASONS:
        value = counts.get(reason)
        lines.append(f"- {reason} {_fig(value, '{}') if value is not None else SKIP_GAP_LABEL}")

    route_line = span_route_line(agg)
    lines += ["", "## Span alignment routes", "",
              route_line or f"span routes over valid rows: {NOT_RECORDED} (no valid rows)",
              "",
              "The route decided which window was scored; a cell scored by the estimated",
              "route is the weaker measurement, which is why the tally is published above",
              "the verdicts and not in a footnote."]

    lines += ["", "## Frontier", "",
              "`preserves the action` means that cell's next-action agreement against the",
              f"full arm reached the floor {floor}. Budgets are matched per arm by chars, so",
              f"`beats {RUNG4}` is the matched-budget question: no worse agreement at every",
              "shared budget, and on a tie a better ΔCE and no more of the observation kept.",
              ""]
    for compressor in COMPRESSORS:
        entry = front.get(compressor)
        if entry is None:
            lines.append(f"- {compressor}: {NOT_RECORDED} (no records in this run)")
            continue
        clear = entry.get("clearing_budget")
        lines.append(f"- {compressor}: smallest preserving budget: {clear:.2f}"
                     if clear is not None
                     else f"- {compressor}: no budget reached the floor {floor}")
    beaten = False
    for compressor in COMPRESSORS:
        entry = front.get(compressor)
        if entry is None or compressor == RUNG4:
            continue
        beats = bool(entry.get("beats_baseline"))
        beaten = beaten or beats
        lines.append(f"- {compressor}: beats {RUNG4} at matched budget: {beats}")
    if not beaten:
        lines.append(f"- {NO_PRESERVING_BUDGET_LINE}")

    lines += ["", "## Per-budget cells", "",
              "`ΔCE mean` is the candidate arm's action-span cross-entropy minus the full",
              "arm's, nats per span token; positive means the compressed observation made",
              "the recorded action harder. `no-op rows` counts the rows whose compressed",
              "arm kept every byte — there the arm measured the observation it was handed,",
              "not a compression — and `declines` and `undecidable` are the rows with no",
              "comparable decision, which are excluded from the agreement rate.", ""]
    for compressor in COMPRESSORS:
        for budget in BUDGETS:
            cell = agg.get((compressor, budget))
            if cell is None:
                continue
            tally = noop.get((compressor, budget)) or {"rows": 0, "no_op": 0, "unknown": 0}
            no_op = (f"{tally['no_op']}/{tally['rows']}"
                     if not tally["unknown"] else NO_KL_CELL)
            lines.append(
                f"- {compressor} budget {budget:.2f}: "
                f"ΔCE mean {_fig(cell.get('mean_delta_ce_nats'), '{:+.4f}')} nats"
                f", agreement {_fig(cell.get('agreement_rate'), '{:.3f}')}"
                f", valid {cell['valid_rows']}/{cell['rows']}"
                f", invalid {cell['invalid_rows']}"
                f", declines {cell['declines']}, undecidable {cell['undecidable']}"
                f", no-op rows {no_op}"
                f", KL@1 {_fig(cell.get('mean_kl_first_position'), '{:.4f}')}"
                f", mean retained {_fig(cell.get('mean_tokens_retained'), '{:.0f}')} chars"
                f" ({READOUT_SHAPE})")

    lines += ["", "## Reproduce", "", "```",
              "eval/run_observation_compression_eval.py score --corpus <corpus.yaml> \\",
              "    --compressor <name> --budget <frac> --out <run.jsonl>   # once per arm",
              "eval/run_observation_compression_eval.py harvest --out <corpus.yaml> \\",
              "    # prints the scanned row count and the four skip buckets; save",
              "    # HarvestSkips.as_dict() as <skips.json> for the line below",
              "eval/run_observation_compression_eval.py report --run <run.jsonl> \\",
              f"    --corpus <corpus.yaml> --skips <skips.json> --floor {floor}",
              "```", ""]
    return "\n".join(lines)


def write_note(run, corpus, *, out_dir=None, skips=None, floor: float,
               date: str = "") -> Path:
    """Render the note and put it in `eval/measurements/`, returning the path written.

    The text is complete before anything touches disk, so a `ReportRefused` leaves no
    half-note behind. `date` defaults to today in UTC — the note's filename is a citation
    key, and a naive local stamp would be read as UTC by every later reader.
    """
    records = load_run_records(run)
    counts = load_skips(skips)
    stamp = date or datetime.now(timezone.utc).date().isoformat()
    text = render_note(records, corpus=corpus, skips=counts, floor=floor, date=stamp,
                       run_path=Path(run))
    target = Path(out_dir) if out_dir else NOTE_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{NOTE_STEM}-{stamp}.md"
    path.write_text(text, encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="write the pinned corpus from real sessions")
    h.add_argument("--sessions-dir", default="")
    h.add_argument("--min-bytes", type=int, default=MIN_OBSERVATION_BYTES)
    h.add_argument("--max-rows", type=int, default=0)
    h.add_argument("--out", required=True)

    s = sub.add_parser("score", help="score arms against a live engine")
    s.add_argument("--corpus", required=True)
    s.add_argument("--compressor", choices=COMPRESSORS, action="append")
    s.add_argument("--budget", type=float, action="append")
    s.add_argument("--engine-url", default=DEFAULT_BASE_URL)
    s.add_argument("--model", default="")
    s.add_argument("--summary-url", default="",
                   help="POST target for the summary arm; required by --compressor summary")
    s.add_argument("--session-id", default="",
                   help="pass one to score the SPILL-BACKED rung-4 arm (#1499 shape)")
    s.add_argument("--floor", type=float, default=0.0,
                   help="agreement floor from the full-arm self-consistency repeats")
    s.add_argument("--out", required=True)

    rp = sub.add_parser("report",
                        help="publish eval/measurements/observation-compression-<date>.md "
                             "from a score run record")
    rp.add_argument("--run", required=True,
                    help="the JSONL `score --out` wrote")
    rp.add_argument("--corpus", required=True,
                    help="the corpus YAML that run scored; the note prints its path and "
                         "byte size beside every published row id")
    rp.add_argument("--skips", default="",
                    help="a JSON HarvestSkips.as_dict() from the harvest run; omit it and "
                         "the note prints a named gap per bucket")
    rp.add_argument("--floor", type=float, required=True,
                    help="agreement floor from the full-arm self-consistency repeats; "
                         "required, because a silently-zero floor would publish a "
                         "compressor that agrees on nothing as preserving the action")
    rp.add_argument("--date", default="", help="YYYY-MM-DD stamp; defaults to today (UTC)")
    rp.add_argument("--out-dir", default="", help=f"default {NOTE_DIR}")

    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "report":
        try:
            printed = write_note(
                Path(args.run), Path(args.corpus),
                out_dir=Path(args.out_dir) if args.out_dir else None,
                skips=Path(args.skips) if args.skips else None,
                floor=args.floor, date=args.date)
        except ReportRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(printed)
        return 0

    if args.cmd == "harvest":
        got = harvest(
            Path(args.sessions_dir) if args.sessions_dir else None,
            min_bytes=args.min_bytes, max_rows=args.max_rows,
        )
        rows = got.rows
        # The item's bar is a published number, never a quietly lowered one: the
        # count prints even when it is short of 40, because "fewer than 40 exist at
        # this band" IS the result the owed check asks for.
        out = save_corpus(rows, Path(args.out))
        print(f"harvested {len(rows)} rows (min {args.min_bytes} bytes) -> {out}")
        # Every scanned row is accounted for by name, so a yield that is short says
        # WHICH requirement cost it. The tally is printed as an identity and
        # self-polices: a broken bucket prints rather than passing.
        print(f"scanned {got.scanned} tool rows = {len(rows)} rows + "
              + " + ".join(f"{k} {v}" for k, v in got.skips.as_dict().items()))
        if got.skips.total + len(rows) != got.scanned:
            print(f"SKIP TALLY BROKEN: {got.skips.total + len(rows)} accounted "
                  f"against {got.scanned} scanned")
        print(f"sessions not scanned at all: {got.sessions_unscanned} "
              "(no spill directory beside the record, or unreadable record)")
        if len(rows) < 40:
            print(f"SHORT OF THE ITEM'S BAR: {len(rows)} rows at this band; "
                  "publish the count rather than lowering --min-bytes")
        return 0

    rows = load_corpus(Path(args.corpus))
    if not rows:
        print(f"corpus {args.corpus} has no rows", file=sys.stderr)
        return 2
    compressors = args.compressor or list(COMPRESSORS)
    budgets = args.budget or list(BUDGETS)
    engine = LogprobClient(args.engine_url, model=args.model)
    summariser = _http_summariser(args.summary_url) if args.summary_url else None
    if SUMMARY in compressors and summariser is None:
        print("the summary arm needs --summary-url; skipping it is a result, "
              "scoring it as something else is not", file=sys.stderr)
        return 2
    records = [
        score_row(row, engine, compressor=c, budget=b, summariser=summariser,
                  session_id=args.session_id)
        for row in rows for c in compressors for b in budgets
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    agg = aggregate(records)
    print(render_report(agg, floor=args.floor))
    print(f"\n{len(records)} rows of records -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
