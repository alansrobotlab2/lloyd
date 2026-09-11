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

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]

#: The retrieval non-regression threshold the triaged items hard-code. It is
#: recorded here so the code that replaces it can be pointed at what it replaced.
RETRIEVAL_GATE_HARDCODE = 0.95

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

#: Presence source for `USER.md` / `MEMORY.md` bullets: loaded into every turn,
#: so "present" carries no information on its own and must be weighted.
ALWAYS_IN_FORCE = "always_in_force:system_prompt"

#: Skill presence proxy until #435 ships per-injection telemetry: derived from
#: `skills_read` tool-call payloads in `event_logs/*.events.jsonl` plus the
#: `<skill name=…>` blocks visible in the turn's own context.
SKILL_PRESENCE_PROXY = "proxy:skills_read+injected_context (#435 pending)"

#: The #435 dependency, in the words a consolidator reads.
SKILL_PRESENCE_NOTE = (
    "Not per-injection telemetry: #435 (per-skill injection events) is still "
    "draft, so presence here is skills_read payloads + injected <skill name> "
    "blocks. Treat sub-90% coverage as unresolved, not as a miss."
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
    as health."""
    d = base / "sessions"
    return d.is_dir() and next(d.glob("*.json"), None) is not None


def lloyd_root() -> Path:
    """The checkout whose *logs* this measures.

    `sessions/`, `event_logs/` and `eval/baselines/nightly-*.json` are all
    untracked, so an automod round running from a worktree would otherwise look
    at an empty tree and report a perfect uptake score over zero turns. Falling
    back to the live checkout is what makes the probe mean the same thing from
    either place; `LLOYD_ROOT` overrides for tests.
    """
    env = os.environ.get("LLOYD_ROOT")
    if env:
        return Path(env).expanduser()
    if _has_sessions(REPO):
        return REPO
    return Path.home() / "lloyd"


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
_VAULT_CTX_ITEM = re.compile(r"^- \*\*(.{5,120}?)\*\*", re.M)


def _injected_skills(text: str) -> list[str]:
    return _SKILL_INJECTION.findall(text)


def _vault_context_titles(text: str) -> list[str]:
    block = text.split("<vault-context>", 1)
    if len(block) < 2:
        return []
    return [t.strip() for t in _VAULT_CTX_ITEM.findall(block[1].split("</vault-context>", 1)[0])]


def human_turns(root: Path | str | None = None, days: int = 30) -> list[Turn]:
    """Human-authored user turns from the last `days` of session JSONs.

    Filtered three ways, each one earned by a shape actually seen in the data:
    interactive sessions only (session `source` unset), user messages with no
    inject `source`, and no bracketed scheduler/inner-voice prefix. Ordered by
    session name then position, so `turn_id` is reproducible.
    """
    root = Path(root) if root else lloyd_root()
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    turns: list[Turn] = []
    for path in sorted((root / "sessions").glob("*.json")):
        if path.stat().st_mtime < cutoff:
            continue
        try:
            doc = json.loads(path.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or doc.get("source"):
            continue
        session = path.stem
        if _synthetic_session(session):
            continue
        prev: str | None = None
        ordinal = 0
        for msg in doc.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = _text_of(msg.get("content"))
            if role == "assistant":
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
                turns.append(Turn(
                    session=session, ts=msg.get("timestamp"), user_text=text.strip(),
                    ordinal=ordinal, session_source=None, prev_assistant=prev,
                    injected_skills=_injected_skills(blob),
                    vault_context=_vault_context_titles(blob),
                ))
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


def _post_secondary(payload: dict[str, Any]) -> dict[str, Any]:
    from app.secondary_models import _endpoint  # the tree's own resolver

    url, resolved = _endpoint()
    body = dict(payload)
    body["model"] = resolved
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def classify_dispute(prev_assistant: str | None, user_text: str, *,
                     transport: Callable[[dict], dict] | None = None) -> bool | None:
    """One dispute verdict from the secondary engine, or `None`.

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
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user", "content": _classify_user(prev_assistant, user_text)},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = post(payload)
    except (OSError, urllib.error.URLError, TimeoutError):
        return None
    except Exception:  # noqa: BLE001 - a transport that dies oddly is still "no answer"
        return None
    try:
        content = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, AttributeError, TypeError):
        return None
    return _parse_verdict(content)


# ------------------------------------------------------------ label corpus --

LABEL_GLOB = "eval/uptake/labels/hand-*.json"
LABELER = "hand:alan-turns-2026-09-11"


def labels_path(root: Path | str | None = None) -> Path | None:
    root = Path(root) if root else REPO
    files = sorted(root.glob(LABEL_GLOB))
    return files[-1] if files else None


def load_labels(root: Path | str | None = None) -> list[dict[str, Any]]:
    path = labels_path(root)
    if not path:
        return []
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    items = doc.get("items") if isinstance(doc, dict) else doc
    return [i for i in (items or []) if isinstance(i, dict) and "turn_id" in i]


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


def memory_entries(root: Path | str | None = None,
                   docs: Sequence[str] = ("lloyd/USER.md", "lloyd/MEMORY.md")) -> list[Entry]:
    """Durable memory bullets, one `Entry` each.

    Read from the live vault when present, else the checkout — a top-level
    bullet is what actually gets loaded into the system prompt, which is the
    surface the item is about.
    """
    root = Path(root) if root else lloyd_root()
    vault = Path.home() / "obsidian"
    out: list[Entry] = []
    seen: set[str] = set()
    for doc in docs:
        for base in (vault, root):
            path = base / doc
            if not path.is_file():
                continue
            for line in path.read_text(errors="replace").splitlines():
                m = _MEMORY_DOC_RE.match(line.rstrip())
                if not m:
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


def skills_read_by_session(root: Path | str | None = None) -> dict[str, set[str]]:
    """Skill names each session explicitly read, from the turn event log.

    The events are written by the agent loop, a different process, and the
    payload is only half-structured: `data.args` is a JSON *string*, so the
    skill name has to be parsed back out of it. A log line that does not
    parse is skipped rather than fatal — these files have unparseable lines in
    them today. Note this is a proxy for #435's per-injection telemetry, not
    that telemetry: it says the model asked for a skill, not that one was
    force-injected.
    """
    root = Path(root) if root else lloyd_root()
    out: dict[str, set[str]] = {}
    for path in root.glob(_EVENT_GLOB):
        try:
            handle = path.open(errors="replace")
        except OSError:
            continue
        with handle:
            for line in handle:
                if "skills_read" not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = rec.get("data")
                if not isinstance(data, dict) or data.get("name") != "skills_read":
                    continue
                args = data.get("args")
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    continue
                name = args.get("name") if isinstance(args, dict) else None
                session = rec.get("session_id")
                if name and session:
                    out.setdefault(str(session), set()).add(str(name))
    return out


def active_skill_names(vault: Path | None = None) -> list[str]:
    vault = vault or (Path.home() / "obsidian")
    return sorted(p.parent.name for p in (vault / "skills").glob("*/SKILL.md"))


def _row(entry: str, presence_source: str, present: int, disputes: int,
         weighted: float, overlap_max: float, extra: dict | None = None) -> dict:
    row = {
        "entry": entry,
        "presence_source": presence_source,
        "present_in_turns": present,
        "disputes": disputes,
        "weighted_disputes": round(weighted, 4),
        "overlap_max": round(overlap_max, 4),
        "dispute_rate": round(disputes / present, 4) if present else 0.0,
    }
    if extra:
        row.update(extra)
    return row


def build_uptake_table(
    turns: Sequence[Turn],
    dispute_flags: dict[int, Any],
    memory_entries: Sequence[Entry] = (),
    skills_read: dict[str, set[str]] | None = None,
    active_skills: Sequence[str] = (),
) -> dict[str, Any]:
    """Per-entry uptake table. Measurement only — nothing here prunes.

    `dispute_flags` maps `Turn.ordinal` -> verdict (`True` / `False` / `None`
    for "the classifier did not answer"). `None` verdicts are counted in
    `unresolved_turns` rather than folded into the negative class, so a stalled
    engine shows up as missing evidence instead of a clean bill of health.
    """
    skills_read = skills_read or {}
    flagged = {k: v for k, v in dispute_flags.items() if v is not None}
    disputed = [t for t in turns if flagged.get(t.ordinal) is True]
    unresolved = sum(1 for t in turns if dispute_flags.get(t.ordinal) is None)

    # --- skills: evidence-bound presence, never "all skills everywhere" ------
    skill_present: dict[str, list[Turn]] = {}

    def note_skill(name: str, turn: Turn) -> None:
        bucket = skill_present.setdefault(name, [])
        if turn not in bucket:
            bucket.append(turn)

    for t in turns:
        for name in t.injected_skills:
            note_skill(name, t)
        for name in skills_read.get(t.session, ()):
            note_skill(name, t)

    entries: list[dict[str, Any]] = []
    for name in sorted(skill_present):
        present = skill_present[name]
        hits = [t for t in present if t in disputed]
        weight = sum(max(overlap(name, t.user_text), 0.05) for t in hits)
        entries.append(_row(
            f"skill:{name}", SKILL_PRESENCE_PROXY, len(present), len(hits), weight,
            max((overlap(name, t.user_text) for t in hits), default=0.0),
            {"kind": "skill", "name": name,
             "present_turn_ids": [t.turn_id for t in present][:20]},
        ))

    # --- memory: always in force, so presence is uninformative by design ----
    for e in memory_entries:
        hits = disputed
        weight = sum(overlap(e.text, t.user_text) for t in hits)
        entries.append(_row(
            e.key, ALWAYS_IN_FORCE, len(turns), len(hits), weight,
            max((overlap(e.text, t.user_text) for t in hits), default=0.0),
            {"kind": "memory_entry", "source_doc": e.source,
             "text": e.text[:300],
             "note": "present in every prompt: dispute_rate is an upper bound, "
                     "weighted_disputes is the usable signal"},
        ))

    # --- notes/facts that arrived through prefetch --------------------------
    note_present: dict[str, list[Turn]] = {}
    for t in turns:
        for title in t.vault_context:
            bucket = note_present.setdefault(f"note:{title}", [])
            if t not in bucket:
                bucket.append(t)
    for key in sorted(note_present):
        present = note_present[key]
        hits = [t for t in present if t in disputed]
        title = key.split(":", 1)[1]
        entries.append(_row(
            key, "prefetch:vault_context", len(present), len(hits),
            sum(overlap(title, t.user_text) for t in hits),
            max((overlap(title, t.user_text) for t in hits), default=0.0),
            {"kind": "note", "presence_turn_ids": [t.turn_id for t in present][:20]},
        ))

    cov_skills = sorted(set(skill_present) & set(active_skills or ()))
    coverage = {
        "user_md_entries": {
            "covered": len(memory_entries),
            "total": len(memory_entries),
            "ratio": 1.0 if memory_entries else 0.0,
            "presence_source": ALWAYS_IN_FORCE,
        },
        "active_skills": {
            "covered": len(cov_skills),
            "total": len(active_skills or ()),
            "ratio": (round(len(cov_skills) / len(active_skills), 4) if active_skills else 0.0),
            "presence_source": SKILL_PRESENCE_PROXY,
            "note": SKILL_PRESENCE_NOTE,
        },
        "notes_seen": len(note_present),
    }

    return {
        "probe_timestamp": _now(),
        "corpus": {
            "turns": len(turns), "disputes": len(disputed),
            "unresolved_turns": unresolved,
            "dispute_rate_over_turns": round(len(disputed) / len(turns), 4) if turns else 0.0,
        },
        "coverage": coverage,
        "entries": entries,
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


def retrieval_gate(baselines_dir: Path | str | None = None,
                   metrics: Sequence[str] = ("doc_hit_rate", "ndcg10")) -> dict[str, Any]:
    """A non-regression gate read from what the eval actually does at rest.

    Triage item #801 measured six identical-config nightly runs at
    0.95/0.90/0.90/0.90/0.95/0.85 doc_hit_rate — a 0.10 spread. Any item whose
    acceptance hard-codes `doc_hit_rate >= 0.95` is therefore decided by which
    night it ran, not by the change under test. This returns newest-value minus
    the observed spread of comparable nights, so the tolerance is measured.

    Comparability is by run shape (`limit`, `matches_production_defaults`), not
    by filename: folding a 40-query or rerank-on night into the spread would
    widen the band with somebody else's experiment.
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
    prod = [r for r in nightly if bool(r[2].get("matches_production_defaults"))]
    basis = prod or nightly or rows
    shapes: dict[Any, list] = {}
    for r in basis:
        shapes.setdefault((r[2].get("limit"),
                           bool(r[2].get("matches_production_defaults"))), []).append(r)
    shape, pool = max(shapes.items(), key=lambda kv: (len(kv[1]), kv[1][0][0]))
    newest_mtime, newest_path, newest_doc, newest_met = pool[0]

    out: dict[str, Any] = {
        "baselines_dir": str(d),
        "shape": {"limit": shape[0], "matches_production_defaults": shape[1]},
        "nights": len(pool),
        "latest": {"label": newest_doc.get("label") or newest_path.name,
                   "measured_at": newest_doc.get("ran_at") or newest_doc.get("measured_at"),
                   "path": str(newest_path)},
        "hardcoded_gate": RETRIEVAL_GATE_HARDCODE,
        "hardcoded_gate_would_have_failed_nights": 0,
    }
    for key in metrics:
        vals = [float(r[3][key]) for r in pool if isinstance(r[3].get(key), (int, float))]
        if not vals:
            continue
        latest = float(newest_met[key]) if isinstance(newest_met.get(key), (int, float)) else min(vals)
        band = max(vals) - min(vals)
        out[key] = {
            "latest": latest, "min": min(vals), "max": max(vals),
            "band": round(band, 4), "floor": round(latest - band, 4),
            "n_nights": len(vals),
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

def write_table(table: dict[str, Any], out_dir: Path | str | None = None,
                date: str | None = None, classifier: dict | None = None,
                stores: dict | None = None, extra: dict | None = None) -> Path:
    """Emit `eval/uptake/uptake-<date>.json`. Enforces the timestamp rule."""
    out = Path(out_dir) if out_dir else (REPO / "eval" / "uptake")
    out.mkdir(parents=True, exist_ok=True)
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc = dict(table)
    doc.setdefault("item", "552")
    doc.setdefault("probe_timestamp", _now())
    if classifier is not None:
        doc["classifier"] = {**classifier, "threshold": PRECISION_FLOOR}
    if stores:
        for name, block in stores.items():
            if not isinstance(block, dict):
                raise ValueError(f"stores.{name} must be a mapping")
            if any(isinstance(v, int) and not isinstance(v, bool) for v in block.values()) \
                    and not block.get("probed_at"):
                raise ValueError(f"stores.{name} carries a count with no probed_at")
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
