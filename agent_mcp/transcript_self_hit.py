"""Transcript self-hits: a retrieved session note that is this prompt's own echo (#1511).

`app/post_capture.py::_export_session_markdown` writes every chat transcript into
`_pipeline/vault-derived/sessions/`, which qmd indexes as the `sessions`
collection and prefetch's vault leg searches. So a prompt that has been sent
before retrieves the transcript that holds it — and that transcript holds the
previous run's answer and tool results right under it. On 2026-09-24/25 the
same "E2E harness check … Read app/harness/client.py …" probe was run five
times, and from the second run on the hybrid leg put the previous run's
transcript at score 1.00 on the top of `<vault-context>`: a pass on the re-run
was then equally consistent with "the model read the file" and "the model read
last night's answer to reading the file".

Two cases are a self-hit, and only these two:

- ``own_session`` — the note IS the current session's own export. It is written
  after the first turn and re-written as the chat grows, so from the second turn
  on a session could retrieve itself.
- ``verbatim`` — the note is another session whose USER turn holds this prompt
  near-verbatim: at least ``SELF_HIT_MIN_CONTAINMENT`` of the prompt's word
  3-shingles, and ``SELF_HIT_MIN_WORD_CONTAINMENT`` of its distinct words, appear
  in one user turn of the note. A related conversation about
  the same topic shares words, not runs of them, so it stays; a prompt pasted
  again is caught however it was re-wrapped.

A prompt under ``SELF_HIT_MIN_WORDS`` words is never judged ``verbatim``: "what
did we decide about qmd" legitimately recalls the session where it was asked,
and three words cannot tell a repeat from a topic. Only notes in the
``sessions`` collection are ever judged — a backlog item quoting the prompt is a
document about it, not an echo of it.

Stdlib plus yaml, and it never raises: a note it cannot read is judged on its snippet
alone, and anything unexpected reads as "not a self-hit", because a wrongly kept
hit costs one stale line while a wrongly dropped one costs a real memory.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

SESSIONS_COLLECTION = "sessions"
SELF_HIT_MIN_WORDS = 6
SELF_HIT_MIN_CONTAINMENT = 0.8
# ...and every distinct word of it but one in twenty. A reused template with a
# new argument ("pull the youtube transcript and give me the highlights <url>",
# a different video each time) shares 13 of its 14 shingles with the last run
# but not the id that makes it a different request; the word test is what keeps
# that prior session, which answered a different question.
SELF_HIT_MIN_WORD_CONTAINMENT = 0.95
SHINGLE = 3
# A transcript export caps each user turn at 600 chars and each note is a few KB
# to a few hundred; the user turns sit throughout, so read enough to see them all
# for any note a chat produces without letting one pathological file cost the
# prefetch budget.
_READ_MAX_BYTES = 512 * 1024
_TURN_START = re.compile(r"^(user|lloyd|tool_call): |^  → ", re.M)
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def note_path(file: str) -> str | None:
    """Collection-relative path of a `sessions` hit, or None for any other hit.

    Accepts qmd's `qmd://sessions/<rel>` and the bare `sessions/<rel>` form.
    """
    f = (file or "").strip()
    f = f.removeprefix("qmd://")
    head, _, rest = f.partition("/")
    if head != SESSIONS_COLLECTION or not rest:
        return None
    return rest


def is_transcript(file: str) -> bool:
    return note_path(file) is not None


@lru_cache(maxsize=1)
def _sessions_root() -> Path | None:
    """Where qmd's `sessions` collection lives, read from qmd's own config.

    Not `app.paths.VAULT_SESSIONS_DIR`: that follows the checkout's data root, so
    from a worktree it names an empty tree, while the hits come from the daemon's
    index of production's. The collection map is the one place that says which
    directory a `qmd://sessions/…` hit was read from.
    """
    cfg = Path.home() / ".config" / "qmd" / "index.yml"
    try:
        import yaml
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        path = ((data.get("collections") or {}).get(SESSIONS_COLLECTION) or {}).get("path")
        return Path(path) if path else None
    except Exception:  # noqa: BLE001 — no config: judge on the snippet
        return None


def user_turns(note_text: str) -> list[str]:
    """The `user: …` turns of an exported transcript, continuation lines kept."""
    out: list[str] = []
    marks = list(_TURN_START.finditer(note_text or ""))
    for i, m in enumerate(marks):
        if m.group(1) != "user":
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(note_text)
        out.append(note_text[m.end():end])
    return out


def _note_user_turns(rel: str) -> list[str] | None:
    root = _sessions_root()
    if root is None:
        return None
    try:
        p = (root / rel).resolve()
        if root.resolve() not in p.parents:
            return None
        with open(p, "rb") as fh:
            raw = fh.read(_READ_MAX_BYTES)
        return user_turns(raw.decode("utf-8", errors="replace"))
    except OSError:
        return None


def containment(query: str, candidate: str) -> float:
    """Share of the query's word 3-shingles present in `candidate` (0..1)."""
    q = _shingles(_words(query))
    if not q:
        return 0.0
    return len(q & _shingles(_words(candidate))) / len(q)


def word_containment(query: str, candidate: str) -> float:
    """Share of the query's distinct words present in `candidate` (0..1)."""
    q = set(_words(query))
    if not q:
        return 0.0
    return len(q & set(_words(candidate))) / len(q)


def echoes(query: str, candidate: str) -> bool:
    """Does `candidate` hold `query` near-verbatim? Both tests, see the constants."""
    return (containment(query, candidate) >= SELF_HIT_MIN_CONTAINMENT
            and word_containment(query, candidate) >= SELF_HIT_MIN_WORD_CONTAINMENT)


def self_hit_reason(query: str, hit: dict, session_id: str | None = None,
                    read_note: bool = True) -> str | None:
    """"own_session", "verbatim", or None for a hit that is not a self-hit."""
    try:
        rel = note_path(str(hit.get("file") or ""))
        if rel is None:
            return None
        if session_id:
            stem = Path(rel).stem
            safe = session_id.replace("/", "--")[:30]
            if stem in (session_id, safe):
                return "own_session"
        if len(_words(query)) < SELF_HIT_MIN_WORDS:
            return None
        candidates = user_turns(str(hit.get("snippet") or ""))
        # A snippet can open mid-turn, before any `user:` marker: judge it whole.
        candidates.append(str(hit.get("snippet") or ""))
        if read_note:
            candidates.extend(_note_user_turns(rel) or [])
        if any(echoes(query, c) for c in candidates):
            return "verbatim"
        return None
    except Exception:  # noqa: BLE001 — never lose a turn to this check
        return None


def drop_self_hits(query: str, hits: list[dict], session_id: str | None = None,
                   read_note: bool = True) -> tuple[list[dict], list[dict]]:
    """(kept, dropped) — `dropped` rows carry their `self_hit` reason."""
    kept: list[dict] = []
    dropped: list[dict] = []
    for h in hits or []:
        reason = self_hit_reason(query, h, session_id, read_note=read_note)
        if reason:
            dropped.append({**h, "self_hit": reason})
        else:
            kept.append(h)
    return kept, dropped
