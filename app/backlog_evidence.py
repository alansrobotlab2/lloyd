"""Is a blocker's quoted gate evidence really in the artifact it names?

A *blocker* — an item tagged `spawned-by-autocode` + `blocker`, first line
"Blocks #N" — is the one thing an implement round may file of its own, and it is
the most privileged item on the board: the `a BLOCKER of #…` line of
`workers/sources/autotriage.py:542-544` tells the triage that "an implement round
deferred a clause of #… to this item" which "skips the depth gate and is taken
early", and `:1123-1124` holds the implement pool behind the blocked item while
one waits. Everything the loop does next is downstream of a sentence a round
wrote about a gate report.

`#3720` (filed 2026-11-03, `priority: high`, blocker of `#3609`) showed what that
buys. It named its source path explicitly —
`/home/alansrobotlab/.local/state/lloyd-automod/rounds/SM_20261103_035337/gate.json`
— and put seven strings in quotation marks "out of" it. `grep -Fc` finds none of
the seven in that file or in its sibling `gate.log` (re-run 2026-11-04 at HEAD
`86cb6be7`); the artifact says `"base": "e4e8b0a8…`, `"passed": 22907` and
`base_probe: … 0 already failing` instead, and one phrase the item quoted —
`full-suite failures that pass in isolation` — appears in no report line
`scripts/automod/gate.py` writes (`git grep -c "pass in isolation" --
scripts/automod/gate.py` → no match). The prompt already forbids it
(`workers/sources/autocode.py:174-181`: "copy any gate or ledger line you quote
out of that artifact **re-read from disk** … and name the path you copied from"),
and prose is what stopped it: one triage's refutation, or the pool moved.

This module turns that sentence into a measurement on the write path. It is
deliberately narrow, because the alternative — a check that refuses honest
blockers — is worse than the forgery it prevents. Two rulings carry that, both
measured off the real artifact:

* **A truthful quote is not a substring of the file the body names.** `gate.json`
  stores the rung's data as JSON (`"passed": 22907`) while the sentence a round
  would copy, `[FAIL] tests (1220.1s) all 5 failure(s) are new in this round (…)`,
  exists only in the SIBLING `gate.log`. So an absence is judged against **every
  readable file in the named round directory**, never against the one path the
  body happened to cite. This is also why a *ledger* path (`promotions.jsonl`) is
  left alone: a ledger has no rendered sibling to catch a legitimately-quoted
  line, so a miss there proves nothing.
* **A quote is only evidence when it could have been a line of the file.** A span
  with an ellipsis in it says "this text is cut"; a single bare token (a node id,
  a path, a sha) carries no phrasing to compare; a shell command is something the
  author wrote, not something the gate printed; a round directory the retention
  sweep reclaimed cannot be re-read at all. Each is read as **unverifiable, not
  absent**, and the write goes through. `#3720` is unambiguous because all seven
  of its multi-word quotes miss both of its files.

A backtick, not a quotation mark, is what marks the claim: this corpus writes
verbatim copies as inline code and its own prose in ordinary sentences, so prose
outside backticks is never checked here. The judgement is checkable by hand
either way (`grep -F '<quote>' <artifact>`), which is the property the prompt
asked a blocker's evidence to have.

Lives in `app/` beside the other backlog helpers so both board writers reach one
judgement; it imports only the standard library at module scope, and the one
automod path it needs (`state.ROUNDS_DIR`) is read from the module attribute at
call time, which is how the gate's own rungs stay movable by a test.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app.backlog_tags import BLOCKER_TAG, normalize_tags

logger = logging.getLogger(__name__)

#: `SM_20261103_035337` — the gate's round id (`scripts/automod/round.py`). A body
#: that names a round at all can be tested against what that round wrote, whether
#: it spelled the whole path, a `~/…` path, a bare `rounds/<id>/gate.json`, or
#: only the id.
_ROUND_ID = re.compile(r"\bSM_\d{8}_\d{6}\b")

#: One inline code span. A span cannot contain a newline, which is why an inline
#: span is the right grain: a quote that had to be a whole line of a report cannot
#: be smuggled across lines the way a fenced block can.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

_FENCE = re.compile(r"^\s*(?:```|~~~)")

#: A quote shorter than this is a value, not a line: `"passed": 22907` clears it,
#: a bare sha, node id or tag does not. Set low on purpose — every real gate line
#: this file has to catch is a sentence.
MIN_QUOTE_CHARS = 12

#: An ellipsis anywhere in the span means the author cut the text, so the span is
#: not one line of any file and a miss proves nothing.
ELLIPSES = ("…", "...", "⋯", "‥")

#: The verbs a body starts its own instructions with. A blocker's house shape is
#: "Check: `cd ~/lloyd && git grep -n …`", and a command is never in the report,
#: so judging one against the artifact would refuse an honest item for style.
_SHELL_VERBS = frozenset({
    "awk", "bash", "cat", "cd", "chmod", "chown", "cp", "curl", "diff", "docker",
    "echo", "export", "find", "git", "grep", "gh", "head", "hyprctl", "jq",
    "journalctl", "kill", "ls", "make", "mkdir", "mv", "node", "npm", "npx",
    "pgrep", "pkill", "pip", "pip3", "printf", "ps", "pytest", "python",
    "python3", "rg", "rm", "ruff", "sed", "sh", "sleep", "sort", "source",
    "sqlite3", "sudo", "supervisorctl", "systemctl", "tail", "tar", "tee",
    "timeout", "touch", "tree", "tsc", "uniq", "uv", "wc", "wget", "which",
    "xargs", "yt-dlp", "zsh",
})

#: The round directory holds the report, the log, the run spec and the base
#: baseline; all of them are kilobytes. The caps exist so one huge artifact cannot
#: make a tool call read a gigabyte.
_MAX_ARTIFACT_BYTES = 4_000_000
_MAX_ARTIFACTS = 32


def rounds_root() -> Path:
    """Where a round's report lives. Read at call time so a test can move it."""
    from scripts.automod import state as S
    return Path(S.ROUNDS_DIR)


def _paragraphs(body: str) -> list[str]:
    """The body cut into attribution units: blank-line blocks, fenced code gone.

    A quote is evidence *for* the artifact named near it, so the paragraph is the
    unit that carries the claim — `#3720`'s fabricated line sits on the line after
    the sentence that named `gate.json`, so a sentence-sized window missed it. A
    fenced block is dropped: a blocker's fences are commands it asks a reader to
    run, and none of them is in the report.
    """
    lines: list[str] = []
    fenced = False
    for line in body.split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
            lines.append("")
            continue
        lines.append("" if fenced else line)
    return [p for p in re.split(r"\n[ \t]*\n", "\n".join(lines)) if p.strip()]


def _is_verifiable(span: str) -> bool:
    """Could this span have been copied out of a file, unedited?"""
    quote = span.strip()
    if len(quote) < MIN_QUOTE_CHARS or " " not in quote:
        return False
    if any(e in quote for e in ELLIPSES):
        return False
    first = quote.split()[0].strip("`\"'([$(")
    if "/" in first or first.startswith("-") or first.lower() in _SHELL_VERBS:
        return False
    return "&&" not in quote and "|" not in quote


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _artifacts(round_dir: Path) -> list[tuple[Path, str]]:
    """Every readable file one round wrote, with its text.

    Swept, absent or unreadable, this returns the empty list, which the caller
    reads as "no verdict" rather than "absent from everything" — the retention
    sweep reclaims round directories, and an item citing an old round is not
    thereby a forgery.
    """
    try:
        paths = sorted(p for p in round_dir.iterdir() if p.is_file())
    except OSError:
        return []
    out: list[tuple[Path, str]] = []
    for path in paths[:_MAX_ARTIFACTS]:
        try:
            if path.stat().st_size > _MAX_ARTIFACT_BYTES:
                continue
            out.append((path, path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return out


def absent_quotes(body: str, *, rounds_dir: Path | None = None) -> list[dict]:
    """Quotes the body presents as copied out of a round's report that no such report holds.

    `[{quote, round_ids, searched}]`, one entry per absent quote. An empty list
    means every verifiable quote reproduced **or** there was nothing verifiable to
    check — the two are deliberately not distinguished, because "unverifiable" is
    the same green answer as "true" for a write. `rounds_dir` overrides the state
    directory (tests, and any caller holding another data root).
    """
    if not body:
        return []
    root = Path(rounds_dir) if rounds_dir is not None else rounds_root()
    cache: dict[str, list[tuple[Path, str]]] = {}
    absent: list[dict] = []
    seen: set[str] = set()
    for paragraph in _paragraphs(body):
        round_ids = list(dict.fromkeys(_ROUND_ID.findall(paragraph)))
        if not round_ids:
            continue
        quotes = [s.strip() for s in _INLINE_CODE.findall(paragraph) if _is_verifiable(s)]
        if not quotes:
            continue
        artifacts: list[tuple[Path, str]] = []
        for rid in round_ids:
            if rid not in cache:
                cache[rid] = _artifacts(root / rid)
            artifacts += cache[rid]
        if not artifacts:
            continue  # nothing readable: unverifiable, not absent
        rendered = [(path, text, _collapse(text)) for path, text in artifacts]
        for quote in quotes:
            if quote in seen:
                continue
            flat = _collapse(quote)
            if any(quote in text or flat in fold for _, text, fold in rendered):
                continue
            seen.add(quote)
            absent.append({"quote": quote,
                           "round_ids": round_ids,
                           "searched": [str(path) for path, _ in artifacts]})
    return absent


def blocker_quote_refusal(name: Any, description: Any, tags: Any) -> str | None:
    """The error a `blocker` create must be answered with, or None to let it through.

    Only the `blocker` tag is checked, because that tag is the privilege: an
    ordinary item whose body happens to mention a round is not claiming a line was
    copied out of it. Fails open on any internal error — this sits on the path that
    writes every self-spawn and every triage note, and a guard that broke the board
    would be the worse bug; the failure is logged, not swallowed.
    """
    try:
        if BLOCKER_TAG not in normalize_tags(tags):
            return None
        title = str(name).strip() if name else ""
        body = f"# {title}\n\n{description}" if title else str(description or "")
        absent = absent_quotes(body)
        if not absent:
            return None
        first = absent[0]
        more = (f" It quotes {len(absent) - 1} more line(s) absent from the same "
                f"round directory." if len(absent) > 1 else "")
        return (
            f"Refused: this blocker quotes `{first['quote']}` as a line copied out of "
            f"round {', '.join(first['round_ids'])}'s report, and that string is in "
            f"none of the files the round wrote ({', '.join(first['searched'])})."
            f"{more} Re-read that report off disk for the round's current head and "
            "re-quote a line verbatim out of it (either file in the round directory "
            "counts), name the file the line really comes from, or drop the quote: a "
            "blocker whose evidence is not in the artifact it names cannot be taken "
            "early, and a second refusal sends the round back."
        )
    except Exception:  # noqa: BLE001 — fail open, but say so in the log
        logger.exception("blocker evidence check errored; the write goes through")
        return None
