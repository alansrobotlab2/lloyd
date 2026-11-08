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

* **A `key: value` rendering of a field the artifact holds is the same quote.** A
  round copying `gate.json`'s `"head": "a8773819…"` into prose writes `head:
  a8773819`, which is no substring of the file it came from — so the match step
  also compares the two as *fields*, not as bytes (`_key_value_reproduces`). Only
  an abbreviation of a value that long is read that way: a prefix shorter than
  `MIN_VALUE_PREFIX_CHARS` is a guess about the number, and stays absent.

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


#: One `key: value` pair and nothing else — a key token, a colon, one value token.
#: The shape a round writes when it copies a JSON field into prose.
_KEY_VALUE = re.compile(r"[\"']?([A-Za-z_][A-Za-z0-9_.\-]*)[\"']?[ \t]*:[ \t]*(\S+)")

#: How much of a value a prefix must carry to count as copied rather than guessed.
#: `passed: 2290` against `"passed": 22907` is a guess about a count: the span is
#: 12 characters, so `MIN_QUOTE_CHARS` lets it through, and only 4 of them are the
#: value. `head: a8773819` is an abbreviation of a sha that means nothing except as
#: a copy of the field it names, so a floor on the VALUE is what separates them.
MIN_VALUE_PREFIX_CHARS = 6

#: Punctuation that belongs to the artifact and not to the value: JSON's quoting
#: and its trailing comma, a shell's quoting. Stripped off both ends of a token.
_VALUE_EDGE_CHARS = "\"'`, \t"

#: The same pair scanned out of an artifact's own lines: the key quoted as JSON
#: writes it or bare as YAML and a log's summary line write it, the value run up to
#: the next space or comma.
_LINE_FIELD = re.compile(
    r"[\"']?([A-Za-z_][A-Za-z0-9_.\-]*)[\"']?[ \t]*:[ \t]*([^\s,]+)")


def _key_value(span: str) -> tuple[str, str] | None:
    """The (key, value) of a span that is exactly one `key: value` pair, else None."""
    match = _KEY_VALUE.fullmatch(span.strip())
    if match is None:
        return None
    return match.group(1), match.group(2).strip(_VALUE_EDGE_CHARS)


def _merged_fields(artifacts: list[tuple[Path, str]]) -> dict[str, set[str]]:
    """Every `key: value` field the round's own files carry, merged across them."""
    out: dict[str, set[str]] = {}
    for _, text in artifacts:
        for match in _LINE_FIELD.finditer(text):
            out.setdefault(match.group(1), set()).add(
                match.group(2).strip(_VALUE_EDGE_CHARS))
    return out


def _key_value_reproduces(span: str, fields: dict[str, set[str]]) -> bool:
    """Is this `key: value` span the artifact's own field, abbreviated honestly?

    **The false refusal this exists for, re-runnable.** The only blocker this guard
    has ever refused was an honest one: `mcp.err:13135` (in
    `~/lloyd-data/logs/mcp.err`, 2026-11-07) reads ``Refused: this blocker quotes
    `head: a8773819` as a line copied out of round SM_20261107_150028's report``,
    while `gate.json:4` of that very round holds
    `"head": "a8773819348e8795b922e45901be1f5f98de1159"` and `gate.log:290` holds
    `(a8773819)`. The substring test can never see it: prose
    renders the field `head: …`, the JSON renders it `"head": "…"`, so even a
    byte-exact `head: <full sha>` was refused. Re-run:

        cd ~/lloyd && .venvs/lloyd/bin/python -c 'from pathlib import Path; \\
        from app.backlog_evidence import absent_quotes as A; \\
        print(len(A("round SM_20261107_150028 `head: a8773819`", rounds_dir=Path("/home/alansrobotlab/.local/state/lloyd-automod/rounds"))))'

    1 before this normaliser, 0 after it — while `head: 9999dead` (key present,
    value in no file) and `passed: 2290` (a 4-character prefix of `"passed": 22907`,
    under `MIN_VALUE_PREFIX_CHARS`) both stay at 1.
    """
    pair = _key_value(span)
    if pair is None:
        return False
    key, quoted = pair
    for value in fields.get(key, ()):
        if quoted == value:
            return True
        if len(quoted) >= MIN_VALUE_PREFIX_CHARS and value.startswith(quoted):
            return True
    return False


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
        # Built only when some span in this paragraph is a `key: value` pair, so the
        # every-day path reads the artifacts exactly as many times as it did.
        fields = (_merged_fields(artifacts)
                  if any(_key_value(q) for q in quotes) else {})
        for quote in quotes:
            if quote in seen:
                continue
            flat = _collapse(quote)
            if any(quote in text or flat in fold for _, text, fold in rendered):
                continue
            if _key_value_reproduces(quote, fields):
                continue  # the artifact's own field, rendered as prose `key: value`
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
