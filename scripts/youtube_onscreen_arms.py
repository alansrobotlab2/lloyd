#!/usr/bin/env python3
"""youtube_onscreen_arms — the deterministic scorer for the #3983 arms comparison (#4201).

The on-screen-text leg landed inert at `e30ce708`: `workers/sources/youtube_digest.py` can
write a bundle's `onscreen.jsonl`, and `onscreen_field()` renders `none` for all 696
bundles because no frame has ever been sampled. Before that arm can reach the nightly path
its gain has to be measured, and the measurement is not "did the note read better" — it is
two counts nothing in this tree produces:

  * ``missing_or_unverifiable`` — a literal the note asserts that neither source contains,
    so the note took it from model priors rather than from the video; and
  * ``contamination`` — a literal that exists ONLY on the screen and that the note goes on
    to assert as something the speaker said.

The second is the whole risk of the leg and the reason the comparison is gated. A frame can
spell a name the captions garbled; a model that reads `KernelBench` on a slide and writes
"the speaker says KernelBench" has invented a claim about a person. Item #4201 step 5 says
any nonzero contamination closes the arm rather than tuning it, so this scorer is a gate,
not a report, and a gate has to be runnable by someone who does not trust the model.

It is deterministic and offline on purpose: no LLM may judge an LLM's provenance, and the
paired run needs identical arithmetic over 10 videos in two arms. Importing this file opens
no socket and calls no engine (pinned by
`tests/test_youtube_onscreen_arms.py::test_importing_the_scorer_opens_no_socket_and_calls_no_engine`);
scoring is a pure function over three text files. It is deliberately NOT wired into
`build_prompt` or into any job. The two arms are two scratch copies of one bundle
directory, one with a sidecar and one without, and `onscreen_field()` already chooses the
arm from `sidecar.is_file()` (`workers/sources/youtube_digest.py:833-847`) — a kwarg here
would be a second writer of a decision the filesystem makes, and it would move the live
nightly prompt builder for the sake of an offline experiment.

The tree already has a literal extractor — `scripts/memory_source_grounding.py:186`, which
grounds a memory bullet's paths, dates, shas and entity phrases against a source — and it is
not reusable here for one reason that is the whole difference between the two surfaces: it
treats a backticked span as PROTECTED and skips it, because in a memory bullet backticks mean
"this is the code I am talking about, not a claim". In a digest note a backticked name IS the
claim, and the one the leg is most likely to have supplied. Its entity-phrase rule is also
position-free by design (`driver. Mission Control` is the shape a bullet puts a name in) where
this scorer needs sentence position to tell `It Uses` from `AI Engineer`.

Run it on one video:

    python scripts/youtube_onscreen_arms.py --note /path/note.md \\
        --bundle ~/.local/share/ai-engineer/bundles/<video_id>

Exits 2 when the note or the transcript is absent: scoring a file that is not there reads
as "every literal unverifiable", which is a confident null and not a result. A missing
`onscreen.jsonl` is NOT an error — that is the control arm, and it scores `contamination: 0`
because on it nothing was only on the screen.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

#: The two names the leg writes and the prompt reads. Spelled here rather than imported —
#: `workers/sources/youtube_digest.py:129` is the authority for `onscreen.jsonl` — because
#: this file's import must stay stdlib-only, and the values are pinned equal to the leg's
#: by `tests/test_youtube_onscreen_arms.py` so a rename cannot drift silently.
ONSCREEN_FILENAME = "onscreen.jsonl"
TRANSCRIPT_FILENAME = "transcript.txt"

#: How far a note's timestamp may sit from the sidecar's frame `ts` and still count as
#: pointing at that frame. The sidecar stores frame seconds
#: (`workers/sources/youtube_digest.py:492-498`); a note writing `0:41` for the frame at
#: 41.4s points at it, and one writing `9:99` points at nothing.
TS_TOLERANCE_SECONDS = 2.0

#: The leg's own ceiling, restated: a literal over this length is a sentence about a slide,
#: and a sentence is where contamination comes from
#: (`workers/sources/youtube_digest.py:146-150`).
MAX_LITERAL_CHARS = 80

#: A screen attribution: the note saying out loud that it read this off the screen. The
#: wording tracks the prompt's own `onscreen:` line
#: (`workers/sources/youtube_digest.py:740`) — "Anything taken from it was seen on screen,
#: not heard: it is not a thing the speaker said" — so the marker is the phrase the
#: instruction already teaches. #4203 is the parked sibling that puts a provenance clause in
#: the digest skill; until that lands this marker set is the whole convention, which is
#: exactly why an unmarked screen-only literal counts as contamination: the reading a
#: human takes from an unmarked sentence is "the speaker said it".
SCREEN_MARK_RE = re.compile(r"on[\s_\-]?screen|keyframe|\bslide\b", re.IGNORECASE)

#: Timestamps a note may point at a frame with: `0:41`, `1:02:03`, `41s`, `41 seconds`,
#: `@41`. Every form normalises to seconds against a row's `ts`.
_TS_HMS_RE = re.compile(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b")
_TS_UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
                         re.IGNORECASE)
_TS_AT_RE = re.compile(r"@[\t ]*(\d+(?:\.\d+)?)\b")

_QUOTED_RE = re.compile(r"[\"“”]([^\"“”\n]{2,60})[\"“”]")
_BACKTICK_RE = re.compile(r"`([^`\n]{2,60})`")
_TITLE_WORD_RE = re.compile(r"[A-Z][A-Za-z0-9'’]*(?:[.-][A-Za-z0-9'’]+)*")
# A figure carries a name on its own only with a unit, a decimal or a percent — `70B`,
# `3.2`, `78.5%`. A bare `3` in prose is a count, and the leg's filter drops bare figures
# for the same reason (`workers/sources/youtube_digest.py:250-252`).
_FIGURE_WORD_RE = re.compile(r"\d+(?:[.,]\d+)+(?:%|[A-Za-z]{1,3})?"
                             r"|\d+(?:%|[A-Za-z]{1,3})")
# A bare figure inside a name — the `4` of `Gemini 4 Carbon`, the `3` of `Claude 3 Opus`. It
# may join a run that has already started on a word with a letter in it and may never start
# one, because a sentence that opens `4 things to know about agents` is a count, not a name.
_JOINER_FIGURE_RE = re.compile(r"\d+(?:[.,]\d+)*")
# A word, keeping the internal punctuation that belongs to a name (`GPT-4o`, `2.5`,
# `Top-5`) and refusing the punctuation that does not: `[.-]` is consumed only when another
# word character follows it, so the full stop ending a sentence is not part of the word.
_WORD_RE = re.compile(r"[0-9A-Za-z](?:[0-9A-Za-z'’]|[.-](?=[0-9A-Za-z]))*")
_SPLIT_RE = re.compile(r"[^0-9A-Za-z]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_REPO_ROOT = Path(__file__).resolve().parents[1]

_general_vocab_cache: frozenset[str] | None = None


def general_vocab() -> frozenset[str]:
    """The leg's own list of ordinary English, loaded on first use, never copied.

    `GENERAL_VOCAB` (`workers/sources/youtube_digest.py:166`, 216 words) is the definition
    this repo already holds of "a token that is vocabulary, not a name", and it is the one
    that must decide prose here too: it knows `code`, `model`, `uses`, `this` and `week`,
    and it does NOT know `roofline`, `agent`, `claude`, `gemini` or `ai`. A second list in
    this file would drift from it within a quarter, and the drift would be invisible
    because both lists would look plausible.

    Lazy and cached, the way `scripts/maintenance/corpus_shape.py:271-275` reaches the
    intel-pipeline guards: importing this module must stay stdlib-only, so nothing heavier
    than the standard library loads until somebody actually scores something.
    """
    global _general_vocab_cache
    if _general_vocab_cache is None:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from workers.sources.youtube_digest import GENERAL_VOCAB   # noqa: E402
        _general_vocab_cache = GENERAL_VOCAB
    return _general_vocab_cache


def fold(text: str) -> list[str]:
    """`text` as the token sequence a name comparison should use.

    Lower-cased, accents stripped (`Café` and `Cafe` are one name), CamelCase split at its
    internal capitals (`KernelBench` → `kernel bench`), and split on everything that is not
    a letter or a digit — so `kernel-bench`, `kernel.bench`, `Kernel Bench` and
    `KERNELBENCH` all answer `["kernel", "bench"]`. That is the item's "after case- and
    punctuation-normalization", and the CamelCase half is not decoration: the failure this
    leg exists for is auto-captions splitting a proper noun across a space, so a scorer that
    could not rejoin the halves would report a recovered name as still missing.

    Tokens, not one joined string, because substring containment on a joined string is how a
    two-letter literal gets "found" inside an unrelated word — `ai` is inside `said`. A
    literal is verified only where its whole token tuple appears, and `_literal_matches`
    decides what "appears" means.

    Accents go first, before the split, and not per word: `é` is a letter in `Café` but a
    separator to `_SPLIT_RE`, so stripping after splitting would fold `Café` to `caf` and
    match a source that never said it.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    out: list[str] = []
    for raw in _SPLIT_RE.split(stripped):
        if not raw:
            continue
        for piece in _CAMEL_BOUNDARY_RE.split(raw.lower()):
            if piece:
                out.append(piece)
    return out


def _words(text: str) -> list[tuple[int, str]]:
    """`(offset, word)` pairs, so a run can be sliced back out of the original spelling."""
    return [(m.start(), m.group(0)) for m in _WORD_RE.finditer(text or "")]


def _is_name_word(word: str) -> bool:
    """A word that can belong to a name: capital-initial (title, CamelCase, acronym) or a figured quantity."""
    return bool(_TITLE_WORD_RE.fullmatch(word) or _FIGURE_WORD_RE.fullmatch(word))


def _prose_only(tokens: list[str]) -> bool:
    """True when every lettered token is ordinary English, so shape is all this candidate has.

    Digits are ordinary tokens, as in the leg's filter: a quantity becomes a candidate only
    inside a named thing. The rule deliberately differs from `is_general_vocabulary()`
    (`workers/sources/youtube_digest.py:244-265`) on one point — that function exempts
    title-cased text, because there a two-word proper noun like the channel's own name
    `AI Engineer` would otherwise be dropped from a capped proposal list. Here the
    title-cased shape is the very false positive to catch: prose begins sentences with
    capitalised ordinary words, and exempting that shape would score `This Week` as a claim.
    Nothing is lost by the difference, because the exemption's purpose is served by the
    list itself: `ai` and `engineer` are both absent from `GENERAL_VOCAB`, so
    `AI Engineer` is not prose-only anywhere.
    """
    lettered = [t for t in tokens if any(c.isalpha() for c in t)]
    return bool(lettered) and all(t in general_vocab() for t in lettered)


def _appears_mid_sentence(candidate: str, sentences: list[str]) -> bool:
    """Does the candidate occur somewhere other than the very start of a sentence?

    The tie-breaker for a prose-shaped candidate: ordinary English capitalised mid-sentence
    is a name (`the AI Engineer episode`), and the same words at a sentence start are prose
    (`AI Engineer is a channel.`). Only candidates that `_prose_only` flagged need this
    question; the rest are kept outright.
    """
    needle = tuple(fold(candidate))
    for sentence in sentences:
        for hit in _subsequence_positions(needle, fold(sentence)):
            if hit > 0:
                return True
    return False


def extract_literals(note: str, sentences: list[str] | None = None) -> list[str]:
    """The literals the note asserts, in the note's own spelling, longest-first, deduped.

    Three signals, in the order a reader of markdown trusts them:

      1. a backticked or double-quoted span — a name the author fenced away from the prose;
      2. a run of two or more adjacent name-words — `Gemini 2.5 Pro`, `Claude Code`,
         `KernelBench eval`;
      3. a single word marked by its own shape rather than by its position: CamelCase
         (`KernelBench`), an acronym of two letters or more (`GPT`), or a figured quantity
         (`70B`).

    A lone word that is merely capitalised is NOT a literal: in prose that shape is a
    sentence or heading initial, and scoring it would count English sentences as
    unverifiable claims. The cost is real and accepted knowingly — a single-word name that
    appears ONLY at a sentence start is missed — so these counts are a LOWER BOUND on both
    arms, and a lower bound computed by identical rules on both sides still answers what the
    comparison asks, which is the difference between the arms.

    A candidate that is nothing but ordinary vocabulary (`_prose_only`) survives only where
    it appears mid-sentence, and anything longer than `MAX_LITERAL_CHARS` is dropped for the
    reason the leg drops it: it is a sentence, not a name.
    """
    sentences = _note_sentences(note) if sentences is None else sentences
    found: list[str] = []
    for pattern in (_BACKTICK_RE, _QUOTED_RE):
        found.extend(match.group(1) for match in pattern.finditer(note or ""))

    words = _words(note or "")
    run: list[tuple[int, str]] = []
    prev_end: int | None = None

    def emit() -> None:
        if len(run) >= 2:
            last_offset, last_word = run[-1]
            found.append(note[run[0][0]:last_offset + len(last_word)])

    for offset, word in words:
        # A run is a NAME, and a name is spelled with spaces in it and nothing else: the gap
        # between two words of one name is whitespace and no more. Without that test the
        # ordinary end of a sentence welds the last word of one sentence to the first of the
        # next and "…shipped Friday.\nAI Engineer is…" becomes a literal called
        # `Friday. AI Engineer` — a run across a full stop is two sentences, never a name.
        adjacent = prev_end is not None and not note[prev_end:offset].strip()
        if not adjacent:
            emit()
            run = []
        # A bare figure joins a run it sits inside but never starts one: `Gemini 4 Carbon` is
        # one name and the `4` is as much a part of it as the `2.5` inside `Gemini 2.5 Pro`,
        # while a sentence beginning `4 things to know` must not become a literal at all.
        if _is_name_word(word) or (run and _JOINER_FIGURE_RE.fullmatch(word)):
            run.append((offset, word))
            prev_end = offset + len(word)
            continue
        emit()
        run = []
        prev_end = offset + len(word)
    emit()

    for _, word in words:
        lettered = [c for c in word if c.isalpha()]
        if not lettered:
            continue
        if (_CAMEL_BOUNDARY_RE.search(word)
                or (word.isupper() and len(lettered) >= 2)
                or _FIGURE_WORD_RE.fullmatch(word)):
            found.append(word)

    out: dict[tuple[str, ...], str] = {}
    for text in found:
        text = text.strip(" \t\"'“”«»·-—")
        if not text or len(text) > MAX_LITERAL_CHARS:
            continue
        tokens = tuple(fold(text))
        if not tokens:
            continue
        if _prose_only(list(tokens)) and not _appears_mid_sentence(text, sentences):
            continue
        previous = out.get(tokens)
        if previous is None or len(text) > len(previous):
            out[tokens] = text

    # `AI` inside `AI Engineer` is the same name mentioned once. Both shapes arrive here at
    # once — the run rule hands over the phrase and the all-caps rule hands over the acronym
    # inside it — and scoring both would count one claim twice, in whichever column it lands.
    kept = list(out.items())
    nested = {short for short, _ in kept
              if any(_subsequence_positions(short, list(long)) 
                     for long, _ in kept if len(short) < len(long))}
    return sorted((text for tokens, text in out.items() if tokens not in nested),
                  key=lambda t: (-len(t), t))


def _subsequence_positions(needle: tuple[str, ...], hay: list[str]) -> list[int]:
    """Every index at which `needle` starts inside `hay`, contiguously."""
    hits: list[int] = []
    size = len(needle)
    if not size or len(hay) < size:
        return hits
    wanted = list(needle)
    first = wanted[0]
    for i in range(len(hay) - size + 1):
        if hay[i] == first and hay[i:i + size] == wanted:
            hits.append(i)
    return hits


def _literal_matches(needle: tuple[str, ...], hay: list[str]) -> bool:
    """Does `hay` contain the literal `needle` folds to?

    Token-for-token first, and then one more way a name can be spelled the same: the same
    letters cut at different places. A sidecar row read off a slide as `KernelBench` folds to
    two tokens, and the same slide captioned in all caps by an OCR that cannot see a word
    boundary folds to one, and those are the same name — `KERNELBENCH` gives `fold` no
    lower-to-upper transition to split on. So a needle also matches any window of `hay`
    whose tokens concatenate to the needle's own concatenation.

    Windows are joined across token boundaries only, never into the middle of a token, which
    is what keeps `ai` from being found inside `said`: the accumulator is built from whole
    tokens and abandoned the moment it outgrows the needle. The window is capped at three
    tokens beyond the needle's length because a name does not spread wider than that, and a
    match at the cap would be a sentence that happens to have the same letters in it.
    """
    if _subsequence_positions(needle, hay):
        return True
    squash = "".join(needle)
    if not squash:
        return False
    for start in range(len(hay)):
        acc = ""
        for end in range(start, min(start + len(needle) + 3, len(hay))):
            acc += hay[end]
            if len(acc) > len(squash):
                break
            if acc == squash:
                return True
    return False


def read_onscreen(path: Path) -> dict:
    """The sidecar as rows, folded literal tokens, and the frame seconds each is seen at.

    A row is `{"ts", "frame", "frame_index", "literals": [{"text", "kind"}], "dropped"}`
    (`workers/sources/youtube_digest.py:492-498` and `:530-531`). The frame `ts` is the
    point of the whole format and the reason contamination is checkable at all: the digest
    may assert a screen-derived literal only with the timestamp it was read at, so this
    returns `literal_ts` keyed by folded tokens and lets the caller re-verify the claim.

    A row that will not parse, or that carries no usable `ts`, is counted in `unparsable`
    and skipped rather than guessed at: an unverifiable frame must not rescue a claim, and
    it must not be silently missing either.
    """
    sidecar: dict = {"rows": 0, "unparsable": 0, "literal_ts": {}, "literal_text": {},
                     "tokens": []}
    if not path.is_file():
        return sidecar
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            ts_seconds = float(row.get("ts"))
        except (AttributeError, TypeError, ValueError):
            sidecar["unparsable"] += 1
            continue
        if not isinstance(row, dict):
            sidecar["unparsable"] += 1
            continue
        sidecar["rows"] += 1
        for item in row.get("literals") or []:
            text = item.get("text") if isinstance(item, dict) else item
            if not isinstance(text, str) or not text.strip():
                continue
            tokens = tuple(fold(text))
            if not tokens:
                continue
            sidecar["literal_ts"].setdefault(tokens, []).append(ts_seconds)
            sidecar["literal_text"].setdefault(tokens, text)
            sidecar["tokens"].extend(tokens)
    return sidecar


def _note_sentences(note: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(note or "") if s.strip()]


def _timestamps(text: str) -> list[float]:
    """Every timestamp a sentence points at, in seconds."""
    stamps: list[float] = []
    for match in _TS_HMS_RE.finditer(text):
        stamps.append(int(match.group(1) or 0) * 3600
                      + int(match.group(2)) * 60 + int(match.group(3)))
    for pattern in (_TS_UNIT_RE, _TS_AT_RE):
        for match in pattern.finditer(text):
            stamps.append(float(match.group(1)))
    return stamps


def score(note: str, transcript: str, onscreen: dict) -> dict:
    """The two counts for one video. Pure: three texts in, one dict out, nothing written.

    The two counts look up in opposite directions, and that is not an accident of style:

    ``missing_or_unverifiable`` reads the NOTE'S literals out with `extract_literals` and
    asks whether either source contains them — a note can claim a name no source ever
    contained, and only the note knows it tried to.

    ``contamination`` iterates the SIDECAR'S literals and asks whether the note asserts
    them. It cannot be note-driven: the orthography rules that keep the note-side count from
    scoring English sentences are exactly the rules that would miss the contamination case,
    because a model writing an invented spoken claim writes it as plain prose —
    "Swarmskills is the repo" at a sentence start, no backticks, no CamelCase. The sidecar
    already carries the name, filtered and capped by the leg, so the screen is the
    authoritative list of what could have been seen and the note is the text being asked
    about. A literal is contaminated when it is in the sidecar, absent from the transcript,
    present in the note, and NO sentence carrying it both names the screen
    (`SCREEN_MARK_RE`) and points at a frame timestamp the sidecar carries for THAT literal
    within `TS_TOLERANCE_SECONDS`. A marker with a timestamp the sidecar does not carry is
    still contamination: a citation to a frame that does not exist is a claim, not a source.

    Two exclusions keep the count honest, both of them false-positive guards on purpose:
    a literal the transcript contains is never contamination, however the note attributes
    it, because it was audible and "the speaker said it" is a claim the video supports; and
    a sidecar literal that is nothing but ordinary vocabulary (`_prose_only`) is skipped,
    because the leg's own filter cannot have kept one — its presence means a malformed
    sidecar, and flagging it would book the note a contamination for writing English.
    """
    transcript_tokens = fold(transcript)
    onscreen_tokens = list(onscreen.get("tokens") or [])
    literal_ts = onscreen.get("literal_ts") or {}
    literal_text = onscreen.get("literal_text") or {}
    sentences = _note_sentences(note)
    literals = extract_literals(note, sentences)

    missing: list[str] = []
    for literal in literals:
        needle = tuple(fold(literal))
        if not _literal_matches(needle, transcript_tokens) \
                and not _literal_matches(needle, onscreen_tokens):
            missing.append(literal)

    contaminated: list[dict] = []
    screen_only = 0
    for needle, stamps in sorted(literal_ts.items(), key=lambda kv: kv[1][0]):
        if _literal_matches(needle, transcript_tokens):
            continue          # audible: whatever the note says about it is supportable
        if _prose_only(list(needle)):
            continue          # a malformed sidecar row, not a name the screen showed
        asserting = [s for s in sentences if _literal_matches(needle, fold(s))]
        if not asserting:
            continue          # the screen showed it and the note never used it
        screen_only += 1
        attributed = any(SCREEN_MARK_RE.search(sentence)
                         and any(abs(stamp - ts) <= TS_TOLERANCE_SECONDS
                                 for stamp in _timestamps(sentence) for ts in stamps)
                         for sentence in asserting)
        if not attributed:
            contaminated.append({
                "literal": literal_text.get(needle, " ".join(needle)),
                "frame_ts": sorted(stamps),
                "reason": "asserted with no screen attribution carrying a timestamp the "
                          "sidecar carries"})
    return {
        "literals": len(literals),
        "screen_only_literals": screen_only,
        "missing_or_unverifiable": len(missing),
        "contamination": len(contaminated),
        "missing_literals": missing,
        "contaminated_literals": contaminated,
        "transcript_tokens": len(transcript_tokens),
        "onscreen_rows": onscreen.get("rows", 0),
        "onscreen_unparsable_rows": onscreen.get("unparsable", 0),
        "onscreen_literals": len(literal_ts),
    }


def score_bundle(note_path: Path, bundle_dir: Path) -> dict:
    """Score one note against one bundle directory. Opens three files to read, writes nothing.

    The bundle directory is never opened for writing, and this scorer does not create one:
    the paired run has to leave `~/.local/share/ai-engineer/bundles` exactly as it found it,
    and the arms are made by copying a bundle into scratch, never by editing it in place.
    """
    note_path = Path(note_path)
    bundle_dir = Path(bundle_dir)
    transcript_path = bundle_dir / TRANSCRIPT_FILENAME
    sidecar_path = bundle_dir / ONSCREEN_FILENAME
    absent = [str(p) for p in (note_path, transcript_path) if not p.is_file()]
    if absent:
        raise FileNotFoundError("cannot score: missing " + ", ".join(absent))
    result = score(note_path.read_text(encoding="utf-8", errors="replace"),
                   transcript_path.read_text(encoding="utf-8", errors="replace"),
                   read_onscreen(sidecar_path))
    result["video_id"] = bundle_dir.name
    result["note_path"] = str(note_path)
    result["bundle_dir"] = str(bundle_dir)
    result["onscreen_path"] = str(sidecar_path) if sidecar_path.is_file() else None
    return result


def render(result: dict) -> str:
    """The two counts first, then the sizes that let a reader doubt them."""
    sidecar = (f"{result['onscreen_rows']} row(s)"
               if result.get("onscreen_path") else "absent (control arm)")
    detail = (f"note literals: {result['literals']}; screen-only literals the note "
              f"asserts: {result['screen_only_literals']}; transcript tokens: "
              f"{result['transcript_tokens']}; onscreen: {sidecar}")
    if result["onscreen_unparsable_rows"]:
        detail += f"; unparsable rows: {result['onscreen_unparsable_rows']}"
    lines = [f"video: {result.get('video_id', '')}",
             f"note: {result.get('note_path', '')}",
             f"missing_or_unverifiable: {result['missing_or_unverifiable']}",
             f"contamination: {result['contamination']}",
             detail]
    lines += [f"  contamination: {row['literal']!r} frame ts available "
              f"{row['frame_ts']} — {row['reason']}"
              for row in result["contaminated_literals"]]
    lines += [f"  unverifiable: {literal!r}" for literal in result["missing_literals"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="youtube_onscreen_arms.py",
        description="Score one digest note against its bundle: unverifiable literals and "
                    "on-screen contamination. Deterministic — no engine, no network.")
    parser.add_argument("--note", required=True, type=Path,
                        help="the note one arm wrote")
    parser.add_argument("--bundle", required=True, type=Path,
                        help=f"bundle directory holding {TRANSCRIPT_FILENAME}, and "
                             f"{ONSCREEN_FILENAME} for the treatment arm")
    parser.add_argument("--json", action="store_true",
                        help="print the whole result as JSON instead of the counts")
    args = parser.parse_args(argv)
    try:
        result = score_bundle(args.note, args.bundle)
    except FileNotFoundError as exc:
        print(f"youtube_onscreen_arms: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
