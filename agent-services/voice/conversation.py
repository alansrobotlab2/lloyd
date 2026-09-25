"""The words a conversation is managed with, as opposed to the words it is about.

Three small vocabularies and the rules that read them. They are fixed lists on
purpose: every shipped assistant that drops the wake word closes on a fixed
closer vocabulary ("stop", "that's all", "never mind"), and the backchannels a
listener makes while someone else talks are a short, well-known set.

**Backchannels are not turns.** On 2026-09-24 one evening's voice session
dropped 26 utterances as "not addressed" — 'Yeah.', 'Okay.', 'Mm.', 'Thank
you.' — which was Alan reacting to a long answer being read aloud, and one
'Yeah.' that landed inside the window was injected as a turn, which Lloyd then
answered with 59 seconds of speech. A backchannel inside a conversation keeps
it open and is never sent to the model, never interrupts.

The exception is an answer: when Lloyd's last sentence was a question, "yeah"
means yes. `is_backchannel(text, after_question=True)` therefore holds back only
the pure acknowledgements that cannot answer anything ("mm", "uh-huh", "I
see").

**Closers end the conversation** and, if Lloyd is talking, stop him.

**Captions become progress.** Every tool call carries a model-written caption
("Checking supervisor status"); `caption_to_speech` turns one into something
worth saying during a long tool turn, or returns "" when it is not (a path, a
hash, a single word).
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z']+")

#: Acknowledgements that answer nothing — never a turn, even after a question.
PURE_BACKCHANNELS = frozenset({
    "mm", "mmm", "mhm", "mm-hmm", "mmhmm", "hmm", "uh-huh", "uh huh", "uhhuh",
    "huh", "ah", "oh", "i see", "gotcha", "got it", "cool", "nice", "wow",
    "interesting", "crazy", "ha", "haha", "right right",
})

#: Acknowledgements that ARE answers when Lloyd just asked something.
ANSWERING_BACKCHANNELS = frozenset({
    "yeah", "yea", "yep", "yup", "okay", "ok", "right", "sure", "alright",
    "all right", "thanks", "thank you", "okay thanks", "ok thanks",
    "yeah okay", "okay cool", "yeah yeah", "okay okay",
})

#: Said to end the conversation. Matched on the whole utterance, so "stop the
#: build" is a request, not a closer.
CLOSERS = frozenset({
    "stop", "stop it", "stop talking", "that's all", "thats all", "that is all",
    "that's it", "thats it", "that's it for now", "that's all for now",
    "never mind", "nevermind", "i'm done", "im done", "we're done",
    "thanks lloyd", "thank you lloyd", "thanks that's all", "thanks done",
    "thanks that's it", "okay that's all", "ok that's all", "goodbye",
    "bye", "bye lloyd", "goodbye lloyd", "that'll do", "that will do",
    "quiet", "be quiet", "shush", "hush",
})

#: The subset of closers that also stops speech mid-reply — a barge-in word.
STOPPERS = frozenset({
    "stop", "stop it", "stop talking", "quiet", "be quiet", "shush", "hush",
    "enough", "never mind", "nevermind", "wait", "hold on", "hang on",
    "lloyd stop", "okay stop", "ok stop",
})


def normalize(text: str) -> str:
    """Lower-case words only, single-spaced: 'Yeah, okay.' -> 'yeah okay'."""
    t = (text or "").lower().replace("mm-hmm", "mmhmm").replace("uh-huh", "uhhuh")
    return " ".join(_WORD.findall(t))


def words(text: str) -> list[str]:
    return normalize(text).split()


def is_backchannel(text: str, after_question: bool = False) -> bool:
    """A short acknowledgement rather than something to answer.

    At most three words, and the whole utterance must be in the vocabulary —
    "yeah, check the other one" is a request that happens to open with yeah.
    """
    n = normalize(text)
    if not n or len(n.split()) > 3:
        return False
    if n in PURE_BACKCHANNELS:
        return True
    if after_question:
        return False
    if n in ANSWERING_BACKCHANNELS:
        return True
    # "yeah yeah okay", "okay cool thanks": every word an acknowledgement.
    vocab = PURE_BACKCHANNELS | ANSWERING_BACKCHANNELS
    return all(w in vocab for w in n.split())


_POLITE_TAIL = (" thanks", " thank you", " lloyd", " please")
_POLITE_HEAD = ("thanks ", "thank you ", "okay ", "ok ", "alright ", "all right ",
                "lloyd ")


def _strip_polite(n: str) -> str:
    """'okay that's all thanks lloyd' -> "that's all"."""
    changed = True
    while changed and n:
        changed = False
        for tail in _POLITE_TAIL:
            if n.endswith(tail):
                n, changed = n[: -len(tail)], True
        for head in _POLITE_HEAD:
            if n.startswith(head):
                n, changed = n[len(head):], True
    return n


def is_closer(text: str) -> bool:
    n = normalize(text)
    if n in CLOSERS:
        return True
    # Politeness around the closer: "That's all, thanks." / "Okay, never mind,
    # Lloyd." — but never down to nothing ("Thanks, Lloyd." is in the list as
    # itself, and a bare "thanks" is a backchannel, not a goodbye).
    core = _strip_polite(n)
    return bool(core) and core != n and core in CLOSERS


def is_stopper(text: str) -> bool:
    n = normalize(text)
    if n.startswith("lloyd ") and n[6:] in STOPPERS:
        return True
    if n.endswith(" lloyd") and n[:-6] in STOPPERS:
        return True
    return n in STOPPERS


def ends_with_question(text: str) -> bool:
    """Did this spoken text end by asking something?

    Home Assistant's whole `continue_conversation` implementation is this test
    on the assistant's last message, and it is the structural signal every open
    system uses to keep listening without a wake word.
    """
    t = (text or "").rstrip().rstrip("\"'”’)")
    return t.endswith("?")


_PATHLIKE = re.compile(r"(?:\S*/\S*|\S+\.(?:py|ts|tsx|js|json|yaml|yml|md|sh|txt|log|db|conf|toml)\b|\b[0-9a-f]{7,40}\b)")
_BRACKETS = re.compile(r"[`*_#<>\[\]{}|\\]")


def caption_to_speech(caption: str, max_words: int = 9) -> str:
    """A tool-call caption as a short spoken progress line, or "".

    Captions are written for a transcript ("Reading server.py", "grep for
    enqueue_turn"), so paths, hashes and identifiers come out and what is left
    must still be a phrase — at least two words, the first a word a person
    would say. Capitalised, ending in a full stop.
    """
    t = _PATHLIKE.sub("", caption or "")
    t = re.sub(r"\b\w*_\w*\b", "", t)       # snake_case identifiers, before _ goes
    t = _BRACKETS.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" .,:;—–-")
    ws = t.split()
    # Removing an identifier can leave the phrase dangling ("grep for in"):
    # a trailing preposition means the object is gone, so drop back to what
    # still stands on its own.
    while ws and ws[-1].lower() in _DANGLING:
        ws.pop()
    if len(ws) < 2 or not re.fullmatch(r"[A-Z][a-z']*", ws[0]):
        # Lower-case first word = tool jargon written as a command ("grep
        # for", "ls the dir"), not a sentence someone would say.
        return ""
    if len(ws) > max_words:
        ws = ws[:max_words]
        while ws and ws[-1].lower() in _DANGLING:
            ws.pop()
    out = " ".join(ws).rstrip(" ,:;—–-")
    # "Checking supervisor status" -> "Checking supervisor status now." reads as
    # an update rather than a label; anything longer just gets its full stop.
    if ws[0].lower().endswith("ing") and len(ws) <= 4:
        out += " now"
    return out + "."


_DANGLING = frozenset({"in", "for", "to", "at", "on", "of", "from", "with",
                       "into", "by", "the", "a", "an", "and", "or"})
