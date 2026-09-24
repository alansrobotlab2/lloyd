"""Text -> the words the vocoder should say (#1165).

`voice/speakable.py` decides what not to say — it strips markdown and deletes
URLs — but expands nothing, so everything Lloyd says that is not an English
word reaches Qwen3-TTS exactly as written: `qmd`, `vLLM`, an email address, a
phone number, `2026-09-15`, `3.1x`, `~23.6k`, `~/obsidian/knowledge/foo.md`.
How the vocoder reads those is its own guess, and nothing measured it until
`eval/tts_fidelity_eval.py`.

`for_speech(text)` is the deterministic half of the fix: a small pronunciation
lexicon and spoken-form expansion for emails, phone numbers, ISO dates, ratios,
approximate counts and home paths. Two properties are the contract:

* **Anything benign passes byte-identical.** Every rule is anchored on
  something an ordinary sentence does not contain — a digit, `@`, `~/`, or a
  lexicon token that is not an English word — so a plain sentence returns the
  very string it was given, and the text pass cannot move the control arm.
* **No `@` and no slash-path survive** an expanded email or path.

It is called from `TTSStreamer.speak()` behind `livekit.tts.normalise_text`,
off by default: a rule stays only when the round-trip corpus shows it raising
its item's score (Alan's measure-first rule), which is a live-GPU run.
Per-entity slowdown is deliberately not here — it needs `OutputShaper` made
span-aware (`tts_shaping.py`), and is a separate, by-ear decision.
"""
from __future__ import annotations

import re

# ── numbers ─────────────────────────────────────────────────────────────────

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third",
                      "five": "fifth", "eight": "eighth", "nine": "ninth",
                      "twelve": "twelfth"}


def int_words(n: int) -> str:
    """0 <= n < 10**12 as English words ("twenty-three thousand six hundred")."""
    if n < 0:
        return "minus " + int_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("-" + _ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + int_words(r) if r else "")
    for size, name in _SCALES:
        if n >= size:
            q, r = divmod(n, size)
            return int_words(q) + " " + name + (" " + int_words(r) if r else "")
    raise AssertionError("unreachable")


def number_words(s: str) -> str:
    """A decimal string ("23.6", "3", "0.25") as words, fraction digit by digit."""
    whole, _, frac = s.partition(".")
    out = int_words(int(whole or "0"))
    if frac:
        out += " point " + " ".join(_ONES[int(d)] for d in frac)
    return out


def ordinal_words(n: int) -> str:
    words = int_words(n)
    head, sep, last = words.rpartition("-") if "-" in words else ("", "", words)
    if last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def year_words(y: int) -> str:
    """How a year is said: 1999 "nineteen ninety-nine", 2005 "two thousand
    five", 2026 "twenty twenty-six", 1900 "nineteen hundred"."""
    if 2000 <= y <= 2009:
        return int_words(y)
    if 1100 <= y <= 2099:
        hi, lo = divmod(y, 100)
        if lo == 0:
            return int_words(hi) + " hundred"
        return int_words(hi) + " " + ("oh " + _ONES[lo] if lo < 10 else int_words(lo))
    return int_words(y)


def spell_digits(s: str) -> str:
    return " ".join(_ONES[int(d)] for d in s if d.isdigit())


# ── lexicon ─────────────────────────────────────────────────────────────────

#: Tokens the vocoder is not trusted to read, as the words that should come
#: out. Only tokens that are not English words belong here — that is what
#: keeps a benign sentence byte-identical. Order matters where one is a prefix
#: of another (LLMs before LLM).
LEXICON: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bqmd\b", re.IGNORECASE), "Q M D"),
    (re.compile(r"\bvLLM\b", re.IGNORECASE), "V L L M"),
    (re.compile(r"\bLLMs\b"), "L L Ms"),
    (re.compile(r"\bLLM\b"), "L L M"),
    (re.compile(r"\bMCP\b"), "M C P"),
    (re.compile(r"\bTTS\b"), "T T S"),
    (re.compile(r"\bASR\b"), "A S R"),
    (re.compile(r"\bsupervisorctl\b", re.IGNORECASE), "supervisor control"),
    (re.compile(r"\bsupervisord\b", re.IGNORECASE), "supervisor D"),
    (re.compile(r"\bLiveKit\b", re.IGNORECASE), "Live Kit"),
    (re.compile(r"\bFastAPI\b", re.IGNORECASE), "Fast A P I"),
    (re.compile(r"\bopenWakeWord\b", re.IGNORECASE), "open wake word"),
    (re.compile(r"\bgr00t\b", re.IGNORECASE), "Groot"),
]

# ── spoken forms ────────────────────────────────────────────────────────────

_EMAIL = re.compile(
    r"(?<![\w.+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
# A home path, or an absolute one of at least two segments. The lookbehind
# keeps "and/or" and a URL's tail (already deleted upstream) out.
_PATH = re.compile(r"(?<![\w/~.])(~/|/(?=[\w.-]+/))[\w.+-]+(?:/[\w.+-]+)*/?")
_ISO_DATE = re.compile(r"(?<![\w-])(\d{4})-(\d{2})-(\d{2})(?![\w-])")
_PHONE = re.compile(
    r"(?<![\w+])(\+1[\s.-]?)?(?:\((\d{3})\)\s?|(\d{3})[\s.-])(\d{3})[\s.-](\d{4})(?![\w-])")
_SCALED = re.compile(r"(~|≈)?(?<![\w.])(\d+(?:\.\d+)?)([kKMB])\b")
_RATIO = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)[x×](?!\w)")
_APPROX = re.compile(r"(?:~|≈)(?=\d)")
_SCALE_WORD = {"k": "thousand", "K": "thousand", "M": "million", "B": "billion"}
_EMAIL_SYM = {".": " dot ", "_": " underscore ", "-": " dash ", "+": " plus ",
              "%": " percent "}


def _spell_email_part(part: str) -> str:
    out = []
    for run in re.split(r"([._+%-])", part):
        if run in _EMAIL_SYM:
            out.append(_EMAIL_SYM[run])
        elif run.isdigit():
            out.append(" " + spell_digits(run) + " ")
        else:
            out.append(run)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _email(m: re.Match) -> str:
    return f"{_spell_email_part(m.group(1))} at {_spell_email_part(m.group(2))}"


def _segment(seg: str) -> str:
    seg = seg.replace(".", " dot ")
    seg = re.sub(r"[_+-]", " ", seg)
    return re.sub(r"\s+", " ", seg).strip()


def _path(m: re.Match) -> str:
    raw = m.group(0)
    # A sentence that ends on a path ends on its full stop, not a file named ".".
    segs = [s for s in raw[2 if raw.startswith("~/") else 1:].split("/") if s]
    spoken = [_segment(s) for s in segs]
    if raw.startswith("~/"):
        spoken.insert(0, "home folder")
    return ", ".join(s for s in spoken if s)


def _date(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    return f"{_MONTHS[mo - 1]} {ordinal_words(d)}, {year_words(y)}"


def _phone(m: re.Match) -> str:
    groups = [m.group(2) or m.group(3), m.group(4), m.group(5)]
    spoken = [spell_digits(g) for g in groups]
    if m.group(1):
        spoken.insert(0, "plus one")
    return ", ".join(spoken)


def _scaled(m: re.Match) -> str:
    about = "about " if m.group(1) else ""
    return f"{about}{number_words(m.group(2))} {_SCALE_WORD[m.group(3)]}"


def _ratio(m: re.Match) -> str:
    return f"{number_words(m.group(1))} times"


def _strip_path_period(text: str) -> str:
    """`_PATH` is greedy over `.`, so "…/foo.md." would take the full stop."""
    def fix(m: re.Match) -> str:
        raw = m.group(0)
        tail = ""
        while raw and raw[-1] in ".":
            raw, tail = raw[:-1], raw[-1] + tail
        return _path(_PATH.fullmatch(raw) or m) + tail if raw else m.group(0)
    return _PATH.sub(fix, text)


def for_speech(text: str) -> str:
    """Expand what the vocoder should not be left to guess at.

    Returns `text` itself (the same object) when no rule applies."""
    if not text:
        return text
    out = _EMAIL.sub(_email, text)
    out = _strip_path_period(out)
    out = _ISO_DATE.sub(_date, out)
    out = _PHONE.sub(_phone, out)
    out = _SCALED.sub(_scaled, out)
    out = _RATIO.sub(_ratio, out)
    out = _APPROX.sub("about ", out)
    for pattern, spoken in LEXICON:
        out = pattern.sub(spoken, out)
    return text if out == text else out
