"""Refuse a self-contradictory date before it reaches a human.

A date written as a weekday word plus an ordinal — `Fri Sept 19` — carries two
claims, and for any given year at most one of them can be true. 2026-09-19 is a
Saturday. So that pair is decidable as false with no calendar lookup, no email
read, and no clock beyond the year: it needs nothing from the document the date
was supposedly copied from, which is exactly why it is worth refusing at the
boundary instead of trusting the prose.

Why this sits on the ambient injection path (#1149). On 2026-09-14 the morning
brief (`autonomy` task #68) delivered a real deadline twice and disagreed with
itself: `deadline Fri Sep 18 at 11:59 PM` in one cycle, then `nominating window
closes Fri Sept 19` fourteen minutes later, for the same source email whose body
said `Nominations will be accepted until Friday, September 18, 2026 at 11:59
p.m.` The second value is not a reading of anything, and it is worse than a wrong
clock header (#1079) because it is invisible and self-consistent: the entry names
the right sender, right cities, right URL, and grants one extra day of slack that
does not exist. A reader acting on "Sept 19" loses Friday.

Scope, chosen deliberately narrow:

* Only a *weekday word bound to an ordinal* is judged. A bare ordinal
  (`Sept 19 at 10am`) has nothing to contradict itself, so it passes; so does
  date-free text.
* Relative words (`today`, `tonight`, `tomorrow`) are **not** judged here, even
  though `tomorrow Sep 19` in a run whose local day was Sep 16 is one of the
  artifacts in #1149. `Tonight! Sept 15` written at 22:49 on Sept 14 is legitimate
  span-midnight usage, and 4 of the 6 offending statements in the 173 task-#68
  reports since 2026-09-14 are relative-label cases — a 400 on those would reject
  real briefs. Relative-day drift belongs to the skill's day-bucket rule
  (`skills/morning-brief-and-triage/SKILL.md` Step 4b), not to this guard.
* A year stated in the text is honoured; otherwise the server's current year is
  used, which is what makes `Fri Sept 19` rejectable in 2026 and also makes
  `Fri Sept 19, 2025` pass — 2025-09-19 really was a Friday.

Known limits, all of them "says nothing", none of them "says a date is right":
the shapes recognised are `<weekday> <month> <ordinal>[, <year>]` with the
weekday first, so `Sept 19 (Fri)` is not matched; a sentence terminator between
the weekday word and the month ends the candidate, so `since Monday. September 19
is the payment date` is two statements and is left alone; and the two words have
to sit on the same line.
"""

from __future__ import annotations

import datetime
import re
from typing import NamedTuple, Optional, Sequence

_WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday"]

# The gap between the weekday word and the month: spaces and commas always, and
# a period only where it is an abbreviation mark — a period with whitespace after
# it ends a sentence, and two statements are not one date. Without that,
# `Nominations were open since Monday. September 19 is when the window reopens`
# — both halves true — reads as the pair `Monday. September 19` and a correct
# brief takes a 400. `Mon. September 19` is missed as a result: a missed pair
# costs a brief its guard, a false one costs it the delivery, so the ambiguity
# resolves toward not matching.
_WORD_GAP = r"(?:[ ,]|(?<!\s)\.(?!\s))*"

# `<weekday>[,] <month> <ordinal>[st|nd|rd|th][, <year>]` — both name slots are
# captured loosely and then validated against the tables above, because an
# alternation over the accepted spellings is how a matcher ends up silently
# matching nothing (see the `sept`-as-month-10 probe defect in #1149's own
# verification pass).
#
# The month→ordinal gap still admits `Sept.` because an abbreviated month really
# does carry a period with a space after it (`Friday, Sept. 19` is a pair worth
# refusing); a sentence ending in a bare month name is the shape this leaves
# unmatched, and it needs a weekday word in front of it to matter at all. The
# year slot takes spaces and tabs only, so a year on the *next* line of a bullet
# list is not pulled up into this fragment.
_PAIR = re.compile(
    r"\b(?P<weekday>[A-Za-z]{3,9})" + _WORD_GAP +
    r"(?:the\s+)?"
    r"(?P<month>[A-Za-z]{3,9})\.?[ ,]*"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,?[ \t]*(?P<year>(?:1[6-9]|2[0-9])\d{2}))?\b"
)


class BadDate(NamedTuple):
    """One date statement whose weekday word and ordinal cannot both be true."""

    fragment: str
    iso: str
    stated_weekday: str
    actual_weekday: str

    @property
    def message(self) -> str:
        return (
            f"{self.fragment!r} cannot be right: {self.iso} is a "
            f"{self.actual_weekday}, not a {self.stated_weekday}"
        )


def _weekday_of(token: str) -> Optional[int]:
    return _WEEKDAYS.get(token.lower().strip("."))


def _month_of(token: str) -> Optional[int]:
    return _MONTHS.get(token.lower().strip("."))


def _today() -> datetime.date:
    """The server's local day, behind one indirection so a test can fix the clock."""
    return datetime.date.today()


def find_contradictory_dates(
    text: str, today: Optional[datetime.date] = None
) -> list[BadDate]:
    """Return each date pair in *text* whose weekday word and ordinal disagree.

    Empty list means nothing here is decidable as false: it is not a proof that
    every date is *right*, only that no stated weekday contradicts its own
    ordinal. An unknown month or weekday word, and an ordinal that no calendar
    has in that month (`Feb 30`), are skipped — this guard rejects statements
    that are impossible, and says nothing about ones it cannot evaluate.
    """
    if not text:
        return []
    year = (today or _today()).year
    found: list[BadDate] = []
    seen: set[str] = set()
    for m in _PAIR.finditer(text):
        wd = _weekday_of(m.group("weekday"))
        month = _month_of(m.group("month"))
        if wd is None or month is None:
            continue
        day = int(m.group("day"))
        stated_year = int(m.group("year")) if m.group("year") else year
        try:
            date = datetime.date(stated_year, month, day)
        except ValueError:
            continue
        if date.weekday() == wd:
            continue
        fragment = m.group(0).strip()
        if fragment in seen:
            continue
        seen.add(fragment)
        found.append(BadDate(
            fragment=fragment,
            iso=date.isoformat(),
            stated_weekday=_WEEKDAY_NAMES[wd],
            actual_weekday=_WEEKDAY_NAMES[date.weekday()],
        ))
    return found


def refusal_detail(fields: Sequence[str],
                   today: Optional[datetime.date] = None) -> Optional[str]:
    """A 400 detail naming every offending fragment in *fields*, or None.

    `fields` are the text fields of one injection payload (the body text, or the
    summary and content of a prefetch entry): one pass judges the whole payload,
    so a producer hears about all of its bad dates at once.
    """
    bad: list[BadDate] = []
    for field in fields:
        bad.extend(find_contradictory_dates(field or "", today=today))
    if not bad:
        return None
    reasons = "; ".join(b.message for b in bad)
    return (
        f"{reasons}. Quote the date as the source document states it instead of "
        "restating it — a weekday word and an ordinal that cannot both be true "
        "give the reader a deadline that does not exist (#1149)"
    )
