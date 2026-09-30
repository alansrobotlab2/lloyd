"""The one builder for a fresh daily note's front matter — #1887.

`memory/<date>.md` has three writers. Two of them (`app/post_capture._append_daily_note`
and `app/autonomy.append_daily_alert_line`) each carried their own `yaml.safe_dump`
copy of the `segment/tags/type/timestamp` block; the third
(`notify.Notifier._vault_note`, the watchdog's own alert section) had none at all, so a
day the watchdog reached before any session capture arrived with a blank lead and a
`## Self-mod guardian:` heading and no front matter. That is a conformance violation on
arrival, not a cosmetic one: `scripts/vault/segment_scan.py` scores a file it cannot
parse as missing BOTH required keys, which is how the bare note turned
`tests/test_okf_segment_producers.py::test_scan_exits_0_on_the_live_vault` red on
2026-09-30.

This module is the single origin of that block. The backend reaches it through
`app/daily_note.py`, so there is one rendering, not a third copy.

Why the canonical copy sits here rather than under `app/`, with the other two writers:
the watchdog does not execute the repo. `lloyd-guardian.service` runs
`~/.local/state/lloyd-guardian/bin/guardian.py`, promoted by
`agent-services/bin/guardian-stage.sh`, which copies `agent-services/guardian/*.py` and
nothing else — precisely so the watchdog can still alert while `~/lloyd` is broken. A
shared header under `app/` would drag `yaml` and the app package's own imports into the
moment of the alert, so this file is **stdlib only**, under the same rule
`notify.py`'s module docstring states.

The emitted text is deliberately identical to what `yaml.safe_dump(...)` produced with
`sort_keys=False, allow_unicode=True, default_flow_style=False` — the kwargs
`scripts/vault/okf_migrate.py` uses for the same repair — because the block's byte shape
is what `validate_okf.STRICT_FM_RE` matches at offset 0 and what the nightly validators
compare. `tests/test_daily_note_shared_header.py` pins that equivalence against a live
`safe_dump`, so this emitter cannot drift from the library it replaces.
"""

from __future__ import annotations

from datetime import date, datetime

#: Key order is part of the byte shape the shipped writers produced and the tests
#: compare; `tags` is the only list, hence the `- ` lines under it.
_SEGMENT = "memory"
_TAGS = ("memory", "daily-notes")
_TYPE = "note"


def _stamp(now: datetime | date | str) -> str:
    """The `timestamp` value, always full-width.

    Every daily note in the vault carries `YYYY-MM-DDTHH:MM:SS`, and `timestamp` is
    `str | None` per `scripts/vault/okf_schema.py`, so a bare date would parse but
    read as a different kind of record: a `date` — which is all the watchdog's
    `_today()` clock seam yields — is normalised to midnight of that day. A string
    is passed through untouched, for a caller that already formatted one.
    """
    if isinstance(now, datetime):
        return now.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(now, date):
        return now.isoformat() + "T00:00:00"
    return str(now)


def fresh_front_matter(now: datetime | date | str) -> str:
    """The front-matter body for a day's note: six lines, no surrounding fences.

    `now` is the caller's reading, never a `datetime.now()` from here — the whole
    point of #1887 is that the writers agree on one header, and a `timestamp` each
    writer invented would make them differ on every day they wrote.
    """
    lines = [f"segment: {_SEGMENT}", "tags:"]
    lines += [f"- {tag}" for tag in _TAGS]
    lines += [f"type: {_TYPE}", f"timestamp: '{_stamp(now)}'"]
    return "\n".join(lines)


def fresh_header(now: datetime | date | str, day: str) -> str:
    """A whole fresh note head: the fenced block, a blank line, and the H1.

    Ends with a newline and nothing else, so each writer appends exactly what it
    appended before: capture adds `"\\n## Sessions\\n{entry}"`, the alert writer adds
    the entry line, the watchdog's `_vault_note` lets its own append write the
    `\\n\\n## Self-mod guardian: …` section after it.
    """
    return f"---\n{fresh_front_matter(now)}\n---\n\n# {day} Daily Notes\n"
