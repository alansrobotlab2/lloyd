"""One fact-ID scheme, shared by every writer.

Two writers used to invent IDs independently: the extractor numbered facts
`pref-001` upward but restarted at 1 on every run, and `fact_add` minted
`pref-a3f9` from a UUID. The restart is why 43% of fact files carried
duplicate IDs on 2026-09-03 — a merge appends new facts, the numbering starts
over, and `pref-001` names two different facts in one file. Anything that
addresses a fact by ID (fact_invalidate, fact_resolve, a revert report) then
acts on whichever it finds first.

`next_fact_id` continues from the highest number already present, so an ID is
unique within its file and stays stable once assigned.
"""
from __future__ import annotations

import re

# Category → ID prefix. Four characters, so an ID reads as `<what>-<n>`.
# Keys are the singular CATEGORY_VOCAB forms; plurals and unknown categories
# are handled by `category_prefix`.
CATEGORY_PREFIX = {
    "state": "stat",
    "event": "evnt",
    "decision": "deci",
    "preference": "pref",
    "goal": "goal",
    "skill": "skil",
    "relationship": "rel",
    "capability": "capa",
    "constraint": "cons",
    "configuration": "conf",
    "hardware": "hard",
    "research": "resr",
    "general": "fact",
    # Legacy spellings still present in the tree.
    "status": "stat",
    "temporary": "temp",
    "overview": "ovrv",
}

_ID_RE = re.compile(r"^(?P<prefix>[a-z]+)-(?P<num>\d+)$")


def category_prefix(category: str | None) -> str:
    """ID prefix for a category. Plurals fold to the singular form."""
    c = (category or "general").strip().lower()
    if c in CATEGORY_PREFIX:
        return CATEGORY_PREFIX[c]
    if c.endswith("s") and c[:-1] in CATEGORY_PREFIX:
        return CATEGORY_PREFIX[c[:-1]]
    # Unknown category: a stable 4-char stem beats a shared "fact" bucket,
    # because it keeps IDs from two categories in one file distinguishable.
    stem = re.sub(r"[^a-z]", "", c)[:4]
    return stem or "fact"


def next_fact_id(existing_ids, prefix: str) -> str:
    """The next free `<prefix>-NNN` given the IDs already in the file.

    Continues from the highest number seen for this prefix, so re-running an
    extraction over a file that already holds `pref-001..pref-012` produces
    `pref-013`, not a second `pref-001`.
    """
    highest = 0
    for fid in existing_ids or ():
        m = _ID_RE.match(str(fid or "").strip())
        if m and m.group("prefix") == prefix:
            highest = max(highest, int(m.group("num")))
    return f"{prefix}-{highest + 1:03d}"


def assign_ids(facts: list[dict], category: str | None) -> list[dict]:
    """Give every fact without an `id` the next free one. Mutates in place.

    Facts that already carry an ID keep it — an ID is a handle other records
    hold, so re-numbering would break `fact_invalidate` and every revert
    report that names one.
    """
    prefix = category_prefix(category)
    taken = [f.get("id") for f in facts if isinstance(f, dict) and f.get("id")]
    seen = set(taken)
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("id"):
            continue
        fid = next_fact_id(taken, prefix)
        # A file can legitimately hold facts of several categories; guard
        # against a collision with an ID that is present but out of sequence.
        while fid in seen:
            taken.append(fid)
            fid = next_fact_id(taken, prefix)
        fact["id"] = fid
        taken.append(fid)
        seen.add(fid)
    return facts
