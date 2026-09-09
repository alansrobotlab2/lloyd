"""The OKF ``type`` vocabulary, defined once.

OKF itself refuses to fix a taxonomy — "consumers must tolerate unknown types" —
so the vocabulary is ours to choose, and unchosen it drifts. On 2026-09-08
``knowledge/`` held 2,357 typed files across **50** distinct ``type`` values:
``research`` 249 / ``research-note`` 21 / ``research_note`` 16 / ``research note``
1; ``deep-research`` 31; ``quick-research`` 11 beside ``medium-research`` 81;
``knowledge`` 37 / ``knowledge-note`` 33 / ``reference`` 45. The same documents,
spelled differently, and every consumer that branches on ``type`` had to
enumerate every spelling to behave at all.

Backlog item #370 chose the 12 canonical values below and renamed the vault onto
them (vault commits ``6eed1445``, ``c2b0962``). This module is the code half,
#441: before it, three scripts each carried their own copy of the *fragmented*
literals — ``validate_okf.py``'s ``KNOWN_TYPES``, ``okf_migrate.infer_type`` and
``knowledge-frontmatter-backfill.py``'s type sets — so a rename was undone by the
next migrator run, and ``validate_okf.py --strict`` counted the freshly renamed
files as unknown types (155 warnings at triage, **194** once the rename landed
without this half).

Three rules this module exists to keep:

  * ``CANONICAL_TYPES`` is the whole vocabulary for ``knowledge/``. Add a 13th
    value deliberately; never mint a synonym again.
  * ``TYPE_ALIASES`` maps a value that once existed to the canonical it became.
    It exists for *reading* old files and for ``--strict`` tolerance. Nothing
    writes an alias: a writer emits canonical, or emits nothing.
  * Everything else lives here once. A ``type:`` literal spelled out in
    ``validate_okf.py``, ``okf_migrate.py`` or ``knowledge-frontmatter-backfill.py``
    is precisely the defect this module was written to prevent —
    ``tests/test_okf_type_taxonomy.py`` fails on it.
"""
from __future__ import annotations

# ── the canonical 12 (#370) ──────────────────────────────────────────────────
CANONICAL_TYPES = frozenset({
    "research",         # a research finding, any depth
    "research-deep",    # deep-dive pass
    "research-quick",   # quick / medium pass
    "reference",        # stable reference material, incl. source summaries
    "synthesis",        # cross-source analysis
    "video-note",       # a note whose source is one video
    "book-note",        # a note whose source is one book (or long-form work)
    "notes",            # primary notes, incl. paper notes and transcripts
    "stack-update",     # dated work/stack change log
    "infrastructure",   # indexes, hubs, handoffs, maintenance logs
    "agent-pattern",    # agent/system knowledge
    "gap-analysis",     # benchmarks and evaluations
})

# Values legitimate *outside* ``knowledge/``, which the 12-value taxonomy does not
# govern. Held here so the validator keeps tolerating them: dropping one would
# turn every existing file that uses it into a new warning, and #370's acceptance
# is that warnings fall. ``note`` is the vault-wide catch-all ``okf_migrate``
# manufactures outside knowledge/ (494 files on 2026-09-09); retiring it is #442.
OUTSIDE_KNOWLEDGE_TYPES = frozenset({
    "autonomy", "facts", "overview", "compiled_wiki",
    "skill", "person", "daily-note", "reflection",
    "how-to", "comparison", "talk",
    "note",
})

# Fragmented value -> the canonical it was consolidated into. Keys are in
# ``normalize_type`` form (lowercase, hyphens). This is the map the vault rename
# was applied through, so re-running a rename and reading an old file agree.
TYPE_ALIASES: dict[str, str] = {
    "research-note": "research",
    "knowledge": "reference",
    "knowledge-note": "reference",
    "source-summary": "reference",
    "deep-research": "research-deep",
    "deep-dive-research": "research-deep",
    "quick-research": "research-quick",
    "medium-research": "research-quick",
    "analysis": "synthesis",
    "comparative-analysis": "synthesis",
    "concept-synthesis": "synthesis",
    "video": "video-note",
    "youtube-note": "video-note",
    "youtube-video": "video-note",
    "youtube-summary": "video-note",
    "biography": "book-note",
    "book-research": "book-note",
    "collected-works": "book-note",
    "short-story": "book-note",
    "interview-notes": "notes",
    "note": "notes",
    "paper": "notes",
    "paper-note": "notes",
    "project-notes": "notes",
    "transcript": "notes",
    "vault-note": "notes",
    "work-notes": "notes",
    "alias": "infrastructure",
    "directory-index": "infrastructure",
    "handoff": "infrastructure",
    "redirect": "infrastructure",
    "hub": "infrastructure",
    "maintenance-log": "infrastructure",
    "architecture-reference": "agent-pattern",
    "entity-overview": "agent-pattern",
    "strategy": "agent-pattern",
    "benchmark": "gap-analysis",
    "research-evaluation": "gap-analysis",
    "review": "gap-analysis",
}

# What ``validate_okf.py`` treats as a recognised vocabulary. Deliberately a
# superset of the 12: OKF tolerates unknown types, and a value already on disk
# outside ``knowledge/`` is not a reason to start warning. Unknown-to-this-module
# values still warn — that is how the next synonym gets noticed.
KNOWN_TYPES = CANONICAL_TYPES | frozenset(TYPE_ALIASES) | OUTSIDE_KNOWLEDGE_TYPES

# ── classification used by the frontmatter backfill ──────────────────────────
# The images of the three sets ``knowledge-frontmatter-backfill.py`` carried, but
# expressed in canonical values, so a file that has been renamed and a file that
# has not are classified the same way (``normalize_type`` is applied first).
SYNTHESIZED_TYPES = frozenset({
    "research", "research-deep", "research-quick", "synthesis",
})
CAPTURED_TYPES = frozenset({"stack-update"})
PRIMARY_TYPES = frozenset({"notes", "reference", "infrastructure"})


def normalize_type(value: object) -> str:
    """Canonical spelling of a ``type`` value, or the value as-is if unknown.

    Tolerates the casing and separator drift that produced the fragmentation in
    the first place (``Research_Note``, ``research note``); maps an alias to its
    canonical. Idempotent: feeding it a canonical value returns that value, so
    the result is always safe to write back.
    """
    if value is None:
        return ""
    text = str(value).strip().strip("\"'").lower()
    if not text:
        return ""
    for sep in ("_", " ", "\t"):
        text = text.replace(sep, "-")
    if text in CANONICAL_TYPES:
        return text
    return TYPE_ALIASES.get(text, text)
