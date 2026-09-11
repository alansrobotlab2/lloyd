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
  * Reading and writing are different rules, and both live here.
    ``normalize_type`` tolerates anything, because a 2,500-file tree predates
    the map. ``canonical_for_write`` / ``normalize_document_type`` refuse: an
    alias is normalised onto its canonical value and an invented value never
    lands. That half (#872) exists because the tree was consolidated onto this
    vocabulary while every writer — four research skills and the schema
    document — went untouched, and ``vault_write`` checked nothing, which is how
    a single invented ``type: note`` blocked every automod promotion on the box
    (#780).
  * Everything else lives here once. A ``type:`` literal spelled out in
    ``validate_okf.py``, ``okf_migrate.py`` or ``knowledge-frontmatter-backfill.py``
    is precisely the defect this module was written to prevent —
    ``tests/test_okf_type_taxonomy.py`` fails on it.
"""
from __future__ import annotations

import re

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


def _lookup_form(value: object) -> str:
    """Separator/casing-normalised form of a value, *before* alias mapping.

    Split out of ``normalize_type`` so a caller can ask which set a spelling
    fell into (canonical? alias? legal-only-outside?) instead of only getting
    the answer. Returns ``""`` for nothing to look up.
    """
    if value is None:
        return ""
    text = str(value).strip().strip("\"'").lower()
    for sep in ("_", " ", "\t"):
        text = text.replace(sep, "-")
    return text


def normalize_type(value: object) -> str:
    """Canonical spelling of a ``type`` value, or the value as-is if unknown.

    Tolerates the casing and separator drift that produced the fragmentation in
    the first place (``Research_Note``, ``research note``); maps an alias to its
    canonical. Idempotent: feeding it a canonical value returns that value, so
    the result is always safe to write back.

    This is the *reading* rule — it never refuses anything, which is what makes
    it safe for a 2,500-file tree whose spellings predate the map. For writing,
    use ``canonical_for_write``: reading tolerates, writing does not.
    """
    text = _lookup_form(value)
    if not text:
        return ""
    if text in CANONICAL_TYPES:
        return text
    return TYPE_ALIASES.get(text, text)


# ── the writer-side rule (#872) ──────────────────────────────────────────────
class KnowledgeTypeError(ValueError):
    """A ``type`` value that may not be written into ``knowledge/``.

    ``.value`` carries the offending spelling and the message names it, because
    the failure this exists for — backlog #780 — was a knowledge note whose
    ``type`` its writer invented, which surfaced seven hours later as a red
    promotion gate and could not be traced to the value that caused it.
    """

    def __init__(self, value: str) -> None:
        self.value = value
        target = TYPE_ALIASES.get(value)
        if value in OUTSIDE_KNOWLEDGE_TYPES:
            why = (f"`{value}` is a vault-wide value that is legal only *outside* "
                   f"knowledge/" + (f"; inside knowledge/ it means `{target}`" if target else ""))
        elif target:
            why = (f"`{value}` was retired in the #370 consolidation; "
                   f"write `{target}`")
        else:
            why = "no such type exists"
        super().__init__(
            f"type {value!r} may not be written into knowledge/: {why}. The "
            f"vocabulary is okf_taxonomy.CANONICAL_TYPES: "
            f"{', '.join(sorted(CANONICAL_TYPES))}."
        )


def canonical_for_write(value: object) -> str:
    """The ``type`` a ``knowledge/`` file may carry, or ``KnowledgeTypeError``.

    Three outcomes, and the gap between the last two is the point:

      * a canonical value, or a retired spelling in ``TYPE_ALIASES`` — returns
        the canonical value, so a writer that still says ``deep-research`` lands
        ``research-deep`` rather than re-fragmenting the tree (#649's four skill
        templates);
      * a value legitimate only *outside* ``knowledge/`` — refused, even where
        an alias happens to exist (``note`` → ``notes``). Such a value means the
        writer is guessing at the vocabulary instead of carrying an old spelling,
        and silently rewriting a guess is what let #39's invention into the tree;
      * anything else — refused. An invented value cannot reach the tree.

    Retiring ``note`` itself is #442 and this does not decide it; it only
    declines to manufacture a new file that carries it.
    """
    form = _lookup_form(value)
    if not form:
        raise KnowledgeTypeError("")
    if form in CANONICAL_TYPES:
        return form
    if form in OUTSIDE_KNOWLEDGE_TYPES:
        raise KnowledgeTypeError(form)
    canonical = TYPE_ALIASES.get(form)
    if canonical is None:
        raise KnowledgeTypeError(form)
    return canonical


_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(\r?\n|\Z)", re.S)
# Top-level only (^, no indent): an indented ``type:`` is inside some nested
# mapping — e.g. one entry of a ``sources:`` list — and is not the page's type.
_TOP_LEVEL_TYPE_RE = re.compile(r"^type:[ \t]*([^#\r\n]*)", re.M)


def normalize_document_type(text: str) -> tuple[str, str | None]:
    """Rewrite a knowledge note's frontmatter ``type`` onto its canonical value.

    Returns ``(text_to_write, value_that_was_replaced)`` — the second is
    ``None`` whenever the text comes back untouched, so a caller can tell a
    writer what it was quietly corrected away from.

    Hands the text back unchanged when there is no frontmatter, no top-level
    ``type`` key, or an empty one: an *absent* type is OKF's other complaint and
    belongs to #478's 221 orphan-frontmatter files, not to a vocabulary guard.
    Raises ``KnowledgeTypeError`` when a type is present but may not land in
    ``knowledge/`` — the caller must then write nothing.
    """
    fm = _FRONTMATTER_RE.match(text)
    if fm is None:
        return text, None
    block = fm.group(1)
    found = _TOP_LEVEL_TYPE_RE.search(block)
    if found is None:
        return text, None
    raw = found.group(1).strip().strip("\"'")
    if not raw:
        return text, None
    canonical = canonical_for_write(raw)
    if canonical == raw:
        return text, None
    line_end = block.find("\n", found.start())
    if line_end == -1:
        line_end = len(block)
    # Keep any trailing comment: the value was wrong, the reason written next to
    # it usually was not, and dropping it silently loses a line of the author's.
    comment = block[found.end():line_end].strip()
    new_block = (block[:found.start()] + f"type: {canonical}"
                 + (f"  {comment}" if comment else "") + block[line_end:])
    return text[:fm.start(1)] + new_block + text[fm.end(1):], raw


def rejected_document_type(text: str) -> str | None:
    """The ``type`` this note carries that may NOT land as-is, or ``None``.

    The read-only half of ``normalize_document_type``, for a caller that must
    check a file without rewriting it — the vault lander, which commits whatever
    is on disk. That difference makes this rule STRICTER than the write rule, not
    looser: ``vault_write`` may rewrite ``deep-research`` to ``research-deep``
    because it controls the bytes, while a lander that accepted the alias would
    put a non-canonical value straight back into the tree #370 emptied and fail
    ``test_knowledge_frontmatter_uses_only_the_canonical_set`` on the next round.
    So: only a value that is *already* canonical lands, and the refusal names the
    value to write instead. An absent or empty ``type`` is not an answer either
    way — #478's orphan-frontmatter files must stay landable.
    """
    fm = _FRONTMATTER_RE.match(text)
    if fm is None:
        return None
    found = _TOP_LEVEL_TYPE_RE.search(fm.group(1))
    if found is None:
        return None
    raw = found.group(1).strip().strip("\"'")
    if not raw:
        return None
    return None if raw in CANONICAL_TYPES else raw
