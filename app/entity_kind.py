"""What kind of thing an entity is. One rule, shared.

The v4 classifier had its own version whose PERSON test ran before its
SYSTEM test, so any Capitalized Two-Word name matched `First Last` first:
`Knowledge Graph`, `Claude Code` and `Isaac Lab` were all typed PERSON, and
the classifier used that type to gate relationship verbs — `created_by`
between two systems is nonsense, and it was being allowed.

Order here is: the document that produced the entity, then unambiguous
shapes (file, task, skill), then SYSTEM, then PERSON. A person's name is the
weakest signal of the three because it is the least constrained shape.
"""
from __future__ import annotations

import re

# Kinds the UI legends and the store's `entities.kind` column use.
KINDS = ("person", "project", "system", "concept", "skill", "task", "doc", "entity")

# Vault directory → kind. The strongest signal there is: someone filed the
# source document under it.
_SOURCE_KIND = {
    "people": "person",
    "projects": "project",
    "work": "project",
    "knowledge": "concept",
    "personal": "concept",
    "skills": "skill",
}

_FILE_EXTS = (".md", ".py", ".json", ".yaml", ".yml", ".sh", ".ts", ".tsx", ".js")

# Ends in one of these → a system/component, whatever its capitalisation.
_SYSTEM_TAIL = (
    "system", "agent", "sdk", "service", "engine", "pipeline", "loop",
    "orchestrator", "controller", "server", "daemon", "worker", "harness",
    "store", "index", "api", "cli", "runtime", "scheduler",
)

# `Task #67`, `Autonomy Task 33`, `backlog_item_235`, `run_39a`. Underscores
# are word characters, so `\b` will not find `item` inside `backlog_item_235`
# — the boundary is written out as "start, or a non-alphanumeric".
_TASK_RE = re.compile(r"(?:^|[^a-z0-9])(task|item|run|backlog)[ _#-]*\d", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"^\d{8}[_-]\d{6}")
_SKILL_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)+$")
_CAMEL_RE = re.compile(r"[A-Z][a-z]+[A-Z]")
# A single token carrying internal capitals or digits is a product name:
# vLLM, GR00T, SQLite, PyTorch.
_PRODUCT_TOKEN_RE = re.compile(r"^(?=.*[A-Z0-9])(?=.*[a-z])[A-Za-z0-9.+-]+$")
# NECESSARY but not sufficient for a person. `Isaac Lab`, `Claude Code` and
# `Knowledge Graph` all satisfy this shape, which is exactly why the old rule
# typed them PERSON and let the classifier propose `created_by` between two
# systems. A name only reads as a person with corroboration: the source
# document is under people/, or the vault holds a people note for it.
_PERSON_SHAPE_RE = re.compile(r"^[A-Z][a-z]+(\s+[A-Z]\.?)?(\s+[A-Z][a-z]+){1,2}$")

_CONCEPT_TOKENS = frozenset({
    "memory", "retrieval", "reasoning", "planning", "safety", "alignment",
    "latency", "throughput", "embedding", "attention", "quantization",
    "architecture", "protocol", "benchmark", "dataset", "policy",
})
_ROLE_TOKENS = frozenset({
    "user", "developer", "engineer", "researcher", "author", "maintainer",
    "reviewer", "operator", "admin", "owner", "assistant",
})

_people_cache: set[str] | None = None


def known_people() -> set[str]:
    """Lowercased stems of the vault's `people/` notes.

    The corroboration a person-shaped name needs. Read once per process;
    the set changes when someone writes a note, which is rare enough that a
    restart is an acceptable refresh.
    """
    global _people_cache
    if _people_cache is None:
        try:
            from app.paths import VAULT_ROOT
            _people_cache = {p.stem.lower().replace("-", " ").replace("_", " ")
                             for p in (VAULT_ROOT / "people").rglob("*.md")}
        except Exception:
            _people_cache = set()
    return _people_cache


def reset_people_cache() -> None:
    global _people_cache
    _people_cache = None


def derive_kind(name: str, source_doc: str | None = None) -> str:
    """One of KINDS. Pure string rules, no LLM."""
    n = (name or "").strip()
    if not n:
        return "entity"

    # 1. Where the fact came from beats any guess about the name.
    if source_doc:
        top = str(source_doc).lstrip("./").split("/", 1)[0].lower()
        if top in _SOURCE_KIND:
            return _SOURCE_KIND[top]

    low = n.lower()

    # 2. Unambiguous shapes.
    if _TASK_RE.search(low) or _TIMESTAMP_RE.match(n):
        return "task"
    if low.endswith(_FILE_EXTS) or ("/" in n and not n.startswith("http")):
        return "doc"
    if _SKILL_RE.fullmatch(n):
        return "skill"

    # 3. SYSTEM before PERSON. `Knowledge Graph`, `Claude Code`, `Isaac Lab`
    #    all satisfy the person shape and are not people.
    tokens = low.split()
    if tokens and tokens[-1] in _SYSTEM_TAIL:
        return "system"
    if any(t in _SYSTEM_TAIL for t in tokens):
        return "system"
    if _CAMEL_RE.search(n):
        return "system"
    if len(tokens) == 1 and _PRODUCT_TOKEN_RE.fullmatch(n) and not n.istitle():
        return "system"     # vLLM, GR00T, PyTorch

    # 4. Concepts and roles are named by their vocabulary.
    if low in _ROLE_TOKENS or (len(tokens) == 1 and low in _CONCEPT_TOKENS):
        return "concept"
    if any(t in _CONCEPT_TOKENS for t in tokens):
        return "concept"

    # 5. Person, only with corroboration. Shape alone is not enough.
    if _PERSON_SHAPE_RE.fullmatch(n) and not any(c.isdigit() for c in n):
        if low.replace("-", " ").replace("_", " ") in known_people():
            return "person"

    # A capitalised multiword that is none of the above is a made thing far
    # more often than a person — the vault has 31 people notes and thousands
    # of systems, tools and papers.
    if " " in n and n[0].isupper():
        return "system"
    return "entity"


# The classifier speaks in upper case and has two kinds this module folds
# into others; keep the mapping in one place.
_TO_CLASSIFIER = {
    "person": "PERSON", "project": "SYSTEM", "system": "SYSTEM",
    "concept": "CONCEPT", "skill": "SKILL", "task": "TASK",
    "doc": "FILE", "entity": "ENTITY",
}


def derive_entity_type(name: str, source_doc: str | None = None) -> str:
    """`derive_kind` in the classifier's uppercase vocabulary."""
    return _TO_CLASSIFIER.get(derive_kind(name, source_doc), "ENTITY")
