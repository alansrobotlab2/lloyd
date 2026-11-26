"""Serve the conversation linker's note-to-note edges where a document is the unit (#4033).

`scripts/memory/conversation_relations.py` lands its accepted proposals as typed
edges with `origin='conversation'`, and every one of them joins a vault note path
to a vault note path (`autonomy/82-nightly-retrieval-eval.md --implements-->
skills/retrieval-eval/SKILL.md`). Every graph reader starts from an entity name
and none of the 494 endpoints is a registered entity, so until this module no
recall path could reach them. They are served here instead, beside the note
itself: `vault_read` of a note, and each document `vault_recall` returns, carries
`linked_notes` — at most `LINKS_CAP` `{path, type, direction}` rows, highest
confidence first, `direction` "out" when the note is the edge's source and "in"
when it is the target.

Cost: one `store().edges.active(either=<path>)` per note — an indexed lookup on
`edges_source` / `edges_target` — and nothing else.

A store that cannot be read is NOT "no links". `app.kg_store` raises
`StoreUnavailable` precisely so a failure never reads as an empty graph, and the
note read must not fail with it either: the read is about the note, the links
are an addition to it. So the caller gets the note and, in place of
`linked_notes`, a `linked_notes_unavailable` string naming the failure. An absent
`linked_notes` with no such marker means the store answered and the note has no
active conversation edge.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("lloyd-vault")

#: The writer's `EDGE_ORIGIN` (scripts/memory/conversation_relations.py). Restated,
#: not imported: the aggregator does not import scripts/, and
#: tests/test_document_link_serving.py pins the two spellings equal.
LINK_ORIGIN = "conversation"
LINKS_CAP = 8
LINKS_KEY = "linked_notes"
LINKS_UNAVAILABLE_KEY = "linked_notes_unavailable"


class LinksUnavailable(Exception):
    """The store could not answer; `str()` is the reason to hand the caller."""


def linked_notes(path: str, st: Any = None) -> list[dict]:
    """The notes `path`'s active conversation edges join it to, best first.

    Raises `LinksUnavailable` when the store cannot be opened or read — never
    returns `[]` for a failure.
    """
    try:
        from app import kg_store
        st = st if st is not None else kg_store.store()
        rows = st.edges.active(either=path)
    except Exception as exc:  # noqa: BLE001 — StoreUnavailable, or a store that would not open
        raise LinksUnavailable(f"knowledge graph store unavailable "
                               f"({type(exc).__name__}: {exc})") from exc
    rows = [e for e in rows if e.get("origin") == LINK_ORIGIN]
    rows.sort(key=lambda e: (-float(e.get("confidence") or 0.0), int(e.get("id") or 0)))
    out: list[dict] = []
    seen: set[tuple] = set()
    for e in rows:
        if e.get("source") == path:
            other, direction = e.get("target"), "out"
        else:
            other, direction = e.get("source"), "in"
        if not other or other == path:
            continue
        key = (other, e.get("type"), direction)
        if key in seen:
            continue
        seen.add(key)
        out.append({"path": other, "type": e.get("type"), "direction": direction})
        if len(out) >= LINKS_CAP:
            break
    return out


def attach_to_read(result: dict, path: str) -> dict:
    """`vault_read`'s result with the note's links, or the unavailability marker.

    A note with no active conversation edge gets the result back unchanged.
    """
    try:
        links = linked_notes(path)
    except LinksUnavailable as exc:
        logger.warning("vault_read %s: linked notes not served: %s", path, exc)
        return {**result, LINKS_UNAVAILABLE_KEY: str(exc)}
    return {**result, LINKS_KEY: links} if links else result


def attach_to_recall(result: dict) -> dict:
    """Put `linked_notes` on each of `result["documents"]` that has any, in place.

    The first store failure stops the lookups (the next nineteen would fail the
    same way) and is named once, at the result's top level.
    """
    for doc in result.get("documents") or []:
        if not isinstance(doc, dict) or not doc.get("path"):
            continue
        try:
            links = linked_notes(str(doc["path"]))
        except LinksUnavailable as exc:
            logger.warning("vault_recall: linked notes not served: %s", exc)
            result[LINKS_UNAVAILABLE_KEY] = str(exc)
            break
        if links:
            doc[LINKS_KEY] = links
    return result
