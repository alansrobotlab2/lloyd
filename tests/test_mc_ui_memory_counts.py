"""Mission Control's memory count covers what its memory search searches (#3160).

`_summarize_memory` answers the navigate brief's `vault_file_count` — the number
behind the Memory tab — by walking `_VAULT_SEGMENTS` under the vault root. Until
#3160 that was a second hand-maintained copy of the recall's segment list, in a
tuple, in this router: the same seven names `app/routers/memory.py` was sending
to qmd, `agents` among them and the five real segments it omitted absent. So the
tab's headline number and the set of notes the tab's own search box could return
were decided by two lists that no test read against each other, and the phantom
`agents` entry fell through the loop's own `is_dir()` guard without saying so.

The two lists are now one import of `agent_mcp.vault.VAULT_SEGMENTS`. These tests
pin the count's coverage, and the equality that makes the divergence impossible.
"""
import pytest

from agent_mcp.vault import VAULT_SEGMENTS
from app.routers import mc_ui as MC
from app.routers import memory as MR

# A different number of notes in each segment — 1, 2, … one per segment — so a
# walk that skips one cannot land on the same total. An even count per segment
# would let two segments' worth of omissions cancel out, and a walk that
# double-counted one segment while missing another would pass.
NOTES = {segment: i + 1 for i, segment in enumerate(VAULT_SEGMENTS)}


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A tmp vault seeded to `NOTES`, with this router pointed at it."""
    root = tmp_path / "obsidian"
    for segment, n in NOTES.items():
        seg_dir = root / segment
        seg_dir.mkdir(parents=True)
        for i in range(n):
            (seg_dir / f"note{i}.md").write_text(
                f"---\ntitle: {segment} note {i}\n---\n\nBody.\n", encoding="utf-8")
    monkeypatch.setattr(MC, "_VAULT", root)
    return root


def test_navigate_memory_count_covers_every_searchable_segment(vault):
    """#3160 clause 4: `vault_file_count` counts the notes in every segment the
    memory search can search — including `backlog/` and `architecture/`, which
    the seven-name tuple left out."""
    assert MC._summarize_memory() == {"vault_file_count": sum(NOTES.values())}, (
        f"`vault_file_count` is {MC._summarize_memory().get('vault_file_count')} over a "
        f"vault seeded with {sum(NOTES.values())} notes spread one segment per "
        "distinct count, so some segment is not being walked")


def test_dropping_a_segment_s_notes_moves_the_number_by_theirs(vault):
    """Per-segment sensitivity, which a total alone cannot show: delete the notes
    of the two segments #3160 names and the count has to fall by exactly their
    shares — no more (double-counting the segment) and no less (never having
    counted it)."""
    full = MC._summarize_memory()["vault_file_count"]
    dropped = NOTES["backlog"] + NOTES["architecture"]
    for segment in ("backlog", "architecture"):
        for f in (vault / segment).glob("*.md"):
            f.unlink()
    after = MC._summarize_memory()["vault_file_count"]
    assert full - after == dropped, (
        f"deleting the backlog and architecture notes moved the count by "
        f"{full - after}, not their {dropped} notes")


def test_the_counted_set_and_the_searched_set_are_one_list():
    """The clause's real guarantee: the Memory tab's number and its searchable
    set cannot diverge, because there is one list behind both. A third copy in
    either router — or a name that is a qmd collection but not a segment, like
    `sessions` — fails this by construction."""
    assert list(MC._VAULT_SEGMENTS) == list(VAULT_SEGMENTS), (
        f"`mc_ui.py` counts {list(MC._VAULT_SEGMENTS)} while the recall searches "
        f"{list(VAULT_SEGMENTS)}")
    assert list(MC._VAULT_SEGMENTS) == list(MR._VAULT_SEGMENTS), (
        "the two routers are back to restating the set independently: the count and "
        "the search will diverge again with nothing to notice it")
