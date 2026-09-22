"""Backlog #933 — the Memory-tab routes end a front-matter block at a
line-anchored `---` over the whole file, not at a byte-capped prefix and not at
the first `---` substring.

Two defects, both live in `app/routers/memory.py` before this fix:

  * `read_text(...)[:2000]` (stats) and `[:500]` (browse) cap the text before
    the fence is looked for, so a note whose front matter runs past the cap has
    no closing fence in hand and is silently skipped. Measured 2026-09-22 over
    the seven vault segments: 18 notes lose their tags and 32 more are listed
    by filename in the tree although their front matter carries a `title:` key.
    Counting every `title:` substring past byte 500 instead of a parsed
    top-level key gives 178 — that overstates the visible symptom 5x and is
    the number this file deliberately does not use.
  * `str.split("---", 2)` cuts on a `---` anywhere, including one inside a
    quoted value — a markdown table row (`|---|---|`) is the common one — so a
    well-formed note is sliced mid-scalar and reports as unparseable.

Every test here goes through `TestClient` over the real router, because the
only consumers of these four handlers are HTTP: `web/src/api.ts` (the Memory
tab) and nothing imports the handlers as Python (depth-1 importer is
`server.py:319`, `include_router`). A direct call on the coroutine would not
prove the JSON a browser receives.
"""

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.routers.memory as MR

# Same matcher the fix installs, used here to pin the fixtures: a fixture whose
# fence is NOT past its threshold would silently stop testing anything.
_FENCE = re.compile(r"^---[ \t]*$", re.M)


def _closing_fence(text: str) -> int:
    """Byte offset of the closing `---`, or -1 if the block never closes."""
    m = _FENCE.search(text, 3)
    return m.start() if m else -1


# --- fixtures ----------------------------------------------------------------
# A short note: parses correctly under the old code too, so it is the control
# that keeps every assertion below from passing on "counts everything" or
# "counts nothing".
SHORT_NOTE = (
    "---\n"
    "title: Short note\n"
    "tags: [shortnote]\n"
    "---\n"
    "\n"
    "# Short body\n"
)

# Front matter whose closing fence sits at byte 2459 — past the stats route's
# 2,000-char prefix, so the old code never saw the closing fence.
LONG_FENCE_NOTE = (
    "---\n"
    "title: Long front matter\n"
    "tags: [longfence]\n"
    'summary: "' + ("padding " * 300) + '"\n'
    "---\n"
    "\n"
    "# Long front matter body\n"
)

# Front matter whose closing fence sits at byte 647 — past the browse route's
# 500-char prefix. Past 500 is the threshold that matters for the tree listing.
FENCE_PAST_500_NOTE = (
    "---\n"
    "title: Fence past five hundred\n"
    'summary: "' + ("pad " * 150) + '"\n'
    "---\n"
    "\n"
    "# Past five hundred body\n"
)

# Same length, no `title:` key: the tree must fall back to the filename stem.
NO_TITLE_PAST_500_NOTE = (
    "---\n"
    "tags: [untitled]\n"
    'summary: "' + ("pad " * 150) + '"\n'
    "---\n"
    "\n"
    "# No title body\n"
)

# A `|---|---|` table row inside a *quoted* front-matter value. The `---` run
# is never at the start of a line, so it is not a fence — but `split("---")`
# cuts there and the remaining half of the quoted scalar fails to parse.
TABLE_ROW_NOTE = (
    "---\n"
    'title: "Column widths |---|---|"\n'
    "tags: [tablerow]\n"
    "---\n"
    "\n"
    "# Table row body\n"
    "\n"
    "Nothing to do with the front matter.\n"
)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A tmp vault with every stats segment present, and no cached payload.

    `_STATS_CACHE` is module state: without the reset a payload cached by an
    earlier call answers the assertion instead of the file just written.
    """
    root = tmp_path / "obsidian"
    for segment in MR._VAULT_SEGMENTS:
        (root / segment).mkdir(parents=True)
    monkeypatch.setattr(MR, "_VAULT", root)
    monkeypatch.setattr(MR, "_STATS_CACHE", {"sig": None, "payload": None})
    return root


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(MR.router)
    with TestClient(app) as c:
        yield c


def stats_tags(client, vault) -> dict:
    """`{tag: count}` from GET /api/memory/stats over the tmp `vault`.

    The `vault` fixture is a dependency for its monkeypatches, not this body.
    """
    body = client.get("/api/memory/stats").json()
    counts = {t["tag"]: t["count"] for t in body["topTags"]}
    assert body["tagCount"] == len(counts), "topTags truncated below tagCount"
    return counts


# --- clause 1: fence beyond 2,000 chars still counts its tags ----------------


def test_the_fixture_fence_really_is_past_the_stats_route_prefix():
    assert _closing_fence(LONG_FENCE_NOTE) == 2459
    assert _closing_fence(FENCE_PAST_500_NOTE) == 647
    assert _closing_fence(LONG_FENCE_NOTE) > 2000
    assert _closing_fence(FENCE_PAST_500_NOTE) > 500


def test_stats_counts_the_tags_of_a_note_whose_fence_is_past_2000_chars(client, vault):
    (vault / "knowledge" / "short.md").write_text(SHORT_NOTE, encoding="utf-8")
    (vault / "knowledge" / "long.md").write_text(LONG_FENCE_NOTE, encoding="utf-8")

    counts = stats_tags(client, vault)

    # Exactly the two notes' tags: the long one is counted identically to the
    # short control, and nothing else leaks in.
    assert counts == {"shortnote": 1, "longfence": 1}


# --- clause 2: a line-internal `---` inside a quoted value does not skip ------


def test_stats_counts_a_note_whose_front_matter_quotes_a_table_row(client, vault):
    (vault / "knowledge" / "short.md").write_text(SHORT_NOTE, encoding="utf-8")
    (vault / "knowledge" / "table.md").write_text(TABLE_ROW_NOTE, encoding="utf-8")

    counts = stats_tags(client, vault)

    assert counts == {"shortnote": 1, "tablerow": 1}


# --- clause 3: browse titles past 500 chars, filename fallback kept ----------


def test_browse_uses_the_title_past_500_chars_and_falls_back_without_one(client, vault):
    (vault / "knowledge" / "titled.md").write_text(FENCE_PAST_500_NOTE, encoding="utf-8")
    (vault / "knowledge" / "untitled.md").write_text(NO_TITLE_PAST_500_NOTE, encoding="utf-8")

    entries = client.get("/api/memory/browse", params={"path": "knowledge"}).json()["entries"]
    titles = {e["name"]: e["title"] for e in entries}

    assert titles == {
        "titled.md": "Fence past five hundred",   # from front matter, past the 500 cap
        "untitled.md": "untitled",                # no `title:` key -> filename stem
    }


# --- clause 4: read returns the body, not front-matter text ------------------


def test_read_returns_the_body_and_frontmatter_of_a_note_quoting_a_table_row(client, vault):
    path = "knowledge/table.md"
    (vault / "knowledge" / "table.md").write_text(TABLE_ROW_NOTE, encoding="utf-8")

    data = client.get("/api/memory/read", params={"path": path}).json()

    assert data["content"] == "# Table row body\n\nNothing to do with the front matter."
    assert data["content"].startswith("# Table row body")
    # The old mis-slice returned the tail of the quoted scalar here — 45 lines
    # of one real backlog note's front matter leaked into the editor body.
    assert "|---|---|" not in data["content"]
    assert "tags:" not in data["content"]
    assert data["frontmatter"] == {"title": "Column widths |---|---|", "tags": ["tablerow"]}


# --- clause 5: save keeps accepting a quoted fence run, still rejects junk ----


def test_save_accepts_embedded_front_matter_quoting_a_fence_run(client, vault):
    content = (
        "---\n"
        'title: "Keep this |---|---| row"\n'
        "tags: [saved]\n"
        "---\n"
        "\n"
        "# Saved body\n"
    )

    r = client.post("/api/memory/save", json={"path": "knowledge/saved.md", "content": content})

    assert r.status_code == 200, r.text
    assert (vault / "knowledge" / "saved.md").read_text(encoding="utf-8") == content


def test_save_still_rejects_invalid_front_matter_yaml(client, vault):
    # An unterminated quoted scalar: invalid whether or not the fence is found.
    content = '---\ntitle: "unterminated\n---\n\n# Body\n'

    r = client.post("/api/memory/save", json={"path": "knowledge/bad.md", "content": content})

    assert r.status_code == 422, r.text
    assert "Invalid frontmatter YAML" in r.text
    assert not (vault / "knowledge" / "bad.md").exists()
