"""The component registry's counters are readable over HTTP (#1880 clause 4).

`stats()` has carried `evictions` since #1782, and the round that added it
justified the counter as "readable without opening the confidential store" —
which was true of the test that pinned it and of nothing else. Outside
`app/component_manifest.py` and its tests the module is imported at nine call
sites and every one of them calls `record_request`,
`components_from_payload`, `note_components` or `note_prefetch`; no router
imports it, and the `/health` payload of the time ended at `checks_failed`. So
the drop rate the 1024-entry bound was counted against was readable only by
importing Lloyd's internals into a scratch process — the one route the store's
own `POLICY.md` ("treat it with the same confidentiality as ~/obsidian") makes
you think twice about.

What the clause needed already exists: `/health` is the pure in-memory root GET
the guardian and the self-mod promoter's idle gate poll
(`app/routers/health.py`), so the counters go on it as one more section. Each
node here drives that route through the real backend app:

* **every counter is on the surface** — nothing the module keeps may be missing
  from the section, and `evictions` is named explicitly
  (`test_health_reports_every_counter_the_module_keeps`);
* **a fresh process reports zero** — the state a boot is in
  (`test_a_fresh_process_reports_no_evictions`);
* **the number is live, not a literal** — forcing the bound to drop three
  sessions moves the endpoint by exactly three
  (`test_the_section_moves_when_the_bound_actually_drops`);
* **it is the read-only route that was already mounted** — a GET on `/health`,
  one entry in the route table, and the counters unmoved by the request that
  read them (`test_reading_the_section_is_the_same_read_only_get_it_was`).

Against HEAD before this diff all four fail on the same fact, which is the
item's own premise: the `/health` response has no `manifest` key. Measured here
by booting the app in the pre-change tree — `has manifest: False`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import component_manifest as cm  # noqa: E402


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """An own store and a clean process: the counters and the registry are
    module globals, so a leftover entry from another file would move the very
    number these nodes read."""
    monkeypatch.setenv("LLOYD_MANIFEST_STORE", str(tmp_path / "manifest-store"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    cm.reset_stats()
    cm._reset_registry()
    yield tmp_path / "manifest-store"
    cm._reset_registry()
    cm.reset_stats()


@pytest.fixture
def health_client(monkeypatch):
    """The real backend app, as `server.py` mounts it, without its startup hooks.

    `TestClient(app)` outside its context manager sends requests but never runs
    the lifespan, so this is the route table and the handler the backend serves
    with none of its tickers, pools or MCP connections behind it.
    `_startup_complete` belongs to the module's own boot hook; setting it through
    the documented `mark_startup_complete` path is what makes the answer the 200
    a poller sees, and `monkeypatch` puts it back.
    """
    import server
    from fastapi.testclient import TestClient

    from app.routers import health
    monkeypatch.setattr(health, "_startup_complete", True)
    return TestClient(server.app)


def test_health_reports_every_counter_the_module_keeps(health_client):
    """The section must carry the whole vocabulary, not the one field this item names.

    A surface that exposes `evictions` alone would answer this clause and leave
    the next question — did the writer fall behind, did a prune delete what it
    should not — back where it was: in-process. So the assertion is over the
    set, and a counter added to `_stats` later has to appear here too or this
    node goes red.
    """
    counters = cm.stats()
    assert "evictions" in counters, sorted(counters)

    body = health_client.get("/health")
    assert body.status_code == 200, body.status_code
    section = body.json().get("manifest")
    assert isinstance(section, dict), (
        f"`/health` has no manifest section: {sorted(body.json())}")

    missing = set(counters) - set(section)
    assert not missing, (
        f"counters the module keeps but the health section does not publish: "
        f"{sorted(missing)}")
    for key, value in counters.items():
        assert section[key] == value, f"{key}: the section is not `stats()`"
    assert section["max_sessions"] == cm.MAX_SESSIONS, section
    assert section["sessions"] == 0, section


def test_a_fresh_process_reports_no_evictions(health_client):
    """The clause's own reading: a process that has served nothing says zero.

    `_stats` starts at zero and `_registry` starts empty, which is what
    `reset_stats()` and `_reset_registry()` in the fixture leave behind — the
    same state a freshly started backend is in. The node above checks the
    section agrees with the module; this one checks the number a boot reports
    is 0 rather than inherited, negative or absent.
    """
    section = health_client.get("/health").json()["manifest"]
    assert section["evictions"] == 0, section
    assert section["sessions"] == 0, section
    assert section["recorded"] == 0, section


def test_the_section_moves_when_the_bound_actually_drops(health_client):
    """The falsifiable half: the endpoint reports the counter, not a literal zero.

    Note one session more than the bound holds, three times over, and the
    section's `evictions` has to have moved by exactly three — the same three
    `stats()` counts. Reading the delta rather than the absolute is what makes
    this survive another test's writes in a shared process; `sessions` is read
    absolutely because the registry is the fixture's to keep empty.
    """
    before = health_client.get("/health").json()["manifest"]["evictions"]
    for n in range(cm.MAX_SESSIONS + 3):
        cm.note_components(f"session-{n}", {"system_prompt": "p"})

    section = health_client.get("/health").json()["manifest"]
    assert section["evictions"] - before == 3, (
        f"three entries went over the bound and the section moved by "
        f"{section['evictions'] - before}: the number on the wire is not the "
        "registry's drop count")
    assert section["evictions"] == cm.stats()["evictions"], section
    assert section["sessions"] == cm.MAX_SESSIONS, (
        f"the occupancy gauge reads {section['sessions']} against a cap of "
        f"{cm.MAX_SESSIONS} — a drop rate beside the wrong denominator is not a "
        "rate")


def test_reading_the_section_is_the_same_read_only_get_it_was(health_client):
    """No new surface, and no side effect: `/health` is watched every few seconds.

    The clause says "an already-mounted read-only health GET", so the section
    may not arrive on a route of its own: `/health` appears exactly once in the
    backend's route table and its only method is GET. And because the guardian
    and the promoter poll it, reading the diagnostics must not itself be an
    event — every counter is unchanged across the request that read them.
    """
    paths = [getattr(r, "path", "") for r in health_client.app.routes]
    health_routes = [p for p in paths if p == "/health"]
    assert health_routes == ["/health"], (
        f"`/health` is mounted {len(health_routes)} times over, so which one a "
        "poller reaches is an accident")
    route = next(r for r in health_client.app.routes
                 if getattr(r, "path", "") == "/health")
    assert set(route.methods) == {"GET"}, route.methods

    before = cm.stats()
    response = health_client.get("/health")
    assert response.status_code == 200, response.status_code
    assert cm.stats() == before, "reading the health endpoint changed the counters"
    assert response.json()["manifest"]["evictions"] == before["evictions"]
