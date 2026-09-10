"""Test isolation from live runtime state.

The suite must not depend on the state of the running system. Two things
leaked in and broke tests that had nothing to do with them:

  * `config.yaml knowledge_graph.write_enabled`, which a knowledge-graph
    rebuild sets to false — six fact-write tests started failing because a
    rebuild was in progress on the machine.
  * `app.kg_store`'s process-default store, which points at the live
    database unless a test configures it.

Both are forced to a known value here. A test that wants the other value
patches it explicitly.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _no_voice_alerts_in_tests(monkeypatch):
    """The guardian's sixth channel synthesises speech and plays it aloud.

    `Notifier.alert` fans out to it like any other channel, so without this an
    ordinary `pytest tests/` would talk to the room — and, worse, would do it
    from a detached process that outlives the test. Muted for every test; the
    ones that exercise the channel assert on the dispatch decision instead.
    """
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "0")


@pytest.fixture(autouse=True)
def _no_desktop_or_journal_alerts_in_tests(monkeypatch):
    """The other two channels that reach the room from a test process.

    `Notifier`'s `external` gate suppresses vault notes and backlog tasks, and
    was added after drill rollbacks polluted the live vault on 2026-09-06. It
    never covered the toast or the journal line, because those need nothing
    but a session bus — so a test that builds a default `Notifier` and only
    cares whether *voice* dispatched still paints the user's screen.

    That is what happened on 2026-09-07: every self-mod gate run toasted
    `Lloyd guardian: real rollback / body` and wrote
    `STILL BROKEN :: 2026-09-06 liveness failed` to the live journal at
    priority 2, from the fixture strings in `test_guardian_speak.py` and
    `test_guardian_predicates.py`. A fake critical incident in the journal is
    worse than a stray toast: it is the record you consult *after* an
    incident, and it now contains fiction.

    DBUS is blanked as well as the switch flipped, so the nag unit's
    bash `notify-send` fallback — which reads the environment, not
    `LLOYD_DESKTOP_ALERTS` — cannot fire from a test either. Tests that assert
    on the channel's dispatch decision set these explicitly.
    """
    monkeypatch.setenv("LLOYD_DESKTOP_ALERTS", "0")
    monkeypatch.setenv("LLOYD_JOURNAL_ALERTS", "0")
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)


@pytest.fixture(autouse=True)
def _writes_enabled_in_tests(monkeypatch):
    """Fact writes are on unless a test says otherwise."""
    try:
        from agent_mcp import facts as facts_mod
    except Exception:          # module not importable in this test's env
        return
    monkeypatch.setattr(facts_mod, "_writes_enabled", lambda: True, raising=False)


@pytest.fixture(autouse=True)
def _isolate_default_store(request, tmp_path_factory):
    """No test writes to the live knowledge-graph store.

    Tests that need a store call `kg_store.configure(...)` themselves; this
    only guarantees the *default* never resolves to the production file, and
    puts it back afterwards.
    """
    from app import kg_store
    original = kg_store._default_path
    kg_store.reset()
    kg_store._default_path = tmp_path_factory.mktemp("kg") / "kg.sqlite"
    yield
    kg_store.reset()
    kg_store._default_path = original


@pytest.fixture(autouse=True)
def _isolate_research_store(tmp_path_factory):
    """No test writes to the live research registry.

    Same shape and same reason as `_isolate_default_store` above: the default
    must never resolve to `~/lloyd/research.db`, because a test that proposes
    a topic would otherwise put it in front of the deep-research worker. Tests
    that want a registry call `research_store.configure(...)` themselves.
    """
    from app import research_store
    original = research_store._default_path
    research_store.reset()
    research_store._default_path = tmp_path_factory.mktemp("research") / "research.db"
    yield
    research_store.reset()
    research_store._default_path = original


@pytest.fixture(autouse=True)
def _isolate_background_records(tmp_path_factory, monkeypatch):
    """No test writes a session or an event log into the live tree.

    Recording became universal on 2026-09-10: every autonomy run and every
    `run_prompt_on_primary` worker turn now writes a session JSON and an event
    log. A test that drives one of those paths — and several do, for reasons
    that have nothing to do with recording — would otherwise leave a real
    transcript in `~/lloyd/sessions/`, where it shows up in the history list,
    in session recall and in the retention sweep.

    Same shape as the two isolators above: the *default* never resolves to the
    production directory. A test that patches these itself still wins, because
    its `monkeypatch` runs after this fixture's.
    """
    root = tmp_path_factory.mktemp("lloyd-records")
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", root / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", root / "event_logs")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", root / "event_logs" / "blobs")


@pytest.fixture(autouse=True)
def _isolate_usage_store(tmp_path_factory, monkeypatch):
    """No test writes a usage row into the live `usage.db`.

    `usage_store.DB_PATH` resolves from `__file__`, so in `~/lloyd` it is the
    production database behind the dashboard's token panel and its
    prefix-miss counter. Until 2026-09-10 only the chat path wrote rows; the
    background recorder writes one per run now, and several tests drive it.
    `usage_store._conn` reopens when `DB_PATH` moves, so the per-thread
    connection cache cannot carry a test's writes into the next one's file.
    """
    monkeypatch.setattr("usage_store.DB_PATH",
                        tmp_path_factory.mktemp("usage") / "usage.db")
