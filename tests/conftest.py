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
