"""The research registry's missing-artifact count has a scheduled reader (#4273).

`ResearchStore.stats()["missing_artifacts"]` reports every topic row whose note is gone
from disk, and until #4273 only a chat `research_stats` call or the CLI ever read it.
`workers.fleet_watchdog._check_research_artifacts` is the deterministic reader: through
`app.research_store`, recorded every run, alerting only when the count rises above the
last record, naming the paths that went missing since.

The registry here is the real `ResearchStore`, on the scratch path
`tests/conftest.py::_isolate_research_store` gives every test; the queue is a scratch
`WorkQueue`; `_alert` is captured through monkeypatch.
"""
import ast
import datetime as dt
import inspect
import json
import textwrap

import pytest

import workers.fleet_watchdog as fw
from app import research_store
from workers.queue import WorkQueue

NOW = dt.datetime(2026, 11, 25, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setattr(fw, "_state", {})
    return WorkQueue(str(tmp_path / "watch.db"))


@pytest.fixture
def sent(monkeypatch):
    out: list[str] = []

    async def fake_alert(message):
        out.append(message)

    monkeypatch.setattr(fw, "_alert", fake_alert)
    return out


def _lose(path: str) -> None:
    """A row naming a note that is not on disk — the shape `stats()` counts."""
    s = research_store.store()
    tid = s.propose(f"a topic whose note {path} went missing")["id"]
    with s._connect() as conn:
        conn.execute("UPDATE topics SET status='archived', artifact_path=? WHERE id=?",
                     (path, tid))


def _record(queue) -> dict:
    return json.loads(queue.wm_get(fw._RESEARCH_ARTIFACT_WM_SOURCE,
                                   fw._RESEARCH_ARTIFACT_RECORD_KEY))


def _later(hours: float) -> dt.datetime:
    return NOW + dt.timedelta(hours=hours)


def test_the_check_reads_the_count_and_paths_through_the_store(queue):
    """Clause 1: the figure is the store's own `stats()` payload, and the reader names
    no database file and no subprocess — `app.research_store` is the only opener."""
    _lose("/nonexistent/research/one.md")
    want = research_store.store().stats()["missing_artifacts"]
    got = fw._research_missing_artifacts()
    assert got == {"count": want["count"], "paths": want["paths"]} == {
        "count": 1, "paths": ["/nonexistent/research/one.md"]}
    for fn in (fw._research_missing_artifacts, fw._check_research_artifacts):
        # The executed code only: `ast.unparse` drops comments, and the docstring is
        # removed, since both may NAME the file this reader is forbidden to open.
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        body = tree.body[0].body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            tree.body[0].body = body[1:]
        code = ast.unparse(tree)
        assert "research_store" in code or fn is fw._check_research_artifacts
        for banned in ("sqlite3", "research.db", "subprocess", "_connect(", "open("):
            assert banned not in code, (fn.__name__, banned)


async def test_the_first_run_records_a_baseline_and_alerts_nothing(queue, sent):
    """Clause 2."""
    _lose("/nonexistent/research/a.md")
    _lose("/nonexistent/research/b.md")
    out = await fw._check_research_artifacts(queue, now=NOW)
    assert out["previous"] is None and out["alerted"] is False, out
    assert sent == []
    rec = _record(queue)
    assert rec["count"] == 2
    assert sorted(rec["paths"]) == ["/nonexistent/research/a.md",
                                    "/nonexistent/research/b.md"]


async def test_a_rise_alerts_with_both_counts_and_only_the_new_paths(queue, sent):
    """Clause 3."""
    _lose("/nonexistent/research/old.md")
    await fw._check_research_artifacts(queue, now=NOW)
    _lose("/nonexistent/research/new-1.md")
    _lose("/nonexistent/research/new-2.md")
    out = await fw._check_research_artifacts(queue, now=_later(7))
    assert out["alerted"] is True and len(sent) == 1, (out, sent)
    msg = sent[0]
    assert "from 1 to 3" in msg, msg
    assert "/nonexistent/research/new-1.md" in msg and "/nonexistent/research/new-2.md" in msg
    assert "/nonexistent/research/old.md" not in msg, (
        "a path already missing at the baseline is not news")
    assert sorted(out["new_paths"]) == ["/nonexistent/research/new-1.md",
                                        "/nonexistent/research/new-2.md"]


async def test_a_tie_or_a_fall_alerts_nothing_and_still_rewrites_the_record(queue, sent):
    """Clause 4: no alert unless the count rose, and the record follows every run, so
    the next comparison is against this run and not a stale high-water mark."""
    _lose("/nonexistent/research/x.md")
    _lose("/nonexistent/research/y.md")
    await fw._check_research_artifacts(queue, now=NOW)

    tie = await fw._check_research_artifacts(queue, now=_later(7))
    assert tie["alerted"] is False and sent == []
    assert _record(queue)["at"] == _later(7).isoformat()

    # A fall: one note restored by settling its row onto a path that now exists.
    s = research_store.store()
    with s._connect() as conn:
        conn.execute("UPDATE topics SET artifact_path=NULL WHERE artifact_path=?",
                     ("/nonexistent/research/x.md",))
    fall = await fw._check_research_artifacts(queue, now=_later(14))
    assert fall["alerted"] is False and sent == [], (fall, sent)
    rec = _record(queue)
    assert rec["count"] == 1 and rec["paths"] == ["/nonexistent/research/y.md"]

    # ...so a rise back to 2 is measured from 1, not from the old record of 2.
    _lose("/nonexistent/research/z.md")
    rise = await fw._check_research_artifacts(queue, now=_later(21))
    assert rise["alerted"] is True and "from 1 to 2" in sent[0], sent


async def test_the_cadence_holds_and_an_unreadable_registry_records_nothing(queue, sent):
    """The 6 h cadence (inside clause 5's 24 h bound), and a store that raises is
    recorded as no reading, never as a zero that a later rise would be measured from."""
    assert fw._RESEARCH_ARTIFACT_SCAN_SECONDS <= 24 * 3600
    assert await fw._check_research_artifacts(queue, now=NOW, reader=lambda: {
        "count": 0, "paths": []}) is not None
    assert await fw._check_research_artifacts(queue, now=_later(1)) is None

    def boom():
        raise research_store.StoreUnavailable("registry unreadable")

    assert await fw._check_research_artifacts(queue, now=_later(7), reader=boom) is None
    assert _record(queue)["count"] == 0, "the failed read overwrote the record"
    assert sent == []
