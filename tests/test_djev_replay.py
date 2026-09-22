"""djev's replay: one answer per request inside one comparison.

djev does not repeat itself. On 2026-09-21 a 32-row recall rank replayed
straight at vLLM kept its argmax at all 123 canvas positions while the label
logprobs the rank score is built from moved by 1-3 nats between identical
requests, and two promotions that touched no retrieval code were rolled back
for the difference. The regression check now runs its arms under one replay
file (`app.djev.replay_env`). These tests pin the client half: an identical
request is answered once, a moved one is drawn, a failure is counted and never
stored, and nothing replays outside an eval.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from app import djev


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(djev, "enabled", lambda: True)
    for name in (djev.REPLAY_ENV, djev.REPLAY_ARM_ENV, djev.REPLAY_ANCHOR_ENV):
        monkeypatch.delenv(name, raising=False)
    djev.reset_stats()
    yield
    djev.reset_stats()


class _NoisyServer:
    """A djev that never gives the same order twice, like the real one."""

    def __init__(self):
        self.calls = 0
        self.fail = False

    def __call__(self, req, timeout=None):
        self.calls += 1
        if self.fail:
            raise urllib.error.URLError("connection refused")
        n = len(json.loads(req.data)["questions"])
        # Rotate the best candidate on every call.
        best = self.calls % n
        answers = {f"c{i}": {"type": "score", "score": 0.9 if i == best else 0.1,
                             "legend": {"0": "a", "1": "b"},
                             "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.5}
                   for i in range(n)}
        body = {"model": "djev", "answers": answers,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "diagnostics": {"timing": {"total_ms": 40.0, "reads": 1},
                                "chunks": [list(answers)],
                                "questions": {k: {"label_mass": 0.9, "argmax_is_label": True}
                                              for k in answers}}}

        class _Resp:
            def read(self):
                return json.dumps(body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return _Resp()


@pytest.fixture
def server(monkeypatch):
    s = _NoisyServer()
    monkeypatch.setattr(djev.urllib.request, "urlopen", s)
    return s


def _arm(monkeypatch, db, arm, anchor="baseline"):
    for k, v in djev.replay_env(db, arm, anchor).items():
        monkeypatch.setenv(k, v)


def _top(rows):
    return rows[0]["index"]


CANDIDATES = ["alpha", "beta", "gamma"]


def test_the_real_server_shape_this_file_fakes_is_noisy(server):
    """The fake has to reproduce the defect, or nothing below means anything."""
    firsts = {_top(djev.rank("q", CANDIDATES)) for _ in range(3)}
    assert len(firsts) == 3 and server.calls == 3


def test_an_identical_request_is_answered_once_across_arms(monkeypatch, tmp_path, server):
    db = tmp_path / "replay.sqlite"
    _arm(monkeypatch, db, "baseline")
    first = djev.rank("q", CANDIDATES)
    _arm(monkeypatch, db, "current")
    again = djev.rank("q", CANDIDATES)
    assert [r["index"] for r in again] == [r["index"] for r in first]
    assert server.calls == 1
    assert djev.replay_stats(db) == {"baseline": {"fresh": 1},
                                     "current": {"replayed_anchor": 1}}


def test_a_request_only_this_arm_asks_is_a_fresh_draw(monkeypatch, tmp_path, server):
    db = tmp_path / "replay.sqlite"
    _arm(monkeypatch, db, "baseline")
    djev.rank("q", CANDIDATES)
    _arm(monkeypatch, db, "current")
    djev.rank("q", CANDIDATES + ["delta"])      # the change moved the ranker's input
    assert server.calls == 2
    assert djev.replay_stats(db)["current"] == {"fresh": 1}


def test_the_confirm_arm_redraws_what_the_change_moved(monkeypatch, tmp_path, server):
    """A second look that replayed the first current run would confirm its noise."""
    db = tmp_path / "replay.sqlite"
    _arm(monkeypatch, db, "baseline")
    djev.rank("q", CANDIDATES)
    _arm(monkeypatch, db, "current")
    djev.rank("q", ["x", "y"])
    _arm(monkeypatch, db, "confirm")
    djev.rank("q", ["x", "y"])                  # not the baseline's: drawn again
    djev.rank("q", CANDIDATES)                  # the baseline's: replayed
    assert server.calls == 3
    assert djev.replay_stats(db)["confirm"] == {"fresh": 1, "replayed_anchor": 1}


def test_an_arm_repeating_itself_replays_its_own_answer(monkeypatch, tmp_path, server):
    db = tmp_path / "replay.sqlite"
    _arm(monkeypatch, db, "baseline")
    a = djev.rank("q", CANDIDATES)
    b = djev.rank("q", CANDIDATES)
    assert _top(a) == _top(b) and server.calls == 1
    # The anchor's own rows are found first; for the anchor that is itself.
    assert djev.replay_stats(db)["baseline"] == {"fresh": 1, "replayed_anchor": 1}


def test_a_failure_is_counted_and_never_replayed(monkeypatch, tmp_path, server):
    db = tmp_path / "replay.sqlite"
    _arm(monkeypatch, db, "baseline")
    server.fail = True
    assert djev.rank("q", CANDIDATES) is None
    server.fail = False
    assert djev.rank("q", CANDIDATES) is not None
    assert server.calls == 2
    assert djev.replay_stats(db)["baseline"] == {"unreachable": 1, "fresh": 1}
    assert "unreachable" in djev.REPLAY_FAILURES


def test_a_5xx_is_a_replay_failure_and_a_4xx_is_not(monkeypatch, tmp_path):
    """A 500 is a hung upstream read, the instrument. A 422 is a schema this
    client built wrong, which is what a change under test can do."""
    db = tmp_path / "replay.sqlite"
    _arm(monkeypatch, db, "current")
    for code in (500, 422):
        def _open(req, timeout=None, code=code):
            raise urllib.error.HTTPError("u", code, "x", None, None)
        monkeypatch.setattr(djev.urllib.request, "urlopen", _open)
        assert djev.rank("q", CANDIDATES) is None
    assert djev.replay_stats(db)["current"] == {"http_5xx": 1, "http_4xx": 1}
    assert "http_5xx" in djev.REPLAY_FAILURES and "http_4xx" not in djev.REPLAY_FAILURES


def test_production_never_replays(tmp_path, server):
    """No replay variable, no file, and every request goes to djev."""
    djev.rank("q", CANDIDATES)
    djev.rank("q", CANDIDATES)
    assert server.calls == 2
    assert not list(tmp_path.iterdir())


def test_a_broken_replay_file_costs_the_replay_not_the_answer(monkeypatch, tmp_path, server):
    bad = tmp_path / "not-a-db"
    bad.write_text("this is not sqlite")
    _arm(monkeypatch, bad, "baseline")
    assert djev.rank("q", CANDIDATES) is not None
    assert server.calls == 1
    assert djev.replay_stats(bad) == {}


def test_the_regression_check_uses_this_file_s_names():
    """The check reads the same arm names and failure set the client writes."""
    from workers.sources import automod_regression as R
    assert R.REPLAY_FAILURES is djev.REPLAY_FAILURES
    env = R._replayed({"X": "1"}, tmp := "/tmp/x.sqlite", R.ARM_CURRENT)
    assert env == {"X": "1", djev.REPLAY_ENV: tmp, djev.REPLAY_ARM_ENV: "current",
                   djev.REPLAY_ANCHOR_ENV: "baseline"}
