"""`app/djev.py`: the three-valued contract, the normalizations, the two flags.

The client's whole job is to be boring in the two directions that matter —
never raise, and never round a measurement into a verdict. These pin both.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from app import djev


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    """Every test here is about the client, not about the slot flag."""
    monkeypatch.setattr(djev, "enabled", lambda: True)
    djev.reset_stats()
    yield
    djev.reset_stats()


def _response(answers: dict, questions: dict, **diag) -> dict:
    return {
        "model": "djev",
        "answers": answers,
        "usage": {"input_tokens": 123, "output_tokens": 9},
        "diagnostics": {
            "timing": {"total_ms": 41.0, "reads": 1},
            "chunks": [list(questions)],
            "questions": questions,
            **diag,
        },
    }


def _serve(monkeypatch, payload):
    """Stand in for the HTTP call. A string body is what the server sends."""
    class _Resp:
        def __init__(self, body):
            self._body = body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _open(req, timeout=None):
        return _Resp(json.dumps(payload).encode())

    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)


# ---------------------------------------------------------------------------
# Three-valued: None on ANY failure, never [] and never an exception
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("boom", [
    ConnectionRefusedError("no engine"),
    TimeoutError("hung"),
    urllib.error.URLError("dns"),
    ValueError("not json"),
])
def test_any_transport_failure_is_none(monkeypatch, boom):
    """`None`, matching `backlog_similar.semantic_candidates`. A caller must be
    able to tell "djev had no opinion" from "djev did not answer" — only the
    first is evidence — and every seam in the tree reads `None` as "carry on
    unchanged"."""
    def _open(req, timeout=None):
        raise boom
    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)
    assert djev.ask_sync("s", {"q": {"type": "noul"}}) is None


def test_http_error_is_none_not_an_exception(monkeypatch):
    """422 is a schema this client built wrong and 500 is what a hung upstream
    read looks like — there is no 504 on that path. Neither may escape."""
    def _open(req, timeout=None):
        raise urllib.error.HTTPError("u", 500, "boom", {}, None)
    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)
    assert djev.ask_sync("s", {"q": {"type": "noul"}}) is None


def test_service_down_returns_none_not_empty_list():
    """Belt and braces on the shape: `[]` is falsy and would read as "no
    candidates matched", which is the opposite of what happened."""
    out = djev.ask_sync("s", {})
    assert out is None and out != []


def test_disabled_slot_never_opens_a_socket(monkeypatch):
    monkeypatch.setattr(djev, "enabled", lambda: False)
    def _open(req, timeout=None):  # pragma: no cover - must not run
        raise AssertionError("opened a socket with the slot switched off")
    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)
    assert djev.ask_sync("s", {"q": {"type": "noul"}}) is None


def test_malformed_body_is_none(monkeypatch):
    _serve(monkeypatch, {"answers": "not a mapping"})
    assert djev.ask_sync("s", {"q": {"type": "noul"}}) is None


# ---------------------------------------------------------------------------
# Normalizing the server's three shapes
# ---------------------------------------------------------------------------

def test_noul_gets_a_confidence_and_probabilities(monkeypatch):
    """The server sends `{"type": "noul", "noul": p}` and NOTHING else. A
    caller reading `confidence` off a mixed answer set would get None for
    exactly the yes/no questions and, unguarded, treat it as zero."""
    _serve(monkeypatch, _response(
        {"urgent": {"type": "noul", "noul": 0.8}},
        {"urgent": {"label_mass": 0.99, "argmax_is_label": True}}))
    out = djev.ask_sync("s", {"urgent": {"type": "noul"}})
    a = out["urgent"]
    assert a.value == pytest.approx(0.8)
    assert a.confidence == pytest.approx(0.8)          # max(p, 1-p)
    assert a.probabilities == pytest.approx({"yes": 0.8, "no": 0.2})
    assert a.label == "yes"


def test_noul_confidence_is_modal_not_p_yes(monkeypatch):
    """P(yes)=0.1 is a CONFIDENT no, not an unconfident yes."""
    _serve(monkeypatch, _response(
        {"q": {"type": "noul", "noul": 0.1}},
        {"q": {"label_mass": 0.99, "argmax_is_label": True}}))
    a = djev.ask_sync("s", {"q": {"type": "noul"}})["q"]
    assert a.label == "no" and a.confidence == pytest.approx(0.9)


def test_score_keeps_expected_value_and_modal_confidence_apart(monkeypatch):
    """`score` is the 0-based EXPECTED level and `confidence` is the modal
    probability. They answer different questions, and collapsing them turns
    "how severe is this" into "how sure are you"."""
    _serve(monkeypatch, _response(
        {"sev": {"type": "score", "score": 2.4,
                 "legend": {"0": "low", "1": "mid", "2": "high"},
                 "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
                 "confidence": 0.5}},
        {"sev": {"label_mass": 0.95, "argmax_is_label": True}}))
    a = djev.ask_sync("s", {"sev": {"type": "score"}})["sev"]
    assert a.value == pytest.approx(2.4)
    assert a.confidence == pytest.approx(0.5)
    assert a.value != a.confidence
    assert a.label == "high"          # the modal LEVEL NAME, via the legend


def test_label_mass_is_lifted_out_of_diagnostics(monkeypatch):
    """It lives in `diagnostics.questions.<id>`, one nesting level away from
    the answers, so the natural read of the response misses the only honest
    confidence signal the server produces."""
    _serve(monkeypatch, _response(
        {"q": {"type": "noul", "noul": 0.99}},
        {"q": {"label_mass": 0.07, "argmax_is_label": False}}))
    a = djev.ask_sync("s", {"q": {"type": "noul"}})["q"]
    assert a.label_mass == pytest.approx(0.07)
    assert a.argmax_is_label is False


# ---------------------------------------------------------------------------
# The two trust flags
# ---------------------------------------------------------------------------

def test_low_trust_fires_under_the_floor_with_the_value_intact(monkeypatch):
    """A flag, never a refusal. The value is still there — the caller
    decides."""
    _serve(monkeypatch, _response(
        {"q": {"type": "noul", "noul": 0.97}},
        {"q": {"label_mass": 0.08, "argmax_is_label": False}}))
    out = djev.ask_sync("s", {"q": {"type": "noul"}}, floor=0.4)
    a = out["q"]
    assert a.low_trust is True
    assert a.value == pytest.approx(0.97)      # intact
    assert out.floor == 0.4


def test_no_floor_means_no_low_trust_verdict(monkeypatch):
    """`None` is the honest starting state for an uncalibrated schema. "No
    answer was low-trust" and "nothing has been calibrated yet" must not read
    the same, which is why `Answers.floor` is reported beside the flags."""
    _serve(monkeypatch, _response(
        {"q": {"type": "noul", "noul": 0.97}},
        {"q": {"label_mass": 0.01, "argmax_is_label": False}}))
    out = djev.ask_sync("s", {"q": {"type": "noul"}})
    assert out["q"].low_trust is False
    assert out.floor is None
    assert out["q"].label_mass == pytest.approx(0.01)   # still reported


def test_uninformative_catches_what_a_floor_cannot(monkeypatch):
    """The n=4 listwise run returned every score 0.0 with `label_mass` 0.987:
    the mass was legal and the answer was empty. A floor is structurally
    unable to see that shape, so the equality check is a second, separate
    flag."""
    answers = {f"c{i}": {"type": "score", "score": 0.0,
                         "legend": {"0": "a", "1": "b"},
                         "probabilities": {"0": 1.0, "1": 0.0},
                         "confidence": 1.0} for i in range(4)}
    diag = {f"c{i}": {"label_mass": 0.987, "argmax_is_label": True}
            for i in range(4)}
    _serve(monkeypatch, _response(answers, diag))
    out = djev.ask_sync("s", {f"c{i}": {"type": "score"} for i in range(4)})
    assert out.uninformative is True
    assert all(a.uninformative for a in out)
    assert all(not a.low_trust for a in out)   # the mass is fine; the answer is not


def test_a_mixed_ticket_schema_is_never_uninformative(monkeypatch):
    """The flag is about a RANKING. Two questions of different types that
    happen to agree are not a degenerate read."""
    _serve(monkeypatch, _response(
        {"a": {"type": "noul", "noul": 0.5},
         "b": {"type": "choice", "choice": "x",
               "probabilities": {"x": 0.5, "y": 0.5}, "confidence": 0.5}},
        {"a": {"label_mass": 0.9, "argmax_is_label": True},
         "b": {"label_mass": 0.9, "argmax_is_label": True}}))
    out = djev.ask_sync("s", {"a": {"type": "noul"}, "b": {"type": "choice"}})
    assert out.uninformative is False


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_rank_refuses_above_the_ceiling_rather_than_truncating():
    """A caller handing over 40 rows means to rank 40. Silently scoring the
    first 16 returns a confident ordering of a slice nobody chose."""
    with pytest.raises(ValueError, match="at most"):
        djev.rank("q", [f"c{i}" for i in range(djev.RANK_MAX_N + 1)])


def test_rank_default_is_below_the_ceiling():
    """12 and 16 are different numbers for a reason: listwise `label_mass` at
    n=16 measured 0.446, 0.807 and 0.965 across three corpora, so 16 is the
    edge of the safe window and not its middle."""
    assert djev.RANK_DEFAULT_N < djev.RANK_MAX_N <= djev.CANVAS_CHUNK_QUESTIONS


def test_rank_refuses_an_answer_split_across_canvas_chunks(monkeypatch):
    """Different chunks are different shared contexts. Upstream says plainly
    that a partitioned listwise score is not comparable across them, and the
    n=64 run ranked first a candidate that was the LONE member of a one-item
    chunk — scored against nothing. Sorting the union looks exactly like a
    ranking."""
    answers = {f"c{i}": {"type": "score", "score": float(i),
                         "legend": {"0": "a", "1": "b"},
                         "probabilities": {"0": 0.5, "1": 0.5},
                         "confidence": 0.5} for i in range(3)}
    diag = {f"c{i}": {"label_mass": 0.9, "argmax_is_label": True} for i in range(3)}
    payload = _response(answers, diag)
    payload["diagnostics"]["chunks"] = [["c0", "c1"], ["c2"]]
    _serve(monkeypatch, payload)
    assert djev.rank("q", ["a", "b", "c"]) is None


def test_rank_orders_best_first(monkeypatch):
    answers = {f"c{i}": {"type": "score", "score": s,
                         "legend": {"0": "a", "1": "b"},
                         "probabilities": {"0": 1 - s, "1": s},
                         "confidence": max(s, 1 - s)}
               for i, s in enumerate((0.2, 0.9, 0.5))}
    diag = {f"c{i}": {"label_mass": 0.9, "argmax_is_label": True} for i in range(3)}
    _serve(monkeypatch, _response(answers, diag))
    rows = djev.rank("q", ["a", "b", "c"])
    assert [r["index"] for r in rows] == [1, 2, 0]


def test_rank_of_nothing_is_an_empty_list_not_none():
    """Zero candidates is an answer ("nothing to rank"); `None` means the
    engine did not speak. The two must not collapse."""
    assert djev.rank("q", []) == []


# ---------------------------------------------------------------------------
# Config and status
# ---------------------------------------------------------------------------

def test_url_falls_back_to_the_default_when_config_is_unreadable(monkeypatch):
    import app.config
    monkeypatch.setattr(app.config, "CONFIG", None, raising=False)
    assert djev.structured_url().startswith("http://")


def test_stats_is_offline_and_counts_outcomes(monkeypatch):
    def _open(req, timeout=None):
        raise ConnectionRefusedError()
    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)
    djev.ask_sync("s", {"q": {"type": "noul"}}, seam="probe")
    st = djev.stats()
    assert st["seams"]["probe"]["unreachable"] == 1
