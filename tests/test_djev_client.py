"""`app/djev.py`: the three-valued contract, the normalizations, the two flags.

The client's whole job is to be boring in the two directions that matter —
never raise, and never round a measurement into a verdict. These pin both.
"""

from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------------------
# #3376 — the `reject` delta: `None` stays the return, and a 4xx explains itself
#
# `None` for every failure is the contract above, and this does not change it.
# What was missing is that a 422 and a dead socket returned the same thing, so
# the tool-facing sentence for both claimed an outage — 9 of the 15
# `djev_decide` calls ever made, measured 2026-10-26, were the engine refusing a
# prose `noul` criteria reported as "engine unreachable". `reject` is an optional
# out-parameter: the engine's own sentence for the one failure it explains, and
# nothing for the rest. Callers that never pass the dict are unaffected, which is
# the reason every node above still passes.
# ---------------------------------------------------------------------------

#: Verbatim from `POST /v1/systemone` on 2026-10-26 for
#: `{"type": "noul", "criteria": "needs a reply now"}`.
_422_BODY = (b'{"error": {"message": "question \'a\': noul criteria must be an '
             b'object with true and false", "type": "validation_error"}}')


class _Body:
    """A response body that reads once, like the socket behind a real
    `HTTPError` — `exc.read()` is the only access the client gets.

    `close()` is not decoration: `HTTPError` adopts the fp it is handed and the
    stdlib later finalises it, so a stub without it surfaces as an unraisable
    `AttributeError` in whatever test happens to trigger the GC — a warning pinned
    to nothing, which is how a suite starts collecting failures it did not cause.
    """

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        data, self._data = self._data, b""
        return data

    def close(self):
        self._data = b""


def _raise_http(monkeypatch, code: int, body: bytes) -> None:
    def _open(req, timeout=None):
        raise urllib.error.HTTPError("u", code, "nope", {}, _Body(body))
    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)


def test_a_4xx_fills_reject_with_the_engines_own_sentence(monkeypatch):
    """The cause has to leave the process, because it names the field the caller
    got wrong. Note the return value is still `None`: the caller's three-valued
    reading is untouched, and the explanation travels beside it."""
    _raise_http(monkeypatch, 422, _422_BODY)
    reject: dict = {}
    assert djev.ask_sync("s", {"a": {"type": "noul",
                                     "criteria": "needs a reply now"}},
                         reject=reject) is None
    assert reject["status"] == 422
    assert reject["message"] == ("question 'a': noul criteria must be an object "
                                 "with true and false")


def test_a_4xx_that_is_not_json_reports_its_status_and_quotes_nothing(monkeypatch):
    """A proxy's page or a login wall is not the engine's sentence, so it is not
    quoted into a prompt: the status alone is the fact that survives, and it is
    the one the caller needs — it says the engine answered and refused."""
    _raise_http(monkeypatch, 400, b"<html>the front proxy said no</html>")
    reject: dict = {}
    assert djev.ask_sync("s", {"a": {"type": "noul"}}, reject=reject) is None
    assert reject == {"status": 400}, reject


def test_a_4xx_json_without_the_message_field_keeps_its_own_head(monkeypatch):
    """`error.message` is what this engine sends; `detail` is what a FastAPI
    validator in front of it sends. Both are the engine's words, so either is safe
    to quote, and the fallback is why a shape change upstream does not silently
    turn a refusal back into an unexplained `None`."""
    _raise_http(monkeypatch, 422, b'{"detail":"canvas too wide"}')
    reject: dict = {}
    assert djev.ask_sync("s", {"a": {"type": "noul"}}, reject=reject) is None
    assert reject["status"] == 422
    assert "canvas too wide" in reject["message"], reject


def test_a_5xx_leaves_reject_empty_so_the_caller_says_unreachable(monkeypatch):
    """The distinction the whole change rests on, at the layer that makes it. The
    body here is deliberately the *persuasive* 422 sentence, because that is the
    shape of the risk: a vLLM upstream that dies mid-generation comes back 500
    with text in it, and quoting that text would tell the caller its arguments
    were wrong when nothing of the kind was established. A 500 on this path is
    also what a hung upstream read looks like — there is no 504 — so the status
    itself is not evidence of a refusal."""
    _raise_http(monkeypatch, 500, _422_BODY)
    reject: dict = {}
    assert djev.ask_sync("s", {"a": {"type": "noul"}}, reject=reject) is None
    assert reject == {}, reject


@pytest.mark.parametrize("boom", [
    urllib.error.URLError("dns"),
    ConnectionRefusedError("no engine"),
    TimeoutError("hung"),
])
def test_a_transport_failure_leaves_reject_empty(monkeypatch, boom):
    """Same reason, the more common cause: nothing answered, so there is nothing
    to quote and `reject` stays empty for the caller to read as an outage."""
    def _open(req, timeout=None):
        raise boom
    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)
    reject: dict = {}
    assert djev.ask_sync("s", {"a": {"type": "noul"}}, reject=reject) is None
    assert reject == {}


def test_a_malformed_200_body_leaves_reject_empty(monkeypatch):
    """200 with an unreadable body is a malformed answer, not a refusal: the
    engine never said the arguments were wrong, so the caller must not be told it
    can fix them."""
    def _open(req, timeout=None):
        class _R:
            def read(self):
                return b"<html>not json</html>"
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return _R()
    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)
    reject: dict = {}
    assert djev.ask_sync("s", {"a": {"type": "noul"}}, reject=reject) is None
    assert reject == {}


def test_a_caller_that_passes_no_reject_dict_is_unaffected(monkeypatch):
    """The back-compat half: four seam sites and the eval arms call this with no
    new argument, and the fold they depend on must behave exactly as before."""
    _raise_http(monkeypatch, 422, _422_BODY)
    assert djev.ask_sync("s", {"a": {"type": "noul"}}) is None


def test_a_4xx_message_over_the_cap_is_marked_not_quoted(monkeypatch):
    """The message is bound before it reaches a prompt: a tool reply is
    re-hydrated into the primary's context, and an engine error body is text
    someone else controls. Long becomes a length — `[:300]` alone would cut
    mid-sentence and read as if that were all the engine said. The body is
    parseable JSON on purpose: a non-JSON page is dropped whole by
    `test_a_4xx_that_is_not_json_reports_its_status_and_quotes_nothing`, and this
    node is about the cap, so nothing here asserts unless the message survives
    parsing to be capped at all."""
    _raise_http(monkeypatch, 422,
                json.dumps({"error": {"message": "y" * 5000}}).encode())
    reject: dict = {}
    assert djev.ask_sync("s", {"a": {"type": "noul"}}, reject=reject) is None
    assert reject["status"] == 422
    msg = reject["message"]
    assert "y" * djev.REJECT_MESSAGE_MAX_CHARS in msg, len(msg)
    assert "y" * (djev.REJECT_MESSAGE_MAX_CHARS + 1) not in msg, len(msg)
    assert "not quoted" in msg, msg
    assert len(msg) < djev.REJECT_MESSAGE_MAX_CHARS + 40, msg


def test_the_reject_cap_is_a_number_and_not_an_open_quote():
    """The bound exists so that a validation dump cannot ride a tool reply into
    the primary's context, where it is re-hydrated every turn it is repeated. A
    node asserting the exact figure would be decorum and would go red on any
    re-tune; the property worth pinning is that the field is capped at all — an
    unbounded pass-through is the shape that does the damage."""
    cap = djev.REJECT_MESSAGE_MAX_CHARS
    assert isinstance(cap, int) and 0 < cap <= 1000, cap


def _serve_httpx(monkeypatch, status: int, content: bytes, payload: dict) -> None:
    """Stand in for `httpx.AsyncClient`, the transport `ask` uses and `ask_sync`
    does not.

    The tool handler reaches the engine through `ask`, so a file that mocks only
    `urllib.request.urlopen` proves the sync twin and leaves the branch that
    actually spent the 9 calls unexercised. `json()` returns a real, parsable
    answer on purpose: were the client to parse a non-200 body, `ask` would return
    `Answers` and the `is None` assertion below would fail, instead of the node
    passing on an exception the client swallows.
    """
    import httpx

    class _Resp:
        status_code = status          # the class body reads the caller's scope

        def __init__(self):
            self.content = content

        def json(self):
            return payload

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


@pytest.mark.parametrize("status,should_quote", [(422, True), (500, False)])
def test_the_async_twin_quotes_a_4xx_and_never_a_5xx(monkeypatch, status,
                                                     should_quote):
    """The tool's own path, across the transport the tool actually crosses.

    `djev_decide` awaits `ask`, so the 4xx/5xx line clause 4 draws is really made
    here, not in the urllib twin. Both arms are served the same persuasive body —
    the real 422 sentence — because the failure worth guarding is a dead upstream
    that answers 500 with text in it: quoting that would tell the caller its
    arguments were wrong when nothing of the kind was established.
    """
    parsable = _response({"a": {"type": "noul", "noul": 0.9}},
                         {"a": {"label_mass": 0.99, "argmax_is_label": True}})
    _serve_httpx(monkeypatch, status, _422_BODY, parsable)
    reject: dict = {}
    assert asyncio.run(djev.ask("s", {"a": {"type": "noul"}},
                                reject=reject)) is None
    if should_quote:
        assert reject["status"] == 422, reject
        assert "noul criteria" in reject["message"], reject
    else:
        assert reject == {}, reject


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


def test_the_row_ceiling_is_derived_from_one_declared_canvas_width(monkeypatch):
    """The ceiling used to be a hand-written 32 with nothing behind it, and the
    recall's pool was a SECOND hand-written 32 in another module — two numbers
    that agreed only for as long as someone kept both in their head. Now one
    declared width does the arithmetic: a rank-shaped question costs 4 rows of
    canvas, measured at both widths the engine has been served at (33 questions
    fitted in one chunk at 128, and the first split at 256 was at 65 —
    `architecture/djev.md` §3.3, and #1345's widening measurement). So 128 // 4
    = 32 is the number §3.3 measured there, one row under its tested capacity,
    and 256 // 4 = 64 is one row under the 65 measured at 256.
    """
    assert djev.SERVED_CANVAS_ROWS == 128          # the conf's `CANVAS="128"`
    assert djev.CANVAS_ROWS_PER_RANK_QUESTION == 4
    assert djev.CANVAS_CHUNK_QUESTIONS == djev.canvas_rows() == 32
    monkeypatch.setattr(djev, "SERVED_CANVAS_ROWS", 64)
    assert djev.canvas_rows() == 16, "the width is read, not remembered"


def test_rank_refuses_an_answer_split_across_canvas_chunks(monkeypatch):
    """Different chunks are different shared contexts. Upstream says plainly
    that a partitioned listwise score is not comparable across them, and the
    n=64 run ranked first a candidate that was the LONE member of a one-item
    chunk — scored against nothing. Sorting the union looks exactly like a
    ranking.

    The refusal is `CANVAS_SPLIT` and not `None`. This node used to assert
    `is None` — byte-for-byte the shape of the transport-failure tests above —
    and that shared value is precisely how the recall came to book a split as an
    engine outage (#2112). The witness stays falsy, so a caller that only asks
    `if not rows:` carries on with the fallback exactly as it did.
    """
    answers = {f"c{i}": {"type": "score", "score": float(i),
                         "legend": {"0": "a", "1": "b"},
                         "probabilities": {"0": 0.5, "1": 0.5},
                         "confidence": 0.5} for i in range(3)}
    diag = {f"c{i}": {"label_mass": 0.9, "argmax_is_label": True} for i in range(3)}
    payload = _response(answers, diag)
    payload["diagnostics"]["chunks"] = [["c0", "c1"], ["c2"]]
    _serve(monkeypatch, payload)
    rows = djev.rank("q", ["a", "b", "c"])
    assert rows is djev.CANVAS_SPLIT
    assert not rows, "a split carries no ordering, exactly like `None`"


def test_an_engine_that_never_answered_is_none_and_not_a_split(monkeypatch):
    """The other half of the distinction #2112 asks for: a transport failure
    reaches `rank()` as the plain `None` it always was, so neither cause borrows
    the other's story — an outage is not a split, and a split is not an outage."""
    def _open(req, timeout=None):
        raise ConnectionRefusedError("no engine")
    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)
    assert djev.rank("q", ["a", "b"]) is None


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
