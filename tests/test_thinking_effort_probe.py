"""#2576 clause 4: the thinking-effort spelling probe, arm set and all.

Clause 4: post one fixed prompt to the primary under each spelling the box has
accepted — `reasoning_effort` at top level, `chat_template_kwargs.enable_thinking`
true and false, `reasoning_effort: "none"` — and record `reasoning_tokens`, content
length and `finish_reason` per spelling, writing an explicit `unreachable` row when
the engine refuses rather than omitting the setting.

Clause → node map:

* **Row per spelling, three figures each:**
  `test_clause_4_every_arm_records_reasoning_tokens_content_and_finish_reason`,
  `test_clause_4_the_arm_set_is_the_spellings_this_box_has_ever_accepted`.
* **One prompt, only the knob differs:**
  `test_clause_4_every_arm_proves_it_sent_the_same_prompt_by_prompt_sha`,
  `test_the_transport_always_sees_the_same_messages_and_ceiling`.
* **An omitted row is never allowed:**
  `test_a_down_engine_still_prints_every_arm_row_never_none`,
  `test_an_omitted_row_would_be_mistaken_for_a_free_setting`.
* **The process boundary itself** (script process → engine process, over real HTTP):
  `test_the_real_http_route_resolves_the_model_and_writes_the_witness`,
  `test_a_closed_port_is_a_row_not_an_absence`.

Why a stub transport and a real socket, both
--------------------------------------------
`probe()` takes its HTTP client as an argument, so the interesting failures —
connection refused, a 400 naming a field, a response that simply has no
`completion_tokens_details` — are constructed here rather than waited for. That
covers the semantics. It does NOT cover the wiring: the url the script builds from
config, the model id it asks the engine for, and the flags that write the witness the
report cites are only exercised by running `main()` against a socket, which the last
node does against a loopback server on an ephemeral port. Both halves are in this file
because the last review of this round said the seam was unverified, and a stub cannot
catch a wrong url.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from pathlib import Path

import httpx
import pytest

from scripts.thinking_effort_probe import (FIXED_PROMPT, SPELLINGS,
                                           chat_completions_url, main,
                                           markdown_rows, probe, render,
                                           resolve_model)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_WITNESS = REPO_ROOT / "eval" / "measurements" / "2576-probe-1500-2026-10-11.md"

# A stub whose arms cover every branch the row builder has: a graded level that
# costs tokens, a switch that costs none, a counter the engine never sent, a 400, and
# a refused connection. `content` is CHARACTERS as returned; the row records its
# length, which is not a token count and must not be read as one.
STUB_RESPONSES: dict[str, object] = {
    "default (no knob sent)": (200, {"choices": [{"message": {"content": ""},
                                                  "finish_reason": "length"}],
                                     "usage": {"completion_tokens": 120,
                                               "completion_tokens_details":
                                                   {"reasoning_tokens": 120}}}),
    "chat_template_kwargs.enable_thinking=true": (
        200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
              "usage": {"completion_tokens": 120,
                        "completion_tokens_details": {"reasoning_tokens": 120}}}),
    "chat_template_kwargs.enable_thinking=false": (
        200, {"choices": [{"message": {"content": "The train is overtaken at 11:05."},
                           "finish_reason": "stop"}],
              "usage": {"completion_tokens": 40}}),
    "reasoning_effort=xhigh": (200, {"choices": [{"message": {"content": ""},
                                                  "finish_reason": "length"}],
                                     "usage": {"completion_tokens": 120,
                                               "completion_tokens_details":
                                                   {"reasoning_tokens": 120}}}),
    "reasoning_effort=medium": (
        200, {"choices": [{"message": {"content": "Overtake at 11:05, working below."},
                           "finish_reason": "stop"}],
              "usage": {"completion_tokens": 90,
                        "completion_tokens_details": {"reasoning_tokens": 60}}}),
    "reasoning_effort=low": (
        200, {"choices": [{"message": {"content": "11:05."}, "finish_reason": "stop"}],
              "usage": {"completion_tokens": 30,
                        "completion_tokens_details": {"reasoning_tokens": 12}}}),
    # A real shape, not a hypothetical one: vLLM has shipped a response with the
    # details block absent, which is a MISSING counter and not a zero-cost arm.
    "reasoning_effort=high": (
        400, {"error": {"message": "Unexpected reasoning effort high. Supported types "
                                   "are xhigh (default), medium, and low."}}),
    "reasoning_effort=none": (200, {"choices": [{"message": {"content": "11:05"},
                                                 "finish_reason": "stop"}],
                                    "usage": {"completion_tokens": 25}}),
}


class StubResponse:
    def __init__(self, status: int, body: dict):
        self.status_code, self._body = status, body

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("stub", request=None, response=None)

    @property
    def text(self) -> str:
        return json.dumps(self._body)


def _stub_post(url: str, *, json: dict = None, headers=None, timeout=None,
               fail_for: frozenset = frozenset()):
    """Transport seam: dispatch on the knob in the body, refuse the `fail_for` arms."""
    effort = json.get("reasoning_effort")
    kwargs = json.get("chat_template_kwargs") or {}
    if effort is not None:
        name = f"reasoning_effort={effort}"
    elif "enable_thinking" in kwargs:
        name = f"chat_template_kwargs.enable_thinking={str(kwargs['enable_thinking']).lower()}"
    else:
        name = "default (no knob sent)"
    if name in fail_for:
        raise httpx.ConnectError("connection refused", request=None)
    status, body = STUB_RESPONSES[name]
    return StubResponse(status, body)


def _rows(**kw):
    # `probe` calls the transport as `post(url, json=…, headers=…, timeout=…)`, so the
    # stub takes the url positionally exactly as httpx.post does — a `**kwargs`-only
    # lambda here would TypeError inside probe and every arm would come back `error`.
    return probe(post=lambda *a, **k: _stub_post(*a, **k), url="http://stub",
                 model="M", headers={}, **kw)


def _by_spelling(rows):
    return {r["spelling"]: r for r in rows}


# ── clause 4: one row per spelling, three figures each ───────────────────────


def test_clause_4_every_arm_records_reasoning_tokens_content_and_finish_reason():
    """Each of the eight arms yields its own three figures.

    The three are what the clause asks for and each carries a different fact: the
    token bill, what the tokens bought, and whether the ceiling cut the arm off
    before it finished. `enable_thinking=false` records 0 because the stub's usage
    block has no reasoning counter at all and the row builder reads a MISSING counter
    as `None` — the arm whose knob turned thinking OFF must not be recorded as
    "measured: cost zero" unless the engine actually said so.
    """
    rows = _rows()
    got = _by_spelling(rows)
    assert len(rows) == len(SPELLINGS) == 8
    assert set(got) == {a["spelling"] for a in SPELLINGS}

    low = got["reasoning_effort=low"]
    assert low["status"] == "ok" and low["http_status"] == 200
    assert low["reasoning_tokens"] == 12
    assert low["content_len"] == len("11:05.")
    assert low["finish_reason"] == "stop"
    assert low["completion_tokens"] == 30

    off = got["chat_template_kwargs.enable_thinking=false"]
    assert off["reasoning_tokens"] is None, (
        "a missing counter became a zero, which is the difference between 'the "
        "engine did not report' and 'thinking costs nothing'")
    assert off["content_len"] == len("The train is overtaken at 11:05.")

    med = got["reasoning_effort=medium"]
    assert med["reasoning_tokens"] == 60 < med["completion_tokens"] == 90


def test_clause_4_the_arm_set_is_the_spellings_this_box_has_ever_accepted():
    """The arm list is the clause's spellings, and the level names are the engine's.

    `high` is in the set although the primary refuses it: the refusal IS the
    measurement (it is how the supported set — xhigh default, medium, low — became
    known), and dropping refused spellings from the arm list is how a ladder looks
    three rungs deep because someone only asked about three.
    """
    names = [a["spelling"] for a in SPELLINGS]
    assert names == ["default (no knob sent)",
                     "chat_template_kwargs.enable_thinking=true",
                     "chat_template_kwargs.enable_thinking=false",
                     "reasoning_effort=xhigh", "reasoning_effort=medium",
                     "reasoning_effort=low", "reasoning_effort=high",
                     "reasoning_effort=none"]
    assert {"reasoning_effort": "none"} == SPELLINGS[-1]["extra"], (
        "the clause names `reasoning_effort: 'none'` by name; it is a spelling, not a "
        "stand-in for the enable_thinking switch")
    assert SPELLINGS[0]["extra"] == {}, (
        "arm zero is the control: what app/harness/client.py sends today, no knob")

    refused = _by_spelling(_rows())["reasoning_effort=high"]
    assert refused["status"] == "error" and refused["http_status"] == 400
    assert "Supported types are xhigh" in refused["error"], (
        "the engine's own words are the evidence for the level set; a truncated error "
        "field would lose them")


def test_clause_4_every_arm_proves_it_sent_the_same_prompt_by_prompt_sha():
    """One prompt across the table, provable from the table.

    Every row carries `prompt_sha` = sha256(FIXED_PROMPT)[:12]. If an arm's row
    disagrees, that arm is a different experiment — the source talk's "change any part
    and it's a different experiment" rule, made checkable by a reader holding only the
    committed witness, without the script's source.
    """
    want = hashlib.sha256(FIXED_PROMPT.encode()).hexdigest()[:12]
    for row in _rows():
        assert row["prompt_sha"] == want, row["spelling"]
    # And the committed measurement agrees with the prompt in this tree's script,
    # so the note's `prompt_sha` sentence still names what was actually sent.
    payload = json.loads(PROBE_WITNESS.read_text().split("```json")[1].split("```")[0])
    assert payload["prompt_sha"] == want == payload["rows"][0]["prompt_sha"]
    assert payload["max_tokens"] == 1500, (
        "the ceiling is part of the experiment and the witness names it: the report "
        "says 1500 because these bytes say 1500")


def test_the_probe_crosses_a_real_http_boundary_and_writes_its_witness(tmp_path):
    """The transport seam is `httpx.post`; this crosses it with a socket, not a fake.

    An OpenAI-compatible engine on 127.0.0.1:0 (the idiom
    `tests/test_vllm_metrics.py` uses) serving `/v1/models` and
    `/v1/chat/completions`, driven through `main()` so URL assembly, model resolution
    off `/v1/models`, all eight arms and the witness write run against a live server.
    Those are the parts a stub lambda cannot check: that the body the probe sends is a
    body an OpenAI-compatible server actually parses, that the model named by the
    witness is one the engine reported rather than one that was typed, and that
    `--witness-out` produces the file the report cites.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    bodies: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                    # silence per-request logging
            pass

        def _send(self, code: int, obj: dict) -> None:
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send(200, {"object": "list",
                                 "data": [{"id": "Served-Model-v1"}]})
            else:
                self._send(404, {"error": {"message": "no such path"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", "0")) or 0))
            bodies.append(body)
            effort = str(body.get("reasoning_effort"))
            if effort == "high":                      # like the template: refuse it
                self._send(400, {"error": {"message": "Unexpected reasoning effort"}})
                return
            knob = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
            off = knob is False or effort == "none"
            reasoning = 0 if off else 137
            self._send(200, {"choices": [{"message": {"content": "answer" * 30},
                                          "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 40,
                                       "completion_tokens": reasoning + 20,
                                       "reasoning_tokens": reasoning}})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    out = tmp_path / "witness.md"
    try:
        assert main(["--base-url", f"http://127.0.0.1:{port}", "--max-tokens", "1500",
                     "--witness-out", str(out)]) == 0
    finally:
        srv.shutdown()

    payload = json.loads(out.read_text().split("```json")[1].split("```")[0])
    assert payload["model"] == "Served-Model-v1", (
        "the witness names a model the engine never reported")
    # The url in the witness is the script's own builder applied to the base it was
    # given, not a path typed into this test: if the builder ever points somewhere
    # else, the request the engine sees changes, and a literal here would let the
    # witness and the code disagree in silence.
    base = f"http://127.0.0.1:{port}"
    assert payload["url"] == chat_completions_url(base) == f"{base}/v1/chat/completions"
    assert len(payload["rows"]) == len(SPELLINGS)
    assert {r["status"] for r in payload["rows"]} == {"ok", "error"}, (
        "over a live server an arm should be ok or an engine 400, nothing else")
    by = {r["spelling"]: r for r in payload["rows"]}
    assert by["reasoning_effort=high"]["http_status"] == 400
    assert by["chat_template_kwargs.enable_thinking=false"]["reasoning_tokens"] == 0
    assert by["default (no knob sent)"]["reasoning_tokens"] == 137
    assert len(bodies) == len(SPELLINGS), "an arm never reached the server"
    assert {b["messages"][0]["content"] for b in bodies} == {FIXED_PROMPT}, (
        "a body carried a different prompt than its siblings")
    assert {b["max_tokens"] for b in bodies} == {1500}
    assert {b["model"] for b in bodies} == {"Served-Model-v1"}


def test_the_transport_always_sees_the_same_messages_and_ceiling():
    """`prompt_sha` agrees, but the request body is the other half of "same prompt".

    The sha covers the prompt string, not the assembled payload. Asserted on the
    bodies handed to the transport: `messages`, `max_tokens`, `temperature` and
    `model` are identical across the arms, and the ONLY keys that differ are the ones
    in the arm's own `extra`.
    """
    seen: list[dict] = []
    probe(post=lambda url, json=None, headers=None, timeout=None:
          (seen.append(json), StubResponse(200, {"choices": [], "usage": {}}))[1],
          url="http://stub", model="M", headers={}, max_tokens=77)
    assert len(seen) == 8
    shared = {"messages", "max_tokens", "temperature", "model"}
    first = {k: v for k, v in seen[0].items() if k in shared}
    for body in seen:
        assert {k: v for k, v in body.items() if k in shared} == first
    knob_keys = {frozenset(body) - shared for body in seen}
    assert frozenset() in knob_keys, "one arm must send no knob at all"
    assert all(len(keys) <= 2 for keys in knob_keys), (
        f"an arm carried more than its knob: {knob_keys}")


# ── no arm may be omitted ───────────────────────────────────────────────────


def test_a_down_engine_still_prints_every_arm_row_never_none():
    """A refused connection yields EIGHT rows, not none.

    The count is the point, so it is asserted as `len(rows) == len(SPELLINGS)` rather
    than as a status string: a probe that swallowed the exception and returned `[]`
    would print a table with no rows, and an empty table reads as "nothing to report"
    when it means "nothing was measured". Every arm also says `unreachable` and keeps
    its knob in `extra`, so the setting is on the record even though no number is.
    """
    def refusing(url, *, json=None, headers=None, timeout=None):
        raise httpx.ConnectError("connection refused", request=None)

    rows = probe(post=refusing, url="http://127.0.0.1:9/v1/chat/completions",
                 model="M", headers={})
    assert len(rows) == len(SPELLINGS) == 8
    assert {r["status"] for r in rows} == {"unreachable"}
    for r in rows:
        assert r["reasoning_tokens"] is None
        assert r["error"], f"{r['spelling']}: unreachable with no reason recorded"
    assert "connection refused" in render(rows, url="u", model="M")


def test_mixed_failures_keep_their_own_rows_and_their_own_status():
    """`unreachable` and `error` are different facts and must not collapse.

    Two arms refused at the socket, one answered-and-refused at the HTTP layer, the
    rest fine: eight rows, three statuses. Collapsing the two failure kinds would
    lose the difference between "cannot measure this box" and "measured: the engine
    does not implement the argument", which is the difference between waiting for the
    engine and changing the question.
    """
    def flaky(url, *, json=None, headers=None, timeout=None):
        if json.get("reasoning_effort") in ("medium", "low"):
            raise httpx.ConnectTimeout("timed out", request=None)
        return _stub_post(url, json=json)

    rows = probe(post=flaky, url="http://stub", model="M", headers={})
    assert len(rows) == 8
    by = _by_spelling(rows)
    assert by["reasoning_effort=medium"]["status"] == "unreachable"
    assert by["reasoning_effort=low"]["status"] == "unreachable"
    assert by["reasoning_effort=high"]["status"] == "error"
    assert by["reasoning_effort=xhigh"]["status"] == "ok"
    assert by["reasoning_effort=medium"]["reasoning_tokens"] is None


def test_an_omitted_row_would_be_mistaken_for_a_free_setting():
    """The markdown table renders a missing figure as `–`, never as 0.

    Asserted per arm LINE, cell by cell, on rows built to be the misleading shapes:
    an unreachable arm and a 2xx with no reasoning counter. A whole-table substring
    census ("no ` 0` anywhere in the body") passes on a truncated or empty table, so
    it cannot catch what it claims to; this finds the arm's own line, splits its
    cells, and fails on the cell.
    """
    rows = _rows() + probe(
        post=lambda *a, **k: _stub_post(*a, **k, fail_for={"reasoning_effort=none"}),
        url="http://stub", model="M", headers={}, arms=[SPELLINGS[-1]])
    text = markdown_rows(rows)
    lines = {l.split("|")[1].strip().strip("`"): l
             for l in text.splitlines() if l.startswith("| ") and "---" not in l}
    lines.pop("spelling", None)
    assert len(lines) == len(SPELLINGS), "the table lost an arm"
    for name in ("reasoning_effort=high", "reasoning_effort=none"):
        cells = [c.strip() for c in lines[name].split("|")[1:-1]]
        assert cells[2] == "–", (
            f"{name}: a refused arm rendered {cells[2]!r} in the reasoning column, "
            "which is the cheapest-looking cell in the table")
        assert cells[3] == "–" and cells[4] == "–", (
            f"{name}: a refused arm rendered a content or completion figure")
    ok_line = [c.strip() for c in lines["reasoning_effort=low"].split("|")[1:-1]]
    assert ok_line[2] == "12", "a measured arm lost its number next to the refused ones"


def test_the_aligned_render_dashes_a_missing_counter_rather_than_padding_a_zero():
    """Same rule in the human-readable table, cell by cell.

    `str(0).rjust(8)` and `"-".rjust(8)` are both 8 characters, so the column widths
    say nothing; the assertion reads the arm's own field out of the aligned line.
    """
    rows = _rows()
    text = render(rows, url="http://stub/v1/chat/completions", model="M")
    lines = {l.split()[0]: l for l in text.splitlines() if l.startswith("reasoning_effort=")}
    assert set(lines) == {"reasoning_effort=xhigh", "reasoning_effort=medium",
                          "reasoning_effort=low", "reasoning_effort=high",
                          "reasoning_effort=none"}

    # Fixed-width cells read at the widths the renderer pads to: spelling 42, status
    # 13, http 5, then reason 8 / content 8 / completion 7 / finish 9. Cell by cell,
    # because `str(0).rjust(8)` and `"-".rjust(8)` are both 8 characters and a
    # whole-line substring search cannot tell a dashed column from a zeroed one.
    def cells(line: str) -> list[str]:
        return [line[42:55], line[55:60], line[60:68], line[68:76], line[76:83],
                line[83:92]]

    assert [c.strip() for c in cells(lines["reasoning_effort=low"])] == [
        "ok", "200", "12", "6", "30", "stop"]
    assert [c.strip() for c in cells(lines["reasoning_effort=high"])] == [
        "error", "400", "-", "-", "-", "-"], (
        "a refused arm is showing a figure in one of the token columns")
    assert "NEVER zero" in text, "the legend that defines the dash went missing"


# ── the real socket: url building, model resolution, flags ──────────────────


class _Engine(http.server.BaseHTTPRequestHandler):
    """A loopback stand-in for vLLM: /v1/models, and a chat route that refuses `high`."""

    seen_bodies: list[dict] = []

    def do_GET(self):                                          # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"object": "list",
                               "data": [{"id": "Stub/Qwen-1", "object": "model"}]}).encode()
            self.send_response(200)
        else:
            body = b"{}"
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                                          # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen_bodies.append(body)
        assert self.path == "/v1/chat/completions", self.path
        if body.get("reasoning_effort") == "high":
            self.send_response(400)
            payload = {"error": {"message": "Unexpected reasoning effort high."}}
        else:
            self.send_response(200)
            payload = {"choices": [{"message": {"content": "11:05"},
                                    "finish_reason": "stop"}],
                       "usage": {"completion_tokens": 9,
                                 "completion_tokens_details": {"reasoning_tokens": 4}}}
        raw = json.dumps(payload).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def engine():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Engine.seen_bodies = []
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_real_http_route_resolves_the_model_and_writes_the_witness(engine, tmp_path,
                                                                       capsys):
    """`main()` end to end over a real socket: the seam the stub cannot check.

    Everything above this node injects a transport, which means none of it can catch
    the three ways the real route fails: a base url whose trailing slash makes every
    arm 404 (eight `error` rows that look exactly like an engine refusing every
    spelling), a model id from config the engine never loaded, and a `--witness-out`
    that writes the wrong shape. Passing a slash-terminated url is deliberate: it is
    the spelling a human types.
    """
    witness = tmp_path / "probe.md"
    rc = main([f"--base-url={engine}/", "--max-tokens", "1500", "--markdown",
               f"--witness-out={witness}"])
    assert rc == 0
    table = capsys.readouterr().out
    assert table.count("\n|") == 9, "eight arms plus the header row"
    assert "`reasoning_effort=high` | error 400" in table

    text = witness.read_text()
    payload = json.loads(text.split("```json")[1].split("```")[0])
    assert payload["model"] == "Stub/Qwen-1", (
        "the model asked of the engine's /v1/models, not the one in config")
    assert payload["url"] == f"{engine}/v1/chat/completions", (
        "the trailing slash in --base-url must not become //v1/ or a 404 on every arm")
    assert payload["max_tokens"] == 1500 and payload["prompt_sha"]
    assert len(payload["rows"]) == 8
    assert sum(1 for r in payload["rows"] if r["status"] == "ok") == 7
    assert _Engine.seen_bodies and all(
        b["messages"][0]["content"] == FIXED_PROMPT for b in _Engine.seen_bodies), (
        "eight arms of one prompt is the whole premise; the server saw otherwise")


def test_a_closed_port_is_a_row_not_an_absence(tmp_path, capsys):
    """`main()` against nothing at all: eight `unreachable` rows and an exit 0.

    Exit code matters here. A nonzero exit would look like the right behaviour and
    still be wrong — the probe's job is to publish that the box could not be reached,
    and a traceback prints nothing to pin. The report is allowed to say "the probe was
    not runnable", and owed item 4 asks whether it was runnable at all: this node is
    what that sentence would be citing.
    """
    witness = tmp_path / "down.md"
    rc = main(["--base-url=http://127.0.0.1:1", "--markdown",
               f"--witness-out={witness}"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("unreachable") == 8
    payload = json.loads(witness.read_text().split("```json")[1].split("```")[0])
    assert [r["status"] for r in payload["rows"]] == ["unreachable"] * 8
    assert all(r["reasoning_tokens"] is None for r in payload["rows"])


def test_resolve_model_degrades_to_an_empty_id_rather_than_to_no_probe(capsys):
    """No /v1/models answer still runs the eight arms.

    vLLM serves one model and accepts an empty id, so an unanswerable
    `/v1/models` is a naming gap, not a reason to print nothing — and printing
    nothing is the failure this whole file is written against.
    """
    got = resolve_model("http://127.0.0.1:1", get=httpx.get, headers={})
    assert got == ""
    assert "/v1/models" in capsys.readouterr().err
    assert resolve_model("http://stub",
                         get=lambda *a, **k: StubResponse(200, {"data": []}),
                         headers={}) == ""
