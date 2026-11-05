"""Tests for the job-path KV pre-warm (#3771).

Five clauses, one test node each, and each node is the thing the item's acceptance
says is checkable:

  1. `test_warm_payload_is_the_real_jobs_prompt_head` — the warm's payload is what
     the job builder produced, byte for byte (same sha256), not a typed copy.
  2. `test_warm_is_one_request_with_max_tokens_1_and_no_item_content` — one call to
     the engine, `max_tokens=1`, and nothing but the head in it.
  3. `test_admission_refuses_for_exactly_the_four_named_reasons` — the pure function
     and all four refusals.
  4. `test_repeated_triggers_warm_once_and_a_changed_head_warms_again` — the trigger
     storm costs one warm; a moved head costs a second.
  5. `test_every_attempt_appends_one_parseable_row` — the ledger, from which warm
     count, warm tokens and hits-per-warm are derived.

Plus the two seams the change crosses but no in-process test would see: the
loopback POST another process fires (`test_the_trigger_route_...`), and the
`vault_write` hook that fires it (`test_a_prompt_surface_write_...`).

Nothing here touches an engine or the vault: the engine is a fake client that
records what it was sent, the prompt builder is a sentinel, and the ledger is a
tmp_path file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest


SENTINEL = "WORKER PROMPT SENTINEL\nposition-0 head, ~20k tokens in production"


class FakeEngine:
    """The primary, as a recorder. Replaces `app.harness.run_query`.

    Answers with the usage shape the real loop emits on its `assistant_message`
    event (`app/run_recorder.py:257` reads exactly these keys), because the warm's
    own token count is one of the numbers the item is priced on.
    """

    def __init__(self, *, raise_on_call: bool = False):
        self.calls: list[dict] = []
        self.raise_on_call = raise_on_call

    async def run_query(self, messages, options, *, turn_id=None):
        self.calls.append({"messages": list(messages), "options": options})
        if self.raise_on_call:
            raise ConnectionError("engine refused the warm")
        yield {
            "type": "assistant_message",
            "iteration": 1,
            "text": "k",
            "usage": {"input_tokens": 21_940, "output_tokens": 1,
                      "cache_read": 0, "cache_create": 21_940},
        }


@pytest.fixture
def wp(tmp_path, monkeypatch):
    """The module with its three outside dependencies replaced.

    The prompt builder becomes a sentinel so a test can tell "the builder ran" from
    "a string that happens to look like its output"; the engine becomes a recorder;
    the ledger lands in tmp_path. The engine's pressure gauge is pinned to "no
    sample yet" — which `app.engine_pressure` itself reads as unpressured — so every
    test but the busy one is about firing, and `engine_busy` stays the real function
    reading that pinned gauge.
    """
    from app import worker_prewarm

    worker_prewarm.reset_state()
    _patch_head(monkeypatch, SENTINEL)
    monkeypatch.setattr(worker_prewarm, "log_path",
                        lambda: tmp_path / "worker-prewarm.jsonl")
    monkeypatch.setattr("app.engine_pressure.latest", lambda *a, **kw: None)
    yield worker_prewarm
    worker_prewarm.reset_state()


@pytest.fixture
def engine(wp, monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr("app.harness.run_query", fake.run_query)
    return fake


def _patch_head(monkeypatch, text: str) -> None:
    """Replace the prompt builder at *every* name that resolves it.

    Two names, because one is not enough: `app.prompt_builder` holds the
    definition, and `app.routers.turn_options` bound it by value at import
    (`from app.prompt_builder import build_system_prompt`,
    `app/routers/turn_options.py:66`). Patching only the definition site leaves the
    backend's builder — which is what `warm_payload` now calls — rendering the real
    prompt, so a comparison against a patched worker-path builder would be a
    sentinel matched against SOUL.md, and every sha assertion in this file would be
    about nothing. The same trap `_refusal`'s docstring records for the notify
    check: patch the name the caller actually reaches.
    """
    import app.prompt_builder as pb
    import app.routers.turn_options as turn_options

    monkeypatch.setattr(pb, "build_system_prompt", lambda **kw: text)
    monkeypatch.setattr(turn_options, "build_system_prompt", lambda **kw: text)


class _FakeBackend:
    """The backend, as the class `httpx.AsyncClient`: a recorder plus one canned turn.

    The class itself is replaced, not an attribute of `workers.sources._common`:
    `run_prompt_in_session` does `import httpx` inside its own body
    (`workers/sources/_common.py:1220`), so a module attribute on the worker module
    is a name that function never looks up. `.stream` is the turn — the POST is
    `client.stream("POST", f"{backend}/api/message/stream", json=payload)`
    (`workers/sources/_common.py:1325`) — and `.post` is the turn-cancel call the
    wall-clock branch makes. Neither reaches a socket here.
    """

    def __init__(self, posted: list):
        self.posted = posted

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None, **kwargs):
        self.posted.append(json)
        return _CannedTurn()

    async def post(self, url, **kwargs):
        return _CannedTurn()


class _CannedTurn:
    """One accepted turn that ends after a single `usage` event.

    `_SSEParser` carries an `event:` line across to the data line that follows it
    (`workers/sources/_common.py:1014-1043`), so the blank lines below are part of
    the fixture and not decoration.
    """

    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aread(self) -> bytes:
        return b"{}"

    async def aclose(self) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        yield "event: usage"
        yield '{"input_tokens": 21940, "output_tokens": 7, "cached_tokens": 0}'
        yield ""
        yield "event: done"
        yield "{}"
        yield ""

    async def aiter_raw(self):
        if False:  # pragma: no cover — the line reader is the only path used here
            yield b""


def _job_post(monkeypatch, *, extra_disallowed=None, max_turns: int = 60,
              prompt: str = "Item 4242 needs triage."):
    """Dispatch a real autotriage turn; return the body the worker POSTed.

    The dispatch is autotriage's own — `run_prompt_in_session`, which is what
    `workers/sources/autotriage.py:1163` calls for a triage turn — with its
    `extra_disallowed` argument (None for single triage, the sweep's list for
    re-triage) and its `max_turns` default of 60 (`_common.py:1146`). Nothing here
    re-derives the request: the body is read out of the socket the worker would have
    opened, so a change to the payload's shape lands here rather than in a fixture.
    """
    import httpx

    from workers.sources._common import new_worker_session, run_prompt_in_session

    posted: list[dict] = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeBackend(posted))
    sid = new_worker_session(title="clause 1 probe", source="autotriage",
                             model="primary")
    asyncio.run(run_prompt_in_session(
        prompt, title="clause 1 probe", source="autotriage",
        max_turns=max_turns, timeout_seconds=30.0,
        extra_disallowed=extra_disallowed, session_id=sid))

    assert len(posted) == 1, f"expected exactly one POST, got {len(posted)}"
    body = posted[0]
    # The POST's shape is half of what clause 1 is about. The worker composes no
    # system prompt for itself: the body carries the user text — which is where the
    # denied-tools block rides on this path (`_common.py:1280`) — plus the model
    # name, and the head is composed backend-side from the session. If a
    # `system_prompt` key ever appeared here, the equality asserted below would stop
    # being a claim about the backend's head, and this assert says so.
    assert "system_prompt" not in body, (
        "the worker shipped its own system prompt: the warm has to follow whichever "
        "side ends up owning the head, and it currently follows the backend's")
    return body


def _backend_head(body: dict) -> str:
    """The head the backend composes for that POST, through the route's own builder.

    `sessions_dir` is passed explicitly, and the reason is worth keeping: the test
    conftest redirects `app.sessions_io.SESSIONS_DIR` per test
    (`tests/conftest.py:633`), while `turn_options` bound `SESSIONS_DIR` by value out
    of `app.paths` at import (`app/routers/turn_options.py:64`), so `load`'s default
    reads a different directory from the one that wrote the session and answers
    "brand-new chat" — which is exactly the `mission-control` platform this assert
    caught on the first run. Reading from the writer's own name is what the route
    does in production, where the two are the same path.
    """
    from app import sessions_io
    from app.routers.turn_options import SessionSnapshot, build_turn_options

    snapshot = SessionSnapshot.load(body["session_id"],
                                    sessions_dir=sessions_io.SESSIONS_DIR)
    assert snapshot.platform == "worker", (
        "the session the job runs in must read as a worker session, or the head "
        "below is a chat's and not a job's")
    return build_turn_options(snapshot, body, "stream").system_prompt



def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── clause 1 ─────────────────────────────────────────────────────────────


def test_warm_payload_is_the_head_the_real_job_sends(wp, tmp_path, monkeypatch):
    """The warm's bytes are the head the backend composes for a real dispatch.

    A prefix one byte off from the real head warms blocks no run will ever ask for,
    and it fails silently: the warm answers `sent`, the ledger looks healthy, and the
    iteration-1 row never moves. So the comparison is against the job's own head read
    across the process boundary. `_worker_post_body` captures the JSON autotriage
    actually POSTs — built by `_run_turn` with the triage dispatch's own arguments
    (`workers/sources/autotriage.py:1163-1173`: `max_turns=25`, `model="primary"`,
    `extra_disallowed=None`) — and `_backend_head` runs that POST through
    `build_turn_options`, the function `/api/messages/stream` composes the head with.
    Neither side of the equality is a builder call this test chose to make.

    Both sides do end at `build_turn_options`, and that is the honest extent of what
    equality here proves: it pins the warm to the backend's composition for a worker
    session — session file, posted model name, `resolve_model`'s resolution included —
    rather than to a hand-typed copy. What it cannot see is a backend that composed a
    head differently from the builder it exposes; that builder is the route, so the
    witness for that would be the engine's own `prompt_tokens` against the ledger's
    `prompt_chars`, not a test. The version this node replaces compared two
    `_worker_run_options` calls and so could not see at all that the worker pool and
    the backend put the denial text in different places — see
    `test_the_two_worker_head_shapes_part_where_the_denial_text_goes`.

    The turn cap is asserted not to move the head, because the loop appends the state
    anchor after the history (`workers/sources/_common.py:778`); if the cap ever did
    move the head, one warm per source could not serve both the plain triage turn and
    the sweep turn.
    """
    from workers.sources._common import DENIED_TOOLS_HEADING

    warm_head, warm_opts = wp.warm_payload()
    post = _job_post(monkeypatch)
    real_head = _backend_head(post)

    assert _sha(warm_head) == _sha(real_head), (
        "the warm would prefill blocks the job never sends")
    assert warm_opts.system_prompt == real_head

    # Where the denial text lives, as an assert rather than as prose, because it is
    # the fact the two worker shapes part on. The real job's user text ends with the
    # block (`_common.py:1280`); its system prompt does not contain it, and neither
    # does the warm's — which is exactly why a warm built from `_worker_run_options`,
    # which appends that block to the system prompt at :710, is a warm the job busts.
    # Single triage passes `extra_disallowed=None`, so `build_denied_tools_block`
    # returns "" for it ("Returns \"\" for a turn with no list", `_common.py:137`) and
    # the posted user text is the prompt alone. The sweep dispatch passes a list, and
    # then the block lands at the END of that same user text — never in the head. So
    # the invariant that holds for both shapes is what is asserted: the user text
    # starts with the item content, and wherever the denial text appears it is a
    # suffix of the user text and absent from the head.
    assert post["text"].startswith("Item 4242"), (
        "the POST's user key is `text`, and the per-run item content is what follows "
        "the head — the part the warm must not contain")
    # The positive side, on the shape that does pass a list: the sweep dispatch
    # (`workers/sources/autotriage.py:1487`, `extra_disallowed=list(SWEEP_DISALLOWED)`).
    # Without this leg the paragraph above would be an `if` that one real dispatch
    # never enters, which is the vacuity this node exists to avoid.
    from workers.sources.autotriage import SWEEP_DISALLOWED

    swept = _job_post(monkeypatch, extra_disallowed=list(SWEEP_DISALLOWED),
                      prompt="Sweep item 9.")
    assert DENIED_TOOLS_HEADING in swept["text"], (
        "a dispatch that passes a denial list appends the block to the user text "
        "(`_common.py:1280`); if it stopped, the shape this node asserts is gone")
    assert swept["text"].startswith("Sweep item 9."), (
        "the block is appended after the item content, not before it, so the head is "
        "untouched by which denial list a dispatch happens to carry")
    assert DENIED_TOOLS_HEADING not in _backend_head(swept), (
        "and the head is the same builder output for both dispatch shapes — which is "
        "what lets one warm serve the triage turn and the sweep turn")
    assert DENIED_TOOLS_HEADING not in real_head, (
        "the head is the builder's output and nothing appended to it")
    assert DENIED_TOOLS_HEADING not in warm_head

    # The head really is the builder's output — not the sentinel re-typed here, and
    # not the builder's output with something appended.
    assert warm_head == SENTINEL
    assert "## Skills" not in warm_head, "the per-job skill body is not head content"
    # Nothing run-specific rode along: not the item the real POST carried, not a
    # round id, not the session title the snapshot was loaded from.
    for run_specific in ("3770", "SM_2026", "Item triage", "triaged today"):
        assert run_specific not in warm_head


def test_the_turn_cap_does_not_move_the_head(wp, tmp_path, monkeypatch):
    """One warm can serve a 25-turn triage and a longer turn: the cap is not head.

    the dispatcher puts `max_turns` in the POST body (`_common.py:1282`) and
    `build_turn_options` never reads it, so the head a job is shown does not depend
    on how many turns it may take. If that stopped being true, the single warm per
    source would stop covering the sweep dispatch, which passes its own cap.
    """
    post = _job_post(monkeypatch)

    assert _backend_head(post) == _backend_head(dict(post, max_turns=120))
    assert _sha(wp.warm_payload()[0]) == _sha(_backend_head(dict(post,
                                                                 max_turns=120)))


def test_the_two_worker_head_shapes_part_where_the_denial_text_goes(
        wp, tmp_path, monkeypatch):
    """The in-process `_worker_run_options` head is NOT the dispatch head.

    Recorded because this is the trap the previous round fell into, and because both
    shapes live in one file. `_worker_run_options`
    (`workers/sources/_common.py:644`) builds the turn for the in-process
    loopback-`RunOptions` callers — the voice prewarm, the self-modification gate's
    own turn — and appends the denial block to the **system prompt** at :710.
    autotriage never calls it: it dispatches through `run_prompt_in_session`, whose
    `_run_turn` appends the block to the **user text** at :1280 and POSTs no system
    prompt at all. A warm built from the first shape is therefore wasted against the
    second, and wasted silently — which is why `warm_payload` builds through
    `build_turn_options` instead of calling the worker-side builder, in spite of the
    worker-side builder being the thing the clause's wording named.

    The assertions are deliberate: the two heads differ, the in-process one is the
    longer (one builder output plus a block), and the warm follows the dispatch shape.
    """
    from workers.sources._common import DENIED_TOOLS_HEADING, _worker_run_options

    in_process = _worker_run_options(max_turns=25, source="autotriage",
                                     session_id="")
    post = _job_post(monkeypatch)
    dispatch_head = _backend_head(post)
    warm_head = wp.warm_payload()[0]

    assert _sha(in_process.system_prompt) != _sha(dispatch_head), (
        "the two worker shapes were supposed to disagree about where the denial text "
        "goes; if they now agree, `warm_payload`'s reason for avoiding "
        "`_worker_run_options` has to be re-read, not left standing")
    assert in_process.system_prompt.startswith(SENTINEL)
    assert DENIED_TOOLS_HEADING in in_process.system_prompt
    assert len(in_process.system_prompt) > len(dispatch_head), (
        "the in-process head is the builder output *plus* the denial block, so it is "
        "the longer one, and their bytes part at the end of the builder's output")
    assert dispatch_head == warm_head, (
        "the warm follows the dispatch shape, not the in-process one")


# ── clause 2 ─────────────────────────────────────────────────────────────


def test_warm_is_one_request_with_max_tokens_1_and_no_item_content(wp, engine):
    """One call, `max_tokens=1`, head only.

    `max_tokens=1` is the difference between a warm and an unrequested second
    inference: the prefill is the thing being paid for, and any generated tokens on
    top of it are cost with no product. The no-item-content half is the other half
    of the same money — a payload that carried the job's item text would be a
    second, non-reusable prompt.
    """
    row = asyncio.run(wp.prewarm("engine-ready"))

    assert row["outcome"] == "sent"
    assert len(engine.calls) == 1, "a warm is one request"
    options = engine.calls[0]["options"]
    assert options.extra_body["max_tokens"] == 1
    assert options.model == "primary"
    assert options.max_turns == 1
    # The payload is the head (as the system prompt the loop puts at position 0)
    # plus the one-character seed user turn that makes it a request at all. No item
    # block, no skill body, no run state.
    assert engine.calls[0]["messages"] == [{"role": "user", "content": "."}]
    assert options.system_prompt == wp.warm_payload()[0]


# ── clause 3 ─────────────────────────────────────────────────────────────


def test_admission_refuses_for_exactly_the_four_named_reasons(wp):
    """All four refusals, and the one case that admits.

    The rules belong to a pure function over (switch, in-flight, busy, already
    warmed) plus the module's registers — the shape the voice path uses for the same
    decision (`app/routers/voice.py:361`), so the policy is readable and testable
    without an engine under it.

    The in-flight register is keyed on the head sha rather than on the source
    because two triggers can land on one head — a boot and the nightly rewrite
    finishing together — and the second of them is the prefill the first is already
    paying for. The busy refusal is about the human: a warm beside a live turn
    steals prefill from the one request on this box somebody is waiting on.
    """
    sha = wp.warm_sha("a head")
    other = wp.warm_sha("a different head")

    assert wp._refusal("engine-ready", sha, enabled=False, busy=False) == "disabled"
    assert wp._refusal("engine-ready", sha, enabled=True, busy=False) is None

    wp._INFLIGHT.add(sha)
    assert wp._refusal("prompt-surface-write", sha,
                       enabled=True, busy=True) == "in flight"
    assert wp._refusal("prompt-surface-write", other,
                       enabled=True, busy=False) is None, \
        "a warm over a different head is not in flight"
    wp._INFLIGHT.discard(sha)

    assert wp._refusal("engine-ready", sha, enabled=True, busy=True) == "busy"

    wp._WARMED[sha] = "engine-ready"
    assert wp._refusal("head-sha-changed", sha, enabled=True, busy=False) \
        == "already warmed"

    # Switch off admits nothing, whatever the registers say.
    assert wp._refusal("engine-ready", other, enabled=False, busy=False) == "disabled"


def test_a_busy_engine_refuses_the_warm_and_the_row_says_so(wp, monkeypatch, engine):
    """`engine_busy` is the gate, and it is read per attempt, not cached.

    The gauge is `app/engine_pressure`'s sampled one — 1 Hz from `/metrics`, stale
    after 30 s — because an inline engine query at the moment of asking is the thing
    the worker pool's KV gate deliberately stopped doing.
    """
    from app import engine_pressure

    wp.reset_state()
    monkeypatch.setattr(engine_pressure, "latest", lambda *a, **kw: (
        engine_pressure.Sample(t=0.0, wall=0.0, kv=0.10, running=1, waiting=0)))
    row = asyncio.run(wp.prewarm("prompt-surface-write"))
    assert row["outcome"] == "refused" and row["reason"] == "busy"
    assert engine.calls == []


# ── clause 4 ─────────────────────────────────────────────────────────────


def test_repeated_triggers_warm_once_and_a_changed_head_warms_again(
        wp, engine, monkeypatch):
    """Every trigger, one warm per head version.

    This is what makes the trigger design safe to wire to things that fire often:
    `vault_write` sees MEMORY.md hundreds of times a night, and the nightly
    consolidation is not the only writer. A warm that served nothing is the item's
    named way to lose money, so an unchanged head cannot be warmed twice however
    many events announce it — and when the head really does move, the next trigger
    pays, because the old blocks are then worth nothing.
    """
    first = asyncio.run(wp.prewarm("engine-ready"))
    second = asyncio.run(wp.prewarm("prompt-surface-write"))
    third = asyncio.run(wp.prewarm("head-sha-changed"))

    assert [r["outcome"] for r in (first, second, third)] == \
        ["sent", "refused", "refused"]
    assert [r["reason"] for r in (second, third)] == ["already warmed"] * 2
    assert len(engine.calls) == 1

    # The prompt surface moves: SOUL.md got rewritten, so the head is different
    # bytes and the warm blocks are stale.
    _patch_head(monkeypatch, SENTINEL + "\nrewritten by the nightly pass")
    assert wp.head_sha() != first["sha256"], "the poll's own question sees the move"

    after = asyncio.run(wp.prewarm("prompt-surface-write"))
    assert after["outcome"] == "sent"
    assert after["sha256"] != first["sha256"]
    assert len(engine.calls) == 2


def test_a_boot_warms_once_more_because_the_engine_forgot(wp, engine):
    """The `engine-ready` exception to the once-per-head rule, and its one-use limit.

    A restart empties the engine's cache, so an `already warmed` answer after one
    names blocks that no longer exist and would refuse the single warm that is
    certain to be needed. The exception is spent, though: it is keyed on a boot
    having happened, not on somebody naming the trigger, because a rule any local
    caller could invoke at will is a way to re-pay the same prefill repeatedly.
    """
    before = asyncio.run(wp.prewarm("prompt-surface-write"))
    assert before["outcome"] == "sent"
    assert asyncio.run(wp.prewarm("head-sha-changed"))["reason"] == "already warmed"
    assert len(engine.calls) == 1

    after_boot = asyncio.run(wp.prewarm("engine-ready"))
    assert after_boot["outcome"] == "sent"
    assert after_boot["sha256"] == before["sha256"], "same head, new engine"
    assert len(engine.calls) == 2

    assert asyncio.run(wp.prewarm("engine-ready"))["reason"] == "already warmed"
    assert len(engine.calls) == 2, "the boot exception is spent, not a standing rule"


def test_an_unknown_trigger_is_refused_and_named(wp, engine):
    """A caller that invents a trigger gets a row naming it, and no engine call."""
    row = asyncio.run(wp.prewarm("on-tuesday"))
    assert row["outcome"] == "refused" and row["reason"] == "unknown trigger"
    assert engine.calls == []


# ── clause 5 ─────────────────────────────────────────────────────────────


def test_every_attempt_appends_one_parseable_row(wp, monkeypatch, tmp_path):
    """The ledger is the item's price tag, so it has to survive a failure too.

    Three attempts — one sent, one refused, one failed — and three rows, each one
    JSON on its own line, each carrying trigger, the head's sha256, the prompt size
    and the outcome. Warms issued and warm tokens come off these rows directly;
    hits-per-warm is the join, because the sha is what attributes a later run's
    cached fraction to this warm (`scripts/prefix_cache_iteration_table.py` is what
    prints that fraction, and the daily prefix-miss instrument is explicitly not the
    judge here). A ledger of successes only would read as a perfect ratio over three
    attempts.
    """
    path = wp.log_path()

    failing = FakeEngine(raise_on_call=True)
    monkeypatch.setattr("app.harness.run_query", failing.run_query)
    assert asyncio.run(wp.prewarm("engine-ready"))["outcome"] == "failed"

    ok = FakeEngine()
    monkeypatch.setattr("app.harness.run_query", ok.run_query)
    assert asyncio.run(wp.prewarm("prompt-surface-write"))["outcome"] == "sent"
    assert asyncio.run(wp.prewarm("head-sha-changed"))["outcome"] == "refused"

    lines = [json.loads(ln) for ln in
             path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3, "one row per attempt, failure included"
    assert [r["outcome"] for r in lines] == ["failed", "sent", "refused"]

    head = wp.warm_payload()[0]
    for r in lines:
        assert r["kind"] == "worker_prewarm"
        assert r["trigger"] in wp.PREWARM_TRIGGERS
        assert len(r["sha256"]) == 64, "the join key to the runs that hit the warm"
        assert r["prompt_chars"] == len(head), "the size of the prefix, in chars"
        assert r["source"] == wp.source()
        assert "ts" in r
        assert ("reason" in r) == (r["outcome"] != "sent"), \
            "an attempt that did not send says why"

    sent_row = lines[1]
    assert sent_row["max_tokens"] == 1
    assert sent_row["prompt_tokens"] == 21_940, "the warm's own cost, from the engine"
    assert sent_row["cached_tokens"] == 0, \
        "a cold head reads 0 here — see app/prefix_miss.py:25"
    assert lines[0]["reason"].startswith("ConnectionError")
    assert lines[2]["reason"] == "already warmed"

    warms = [r for r in lines if r["outcome"] == "sent"]
    assert len(warms) == 1
    assert sum(r["prompt_tokens"] or 0 for r in warms) == 21_940


# ── the seams this change crosses ────────────────────────────────────────


def test_the_trigger_route_answers_an_out_of_process_caller(wp, engine):
    """The writer that fires this lives in another process — the MCP server.

    The nightly consolidation and `automod_vault_land` rewrite the prompt surface
    from `agent_mcp`, not from the backend, so the trigger arrives over loopback
    HTTP. A refusal is a 200 with the reason (every caller is a fire-and-forget
    notifier, and a 4xx on a correctly-declined warm reads as a broken caller); an
    unknown trigger is a 400, because that is a caller bug the caller can fix.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import worker_prewarm as route

    client = TestClient(FastAPI())
    client.app.include_router(route.router)

    ok = client.post("/api/worker/prewarm",
                     json={"trigger": "prompt-surface-write"})
    assert ok.status_code == 200
    assert ok.json()["ok"] is True
    assert ok.json()["outcome"] == "sent"
    assert len(engine.calls) == 1

    again = client.post("/api/worker/prewarm",
                        json={"trigger": "prompt-surface-write"})
    assert again.status_code == 200
    assert again.json() == {**again.json(), "ok": False,
                            "outcome": "refused", "reason": "already warmed"}

    bad = client.post("/api/worker/prewarm", json={"trigger": "on-tuesday"})
    assert bad.status_code == 400
    assert len(engine.calls) == 1, "a 400 never reaches the engine"


def test_a_prompt_surface_write_notifies_the_backend_and_nothing_else(monkeypatch, tmp_path):
    """The MCP-side hook: three paths announce themselves, every other note does not.

    The trigger has to be the write completing, not an hour — a warm fired while the
    consolidation pass is still writing warms a prefix that dies minutes later,
    which is this item's named sequencing risk. The list is asserted against
    `app/prompt_surface`'s own three loaded files, so a fourth loaded surface cannot
    be added on one side and quietly missing on the other.

    The call is fire-and-forget and its failure is silent: the backend is routinely
    down at the instant a nightly job writes, and the consequence is that the poll
    warms the head instead — never a failed write.
    """
    import agent_mcp.vault as vault
    import urllib.request

    posted: list[str] = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        posted.append(req.full_url)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    for path in vault.PROMPT_SURFACE_PATHS:
        vault._notify_prompt_surface_write(path)
    assert len(posted) == len(vault.PROMPT_SURFACE_PATHS)
    assert all(u.endswith("/api/worker/prewarm") for u in posted)

    for path in ("memory/learnings/DAILY_NOTES.md", "knowledge/x.md",
                 "lloyd/bench/bench_001_x.md", "skills/foo/SKILL.md"):
        vault._notify_prompt_surface_write(path)
    assert len(posted) == len(vault.PROMPT_SURFACE_PATHS), \
        "an ordinary note is not a prompt-surface write"

    # The same claim, driven rather than asserted beside: the notify sits BELOW the
    # loaded-memory ceiling inside the real `_vault_write`, so a write the ceiling
    # refused changed no bytes and must notify nobody, while the identical write under
    # the ceiling must notify exactly once. Two calls, one path, one difference — the
    # size — so what is measured is the guard order and nothing else.
    _the_ceiling_refusal_notifies_nobody_and_the_accepted_write_notifies_once(
        monkeypatch, tmp_path, vault)


def test_a_failed_warm_is_a_row_and_not_a_raise(wp, monkeypatch):
    """A warm that cannot be sent is a cost the ledger records, never an exception.

    Both callers are fire-and-forget — a nightly writer's notify and a startup
    coroutine — and neither can use a traceback. The register is dropped too, so the
    next trigger may try again rather than being refused by a warm that never
    happened.
    """
    bad = FakeEngine(raise_on_call=True)
    monkeypatch.setattr("app.harness.run_query", bad.run_query)

    row = asyncio.run(wp.prewarm("engine-ready"))
    assert row["outcome"] == "failed"
    assert "ConnectionError" in row["reason"]
    assert wp._INFLIGHT == set()

    good = FakeEngine()
    monkeypatch.setattr("app.harness.run_query", good.run_query)
    wp._WARMED.clear()  # as the engine-ready trigger would
    assert asyncio.run(wp.prewarm("engine-ready"))["outcome"] == "sent"


def test_the_switch_turns_the_warm_off_without_touching_anything(
        wp, monkeypatch, engine):
    """One config line stops the mechanism, and it is checked before the build.

    #3771's acceptance is that a hits-per-warm below 1 leaves the switch off by
    default, so the off position has to be real: `disabled` is the first refusal
    checked, ahead of the prompt build, so a disabled warm reaches neither the vault
    nor the engine.
    """
    monkeypatch.setattr(wp, "enabled", lambda: False)
    row = asyncio.run(wp.prewarm("engine-ready"))
    assert row["outcome"] == "refused" and row["reason"] == "disabled"
    assert engine.calls == []
    assert wp.log_path().exists(), "a refusal is still an attempt, and is logged"


def test_startup_is_a_no_op_when_disabled(wp, monkeypatch):
    """The boot hook must not create a task, or a poller, or a warm, when off."""
    monkeypatch.setattr(wp, "enabled", lambda: False)
    asyncio.run(wp.start())
    assert wp._task is None


def test_the_warm_reaches_the_engine_as_one_body_with_max_tokens_1(wp, monkeypatch):
    """The seam the review named: the warm's intent survives into the /v1 request.

    `app.harness.client.stream_chat` is the send site (its own docstring: the sole
    place the OpenAI-format POST is composed), so this replaces only the socket and
    reads the JSON that would have gone to the engine. Three things have to be true
    of that body for the warm to be a warm and not a job:

      * ``max_tokens`` is 1 — it arrives through the ``extra_body`` spread at
        ``app/harness/client.py:177-178``, which is the last thing that touches the
        payload before the send, so nothing downstream can re-raise the cap;
      * ``messages`` is exactly the one system message holding the head, with the
        seed user turn as the only other line — a warm that shipped the item content
        would warm a prefix that includes it and bust on the next item;
      * the session the request is *logged* under is empty. ``session_id`` is not a
        body field at all on this path: ``stream_chat`` takes it for the #581
        component manifest alone (``record_request``, ``app/harness/client.py:196``,
        whose docstring says nothing on the request path reads it), and the headers
        are auth, content-type and accept. So the checkable form of 'the warm belongs
        to no run' is that the manifest line carries an empty session — a warm
        recorded under a live session would be counted as that session's iteration 1
        and would move the very number this item is measured by.

    What this does NOT cover, stated rather than implied: ``max_turns=1`` is the
    loop's stop condition, not a body field, so what proves it here is
    ``test_warm_is_one_request_with_max_tokens_1_and_no_item_content`` counting the
    sends; and the shape of a 1-token completion that happens to carry a tool call is
    not exercised at all — no test in this file drives the loop past a truncated
    tool-call delta, which is recorded as a gap on #3771.
    """
    import httpx

    import app.harness.client as client

    head, options = wp.warm_payload()
    sent: dict = {}
    manifest: list[dict] = []

    class _Resp:
        status_code = 200

        def request(self):  # only reached on a 4xx/5xx branch we never take
            return None

        async def aread(self) -> bytes:
            return b""

        async def aclose(self) -> None:
            return None

        async def aiter_lines(self):
            # One line, then the end of the stream: the fake's job is to let the
            # payload be observed, and [DONE] ends the turn with no chunks.
            yield "data: [DONE]\n"

    class _Stream:
        def __init__(self, method, url, headers=None, json=None):
            sent["method"], sent["url"] = method, url
            sent["headers"], sent["body"] = headers, json

        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, headers=None, json=None):
            return _Stream(method, url, headers, json)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(
        client, "record_request",
        lambda **kw: manifest.append({"session_id": kw.get("session_id"),
                                      "send_site": kw.get("send_site")}))

    async def _drive():
        # The warm's own call, with the loop's own stop condition standing behind
        # it: one send, then the fake ends the stream so the loop returns.
        # The loop prepends the system prompt to the history it is handed
        # (`app/harness/loop.py:291-292`), so this is the message list the warm
        # actually sends — head first, seed turn second.
        async for _ in client.stream_chat(
                base_url=options.base_url, model=options.model,
                messages=[{"role": "system", "content": head},
                          dict(wp._SEED_MESSAGE)],
                tools=None, extra_body=options.extra_body,
                cancel_event=None, timeout_s=90.0, api_key=options.api_key,
                priority=options.priority, session_id=options.session_id,
                iteration=1):
            pass

    asyncio.run(_drive())

    assert sent["url"].endswith("/v1/chat/completions")
    body = sent["body"]
    assert body["max_tokens"] == 1, (
        "the cap is the whole cost bound: without it a warm is an unbounded "
        "generation paid for out of the primary's idle window")
    assert body["stream"] is True
    assert body["messages"] == [{"role": "system", "content": head},
                                dict(wp._SEED_MESSAGE)], (
        "the body is the head plus the seed turn, in that order — the head must be "
        "the first thing the engine sees, since it is the first thing the job's "
        "engine sees too")
    assert "tools" not in body
    assert manifest and manifest[0]["session_id"] == "", (
        "the component-manifest line is the only place a warm's session id lands; a "
        "non-empty one would book the warm as some run's iteration 1")


def _the_ceiling_refusal_notifies_nobody_and_the_accepted_write_notifies_once(
        monkeypatch, tmp_path, vault):
    from app.prompt_surface import MEMORY_MD_CEILING_BYTES
    """Drive `_vault_write`'s real guards, in a scratch vault, with no stub of itself.

    The point of the shape is that the *only* difference between the two calls is the
    size of the content, so the thing being measured is the guard order and nothing
    else: same path, same caller, same vault.

    The refusal has to be identified, not just observed. `_vault_write` has half a
    dozen guards above the write — protected-path, bench-assertion, dispatch-field,
    route, the two OKF ones, the note-clobber guard — and a node that merely asserted
    "some error came back" would pass if the write were refused for a reason with
    nothing to do with the ceiling, which is the mistake this node replaces. So the
    assertion is on the ceiling guard's own wording, taken from the same module that
    sets the number.
    """
    calls: list[str] = []
    # `agent_mcp/vault.py` binds `VAULT` by value out of `agent_mcp._shared` at
    # import, and the write resolves `target = VAULT / path` from its own globals —
    # so the scratch tree is redirected at the name the function actually looks up,
    # which is the same lesson the notify patch below carries.
    scratch = tmp_path / "vault"
    (scratch / "lloyd").mkdir(parents=True)
    monkeypatch.setattr(vault, "VAULT", scratch)
    monkeypatch.setattr(vault, "AUDIT_LOG_DIR", scratch / "memory" / "audit")
    # The ceiling guard identifies a loaded-memory file by comparing the target's
    # parent to `MEMORIES_DIR` (`app/memory_ceiling.py:231-245`), so a scratch vault
    # that only redirected `VAULT` would leave the guard abstaining — which is
    # exactly how the 200 KB write below slipped through on the first run of this
    # node. Redirecting the directory is what makes the guard live in the scratch
    # tree; without this line the node would be asserting a refusal that its own
    # fixture had switched off.
    import app.memory_ceiling as memory_ceiling

    monkeypatch.setattr(memory_ceiling, "MEMORIES_DIR", scratch / "lloyd")
    # The spy is the seam the notify actually crosses: one POST, to one route, and
    # `urllib.request.urlopen` is the name it reaches for (`agent_mcp/vault.py`
    # imports the module, not the function), which is also what the node above
    # patches when it needs to see a failure.
    def _record(url, *args, **kwargs):
        calls.append(getattr(url, "full_url", str(url)))
        raise AssertionError("the notify must not reach the network from a test")

    monkeypatch.setattr("urllib.request.urlopen", _record)
    surface = "lloyd/MEMORY.md"

    # 200 KB against a 25,600-byte ceiling, so the arithmetic of the refusal is not
    # in doubt and the content's shape is nothing but frontmatter plus filler.
    over = "---\ntype: note\n---\n" + ("x" * 200_000)
    refused = vault._vault_write({"path": surface, "content": over})
    assert refused.get("error"), refused
    # The guard is identified by its own sentence and by the number it prints, which
    # is read from the same constant the guard reads. An unadorned
    # `if refused.get("error")` would stay green if any of the six guards above the
    # write had fired instead — that is the vacuity this node replaces, and the first
    # run of it earned the replacement: the error it caught was a *success*, because
    # the fixture had not redirected the directory the guard compares against.
    err = str(refused["error"])
    assert "over its" in err and "ceiling" in err, err[:300]
    assert err.startswith(f"MEMORY.md is {len(over):,} bytes"), err[:300]
    assert f"{MEMORY_MD_CEILING_BYTES:,}-byte ceiling" in err, err[:300]
    assert not (scratch / surface).exists(), (
        "the guard returns before the lock, the ledger and the write, so a refusal "
        "leaves no file behind — its own comment's promise")
    assert calls == [], (
        "a refused write changed no prefix, so nobody may be told — the notify sits "
        f"below the ceiling for a reason, and it fired anyway: {calls}")
    assert not (tmp_path / "vault" / surface).exists(), (
        "and it left no file behind, which is what the guard's own comment promises")

    # Same path, same frontmatter, 1/3000th the size: under the floor AND the ceiling,
    # so the only difference between this call and the one above is the guard order.
    under = "---\ntype: note\n---\n" + ("x" * 64)
    accepted = vault._vault_write({"path": surface, "content": under})
    assert accepted.get("success") is True, accepted
    assert len(calls) == 1 and calls[0].endswith("/api/worker/prewarm"), (
        "the accepted write is the one the notify is armed for: the prefix at "
        f"position 0 changed, which is exactly what the warm exists to re-establish "
        f"— got {calls}")
    assert (tmp_path / "vault" / surface).read_text() == under




# ── review retry (round SM_20261105_054745): head and policy, unpatched ───
#
# The three nodes below exist because the review refused the previous commit
# for a defect no node in this file could see. `warm_payload` builds its
# snapshot as `SessionSnapshot(session_id="", path=Path(""), raw_platform=
# "worker")`, and `SessionSnapshot.identity` (`app/routers/turn_options.py`
# :129-131) answers `("mission-control", "")` for any snapshot that does not
# exist, while `build_turn_options` takes the platform from `identity[0]`
# (:193). The kwarg was therefore inert and the warm shipped a chat head.
# The node above could not catch it for two reasons, and each is repaired
# here rather than argued: it patched `build_system_prompt` at both names
# through the `wp` fixture, so both sides of the sha equality were the same
# sentinel; and it took the comparison side from a snapshot that was itself
# built in-process, so nothing on either side was ever a session file.


def _memory_surface(tmp_path, monkeypatch):
    """Put a real memory surface under the builder, so the platform split exists.

    `_load_memories` resolves each member of `_memory_files_for(platform)` from the
    variant overlay, else `_CANON_MEMORIES_DIR` (`app/prompt_builder.py:988-991`),
    and renders it as `## <filename>` inside a `<memory>` block (:565, :1028). Under
    the test data root that directory holds nothing, the block renders empty, and a
    chat head and a worker head are then the SAME bytes — which is one of the two
    reasons the previous clause-1 node could assert a sha equality and mean nothing.
    Two files with distinct bodies make the split observable: MEMORY.md rides every
    turn, USER.md is dropped for a platform in `NON_USER_PLATFORMS`
    (`_memory_files_for`, :933-950).

    `freeze_enabled` is switched off so both sides read that one surface. The
    session side would otherwise take `frozen_memories`
    (`app/memory_snapshot.py:118-119`), which is a mechanism about one session's
    snapshot history and not about the platform split this node is about; the warm's
    own side has no session id and so never freezes.
    """
    mem = tmp_path / "memories"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("the index line every turn carries\n",
                                   encoding="utf-8")
    (mem / "USER.md").write_text("who Alan is, carried by a chat turn only\n",
                                 encoding="utf-8")
    import app.memory_snapshot as ms
    import app.prompt_builder as pb

    monkeypatch.setattr(pb, "_CANON_MEMORIES_DIR", mem)
    monkeypatch.setattr(ms, "freeze_enabled", lambda: False)
    return mem


def _session_built_worker_turn(tmp_path, monkeypatch):
    """A real worker session file, and the turn the backend builds for it.

    `create_session` (`app/sessions_io.py:398`) is the one writer
    `new_worker_session` uses, so the file is the one an autotriage run's
    session is; the turn is `build_turn_options` over the snapshot loaded back
    from it. No `build_system_prompt` is replaced at either name, which is what
    makes the two asserts about the memory block meaningful: with the real
    builder, a worker head ends in `</memory>` and carries no `## USER.md`
    section, and a chat head carries one — the split the review measured as
    43,988 bytes against 60,315.
    """
    from app.routers.turn_options import SessionSnapshot, build_turn_options
    from app.sessions_io import create_session

    _memory_surface(tmp_path, monkeypatch)
    sid = create_session("prewarm-clause-head", platform="worker",
                         model="primary", title="clause head",
                         source="autotriage", sessions_dir=tmp_path)
    snapshot = SessionSnapshot.load(sid, sessions_dir=tmp_path)
    assert snapshot.exists and snapshot.platform == "worker", (
        "the comparison side must be a worker session read back from a file; "
        "two synthetic snapshots are how the wrong head went unmeasured")
    body = {"model": "primary", "session_id": sid}
    return build_turn_options(snapshot, body, "stream"), body


def test_the_warm_head_is_the_builder_output_for_a_worker_session_file(
        tmp_path, monkeypatch):
    """Clause 1, unpatched: the warm's head is the job's head, byte for byte.

    The byte equality is the whole item: a prefix one byte off warms blocks no
    run asks for, and the failure is silent — the ledger still says `sent`.
    The two memory asserts are the node's teeth against the specific bug: they
    can only hold for a head built with `platform="worker"`, so a warm that
    reverted to the chat head reddens here even if some future refactor made
    the sha comparison pass by other means.
    """
    from pathlib import Path

    from app import worker_prewarm

    built, _ = _session_built_worker_turn(tmp_path, monkeypatch)
    warm_head, warm_opts = worker_prewarm.warm_payload()

    assert warm_head == built.system_prompt, (
        "the warm would prefill blocks the job never sends")
    assert warm_opts.system_prompt == built.system_prompt
    assert "</memory>" in warm_head, (
        "the head must be the real rendered prompt, not a stub: a test whose "
        "builder returns a sentinel cannot see a platform split at all")
    assert "## USER.md" not in warm_head, (
        "a head carrying a USER.md section is a chat's: the worker platform "
        "omits it, which is the visible half of the 43,988/60,315 split")

    # And the surface must be able to express that split at all. Without this
    # the assert above could pass because USER.md was simply missing from the
    # tree the builder reads, which is the state `_memory_surface` exists to
    # end — so the same builder, over the same two files, is pointed at a
    # snapshot with no worker platform and has to produce the section.
    from app.routers.turn_options import SessionSnapshot, build_turn_options

    chat = build_turn_options(
        SessionSnapshot(session_id="", path=Path("nowhere.json"), exists=True),
        {"model": "primary"}, "stream")
    assert "## USER.md" in chat.system_prompt and "## MEMORY.md" in chat.system_prompt, (
        "the memory surface in this test is not rendering both members, so "
        "'no ## USER.md' on the warm would be an absence of files, not a "
        "platform split")


def test_the_warm_turn_carries_the_real_worker_turn_s_tool_policy(
        tmp_path, monkeypatch):
    """Clause 2: the warm is banned what an autotriage turn is banned.

    `disallowed_tools_refresh` is the refresher `build_turn_options` installs
    (:284) and the loop re-reads every iteration (`app/harness/loop.py:565`),
    so it is the live policy rather than a snapshot of it. The three names are
    the point: `automod_start`/`automod_land` come from
    `messages._ban_automod_for_workers`, armed on the session's own platform
    (#709), and `grant_create` from `_ban_grant_minting`. A warm whose snapshot
    does not exist gets identity `("mission-control", "")`, so neither ban is
    written and its policy is a human's — which is both the bug and, on the
    engine, a different tools array, so a different rendered prefix and a miss.
    """
    from app import worker_prewarm

    built, _ = _session_built_worker_turn(tmp_path, monkeypatch)
    _, warm_opts = worker_prewarm.warm_payload()

    real = sorted(built.options.disallowed_tools_refresh())
    warm = sorted(warm_opts.disallowed_tools_refresh())
    assert warm == real, (
        "the warm's tool policy differs from the job's, so its tools array "
        "differs, so the prefix the engine renders is not the job's")
    for name in ("automod_start", "automod_land", "grant_create"):
        assert name in warm, (
            f"{name} is banned on every worker turn by the session's own "
            "platform; a warm missing it is not a worker turn")


def test_the_warm_body_at_the_send_seam_is_the_head_plus_the_worker_surface(
        tmp_path, monkeypatch):
    """Clause 3: what reaches the socket, across the loop.

    Driven through `run_query`, not by calling `stream_chat` with arguments
    this test chose, because the two things clause 3 is about are both the
    loop's doing: the system prompt is prepended at `app/harness/loop.py:291`
    and the advertised tools array is built from the MCP pool minus the turn's
    disallowed set and its surface. So the worker turn is run over the same
    fake catalog and the same refresher, and its advertised names are the
    standard the warm's are compared to — a warm that lost a ban would advertise
    `automod_start` and reddens on the equality, not on a hand-written list.
    """
    from app import worker_prewarm
    from app.harness.tests.test_loop_tool_search import (
        _drain, _FakePool, _mcp_tool, _patch_pool)

    catalog = ["Bash", "vault_search", "automod_start", "automod_land",
               "grant_create", "session_inject_context"]
    built, _ = _session_built_worker_turn(tmp_path, monkeypatch)
    head, options = worker_prewarm.warm_payload()

    class _Seam:
        """`app.harness.loop.stream_chat`, as a recorder of whole kwargs."""

        def __init__(self):
            self.calls: list[dict] = []

        def __call__(self, **kwargs):
            # A shallow copy is not enough: the loop appends the assistant
            # reply to the SAME list object it passed in, so the snapshot has
            # to copy the messages and tools it was handed at call time.
            snap = dict(kwargs)
            snap["messages"] = [dict(m) for m in (kwargs.get("messages") or [])]
            snap["tools"] = [dict(t) for t in (kwargs.get("tools") or [])]
            self.calls.append(snap)

            async def _gen():
                yield {"choices": [{"delta": {"content": "k"},
                                    "finish_reason": "stop"}]}
            return _gen()

    def _run(options, messages):
        seam = _Seam()
        _patch_pool(monkeypatch, _FakePool("lloyd-mcp",
                                           [_mcp_tool(n) for n in catalog]))
        monkeypatch.setattr("app.harness.loop.stream_chat", seam)
        asyncio.run(_drain(messages, options))
        return seam

    warm_seam = _run(options, [dict(worker_prewarm._SEED_MESSAGE)])
    job_seam = _run(built.options,
                    [{"role": "user", "content": "Item 4242 needs triage."}])

    assert len(warm_seam.calls) == 1, (
        "a warm is one request; more than one send is a turn, paid in full")
    body = warm_seam.calls[0]
    assert body["messages"] == [{"role": "system", "content": head},
                                dict(worker_prewarm._SEED_MESSAGE)], (
        "the head is position 0 and the seed turn is the only other line: the "
        "item content must not ride along, or the warm busts on the next item")
    assert body["extra_body"]["max_tokens"] == 1, (
        "max_tokens rides in extra_body at this call site "
        "(app/harness/loop.py:652) and is spread last into the payload; "
        "without it a warm is an unbounded generation")

    warm_names = {t["function"]["name"] for t in (body["tools"] or [])}
    job_names = {t["function"]["name"]
                 for t in (job_seam.calls[0]["tools"] or [])}
    assert warm_names, "a toolless warm renders a prefix no job's matches"
    assert warm_names == job_names, (
        "the tools array is part of the rendered prompt: an advertised set "
        "that differs from the job's is a different prefix and a cache miss")
    assert not ({"automod_start", "automod_land", "grant_create"} & warm_names), (
        "the worker bans have to reach the advertised array, not only the "
        "policy function")
