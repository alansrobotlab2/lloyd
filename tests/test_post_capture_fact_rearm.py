"""Item #1159 — fact extraction must re-arm, and the `captured` latch must not stop it.

`_post_session_capture` runs after **every** turn (`app/routers/messages.py:1309`,
`:1395`). Before this fix it opened with `if data.get("captured"): return` and
extracted facts only inside `if user_msg_count >= 3:` on that same single pass.
One boolean could mean one thing, and the summary half won it: turn 1 nearly
always yields a non-trivial summary, so `captured` was true before message 3
existed and the extraction branch was unreachable by construction. Measured on
the tree this fix was written against (HEAD `b6d19fc2`, 2026-09-20): 65 sessions
were `captured` with ≥3 user messages, and 13 of them had ever produced a
`session-extracted` fact. The retained server logs (`logs/server.err` …
`server.err.10`, 09-16 → 09-20) carry the string `facts extracted` **exactly
once** — 2026-09-18 11:17:58 — and the three lines around it put the export, the
extraction and the summary in the *same* pass: that session was lucky, its first
capture pass did not arrive until it already held 4 user messages. No session
has ever extracted on a second pass, which is what the watermark is for.

The fix separates the two gates: `captured` still guards the once-per-session
summary, and a new `fact_watermark` message index guards extraction. These tests
pin each half of that, and pin the cost of getting the re-arm wrong — the
secondary engine is single-tenant (`llama.cpp --parallel 1`), so a pass with
nothing new must issue **zero** calls. The store refuses a byte-identical re-add
on `(entity, text_hash)` (#499) but is blind to a paraphrase, and this extractor
asks the model to restate — so the watermark is what stands between a long voice
session and the same fact rewritten, in different words, every pass.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
from datetime import datetime

import pytest

import agent_mcp.facts as facts_mod
from app import post_capture, sessions_io

SID = "20260920_120000_aaaaaa"

# One fact per extraction, so `len(calls["fact_add"])` is also a call count.
EXTRACTED = [{"entity": "Goliath", "fact": "runs the tts service on port 8090"}]


def _msg(role: str, text: str, n: int) -> dict:
    return {
        "id": f"{role}{n}",
        "role": role,
        "content": [{"type": "text", "text": text}],
        "timestamp": "2026-09-20T12:00:00",
    }


def _exchange(n: int, text: str) -> list[dict]:
    """A user turn and the reply to it, the shape a chat session actually holds."""
    return [_msg("user", text, n), _msg("assistant", f"Noted: {text[:24]}", n)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A temp session dir, a temp `~` for the daily note, and counting engines.

    Both secondary calls are replaced: the real ones are HTTP to a single-slot
    llama.cpp, so a suite that reached them would queue behind the machine's own
    background work. `agent_mcp.facts._fact_add` is replaced rather than
    `post_capture._write_extracted_facts`, so the assertion lands on the seam
    post_capture actually writes through (the import inside
    `_write_extracted_facts` resolves this attribute at call time).
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    home = tmp_path / "home"
    (home / "obsidian" / "memory").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("USERPROFILE", raising=False)

    monkeypatch.setattr(post_capture, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(post_capture, "VAULT_SESSIONS_DIR", tmp_path / "vault-sessions")
    monkeypatch.setattr(post_capture, "VAULT_BACKGROUND_SESSIONS_DIR",
                        tmp_path / "vault-background")

    calls: dict = {"summary": [], "facts": [], "fact_add": []}

    def fake_summary(transcript: str) -> str:
        calls["summary"].append(transcript)
        return "Talked through the capture worker change; shipped the model key fix."

    def fake_facts(transcript: str) -> list[dict]:
        calls["facts"].append(transcript)
        return list(EXTRACTED)

    def fake_fact_add(payload: dict) -> dict:
        calls["fact_add"].append(payload)
        return {"success": True}

    monkeypatch.setattr(post_capture, "_sync_secondary_capture_call", fake_summary)
    monkeypatch.setattr(post_capture, "_sync_secondary_fact_extraction", fake_facts)
    monkeypatch.setattr(facts_mod, "_fact_add", fake_fact_add)

    def write(messages: list[dict], sid: str = SID, **extra) -> dict:
        """Replace a session's messages, keeping whatever else the file holds.

        Merging rather than rewriting is what makes the first test mean what it
        says: the session under test has to still carry `captured: true` written
        by its own first pass, since surviving that flag is the clause.
        """
        path = sessions / f"{sid}.json"
        data = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        data.update({
            "session_id": sid,
            "created_at": "2026-09-20T12:00:00",
            "model": "primary",
            "messages": messages,
        })
        data.update(extra)
        (sessions / f"{sid}.json").write_text(json.dumps(data), encoding="utf-8")
        return data

    def read(sid: str = SID) -> dict:
        return json.loads((sessions / f"{sid}.json").read_text(encoding="utf-8"))

    def daily_note() -> Path:
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        return home / "obsidian" / "memory" / f"{today}.md"

    return SimpleNamespace(sessions=sessions, calls=calls, write=write, read=read,
                           daily_note=daily_note)


async def test_first_turn_latch_does_not_close_the_extraction_gate(env):
    """Clause 1: a session that latched on turn 1 still extracts at message 3.

    This is the exact shape of the failure, not an abstraction of it: the
    spoken-conversation case is a long-lived session whose single pass ran on its
    first turn. Pass one here has one user message, so it must summarise and must
    NOT extract (below the threshold). The two later turns then arrive; the
    second pass finds `captured` already true — which is the condition that used
    to mean "return" — and has to extract anyway while still writing no second
    summary.
    """
    env.write(_exchange(1, "Set the built-in tts voices back up on goliath"))

    await post_capture._post_session_capture(SID)

    assert len(env.calls["summary"]) == 1, "the first pass must produce the summary"
    assert env.calls["facts"] == [], (
        "one user message is below the threshold; no extraction call is owed"
    )
    assert env.read()["captured"] is True, "the summary pass must latch `captured`"
    assert "fact_watermark" not in env.read(), (
        "a pass that never extracted must not claim a watermark"
    )

    messages = env.read()["messages"] + _exchange(2, "Also the wake word needs a media port check") \
        + _exchange(3, "And note that the voice room session is the chat session")
    env.write(messages)

    await post_capture._post_session_capture(SID)

    assert len(env.calls["summary"]) == 1, (
        "the latch still guards the summary: re-arming extraction must not add "
        "a second one"
    )
    assert len(env.calls["facts"]) == 1, (
        "3 user messages with no watermark is exactly the threshold; the "
        "`captured: true` already in the file must not suppress it"
    )
    assert len(env.calls["fact_add"]) == 1, env.calls["fact_add"]
    written = env.calls["fact_add"][0]
    assert written["category"] == "session-extracted"
    assert written["source_doc"] == f"sessions/{SID}", (
        "the acceptance check counts facts by `source_doc: sessions/<id>`, so a "
        "fact that names no session is invisible to it"
    )
    assert env.read()["fact_watermark"] == len(messages), (
        "the watermark must reach the end of what the extractor was shown"
    )


async def test_two_new_user_messages_issue_zero_extraction_calls(env):
    """Clause 2: the gate is 3 NEW user messages past the watermark, not 3 total.

    Two sessions, one file each, differing only in how many user messages sit
    past the watermark — the boundary has to bite on one side and not the other,
    or "event-gated" is unenforced. Per-turn extraction is what the single-tenant
    secondary engine cannot carry, and a pass that re-asks about messages it has
    already seen spends that slot on a window already answered — and gets back a
    paraphrase, which is the one shape the store's `(entity, text_hash)` guard
    (#499) cannot refuse.
    """
    # Five exchanges, ten messages: user turns at indices 0, 2, 4, 6, 8. A
    # watermark of 6 leaves turns 4 and 5 unseen (2 new user messages); 4 leaves
    # turns 3, 4 and 5 unseen (3). Both sessions are latched and differ only in
    # that number.
    messages = [m for i in range(1, 6) for m in _exchange(i, f"Turn {i} about the capture worker")]
    env.write(messages, sid="20260920_120000_twoooo", captured=True, fact_watermark=6)
    env.write(messages, sid="20260920_120000_threeee", captured=True, fact_watermark=4)

    await post_capture._post_session_capture("20260920_120000_twoooo")
    assert env.calls["facts"] == [], "2 new user messages must issue zero calls"
    assert env.calls["fact_add"] == []
    assert env.calls["summary"] == [], "a latched session owes no second summary"

    await post_capture._post_session_capture("20260920_120000_threeee")
    assert len(env.calls["facts"]) == 1, "3 new user messages is the threshold"
    assert len(env.calls["fact_add"]) == 1
    assert env.read("20260920_120000_threeee")["fact_watermark"] == len(messages)


async def test_rerunning_the_pass_with_no_new_messages_writes_no_new_facts(env):
    """Clause 2: the pass is idempotent with nothing new, because the engine cannot be asked twice.

    The store refuses a byte-identical re-add on `(entity, text_hash)` (#499), so
    a re-sent tail would not multiply that exact row — but the refusal lands
    *after* the secondary-model call, and on a `--parallel 1` engine that call is
    the cost. The model's second answer is also a paraphrase, the one shape the
    key cannot see, so a re-sent tail still lands a near-duplicate. Both halves
    make the same demand: idempotence lives at the watermark, not downstream.
    """
    env.write([m for i in range(1, 5)
               for m in _exchange(i, f"Turn {i}: redeploy the livekit worker and recheck")])

    await post_capture._post_session_capture(SID)
    first_writes = len(env.calls["fact_add"])
    first_watermark = env.read()["fact_watermark"]
    assert first_writes == 1 and first_watermark == 8

    for _ in range(3):
        await post_capture._post_session_capture(SID)

    assert len(env.calls["facts"]) == 1, (
        f"three more passes with no new messages issued {len(env.calls['facts']) - 1} "
        "more secondary calls; expected none"
    )
    assert len(env.calls["fact_add"]) == first_writes, (
        "the same window was written a second time"
    )
    assert env.read()["fact_watermark"] == first_watermark


async def test_rearming_does_not_add_a_second_auto_captured_section(env):
    """Clause 4: daily-note behaviour is unchanged, exactly one section per session.

    The heading is what daily notes are grepped by, and a re-armed extraction
    that re-summarised would put a second `Auto-captured` section in the user's
    own record of the day for a conversation they already saw summarised.
    """
    env.write(_exchange(1, "Walk me through the post-capture latch"))
    await post_capture._post_session_capture(SID)

    text = env.daily_note().read_text(encoding="utf-8")
    assert text.count("— Auto-captured") == 1

    env.write(env.read()["messages"] + _exchange(2, "Second turn about the same latch")
              + _exchange(3, "Third turn, now enough to extract"))
    await post_capture._post_session_capture(SID)

    after = env.daily_note().read_text(encoding="utf-8")
    assert after.count("— Auto-captured") == 1, (
        "re-arming extraction added a daily-note section"
    )
    assert len(env.calls["facts"]) == 1, "the extraction half still has to run"


async def test_a_failed_extraction_call_leaves_the_watermark_behind(env, monkeypatch):
    """A refused call is not a consumed window: the next pass retries it.

    Advancing the watermark on a transport failure would silently drop a
    stretch of conversation, which is the failure mode #1159 exists to remove.
    """
    attempts: list[str] = []

    def vllm_502(transcript: str):
        attempts.append(transcript)
        raise RuntimeError("vllm 502")

    monkeypatch.setattr(post_capture, "_sync_secondary_fact_extraction", vllm_502)
    env.write([m for i in range(1, 4)
               for m in _exchange(i, f"Turn {i}: a durable claim worth extracting")])

    await post_capture._post_session_capture(SID)

    assert attempts, "the fixture must have exercised the failure path"
    assert env.read().get("fact_watermark", 0) == 0, (
        "a failed call advanced the watermark, so those messages are lost"
    )
    assert env.calls["fact_add"] == []
    assert env.read()["captured"] is True, "the summary half is unaffected by this"

    monkeypatch.setattr(post_capture, "_sync_secondary_fact_extraction",
                        lambda t: list(EXTRACTED))
    await post_capture._post_session_capture(SID)

    assert len(env.calls["fact_add"]) == 1, (
        "the retry on the next pass must extract what the failure skipped"
    )


async def test_a_background_session_gets_neither_summary_nor_facts(env):
    """The platform rail survived the split: a worker's notes to itself stay out.

    Extraction used to sit behind the summary's `is_user_session` early return.
    It now has its own, so this has to be pinned — otherwise "the fact pass is
    re-armed" quietly becomes "every autocode run writes facts about itself into
    the knowledge graph", at ~240 background sessions a day against ~14 chats.
    """
    env.write([m for i in range(1, 5)
               for m in _exchange(i, f"Turn {i}: tool output from an autocode run")],
              platform="worker")

    await post_capture._post_session_capture(SID)

    assert env.calls["summary"] == []
    assert env.calls["facts"] == []
    assert env.calls["fact_add"] == []
    assert env.read()["captured"] is True, "a latched background session is not re-walked"


async def test_the_watermark_only_moves_forward_and_never_off_the_end(env):
    """What `_set_fact_watermark` keeps under `mutate_session`, and how it says so.

    Every case goes through the real `mutate_session`, because that is the only
    caller and it is what makes the claim here observable in the place it matters:
    `sessions_io.mutate_session:819-820` calls the callback and then writes the
    file unconditionally, whatever the callback decided. So a refusal that only
    refrained from touching the dict would still be persisted — and the assertion
    that carries this test is the value read back out of the session file on disk,
    not the dict the callback saw. Calling the callback directly would test a
    dict and prove nothing about the file the next pass reads.

    Backwards would re-extract an old window; past the end of the message list
    would mark messages never sent as already seen, and a message skipped
    silently is a fact lost with no record that it was ever owed. Both refusals
    leave the on-disk value where it was.
    """
    async def apply(covered: int, sid: str, expect: int | None = None):
        """Write `covered` through `mutate_session`; returns (verdict, stored).

        `expect` defaults to the value in the file now. Passing it explicitly is
        how the lost-race case is built: `expect` is what the losing pass read
        before it called the model, and the stored value has since moved.
        """
        expect = post_capture._fact_watermark(env.read(sid)) if expect is None else expect
        callback = post_capture._set_fact_watermark(covered, expect=expect)
        # True means the file existed and the callback ran; it says nothing about
        # whether the callback accepted the write, which is why `.result` exists.
        assert await sessions_io.mutate_session(sid, callback) is True
        return callback.result, env.read(sid).get(post_capture.FACT_WATERMARK_KEY, 0)

    # Sixteen messages, watermark at 6: three user turns sit unseen, so the
    # session is one the gate would open for.
    msgs = [m for i in range(1, 9) for m in _exchange(i, f"Turn {i} about the tts port")]
    env.write(msgs, sid="a_lowered", fact_watermark=6)
    assert await apply(2, "a_lowered") == ("lowered", 6), (
        "a target below the stored value reached the file, rewinding the window "
        "so the same turns get re-extracted and the model restates facts the "
        "store's verbatim key cannot refuse"
    )

    env.write(msgs, sid="a_shrink", fact_watermark=6)
    assert await apply(len(msgs) + 5, "a_shrink") == ("shrink", 6), (
        "a target past the end of the message list reached the file, marking "
        "messages that do not exist as already extracted"
    )

    env.write(msgs, sid="a_advanced", fact_watermark=6)
    assert await apply(len(msgs), "a_advanced") == ("advanced", len(msgs)), (
        "the honest case must persist — a guard that refuses everything looks "
        "identical to one that works"
    )

    env.write(msgs, sid="a_raced", fact_watermark=8)
    assert await apply(len(msgs), "a_raced", expect=6) == ("already-advanced", 8), (
        "a pass that lost the race has to be told its write did not land, not "
        "left believing it advanced the window"
    )

    for junk in (None, "x", -3, True, 1.5, float("nan"), float("inf")):
        assert post_capture._fact_watermark({"fact_watermark": junk}) == 0, (
            f"`fact_watermark: {junk!r}` must read as 'never extracted', not as an offset"
        )

async def test_a_window_over_one_budget_leaves_the_watermark_behind_and_drains(env):
    """Seam 2: the window handed the model is not the window the file is stamped with.

    `_transcript_line` caps a line at 600 chars, so nine oversized user turns
    render to about 5.5k characters — over the 4000-char budget, which is what
    makes this a trimming case rather than a fixture that merely looks big. The
    first pass covers the prefix that fits and stops; the residue has to still be
    owed. Stamping `len(messages)` off a trimmed string would mark the unseen
    turns read, which is #1159's loss made permanent by the thing that fixes it.
    """
    huge = [{"role": "user", "content": ("u" * 700) + f" turn {i}"} for i in range(1, 10)]
    env.write(huge)

    await post_capture._post_session_capture(SID)

    sent = env.calls["facts"]
    assert len(sent) == 1, "the first pass must call out, gate satisfied at 9 user turns"
    assert len(sent[0]) <= post_capture.FACT_TRANSCRIPT_BUDGET, (
        f"the extractor was handed {len(sent[0])} chars, over the "
        f"{post_capture.FACT_TRANSCRIPT_BUDGET}-char budget"
    )
    covered = env.read()[post_capture.FACT_WATERMARK_KEY]
    assert 0 < covered < len(huge), (
        f"a {len(sent[0])}-char transcript over {len(huge)} unseen messages "
        f"stamped all of them read (watermark {covered})"
    )

    # Drain it: successive passes take the next slice until nothing is unseen.
    for _ in range(10):
        if env.read()[post_capture.FACT_WATERMARK_KEY] == len(huge):
            break
        before = len(env.calls["facts"])
        await post_capture._post_session_capture(SID)
        after_wm = env.read()[post_capture.FACT_WATERMARK_KEY]
        assert after_wm > covered, (
            f"a pass covered nothing and left the watermark at {covered}"
        )
        # Exactly one more call. `calls["facts"]` is append-only, so `>=` could
        # never fail and asserted nothing about the pass it followed; equality is
        # what makes "a window advanced" and "the model was asked about it" one
        # claim instead of two. A watermark that moved on a pass which did not
        # call the model marks messages read that nobody read.
        assert len(env.calls["facts"]) == before + 1, (
            f"the watermark moved from {covered} to {after_wm} on a pass that "
            f"issued {len(env.calls['facts']) - before} extraction calls"
        )
        covered = after_wm

    assert env.read()[post_capture.FACT_WATERMARK_KEY] == len(huge), (
        "the backlog never drained: some messages stay unseen forever, which is "
        "the loss this item exists to remove"
    )
    calls = len(env.calls["facts"])
    await post_capture._post_session_capture(SID)
    assert len(env.calls["facts"]) == calls, (
        "a drained session issues another extraction call with nothing to say"
    )


async def test_a_compaction_that_replaces_the_messages_reshapes_the_window_it_stamps(env):
    """The session file is shared mutable state across a process boundary.

    Manual `/compact` replaced `data["messages"]` wholesale until D11 (a hand
    edit still can): its `_swap` callback assigned the compacted list, refreshed
    `last_active` and `message_count`, and writes nothing else — so the
    `fact_watermark` it leaves behind was counted against a list that no longer
    exists. `mutate_session` then persists the dict whatever the callback decided,
    and the file ends up claiming coverage 10 messages past its own end.

    That is fatal in the specific way #1159 is about: the reader's own rail is
    `messages[watermark:]`, which on a stale watermark is the empty list. Zero new
    user messages, forever — the gate reports "nothing new" about a conversation
    that keeps growing, and the session rejoins the 63-eligible/12-with-a-fact
    population from the item's own count, permanently. So the watermark is clamped
    to what the file can actually vouch for at read time (`_fact_watermark`),
    which repairs on the next pass and never needs the writer's cooperation.

    `_swap` is reproduced here rather than imported because it is a closure inside
    the compaction generator; what matters is the three keys it writes and the one
    it does not, and those are copied.
    """
    msgs = [m for i in range(1, 11) for m in _exchange(i, f"Turn {i} on the livekit media ports")]
    env.write(msgs, captured=True)

    # An extraction pass over the full 20-message session: 10 user messages, all
    # unseen, drains inside one budget, so the watermark lands on the list's end.
    await post_capture._post_session_capture(SID)
    stale = env.read()[post_capture.FACT_WATERMARK_KEY]
    assert stale == len(msgs), "the fixture must have a covered, stamped session"
    assert len(env.calls["facts"]) == 1

    # /compact: keep the last ten messages, say nothing about the watermark.
    kept = msgs[-10:]

    def _swap(data: dict) -> None:
        data["messages"] = kept
        data["last_active"] = "2026-09-20T12:30:00"
        data["message_count"] = len(kept)

    assert await sessions_io.mutate_session(SID, _swap) is True
    assert env.read()[post_capture.FACT_WATERMARK_KEY] == stale == 20, (
        "the compaction as written no longer leaves the stale stamp behind, so "
        "this test is no longer describing the seam it was written for"
    )
    assert stale > len(env.read()["messages"]), (
        "the fixture must leave the file claiming coverage past its own end"
    )

    # Three more spoken turns arrive on the compacted session — the threshold.
    env.write(env.read()["messages"]
              + _exchange(21, "Also note goliath runs tts on 8090")
              + _exchange(22, "And the wake word needs a media port check")
              + _exchange(23, "Third new turn after the compaction"))
    unseen_now = env.read()["messages"][stale:]
    assert unseen_now == [], (
        "the fixture must show why the stale stamp is fatal read raw: slicing at "
        f"20 on a {len(env.read()['messages'])}-message list yields nothing"
    )

    await post_capture._post_session_capture(SID)

    assert len(env.calls["facts"]) == 2, (
        f"a compaction left the watermark at {stale} over a "
        f"{len(env.read()['messages'])}-message file, so every later pass read "
        "`messages[20:]` — empty — and reported nothing new"
    )
    assert env.read()[post_capture.FACT_WATERMARK_KEY] == len(env.read()["messages"]), (
        "the pass that reset a stale stamp has to re-stamp the real coverage, or "
        "every later pass re-extracts from the same point and spends the engine "
        "slot restating facts the store's verbatim key cannot refuse"
    )


async def test_the_pass_is_not_reentrant_per_session(env, monkeypatch):
    """Two concurrent dispatches for one session must not both issue a call.

    `messages.py` fires the post-turn task on every turn, so a session whose
    turns arrive faster than the single-slot engine drains them does overlap —
    and two passes over the same window spend two calls on one slot and land two
    answers, since the second is a paraphrase the store's verbatim key cannot
    refuse. The first pass is parked inside the
    secondary-model call so the overlap is real rather than asserted.
    """
    import threading

    hold = threading.Event()
    release = threading.Event()
    started: list[str] = []

    def parked(transcript: str) -> list[dict]:
        started.append(transcript)
        hold.set()
        release.wait(timeout=10)
        return list(EXTRACTED)

    monkeypatch.setattr(post_capture, "_sync_secondary_fact_extraction", parked)
    env.write([m for i in range(1, 4) for m in _exchange(i, f"Turn {i} at {i * 3} seconds")])

    first = asyncio.create_task(post_capture._post_session_capture(SID))
    while not started:
        await asyncio.sleep(0.01)

    await post_capture._post_session_capture(SID)
    assert len(started) == 1, "the second pass issued its own call while the first ran"

    release.set()
    await first
    assert env.read()[post_capture.FACT_WATERMARK_KEY] == 6, (
        "the parked pass lost its watermark write on the way out"
    )
