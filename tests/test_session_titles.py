"""Session titles — cleaning, the re-title schedule, and the read cache.

Three things here are easy to get wrong in a way that looks fine:

* **A bad title is worse than no title.** ``Here is a title for the
  conversation`` in the sidebar looks like a bug; the session id at least
  identifies the row. `clean_title` is therefore strict, and returning ""
  is a normal outcome, not a failure.
* **Re-titling is geometric, not per-turn.** The secondary is a
  single-tenant llama.cpp slot that agent turns already queue behind. A
  title call on every turn completion would put a model call in that queue
  for a label nobody asked to be refreshed.
* **The read cache must not key on mtime.** A session's JSON is rewritten
  on every appended message, so an mtime-keyed cache re-parses a
  multi-megabyte transcript on every 2-second dashboard poll — which is
  exactly the cost the cache exists to avoid.
"""

from __future__ import annotations

import json

import pytest

from app import session_titles as st


# ── Cleaning ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,want",
    [
        ("Setting up TTS with cloned voice", "Setting up TTS with cloned voice"),
        # Wrappers a model volunteers instead of doing the task.
        ('"Dashboard session titles"', "Dashboard session titles"),
        ("Title: guardian rollback bug", "Guardian rollback bug"),
        ("**Kv cache tuning**", "Kv cache tuning"),
        ("`harness stream stall`", "Harness stream stall"),
        ("## vLLM restart loop.", "vLLM restart loop"),
        ("- Fixing the model identity sweep", "Fixing the model identity sweep"),
        # Commentary after the title is dropped, not concatenated.
        ("Naming the session\nThis conversation covers…", "Naming the session"),
        # Refusals and non-answers are not titles.
        ("NONE", ""),
        ("trivial", ""),
        ("Untitled", ""),
        ("", ""),
        ("   ", ""),
    ],
)
def test_clean_title(raw, want):
    assert st.clean_title(raw) == want


def test_clean_title_preserves_internal_capitals():
    """Sentence-casing must not flatten a name the model got right."""
    assert st.clean_title("vLLM restart loop") == "vLLM restart loop"
    assert st.clean_title("iPhone sync failure") == "iPhone sync failure"
    assert st.clean_title("guardian policy audit") == "Guardian policy audit"


def test_clean_title_bounds_length_on_a_word_boundary():
    long = "Investigating the extremely verbose and rambling failure mode again"
    got = st.clean_title(long)
    assert len(got) <= st._MAX_CHARS
    # Truncation must not leave a half word.
    assert long.startswith(got)
    assert got.split()[-1] in long.split()


# ── Message accounting ─────────────────────────────────────────────────


def _session(*messages, **extra) -> dict:
    return {"session_id": "s1", "messages": list(messages), **extra}


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def test_injected_blocks_are_not_user_messages():
    """A session whose visible user turns are all prefetch would otherwise
    be titled after the prefetcher."""
    data = _session(
        _user("<context>vault hits…</context>"),
        _user("[autonomy:task-42] nightly sweep"),
        _user("<system-reminder>be nice</system-reminder>"),
        _user("actually fix the KV cache sizing"),
    )
    assert st.user_message_count(data) == 1
    transcript = st.build_transcript(data)
    assert "USER: actually fix the KV cache sizing" in transcript
    assert "<context>" not in transcript
    assert "[autonomy:" not in transcript


def test_transcript_takes_the_head_not_the_tail():
    """The subject is established early. Titling a long debugging session
    from its tail names it after whatever it happened to do last."""
    data = _session(
        *[m for i in range(20) for m in (_user(f"step {i}"), _assistant(f"ok {i}"))]
    )
    transcript = st.build_transcript(data)
    assert "USER: step 0" in transcript
    assert "step 19" not in transcript


def test_string_content_messages_are_read():
    """Session JSON carries both block-list and bare-string content."""
    data = _session({"role": "user", "content": "plain string turn"})
    assert st.user_message_count(data) == 1
    assert "plain string turn" in st.build_transcript(data)


# ── Re-title schedule ──────────────────────────────────────────────────


def test_untitled_session_with_one_user_message_is_due():
    assert st.should_title(_session(_user("set up the voice clone"))) is True


def test_empty_session_is_not_due():
    assert st.should_title(_session()) is False
    assert st.should_title(_session(_assistant("hello"))) is False


def test_autonomy_sessions_are_never_titled():
    data = _session(_user("run the sweep"), platform="autonomy")
    assert st.should_title(data) is False


def test_retitle_waits_for_geometric_growth():
    """Titled at 1 message: not due again until 3, then not until 9."""
    at_one = _session(*[_user(f"m{i}") for i in range(2)],
                      title="Voice clone setup", title_at_count=1)
    assert st.should_title(at_one) is False

    at_three = _session(*[_user(f"m{i}") for i in range(3)],
                        title="Voice clone setup", title_at_count=1)
    assert st.should_title(at_three) is True

    at_eight = _session(*[_user(f"m{i}") for i in range(8)],
                        title="Voice clone setup", title_at_count=3)
    assert st.should_title(at_eight) is False

    at_nine = _session(*[_user(f"m{i}") for i in range(9)],
                       title="Voice clone setup", title_at_count=3)
    assert st.should_title(at_nine) is True


def test_title_without_a_recorded_count_is_left_alone():
    """A title written by a build that didn't record `title_at_count` must
    anchor, not re-fire the model on every single turn forever."""
    data = _session(*[_user(f"m{i}") for i in range(30)], title="Old title")
    assert st.should_title(data) is False


# ── Generation ─────────────────────────────────────────────────────────


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path)
    st._cache.clear()
    st._in_flight.clear()
    yield tmp_path
    st._cache.clear()
    st._in_flight.clear()


def _write(sessions_dir, session_id: str, data: dict) -> None:
    (sessions_dir / f"{session_id}.json").write_text(json.dumps(data))


async def test_maybe_title_session_writes_title_and_anchor(sessions_dir, monkeypatch):
    _write(sessions_dir, "s1", _session(
        _user("we need to set up TTS with the cloned voice"),
        _assistant("Checking the repo for the voice pipeline."),
        session_id="s1",
    ))
    monkeypatch.setattr(
        "app.secondary_models._sync_secondary_title",
        lambda transcript, **kw: "Setting up TTS with cloned voice",
    )

    got = await st.maybe_title_session("s1")
    assert got == "Setting up TTS with cloned voice"

    on_disk = json.loads((sessions_dir / "s1.json").read_text())
    assert on_disk["title"] == "Setting up TTS with cloned voice"
    # The anchor is what stops the next turn re-firing the model.
    assert on_disk["title_at_count"] == 1
    assert on_disk["title_generated_at"]


async def test_unusable_model_output_leaves_the_session_untitled(sessions_dir, monkeypatch):
    """Better an untitled session than "Here is a title:" in the sidebar."""
    _write(sessions_dir, "s1", _session(
        _user("we need to set up TTS with the cloned voice"), session_id="s1",
    ))
    monkeypatch.setattr(
        "app.secondary_models._sync_secondary_title",
        lambda transcript, **kw: "NONE",
    )

    assert await st.maybe_title_session("s1") is None
    assert "title" not in json.loads((sessions_dir / "s1.json").read_text())


async def test_a_dead_secondary_does_not_raise(sessions_dir, monkeypatch):
    """This runs fire-and-forget off turn completion. An exception here
    must not surface as an unhandled task exception on every turn."""
    _write(sessions_dir, "s1", _session(
        _user("we need to set up TTS with the cloned voice"), session_id="s1",
    ))

    def _boom(transcript, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr("app.secondary_models._sync_secondary_title", _boom)
    assert await st.maybe_title_session("s1") is None


async def test_not_due_means_no_model_call(sessions_dir, monkeypatch):
    _write(sessions_dir, "s1", _session(
        _user("m0"), title="Already titled", title_at_count=1, session_id="s1",
    ))
    calls: list[str] = []
    monkeypatch.setattr(
        "app.secondary_models._sync_secondary_title",
        lambda transcript, **kw: calls.append(transcript) or "Nope",
    )

    assert await st.maybe_title_session("s1") is None
    assert calls == []


async def test_missing_session_is_a_no_op(sessions_dir):
    assert await st.maybe_title_session("does-not-exist") is None


# ── Read cache ─────────────────────────────────────────────────────────


def test_title_for_caches_across_a_rewritten_transcript(sessions_dir):
    """The session file is rewritten on every appended message. The cache
    must survive that, or the dashboard re-parses the whole transcript on
    every 2-second poll."""
    _write(sessions_dir, "s1", _session(_user("m0"), title="Voice clone setup"))
    assert st.title_for("s1") == "Voice clone setup"

    # Simulate a turn appending messages — same title, new file contents.
    _write(sessions_dir, "s1", _session(
        _user("m0"), _assistant("a" * 5000), title="Voice clone setup",
    ))
    (sessions_dir / "s1.json").write_text("{ not json at all")
    # Still served from cache rather than re-read.
    assert st.title_for("s1") == "Voice clone setup"


def test_invalidate_forces_a_reread(sessions_dir):
    _write(sessions_dir, "s1", _session(title="First"))
    assert st.title_for("s1") == "First"

    _write(sessions_dir, "s1", _session(title="Second"))
    assert st.title_for("s1") == "First"      # cached
    st.invalidate("s1")
    assert st.title_for("s1") == "Second"


def test_expired_entry_survives_a_torn_read(sessions_dir):
    """A half-written session file must not blank a panel that already had
    a good title — the next poll picks up the real one."""
    _write(sessions_dir, "s1", _session(title="Voice clone setup"))
    assert st.title_for("s1") == "Voice clone setup"

    # Age the cached entry past its TTL without waiting 30 real seconds.
    stamp, value = st._cache["s1"]
    st._cache["s1"] = (stamp - st._TITLE_TTL_S - 1, value)
    (sessions_dir / "s1.json").write_text("{ half-written")

    assert st.title_for("s1") == "Voice clone setup"


def test_unknown_session_reads_as_untitled(sessions_dir):
    """With nothing cached there is nothing to fall back to, and "" is the
    signal every consumer uses to render the preview or the id instead."""
    (sessions_dir / "s1.json").write_text("{ half-written")
    assert st.title_for("s1") == ""


def test_titles_for_missing_session_is_empty_not_an_error(sessions_dir):
    assert st.titles_for(["nope"]) == {"nope": ""}


# ── History ordering ───────────────────────────────────────────────────


def test_history_orders_on_last_active_not_file_mtime(tmp_path):
    """Writing a title rewrites the session file. If the history sorted on
    mtime, titling would silently promote whichever session was titled last
    — and a backfill, which touches every session at once, would reorder
    the entire list into backfill order."""
    import os

    from app.routers.sessions import _last_active_ts

    old = tmp_path / "old.json"
    old.write_text(json.dumps({"last_active": "2026-01-01T09:00:00"}))
    # Freshly rewritten on disk, as the titler leaves it.
    os.utime(old, None)

    recent = tmp_path / "recent.json"
    recent.write_text(json.dumps({"last_active": "2026-09-06T15:20:00"}))
    os.utime(recent, (0, 0))

    assert _last_active_ts(recent, json.loads(recent.read_text())) > \
        _last_active_ts(old, json.loads(old.read_text()))


def test_session_without_last_active_falls_back_to_mtime(tmp_path):
    """Sessions written before `last_active` existed still have to sort."""
    from app.routers.sessions import _last_active_ts

    path = tmp_path / "legacy.json"
    path.write_text("{}")
    assert _last_active_ts(path, {}) == pytest.approx(path.stat().st_mtime)
    assert _last_active_ts(path, {"last_active": "not a date"}) == \
        pytest.approx(path.stat().st_mtime)
