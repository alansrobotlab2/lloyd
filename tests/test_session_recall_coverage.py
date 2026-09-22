"""`session_recall` searches whole turns, not the head of a session.

The session index used to build one searchable string per session by clipping
every user message to 500 characters and every assistant message to 300, then
joining them and cutting the result at 5,000 characters. Scoring read that
string and nothing else, so a term appearing late in a long transcript — or
merely past the first 500 characters of a single message — was not in the
corpus at any point, and no query could reach it. Worse, a session that
*could* not be found returned nothing in exactly the shape a short session
that never mentioned the term would, which reads as "not discussed" rather
than "not indexed". The reported snippet came from the first eight user turns
only, so even a correctly found late match was illustrated with text that did
not contain the query term.

Each test below plants a marker in text the old triple of caps excluded, and
pins the one property the fix owes for that marker alone:

  * a marker in the FINAL user turn of a >20,000-character session — outside
    the 5,000-character corpus (`test_a_term_in_the_last_turn…`)
  * a marker past character 500 of a user message, or past character 300 of an
    assistant message, in a session short enough that the 5,000-character cap
    was never the reason it was missed (`test_a_term_past_the_old_per_message…`)
  * a marker in a user turn beyond the first eight, whose hit has to name the
    turn it came from (`test_a_hit_beyond_the_first_eight_user_turns…`)

Two fixtures — the long session and the late-user-turn one — run past the old
5,000-character cap, which is what they are testing. The clause-2 fixture keeps
itself under that cap and asserts that it does, so a per-message clip failure
cannot be excused by the cap; the assistant-turn fixture is under it too.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from agent_mcp import session as session_mod

TODAY = datetime.datetime.now().strftime("%Y%m%d")

# One token each (no punctuation: `re.findall(r"\w+")` would split it), and no
# word here is a stopword in `_ENTITY_STOPWORDS`.
MARKER_LAST_TURN = "zitherfennel42"
MARKER_PAST_USER_CLIP = "quorlumbridge9"
MARKER_PAST_ASSISTANT_CLIP = "vellumspindle3"
MARKER_LATE_USER_TURN = "cinnabarhold55"
MARKER_ASSISTANT_TURN = "pannuclockwork8"
MARKER_ABSENT = "umbrellathistle77"

# The caps the fix removes, restated as numbers the tests can fail against.
OLD_USER_CLIP = 500
OLD_ASSISTANT_CLIP = 300
OLD_CORPUS_MAX = 5000


def _turn(index: int, role: str, text: str) -> dict:
    return {
        "id": f"{role}{index}",
        "role": role,
        "content": [{"type": "text", "text": text}],
        "timestamp": "2026-09-20T12:00:00",
    }


def _text_of(messages: list[dict]) -> int:
    """Total user+assistant characters a session file carries."""
    return sum(len(m["content"][0]["text"]) for m in messages)


def _pad(words: int) -> str:
    """Deterministic marker-free filler, roughly 13 characters a word."""
    return " ".join(f"fillerword{n}" for n in range(words))


def _write(sessions_dir: Path, name: str, messages: list[dict], **extra) -> dict:
    data = {
        "session_id": name[:-5],
        "created_at": "2026-09-20T12:00:00",
        "model": "primary",
        "preview": "a session about filler words",
        "message_count": len(messages),
        "messages": messages,
    }
    data.update(extra)
    (sessions_dir / name).write_text(json.dumps(data), encoding="utf-8")
    return data


@pytest.fixture()
def indexed(tmp_path, monkeypatch):
    """Point the index at a scratch sessions dir and hand back the writer."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(session_mod, "_session_index_cache", None)
    return sessions


def _indexed_turn_texts(index_row: dict) -> list[str]:
    """Each indexed turn's own text, read back out of the row's corpus."""
    corpus = index_row["corpus"]
    return [corpus[start:end] for _i, _role, start, end in index_row["turns"]]


# ── clause 1: the corpus is not a head-truncated digest ──────────────────────

def test_a_term_in_the_last_turn_of_a_long_session_finds_that_session(indexed):
    """A term past the old 5,000-character corpus was unreachable.

    The fixture carries over 20,000 characters of user+assistant text with
    `MARKER_LAST_TURN` planted only in the final user turn, so the old corpus
    (`" ".join(...).lower()[:5000]`) could not contain it and `_score_session`,
    which reads that string and no file, scored the session zero. Without the
    fix this node fails: the recall returns 0 sessions.
    """
    messages: list[dict] = []
    for n in range(20):
        role = "user" if n % 2 == 0 else "assistant"
        messages.append(_turn(n, role, _pad(120)))
    last = len(messages)
    messages.append(_turn(last, "user", f"can you re-check the {MARKER_LAST_TURN} setting?"))
    _write(indexed, f"{TODAY}_120000_longone.json", messages)

    assert _text_of(messages) > 20_000, "the fixture must be longer than the old corpus cap"

    hits = session_mod._session_recall({"query": MARKER_LAST_TURN, "days": 14, "limit": 5})
    assert [s["session_id"] for s in hits["sessions"]] == [f"{TODAY}_120000_longone"]

    # Positive control: the same recall path still discriminates. A marker no
    # session carries must return nothing, so the assertion above is not
    # satisfied by a scorer that now matches everything.
    misses = session_mod._session_recall({"query": MARKER_ABSENT, "days": 14, "limit": 5})
    assert misses["sessions"] == []


def test_the_indexed_corpus_holds_every_turn_whole(indexed):
    """The per-session corpus is the whole indexed text, and offsets agree.

    `_session_recall` reports a hit's position as an offset into this string, so
    the offset is only meaningful if the string is complete and the rows point
    into it. This pins both: the corpus length equals the joined turn texts, and
    every turn row slices back to that turn's own lowercased text.
    """
    messages = [
        _turn(0, "user", _pad(90)),
        _turn(1, "assistant", _pad(90)),
        _turn(2, "user", "what happened to the indexer"),
        _turn(3, "assistant", "nothing at all"),
    ]
    _write(indexed, f"{TODAY}_130000_offsets.json", messages)

    row = session_mod._load_session_index(max_days=14)[f"{TODAY}_130000_offsets.json"]
    corpus = row["corpus"]

    assert len(corpus) >= _text_of(messages), (
        "the corpus must carry the whole indexed text, not a clip of it")
    turn_chars = sum(end - start for _i, _r, start, end in row["turns"])
    assert len(corpus) == turn_chars + len(row["turns"]) - 1, (
        "the corpus must be every turn joined whole, with nothing cut")

    texts = _indexed_turn_texts(row)
    assert len(texts) == 4
    assert texts[0] == _pad(90).lower()
    assert texts[2] == "what happened to the indexer"
    assert texts[3] == "nothing at all"
    assert [i for i, _r, _s, _e in row["turns"]] == [0, 1, 2, 3]


# ── clause 2: the per-message clips are gone ─────────────────────────────────

def test_a_term_past_the_old_per_message_clips_is_found(indexed):
    """Both per-message clips hid text inside sessions the corpus held whole.

    Two sessions, each one short enough that the old 5,000-character corpus cap
    was not the reason for the miss — the assertion in the loop below proves
    that, so the fix cannot be credited with a truncation it did not remove. In
    the first the marker sits past character 500 of a single user message (the
    old user clip); in the second, past character 300 of a single assistant
    message (the old assistant clip). Without the fix this node fails on the
    loop's corpus assertion — the marker is not in the session's indexed text at
    all — and the two recalls below would each return 0 sessions.
    """
    user_body = f"{_pad(80)} {MARKER_PAST_USER_CLIP} is the one I meant"
    assert len(user_body) > OLD_USER_CLIP
    messages_u = [_turn(0, "user", user_body), _turn(1, "assistant", "noted")]
    _write(indexed, f"{TODAY}_140000_userclip.json", messages_u)
    assert user_body.index(MARKER_PAST_USER_CLIP) > OLD_USER_CLIP

    asst_body = f"{_pad(50)} {MARKER_PAST_ASSISTANT_CLIP} covers that case"
    assert len(asst_body) > OLD_ASSISTANT_CLIP
    messages_a = [_turn(0, "user", "how does the indexer decide?"), _turn(1, "assistant", asst_body)]
    _write(indexed, f"{TODAY}_140100_asstclip.json", messages_a)
    assert asst_body.index(MARKER_PAST_ASSISTANT_CLIP) > OLD_ASSISTANT_CLIP

    index = session_mod._load_session_index(max_days=14)
    for name, messages, marker in ((f"{TODAY}_140000_userclip.json", messages_u, MARKER_PAST_USER_CLIP),
                                   (f"{TODAY}_140100_asstclip.json", messages_a, MARKER_PAST_ASSISTANT_CLIP)):
        assert _text_of(messages) < OLD_CORPUS_MAX, (
            f"{name} must stay under the old corpus cap, or the cap and the "
            "per-message clip stop being separable here")
        assert marker in index[name]["corpus"], name

    hits_u = session_mod._session_recall({"query": MARKER_PAST_USER_CLIP, "days": 14, "limit": 5})
    assert [s["session_id"] for s in hits_u["sessions"]] == [f"{TODAY}_140000_userclip"]

    hits_a = session_mod._session_recall({"query": MARKER_PAST_ASSISTANT_CLIP, "days": 14, "limit": 5})
    assert [s["session_id"] for s in hits_a["sessions"]] == [f"{TODAY}_140100_asstclip"]


# ── clause 3: a hit reports where in the transcript it came from ─────────────

def test_a_hit_beyond_the_first_eight_user_turns_reports_its_turn(indexed):
    """The result names the matched turn instead of falling back to the preview.

    The old snippet source was `[t[:300] for t in user_texts[:8]]`, so a match in
    the eleventh user turn could never show text containing the term: the result
    carried the session `preview` as its snippet, which says nothing about where
    the match was. This fixture plants the marker in the eleventh user turn,
    asserts the first eight user snippets cannot contain it — the assertion that
    stops the old fallback from satisfying the node — and then that the result
    both quotes the matched text and names its turn index and role.

    Stated because it is not isolated here: that turn also sits past the old
    5,000-character corpus, so without the fix this node fails at the recall
    itself, with 0 sessions returned, before it reaches the reporting.
    """
    messages: list[dict] = []
    for n in range(22):
        role = "user" if n % 2 == 0 else "assistant"
        messages.append(_turn(n, role, _pad(40)))
    late_user_turn = 20  # the eleventh user message (0,2,…,20)
    messages[late_user_turn] = _turn(
        late_user_turn, "user", f"bring the {MARKER_LATE_USER_TURN} back up please")
    _write(indexed, f"{TODAY}_150000_late.json", messages)

    row = session_mod._load_session_index(max_days=14)[f"{TODAY}_150000_late.json"]
    assert len(row["user_snippets"]) == 8
    assert not any(MARKER_LATE_USER_TURN in s for s in row["user_snippets"]), (
        "the fixture is only a test of the fallback if the first eight user "
        "snippets cannot contain the marker")

    hits = session_mod._session_recall({"query": MARKER_LATE_USER_TURN, "days": 14, "limit": 5})
    assert len(hits["sessions"]) == 1
    result = hits["sessions"][0]

    assert any(MARKER_LATE_USER_TURN in s.lower() for s in result["snippets"]), (
        "the snippet must contain the matched term, not the session preview")
    location = result["match_location"]
    assert location["turn_index"] == late_user_turn
    assert location["role"] == "user"
    assert location["char_offset"] == row["turns"][-2][2]


def test_a_hit_in_an_assistant_turn_reports_that_turn(indexed):
    """The location names whichever turn matched, not always a user turn.

    Same reporting path, the other role: the marker is only in an assistant
    message, so `role` has to come from the matched turn rather than being a
    constant, and `char_offset` has to be that turn's offset rather than the
    session's start.
    """
    messages = [
        _turn(0, "user", "and the second one?"),
        _turn(1, "assistant", _pad(40)),
        _turn(2, "assistant", f"the {MARKER_ASSISTANT_TURN} branch is the one that retries"),
    ]
    _write(indexed, f"{TODAY}_151000_assthit.json", messages)

    hits = session_mod._session_recall({"query": MARKER_ASSISTANT_TURN, "days": 14, "limit": 5})
    assert len(hits["sessions"]) == 1
    row = session_mod._load_session_index(max_days=14)[f"{TODAY}_151000_assthit.json"]
    location = hits["sessions"][0]["match_location"]
    assert location["role"] == "assistant"
    assert location["turn_index"] == 2
    assert location["char_offset"] == row["turns"][2][2]


# ── seam: the index prefetch scores is the same widened one ─────────────────

def test_prefetch_scores_and_renders_the_widened_index(indexed):
    """`prefetch` imports the index and the scorer, so it gets whole turns too.

    `_search_recent_sessions` builds the same index (`prefetch.py:842`) and
    scores it with the same `_score_session` (`prefetch.py:861`) to render the
    per-turn `<recent-sessions>` block. Nothing in `prefetch.py` changed, so this
    is the test that crosses that module boundary rather than a grep.

    The marker sits in the LAST turn of a session whose text runs past the old
    5,000-character corpus. Without the fix that session scores 0.0 on the term
    overlap the keyword branch ranks by, so it is not returned at all — the
    `long_row is not None` assertion below is what fails, with the four filler
    sessions in its place. After the fix it outranks them, which is what makes
    the ranking and the rendered line below deterministic rather than a property
    of directory order.
    """
    import prefetch

    long_messages: list[dict] = [_turn(0, "user", _pad(120))]
    for n in range(1, 24):
        long_messages.append(_turn(n, "user" if n % 2 else "assistant", _pad(120)))
    long_messages.append(_turn(24, "user", f"back to {MARKER_LAST_TURN} what did we decide"))
    assert sum(len(m["content"][0]["text"]) for m in long_messages) > OLD_CORPUS_MAX, (
        "the fixture must overrun the old corpus cap or the marker is indexable")
    _write(indexed, f"{TODAY}_160000_prefetchlong.json", long_messages,
           created_at="2026-09-20T09:12:00", message_count=len(long_messages))
    for s in range(4):
        _write(indexed, f"{TODAY}_16010{s}_other.json",
               [_turn(0, "user", _pad(6)), _turn(1, "assistant", _pad(6))])

    rows = prefetch._search_recent_sessions(f"what did we discuss about {MARKER_LAST_TURN} today")
    assert rows, "a temporal query must still return recent sessions"
    long_row = next((r for r in rows if r["filename"] == f"{TODAY}_160000_prefetchlong.json"), None)
    assert long_row is not None, "the session that answers the query must be returned"
    tokens = {MARKER_LAST_TURN}
    assert prefetch._score_session(long_row, tokens) > 0.0, (
        "the row prefetch scored must carry the marker: the whole-turn corpus "
        "is what this seam receives now")
    assert rows[0]["filename"] == f"{TODAY}_160000_prefetchlong.json"

    block = prefetch._format_context([], [], session_results=rows)
    assert "<recent-sessions>" in block
    assert "2026-09-20T09:12" in block
