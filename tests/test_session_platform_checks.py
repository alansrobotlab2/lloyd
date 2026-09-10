"""One definition of "is a human reading this session".

Six readers hand-rolled `data.get("platform") == "autonomy"` and not one of
them had learned about `worker`. Worker turns arrive through the chat path, so
until 2026-09-10 the sessions the machine ran for itself were:

  * listed in the user's chat history (`/api/sessions`),
  * counted as recent chats on Mission Control's landing tab,
  * titled by the LLM titler on the single-tenant secondary,
  * exported into the daily note as the user's day,
  * fact-extracted into the knowledge graph as things the user said,
  * and recalled back to the model by `session_recall` as the user's own
    conversations.

`sessions_io.is_user_session` is the one definition. The literal is what this
file greps for, because a seventh reader written next month is exactly how
this comes back — and it comes back silently, since a background session that
leaks into the history looks like a session.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the definition lives, and the two places allowed to restate it.
_DEFINITION = ROOT / "app" / "sessions_io.py"
#: The retention sweep is stdlib-only and runs from cron with no venv, so it
#: cannot import the definition. It restates the tuple and this file pins that
#: the two agree, which is the whole point of allowing the exception.
_RESTATES = {ROOT / "scripts" / "groundskeeper" / "retention-sweep.py"}

_LITERAL = re.compile(r'platform["\']?\s*\)?\s*==\s*["\']autonomy["\']')


def _sources():
    for pkg in ("app", "agent_mcp", "workers"):
        for path in (ROOT / pkg).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_nobody_hand_rolls_the_autonomy_platform_check():
    offenders = []
    for path in _sources():
        if path == _DEFINITION or path in _RESTATES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if _LITERAL.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{i}")
    assert not offenders, (
        "hand-rolled platform checks — use `sessions_io.is_user_session` so a "
        "new non-user platform is excluded everywhere at once: "
        + ", ".join(offenders)
    )


def test_the_definition_covers_both_background_platforms():
    from app.sessions_io import NON_USER_PLATFORMS, is_user_session

    assert NON_USER_PLATFORMS == frozenset({"autonomy", "worker"})
    assert not is_user_session({"platform": "autonomy"})
    assert not is_user_session({"platform": "worker"})
    # A missing platform is the web UI. Every session that predates the field
    # reads that way, and so does a chat session mid-creation.
    assert is_user_session({})
    assert is_user_session({"platform": "mission-control"})
    # A platform this list has never heard of is a USER session, deliberately:
    # a deny-list, so a new client keeps receiving its briefs rather than
    # silently losing them.
    assert is_user_session({"platform": "some-future-client"})


def test_the_retention_sweep_agrees_with_the_definition():
    """It cannot import — stdlib-only, no venv under cron — so it restates.
    A restatement that drifts would archive real conversations on the
    background schedule, or keep background runs for three months."""
    from app.sessions_io import NON_USER_PLATFORMS

    text = (ROOT / "scripts" / "groundskeeper" / "retention-sweep.py").read_text()
    m = re.search(r"NON_USER_PLATFORMS\s*=\s*\((.*?)\)", text, re.DOTALL)
    assert m, "retention-sweep.py no longer restates NON_USER_PLATFORMS"
    assert set(re.findall(r'"([a-z_-]+)"', m.group(1))) == set(NON_USER_PLATFORMS)


def test_every_reader_that_used_to_hand_roll_it_now_calls_the_helper():
    """Named individually, because "the grep is clean" is also true of a file
    that dropped the check altogether."""
    for rel in ("app/routers/sessions.py", "app/routers/dashboard.py",
                "app/session_titles.py", "agent_mcp/session.py",
                "app/post_capture.py", "app/routers/mc_ui.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "is_user_session" in text, f"{rel} no longer filters at all"


# ── The filename fast path ─────────────────────────────────────────────

def test_a_background_id_is_recognisable_without_opening_the_file():
    from app.sessions_io import is_background_session_name

    assert is_background_session_name("20260910_120001_autocode_9f2a.json")
    assert is_background_session_name("20260910_120001_autonomy_9f2a")
    # A chat id has three parts.
    assert not is_background_session_name("20260910_120001_9f2a1c.json")
    # And anything this rule does not recognise is read as a chat and then
    # classified by its `platform` — one wasted read. The opposite error would
    # hide a real conversation from the history list.
    assert not is_background_session_name("notes")
    assert not is_background_session_name("bench_baseline_1788_x_y")


def test_no_user_session_creator_mints_a_background_shaped_id():
    """The chat listing and the recent-chats panel skip a four-part id WITHOUT
    opening it, so for that shape the name is the whole decision, not a hint.
    That is safe only while nothing that creates a user session mints one, so
    the creators are pinned here rather than trusted. A future producer that
    named a user session in four parts would vanish from the history, and
    nothing would say so.
    """
    import uuid
    from datetime import datetime
    from app.sessions_io import (is_background_session_name,
                                 new_background_session_id)

    # Every `session_id = f"..."` mint in the packages this file sweeps. Three
    # today: two in the chat router, one in POST /api/sessions/create. A fourth is a
    # new creator, and it has to be looked at before it is allowed.
    mints = []
    for path in _sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r'session_id\s*=\s*f["\']', line):
                mints.append(f"{path.relative_to(ROOT)}:{i}")
    assert len(mints) == 3, mints

    # The chat path: `<ts>_<6 hex>`, three parts.
    router = (ROOT / "app" / "routers" / "messages.py").read_text()
    chat_mint = ('session_id = f"{datetime.now():%Y%m%d_%H%M%S}_'
                 '{uuid.uuid4().hex[:6]}"')
    assert router.count(chat_mint) == 2
    assert not is_background_session_name(
        f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}")

    # POST /api/sessions/create — the Inner Voice "+ new chat" button and the
    # right-hand chat sidebar: `<ts>_iv<4 hex>`, three parts. Pinned end to
    # end in tests/test_api_contracts.py.
    create = (ROOT / "app" / "routers" / "sessions.py").read_text()
    assert 'suffix = "iv" + secrets.token_hex(2)' in create
    assert 'session_id = f"{ts}_{suffix}"' in create

    # And the one mint that is supposed to be four parts, is.
    assert is_background_session_name(new_background_session_id("autocode"))
    assert is_background_session_name(new_background_session_id("autonomy"))
