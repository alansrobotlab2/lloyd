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
    # right-hand chat sidebar: `<ts>_iv<4 hex>`, three parts. Pinned against the
    # running endpoint by
    # `test_the_create_endpoint_still_mints_a_three_part_id` below, and its
    # written file by tests/test_api_contracts.py.
    create = (ROOT / "app" / "routers" / "sessions.py").read_text()
    assert 'suffix = "iv" + secrets.token_hex(2)' in create
    assert 'session_id = f"{ts}_{suffix}"' in create

    # And the one mint that is supposed to be four parts, is.
    assert is_background_session_name(new_background_session_id("autocode"))
    assert is_background_session_name(new_background_session_id("autonomy"))


async def test_the_create_endpoint_still_mints_a_three_part_id(tmp_path,
                                                              monkeypatch):
    """The shape rule, checked against the endpoint rather than its source text.

    The scan above pins the literal; this one crosses the HTTP boundary, because
    a mint that only looks right in the file is the failure that matters —
    `is_background_session_name` is decided from the *filename* before the JSON
    is opened, so a four-part id minted here would hide a live user conversation
    from `/api/sessions` entirely. Written file, response body, the name rule and
    the listing all have to agree on the same id.
    """
    import json

    import httpx

    import server
    from app.routers import sessions as sessions_router
    from app.sessions_io import is_background_session_name

    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 9999))
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://lloyd-test") as client:
        created = await client.post("/api/sessions/create",
                                    json={"inner_voice": True})
        assert created.status_code == 200, created.text
        body = created.json()
        listed = {s["id"] for s in
                  (await client.get("/api/sessions")).json()["sessions"]}

    session_id = body["session_key"]
    assert session_id == body["session_id"]
    assert re.fullmatch(r"\d{8}_\d{6}_iv[0-9a-f]{4}", session_id), session_id
    assert not is_background_session_name(session_id + ".json")
    assert session_id in listed, "a chat created here vanished from the history"
    written = json.loads((tmp_path / f"{session_id}.json").read_text())
    assert written["session_id"] == written["id"] == session_id
    assert written["platform"] == "mission-control", (
        "a stub that is not a user session would be listed by name and then "
        "classified away by its platform")


# ── One writer, one write path ───────────────────────────────────────────

#: The router that serves `/api/sessions*`, and the two things a second writer
#: needs in order to exist: a write of its own, and no call to the helper.
_SESSIONS_ROUTER = ROOT / "app" / "routers" / "sessions.py"

#: Every way this module could put bytes on disk. `write_text`/`write_bytes`
#: truncate in place, `atomic_write_text` is the one allowed form and lives in
#: `sessions_io`, and an `open(..., "w")` is the same truncate with more steps.
_WRITE_CALL = re.compile(
    r"\.(?:write_text|write_bytes|atomic_write_text)\s*\("
    r"|\bopen\([^)]*[\"'][wax]")


def test_the_sessions_router_writes_no_session_file_itself():
    """#1275 clause 1: `POST /api/sessions/create` creates through `sessions_io`.

    Until 2026-09-20 that endpoint built its own dict and wrote it with a bare
    `write_text`, so the sessions it minted — ~150 of them on 2026-09-19 —
    carried no `id` and no `source`, while `sessions_io.create_session`, the
    helper whose own docstring opens "one writer for every session", guaranteed
    both. A second writer leaves no other trace: the file simply looks slightly
    different, and each reader that trusts a guaranteed key misjudges those
    sessions quietly. Hence a source scan — it fails at the shape, not one poll
    after a torn write.

    The scan is whole-module on purpose. `app/routers/sessions.py` answers
    questions about sessions and delegates every write; a write appearing here
    at all is the thing to notice, and there were none before the endpoint was
    added. Creation must still be reachable: the module has to name the helper.
    """
    source = _SESSIONS_ROUTER.read_text(encoding="utf-8")
    offenders = [f"{_SESSIONS_ROUTER.relative_to(ROOT)}:{i}"
                 for i, line in enumerate(source.splitlines(), 1)
                 if _WRITE_CALL.search(line)]
    assert not offenders, (
        "the sessions router writes a file itself; create through "
        "sessions_io.create_session so one field set and one atomic write "
        "decide what a session JSON holds: " + ", ".join(offenders))
    assert "sessions_io.create_session(" in source, (
        "nothing in this module creates a session through the shared helper")


# ── The one prose copy of the rule ─────────────────────────────────────

#: The endpoint that answers "which session is the user looking at?" for every
#: ambient producer (`agent_mcp/ambient.py` resolves through it).
_ACTIVE_ROUTE = "app/routers/sessions.py"


def _active_endpoint_doc():
    from app.routers.sessions import get_active_session_endpoint

    return get_active_session_endpoint.__doc__ or ""


def _flat(text: str) -> str:
    """Collapse runs of whitespace, so the prose is pinned but its wrapping is
    not — re-indenting a docstring is not a finding."""
    return " ".join(text.split())


def test_the_active_session_doc_states_the_deny_list_not_an_allow_list():
    """`/api/sessions/active` is the text an operator reads when a producer
    reports "no active user session", and the text an editor edits against.

    Its docstring described the pre-2026-09-07 rule — resolve to the most
    recent `platform: mission-control` session, exclude only `platform:
    autonomy` — while `app.sessions_io` had been a deny-list since `1a7d50e`
    excluded `worker` as well. The contradiction outlived the change because
    the commit that introduced the deny-list added a 409 guard to the inject
    endpoint and never touched this prose, and it had already been extracted
    verbatim into the knowledge graph. Editing to the doc would re-narrow the
    rule to one client, and the behaviour tests would only catch it if they
    were edited too. So what the docstring may NOT contain is pinned, in the
    file that already owns "one definition of who is reading this session".
    """
    doc = _active_endpoint_doc()
    source = (ROOT / _ACTIVE_ROUTE).read_text(encoding="utf-8")

    # Clause 1: no allow-list is reconstructable — from the docstring, and
    # from anywhere else in the module.
    assert "platform: mission-control" not in source
    assert "Explicitly excludes" not in source
    assert "mission-control" not in doc, (
        "the endpoint docstring names a platform again; it defers to "
        "`is_user_session` / `NON_USER_PLATFORMS` instead"
    )


def test_the_active_session_doc_defers_to_the_one_definition():
    """Clause 2: the prose says where the decision lives and that BOTH rules
    run it — the in-memory hint and the mtime scan — so neither reads as an
    unconditional shortcut to the last session that touched."""
    doc = _flat(_active_endpoint_doc())
    # Identifiers are matched as written; prose is matched case-insensitively,
    # because "Both rules apply" and "both rules apply" are the same sentence.
    low = doc.lower()

    assert "is_user_session" in doc
    assert "NON_USER_PLATFORMS" in doc
    assert "both rules" in low
    assert "in-memory" in low and "mtime" in low
    # Both excluded platforms are named, so a reader who only skims the
    # endpoint learns that `worker` is not the user either — that is the half
    # the old prose got wrong and the half the 2026-09-07 brief missed.
    assert "worker" in low and "autonomy" in low
    # And the direction of the list, because "excluded platforms" alone reads
    # the same as an allow-list to anyone typing the next edit.
    assert "deny-list" in low
    assert "eligible" in low


def test_the_active_session_doc_keeps_the_operational_facts():
    """Clause 3: what a producer actually needs from this text — how stale is
    too stale, and what a null answer means. Rewriting prose about platforms
    must not drop them with the platform names."""
    doc = _flat(_active_endpoint_doc())
    low = doc.lower()

    assert "24h" in low, "the recency window on rule 2 went missing"
    assert '{"session_id": null}' in low
    assert "skip injection" in low
