"""One definition of "is a human reading this session" — and two answers.

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

`sessions_io.is_user_session` fixed that, and #1064 found the second question it
had been quietly answering as well. `is_user_session` is a DENY-list, because a
client it has never heard of must keep receiving its ambient briefs rather than
silently losing them. But the same bool also decided whether a session is
*listed as a conversation* and *exported into the corpus qmd embeds* — and for
that question the generous default is the bug. The three `e2e-harness` automod
smokes and the gate's `canary` turn are not brief-delivery problems (nobody ever
meant to brief them) and they are not conversations; a deny-list could only ever
call them conversations, because their platform is simply absent from it.

So there are now two predicates, and this file pins that they stay apart:

  * `is_user_session(data)` / `NON_USER_PLATFORMS` — may this session receive a
    brief, and may it be "the user's session". Deny-list. Unchanged.
  * `is_conversation_session(name, data)` / `INTERACTIVE_PLATFORMS` — is this a
    conversation a human will read, and therefore listed and embedded.
    Allow-list, and it refuses a four-part id whatever its platform says.

The literal is what this file greps for, because a seventh reader written next
month is exactly how the old shape comes back — and it comes back silently,
since a background session that leaks into the history looks like a session.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Where the definition lives, and the two places allowed to restate it.
_DEFINITION = ROOT / "app" / "sessions_io.py"
#: The retention sweep is stdlib-only and runs from cron with no venv, so it
#: cannot import the definition. It restates the tuple and this file pins that
#: the two agree, which is the whole point of allowing the exception.
_RESTATES = {ROOT / "scripts" / "groundskeeper" / "retention-sweep.py"}

_LITERAL = re.compile(r'platform["\']?\s*\)?\s*==\s*["\']autonomy["\']')

#: Every package the rule reaches. `scripts/` joined with #1064: the corpus
#: backfill was reader number seven all along — it had its own
#: `platform == "autonomy"`, wrote every `worker` transcript it saw into the
#: collection qmd EMBEDS (469 of them on 2026-09-17), and sat outside the swept
#: roots, so the guard that exists for exactly this class could not see it.
#: Roots swept for a hand-rolled platform check. `scripts/` is here because of
#: the offender that made this item true —
#: `scripts/memory/backfill-session-markdown.py`, which sat outside the old
#: three roots and so sat outside the one guard written for its own class.
_SWEEP_ROOTS = ("app", "agent_mcp", "workers", "scripts")

#: Roots swept for id MINTS. Deliberately narrower: `scripts/` also holds
#: legitimate *machine* minters (`automod/review.py` writes a four-part review
#: transcript stamped `worker`), and a regex over a string literal cannot see the
#: platform a file stamps four lines later. Those are pinned by name in
#: `test_a_scripts_minter_of_a_background_shape_names_a_machine_platform`.
_MINT_ROOTS = ("app", "agent_mcp", "workers")


def _sources(roots=_SWEEP_ROOTS, base=ROOT):
    """Every `.py` under `roots`, walked from `base`.

    `base` is a parameter so the positive-control test can run THIS walker —
    roots, skip rules and all — against a planted tree instead of writing the plant
    into the checkout, which pytest under 8 xdist workers makes visible to every
    sibling test that sweeps `scripts/`. A control that swept a tree hand-built in
    the test would prove nothing about the one that guards the real code, so this
    is the same generator, not a copy of its logic.
    """
    for pkg in roots:
        for path in (base / pkg).rglob("*.py"):
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


def test_every_reader_that_used_to_hand_roll_it_now_calls_a_helper():
    """Named individually, because a file that dropped the check altogether also
    passes a clean grep. Since #1064 the six readers split across two
    predicates, so the requirement is "calls one of them", not "calls the old
    one" — which one each file must call is pinned per-file below.
    """
    for rel in ("app/routers/sessions.py", "app/routers/dashboard.py",
                "app/session_titles.py", "agent_mcp/session.py",
                "app/post_capture.py", "app/routers/mc_ui.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert ("is_user_session" in text
                or "is_conversation_session" in text), f"{rel} no longer filters at all"


def test_the_chat_readers_name_the_allow_list_predicate_and_the_others_do_not():
    """Which predicate each reader calls, per file.

    "Some helper is called" would still be true with every reader back on the
    delivery deny-list — the state that leaked 3 `e2e-harness` automod smokes
    (93, 203 and 134 messages) into the chat history AND into the corpus qmd
    embeds. Four readers decide what a human sees or which corpus a transcript
    lands in; those must name `is_conversation_session`. The titler and the
    recall tool answer the delivery question and must NOT have been moved,
    because a background run still gets titled and still must not be recalled as
    the user's own words.
    """
    for rel in ("app/routers/sessions.py", "app/routers/dashboard.py",
                "app/routers/mc_ui.py", "app/post_capture.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "is_conversation_session(" in text, (
            f"{rel} decides chat membership or corpus placement with the "
            f"delivery deny-list again — the leak #1064 is about")
    for rel in ("app/session_titles.py", "agent_mcp/session.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "is_user_session(" in text, (
            f"{rel} answers the delivery question with the listing predicate: a "
            f"background run keeps its title, and is still never 'the user's session'")


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
    for path in _sources(_MINT_ROOTS):
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


# ── Two questions, two predicates (#1064) ───────────────────────────────────
#
# `is_user_session` answers "may this session receive a brief", and its generous
# default is deliberate. The readers below answer "is this a conversation a human
# will read" — for which that same default IS the defect, because a listed
# session is also exported into the corpus qmd embeds and later recalled as
# something the user discussed. A deny-list cannot answer the second question at
# all: an unclassifiable platform is exactly what a new client invents, and a
# deny-list files it under "human". So `is_conversation_session` is an
# allow-list, and it also refuses a four-part id whatever its platform says.

def test_there_are_two_lists_and_the_new_one_is_an_allow_list():
    from app import sessions_io

    assert sessions_io.INTERACTIVE_PLATFORMS == frozenset(
        {"mission-control", "browser"}
    ), ("the platforms that name a surface a person reads their own conversation "
        "on: the web UI, and the Chrome side panel whose transcripts carry "
        "browser_metadata and 2-375 messages each")
    # Not complements, and the gap between them is the whole point. `e2e-harness`
    # (3 automod smokes, 93/203/134 messages) and `canary` (the gate's one
    # synthetic turn) sit in NEITHER list on disk today.
    assert not (sessions_io.INTERACTIVE_PLATFORMS & sessions_io.NON_USER_PLATFORMS)
    for unclassified in ("e2e-harness", "canary", "a-client-nobody-has-named"):
        assert sessions_io.is_user_session({"platform": unclassified}) is True, (
            f"{unclassified} must stay eligible for briefs — replacing the "
            f"deny-list with the allow-list would cut a new client off silently")
        assert sessions_io.is_conversation_session(
            "20260921_101112_abc123", {"platform": unclassified}) is False, (
            f"{unclassified} must not be filed as a human conversation by "
            f"default; widening INTERACTIVE_PLATFORMS is the way to say it is")


def test_a_four_part_id_is_never_a_conversation_whatever_its_platform():
    from app.sessions_io import is_conversation_session, is_user_session

    orphan = {"platform": "mission-control"}
    assert is_user_session(orphan) is True
    # The two predicates MUST disagree on this shape. 12 sessions on disk are
    # exactly it; the chat listings skip a four-part name unread, so a parsed
    # path that called them conversations would leave them in no listing at all.
    assert is_conversation_session("20260917_105134_autocode_029c", orphan) is False
    assert is_conversation_session("20260917_105134_029c", orphan) is True
    # A chat-shaped name with no platform is the pre-field web UI and stays a
    # conversation, or history that predates the field vanishes from the tab.
    assert is_conversation_session("20260916_135908_abc123", {}) is True
    # ...while a background-shaped name with no platform is a machine run.
    assert is_conversation_session("20260916_135908_autotriage_3674", {}) is False


@pytest.mark.anyio
async def test_the_two_listings_partition_the_directory(tmp_path, monkeypatch):
    """Every session lands in exactly one listing: never both, never neither.

    Through the HTTP boundary, because the two halves are written separately
    there — the chat listing skips a four-part name UNREAD, the background
    listing then dropped anything user-labelled — and a four-part id stamped
    `mission-control` fell into the seam. 12 of them were on disk when triage
    counted, newest `20260917_105134_autocode_029c`, and no single-endpoint unit
    test could see the gap.
    """
    import httpx

    import server
    from app.routers import sessions as sessions_router

    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)

    def _write(sid: str, platform: str) -> str:
        body = {"session_id": sid, "title": sid, "messages": []}
        if platform:
            body["platform"] = platform
        (tmp_path / f"{sid}.json").write_text(json.dumps(body), encoding="utf-8")
        return sid

    humans = [
        _write("20260916_135908_abc123", "mission-control"),
        _write("20260916_140210_iv9f8e", "browser"),
        _write("20260916_140311_77aa55", ""),        # predates the platform field
    ]
    machines = [
        _write("20260916_140412_def456", "worker"),
        _write("e2e_selfmod_1788756141", "e2e-harness"),
        _write("canary_1757526350_a1b2c3", "canary"),
        _write("20260916_150000_autonomy_ab12", "autonomy"),
        _write("20260917_105134_autocode_029c", "mission-control"),   # the orphan
        _write("20260917_104051_autocode_f9fe", ""),                  # orphan, no platform
    ]

    app = server.app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        chats = {s["id"] for s in (await client.get("/api/sessions")).json()["sessions"]}
        bg = {s["id"] for s in
              (await client.get("/api/background/sessions")).json()["sessions"]}

    assert chats == set(humans), (
        f"chat listing leaked {sorted(chats - set(humans))} and lost "
        f"{sorted(set(humans) - chats)}: the e2e-harness and canary runs must "
        f"leave it, while the 144 mission-control and 54 browser rows on disk "
        f"today (2026-09-21) must all stay")
    assert set(machines) <= bg, f"machine runs in no listing: {sorted(set(machines) - bg)}"
    assert not (chats & bg), (
        f"in both listings: {sorted(chats & bg)} — a session in both is the same "
        f"defect as one in neither, seen from the other side")


def test_the_landing_panel_and_both_tab_summaries_apply_the_same_predicate(tmp_path, monkeypatch):
    """Three more readers of the same directory, each with its own old opinion.

    Called directly rather than over HTTP (the dashboard panel needs a dozen
    unrelated sections to serialise), which still catches the defect — the defect
    was always which predicate each file chose. `mc_ui._summarize_background` is
    the one that mattered most: it skipped every four-part name unread AND then
    dropped anything user-labelled, so the orphan sessions were invisible to the
    model's own picture of the Background tab as well as to the tab itself.
    """
    from app.routers import dashboard as dash
    from app.routers import mc_ui

    monkeypatch.setattr("app.paths.SESSIONS_DIR", tmp_path)   # dash imports it per call
    monkeypatch.setattr(mc_ui, "SESSIONS_DIR", tmp_path)

    def _write(sid: str, platform: str):
        body = {"session_id": sid, "title": sid, "messages": []}
        if platform:
            body["platform"] = platform
        (tmp_path / f"{sid}.json").write_text(json.dumps(body), encoding="utf-8")

    _write("20260916_135908_abc123", "mission-control")
    _write("20260916_140210_iv9f8e", "browser")
    _write("e2e_selfmod_1788756141", "e2e-harness")
    _write("20260917_105134_autocode_029c", "mission-control")   # the orphan
    human_ids = {"20260916_135908_abc123", "20260916_140210_iv9f8e"}

    recent = {r["session_id"] for r in dash._scan_recent_sessions()}
    assert recent == human_ids, f"the landing tab's recent chats said {sorted(recent)}"

    chats = {s["id"] for s in mc_ui._summarize_chat()["recent_sessions"]}
    bgs = {s["id"] for s in mc_ui._summarize_background()["recent"]}
    assert chats == human_ids, f"the model was told the Chat tab holds {sorted(chats)}"
    assert bgs == {"e2e_selfmod_1788756141", "20260917_105134_autocode_029c"}, (
        f"the model was told the Background tab holds {sorted(bgs)}; the orphan "
        f"belongs there and was previously counted as neither chat nor background")


def test_the_markdown_export_chooses_the_corpus_the_platform_earns(tmp_path, monkeypatch):
    """Which corpus qmd embeds is decided by the allow-list, not the deny-list.

    `agent-services/scripts/qmd-watcher.sh` watches `_pipeline/vault-derived/
    sessions` and feeds it to `qmd update`/`embed`, so anything written there is
    later recalled as something the user discussed. Before #1064 the export
    asked the delivery question, which put the 3 `e2e-harness` automod smokes —
    ~430 messages of round transcripts — and every four-part `mission-control`
    orphan into the embedded corpus.
    """
    from app import post_capture

    chat_corpus = tmp_path / "sessions"
    bg_corpus = tmp_path / "sessions-background"
    monkeypatch.setattr(post_capture, "VAULT_SESSIONS_DIR", chat_corpus)
    monkeypatch.setattr(post_capture, "VAULT_BACKGROUND_SESSIONS_DIR", bg_corpus)

    def _data(sid, platform):
        return {
            "session_id": sid, "platform": platform,
            "created_at": "2026-09-21T00:00:00",
            "messages": [
                {"role": "user", "content": "what did the review rung say about it?"},
                {"role": "assistant", "content": "it refused the commit and named the clause."},
            ],
        }

    human = post_capture._export_session_markdown(
        "20260921_101112_abc123", _data("20260921_101112_abc123", "mission-control"))
    smoke = post_capture._export_session_markdown(
        "e2e_selfmod_1788756141", _data("e2e_selfmod_1788756141", "e2e-harness"))
    orphan = post_capture._export_session_markdown(
        "20260917_105134_autocode_029c", _data("20260917_105134_autocode_029c", "mission-control"))

    assert human is not None and str(human).startswith(str(chat_corpus)), (
        f"a mission-control chat must stay in the embedded corpus, got {human}")
    for path, what in ((smoke, "an e2e-harness smoke"), (orphan, "a four-part orphan")):
        assert path is not None and str(path).startswith(str(bg_corpus)), (
            f"{what} was written into the corpus qmd embeds at {path}")


@pytest.mark.anyio
async def test_a_background_shaped_turn_on_the_chat_path_is_not_stamped_a_chat(tmp_path, monkeypatch):
    """The lazy create is the minter, and the old guard could not see it.

    `tests/test_uptake.py` pins that `_save_session_meta` defaults to
    `mission-control` for a chat-shaped id — correct, and the reason: the web UI
    posts a message before `POST /api/sessions` has run. Reached with a
    four-part id the same line minted an invisible session instead; three of the
    12 orphans were minted that way on 2026-09-17 alone, so the leak was live the
    day triage measured it. The endpoint's `platform` now threads down here, and
    the four-part shape overrides even an interactive-looking value.
    """
    from app import sessions_io

    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", tmp_path)

    await sessions_io._save_session_meta(
        "20260921_101112_autocode_abcd", "primary", "hi", platform="mission-control")
    await sessions_io._save_session_meta(
        "20260921_101113_ef5678", "primary", "hi", platform="mission-control")
    await sessions_io._save_session_meta(
        "20260921_101114_autotriage_99aa", "primary", "hi")

    stamped = {p.stem: json.loads(p.read_text(encoding="utf-8")).get("platform")
               for p in tmp_path.glob("*.json")}

    assert stamped["20260921_101113_ef5678"] == "mission-control", (
        "a three-part lazy create must stay a chat — test_uptake.py pins that default")
    for four_part in ("20260921_101112_autocode_abcd", "20260921_101114_autotriage_99aa"):
        assert stamped[four_part] != "mission-control", (
            f"{four_part} was stamped mission-control by the lazy create, which is "
            f"how a live run becomes invisible to both listings")
        assert sessions_io.is_user_session({"platform": stamped[four_part]}) is False


#: What the worker's loopback POST must carry, and what the backend must file it
#: as. Kept a tuple so the two assertions below cannot drift apart from each other.
WORKER_SEAM_PLATFORM = "worker"


@pytest.mark.anyio
async def test_the_worker_s_loopback_post_arrives_with_its_platform(tmp_path, monkeypatch):
    """The worker → backend hop is a process boundary, and this pins the wire.

    `workers/sources/_common.py:729` builds a JSON body, POSTs it to
    `127.0.0.1:8080/api/message/stream`, and `app/routers/messages.py` reads
    `data.get("platform")` out of that body and hands it to
    `sessions_io._save_session_meta`. The earlier commit at this item added the
    field and a test that `_save_session_meta` honours it, and the review rung
    refused both: a test that calls the sink with a value is not a test that the
    source sends one, so the field could be dropped from the payload, renamed on
    either side, or stripped by a future body-builder and every test here would
    stay green while the orphan class came back.

    So both real sides run, across a real serialization. `run_prompt_in_session`
    executes unpatched — its payload build, `new_worker_session`, the httpx stream
    and the SSE parser — and only its transport is swapped, onto the ASGI app, so
    the body is encoded to bytes, decoded by FastAPI, and validated by the endpoint
    exactly as over loopback. The backend then does its real lazy create. What is
    patched is the agent turn and the machine-level writes it would perform; see
    the notes at each patch.

    And the fault is reproduced, not assumed: `create_session` is wrapped so the
    file it writes disappears before the POST, which is the exact state the payload
    exists for — a worker whose session file is gone by the time the turn arrives.
    Without that, the endpoint finds a file, copies its platform forward, and never
    exercises either branch.

    Two legs, because the field is only half-visible on either side alone. For a
    four-part id the endpoint classifies by SHAPE — the create branch stamps
    `worker` whatever the body says — so the file alone cannot show the body
    arriving; leg 1 reads the serialised bytes and catches the field going missing
    at the source. For a chat-shaped id there is no shape to fall back on and the
    body is the only evidence, so leg 2 replays the same bytes under a chat-shaped
    id and catches the endpoint stop forwarding it. Together they cover both ends of
    the hop; either alone leaves one side free to regress.
    """
    import httpx

    import app.config
    import server
    import workers.sources._common as common
    from app import sessions_io
    from app.routers import automod as automod
    from app.routers import messages as messages

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", sessions)

    # Reproduce the fault: the id is minted, its file is not there when the POST
    # lands. A collaborator's side effect is removed; no code under test is patched.
    real_create = sessions_io.create_session

    def _mint_without_leaving_a_file(*a, **k):
        sid = real_create(*a, **k)
        (sessions / f"{sid}.json").unlink(missing_ok=True)
        return sid

    monkeypatch.setattr(sessions_io, "create_session", _mint_without_leaving_a_file)

    sent: list[dict] = []

    class _Recording(httpx.AsyncBaseTransport):
        """Delegates to the ASGI app and keeps the bytes that crossed it.

        The recorded value is `json.loads(request.content)`: the payload after the
        worker serialised it, which is the only thing the backend can read. Reading
        `payload` off a patched-in-memory object instead would let a field be
        dropped by the serialiser, renamed by a future `json=`, or stripped by a
        body-builder, and this test would stay green while the orphan class came
        back — which is exactly the gap the review rung named.
        """

        def __init__(self, inner):
            self._inner = inner

        async def handle_async_request(self, request):
            if request.url.path.endswith("/message/stream"):
                sent.append(json.loads(request.content))
            return await self._inner.handle_async_request(request)

    class _AsgiOnlyClient(httpx.AsyncClient):
        """Every client in this test talks to the ASGI app, never to a port.

        Patched globally rather than on the caller because the backend opens its
        own clients while the request is being served — including the model engine
        call the aborted turn attempts — and an engine call must fail loudly
        instead of reaching a GPU. That is also what keeps this test honest as a
        process-boundary test: nothing on either side can shortcut the wire.
        """

        def __init__(self, *a, **k):
            k["transport"] = _Recording(httpx.ASGITransport(app=server.app))
            k["base_url"] = "http://asgi"
            super().__init__(*a, **k)

    import httpx as _httpx_module
    monkeypatch.setattr(_httpx_module, "AsyncClient", _AsgiOnlyClient)
    monkeypatch.setattr(app.config, "service_url",
                        lambda *a, **k: "http://asgi")

    # The agent turn the endpoint would otherwise run on this machine. `enqueue_turn`
    # is the handoff to the session queue; the stub plays the consumer's part — it
    # closes the turn the way a finished turn is closed (`done`, then the sentinel)
    # — so the SSE the worker reads is well-formed and nothing reaches a model.
    # Everything up to and including `_save_session_meta`, the write whose value is
    # in question, runs unpatched and in the real order.
    async def _turn_finished_elsewhere(_sid, turn, *a, **k):
        await turn.events.put({"event": "done", "data": {
            "response": "stubbed: no model was called", "stop_reason": "end_turn",
            "num_turns": 1, "session_id": _sid}})
        await turn.events.put(None)
        return turn

    monkeypatch.setattr(messages, "enqueue_turn", _turn_finished_elsewhere)
    # Transcript persistence, which is the same write `post_capture`'s own test
    # stubs out: here it would file markdown under the real vault corpora.
    async def _no_persist(*a, **k):
        return None

    monkeypatch.setattr(messages, "_post_session_capture", _no_persist)
    monkeypatch.setattr(automod, "drain_active", lambda: False)

    out = await common.run_prompt_in_session(
        "classify the turn you are about to run", title="seam probe",
        source="autotriage", timeout_seconds=25.0)

    sid = out["session_id"]
    assert sessions_io.is_background_session_name(sid), (
        f"{sid} is not four-part, so this no longer crosses the seam that matters")
    path = sessions / f"{sid}.json"
    assert path.exists(), (
        "the backend never filed the session the worker POSTed for, so the "
        "platform assertion below could not have anything to read")
    filed = json.loads(path.read_text(encoding="utf-8"))
    assert sessions_io.is_user_session(filed) is False, (
        "the filed session reads as a human conversation, which is the whole "
        "defect this item is about")

    # Leg 1 — the SOURCE. What the worker actually put on the wire, after its own
    # serialisation. This is the half a test of `_save_session_meta` cannot see, and
    # it is the half that can silently regress: the shape rule below means the
    # four-part file says `worker` whether or not the body carries the field, so
    # nothing else in this file, or anywhere, notices the field going missing.
    assert sent, "the worker never reached /api/message/stream, so nothing crossed"
    body = sent[0]
    assert body.get("platform") == WORKER_SEAM_PLATFORM, (
        f"the POST body carries platform={body.get('platform')!r}, not "
        f"{WORKER_SEAM_PLATFORM!r}. The file above still reads "
        f"{filed.get('platform')!r} because a four-part id is stamped by shape, so "
        f"this is invisible everywhere else: the body is how a session that ALREADY "
        f"exists keeps its identity, and how a chat-shaped turn is classified at "
        f"all (leg 2, next) — `app/routers/messages.py` reads only this field")
    assert body.get("session_id") == sid, (
        "the body's session_id is not the id the worker minted, so the seam moved")

    # Leg 2 — the SINK, given those same bytes. A chat-shaped id has no shape to
    # fall back on: `worker` reaching the file is proof the endpoint read the body
    # and forwarded it. Drop `platform=data.get("platform")` from the endpoint and
    # this lands `mission-control`, which is a machine turn filed into the corpus
    # qmd embeds.
    chat_shaped = "20260921_000000_seam01"
    async with httpx.AsyncClient() as client:
        again = await client.post("/api/message/stream",
                                 json={**body, "session_id": chat_shaped})
    assert again.status_code == 200, f"the replayed body was refused: {again.status_code}"
    chat_filed = json.loads(
        (sessions / f"{chat_shaped}.json").read_text(encoding="utf-8"))
    assert chat_filed.get("platform") == WORKER_SEAM_PLATFORM, (
        f"a machine turn that named itself {WORKER_SEAM_PLATFORM!r} in its body was "
        f"filed as {chat_filed.get('platform')!r}: the endpoint stopped forwarding "
        f"the field, and a chat-shaped id defaults to a human conversation")


@pytest.mark.anyio
async def test_the_create_endpoint_refuses_a_platform_it_cannot_classify(tmp_path, monkeypatch):
    """`platform` is a choice from a list, not free text.

    It was `isinstance(platform, str)` and then stored verbatim, which is how
    `browser` and `e2e-harness` got minted at all. Rejecting an unknown word is
    the honest answer precisely because `is_user_session`'s default is generous:
    the endpoint must not be the place that quietly decides a new client is
    human. Existing senders keep working — the Chrome side panel posts `browser`
    (`chrome-extension/src/background/lloyd-client.ts:15`), the workers post
    `worker`, autonomy posts `autonomy`, and the four-part mint in the endpoint
    is unchanged (`<ts>_iv<4hex>`).
    """
    import httpx

    import server
    from app.routers import sessions as sessions_router
    from app.sessions_io import is_background_session_name

    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    app = server.app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/sessions/create",
                              json={"platform": "e2e-harness", "source": "e2e"})
        assert r.status_code == 400, f"unknown platform accepted: {r.status_code} {r.text}"
        assert "e2e-harness" in r.json()["detail"]
        assert "mission-control" in r.json()["detail"], (
            "the refusal must name the real choices, or the caller just guesses again")
        assert not list(tmp_path.glob("*.json")), "a refused create must write nothing"

        for good in ("browser", "worker", "autonomy", "mission-control"):
            r = await client.post("/api/sessions/create", json={"platform": good})
            assert r.status_code == 200, f"{good} rejected: {r.text}"

        r = await client.post("/api/sessions/create", json={"platform": 7})
        assert r.status_code == 400 and "string" in r.json()["detail"]

        r = await client.post("/api/sessions/create", json={})
        sid = r.json()["session_id"]
        assert not is_background_session_name(sid), f"shape drifted: {sid!r}"
        assert json.loads((tmp_path / f"{sid}.json").read_text())["platform"] == "mission-control"


#: The two client trees that mint a session over HTTP, and therefore the two that
#: can send the create endpoint a word it will refuse. `dist/` is build output —
#: it holds a minified copy of the same call, and sweeping it would pin a bundle
#: nobody edits.
_CLIENT_TREES = ("web/src", "chrome-extension/src")
_CLIENT_PLATFORM_LITERAL = re.compile(
    r"""platform['"]?\s*:\s*['"]([A-Za-z0-9_-]+)['"]""")


def test_every_client_that_names_a_platform_to_the_endpoint_names_a_known_one():
    """The 400 is only safe because every real sender was re-read, not assumed.

    `POST /api/sessions/create` used to accept any string, so a client could send
    anything. It now refuses a word outside `known_platforms()` — which turns a
    silently-mis-classified session into a hard failure at the caller, across a
    process boundary the code graph does not cross and a language this file cannot
    import. That is the right trade, but only if no live sender is newly refused:
    the Chrome side panel posts `browser`
    (`chrome-extension/src/background/lloyd-client.ts:15`) and Mission Control
    posts `mission-control` twice (`web/src/components/RightChatSidebar.tsx:196`,
    `web/src/components/pages/InnerVoicePage.tsx:145`). This sweep is the check
    that they still do, so a rename on either side fails HERE, in Python, rather
    than as a side panel that can no longer start a conversation.

    Both directions are asserted: an unknown word the endpoint would refuse is a
    broken client, and a sweep that found nothing is a broken sweep — the empty
    census is the failure mode this item keeps hitting.
    """
    from app.sessions_io import known_platforms

    known = set(known_platforms())
    found: dict[str, set[str]] = {}
    for tree in _CLIENT_TREES:
        root = ROOT / tree
        assert root.is_dir(), f"{tree} is gone, so this sweep cannot see that client"
        for path in root.rglob("*.ts*"):
            words = set(_CLIENT_PLATFORM_LITERAL.findall(
                path.read_text(encoding="utf-8", errors="replace")))
            if words:
                found[str(path.relative_to(ROOT))] = words

    assert found, (
        f"no client sends a `platform` literal any more, so the create endpoint's "
        f"allow-list is unverified against every caller — re-read "
        f"{_CLIENT_TREES} before trusting this pass")
    seen = {w for words in found.values() for w in words}
    assert {"browser", "mission-control"} <= seen, (
        f"clients name only {sorted(seen)}; the two senders this endpoint must "
        f"never refuse are `browser` (the side panel) and `mission-control` "
        f"(the web UI), and one of them has stopped naming itself")
    refused = sorted(seen - known)
    assert not refused, (
        f"{_format_sites(found, refused)} send a platform the create endpoint "
        f"refuses with a 400, so that client cannot mint a session at all. Either "
        f"classify the word in `sessions_io` (interactive, or a machine platform) "
        f"or fix the caller")


def _format_sites(found: dict, wanted) -> str:
    """`web/src/x.tsx:196`-style sites for an assertion message."""
    return ", ".join(f"{rel} ({sorted(words & set(wanted))})"
                     for rel, words in sorted(found.items())
                     if words & set(wanted))


def test_the_backfill_script_classifies_through_the_one_definition(tmp_path, monkeypatch):
    """The corpus backfill is a second writer and must ask the same question.

    Loaded by path the way the retention-sweep test does: `scripts/` is not an
    importable package, which is why this file greps it instead of importing it
    and why this one test does the import by hand. Before #1064 it skipped only
    `platform == "autonomy"` and wrote everything else into the chat corpus — 469
    `worker` exports sat under `_pipeline/vault-derived/sessions/` on 2026-09-17,
    under corpus dates 09-08 through 09-12, i.e. minted after `worker` joined the
    deny-list on 09-07 and never heard about it.

    `export_session` returns `(filename, ok, reason)`, so "which corpus" is read
    off the filesystem: `ok` is True for a machine run (it is exported, to the
    corpus that is not embedded) and the discriminator is the directory.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "backfill_session_markdown",
        ROOT / "scripts" / "memory" / "backfill-session-markdown.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    chat_corpus = tmp_path / "sessions"
    bg_corpus = tmp_path / "sessions-background"
    monkeypatch.setattr(mod, "VAULT_SESSIONS_DIR", chat_corpus)
    monkeypatch.setattr(mod, "VAULT_BACKGROUND_SESSIONS_DIR", bg_corpus)

    def _one(sid: str, platform: str) -> Path:
        src = tmp_path / f"{sid}.json"
        src.write_text(json.dumps({
            "session_id": sid, "platform": platform,
            "created_at": "2026-09-12T13:43:12",
            "messages": [{"role": "user", "content": "please review this diff carefully"},
                         {"role": "assistant", "content": "the gate refused it twice."}],
        }), encoding="utf-8")
        return src

    def _dest(sid: str, platform: str) -> tuple[bool, bool]:
        # `export_session` returns the SOURCE filename as its first element, so
        # which corpus it chose is read off the filesystem, not off the return.
        _name, ok, reason = mod.export_session(_one(sid, platform))
        assert ok, f"{sid} ({platform}) was not exported at all: {reason}"
        in_chat = any(chat_corpus.rglob(f"{sid}.md"))
        in_bg = any(bg_corpus.rglob(f"{sid}.md"))
        return in_chat, in_bg

    # Machine transcripts leave the embedded corpus and join the other one.
    for sid, platform in (("20260912_134312_review_9285", "worker"),
                          ("e2e_selfmod363_1788758995", "e2e-harness"),
                          ("20260917_105134_autocode_029c", "mission-control")):
        in_chat, in_bg = _dest(sid, platform)
        assert in_bg and not in_chat, (
            f"{sid} ({platform}) landed in the corpus qmd embeds; a four-part id "
            f"is refused whatever its platform says")

    # Human transcripts stay where the retrieval corpus is built.
    for sid, platform in (("20260912_140000_abc123", "mission-control"),
                          ("20260912_140100_iv1234", "browser"),
                          ("20260912_140200_9f2a1c", "")):
        in_chat, in_bg = _dest(sid, platform)
        assert in_chat and not in_bg, f"{sid} ({platform or 'no platform'}) left the chat corpus"


def test_the_guard_can_see_the_tree_the_backfill_lives_in():
    """`scripts/` is a swept root now, or this class re-files itself.

    The offender that made this item true sat at
    `scripts/memory/backfill-session-markdown.py:31`, inside a guard whose scan
    roots were `("app", "agent_mcp", "workers")`. A guard whose roots exclude the
    directory the bug lives in reports the entire class as clean.
    """
    assert "scripts" in _SWEEP_ROOTS
    swept = {p.relative_to(ROOT).parts[0] for p in _sources(_SWEEP_ROOTS)}
    assert swept == set(_SWEEP_ROOTS), (
        f"no .py found under {sorted(set(_SWEEP_ROOTS) - swept)} — a swept root "
        f"that holds nothing is a root that stopped existing")


_BACKFILL = "scripts/memory/backfill-session-markdown.py"


def _run_backfill_cli(args):
    """Run the backfill the way its own docstring tells an operator to run it.

    A subprocess, cwd the repo root, the script named by RELATIVE path, and
    `PYTHONPATH` removed — the combination the module docstring promises works. Run
    by path, Python puts `scripts/memory/` on `sys.path` and not the repo root, so
    this is the only shape that exercises the `_import_root_on_path()` insertion
    the script exists to need; the in-process test imports the module with the
    checkout already on the path and so cannot see the import break.
    """
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, _BACKFILL, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


def test_the_backfill_command_line_itself_runs_and_classifies(tmp_path):
    """The classification is only one half of the claim; the entry point is the other.

    `test_the_backfill_script_classifies_through_the_one_definition` loads the file
    with `importlib` inside pytest, which asserts the FUNCTION is right and says
    nothing about whether the documented command line still starts: the `sys.path`
    insertion, the module-level `from app.paths`/`from app.sessions_io` imports,
    and `main()`'s own wiring are all outside what an in-process import can see.
    The review rung named exactly that gap on 2026-09-21, and the failure this file
    exists to prevent is a corpus writer that silently stopped running.
    """
    src = tmp_path / "sessions"
    chat_corpus = tmp_path / "chat"
    bg_corpus = tmp_path / "bg"
    src.mkdir()
    for sid, platform in (("20260912_134312_review_9285", "worker"),
                          ("20260917_105134_autocode_029c", "mission-control"),
                          ("20260912_140000_abc123", "mission-control"),
                          ("20260912_140100_iv1234", "browser")):
        (src / f"{sid}.json").write_text(json.dumps({
            "session_id": sid, "platform": platform,
            "created_at": "2026-09-12T13:43:12",
            "messages": [{"role": "user", "content": "Did the command line run?"},
                         {"role": "assistant",
                          "content": "It did, and it classified the transcript the "
                                     "way the two listings classify it."}],
        }), encoding="utf-8")

    r = _run_backfill_cli(["--sessions-dir", str(src),
                           "--chat-dir", str(chat_corpus),
                           "--background-dir", str(bg_corpus)])
    assert r.returncode == 0, (
        f"`python3 {_BACKFILL}` exits {r.returncode} on the documented route.\n"
        f"stdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-2500:]}")
    # Exit code is the real signal; stderr is scanned for the ONE failure this
    # route is fragile about — `import app` not resolving, which is what the
    # `sys.path` insertion exists for. Other stderr is environmental noise the
    # script correctly passes through: inside a round worktree `app.paths` warns,
    # loudly and on purpose, that every state path it derives is the worktree's.
    for shape in ("Traceback", "ImportError", "ModuleNotFoundError"):
        assert shape not in r.stderr, (
            f"`python3 {_BACKFILL}` reports {shape} on its documented route:\n"
            f"{r.stderr[-2000:]}")
    assert "4 exported" in r.stdout, r.stdout[-800:]

    chat = sorted(p.stem for p in chat_corpus.rglob("*.md"))
    bg = sorted(p.stem for p in bg_corpus.rglob("*.md"))
    assert chat == ["20260912_140000_abc123", "20260912_140100_iv1234"], (
        f"the corpus qmd embeds holds {chat}; a machine run is in it again, or a "
        f"conversation has been pushed out of it")
    assert bg == ["20260912_134312_review_9285", "20260917_105134_autocode_029c"], (
        f"the non-embedded corpus holds {bg}")


def test_the_guard_fires_on_a_hand_rolled_check_added_under_scripts(tmp_path):
    """Positive control: the widened root is load-bearing, not decorative.

    A grep empty against the corpus reads identically whether the rule is
    satisfied or the pattern cannot reach the file — the failure mode this whole
    item is built on. So plant the exact old shape in the tree the backfill lives
    in and require the guard to report it.

    Planted in a tmp tree, not the checkout. The first pass wrote
    `scripts/_platform_guard_canary.py` into the shared source tree for the length
    of the test, and the gate runs pytest under 8 xdist workers — every other test
    that sweeps `scripts/` was reading a tree that one worker had a hand-rolled
    platform check in, and `finally: victim.unlink()` is not a lock: a second
    worker's sweep between the write and the unlink sees the plant and fails on an
    offender it did not plant. A sweep that can be poisoned by a sibling is a
    sweep whose green means nothing either way.

    So the sweep runs over both trees at once and the two verdicts are asserted
    separately: the planted file must be reported, and the live tree must report
    nothing. The first half is the control; the second is the claim.
    """
    planted_root = tmp_path / "tree"
    (planted_root / "scripts").mkdir(parents=True)
    planted = planted_root / "scripts" / "_platform_guard_canary.py"
    planted.write_text('x = d.get("platform") == "autonomy"\n', encoding="utf-8")

    def _sweep(roots, base):
        found = []
        for path in _sources(roots, base=base):
            if path == _DEFINITION or path in _RESTATES:
                continue
            if _LITERAL.search(path.read_text(encoding="utf-8")):
                found.append(str(path.relative_to(base)))
        return found

    assert _sweep(_SWEEP_ROOTS, base=planted_root) == ["scripts/_platform_guard_canary.py"], (
        "a hand-rolled platform check inside scripts/ went unseen")
    assert _sweep(_SWEEP_ROOTS, base=ROOT) == [], (
        "the live tree itself now carries a hand-rolled platform check")


def test_brief_delivery_keeps_the_deny_list_and_the_split_stays_honest():
    """Delivery readers still call `is_user_session`; listing readers no longer do.

    The dangerous later merge is the one that makes these two "consistent" by
    routing delivery through the allow-list: an ambient brief would then silently
    stop reaching any client whose platform string is new, which is the exact
    failure the deny-list was written for.
    """
    for rel in ("app/sessions_io.py",     # get_active_session_id
                "agent_mcp/session.py",   # _resolve_target_session
                "app/session_titles.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "is_user_session(" in text, (
            f"{rel} stopped answering the delivery question with the deny-list")
    for rel in ("app/routers/sessions.py", "app/routers/dashboard.py",
                "app/routers/mc_ui.py", "app/post_capture.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "is_conversation_session(" in text, (
            f"{rel} decides listing membership or corpus placement with the "
            f"delivery deny-list again — the leak #1064 is about")
        corpus_line = [ln for ln in text.splitlines()
                       if ln.strip().startswith("root = (")]
        if corpus_line:   # only post_capture picks a corpus
            assert "is_conversation_session(" in corpus_line[0], (
                f"{rel} chose the embedded corpus with the delivery deny-list: "
                f"{corpus_line[0].strip()}")


#: Every `scripts/` function that mints a session id AND stamps a `platform`,
#: found by the AST census below, with the reason each one is safe. The table IS
#: the guard: a new unattended minter has to be added here with its classification
#: or the test fails. Same shape as `_RESTATES`, which is how the one definition a
#: stdlib-only script may copy is kept honest.
_SCRIPT_MINTERS = {
    ("scripts/automod/review.py", "write_session"):
        "four-part `<ts>_review_<4hex>`, stamped `worker`; proven behaviourally",
    ("scripts/automod/canary_smoke.py", "run"):
        "chat-shaped `canary_<epoch>_<6hex>`, stamped `canary`, off the allow-list",
}


def _scripts_session_minters():
    """`(rel_path, function, mint_literal_chunks, platform_words)` for `scripts/`.

    An AST walk, not a regex over the literal. The first pass at this census
    grep'd `session_id = f"..."` and the review rung refused it on 2026-09-21:
    that pattern stops at the FIRST inner quote, so `review.py`'s mint —
    `f"{time.strftime('%Y%m%d_%H%M%S')}_review_{secrets.token_hex(2)}"` — was
    captured as the fourteen characters `{time.strftime(`, and substituting
    `{`→`2026` to probe its shape invented a token no writer ever emits
    (`2026time.strftime(`). Re-measured over this checkout, the old census yielded
    ZERO background-shaped suspects in every tree: its loop body never ran and the
    test stayed green while the one file it exists to watch went unseen. A census
    whose loop is vacuous reports its class as clean, which is the failure mode
    this whole item is built on.

    A parser sees the whole `JoinedStr` however many quotes sit inside it. The
    literal chunks are returned rather than a rendered id, because an f-string
    cannot be rendered without executing the mint's own calls — and where a shape
    decision actually carries weight, the test below *runs* the minter instead of
    guessing what it produces.
    """
    import ast

    found = []
    for path in _sources(("scripts",)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for fn in (n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            mint = None
            stamps: set[str] = set()
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign):
                    if (mint is None
                            and any(isinstance(t, ast.Name) and t.id == "session_id"
                                    for t in node.targets)
                            and isinstance(node.value, (ast.JoinedStr, ast.Constant))):
                        mint = ([v.value for v in node.value.values
                                 if isinstance(v, ast.Constant)]
                                if isinstance(node.value, ast.JoinedStr)
                                else [node.value.value])
                elif isinstance(node, ast.Dict):
                    for key, val in zip(node.keys, node.values):
                        if (isinstance(key, ast.Constant) and key.value == "platform"
                                and isinstance(val, ast.Constant)
                                and isinstance(val.value, str)):
                            stamps.add(val.value)
            if mint is not None and stamps:
                found.append((str(path.relative_to(ROOT)), fn.name,
                              [c for c in mint if isinstance(c, str)], stamps))
    return found


def test_a_scripts_minter_of_a_background_shape_names_a_machine_platform(tmp_path):
    """No `scripts/` minter may hand a machine run an interactive platform.

    `scripts/` mints ids too, and the mint census above covers only
    `app`/`agent_mcp`/`workers`, so it cannot see them. Two files write a session
    JSON directly: `scripts/automod/review.py:630` mints `<ts>_review_<4hex>`,
    which IS background-shaped, and stamps it `worker` in the same function;
    `scripts/automod/canary_smoke.py:61` mints `canary_<epoch>_<6hex>` — three
    parts, a chat shape — and stamps `canary`, a word on neither list. So the rule
    is not "three parts means a conversation". The rule `is_conversation_session`
    relies on is that a transcript written by a script is never a conversation,
    because there is no human on the script's side of it: a script that stamped
    `mission-control` or `browser` would file a machine run into the corpus qmd
    embeds, and no listing code would be at fault — which is #1064 with the router
    removed.

    Pinned per file, and the minter table asserted as a set, so a third minter
    cannot join the tree unnoticed. An allow-list nobody re-counts is exactly how
    the swept roots came to exclude the corpus backfill in the first place.
    """
    from app.sessions_io import (INTERACTIVE_PLATFORMS, NON_USER_PLATFORMS,
                                 is_background_session_name,
                                 is_conversation_session)

    minters = _scripts_session_minters()
    keys = {(rel, fn) for rel, fn, _chunks, _stamps in minters}
    assert keys == set(_SCRIPT_MINTERS), (
        "the scripts/ minter census disagrees with the table. New census hit: "
        f"{sorted(keys - set(_SCRIPT_MINTERS))} — classify it (what platform, and "
        f"why is it not a conversation) and add it to _SCRIPT_MINTERS. Stale "
        f"entry: {sorted(set(_SCRIPT_MINTERS) - keys)} — that minter is gone. "
        f"Census found: {sorted(keys)}")

    for rel, fn, chunks, stamps in minters:
        assert stamps, f"{rel}:{fn} minted an id but stamped no platform"
        assert not (stamps & set(INTERACTIVE_PLATFORMS)), (
            f"{rel}:{fn} mints {chunks!r} and stamps {sorted(stamps)}: an "
            f"allow-listed platform from an unattended script puts a machine "
            f"transcript in the corpus qmd embeds and in the user's chat history")

    # Behavioural leg — the reason this is a test and not only a census. Of the two
    # minters, review.py's is the background-shaped one, and the only honest way to
    # know that is to run it. `write_session` is stdlib-only and writes wherever it
    # is pointed, so it runs for real here; it is what produces the gate's own
    # grader transcript.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "automod_review_minter", ROOT / "scripts" / "automod" / "review.py")
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    sid = review.write_session(tmp_path, item_id=1064,
                               round_id="SM_PROBE", model="primary")
    data = json.loads((tmp_path / f"{sid}.json").read_text(encoding="utf-8"))
    assert is_background_session_name(sid), (
        f"review.write_session minted {sid!r}, not the four-part shape its table "
        f"entry claims — so the platform it stamps stopped mattering and "
        f"_SCRIPT_MINTERS is describing a minter that no longer exists")
    assert data["platform"] in set(NON_USER_PLATFORMS), (
        f"a four-part review transcript is stamped {data['platform']!r}, which is "
        f"not a platform the brief-delivery deny-list refuses")
    assert is_conversation_session(f"{sid}.json", data) is False, (
        "the review transcript would be listed as a conversation and embedded")

    # The canary mint cannot be run — `run()` POSTs to the gate's backend — so what
    # is pinned is what its literals state: a `canary_` prefix, hence a chat-shaped
    # id, whose platform is off the allow-list. That pairing is deliberate, not an
    # accident: the turn is a synthetic tool-dispatch probe, so it belongs neither
    # in the human history nor in the embedded corpus.
    canary = [m for m in minters if m[0].endswith("canary_smoke.py")][0]
    prefix, canary_platform = canary[2][0], next(iter(canary[3]))
    assert prefix.startswith("canary"), (
        f"canary_smoke's mint now starts {prefix!r}, so its id shape changed and "
        f"the pairing below has to be re-read rather than inherited")
    assert is_conversation_session(f"{prefix}1712345678_abcdef",
                                   {"platform": canary_platform}) is False, (
        f"the gate's synthetic turn ({prefix!r}…, platform {canary_platform!r}) "
        f"reads as a human conversation, so a tool-dispatch probe would be listed "
        f"in the chat history and embedded")




def test_an_unclassified_platform_still_receives_its_brief(tmp_path, monkeypatch):
    """The delivery deny-list, proven behaviourally rather than by grep.

    `get_active_session_id` is where the generous default earns its keep: a
    producer asks "which session may I inject into", and a client this codebase
    has never heard of must answer *yes* there while answering *no* to the chat
    listing. Both halves have to hold at once, which is precisely why one bool
    cannot serve both questions — and why routing delivery through the allow-list
    later would silently stop briefs reaching any new client.
    """
    from app import sessions_io

    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sessions_io, "_last_user_session_id", None)

    def _fresh(sid: str, platform: str, age_s: int):
        body = {"session_id": sid, "platform": platform, "messages": []}
        path = tmp_path / f"{sid}.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        os.utime(path, (time.time() - age_s, time.time() - age_s))

    _fresh("20260921_101112_abc123", "brand-new-client", 60)
    assert sessions_io.get_active_session_id() == "20260921_101112_abc123", (
        "a platform the deny-list has never heard of must keep receiving briefs")

    (tmp_path / "20260921_101112_abc123.json").unlink()
    _fresh("20260921_101500_worker_zz99", "worker", 60)
    assert sessions_io.get_active_session_id() is None, (
        "a deny-listed platform must never receive an injection")


#: The row counts the Background tab actually asks for: 150 from
#: `web/src/components/pages/BackgroundPage.tsx:101`, and the endpoint's own
#: default of 100 for every other caller.
BACKGROUND_TAB_LIMITS = (100, 150)


@pytest.mark.anyio
@pytest.mark.parametrize("limit", BACKGROUND_TAB_LIMITS)
async def test_the_orphans_reach_the_listing_past_both_budgets(
        tmp_path, monkeypatch, limit: int):
    """A rule either budget cannot reach is not a rule — and there are TWO budgets.

    The listing bounds how many files it opens, which is right for a directory of
    whole transcripts and wrong for membership. It also bounds how many ROWS it
    returns, which is right for a browser tab and, on the first pass at this fix,
    just as wrong for membership: candidates were sliced to `limit * 2` before any
    row was built, so a four-part run sitting past that index in mtime order was
    invisible whatever its platform said — the same defect one stage later, at a
    bigger number. This directory is sized to reproduce that and nothing else:
    318 ordinary four-part runs and one orphan older than all of them, so the
    orphan is at mtime index 319 of 320 while the tab asks for 150 rows. The
    review rung's measurement against live data put the 12 real orphans at
    candidate index 1222-3578 of 3,853 for the same reason.

    Both dimensions are pinned at once on purpose. The read ceiling is dropped to
    1 — what 600 is against 4,045 files on this box — so the four-part half must be
    settled by `is_background_session_name` plus a head-window platform read, and
    the row budget is the tab's own number, so the exemption for a mis-labelled
    run has to survive a directory two orders of magnitude bigger than it. The
    newest human chat and the newest ordinary run must still be returned: the
    exemption may not turn the listing into an orphan-only view.
    """
    import httpx

    import server
    from app import sessions_io
    from app.routers import sessions as sessions_router

    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sessions_router, "_BACKGROUND_PLATFORM_SCAN_CEILING", 1)

    def _write(sid: str, platform: str, minutes_ago: int) -> str:
        path = tmp_path / f"{sid}.json"
        path.write_text(
            json.dumps({"session_id": sid, "title": sid, "platform": platform,
                        "messages": []}), encoding="utf-8")
        stamp = time.time() - minutes_ago * 60
        os.utime(path, (stamp, stamp))
        return sid

    newest_chat = _write("20260921_100000_chat0000", "mission-control", 1)
    # 318 four-part `worker` runs, each older than the last, so the mtime order the
    # endpoint walks puts them all ahead of the orphan behind them.
    pool = [_write(f"2026091{i % 9}_1{i % 10:01d}000{i % 10}_worker_{i:04d}", "worker",
                   2 + i)
            for i in range(318)]
    newest_run = pool[0]
    # `20260909_153801_autonomy_2cf3` is a real orphan: four-part, `platform:
    # mission-control`, and older than every run on this box.
    orphan = _write("20260909_153801_autonomy_2cf3", "mission-control", 10_000)
    # A second mis-labelled run, at the OTHER end of mtime order, so a listing
    # that special-cased "the oldest file" would still fail here.
    newest_orphan = _write("20260921_100030_autocode_a911", "mission-control", 0)

    app = server.app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as client:
        payload = (await client.get(f"/api/background/sessions?limit={limit}")).json()
        chats = {s["id"] for s in (await client.get("/api/sessions")).json()["sessions"]}

    # Premise check, so a future edit that shrinks this directory cannot quietly
    # turn the test into one that passes for the wrong reason: the buried orphan
    # has to sit past the row budget the first pass sliced at, which is the whole
    # reason it was invisible. (`limit * 2` is that first pass's slice, not a
    # budget this endpoint still has.)
    ranked = sorted(tmp_path.glob("*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    rank = [i for i, p in enumerate(ranked) if p.stem == orphan][0]
    assert rank > limit * 2, (
        f"the buried orphan is mtime rank {rank} of {len(ranked)}, inside the "
        f"sliced-away window at limit={limit}: this fixture no longer reproduces "
        f"the reach defect and proves nothing about it")

    bg = {s["id"] for s in payload["sessions"]}
    for sid in (orphan, newest_orphan):
        assert sid in bg, (
            f"{sid} is four-part, `platform: mission-control`, and reached no "
            f"listing at limit={limit} — the row budget is dropping a candidate "
            f"the name rule settled for free")
        assert sid not in chats, f"{sid} is listed as a human conversation"
    assert newest_run in bg, (
        f"exempting the mis-labelled class at limit={limit} aged the newest "
        f"ordinary run out of the listing instead — the exemption is for the "
        f"class, not a re-purposed orphan-only view")
    assert newest_chat not in bg and newest_chat in chats, (
        "the human chat at the head of mtime order moved listings")
    assert payload["count"] == limit, (
        f"limit={limit} returned {payload['count']} rows; the exemption is for "
        f"the mis-labelled class, not an extra page of everything else")
    assert len(bg) == limit
    # `total` reports what the pass settled — every four-part run, whose shape
    # decides without a read, plus any chat-shaped file the one permitted read
    # found to be a machine run. Here the only chat-shaped file in the directory
    # IS a human chat, so the settled count is "everything except that one":
    # derived from the fixture rather than a literal, because a literal pinned to
    # 320 forces an unrelated edit to THIS assertion every time the fixture grows,
    # and an assertion that has to be edited in step with its own fixture is one a
    # future edit will simply update to whatever the code now returns.
    chat_shaped = [p for p in tmp_path.glob("*.json")
                   if not sessions_io.is_background_session_name(p.stem)]
    assert [p.stem for p in chat_shaped] == [newest_chat], (
        "the fixture grew a second chat-shaped file, so `total` is no longer "
        "derivable from the shape alone — re-read what the endpoint counts")
    assert payload["total"] == len(list(tmp_path.glob("*.json"))) - len(chat_shaped), (
        f"`total` reported {payload['total']}; the directory holds "
        f"{len(list(tmp_path.glob('*.json')))} files of which {len(chat_shaped)} "
        f"is a conversation, and a run cannot be both settled and missing")
    assert payload["scanned"] == 1, (
        "the read ceiling was not the one this test set, so the reach being "
        "proved here is not the reach that was missing")
