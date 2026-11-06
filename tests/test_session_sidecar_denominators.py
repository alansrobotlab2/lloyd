"""No session denominator counts a reduction sidecar (#3815, acceptance clause 3).

`app/harness/microcompact.reduction_sidecar_path` writes
`sessions/<sid>.microcompact-reduced.json` into the same directory and the same
`*.json` glob as the transcripts it records. On 2026-11-05 the store held 107 of them
beside 11,877 heads, and every one of the readers below globbed the directory raw —
so Mission Control's Chat tab printed `total_session_count: 11984` for a corpus of
11,877 conversations, the Background tab filed the 105 four-part ones as
`mislabelled` machine runs, and `app/tool_quality.py`'s 7-day window walked all 107 as
transcripts that cannot contain a tool call.

One directory, N heads and M sidecars, is built by the fixture below and read back
through every denominator the item names. The naming rule itself is
`app.harness.microcompact.is_reduction_sidecar`, pinned by
tests/test_microcompact_sidecar_naming.py; what this file pins is that each of these
sites reaches that rule rather than counting files.
"""
import json
import time

import pytest

from app.harness.microcompact import REDUCTION_SIDECAR_SUFFIX

#: Four heads, two sidecars — N=4 and M=2, deliberately unequal so a denominator that
#: counted files instead of heads answers 6 and the assertion says so.
#: Three underscore-separated parts: `is_background_session_name` says False, which
#: is what makes them Chat-tab material. (A name whose third token is a known agent —
#: `chat`, `autocode`, `worker` — is four-part-shaped to that predicate, so an id
#: invented as `..._chat_alpha` would be a background row, not a chat.)
CHAT_HEADS = ["20261105_100000_a1b2c3", "20261105_100100_d4e5f6"]
BACKGROUND_HEADS = ["20261105_100200_autocode_charlie",
                    "20261105_100300_worker_delta"]
HEADS = CHAT_HEADS + BACKGROUND_HEADS
#: One sidecar per shape: a chat-shaped head's and a four-part head's. Both are on the
#: real box in both shapes (2 chat-shaped, 105 four-part).
SIDECARS = [CHAT_HEADS[0] + REDUCTION_SIDECAR_SUFFIX,
            BACKGROUND_HEADS[0] + REDUCTION_SIDECAR_SUFFIX]

#: What relief actually writes: no `platform`, no `source`, no `session_id`, no
#: `messages`. Every "it would have been classified as X" below follows from this.
SIDECAR_PAYLOAD = {"schema": "microcompact-reduced.v1",
                   "cleared_call_ids": ["toolu_01A"],
                   "reduced_call_ids": ["toolu_01B"]}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A sessions directory with 4 heads and 2 sidecars, every site pointed at it.

    Each site is patched where IT binds the directory: the routers hold their own
    module-level `SESSIONS_DIR`, `app/routers/dashboard.py` imports it from
    `app.paths` inside the function, and the two collectors take it as an argument.
    """
    d = tmp_path / "sessions"
    d.mkdir()

    def head(sid, *, platform, inner_voice=False):
        body = {"session_id": sid, "title": sid, "preview": f"head {sid}",
                "platform": platform, "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "created_at": "2026-11-05T10:00:00",
                "last_active": "2026-11-05T10:00:00"}
        if inner_voice:
            body["inner_voice"] = True
        (d / f"{sid}.json").write_text(json.dumps(body), encoding="utf-8")

    # Only two platforms make a conversation: `sessions_io.INTERACTIVE_PLATFORMS` is
    # {"mission-control", "browser"}. Anything else is not a chat, so inventing a
    # plausible-sounding one here would silently drop a head out of both listings.
    head(CHAT_HEADS[0], platform="mission-control", inner_voice=True)
    head(CHAT_HEADS[1], platform="browser")
    head(BACKGROUND_HEADS[0], platform="autonomy", inner_voice=True)
    head(BACKGROUND_HEADS[1], platform="worker")
    for name in SIDECARS:
        (d / name).write_text(json.dumps(SIDECAR_PAYLOAD), encoding="utf-8")

    from app.routers import inner_voice, mc_ui, sessions
    import app.paths as app_paths
    import app.sessions_io as sessions_io

    monkeypatch.setattr(mc_ui, "SESSIONS_DIR", d)
    monkeypatch.setattr(inner_voice, "SESSIONS_DIR", d)
    monkeypatch.setattr(sessions, "SESSIONS_DIR", d)
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", d)
    monkeypatch.setattr(app_paths, "SESSIONS_DIR", d)
    # The row cache is keyed by path string; a sibling test's tmp dir cannot
    # collide, but a re-run of THIS one inside one process can.
    inner_voice._SESSION_ROW_CACHE.clear()
    return d


def _ids(rows):
    return {r.get("id") or r.get("session_id") for r in rows}


def test_the_chat_listers_report_n_heads_not_n_plus_m_heads_and_sidecars(store):
    """`GET /api/sessions`'s collector: 2 chat heads, so 2 rows — and no row whose id
    carries the suffix.

    Without the exclusion the chat-shaped sidecar reaches `is_conversation_session`
    with a payload that has no `platform`, the `or "mission-control"` default fires,
    and the listing grows a third row whose `model` is empty and whose
    `message_count` is zero — the shape
    `app/behaviour_exhibit.py`'s comment already describes from a committed run that
    counted 84 of them in a 7-day window.
    """
    from app.routers.sessions import _collect_chat_sessions

    out = _collect_chat_sessions(store)
    assert out["count"] == len(CHAT_HEADS), out["count"]
    assert _ids(out["sessions"]) == set(CHAT_HEADS)
    assert not [i for i in _ids(out["sessions"])
                if i.endswith(REDUCTION_SIDECAR_SUFFIX)]


def test_the_background_listers_report_n_heads_not_n_plus_m_heads_and_sidecars(store):
    """The Background collector: 2 background heads, and the sidecars are not among
    them in ANY of its three classes.

    This is the site the old code got most wrong in the most legible way. A four-part
    sidecar name passes the shape test, then reads as no-platform, and #1228's
    no-platform class files it as `mislabelled` — so 105 of the 107 live sidecars were
    `mislabelled` machine runs, and `total` (settled machine runs plus mislabelled)
    counted them. `unclassified` is derived from `paths`, so a sidecar left in the
    candidate list would move that number too.
    """
    from app.routers.sessions import _collect_background_sessions

    out = _collect_background_sessions(store, 50)
    assert out["count"] == len(BACKGROUND_HEADS), out["count"]
    assert _ids(out["sessions"]) == set(BACKGROUND_HEADS)
    assert out["total"] == len(BACKGROUND_HEADS)
    assert out["unclassified"] == 0


def test_the_mission_control_totals_report_n_heads_not_n_plus_m(store):
    """All three of Mission Control's session totals: the Chat tab's
    `total_session_count` (the number `mc_navigate` puts in the model's context), the
    Inner Voice tab's `sessions_total`, and the Background tab's `sessions_total`.

    Before #3815 the first of these was a bare `sum(1 for sf in
    SESSIONS_DIR.glob("*.json"))` — the published figure that was reading 11,984 where
    the corpus was 11,877. The Background one also used to spend its 400-file read
    budget on sidecars and file each under the `unknown` source bucket, since a
    sidecar names no `source`.
    """
    from app.routers import mc_ui

    assert mc_ui._summarize_chat()["total_session_count"] == len(HEADS)
    assert mc_ui._summarize_inner_voice()["sessions_total"] == len(HEADS)
    bg = mc_ui._summarize_background()
    # The Background tab's own denominator is the complement of the chat half: the 2
    # unattended heads. Before the exclusion it was 4 — the 2 sidecars are not
    # conversations either, so each was bucketed under `source`/"unknown" and added.
    assert bg["sessions_total"] == len(BACKGROUND_HEADS), bg["by_source"]
    assert bg["by_source"] == {"autonomy": 1, "worker": 1}, bg["by_source"]


def test_the_mcp_session_list_tool_reports_n_heads_not_n_plus_m(store, monkeypatch):
    """`chat_list_sessions` across the boundary it is actually served from: the MCP
    handler, its serialized JSON, and the rows the model receives.

    The App-side totals above are one process. This tool is the other one: the
    `agent_mcp` server binds `SESSIONS_DIR` at import (`from app.paths import
    SESSIONS_DIR`), so the fixture's patch of `app.paths` cannot reach it and the
    module attribute is patched here — the same "patch it where it binds" rule the
    fixture states for the routers. `agent_mcp.mission_control` imports the writer's
    predicate at module load and filters its own glob; this node is what keeps that
    filter from being a claim about a file no test runs.

    The published id is `data.get("session_id", sf.stem)` and a sidecar payload carries
    no `session_id` key, so unfiltered the handler advertised
    `<sid>.microcompact-reduced` as a session id — one `chat_get_session` away from a
    lookup for a file that is a reducer's bookkeeping record, not a transcript. The two
    sidecars are also given the FRESHEST mtimes below, because the handler returns
    `sessions[:50]` sorted by mtime descending: on the real store the newest files ARE
    the sidecars, so a filter that only fixed `count` and left the rows ordered by
    mtime would still have spent the window the model sees.
    """
    import asyncio
    import os

    from agent_mcp import mission_control as mc_server

    monkeypatch.setattr(mc_server, "SESSIONS_DIR", store)
    future = time.time() + 3600
    for name in SIDECARS:
        os.utime(store / name, (future, future))

    payload = json.loads(mc_server._list_sessions())
    assert payload["success"] is True, payload
    assert payload["count"] == len(HEADS), (
        f"count is {payload['count']}, the corpus is {len(HEADS)}: a sidecar counted")
    assert {r["session_id"] for r in payload["sessions"]} == set(HEADS)
    phantom = REDUCTION_SIDECAR_SUFFIX.removesuffix(".json")
    assert not [r["session_id"] for r in payload["sessions"]
                if r["session_id"].endswith(phantom)], payload["sessions"]
    # The freshest-file claim above, pinned: were the sidecars still in the listing,
    # their future mtimes would put them in rows 0 and 1 of the window.
    assert payload["sessions"][0]["session_id"] in set(HEADS)

    served = asyncio.run(mc_server.call_tool("chat_list_sessions", {}))
    assert served.is_error is False
    assert json.loads(served.content[0].text)["count"] == len(HEADS)


def test_the_dashboard_scan_reports_n_heads_not_n_plus_m(store):
    """`_scan_recent_sessions` selects its candidates by mtime, and a sidecar's mtime
    is the end of the turn that reduced — i.e. among the newest in the store.

    The row builder is where the shape showed: a sidecar has no `messages` and no
    `platform`, so a chat-shaped one reached `is_conversation_session`, the
    `or "mission-control"` default fired, and it entered the "recent chats" panel as a
    row with nothing in it — pushing a real transcript out of the scan window. The
    105 four-part sidecars were already skipped by the shape rule, which is exactly
    why a fix that leaned on shape alone would have looked done.
    """
    from app.routers.dashboard import _scan_recent_sessions

    rows = _scan_recent_sessions()
    # This panel is the CHAT panel: it skips a background-named file unread and then
    # applies `is_conversation_session`, so its denominator is the 2 chat heads — and
    # the 2 sidecars are not in it either.
    assert len(rows) == len(CHAT_HEADS), [r.get("session_id") for r in rows]
    assert {r["session_id"] for r in rows} == set(CHAT_HEADS)
    assert not [r for r in rows
                if str(r.get("session_id", "")).endswith(REDUCTION_SIDECAR_SUFFIX)]


def test_the_inner_voice_listing_never_scans_a_sidecar(store):
    """`_scan_inner_voice_sessions` stats every candidate before the read, so the
    exclusion has to happen at the stat — and the control here is the payload.

    Both IV-flagged files in the fixture are heads, so an assertion that only counts
    rows could be satisfied by a reader that opened the sidecar and found nothing in
    it. This one gives each sidecar `"inner_voice": True`, which is exactly the key
    `_session_row` tests: were the sidecar scanned at all, it would produce a row with
    `session_id` equal to its stem — `<sid>.microcompact-reduced`, an id with no
    session behind it. The exclusion is therefore pinned to be the NAME, not the
    emptiness of the payload.
    """
    from app.routers import inner_voice

    for name in SIDECARS:
        p = store / name
        body = dict(SIDECAR_PAYLOAD)
        body["inner_voice"] = True
        p.write_text(json.dumps(body), encoding="utf-8")
    inner_voice._SESSION_ROW_CACHE.clear()

    rows = inner_voice._scan_inner_voice_sessions(limit=50)
    assert _ids(rows) <= set(HEADS), _ids(rows)
    assert not [r for r in rows
                if str(r["session_id"]).endswith(REDUCTION_SIDECAR_SUFFIX)]
    # The two real IV heads are still there: the exclusion is not a filter that
    # dropped the whole listing.
    assert _ids(rows) == {CHAT_HEADS[0], BACKGROUND_HEADS[0]}


def test_the_active_session_lookup_never_names_a_sidecar(store):
    """`sessions_io.get_active_session_id` picks the newest `*.json` by mtime, which
    is precisely what a sidecar is.

    Its payload has no `session_id`, so the `sf.stem` fallback would have returned
    `20261105_100000_a1b2c3.microcompact-reduced` — a session that has no file —
    to whoever injects a brief or an ambient turn. The sidecars are backdated to be
    the freshest files in the fixture so the mtime race actually runs, and the heads
    are stamped now.
    """
    import os

    import app.sessions_io as sessions_io

    now = time.time()
    for name in HEADS:
        os.utime(store / f"{name}.json", (now, now))
    for name in SIDECARS:
        os.utime(store / name, (now + 60, now + 60))

    found = sessions_io.get_active_session_id(max_age_hours=24.0)
    assert found in HEADS, found


def test_the_tool_quality_window_walk_reports_n_heads_not_n_plus_m(store):
    """`app.tool_quality.py`'s dated window walk, the denominator of every rate in the
    daily tool-quality row.

    The date guard does not exclude a sidecar: its name begins with the head's
    `YYYYMMDD`, so `_FILE_DATE_RE.match` accepts it and a `days=7` window swept all of
    them into the corpus as transcripts with zero tool calls — diluting every rate with
    rows that cannot contain one. The report now names them separately, beside
    `files_undated_in_dir`, so the gap between "files in the directory" and "files
    scanned" stays explained instead of moving here.
    """
    import app.tool_quality as tq

    paths = tq.transcript_paths(store, days=7, today="20261105")
    assert {p.name for p in paths} == {f"{h}.json" for h in HEADS}

    report = tq.sweep(sessions_dir=store, out_dir=store / "tq",
                      days=7, today="2026-11-05")
    assert report["corpus"]["files_scanned"] == len(HEADS)
    assert report["corpus"]["files_sidecar_in_dir"] == len(SIDECARS)
