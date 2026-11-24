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


# ── #3934: the session-distill producer is a denominator too ──────────────────
#
# `workers/sources/session_distill.py` decides what to distil with three gates of its
# own (quiet, not mid-turn, a user platform) and globbed the directory raw, so a
# sidecar became a conversation queued for the primary model to write notes about. It
# passes every gate the source asks: the payload has no `platform`, and
# `is_user_session` reads a missing platform as the web UI — its own docstring,
# "Missing platform means the web UI, on both lists". Measured on the live store on
# 2026-11-08: `_eligible` answered `(True, "")` for 145 of 145 files carrying the
# suffix, and `_scan` over the same tree with the 105 live `done:` markers returned 156
# candidates, 145 of them sidecars, oldest-first, with a sidecar sitting third in
# `_MAX_ENQUEUE_PER_TICK = 3`. `is_session_active(path.stem)` is False for a sidecar
# too, precisely because the stem carries the suffix.
#
# So the filter can only be the NAME, and it belongs in `_eligible` — `_scan` calls it
# per candidate and is its only caller, so one site covers the scan and anything added
# beside it, and the two cannot disagree about what a sidecar is.


def _distill_source(store, monkeypatch):
    """The session-distill source, pointed at the fixture directory.

    It binds `SESSIONS_DIR` at import (`from app.paths import SESSIONS_DIR`), so the
    fixture's patch of `app.paths` cannot reach it. Patched where it binds, exactly as
    the Mission Control node above patches `agent_mcp`.
    """
    from workers.sources import session_distill

    monkeypatch.setattr(session_distill, "SESSIONS_DIR", store)
    return session_distill


def test_the_distill_source_refuses_a_sidecar_by_name_and_by_the_shared_predicate(
        store, monkeypatch):
    """Clause 1: `_eligible` answers False, with a reason, for every name #3815's
    predicate accepts — and it reaches that predicate instead of spelling its suffix.

    The reason is part of the clause and not decoration, even though `_scan` discards it
    today (`ok, _why`): `""` is exactly what `_eligible` returns for a file it ACCEPTED,
    so a refusal that arrives blank is indistinguishable from a non-refusal to any reader
    of the tuple — which is the only reader this function has, and the one a caller that
    ever logs the reason would read.

    The second half is why this node reads the module's own source. A copy of the
    suffix here would work today and rot the day the writer changes it — the tree
    already carries a second spelling in
    `scripts/groundskeeper/retention-sweep.py::_is_reduction_sidecar`, which is
    #3815's own leftover, so "one rule" is a rule about new sites.
    """
    from pathlib import Path

    from app.harness import microcompact

    sd = _distill_source(store, monkeypatch)
    now = time.time() + sd._QUIET_SECONDS + 60

    for name in SIDECARS:
        ok, why = sd._eligible(store / name, (store / name).stat().st_mtime, now)
        assert ok is False, f"{name} is a reducer's bookkeeping record, not a transcript"
        assert why, f"a refusal whose reason is empty reads as no opinion: {name}"

    assert sd.is_reduction_sidecar is microcompact.is_reduction_sidecar, (
        "the source must call #3815's one naming rule, not keep a local copy of it")
    src = Path(sd.__file__).read_text(encoding="utf-8")
    assert "microcompact-reduced" not in src, (
        "the suffix is spelled once, in app.harness.microcompact — a second literal "
        "here is the second naming rule #3815 removed")


def test_the_distill_scan_offers_only_the_chat_heads_when_sidecars_share_the_directory(
        store, monkeypatch):
    """Clause 2: `_scan(now, set())` returns the 2 chat-platform heads — no sidecar,
    and no machine head.

    `set()` for `already_done` so the answer is the gates' alone and nothing is hidden
    by a marker. The directory holds 6 files and the expected answer is 2, not 4 and
    not 6: of the 4 heads, the `autonomy` and `worker` ones fail `is_user_session` on
    their own merits, which is why the item's "returns only the N heads" has to read
    N=2 of 4 — a node that asserted 4 would fail on a correct change.

    Both halves matter together. Removing the sidecars by removing every unreadable
    file would also remove a conversation, which is the failure #3815's denominators
    pinned from the other side.
    """
    sd = _distill_source(store, monkeypatch)
    now = time.time() + sd._QUIET_SECONDS + 60

    out = sd._scan(now, set())
    names = sorted(p.name for _, p in out)
    assert names == sorted(f"{h}.json" for h in CHAT_HEADS), names
    assert not [n for n in names if n.endswith(REDUCTION_SIDECAR_SUFFIX)], names


def test_the_distill_source_keeps_all_three_original_gates_biting(store, monkeypatch):
    """Clause 3: the sidecar gate is added to the source's three original gates, not
    swapped in place of any of them.

    Each was written against a measured loss and is re-pinned here on a HEAD, so a
    change that stopped calling them cannot pass clause 2 and fail nothing:

      * *quiet* — a chat still being typed into is not a transcript to learn from;
      * *not mid-turn* — one chat was distilled 44 times before the `done:` markers;
      * *a user session* — `platform: worker` turns go through the chat path, and 12
        machine transcripts were mined as observations about the user.

    The active-session gate is forced because `_session_queues` is in-process state no
    fixture file can populate: `_eligible` imports it at call time from
    `app.sessions_io`, so patching that attribute is what the call actually reads.
    """
    import app.sessions_io as sessions_io

    sd = _distill_source(store, monkeypatch)
    now = time.time() + sd._QUIET_SECONDS + 60

    quiet_head = store / f"{CHAT_HEADS[0]}.json"
    mtime = quiet_head.stat().st_mtime

    ok, why = sd._eligible(quiet_head, mtime, mtime + sd._QUIET_SECONDS - 1)
    assert ok is False and "still active" in why, (quiet_head, why)

    monkeypatch.setattr(sessions_io, "is_session_active", lambda sid: True)
    ok, why = sd._eligible(quiet_head, mtime, now)
    assert ok is False and "a turn is running" in why, (quiet_head, why)
    monkeypatch.setattr(sessions_io, "is_session_active", lambda sid: False)

    machine_head = store / f"{BACKGROUND_HEADS[1]}.json"   # platform: worker
    ok, why = sd._eligible(machine_head, machine_head.stat().st_mtime, now)
    assert ok is False and "not a user session" in why, (machine_head, why)

    ok, why = sd._eligible(quiet_head, mtime, now)
    assert ok is True and why == "", ("a quiet chat head must still be distillable", why)


def test_the_legacy_cursor_migration_seeds_no_marker_for_a_sidecar(
        store, monkeypatch, tmp_path):
    """Clause 4: `last_mtime` -> `done:` markers, for the head names only.

    The migration is a second, independent walk of the same glob: it does not consult
    `_eligible`, because its question is "was this ever enqueued", not "is this worth
    distilling now". A sidecar seeded here would be marked done forever while nothing
    had ever offered it, and the 105-row marker set would stop being a record of
    sessions.

    Dead in production, and pinned anyway: the live `watermarks` table holds
    `last_enqueue_check`, `last_enqueue_ok` and the 105 `done:` rows, because the
    cursor is deleted at the end of the first run that finds it, so this path returns 0
    on the live box today. It is testable only by seeding the cursor, which is what
    happens here. Expect no live marker count to move when this lands.
    """
    from workers.queue import WorkQueue

    sd = _distill_source(store, monkeypatch)
    queue = WorkQueue(tmp_path / "workers.db")

    newest = max((store / n).stat().st_mtime
                 for n in [f"{h}.json" for h in HEADS] + list(SIDECARS))
    queue.wm_set(sd.NAME, sd._LEGACY_CURSOR, str(newest + 10))

    seeded = sd._migrate_legacy_cursor(queue)

    assert seeded == len(HEADS), (seeded, [n for n in HEADS])
    for head in HEADS:
        assert queue.wm_get(sd.NAME, sd._done_key(f"{head}.json")) is not None, head
    for sidecar in SIDECARS:
        assert queue.wm_get(sd.NAME, sd._done_key(sidecar)) is None, (
            f"a sidecar was filed as a distilled session: {sidecar}")
    assert queue.wm_get(sd.NAME, sd._LEGACY_CURSOR) is None, (
        "the cursor is deleted once converted, so no session can be stranded behind it")


# ── #4447: the Inner Voice brief is ceilingless BY DECISION, and says so ────────
#
# `_summarize_inner_voice` is the one session enumeration in `app/routers/mc_ui.py`
# that opens every file in the directory, and #4267's owed-check ruled on 2026-11-20
# that it stays that way. The reason is not cost — it is meaning: `sessions_total`
# and `iv_enabled_count` are full-directory denominators, so a read budget would not
# make the brief cheaper, it would make the number describe a window while still
# being labelled as the store. The other two briefs were moved onto
# `sessions_io.scan_session_heads` with `SESSION_SCAN_CEILING` (400) precisely because
# what they publish IS a newest-first window.
#
# That ruling existed nowhere in the tree: `iv_enabled_count` appeared in exactly one
# line repo-wide (the `return` that builds it), zero test files, and the function had
# no docstring at all. A ruling only one reader ever saw is a ruling the next reader
# re-opens, so the four nodes below pin it four ways: the count past the shared
# ceiling (behaviour), the second denominator (first test ever to name it), the
# source that enumerates `session_head_files` instead of globbing (shape), and the
# docstring that states the reopen condition (contract). The `store` fixture above is
# 4 heads and cannot tell a 400-file budget from no budget, which is why this section
# builds its own, bigger, directory.

#: One more head than the shared budget, so a walk that adopted
#: `SESSION_SCAN_CEILING` answers 400 here and this section's first node says so.
IV_HEADS = 401

#: Two sidecars, both carrying `"inner_voice": True` in their payload, so a reader
#: that stopped excluding sidecars moves BOTH denominators (401→403, 101→103) rather
#: than only one. Same payload #3815 says relief actually writes, plus the one key
#: this brief reads.
IV_SIDECARS = 2

#: Every 4th head is flagged (indices 0, 4, …, 400 → 101 of 401). Deliberately not
#: half, not all, and not coprime with the head count: an implementation that
#: counted files, or heads, or sidecars instead of IV-flagged heads answers a
#: different number, and 101 ≠ 401 ≠ 403 ≠ 0.
IV_EVERY_4TH = 4


def _iv_head_names():
    """The 401 fixture head names. Name shape is free here: `is_session_head` is a
    NAME test (not a sidecar ⇒ a head), and this brief counts every head regardless
    of whether the name looks like a chat or a machine run."""
    return [f"20261105_{120000 + i:06d}_iv{i:04d}" for i in range(IV_HEADS)]


def _iv_enabled_head_names():
    """The subset of those names whose payload carries `inner_voice` — 101 of them.

    Derived from the same enumeration the fixture writes, so the assertion below
    reads as the clause it pins ("equals the number of heads whose payload carries
    inner_voice true") instead of as a constant nobody can re-derive.
    """
    return [n for i, n in enumerate(_iv_head_names()) if i % IV_EVERY_4TH == 0]


@pytest.fixture
def iv_store(tmp_path, monkeypatch):
    """A sessions directory of 401 heads (101 IV-flagged) plus 2 IV-flagged sidecars.

    Patched only in `app.routers.mc_ui`, where the brief binds its own
    `SESSIONS_DIR` — the "patch it where it binds" rule the 4-head fixture states.
    Nothing else walks the directory in these four nodes.
    """
    d = tmp_path / "sessions"
    d.mkdir()

    for i, sid in enumerate(_iv_head_names()):
        body = {"session_id": sid, "title": sid, "preview": f"head {sid}",
                "platform": "mission-control", "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "created_at": "2026-11-05T12:00:00",
                "last_active": "2026-11-05T12:00:00"}
        if i % IV_EVERY_4TH == 0:
            body["inner_voice"] = True
        (d / f"{sid}.json").write_text(json.dumps(body), encoding="utf-8")

    for name in [f"{n}{REDUCTION_SIDECAR_SUFFIX}" for n in _iv_head_names()[:IV_SIDECARS]]:
        (d / name).write_text(json.dumps({**SIDECAR_PAYLOAD, "inner_voice": True}),
                              encoding="utf-8")

    from app.routers import mc_ui
    monkeypatch.setattr(mc_ui, "SESSIONS_DIR", d)
    return d


def test_the_inner_voice_brief_opens_every_head_past_the_shared_ceiling(iv_store):
    """Clause 1: 401 heads in, `sessions_total: 401` out — ceilingless by behaviour.

    The last node is the control that makes this a test rather than a comment: the
    same directory walked by `sessions_io.scan_session_heads` at its own default
    budget stops at 400. So if anyone moves this brief onto the bounded walk, or
    hands it a budget of its own, `sessions_total` answers 400 and this node fails —
    which is the failure #4447 exists to make loud, because the number the model is
    handed would then be a window labelled as the store.

    The second assertion is the identity the ruling rests on, and the one a swap to
    the shared listing must never break: the brief's count equals
    `len(session_head_files(...))` over the same directory, sidecars excluded.
    """
    from app import sessions_io
    from app.routers import mc_ui

    out = mc_ui._summarize_inner_voice()
    assert out["sessions_total"] == IV_HEADS, out
    assert out["sessions_total"] == len(sessions_io.session_head_files(iv_store)), out
    assert IV_HEADS > sessions_io.SESSION_SCAN_CEILING, sessions_io.SESSION_SCAN_CEILING
    assert (len(list(sessions_io.scan_session_heads(iv_store)))
            == sessions_io.SESSION_SCAN_CEILING), (
        "the fixture no longer exceeds the shared budget, so clause 1's "
        "'past the ceiling' has nothing to be past")


def test_the_inner_voice_brief_counts_every_iv_flagged_head_in_the_directory(iv_store):
    """Clause 2: `iv_enabled_count` == the 101 heads whose payload says
    `inner_voice` — that denominator pinned for the first time (`git grep
    iv_enabled_count -- tests` was 0 hits before this file).

    It is the numerator the tab publishes beside `sessions_total`, and until now
    nothing in the tree said what it counts. Both sidecars here are IV-flagged too,
    so an answer of 103 means sidecars were read as sessions, and an answer of 401
    means the IV flag was ignored and every head counted; neither failure is
    distinguishable on the 4-head fixture, where 2 of 4 heads are flagged.
    """
    from app.routers import mc_ui

    out = mc_ui._summarize_inner_voice()
    assert out["iv_enabled_count"] == len(_iv_enabled_head_names()), out
    assert out["iv_enabled_count"] < out["sessions_total"], out
    assert len(_iv_enabled_head_names()) == 101, _iv_enabled_head_names()


def test_the_inner_voice_denominators_reach_the_agent_through_the_navigate_route(iv_store):
    """The seam the two numbers actually travel: `POST /api/mc/navigate` →
    `_summarize_tab` → `_SUMMARIZERS["inner_voice"]` → `_summarize_inner_voice`, and
    the envelope the agent's `mc_navigate` tool reads back as `detail`.

    The node above calls the brief in-process; this one calls it the way the tool's
    loopback POST does, so the registry entry, the `_safe` wrapper that turns a raise
    into `{}`, and the route's `{tab, focus_id, detail}` shape are all on the path.
    No frontend subscribes — `mc_state.publish_navigate` fans out to nobody, which is
    the cold case this route usually is.

    It re-asserts the same pair rather than a copy of it: a `_safe` that swallowed an
    exception would return `{}` and fail here on the missing key, not pass.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import mc_ui

    app = FastAPI()
    app.include_router(mc_ui.router)
    res = TestClient(app).post(
        "/api/mc/navigate", json={"tab": "inner_voice", "focus_id": None})
    assert res.status_code == 200, res.text
    detail = res.json()["detail"]
    assert detail["sessions_total"] == IV_HEADS, detail
    assert detail["iv_enabled_count"] == len(_iv_enabled_head_names()), detail


def test_the_inner_voice_brief_enumerates_the_shared_listing_and_carries_no_budget():
    """Clause 3: the function's own source reaches `session_head_files(SESSIONS_DIR)`
    and holds no budget of any kind.

    Behaviour cannot tell a raw glob with a hand-rolled skip from the shared listing
    — that is what the #3815 node above is for — but it is the enumeration itself
    #4267 consolidated: the last `SESSIONS_DIR.glob("*.json")` left in this module sat
    in this function, applying `is_session_head` by hand two lines below the place
    that already imported the helper. So this node reads the source, which is the
    only way to pin that the site is GONE rather than merely still correct.

    The negative half is the ruling: no `SESSION_SCAN_CEILING`, no `ceiling` argument,
    no `break`, no slice. A docstring may discuss ceilings — it is stripped from the
    text under test so the body is what is judged.
    """
    import ast
    import inspect
    import re
    import textwrap

    from app.routers import mc_ui

    code = textwrap.dedent(inspect.getsource(mc_ui._summarize_inner_voice))
    fdef = ast.parse(code).body[0]
    assert isinstance(fdef, ast.FunctionDef), ast.dump(fdef)

    # The docstring is the function's first statement when it has one; everything
    # after it is the body, and only the body is judged below.
    skip = 1 if (isinstance(fdef.body[0], ast.Expr)
                 and isinstance(fdef.body[0].value, ast.Constant)
                 and isinstance(fdef.body[0].value.value, str)) else 0
    lines = code.splitlines()
    body_src = "\n".join(line
                         for stmt in fdef.body[skip:]
                         for line in lines[stmt.lineno - 1:stmt.end_lineno])

    assert "session_head_files(SESSIONS_DIR)" in body_src, body_src
    assert "SESSIONS_DIR.glob(" not in body_src, (
        "the raw whole-directory glob was #4447's drift (a): the shared listing is "
        "one enumeration for every session count in this module")
    assert "is_session_head(" not in body_src, (
        "the hand-rolled skip is what session_head_files replaced")
    assert "scan_session_heads" not in body_src, (
        "the bounded walk is for windows; this brief counts the directory")
    assert "SESSION_SCAN_CEILING" not in body_src, body_src
    assert re.search(r"\bceiling\b", body_src, re.IGNORECASE) is None, body_src
    assert not [n for n in ast.walk(fdef) if isinstance(n, ast.Break)], (
        "a break is how a budget enters a loop quietly")
    assert not [n for n in ast.walk(fdef) if isinstance(n, ast.Slice)], (
        "a slice is how a window enters a count quietly")


def test_the_inner_voice_brief_docstring_records_the_ceilingless_ruling():
    """Clause 4: the docstring states the ruling in words a later reader can grep,
    and names the condition under which it may be reopened.

    Drift (b) of #4447 was that the decision lived only in #4267's owed entry:
    `architecture/mission-control.md` records the ceiling for the Background and Chat
    briefs and never mentioned this one, which is exactly why the next reader of a
    9,700-file loop reaches for a budget. So the docstring is graded here as text, on
    the four things it must say: what the two keys are, that the walk is
    deliberately ceilingless, who calls it, and when a ceiling becomes acceptable.
    """
    import inspect

    from app.routers import mc_ui

    doc = inspect.getdoc(mc_ui._summarize_inner_voice) or ""
    for phrase in ("sessions_total", "iv_enabled_count",
                   "full-directory denominator", "ceilingless",
                   "POST /api/mc/navigate"):
        assert phrase in doc, (phrase, doc)
    assert "Reopen condition" in doc, doc
    assert "window" in doc and "head count" in doc, (
        "the reopen condition must say when a ceiling becomes acceptable: "
        "once a caller wants a window and sessions_total stops being the "
        "directory's head count")
