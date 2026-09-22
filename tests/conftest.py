"""Test isolation from live runtime state.

The suite must not depend on the state of the running system. Two things
leaked in and broke tests that had nothing to do with them:

  * `config.yaml knowledge_graph.write_enabled`, which a knowledge-graph
    rebuild sets to false — six fact-write tests started failing because a
    rebuild was in progress on the machine.
  * `app.kg_store`'s process-default store, which points at the live
    database unless a test configures it.

Both are forced to a known value here. A test that wants the other value
patches it explicitly.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _default_state_dirs_to_scratch() -> None:
    """No pytest run reads or writes the machine's automod, guardian or request-manifest state.

    `scripts.automod.state` and the guardian's `policy` resolve their state
    dir from the environment **at import**, and default to the production
    path. `gate._child_env` sets both variables so a candidate's tests cannot
    reach the production ledger — but that covers the gate's own pytest only.
    A round's model runs the suite from Bash with a plain environment, thirty
    times a day (31 launches across 22 turns on 2026-09-18), and every one of
    those addressed `~/.local/state/lloyd-automod` by default, relying on each
    test to patch each path it touches. One did not: a `land()` whose leaked
    SIGTERM handler wrote `land_failed` for the fixture round `SM_L` into the
    live ledger, twice in one day, when the model `pkill`ed its own run.

    Set here, at conftest import, because that is the one moment that is
    before every test module's `from scripts.automod import state`. A caller
    that already chose a state dir (the gate, `review_tools`) keeps it.

    `LLOYD_MANIFEST_STORE` joined that list on 2026-09-19, for the same reason in
    a sharper form: `app/component_manifest.py` (#581) writes one NDJSON line per
    model request to `~/.local/state/lloyd-request-manifests` by default, and the
    first full-suite gate run that had the module in its tree put **48 fabricated
    lines** from fixture strings into that directory (`app/harness/finalizer.py`
    38, `app/secondary_models.py` 6, `app/harness/client.py` 4). The store is the
    artifact #581 exists to produce, and the clause that closes it is read off it
    — "24 h of mixed traffic read off the live store: zero manifest-write errors"
    — so a suite that writes there is a suite that corrupts its own acceptance
    evidence. Tests that own a store point the variable at their own `tmp_path`,
    as the three `test_component_manifest*`/`test_prompt_diff` files already do.

    `LLOYD_DAILY_NOTE_DIR` joined on 2026-09-21 for the same reason one level up
    the tree: `autonomy._append_fast_failure_alert` (#1209) appends a line to
    today's note under `~/obsidian/memory/` by default, and an existing
    failure-backoff test already drives a task through three sub-second failures
    without caring where an alert might go. Measured with this guard removed and
    the variable aimed at an empty directory: the scheduler, timeout, silence and
    grant-policy suites (173 tests, all passing) wrote one `2026-09-21.md`
    carrying one `Autonomy #` alert line, 387 bytes. A fixture writing "Autonomy
    #N failed 3 times in a row" into the live note is the same class of pollution
    as the manifest lines above, and worse, because the note is the surface a
    human reads to decide whether the fleet is healthy. Tests that assert on the
    alert point it at their own `tmp_path`.
    """
    scratch: Path | None = None
    for var, sub in (("LLOYD_AUTOMOD_STATE", "automod"), ("LLOYD_GUARDIAN_STATE", "guardian"),
                     ("LLOYD_MANIFEST_STORE", "request-manifests"),
                     ("LLOYD_DAILY_NOTE_DIR", "daily-notes")):
        if os.environ.get(var):
            continue
        if scratch is None:
            scratch = Path(tempfile.mkdtemp(prefix="lloyd-test-state-"))
            atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        (scratch / sub).mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(scratch / sub)


_default_state_dirs_to_scratch()


@pytest.fixture(autouse=True)
def _no_voice_alerts_in_tests(monkeypatch):
    """The guardian's sixth channel synthesises speech and plays it aloud.

    `Notifier.alert` fans out to it like any other channel, so without this an
    ordinary `pytest tests/` would talk to the room — and, worse, would do it
    from a detached process that outlives the test. Muted for every test; the
    ones that exercise the channel assert on the dispatch decision instead.
    """
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "0")


@pytest.fixture(autouse=True)
def _no_desktop_or_journal_alerts_in_tests(monkeypatch):
    """The other two channels that reach the room from a test process.

    `Notifier`'s `external` gate suppresses vault notes and backlog tasks, and
    was added after drill rollbacks polluted the live vault on 2026-09-06. It
    never covered the toast or the journal line, because those need nothing
    but a session bus — so a test that builds a default `Notifier` and only
    cares whether *voice* dispatched still paints the user's screen.

    That is what happened on 2026-09-07: every self-mod gate run toasted
    `Lloyd guardian: real rollback / body` and wrote
    `STILL BROKEN :: 2026-09-06 liveness failed` to the live journal at
    priority 2, from the fixture strings in `test_guardian_speak.py` and
    `test_guardian_predicates.py`. A fake critical incident in the journal is
    worse than a stray toast: it is the record you consult *after* an
    incident, and it now contains fiction.

    DBUS is blanked as well as the switch flipped, so the nag unit's
    bash `notify-send` fallback — which reads the environment, not
    `LLOYD_DESKTOP_ALERTS` — cannot fire from a test either. Tests that assert
    on the channel's dispatch decision set these explicitly.
    """
    monkeypatch.setenv("LLOYD_DESKTOP_ALERTS", "0")
    monkeypatch.setenv("LLOYD_JOURNAL_ALERTS", "0")
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)


@pytest.fixture(autouse=True)
def _promoter_cannot_reach_the_live_backend(monkeypatch):
    """No test asks the RUNNING backend whether a round is in flight.

    `round.land` waits for the other round's turn by reading the live pool
    (`promote.wait_for_rounds`), up to 75 minutes. A test that reaches it
    unpatched is then a test whose duration depends on what production is
    doing — it hung the first time it ran beside a real round (2026-09-17).
    The discard port refuses at once, which reads as "pool state unreadable"
    and returns. A test about the promoter's HTTP patches `_get`, as before.

    One unreadable poll is enough here: production waits out
    `ROUNDS_UNREADABLE_POLLS` of them, a minute, which no test should spend.
    And the aggregator is as unreachable as the backend, so
    `promote.restart_needed` never asks the LIVE services what they have
    loaded — it fails closed to "restart", the landing every older test means.
    """
    try:
        from scripts.automod import promote
        monkeypatch.setattr(promote, "BACKEND", "http://127.0.0.1:9")
        monkeypatch.setattr(promote, "MCP_HEALTH", "http://127.0.0.1:9/health")
        monkeypatch.setattr(promote, "ROUNDS_UNREADABLE_POLLS", 1)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_automod_lock(tmp_path, monkeypatch):
    """No test reads the machine's automod lock to decide anything.

    `autocode._loop_is_free` asks whether a landing holds that lock before it
    will call anything free, so read unpatched, every `(True, "free")` the
    suite asserts would depend on whether a real landing happened to be in
    flight while the test ran — and on this box one usually is. Same shape of
    hazard as the fixture above (a live process's state reaching into a test's
    verdict), different mechanism and different file.

    It redirects rather than stubs: the lock file still exists and `S.Lock`
    still takes a real `flock` on it, so lock behaviour is tested on the real
    class, only not on production's file. A test that wants to *be* the holder
    constructs `S.Lock(its_own_path)`."""
    from scripts.automod import state as S
    monkeypatch.setattr(S, "LOCK_PATH", tmp_path / "lloyd-automod" / "lock")


@pytest.fixture(autouse=True)
def _writes_enabled_in_tests(monkeypatch):
    """Fact writes are on unless a test says otherwise."""
    try:
        from agent_mcp import facts as facts_mod
    except Exception:          # module not importable in this test's env
        return
    monkeypatch.setattr(facts_mod, "_writes_enabled", lambda: True, raising=False)


@pytest.fixture(autouse=True)
def _isolate_default_store(request, tmp_path_factory):
    """No test writes to the live knowledge-graph store.

    Tests that need a store call `kg_store.configure(...)` themselves; this
    only guarantees the *default* never resolves to the production file, and
    puts it back afterwards.
    """
    from app import kg_store
    original = kg_store._default_path
    kg_store.reset()
    kg_store._default_path = tmp_path_factory.mktemp("kg") / "kg.sqlite"
    yield
    kg_store.reset()
    kg_store._default_path = original


@pytest.fixture(autouse=True)
def _isolate_research_store(tmp_path_factory):
    """No test writes to the live research registry.

    Same shape and same reason as `_isolate_default_store` above: the default
    must never resolve to `~/lloyd/research.db`, because a test that proposes
    a topic would otherwise put it in front of the deep-research worker. Tests
    that want a registry call `research_store.configure(...)` themselves.
    """
    from app import research_store
    original = research_store._default_path
    research_store.reset()
    research_store._default_path = tmp_path_factory.mktemp("research") / "research.db"
    yield
    research_store.reset()
    research_store._default_path = original


@pytest.fixture(autouse=True)
def _isolate_background_records(tmp_path_factory, monkeypatch):
    """No test writes a session or an event log into the live tree.

    Recording became universal on 2026-09-10: every autonomy run and every
    `run_prompt_on_primary` worker turn now writes a session JSON and an event
    log. A test that drives one of those paths — and several do, for reasons
    that have nothing to do with recording — would otherwise leave a real
    transcript in `~/lloyd/sessions/`, where it shows up in the history list,
    in session recall and in the retention sweep.

    Same shape as the two isolators above: the *default* never resolves to the
    production directory. A test that patches these itself still wins, because
    its `monkeypatch` runs after this fixture's.
    """
    root = tmp_path_factory.mktemp("lloyd-records")
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", root / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", root / "event_logs")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", root / "event_logs" / "blobs")


@pytest.fixture(autouse=True)
def _isolate_usage_store(tmp_path_factory, monkeypatch):
    """No test writes a usage row into the live `usage.db`.

    `usage_store.DB_PATH` resolves from `__file__`, so in `~/lloyd` it is the
    production database behind the dashboard's token panel and its
    prefix-miss counter. Until 2026-09-10 only the chat path wrote rows; the
    background recorder writes one per run now, and several tests drive it.
    `usage_store._conn` reopens when `DB_PATH` moves, so the per-thread
    connection cache cannot carry a test's writes into the next one's file.
    """
    monkeypatch.setattr("usage_store.DB_PATH",
                        tmp_path_factory.mktemp("usage") / "usage.db")


@pytest.fixture(autouse=True)
def _isolate_backlog_dedupe(tmp_path_factory, monkeypatch):
    """No test appends to the live dedupe log or queries the live qmd daemon.

    `backlog_write_task` runs the write-time dedupe on every create, and its
    log path is a `Path.home()` literal. Tests that patched `BACKLOG_DIR` and
    nothing else — `test_backlog_okf_frontmatter.py`, `test_backlog_tags_shape.py`
    — wrote a real row per run and POSTed their fixture text to the daemon. On
    2026-09-13, 656 of `dedupe.jsonl`'s 1004 rows were fixtures ("A newly
    written task" 498, "Filed by a digest run" 158): the log that exists to
    tune the merge threshold was two-thirds noise.

    `dedupe_config` is pinned to the defaults so `config.yaml` cannot flip
    `merge` under a test. Test-local patches still win — their `monkeypatch`
    runs after this one — which is how `test_backlog_dedupe.py` feeds rows in.
    """
    try:
        from agent_mcp import backlog_similar as SIM
    except Exception:          # module not importable in this test's env
        return
    monkeypatch.setattr(SIM, "DEDUPE_LOG",
                        tmp_path_factory.mktemp("dedupe") / "dedupe.jsonl")
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: [])
    monkeypatch.setattr(SIM, "dedupe_config", lambda: dict(SIM.DEFAULTS))


@pytest.fixture(autouse=True)
def _recall_ranked_by_qmd_in_tests(monkeypatch):
    """No test's recall reaches the LIVE djev on GPU 2 unless it asks to (#1336).

    `agent_mcp.vault.recall_reranker()` is "djev" whenever `djev.enabled` is on in
    the live config, and a test that stubs qmd's reply with two rows or more would
    then send a real ranking to :8011 — production load and a result that depends
    on a GPU. Pinned to the cross-encoder path the existing recall tests were
    written against; `tests/test_recall_djev_ranker.py` opts back in, with djev
    stubbed.
    """
    import agent_mcp.vault as _vault
    monkeypatch.setattr(_vault, "RECALL_RERANKER", "qmd")
    yield


@pytest.fixture(scope="session", autouse=True)
def _isolate_djev_shadow(tmp_path_factory):
    """No test appends a row to the log the djev floors are calibrated from.

    `~/.local/state/lloyd-djev/shadow.jsonl` is what `eval/djev/replay.py
    --floors` reads, and step 5 of `architecture/djev.md` §9.3 is "let the
    shadow rows accumulate, then set the floor from them" — a corpus that is
    mostly fixtures yields a floor set from fixtures. Its three paths are
    `Path.home()` literals bound at import (`app/djev_shadow.py:68-71`), and
    three seams (`agent_mcp/backlog.py:529`, `agent_mcp/vault.py:1289`,
    `scripts/memory/entity_semantic_gate.py:225`) call the recorder from code
    that has no idea a test is running. Re-measured 2026-09-21 08:19Z over the
    live log: 36 of its 44 `dedupe` rows carried one of two fixture titles and
    27 of its 32 `rerank` rows carried an empty `meta`, in a file six hours
    old — nothing decays, `_write()` only appends.

    SESSION scope, not per test, because the recorder's worker is a daemon
    thread that drains whenever it likes and `_write()` re-reads `SHADOW_LOG`
    at write time (`:295`). `tests/test_djev_shadow.py` has patched these names
    per test since the recorder landed and still leaked: after its `monkeypatch`
    tears down, the module points at the home path again and the row arrives
    there. Its own per-test patch still wins while the test runs — a narrower
    patch applied later simply covers it — and restores TO this one afterwards.

    **And deliberately no teardown, for the same reason.** The first cut of
    this fixture restored the three names on session teardown and still left a
    row in `$HOME/.local/state/lloyd-djev/shadow.jsonl` after
    `HOME=$FH pytest tests/test_backlog_dedupe.py tests/test_backlog_spawn_loop.py`:
    neither module calls `flush()`, so a job whose djev read outlasts pytest's
    teardown lands wherever the module points then, and restoring put the home
    path back exactly when a drain was in flight. Nothing reads these globals
    after the session ends — the process exits — so the restore bought nothing
    and cost the isolation. Measured after dropping it: the file is not created
    at all.

    Paths, not `LLOYD_DJEV_SHADOW=0`, because that variable is itself asserted
    on: `test_djev_rerank_arm.py::test_the_eval_mutes_the_shadow_recorder`
    reads it out of the process to prove `eval/run_eval.py:35` set it, and an
    ambient mute makes that assertion hold with the line under test deleted. A
    redirected path silences the same writes without touching the switch.

    All three globals move together, because they are three independent leaks:
    `_write()` opens `SHADOW_LOG`, `_record_pending_drops()` writes
    `PENDING_DROPS`, and both `mkdir` `STATE_DIR`.

    Pinned by `tests/test_djev_shadow_isolation.py`, whose child-pytest node is
    #1324's reproduction (`HOME=$FH pytest tests/test_backlog_dedupe.py
    tests/test_backlog_spawn_loop.py` must leave no shadow log in `$FH`) run as
    an assertion, so the automod gate's own full-suite run — which inherits
    `HOME` from `_child_env` — is covered by the same property.
    """
    try:
        from app import djev_shadow
    except Exception:          # module not importable in this test's env
        return
    root = tmp_path_factory.mktemp("djev-shadow")
    djev_shadow.STATE_DIR = root
    djev_shadow.SHADOW_LOG = root / "shadow.jsonl"
    djev_shadow.PENDING_DROPS = root / "dropped_at_shutdown.json"


@pytest.fixture
def worker_turn_post(monkeypatch):
    """Capture the body of a worker's loopback POST, with no backend to POST to.

    `workers.sources._common.run_prompt_in_session` is the one entry every
    session-backed source uses (`deep-research`, `youtube-digest`, `autotriage`,
    `autocode`), and what it puts in `payload["text"]` is the whole of the
    instruction the turn gets. A test cannot assert that from a stubbed
    `run_prompt_in_session` — the source's own test replaces that function, so
    anything it does downstream is invisible there. This swaps only the
    transport: the function runs unpatched through its payload build, and the
    list returned is the JSON body in request order.

    Not autouse: it replaces `httpx.AsyncClient` globally for the test, which
    is exactly what a test that is verifying a real POST wants and what every
    other test would choke on. `tests/test_session_platform_checks.py` owns the
    variant that crosses the wire for real.
    """
    import httpx

    captured: list[dict] = []

    class _Streamed:
        """A finished turn, in the SSE shape `_aiter_sse` parses."""

        status_code = 200

        async def aread(self) -> bytes:
            return b""

        async def aiter_lines(self):
            yield "event: done"
            yield 'data: {"response": "stubbed", "stop_reason": "end_turn", ' \
                  '"num_turns": 1}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            captured.append(dict(json or {}))
            return _Streamed()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return captured
