"""The properties added after the 2026-09-06 review of the self-modification loop.

Every test here pins a failure that was reachable in the shipped design and
that no existing test could see. They are grouped by the question they answer:

  * can the loop be driven by the AGENT, not just by a human at a terminal?
  * does a rollback destroy work it was never asked to judge?
  * can the watchdog survive performing its own rescue?
  * does a check that reports "healthy" have any way to say otherwise?

The common shape of the bugs below is a mechanism that looks like it is doing
something and is structurally incapable of it — a knob nothing reads, a
detector keyed on state that has already been deleted, a floor satisfied by a
suite that ran nothing. Tests that only exercise the happy path cannot find
those, so most of what follows asserts a negative.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import detect      # noqa: E402
import gstate      # noqa: E402
import policy      # noqa: E402
import rollback as rb  # noqa: E402

from scripts.automod import state as S  # noqa: E402


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)


@pytest.fixture()
def isolated_state(tmp_path, monkeypatch):
    """Point the state module at a scratch dir. Never the live one — on the
    way OUT as well as on the way in.

    The teardown used to `delenv` and reload. A reload re-reads the
    environment, so with the variable deleted `S.STATE_DIR` came back as
    `~/.local/state/lloyd-automod` and stayed there for **every test that ran
    after this file**, the gate's tests rung included: `gate._child_env` sets
    `LLOYD_AUTOMOD_STATE` precisely so candidate tests cannot reach the
    production ledger, and this fixture unset it one file into the run. From
    2026-09-09 to 2026-09-18 that was latent, then it cost twice in a day —
    `land_failed` rows for a fixture round (`SM_L`) in the live ledger, and
    `test_a_landing_owns_its_marker…` spinning in `round._land_lock` for as
    long as PRODUCTION had a promotion under observation (822 s in one run).
    Restore what was there; `tests/conftest.py` guarantees something was.
    """
    import importlib
    before = os.environ.get("LLOYD_AUTOMOD_STATE")
    monkeypatch.setenv("LLOYD_AUTOMOD_STATE", str(tmp_path / "state"))
    importlib.reload(S)
    S.ensure_dirs()
    yield tmp_path / "state"
    if before is None:
        monkeypatch.delenv("LLOYD_AUTOMOD_STATE", raising=False)
    else:
        monkeypatch.setenv("LLOYD_AUTOMOD_STATE", before)
    importlib.reload(S)


@pytest.fixture()
def repo(tmp_path):
    """A repo with parent A, promotion B, and a later human commit C."""
    r = tmp_path / "lloyd"
    (r / "app").mkdir(parents=True)
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "app" / "mod.py").write_text("VALUE = 'A'\n", encoding="utf-8")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "A")
    a = git(r, "rev-parse", "HEAD").stdout.strip()

    (r / "app" / "mod.py").write_text("VALUE = 'B'\n", encoding="utf-8")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "B: the promotion")
    b = git(r, "rev-parse", "HEAD").stdout.strip()

    # A nightly job commits straight to live `main` during the window.
    (r / "app" / "nightly.py").write_text("KEEP = True\n", encoding="utf-8")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "C: unrelated nightly work")
    c = git(r, "rev-parse", "HEAD").stdout.strip()
    return {"path": r, "a": a, "b": b, "c": c}


# ===========================================================================
# 1. The loop has to survive being driven by the agent
# ===========================================================================

def test_a_detached_child_escapes_its_parents_process_group(tmp_path):
    """The single defect that made the autonomous path unusable.

    A landing restarts lloyd-mcp, and when driven from an MCP tool the
    promoter runs INSIDE lloyd-mcp. Both supervisor confs set
    `stopasgroup`/`killasgroup`, so supervisord signals the whole process
    group — killing the promoter partway through its own restart. The tree is
    already fast-forwarded, the aggregator is stopped, the backend serves old
    code with no tools, and `current.json` is stuck in `landing`, which the
    guardian is explicitly told to ignore. Nothing watches it, nothing reverts.

    A new session is precisely what a process-group signal cannot reach.
    """
    log = tmp_path / "child.log"
    pid = S.spawn_detached([sys.executable, "-c", "import time; time.sleep(30)"], log)
    try:
        assert os.getsid(pid) != os.getsid(0), "child shares our session; a group kill takes it"
        assert os.getpgid(pid) != os.getpgid(0), "child shares our process group"
    finally:
        os.kill(pid, 9)


def test_the_detached_child_survives_a_group_kill(tmp_path):
    """The same property, demonstrated the way supervisord would do it."""
    log = tmp_path / "child.log"
    pid = S.spawn_detached([sys.executable, "-c", "import time; time.sleep(30)"], log)
    try:
        os.killpg(os.getpgid(0), 0)          # our group exists and is signalable
        assert os.getpgid(pid) != os.getpgid(0)
        os.kill(pid, 0)                       # still alive, in its own group
    finally:
        os.kill(pid, 9)


def test_the_master_switch_covers_the_cli_not_only_the_tool(isolated_state, tmp_path):
    """`automod.enabled` was checked in the MCP wrapper alone.

    The skill's own worked examples drive `python -m scripts.automod.round`,
    so "the loop ships inert" was true of the surface nobody used and false of
    the one they did.
    """
    (tmp_path / "config.yaml").write_text("automod:\n  enabled: false\n", encoding="utf-8")
    assert S.is_enabled(tmp_path) is False
    with pytest.raises(S.AutomodDisabled):
        S.require_enabled("land a round", tmp_path)

    (tmp_path / "config.yaml").write_text("automod:\n  enabled: true\n", encoding="utf-8")
    assert S.is_enabled(tmp_path) is True
    S.require_enabled("land a round", tmp_path)          # does not raise


def test_a_missing_config_reads_as_disabled(tmp_path):
    """Fail closed. An unreadable config must not enable a live promotion."""
    assert S.is_enabled(tmp_path / "nope") is False


def test_worker_turns_cannot_drive_the_loop():
    """`domain-research` reads arbitrary web pages into its context.

    The automod tools were advertised to every worker prompt, so the machinery
    that rewrites production sat one prompt injection away from a source whose
    whole job is ingesting untrusted text. The backlog worker was told not to
    use them in its PROMPT, which is not a control.
    """
    src = (ROOT / "workers" / "sources" / "_common.py").read_text()
    for tool in ("automod_start", "automod_gate", "automod_land", "automod_rollback"):
        assert tool in src, f"{tool} is not disallowed for worker turns"


def test_subagents_cannot_drive_the_loop():
    """A Task runs inside the aggregator a landing restarts."""
    src = (ROOT / "agent_mcp" / "builtin_task.py").read_text()
    for tool in ("automod_start", "automod_land", "automod_rollback"):
        assert tool in src, f"{tool} is not disallowed for subagents"


def test_the_drain_endpoint_is_loopback_only():
    """Its only legitimate caller is the promoter, on this box.

    `server.py`'s auth middleware enforces the client-cert allowlist only when
    a fingerprint is actually forwarded, so on the tailnet this was an
    unauthenticated way to make the backend refuse every turn for 10 minutes.
    """
    src = (ROOT / "app" / "routers" / "automod.py").read_text()
    assert "_is_loopback" in src and "loopback-only" in src


# ===========================================================================
# 2. A rollback must not destroy work it was never asked to judge
# ===========================================================================

def test_a_surgical_revert_keeps_later_commits(repo):
    """The 26-commit incident, one level down.

    There the wrong TARGET was chosen. Here the right target is reached by the
    wrong ROUTE: `reset --hard` to the promotion's parent is only correct
    while HEAD still IS the promotion, and nightly jobs commit straight to
    live `main` during the 15-minute window.
    """
    r = repo["path"]
    new_head = rb.revert_commit(str(r), repo["b"])

    assert new_head not in (repo["a"], repo["b"], repo["c"])
    # The promotion is undone...
    assert (r / "app" / "mod.py").read_text() == "VALUE = 'A'\n"
    # ...and the unrelated nightly work is still there.
    assert (r / "app" / "nightly.py").exists(), "a rollback ate work it never judged"
    assert rb.head_branch(str(r)) == "refs/heads/main"
    assert rb.commits_between(str(r), repo["c"], new_head) == 1


def test_a_conflicting_revert_raises_rather_than_guessing(repo):
    """Overlapping later work has no safe automatic answer, so escalate."""
    r = repo["path"]
    (r / "app" / "mod.py").write_text("VALUE = 'C-touched-the-same-line'\n", encoding="utf-8")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "D: conflicts with the revert")
    with pytest.raises(rb.RollbackError):
        rb.revert_commit(str(r), repo["b"])
    # And it must not leave a revert half-applied.
    assert not git(r, "status", "--porcelain").stdout.strip()


def test_the_denylist_catches_the_same_change_under_a_new_sha(isolated_state, repo):
    """Keying on the SHA alone was never going to catch anything.

    A round that is re-cut produces a new SHA for identical content and sails
    straight past. The content hash of the touched paths is the half that
    implements "do not re-land this change".
    """
    r = repo["path"]
    h_b = S.changed_tree_hash(r, repo["b"], ["app/mod.py"])
    assert h_b

    S.deny(repo["b"], tree_hash=h_b)
    assert S.is_denied(commit=repo["b"])

    # Re-derive the identical change on a fresh branch: new SHA, same content.
    git(r, "checkout", "-q", repo["a"])
    (r / "app" / "mod.py").write_text("VALUE = 'B'\n", encoding="utf-8")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "B again, re-derived")
    redone = git(r, "rev-parse", "HEAD").stdout.strip()

    assert redone != repo["b"]
    assert S.is_denied(commit=redone) is False, "different SHA, as expected"
    assert S.is_denied(tree_hash=S.changed_tree_hash(r, redone, ["app/mod.py"])) is True


def test_a_tree_hash_that_git_cannot_answer_is_not_a_denial(isolated_state, repo):
    """Refusing to promote because git hiccuped is worse than the bug."""
    assert S.changed_tree_hash(repo["path"], repo["b"], []) is None
    assert S.changed_tree_hash(repo["path"], "0" * 40, ["app/mod.py"]) is None
    assert S.is_denied(tree_hash=None) is False


def test_the_writer_drain_ignores_readers_and_log_holders(tmp_path):
    """The old predicate was "cwd is under the repo", true of every shell.

    It was therefore never empty: the drain burned its full 20s on every
    rollback and logged 44 pids that meant nothing, so the one thing it
    existed to detect was invisible inside the noise. Scoping matters as much
    as the fd check — supervisord holds `logs/server.err` open for append
    forever, so an unscoped scan just swaps one always-true predicate for
    another.
    """
    repo = tmp_path / "lloyd"
    (repo / "app").mkdir(parents=True)
    (repo / "logs").mkdir()
    (repo / "app" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "logs" / "server.err").write_text("", encoding="utf-8")
    watch = ("app", "agent_mcp", "workers", "scripts", ".git")
    # skip_pids=() so this process can be its own subject; production excludes
    # the guardian itself, which holds the repo open to tag/stash/reset.
    def holders():
        return rb.repo_write_holders(str(repo), watch, skip_pids=())

    with open(repo / "app" / "mod.py", "r"):                     # a READER
        assert os.getpid() not in holders(), "a reader cannot corrupt a checkout"
    with open(repo / "logs" / "server.err", "a"):                # a LOG writer
        assert os.getpid() not in holders(), "supervisord holds this open forever"
    with open(repo / "app" / "mod.py", "a"):                     # a CODE writer
        assert os.getpid() in holders(), "the one case the drain exists for"


# ===========================================================================
# 3. The watchdog has to survive performing its own rescue
# ===========================================================================

def test_wait_healthy_beats_the_watchdog_while_it_waits():
    """WATCHDOG=1 was sent once per loop iteration, and a rollback IS one.

    Two blocking stops plus a 90s health wait comfortably exceeds the unit's
    WatchdogSec=90, so systemd killed the guardian mid-rescue, restarted it,
    and the resume path began the same rollback again — a loop that can never
    reach BROKEN, in exactly the case BROKEN exists to report.
    """
    import probes
    beats = []
    ok, _ = probes.wait_healthy("http://127.0.0.1:1/nothing", 2.0, 0.2,
                                interval=0.2, on_tick=lambda: beats.append(1))
    assert ok is False
    assert beats, "no watchdog ping during the wait"


def test_the_beat_is_tied_to_progress_not_a_background_thread():
    """A thread pinging on its own would defeat the point: a HUNG guardian
    would keep the watchdog satisfied forever."""
    src = (GUARDIAN_DIR / "guardian.py").read_text()
    assert "def _beat" in src
    assert "threading" not in src, "a heartbeat thread would hide a hung guardian"


def test_a_stale_rollback_request_is_discarded_not_obeyed():
    """It reasoned about state that has since moved on."""
    src = (GUARDIAN_DIR / "guardian.py").read_text()
    assert "ROLLBACK_REQUEST_MAX_AGE_SECONDS" in src
    assert policy.ROLLBACK_REQUEST_MAX_AGE_SECONDS > 0


def test_a_rollback_request_round_trips(isolated_state):
    S.request_rollback(reason="because", trigger="regression",
                       target="a" * 40, commit="b" * 40,
                       changed_paths=["app/x.py"])
    back = S.read_rollback_request()
    assert back["target"] == "a" * 40 and back["commit"] == "b" * 40
    assert back["changed_paths"] == ["app/x.py"]
    S.clear_rollback_request()
    assert S.read_rollback_request() is None


def test_nothing_inside_the_blast_radius_rolls_back_inline():
    """The backend and the aggregator are both STOPPED by a rollback.

    Calling it inline meant issuing the stop that kills your own caller and
    then never reaching `git reset`: the stack goes down and the tree does not
    move, which is worse than either doing it or not doing it.
    """
    for rel in ("agent_mcp/automod.py", "workers/sources/automod_regression.py"):
        src = (ROOT / rel).read_text()
        assert "_rollback_inline(" not in src, f"{rel} still reverts inline"
        assert "request_rollback" in src, f"{rel} does not hand the rollback over"


# ===========================================================================
# 4. Predicates that could not say "unhealthy"
# ===========================================================================

def _running(**kw):
    d = {"statename": "RUNNING", "start": 0}
    d.update(kw)
    return d


def _down(**kw):
    base = dict(now=1e9, grace=1.0, probe_fail_streak=0, probe_threshold=3,
                start_history=[], crash_loop_starts=3, crash_loop_window=180.0)
    base.update(kw)
    info = base.pop("info", _running())
    return detect.process_down(info, **base)


def test_a_degraded_aggregator_is_not_a_dead_one():
    """A 503 from /health means "some module is degraded" and it ANSWERED.

    Counting that as a refused connection gave a closed Thunderbird bridge
    three ticks to look like a dead aggregator, while the careful
    newly-degraded-since-LKG check sat after the down predicate and never got
    a vote.
    """
    down, why = _down(probe_http_streak=3, probe_http_threshold=policy.PROBE_HTTP_ERROR_STREAK)
    assert down is False, why


def test_a_backend_parked_in_degraded_eventually_counts():
    """It is alive, but a router that never mounted is a real bad promotion."""
    down, why = _down(probe_http_streak=policy.PROBE_HTTP_ERROR_STREAK,
                      probe_http_threshold=policy.PROBE_HTTP_ERROR_STREAK)
    assert down is True and "non-200" in why


def test_the_three_probe_budgets_are_ordered_by_what_they_mean():
    """refused < timeout < http-error: nothing listening, busy, answering badly."""
    assert policy.PROBE_FAIL_STREAK < policy.PROBE_TIMEOUT_STREAK
    assert policy.PROBE_TIMEOUT_STREAK < policy.PROBE_HTTP_ERROR_STREAK


def test_our_own_quarantine_stop_is_not_an_outage():
    """Flap protection stops the backend on purpose.

    Reading that back as death produced a "service down" alert every 15
    minutes for as long as the quarantine lasted.
    """
    down, _ = _down(info={"statename": "STOPPED", "start": 0}, intentional_stop=True)
    assert down is False


def test_exited_is_never_excused_by_an_intentional_stop():
    """STOPPED is somebody's deliberate stop; EXITED is the process ending."""
    down, why = _down(info={"statename": "EXITED", "start": 0}, intentional_stop=True)
    assert down is True and "EXITED" in why


def test_the_chronic_set_expires():
    """Learned once and kept forever, it describes the box as it was on first
    boot — so every recurring error that starts later is "novel" indefinitely
    and the next promotion is reverted for a failure it did not cause."""
    assert policy.CHRONIC_REFRESH_SECONDS > 0
    src = (GUARDIAN_DIR / "guardian.py").read_text()
    assert "_chronic_built_ts" in src
    assert "_chronic_loaded" not in src, "the load-once guard makes the TTL unreachable"


# ===========================================================================
# 5. Checks satisfied by having done nothing
# ===========================================================================

def test_the_test_rung_requires_tests_to_have_RUN():
    """`pytest -q` exits 0 having collected everything and run none of it.

    A collected-count floor alone is satisfied by a suite that skipped itself
    wholesale, which is one conftest line away.
    """
    from scripts.automod import gate as G
    counts = G._parse_pytest_summary("collected 1632 items\n\n1632 skipped in 2s")
    assert counts["collected"] == 1632 and counts["passed"] == 0
    assert counts["collected"] >= G.PYTEST_MIN_COLLECTED, "the old floor passes this"
    assert counts["passed"] < G.PYTEST_MIN_PASSED, "the new floor must not"


def test_skips_are_parsed_at_all():
    from scripts.automod import gate as G
    assert G._parse_pytest_summary("1600 passed, 12 skipped in 30s")["tests_skipped"] == 12


def test_the_skipped_test_count_does_not_masquerade_as_a_skipped_rung():
    """`_rung` records a rung as skipped from `data["skipped"]`, and
    `rung_tests` returns the pytest counts as its data — so a suite with three
    skipped tests recorded the whole `tests` rung as skipped on the ledger,
    where `scorecard.py` reads it. A round that ran its entire suite was
    indistinguishable from one that never ran it.
    """
    from scripts.automod import gate as G
    counts = G._parse_pytest_summary("1600 passed, 12 skipped in 30s")
    assert "skipped" not in counts, "the count must not collide with the flag"
    assert counts["tests_skipped"] == 12


def test_gate_rungs_run_candidate_code_against_scratch_state():
    """static/tests/venv ran candidate code against the LIVE state dir.

    A candidate test that forgot its isolation fixture could write a real
    BROKEN or promotions-halted flag, or append to the production audit trail,
    from inside the gate that is supposed to be read-only judgment.
    """
    from scripts.automod import gate as G
    g = G.Gate.__new__(G.Gate)
    g.round_id = "SM_TEST"
    g.worktree = Path("/tmp/nonexistent-worktree")
    env = G.Gate._child_env(g)
    assert "/lloyd-automod" not in env["LLOYD_AUTOMOD_STATE"], "points at live state"
    assert env["LLOYD_VOICE_ALERTS"] == "0"
    assert env["LLOYD_GUARDIAN_STATE"] != env["LLOYD_AUTOMOD_STATE"]


def test_the_idle_gate_can_see_turns_that_never_touch_a_session_queue():
    """Worker jobs call run_query directly.

    A landing that restarts the backend during a ten-minute research job kills
    it, and the ConnectErrors it logs on the way down land inside the window
    the error-rate detector is watching — so the promotion is reverted for the
    damage its own landing caused. That is the 2026-09-06 20:14 signature.
    """
    from app.harness.loop import active_run_count, _run_started, _run_finished
    from app.sessions_io import active_turn_summary

    before = active_run_count()
    _run_started()
    try:
        assert active_run_count() == before + 1
        assert active_turn_summary()["harness_runs"] >= 1
    finally:
        _run_finished()
    assert active_run_count() == before


def test_the_promoter_treats_a_harness_run_as_busy():
    src = (ROOT / "scripts" / "automod" / "promote.py").read_text()
    assert "harness_runs" in src, "wait_idle cannot see worker turns"


# ===========================================================================
# 6. State that has to outlive the thing that wrote it
# ===========================================================================

def test_settling_records_what_landed_and_what_it_replaced(tmp_path):
    """`current.json` is DELETED at settle.

    Anything asking "what landed recently, and what did it replace?" after the
    window closed had nothing to read — which is why the nightly quality check
    found no promotion under observation essentially always.
    """
    st = gstate.AutomodState(tmp_path)
    st.write_last_settled({"commit": "b" * 40, "parent": "a" * 40,
                           "settled_ts": time.time()})
    back = gstate.read_json(st.last_settled)
    assert back["parent"] == "a" * 40


def test_the_guardian_reads_the_eval_baseline_and_never_writes_it(tmp_path):
    """LKG has exactly one writer, and that is what makes it mean
    observed-healthy-in-production rather than measured-by-somebody."""
    st = gstate.AutomodState(tmp_path)
    gstate.write_json_atomic(st.eval_last, {"commit": "b" * 40, "overall": {"ndcg10": 0.5}})
    assert st.read_eval_last()["commit"] == "b" * 40
    assert not hasattr(st, "write_eval_last"), "the guardian must not write the measurement"

    st.set_lkg("b" * 40, eval_baseline=st.read_eval_last())
    assert gstate.read_json(st.lkg_path)["eval"]["overall"]["ndcg10"] == 0.5
    # A measurement of THIS commit reads as this commit's.
    assert gstate.read_json(st.lkg_path)["eval_for_recorded_commit"] is True


def test_the_measurement_states_which_axis_it_measured(tmp_path):
    """Clause 4 of #829, across the seam the number actually travels.

    The worker's measurement is handed to the guardian, folded into
    `last_known_good.json`'s `eval` slot, and served by `app/routers/automod.py` to
    Mission Control as the green mark beside a promotion. That slot carried seven
    query scores and nothing stating what they were scores OF, so a retrieval pass
    over pinned questions — an eval that issues no model request at all — could be
    read as evidence about behaviour. The payload is built by one function with one
    call site, so the field list is assertable without running an arm, and the
    assertion here runs the real fold into the record rather than stopping at the
    payload: a field the guardian dropped on the way in would satisfy a
    payload-only test and tell a reader nothing.
    """
    import workers.sources.automod_regression as R

    payload = R.eval_last_payload(
        commit="b" * 40, measured_at="2026-09-22T00:00:00Z", baseline_commit="a" * 40,
        stage="detached", current={"overall": {"ndcg10": 0.5}, "corpus": {}},
        pin={"queries_fingerprint": "deadbeef"}, regressed=False, reasons=[],
        latency_reading={"latency_ms_avg": 20.0}, latency_verdict=None)

    axis = payload[R.AXIS_FIELD]
    measures = axis["measures"].lower()
    assert "retrieval" in measures, axis
    assert "paired" in measures, "the paired A/B design is not stated"
    refused = " ".join(axis["does_not_measure"]).lower()
    for phrase in ("tool call", "turn count", "model decision"):
        assert phrase in refused, f"the payload never lists a {phrase} as unmeasured"
    assert "edge" in refused, "graph edge quality is not listed as uncovered"
    assert "prompt_surface" in refused, (
        "the uncovered-axis line must also say where the only loop-side check is")

    # Parity with the live gate, not just with the worker source. The worker's
    # limit block is asserted against `Gate.PROMPT_SURFACE_PATHS + PROMPT_SURFACE_
    # VAULT` in `test_automod_doc_claims`, so without this the constant shipped
    # into the record could sit on its own and drift from the trigger it describes.
    # It is partial by construction and stays that way: the gate also matches the
    # vault paths by BASENAME (`_touches_prompt_surface` tests `base` as well as
    # `path`), so no assertion here proves that every name a future rung adds was
    # written into the constant. What IS provable — and is the asymmetry worth
    # guarding — is that the record never lists a path the rung does not trigger
    # on, since a false "we did check that surface" is the misleading direction.
    from scripts.automod.gate import Gate

    surfaces = Gate.PROMPT_SURFACE_PATHS + Gate.PROMPT_SURFACE_VAULT
    assert len(surfaces) == 5, f"the five-path claim no longer matches the gate: {surfaces}"
    named = [name for name in surfaces if name.split("/")[-1].lower() in refused]
    assert len(named) == len(surfaces), (
        f"the payload omits a trigger the gate actually fires on: "
        f"{[n for n in surfaces if n not in named]}")
    for bogus in ("run_eval.py", "loop.py", "gate.py", "messages.py"):
        assert bogus not in refused, f"the payload claims {bogus} is a prompt surface"

    st = gstate.AutomodState(tmp_path)
    st.set_lkg("b" * 40, eval_baseline=payload)
    stored = gstate.read_json(st.lkg_path)["eval"]
    assert stored[R.AXIS_FIELD]["measures"] == axis["measures"], "the fold dropped the axis"
    assert stored["pin"]["queries_fingerprint"] == "deadbeef", (
        "a test that only looked at the axis would pass while the fold dropped "
        "the measurement beside it")


def test_a_carried_over_eval_says_so_in_the_record(tmp_path):
    """Clause 5 of #829: the slot must not be able to pose as this promotion's
    baseline.

    `guardian.py` folds a measurement only when it was taken for exactly the commit
    being recorded, and `set_lkg` otherwise keeps the previous slot — carrying is
    right, a promotion that has not been measured yet must not lose the measurement
    that exists. What was missing is that nothing said which of the two a reader
    was looking at. Read 2026-09-18T04:39Z the live record held commit 4a3ac776
    recorded 2026-09-17T23:51Z carrying an eval for fadbc235 measured
    2026-09-14T11:03Z: a three-day-old measurement of a different commit, in the
    slot every later "observed healthy in production" claim reads, unmarked.
    """
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("a" * 40, eval_baseline={"commit": "a" * 40,
                                        "measured_at": "2026-09-14T11:03:59Z",
                                        "overall": {"ndcg10": 0.5}})
    assert gstate.read_json(st.lkg_path)["eval_for_recorded_commit"] is True

    # The next promotion settles with no measurement of its own.
    st.set_lkg("b" * 40, health={"mcp_degraded_modules": []})
    rec = gstate.read_json(st.lkg_path)
    assert rec["commit"] == "b" * 40
    assert rec["eval"]["overall"]["ndcg10"] == 0.5, "the measurement must still be carried"
    assert rec["eval_for_recorded_commit"] is False, (
        "a carried-over eval reads as this promotion's baseline")
    assert rec["eval_commit"] == "a" * 40, "the record does not say whose measurement it holds"
    assert rec["eval_measured_at"] == "2026-09-14T11:03:59Z"


def test_a_record_with_no_measurement_says_it_was_never_measured(tmp_path):
    """The third case, which a `False` would swallow.

    An empty slot and a measurement of another commit are different facts — one is
    "nobody has measured this", the other is "here is a number from somewhere else"
    — and a reader who has to infer which one happened infers wrong. `None`, not
    `False`, and the field is present either way so the question can be asked of
    every record.
    """
    st = gstate.AutomodState(tmp_path)
    st.set_lkg("a" * 40)
    rec = gstate.read_json(st.lkg_path)
    assert rec["eval"] == {}
    assert "eval_for_recorded_commit" in rec, "a fresh record must answer the question too"
    assert rec["eval_for_recorded_commit"] is None
    assert rec["eval_commit"] is None


def test_the_bless_route_stamps_the_same_attribution_the_guardian_does(isolated_state,
                                                                      monkeypatch):
    """Clause 5 on the OTHER writer of the record.

    `bless` advances the pointer by hand when settle will not, and it writes through
    `scripts/automod/state.py:write_lkg`, not `gstate.set_lkg` — two writers, one
    file. A stamp only one of them computes would make the field answer "who wrote
    this" instead of "is this measurement of this commit", and the hand route is
    exactly where it matters: bless passes no `eval_baseline`, so it carries the
    previous slot over, and an unattributed bless would sit a stale measurement
    beside a fresh commit with nothing marking it — the #829 defect reintroduced by
    the escape hatch. The bless CLI tests stub `write_lkg` out entirely, so the
    mirror was asserted by nothing.

    Driven through the real `bless` on isolated state; the only stub is the
    backend's reported commit, an outside-world read that is not the seam.
    """
    import scripts.automod.round as R

    a = "a" * 40
    measurement = {"commit": a, "measured_at": "t1", "overall": {"ndcg10": 0.4}}
    S.write_lkg(a, eval_baseline=measurement)
    assert S.read_lkg()["eval_for_recorded_commit"] is True, (
        "a promotion that settled with its own number must say so first")

    # bless verifies against the RUNNING process before it moves the pointer, so
    # the one outside-world read is `/health`'s `commit`. Answering "the service
    # reports nothing" is the honest stub: bless then blesses `git rev-parse HEAD`
    # of the live root, and the branch under test — what `write_lkg` stamps — is
    # reached either way. Left unstubbed the test would ask the running production
    # backend which commit it serves and then bless whatever fell out.
    import scripts.automod.promote as P

    monkeypatch.setattr(P, "_get", lambda url, *a, **k: (200, {"commit": None}))
    R.bless("by hand for the test")

    rec = S.read_lkg()
    assert rec["commit"] not in (a, None), "bless did not advance the pointer"
    assert rec["eval"]["overall"]["ndcg10"] == 0.4, "bless dropped the measurement"
    assert rec["eval_for_recorded_commit"] is False, (
        "a bless carrying somebody else's measurement reads as this commit's baseline")
    assert rec["eval_commit"] == a, "the carried number no longer names its own commit"
    assert rec["eval_measured_at"] == "t1"

    # Parity with the guardian's route on the same facts: two writers, one record,
    # so the stamp must not depend on which one held the pen.
    guardian_route = gstate.AutomodState(isolated_state / "guardian-parity")
    guardian_route.set_lkg(a, eval_baseline=measurement)
    guardian_route.set_lkg(rec["commit"])
    theirs = gstate.read_json(guardian_route.lkg_path)
    for field in ("eval_for_recorded_commit", "eval_commit", "eval_measured_at"):
        assert rec[field] == theirs[field], (
            f"{field}: bless says {rec[field]!r}, the guardian says {theirs[field]!r}")


@pytest.mark.parametrize("settling", ["a" * 40, "b" * 40])
def test_settling_attributes_the_eval_slot_it_hands_out(tmp_path, monkeypatch, settling):
    """The same seam end to end, through the guardian rather than around it.

    The worker leaves a measurement for commit `a…` in `eval_last.json`; the
    guardian settles a promotion and is the only writer of the record. When it
    settles `a…` the slot is its own; when it settles `b…` the compare at
    `guardian.maybe_settle` refuses the fold, the previous slot is carried, and the
    record now says which of those happened. Asserted through `maybe_settle` on
    purpose: the comparison and the stamp live in different files, and pinning the
    stamp alone would let the two disagree without a test noticing.
    """
    import guardian as G
    import probes as P

    # Arguments from the real parser, not a hand-written namespace: this is the
    # settle path that hands out the eval slot, and a constructor that starts
    # reading a new option should make this test fail loudly rather than run
    # against a fixture that quietly stops describing a real daemon. Same route
    # the tick test uses for the same reason.
    g = G.Guardian(G.build_parser().parse_args([
        "--repo", str(tmp_path),
        "--state", str(tmp_path / "state"),
        "--guardian-state", str(tmp_path / "gstate"),
        "--supervisor-sock", "/nonexistent",
        "--programs", "lloyd-mc:lloyd-backend",
        "--no-external-alerts",
    ]))
    monkeypatch.setattr(P, "probe", lambda url, t: {
        "ok": True, "status": 200,
        "body": {"status": "ok", "degraded_modules": []}, "error": None,
        "latency_ms": 1.0})
    monkeypatch.setattr(G.gstate, "append_event", lambda *a, **k: None)

    g.state.set_lkg("a" * 40, eval_baseline={"commit": "a" * 40, "measured_at": "t1",
                                             "overall": {"ndcg10": 0.4}})
    gstate.write_json_atomic(g.state.eval_last, {"commit": "a" * 40, "measured_at": "t1",
                                                 "overall": {"ndcg10": 0.4}})
    g.maybe_settle({"commit": settling, "errors_until_ts": time.time() - 1_000_000})

    rec = gstate.read_json(g.state.lkg_path)
    assert rec["commit"] == settling
    assert rec["eval"]["overall"]["ndcg10"] == 0.4, "the measurement must survive either way"
    assert rec["eval_for_recorded_commit"] is (settling == "a" * 40)
    assert rec["eval_commit"] == "a" * 40


def test_installed_units_are_compared_against_the_repo():
    """systemd reads ~/.config/systemd/user, not the repo.

    A unit edit that was never installed is a change that looks landed and
    does nothing — for a watchdog unit, the worst kind of silent no-op.
    """
    from scripts.automod.round import _unit_drift
    assert isinstance(_unit_drift(), list)


def test_protected_service_definitions_are_actually_applied():
    """spec.py calls them protected — allowed with a drill — but nothing ever
    applied one: supervisord needs reread/update, units need installing, and
    the guardian runs a pinned snapshot re-staged only on a unit restart."""
    src = (ROOT / "scripts" / "automod" / "promote.py").read_text()
    assert "_apply_service_changes" in src
    for needed in ("reread", "update", "daemon-reload", "lloyd-guardian"):
        assert needed in src, f"{needed} is never applied after landing"


# ===========================================================================
# 7. Dead knobs
# ===========================================================================

@pytest.mark.parametrize("name", [
    "ERROR_RATE_FLOOR_PER_MIN", "ERROR_RATE_MULTIPLIER", "CUSUM_P1",
    "CUSUM_THRESHOLD", "CUSUM_P0_FLOOR", "WORKERS_DB", "CHRONIC_LOOKBACK_DAYS",
    "ADVISORY",
])
def test_policy_carries_no_knob_that_nothing_reads(name):
    """An operator tunes these expecting an effect. Each of these had none."""
    assert not hasattr(policy, name), f"policy.{name} is read by nothing"


def test_the_promotion_record_carries_no_field_nothing_reads():
    src = (ROOT / "scripts" / "automod" / "promote.py").read_text()
    guardian = (GUARDIAN_DIR / "guardian.py").read_text()
    for field in ("err_offset", "liveness_until_ts"):
        assert f'"{field}"' not in src, f"{field} is written and never read"
        assert field not in guardian


# ===========================================================================
# 8. The guardian must actually complete a tick
# ===========================================================================

def test_a_real_tick_completes_against_the_live_state(tmp_path):
    """The regression test for shipping an AttributeError into the loop.

    `tick()` catches every exception and logs "tick error (continuing)", which
    is the right shape for a watchdog — it must not die on a transient — but
    it means a plain typo degrades it to a process that runs forever and
    watches nothing. Exactly that happened when `intentional_stop` started
    calling `state.is_halted()`, which gstate did not have: the unit stayed
    `active`, the heartbeat kept updating, and no unit test noticed because
    they all exercise the predicates rather than the loop.

    So: build a real Guardian against scratch state and a supervisord that
    cannot be reached, and require a clean tick with nothing swallowed.
    """
    import guardian as G

    args = G.build_parser().parse_args([
        "--repo", str(ROOT),
        "--state", str(tmp_path / "automod"),
        "--guardian-state", str(tmp_path / "guardian"),
        "--supervisor-sock", str(tmp_path / "no-such.sock"),
        "--no-external-alerts",
    ])
    g = G.Guardian(args)

    raised = []
    real_tick = g.tick
    try:
        state = real_tick()
    except Exception as exc:            # noqa: BLE001 - that is the point
        raised.append(exc)
        state = None
    assert not raised, f"tick raised instead of completing: {raised}"
    # Unreachable supervisord is invariant 2: infrastructure, never a code trigger.
    assert state == "infra_down", state


def test_every_state_object_attribute_the_guardian_calls_exists(tmp_path):
    """A cheap structural guard against the same class of typo."""
    st = gstate.AutomodState(tmp_path)
    for name in ("lkg", "current", "floor", "rollback_target", "set_lkg",
                 "clear_current", "write_last_settled", "read_eval_last",
                 "read_rollback_request", "clear_rollback_request",
                 "is_broken", "is_halted", "set_broken", "set_halted",
                 "deny", "recent_rollbacks", "unfinished_rollback",
                 "pause_remaining"):
        assert hasattr(st, name), f"AutomodState.{name} is called but does not exist"


# ===========================================================================
# 9. The caller must stop talking after landing
# ===========================================================================

def test_landing_tells_the_caller_to_end_its_turn():
    """Found by driving the loop end to end as the agent, not from a terminal.

    The idle gate counts the CALLING turn too. An agent that lands and then
    polls `automod_status` in a loop is itself the reason the backend never
    goes idle, so the landing waits its full 15 minutes and gives up. The
    landing restarts the backend and ends that turn regardless, so the only
    correct move after `automod_land` returns is to stop.

    This is only reachable on the path that had never been exercised, which is
    the whole reason the loop was driven by the agent before being trusted.
    """
    src = (ROOT / "agent_mcp" / "automod.py").read_text()
    assert "END YOUR TURN" in src
    # ...and the tool description must not still say to poll, or the two
    # halves of the contract contradict each other.
    assert "Poll automod_status to follow it" not in src


def test_the_skill_says_to_stop_after_landing():
    skill = Path.home() / "obsidian" / "skills" / "automod-change-own-code" / "SKILL.md"
    if not skill.exists():
        pytest.skip("vault skill not present")
    text = skill.read_text()
    assert "End your turn" in text or "end your turn" in text


# ===========================================================================
# 10. The promoter's refusals, exercised rather than grepped
# ===========================================================================

@pytest.fixture()
def candidate(tmp_path):
    """A throwaway repo standing in for a round worktree."""
    r = tmp_path / "wt"
    r.mkdir()
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "f.py").write_text("A = 1\n", encoding="utf-8")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    (r / "f.py").write_text("A = 2\n", encoding="utf-8")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "gated")
    gated = git(r, "rev-parse", "HEAD").stdout.strip()
    return {"path": r, "base": base, "gated": gated}


def test_promote_refuses_a_commit_made_after_the_gate_passed(isolated_state, candidate,
                                                             monkeypatch):
    """`land` passed the base along, but the candidate HEAD was re-read from
    the worktree — so a commit made after a passing gate landed ungated."""
    from scripts.automod import promote as P
    r = candidate["path"]
    (r / "f.py").write_text("A = 3  # snuck in after the gate\n", encoding="utf-8")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "ungated")

    monkeypatch.setattr(P.S, "read_current", lambda: None)
    with pytest.raises(P.PromoteError) as exc:
        P.promote("SM_TEST", r, candidate["base"],
                  gate_report={"head": candidate["gated"]}, dry_run=True)
    assert "re-gate before landing" in str(exc.value)


def test_promote_allows_exactly_the_gated_commit(isolated_state, candidate, monkeypatch):
    """The same guard must not block the normal case.

    Asserted narrowly: the promotion still fails here, because the live tree in
    a unit test is not this scratch repo. What matters is that it does not fail
    on the gate-head check — over-mocking `subprocess` to get further patches
    the module for `worktree.git` too, which makes `W.head` return the wrong
    commit and the test pass for the wrong reason.
    """
    from scripts.automod import promote as P
    monkeypatch.setattr(P.S, "read_current", lambda: None)
    try:
        P.promote("SM_TEST", candidate["path"], candidate["base"],
                  gate_report={"head": candidate["gated"]}, dry_run=True)
    except P.PromoteError as exc:
        assert "re-gate before landing" not in str(exc), (
            "the gated commit itself was rejected as ungated")


def test_promote_refuses_while_another_promotion_is_observed(isolated_state, candidate,
                                                             monkeypatch):
    """A second landing overwrote current.json: the first never settled, and
    the new rollback target had never survived a window."""
    from scripts.automod import promote as P
    monkeypatch.setattr(P.S, "read_current", lambda: {
        "commit": "c" * 40, "state": "observing",
        "errors_until_ts": time.time() + 600})
    with pytest.raises(P.PromoteError) as exc:
        P.promote("SM_TEST", candidate["path"], candidate["base"],
                  gate_report={"head": candidate["gated"]}, dry_run=True)
    msg = str(exc.value)
    assert "under observation" in msg and "min left" in msg


def test_promote_refuses_a_change_denied_by_content(isolated_state, candidate, monkeypatch):
    """Re-deriving a reverted change under a new SHA must not walk past."""
    from scripts.automod import promote as P
    monkeypatch.setattr(P.S, "read_current", lambda: None)
    tree_hash = S.changed_tree_hash(candidate["path"], candidate["gated"], ["f.py"])
    S.deny("does-not-matter", tree_hash=tree_hash)
    with pytest.raises(P.PromoteError) as exc:
        P.promote("SM_TEST", candidate["path"], candidate["base"],
                  gate_report={"head": candidate["gated"]}, dry_run=True)
    assert "denylist" in str(exc.value)


# ===========================================================================
# 11. Applying changed service definitions
# ===========================================================================

def test_service_definition_changes_are_applied_by_kind(monkeypatch, tmp_path):
    """Each kind needs a different action, and none of them happened before.

    supervisord includes conf.d straight out of the repo and needs
    reread/update; systemd reads ~/.config/systemd/user, so a unit edited in
    the repo reached nothing at all; and the guardian runs a pinned snapshot
    re-staged only when its unit restarts. A round could pass the drill, land,
    look healthy, and leave the running system on the old definition.
    """
    from scripts.automod import promote as P
    calls = []
    monkeypatch.setattr(P, "_run", lambda argv, timeout=60.0: calls.append(
        [str(a) for a in argv]) or type("R", (), {"returncode": 0, "stdout": "",
                                                  "stderr": ""})())
    monkeypatch.setattr(P, "SYSTEMD_USER_DIR", tmp_path / "systemd")

    notes = P._apply_service_changes([
        "agent-services/supervisor/conf.d/lloyd-backend.conf",
        "agent-services/systemd/lloyd-guardian.service",
        "agent-services/guardian/policy.py",
    ])
    flat = [" ".join(c) for c in calls]
    assert any("reread" in c for c in flat), "supervisord never re-read conf.d"
    assert any("update" in c for c in flat), "supervisord never applied the change"
    assert any("daemon-reload" in c for c in flat), "systemd never reloaded"
    assert any("restart lloyd-guardian" in c for c in flat), \
        "a landed guardian change stays inert until its unit restarts"
    # The unit must actually be copied to where systemd reads it.
    assert (tmp_path / "systemd" / "lloyd-guardian.service").exists()
    assert any("installed lloyd-guardian.service" in n for n in notes)


def test_a_code_only_change_touches_no_service_machinery(monkeypatch, tmp_path):
    """The common case must not restart the watchdog for nothing."""
    from scripts.automod import promote as P
    calls = []
    monkeypatch.setattr(P, "_run", lambda argv, timeout=60.0: calls.append(argv)
                        or type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(P, "SYSTEMD_USER_DIR", tmp_path / "systemd")
    notes = P._apply_service_changes(["app/inner_voice/guards.py", "tests/test_x.py"])
    assert calls == [] and notes == []


def test_an_aggregator_verdict_is_confirmed_across_ticks(tmp_path):
    """`mcp_degraded_is_fatal` fires on a body reporting zero tools.

    An aggregator answering 500 mid-restart parses to exactly that, so without
    a streak one bad response reverts a promotion on its own. Every other
    detector here confirms across ticks; this one was reached only via a 503,
    which hid how sharp it was.
    """
    import guardian as G

    args = G.build_parser().parse_args([
        "--repo", str(ROOT),
        "--state", str(tmp_path / "automod"),
        "--guardian-state", str(tmp_path / "guardian"),
        "--supervisor-sock", str(tmp_path / "no-such.sock"),
        "--no-external-alerts",
    ])
    g = G.Guardian(args)
    mcp = "lloyd-mc:lloyd-mcp"
    g.programs = (mcp,)
    g.probe_fail = {mcp: 0}
    g.probe_timeout = {mcp: 0}
    g.probe_http = {mcp: 0}

    snap = {
        "now": time.time(),
        "procs": {mcp: {"statename": "RUNNING", "start": 1}},
        "probes": {mcp: {"ok": False, "status": 500, "kind": "http_error",
                         "body": None, "error": None}},
    }
    seen = [g.evaluate_liveness(snap)[0] for _ in range(policy.MCP_FATAL_STREAK)]
    assert seen[:-1] == [False] * (policy.MCP_FATAL_STREAK - 1), \
        "a single bad response must not revert code"
    assert seen[-1] is True, "a persistent zero-tool aggregator must still fire"

    # ...and one good answer clears it.
    snap["probes"][mcp] = {"ok": True, "status": 200, "kind": "ok",
                           "body": {"tools": 130, "degraded_modules": []}}
    assert g.evaluate_liveness(snap)[0] is False
    assert g.mcp_fatal_streak == 0


# ===========================================================================
# 12. The pinned document corpus
# ===========================================================================

def test_the_grep_corpus_can_be_repointed(monkeypatch, tmp_path):
    """The retriever greps the repository it ships in.

    So in a paired comparison the code under test is also part of the corpus,
    and each arm searched its own source. Measured 2026-09-07: the query
    `lloyd-vllm-rel` returned six different files per arm, and ndcg10 and
    mrr_doc each moved 0.0060 with the qmd corpus already pinned and no
    retrieval code changed at all.
    """
    import importlib
    (tmp_path / "agent_mcp").mkdir()
    (tmp_path / "app").mkdir()
    before = os.environ.get("LLOYD_CODE_ROOT")
    monkeypatch.setenv("LLOYD_CODE_ROOT", str(tmp_path))
    import agent_mcp.vault as V
    importlib.reload(V)
    try:
        assert V.LLOYD_CODE_ROOTS[0] == tmp_path / "agent_mcp"
        assert V.LLOYD_CODE_PREFIX == str(tmp_path) + "/"
    finally:
        # Restore, never delete: a reload re-reads the environment, and a
        # caller that set this for the whole run (an eval arm does) would
        # otherwise lose it for every test after this one. Same shape as
        # `isolated_state` above.
        if before is None:
            monkeypatch.delenv("LLOYD_CODE_ROOT", raising=False)
        else:
            monkeypatch.setenv("LLOYD_CODE_ROOT", before)
        importlib.reload(V)


def test_the_grep_corpus_defaults_to_this_checkout():
    """Unset, behaviour must be exactly as before."""
    import agent_mcp.vault as V
    assert V.LLOYD_CODE_ROOTS[0] == V.LLOYD_HOME / "agent_mcp"


def test_the_pin_refuses_a_port_it_did_not_start(tmp_path):
    """Comparing against somebody else's daemon is comparing against unknown
    data, which is the failure the pin exists to remove."""
    import socket
    from scripts.automod import evalpin

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        assert evalpin.port_free(busy) is False
        with pytest.raises(evalpin.PinError) as exc:
            with evalpin.PinnedCorpus(tmp_path, port=busy):
                pass
    assert "already in use" in str(exc.value)


def test_the_overlay_repoints_only_the_qmd_service(tmp_path):
    """It rides the same mechanism as the gate's canary, so it must not carry
    anything else that could change how an arm behaves."""
    import yaml
    from scripts.automod import evalpin

    path = evalpin.write_overlay(tmp_path / "o.yaml", port=18999)
    doc = yaml.safe_load(path.read_text())
    assert doc == {"services": {"qmd": "http://localhost:18999/query"}}


def test_env_for_pins_both_halves_of_the_corpus(tmp_path):
    from scripts.automod import evalpin
    pin = evalpin.PinnedCorpus(tmp_path)
    pin.overlay = tmp_path / "o.yaml"
    env = pin.env_for({}, code_root="/some/tree")
    assert env["LLOYD_CONFIG_OVERLAY"] == str(pin.overlay)
    assert env["LLOYD_CODE_ROOT"] == "/some/tree"


def test_env_for_names_the_pinned_index_file(tmp_path, monkeypatch):
    """The THIRD half of a pinned arm's corpus, and the one #1374 found unclaimed.

    A pinned arm reaches its daemon over HTTP and cannot tell which file that daemon
    was served (`evalpin.py:183-196` passes `--index <name>`; `pin_index_path` at
    `:133` puts the pin at `~/.cache/qmd/<name>.sqlite`). Without this key the arm's
    artifact records the LIVE index's path, mtime and row count as the identity of a
    frozen snapshot — a provenance field that reads as precise and is wrong, which is
    the defect #1374 was filed under rather than one it repairs. `app.doc_corpus`
    refuses the live default under an overlay and writes null plus a reason instead;
    this is what makes the null not happen.

    The key is asserted against `doc_corpus.INDEX_PATH_ENV` rather than as a literal,
    because the two sides sit across a process boundary and a renamed constant on one
    side would otherwise silently disable the whole mechanism.
    """
    from app import doc_corpus
    from scripts.automod import evalpin

    # Redirect the pin's home into the fixture dir: `pin_index_path` resolves through
    # the module's QMD_CACHE, and this test must not create a file beside the real
    # ~/.cache/qmd/index.sqlite that a concurrent evalpin could then serve.
    monkeypatch.setattr(evalpin, "QMD_CACHE", tmp_path)
    pin = evalpin.PinnedCorpus(tmp_path, name="evalpin-smoke")
    pin.overlay = tmp_path / "o.yaml"
    env = pin.env_for({}, code_root="/some/tree")
    assert env[doc_corpus.INDEX_PATH_ENV] == str(evalpin.pin_index_path("evalpin-smoke"))
    assert env[doc_corpus.INDEX_PATH_ENV].endswith("evalpin-smoke.sqlite")

    # And the consumer agrees: the env this produces is exactly the one under which
    # `index_identity` stops refusing the path, so the arm's artifact carries the
    # pin's identity instead of `PIN_UNNAMED_REASON`.
    monkeypatch.setenv(doc_corpus.CONFIG_OVERLAY_ENV, str(pin.overlay))
    monkeypatch.delenv(doc_corpus.INDEX_PATH_ENV, raising=False)
    refused = doc_corpus.index_identity()
    assert refused["index_path"] is None and refused["content_vectors"] is None
    monkeypatch.setenv(doc_corpus.INDEX_PATH_ENV, env[doc_corpus.INDEX_PATH_ENV])
    # The named snapshot has to exist for its identity to be readable — a missing
    # file is the OTHER refusal reason, asserted separately above.
    Path(env[doc_corpus.INDEX_PATH_ENV]).write_bytes(b"VACUUM INTO copy")
    named = doc_corpus.index_identity()
    assert named["index_path"] == env[doc_corpus.INDEX_PATH_ENV]
    assert named["index_reason"] is None, named


def test_the_retargeted_eval_query_is_satisfiable():
    """The old `qwen35-users` expected files from ~/lloyd while qmd indexes
    ~/obsidian, so no retrieval quality could satisfy it."""
    import yaml
    spec = yaml.safe_load((ROOT / "eval" / "vault_recall_queries.yaml").read_text())
    ids = {q["id"] for q in spec["queries"]}
    assert "qwen35-users" not in ids, "the unsatisfiable query is still armed"

    q = next(q for q in spec["queries"] if q["id"] == "qwen38-local-serving")
    vault = Path.home() / "obsidian"
    for fragment in q["expect_docs"]:
        hits = list(vault.rglob(f"*{fragment}*"))
        assert hits, f"{fragment} matches nothing in the indexed vault"


def test_the_pin_does_not_outlive_its_owner():
    """`start_new_session=True` is the PROMOTER's requirement, and the exact
    opposite of this one's.

    A pinned corpus exists only for one comparison and holds an embedding
    model. Orphaned, it keeps answering on :8182 and the next run refuses the
    port it cannot verify.
    """
    import ast

    src = (ROOT / "scripts" / "automod" / "evalpin.py").read_text()
    popens = [n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"]
    assert popens, "no subprocess.Popen found in evalpin"
    kwargs = {k.arg for call in popens for k in call.keywords}
    assert "start_new_session" not in kwargs, "the pin would survive its owner"
    assert "process_group" in kwargs, "stop() could not reach the whole tree"
    # Neither of those makes the title true, and on 2026-09-18 both held while
    # an orphaned pin kept :8182 for five and a half hours: its own process
    # group is exactly what a group signal to its owner cannot reach, and a
    # child whose parent dies is re-parented, not killed. What makes it true is
    # asking the kernel; `test_regression_runner` kills a real owner to see.
    assert "preexec_fn" in kwargs, "nothing ends the pin when its owner dies"


def test_the_pin_frees_its_snapshot_even_when_interrupted():
    """The snapshot is 1 GB. `discard` on the happy path only meant an
    interrupted run leaked it, which is what happened on 2026-09-07."""
    import inspect
    from scripts.automod import evalpin
    body = inspect.getsource(evalpin.PinnedCorpus.__exit__)
    assert "self.stop()" in body and "self.discard()" in body


def test_a_noise_floor_records_the_questions_it_was_measured_against():
    """A floor describes one experiment; retargeting a query changes it.

    And since #1352 the mismatch IS a gate: `check_promotion` compares the
    artifact's fingerprint with the live one before spending an arm on a second
    look, and a mismatched artifact reports its deltas without asking for a
    rollback. The old ruling — "not a gate, because a pinned corpus is
    deterministic so the floor falls back to MIN_SIGMA either way" — held only
    while the doc leg was a cross-encoder. djev took over the ranking on
    2026-09-21, one settled commit measured itself at Δ-0.0090 and Δ+0.0010 on
    `ndcg10` inside a single check, and two promotions were reverted on a floor
    that every check that day had already flagged stale.
    """
    from workers.sources import automod_regression as R
    fp = R.queries_fingerprint()
    assert fp and len(fp) == 12
    src = (ROOT / "workers" / "sources" / "automod_regression.py").read_text()
    assert '"queries_fingerprint": queries_fingerprint()' in src
    assert "noise_floor_stale" in src


# ===========================================================================
# 13. A round must be observed
# ===========================================================================

def _iv_gate(monkeypatch, tmp_path, session_id, session_data=None, require=True,
             turn_id=None):
    """Drive `_inner_voice_gate` with a scratch sessions dir.

    `turn_id` is the turn this call belongs to. Left as None the context var is
    not touched at all, which is what a CLI, a detached promoter or a direct
    `run_query` caller looks like to the gate; `""` and a real id are the two
    harness shapes. It is a separate control from `session_id` on purpose: the
    refused-turn marker is keyed on the turn, so a test has to be able to move
    the turn under one session file. `_restore_turn_id` puts the var back.
    """
    import json

    import agent_mcp.automod as M
    if session_data is not None:
        (tmp_path / f"{session_id}.json").write_text(json.dumps(session_data),
                                                     encoding="utf-8")
    monkeypatch.setattr(M, "get_bound_session", lambda: session_id)
    monkeypatch.setattr(M, "_require_inner_voice", lambda: require)
    if turn_id is not None:
        import agent_mcp._task_registry as TR
        TR.current_turn_id.set(turn_id)
    import app.paths
    monkeypatch.setattr(app.paths, "SESSIONS_DIR", tmp_path)
    return M._inner_voice_gate("open a round")


@pytest.fixture(autouse=True)
def _restore_turn_id():
    """`current_turn_id` is a context var and tests here bind it, so hand the
    next test the value it started with — a leaked id would otherwise make a
    later test's second call look sticky."""
    import agent_mcp._task_registry as TR
    tok = TR.current_turn_id.set(TR.current_turn_id.get())
    try:
        yield
    finally:
        TR.current_turn_id.reset(tok)


def test_a_worker_session_opens_a_round_without_the_observer(monkeypatch, tmp_path):
    """Cut 1 of senses-not-supervision: `inner_voice: false` on the autocode
    source. The gate's two reasons were written for a chat turn and neither
    holds unattended — the observer's measured effect on rounds was negative,
    and every background run is recorded and listed in the Background tab.
    Without this exemption the config switch would have stopped every round
    from opening, which is the opposite of a switch.
    """
    for platform in ("worker", "autonomy"):
        gate = _iv_gate(monkeypatch, tmp_path, f"s-{platform}",
                        {"id": f"s-{platform}", "inner_voice": False,
                         "platform": platform})
        assert gate is None, platform


def test_a_chat_session_is_still_refused_without_the_observer(monkeypatch, tmp_path):
    """The exemption is by platform, not a loosening of the chat rule."""
    gate = _iv_gate(monkeypatch, tmp_path, "s-chat",
                    {"id": "s-chat", "inner_voice": False, "platform": "mission-control"})
    assert gate is not None


def test_the_unattended_sources_ship_with_the_observer_off():
    """The switch itself, pinned: cut 1 is a config change and a config
    change can be reverted by a UI toggle without anyone noticing."""
    from app.config import CONFIG
    src = CONFIG["workers"]["sources"]
    for name in ("autocode", "autotriage", "arch-review", "youtube-digest"):
        assert src[name].get("inner_voice") is False, name


def test_a_round_is_refused_from_an_unobserved_turn(monkeypatch, tmp_path):
    """The observer attaches at turn START, so enabling the flag mid-call
    cannot cover the turn that enabled it.

    A round driven in a single turn would otherwise report itself observed
    while running blind, which is worse than being plainly unobserved.
    """
    gate = _iv_gate(monkeypatch, tmp_path, "s1", {"id": "s1", "inner_voice": False})
    assert gate is not None
    assert "refusing to open a round" in gate["error"]
    assert gate["inner_voice_enabled_for_next_turn"] is True

    # ...and it enabled it, so the retry is observed rather than refused again.
    import json as _json
    written = _json.loads((tmp_path / "s1.json").read_text())
    assert written["inner_voice"] is True
    assert written["inner_voice_evaluate_user_turns"] is True


def test_an_observed_turn_proceeds(monkeypatch, tmp_path):
    assert _iv_gate(monkeypatch, tmp_path, "s2", {"id": "s2", "inner_voice": True}) is None


def test_the_cli_is_exempt(monkeypatch, tmp_path):
    """No bound session means the CLI or the detached promoter. A human at a
    terminal is their own observer, and blocking that path would make the
    documented recovery commands unusable."""
    assert _iv_gate(monkeypatch, tmp_path, "") is None


def test_an_unreadable_session_does_not_block_self_modification(monkeypatch, tmp_path):
    """Fail closed on the observer, not on the round. A missing session file
    should not be able to disable self-modification entirely."""
    assert _iv_gate(monkeypatch, tmp_path, "gone") is None


def test_the_requirement_is_switchable(monkeypatch, tmp_path):
    assert _iv_gate(monkeypatch, tmp_path, "s3", {"id": "s3", "inner_voice": False},
                    require=False) is None


def _iv_session(tmp_path, session_id):
    import json
    return json.loads((tmp_path / f"{session_id}.json").read_text(encoding="utf-8"))


CHAT = {"id": "s", "inner_voice": False, "platform": "mission-control"}


def test_a_refusal_is_sticky_for_the_turn_that_issued_it(monkeypatch, tmp_path):
    """#1226: a second call inside the refused turn must not get a pass.

    The refusing call writes `inner_voice: true` into the session file, and the
    flag was the first thing every later call read — so one retry, with no turn
    boundary and therefore no observer, walked straight through. This happened
    on 2026-09-17: refused at msg39, `round_id SM_20260917_174742` returned 71
    seconds later at msg76 of the same turn
    (`sessions/20260917_104306_autocode_f9fe.json`, 366 messages, exactly one
    `role:"user"`), and the round it opened ran four gates and two land attempts
    with nothing attached to any of them.

    The first call in this test is the one `test_a_round_is_refused_from_an_
    unobserved_turn` already pins; the second and third are the bug. Asserting
    them against the file that call 1 rewrote is the point — a marker that only
    lived in the refusing call's own memory would pass a test like this one and
    still leak in production, where the retry is a fresh call and can even be a
    restarted process.
    """
    for call in (1, 2, 3):
        gate = _iv_gate(monkeypatch, tmp_path, "s-sticky",
                        dict(CHAT, id="s-sticky") if call == 1 else None,
                        turn_id="turn-a")
        assert gate is not None, f"call {call} opened a round after a refusal"
        assert "refusing to open a round" in gate["error"]
        assert gate["next"].startswith("End your turn"), (
            "the retry copy must still send the caller to the next turn, not "
            "invite another call in this one")

    written = _iv_session(tmp_path, "s-sticky")
    assert "turn-a" in written["inner_voice_refused_turns"]
    # Refusing the rest of the turn does not undo the enabling: the two flags the
    # first call wrote are what make the NEXT turn observed — `inner_voice`
    # attaches the observer and `inner_voice_evaluate_user_turns` is what lets it
    # judge a turn that arrives without the human typing. Staying silent for the
    # rest of the turn must not cost the session the enabling it was promised.
    assert written["inner_voice"] is True
    assert written["inner_voice_evaluate_user_turns"] is True


def test_a_later_turn_under_the_same_session_file_is_not_locked_out(monkeypatch,
                                                                    tmp_path):
    """The marker is turn-scoped. Refusing every later call too would be the
    safer-looking patch and would stop the session ever opening a round: the
    flag the refusal writes is exactly what makes the next turn observed, so
    the next turn has to pass. That is the whole design — refuse once, and the
    retry that crosses a turn boundary runs with an observer."""
    first = _iv_gate(monkeypatch, tmp_path, "s-next",
                     dict(CHAT, id="s-next"), turn_id="turn-a")
    assert first is not None
    assert _iv_gate(monkeypatch, tmp_path, "s-next", turn_id="turn-b") is None


def test_a_turn_id_that_was_never_refused_proceeds_once_the_flag_is_on(monkeypatch,
                                                                       tmp_path):
    """A marker left by an earlier turn must not act as a session-wide ban.

    Same file, marker present, different turn id, flag true: the observer is
    attached by definition (the flag was set for that turn), so proceeding is
    correct — and the marker list having a stale entry is normal, not a state
    to honour.
    """
    assert _iv_gate(monkeypatch, tmp_path, "s-stale",
                    dict(CHAT, id="s-stale", inner_voice=True,
                         inner_voice_refused_turns=["turn-a"]),
                    turn_id="turn-z") is None


def test_a_call_with_no_turn_id_keeps_todays_behaviour(monkeypatch, tmp_path):
    """Nothing keys the marker, so nothing is sticky — deliberately.

    The turn id defaults to `""` for the CLI, the detached promoter and a
    direct `run_query` caller (`app/harness/options.py:110`), and the marker
    lives in the *session* file: a refusal recorded against `""` would refuse
    every later turn of that session and lock a real operator out of the loop.
    So the no-turn-id path is exactly what it was before #1226 — refused once,
    then proceeds because the flag is on.
    """
    assert _iv_gate(monkeypatch, tmp_path, "s-empty",
                    dict(CHAT, id="s-empty"), turn_id="") is not None
    assert _iv_gate(monkeypatch, tmp_path, "s-empty", turn_id="") is None
    # The other no-turn-id shape: the context var never set (its default).
    assert _iv_gate(monkeypatch, tmp_path, "s-plain",
                    dict(CHAT, id="s-plain")) is not None
    assert _iv_gate(monkeypatch, tmp_path, "s-plain") is None
    assert "inner_voice_refused_turns" not in _iv_session(tmp_path, "s-empty"), (
        "a marker keyed on nothing is a ban, not a turn marker")


def test_an_exempt_session_with_a_stale_marker_is_still_exempt(monkeypatch, tmp_path):
    """The platform exemption is read before the marker, so a session that
    changed platform (or an autonomy session minted a file by #1064's
    hard-coded default and later rewritten) cannot be refused by a marker the
    exempt platform never needed. The exemption is a statement that no observer
    is required, which a refusal cannot answer."""
    for platform in ("worker", "autonomy"):
        gate = _iv_gate(monkeypatch, tmp_path, f"s-{platform}",
                        {"id": f"s-{platform}", "inner_voice": False,
                         "platform": platform,
                         "inner_voice_refused_turns": ["turn-a"]},
                        turn_id="turn-a")
        assert gate is None, platform


def test_the_refused_turn_list_stays_bounded(monkeypatch, tmp_path):
    """One marker per refused turn is unbounded on its own: a session whose
    `inner_voice` is cleared and re-refused accumulates forever. The marker only
    has to outlive the turn that wrote it, so the kept list is capped and keeps
    the most recent ids — the only ones a live turn can carry."""
    old = [f"turn-{i}" for i in range(12)]
    gate = _iv_gate(monkeypatch, tmp_path, "s-bound",
                    dict(CHAT, id="s-bound", inner_voice_refused_turns=old),
                    turn_id="turn-new")
    assert gate is not None
    kept = _iv_session(tmp_path, "s-bound")["inner_voice_refused_turns"]
    assert len(kept) == 8
    assert kept[-1] == "turn-new"
    assert kept[0] == "turn-5", kept


def _dispatch_automod(monkeypatch, tmp_path, session_id):
    """Send automod calls the way the harness sends them, through the aggregator.

    `agent_mcp.main.call_tool` is the only place a turn id crosses from the
    client's `_meta` into `current_turn_id`, so the sticky marker has to be
    proven there: the gate reads a context var that only that dispatch sets, and
    `automod.call_tool` runs its body in `asyncio.to_thread`, so the seam is
    invisible from the gate side. The returned helper takes
    `(tool, arguments, turn_id)` and answers with the payload plus `isError`.

    The effect ledger is stubbed out. It is not the seam under test, and its
    duplicate-call guard answers a repeat with the same arguments before the
    module handler is ever reached — which would refuse the retry for the wrong
    reason and write a test row into the live ledger.
    """
    import json

    import agent_mcp.main as M
    import app.paths

    (tmp_path / f"{session_id}.json").write_text(
        json.dumps(dict(CHAT, id=session_id)), encoding="utf-8")
    monkeypatch.setattr(app.paths, "SESSIONS_DIR", tmp_path)

    async def _claim(*a, **k):
        # An unledgered claim: dispatch proceeds, and with no key the dispatcher
        # never writes a result row, so the suite stays out of the live ledger.
        return M._tool_effects.Claim()

    monkeypatch.setattr(M._tool_effects, "claim", _claim)

    async def _call(tool, args, turn_id):
        res = await M.call_tool(tool, args, {
            M.META_SESSION_ID: session_id, M.META_TURN_ID: turn_id})
        return json.loads(res.content[0].text), res.is_error

    return _call


async def test_a_same_turn_retry_is_refused_across_the_mcp_dispatch_seam(
        monkeypatch, tmp_path):
    """Clause 1 where it actually bites: two real dispatches, one turn id.

    Each dispatch is its own request whose body runs in its own thread, which is
    the shape the 2026-09-17 retry had 71 seconds after the refusal. A marker
    held in the refusing call's memory would pass the in-process tests above and
    still leak here; the payload has to come back from the *file*.

    `automod_start` is never allowed through the gate in this test — the round
    it would open is a real worktree off the live tree — so the next-turn half
    of the property is proven on the vault route, below.
    """
    call = _dispatch_automod(monkeypatch, tmp_path, "s-seam")

    first, is_error = await call("automod_start", {"goal": "x"}, "turn-1")
    assert is_error
    assert "inner voice is not attached" in first["error"]
    assert "already_refused_this_turn" not in first

    retry, is_error = await call("automod_start", {"goal": "x"}, "turn-1")
    assert is_error, "a same-turn retry opened a round"
    assert "inner voice is not attached" in retry["error"]
    assert retry["already_refused_this_turn"] is True
    assert "round_id" not in retry


async def test_a_same_turn_retry_of_the_vault_route_is_refused_and_writes_nothing(
        monkeypatch, tmp_path):
    """Clause 5. `automod_vault_land` commits straight to the live vault, so a
    bypass there is an unobserved commit rather than a worktree — and all ten
    refusals counted before the autonomy exemption happened on this route.

    The lander is replaced with a recorder, so the retry reaching it fails here
    on the spot and no real vault is in reach. The last leg is clause 2 across
    the same seam: a different turn id on that same session file, marker still
    present, proceeds — the refusal must not become a ban.
    """
    import agent_mcp.automod as AM
    from scripts.automod import vault_round as VR

    landed = []
    monkeypatch.setattr(VR, "land",
                        lambda *a, **k: landed.append(a) or {"sha": "not-landed"})
    args = {"paths": ["knowledge/x.md"], "message": "m"}
    call = _dispatch_automod(monkeypatch, tmp_path, "s-vault")

    first, is_error = await call("automod_vault_land", args, "turn-1")
    assert is_error and "land a vault change" in first["error"]

    retry, is_error = await call("automod_vault_land", args, "turn-1")
    assert is_error, "a same-turn retry landed a vault commit unobserved"
    assert "land a vault change" in retry["error"]
    assert retry["already_refused_this_turn"] is True
    assert not landed, "the refusal still reached the vault lander"

    out, is_error = await call("automod_vault_land", args, "turn-2")
    assert not is_error, out
    assert landed, "the next turn never reached the vault route"
    assert AM.IV_REFUSAL_KEY in _iv_session(tmp_path, "s-vault")


def test_the_gate_is_actually_wired_into_automod_start():
    """A gate nothing calls is the failure this whole review keeps finding."""
    src = (ROOT / "agent_mcp" / "automod.py").read_text()
    start = src.index('if name == "automod_start"')
    body = src[start:start + 500]
    assert "_inner_voice_gate(" in body, "automod_start does not consult the gate"


def test_the_two_unobserved_paths_are_deliberate_and_say_so():
    """The canary smoke turn and worker turns both run without an observer.
    Neither is an oversight, and both must explain themselves where someone
    would otherwise 'fix' them."""
    smoke = (ROOT / "scripts" / "automod" / "canary_smoke.py").read_text()
    assert "not an automod job" in smoke.lower() or "gate rung, not an automod job" in smoke

    common = (ROOT / "workers" / "sources" / "_common.py").read_text()
    # The phrasing moved on 2026-09-10, when `run_prompt_on_primary` gained a
    # session and a transcript: it is now *recorded* and still not *observed*,
    # which is a sharper claim than "no session" and the one worth pinning.
    # Observation stays wired in `app/routers/messages.py` and nowhere else.
    assert "Recorded, not observed" in common
    assert "no Inner Voice" in common


def test_the_state_route_hands_out_the_axis_and_the_attribution(isolated_state):
    """The record's new fields have to survive the route a person actually reads.

    `GET /api/automod/status` is Mission Control's banner feed and it passes
    `S.read_lkg()` through inside a hand-built response dict
    (`app/routers/automod.py`, `get_status`). A passthrough that picked keys — or a
    later refactor that re-projected the record — would leave the axis claim and the
    attribution on disk and off the screen, which is the half of the gap a reader
    can see. So this drives the real route over an in-process ASGI transport and
    asserts the served JSON, not the file: `isolated_state` repoints `LKG_PATH`, and
    the suite-wide `LLOYD_AUTOMOD_STATE` isolation (`tests/conftest.py:63-81`) keeps
    the route off the real state directory regardless.

    The file-writing tests in `tests/test_landing_without_restart.py` drive a route
    by POSTing to the live backend, which is exactly what a gated change must not
    do — the restart it triggers is what the round is waiting for. In-process is the
    only honest way to reach a GET here.
    """
    import asyncio

    import httpx

    import server  # the root ASGI app; `tests/test_api_contracts.py:22` imports it the same way
    import scripts.automod.state as S

    measurement = {
        "commit": "b" * 40, "measured_at": "2026-09-22T00:00:00Z",
        "axis": {"measures": "retrieval quality, paired A/B",
                 "does_not_measure": ["agent-loop behaviour"]},
        "overall": {"ndcg10": 0.4},
    }
    S.write_lkg("b" * 40, eval_baseline=measurement)

    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 9999))

    async def call():
        async with httpx.AsyncClient(transport=transport, timeout=30) as client:
            return await client.get("http://127.0.0.1:8096/api/automod/status")

    resp = asyncio.run(call())
    assert resp.status_code == 200, resp.text[:200]
    served = resp.json()["last_known_good"]
    assert served["eval"]["axis"]["measures"] == "retrieval quality, paired A/B", (
        "the axis claim does not reach the surface a human reads")
    assert served["eval_for_recorded_commit"] is True, (
        "the attribution does not reach the surface a human reads")
    assert served["eval_commit"] == "b" * 40


def test_the_production_write_carries_the_axis_into_the_record(monkeypatch, tmp_path):
    """The axis has to be in the payload the REAL write path emits.

    `test_the_measurement_states_which_axis_it_measured` builds the payload through
    `eval_last_payload`, which would still pass if `check_promotion` had kept its own
    inline literal and never called the function — a field declared on one side of a
    seam and never handed over, the bug shape this file keeps writing tests for. So
    here the worker's own blocking run executes, with only the two arms, the noise
    floor, the pin and the worktree stubbed, and the payload captured from
    `write_eval_last` is what gets asserted.

    The fakes are imported from `tests/test_automod_regression.py` rather than
    re-implemented: they encode what a well-formed arm looks like to `check_promotion`
    (81 answered questions, a corpus that is fine, a fingerprint matching the live
    query file), and a second copy would drift from the harness that already pins
    those preconditions.
    """
    import json

    import scripts.automod.state as S
    import test_automod_regression as T
    import workers.sources.automod_regression as R

    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(T.ZERO_NOISE), encoding="utf-8")
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: T._arm(20, []))
    monkeypatch.setattr(R, "PinnedCorpus", lambda *a, **k: T._FakePin())
    monkeypatch.setattr(R, "_baseline_worktree", T._fake_worktree)
    monkeypatch.setattr(S, "read_current", lambda: T._observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(S, "request_rollback", lambda **k: k)
    captured: dict = {}
    monkeypatch.setattr(S, "write_eval_last", lambda p: captured.update(p))

    out = R._execute_blocking()
    assert out.get("status") == "success", out
    assert captured, "the run wrote no measurement at all"
    assert captured[R.AXIS_FIELD]["measures"], (
        "the production write emitted no axis, so the LKG eval slot still states "
        "numbers with no stated coverage")
    refused = " ".join(captured[R.AXIS_FIELD]["does_not_measure"]).lower()
    assert "tool call" in refused and "turn count" in refused and "model decision" in refused
    assert captured["commit"] == "b" * 40 and captured["baseline_commit"] == "a" * 40, (
        "the axis arrived but the measurement it describes lost its subject")
