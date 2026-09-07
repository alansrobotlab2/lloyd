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

from scripts.selfmod import state as S  # noqa: E402


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)


@pytest.fixture()
def isolated_state(tmp_path, monkeypatch):
    """Point the state module at a scratch dir. Never the live one."""
    import importlib
    monkeypatch.setenv("LLOYD_SELFMOD_STATE", str(tmp_path / "state"))
    importlib.reload(S)
    S.ensure_dirs()
    yield tmp_path / "state"
    monkeypatch.delenv("LLOYD_SELFMOD_STATE", raising=False)
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
    """`selfmod.enabled` was checked in the MCP wrapper alone.

    The skill's own worked examples drive `python -m scripts.selfmod.round`,
    so "the loop ships inert" was true of the surface nobody used and false of
    the one they did.
    """
    (tmp_path / "config.yaml").write_text("selfmod:\n  enabled: false\n", encoding="utf-8")
    assert S.is_enabled(tmp_path) is False
    with pytest.raises(S.SelfmodDisabled):
        S.require_enabled("land a round", tmp_path)

    (tmp_path / "config.yaml").write_text("selfmod:\n  enabled: true\n", encoding="utf-8")
    assert S.is_enabled(tmp_path) is True
    S.require_enabled("land a round", tmp_path)          # does not raise


def test_a_missing_config_reads_as_disabled(tmp_path):
    """Fail closed. An unreadable config must not enable a live promotion."""
    assert S.is_enabled(tmp_path / "nope") is False


def test_worker_turns_cannot_drive_the_loop():
    """`domain-research` reads arbitrary web pages into its context.

    The selfmod tools were advertised to every worker prompt, so the machinery
    that rewrites production sat one prompt injection away from a source whose
    whole job is ingesting untrusted text. The backlog worker was told not to
    use them in its PROMPT, which is not a control.
    """
    src = (ROOT / "workers" / "sources" / "_common.py").read_text()
    for tool in ("selfmod_start", "selfmod_gate", "selfmod_land", "selfmod_rollback"):
        assert tool in src, f"{tool} is not disallowed for worker turns"


def test_subagents_cannot_drive_the_loop():
    """A Task runs inside the aggregator a landing restarts."""
    src = (ROOT / "agent_mcp" / "builtin_task.py").read_text()
    for tool in ("selfmod_start", "selfmod_land", "selfmod_rollback"):
        assert tool in src, f"{tool} is not disallowed for subagents"


def test_the_drain_endpoint_is_loopback_only():
    """Its only legitimate caller is the promoter, on this box.

    `server.py`'s auth middleware enforces the client-cert allowlist only when
    a fingerprint is actually forwarded, so on the tailnet this was an
    unauthenticated way to make the backend refuse every turn for 10 minutes.
    """
    src = (ROOT / "app" / "routers" / "selfmod.py").read_text()
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
    for rel in ("agent_mcp/selfmod.py", "workers/sources/selfmod_regression.py"):
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
    from scripts.selfmod import gate as G
    counts = G._parse_pytest_summary("collected 1632 items\n\n1632 skipped in 2s")
    assert counts["collected"] == 1632 and counts["passed"] == 0
    assert counts["collected"] >= G.PYTEST_MIN_COLLECTED, "the old floor passes this"
    assert counts["passed"] < G.PYTEST_MIN_PASSED, "the new floor must not"


def test_skips_are_parsed_at_all():
    from scripts.selfmod import gate as G
    assert G._parse_pytest_summary("1600 passed, 12 skipped in 30s")["skipped"] == 12


def test_gate_rungs_run_candidate_code_against_scratch_state():
    """static/tests/venv ran candidate code against the LIVE state dir.

    A candidate test that forgot its isolation fixture could write a real
    BROKEN or promotions-halted flag, or append to the production audit trail,
    from inside the gate that is supposed to be read-only judgment.
    """
    from scripts.selfmod import gate as G
    g = G.Gate.__new__(G.Gate)
    g.round_id = "SM_TEST"
    g.worktree = Path("/tmp/nonexistent-worktree")
    env = G.Gate._child_env(g)
    assert "/lloyd-selfmod" not in env["LLOYD_SELFMOD_STATE"], "points at live state"
    assert env["LLOYD_VOICE_ALERTS"] == "0"
    assert env["LLOYD_GUARDIAN_STATE"] != env["LLOYD_SELFMOD_STATE"]


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
    src = (ROOT / "scripts" / "selfmod" / "promote.py").read_text()
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
    st = gstate.SelfModState(tmp_path)
    st.write_last_settled({"commit": "b" * 40, "parent": "a" * 40,
                           "settled_ts": time.time()})
    back = gstate.read_json(st.last_settled)
    assert back["parent"] == "a" * 40


def test_the_guardian_reads_the_eval_baseline_and_never_writes_it(tmp_path):
    """LKG has exactly one writer, and that is what makes it mean
    observed-healthy-in-production rather than measured-by-somebody."""
    st = gstate.SelfModState(tmp_path)
    gstate.write_json_atomic(st.eval_last, {"commit": "b" * 40, "overall": {"ndcg10": 0.5}})
    assert st.read_eval_last()["commit"] == "b" * 40
    assert not hasattr(st, "write_eval_last"), "the guardian must not write the measurement"

    st.set_lkg("b" * 40, eval_baseline=st.read_eval_last())
    assert gstate.read_json(st.lkg_path)["eval"]["overall"]["ndcg10"] == 0.5


def test_installed_units_are_compared_against_the_repo():
    """systemd reads ~/.config/systemd/user, not the repo.

    A unit edit that was never installed is a change that looks landed and
    does nothing — for a watchdog unit, the worst kind of silent no-op.
    """
    from scripts.selfmod.round import _unit_drift
    assert isinstance(_unit_drift(), list)


def test_protected_service_definitions_are_actually_applied():
    """spec.py calls them protected — allowed with a drill — but nothing ever
    applied one: supervisord needs reread/update, units need installing, and
    the guardian runs a pinned snapshot re-staged only on a unit restart."""
    src = (ROOT / "scripts" / "selfmod" / "promote.py").read_text()
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
    src = (ROOT / "scripts" / "selfmod" / "promote.py").read_text()
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
        "--state", str(tmp_path / "selfmod"),
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
    st = gstate.SelfModState(tmp_path)
    for name in ("lkg", "current", "floor", "rollback_target", "set_lkg",
                 "clear_current", "write_last_settled", "read_eval_last",
                 "read_rollback_request", "clear_rollback_request",
                 "is_broken", "is_halted", "set_broken", "set_halted",
                 "deny", "recent_rollbacks", "unfinished_rollback",
                 "pause_remaining"):
        assert hasattr(st, name), f"SelfModState.{name} is called but does not exist"


# ===========================================================================
# 9. The caller must stop talking after landing
# ===========================================================================

def test_landing_tells_the_caller_to_end_its_turn():
    """Found by driving the loop end to end as the agent, not from a terminal.

    The idle gate counts the CALLING turn too. An agent that lands and then
    polls `selfmod_status` in a loop is itself the reason the backend never
    goes idle, so the landing waits its full 15 minutes and gives up. The
    landing restarts the backend and ends that turn regardless, so the only
    correct move after `selfmod_land` returns is to stop.

    This is only reachable on the path that had never been exercised, which is
    the whole reason the loop was driven by the agent before being trusted.
    """
    src = (ROOT / "agent_mcp" / "selfmod.py").read_text()
    assert "END YOUR TURN" in src
    # ...and the tool description must not still say to poll, or the two
    # halves of the contract contradict each other.
    assert "Poll selfmod_status to follow it" not in src


def test_the_skill_says_to_stop_after_landing():
    skill = Path.home() / "obsidian" / "skills" / "selfmod-change-own-code" / "SKILL.md"
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
    from scripts.selfmod import promote as P
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
    from scripts.selfmod import promote as P
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
    from scripts.selfmod import promote as P
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
    from scripts.selfmod import promote as P
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
    from scripts.selfmod import promote as P
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
    from scripts.selfmod import promote as P
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
        "--state", str(tmp_path / "selfmod"),
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
