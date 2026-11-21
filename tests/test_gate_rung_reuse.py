"""A re-gate should not replay what cannot have changed.

A gate with review runs 7-12 minutes and every re-gate replays the whole
ladder, so a 60-minute round gets about one fix cycle. The ladder
short-circuits at the first failure, so after a review refusal only the rungs
*before* review have a cached pass — about 146s. The larger win is the
promoter's rebase chase, which re-runs the ladder twice more against a moved
base for a diff that has not changed at all.

Four conditions, all of them one question — is this still the same question:
the config allows it, the base has not moved (a rebase changes what the diff
MEANS), the entry is fresh, and the cached head is an ancestor of this one.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest

from scripts.automod import gate as G


class _Gate:
    """A Gate with just enough state for the reuse machinery."""

    def __init__(self, tmp_path, base="BASE", head="HEAD2", cached=None):
        self.round_id = "SM_TEST"
        self.base = base
        self.worktree = tmp_path / "wt"
        self.worktree.mkdir(parents=True, exist_ok=True)
        self._dir = tmp_path / "round"
        (self._dir / "gate-state").mkdir(parents=True, exist_ok=True)
        self.report = type("R", (), {"head": head, "rungs": [], "changed_paths": []})()
        self._head = head
        if cached is not None:
            (self._dir / "gate-state" / "rung_cache.json").write_text(
                json.dumps(cached))

    # bind the real methods
    _quiet_skip = G.Gate._quiet_skip
    QUIET_SKIP_RUNGS = G.Gate.QUIET_SKIP_RUNGS
    EVENT_ID_CAP = G.Gate.EVENT_ID_CAP
    _reuse = G.Gate._reuse
    _reuse_inner = G.Gate._reuse_inner
    _reuse_rule = staticmethod(G.Gate._reuse_rule)
    _reuse_load = G.Gate._reuse_load
    _reuse_save = G.Gate._reuse_save
    REUSE_MAX_AGE_S = G.Gate.REUSE_MAX_AGE_S

    def _reuse_path(self):
        return self._dir / "gate-state" / "rung_cache.json"


@pytest.fixture
def stub_git(monkeypatch):
    """`merge-base --is-ancestor` succeeds; `diff --name-only` returns state."""
    state = {"delta": [], "ancestor": True}

    def _run(cmd, cwd=None, env=None, timeout=900.0):
        import types
        if "merge-base" in cmd:
            return types.SimpleNamespace(
                returncode=0 if state["ancestor"] else 1, stdout="", stderr="")
        if "diff" in cmd:
            return types.SimpleNamespace(
                returncode=0, stdout="\n".join(state["delta"]), stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(G, "_run", _run)
    monkeypatch.setattr(G.W, "head", lambda wt: "HEAD2")
    return state


def _entry(head="HEAD1", base="BASE", age_s=10.0, data=None):
    return {"base": base, "head": head, "ts": time.time() - age_s,
            "data": data or {"ok": 1}}


def test_a_rung_is_reused_when_nothing_relevant_changed(tmp_path, stub_git):
    stub_git["delta"] = ["app/harness/loop.py"]
    g = _Gate(tmp_path, cached={"frontend": _entry()})
    got = g._reuse("frontend")
    assert got is not None
    data, why = got
    assert data == {"ok": 1}
    # #3770 clause 3 renamed this rule's excuse, because the branch now asks
    # `G.frontend_changed(delta)` — which admits the two granted chrome-extension paths as
    # well as `web/` — so "no web/ path" was no longer the measurement it reported. The
    # chrome-only delta that DOES re-run the rung is pinned beside it, in
    # tests/test_automod_gate.py::test_a_chrome_only_delta_re_rungs_the_rung_rather_than_reusing_a_green.
    assert "no frontend path" in why


def test_a_rung_is_re_run_when_its_own_files_moved(tmp_path, stub_git):
    stub_git["delta"] = ["web/src/api.ts"]
    g = _Gate(tmp_path, cached={"frontend": _entry()})
    assert g._reuse("frontend") is None


def test_a_panel_only_commit_re_rungs_the_rung_rather_than_reusing_a_green(tmp_path,
                                                                          stub_git):
    """#4171 clause 1's second half, at the seam the reuse rule owns.

    `frontend_changed` answering non-empty is only half of it: a cached green from an
    earlier round is keyed on the SAME predicate, so if the reuse rule resolved the panel
    paths a different way, a commit that rewrites `dist/sidepanel.html` and a chunk would
    inherit the previous round's verdict and ship a panel no build has ever emitted. This
    is the leg the worker pair has had since #3770 and the panel half never did.

    The byte that matters is the one on the line above: `test_a_rung_is_reused_when_nothing_
    relevant_changed` asserts `web/src/api.ts` alone still reuses. So the reuse reader is
    the same function the rung's own gate reads, and a widening that reached one and not the
    other would leave a rebuild commit reusing a green — which is what this node exists to
    refuse.
    """
    stub_git["delta"] = ["chrome-extension/dist/sidepanel.html",
                        "chrome-extension/dist/assets/sidepanel-Bf8Ekfot.js"]
    g = _Gate(tmp_path, cached={"frontend": _entry()})
    assert g._reuse("frontend") is None, (
        "a commit naming only the shipped panel half reused a previous round's green, so "
        "the bytes it shipped were never compared to any build")


def test_a_moved_base_invalidates_everything(tmp_path, stub_git):
    """A rebase changes what the diff MEANS, so every cached answer is about
    a different question.
    """
    stub_git["delta"] = ["docs/x.md"]
    g = _Gate(tmp_path, base="NEWBASE", cached={"frontend": _entry(base="BASE")})
    assert g._reuse("frontend") is None


def test_a_stale_entry_is_not_reused(tmp_path, stub_git):
    stub_git["delta"] = ["docs/x.md"]
    g = _Gate(tmp_path, cached={"frontend": _entry(age_s=99_999)})
    assert g._reuse("frontend") is None


def test_a_head_that_is_not_an_ancestor_is_not_reused(tmp_path, stub_git):
    """An amended or reset commit is not "the same work plus more"."""
    stub_git["delta"] = ["docs/x.md"]
    stub_git["ancestor"] = False
    g = _Gate(tmp_path, cached={"frontend": _entry()})
    assert g._reuse("frontend") is None


@pytest.mark.parametrize("rung", ["preflight", "static", "review"])
def test_three_rungs_always_run(tmp_path, stub_git, rung):
    """`preflight` rebases and must always run; `static` is milliseconds;
    `review` has its own patch-id reuse, which is a stronger test than
    "no relevant file changed".
    """
    stub_git["delta"] = ["docs/x.md"]
    g = _Gate(tmp_path, cached={rung: _entry()})
    assert g._reuse(rung) is None


def test_tests_is_never_reused_bare(tmp_path, stub_git):
    """A cached pass says the suite was green at an earlier commit; any code
    change since could have broken anything.
    """
    stub_git["delta"] = ["tests/test_x.py"]
    g = _Gate(tmp_path, cached={"tests": _entry()})
    assert g._reuse("tests") is None


@pytest.mark.parametrize("delta,reused", [
    (["tests/test_a.py"], True),
    (["docs/x.md"], True),
    (["app/harness/loop.py"], False),
    (["tests/test_a.py", "app/x.py"], False),
])
def test_a_canary_is_reused_only_for_tests_and_docs(tmp_path, stub_git,
                                                    delta, reused):
    """A canary boots the candidate: any code change invalidates it.

    Both canary entries are cached green here, which is the only state in which
    a boot may be reused at all (#4347: a boot reused with no smoke entry leaves
    `canary_smoke` dereferencing an unset `_canary`). With the pair present the
    delta rule is what decides, identically for both names.
    """
    stub_git["delta"] = delta
    g = _Gate(tmp_path, cached={"canary_boot": _entry(), "canary_smoke": _entry()})
    assert (g._reuse("canary_boot") is not None) is reused
    assert (g._reuse("canary_smoke") is not None) is reused


def test_an_empty_delta_reuses_everything(tmp_path, stub_git):
    """An empty delta still re-runs one rung: `canary_boot` alone.

    #4347: a boot reuse without a smoke reuse leaves this process with
    `_canary` unset while `canary_smoke` runs, which is the
    `'NoneType' object has no attribute 'smoke'` death in 5 gate rows over 4
    rounds (newest SM_20261117_132548, 2026-11-17). So the empty delta re-runs
    the boot, and everything on the reuse path beside it is unchanged.
    """
    stub_git["delta"] = []
    g = _Gate(tmp_path, cached={"canary_smoke": _entry()})
    got = g._reuse("canary_smoke")
    assert got is not None and "nothing committed" in got[1]
    boot = _Gate(tmp_path, cached={"canary_boot": _entry()})
    assert boot._reuse("canary_boot") is None, (
        "the boot was reused with no smoke beside it, so canary_smoke would run "
        "against a process that never built a canary object")


def test_a_reused_boot_without_a_cached_smoke_is_refused(tmp_path, stub_git):
    """The invariant, on the two deltas that used to reuse the boot anyway.

    A reused `canary_boot` implies a reused `canary_smoke`: only a PASSING rung
    is cached (`if ok: self._reuse_save(...)`), so a smoke that genuinely
    FAILED leaves no entry — and that is exactly the re-gate where the live turn
    most needs to run again. SM_20261117_132548 is that state: boot green at
    14:14:24Z, smoke FAIL at 14:14:29Z, then both re-gates show `canary_boot`
    REUSED 0.0s followed by `canary_smoke` FAIL 0.0s.
    """
    for delta in ([], ["docs/x.md"]):
        stub_git["delta"] = delta
        g = _Gate(tmp_path, cached={"canary_boot": _entry()})
        assert g._reuse("canary_boot") is None, (
            f"boot-only reuse survived a {delta} delta")


def test_a_boot_still_reuses_when_its_smoke_reuses_beside_it(tmp_path, stub_git):
    """Bounded guard, not a capability removal: with both entries green the
    boot still reuses, and its reason string is the unchanged one, so no rung
    comes off the reuse path wholesale.
    """
    stub_git["delta"] = []
    g = _Gate(tmp_path, cached={"canary_boot": _entry(), "canary_smoke": _entry()})
    got = g._reuse("canary_boot")
    assert got is not None, "the guard took the boot off the reuse path"
    assert got[1] == "nothing committed since", f"the reason changed: {got[1]!r}"


def test_a_skipped_smoke_keeps_its_boot_reusable(tmp_path, stub_git):
    """The engine-unreachable case pays nothing.

    A skipped smoke is a passing rung, so it is cached like any green and the
    boot keeps reusing; `skip_smoke` never has to be read here, which is what
    keeps `_reuse_rule` the staticmethod the harness binds.
    """
    stub_git["delta"] = ["docs/x.md"]
    g = _Gate(tmp_path, cached={"canary_boot": _entry(),
                                "canary_smoke": _entry(data={"skipped": True})})
    got = g._reuse("canary_boot")
    assert got is not None, (
        "the guard refused a boot whose smoke is cached as SKIPPED, so an "
        "engine-unreachable box would pay a boot it does not need")
    assert got[1] == "only tests and docs in the delta", f"reason: {got[1]!r}"


@pytest.mark.parametrize("name,delta,cached,expect", [
    # Every delta `_reuse_rule` answers for a name, over the cache shapes that
    # decide it: the boot guard must not touch any of these answers.
    ("drill", ["docs/x.md"], {"drill": _entry()}, "only tests and docs in the delta"),
    ("drill", [], {"drill": _entry()}, "nothing committed since"),
    ("venv", ["SETUP.md"], {"venv": _entry()}, "no requirements path in the delta"),
    ("venv", ["requirements.txt"], {"venv": _entry()}, None),
    ("frontend", ["docs/x.md"], {"frontend": _entry()}, "no frontend path in the delta"),
    ("frontend", ["web/src/api.ts"], {"frontend": _entry()}, None),
    ("prompt_surface", ["docs/x.md"], {"prompt_surface": _entry()},
     "no prompt-surface path in the delta"),
    ("prompt_surface", ["app/prompt_builder.py"], {"prompt_surface": _entry()}, None),
    ("tests", ["tests/test_a.py"], {"tests": _entry()}, None),
])
def test_the_boot_guard_moves_no_other_rung_s_answer(tmp_path, stub_git, name, delta,
                                                    cached, expect):
    stub_git["delta"] = delta
    g = _Gate(tmp_path, cached=cached)
    got = g._reuse(name)
    assert (got[1] if got else None) == expect


@pytest.mark.parametrize("name,delta,expect", [
    ("drill", ["docs/x.md"], "only tests and docs in the delta"),
    ("drill", [], "nothing committed since"),
    ("venv", ["SETUP.md"], "no requirements path in the delta"),
    ("frontend", ["docs/x.md"], "no frontend path in the delta"),
    ("prompt_surface", ["docs/x.md"], "no prompt-surface path in the delta"),
    ("tests", ["tests/test_a.py"], None),
])
def test_the_rule_still_answers_a_two_argument_call_with_no_cache(name, delta, expect):
    """`_reuse_rule(name, delta)` is the call the harness and any other reader
    makes: the cache arrives as an optional third argument, so a two-argument
    call for a name the guard does not own answers exactly as it did before.
    """
    assert G.Gate._reuse_rule(name, delta) == expect


def test_a_boot_rule_call_with_no_cache_refuses_the_boot():
    """The one name the guard owns is refused on an UNKNOWN cache.

    A caller that cannot show a green `canary_smoke` entry cannot show the
    invariant, and the cost of refusing is one ~30 s boot against a gate that
    dies at `canary_smoke` with no canary object at all.
    """
    assert G.Gate._reuse_rule("canary_boot", []) is None
    assert G.Gate._reuse_rule("canary_boot", ["docs/x.md"]) is None
    assert (G.Gate._reuse_rule("canary_boot", [], {"canary_smoke": _entry()})
            == "nothing committed since")


def test_a_smoke_without_its_canary_fails_by_name_rather_than_raising():
    """The residual case, diagnosable instead of an engine failure.

    Before #4347 this raised `AttributeError: 'NoneType' object has no attribute
    'smoke'` at gate.py's bare `self._canary.smoke(...)`, and the rung wrapper
    reported it as `AttributeError: ...` at 0.0s — what a reader saw in the
    ledger was an engine fault, not "the boot was reused without the smoke".
    """
    class _Bare:
        skip_smoke = False
        _canary = None
        rung_canary_smoke = G.Gate.rung_canary_smoke

    ok, detail, data = _Bare().rung_canary_smoke()
    assert ok is False, "a smoke with no canary object passed"
    assert "canary_smoke" in detail, f"the detail does not name the rung: {detail!r}"
    assert "boot" in detail.lower(), f"the detail does not name the missing boot: {detail!r}"
    assert isinstance(data, dict)


def test_the_ladder_still_boots_the_canary_immediately_before_the_smoke(monkeypatch):
    """The pair is the invariant's other half: the boot builds the object the
    next rung dereferences, so a reorder would put the smoke in front of the
    only thing that ever sets `_canary`.
    """
    monkeypatch.setattr(G, "engine_reachable", lambda root, timeout=4.0: (False, "stub"))
    names = []

    class _StubGate(G.Gate):
        def __init__(self):
            super().__init__("SM_TEST_LADDER", G.Path(__file__).resolve().parents[1],
                             "HEAD", skip_smoke=True)

    def _record(name, fn):
        """Stand in for the rung wrapper: record the rung reached, answer PASS.

        The PASS is this stub's own return value and carries no claim — it is
        there so the ladder, which short-circuits on a failing rung, walks to
        the end and `names` becomes the ladder's order. The two asserts below
        are the whole check, and a reorder fails them.
        """
        names.append(name)
        return True

    g = _StubGate()
    monkeypatch.setattr(g, "_rung", _record)
    g.run()
    assert names.index("canary_boot") == names.index("canary_smoke") - 1, names
    assert names == ["preflight", "vet", "code_rules", "recovery", "static", "pyright",
                     "prose_consumers", "frontend", "tests", "prompt_surface", "review",
                     "venv", "canary_boot", "canary_smoke", "drill"], names


@pytest.mark.parametrize("delta,reused", [
    (["requirements.txt"], False),
    (["requirements.lock"], False),
    # #1073: the dev-only requirements file is neither a trigger here nor an
    # install target, which is what makes it a safe home for an optional solver.
    (["requirements-dev.txt"], True),
    (["SETUP.md"], True),
    (["requirements.txt", "requirements-dev.txt"], False),
])
def test_the_venv_rung_reuses_unless_a_real_install_target_moved(tmp_path, stub_git,
                                                                 delta, reused):
    """Two surfaces decide whether a candidate venv gets rebuilt, and #1073's
    hazard has to be closed on both: `_reuse_rule` (gate.py:665, whose delta
    check is :674) and `spec.touches_requirements` inside `rung_venv`
    (gate.py:1800). They name the
    same two files, so a `requirements-dev.txt` delta keeps the cached verdict
    instead of resurrecting a rebuild whose install list cannot contain it.
    """
    stub_git["delta"] = delta
    g = _Gate(tmp_path, cached={"venv": _entry()})
    assert (g._reuse("venv") is not None) is reused


def test_the_venv_rung_installs_from_two_named_files():
    """The install list is a two-name expression, not a glob over `requirements*`.

    `lock if lock.exists() else requirements.txt`, cloned from the live venv —
    so a package that lives only in `requirements-dev.txt` cannot reach a
    candidate, and the live venv and the candidate cannot disagree about whether
    `import z3` succeeds. If a third name ever appears here, or a wildcard, this
    is the test that says the #1073 route stopped being a dev-only route.
    """
    src = inspect.getsource(G.Gate.rung_venv)
    assert "requirements-dev" not in src
    assert 'lock = self.worktree / "requirements.lock"' in src
    assert 'req = lock if lock.exists() else self.worktree / "requirements.txt"' in src


def test_the_kill_switch_disables_reuse(tmp_path, stub_git, monkeypatch):
    monkeypatch.setattr(G, "_gate_cfg",
                        lambda k, d: False if k == "reuse_rungs" else d)
    stub_git["delta"] = ["docs/x.md"]
    g = _Gate(tmp_path, cached={"frontend": _entry()})
    assert g._reuse("frontend") is None


def test_an_unreadable_cache_runs_the_rung(tmp_path, stub_git):
    """Fail toward running it. A failure to READ the cache must never skip a
    check.
    """
    g = _Gate(tmp_path)
    g._reuse_path().write_text("{ this is not json")
    assert g._reuse("frontend") is None


def test_a_save_that_cannot_write_does_not_fail_the_rung(tmp_path, monkeypatch):
    """Every attribute `_reuse_save` touches is one a caller could have left
    off, and a rung that passed must never be reported as failed because its
    bookkeeping did.
    """
    class _Bare:
        _reuse_save = G.Gate._reuse_save
        _reuse_load = G.Gate._reuse_load
        _reuse_path = G.Gate._reuse_path
    # No worktree, no report, no round_id: everything raises.
    _Bare()._reuse_save("frontend", {})      # must not raise


def test_a_reused_rung_is_recorded_as_such(tmp_path, stub_git, monkeypatch):
    """The report and the ledger have to say a rung was not actually run, or
    the gate's own record stops meaning what it says.
    """
    events = []
    monkeypatch.setattr(G.S, "append_event", lambda e: events.append(e))
    stub_git["delta"] = ["docs/x.md"]

    g = _Gate(tmp_path, cached={"frontend": _entry(data={"changed_web": []})})
    g._rung = G.Gate._rung.__get__(g)
    ran = []
    assert g._rung("frontend", lambda: ran.append(1) or (True, "ok", {})) is True
    assert not ran, "the rung was actually executed"
    assert g.report.rungs[0].detail.startswith("REUSED")
    assert g.report.rungs[0].data["reused"] is True
    assert events[0]["reused"] is True
    assert events[0]["ok"] is True


# ---------------------------------------------------------------------------
# the tests rung's partial run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta,expected", [
    (["tests/test_a.py"], ["tests/test_a.py"]),
    (["tests/test_a.py", "tests/test_b.py"], ["tests/test_a.py", "tests/test_b.py"]),
    # A conftest changes how EVERY other test runs.
    (["tests/test_a.py", "tests/conftest.py"], None),
    # So does a shared helper.
    (["tests/test_a.py", "tests/_helpers.py"], None),
    # Any code path: the suite has to run.
    (["tests/test_a.py", "app/x.py"], None),
    (["app/x.py"], None),
    # Docs are not test files either — the rule is "only test files", not
    # "nothing risky", because a partial run skips the floors.
    (["tests/test_a.py", "README.md"], None),
    ([], None),
])
def test_which_deltas_earn_a_partial_test_run(tmp_path, stub_git, delta, expected):
    stub_git["delta"] = delta
    g = _Gate(tmp_path, cached={"tests": _entry()})
    g._tests_delta_only = G.Gate._tests_delta_only.__get__(g)
    assert g._tests_delta_only() == expected


def test_no_cached_tests_run_means_no_partial(tmp_path, stub_git):
    stub_git["delta"] = ["tests/test_a.py"]
    g = _Gate(tmp_path)          # empty cache
    g._tests_delta_only = G.Gate._tests_delta_only.__get__(g)
    assert g._tests_delta_only() is None


def test_a_moved_base_means_no_partial(tmp_path, stub_git):
    stub_git["delta"] = ["tests/test_a.py"]
    g = _Gate(tmp_path, base="NEWBASE", cached={"tests": _entry(base="BASE")})
    g._tests_delta_only = G.Gate._tests_delta_only.__get__(g)
    assert g._tests_delta_only() is None


# ---------------------------------------------------------------------------
# the skipped flag
# ---------------------------------------------------------------------------

def test_the_pytest_skip_count_is_not_the_rung_flag():
    counts = G._parse_pytest_summary("1600 passed, 12 skipped in 30s")
    assert counts["tests_skipped"] == 12
    assert "skipped" not in counts


@pytest.mark.parametrize("data,flag", [
    ({"skipped": True}, True),
    ({"skipped": False}, False),
    ({"tests_skipped": 12, "passed": 1600}, False),   # the bug
    ({}, False),
    ({"skipped": 3}, False),                          # truthy but not a flag
])
def test_only_a_real_flag_records_a_skipped_rung(tmp_path, monkeypatch, data, flag):
    events = []
    monkeypatch.setattr(G.S, "append_event", lambda e: events.append(e))
    g = _Gate(tmp_path)
    g._rung = G.Gate._rung.__get__(g)
    g._rung("tests", lambda: (True, "ok", data))
    assert events[0]["skipped"] is flag
