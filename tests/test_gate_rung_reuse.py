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
    assert "no web/ path" in why


def test_a_rung_is_re_run_when_its_own_files_moved(tmp_path, stub_git):
    stub_git["delta"] = ["web/src/api.ts"]
    g = _Gate(tmp_path, cached={"frontend": _entry()})
    assert g._reuse("frontend") is None


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
    """A canary boots the candidate: any code change invalidates it."""
    stub_git["delta"] = delta
    g = _Gate(tmp_path, cached={"canary_boot": _entry()})
    assert (g._reuse("canary_boot") is not None) is reused


def test_an_empty_delta_reuses_everything(tmp_path, stub_git):
    stub_git["delta"] = []
    g = _Gate(tmp_path, cached={"canary_smoke": _entry()})
    got = g._reuse("canary_smoke")
    assert got is not None and "nothing committed" in got[1]


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
