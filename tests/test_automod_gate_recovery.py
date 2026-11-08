"""The `recovery` rung on the gate ladder, and the record it leaves (#3930 clause 4).

The clause: the gate records ONE observe-only recovery record per round — declared
surfaces, witness captured?, the verifier's verdict, or `unrecoverable: <surface>`
— on the durable automod ledger event, the way the `vet` record rides it, and its
presence never changes the round's pass/fail.

Every rung but `recovery` is stubbed, so the code that runs for real is
`Gate._rung` (which is where the ledger event is written) plus
`Gate.rung_recovery` plus the record builder it calls. The changed-path list comes
from a real scratch git repo, because the declaration being driven by preflight's
resolved diff is the process boundary here. The witness READERS are injected, and
that is deliberate: a test must never digest the live tree or open a live store —
measurement belongs to a scratch root, which is the same rule #3930's risk section
writes for the rehearsal.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.automod import gate as G
from scripts.automod import recovery as REC
from scripts.automod import state as S
from scripts.automod import worktree as W

# The real ladder, so a rung added to the gate without being accounted for here
# shows up as a missing record rather than a passing test. `test_automod_gate.py`
# pins this tuple against `Gate.run`; this file pins the recovery row's content.
LADDER = ("preflight", "vet", "code_rules", "recovery", "static", "pyright",
          "prose_consumers", "frontend", "tests", "prompt_surface",
          "review", "venv", "canary_boot", "canary_smoke", "drill")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()


def _scratch_repo(tmp: Path, extra: dict[str, str] | None = None) -> dict:
    """A one-commit repo whose HEAD changes `app/gate.py` plus `extra`.

    `extra` is how a round 'touches' a surface: a file under
    `agent-services/supervisor/conf.d/` is a real tracked path, so the
    declaration comes out of a real diff instead of a hand-written list.
    """
    wt = tmp / "wt"
    (wt / "app").mkdir(parents=True)
    (wt / "app" / "gate.py").write_text("X = 1\n")
    _git(wt, "init")
    _git(wt, "config", "user.email", "recovery@example.invalid")
    _git(wt, "config", "user.name", "Test")
    _git(wt, "add", ".")
    _git(wt, "commit", "-qm", "base")
    base = _git(wt, "rev-parse", "HEAD")
    for rel, body in (extra or {}).items():
        (wt / rel).parent.mkdir(parents=True, exist_ok=True)
        (wt / rel).write_text(body)
    (wt / "app" / "gate.py").write_text("X = 2\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-qm", "head")
    return {"wt": wt, "base": base}


class _Stub(G.Gate):
    """A real Gate whose every rung but `recovery` is scripted.

    Deliberately built without calling `Gate.__init__`, like `_StubGate` in
    `test_automod_gate.py`: the constructor resolves live-tree paths (the venv,
    pyright) that a unit test has no business touching. The two attributes the
    recovery rung actually reads — `report.changed_paths` and `self.live` — are
    set explicitly, from the scratch repo and a scratch root.
    """

    def __init__(self, changed: list[str], *, live_root: Path):
        self.round_id = "SM_RECOVERY"
        self.base = "a" * 40
        self.worktree = Path("/nonexistent-worktree")   # noqa: S106 — never read
        self.live = Path(live_root)
        self.skip_smoke = False
        self._canary = None
        self.item_id = None
        self.report = G.GateReport(round_id=self.round_id, base=self.base,
                                   head="b" * 40)
        self.report.changed_paths = list(changed)
        self.calls: list[str] = []

    def _make(self, name):
        def rung():
            self.calls.append(name)
            return True, "stub", {}
        return rung

    def _rung(self, name: str, fn) -> bool:
        if name != "recovery":
            res = G.RungResult(name, True, "stub", 0.0, {})
            self.report.rungs.append(res)
            self.calls.append(name)
            return True
        return super()._rung(name, fn)          # the real rung, the real event

    def run(self):
        for name in LADDER:
            fn = self.rung_recovery if name == "recovery" else self._make(name)
            if not self._rung(name, fn):
                self.report.ok = False
                return self.report
        self.report.ok = True
        return self.report


@pytest.fixture
def isolate(monkeypatch, tmp_path):
    """No ledger write escapes to disk, no rung cache, no live reader."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    events: list[dict] = []
    monkeypatch.setattr(S, "append_event", lambda entry, *a, **k: events.append(dict(entry)))
    return events


def _readers(calls: list[str]):
    """A scratch reader per op that has an inverse, counting which ones ran."""
    def make(op: str):
        def read():
            calls.append(op)
            return {"digest": "d" * 16, "n": "1"}
        return read
    return {s.op: make(s.op) for s in REC.SURFACES if s.inverse is not None}


def _run(tmp_path, monkeypatch, isolate, *, changed: list[str], calls=None,
         readers=None):
    monkeypatch.setattr(REC, "default_readers",
                        lambda root: dict(readers if readers is not None
                                          else _readers(calls if calls is not None
                                                        else [])))
    g = _Stub(changed, live_root=tmp_path / "live")
    report = g.run()
    return g, report, [e for e in isolate if e.get("rung") == "recovery"]


def _recovery_event(events: list[dict]) -> dict:
    hits = [e for e in events if e.get("rung") == "recovery"]
    assert len(hits) == 1, f"expected exactly one recovery row, got {len(hits)}"
    return hits[0]


# ── the rung is on the ladder, and the round still passes ───────────────────


def test_the_recovery_rung_runs_on_every_round_and_never_fails_it(tmp_path, monkeypatch,
                                                                 isolate):
    calls: list[str] = []
    _, report, events = _run(tmp_path, monkeypatch, isolate,
                             changed=["agent-services/supervisor/conf.d/x.conf",
                                      "app/gate.py"], calls=calls)
    assert [r.name for r in report.rungs] == list(LADDER), \
        "the recovery rung is not on the ladder the gate runs"
    assert report.ok, [r.detail for r in report.rungs if not r.ok]
    rung = next(r for r in report.rungs if r.name == "recovery")
    assert rung.ok is True, "an observe-only rung must never fail a round"
    assert rung.detail.startswith("observe-only:")
    ev = _recovery_event(events)
    assert ev["event"] == "gate" and ev["ok"] is True


def test_the_record_rides_the_durable_event_in_the_shape_the_census_reads(
        tmp_path, monkeypatch, isolate):
    """Field by field, on the row that outlives the worktree. `gate.json` is
    deleted with the round dir, so #3930's published split can only be counted
    from here — the same reason `vet`'s and `pyright`'s records are copied."""
    _, _, events = _run(tmp_path, monkeypatch, isolate,
                        changed=["agent-services/supervisor/conf.d/x.conf",
                                 "app/gate.py"])
    rec = _recovery_event(events)["recovery"]
    assert rec["observe_only"] is True
    assert rec["contract"] == REC.CONTRACT_DECLARED
    assert rec["declared"] == ["supervisor-program-definitions"]
    assert rec["unrecoverable"] == []
    assert rec["verdict"] == REC.UNEVALUATED, \
        "nothing is applied-and-unwound yet, so no round may be credited a restore"
    assert rec["rehearsed"] is False
    assert rec["witness_captured"] == {"supervisor-program-definitions": True}


def test_a_pure_code_round_still_gets_a_record_and_reads_no_witness(
        tmp_path, monkeypatch, isolate):
    """A per-ROUND record, not a per-effect one: the empty contract is one of the
    three buckets the acceptance publishes, so it has to be on the event too — and
    the round pays no witness read for it."""
    calls: list[str] = []
    _, report, events = _run(tmp_path, monkeypatch, isolate,
                             changed=["app/gate.py", "tests/test_x.py"], calls=calls)
    assert report.ok
    rec = _recovery_event(events)["recovery"]
    assert rec["contract"] == REC.CONTRACT_EMPTY
    assert rec["declared"] == []
    assert calls == [], "a round declaring no surface should read no witness at all"


def test_a_surface_with_no_inverse_is_named_unrecoverable_in_the_row_and_the_line(
        tmp_path, monkeypatch, isolate):
    _, report, events = _run(tmp_path, monkeypatch, isolate,
                             changed=["agent-services/guardian/policy.py"])
    assert report.ok, [r.detail for r in report.rungs if not r.ok]
    ev = _recovery_event(events)
    assert ev["recovery"]["unrecoverable"] == ["guardian-pinned-snapshot"]
    assert "unrecoverable: guardian-pinned-snapshot" in ev["detail"], \
        "the word has to be in the line a person reads on the ladder, not only in " \
        "the structured keys that die in `gate.json`"


def test_a_recovery_mismatch_is_reported_and_still_does_not_change_the_verdict(
        tmp_path, monkeypatch, isolate):
    """Observe-only means the VERDICT may be `failed` and the round may not. The
    ladder line says so in words, because a green build beside a red 'undo' is
    precisely the case a reader would otherwise take as fine."""
    monkeypatch.setattr(
        G.REC, "build_recovery_record",
        lambda paths, *, readers=None, live_root=None: {
            "observe_only": True, "contract": REC.CONTRACT_DECLARED,
            "declared": ["knowledge-graph-store"], "unrecoverable": [],
            "verdict": REC.FAILED, "rehearsed": True,
            "surfaces": [{"name": "knowledge-graph-store", "has_inverse": True,
                          "witness_captured": True, "verdict": REC.FAILED,
                          "reason": "kg_rows 41820 -> 41821"}]})
    _, report, events = _run(tmp_path, monkeypatch, isolate, changed=["app/kg_store.py"])
    assert report.ok, "clause 4: its presence never changes the round's pass/fail"
    rung = next(r for r in report.rungs if r.name == "recovery")
    assert rung.ok is True
    assert rung.detail.startswith("RECOVERY MISMATCH")
    assert _recovery_event(events)["recovery"]["verdict"] == REC.FAILED


def test_a_surface_that_cannot_be_read_is_unevaluated_and_never_clean(
        tmp_path, monkeypatch, isolate):
    """The clause's other half on the gate's own path: an unreadable surface must
    not be recorded as a passing check, and a rung that cannot see the state must
    not raise — a raising rung is a FAILED rung to `_rung`, which would be this
    observe-only rung failing a round for the wrong reason."""
    def bad():
        raise FileNotFoundError("kg.sqlite absent")
    _, report, events = _run(tmp_path, monkeypatch, isolate,
                             changed=["app/kg_store.py"], calls=[],
                             readers={"kg_rows": bad})
    assert report.ok, [r.detail for r in report.rungs if not r.ok]
    ev = _recovery_event(events)
    rec = ev["recovery"]
    assert rec["verdict"] == REC.UNEVALUATED
    assert rec["witness_captured"] == {"knowledge-graph-store": False}
    assert "kg.sqlite absent" in rec["unevaluated_reasons"]["knowledge-graph-store"]


def test_the_declaration_follows_the_real_diff_of_the_round(tmp_path, monkeypatch,
                                                           isolate):
    """The boundary the clause lives across: preflight resolves the changed-path
    list from `git diff base...HEAD`, and the declaration must be the answer to
    THAT list. Built from a scratch repo rather than asserted from a literal."""
    repo = _scratch_repo(tmp_path, extra={
        "agent-services/systemd/lloyd-backend.service": "[Service]\nType=simple\n"})
    changed = W.changed_paths(repo["wt"], repo["base"])
    assert "agent-services/systemd/lloyd-backend.service" in changed
    calls: list[str] = []
    monkeypatch.setattr(REC, "default_readers", lambda root: _readers(calls))
    g = _Stub(changed, live_root=tmp_path / "live")
    report = g.run()
    assert report.ok
    declared = REC.declare_surfaces(changed)
    assert "systemd-user-unit-set" in [s.name for s in declared]
    assert set(calls) == {s.op for s in declared if s.inverse}, \
        "the witness was not captured for every declared surface with an inverse"
