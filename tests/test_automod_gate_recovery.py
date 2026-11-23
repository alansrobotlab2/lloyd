"""The `recovery` rung on the gate ladder, and the record it leaves (#3930 clause 4).

The clause: the gate records ONE observe-only recovery record per round — declared
surfaces, witness captured?, the verifier's verdict, or `unrecoverable: <surface>`
— on the durable automod ledger event, the way the `vet` record rides it, and its
presence never changes the round's pass/fail.

Every rung but `recovery` is stubbed, so the code that runs for real is
`Gate._rung` (which is where the ledger event is written) plus
`Gate.rung_recovery` plus the record builder it calls. The changed-path list comes
from a real scratch git repo, because the declaration being driven by preflight's
resolved diff is the process boundary here. The witness READERS are injected where a
test is about the ledger shape or an unreadable surface, and left REAL — reading a
scratch root inside `tmp_path` — wherever a test is about #4082's rehearsal, because
a rehearsal measured through a reader that ignores its root proves nothing. Either
way no test digests the live checkout or opens a live store: measurement belongs to a
scratch root, which is the same rule #3930's risk section writes for the rehearsal.

Clause 4 of #3930 is the row and the ladder line; the #4082 clauses are that the rung
supplies the disposable copy and deletes it, that it writes nothing outside it, and
that a rehearsal which comes back `failed` still leaves the round passing.
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
          "review", "venv", "judgment_replay", "canary_boot", "canary_smoke", "drill")


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
    pyright) that a unit test has no business touching. The three attributes the
    recovery rung actually reads — `report.changed_paths`, `self.live` and
    (since #4082) `self.worktree`, whose bytes the rehearsal applies — are set
    explicitly, from a scratch root and, for a file-scope round, a scratch
    worktree that really holds the round's changed files.
    """

    def __init__(self, changed: list[str], *, live_root: Path,
                 worktree: Path | None = None):
        self.round_id = "SM_RECOVERY"
        self.base = "a" * 40
        # A round that declares a FILE-scope surface makes the rung copy and read
        # this, so such a round must pass a real tree; a pure-code round never does.
        self.worktree = Path(worktree or "/nonexistent-worktree")
        self.live = Path(live_root)
        self.skip_smoke = False
        self._canary = None
        self.item_id = None
        self.report = G.GateReport(round_id=self.round_id, base=self.base,
                                   head="b" * 40)
        self.report.changed_paths = list(changed)
        self.calls: list[str] = []
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
    # `Gate._reuse_path` is `W.round_dir(round_id)/gate-state/rung_cache.json`, and
    # every stub in this file shares the round id `SM_RECOVERY`, so unpatched the
    # rung cache of one test is the next test's `REUSED (nothing committed since)`
    # line within `REUSE_MAX_AGE_S` — and the write lands in the real ~/lloyd-work.
    # A recovery test that reports a REUSED verdict is measuring the previous test.
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "lloyd-work")
    # And no rung is carried over from a previous round: `_reuse_inner` reads the
    # cache through `W.WORK_ROOT` too, so with the root redirected to a fresh
    # `tmp_path` there is nothing to read and `rung_recovery` really runs.
    monkeypatch.setattr(G, "_gate_cfg",
                        lambda key, default=None: {
                            "reuse_rungs": False,
                            "skip_smoke": False,
                            "reuse_max_age_s": 0}.get(key, default))
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
         readers=None, worktree: Path | None = None):
    # `unit_dir` is in the signature because the rehearsal asks `default_readers`
    # for a reader over a SCRATCH copy of the unit directory: a seam that dropped
    # the root would silently measure the live machine (#4082).
    monkeypatch.setattr(REC, "default_readers",
                        lambda root, unit_dir=None: dict(
                            readers if readers is not None
                            else _readers(calls if calls is not None else [])))
    g = _Stub(changed, live_root=tmp_path / "live", worktree=worktree)
    report = g.run()
    return g, report, [e for e in isolate if e.get("rung") == "recovery"]


def _recovery_event(events: list[dict]) -> dict:
    hits = [e for e in events if e.get("rung") == "recovery"]
    assert len(hits) == 1, f"expected exactly one recovery row, got {len(hits)}"
    return hits[0]


CONF_REL = "agent-services/supervisor/conf.d/lloyd-backend.conf"


def _live_and_candidate(tmp: Path, rel: str = CONF_REL,
                        base: str = "command=/old/start\n",
                        head: str = "command=/new/start\n") -> tuple[Path, Path]:
    """A scratch machine and a scratch round, as two directories.

    `live` is what the box holds before the landing; `cand` is the round's own
    worktree, which is where #4082's rehearsal reads the changed bytes from. Both
    are inside the test's tmp_path, so the rung's readers — the REAL ones, not
    injected — can be used without ever pointing at the live checkout. That is the
    point: the gate test exercises the rung, and a rung whose scratch root is a
    `/nonexistent` stub measures nothing.
    """
    live, cand = tmp / "live", tmp / "cand"
    for root, body in ((live, base), (cand, head)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return live, cand


CONF_REL = "agent-services/supervisor/conf.d/lloyd-backend.conf"


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


def test_the_record_rides_the_durable_event_in_the_shape_the_census_reads(tmp_path,
                                                                          isolate):
    """Field by field, on the row that outlives the worktree. `gate.json` is
    deleted with the round dir, so #3930's published split can only be counted
    from here — the same reason `vet`'s and `pyright`'s records are copied.

    This test used to assert `unevaluated` and `rehearsed: False`, because the rung
    had no scratch to work in. #4082 gave it one, so the row this rung writes for a
    file-scope round is a MEASURED round-trip, and the per-surface verdicts are on
    the event too — which is the only form in which the census can count it.
    """
    live, cand = _live_and_candidate(tmp_path)
    g = _Stub(["app/gate.py", CONF_REL], live_root=live, worktree=cand)
    report = g.run()
    assert report.ok, [r.detail for r in report.rungs if not r.ok]
    rec = _recovery_event(isolate)["recovery"]
    assert rec["observe_only"] is True
    assert rec["contract"] == REC.CONTRACT_DECLARED
    assert rec["declared"] == ["supervisor-program-definitions"]
    assert rec["unrecoverable"] == []
    assert rec["verdict"] == REC.VERIFIED, \
        "the rung supplies a scratch copy now, so a file-scope round is credited " \
        "with the round-trip it was measured, not left at `unevaluated`"
    assert rec["rehearsed"] is True
    assert rec["surface_verdicts"] == {"supervisor-program-definitions":
                                       REC.VERIFIED}
    assert rec["witness_captured"] == {"supervisor-program-definitions": True}


def test_a_round_with_no_file_scope_surface_makes_no_scratch_copy(tmp_path, monkeypatch,
                                                                 isolate):
    """The cost clause: the copy exists only for a round that has a file-level undo
    to rehearse, so an ordinary code round still pays nothing for this rung."""
    made: list[Path] = []
    real = G.tempfile.mkdtemp
    monkeypatch.setattr(G.tempfile, "mkdtemp",
                        lambda *a, **k: (made.append(Path(real(*a, **k))) or
                                         made[-1]))
    _, report, _ = _run(tmp_path, monkeypatch, isolate,
                        changed=["app/gate.py", "tests/test_x.py"], calls=[])
    assert report.ok
    assert made == [], "a round declaring no surface must not get a scratch copy"


def test_the_rung_makes_its_scratch_copy_under_temp_and_removes_it(tmp_path, monkeypatch,
                                                                  isolate):
    """Where the rehearsal worked, and that it cleaned up afterwards.

    The rung is the caller that supplies the disposable copy, so two things have to
    hold on the gate's own path: the directory is a fresh temp dir rather than
    anything under the live tree, and it is gone when the rung returns. A scratch
    rehearsal that left its applied bytes behind would be a gate littering the box
    with a copy of its own service definitions.
    """
    live, cand = _live_and_candidate(tmp_path)
    made: list[Path] = []
    real = G.tempfile.mkdtemp

    def spy(*a, **k):
        d = Path(real(*a, **k))
        made.append(d)
        return d

    monkeypatch.setattr(G.tempfile, "mkdtemp", spy)
    before = (live / CONF_REL).read_text()
    g = _Stub([CONF_REL], live_root=live, worktree=cand)
    report = g.run()
    assert report.ok
    assert len(made) == 1, f"one copy per round that declares one: {made}"
    assert not made[0].exists(), "the rung must remove the copy it made"
    assert (live / CONF_REL).read_text() == before, \
        "the live tree is seeded into the copy and never written by it"
    assert _recovery_event(isolate)["recovery"]["verdict"] == REC.VERIFIED


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
        lambda paths, *, readers=None, live_root=None, **kw: {
            "observe_only": True, "contract": REC.CONTRACT_DECLARED,
            "declared": ["knowledge-graph-store"], "unrecoverable": [],
            "verdict": REC.FAILED, "rehearsed": True,
            "surfaces": [{"name": "knowledge-graph-store", "has_inverse": True,
                          "witness_captured": True, "verdict": REC.FAILED,
                          "rehearsed": True,
                          "reason": "kg_rows 41820 -> 41821",
                          "mismatch": ["kg_rows"]}]})
    _, report, events = _run(tmp_path, monkeypatch, isolate, changed=["app/kg_store.py"])
    assert report.ok, "clause 4: its presence never changes the round's pass/fail"
    rung = next(r for r in report.rungs if r.name == "recovery")
    assert rung.ok is True
    assert rung.detail.startswith("RECOVERY MISMATCH")
    assert _recovery_event(events)["recovery"]["verdict"] == REC.FAILED


def test_a_failed_rehearsal_on_the_rungs_own_path_still_leaves_the_round_passing(
        tmp_path, monkeypatch, isolate):
    """Clause 4, with no synthetic record: the rehearsal REALLY runs, its undo is
    stubbed to a no-op, the verdict comes back `failed`, and the round is untouched.

    The one thing this rung must never do is fail a build — so `report.ok` stays
    True and the ladder line says `observe-only` in the same breath as the mismatch,
    because a green build beside a red 'undo' is exactly what a reader would
    otherwise take as fine.
    """
    live, cand = _live_and_candidate(tmp_path)
    monkeypatch.setitem(REC.SCRATCH_INVERSES, "supervisor_conf",
                        lambda ctx: (lambda: None))
    g = _Stub([CONF_REL], live_root=live, worktree=cand)
    report = g.run()

    assert report.ok, "clause 4: its presence never changes the round's pass/fail"
    rung = next(r for r in report.rungs if r.name == "recovery")
    assert rung.ok is True
    assert rung.detail.startswith("RECOVERY MISMATCH — observe-only:"), rung.detail
    rec = _recovery_event(isolate)["recovery"]
    assert rec["verdict"] == REC.FAILED
    assert rec["rehearsed"] is True
    assert rec["surface_verdicts"] == {"supervisor-program-definitions": REC.FAILED}
    assert rec["mismatch"] == {"supervisor-program-definitions": ["tree"]}, \
        "the field that did not come back has to survive on the event"


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
