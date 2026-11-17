"""`scripts/automod/recovery.py` — the witness/verify pair (#3930).

Clauses this file pins:

  1. a PURE function maps a round's changed paths to the declared non-git effect
     surfaces from a checked-in table that holds a supervisor path, a systemd-unit
     path, a store-owning module path and a `config.yaml` key, and returns NOTHING
     for a pure-code diff;
  2. witness capture and restore verification return exactly three verdicts —
     `verified` when every captured field matches, `failed` when any differs,
     `unevaluated` when the surface is unreadable or has no restore op — and an
     empty field list is NEVER read as verified;
  3. on a scratch fixture, registering the restore op as a NO-OP makes the same
     check return `failed`, and registering the real restore op round-trips it to
     `verified` with the digest restored exactly — so the check cannot pass
     unconditionally.

Clause 3 is the one that matters. A recovery check that passes whatever happens is
the same instrument #703's drill had to carry a broken-path clause for, and the
reason `vet` refuses an empty diff: a guard reporting OK on its own missing input.

Clauses added by #4082 (#3930 step 3, scoped to file scope), pinned at the bottom
of this file:

  4. a round touching a FILE-scope surface — supervisor-program-definitions,
     systemd-user-unit-set or effective-runtime-config — is applied to a copy of the
     affected paths and unwound there, and comes back `verified` with
     `rehearsed: True`, while a `runner` that raises on any call is never invoked and
     no path outside the copy is created or modified;
  5. on the same fixture, stubbing that surface's `scratch_inverse` as a no-op makes
     the rehearsal return `failed` and name the mismatched witness field — the
     rehearsal cannot pass unconditionally;
  6. a round touching only a STORE surface keeps `unevaluated` with the reason that
     the available restore is coarser than one round, and
     `agent-services/guardian/**` keeps `unrecoverable: guardian-pinned-snapshot`,
     whether or not a scratch copy is on offer.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from scripts.automod import recovery as R

# ── clause 1: the checked-in table and the pure declaration ─────────────────


def _triggers():
    return [t for s in R.SURFACES for t in s.triggers]


def test_the_table_holds_a_service_path_a_store_module_and_a_config_key():
    """The table's minimum coverage, named the way the clause names it."""
    trig = _triggers()
    assert any(t.startswith("agent-services/supervisor/") for t in trig), \
        "no supervisor/systemd-unit path in the surface table"
    assert any(t.startswith("agent-services/systemd/") for t in trig), \
        "no systemd unit path in the surface table"
    # A store-OWNING module: the module that owns the KG store and the module
    # that owns the job queue's `jobs`/`runs` tables.
    assert "app/kg_store.py" in trig, "the KG store's owning module is not a trigger"
    assert "workers/queue.py" in trig, "the job queue's owning module is not a trigger"
    # A config.yaml-consumed key, named rather than left inside the file's bytes.
    assert "config.yaml" in trig
    assert R.CONFIG_KEYS, "no config.yaml key is named in the table"


def test_every_surface_names_a_measurable_digest_and_either_an_inverse_or_none():
    for s in R.SURFACES:
        assert s.name and s.op, s
        assert len(s.digest) > 20, f"{s.name}: digest is not a measurement description"
        assert s.inverse is None or len(s.inverse) > 10, f"{s.name}: vague inverse"
    assert any(s.inverse is None for s in R.SURFACES), \
        ("no surface says 'no inverse' — the honest category the item insists on "
         "is missing, which is how 'unrecoverable' would quietly disappear")


def test_every_surface_with_an_inverse_has_a_live_reader():
    """`build_recovery_record` looks readers up by `op`; a gap there is a typo that
    would read as an unreadable surface forever."""
    readers = R.default_readers(R.Path(__file__).resolve().parent.parent)
    for s in R.SURFACES:
        if s.inverse is None:
            continue
        assert readers.get(s.op) is not None, f"{s.name}: op {s.op} has no reader"


def test_a_pure_code_round_declares_no_surface():
    """Real paths from this repo, so the empty answer is about the KIND of change."""
    pure = ["app/harness/protected_paths.py", "app/gate.py", "scripts/automod/gate.py",
            "tests/test_automod_recovery.py", "skills/foo/SKILL.md",
            "architecture/automod.md"]
    assert R.declare_surfaces(pure) == ()
    assert R.declare_surfaces([]) == ()


def test_a_change_to_the_service_definitions_declares_the_state_it_reaches():
    got = [s.name for s in R.declare_surfaces(
        ["agent-services/supervisor/conf.d/lloyd-backend.conf"])]
    assert got == ["supervisor-program-definitions"]
    got = [s.name for s in R.declare_surfaces(
        ["agent-services/systemd/lloyd-guardian.service"])]
    assert "systemd-user-unit-set" in got, got
    # The guardian unit is ALSO the surface whose ExecStartPre re-stages the
    # guardian's pinned snapshot: one path, two surfaces, both named.
    assert "guardian-pinned-snapshot" in got, got


def test_a_store_owning_module_declares_the_store_and_config_yaml_declares_the_config():
    assert [s.name for s in R.declare_surfaces(["app/kg_store.py"])] == \
        ["knowledge-graph-store"]
    assert [s.name for s in R.declare_surfaces(["workers/queue.py"])] == \
        ["job-queue-store"]
    assert [s.name for s in R.declare_surfaces(["config.yaml"])] == \
        ["effective-runtime-config"]


def test_declaration_matches_exact_paths_so_a_lookalike_cannot_trigger_a_surface():
    assert R.declare_surfaces(["config.yaml.bak"]) == ()
    assert R.declare_surfaces(["app/kg_store.py.bak"]) == ()
    assert R.declare_surfaces(["docs/config.yaml.md"]) == ()


def test_declaration_is_pure_and_touches_no_filesystem(monkeypatch, tmp_path):
    """The clause says PURE. Anything that reads the disk could disagree with the
    promoter about what a round changed, and could be slow."""
    def boom(*a, **k):
        raise AssertionError("declare_surfaces touched the filesystem")
    monkeypatch.setattr(R.Path, "exists", boom)
    monkeypatch.setattr(R.Path, "is_file", boom)
    monkeypatch.setattr(R.Path, "read_bytes", boom)
    monkeypatch.setattr(R.Path, "is_dir", boom)
    monkeypatch.setattr("os.path.exists", boom)
    got = [s.name for s in R.declare_surfaces(
        ["agent-services/supervisor/conf.d/x.conf", "app/harness/thing.py"])]
    assert got == ["supervisor-program-definitions"]


# ── clause 2: exactly three verdicts, and empty is never clean ──────────────


def test_the_verdict_vocabulary_is_exactly_the_three_states():
    assert set(R.VERDICTS) == {"verified", "failed", "unevaluated"}


def test_verified_only_when_every_captured_field_matches():
    w = R.Witness({"sha256": "a" * 64, "files": "3"}, {})
    assert R.verify_restore(w, R.Witness({"sha256": "a" * 64, "files": "3"}, {})) \
        == R.VERIFIED
    assert R.verify_restore(w, R.Witness({"sha256": "b" * 64, "files": "3"}, {})) \
        == R.FAILED
    assert R.restore_mismatch(w, R.Witness({"sha256": "b" * 64, "files": "3"}, {})) \
        == ["sha256"]


def test_an_empty_field_list_is_never_read_as_verified():
    """The clause's own words, in three directions."""
    empty = R.Witness({}, {})
    full = R.Witness({"sha256": "a" * 64}, {})
    assert R.verify_restore(empty, empty) == R.UNEVALUATED
    assert R.verify_restore(full, empty) == R.UNEVALUATED
    assert R.verify_restore(empty, full) == R.UNEVALUATED


def test_a_vanished_field_and_an_appeared_field_are_both_mismatches():
    pre = R.Witness({"a": "1", "b": "2"}, {})
    assert R.verify_restore(pre, R.Witness({"a": "1"}, {})) == R.FAILED
    assert R.restore_mismatch(pre, R.Witness({"a": "1"}, {})) == ["b"]
    assert R.verify_restore(pre, R.Witness({"a": "1", "b": "2", "c": "3"}, {})) \
        == R.FAILED
    assert R.restore_mismatch(pre, R.Witness({"a": "1", "b": "2", "c": "3"}, {})) \
        == ["c"]


def test_capture_witness_reports_a_raising_reader_as_unreadable_not_empty():
    def bad():
        raise PermissionError("db locked")
    w = R.capture_witness(bad)
    assert w.readable is False
    assert w.errors and "PermissionError" in w.errors["reader"]
    assert R.verify_restore(w, R.Witness({"a": "1"}, {})) == R.UNEVALUATED


def test_capture_witness_reports_a_missing_reader_and_a_non_mapping():
    assert R.capture_witness(None).readable is False
    assert R.capture_witness(lambda: "not a mapping").readable is False


def test_capture_witness_reports_a_reader_that_returns_no_fields_as_unreadable():
    """A reader that finds nothing is exactly the vacuity the vet rung refuses."""
    w = R.capture_witness(lambda: {})
    assert w.readable is False
    assert "no fields" in w.errors["reader"]


# ── clause 3: the check can FAIL — the no-op restore is caught ──────────────


@pytest.fixture
def scratch(tmp_path):
    """A file standing in for a non-git surface, with its reader, its applier and
    its real restore op. Everything is injected, so the rung's own code path is the
    thing under test and no live store is anywhere near it."""
    target = tmp_path / "state" / "unit.link"
    target.parent.mkdir(parents=True)
    target.write_text("the state before the round\n")

    def read():
        return {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "bytes": str(target.stat().st_size)}

    def apply_change():
        target.write_text("the candidate wrote this into the state\n")

    def restore():
        target.write_text("the state before the round\n")

    return {"path": target, "read": read, "apply": apply_change, "restore": restore}


def test_a_no_op_restore_op_makes_the_same_check_return_failed(scratch):
    """#3930's falsifier: the restore command 'ran' and nothing came back."""
    rec = R.rehearse_restore(scratch["read"], scratch["apply"], lambda: None,
                             surface="scratch")
    assert rec["verdict"] == R.FAILED
    assert rec["mismatch"] == ["bytes", "sha256"]
    assert rec["witness_captured"] is True
    assert rec["unrecoverable"] is False


def test_the_real_restore_op_round_trips_to_verified_with_the_digest_restored(scratch):
    before = scratch["read"]()["sha256"]
    rec = R.rehearse_restore(scratch["read"], scratch["apply"], scratch["restore"],
                             surface="scratch")
    assert rec["verdict"] == R.VERIFIED, rec
    assert rec["mismatch"] == []
    assert scratch["read"]()["sha256"] == before, "the digest was not restored exactly"


def test_a_restore_op_that_raises_is_failed_not_unevaluated(scratch):
    def explode():
        raise RuntimeError("restore route is gone")
    rec = R.rehearse_restore(scratch["read"], scratch["apply"], explode,
                             surface="scratch")
    assert rec["verdict"] == R.FAILED
    assert "restore raised" in rec["reason"]


def test_no_restore_op_is_unevaluated_and_names_the_surface_unrecoverable(scratch):
    rec = R.rehearse_restore(scratch["read"], scratch["apply"], None,
                             surface="guardian-pinned-snapshot")
    assert rec["verdict"] == R.UNEVALUATED
    assert rec["unrecoverable"] is True
    assert rec["reason"].startswith("unrecoverable:")


def test_an_apply_that_moves_no_field_is_unevaluated_not_verified(scratch):
    """The witness did not change, so 'the state came back' would be vacuous."""
    rec = R.rehearse_restore(scratch["read"], lambda: None, scratch["restore"],
                             surface="scratch")
    assert rec["verdict"] == R.UNEVALUATED
    assert "moved no witness field" in rec["reason"]


def test_an_unreadable_surface_is_never_applied_to(scratch):
    """Rehearsing against a store we cannot measure could damage it unmeasured."""
    applied = []

    def read():
        raise PermissionError("locked")

    def apply_change():
        applied.append(1)

    rec = R.rehearse_restore(read, apply_change, scratch["restore"], surface="scratch")
    assert rec["verdict"] == R.UNEVALUATED
    assert rec["witness_captured"] is False
    assert applied == [], "the change was applied to a surface with no witness"


# ── the per-round record the rung builds ───────────────────────────────────


def _counting_readers(names=("supervisor_conf", "kg_rows", "runtime_config")):
    calls: list[str] = []

    def make(name):
        def read():
            calls.append(name)
            return {"digest": "d" * 8, "n": "1"}
        return read

    return ({n: make(n) for n in names}, calls)


def test_a_pure_code_round_carries_an_empty_contract_and_reads_no_witness():
    readers, calls = _counting_readers()
    rec = R.build_recovery_record(["app/gate.py", "tests/test_x.py"], readers=readers)
    assert rec["contract"] == R.CONTRACT_EMPTY
    assert rec["declared"] == []
    assert rec["verdict"] == R.UNEVALUATED
    assert rec["observe_only"] is True
    assert rec["rehearsed"] is False
    assert calls == [], "an ordinary code round should pay no witness read at all"


def test_a_declared_surface_gets_its_witness_recorded_per_surface():
    readers, calls = _counting_readers(names=("supervisor_conf",))
    rec = R.build_recovery_record(["agent-services/supervisor/conf.d/x.conf"],
                                  readers=readers)
    assert rec["contract"] == R.CONTRACT_DECLARED
    assert rec["declared"] == ["supervisor-program-definitions"]
    # This used to read "no rehearsal exists yet". #4082 built the rehearsal, so the
    # sentence here is about THIS call: it supplied no scratch copy, and a surface
    # nobody applied-and-unwound in a copy may not be reported as verified either.
    assert rec["verdict"] == R.UNEVALUATED, \
        "no scratch copy was supplied to this call, so nothing may be reported " \
        "as verified: #4082 rehearses only inside a directory the caller provides"
    assert rec["surfaces"][0]["rehearsed"] is False
    assert "no scratch copy" in rec["surfaces"][0]["reason"]
    assert rec["surfaces"][0]["witness_captured"] is True
    assert calls == ["supervisor_conf"]


def test_a_surface_with_no_inverse_is_recorded_unrecoverable_not_admitted():
    """`build_recovery_record` never reads a witness for such a surface — there is
    nothing to put back — and says which surface out loud."""
    rec = R.build_recovery_record(["agent-services/guardian/policy.py"], readers={})
    assert rec["declared"] == ["guardian-pinned-snapshot"]
    assert rec["unrecoverable"] == ["guardian-pinned-snapshot"]
    assert rec["surfaces"][0]["has_inverse"] is False
    assert rec["surfaces"][0]["reason"].startswith("unrecoverable: ")


def test_an_unreadable_surface_is_recorded_unevaluated_with_a_reason():
    def bad():
        raise FileNotFoundError("kg.sqlite absent")
    rec = R.build_recovery_record(["app/kg_store.py"],
                                  readers={"kg_rows": bad})
    surface = rec["surfaces"][0]
    assert surface["witness_captured"] is False
    assert surface["verdict"] == R.UNEVALUATED
    assert "FileNotFoundError" in surface["reason"]
    assert "kg.sqlite absent" in surface["reason"]


def test_the_record_summaries_name_the_three_buckets():
    empty = R.build_recovery_record(["app/gate.py"], readers={})
    assert "no non-git effect declared" in R.record_summary(empty)
    declared = R.build_recovery_record(["agent-services/supervisor/conf.d/x.conf"],
                                       readers=_counting_readers(("supervisor_conf",))[0])
    summary = R.record_summary(declared)
    assert summary.startswith("observe-only:")
    assert "supervisor-program-definitions=" in summary
    unrec = R.build_recovery_record(["agent-services/guardian/policy.py"], readers={})
    assert "unrecoverable: guardian-pinned-snapshot" in R.record_summary(unrec)


def test_the_event_form_carries_the_declaration_and_witness_flags():
    rec = R.build_recovery_record(["app/kg_store.py"],
                                  readers={"kg_rows": lambda: {"kg_rows": "41820"}})
    ev = R.to_event(rec)
    assert ev["observe_only"] is True
    assert ev["contract"] == R.CONTRACT_DECLARED
    assert ev["verdict"] == R.UNEVALUATED
    assert ev["rehearsed"] is False
    assert ev["declared"] == ["knowledge-graph-store"]
    assert ev["witness_captured"] == {"knowledge-graph-store": True}
    assert "unevaluated_reasons" not in ev


def test_the_event_form_keeps_the_reason_when_the_could_not_is_the_point():
    def bad():
        raise FileNotFoundError("kg.sqlite absent")
    rec = R.build_recovery_record(["app/kg_store.py"], readers={"kg_rows": bad})
    ev = R.to_event(rec)
    assert ev["witness_captured"] == {"knowledge-graph-store": False}
    assert "kg.sqlite absent" in ev["unevaluated_reasons"]["knowledge-graph-store"]
    assert len(ev["unevaluated_reasons"]["knowledge-graph-store"]) <= R.REASON_CAP


# ── #4082 clauses 4-6: the file-scope rehearsal in a disposable copy ─────────
#
# Every fixture below is a REAL tree of bytes, and that is the point. A reader that
# returns a constant dict whatever the root gives pre == post no matter what the
# scratch copy holds, and `rehearse_restore` then answers `unevaluated` ("the applied
# change moved no witness field") — so a no-op-inverse falsifier built on one would
# pass without proving the rehearsal can fail. These fixtures put different bytes at
# the round's changed paths, so a real apply moves the digest and the only thing that
# can hold it back is an inverse that does not put them back.

CONF = "agent-services/supervisor/conf.d/lloyd-backend.conf"
UNIT = "agent-services/systemd/lloyd-backend.service"
KEEP = "agent-services/systemd/lloyd-guardian.service"
NOUNIT = "agent-services/systemd/README.md"   # under the surface root, not a unit
OTHER = "lloyd-guardian.service"        # the link already in the manager's dir
OLD, NEW = "command=/old/start\n", "command=/new/start\n"
CONFIG_OLD = "automod:\n  enabled: false\n"
CONFIG_NEW = "automod:\n  enabled: true\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _scratch_roots(tmp_path: Path) -> dict:
    """live/ = the machine before the round, cand/ = the round's own tree,
    units/ = a stand-in for ~/.config/systemd/user, scratch/ = the copy the
    rehearsal is confined to (and the only thing it may write).

    The candidate differs from live at exactly the three file-scope surfaces' paths,
    so the applied change demonstrably moves every witness it should. The unit
    directory starts with only the guardian's link: the round's unit is not linked
    yet, which is the state a rehearsal has to reproduce and then take back.
    """
    live, cand = tmp_path / "live", tmp_path / "cand"
    for root, (a, b, c, d) in ((live, (OLD, OLD, CONFIG_OLD, "one line\n")),
                               (cand, (NEW, NEW, CONFIG_NEW, "another line\n"))):
        _write(root / CONF, "[program:lloyd-backend]\n" + a)
        _write(root / UNIT, "[Service]\nExecStart=" + b)
        _write(root / KEEP, "[Service]\nExecStart=/guardian/keep\n")
        _write(root / NOUNIT, "# units\n" + d)
        _write(root / "config.yaml", c)
    units = tmp_path / "units"
    units.mkdir(parents=True)
    os.symlink(str(live / KEEP), str(units / OTHER))
    return {"root": tmp_path, "live": live, "cand": cand, "units": units,
            "scratch": tmp_path / "scratch"}


def _snapshot(root: Path, skip: Path | None = None) -> dict:
    """relpath -> what the entry IS, for every path under ROOT except under SKIP.
    Bytes by hash, links by raw target, directories and files by mtime and size:
    the witness that no path outside the scratch copy was created or modified."""
    out: dict[str, tuple] = {}
    for p in sorted(root.rglob("*")):
        if skip is not None and (p == skip or skip in p.parents):
            continue
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = ("link", os.readlink(str(p)))
        elif p.is_dir():
            out[rel] = ("dir", p.stat().st_mtime_ns)
        else:
            out[rel] = ("file", p.stat().st_mtime_ns, p.stat().st_size,
                        hashlib.sha256(p.read_bytes()).hexdigest())
    return out


class _NeverRunner:
    """A `runner` that fails the call if the rehearsal execs anything.

    The clause is "no supervisorctl, no systemctl --user daemon-reload, no service
    restart", and the only way to pin a callable's ABSENCE is to hand the builder one
    that raises and record whether it was reached.
    """

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, *argv, **kw):
        self.calls.append((argv, kw))
        raise AssertionError(f"the scratch rehearsal ran something: {argv!r}")


def _noop_inverses() -> dict:
    """`scratch_inverse` stubbed to do nothing, for every surface that has one: the
    same apply, the same reader, the same machinery, and no undo."""
    return {op: (lambda ctx: (lambda: None)) for op in R.SCRATCH_INVERSES}


@pytest.mark.parametrize("changed,surface,mismatched", [
    (CONF, "supervisor-program-definitions", "tree"),
    (UNIT, "systemd-user-unit-set", "unit_links"),
    ("config.yaml", "effective-runtime-config", "config_sha"),
])
def test_a_file_scope_surface_is_rehearsed_in_the_copy_and_comes_back_verified(
        tmp_path, changed, surface, mismatched):
    """Clause 1: the round's bytes are applied to a caller-supplied copy, the
    surface's `scratch_inverse` undoes them there, and the re-digest equals pre.

    `rehearse_restore` is what compares the digests, so `verified` here is the
    statement that the copy's witness moved on the apply AND came back byte-equal
    after the undo — not that an inverse was invoked.
    """
    t = _scratch_roots(tmp_path)
    boom = _NeverRunner()
    before = _snapshot(t["root"], skip=t["scratch"])

    rec = R.build_recovery_record([changed], live_root=t["live"],
                                 candidate_root=t["cand"],
                                 scratch_root=t["scratch"],
                                 unit_dir=t["units"], runner=boom)

    s = rec["surfaces"][0]
    assert s["name"] == surface
    assert s["verdict"] == R.VERIFIED, s["reason"]
    assert s["rehearsed"] is True
    assert rec["verdict"] == R.VERIFIED
    assert rec["rehearsed"] is True
    assert s["reason"] == "all captured fields came back byte-equal"
    # The rehearsal execs nothing: not `supervisorctl reread`, not `systemctl --user
    # daemon-reload`, not a restart. `boom` would have raised and recorded the call.
    assert boom.calls == []
    # Confined to the copy: live/, cand/, units/ and every other path the test owns
    # are byte-for-byte and mtime-for-mtime what they were.
    assert _snapshot(t["root"], skip=t["scratch"]) == before
    # And the copy really is where it happened: it exists, and after the undo it
    # holds the BASE bytes again, not the candidate's.
    assert (t["scratch"] / "work" / changed).is_file()
    assert mismatched not in s["mismatch"]


def test_the_rehearsal_links_the_unit_into_its_own_copy_not_the_live_unit_dir(tmp_path):
    """Clause 1 again, on the surface whose undo is a re-LINK (#3930's risk).

    The unit witness is the user manager's link set, so a rehearsal that only
    restored files would leave a fresh symlink in ~/.config/systemd/user — the gate
    creating a service definition that outlives the round. Here the manager's
    directory is the fixture's, and it must still hold exactly the guardian's link
    afterwards while the scratch copy is what gained and lost one.
    """
    t = _scratch_roots(tmp_path)
    boom = _NeverRunner()
    rec = R.build_recovery_record([UNIT], live_root=t["live"],
                                 candidate_root=t["cand"],
                                 scratch_root=t["scratch"],
                                 unit_dir=t["units"], runner=boom)

    s = rec["surfaces"][0]
    assert s["name"] == "systemd-user-unit-set"
    assert s["verdict"] == R.VERIFIED, s["reason"]
    assert s["rehearsed"] is True
    assert boom.calls == []
    assert sorted(os.listdir(t["units"])) == [OTHER], \
        "the live unit directory must be exactly as it was: the rehearsal has no " \
        "route to the manager's directory, only to a copy of it"
    scratch_units = t["scratch"] / "units"
    assert sorted(os.listdir(scratch_units)) == [OTHER], \
        "the copy is what the apply linked into and the inverse un-linked"
    assert not (scratch_units / "lloyd-backend.service").exists()


@pytest.mark.parametrize("changed,surface,mismatched", [
    (CONF, "supervisor-program-definitions", "tree"),
    (UNIT, "systemd-user-unit-set", "unit_links"),
    ("config.yaml", "effective-runtime-config", "config_sha"),
])
def test_a_no_op_scratch_inverse_makes_the_same_rehearsal_come_back_failed(
        tmp_path, changed, surface, mismatched):
    """Clause 2: same fixture, same apply, no-op undo — `failed`, not `unevaluated`,
    and it names the witness field that did not come back.

    The word `failed` is the whole falsifiability claim. A reader that ignored the
    scratch root would report `unevaluated` ("the applied change moved no witness
    field") and this test would pass having proved nothing, so the fixture's bytes
    differ between live/ and cand/ and the apply demonstrably moves the digest.
    """
    t = _scratch_roots(tmp_path)
    boom = _NeverRunner()
    before = _snapshot(t["root"], skip=t["scratch"])
    rec = R.build_recovery_record([changed], live_root=t["live"],
                                 candidate_root=t["cand"],
                                 scratch_root=t["scratch"],
                                 unit_dir=t["units"], runner=boom,
                                 scratch_inverses=_noop_inverses())

    s = rec["surfaces"][0]
    assert s["name"] == surface
    assert s["verdict"] == R.FAILED, \
        f"a no-op undo must be a failure, not a shrug: {s['reason']}"
    assert s["rehearsed"] is True
    assert rec["verdict"] == R.FAILED
    assert mismatched in s["mismatch"], s["mismatch"]
    # The field has to be in the REASON too: `mismatch` is a gate.json key, and
    # gate.json dies with the worktree.
    assert mismatched in s["reason"], s["reason"]
    assert boom.calls == []
    # The apply really did happen inside the copy, the undo did not undo it, and
    # nothing at all escaped the copy.
    assert (t["scratch"] / "work" / changed).read_text() == \
        (t["cand"] / changed).read_text(), \
        "the copy must still hold the candidate's bytes: a no-op undid nothing"
    assert _snapshot(t["root"], skip=t["scratch"]) == before


def test_the_scratch_copy_holds_the_candidate_bytes_and_the_undo_returns_the_base(tmp_path):
    """Where the rehearsal physically worked, read out of the copy afterwards.

    A `verified` verdict is consistent with the rehearsal having quietly done
    nothing, so this reads the copy: the undo left the BASE bytes at the changed
    path, and the live tree still holds the base bytes it started with.
    """
    t = _scratch_roots(tmp_path)
    rec = R.build_recovery_record([CONF], live_root=t["live"],
                                 candidate_root=t["cand"],
                                 scratch_root=t["scratch"])
    assert rec["verdict"] == R.VERIFIED
    work_conf = t["scratch"] / "work" / CONF
    assert work_conf.read_text() == (t["live"] / CONF).read_text()
    assert (t["live"] / CONF).read_text().find(OLD) >= 0
    assert (t["cand"] / CONF).read_text().find(NEW) >= 0


def test_a_change_the_witness_cannot_see_is_not_credited_a_rehearsal(tmp_path):
    """The vacuity guard, on the round that really meets it: a file under the
    systemd surface's root that is not a unit. The witness is the link set plus the
    count of `*.service`, so editing `README.md` there moves nothing — and
    `rehearse_restore` refuses to call an unchanged digest a round-trip. `unevaluated`
    with the guard's own reason, and `rehearsed` False, which is why a surface whose
    undo ran is credited and one that never got to run is not."""
    t = _scratch_roots(tmp_path)
    rec = R.build_recovery_record([NOUNIT], live_root=t["live"],
                                 candidate_root=t["cand"],
                                 scratch_root=t["scratch"], unit_dir=t["units"])
    s = rec["surfaces"][0]
    assert s["name"] == "systemd-user-unit-set"
    assert s["verdict"] == R.UNEVALUATED
    assert s["rehearsed"] is False
    assert "moved no witness field" in s["reason"], s["reason"]
    # The apply DID happen — the copy holds the candidate's README — and the
    # rehearsal still declined to report a restore, because nothing was measured.
    assert (t["scratch"] / "work" / NOUNIT).read_text() == \
        (t["cand"] / NOUNIT).read_text()


def test_a_store_surface_says_the_available_restore_is_coarser_than_one_round(tmp_path):
    """Clause 3: `app/kg_store.py` and `workers/queue.py` keep `unevaluated`, but the
    reason now names what they actually have — a day- and an hour-granular snapshot —
    instead of the step-3-deferred placeholder. Supplied a scratch copy and still
    unevaluated: a rehearsal is not on offer for a store's rows."""
    t = _scratch_roots(tmp_path)
    for changed, op, gran in (("app/kg_store.py", "kg_rows", "day-granular"),
                              ("workers/queue.py", "queue_rows", "hour-granular")):
        readers, _ = _counting_readers(names=(op,))
        rec = R.build_recovery_record([changed], readers=readers,
                                      live_root=t["live"],
                                      candidate_root=t["cand"],
                                      scratch_root=t["scratch"],
                                      unit_dir=t["units"])
        s = rec["surfaces"][0]
        assert s["verdict"] == R.UNEVALUATED
        assert s["rehearsed"] is False
        assert gran in s["reason"], s["reason"]
        assert "coarser" in s["reason"], s["reason"]
        assert "step 3 deferred" not in s["reason"]


def test_the_guardian_surface_keeps_its_unrecoverable_verdict_with_a_copy_present(tmp_path):
    """Clause 3 again: `agent-services/guardian/**` still answers
    `unrecoverable: guardian-pinned-snapshot`. Its only undo is a unit restart, so
    the rehearsal the other three surfaces get is not offered to it, and a scratch
    copy on the table changes nothing about that."""
    t = _scratch_roots(tmp_path)
    rec = R.build_recovery_record(["agent-services/guardian/guardian.sh"],
                                 live_root=t["live"], candidate_root=t["cand"],
                                 scratch_root=t["scratch"], unit_dir=t["units"])
    s = rec["surfaces"][0]
    assert s["name"] == "guardian-pinned-snapshot"
    assert s["has_inverse"] is False
    assert s["verdict"] == R.UNEVALUATED
    assert s["rehearsed"] is False
    assert s["reason"].startswith("unrecoverable: guardian-pinned-snapshot")
    assert rec["unrecoverable"] == ["guardian-pinned-snapshot"]


def test_the_rehearsal_never_writes_outside_the_directory_the_caller_supplied(tmp_path):
    """Clause 1's confinement stated as the strongest form: one call, three
    surfaces, every witness read for real, and the ONLY path that differs
    afterwards is the scratch directory the test handed it."""
    t = _scratch_roots(tmp_path)
    boom = _NeverRunner()
    before = _snapshot(t["root"], skip=t["scratch"])

    rec = R.build_recovery_record([CONF, UNIT, "config.yaml"],
                                 live_root=t["live"], candidate_root=t["cand"],
                                 scratch_root=t["scratch"],
                                 unit_dir=t["units"], runner=boom)

    assert rec["verdict"] == R.VERIFIED, [s["reason"] for s in rec["surfaces"]]
    assert rec["rehearsed"] is True
    assert [s["verdict"] for s in rec["surfaces"]] == [R.VERIFIED] * 3
    assert boom.calls == []
    assert _snapshot(t["root"], skip=t["scratch"]) == before
    assert sorted(os.listdir(t["units"])) == [OTHER]


def test_the_rehearsal_verdict_reaches_the_event_so_the_ledger_keeps_it():
    """The per-surface verdict is the only form in which a rehearsal outlives the
    round: `gate.json` is copied into the round directory and deleted with the
    worktree, so `to_event` has to carry it per surface, not just one word for the
    whole record."""
    ev = R.to_event({"observe_only": True, "contract": R.CONTRACT_DECLARED,
                     "verdict": R.FAILED, "rehearsed": True,
                     "declared": ["supervisor-program-definitions",
                                  "knowledge-graph-store"],
                     "unrecoverable": [],
                     "surfaces": [
                         {"name": "supervisor-program-definitions",
                          "has_inverse": True, "witness_captured": True,
                          "verdict": R.FAILED, "rehearsed": True,
                          "reason": "differed after restore: tree",
                          "mismatch": ["tree"]},
                         {"name": "knowledge-graph-store", "has_inverse": True,
                          "witness_captured": True, "verdict": R.UNEVALUATED,
                          "rehearsed": False, "reason": "coarser", "mismatch": []}]})
    assert ev["surface_verdicts"] == {"supervisor-program-definitions": R.FAILED,
                                      "knowledge-graph-store": R.UNEVALUATED}
    assert ev["rehearsed"] is True
    assert ev["mismatch"] == {"supervisor-program-definitions": ["tree"]}
    assert ev["verdict"] == R.FAILED
