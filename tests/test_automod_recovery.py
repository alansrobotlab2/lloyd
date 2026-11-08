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
"""
from __future__ import annotations

import hashlib

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
    assert rec["verdict"] == R.UNEVALUATED, \
        "no rehearsal exists yet, so nothing may be reported as verified"
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
