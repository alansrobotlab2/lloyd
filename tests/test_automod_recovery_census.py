"""`python -m scripts.automod.recovery_census --last 50` (#3930 clause 5).

The clause: one command regenerates the surface → measurable digest → available
inverse table over landed-round records, and prints an explicit `none` row for a
surface with no inverse — never an empty table.

An empty table is the failure mode this file spends most of its effort on. A
census that prints nothing when it has seen nothing is indistinguishable, in six
weeks, from a census that found no surfaces to recover — which is how a report
that was supposed to catch `unrecoverable` becoming a rubber stamp would itself
go quiet. So the table comes from the checked-in contract, not from the ledger,
and the ledger only fills in how many rounds DECLARED each one.

#4082 adds a fourth clause at the bottom of this file: a measured restore is counted
PER SURFACE, from `surface_verdicts` on the durable `rung=recovery` ledger row alone
— no `gate.json`, no round directory — so a round that rehearses one surface and
cannot rehearse a store still contributes its one verified round-trip, and rows
written before that field existed are reported as no coverage rather than as zero
restores.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.automod import recovery as R
from scripts.automod import recovery_census as RC

ROOT = Path(__file__).resolve().parent.parent


def _rec(*, contract: str = R.CONTRACT_EMPTY, declared=(), unrec=(),
         verdict: str = R.UNEVALUATED, captured=None, surfaces=None) -> dict:
    rec = {"observe_only": True, "contract": contract,
           "declared": list(declared), "unrecoverable": list(unrec),
           "verdict": verdict, "rehearsed": bool(surfaces),
           "witness_captured": dict(captured or {})}
    if surfaces is not None:
        # The field #4082 added: `R.to_event` writes it, the census reads it. An
        # event from before that change has no such key, and the census must count
        # those rows as no coverage rather than as zero verified restores.
        rec["surface_verdicts"] = dict(surfaces)
    return rec


def _gate_ev(round_id: str, ts: float, *, recovery: dict | None) -> dict:
    ev = {"event": "gate", "round_id": round_id, "rung": "preflight",
          "ok": True, "ts": ts}
    if recovery is not None:
        ev = {"event": "gate", "round_id": round_id, "rung": "recovery",
              "ok": True, "ts": ts, "recovery": recovery}
    return ev


def _write_ledger(path: Path, events: list[dict]) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


# ── the table is always there, and `none` is always said out loud ───────────


def test_the_census_prints_the_whole_table_when_the_window_has_never_seen_a_round(
        tmp_path):
    """Never an empty table: the surfaces come from the contract, not the ledger."""
    out = RC.render(RC.census([], last=50))
    for s in R.SURFACES:
        assert s.name in out, f"{s.name} missing from the census table"
        assert s.digest in out, f"{s.name}'s digest is not printed"
    assert "rounds in the window: 0, rounds with a recovery record: 0 of 0" in out
    assert "none" in out, "a surface with no inverse must print `none`"
    assert len(out.splitlines()) > len(R.SURFACES)


def test_the_census_prints_none_as_the_inverse_rather_than_blanking_the_cell():
    rows = {row["surface"]: row for row in RC.census([], last=50)["surfaces"]}
    name = next(s.name for s in R.SURFACES if s.inverse is None)
    assert rows[name]["inverse"] == "none", \
        "an empty cell reads as 'not recorded', not as 'no inverse exists'"
    for s in R.SURFACES:
        if s.inverse:
            assert rows[s.name]["inverse"] == s.inverse


def test_the_census_reports_the_named_no_inverse_surface():
    c = RC.census([], last=50)
    assert "guardian-pinned-snapshot" in c["no_inverse"]
    assert "surfaces with no inverse: guardian-pinned-snapshot" in RC.render(c)


# ── the ledger fills in what real rounds actually declared ──────────────────


def test_the_census_counts_the_rounds_that_declared_each_surface(tmp_path):
    events = [
        _gate_ev("SM_A", 1.0, recovery=_rec(
            contract=R.CONTRACT_DECLARED, declared=["supervisor-program-definitions"],
            captured={"supervisor-program-definitions": True})),
        _gate_ev("SM_B", 2.0, recovery=_rec(
            contract=R.CONTRACT_DECLARED, declared=["supervisor-program-definitions"],
            captured={"supervisor-program-definitions": False})),
        _gate_ev("SM_C", 3.0, recovery=_rec(
            contract=R.CONTRACT_DECLARED,
            declared=["guardian-pinned-snapshot"],
            unrec=["guardian-pinned-snapshot"])),
        _gate_ev("SM_D", 4.0, recovery=_rec()),
    ]
    c = RC.census(events, last=50)
    rows = {row["surface"]: row for row in c["surfaces"]}
    assert rows["supervisor-program-definitions"]["rounds"] == 2
    assert rows["guardian-pinned-snapshot"]["unrecoverable_rounds"] == 1
    assert rows["knowledge-graph-store"]["rounds"] == 0
    out = RC.render(c)
    assert "rounds with an empty effect contract: 1" in out
    assert "rounds declaring at least one surface: 3" in out


def test_the_published_three_way_split_is_over_rounds_not_rungs(tmp_path):
    """A round gated twice (a re-gate) counts once, at its latest record — the
    split is over ROUNDS, and counting rungs would inflate the denominator."""
    events = [
        _gate_ev("SM_A", 1.0, recovery=_rec(contract=R.CONTRACT_DECLARED,
                                           declared=["knowledge-graph-store"])),
        _gate_ev("SM_A", 9.0, recovery=_rec()),                      # re-gated, now empty
        _gate_ev("SM_A", 5.0, recovery=_rec(contract=R.CONTRACT_DECLARED,
                                           declared=["job-queue-store"])),
        _gate_ev("SM_B", 2.0, recovery=_rec(contract=R.CONTRACT_DECLARED,
                                           declared=["job-queue-store"])),
    ]
    c = RC.census(events, last=50)
    assert c["rounds_in_window"] == 2
    assert c["rounds_with_record"] == 2
    assert c["contracts"] == {"empty": 1, "declared": 1}
    rows = {row["surface"]: row for row in c["surfaces"]}
    assert rows["knowledge-graph-store"]["rounds"] == 0, \
        "the superseded record of the re-gated round must not be counted"
    assert rows["job-queue-store"]["rounds"] == 1


def test_rounds_gated_before_the_rung_existed_are_reported_as_a_gap_not_an_empty_contract(
        tmp_path):
    """Sixty gate events with no recovery row are 60 rounds this rung never saw.
    Folding them into `empty contract` would be the census lying about the split
    the acceptance publishes."""
    events = [_gate_ev(f"SM_{i}", float(i), recovery=None) for i in range(5)]
    c = RC.census(events, last=50)
    assert c["rounds_in_window"] == 5
    assert c["rounds_with_record"] == 0
    assert c["contracts"] == {"empty": 0, "declared": 0}
    assert "with a recovery record: 0" in RC.render(c)


def test_the_window_is_the_last_n_rounds_not_the_last_n_events():
    events = [_gate_ev(f"SM_{i}", float(i), recovery=None) for i in range(60)]
    assert len(RC.window_rounds(events, 50)) == 50
    assert RC.window_rounds(events, 50)[0] == "SM_10", "oldest kept round is wrong"
    assert RC.window_rounds(events, 5) == ["SM_55", "SM_56", "SM_57", "SM_58", "SM_59"]


# ── the one-command clause ──────────────────────────────────────────────────


def test_the_command_line_entry_point_prints_the_table_from_a_ledger_file(tmp_path,
                                                                         capsys):
    ledger = _write_ledger(tmp_path / "promotions.jsonl", [
        _gate_ev("SM_X", 10.0, recovery=_rec(
            contract=R.CONTRACT_DECLARED, declared=["effective-runtime-config"],
            captured={"effective-runtime-config": True})),
    ])
    assert RC.main(["--last", "50", "--ledger", str(ledger)]) == 0
    out = capsys.readouterr().out
    assert "effective-runtime-config" in out
    assert "rounds with a recovery record: 1" in out
    for s in R.SURFACES:
        assert s.name in out


def test_the_module_runs_under_dash_m_with_the_named_flag(tmp_path):
    """`python -m scripts.automod.recovery_census --last 50` is the command the
    acceptance names, so the entry point and the flag are what gets proven here —
    against a scratch ledger, because a test may not read the live one's shape."""
    ledger = _write_ledger(tmp_path / "promotions.jsonl",
                           [_gate_ev("SM_Y", 3.0, recovery=_rec())])
    r = subprocess.run([sys.executable, "-m", "scripts.automod.recovery_census",
                        "--last", "50", "--ledger", str(ledger)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "rounds with a recovery record: 1" in r.stdout
    assert "-> none" in r.stdout, "the no-inverse row must be printed by the command"
    assert "--last" not in r.stdout


def test_the_surface_table_command_lists_every_declared_inverse(tmp_path, capsys):
    """The regeneration half of the clause: the table itself, on demand, without a
    ledger at all."""
    assert R.main(["--surfaces"]) == 0
    out = capsys.readouterr().out
    for s in R.SURFACES:
        assert f"{s.name}\n" in out
        assert s.digest in out
    assert "inverse: none" in out


# ── #4082: the measured restore, counted per surface over ledger events alone ─
#
# The record-level `verdict` is ONE word per round, and a round that rehearses one
# surface and cannot rehearse a store still aggregates to `unevaluated`. So the
# verified-restore count has to come from `surface_verdicts`, which is what
# `R.to_event` writes and this census reads — and the ledger is the only artifact
# that survives the round, `gate.json` being copied into the round directory and
# deleted with the worktree.


def _rehearsed_round(tmp_path: Path, round_id: str = "SM_R") -> dict:
    """ONE real round, recorded the way the gate records it: the real builder over a
    scratch live tree and a scratch candidate tree, then `R.to_event`.

    It touches a supervisor conf (a file-scope surface with a scratch inverse) and
    `app/kg_store.py` (a store whose restore is coarser than one round), so the
    ROUND aggregates to `unevaluated` while one surface carries a measured
    round-trip — the split the round-level count cannot express.

    The two LIVE witness readers are injected, because no test may digest the live
    checkout or open the live KG store. The rehearsal does not use them: it builds
    its own readers over the scratch copy, which is where the measurement this
    census counts actually comes from.
    """
    live, cand = tmp_path / "live", tmp_path / "cand"
    rel = "agent-services/supervisor/conf.d/lloyd-backend.conf"
    for root, body in ((live, "command=/old\n"), (cand, "command=/new\n")):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    rec = R.build_recovery_record(
        [rel, "app/kg_store.py"],
        readers={"supervisor_conf": lambda: {"tree": "d" * 16, "files": "1"},
                 "kg_rows": lambda: {"kg_rows": "41820"}},
        live_root=live, candidate_root=cand,
        scratch_root=tmp_path / "scratch")
    assert rec["verdict"] == R.UNEVALUATED          # the aggregate, one word
    assert rec["surfaces"][0]["verdict"] == R.VERIFIED, rec["surfaces"][0]["reason"]
    return R.to_event(rec)


def test_a_measured_restore_is_counted_per_surface_even_when_the_round_is_not(tmp_path):
    """Clause 5's whole point in one round: the verifier measured a round-trip on
    the supervisor surface, the store surface stayed `unevaluated`, and the round's
    own verdict is therefore `unevaluated` too. Counting rounds would report
    `verified=0` and hide the restore that was actually measured."""
    ev = [_gate_ev("SM_R", 5.0, recovery=_rehearsed_round(tmp_path))]
    c = RC.census(ev, last=50)
    assert c["verdicts"]["verified"] == 0, "the round-level bucket is unchanged"
    assert c["restores"]["verified"] == 1, c["restores"]
    assert c["restores"]["unevaluated"] == 1
    assert c["restores"]["events_with_surface_verdicts"] == 1
    rows = {row["surface"]: row for row in c["surfaces"]}
    assert rows["supervisor-program-definitions"]["verified"] == 1
    assert rows["knowledge-graph-store"]["verified"] == 0
    assert rows["knowledge-graph-store"]["unevaluated"] == 1


def test_the_verified_restore_count_is_printed_and_named_as_per_surface():
    """The printed census has to say WHICH denominator is zero, or a `verified=0`
    round column will be read as 'no round-trip was ever measured'."""
    out = RC.render(RC.census(
        [_gate_ev("SM_R", 5.0, recovery=_rec(
            contract=R.CONTRACT_DECLARED, declared=["effective-runtime-config"],
            captured={"effective-runtime-config": True},
            surfaces={"effective-runtime-config": R.VERIFIED}))], last=50))
    assert "measured restores (per surface, #4082): verified=1" in out
    assert "verifier verdicts: verified=0" in out, \
        "both denominators print, so neither is mistaken for the other"


def test_an_event_from_before_the_per_surface_field_is_no_coverage_not_zero_restores(
        tmp_path):
    """Rows written before #4082 carry no per-surface verdict. They must not be
    folded into the verified-restore count as zeros, and the census says how many of
    its rows could be counted at all — a coverage statement, not a verdict."""
    c = RC.census([_gate_ev("SM_OLD", 1.0, recovery=_rec(
        contract=R.CONTRACT_DECLARED, declared=["supervisor-program-definitions"],
        captured={"supervisor-program-definitions": True}))], last=50)
    assert c["restores"] == {"verified": 0, "failed": 0, "unevaluated": 0,
                            "events_with_surface_verdicts": 0}
    out = RC.render(c)
    assert "measured restores (per surface, #4082): verified=0" in out
    assert "0 of 1 round(s) whose ledger row carries per-surface verdicts" in out


def test_the_command_counts_a_measured_restore_from_the_ledger_file_alone(tmp_path,
                                                                        capsys):
    """`--ledger` and nothing else: no `gate.json`, no round directory, no worktree.
    The owed check that asks whether any surface ever measured a round-trip runs this
    command months from now, against rows a deleted round left behind."""
    ledger = _write_ledger(tmp_path / "promotions.jsonl", [
        _gate_ev("SM_R", 7.0, recovery=_rehearsed_round(tmp_path))])
    assert RC.main(["--last", "50", "--ledger", str(ledger)]) == 0
    out = capsys.readouterr().out
    assert "measured restores (per surface, #4082): verified=1" in out
    assert "failed=0" in out
