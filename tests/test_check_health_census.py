"""The per-check health census (#3418): `scripts/maintenance/check_health_census.py`.

The gate and the eval tree together are ~90 checks, and until this module nothing in the tree
could answer, for any one of them, how often it ran, how often it actually failed, when it last
failed and what it cost. The measurements this item was opened on — `vet` 0 fails of 2,466,
`pyright` 0 of 1,160, against `review` 871 of 2,909 — were computed by hand over the ledger in
triage, which is the same evidence #2378 left in a `--record` flag with no caller.

What is pinned here, clause by clause:

1. `test_the_rung_rows_are_enumerated_from_the_ledgers_own_rung_values` — one row per distinct
   `rung` value on the `event:"gate"` rows the reader actually read, each carrying
   passes/fails over n, `days_since_last_fail`, mean and median seconds, and the window's first
   and last stamps. Invented rung names prove enumeration is data-driven, not a constant.
2. `test_a_rung_present_only_in_an_archive_is_counted_from_the_archive` — the population is the
   live `promotions.jsonl` UNION every `promotions-archive-YYYYMM.jsonl.gz` beside it, two
   archives deep.
3. `test_the_degenerate_prints_never_execute_never_fail_or_a_zero_second_pass` — `0/0 (never
   executed)`, `below floor (n=…)`, and the zero-second share on every row.
4. `test_every_eval_file_prints_one_class_beside_the_command_that_established_it` — over a
   throwaway git tree: the fixed vocabulary, the command, the standing/one-off boundary rule
   printed before any health figure, and the no-caller list printed when empty.
5. `test_every_never_failed_check_carries_an_injection_verdict_and_the_blinded_control_is_vacuous`
   and `test_record_appends_exactly_one_dated_row_beside_the_landed_outcomes_journal`.

Hermetic throughout: every ledger row is a fixture, every classification runs over a throwaway
git tree this file builds and deletes, and no node reads or writes the live automod state dir.
The one assertion against the live filesystem names a PATH and never its contents.

The shape of the fixture ledger — a live `promotions.jsonl`, monthly
`promotions-archive-YYYYMM.jsonl.gz` files beside it, and non-gate rows like
`event:"round_aborted"` mixed in — is the live automod state dir's own, restated here rather
than imported from `tests/test_gate_to_land_lag.py`: that file's `_store` helper builds rows for
a JOIN on `round_id` and carries no per-rung population, so borrowing it would import its
fixture shape and not its ledger format.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.maintenance import check_health_census as CHC  # noqa: E402
NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(days_ago: int, hour: int) -> float:
    """A ledger `ts` (epoch seconds) a whole number of days before NOW at `hour`."""
    return NOW - (days_ago * 86400 + hour * 3600)

#: Every rung the fixture ledger carries, including one invented name. `probe_rung_zzz` exists
#: nowhere in Lloyd's source, so printing it at all is the proof the rung list came from rows.
LEDGER_ROWS = [
    # A zero-fail rung with real seconds: the flag needs a floor, not a verdict.
    *({"event": "gate", "round_id": f"SM_A_{i:02d}", "rung": "vet", "ok": True,
       "ts": _stamp(20, 4), "seconds": 0.5, "data": {}} for i in range(30)),
    # A zero-fail rung whose rows are ALL zero-second: skips, not 554 healthy passes.
    *({"event": "gate", "round_id": f"SM_B_{i:02d}", "rung": "frontend", "ok": True,
       "ts": _stamp(20, 4), "seconds": 0.0, "data": {}} for i in range(25)),
    # A failing rung, so the census is not a wall of greens.
    *({"event": "gate", "round_id": f"SM_C_{i:02d}", "rung": "tests",
       "ok": (i % 3 != 0), "ts": _stamp(10 + i, 6), "seconds": 300.0 + i, "data": {}}
      for i in range(12)),
    # Invented: enumeration must follow the data.
    *({"event": "gate", "round_id": f"SM_D_{i:02d}", "rung": "probe_rung_zzz", "ok": True,
       "ts": _stamp(5, 3), "seconds": 1.0, "data": {}} for i in range(6)),
]

#: Archive-only rungs, in a second monthly file. `venv` is the live case: 0 rows in
#: `promotions.jsonl`, 554 in `promotions-archive-202609.jsonl.gz`, every one `seconds: 0.0`.
ARCHIVE_ROWS = [
    *({"event": "gate", "round_id": f"SM_E_{i:02d}", "rung": "venv", "ok": True,
       "ts": _stamp(50, 2), "seconds": 0.0, "data": {}} for i in range(22)),
    *({"event": "gate", "round_id": f"SM_F_{i:02d}", "rung": "old_rung_yyy", "ok": True,
       "ts": _stamp(80, 2), "seconds": 2.0, "data": {}} for i in range(5)),
]


@pytest.fixture
def census_dir(tmp_path):
    """A state dir holding a swept live ledger and TWO monthly archives.

    Two, not one: clause 2 says "union every archive", and a fixture with a single archive
    cannot tell that from a reader that stopped after the first.
    """
    dirn = tmp_path / "automod-state"
    dirn.mkdir()
    live = dirn / "promotions.jsonl"
    with live.open("w", encoding="utf-8") as fh:
        for row in LEDGER_ROWS:
            fh.write(json.dumps(row) + "\n")
        # A non-gate row carrying none of the fields the census keys on. A reader that
        # filtered on `rung` alone would count it; one that filters on the event will not.
        fh.write(json.dumps({"ts": _stamp(30, 1), "created_at": _utc(_stamp(30, 1)),
                             "event": "round_aborted", "round_id": "SM_Z_00",
                             "reason": "operator"}) + "\n")
    for name, rows in (("promotions-archive-202609.jsonl.gz", ARCHIVE_ROWS),
                       ("promotions-archive-202608.jsonl.gz", [])):
        with gzip.open(dirn / name, "wt", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    return dirn


def run_census(census_dir, root, vault, baseline_dir, **kw):
    return CHC.build_census(dirn=census_dir, root=root, vault=vault,
                            baseline_dir=baseline_dir, now_ts=NOW, **kw)


def flag_of_row(census, name):
    return row_for(census, name)["flag"]


def lines(census):
    return CHC.render(census, now_ts=NOW)


def row_for(census, name):
    hits = [c for c in census["checks"] if c["name"] == name]
    assert len(hits) == 1, f"{name} printed {len(hits)} times: " \
                           f"{[c['name'] for c in census['checks']]}"
    return hits[0]


# --------------------------------------------------------------------------- clauses 1-3
def test_the_rung_rows_are_enumerated_from_the_ledgers_own_rung_values(census_dir, tmp_path):
    """One row per rung the ledger NAMES, with the figures clause 1 lists and both stamps.

    `probe_rung_zzz` is the load-bearing half: it appears in no source file, in no rung list,
    and in nothing but the fixture rows — so printing it is what proves the census enumerates
    the window's distinct `rung` values instead of iterating a constant someone maintains.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    census = run_census(census_dir, REPO_ROOT, empty, empty)
    printed = {c["name"] for c in census["checks"]}
    assert {"vet", "frontend", "tests", "probe_rung_zzz"} <= printed, sorted(printed)
    assert "probe_rung_zzz" not in (REPO_ROOT / "scripts/automod/gate.py").read_text(), (
        "the invented rung name stopped being invented, and with it the proof that the rung "
        "list is data-driven rather than a constant")

    out = "\n".join(lines(census))
    for name in ("vet", "frontend", "tests", "probe_rung_zzz", "venv", "old_rung_yyy"):
        assert f"gate_rung {name}:" in out, f"{name} printed no row:\n{out[:400]}"

    # The row's own figures, checked against the fixture arithmetic rather than re-derived
    # from the same code path: 12 `tests` rows, every third one a fail (i%3==0 → 4 fails).
    tests = row_for(census, "tests")
    assert (tests["passes"], tests["fails"], tests["n"]) == (8, 4, 12), tests
    assert tests["class"] == "gate_rung", tests
    assert tests["mean_seconds"] == pytest.approx(305.5)
    assert tests["median_seconds"] == pytest.approx(305.5)
    assert tests["window_stamps"] == (_utc(_stamp(21, 6)), _utc(_stamp(10, 6))), tests
    head = CHC.render_check_row(tests, floor=CHC.FLOORS["gate_rung"], now_ts=NOW)
    assert "8 pass / 4 fail over n=12" in head, head
    assert "mean=305.5s" in head and "median=305.5s" in head, head
    assert "days_since_last_fail=10.25d" in head, head

    # The window the reader actually saw, printed with both stamps.
    assert (f"window {census['window']['first_ts']} .. {census['window']['last_ts']}" in out), out[:600]


def test_a_rung_present_only_in_an_archive_is_counted_from_the_archive(census_dir, tmp_path):
    """`venv` has zero live rows and lives only in a gzip archive: it must print its archived n.

    This is the live box's own trap, measured in triage: 0 rows in `promotions.jsonl` and 554
    in `promotions-archive-202609.jsonl.gz`, and #2758/#2759 each shipped a reader that saw
    only one file and printed an `n=` that was quietly half the population.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    census = run_census(census_dir, REPO_ROOT, empty, empty)
    venv = row_for(census, "venv")
    assert venv["n"] == 22 and venv["archived_n"] == 22, (
        f"the archive was not read into the denominator: {venv}")
    out = "\n".join(lines(census))
    assert "0/0 (never executed)" not in out.split("gate_rung venv")[1].split("\n")[0], out
    assert census["population"]["archived_rows"] == 27, census["population"]
    assert census["window"]["files"] == ["promotions.jsonl",
                                        "promotions-archive-202608.jsonl.gz",
                                        "promotions-archive-202609.jsonl.gz"], census["window"]
    assert row_for(census, "old_rung_yyy")["n"] == 5, (
        "the SECOND archive went unread: the union is with every file beside the live one, "
        "not with the newest")


def test_the_degenerate_prints_never_execute_never_fail_or_a_zero_second_pass(census_dir,
                                                                              tmp_path):
    """Three lies a census can tell by accident, each pinned in its printed form.

    A dash or a silent omission for a check with no rows reads as "nothing to report" rather
    than "this check has never run here", which is the #2378 `landed_outcomes` finding in
    typographic form. A confident `0 fails` over two runs reads as health. And a rung whose
    rows are all `seconds: 0.0` reads as 554 cheap passes when they are skips — the rung was
    never run and never recorded why, which is exactly what #2758 was opened on.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    census = run_census(census_dir, REPO_ROOT, empty, empty)
    out = "\n".join(lines(census))

    # (a) no rows at all, for a check that was named.
    ghost = CHC.check_stats("named_nowhere", "gate_rung", [])
    assert "0/0 (never executed)" in CHC.render_check_row(ghost, floor=20, now_ts=NOW)
    assert ghost["flag"] == CHC.NEVER_EXECUTED, ghost

    # (b) zero fails BELOW the floor: no never-failed claim, and the n is printed.
    thin = CHC.check_stats("thin_check", "eval_check",
                           [{"event": "gate", "rung": "thin_check", "ok": True,
                             "ts": _stamp(2, 3), "seconds": 1.0}] * 3)
    assert CHC.flag_of(thin, CHC.FLOORS["eval_check"]) == "below floor (n=3)", thin
    head = CHC.render_check_row(thin, floor=CHC.FLOORS["eval_check"], now_ts=NOW)
    assert "below floor (n=3)" in head and "never_failed" not in head, head

    # (c) zero-second rows surface as a share on EVERY row, and 100% is said so.
    venv = row_for(census, "venv")
    assert (venv["zero_second"], venv["n"], venv["zero_second_share"]) == (22, 22, 1.0), venv
    assert "zero-second=22/22 (100%: rung never ran)" in out, out
    assert "zero-second=0/22" not in out
    assert all("zero-second=" in ln for ln in out.splitlines()
               if ln.lstrip().startswith("gate_rung ")) , (
        "a health row printed without its skip share")
    assert CHC.ADDABLE_FIELDS == ("seconds",), (
        "a field joined this sum without the nesting check that `review.rung_wait_s` (queue "
        "wait, rung ran concurrently) and `review.waited_s` (a retry sleep inside `seconds`) "
        "and `restart_flushed.waited_s` (no round_id) each failed")


def test_the_addable_seconds_sum_is_smaller_than_the_naive_one():
    """The denominator trap, tested rather than commented.

    `ADDABLE_FIELDS` could be widened by anyone who wants a bigger number, and every figure
    the census prints would move with it. What is pinned is the RELATIONSHIP: summing every
    numeric field in the data block is strictly larger than the addable sum, so the smaller
    one is a choice and not an accident.
    """
    rows = [{"event": "gate", "rung": "review", "ok": False, "ts": _stamp(1, 3),
             "seconds": 500.0,
             "data": {"rung_wait_s": 120.0, "waited_s": 60.0, "turns": 9}},
            {"event": "gate", "rung": "review", "ok": True, "ts": _stamp(2, 3),
             "seconds": 400.0, "data": {"rung_wait_s": 30.0}}]
    addable = sum(CHC.addable_seconds(r) for r in rows)
    naive = sum(CHC.naive_seconds(r) for r in rows)
    assert addable == 900.0, addable
    assert naive > addable, f"the naive sum ({naive}) must exceed the addable one ({addable})"


# --------------------------------------------------------------------------- clause 4
@pytest.fixture
def eval_tree(tmp_path):
    """A throwaway git tree with four `eval/*.py` files, one per classification path.

    A real `git init` + `git add`, because the classifier's evidence is `git grep` and a
    fixture that faked the grep would also fake the orphans.
    """
    root = tmp_path / "repo"
    (root / "eval").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "agent-services" / "bin").mkdir(parents=True)
    src = {
        "eval/alpha_check.py": "print('alpha')\n",
        "eval/beta_check.py": "print('beta')\n",
        "eval/gamma_probe.py": "print('gamma')\n",
        "eval/lonely_thing.py": "print('lonely')\n",
    }
    for rel, text in src.items():
        (root / rel).write_text(text, encoding="utf-8")
    # alpha: asserted on by a test node that imports it -> test, standing.
    (root / "tests/test_alpha.py").write_text(
        "from eval import alpha_check\n", encoding="utf-8")
    # beta: driven by a shell line -> scheduled job, standing.
    (root / "agent-services/bin/beta-step.sh").write_text(
        "#!/bin/sh\npython eval/beta_check.py --record\n", encoding="utf-8")
    (root / ".gitignore").write_text("", encoding="utf-8")
    for args in (["init", "-q"], ["config", "user.email", "t@example.invalid"],
                 ["config", "user.name", "fixture"], ["add", "-A"],
                 ["commit", "-q", "-m", "fixture"]):
        subprocess.run(["git", *args], cwd=str(root), check=True,
                       capture_output=True, text=True)
    vault = tmp_path / "vault"
    (vault / "autonomy").mkdir(parents=True)
    (vault / "autonomy/105-beta-check-weekly.md").write_text("---\n---\nbody\n",
                                                             encoding="utf-8")
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    (baselines / "gamma_probe-20260101-000000.json").write_text("{}", encoding="utf-8")
    return root, vault, baselines


def test_every_eval_file_prints_one_class_beside_the_command_that_established_it(eval_tree):
    """Vocabulary, evidence, the boundary rule BEFORE any figure, and an orphan list that
    prints even when empty.

    The item's own scope hazard is the thing pinned here: the eval tree is ~72 top-level
    scripts and some are one-off measurement scripts, so a file classified `no caller found`
    has to arrive with the search that established it. An evidence string that cannot be
    copied into a terminal and re-run is not evidence.
    """
    root, vault, baselines = eval_tree
    census = run_census(root / "state", root, vault, baselines)
    by_name = {e["file"]: e for e in census["eval_files"]}
    assert {"eval/alpha_check.py", "eval/beta_check.py", "eval/gamma_probe.py",
            "eval/lonely_thing.py"} <= set(by_name), sorted(by_name)
    assert by_name["eval/alpha_check.py"]["class"] == "test", by_name["eval/alpha_check.py"]
    assert by_name["eval/beta_check.py"]["class"] == "scheduled job", by_name["eval/beta_check.py"]
    assert by_name["eval/gamma_probe.py"]["class"] == "nightly", by_name["eval/gamma_probe.py"]
    assert by_name["eval/lonely_thing.py"]["class"] == "no caller found", \
        by_name["eval/lonely_thing.py"]
    assert set(CHC.CALLER_CLASSES) == {"gate rung", "scheduled job", "nightly", "test",
                                       "no caller found"}, CHC.CALLER_CLASSES
    for c in by_name.values():
        assert c["class"] in CHC.CALLER_CLASSES, c
        assert isinstance(c["standing"], bool), c
        assert any("git grep" in e or "ls " in e for e in c["evidence"]), c

    out = lines(census)
    body = "\n".join(out)
    # The boundary is printed as a rule, and printed BEFORE the first health figure.
    rule_at = next(i for i, ln in enumerate(out) if ln.startswith("standing boundary:"))
    first_row = next(i for i, ln in enumerate(out) if ln.startswith("eval tree:"))
    assert rule_at < first_row, "the standing/one-off boundary came after a health figure"
    assert "STANDING CHECK" in out[rule_at] and "ONE-OFF PROBE" in out[rule_at], out[rule_at]
    assert "no caller found: 1 file(s): eval/lonely_thing.py" in body, body[body.find("no caller"):]
    assert any("zero rows in the window" in ln for ln in out), (
        "a classified file printed with no row of its own — classification and health figure "
        "went missing together")

    # An empty orphan list still prints: silence and "no orphans" are different claims.
    (root / "eval/lonely_thing.py").write_text("print('now called')\n", encoding="utf-8")
    (root / "tests/test_lonely.py").write_text("from eval import lonely_thing\n",
                                               encoding="utf-8")
    for args in (["add", "-A"], ["commit", "-q", "-m", "a caller appeared"]):
        subprocess.run(["git", *args], cwd=str(root), check=True, capture_output=True,
                       text=True)
    out2 = "\n".join(lines(run_census(root / "state", root, vault, baselines)))
    assert "no caller found: 0 file(s) (list empty, printed anyway)" in out2, out2[-400:]


def test_a_file_named_only_inside_eval_is_not_a_caller(eval_tree):
    """An eval script importing another is a cross-reference, not a cadence — and the report
    says that out loud instead of printing `-> nothing` about a search that matched.

    The first version of `_grep_found` dropped `eval/` hits and then printed `nothing`, so the
    evidence line was false whenever the only hits were inside eval/. The claim was the bug.
    """
    root, vault, baselines = eval_tree
    (root / "eval/alpha_check.py").write_text("import eval.lonely_thing\n", encoding="utf-8")
    for args in (["add", "-A"], ["commit", "-q", "-m", "a cross-reference"]):
        subprocess.run(["git", *args], cwd=str(root), check=True, capture_output=True,
                       text=True)
    census = run_census(root / "state", root, vault, baselines)
    lonely = [e for e in census["eval_files"] if e["file"].endswith("lonely_thing.py")][0]
    assert lonely["class"] == "no caller found", lonely
    assert any("only inside eval/" in e for e in lonely["evidence"]), lonely["evidence"]


# --------------------------------------------------------------------------- clause 5
def test_every_never_failed_check_carries_an_injection_verdict(eval_tree, census_dir):
    """A green row with no injection recorded is an acceptance failure, so the census makes
    the pairing structural: `required: True` for every check it flags `never_failed`.

    `frontend` in the fixture is the interesting case — zero fails, 25 rows, and every row
    `seconds: 0.0`. It gets the flag AND a verdict, because a rung that records zero seconds
    and zero fails is precisely the shape that must not go unprobed.
    """
    root, vault, baselines = eval_tree
    inj = [CHC.Injection(name="vet", fault="injected fault A", probe=lambda: True, blind=False),
           CHC.Injection(name="frontend", fault="injected fault B", probe=lambda: False,
                         blind=False),
           CHC.Injection(name="probe_rung_zzz", fault="injected fault C",
                         probe=lambda: True, blind=False)]
    census = CHC.build_census(dirn=census_dir, root=root, vault=vault,
                              baseline_dir=baselines, now_ts=NOW, injections=inj)
    flaggable = {c["name"] for c in census["checks"] if c["flag"] == "never_failed"}
    # `vet` (30 rows) and `frontend` (25) clear the 20-row gate-rung floor; `venv` does too at
    # 22, and MUST — a rung whose every archived row is `seconds: 0.0` and whose every row is a
    # pass is the exact shape that must not go unprobed, which is what #2758 was opened on.
    # `probe_rung_zzz` is the opposite case and is asserted below: 6 rows is under the floor, so
    # the census refuses the never-failed claim instead of printing "0 fails in 6".
    assert {"vet", "frontend", "venv"} == flaggable, sorted(flaggable)
    assert row_for(census, "probe_rung_zzz")["flag"] == "below floor", census["checks"]
    required = {v["check"] for v in census["injections"] if v["required"]}
    assert required == flaggable, (
        f"flagged {sorted(flaggable)} but carried verdicts for {sorted(required)}")
    verdicts = {v["check"]: v["verdict"] for v in census["injections"]}
    assert verdicts["vet"] == "alive (injected fault A caught)", verdicts
    assert verdicts["frontend"].startswith("vacuous (injected fault B not caught)"), verdicts
    out = "\n".join(lines(census))
    assert "VACUOUS" in out and "alive (injected fault A caught)" in out, out
    # A never-failed check the harness did NOT probe is a hole, and the census prints it.
    census2 = CHC.build_census(dirn=census_dir, root=root, vault=vault,
                               baseline_dir=baselines, now_ts=NOW, injections=[])
    assert "not injectable (no probe registered)" in "\n".join(lines(census2))


def test_the_blinded_control_check_comes_out_vacuous(census_dir, tmp_path):
    """The meta-check, and the reason the other verdicts mean anything.

    `flag_of` is handed stats built from rows stripped of everything but a name: it cannot
    tell a check that failed from one that never did, and it reports `never_failed` anyway.
    That is `vacuous`, and it must be REPORTED as vacuous. A harness whose probes could not
    distinguish "the fault was caught" from "the probe was never told" would print `alive` for
    all fifteen rungs and read as a clean bill of health — which is the failure this whole
    item exists to catch, one level up.
    """
    assert CHC._inject_blinded_rung_reader() is False, (
        "the blinded control came out non-vacuous, so the flag no longer depends on the rows")
    stripped = {"n": 30, "fails": 0}
    assert stripped["fails"] == 0 and stripped["n"] == 30, stripped
    assert CHC.flag_of(stripped, CHC.FLOORS["gate_rung"]).startswith("never_failed"), stripped

    # And the harness REPORTS it as vacuous: a probe that correctly finds itself blind but is
    # printed as `alive`, or dropped from the row set, is the same failure one level up.
    def _control(census):
        return [v for v in census["injections"] if v["check"] == "BLINDED-CONTROL"]

    with_control = run_census(census_dir, REPO_ROOT, tmp_path, tmp_path)
    hits = _control(with_control)
    assert len(hits) == 1, f"the control printed {len(hits)} times: {census_injections(with_control)}"
    assert hits[0]["blind"] is True and hits[0]["verdict"].startswith("vacuous"), hits[0]
    assert hits[0]["required"] is True and hits[0]["tried"] is True, hits[0]
    body = "\n".join(lines(with_control))
    assert "injection control BLINDED-CONTROL: vacuous" in body, body[:600]
    # The default registry's real probes all catch their faults on this box, so the only
    # VACUOUS line the live report can carry is the control's — and the summary says so rather
    # than listing the control as if it were a rung.
    assert "VACUOUS checks (fault seeded, check did not catch it): no vacuous check found" in body
    # No probe registered at all: no control line invented, and the flagged checks show the hole.
    without = CHC.build_census(dirn=census_dir, root=REPO_ROOT, vault=tmp_path,
                               baseline_dir=tmp_path, now_ts=NOW, injections=[])
    assert _control(without) == [], "a control verdict was reported for a probe that never ran"


def census_injections(census):
    return sorted(v["check"] for v in census["injections"])


def test_the_shipped_rung_probes_really_catch_their_faults():
    """The two real injections call PRODUCTION detectors, so both are checkable here.

    `code_rules.naive_now_in_artifact_name` must fire on a no-arg `datetime.now()` inside a
    `Path(...)` AND stay silent on the same expression with `timezone.utc` — a probe that only
    demanded the first half would pass on a rule that flags every `Path` call whatsoever.
    The `static` rung is proven against a file that does not compile, not a file with a style
    complaint.
    """
    from scripts.automod import code_rules as CR
    assert CHC._inject_code_rules_naive_now() is True
    assert len(CR.naive_now_in_artifact_name(CHC._ctx(CHC.NAIVE_NOW_BENIGN))) == 0
    assert CHC._inject_static_does_not_compile() is True


def test_record_appends_exactly_one_dated_row_beside_the_landed_outcomes_journal(
        eval_tree, census_dir, tmp_path, monkeypatch, capsys):
    """One row, next to `landed_outcomes.jsonl`, carrying every printed figure.

    The journal's existence is the acceptance clause: #3418 names "a verdict that is only
    printed" as a standing failure mode, and #2378 shipped a `--record` flag with no caller so
    its own journal never existed. The assertion here is that the printed lines and the stored
    row are the SAME numbers, because a journal that drifts from the print is a second
    instrument nobody reads.
    """
    root, vault, baselines = eval_tree
    journal = tmp_path / "state" / "check_health_census.jsonl"
    monkeypatch.setattr(CHC, "census_path", lambda: journal)
    rc = CHC.main(["--record", "--state-dir", str(census_dir), "--repo-root", str(root),
                   "--vault", str(vault), "--baselines", str(baselines)])
    assert rc == 0, rc
    printed = capsys.readouterr().out.splitlines()
    assert journal.exists(), "the report printed and journaled nothing"
    rows = [json.loads(ln) for ln in journal.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1, f"--record appended {len(rows)} rows, expected exactly 1"
    row = rows[0]
    assert row["kind"] == "check_health_census" and row["schema"] == 1, sorted(row)
    assert row["generated_at"].endswith("Z"), row["generated_at"]
    # Every journaled check row has a printed twin and the counts match: the census prints BOTH
    # classes (`gate_rung …` and `eval_check …`), so comparing the journal's length against the
    # gate-rung lines alone would compare a 20-item list to a 6-item filter and pass only by
    # leaving the eval arms unjournaled.
    check_lines = [ln for ln in printed
                   if ln.startswith("  gate_rung ") or ln.startswith("  eval_check ")]
    assert len(row["checks"]) == len(check_lines), (len(row["checks"]), len(check_lines))
    for check in row["checks"]:
        line = [ln for ln in check_lines
                if ln.startswith(f"  {check['class']} {check['name']}: ")]
        assert line, (check["name"], "journaled but printed nowhere")
        if check["n"] == 0:
            # Clause 3(a): a check with no rows prints `0/0 (never executed)` and NOT a fake
            # "0 pass / 0 fail", which would read as two runs that went fine.
            assert CHC.NEVER_EXECUTED in line[0], (check["name"], line)
            continue
        assert (f"{check['passes']} pass / {check['fails']} fail over n={check['n']}"
                in line[0]), (check, line)
    for key in ("window", "floors", "addable_fields", "injections", "counts", "files",
                "ledger"):
        assert key in row, f"{key} is printed but not journaled"
    assert row["addable_fields"] == list(CHC.ADDABLE_FIELDS)
    assert row["floors"] == CHC.FLOORS


def test_the_journal_path_sits_beside_landed_outcomes_and_outside_the_checkout():
    """The path is the clause. #1667's ruling is that runtime data in the tree makes the
    guardian alert hourly, so the journal must sit beside `landed_outcomes.jsonl`."""
    from scripts.automod import landed_outcomes as LO
    assert CHC.census_path().parent == LO.landed_outcomes_path().parent, (
        f"{CHC.census_path()} is not beside {LO.landed_outcomes_path()}")
    assert CHC.census_path().name == "check_health_census.jsonl"
    assert not str(CHC.census_path()).startswith(str(REPO_ROOT)), (
        "the journal resolves inside the checkout, where the guardian alerts on runtime data")


def test_the_census_names_the_live_rungs_it_would_print_and_refuses_to_fabricate(tmp_path):
    """A state dir with no ledger at all prints zero rungs and says so, rather than inventing
    the fifteen it expects.

    The counterfactual is the risk: a census that fell back to a known rung list when its
    population came up empty would print a full, plausible, entirely fictitious report — and
    the row it journaled would be indistinguishable from a real one.
    """
    empty = tmp_path / "no-state"
    empty.mkdir()
    census = run_census(empty, REPO_ROOT, empty, empty)
    assert census["counts"]["gate_rungs"] == 0, census["counts"]
    assert census["population"]["files"] == [], census["population"]
    out = "\n".join(lines(census))
    assert "gate_rung " not in out
    assert "no caller found" in out, "the caller half vanished with the ledger half"
