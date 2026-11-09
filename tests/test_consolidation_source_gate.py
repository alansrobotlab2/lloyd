"""Phase 1.3's source gate must be the emitter's rule, not a copy of it (#1287).

`scripts/mine-trajectories.py::is_emittable()` stopped emitting a `sequence`
candidate flagged `has_error_recovery: false` under #1181, and #1181 recorded the
consequence it did *not* cure: the historical candidate files stay on disk, so
`nightly-skill-consolidation` Phase 1.3 kept hand-adjudicating them five keys a
night. Measured 2026-09-20T08:0xZ over the 3,930 files in
`_pipeline/skills/candidates/`: 761 keys clear Phase 0's verdict drop, `sessions >= 3`
and the 2-snapshot persistence gate; all 761 are `type: sequence`; 653 carry
`has_error_recovery: false`; at the runbook's 5-patterns-per-run cap that is ~152
nights to reach a verdict the emitter computes in one comparison. (The item's own
stricter figure, 592, additionally requires zero error markers in any example step —
the right count for "cannot carry a lesson", the wrong one for a gate keyed on
`is_emittable`, which discriminates on the flag alone. Post-gate pool: 108 keys, not
the 169 that a 761−592 subtraction would suggest.)

These tests pin three things the item asks for and a naive fix gets wrong:

* the gate *delegates* — patch the miner's predicate and the gate's answer moves with
  it, so nothing here is a hand-copied `if` (clause 1, clause 3);
* a key is dropped only for the flag, never for its age, and never for a field the
  front matter does not record (clause 4, clause 5);
* the drop is reported under the name `dropped_by_source_gate`, including when it is
  zero, which is Phase 0.4's rule for the same reason (clause 2).

Corpus figures are quoted, never re-measured here: `_pipeline/` is gitignored, so a
count over it is a fact about one machine on one day. The machinery is tested on
synthetic corpora built in `tmp_path`.
"""
from __future__ import annotations

import importlib.util
import json
import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GATE_PATH = ROOT / "scripts" / "consolidation_source_gate.py"
MINER_PATH = ROOT / "scripts" / "mine-trajectories.py"
VERDICTS_PATH = ROOT / "scripts" / "skill_verdicts.py"
APPLIER_PATH = ROOT / "scripts" / "maintenance" / "apply-1287-phase-1-3-source-gate.py"
SKILL = Path.home() / "obsidian" / "skills" / "nightly-skill-consolidation" / "SKILL.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


csg = _load(GATE_PATH, "csg_under_test")
applier = _load(APPLIER_PATH, "applier_under_test")

# An independent load of the miner, so a test can compare the gate's decision against
# the emitter without borrowing the module object the gate itself chose.
miner_ref = _load(MINER_PATH, "miner_for_source_gate")

# Likewise the ledger: the lapsed-leg tests build a fixture ledger file in its own JSONL
# shape and compare `check`'s release line against its own line renderer, so neither is
# paraphrased here (#4003).
sv = _load(VERDICTS_PATH, "sv_for_source_gate")


# ── the five shapes clause 3 names, built the way the miner builds them ──────

SEQ_TRUE = {"type": "sequence", "ngram_size": 2, "has_error_recovery": True}
SEQ_FALSE = {"type": "sequence", "ngram_size": 2, "has_error_recovery": False}
SEQ_NOFLAG = {"type": "sequence", "ngram_size": 2}
# `mine_error_patterns` never sets `has_error_recovery`; the flag it does not set is
# the whole reason #1181's gate tests the flag with `is False`.
ERROR = {"type": "error", "tool_name": "Bash", "error_type": "not_found"}
SUCCESS = {"type": "success", "tool_name": "Read",
           "params_signature": "file_path_limit_offset"}


def front_matter(pattern: dict, *, sessions: int = 9, status: str = "pending_review",
                 pattern_key: str = "seq-test", record_signature: bool = False) -> dict:
    """The front-matter dict a candidate file carries for `pattern`.

    Mirrors `write_candidate_file`: booleans arrive as the text `true`/`false`, and a
    `success` file records no `params_signature` unless `record_signature` asks for the
    case that cannot occur on disk today.
    """
    fm = {"candidate": "true", "pattern": pattern_key, "type": str(pattern.get("type")),
          "sessions": str(sessions), "first_seen": "2026-04-01",
          "last_seen": "2026-04-12", "status": status}
    if "has_error_recovery" in pattern:
        fm["has_error_recovery"] = "true" if pattern["has_error_recovery"] else "false"
    if record_signature and "params_signature" in pattern:
        fm["params_signature"] = str(pattern["params_signature"])
    return fm


def write_candidate(dir_: Path, *, key: str, date: str, pattern: dict,
                   sessions: int = 9, status: str = "pending_review") -> Path:
    fm = front_matter(pattern, sessions=sessions, status=status, pattern_key=key)
    body = "---\n" + "\n".join(f"{k}: {v}" for k, v in fm.items()) + "\n---\n\n"
    body += f"# Skill Candidate: {key}\n\n- Step 1 (`Bash`) [OK]\n"
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / f"candidate-{key}-{date}.md"
    path.write_text(body, encoding="utf-8")
    return path


def write_pair(dir_: Path, *, key: str, pattern: dict, sessions: int = 9,
               status: str = "pending_review") -> None:
    """Two dated snapshots, so the key clears the 2-snapshot persistence gate."""
    for date in ("20260912", "20260913"):
        write_candidate(dir_, key=key, date=date, pattern=pattern,
                        sessions=sessions, status=status)


# ── clause 1 / 3: the decision is the miner's, not a copy of it ──────────────

def test_the_gate_loads_the_miner_it_delegates_to():
    assert Path(csg.miner().__file__).resolve() == MINER_PATH.resolve()
    assert callable(csg.miner().is_emittable)


def test_patching_the_miners_predicate_moves_the_gates_answer(monkeypatch):
    """Clause 1: the gate decides *through* `is_emittable`, so no shadow copy exists.

    A hand-copied `if pattern["type"] == "sequence" and not flag` passes every
    value-based assertion in this file and then drifts the first time the emitter
    changes — the exact failure #1287's second required-shape bullet names. Replacing
    the loaded miner's attribute must move the gate's answer for a key the real emitter
    keeps, and it must receive a typed dict, not the file's text.
    """
    seen: list[dict] = []

    def refuse_everything(pattern: dict) -> bool:
        seen.append(dict(pattern))
        return False

    monkeypatch.setattr(csg.miner(), "is_emittable", refuse_everything)
    assert csg.drop_reason(front_matter(SEQ_TRUE)) is not None
    assert seen and seen[0]["type"] == "sequence"
    assert seen[0]["has_error_recovery"] is True, "the gate handed a string to the emitter"

    monkeypatch.setattr(csg.miner(), "is_emittable", lambda pattern: True)
    assert csg.drop_reason(front_matter(SEQ_FALSE)) is None


def test_the_gate_hands_the_emitter_typed_fields_not_file_text():
    """What the gate may own: reading a text file. What it may not: the rule.

    `decidable()` and `pattern_from_frontmatter()` exist only to turn a candidate's
    front matter into the dict `is_emittable()` already judges, so this pins the shape
    of the handoff — `type` a string, the flag a real bool, and a field the file does
    not record left out rather than invented.
    """
    assert csg.pattern_from_frontmatter(front_matter(SEQ_FALSE)) == {
        "type": "sequence", "has_error_recovery": False}
    assert csg.pattern_from_frontmatter(front_matter(ERROR)) == {"type": "error"}
    assert csg.pattern_from_frontmatter(
        front_matter(SUCCESS, record_signature=True)) == {
        "type": "success", "params_signature": "file_path_limit_offset"}
    # A non-success key is decidable with or without the flag: for those types the
    # emitter's only refusal input is the flag, and an absent flag is a keep.
    assert csg.decidable(front_matter(ERROR)) is True
    assert csg.decidable(front_matter(SEQ_NOFLAG)) is True
    # A success key is decidable only if its file recorded the signature it is judged on.
    assert csg.decidable(front_matter(SUCCESS)) is False
    assert csg.decidable(front_matter(SUCCESS, record_signature=True)) is True


PARITY_SHAPES = [
    pytest.param(SEQ_TRUE, id="sequence-flagged-true"),
    pytest.param(SEQ_FALSE, id="sequence-flagged-false"),
    pytest.param(ERROR, id="error"),
    pytest.param(SUCCESS, id="success"),
    pytest.param(SEQ_NOFLAG, id="pattern-dict-with-no-has_error_recovery-key"),
]


@pytest.mark.parametrize("pattern", PARITY_SHAPES)
def test_the_gate_and_the_emitter_agree_on_every_shape(pattern):
    """Clause 3: runbook and emitter cannot disagree about what is eligible."""
    expected = miner_ref.is_emittable(dict(pattern))
    reason = csg.drop_reason(front_matter(pattern, record_signature=True))
    assert (reason is None) is expected, (pattern, reason, expected)


def test_the_gate_follows_the_emitter_into_a_class_it_does_not_name():
    """The delegation is real, not a sequence-only special case.

    `is_emittable` also refuses a `success` pattern whose `params_signature` is a bare
    program name or a `*_signature` key-set (#1181's own clause-4 tests). The gate has
    no `success` branch anywhere; if it were re-typing the sequence rule it would keep
    this key.
    """
    bare = {"type": "success", "tool_name": "Bash", "params_signature": "cmd:cd"}
    assert miner_ref.is_emittable(dict(bare)) is False
    fm = front_matter(bare, record_signature=True)
    assert "has_error_recovery" not in fm, "the drop must not depend on the sequence flag"
    reason = csg.drop_reason(fm)
    assert reason is not None and reason.startswith("not_emittable :: type='success'")


def test_a_refusal_reason_names_the_class_without_recomputing_it():
    reason = csg.drop_reason(front_matter(SEQ_FALSE))
    assert reason is not None
    assert reason.startswith("not_emittable :: ")
    assert "has_error_recovery=False" in reason
    assert "type='sequence'" in reason


# ── the front-matter trap: text "false" is not the bool False ────────────────

def test_front_matter_text_false_is_decided_as_the_bool_false():
    """A gate that forwards the raw string silently drops nothing at all.

    `is_emittable` tests the flag with `is False` on purpose (#1181 clause 3 — a falsy
    test would have suppressed every mined `error` pattern too), and a candidate file's
    front matter carries the *text* `false`, which is a non-empty string and therefore
    not `False`. Handing the text over unconverted is the one implementation bug that
    would leave this gate reporting `dropped_by_source_gate: 0` over a corpus of 653
    refusals, so the emitter is asserted on the unconverted value to prove the
    coercion — not the emitter — is what carries the decision.
    """
    assert miner_ref.is_emittable({"type": "sequence",
                                   "has_error_recovery": "false"}) is True
    assert csg.drop_reason({"type": "sequence", "has_error_recovery": "false"}) is not None
    assert csg.drop_reason({"type": "sequence", "has_error_recovery": "False"}) is not None
    assert csg.drop_reason({"type": "sequence", "has_error_recovery": "true"}) is None
    assert csg.drop_reason({"type": "sequence", "has_error_recovery": False}) is not None


# ── clause 4: age is never a reason ─────────────────────────────────────────

def test_a_flagged_true_key_still_reaches_the_evidence_gates(tmp_path):
    """Clause 4: `has_error_recovery: true` survives the source gate at any age.

    #1181's own measurement is why this clause exists: 672 of the 780 actionable
    candidate keys read false, and the 108 that carried a real recovery are the pool
    this runbook is for. The key below is deliberately stale — two snapshots from
    April — because the 2026-09-20 pool is 761 keys whose newest snapshot is on or
    before 2026-09-13, and a gate that dropped on age would have cleared it.
    """
    for date in ("20260407", "20260412"):
        write_candidate(tmp_path, key="seq-recovering", date=date,
                        pattern=SEQ_TRUE, sessions=12)
    result = csg.scan(tmp_path)
    assert result["dropped_by_source_gate"] == 0
    assert [key for key, _fm, _reason in result["kept"]] == ["seq-recovering"]
    assert result["eligible_after_source_gate"] == 1


def test_moving_snapshot_dates_does_not_move_the_drop(tmp_path):
    """No key is dropped for the age of its newest snapshot: dates are not an input."""
    shapes = {"stale-a": SEQ_FALSE, "stale-b": SEQ_TRUE}
    old, new = tmp_path / "old", tmp_path / "new"
    for dir_, dates in ((old, ("20260407", "20260412")), (new, ("20260918", "20260920"))):
        for key, pattern in shapes.items():
            for date in dates:
                write_candidate(dir_, key=key, date=date, pattern=pattern, sessions=8)
    assert [k for k, _f, _r in csg.scan(old)["dropped"]] == \
        [k for k, _f, _r in csg.scan(new)["dropped"]] == ["stale-a"]


# ── clause 5: the drop never covers a class the emitter keeps ────────────────

def test_error_and_flagless_keys_are_never_counted_in_the_drop(tmp_path):
    """Clause 5: `dropped_by_source_gate` counts only what `is_emittable` refuses."""
    keepers = {"err-not-found": ERROR, "succ-decision": SUCCESS, "seq-no-flag": SEQ_NOFLAG}
    refusals = {"seq-order-one": SEQ_FALSE, "seq-order-two": SEQ_FALSE}
    for key, pattern in {**keepers, **refusals}.items():
        write_pair(tmp_path, key=key, pattern=pattern, sessions=20)
    result = csg.scan(tmp_path)
    assert result["eligible"] == 5
    assert result["dropped_by_source_gate"] == 2
    assert sorted(key for key, _f, _r in result["dropped"]) == sorted(refusals)
    assert sorted(key for key, _f, _r in result["kept"]) == sorted(keepers)


def test_front_matter_the_miner_actually_writes_is_kept_when_the_emitter_keeps_it(tmp_path):
    """The shapes above are hand-built; these came out of `write_candidate_file`.

    A hand-built front-matter dict can drift from what the writer emits, and then this
    file would be pinning a fiction. `success` and `error` files are asserted to carry
    no `has_error_recovery` field at all — the census claim (0 of 102 `success` keys
    record a `params_signature` either) restated against the writer rather than a count
    over a gitignored directory.
    """
    emitted = {}
    for label, pattern in (
        ("error", {"type": "error", "tool_name": "Bash", "error_type": "not_found",
                   "params_signature": "protocol", "sessions": {"s1", "s2"},
                   "examples": [], "dates": {"2026-04-01"}, "total_calls": 5,
                   "first_seen": "2026-04-01", "last_seen": "2026-04-12"}),
        ("success", {"type": "success", "tool_name": "Read",
                     "params_signature": "read_limit_on_files_over_2000_lines",
                     "sessions": {"s1", "s2"}, "examples": [], "dates": {"2026-04-01"},
                     "total_calls": 9, "error_count": 0, "error_rate": 0.0,
                     "first_seen": "2026-04-01", "last_seen": "2026-04-12"}),
    ):
        assert miner_ref.is_emittable(dict(pattern)) is True, label
        path = miner_ref.write_candidate_file(pattern, tmp_path)
        assert path, label
        fm = csg.parse_frontmatter(Path(path).read_text(encoding="utf-8"))
        assert fm.get("type") == label
        assert "has_error_recovery" not in fm, f"{label}: writer added the flag"
        assert "params_signature" not in fm, f"{label}: writer added the signature"
        emitted[label] = fm
    assert all(csg.drop_reason(fm) is None for fm in emitted.values()), \
        "a file the emitter wrote must never land in dropped_by_source_gate"


def test_the_drop_is_never_wider_than_the_emitters_refusal():
    """Anti-drift, in the direction that matters: kept-by-the-emitter ⇒ never dropped.

    Over-dropping loses a lesson; under-dropping only costs hand-adjudication. This
    asserts the safe half as a property, so a future `is_emittable()` that stops
    refusing the sequence class moves the gate down and never the other way.
    """
    for pattern in (SEQ_TRUE, SEQ_NOFLAG, ERROR, SUCCESS):
        assert miner_ref.is_emittable(dict(pattern)) is True
        assert csg.drop_reason(front_matter(pattern)) is None, pattern
    assert miner_ref.is_emittable(dict(SEQ_FALSE)) is False
    assert csg.drop_reason(front_matter(SEQ_FALSE)) is not None


# ── clauses 1 / 2: the report, including when it is empty ────────────────────

def test_a_dropped_key_is_counted_under_the_name_dropped_by_source_gate(tmp_path):
    """Clause 1's second half: the count leaves the tool under that exact name."""
    write_pair(tmp_path, key="seq-plain-order", pattern=SEQ_FALSE, sessions=40)
    result = csg.scan(tmp_path)
    assert result["dropped_by_source_gate"] == 1
    assert result["eligible_after_source_gate"] == 0
    assert result["dropped"][0][0] == "seq-plain-order"


def test_the_totals_line_is_last_and_names_the_count_even_at_zero(tmp_path, capsys):
    """Phase 0.4's rule for the same reason: an unexplained zero reads as a broken job.

    `skill_verdicts.py check` keeps its counts on the final line because the runbook
    and the nightly greps read them off `splitlines()[-1]`; a source-gate line that
    displaced them would break a reader that is not looking for it.
    """
    write_pair(tmp_path, key="seq-recovering", pattern=SEQ_TRUE, sessions=40)
    assert csg.main(["check", "--candidates", str(tmp_path)]) == 0
    totals = capsys.readouterr().out.strip().splitlines()[-1]
    assert "dropped_by_source_gate: 0" in totals
    assert "eligible_after_source_gate: 1" in totals


def test_a_run_that_drops_every_eligible_key_still_reports_the_count(tmp_path, capsys):
    """The 2026-09-20 corpus *is* this case: 653 of 761 eligible keys are refused."""
    write_pair(tmp_path, key="seq-plain-order", pattern=SEQ_FALSE, sessions=84)
    assert csg.main(["check", "--candidates", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "dropped_by_source_gate: 1" in out.splitlines()[-1]
    assert "DROP seq-plain-order :: not_emittable ::" in out


def test_a_key_the_gate_cannot_decide_is_reported_as_kept(tmp_path, capsys):
    """The safe direction, stated as behaviour: no field, no drop, and it says so."""
    write_pair(tmp_path, key="succ-decision", pattern=SUCCESS, sessions=7)
    assert csg.main(["check", "--candidates", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "KEEP succ-decision" in out
    assert "dropped_by_source_gate: 0" in out.splitlines()[-1]


def test_per_key_drop_lines_name_the_pattern_the_way_phase_0_ones_do(tmp_path, capsys):
    for date in ("20260912", "20260913"):
        write_candidate(tmp_path, key="seq-plain-order", date=date,
                        pattern=SEQ_FALSE, sessions=84)
    csg.main(["check", "--candidates", str(tmp_path)])
    out = capsys.readouterr().out
    assert re.search(r"^DROP seq-plain-order :: not_emittable :: .+ \(newest snapshot, sessions=84\)$",
                     out, re.M)


# ── the evidence gates the source gate sits behind ──────────────────────────

def test_a_key_with_one_snapshot_or_a_thin_session_count_is_not_eligible(tmp_path):
    """The source gate counts only inside Phase 1.2/1.3's evidence pool."""
    write_candidate(tmp_path, key="seq-one-snapshot", date="20260913",
                    pattern=SEQ_FALSE, sessions=9)
    for date in ("20260912", "20260913"):
        write_candidate(tmp_path, key="seq-two-thin", date=date, pattern=SEQ_FALSE, sessions=2)
    result = csg.scan(tmp_path)
    assert result["eligible"] == 0
    assert result["dropped_by_source_gate"] == 0
    assert result["scanned"] == 2


def test_a_dispositioned_key_is_excluded_before_the_source_gate(tmp_path):
    """Phase 0's drop happens upstream, so a terminal key is in neither count."""
    write_pair(tmp_path, key="seq-already-verdicted", pattern=SEQ_FALSE, sessions=30,
               status="rejected_artifact_class — verdicted 2026-09-20")
    result = csg.scan(tmp_path)
    assert result["eligible"] == 0 and result["dropped_by_source_gate"] == 0


def test_terminal_statuses_are_read_from_the_ledger_not_listed_here():
    """#830 is adding a terminal verdict to `skill_verdicts.TERMINAL_VERDICTS`.

    Deriving the skip list from that module means the addition lands in Phase 1.3's
    evidence filter for free; a second hard-coded list here would drift, and a drift in
    that direction re-admits keys Phase 0 has already retired — the failure Phase 0
    exists to remove.
    """
    from_ledger = set(csg.verdicts().TERMINAL_VERDICTS)
    assert from_ledger <= set(csg.terminal_statuses())
    assert csg.verdicts().SUPERSEDED_STATUS in csg.terminal_statuses()
    assert {"consolidated", "noise"} <= set(csg.terminal_statuses())


def test_keys_are_grouped_on_their_own_pattern_field_not_a_filename_glob(tmp_path):
    """A short key is a strict prefix of another key's filename (Phase 5.1).

    Globbing per slug would let `seq-2-bash-fs-read` absorb the snapshots of
    `seq-2-bash-fs-read-and-more` and mis-count both pools.
    """
    for key in ("seq-2-bash-fs-read", "seq-2-bash-fs-read-and-more"):
        write_pair(tmp_path, key=key, pattern=SEQ_FALSE, sessions=9)
    result = csg.scan(tmp_path)
    assert result["eligible"] == 2
    assert sorted(k for k, _f, _r in result["dropped"]) == [
        "seq-2-bash-fs-read", "seq-2-bash-fs-read-and-more"]


# ── clauses 1–4: a superseded snapshot whose ledger leg has lapsed (#4003) ───

#: The status `mine-trajectories.py::status_block` writes when the ledger blocks a key.
SUPERSEDED = "superseded_by_verdict"
LAPSED = "seq-2-lapsed-err-read"
BINDING = "seq-2-binding-err-read"


def verdict_row(key: str, *, age_days: int,
                verdict: str = "reviewed_no_skill",
                occurrences_at_decision: int = 4) -> dict:
    """One terminal ledger line, anchored `age_days` ago.

    Written in the ledger's own JSONL shape (`tests/test_skill_verdicts.py`'s
    `_crossed_cohort_ledger` is the precedent) so the gate is tested against the file
    `load_verdicts` parses, not against a dict this file invented. `first_decided_at` is
    the anchor the 60-day leg weighs on (#2746); `occurrences_at_decision: 4` keeps the
    growth leg out of every verdict below, so each one turns on the date alone.
    """
    stamp = (datetime.now(tz=timezone.utc) - timedelta(days=age_days))
    return {"pattern_key": key, "verdict": verdict, "reason": "fixture",
            "decided_at": stamp.isoformat(), "first_decided_at": stamp.isoformat(),
            "occurrences_at_decision": occurrences_at_decision}


def point_ledger_at(tmp_path, monkeypatch, rows: list[dict]) -> Path:
    """A live ledger at `$SKILL_VERDICTS_STORE`, carrying `rows`."""
    ledger = tmp_path / "reviews" / "verdicts.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                      encoding="utf-8")
    monkeypatch.setenv("SKILL_VERDICTS_STORE", str(ledger))
    return ledger


def check_lines(corpus: Path, capsys) -> list[str]:
    """`check`'s whole stdout for a corpus, through the real CLI entry point.

    The nightly greps read this program's stdout, not a Python return value, so the
    clause that promises a line is graded on the process boundary it names.
    """
    assert csg.main(["check", "--candidates", str(corpus)]) == 0
    return capsys.readouterr().out.strip().splitlines()


def test_a_superseded_key_whose_ledger_leg_crossed_reaches_the_evidence_gates(
        tmp_path, monkeypatch):
    """Clause 1: the file's status is a *copy* of a row, and the row has lapsed.

    The fixture is one key with two dated snapshots and `sessions: 3` — the shape
    `seq-2-edit-err-read` had on the live corpus the night its leg crossed: `check`
    printed `REOPEN` for it while this gate's last line read `eligible: 0
    dropped_by_source_gate: 0 eligible_after_source_gate: 0`, so the key sat in none of
    the three buckets and no report named it.
    """
    corpus = tmp_path / "candidates"
    write_pair(corpus, key=LAPSED, pattern=SEQ_TRUE, sessions=3, status=SUPERSEDED)
    point_ledger_at(tmp_path, monkeypatch,
                    [verdict_row(LAPSED, age_days=sv.REOPEN_AFTER_DAYS + 1)])

    result = csg.scan(corpus)

    assert result["scanned"] == 1
    assert result["eligible"] == 1, "a lapsed verdict must not keep filtering"
    assert [k for k, _f, _r in result["kept"]] == [LAPSED]
    assert result["eligible_after_source_gate"] == 1
    assert result["dropped_by_source_gate"] == 0
    assert result["released_lapsed_leg_keys"] == [LAPSED]


def test_the_same_fixture_reads_eligible_zero_with_no_lapsed_leg_to_release_it(
        tmp_path):
    """Clause 1's other half: today's reading of that same corpus is `eligible: 0`.

    `ledger={}` is "no row behind any of these keys", which is what the status filter
    already saw — an absent or emptied ledger releases nothing, because releasing on it
    is the 09-09 re-adjudication loop #530 exists to prevent. The last assertion is the
    pre-fix call itself, unchanged: `evidence_eligible` with its default flag.
    """
    corpus = tmp_path / "candidates"
    write_pair(corpus, key=LAPSED, pattern=SEQ_TRUE, sessions=3, status=SUPERSEDED)

    assert csg.scan(corpus, ledger={})["eligible"] == 0
    assert csg.scan(corpus, ledger={})["released_by_lapsed_leg"] == 0
    rows = csg.group_candidates(corpus)[LAPSED]
    assert csg.evidence_eligible(rows, csg.terminal_statuses()) is False


def test_a_still_binding_verdict_keeps_the_key_filtered_and_the_status_terminal(
        tmp_path, monkeypatch):
    """Clause 2: the same front matter, a leg that has NOT crossed, filters as today.

    The guard #530 exists for is that the ledger is *asked*, not that the status is
    ignored, so this pins both sides: the row really is read and really binds, and
    `SUPERSEDED_STATUS` is still in the skip list it was filtered through.
    """
    corpus = tmp_path / "candidates"
    write_pair(corpus, key=BINDING, pattern=SEQ_TRUE, sessions=9, status=SUPERSEDED)
    ledger = point_ledger_at(tmp_path, monkeypatch,
                             [verdict_row(BINDING, age_days=sv.REOPEN_AFTER_DAYS - 20)])
    assert ledger.exists()

    table = csg.ledger_verdict_table()
    assert table is not None and BINDING in table
    assert csg.terminal_verdict_binds(BINDING, 0, table) is True, \
        "the row must still bind, or this test proves nothing"

    result = csg.scan(corpus, ledger=table)
    assert result["eligible"] == 0
    assert result["dropped_by_source_gate"] == 0
    assert result["released_by_lapsed_leg"] == 0
    assert csg.verdicts().SUPERSEDED_STATUS in csg.terminal_statuses()


def test_the_ledger_is_never_read_for_a_status_the_ledger_did_not_mint(
        tmp_path, monkeypatch):
    """Clause 3: `consolidated`, `noise` and a `TERMINAL_VERDICTS` value stay filtered.

    Each of the three carries a ledger row whose leg crossed, so if the ask were keyed on
    anything but the one status the miner mints, all three would be released. They are
    dispositions a consolidator or a hand adjudication *wrote*, so the file is the record
    and asking the ledger is not owed — and the assertion that no read happens at all is
    what makes "consulted only for that status" checkable rather than a comment.
    """
    def no_ledger_read():
        raise AssertionError("the ledger was read for a status it did not mint")

    monkeypatch.setattr(csg, "ledger_verdict_table", no_ledger_read)
    corpus = tmp_path / "candidates"
    for key, status in (("seq-consolidated", "consolidated"),
                        ("seq-noise", "noise"),
                        ("seq-verdicted", "reviewed_no_skill")):
        write_pair(corpus, key=key, pattern=SEQ_TRUE, sessions=9, status=status)

    result = csg.scan(corpus, ledger={
        "seq-consolidated": verdict_row("seq-consolidated",
                                        age_days=sv.REOPEN_AFTER_DAYS + 1),
        "seq-noise": verdict_row("seq-noise", age_days=sv.REOPEN_AFTER_DAYS + 1),
        "seq-verdicted": verdict_row("seq-verdicted",
                                     age_days=sv.REOPEN_AFTER_DAYS + 1)})

    assert result["scanned"] == 3
    assert result["eligible"] == 0
    assert result["released_by_lapsed_leg"] == 0


def test_a_superseded_key_is_released_beside_a_consolidated_one_that_is_not(
        tmp_path, monkeypatch):
    """Clause 3 again, from the other side, on one corpus and one ledger read.

    `seq-lapsed-pair` has NO ledger row at all and is released; `seq-consolidated-pair`
    has a row whose leg crossed and is not, because for that status the row is never
    asked. A fix that released on "the ledger has no binding row" — i.e. that dropped
    `SUPERSEDED_STATUS` from the skip list wholesale — would let the second key through.
    """
    corpus = tmp_path / "candidates"
    write_pair(corpus, key="seq-lapsed-pair", pattern=SEQ_TRUE, sessions=4,
               status=SUPERSEDED)
    write_pair(corpus, key="seq-consolidated-pair", pattern=SEQ_TRUE, sessions=4,
               status="consolidated")
    point_ledger_at(tmp_path, monkeypatch,
                    [verdict_row("seq-consolidated-pair",
                                 age_days=sv.REOPEN_AFTER_DAYS + 1)])

    result = csg.scan(corpus)

    assert result["released_lapsed_leg_keys"] == ["seq-lapsed-pair"]
    assert [k for k, _f, _r in result["kept"]] == ["seq-lapsed-pair"]
    assert result["eligible"] == 1 and result["scanned"] == 2


def test_check_names_the_released_keys_in_the_ledger_reopened_grammar(
        tmp_path, monkeypatch, capsys):
    """Clause 4: the release is reported, in `ledger_reopened_keys`' grammar.

    `dropped_by_source_gate: 0` was published over a pool emptied upstream of the
    emitter, which is how a silent drop read as a clean emitter. So the line names the
    count and every key, and prints the zero case too — and the totals stay the last
    line, which is the contract the runbook greps.
    """
    corpus = tmp_path / "candidates"
    write_pair(corpus, key=LAPSED, pattern=SEQ_TRUE, sessions=3, status=SUPERSEDED)
    write_pair(corpus, key=BINDING, pattern=SEQ_TRUE, sessions=9, status=SUPERSEDED)
    point_ledger_at(tmp_path, monkeypatch, [
        verdict_row(LAPSED, age_days=sv.REOPEN_AFTER_DAYS + 1),
        verdict_row(BINDING, age_days=sv.REOPEN_AFTER_DAYS - 20)])

    lines = check_lines(corpus, capsys)

    released = [ln for ln in lines if ln.startswith("released_by_lapsed_leg:")]
    assert released == [f"released_by_lapsed_leg: 1 :: {LAPSED}"]
    # The same grammar and unit as the verdicts module's own line, byte for byte:
    # swap the label and this is what `ledger_reopened_line` would have printed.
    assert released[0].replace("released_by_lapsed_leg",
                               sv.LEDGER_REOPENED_LABEL, 1) == \
        sv.ledger_reopened_line([LAPSED])
    assert lines[-1].startswith("scanned:")
    assert "eligible: 1" in lines[-1] and "dropped_by_source_gate: 0" in lines[-1]


def test_check_prints_the_zero_release_case_when_no_leg_has_lapsed(
        tmp_path, monkeypatch, capsys):
    """Clause 4's zero case: a line that appears only on findings cannot tell a quiet
    night from a comparison nobody ran (#2644's reason for the line it sits beside)."""
    corpus = tmp_path / "candidates"
    write_pair(corpus, key=BINDING, pattern=SEQ_TRUE, sessions=9, status=SUPERSEDED)
    point_ledger_at(tmp_path, monkeypatch,
                    [verdict_row(BINDING, age_days=sv.REOPEN_AFTER_DAYS - 20)])

    lines = check_lines(corpus, capsys)

    assert "released_by_lapsed_leg: 0 :: none" in lines
    assert lines[-1].startswith("scanned:")


# ── clause 2: the runbook says it, in the right place ───────────────────────

def _runbook() -> str:
    """The runbook text this clause has to be true of.

    The skill lives in the vault, which the code round's diff cannot reach, so the
    clause arrives through the vault route from `APPLIER_PATH`. Grading the live file
    only would make every test below vacuous until that landing happened — the shape
    this repo already rejects, where "#530's skill clause was unverifiable, so it was
    held as a patch, and the patch was never applied". So: read the vault file, and if
    the clause is not there yet, grade exactly the text the applier will write. When the
    applier can no longer place it, `patched_text` raises and this fails loudly rather
    than passing on stale prose. Once the vault carries the clause the applier is a
    no-op and the assertions read the live runbook.
    """
    assert SKILL.exists(), f"{SKILL} is absent: the runbook this clause binds is not here"
    text = SKILL.read_text(encoding="utf-8")
    if applier.MARKER not in text:
        warnings.warn(f"{SKILL} does not carry the source-gate clause yet; grading "
                      "scripts/maintenance/apply-1287-phase-1-3-source-gate.py's output",
                      stacklevel=2)
        text = applier.patched_text(text)
    return text


@pytest.fixture(scope="module")
def phase_1_3() -> str:
    body = _runbook()
    start = body.index("### 1.3 Build work list")
    return body[start:body.index("## Phase 2:")]


def test_phase_1_3_names_the_source_gate_and_its_report_line(phase_1_3):
    """Clause 2: the drop, the command, and the count, between the two anchors."""
    assert "consolidation_source_gate.py" in phase_1_3
    assert "check --candidates ~/lloyd-data/_pipeline/skills/candidates/" in phase_1_3
    assert "dropped_by_source_gate:" in phase_1_3
    assert "eligible_after_source_gate" in phase_1_3
    assert "script_absent" in phase_1_3, "an unavailable gate must be reported, not skipped"
    # Before the cap: the gate has to be a work-list rule, not a reporting footnote.
    assert phase_1_3.index("consolidation_source_gate.py") < \
        phase_1_3.index("5 patterns per run")
    assert "is_emittable" in phase_1_3, "the runbook must say whose rule this is"
    assert "Phase 0" in phase_1_3, "the source gate sits behind Phase 0's verdict drop"


def test_the_runbook_names_the_command_the_way_the_tool_spells_it(phase_1_3):
    """A command in a runbook that cannot run is worse than no command (Phase 0.6)."""
    invoked = re.search(r"([^\s`]+consolidation_source_gate\.py)\s*\\?\s+check --candidates",
                        phase_1_3)
    assert invoked, phase_1_3
    assert Path(invoked.group(1).replace("~/lloyd", str(ROOT))).resolve() == GATE_PATH.resolve()


def test_phase_0_still_owns_the_verdict_drop_after_the_source_gate_landed(phase_1_3):
    """#830 and #1287 edit the same section; the verdict drop must survive intact."""
    body = SKILL.read_text(encoding="utf-8")
    assert "skipped_by_verdict: N" in body
    assert "skill_verdicts.py check" in body
    assert body.index("skill_verdicts.py check") < body.index("### 1.3 Build work list")
    assert "Phase 0 dropped nothing" in phase_1_3


def test_the_source_gate_reports_a_post_gate_pool_so_a_zero_is_explained(phase_1_3):
    """A future reader must be able to tell 108 from 761 without re-deriving either.

    The numbers in the runbook are dated on purpose:
    `falsifier-20260920-consolidation.py --corpus` still prints `eligible_keys=761`
    because it implements the *pre-gate* Phase 1.3 filter, so only this tool's own
    output is evidence about the post-gate pool.
    """
    assert "falsifier-20260920" in phase_1_3
    assert "152" in phase_1_3, "the drain cost is what justifies the rule living here"
    assert "1181" in phase_1_3, "the clause this closes is #1181's own 'does NOT cure'"
