"""Tests for the skill-candidate verdict ledger (#530).

The defect these pin, in one sentence: the nightly skill pipeline rejects a pattern,
the miner regenerates it the next night as `status: pending_review`, and the rejection
is re-made by hand. Run #57 on 2026-09-01: "the miner regenerated both files as
`pending_review`, silently wiping the previous review — that's the real process bug
here."

Nothing here touches `~/lloyd/_pipeline` or the vault: every assertion runs against a
tmp ledger and a tmp candidates dir, so a nightly job rewriting the live corpus cannot
fail this file for somebody else's change.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sv = _load("skill_verdicts", "scripts/skill_verdicts.py")
mt = _load("mine_trajectories_530", "scripts/mine-trajectories.py")


def error_pattern(tool="Bash", error_type="timeout", calls=13):
    return {
        "type": "error",
        "tool_name": tool,
        "error_type": error_type,
        "total_calls": calls,
        "sessions": {"s1", "s2", "s3"},
        "first_seen": "2026-09-03",
        "last_seen": "2026-09-09",
        "examples": [{
            "session_key": "s1",
            "date": "2026-09-09",
            "tool": tool,
            "error_type": error_type,
            "params_summary": {"command": "sleep 900"},
            "result_summary": "command timed out after 120000ms",
        }],
    }


def sequence_pattern(sequence=("bash_fs", "read")):
    return {
        "type": "sequence",
        "ngram_size": len(sequence),
        "sequence_str": " -> ".join(sequence),
        "sessions": {"s1", "s2"},
        "first_seen": "2026-09-02",
        "last_seen": "2026-09-09",
        "has_error_recovery": False,
        "examples": [{
            "session_key": "s1",
            "date": "2026-09-09",
            "steps": [{"tool": "Bash", "label": "list files", "params_summary": {},
                       "is_error": False, "result_summary": "ok"}],
        }],
    }


def frontmatter(text: str) -> str:
    return text.split("---", 2)[1]


def status_of(path: Path) -> str:
    """The candidate's `status:` value — the exact thing the acceptance check greps."""
    line = next(
        (ln for ln in frontmatter(path.read_text()).splitlines() if ln.startswith("status:")),
        "",
    )
    return line.split(":", 1)[1].split()[0] if line else ""


@pytest.fixture
def store(tmp_path) -> Path:
    return tmp_path / "reviews" / "verdicts.jsonl"


@pytest.fixture
def seeded(store) -> Path:
    """A ledger carrying the real 2026-09-09 disposition for Bash/timeout: real signal,
    already owned by the installed `bash-timeout` skill."""
    sv.record_verdict(
        store=store,
        pattern_key="Bash/timeout",
        verdict="reviewed_no_skill",
        reason="installed skill bash-timeout Pattern 3 cites this exact signature",
        evidence_cmd="grep -c 'command timed out' ~/obsidian/skills/bash-timeout/SKILL.md",
        occurrences=13,
        source_candidate="candidate-bash-timeout-20260909.md",
    )
    return store


# ── The acceptance check, pinned ─────────────────────────────────────────────

def test_terminal_key_cannot_be_emitted_as_pending_review(seeded, tmp_path):
    """#530's core line: a key carrying a terminal verdict is never `pending_review`."""
    out = tmp_path / "candidates"
    path = Path(mt.write_candidate_file(error_pattern(), out, verdict_store=seeded))

    assert status_of(path) == "superseded_by_verdict"
    assert "status: pending_review" not in path.read_text()


def test_verdict_travels_with_the_file(seeded, tmp_path):
    """The reason must be readable where the next run looks at it, or the next run
    re-adjudicates from scratch — which is the defect."""
    path = Path(mt.write_candidate_file(error_pattern(), tmp_path / "c", verdict_store=seeded))
    head = frontmatter(path.read_text())

    assert "verdict: reviewed_no_skill" in head
    assert "grep -c" in head  # evidence_cmd: the check that can falsify the verdict
    assert "bash-timeout Pattern 3" in head


def test_pattern_with_no_verdict_still_reaches_pending_review(store, tmp_path):
    """The gate must not blind the pipeline: an unadjudicated pattern still proposes."""
    assert not store.exists()
    path = Path(mt.write_candidate_file(
        error_pattern(tool="Read", error_type="validation"), tmp_path / "c", verdict_store=store))

    assert status_of(path) == "pending_review"


def test_non_terminal_verdict_does_not_block(store, tmp_path):
    """`consolidated`/`proposed` are dispositions, not verdicts — a below-threshold
    patch must keep accumulating evidence."""
    sv.record_verdict(
        store=store, pattern_key="Bash/timeout", verdict="proposed",
        reason="patch below auto-apply threshold", evidence_cmd="ls ~/x", occurrences=13,
    )
    path = Path(mt.write_candidate_file(error_pattern(), tmp_path / "c", verdict_store=store))

    assert status_of(path) == "pending_review"


def test_key_shape_is_what_the_written_file_records(tmp_path, store):
    """The ledger and the corpus join on the pattern key and nothing else. If the
    derivation ever drifts from the `pattern:` field the miner writes, every lookup
    silently misses — a gate that reads its own missing input reports 'no verdicts'."""
    for pattern in (error_pattern(), sequence_pattern()):
        path = Path(mt.write_candidate_file(pattern, tmp_path / "c", verdict_store=store))
        field = [ln for ln in frontmatter(path.read_text()).splitlines()
                 if ln.startswith("pattern:")][0].split(":", 1)[1].strip()
        assert mt.candidate_pattern_key(pattern) == field


def test_sequence_keys_are_gated_too(seeded, store, tmp_path):
    """Error slugs are the common case, not the only one."""
    sv.record_verdict(
        store=store, pattern_key="seq-2-bash-fs-read", verdict="rejected_false_positive",
        reason="co-occurs by accident, no causal link", evidence_cmd="grep -r x ~/y",
        occurrences=40,
    )
    path = Path(mt.write_candidate_file(sequence_pattern(), tmp_path / "c", verdict_store=store))

    assert status_of(path) == "superseded_by_verdict"


def test_superseded_pattern_keys_names_the_blocked_ones_only(seeded, tmp_path):
    """The report number the runbook must print, and the reason a silent zero is a
    defect: this is the list `consolidation-*.md` reports per key."""
    patterns = [
        error_pattern(),                                              # blocked
        error_pattern(tool="Edit", error_type="not_found"),           # free
        sequence_pattern(),                                           # free
    ]
    assert mt.superseded_pattern_keys(patterns, store=seeded) == ["Bash/timeout"]


# ── Ledger semantics ─────────────────────────────────────────────────────────

def test_record_requires_a_falsifiable_check(store):
    """A verdict with no re-executable evidence is an assertion the next run can only
    inherit — the #525 weakness, formalised away here rather than trusted."""
    with pytest.raises(ValueError, match="evidence_cmd"):
        sv.record_verdict(store=store, pattern_key="Bash/timeout",
                          verdict="reviewed_no_skill", reason="seen before", evidence_cmd="")
    with pytest.raises(ValueError, match="reason"):
        sv.record_verdict(store=store, pattern_key="Bash/timeout",
                          verdict="reviewed_no_skill", reason="", evidence_cmd="ls")


def test_decisions_track_lines_one_to_one(seeded, store):
    """Acceptance: `verdicts.jsonl` line count tracks decisions 1:1, and a later run
    does not re-adjudicate a key that already has a line."""
    assert len(store.read_text().strip().splitlines()) == 1
    sv.record_verdict(store=store, pattern_key="Edit/not_found", verdict="reviewed_no_skill",
                      reason="owned by file-mutation-safety", evidence_cmd="grep x", occurrences=10)

    lines = store.read_text().strip().splitlines()
    assert len(lines) == 2 == len({json.loads(ln)["pattern_key"] for ln in lines})


def test_latest_line_per_key_wins(store):
    """Append-only with a reopen as its own line — history is never rewritten, so a
    wrong verdict stays auditable."""
    sv.record_verdict(store=store, pattern_key="Bash/logic", verdict="rejected_unverifiable",
                      reason="no error text to ground a skill in", evidence_cmd="grep x")
    sv.record_verdict(store=store, pattern_key="Bash/logic", verdict="reviewed_no_skill",
                      reason="mechanised since: error_tools[] now carries result_summary",
                      evidence_cmd="grep result_summary scripts/mine-trajectories.py")
    rows = sv.load_verdicts(store)

    assert len(rows) == 1
    assert rows["Bash/logic"]["verdict"] == "reviewed_no_skill"


def test_terminal_verdict_lookup(seeded):
    assert sv.terminal_verdict("Bash/timeout", store=seeded, occurrences=13)
    assert sv.terminal_verdict("Bash/network", store=seeded, occurrences=3) is None


def test_verdict_reopens_after_its_ttl(store):
    """A sticky verdict must not bury a pattern that becomes real later (item risk,
    and WikiSkill's own unbounded-growth gap)."""
    old = (datetime.now(tz=timezone.utc) - timedelta(days=sv.REOPEN_AFTER_DAYS + 5)
           ).strftime("%Y-%m-%dT%H:%M:%SZ")
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="owned elsewhere", evidence_cmd="grep x", occurrences=13,
                      decided_at=old)

    assert sv.terminal_verdict("Bash/timeout", store=store, occurrences=13) is None


def test_verdict_reopens_when_the_pattern_grew(store):
    """>10x the occurrences at the decision means the evidence moved; re-ask."""
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="n=1 signature over 3 sessions is not a pattern",
                      evidence_cmd="grep x", occurrences=3)

    assert sv.terminal_verdict("Bash/timeout", store=store, occurrences=30)
    assert sv.terminal_verdict("Bash/timeout", store=store, occurrences=31) is None


def test_unparseable_stamp_does_not_silently_reopen(store):
    """The reopen path is the one that can re-mint a rejected skill, so a verdict whose
    date cannot be read stays binding rather than failing open."""
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="owned elsewhere", evidence_cmd="grep x", occurrences=13,
                      decided_at="not-a-timestamp")

    assert sv.terminal_verdict("Bash/timeout", store=store, occurrences=13)


# ── The two runbooks' entry points ───────────────────────────────────────────

def test_check_reports_skipped_by_verdict_with_reasons(seeded, tmp_path, capsys):
    """`skipped_by_verdict: N` must be a number a report can carry, with a reason per
    key — replacing the silent zero #391 attributes to #58."""
    cands = tmp_path / "candidates"
    cands.mkdir()
    mt.write_candidate_file(error_pattern(), cands, verdict_store=seeded)
    mt.write_candidate_file(error_pattern(tool="Edit", error_type="not_found"), cands,
                            verdict_store=seeded)

    assert sv.main(["check", "--candidates", str(cands), "--store", str(seeded)]) == 0
    out = capsys.readouterr().out

    assert "skipped_by_verdict: 1" in out
    assert "SKIP Bash/timeout :: reviewed_no_skill ::" in out
    assert "PROCEED Edit/not_found" in out


def test_seed_harvests_the_dispositions_already_on_disk(tmp_path, store, capsys):
    """Re-seed source named by the acceptance: per-file `status:` lines from the 09-08
    and 09-09 candidates, keyed on the (tool, error_type) slug."""
    cands = tmp_path / "candidates"
    cands.mkdir()
    (cands / "candidate-read-logic-20260909.md").write_text(
        "---\ncandidate: true\npattern: Read/logic\ntype: error\noccurrences: 5\n"
        "status: reviewed_no_skill — REAL, mechanised; nearest coverage file-path-resolution\n"
        "---\n\n# body\n\nstatus: pending_review\n",
        encoding="utf-8",
    )
    (cands / "candidate-seq-noise-20260909.md").write_text(
        "---\npattern: seq-2-bash-fs-read\nstatus: noise\n---\n", encoding="utf-8")

    assert sv.main(["seed", "--candidates", str(cands), "--store", str(store)]) == 0
    rows = sv.load_verdicts(store)

    assert list(rows) == ["Read/logic"], "only terminal statuses are verdicts"
    assert rows["Read/logic"]["verdict"] == "reviewed_no_skill"
    assert "file-path-resolution" in rows["Read/logic"]["reason"]
    assert rows["Read/logic"]["evidence_cmd"].startswith("grep ")

    capsys.readouterr()
    sv.main(["seed", "--candidates", str(cands), "--store", str(store)])
    assert "keep: Read/logic already in the ledger" in capsys.readouterr().out
    assert len(store.read_text().strip().splitlines()) == 1, "re-seeding must not re-adjudicate"


def test_body_prose_is_not_mistaken_for_the_status_field(tmp_path, store):
    """`status: pending_review` appears in mined body prose. Reading the body would let
    a candidate that was dispositioned look undecided, and vice versa."""
    cands = tmp_path / "candidates"
    cands.mkdir()
    (cands / "candidate-x-20260909.md").write_text(
        "---\npattern: Bash/logic\nstatus: rejected_unverifiable\n---\n"
        "\nstatus: pending_review\n", encoding="utf-8")

    assert sv.read_candidate(next(cands.glob("*.md")))[:2] == ("Bash/logic", "rejected_unverifiable")


def test_missing_ledger_is_an_empty_answer_not_an_error(tmp_path, capsys):
    """The gate fails open exactly as far as the pre-#530 miner already did, and says so
    as a zero rather than a crash — a nightly job must still finish."""
    cands = tmp_path / "candidates"
    cands.mkdir()
    mt.write_candidate_file(error_pattern(), cands, verdict_store=tmp_path / "nope.jsonl")

    assert sv.main(["check", "--candidates", str(cands),
                    "--store", str(tmp_path / "nope.jsonl")]) == 0
    assert "skipped_by_verdict: 0" in capsys.readouterr().out
