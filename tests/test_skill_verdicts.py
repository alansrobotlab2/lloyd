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


# ── #1131 clause 4: widening a sequence key must orphan no ledger row ────────
#
# #1131 disambiguates a sequence key whose slug reaches `slugify`'s 50-character
# cap. The one way that repair becomes a new defect is by changing a key the
# verdict ledger already adjudicated: the ledger and the corpus join on the key
# string and nothing else, so a widened key reads as a brand-new pattern and a
# recorded rejection silently stops binding. #515's ledger half is the cautionary
# case — widening a key with no migration left 21 coarse rows unreachable. The
# disambiguator is therefore scoped to the cap, and the two tests below are both
# halves of that scoping: the real ledger, whose `seq-*` rows must still resolve
# through the miner's own join, and one written candidate that carries a widened
# key read back by the ledger's own reader.

LIVE_LEDGER = Path.home() / "lloyd" / "_pipeline" / "skills" / "reviews" / "verdicts.jsonl"

# Measured 2026-09-15 on the ledger above: 53 seq-* rows under 28 distinct keys,
# every one of them under 48 characters, so none of them is touched by a
# cap-scoped disambiguator. The minimums below are the item's filing figures
# (43 rows / 23 keys), which the append-only ledger can only exceed — they exist
# so the loop below can never degenerate into checking zero keys.
MIN_UNDERCAP_SEQ_ROWS = 43
MIN_UNDERCAP_SEQ_KEYS = 23
SLUG_CAP = 50


def seq_slug_of(key: str) -> str:
    """The slug part of a `seq-{n}-{slug}` key, the piece the cap applies to."""
    return key[len("seq-"):].partition("-")[2]


def seq_pattern_for_key(key: str) -> dict:
    """The sequence pattern `candidate_pattern_key` derives `key` from.

    A stored seq key is `seq-{n}-{slug}` and a slug is already lowercase
    alphanumerics plus hyphens, so feeding the slug back in as the n-gram string
    re-derives that key — and only re-derives it if the key rule is still
    identity below the cap, which is the whole of clause 4.
    """
    ngram_size, _, slug = key[len("seq-"):].partition("-")
    return {"type": "sequence", "ngram_size": int(ngram_size),
            "sequence_str": slug, "sequence": tuple(slug.split("-")),
            "sessions": {"s1", "s2"}, "has_error_recovery": False,
            "first_seen": "2026-09-01", "last_seen": "2026-09-14",
            "examples": [], "total_calls": 0}


def copy_live_ledger(tmp_path) -> Path:
    """A tmp copy of the live ledger. The rule this file runs under is that it
    never reads the nightly's state through a path it could write, so the real
    rows are copied in: the assertion stays read-only and the data stays real."""
    dest = tmp_path / "reviews" / "verdicts.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(LIVE_LEDGER.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


@pytest.mark.skipif(not LIVE_LEDGER.exists(), reason="no verdict ledger here")
def test_every_stored_sequence_verdict_still_resolves_after_the_widening(tmp_path):
    """No `seq-*` row in the real ledger may stop binding because of #1131.

    For each stored key whose slug is under the cap: the pattern it came from
    re-derives that exact key — an unscoped disambiguator breaks here, which is the
    point — and the miner's join (`verdict_for`) reaches the row that
    `terminal_verdict` reaches by key. Measured today every stored seq-* key is
    under the cap, so this loop is all of them. A key whose slug reached the cap
    could only have been minted after this change, and its original `sequence_str`
    is not recoverable from the row, so it is out of scope here by construction
    rather than by choice.
    """
    store = copy_live_ledger(tmp_path)
    rows = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    seq_keys = sorted({r["pattern_key"] for r in rows
                       if (r.get("pattern_key") or "").startswith("seq-")})
    under_cap = [k for k in seq_keys if len(seq_slug_of(k)) < SLUG_CAP]
    under_cap_rows = [r for r in rows
                      if (r.get("pattern_key") or "").startswith("seq-")
                      and len(seq_slug_of(r["pattern_key"])) < SLUG_CAP]
    assert len(under_cap_rows) >= MIN_UNDERCAP_SEQ_ROWS, (
        f"only {len(under_cap_rows)} under-cap seq-* rows: the ledger this test guards "
        "against orphaning no longer holds the rows it held at filing, so the loop "
        "below would be checking almost nothing")
    assert len(under_cap) >= MIN_UNDERCAP_SEQ_KEYS, (
        f"only {len(under_cap)} under-cap seq-* keys, so the loop below proves nothing")

    table = sv.load_verdicts(store)
    for key in under_cap:
        pattern = seq_pattern_for_key(key)
        assert mt.candidate_pattern_key(pattern) == key, (
            f"{key}: the key rule changed below the cap, so this ledger row is now "
            "unreachable and its recorded verdict silently stops binding")
        row = table.get(key)
        assert row is not None, f"{key} missing from the loaded ledger"
        occ = int(row.get("occurrences_at_decision") or 0)
        pattern["total_calls"] = occ
        joined = mt.verdict_for(pattern, store=store)
        if sv.is_terminal(row):
            assert sv.terminal_verdict(key, store=store, occurrences=occ) is not None
            assert joined is not None, f"{key}: the miner's join missed a terminal row"
            assert joined["verdict"] == row["verdict"]
        else:
            assert joined is None, f"{key}: a non-terminal row now blocks a candidate"


def test_a_widened_sequence_key_survives_the_candidate_round_trip(tmp_path, store):
    """The other half of the join: a candidate written with a disambiguated key is
    read back at that key by `read_candidate`, so a verdict recorded against the
    long key binds the next night's file for the same n-gram."""
    long_seq = {
        "type": "sequence", "ngram_size": 5,
        "sequence_str": ("backlog_write_task → bash:fs → automod_gate_wait "
                         "→ automod_land → backlog_tasks"),
        "sequence": ("backlog_write_task", "bash:fs", "automod_gate_wait",
                     "automod_land", "backlog_tasks"),
        "sessions": {"s1", "s2"}, "has_error_recovery": False,
        "first_seen": "2026-09-14", "last_seen": "2026-09-15",
        "examples": [], "total_calls": 6,
    }
    key = mt.candidate_pattern_key(long_seq)
    assert len(key) > SLUG_CAP, "the fixture must carry a key past the slug cap"

    cands = tmp_path / "candidates"
    path = Path(mt.write_candidate_file(long_seq, cands, verdict_store=store))
    assert sv.read_candidate(path)[:2] == (key, "pending_review")

    sv.record_verdict(store, key, "reviewed_no_skill",
                      reason="covered by an installed skill",
                      evidence_cmd="grep -n 'seq-5' _pipeline/skills/candidates/*.md")
    row = mt.verdict_for(long_seq, store=store)
    assert row is not None and row["verdict"] == "reviewed_no_skill"

    # The next night's file — same n-gram, now written with the verdict already on
    # it — carries the long key in its frontmatter and a superseded status, so the
    # loop cannot re-propose it.
    status, front, _verdict = mt.status_block(long_seq, store=store)
    assert status == "superseded_by_verdict"
    assert "verdict: reviewed_no_skill" in front

    second = Path(mt.write_candidate_file(long_seq, cands, verdict_store=store))
    body = second.read_text(encoding="utf-8")
    assert f"pattern: {key}" in body
    assert f"status: {status}" in body.split("---", 2)[1]
