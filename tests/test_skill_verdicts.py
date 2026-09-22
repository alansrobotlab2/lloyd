"""Tests for the skill-candidate verdict ledger (#530).

The defect these pin, in one sentence: the nightly skill pipeline rejects a pattern,
the miner regenerates it the next night as `status: pending_review`, and the rejection
is re-made by hand. Run #57 on 2026-09-01: "the miner regenerated both files as
`pending_review`, silently wiping the previous review — that's the real process bug
here."

Every assertion *writes* to a tmp ledger and a tmp candidates dir, never to
production state — with one read:
`test_every_stored_sequence_verdict_still_resolves_after_the_widening`
(#1131 clause 4) reads the real append-only ledger
`~/lloyd/_pipeline/skills/reviews/verdicts.jsonl`, copies it into tmp before doing
anything, and asserts against the rows the nightly has actually recorded. A committed
fixture cannot say that a live rejection still binds. The cost of that read is stated
plainly: a nightly that appends a `seq-*` row can now fail this file, and that is the
point — it fails exactly when the miner's key rule stops agreeing with the ledger.
Everywhere else, a nightly rewriting the live corpus cannot fail this file for
somebody else's change.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
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

#: An `evidence_cmd` fixture that observes something: a real grep, printing its count.
#: Since #736 clause 2 `record` refuses a command whose combined output is empty, so the
#: fixtures that used to read `true`, `false` and `grep x` (which blocks on stdin) can no
#: longer be recorded at all. A test that needs a particular exit code, or a particular
#: silence, builds its own command and says why in its docstring.
PRINTING_CMD = f"grep -c '^def ' {_ROOT / 'scripts' / 'skill_verdicts.py'}"


def stored_row(store: Path, pattern_key: str, evidence_cmd: str,
               verdict: str = "reviewed_no_skill", reason: str = "stored grounds",
               occurrences: int = 0) -> dict:
    """Append one ledger line straight to the store, bypassing `record`.

    Only for a row a *reader* has to see that `record` would refuse, or would pay a real
    timeout to write: since #736 clause 2 a command that prints nothing cannot be
    recorded, and rows exactly like that already sit in the live ledger, written before
    the clause landed. The `evidence_cmd_status` tests are about reading a stored command,
    not about the write that now refuses one, so they state their row the way the
    pre-#736 ledger did.
    """
    row = {"pattern_key": pattern_key, "verdict": verdict, "reason": reason,
           "evidence_cmd": evidence_cmd, "occurrences_at_decision": occurrences,
           "decided_at": sv.now_iso(), "decided_by": "test"}
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


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
    """A sequence pattern that reaches a candidate file at all.

    `has_error_recovery` is True because since #1181 `write_candidate_file`
    refuses a sequence flagged False, and every test in this file that uses this
    fixture writes a file to inspect its `status:` — with the flag False the
    writer returns None and each of those assertions dies on `Path(None)` rather
    than testing the verdict plumbing they name. The flag's own gate is pinned in
    `test_trajectory_extraction.py::test_a_sequence_with_no_recovery_in_it_is_not_emittable`."""
    return {
        "type": "sequence",
        "ngram_size": len(sequence),
        "sequence_str": " -> ".join(sequence),
        "sessions": {"s1", "s2"},
        "first_seen": "2026-09-02",
        "last_seen": "2026-09-09",
        "has_error_recovery": True,
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
    """`proposed` is the one disposition that must not end a pattern's life: a patch
    below the auto-apply threshold keeps accumulating evidence until it crosses it
    (`nightly-skill-consolidation` Phase 5.1).

    It used to be asserted beside `consolidated`, which was non-terminal here and
    terminal nowhere — #830 moved `consolidated` into `TERMINAL_VERDICTS`, and the
    paragraph below pins the three values that went with it."""
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
                      reason="owned by file-mutation-safety", evidence_cmd=PRINTING_CMD,
                      occurrences=10)

    lines = store.read_text().strip().splitlines()
    assert len(lines) == 2 == len({json.loads(ln)["pattern_key"] for ln in lines})


def test_latest_line_per_key_wins(store):
    """Append-only with a reopen as its own line — history is never rewritten, so a
    wrong verdict stays auditable."""
    sv.record_verdict(store=store, pattern_key="Bash/logic", verdict="rejected_unverifiable",
                      reason="no error text to ground a skill in", evidence_cmd=PRINTING_CMD)
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
                      reason="owned elsewhere", evidence_cmd=PRINTING_CMD, occurrences=13,
                      decided_at=old)

    assert sv.terminal_verdict("Bash/timeout", store=store, occurrences=13) is None


def test_verdict_reopens_when_the_pattern_grew(store):
    """>10x the occurrences at the decision means the evidence moved; re-ask."""
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="n=1 signature over 3 sessions is not a pattern",
                      evidence_cmd=PRINTING_CMD, occurrences=3)

    assert sv.terminal_verdict("Bash/timeout", store=store, occurrences=30)
    assert sv.terminal_verdict("Bash/timeout", store=store, occurrences=31) is None


def test_unparseable_stamp_does_not_silently_reopen(store):
    """The reopen path is the one that can re-mint a rejected skill, so a verdict whose
    date cannot be read stays binding rather than failing open."""
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="owned elsewhere", evidence_cmd=PRINTING_CMD, occurrences=13,
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
    and 09-09 candidates, keyed on the (tool, error_type) slug.

    The `status: noise` file is harvested too, and that is #830 rather than a slipped
    assertion: the filter below is `is_terminal`, so widening `TERMINAL_VERDICTS` with the
    three runbook-prescribed dispositions is what let the ~125 live keys on `status: noise`
    into the store at all. Before it, `seed` silently skipped every one of them — the
    disposition had nowhere to go, which is the ledger half of the defect this file exists
    to close. `reviewed_no_skill` is still the case that was always harvestable, so both
    halves of the vocabulary are asserted here: the one that bound before #830 and the one
    that only binds now."""
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

    assert sorted(rows) == ["Read/logic", "seq-2-bash-fs-read"], (
        "both statuses this corpus carries are terminal: `reviewed_no_skill` always was, "
        "`noise` since #830 widened `TERMINAL_VERDICTS` with it")
    assert rows["Read/logic"]["verdict"] == "reviewed_no_skill"
    assert "file-path-resolution" in rows["Read/logic"]["reason"]
    assert rows["Read/logic"]["evidence_cmd"].startswith("grep ")
    assert rows["seq-2-bash-fs-read"]["verdict"] == "noise"

    capsys.readouterr()
    sv.main(["seed", "--candidates", str(cands), "--store", str(store)])
    assert "keep: Read/logic already in the ledger" in capsys.readouterr().out
    assert len(store.read_text().strip().splitlines()) == 2, "re-seeding must not re-adjudicate"


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
    """The n-gram part of a `seq-{n}-{slug}` key.

    Not what the cap is measured on — that is the whole key, prefix included — but
    it is the field that distinguishes a legacy truncated key (slug exactly
    `SLUG_CAP`) from one this change disambiguated (slug longer than it)."""
    return key[len("seq-"):].partition("-")[2]


def seq_pattern_for_key(key: str) -> dict:
    """The sequence pattern `candidate_pattern_key` derives `key` from.

    A stored seq key is `seq-{n}-{slug}` and a slug is already lowercase
    alphanumerics plus hyphens, so feeding the slug back in as the n-gram string
    re-derives that key — and only re-derives it if the key rule is still
    identity below the cap, which is the whole of clause 4.
    """
    ngram_size, _, slug = key[len("seq-"):].partition("-")
    # The flag is True so the pattern this builds reaches a file: since #1181 a
    # False-flagged sequence is refused by `write_candidate_file`, and the tests
    # here write the file to read its `pattern:`/`status:` back. The flag plays no
    # part in deriving the key.
    return {"type": "sequence", "ngram_size": int(ngram_size),
            "sequence_str": slug, "sequence": tuple(slug.split("-")),
            "sessions": {"s1", "s2"}, "has_error_recovery": True,
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


def test_every_stored_sequence_verdict_still_resolves_after_the_widening(tmp_path):
    """No `seq-*` row in the real ledger may stop binding because of #1131.

    For each stored key under the cap: the pattern it came from re-derives that
    exact key — an unscoped disambiguator breaks here, which is the point — and the
    miner's join (`verdict_for`) reaches the row that `terminal_verdict` reaches by
    key. "Under the cap" is measured on the whole key, because that is what the rule
    measures: `seq-3-` spends 6 of the 50 characters. Measured 2026-09-15 every one
    of the 53 seq-* rows / 28 distinct keys is under it (longest key 47 characters),
    so the loop below is every row the ledger holds.

    A stored key *at or past* the cap is out of this loop's reach for a mechanical
    reason: the row stores the key, not the n-gram, so `seq_pattern_for_key` cannot
    rebuild the input it was derived from. Two shapes get there and only one is a
    defect — a legacy key whose slug was cut to exactly `SLUG_CAP`, versus a key this
    change disambiguated (50 slug characters plus `-` plus 8 hex). The legacy shape is
    asserted away rather than filtered in silence, because it is precisely the class a
    widened key would orphan without this loop ever noticing.

    The ledger is asserted, never skipped: a guard against orphaning existing rows
    that can pass by not finding the ledger is not that guard. `_pipeline/` is
    gitignored, so this reads the machine's real append-only ledger rather than a
    committed fixture — the same convention as `LIVE_CORPUS` in
    `test_trajectory_extraction.py`.
    """
    assert LIVE_LEDGER.is_file(), f"verdict ledger absent: {LIVE_LEDGER}"
    store = copy_live_ledger(tmp_path)
    rows = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    seq_keys = sorted({r["pattern_key"] for r in rows
                       if (r.get("pattern_key") or "").startswith("seq-")})
    # The rule measures the cap on the whole key, so that is the line between the
    # keys this loop can re-derive and the ones it cannot: at or past 50 characters
    # the key carries a suffix this function cannot undo, because the row stores the
    # key and not the n-gram it was cut from.
    rederivable = [k for k in seq_keys if len(k) < SLUG_CAP]
    # Of the keys past the cap, one shape is a defect and one is expected: a slug cut
    # to exactly SLUG_CAP is the legacy truncation's signature — the class a widened
    # key would silently strand — while a longer slug is a disambiguator this change
    # minted and could not have orphaned. Asserted, never filtered quietly.
    legacy_cut = [k for k in seq_keys if len(seq_slug_of(k)) == SLUG_CAP]
    assert not legacy_cut, (
        f"{len(legacy_cut)} stored seq-* key(s) carry a slug cut to exactly the cap "
        f"({legacy_cut[:3]}): legacy truncated keys, whose original n-gram cannot be "
        "re-derived here, so a widened key would orphan them and they need a "
        "migration rather than a filter")
    under_cap = rederivable
    under_cap_rows = [r for r in rows
                      if (r.get("pattern_key") or "").startswith("seq-")
                      and len(r["pattern_key"]) < SLUG_CAP]
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
        # True for the same reason as `sequence_pattern` above: this test writes
        # the candidate and reads its front matter back, and since #1181 the
        # writer refuses a sequence flagged False. The flag plays no part in
        # deriving or widening the key, which is what this test is about.
        "sessions": {"s1", "s2"}, "has_error_recovery": True,
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


# ── #772: the ledger exists in two trees, and a lost artifact says so ────────
#
# Everything above pins that a verdict binds. This section pins that a verdict
# *survives*, which is a different failure: the ledger lived only under
# `~/lloyd/_pipeline/`, which `.gitignore` excludes from every repo on the box and no
# backup job reads (the 15-minute vault snapshotter covers `~/obsidian` only), so one
# `rm -rf _pipeline` erased 132 lines of decisions — prose reasons that exist nowhere
# else — and `check` then printed `skipped_by_verdict: 0` as though the pipeline had
# never rejected anything. A second, quieter half: a stored `evidence_cmd` invokes
# scripts inside that same tree (20 of the ledger's 81 keys on 2026-09-19), so a wipe
# left the verdict binding while the check meant to overturn it returned "No such file
# or directory" — and nothing printed that, because no reader executed the command.
#
# Every test here redirects the mirror at its own tmp file or unsets it outright; none
# of them may touch the real vault copy at `~/obsidian/memory/skill-verdicts/`.
# `_mirror_target` makes that the default for a scratch `--store` (env unset → no
# mirror), and `test_the_mirror_is_seeded_with_verdicts_recorded_before_it_existed`
# relies on it: the two verdicts it calls "pre-change" are recorded with the env unset,
# which is what makes the seeding branch below the only thing that can move them.

@pytest.fixture
def mirror(tmp_path, monkeypatch) -> Path:
    durable = tmp_path / "vault" / "memory" / "skill-verdicts" / "verdicts.jsonl"
    durable.parent.mkdir(parents=True)
    monkeypatch.setenv("SKILL_VERDICTS_MIRROR", str(durable))
    return durable


def test_record_lands_the_identical_line_in_both_trees(store, mirror, monkeypatch):
    """Clause 1: one `record` call, one decision, two trees, byte-identical lines.

    The mirror path is derived in code from the vault root — `DEFAULT_MIRROR` is
    `Path.home()/"obsidian"/...`, never a function of the store, because a mirror
    computed from `_pipeline` would be erased by the same command that erased the
    ledger. `$SKILL_VERDICTS_MIRROR` is the seam this suite writes through.
    """
    assert sv.DEFAULT_MIRROR.is_relative_to(Path.home() / "obsidian")
    assert "_pipeline" not in str(sv.DEFAULT_MIRROR)
    monkeypatch.delenv("SKILL_VERDICTS_MIRROR")
    assert sv.mirror_path() == sv.DEFAULT_MIRROR
    monkeypatch.setenv("SKILL_VERDICTS_MIRROR", str(mirror))
    assert sv.mirror_path() == mirror

    row = sv.record_verdict(
        store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
        reason="installed skill bash-timeout Pattern 3 cites this exact signature",
        evidence_cmd="grep -c 'command timed out' ~/obsidian/skills/bash-timeout/SKILL.md",
        occurrences=13)

    live, durable = store.read_text().splitlines(), mirror.read_text().splitlines()
    assert len(live) == len(durable) == 1
    assert durable[0] == live[0], "the mirror must not be a re-serialisation"
    assert json.loads(durable[0])["pattern_key"] == row["pattern_key"]


def test_the_mirror_is_seeded_with_verdicts_recorded_before_it_existed(store, tmp_path,
                                                                       monkeypatch):
    """Clause 2: the copy created by the next `record` also holds every prior line.

    The two pre-existing verdicts are recorded with `$SKILL_VERDICTS_MIRROR` **unset** —
    so `_mirror_target` answers None and they exist in the live file alone. That is the
    state every verdict on this box is in today, and it is the only state in which the
    seeding branch of `_seed_mirror` runs: this test deliberately does **not** take the
    `mirror` fixture, because a fixture that sets the env for the whole test lets those
    two records mirror themselves and leaves the seeding body unexercised (review
    finding, 2026-09-19). The durable path is then pointed at a file that does not exist
    until the third `record` creates it. Deleting `_seed_mirror`'s copy body now fails
    the first assertion below; deleting the append fails the second.

    A copy that started empty would be a second file that loses history and makes a
    restore look like it worked, which is the failure #772 was filed for.
    """
    for key in ("Bash/timeout", "Edit/not_found"):
        sv.record_verdict(store=store, pattern_key=key, verdict="reviewed_no_skill",
                          reason="pre-existing decision", evidence_cmd=PRINTING_CMD)
    before = store.read_text()
    assert len(before.splitlines()) == 2, "two verdicts exist before any copy does"

    durable = tmp_path / "vault" / "memory" / "skill-verdicts" / "verdicts.jsonl"
    durable.parent.mkdir(parents=True)  # the vault directory exists; the copy does not
    assert not durable.exists()
    monkeypatch.setenv("SKILL_VERDICTS_MIRROR", str(durable))

    sv.record_verdict(store=store, pattern_key="Bash/logic", verdict="rejected_false_positive",
                      reason="decided after the mirror existed",
                      evidence_cmd=PRINTING_CMD)

    assert durable.read_text().splitlines() == store.read_text().splitlines(), \
        "the new copy must hold the two pre-change verdicts as well as the new one"
    assert len(durable.read_text().splitlines()) == 3
    # Seeding is one-time: a later record appends, it does not re-copy the live file
    # over the copy's own history.
    durable.write_text(durable.read_text() + '{"pattern_key":"only-in-the-mirror"}\n')
    sv.record_verdict(store=store, pattern_key="Read/missing", verdict="reviewed_no_skill",
                      reason="fourth decision", evidence_cmd=PRINTING_CMD)
    assert "only-in-the-mirror" in durable.read_text()
    assert len(durable.read_text().splitlines()) == 5


def test_check_answers_from_the_mirror_and_says_it_did(store, mirror, tmp_path, capsys):
    """Clause 3: a wiped ledger reports itself instead of reading as zero verdicts.

    The counts are asserted equal to the ones the same candidates produce from the live
    ledger a moment earlier — the acceptance asks for the *same* `checked:` and
    `skipped_by_verdict:` after the live file is deleted, and an equality between two
    runs is the only form of that claim a test can check without pinning a number the
    nightly moves under it. The verdict is recorded through the real route, not the
    `seeded` fixture, so the mirror holds it too: `seeded` fires while the mirror env is
    still unset and would leave the copy empty by the guard in `_mirror_target`.
    """
    sv.record_verdict(
        store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
        reason="installed skill bash-timeout Pattern 3 cites this exact signature",
        evidence_cmd="grep -c 'command timed out' ~/obsidian/skills/bash-timeout/SKILL.md",
        occurrences=13)
    cands = tmp_path / "candidates"
    cands.mkdir()
    mt.write_candidate_file(error_pattern(), cands, verdict_store=store)

    capsys.readouterr()
    assert sv.main(["check", "--candidates", str(cands), "--store", str(store)]) == 0
    with_live = capsys.readouterr().out.splitlines()[-1]
    assert with_live == "checked: 1  skipped_by_verdict: 1"

    store.unlink()
    # #736 clause 4 rides along on this route: the corpus holds a key only a ledger can
    # mint (`superseded_by_verdict`, written because a verdict blocked it), so the absent
    # store is reported as an incident and fails — even though the mirror-rescued counts
    # are still what the runbook parses off the last line.
    assert sv.main(["check", "--candidates", str(cands), "--store", str(store)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == with_live, "the mirror must answer the same counts, not zero"
    naming = [ln for ln in lines if ln.startswith("verdict source:")]
    assert naming == [f"verdict source: {mirror} (live ledger {store} is absent)"]
    assert any(ln.startswith("LEDGER_ABSENT") for ln in lines), lines


def test_check_reports_a_verdict_whose_check_can_no_longer_run(tmp_path, store, mirror, capsys):
    """Clause 4: an unexecutable falsifier is an incident; rc 0 and rc 1 are results.

    Four keys, four outcomes. The echoing key that exits 1 is the falsifier *doing its
    job* — reporting that the verdict's grounds no longer hold — and printing a warning
    for it would train the nightly to ignore the warning. (Both commands print, since
    #736 clause 2 refuses to record one that does not.) The absent-script key is the #772
    fail-shut case: the verdict keeps suppressing candidates while nothing can overturn
    it, which nothing could see before because `scan_candidates` never executed these
    commands. The fourth is not hypothetical: the live ledger's `seq-2-read-edit` command
    exits 2 with bash's own `unexpected EOF while looking for matching '"'` (measured
    2026-09-19), a stored string that lost a quote, so exit 127 alone would have left the
    one falsifier actually broken on this box unreported.
    """
    cands = tmp_path / "candidates"
    cands.mkdir()
    for key, cmd in (("Bash/timeout", "echo 'bash-timeout owns this signature'"),
                     ("Edit/not_found", "echo 'the grounds are gone'; exit 1"),
                     ("Bash/logic", f"{tmp_path}/gone/falsifier.sh"),
                     ("Write/logic", 'echo "unbalanced')):
        sv.record_verdict(store=store, pattern_key=key, verdict="reviewed_no_skill",
                          reason=f"grounds for {key}", evidence_cmd=cmd)
        mt.write_candidate_file(error_pattern(tool=key.split("/")[0],
                                              error_type=key.split("/")[1]),
                                cands, verdict_store=store)

    assert sv.main(["check", "--candidates", str(cands), "--store", str(store)]) == 0
    out = capsys.readouterr().out
    unrunnable = [ln for ln in out.splitlines() if ln.startswith("EVIDENCE_CMD_UNRUNNABLE")]
    assert [ln.split(" ::")[0] for ln in unrunnable] == [
        "EVIDENCE_CMD_UNRUNNABLE Bash/logic", "EVIDENCE_CMD_UNRUNNABLE Write/logic"], out
    assert "No such file or directory" in unrunnable[0]
    assert "unexpected EOF" in unrunnable[1]
    assert "skipped_by_verdict: 4" in out, "a broken falsifier must not silently un-block"


def test_a_hung_falsifier_is_reported_too(tmp_path, store, mirror):
    """The bound in clause 4 is only worth having if firing it is an answer, not a hang.

    `EVIDENCE_TIMEOUT_SECONDS` is 15 against a live ledger whose 75 blocking commands
    re-execute in 2.0 s together, so this fires on a wedged grep, not on a slow pipeline.
    """
    # Stated as a stored row, not through `record`: #736 clause 2 refuses a command that
    # prints nothing, and `record` would itself block the full 5 s on a hanging one. A
    # wedged falsifier is a condition of a *stored* line — including the ones the live
    # ledger holds from before the clause.
    stored_row(store, "Bash/sleepy", "sleep 5",
               reason="grounds that cannot be re-checked while the box waits")
    rc, detail = sv.evidence_cmd_status(sv.load_verdicts(store)["Bash/sleepy"], timeout=1)
    assert rc == sv.UNRUNNABLE
    assert "still running after 1s" in detail


def test_a_recorded_check_script_is_copied_into_the_mirror(tmp_path, store, mirror):
    """Clause 5: re-executability outlives `_pipeline` too, not just the ledger.

    Both halves run against a real `git init` repository under tmp, not a directory that
    merely looks like one. `_git_tracked` shells out to `git ls-files --error-unmatch`,
    and in a `.git`-less tree that guard is inert — "not a repository" and "not tracked"
    are the same non-zero exit — so the negative assertion at the end would pass whether
    or not tracking had ever been consulted (review finding, 2026-09-19). Here the two
    scripts are siblings in one repo, and the only difference between them is that
    `git add` named one of them.

    The copy is byte-identical under its own basename and the stored command is *not*
    rewritten, so the mirror stays a restore source and never becomes a shadow a later
    run mistakes for the live falsifier. A script git already tracks is deliberately not
    copied: a second copy of a tracked module in the vault is a fork waiting to happen.
    """
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    untracked = repo / "tools" / "seq_falsifier.py"
    untracked.write_text("print('seq falsifier v1')\n", encoding="utf-8")
    tracked = repo / "tools" / "tracked_falsifier.py"
    tracked.write_text("print('tracked falsifier')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "tools/tracked_falsifier.py"], check=True)
    assert subprocess.run(["git", "-C", str(tracked.parent), "ls-files", "--error-unmatch",
                           tracked.name], capture_output=True).returncode == 0, \
        "control: git really tracks this script, so the skip below has something to bite on"

    sv.record_verdict(store=store, pattern_key="seq-3-bash-fs-bash-other",
                      verdict="reviewed_no_skill", reason="grounds",
                      evidence_cmd=f"python3 {untracked}")

    copied = mirror.parent / untracked.name
    assert copied.is_file()
    assert copied.read_bytes() == untracked.read_bytes()
    assert json.loads(mirror.read_text().splitlines()[-1])["evidence_cmd"] == f"python3 {untracked}"

    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="grounds checked by a tracked module",
                      evidence_cmd=f"python3 {tracked}")
    assert not (mirror.parent / tracked.name).exists(), \
        "a tracked script is durable already; a vault copy of it is a fork"


def test_the_stored_check_is_run_by_a_real_child_process(tmp_path, store, mirror):
    """Seam, not clause: ledger JSON → `bash -c` → exit code, measured on a child.

    `evidence_cmd` is a string read out of a JSONL line and handed to
    `subprocess.run(["bash", "-c", cmd])`, so the value under test only exists on the
    far side of a process boundary. The two tests that read this seam via `sv.main`
    assert on the *report*; this one asserts the child itself ran, three ways:

    * the falsifier writes a file, so execution has an effect no in-process mock fakes;
    * what it writes is its own pid, bash's `$$` expanded in the child and nowhere else,
      and that pid is not this process's — so the stored string really became a shell;
    * the same command, re-read from the ledger JSON and executed again after the script
      file was deleted, is `UNRUNNABLE` — the clause-4 predicate decided by the OS
      against a path, not by a stubbed return code.
    """
    marker = tmp_path / "child-pid"
    script = tmp_path / "falsifier.sh"
    # It prints as well as writes: since #736 clause 2 a falsifier whose output is empty
    # cannot be recorded at all, and writing a marker file is not output.
    script.write_text(f'echo "$$" > {marker}; echo "child $$ wrote {marker}"\n',
                      encoding="utf-8")
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="grounds with a real re-executable check",
                      evidence_cmd=f"bash {script}")

    row = sv.load_verdicts(store)["Bash/timeout"]
    assert row["evidence_cmd"] == f"bash {script}", "the seam's input is the stored string"
    rc, detail = sv.evidence_cmd_status(row)
    assert (rc, detail) == (0, ""), f"a runnable falsifier must return its own exit code: {detail}"
    child_pid = int(marker.read_text().strip())
    assert child_pid > 0 and child_pid != os.getpid(), \
        f"the stored string was not executed by a separate shell: pid {child_pid}"

    script.unlink()
    rc, detail = sv.evidence_cmd_status(sv.load_verdicts(store)["Bash/timeout"])
    assert rc == sv.UNRUNNABLE
    assert "No such file or directory" in detail

    # And the same child executes against the caller-supplied bound, in a process whose
    # wall clock the test cannot see: the bound must stop the child, not merely describe it.
    # Stated as a stored row rather than recorded: `record` would itself block the whole
    # 5 s executing it, and #736 clause 2 refuses a command that has printed nothing by the
    # time it is refused. Reading a stored bound is the half under test.
    slow = tmp_path / "slow.sh"
    slow.write_text('sleep 5\n', encoding="utf-8")
    stored_row(store, "Edit/not_found", f"bash {slow}",
               reason="grounds whose check cannot finish")
    started = time.monotonic()
    rc, detail = sv.evidence_cmd_status(sv.load_verdicts(store)["Edit/not_found"], timeout=1)
    elapsed = time.monotonic() - started
    assert rc == sv.UNRUNNABLE and "still running after 1s" in detail
    assert elapsed < 4, f"the bound was not enforced, it was observed: {elapsed:.1f}s"


def test_the_shipped_cli_writes_both_trees_and_answers_from_a_child(tmp_path, store, mirror):
    """Seam, not clause: the nightly runs this as a *program*, not as an imported function.

    `nightly-skill-consolidation` Phase 0 invokes `skill_verdicts.py check`, and Phase 5
    invokes `record`, as subprocesses. So the ledger file, the env-derived mirror path
    and the printed counts cross an interpreter boundary, and nothing about the durable
    half has ever been exercised that way — an in-process `sv.main` call proves the
    function, not the shipped CLI. This spawns the real module out of this checkout four
    times with `$SKILL_VERDICTS_MIRROR` aimed at the tmp copy: one `record` (which must
    leave the identical line in both trees from the child), one `record` that must be
    refused and leave both trees untouched (#736 clause 2's exit status is only a
    refusal if it survives `sys.exit`), one `check` (whose last stdout line is the count
    line the runbook parses), and one more `check` after the live ledger is deleted under
    a fresh child process.
    """
    cands = tmp_path / "candidates"
    cands.mkdir()
    durable = mirror
    env = dict(os.environ, SKILL_VERDICTS_MIRROR=str(durable))
    run = lambda *args: subprocess.run(
        [sys.executable, "-m", "scripts.skill_verdicts", *args],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=120)

    proc = run("record", "--pattern", "Bash/timeout", "--verdict", "reviewed_no_skill",
               "--reason", "installed skill bash-timeout Pattern 3 cites this exact signature",
               "--evidence-cmd", PRINTING_CMD, "--occurrences", "13",
               "--store", str(store))
    assert proc.returncode == 0, proc.stderr
    assert len(store.read_text().splitlines()) == 1, "the child wrote the live ledger"
    assert durable.read_text().splitlines() == store.read_text().splitlines(), \
        "a child-process `record` must leave the identical line in both trees"
    assert json.loads(store.read_text())["evidence_observed"], \
        "the measurement the decision rested on belongs in the row"

    refused = run("record", "--pattern", "Edit/not_found", "--verdict", "reviewed_no_skill",
                  "--reason", "grounds a silent command cannot support",
                  "--evidence-cmd", "true", "--store", str(store))
    assert refused.returncode != 0, "a vacuous falsifier must fail the shipped CLI too"
    assert "observed nothing" in refused.stderr and "true" in refused.stderr, refused.stderr
    assert len(store.read_text().splitlines()) == 1, "a refusal writes no line"
    assert durable.read_text().splitlines() == store.read_text().splitlines(), \
        "a refusal must not reach the durable copy either"

    mt.write_candidate_file(error_pattern(), cands, verdict_store=store)
    proc = run("check", "--candidates", str(cands), "--store", str(store))
    assert proc.returncode == 0, proc.stderr
    with_live = proc.stdout.strip().splitlines()[-1]
    assert with_live == "checked: 1  skipped_by_verdict: 1", proc.stdout

    store.unlink()
    proc = run("check", "--candidates", str(cands), "--store", str(store))
    # Non-zero: the corpus still holds `superseded_by_verdict`, a status only a ledger
    # mints, so the missing store is an incident (#736 clause 4) even with the copy in place.
    assert proc.returncode == 1, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[-1] == with_live, "a second child process must answer from the copy alone"
    assert f"verdict source: {durable} (live ledger {store} is absent)" in "\n".join(lines[:-1])
    assert any(ln.startswith("LEDGER_ABSENT") for ln in lines), proc.stdout


def test_check_names_a_verdict_the_two_trees_do_not_agree_on(tmp_path, store, mirror, capsys):
    """Clause 2's invariant, enforced on every read instead of at one `record`.

    Seeding and appending make the copy a *superset*; neither keeps the live file honest,
    and `resolve_verdict_source` falls back only when the live file is *entirely* absent —
    so a live ledger that dropped one of its 132 lines, in a file whose only documented
    writer appends, would print exactly the counts two healthy trees print. `check`
    therefore compares the two latest-per-key tables and names what disagrees. Three
    incidents, three lines: the copy never receiving a verdict, the live file losing one,
    and the two sides holding different decisions for one key. One shared silent symptom is
    why all three get a line, and none of them has happened on this box yet — as of
    2026-09-19T11:52Z both trees hold 132 lines and `cmp` reports them identical, which is
    the state `test_agreeing_trees_and_a_scratch_ledger_print_no_divergence` pins as silent.
    """
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="decided in both trees", evidence_cmd=PRINTING_CMD)
    sv.record_verdict(store=store, pattern_key="Edit/not_found", verdict="reviewed_no_skill",
                      reason="decided before the copy lagged", evidence_cmd=PRINTING_CMD)
    # A third verdict appended straight to the live file, bypassing `record`: the nightly
    # running pre-change code, `skill_verdicts.py seed`, or a hand `>>`.
    with store.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pattern_key": "tool_calls/truncated_args",
                             "verdict": "reviewed_no_skill", "reason": "appended by a writer",
                             "decided_at": "2026-09-19T11:00:00Z"}) + "\n")
    cands = tmp_path / "candidates"
    cands.mkdir()

    assert sv.main(["check", "--candidates", str(cands), "--store", str(store)]) == 0
    lines = capsys.readouterr().out.splitlines()
    missing = [ln for ln in lines if ln.startswith("MIRROR_MISSING")]
    assert missing == [
        "MIRROR_MISSING tool_calls/truncated_args :: reviewed_no_skill decided "
        f"2026-09-19T11:00:00Z is in the live ledger but not in the durable copy {mirror}"], lines
    assert not [ln for ln in lines if ln.startswith(("LEDGER_LOST", "LEDGER_MIRROR_CONFLICT"))]

    # The other direction: the mirror still holds a verdict the live file has lost.
    store.write_text("".join(
        ln for ln in store.read_text().splitlines(keepends=True)
        if "Edit/not_found" not in ln), encoding="utf-8")
    assert sv.main(["check", "--candidates", str(cands), "--store", str(store)]) == 0
    lost = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("LEDGER_LOST")]
    assert len(lost) == 1 and lost[0].startswith("LEDGER_LOST Edit/not_found ::"), lost
    assert "but the live ledger no longer holds it" in lost[0]

    # And disagreement about one key, from a reopen one tree saw and the other did not.
    with store.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pattern_key": "Bash/timeout", "verdict": "noise",
                             "reason": "reopened in the live ledger only",
                             "decided_at": "2026-09-19T12:00:00Z"}) + "\n")
    assert sv.main(["check", "--candidates", str(cands), "--store", str(store)]) == 0
    final = capsys.readouterr().out.splitlines()
    conflict = [ln for ln in final if ln.startswith("LEDGER_MIRROR_CONFLICT")]
    assert conflict == [
        f"LEDGER_MIRROR_CONFLICT Bash/timeout :: live=noise "
        f"durable=reviewed_no_skill decided 2026-09-19T12:00:00Z/"
        f"{sv.load_verdicts(mirror)['Bash/timeout']['decided_at']}"], conflict
    assert final[-1] == "checked: 0  skipped_by_verdict: 0", \
        "the count line stays last: the nightly parses splitlines()[-1]"


def test_agreeing_trees_and_a_scratch_ledger_print_no_divergence(tmp_path, store, mirror,
                                                                 monkeypatch, capsys):
    """The divergence line must be able to stay silent, or the nightly stops reading it.

    Two trees holding the same verdicts print nothing — that is the healthy case, and a
    warning that fires on it joins the noise that trains a run to ignore warnings. A
    scratch run with no mirror prints nothing either, which is the state the nightly is in
    whenever `$SKILL_VERDICTS_MIRROR` is unset: `_mirror_target` then refuses an
    off-default `--store` so a scratch ledger cannot append scratch verdicts into the real
    vault copy, and reporting that deliberate guard as a divergence would turn every
    scratch run into a false alarm.
    """
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="decided in both trees", evidence_cmd=PRINTING_CMD)
    capsys.readouterr()
    assert sv.main(["check", "--candidates", str(tmp_path), "--store", str(store)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert not [ln for ln in out if ln.startswith(("MIRROR_MISSING", "LEDGER_LOST",
                                                   "LEDGER_MIRROR_CONFLICT"))], out
    assert out[-1] == "checked: 0  skipped_by_verdict: 0"

    unmirrored = tmp_path / "scratch.jsonl"
    monkeypatch.delenv("SKILL_VERDICTS_MIRROR")
    assert sv._mirror_target(unmirrored) is None, \
        "an off-default ledger with no mirror env gets no durable copy"
    assert sv._mirror_target(sv.DEFAULT_STORE) == sv.DEFAULT_MIRROR
    sv.record_verdict(store=unmirrored, pattern_key="Write/logic", verdict="reviewed_no_skill",
                      reason="scratch decision", evidence_cmd=PRINTING_CMD)
    capsys.readouterr()
    assert sv.main(["check", "--candidates", str(tmp_path), "--store", str(unmirrored)]) == 0
    silent = capsys.readouterr().out.splitlines()
    assert not [ln for ln in silent if ln.startswith(("MIRROR_MISSING", "LEDGER_LOST",
                                                      "LEDGER_MIRROR_CONFLICT"))], silent
    assert silent[-1] == "checked: 0  skipped_by_verdict: 0"


# ── the mining gate must not promote a bare non-zero exit (#500) ─────────────
#
# `stats.is_error` in the session store fires on any non-zero shell exit, so a
# `grep` that found nothing and a command that actually broke arrive identical.
# Keying the gate on `error_source` made it a no-op — over `_pipeline/trajectories/*.jsonl`
# 2026-09-09→09-17, 2,330 error steps carried {protocol: 2327, exit_code: 3} and
# `is_corroborated_error` admitted all 2,330. These pin the gate on
# `failure_class` instead, which the extractor derives from payload shape.

et500 = _load("extract_trajectories_500", "scripts/extract-trajectories.py")


def flagged_step(**fields):
    """An `error_tools[]` row as the extractor writes it: by construction an
    error, so it carries no `is_error` key."""
    step = {"name": "Bash", "sequence": 0, "error_type": "logic",
            "error_source": "protocol", "params_summary": {"command": "true"}}
    step.update(fields)
    return step


def test_the_gate_rejects_a_bare_nonzero_exit_but_keeps_signal_and_structured():
    """Clause 3. The three shapes that decide the gate, as real payloads.

    The `nonzero_exit` row is the one class that must stop reaching skill
    authoring: a `grep` with no match exits 1 and an intentionally failing
    `pytest` exits 1, and 1,732 flagged messages in an 8-day read of
    `~/lloyd/sessions/*.json` were that class. The -15 row is a SIGTERM — the
    signature of a real Bash timeout — and the structured row is a tool saying
    it failed; both must stay promotable.
    """
    assert mt.is_corroborated_error(flagged_step(failure_class="nonzero_exit",
                                                 exit_code=1)) is False
    assert mt.is_corroborated_error(flagged_step(failure_class="timeout_or_signal",
                                                 exit_code=-15)) is True
    assert mt.is_corroborated_error(flagged_step(
        failure_class="structured_error", exit_code=None,
        params_summary={"query": "x"})) is True
    assert mt.is_corroborated_error(flagged_step(failure_class="harness_block")) is True
    assert mt.is_corroborated_error(flagged_step(failure_class="protocol_flagged")) is True


def test_the_gate_still_falls_back_for_rows_written_before_the_class():
    """Pre-#500 trajectory rows have no `failure_class`, and extraction is
    incremental, so a gate that required the field would blank the 7-day window
    the night before any backfill. The `error_source` reading survives as the
    fallback; a row with no class and no source still needs an explicit exit code.
    """
    assert mt.is_corroborated_error(flagged_step()) is True          # protocol, legacy
    assert mt.is_corroborated_error({**flagged_step(), "error_source": "semantic"}) is False
    no_source = {k: v for k, v in flagged_step().items() if k != "error_source"}
    assert mt.is_corroborated_error({**no_source, "exit_code": 2}) is True
    assert mt.is_corroborated_error({**no_source, "exit_code": None}) is False
    assert mt.is_corroborated_error({"name": "Bash", "is_error": False,
                                     "error_source": "protocol"}) is False


MIXED_CALLS = [
    # (tool, args, result, stats.is_error)
    ("Bash", {"command": "grep -rn failure_class scripts/"},
     "no matches found\n\n[exit code: 1]", True),
    ("Bash", {"command": "sleep 900"},
     "Command timed out after 120000ms\n\n[exit code: -15]", True),
    ("mcp____tools_email_search", {"query": "invoice"},
     '{"error": "goal is required"}', True),
]


def mixed_session_rows(tmp_path, n=2):
    """N sessions with the same three flagged steps, run through the real
    extractor so the miner sees written rows, not hand-made dicts. The threshold
    is distinct sessions, so one session would emit nothing for any class."""
    rows = []
    for i in range(n):
        path = tmp_path / f"sess-mixed-{i}.json"
        messages = []
        for j, (tool, args, result, is_err) in enumerate(MIXED_CALLS):
            messages.append({"role": "assistant", "tool_calls": [
                {"id": f"call_{j}", "function": {"name": tool,
                                                 "arguments": json.dumps(args)}}]})
            messages.append({"role": "tool", "tool_call_id": f"call_{j}",
                             "content": [{"type": "text", "text": result}],
                             "stats": {"result_chars": len(result),
                                       "is_error": is_err}})
        path.write_text(json.dumps({"session_id": f"sess-mixed-{i}",
                                    "messages": messages}), encoding="utf-8")
        row = et500.parse_session(path)
        assert row is not None
        rows.append(row)
    return rows


def test_the_miner_emits_a_candidate_only_for_the_two_real_failures(tmp_path):
    """Clause 4, end to end: extractor writes the class, miner gates on it.

    Three flagged steps in the window — a `grep` no-match (exit 1), a
    SIGTERM-killed `sleep` (exit -15), a structured tool error — and exactly two
    candidates come out. Every example in every emitted candidate carries a
    promotable class, and no example carries `nonzero_exit`: a candidate whose
    examples are all bare exits is the artifact this item exists to stop.
    """
    rows = mixed_session_rows(tmp_path)
    patterns = mt.mine_error_patterns(rows, threshold=2)

    assert len(patterns) == 2, [p["tool_name"] for p in patterns]
    promoted = {(p["tool_name"], p["error_type"]) for p in patterns}
    assert promoted == {("Bash", "timeout"),
                        ("mcp____tools_email_search", "logic")}, promoted
    assert not any("grep" in str(p["examples"][0]["params_summary"]) for p in patterns)

    classes = {ex["failure_class"] for p in patterns for ex in p["examples"]}
    assert classes == {"timeout_or_signal", "structured_error"}, classes


def test_the_emitted_candidate_files_carry_no_nonzero_exit_example(tmp_path):
    """The same window through `write_candidate_file`, which is what the nightly
    actually leaves on disk for a human to read. Two files, and the string
    `nonzero_exit` appears in neither."""
    rows = mixed_session_rows(tmp_path)
    patterns = mt.mine_error_patterns(rows, threshold=2)
    out = tmp_path / "candidates"
    paths = [Path(mt.write_candidate_file(p, out, verdict_store=tmp_path / "v.jsonl"))
             for p in patterns]
    assert len(paths) == 2
    names = sorted(p.name for p in paths)
    assert not [n for n in names if "grep" in n], names
    for p in paths:
        text = p.read_text(encoding="utf-8")
        assert "nonzero_exit" not in text, p.name
        assert "timeout_or_signal" in text or "structured_error" in text, p.name


# ── #736: a new line may not weaken the row it supersedes ────────────────────

def test_a_correction_without_a_new_count_keeps_the_reopen_baseline(store):
    """Clause 1: an omitted `--occurrences` carries the superseded row's count forward.

    The failure this pins happened on 2026-09-15. A wrong phrase in a prior line has to
    be superseded by an appended correction — never edited — and the corrections that
    night omitted `--occurrences`, so each stored `occurrences_at_decision: 0` through
    `int(occurrences or 0)`. `reopen_reason`'s growth guard is
    `if baseline > 0 and occurrences > baseline * 10`, so a zero baseline turns growth
    reopen off permanently for that key: 10 live keys, including one whose decision count
    was 367, went on binding their full 60 days however far the pattern grew. The act of
    correcting the ledger silently disarmed the cap that exists to stop a stale verdict
    burying real work — in the one file whose purpose is that a later run can check it.
    """
    key = "seq-2-bash-explore-bash-fs"
    sv.record_verdict(store=store, pattern_key=key, verdict="reviewed_no_skill",
                      reason="names the wrong owning skill",
                      evidence_cmd="echo '367 occurrences over 41 sessions'",
                      occurrences=367)
    sv.record_verdict(store=store, pattern_key=key, verdict="reviewed_no_skill",
                      reason="correction: the owning skill is bash-fs, not bash-explore",
                      evidence_cmd="echo '367 occurrences over 41 sessions'",
                      decided_by="self-correction")

    row = sv.load_verdicts(store)[key]
    assert row["decided_by"] == "self-correction", "the correction is still the latest line"
    assert row["occurrences_at_decision"] == 367, \
        "an omitted count is carried forward, never reset to 0"
    # And the cap it protects fires again: past 10x the carried baseline the verdict
    # reopens. With a zero baseline both of these would report the verdict still binding.
    assert sv.terminal_verdict(key, store=store, occurrences=3670), "3670 is not yet >10x"
    assert sv.terminal_verdict(key, store=store, occurrences=3671) is None, \
        "3671 is past 10x of the carried baseline, so the verdict must reopen"


def test_an_explicit_zero_count_is_still_stored_as_zero(store):
    """The counterpart to the carry-forward: `None` means 'not named', `0` means 0.

    `cmd_seed` passes 0 deliberately for a merged candidate (#515), whose `occurrences:`
    sums every signature bucket behind the key and so cannot be compared with a baseline
    recorded from one bucket. Rewriting that 0 into a carried-forward count would put two
    units on the same axis and re-arm a growth ratio that measures nothing.
    """
    key = "Bash/network"
    sv.record_verdict(store=store, pattern_key=key, verdict="reviewed_no_skill",
                      reason="one signature's count, recorded first",
                      evidence_cmd="echo '90 over 6 sessions'", occurrences=90)
    sv.record_verdict(store=store, pattern_key=key, verdict="reviewed_no_skill",
                      reason="re-seeded as a merged unit; units not comparable",
                      evidence_cmd=PRINTING_CMD,
                      occurrences=0)
    assert sv.load_verdicts(store)[key]["occurrences_at_decision"] == 0


def test_the_cli_carries_the_count_forward_when_the_flag_is_omitted(store, capsys):
    """The same clause across the surface the runbook actually uses.

    `--occurrences` had an argparse default of 0, so an omitted flag was indistinguishable
    from `--occurrences 0` before the value ever reached `record_verdict` — a fix in the
    function alone would leave the CLI writing zeros. This calls `main`, so the default is
    part of what is under test.
    """
    base = ["record", "--pattern", "Bash/timeout", "--verdict", "reviewed_no_skill",
            "--evidence-cmd", "echo '13 over 3 sessions'", "--store", str(store)]
    assert sv.main([*base, "--reason", "installed skill bash-timeout owns this signature",
                    "--occurrences", "13"]) == 0
    assert sv.main([*base, "--reason", "correction: it is Pattern 3, not Pattern 4",
                    "--decided-by", "self-correction"]) == 0
    capsys.readouterr()
    assert sv.load_verdicts(store)["Bash/timeout"]["occurrences_at_decision"] == 13


def test_a_check_that_observes_nothing_is_refused_and_writes_no_line(store, mirror, capsys):
    """Clause 2: `--evidence-cmd` enforced presence, and presence was never the point.

    #530's stated purpose is that a re-executable check lets a later run *falsify* a
    verdict; a command that runs successfully and prints nothing satisfies "re-executable"
    while falsifying nothing. Measured on the live ledger 2026-09-13: 12 of its 41 live
    keys printed nothing at all — `test $(grep -c …) -eq 0`, silent on success, and
    `grep -rl` for a string the target file does not contain. The symptom was a
    `skipped_by_verdict: 0` line that read as a quiet night.
    """
    for cmd in ("true",
                "false",
                "test $(grep -c 'never-written-token' /etc/hostname) -eq 0"):
        rc = sv.main(["record", "--pattern", "Bash/timeout", "--verdict", "reviewed_no_skill",
                      "--reason", "grounds a silent command cannot support",
                      "--evidence-cmd", cmd, "--occurrences", "5", "--store", str(store)])
        err = capsys.readouterr().err
        assert rc != 0, f"a check that prints nothing must not be recordable: {cmd}"
        assert "observed nothing" in err, err
        assert cmd in err, f"the refusal has to name the command it refused: {err}"
        assert not store.exists(), "a refusal writes no line to either tree"
        assert not mirror.exists()

    # The same call with a command that prints a measurement is accepted.
    assert sv.main(["record", "--pattern", "Bash/timeout", "--verdict", "reviewed_no_skill",
                    "--reason", "installed skill bash-timeout owns this signature",
                    "--evidence-cmd", "echo '13 occurrences over 3 sessions'",
                    "--occurrences", "13", "--store", str(store)]) == 0
    assert len(store.read_text().splitlines()) == 1


def test_a_check_that_times_out_observes_nothing_either(store, capsys):
    """The bound has to be a refusal, not a pass: a command still running when the
    15 s cap fires has printed nothing, which is the same observation gap clause 2
    refuses — and `record` runs synchronously inside a nightly turn, so it may not hang."""
    started = time.monotonic()
    rc = sv.main(["record", "--pattern", "Bash/timeout", "--verdict", "reviewed_no_skill",
                  "--reason", "grounds a wedged command cannot support",
                  "--evidence-cmd", "sleep 30", "--store", str(store)])
    elapsed = time.monotonic() - started
    err = capsys.readouterr().err
    assert rc != 0 and "observed nothing" in err, err
    assert elapsed < sv.EVIDENCE_TIMEOUT_SECONDS + 10, \
        f"record waited out the bound rather than bounding it: {elapsed:.1f}s"
    assert not store.exists()


def test_an_accepted_row_stores_what_the_check_printed(tmp_path, store):
    """Clause 3: the decision carries the measurement it was made on.

    A later run sees `evidence_observed` beside the `evidence_cmd` that re-makes it, so a
    falsifier whose *value* has since moved is visible as a disagreement rather than as a
    green check. Only the first line is kept, and truncated: the ledger is a JSONL of
    decisions, not a log.
    """
    script = tmp_path / "seq_falsifier.py"
    script.write_text("print('sess=7 steps_ok=4 steps_err=3 has_error_recovery=True')\n"
                      "print('this line is not quoted')\n", encoding="utf-8")
    assert sv.main(["record", "--pattern", "seq-2-read-write", "--verdict", "reviewed_no_skill",
                    "--reason", "3 of 7 steps have no recovery, under the threshold",
                    "--evidence-cmd", f"{sys.executable} {script}",
                    "--occurrences", "275", "--store", str(store)]) == 0
    row = sv.load_verdicts(store)["seq-2-read-write"]
    assert row["evidence_observed"] == "sess=7 steps_ok=4 steps_err=3 has_error_recovery=True"
    assert "not quoted" not in row["evidence_observed"], "the first line only"
    assert row["evidence_cmd"] == f"{sys.executable} {script}", "the command is unchanged"

    # A command reporting through stderr is stored with what it printed, not with nothing.
    assert sv.main(["record", "--pattern", "Write/logic", "--verdict", "reviewed_no_skill",
                    "--reason", "the check itself fails on this box, and says so",
                    "--evidence-cmd", "grep -c 'x' /nope/nothing-here",
                    "--occurrences", "4", "--store", str(store)]) == 0
    assert "No such file" in sv.load_verdicts(store)["Write/logic"]["evidence_observed"]

    # A long first line is a quotation, not an attachment.
    long_line = "M" * (sv.EVIDENCE_OBSERVED_MAX + 50)
    assert sv.main(["record", "--pattern", "Bash/logic", "--verdict", "reviewed_no_skill",
                    "--reason", "a falsifier that prints a wall of text",
                    "--evidence-cmd", f"printf '{long_line}\\n'",
                    "--occurrences", "2", "--store", str(store)]) == 0
    observed = sv.load_verdicts(store)["Bash/logic"]["evidence_observed"]
    assert len(observed) == sv.EVIDENCE_OBSERVED_MAX + 1, \
        f"expected the cap plus an ellipsis, got {len(observed)}"


def test_a_missing_ledger_is_an_alarm_when_the_corpus_still_carries_its_verdicts(tmp_path,
                                                                                capsys):
    """Clause 4: a wiped ledger and a fresh install used to print the same quiet night.

    `load_verdicts` returns an empty dict for a missing store, which is right for the
    empty case and indistinguishable from the deleted one — so an empty ledger answers
    "no verdicts" for every key and the consolidator and the miner both resume proposing
    content rejected months ago, with `skipped_by_verdict: 0` as the only symptom. The
    2026-08-22 entity-graph incident is the precedent: a nightly writer deleted
    `_pipeline/memory-graph/` and `_pipeline/` is gitignored, so nothing was recoverable.
    """
    cands = tmp_path / "candidates"
    cands.mkdir()
    raw_candidate(cands, "candidate-bash-timeout-20260922.md", "reviewed_no_skill")
    capsys.readouterr()

    rc = sv.main(["check", "--candidates", str(cands),
                  "--store", str(tmp_path / "gone" / "verdicts.jsonl")])
    out = capsys.readouterr().out
    assert rc != 0, "a missing ledger beside adjudicated candidates must fail, not shrug"
    assert any(ln.startswith("LEDGER_ABSENT") for ln in out.splitlines()), out
    assert "reviewed_no_skill" in out, "the alarm names the statuses that prove a ledger existed"
    assert out.splitlines()[-1] == "checked: 1  skipped_by_verdict: 0", \
        "the counts stay the last line: the runbook parses them off splitlines()[-1]"


def test_a_missing_ledger_with_nothing_adjudicated_is_a_quiet_zero(tmp_path, capsys):
    """The other half of clause 4: a fresh install is not an alarm.

    `noise` is a candidate disposition the pipeline writes about a candidate's own
    content — 923 of the live corpus's files carry it, and the great majority never had a
    ledger row — so counting it here would turn every scratch run over that corpus into a
    false alarm, which is how a warning becomes the thing people skip.
    """
    cands = tmp_path / "candidates"
    cands.mkdir()
    raw_candidate(cands, "candidate-bash-noise-20260922.md", "noise")
    raw_candidate(cands, "candidate-read-pending-20260922.md", "pending_review")
    capsys.readouterr()

    rc = sv.main(["check", "--candidates", str(cands),
                  "--store", str(tmp_path / "gone" / "verdicts.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "LEDGER_ABSENT" not in out, out
    assert out.splitlines()[-1] == "checked: 2  skipped_by_verdict: 0"


def raw_candidate(candidates_dir: Path, name: str, status: str) -> Path:
    """One candidate file with a hand-written frontmatter `status:`.

    The shape the corpus actually holds after a runbook dispositions a pattern by hand —
    which is what a wiped ledger has to be compared against, and which the miner's own
    writer cannot produce any more once a key carries a verdict.
    """
    file = candidates_dir / name
    file.write_text(f"---\npattern: Bash/timeout\nstatus: {status}\noccurrences: 9\n---\n\n"
                    "## Examples\n\n- one mined example\n", encoding="utf-8")
    return file


# ── #736 clause 5: the runbook half, pinned from the code tree ───────────────

#: The mined-skill runbook. Read live, the way
#: `tests/test_consolidation_source_gate.py:43` reads the consolidator's: step 3.5 of this
#: file is the write-back that populates the ledger under test, so its shape is this
#: module's contract as much as the code's.
MINING_RUNBOOK = (Path.home() / "obsidian" / "skills" / "trajectory-skill-mining"
                  / "SKILL.md")


def _unclosed_span_before_heading(lines: list[str]) -> list[tuple[int, str]]:
    """Lines that end inside an inline-code span with a heading as the next paragraph.

    Prose wraps, so an odd backtick count on its own is legal: a span opened at the end of
    one line closes on the next, and `documentation-digester`, `service-health-check`,
    `skill-lint` and `task-notification-handling` all trip that crude reading today from
    tables and wrapped prose. What a reader cannot recover from is a span still open when
    the next paragraph is a heading — the sentence ended, and whatever token was going to
    close the span is gone. Fenced blocks are skipped: their backticks are delimiters.
    """
    fence = False
    offenders = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence or line.count("`") % 2 == 0:
            continue
        nxt = next((n for n in lines[i + 1:] if n.strip()), "")
        if nxt.lstrip().startswith("#"):
            offenders.append((i + 1, line))
    return offenders


def test_the_runbook_install_step_names_the_status_token_to_write():
    """Clause 5: step 3 told the miner to mark candidates as something, and not what.

    The bullet read `- Mark source candidates as `` — an unterminated inline-code span with
    a heading next, which renders as a swallowed token for the model consuming this file as
    a runbook. #530 then introduced two vocabularies (`status: superseded_by_verdict` in the
    corpus, `verdict:` values in the ledger), so a reader could not even tell whether the
    lost token predates #530 or names one of them.

    The token is `reviewed_authored`, not an invention: step 3.5's own example records
    `--verdict reviewed_authored` for an authored skill, and the corpus carries the
    on-disk precedent `status: reviewed_authored — skill bash-transient-error-handling
    authored from this candidate (08-27 mining)`. Nothing earlier was recoverable — the
    bullet is already truncated in `05ce35ae`, the vault's 2026-08-22 baseline commit, so
    every version the vault history holds ends mid-sentence.
    """
    assert MINING_RUNBOOK.exists(), \
        f"{MINING_RUNBOOK} is absent: the runbook this clause binds is not here"
    lines = MINING_RUNBOOK.read_text(encoding="utf-8", errors="replace").splitlines()

    bullets = [ln for ln in lines if ln.strip().startswith("- Mark source candidates")]
    assert len(bullets) == 1, f"step 3 must state the disposition exactly once: {bullets}"
    assert bullets[0].count("`") == 2, f"the code span must be closed: {bullets[0]!r}"
    assert "reviewed_authored" in bullets[0], \
        f"the bullet must name the status token to write: {bullets[0]!r}"

    assert _unclosed_span_before_heading(lines) == []


# ── #830: the ledger has to honour the dispositions the runbooks already write ─
#
# The defect, in one sentence: `nightly-skill-consolidation` Phase 5.1 routes `consolidated`
# and `noise` to this ledger, `trajectory-skill-mining` step 3.5 records `reviewed_authored`,
# and `record_verdict` validates only that a verdict is non-empty — so the store accepted all
# three, appended the line, and `is_terminal` honoured none of them. A run that followed the
# runbook wrote a decision that bound nothing: 5 `reviewed_authored` rows and 1 `noise` row
# in the 116-row ledger as measured at triage (2026-09-18), and the 248 of 1,132 candidate
# keys whose `noise` lived only in frontmatter were re-emitted `pending_review` by the next
# mining run. Nothing pinned the set before this section, which is why it stayed at five.

#: The dispositions the runbooks prescribe, inert until #830. Each is a stronger statement
#: about a pattern than the weakest binding verdict (`reviewed_no_skill`): a skill was
#: authored from it, its evidence folded into a proposal, or it is not skill-worthy at all.
RUNBOOK_PRESCRIBED_VERDICTS = ("reviewed_authored", "noise", "consolidated")

#: One candidate key per prescribed verdict, so every test here iterates one mapping
#: instead of zipping the constant against a literal of the same length — a zip against a
#: literal 3-tuple truncates silently the moment a fourth disposition joins the vocabulary,
#: and the set-pin test would then be forcing a value these tests never exercised.
RUNBOOK_CASES = {
    "reviewed_authored": "Edit/not_found",
    "noise": "Bash/timeout",
    "consolidated": "Read/validation",
}


def hand_candidate(candidates_dir: Path, name: str, pattern_key: str,
                   occurrences: int) -> Path:
    """One undecided candidate file, with the key and count this section needs.

    `raw_candidate` above pins a hand-written `status:` and fixes both the key and the
    count, which is what its own clause needs; the reopen rules below are about a count, so
    these tests say theirs. `pending_review` is what the miner regenerates every night, so
    it is the state a verdict has to survive in production: with a disposition already
    written into the file, a run could read a non-`PROCEED` outcome as the ledger working
    when it was only the file agreeing with itself.
    """
    candidates_dir.mkdir(parents=True, exist_ok=True)
    file = candidates_dir / name
    file.write_text(f"---\npattern: {pattern_key}\nstatus: pending_review\n"
                    f"occurrences: {occurrences}\n---\n\n## Examples\n\n- one\n",
                    encoding="utf-8")
    return file


def test_the_terminal_set_is_pinned_to_the_whole_disposition_vocabulary():
    """Clause 1: the set is exactly the eight values that end a pattern's life, and `proposed`
    is still not among them.

    Exact equality, not containment, is the pin the item asked for: before #830 nothing in
    this file named `TERMINAL_VERDICTS` at all, and the defect was the set stopping at five
    while the runbooks prescribed eight — `rejected_false_positive`, `reviewed_no_skill`,
    `rejected_unverifiable`, `rejected_artifact_class` and `archived_content` bound, while
    `reviewed_authored`, `noise` and `consolidated` were written by the runbooks and
    honoured by nobody.

    `proposed` is asserted away from the set explicitly because it is the one disposition
    that must keep accumulating evidence: a patch below the auto-apply threshold stays live
    until it crosses the threshold (`nightly-skill-consolidation` Phase 5.1), and a set that
    swept it up would freeze every proposed patch at its proposal count.
    """
    assert sv.TERMINAL_VERDICTS == frozenset({
        "rejected_false_positive",
        "reviewed_no_skill",
        "rejected_unverifiable",
        "rejected_artifact_class",
        "archived_content",
        "reviewed_authored",
        "noise",
        "consolidated",
    }), "the terminal set is the adjudicated vocabulary; changing it is a deliberate edit"
    assert "proposed" not in sv.TERMINAL_VERDICTS


def test_each_runbook_prescribed_verdict_supersedes_the_candidate(store, tmp_path):
    """Clause 2: a latest row of each of the three values stamps a re-mined candidate
    `superseded_by_verdict`, exactly as the five values that already bound do.

    The seam is `mine-trajectories.write_candidate_file`, not the ledger: the nightly's
    symptom was a candidate arriving as `status: pending_review` after a decision had been
    recorded, so the assertion has to be about the file the next phase reads. One store
    holds all three rows, each on its own key, which is also the shape that proves the
    latest-per-key table is consulted per key rather than once per run.
    """
    assert set(RUNBOOK_CASES) == set(RUNBOOK_PRESCRIBED_VERDICTS), \
        "every prescribed verdict needs its own key here, or clause 2 stops covering it"
    for verdict, key in RUNBOOK_CASES.items():
        tool, error_type = key.split("/", 1)
        sv.record_verdict(
            store=store, pattern_key=key, verdict=verdict,
            reason=f"{verdict}: adjudicated during the #830 widening",
            evidence_cmd=PRINTING_CMD, occurrences=13,
        )
        path = Path(mt.write_candidate_file(
            error_pattern(tool=tool, error_type=error_type), tmp_path / "c",
            verdict_store=store))

        assert status_of(path) == "superseded_by_verdict", (verdict, path.read_text())
        assert "status: pending_review" not in path.read_text(), verdict
        assert f"verdict: {verdict}" in frontmatter(path.read_text()), verdict


def test_check_skips_a_candidate_whose_only_row_is_a_runbook_verdict(tmp_path, capsys):
    """Clause 3: `check` prints `SKIP <key> :: <verdict>` for each of the three values and
    counts it in `skipped_by_verdict:`.

    This is the Phase 0 command the runbook actually runs, so it is asserted through
    `main` on stdout rather than through `is_terminal`: the nightly report carries the SKIP
    lines and the count, and a value that prints `PROCEED` while a decision sits in the
    store is the exact failure this item closes — `automod_gate/logic` was recorded
    `reviewed_authored` on 2026-09-11 when a skill was authored from it and `check` still
    printed `PROCEED` for it, which is what the item's own proof command was.

    One candidate per key, one ledger row per key: `pending_review` on disk and nothing
    else, so the only thing that can produce a SKIP is the verdict itself.
    """
    cands = tmp_path / "candidates"
    for verdict, key in RUNBOOK_CASES.items():
        hand_candidate(cands, f"candidate-{key.replace('/', '-')}-20260922.md", key, 13)
        sv.record_verdict(store=tmp_path / "verdicts.jsonl", pattern_key=key,
                          verdict=verdict, reason=f"decided: {verdict}",
                          evidence_cmd=PRINTING_CMD, occurrences=13)

    assert sv.main(["check", "--candidates", str(cands),
                    "--store", str(tmp_path / "verdicts.jsonl")]) == 0
    out = capsys.readouterr().out

    for verdict, key in RUNBOOK_CASES.items():
        assert f"SKIP {key} :: {verdict} ::" in out, (verdict, out)
    assert out.splitlines()[-1] == "checked: 3  skipped_by_verdict: 3", out


def test_a_runbook_verdict_still_reopens_on_age_or_growth(tmp_path, capsys):
    """Clause 4: making the three values terminal must not make them permanent — the same
    two triggers that lift the original five lift them.

    Four rows, and the point of the fourth is that each value is shown lifting on exactly
    one trigger at a time. `reopen_reason` returns on expiry before it looks at growth, so
    a row that is both aged and grown only ever demonstrates the age lift: `noise` and
    `reviewed_authored` are one trigger each, and `consolidated` — the strongest disposal
    in the vocabulary, and the one added on the argument that the reopen path is what makes
    stickiness safe — gets a row per trigger. The grown rows carry the live shape: the two
    keys sitting mis-disposed at triage held 23 and 38 occurrences at decision, so
    `REOPEN_OCCURRENCE_GROWTH` could already lift them.

    Each assertion names the key beside the reason it expects, so a lift that fired for the
    wrong reason on the wrong row cannot pass; and every lift must print `REOPEN`, never a
    silent `PROCEED`, because a report that cannot tell a lifted verdict from a missing
    ledger is the report that hides a wipe.
    """
    aged = (datetime.now(tz=timezone.utc)
            - timedelta(days=sv.REOPEN_AFTER_DAYS + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    decision_count = 13
    grew = int(decision_count * sv.REOPEN_OCCURRENCE_GROWTH) + 1
    plan = [
        # verdict, key, decided_at (None = now), candidate occurrences, lift expected
        ("noise", RUNBOOK_CASES["noise"], aged, decision_count, "expired"),
        ("reviewed_authored", RUNBOOK_CASES["reviewed_authored"], None, grew, "grew"),
        ("consolidated", RUNBOOK_CASES["consolidated"], aged, decision_count, "expired"),
        ("consolidated", "Write/permission", None, grew, "grew"),
    ]
    cands = tmp_path / "candidates"
    for i, (verdict, key, decided_at, occurrences, _) in enumerate(plan):
        hand_candidate(cands, f"candidate-{i}-{key.replace('/', '-')}-20260922.md",
                       key, occurrences)
        sv.record_verdict(store=tmp_path / "verdicts.jsonl", pattern_key=key,
                          verdict=verdict, reason=f"decided: {verdict}",
                          evidence_cmd=PRINTING_CMD, occurrences=decision_count,
                          **({"decided_at": decided_at} if decided_at else {}))

    assert sv.main(["check", "--candidates", str(cands),
                    "--store", str(tmp_path / "verdicts.jsonl")]) == 0
    out = capsys.readouterr().out

    for verdict, key, _, _, lift in plan:
        expected = (f"REOPEN {key} :: expired after {sv.REOPEN_AFTER_DAYS}d" if lift == "expired"
                    else f"REOPEN {key} :: occurrences grew {grew}>")
        assert expected in out, (verdict, key, lift, out)
    assert out.splitlines()[-1] == f"checked: {len(plan)}  skipped_by_verdict: 0", out


#: The consolidator's runbook, read the way `tests/test_consolidation_source_gate.py:43`
#: reads it and `MINING_RUNBOOK` above reads the mining one: Phase 1.3 is the filter that
#: re-admits an authored pattern every night, and it lives in this tree's contract because
#: the code's `TERMINAL_VERDICTS` is what it is supposed to mirror.
CONSOLIDATION_RUNBOOK = (Path.home() / "obsidian" / "skills"
                         / "nightly-skill-consolidation" / "SKILL.md")


def test_phase_1_3_skips_every_status_the_ledger_considers_terminal():
    """Clause 5: Phase 1.3's skip-list and `TERMINAL_VERDICTS` name the same endings.

    `reviewed_authored` is the token this item added: `trajectory-skill-mining` step 3
    writes it into a candidate's frontmatter when a skill was authored from it, so the most
    disposed outcome in the vocabulary was the one status the work-list filter did not skip
    — and the pattern came back every night, which is the title of this item.

    The list is derived from the code rather than repeated here for the reason
    `test_terminal_statuses_are_read_from_the_ledger_not_listed_here` gives: the next value
    added to `TERMINAL_VERDICTS` must fail here and get the runbook line written, instead of
    binding in Phase 0 while Phase 1.3 walks past it. `superseded_by_verdict` is in the
    expected set because the runbook skips the status the miner writes, even though it is
    not itself a verdict value.

    Asserted, never skipped: this clause's subject is a tree outside the gated repo, and a
    guard that can go quietly unverified repeats the defect it guards.
    """
    assert CONSOLIDATION_RUNBOOK.is_file(), \
        f"clause 5's subject is absent: {CONSOLIDATION_RUNBOOK} does not exist"
    body = CONSOLIDATION_RUNBOOK.read_text(encoding="utf-8", errors="replace")
    start = body.index("### 1.3 Build work list")
    section = body[start:body.index("## Phase 2:", start)]

    expected = set(sv.TERMINAL_VERDICTS) | {sv.SUPERSEDED_STATUS}
    missing = sorted(status for status in expected if f"`{status}`" not in section)
    assert missing == [], (
        f"Phase 1.3's skip-list does not name {missing}, so a pattern carrying one of "
        f"them is skipped by Phase 0's ledger and re-admitted by this filter: {section}")
    assert "`proposed`" in section, \
        "the filter must keep stating that a proposed patch is not skipped"


def test_widening_the_terminal_set_did_not_make_a_hand_written_status_ledger_minted(
        tmp_path, capsys):
    """The regression #830 could have introduced, pinned: three of the newly terminal
    dispositions are statuses a runbook writes into a candidate's own frontmatter about the
    candidate's own content, so their presence proves a decision was made, never that a
    verdict store existed.

    `MINTED_BY_LEDGER` is what turns an absent ledger into `LEDGER_ABSENT` and a non-zero
    exit. Deriving it from `TERMINAL_VERDICTS` — which is what it used to do, and every
    value in the old five plus `superseded_by_verdict` was ledger-only — would have made a
    scratch run over the real corpus alarm on its own 930 `status: noise` files and 3
    `status: reviewed_authored` ones, most of which never had a ledger row to lose. That is
    the same fail-loud-when-nothing-is-wrong shape as a `LEDGER_ABSENT` that fires on a
    fresh install, and it would train a run to ignore the one alarm that means a decision
    history is gone.
    """
    cands = tmp_path / "candidates"
    cands.mkdir()
    for status in RUNBOOK_PRESCRIBED_VERDICTS:
        raw_candidate(cands, f"candidate-{status}-20260922.md", status)
    capsys.readouterr()

    rc = sv.main(["check", "--candidates", str(cands),
                  "--store", str(tmp_path / "gone" / "verdicts.jsonl")])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "LEDGER_ABSENT" not in out, out
    assert set(RUNBOOK_PRESCRIBED_VERDICTS).isdisjoint(sv.MINTED_BY_LEDGER), (
        "a runbook-written frontmatter status must not count as proof a ledger existed")
