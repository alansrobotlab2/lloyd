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
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests._live_data import require_live_data
from app.paths import production_data_root  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
#: This module's own namespace, so a pin test can redirect `LIVE_LEDGER` by name and
#: the guard beneath it reads the redirected value on its next global lookup.
_THIS = sys.modules[__name__]


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
#:
#: Since #2052 the mint also refuses an rc-0 command whose stdout declares no input count,
#: and this fixture is the one most nodes mint through, so it declares: `input_rows=1` is the
#: honest count for a `grep -c` over the one file it names — the input it inspected. Two
#: shapes here are load-bearing, and both are why adopting the field is cheap: the declaration
#: rides on a SECOND line, because `run_evidence` stores the FIRST non-empty line as
#: `evidence_observed` and nodes assert against that string; and the exit code is captured and
#: re-raised, because `check` and `audit` read rc, and `cmd; echo …` alone would turn a
#: falsifier answering no (rc 1) into one answering yes (rc 0).
PRINTING_CMD = (f"grep -c '^def ' {_ROOT / 'scripts' / 'skill_verdicts.py'}; "
                "rc=$?; echo 'input_rows=1'; exit $rc")


def declares(cmd: str, rows: int = 1) -> str:
    """Add the #2052 declaration to a fixture command, changing nothing else about it.

    For a node minting a verdict in order to test some other property. The second line and
    the preserved exit status are `PRINTING_CMD`'s reasons above. A node that IS about the
    denominator builds its command text itself and says so, because a fixture declaring a
    count it did not read is the field-with-no-measurement state: as in the ledger,
    `input_rows=<N>` is a claim about inputs, and `rows` here is only ever a file count the
    calling node can point at.
    """
    return f'{cmd}; rc=$?; echo "{sv.INPUT_ROWS_FIELD}={rows}"; exit $rc'


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
        evidence_cmd=declares("grep -ci 'command timed out' "
                              "~/obsidian/skills/bash-timeout/SKILL.md"),
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
        reason="patch below auto-apply threshold",
        evidence_cmd=declares("echo 'proposed: patch below the auto-apply threshold'"), occurrences=13,
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
        reason="co-occurs by accident, no causal link",
        evidence_cmd=declares("echo 'seq-2-bash-fs-read co-occurs without a causal link'"),
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


# ── #1586: a falsifier that cannot run is not evidence ───────────────────────────
#
# `run_evidence` quotes output, and a command that never started prints output too — its
# own failure. So #736 clause 2's emptiness test let three UNRUNNABLE shapes be recorded:
# measured through the real `record_verdict` on 2026-09-27, each of the three below was
# ACCEPTED and appended with `evidence_observed` holding its own crash
# (`grep: /nonexistent-file-xyz.md: No such file or directory`, bash's parse error,
# `bash: line 1: …: command not found`). Such a verdict is then honoured by `check` for its
# full 60 days while that check's own re-execution prints EVIDENCE_CMD_UNRUNNABLE — the
# state #1533's `audit` counts as `keys: 104 unrunnable: 79`, minted at write time. The
# fourth shape `evidence_cmd_status` names, a check that outlives its timeout, prints
# nothing and was already refused by the emptiness guard.

#: The three shapes that reach `record`'s classifier with output to quote, keyed by what
#: broke. Each is a command no falsifier can legitimately be.
UNRUNNABLE_SHAPES = {
    "absent-file": "grep -c zzz /nonexistent-file-xyz.md",
    "bash-parse-error": "this is prose (not a command) and bash cannot parse it",
    "not-found-binary": "this-command-does-not-exist-xyz",
}

#: The fragment each shape's classifier detail must reach the refusal with, so the nightly
#: log names which of the three broke without anyone re-running the command.
UNRUNNABLE_SHAPE_DETAIL = {
    "absent-file": "No such file or directory",
    "bash-parse-error": "syntax error",
    "not-found-binary": "command not found",
}

#: A falsifier that reports its measurement on stderr and exits non-zero. Legal by
#: construction (`run_evidence` falls back to stderr precisely for this) and the case a
#: refusal keyed on exit code instead of runnability would wrongly kill.
STDERR_MEASUREMENT = "printf 'count=3\\n' >&2; exit 3"

#: Assembled rather than literal: a token this file spells out would be found by the very
#: `grep -c` that is supposed to come back 0, and the falsifier would answer yes.
NO_SUCH_TOKEN = "ZZZ" + "_NO_SUCH_TOKEN_" + "7F"


def test_record_refuses_an_evidence_cmd_that_cannot_run(store, mirror):
    """Clause 1: each UNRUNNABLE shape raises, and neither the ledger nor its durable
    copy gains a line.

    The accepted verdict first is what makes the line counts a test rather than a
    tautology: with both trees empty, "still zero" would pass a refusal that crashed
    before writing and one that wrote then raised.
    """
    sv.record_verdict(store=store, pattern_key="Bash/logic", verdict="reviewed_no_skill",
                      reason="the accepted baseline whose lines the assertions below count",
                      evidence_cmd=PRINTING_CMD)
    assert len(store.read_text().splitlines()) == 1
    assert mirror.read_text().splitlines() == store.read_text().splitlines()

    for shape, cmd in UNRUNNABLE_SHAPES.items():
        with pytest.raises(ValueError, match="UNRUNNABLE"):
            sv.record_verdict(store=store, pattern_key=f"test/{shape}",
                              verdict="reviewed_no_skill",
                              reason=f"{shape} cannot support a verdict",
                              evidence_cmd=cmd)

    assert len(store.read_text().splitlines()) == 1, "a refusal writes no line"
    assert mirror.read_text().splitlines() == store.read_text().splitlines(), \
        "a refusal must not reach the durable copy either"


def test_the_refusal_names_the_shape_and_quotes_the_command(store):
    """Clause 2: the refusal says which shape the command has and repeats the command, so
    `refused: …` in the nightly log is actionable without re-running anything.

    Three assertions per shape: the word `UNRUNNABLE` (the same token `check` and `audit`
    print, so one grep finds both ends), the command verbatim, and the fragment of the
    classifier's own detail that distinguishes this shape from the other two.
    """
    for shape, cmd in UNRUNNABLE_SHAPES.items():
        with pytest.raises(ValueError) as excinfo:
            sv.record_verdict(store=store, pattern_key=f"test/{shape}",
                              verdict="reviewed_no_skill",
                              reason=f"{shape} cannot support a verdict",
                              evidence_cmd=cmd)
        message = str(excinfo.value)
        assert "UNRUNNABLE" in message, shape
        assert cmd in message, f"{shape}: the refusal must quote the command: {message}"
        assert UNRUNNABLE_SHAPE_DETAIL[shape] in message, \
            f"{shape}: the refusal must name the shape: {message}"


def test_a_nonzero_exit_beside_a_printed_count_still_records(store):
    """Clause 3: rc 1 beside a measurement is a falsification, not a hole, and the
    emptiness guard is untouched.

    Both halves are the same boundary seen from either side of the new guard: a `grep -c`
    that prints `0` and exits 1 must still be recorded with that `0` as its
    `evidence_observed` (a falsifier reporting its verdict false is the ledger working),
    while a command that prints nothing is still refused with #736's own `observed nothing`
    message — the classifier check sits *after* the emptiness check, so no command that
    #736 refused now reaches a different error, and no command that #736 accepted and
    printed something is now accepted unexamined.
    """
    count_cmd = (f"grep -c '{NO_SUCH_TOKEN}' "
                 f"{_ROOT / 'scripts' / 'skill_verdicts.py'}")
    assert sv.run_evidence(count_cmd)[0] == 1, \
        "the fixture must exit 1: `grep -c` with no match is a falsifier answering no"
    row = sv.record_verdict(store=store, pattern_key="Bash/timeout",
                            verdict="reviewed_no_skill",
                            reason="the count is the measurement, and it is zero",
                            evidence_cmd=count_cmd)
    assert row["evidence_observed"] == "0"

    with pytest.raises(ValueError, match="observed nothing"):
        sv.record_verdict(store=store, pattern_key="Edit/not_found",
                          verdict="reviewed_no_skill",
                          reason="a command that says nothing cannot be overturned",
                          evidence_cmd="true")


def test_record_refuses_exactly_what_the_classifier_calls_unrunnable(store):
    """Clause 4: the refused set is `evidence_cmd_status`'s, imported rather than invented.

    For every command here, `record` accepts iff the classifier the ledger's readers run
    (`check`, `audit`) reports as not-UNRUNNABLE. Every case prints at least one line, so
    #736's emptiness guard is inert across this table and the classifier is the only
    refusal on the table — `true`, runnable but silent, is clause 3's case, not one here.
    Since #2052 the accepted group has one more thing to satisfy: an rc-0 command that
    declares no input count is refused too, which is why the prose-only case below carries
    `declares(...)` while the two cases that ride on a non-zero exit (`grep -c` at rc 1, the
    stderr measurement at rc 3) do not need it — the mandate is keyed on rc 0, so a falsifier
    answering no stays recordable exactly as #2048 left it.
    The accepted group includes the two shapes a rule keyed on exit code would wrongly
    refuse: `grep -c` exiting 1 with its `0`, and a measurement on stderr at exit 3.
    """
    count_cmd = (f"grep -c '{NO_SUCH_TOKEN}' "
                 f"{_ROOT / 'scripts' / 'skill_verdicts.py'}")
    cases = [
        (UNRUNNABLE_SHAPES["absent-file"], True),
        (UNRUNNABLE_SHAPES["bash-parse-error"], True),
        (UNRUNNABLE_SHAPES["not-found-binary"], True),
        (declares("echo 'bash-timeout owns this signature'"), False),
        (PRINTING_CMD, False),
        (count_cmd, False),
        (STDERR_MEASUREMENT, False),
    ]
    for index, (cmd, expect_unrunnable) in enumerate(cases):
        status_rc, _detail = sv.evidence_cmd_status({"evidence_cmd": cmd})
        assert (status_rc == sv.UNRUNNABLE) == expect_unrunnable, \
            f"the classifier itself disagrees about {cmd!r}"
        refused = False
        try:
            sv.record_verdict(store=store, pattern_key=f"test/iff-{index}",
                              verdict="reviewed_no_skill",
                              reason="one row per case, so the accepted ones are countable",
                              evidence_cmd=cmd)
        except ValueError:
            refused = True
        assert refused == expect_unrunnable, \
            f"record {'refused' if refused else 'accepted'} {cmd!r}; " \
            f"classifier UNRUNNABLE={expect_unrunnable}"

    stored = {json.loads(ln)["evidence_cmd"]: json.loads(ln)["evidence_observed"]
              for ln in store.read_text().splitlines()}
    assert stored[STDERR_MEASUREMENT] == "count=3", \
        "a check reporting through stderr records its measurement, not its silence"
    assert stored[count_cmd] == "0"


def test_the_shipped_cli_refuses_an_unrunnable_check_without_touching_either_tree(tmp_path,
                                                                                  mirror):
    """The same refusal across the boundary the nightly actually crosses.

    `nightly-skill-consolidation` Phase 5.1 records verdicts by invoking
    `skill_verdicts.py record` as a subprocess, so the guard's exit status and its stderr
    line — not the `ValueError` — are what a nightly sees, and the durable copy is written
    by that child, not by this process. This spawns the shipped module from this checkout
    with `$SKILL_VERDICTS_MIRROR` aimed at the tmp copy: one accepted record to give both
    trees a line, then one refused record, which must exit non-zero, name UNRUNNABLE and
    quote the command on stderr, and leave both trees byte-identical to the accepted line.
    """
    store = tmp_path / "cli-reviews" / "verdicts.jsonl"
    env = dict(os.environ, SKILL_VERDICTS_MIRROR=str(mirror))
    run = lambda *args: subprocess.run(
        [sys.executable, "-m", "scripts.skill_verdicts", *args],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=120)

    accepted = run("record", "--pattern", "Bash/timeout", "--verdict", "reviewed_no_skill",
                   "--reason", "installed skill bash-timeout owns this signature",
                   "--evidence-cmd", PRINTING_CMD, "--occurrences", "13",
                   "--store", str(store))
    assert accepted.returncode == 0, accepted.stderr
    assert len(store.read_text().splitlines()) == 1

    refused = run("record", "--pattern", "Write/logic", "--verdict", "reviewed_no_skill",
                  "--reason", "grounds a crash message cannot support",
                  "--evidence-cmd", UNRUNNABLE_SHAPES["absent-file"],
                  "--occurrences", "4", "--store", str(store))
    assert refused.returncode != 0, "an unrunnable falsifier must fail the shipped CLI too"
    assert "UNRUNNABLE" in refused.stderr, refused.stderr
    assert UNRUNNABLE_SHAPES["absent-file"] in refused.stderr, refused.stderr
    assert "No such file or directory" in refused.stderr, refused.stderr
    assert len(store.read_text().splitlines()) == 1, "a refusal writes no line"
    assert mirror.read_text().splitlines() == store.read_text().splitlines(), \
        "a refusal must not reach the durable copy either"


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
                      evidence_cmd=declares("grep result_summary scripts/mine-trajectories.py"))
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

LIVE_LEDGER = production_data_root() / "_pipeline" / "skills" / "reviews" / "verdicts.jsonl"

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

    An absent ledger skips and names its path; a ledger that EXISTS always runs the
    loop. `_pipeline/` is gitignored, so this reads the machine's real append-only
    ledger rather than a committed fixture — the same convention as `LIVE_CORPUS` in
    `test_trajectory_extraction.py`, and the same #1377 rule. After the 2026-09-22 wipe
    this assert was not guarding against orphaning; it was one of the nodes making every
    round's `tests` rung refuse to promote onto a tree it had not broken. What still
    fails over a present ledger is the orphaning itself —
    `test_a_legacy_truncated_seq_key_in_a_present_ledger_still_fails` is that proof.
    """
    require_live_data(LIVE_LEDGER, "verdict ledger", kind="file")
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


def test_the_sequence_verdict_guard_skips_when_the_verdict_ledger_is_absent(
        tmp_path, monkeypatch):
    """Clause 4 of backlog #1377, from the same rule as the seven guards in
    `test_trajectory_extraction.py`.

    This node asserted the ledger's presence, and after the 2026-09-22 wipe the ledger
    did not exist, so it reproduced as a failure *at base* in every round's `tests` rung
    — one of the 106 in `base_probe: probed 21 file(s) at base 1842b8cf: 106 already
    failing`, copied from
    `~/.local/state/lloyd-automod/rounds/SM_20260922_201206/gate.json`. Absence of
    derived data is a property of the machine and becomes a named skip; the reason has
    to name the path, because a bare `skipped` would hide which root went missing.

    Run by calling the guard with `LIVE_LEDGER` redirected, so this pins the rule on a
    machine that HAS a ledger too rather than only on one that lost it.
    """
    missing = tmp_path / "gone" / "verdicts.jsonl"
    monkeypatch.setattr(_THIS, "LIVE_LEDGER", missing)
    with pytest.raises(pytest.skip.Exception) as caught:
        test_every_stored_sequence_verdict_still_resolves_after_the_widening(tmp_path)
    assert str(missing) in str(caught.value), (
        f"skipped for a reason that does not name the missing ledger: {caught.value}")


def test_a_legacy_truncated_seq_key_in_a_present_ledger_still_fails(
        tmp_path, monkeypatch):
    """The other side of that skip, and the invariant the guard was written for.

    A synthetic ledger in a tmp root holding one `seq-*` row whose slug is cut to
    exactly `SLUG_CAP` — the legacy truncation's signature, the class a widened key
    would silently orphan — must make the guard FAIL, not skip. A skip that could fire
    over a ledger that is present would convert a red orphaning-detector into a green
    nothing.
    """
    legacy_key = f"seq-3-{'reuse1377' + 'x' * 41}"
    assert len(seq_slug_of(legacy_key)) == SLUG_CAP, "the fixture must be the legacy shape"
    ledger = tmp_path / "reviews" / "verdicts.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({
        "pattern_key": legacy_key, "verdict": "rejected",
        "reason": "fixture", "decided_at": "2026-09-22T00:00:00+00:00",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(_THIS, "LIVE_LEDGER", ledger)
    with pytest.raises(AssertionError) as caught:
        test_every_stored_sequence_verdict_still_resolves_after_the_widening(tmp_path)
    assert "legacy truncated" in str(caught.value), caught.value


def test_an_empty_but_present_ledger_fails_rather_than_skips(tmp_path, monkeypatch):
    """Clause 3's shape for this guard: the ledger EXISTS and holds nothing, so the
    skip's precondition is not met and the row floors below have to fire. Emptying the
    file and deleting it are different events — `skill_verdicts.resolve_verdict_source`
    says so for the same reason — and a guard that skipped on both would report green
    over a ledger whose rows had been lost."""
    ledger = tmp_path / "reviews" / "verdicts.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("", encoding="utf-8")
    monkeypatch.setattr(_THIS, "LIVE_LEDGER", ledger)
    with pytest.raises(AssertionError) as caught:
        test_every_stored_sequence_verdict_still_resolves_after_the_widening(tmp_path)
    assert "under-cap seq-* rows" in str(caught.value), caught.value


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
                      evidence_cmd=PRINTING_CMD)
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
        evidence_cmd=declares("grep -ci 'command timed out' "
                              "~/obsidian/skills/bash-timeout/SKILL.md"),
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
        evidence_cmd=declares("grep -ci 'command timed out' "
                              "~/obsidian/skills/bash-timeout/SKILL.md"),
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
    # The two runnable keys go through `record`. The two that must sit in the ledger
    # *unrunnable* are stated as stored rows instead, because since #1586 `record` refuses
    # to create them — an absent script and a command bash cannot parse are exactly what
    # the write path now turns away, and that refusal is why a broken falsifier can only
    # arrive here as a row written before it: 99 of the live ledger's keys were recorded
    # before any guard existed and will be read long after this one.
    for key, cmd in (("Bash/timeout", declares("echo 'bash-timeout owns this signature'")),
                     ("Edit/not_found", "echo 'the grounds are gone'; exit 1")):
        sv.record_verdict(store=store, pattern_key=key, verdict="reviewed_no_skill",
                          reason=f"grounds for {key}", evidence_cmd=cmd)
    for key, cmd in (("Bash/logic", f"{tmp_path}/gone/falsifier.sh"),
                     ("Write/logic", 'echo "unbalanced')):
        stored_row(store, key, cmd, reason=f"grounds for {key}")
    for key in ("Bash/timeout", "Edit/not_found", "Bash/logic", "Write/logic"):
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

    untracked_cmd = declares(f"python3 {untracked}")
    sv.record_verdict(store=store, pattern_key="seq-3-bash-fs-bash-other",
                      verdict="reviewed_no_skill", reason="grounds",
                      evidence_cmd=untracked_cmd)

    copied = mirror.parent / untracked.name
    assert copied.is_file()
    assert copied.read_bytes() == untracked.read_bytes()
    assert json.loads(mirror.read_text().splitlines()[-1])["evidence_cmd"] == untracked_cmd

    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="grounds checked by a tracked module",
                      evidence_cmd=declares(f"python3 {tracked}"))  # stored verbatim
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
    minted = declares(f"bash {script}")
    sv.record_verdict(store=store, pattern_key="Bash/timeout", verdict="reviewed_no_skill",
                      reason="grounds with a real re-executable check",
                      evidence_cmd=minted)

    row = sv.load_verdicts(store)["Bash/timeout"]
    assert row["evidence_cmd"] == minted, "the seam's input is the stored string"
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
    # The line carries the class word `DISAGREEING` since #1717 clause 3, which is what
    # tells it apart from a `LEDGER_MIRROR_LAGGED` line (the copy keeping behind with the
    # *same* verdict). The `LEDGER_MIRROR_CONFLICT` token itself is deliberately unchanged:
    # the nightly and this item's acceptance grep it, and re-naming it would have made that
    # grep go quiet for the wrong reason rather than because the divergence is gone.
    assert conflict == [
        f"LEDGER_MIRROR_CONFLICT Bash/timeout :: DISAGREEING live=noise "
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
                      evidence_cmd=declares("echo '367 occurrences over 41 sessions'"),
                      occurrences=367)
    sv.record_verdict(store=store, pattern_key=key, verdict="reviewed_no_skill",
                      reason="correction: the owning skill is bash-fs, not bash-explore",
                      evidence_cmd=declares("echo '367 occurrences over 41 sessions'"),
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
                      evidence_cmd=declares("echo '90 over 6 sessions'"), occurrences=90)
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
            "--evidence-cmd", declares("echo '13 over 3 sessions'"), "--store", str(store)]
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
                    "--evidence-cmd", declares("echo '13 occurrences over 3 sessions'"),
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
    minted = declares(f"{sys.executable} {script}")
    assert sv.main(["record", "--pattern", "seq-2-read-write", "--verdict", "reviewed_no_skill",
                    "--reason", "3 of 7 steps have no recovery, under the threshold",
                    "--evidence-cmd", minted, "--occurrences", "275", "--store", str(store)]) == 0
    row = sv.load_verdicts(store)["seq-2-read-write"]
    assert row["evidence_observed"] == "sess=7 steps_ok=4 steps_err=3 has_error_recovery=True"
    assert "not quoted" not in row["evidence_observed"], "the first line only"
    assert row["evidence_cmd"] == minted, "the command is stored byte-for-byte as minted"

    # A command reporting through stderr is stored with what it printed, not with nothing.
    # The fixture here used to be `grep -c 'x' /nope/nothing-here`, whose entire output was
    # grep's own `No such file or directory` — the shape #1586 now refuses at write time —
    # so the case rides on a falsifier that measures something, prints it on stderr, and
    # exits non-zero: the reason `run_evidence` falls back to stderr in the first place.
    assert sv.main(["record", "--pattern", "Write/logic", "--verdict", "reviewed_no_skill",
                    "--reason", "the check reports its count on stderr and exits 3",
                    "--evidence-cmd", STDERR_MEASUREMENT,
                    "--occurrences", "4", "--store", str(store)]) == 0
    assert sv.load_verdicts(store)["Write/logic"]["evidence_observed"] == "count=3"

    # A long first line is a quotation, not an attachment.
    long_line = "M" * (sv.EVIDENCE_OBSERVED_MAX + 50)
    assert sv.main(["record", "--pattern", "Bash/logic", "--verdict", "reviewed_no_skill",
                    "--reason", "a falsifier that prints a wall of text",
                    "--evidence-cmd", declares(f"printf '{long_line}\\n'"),
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


# ═══════════════════════════════════════════════════════════════════════════
# `audit` — the ledger's falsifiers get re-executed, not inherited (#1533).
#
# #530/#525 made `evidence_cmd` mandatory so that a verdict is a check rather than
# an assertion. `~/lloyd/_pipeline` was deleted on 2026-09-22 (#1377) and the
# ledger's stored commands still name it, so for most of the ledger the check
# cannot run while `check` goes on honouring the decision it carried: 78 of 103
# latest-wins keys tonight, per the item's proving command. Nothing noticed,
# because `check` only executes the commands of keys that blocked a candidate in
# the scanned directory — 2 tonight, both re-recorded with live commands.
#
# Each fixture command below was run through the classifier before it was written
# into a fixture, so the shapes are the classifier's own four reasons and not
# guesses: a missing file (rc relabelled 127, stderr names it), a command bash
# cannot parse (rc 2 plus bash's parse-error prefix), rc 127 itself, and a timeout.
# ═══════════════════════════════════════════════════════════════════════════

AUDIT_SEES = "sweep/observed_x"                  # exits 0 after printing its count
AUDIT_SEES_NONE = "sweep/observed_none"          # prints count=0, exits 1: still an observation
AUDIT_MISSING_FILE = "sweep/missing_file"        # stderr: No such file or directory
AUDIT_PARSE_ERROR = "sweep/parse_error"          # bash: …: unexpected EOF while …
AUDIT_NO_SUCH_COMMAND = "sweep/no_such_command"  # rc 127
AUDIT_HANGS = "sweep/hangs"                      # exceeds --timeout
AUDIT_FAILED_BUT_RAN = "sweep/error_2"           # rc 2, nothing missing: ran and failed
#: #2048's fifth state and its control: the first exits 0 and declares it read no input,
#: the second exits 0 declaring the 9 rows it read. Both print a count, so the pair is the
#: exact shape the item's proof has — rc 0 and a number — and only the denominator tells
#: them apart. Neither is in AUDIT_DEAD: `unrunnable:` is about executing, not grounding.
AUDIT_EMPTY_INPUT = "sweep/zero_denominator"     # rc 0, stdout declares input_rows=0
AUDIT_DECLARES = "sweep/declared_nonzero"        # rc 0, stdout declares input_rows=9

AUDIT_CMDS = {
    AUDIT_SEES: "echo 'sweep/observed_x count=4'",
    AUDIT_SEES_NONE: "echo 'sweep/observed_none count=0'; exit 1",
    AUDIT_MISSING_FILE: "grep -c x /tmp/definitely-missing-falsifier-file-1533.md",
    AUDIT_PARSE_ERROR: 'echo "unbalanced',
    AUDIT_NO_SUCH_COMMAND: "definitely-not-a-command-1533",
    AUDIT_HANGS: "sleep 5",
    AUDIT_FAILED_BUT_RAN: "grep -c x .",
    AUDIT_EMPTY_INPUT: "echo 'sweep/zero_denominator input_rows=0 matched=0'",
    AUDIT_DECLARES: "echo 'sweep/declared_nonzero input_rows=9 matched=2'",
}
AUDIT_DEAD = {AUDIT_MISSING_FILE, AUDIT_PARSE_ERROR, AUDIT_NO_SUCH_COMMAND, AUDIT_HANGS}


def write_ledger(store: Path, keys) -> None:
    """A ledger holding exactly `keys`, one latest-wins row each.

    Written by hand rather than with `record`, because `record` executes the command
    it is asked to store — right for a verdict, but it would put the fixture's own
    exit codes in front of the thing under test.
    """
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("w", encoding="utf-8") as fh:
        for key in keys:
            fh.write(json.dumps({"logged_at": "2026-09-26T07:00:00+00:00",
                                 "pattern_key": key, "candidate": key, "occurrences": 1,
                                 "verdict": "rejected_false_positive",
                                 "reason": "fixture: pattern is a traceback line",
                                 "scope": "project", "skill": "",
                                 "evidence_cmd": AUDIT_CMDS[key],
                                 "source": "skill-sweep"}) + "\n")


def run_audit(store: Path, timeout: int = 1) -> tuple[int, str]:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = sv.main(["audit", "--store", str(store), "--timeout", str(timeout)])
    return rc, buf.getvalue()


def test_audit_names_every_key_whose_stored_check_cannot_run(tmp_path):
    """Clause 1: one `UNRUNNABLE <pattern_key> :: <detail>` line per dead key, and
    none for a key whose check ran."""
    store = tmp_path / "verdicts.jsonl"
    write_ledger(store, [AUDIT_SEES, AUDIT_MISSING_FILE, AUDIT_PARSE_ERROR])

    rc, out = run_audit(store)
    lines = [ln for ln in out.splitlines() if ln.startswith("UNRUNNABLE ")]

    assert len(lines) == 2, out
    assert {ln.split(" :: ")[0] for ln in lines} == {
        f"UNRUNNABLE {AUDIT_MISSING_FILE}", f"UNRUNNABLE {AUDIT_PARSE_ERROR}"}, out
    # The detail is the classifier's, so a reader can tell a deleted file from an
    # unparseable command without executing anything themselves.
    missing = next(ln for ln in lines if AUDIT_MISSING_FILE in ln)
    assert "No such file or directory" in missing, missing
    parse = next(ln for ln in lines if AUDIT_PARSE_ERROR in ln)
    assert "bash:" in parse, parse
    assert AUDIT_SEES not in out, "a check that observed something was reported dead"


def test_audit_tally_is_its_final_line(tmp_path):
    """Clause 2: `keys: N unrunnable: M` last, because a nightly reading of this
    surface takes `splitlines()[-1]` — `check`'s existing contract, extended.

    The two numbers are asserted independently of each other and of the finding
    lines: `keys` is the ledger's latest-wins key count (7 keys written, 4 dead),
    so an implementation that printed the findings as the total, or counted rows
    instead of keys, fails here rather than downstream.
    """
    store = tmp_path / "verdicts.jsonl"
    write_ledger(store, sorted(AUDIT_DEAD) + [AUDIT_SEES, AUDIT_SEES_NONE,
                                              AUDIT_FAILED_BUT_RAN])

    rc, out = run_audit(store)
    last = out.splitlines()[-1]

    assert last == f"keys: {len(AUDIT_DEAD) + 3} unrunnable: {len(AUDIT_DEAD)}", out
    # +3, not +1: #2048 added one line above the tally (`denominators: …`) and #2103 another
    # above that one (`stranded: case_sensitive_grep …`), and the whole point of counting
    # lines here is that the published figure stays LAST — a nightly takes splitlines()[-1],
    # so any new tally has to arrive above it. Nothing in this ledger is case-stranded, which
    # is why the line reads 0 rather than being absent: a tally that only appears with
    # findings cannot tell a clean night from a check nobody ran.
    # +4 as of #2166, which added a third tally (`candidate_body_scoping: …`) above `stranded:`
    # for the same reason the other two are there: a clean night has to print a zero. Only this
    # arithmetic moves — `keys:` stays last, `denominators:` -2 and `stranded:` -3 below it.
    assert len(out.splitlines()) == len(AUDIT_DEAD) + 4, (
        f"tally must be the {len(AUDIT_DEAD) + 4}th and last line of its own output: {out}")
    assert out.splitlines()[-2] == "denominators: empty_input 0 undeclared 7", out
    assert out.splitlines()[-3] == "stranded: case_sensitive_grep 0", out
    assert out.count("UNRUNNABLE ") == len(AUDIT_DEAD), out


def test_audit_reports_exactly_the_unrunnable_classification(tmp_path):
    """Clause 3: the reported set is `evidence_cmd_status`'s decision, not a new rule.

    Four shapes must be reported, three must not. The three that must not are what
    makes this falsifiable: `grep -c x .` exits 2, and a "`rc != 0` means dead"
    implementation reports it; the two observing commands exit 0 and 1, which an
    "`anything` non-zero is dead" implementation reports too; `sleep 5` is caught
    only by something that enforces a timeout. The parity assertion at the end runs
    the same function `check` already uses over the same rows, so `audit` cannot
    answer a different question about a row than the one `check` asks of the two it
    happens to execute.
    """
    store = tmp_path / "verdicts.jsonl"
    write_ledger(store, list(AUDIT_CMDS))

    rc, out = run_audit(store)
    reported = {ln.split(" :: ")[0].removeprefix("UNRUNNABLE ")
                for ln in out.splitlines() if ln.startswith("UNRUNNABLE ")}

    assert reported == AUDIT_DEAD, out
    table = sv.load_verdicts(store)
    assert len(table) == len(AUDIT_CMDS), "fixture keys collided in the ledger"
    assert reported == {k for k, row in table.items()
                        if sv.evidence_cmd_status(row, timeout=1)[0] == sv.UNRUNNABLE}, (
        "audit's set disagrees with evidence_cmd_status on the same rows")


def test_audit_exits_1_when_anything_is_unrunnable_and_0_when_nothing_is(tmp_path):
    """Clause 4, first half: the exit code follows the tally in both directions.

    A nightly that reads only the tally line must still be failed by a red exit, and
    a ledger whose falsifiers all run must not train anyone to ignore one.
    """
    dead = tmp_path / "dead.jsonl"
    write_ledger(dead, [AUDIT_MISSING_FILE, AUDIT_SEES])
    rc_dead, out_dead = run_audit(dead)
    assert rc_dead == 1, out_dead
    assert out_dead.splitlines()[-1] == "keys: 2 unrunnable: 1", out_dead

    alive = tmp_path / "alive.jsonl"
    write_ledger(alive, [AUDIT_SEES, AUDIT_SEES_NONE])
    rc_alive, out_alive = run_audit(alive)
    assert rc_alive == 0, out_alive
    assert out_alive.splitlines()[-1] == "keys: 2 unrunnable: 0", out_alive
    assert "UNRUNNABLE " not in out_alive, out_alive


def test_check_still_reports_its_slice_and_its_own_final_line(tmp_path):
    """Clause 4, second half: `check` is untouched — same two numbers, same
    `checked: N  skipped_by_verdict: M` last line, same rc 0 when it blocks work on
    a verdict it could not verify (and still says `EVIDENCE_CMD_UNRUNNABLE`).

    The candidate is `Bash/timeout` at 9 occurrences and the ledger row for that key
    carries a falsifier naming a file that is not there, which is tonight's ledger
    in miniature: the decision is honoured, and only this line says it was honoured
    on an uncheckable basis.
    """
    cands = tmp_path / "candidates"
    cands.mkdir()
    raw_candidate(cands, "candidate-a.md", "rejected_artifact_class")
    store = tmp_path / "verdicts.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"logged_at": "2026-09-26T07:00:00+00:00",
                             "pattern_key": "Bash/timeout", "candidate": "x",
                             "occurrences": 9,
                             "verdict": "rejected_artifact_class",
                             "reason": "fixture: artifact-class pattern",
                             "scope": "project", "skill": "",
                             "evidence_cmd": AUDIT_CMDS[AUDIT_MISSING_FILE],
                             "source": "nightly-skill-consolidation"}) + "\n")

    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = sv.main(["check", "--candidates", str(cands), "--store", str(store)])
    out = buf.getvalue()

    assert rc == 0, out
    assert out.splitlines()[-1] == "checked: 1  skipped_by_verdict: 1", out
    assert "EVIDENCE_CMD_UNRUNNABLE Bash/timeout" in out, out


# ── #1588: the ledger repair pass over a moved data root ─────────────────────

def moved_root(tmp_path):
    """A dead in-tree `_pipeline` and the live sibling it moved to, as two tmp trees.

    The live half holds one candidate a real `grep` can find, so a repaired check
    *observes* something: `record` refuses a command that prints nothing (#736 clause 2),
    which is exactly the #1586 hazard — a "repair" that lands a check whose only output
    is its own error text is a fresh born-dead row, not a fix.
    """
    name = sv.PIPELINE_DIR.name
    dead = tmp_path / "checkout" / name
    live = tmp_path / "data" / name
    found = live / "skills" / "candidates" / "candidate-x-20260927.md"
    found.parent.mkdir(parents=True)
    found.write_text("---\npattern: Bash/timeout\nstatus: reviewed_no_skill\n---\n",
                     encoding="utf-8")
    return str(dead), str(live)


def root_move_row(store: Path, key: str, dead_root: str, stem: str,
                  verdict: str = "reviewed_no_skill", occurrences: int = 4) -> dict:
    """A stored verdict whose falsifier names the dead root for `stem`.

    `stem` ends in a date: the shape that decides which of #1588's three buckets the
    key lands in. A date retention still holds (20260927) is a key the substitution
    saves; one it pruned (20260908) is a key no rewrite can reach, because the file is
    missing from the old root *and* the new one.
    """
    return stored_row(store, key,
                      f"grep -c '^status:' {dead_root}/skills/candidates/{stem}.md",
                      verdict=verdict, reason="installed skill already owns this shape",
                      occurrences=occurrences)


def run_repair(store: Path, dead: str, live: str, *extra) -> tuple[int, str]:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = sv.main(["repair", "--store", str(store), "--rewrite", dead, live, *extra])
    return rc, buf.getvalue()


# ── #1717: the two trees are written together, or the skip is said out loud ──────
#
# On 2026-09-27 a repair pass appended 79 rows to the live ledger and none of them to the
# vault copy, and nothing on any surface said so. `check` then spent its whole divergence
# alarm on that lag: 79 lines of one label, 25 of which were the two trees holding
# *different verdicts* — the case the detector was built for, hidden inside noise its own
# blind spot produced. Three separate failures are pinned below: the silent single-tree
# write, the merged label, and the missing bring-up route.

def _pair(key: str, verdict: str, at: str, **extra) -> dict:
    """One ledger row with the fields these fixtures need, spelled once."""
    row = {"pattern_key": key, "verdict": verdict, "reason": "fixture grounds",
           "evidence_cmd": "grep -c '^status:' /dev/null || true",
           "occurrences_at_decision": 4, "decided_at": at, "decided_by": "test"}
    row.update(extra)
    return row


def _write_rows(path: Path, rows: list[dict]) -> Path:
    """Create a ledger (or its durable copy) holding exactly `rows`, oldest line first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _lines(path: Path) -> list[str]:
    """The file's raw lines — what the two trees have to agree on, byte for byte."""
    return path.read_text(encoding="utf-8").splitlines()


def test_a_lagged_key_is_labelled_lagged_and_a_relabelled_key_is_not(tmp_path):
    """Clause 3, at the classifier: the shapes, both directions, plus the tie-break.

    `LAGGED` is the class `sync` copies without asking anyone, so the burden sits on it: a
    key qualifies only when the verdicts match AND the live stamp is provably newer. Every
    other difference — a different verdict, a copy stamped *newer* than the ledger (which no
    append-only writer produces), or a date that cannot be parsed — is `DISAGREEING`,
    because an unexplained difference must never be filed where an unattended job will
    overwrite it.
    """
    lag_live = _pair("Bash/timeout", "rejected_unverifiable", "2026-09-27T08:59:12Z")
    lag_durable = _pair("Bash/timeout", "rejected_unverifiable", "2026-09-20T09:00:00Z")
    assert sv.divergence_class(lag_live, lag_durable) == sv.LAGGED

    relabelled = _pair("Read/logic", "rejected_unverifiable", "2026-09-27T08:59:12Z")
    durable_view = _pair("Read/logic", "reviewed_no_skill", "2026-09-20T09:00:00Z")
    assert sv.divergence_class(relabelled, durable_view) == sv.DISAGREEING

    # Both cases below carry the SAME verdict on both sides, each paired with a control
    # that is the same two rows the other way round and LAGGED. Otherwise the verdict check
    # answers them first and the stamp rule — the only thing these assertions are about —
    # is never reached.
    ahead_live = _pair("Bash/err", "noise", "2026-09-20T09:00:00Z")
    ahead_durable = _pair("Bash/err", "noise", "2026-09-27T08:59:12Z")
    assert sv.divergence_class(ahead_live, ahead_durable) == sv.DISAGREEING, (
        "a copy stamped ahead of the ledger is not lag in either direction")
    assert sv.divergence_class(ahead_durable, ahead_live) == sv.LAGGED, (
        "control: the same two rows the other way round ARE lag, so the verdicts agree "
        "here and the stamp rule is what decides")

    unreadable = _pair("Bash/timeout", "noise", "not a date")
    readable_ahead = _pair("Bash/timeout", "noise", "2026-09-20T09:00:00Z")
    readable_behind = _pair("Bash/timeout", "noise", "2026-09-10T09:00:00Z")
    assert sv.divergence_class(unreadable, readable_ahead) == sv.DISAGREEING, (
        "an unparseable stamp cannot be *proved* to be lag, so it is not filed as lag")
    assert sv.divergence_class(readable_ahead, readable_behind) == sv.LAGGED, (
        "control: same verdicts, readable stamps, live ahead — the verdicts agree too")

    # One lagged row cannot be reported under the disagreeing label, or the other way
    # round: both counts come out of one loop over the same two trees.
    report = sv.divergence_report(
        {"Bash/timeout": lag_live, "Read/logic": relabelled},
        {"Bash/timeout": lag_durable, "Read/logic": durable_view},
        tmp_path / "durable.jsonl")
    assert report[sv.LAGGED] == 1 and report[sv.DISAGREEING] == 1, report
    assert report["keys_compared"] == 2, report
    joined = "\n".join(report["lines"])
    assert "LEDGER_MIRROR_LAGGED Bash/timeout" in joined, report
    assert "LEDGER_MIRROR_CONFLICT Read/logic" in joined, report
    assert "LEDGER_MIRROR_CONFLICT Bash/timeout" not in joined, (
        "the lagged key must not also appear under the disagreement label")
    assert "LEDGER_MIRROR_LAGGED Read/logic" not in joined, report
    assert sv.divergence_lines({"Bash/timeout": lag_live}, {"Bash/timeout": lag_live},
                              tmp_path / "durable.jsonl") == []


def test_check_separates_lag_from_disagreement_and_prints_both_counts(
        store, mirror, capsys):
    """Clauses 3 and 4 on the printed surface: #1717's 54-lag/25-disagreement night, small.

    Two lagged keys, one relabelled key, and one the copy never got. Before this change the
    first three printed the same string and there was no denominator at all, which is how a
    third of that night's lines were real disagreements nobody could pick out. Now the
    CONFLICT token fires only on the relabelled key, the two classes are two numbers, and
    those numbers sit against the number of keys compared.
    """
    _write_rows(store, [
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-10T09:00:00Z"),
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-27T08:59:09Z"),
        _pair("Bash/err", "rejected_artifact_class", "2026-09-10T09:00:00Z"),
        _pair("Bash/err", "rejected_artifact_class", "2026-09-27T08:59:10Z"),
        _pair("Read/logic", "reviewed_no_skill", "2026-09-10T09:00:00Z"),
        _pair("Read/logic", "rejected_unverifiable", "2026-09-27T08:59:11Z"),
        _pair("Grep/formatting", "reviewed_no_skill", "2026-09-11T09:00:00Z"),
    ])
    _write_rows(mirror, [
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-10T09:00:00Z"),
        _pair("Bash/err", "rejected_artifact_class", "2026-09-10T09:00:00Z"),
        _pair("Read/logic", "reviewed_no_skill", "2026-09-10T09:00:00Z"),
    ])
    cands = store.parent / "candidates"
    cands.mkdir()

    assert sv.main(["check", "--candidates", str(cands), "--store", str(store)]) == 0
    out = capsys.readouterr().out

    assert out.count("LEDGER_MIRROR_CONFLICT") == 1, out
    assert "LEDGER_MIRROR_CONFLICT Read/logic" in out, out
    assert "live=rejected_unverifiable durable=reviewed_no_skill" in out, out
    assert out.count("LEDGER_MIRROR_LAGGED") == 2, out
    assert "LEDGER_MIRROR_LAGGED Bash/timeout" in out, out
    assert "MIRROR_MISSING Grep/formatting" in out, out
    tally = [ln for ln in out.splitlines() if ln.startswith("ledger_mirror:")]
    assert tally == ["ledger_mirror: keys_compared: 4  lagged: 2  disagreeing: 1  "
                     f"missing: 1  lost: 0  durable: {mirror}"], out
    assert out.splitlines()[-1] == "checked: 0  skipped_by_verdict: 0", \
        "the count line stays last: the nightly parses splitlines()[-1]"


def test_reanchor_lands_the_identical_row_in_both_trees(store, mirror):
    """Clause 1, first half: the single-key re-anchor appends the same bytes to both trees.

    Six of the 79 one-tree rows were written through `reanchor_verdict`; the other 73 came
    from the pass around it, which is the next test. A row that exists in only one tree is
    the exact failure #772 keeps a second copy to prevent, so the assertion is on the lines
    themselves — the two files ending byte-identical is what "the same row" has to mean for
    a restore — and on the decision the re-anchor is not allowed to change.
    """
    genuine = store.parent / "genuine.md"
    genuine.parent.mkdir(parents=True, exist_ok=True)
    genuine.write_text("one signature line that survived the move\n", encoding="utf-8")
    stored_row(store, "Bash/timeout", "grep -c x '/nonexistent/shape/*.md' | head",
               verdict="reviewed_no_skill", reason="hand-authored re-anchor", occurrences=7)
    _write_rows(mirror, [json.loads(ln) for ln in _lines(store)])

    rc = sv.main(["reanchor", "--store", str(store), "--pattern", "Bash/timeout",
                  "--evidence-cmd", declares(f"grep -c 'survived' {genuine}")])

    assert rc == 0, "an authored falsifier that executes is accepted"
    assert len(_lines(store)) == len(_lines(mirror)) == 2, "each tree gained exactly one row"
    assert _lines(store) == _lines(mirror), "the two trees hold different bytes"
    assert sv.load_verdicts(store) == sv.load_verdicts(mirror)
    latest = sv.load_verdicts(store)["Bash/timeout"]
    assert latest["decided_by"] == sv.REANCHOR_DECIDED_BY, latest
    assert latest["verdict"] == "reviewed_no_skill", "a re-anchor keeps the decision"


def test_the_repair_pass_writes_both_trees_for_every_class_it_records(
        store, mirror, tmp_path):
    """Clause 1, second half: fixing `reanchor_verdict` alone would have reached 6 of 79.

    The live-only rows split 46 `root-move-repair` + 27 `root-move-repair-disposal` + 6
    `root-move-repair-reanchor`, so the path under test has to be the shared one — the pass,
    driving all three of its recording buckets in one run: a key whose moved root a
    substitution reaches, one whose dated corpus retention already pruned (disposed to a
    tombstone), and one whose dead path sits inside a script only an authored command can
    reach. The two files are byte-identical when it finishes.
    """
    dead, live = nested_root_pair(tmp_path)
    root_move_row(store, "Bash/timeout", dead, "candidate-x-20260927")   # substitution
    root_move_row(store, "Bash/err", dead, "candidate-x-20260908")       # retention pruned
    stored_row(store, "ledger/evidence_cmd_syntax", f"python3 {dead}/skills/tools/m.py",
               verdict="reviewed_no_skill", reason="guard lives in the tool", occurrences=7)
    authored = store.parent / "reanchors.json"
    # The authored falsifier declares its denominator (#2052): a re-anchor is a freshly
    # written command, so the mandate applies to it, unlike the two commands this pass
    # writes itself. The count is one file — the module it greps.
    authored.write_text(json.dumps({"ledger/evidence_cmd_syntax":
                                    declares(f"grep -c . {live}/skills/tools/m.py")}),
                        encoding="utf-8")
    _write_rows(mirror, [json.loads(ln) for ln in _lines(store)])

    rc, out = run_repair(store, dead, live, "--dispose-unverifiable",
                         "--reanchor-file", str(authored))

    assert rc == 0, out
    assert "repaired: 1  reanchored: 1  disposed: 1" in out, out
    assert _lines(store) == _lines(mirror), out
    assert len(_lines(store)) == 6, "three rows in, three appended, none in one tree only"
    assert sv.MIRROR_NOT_WRITTEN not in out, "a copy was attached, so none was skipped"


def test_a_repair_with_no_durable_copy_names_the_copy_it_could_not_write(
        store, monkeypatch, tmp_path):
    """Clause 2: record anyway, and *say* so — the policy `mirror_for`'s docstring states.

    Deliberately no mirror: a tmp `--store` with `$SKILL_VERDICTS_MIRROR` unset is the
    legitimate state, and `_mirror_target` refuses to push an off-default ledger through the
    production vault copy. Refusing the record instead would lose a decision, which clause 1
    and that docstring both forbid — so the half left standing is the announcement, carrying
    the count of rows that went out single-tree. This is the shape the 2026-09-27 pass ran
    in for all 79 of its appends, and the pass's stdout is where a nightly could have seen
    it and did not.
    """
    dead, live = moved_root(tmp_path)
    root_move_row(store, "Bash/timeout", dead, "candidate-x-20260927")
    monkeypatch.delenv("SKILL_VERDICTS_MIRROR", raising=False)
    assert sv.mirror_for(store) is None, "the fixture must be in the skipped shape"

    rc, out = run_repair(store, dead, live)

    assert rc == 0, out
    assert sv.MIRROR_NOT_WRITTEN in out, out
    assert str(store) in out and str(sv.DEFAULT_MIRROR) in out, out
    assert "rows_this_pass=1" in out, out
    assert len(_lines(store)) == 2, "the live row was still recorded"
    assert out.splitlines()[-1].startswith("repaired: 1"), \
        "the tally line stays last: the nightly parses splitlines()[-1]"


def test_record_verdict_says_which_copy_it_could_not_write(tmp_path, monkeypatch, capfd):
    """Clause 2 at the single writer, not only at the pass: stderr, one row, no exception.

    Every caller that is not `repair` — `record`, `dispose`, a runbook — gets the notice
    from here rather than from a tally it does not have, so the announce lives in
    `record_verdict` itself. The row still lands: the ledger is the live artifact, and an
    unwritable vault costs the second copy, not the decision.
    """
    scratch = tmp_path / "scratch" / "verdicts.jsonl"
    monkeypatch.delenv("SKILL_VERDICTS_MIRROR", raising=False)

    sv.record_verdict(scratch, pattern_key="Bash/timeout", verdict="noise",
                      reason="fixture grounds",
                      evidence_cmd=declares("grep -c '^status:' /dev/null || true"))

    err = capfd.readouterr().err
    assert sv.MIRROR_NOT_WRITTEN in err, err
    assert str(scratch) in err and str(sv.DEFAULT_MIRROR) in err, err
    assert len(_lines(scratch)) == 1, "the live row was recorded anyway"


def test_sync_brings_the_copy_up_to_the_ledger_and_adds_nothing_twice(
        store, mirror, capsys):
    """Clause 5, first half: a route for the 79 rows already on disk, runnable twice.

    No amount of correct future writing puts rows into a copy that is 79 lines behind, so
    #1717 needs a bring-up: it appends the live ledger's latest row per key — byte for byte,
    the copy's own history intact — and the second run finds every key already matching and
    adds 0 lines, which is the property that makes it safe to put in a nightly at all.
    """
    _write_rows(store, [
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-10T09:00:00Z"),
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-27T08:59:09Z"),
        _pair("Read/logic", "reviewed_no_skill", "2026-09-10T09:00:00Z"),
        _pair("Read/logic", "rejected_unverifiable", "2026-09-27T08:59:11Z"),
        _pair("Grep/formatting", "reviewed_no_skill", "2026-09-11T09:00:00Z"),
    ])
    _write_rows(mirror, [
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-10T09:00:00Z"),
        _pair("Read/logic", "reviewed_no_skill", "2026-09-10T09:00:00Z"),
    ])
    copy_before = _lines(mirror)

    rc = sv.main(["sync", "--store", str(store)])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert out.count("SYNCED ") == 3, out
    assert "SYNCED Bash/timeout :: rejected_artifact_class LAGGED" in out, out
    assert "SYNCED Read/logic :: rejected_unverifiable DISAGREEING" in out, out
    assert "SYNCED Grep/formatting :: reviewed_no_skill MIRROR_MISSING" in out, out
    assert _synced_line(out).startswith(
        "synced: 3  lagged: 1  disagreeing: 1  missing: 1  held: 0"), out
    assert ("ledger_mirror: keys_compared: 3  lagged: 0  disagreeing: 0  missing: 0  "
            "lost: 0") in out, out

    after = _lines(mirror)
    assert after[:len(copy_before)] == copy_before, \
        "the copy stays append-only: its own history survives the bring-up"
    assert [json.loads(ln) for ln in after[len(copy_before):]] == [
        sv.load_verdicts(store)[k]
        for k in ("Bash/timeout", "Grep/formatting", "Read/logic")], \
        "what landed is the live ledger's latest row per key, unmodified"
    assert sv.load_verdicts(store) == sv.load_verdicts(mirror)

    rc2 = sv.main(["sync", "--store", str(store)])
    out2 = capsys.readouterr().out
    assert rc2 == 0, out2
    assert "SYNCED " not in out2, out2
    assert _synced_line(out2).startswith(
        "synced: 0  lagged: 0  disagreeing: 0  missing: 0  held: 0"), out2
    assert _lines(mirror) == after, "a second run added lines"


def test_sync_holds_a_relabelled_key_when_told_to(store, mirror, capsys):
    """Clause 5's `--hold-disagreements`: lag is copied, a relabel is left for a human.

    #1717's owed ruling is whether the 25 keys the repair pass relabelled to
    `rejected_unverifiable` should become the durable answer unreviewed. This flag is what
    makes that ruling executable rather than a re-derivation: the same run still closes the
    lag gap, and the relabelled key is printed with both verdicts and left untouched.
    """
    _write_rows(store, [
        _pair("Bash/timeout", "noise", "2026-09-10T09:00:00Z"),
        _pair("Bash/timeout", "noise", "2026-09-27T08:59:09Z"),
        _pair("Read/logic", "rejected_unverifiable", "2026-09-27T08:59:11Z"),
    ])
    _write_rows(mirror, [
        _pair("Bash/timeout", "noise", "2026-09-10T09:00:00Z"),
        _pair("Read/logic", "reviewed_no_skill", "2026-09-10T09:00:00Z"),
    ])
    before = _lines(mirror)

    rc = sv.main(["sync", "--store", str(store), "--hold-disagreements"])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "SYNCED Bash/timeout :: noise LAGGED" in out, out
    assert "SYNC_HELD Read/logic :: live=rejected_unverifiable " \
           "durable=reviewed_no_skill" in out, out
    assert _synced_line(out).startswith(
        "synced: 1  lagged: 1  disagreeing: 0  missing: 0  held: 1"), out
    assert len(_lines(mirror)) == len(before) + 1, out
    assert sv.load_verdicts(mirror)["Read/logic"]["verdict"] == "reviewed_no_skill", \
        "the held key keeps the verdict the human has not yet ruled on"


def test_sync_refuses_to_write_the_production_copy_for_an_off_default_ledger(
        store, monkeypatch, tmp_path, capsys):
    """Clause 5, second half: a `--store` never reaches the vault copy through a sync.

    The ledger here is a tmp file and `$SKILL_VERDICTS_MIRROR` is unset, so `_mirror_target`
    has no durable copy for it — the guard that exists precisely so a scratch run cannot
    push scratch decisions into `~/obsidian/memory/skill-verdicts/`. A sync that wrote them
    there anyway would be the loudest possible way to defeat it, so it appends nothing and
    exits 1. The pair to this test is the one above: name a copy with
    `$SKILL_VERDICTS_MIRROR` and the same run writes it.
    """
    _write_rows(store, [_pair("Bash/timeout", "noise", "2026-09-20T09:00:00Z")])
    monkeypatch.delenv("SKILL_VERDICTS_MIRROR", raising=False)
    assert sv.mirror_for(store) is None, "the fixture must be in the refused shape"
    # The production copy re-aimed at a tmp file, so "nothing was written to it" is a claim
    # about a file this test owns: asserted against the real path it says nothing at all
    # whenever the vault copy happens not to exist, which is the state a fresh round home
    # runs in.
    pretend_vault = tmp_path / "production-copy.jsonl"
    monkeypatch.setattr(sv, "DEFAULT_MIRROR", pretend_vault)

    rc = sv.main(["sync", "--store", str(store)])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "SYNC_REFUSED" in out, out
    assert str(store) in out and "$SKILL_VERDICTS_MIRROR" in out, out
    assert str(pretend_vault) in out, out
    assert out.splitlines()[-1] == "synced: 0  refused: 1  durable: unresolved", out
    assert len(_lines(store)) == 1, "a refusal appended to the ledger it was given"
    assert not pretend_vault.exists(), (
        "the refusal created the copy it refused to write")


def test_the_shipped_cli_keeps_its_tally_last_and_names_both_classes(tmp_path, mirror):
    """Seam, not clause: the nightly parses the CHILD's stdout, and that surface changed.

    Every other test in this section calls `sv.main` in-process, so none proves what the
    shipped module prints through a real interpreter in a real environment — which is the
    boundary the runbook reads, and the one whose `splitlines()[-1]` contract the new
    `ledger_mirror:` tally could have broken. Asserted across it: the two classes are two
    numbers, the counts line is still the last line, `sync`'s own tally is well formed, and
    the same parse after a sync reports `disagreeing: 0` with no CONFLICT line left.
    """
    store = tmp_path / "cli" / "verdicts.jsonl"
    _write_rows(store, [
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-10T09:00:00Z"),
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-27T08:59:09Z"),
        _pair("Read/logic", "rejected_unverifiable", "2026-09-27T08:59:11Z"),
    ])
    _write_rows(mirror, [
        _pair("Bash/timeout", "rejected_artifact_class", "2026-09-10T09:00:00Z"),
        _pair("Read/logic", "reviewed_no_skill", "2026-09-10T09:00:00Z"),
    ])
    cands = tmp_path / "candidates"
    cands.mkdir()
    env = dict(os.environ, SKILL_VERDICTS_MIRROR=str(mirror))
    run = lambda *args: subprocess.run(
        [sys.executable, "-m", "scripts.skill_verdicts", *args],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=120)

    checked = run("check", "--candidates", str(cands), "--store", str(store))
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.count("LEDGER_MIRROR_CONFLICT") == 1, checked.stdout
    assert checked.stdout.count("LEDGER_MIRROR_LAGGED") == 1, checked.stdout
    assert ("ledger_mirror: keys_compared: 2  lagged: 1  disagreeing: 1  missing: 0  "
            f"lost: 0  durable: {mirror}") in checked.stdout, checked.stdout
    assert checked.stdout.splitlines()[-1] == "checked: 0  skipped_by_verdict: 0", \
        "the tally must not become the last line: the runbook parses splitlines()[-1]"

    synced = run("sync", "--store", str(store))
    assert synced.returncode == 0, synced.stderr
    assert _synced_line(synced.stdout).startswith(
        "synced: 2  lagged: 1  disagreeing: 1  missing: 0  held: 0"), synced.stdout

    after = run("check", "--candidates", str(cands), "--store", str(store))
    assert after.returncode == 0, after.stderr
    assert "LEDGER_MIRROR_CONFLICT" not in after.stdout, after.stdout
    assert "LEDGER_MIRROR_LAGGED" not in after.stdout, after.stdout
    assert ("ledger_mirror: keys_compared: 2  lagged: 0  disagreeing: 0  missing: 0  "
            "lost: 0") in after.stdout, after.stdout
    assert after.stdout.splitlines()[-1] == "checked: 0  skipped_by_verdict: 0", after.stdout


def _synced_line(out: str) -> str:
    """The `sync` tally line, wherever it sits among the findings above it."""
    return [ln for ln in out.splitlines() if ln.startswith("synced: ")][0]


def test_the_default_rewrites_point_at_the_live_data_root_not_the_code_tree():
    """Clause 2's mechanism: what a correction is re-pointed *at*.

    Both dead spellings the live ledger holds its commands in — the absolute one and the
    tilde one — have to map onto the pipeline dir `app.paths` resolves today, and the old
    one must not be inside the data root. Asserted structurally rather than as a literal home path: the same rule
    `tests/test_no_runtime_paths_in_code.py` enforces is what forbids spelling the dead
    root into this file's own source.

    Containment is decided with `Path.is_relative_to`, not `str.startswith`, and that is
    the whole point of the last line: the data root is a *sibling* whose name starts with
    the code tree's own (`~/lloyd` and `~/lloyd-data`), so a string prefix reports the two
    as nested when they are not, and the substitution would look like a no-op wherever the
    two trees are not exactly `/home/<user>/lloyd`. Under the gate's round home — where
    `app.paths` anchors to `<round>/home/lloyd` and the data root to
    `<round>/home/lloyd-data` — a prefix test fails on a change that is correct.
    """
    pairs = dict(sv.dead_root_spellings())
    assert len(pairs) == 2, pairs
    assert all(new == str(sv.PIPELINE_DIR) for new in pairs.values()), pairs
    dead = [old for old in pairs if Path(old).is_relative_to(sv.LIVE_CHECKOUT)]
    assert len(dead) == 1, pairs
    assert not sv.PIPELINE_DIR.is_relative_to(sv.LIVE_CHECKOUT), (
        "app.paths no longer resolves the pipeline dir outside the code tree")


def test_repair_repoints_a_falsifier_the_moved_root_left_behind(tmp_path):
    """Clause 2: the root-substitution case, end to end through the CLI.

    One `repair` run must append a row whose `evidence_cmd` names the live root, keep the
    superseded row's `verdict` and `reason` byte-identical, leave that superseded row on
    disk untouched (#530 append-only), and hand `audit` a ledger with nothing left to
    report for the key.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    root_move_row(store, "Bash/network", dead, "candidate-x-20260927")
    before = store.read_text()

    rc, out = run_repair(store, dead, live)

    assert rc == 0, out
    assert out.splitlines()[-1].startswith("repaired: 1  reanchored: 0  disposed: 0"), out
    latest = sv.load_verdicts(store)["Bash/network"]
    assert latest["evidence_cmd"] == (
        f"grep -c '^status:' {live}/skills/candidates/candidate-x-20260927.md"), latest
    assert latest["verdict"] == "reviewed_no_skill", latest
    assert latest["reason"] == "installed skill already owns this shape", latest
    assert latest["decided_by"] == sv.REPAIR_DECIDED_BY, latest
    assert dead not in json.dumps(latest), "the repaired row still names the dead root"
    # Append-only: the row it supersedes is still there, byte for byte.
    lines = store.read_text().splitlines()
    assert len(lines) == 2, lines
    assert lines[0] + "\n" == before, "a repair edited the row it supersedes (#530)"

    audit_rc, audit_out = run_audit(store)
    assert audit_out.splitlines()[-1] == "keys: 1 unrunnable: 0", audit_out
    assert audit_rc == 0, audit_out


def test_repair_never_calls_a_command_repaired_because_of_its_exit_code(tmp_path):
    """The trap #1588 names, pinned: a rewritten command is judged by the classifier.

    Its first pass judged each rewritten command by `rc in (0,1)` and reported roughly
    twice the number of repaired keys that the classifier agreed with. This fixture is one of the shapes
    that lie that way: a `grep` of a file retention pruned, piped into `head`. The
    substitution runs, the pipeline exits **0**, and the thing the check measures is
    gone. So the node asserts the clean exit code first (the premise a naive pass would
    believe), then that `falsifier_repair` classifies it `NEEDS_RERECORD` and appends
    nothing — which is only true because the classifier reads stderr, not `$?`.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    # Its own command, not the shared helper's: the trailing `| head` is the whole trap,
    # since it is what hides grep's exit 2 behind a pipeline that exits 0.
    row = stored_row(store, "Bash/network",
                     f"grep -c '^status:' {dead}/skills/candidates/candidate-x-20260908.md"
                     " | head",
                     reason="installed skill already owns this shape", occurrences=4)
    rewritten = row["evidence_cmd"].replace(dead, live)

    naive = subprocess.run(["bash", "-c", rewritten], capture_output=True, text=True)
    assert naive.returncode == 0, "fixture broke: this shape is only a trap if rc looks clean"

    cls, new_cmd, detail = sv.falsifier_repair(row, rewrites=[(dead, live)])
    assert cls == sv.NEEDS_RERECORD, (cls, new_cmd, detail)
    assert new_cmd == "", "a key that needs re-recording must not be handed a command"
    assert "No such file or directory" in detail, detail

    rc, out = run_repair(store, dead, live)
    assert rc == 1, out
    assert out.splitlines()[-1].startswith("repaired: 0  reanchored: 0  disposed: 0  needs_rerecord: 1"), out
    assert store.read_text().splitlines()[0] == json.dumps(row, ensure_ascii=False), (
        "a key the rewrite does not save must be left exactly as it was")


def test_a_repair_correction_carries_the_reopen_baseline_and_appends_over_the_old_row(tmp_path):
    """Clause 2, third half: `occurrences_at_decision` travels, and a second run is a no-op.

    The count at decision is the baseline the >10x growth reopen measures against
    (#736 clause 1), and `carried_forward_occurrences` only runs when the caller passes
    no count at all — so a repair that passed its own `0` would silently disarm that
    reopen for the key while looking like a routine append. 12 is carried here; the
    second run proves the pass is idempotent rather than an appending machine.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    root_move_row(store, "Bash/network", dead, "candidate-x-20260927", occurrences=12)

    rc, out = run_repair(store, dead, live)
    assert rc == 0, out
    assert sv.load_verdicts(store)["Bash/network"]["occurrences_at_decision"] == 12, out

    rc2, out2 = run_repair(store, dead, live)
    assert rc2 == 0, out2
    assert out2.splitlines()[-1].startswith("repaired: 0  reanchored: 0"), out2
    assert len(store.read_text().splitlines()) == 2, "the second run appended again"


def test_repair_and_audit_answer_one_ledger_with_one_classifier(tmp_path):
    """One ledger, three causes, the same numbers from both surfaces.

    #1588's whole objection to the exit-code pass is that two tools would report two
    numbers and one would be believed. Here `repair` classifies a moved root, a pruned
    dated input and a field holding prose; its tally must account for all three keys, and
    the keys it *refuses* to touch must be exactly what `audit` then calls UNRUNNABLE —
    one classifier deciding both answers, or this fails.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    root_move_row(store, "key/moved", dead, "candidate-x-20260927")
    root_move_row(store, "key/pruned", dead, "candidate-x-20260908")
    stored_row(store, "key/prose", 'echo "unbalanced', reason="a sentence, not a command")

    rc, out = run_repair(store, dead, live)

    assert rc == 1, out
    assert out.splitlines()[-1] == ("repaired: 1  reanchored: 0  disposed: 0  "
                                    "needs_rerecord: 1  unparseable: 1  refused: 0  "
                                    "keys: 3"), out
    assert "NEEDS_RERECORD key/pruned" in out, out
    assert "UNPARSEABLE key/prose" in out, out

    audit_rc, audit_out = run_audit(store)
    reported = {ln.split(" :: ")[0].removeprefix("UNRUNNABLE ")
                for ln in audit_out.splitlines() if ln.startswith("UNRUNNABLE ")}
    assert reported == {"key/pruned", "key/prose"}, audit_out
    assert audit_out.splitlines()[-1] == "keys: 3 unrunnable: 2", audit_out
    assert audit_rc == 1, audit_out


def test_a_disposal_relabels_a_terminal_verdict_without_unblocking_anything(tmp_path):
    """Clause 4: the honest record for a verdict whose corpus retention deleted.

    `rejected_unverifiable` is the disposal #1588 offers, and the reason it is safe for a
    machine to make it is that the word is *terminal*, like the one it supersedes: the
    ledger stops claiming grounds it cannot re-run, and the same candidate stays blocked.
    Asserted across that boundary with the miner's own lookup, `terminal_verdict` — a
    verdict that stops binding is the release nobody authorised. The appended check is a
    count of the pruned corpus, so the disposal is falsifiable (a restore makes it
    nonzero) rather than an assertion, and it exits 0, which is what takes the key out of
    `audit`'s dead set honestly instead of by rewriting a path.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    root_move_row(store, "seq-2-x-y", dead, "candidate-x-20260908", occurrences=5)
    assert sv.terminal_verdict("seq-2-x-y", store=store) is not None, "fixture: not blocking"

    rc, out = run_repair(store, dead, live, "--dispose-unverifiable")

    assert rc == 0, out
    assert out.splitlines()[-1].startswith("repaired: 0  reanchored: 0  disposed: 1"), out
    latest = sv.load_verdicts(store)["seq-2-x-y"]
    assert latest["verdict"] == sv.UNVERIFIABLE_VERDICT, latest
    assert sv.is_terminal(latest), "a disposal must not unblock a blocked pattern"
    assert sv.terminal_verdict("seq-2-x-y", store=store) is not None, (
        "the candidate stopped being blocked: a disposal released a verdict")
    assert "installed skill already owns this shape" in latest["reason"], latest
    assert latest["decided_by"] == sv.DISPOSE_DECIDED_BY, latest
    assert latest["occurrences_at_decision"] == 5, latest

    ran = subprocess.run(["bash", "-c", latest["evidence_cmd"]],
                         capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    assert "dated_corpus_files=0" in ran.stdout, ran.stdout
    assert "No such file or directory" not in ran.stderr, ran.stderr
    assert sv.evidence_cmd_status(latest)[0] != sv.UNRUNNABLE, latest

    audit_rc, audit_out = run_audit(store)
    assert audit_out.splitlines()[-1] == "keys: 1 unrunnable: 0", audit_out
    assert audit_rc == 0, audit_out


def test_a_disposal_never_mints_a_block_a_non_terminal_verdict_never_had(tmp_path):
    """The other half of the terminality rule, and the only dangerous direction.

    `proposed` and `below_threshold` do not block a candidate (#736's reopen rules,
    `test_non_terminal_verdict_does_not_block`). If a repair pass over bookkeeping
    relabelled one to `rejected_unverifiable`, it would author a rejection that was never
    decided — a skill candidate silently suppressed every night with no diff anywhere. So
    the verdict stays, the key is disposed only in the sense of being re-checked, and
    `terminal_verdict` must go on answering None for it before and after.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    root_move_row(store, "seq-3-a-b-c", dead, "candidate-x-20260908", verdict="proposed")

    assert sv.terminal_verdict("seq-3-a-b-c", store=store) is None
    verdict, kept = sv.disposal_verdict({"verdict": "proposed"})
    assert verdict == "proposed", kept
    assert not sv.is_terminal({"verdict": verdict}), kept

    rc, out = run_repair(store, dead, live, "--dispose-unverifiable")
    assert rc == 0, out
    latest = sv.load_verdicts(store)["seq-3-a-b-c"]
    assert latest["verdict"] == "proposed", latest
    assert sv.terminal_verdict("seq-3-a-b-c", store=store) is None, (
        "the repair minted a block the ledger never held")


def test_dry_run_reports_the_repairs_it_does_not_append(tmp_path):
    """`--dry-run` promises "append nothing", for both classes, and means it.

    Caught on the live ledger while running the very pass this subcommand exists for: the
    first draft honoured `dry_run` only on the disposal branch, so a rehearsal of the
    machine repair appended its 46 correction rows and reported them as if nothing had
    been written — a preview of a runtime-data write that performed it. A rehearsal that
    writes is worse than no flag, because the next invocation of the same command is then
    a second pass over data that was already changed. So the file is asserted byte for
    byte unchanged across a dry run that still reports one repair and one disposal.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    root_move_row(store, "key/moved", dead, "candidate-x-20260927")
    root_move_row(store, "key/pruned", dead, "candidate-x-20260908")
    before = store.read_bytes()

    rc, out = run_repair(store, dead, live, "--dry-run", "--dispose-unverifiable")

    assert rc == 1, out
    assert out.splitlines()[-1].startswith("repaired: 1  reanchored: 0  disposed: 1"), out
    assert "(dry-run)" in out, out
    assert store.read_bytes() == before, "a dry run wrote to the ledger"

    rc2, out2 = run_repair(store, dead, live, "--dispose-unverifiable")
    assert rc2 == 0, out2
    assert out2.splitlines()[-1].startswith("repaired: 1  reanchored: 0  disposed: 1"), out2
    assert len(store.read_text().splitlines()) == 4, "the real run did not follow the rehearsal"


# ── #1588 clause 3: the two shapes a substitution cannot reach ───────────────

def nested_root_pair(tmp_path, key_stem: str = "candidate-x-20260927"):
    """A command whose dead path lives one level down, in the script it invokes.

    This is the shape the item's own heredoc could not repair and the reason 15 of its
    45 turned out to be repairable after all: substituting both spellings *in the stored
    command* succeeds, the script then runs, and the traceback it prints names the OLD
    root — because that path is in the script's own source, where a rewrite of the
    command cannot reach. The script is real and lives in the live tree, so the rewritten
    command genuinely executes: only what it reports about is gone.
    """
    dead, live = moved_root(tmp_path)
    tool_dir = Path(live) / "skills" / "tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "m.py").write_text(
        "import sys\n"
        f"try:\n    open({dead!r} + '/data/x.jsonl')\n"
        "except OSError as exc:\n    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)\n"
        "sys.exit(1)\n", encoding="utf-8")
    return dead, live


def test_a_nested_root_falsifier_is_classified_as_the_shape_a_rewrite_cannot_reach(tmp_path):
    """The nested-string shape, decided from what the rewritten command *prints*.

    `falsifier_repair` must not call this repaired — the substitution ran, the script
    ran, and the dead root came back in the detail. `rerecord_shape` then has to name it
    `nested_root` rather than `dated_corpus`, because the two have different owners: one
    is an edit to a script, the other is a corpus retention deleted. Both assertions are
    about the classifier's judgement, not an exit code.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = nested_root_pair(tmp_path)
    row = stored_row(store, "ledger/evidence_cmd_syntax",
                     f"python3 {dead}/skills/tools/m.py",
                     reason="falsifier prints a missing input", occurrences=3)

    cls, new_cmd, detail = sv.falsifier_repair(row, rewrites=[(dead, live)])
    assert cls == sv.NEEDS_RERECORD, (cls, new_cmd, detail)
    assert dead in detail, detail
    assert sv.rerecord_shape(detail, rewrites=[(dead, live)]) == sv.NESTED_ROOT, detail


def test_a_quoted_glob_is_classified_as_the_shape_no_shell_ever_expanded(tmp_path):
    """The quoted-glob shape: a `*` inside quotes is a filename, not a wildcard.

    Rewriting the root leaves the command grepping for a path with a literal asterisk in
    it, so the detail still says the file is missing and still carries the `*` — which is
    the tell. Named separately from `nested_root` because the edit is a re-quote, and
    named separately from `dated_corpus` because re-quoting might yet find files. Here it
    must not: the glob is offered over a directory that holds none of the dated inputs, so
    `rerecord_shape` still says the corpus is what is missing, and a pass that re-quoted
    it would be measuring nothing.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    row = stored_row(store, "automod_vault_land/validation",
                     f"grep -c x '{dead}/trajectories/2026-09-1*.jsonl' | head",
                     reason="falsifier greps a quoted glob", occurrences=2)

    cls, _new, detail = sv.falsifier_repair(row, rewrites=[(dead, live)])
    assert cls == sv.NEEDS_RERECORD, (cls, detail)
    assert "*" in detail, detail
    assert sv.rerecord_shape(detail, rewrites=[(dead, live)]) == sv.QUOTED_GLOB, detail


def test_the_pass_appends_an_authored_reanchor_and_keeps_the_decision(tmp_path):
    """Clause 3's mechanism, shipped: a hand-authored falsifier reaches the ledger.

    Through `repair --reanchor-file`, so the command that fixes a nested-root key is code
    in this repo, not a script run outside it. The appended row must carry the superseded
    verdict, reason and `occurrences_at_decision` unchanged, the re-anchor provenance
    exactly, and a command the classifier accepts; and the pass's tally has to count it,
    not leave it in `needs_rerecord`.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = nested_root_pair(tmp_path)
    root = stored_row(store, "ledger/evidence_cmd_syntax", f"python3 {dead}/skills/tools/m.py",
                      verdict="reviewed_no_skill", reason="guard lives in the tool",
                      occurrences=7)
    authored = tmp_path / "reanchors.json"
    authored.write_text(json.dumps(
        {"ledger/evidence_cmd_syntax": declares(f"grep -c . {live}/skills/tools/m.py")}),
        encoding="utf-8")
    before = store.read_text()

    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = sv.main(["repair", "--store", str(store), "--rewrite", dead, live,
                      "--reanchor-file", str(authored)])
    out = buf.getvalue()

    assert rc == 0, out
    assert "REANCHORED ledger/evidence_cmd_syntax" in out, out
    assert out.splitlines()[-1].startswith("repaired: 0  reanchored: 1"), out
    latest = sv.load_verdicts(store)["ledger/evidence_cmd_syntax"]
    assert latest["evidence_cmd"] == json.loads(authored.read_text())["ledger/evidence_cmd_syntax"], latest
    assert latest["verdict"] == root["verdict"] == "reviewed_no_skill", latest
    assert latest["reason"] == root["reason"] == "guard lives in the tool", latest
    assert latest["occurrences_at_decision"] == 7, latest
    assert latest["decided_by"] == sv.REANCHOR_DECIDED_BY, latest
    assert dead not in json.dumps(latest), latest
    assert store.read_text().startswith(before), "the superseded row was edited (#530)"
    assert sv.evidence_cmd_status(latest)[0] != sv.UNRUNNABLE, latest

    audit_rc, audit_out = run_audit(store)
    assert audit_out.splitlines()[-1] == "keys: 1 unrunnable: 0", audit_out
    assert audit_rc == 0, audit_out


def test_a_reanchor_refuses_a_command_that_is_itself_unrunnable(tmp_path):
    """The #1586 hazard, closed at the point a human hand is involved.

    `record_verdict` refuses only a command that prints *nothing*, so a probe whose sole
    output is its own traceback would be accepted and would arrive in next night's
    `audit` as fresh damage from this pass. `reanchor_verdict` therefore classifies the
    authored command first and refuses it here — and the ledger is asserted unchanged,
    because a refused re-anchor that still appended would be worse than no guard: the key
    would look repaired while its check is dead on arrival.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    root_move_row(store, "Bash/network", dead, "candidate-x-20260927")
    before = store.read_text()

    import contextlib
    import io
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = sv.main(["reanchor", "--store", str(store), "--pattern", "Bash/network",
                      "--evidence-cmd", f"cat {dead}/skills/candidates/gone.md"])

    assert rc == 2, (rc, buf.getvalue(), err.getvalue())
    assert "unrunnable" in err.getvalue(), err.getvalue()
    assert store.read_text() == before, "a refused re-anchor still wrote a row"


def test_a_pinned_input_with_shell_metacharacters_is_reported_not_disposed(tmp_path):
    """The disposal screen's refusal branch, which the green half cannot show.

    A tombstone embeds the pruned path inside a command the pass authors itself, so a path
    carrying a quote, a `$`, a backtick or a backslash would be spliced into a check that
    then runs through `record_verdict`'s `run_evidence` — inside the only durable copy of
    the ledger. The pass refuses rather than quotes, so the key comes back as work. The
    fixture single-quotes the path so bash hands the `$` through literally, which is what
    puts it in the detail the screen reads; the dead root stays in the path too, because
    `falsifier_repair` takes no position on a command that never named it.
    """
    store = tmp_path / "verdicts.jsonl"
    dead, live = moved_root(tmp_path)
    stored_row(store, "Bash/network",
               f"ls -1 '{dead}/skills/candidates/candidate-$pruned-20260908.md' | head",
               reason="installed skill already owns this shape")

    rc, out = run_repair(store, dead, live, "--dispose-unverifiable")

    assert rc == 1, out
    assert "NEEDS_RERECORD" in out, out
    assert "DISPOSED" not in out, out
    assert len(store.read_text().splitlines()) == 1, "a screened key was still written"


# ── the surfaces the prose names (#1795) ─────────────────────────────────────
#
# #718 closed the store half of this: `verdicts.jsonl` exists, is mirrored, and both
# runbooks mandate read-back. What survived is the description — six lines across four
# surfaces pointing a reader at a markdown review log in `reviews/` as the readable view
# of the ledger, and the one sentence that read like a correction
# (`nightly-skill-consolidation/SKILL.md:75`) was itself the sentence asserting the file
# still exists. It does not, and no writer can produce it: `record` appends to exactly
# two destinations, `DEFAULT_STORE` and `DEFAULT_MIRROR`, both JSONL.
# `eval/iv/recovered-2026-08-22..09-21.jsonl:197` preserves a 2026-09-09 `ls` of
# `_pipeline/skills/reviews/` in which that markdown was the *sole* file and the JSONL
# was unreadable — the two surfaces swapped, and the prose kept describing the retired
# one. These nodes keep it swapped the other way, and obey the rail themselves: the
# file's name does not appear anywhere below, because a dead path quoted even to be
# denied is a path a later grep-and-follow run walks.

VAULT_SKILLS = Path.home() / "obsidian" / "skills"
#: The live ledger, which lives under the data root's `_pipeline/` (`PIPELINE_DIR`).
LIVE_LEDGER_REL = "_pipeline/skills/reviews/verdicts.jsonl"
PHANTOM = "REVIEW-" + "LOG"

def _read(p: Path) -> str:
    assert p.is_file(), f"{p} is not in the tree; a missing file passes every absence below"
    return p.read_text(encoding="utf-8")


def test_no_surface_names_a_markdown_review_log_as_reading_material():
    """Clauses 1-4's shared rail: nothing points a reader at a markdown ledger.

    The acceptance grep is over *tracked* files and the vault's skills, so it is run
    here rather than trusted from a shell: a phantom path quoted even to be denied is a
    path a later grep-and-follow run walks, which is how #718's "demoted" sentence left
    two live skills still sending #57 to a file that is not there. `eval/iv` is excluded
    — it is the archived transcript that *proves* the file once existed, and rewriting a
    recovered session would destroy the evidence.
    """
    hits = subprocess.run(
        ["git", "grep", "-l", PHANTOM, "--", ":!eval/iv", ":!*__pycache__*"],
        cwd=_ROOT, capture_output=True, text=True)
    assert hits.returncode in (0, 1), hits.stderr
    assert not [f for f in hits.stdout.split() if f.strip()], (
        "a tracked file still names the retired markdown ledger: " + hits.stdout)

    skills = subprocess.run(
        ["grep", "-rl", PHANTOM, str(VAULT_SKILLS)],
        capture_output=True, text=True)
    assert skills.stdout.strip() == "", skills.stdout
    for slug in ("nightly-skill-consolidation", "trajectory-skill-mining"):
        text = _read(VAULT_SKILLS / slug / "SKILL.md")
        assert "human-readable" not in text.lower(), (
            f"{slug} still advertises a human-readable markdown ledger")
    assert "human-readable" not in _read(_ROOT / "scripts/skill_verdicts.py").lower()


def test_the_skills_name_the_two_jsonl_surfaces_that_exist():
    """Clauses 1 and 3's positive half: each skill points at the live JSONL, the vault
    mirror, and the README that is actually the prose of this store.

    Absence alone would be satisfied by deleting the Key-paths bullet outright, which
    would leave the next run with no ledger to read at all.
    """
    cons = _read(VAULT_SKILLS / "nightly-skill-consolidation" / "SKILL.md")
    mine = _read(VAULT_SKILLS / "trajectory-skill-mining" / "SKILL.md")
    for text, label in ((cons, "nightly-skill-consolidation"),
                        (mine, "trajectory-skill-mining")):
        assert LIVE_LEDGER_REL in text, f"{label} lost the live ledger"
        assert "memory/skill-verdicts/verdicts.jsonl" in text, (
            f"{label} does not name the durable mirror")
        assert "memory/skill-verdicts/README.md" in text, (
            f"{label} names no prose surface for a human reader")

    # The three named paths are the three files that exist — a doc naming a fourth
    # surface, or a surface that moved, is the same class of bug from the other side.
    assert (production_data_root() / LIVE_LEDGER_REL).is_file()
    assert (Path.home() / "obsidian/memory/skill-verdicts/verdicts.jsonl").is_file()
    assert (Path.home() / "obsidian/memory/skill-verdicts/README.md").is_file()
    assert not list((production_data_root() / "_pipeline/skills/reviews").glob("*.md")), (
        "a markdown file appeared in reviews/, so 'no markdown ledger exists' is now "
        "the false half of this sentence and the prose needs the generator decision")


def test_the_09_06_rejections_are_pointed_at_the_rows_that_carry_them():
    """Clause 2: the Phase 0 citation points at the rows carrying those verdicts, and
    the rows it promises are there, are terminal, and are dated 09-10 — never 09-06.

    The item this fixes claimed the rejections were *lost*; they are not. The 16
    tool/error rows of the 2026-09-10 pass cover the 15 pattern keys the 09-06 batch
    became, and all 15 carry a `TERMINAL_VERDICTS` value, so Phase 0's SKIP still fires
    for every one. Had the reword said "gone", the next run would re-adjudicate 15
    patterns from scratch — the cost #530 exists to remove. The count is pinned from the
    ledger side too: a passage claiming a key the ledger does not carry would tell the
    next run to SKIP on a pattern nothing ever decided.
    """
    import scripts.skill_verdicts as sv  # noqa: PLC0415 - constant under test

    cons = _read(VAULT_SKILLS / "nightly-skill-consolidation" / "SKILL.md")
    passage = cons[cons.index("Phase 0"):cons.index("**0.0")]
    TERMINAL = sv.TERMINAL_VERDICTS

    ledger = production_data_root() / LIVE_LEDGER_REL
    rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    d10 = [r for r in rows if r["decided_at"].startswith("2026-09-10")]
    tool = [r for r in d10 if not r["pattern_key"].startswith("seq-")]
    keys = sorted({r["pattern_key"] for r in tool})
    assert len(d10) == 21 and len(tool) == 16 and len(keys) == 15, (
        len(d10), len(tool), len(keys))
    assert min(r["decided_at"] for r in rows) == "2026-09-10T01:50:37Z"
    assert not [r for r in rows if r["decided_at"].startswith("2026-09-06")], (
        "the ledger gained 09-06 rows, so the passage's 'provenance is gone' is false")

    # The passage points, in words the reader can act on: the file, the date, the
    # timestamp, every one of the 15 keys, and what was lost.
    assert "verdicts.jsonl" in passage, (
        "the passage no longer names the file the verdicts live in: " + passage)
    assert "2026-09-10" in passage and "01:50:37" in passage, passage
    assert "21 rows" in passage and "16" in passage and "15 keys" in passage, passage
    for k in keys:
        assert k in passage, f"{k} is in the ledger but not named by the passage"
        latest = max((r for r in tool if r["pattern_key"] == k),
                     key=lambda r: r["decided_at"])
        assert latest["verdict"] in TERMINAL, (k, latest["verdict"])
    assert "provenance" in passage.lower(), (
        "the passage must say what was lost, not only what survived")


def test_the_module_describes_its_own_write_surfaces(tmp_path):
    """Clause 4: the docstring names the two paths the two store constants actually
    point at, and the comment above `DEFAULT_STORE` no longer claims a third.

    This is the join, not a word search. The prose is a claim about what `record` opens
    (`DEFAULT_STORE`, `DEFAULT_MIRROR`), so each backticked path in the docstring is
    compared to the constant that writes it — a docstring that drifts off the constant
    fails here even if the sentence still reads well. And `record` is run for real into
    a tmp pair to prove those two destinations are the only places a row lands, which is
    what makes a docstring naming a markdown log a claim about an append that does not
    exist.
    """
    import scripts.skill_verdicts as sv  # noqa: PLC0415 - module under test

    src = _read(_ROOT / "scripts/skill_verdicts.py")
    doc = src.split('"""')[1]

    live_rel = "_pipeline/skills/reviews/verdicts.jsonl"
    assert f"`{live_rel}`" in doc, doc[:700]
    assert str(sv.DEFAULT_STORE).endswith("skills/reviews/verdicts.jsonl"), sv.DEFAULT_STORE
    assert str(sv.DEFAULT_STORE).endswith(str(sv.PIPELINE_DIR / "skills/reviews"
                                               / "verdicts.jsonl"))
    assert "`~/obsidian/memory/skill-verdicts/verdicts.jsonl`" in doc, doc[:700]
    assert sv.DEFAULT_MIRROR == (Path.home() / "obsidian" / "memory" / "skill-verdicts"
                                 / "verdicts.jsonl"), sv.DEFAULT_MIRROR

    comment = src[src.rindex("#", 0, src.index("DEFAULT_STORE =")):
                  src.index("DEFAULT_STORE =")]
    named = [tok for tok in comment.replace(",", " ").split()
             if tok.strip("`.:" "").endswith(".md")]
    assert not named, (
        "the comment above DEFAULT_STORE names a markdown file as a surface: " + comment)
    assert "two surfaces" in comment, comment

    # Both destinations redirected by env (`_resolve_store` :268, `_mirror_path` :283),
    # so this run cannot append a probe row to the live ledger or the vault mirror.
    store, mirror = tmp_path / "v.jsonl", tmp_path / "m.jsonl"
    os.environ["SKILL_VERDICTS_STORE"] = str(store)
    os.environ["SKILL_VERDICTS_MIRROR"] = str(mirror)
    try:
        rc = sv.main(["record", "--pattern", "Bash/prose-check", "--verdict",
                      "reviewed_no_skill", "--reason", "surface count",
                      "--occurrences", "3", "--evidence-cmd",
                      declares("echo prose-check: 3 keys")])
    finally:
        os.environ.pop("SKILL_VERDICTS_STORE", None)
        os.environ.pop("SKILL_VERDICTS_MIRROR", None)
    assert rc == 0
    assert store.is_file() and mirror.is_file(), "record did not write both surfaces"
    md = [p for p in tmp_path.iterdir() if p.suffix == ".md"]
    assert not md, f"record wrote a markdown file after all: {[p.name for p in md]}"


# ═══════════════════════════════════════════════════════════════════════════
# #2048: a falsifier that declares it read nothing is a state, not a healthy zero
#
# `audit`'s published `unrunnable:` figure is about whether a stored check can execute.
# The live ledger proves that is not the same question as whether it can see its input:
# tonight it published `keys: 109 unrunnable: 0` with exit 0 while 28 of those 109
# latest-wins keys reference no absolute path that resolves. The item's executed proof is
# one of them verbatim — the `backlog_write_task/validation` falsifier's
# `print(sum(...))` over `/home/alansrobotlab/lloyd/_pipeline/trajectories/2026-09-*.jsonl`,
# a root deleted on 2026-09-22 — which exits 0 printing `0`, exactly the bytes a healthy
# count prints. #1588 repaired the paths it was handed and cannot reach this class at all,
# because its repair pass is driven by the keys `audit` calls UNRUNNABLE. So the fix is in
# the executor: a command that declares its own denominator as zero is reported, named and
# refused, and a command that declares nothing is tallied, not refused.
# ═══════════════════════════════════════════════════════════════════════════

#: rc 0, and its stdout says it read nothing. The shape of the item's proof, in the
#: `name=value` form Phase 0.6 of `nightly-skill-consolidation` already asks for.
ZERO_DENOMINATOR_CMD = "echo 'tool=x input_rows=0 matched=0'"

#: The control: rc 0 as well, and it read 9 rows. Both print a count, so only the
#: declared denominator separates an observation from a measurement of nothing.
NONZERO_DENOMINATOR_CMD = "echo 'tool=x input_rows=9 matched=2'"


def test_a_check_that_declares_it_read_nothing_is_neither_a_healthy_zero_nor_unrunnable():
    """Clause 1: `EMPTY_INPUT` is a third answer, and its detail names the declared zero.

    Three refusals of the same temptation, each pinned beside the case it would corrupt:
    labelling the state `UNRUNNABLE` would hand these keys to #1588's repair pass, which
    repairs paths and would "fix" a command whose input is an empty glob into a different
    empty glob; folding it into rc 0 is the status quo that let 28 keys be honoured on
    nothing; and re-running the command to read its denominator would give `audit` two
    answers about one ledger. And a falsifier that exits nonzero beside `input_rows=0`
    stays exactly the rc it returned — that is a check answering no, which is the ledger
    working, and relabelling it would train `check` to distrust a real falsification.
    """
    # One three-way comparison, because the three answers must differ from each other:
    # pinned separately from the ledger's healthy rc 0 (the status quo this item exists to
    # end) and from `UNRUNNABLE` (a different repair route — #1588 repairs a missing path
    # and would "fix" a command whose glob matched nothing into a different glob matching
    # nothing). Asserting the triple also pins the sentinel's *value*, which a bare
    # `rc != sv.UNRUNNABLE` cannot: it can never fail once `rc == sv.EMPTY_INPUT` is
    # pinned above it, since the two are different objects by definition.
    triple = (sv.evidence_cmd_status({"evidence_cmd": ZERO_DENOMINATOR_CMD})[0],
              sv.evidence_cmd_status({"evidence_cmd": "echo 0"})[0],
              sv.evidence_cmd_status(
                  {"evidence_cmd": "grep -c x /tmp/definitely-missing-2048-b.md"})[0])
    assert triple == (sv.EMPTY_INPUT, 0, sv.UNRUNNABLE), \
        f"zero-denominator / healthy / unrunnable came back {triple}"
    assert sv.EMPTY_INPUT != 0 and sv.EMPTY_INPUT != sv.UNRUNNABLE, \
        "a sentinel that collided with rc 0 or with 127 would silently merge the states"
    assert sv.evidence_cmd_status({"evidence_cmd": ZERO_DENOMINATOR_CMD})[1] \
        == "input_rows=0", "the detail must name the declared zero, not just the state"

    assert sv.evidence_cmd_status({"evidence_cmd": "echo 0"}) == (0, ""), \
        "the proof's own output — a bare 0 with no claim — is still an observation"
    assert sv.evidence_cmd_status({"evidence_cmd": NONZERO_DENOMINATOR_CMD}) == (0, "")
    assert sv.evidence_cmd_status(
        {"evidence_cmd": f"{NONZERO_DENOMINATOR_CMD}; exit 3"}) == (3, ""), \
        "a declared denominator must not disturb a real exit status"
    assert sv.evidence_cmd_status(
        {"evidence_cmd": "echo 'input_rows=0'; exit 1"}) == (1, ""), \
        "rc 1 beside a zero denominator is the falsifier answering no, not an empty input"
    assert sv.evidence_cmd_status({"evidence_cmd": "echo 'input_rows=0' >&2"}) == (0, ""), \
        "the claim is read on stdout, where a measurement goes; stderr is not a denominator"
    assert sv.evidence_cmd_status({"evidence_cmd": "echo 'total_input_rows=0'"}) == (0, ""), \
        "a longer field name is not this one"
    assert sv.evidence_cmd_status(
        {"evidence_cmd": "grep -c x /tmp/definitely-missing-2048-a.md; echo 'input_rows=0'"})[0] \
        == sv.UNRUNNABLE, \
        "a command whose input file is gone is #1588's to repair, not this state's"


def test_record_refuses_a_check_that_declares_it_read_no_input_and_a_check_saying_nothing(store):
    """Clause 2 of #2048, and the half of it #2052 reversed.

    The zero-denominator refusal is for the same reason #1586 refuses an unrunnable command:
    a verdict born on a check that read nothing can never be falsified, and this shape is
    worse — it looks healthy to every later tally. That half is untouched.

    What moved is the silence beside it. This node used to accept `echo 'count=0'` as proof
    that "making the field mandatory is a ruling on the authoring convention, not a fact this
    function gets to decide alone". The ruling was made on 2026-10-02 (#2052), so the mint
    refuses an undeclared rc-0 command too, and the accepted group is now the commands that
    DO declare: a count over a real input, and that same count at a non-zero exit, which is a
    falsifier answering no and stays recordable because the mandate is keyed on rc 0. The
    109 legacy keys that declare nothing are not refused anywhere here — they are history, and
    `audit` goes on counting them `undeclared` without faulting them, which is what the
    re-cut row below is for.
    """
    with pytest.raises(ValueError, match="input_rows=0"):
        sv.record_verdict(store=store, pattern_key="Bash/validation",
                          verdict="reviewed_no_skill",
                          reason="falsifier globs a root deleted on 2026-09-22",
                          evidence_cmd=ZERO_DENOMINATOR_CMD, occurrences=4)
    assert not store.exists(), "a refused verdict reached the ledger"

    accepted = [declares("echo 'count=0'"),          # zero matched over one read input: a
                                                            # conclusion, not an empty input
                NONZERO_DENOMINATOR_CMD,
                f"{NONZERO_DENOMINATOR_CMD}; exit 1"]            # falsified, and recorded
    for index, cmd in enumerate(accepted):
        sv.record_verdict(store=store, pattern_key=f"test/denominator-{index}",
                          verdict="reviewed_no_skill",
                          reason="one row per case, so the accepted ones are countable",
                          evidence_cmd=cmd)
    rows = [json.loads(ln) for ln in store.read_text().splitlines()]
    assert [r["pattern_key"] for r in rows] == ["test/denominator-0", "test/denominator-1",
                                                "test/denominator-2"], rows
    assert rows[0]["evidence_observed"] == "count=0", \
        "a zero conclusion measured over a real input records; only a zero INPUT is refused"
    with pytest.raises(ValueError, match="input_rows="):
        sv.record_verdict(store=store, pattern_key="test/undeclared",
                          verdict="reviewed_no_skill",
                          reason="the shape 109 live keys are stored in",
                          evidence_cmd="echo 'count=0'")
    assert len(store.read_text().splitlines()) == 3, \
        "the refused undeclared mint still wrote a row"


def test_the_shipped_cli_refuses_a_zero_denominator_check_and_leaves_the_ledger_alone(tmp_path):
    """Clause 2 across its process boundary: the CLI nightly jobs call, not the import.

    `nightly-skill-consolidation` Phase 5.1 records through `python scripts/skill_verdicts.py
    record`, so the refusal has to survive argv, `main()`'s error path and the exit code —
    an in-process `ValueError` that `cmd_record` swallowed into a 0 would be invisible to a
    nightly that reads only exit codes.
    """
    store = tmp_path / "verdicts.jsonl"
    # `record` writes two or three trees, and the refusal has to leave all of them
    # alone: the scratch ledger, the mirror `sync_orphans` appends alongside it, and the
    # real vault copy at `mirror_path()` — which is the shipped ledger the nightly reads,
    # so a refusal that reached it would be an unobserved write to the vault. Pointing
    # $SKILL_VERDICTS_MIRROR at a tmp file keeps that vault file out of reach of the
    # subprocess at all, and leaves a witness that says the same thing.
    mirror = tmp_path / "mirror.jsonl"
    vault_copy = Path(sv.DEFAULT_MIRROR)   # the shipped ledger, not a scratch copy
    before = (store.read_bytes() if store.is_file() else b"",
              mirror.read_bytes() if mirror.is_file() else b"",
              vault_copy.read_bytes() if vault_copy.is_file() else None)

    proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                           "record", "--store", str(store),
                           "--pattern", "Bash/validation", "--verdict", "reviewed_no_skill",
                           "--reason", "globs a deleted root",
                           "--occurrences", "4",
                           "--evidence-cmd", ZERO_DENOMINATOR_CMD],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                               "SKILL_VERDICTS_MIRROR": str(mirror)})

    assert proc.returncode != 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "input_rows=0" in proc.stderr, proc.stderr
    assert (store.read_bytes() if store.is_file() else b"",
            mirror.read_bytes() if mirror.is_file() else b"",
            vault_copy.read_bytes() if vault_copy.is_file() else None) == before, \
        "a refused verdict reached the ledger, the mirror or the vault copy"


def test_audit_names_the_zero_denominator_keys_and_tallies_denominators_above_its_tally(tmp_path):
    """Clauses 3 and 4 together: one named line per key, one tally line, `unrunnable:` intact.

    Four keys, four states, so the tallies are read against each other rather than against
    an empty ledger: one that cannot run, one that ran and declared a zero, one that ran and
    declared 9 rows, and one that ran and declared nothing (`AUDIT_SEES` is tonight's ledger
    shape — a bare count). `undeclared` counts all four of the keys whose command named no
    field, the unrunnable one included, which is why `empty_input + undeclared + declared`
    adds to `keys:` and a reader can check that arithmetic on the printed line alone.
    """
    store = tmp_path / "verdicts.jsonl"
    write_ledger(store, [AUDIT_MISSING_FILE, AUDIT_EMPTY_INPUT, AUDIT_DECLARES, AUDIT_SEES])

    rc, out = run_audit(store)
    lines = out.splitlines()

    assert f"EMPTY_INPUT {AUDIT_EMPTY_INPUT} :: input_rows=0" in lines, out
    assert lines.count(f"EMPTY_INPUT {AUDIT_EMPTY_INPUT} :: input_rows=0") == 1, out
    assert f"UNRUNNABLE {AUDIT_MISSING_FILE} " in "\n".join(lines), out
    assert AUDIT_DECLARES not in "\n".join(ln for ln in lines
                                          if ln.startswith(("EMPTY_INPUT", "UNRUNNABLE"))), out
    assert lines[-2] == "denominators: empty_input 1 undeclared 2", out
    assert lines[-1] == "keys: 4 unrunnable: 1", out
    assert rc == 1, out


def test_the_denominators_tally_never_touches_the_published_unrunnable_figure(tmp_path):
    """Clause 4: a zero or undeclared denominator is never counted in `unrunnable:`.

    The figure is the nightly's published health number
    (`skills/nightly-skill-consolidation/SKILL.md` writes it as `ledger_unrunnable:`), so a
    key counted there twice would report the new blind spot using the number #1587 was added
    to expose — and a key counted in neither would vanish. Both ledgers here hold the same
    four states; only the ledger without the dead key is allowed to exit 0.
    """
    mixed = tmp_path / "mixed.jsonl"
    write_ledger(mixed, [AUDIT_EMPTY_INPUT, AUDIT_DECLARES, AUDIT_SEES, AUDIT_SEES_NONE])
    rc_mixed, out_mixed = run_audit(mixed)
    assert out_mixed.splitlines()[-1] == "keys: 4 unrunnable: 0", out_mixed
    assert rc_mixed == 1, "an empty-input ledger exits 1 on its own, with nothing unrunnable"

    clean = tmp_path / "clean.jsonl"
    write_ledger(clean, [AUDIT_DECLARES, AUDIT_SEES, AUDIT_SEES_NONE])
    rc_clean, out_clean = run_audit(clean)
    assert out_clean.splitlines()[-1] == "keys: 3 unrunnable: 0", out_clean
    assert out_clean.splitlines()[-2] == "denominators: empty_input 0 undeclared 2", out_clean
    assert "EMPTY_INPUT " not in out_clean, out_clean


def test_an_undeclared_denominator_is_published_but_never_fails_a_run(tmp_path):
    """Clause 5's other half: exit 1 follows the zero denominator, not the missing field.

    A ledger of nothing but undeclared keys is tonight's live ledger — 109 keys, every one
    recorded before the field existed — so this exit code is what keeps the rail from being
    a retroactive refusal: the count is published so the trend is measurable, and a run is
    not failed for an authoring convention nobody had taught it. Adding one zero-denominator
    key to the same ledger must flip it, which is the pair the clause is stated as.
    """
    undeclared = tmp_path / "undeclared.jsonl"
    write_ledger(undeclared, [AUDIT_SEES, AUDIT_SEES_NONE, AUDIT_FAILED_BUT_RAN])
    rc, out = run_audit(undeclared)
    assert rc == 0, out
    assert out.splitlines()[-2] == "denominators: empty_input 0 undeclared 3", out

    one_zero = tmp_path / "one-zero.jsonl"
    write_ledger(one_zero, [AUDIT_SEES, AUDIT_SEES_NONE, AUDIT_FAILED_BUT_RAN,
                            AUDIT_EMPTY_INPUT])
    rc_zero, out_zero = run_audit(one_zero)
    assert rc_zero == 1, out_zero
    assert out_zero.splitlines()[-2] == "denominators: empty_input 1 undeclared 3", out_zero


def test_the_shipped_cli_prints_the_denominator_state_on_its_check_surface(tmp_path):
    """The advisory's seam: `check`'s new line over a real pipe, not a redirected buffer.

    Phase 5 reads `check`'s stdout from a subprocess, and every existing node on this
    surface captures it in-process with `capsys` or `redirect_stdout`, which cannot see a
    write that never reaches a file descriptor — a `sys.stdout` buffering difference or a
    message that reaches stderr instead would pass those nodes and print nothing for the
    nightly. So this node runs the shipped script as a child over a pipe, and asserts the
    line there: the state named, its detail naming the declared zero, `unrunnable` not
    borrowed, the tally line still last, exit 0 for a ledger that is un-falsifiable rather
    than un-runnable, and the candidate's own `status:` unchanged — naming the state
    publishes it, it does not un-honour the verdict.
    """
    cands = tmp_path / "candidates"
    cands.mkdir()
    raw_candidate(cands, "candidate-bash-timeout-zero-denominator.md", "reviewed_no_skill")
    store = tmp_path / "verdicts.jsonl"
    # The row is written by hand for the reason `write_ledger` gives: `record_verdict` now
    # refuses exactly this command, so a row of this shape can only have arrived from
    # before the rail — which is the population this node is about.
    with store.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"logged_at": "2026-10-02T07:00:00+00:00",
                             "pattern_key": "Bash/timeout", "candidate": "x",
                             "occurrences": 9, "verdict": "reviewed_no_skill",
                             "reason": "fixture: zero-denominator falsifier",
                             "scope": "project", "skill": "",
                             "evidence_cmd": ZERO_DENOMINATOR_CMD,
                             "source": "nightly-skill-consolidation"}) + "\n")

    proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                           "check", "--candidates", str(cands), "--store", str(store)],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

    assert "EVIDENCE_CMD_EMPTY_INPUT Bash/timeout :: input_rows=0" in proc.stdout, proc.stdout
    assert "EVIDENCE_CMD_UNRUNNABLE" not in proc.stdout, \
        "the two states must not be reported by one name; only one of them is repairable"
    assert proc.stdout.splitlines()[-1] == "checked: 1  skipped_by_verdict: 1", proc.stdout
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    # And the verdict still honoured, untouched: `check` only reads candidates here (its
    # write-back is `mine-trajectories.py`'s `superseded_by_verdict`, scripts/skill_verdicts.py:196,
    # which this invocation does not perform), so a file that arrived as
    # `reviewed_no_skill` — the status that makes it a skip in the first place — leaves as
    # the same thing. Naming the state is a publication about the check, not a re-decision.
    assert status_of(next(cands.glob("*.md"))) == "reviewed_no_skill"


def test_the_shipped_cli_publishes_both_tallies_over_stdout_in_the_agreed_order(tmp_path):
    """Clause 3 and 4 across their process boundary: the bytes `splitlines()[-1]` sees.

    Phase 0.0 of `nightly-skill-consolidation` parses the LAST line of this surface's
    stdout, and every audit node above captures it in-process with `capsys`, which cannot
    see a write that never reaches a file descriptor nor the order two `print` calls reach
    a pipe in. So the child's stdout is read here as the nightly reads it: the published
    figure is last and byte-identical, the new tally sits immediately above it, one
    `EMPTY_INPUT` line names the key and its declared zero, and the two counts are of
    different keys — this ledger holds one unrunnable key, one empty-input key and one
    healthy declared key, so `unrunnable: 1` cannot have absorbed the empty input and
    `undeclared 1` counts only the key whose command named no field. Exit 1 is the
    `unrunnable` rule extended to `empty_input`; the all-undeclared 0 case is pinned by
    the node named for it.
    """
    store = tmp_path / "verdicts.jsonl"
    stored_row(store, AUDIT_MISSING_FILE, AUDIT_CMDS[AUDIT_MISSING_FILE])
    stored_row(store, AUDIT_EMPTY_INPUT, AUDIT_CMDS[AUDIT_EMPTY_INPUT])
    stored_row(store, AUDIT_DECLARES, AUDIT_CMDS[AUDIT_DECLARES])

    proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                           "audit", "--store", str(store), "--timeout", "1"],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    lines = proc.stdout.splitlines()

    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert lines[-1] == "keys: 3 unrunnable: 1", lines
    assert lines[-2] == "denominators: empty_input 1 undeclared 1", lines
    assert f"EMPTY_INPUT {AUDIT_EMPTY_INPUT} :: input_rows=0" in lines, lines
    assert sum(ln.startswith("UNRUNNABLE ") for ln in lines) == 1, lines


#: #2048 clause 6's witness: the verdicts ledger the item's figures are measured over, as
#: committed bytes with history. `wc -l` on it is the row count the item quotes; the
#: latest-wins table read out of it is the 109 keys every percentage divides by. The live
#: file under `_pipeline` gains a row with every consolidation run and is folded by
#: retention, so a figure quoted from it rots (#1193) — these bytes cannot.
#:
#: The dated name is #2103's doing, and it is a move, not a rewrite. #2048 landed these bytes
#: at `backlog/data/verdicts.jsonl` (vault `8cbecceb`) and nothing ever refreshed that path, so
#: by 2026-10-03 `wc -l < backlog/data/verdicts.jsonl` printed 282 while #2103 was quoting a
#: 291-row ledger — the same rot this witness exists to prevent, reached by the mechanism
#: built to prevent it. #2103 refreshed the undated path to the bytes IT measured (vault
#: `ef42b40a`) and moved these 282 rows here, byte-identical to the copy it replaced and to the
#: live ledger's first 282 lines (`cmp` clean against each; the ledger is append-only, so an
#: older population is exactly a prefix). Two things follow, and both are the point: the three
#: assertions below are #2048's, unchanged, and a witness is now pinned to a path nothing can
#: refresh under it — `verdicts.jsonl` is the live-facing copy, for whoever needs the clause's
#: own command to answer about tonight's ledger.
WITNESS_2048_ROWS = 282
WITNESS_2048_KEYS = 109
WITNESS_2048_VAULT_PATH = "backlog/data/2026-10-02.2048-verdicts-witness.jsonl"


def test_the_verdicts_ledger_the_item_measures_is_committed_and_declares_nothing():
    """Clause 6: re-derive the item's two denominators from bytes a reader can hold.

    Both figures come from one read of the durable copy: the row count the clause's own
    `wc -l` command gives (282), and the latest-wins key count those rows collapse to (109),
    which is the denominator of every percentage in the item — 28 of 109 with no resolving
    path, 95 of 109 naming an absolute path. Any other key count contradicts #2048's report
    of the same ledger, so the node refuses rather than restating it.

    The third assertion is why the rail lands with zero findings instead of quarantining a
    ledger: not one of those 109 commands even mentions `input_rows`, so none of them can be
    reporting a denominator, which is exactly what `audit`'s `undeclared:` figure counts and
    why it reads 109 on this copy. It is a text test on the stored commands, deliberately —
    the alternative is executing 109 of them inside a unit node, and a claim about what the
    ledger *holds* does not need a run to be true. The count can only fall by new records.

    The copy is on the vault's main at WITNESS_2048_VAULT_PATH (landed through
    `automod_vault_land`, so it is not in this diff), which is also what makes the ledger
    citable by the review's own resolver; with no vault present the node skips, as
    `write_ledger`'s durable-copy leg does. It reached that dated name in #2103, which needed
    the undated `verdicts.jsonl` to carry its own figures — the move is `cmp`-proven lossless
    and none of the three assertions below moved with it.
    """
    durable = Path.home() / "obsidian" / WITNESS_2048_VAULT_PATH
    if not durable.is_file():
        pytest.skip("no vault durable copy of the verdicts ledger here")

    rows = [ln for ln in durable.read_text(errors="replace").splitlines() if ln.strip()]
    assert len(rows) == WITNESS_2048_ROWS, (
        f"#2048's report re-derived {WITNESS_2048_ROWS} rows from this file and this copy has "
        f"{len(rows)}, so the ratio quoted over it is no longer the one the item measured")
    table = sv.load_verdicts(durable)
    assert len(table) == WITNESS_2048_KEYS, (
        f"latest-wins gives {len(table)} keys, not the {WITNESS_2048_KEYS} the item divides "
        "its figures by — the copy and the report disagree")
    mentioning = sorted(k for k, r in table.items()
                        if sv.INPUT_ROWS_FIELD in (r.get("evidence_cmd") or ""))
    assert mentioning == [], (
        f"{mentioning[:3]} mention the field, so this ledger is no longer the "
        "all-undeclared population the witness is for — re-cut the figures from a copy "
        "taken before the rail landed")


# ── the mandate at the mint (#2052) ──────────────────────────────────────────
#
# #2048 built the zero-denominator rail and stopped at the authoring convention: it
# published `undeclared` and refused to fault a field nothing in the code or the
# convention asked anybody to write. That left the rail covering 0% of future verdicts
# — 0 of the 109 latest-wins keys in the committed witness mention `input_rows=` — so
# #2052 makes the mint ask, with the convention written first, in
# `nightly-skill-consolidation`'s Phase 0.6. The five nodes below are the two halves of
# that pair and the one thing it must not do: reach backwards into the ledger.

#: rc 0, a real observation, and no denominator anywhere in it — the shape 109 witness
#: keys are stored in, and the one the mint now turns away.
UNDECLARED_CMD = "echo 'matched=3 of 9 candidate files'"

#: The same observation with its denominator: three matches over three files READ. The count
#: is of the input, per the vault bullet, and it is a non-zero count, which is what keeps this
#: out of `EMPTY_INPUT` — the clause's other half is the zero that stays refused.
DECLARED_THREE_CMD = "echo 'matched=3 input_rows=3 candidate_files=3'"


def test_the_mint_refuses_an_undeclared_command_naming_the_field_on_both_surfaces(store,
                                                                                   tmp_path):
    """Clause 1: a verdict cannot be born without saying what its check read.

    Two seams, because the mandate has two callers reaching one function. The direct call is
    the runbook's (`record_verdict` is what `nightly-skill-consolidation` Phase 0.6 binds),
    and the CLI is what a person at a terminal and any future worker hit — and the CLI has to
    be asserted separately because `main` turns the `ValueError` into an exit code, so a
    handler that swallowed it would still satisfy the first half.

    Both halves assert that NOTHING landed, which is the property that makes a mint-time
    refusal worth having at all: a row written beside a refusal is a verdict with no
    denominator and a run that believes it passed. The message must name the literal
    `input_rows=` — a refusal that says "declare your denominator" in prose the author cannot
    grep for is a refusal that has to be decoded before it can be obeyed.
    """
    cli_store = tmp_path / "cli.jsonl"

    with pytest.raises(ValueError, match="input_rows="):
        sv.record_verdict(store=store, pattern_key="Bash/validation",
                          verdict="reviewed_no_skill", reason="grounds, no denominator",
                          evidence_cmd=UNDECLARED_CMD, occurrences=2)
    assert not store.exists(), "a refused mint still wrote a ledger"

    proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                           "record", "--store", str(cli_store), "--pattern", "Bash/validation",
                           "--verdict", "reviewed_no_skill",
                           "--reason", "grounds, no denominator", "--occurrences", "2",
                           "--evidence-cmd", UNDECLARED_CMD],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0, f"the CLI accepted what the function refuses: {proc.stdout}"
    assert "input_rows=" in proc.stderr + proc.stdout, proc.stderr or proc.stdout
    assert not cli_store.exists(), "the refused CLI record still wrote a ledger"


def test_a_declared_denominator_records_and_a_declared_zero_keeps_the_other_refusal(store):
    """Clause 2: the new refusal did not swallow the two verdicts that already worked.

    `input_rows=3` over a real input records a row exactly as it did before the mandate —
    same fields, command stored byte-for-byte as minted — because the field is a claim to be
    checked, not a tax that changes what a passing row looks like.

    And `input_rows=0` is still refused by #2048's `EMPTY_INPUT`, not by the new branch. The
    distinction is the clause: both messages name `input_rows=`, so matching on that string
    alone cannot tell them apart, and the two refusals mean different things. One says the
    command measured nothing and its conclusion is vacuous; the other says the command never
    said what it measured. Re-routing the zero through the new branch would publish a fix
    that does not fix the thing #2048 found, and the ledger would fill with rows whose check
    saw an empty directory while every message blamed the missing field.
    """
    sv.record_verdict(store=store, pattern_key="Sweep/declared",
                      verdict="reviewed_no_skill", reason="3 of 9 files matched",
                      evidence_cmd=DECLARED_THREE_CMD, occurrences=1)
    row = sv.load_verdicts(store)["Sweep/declared"]
    assert row["evidence_cmd"] == DECLARED_THREE_CMD, "the command was not stored as minted"
    assert row["verdict"] == "reviewed_no_skill" and row["occurrences_at_decision"] == 1, row
    assert len(store.read_text().splitlines()) == 1, "the accepted mint wrote more than one row"

    with pytest.raises(ValueError) as exc:
        sv.record_verdict(store=store, pattern_key="Sweep/zero",
                          verdict="reviewed_no_skill", reason="globs a pruned root",
                          evidence_cmd=ZERO_DENOMINATOR_CMD)
    assert "input_rows=0" in str(exc.value), str(exc.value)
    assert "empty input" in str(exc.value).lower(), (
        f"the zero was refused by the wrong rail: {exc.value}")
    assert len(store.read_text().splitlines()) == 1, "the refused zero wrote a row"


def test_the_mint_and_the_read_rail_agree_when_the_zero_hides_behind_a_header_line(store):
    """Clause 3: one parse of one stream, so the two rails cannot disagree on one run.

    The command prints its findings header first and the `input_rows=0` claim on the second
    line. `run_evidence` keeps only the first non-empty line as `evidence_observed`, so a mint
    guard written against that line would see "no denominator" and refuse the command as
    undeclared — while `audit`, which scans the whole stdout through `declared_denominator`,
    reports the same bytes as `EMPTY_INPUT`. That pair is the failure this node exists for:
    the author is told to add a field that is already there, the real reason (an empty input)
    stays invisible, and the two surfaces of one tool tell one story each.

    The clause has a positive half, and it is the half that catches the cheap implementation:
    a command whose declaration sits on the second line and is NON-zero must RECORD. A mint
    guard written against `run_evidence`'s first line alone refuses that verdict outright —
    the author is told their command declares nothing while `audit`, reading the whole stream,
    reports the count sitting in it — which is the divergence this clause exists to forbid, and
    the reason the guard reads `declared_denominator` over the same bytes `_run_stored_check`
    reads rather than the one-line summary the row stores.

    Every assertion therefore goes through the same helper the rails use, and the expected
    verdict is named explicitly: `EMPTY_INPUT` on both surfaces, never the undeclared refusal.
    """
    header_then_zero = ("printf 'candidate sweep, 9 files considered\\n'; "
                        "echo 'sweep/hidden_zero input_rows=0 matched=0'")
    assert sv._run_stored_check({"evidence_cmd": header_then_zero})[2] == 0, (
        "the read rail does not see the zero behind the header line, so there is nothing for "
        "the mint to agree with")

    with pytest.raises(ValueError) as exc:
        sv.record_verdict(store=store, pattern_key="Sweep/hidden_zero",
                          verdict="reviewed_no_skill", reason="header line first",
                          evidence_cmd=header_then_zero)
    assert "input_rows=0" in str(exc.value), str(exc.value)
    assert "empty input" in str(exc.value).lower(), (
        f"the mint refused a declared zero as undeclared: {exc.value}")
    assert not store.exists(), "a refused mint reached the ledger"

    write_ledger(store, [AUDIT_EMPTY_INPUT])
    state, detail = sv.evidence_cmd_status({"evidence_cmd": AUDIT_CMDS[AUDIT_EMPTY_INPUT]})
    assert state == sv.EMPTY_INPUT, (state, detail)
    rc, out = run_audit(store)
    assert rc == 1 and "EMPTY_INPUT" in out, out
    assert "undeclared 0" in out, out

    # The positive half: a NON-zero declaration on the second line is a real denominator, so
    # the verdict records and `audit` counts it declared rather than undeclared.
    header_then_nine = ("printf 'candidate sweep, 9 files considered\\n'; "
                        "echo 'sweep/hidden_nine input_rows=9 matched=0'")
    assert sv._run_stored_check({"evidence_cmd": header_then_nine})[2] == 9, (
        "the read rail does not see a declaration behind the header line")
    sv.record_verdict(store=store, pattern_key="Sweep/hidden_nine",
                      verdict="reviewed_no_skill", reason="header line first, then a count",
                      evidence_cmd=header_then_nine)
    assert "Sweep/hidden_nine" in sv.load_verdicts(store), (
        "the mint refused a command whose denominator the read rail can see, which is the "
        "divergence this clause forbids")
    assert sv.evidence_cmd_status({"evidence_cmd": header_then_nine})[0] not in (
        sv.UNRUNNABLE, sv.EMPTY_INPUT), "the read rail called a 9-row input empty"


def test_reading_a_ledger_of_legacy_undeclared_rows_faults_nothing_and_writes_nothing(
        tmp_path):
    """Clause 4: the mandate points forward, and both read surfaces stayed where #2048 left them.

    `AUDIT_SEES` and `AUDIT_SEES_NONE` are the two legacy shapes in the fixture table — an rc-0
    run printing a bare count, an rc-1 run printing one — and neither mentions the field, which
    is the population the live ledger and the committed witness are made of. #2048's own
    `test_an_undeclared_denominator_is_published_but_never_fails_a_run` pins the exit code and
    the tally line over three such keys; this node adds the two halves the mandate could have
    broken and did not.

    The byte comparison is the half that cannot be asserted from stdout. A read path that
    "helped" by stamping `input_rows=1` into the rows it audited would print the same report,
    exit the same, and quietly re-judge 109 keys nobody authored wrongly — which is exactly the
    retroactive quarantine the item's clause 4 forbids and the vault paragraph now states twice.
    """
    ledger = tmp_path / "legacy.jsonl"
    write_ledger(ledger, [AUDIT_SEES, AUDIT_SEES_NONE])
    before = ledger.read_bytes()

    rc, out = run_audit(ledger)
    assert rc == 0, out
    assert "denominators: empty_input 0 undeclared 2" in out, out

    assert ledger.read_bytes() == before, (
        "audit rewrote the ledger it was asked to read: a read path that stamps "
        "`input_rows=` into legacy rows re-judges history one silent write at a time")

    for key in (AUDIT_SEES, AUDIT_SEES_NONE):
        state, detail = sv.evidence_cmd_status({"evidence_cmd": AUDIT_CMDS[key]})
        assert state not in (sv.UNRUNNABLE, sv.EMPTY_INPUT), (key, state, detail)


def test_the_mandate_is_taught_in_the_vault_before_it_is_enforced_in_code():
    """Clause 5: a refusal nobody was taught is a bug report, not a convention.

    Three assertions, one per surface the ruling had to reach, and the order matters more than
    the wording: `nightly-skill-consolidation` Phase 0.6 is where a runbook authors its
    `--evidence-cmd`, so the mandate is useless if it is not there; the two #2051 retirement
    paragraphs in both templates used to state the OPPOSITE — "The ruling is owed-check's, not a
    run's. Until it is made, publish the figure", and "mandating the field at `record` is #58's
    Phase 0.6 ruling, not this stage's call" — and a stale pending sentence there now contradicts
    code that refuses; `nightly-skills-management` is the other template that publishes
    `undeclared` and has to describe the same rule in the same terms.

    Each paragraph must ALSO keep publishing the figure with no target value: the mandate does
    not make `undeclared` a fault, and a paragraph that quietly became a goal line would put a
    target on a number only ageing can move. That is what the last assertion reads for.
    """
    cons = CONSOLIDATION_RUNBOOK.read_text(encoding="utf-8", errors="replace")
    start = cons.index("**0.6 Authoring")
    phase06 = cons[start:cons.index("## Phase 1:", start)]
    assert "input_rows=" in phase06, (
        "Phase 0.6 does not teach the field the mint now refuses without")
    assert "input_rows=0" in phase06, (
        "Phase 0.6 must distinguish the refused zero-input from an absence probe's zero")

    retire_start = cons.index("**When the key retires (#2051).**")
    retirement = cons[retire_start:cons.index("## Guardrails", retire_start)]
    assert "mandatory at `record`" in retirement, (
        "the retirement paragraph still states the ruling as pending while the code enforces it")
    assert "no target" in retirement, retirement
    assert "owed-check" in retirement, (
        "the key's own retirement is still owed-check's; the paragraph must not claim otherwise")

    mgmt = (Path.home() / "obsidian" / "skills" / "nightly-skills-management"
            / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    assert "mandatory at `record`" in mgmt or "mandated now" in mgmt, (
        "nightly-skills-management still describes the field as somebody else's pending call")
    assert "input_rows=$(ls -1" in mgmt, (
        "the mirror must carry the same two shapes the mandate teaches: a count of the input, "
        "and an absence probe printing the rows it inspected")


"""#2103: the two reads no exit code reports.

Both holes are the same shape — a falsifier that ran, answered, and left the ledger unable
to tell what its answer measured — and both were invisible to every existing rail because
each rail reads the exit status. A case-sensitive `grep -c` with no match exits 1 while
printing `0`, so a verdict can only ever be *provisionally* blocked by a check whose casing
the owner's own nightly rewrite is free to move (#530's caveat, unenforced for six weeks);
and a command that exits 0 while leaving three of its fields empty satisfies a rail that
only reads the fourth (#2052's `input_rows`). So the fix is not a fourth exit code: the
classifier asks the stored command its question a second way and publishes `STRANDED_CASE`,
and the write path refuses to mint the two shapes it can see coming.
"""


def grep_owner(skill: Path, literal: str, flags: str = "-c") -> str:
    """The owner-coverage falsifier shape #57 minted, on a temp skill file instead of the vault.

    Built by hand rather than through `declares` because these nodes are about the read: the
    declaration is the honest count of the one file the command inspects, and the captured
    exit status is what makes a no-match grep (rc 1, stdout `0`) distinguishable from a match
    — the exact pair whose readings the classifier now has to tell apart.
    """
    return (f"grep {flags} '{literal}' {skill}; rc=$?; "
            f"echo '{sv.INPUT_ROWS_FIELD}=1'; exit $rc")


def skill_fixture(tmp_path) -> Path:
    """A skill file holding the witness sentence, capitalised as its owner wrote it.

    Mirrors `file-mutation-safety:96` after its 09-29 refresh: the sentence sits
    sentence-initially, so the ledger's lower-case literal no longer matches it even though
    the coverage is verbatim present. Deliberately under `tmp_path`, so a mint-time refusal
    of a `skills/**/SKILL.md` grep needs no vault write to be exercised.
    """
    skill = tmp_path / "vault" / "skills" / "file-mutation-safety" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Skill\n\nSix such sessions in the window. One mechanism, two "
                     "symptoms.\n", encoding="utf-8")
    return skill


def test_record_refuses_a_falsifier_whose_rc_zero_line_leaves_its_fields_empty(tmp_path, store):
    """Clause 1: the shape that minted `run:2026-10-03-nightly-mining` cannot be minted again.

    Three of that line's four words are field names and three lost their value, so the
    durable record carried `input_rows=2812` (real, and matching the miner's own printed
    window) beside `newest_bucket=`, `newest_rows=` and `prev_rows=` printing nothing — which
    is the whole window denominator, missing from the record while every rail read rc 0 and a
    declared count. The refusal has to name each empty field, because the honest repair is to
    re-run the command with its variables quoted, not to invent three numbers.
    """
    with pytest.raises(ValueError) as exc:
        sv.record_verdict(store=store, pattern_key="run:2026-10-03-nightly-mining",
                          verdict="reviewed_no_skill",
                          reason="no signature clears the threshold",
                          evidence_cmd="echo 'newest_bucket= newest_rows= prev_rows= "
                                       f"{sv.INPUT_ROWS_FIELD}=2812'")

    msg = str(exc.value)
    for field in ("newest_bucket", "newest_rows", "prev_rows"):
        assert field in msg, f"the refusal must name the empty field {field}: {msg}"
    assert sv.empty_valued_fields("newest_bucket= newest_rows= prev_rows= input_rows=2812") == [
        "newest_bucket", "newest_rows", "prev_rows"], (
        "the named fields are the ones that lost their value, and the denominator that "
        "carried its number is not one of them")
    assert not store.exists(), "a refused mint writes nothing to either tree"


def test_the_empty_field_refusal_never_fires_on_a_no_answer_or_a_populated_line(tmp_path, store):
    """Clause 2: the rail is keyed on rc 0 and on a genuinely empty value, both halves pinned.

    The first case is the one the item's own correction makes load-bearing: the ledger
    answering *no* (nonzero rc beside its fields) is the falsifier working, and refusing it
    would teach `record` to reject a real falsification — the same mistake #1586's fix had to
    be careful of on the other axis. The second is a line whose every field carries a value.
    The third is a `0`: a zero is a measurement, not an absence, and `empty_input` already
    owns that state with its own name and its own tally.
    """
    line = (f"newest_bucket= newest_rows= prev_rows= {sv.INPUT_ROWS_FIELD}=2812")
    sv.record_verdict(store=store, pattern_key="run:no-answer",
                      verdict="rejected_false_positive", reason="falsifier answered no",
                      evidence_cmd=f"echo '{line}'; exit 1")
    sv.record_verdict(store=store, pattern_key="run:fully-populated",
                      verdict="reviewed_no_skill", reason="window auditable",
                      evidence_cmd="echo 'newest_bucket=2026-10-03 newest_rows=12 "
                                   f"prev_rows=9 {sv.INPUT_ROWS_FIELD}=2812'")
    sv.record_verdict(store=store, pattern_key="run:zero-is-a-value",
                      verdict="reviewed_no_skill", reason="zero counted, not missing",
                      evidence_cmd=f"echo 'newest_rows=0 {sv.INPUT_ROWS_FIELD}=2812'")

    rows = sv.load_verdicts(store)
    assert set(rows) == {"run:no-answer", "run:fully-populated", "run:zero-is-a-value"}, rows
    assert rows["run:no-answer"]["evidence_observed"] == line, (
        "a falsifier answering no is recorded unchanged, including the empties it printed — "
        "that row's rc is what makes it readable, and no rail may rewrite it")
    assert rows["run:fully-populated"]["evidence_observed"] == (
        "newest_bucket=2026-10-03 newest_rows=12 prev_rows=9 input_rows=2812")


def test_audit_names_one_stranded_case_line_per_key_and_tallies_it_outside_unrunnable(tmp_path):
    """Clause 3: `audit` asks the third question, and its figure is not `unrunnable`'s.

    Three keys, one line each expected. `k:stranded` is the repaired witness's own command
    text — case-sensitive, rc 1, stdout `0` — which is the state every existing rail reads as
    the ledger working. `k:absent` is a genuinely missing string and must NOT be reported: the
    rail's whole cost is a second execution, so a key whose case-insensitive re-read is also
    empty has to fall through to its real exit status. `k:present` already matched and must
    not be run twice.

    The strand is published beside the key because a nightly reading `evidence_observed: "0"`
    concludes the owner LOST the section, and that is how a closed key gets re-adjudicated on
    an instrument artefact. It gets its own count, and `unrunnable: 0` stays 0, because a
    case-stranded key ran fine — folding it in would both hide #1533's real figure and point
    the repair at a re-anchored path when the fix is `grep -i`.
    """
    skill = skill_fixture(tmp_path)
    store = tmp_path / "verdicts.jsonl"
    stored_row(store, "k:stranded", grep_owner(skill, "one mechanism, two symptoms"))
    stored_row(store, "k:absent", grep_owner(skill, "no such section anywhere"))
    stored_row(store, "k:present", grep_owner(skill, "one mechanism, two symptoms", flags="-ci"))

    proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                           "audit", "--store", str(store), "--timeout", "5"],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    out = proc.stdout

    found = [ln for ln in out.splitlines() if ln.startswith("STRANDED_CASE")]
    assert len(found) == 1 and found[0].startswith("STRANDED_CASE k:stranded :: "), out
    assert "k:absent" not in out and "k:present" not in out, (
        f"a string absent under `-i` too is the ledger answering no, and a key that matched "
        f"is not measured twice — both belong in no line at all: {out}")
    detail = [ln for ln in out.splitlines() if ln.startswith("STRANDED_CASE")][0]
    assert "grep -i -c" in detail, f"the line has to carry the read that does match: {detail}"
    assert "stranded: case_sensitive_grep 1" in out, out
    assert "keys: 3 unrunnable: 0" == out.splitlines()[-1], out
    assert "unrunnable: 1" not in out, "the count is never folded into the published figure"
    assert proc.returncode == 1, "a stranded key fails the run: 1 if dead or stranded"


def test_the_shipped_cli_publishes_the_stranded_count_without_moving_the_published_tallies(
        tmp_path):
    """Clause 3's process boundary: the numbers a nightly reads, at the positions it reads them.

    In-process assertions can miss a print that goes to the wrong stream or a tally pushed off
    the end of the output, and this surface's contract is positional — `splitlines()[-1]` is the
    ledger tally and `[-2]` is the denominator tally, both published before #2103 and both
    pinned by existing nodes. So the new line enters above them, and this node runs the real
    program over a temp ledger with exactly one stranded key to pin all four at once.
    """
    skill = skill_fixture(tmp_path)
    store = tmp_path / "verdicts.jsonl"
    stored_row(store, "seq-2-edit-err-read", grep_owner(skill, "one mechanism, two symptoms"))
    stored_row(store, "k:healthy", grep_owner(skill, "one mechanism, two symptoms", flags="-ci"))

    proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                           "audit", "--store", str(store), "--timeout", "5"],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    lines = proc.stdout.splitlines()

    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert lines[-1] == "keys: 2 unrunnable: 0", lines
    assert lines[-2] == "denominators: empty_input 0 undeclared 0", lines
    assert lines[-3] == "stranded: case_sensitive_grep 1", lines
    assert lines[0].startswith("STRANDED_CASE seq-2-edit-err-read ::"), lines
    assert proc.stdout.count("STRANDED_CASE ") == 1, (
        "exactly one finding line: the tally is spelled `stranded: case_sensitive_grep`, so a "
        f"reader counting the uppercase token counts keys and never tally lines: {lines}")


def test_check_names_the_case_stranded_key_on_its_existing_evidence_cmd_line(tmp_path):
    """Clause 4: the surface a step-0 SKIP falsifier actually re-executes says so too.

    The failure this closes is not a wrong block. The key is terminal, so it blocks either
    way; the failure is that the nightly which then reads `evidence_observed: "0"` writes a
    finding saying the owner skill no longer holds the section, and the run after that
    re-adjudicates or reopens a closed key on the casing of a grep. Named in the
    `EVIDENCE_CMD_*` shape the loop already prints and already teaches, and the ledger's own
    figure stays untouched — no `LEDGER_UNVERIFIABLE`, no exit 1: an instrument note is not a
    veto, exactly as an absent denominator is not one (#2052's ruling, adopted here unchanged).
    """
    skill = skill_fixture(tmp_path)
    store = tmp_path / "verdicts.jsonl"
    stored_row(store, "Bash/timeout", grep_owner(skill, "one mechanism, two symptoms"))
    cands = tmp_path / "candidates"
    mt.write_candidate_file(error_pattern(), cands)

    proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                           "check", "--candidates", str(cands), "--store", str(store)],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

    assert "EVIDENCE_CMD_STRANDED_CASE Bash/timeout ::" in proc.stdout, proc.stdout
    assert "SKIP Bash/timeout" in proc.stdout, (
        "the key blocks exactly as it did before this line existed")
    assert "LEDGER_UNVERIFIABLE" not in proc.stdout, proc.stdout
    assert proc.returncode == 0, (proc.returncode, proc.stdout)
    assert proc.stdout.splitlines()[-1] == "checked: 1  skipped_by_verdict: 1", proc.stdout


def test_the_stranded_note_changes_the_line_and_never_the_block_it_reports(tmp_path):
    """Clause 4's other half: the strand is a note about a grep, never a second verdict.

    Two ledgers, same key, same candidate, differing only in the casing of the stored read:
    the case-insensitive one matches, so it is not stranded, and the case-sensitive one reads
    `0` at rc 1 and is. Both must `SKIP` identically and exit 0, and only the stranded one
    carries the line — because if publishing the artefact could move a decision, this surface
    would be silently reopening verdicts on a casing change, the exact instability #2103 exists
    to remove. The `check` loop reaches a stored command only for a key its verdict is
    currently blocking, which is the right scope and not an accident to widen: a key with no
    terminal verdict stands in front of nothing, so there is no "owner lost the section"
    inference for the artefact to poison.

    The third ledger is the strand's other origin: a stored command that swallows grep's exit
    status (`cmd; echo …` with no `exit $rc`, the shape this file's own fixtures had to stop
    writing) prints `0` and exits 0, so its answer is a count rather than a falsification — and
    it is reported exactly as the rc-1 one is, because `check` receives the instrument's state
    from the classifier and not the row's underlying exit status. What neither one touches is
    `unverified`, and that is deliberate: folding a strand into the unverifiable figure would
    need a third execution to learn which rc produced it, for a signal `audit` already fails the
    run over. Both exit 0: an instrument note has never been a veto on this surface (#2052's
    ruling, adopted unchanged).
    """
    skill = skill_fixture(tmp_path)
    ledgers = {
        "stranded": grep_owner(skill, "one mechanism, two symptoms"),
        "matched": grep_owner(skill, "one mechanism, two symptoms", flags="-ci"),
        "swallowed": (f"grep -c 'one mechanism, two symptoms' {skill}; "
                      f"echo '{sv.INPUT_ROWS_FIELD}=1'"),
    }

    outputs = {}
    for name, cmd in ledgers.items():
        store = tmp_path / name / "verdicts.jsonl"
        stored_row(store, "Bash/timeout", cmd, verdict="reviewed_no_skill")
        cands = tmp_path / name / "candidates"
        mt.write_candidate_file(error_pattern(), cands)
        proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                               "check", "--candidates", str(cands), "--store", str(store)],
                              capture_output=True, text=True,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert proc.returncode == 0, (name, proc.returncode, proc.stdout)
        assert any(ln.startswith("SKIP Bash/timeout :: reviewed_no_skill")
                   for ln in proc.stdout.splitlines()), (name, proc.stdout)
        assert "REOPEN" not in proc.stdout, (name, proc.stdout)
        outputs[name] = proc.stdout

    stranded_lines = [ln for ln in outputs["stranded"].splitlines()
                      if ln.startswith("EVIDENCE_CMD_")]
    assert [ln.split(" :: ")[0] for ln in stranded_lines] == [
        "EVIDENCE_CMD_STRANDED_CASE Bash/timeout"], outputs["stranded"]
    assert not [ln for ln in outputs["matched"].splitlines()
                if ln.startswith("EVIDENCE_CMD_")], (
        "a read that matches is not an artefact, and the block above proves the line is the "
        "only difference between the two runs")
    assert "LEDGER_UNVERIFIABLE" not in outputs["stranded"], (
        "rc 1 beside a strand is a falsifier answering no, which is the ledger working")
    assert "EVIDENCE_CMD_STRANDED_CASE Bash/timeout ::" in outputs["swallowed"], (
        outputs["swallowed"])
    assert "LEDGER_UNVERIFIABLE" not in outputs["swallowed"], outputs["swallowed"]


def test_record_refuses_a_new_case_sensitive_owner_grep_and_names_the_case_insensitive_read(
        tmp_path, store):
    """Clause 5: a falsifier born falsifiable by somebody else's nightly is refused at the mint.

    Refused before the command runs, because the defect is the shape of the read and not its
    answer — the same command that matches tonight is the one that reads `0` the night the
    owner re-cases its sentence, and by then the verdict is six weeks old and a stored `0` is
    indistinguishable from the owner deleting the section. Naming `grep -i` in the message is
    the whole repair, so the message has to carry it: #83's nightly write path is the one that
    strands these greps and it does not read this file.
    """
    skill = skill_fixture(tmp_path)
    cmd = grep_owner(skill, "one mechanism, two symptoms")

    with pytest.raises(ValueError) as exc:
        sv.record_verdict(store=store, pattern_key="seq-2-edit-err-read",
                          verdict="reviewed_no_skill", reason="owner holds the section",
                          evidence_cmd=cmd, occurrences=11)

    msg = str(exc.value)
    assert str(skill) in msg, f"the refusal must name the file it refuses to read: {msg}"
    assert "grep -ci" in msg, f"the refusal must name the read that survives a rewrite: {msg}"
    assert "#2103" in msg, msg
    assert not store.exists(), "a refused mint writes nothing to either tree"

    # The same command with the case-insensitive read mints, and mints *clean*: the mandate is
    # about casing, and a refusal that also rejected `-i` would teach a mint to drop the
    # falsifier rather than fix it.
    sv.record_verdict(store=store, pattern_key="seq-2-edit-err-read",
                      verdict="reviewed_no_skill", reason="owner holds the section",
                      evidence_cmd=grep_owner(skill, "one mechanism, two symptoms", flags="-ci"),
                      occurrences=11)
    assert sv.load_verdicts(store)["seq-2-edit-err-read"]["evidence_observed"] == "1"


def test_the_case_sensitive_mint_rail_leaves_other_shapes_and_the_stored_history_alone(
        tmp_path, store):
    """Clause 5's second half: the mandate is mint-time, and the ledger's history is untouched.

    Three things have to stay true while that rail fires, and each is a way a naive version
    would be wrong. A case-sensitive grep of a file that is not an installed skill is the
    common falsifier in this very test file (`PRINTING_CMD` greps `skill_verdicts.py`), and a
    rail widened to all greps would refuse most of the ledger's healthy keys; a key stored
    before the mandate is the recorded history #2103 explicitly does not re-record — the 25
    case-sensitive `SKILL.md` falsifiers stay in the ledger, still blocking, and `audit`
    publishes them nightly rather than `record` pretending they were never written.
    """
    skill = skill_fixture(tmp_path)
    sv.record_verdict(store=store, pattern_key="k:source-grep", verdict="reviewed_no_skill",
                      reason="greps the script, not a skill",
                      evidence_cmd=grep_owner(_ROOT / "scripts" / "skill_verdicts.py",
                                              "def load_verdicts"))
    # And the rail is about the READ, not the string: a command that greps some other file and
    # happens to name a skill path in prose is not an owner-coverage falsifier, and a rail that
    # refused on the mention would be refusing on a substring — the same over-broad reflex that
    # made this class of rail wrong before it shipped.
    other = tmp_path / "window.txt"
    other.write_text("newest_bucket=2026-10-03\n", encoding="utf-8")
    sv.record_verdict(store=store, pattern_key="k:mentions-a-skill", verdict="reviewed_no_skill",
                      reason="greps the window, names a skill in passing",
                      evidence_cmd=declares(f"grep -c 'newest_bucket=' {other}; "
                                           f"echo 'see {skill}'"))
    legacy = grep_owner(skill, "one mechanism, two symptoms")
    stored_row(store, "Bash/timeout", legacy)

    with pytest.raises(ValueError):
        sv.record_verdict(store=store, pattern_key="k:second-attempt",
                          verdict="reviewed_no_skill", reason="same shape refused",
                          evidence_cmd=legacy)

    rows = sv.load_verdicts(store)
    assert set(rows) == {"k:source-grep", "k:mentions-a-skill", "Bash/timeout"}, rows
    assert rows["Bash/timeout"]["evidence_cmd"] == legacy, (
        "a stored row is never rewritten: append-only, latest-wins, and the 24 legacy "
        "SKILL.md greps are history that `audit` reports, not that `record` repairs")

    cands = tmp_path / "candidates"
    mt.write_candidate_file(error_pattern(), cands)
    proc = subprocess.run([sys.executable, str(_ROOT / "scripts" / "skill_verdicts.py"),
                           "check", "--candidates", str(cands), "--store", str(store)],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert "EVIDENCE_CMD_STRANDED_CASE Bash/timeout ::" in proc.stdout, proc.stdout
    assert "SKIP Bash/timeout" in proc.stdout, (
        "a row stored before the mandate blocks exactly as it did the night it was written")
    assert proc.returncode == 0, (proc.returncode, proc.stdout)


def test_the_mint_rail_catches_a_skill_path_built_through_a_variable(tmp_path, store):
    """Clause 5's bypass, closed on the shape the live ledger actually uses.

    The strand rail may only rewrite a command whose operand it can prove is a plain path,
    because it re-executes what it rewrites. The mint rail only decides, so it is wider: a
    command whose `SKILL.md` arrives through `$K` is refused on its text. Measured on
    2026-10-03, one of the ledger's 27 latest-wins `SKILL.md` falsifiers is exactly this shape —
    `K=/home/<user>/obsidian/skills; grep -Fc 'command (string) is required'
    "$K/tool-parameter-validation/SKILL.md"` — and a rule that required a literal `skills/`
    prefix in the command would let every one of them be re-minted behind a variable, which is
    #2052's `--no-input-rows` hole wearing different clothes.
    """
    cmd = ("K=" + str(tmp_path / "vault" / "skills")
           + "; grep -Fc 'command (string) is required' \"$K/tool-parameter-validation/SKILL.md\""
           + "; rc=$?; echo 'input_rows=1'; exit $rc")
    (tmp_path / "vault" / "skills" / "tool-parameter-validation").mkdir(parents=True)
    (tmp_path / "vault" / "skills" / "tool-parameter-validation" / "SKILL.md").write_text(
        "Tool Parameter Validation\n\ncommand (string) is required\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        sv.record_verdict(store=store, pattern_key="k:variable-path",
                          verdict="reviewed_no_skill", reason="fixture: greps via $K",
                          evidence_cmd=cmd)
    assert "grep -ci" in str(exc.value), str(exc.value)
    assert not store.exists()


def test_a_falsifier_that_records_both_casings_is_not_refused(tmp_path, store):
    """The exemption's own shape: the ledger's repair idiom stays mintable.

    `seq-2-edit-err-read` was hand-repaired on 2026-10-03 by recording the coverage
    case-insensitively *beside* the case-sensitive read that had gone to `0`, so the artefact
    and the claim sit in one line. A mint rail that refused that would forbid the only
    authoring pattern that has actually demonstrated it knows about casing, and would disagree
    with the strand rail, which asks no second question of a command that already reads
    case-insensitively — a key must not be exempt at the mint and stranded at the audit.
    """
    skill = skill_fixture(tmp_path)
    cmd = (f"echo {sv.INPUT_ROWS_FIELD}=1 "
           f"owner_section_ci=$(grep -ci 'one mechanism, two symptoms' {skill}) "
           f"owner_section_case_sensitive=$(grep -c 'one mechanism, two symptoms' {skill})")
    sv.record_verdict(store=store, pattern_key="seq-2-edit-err-read",
                      verdict="reviewed_no_skill",
                      reason="fixture: both casings recorded", evidence_cmd=cmd)
    observed = sv.load_verdicts(store)["seq-2-edit-err-read"]["evidence_observed"]
    assert "owner_section_ci=1" in observed, observed
    assert "owner_section_case_sensitive=0" in observed, (
        "the line the ledger was repaired to write is the line this mint must still produce")


def test_the_case_insensitive_reread_survives_a_tilde_spelled_operand(tmp_path):
    """Clause 3 on the spelling most of the ledger actually uses.

    The live falsifiers name `~/obsidian/skills/<x>/SKILL.md`, and a `~` survives `shlex.split`
    only to be single-quoted back by `shlex.join` — which bash does not expand, so the re-read
    targets a path literally named `'~/obsidian/…'`, dies at rc 2, and the key is published as
    a healthy rc-1 falsifier instead of `STRANDED_CASE`. Measured on 2026-10-03: of the 12
    latest-wins `SKILL.md` falsifiers this module can parse at all, 10 are tilde-spelled, so a
    rail built that way would be blind to five sixths of the corpus it was written for. The fixture therefore sits
    under the real home, in a uniquely named directory removed afterwards.
    """
    home = Path.home()
    rel = ".cache/lloyd-skill-verdicts-tilde-2103/skills/file-mutation-safety"
    target = home / rel
    target.mkdir(parents=True, exist_ok=True)
    try:
        (target / "SKILL.md").write_text(
            "# Skill\n\nSix such sessions in the window. One mechanism, two symptoms.\n",
            encoding="utf-8")
        cmd = f"grep -c 'one mechanism, two symptoms' ~/{rel}/SKILL.md"
        variant = sv.case_insensitive_reread(cmd)
        assert variant is not None and f"~/{rel}/SKILL.md" in variant and "'~" not in variant, variant
        state, detail = sv.evidence_cmd_status({"evidence_cmd": cmd})
        assert state == sv.STRANDED_CASE, (state, detail)
    finally:
        shutil.rmtree(home / ".cache" / "lloyd-skill-verdicts-tilde-2103", ignore_errors=True)


# ── the witness bytes behind this item's figures (#2103 clause 6) ─────────────
#
# Everything #2103 asserts about the ledger — "110 keys", "25 case-sensitive greps", "5 rows
# with an empty field", "undeclared 102" — is a claim about a file under `_pipeline/`, which
# `.gitignore` excludes from every repo on the box and gains a row from every consolidation
# run. A figure quoted from a moving file cannot be re-checked by the reader who is asked to
# act on it, which is the same defect #1193 named and #2048's clause 6 answered by committing
# the copy it measured. These bytes are that copy for this item: the live ledger as of
# 2026-10-03T07:18:12Z, byte-identical (sha256 4c5ed7beb86c1320bc4d84245f1bfbe48227265f212d85928b1ede352d6a03d0),
# landed on the vault's main and dated so it can never be re-cut under a quoted figure.
#
# Why a dated copy and not the undated one. `backlog/data/verdicts.jsonl` IS refreshed to these
# bytes in this round (vault `ef42b40a`), so the clause's own command now answers about
# tonight's ledger instead of printing #2048's 282 — that stale contradiction is what the first
# review of this round refused on. But an item's figures stay pinned here, because a path that
# gets refreshed for the next item goes red under the previous one, and that is exactly how
# #2048's witness came to disagree with this item's report. So the rule the round landed on:
# the undated name is live-facing, every quoted figure reads a dated copy, and a refresh moves
# the old population to its own dated sibling rather than overwriting it — #2048's 282 rows are
# at `2026-10-02.2048-verdicts-witness.jsonl`, `cmp`-identical to what was there before.

#: The committed copy, as a path relative to the vault root (the same resolution
#: `sv.DEFAULT_MIRROR` uses for the durable ledger copy, and for the same reason: the vault is
#: a second tree, so it is never a function of `--store`).
WITNESS_2103_VAULT_PATH = "backlog/data/2026-10-03.2103-verdicts-witness.jsonl"

#: Every figure #2103's report and its triage quote, measured off those bytes on
#: 2026-10-03. They are the item's claims, not this file's inventions: the row count is what
#: the clause's own `wc -l` command prints, 110 is the `keys:` line `audit` published at
#: 06:14Z, 102 is the `undeclared` figure on the line above it, and 27/24 is the exposure
#: count the automod triage recorded ("25 of 26" there is corrected in the round-findings
#: section of the item, because that scan counted only keys whose path it could parse and
#: missed 3 keys that read case-insensitively and one that reads a `SKILL.md` through `$K`).
WITNESS_2103_ROWS = 291
WITNESS_2103_KEYS = 110
WITNESS_2103_SKILL_MD_KEYS = 27
WITNESS_2103_CASE_SENSITIVE_KEYS = 24
WITNESS_2103_UNDECLARED_KEYS = 102

#: The three exempt keys by name, because "exempt" here is a claim with a reason: each one
#: reads something case-insensitively already, which is the repair, so the mint rail asks it
#: no second question. One is the hand-repaired witness key; two were never in the triage
#: scan's population at all.
WITNESS_2103_EXEMPT_KEYS = [
    "seq-2-edit-err-bash-fs",
    "seq-2-edit-err-read",
    "seq-3-bash-fs-bash-explore-bash-fs-err",
]

#: The five rows whose stored `evidence_observed` carries a field name with no value — the
#: population clause 1's rail exists for, and the count `audit` could not see at 06:14Z. Four
#: are the 2026-09-27 mining batch (`snapshot=`/`sessions=`), one is the run-level line filed
#: the night this item opened (`newest_bucket=`/`newest_rows=`/`prev_rows=`).
WITNESS_2103_EMPTY_FIELD_ROWS = [
    "run:2026-10-03-nightly-mining",
    "seq-2-bash-fs-edit",
    "seq-3-bash-explore-bash-fs-read",
    "seq-3-bash-fs-backlog-write-task-bash-fs",
    "seq-3-read-edit-bash-fs",
]


def _witness_2103() -> Path:
    """The committed witness bytes, or a skip when this machine has no vault.

    The same shape as #2048's witness node: the durable copy is a second tree, and a ledger
    on a box without one is not a failing ledger.
    """
    durable = Path.home() / "obsidian" / WITNESS_2103_VAULT_PATH
    if not durable.is_file():
        pytest.skip(f"no committed verdicts witness at {WITNESS_2103_VAULT_PATH} here")
    return durable


def test_the_verdicts_ledger_this_item_measures_is_committed_and_re_derivable():
    """Clause 6: the item's denominators come out of bytes a reader can hold.

    Four figures, one read. The row count is taken by the clause's own command — a real
    `wc -l` child process, not a line count this file invents, because the clause publishes
    that command as the re-check and a test that computes the number another way proves
    nothing about it; the splitlines count is asserted to agree, since a file whose last
    line has no newline would make the two disagree and every figure below would then be a
    count of a different corpus. The key count is `load_verdicts`, i.e. the latest-wins
    table `audit` and `check` both read, so the published `keys: 110` is the same number.
    `undeclared` is a text count of the commands that name no `input_rows=`, which is exactly
    how `audit` defines it, unrunnable keys included — deliberately not a re-execution of
    110 stored commands inside a unit node.

    The 27/24 pair is the exposure figure the round re-measured at head `8e9befbd` and the
    only one of the four that depends on this diff, since 24 is the count `
    case_sensitive_skill_md_grep` would refuse **as a new mint**: the rule that lands here,
    read back over the history it deliberately does not rewrite. A witness that stopped
    matching would mean either the copy is not the one the item measured or the mint rail
    moved under the report, and both are reasons to stop and re-derive the item, so each
    assertion names which figure broke rather than just failing.
    """
    durable = _witness_2103()

    counted = subprocess.run(["wc", "-l", str(durable)], capture_output=True, text=True)
    assert counted.returncode == 0, counted.stderr
    assert int(counted.stdout.split()[0]) == WITNESS_2103_ROWS, counted.stdout
    rows = [ln for ln in durable.read_text(errors="replace").splitlines() if ln.strip()]
    assert len(rows) == WITNESS_2103_ROWS, (
        f"`wc -l` and a filtered read disagree, so this copy is not newline-terminated and "
        f"the {WITNESS_2103_ROWS}-row figure the item quotes is ambiguous")

    table = sv.load_verdicts(durable)
    assert len(table) == WITNESS_2103_KEYS, (
        f"latest-wins gives {len(table)} keys, not the 110 in `audit`'s published line, so "
        "this copy is not the ledger the item's percentages divide by")

    skill_keys = sorted(k for k, r in table.items()
                        if "SKILL.md" in (r.get("evidence_cmd") or ""))
    assert len(skill_keys) == WITNESS_2103_SKILL_MD_KEYS, (
        f"{len(skill_keys)} latest-wins keys name a SKILL.md, not "
        f"{WITNESS_2103_SKILL_MD_KEYS}, so the exposure figure is not this population")
    refused = sorted(k for k in skill_keys
                     if sv.case_sensitive_skill_md_grep(table[k]["evidence_cmd"]))
    assert len(refused) == WITNESS_2103_CASE_SENSITIVE_KEYS, (
        f"the mint rail would refuse {len(refused)} of those keys as new mints, not "
        f"{WITNESS_2103_CASE_SENSITIVE_KEYS} — the rail and the item's report no longer agree")
    assert sorted(set(skill_keys) - set(refused)) == WITNESS_2103_EXEMPT_KEYS, (
        "a different set of keys reads something case-insensitively, so the exemption rule "
        "moved and the item's 3-exempt figure is stale")

    undeclared = sorted(k for k, r in table.items()
                        if sv.INPUT_ROWS_FIELD not in (r.get("evidence_cmd") or ""))
    assert len(undeclared) == WITNESS_2103_UNDECLARED_KEYS, (
        f"{len(undeclared)} commands name no {sv.INPUT_ROWS_FIELD}=, not the 102 `audit` "
        "printed beside `keys: 110`, so these bytes are not the ledger that printed it")


def test_the_committed_witness_carries_the_rows_with_a_field_and_no_value():
    """Clause 6 on the second defect's population: the five half-empty rows are in the copy.

    Clause 1's rail is decided by `empty_valued_fields`, so reading that same predicate over
    the committed bytes is what turns "5 ledger rows carry an empty-valued field today" from
    a sentence into a check. It runs over every row and not the latest-wins table on purpose:
    a half-empty line stays in the durable record after a later verdict supersedes its key,
    which is the whole reason `audit`, not `record`, has to be the surface that reads them.
    The list is sorted by key rather than by row number because the item's row numbering is
    zero-based and a reader re-deriving it from `enumerate(..., 1)` gets numbers one higher —
    the keys cannot be off by one.
    """
    durable = _witness_2103()
    rows = [json.loads(ln) for ln in durable.read_text(errors="replace").splitlines()
            if ln.strip()]

    blanked = sorted({r.get("pattern_key", "") for r in rows
                      if sv.empty_valued_fields(r.get("evidence_observed") or "")})
    assert blanked == WITNESS_2103_EMPTY_FIELD_ROWS, (
        f"{blanked} carry a field with no value, not the five the item names — so either "
        "this copy is not the ledger it measured or the rail's predicate does not fire on "
        "the stored shape")
    run_row = [r for r in rows if r.get("pattern_key") == "run:2026-10-03-nightly-mining"]
    assert len(run_row) == 2, (
        f"the run-level key has {len(run_row)} rows; the item's account is a half-empty mint "
        "and one hand correction, and clause 1's refusal is what keeps the first shape from "
        "being minted again")
    assert sv.empty_valued_fields(run_row[0].get("evidence_observed") or "") == [
        "newest_bucket", "newest_rows", "prev_rows"]
    assert not sv.empty_valued_fields(run_row[1].get("evidence_observed") or ""), (
        "the correcting line is itself half-empty, so the mint that clears the rail is wrong")


# ---------------------------------------------------------------------------
# #2166 — a falsifier that counts a candidate file whole counts the ledger's own prose
#
# `mine-trajectories.status_block()` (scripts/mine-trajectories.py:1658-1677) writes the
# decision's own `verdict_reason:` into the FRONT MATTER of every superseded snapshot for
# that key. A `grep -c` over the whole file therefore counts the verdict re-injecting
# itself, and the count can only inflate: a reason that mentions the very path or phrase
# the literal searches for keeps a dead key looking alive. Measured on
# `candidate-edit-logic-20261004.md` the night this was filed: whole=3, body=2, with the
# surplus match at line 17, `verdict_reason:`. Body-scoping — strip the front matter with
# awk and count the rest — is the fix, and it is what four keys re-minted themselves to.
# ---------------------------------------------------------------------------

#: The path shape `Edit/logic`'s falsifier counts, and the one its own verdict_reason names.
WORKTREE_PATH_MARK = "lloyd-work/SM_20261003_042353/home/lloyd/"
#: How many times that path appears in the fixture's front matter and in its examples.
#: 1 + 2 = the whole-file read, 2 = the body read: the item's own whole=3 body=2.
FM_MENTIONS, BODY_MENTIONS = 1, 2


def candidate_2166(tmp_path: Path) -> Path:
    """A dated candidate snapshot in the shape the miners write, with a `verdict_reason`.

    One mention of `WORKTREE_PATH_MARK` in the front matter (the re-injected verdict) and two
    in the example lines below the closing `---` — the exact 3-versus-2 shape the filing
    measured. Returns the candidates DIRECTORY, because the stored falsifiers glob it.
    """
    cand = tmp_path / "candidates"
    cand.mkdir(parents=True, exist_ok=True)
    (cand / "candidate-edit-logic-20261004.md").write_text(
        "---\n"
        "pattern_key: Edit/logic\n"
        "occurrences: 3\n"
        "sessions: 2\n"
        f"verdict_reason: the count is fine, it just moved — see {WORKTREE_PATH_MARK}tests/x.py\n"
        "---\n"
        "\n"
        "## Example 1  (session s1, 2026-10-03)\n"
        f"- **Input:** `{{'file_path': '{WORKTREE_PATH_MARK}tests/x.py'}}`\n"
        "- **Error:** `Edit refused: old_string not found in file`\n"
        "\n"
        "## Example 2  (session s2, 2026-10-03)\n"
        f"- **Input:** `{{'file_path': '{WORKTREE_PATH_MARK}tests/y.py'}}`\n"
        "- **Error:** `Edit refused: old_string not found in file`\n",
        encoding="utf-8")
    return cand


def counting_cmd(cand: Path, scope: str) -> str:
    """The `Edit/logic` falsifier in one of its two historical scopes.

    `whole` is the shape the key stored until 2026-10-04: a `grep -c` whose operand is the
    candidate file itself. `body` is the shape it stores now, and the shape #2166 makes the
    only acceptable one — an awk pass that drops everything up to and including the second
    `---`, then a count of that copy. Both keep `input_rows` as a count of files, which is
    the honest denominator either way, and both are runnable, so a node can watch the two
    scopes DISAGREE on the same file rather than only read about it.
    """
    glob = f"{cand}/candidate-edit-logic-*.md"
    strip = "b=$(awk '/^---/{n++; next} n>=2' $f); " if scope == "body" else ""
    read = (f"printf '%s\\n' \"$b\" | grep -c '{WORKTREE_PATH_MARK}'" if scope == "body"
            else f"grep -c '{WORKTREE_PATH_MARK}' $f")
    field = "body_worktree_shape" if scope == "body" else "worktree_shape"
    return (f"f=$(ls -t {glob} | head -1); {strip}"
            f"echo \"input_rows=$(ls -1 {glob} | wc -l) {field}=$({read})\"")


def test_the_whole_file_candidate_count_measures_the_verdict_it_decided(tmp_path):
    """The premise, measured rather than quoted: whole 3, body 2, on one file.

    Both commands run against the same snapshot. The difference is the front matter, and the
    front matter is `mine-trajectories.status_block()`'s re-injected `verdict_reason` — the
    decision counting its own sentence as traffic and so keeping its own key alive. This is
    the 2026-09-21 class rule, "bind a count to the record set, never to narrative prose",
    caught in the act of failing toward the tool's own liveness."""
    cand = candidate_2166(tmp_path)
    rc, whole = sv.run_evidence(counting_cmd(cand, "whole"))
    assert rc == 0, whole
    rc2, body = sv.run_evidence(counting_cmd(cand, "body"))
    assert rc2 == 0, body
    assert f"worktree_shape={FM_MENTIONS + BODY_MENTIONS}" in whole, whole
    assert f"body_worktree_shape={BODY_MENTIONS}" in body, body
    assert sv.candidate_body_defect(counting_cmd(cand, "whole")) == (
        "whole_file", f"grep -c '{WORKTREE_PATH_MARK}' $f"), (
        "the shape that produced the number above must be the shape the rule names")
    assert sv.candidate_body_defect(counting_cmd(cand, "body")) is None, (
        "the shape that produced the honest number above must be the shape the rule accepts")


def test_record_refuses_a_candidate_count_that_greps_the_whole_file(store, tmp_path):
    """Clause 1: the mint refuses both spellings of the operand and names body-scoping.

    One via the `f=$(ls -t …)` the minter writes, one via the inline `$(ls -t … | head -1)`
    substitution — the two ways tonight's ledger reaches a candidate file. The message has to
    carry the remedy, because the run that hits it is a nightly with nobody reading: `#750`
    minted this key, `#2103`'s rail is the precedent for refusing at the mint rather than
    logging a complaint nobody executes."""
    cand = candidate_2166(tmp_path)
    skill = skill_fixture(tmp_path)
    with pytest.raises(ValueError) as raised:
        sv.record_verdict(store, "Edit/logic", "reviewed_no_skill", "stored",
                          counting_cmd(cand, "whole") + f"; {grep_owner(skill, 'old string', '-ci')}")
    msg = str(raised.value)
    assert "front matter" in msg and "verdict_reason" in msg, msg
    assert "body-scope" in msg.lower() and "n>=2" in msg, (
        f"the refusal must name the fix, not just the fault: {msg}")

    inline = (f"echo \"errlegs=$(grep -c ':ERR.*\\[ERROR\\]' "
              f"'{cand}/candidate-edit-logic-20261004.md')\"; echo '{sv.INPUT_ROWS_FIELD}=1'")
    assert sv.candidate_body_defect(inline) == (
        "whole_file", f"grep -c ':ERR.*\\[ERROR\\]' '{cand}/candidate-edit-logic-20261004.md'"), (
        "an inline candidate path is the same read spelled without a variable")
    with pytest.raises(ValueError) as raised2:
        sv.record_verdict(store, "Bash/fs", "reviewed_no_skill", "stored", inline)
    assert "body-scope" in str(raised2.value).lower(), str(raised2.value)
    assert not store.exists() or sv.load_verdicts(store) == {}, "a refused mint writes nothing"


def test_the_body_scoped_falsifier_records_unchanged(store, tmp_path):
    """Clause 2: the shape the two repaired keys store today still mints, byte for byte.

    `Edit/logic` and `Bash/logic` re-minted themselves to this on 2026-10-04 and their counts
    then equalled a hand count of the example lines. A rail that refused this shape would be
    worse than no rail: the nightly's only compliant spelling would be undepressable, and the
    run that noticed would conclude the rule is wrong and drop the scoping too."""
    cand = candidate_2166(tmp_path)
    row = sv.record_verdict(store, "Edit/logic", "reviewed_no_skill",
                            "the body count is the traffic count",
                            counting_cmd(cand, "body"))
    assert f"body_worktree_shape={BODY_MENTIONS}" in row["evidence_observed"], row
    assert sv.load_verdicts(store)["Edit/logic"]["evidence_cmd"] == counting_cmd(cand, "body"), (
        "the stored command is the accepted one, not a rewritten copy of it")


def test_the_rule_leaves_front_matter_reads_and_owner_greps_alone(store, tmp_path):
    """Clause 3: the deliberate reads still mint, so the rule costs the nightly nothing.

    `grep -m1 '^occurrences:' $f` and `grep -m1 '^sessions:' $f` target the front matter ON
    PURPOSE: they are reading the metadata, which is exactly where a `^`-anchored pattern
    lives. Refusing them would refuse the ledger's own most common read. The owner SKILL.md
    grep is exempt for the simpler reason that its operand is not a candidate file — the same
    file `seed` writes and `repair` writes into every tombstone."""
    cand = candidate_2166(tmp_path)
    skill = skill_fixture(tmp_path)
    front_matter = (f"f=$(ls -t {cand}/candidate-edit-logic-*.md | head -1); "
                    f"echo occ=$(grep -m1 -i '^occurrences:' $f | tr -dc '0-9') "
                    f"sessions=$(grep -m1 '^sessions:' $f) "
                    f"{grep_owner(skill, 'old string', '-ci')}")
    assert sv.candidate_body_defect(front_matter) is None, (
        "a `^`-anchored read has already scoped itself away from the examples")
    row = sv.record_verdict(store, "Edit/logic", "reviewed_no_skill", "front matter only",
                            front_matter)
    assert "occ=3" in row["evidence_observed"], row["evidence_observed"]

    tombstone = sv.TOMBSTONE_TEMPLATE.format(input=f"{cand}/candidate-edit-logic-*.md")
    assert sv.candidate_body_defect(tombstone) is None, (
        "#1588's disposal check counts PATHNAMES under a candidate glob; refusing it would "
        "make every future disposal unrecordable")

    # The anchor, isolated from `-m1`. `cmd_repair` re-mints `grep -c '^status:' <snapshot>`
    # when the data move strands a falsifier's root, and that read COUNTS — so `not counting`
    # does not exempt it and only the `^` can. Its unanchored twin counts the same word in the
    # example lines, which is the leak, so the two have to land on opposite sides here.
    anchored = f"echo snap_status=$(grep -c '^status:' {cand}/candidate-edit-logic-20261004.md)"
    assert sv.candidate_body_defect(anchored) is None, (
        "the shape `repair` mints must be acceptable, or a stranded falsifier cannot be "
        "repaired at all")
    unanchored = anchored.replace("^status:", "status:")
    assert sv.candidate_body_defect(unanchored) is not None, (
        "dropping the anchor is what makes the count a prose count: this file's body has no "
        "`status:` at column 0, so an unanchored read counting it is counting something else")


def test_audit_names_the_keys_whose_falsifier_counts_a_candidate_whole(store, tmp_path):
    """Clause 4: `audit` counts the stored rows that read prose, and names them.

    Against a fixture ledger holding one offender in tonight's real shape — `seq-2-bash-fs-err-bash-fs`
    with its `errlegs=$(grep -c ':ERR.*\\[ERROR\\]' $f)` — and one compliant key. The count is
    the point: `audit`'s published numbers are what the nightly reads, and a key that has been
    inflating all along reports rc 0 doing it, so no execution-based tally can ever reach it.
    The exit code goes 1 for the same reason the UNRUNNABLE list does: this is a finding with a
    fix, not a caption."""
    cand = candidate_2166(tmp_path)
    offender = (f"f=$(ls -t {cand}/candidate-seq-2-bash-fs-err-bash-fs-*.md | head -1); "
                f"echo errlegs=$(grep -c ':ERR.*\\[ERROR\\]' $f); echo '{sv.INPUT_ROWS_FIELD}=1'")
    (cand / "candidate-seq-2-bash-fs-err-bash-fs-20261004.md").write_text(
        "---\nverdict_reason: the :ERR leg [ERROR] is what grew here\n---\n"
        "- step 2 :ERR [ERROR] no such file\n", encoding="utf-8")
    stored_row(store, "seq-2-bash-fs-err-bash-fs", offender)
    stored_row(store, "Edit/logic", counting_cmd(cand, "body"))
    rc, out = run_audit(store)
    assert rc == 1, out
    assert "candidate_body_scoping: whole_file 1 dead_strip 0" in out, out
    assert any(line.startswith("WHOLE_CANDIDATE_COUNT seq-2-bash-fs-err-bash-fs")
               for line in out.splitlines()), out
    assert not any("Edit/logic" in line for line in out.splitlines()
                   if "CANDIDATE" in line or "STRIP" in line), out


def test_a_dead_front_matter_strip_is_refused_at_the_mint_and_named_by_audit(store, tmp_path):
    """Clause 5: half the applied fix was a strip nobody reads, and that is a defect, not a fix.

    `Bash/timeout` stores `b=$(awk '/^---/{n++; next} n>=2' $f)` and then emits only `owner_*`
    greps, so running it verbatim yields no `body_*` field at all: the key looks repaired, and
    its candidate traffic is now measured by nobody. Refusing the shape at the mint is the
    cheap half; `audit` naming the already-stored row is the half that reaches the ledger
    without waiting for a re-mint that may never come."""
    cand = candidate_2166(tmp_path)
    skill = skill_fixture(tmp_path)
    dead = (f"f=$(ls -t {cand}/candidate-edit-logic-*.md | head -1); "
            f"b=$(awk '/^---/{{n++; next}} n>=2' $f); "
            f"echo occ=$(grep -m1 '^occurrences:' $f) {grep_owner(skill, 'old string', '-ci')}")
    assert sv.candidate_body_defect(dead) == (
        "dead_strip", "b=$(awk '/^---/…' …) defined and never read"), dead
    with pytest.raises(ValueError) as raised:
        sv.record_verdict(store, "Bash/timeout", "reviewed_no_skill", "looks scoped", dead)
    assert "never read" in str(raised.value), str(raised.value)

    stored_row(store, "Bash/timeout", dead + f"; echo '{sv.INPUT_ROWS_FIELD}=1'")
    stored_row(store, "Edit/logic", counting_cmd(cand, "body"))
    rc, out = run_audit(store)
    assert rc == 1, out
    assert "candidate_body_scoping: whole_file 0 dead_strip 1" in out, out
    assert any(line.startswith("DEAD_FRONT_MATTER_STRIP Bash/timeout")
               for line in out.splitlines()), out


def test_a_strip_that_is_read_is_not_reported_as_dead(tmp_path):
    """The dead-strip rule's own falsifier: consume the copy and the finding is gone.

    Without this node the predicate could be satisfied only by deleting the strip, which is
    the wrong fix and would leave the whole-file read standing. The one line that changes is
    `echo occ=` reading `"$b"` instead of `$f` — the same edit the nightly has to make."""
    cand = candidate_2166(tmp_path)
    skill = skill_fixture(tmp_path)
    dead = (f"f=$(ls -t {cand}/candidate-edit-logic-*.md | head -1); "
            f"b=$(awk '/^---/{{n++; next}} n>=2' $f); echo ex=$(printf '%s\\n' \"$b\" | "
            f"grep -c 'Error:') {grep_owner(skill, 'old string', '-ci')}")
    assert sv.candidate_body_defect(dead) is None, dead
    rc, out = sv.run_evidence(dead)
    assert rc == 0 and "ex=2" in out, out      # two example lines, zero front-matter matches


def test_a_double_quoted_candidate_substitution_is_not_read_as_inert(tmp_path):
    """The one spelling of the inline substitution that is NOT the same as the other two.

    `'$(ls -t …/candidate-x-*.md | head -1)'` hands `grep` a file literally named that, which is
    why the rule calls it inert and lets `audit`'s UNRUNNABLE list handle it; `"$( … )"` is
    expanded by bash and opens the snapshot. A rule that treated every quoted word containing
    `$( ` as inert would call the second shape clean — the clause-1 inline form, passing.
    """
    cand = candidate_2166(tmp_path)
    glob = f"ls -t {cand}/candidate-edit-logic-*.md | head -1"
    expanded = f"echo errlegs=$(grep -c 'Error' \"$({glob})\"); echo '{sv.INPUT_ROWS_FIELD}=1'"
    assert sv.candidate_body_defect(expanded) is not None, (
        "a double-quoted substitution expands, so the read is of the candidate file")
    inert = f"echo errlegs=$(grep -c 'Error' '$({glob})'); echo '{sv.INPUT_ROWS_FIELD}=1'"
    assert sv.candidate_body_defect(inert) is None, (
        "a single-quoted substitution is never expanded, so `grep` opens a file literally named "
        "that and finds none: UNRUNNABLE's class, decided by running it, not this rule's")
