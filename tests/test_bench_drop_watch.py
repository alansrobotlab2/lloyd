#!/usr/bin/env python3
"""The bench-drop watch (#4555): one command that answers #4554's owed RE-ARM row.

`scripts/automod/owed.py:78` quotes #4185's ruling — RE-ARM on a measured fire, ">=1
`BENCH ASSERTION DROP` on a job's stderr whose run record does NOT carry the copied
line". Answering that by hand means grepping 1,147 run records, and the corpus punishes
it: all 42 `BENCH ASSERTION` mentions in `~/lloyd-data/autonomy-runs/` are negations, so
a marker grep finds "a fire" in a sentence that says the marker did not appear. Every
node here is a line the watcher must get right, and the seeds are that corpus's own
bytes, copied verbatim from the run records named beside them.

Fixtures only. No node reads ~/lloyd-data or ~/obsidian: the live figures are quoted as
measurements taken 2026-11-25 and behaviour is pinned against tmp_path, so the next
author is not red because a store moved.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "util" / "bench_drop_watch.py"

# Load the script as a module: scripts/util has no package __init__ and the file is a
# CLI, which is how the sibling finding scripts are imported elsewhere.
_SPEC = importlib.util.spec_from_file_location("bench_drop_watch", SCRIPT)
bdw = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_drop_watch"] = bdw
_SPEC.loader.exec_module(bdw)

# ── the two landed forms, byte-exact ───────────────────────────────────────────────
# The door's drop line, as `scripts/util/vault-commit.sh:628-629` emits it
# (printf '%s: BENCH ASSERTION DROP: %s is out of this commit — %s\n') and as
# tests/test_vault_commit_bench_assertion_refusal.py:271 pins it on the emitter side.
DOOR_DROP = (
    "vault-commit.sh: BENCH ASSERTION DROP: lloyd/bench/bench_99_probe_card.md is out of "
    "this commit — its stem has no assertion set in eval/autoresearch_assertions.yaml, "
    "which is a file of the lloyd repo (the checkout this script runs from): add a "
    "`bench_99_probe_card:` entry there first. The card was added to the index before "
    "the rail ran, so the table is the only thing that knows this card exists, and a "
    "commit that ships the card without shipping its rows ships an assertion the judge "
    "will never run. The card is NOT in this commit; bench/ stays a draft directory. A "
    "line that must not appear in this commit's paths has also been removed."
)

# The rung's copy line, as `scripts/util/bench_assertion_findings.py:477-484` prints it.
RUNG_COPY = (
    "BENCH ASSERTION: 2 bench card(s) added under bench/ and no assertion-set row was "
    "found for them in eval/autoresearch_assertions.yaml. Copy this exact line into the "
    "invoking job's run record (autonomy-runs/<task>/run_<...>.md) under a "
    "`## Bench assertion findings` heading — the commit itself is not the witness. The "
    "card was still committed: bench/ is a DRAFT directory, and a missing assertion set "
    "is a finding for the nightly judge, not a reason to refuse a card."
)

# ── the corpus, verbatim ───────────────────────────────────────────────────────────
# Every shape the 42-line corpus takes, copied out of the run record named beside it.
# Two of them are the shapes a cue-word-only guard would wrongly call a fire.
NEGATION_SEEDS: dict[str, str] = {
    # autonomy-runs/24/run_24_20261120_031938.md:88 — marker mid-sentence, "no" cue.
    "no_bench_assertion_drop_em_dash": (
        "**Commits and run record.** Pre-flight `2ad13f3` (174 unattributed paths; "
        "`lloyd/MEMORY.md` +1 −0 growth, no curated shrink/deletion; no `BENCH ASSERTION "
        "DROP` — re-derived, the tree was clean after it)"
    ),
    # autonomy-runs/24/run_24_20261121_012514.md:96 — sentence-initial "No".
    "no_bench_assertion_drop_printed": (
        "**Commits:** pre-flight `9aaa09098`; post-flight `651bd21e7` (+209 −11); log "
        "corrections `75ca5a2a8`, `a5e412d8f`, `ed45dbdba`. Unattributed paths the doors "
        "carried and this record names: `backlog/4360-x.md`, `memory/audit/writes.jsonl`. "
        "No `BENCH ASSERTION DROP` printed by either door call, and `git status "
        "--porcelain -- lloyd/bench/` was clean at both."
    ),
    # autonomy-runs/24/run_24_20261120_075438.md:101 — the RUNNER's literal in a bolded
    # list item. The shape clause 2 names, and the one that reads as a fire if the copy
    # pattern is a bare `BENCH ASSERTION:` search instead of an anchored one.
    "no_rung_line_printed": (
        "1. **No `BENCH ASSERTION:` line printed** on either pre-flight call, so no card "
        "was dropped for a missing assertion set this run — but I only know that because "
        "I ran the wrapper a **second** time to capture stderr."
    ),
    # autonomy-runs/24/run_24_20261121_184454.md:72 — negated by a DIGIT, which is why
    # the cue set carries `\b0\b`: a cue-WORD-only guard calls this line a fire.
    "zero_rung_lines": (
        "**Pre-flight snapshot:** `bcb222662`, 48 files +3249 −194, 0 MEMORY/USER/SOUL "
        "rows, **0 `BENCH ASSERTION:` lines**."
    ),
}

# Why the fire side refuses a lead that carries a word: the 234 lines across 112 files
# under ~/lloyd-data/sessions/ that contain the door's prefix (measured 2026-11-25) are
# quotations and test asserts, not observed drops. These are its shapes.
QUOTED_SEEDS: dict[str, str] = {
    "echo_quote": '    echo "' + DOOR_DROP + '"',
    "assert_quote": '    assert line.startswith("vault-commit.sh: BENCH ASSERTION DROP"), \\',
    "file_line_quote": "scripts/util/vault-commit.sh:628:" + DOOR_DROP,
}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True)


def _vault(tmp_path: Path, *, duty_skill_a: bool = True) -> dict[str, Path]:
    """A vault-shaped tree: skills, two of which carry the copy duty, and task files.

    The duty sentences are the two landed ones (`step1-fact-extraction.md:325`,
    `nightly-reflection-signals/SKILL.md:383`) copied so the derivation is graded against
    the real wording, not a paraphrase of it.

    `git init` and one commit are part of the fixture, not decoration: the `vault_dirty`
    witness runs `git status`, which exits 128 in a non-repo (the watcher would then
    correctly print `UNREADABLE`, and every node here would assert the opposite of what it
    means to check), and a repo with nothing committed lists its whole tree untracked, so
    the baseline would read dirty. Committing gives the nodes a clean baseline that a node
    can then dirty on purpose.
    """
    _git(tmp_path, "init", "-q")
    skills = tmp_path / "skills"
    a = skills / "autonomy-data-pipeline"
    b = skills / "nightly-reflection-signals"
    c = skills / "nightly-vault-maintenance"
    for d in (a, b, c):
        d.mkdir(parents=True, exist_ok=True)
    if duty_skill_a:
        (a / "step1-fact-extraction.md").write_text(
            "**A `BENCH ASSERTION:` line on that stderr is copied verbatim into this "
            "run's record, not summarised.**\n",
            encoding="utf-8",
        )
    else:  # the door, no duty: mentions the marker with no copy verb
        (a / "step1-fact-extraction.md").write_text(
            "A `BENCH ASSERTION:` line may appear on that stderr.\n", encoding="utf-8"
        )
    (b / "SKILL.md").write_text(
        "- A line starting `BENCH ASSERTION:` is copied **verbatim** into the same "
        "section: the commit itself is not the witness.\n",
        encoding="utf-8",
    )
    # A door-invoking skill with no copy duty: finding 2's duty-less majority.
    (c / "SKILL.md").write_text(
        "Commit with `scripts/util/vault-commit.sh`; read its stderr.\n", encoding="utf-8"
    )

    autonomy = tmp_path / "autonomy"
    autonomy.mkdir(exist_ok=True)
    for tid, name, skill in (
        (24, "Data Pipeline", "autonomy-data-pipeline"),
        (38, "Nightly Reflection — Signals", "nightly-reflection-signals"),
        (44, "Nightly Vault Maintenance", "nightly-vault-maintenance"),
    ):
        (autonomy / f"{tid}-{skill}.md").write_text(
            "---\n"
            + yaml.safe_dump({"id": tid, "name": name, "skill_name": skill,
                              "status": "up_next"}, sort_keys=False)
            + "---\n\nbody\n",
            encoding="utf-8",
        )
    return {"vault": tmp_path, "skills": skills, "autonomy": autonomy}


def _marker_free_stores(tmp_path: Path) -> dict[str, Path]:
    """server.err and workers.db present and marker-free, so their zeros are earned.

    Built only where the test has not already written one: a test that seeds its own
    store passes the same path, and creating a default over it would silently replace
    the fixture with an empty file and leave the node asserting against nothing.
    """
    err = tmp_path / "server.err"
    if not err.exists():
        err.write_text("INFO: booting\n", encoding="utf-8")
    db = tmp_path / "workers.db"
    if not db.exists():
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, source TEXT,"
                    " summary TEXT)")
        con.execute("INSERT INTO runs VALUES ('r1', 'autocode',"
                    " 'nothing bench-shaped')")
        con.commit()
        con.close()
    return {"server_err": err, "workers_db": db}


def _job(runs_dir: Path, tid: int, body: str) -> None:
    (runs_dir / str(tid)).mkdir(parents=True, exist_ok=True)
    (runs_dir / str(tid) / f"run_{tid}_20261125_010101.md").write_text(
        f"---\ntask_id: {tid}\nstatus: success\n---\n\n{body}\n", encoding="utf-8"
    )


def _watch(**over) -> subprocess.CompletedProcess[str]:
    """One run of the command. tmp_path defaults are built fresh per call, and any path
    can be overridden — including to a path that does not exist, which is how clause 5's
    unreadable store is reached.

    `--no-git` is the default because shelling out to git on every node is slow and the
    store cannot attribute anything; `use_git=True` opens it for the nodes that are about
    its own reporting.
    """
    tmp = over.pop("tmp_path")
    use_git = over.pop("use_git", False)
    vault = _vault(tmp)
    stores = _marker_free_stores(tmp)
    runs = tmp / "autonomy-runs"
    runs.mkdir(exist_ok=True)
    argv = [sys.executable, str(SCRIPT),
            "--runs-dir", str(runs),
            "--server-err", str(stores["server_err"]),
            "--workers-db", str(stores["workers_db"]),
            "--vault", str(vault["vault"]),
            "--skills-dir", str(vault["skills"]),
            "--autonomy-dir", str(vault["autonomy"])]
    if not use_git:
        argv.append("--no-git")
    for key, value in over.items():
        argv[argv.index(f"--{key.replace('_', '-')}") + 1] = str(value)
    return subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)


def _job_line(proc: subprocess.CompletedProcess[str], tid: int) -> list[str]:
    return [l for l in proc.stdout.splitlines() if f"task={tid} " in l]


def _store_line(proc: subprocess.CompletedProcess[str], name: str) -> str:
    hits = [l for l in proc.stdout.splitlines() if l.startswith(f"store {name}:")]
    assert len(hits) == 1, f"store {name} missing from:\n{proc.stdout}"
    return hits[0]


# ── clause 1: one token per duty job; exit 0 clean, non-zero on an uncopied drop ─────
def test_one_status_token_per_duty_job_and_exit_zero_when_none_is_uncopied(tmp_path):
    """Two duty jobs (task 24 Data Pipeline, task 38 Nightly Reflection — Signals), one
    line each, exactly one status token each, exit 0 because neither is uncopied.

    Not vacuous: task 24's record carries a drop AND its copy, so its token is earned by
    a real pairing rather than by an empty file.
    """
    runs = tmp_path / "autonomy-runs"
    _job(runs, 24, f"## Bench assertion findings\n\n{DOOR_DROP}\n\n{RUNG_COPY}")
    _job(runs, 38, NEGATION_SEEDS["zero_rung_lines"])
    proc = _watch(tmp_path=tmp_path, runs_dir=runs)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    duty = _job_line(proc, 24) + _job_line(proc, 38)
    assert len(duty) == 2, f"expected one line per duty job:\n{proc.stdout}"
    assert sorted(l.split()[0] for l in duty) == [bdw.DROPPED_COPIED, bdw.NO_DROPS], (
        proc.stdout)
    for line in duty:
        tokens = [t for t in (bdw.DROPPED_COPIED, bdw.DROPPED_UNCOPIED, bdw.NO_DROPS)
                  if t in line]
        assert len(tokens) == 1, f"a job line carries {len(tokens)} status tokens: {line}"


def test_a_duty_job_with_an_uncopied_drop_exits_nonzero(tmp_path):
    """The same tree with task 24's copy line removed: `DROPPED_UNCOPIED`, exit 1, and
    task 38 still reads NO_DROPS on the same output."""
    runs = tmp_path / "autonomy-runs"
    _job(runs, 24, DOOR_DROP)
    _job(runs, 38, "nothing bench-shaped here")
    proc = _watch(tmp_path=tmp_path, runs_dir=runs)
    assert proc.returncode == 1, f"exit {proc.returncode}:\n{proc.stdout}"
    assert _job_line(proc, 24)[0].startswith(bdw.DROPPED_UNCOPIED), proc.stdout
    assert _job_line(proc, 38)[0].startswith(bdw.NO_DROPS), proc.stdout
    assert "armed_dropped_uncopied=1" in proc.stdout


# ── clause 2: a negation-only record is NO_DROPS; plus a drop it is not ─────────────
@pytest.mark.parametrize("key", sorted(NEGATION_SEEDS))
def test_a_negation_of_either_marker_reads_as_no_drops(tmp_path, key):
    """All four corpus shapes, verbatim, inside a duty job's own record: `NO_DROPS`,
    exit 0. `zero_rung_lines` and `no_rung_line_printed` are the two the COPY pattern
    must not catch; `no_bench_assertion_drop_em_dash` is the one the FIRE pattern must
    not catch."""
    runs = tmp_path / "autonomy-runs"
    _job(runs, 24, NEGATION_SEEDS[key])
    proc = _watch(tmp_path=tmp_path, runs_dir=runs)
    assert proc.returncode == 0, f"{key} armed the watch:\n{proc.stdout}"
    assert _job_line(proc, 24)[0].startswith(bdw.NO_DROPS), proc.stdout


def test_that_same_record_plus_a_carried_drop_and_no_copy_is_dropped_uncopied(tmp_path):
    """The clause's second half over the SAME bytes: the negation line plus a carried
    drop line and no copy line. A negation neither arms nor disarms on its own."""
    runs = tmp_path / "autonomy-runs"
    _job(runs, 24, NEGATION_SEEDS["no_rung_line_printed"] + "\n\n" + DOOR_DROP)
    proc = _watch(tmp_path=tmp_path, runs_dir=runs)
    assert proc.returncode == 1, proc.stdout
    line = _job_line(proc, 24)[0]
    assert line.startswith(bdw.DROPPED_UNCOPIED), proc.stdout
    assert "drops=1 copies=0" in line, line


# ── clause 3: two patterns, neither matching the other's form ───────────────────────
def test_the_two_patterns_never_match_the_others_form():
    """Cross-matching would collapse the two literals, which
    `scripts/util/bench_assertion_findings.py:229` records as deliberate: "a caller that
    greps the door's stderr for the rung's decision must not find a bookkeeping line
    wearing the decision's prefix". Each landed form is run against the OTHER pattern, and
    each classifies as exactly one of the two — including as a record writes it, inside a
    numbered list item."""
    assert bdw.FIRE_RE.search(RUNG_COPY) is None, (
        "the fire pattern reaches the rung's copy line — a copied line would count as a "
        "drop and every clean run would read DROPPED_UNCOPIED")
    assert bdw.COPY_RE.search(DOOR_DROP) is None, (
        "the copy pattern reaches the door's drop line — a drop would count as its own "
        "witness and an uncopied duty could never be seen")
    assert bdw.classify_line(DOOR_DROP) == (True, False)
    assert bdw.classify_line(RUNG_COPY) == (False, True)
    assert bdw.classify_line(f"3. {RUNG_COPY}") == (False, True)


def test_the_copy_pattern_matches_the_form_the_duty_tells_a_job_to_copy():
    """The duty names `BENCH ASSERTION:` and the emitter writes `BENCH ASSERTION: N bench
    card(s)`. A record that pasted the line under a heading, indented four, is the copy
    the duty is satisfied by — so indentation is decoration, not a disqualifier. What the
    pattern will not do is reach into a sentence: the corpus's `**0 `BENCH ASSERTION:`
    lines**` and `**No `BENCH ASSERTION:` line printed**` carry no card count and are
    negated besides."""
    assert bdw.classify_line("    " + RUNG_COPY) == (False, True)
    for shape in (NEGATION_SEEDS["zero_rung_lines"],
                  NEGATION_SEEDS["no_rung_line_printed"]):
        assert bdw.classify_line(shape) == (False, False), shape


def test_a_quoted_or_discussed_drop_is_not_a_fire():
    """The fire side refuses a lead carrying a word: a quotation of the emitter is not an
    observed drop. 234 lines across 112 files under ~/lloyd-data/sessions/ hold the
    door's prefix (measured 2026-11-25) and every one is a quote or an assert; these are
    its shapes."""
    for key, line in QUOTED_SEEDS.items():
        assert bdw.classify_line(line)[0] is False, (
            f"{key}: a quotation of the emitter read as an observed drop")


def test_the_fire_pattern_is_the_prefix_the_emitter_test_pins():
    """`tests/test_vault_commit_bench_assertion_refusal.py:271` pins
    `line.startswith("vault-commit.sh: BENCH ASSERTION DROP")` on the emitter's stderr, so
    the watch accepts exactly that prefix and refuses a moved one: an emitter edit turns
    this node red instead of turning the witness silently blind."""
    assert bdw.FIRE_RE.match(DOOR_DROP)
    assert bdw.FIRE_RE.match("vault-commit.sh: BENCH ASSERTION DROP: x is out of this commit")
    for reworded in (
        "BENCH ASSERTION DROP: x is out of this commit",   # the door's prefix is gone
        "vault-commit.sh: bench assertion drop: x",        # case moved
        "vault-commit.sh:BENCH ASSERTION DROP: x",         # separator gone
    ):
        assert bdw.FIRE_RE.match(reworded) is None, (
            f"the fire pattern tolerated a reworded emitter: {reworded!r}")


# ── clause 4: a duty-less job's fire is reported, never armed ───────────────────────
def test_a_fire_on_a_job_that_carries_no_copy_duty_never_moves_the_exit_status(tmp_path):
    """Task 44 (nightly-vault-maintenance) invokes the door and owes no copy — 10 skill
    files across 8 skills invoke `vault-commit.sh` and only 2 carry the duty, so this is
    the ordinary case. Its drop prints, is marked not armed, exit stays 0."""
    runs = tmp_path / "autonomy-runs"
    _job(runs, 44, DOOR_DROP)
    _job(runs, 24, "clean run")
    proc = _watch(tmp_path=tmp_path, runs_dir=runs)
    assert proc.returncode == 0, f"a duty-less drop armed the watch:\n{proc.stdout}"
    reported = _job_line(proc, 44)
    assert len(reported) == 1, proc.stdout
    assert reported[0].startswith(bdw.NO_DUTY), reported[0]
    assert "drops=1" in reported[0] and "not_armed" in reported[0], reported[0]
    assert bdw.DROPPED_UNCOPIED not in proc.stdout
    assert "armed_dropped_uncopied=0" in proc.stdout


def test_the_duty_set_is_derived_and_is_two_files_in_two_skills(tmp_path):
    """The scope lives in the vault, so it is scanned rather than hard-coded: 2 files in
    2 skills today, and removing the copy verb from one moves the set — the derivation is
    load-bearing, not decoration. The live vault has the same two files
    (`skills/autonomy-data-pipeline/step1-fact-extraction.md:325`,
    `skills/nightly-reflection-signals/SKILL.md:383`), measured 2026-11-25."""
    vault = _vault(tmp_path)
    assert bdw.duty_files(vault["skills"]) == [
        "autonomy-data-pipeline/step1-fact-extraction.md",
        "nightly-reflection-signals/SKILL.md"]
    assert bdw.duty_skills(vault["skills"]) == ["autonomy-data-pipeline",
                                               "nightly-reflection-signals"]
    stripped = _vault(tmp_path / "b", duty_skill_a=False)
    assert bdw.duty_skills(stripped["skills"]) == ["nightly-reflection-signals"], (
        "removing the copy verb left the skill in the duty set")


# ── clause 5: each store read and attributed; unreadable is not a zero ───────────────
def test_every_store_prints_its_own_count_and_an_unreadable_one_contributes_no_zero(tmp_path):
    """Four store lines, each naming itself, its counts, and whether it can attribute per
    job. Then the per-job store is removed: duty jobs must print `STORE_UNREADABLE` and
    the headline `answerable=no`, never `NO_DROPS` — "I could not look" must not print as
    "nothing happened". Clause 1 would call every absent job NO_DROPS; clause 5 wins."""
    runs = tmp_path / "autonomy-runs"
    _job(runs, 24, "clean run")
    # use_git=True: this node is about all four stores reporting, so the git one has to be
    # opened. The fixture vault IS a repo with a committed baseline, so `git status`
    # answers, and the line carries its count like the other three.
    proc = _watch(tmp_path=tmp_path, runs_dir=runs, use_git=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in ("run_records", "server_err", "workers_db", "vault_dirty"):
        line = _store_line(proc, name)
        assert "fires=" in line and "copies=" in line, line
    assert "bench_dirty_paths=0" in _store_line(proc, "vault_dirty"), (
        "the git witness did not report its own count")
    assert "per_job=yes" in proc.stdout and "per_job=no" in proc.stdout
    assert "answerable=yes" in proc.stdout and "UNREADABLE" not in proc.stdout

    # --no-git prints SKIPPED, not UNREADABLE: a store the caller chose not to open must
    # not wear the same word as one that failed, or the flag becomes a way to hide a real
    # failure. And a store nobody opened still contributes no zero.
    skipped = _watch(tmp_path=tmp_path, runs_dir=runs)
    assert "store vault_dirty: STORE_SKIPPED" in skipped.stdout, skipped.stdout
    assert "store vault_dirty: UNREADABLE" not in skipped.stdout

    gone = _watch(tmp_path=tmp_path, runs_dir=tmp_path / "no-such-runs")
    assert "store run_records: UNREADABLE" in gone.stdout, gone.stdout
    assert bdw.STORE_UNREADABLE in gone.stdout, gone.stdout
    assert not [l for l in gone.stdout.splitlines()
                if l.startswith(bdw.NO_DROPS) and "task=24" in l], (
        "an unreadable per-job store produced a NO_DROPS zero")
    assert "answerable=no" in gone.stdout
    assert gone.returncode == 0, gone.stdout  # unreadable never arms


def test_the_workers_db_read_counts_summary_rows_without_ordering_the_store(tmp_path):
    """`~/lloyd-data/workers.db` sits over the WITNESS_MAX_BYTES = 1,048,576 ceiling, so
    the watch reads `runs.summary` and nothing else. A marker in `summary` is counted and
    attributed to the store (never to a job); one in another column is not seen, and the
    line says how many candidate rows it looked at. Live figures 2026-11-25: 11,202 runs,
    0 summary candidates, and 0 rows whose run_id matches `run_24_%` — no autonomy run is
    in this table at all, which is why it cannot attribute."""
    db = tmp_path / "workers.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, source TEXT, summary TEXT)")
    con.execute("INSERT INTO runs VALUES ('r1','autocode',?)", (DOOR_DROP,))
    con.execute("INSERT INTO runs VALUES ('r2','autotriage',?)",
                (NEGATION_SEEDS["zero_rung_lines"],))
    con.execute("INSERT INTO runs VALUES ('r3','autocode','nothing bench-shaped')")
    con.commit()
    con.close()
    runs = tmp_path / "autonomy-runs"
    _job(runs, 24, "clean run")
    proc = _watch(tmp_path=tmp_path, runs_dir=runs, workers_db=db)
    line = _store_line(proc, "workers_db")
    assert "runs=3" in line and "summary_candidates=2" in line, line
    assert "fires=1" in line and "copies=0" in line, line
    assert "per_job=no" in line, (
        "workers.db cannot pin a drop to an autonomy job; it must not claim it can")


def test_server_err_prints_both_the_raw_mention_count_and_the_affirmative_one(tmp_path):
    """The item's premise — "server.err carries 0 door lines at all" (measured 2026-11-25:
    0 `vault-commit.sh:` lines in 70,408) — is the RAW count, and an affirmative 0 alone
    would not tell an empty store from one whose every line is a negation. So both print:
    `mentions=` raw, `fires=`/`copies=` the affirmative reading."""
    err = tmp_path / "server.err"
    err.write_text("INFO: booting\n"
                   + NEGATION_SEEDS["no_bench_assertion_drop_em_dash"] + "\n",
                   encoding="utf-8")
    runs = tmp_path / "autonomy-runs"
    _job(runs, 24, "clean run")
    proc = _watch(tmp_path=tmp_path, runs_dir=runs, server_err=err)
    line = _store_line(proc, "server_err")
    assert "mentions=1" in line, line
    assert "fires=0" in line and "copies=0" in line, line
    assert proc.returncode == 0


def test_a_fenced_example_is_counted_and_printed_rather_than_dropped(tmp_path):
    """A run record documenting the format inside a code block is not an observation.
    Skipped silently would be a hidden count, so it prints as `fenced_mentions=1`."""
    runs = tmp_path / "autonomy-runs"
    _job(runs, 24, "The line looks like this:\n\n```\n" + DOOR_DROP + "\n```\n\ndone.")
    proc = _watch(tmp_path=tmp_path, runs_dir=runs)
    assert proc.returncode == 0, f"a fenced example armed the watch:\n{proc.stdout}"
    assert _job_line(proc, 24)[0].startswith(bdw.NO_DROPS), proc.stdout
    line = _store_line(proc, "run_records")
    assert "fenced_mentions=1" in line and "fires=0" in line, line
