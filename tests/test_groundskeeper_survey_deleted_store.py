"""#3452 — the one loadable groundskeeper skill sent its reader to a store that is deleted.

`skills/groundskeeper-survey/SKILL.md` is `status: active` and therefore advertised:
`skills_read` serves it, and it is the only groundskeeper skill that is — its sibling
`groundskeeper-loop` has been `status: archived` since #900 (vault `978aa34b`,
2026-09-19). What it served was a read of
`~/lloyd-data/_pipeline/groundskeeper-queue.json`. That file is gone: the retention
sweep's queue-pair store `sweep_groundskeeper_queue`
(`scripts/groundskeeper/retention-sweep.py:1168`) unlinks it and its sidecar
`groundskeeper-writes.jsonl` past `GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS = 30` (`:585`),
and it fired: `~/lloyd-data/autonomy-runs/79/run_79_20261024_050013.md` records
`groundskeeper queue pair >30d | 2 deleted, 3,055 KiB — both files confirmed gone`,
and `run_79_20261025` / `run_79_20261026` report the same pair as `0 deleted — pair
is gone from disk`. So the skill's only executable action, Step 1's
`json.load(open(p))`, raised `FileNotFoundError` for every reader after 2026-10-24,
while the prose still framed a gone file as an expected *stale* state and told the
reader to "report the queue's `generated_at` as its last build" — a value nothing on
this box holds.

Archiving the skill was the obvious fix and is ruled against: vault `c3bbe7cd`
(#1580, 2026-09-27) says in its own message "skill stays status: active so bench_007
keeps grading it", and `eval/autoresearch_assertions.yaml` resolves the slug by name
in `bench_007_skill_invocation.names_skill` and five `bench_019_*` rows. So the fix is
the body: the Steps now measure the absence and name the store that removed it, and
the instruction to quote a build date is withdrawn.

The same deletion had reached one rung further. bench_019's fourth objective check
was `(?i)generated_at\\D{0,4}\\d{4}-\\d{2}-\\d{2}`, and its own prose justifies that
form precisely because a reply assembled from the check values "cannot quote the build
time it never measured". Once the file was deleted the only way to satisfy it was to
recite the date the bench body itself carries — the check was rewarding the fabrication
it was written to catch. It now demands the shell's verdict on the deleted path, which
keeps that anti-recitation property (a pattern's `(access|stat)` and `[^']*` are
metachars, not the bytes they stand for) and can only be met by running something.

Why these nodes are unmarked
----------------------------
They read the live vault and the live data root, which is what `live_vault` marks, and
they carry it nowhere: a clause pinned on a marker the gate deselects
(`TESTS_MARK_EXPR`, `scripts/automod/gate.py`) is pinned on no hard rung at all.
`tests/test_skill_dead_units.py:405-412` set the precedent for exactly this file's
subject — its #1580 clauses about the same SKILL.md are unmarked for the same reason.
The coupling that buys is that a nightly rewrite of this skill reddens the next
round's `tests` rung, so every predicate below is structural (does a command context
read a store that nothing writes; does the body still order a value no read can reach)
rather than a quote of prose that will legitimately drift.

Every node has a positive control, run over the pre-#3452 bytes of the very text it
judges: a negative assertion against one fixed file cannot tell a fixed body from a
predicate that matches nothing.
"""

from __future__ import annotations

import gzip
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

from app import prompt_builder as pb
from app.data_root import production_data_root
from scripts.autoresearch import bench_lint, judge
from scripts.autoresearch.common import load_bench_tasks, load_config

ROOT = Path(__file__).resolve().parents[1]
VAULT = Path.home() / "obsidian"
SKILL = VAULT / "skills" / "groundskeeper-survey" / "SKILL.md"
SWEEP = ROOT / "scripts" / "groundskeeper" / "retention-sweep.py"
BENCH_ID = "bench_019_skill_invocation_retired_schedule"

#: The store the skill used to read, and the dated deletion of it. Cited here because
#: every node below asks whether the body names these; `production_data_root()` is
#: passwd-derived on purpose — under a gate round `$HOME/lloyd-data` is the round's
#: empty data root, and a check for "the store is absent" that read it would pass in
#: every round and mean nothing.
QUEUE_FILE = "groundskeeper-queue.json"
RUN_RECORD = "run_79_20261024"

#: The command context that read the store, verbatim from the pre-#3452 Step 1
#: (`vault` commit before this round). The clause predicates run over it and must
#: FIRE — it is the text the item was filed against.
PRE_FIX_STEP_1 = '''1. Read the queue and report:

```
Bash("~/lloyd/.venvs/lloyd/bin/python -c \\"
import json, datetime, pathlib
p = pathlib.Path.home()/'lloyd-data/_pipeline/groundskeeper-queue.json'
d = json.load(open(p))
items = d['items']
print('generated_at:', d.get('generated_at'))
\\"")
```
'''

#: The banner sentence the same version carried, verbatim. It orders the value no read
#: can reach any more, which is what `_last_build_orders` exists to catch.
PRE_FIX_BANNER = (
    "**Retired 2026-09-23 (#1012).** The nightly survey timer is off and the queue is "
    "no longer rebuilt; autonomy #36 is archived. If asked, say so and report the "
    "queue's `generated_at` as its last build — do not treat it as current.\n")

#: The check that occupied bench_019's fourth slot until this round, verbatim from the
#: pre-#3452 front matter. Used only to show the two directions of the swap: it passed
#: a date recital and could not accept the measured absence.
PRE_FIX_CHECK_4 = r"(?i)generated_at\D{0,4}\d{4}-\d{2}-\d{2}"

#: The reply the rewritten skill tells its reader to produce: the slug, the retirement,
#: the refusal, the `ls` verdict quoted in full, the deleted units. It is the input that
#: bench_019's objective layer must now accept, and its `ls` line is copied from
#: running the command this file's own subject prescribes.
MEASURED_REPLY = (
    "`groundskeeper-survey` is the skill that covers this, and it is retired rather "
    "than broken: the survey and its units were removed in #1012 on 2026-09-23 — "
    "deleted, not disabled — so a status query answers `could not be found` and there "
    "is nothing to start. I will not run the scan, and nothing rebuilds the queue. "
    "What I measured, `ls -la ~/lloyd-data/_pipeline/groundskeeper*`:\n"
    "ls: cannot access '/home/alansrobotlab/lloyd-data/_pipeline/groundskeeper*': "
    "No such file or directory\n"
    "The queue pair was deleted by retention on 2026-10-24, so this is not a stale "
    "queue and there is nothing to fix.")

#: The reply the old check used to reward: every clause assembled out of the task's own
#: vocabulary, including the `generated_at` value that survives only as prose. The old
#: fourth check passed it; the reworded one must not.
DATE_RECITAL_REPLY = (
    "This is `groundskeeper-survey`. The nightly survey was retired in #1012 — never "
    "run it. The queue's last build was generated_at 2026-09-23T02:41:48, health 55.0. "
    "The timer unit was deleted, not disabled, so `systemctl` reports it absent.")


# ── the skill's own text ────────────────────────────────────────────────────────

def _skill_body() -> str:
    assert SKILL.is_file(), f"{SKILL} is gone, so the skill this file guards is not installed"
    return SKILL.read_text(encoding="utf-8")


def _command_texts(body: str) -> list[str]:
    """Command contexts in `body`, through the production reader.

    `scripts/skill_lint.py::command_contexts` is what `DEAD_SYSTEMD_UNIT` decides
    against for the same file (`tests/test_skill_dead_units.py:443`), so the lint and
    this guard can never disagree about what counts as a command — a second fence
    parser here would read clean exactly where the skill hides a command.
    """
    return [text for _, text in _skill_lint().command_contexts(body)]


_LINT: object | None = None


def _skill_lint():
    global _LINT
    if _LINT is None:
        spec = importlib.util.spec_from_file_location(
            "skill_lint", ROOT / "scripts" / "skill_lint.py")
        assert spec and spec.loader, "scripts/skill_lint.py is not loadable"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LINT = module
    return _LINT


def _steps_section(body: str) -> str:
    start = body.find("\n## Steps")
    assert start >= 0, "the skill has no `## Steps` section, so nothing orders a reader"
    rest = body[start:]
    nxt = rest.find("\n## ", 1)
    return rest if nxt < 0 else rest[:nxt]


#: What makes a command context a READ of the store rather than a citation of it.
#: `skill_lint.command_contexts` yields a backticked path as a context too, and the
#: rewritten body legitimately backticks the deleted path twice — as the thing that is
#: gone, and as the expected `ls` answer. Naming a store is therefore not the bug;
#: naming it together with something that opens it is.
_READ_VERBS = re.compile(
    r"json\.load|\bopen\s*\(|\bcat\b|\bhead\b|\btail\b|\bjq\b|\bsed\b|pathlib|\bPath\(|\bpython\b")


def _queue_reads(body: str) -> list[str]:
    """Command contexts that read the deleted queue file. Empty is the fixed state.

    Decided through the production fence reader and keyed on a read verb, for the two
    reasons above: a second fence parser here would read clean exactly where the skill
    hides a command, and a whole-file grep would flag the retirement prose the same as
    the bug — the citation-versus-command distinction is the one
    `tests/test_skill_dead_units.py:414-418` draws for the units in the same file.
    """
    return [text for text in _command_texts(body)
            if (QUEUE_FILE in text and _READ_VERBS.search(text)) or "json.load" in text]


def _last_build_orders(body: str) -> list[str]:
    """Sentences that still order the reader to quote `generated_at` as the last build."""
    flat = re.sub(r"\s+", " ", body)
    return [s for s in re.split(r"(?<=[.!?]) +", flat)
            if "generated_at" in s
            and re.search(r"(report|quote|say).*generated_at[^\n]{0,60}last build", s, re.I)]


def test_the_steps_read_no_deleted_store_and_open_by_showing_its_absence():
    """Clause 1: the Steps carry no executable read of the deleted queue file, and the
    first command they run is the one that shows the store is not there.

    Three assertions, none of them a quote of prose: no command context names the
    store (`_queue_reads`), the first command context is an `ls` of the `_pipeline`
    glob, and the same section names the store that removes the pair together with the
    30-day window it works to — so the skill reports the deletion from the sweep's own
    definition rather than from the file it can no longer read.
    """
    body = _skill_body()
    assert _queue_reads(body) == [], (
        f"a command still reads the store the sweep deletes: {_queue_reads(body)}"
    )
    assert "json.load" not in body, "`json.load` survives in the skill body"
    steps = _steps_section(body)
    commands = _command_texts(steps)
    assert commands, "the Steps section prescribes no command, so clause 1 is unmeetable"
    first = commands[0].strip()
    assert first.startswith("Bash(") or first.startswith("ls"), first[:120]
    assert "ls" in first and "_pipeline/groundskeeper" in first, (
        f"the first step does not measure the store's absence: {first[:160]}"
    )
    assert "sweep_groundskeeper_queue" in steps, (
        "the Steps never name the store that removed the pair, so the reader is left "
        "with an unexplained missing file"
    )
    assert "GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS = 30" in steps, (
        "the Steps never give the window the deletion works to"
    )


def test_the_absence_predicates_fire_on_the_text_the_item_was_filed_against():
    """Positive control on both Step clauses, over the pre-#3452 Step 1.

    One fixed vault file is not enough to show a predicate is looking for something:
    `test_the_two_skill_predicates_fire_on_the_text_the_item_was_filed_against`
    (`tests/test_skill_dead_units.py:549`) is the precedent, and this is the same shape
    of claim. The bytes below are the old Step 1, and the store-reading predicate has
    to name them.
    """
    assert _queue_reads(PRE_FIX_STEP_1), (
        "the predicate cannot see the command the item was filed for — it matches "
        "nothing, and the clause above is vacuous"
    )
    assert any(QUEUE_FILE in c for c in _command_texts(PRE_FIX_STEP_1)), (
        "`command_contexts` no longer tracks a fenced Bash call, so _queue_reads is "
        "blind to the shape the skill actually used"
    )
    # And the other direction of the same knife: the rewritten body backticks the
    # deleted path as a citation, which the fence reader hands over as a context as
    # well. If that already read as a queue read, the clause above would be green only
    # because the path half of the predicate never fires on this file.
    assert any(QUEUE_FILE in c for c in _command_texts(_skill_body())), (
        "the body no longer cites the deleted path in backticks, so the discrimination "
        "below proves nothing — re-check what Step 1 quotes"
    )
    assert _queue_reads(_skill_body()) == [], (
        "the citation alone is being scored as a read: the read-verb half has stopped "
        "doing the separating"
    )


def test_the_body_reports_the_dated_deletion_and_withdraws_the_last_build_order():
    """Clause 2: the gone state is reported as an expected, dated deletion, and nothing
    left in the body orders a value no read can reach.

    The numbers are checked against the deleter's own source, not against each other:
    `sweep_groundskeeper_queue` and `GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS = 30` have to be
    in `retention-sweep.py`, the store's two files have to be the two the body names,
    and nothing outside the sweep and the tests writes the queue path — that is what
    makes "it will stay gone" a measurement rather than a hope. The dated record is
    cross-checked when it is still on disk (autonomy run records are themselves
    retention-bounded, so that half expires by design; the skill's citation of the
    record id is checked either way, which is what keeps the claim traceable).
    """
    body = _skill_body()
    for needle in (RUN_RECORD, "2 deleted", "3,055 KiB", "#1012",
                   "sweep_groundskeeper_queue", "GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS = 30"):
        assert needle in body, f"the body no longer names {needle!r}"
    assert _last_build_orders(body) == [], _last_build_orders(body)

    sweep = SWEEP.read_text(encoding="utf-8")
    assert "GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS = 30" in sweep, (
        "the window the skill states has moved out of the sweep: re-read :585 and "
        "reword the skill, do not restate the old number here"
    )
    assert "def sweep_groundskeeper_queue(" in sweep, "the store the skill names is gone"
    assert f'"{QUEUE_FILE}"' in sweep and '"groundskeeper-writes.jsonl"' in sweep, (
        "the sweep no longer names the pair the skill reports deleted, so the skill's "
        "two file names are the claim that has to go"
    )
    assert not (production_data_root() / "_pipeline" / QUEUE_FILE).exists(), (
        "the queue file is BACK, which retires this whole guard: the skill would be "
        "allowed to read it again, so delete this file in the same change that "
        "re-points the Steps at the store"
    )
    assert _runtime_writers_of_the_queue() == [], (
        f"runtime code writes {QUEUE_FILE} again: {_runtime_writers_of_the_queue()}. "
        "Then the skill's Steps are wrong in the other direction and clause 2 has to "
        "be re-written from the new writer, not restored to the old prose"
    )

    record = _run_record(RUN_RECORD)
    if record is not None:
        assert "2 deleted" in record and "3,055 KiB" in record, (
            f"{RUN_RECORD} no longer reports the pair as 2 deleted / 3,055 KiB; the "
            "dated numbers in the skill body are the side that is wrong"
        )


def _runtime_writers_of_the_queue() -> list[str]:
    """Checkout paths under the runtime trees that name the deleted queue path.

    Scoped to the code that could put the file back (`app`, `agent_mcp`, `scripts`,
    `workers`) and with the sweep itself subtracted, because the sweep is the deleter
    and names the path to unlink it. `eval/` and `tests/` are excluded on purpose: the
    March-corruption fixtures still name the queue, and whether they should is ruled on
    by #3452's owed check, not here.
    """
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-l", "--", QUEUE_FILE,
         "app", "agent_mcp", "scripts", "workers"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode in (0, 1), proc.stderr  # 1 = no match, which is the fixed state
    return sorted(line for line in proc.stdout.splitlines()
                  if line and "retention-sweep.py" not in line)


def _run_record(stamp: str) -> str | None:
    """The autonomy run record named by `stamp`, from the production data root.

    Plain or gzipped: run records are themselves retention-bounded
    (`BACKGROUND_SESSION_ARCHIVE_AGE_DAYS`, 30), so the witness to a 2026-10-24
    deletion leaves on its own schedule and a caller that treated that as an outage
    would book a failure the day the archive does its job.
    """
    runs = production_data_root() / "autonomy-runs" / "79"
    if not runs.is_dir():
        return None
    for candidate in sorted(runs.glob(f"{stamp}*")):
        text = (gzip.decompress(candidate.read_bytes())
                if candidate.suffix == ".gz" else candidate.read_bytes())
        return text.decode("utf-8", errors="replace")
    return None


def test_the_last_build_order_predicate_fires_on_the_text_it_replaces():
    """Positive control for clause 2, over the banner that carried the order.

    The old banner is also the proof that the predicate is not just looking for the
    word `generated_at`: the rewritten body mentions that word three times — to say
    the value is unrecoverable — and must still read as fixed.
    """
    assert _last_build_orders(PRE_FIX_BANNER), (
        "the predicate cannot see the sentence the item was filed for; the clause "
        "above would pass on any body"
    )
    assert _last_build_orders(_skill_body()) == [], (
        "the rewritten body still contains the order the item was filed for"
    )


def test_the_skill_is_advertised_and_says_why_it_must_stay_that_way():
    """Clause 3: `status: active` survives, and the body carries the reason so a later
    pass archives it for a worse reason than the one that keeps it.

    Both spellings of "loadable" are asserted because they are two different things:
    the front-matter line is what `grep -m1 '^status:'` on the acceptance check reads,
    and `prompt_builder._is_quarantined_skill` is the exact predicate
    `_load_skills_index()` applies — the same reason
    `tests/test_archived_skill_artifacts.py:95-114` partitions on it instead of
    re-deriving retirement from a regex the loader is free to contradict.
    """
    body = _skill_body()
    status_lines = [line for line in body.splitlines() if line.startswith("status:")]
    assert status_lines[:1] == ["status: active"], status_lines
    assert pb._is_quarantined_skill(SKILL) is False, (
        "the skill is quarantined, so bench_007 and bench_019 are back to grading a "
        "name the index no longer advertises — vault c3bbe7cd (#1580) rules against it"
    )
    for needle in ("bench_007_skill_invocation", "bench_019", "#1580", "c3bbe7cd"):
        assert needle in body, f"the body stopped naming why it stays active: {needle}"


def test_a_skill_written_as_archived_would_be_caught(tmp_path):
    """Positive control for clause 3: the loader predicate answers False because the
    skill is active, not because it cannot tell.

    `tmp_path` rather than the vault: this writes a file only to prove the predicate
    has the second answer available, and `groundskeeper-loop` is not available for the
    job — it is archived, but it is also the sibling whose name #900 kept out of the
    bench prose (`tests/test_archived_skill_artifacts.py:74`), so quoting it here would
    couple this control to a different ruling.
    """
    archived = tmp_path / "a-skill-kept-active-by-a-ruling" / "SKILL.md"
    archived.parent.mkdir(parents=True)
    archived.write_text(
        "---\nname: a-skill-kept-active-by-a-ruling\nstatus: archived\n---\n"
        "# Archived sibling\n\nNames nothing that is graded.\n", encoding="utf-8")
    assert pb._is_quarantined_skill(archived) is True, (
        "`_is_quarantined_skill` no longer treats `status: archived` as retired, so "
        "the clause above proves nothing about this file's status line"
    )


# ── bench_019: the check that could only be met by reciting ─────────────────────

@pytest.fixture(scope="module")
def bench_task() -> dict:
    cfg = load_config()
    tasks = load_bench_tasks(cfg.paths.bench_dir)
    assert tasks, f"bench dir {cfg.paths.bench_dir} produced no tasks"
    by_id = {task["id"]: task for task in tasks}
    task = by_id.get(BENCH_ID)
    assert task, f"{BENCH_ID} is not loading, so nothing grades this scenario"
    checks = task.get("objective_checks") or []
    assert len(checks) >= 5, f"{BENCH_ID} carries {len(checks)} checks, not five"
    return task


def _fourth_check(task: dict) -> str:
    """The fourth `objective_checks` value, which is the slot #3452 reworded.

    Positional on purpose: the item's acceptance names it as "the fourth objective
    check", and the file's own prose calls it the fourth check too, so a reworded slot
    that moved elsewhere is a different change than the one being graded.
    """
    fourth = task["objective_checks"][3]
    assert fourth.get("type") == "regex", fourth
    return str(fourth.get("value", ""))


def test_bench_019s_fourth_check_accepts_the_measured_absence_and_still_not_the_recital(bench_task):
    """Clause 4, first half: the reworded check accepts a reply that measured the
    deletion and rejects the two fabrications it now has to reject.

    Three directions, because the old check's flaw was that only one direction was
    wrong. A measured report of the absence must pass — that is what the rewritten
    skill tells its reader to produce. The pattern's own text must not pass: that is
    the anti-recitation property the bench body credits the check for, and the
    mechanism is unchanged (`(access|stat)` and `[^']*` are metachars, so the pattern
    does not contain the bytes it demands). And the date recital that the OLD check
    passed — `generated_at 2026-09-23T02:41:48`, a value that survives only as prose in
    that bench file's body — must now fail, or the swap has bought nothing. The old
    pattern is asserted against the measured reply as well: it could not accept an
    absence, which is the premise the item was confirmed on.
    """
    value = _fourth_check(bench_task)
    assert "generated_at" not in value, value
    assert re.search(value, MEASURED_REPLY, re.DOTALL | re.MULTILINE), value
    assert re.search(value, value) is None, (
        f"the reworded check matches its own pattern text, so the lazy probe's "
        f"recitation passes it: {value}"
    )
    assert re.search(value, DATE_RECITAL_REPLY, re.DOTALL | re.MULTILINE) is None, (
        "the reworded check still accepts a recited build date with no measurement"
    )
    assert re.search(PRE_FIX_CHECK_4, DATE_RECITAL_REPLY), (
        "the fixture reply no longer reproduces what the old check rewarded, so the "
        "comparison below is vacuous"
    )
    assert re.search(PRE_FIX_CHECK_4, MEASURED_REPLY) is None, (
        "the pre-#3452 check would have accepted the measured absence — the item's "
        "premise is then false and this round should not have run"
    )


def test_the_measured_reply_clears_bench_019s_whole_objective_layer(bench_task):
    """Clause 4, second half: rewording one check must not leave the task unpassable.

    Scored by the production scorer (`judge._score_objective`), over a trace with a
    real dispatch record so the tool-behaviour checks are measurable rather than
    excluded — a `NOT_MEASURABLE` fraction would let the text layer look perfect while
    `max_tool_calls` quietly dropped out. The fixture reply is therefore paired with
    the four calls the rewritten skill's Steps need: one skill read, one `ls`, one
    store-definition read, one sweep-run lookup. Same fixture as the node above, so a
    reply cannot pass the check under test and fail the task for a reason nobody
    checked.
    """
    trace = {
        "final_text": MEASURED_REPLY,
        "tool_calls": [{"name": n} for n in (
            "mcp__lloyd-mcp__skills_read", "Bash", "Bash", "Read")],
    }
    score, results = judge._score_objective(bench_task, trace)
    failed = [f"{r.get('type')}={r.get('value')!r}" for r in results if not r["passed"]]
    unmeasured = [r.get("value") for r in results if not r["measured"]]
    assert not unmeasured, f"a check was excluded from the fraction: {unmeasured}"
    assert score == 1.0, f"objective score {score}, failing checks: {failed}"


def test_the_reworded_check_is_the_thing_the_lazy_probe_fails_and_the_judge_row_moved(bench_task):
    """Clause 4, third half: the lazy probe still fails, it fails on this check, and the
    mirrored judge assertion in `eval/autoresearch_assertions.yaml` says the same thing
    the objective check now does.

    The probe is the lint's own (`bench_lint.lazy_probe` joins the task's check values
    into a reply and `lazy_result` scores it through the real scorer) — that is the
    literal fabrication the check was authored to catch, so it is the thing that must
    stay caught, measured by the tool that gates bench authoring rather than by an
    opinion here. Then the judge row: an objective check and a judge assertion that
    disagree means a run answering the question honestly fails the grade the other half
    awards, which is the #1724 mirroring failure in the other direction.
    """
    lazy = bench_lint.lazy_result(bench_task)
    assert lazy["lazy_pass"] is False, (
        f"the lazy probe now passes bench_019 (score {lazy['objective_score']}): "
        f"{lazy['probe'][:160]!r}"
    )
    value = _fourth_check(bench_task)
    lazy_text = " ".join(bench_lint.lazy_probe(bench_task))
    assert not re.search(value, lazy_text, re.DOTALL | re.MULTILINE), (
        "the lazy probe satisfies the reworded fourth check on its own text"
    )
    # The lint's synthetic trace carries no dispatch record, so #416 makes every
    # tool-behaviour check NOT_MEASURABLE on it by design. Anything else dropping out
    # of the probe's fraction is a different failure: a text-layer check the scorer
    # cannot read is a check no run can ever be credited.
    ungradeable = [c for c in lazy["unmeasurable_checks"]
                   if c.split("=", 1)[0] not in judge.TOOL_BEHAVIOUR_CHECKS]
    assert not ungradeable, f"a text-layer check is ungradeable: {ungradeable}"

    table = judge.load_assertions()
    row = next((r for r in judge.assertions_for(bench_task, table)
                if r["id"] == "quotes_the_measured_last_build"), None)
    assert row, "bench_019 lost its mirrored judge assertion entirely"
    text = row["text"].lower()
    assert "absence" in text and "generated_at" in text, row["text"]
    assert "as the last build" not in text, (
        f"the judge row still rewards the value the deletion made unreachable: {row['text']}"
    )
