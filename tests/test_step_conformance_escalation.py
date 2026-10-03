"""#2104 clauses 2, 3, 4 and 5 — the one non-alerting escalation the replay may take.

The detector shipped as a report (`replay` prints and exits 0, and its own NOTE line says
`Nothing here alerts`) while the thing it detects — a run that recorded `success` after
skipping the stage that writes an artifact — is invisible unless somebody reads that print
every morning. The guardian's record says every rollback so far was a false positive, so the
answer is not an alert channel: a deviation naming a missing *written* artifact files a backlog
**draft** for a human to adjudicate, and nothing pages anybody.

The clauses, each pinned by the test that names it:

2. a flagged deviation whose missing step begins `write:` files exactly one draft; a flagged
   run whose missing step begins `tool:` files zero, and one missing a `read:` step files none;
3. the filed draft names the task id, the run id, the offending step and that deviation's
   `learned_from` run ids, and one escalating run never produces a second draft;
4. the escalation is confined — its only write is the draft, it makes no alert call, and the
   replay around it still mutates neither the ledger nor the trajectory store;
5. the autonomy task that runs it nightly names the read-only replay at strict support and no
   alert surface (marked `live_vault`: that clause is about a file in the live vault, which the
   gate's `-m "not live_vault"` run therefore does not judge).

The corpora are built with the helpers from `tests/test_step_conformance.py`, so every store
here is a real trajectory JSONL plus a real sqlite `runs` table read through the same read-only
loaders the nightly replay uses, and the filing is exercised both ways: through
`file_escalations` directly, and through the shipped CLI as a subprocess.

Why every corpus here has exactly one faulty run: at strict `support` an expectation needs
`n_runs - 1` of `n_runs`, so the moment two runs of one task miss the same step the step stops
being expected of either of them and nothing flags. Two escalating runs therefore come from two
tasks, which is also how the real board looks.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import test_step_conformance as base  # tests/ is on sys.path, same as board_presence


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "step_conformance.py"
AUTONOMY_DIR = Path("~/obsidian/autonomy").expanduser()

# The steps the escalation corpora learn, one of each kind, plus a second call of every kind
# that is never dropped: without it, removing `Write` would also remove `tool:Write`, and a
# test that flags two steps could not say which one the filer acted on.
PLAN = "/lloyd/_pipeline/reflection/plan-latest.md"
NOTES = "/lloyd/_pipeline/reflection/notes-latest.md"
SCRATCH = "/lloyd/_pipeline/reflection/scratch-latest.md"

DROPPABLE = {
    "write": ("Write", "/lloyd/_pipeline/reflection/report-latest.md"),
    "read": ("Read", PLAN),
    "tool": ("TodoWrite", None),
}
MISSING_STEP = {"write": "write:report-latest.md", "read": "read:plan-latest.md",
                "tool": "tool:TodoWrite"}


def _load_module():
    spec = importlib.util.spec_from_file_location("step_conformance", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load_module()


def _clean_calls():
    """Every clean run of a fixture task: read two files, log the plan, count the report,
    write the report, write a scratch note."""
    return [
        base._tool("Read", path=PLAN, seq=1),
        base._tool("Read", path=NOTES, seq=2),
        base._tool("TodoWrite", seq=3),
        base._tool("Bash", command=f"wc -c {base.REPORT}", seq=4),
        base._tool("Write", path=base.REPORT, seq=5),
        base._tool("Write", path=SCRATCH, seq=6),
    ]


def _drop(missing: str):
    """The same run with exactly one call never made."""
    name, path = DROPPABLE[missing]
    return [
        call for call in _clean_calls()
        if not (call["name"] == name
                and (path is None
                     or call["params_summary"].get("file_path") == path))
    ]


def _specs(count: int, *, faulty_at: int, missing: str | None,
           source: str = base.TASK, prefix: str = "run_ok"):
    """`count` runs of one fixture task, one of which never made one of its calls."""
    specs = []
    for i in range(count):
        calls = _drop(missing) if (i == faulty_at and missing) else _clean_calls()
        specs.append({
            "run_id": f"{prefix}_{i}",
            "session_key": f"2026091{i}_0500{i}0_autonomy_{prefix}{i}",
            "status": "success",
            "source": source,
            "task_id": int(source.rsplit(":", 1)[1]),
            "calls": calls,
        })
    return specs


def _replay(root: Path, specs, **kwargs):
    """Score a corpus the way the nightly job does: the real loaders, the real `replay`."""
    traj, db = base._write_corpus(root, specs)
    return sc.replay(trajectory_dir=traj, db_path=db, **kwargs)


def _flagged(result) -> list[dict]:
    return [r for r in result["reports"] if r["deviations"]]


def _steps(report: dict) -> list[str]:
    return [d["step"] for d in report["deviations"]]


def _drafts(backlog: Path) -> list[Path]:
    return sorted(backlog.glob("*.md"))


# ---------------------------------------------------------------------------
# the input the escalation path is handed
# ---------------------------------------------------------------------------


def test_a_run_that_never_wrote_its_artifact_is_flagged_on_one_missing_write_step(tmp_path):
    """Precondition for clauses 2 and 3, and the control on the fixture: the flagged run names
    `write:<artifact>` and nothing else, carries the run id it was judged on, and carries the
    run ids that expectation was learned from. If this corpus flagged two steps the filing tests
    below could not say which one the filer acted on; if it flagged none they would be grading
    an empty input."""
    result = _replay(tmp_path, _specs(6, faulty_at=5, missing="write"))
    flagged = _flagged(result)
    assert len(flagged) == 1, [r["run_id"] for r in flagged]
    assert _steps(flagged[0]) == ["write:report-latest.md"]
    assert flagged[0]["run_id"] == "run_ok_5"
    assert flagged[0]["deviations"][0]["learned_from"] == [f"run_ok_{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# clause 2 — write: files one draft; tool: and read: file nothing
# ---------------------------------------------------------------------------


def test_a_missing_written_artifact_files_exactly_one_backlog_draft(tmp_path):
    """Clause 2, first half. One flagged run whose missing step is `write:` produces one
    `status: draft` item on disk — not `up_next`, because a deviation on unlabelled traffic is
    a candidate, and promoting it would hand the autocode loop work nobody adjudicated."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    filed = sc.file_escalations(result, backlog_dir=backlog)
    assert len(filed) == 1
    drafts = _drafts(backlog)
    assert len(drafts) == 1
    assert str(drafts[0]) == filed[0]["path"]
    front_matter = drafts[0].read_text(encoding="utf-8").split("---")[1]
    assert "status: draft" in front_matter, front_matter
    assert sc.ESCALATION_TAG in front_matter, front_matter


@pytest.mark.parametrize("missing", ["tool", "read"])
def test_a_missing_tool_call_or_read_files_no_draft_at_all(tmp_path, missing):
    """Clause 2, second half — the half that keeps the filing rate readable. A run that never
    called `TodoWrite`, or never read a file every other run read, may have taken an equivalent
    path: those steps name a *tool*, not an artifact. Only a missing written artifact is evidence
    that a stage did not happen, so both of these flag and neither files."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing=missing))
    flagged = _flagged(result)
    assert _steps(flagged[0]) == [MISSING_STEP[missing]]
    assert sc.escalation_targets(result) == []
    assert sc.file_escalations(result, backlog_dir=backlog) == []
    assert _drafts(backlog) == []


def test_a_clean_corpus_files_no_draft(tmp_path):
    """The control over the filer itself: six runs that all took the same shape flag nothing, so
    the one draft above cannot be credited to a filer that files whenever it is called."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=None, missing=None))
    assert _flagged(result) == []
    assert sc.file_escalations(result, backlog_dir=backlog) == []
    assert _drafts(backlog) == []


# ---------------------------------------------------------------------------
# clause 3 — the draft carries the evidence, and one run never files twice
# ---------------------------------------------------------------------------


def test_the_filed_draft_names_the_task_run_offending_step_and_learned_from_runs(tmp_path):
    """Clause 3, first half. Every identifier a reader needs to reproduce the judgement is in the
    item body: task id, session source, run id, session key, the offending step id, and the run
    ids the expectation was learned from — the same evidence `--json` prints, in prose."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    sc.file_escalations(result, backlog_dir=backlog)
    text = _drafts(backlog)[0].read_text(encoding="utf-8")
    assert "task_id: 7" in text
    assert "session_source: autonomy-task:7" in text
    assert "run_id: run_ok_5" in text
    assert "missing_step: write:report-latest.md" in text
    assert "session_key: 20260915_050050_autonomy_run_ok5" in text
    for run_id in [f"run_ok_{i}" for i in range(5)]:
        assert run_id in text, f"learned_from run {run_id} missing from the draft"


def test_one_escalating_run_never_produces_a_second_draft(tmp_path):
    """Clause 3, second half. The nightly replay re-scores the same history for days: Monday's
    deviation is still Tuesday's. Dedupe is against every name already on the board, so the
    second pass files nothing and allocates no id."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    first = sc.file_escalations(result, backlog_dir=backlog)
    second = sc.file_escalations(result, backlog_dir=backlog)
    assert len(first) == 1
    assert second == [], second
    assert len(_drafts(backlog)) == 1, [p.name for p in _drafts(backlog)]


def test_two_escalating_runs_file_two_drafts(tmp_path):
    """Dedupe is per run, not per task or per step: the incident #673 was filed over was a task
    failing its stage twice. At strict support two runs of *one* task missing the same step stop
    making it expected at all, so the two runs come from two tasks — which is also the case where
    collapsing them would hide half the evidence."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    specs = _specs(6, faulty_at=5, missing="write")
    specs += _specs(6, faulty_at=2, missing="write",
                    source="autonomy-task:8", prefix="run_other")
    result = _replay(corpus, specs)
    assert len(_flagged(result)) == 2, [(r["task_id"], _steps(r)) for r in _flagged(result)]
    assert len(sc.file_escalations(result, backlog_dir=backlog)) == 2
    assert len(_drafts(backlog)) == 2


def test_a_dry_run_prints_the_answer_and_writes_nothing(tmp_path):
    """The operator's way to see the filing rate before accepting it — and the control that the
    one draft above came from `new_item`, not from a filer that always writes."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    would = sc.file_escalations(result, backlog_dir=backlog, dry_run=True)
    assert len(would) == 1
    assert would[0]["path"] == ""
    assert _drafts(backlog) == []


# ---------------------------------------------------------------------------
# clause 4 — confined and non-alerting, across the seam the nightly job crosses
# ---------------------------------------------------------------------------


def _hash_tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_shipped_cli_files_the_draft_and_leaves_both_stores_byte_identical(tmp_path):
    """Clause 4 across the process boundary the nightly job actually crosses: `python3
    scripts/step_conformance.py replay --escalate-writes` as a subprocess, which is how the
    autonomy task invokes it. The draft appears, the escalation count prints beside its
    denominator, and the two stores the replay read — the ledger and the trajectory JSONL — are
    byte-for-byte what they were. Without this the filer is only proven against an in-process
    call no scheduler makes."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    base._write_corpus(corpus, _specs(6, faulty_at=5, missing="write"))
    before = _hash_tree(corpus)
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "replay", "--escalate-writes",
         "--trajectories", str(corpus / "trajectories"), "--db", str(corpus / "workers.db"),
         "--backlog-dir", str(backlog)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "escalations_filed=1/1" in proc.stdout, proc.stdout[-2000:]
    assert len(_drafts(backlog)) == 1
    assert "write:report-latest.md" in _drafts(backlog)[0].read_text(encoding="utf-8")
    assert _hash_tree(corpus) == before, "replay mutated the ledger or the trace store"


def test_replay_without_the_flag_files_nothing(tmp_path):
    """Clause 4's other half, and the item's clause 2: read-only is the default. Same corpus,
    same subprocess, no `--escalate-writes` — the report prints, the flagged run is still
    reported, and no draft is filed. A human running the replay to look at something cannot put
    work on the board by looking at it."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    base._write_corpus(corpus, _specs(6, faulty_at=5, missing="write"))
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "replay",
         "--trajectories", str(corpus / "trajectories"), "--db", str(corpus / "workers.db")],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "runs_flagged=1/" in proc.stdout
    assert "escalations_filed=" not in proc.stdout, "escalated without being asked"
    assert _drafts(backlog) == []


def test_the_escalation_path_makes_no_alert_call_and_writes_no_store():
    """Clause 4's confinement, read off the shipped source rather than asserted in prose, in two
    shapes. (a) The module names no alert surface anywhere: not the injection tool, and not the
    two elevated priorities it would be injected at. (b) Inside `file_escalations` — the one
    function allowed to write — the only write call is the shared backlog writer's own
    `new_item`: no sqlite handle, no statement, no notification symbol. A whole-module grep could
    not say (b), because `synth_fixtures` legitimately `INSERT`s into a temp ledger it owns."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("session_inject_context", "notable", "urgent"):
        assert forbidden not in source, f"{forbidden} appears in the shipped detector"

    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "file_escalations")
    called = {node.func.id for node in ast.walk(fn)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    called |= {node.func.attr for node in ast.walk(fn)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "new_item" in called, "the filer must write through the shared backlog writer"
    for forbidden in ("connect", "execute", "executescript", "session_inject_context",
                      "discord_notify", "notify", "urlopen", "requests", "post", "send"):
        assert forbidden not in called, f"file_escalations calls {forbidden}"
    body = ast.get_source_segment(source, fn) or ""
    for statement in ("INSERT ", "UPDATE ", "DELETE "):
        assert statement not in body, f"file_escalations carries the statement {statement!r}"


def test_the_draft_is_the_only_file_the_escalation_creates(tmp_path):
    """Confinement measured rather than read: the whole temp root is hashed before and after a
    filing call, so anything the escalation writes outside the backlog directory — a cache, a
    state file, a copy of its own report — shows up as an unexpected new path."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "escalation"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    before = set(_hash_tree(tmp_path))
    sc.file_escalations(result, backlog_dir=backlog)
    added = set(_hash_tree(tmp_path)) - before
    assert len(added) == 1, sorted(added)
    assert added.pop().startswith("escalation/"), "wrote outside the backlog directory"


# ---------------------------------------------------------------------------
# clause 5 — the nightly task definition, in the live vault (marked, not skipped)
# ---------------------------------------------------------------------------


@pytest.mark.live_vault
def test_the_nightly_task_file_runs_the_replay_at_strict_support_with_no_alert_surface():
    """Clause 5. The task that makes the shipped detector run is a vault file landed through
    `automod_vault_land`, so it is read back through the real autonomy parser — a front-matter
    field the loader cannot read is a task that never dispatches, which is the exact failure this
    item is about. Marked `live_vault` because the gate's `-m "not live_vault"` run must not
    judge a file a human or a nightly job can rewrite between rounds."""
    from app.autonomy import _parse_task_file  # the real loader, not a re-implementation

    matches = sorted(AUTONOMY_DIR.glob("*-step-conformance-replay.md"))
    assert len(matches) == 1, [p.name for p in matches]
    path = matches[0]
    task = _parse_task_file(path)
    assert task, f"{path} does not parse as an autonomy task"
    description = str(task.get("description") or "")
    assert "scripts/step_conformance.py replay --days 7 --support 1.0" in description
    assert "--escalate-writes" in description, "the nightly run is the filer's only caller"
    assert "session_inject_context" not in description
    assert str(task.get("status", "")).strip() != "draft", "a draft never dispatches"
    # Both readers default a missing field to True, so the file has to carry the `false`: a
    # completion notice is an alert surface, which is the one thing this task may not name.
    assert task.get("notify_on_complete", True) is False, "the task declares a completion notice"
    skill = str(task.get("skill_name") or "").strip()
    assert skill, "dispatch refuses a task with no skill_name"
    skill_path = (Path("~/obsidian/skills") / skill / "SKILL.md").expanduser()
    assert skill_path.is_file(), f"skill_name names no skill on disk: {skill}"
    skill_text = skill_path.read_text(encoding="utf-8")
    assert "replay --days 7 --support 1.0" in skill_text
    for surface in ("session_inject_context", "notable", "urgent", "discord", "email"):
        assert surface not in skill_text, f"the skill names an alert surface: {surface}"


#: The clause-5 witness, committed in the diff rather than read from the live vault. The
#: autonomy file itself is landed by `automod_vault_land` and so cannot ride in this diff —
#: a gate that judges a file its own diff cannot carry grades an absence — and the
#: `live_vault` node above is deselected by the gate's `-m "not live_vault"`. This copy is
#: what makes the diff carry a witness: the real loader parses it, and the live node is what
#: notices the copy has drifted from the file the scheduler actually reads.
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/step_conformance_replay"
FIXTURE_TASK = FIXTURE_DIR / "nightly-task.md"
FIXTURE_SKILL = FIXTURE_DIR / "SKILL.md"


@pytest.fixture(scope="module")
def nightly_task():
    from app.autonomy import _parse_task_file  # the real loader, not a re-implementation

    task = _parse_task_file(FIXTURE_TASK)
    assert task is not None, f"the real loader cannot read {FIXTURE_TASK}"
    return task


def test_the_nightly_task_fixture_runs_the_replay_at_strict_support(nightly_task):
    """Clause 5, graded on a copy the diff itself carries: the command is the read-only
    replay at strict support (support 1.0 is the shipped default, so the task states it
    anyway and this node asserts the number) and the file names no alert surface."""
    task = nightly_task
    # `_parse_task_file` exposes no `command` key, so the command the scheduler runs is the one
    # in the description — which is also what `_build_task_prompt` renders to the model.
    description = str(task.get("description") or "")
    assert "scripts/step_conformance.py replay --days 7 --support 1.0" in description
    assert str(task.get("status", "")).strip() == "up_next", "a draft never dispatches"
    for surface in ("session_inject_context", "notable", "urgent", "email"):
        assert surface not in description, f"the description names an alert surface: {surface}"
    assert task.get("notify_on_complete", True) is False, "the task declares a completion notice"


def test_the_nightly_task_fixture_names_a_skill_that_exists_on_disk(nightly_task):
    """Clause 5's other half: dispatch refuses a task with no `skill_name`, and
    `_build_task_prompt` renders a task whose skill file is missing into a stub, so the
    skill has to exist and be the protocol the report is run from."""
    skill = str(nightly_task.get("skill_name") or "").strip()
    assert skill, "dispatch refuses a task with no skill_name"
    assert FIXTURE_SKILL.is_file(), "the committed skill copy is missing"
    text = FIXTURE_SKILL.read_text(encoding="utf-8")
    assert f"name: {skill}" in text, f"the copy is not the skill the task names: {skill}"
    assert "scripts/step_conformance.py replay --days 7 --support 1.0" in text
    for surface in ("session_inject_context", "notable", "urgent", "discord"):
        assert surface not in text, f"the skill names an alert surface: {surface}"


def test_the_nightly_task_fixture_copy_has_not_drifted_from_the_live_task():
    """The fixture is a copy, so it only witnesses clause 5 while it matches the file the
    scheduler reads. Red here means the vault task changed and the committed witness is
    stale — update the copy, or the file, not this node."""
    live_task = sorted(AUTONOMY_DIR.glob("*-step-conformance-replay.md"))
    live_skill = (Path("~/obsidian/skills") / "step-conformance-replay" / "SKILL.md").expanduser()
    if not live_task or not live_skill.is_file():
        pytest.skip("the vault task or its skill has not landed yet")
    assert FIXTURE_TASK.read_text(encoding="utf-8") == live_task[0].read_text(encoding="utf-8"), (
        f"{FIXTURE_TASK} and {live_task[0].name} disagree")
    assert FIXTURE_SKILL.read_text(encoding="utf-8") == live_skill.read_text(encoding="utf-8"), (
        f"{FIXTURE_SKILL} and {live_skill} disagree")
