"""A dispatch-killing autonomy status change is named before commit, and every
non-scheduler writer records it: backlog #1127.

Two halves, and the split is the point.

The commit side: `status: draft` and `status: paused` are dispatch kill switches
— `_all_runnable_tasks` (`autonomy.py:323`) keeps only
`("up_next", "in_progress", "failed")`, and the queue's own source
(`workers/sources/scheduled_task.py:174`) skips anything that is not `up_next` —
yet no rung anywhere named such a transition before it was committed. Vault commit
`6f657fa9` moved #68 (`frequency: every-15min`, the fleet's highest-volume task)
from `up_next` to `draft` inside a nightly pre-flight commit whose own message
certified three *unrelated* prose files as unshrunk; #68 ran nothing for ~30 h
(~120 missed cycles) while looking configured.

The write side: three writers can set `status` from outside the scheduler
(`POST /api/autonomy/task-write`, `autonomy_write_task`, and
`autonomy_delete_task(archive=True)`), and none of them appended a line saying
which value it moved from and to. That is why #68's disable had no reason anywhere
and read as unattributable for a day.

Both halves REPORT. Neither blocks: `up_next -> in_progress` is a job claiming its
own task, and a guard that stops legitimate work gets switched off inside a week.

`scripts/util/vault-commit.sh` invokes `scripts/util/autonomy_status_findings.py`
between staging and committing, and the two tests that go through the shell
(`test_wrapper_reports_a_dispatch_stopping_transition`,
`test_wrapper_reports_the_same_finding_for_a_different_committer`) are the ones
that cross the process boundary — `--repo <scratch>` is how the same check is
reached from any job's own inline commit.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import autonomy as A
from app.routers import autonomy as ROUTER

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNG = REPO_ROOT / "scripts" / "util" / "autonomy_status_findings.py"
WRAPPER = REPO_ROOT / "scripts" / "util" / "vault-commit.sh"

TASK_BODY = "\n## Activity Log\n\n- 2026-09-13T02:45:25Z: run completed\n"


def _task_file(dirn: Path, task_id: int, name: str, status: str, **extra) -> Path:
    fm = {"type": "autonomy", "segment": "autonomy", "id": task_id, "name": name,
          "status": status, "frequency": "every-15min"}
    fm.update(extra)
    path = dirn / f"{task_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n{TASK_BODY}",
                    encoding="utf-8")
    return path


@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    import agent_mcp.autonomy as MCP
    monkeypatch.setattr(A, "AUTONOMY_DIR", dirn)
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", dirn)
    monkeypatch.setattr(MCP, "AUTONOMY_DIR", dirn)
    return dirn


@pytest.fixture
def client(autonomy_dir):
    app = FastAPI()
    app.include_router(ROUTER.router)
    with TestClient(app) as c:
        yield c


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True, text=True)


@pytest.fixture
def vault_repo(tmp_path):
    """A real git repo with a real commit, because the rung reads the index."""
    repo = tmp_path / "vault"
    repo.mkdir()
    (repo / "autonomy").mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _task_file(repo / "autonomy", 68, "Morning Brief Triage", "up_next")
    _task_file(repo / "autonomy", 40, "Nightly Reflection Config", "up_next")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


_STATUS_LINE = re.compile(r"(?m)^status:.*$")


def _set_status(repo: Path, filename: str, new_status: str) -> None:
    """Rewrite the file's `status:` to `new_status`, whatever it says now.

    Deliberately not a string replace of a fixed from-value: a test that walks
    `up_next -> paused -> up_next -> paused` has to be able to state the status it
    is leaving behind, or the second flip silently writes nothing and the test
    passes on an empty diff.
    """
    path = repo / "autonomy" / filename
    text = path.read_text(encoding="utf-8")
    new_text, n = _STATUS_LINE.subn(f"status: {new_status}", text, count=1)
    assert n == 1, f"no front-matter status line in {filename}"
    path.write_text(new_text, encoding="utf-8")


def _finding_line(printed: str, filename: str) -> str:
    """The single finding line the rung printed for `filename`, or fail.

    Compared whole, not with `in`: two committers printing the *same* sentence is
    the claim, and a substring test would accept half of it.
    """
    lines = [ln.strip() for ln in printed.splitlines()
             if "FINDING" in ln and filename in ln]
    assert len(lines) == 1, f"expected exactly one finding for {filename}, got {lines!r}"
    return lines[0]


def _run_rung(repo: Path) -> str:
    """Drive the rung the way a job's shell step does: system python3, --repo."""
    proc = subprocess.run(("python3", str(RUNG), "--repo", str(repo)),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"rung must never block a commit: {proc.stderr}"
    return proc.stdout


def _activity_lines(text: str) -> list[str]:
    """The `## Activity Log` bullets, whitespace-normalised.

    Compared as a list of bullets rather than as a body string because both
    writers re-serialise the whole file — `\n\n{body}` onto a body that already
    begins with a newline — so a no-status write legitimately shifts blank lines
    while adding no bullet. The claim here is about log LINES, which is what
    clause 5 is about; blank-line drift is a separate, pre-existing quirk of the
    two writers and not this item's to change.
    """
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ")]


def _status_of(path: Path) -> str:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---\n", 2)[1])["status"]


# ── the note itself ───────────────────────────────────────────────────────────


def test_a_transition_into_a_dispatch_stopping_status_names_both_values():
    note = A.status_change_note("up_next", "draft")
    assert "up_next" in note and "draft" in note
    assert "dispatch-stopping" in note


def test_a_transition_out_of_a_stopping_status_is_recorded_without_that_label():
    note = A.status_change_note("draft", "up_next")
    assert "draft" in note and "up_next" in note
    assert "dispatch-stopping" not in note


def test_no_note_when_the_status_does_not_move_or_is_absent():
    assert A.status_change_note("up_next", "up_next") is None
    assert A.status_change_note("up_next", None) is None
    assert A.status_change_note("up_next", "") is None


def test_the_rung_scripts_stopping_set_is_the_module_s():
    """Two definitions of one fact must not be allowed to drift.

    `autonomy_status_findings.py` spells the tuple out instead of importing
    `autonomy`, because it runs under whatever interpreter a shell has and
    `autonomy` pulls `yaml` / `agent_mcp._shared` / `app.paths` at module level.
    That is only safe while both name the same values, and this is the only thing
    that notices when they stop.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("asf", RUNG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert tuple(mod.DISPATCH_STOPPING_STATUSES) == tuple(A.DISPATCH_STOPPING_STATUSES)
    assert set(A.DISPATCH_STOPPING_STATUSES) == {"draft", "paused"}


# ── clauses 1-3: the pre-flight rung, driven through its own CLI ─────────────


def test_rung_names_file_and_both_values_for_draft_and_still_exits_zero(vault_repo):
    """Clause 1: `up_next` -> `draft` is printed with file, from-value, to-value."""
    _set_status(vault_repo, "68-morning-brief-triage.md", "draft")
    _git(vault_repo, "add", "-A")
    out = _run_rung(vault_repo)
    assert "autonomy-status FINDING:" in out
    assert "autonomy/68-morning-brief-triage.md" in out
    assert "up_next -> draft" in out
    assert "dispatch-stopping" in out


def test_rung_prints_the_equivalent_finding_for_paused(vault_repo):
    """Clause 2, first half: `paused` is a finding on the same code path."""
    _set_status(vault_repo, "68-morning-brief-triage.md", "paused")
    _git(vault_repo, "add", "-A")
    out = _run_rung(vault_repo)
    assert "autonomy/68-morning-brief-triage.md: status up_next -> paused" in out
    assert "dispatch-stopping" in out


def test_rung_prints_no_finding_when_front_matter_moves_but_status_does_not(vault_repo):
    """Clause 3: not a permanent alarm. `last_run` churn is what the scheduler
    writes into task files on every single run, so a rung that fired on it would
    be silenced the same night."""
    path = vault_repo / "autonomy" / "40-nightly-reflection-config.md"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "frequency: every-15min",
        "frequency: every-15min\nlast_run: '2026-09-18T06:48:56.217392+00:00'\n"
        "next_run: '2026-09-18T10:48:56.217392+00:00'", 1), encoding="utf-8")
    _git(vault_repo, "add", "-A")
    out = _run_rung(vault_repo)
    assert "FINDING" not in out
    # The denominator travels with the count: "0 findings" over "0 files checked"
    # is not the same verdict and must not read like one. Only the one file is
    # staged here, so the rung must say 1 checked — a rung that reported
    # "0 checked, 0 findings" could not tell a clean tree from a silent one.
    assert "1 staged autonomy task file(s) checked, 0 status finding(s)" in out


def test_rung_reports_a_claim_the_same_way_it_reports_a_clobber(vault_repo):
    """`up_next -> in_progress` is a job claiming its own task: named, unlabelled.

    The reason a reader needs the from/to pair at all is to tell these two apart
    without opening the file, so a claim prints and does NOT carry
    `dispatch-stopping`.
    """
    _set_status(vault_repo, "40-nightly-reflection-config.md", "in_progress")
    _git(vault_repo, "add", "-A")
    out = _run_rung(vault_repo)
    assert "autonomy/40-nightly-reflection-config.md: status up_next -> in_progress" in out
    assert "dispatch-stopping" not in out


def test_an_added_task_file_is_not_reported_as_a_transition(vault_repo):
    """New tasks legitimately start in `draft`; calling that a transition would
    make the rung fire on every task creation and teach every job to ignore it."""
    _task_file(vault_repo / "autonomy", 99, "Brand New Task", "draft")
    _git(vault_repo, "add", "-A")
    out = _run_rung(vault_repo)
    assert "99-brand-new-task.md" not in out
    assert "1 staged autonomy task file(s) checked, 0 status finding(s)" in out


def test_the_rung_reads_the_staged_tree_not_the_working_tree(vault_repo):
    """Staging is the unit. A dirty-but-unstaged flip is somebody else's in-flight
    edit — the same line vault-commit.sh draws everywhere else."""
    _set_status(vault_repo, "68-morning-brief-triage.md", "draft")
    out = _run_rung(vault_repo)                      # nothing staged
    assert "FINDING" not in out
    _git(vault_repo, "add", "autonomy/68-morning-brief-triage.md")
    assert "up_next -> draft" in _run_rung(vault_repo)


def test_front_status_reads_values_in_every_form_the_vault_holds():
    """The rung parses front matter with a regex, not a YAML loader. These are the
    shapes on disk: bare, single-quoted, double-quoted, and a quoted value
    carrying a trailing comment. Miss one and a flip reads as 'status unchanged'
    — which is the false all-clear this item exists to remove."""
    for value in ("up_next", "'up_next'", '"up_next"', "up_next  # claimed 09-17"):
        text = f"---\nid: 68\nstatus: {value}\n---\n\nbody\n"
        assert RUNG_FRONT(text) == "up_next", value
    assert RUNG_FRONT("---\nid: 68\n---\n\nbody\n") == ""      # no status key
    assert RUNG_FRONT("# just a note\n") == ""                  # no front matter


def RUNG_FRONT(text: str) -> str:
    import importlib.util

    spec = importlib.util.spec_from_file_location("asf_front", RUNG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.front_status(text)


# ── clauses 1-2 through the real boundary: the shell wrapper ─────────────────


def _write_wrapper_env(repo: Path):
    return {"VAULT_DIR": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin"}


def test_wrapper_reports_a_dispatch_stopping_transition_and_still_commits(vault_repo):
    """Clause 1 end to end: the finding is printed by the committing process and
    the commit lands anyway."""
    _set_status(vault_repo, "68-morning-brief-triage.md", "draft")
    proc = subprocess.run(("bash", str(WRAPPER), "pre-flight 2026-09-18 (demo)"),
                          capture_output=True, text=True, timeout=120,
                          cwd=str(vault_repo), env=_write_wrapper_env(vault_repo))
    assert proc.returncode == 0, proc.stderr
    printed = proc.stdout + proc.stderr
    assert "autonomy-status FINDING:" in printed
    assert "autonomy/68-morning-brief-triage.md: status up_next -> draft" in printed
    assert "dispatch-stopping" in printed
    head = _git(vault_repo, "log", "--format=%s", "-1")
    assert head.stdout.strip() == "pre-flight 2026-09-18 (demo)"
    assert "autonomy/68-morning-brief-triage.md" in _git(
        vault_repo, "show", "--stat", "--format=", "HEAD").stdout


def test_wrapper_reports_the_same_finding_for_a_different_committer(vault_repo):
    """Clause 2's 'regardless of which job is committing', tested rather than
    asserted: nothing here passes a job name, and the rung's only inputs are the
    repo and its index — so a nightly job, a pipeline, and this test get the same
    line for the same staged diff."""
    task = "68-morning-brief-triage.md"
    # Committer 1: the nightly pre-flight, through the wrapper.
    _set_status(vault_repo, task, "paused")
    first = subprocess.run(("bash", str(WRAPPER), "nightly-reflection: pre-flight"),
                           capture_output=True, text=True, timeout=120,
                           cwd=str(vault_repo), env=_write_wrapper_env(vault_repo))
    assert first.returncode == 0, first.stderr
    first_line = _finding_line(first.stdout + first.stderr, task)
    assert first_line.endswith("status up_next -> paused (dispatch-stopping)")

    # Committer 2: a different job, which restores the task and later re-parks it.
    # A job that stages named directories and commits inline
    # (nightly-reflection-knowledge-write does exactly that) reaches the identical
    # check by calling the script itself, so it gets the identical line.
    _set_status(vault_repo, task, "up_next")
    restore = subprocess.run(("bash", str(WRAPPER), "autonomy-data-pipeline: restore #68"),
                             capture_output=True, text=True, timeout=120,
                             cwd=str(vault_repo), env=_write_wrapper_env(vault_repo))
    # Every transition is reported and only the stopping direction is labelled, so
    # a restore prints and reads as a restore.
    restore_line = _finding_line(restore.stdout + restore.stderr, task)
    assert restore_line.endswith("status paused -> up_next")
    assert "dispatch-stopping" not in restore_line
    _set_status(vault_repo, task, "paused")
    _git(vault_repo, "add", "-A")
    second = subprocess.run(_rung_cmd(vault_repo), capture_output=True, text=True,
                            timeout=60)
    assert second.returncode == 0, second.stderr
    assert _finding_line(second.stdout, task) == first_line


def _rung_cmd(repo: Path) -> list[str]:
    return ["python3", str(RUNG), "--repo", str(repo)]


def test_wrapper_prints_no_status_finding_on_a_clean_front_matter_change(vault_repo):
    """Clause 3 through the shell: the wrapper's commit of a `last_run`-only change
    (what the scheduler writes on every run) prints no status finding, and commits.
    """
    path = vault_repo / "autonomy" / "40-nightly-reflection-config.md"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "id: 40", "id: 40\nlast_run: '2026-09-18T06:48:56+00:00'", 1), encoding="utf-8")
    proc = subprocess.run(("bash", str(WRAPPER), "autonomy-data-pipeline: 2026-09-18"),
                          capture_output=True, text=True, timeout=120,
                          cwd=str(vault_repo), env=_write_wrapper_env(vault_repo))
    assert proc.returncode == 0, proc.stderr
    assert "FINDING" not in proc.stdout + proc.stderr
    assert "0 status finding(s)" in proc.stdout + proc.stderr
    assert _git(vault_repo, "log", "--format=%s", "-1").stdout.strip() == \
        "autonomy-data-pipeline: 2026-09-18"


# ── clauses 4-5: POST /api/autonomy/task-write ───────────────────────────────


def test_the_http_writer_records_old_and_new_status_and_is_not_blocked(client, autonomy_dir):
    """Clause 4: the endpoint appends the line to the task's own markdown, still
    writes the new status, and answers 200 — a record, not a gate."""
    path = _task_file(autonomy_dir, 68, "Morning Brief Triage", "up_next")
    r = client.post("/api/autonomy/task-write", json={"id": 68, "status": "draft"})
    assert r.status_code == 200, r.text
    assert _status_of(path) == "draft"
    body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "status changed: up_next -> draft" in body
    assert "dispatch-stopping" in body
    assert "## Activity Log" in body


def test_the_http_writer_labels_only_the_stopping_direction(client, autonomy_dir):
    """The same endpoint restoring a task must not be made to look like a kill."""
    path = _task_file(autonomy_dir, 68, "Morning Brief Triage", "draft")
    r = client.post("/api/autonomy/task-write", json={"id": 68, "status": "up_next"})
    assert r.status_code == 200, r.text
    body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "status changed: draft -> up_next" in body
    assert "dispatch-stopping" not in body


def test_a_write_that_leaves_status_unchanged_appends_no_status_line(client, autonomy_dir):
    """Clause 5: the scheduler's own claim path (`up_next` -> `in_progress` via a
    body/front-matter touch, status identical) stays silent — otherwise every run
    appends a status line and the log stops meaning anything."""
    path = _task_file(autonomy_dir, 68, "Morning Brief Triage", "up_next")
    before = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    r = client.post("/api/autonomy/task-write",
                    json={"id": 68, "status": "up_next", "priority": "high"})
    assert r.status_code == 200, r.text
    after = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "status changed" not in after
    assert _activity_lines(after) == _activity_lines(before), \
        "an unchanged status must not add a log bullet"
    assert yaml.safe_load(path.read_text(encoding="utf-8").split("---\n", 2)[1])["priority"] == "high"


def test_the_http_writer_records_a_pause_as_dispatch_stopping(client, autonomy_dir):
    path = _task_file(autonomy_dir, 68, "Morning Brief Triage", "up_next")
    client.post("/api/autonomy/task-write", json={"id": 68, "status": "paused"})
    body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "status changed: up_next -> paused (dispatch-stopping)" in body


# ── clause 6: autonomy_write_task and its archive-to-draft path ─────────────


def _mcp():
    import agent_mcp.autonomy as MCP
    return MCP


def test_mcp_write_records_old_and_new_status(autonomy_dir):
    """Clause 6, first half: the tool agents actually call."""
    MCP = _mcp()
    _task_file(autonomy_dir, 68, "Morning Brief Triage", "up_next")
    out = json.loads(MCP._handle_write({"id": 68, "status": "draft"}))
    assert "error" not in out
    path = autonomy_dir / "68-morning-brief-triage.md"
    assert _status_of(path) == "draft"
    body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "status changed: up_next -> draft (dispatch-stopping)" in body


def test_mcp_write_keeps_a_callers_activity_note_and_adds_its_own(autonomy_dir):
    """The pre-existing `activity_note` behaviour must survive: one write, two
    lines, the caller's note still there."""
    MCP = _mcp()
    _task_file(autonomy_dir, 68, "Morning Brief Triage", "up_next")
    MCP._handle_write({"id": 68, "status": "paused",
                       "activity_note": "parked until the Thunderbird bridge returns"})
    body = (autonomy_dir / "68-morning-brief-triage.md").read_text(encoding="utf-8")
    assert "parked until the Thunderbird bridge returns" in body
    assert "status changed: up_next -> paused (dispatch-stopping)" in body


def test_mcp_write_with_no_status_change_appends_no_status_line(autonomy_dir):
    MCP = _mcp()
    _task_file(autonomy_dir, 68, "Morning Brief Triage", "up_next")
    before = (autonomy_dir / "68-morning-brief-triage.md").read_text(encoding="utf-8")
    MCP._handle_write({"id": 68, "priority": "high"})
    after = (autonomy_dir / "68-morning-brief-triage.md").read_text(encoding="utf-8")
    assert "status changed" not in after
    fm = yaml.safe_load(after.split("---\n", 2)[1])
    assert fm["priority"] == "high" and fm["status"] == "up_next"
    assert _activity_lines(after) == _activity_lines(before), \
        "an unchanged status must not add a log bullet"


def test_archive_to_draft_records_the_transition(autonomy_dir):
    """Clause 6, second half: `archive=True` writes `status: draft`, which is the
    kill switch, so archiving now says so — the case where 'nobody can say why
    this grant exists' applies to a task."""
    MCP = _mcp()
    _task_file(autonomy_dir, 68, "Morning Brief Triage", "up_next")
    out = json.loads(MCP._handle_delete({"id": 68, "archive": True}))
    assert out.get("success") is True
    path = autonomy_dir / "68-morning-brief-triage.md"
    assert _status_of(path) == "draft"
    body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "status changed: up_next -> draft (dispatch-stopping)" in body


def test_archiving_an_already_draft_task_records_nothing(autonomy_dir):
    MCP = _mcp()
    _task_file(autonomy_dir, 68, "Morning Brief Triage", "draft")
    json.loads(MCP._handle_delete({"id": 68, "archive": True}))
    body = (autonomy_dir / "68-morning-brief-triage.md").read_text(encoding="utf-8")
    assert "status changed" not in body


# ── the write side does not disturb dispatch, and the log stays parseable ────


def test_the_recorded_line_keeps_the_file_parseable_and_the_stopping_status_still_bites(autonomy_dir):
    """Append-only body text must not break the scheduler's own reader, and must
    not change what dispatch does with the file: `draft` is still not runnable."""
    MCP = _mcp()
    _task_file(autonomy_dir, 68, "Morning Brief Triage", "up_next")
    MCP._handle_write({"id": 68, "status": "draft"})
    parsed = A._parse_task_file(autonomy_dir / "68-morning-brief-triage.md")
    assert parsed["status"] == "draft"
    assert "run completed" in parsed["body"]   # the pre-existing log survived
    runnable = [t["id"] for t in A._all_runnable_tasks([parsed])]
    assert runnable == []
