"""#2877 — `scripts/maintenance/commit_injection_canary_corpus.py`, the corpus commit step.

The persistence-window unit's second `ExecStartPost` runs this script so the rows a fire
appends to the git-tracked `eval/measurements/injection-canary/` get recorded instead of
sitting in `git status --porcelain` until a human commits them — the state #2866 closed
with owed and never made: as of 2026-10-17 the whole-checkout status is exactly ` M
eval/measurements/injection-canary/rows.jsonl` and `??
eval/measurements/injection-canary/run-2026-10-15-persistence.md`.

What these nodes pin, in the item's clause order:

  * the ONE staging command, spelled, and that no other mutating git verb runs (#2);
  * that the commit takes the corpus and nothing else, carries the message the item
    specifies, and is authored at the repo default (#2);
  * that every path which commits nothing EXITS 0 — nothing staged, a foreign path
    already staged, a `git` that fails outright — because a non-zero `ExecStartPost` marks
    a `Type=oneshot` unit `Failed` and would bury a ~34 min window (#3);
  * that the caller has to BE the window unit, read off `/proc/self/cgroup` and not off
    `INVOCATION_ID` alone (#4), with the raw verbs still refused by
    `tests/test_safety_gate_discrimination.py`.

**How a test gets each arm, and why it is split that way.** The step's caller check reads
`/proc/self/cgroup`, and no test can put itself in the window unit's cgroup without
starting the window — which a round must never do. So the arms are split by what they need
to prove:

  * **behaviour past the gate** runs in-process, through `commit_corpus(...,
    cgroup_file=<witness>)`, where the witness is a temp file holding the cgroup line
    systemd writes for a unit's processes. That reaches the staging, commit, index and
    git-error arms deterministically. What is owed — and only the 2026-10-19 fire can close
    it — is that the kernel writes that line for the real fire.
  * **the gate itself** runs as a real subprocess of this test process, so its cgroup is
    the real one and no witness file is involved. Its env carries a real `INVOCATION_ID`,
    the way an agent Bash turn's does — measured: two Bash forks this session each printed
    `INVOCATION_ID=862d3b6355d542dea3b564f72aae7174`, one value inherited down the whole
    process tree from the backend's own unit, the point being only that the variable IS set
    for a turn. That is the positive arm of the actual threat: a caller that clears the
    variable and still gets nothing.

Every node writes to a scratch `git init` repo under `tmp_path`, never a checkout: the
seeds are a dirty index and a dirty worktree, and seeding those in
`/home/alansrobotlab/lloyd` would be the repository write this script is guarded against.
`--repo` is the seam that makes that possible; the unit passes no argument, and
`tests/test_persistence_window_unit.py::test_the_unit_commits_the_corpus_it_writes_after_rendering_the_report`
pins both the token count and that the line's script path is inside `LIVE_REPO`.

The `git` calls a node makes are `subprocess.run(["git", …])`, never a shell string, and
assertions read the repo back through `git show` / `git status --porcelain` rather than
through this script's own output — the claim under test is what landed on disk.

Run: .venvs/lloyd/bin/python -m pytest tests/test_commit_injection_canary_corpus.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts" / "maintenance" / "commit_injection_canary_corpus.py"


def _load_step():
    spec = importlib.util.spec_from_file_location("commit_injection_canary_corpus",
                                                  SCRIPT)
    assert spec is not None and spec.loader is not None, f"{SCRIPT} will not import"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STEP = _load_step()

#: The corpus, as the script's own relative path and as the pathspec it must pass git.
CORPUS = "eval/measurements/injection-canary"
PATHSPEC = CORPUS + "/"

#: A dirty path OUTSIDE the corpus, which is the thing the step must never commit.
FOREIGN = "architecture/foreign-notes.md"

#: The two cgroup strings measured on this box, kept verbatim because they are the pair
#: the discriminator has to separate: the main process of a live user unit, whose leaf is
#: the unit name (`systemctl --user show --value -p MainPID agent-supervisord.service` →
#: 1695, `cat /proc/1695/cgroup`), and an agent Bash fork of this session, whose leaf is a
#: generated scope that names no service at all.
REAL_UNIT_CGROUP = ("0::/user.slice/user-1000.slice/user@1000.service"
                    "/lloyd.slice/agent-supervisord.service")
REAL_BASH_FORK_CGROUP = ("0::/user.slice/user-1000.slice/user@1000.service"
                         "/lloyd.slice/lloyd-bash-ceiling-15230-b609326e.scope")

IV = "8f14e45fceea167a5a36dedd4bea2543"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    """A scratch checkout with one commit, a configured identity, and no dirt.

    The identity is set locally (`git config`, not `GIT_AUTHOR_*`) so a node that reads
    `%an <%ae>` back is asking the same question the unit's fire will ask: does the script
    pass an author, or leave it to the repo?
    """
    r = tmp_path / "checkout"
    (r / CORPUS).mkdir(parents=True)
    (r / "architecture").mkdir(parents=True)
    _git(r, "-c", "init.defaultBranch=main", "init", "-q")
    _git(r, "config", "user.name", "Canary Window")
    _git(r, "config", "user.email", "canary@example.invalid")
    (r / CORPUS / "rows.jsonl").write_text('{"rep": 1}\n', encoding="utf-8")
    (r / FOREIGN).write_text("base\n", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "base").check_returncode()
    return r


def _dirty_window(repo: Path) -> list[str]:
    """Simulate one fire's output: rows appended, one dated report added, a foreign file
    touched by something else. Returns the corpus paths the fire produced."""
    (repo / CORPUS / "rows.jsonl").write_text('{"rep": 1}\n{"rep": 2}\n',
                                              encoding="utf-8")
    (repo / CORPUS / "run-2026-10-19-persistence.md").write_text("# window\n",
                                                                encoding="utf-8")
    (repo / FOREIGN).write_text("base\nsomebody else's work\n", encoding="utf-8")
    return [f"{CORPUS}/rows.jsonl", f"{CORPUS}/run-2026-10-19-persistence.md"]


@pytest.fixture
def as_unit(tmp_path):
    """Keyword arguments that make an in-process call look like the window unit's fire.

    The witness file holds the cgroup line in the shape systemd writes for a unit's
    processes, with THIS step's unit name as the leaf — the shape `REAL_UNIT_CGROUP` shows
    for a different unit on this box. Passing it is what lets a node reach the staging and
    commit arms; the file is never on argv, because an option that took it would be a
    switch the refused caller could pass itself.
    """
    witness = tmp_path / "cgroup.witness"
    witness.write_text("0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
                       f"{STEP.PERSISTENCE_UNIT}\n", encoding="utf-8")
    return {"env": {"INVOCATION_ID": IV}, "cgroup_file": str(witness)}


def _run_as_agent_turn(repo: Path, extra_env: dict[str, str] | None = None
                       ) -> subprocess.CompletedProcess:
    """The step as a real subprocess of this test process, carrying `INVOCATION_ID`.

    Deliberately a bare env plus the variable: the shape an agent Bash turn actually has on
    this box is `INVOCATION_ID` set (inherited from the backend's own unit) in a cgroup that
    names a generated `.scope`, which is why presence alone cannot be the check. The
    subprocess inherits THIS process's cgroup, and a node that calls this first asserts that
    cgroup is not the window unit's, so the arm is measuring a refusal and not an accident.
    """
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(repo.parent), "LANG": "C",
           "INVOCATION_ID": IV}
    env.update(extra_env or {})
    return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo)],
                          capture_output=True, text=True, env=env, timeout=180)


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _staged(repo: Path) -> list[str]:
    return _git(repo, "diff", "--cached", "--name-only").stdout.split()


def _status(repo: Path) -> list[str]:
    return _git(repo, "status", "--porcelain").stdout.splitlines()


def _witness(tmp_path: Path, body: str, name: str = "cgroup") -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_the_only_staging_command_is_the_corpus_directory_alone(tmp_path, repo,
                                                               monkeypatch, as_unit):
    """Clause 2: the step's ONE staging command is `git -C <repo> add --
    eval/measurements/injection-canary/`, and it runs no other mutating git verb.

    Recorded off the argv of the real calls rather than asserted against a regex over the
    source, because what has to be true is what git is *asked*: a directory-wide `git add
    .`, a `git add -A`, a `git commit -a`, or a `git stash` to get the index clean first are
    all shapes a source regex can miss and a repository cannot tolerate from a job that runs
    unattended at 02:10 in the production checkout.

    The default argument is pinned too: with no `--repo`, the command the unit line produces
    is literally `git -C /home/alansrobotlab/lloyd add --
    eval/measurements/injection-canary/`.
    """
    recorded: list[list[str]] = []
    real_run = subprocess.run

    def spy(argv, *a, **kw):
        if argv and argv[0] == "git":
            recorded.append(list(argv))
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(STEP.subprocess, "run", spy)
    _dirty_window(repo)
    assert STEP.commit_corpus(str(repo), **as_unit) == 0

    adds = [a for a in recorded if a[3:5] == ["add", "--"]]
    assert len(adds) == 1, f"expected one staging command, got {recorded!r}"
    assert adds[0] == ["git", "-C", str(repo), "add", "--", PATHSPEC], \
        f"the staging command is not the corpus directory alone: {adds[0]!r}"
    assert STEP.LIVE_REPO == "/home/alansrobotlab/lloyd", \
        f"with no --repo the step would stage in {STEP.LIVE_REPO!r}"
    assert STEP.CORPUS_PATHSPEC == PATHSPEC

    mutating_verbs = {"add", "commit", "checkout", "reset", "restore", "stash", "rm",
                      "mv", "clean", "revert", "merge", "rebase", "switch"}
    ran = {a[3] for a in recorded if a[1] == "-C" and len(a) > 3}
    assert ran == {"add", "commit", "diff", "rev-parse"}, \
        f"the step's git verbs are not add, diff, commit, rev-parse — {sorted(ran)}: " \
        f"{recorded!r}"
    assert ran <= mutating_verbs | {"diff", "rev-parse"}, \
        f"the step ran a git verb beyond add, commit and its two reads: {sorted(ran)!r}"
    commits = [a for a in recorded if a[3:5] == ["commit", "--only"]]
    assert len(commits) == 1, f"expected one `commit --only`: {recorded!r}"
    assert commits[0][:6] == ["git", "-C", str(repo), "commit", "--only", "-m"], \
        f"the commit is not `git commit --only -m …`: {commits[0]!r}"
    assert commits[0][-2:] == ["--", PATHSPEC], \
        f"the commit is not confined to the corpus by pathspec: {commits[0]!r}"
    assert "-a" not in commits[0] and "--all" not in commits[0], \
        "a commit -a would take the whole worktree, foreign dirt included"
    assert "-c" not in commits[0], \
        "a `-c user.*` here would be an author identity the script invented"


def test_the_commit_takes_the_corpus_and_nothing_else(repo, as_unit):
    """Clause 2, on disk: the fire's rows and its dated report land in one commit, the
    foreign dirty file is untouched and unstaged, and the message and author are the ones
    the item specifies.

    The message is `#<item>: record the <UTC date> persistence-arm window
    (auto-committed by lloyd-persistence-window.service)`, quoted exactly — it is the only
    thing a reader of `git log` gets about who wrote this and why. The author is what the
    scratch repo's own `git config` says, which is the repo-default shape `1cdb06bd` has (on
    the live box `git config --get user.name` and `--get user.email` give `alansrobotlab`
    and `alansrobotlab@gmail.com`, the same pair that authored `1cdb06bd`): no
    `GIT_AUTHOR_*`, no `-c`, so the sha-commit audit path reads this the way it reads any
    other commit until #668 settles the question of labelling one.
    """
    produced = _dirty_window(repo)
    before = _head(repo)
    assert STEP.commit_corpus(str(repo), **as_unit) == 0

    assert _head(repo) != before, "no commit was made"
    assert _git(repo, "rev-parse", "HEAD~1").stdout.strip() == before, \
        "the step made more than one commit"
    names = _git(repo, "show", "--pretty=format:", "--name-only", "HEAD").stdout.split()
    assert sorted(names) == sorted(produced), \
        f"the commit touched {names}, not the fire's corpus alone"

    subject = _git(repo, "log", "-1", "--pretty=%s").stdout.strip()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert subject == (f"#{STEP.ITEM}: record the {today} persistence-arm window "
                       f"(auto-committed by {STEP.PERSISTENCE_UNIT})"), \
        f"unexpected commit subject: {subject!r}"

    author = _git(repo, "log", "-1", "--pretty=%an <%ae>").stdout.strip()
    configured = (f"{_git(repo, 'config', 'user.name').stdout.strip()} "
                  f"<{_git(repo, 'config', 'user.email').stdout.strip()}>")
    assert author == configured, \
        f"the script authored as {author!r} instead of leaving the repo default " \
        f"{configured!r}"

    assert _status(repo) == [f" M {FOREIGN}"], \
        f"the foreign path was not left exactly as found: {_status(repo)!r}"
    assert not _staged(repo), f"the step left an index behind: {_staged(repo)!r}"


@pytest.mark.parametrize("tz", ["Pacific/Kiritimati", "Pacific/Pago_Pago"])
def test_the_dated_part_of_the_subject_is_the_utc_date_under_any_local_clock(
        tmp_path, repo, as_unit, monkeypatch, tz):
    """Clause 2's `<UTC date>`, proven rather than echoed: the subject's date must be
    today in UTC even when the process clock says otherwise.

    The two zones are chosen as a pair, not as company. Kiritimati (UTC+14) is a day ahead
    of UTC whenever the UTC hour is 10 or later, and Pago_Pago (UTC−11) is a day behind it
    for every earlier hour, so between them a `datetime.now()` with no `timezone.utc`
    produces the wrong subject at every hour of every day — which is the class the repo's
    own `naive-now-in-artifact-name` rule exists for: a naive local stamp in a
    machine-facing payload is read as UTC by every later reader, and here the report filename
    the step commits is dated by the same convention.

    Run in-process with `TZ` set and `time.tzset()` called, because the alternative —
    handing a subprocess a cgroup witness through the environment — would be a switch the
    refused caller could pass itself, which is the exact weakness the caller check exists to
    close. `TZ` is the one input a test may set from outside: it moves the local clock for
    nobody but this process.
    """
    import time
    monkeypatch.setenv("TZ", tz)
    time.tzset()
    try:
        _dirty_window(repo)
        assert STEP.commit_corpus(str(repo), **as_unit) == 0
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()

    subject = _git(repo, "log", "-1", "--pretty=%s").stdout.strip()
    utc_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert f"record the {utc_today} persistence-arm window" in subject, \
        f"under TZ={tz} the subject is dated wrong: {subject!r}"


def test_a_foreign_staged_path_makes_it_commit_nothing_and_still_exit_zero(repo,
                                                                          as_unit):
    """Clauses 2 and 3: somebody else's path already in the index stops the step dead.

    This is the shape that makes the guard worth having — an automod landing, a human's
    half-typed `git add`, or the residue of a crashed round leaves a foreign path staged,
    and `git commit --only -- <dir>` would otherwise be the moment their work got attributed
    to a window and to this message. The required behaviour is the least damaging one: log
    one line, commit nothing, exit 0, and STAGE NOTHING BACK — an index that a job silently
    cleaned up is a second job's lost work, and #573's 136-line diff came out of the stash
    stack exactly that way.

    Exit 0 is not tidiness. systemd marks a `Type=oneshot` `Failed` when an `ExecStartPost`
    returns non-zero, so a refusal that has nothing to do with the window would read as a
    failed 34-minute window — the same reasoning the unit gives for its own `--force`.
    """
    _dirty_window(repo)
    assert _git(repo, "add", "--", FOREIGN).returncode == 0, "seed failed"
    foreign_staged = _staged(repo)
    before = _head(repo)
    out = _captured_log(lambda: STEP.commit_corpus(str(repo), **as_unit))

    assert out["rc"] == 0, f"a refused commit must not fail the window: {out!r}"
    assert _head(repo) == before, "the step committed despite a foreign staged path"
    after = _staged(repo)
    assert FOREIGN in after, \
        f"the step unstaged somebody else's work: {foreign_staged!r} -> {after!r}"
    assert FOREIGN in out["text"], f"the one line must name what it refused on: {out!r}"
    assert out["lines"] == 1, f"expected exactly one journal line, got {out!r}"
    assert (repo / FOREIGN).read_text(encoding="utf-8") == \
        "base\nsomebody else's work\n", "the foreign file's contents changed"


def _captured_log(fn) -> dict:
    """Run `fn` with `_log` captured, returning the exit code, the line count and the text.

    The line count is measured at the printing function rather than by parsing a subprocess's
    stdout because most nodes run the step in-process to reach the arms behind the caller
    check; the one-line contract is checked end-to-end on a real subprocess in
    `test_a_real_subprocess_carrying_invocation_id_is_still_refused`.
    """
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fn()
    text = buf.getvalue()
    return {"rc": rc, "text": text, "lines": text.count("\n")}


def test_a_fire_that_wrote_nothing_exits_zero_without_an_empty_commit(repo, as_unit):
    """Clause 3: nothing staged means nothing to commit, and that is a 0.

    Reached on a real fire too — the unit can complete with `ExecStart` writing no new rows
    if the driver's reps all refused, and a `git commit` with an empty index exits 1 by
    itself, which under `ExecStartPost` is a `Failed` result for a run that did nothing
    wrong. So the step checks the index it just staged rather than discovering the emptiness
    as git's exit status.
    """
    before = _head(repo)
    assert not _status(repo), "the fixture must start clean for this node"

    out = _captured_log(lambda: STEP.commit_corpus(str(repo), **as_unit))
    assert out["rc"] == 0, f"exit {out['rc']}"
    assert _head(repo) == before, "an empty commit was made"
    assert out["lines"] == 1, f"one line, always: {out['text']!r}"
    assert "nothing to commit" in out["text"], \
        f"a no-op fire has to be readable as one: {out['text']!r}"


def test_a_git_that_fails_exits_zero_and_leaves_the_tree_alone(tmp_path, repo, as_unit):
    """Clause 3, both git-error shapes: no repository at all, and an index that is locked.

    A `--repo` at a directory that is not a checkout is the misconfiguration arm; the
    `index.lock` arm is the one production will actually meet, because 02:10 is inside the
    nightly window in which an automod landing may be holding the lock, and a step that
    could not take the index has to leave the window's result alone and let the next Monday's
    fire pick the rows up. Neither arm may raise a traceback into the journal: the exception
    path is the last-resort exit-0 branch, so the failure must still be one named line.
    """
    not_a_repo = tmp_path / "not-a-checkout"
    not_a_repo.mkdir()
    out = _captured_log(lambda: STEP.commit_corpus(str(not_a_repo), **as_unit))
    assert out["rc"] == 0, f"a step that cannot even ask git must exit 0: {out!r}"
    assert out["lines"] == 1, f"one line, no traceback: {out['text']!r}"
    assert "not a git repository" in out["text"], \
        "the line must carry git's own reason, not just the fact of failure"

    _dirty_window(repo)
    before = _head(repo)
    lock = repo / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    try:
        out = _captured_log(lambda: STEP.commit_corpus(str(repo), **as_unit))
    finally:
        lock.unlink()
    assert out["rc"] == 0, f"a locked index must not fail the window: {out!r}"
    assert _head(repo) == before, "the step committed through a locked index"
    assert out["lines"] == 1, f"one line for a locked index too: {out['text']!r}"
    assert any("rows.jsonl" in line for line in _status(repo)), \
        "the corpus dirt has to survive a refused step for the next fire to commit"


def test_a_real_subprocess_carrying_invocation_id_is_still_refused(repo):
    """Clause 4, the arm that matters: a caller holding a real `INVOCATION_ID` and no unit
    cgroup gets no commit — which is exactly what an agent Bash turn on this box is.

    The refusal is measured, not assumed. Two Bash forks in the session that produced the
    first gate run each reported `INVOCATION_ID=862d3b6355d542dea3b564f72aae7174` — the same
    value in both, because the backend is itself a unit and systemd exports the variable
    down the whole tree — while their `/proc/self/cgroup` read
    `…/lloyd.slice/lloyd-bash-ceiling-15230-b609326e.scope`. The review rung reproduced the
    same thing independently, and it is why this node refuses a subprocess that clears the
    variable rather than one that lacks it.

    The precondition is asserted first, because the node measures a refusal only if THIS
    process really is outside the window unit's cgroup; run from inside the unit, it would
    otherwise pass for the wrong reason.
    """
    assert not STEP.runs_in_window_unit(), \
        "this test process is inside lloyd-persistence-window.service's cgroup, so the " \
        "refusal below would be measuring nothing"
    produced = _dirty_window(repo)
    before = _head(repo)

    proc = _run_as_agent_turn(repo)
    assert proc.returncode == 0, \
        f"a refused commit must not fail anything: exit {proc.returncode}, " \
        f"stderr {proc.stderr!r}"
    assert _head(repo) == before, \
        "the step committed for a caller that is not the window unit"
    assert not _staged(repo), \
        f"the step staged behind the refusal: {_staged(repo)!r}"
    assert sorted(_status(repo)) == sorted(
        [f" M {produced[0]}", f"?? {produced[1]}", f" M {FOREIGN}"]), \
        f"the fire's residue must be left exactly as found: {_status(repo)!r}"
    assert proc.stdout.count("\n") == 1 and proc.stderr == "", \
        f"one journal line and no traceback: {proc.stdout!r} {proc.stderr!r}"
    assert "cgroup" in proc.stdout, \
        f"the line has to name the arm that refused: {proc.stdout!r}"
    assert str(repo) in proc.stdout, \
        f"…and the checkout it declined to write, which is the one thing a reader needs " \
        f"to tell this step from the one that commits: {proc.stdout!r}"
    assert STEP.PREFIX in proc.stdout, \
        f"the line has to be attributable to this step in the journal: {proc.stdout!r}"


def test_a_caller_with_no_invocation_id_is_refused_before_any_git_call(tmp_path, repo):
    """Clause 4's letter: with no `INVOCATION_ID` the step commits nothing, and it decides
    that before touching the index.

    The clause names the variable, so the variable is honoured as an arm — but the reason it
    is not the whole check is measured above. What this node adds over the subprocess node is
    ORDER: the refusal precedes `git add`, so a turn typing the wrapper does not so much as
    move the index. That is pinned by recording argv, because `git status` after the fact
    cannot distinguish "refused first" from "staged then decided not to commit" — and the
    second shape would still be a repository write by an unauthorised caller.
    """
    recorded: list[list[str]] = []
    real_run = subprocess.run

    def spy(argv, *a, **kw):
        if argv and argv[0] == "git":
            recorded.append(list(argv))
        return real_run(argv, *a, **kw)

    import unittest.mock as mock
    _dirty_window(repo)
    before = _head(repo)
    with mock.patch.object(STEP.subprocess, "run", spy):
        out = _captured_log(lambda: STEP.commit_corpus(
            str(repo), env={}, cgroup_file=str(tmp_path / "absent")))
    assert out["rc"] == 0
    assert recorded == [], f"git was called before the caller check: {recorded!r}"
    assert _head(repo) == before and not _staged(repo)
    assert "INVOCATION_ID" in out["text"], \
        f"the line must say which arm refused: {out['text']!r}"


def test_the_cgroup_reader_separates_a_unit_from_a_generated_scope(tmp_path):
    """The discriminator's own input, measured against the two strings this box produces.

    `REAL_UNIT_CGROUP` is `cat /proc/1695/cgroup` on this machine, PID 1695 being
    `agent-supervisord.service`'s `MainPID`; `REAL_BASH_FORK_CGROUP` is
    `cat /proc/self/cgroup` inside an agent Bash fork. The first has a component naming a
    unit and the second names only a generated `.scope`, and the step must accept the shape
    of the first for its own unit and reject the second outright.

    The `.d` arm is the trap in a substring match: `lloyd-persistence-window.service.d` is a
    drop-in directory that exists beside a unit on any box systemd has ever touched, and
    testing `unit_name in line` against either accepted name would read it as an invocation.
    A component equality against a key of `WINDOW_UNITS` does not."""
    unit_witness = _witness(tmp_path, f"0::/user.slice/user-1000.slice/"
                                      f"user@1000.service/app.slice/{STEP.PERSISTENCE_UNIT}",
                            "unit")
    assert STEP.runs_in_window_unit(unit_witness) is True, \
        "the shape systemd writes for a unit's processes has to be accepted, or the " \
        "2026-10-19 fire commits nothing and #2866's owed commit silently survives"
    assert STEP.cgroup_components(unit_witness)[-1] == STEP.PERSISTENCE_UNIT

    other_unit = _witness(tmp_path, REAL_UNIT_CGROUP, "other-unit")
    assert STEP.runs_in_window_unit(other_unit) is False, \
        "a different unit's cgroup must not read as this unit's invocation"
    assert STEP.cgroup_components(other_unit)[-1] == "agent-supervisord.service"

    bash_fork = _witness(tmp_path, REAL_BASH_FORK_CGROUP, "bash-fork")
    assert STEP.runs_in_window_unit(bash_fork) is False, \
        "the cgroup an agent Bash fork actually runs in was accepted"
    assert STEP.cgroup_components(bash_fork)[-1] == \
        "lloyd-bash-ceiling-15230-b609326e.scope"

    dropin = _witness(tmp_path, f"0::/user.slice/x/{STEP.PERSISTENCE_UNIT}.d", "dropin")
    assert STEP.runs_in_window_unit(dropin) is False, \
        "a `.d` drop-in directory component was read as an invocation of the unit"

    assert STEP.cgroup_components(str(tmp_path / "no-such-file")) == [], \
        "an unreadable cgroup must fail closed, not raise"
    assert STEP.runs_in_window_unit(str(tmp_path / "no-such-file")) is False
    assert STEP.is_systemd_invocation({}) is False, \
        "INVOCATION_ID absent must still be a refusal on its own arm"
    assert STEP.is_systemd_invocation({"INVOCATION_ID": IV}) is True


def test_the_step_is_not_a_package_and_the_window_pair_are_its_only_invocation():
    """The seam the two files have to agree on: this script is invoked by absolute path
    from the unit, and nothing in the repo imports it or wires it into anything else.

    `scripts/` is not a package and the step has no side effects at import, so the only way
    it runs in production is an `ExecStartPost` line — which
    `tests/test_persistence_window_unit.py` and `tests/test_egress_window_unit.py` each pin
    to exactly two tokens. What this node adds is the other direction: a second caller — a
    timer, an autonomy task, an `__init__`-time hook — would give an unattended job a
    committer designed for a window's own invocation.

    The set is TWO window units as of #2976, and it is now checked against the guard's own
    authorized set rather than listed freehand: a caller named here but absent from
    `WINDOW_UNITS` is the silent-green-no-op bug (the unit fires a step that refuses and
    exits 0), and one present in `WINDOW_UNITS` but absent here is a unit that gained the
    right to commit the live checkout without this rail knowing. Either drift fails, so the
    units in the allowlist and the keys of the guard cannot come apart.
    """
    assert SCRIPT.is_file() and not (SCRIPT.parent / "__init__.py").exists(), \
        "`scripts/` is a package, so importing this module could run it — the whole design " \
        "assumes the unit's argv is the only invocation"
    importers = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-l", "commit_injection_canary_corpus"],
        capture_output=True, text=True).stdout.split()
    units = {"agent-services/systemd/" + unit for unit in STEP.WINDOW_UNITS}
    allowed = units | {
        "scripts/maintenance/commit_injection_canary_corpus.py",
        "tests/test_commit_injection_canary_corpus.py",
        "tests/test_persistence_window_unit.py",
        "tests/test_egress_window_unit.py",
        "tests/test_safety_gate_discrimination.py",
    }
    assert set(importers) <= allowed, \
        f"something besides the window units and their tests names the committer: " \
        f"{sorted(set(importers - allowed))}"
    naming = set(importers) & units
    assert naming == units, \
        f"the units named here {sorted(naming)} are not the units the guard authorizes " \
        f"{sorted(units)}: one of them is a step that would be refused in silence, or an " \
        f"authorized caller this rail stopped watching"


# ── #2976: a second window unit drives this step ───────────────────────────────────────
#
# `lloyd-egress-window.service` appends into this same tracked corpus and carried no commit
# step of its own, so its rows reached HEAD only at the NEXT Monday's persistence fire — this
# timer goes off at 06:23 plus jitter, after 02:10 has already passed. The unit line is half
# the change; what a unit line cannot do is make the guard accept the fire that runs it.
# `commit_corpus` returns 0 on a refusal — deliberately, so a refused commit cannot mark a
# ~34-minute window `Failed` — which is precisely what makes a step refused by NAME
# indistinguishable, from the unit's side, from a step that committed. So the arms below
# drive the whole function from an egress witness rather than reading the guard, and then
# re-check the refusals against the WIDENED guard rather than the old one.

#: The cgroup systemd writes for a process it started for the EGRESS window, in the shape
#: `REAL_UNIT_CGROUP` measures for a live user unit on this box: the unit name as the leaf.
EGRESS_CGROUP = ("0::/user.slice/user-1000.slice/user@1000.service"
                 f"/lloyd.slice/{STEP.EGRESS_UNIT}")


def _as_fire(tmp_path: Path, cgroup_body: str, name: str = "fire") -> dict:
    """`commit_corpus` kwargs making an in-process call a fire of the unit whose cgroup line
    is `cgroup_body` — the kwargs `as_unit` builds, for whichever unit is asked."""
    return {"env": {"INVOCATION_ID": IV},
            "cgroup_file": _witness(tmp_path, cgroup_body, name)}


def test_an_egress_fire_stages_the_corpus_and_commits_it(tmp_path, repo):
    """#2976 clause 2, on disk: a fire whose cgroup leaf is
    `lloyd-egress-window.service` stages the corpus directory, commits the fire's rows and
    report, and leaves the index clean.

    Reading `authorized_unit` cannot give this answer. The refusal arm exits 0, so a widened
    name that never reached the guard, or a guard that still knew only the persistence unit,
    would produce a green egress fire with no commit behind it AND a passing read of the
    constant. Driving `commit_corpus` from a witness whose leaf is the egress unit is what
    shows the rows that unit appends actually reach HEAD in the same fire that wrote them.
    """
    produced = _dirty_window(repo)
    before = _head(repo)
    assert STEP.commit_corpus(str(repo), **_as_fire(tmp_path, EGRESS_CGROUP)) == 0, \
        "an egress fire was refused, so its rows wait a week for the sibling's next fire"

    assert _head(repo) != before, \
        "the egress fire made no commit: its ExecStartPost would still exit 0 and read green"
    assert _git(repo, "rev-parse", "HEAD~1").stdout.strip() == before, \
        "the step made more than one commit"
    names = _git(repo, "show", "--pretty=format:", "--name-only", "HEAD").stdout.split()
    assert sorted(names) == sorted(produced), \
        f"the egress commit touched {names}, not the corpus that fire wrote"
    assert not _staged(repo), f"the step left an index behind: {_staged(repo)!r}"


@pytest.mark.parametrize("leaf,kind,other", [
    (STEP.PERSISTENCE_UNIT, "persistence-arm", "egress"),
    (STEP.EGRESS_UNIT, "egress-arm", "persistence"),
])
def test_the_commit_subject_names_the_unit_and_window_that_actually_fired(
        tmp_path, repo, leaf, kind, other):
    """#2976 clause 4: both halves of the subject — the window kind and the unit in
    `auto-committed by` — come from the cgroup component that authorized the commit, so a
    fire of one window is never filed under the other.

    Both were hardcoded before: `f"#{ITEM}: record the {today} persistence-arm window
    (auto-committed by {UNIT_NAME})"`, with `UNIT_NAME` the same constant the guard read.
    Once the guard could accept an egress fire, that literal would have recorded a fire's
    egress rows as a `persistence-arm window (auto-committed by
    lloyd-persistence-window.service)` — rows attributed to a window that did not write them,
    in the one place a later reader looks to ask which fire produced a commit. Pinned both
    ways round, and against the firing unit rather than any constant, which is what makes a
    hardcoded subject impossible here.
    """
    _dirty_window(repo)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = ("0::/user.slice/user-1000.slice/user@1000.service/app.slice/" + leaf)
    assert STEP.commit_corpus(str(repo), **_as_fire(tmp_path, body, name=leaf)) == 0

    subject = _git(repo, "log", "-1", "--pretty=%s").stdout.strip()
    assert subject == (f"#{STEP.ITEM}: record the {today} {kind} window "
                       f"(auto-committed by {leaf})"), \
        f"a fire of {leaf} produced the subject {subject!r}"
    assert other not in subject.lower(), \
        f"the subject of a {leaf} fire mentions {other!r}: {subject!r}"


@pytest.mark.parametrize("label,cgroup_body,env", [
    ("agent-bash-scope-and-no-invocation-id", REAL_BASH_FORK_CGROUP, {}),
    ("agent-bash-scope-with-inherited-invocation-id", REAL_BASH_FORK_CGROUP,
     {"INVOCATION_ID": IV}),
    ("parent-user-scope", "0::/user.slice/user-1000.slice/user@1000.service",
     {"INVOCATION_ID": IV}),
    ("another-lloyd-unit-leaf", REAL_UNIT_CGROUP, {"INVOCATION_ID": IV}),
    ("persistence-drop-in-dir", f"0::/user.slice/x/{STEP.PERSISTENCE_UNIT}.d/exec.conf",
     {"INVOCATION_ID": IV}),
    ("egress-drop-in-dir", f"0::/user.slice/x/{STEP.EGRESS_UNIT}.d/exec.conf",
     {"INVOCATION_ID": IV}),
])
def test_widening_the_accepted_units_did_not_loosen_the_guard(tmp_path, repo, label,
                                                             cgroup_body, env):
    """#2976 clause 3: after accepting a second unit, a non-window caller still commits
    nothing, leaves the tracked tree untouched and exits 0 — including a `.scope` leaf (an
    agent Bash fork's real shape) and a `.d` drop-in component of EITHER window unit.

    Re-checked against the widened guard rather than inherited from the nodes above, because
    taking a second name is exactly how a whole-component match decays into a substring
    match: `lloyd-egress-window.service.d/exec.conf` is a directory that exists beside the
    accepted leaf on any box systemd has touched, and shares a prefix with it, so a guard
    that stopped splitting on `/` would wave a drop-in through. The `.d` sweep now runs over
    both unit names for that reason.

    Exit 0 is the contract, not a courtesy — see the module docstring: a refused commit is a
    success-by-refusal, and a non-zero exit marks the window `Failed`. What must not move is
    the index: the corpus keeps its rows either way, and the outcome an agent turn must never
    reach is removing them from `git status` in the production checkout.
    """
    _dirty_window(repo)
    before = _head(repo)
    w = _witness(tmp_path, cgroup_body, "refuse")
    assert STEP.runs_in_window_unit(w) is False, \
        f"{label}: a cgroup naming no window unit authorized the step"
    assert STEP.commit_corpus(str(repo), env, cgroup_file=w) == 0, \
        f"{label}: refusing a non-window caller must exit 0, not fail the window"
    assert _head(repo) == before, f"{label} committed into the checkout"
    assert not _staged(repo), f"{label} staged: {_staged(repo)!r}"


def test_a_turn_without_an_invocation_id_still_exits_zero_at_the_boundary(repo):
    """Clause 3's exit code across the real process boundary rather than the function's: a
    subprocess whose environment carries no `INVOCATION_ID` — the environment an agent Bash
    fork has when the backend itself is not the caller — exits 0, stages nothing and commits
    nothing, measured against the widened guard.

    `_run_as_agent_turn` deliberately carries `INVOCATION_ID`, because the shape it exists to
    test is the cgroup arm; this is the other arm, and it has to be a real process: an
    in-process `{}` passes even if `is_systemd_invocation` were inverted, since neither an
    absent key nor an empty dict carries the string.

    No cgroup witness is handed over, because there is no flag that would take one and that is
    deliberate — an option naming a cgroup file would be a switch a refused caller could pass
    itself. The child inherits THIS process's cgroup, so the first assertion asks the real
    `/proc/self/cgroup` to name no window unit: that precondition is what makes the inherited
    cgroup a refusal rather than an accident, and it is the same precondition
    `test_a_real_subprocess_carrying_invocation_id_is_still_refused` states for its arm.
    """
    _dirty_window(repo)
    before = _head(repo)
    assert STEP.runs_in_window_unit() is False, \
        "this test process is itself running in a window unit's cgroup, so a child would " \
        "inherit authorization and the arm would measure nothing"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(repo.parent), "LANG": "C"}
    assert "INVOCATION_ID" not in env
    proc = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo)],
                          capture_output=True, text=True, env=env, timeout=180)
    assert proc.returncode == 0, \
        f"a refused commit from a turn exited {proc.returncode}: {proc.stderr[-300:]}"
    assert proc.stderr == "" and proc.stdout.count("\n") == 1, \
        f"one journal line and no traceback: {proc.stdout!r} {proc.stderr!r}"
    assert STEP.PREFIX in proc.stdout and "invocation" in proc.stdout.lower(), \
        "the refusal has to be attributable to this step and name the arm that refused: " \
        f"{proc.stdout!r}"
    assert _head(repo) == before and not _staged(repo), \
        "a turn with no INVOCATION_ID changed the checkout"
