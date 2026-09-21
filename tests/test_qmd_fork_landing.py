"""The qmd fork's landed state must be answerable by running one command.

`~/lloyd/qmd` — the fork that supplies production's vector leg — is ignored out
of the gated tree (`.gitignore` `/qmd/`, `git ls-files qmd` empty, no
`.gitmodules`), so no gate rung builds it, no promotion record carries its sha,
and the guardian's `reset --hard` / `clean -fd` cannot revert it. #854 settled
the route: a round does not edit the fork; fork changes are human-landed, and
`scripts/qmd_fork_landing.py` is what makes "fork sha X is landed" checkable —
the fork's own build, the fork's own suite, then branch/HEAD sha, clean tree,
and not ahead of its upstream.

The seam these tests cross is a *process boundary*: the script spawns
`node scripts/build.mjs` and `node scripts/test-all.mjs` and shells out to
`git`, so importing it proves nothing about the verdict it reports. Every test
here runs the documented command line — `python -m scripts.qmd_fork_landing
--fork <path>` — against a fixture fork it builds itself, and asserts on the
exit status. The fixtures are real git repositories with upstreams, so the
git-state half is exercised across a real `git push`, not a mock.

Nothing here needs the live fork to exist: it is absent from every automod
worktree, and a fixture whose build script fails is the case the check exists
to catch — the one that must not be able to report success.

Clause map, so a reader grading #854 can find each clause's node without
searching. Clauses 1 and 2 are the two documents and live in
``tests/test_automod_doc_claims.py``.

    clause 3 — build and suite both run, non-zero if either fails:
        ``test_a_fork_that_builds_passes_its_suite_and_is_pushed_reports_landed``
        ``test_a_fork_whose_build_fails_exits_nonzero_and_names_the_build_step``
        ``test_a_fork_whose_suite_fails_exits_nonzero_and_names_the_suite_step``
        ``test_a_fork_missing_either_step_script_fails_rather_than_passing_quietly``
    clause 4 — branch and HEAD sha printed, dirty or unpushed refused:
        ``test_a_fork_that_builds_passes_its_suite_and_is_pushed_reports_landed``
        ``test_a_fork_with_uncommitted_changes_is_not_landed``
        ``test_a_fork_with_commits_origin_never_received_is_not_landed``
        ``test_a_fork_branch_with_no_upstream_cannot_claim_a_landed_sha``
    clause 5 — a fixture fork whose build fails exits non-zero and the failure
    output names the failed step:
        ``test_a_fork_whose_build_fails_exits_nonzero_and_names_the_build_step``
        asserts ``FAIL build:`` (the step name), the quoted fixture error line,
        and that the suite leg is not falsely blamed (no ``FAIL suite:``).
        ``test_a_fork_whose_suite_fails_exits_nonzero_and_names_the_suite_step``
        is the same assertion for the other step, and
        ``test_editing_the_forks_package_json_cannot_change_what_the_check_runs``
        is the inverted case: the fork's own package.json points `build` and
        `test` at two decoys that *pass* while the documented build fails, so a
        check that followed package.json would have reported success from the
        decoy. Two further legs name a step failure when the step cannot run at
        all:
        ``test_a_fork_missing_either_step_script_fails_rather_than_passing_quietly``
        ``test_a_fork_with_no_node_fails_by_name_rather_than_traceback``

    Every leg runs under a fixture environment, not this machine's: `_run_check`
    sets `PATH` to a node/git/bun trio this file writes and `HOME` to a directory
    with no bun install, so a green leg cannot mean "the developer's shell had
    bun". Which bun the suite resolved is proved by the bun's own self-reported
    version landing in the fork's working tree, not by a return code:
        ``test_the_suite_step_reaches_bun_through_the_home_fallback``
        ``test_a_bun_on_the_path_wins_over_the_home_fallback``
        ``test_a_fork_with_no_bun_anywhere_fails_the_suite_by_name``
    `DEFAULT_FORK`, the one value the shipped command line depends on and no
    `--fork` argument can override, is pinned alongside the two step scripts in
    ``test_the_check_runs_the_documented_pair_at_the_default_fork_path``. The
    fork's own package.json is a *cross-tree* claim, so it is its own test and
    reports `skipped` — never passed — where the ignored fork is absent:
    ``test_the_live_fork_still_declares_the_pair_the_outer_tree_names``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORK = Path.home() / "lloyd" / "qmd"

# A fixture must not read the machine's git config (a global user.name, a
# non-default init.defaultBranch, a credential helper): every git call runs
# against the null config files.
GIT_ENV = dict(
    os.environ,
    GIT_AUTHOR_NAME="fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
    GIT_COMMITTER_NAME="fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid",
    GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, env=GIT_ENV)
    assert done.returncode == 0, f"git {' '.join(args)} in {root}: {done.stderr}"
    return done.stdout.strip()


def _step(root: Path, rel: str, *, ok: bool) -> None:
    """A stand-in for one of the fork's scripts: prints an identifying line,
    and exits 3 with a named error when it is meant to fail."""
    body = f'console.log("{rel}: fixture step ran");\n'
    if ok:
        body += "process.exit(0);\n"
    else:
        body += f'console.error("fixture {rel} fails on purpose");\nprocess.exit(3);\n'
    (root / "scripts" / rel).write_text(body, encoding="utf-8")


def _init_fork(root: Path, remote: Path) -> Path:
    """Turn an existing directory into a fork checkout on branch `lloyd`,
    pushed to `remote` and in sync with it, whose two steps pass — the shape of
    a landed fork. Split out of `_make_fork` because one test has to build the
    fixture at a *specific* path (the check's own default target), not under a
    name of its own."""
    (root / "scripts").mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    (root / "package.json").write_text('{"name": "fixture-qmd", "version": "0.0.0"}\n',
                                       encoding="utf-8")
    _step(root, "build.mjs", ok=True)
    _step(root, "test-all.mjs", ok=True)
    (root / "src" / "vecindex.ts").write_text("export const normalise = 1;\n",
                                             encoding="utf-8")
    _git(root, "init", "-q", "-b", "lloyd")
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "fixture source")
    _git(root, "push", "-q", "-u", "origin", "lloyd")
    return root


def _make_fork(tmp_path: Path, name: str, remote: Path) -> Path:
    """A landed-shape fixture fork at `tmp_path/name`."""
    return _init_fork(tmp_path / name, remote)


def _real_node() -> str:
    """The node binary itself, not whatever shim `PATH` happens to resolve.

    On this box `which node` is a mise shim, and a shim symlinked out of its own
    directory is a different program from the interpreter the fork's steps need.
    `process.execPath` is the real one, so a fixture PATH built from it tests the
    check rather than the launcher.
    """
    done = subprocess.run([shutil.which("node"), "-p", "process.execPath"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _bin_dir(tmp_path: Path, name: str, *, with_bun: bool) -> Path:
    """A PATH holding exactly the interpreters this check is meant to find.

    Every environment-leg test needs a *known* PATH. `/usr/bin:/bin` happens to
    hold node and git here and `~/.bun/bin/bun` happens to exist, so a test built
    on either passed on this box and said nothing anywhere else — including in a
    worktree on a machine where the fork was never cloned. node and git are
    symlinked from the binaries this process resolves; bun is a fixture script
    that prints a version no real bun prints, present only in the leg about the
    bun the check *should* find.
    """
    bin_dir = tmp_path / name
    bin_dir.mkdir()
    (bin_dir / "node").symlink_to(_real_node())
    (bin_dir / "git").symlink_to(shutil.which("git"))
    if with_bun:
        _write_bun(bin_dir / "bun")
    return bin_dir


def _write_bun(path: Path) -> None:
    """A stand-in bun that identifies itself on `--version`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho 9.9.9-fixture\n", encoding="utf-8")
    path.chmod(0o755)


def _fake_home(tmp_path: Path, *, with_bun: bool) -> Path:
    """A home directory, optionally carrying the per-user bun install the check
    falls back to at `~/.bun/bin/bun` when `PATH` has none.

    `HOME` is what makes the check's two resolution paths testable at all: it is
    `Path.home()` that the fallback reads, and it is unset in exactly the
    environments the fallback exists for.
    """
    home = tmp_path / "fake-home"
    if with_bun:
        _write_bun(home / ".bun" / "bin" / "bun")
    else:
        # exist_ok: `_run_check` hands every leg this default too, so a leg that
        # first asks for a home *with* a bun and then runs the check must not
        # collide with its own directory.
        home.mkdir(exist_ok=True)
    return home


def _commit_and_push(root: Path, message: str) -> str:
    """Land a fixture edit the way the rule says to: commit on `lloyd`, push to
    `origin`. Returns the new HEAD sha."""
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message)
    _git(root, "push", "-q", "origin", "lloyd")
    return _git(root, "rev-parse", "HEAD")


def _break_step(root: Path, rel: str) -> str:
    """Make one of the fork's own steps fail, committed and pushed. Isolates a
    step failure from the git-state checks, so a build failure can never be
    reported as a dirty tree instead."""
    _step(root, rel, ok=False)
    return _commit_and_push(root, f"{rel} fails on purpose")


def _run_check(tmp_path: Path, *fork_args: str,
               env: dict | None = None) -> subprocess.CompletedProcess:
    """The documented invocation, from the repo root, as a subprocess — the only
    form that can report a landed fork wrongly.

    The default environment is a fixture, not this machine's: `PATH` holds the
    node/git/bun trio this file created and `HOME` has no bun install under it.
    Inheriting the ambient environment instead lets a green leg mean "the
    developer's shell happened to have bun on `$PATH`", which is the dependency
    the check exists to remove — and the dependency that made the pass leg here
    host-dependent. A leg that wants to *prove* an absence (no bun anywhere, no
    node at all) passes `env`, which overrides the fixture's `PATH` or `HOME`.
    """
    run_env = dict(os.environ)
    run_env.update({
        "PATH": str(_bin_dir(tmp_path, "bins", with_bun=True)),
        "HOME": str(_fake_home(tmp_path, with_bun=False)),
    })
    if env is not None:
        run_env.update(env)
    return subprocess.run([sys.executable, "-m", "scripts.qmd_fork_landing", *fork_args],
                          cwd=ROOT, capture_output=True, text=True, timeout=300,
                          env=run_env)


# ── the environment the steps run under is a seam ────────────────────────────
#
# `qmd/scripts/test-all.mjs:37` runs the Bun suite as
# `run("Bun test suite", "bun", ...)` — a bare-name `spawnSync`. Two ordinary
# callers cannot satisfy that name, and neither is a fork defect: the gate hands
# its rungs `PATH=/usr/bin:/bin` on purpose (`scripts/automod/gate.py:508`), and
# bun is a per-user install whose `~/.bun/bin` reaches `$PATH` only through a
# login shell. A check that resolved neither would print a green build and a red
# suite — which reads as "the fork is broken", a different finding filed against
# the wrong tree.

_NO_LOGIN_ENV = {"PATH": "/usr/bin:/bin"}


def _bare_name_suite(root: Path, name: str) -> str:
    """Replace the fixture's suite step with one that spawns `bun` by bare name,
    the way the fork's does, committed and pushed.

    It also records the answer in `bun_used.txt`. A successful step only proves
    *a* bun answered; the file is what lets a test assert it was the fixture bun
    the check resolved, rather than any bun the machine had lying around.
    """
    (root / "scripts" / "test-all.mjs").write_text(
        'import { spawnSync } from "node:child_process";\n'
        'import { writeFileSync } from "node:fs";\n'
        'const r = spawnSync("bun", ["--version"], {encoding: "utf8"});\n'
        'if (r.error || r.status !== 0) {\n'
        '  console.error("bare-name bun spawn failed: " + (r.error && r.error.code));\n'
        '  process.exit(1);\n'
        '}\n'
        'writeFileSync("bun_used.txt", r.stdout.trim() + "\\n");\n'
        'console.log("bun answered " + r.stdout.trim());\n',
        encoding="utf-8")
    return _commit_and_push(root, f"{name}: suite spawns bun by bare name")


def test_the_suite_step_reaches_bun_through_the_home_fallback(tmp_path, remote):
    """"Fork sha X is landed" must not depend on who asks. A gate-shaped or
    cron-shaped environment has no bun on `PATH` — the gate hands its rungs
    `PATH=/usr/bin:/bin` on purpose (`scripts/automod/gate.py:508`) and bun
    reaches `$PATH` only through a login shell — so the step still has to satisfy
    the fork's bare-name spawn out of `~/.bun/bin`, or the check is only runnable
    from a terminal.

    Both the absent bun on `PATH` and the fallback bun under `HOME` are fixtures
    here, and the answer is read back out of the fork: only the fixture bun prints
    `9.9.9-fixture`, so the file the step writes proves the child spawned the one
    the check named rather than some bun the machine had lying around."""
    fork = _make_fork(tmp_path, "barename", remote)
    _bare_name_suite(fork, "barename")
    home = _fake_home(tmp_path, with_bun=True)
    done = _run_check(tmp_path, "--fork", str(fork), env={
        "PATH": str(_bin_dir(tmp_path, "bin-no-bun", with_bun=False)),
        "HOME": str(home)})
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-500:]
    out = done.stdout
    assert "step suite: ok" in out, out[-1500:]
    # The bun that got used is named before any step output, so a reader sees
    # which one the suite resolved instead of inferring it from a green step.
    used = out.split("tool bun: ")[1].splitlines()[0].strip()
    assert used == str(home / ".bun" / "bin" / "bun"), out
    assert (fork / "bun_used.txt").read_text(encoding="utf-8").strip() == "9.9.9-fixture", \
        "the suite spawned a different bun than the check named"


def test_a_bun_on_the_path_wins_over_the_home_fallback(tmp_path, remote):
    """The precedence the resolver documents — a bun the calling environment can
    already see beats the per-user install — is the difference between checking a
    fork with the bun its developer intended and checking it with whatever the
    account happens to have installed. Asserted by both sides: the name printed
    and the version the child actually reported."""
    fork = _make_fork(tmp_path, "pathbun", remote)
    _bare_name_suite(fork, "pathbun")
    path_bun = _bin_dir(tmp_path, "bin-with-bun", with_bun=True) / "bun"
    home = _fake_home(tmp_path, with_bun=False)
    done = _run_check(tmp_path, "--fork", str(fork),
                      env={"PATH": str(path_bun.parent), "HOME": str(home)})
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-500:]
    out = done.stdout
    assert out.split("tool bun: ")[1].splitlines()[0].strip() == str(path_bun), out
    assert (fork / "bun_used.txt").read_text(encoding="utf-8").strip() == "9.9.9-fixture", out


def test_a_fork_with_no_bun_anywhere_fails_the_suite_by_name(tmp_path, remote):
    """With no bun on `PATH` and none under `HOME` either, the verdict is a named
    refusal naming the bare-name spawn — not an ENOENT delivered after the node
    halves printed green, which is what the fork's own runner would report. Both
    absences are fixtures rather than the machine's state: on this box
    `~/.bun/bin/bun` exists, so a test that only stripped `PATH` was asserting
    the fallback was broken."""
    fork = _make_fork(tmp_path, "nobun", remote)
    _bare_name_suite(fork, "nobun")
    done = _run_check(tmp_path, "--fork", str(fork), env={
        "PATH": str(_bin_dir(tmp_path, "bin-no-bun2", with_bun=False)),
        "HOME": str(_fake_home(tmp_path, with_bun=False))})
    assert done.returncode != 0
    out = done.stdout
    assert "FAIL suite:" in out, out[-1500:]
    assert "bun" in out.split("FAIL suite:")[1][:400], out[-1500:]
    assert "step suite: ok" not in out, "the suite ran without a bun to spawn"
    assert "tool bun: NOT FOUND" in out


def test_a_fork_with_no_node_fails_by_name_rather_than_traceback(tmp_path, remote):
    """The check must print a named `node` failure and exit non-zero; a traceback
    escaping `main` is not a verdict a person can act on. node is the one
    interpreter deliberately absent from the fixture PATH (bun present, git
    present — git is what the check's own git-state leg runs, so the run has to
    reach the node check) rather than left to a `/usr/bin:/bin` that may or may
    not hold a node."""
    fork = _make_fork(tmp_path, "nonode", remote)
    binonly = tmp_path / "bin-no-node"
    binonly.mkdir()
    (binonly / "git").symlink_to(shutil.which("git"))
    _write_bun(binonly / "bun")
    assert not (binonly / "node").exists()
    done = _run_check(tmp_path, "--fork", str(fork), env={"PATH": str(binonly)})
    assert done.returncode != 0
    assert "FAIL node:" in done.stdout, done.stdout[-600:] + done.stderr[-800:]
    assert "LANDED:" not in done.stdout
    assert "Traceback" not in done.stderr


def test_a_dirty_fork_is_refused_before_its_build_touches_the_served_tree(tmp_path, remote):
    """Order is load-bearing, not cosmetic: `node scripts/build.mjs` writes
    `dist/`, and `dist/` is the directory `agent-qmd-daemon` serves on :8181. A
    check that built first would compile an uncommitted working tree into the
    artifact production is running, then report the dirt it just shipped."""
    fork = _make_fork(tmp_path, "distwatch", remote)
    (fork / "scripts" / "build.mjs").write_text(
        'import { mkdirSync, writeFileSync } from "node:fs";\n'
        'mkdirSync("dist/cli", {recursive: true});\n'
        'writeFileSync("dist/cli/qmd.js", "// compiled\\n");\n'
        'console.log("wrote dist/cli/qmd.js");\n',
        encoding="utf-8")
    _commit_and_push(fork, "build writes dist like the real fork does")
    (fork / "src" / "vecindex.ts").write_text("export const normalise = 2; // uncommitted\n",
                                             encoding="utf-8")
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode != 0
    assert "FAIL dirty:" in done.stdout
    assert "steps: skipped" in done.stdout, done.stdout[-1200:]
    assert "step build:" not in done.stdout, "the build ran on a dirty fork"
    assert not (fork / "dist" / "cli" / "qmd.js").exists(), \
        "the check compiled an uncommitted tree into the served dist/"


def test_the_verdict_names_the_fork_and_both_interpreters_it_resolved(tmp_path, remote):
    """What makes a red run attributable: which fork was checked and which
    node/bun it got. Without them a failure from a worker is indistinguishable
    from a failure in someone's shell, which is the ambiguity that sent the
    first gate back."""
    fork = _make_fork(tmp_path, "attrib", remote)
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-500:]
    assert done.stdout.splitlines()[0] == f"fork: {fork}"
    assert "tool node: " in done.stdout and "tool bun: " in done.stdout


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    bare = tmp_path / "remote.git"
    bare.mkdir()
    _git(bare, "init", "-q", "--bare")
    return bare


# ── the command exists, and is part of the gated tree ────────────────────────

def test_the_landing_check_is_a_tracked_script_in_the_outer_repo():
    """Route (b) still needs a command in the tree that every gate reads from —
    a check that lives only in the ignored fork is unreachable from a round."""
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files",
                              "scripts/qmd_fork_landing.py"],
                             capture_output=True, text=True).stdout.strip()
    assert tracked == "scripts/qmd_fork_landing.py"


# ── clause: one command answers "is fork sha X landed" ───────────────────────

def test_a_fork_that_builds_passes_its_suite_and_is_pushed_reports_landed(tmp_path, remote):
    fork = _make_fork(tmp_path, "landed", remote)
    sha = _git(fork, "rev-parse", "HEAD")
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    # Both of the fork's own steps actually ran, in this order, and each is
    # reported by the exact script the fork's package.json names.
    assert "step build: ok — node scripts/build.mjs" in out
    assert "step suite: ok — node scripts/test-all.mjs" in out
    assert out.index("step build: ok") < out.index("step suite: ok")
    # And the answer names the branch and the sha, which is the whole point.
    assert "LANDED:" in out
    assert "branch: lloyd" in out
    assert sha in out


def test_the_landed_verdict_names_the_upstream_the_sha_was_pushed_to(tmp_path, remote):
    """"fork sha X is landed" is only a claim about a remote, so the verdict
    must say which remote branch holds it."""
    fork = _make_fork(tmp_path, "landed", remote)
    out = _run_check(tmp_path, "--fork", str(fork)).stdout
    assert "origin/lloyd" in out


# ── clause: build and suite failures are non-zero and attributable ───────────

def test_a_fork_whose_build_fails_exits_nonzero_and_names_the_build_step(tmp_path, remote):
    """The failure mode #577 would otherwise hit: an edit to `qmd/src/**` that
    nothing compiled, reported as a change. The fixture is committed and pushed,
    so the only thing wrong with it is the failing step."""
    fork = _make_fork(tmp_path, "badbuild", remote)
    _break_step(fork, "build.mjs")
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode != 0
    out = done.stdout
    assert "LANDED:" not in out
    assert "NOT LANDED (" in out
    assert "FAIL build:" in out, out
    # The failure output names the step AND quotes what it said, so a reader
    # does not have to re-run it to know which command broke.
    assert "fixture build.mjs fails on purpose" in out
    assert "FAIL suite:" not in out


def test_a_fork_whose_suite_fails_exits_nonzero_and_names_the_suite_step(tmp_path, remote):
    """A green build is not a landed fork: the fork's own suite is the other
    half of the one command."""
    fork = _make_fork(tmp_path, "badsuite", remote)
    _break_step(fork, "test-all.mjs")
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode != 0
    out = done.stdout
    assert "FAIL suite:" in out, out
    assert "fixture test-all.mjs fails on purpose" in out
    # The build passed, and the check says so rather than failing everything.
    assert "step build: ok" in out
    assert "FAIL build:" not in out


def test_a_fork_missing_either_step_script_fails_rather_than_passing_quietly(tmp_path, remote):
    """A fixture with no scripts to run must not read as success — the check
    could otherwise be vacuously green on any tree it cannot build."""
    fork = _make_fork(tmp_path, "nosteps", remote)
    for rel in ("build.mjs", "test-all.mjs"):
        (fork / "scripts" / rel).unlink()
    _commit_and_push(fork, "remove the fork's step scripts")
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode != 0
    assert "FAIL build:" in done.stdout
    assert "FAIL suite:" in done.stdout


def test_a_path_that_is_not_a_fork_fails_rather_than_reporting_landed(tmp_path):
    done = _run_check(tmp_path, "--fork", str(tmp_path / "no-such-fork"))
    assert done.returncode != 0
    assert "FAIL fork:" in done.stdout
    assert "LANDED:" not in done.stdout


# ── clause: dirty or unpushed is not landed ─────────────────────────────────

def test_a_fork_with_uncommitted_changes_is_not_landed(tmp_path, remote):
    """An edit sitting in the working tree is exactly the state a round must
    not be able to call a shipped change."""
    fork = _make_fork(tmp_path, "dirty", remote)
    (fork / "src" / "vecindex.ts").write_text("export const normalise = 2;\n",
                                              encoding="utf-8")
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode != 0
    assert "FAIL dirty:" in done.stdout
    assert "src/vecindex.ts" in done.stdout
    assert "LANDED:" not in done.stdout
    # Still answers the sha question, because that is what was asked.
    assert _git(fork, "rev-parse", "HEAD") in done.stdout


def test_a_fork_with_commits_origin_never_received_is_not_landed(tmp_path, remote):
    """Built, tested and committed is not landed until `origin` has it — the
    half the fork's history cannot show on its own."""
    fork = _make_fork(tmp_path, "ahead", remote)
    (fork / "src" / "vecindex.ts").write_text("export const normalise = 3;\n",
                                              encoding="utf-8")
    _git(fork, "add", "-A")
    _git(fork, "commit", "-qm", "never pushed")
    unpushed = _git(fork, "rev-parse", "HEAD")
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode != 0
    assert "FAIL ahead:" in done.stdout
    assert "1 commit" in done.stdout
    assert "LANDED:" not in done.stdout
    assert unpushed in done.stdout


def test_a_fork_branch_with_no_upstream_cannot_claim_a_landed_sha(tmp_path, remote):
    """A local-only branch has no remote that could have received the sha, so
    the verdict is 'not landed', not a silent pass on `0 ahead`."""
    fork = _make_fork(tmp_path, "noup", remote)
    _git(fork, "push", "-q", "origin", "lloyd:refs/heads/scratch")
    _git(fork, "branch", "--unset-upstream")
    _git(fork, "branch", "-m", "lloyd", "scratch")
    done = _run_check(tmp_path, "--fork", str(fork / "."))
    assert done.returncode != 0
    assert "FAIL upstream:" in done.stdout
    assert "LANDED:" not in done.stdout


# ── the pair the check runs is fixed by the outer tree, not by the fork ──────

def test_the_check_runs_the_documented_pair_at_the_default_fork_path():
    """Three places name the two step scripts and they must agree: `STEPS`,
    CLAUDE.md's Automod section (the sentence a round reads), and the fork's own
    package.json — whose agreement with the *live* fork is the separate leg at the
    bottom of this file, reported as skipped rather than passed when the fork is
    not on disk.

    `DEFAULT_FORK` is pinned here as well, because it is the one value the shipped
    command line depends on and no other test in this file passes it: `--fork`
    overrides it in every other leg. Rename the default and every other green leg
    would keep its green while `python -m scripts.qmd_fork_landing` checked a
    stale path."""
    from scripts.qmd_fork_landing import DEFAULT_FORK, STEPS

    assert [rel for _, rel in STEPS] == ["scripts/build.mjs", "scripts/test-all.mjs"], STEPS
    assert DEFAULT_FORK == FORK, DEFAULT_FORK

    doc = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    heading = "## Automod (self-modification)"
    body = doc[doc.index(heading) + len(heading):]
    section = body[:body.index("\n## ")]
    for _, rel in STEPS:
        assert f"node {rel}" in section, \
            f"CLAUDE.md's Automod section no longer names `node {rel}`"


def test_editing_the_forks_package_json_cannot_change_what_the_check_runs(tmp_path, remote):
    """The pair is invoked directly, not through `npm run` / `bun run`, and this
    proves the consequence: a fork whose package.json points `build` and `test`
    somewhere else does not redirect the check. Read the other way the property is
    the hazard — a check that followed package.json could not notice the build step
    it was written to run had gone, while the outer doc kept telling rounds it
    ran, and a fork author could point the "landed" verdict at a step that proves
    nothing.

    The decoy is the one that *passes* and the documented build is the one that
    fails, so a redirected check would have reported success from the decoy. That
    inversion is the fixture the review rung asked clause 5 to name."""
    fork = _make_fork(tmp_path, "decoypkg", remote)
    (fork / "package.json").write_text(
        '{"name": "fixture-qmd", "version": "0.0.0", "scripts": {'
        '"build": "node scripts/decoy-build.mjs", '
        '"test": "node scripts/decoy-test.mjs"}}\n', encoding="utf-8")
    _step(fork, "decoy-build.mjs", ok=True)
    _step(fork, "decoy-test.mjs", ok=True)
    _break_step(fork, "build.mjs")
    done = _run_check(tmp_path, "--fork", str(fork))
    assert done.returncode != 0
    out = done.stdout
    assert "FAIL build:" in out, out[-1500:]
    assert "fixture build.mjs fails on purpose" in out, out[-1500:]
    assert "decoy-build" not in out, "the check followed the fork's package.json"
    assert "decoy-test" not in out, "the check followed the fork's package.json"


def test_the_live_fork_still_declares_the_pair_the_outer_tree_names():
    """The third naming of the pair is the fork's own package.json. If it ever
    stops saying `node scripts/build.mjs` / `node scripts/test-all.mjs`, the check
    above is running a pair its own doc no longer describes, and CLAUDE.md is
    pointing rounds at steps the fork has renamed.

    A cross-tree claim can only be checked where the other tree exists, so an
    absent fork is reported as *skipped*, not as a pass: the previous shape
    returned from inside a test that also asserted outer-tree facts, which made
    the unrun half invisible on every machine that has no fork — including every
    automod worktree on a machine without one."""
    if not FORK.is_dir():
        pytest.skip(f"{FORK} is not on disk here, so the fork's package.json "
                    "cannot be read; the outer-tree legs above do not depend on it")
    declared = json.loads((FORK / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert declared["build"].startswith("node scripts/build.mjs"), declared["build"]
    assert declared["test"].startswith("node scripts/test-all.mjs"), declared["test"]
