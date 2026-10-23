"""backup-vault.sh must clear the lock a DEAD run left, and never one a LIVE run holds.

The vault snapshot died on 2026-10-16 at 11:03, when the run holding
`index.lock` was killed by a boot; every one of the 370 runs after it died at
`g add --all --force .` ("Unable to create '…/index.lock': File exists") and
exited 128, with the snapshot frozen at `da894bfe` and 8145 files while the live
vault had grown to 8516. `gc.pid.lock` was already sitting there from
2026-10-14 17:15, left by a killed `git gc`, and the `|| true` on that line had
made the run that found it exit 0 and look healthy. The timer guard in
`scripts/service_health_check.py` (#1891) passed the whole time because the
timer really was firing every 15 minutes: only the runs were failing.

So a lock left by a dead process must be removed by the script itself, but only
on two proofs — no live holder, and older than one timer period — because the
same file is also what a *concurrent* run holds, and bulldozing that corrupts
the index being written. Every node here drives the real script through a fake
`$HOME` and the three env overrides it defines (`LLOYD_VAULT_ROOT`,
`LLOYD_VAULT_BACKUP_REPO`, `LLOYD_VAULT_TRIP_MARKER`), the way
`tests/test_backup_graph.py` drives its script, and none of them touches the
live backup repository.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "backup" / "backup-vault.sh"
TIMER = ROOT / "agent-services" / "systemd" / "lloyd-vault-backup.timer"

#: One `lloyd-vault-backup.timer` period. `test_the_stale_limit_is_exactly_one_timer_period`
#: re-derives it from the unit rather than trusting this number.
TIMER_PERIOD_SECONDS = 900


def _vault(root: Path, files: int = 40) -> Path:
    for i in range(files):
        folder = ("backlog", "skills", "memory", "knowledge")[i % 4]
        p = root / folder / f"note-{i}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"note {i}\n")
    return root


def _run(tmp_path: Path, vault: Path, repo: Path, marker: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=str(tmp_path / "home"), LLOYD_VAULT_ROOT=str(vault),
               LLOYD_VAULT_BACKUP_REPO=str(repo), LLOYD_VAULT_TRIP_MARKER=str(marker))
    return subprocess.run([str(SCRIPT)], capture_output=True, text=True, env=env,
                          timeout=180, cwd=str(ROOT))


def _head(repo: Path) -> str | None:
    """HEAD's sha in the backup repo, or None when it has no commit yet. A read
    only: `log` never moves anything."""
    r = subprocess.run(["git", "--git-dir", str(repo), "log", "-1", "--format=%H"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


def _head_ct(repo: Path) -> str | None:
    """HEAD's `%ct` — the stamp `snapshot_age_state` weighs — or None with no
    commit. The beat tests compare this before and after a run: a `no change`
    delivery must leave it exactly where it was."""
    r = subprocess.run(["git", "--git-dir", str(repo), "log", "-1", "--format=%ct"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


def _snapshot(tmp_path: Path, vault: Path, repo: Path, marker: Path) -> str:
    """Get the repo to a committed state, and return that commit."""
    out = _run(tmp_path, vault, repo, marker)
    assert out.returncode == 0 and "committed" in out.stdout, out.stdout + out.stderr
    head = _head(repo)
    assert head, out.stdout
    return head


def _plant(repo: Path, name: str, *, age_seconds: float) -> Path:
    """A lock file aged `age_seconds`, the state a killed run leaves behind."""
    repo.mkdir(parents=True, exist_ok=True)
    f = repo / name
    f.write_text("")
    old = time.time() - age_seconds
    os.utime(f, (old, old))
    return f


def _wait_until_fuser_sees(lock: Path, pid: int, timeout: float = 15.0) -> str:
    """Block until `fuser` reports `pid` holding `lock`, or fail.

    The positive control of the holder probe: without it, a node asserting
    "refused because held" would pass just as well on a box where `fuser` cannot
    see the holder, which is the same probe the script is about to run.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = subprocess.run(["fuser", str(lock)], capture_output=True, text=True)
        last = (r.returncode, r.stdout)
        if r.returncode == 0 and str(pid) in r.stdout:
            return r.stdout.strip()
        time.sleep(0.2)
    raise AssertionError(
        f"fuser never reported pid {pid} holding {lock} within {timeout}s; last was {last}. "
        "The script's holder probe is fuser, so a live-holder test cannot pass without it.")


def _hold_open(path: Path, seconds: float = 120) -> subprocess.Popen:
    """A live process that keeps `path` open, which is what `fuser` reports.

    The handle must be BOUND: `open(p, 'a')` as a bare statement returns a file
    object with no reference, CPython collects it at once, the fd is closed
    before the run under test asks `fuser`, and the node fails for a reason that
    has nothing to do with the script.
    """
    return subprocess.Popen([sys.executable, "-c",
                             "import sys, time; handle = open(sys.argv[1], 'a'); "
                             f"time.sleep({seconds})", str(path)])


def _dead_pid() -> int:
    """A pid the kernel has reaped, so `kill -0` on it fails."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(timeout=30)
    return p.pid


# ── clause 1: a stale, holder-less lock is cleared and that run commits ────────

def test_a_stale_holder_less_index_lock_is_cleared_and_the_run_commits(tmp_path):
    """The 2026-10-16 case: `index.lock` from a run killed mid-`git add`.

    The order the clause demands is visible in the output, not assumed: a sweep
    that ran AFTER `g add` would have died at `g add` with status 128 and never
    printed the clearing line at all, so asserting both the line and the commit
    is what pins "before any git call".
    """
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    before = _snapshot(tmp_path, vault, repo, marker)

    (vault / "backlog" / "written-while-dead.md").write_text("new\n")
    lock = _plant(repo, "index.lock", age_seconds=2 * 3600.0)

    out = _run(tmp_path, vault, repo, marker)

    assert out.returncode == 0, out.stdout + out.stderr
    assert "cleared stale index.lock" in out.stdout, out.stdout
    assert not lock.exists(), "the stale lock is still there"
    assert "committed" in out.stdout, out.stdout
    assert _head(repo) != before, "the run cleared the lock but wrote no commit"
    # The sweep is scoped to `*.lock`: the count file it lives beside is data.
    assert (repo / "lloyd-last-count").read_text().strip() == "41"


def test_a_stale_gc_pid_lock_is_cleared_too_and_the_dead_pid_file_goes_with_it(tmp_path):
    """The lock nobody saw, which is the one the item insists must not stay silent.

    `gc.pid.lock` has been sitting here since 2026-10-14 17:15 and no run ever
    reported it, because `g gc --auto --quiet || true` swallowed the fatal and
    the run exited 0. A stale `gc.pid` is removed with its own lock: a pid file
    naming a process that no longer exists is nothing to refuse on, and if it
    stayed and that pid were recycled to a live process, the probe would refuse
    forever on a pid that has nothing to do with git.
    """
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    before = _snapshot(tmp_path, vault, repo, marker)

    lock = _plant(repo, "gc.pid.lock", age_seconds=3 * 3600.0)
    pid_file = repo / "gc.pid"
    pid_file.write_text(f"{_dead_pid()}\n")
    os.utime(pid_file, (time.time() - 3 * 3600.0, time.time() - 3 * 3600.0))
    (vault / "backlog" / "written-while-dead.md").write_text("new\n")

    out = _run(tmp_path, vault, repo, marker)

    assert out.returncode == 0, out.stdout + out.stderr
    assert "cleared stale gc.pid.lock" in out.stdout, out.stdout
    assert not lock.exists() and not pid_file.exists(), (list(repo.iterdir()))
    assert _head(repo) != before, out.stdout


def test_both_locks_left_by_the_two_deaths_are_cleared_in_one_run(tmp_path):
    """The live state: `index.lock` from the boot-killed run, `gc.pid.lock` from
    the killed gc, planted together as they are on the box."""
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    before = _snapshot(tmp_path, vault, repo, marker)

    index = _plant(repo, "index.lock", age_seconds=93.5 * 3600.0)
    gc = _plant(repo, "gc.pid.lock", age_seconds=141 * 3600.0)
    (vault / "backlog" / "written-while-dead.md").write_text("new\n")

    out = _run(tmp_path, vault, repo, marker)

    assert out.returncode == 0, out.stdout + out.stderr
    assert not index.exists() and not gc.exists()
    assert out.stdout.count("cleared stale") == 2, out.stdout
    assert _head(repo) != before
    # What must NOT go with them. Strip `.lock` from `index.lock` and the name
    # left is `<repo>/index` — git's staging index, a binary file, not a pid file
    # — so a sweep that removed "whatever the lock is named after" would turn a
    # wedged snapshot into a corrupted one. This is the assertion that catches it.
    assert (repo / "index").exists(), "the sweep deleted git's staging index"


# ── clause 2: a lock a live process owns is left exactly where it is ──────────

def test_a_lock_held_by_a_live_process_is_left_and_the_run_fails_without_committing(tmp_path):
    """Age alone must never clear a lock: a run in flight owns it.

    The lock is deliberately 2 h old, so the only thing that can make this run
    refuse is the holder probe — and after the holder exits, the very same lock
    is stale and the next run clears it and commits. That pair of runs is the
    proof that the holder, not the age, was the reason.
    """
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    before = _snapshot(tmp_path, vault, repo, marker)

    (vault / "backlog" / "written-while-dead.md").write_text("new\n")
    lock = _plant(repo, "index.lock", age_seconds=2 * 3600.0)
    holder = _hold_open(lock)
    try:
        _wait_until_fuser_sees(lock, holder.pid)

        refused = _run(tmp_path, vault, repo, marker)

        assert refused.returncode != 0, refused.stdout + refused.stderr
        assert "held by a live process" in refused.stdout, refused.stdout
        assert str(holder.pid) in refused.stdout, refused.stdout
        assert lock.exists(), "a lock a live process holds was deleted"
        assert _head(repo) == before, "a commit was written over a live run's lock"
        assert "committed" not in refused.stdout, refused.stdout
    finally:
        holder.terminate()
        holder.wait(timeout=30)

    after = _run(tmp_path, vault, repo, marker)
    assert after.returncode == 0, after.stdout + after.stderr
    assert "cleared stale index.lock" in after.stdout, after.stdout
    assert _head(repo) != before


def test_a_gc_pid_lock_whose_pid_file_names_a_live_process_is_refused(tmp_path):
    """The second half of the probe: git's own pid file, not an open fd.

    git renames `gc.pid.lock` to `gc.pid` holding the live gc's pid, so a gc in
    flight may not have the lock open by then at all and `fuser` alone would
    clear it out from under it. `os.getpid()` is the live process here — this
    test process, alive for the whole run — so no timing is involved.
    """
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    before = _snapshot(tmp_path, vault, repo, marker)

    lock = _plant(repo, "gc.pid.lock", age_seconds=2 * 3600.0)
    (repo / "gc.pid").write_text(f"{os.getpid()}\n")

    out = _run(tmp_path, vault, repo, marker)

    assert out.returncode != 0, out.stdout
    assert "held by a live process" in out.stdout, out.stdout
    assert str(os.getpid()) in out.stdout, out.stdout
    assert lock.exists() and (repo / "gc.pid").exists()
    assert _head(repo) == before, out.stdout


# ── clause 3: a young holder-less lock is left too ────────────────────────────

def test_a_holder_less_lock_younger_than_one_timer_period_is_left_and_fails(tmp_path):
    """No holder yet does not mean no run: `git add` creates the lock, and a
    snapshot that started 30 s ago is mid-flight on a 5.5k-file vault. The timer
    period is the bound on how old a live run can be, so a younger lock is
    refused rather than bulldozed."""
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    before = _snapshot(tmp_path, vault, repo, marker)

    lock = _plant(repo, "index.lock", age_seconds=30.0)

    out = _run(tmp_path, vault, repo, marker)

    assert out.returncode != 0, out.stdout
    assert "REFUSING" in out.stdout, out.stdout
    assert "900" in out.stdout, out.stdout
    assert lock.exists(), "a lock inside the current timer period was deleted"
    assert _head(repo) == before, out.stdout


def test_the_limit_that_decides_is_one_timer_period_and_not_just_any_age(tmp_path):
    """860 s refuses, 910 s clears: the number in the script is the 15 minutes
    of `OnCalendar=*:0/15`, not a token older-than-a-moment."""
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    before = _snapshot(tmp_path, vault, repo, marker)
    (vault / "backlog" / "written-while-dead.md").write_text("new\n")

    young = _plant(repo, "index.lock", age_seconds=float(TIMER_PERIOD_SECONDS) - 40)
    refused = _run(tmp_path, vault, repo, marker)
    assert refused.returncode != 0 and young.exists(), refused.stdout

    old = _plant(repo, "index.lock", age_seconds=float(TIMER_PERIOD_SECONDS) + 10)
    cleared = _run(tmp_path, vault, repo, marker)
    assert cleared.returncode == 0, cleared.stdout + cleared.stderr
    assert not old.exists()
    assert _head(repo) != before


def test_the_stale_limit_is_exactly_one_timer_period_of_the_timer_that_fires_it(tmp_path):
    """The script's bound and the unit's period are one number across a file
    boundary the script never reads: parse `OnCalendar=*:0/15` and compare, so
    moving the timer to 10 minutes turns this red instead of quietly leaving a
    15-minute bulldoze window."""
    cal = re.search(r"^OnCalendar=(.+)$", TIMER.read_text(encoding="utf-8"), re.M).group(1)
    minutes = int(re.search(r"/(\d+)$", cal).group(1))
    assert cal == "*:0/15" and minutes * 60 == TIMER_PERIOD_SECONDS, cal

    declared = int(re.search(r"^LOCK_STALE_SECONDS=(\d+)",
                             SCRIPT.read_text(encoding="utf-8"), re.M).group(1))
    assert declared == TIMER_PERIOD_SECONDS == minutes * 60, (declared, minutes)


# ── clause 5: the two refusals that already existed are untouched ─────────────

def test_the_tripwire_refusal_still_exits_zero_and_says_so(tmp_path):
    """A refusal while an incident is open is the intended state, and the timer
    must not be shown it as a failure. Unchanged by the lock sweep, which is
    why the tripwire test stays ahead of it in the script."""
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    _snapshot(tmp_path, vault, repo, marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"reason": "test"}')

    out = _run(tmp_path, vault, repo, marker)

    assert out.returncode == 0, out.stdout + out.stderr
    assert "vault tripwire is set" in out.stdout, out.stdout


def test_the_shrunken_vault_refusal_still_exits_zero_and_names_both_counts(tmp_path):
    """The second net under the tripwire, with its counts, still exiting 0."""
    vault = _vault(tmp_path / "obsidian", files=120)
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    _snapshot(tmp_path, vault, repo, marker)
    assert (repo / "lloyd-last-count").read_text().strip() == "120"

    for folder in ("backlog", "skills"):
        for f in (vault / folder).glob("*.md"):
            f.unlink()

    out = _run(tmp_path, vault, repo, marker)

    assert out.returncode == 0, out.stdout + out.stderr
    assert "REFUSING: vault holds 60 files, the last snapshot held 120" in out.stdout, out.stdout


# ── the silence that hid the first lock ───────────────────────────────────────

def test_the_gc_line_no_longer_swallows_its_own_failure():
    """`g gc --auto --quiet || true` is how the 2026-10-14 lock went unnoticed:
    the run committed, gc printed `fatal: Unable to create '…/gc.pid.lock':
    File exists`, and `|| true` made the exit 0 that a healthy snapshot looks
    like. Clearing the next run's lock is no help if nothing ever reported that
    one, so the failure has to reach the journal as a word, not a silence."""
    source = SCRIPT.read_text(encoding="utf-8")

    gc_lines = [ln.strip() for ln in source.splitlines() if "gc --auto" in ln]
    assert len(gc_lines) == 1, gc_lines
    assert "|| true" not in gc_lines[0], gc_lines[0]
    assert "REFUSING" not in gc_lines[0], (
        f"a gc failure must not fail the run — the commit above it succeeded: {gc_lines[0]}")
    assert "WARNING" in gc_lines[0], gc_lines[0]


# ── #3184: the beat the guardian now weighs, and what must never write it ──────

def test_an_unchanged_vault_advances_the_beat_without_moving_head(tmp_path):
    """The asymmetry the check rests on: `echo "$count" > "$last_file"` runs
    after BOTH branches of the commit step, so a `no change` delivery re-beats
    while HEAD's `%ct` does not move at all. If this ever stops holding — if the
    write moves above the branch, or the branch starts exiting early — the
    guardian's beat age would say "alive" about a producer that wrote nothing,
    or "stopped" about one delivering, which is the exact confusion #3184 was
    filed to end."""
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    _snapshot(tmp_path, vault, repo, marker)
    beat = repo / "lloyd-last-count"
    ct_before, beat_before = _head_ct(repo), beat.stat().st_mtime

    out = _run(tmp_path, vault, repo, marker)          # vault untouched: idle beat

    assert out.returncode == 0, out.stdout + out.stderr
    assert "no change (40 files)" in out.stdout, out.stdout
    assert _head_ct(repo) == ct_before, (
        "the idle run moved HEAD; %ct is no longer the witness of an unchanged vault")
    assert beat.stat().st_mtime > beat_before, (
        "a delivering run did not advance lloyd-last-count — the beat is dead as a witness")


def test_a_refused_run_beats_no_more_than_a_dead_one(tmp_path):
    """The other edge of the same property: the two refusals that exit 0 — the
    tripwire (`vault-tripped.json` present) and the shrink guard — return BEFORE
    the beat write, so neither advances the beat nor HEAD. If either started
    beating, the guardian could never distinguish a producer refusing on purpose
    from one delivering, and #3184's predicate would page on the wrong shape."""
    vault = _vault(tmp_path / "obsidian")
    repo, marker = tmp_path / "state" / "vault.git", tmp_path / "gstate" / "vault-tripped.json"
    _snapshot(tmp_path, vault, repo, marker)
    beat = repo / "lloyd-last-count"
    ct_before, beat_before = _head_ct(repo), beat.stat().st_mtime

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"reason": "200 of 401 vault files disappeared"}')
    out = _run(tmp_path, vault, repo, marker)
    assert out.returncode == 0 and "tripwire is set" in out.stdout, out.stdout
    assert (beat.stat().st_mtime, _head_ct(repo)) == (beat_before, ct_before), \
        "the tripwire refusal wrote a beat — a muted incident now reads as delivering"
    marker.unlink()

    for f in (vault / "backlog").glob("*.md"):          # 30 of 40 < the 90% floor
        f.unlink()
    out = _run(tmp_path, vault, repo, marker)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "REFUSING: vault holds 30 files, the last snapshot held 40" in out.stdout, out.stdout
    assert (beat.stat().st_mtime, _head_ct(repo)) == (beat_before, ct_before), \
        "the shrink refusal wrote a beat — a refusing producer now reads as delivering"
