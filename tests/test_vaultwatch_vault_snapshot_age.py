"""The vault snapshot's own age is now a reading, where 370 failures were invisible.

`backup-vault.sh` is the mechanism Alan's 2026-09-24 ruling rests on — "the vault
is covered by btrfs snapshots" (#1142) — and the btrfs half covers only
`~/lloyd-data` (`scripts/backup/snapshot-data.sh` takes `DATA=$HOME/lloyd-data`),
so this git snapshot is the only copy of the vault outside the vault itself. It
stopped on 2026-10-16 10:45 and nothing said so for four days:
`lloyd-vault-backup.timer` stayed enabled and fired on schedule, so the timer
guard (#1891) passed; `vaultwatch.py`, the module that already reads vault health
for the guardian, never looked at the backup repo at all; and
`pre-reimage-backup.sh` tests only that the directory exists, so a four-day-frozen
repo was still reported as a present backup.

This is #1416's bounded-age check, mirrored onto the vault:
`datawatch.snapshot_state` and `snapshot_report` are one read of the data-root
snapshot directory feeding two readers, and `snapshot_age_state` /
`snapshot_age_report` do the same for this repository's HEAD, against the same
3 h bound as `policy.SNAPSHOT_MAX_AGE_SECONDS`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import policy  # noqa: E402
import vaultwatch as VW  # noqa: E402

THREE_HOURS = 3 * 3600.0


def _git(*args: str, env: dict[str, str] | None = None) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True,
                       env=dict(os.environ, **(env or {})))
    assert r.returncode == 0, f"{args}: {r.stdout}{r.stderr}"
    return r.stdout.strip()


def _repo_with_a_head(tmp_path: Path, *, age_seconds: float) -> tuple[Path, str, int]:
    """A backup-shaped repo whose HEAD commit is `age_seconds` old.

    The age goes into the commit object (`GIT_COMMITTER_DATE`), not into a file
    mtime: `%ct` is what the check reads, and it is read back here to prove git
    honoured the planted epoch rather than the node passing on a stamp it never
    set.
    """
    repo = tmp_path / "vault.git"
    work = tmp_path / "vault"
    work.mkdir(parents=True)
    (work / "note.md").write_text("a note\n")
    _git("init", "--quiet", "--bare", str(repo))
    _git("--git-dir", str(repo), "--work-tree", str(work), "add", "--all", ".")
    stamp = int(time.time() - age_seconds)
    _git("--git-dir", str(repo), "--work-tree", str(work), "commit", "--quiet",
         "--no-verify", "-m", "snapshot",
         env={"GIT_AUTHOR_DATE": f"@{stamp} +0000",
              "GIT_COMMITTER_DATE": f"@{stamp} +0000"})
    shown = _git("--git-dir", str(repo), "log", "-1", "--format=%ct")
    assert int(shown) == stamp, f"git stored committer time {shown}, not the planted {stamp}"
    sha = _git("--git-dir", str(repo), "log", "-1", "--format=%H")
    return repo, sha, stamp


# ── the judgement ────────────────────────────────────────────────────────────

def test_a_head_older_than_three_hours_is_not_fresh_and_names_its_age(tmp_path):
    """93 h of dead snapshot is the state this node pins at 4 h: non-green, and
    the age inside the sentence, because a reading that says only "stale"
    cannot tell one skipped cycle from a mechanism that died on Tuesday."""
    repo, sha, _stamp = _repo_with_a_head(tmp_path, age_seconds=4 * 3600.0)
    now = time.time()

    state = VW.snapshot_age_state(str(repo), THREE_HOURS, now=now)

    assert state["sha"] == sha
    assert state["fresh"] is False, state
    assert 4 * 3600.0 <= state["age"] < 4 * 3600.0 + 120, state
    assert "4 h 0 m old" in state["text"], state["text"]
    assert "past the 3 h limit" in state["text"], state["text"]
    assert sha[:8] in state["text"], state["text"]


def test_a_head_fresh_from_the_timer_is_green(tmp_path):
    repo, sha, _stamp = _repo_with_a_head(tmp_path, age_seconds=20.0)

    fresh, text = VW.snapshot_age_report(str(repo), THREE_HOURS, now=time.time())

    assert fresh is True, text
    assert re.search(r"is \d+ m old", text), text
    assert "past the" not in text, text
    assert sha[:8] in text, text


def test_the_limit_is_the_bounded_three_hours_the_data_root_check_already_uses():
    """The threshold is #1416's, not a new opinion: `policy.py` states 3 h for
    the data root so that one skipped cycle is not an alarm, and this item asks
    for the same bound."""
    assert VW.SNAPSHOT_MAX_AGE_SECONDS == THREE_HOURS == policy.SNAPSHOT_MAX_AGE_SECONDS


def test_a_repo_with_no_readable_head_is_not_green_by_absence(tmp_path):
    """The trap the check has to fall on the right side of: `git --git-dir=…`
    against a path that does not resolve answers `fatal: not a git repository`,
    and a check that reads "no commit time" as "no problem" would have reported
    the dead four days as fresh. A missing repo is the worst age there is."""
    missing = tmp_path / "gone.git"
    assert not missing.exists()

    state = VW.snapshot_age_state(str(missing), THREE_HOURS, now=time.time())

    assert state["fresh"] is False, state
    assert state["sha"] is None and state["age"] is None, state
    assert str(missing) in state["text"], state["text"]

    empty = tmp_path / "empty.git"
    _git("init", "--quiet", "--bare", str(empty))
    bare = VW.snapshot_age_state(str(empty), THREE_HOURS, now=time.time())
    assert bare["fresh"] is False and bare["sha"] is None, bare


def test_the_two_readers_report_the_one_measurement(tmp_path):
    """`snapshot_age_state` is the single read, exactly as `snapshot_state` is
    for the data root: the age an operator prints and the age a check alerts on
    must not be two reads that can disagree."""
    repo, sha, _stamp = _repo_with_a_head(tmp_path, age_seconds=93.5 * 3600.0)
    now = time.time()

    state = VW.snapshot_age_state(str(repo), THREE_HOURS, now=now)
    fresh, text = VW.snapshot_age_report(str(repo), THREE_HOURS, now=now)

    assert (fresh, text) == (state["fresh"], state["text"]), (fresh, state)
    assert state["sha"] == sha
    assert "3 d 21 h old" in text, text


# ── #3184: the per-run beat, which is how an idle producer differs from a dead one ──

def _beat(repo: Path, *, age_seconds: float) -> Path:
    """Plant `<repo>/lloyd-last-count` with its mtime `age_seconds` old.

    This is the file `backup-vault.sh:175` writes (`echo "$count" > "$last_file"`)
    after BOTH branches of its commit step, so its mtime advances on every run
    that reaches that line — including a run over an unchanged vault, which
    commits nothing and moves HEAD's `%ct` not at all. The mtime is the beat;
    the content is the file count and none of this weighs it.
    """
    beat = repo / VW.SNAPSHOT_BEAT_FILE
    beat.write_text("8516\n")
    old = time.time() - age_seconds
    os.utime(beat, (old, old))
    return beat


def test_an_idle_but_delivering_producer_is_fresh_on_its_beat(tmp_path):
    """The 2026-10-22 07:09 page, replayed at its measured numbers (#3184).

    HEAD was 11,318 s old — past the 3 h bound by 518 s — while the journal shows
    twelve consecutive `no change` runs from 04:15:20 to 07:00:57, the last of
    which wrote a beat 519 s before the alarm read the repo. Weighing HEAD alone
    paged a producer that had delivered twelve times in a row; the beat predicate
    must read this state fresh, and the sentence must say why HEAD's age alone
    stopped being the whole story.
    """
    repo, sha, _stamp = _repo_with_a_head(tmp_path, age_seconds=11_318.0)
    _beat(repo, age_seconds=519.0)

    state = VW.snapshot_age_state(str(repo), THREE_HOURS, now=time.time())

    assert 11_318.0 <= state["age"] < 11_318.0 + 120, state
    assert state["age"] > THREE_HOURS, state            # HEAD alone would page…
    assert 519.0 <= state["beat_age"] < 639.0, state    # …the beat says otherwise
    assert state["fresh"] is True, state
    assert "3 h 8 m old" in state["text"], state["text"]
    assert "last beat was 8 m old" in state["text"], state["text"]
    assert "past the 3 h limit" not in state["text"].split("HEAD is past")[0], state["text"]


def test_both_ages_past_the_bound_is_not_fresh_and_names_both(tmp_path):
    """The clause's other side: HEAD 11,318 s AND beat 11,400 s are both past
    10,800 s, so nothing argues the producer is merely idle, and the sentence a
    person gets must carry both ages it weighed — the commit's with its sha
    already there from #3132, and the beat's added by #3184."""
    repo, sha, _stamp = _repo_with_a_head(tmp_path, age_seconds=11_318.0)
    _beat(repo, age_seconds=11_400.0)

    state = VW.snapshot_age_state(str(repo), THREE_HOURS, now=time.time())

    assert state["beat_age"] > THREE_HOURS, state
    assert state["fresh"] is False, state
    assert sha[:8] in state["text"], state["text"]
    assert "3 h 8 m old" in state["text"], state["text"]
    assert "past the 3 h limit" in state["text"], state["text"]
    assert "the producer's last beat is 3 h 10 m old" in state["text"], state["text"]
    assert "past the same 3 h limit" in state["text"], state["text"]


def test_a_stale_head_with_no_beat_written_is_no_beat_and_not_alive(tmp_path):
    """The freeze shape the alarm exists to catch (#3184 clause 3). During the
    2026-10-16→10-20 freeze every one of the 370 failed runs died inside
    `git add` (exit 128) — before the beat write at `backup-vault.sh:175` — and
    the tripwire refusal (:56-59) and the shrink refusal (:158-163) exit 0 even
    earlier. So a producer that stopped leaves no beat (or none that moves), and
    absence must weigh as NO witness, never as alive: HEAD stale plus a missing
    beat is still not fresh. An unreadable beat (a dangling symlink `stat`
    cannot resolve) is the same nothing, not a green."""
    repo, _sha, _stamp = _repo_with_a_head(tmp_path, age_seconds=93.5 * 3600.0)
    assert not (repo / VW.SNAPSHOT_BEAT_FILE).exists()

    state = VW.snapshot_age_state(str(repo), THREE_HOURS, now=time.time())

    assert state["beat_age"] is None, state
    assert state["fresh"] is False, state
    assert "3 d 21 h old" in state["text"], state["text"]
    assert f"no per-run beat ({VW.SNAPSHOT_BEAT_FILE})" in state["text"], state["text"]

    (repo / VW.SNAPSHOT_BEAT_FILE).symlink_to(repo / "written-then-gone")
    broken = VW.snapshot_age_state(str(repo), THREE_HOURS, now=time.time())
    assert broken["beat_age"] is None and broken["fresh"] is False, broken


# ── the surface a person or a check actually runs ─────────────────────────────

def _cli(*args: str, home_repo: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if home_repo is not None:
        env["LLOYD_VAULT_BACKUP_REPO"] = home_repo
    return subprocess.run([sys.executable, str(GUARDIAN_DIR / "vaultwatch.py"), *args],
                          capture_output=True, text=True, env=env, timeout=60,
                          cwd=str(GUARDIAN_DIR))


def test_the_cli_goes_non_zero_on_a_stale_head_and_zero_on_a_fresh_one(tmp_path):
    """Green/non-green has to be a status code and not only a word, or the check
    can be run and still be ignored by whatever runs it — which is how 370
    exit-128 runs went unreported."""
    stale, _sha, _stamp = _repo_with_a_head(tmp_path / "stale", age_seconds=4 * 3600.0)
    out = _cli("snapshot-age", str(stale))
    assert out.returncode == 1, (out.stdout, out.stderr)
    # The refusal goes to stderr, which is where this module already puts the
    # verdict of a check that is refusing (`sync-gate`'s "REFUSING: ").
    assert out.stderr.startswith("STALE:"), (out.stdout, out.stderr)
    assert "4 h 0 m old" in out.stderr, out.stderr

    fresh, _sha2, _s2 = _repo_with_a_head(tmp_path / "fresh", age_seconds=10.0)
    ok = _cli("snapshot-age", str(fresh))
    assert ok.returncode == 0, (ok.stdout, ok.stderr)
    assert ok.stdout.startswith("fresh:"), ok.stdout


def test_the_cli_reads_the_default_repo_from_the_same_place_the_script_writes_it(tmp_path):
    """No argument, no second config: `vaultwatch.py snapshot-age` and
    `backup-vault.sh` must agree on which repository holds the vault, through
    the same `LLOYD_VAULT_BACKUP_REPO` override the script already takes."""
    repo, _sha, _stamp = _repo_with_a_head(tmp_path, age_seconds=10.0)

    out = _cli("snapshot-age", home_repo=str(repo))

    assert out.returncode == 0, (out.stdout, out.stderr)
    assert str(repo) in out.stdout, out.stdout


def test_status_carries_the_snapshot_alongside_the_tripwire(tmp_path):
    """`vaultwatch.py status` is the reading of vault health that exists today,
    and for four days it said "tripwire not set, N files" while the thing a
    restore would have been taken from had not moved. The snapshot age now rides
    in the same JSON."""
    repo, sha, _stamp = _repo_with_a_head(tmp_path, age_seconds=93.5 * 3600.0)
    vault = tmp_path / "obsidian"
    (vault / "backlog").mkdir(parents=True)
    (vault / "backlog" / "note.md").write_text("x\n")

    out = subprocess.run(
        [sys.executable, str(GUARDIAN_DIR / "vaultwatch.py"), "status"],
        capture_output=True, text=True, timeout=60, cwd=str(GUARDIAN_DIR),
        env=dict(os.environ, LLOYD_VAULT_ROOT=str(vault), LLOYD_VAULT_BACKUP_REPO=str(repo),
                 LLOYD_GUARDIAN_STATE=str(tmp_path / "gstate")))

    assert out.returncode == 0, out.stderr
    body = json.loads(out.stdout)
    snap = body["snapshot"]
    assert snap["sha"] == sha, snap
    assert snap["fresh"] is False, snap
    assert "3 d 21 h" in snap["text"], snap


def test_the_default_repo_is_the_one_backup_vault_sh_writes_by_default():
    """The two defaults are one path, stated across a language boundary: the
    shell script's `${LLOYD_VAULT_BACKUP_REPO:-$HOME/…}` and this module's
    default. Drift here is the silent kind — a check reading a repo that is not
    the one being written sees a stale HEAD forever and looks like a check that
    passed."""
    script = (ROOT / "scripts" / "backup" / "backup-vault.sh").read_text(encoding="utf-8")
    shell_default = re.search(r'^REPO="\$\{LLOYD_VAULT_BACKUP_REPO:-(.+)\}"', script,
                              re.M).group(1)

    assert shell_default.startswith("$HOME/"), shell_default
    assert f"$HOME/{VW.SNAPSHOT_REPO_REL}" == shell_default, (shell_default, VW.SNAPSHOT_REPO_REL)
    assert VW.SNAPSHOT_REPO == Path.home() / VW.SNAPSHOT_REPO_REL, VW.SNAPSHOT_REPO


def test_the_beat_is_the_same_file_the_producer_writes_into_the_repo():
    """The beat crosses a language boundary too: the check reads
    `<repo>/lloyd-last-count` because that is the name
    `backup-vault.sh:157/175` writes it under, inside the very repo dir
    `snapshot_repo_is_owned` confines. Drift here is drift of the repo-default
    test above in a quieter key — the check would weigh a file nobody touches and
    every idle streak would page again."""
    script = (ROOT / "scripts" / "backup" / "backup-vault.sh").read_text(encoding="utf-8")
    named = re.search(r'^last_file="\$REPO/([^"]+)"', script, re.M)
    assert named, "backup-vault.sh no longer spells its beat file as $REPO/<name>"

    assert named.group(1) == VW.SNAPSHOT_BEAT_FILE, (named.group(1), VW.SNAPSHOT_BEAT_FILE)
