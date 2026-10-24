"""The SQLite sidecar refusal at the `vault-commit.sh` door (#3236).

`backlog/data/` holds 33 tracked store primaries (`*.db` / `*.sqlite` /
`*.sqlite3`, `kg.sqlite` among the 33) and some of those stores are LIVE:
`ls -l backlog/data/usage.db*` shows a `-wal` and a `-shm` beside the tracked
primary there. A WAL/SHM pair committed next to a copy of the DB it belongs to is
not a second witness, it is a corruption recipe — restoring that pair against any
other copy of the store replays transaction history from the instant the sidecars
were read, not from the page image committed beside them. And the sidecar is never
what the commit was for.

`#3230` closed part of this with a `.gitignore` pair, which stays in force and is
NOT touched here. A pattern list is a hand-list over an open set — the 2026-09-20
class rule — and it fails two ways, both reproduced in the fixture below because
both are what the live vault actually does:

  * suffix coverage. `backlog/data/usage.db-wal` is matched by
    `.gitignore:48`; `backlog/data/usage.db-journal` — the rollback journal the
    non-WAL transaction family leaves — matches nothing, so it is an ordinary
    dirty file that the wrapper's own `git add -A -- "${CHANGED[@]}"`
    (`vault-commit.sh:311`; `:299` until #3236 lifted the `$PY`/`SCRIPT_DIR`
    definitions out of the first rung and up to the top of the file, +12 lines)
    stages. The fixture asserts `git check-ignore` exits 1 for exactly that path,
    which is the positive control that the probe can see coverage: the same probe
    on the `-wal` sibling exits 0.
  * directory coverage. Every pattern is anchored at `backlog/data/`, while
    `git ls-files | grep -E '\\.(db|sqlite|sqlite3)$'` names primaries elsewhere —
    `knowledge/graph.db` — whose companions are unignored too.

Plus the case no ignore file can reach: a path that is already in the index — one
`git add -f` by any writer — is committed whatever the ignore file says. So the
guard reads `git diff --cached` and nothing else.

What the guard is NOT is pinned too. It keys on the companion suffixes only,
never on `*.db` / `*.sqlite` / `*.sqlite3`: a deliberate witness copy of a
store's primary is wanted in the index (that is what #3223 asks for, and 33
witnesses already sit there), so a primary named by the run must still be staged
and committed. And it never refuses the COMMIT — the ruling the wrapper records
above its second pre-flight rung (`vault-commit.sh:481-488`; the contract for this
item cites it as `:412-419`, which is where the same words sat before this change's
+69 lines) for the other three pre-flight rungs: this wrapper also
commits other writers' state, so one job's stray WAL must not strand every later
job's snapshot, which is #2245's ~20 minutes of stranded task #4 commits. The
companion is unstaged (`git reset -q --`, which leaves the bytes on disk —
deleting a live store's WAL would be the destructive version of this fix), named
on stderr with its primary and the `git rm --cached` that would untrack it, and
the rest of the commit lands with exit 0.

ONE MATCHER, TWO DOORS. `scripts/util/sqlite_sidecar_findings.py` is imported by
this wrapper and by `scripts/automod/vault_round.py`'s `land()` — the other door,
which has its own `git add` (`:1876`) and its own `git commit` (`:1962`) — `:1856`
and `:1887` in the tree this item was triaged against — and never
calls the wrapper. #2468 is what guarding one door and calling it coverage costs;
the node that walks the second door lives in `tests/test_automod_vault_round.py`,
next to the land tests it belongs to.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_WRAPPERS = Path(__file__).resolve().parents[1] / "scripts" / "util"
_WRAPPER = _WRAPPERS / "vault-commit.sh"
_MODULE = _WRAPPERS / "sqlite_sidecar_findings.py"

# The pair #3230 shipped in the vault's real .gitignore (lines 48-49 and 68-69),
# copied verbatim so the fixture's ignore coverage matches the live vault's: this
# item's owed half is the REFUSAL, and the ignore file stays exactly as it is.
_VAULT_GITIGNORE = """\
backlog/data/*.db-wal
backlog/data/*.db-shm
backlog/data/*.sqlite*-wal
backlog/data/*.sqlite*-shm
"""


def _load_rung():
    # The matcher is a scripts/util module and `scripts.util` is not a package
    # (no __init__.py), so the import the two doors use is by path.
    spec = importlib.util.spec_from_file_location("sqlite_sidecar_findings", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCF = _load_rung()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


@pytest.fixture()
def vault_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch vault carrying the #3230 ignore pair and a tracked store primary.

    `app.paths.VAULT_DIR` is a property that re-reads `LLOYD_VAULT`, so one env
    var moves the wrapper's default vault — the same lever
    `tests/test_vault_commit_witness_warning.py:59` uses.
    """
    repo = tmp_path / "vault"
    repo.mkdir()
    monkeypatch.setenv("LLOYD_VAULT", str(repo))
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@e")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / ".gitignore").write_text(_VAULT_GITIGNORE, encoding="utf-8")
    (repo / "backlog" / "data").mkdir(parents=True, exist_ok=True)
    (repo / "knowledge").mkdir(parents=True, exist_ok=True)
    (repo / "memory").mkdir(parents=True, exist_ok=True)
    # Tracked in the base commit: a store primary under each of the two
    # directories this matters in. `knowledge/graph.db` is the real shape of the
    # directory-coverage gap — a primary OUTSIDE the ignore pair's scope.
    (repo / "backlog" / "data" / "usage.db").write_bytes(b"PRIMARY-BYTES")
    (repo / "knowledge" / "graph.db").write_bytes(b"PRIMARY-BYTES")
    (repo / "seed.md").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-qm", "base").returncode == 0
    return repo


def _run_wrapper(vault: Path, msg: str, paths: list[str] | None = None,
                 job_id: str = "test: sidecar") -> subprocess.CompletedProcess:
    # `VAULT_DIR` is what the wrapper reads (`vault-commit.sh:129`
    # `VAULT="${VAULT_DIR:-$HOME/obsidian}"`); `LLOYD_VAULT` is what `app.paths`
    # reads. The matcher under test is a python module reached from the bash door, so
    # an env var that moved only one of them would have the wrapper guard a vault
    # this test is not asserting on — and guard the live one, of all things.
    env = dict(os.environ, LLOYD_VAULT=str(vault), VAULT_DIR=str(vault),
               LLOYD_JOB_ID=job_id)
    env.pop("LLOYD_JOB_WRITES", None)
    argv = ["bash", str(_WRAPPER), msg]
    if paths is not None:
        argv.append("--")
        argv.extend(paths)
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)


def _commit_paths(repo: Path) -> list[str]:
    """Paths in HEAD's commit, excluding the pre-test base commit."""
    if _git(repo, "rev-parse", "--verify", "HEAD~1").returncode != 0:
        return []
    out = _git(repo, "show", "--name-only", "--format=", "HEAD").stdout
    return sorted(l for l in out.splitlines() if l.strip())


def _tracked(repo: Path) -> list[str]:
    """What the INDEX holds — `git ls-files` reads the index, which is the right
    place for the force-add assertion: a WAL that `git add -f` put there and the
    guard pulled out must be absent from the index AND absent from HEAD, and
    `_commit_paths` only answers the second half."""
    out = _git(repo, "ls-files").stdout
    return sorted(l for l in out.splitlines() if l.strip())


def _on_disk(repo: Path, rel: str) -> bool:
    return (repo / rel).exists()


def _check_ignore(repo: Path, rel: str) -> int:
    """`git check-ignore -v` exit code: 0 = an ignore pattern covers it, 1 = not.

    The positive control of this file's whole argument, so it is asserted in both
    directions — an ignore probe that cannot detect a covered path would let a
    vacuous fixture pass every refusal here.
    """
    return _git(repo, "check-ignore", "-v", rel).returncode


# ---------------------------------------------------------------------------
# Clause 1, first half: an UNIGNORED companion of a tracked primary, outside
# backlog/data/ — the open set no ignore pattern reaches.
# ---------------------------------------------------------------------------

def test_an_unignored_journal_companion_outside_backlog_data_is_never_committed(vault_repo):
    """`-journal` is the shape `.gitignore` cannot see, at a primary it also
    cannot see — so this is the open-set gap in one assertion."""
    assert _check_ignore(vault_repo, "knowledge/graph.db-journal") == 1, \
        "the fixture must reproduce the live gap: no pattern covers a journal"
    (vault_repo / "knowledge" / "graph.db").write_bytes(b"GROWN-PRIMARY")
    (vault_repo / "knowledge" / "graph.db-journal").write_bytes(b"JOURNAL-BYTES")
    (vault_repo / "knowledge" / "graph.md").write_text("# notes\n", encoding="utf-8")

    proc = _run_wrapper(vault_repo, "job: my snapshot")
    assert proc.returncode == 0, proc.stderr
    committed = _commit_paths(vault_repo)
    assert "knowledge/graph.db-journal" not in committed, committed
    assert "knowledge/graph.db" in committed, committed
    assert "knowledge/graph.md" in committed, committed
    # Named on stderr, with everything a reader needs to act: the path, the
    # primary it belongs to, and the command that would untrack it.
    err = proc.stderr
    assert "knowledge/graph.db-journal" in err, err
    assert "knowledge/graph.db" in err, err
    assert "git rm --cached knowledge/graph.db-journal" in err, err


def test_an_unignored_journal_companion_inside_backlog_data_is_never_committed(vault_repo):
    """Same family, inside the directory the ignore pair DOES scope: the pair
    covers `-wal`/`-shm` there and still covers no `-journal`."""
    assert _check_ignore(vault_repo, "backlog/data/usage.db-wal") == 0
    assert _check_ignore(vault_repo, "backlog/data/usage.db-journal") == 1
    (vault_repo / "backlog" / "data" / "usage.db").write_bytes(b"GROWN-PRIMARY")
    (vault_repo / "backlog" / "data" / "usage.db-journal").write_bytes(b"J")

    proc = _run_wrapper(vault_repo, "job: my snapshot")
    assert proc.returncode == 0, proc.stderr
    committed = _commit_paths(vault_repo)
    assert "backlog/data/usage.db-journal" not in committed, committed
    assert "backlog/data/usage.db" in committed, committed


# ---------------------------------------------------------------------------
# Clause 1, second half: a companion the ignore pair DOES cover, force-added into
# the index by another writer — the case only an index read can see.
# ---------------------------------------------------------------------------

def test_a_force_added_wal_the_ignore_pair_covers_is_dropped_too(vault_repo):
    """Clause 1's second half: the pair covers `-wal`, yet a forced add must not
    land.

    The force-add is left STAGED, not committed: `git add -f` is the hole, and the
    guard reads the index, which is exactly where a force-add lives. Committing it
    and resetting would leave an index the guard has nothing to refuse. A sibling
    edit rides along because clause 2's point is that the REST of the commit lands.
    """
    rel = "backlog/data/usage.db-wal"
    assert _check_ignore(vault_repo, rel) == 0, \
        "fixture must reproduce the covered case: the pair matches wal"
    (vault_repo / "backlog" / "data" / "usage.db-wal").write_bytes(b"WAL-DO-NOT-COMMIT")
    _git(vault_repo, "add", "-f", "--", rel)
    (vault_repo / "seed.md").write_text("seed, edited by this job\n", encoding="utf-8")
    assert _git(vault_repo, "diff", "--cached", "--name-only").stdout.split() == [rel], \
        "the fixture has to hand the guard a forced entry in the index"

    proc = _run_wrapper(vault_repo, "job: my snapshot")
    assert proc.returncode == 0, proc.stderr
    committed = _commit_paths(vault_repo)
    assert rel not in committed, committed
    assert "seed.md" in committed, committed
    assert rel in proc.stderr, proc.stderr
    # Out of the index again, so the ignore rule covers it once more and its own
    # writer still owns it — and untouched on disk: unstaging is not delete.
    assert rel not in _tracked(vault_repo), "a refused force-add must not stay in the index"
    assert (vault_repo / rel).read_bytes() == b"WAL-DO-NOT-COMMIT"

def test_the_matcher_reads_the_index_and_not_the_ignore_file(vault_repo):
    """The mechanism clause 1 pins, both directions in one call.

    One vault, two companions staged the same way: the `-wal` the ignore pair DOES
    cover and the `-journal` it cannot. A guard implemented over `git status`
    (ignore-aware) would drop only the first; one over `git ls-files` alone would
    miss a path added in the same invocation. `check-ignore` is re-asserted inside
    the node so editing the fixture's `.gitignore` cannot silently turn clause 1
    into a test of the ignore file.
    """
    wal, journal = "backlog/data/usage.db-wal", "backlog/data/usage.db-journal"
    assert _check_ignore(vault_repo, wal) == 0
    assert _check_ignore(vault_repo, journal) == 1
    (vault_repo / "backlog" / "data" / "usage.db-wal").write_bytes(b"W")
    (vault_repo / "backlog" / "data" / "usage.db-journal").write_bytes(b"J")
    _git(vault_repo, "add", "-f", "--", wal, journal)
    (vault_repo / "seed.md").write_text("seed, edited by this job\n", encoding="utf-8")

    proc = _run_wrapper(vault_repo, "job: my snapshot")
    assert proc.returncode == 0, proc.stderr
    committed = _commit_paths(vault_repo)
    assert wal not in committed and journal not in committed, committed
    assert "seed.md" in committed, committed
    assert wal in proc.stderr and journal in proc.stderr, proc.stderr
# ---------------------------------------------------------------------------

def test_dropping_a_sidecar_still_commits_the_sibling_file_and_exits_zero(vault_repo):
    """The report-only ruling: this wrapper commits other writers' state, so one
    job's stray WAL must not strand every later job's snapshot (#2245, the
    ruling at `vault-commit.sh:412`)."""
    (vault_repo / "memory" / "notes.md").write_text("# today\n", encoding="utf-8")
    (vault_repo / "backlog" / "data" / "usage.db-shm").write_bytes(b"S")

    proc = _run_wrapper(vault_repo, "job: my snapshot")
    assert proc.returncode == 0, proc.stderr
    assert "memory/notes.md" in _commit_paths(vault_repo), _commit_paths(vault_repo)
    assert "backlog/data/usage.db-shm" not in _commit_paths(vault_repo)
    # The commit is a real commit, not a clean exit: HEAD moved and says what it
    # was for.
    assert _git(vault_repo, "log", "-1", "--format=%s").stdout.startswith("job: my snapshot")


def test_the_unstaged_companion_survives_on_disk_untouched(vault_repo):
    """"Dropped from the commit" must never be read as "deleted": a live store's
    WAL is open by that store, and removing it is the corruption the guard is
    trying to avoid. Bytes identical, file present, and back to untracked."""
    rel = "knowledge/graph.db-wal"
    (vault_repo / rel).write_bytes(b"MUST-NOT-BE-TOUCHED")

    proc = _run_wrapper(vault_repo, "job: my snapshot")
    assert proc.returncode == 0, proc.stderr
    assert (vault_repo / rel).read_bytes() == b"MUST-NOT-BE-TOUCHED"
    status = _git(vault_repo, "status", "--porcelain=v1", "--untracked-files=all").stdout
    assert any(line.endswith(rel) and line.startswith("??") for line in status.splitlines()), \
        f"expected the file left unstaged and untracked, got: {status!r}"
    assert rel not in _git(vault_repo, "diff", "--cached", "--name-only").stdout.splitlines()


# ---------------------------------------------------------------------------
# Clause 3: store primaries stay stageable — the guard keys on companions only.
# ---------------------------------------------------------------------------

def test_a_store_primary_named_by_the_run_is_still_staged_and_committed(vault_repo):
    """The constraint on the fix shape: 33 tracked witnesses and #3223's next one
    are the archive this `git add` EXISTS to make. Each primary here travels with
    its own WAL companion, so a single rule keyed on the primary suffix would hide
    all three witnesses from the very commit meant to archive them."""
    for rel in ("backlog/data/witness.db", "backlog/data/kg.sqlite",
                "backlog/data/graph.sqlite3"):
        (vault_repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault_repo / rel).write_bytes(b"WITNESS")
        (vault_repo / f"{rel}-wal").write_bytes(b"COMPANION")

    proc = _run_wrapper(vault_repo, "job: archive a witness",
                        paths=["backlog/data/witness.db", "backlog/data/witness.db-wal",
                               "backlog/data/kg.sqlite", "backlog/data/kg.sqlite-wal",
                               "backlog/data/graph.sqlite3", "backlog/data/graph.sqlite3-wal"])
    assert proc.returncode == 0, proc.stderr
    committed = _commit_paths(vault_repo)
    for rel in ("backlog/data/witness.db", "backlog/data/kg.sqlite",
                "backlog/data/graph.sqlite3"):
        assert rel in committed, f"witness hidden by the guard: {committed}"
        assert f"{rel}-wal" not in committed, committed


def test_the_matcher_never_refuses_a_store_primary():
    """The matcher-level pin of the same rule: a primary is not companion-shaped,
    whatever it sits next to."""
    primaries = ["backlog/data/usage.db", "backlog/data/kg.sqlite",
                 "backlog/data/graph.sqlite3", "knowledge/graph.db",
                 "backlog/data/lloydmemeval-baseline-2026-09-25.json"]
    assert SCF.companions_in(primaries) == []
    for rel in primaries[:4]:
        assert SCF.is_store_primary(rel), rel
    assert SCF.companion_suffix("backlog/data/usage.db") is None


def test_the_matcher_keys_on_the_three_companion_suffixes_only():
    """`-wal` / `-shm` / `-journal` case-insensitively, and nothing else; the
    refusal names the primary it is derived from, which is what the stderr line
    quotes."""
    assert SCF.COMPANION_SUFFIXES == ("-wal", "-shm", "-journal")
    found = SCF.companions_in([
        "backlog/data/usage.db-wal", "backlog/data/usage.db-WAL",
        "backlog/data/usage.db-shm", "knowledge/graph.db-journal",
        "memory/plain.md", "backlog/data/kg.sqlite"])
    assert [r["path"] for r in found] == [
        "backlog/data/usage.db-wal", "backlog/data/usage.db-WAL",
        "backlog/data/usage.db-shm", "knowledge/graph.db-journal"], found
    assert [r["primary"] for r in found] == [
        "backlog/data/usage.db", "backlog/data/usage.db",
        "backlog/data/usage.db", "knowledge/graph.db"], found
    assert [r["suffix"] for r in found] == ["-wal", "-wal", "-shm", "-journal"], found


# ---------------------------------------------------------------------------
# The CLI the wrapper calls: one subprocess, refuses on stderr, exit is a hint.
# ---------------------------------------------------------------------------

def test_the_cli_refuses_on_stderr_and_its_exit_code_never_decides_the_commit(vault_repo):
    """The wrapper depends on exactly three things from the CLI: the refusal is on
    stderr (its stdout is a transcript), `--unstage` performs the drop, and exit 1
    means only "I found one" — the door commits the rest either way."""
    rel = "backlog/data/kg.sqlite-wal"
    (vault_repo / rel).write_bytes(b"W")
    # `-f`, because the path is ignored: a plain `add` on an ignored path is git
    # refusing, which would test git rather than the guard.
    assert _git(vault_repo, "add", "-f", "--", rel).returncode == 0

    proc = subprocess.run(
        [sys.executable, str(_MODULE), "--vault", str(vault_repo), "--unstage"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1, proc.stderr
    assert rel in proc.stderr, proc.stderr
    assert f"git rm --cached {rel}" in proc.stderr, proc.stderr
    assert proc.stdout == "", f"stdout must stay empty for a caller that pipes: {proc.stdout!r}"
    assert rel not in _git(vault_repo, "diff", "--cached", "--name-only").stdout.splitlines()
    assert _on_disk(vault_repo, rel)

    # A clean index is exit 0 with nothing said: the wrapper runs this on every
    # nightly commit and silence is the normal answer.
    again = subprocess.run(
        [sys.executable, str(_MODULE), "--vault", str(vault_repo), "--unstage"],
        capture_output=True, text=True, timeout=60)
    assert again.returncode == 0, again.stderr
    assert again.stdout == "" and again.stderr == ""

    records = subprocess.run(
        [sys.executable, str(_MODULE), "--vault", str(vault_repo), "--path", rel, "--json"],
        capture_output=True, text=True, timeout=60)
    assert json.loads(records.stdout) == [
        {"path": rel, "suffix": "-wal", "primary": "backlog/data/kg.sqlite"}], records.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))


def test_the_measurement_seam_stands_the_guard_down_and_production_never_sets_it(
        vault_repo, monkeypatch):
    """`LLOYD_VAULT_SIDECAR_REFUSAL=off` is for controls, and nothing on the live
    path sets it.

    #2960's and #3230's controls must SHOW the leak — run the wrapper with the ignore
    lines deleted and require the WAL+SHM in its commit — and a suffix refusal sits in
    front of the `git add` they measure, so they need the guard to stand down. That
    seam is only honest if (a) it changes nothing else about the run and (b) no
    caller can leave it set. `tests/test_witness_artifact_clause.py` is the (a)
    consumer; the second half below is the (b) pin, over the whole repo.
    """
    rel = "backlog/data/usage.db-wal"
    assert _check_ignore(vault_repo, rel) == 0
    (vault_repo / "backlog" / "data" / "usage.db-wal").write_bytes(b"W")
    _git(vault_repo, "add", "-f", "--", rel)
    monkeypatch.setenv("LLOYD_VAULT_SIDECAR_REFUSAL", "off")
    proc = _run_wrapper(vault_repo, "job: a control run")
    assert proc.returncode == 0, proc.stderr
    assert rel in _commit_paths(vault_repo), (
        "with the seam set the wrapper still refused, so a #2960 control could not "
        f"reproduce the leak: {proc.stderr}")
    assert "sidecar REFUSED" not in proc.stderr, proc.stderr

    # Production never sets it: no source file, no config, no autonomy task writes
    # the name. Scoped to the repo's own text so a fixture that merely READS the
    # seam (this file, the #2960 helper) is not mistaken for a setter.
    setters = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "grep", "-l",
        "-e", "LLOYD_VAULT_SIDECAR_REFUSAL=", "-e", 'LLOYD_VAULT_SIDECAR_REFUSAL": ',
        "--", ":!tests/"],
        capture_output=True, text=True)
    assert setters.stdout.strip() == "", (
        f"something outside tests/ sets the seam: {setters.stdout}")
