"""A 0-byte database inside the data root whose name is populated elsewhere under the
same root is a shadow, and the guardian must name it (`datawatch.data_root_shadows`).

The live shape this check was written for: `/home/alansrobotlab/lloyd-data/kg.sqlite`
sitting at 0 bytes while the knowledge graph `app.paths.VAULT_KG_DB` resolves to
`_pipeline/vault-derived/kg.sqlite` inside that same root at 160,882,688 bytes, with
`entities 15483 / facts_idx 165971 / edges 51965` readable through `app.kg_store`. A
probe opened the bare name with the data root as its cwd and `sqlite3` created the file
to open it — the same origin as the 2026-10-02 in-tree `workers.db` incident, one root
away from the tree the code-tree sweep watches. 0 bytes is a valid *empty* SQLite
database, so the file answers a count with 0 rather than failing: the false "the KG is
empty / 0 relationships" verdict of the 2026-08-23/24 incident class, printed against a
populated store. The reverse instance is live too — `_pipeline/workers.db` at 0 bytes
beside the populated queue store at the root's top level (`app/paths.py:97`) — which is
why the check runs in both directions.

Everything here is the check's own measurement over a fixture root under `tmp_path`.
Removing the live files is a person's call under `~/lloyd-data` and no clause here
tries it: the check reads and never moves
(`test_running_the_check_moves_and_deletes_nothing`). The alert half — the finding
reaching `RUNTIME_DATA_ALERT_TITLE` with `needs_human=True`, and the clearance that
retracts it — is pinned in `tests/test_guardian_predicates.py`.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datawatch  # noqa: E402
import policy  # noqa: E402

NOW = 1_800_000_000.0
#: Old enough that no writer can still be mid-open on it — the same age floor the
#: code-tree residue rule uses, read from the module rather than restated here.
OLD = NOW - 2 * datawatch.INERT_MIN_AGE_SECONDS
#: The shipped store location, spelled in this file only as the EXPECTATION a test
#: compares against. The code under test must reach it through `KG_DB_RELATIVE`.
SHIPPED_KG = "_pipeline/vault-derived/kg.sqlite"


def _repo_declaring(tmp_path, rel_literal: str) -> Path:
    """A checkout whose `app/data_root.py` declares `KG_DB_RELATIVE` as `rel_literal`.

    The guardian cannot import `app.paths` (no venv, `/usr/bin/python3`, the repo is not
    on `sys.path`), so it reads the data layout by file path through
    `policy._data_root_module` — the route `policy.quarantine_dir` and `policy.kg_db_path`
    already use (#2110, #1525). A fixture repo that wants a store located has to carry
    that file, and the literal it declares is the variable this file turns.
    """
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "data_root.py").write_text(
        "from pathlib import Path\n\n"
        f"KG_DB_RELATIVE = {rel_literal}\n"
        'QUARANTINE_DIR_RELATIVE = Path("quarantine") / "tree-strays"\n',
        encoding="utf-8")
    return repo


def _shipped_layout_repo(tmp_path) -> Path:
    """A checkout carrying the real layout module, copied from the tree under test.

    So the constant these nodes exercise is the one production reads, not a paraphrase
    of it written here.
    """
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    shutil.copy(ROOT / "app" / "data_root.py", repo / "app" / "data_root.py")
    return repo


def _store(path: Path, size: int, *, now: float = OLD) -> Path:
    """Create a database-shaped file of `size` bytes (0 = the empty shape), aged to
    `now`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"SQLite format 3\x00" + b"x" * max(0, size - 16) if size else b"")
    os.utime(path, (now, now))
    return path


def _shadows(data: Path, repo: Path):
    return datawatch.data_root_shadows(str(data), repo=str(repo), now=NOW)


def _reported(findings) -> list[str]:
    return [f["shadow"] for f in findings]


def _fingerprint(path: Path) -> tuple:
    """Everything a moving or rewriting process would change: size, inode, mtime, mode."""
    st = os.lstat(path)
    return (st.st_size, st.st_ino, st.st_mtime, stat.S_IMODE(st.st_mode))


# ── clause 1: the top-level empty copy of a store that lives below it ──

def test_a_zero_byte_kg_at_the_root_is_a_shadow_of_the_store_below_it(tmp_path):
    """The live incident, at fixture scale: `/home/alansrobotlab/lloyd-data/kg.sqlite`
    at 0 bytes while `_pipeline/vault-derived/kg.sqlite` holds the graph.

    `kg.sqlite` is deliberately NOT in `RUNTIME_NAMES` — adding the name there is the
    wrong fix the item rules out, because that list is a prediction of names and the
    property is about depths. The check has to reach this file by looking at the root.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _store(data / "kg.sqlite", 0)
    real = _store(data / SHIPPED_KG, 4096)

    assert "kg.sqlite" not in datawatch.RUNTIME_NAMES, (
        "the check must not be reached by adding the name to the watch list")
    found = _shadows(data, repo)
    assert _reported(found) == [str(stray)]
    assert found[0]["trusted"] == str(real)
    assert found[0]["trusted_size"] == 4096
    assert found[0]["declared"] is True, (
        "kg.sqlite is the one name the repo's own layout module declares, so the "
        "check must say the trusted store came from `KG_DB_RELATIVE` and not from a "
        "guess")


def test_the_populated_half_is_read_from_the_layout_constant(tmp_path):
    """Clause 1's real proof: the trusted path is whatever `KG_DB_RELATIVE` says.

    The fixture repo declares the store at a path no re-spelling could produce
    (`_elsewhere/nested/kg.sqlite`) and the populated file is built there. Code holding
    the shipped `_pipeline/vault-derived/kg.sqlite` literal would find no populated
    counterpart and report nothing; code reading the constant reports the stand-in
    path. Then the constant is moved again and the answer moves with it over an
    unchanged scan — the #1525/#2110 rule this item re-cites: a restated copy rots when
    the layout moves.
    """
    data = tmp_path / "lloyd-data"
    repo = _repo_declaring(tmp_path, 'Path("_elsewhere") / "nested" / "kg.sqlite"')
    stray = _store(data / "kg.sqlite", 0)
    real = _store(data / "_elsewhere" / "nested" / "kg.sqlite", 2048)

    found = _shadows(data, repo)
    assert _reported(found) == [str(stray)]
    assert found[0]["trusted"] == str(real), found[0]
    assert found[0]["declared"] is True

    (repo / "app" / "data_root.py").write_text(
        'from pathlib import Path\n\n'
        'KG_DB_RELATIVE = Path("_moved") / "kg.sqlite"\n', encoding="utf-8")
    moved = _store(data / "_moved" / "kg.sqlite", 2048)

    found = _shadows(data, repo)
    assert found[0]["trusted"] == str(moved), (
        "the answer still names the old location, so it was never read from the "
        f"constant: {found[0]}")


def test_a_shallower_undeclared_copy_does_not_outvote_the_declared_store(tmp_path):
    """The declaration outranks depth AND size, in both fields of the finding.

    The review's probe: declared store at depth 3, a populated `sub/kg.sqlite` at depth 2
    and 8 times the bytes. Depth order alone names `sub/kg.sqlite` — a file no module
    declares — and labels the finding `declared=False`, which is the check telling a human
    to go and count rows in the wrong database while claiming it had not consulted the
    layout module at all. So the declared copy wins whenever it is populated; the sort is
    the fallback the node below pins, for a name nothing declares.
    """
    data = tmp_path / "lloyd-data"
    repo = _repo_declaring(
        tmp_path, 'Path("_pipeline") / "vault-derived" / "kg.sqlite"')
    _store(data / "kg.sqlite", 0)
    _store(data / "sub" / "kg.sqlite", 8192)             # shallower AND larger
    declared_store = _store(
        data / "_pipeline" / "vault-derived" / "kg.sqlite", 1024)

    found = _shadows(data, repo)
    assert _reported(found) == [str(data / "kg.sqlite")], (
        f"two populated copies must still be one shadow pair, not two findings: {found}")
    assert found[0]["trusted"] == str(declared_store), (
        f"the sort outvoted the declaration: named {found[0]['trusted']} at depth "
        f"{found[0]['trusted_depth']}, not the store KG_DB_RELATIVE points at")
    assert found[0]["trusted_size"] == 1024, (
        "the byte count a human will read must be the declared store's, not the "
        "bigger undeclared copy's 8192")
    assert found[0]["declared"] is True
    assert found[0]["declared_relative"] == "_pipeline/vault-derived/kg.sqlite"


def test_an_undeclared_name_is_judged_by_the_scan_alone(tmp_path):
    """No layout constant, no `declared` claim — and still a reported shadow.

    The constant is a precedence, not a gate: a repo whose `app/data_root.py` cannot be
    loaded (a pinned stage one commit behind, as in `test_guardian_inert_stray.py`'s
    None branches) must keep finding shadows on enumeration rather than go blind, or a
    layout-module failure becomes a silent all-clear for the whole check.
    """
    data = tmp_path / "lloyd-data"
    repo = tmp_path / "repo-not-a-checkout"
    repo.mkdir()
    _store(data / "_pipeline" / "vault-derived" / "kg.sqlite", 4096)
    stray = _store(data / "kg.sqlite", 0)

    found = _shadows(data, repo)
    assert _reported(found) == [str(stray)]
    assert found[0]["trusted"] == str(data / SHIPPED_KG)
    assert found[0]["declared"] is False


# ── clause 2: the opposite direction, and a name nobody listed ──

def test_a_zero_byte_copy_below_the_root_is_a_shadow_of_the_store_at_the_top(tmp_path):
    """The live reverse instance: `_pipeline/workers.db` at 0 bytes beside the
    populated queue store at the root's top level (`app/paths.py:97`).

    `inert_residue` cannot reach this shape by construction — it skips any name
    containing `/` — so a nested empty copy stays invisible to the code-tree rule even
    when that rule is handed the data root as its tree.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    real = _store(data / "workers.db", 8192)
    nested = _store(data / "_pipeline" / "workers.db", 0)

    found = _shadows(data, repo)
    assert _reported(found) == [str(nested)]
    assert found[0]["trusted"] == str(real)
    assert found[0]["trusted_size"] == 8192
    assert found[0]["declared"] is False, (
        "the layout module declares kg.sqlite and nothing else; this pair is judged "
        "by the scan alone")


def test_a_name_nobody_predicted_is_reported_too(tmp_path):
    """Clause 2's open-set half: `foo.db` is in no constant in the repository.

    A hand-maintained allowlist cannot close a property over an open set — the standing
    2026-09-20 ruling that makes `KNOWN_GOOD_TOPLEVEL` an exclusion list rather than an
    allowlist. The candidate set here is `os.scandir`, so the only thing that can make
    an unpredicted name invisible is the scan not running.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    assert "foo.db" not in datawatch.RUNTIME_NAMES
    real = _store(data / "_pipeline" / "nested" / "foo.db", 1024)
    empty = _store(data / "foo.db", 0)

    found = _shadows(data, repo)
    assert _reported(found) == [str(empty)]
    assert found[0]["trusted"] == str(real)


# ── clause 3: what is not a shadow ──

def test_two_populated_copies_are_no_shadow(tmp_path):
    """A non-empty file is a store, wherever a second copy of it also lives."""
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _store(data / "kg.sqlite", 4096)
    _store(data / SHIPPED_KG, 4096)

    assert _shadows(data, repo) == []


def test_a_zero_byte_file_with_no_counterpart_is_not_reported(tmp_path):
    """Nothing to trust beside it, so nothing is being shadowed.

    `~/lloyd-data/backlog.db` and `state.db` are this shape live: 0 bytes, no populated
    namesake anywhere, and no constant in `app/paths.py` or `app/data_root.py` that owns
    them. Cruft, and a person's call, but not a wrong-path hazard — and a check that
    cried about both would be trained off inside a week.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _store(data / "orphan.db", 0)
    _store(data / "state.db", 0)
    _store(data / SHIPPED_KG, 4096)

    assert _shadows(data, repo) == []


def test_a_sidecar_beside_an_empty_copy_means_a_writer_has_it_open(tmp_path):
    """A `-wal` is a live handle: the same rule `inert_residue` applies in the tree.

    Naming a database that is mid-open as a shadow to delete is how a check gets
    switched off. It is left for the next tick, which is what the age floor below is
    for as well.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _store(data / "kg.sqlite", 0)
    _store(data / "kg.sqlite-wal", 0)
    _store(data / SHIPPED_KG, 4096)

    assert _shadows(data, repo) == []


def test_a_sidecar_is_never_itself_reported(tmp_path):
    """A 0-byte `-shm` is not a store, and its basename is not a store's name.

    `usage.db-shm` beside a populated `usage.db` is the shape that would read as a
    shadow of `usage.db` to any check matching on a prefix instead of a whole name.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _store(data / "usage.db-shm", 0)
    _store(data / "usage.db-journal", 0)
    _store(data / "usage.db", 4096)

    assert _shadows(data, repo) == []


def test_a_cache_directory_is_not_scanned(tmp_path):
    """`CACHE_DIRS` is the data root's rebuildable-cache set and the check honours it.

    `code-graph` is `app.paths.CODE_GRAPH_DIR`: a nested subvolume the hourly snapshots
    skip for the same reason, and one emptying of which must not trip anything
    (`DataWatch.skip_top`). An empty store inside it is a rebuild in progress.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    assert "code-graph" in datawatch.CACHE_DIRS, (
        "this node names the cache directory by hand; if the constant moved, the node "
        "would pass while testing nothing")
    _store(data / "code-graph" / "kg.sqlite", 0)
    _store(data / SHIPPED_KG, 4096)

    assert _shadows(data, repo) == []


def test_an_empty_store_created_this_minute_is_left_for_the_next_check(tmp_path):
    """The age floor is the tree rule's own constant, not a second number.

    `sqlite3` creates the file on connect and writes the header on first use, so a
    young 0-byte file is indistinguishable from a store being written now.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _store(data / "kg.sqlite", 0, now=NOW - 5)
    _store(data / SHIPPED_KG, 4096)

    assert _shadows(data, repo) == []


def test_a_directory_named_like_a_store_is_not_a_shadow(tmp_path):
    """A directory is not something a probe can open as a database.

    `inert_residue` refuses a directory for the same reason: the empty-file test is
    `S_ISREG` on an `lstat`, not an absence of bytes.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    (data / "kg.sqlite").mkdir(parents=True)
    _store(data / SHIPPED_KG, 4096)

    assert _shadows(data, repo) == []


def test_an_empty_copy_with_a_second_hardlink_is_not_reported(tmp_path):
    """Two names for one inode is the shape `inert_residue` also refuses (`st_nlink`).

    The `.dup` name is not database-shaped, so the scan never sees it — which is what
    makes this a real `st_nlink == 2` file rather than a fixture that quietly tests the
    single-link path twice.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _store(data / "kg.sqlite", 0)
    os.link(stray, data / "kg.sqlite.dup")
    _store(data / SHIPPED_KG, 4096)
    assert os.lstat(stray).st_nlink == 2, "the fixture is no longer a second link"

    assert _shadows(data, repo) == []


# ── clause 4: reading the root must cost the root nothing ──

def test_running_the_check_moves_and_deletes_nothing(tmp_path):
    """The check is report-only, and the difference is measurable per inode.

    `quarantine_inert` moves code-tree residue into `QUARANTINE_DIR_RELATIVE`; the
    item's own reasoning — a data-root deletion reads as data loss and needs an owner —
    says that mechanism must not run against the data root until a person rules
    otherwise. So: both halves of the pair still there, same size, same inode, same
    mtime, same mode, and not one file created at the destination a code-tree move would
    have used.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _store(data / "kg.sqlite", 0)
    real = _store(data / SHIPPED_KG, 4096)
    dest = Path(policy.quarantine_dir(repo=str(repo), data_root=str(data)))
    before = {stray: _fingerprint(stray), real: _fingerprint(real)}
    assert dest.parent.exists() is False, (
        "the fixture must start with no quarantine directory, or 'nothing was created' "
        "is not a claim about this run")

    found = _shadows(data, repo)

    assert _reported(found) == [str(stray)], "the fixture must hold a real shadow"
    for path, was in before.items():
        assert _fingerprint(path) == was, f"{path} changed under a read-only check"
    assert not dest.exists(), f"the check created its code-tree quarantine at {dest}"
    assert sorted(os.listdir(data)) == ["_pipeline", "kg.sqlite"], (
        "the root holds exactly the entries it started with")


def test_a_missing_data_root_is_an_error_and_not_an_empty_result(tmp_path):
    """A root that cannot be read must not answer `[]`.

    The stray incident's #2056 lesson applied to this leg: substituting an empty list
    for a failed measurement is what sealed three open sections in
    `memory/2026-10-02.md` while the same title was still alerting an hour later. The
    guardian catches the raise, logs it, and claims nothing in either direction.
    """
    repo = _shipped_layout_repo(tmp_path)
    with pytest.raises(OSError):
        datawatch.data_root_shadows(str(tmp_path / "no-such-data-root"),
                                    repo=str(repo), now=NOW)
