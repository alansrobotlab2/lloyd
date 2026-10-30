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
reaching an alert with `needs_human=True`, under the heading a person reads — is pinned
here from `test_a_shadows_only_check_heads_itself_as_a_shadow_not_a_writer` onwards
(#3124: a shadow is not a stray, and does not get the code tree's words); the incident
cycle around it, including the clearance that retracts both headings, is in
`tests/test_guardian_predicates.py`.

A schema-current, content-empty duplicate is the SAME hazard at a different byte size, and
the check used to be blind to it (#3481): `KGStore('/home/alansrobotlab/lloyd-data/kg.sqlite')`
run from an agent's Bash heredoc on 2026-10-26 23:47 laid an 81,920-byte file with the whole
shipped schema and a `meta` version row and NOTHING else — 0 rows in `entities`, `aliases`,
`edges` and `facts_idx`, while the declared store it shadows held 178,001 facts. A 0-byte
store answers a count with an error; a provisioned one answers `select count(*)` with `0`,
cleanly, which is the confident false "the KG is empty" of the 2026-08-23/24 incident class.
`empties = sorted((p, d) for d, sz, p in recs if sz == 0)` could not see it, so the check now
measures rows instead of bytes and the alert says which. Nodes from
`test_an_80_kib_schema_only_copy_of_the_declared_store_is_a_shadow` onwards (#3481); the
row probe's own non-mutation and the lock query that keeps a live writer unnamed are pinned
beside them.
"""
from __future__ import annotations

import ast
import os
import shutil
import sqlite3
import stat
import subprocess
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


def test_a_sidecar_beside_a_copy_amends_to_a_lock_and_not_to_its_existence(tmp_path):
    """#3109 clause 4's rule, AMENDED by #3481: existence is not evidence of a writer.

    This node pinned "a `-wal` beside it means a writer has it open", and that inference is
    false on this box: the live `/home/alansrobotlab/lloyd-data/kg.sqlite` has a 0-byte
    `-wal` stamped 2026-10-27 07:32 and a 32,768-byte `-shm` stamped 07:35 — over forty
    hours after the file itself — and no descriptor on the machine held either, because
    SQLite unlinks that pair only when the last connection closes cleanly. Under the old
    rule the file this item exists for stayed invisible. So the rail is now the question it
    was standing in for: does a process HOLD the file, asked of the kernel with `F_GETLK`
    (`_held_by_another_process`), over the candidate and each sidecar.

    The other half of #3109 clause 4 is untouched and still pinned: a database that really
    is mid-write is never named —
    `test_a_process_holding_the_candidate_locked_keeps_it_unreported` and
    `test_a_process_holding_the_wal_locked_keeps_it_unreported` both hold a real `F_SETLK`
    lock in a child process across the check, which is what makes those rails measurable
    rather than declarative.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _store(data / "kg.sqlite", 0)
    _store(data / "kg.sqlite-wal", 0)
    _store(data / SHIPPED_KG, 4096)

    assert _reported(_shadows(data, repo)) == [str(data / "kg.sqlite")], (
        "an unheld `-wal` is residue, and the file beside it is the finding #3481 was "
        "written to stop losing")


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


# ── #3124: what a shadows-only finding says, on the surfaces a person reads ──

def test_a_shadows_only_check_heads_itself_as_a_shadow_not_a_writer(tmp_path, monkeypatch):
    """A check that measured only shadows does not say "code tree" and does not say
    anything "is being written" — on the alert, on ALERT.md, or in the daily note.

    `agent-services/guardian/guardian.py` coalesces this finding and the code-tree stray
    finding into ONE incident under one title, and until #3124 that title was
    `RUNTIME_DATA_ALERT_TITLE` for both. This fixture measures the shadow half alone, so
    both claims in those words were false of it: nothing here is inside `policy.REPO`,
    and the file is 0 bytes, untouched past `INERT_MIN_AGE_SECONDS`, with no `-wal` or
    `-shm` beside it — the exact evidence that no writer has it open. On 2026-10-21 that
    heading toasted hourly from 06:49 to 10:49 PT over four files every one of which was
    under `/home/alansrobotlab/lloyd-data`, and the operator went looking for a writer in
    `~/lloyd`.

    The phrases are asserted as strings, not against a constant: the point of the clause
    is what a reader can no longer be told, and a node that compared the title to
    `RUNTIME_DATA_SHADOW_ALERT_TITLE` alone would keep passing if that constant were
    renamed back to the stray wording. `needs_human` is asserted beside them because it
    is what sends this finding to the board as `[guardian] <title>`
    (`agent-services/guardian/notify.py`), which is the same string one hop further out —
    the filing itself needs a live firing and is recorded as owed on the item.
    """
    import types

    import guardian as gmod
    import notify

    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    shadow = _store(data / "kg.sqlite", 0)
    _store(data / SHIPPED_KG, 4096)

    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True)
    monkeypatch.setattr(gmod.policy, "REPO", str(repo))
    monkeypatch.setattr(gmod.datawatch, "DATA_ROOT", str(data))
    # The tree is empty: this check measured one condition, not two.
    monkeypatch.setattr(gmod.datawatch, "stray_in_tree", lambda _tree: [])

    sent: list[tuple[tuple, dict]] = []
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(vault), backend_url="http://127.0.0.1:1")
    real_alert = n.alert

    def _alert(*args, **kwargs):
        sent.append((args, kwargs))
        return real_alert(*args, **kwargs)

    n.alert = _alert
    n.announce = lambda *a, **k: {}
    g = gmod.Guardian.__new__(gmod.Guardian)
    g.notifier = n
    g.data = types.SimpleNamespace(armed=True)
    g._alert_seen = {}
    g.last_alert = ""

    g._runtime_data_incident(NOW)

    assert len(sent) == 1, f"one check, one incident, whatever it found: {sent}"
    title = sent[0][0][1]
    assert title == gmod.RUNTIME_DATA_SHADOW_ALERT_TITLE, (
        f"a shadows-only check raised under {title!r}")
    low = title.lower()
    assert "code tree" not in low, f"the heading names a tree this check did not find: {title!r}"
    assert "being written" not in low, (
        f"the heading asserts a live writer, and an inert 0-byte stand-in is the "
        f"opposite of one: {title!r}")
    assert sent[0][1].get("needs_human") is True, (
        "an empty stand-in shadowing a populated store is a person's call to delete, "
        "so it still goes to the board (#3109)")
    assert str(shadow) in sent[0][0][2], "the finding itself must survive the retitle"

    alert_md = (tmp_path / "ALERT.md").read_text(encoding="utf-8")
    assert alert_md.splitlines()[0] == f"# {title}", (
        "ALERT.md's H1 is written from the same title and must not carry the retired "
        "wording")
    notes = sorted((vault / "memory").glob("*.md"))
    assert len(notes) == 1, f"expected one daily note, got {[str(m) for m in notes]}"
    daily = notes[0].read_text(encoding="utf-8")
    assert f"## Self-mod guardian: {title}" in daily, daily
    tree_heading = "## Self-mod guardian: Runtime data is being written into the code tree"
    assert tree_heading not in daily, (
        "the daily note still heads a data-root shadow with the wording a stray owns")


# ── #3481: bytes are not rows — the schema-current, content-empty duplicate ──

#: The data tables of the shipped schema, i.e. every table but the `meta` bookkeeping
#: one. Named here so a node can say what "no rows" was measured over.
KG_DATA_TABLES = ("entities", "aliases", "edges", "facts_idx")


def _kg_schema() -> str:
    """The `_SCHEMA` literal of `app/kg_store.py`, read out of the source text.

    Read, not imported: the guardian runs on `/usr/bin/python3` with no venv and cannot
    import `app.*` at all (which is why the layout module is loaded by file path in
    `_repo_declaring`), and a fixture schema typed out here would drift from the shipped
    one the moment a migration landed — the #1525/#2110 rule this file already holds for
    `KG_DB_RELATIVE`. Parsing the assignment means the fixture provisions the database the
    incident actually provisioned, including `meta`, which is the table this item's
    predicate has to look past.
    """
    src = (ROOT / "app" / "kg_store.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "_SCHEMA"):
            return ast.literal_eval(node.value)
    raise AssertionError("app/kg_store.py no longer assigns a _SCHEMA literal")


def _provisioned(path: Path, *, facts_rows: int = 0, pad_rows: int = 0,
                 now: float = OLD) -> Path:
    """A REAL SQLite database carrying the shipped KG schema in WAL mode, aged to `now`.

    What `KGStore(path)` leaves behind: `_SCHEMA` executed, one `meta` row carrying
    `schema_version`, and no data. `pad_rows` inserts that many rows into `facts_idx`
    and deletes them again WITHOUT vacuuming, which is how a database ends up holding
    free pages: bytes, zero rows, and a fully valid file. `facts_rows` leaves its rows
    in place, which is the populated side of the pair.

    **WAL mode, because that is what the shipped store is.** `KGStore._open()` runs
    `PRAGMA journal_mode=WAL` (`app/kg_store.py:228-229`) on the connection it hands out, so
    every database `KGStore(path)` has ever provisioned carries `journal_mode=wal` in its
    header — the live `/home/alansrobotlab/lloyd-data/kg.sqlite` of this incident among them
    (it answers `wal`). This is not decoration: the journal mode decides which read open the
    probe can use, and a rollback-journal fixture made the non-mutation node below
    trivially true of its own fixtures, which is how the review found clause 3 unmet —
    `mode=ro` creates `-wal` and `-shm` beside a WAL-mode file and nothing beside a
    rollback-journal one (measured 2026-10-29 on SQLite 3.54.0 and 3.53.1). Closing the
    last connection checkpoints the log and unlinks both sidecars, so what the fixture
    leaves on disk is a WAL-mode store with no residue beside it: the one shape where the
    WAL-aware open manufactures files that were not there.

    Unlike `_store`, this writes bytes SQLite can open, so the row probe reaches a real
    verdict instead of declining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.executescript(_kg_schema())
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '1') "
                    "ON CONFLICT(key) DO NOTHING")
        for i in range(facts_rows):
            con.execute("INSERT INTO facts_idx(entity, category, text_hash, file_path) "
                        "VALUES (?, 'state', ?, ?)",
                        (f"E{i}", f"h{i}", f"memory/{i}.md"))
        for i in range(pad_rows):
            con.execute("INSERT INTO facts_idx(entity, category, text_hash, file_path) "
                        "VALUES (?, 'state', ?, ?)",
                        (f"P{i}", f"p{i}", f"pad/{i}.md"))
        if pad_rows:
            con.execute("DELETE FROM facts_idx WHERE entity LIKE 'P%'")
        con.commit()
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal", (
            f"{path} is not a WAL-mode store, so every non-mutation claim built on this "
            "fixture is a claim about a shape the shipped store never has")
    finally:
        con.close()
    for side in ("-wal", "-shm"):
        assert not os.path.lexists(str(path) + side), (
            f"{path}{side} survived the close, so the fixture starts with residue and "
            "'the probe created nothing' below would not be this run's claim")
    os.utime(path, (now, now))
    return path


_LOCK_CHILD = (
    "import fcntl, os, struct, sys, time\n"
    "fd = os.open(sys.argv[1], os.O_RDWR)\n"
    "fcntl.fcntl(fd, fcntl.F_SETLK, "
    "struct.pack('hhqqi', fcntl.F_WRLCK, 0, 0, 0, 0))\n"
    "sys.stdout.write('locked\\n')\n"
    "sys.stdout.flush()\n"
    "time.sleep(60)\n")


def _lock_held_by_another_process(path: Path) -> subprocess.Popen:
    """Hold an exclusive POSIX record lock on `path` from a CHILD process.

    A child and not this one, because POSIX record locks are per-process: a lock this
    process takes cannot conflict with this process's own query, so a same-process
    fixture would pass whether or not the probe can see a writer at all. The child is
    the same class of writer the check must leave alone — SQLite takes exactly these
    `F_SETLK` locks on the main file and on its `-shm` index.
    """
    child = subprocess.Popen([sys.executable, "-c", _LOCK_CHILD, str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    first = child.stdout.readline()
    assert first == "locked\n", (
        f"the child never took the lock on {path}: {first!r}")
    return child


def _kill(child: subprocess.Popen) -> None:
    child.kill()
    child.wait(timeout=10)


def test_an_80_kib_schema_only_copy_of_the_declared_store_is_a_shadow(tmp_path):
    """Clause 1, at the live incident's own byte count: 81,920 bytes, 0 rows.

    The stray this item exists for is not 0 bytes — it is a database the shipped schema
    was provisioned into, so `select count(*) from facts_idx` answers `0` instead of
    failing. `empties = sorted((p, d) for d, sz, p in recs if sz == 0)` dropped it before
    the pairing test ever ran, which is why three checks that could have named it on
    2026-10-26 named nothing and this item is the only report it will ever get. The
    fixture is built from `_SCHEMA`, so its size is the shipped schema's size, and the
    predicate under test must be rows and not bytes.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _provisioned(data / "kg.sqlite")
    real = _provisioned(data / SHIPPED_KG, facts_rows=40)

    found = _shadows(data, repo)
    # The live stray is 81,920 bytes; provisioning the same schema from `KGStore` right
    # now lands at 77,824, because the live file's page count also carries a WAL
    # checkpoint and a fill factor this fixture never had. Tens of kilobytes and zero
    # rows is the shape the clause is about; the exact page count is not the predicate,
    # and `shadow_size` below is asserted to be whatever the file actually measures.
    assert os.path.getsize(stray) > 65_536, (
        f"the fixture has collapsed to {os.path.getsize(stray)} bytes, so this node is "
        "no longer about a file the old 0-byte predicate could not see")
    assert _reported(found) == [str(stray)], (
        f"a provisioned 0-row copy of the declared store must be named: {found}")
    assert found[0]["declared"] is True, (
        "the trusted half is the store `KG_DB_RELATIVE` declares, which is what tells "
        "the reader which of the four live kg.sqlite files to count")
    assert found[0]["trusted"] == str(real)
    assert found[0]["data_rows"] == 0, found[0]
    assert found[0]["data_tables"] == len(KG_DATA_TABLES), (
        "the emptiness has to be measured over the four data tables the schema "
        f"provisions, not over nothing: {found[0]}")
    assert found[0]["shadow_size"] == os.path.getsize(stray), (
        "the alert prints this number, so it must be the file's measured bytes and not "
        f"the 0 the old line printed: {found[0]}")


def test_a_larger_zero_row_copy_is_reported_on_the_same_measurement(tmp_path):
    """Clause 1's "whatever its byte size" half, at 4x the byte count.

    The same four tables, still empty, but the file is 327,680 bytes because it once
    held rows that were deleted without a vacuum. Any predicate that reads size as a
    proxy for content gets one finding here and not two, so both files are reported and
    the larger one is not silently demoted to "a store".
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    thin = _provisioned(data / "notes.sqlite")
    fat = _provisioned(data / "sub" / "notes.sqlite", pad_rows=4000)
    _provisioned(data / "_pipeline" / "notes.sqlite", facts_rows=200)
    fat_bytes = os.path.getsize(fat)
    assert fat_bytes > 4 * os.path.getsize(thin), (
        f"the padded fixture is {fat_bytes} bytes against "
        f"{os.path.getsize(thin)}, so this node no longer tests a bigger file")

    found = _shadows(data, repo)
    assert _reported(found) == [str(thin), str(fat)], (
        f"both zero-row copies are shadows; got {found}")
    assert {f["data_rows"] for f in found} == {0}
    assert sorted(f["shadow_size"] for f in found) == sorted(
        [os.path.getsize(thin), fat_bytes])


def test_a_version_row_alone_does_not_make_a_copy_populated(tmp_path):
    """What `meta` is, and why the row count looks past it.

    `KGStore._init_schema` inserts `('schema_version', '1')` as part of PROVISIONING a
    database, so every freshly-created store — including the live stray this item names —
    has exactly one row in `meta` and none anywhere else. Counting it would make the
    predicate unsatisfiable at the exact byte size the incident has, and the check would
    answer "0 findings" about the file it was written for.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _provisioned(data / "kg.sqlite")
    _provisioned(data / SHIPPED_KG, facts_rows=5)
    with sqlite3.connect(str(stray)) as con:
        assert list(con.execute("SELECT key, value FROM meta")) == [
            ("schema_version", "1")], "the fixture lost the provisioning row"
        for table in KG_DATA_TABLES:
            assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0

    assert _reported(_shadows(data, repo)) == [str(stray)]


def test_one_row_in_facts_idx_is_a_populated_copy_and_not_a_shadow(tmp_path):
    """Clause 2: a candidate holding a row is a store, at any depth.

    This is also the false-positive bound on the whole leg. The populated declared store
    shares the basename, so a predicate that misread it would print an instruction to
    delete a 169 MB store of 178,001 facts — and the reverse direction matters just as
    much: the shallow copy below the top is a real store the moment it has a row in it.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _provisioned(data / "kg.sqlite", facts_rows=1)
    _provisioned(data / SHIPPED_KG, facts_rows=40)
    assert _shadows(data, repo) == [], (
        "a copy with one row is a store; naming it is how a check gets switched off")

    data2 = tmp_path / "lloyd-data2"
    _provisioned(data2 / "kg.sqlite", facts_rows=40)
    _provisioned(data2 / "sub" / "kg.sqlite", facts_rows=1)
    assert _shadows(data2, repo) == []


def test_bytes_that_are_not_a_database_are_never_claimed_empty(tmp_path):
    """A probe that cannot read the file cannot prove it empty, so it reports nothing.

    `_store` writes the SQLite magic plus filler: `sqlite3` opens that and answers
    "file is not a database". Answering `0 rows` for bytes the probe could not parse
    would be #2056's unreadable-is-not-empty in a new costume, and here it would be the
    sentence on an alert telling a person to delete the file.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    junk = _store(data / "kg.sqlite", 81_920)
    _provisioned(data / SHIPPED_KG, facts_rows=40)

    assert _shadows(data, repo) == [], (
        f"an unparseable 81,920-byte file was reported as empty: {junk}")


def test_testing_a_candidates_content_mutates_nothing_on_disk(tmp_path):
    """Clause 3: the row probe is a read, provably, per inode and per directory entry.

    The check opens every database-named candidate on every hourly tick, including stores a
    person may still care about, so "it only reads" has to be a measurement and not a
    description of the URI. Both fixtures are WAL-mode, and the journal mode is what makes
    this node able to fail: a `mode=ro` open **creates `-wal` AND `-shm` beside a WAL-mode
    file that had none** (measured 2026-10-29 on `/usr/bin/python3`'s SQLite 3.54.0 and the
    venv's 3.53.1 — sidecars absent before the open, both present after it) and creates
    nothing beside a rollback-journal one. The first version of this node built its fixtures
    in the second shape and the review found clause 3 unmet on it: every claim it made was
    true of files `KGStore` never produces, while the stores it claimed to protect took the
    open that writes. `_provisioned` now sets the shipped store's own journal mode, so the
    fixture reproduces the failure.

    So: both stores keep their bytes and their mtime, no `-wal`, `-shm` or `-journal`
    appears beside either, and the root's directory entries are the ones it started with. A
    manufactured sidecar is worse than litter — the check reads a sidecar as residue to
    explain and asks the kernel who holds it, so inventing one rewrites the next tick's
    reading of the file — and it is still a file the watchdog wrote into a person's data
    root. That is why `_read_open` takes `immutable=1` whenever no log with bytes in it
    stands beside the candidate, and why the WAL-aware open is only used where a `-shm`
    already exists for it to use.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _provisioned(data / "kg.sqlite")
    real = _provisioned(data / SHIPPED_KG, facts_rows=40)
    # That `_provisioned` came back at all is the WAL claim: it asserts
    # `PRAGMA journal_mode` answers `wal` on its own way out, then asserts both sidecars are
    # gone, because a WAL-mode file with a live connection HAS them and the check would
    # legitimately decline to explain a store somebody has open.
    before = {stray: _fingerprint(stray), real: _fingerprint(real)}
    for path in before:
        for suffix in ("-wal", "-shm", "-journal"):
            assert not os.path.lexists(str(path) + suffix), (
                f"{path}{suffix} exists before the check ran, so 'created nothing' "
                "below is not this run's claim")

    found = _shadows(data, repo)

    assert _reported(found) == [str(stray)], "the fixture must hold a real shadow"
    assert found[0]["trusted"] == str(real) and found[0]["declared"] is True, (
        "and the probe has to have READ the populated store to know that: a declined "
        f"declared store would fall through to this assertion anyway: {found[0]}")
    for path, was in before.items():
        assert _fingerprint(path) == was, f"{path} changed under a read-only probe"
        for suffix in ("-wal", "-shm", "-journal"):
            assert not os.path.lexists(str(path) + suffix), (
                f"the probe created {path}{suffix}: the next tick reads that as a store "
                "with a connection of its own, and the data root is not ours to write")
    assert sorted(os.listdir(data)) == ["_pipeline", "kg.sqlite"], (
        "the root holds exactly the entries it started with")


def test_a_wal_log_without_its_index_is_declined_and_never_written_into(
        tmp_path, monkeypatch):
    """The one shape where BOTH opens are unsafe is declined, and both wrong answers are
    reproduced here so the decline is what the node pins.

    A `-wal` holding frames with no `-shm` beside it. `mode=ro&immutable=1` reads the main
    file alone and answers 0 rows for a store whose committed rows are sitting in that log
    (measured 2026-10-29, both SQLite builds: 3 rows committed un-checkpointed, `mode=ro`
    answers 3 and `immutable=1` answers 0) — clause 2's forbidden error, an alert telling a
    person to delete a populated store. `mode=ro` answers correctly but has to BUILD the
    missing index, measured the same day as a 32,768-byte `-shm` appearing where none
    stood: the check writing into the data root to answer a question it asked for its own
    bookkeeping. So `_read_open` returns `None` and the candidate goes unmeasured — the
    safe direction, because an unanswered question is not emptiness (#2056), and the file
    is measured for real on the tick after its log is checkpointed or its index returns.

    Like its sibling node this one puts the WAL copy in the CANDIDATE's seat, which the
    first draft did not: as the declared deeper store it was the trusted pick, the sweep
    skips the trusted pick, and the review measured that the lying open left the findings
    unchanged — the node passed on a store the check never asked about. With the copy
    shallow and the declared deeper store populated in its main file, the two controls
    below each produce their own consequence: the lying open names the populated store for
    deletion, and the WAL-aware open materialises the `-shm` this node says must not exist.

    The writer is SIGKILLed BEFORE the check runs, exactly as in
    `test_a_candidate_store_whose_rows_are_still_in_its_wal_is_not_reported`: a live writer
    would be excluded by the lock query instead, and the decline would never be reached.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    declared = _provisioned(data / SHIPPED_KG, facts_rows=20)
    wal_store = data / "kg.sqlite"          # the CANDIDATE, as in the sibling node
    wal_store.parent.mkdir(parents=True, exist_ok=True)

    child = subprocess.Popen(
        [sys.executable, "-c", _WAL_WRITER, str(wal_store)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        child.stdin.write(_kg_schema())
        child.stdin.close()                # the child reads the schema to EOF: leave no pipe
        assert child.stdout.readline().strip() == "written", "the writer did not commit"
        child.kill()
        child.wait()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert os.path.getsize(str(wal_store) + "-wal") > 0, (
        "the log holds nothing, so the node is no longer about content that only a "
        "WAL-aware open can see")
    os.unlink(str(wal_store) + "-shm")     # the index that connection never finished
    assert not os.path.lexists(str(wal_store) + "-shm")
    os.utime(wal_store, (OLD, OLD))
    os.utime(Path(str(wal_store) + "-wal"), (OLD, OLD))
    assert datawatch._read_open(str(wal_store)) is None, (
        "a frames-bearing log with no index is the shape the decline exists for")

    found = _shadows(data, repo)

    assert _reported(found) == [], (
        f"a declined candidate was reported anyway: {found}")
    assert not os.path.lexists(str(wal_store) + "-shm"), (
        "the check built the missing wal-index to get its answer, writing a file into the "
        "data root it has no right to create")

    # ── control 1: the lying open. It must NAME the store, or the decline above is
    # indistinguishable from the check not looking, which is the defect the review found.
    monkeypatch.setattr(datawatch, "_read_open", lambda path: "mode=ro&immutable=1")
    forced = _shadows(data, repo)
    assert _reported(forced) == [str(wal_store)], (
        "forcing the sidecar-safe open produced no finding, so the decline is not what "
        f"keeps this store out of the alert: {forced}")
    assert forced[0]["data_rows"] == 0, (
        "the control's whole point is the wrong number over a populated file: "
        f"{forced[0]}")
    assert forced[0]["trusted"] == str(declared) and forced[0]["declared"] is True, (
        "and the alert it would print names the WAL copy as the empty half while pointing "
        f"at the declared store as the one to keep: {forced[0]}")
    monkeypatch.undo()

    # ── control 2: the WAL-aware open. It answers the rows correctly AND writes, which is
    # why correctness alone does not license it here.
    monkeypatch.setattr(datawatch, "_read_open", lambda path: "mode=ro")
    _shadows(data, repo)
    assert os.path.lexists(str(wal_store) + "-shm"), (
        "the WAL-aware open did NOT create the missing index, so the decline above was "
        "avoiding a cost this fixture does not actually have and the node's reason is "
        "wrong even though its verdict is right")


def test_a_stale_sidecar_no_process_holds_does_not_hide_the_shadow(tmp_path):
    """The live stray has a `-wal` and a `-shm` beside it, and no process has either.

    `kg.sqlite-wal` is 0 bytes and `kg.sqlite-shm` is 32,768, both stamped 2026-10-27
    07:32 and 07:35 — a day and a half after the file itself, two days before this node
    was written, with no descriptor on either. SQLite allocates those two files to the
    first WAL-mode connection and unlinks them when the last one closes, so a process
    killed mid-session leaves them standing over a store nobody has open. Reading their
    mere existence as "a writer has it open" — which is what the 0-byte class does, and
    keeps doing — would have kept the one finding this item owns unreported, so for a
    provisioned candidate the check asks the kernel who holds the file instead.

    `os.utime` on the sidecars is the only fiction here: the fixture's own bytes and
    sizes are the live ones.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _provisioned(data / "kg.sqlite")
    _provisioned(data / SHIPPED_KG, facts_rows=40)
    (data / "kg.sqlite-wal").write_bytes(b"")
    (data / "kg.sqlite-shm").write_bytes(b"\0" * 32_768)
    for side in ("kg.sqlite-wal", "kg.sqlite-shm"):
        os.utime(data / side, (OLD, OLD))

    found = _shadows(data, repo)
    assert _reported(found) == [str(stray)], (
        "a `-wal`/`-shm` pair that no process holds is residue of a dead connection, "
        f"not a writer: {found}")


def test_a_process_holding_the_candidate_locked_keeps_it_unreported(tmp_path):
    """The other side of that query: a real writer, in another process, is still a rail.

    Nothing about this fixture is mocked — a child process takes the `F_SETLK` write lock
    SQLite takes during a transaction and holds it across the check, which is the process
    boundary the probe is read across. The same scenario for a 0-byte file is pinned by
    `test_a_sidecar_beside_an_empty_copy_means_a_writer_has_it_open`; this is the class
    whose file is provisioned, where existence no longer decides.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _provisioned(data / "kg.sqlite")
    _provisioned(data / SHIPPED_KG, facts_rows=40)

    child = _lock_held_by_another_process(stray)
    try:
        assert _shadows(data, repo) == [], (
            "the check named a database another process is writing to")
    finally:
        _kill(child)

    assert _reported(_shadows(data, repo)) == [str(stray)], (
        "with the lock released the same candidate must be named again, or the rail "
        "above is the fixture failing rather than the probe answering")


def test_a_process_holding_the_wal_locked_keeps_it_unreported(tmp_path):
    """A WAL-mode writer locks the `-shm` index, not the main file — so sidecars are
    probed too, not just the candidate.

    Without this the check would see a free main file and report a store whose write-ahead
    log is mid-append, which is the one case the original existence rule existed for.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _provisioned(data / "kg.sqlite")
    _provisioned(data / SHIPPED_KG, facts_rows=40)
    shm = data / "kg.sqlite-shm"
    shm.write_bytes(b"\0" * 32_768)
    os.utime(shm, (OLD, OLD))

    child = _lock_held_by_another_process(shm)
    try:
        assert _shadows(data, repo) == [], (
            "the check named a store whose WAL index another process holds")
    finally:
        _kill(child)


def test_the_declared_store_is_trusted_whatever_its_byte_size(tmp_path):
    """The probe has no byte bound, and this node is the reason one was never kept.

    A draft of the fix carried an 8 MiB cap on the probe, reasoned from the cost of opening
    big stores. On the live root that cap made the 172 MB declared KG store unreadable to
    the probe, so the pairing fell back to a 24,576-byte witness copy buried under
    `_scratch/` and reported the very stray this item names as `declared: false` — the
    acceptance check's shape, missed. The declared store here is 9 MiB, over that cap and
    over every size a cost bound would plausibly sit at, and it must still be the trusted
    half with `declared: True`.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    stray = _provisioned(data / "kg.sqlite")
    real = _provisioned(data / SHIPPED_KG, facts_rows=1)
    _pad_to_over(real, 8 * 1024 * 1024)
    assert os.path.getsize(real) > 8 * 1024 * 1024, (
        f"the declared store is {os.path.getsize(real)} bytes, so this node no longer "
        "tests a store a byte cap would have had to decline")
    assert os.path.getsize(stray) < 1024 * 1024

    found = _shadows(data, repo)
    assert _reported(found) == [str(stray)], found
    assert found[0]["declared"] is True, (
        "a byte bound that makes the declared store unmeasurable silently re-points the "
        f"alert at some other copy: {found[0]}")
    assert found[0]["trusted"] == str(real)


_WAL_WRITER = (
    "import sqlite3, sys, time\n"
    "con = sqlite3.connect(sys.argv[1])\n"
    "con.execute('PRAGMA journal_mode=WAL')\n"
    "con.execute('PRAGMA wal_autocheckpoint=0')\n"
    "con.executescript(sys.stdin.read())\n"
    "con.execute(\"INSERT INTO facts_idx(entity, category, text_hash, file_path) \"\n"
    "            \"VALUES ('E0', 'state', 'h0', 'memory/0.md')\")\n"
    "con.commit()\n"
    "sys.stdout.write('written\\n'); sys.stdout.flush()\n"
    "time.sleep(60)\n")


def test_a_candidate_store_whose_rows_are_still_in_its_wal_is_not_reported(tmp_path,
                                                                           monkeypatch):
    """Clause 2 against the one open that can lie, on a store the sweep can actually reach.

    `immutable=1` is what keeps the probe from creating sidecars, and its price is that
    SQLite will not consult a `-wal` with it — a database whose committed rows are still
    sitting un-checkpointed in the log reads back as an EMPTY main file under that open
    alone. `_read_open` therefore hands a candidate whose `-wal` holds bytes the WAL-aware
    `mode=ro` open and never offers it `immutable=1`: naming such a file for a human to
    delete is the one error this check could make.

    **The store under test stands in the candidate's seat, which the first draft of this
    node did not.** The review measured that the earlier version's headline assertion could
    not fail for the open it named: the WAL store was the DECLARED deeper copy, the declared
    copy is ordered first as the trusted pick, and the trusted pick is skipped by the
    candidate sweep — so the lying open left the finding list unchanged at `[kg.sqlite]`
    and the node passed on a store it never asked. A WAL-mode store now sits at the top of
    the root as a copy of a name whose declared deeper store is populated in its main file,
    which is the shape that makes the assertion bite: the top-level copy is a candidate, so
    the two opens disagree about whether it is reported at all. The control below forces the
    lying open and shows the finding appearing; the node therefore pins the open choice and
    not merely the check's indifference.

    The fixture is a writer process SIGKILLed after committing with autocheckpoint off,
    which is how the live `kg.sqlite` came to carry sidecars at all: the connection died
    and SQLite never got to checkpoint and unlink them. A `-wal` of hand-written zeros
    would prove nothing, because SQLite ignores a log whose header it cannot parse.
    `test_a_wal_log_without_its_index_is_declined_and_never_written_into` is this node's
    other half: the same log with its `-shm` taken away can only be measured by writing a
    file into the root, so the candidate is declined instead.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    declared = _provisioned(data / SHIPPED_KG, facts_rows=20)
    wal_store = data / "kg.sqlite"          # the CANDIDATE: a copy of the declared name
    wal_store.parent.mkdir(parents=True, exist_ok=True)

    child = subprocess.Popen(
        [sys.executable, "-c", _WAL_WRITER, str(wal_store)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        child.stdin.write(_kg_schema())
        child.stdin.close()                # the child reads the schema to EOF: leave no pipe
        assert child.stdout.readline().strip() == "written", "the writer did not commit"
        assert (data / "kg.sqlite-wal").stat().st_size > 0, (
            "the rows went to the main file, so this node no longer tests a store that "
            "an immutable open would read as empty")
        assert _data_rows_of(str(wal_store)) == 1
        assert _immutable_open_sees_nothing(str(wal_store)), (
            "the premise of this node is that `immutable=1` alone answers 0 rows here; if "
            "the engine changed and applies the log anyway, the node is vacuous")
        child.kill()
        child.wait()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    for f in (wal_store, Path(str(wal_store) + "-wal"), Path(str(wal_store) + "-shm")):
        if f.exists():
            os.utime(f, (OLD, OLD))
    assert os.path.lexists(str(wal_store) + "-shm"), (
        "the index is what lets the WAL-aware open answer without writing, so its absence "
        "here would silently swap in the decline node's shape")
    wal_bytes = (data / "kg.sqlite-wal").stat().st_size
    declared_before = _fingerprint(declared)
    # The branch itself, not just its consequence: a frames-bearing log with its index
    # standing is the one input that must select the WAL-aware open.
    assert datawatch._read_open(str(wal_store)) == "mode=ro"
    assert datawatch._read_open(str(declared)) == "mode=ro&immutable=1", (
        "a store with no log beside it must take the open that creates nothing")

    found = _shadows(data, repo)

    assert _reported(found) == [], (
        f"a populated WAL store was reported as an empty shadow to delete: {found}")
    assert (data / "kg.sqlite-wal").stat().st_size == wal_bytes, (
        "the check checkpointed or truncated a log it was only measuring, which is a write "
        "into somebody's store")
    assert _data_rows_of(str(wal_store)) == 1, (
        "the store no longer holds the row the check declined to report it for")
    assert _fingerprint(declared) == declared_before, (
        "the declared store changed under a read-only probe")

    # ── control: the lying open, forced. The finding MUST appear, or the assertion above
    # is this suite's own vacuity — the exact defect the review found in the first draft.
    monkeypatch.setattr(datawatch, "_read_open", lambda path: "mode=ro&immutable=1")
    forced = _shadows(data, repo)
    assert _reported(forced) == [str(wal_store)], (
        "forcing the sidecar-safe open produced no finding, so the node's headline "
        f"assertion cannot fail and proves nothing about the open: {forced}")
    assert forced[0]["data_rows"] == 0 and forced[0]["shadow_size"] > 0, (
        "the control has to reproduce the harm, not just a finding: an alert reading "
        f"`0 rows` over a store with a row in it: {forced[0]}")
    assert forced[0]["trusted"] == str(declared) and forced[0]["declared"] is True, (
        "and it points at the populated WAL copy as the file to remove while naming the "
        f"real store as the one to keep: {forced[0]}")


def _data_rows_of(path: str) -> int:
    """Rows across the data tables, measured WAL-aware: what the check has to agree with."""
    con = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "select name from main.sqlite_master "
            "where type = 'table' and name not like 'sqlite_%'")
            if r[0] not in datawatch.SHADOW_BOOKKEEPING_TABLES]
        return sum(con.execute('select count(*) from main."%s"'
                               % t.replace('"', '""')).fetchone()[0] for t in tables)
    finally:
        con.close()


def _immutable_open_sees_nothing(path: str) -> bool:
    """Does an `immutable=1` open of `path` answer 0 rows for every data table?"""
    con = sqlite3.connect("file:" + path + "?mode=ro&immutable=1", uri=True)
    try:
        for row in con.execute("select name from main.sqlite_master "
                               "where type = 'table' and name not like 'sqlite_%'"):
            t = row[0]
            if t in datawatch.SHADOW_BOOKKEEPING_TABLES:
                continue
            if con.execute('select exists(select 1 from main."%s")'
                           % t.replace('"', '""')).fetchone()[0]:
                return False
        return True
    finally:
        con.close()


def _pad_to_over(path: Path, target: int) -> None:
    """Grow `path` past `target` bytes with real pages, then leave it closed.

    Deliberately a fill rather than a sparse `truncate`: a hole costs SQLite nothing to
    open and so would not distinguish a real byte bound from a size read.
    """
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE IF NOT EXISTS pad (b BLOB)")
        con.executemany("INSERT INTO pad VALUES (zeroblob(4096))", [()] * 3000)
        con.commit()
    finally:
        con.close()
    assert os.path.getsize(path) > target


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


# ── seam: the engine the guardian runs, which is not the engine this suite runs ──
#
# The review named this as unverified and it is a real boundary: the guardian is a systemd
# unit, `agent-services/systemd/lloyd-guardian.service`, whose interpreter is
# `/usr/bin/python3` (SQLite 3.54.0, `sqlite3.sqlite_version` measured 2026-10-29), while
# every node above runs the venv's Python, whose SQLite is 3.53.1 (same measurement).
# `mode=ro` creating a wal-index and `immutable=1` skipping a log are engine behaviours,
# not Lloyd's, so a node that measures one engine pins its own interpreter. The script
# below is run whole by BOTH and its readings are asserted to agree, which is the claim
# `datawatch._read_open`'s docstring makes when it cites "both builds this box runs".

#: Standalone: stdlib only, argv is a directory to work in, prints one JSON line. It
#: decides nothing about the check — the assertions stay below, in Python, where a failure
#: names the engine and the shape that disagreed.
_ENGINE_PROBE = r'''
import json
import os
import sqlite3
import subprocess
import sys
import time

ROOT = sys.argv[1]
SCHEMA = "CREATE TABLE t (a); CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);"
WRITER = (
    "import sqlite3, sys, time\n"
    "con = sqlite3.connect(sys.argv[1])\n"
    "con.executescript(sys.argv[2])\n"
    "con.execute('PRAGMA journal_mode=WAL')\n"
    "con.execute('PRAGMA wal_autocheckpoint=0')\n"
    "for i in range(3): con.execute('INSERT INTO t VALUES (?)', (i,))\n"
    "con.commit()\n"
    "open(sys.argv[3], 'w').write('ok')\n"
    "time.sleep(30)\n")


def build(path, checkpoint):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    for i in range(3):
        con.execute("INSERT INTO t VALUES (?)", (i,))
    con.commit()
    if checkpoint:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
    con.close()


def run_writer(path):
    ready = path + ".ready"
    proc = subprocess.Popen([sys.executable, "-c", WRITER, path, SCHEMA, ready],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline and not os.path.exists(ready):
        time.sleep(0.05)
    if not os.path.exists(ready):
        proc.kill(); proc.wait()
        raise SystemExit("the writer never signalled its commit")
    proc.kill()
    proc.wait()


def sides(path):
    return sorted(s for s in ("-shm", "-wal") if os.path.lexists(path + s))


def read(path, query):
    before = sides(path)
    con = sqlite3.connect("file:" + path + "?" + query, uri=True)
    rows = con.execute("select count(*) from main.t").fetchone()[0]
    con.close()
    return {"rows": rows, "created": [s for s in sides(path) if s not in before]}


out = {"sqlite_version": sqlite3.sqlite_version, "executable": sys.executable}
build(os.path.join(ROOT, "a.sqlite"), True)
assert sides(os.path.join(ROOT, "a.sqlite")) == [], "fixture left residue beside itself"
out["no_log"] = {"mode_ro": read(os.path.join(ROOT, "a.sqlite"), "mode=ro")}
build(os.path.join(ROOT, "b.sqlite"), True)
out["no_log"]["immutable"] = read(os.path.join(ROOT, "b.sqlite"), "mode=ro&immutable=1")
run_writer(os.path.join(ROOT, "c.sqlite"))
assert "-shm" in sides(os.path.join(ROOT, "c.sqlite")), "no index to use"
assert os.path.getsize(os.path.join(ROOT, "c.sqlite") + "-wal") > 0, "log has no frames"
out["frames_with_index"] = {
    "mode_ro": read(os.path.join(ROOT, "c.sqlite"), "mode=ro")}
run_writer(os.path.join(ROOT, "d.sqlite"))
out["frames_with_index"]["immutable"] = read(
    os.path.join(ROOT, "d.sqlite"), "mode=ro&immutable=1")
run_writer(os.path.join(ROOT, "e.sqlite"))
os.unlink(os.path.join(ROOT, "e.sqlite") + "-shm")
run_writer(os.path.join(ROOT, "f.sqlite"))
os.unlink(os.path.join(ROOT, "f.sqlite") + "-shm")
out["frames_no_index"] = {
    "mode_ro": read(os.path.join(ROOT, "e.sqlite"), "mode=ro"),
    "immutable": read(os.path.join(ROOT, "f.sqlite"), "mode=ro&immutable=1")}
print(json.dumps(out))
'''


@pytest.mark.parametrize("interp", [sys.executable, "/usr/bin/python3"],
                         ids=["this-suites-interpreter", "the-guardians-interpreter"])
def test_the_two_opens_behave_identically_on_the_engine_the_guardian_runs(tmp_path,
                                                                         interp):
    """Every premise `_read_open` chooses between is measured on both engines, not one.

    The decision table this check is built on is SQLite's behaviour, so the suite running
    on one SQLite and the unit running on another means the nodes above pin an interpreter,
    not the box. `/usr/bin/python3` here is the guardian's ExecStart interpreter
    (`agent-services/supervisord.conf`, `[program:lloyd-guardian]`), and the venv's Python
    is what the gate and this file run on.

    What must agree, shape by shape:
      * WAL store, log checkpointed, nothing beside it: `mode=ro` CREATES both sidecars —
        the write clause 3 forbids — and `immutable=1` creates none and still reads all 3
        rows, which is why the empty-log branch is safe.
      * a frames-bearing log with its index standing: `mode=ro` reads the 3 rows and
        creates nothing; `immutable=1` reads 0, the lie clause 2 forbids.
      * the same log with its index gone: `mode=ro` answers 3 rows but creates `-shm`, and
        `immutable=1` lies again — the pair of costs the decline avoids.
    """
    import json as _json

    if not os.path.exists(interp):
        pytest.skip(f"{interp} is not on this box, so this engine's half is unmeasured")
    script = tmp_path / "engine_probe.py"
    script.write_text(_ENGINE_PROBE, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    proc = subprocess.run([interp, str(script), str(work)],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        f"{interp} ({os.uname().sysname}) failed the probe: {proc.stderr[-1200:]}")
    out = _json.loads(proc.stdout.strip().splitlines()[-1])
    who = f"{out['executable']} sqlite {out['sqlite_version']}"

    assert out["no_log"]["mode_ro"]["created"] == ["-shm", "-wal"], (
        f"{who}: `mode=ro` created {out['no_log']['mode_ro']['created']} beside a WAL "
        "store with no log, and the clause-3 fix is keyed on that being the hazard")
    assert out["no_log"]["mode_ro"]["rows"] == 3, who
    assert out["no_log"]["immutable"]["created"] == [], (
        f"{who}: `immutable=1` created {out['no_log']['immutable']['created']}, so the "
        "open this fix routes every plain candidate to is not the non-writing one")
    assert out["no_log"]["immutable"]["rows"] == 3, (
        f"{who}: `immutable=1` read {out['no_log']['immutable']['rows']} of 3 rows from a "
        "checkpointed WAL store whose log was empty, so the empty-log branch would be "
        "reporting populated stores as empty")
    assert out["frames_with_index"]["mode_ro"] == {"rows": 3, "created": []}, (
        f"{who}: the WAL-aware open over a frames-bearing log with its index standing "
        f"answered {out['frames_with_index']['mode_ro']}")
    assert out["frames_with_index"]["immutable"] == {"rows": 0, "created": []}, (
        f"{who}: the lying open stopped lying — answered "
        f"{out['frames_with_index']['immutable']} — which makes the branch this fix added "
        "unmotivated on this engine and the node above vacuous there")
    assert "-shm" in out["frames_no_index"]["mode_ro"]["created"], (
        f"{who}: `mode=ro` did not build the missing wal-index "
        f"({out['frames_no_index']['mode_ro']}), so declining that shape avoids a cost it "
        "does not pay here")
    assert out["frames_no_index"]["mode_ro"]["rows"] == 3, who
    assert out["frames_no_index"]["immutable"]["rows"] == 0, (
        f"{who}: `{out['sqlite_version']}` read the log under immutable=1")


def test_a_quarantined_empty_copy_is_not_a_shadow(tmp_path):
    """A store moved into the quarantine has been ruled on, so it is not reported.

    The 2026-10-26 `kg.sqlite` was moved to `quarantine/tree-strays/<stamp>/` by hand and
    the next scan reported it again from there; `quarantine_inert` moves empty tree-stray
    stores to the same directory, so every one of those would have alerted too.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    _store(data / "quarantine" / "tree-strays" / "2026-10-29-kg-sqlite" / "kg.sqlite", 0)
    _store(data / SHIPPED_KG, 4096)

    assert _shadows(data, repo) == []

    # Positive control: the same empty copy outside the quarantine is still a finding.
    _store(data / "kg.sqlite", 0)
    assert _reported(_shadows(data, repo)) == [str(data / "kg.sqlite")]
