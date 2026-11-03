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
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
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


# ── #3628: the leg hands over the root it measured, and a foreign seal is labelled ──


def test_a_shadow_finding_sealed_from_another_root_re_alerts_with_the_label(
        tmp_path, monkeypatch):
    """#3628 clause 3: end to end through the check, not through a hand-built write.

    A real shadow finding under `tmp_path/lloyd-data` alerts, a clearance naming a root
    THIS process never measured seals its section, and the next hourly pass of
    `_runtime_data_incident` re-alerts the same finding. Two things are pinned here that
    no node in `tests/test_guardian_predicates.py` can pin: that the shadow leg hands
    the notifier the same `datawatch.DATA_ROOT` string its alert body and its clearance
    already print (so the two are comparable with no new state), and that the second
    section — reached by running the check, never by calling `_vault_note` — carries the
    dated `cross-root seal:` line naming both roots.

    This is the shape of 2026-10-29: the guardian alerted hourly on
    `/home/alansrobotlab/lloyd-data/kg.sqlite` while five automod rounds, each walking
    its own `home/lloyd-data`, sealed the section between the ticks. Nothing in the note
    said so; five sections each read as a fresh incident that had honestly cleared.
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
    monkeypatch.setattr(gmod.datawatch, "stray_in_tree", lambda _tree: [])
    # The arrivals leg of the same hourly check persists a baseline, and a whole pytest
    # session shares one scratch data root; pointed at `tmp_path` it cannot reach the
    # live guardian's state directory (the reason is in `_stray_incident_with`).
    monkeypatch.setattr(gmod.policy, "GUARDIAN_STATE", str(tmp_path / "guardian-state"))
    # `Guardian.alert`'s 900 s repeat guard reads `time.time()` while the check takes
    # `now` as an argument, so the two must come from ONE clock or the second hourly
    # pass is silently suppressed. Started at `NOW` because that is the instant the
    # stores' mtimes were laid relative to: an hour later they are still inert.
    clock = {"t": NOW}
    monkeypatch.setattr(gmod.time, "time", lambda: clock["t"])

    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(vault), backend_url="http://127.0.0.1:1")
    fired: list[dict] = []
    real_vault_note = n._vault_note

    def spy(title, text, *, coalesce=False, measured_root=""):
        fired.append({"title": title, "text": text, "coalesce": coalesce,
                      "measured_root": measured_root})
        return real_vault_note(title, text, coalesce=coalesce,
                               measured_root=measured_root)

    n._vault_note = spy
    n.announce = lambda *a, **k: {}
    g = gmod.Guardian.__new__(gmod.Guardian)
    g.notifier = n
    g.data = types.SimpleNamespace(armed=True)
    g._alert_seen = {}
    g.last_alert = ""

    def check():
        clock["t"] += gmod.policy.STRAY_CHECK_SECONDS
        g._runtime_data_incident(clock["t"])

    check()
    title = gmod.RUNTIME_DATA_SHADOW_ALERT_TITLE
    assert len(fired) == 1, f"one shadow finding, one note write: {fired}"
    assert fired[0]["measured_root"] == str(data), (
        "the shadow leg alerted without handing the notifier the root it measured, so "
        "its body's root and its clearance's root cannot be compared")
    assert str(shadow) in fired[0]["text"]

    foreign = "/home/alansrobotlab/lloyd-work/SM_FAKE0000_000000/home/lloyd-data"
    assert n.resolve(title, (
        f"no runtime stores inside the code tree of {repo}, and no database with zero "
        f"data rows shadowing a populated store under {foreign} — absent with no move "
        "recorded by the guardian")) is True

    check()
    daily = sorted((vault / "memory").glob("*.md"))
    assert len(daily) == 1, daily
    text = daily[0].read_text(encoding="utf-8")
    assert len(fired) == 2, (
        f"the foreign seal closed the section but the next check did not re-alert:\n{text}")
    assert fired[1]["measured_root"] == str(data), fired[1]
    secs = [text[a:b] for (t, a, b) in notify.Notifier._daily_sections(text)
            if t == title]
    assert len(secs) == 2, f"one section, so the foreign seal was never honoured:\n{text}"
    seals = [ln for ln in secs[1].splitlines()
             if ln.startswith(notify.DAILY_CROSS_ROOT_PREFIX)]
    assert len(seals) == 1, f"expected one label in the new section:\n{secs[1]}"
    assert foreign in seals[0] and str(data) in seals[0], seals[0]
    # The label is detection only: the finding, its root and the ask are all still the
    # body's own words, unchanged by the label beside them.
    assert str(shadow) in secs[1], secs[1]
    assert notify.DAILY_CROSS_ROOT_PREFIX not in secs[0], secs[0]


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


# ── #3577: the read the shadow paragraph names must not be able to touch the store ──

#: A writer that checkpoints its log away and then holds the file open, so the
#: store is left in the shape `_read_open` picks the immutable branch for: WAL mode,
#: a `-wal` of NO bytes, and a `-shm` index. `WRITER` above never checkpoints, so it
#: leaves bytes in the log — the other branch, where an immutable read is refused.
CHECKPOINTED_WRITER = r"""
import os, sqlite3, sys, time
path, schema, ready = sys.argv[1], sys.argv[2], sys.argv[3]
con = sqlite3.connect(path)
con.executescript(schema)
con.execute("PRAGMA journal_mode=WAL")
con.execute("PRAGMA wal_autocheckpoint=0")
con.execute("INSERT INTO meta(key,value) VALUES ('schema_version','1')")
con.commit()
con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
con.commit()
open(ready, "w").write("written")
time.sleep(60)
"""


def _checkpointed_store(path: Path, schema: str, ready: Path) -> subprocess.Popen:
    """Start `CHECKPOINTED_WRITER`, wait for its commit, SIGKILL it mid-hold.

    Killed rather than closed for the same reason `run_writer` kills: a clean close
    checkpoints and unlinks the sidecars, and the shape under test is the one left
    behind by a process that did not — which is how the live `kg.sqlite` was found.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", CHECKPOINTED_WRITER, str(path), schema, str(ready)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline and not ready.exists():
        time.sleep(0.05)
    proc.kill()
    proc.wait()
    if proc.returncode >= 0:
        raise AssertionError("the checkpointing writer exited on its own before the kill")
    return proc


def test_the_immutable_read_the_paragraph_names_answers_zero_and_touches_nothing(
        tmp_path):
    """The instruction #3577 ships, executed against the bytes it would be run on.

    Clause 4, both halves, over a real WAL store with a `-wal` of no bytes and a
    `-shm` index — the live `kg.sqlite` shape, and the one case
    `datawatch._read_open` answers with the immutable URI
    (`agent-services/guardian/datawatch.py:719-721`, read at them).

    1. the query string is taken OUT OF THE RENDERED PARAGRAPH, not retyped here, so
       the node fails if the paragraph ever names a query that does not work; and
    2. it answers 0 for every data table while leaving BOTH sidecar mtimes and sizes
       exactly where they were. The mutating form is run beside it on the same
       shape as the control: `mode=ro` — what `sqlite3 -readonly` is implemented
       with — re-stamps the `-shm` and does not the `-wal`, which is the #3559
       signature the quarantined witness carries.
    """
    import guardian as gmod

    data = tmp_path / "lloyd-data"
    data.mkdir()
    repo = _repo_declaring(tmp_path, SHIPPED_KG.rsplit("/", 1)[0])
    schema = _kg_schema()
    shadow = data / "kg.sqlite"
    _provisioned(data / SHIPPED_KG, facts_rows=5)      # the trusted half
    _checkpointed_store(shadow, schema, data / "ready")

    wal = shadow.with_name(shadow.name + "-wal")
    shm = shadow.with_name(shadow.name + "-shm")
    assert wal.exists() and wal.stat().st_size == 0 and shm.exists(), (
        "the fixture is not the zero-byte-log WAL shape the immutable branch is "
        f"chosen for: wal={'absent' if not wal.exists() else wal.stat().st_size} "
        f"shm={'absent' if not shm.exists() else 'present'}")
    # Age the store and both sidecars past the check's `age_min`: a young empty
    # file is indistinguishable from one being provisioned right now, so the age
    # is what makes this a finding at all rather than a skip.
    for p in (shadow, wal, shm):
        os.utime(p, (OLD, OLD))

    finding, = datawatch.data_root_shadows(str(data), repo=str(repo), now=NOW)
    body = gmod._shadow_alert_body(str(data), [finding], NOW)

    found = re.findall(r"`(mode=[A-Za-z0-9&;=_]+)`", body)
    assert len(found) == 1, (
        f"the paragraph must name exactly one query string for a reader to paste, "
        f"and this node has to know which one to run; found {found} in:\n{body}")
    query = found[0]
    assert query == "mode=ro&immutable=1", (
        f"the read the paragraph names drifted from the one `_read_open` picks "
        f"({datawatch._read_open}): {query}")

    def answer(q: str):
        before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (wal, shm)}
        con = sqlite3.connect(f"file:{shadow}?{q}", uri=True)
        try:
            tables = [r[0] for r in con.execute(
                "select name from main.sqlite_master where type='table' "
                "and name not like 'sqlite_%'")]
            rows = {t: con.execute(f'select count(*) from main."{t}"').fetchone()[0]
                    for t in tables}
        finally:
            con.close()
        after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (wal, shm)}
        return rows, before, after

    rows, before, after = answer(query)
    assert rows["meta"] == 1 and all(
        rows[t] == 0 for t in KG_DATA_TABLES if t in rows), (
        f"the immutable read did not reproduce what the check measured "
        f"(0 rows over the data tables, `meta`'s version row past it): {rows}")
    assert before == after, (
        "the read the paragraph NAMES moved a sidecar, so the paragraph is the "
        f"#3559 bug again: {(before[wal], after[wal])} {(before[shm], after[shm])}")

    # The control, on the same bytes: the ordinary read-only open — which is what
    # `sqlite3 -readonly` opens with — re-stamps the -shm and leaves the -wal alone.
    _, c_before, c_after = answer("mode=ro")
    assert c_before[shm] != c_after[shm], (
        "the CONTROL did not re-stamp the -shm, so this node can no longer tell an "
        "immutable open from a mutating one and the assertion above proves nothing")
    assert c_before[wal] == c_after[wal], (
        "the control moved the -wal too, which is a different mutation than the one "
        "this item is about and the paragraph's `why` would need re-deriving")
    # "Re-stamped", not "grew": the fixture is aged to a stamp AHEAD of this box's
    # clock (`NOW` is a fixed future epoch), so the new stamp is numerically lower
    # and the only honest claim is that the bytes are identical and the mtime is not.
    assert c_after[shm][1] == c_before[shm][1] and c_after[shm][0] != c_before[shm][0], (
        f"the -shm changed in SIZE rather than being re-stamped, so this is not the "
        f"#3559 signature (same 32,768 bytes, a new mtime): "
        f"{c_before[shm]} -> {c_after[shm]}")


# ── #3584: does the quarantine's ledger account for what stands in the quarantine? ──
#
# `quarantine_inert` appends one JSONL row per move into `log.jsonl` BESIDE the payload it
# moved, and that row is the only statement in this repository about where a quarantined
# file came from: "the quarantine directory is its own audit and not just a pile of empty
# files" is a property of the mover's loop, not of the directory. A payload that gets there
# by any other route — a session's `mv`, a person clearing a root by hand — is bytes with no
# provenance, and on 2026-10-29 ALERT.md had to say exactly that. The directory on this box
# is in that state right now: `~/lloyd-data/quarantine/tree-strays/` holds
# `2026-10-29-kg-sqlite/` with an 81,920-byte `kg.sqlite`, a 32,768-byte `-shm` and a 0-byte
# `-wal`, while `find ~/lloyd-data/quarantine -maxdepth 3 -name log.jsonl` answers nothing.
#
# The mover did not make that payload and its own shapes are the proof: a destination of
# `{%Y%m%dT%H%M%SZ}-{name}`, FLAT, of exactly 0 bytes. So the reconciliation has to name a
# DIRECTORY payload, and name it as ONE entry — the one shape `top_level_regular_files`,
# this module's other enumeration, filters out by design ("No name filter of any kind", but
# regular files only, #3210). Report-only throughout, inheriting #3109 for the data root and
# #2110 for the checkout half: the guardian cannot know whether a payload someone set aside
# by hand is theirs, so nothing in these nodes may move, rename or remove anything.


def _quarantine_of(tmp_path, *, create: bool = True):
    """(data root, quarantine dir) for a checkout carrying the shipped layout module.

    The address comes from `policy.quarantine_dir`, the same route the guardian's hourly
    leg uses, so these nodes fail if the reconciliation is ever pointed at a directory the
    mover does not write to — two quarantine directories, one of them checked, is the
    failure #2110 names for the candidate roots.
    """
    data = tmp_path / "lloyd-data"
    repo = _shipped_layout_repo(tmp_path)
    qdir = Path(policy.quarantine_dir(repo=str(repo), data_root=str(data)))
    assert qdir.parts[-2:] == ("quarantine", "tree-strays"), (
        f"the fixture's address must be the mover's own: {qdir}")
    if create:
        qdir.mkdir(parents=True, exist_ok=True)
    return data, qdir


def test_a_payload_standing_in_the_quarantine_with_no_ledger_is_named_whole(tmp_path):
    """Clause 1: no ledger accounts for nothing, and a directory is one payload.

    The fixture is the live one, transcribed: a stamped DIRECTORY holding a store and its
    two sidecars, plus a flat 0-byte file of the shape the mover does make, and no
    `log.jsonl`. Both are named; the sidecars are not named alongside their container,
    because three lines about one payload is three lines that contradict each other the
    hour someone deletes the directory, and the container is the path a person acts on.

    Read-only is measured, not asserted in prose: every payload file keeps its size, inode,
    mtime and mode, and the directory's own listing is what it was — including no ledger,
    since a check that "fixed" the missing file by creating one would be this leg writing
    the data root, which is the thing the item's own contract forbids.
    """
    data, qdir = _quarantine_of(tmp_path)
    live = qdir / "2026-10-29-kg-sqlite"
    _store(live / "kg.sqlite", 81_920)
    _store(live / "kg.sqlite-shm", 32_768)
    _store(live / "kg.sqlite-wal", 0)
    flat = qdir / "20261029T183102Z-usage.sqlite"
    flat.write_bytes(b"")
    before = {p: _fingerprint(p) for p in sorted(live.iterdir())}
    listing = sorted(os.listdir(qdir))

    found = datawatch.quarantine_ledger_gaps(str(qdir))

    assert [g["name"] for g in found] == ["2026-10-29-kg-sqlite", flat.name], (
        f"with no ledger, every standing entry is unaccounted for: {found}")
    assert found[0]["directory"] is True, (
        "the mover only ever makes a flat file, so the word 'directory' is what tells the "
        f"reader something other than the guardian put this here: {found[0]}")
    assert found[1]["directory"] is False, f"a flat payload is not a directory: {found[1]}"
    assert Path(found[0]["path"]) == live, (
        "the report must carry the path a person looks in, not the bare name: "
        f"{found[0]['path']}")
    assert sorted(os.listdir(qdir)) == listing, (
        "naming a payload must not create, move or remove anything in the quarantine")
    for path, was in before.items():
        assert _fingerprint(path) == was, f"{path} changed under a read-only check"
    assert not (qdir / "log.jsonl").exists(), (
        "the check that reports a missing ledger must not answer by writing one")


def test_a_ledger_with_no_rows_in_it_accounts_for_no_payload(tmp_path):
    """Clause 1's other half: an empty ledger and no ledger say the same true thing.

    Worth a node because the two look different on disk and the difference is where a
    lenient reader would go wrong: `FileNotFoundError` is excused (a quarantine that has
    never seen a move has no file, which is not a failure), while an existing file with no
    destination rows is a ledger that has looked at nothing — and answering "everything is
    accounted for" to that would clear the exact payload this item is about.
    """
    data, qdir = _quarantine_of(tmp_path)
    (qdir / "2026-11-01-sessions").mkdir()
    (qdir / "stray.json").write_text("{}\n", encoding="utf-8")

    empty = qdir / "log.jsonl"
    empty.write_text("", encoding="utf-8")
    assert [g["name"] for g in datawatch.quarantine_ledger_gaps(str(qdir))] == [
        "2026-11-01-sessions", "stray.json"], (
        "a ledger of zero rows accounts for zero payloads")

    # The writer terminates every row with `\n`, so a healthy ledger's last read line is
    # empty. Treating that as corruption would make the check raise on its own happy path.
    empty.write_text("\n", encoding="utf-8")
    assert len(datawatch.quarantine_ledger_gaps(str(qdir))) == 2, (
        "the trailing newline of a well-formed ledger is not a corrupt line")

    empty.write_text('{"destination": "' + str(qdir / "stray.json") + '"}\n',
                     encoding="utf-8")
    assert [g["name"] for g in datawatch.quarantine_ledger_gaps(str(qdir))] == [
        "2026-11-01-sessions"], (
        "one real destination row clears exactly the payload it names")


def test_what_the_stray_leg_moved_and_journalled_is_not_reported(tmp_path):
    """Clause 2: the reconciliation cannot fire on the mover's own work.

    Both the payload and the ledger row are PRODUCED here by `quarantine_inert` rather than
    typed in, so the node breaks the day the two disagree about the destination's spelling
    or the row's field names — which is the only way this pair can silently drift apart,
    since nothing else in the tree reads the ledger. A check that alerted on every payload
    the guardian itself moved would be noise within an hour, and noise is how a report-only
    leg gets switched off.

    A quarantine directory that does not exist is the other silent case, and it is silent
    by REFUSAL rather than by answering empty: this fixture never creates one, because the
    mover creates its own lazily, and `quarantine_ledger_gaps` raises for it the way
    `data_root_shadows` raises for a missing root. An empty list is the answer to "every
    payload here has a row", and a directory that was never created has not been
    reconciled — so the caller that catches the raise makes no statement, which is what a
    machine that has never moved anything should hear back.
    """
    tree = _shipped_layout_repo(tmp_path)
    data = tmp_path / "lloyd-data"
    data.mkdir()
    (data / "workers.db").write_bytes(b"SQLite format 3\x00" + b"x" * 100)
    stray = tree / "workers.db"
    stray.write_bytes(b"")
    os.utime(stray, (OLD, OLD))
    never = Path(policy.quarantine_dir(repo=str(tree), data_root=str(data)))
    assert not never.exists()
    try:
        datawatch.quarantine_ledger_gaps(str(never))
        raise AssertionError("a quarantine directory that does not exist was reconciled")
    except FileNotFoundError:
        pass

    moved = datawatch.quarantine_inert(str(tree), ["workers.db"], str(data), NOW)
    assert len(moved) == 1, "the fixture must produce a real moved payload and a real row"
    qdir = Path(moved[0][1]).parent
    assert (qdir / "log.jsonl").is_file(), "the mover writes its own ledger"
    assert datawatch.quarantine_ledger_gaps(str(qdir)) == [], (
        "the guardian must not be alerted about a payload the guardian moved")


def test_a_ledger_that_names_files_inside_a_directory_payload_accounts_for_it(tmp_path):
    """Clause 2 read against a directory: the FIRST SEGMENT under the quarantine matches.

    `quarantine_inert`'s destinations are flat, so for its own moves a basename and a
    first-segment are the same string and this node would pass either way. It exists for
    the payload the live box actually has: someone sets a store aside as a directory, and
    a ledger row per file inside it. Matching basenames would report `2026-10-29-kg-sqlite`
    as unaccounted for while its ledger listed all three of its files — a false alarm about
    a payload that is fully documented, which is precisely the kind that gets the leg
    muted. A row whose destination is somewhere OUTSIDE the quarantine still accounts for
    nothing here, so the negative control is in the same node.
    """
    data, qdir = _quarantine_of(tmp_path)
    payload = qdir / "hand-set-aside"
    _store(payload / "kg.sqlite", 4096)

    def _row(dest: Path) -> str:
        import json as _json
        return _json.dumps({"source": "/somewhere/kg.sqlite",
                            "destination": str(dest), "size": 4096}) + "\n"

    (qdir / "log.jsonl").write_text(_row(payload / "kg.sqlite"), encoding="utf-8")
    assert datawatch.quarantine_ledger_gaps(str(qdir)) == [], (
        "a ledger that names a payload's inner file documents that payload")

    (qdir / "log.jsonl").write_text(_row(qdir / "a-different-payload"), encoding="utf-8")
    assert [g["name"] for g in datawatch.quarantine_ledger_gaps(str(qdir))] == [
        "hand-set-aside"], (
        "a row about another payload cannot clear this one")

    (qdir / "log.jsonl").write_text(
        _row(Path("/home/alansrobotlab/lloyd-data/elsewhere/kg.sqlite")), encoding="utf-8")
    assert [g["name"] for g in datawatch.quarantine_ledger_gaps(str(qdir))] == [
        "hand-set-aside"], (
        "a destination outside the quarantine accounts for nothing standing inside it")


def test_an_unaccounted_payload_is_reported_and_the_tick_decides_no_differently(
        tmp_path, monkeypatch):
    """Clause 3: the report is the whole action, proved by differencing two ticks.

    The same guardian, the same clock, the same data root, run twice — once over an empty
    quarantine, once over one holding a payload its ledger cannot account for — with every
    alert and every retraction recorded. The difference between the two runs must be one
    `alert` and nothing else: one extra heading, `needs_human=True` on it, `resolve` called
    identically the other way, the method still returning `None`, and the arm flag
    untouched. That is what "reaches the same arm/liveness decision" means when it is
    measured rather than asserted, and a second branch in the tick that suppressed an alert
    or armed the mover would land here.

    The mover is patched to a stub that fails the node if it is reached: `stray_in_tree`
    already returns nothing so the stray leg cannot fire, and the stub is what proves it
    for the payload under test rather than for the fixture's shape. The clean baseline is
    measured on the tick AFTER the first, because the first pass records a name set and
    compares it against nothing — a retraction it is not entitled to make — so comparing a
    first pass with a comparing pass would difference the two legs' arming rules and not
    the presence of a payload.
    """
    import types

    import guardian as gmod
    import notify

    def _move_never(*a, **k):
        raise AssertionError("a quarantine payload with no ledger row must not be moved, "
                             "re-moved or tidied by the check that reports it")

    data, qdir = _quarantine_of(tmp_path)
    repo = tmp_path / "repo"
    monkeypatch.setattr(gmod.policy, "REPO", str(repo))
    monkeypatch.setattr(gmod.datawatch, "DATA_ROOT", str(data))
    monkeypatch.setattr(gmod.policy, "GUARDIAN_STATE", str(tmp_path / "gstate"))
    monkeypatch.setattr(gmod.datawatch, "stray_in_tree", lambda _tree: [])
    monkeypatch.setattr(gmod.datawatch, "quarantine_inert", _move_never)

    def _run_tick():
        sent: list[tuple] = []
        sealed: list[str] = []
        vault = tmp_path / "obsidian"
        (vault / "memory").mkdir(parents=True, exist_ok=True)
        n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                            vault_root=str(vault), backend_url="http://127.0.0.1:1")
        n.alert = lambda *a, **k: sent.append((a, k))
        n.resolve = lambda title, *a, **k: sealed.append(title)
        n.announce = lambda *a, **k: {}
        g = gmod.Guardian.__new__(gmod.Guardian)
        g.notifier = n
        g.data = types.SimpleNamespace(armed=True)
        g._alert_seen = {}
        g.last_alert = ""
        assert g._runtime_data_incident(NOW) is None, (
            "this check reports through alerts; a return value would be a decision")
        assert g.data.armed is True, "the tick that found a payload also un-armed itself"
        return sent, sealed

    # TWO clean ticks, deliberately: the first records the arrivals baseline and compares
    # it against nothing, so only the second is in the same measuring-against-a-baseline
    # state the dirty tick will be in. Differencing a first pass against a comparing pass
    # would difference the two arrivals legs' arming rules, not the presence of a payload.
    _run_tick()
    clean_sent, clean_sealed = _run_tick()
    assert clean_sent == [], (
        f"an empty quarantine is not an incident: {clean_sent}")
    assert gmod.RUNTIME_DATA_QUARANTINE_LEDGER_ALERT_TITLE in clean_sealed, (
        "the clean tick has to REACH the ledger check and seal it, or the comparison below "
        f"is describing a leg that never ran: {clean_sealed}")

    payload = qdir / "2026-10-29-kg-sqlite"
    _store(payload / "kg.sqlite", 81_920)
    _store(payload / "kg.sqlite-wal", 0)
    _store(payload / "kg.sqlite-shm", 32_768)
    before = {p: _fingerprint(p) for p in sorted(payload.iterdir())}
    listing = sorted(os.listdir(qdir))

    gap_sent, gap_sealed = _run_tick()

    assert len(gap_sent) == 1, (
        f"one unaccounted payload, one alert, no other statement: {gap_sent}")
    (level, title, body), kwargs = gap_sent[0]
    assert title == gmod.RUNTIME_DATA_QUARANTINE_LEDGER_ALERT_TITLE, (
        f"the finding needs its own heading, not a shared one (#3124): {title!r}")
    assert kwargs["needs_human"] is True, (
        f"a quarantined payload is a person's file to explain: {kwargs}")
    assert kwargs.get("quarantine") is None and kwargs.get("staged") is None, (
        f"the alert must offer the guardian nothing to do: {sorted(kwargs)}")
    assert str(payload) in body, (
        f"the body names the payload by path, the thing the bare name cannot say: {body}")
    assert "directory" in body, (
        f"a directory payload is labelled as one, because the mover never makes one: {body}")
    # The two ticks' whole difference, enumerated: the ledger check's retraction on the
    # clean run is its alert on the dirty one, and no other section of this hourly check —
    # shadow, arrivals, departures, strays — says anything differently. Read this way
    # (symmetric difference, not "the dirty run also sealed X") because a leg that armed
    # differently, moved something, or suppressed a retraction elsewhere would show up as a
    # second element on one side, which is exactly what clause 3 forbids.
    assert set(clean_sealed) ^ set(gap_sealed) == {
        gmod.RUNTIME_DATA_QUARANTINE_LEDGER_ALERT_TITLE}, (
        f"the two ticks differ ONLY by the ledger check's own line — its retraction on the "
        f"clean run becomes its alert on the dirty one, and the shadow, arrivals and "
        f"departure statements are identical: clean={clean_sealed} dirty={gap_sealed}")
    assert gmod.RUNTIME_DATA_QUARANTINE_LEDGER_ALERT_TITLE not in gap_sealed, (
        "a tick that alerts about a payload must not seal the section it alerted into")

    for path, was in before.items():
        assert _fingerprint(path) == was, f"{path} changed under a report-only tick"
    assert sorted(os.listdir(qdir)) == listing, (
        "the tick created nothing in the quarantine either, including no ledger")


# ── #3584 clause 4: the arrivals journal records departures as well as arrivals ──
#
# The state file `data-root-arrivals.json` persisted `{root, seen, updated}` and nothing
# else, so a name that left the data root's top level was silently dropped on the next
# write and no later reader could say which tick last saw it — the same event as the
# unaccounted quarantine payload above, seen from the side that still has the clock.
#
# The addition must not disturb the key that the leg's whole noise rule rests on. `seen` is
# MONOTONIC on purpose: `usage.db-wal` is removed at a checkpoint and recreated by the next
# writer, and a snapshot-per-pass would report that every hour forever. So a departure goes
# INTO `departed` and the name STAYS in `seen`. The only route that removes a name from
# `seen` remains #3422's prune, because a prune is a person saying the name is gone.


def test_a_name_that_left_the_root_is_journalled_once_and_stays_counted_as_seen(tmp_path):
    """Clause 4: `departed` beside `root`/`seen`/`updated`, reported on one tick only.

    Three passes over the same root, the second of which has one name missing. The finding
    is asserted three ways: in the return value (what the tick alerts on), in the persisted
    file (what a person reads afterwards), and in the third pass's silence (the once-only
    rule `new_top_level_names` states for arrivals, read on the departure side — a
    departure that re-alerts hourly is an alert that gets switched off).

    And `seen` is asserted to STILL hold the departed name. That is the clause's sharp
    edge: the intuitive implementation removes it, and the hourly WAL checkpoint would then
    become a recurring alert about a file nobody deleted.
    """
    data = tmp_path / "lloyd-data"
    data.mkdir()
    state = tmp_path / "gstate"
    first = {"sessions": OLD, "workers.db": OLD, "scratch-notes.txt": OLD}
    assert datawatch.write_arrivals_baseline(state, first, NOW, str(data)) == [], (
        "the first pass has no previous scan, so it can claim no departure")

    second = {"sessions": OLD, "workers.db": OLD}
    newly = datawatch.write_arrivals_baseline(state, second, NOW + 3600, str(data))
    assert [f["name"] for f in newly] == ["scratch-notes.txt"], (
        f"the name missing from this scan is the departure: {newly}")
    assert newly[0]["last_seen"] == OLD, (
        f"the finding must carry when the leg last measured the name: {newly[0]}")
    assert "age_days" in newly[0], (
        f"and how long ago that was, measured on the pass's own clock: {newly[0]}")

    persisted = json.loads((state / "data-root-arrivals.json").read_text(encoding="utf-8"))
    assert set(persisted) == {"root", "seen", "updated", "departed"}, (
        f"the envelope gains one key and keeps the other three: {sorted(persisted)}")
    assert persisted["departed"] == ["scratch-notes.txt"], persisted["departed"]
    assert "scratch-notes.txt" in persisted["seen"], (
        "`seen` is monotonic: a departure is journalled, the arrival history is not "
        f"edited — {sorted(persisted['seen'])}")
    assert datawatch.read_arrivals_departed(str(state), str(data)) == ["scratch-notes.txt"]

    again = datawatch.write_arrivals_baseline(state, second, NOW + 7200, str(data))
    assert again == [], (
        f"a recorded departure is not re-reported on a later tick: {again}")
    assert json.loads((state / "data-root-arrivals.json").read_text(
        encoding="utf-8"))["departed"] == ["scratch-notes.txt"], (
        "and the journal keeps the record even though this pass said nothing")

    report = datawatch.departures_alert(newly)
    assert report["needs_human"] is True, (
        f"a name that left a person's root is their decision, like an arrival: {report}")
    assert "scratch-notes.txt" in report["reason"], report["reason"]
    assert datawatch.departures_alert([]) is None, "no departure, no statement"


def test_a_sidecar_that_vanished_while_its_store_stands_is_not_called_a_departure(tmp_path):
    """The mute the arrivals leg already has, read against a disappearance.

    `_sidecar_of_store_in_this_scan` exists because a `-wal` that is checkpointed away and
    recreated by the next writer is not an event: asked about ARRIVALS it suppresses an
    alert for every sidecar. Asked about departures it does the same job for the same
    reason, and the pairing may only be visible in the OLD scan — the new one is precisely
    where the sidecar is absent — so the question is put to `previous ∪ files`.

    The other half of the node is why the mute cannot be naive: when a store and its sidecar
    BOTH go missing the store's own name must still be reported, with the sidecar folded
    into it rather than said twice. And a sidecar whose base name the leg never saw is an
    unexplained name going missing, which is reported. Three journal directories, one per
    claim, so no node depends on another's leftovers.
    """
    data = tmp_path / "lloyd-data"
    data.mkdir()

    checkpoint = tmp_path / "s1"
    datawatch.write_arrivals_baseline(checkpoint, {"usage.db": OLD, "usage.db-wal": OLD,
                                                   "kg.sqlite": OLD}, NOW, str(data))
    assert datawatch.write_arrivals_baseline(
        checkpoint, {"usage.db": OLD, "kg.sqlite": OLD}, NOW + 3600, str(data)) == [], (
        "a WAL checkpointed away while its store stands is no one's deletion, and the "
        "pairing that says so exists only in the previous scan")

    deleted = tmp_path / "s2"
    datawatch.write_arrivals_baseline(deleted, {"usage.db": OLD, "usage.db-wal": OLD},
                                      NOW, str(data))
    names = [f["name"] for f in datawatch.write_arrivals_baseline(
        deleted, {"kg.sqlite": OLD}, NOW + 3600, str(data))]
    assert names == ["usage.db"], (
        f"the store is the event and its sidecar is that event said twice: {names}")

    orphan = tmp_path / "s3"
    datawatch.write_arrivals_baseline(orphan, {"research.db-wal": OLD}, NOW, str(data))
    names = [f["name"] for f in datawatch.write_arrivals_baseline(
        orphan, {"sessions": OLD}, NOW + 3600, str(data))]
    assert names == ["research.db-wal"], (
        "a sidecar whose base this leg never saw has no explanation available, so its "
        f"disappearance is the news: {names}")
