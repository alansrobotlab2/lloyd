"""#4262: the witness naming-convention report — reading-date stays, self-declaration is checkable.

Every node builds its own temporary vault (a real git repo) so the report reads
`git ls-files` exactly as it does over `~/obsidian`, and never touches the real one.
"""
from __future__ import annotations

import inspect
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from scripts.automod import backlog as B
from scripts.maintenance import witness_naming_report as W

REPO = Path(__file__).resolve().parents[1]
STORE = "backlog/data/2026-01-02.7-store-x.db"


def _git(vault: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(vault), "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True).stdout


def _db(path: Path, *, stamp: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE queue (id INTEGER, kind TEXT)")
    con.execute("INSERT INTO queue VALUES (1, 'scheduled-task')")
    if stamp == "kv":            # 2026-11-06.3850-lloyd-data-usage.db's shape
        con.execute("CREATE TABLE witness_meta (key TEXT, value TEXT)")
        con.execute("INSERT INTO witness_meta VALUES ('source_store', 'usage.db, read-only')")
    elif stamp == "columnar":    # 2026-11-07.3907-lloyd-data-workers.db's shape
        con.execute("CREATE TABLE witness_meta (captured_at TEXT, source TEXT, "
                    "source_bytes INTEGER, rows INTEGER, note TEXT)")
        con.execute("INSERT INTO witness_meta VALUES ('2026-11-10T00:05:30Z', 'workers.db', "
                    "32935936, 14433, 'CAPTURE READING, NOT A 2026-11-07 SNAPSHOT')")
    con.commit()
    con.close()


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "backlog" / "data").mkdir(parents=True)
    _git(v, "init", "-q")
    (v / ".gitignore").write_text("backlog/data/*.db-wal\nbacklog/data/*.db-shm\n"
                                  "backlog/data/*.sqlite*-wal\nbacklog/data/*.sqlite*-shm\n")
    _db(v / STORE)
    _git(v, "add", "-A")
    _git(v, "commit", "-qm", "seed")
    return v


def _commit(v: Path, msg: str = "x") -> None:
    _git(v, "add", "-A")
    _git(v, "commit", "-qm", msg)


@pytest.mark.parametrize("shape", ["kv", "columnar"])
def test_an_unstamped_store_is_named_and_a_witness_meta_table_clears_it(vault, shape):
    """Clause 1: listed with no stamp and no note; gone once `witness_meta` exists, in
    either committed schema — detection is by table name."""
    r = W.report(vault)
    assert r["unstamped_stores"] == [STORE], r
    assert STORE in W.render(r)

    (vault / STORE).unlink()
    _db(vault / STORE, stamp=shape)
    _commit(vault)
    r = W.report(vault)
    assert STORE not in r["unstamped_stores"], (shape, r)
    assert r["dated"] == 1


def test_an_uncommitted_store_is_not_reported(vault):
    _db(vault / "backlog/data/2026-01-03.8-loose.db")
    assert "backlog/data/2026-01-03.8-loose.db" not in W.report(vault)["unstamped_stores"]


@pytest.mark.parametrize("note", [
    "2026-01-02.7-store-x-witness.md",       # `-witness.md`, shared <date>.<id> prefix
    "2026-01-02.7-queue-kinds.witness.md",   # `.witness.md`, prefix shared, stem differs
    "store-x.db.witness.md",                 # no <date>.<id>- prefix at all
    "store-x-witness.md",                    # no prefix, extension dropped
])
def test_either_note_spelling_satisfies_the_sibling_arm(vault, note):
    """Clause 2."""
    assert W.report(vault)["unstamped_stores"] == [STORE]
    (vault / "backlog/data" / note).write_text("captured 2026-01-04T00:00Z\n")
    _commit(vault)
    assert W.report(vault)["unstamped_stores"] == [], note


def test_a_note_for_another_item_does_not_clear_it(vault):
    (vault / "backlog/data/2026-01-02.8-store-x-other-witness.md").write_text("x\n")
    (vault / "backlog/data/other.db.witness.md").write_text("x\n")
    _commit(vault)
    assert W.report(vault)["unstamped_stores"] == [STORE]


def test_text_artifacts_are_counted_apart_and_only_a_note_clears_them(vault):
    (vault / "backlog/data/2026-01-05.9-ledger.jsonl").write_text('{"a": 1}\n')
    (vault / "backlog/data/2026-01-05.9-readme.md").write_text("not an artifact\n")
    _commit(vault)
    r = W.report(vault)
    assert r["unstamped_text"] == ["backlog/data/2026-01-05.9-ledger.jsonl"], r
    assert r["dated"] == 2, "a .md file is never a dated artifact"
    (vault / "backlog/data/2026-01-05.9-ledger.witness.md").write_text("x\n")
    _commit(vault)
    assert W.report(vault)["unstamped_text"] == []


def test_sidecars_are_counted_by_directory_scan_where_git_status_sees_nothing(vault):
    """Clause 3."""
    _db(vault / "backlog/data/kg.sqlite")
    _commit(vault)
    shm = vault / (STORE + "-shm")
    wal = vault / "backlog/data/kg.sqlite-wal"
    shm.write_bytes(b"\0" * 32768)
    wal.write_bytes(b"")
    porcelain = _git(vault, "status", "--porcelain", "--untracked-files=all")
    assert porcelain == "", f"the fixture must hide the sidecars from git status: {porcelain!r}"
    assert os.path.exists(shm) and os.path.exists(wal)

    r = W.report(vault)
    assert sorted(r["sidecars"]) == sorted([STORE + "-shm", "backlog/data/kg.sqlite-wal"]), r
    assert r["sidecars_indexed"] == []
    assert "2 SQLite sidecars beside tracked stores" in W.render(r)


def test_the_report_writes_nothing_and_no_commit_door_imports_it(vault, capsys):
    """Clause 4: report-only, structurally."""
    (vault / STORE).with_name("2026-01-02.7-store-x.db-shm").write_bytes(b"\0" * 8)
    before_status = _git(vault, "status", "--porcelain", "--untracked-files=all",
                         "--ignored")
    index = vault / ".git" / "index"
    before_index = (index.read_bytes(), index.stat().st_mtime_ns)
    before_tree = sorted((str(p.relative_to(vault)), p.stat().st_mtime_ns, p.stat().st_size)
                         for p in vault.rglob("*") if ".git" not in p.parts)

    assert W.main(["--vault", str(vault)]) == 0
    assert W.main(["--vault", str(vault), "--json"]) == 0
    assert STORE in capsys.readouterr().out

    # The index first: the test's own `git status` below may refresh it.
    assert (index.read_bytes(), index.stat().st_mtime_ns) == before_index, (
        "the report touched the vault index")
    assert _git(vault, "status", "--porcelain", "--untracked-files=all",
                "--ignored") == before_status
    after_tree = sorted((str(p.relative_to(vault)), p.stat().st_mtime_ns, p.stat().st_size)
                        for p in vault.rglob("*") if ".git" not in p.parts)
    assert after_tree == before_tree, "the report created, renamed or rewrote a file"

    for door in ("scripts/automod/vault_round.py", "scripts/util/vault-commit.sh"):
        text = (REPO / door).read_text(encoding="utf-8")
        assert "witness_naming_report" not in text, f"{door} wires the report into a land"


@pytest.mark.parametrize("fn", ["_witness_artifact", "_over_bound_extract_demand"])
def test_both_emitters_state_the_naming_ruling(fn):
    """Clause 5. The functions moved from `scripts/automod/backlog.py` into
    `scripts/automod/witness.py` (1868cb27) and are bound back onto `backlog`, so the
    source is read through the `backlog` name the clause cites."""
    src = inspect.getsource(getattr(B, fn))
    flat = " ".join(src.split())
    assert "a filename's date is the READING / demand date" in flat, fn
    assert "capture instant lives INSIDE the artifact" in flat, fn
    assert "Capture-date equality is deliberately NOT checked" in flat \
        or "capture-date equality is deliberately NOT checked" in flat, fn


def test_the_tree_now_names_witness_meta_in_python():
    """The item's own verification: `git grep -c witness_meta -- '*.py'` was empty."""
    out = subprocess.run(["git", "-C", str(REPO), "grep", "-l", "witness_meta", "--", "*.py"],
                         capture_output=True, text=True).stdout
    assert "scripts/maintenance/witness_naming_report.py" in out.split()
