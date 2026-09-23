"""pre-reimage-backup.sh must carry the irreplaceable half of `_pipeline/` off-box.

#947: this script is the only off-box copy mechanism on the box, and before this
round its `_pipeline` entries were the three autoresearch paths `cf688eb` added
(~20 MB). Everything that cannot be re-derived from the vault — the knowledge
graph, the merge evidence, the trajectories, the metrics — stayed single-copy
inside the same tree the KG's own snapshots live in, so one reimage erases the
data and its backups together.

Driven through a fake `$HOME` the way `tests/test_backup_graph.py` drives its
script. One difference matters: `pre-reimage-backup.sh` derives `REPO` from its
own location rather than from `$HOME`, so the file under test is copied
byte-for-byte into `<HOME>/lloyd/scripts/` and run from there. Nothing under the
real `_pipeline` is read or written by these tests.
"""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "pre-reimage-backup.sh"

# The three switches the script reads. Cleared in every run so a developer's
# own environment cannot change what a "default run" means.
SWITCHES = ("INCLUDE_MODELS", "INCLUDE_SESSIONS", "INCLUDE_PIPELINE_BULK")


def _fake_home(tmp_path: Path, dailies: dict[str, bytes] | None = None) -> Path:
    """Build the tree the script reads, minus every path it skips over.

    `dailies` is the contents of `_pipeline/backups/daily/` as name → bytes;
    `None` means one tarball dated 2026-01-01, `{}` means the directory exists
    and holds nothing — the state of a box whose backup timer has not fired yet.
    """
    home = tmp_path / "home"
    repo = home / "lloyd"
    # Runtime data lives in the data root (`~/lloyd-data`), not the tree.
    pipeline = home / "lloyd-data" / "_pipeline"

    # The live store, WAL-mode. A raw copy of a database mid-write is not
    # restorable (SETUP.md Part 0), so these two files are what the script must
    # never ship — the daily tarball is the vehicle instead.
    vault_derived = pipeline / "vault-derived"
    (vault_derived / "facts").mkdir(parents=True)
    (vault_derived / "kg.sqlite").write_bytes(b"LIVE STORE")
    (vault_derived / "kg.sqlite-wal").write_bytes(b"LIVE WAL")
    (vault_derived / "facts" / "a.md").write_text("fact body", encoding="utf-8")

    # Merge/apply/verdict evidence, plus the raw store copies kept beside it.
    memgraph = pipeline / "memory-graph"
    (memgraph / "store-backups").mkdir(parents=True)
    (memgraph / "graph-baseline.json").write_text('{"active_edges": 3894}', encoding="utf-8")
    (memgraph / "merge-plan.json").write_text('{"moves": []}', encoding="utf-8")
    (memgraph / "store-backups" / "kg-v4-x.sqlite").write_bytes(b"RAW STORE COPY")

    daily = pipeline / "backups" / "daily"
    daily.mkdir(parents=True)
    for name, body in ({"graph-20260101.tar.gz": b"CONSISTENT SNAPSHOT"}
                       if dailies is None else dailies).items():
        (daily / name).write_bytes(body)

    (pipeline / "trajectories").mkdir()
    (pipeline / "trajectories" / "run.jsonl").write_text('{"session": "a"}\n', encoding="utf-8")
    (pipeline / "metrics").mkdir()
    (pipeline / "metrics" / "routing.json").write_text('{"n": 1}', encoding="utf-8")

    # Autoresearch: the subset cf688eb already copies, plus the regenerable bulk.
    research = pipeline / "research"
    (research / "rounds").mkdir(parents=True)
    (research / "ledger.jsonl").write_text('{"round": 1}\n', encoding="utf-8")
    (research / "rounds" / "round-1.md").write_text("# round 1\n", encoding="utf-8")
    (research / "snapshots" / "20260101_000000").mkdir(parents=True)
    (research / "snapshots" / "20260101_000000" / "MEMORY.md").write_text("memory\n",
                                                                          encoding="utf-8")
    (research / "variants").mkdir()
    (research / "variants" / "variant-a.json").write_text('{"v": 1}', encoding="utf-8")
    (research / "_debug").mkdir()
    (research / "_debug" / "dump.txt").write_text("debug dump", encoding="utf-8")

    # REPO comes from the script's own directory, so run it from inside the fake
    # tree — and run the real bytes, not a stand-in.
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / "pre-reimage-backup.sh")
    return home


def _run(home: Path, dest: Path, env_extra: dict[str, str] | None = None
         ) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in SWITCHES and k not in ("LLOYD_DATA", "LLOYD_DATA_SNAPSHOTS")}
    env["HOME"] = str(home)
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(home / "lloyd" / "scripts" / "pre-reimage-backup.sh"), str(dest)],
        env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=str(home))


def _fail(result: subprocess.CompletedProcess) -> str:
    return f"exit {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"


def test_the_graph_reaches_the_destination_only_as_the_daily_tarball(tmp_path):
    """The live WAL database must never be what gets copied off-box.

    `_pipeline/vault-derived/kg.sqlite` is the thing the item is about, and the
    tempting one-line fix is `copy … "$REPO/_pipeline/vault-derived/"`. That
    ships a database mid-write that SETUP.md says outright is not restorable, so
    the acceptance names the vehicle instead: the tarball `backup-graph.sh`
    builds nightly from `KGStore.backup()`, which is consistent by construction.
    """
    home = _fake_home(tmp_path)
    dest = tmp_path / "dest"

    result = _run(home, dest)
    assert result.returncode == 0, _fail(result)

    shipped = dest / "lloyd-data/_pipeline/backups/daily/graph-20260101.tar.gz"
    assert shipped.is_file(), f"daily graph tarball missing\n{_fail(result)}"
    assert shipped.read_bytes() == b"CONSISTENT SNAPSHOT"

    raw = sorted(p.name for p in dest.rglob("*")
                 if p.name in {"kg.sqlite", "kg.sqlite-wal", "kg.sqlite-shm"})
    assert raw == [], f"raw store files reached the destination: {raw}"


def test_the_newest_daily_tarball_is_the_one_copied(tmp_path):
    """One consistent snapshot is what a restore needs; the whole window is 1.4 GB.

    `backup-graph.sh` rotates 30 dailies on-box. The default run takes the newest
    only, so the copy stays proportional to what a reimage actually restores from.
    """
    home = _fake_home(tmp_path, dailies={
        "graph-20260101.tar.gz": b"OLD SNAPSHOT",
        "graph-20260920.tar.gz": b"NEWEST SNAPSHOT",
    })
    daily = home / "lloyd-data/_pipeline/backups/daily"
    os.utime(daily / "graph-20260101.tar.gz", (1767225600, 1767225600))  # 2026-01-01
    os.utime(daily / "graph-20260920.tar.gz", (1789862400, 1789862400))  # 2026-09-20
    dest = tmp_path / "dest"

    result = _run(home, dest)
    assert result.returncode == 0, _fail(result)

    landed = sorted(p.name for p in (dest / "lloyd-data/_pipeline/backups/daily").iterdir())
    assert landed == ["graph-20260920.tar.gz"], landed
    assert (dest / "lloyd-data/_pipeline/backups/daily/graph-20260920.tar.gz").read_bytes() \
        == b"NEWEST SNAPSHOT"


def test_a_default_run_carries_the_merge_evidence_trajectories_and_metrics(tmp_path):
    """None of these sits behind an `INCLUDE_*` gate, so all switches are set off.

    The merge plans and `graph-baseline.json` are what makes a bad merge
    revertable, and they are the reason #917's guard can compare against a
    baseline at all; trajectories and metrics are the only record of how each
    routing decision performed. Both are single-copy without this entry.
    """
    home = _fake_home(tmp_path)
    dest = tmp_path / "dest"

    result = _run(home, dest, {"INCLUDE_MODELS": "0", "INCLUDE_SESSIONS": "0",
                               "INCLUDE_PIPELINE_BULK": "0"})
    assert result.returncode == 0, _fail(result)

    pipeline_dest = dest / "lloyd-data/_pipeline"
    for rel in ("memory-graph/graph-baseline.json", "memory-graph/merge-plan.json",
                "trajectories/run.jsonl", "metrics/routing.json",
                "research/ledger.jsonl", "research/rounds/round-1.md",
                "research/snapshots/20260101_000000/MEMORY.md"):
        assert (pipeline_dest / rel).is_file(), f"{rel} missing from a default run"

    # memory-graph/store-backups/ is 1.2 GB of raw db copies beside the evidence.
    # The tarball above already carries a restorable store, so they stay on-box.
    assert not (pipeline_dest / "memory-graph/store-backups").exists()


def test_regenerable_bulk_is_absent_by_default_and_present_on_the_opt_in(tmp_path):
    """`variants/` and `_debug/` are re-derivable, so they cost an opt-in."""
    home = _fake_home(tmp_path)
    default_dest = tmp_path / "default"
    result = _run(home, default_dest)
    assert result.returncode == 0, _fail(result)
    research = default_dest / "lloyd-data/_pipeline/research"
    assert not (research / "variants").exists()
    assert not (research / "_debug").exists()

    opt_in_dest = tmp_path / "opt_in"
    result = _run(home, opt_in_dest, {"INCLUDE_PIPELINE_BULK": "1"})
    assert result.returncode == 0, _fail(result)
    research = opt_in_dest / "lloyd-data/_pipeline/research"
    assert (research / "variants" / "variant-a.json").is_file()
    assert (research / "_debug" / "dump.txt").is_file()


def test_a_missing_daily_tarball_is_recorded_and_not_reported_as_captured(tmp_path):
    """No tarball yet is a named gap in the manifest, not a silent success.

    `copy()` already turns an absent source into a `SKIP` line plus a `missing[]`
    entry; the graph snapshot is found by glob rather than by path, so it needs
    the same treatment explicitly — and the record has to outlive the terminal,
    because the point of this script is a destination you inspect later from
    another machine.
    """
    home = _fake_home(tmp_path, dailies={})
    dest = tmp_path / "dest"

    result = _run(home, dest)
    assert result.returncode == 0, _fail(result)

    assert "SKIP" in result.stdout, _fail(result)
    assert "_pipeline/backups/daily" in result.stdout, _fail(result)
    manifest = (dest / "MANIFEST.txt").read_text(encoding="utf-8")
    assert "NOT captured" in manifest, manifest
    assert "_pipeline/backups/daily" in manifest, manifest
    assert not list((dest / "lloyd-data/_pipeline").rglob("graph-*.tar.gz"))


def test_a_snapshot_is_the_source_when_one_exists(tmp_path):
    """The data root goes whole, and from the newest read-only snapshot when one
    exists: a btrfs snapshot is atomic, so its WAL databases are restorable."""
    home = _fake_home(tmp_path)
    snaps = home / ".lloyd-data-snapshots"
    older = snaps / "20260101T000000Z" / "_pipeline" / "vault-derived"
    newer = snaps / "20260920T000000Z" / "_pipeline" / "vault-derived"
    for d, body in ((older, b"OLD STORE"), (newer, b"SNAPSHOT STORE")):
        d.mkdir(parents=True)
        (d / "kg.sqlite").write_bytes(body)
    dest = tmp_path / "dest"

    result = _run(home, dest)
    assert result.returncode == 0, _fail(result)
    assert (dest / "lloyd-data/_pipeline/vault-derived/kg.sqlite").read_bytes() \
        == b"SNAPSHOT STORE"
