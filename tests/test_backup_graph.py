"""backup-graph.sh must refuse to snapshot a graph that has lost its edges.

The script is driven through a fake $HOME so it never touches the real
_pipeline. Each case builds the minimum tree the script reads:
<HOME>/lloyd/_pipeline/{vault-derived/{facts,kg.sqlite}, memory-graph}.
"""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.kg_store import KGStore  # noqa: E402

SCRIPT = ROOT / "scripts" / "backup" / "backup-graph.sh"


def _fake_home(tmp_path: Path, *, active: int = 0, expired: int = 0,
               baseline: int | None = None, corrupt: bool = False) -> Path:
    home = tmp_path / "home"
    pipeline = home / "lloyd" / "_pipeline"
    facts = pipeline / "vault-derived" / "facts"
    facts.mkdir(parents=True)
    (pipeline / "memory-graph").mkdir()
    db = pipeline / "vault-derived" / "kg.sqlite"
    if corrupt:
        db.write_bytes(b"not a database, just 48 bytes of noise ........")
    else:
        s = KGStore(db)
        for i in range(active):
            s.edges.add({"source": f"A{i}", "target": f"B{i}", "type": "uses"}, origin="test")
        for i in range(expired):
            eid = s.edges.add({"source": f"X{i}", "target": f"Y{i}", "type": "uses"}, origin="test")
            s.edges.expire(eid, "test")
        s.close()
    if baseline is not None:
        (pipeline / "memory-graph" / "graph-baseline.json").write_text(
            json.dumps({"active_edges": baseline}), encoding="utf-8"
        )
    return home


def _run(home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=str(home), LLOYD_PYTHON=sys.executable)
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                          cwd=str(ROOT))


def test_refuses_below_half_baseline_and_keeps_previous_tarball(tmp_path):
    home = _fake_home(tmp_path, active=100, baseline=3894)
    dest = home / "lloyd" / "_pipeline" / "backups" / "daily"
    dest.mkdir(parents=True)
    previous = dest / "graph-20260101.tar.gz"
    previous.write_bytes(b"previous snapshot")

    result = _run(home)

    assert result.returncode == 1, result.stderr
    assert "REFUSING" in result.stderr
    assert "below 50%" in result.stderr
    assert previous.read_bytes() == b"previous snapshot"
    assert list(dest.glob("graph-*.tar.gz")) == [previous], "no new tarball must be written"


def test_refuses_on_an_unreadable_store(tmp_path):
    home = _fake_home(tmp_path, baseline=3894, corrupt=True)
    result = _run(home)
    assert result.returncode == 1
    assert "will not open" in result.stderr


def test_writes_snapshot_when_at_baseline(tmp_path):
    home = _fake_home(tmp_path, active=3900, baseline=3894)
    result = _run(home)
    assert result.returncode == 0, result.stderr
    dest = home / "lloyd" / "_pipeline" / "backups" / "daily"
    tarballs = list(dest.glob("graph-*.tar.gz"))
    assert len(tarballs) == 1
    assert "3900 active edges" in result.stdout


def test_snapshot_contains_a_restorable_store_and_a_json_export(tmp_path):
    home = _fake_home(tmp_path, active=120, baseline=100)
    assert _run(home).returncode == 0
    tarball = next((home / "lloyd" / "_pipeline" / "backups" / "daily").glob("graph-*.tar.gz"))
    with tarfile.open(tarball) as tf:
        names = tf.getnames()
        db_member = next(n for n in names if n.endswith("kg.sqlite"))
        assert any(n.endswith("json-export/_relationships.json") for n in names)
        assert any(n.endswith("json-export/entity-aliases.json") for n in names)
        tf.extract(db_member, tmp_path / "restored", filter="data")
    restored = KGStore(tmp_path / "restored" / db_member)
    assert restored.edges.count() == 120
    assert restored.integrity_check() == "ok"
    restored.close()


def test_staging_dir_is_cleaned_up(tmp_path):
    home = _fake_home(tmp_path, active=120, baseline=100)
    assert _run(home).returncode == 0
    backups = home / "lloyd" / "_pipeline" / "backups"
    assert not list(backups.glob(".staging-*"))


def test_expired_edges_do_not_count_as_active(tmp_path):
    home = _fake_home(tmp_path, active=100, expired=4000, baseline=3894)
    result = _run(home)
    assert result.returncode == 1
    assert "100 active edges" in result.stderr


def test_no_baseline_means_no_comparison(tmp_path):
    home = _fake_home(tmp_path, active=5, baseline=None)
    result = _run(home)
    assert result.returncode == 0, result.stderr
