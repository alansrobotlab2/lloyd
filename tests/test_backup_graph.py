"""backup-graph.sh must refuse to snapshot a graph that has lost its edges.

The script is driven through a fake $HOME so it never touches the real
_pipeline. Each case builds the minimum tree the script reads:
<HOME>/lloyd/_pipeline/{vault-derived/facts, memory-graph}.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backup" / "backup-graph.sh"


def _fake_home(tmp_path: Path, *, edges: list, baseline: int | None, raw: str | None = None) -> Path:
    home = tmp_path / "home"
    pipeline = home / "lloyd" / "_pipeline"
    facts = pipeline / "vault-derived" / "facts"
    facts.mkdir(parents=True)
    (pipeline / "memory-graph").mkdir()
    rel = facts / "_relationships.json"
    if raw is not None:
        rel.write_text(raw, encoding="utf-8")
    else:
        rel.write_text(json.dumps({"edges": edges, "schema_version": 1}), encoding="utf-8")
    (facts / "entity-aliases.json").write_text("{}", encoding="utf-8")
    if baseline is not None:
        (pipeline / "memory-graph" / "graph-baseline.json").write_text(
            json.dumps({"active_edges": baseline}), encoding="utf-8"
        )
    return home


def _run(home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=str(home))
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def _edges(n: int) -> list:
    return [{"source": f"A{i}", "target": f"B{i}", "type": "uses", "expired_at": None} for i in range(n)]


def test_refuses_below_half_baseline_and_keeps_previous_tarball(tmp_path):
    home = _fake_home(tmp_path, edges=_edges(100), baseline=3894)
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


def test_refuses_on_unparseable_index(tmp_path):
    home = _fake_home(tmp_path, edges=[], baseline=3894, raw='{"edges": [{"sou')
    result = _run(home)
    assert result.returncode == 1
    assert "will not parse" in result.stderr


def test_writes_snapshot_when_at_baseline(tmp_path):
    home = _fake_home(tmp_path, edges=_edges(3900), baseline=3894)
    result = _run(home)
    assert result.returncode == 0, result.stderr
    dest = home / "lloyd" / "_pipeline" / "backups" / "daily"
    tarballs = list(dest.glob("graph-*.tar.gz"))
    assert len(tarballs) == 1
    assert "3900 active edges" in result.stdout


def test_expired_edges_do_not_count_as_active(tmp_path):
    edges = _edges(100) + [
        {"source": f"X{i}", "target": f"Y{i}", "type": "uses", "expired_at": "2026-01-01"}
        for i in range(4000)
    ]
    home = _fake_home(tmp_path, edges=edges, baseline=3894)
    result = _run(home)
    assert result.returncode == 1
    assert "100 active edges" in result.stderr


def test_no_baseline_means_no_comparison(tmp_path):
    home = _fake_home(tmp_path, edges=_edges(5), baseline=None)
    result = _run(home)
    assert result.returncode == 0, result.stderr
