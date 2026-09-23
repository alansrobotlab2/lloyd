"""backup-graph.sh must refuse to snapshot a graph whose state it cannot verify.

The script is driven through a fake $HOME so it never touches the real
_pipeline. Each case builds the minimum tree the script reads:
<HOME>/lloyd/_pipeline/{vault-derived/{facts,kg.sqlite}, memory-graph}.

Two refusals are pinned here, one per unreadable input: the store will not open,
and there is no usable active-edge baseline. The second is what #917 is about —
the guard used to skip itself silently when graph-baseline.json was missing or
unparseable, which is the exact state the 2026-08-22 destruction left behind.
"""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.kg_store import KGStore  # noqa: E402

SCRIPT = ROOT / "scripts" / "backup" / "backup-graph.sh"


def _baseline_path(home: Path) -> Path:
    return home / "lloyd-data" / "_pipeline" / "memory-graph" / "graph-baseline.json"


def _tarballs(home: Path) -> list[Path]:
    return sorted((home / "lloyd-data" / "_pipeline" / "backups" / "daily").glob("graph-*.tar.gz"))


def _fake_home(tmp_path: Path, *, active: int = 0, expired: int = 0,
               baseline=None, baseline_raw: str | None = None,
               corrupt: bool = False) -> Path:
    """Build the tree the script reads.

    `baseline` is an active_edges count to record, a full dict to write as the
    baseline JSON verbatim, or None to leave the file absent. `baseline_raw`
    writes the file exactly as given, for a body that is not valid JSON at all,
    and wins over `baseline`.
    """
    home = tmp_path / "home"
    pipeline = home / "lloyd-data" / "_pipeline"
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
    if baseline_raw is not None:
        _baseline_path(home).write_text(baseline_raw, encoding="utf-8")
    elif isinstance(baseline, dict):
        _baseline_path(home).write_text(json.dumps(baseline), encoding="utf-8")
    elif baseline is not None:
        _baseline_path(home).write_text(
            json.dumps({"active_edges": baseline}), encoding="utf-8"
        )
    return home


def _run(home: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=str(home), LLOYD_PYTHON=sys.executable,
               LLOYD_DATA=str(home / "lloyd-data"))
    env.update(extra_env or {})
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
                          cwd=str(ROOT))


def test_the_script_is_executable_because_systemd_execs_it_directly(tmp_path):
    """lloyd-graph-backup.service runs `ExecStart=%h/lloyd/scripts/backup/backup-graph.sh`.

    A oneshot unit execs the file itself rather than handing it to a shell, so
    the mode bit is part of the interface and an editor that rewrote the file
    without it would take the nightly backup down with EACCES. The tests here all
    invoke `bash <script>`, which passes whatever the mode is, so nothing else
    in this file would notice.
    """
    assert SCRIPT.stat().st_mode & 0o111, "backup-graph.sh must stay executable for systemd"

    # And the exec path itself: same invocation shape as the unit, no shell
    # named in front of it. A missing bit surfaces here as EACCES.
    home = _fake_home(tmp_path / "direct", active=5, baseline=None)
    env = dict(os.environ, HOME=str(home), LLOYD_PYTHON=sys.executable,
               LLOYD_DATA=str(home / "lloyd-data"))
    result = subprocess.run([str(SCRIPT)], env=env, capture_output=True, text=True,
                            cwd=str(ROOT))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr


def _guard_line(result: subprocess.CompletedProcess) -> str:
    """The stdout line reporting the counts the guard compared."""
    return next(line for line in result.stdout.splitlines() if "active edges" in line)


def test_refuses_below_half_baseline_and_keeps_previous_tarball(tmp_path):
    home = _fake_home(tmp_path, active=100, baseline=3894)
    dest = home / "lloyd-data" / "_pipeline" / "backups" / "daily"
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


def test_refuses_when_no_baseline_exists_and_keeps_previous_tarball(tmp_path):
    """#917: an absent reference is a refusal, not a skipped comparison.

    This replaces test_no_baseline_means_no_comparison, which asserted
    returncode == 0 for exactly this state and so pinned the fail-open. The
    acceptance for #917 is that a missing baseline is refused the way an
    unreadable store already was: exit 1, the path named, no tarball, and the
    snapshot window not rotated.
    """
    home = _fake_home(tmp_path, active=5, baseline=None)
    dest = home / "lloyd-data" / "_pipeline" / "backups" / "daily"
    dest.mkdir(parents=True)
    previous = dest / "graph-20260101.tar.gz"
    previous.write_bytes(b"previous snapshot")

    result = _run(home)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert str(_baseline_path(home)) in result.stderr, result.stderr
    assert previous.read_bytes() == b"previous snapshot"
    assert _tarballs(home) == [previous], "no new tarball may be written without a reference"


def test_refuses_when_the_baseline_is_not_json(tmp_path):
    home = _fake_home(tmp_path, active=3900, baseline_raw="not json at all {{{")
    result = _run(home)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert str(_baseline_path(home)) in result.stderr
    assert _tarballs(home) == []


def test_refuses_when_the_baseline_carries_no_active_edges(tmp_path):
    home = _fake_home(tmp_path, active=3900,
                      baseline_raw=json.dumps({"recorded_at": "2026-09-16T00:03:08+00:00"}))
    result = _run(home)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert str(_baseline_path(home)) in result.stderr
    assert _tarballs(home) == []


def test_refuses_when_the_baseline_is_not_positive(tmp_path):
    """A reference of 0 edges compares against nothing, so it is no reference."""
    home = _fake_home(tmp_path, active=3900,
                      baseline_raw=json.dumps({"active_edges": 0,
                                               "recorded_at": "2026-09-16T00:03:08+00:00"}))
    result = _run(home)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert str(_baseline_path(home)) in result.stderr
    assert _tarballs(home) == []


def test_the_script_reads_no_override_switch_at_all():
    """The refusal has no escape hatch, and no knob that could grow one.

    Enumerating switch names would only prove those names do nothing, so this
    reads the property instead: every variable the script expands is either its
    own path/loop local or one of the two documented inputs, and none of them is
    a permission flag. Adding `ALLOW_NO_BASELINE=1` later would trip this on the
    name alone, without anyone having to think of the name first.
    """
    import re

    source = SCRIPT.read_text(encoding="utf-8")
    # HOME and LLOYD_PYTHON are the only two the script takes from the
    # environment; everything else in the set is its own path/loop local.
    expanded = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", source))
    switches = {n for n in expanded if re.search(r"ALLOW|SKIP|FORCE|IGNORE|OVERRIDE|BYPASS|NO_BASELINE", n)}
    assert switches == set(), f"guard override switches appeared: {sorted(switches)}"
    assert "HOME" in expanded and "LLOYD_PYTHON" in expanded
    assert not re.search(r"os\.environ|getenv", source), "the guard must not read the environment either"


def test_no_environment_override_disables_the_baseline_check(tmp_path):
    """And the refusal still refuses with the most plausible names set."""
    home = _fake_home(tmp_path, active=5, baseline=None)
    result = _run(home, extra_env={
        "ALLOW_NO_BASELINE": "1",
        "BACKUP_GRAPH_SKIP_GUARD": "1",
    })
    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING" in result.stderr
    assert _tarballs(home) == []


def test_writes_snapshot_when_at_baseline(tmp_path):
    home = _fake_home(tmp_path, active=3900, baseline=3894)
    result = _run(home)
    assert result.returncode == 0, result.stderr
    dest = home / "lloyd-data" / "_pipeline" / "backups" / "daily"
    tarballs = list(dest.glob("graph-*.tar.gz"))
    assert len(tarballs) == 1
    assert "3900 active edges" in result.stdout


def test_success_line_names_when_the_baseline_was_recorded(tmp_path):
    home = _fake_home(tmp_path, active=3900,
                      baseline={"active_edges": 3894, "recorded_at": "2026-09-16T00:03:08+00:00"})
    result = _run(home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _guard_line(result) == (
        "backup-graph: 3900 active edges (baseline 3894, recorded 2026-09-16T00:03:08+00:00)"
    )


def test_success_line_says_unknown_when_recorded_at_is_absent(tmp_path):
    home = _fake_home(tmp_path, active=3900, baseline=3894)
    result = _run(home)
    assert result.returncode == 0, result.stdout + result.stderr
    line = _guard_line(result)
    assert "baseline 3894" in line
    assert "recorded unknown" in line


def test_the_baseline_the_sweep_actually_writes_passes_the_guard(tmp_path):
    """The reference is written by another program and read by this one.

    entity-resolution-sweep.py owns graph-baseline.json: `update_baseline` is a
    max-ratchet that stamps `recorded_at` with a timezone-aware isoformat. The
    shell script never imports it, so nothing but this test puts the writer's
    real bytes in front of the guard.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "entity_resolution_sweep", ROOT / "scripts" / "memory" / "entity-resolution-sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)

    home = _fake_home(tmp_path, active=3900)  # no baseline file yet
    assert sweep.update_baseline(3900, _baseline_path(home)) == 3900
    recorded = json.loads(_baseline_path(home).read_text())["recorded_at"]
    assert recorded.startswith("20"), recorded

    result = _run(home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _guard_line(result) == (
        f"backup-graph: 3900 active edges (baseline 3900, recorded {recorded})")


def test_success_line_prints_a_spaced_timestamp_without_reshaping_it(tmp_path):
    """A stamp written with a space in it survives the field split intact.

    The guard reads the line with `read -r ACTIVE BASE RECORDED`, and a
    hand-edited baseline can carry any string, so the value must print as
    written rather than have its separator deleted.
    """
    stamp = "2026-09-16 00:03:08 +0000"
    home = _fake_home(tmp_path, active=3900,
                      baseline={"active_edges": 3894, "recorded_at": stamp})
    result = _run(home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _guard_line(result) == f"backup-graph: 3900 active edges (baseline 3894, recorded {stamp})"


def test_snapshot_contains_a_restorable_store_and_a_json_export(tmp_path):
    home = _fake_home(tmp_path, active=120, baseline=100)
    assert _run(home).returncode == 0
    tarball = next(iter(_tarballs(home)))
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
    backups = home / "lloyd-data" / "_pipeline" / "backups"
    assert not list(backups.glob(".staging-*"))


def test_expired_edges_do_not_count_as_active(tmp_path):
    home = _fake_home(tmp_path, active=100, expired=4000, baseline=3894)
    result = _run(home)
    assert result.returncode == 1
    assert "100 active edges" in result.stderr
