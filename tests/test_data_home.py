"""Runtime data lives outside the code tree, and every copy of the code gets its own.

On 2026-09-22 a pytest fixture teardown deleted `~/lloyd`, and the sessions,
databases, `_pipeline/` and logs went with the code. They moved to
`~/lloyd-data` (`architecture/data-home.md`). What this pins:

* the resolver: production gets `~/lloyd-data` only when it carries its marker
  (never a silent fallback into the tree), any other checkout keeps its own,
  and `LLOYD_DATA` wins;
* isolation: a round's test rungs, its canary and this suite each get a scratch
  root — the data used to be isolated by living inside the worktree, and a path
  outside it isolates nothing unless someone sets it;
* the guardian's data tripwire, the snapshot and restore scripts, and the
  one-shot migration, all against temp trees.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

import app.paths as paths

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import datawatch as DW  # noqa: E402
import guardian as G  # noqa: E402
import policy  # noqa: E402


# ── the resolver ─────────────────────────────────────────────────────────────

def _resolve(tmp_path, *, env=None, tree=None, worktree=False, marker=True):
    live = tmp_path / "home" / "lloyd"
    live.mkdir(parents=True, exist_ok=True)
    prod = tmp_path / "home" / "lloyd-data"
    prod.mkdir(parents=True, exist_ok=True)
    if marker:
        (prod / paths.DATA_ROOT_MARKER).write_text("{}")
    return paths.resolve_data_root(env=env, lloyd_home=(tree or live).resolve(),
                                   is_worktree=worktree, live_checkout=live,
                                   production_root=prod), prod


def test_production_uses_the_data_root_outside_the_tree(tmp_path):
    root, prod = _resolve(tmp_path)
    assert root == prod
    assert not root.is_relative_to(tmp_path / "home" / "lloyd")


def test_production_without_its_marker_refuses_rather_than_writing_into_the_tree(tmp_path):
    with pytest.raises(paths.DataRootMissing, match="refuses to fall back"):
        _resolve(tmp_path, marker=False)


def test_any_other_checkout_keeps_its_own_data(tmp_path):
    sandbox = tmp_path / "lloyd-sandbox"
    sandbox.mkdir()
    root, prod = _resolve(tmp_path, tree=sandbox)
    assert root == sandbox / ".lloyd-data" and root != prod


def test_a_worktree_of_the_live_path_is_not_production(tmp_path):
    # A linked worktree at the live path (a round layout under a swapped HOME)
    # is judged by `.git` being a file, not by where it sits.
    root, prod = _resolve(tmp_path, worktree=True)
    assert root != prod and root.name == ".lloyd-data"


def test_the_environment_wins(tmp_path):
    root, _ = _resolve(tmp_path, env=str(tmp_path / "elsewhere"))
    assert root == tmp_path / "elsewhere"


def test_this_suite_never_resolves_the_live_root():
    assert paths.DATA_ROOT != paths.PRODUCTION_DATA_ROOT
    assert os.environ["LLOYD_DATA"] == str(paths.DATA_ROOT)
    for const in (paths.SESSIONS_DIR, paths.WORKERS_DB, paths.USAGE_DB, paths.LOGS_DIR,
                  paths.VAULT_KG_DB_DEFAULT, paths.EVENT_LOGS_DIR, paths.TASKS_DIR):
        assert const.is_relative_to(paths.DATA_ROOT), const


def test_the_production_root_is_read_off_passwd_not_home(monkeypatch):
    import pwd
    monkeypatch.setenv("HOME", "/nonexistent-round-home")
    assert paths.production_data_root() == Path(pwd.getpwuid(os.getuid()).pw_dir) / "lloyd-data"


def test_config_expands_the_data_root_it_resolved_without_exporting_it(monkeypatch):
    from app import config
    monkeypatch.delenv("LLOYD_DATA", raising=False)
    got = config._expand({"db": "${LLOYD_DATA}/workers.db", "x": ["${LLOYD_DATA}/_pipeline"]})
    assert got == {"db": f"{paths.DATA_ROOT}/workers.db", "x": [f"{paths.DATA_ROOT}/_pipeline"]}
    assert "LLOYD_DATA" not in os.environ


def test_the_workers_db_follows_the_data_root():
    from workers.queue import configured_db_path
    assert configured_db_path() == paths.WORKERS_DB


# ── a round, its canary ──────────────────────────────────────────────────────

def test_a_round_home_never_links_the_live_data_root(tmp_path, monkeypatch):
    from scripts.automod import worktree as W
    real = tmp_path / "real"
    (real / "lloyd-data" / "sessions").mkdir(parents=True)
    (real / "obsidian").mkdir()
    monkeypatch.setenv("HOME", str(real))
    wt = tmp_path / "work" / "SM_x" / "home" / "lloyd"
    wt.mkdir(parents=True)
    monkeypatch.setattr(W, "worktree_path", lambda rid: wt)
    monkeypatch.setattr(W, "round_home", lambda rid: wt.parent)
    home = W.ensure_round_home("SM_x")
    assert (home / "obsidian").is_symlink()
    data = home / "lloyd-data"
    assert data.is_dir() and not data.is_symlink()
    assert not any(data.iterdir())
    assert W.round_data_root("SM_x") == data


def test_every_gate_child_gets_the_round_data_root_except_live_baselines(tmp_path, monkeypatch):
    from scripts.automod import gate as GT
    from scripts.automod import worktree as W
    monkeypatch.setattr(W, "round_dir", lambda rid: tmp_path / rid)
    monkeypatch.setattr(W, "round_home", lambda rid: tmp_path / rid / "home")
    g = GT.Gate.__new__(GT.Gate)
    g.round_id, g.worktree = "SM_y", tmp_path / "SM_y" / "home" / "lloyd"
    env = g._child_env()
    assert env["LLOYD_DATA"] == str(tmp_path / "SM_y" / "home" / "lloyd-data")
    assert Path(env["LLOYD_DATA"]).is_dir()
    assert "LLOYD_DATA" not in g._child_env(live_data=True)


def test_the_canary_gets_its_own_data_root(tmp_path):
    from scripts.automod import canary_config as CC
    rd = tmp_path / "SM_z"
    env = CC.canary_env(rd, tmp_path / "wt", overlay=tmp_path / "o.yaml",
                        python=Path("/usr/bin/python3"))
    assert env["LLOYD_DATA"] == str(rd / "canary-home" / "lloyd-data")
    assert CC.canary_data_root(rd) == rd / "canary-home" / "lloyd-data"


# ── the guardian's data tripwire ─────────────────────────────────────────────

def _data_root(root: Path, sessions: int = 400) -> Path:
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    for i in range(sessions):
        (root / "sessions" / f"2026_{i}.json").write_text("{}")
    (root / "logs").mkdir(exist_ok=True)
    (root / "workers.db").write_text("x")
    (root / DW.ROOT_MARKER).write_text("{}")
    return root


def test_the_watch_is_not_armed_before_there_is_anything_to_lose(tmp_path):
    w = DW.DataWatch(str(tmp_path / "lloyd-data"), tmp_path / "gstate")
    assert w.tick() == (None, None)
    assert not w.armed


def test_the_watch_trips_on_a_wipe_a_swap_and_a_lost_marker(tmp_path):
    root = _data_root(tmp_path / "lloyd-data")
    w = DW.DataWatch(str(root), tmp_path / "gstate")
    assert w.tick(now=1000.0)[0] is None and w.armed
    (root / DW.ROOT_MARKER).unlink()
    why, _ = w.tick(now=1005.0)
    assert why and "marker" in why

    root2 = _data_root(tmp_path / "d2")
    w2 = DW.DataWatch(str(root2), tmp_path / "g2")
    w2.tick(now=1000.0)
    for f in list((root2 / "sessions").iterdir())[:300]:
        f.unlink()
    why, _ = w2.tick(now=1005.0)
    assert why and "disappeared" in why


@pytest.fixture
def guardian(tmp_path, monkeypatch):
    root = _data_root(tmp_path / "lloyd-data")
    vault = tmp_path / "obsidian"
    vault.mkdir()
    monkeypatch.setattr(policy, "VAULT_ROOT", str(vault))
    monkeypatch.setattr(policy, "DATA_ROOT", str(root))
    monkeypatch.setattr(policy, "LOG_FILES", ())
    monkeypatch.setattr(policy, "REPO", str(tmp_path / "lloyd"))
    (tmp_path / "lloyd").mkdir()
    args = types.SimpleNamespace(
        repo=str(tmp_path / "lloyd"), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    acted: dict = {"workers": 0, "alerts": []}
    monkeypatch.setattr(g, "_stop_sync", lambda: "n/a")
    monkeypatch.setattr(g, "_pause_workers",
                        lambda: acted.__setitem__("workers", acted["workers"] + 1) or "paused")
    monkeypatch.setattr(g, "alert", lambda level, title, body, **kw:
                        acted["alerts"].append((level, title, body)))
    monkeypatch.setattr(g, "_vault_evidence", lambda stamp: None)
    return g, root, acted


def test_the_guardian_acts_on_a_data_wipe_once(guardian):
    g, root, acted = guardian
    g.check_data()
    assert not acted["alerts"]
    import shutil
    shutil.rmtree(root / "sessions")
    g.check_data()
    assert acted["workers"] == 1 and g.state.is_halted()
    [(level, title, body)] = acted["alerts"]
    assert level == "critical" and "restore-data.sh" in body
    g.check_data()
    assert acted["workers"] == 1


def test_the_guardian_names_runtime_data_that_came_back_into_the_tree(guardian):
    g, _, acted = guardian
    g.check_data()  # arms the watch
    (Path(policy.REPO) / "sessions").mkdir()
    g._strays_checked_at = 0.0
    g.check_data()
    [(level, title, body)] = acted["alerts"]
    assert level == "error" and "/sessions" in body


# ── snapshot, prune, restore ─────────────────────────────────────────────────

def _run(script, env, *args):
    return subprocess.run(["bash", str(ROOT / "scripts/backup" / script), *args],
                          capture_output=True, text=True, env={**os.environ, **env}, timeout=60)


def test_snapshots_refuse_a_tripped_or_unmarked_root(tmp_path):
    root = _data_root(tmp_path / "lloyd-data")
    gstate = tmp_path / "gstate"
    gstate.mkdir()
    env = {"LLOYD_DATA": str(root), "LLOYD_GUARDIAN_STATE": str(gstate),
           "LLOYD_DATA_SNAPSHOTS": str(tmp_path / "snaps")}
    (gstate / "data-tripped.json").write_text("{}")
    r = _run("snapshot-data.sh", env)
    assert r.returncode == 0 and "tripwire is set" in r.stdout
    (gstate / "data-tripped.json").unlink()
    (root / DW.ROOT_MARKER).unlink()
    r = _run("snapshot-data.sh", env)
    assert r.returncode == 0 and "REFUSING" in r.stdout
    assert not (tmp_path / "snaps").exists()


def test_pruning_keeps_48_hours_and_one_a_day_for_14_days(tmp_path):
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    names = [(now - dt.timedelta(hours=h)).strftime("%Y%m%dT%H%M%SZ") for h in range(0, 24 * 20, 6)]
    for n in names:
        (tmp_path / n).mkdir()
    r = _run("prune-data-snapshots.sh", {"DRY_RUN": "1"}, str(tmp_path))
    doomed = {line.rsplit("/", 1)[1] for line in r.stdout.splitlines() if line.startswith("would delete")}
    kept = set(names) - doomed
    assert all(n in kept for n in names[:8])            # last 48 h, every one
    old_days = {n[:8] for n in names if n not in names[:9]}
    kept_old = [n for n in kept if n not in names[:9]]
    assert len(kept_old) <= 14 and len({n[:8] for n in kept_old}) == len(kept_old)
    assert doomed and all(n[:8] in old_days for n in doomed)


def test_restore_refuses_the_live_root(tmp_path):
    live = tmp_path / "lloyd-data"
    live.mkdir()
    (tmp_path / "snaps" / "20260922T000000Z").mkdir(parents=True)
    r = _run("restore-data.sh", {"LLOYD_DATA": str(live), "LLOYD_DATA_SNAPSHOTS": str(tmp_path / "snaps")},
             "20260922T000000Z", str(live / "sub"))
    assert r.returncode == 1 and "refusing" in r.stderr


# ── the migration ────────────────────────────────────────────────────────────

def _tree(tree: Path) -> None:
    (tree / "sessions").mkdir(parents=True)
    (tree / "sessions" / "a.json").write_text('{"x": 1}')
    (tree / "_pipeline" / "vault-derived").mkdir(parents=True)
    kg = sqlite3.connect(tree / "_pipeline" / "vault-derived" / "kg.sqlite")
    kg.execute("create table t(x)"); kg.execute("insert into t values (1)"); kg.commit(); kg.close()
    db = sqlite3.connect(tree / "usage.db")
    db.execute("pragma journal_mode=wal"); db.execute("create table u(x)")
    db.executemany("insert into u values (?)", [(i,) for i in range(500)]); db.commit(); db.close()
    (tree / "logs").mkdir()
    (tree / "logs" / "server.err").write_text("e\n")
    (tree / "agent-services" / "logs").mkdir(parents=True)
    (tree / "agent-services" / "logs" / "agent-djev.log").write_text("d\n")
    (tree / "None").write_text("")
    (tree / "server.py").write_text("code")


def _migrate():
    sys.path.insert(0, str(ROOT / "scripts"))
    import migrate_data_home as M
    return M


def test_the_migration_moves_everything_verifies_it_and_holds_the_originals(tmp_path):
    M = _migrate()
    tree, data, hold = tmp_path / "lloyd", tmp_path / "lloyd-data", tmp_path / "hold"
    _tree(tree)
    data.mkdir()
    report = M.apply(tree, data, hold)
    assert (data / ".lloyd-data-root").is_file()
    assert json.loads((data / "sessions" / "a.json").read_text()) == {"x": 1}
    assert (data / "logs" / "services" / "agent-djev.log").read_text() == "d\n"
    con = sqlite3.connect(data / "usage.db")
    assert con.execute("select count(*) from u").fetchone()[0] == 500
    con.close()
    # Nothing runtime is left in the tree, the code is, the originals are held.
    assert sorted(p.name for p in tree.iterdir()) == ["agent-services", "server.py"]
    held = Path(report["hold"])
    assert (held / "sessions" / "a.json").exists() and (held / "None").exists()
    assert (held / "migration-report.json").exists()
    with pytest.raises(SystemExit, match="already carries"):
        M.apply(tree, data, hold)


def test_the_migration_refuses_while_a_process_holds_a_source_open(tmp_path):
    M = _migrate()
    tree, data = tmp_path / "lloyd", tmp_path / "lloyd-data"
    _tree(tree)
    data.mkdir()
    holder = subprocess.Popen([sys.executable, "-c",
                               "import sys,time; f=open(sys.argv[1],'a'); print('ok',flush=True); time.sleep(60)",
                               str(tree / "logs" / "server.err")], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ok"
        with pytest.raises(SystemExit, match="hold files"):
            M.apply(tree, data, tmp_path / "hold")
    finally:
        holder.kill()
        holder.wait()
    assert (tree / "logs" / "server.err").exists() and not (data / ".lloyd-data-root").exists()
