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

import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import app.paths as paths
from app import ww_diag

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


#: The stamp format `snapshot-data.sh` writes (`date -u +%Y%m%dT%H%M%SZ`) and
#: `prune-data-snapshots.sh` matches with `-name '20*T*Z'`.
SNAP_FMT = "%Y%m%dT%H%M%SZ"


def _snap_store(dirs: Path, *ages_seconds: float) -> list[str]:
    """Rebuild `dirs` holding one snapshot subvolume per age, returning the
    stamps in chronological order — newest last, which is the entry
    `ls | sort | tail -1` selects. Ages are seconds before now, so the age a
    reader computes matches the clock the check reads.
    """
    import datetime as dt
    import shutil
    now = dt.datetime.now(dt.timezone.utc)
    if dirs.is_dir():
        for p in dirs.iterdir():
            shutil.rmtree(p)
    dirs.mkdir(parents=True, exist_ok=True)
    stamps = []
    for age in ages_seconds:
        stamp = (now - dt.timedelta(seconds=age)).strftime(SNAP_FMT)
        (dirs / stamp).mkdir()
        stamps.append(stamp)
    return sorted(stamps)


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
    # A box that has been snapshotting: one fresh subvolume, so the snapshot
    # freshness check has a store to read and reports nothing on its own.
    monkeypatch.setattr(policy, "DATA_SNAPSHOTS", str(tmp_path / "snaps"))
    _snap_store(tmp_path / "snaps", 60.0)
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


# ── is the hourly snapshot still arriving? (backlog #1416) ───────────────────

TWO_DAYS = 2 * 24 * 3600.0


def test_a_stale_snapshot_store_alerts_once_and_a_fresh_one_is_quiet(guardian):
    """`snapshot-data.sh` refuses with `exit 0` and the `Type=oneshot` unit
    still reports `Result=success`, so neither the timer nor systemd can show a
    stopped snapshot stream. The data tick is the surface that can (#1416)."""
    g, _, acted = guardian
    snaps = Path(policy.DATA_SNAPSHOTS)
    g.check_data()
    assert not acted["alerts"]                       # a snapshot from a minute ago
    [stale] = _snap_store(snaps, TWO_DAYS)
    g._snapshots_checked_at = 0.0
    g.check_data()
    [(level, title, body)] = acted["alerts"]
    assert level == "error"
    assert stale in body and "2 d" in body           # names the stamp and its age
    assert str(snaps) in body                        # and the directory to look in
    g.check_data()
    assert len(acted["alerts"]) == 1                 # throttled, not every 5 s tick
    g._snapshots_checked_at = 0.0
    g.check_data()
    assert len(acted["alerts"]) == 2                 # and it keeps re-checking
    _snap_store(snaps, 60.0)                         # a snapshot arrives again
    g._snapshots_checked_at = 0.0
    g.check_data()
    assert len(acted["alerts"]) == 2                 # quiet from here on


def test_the_snapshot_alert_is_quiet_while_the_data_tripwire_is_set(guardian):
    """The refusal the tripwire causes is intended and already paged critical;
    a second alarm for the same incident is how an alert gets ignored."""
    g, _, acted = guardian
    _snap_store(Path(policy.DATA_SNAPSHOTS), TWO_DAYS)
    (g.gdir / "data-tripped.json").write_text(
        json.dumps({"reason": "900 of 1000 data files disappeared"}), encoding="utf-8")
    assert g.data.tripped()
    g._snapshots_checked_at = 0.0
    g.check_data()
    assert not acted["alerts"]
    (g.gdir / "data-tripped.json").unlink()
    g._snapshots_checked_at = 0.0
    g.check_data()
    assert [a[0] for a in acted["alerts"]] == ["error"]   # the marker was all that muted it


def test_an_empty_snapshot_directory_alerts_where_a_data_root_exists(guardian):
    """`prune-data-snapshots.sh` never deletes the newest snapshot whatever its
    age, so entry *count* can never report this: only the newest stamp against
    the clock separates an alive store from a deleted one."""
    g, _, acted = guardian
    snaps = Path(policy.DATA_SNAPSHOTS)
    _snap_store(snaps)                               # present, holding nothing
    g._snapshots_checked_at = 0.0
    g.check_data()
    [(level, title, body)] = acted["alerts"]
    assert level == "error" and str(snaps) in body


def test_the_snapshot_check_is_not_armed_on_a_box_without_a_data_root(guardian):
    """No marked data root means nothing to take snapshots of — the same
    not-armed rule `datawatch` applies to the tripwire, so a round home or an
    un-cut-over box is never paged for a missing snapshot store. The store is left
    stale on purpose: with the marker back it alerts, which says the marker guard
    is what muted it and not the state of the directory."""
    g, root, acted = guardian
    _snap_store(Path(policy.DATA_SNAPSHOTS), TWO_DAYS)
    marker = root / DW.ROOT_MARKER
    marker.unlink()
    g._snapshots_checked_at = 0.0
    g.check_data()
    assert not acted["alerts"]
    marker.write_text("{}", encoding="utf-8")               # the root is marked again
    g._snapshots_checked_at = 0.0
    g.check_data()
    assert [a[0] for a in acted["alerts"]] == ["error"]


def test_the_data_tripwire_alert_names_the_snapshot_directory_that_exists(guardian):
    """This alert exists for the one incident in which a human goes looking for
    a snapshot, and it named `/home/.lloyd-data-snapshots` — a path that has
    never existed on this machine."""
    g, root, acted = guardian
    g.check_data()
    import shutil
    shutil.rmtree(root / "sessions")
    g.check_data()
    [(level, title, body)] = acted["alerts"]
    assert level == "critical" and "restore-data.sh" in body
    assert str(policy.DATA_SNAPSHOTS) in body
    assert "/home/.lloyd-data-snapshots" not in body


def test_only_stamp_shaped_directories_count_as_snapshots(tmp_path):
    """Same shape `prune-data-snapshots.sh` matches (`-name '20*T*Z'`), so a
    stray file neither gets pruned nor silences the freshness check."""
    snaps = tmp_path / "snaps"
    (snaps / "not-a-snapshot").mkdir(parents=True)
    (snaps / "2026-09-22").mkdir()
    (snaps / "20260922T000000Z-not").mkdir()
    assert DW.newest_snapshot(str(snaps)) is None


def test_the_snapshot_report_and_its_cli_read_one_measurement(tmp_path):
    """The guardian's alert, the CLI and the restore listing all come from
    `snapshot_state`, so the age an operator prints cannot disagree with the age
    the watchdog alerted on."""
    snaps = tmp_path / "snaps"
    stamps = _snap_store(snaps, TWO_DAYS + 7200.0, TWO_DAYS)     # oldest first
    fresh, why = DW.snapshot_report(str(snaps), 3 * 3600.0)
    assert not fresh and "2 d 0 h" in why and str(snaps) in why
    fresh, why = DW.snapshot_report(str(snaps), 3 * 24 * 3600.0)
    assert fresh and "2 d 0 h" in why
    out = subprocess.run([sys.executable, str(GUARDIAN_DIR / "datawatch.py"),
                          "snapshot-age", "--tsv", str(snaps)],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    stamp, age, stale = out.stdout.strip().split("\t")
    assert stamp == stamps[-1] and age == "2 d 0 h" and stale == "1"


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


def _listing(tmp_path, snaps):
    return _run("restore-data.sh", {"LLOYD_DATA_SNAPSHOTS": str(snaps),
                                    "LLOYD_GUARDIAN_STATE": str(tmp_path / "gstate")})


def test_listing_snapshots_prints_the_age_of_the_newest_one(tmp_path):
    """The bare listing is how an operator checks what is restorable, and a list
    of names read as healthy after a week of silent refusals. The age printed
    here is `datawatch.age_text`, the same one the guardian alerts with."""
    snaps = tmp_path / "snaps"
    ten_days = 10 * 24 * 3600.0
    _, newest = _snap_store(snaps, ten_days + 3600.0, ten_days)
    r = _listing(tmp_path, snaps)
    assert r.returncode == 0, r.stderr
    assert newest in r.stdout
    assert f"newest {newest} is 10 d 0 h old" in r.stdout
    assert "stopped arriving" in r.stdout             # and what to do about it


def test_listing_a_fresh_snapshot_store_prints_its_age_without_an_alarm(tmp_path):
    snaps = tmp_path / "snaps"
    [newest] = _snap_store(snaps, 120.0)
    r = _listing(tmp_path, snaps)
    assert r.returncode == 0, r.stderr
    assert re.search(rf"newest {newest} is \d+ m old", r.stdout)
    assert "stopped arriving" not in r.stdout


def test_listing_an_empty_snapshot_directory_says_so(tmp_path):
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    r = _listing(tmp_path, snaps)
    assert r.returncode == 0, r.stderr
    assert "NO SNAPSHOTS" in r.stdout


def test_the_listing_ages_the_store_through_the_checkout_when_the_pinned_copy_is_older(tmp_path):
    """`restore-data.sh` asks the pinned copy first, because that is the module
    the running watchdog judged — and that copy is re-staged only at unit start
    (`ExecStartPre=guardian-stage.sh`), so a landed change can go minutes without
    the subcommand in it. A listing that printed no age during exactly that
    window would hide the staleness the age line exists to show."""
    snaps = tmp_path / "snaps"
    [newest] = _snap_store(snaps, TWO_DAYS)
    pinned = tmp_path / "gstate" / "bin"
    pinned.mkdir(parents=True)
    # What a pinned copy from before `snapshot-age` does with the command.
    (pinned / "datawatch.py").write_text(
        "import sys\n"
        "print('usage: datawatch.py status|clear|snapshot-gate|strays', file=sys.stderr)\n"
        "sys.exit(2)\n", encoding="utf-8")
    r = _run("restore-data.sh", {"LLOYD_DATA_SNAPSHOTS": str(snaps),
                                 "LLOYD_GUARDIAN_STATE": str(tmp_path / "gstate")})
    assert r.returncode == 0, r.stderr
    assert f"newest {newest} is 2 d 0 h old" in r.stdout
    assert "stopped arriving" in r.stdout


def test_the_pinned_copy_is_the_one_that_answers_when_it_can(tmp_path):
    """Order matters, not just the fallback: the pinned copy is the rule the
    running watchdog applies, so an operator reading a listing should see what
    that process would have alerted on, not what an unrelated checkout says."""
    snaps = tmp_path / "snaps"
    _snap_store(snaps, 60.0)                                # genuinely fresh
    pinned = tmp_path / "gstate" / "bin"
    pinned.mkdir(parents=True)
    (pinned / "datawatch.py").write_text(
        "import sys\nprint('PINNED\\t1 d\\t1')\n", encoding="utf-8")
    r = _run("restore-data.sh", {"LLOYD_DATA_SNAPSHOTS": str(snaps),
                                 "LLOYD_GUARDIAN_STATE": str(tmp_path / "gstate")})
    assert r.returncode == 0, r.stderr
    assert "newest PINNED is 1 d old" in r.stdout
    assert "is 0 m old" not in r.stdout              # the store's own fresh age did not answer


def test_an_age_line_is_never_silently_missing(tmp_path):
    """The fallback is tried too, and if neither copy can answer the listing says
    so instead of printing a bare list that reads like a checked one."""
    snaps = tmp_path / "snaps"
    _snap_store(snaps, TWO_DAYS)
    # A copy of the script in a directory with no checkout two levels above it:
    # the only way both candidate `datawatch.py` paths can be absent at once.
    lone = tmp_path / "sbin"
    lone.mkdir()
    script = lone / "restore-data.sh"
    script.write_text((ROOT / "scripts/backup/restore-data.sh").read_text(encoding="utf-8"),
                      encoding="utf-8")
    pinned = tmp_path / "gstate" / "bin"
    pinned.mkdir(parents=True)
    (pinned / "datawatch.py").write_text("import sys\nsys.exit(2)\n", encoding="utf-8")
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "LLOYD_DATA_SNAPSHOTS": str(snaps),
                            "LLOYD_GUARDIAN_STATE": str(tmp_path / "gstate")})
    assert r.returncode == 0, r.stderr
    assert "could not work out the newest snapshot's age" in r.stderr


def test_the_bash_listing_and_the_python_alert_default_to_the_same_directory(tmp_path):
    """Clause 4 says the alert must name the directory the running system uses,
    and the two resolve it independently: bash with
    `${LLOYD_DATA_SNAPSHOTS:-$HOME/.lloyd-data-snapshots}`, Python with
    `policy.DATA_SNAPSHOTS` (env, then `Path.home()`). Nothing but this test says
    those two defaults land in one place, and every other test passes a directory
    explicitly — so neither the agreement nor its failure would show up anywhere
    else."""
    home = tmp_path / "home"
    snaps = home / ".lloyd-data-snapshots"
    stamps = _snap_store(snaps, TWO_DAYS + 7200.0, TWO_DAYS)   # oldest first
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLOYD_")}
    env["HOME"] = str(home)

    def _py(code):
        return subprocess.run([sys.executable, "-c", code], cwd=str(GUARDIAN_DIR), env=env,
                              capture_output=True, text=True, timeout=60)

    listed = subprocess.run(["bash", str(ROOT / "scripts/backup/restore-data.sh")],
                            env=env, capture_output=True, text=True, timeout=60)
    assert listed.returncode == 0, listed.stderr
    assert f"newest {stamps[-1]}" in listed.stdout          # bash found the store by default
    assert "2 d 0 h" in listed.stdout

    resolved = _py("import policy; print(policy.DATA_SNAPSHOTS)")
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.strip() == str(snaps), (
        "policy.DATA_SNAPSHOTS and restore-data.sh's default snapshot directory "
        "disagreed: the alert would name a directory the timer does not write")

    env["LLOYD_DATA_SNAPSHOTS"] = str(tmp_path / "elsewhere")
    assert _py("import policy; print(policy.DATA_SNAPSHOTS)").stdout.strip() \
        == str(tmp_path / "elsewhere")                     # and the override agrees too


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


# ── the wake-miss diagnostic corpus (#1444) ────────────────────────────────
#
# `WakeMissCapture` wrote its corpus into a dot-directory under the account home
# instead of under the data root, and it was the one voice-produced path the
# 2026-09-22 move could not find by rewriting tree-relative paths, because it
# never lived in the tree. What that cost, each half reproduced by a test below:
#
#   * no snapshot — `scripts/backup/snapshot-data.sh` reads
#     `${LLOYD_DATA:-…/lloyd-data}` alone, and the box's other snapshotter
#     (snapper, one config, `SUBVOLUME="/"`) does not reach that dot-directory;
#   * outside the delete guard — `protected_roots()` protects the data root, so
#     `rm -r` over `<data root>/ww_diag` is refused while the same call over the
#     home's copy was allowed (the guard test is in test_protected_paths.py);
#   * no isolation under a gate — `scripts/automod/worktree.py:HOME_LINK_SKIP`
#     skips `lloyd` and `lloyd-data`, so the home's dot-directory *is* symlinked
#     into a round and candidate code appended to the live corpus. Under the root
#     a round gets `round_data_root`, which the gate exports as `LLOYD_DATA`.


CORPUS_READERS = [
    ("scripts/ww_diag_summary.py", "DEFAULT_PATH", "scores.jsonl"),
    ("scripts/voice/replay.py", "DIAG", None),
    ("scripts/voice/wake_eval.py", "DIAG", None),
    ("scripts/voice/hotword_eval.py", "DIAG", None),
]


def _import_fresh(rel: str, name: str):
    """Import one of this tree's scripts under a name no earlier test has used,
    so its module body re-runs against the environment the caller set up."""
    path = ROOT / rel
    assert path.is_file(), f"{rel} is not in the tree — the check found nothing"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_corpus_layout_is_resolved_once_and_hangs_off_the_data_root(tmp_path, monkeypatch):
    """`app.ww_diag` is the single expression of the layout, and the corpus is a
    *top-level folder* of the root — which is the level `protected_roots()`
    refuses to delete wholesale, so the guard covers it without a second list."""
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))
    corpus = tmp_path / ww_diag.CORPUS_DIR
    assert ww_diag.diag_dir() == corpus
    assert ww_diag.utterances_dir() == corpus / "utterances"
    assert ww_diag.misses_dir() == corpus / "misses"
    assert ww_diag.scores_path() == corpus / "scores.jsonl"
    assert ww_diag.labels_path() == corpus / "labels.jsonl"
    assert ww_diag.diag_dir().parent == ww_diag.data_root()


def test_wake_miss_capture_writes_under_the_data_root_not_the_home(tmp_path, monkeypatch):
    """Clause 1: with `LLOYD_DATA` set, constructing the rig creates the corpus
    there, an utterance lands in it, and the home's dot-directory is never
    created — the directory a round's home symlinks straight to the live corpus."""
    home, data = tmp_path / "home", tmp_path / "data"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LLOYD_DATA", str(data))
    if str(ROOT / "agent-services") not in sys.path:
        sys.path.append(str(ROOT / "agent-services"))
    import livekit_worker

    capture = livekit_worker.WakeMissCapture()
    assert capture.DIAG_DIR == data / ww_diag.CORPUS_DIR
    assert (capture.UTTERANCES_DIR).is_dir() and (capture.MISSES_DIR).is_dir()
    assert not (home / ".lloyd").exists()

    capture.record_utterance(
        utterance_id="tst", room="lloyd-test", identity="alan", duration_s=0.5,
        rms_mean=0.01, rms_peak=0.02, voiced_ratio=0.4, ww_ran=True,
        ww_name="lloyd", ww_score=0.1, ww_threshold=0.4, ww_fired=False,
        in_continuation=False, stt_text="go", stt_latency_s=0.1,
        samples=np.zeros(8000, dtype=np.int16), sample_rate=16000)

    wav = data / ww_diag.CORPUS_DIR / "utterances" / "tst.wav"
    assert wav.is_file()
    row = json.loads((data / ww_diag.CORPUS_DIR / "scores.jsonl").read_text().splitlines()[-1])
    assert row["utterance_id"] == "tst" and row["ww_fired"] is False
    assert Path(row["audio_path"]) == wav, "the record must point inside the corpus"
    assert not (home / ".lloyd").exists(), "nothing may be created under the home"


@pytest.mark.parametrize("rel,attr,child", CORPUS_READERS)
def test_each_corpus_reader_names_the_data_root_corpus(rel, attr, child, tmp_path, monkeypatch):
    """Clause 2: all four readers resolve the corpus through `app.ww_diag`, so
    `LLOYD_DATA` moves a reader together with the writer. Each is imported fresh
    with the variable set, because each binds the path at import."""
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))
    mod = _import_fresh(rel, f"corpus_reader_{Path(rel).stem}")
    got = getattr(mod, attr)
    want = tmp_path / ww_diag.CORPUS_DIR if child is None else tmp_path / ww_diag.CORPUS_DIR / child
    assert got == want, f"{rel} resolves {attr} to {got}, not {want}"
    assert "/.lloyd" not in str(got), f"{rel} still keeps a home-relative corpus path"


#: The pre-#1444 spelling, assembled rather than written: the clause-4 grep
#: forbids the literal in any tracked `.py`, and the sweep's own test file is a
#: tracked `.py`. Every use of this name goes through `_home_relative_hits`, so
#: the sweep and its positive control cannot drift into hunting different strings.
BANNED_SPELLING = ".lloyd" + "/ww_diag"


def _home_relative_hits(root: Path, rels: list[str]) -> list[str]:
    """The clause-4 sweep: which of `rels` carry the pre-move spelling."""
    return [rel for rel in rels
            if BANNED_SPELLING in (root / rel).read_text(encoding="utf-8", errors="replace")]


def test_no_python_file_in_the_repo_names_the_home_relative_corpus():
    """Clause 4, run as the clause's own grep: the corpus has one location, and
    the second spelling is what let the writer and four readers drift.

    The tracked set is the repo: `git ls-files`, because the tree also carries
    `.venvs/` and `qmd/`, and the denominator is printed beside the result — an
    empty listing would otherwise read as a clean sweep the same way a clean
    sweep does."""
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "*.py"],
                            capture_output=True, check=True).stdout
    tracked = [raw.decode() for raw in listed.split(b"\0") if raw]
    assert len(tracked) > 500, f"only {len(tracked)} tracked .py files: the check saw nothing"
    hits = _home_relative_hits(ROOT, tracked)
    assert hits == [], f"still naming the pre-#1444 location: {hits}"


def test_the_home_relative_sweep_catches_the_spelling_that_shipped(tmp_path):
    """The positive control clause 4 asks for, on the same helper the sweep above
    runs — so the empty result it reports is a sweep that works, not a sweep
    pointed at a string no file could ever contain.

    The caught line is the exact form the worker shipped until #1444
    (`Path("~/.lloyd` + `/ww_diag").expanduser()`), assembled here for the same
    reason the sweep assembles its pattern: the literal is what this pair of tests
    exists to keep out of the tree."""
    offender = tmp_path / "worker.py"
    offender.write_text('DIAG_DIR = Path("~/.lloyd' + '/ww_diag").expanduser()\n')
    other = tmp_path / "fine.py"
    other.write_text('DIAG = Path("~/lloyd-data' + "/ww_diag\").expanduser()\n")

    hits = _home_relative_hits(tmp_path, ["worker.py", "fine.py"])
    assert hits == ["worker.py"], (
        f"the sweep matched {hits}: it neither catches the shipped spelling nor "
        "leaves the data-root spelling alone, so the clean repo result means nothing")
    # And the file this control lives in is itself inside the swept set: the
    # assembly above is why that is possible.
    assert _home_relative_hits(ROOT, ["tests/test_data_home.py"]) == []


def test_a_tracked_runtime_name_is_not_a_stray_until_something_untracked_lands(tmp_path):
    """`eval/baselines/` holds committed measurement records; the hourly check
    alerted on the directory existing at all (2026-09-25). Tracked content is
    quiet, an untracked or ignored file beside it is a stray, and a runtime
    name git knows nothing about is a stray however empty."""
    tree = tmp_path / "tree"
    base = tree / "eval" / "baselines"
    base.mkdir(parents=True)
    (tree / ".gitignore").write_text("eval/baselines/*\n*.db\n")
    (base / "measured.json").write_text("{}")
    run = lambda *a: subprocess.run(["git", "-C", str(tree), *a], check=True,  # noqa: E731
                                    capture_output=True)
    run("init", "-q")
    run("add", "-f", ".gitignore", "eval/baselines/measured.json")
    assert DW.stray_in_tree(str(tree)) == []
    (base / "written-by-a-stray.json").write_text("{}")
    assert DW.stray_in_tree(str(tree)) == ["eval/baselines"]
    (base / "written-by-a-stray.json").unlink()
    (tree / "workers.db").write_bytes(b"")
    assert DW.stray_in_tree(str(tree)) == ["workers.db"]
