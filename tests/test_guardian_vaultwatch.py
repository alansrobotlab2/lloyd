"""The vault tripwire, the sync gate, and the snapshots behind them.

Twice — 2026-09-10 04:05 and 2026-09-12 11:28 — the vault was deleted from
inside Lloyd and `agent-obsidian-sync` pushed the deletions to the cloud.
Nothing was watching the vault between promotions, sync would start again over
any tree, and the only copy anyone could restore from was a tarball the
deleting process wrote itself.

Everything here runs against temp vaults and temp state: the guardian is
driven through `tick()` (the placement is the property, as with the log
cursor), the start script is run for real with a fake `ob`, and the backup
scripts are run for real against a temp git repository.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import guardian as G  # noqa: E402
import policy  # noqa: E402
import vaultwatch as VW  # noqa: E402


def _vault(root: Path, files: int = 400) -> Path:
    for i in range(files):
        folder = ("backlog", "skills", "memory", "knowledge")[i % 4]
        p = root / folder / f"note-{i}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"note {i}\n")
    (root / "MEMORY.md").write_text("index\n")
    return root


# ── the judgement ────────────────────────────────────────────────────────────

def test_evaluate_trips_on_each_wipe_shape_and_not_on_churn():
    base = VW.Snapshot(ts=0, total=5000, top={"backlog": 1100, "skills": 700, "images": 7}, inode=9)
    ok = VW.Snapshot(ts=5, total=4970, top={"backlog": 1090, "skills": 690, "images": 7}, inode=9)
    assert VW.evaluate([base], ok) is None
    assert "disappeared" in VW.evaluate([base], VW.Snapshot(ts=5, total=40, top={}, inode=9))
    assert "'skills'" in VW.evaluate(
        [base], VW.Snapshot(ts=5, total=4300, top={"backlog": 1100, "images": 7}, inode=9),
        drop_fraction=0.5)
    # A small folder going away is ordinary (7 < TOPDIR_MIN_FILES).
    assert VW.evaluate([base], VW.Snapshot(ts=5, total=4993,
                                           top={"backlog": 1100, "skills": 700}, inode=9)) is None
    assert "replaced" in VW.evaluate([base], VW.Snapshot(ts=5, total=5000, top=base.top, inode=10))
    assert "missing" in VW.evaluate([base], None)


def test_the_peak_is_taken_over_the_window_so_a_slow_wipe_still_trips():
    hist = [VW.Snapshot(ts=t, total=5000 - t * 20, top={}, inode=1) for t in range(0, 60, 5)]
    now = VW.Snapshot(ts=60, total=3700, top={}, inode=1)
    assert VW.evaluate(hist, now) is not None


def test_a_wipe_while_the_guardian_was_down_trips_on_restart(tmp_path):
    vault = _vault(tmp_path / "obsidian")
    state = tmp_path / "gstate"
    w = VW.VaultWatch(str(vault), state)
    assert w.tick()[0] is None
    w._persist(w.history[-1])
    shutil.rmtree(vault / "backlog")
    shutil.rmtree(vault / "skills")
    again = VW.VaultWatch(str(vault), state)          # a fresh process
    why, _ = again.tick()
    assert why and "disappeared" in why


# ── the guardian acts, from every state ──────────────────────────────────────

@pytest.fixture
def guardian(tmp_path, monkeypatch):
    vault = _vault(tmp_path / "obsidian")
    monkeypatch.setattr(policy, "VAULT_ROOT", str(vault))
    monkeypatch.setattr(G.policy, "VAULT_ROOT", str(vault))
    monkeypatch.setattr(policy, "LOG_FILES", ())
    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    acted: dict = {"sync": 0, "workers": 0, "alerts": []}
    monkeypatch.setattr(g, "_stop_sync", lambda: acted.__setitem__("sync", acted["sync"] + 1) or "stopped")
    monkeypatch.setattr(g, "_pause_workers",
                        lambda: acted.__setitem__("workers", acted["workers"] + 1) or "paused")
    monkeypatch.setattr(g, "alert", lambda level, title, body, **kw:
                        acted["alerts"].append((level, title, body)))
    monkeypatch.setattr(g, "ensure_chronic", lambda: None)
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    return g, vault, acted


@pytest.mark.parametrize("condition", ["supervisord_unreachable", "paused", "broken", "armed"])
def test_a_wipe_trips_whatever_state_the_tick_returns_from(guardian, monkeypatch, condition):
    g, vault, acted = guardian
    snap = {"now": 0.0, "supervisord": "unreachable" if condition == "supervisord_unreachable" else "ok",
            "procs": {}, "probes": {}}
    monkeypatch.setattr(g, "collect", lambda: snap)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 600.0 if condition == "paused" else 0.0)
    monkeypatch.setattr(g.state, "is_broken", lambda: condition == "broken")
    monkeypatch.setattr(g, "evaluate_liveness", lambda s: (False, "ok"))
    monkeypatch.setattr(g.state, "current", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=""))

    g.tick()
    assert acted["sync"] == 0
    for folder in ("backlog", "skills", "memory", "knowledge"):
        shutil.rmtree(vault / folder)
    g.tick()

    assert acted["sync"] == 1 and acted["workers"] == 1
    assert g.state.is_halted() if condition != "broken" else True
    marker = json.loads(g.vault.marker.read_text())
    assert "disappeared" in marker["reason"]
    assert marker["actions"]["sync"] == "stopped"
    [(level, title, body)] = acted["alerts"]
    assert level == "critical" and "sync stopped" in title
    assert "restore-vault.sh" in body and "clear" in body
    # Latched: the next tick does not act again.
    g.tick()
    assert acted["sync"] == 1


def test_evidence_is_captured(guardian):
    g, _, _ = guardian
    out = g._vault_evidence("20260914_000000")
    assert out and (Path(out) / "ps.txt").read_text().strip()


# ── the sync gate ────────────────────────────────────────────────────────────

def _run_start_script(tmp_path, vault: Path, state: Path) -> subprocess.CompletedProcess:
    fake = tmp_path / "ob"
    fake.write_text("#!/bin/bash\n"
                    "case \"$1\" in\n"
                    "  sync-list-local) echo /home/alansrobotlab/obsidian ;;\n"
                    "  sync) echo SYNC-STARTED ;;\n"
                    "esac\n")
    fake.chmod(0o755)
    env = dict(os.environ, OB=str(fake), HOME=str(tmp_path / "home"),
               LLOYD_VAULT_ROOT=str(vault), LLOYD_GUARDIAN_STATE=str(state),
               LLOYD_VAULTWATCH=str(GUARDIAN_DIR / "vaultwatch.py"))
    return subprocess.run(["bash", str(ROOT / "agent-services/bin/start-obsidian-sync.sh")],
                          capture_output=True, text=True, env=env, timeout=60)


def test_sync_will_not_start_while_tripped_or_over_a_gutted_vault(tmp_path):
    vault = _vault(tmp_path / "obsidian")
    state = tmp_path / "gstate"
    w = VW.VaultWatch(str(vault), state)
    w.tick()
    w._persist(w.history[-1])

    healthy = _run_start_script(tmp_path, vault, state)
    assert "SYNC-STARTED" in healthy.stdout, healthy.stderr

    w.trip("test trip", None, {})
    tripped = _run_start_script(tmp_path, vault, state)
    assert tripped.returncode != 0 and "SYNC-STARTED" not in tripped.stdout
    assert "tripwire is set" in tripped.stderr

    w.clear()
    for folder in ("backlog", "skills", "memory"):
        shutil.rmtree(vault / folder)
    gutted = _run_start_script(tmp_path, vault, state)
    assert gutted.returncode != 0 and "SYNC-STARTED" not in gutted.stdout


# ── snapshots ────────────────────────────────────────────────────────────────

def _backup(tmp_path, vault, repo, marker) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=str(tmp_path / "home"), LLOYD_VAULT_ROOT=str(vault),
               LLOYD_VAULT_BACKUP_REPO=str(repo), LLOYD_VAULT_TRIP_MARKER=str(marker))
    return subprocess.run([str(ROOT / "scripts/backup/backup-vault.sh")],
                          capture_output=True, text=True, env=env, timeout=120)


def test_snapshots_record_changes_refuse_a_shrunken_vault_and_restore_aside(tmp_path):
    vault = _vault(tmp_path / "obsidian", files=120)
    (vault / ".git").mkdir()
    (vault / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (vault / ".gitignore").write_text("*.md\n")                  # the backup keeps them anyway
    repo = tmp_path / "state" / "vault.git"
    marker = tmp_path / "gstate" / "vault-tripped.json"

    first = _backup(tmp_path, vault, repo, marker)
    assert first.returncode == 0 and "committed" in first.stdout, first.stdout + first.stderr
    (vault / "backlog" / "note-0.md").write_text("edited\n")
    assert "committed" in _backup(tmp_path, vault, repo, marker).stdout

    for folder in ("backlog", "skills"):
        shutil.rmtree(vault / folder)
    refused = _backup(tmp_path, vault, repo, marker)
    assert refused.returncode == 0 and "REFUSING" in refused.stdout

    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert "tripwire is set" in _backup(tmp_path, vault, repo, marker).stdout

    env = dict(os.environ, HOME=str(tmp_path / "home"), LLOYD_VAULT_BACKUP_REPO=str(repo))
    dest = tmp_path / "restore" / "obsidian"
    restored = subprocess.run([str(ROOT / "scripts/backup/restore-vault.sh"), "HEAD", str(dest)],
                              capture_output=True, text=True, env=env, timeout=120)
    assert restored.returncode == 0, restored.stderr
    assert (dest / "backlog" / "note-0.md").read_text() == "edited\n"
    assert not (dest / ".git").exists(), "the vault's own .git was copied into the backup"

    live = tmp_path / "home" / "obsidian"
    live.mkdir(parents=True)
    into_live = subprocess.run([str(ROOT / "scripts/backup/restore-vault.sh"), "HEAD", str(live)],
                               capture_output=True, text=True, env=env, timeout=60)
    assert into_live.returncode != 0 and "refusing" in into_live.stderr
