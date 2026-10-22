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
import time
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

# ── #3154: the owner of the pause sink ───────────────────────────────────────
#
# The trip below is the deployed guardian's own behaviour, with the pause going over a
# real socket to a real server: `pause_journal` id 21 held dispatch for 7 h 01 m on
# 2026-10-21 on a trip the deployed unit's journal never logged, because a COPY of
# `guardian.py` under a gate tree posted to a URL literal no caller could move. The two
# facts that make a copy a non-owner are the module's own path and the guardian state
# dir, so the nodes here arrange both instead of patching the check away — clause 4's
# claim is that the owner still pauses the fleet, and a node that stubbed the guard
# cannot tell an owner from a refusal.


def _recording_backend(tmp_path: Path):
    """A one-shot HTTP server that records what it was POSTed, as a context manager.

    A real socket rather than a faked `urlopen`, because clause 4 is about bytes that
    leave the process: `holder_kind` has to be absent ON THE WIRE, and the assertion
    that matters most — that a pause reached a backend at all — is a backend's answer.
    It binds port 0 and hands back the URL, so a node can point a sink at it without
    ever aiming a test POST at `127.0.0.1:8080`, which is the live fleet.
    """
    import http.server
    import threading

    hits: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            hits.append({"path": self.path, "method": "POST",
                         "body": self.rfile.read(n).decode(),
                         "content_type": self.headers.get("Content-Type")})
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):  # silence: a suite run is not a server log
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address[:2]
    return srv, hits, f"http://{host}:{port}"


def _owner_guardian(args, monkeypatch, tmp_path: Path, name: str = "watchdog"):
    """A `Guardian` built from a copy of this tree's `guardian.py` IN the deployed shape.

    The copy's file goes to `<tmp>/<name>/bin/guardian.py` — the directory
    `guardian-stage.sh:31` promotes into and `lloyd-guardian.service:29`'s `ExecStart`
    runs from — and `policy.GUARDIAN_STATE` is pointed at that state dir, so
    `policy.deployed_watchdog_refusal` answers "owner" for it (#3154). Everything else
    about it is the shipped class: no patched helper, no patched check.
    """
    import importlib.util

    state = tmp_path / name
    (state / "bin").mkdir(parents=True, exist_ok=True)
    dst = state / "bin" / "guardian.py"
    dst.write_bytes((GUARDIAN_DIR / "guardian.py").read_bytes())
    monkeypatch.setattr(policy, "GUARDIAN_STATE", state)
    spec = importlib.util.spec_from_file_location("guardian_watchdog_copy", dst)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    added = str(state / "bin")
    try:
        spec.loader.exec_module(mod)
    finally:
        # The copy's own `sys.path.insert(0, <its dir>)` runs at import; leaving a tmp
        # directory ahead of the real one would let a later `import guardian` in this
        # process load a file under a deleted temp root.
        sys.path[:] = [p for p in sys.path if p != added]
    return mod, mod.Guardian(args)


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
    acted: dict = {"sync": 0, "workers": 0, "alerts": [], "pause_actors": []}
    monkeypatch.setattr(g, "_stop_sync", lambda: acted.__setitem__("sync", acted["sync"] + 1) or "stopped")

    def _pause(actor):
        # #2846: the trip passes its own name, and what it passed is kept, because
        # `pause_journal` answers "who held the fleet" with exactly this string.
        acted["pause_actors"].append(actor)
        acted["workers"] += 1
        return "paused"

    monkeypatch.setattr(g, "_pause_workers", _pause)
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
    # The one alert the tripwire raised, and only the tripwire's: the blanket
    # `subprocess.run` stub above also starves the vault snapshot reading #3132 put
    # in the same method, which answers `returncode=0, stdout=""` with "no readable
    # HEAD commit" — a stall page with a different title, correct for that lie and
    # not what this node is about. What a second alarm in `check_vault` does to the
    # trip is pinned by `test_a_broken_snapshot_reading_cannot_stop_the_vault_tripwire`.
    [(level, title, body)] = [a for a in acted["alerts"] if a[1] == G.VAULT_TRIP_ALERT_TITLE]
    assert level == "critical" and "sync stopped" in title
    assert "restore-vault.sh" in body and "clear" in body
    # Latched: the next tick does not act again.
    g.tick()
    assert acted["sync"] == 1


def test_evidence_is_captured(guardian):
    g, _, _ = guardian
    out = g._vault_evidence("20260914_000000")
    assert out and (Path(out) / "ps.txt").read_text().strip()


FOUR_HOURS = 4 * 3600.0


def _backup_repo(dirs: Path, name: str = "vault", *, age_seconds: float) -> tuple[Path, str]:
    """A backup-shaped bare repo whose HEAD commit is `age_seconds` old, and its sha.

    Shaped like what `backup-vault.sh` writes — a bare repo driven by
    `--git-dir` with a separate `--work-tree` — because the age that matters is
    git's `%ct`, the one stamp `snapshot_age_state` reads and the one that
    survives the repo being copied. The planted epoch is read back out of git
    below: a node that planted nothing and passed anyway is the failure this
    helper exists to refuse.
    """
    def git(*args: str, env: "dict | None" = None) -> str:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           env=dict(os.environ, **(env or {})))
        assert r.returncode == 0, f"{args}: {r.stdout}{r.stderr}"
        return r.stdout.strip()

    repo, work = dirs / f"{name}.git", dirs / f"{name}-worktree"
    work.mkdir(parents=True)
    (work / "note.md").write_text("a note\n")
    git("init", "--quiet", "--bare", str(repo))
    git("--git-dir", str(repo), "--work-tree", str(work), "add", "--all", ".")
    stamp = int(time.time() - age_seconds)
    git("--git-dir", str(repo), "--work-tree", str(work), "commit", "--quiet", "--no-verify",
        "-m", "snapshot", env={"GIT_AUTHOR_DATE": f"@{stamp} +0000",
                               "GIT_COMMITTER_DATE": f"@{stamp} +0000"})
    stored = git("--git-dir", str(repo), "log", "-1", "--format=%ct")
    assert int(stored) == stamp, f"git stored committer time {stored}, not the planted {stamp}"
    return repo, git("--git-dir", str(repo), "log", "-1", "--format=%H")


#: The owner directory's path relative to a home, spelled by the module under test so
#: a node that plants a repository "in the owner position" is naming the same layout
#: `backup-vault.sh` writes and `lloyd-vault-backup.timer` fires, not a guess at it.
OWNER_REL = str(VW.SNAPSHOT_OWNER_DIR_REL)


def _owner_dir(tmp_path: Path, monkeypatch) -> Path:
    """The owner directory this test owns: present, scratch, and named by the code.

    Redirected by replacing `vaultwatch.snapshot_owner_dir` rather than by setting
    `HOME`, which is the same choice
    `test_owner_check_does_not_silence_a_stale_vault_snapshot_alert` below makes and for
    the same reason it has to be: the gate runs this file under a child whose `HOME` it
    set itself (`scripts/automod/gate.py::_child_env`), and an `expanduser`-backed
    `HOME` swapped mid-test is not what the reading then resolves — a node that moved
    `$HOME` and asserted on the alert that follows was measuring the fixture, which is
    the exact mistake #3149's own trap clause warns about. What the code asks is
    "is this repository under THIS process's owner directory", so the node states the
    owner directory. Its `$HOME/.local/state/lloyd-vault-backup` SHAPE is not left
    unpinned: `SNAPSHOT_OWNER_DIR_REL` is the constant `SNAPSHOT_REPO` is built from,
    and `test_a_directory_that_only_shares_the_owner_name_is_not_the_owner` pins what
    lies inside it.
    """
    owner = tmp_path / "owner" / OWNER_REL
    owner.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(VW, "snapshot_owner_dir", lambda: owner)
    return owner


def _snapshot_override(monkeypatch, owner: Path, repo: Path) -> None:
    """Point the reading's override at `repo`, which sits in `owner`.

    The override is the one `backup-vault.sh` shares, so the check and the writer name
    one repository without a config file. It is the ONLY knob these nodes move: the
    2026-10-21 emitter set exactly this one variable, and its one-variable override
    carried `~/lloyd-work/.t/nonexistent-vault.git` into three sections of the live
    daily note.
    """
    monkeypatch.setenv("LLOYD_VAULT_BACKUP_REPO", str(repo))


def _owner_backup_repo(tmp_path: Path, monkeypatch, name: str, *,
                       age_seconds: float) -> tuple[Path, str]:
    """Plant a repo IN the owner position and point the override at it: `(repo, sha)`.

    The owner position is what makes a stale HEAD here an alert rather than a
    fixture — #3132's alarm has to keep firing there (#3149 clause 4), and the same
    HEAD age one directory away must not (#3149 clause 3).
    """
    owner = _owner_dir(tmp_path, monkeypatch)
    repo, sha = _backup_repo(owner, name, age_seconds=age_seconds)
    _snapshot_override(monkeypatch, owner, repo)
    return repo, sha


def test_the_deployed_shape_still_halts_everything_and_posts_its_body(
        tmp_path, monkeypatch):
    """Clause 4/5: the owner's trip is unchanged, byte for byte, and this node is where
    the wire proves it — a real socket, a real server, no faked `urlopen`.

    A vault trip keeps every action it ever took (#2846 clause 5) and still pauses the
    pool: `guardian.py` resolves under its state dir's `bin/`, the sink is aimed at a
    test server on a free port (never `127.0.0.1:8080`, which is the live fleet), and the
    server records exactly one POST carrying `{paused: true, actor:
    guardian-vault-trip}` and NO `holder_kind` key — that omission is load-bearing and
    documented at `guardian.py` above the helper: naming a kind would let the next backend
    restart resume dispatch in the middle of the incident.

    The owner shape is ARRANGED, not patched: the copy's file and the state dir are the
    two facts `policy.deployed_watchdog_refusal` weighs, so a node that overrode the
    check would be reporting that the guard passed while never having consulted it. The
    marker side is read back from disk because the alert body is a rendering: `actions`
    is what the next reader of the trip marker gets.
    """
    vault = _vault(tmp_path / "obsidian")
    monkeypatch.setattr(policy, "VAULT_ROOT", str(vault))
    monkeypatch.setattr(policy, "LOG_FILES", ())
    srv, hits, base = _recording_backend(tmp_path)
    try:
        args = types.SimpleNamespace(
            repo=str(tmp_path), state=str(tmp_path / "state"),
            guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
            backend_url=base + "/health", mcp_url="http://127.0.0.1:2/health",
            programs="lloyd-mc:lloyd-backend", interval=5.0,
        )
        _mod, g = _owner_guardian(args, monkeypatch, tmp_path)
        alerts: list = []
        monkeypatch.setattr(g, "_stop_sync", lambda: "stopped")
        monkeypatch.setattr(g, "alert",
                            lambda level, title, body, **kw:
                            alerts.append((level, title, body)))
        monkeypatch.setattr(g, "ensure_chronic", lambda: None)
        monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)

        g.check_vault()                                # arm the tripwire
        for folder in ("backlog", "skills", "memory", "knowledge"):
            shutil.rmtree(vault / folder)
        g.check_vault()

        assert len(hits) == 1, (
            f"the owner's trip produced {len(hits)} pause POST(s), not exactly one")
        assert hits[0]["path"] == policy.WORKERS_PAUSE_PATH, (
            f"the pause went to {hits[0]['path']}, not {policy.WORKERS_PAUSE_PATH}")
        posted = json.loads(hits[0]["body"])
        assert posted == {"paused": True, "actor": "guardian-vault-trip"}, (
            f"the owner posted {posted}: the body changed, or the actor or a "
            "`holder_kind` appeared in it")
        assert "holder_kind" not in posted, (
            "a `holder_kind` is back in the trip's body, so the next backend restart "
            "would resume dispatch mid-incident")

        marker = json.loads(g.vault.marker.read_text(encoding="utf-8"))
        assert marker["actions"]["workers"].startswith("paused (HTTP "), (
            f"the trip marker records {marker['actions']['workers']!r}, not a pause the "
            "backend answered — a reader would see a halt that did not happen")
        assert marker["actions"]["promotions"] == "halted", marker["actions"]
        assert "sync stopped" in alerts[0][1], alerts[0][1]
        assert f"Worker pool: {marker['actions']['workers']}" in alerts[0][2], (
            "the alert no longer quotes the pause it took, so whoever reads it cannot "
            "tell a held fleet from a backend that refused the hold")
        assert "Promotions: halted" in alerts[0][2], alerts[0][2]
        assert "restore-vault.sh" in alerts[0][2] and "clear" in alerts[0][2], (
            "#2846's recovery sentence left the trip alert: the reader is told an "
            "incident happened but not how to undo it")
        # Clause 5: everything else the trip does is where it was.
        assert (Path(marker["actions"]["evidence"]) / "ps.txt").read_text().strip(), (
            "the trip no longer captures the process table")
    finally:
        srv.shutdown()


def test_the_trip_prints_the_same_resume_line_and_selftest_still_passes(
        tmp_path, monkeypatch, capfd):
    """Clause 5, the other two surfaces a confinement check could have moved.

    `vaultwatch.py clear` prints the curl a HUMAN runs to lift a real halt. It is a
    literal, not derived from the guardian's sink, and it has to stay that way: a person
    reading it has the vault checkout, not the round's environment. And `selftest.py`'s
    `vault tripwire judges and measures` node is what `guardian-stage.sh` gates promotion
    on, so a check that made it fail would keep a fixed watchdog from ever being staged —
    the exact failure #3149's round hit on the note sink. It is run as a subprocess here
    because that node only fails or passes in a real interpreter with a real home.
    """
    # A subprocess against a scratch vault, because `VW.main` builds its `VaultWatch`
    # from the module defaults — which in-process would be the LIVE vault and the live
    # state dir, and clearing the deployed tripwire from a test is exactly the write
    # this item's class is about. The CLI resolves both from the environment, so a
    # confined run is the human surface, honestly confined.
    vault = _vault(tmp_path / "obsidian")
    state = tmp_path / "gstate"
    w = VW.VaultWatch(str(vault), state)
    w.trip("400 of 401 vault files disappeared", None, {"workers": "paused (HTTP 200)"})
    env = dict(os.environ,
               LLOYD_VAULT_ROOT=str(vault), LLOYD_GUARDIAN_STATE=str(state))
    cleared = subprocess.run([sys.executable, str(GUARDIAN_DIR / "vaultwatch.py"),
                              "clear"], capture_output=True, text=True, timeout=60,
                             env=env, check=False)
    assert cleared.returncode == 0, cleared
    printed = cleared.stdout
    resume_curl = ("curl -X POST localhost:8080/api/workers/pause "
                   "-H 'content-type: application/json' "
                   "-d '" + '{"paused": false}' + "'")
    assert resume_curl in printed, (
        f"`clear` no longer prints the resume curl a human needs: {printed!r}")

    scratch = tmp_path / "scratch-home"
    (scratch / "obsidian" / "memory").mkdir(parents=True)
    (scratch / "obsidian" / "memory" / "note.md").write_text("# note\n",
                                                            encoding="utf-8")
    env = dict(os.environ,
               HOME=str(scratch),
               LLOYD_GUARDIAN_STATE=str(scratch / ".local/state/lloyd-guardian"),
               LLOYD_AUTOMOD_STATE=str(scratch / ".local/state/lloyd-automod"),
               LLOYD_DATA=str(tmp_path / "data"),
               LLOYD_SUPERVISOR_SOCK=str(tmp_path / "no-supervisor.sock"),
               LLOYD_VOICE_ALERTS="0", LLOYD_DESKTOP_ALERTS="0",
               LLOYD_JOURNAL_ALERTS="0")
    out = subprocess.run([sys.executable, str(GUARDIAN_DIR / "selftest.py"),
                          "--profile", "staging"],
                         capture_output=True, text=True, timeout=120, env=env,
                         check=False)
    line = [ln for ln in out.stdout.splitlines()
            if "vault tripwire judges and measures" in ln]
    assert line, f"the selftest printed no vault-tripwire line at all: {out.stdout!r}"
    assert "[ok ]" in line[0], (
        f"`guardian-stage.sh` would now refuse a candidate over #3154's owner check: "
        f"{line[0]}")


def test_a_stale_vault_backup_alerts_once_and_a_fresh_one_is_quiet(guardian, tmp_path,
                                                                   monkeypatch):
    """The alarm itself, once, with the numbers a person needs in its body.

    `backup-vault.sh` refuses with `exit 0` by design and the timer is a
    `Type=oneshot`, so neither systemd nor the timer can show a stopped snapshot
    stream; the vault tick is the surface that can. The same node holds the
    throttle, because the tick runs every 5 s and a standing condition that
    re-alerts every 5 s is an alarm nobody reads (#3132).

    Both repositories sit IN the owner directory (`$HOME/.local/state/
    lloyd-vault-backup`), which is where #3149 leaves this alarm: an owner check in
    front of the alert is only safe to add if a stale HEAD at the owner's own path
    still pages, so this is the node that says so. Its counterpart,
    `test_a_repo_outside_the_owner_directory_raises_no_vault_snapshot_alert`, holds
    the same HEAD age one directory away and expects silence.
    """
    g, _, acted = guardian
    fresh, _fresh_sha = _owner_backup_repo(tmp_path, monkeypatch, "fresh", age_seconds=60.0)
    g.check_vault()
    assert not acted["alerts"], acted["alerts"]          # a snapshot a minute old is fine

    stale, sha = _owner_backup_repo(tmp_path, monkeypatch, "stale", age_seconds=FOUR_HOURS)
    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    [(level, title, body)] = acted["alerts"]
    assert (level, title) == ("error", G.VAULT_SNAPSHOT_ALERT_TITLE), acted["alerts"]
    assert str(stale) in body, body                      # the repo to go and look at
    assert sha[:8] in body, body                         # the commit it is standing on
    assert "4 h 0 m old" in body, body                   # its age, not just "stale"
    assert "past the 3 h limit" in body, body            # and the bound it crossed
    assert "lloyd-vault-backup.service" in body, body    # where the failing runs are

    g.check_vault()
    assert len(acted["alerts"]) == 1                     # throttled, not every 5 s tick

    g._alert_seen.clear()                                # the fan-out's own 15-minute
    g._vault_snapshots_checked_at = 0.0                  # repeat window, and the
    g.check_vault()                                      # check's own throttle
    assert len(acted["alerts"]) == 2                     # a stall standing for an
    assert g._vault_snapshots_checked_at > 0.0           # hour keeps being reported

    later, _s2 = _owner_backup_repo(tmp_path, monkeypatch, "later", age_seconds=60.0)  # delivering again
    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    assert len(acted["alerts"]) == 2                     # quiet from here on


def test_the_vault_snapshot_alert_is_quiet_while_the_vault_tripwire_is_set(guardian,
                                                                            tmp_path,
                                                                            monkeypatch):
    """A stalled HEAD during a trip is intended behaviour, and the trip already
    paged critical: `backup-vault.sh:57` refuses to snapshot while
    `vault-tripped.json` exists and exits 0. A second alarm for the same incident
    is how an alert gets ignored — the data root's twin mutes the same way
    (`tests/test_data_home.py:371`)."""
    g, _, acted = guardian
    stale, _sha = _owner_backup_repo(tmp_path, monkeypatch, "stale", age_seconds=FOUR_HOURS)
    g.vault.marker.write_text(json.dumps({"reason": "200 of 401 vault files disappeared"}),
                              encoding="utf-8")
    assert g.vault.tripped()
    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    assert not acted["alerts"], acted["alerts"]
    g.vault.marker.unlink()
    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    assert [a[1] for a in acted["alerts"]] == [G.VAULT_SNAPSHOT_ALERT_TITLE], acted["alerts"]


def test_a_backup_repo_with_no_readable_head_alerts_as_the_worst_age(guardian, tmp_path,
                                                                     monkeypatch):
    """The data twin's empty-store node (`tests/test_data_home.py:388`) mirrored
    onto a git repo: pruning never deletes the newest snapshot, so entry COUNT
    cannot separate an alive store from a dead one, and `git log` answering
    `fatal: not a git repository` for a path that does not resolve must not read
    as green. Both shapes are checked — a repo the timer never created, and a
    bare repo with no commit — because both are a vault with no off-vault copy.

    Both sit in the owner directory, because that is the only place their silence
    would mean anything: a missing repo is the worst age THERE is, so #3149's owner
    check may not turn it into a shrug. The same shape outside the owner directory is
    `test_a_repo_outside_the_owner_directory_raises_no_vault_snapshot_alert`."""
    g, _, acted = guardian
    owner = _owner_dir(tmp_path, monkeypatch)
    missing = owner / "never-created.git"
    _snapshot_override(monkeypatch, owner, missing)
    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    [(level, title, body)] = acted["alerts"]
    assert (level, title) == ("error", G.VAULT_SNAPSHOT_ALERT_TITLE), acted["alerts"]
    assert "no readable HEAD commit" in body, body
    assert str(missing) in body, body
    assert "not the vault's backup owner" not in body, body   # it IS the owner's repo

    g._alert_seen.clear()                                # the alert's own repeat window
    subprocess.run(["git", "init", "--quiet", "--bare", str(owner / "no-commits.git")],
                   check=True, capture_output=True)
    _snapshot_override(monkeypatch, owner, owner / "no-commits.git")
    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    assert len(acted["alerts"]) == 2, acted["alerts"]


def test_a_broken_snapshot_reading_cannot_stop_the_vault_tripwire(guardian, tmp_path,
                                                                  monkeypatch):
    """`check_vault` must never raise into the tick, and the snapshot read is new
    code inside it: a `git` that hangs the box, a module that fails to import, a
    repo on a mount that went away. The tripwire is the reason — it stops sync
    within one tick of a wipe, and a check bolted above it may not cost that. The
    reading here raises on purpose, and the wipe below still has to trip.

    The override points INSIDE the owner directory so that the raised exception is the
    only thing that can keep this tick quiet: pointed anywhere else it would be
    silenced by #3149's owner check and the node would pass without ever reaching the
    `except` it is about."""
    g, vault, acted = guardian
    owner = _owner_dir(tmp_path, monkeypatch)
    _snapshot_override(monkeypatch, owner, owner / "gone.git")

    def unreadable(*_a, **_k):
        raise RuntimeError("simulated git failure")

    monkeypatch.setattr(VW, "snapshot_age_state", unreadable)
    g.check_vault()                                      # does not raise
    assert not acted["alerts"], acted["alerts"]
    for folder in ("backlog", "skills", "memory", "knowledge"):
        shutil.rmtree(vault / folder)
    g.check_vault()
    assert acted["sync"] == 1 and acted["workers"] == 1
    assert [a[1] for a in acted["alerts"]] == [G.VAULT_TRIP_ALERT_TITLE], acted["alerts"]


def test_the_vault_snapshot_check_runs_from_the_tick_even_while_paused(guardian, monkeypatch,
                                                                       tmp_path):
    """Placement, not just existence: `tick()` calls `check_vault()` above every
    early return, and `paused` is the state a stalled backup is most likely to be
    missed in — the vault trip's own pause, or a promotion window, suppresses
    actions for minutes at a time. Driving `tick()` is what pins that the reading
    happens in the part of the loop that always runs (#3132)."""
    g, _, acted = guardian
    stale, _sha = _owner_backup_repo(tmp_path, monkeypatch, "stale", age_seconds=FOUR_HOURS)
    monkeypatch.setattr(g, "collect", lambda: {"now": time.time(), "supervisord": "ok",
                                               "procs": {}, "probes": {}})
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 600.0)
    monkeypatch.setattr(g.state, "is_broken", lambda: False)
    monkeypatch.setattr(g.state, "current", lambda: None)
    monkeypatch.setattr(g, "evaluate_liveness", lambda s: (False, "ok"))
    real_run = subprocess.run

    def stubbed(cmd, *a, **k):
        # The tick's other shell-outs (a systemctl that must not run here, a `df`
        # whose numbers are not what this node is about) are silenced; the vault
        # snapshot read is the one command allowed to reach the machine, because
        # faking it would leave the placement untested.
        if isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]).endswith("git"):
            return real_run(cmd, *a, **k)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", stubbed)
    assert g.tick() == "paused"
    assert [a[1] for a in acted["alerts"]] == [G.VAULT_SNAPSHOT_ALERT_TITLE], acted["alerts"]


def test_a_repo_outside_the_owner_directory_raises_no_vault_snapshot_alert(guardian,
                                                                          tmp_path,
                                                                          monkeypatch):
    """#3149 clause 3: the stale-snapshot alert may only name a repo the backup owns.

    The 2026-10-21 emitter is rebuilt here as it ran: a real stale HEAD in a repo
    parked under a throwaway-checkout root (`~/lloyd-work/.t/`-shaped, and named
    `nonexistent-vault.git`, the basename that reached the live daily note), reached
    through the one knob that caller turned — `LLOYD_VAULT_BACKUP_REPO`, with the
    other knob left wherever the process happened to have it.

    What those three sections prove is that the reading itself cannot be trusted with
    a page. It measured a genuine 4-hour-old HEAD and reported it accurately, and the
    repository it measured was a fixture. So silence is the property here, while the
    age stays measured and stays printed — the operator's `snapshot-age` CLI reads the
    same string and a person who pointed the override at a repo deserves an answer
    about that repo — with the sentence naming whose repository it is NOT.

    The last block is what makes this node able to fail: the identical HEAD age placed
    IN the owner directory pages on the same guardian, so the silence above is the
    owner check and not a throttled tick, a tripped vault, or a plant that planted
    nothing.
    """
    g, _, acted = guardian
    owner = _owner_dir(tmp_path, monkeypatch)
    throwaway = tmp_path / "lloyd-work" / ".t"
    stale, _sha = _backup_repo(throwaway, "nonexistent-vault", age_seconds=FOUR_HOURS)
    _snapshot_override(monkeypatch, owner, stale)

    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    assert not acted["alerts"], acted["alerts"]

    state = VW.snapshot_age_state(str(stale), VW.SNAPSHOT_MAX_AGE_SECONDS)
    assert state["owned"] is False, state
    assert state["fresh"] is False, state                 # the age is still measured
    assert "4 h 0 m old" in state["text"], state["text"]
    assert "not the vault's backup owner" in state["text"], state["text"]
    assert str(VW.snapshot_owner_dir()) in state["text"], state["text"]

    g.check_vault()                                       # a standing condition stays
    assert not acted["alerts"], acted["alerts"]           # quiet, not once-per-throttle

    _owner_backup_repo(tmp_path, monkeypatch, "stale", age_seconds=FOUR_HOURS)
    g._alert_seen.clear()
    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    assert [a[1] for a in acted["alerts"]] == [G.VAULT_SNAPSHOT_ALERT_TITLE], acted["alerts"]


def test_a_directory_that_only_shares_the_owner_name_is_not_the_owner(tmp_path):
    """The check is `relative_to`, so a sibling whose name merely starts with the
    owner's buys nothing.

    A confinement test written as a string prefix is a confinement test that can be
    walked past by naming a directory, which is the class of guard #2110 names:
    `lloyd-vault-backup-evil` begins with `lloyd-vault-backup` and is not in it. The
    first line below is the positive control — a repo genuinely inside the owner
    directory it is weighed against — so the three assertions differ only in the
    characters a prefix would have ignored.
    """
    owner = tmp_path / OWNER_REL
    assert VW.snapshot_repo_is_owned(owner / "vault.git", owner) is True
    assert VW.snapshot_repo_is_owned(tmp_path / f"{OWNER_REL}-evil" / "x.git",
                                     owner) is False
    assert VW.snapshot_repo_is_owned(tmp_path / "lloyd-work" / ".t" / "vault.git",
                                     owner) is False


def test_a_stale_head_in_the_owner_repo_still_alerts_with_its_numbers(guardian, tmp_path,
                                                                     monkeypatch):
    """#3149 clause 4: the owner check may not weaken the alarm #3132 exists to run.

    #3132 was filed because `lloyd-vault-backup.timer` sat enabled and firing while
    370 runs each exited 128 and the vault's only off-vault copy did not move for four
    days. Any check placed in front of that page buys back exactly that silence if it
    can be tripped by something other than a foreign repository, so this node pins the
    alarm still firing at the owner's own path with the three numbers a person needs:
    the sha it is standing on, its age, and the bound it crossed."""
    g, _, acted = guardian
    repo, sha = _owner_backup_repo(tmp_path, monkeypatch, "vault", age_seconds=FOUR_HOURS)

    state = VW.snapshot_age_state(str(repo), VW.SNAPSHOT_MAX_AGE_SECONDS)
    assert state["owned"] is True, state                  # the owner's own repo
    assert state["fresh"] is False, state

    g._vault_snapshots_checked_at = 0.0
    g.check_vault()
    [(level, title, body)] = acted["alerts"]
    assert (level, title) == ("error", G.VAULT_SNAPSHOT_ALERT_TITLE), acted["alerts"]
    assert sha[:8] in body, body                          # its sha
    assert "4 h 0 m old" in body, body                    # its age
    assert "past the 3 h limit" in body, body             # the bound it crossed
    assert "not the vault's backup owner" not in body, body


def test_the_poller_names_the_state_function_and_the_cli_stays_a_human_surface():
    """Both halves of #3132's acceptance grep, pinned so they cannot rot into
    each other's shape.

    `guardian.py` must name `snapshot_age_state`: the poller has to read the SAME
    measurement the operator's CLI prints, and a check that shelled out to the
    CLI would be a second reader parsing a stdout nobody else reads (the class
    rule #3106 states: a verdict that is only printed is one nobody can read
    after the window closes). And the four paths the item's own grep names must
    still contain no `snapshot-age`: that subcommand is built to be run by a
    human and its exit code by a check that is not in this tree.

    The pattern the item spells is `snapshot.age`, and its `.` matches any
    character — so it is also the shape of `snapshot_age_state`, which the same
    clause requires by name in `guardian.py`. The literal CLI token is the one
    absence this change can honour while naming the function, so that is what is
    asserted here. Positive control below: the same grep against the module that
    defines the subcommand, where the token does live.
    """
    polled = [ROOT / "agent-services/guardian/guardian.py",
              ROOT / "scripts/service_health_check.py",
              ROOT / "agent-services/systemd", ROOT / "agent-services/supervisor"]
    for path in polled:
        assert path.exists(), f"{path} is not on the tree, so grepping it proves nothing"
        text = (path.read_text(encoding="utf-8") if path.is_file() else
                "".join(p.read_text(encoding="utf-8", errors="replace")
                        for p in sorted(path.rglob("*")) if p.is_file()))
        assert "snapshot-age" not in text, (
            f"{path} now shells out to `vaultwatch.py snapshot-age`; the durable "
            "record is the function, not a subprocess stdout")
    guardian_src = (ROOT / "agent-services/guardian/guardian.py").read_text(encoding="utf-8")
    assert "snapshot_age_state" in guardian_src, (
        "the guardian no longer reads the one measurement the CLI prints, so the "
        "two readers can disagree again — which is the premise #3132 filed")
    vw_src = (ROOT / "agent-services/guardian/vaultwatch.py").read_text(encoding="utf-8")
    assert "snapshot-age" in vw_src, "positive control: the CLI token still exists somewhere"


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


# ── #2001: the vault's scratch top-level folder is not counted ────────────────
#
# `~/obsidian/_pipeline/` held one thing, a stray pytest basetemp of 63 files.
# 63 >= TOPDIR_MIN_FILES, so removing the scratch would have read as a top-level
# folder being wiped: sync stopped, pool paused, a human-only `clear`.

def _scratch_vault(root: Path) -> Path:
    v = _vault(root)
    scratch = v / "_pipeline" / "tmp" / "pt1018"
    scratch.mkdir(parents=True)
    for i in range(63):
        (scratch / f"f{i}.json").write_text("{}")
    return v


def test_the_vault_exempts_exactly_one_top_level_folder_and_says_why():
    assert VW.VaultWatch.skip_top == frozenset({"_pipeline"})
    src = Path(VW.__file__).read_text(encoding="utf-8")
    comment = src[src.index("#: Top-level folders `measure` does not count"):
                  src.index("skip_top: frozenset[str] = frozenset({")]
    for said in ("scratch", "top-level-folder rule", "for that folder only"):
        assert said in comment, (said, comment)


def test_measure_counts_none_of_the_scratch_folder_and_every_other_folder_the_same(tmp_path):
    v = _scratch_vault(tmp_path / "vault")
    full = VW.measure(str(v))
    skipped = VW.measure(str(v), skip_top=VW.VaultWatch.skip_top)

    assert full.top["_pipeline"] == 63
    assert "_pipeline" not in skipped.top
    assert skipped.total == full.total - 63
    siblings = {k: n for k, n in full.top.items() if k != "_pipeline"}
    assert siblings and skipped.top == siblings


def test_shipping_the_exemption_against_a_baseline_that_still_holds_the_folder_does_not_trip(
        tmp_path):
    """The persisted `vault_watch.json` on the live box records `"_pipeline": 63`.
    The first tick of the new code measures no such bucket; read as stored, that
    is `top-level folder '_pipeline' (63 files) vanished` and a latched trip
    before anything was deleted."""
    v = _scratch_vault(tmp_path / "vault")
    state = tmp_path / "state"
    state.mkdir()
    old = VW.measure(str(v), now=1000.0)                       # counted, as the old code did
    assert old.top["_pipeline"] == 63
    (state / VW.STATE_FILE).write_text(json.dumps({"root": str(v), "last_good": old.to_dict()}))
    # The hazard is real for the stored snapshot taken as it stands:
    assert "'_pipeline' (63 files) vanished" in VW.evaluate(
        [old], VW.measure(str(v), now=1010.0, skip_top=VW.VaultWatch.skip_top),
        window=float("inf"))

    w = VW.VaultWatch(str(v), state)
    assert "_pipeline" not in w.history[-1].top
    assert w.history[-1].total == old.total - 63
    why, snap = w.tick(now=1010.0)
    assert why is None and w.tripped() is None
    ok, said = VW.sync_gate(str(v), state)
    assert ok, said

    # And removing the scratch afterwards is not an event either.
    shutil.rmtree(v / "_pipeline")
    assert w.tick(now=1020.0)[0] is None
    assert VW.sync_gate(str(v), state)[0]


def test_a_real_top_level_folder_emptying_still_trips_and_an_exempt_one_never_does(tmp_path):
    v = _scratch_vault(tmp_path / "vault")
    state = tmp_path / "state"
    w = VW.VaultWatch(str(v), state)
    assert w.tick(now=1000.0)[0] is None
    big = max((k for k in w.history[-1].top if k != "."), key=lambda k: w.history[-1].top[k])
    assert w.history[-1].top[big] >= VW.TOPDIR_MIN_FILES

    for f in (v / "_pipeline").rglob("*"):                      # the exempt folder empties
        if f.is_file():
            f.unlink()
    assert w.tick(now=1010.0)[0] is None

    shutil.rmtree(v / big)                                      # a real one vanishes
    why, _ = w.tick(now=1020.0)
    assert why is not None and repr(big) in why, why

    # The pure rule, both directions, on hand-written snapshots.
    before = VW.Snapshot(ts=0, total=5000, top={"notes": 40, "_pipeline": 63, ".": 3})
    emptied_exempt = VW.Snapshot(ts=1, total=4937, top={"notes": 40, "_pipeline": 0, ".": 3})
    emptied_real = VW.Snapshot(ts=1, total=4960, top={"notes": 0, "_pipeline": 63, ".": 3})
    skip = VW.VaultWatch.skip_top
    assert VW.evaluate([before], emptied_exempt, skip_top=skip) is None
    assert "'notes' (40 files) emptied" in VW.evaluate([before], emptied_real, skip_top=skip)


# ── a `clear` run in another process reaches the running watcher (#3095) ─────

def test_a_clear_from_another_process_reaches_the_watcher_before_it_judges(tmp_path,
                                                                          capsys):
    """One instance keeps judging, another one clears, and the clear has to land.

    Four minutes after the 2026-10-20 14:24:14 data trip, the operator removed the
    second leftover checkout and ran the documented one-command reset — whose own
    text promises to "re-baseline on the tree as it is now" — and the next guardian
    tick, 14:28:32, tripped critical again on the folder that RESET had just
    removed, with `before.ts` still the 14:24 baseline. It re-paused the worker pool
    and re-halted promotions, and the surviving physical evidence is two incident
    evidence directories four minutes apart. The CLI and the guardian are different
    processes: `clear()` rewrote the state file and its own `history`, while the
    running watcher still held the pre-incident baseline in memory and returned
    early while latched, so it never appended and never re-read. What recovery
    actually took was stopping the guardian unit, clearing, and starting it again —
    which is how a one-command reset became a service restart, and why the fix is a
    reload rather than an alias for `systemctl`.

    Driven as two instances over one state directory, which IS the seam: same
    classes, same files, nothing about the thing under test mocked out.
    """
    vault = _vault(tmp_path / "vault", files=600)
    state = tmp_path / "state"

    running = VW.VaultWatch(root=str(vault), state_dir=state)
    running.clear()
    shutil.rmtree(vault / "backlog")
    reason, snap = running.tick()
    assert reason == "top-level folder 'backlog' (150 files) vanished"
    # `tick()` only judges; writing the latch marker is the CALLER's act, and the
    # guardian is the only caller — `Guardian.check_vault` calls `trip()` with the
    # actions it took. So the actions are stated here rather than left out, and the
    # pause they record is what a re-trip would do a second time.
    running.trip(reason, snap, {"workers_paused": True, "promotions_halted": True})
    assert running.tripped(), "latched: that trip has been acted on"

    # The order is the incident's: the operator removes a folder FIRST, then runs
    # the reset. Getting it the other way round would describe a real loss and ask
    # the watcher to ignore it, which is not what was ever asked of it.
    shutil.rmtree(vault / "skills")

    # The operator's process: a fresh instance, which is exactly what the CLI
    # builds, run against the tree as it stands now.
    operator = VW.VaultWatch(root=str(vault), state_dir=state)
    assert operator.clear() is True
    assert operator.tripped() is None

    reason, _snap = running.tick()
    assert reason is None, (
        "the watcher was armed against a tree that still had `skills`, judged against "
        "that rather than the baseline it had just been told to adopt, and latched on "
        "the folder the operator had removed on purpose — which is the 14:28:32 "
        "re-trip, on a tree its own reset had already blessed")
    out = capsys.readouterr().out
    assert "baseline reloaded" in out, (
        "not tripping is half of it. A watcher that silently adopts another "
        "process's baseline is a watcher whose history a reader cannot reconstruct "
        "afterwards, so the reload is the surprising event and it gets said")


def test_the_reload_keys_on_the_state_file_mtime_not_on_the_marker(tmp_path):
    """The mtime is the trigger; a vanished marker only says when it may be asked.

    Both edges fail, and differently. Fire on "marker gone" alone and the watcher's
    own first healthy tick after a legitimate clear reloads what it just wrote and
    collapses `history` to one entry — the harm `clear` exists to avoid, arriving by
    the fix instead of the bug. Fire only on the marker and a state file rewritten
    while the watcher was never tripped is invisible. So the comparison is against
    the timestamp of OUR OWN last write, and the marker is consulted only to keep
    the stat off the latched path.
    """
    vault = _vault(tmp_path / "vault", files=400)
    state = tmp_path / "state"
    w = VW.VaultWatch(root=str(vault), state_dir=state)
    w.clear()

    for _ in range(4):
        assert w.tick()[0] is None
    assert len(w.history) >= 4, (
        "healthy ticks keep appending: a reload that fired on the watcher's own "
        "writes would leave exactly one entry and disarm the window rules silently")
    assert w._reload_if_rebaselined() is False, (
        "our own persist is not somebody else's re-baseline")

    (state / "vault_watch.json").touch()          # any other writer; no marker involved
    assert w._reload_if_rebaselined() is True, (
        "a moved mtime is the trigger, so a state file edited by hand or by another "
        "watcher's clear is adopted whether or not anyone was ever tripped")


def test_a_clear_that_ran_while_the_watcher_was_never_tripped_still_reloads(tmp_path):
    """Same path, no marker anywhere: what matters is that the file was replaced."""
    vault = _vault(tmp_path / "vault", files=400)
    state = tmp_path / "state"
    w = VW.VaultWatch(root=str(vault), state_dir=state)
    w.clear()
    for _ in range(3):
        w.tick()
    assert len(w.history) == 4

    other = VW.VaultWatch(root=str(vault), state_dir=state)
    other.clear()                                  # no marker existed; still a re-baseline
    other_ts = other.history[0].ts
    assert w.tick()[0] is None
    assert len(w.history) == 2, (
        "the new baseline replaces the long history and then this tick appends to it; "
        "keeping the old entries would mean the clear changed nothing the next "
        "judgement can see")
    assert w.history[0].ts == other_ts, (
        "and what it adopted is the operator's measurement, not its own earlier one")
