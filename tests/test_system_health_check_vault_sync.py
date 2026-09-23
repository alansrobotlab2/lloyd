"""#538 — the system health check must be able to see a vault that is not syncing.

`agent-obsidian-sync` is `ob sync --path ~/obsidian --continuous` under supervisor
with `autorestart=true`. A client that errors on every cycle therefore stays
RUNNING forever, and the check graded it on the single test
`healthy = (status == 'RUNNING') or (status == 'STOPPED' and expected_stopped)`, so
the vault's only off-box copy could stop moving entirely while the script printed
`=== Overall: HEALTHY ===`.

Two things make this awkward to test, and both are load-bearing:

* **The evidence cannot be a log grep.** The sync log carries no timestamps, prints
  `Fully synced` on every idle cycle, and both files rotate at 10 MB with 10
  backups — so `test_the_verdict_never_comes_from_the_logs` plants a log that looks
  perfectly healthy and requires the verdict to be unaffected.
* **The green state is an observed round trip**, which in production means a
  transient pull-only sync device on a paid Obsidian account (a human decision). So
  the tests drive a scripted fake `ob` through the real `exec` boundary — a
  subprocess, not a monkeypatched function — and the fake either really replicates
  the sentinel from a stand-in "server" directory or really does not. The positive
  test therefore proves the mechanism detects a stalled client, not merely that a
  branch exists.

Each test names the acceptance clause it pins.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# The script lives in the vault, which is a live shared tree with no branch; a
# candidate copy is tested before it is written there by pointing this at it.
SCRIPT = Path(os.environ.get("LLOYD_SHC_SCRIPT") or
              Path.home() / "obsidian" / "skills" / "system-health-check" / "system_health_check.py")
SKILL = SCRIPT.parent / "SKILL.md"

DEGRADED_EXIT = 3

# What `ob sync-status --path <linked vault>` really prints on this box, modulo the
# vault id, which every fixture below varies on purpose.
SYNC_STATUS_LINKED = textwrap.dedent(
    """\
    Sync Configuration:
      Vault: obsidian ({vault_id})
      Location: {vault}
      Sync mode: bidirectional
      Conflict strategy: merge
      Device name: goliath (Linux)
      File types: image, audio, pdf, video
      Configs: none (config syncing disabled)
      Excluded folders: .git
    """
)


def _load_module():
    """Load the CLI script as a module.

    It lives in the vault, outside this repo's package, and nothing imports it —
    it is invoked from Bash. Loading by path is how the unit-level assertions
    reach the verdict vocabulary without inventing an import surface.
    """
    assert SCRIPT.is_file(), f"health check script missing: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("shc_vault_sync", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shc = _load_module()

# The fake `ob` is itself a program, not a Python double: the boundary under test is
# `subprocess.run([ob, 'sync-status', ...])` reaching a real executable, so
# double-ing `subprocess` inside the module would leave the resolution, the
# argument construction and the exit-code reading untested.
FAKE_OB = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, shutil, sys
    from pathlib import Path

    args = sys.argv[1:]
    command = args[0] if args else ""

    log = os.environ.get("FAKE_OB_LOG")
    if log:
        with open(log, "a") as handle:
            handle.write(json.dumps(args) + "\\n")

    spec = json.loads(os.environ["FAKE_OB_SPEC"])
    entry = spec.get(command, {"rc": 0, "out": ""})

    # Which config and whether a token each call ran with: the property clause 1
    # rests on is in the environment, not the arguments.
    envlog = os.environ.get("FAKE_OB_ENVLOG")
    if envlog:
        with open(envlog, "a") as handle:
            handle.write(json.dumps({"args": args, "xdg": os.environ.get("XDG_CONFIG_HOME", ""),
                                     "token": bool(os.environ.get("OBSIDIAN_AUTH_TOKEN"))}) + "\\n")

    def opt(name):
        return args[args.index(name) + 1] if name in args else None

    # Registry mode models the shipped client's storage (#1141): one directory per
    # VAULT ID under $XDG_CONFIG_HOME/obsidian-headless/sync/, and `sync-unlink`
    # deleting by the vault id of whatever path it is given (cli.js `Rr(t.vaultId)`).
    if spec.get("_registry"):
        live_root = Path(os.environ["FAKE_LIVE_CONFIG"]) / "obsidian-headless" / "sync"
        own_root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") \\
            / "obsidian-headless" / "sync"
        read_root = live_root if spec.get("_ignore_xdg") else own_root
        write_root = live_root if (spec.get("_ignore_xdg") or spec.get("_ignore_xdg_writes")) \\
            else own_root

        def registrations(root):
            found = {}
            if root.is_dir():
                for entry in sorted(root.iterdir()):
                    cfg = entry / "config.json"
                    if cfg.is_file():
                        found[entry.name] = json.loads(cfg.read_text())
            return found

        here = str(Path(opt("--path") or ".").resolve())
        token_file = own_root.parent / "auth_token"
        has_token = bool(os.environ.get("OBSIDIAN_AUTH_TOKEN")) or token_file.is_file()
        if command in ("sync-setup", "sync") and not has_token:
            sys.stderr.write('No account logged in. Run "ob login" first.\\n')
            sys.exit(2)
        if command in ("sync-config", "sync") and here not in [
                c["vaultPath"] for c in registrations(write_root).values()]:
            sys.stderr.write(f"No sync configuration found for {here}\\n")
            sys.exit(3)
        if command == "sync-status" and spec.get("_status_fail_after"):
            counter = Path(os.environ["FAKE_OB_LOG"] + ".status-count")
            n = int(counter.read_text()) + 1 if counter.exists() else 1
            counter.write_text(str(n))
            if n > spec["_status_fail_after"]:
                sys.stderr.write("Status check failed: Error: socket hang up\\n")
                sys.exit(1)
        if command == "sync-list-local":
            regs = registrations(read_root)
            if not regs:
                print("No vaults configured.")
            else:
                print("Configured vaults:")
                for vid, cfg in regs.items():
                    print(f"  {vid}\\n    Path: {cfg['vaultPath']}")
            sys.exit(0)
        if command == "sync-setup":
            if spec.get("_e2e"):
                print("Fetching vault info...")
                sys.stderr.write("Password not provided.\\n")
                sys.exit(2)
            if spec.get("_setup_refused"):
                print("Fetching vault info...\\nSetting up...")
                sys.stderr.write("Failed to validate password. Error: Vault limit exceeded.\\n")
                sys.exit(2)
            vid = opt("--vault")
            (write_root / vid).mkdir(parents=True, exist_ok=True)
            (write_root / vid / "config.json").write_text(
                json.dumps({"vaultId": vid, "vaultPath": here}))
            print("Setting up...")
            sys.exit(0)
        if command == "sync-unlink":
            for vid, cfg in registrations(write_root).items():
                if cfg["vaultPath"] == here:
                    shutil.rmtree(write_root / vid)
            sys.exit(0)
        if command == "sync-status":
            for vid, cfg in registrations(live_root if spec.get("_ignore_xdg_writes")
                                          else read_root).items():
                if cfg["vaultPath"] == here:
                    sys.stdout.write(spec["sync-status"]["out"].format(vault_id=vid, vault=here))
                    sys.exit(0)
            sys.stderr.write(f"No sync configuration found for {here}\\n")
            sys.exit(3)

    if command == "sync" and spec.get("_replicate"):
        # Model the live client uploading, then the server handing the file to this
        # client. Only a real push can put the sentinel in the server directory,
        # so a sync that is not really moving files replicates nothing.
        vault = Path(os.environ["FAKE_VAULT_DIR"])
        server = Path(os.environ["FAKE_SERVER_DIR"])
        target = Path(opt("--path") or ".")
        for source in sorted((vault / "lloyd").glob("healthcheck-vault-sync-*.md")):
            shutil.copyfile(source, server / "lloyd" / source.name)
            if target != vault:
                (target / "lloyd").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target / "lloyd" / source.name)

    body = entry.get("out", "")
    if body:
        sys.stdout.write(body.format(vault_id=spec.get("_vault_id", ""),
                                     vault=os.environ["FAKE_VAULT_DIR"]))
    if entry.get("err"):
        sys.stderr.write(entry["err"].format(vault_id=spec.get("_vault_id", ""),
                                             vault=os.environ["FAKE_VAULT_DIR"]))
    sys.exit(entry.get("rc", 0))
    """
)


@pytest.fixture()
def rig(tmp_path):
    class Rig:
        """Drives the real script as a subprocess against a scripted `ob`.

        Every sync path the script derives comes from `LLOYD_HEALTH_VAULT_PATH`, so
        the probe can be aimed at a scratch directory and the live `~/obsidian`
        stays out of it entirely.
        """

        linked_id = "566dda1217a7a8dfddbac33d4f280b3a"

        def __init__(self, base):
            self.vault = base / "vault"
            (self.vault / "lloyd").mkdir(parents=True)
            self.server = base / "server"          # the stand-in sync remote
            (self.server / "lloyd").mkdir(parents=True)
            self.fake = base / "ob"
            self.fake.write_text(FAKE_OB)
            self.fake.chmod(0o755)
            self.calls = base / "ob-calls.log"
            self.calls.touch()
            # The "live" ob config for this rig: never the real ~/.config.
            self.live_config = base / "live-config"
            self.registry = self.live_config / "obsidian-headless" / "sync"
            self.registry.mkdir(parents=True)
            (self.live_config / "obsidian-headless" / "auth_token").write_text("fake-token\n")
            self.state_file = base / "state" / "last_sync_success.json"

        def plant_live_registration(self, vault_id=None):
            vid = vault_id or self.linked_id
            (self.registry / vid).mkdir(parents=True, exist_ok=True)
            (self.registry / vid / "config.json").write_text(
                json.dumps({"vaultId": vid, "vaultPath": str(self.vault.resolve())}))
            return self.registry / vid

        def live_ids(self):
            return sorted(p.name for p in self.registry.iterdir() if (p / "config.json").is_file())

        def spec(self, *, linked=True, session="ok", replicate=False,
                 vault_id=None, status_logged_out=False, e2ee_missing=False,
                 registry=False, e2e=False, ignore_xdg=False, ignore_xdg_writes=False,
                 setup_refused=False, status_fail_after=0):
            spec: dict = {"_vault_id": vault_id or self.linked_id,
                          "_replicate": bool(replicate), "_registry": bool(registry),
                          "_e2e": bool(e2e), "_ignore_xdg": bool(ignore_xdg),
                          "_ignore_xdg_writes": bool(ignore_xdg_writes),
                          "_setup_refused": bool(setup_refused),
                          "_status_fail_after": int(status_fail_after)}
            if linked:
                body = SYNC_STATUS_LINKED
                if e2ee_missing:
                    body = "End-to-end encryption key missing. Run setup again.\n" + body
                if status_logged_out:
                    body = "Not logged in.\n" + body
                spec["sync-status"] = {"rc": 0, "out": body}
            else:
                spec["sync-status"] = {
                    "rc": 3,
                    "out": "",
                    "err": "No sync configuration found for {vault}\n",
                }
            if session == "ok":
                spec["sync-list-remote"] = {"rc": 0, "out": "Remotes:\n  obsidian\n"}
            elif session == "logged-out":
                spec["sync-list-remote"] = {
                    "rc": 2, "out": "",
                    "err": 'No account logged in. Run "ob login" first.\n',
                }
            spec.setdefault("sync-setup", {"rc": 0, "out": "Linked.\n"})
            spec.setdefault("sync-config", {"rc": 0, "out": "Sync mode set.\n"})
            spec.setdefault("sync", {"rc": 0, "out": "Fully synced\n"})
            spec.setdefault("sync-unlink", {"rc": 0, "out": "Unlinked.\n"})
            return spec

        def env(self, spec, vault=None):
            env = dict(os.environ)
            env.update({
                "LLOYD_OB_BIN": str(self.fake),
                "LLOYD_HEALTH_VAULT_PATH": str(vault or self.vault),
                "FAKE_OB_SPEC": json.dumps(spec),
                "FAKE_OB_LOG": str(self.calls),
                "FAKE_OB_ENVLOG": str(self.calls) + ".env",
                "FAKE_VAULT_DIR": str(self.vault),
                "FAKE_SERVER_DIR": str(self.server),
                "FAKE_LIVE_CONFIG": str(self.live_config),
                "XDG_CONFIG_HOME": str(self.live_config),
                "LLOYD_VAULT_SYNC_STATE_FILE": str(self.state_file),
                "LLOYD_OB_TIMEOUT": "15",
            })
            env.pop("LLOYD_VAULT_SYNC_ROUND_TRIP", None)
            env.pop("OBSIDIAN_AUTH_TOKEN", None)
            return env

        def ob(self, *args, spec, config_home=None):
            """Run the fake client directly, as the probe would."""
            env = self.env(spec)
            if config_home is not None:
                env["XDG_CONFIG_HOME"] = str(config_home)
            return subprocess.run([str(self.fake), *args], capture_output=True, text=True,
                                  env=env, timeout=60)

        def run(self, *extra, spec=None, round_trip=False, fmt="json", vault=None):
            payload = spec if spec is not None else self.spec()
            argv = [sys.executable, str(SCRIPT), "--format", fmt]
            argv += list(extra)
            if round_trip:
                argv += ["--vault-sync-round-trip", "--round-trip-timeout", "20"]
            return subprocess.run(argv, capture_output=True, text=True,
                                  env=self.env(payload, vault=vault), timeout=120)

        def json(self, *extra, **kw):
            proc = self.run(*extra, **kw)
            assert proc.returncode in (0, DEGRADED_EXIT), (
                f"exit {proc.returncode} is neither healthy nor degraded: "
                f"{proc.stdout[-1500:]}{proc.stderr[-1500:]}"
            )
            return json.loads(proc.stdout), proc

        def logged_env(self):
            path = Path(str(self.calls) + ".env")
            return [json.loads(line) for line in path.read_text().splitlines()
                    if line.strip()] if path.exists() else []

        def logged(self):
            return [json.loads(line) for line in
                    self.calls.read_text().splitlines() if line.strip()]

    return Rig(tmp_path)


# --------------------------------------------------------------- clause 1
# "`python3 system_health_check.py --format json` contains a vault_sync object
#  carrying an explicit state and a reason string, and the text format prints a
#  Vault Sync section."
def test_json_output_carries_a_vault_sync_object_with_state_and_reason(rig):
    payload, proc = rig.json("--component", "vault_sync")
    block = payload["vault_sync"]
    assert isinstance(block, dict), block
    assert block["state"] in shc.VAULT_SYNC_STATES, block
    assert isinstance(block["reason"], str) and block["reason"], block
    assert isinstance(block["detail"], str), block
    assert proc.returncode in (0, DEGRADED_EXIT)


def test_text_format_prints_a_vault_sync_section(rig):
    proc = rig.run("--component", "vault_sync", fmt="text")
    assert "--- Vault Sync ---" in proc.stdout, proc.stdout
    assert "obsidian-sync —" in proc.stdout, proc.stdout


# --------------------------------------------------------------- clause 2
# "With the vault unlinked (ob sync-status --path <path> reports no sync
#  configuration), the check prints DEGRADED, exits non-zero, and reports reason
#  not-configured."
def test_an_unlinked_vault_is_not_configured_degraded_and_exits_non_zero(rig):
    payload, proc = rig.json("--component", "vault_sync", spec=rig.spec(linked=False))
    block = payload["vault_sync"]
    assert block["state"] == "not-configured", block
    assert block["reason"] == "not-configured", block
    assert block["kind"] == "cannot-sync", block
    assert payload["overall_status"] == "degraded", payload
    assert proc.returncode == DEGRADED_EXIT, (proc.returncode, proc.stdout)
    assert "=== Overall: DEGRADED ===" in rig.run(
        "--component", "vault_sync", fmt="text", spec=rig.spec(linked=False)).stdout


# --------------------------------------------------------------- clause 3
# "With the client logged out (ob login reports no session), the reason is
#  not-logged-in — a value distinct from not-configured — and the overall verdict is
#  DEGRADED."
#
# Two real shapes of "logged out": `ob sync-status` prints `Not logged in.` while
# still showing a stored configuration, and a command that needs auth exits 2 with
# `No account logged in.` — the second is what a *stale* configuration looks like
# after the session is gone, and it must not be mistaken for "unlinked".
@pytest.mark.parametrize("shape", ["status-says-so", "auth-command-exits-2"])
def test_a_logged_out_client_is_not_logged_in(rig, shape):
    spec = rig.spec(status_logged_out=True) if shape == "status-says-so" \
        else rig.spec(session="logged-out")
    payload, proc = rig.json("--component", "vault_sync", spec=spec)
    block = payload["vault_sync"]
    assert block["state"] == "not-logged-in", (shape, block)
    assert block["reason"] == "not-logged-in", (shape, block)
    assert payload["overall_status"] == "degraded", payload
    assert proc.returncode == DEGRADED_EXIT


def test_not_logged_in_is_a_different_value_from_not_configured():
    """The two states cannot collapse into one verdict."""
    assert "not-logged-in" != "not-configured"
    assert {"not-logged-in", "not-configured"} <= set(shc.VAULT_SYNC_CANNOT_SYNC)


# --------------------------------------------------------------- clause 4
# "`nothing-to-sync` has its own state value, distinct from every cannot-sync value,
#  and a test asserts a green supervisor row for agent-obsidian-sync plus a failing
#  sync probe still yields DEGRADED — process liveness alone can never produce a
#  green sync state."
def test_nothing_to_sync_is_its_own_state_and_not_a_cannot_sync_value(rig, tmp_path):
    """Reached by making the sentinel unwritable: the probe created no local change,
    so nothing could cross the wire. That is a different report from "the vault
    cannot sync", and it must not be reported as one."""
    vault = tmp_path / "read-only-vault"
    (vault / "lloyd").mkdir(parents=True)
    os.chmod(vault / "lloyd", 0o500)
    try:
        payload, _ = rig.json("--component", "vault_sync",
                              round_trip=True, spec=rig.spec(), vault=vault)
    finally:
        os.chmod(vault / "lloyd", 0o700)
    block = payload["vault_sync"]
    assert block["state"] == "nothing-to-sync", block
    assert block["state"] not in shc.VAULT_SYNC_CANNOT_SYNC
    assert block["kind"] == "unproven", block
    assert block["healthy"] is False, block


def test_a_running_sync_process_never_makes_the_sync_state_green(rig):
    """The clause in one assertion: supervisor says `agent-obsidian-sync` is RUNNING
    (that is the live box, and it is the exact condition under which the old check
    printed HEALTHY), the probe says the sentinel never came back, and the sync
    state is still not green."""
    services = [{"name": "agent-obsidian-sync", "status": "RUNNING",
                 "healthy": True, "expected_stopped": False}]
    failing = {"component": "vault_sync", "state": "sync-failed",
               "kind": "cannot-sync", "healthy": False,
               "reason": "sentinel-not-replicated", "detail": "stub"}
    healthy, reasons = shc.compute_overall(
        disk=[], services=services, endpoints=[],
        vault_sync=failing, components=["services", "vault_sync"])
    assert shc.services_healthy(services) is True, "the supervisor row must read green"
    assert healthy is False
    assert any(reason.startswith("vault_sync:") for reason in reasons), reasons


def test_a_live_running_process_with_a_failing_probe_still_prints_degraded(rig):
    """Same clause across the real process boundary: the real supervisor answer
    (RUNNING) plus a faulted sync probe has to yield exit 3."""
    stalled = rig.spec(replicate=False)
    payload, proc = rig.json("--component", "services", "vault_sync",
                             spec=stalled, round_trip=True)
    rows = {row["name"]: row for row in payload["services"]["details"]}
    assert rows["agent-obsidian-sync"]["status"] == "RUNNING", rows
    assert payload["vault_sync"]["healthy"] is False, payload["vault_sync"]
    assert payload["overall_status"] == "degraded", payload
    assert proc.returncode == DEGRADED_EXIT


def test_only_the_synced_state_is_green():
    """Nothing but an observed round trip can produce a green sync component, so no
    future state string can be added that sneaks liveness in as green."""
    assert shc.VAULT_SYNC_GREEN == "synced"
    assert shc._sync_result("synced", "r")["healthy"] is True
    for state in shc.VAULT_SYNC_STATES:
        if state != "synced":
            assert shc._sync_result(state, "r")["healthy"] is False, state
    with pytest.raises(ValueError):
        shc._sync_result("running", "an unrecognised state must not be constructible")


# --------------------------------------------------------------- clause 5
# "The verdict is computed from the sync probe, not the logs: with
#  agent-obsidian-sync.log/.err written full of green-looking lines (Fully synced,
#  zero errors) and the probe faulted, the check still prints DEGRADED, and no code
#  path derives last-sync time from a log mtime or from ob sync-status."
def test_the_verdict_never_comes_from_the_logs(rig, tmp_path):
    """The log plant is the point: `Fully synced` is printed on every idle cycle and
    the files rotate at 10 MB, so a log full of success is compatible with a vault
    that has not uploaded anything in weeks. The probe disagrees, and the probe wins.
    """
    log = tmp_path / "agent-obsidian-sync.log"
    log.write_text("Fully synced\n" * 200)
    err = tmp_path / "agent-obsidian-sync.err"
    err.write_text("")
    assert log.read_text().count("Fully synced") == 200     # the plant really is green
    assert "error" not in log.read_text().lower()
    assert err.read_text() == ""

    payload, proc = rig.json("--component", "vault_sync",
                             spec=rig.spec(replicate=False), round_trip=True)
    assert payload["vault_sync"]["state"] == "sync-failed", payload["vault_sync"]
    assert payload["overall_status"] == "degraded"
    assert proc.returncode == DEGRADED_EXIT


def test_the_script_reads_no_log_file_and_no_sync_status_timestamp():
    """No code path may derive last-sync time from a log mtime or from
    `ob sync-status`: sync-status carries no timestamp at all, and the log's mtime
    advances on idle cycles. Asserted on the source because the only way to make
    this fail is to write it."""
    source = SCRIPT.read_text()
    assert "agent-obsidian-sync.log" not in source
    assert "agent-obsidian-sync.err" not in source
    assert ".log" not in source.replace(".login", "")
    assert "st_mtime" not in source and "getmtime" not in source
    # #1141 added a last-sync record, and its only source is a dedicated state file
    # written by an observed round trip — never a log, never `ob sync-status`.
    assert shc.LAST_SYNC_STATE_DEFAULT.endswith(".json")
    assert "agent-services" not in shc.LAST_SYNC_STATE_DEFAULT
    assert not any(line.lstrip().startswith("#") is False and "open(" in line
                   and "log" in line.lower() for line in source.splitlines())


def test_the_probe_never_opens_the_log_files_even_when_reading_them_would_persuade_it(rig):
    """Behavioural half of clause 5: with a faulted round trip and `strace`-free
    proof — the fake `ob` sees the calls, and the log paths never enter them."""
    rig.json("--component", "vault_sync", spec=rig.spec(replicate=False),
             round_trip=True)
    for call in rig.logged():
        assert not any("agent-obsidian-sync" in argument for argument in call), call


# --------------------------------------------------------------- clause 6
# "The component is selectable: --component vault_sync runs only it and --skip
#  vault_sync excludes it (both argparse choices lists extended), and skipping the
#  end-to-end leg reports a weaker, named state rather than green."
def test_component_selection_runs_only_the_sync_probe(rig):
    payload, _ = rig.json("--component", "vault_sync")
    assert payload["components"] == ["vault_sync"], payload["components"]
    assert payload["disk"]["volumes"] == [], payload["disk"]
    assert payload["services"]["details"] == [], payload["services"]
    assert payload["tools"]["endpoints"] == [], payload["tools"]


def test_skip_vault_sync_excludes_the_block(rig):
    """`--skip vault_sync` removes the block, and checking nothing is a degraded run.

    Both halves used to ride on one argv: `--skip vault_sync disk services tools` relied
    on `--skip`'s `nargs='+'` swallowing the four names after it, so `components == []`
    was a fact about argument parsing rather than about the check. `voice_media` joined
    `COMPONENTS` when #644 gave the checker a media-plane probe — five components now —
    and the swallowed list left one running, which turned the assertion into whichever
    component happened to sit outside the skip list. Each half is now stated on its own:
    the skip list is built from `COMPONENTS`, so a sixth component cannot quietly reopen
    it, and the block-exclusion claim is asserted against a real selection.
    """
    payload, _ = rig.json("--component", "disk", "services", "tools", "--skip", "vault_sync")
    assert "vault_sync" not in payload, sorted(payload)
    assert payload["components"] == ["disk", "services", "tools"], payload["components"]

    # Selecting nothing is not a pass either — the check measured nothing.
    empty, _ = rig.json("--skip", *shc.COMPONENTS)
    assert empty["components"] == [], empty["components"]
    assert empty["overall_status"] == "degraded", empty
    assert empty["reasons"] == ["no components were checked"], empty["reasons"]

    # And the component #644 added is selectable and excludable like the other four:
    # skipping it removes its block rather than leaving a row that reads as checked.
    skipped, _ = rig.json("--component", "disk", "--skip", "voice_media")
    assert "voice_media" not in skipped, sorted(skipped)
    assert skipped["components"] == ["disk"], skipped["components"]
    assert "voice_media" in shc.COMPONENTS, shc.COMPONENTS


def test_skipping_the_end_to_end_leg_is_a_weaker_named_state_not_green(rig):
    payload, proc = rig.json("--component", "vault_sync")
    block = payload["vault_sync"]
    assert block["state"] == "roundtrip-skipped", block
    assert block["state"] != shc.VAULT_SYNC_GREEN
    assert block["state"] not in shc.VAULT_SYNC_CANNOT_SYNC
    assert block["kind"] == "unproven", block
    assert block["healthy"] is False, block
    assert proc.returncode == DEGRADED_EXIT


def test_the_round_trip_flag_is_what_produces_the_green_state(rig):
    """The other direction, so `roundtrip-skipped` cannot be satisfied by a probe
    that is simply always unhappy."""
    payload, proc = rig.json("--component", "vault_sync",
                             spec=rig.spec(replicate=True), round_trip=True)
    block = payload["vault_sync"]
    assert block["state"] == "synced", block
    assert block["kind"] == "green", block
    assert payload["overall_status"] == "healthy", payload
    assert proc.returncode == 0
    assert block["round_trip"] is True
    assert any("pulled back" in item for item in block["evidence"]), block


@pytest.mark.parametrize("option", ["--component", "--skip"])
def test_both_argparse_choices_lists_offer_the_component(option):
    """The clause names both lists: a component that `--component` accepts but
    `--skip` refuses is half a component. argparse prints the set it actually
    compared against on an invalid choice, so this tests the live lists rather
    than a rendering of them."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), option, "no-such-component"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2, (option, proc.stdout[-300:], proc.stderr[-300:])
    assert "invalid choice" in proc.stderr, (option, proc.stderr[-600:])
    assert f"{{{','.join(shc.COMPONENTS)}}}" in proc.stderr, (option, proc.stderr[-600:])
    assert "vault_sync" in shc.COMPONENTS


# ------------------------------------------- #1030 clauses the mechanism must keep
def test_sync_unlink_is_never_pointed_at_the_live_vault(rig):
    """`sync-unlink` is documented to remove stored credentials, so a scratch-path
    guard that leaks is a way to unlink the real vault."""
    rig.json("--component", "vault_sync", spec=rig.spec(replicate=True),
             round_trip=True)
    unlink_paths = [args[args.index("--path") + 1] for args in rig.logged()
                    if args[:1] == ["sync-unlink"]]
    assert unlink_paths, "the scratch client was never unlinked"
    for path in unlink_paths:
        real = Path(path).resolve()
        assert real != rig.vault.resolve(), real
        assert not str(real).startswith(str(rig.vault.resolve())), real
        assert Path(path).name.startswith(shc.SCRATCH_PREFIX), real
    assert rig.vault.exists()


def test_the_guard_refuses_the_live_vault_and_its_children(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    assert shc._is_probe_scratch(str(vault), str(vault)) is False
    assert shc._is_probe_scratch(str(vault / "lloyd"), str(vault)) is False
    assert shc._is_probe_scratch(None, str(vault)) is False


def test_the_vault_id_is_read_at_run_time_and_never_hardcoded(rig):
    """The id has already changed once (c73df6a0… → 566dda12…), so a pinned id turns
    a healthy vault into an outage on the day it is re-linked."""
    chosen = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    payload, _ = rig.json("--component", "vault_sync",
                          spec=rig.spec(vault_id=chosen), round_trip=True)
    assert payload["vault_sync"]["vault_id"] == chosen, payload["vault_sync"]
    setup = next(args for args in rig.logged() if args[:1] == ["sync-setup"])
    assert setup[setup.index("--vault") + 1] == chosen, setup
    source = SCRIPT.read_text()
    assert "c73df6a055405245f95f936685bb3683" not in source
    assert "566dda1217a7a8dfddbac33d4f280b3a" not in source


def test_an_unresolvable_ob_binary_is_its_own_cannot_sync_state(tmp_path, rig):
    env_spec = rig.spec()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json", "--component", "vault_sync"],
        capture_output=True, text=True, timeout=60,
        env={**rig.env(env_spec), "LLOYD_OB_BIN": str(tmp_path / "no-such-ob")},
    )
    payload = json.loads(proc.stdout)
    assert payload["vault_sync"]["state"] == "ob-missing", payload["vault_sync"]
    assert payload["vault_sync"]["kind"] == "cannot-sync"
    assert proc.returncode == DEGRADED_EXIT


def test_a_missing_encryption_credential_is_its_own_state_never_a_crash(rig):
    payload, proc = rig.json("--component", "vault_sync",
                             spec=rig.spec(e2ee_missing=True))
    block = payload["vault_sync"]
    assert block["state"] == "needs-credential", block
    assert block["kind"] == "cannot-sync", block
    assert proc.returncode == DEGRADED_EXIT
    assert "Traceback" not in proc.stdout + proc.stderr


def test_the_sentinel_is_cleaned_up_after_both_verdicts(rig):
    """The probe writes into a live, synced vault; a leftover per run is litter the
    sync client would happily upload forever."""
    rig.json("--component", "vault_sync", spec=rig.spec(replicate=True),
             round_trip=True)
    rig.json("--component", "vault_sync", spec=rig.spec(replicate=False),
             round_trip=True)
    assert list((rig.vault / "lloyd").glob("healthcheck-vault-sync-*.md")) == []


def test_the_scratch_directory_is_removed_after_a_run(rig):
    rig.json("--component", "vault_sync", spec=rig.spec(replicate=True),
             round_trip=True)
    scratch = [args[args.index("--path") + 1] for args in rig.logged()
               if args[:1] == ["sync-unlink"]]
    assert scratch
    for path in scratch:
        assert not Path(path).exists(), f"scratch client left behind: {path}"


# ------------------------------------------- the two formatters cannot disagree
def test_text_and_json_report_the_same_verdict_for_the_same_inputs(rig):
    for label, spec, round_trip in [
        ("linked, leg skipped", rig.spec(), False),
        ("linked, leg ran, nothing replicated", rig.spec(replicate=False), True),
        ("linked, leg ran, replicated", rig.spec(replicate=True), True),
        ("unlinked", rig.spec(linked=False), False),
        ("logged out", rig.spec(session="logged-out"), False),
    ]:
        payload, json_proc = rig.json("--component", "vault_sync", spec=spec,
                                      round_trip=round_trip)
        text_proc = rig.run("--component", "vault_sync", spec=spec,
                            round_trip=round_trip, fmt="text")
        word = "HEALTHY" if payload["overall_status"] == "healthy" else "DEGRADED"
        assert f"=== Overall: {word} ===" in text_proc.stdout, (label, text_proc.stdout)
        assert text_proc.returncode == json_proc.returncode, label


# ================================================================ #1141
# The probe deleted the live sync registration twice on 2026-09-14 (12:03 and
# 14:15 PDT, from a chat): `ob` keeps registrations per vault id and
# `sync-unlink` deletes by vault id, so a scratch client sharing the live config
# took the live registration with it. The fake below models exactly that storage.

# --------------------------------------------------------------- clause 1
def test_the_stub_deletes_by_vault_id_the_way_the_shipped_client_does(rig, tmp_path):
    """The counterfactual, so clause 1 cannot pass against a stub too kind to
    reproduce the incident: a scratch client in the SHARED config, set up and
    unlinked exactly as the old probe did, deletes the live registration."""
    spec = rig.spec(registry=True)
    rig.plant_live_registration()
    scratch = tmp_path / "old-probe-scratch"
    scratch.mkdir()
    assert rig.ob("sync-setup", "--vault", rig.linked_id, "--path", str(scratch),
                  spec=spec).returncode == 0
    assert rig.ob("sync-unlink", "--path", str(scratch), spec=spec).returncode == 0
    assert rig.live_ids() == [], "the stub failed to model the vault-id deletion"


def test_a_round_trip_leaves_the_live_registration_resolving(rig):
    rig.plant_live_registration()
    spec = rig.spec(registry=True, replicate=True)
    payload, proc = rig.json("--component", "vault_sync", spec=spec, round_trip=True)
    block = payload["vault_sync"]
    assert block["state"] == "synced", block
    assert block["live_registration"] == "intact", block
    assert rig.live_ids() == [rig.linked_id], "the live registration was deleted"
    status = rig.ob("sync-status", "--path", str(rig.vault), spec=spec)
    assert status.returncode == 0 and rig.linked_id in status.stdout, status
    # Every scratch call ran in its own config, with the token handed over in the
    # environment; the live config saw only reads.
    calls = rig.logged_env()
    scratch = [c for c in calls if c["args"][:1] in (["sync-setup"], ["sync-config"],
                                                       ["sync"], ["sync-unlink"])]
    assert {c["args"][0] for c in scratch} == {"sync-setup", "sync-config", "sync", "sync-unlink"}
    for c in scratch:
        assert c["xdg"] and c["xdg"] != str(rig.live_config), c
        assert Path(c["xdg"]).name.startswith(shc.SCRATCH_PREFIX), c
        assert c["token"] is True, c
    checks = [c for c in calls if c["args"][:1] == ["sync-list-local"]]
    assert len(checks) == 1 and checks[0]["xdg"] == scratch[0]["xdg"], checks
    live = [c for c in calls if c["xdg"] == str(rig.live_config)]
    assert live and {c["args"][0] for c in live} <= {"sync-status", "sync-list-remote"}, live


def test_a_client_that_ignores_the_isolated_config_is_refused_before_setup(rig):
    rig.plant_live_registration()
    spec = rig.spec(registry=True, replicate=True, ignore_xdg=True)
    payload, proc = rig.json("--component", "vault_sync", spec=spec, round_trip=True)
    block = payload["vault_sync"]
    assert block["state"] == "sync-probe-failed" and block["reason"] == "scratch-not-isolated", block
    called = [a[0] for a in rig.logged()]
    assert "sync-setup" not in called and "sync-unlink" not in called, called
    assert rig.live_ids() == [rig.linked_id]
    assert proc.returncode == DEGRADED_EXIT


def test_a_lost_live_registration_is_reported_whatever_the_probe_concluded(rig):
    """A client that honours the isolation check but writes to the live config
    anyway: the post-probe re-read is the only thing that can notice."""
    rig.plant_live_registration()
    spec = rig.spec(registry=True, replicate=True, ignore_xdg_writes=True)
    payload, proc = rig.json("--component", "vault_sync", spec=spec, round_trip=True)
    block = payload["vault_sync"]
    assert block["state"] == "sync-failed" and block["reason"] == "live-registration-lost", block
    assert block["live_registration"] == "lost", block
    assert "ob sync-setup" in block["detail"], block
    assert proc.returncode == DEGRADED_EXIT


def test_no_account_token_means_no_scratch_client(rig):
    (rig.live_config / "obsidian-headless" / "auth_token").unlink()
    rig.plant_live_registration()
    payload, _ = rig.json("--component", "vault_sync", spec=rig.spec(registry=True),
                          round_trip=True)
    block = payload["vault_sync"]
    assert block["state"] == "not-logged-in" and block["reason"] == "no-token-for-scratch-client", block
    assert "sync-setup" not in [a[0] for a in rig.logged()]


def test_both_scratch_directories_are_removed(rig, tmp_path, monkeypatch):
    """The check must leave no scratch directory behind.

    Scoped to a TMPDIR of its own rather than globbing the shared one. The
    subprocess builds its env from `dict(os.environ)`, so it lands here too, and
    only this invocation's scratch can appear in the diff. Globbing the real
    temp dir made the assertion a statement about the whole machine: under
    `-n 8` a sibling worker creating a scratch directory with the same prefix
    showed up in `after - before` and failed this test for someone else's file.
    """
    scratch_root = tmp_path / "scratch-tmp"
    scratch_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch_root))

    before = set(scratch_root.glob(shc.SCRATCH_PREFIX + "*"))
    rig.plant_live_registration()
    rig.json("--component", "vault_sync", spec=rig.spec(registry=True, replicate=True),
             round_trip=True)
    after = set(scratch_root.glob(shc.SCRATCH_PREFIX + "*"))
    assert after - before == set(), sorted(after - before)


# --------------------------------------------------------------- clause 2
def test_an_e2e_vault_is_unprovable_not_a_sync_failure(rig):
    """`ob sync-setup` on an end-to-end-encrypted vault prompts for the password;
    with stdin closed it exits 2 "Password not provided." That is a property of
    the vault, and the check has to say so rather than call sync broken."""
    rig.plant_live_registration()
    payload, proc = rig.json("--component", "vault_sync",
                             spec=rig.spec(registry=True, e2e=True), round_trip=True)
    block = payload["vault_sync"]
    assert block["state"] == "unprovable-e2e", block
    assert block["kind"] == "unproven", block
    assert block["state"] not in shc.VAULT_SYNC_CANNOT_SYNC
    assert block["healthy"] is False
    assert payload["overall_status"] == "degraded" and proc.returncode == DEGRADED_EXIT
    assert rig.live_ids() == [rig.linked_id]


# --------------------------------------------------------------- clause 3
def test_a_planted_record_puts_its_timestamp_in_the_json(rig):
    rig.state_file.parent.mkdir(parents=True)
    rig.state_file.write_text(json.dumps({"at": "2026-09-01T10:00:00-07:00",
                                          "observed_by": "planted", "vault_id": rig.linked_id}))
    payload, _ = rig.json("--component", "vault_sync")
    last = payload["vault_sync"]["last_observed_sync"]
    assert last["status"] == "recorded" and last["at"] == "2026-09-01T10:00:00-07:00", last
    assert last["path"] == str(rig.state_file)
    assert last["healthy"] is False, "a record is history, never a verdict"


@pytest.mark.parametrize("shape", ["absent", "corrupt", "no-timestamp", "not-utf8"])
def test_no_usable_record_is_a_named_non_green_value(rig, shape):
    if shape != "absent":
        rig.state_file.parent.mkdir(parents=True)
        if shape == "not-utf8":
            rig.state_file.write_bytes(b"\xff\xfe{\x00")
        else:
            rig.state_file.write_text("{not json" if shape == "corrupt" else json.dumps({"x": 1}))
    payload, _ = rig.json("--component", "vault_sync")
    last = payload["vault_sync"]["last_observed_sync"]
    assert last["status"] == ("none-recorded" if shape == "absent" else "unreadable"), last
    assert last["healthy"] is False, last
    assert payload["vault_sync"]["healthy"] is False


def test_an_observed_round_trip_writes_the_record_and_a_failed_one_does_not(rig):
    rig.plant_live_registration()
    rig.json("--component", "vault_sync", spec=rig.spec(registry=True, replicate=False),
             round_trip=True)
    assert not rig.state_file.exists(), "a failed round trip wrote a success record"
    rig.json("--component", "vault_sync", spec=rig.spec(registry=True, replicate=True),
             round_trip=True)
    record = json.loads(rig.state_file.read_text())
    assert record["vault_id"] == rig.linked_id and record["at"], record
    payload, _ = rig.json("--component", "vault_sync")
    assert payload["vault_sync"]["last_observed_sync"]["at"] == record["at"]


def test_a_record_from_another_vault_is_not_reported_as_this_one(rig):
    rig.state_file.parent.mkdir(parents=True)
    rig.state_file.write_text(json.dumps({"at": "2026-09-01T10:00:00-07:00",
                                          "vault_id": "c73df6a0-old-vault"}))
    payload, _ = rig.json("--component", "vault_sync")
    last = payload["vault_sync"]["last_observed_sync"]
    assert last["status"] == "recorded-other-vault", last
    assert last["healthy"] is False


def test_a_refused_setup_on_a_managed_vault_is_a_sync_failure_not_unprovable(rig):
    """cli.js prints "Failed to validate password." for ANY refused /vault/access
    call — a network error, a quota refusal — so only the unanswered prompt
    ("Password not provided.") means the vault is end-to-end encrypted."""
    rig.plant_live_registration()
    payload, _ = rig.json("--component", "vault_sync",
                          spec=rig.spec(registry=True, setup_refused=True), round_trip=True)
    block = payload["vault_sync"]
    assert block["state"] == "sync-failed" and block["reason"] == "setup-failed", block


def test_an_unreadable_live_registration_after_a_green_trip_is_not_green(rig):
    rig.plant_live_registration()
    payload, proc = rig.json("--component", "vault_sync",
                             spec=rig.spec(registry=True, replicate=True, status_fail_after=1),
                             round_trip=True)
    block = payload["vault_sync"]
    assert block["state"] != "synced" and block["reason"] == "live-registration-unverified", block
    assert block["live_registration"] == "unverified", block
    assert block["probe_verdict"]["state"] == "synced", block
    assert proc.returncode == DEGRADED_EXIT


def test_a_temp_dir_that_cannot_be_made_still_takes_the_sentinel_out(rig, monkeypatch):
    """The sentinel is written into the live vault before the scratch directories
    exist; a full /tmp must not leave it there for the live client to upload."""
    import tempfile
    rig.plant_live_registration()
    for key, value in rig.env(rig.spec(registry=True)).items():
        monkeypatch.setenv(key, value)
    before = set(Path(tempfile.gettempdir()).glob(shc.SCRATCH_PREFIX + "*"))
    real, calls = shc.tempfile.mkdtemp, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real(*args, **kwargs)
    monkeypatch.setattr(shc.tempfile, "mkdtemp", flaky)
    with pytest.raises(OSError):
        shc._vault_sync_round_trip(str(rig.fake), rig.linked_id, str(rig.vault.resolve()), 20)
    assert list((rig.vault / "lloyd").glob("healthcheck-vault-sync-*.md")) == []
    assert set(Path(tempfile.gettempdir()).glob(shc.SCRATCH_PREFIX + "*")) == before
    assert rig.live_ids() == [rig.linked_id]
