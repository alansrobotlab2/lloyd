"""#1727 — the systemd units that give the tailnet leaf renewal an owner.

`scripts/renew-tailnet-cert.sh` is only the owner if something runs it. The box's
other scheduled work arrives as a `lloyd-*.timer` pair under
`agent-services/systemd/` installed into `~/.config/systemd/user/` (SETUP.md:1151
is the route for `lloyd-vault-backup.timer`), and before #1727
`systemctl --user list-timers` counted five timers and none of them a cert one.
So this pins the two files as configuration: parsed through the same INI shape
systemd reads, the service must be a oneshot that runs that script, and the timer
must be a recurring one that wants `timers.target`.

Installing and enabling them is host state and stays with the operator; these
assertions hold whether or not the units are linked.

Run: .venvs/lloyd/bin/python -m pytest tests/test_cert_renew_units.py
"""
import configparser
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIT_DIR = ROOT / "agent-services" / "systemd"
SERVICE = UNIT_DIR / "lloyd-cert-renew.service"
TIMER = UNIT_DIR / "lloyd-cert-renew.timer"
SCRIPT = "scripts/renew-tailnet-cert.sh"


def _parse(path: Path) -> configparser.ConfigParser:
    """Read a unit file the way systemd's loader does: sections, `Key=value`, no
    interpolation (a `%h` in a value is systemd's own syntax, not configparser's),
    and key case preserved."""
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str
    cp.read(path, encoding="utf-8")
    return cp


def _env_lines(path: Path) -> list[str]:
    """Every `Environment=` line in the unit. systemd applies all of them, while
    an INI read keeps only the last, so the PATH assertion has to see them all."""
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip().startswith("Environment=")]


def test_both_units_are_tracked_in_the_repo():
    """Untracked units are host drift: nothing in a fresh checkout or another
    machine would ever get the renewal, which is the failure #1727 is about."""
    for path in (SERVICE, TIMER):
        assert path.exists(), path
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                            str(path.relative_to(ROOT))],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{path.name} is not tracked: {r.stderr.strip()}"


def test_the_service_is_a_oneshot_that_runs_the_renewal_script():
    """Clause 5, first half: `Type=oneshot` and an `ExecStart` naming the script.
    A service that never runs the owner is the same gap with a timer attached."""
    cp = _parse(SERVICE)
    assert cp.has_section("Service"), SERVICE
    assert cp.get("Service", "Type") == "oneshot"
    execstart = cp.get("Service", "ExecStart")
    assert execstart.endswith(SCRIPT), execstart
    assert execstart.startswith("/"), f"ExecStart must be an absolute path: {execstart}"
    assert "renew-tailnet-cert.sh" in Path(execstart.split()[-1]).name


def test_the_service_can_reach_the_tools_the_script_needs():
    """`tailscale` and `openssl` are in /usr/bin, but `supervisorctl` is a
    ~/.local/bin tool, and a unit whose PATH does not name it fails the restart
    at exactly the moment the cert has changed."""
    env = _env_lines(SERVICE)
    path_lines = [l for l in env if l.split("=", 1)[1].startswith("PATH=")]
    assert path_lines, f"no Environment=PATH= in the unit: {env}"
    value = path_lines[0].split("=", 1)[1][len("PATH="):]
    parts = value.split(":")
    assert any(p.endswith(".local/bin") for p in parts), value
    assert "/usr/bin" in parts, value
    assert "/bin" in parts, value


def test_the_timer_is_recurring_and_wanted_by_timers_target():
    """Clause 5, second half: a schedule that fires more than once. OnCalendar is
    the recurring leg (an OnUnitActiveSec-only timer would never start after a
    reboot with the box idle), and `[Install] WantedBy=timers.target` is what
    `systemctl --user enable` needs to act on at all."""
    cp = _parse(TIMER)
    assert cp.has_section("Timer"), TIMER
    on_calendar = cp.get("Timer", "OnCalendar")
    assert on_calendar.strip()
    date_field, time_field = _calendar_fields(TIMER)
    assert date_field.count("-") == 2 or "/" in date_field, on_calendar
    assert time_field.count(":") >= 1, on_calendar
    assert cp.get("Timer", "Persistent", fallback="").lower() == "true"
    assert cp.get("Install", "WantedBy") == "timers.target"


def _calendar_fields(timer_path: Path) -> tuple[str, str]:
    """The (date, time) halves of the unit's OnCalendar. systemd's syntax is
    `[DayOfWeek] Year-Month-Day Hour:Minute:Second`, so a rule with no weekday
    list has two whitespace-separated fields and one with a weekday list has
    three; both are read here as the same two parts."""
    cp = _parse(timer_path)
    fields = cp.get("Timer", "OnCalendar").split()
    assert fields, f"{timer_path}: OnCalendar is empty"
    return fields[-2], fields[-1]


def test_the_timer_checks_at_least_once_a_day():
    """The window is 14 days, so a daily check is what leaves a mintable leaf at
    most one day stale; a weekly timer would burn a third of the runway before
    the first check ever ran."""
    date_field, _ = _calendar_fields(TIMER)
    assert date_field == "*-*-*", f"{date_field} is not every day of every month"


def test_the_units_name_the_script_that_exists():
    """A unit pointing at a script the repo does not have is a timer that fails
    in the journal and nothing else, which is how a renewal owner goes unseen for
    76 days."""
    assert (ROOT / SCRIPT).exists(), SCRIPT
    assert (ROOT / SCRIPT).stat().st_mode & 0o111, "the script is not executable"
    head = (ROOT / SCRIPT).read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("#!") and "bash" in head
