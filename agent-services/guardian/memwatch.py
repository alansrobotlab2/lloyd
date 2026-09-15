"""Memory-pressure evidence: what was holding the memory when oomd fired.

On 2026-09-15 at 04:48:34Z systemd-oomd killed `agent-supervisord.service` —
796 processes, 174.6 GiB — and nothing on the machine could say which process
had grown. oomd logs the cgroup it chose and the journal logs the count; every
process that might have said more was among the 796. The same kill took the
whole unit twice on 2026-09-08. The cost this time was an implement turn
killed mid-round and a loop held closed for thirteen and a half hours.

Two facts about the kill decide the shape of this module. oomd watches
`app.slice` (`/usr/lib/systemd/user/app.slice.d/10-oomd.conf`) and, past 50%
full pressure for 20 s (`/etc/systemd/oomd.conf.d/10-omarchy.conf`), kills the
descendant with the most reclaim activity — which is the stack, whatever else
in the slice caused the pressure. So the snapshot covers every process, not
just the unit's, and says which cgroup each belongs to. And the guardian is a
separate unit, so it survives the kill it is recording.

Every tick reads three small files (the unit's, `app.slice`'s and the host's
`memory.pressure`). Below `TRIGGER_FULL_AVG10` that is all. At or above it a
snapshot is written to `<guardian state>/mem-pressure/`, at most once per
`MIN_INTERVAL_SECONDS`, newest `KEEP` kept: the three pressure readings,
`/proc/meminfo`, the unit's `memory.stat`, and the top processes by RSS with
their anon/file/shmem split. Reading a process's `cmdline` touches its memory,
which can block on a page-in under exactly this pressure, so it is read only
for the processes that make the list, and the walk stops early past
`WALK_BUDGET_SECONDS`.

Stdlib only, like everything the guardian imports. Also a CLI:
    memwatch.py latest [N]     # the newest snapshot's top N processes
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# oomd kills at 50% for 20 s; the snapshots must start well before that.
TRIGGER_FULL_AVG10 = 15.0
MIN_INTERVAL_SECONDS = 15.0
KEEP = 120
TOP_N = 30
WALK_BUDGET_SECONDS = 3.0

UNIT = "agent-supervisord.service"


def unit_cgroup(unit: str = UNIT) -> Path:
    uid = os.getuid()
    return Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}")


def parse_pressure(text: str) -> dict:
    """`some avg10=1.00 avg60=... total=...` lines → {"some": {...}, "full": {...}}."""
    out: dict = {}
    for line in text.splitlines():
        kind, _, rest = line.strip().partition(" ")
        if kind not in ("some", "full"):
            continue
        vals = {}
        for tok in rest.split():
            k, _, v = tok.partition("=")
            try:
                vals[k] = float(v)
            except ValueError:
                continue
        out[kind] = vals
    return out


def read_pressure(path: Path) -> dict | None:
    try:
        return parse_pressure(path.read_text())
    except OSError:
        return None


def _full_avg10(p: dict | None) -> float:
    return float(((p or {}).get("full") or {}).get("avg10") or 0.0)


def _kb_fields(text: str, keys: tuple[str, ...]) -> dict:
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in keys:
            try:
                out[k] = int(v.split()[0])
            except (ValueError, IndexError):
                pass
    return out


def _cgroup_name(pid: str) -> str:
    try:
        line = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[-1]
    except (OSError, IndexError):
        return ""
    path = line.partition("::")[2]
    # The unit or scope is the informative part; a supervisord child sits at
    # the unit itself, a desktop app in its own app-*.scope.
    return path.rsplit("/", 1)[-1]


def top_processes(limit: int = TOP_N, proc: Path = Path("/proc"),
                  budget: float = WALK_BUDGET_SECONDS) -> tuple[list[dict], bool]:
    """Processes by RSS, largest first, and whether the walk finished."""
    deadline = time.monotonic() + budget
    rows: list[dict] = []
    complete = True
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        if time.monotonic() > deadline:
            complete = False
            break
        try:
            status = (proc / pid / "status").read_text()
        except OSError:
            continue
        f = _kb_fields(status, ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"))
        if not f.get("VmRSS"):
            continue   # kernel threads
        name = next((ln.split(":", 1)[1].strip() for ln in status.splitlines()
                     if ln.startswith("Name:")), "")
        rows.append({"pid": int(pid), "name": name,
                     "rss_kb": f.get("VmRSS", 0), "anon_kb": f.get("RssAnon", 0),
                     "file_kb": f.get("RssFile", 0), "shmem_kb": f.get("RssShmem", 0),
                     "swap_kb": f.get("VmSwap", 0)})
    rows.sort(key=lambda r: r["rss_kb"], reverse=True)
    rows = rows[:limit]
    for r in rows:
        r["cgroup"] = _cgroup_name(str(r["pid"]))
        if time.monotonic() > deadline:
            continue
        try:
            raw = (proc / str(r["pid"]) / "cmdline").read_bytes()[:300]
            r["cmdline"] = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            pass
    return rows, complete


class MemWatch:
    def __init__(self, state_dir: Path, cgroup: Path | None = None,
                 host: Path = Path("/proc/pressure/memory"), *,
                 trigger: float = TRIGGER_FULL_AVG10, min_interval: float = MIN_INTERVAL_SECONDS,
                 keep: int = KEEP):
        self.cgroup = cgroup or unit_cgroup()
        self.host = host
        self.out_dir = Path(state_dir) / "mem-pressure"
        self.trigger = trigger
        self.min_interval = min_interval
        self.keep = keep
        self.last_ts = 0.0

    def readings(self) -> dict:
        return {"unit": read_pressure(self.cgroup / "memory.pressure"),
                "slice": read_pressure(self.cgroup.parent / "memory.pressure"),
                "host": read_pressure(self.host)}

    def tick(self, now: float | None = None) -> Path | None:
        """A snapshot path when one was written, else None. Never raises for a
        missing file: the unit's cgroup is absent while it is down."""
        now = time.time() if now is None else now
        pressure = self.readings()
        peak = max(_full_avg10(p) for p in pressure.values())
        if peak < self.trigger or now - self.last_ts < self.min_interval:
            return None
        self.last_ts = now
        return self.write(self.snapshot(pressure, now, peak))

    def snapshot(self, pressure: dict, now: float, peak: float) -> dict:
        procs, complete = top_processes()
        snap = {"ts": now, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "peak_full_avg10": peak, "pressure": pressure,
                "processes": procs, "walk_complete": complete}
        try:
            snap["meminfo_kb"] = _kb_fields(Path("/proc/meminfo").read_text(), (
                "MemTotal", "MemAvailable", "Cached", "Shmem", "SwapTotal", "SwapFree"))
        except OSError:
            pass
        try:
            snap["unit_memory_stat_kb"] = {
                k: v // 1024 for k, v in (
                    (ln.split()[0], int(ln.split()[1]))
                    for ln in (self.cgroup / "memory.stat").read_text().splitlines())
                if k in ("anon", "file", "shmem", "file_mapped", "kernel")}
            snap["unit_memory_current_kb"] = int((self.cgroup / "memory.current").read_text()) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return snap

    def write(self, snap: dict) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime(snap['ts']))}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, indent=1), encoding="utf-8")
        tmp.replace(path)
        for old in sorted(self.out_dir.glob("*.json"))[:-self.keep]:
            try:
                old.unlink()
            except OSError:
                pass
        return path


def _latest(state_dir: Path, n: int) -> int:
    snaps = sorted((state_dir / "mem-pressure").glob("*.json"))
    if not snaps:
        print("no memory-pressure snapshots")
        return 0
    snap = json.loads(snaps[-1].read_text())
    print(f"{snaps[-1]}  ({len(snaps)} kept)  peak full avg10 {snap['peak_full_avg10']:.1f}%")
    avail = (snap.get("meminfo_kb") or {}).get("MemAvailable")
    if avail is not None:
        print(f"MemAvailable {avail / 1048576:.1f} GiB")
    for p in snap["processes"][:n]:
        print(f"{p['rss_kb'] / 1048576:7.1f}G  anon {p['anon_kb'] / 1048576:6.1f}G  "
              f"shm {p['shmem_kb'] / 1048576:6.1f}G  {p['pid']:>8}  {p.get('cgroup', '')[:34]:34}  "
              f"{(p.get('cmdline') or p['name'])[:80]}")
    return 0


if __name__ == "__main__":
    state = Path(os.environ.get("LLOYD_GUARDIAN_STATE",
                                Path.home() / ".local/state/lloyd-guardian"))
    if len(sys.argv) >= 2 and sys.argv[1] == "latest":
        sys.exit(_latest(state, int(sys.argv[2]) if len(sys.argv) > 2 else 15))
    print(__doc__.strip().splitlines()[-1].strip())
    sys.exit(2)
