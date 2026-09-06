"""Host telemetry — CPU, memory, disk, and NVIDIA GPUs.

Feeds the Mission Control dashboard's system panel. Everything here is
best-effort: a missing `nvidia-smi`, an unreadable mount, or a psutil
call that raises on this kernel degrades to `None`/`[]` rather than
failing the dashboard request.

`cpu_percent` deserves a note. `psutil.cpu_percent()` with no interval
returns utilisation since the *previous call in this process*, which for
a polled endpoint is exactly the poll interval — the right window, and
free. Passing an interval instead would block the event loop for that
long on every poll. The first call after boot returns 0.0 (no baseline);
we prime it at import so the dashboard's first render is already real.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from typing import Any

import psutil

logger = logging.getLogger("lloyd-server")

# Mounts worth showing. Anything absent is skipped silently.
_DISK_PATHS = ("/", "/home")

# nvidia-smi costs ~40-80ms and the dashboard polls faster than GPU state
# meaningfully changes, so results are cached briefly. This also stops a
# handful of concurrent dashboard clients from forking a process each per
# poll.
_GPU_CACHE_TTL_S = 2.0
_gpu_cache: tuple[float, list[dict[str, Any]]] | None = None
_gpu_lock = asyncio.Lock()

_NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
)

# Prime psutil's per-process CPU baseline so the first poll isn't 0.0.
psutil.cpu_percent(interval=None)


def _num(raw: str) -> float | None:
    """Parse one nvidia-smi CSV cell; '[N/A]' and junk become None."""
    raw = raw.strip()
    if not raw or raw.startswith("["):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


async def gpus() -> list[dict[str, Any]]:
    """Per-GPU utilisation, memory, temperature and power draw.

    Returns `[]` when nvidia-smi is absent or fails — a CPU-only host is
    a valid configuration, not an error worth surfacing.
    """
    global _gpu_cache

    now = time.monotonic()
    if _gpu_cache is not None and now - _gpu_cache[0] < _GPU_CACHE_TTL_S:
        return _gpu_cache[1]

    async with _gpu_lock:
        # Re-check: another caller may have refreshed while we waited.
        now = time.monotonic()
        if _gpu_cache is not None and now - _gpu_cache[0] < _GPU_CACHE_TTL_S:
            return _gpu_cache[1]

        if shutil.which("nvidia-smi") is None:
            _gpu_cache = (now, [])
            return []

        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                f"--query-gpu={','.join(_NVIDIA_SMI_FIELDS)}",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=4.0)
        except (asyncio.TimeoutError, OSError) as exc:
            logger.debug("nvidia-smi failed: %s", exc)
            _gpu_cache = (now, [])
            return []

        out: list[dict[str, Any]] = []
        for line in stdout.decode("utf-8", "replace").splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < len(_NVIDIA_SMI_FIELDS):
                continue
            used, total = _num(cells[4]), _num(cells[5])
            out.append({
                "index": int(_num(cells[0]) or 0),
                "name": cells[1],
                "gpu_util": _num(cells[2]),
                "mem_util": _num(cells[3]),
                "memory_used_mb": used,
                "memory_total_mb": total,
                "memory_pct": (used / total * 100) if used and total else None,
                "temperature_c": _num(cells[6]),
                "power_draw_w": _num(cells[7]),
                "power_limit_w": _num(cells[8]),
            })
        _gpu_cache = (now, out)
        return out


def _disks() -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for path in _DISK_PATHS:
        try:
            usage = psutil.disk_usage(path)
        except OSError:
            continue
        # /home is often the same filesystem as / — don't list it twice.
        key = f"{usage.total}:{usage.used}"
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "path": path,
            "used_bytes": usage.used,
            "total_bytes": usage.total,
            "percent": usage.percent,
        })
    return out


def _load_average() -> list[float] | None:
    try:
        return list(psutil.getloadavg())
    except (AttributeError, OSError):
        return None


async def collect() -> dict[str, Any]:
    """One host snapshot for the dashboard."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    boot = psutil.boot_time()

    return {
        "cpu": {
            # Since the previous call — i.e. over the last poll interval.
            "percent": psutil.cpu_percent(interval=None),
            "count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "load_average": _load_average(),
        },
        "memory": {
            "used_bytes": vm.total - vm.available,
            "total_bytes": vm.total,
            "percent": vm.percent,
        },
        "swap": {
            "used_bytes": swap.used,
            "total_bytes": swap.total,
            "percent": swap.percent,
        },
        "disks": _disks(),
        "gpus": await gpus(),
        "uptime_seconds": max(0.0, time.time() - boot),
        "boot_time": boot,
    }
