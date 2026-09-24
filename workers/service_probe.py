"""Service probe — a program supervisord keeps up whose port never answers (#1359).

On 2026-09-21 djev crash-looped nine times in 57 minutes: each boot cleared
`startsecs`, so supervisord called it RUNNING, and each died 65–155 s after
spawn on its own readiness check. Every surface that trusts supervisord's state
agreed djev was healthy. The probe that would have caught all nine — RUNNING
with its declared port closed — already existed in two places (the dashboard's
services panel and `scripts/service_health_check.py`), and both are pull-only:
nothing looks unless a person opens a tab or invokes a skill.

This is the push half, run from the pool's scheduler loop beside the poison
sweep, deterministic and on the same 60 s tick. For every infra service with a
declared port (`supervisor_client.infra_services()`, so a slot config.yaml has
switched off is never probed) it tracks how long the port has been closed while
supervisord still has the program in hand. Past the service's grace it
announces once; when the port opens again it announces the recovery once.

Three rules, each the reason for a line below:

  * **The streak survives a respawn.** A crash loop passes through EXITED,
    STARTING and BACKOFF between its RUNNING spells, and a 60 s tick can land
    on any of them. Only an open port or a deliberate STOPPED ends a streak;
    reading a respawn as recovery would reset the clock every cycle and the
    loop would never be reported — the defect being fixed.
  * **The grace outlasts a cold boot.** djev's cold torch.compile/Triton pass
    runs minutes, and the primary's legitimate boot is up to 20 minutes
    (`round restart --only agent-llm-primary` waits that long for /health). A
    probe that fires on every restart is one a person learns to dismiss.
  * **`announce`, not `alert`.** `notify.py`'s `alert` writes a ledger row,
    ALERT.md and, when critical, a backlog task — it is for incidents the
    guardian owns. This is news for whoever is in the room: journal and toast
    at `critical`, no voice, no bookkeeping. The lloyd services are left out
    for the same reason: the guardian already watches the backend and the
    aggregator, and a second voice about the same outage is noise.

An unreachable supervisord skips the tick without touching any streak: it says
nothing about the programs, and a canary pointing `LLOYD_SUPERVISOR_SOCK` at a
dead path must stay silent.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("lloyd-workers.service_probe")

#: Seconds a port may stay closed while its program is in supervisord's hands.
DEFAULT_GRACE_S = 15 * 60
#: The primary's boot legitimately takes up to 20 minutes.
GRACE_S = {"agent-llm-primary": 30 * 60}

# A deliberate stop ends a streak; everything else supervisord reports while it
# still owns the program (RUNNING, STARTING, BACKOFF, EXITED on the way to a
# respawn, FATAL once it gives up) keeps it going.
_STOPPED = "STOPPED"

Announce = Callable[[str, str, str], object]


@dataclass
class _Streak:
    since: float
    state: str
    announced: bool = False


@dataclass
class ServiceProbe:
    announce: Optional[Announce] = None
    clock: Callable[[], float] = time.monotonic
    streaks: dict[str, _Streak] = field(default_factory=dict)

    def tick(self, services: dict, procs: dict,
             port_open: Callable[[int], bool]) -> list[dict]:
        """One pass. `services` is {name: (display, port)}, `procs` is
        supervisord's {name: process_info}. Returns what was announced."""
        now = self.clock()
        events: list[dict] = []
        seen: set[str] = set()
        for name, (display, port) in services.items():
            if not port:
                continue
            seen.add(name)
            proc = procs.get(name)
            state = (proc or {}).get("statename", "")
            if proc is None or state == _STOPPED or port_open(port):
                streak = self.streaks.pop(name, None)
                if streak and streak.announced and proc is not None \
                        and state != _STOPPED:
                    events.append(self._say(
                        "recovered", name,
                        f"{display} is answering on :{port} again",
                        f"{name} was not serving for "
                        f"{int(now - streak.since) // 60} min.", "info"))
                continue
            streak = self.streaks.get(name)
            if streak is None:
                self.streaks[name] = _Streak(since=now, state=state)
                continue
            streak.state = state
            grace = GRACE_S.get(name, DEFAULT_GRACE_S)
            if not streak.announced and now - streak.since >= grace:
                streak.announced = True
                events.append(self._say(
                    "down", name,
                    f"{display} is not serving: :{port} closed while supervisord "
                    f"says {state}",
                    f"{name} has held supervisord state {state} (or cycled "
                    f"through respawns) for {int(now - streak.since) // 60} min "
                    f"without :{port} opening — a crash loop reads as RUNNING "
                    f"to everything that trusts supervisord's state (#1359). "
                    f"Its log: supervisorctl tail {name} stderr.", "critical"))
        # A slot switched off since the last tick is no longer ours to judge.
        for gone in set(self.streaks) - seen:
            self.streaks.pop(gone, None)
        return events

    def _say(self, kind: str, name: str, title: str, body: str,
             level: str) -> dict:
        (logger.error if level == "critical" else logger.info)(
            "service_probe: %s — %s", title, body)
        channels: object = None
        if self.announce is not None:
            try:
                channels = self.announce(title, body, level)
            except Exception as exc:  # noqa: BLE001 — a bell is not the probe
                logger.warning("service_probe: announce failed: %s", exc)
        return {"kind": kind, "service": name, "title": title,
                "level": level, "channels": channels}


def guardian_announce(title: str, body: str, level: str) -> dict:
    """Through the guardian's one fan-out, the route `app/prefix_miss.py` and
    the promoter take: journal and toast, no voice."""
    import sys
    from pathlib import Path

    gdir = Path(__file__).resolve().parents[1] / "agent-services" / "guardian"
    if not (gdir / "notify.py").is_file():
        return {}
    if str(gdir) not in sys.path:
        sys.path.insert(0, str(gdir))
    import gstate
    import notify as notify_mod
    import policy

    notifier = notify_mod.Notifier(
        ledger=gstate.AutomodState(Path(policy.AUTOMOD_STATE)).ledger,
        state_dir=Path(policy.GUARDIAN_STATE),
        vault_root=policy.VAULT_ROOT,
        voice=False,
        voice_window=policy.VOICE_REPEAT_SECONDS,
    )
    return notifier.announce(title, body, level=level)


def run_probe(probe: ServiceProbe) -> list[dict]:
    """Read supervisord and the ports, then tick. Blocking — call off the loop."""
    from app import supervisor_client as sc

    try:
        procs = sc._supervisor_all()
    except sc.SupervisordUnreachable:
        return []
    return probe.tick(sc.infra_services(), procs, sc._port_open)
