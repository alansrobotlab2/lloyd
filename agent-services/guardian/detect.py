"""Pure failure predicates. No I/O, so every branch is table-testable.

The liveness predicate here deliberately **inverts** the priority used by
`app/supervisor_client.py::_health`, whose comment reads "Port being open is
the strongest signal — trust it over supervisord state". That is right for the
Services page and wrong for a watchdog: a FATAL backend whose :8080 is still
held by a zombie worker would read "healthy" and the crash would never be
noticed. Here supervisord state is consulted FIRST, FATAL is decisive, and a
probe may only ADD failure, never subtract it.
`tests/test_guardian_predicates.py` asserts the divergence by name so it stays
intentional.
"""

from __future__ import annotations

import hashlib
import re


STOPPED_STATES = {"STOPPED", "EXITED"}


def is_starting(info: dict, now: float, grace: float) -> bool:
    """True while a process is inside its post-spawn grace window."""
    start = float(info.get("start") or 0)
    if start <= 0:
        return False
    return (now - start) < grace


def crash_looping(start_history: list[float], *, starts: int, window: float, now: float) -> bool:
    """True when the spawn timestamp has advanced `starts` times inside `window`.

    This is the predicate that catches the pathology `autorestart=true` plus a
    too-small `startsecs` produces: supervisord marks the process RUNNING
    before it can actually serve, so a boot failure counts as an *unexpected
    exit* and is retried forever without ever parking in FATAL. Sampling
    `statename` would show RUNNING most of the time. Distinct spawn timestamps
    do not lie.
    """
    recent = [t for t in start_history if now - t <= window]
    return len(set(recent)) >= starts


def process_down(
    info: dict | None,
    *,
    now: float,
    grace: float,
    probe_fail_streak: int,
    probe_threshold: int,
    start_history: list[float],
    crash_loop_starts: int,
    crash_loop_window: float,
    intentional_stop: bool = False,
    probe_timeout_streak: int = 0,
    probe_timeout_threshold: int = 10**9,
) -> tuple[bool, str]:
    """Return (down, reason). See the module docstring for the ordering rule."""
    if info is None:
        return True, "unknown to supervisord"

    state = str(info.get("statename", "")).upper()

    # 1. supervisord state first. FATAL is decisive and needs no corroboration.
    if state == "FATAL":
        return True, f"FATAL: {info.get('spawnerr') or 'no spawnerr'}"
    if state in STOPPED_STATES and not intentional_stop:
        return True, f"{state} without an intentional stop"

    # 2. A crash loop that never reaches FATAL.
    if crash_looping(start_history, starts=crash_loop_starts,
                     window=crash_loop_window, now=now):
        return True, f"crash loop: {crash_loop_starts} spawns in {crash_loop_window:.0f}s"

    # 3. Only now may a probe contribute — and only outside the grace window.
    if state == "RUNNING":
        if is_starting(info, now, grace):
            return False, "starting"
        # A refused connection means nothing is listening. A timeout means the
        # socket accepted but the app was too busy to answer — for this backend
        # that is routine (an hourly autoresearch round pushes 77 bench trials
        # through the same event loop that serves /health), so it needs a much
        # longer budget before it counts as death.
        if probe_fail_streak >= probe_threshold:
            return True, f"health probe refused {probe_fail_streak} consecutive times"
        if probe_timeout_streak >= probe_timeout_threshold:
            return True, (f"health probe timed out {probe_timeout_streak} consecutive "
                          f"times — unresponsive, not merely busy")
        return False, "running"

    if state in ("STARTING", "BACKOFF"):
        return False, state.lower()
    return False, state.lower() or "unknown"


def mcp_degraded_is_fatal(current: dict, baseline_degraded: list[str] | None) -> tuple[bool, str]:
    """Decide whether an MCP `degraded` reading warrants a rollback.

    `agent_mcp/main.py`'s /health returns 503 if ANY module is degraded, but a
    degraded aggregator still serves 100+ of ~124 tools — reverting on one
    degraded module is too aggressive, and the modules that degrade most often
    are the external-app bridges (Thunderbird closed, browser unavailable).

    Fire only when the aggregator serves nothing at all, or when a module that
    was healthy in the last-known-good snapshot has newly broken.
    """
    tools = int(current.get("tools") or 0)
    if tools == 0:
        return True, "aggregator advertises zero tools"
    now_degraded = set(current.get("degraded_modules") or [])
    if not now_degraded:
        return False, "ok"
    was_degraded = set(baseline_degraded or [])
    newly = sorted(now_degraded - was_degraded)
    if newly:
        return True, f"modules degraded since last known good: {newly}"
    return False, f"pre-existing degradation only: {sorted(now_degraded)}"


# ---------------------------------------------------------------------------
# Log signatures
# ---------------------------------------------------------------------------

_NUM = re.compile(r"\d+")
_HEX = re.compile(r"\b[0-9a-f]{7,}\b", re.IGNORECASE)
_PATH = re.compile(r"(/[\w.\-]+){2,}")
_DUR = re.compile(r"\b\d+(\.\d+)?(ms|s|m|h)\b")
_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.]\d+)\s+"
    r"\[(?P<level>[A-Z]+)\]\s+(?P<logger>[\w.\-]+):\s*(?P<msg>.*)$"
)


def normalize_message(msg: str) -> str:
    """Collapse the varying parts of a log message so repeats share a signature.

    `…oldest claimable queue item is 1266 min old` and `…1300 min old` must
    reduce to the same thing, or a chronic error would present as an endless
    stream of novel ones.
    """
    out = _PATH.sub("<path>", msg)
    out = _DUR.sub("<dur>", out)
    out = _HEX.sub("<hex>", out)
    out = _NUM.sub("<n>", out)
    return " ".join(out.split())[:400]


def signature(logger_name: str, msg: str) -> str:
    raw = f"{logger_name}|{normalize_message(msg)}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]


def parse_log_line(line: str) -> dict | None:
    """Parse `%(asctime)s [%(levelname)s] %(name)s: %(message)s`."""
    m = _LINE.match(line.strip())
    if not m:
        return None
    d = m.groupdict()
    return {
        "level": d["level"],
        "logger": d["logger"],
        "message": d["msg"],
        "signature": signature(d["logger"], d["msg"]),
    }


def extract_events(text: str) -> list[dict]:
    """Structured error events from a chunk of log text.

    Traceback headers are counted once per exception rather than per frame, so
    one raised error is one event. `[WARNING]` is never an error: production
    emits 20+ per 20k lines at steady state from the worker pool, the autonomy
    scheduler and the MCP pool.
    """
    events: list[dict] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last):"):
            # A traceback is: header, indented frames, then ONE non-indented
            # exception line. Stop at that line — scanning a fixed window
            # ahead makes consecutive tracebacks all report the *last* one's
            # exception, collapsing three distinct failures into one
            # signature and hiding a real spike.
            exc_line = ""
            end = i + 1
            for j in range(i + 1, min(i + 40, len(lines))):
                candidate = lines[j]
                if candidate.strip() and not candidate.startswith((" ", "\t")):
                    exc_line = candidate.strip()
                    end = j + 1
                    break
                end = j + 1
            events.append({
                "level": "TRACEBACK",
                "logger": "traceback",
                "message": exc_line,
                "signature": signature("traceback", exc_line),
                "text": "\n".join(lines[i:end])[:2000],
            })
            continue
        parsed = parse_log_line(line)
        if parsed and parsed["level"] in ("ERROR", "CRITICAL"):
            parsed["text"] = line[:2000]
            events.append(parsed)
    return events


def error_spike(
    events: list[dict],
    *,
    chronic: set[str],
    changed_paths: list[str],
    novel_threshold: int,
    fatal_distinct_threshold: int,
    changed_path_threshold: int,
) -> tuple[bool, str]:
    """Decide whether a batch of log events warrants a rollback.

    Chronic signatures can never fire. Production already emits recurring
    `[ERROR] lloyd-workers.scheduled_task: autonomy scheduler may be stalled`
    and a `discord_alert (no channel/token configured)` echo — a detector that
    counted those would fire on its first tick, every time.
    """
    novel = [e for e in events if e["signature"] not in chronic]
    if not novel:
        return False, "no novel signatures"

    counts: dict[str, int] = {}
    for e in novel:
        counts[e["signature"]] = counts.get(e["signature"], 0) + 1

    # Strongest rule, and it costs nothing: a novel traceback naming a file the
    # promotion just edited is causal evidence, not correlation.
    if changed_paths:
        for e in novel:
            blob = e.get("text") or e.get("message") or ""
            if any(p and p in blob for p in changed_paths):
                if counts[e["signature"]] >= changed_path_threshold:
                    return True, (f"novel error naming a changed path "
                                  f"({counts[e['signature']]}x): {e['message'][:120]}")

    for sig, n in counts.items():
        if n >= novel_threshold:
            sample = next(e["message"] for e in novel if e["signature"] == sig)
            return True, f"novel signature x{n}: {sample[:120]}"

    distinct_fatal = {e["signature"] for e in novel if e["level"] in ("CRITICAL", "TRACEBACK")}
    if len(distinct_fatal) >= fatal_distinct_threshold:
        return True, f"{len(distinct_fatal)} distinct novel fatal signatures"

    return False, f"{len(novel)} novel events below threshold"


def cusum_update(score: float, failed: bool, p0: float, p1: float, floor: float) -> float:
    """One-sided CUSUM step. Returns the updated score, clamped at 0.

    Rate comparison is unusable here: ~14 worker runs/hour means a 30-minute
    window holds about 7 samples. CUSUM fires as fast as the evidence allows
    and self-calibrates per source — at the measured p0=0.011 for
    `session-distill` two consecutive failures are already decisive, which is
    right, because that source has never failed twice in a row in 322 runs.
    """
    import math
    p0 = max(float(p0), floor)
    p1 = max(float(p1), p0 + 1e-6)
    step = math.log(p1 / p0) if failed else math.log((1.0 - p1) / (1.0 - p0))
    return max(0.0, score + step)


def data_damage(before: int | None, after: int | None, fraction: float) -> tuple[bool, str]:
    """Detect the failure class `git reset --hard` cannot undo.

    The knowledge graph and the vault are gitignored, so a change that deletes
    rows or notes boots fine, logs nothing, passes every eval, and *survives*
    the revert. Two cheap counts cover it.
    """
    if not before or after is None:
        return False, "no baseline"
    if before <= 0:
        return False, "empty baseline"
    drop = (before - after) / before
    if drop > fraction:
        return True, f"dropped {drop * 100:.1f}% ({before} → {after})"
    return False, f"delta {drop * 100:+.1f}%"
