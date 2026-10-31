"""scheduled-task source — wraps the existing autonomy task files.

Reads ~/obsidian/autonomy/*.md each tick, evaluates `due-ness` per task
(frequency / runs_per_day / preferred_hours / depends_on), and enqueues the
highest-priority due tasks. Execution delegates to autonomy.run_task() which
still writes the per-task markdown run records under autonomy-runs/{task_id}/.

Dispatch only, since #1682: the fleet's alarms — both stall alarms and the
unparseable-task-file scan — live in `workers/fleet_watchdog.py` and are driven by
`WorkerPool._scheduler_loop`, so they fire whether or not this source is enabled.
What is left here is the vLLM dispatch gate (#938: it pauses enqueuing, and the
watching went on without it), the stuck-claim reset that gate has always sat behind,
`get_due_tasks` and the enqueue loop, and the execution of a claimed item.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.scheduled_task")

NAME = "scheduled-task"
DEFAULT_PRIORITY = 30

#: This source's work IS one of our task files, so a death that never reached a
#: verdict may charge that file's retry budget (#2037). `WorkerPool._death_meta`
#: reads the flag, and a source without it is classified but never charged —
#: `bench-mine` payloads carry a `task_id` naming an autonomy task the run is
#: MINING, not a task the run is, and charging from the payload alone would
#: retire someone else's schedule.
CHARGE_TASK_ON_DEATH = True

_PRIORITY_MAP = {
    "critical": 10,
    "high": 20,
    "medium": 30,
    "low": 50,
    "background": 70,
}

# vLLM health gate: skip enqueuing when the model server is unreachable so a wedge
# doesn't turn every due task into a ConnectError flood.
#
# This literal is the LAST-RESORT FALLBACK only (#1684). The endpoint the gate
# actually probes is derived from config by `_primary_health_target`, the same way
# a named model's leg always has been: the fleet-wide leg was the one leg reading
# a hardcoded port, so moving the primary's port under `models:` paused ALL
# dispatch on a URL the engine never served, and the alert that eventually arrived
# named that stale URL while the fleet sat stopped on it.
_VLLM_HEALTH_URL = "http://127.0.0.1:8096/health"
#: Provenance to print when the derivation itself could not run, so the outage
#: alert still names where its URL came from (#1684).
_VLLM_URL_FALLBACK_KEY = ("config unreadable — workers/sources/scheduled_task.py "
                          "fallback literal")


def _primary_health_target() -> tuple[str, str]:
    """(health URL, config key) for the fleet gate's primary leg.

    Resolved through `app.config.default_model_base_url_source`, which reads the
    default model's `models.<name>.base_url` (then its `env.ANTHROPIC_BASE_URL`)
    — the key the engine and the rest of the app treat as the source of truth —
    and yields the literal above only when config cannot be read at all.

    The key travels with the URL because a surface that stops the fleet on a
    failed probe has to say which endpoint it probed and where that came from: a
    port move then diagnoses itself instead of producing an alert about a port
    nobody moved (#1684).
    """
    try:
        from app.config import default_model_base_url_source
        base, source_key = default_model_base_url_source()
    except Exception:
        return _VLLM_HEALTH_URL, _VLLM_URL_FALLBACK_KEY
    if not base:
        return _VLLM_HEALTH_URL, _VLLM_URL_FALLBACK_KEY
    return base.rstrip("/") + "/health", source_key


def _read_model_table() -> dict:
    """`models:` from config.yaml as a dict, `{}` when it cannot be read.

    Read off disk for the same reason `_read_primary_health_config` is: the gate
    runs inside the pool's process, which may have imported `app.config` before an
    edit, and `app.autonomy._get_model_env` has always re-read the file per call.
    Never raises — the caller's fail-open verdict depends on `{}` meaning "no
    answer", which is why an unreadable file cannot be reported as an empty table.
    """
    import yaml
    try:
        # `app.autonomy.LLOYD_HOME`, read the way `app.autonomy._get_model_env` reads
        # it — the same file, and the attribute the named leg has always been tested
        # through, so one redirect moves both readers instead of leaving the env leg
        # on a scratch config while the table leg still sees the shipped one.
        from app import autonomy
        cfg = yaml.safe_load(
            (autonomy.LLOYD_HOME / "config.yaml").read_text(encoding="utf-8"))
        models = (cfg or {}).get("models") or {}
        return models if isinstance(models, dict) else {}
    except Exception:
        return {}


def _model_endpoint(model: str) -> str | None:
    """A model's own health endpoint, or None when config does not name one.

    `env.ANTHROPIC_BASE_URL` first and `base_url` as the second source, which is the
    order the app itself resolves a model's endpoint in — `app.config._model_base_url`
    (`env` block, then the flat key) and `default_model_base_url_source`, whose whole
    contract is that the default's endpoint is `<name>.base_url` only when its env
    block has no `ANTHROPIC_BASE_URL`. `_primary_health_target` already reads both;
    this leg read only the env, so a model added with `base_url` and an env block that
    omits the key — the shape #1684 pushed the primary leg toward, and the shape both
    shipped entries happen to avoid by carrying the key twice with the same value —
    answered with the PRIMARY's endpoint, and the gate watched a server the task's runs
    never touched (#3517).
    """
    entry = _read_model_table().get(model)
    if not isinstance(entry, dict):
        return None
    env = entry.get("env") if isinstance(entry.get("env"), dict) else {}
    for source in (env.get("ANTHROPIC_BASE_URL"), entry.get("base_url")):
        if isinstance(source, str) and source.strip():
            return source.strip().rstrip("/") + "/health"
    return None


#: The class `run_task` stamps on its own refusal of an undeclared pin (#1209,
#: `app/autonomy.py:4631`). The gate's skip carries the identical value, so one query
#: over `runs.meta_json` finds every refusal of a bad pin whichever side spoke it —
#: which is only safe because both sides ask ONE predicate, the function below.
_MODEL_REFUSAL_CLASS = "unconfigured_model"


def _gate_task_view(task_id) -> dict | None:
    """The task dict the gate asks the predicate about: the file, else nothing.

    Read from the task file rather than the due dict, because `get_due_tasks` returns
    the fields a scheduler tick needs to decide DUE-NESS (`id`, `name`, `priority`, the
    window) and not the `model:` pin — so a check written against the due dict reads a
    key that is usually absent, passes every test that stubs that call, and does nothing
    in production. That is the difference between a gate and a test fixture, and it is
    why `execute` has always re-parsed the file itself before deciding anything.

    `None` on an unreadable file, which the predicate answers as "cannot adjudicate"
    rather than as a refusal: an unreadable task file is not this gate's business.

    Measured at this round's base: `get_due_tasks`' own source contains no occurrence
    of the word `model`, so the due dict has never carried the pin and the per-task
    `_model_health_url(t.get("model"))` below has always been probing the primary
    whoever the task pinned. Changing that call is out of scope here (it would move
    every enqueue onto a second probe per task); what is not out of scope is refusing a
    pin this gate can see, and seeing it requires the file.
    """
    try:
        from app.autonomy import _find_task_file, _parse_task_file
        path = _find_task_file(task_id)
        if path:
            return _parse_task_file(path) or None
    except Exception:
        pass
    return None


def _model_pin_refusal(task_id, pin: object) -> str | None:
    """Skip text for this pin, `None` when the config can honour it.

    The verdict is the engine's, not a second copy of it: the text, the name list, the
    empty-pin case and the fail-open on an unreadable `models:` block all live in
    `app.autonomy._model_dispatch_refusal` (#1209, `app/autonomy.py:3145`) and are
    tested there. Re-implementing it here would be a second hand-maintained list of
    what config defines — the #3516 class of defect — and the two would drift the way
    the fleet-watch rosters did.

    What this function adds is the two rewrites `run_task` applies before it asks, so
    the gate's question is about the same string the engine would receive: no pin is the
    absence of a name (it resolves through `model.default`, and #1209 lets the empty
    string through on purpose), and `secondary` becomes `primary` while
    `secondary_enabled: false` — refusing the switched-off slot would be a behaviour
    change wearing a bug fix.

    Why ask at the gate when the engine already refuses: a refused `run_task` writes NO
    run record, so the only trace of the dispatch is the queue row, and a row that keeps
    re-failing is the whole witness. Refusing at enqueue means the task never enters the
    queue and the WARNING naming the pin is spoken in the tick that could have run it;
    refusing in `execute` means the item never spends 90 s waiting on a server it did not
    pin. Two surfaces, one predicate, one refusal class.
    """
    pin = str(pin or "").strip()
    if not pin or pin.lower() in ("null", "none"):
        try:
            from app.config import CONFIG
            pin = str(((CONFIG.get("model") or {}).get("default")) or "").strip()
        except Exception:
            return None
    if not pin:
        return None
    try:
        from app.config import resolve_model_alias
        pin = resolve_model_alias(pin)
    except Exception:
        pass
    try:
        from app.autonomy import _model_dispatch_refusal
        return _model_dispatch_refusal(task_id, pin)
    except Exception:
        return None


def _model_health_url(model: str) -> str:
    """Health endpoint for the model a task actually runs on.

    Tasks can pin `model: secondary` in their frontmatter (autonomy.run_task
    resolves it through that model's config entry). Probing only the primary would
    let a dead secondary turn every one of its tasks into a ConnectError flood — the
    exact failure the primary gate prevents.

    Every leg is config-derived: an empty/`primary` pin resolves through
    `_primary_health_target`, a named pin through `_model_endpoint` — its own
    `base_url`, falling back to its `env.ANTHROPIC_BASE_URL` (#3517) — and only the
    `except` leg, a config this process could not read at all, lands on the primary's
    derived endpoint rather than a literal (#1684). A named pin that config does not
    define is a different verdict, and `_model_pin_refusal` is where it is taken; this
    function still answers with the primary's endpoint for it, because the refusal
    runs first at both call sites and a health URL is not a way to say "skip".
    """
    model = str(model or "").strip()
    try:
        from app.config import resolve_model_alias
        model = resolve_model_alias(model)
    except Exception:
        pass
    if not model or model in ("primary", "null", "none"):
        return _primary_health_target()[0]
    try:
        endpoint = _model_endpoint(model)
        if endpoint:
            return endpoint
    except Exception:
        pass
    return _primary_health_target()[0]
# How long the model server can stay unreachable at /health before the outage
# itself is an alert (not one `logger.warning` line deduped for the whole
# outage). The gate in `enqueue_if_due` pauses dispatch, and when this constant
# was written that one line was all an outage produced:
# `agent-services/guardian/policy.py` watches only `lloyd-backend`/`lloyd-mcp`,
# never the model server, and the vault tasks that do probe it are autonomy
# tasks dispatched through this gate, so they die with it (#938).
#
# That is no longer the whole picture. `workers/service_probe.py` (#1359) is the
# CO-WATCHER of this same port: it announces `LLM Primary is not serving: :8096
# closed` once its grace elapses, from the pool's scheduler loop, outside this
# gate. The two do not measure the same thing — the probe watches the PORT under
# supervisord, this gate watches /health — so the probe cannot retire this
# threshold: a wedged vLLM that keeps the socket open, or an unreachable
# supervisord, produces no probe streak at all. What one outage costs is said
# once and measured once: the probe's grace is asserted STRICTLY BELOW this
# threshold (tests/test_service_probe.py::
# test_the_primary_grace_always_precedes_the_dispatch_gate_alert) so the in-room
# line always precedes this one, and `_note_vllm_outage` reads the elapsed
# minutes off the probe's streak rather than counting its own ticks (#1683).
# Neither surface writes a guardian ledger row for an engine outage — the probe
# announces (journal and toast, no bookkeeping) and this path is `logger.error`
# plus discord — and that is a SETTLED ruling (#1683, ruled by owed-check
# 2026-09-28), not a gap to close by accident here. The reason it stays that way:
# the only route to a guardian ledger row is `notify.alert`, which also rewrites
# ALERT.md with `write_text` (last-writer-wins) and can file a backlog task, so an
# engine outage would clobber a live rollback's record to buy a telemetry line.
# Both watchers therefore stay announce-shaped by design.
#
# 45 min is long enough to sit past a vLLM restart (the n-gram table alone takes
# minutes to load) and short enough that a person hears about it within an hour.
_VLLM_DOWN_ALERT_SECONDS = 45 * 60
# Outage accounting only, since #1682 moved the surveillance streaks to
# `workers/fleet_watchdog._state`. Kept here because the two keys below are written
# only by `_note_vllm_outage` and the gate's recovery branch, both dispatch-side.
_state = {"vllm_down_logged": False,
          # Outage accounting: `_vllm_down_since` is the instant the current
          # uninterrupted run of unhealthy ticks began, and `_vllm_down_alerted`
          # says whether THIS outage has already raised its alert. Both reset on
          # the first healthy tick, so the next outage gets one alert of its own
          # rather than inheriting the last one's silence.
          "vllm_down_since": None, "vllm_down_alerted": False}


def _vllm_healthy(timeout: float = 4.0, url: str | None = None) -> bool:
    # Even the no-URL call site derives from config: leaving it to fall through to
    # `_VLLM_HEALTH_URL` would keep a reachable path that probes the literal, which
    # is the exact divergence #1684 is about.
    target = url or _primary_health_target()[0]
    try:
        with urllib.request.urlopen(target, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False

async def _alert(message: str) -> None:
    try:
        from app.discord_notify import discord_alert
        await discord_alert(message)
    except Exception as e:
        logger.error("alert dispatch failed: %s", e)

def _vllm_outage_sentence(outage, own_seconds: float, now,
                          url: str | None = None,
                          url_source: str | None = None) -> str:
    """The one sentence a sustained primary outage produces (#1683).

    Which instrument measured it is stated in the sentence, because the two
    numbers are not the same quantity: `outage.down_seconds` is how long :8096
    has been closed while supervisord holds the program (the co-watcher's
    measurement, the one that also produced the in-room announcement this
    quotes), and `own_seconds` is how long this scheduler has failed to get a
    200 from /health. The second is only ever used when the first does not
    exist, and saying so is what keeps a reader from adding them.

    `url`/`url_source` are the endpoint the gate actually probed and the config
    key it came from (#1684). Before that the sentence interpolated the module's
    own literal, so a port move produced an alert pointing at a port the primary
    had never served while this same gate held the fleet paused; naming the key
    is what makes the outage self-diagnosing. Both default to a fresh
    derivation, so no caller can print an endpoint it did not resolve.
    """
    import datetime as _dt

    if url is None:
        url, url_source = _primary_health_target()
    url_source = url_source or _VLLM_URL_FALLBACK_KEY

    if outage is None:
        measured = own_seconds
        where = ("measured on this scheduler's own run of unhealthy /health "
                 "ticks; the service probe (workers/service_probe.py) holds no "
                 "streak for the primary, so it has announced nothing to quote "
                 "— supervisord unreachable, or the port never closed")
    else:
        measured = outage.down_seconds
        quoted = (f", which already announced \"{outage.announcement}\""
                  if outage.announced else ", which has not reached its grace")
        where = (f"measured on the service probe's :{outage.port} closed-port "
                 f"streak (workers/service_probe.py){quoted}")
    started = now - _dt.timedelta(seconds=measured)
    return (f"primary model server {url} has been unreachable for "
            f"{measured / 60:.0f} min (since {started.isoformat()}, {where}, "
            f"that endpoint is config-derived: {url_source}): "
            f"autonomy dispatch is PAUSED until it recovers, so every stall "
            f"alert from here on is describing a paused fleet, not a broken "
            f"one. Neither that line nor this one writes a guardian ledger row "
            f"for an engine outage (#1683).")


async def _note_vllm_outage() -> None:
    """Account for one unhealthy tick: log the transition, and alert once if the
    outage has run past `_VLLM_DOWN_ALERT_SECONDS`.

    Called on every unhealthy tick and on nothing else. The elapsed time it
    alerts on is measured ONCE, on `workers/service_probe.py`'s closed-port
    streak for the primary when it holds one: the same streak that announced the
    outage in the journal at the probe's grace, so the person on discord is
    reading the probe's number and the alert can quote the sentence they already
    saw instead of arriving as a second, unrelated incident. This module's own
    uninterrupted run of unhealthy ticks is the fallback, and it is load-bearing
    rather than legacy — the probe holds no streak when supervisord is
    unreachable (a skipped tick says nothing about the programs) or when vLLM
    keeps :8096 open while /health stops answering. A healthy tick clears
    `_vllm_down_since`, which is what makes a later outage alert again instead of
    inheriting this one's `vllm_down_alerted`.

    The alert is exactly one per outage and the log line stays transition-only:
    at `tick_interval: 60` an alert-per-tick would be 1440 a day, and the single
    deduped `logger.warning` this replaced was 0 a day where a person reads it —
    `logger` output goes to the unit log, `_alert` goes to discord (#938). The
    co-watcher of this port is `workers/service_probe.py` (#1359), whose grace
    is asserted strictly below this threshold so its line always comes first;
    `agent-services/guardian/policy.py` still watches only
    `lloyd-backend`/`lloyd-mcp`. NEITHER surface writes a guardian ledger row
    for an engine outage — the probe announces (journal and toast, no
    bookkeeping by design) and this path is a log line plus discord — and that is
    a SETTLED ruling (#1683, owed-check 2026-09-28), not one this function is
    waiting on. The reason: the only route to a guardian ledger row is
    `notify.alert`, which also rewrites ALERT.md with `write_text`
    (last-writer-wins) and can file a backlog task, so an engine outage would
    clobber a live rollback's record to buy a telemetry line."""
    import datetime as _dt
    from workers import service_probe

    now = _dt.datetime.now(_dt.timezone.utc)
    if not _state["vllm_down_logged"]:
        logger.warning("scheduled-task: vLLM unhealthy — pausing enqueue until it recovers")
        _state["vllm_down_logged"] = True
        _state["vllm_down_since"] = now
        _state["vllm_down_alerted"] = False
    if _state.get("vllm_down_alerted"):
        return
    own_since = _state.get("vllm_down_since") or now
    own_seconds = (now - own_since).total_seconds()
    outage = service_probe.shared().outage(service_probe.PRIMARY_ENGINE)
    measured = own_seconds if outage is None else outage.down_seconds
    if measured >= _VLLM_DOWN_ALERT_SECONDS:
        _state["vllm_down_alerted"] = True
        # Re-derived here rather than carried from the probe: same config, so the
        # alert names the very endpoint this gate has been failing to reach, and
        # the key it is configured by (#1684).
        url, url_source = _primary_health_target()
        msg = _vllm_outage_sentence(outage, own_seconds, now, url, url_source)
        logger.error("%s", msg)
        await _alert(msg)


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    from app import autonomy
    from app.autonomy import get_due_tasks

    loop = asyncio.get_event_loop()

    # Reset tasks stuck in_progress past their timeout (e.g. worker died
    # mid-run). Cheap relative to the get_due_tasks parse below. (`autonomy` is
    # already bound above, where the scan reads its clock.)
    recovered = await loop.run_in_executor(None, autonomy.recover_stuck_tasks)
    if recovered:
        logger.warning("Recovered %d stuck task(s): %s", len(recovered), recovered)

    # vLLM health gate — pause DISPATCH when the model server is down. It yields
    # a verdict here and acts on it below, after both stall detectors have run:
    # they read only task files and the queue, so neither needs the model server,
    # exactly like the startup scan and `recover_stuck_tasks` above. Returning at
    # this line is what made a multi-day outage invisible — it silenced dispatch
    # AND the detectors together, and left one deduped log line behind (#938).
    # The detectors' own exclusions are untouched, so an outage does not turn
    # every paused or queue-waiting task into noise: only a task that dispatch
    # would have enqueued and did not is flagged.
    # The endpoint is resolved HERE, not left to `_vllm_healthy`'s fallback, so the
    # probe the whole fleet's verdict rests on is the same config-derived URL the
    # outage alert will name — `models.<default>.base_url`, not a literal copied
    # into this module (#1684: a port move under `models:` used to pause every
    # dispatch on a URL the primary never served).
    primary_url, _ = _primary_health_target()
    vllm_ok = await loop.run_in_executor(None, _vllm_healthy, 4.0, primary_url)

    # The gate pauses dispatch here. Since #1682 the stall detectors that used
    # to run above this line run on the pool's own seat before this pass even
    # starts, so the #938 rule — an outage stops work, it does not stop
    # watching — holds by construction rather than by ordering. Returning before
    # `get_due_tasks` preserves the gate's own purpose: a fleet pinned to a dead
    # model server is skipped, not enqueued into a ConnectError retry loop (see
    # the per-task `_model_health_url` check below, which is the same rule one
    # level down).
    if not vllm_ok:
        await _note_vllm_outage()
        return
    if _state["vllm_down_logged"]:
        logger.info("scheduled-task: vLLM healthy again — resuming enqueue")
        _state["vllm_down_logged"] = False
        _state["vllm_down_since"] = None
        _state["vllm_down_alerted"] = False

    due = await loop.run_in_executor(None, get_due_tasks)

    healthy_urls: dict[str, bool] = {}
    for task in due:
        task_id = task.get("id")
        if task_id is None:
            continue
        # #3517: decided before the probe, because a probe is the one thing that
        # cannot answer for an undeclared pin — it would be taken against the
        # primary (that is what `_model_health_url` still returns for such a pin) and
        # read as a green light. Per task rather than a gate-wide return: every step
        # above this loop is fleet-wide, and the fleet does not stop enqueuing
        # because one task file has a typo in its `model:`.
        refusal = _model_pin_refusal(
            task_id, (_gate_task_view(task_id) or {}).get("model"))
        if refusal:
            logger.warning("Skipping #%s (%s): %s",
                           task_id, task.get("name"), refusal)
            continue
        # A task pinned to a model whose server is down should be skipped, not
        # enqueued into a retry loop.
        url = _model_health_url(task.get("model"))
        if url not in healthy_urls:
            healthy_urls[url] = await loop.run_in_executor(None, _vllm_healthy, 4.0, url)
        if not healthy_urls[url]:
            logger.warning("Skipping #%s (%s): model server %s is unhealthy",
                           task_id, task.get("name"), url)
            continue
        dedup_key = f"scheduled-task:{task_id}"
        prio_str = str(task.get("priority", "medium")).lower()
        priority = _PRIORITY_MAP.get(prio_str, DEFAULT_PRIORITY)
        new_id = queue.enqueue(
            source=NAME,
            kind="run",
            payload={"task_id": task_id, "name": task.get("name")},
            priority=priority,
            dedup_key=dedup_key,
        )
        if new_id is not None:
            logger.info("Enqueued scheduled-task #%s (%s) id=%d prio=%d",
                        task_id, task.get("name"), new_id, priority)

def _artifact_on_disk(artifact_path: str) -> bool:
    """True iff `artifact_path` is a non-empty file under the RESOLVED data root.

    `artifact_path` is DATA_ROOT-relative (`autonomy-runs/<task_id>/<run_id>.md`),
    so it has to be resolved against `app.paths.DATA_ROOT` and never against the
    process cwd — checking it with `Path(p).exists()` from `~/lloyd` reports 96 of
    97 real artifacts missing, which is a false alarm worse than the bug: it makes
    the check get disabled. The data root resolves from `$LLOYD_DATA` first and a
    worktree's `.lloyd-data/` third (`app/paths.py:63-83`), so "which root" is a
    real question and this answers it the same way the writer's `AUTONOMY_RUNS_DIR`
    was derived.

    Existence only, and only non-emptiness: the writer side already verifies the
    bytes (`app.autonomy._run_record_readable`); this is the reader's second
    opinion, from the other process, one hop before the row is written.
    """
    from app.paths import DATA_ROOT
    try:
        path = DATA_ROOT / artifact_path
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


async def execute(item: QueueItem) -> dict[str, Any]:
    from app.autonomy import run_task, run_trigger
    from app.discord_notify import _discord_notify_task_complete
    from app.autonomy import _find_task_file, _parse_task_file

    task_id = item.payload.get("task_id")
    if task_id is None:
        raise RuntimeError("scheduled-task item has no task_id in payload")

    # The enqueue-side health gate can't help items already in the queue when
    # vLLM wedges — without this wait they burn all attempts in ~90s of
    # ConnectErrors. Poll briefly for recovery before spending an attempt.
    loop = asyncio.get_event_loop()
    from app.autonomy import _find_task_file as _ftf, _parse_task_file as _ptf
    _t = _ptf(_ftf(task_id)) if _ftf(task_id) else None
    # Same predicate as the enqueue gate, and it has to be here too: an item can sit
    # in the queue across the edit that broke its pin, and the wait below would then
    # spend up to 90 s on a server this task never pinned and raise a deferral naming
    # that server's health URL — which the pool reads as a failure, charges it, and
    # re-queues until the attempts run out. The answer is the shape `run_task` itself
    # gives for this same verdict (#1209): `status: skipped`, which `normalize_result`
    # records without a failure and the pool releases without burning an attempt, and
    # whose text names the pin and the configured names instead of a machine (#3517).
    # Same verdict, arrived at without the wait.
    refusal = _model_pin_refusal(task_id, (_t or {}).get("model"))
    if refusal:
        logger.warning("deferring #%s (%s) without running: %s",
                       task_id, item.payload.get("name"), refusal)
        return {"status": "skipped", "skipped": refusal, "summary": refusal,
                "task_id": task_id, "refusal": _MODEL_REFUSAL_CLASS,
                "meta": _MODEL_REFUSAL_CLASS}
    health_url = _model_health_url((_t or {}).get("model"))
    for _ in range(18):  # up to 90s
        if await loop.run_in_executor(None, _vllm_healthy, 4.0, health_url):
            break
        await asyncio.sleep(5)
    else:
        raise RuntimeError(f"model server {health_url} unhealthy — deferring task")

    # Pass the pool's cap so run_task can keep its own timeout strictly under it
    # and always win the race (otherwise the pool cancels it and no run record
    # is written at all).
    try:
        from workers.sources import get_sources_config
        max_dur = int(get_sources_config().get(NAME, {}).get("max_duration_seconds", 1800))
    except Exception:
        max_dur = 1800
    with run_trigger("scheduler"):
        result = await run_task(int(task_id), max_duration=max_dur)
    # From here the task file has this attempt's verdict and the retry budget
    # moved with it. Anything still raising below — the silent-run check, the
    # artifact probe, the notify post — is a death of this adapter, not of the
    # task, and must be classified without charging a second unit (#2037).
    from workers.pool import mark_task_verdict
    mark_task_verdict()

    preview = (result.get("response_preview") or "")
    # #1507: the run record's verdict (its terminal block, exact match), not a
    # substring of the preview — a report that mentions the token still posts.
    from app.silent_sentinel import run_is_silent
    silent = run_is_silent(result.get("meta"), preview)
    if result.get("success") and preview and not silent:
        try:
            path = _find_task_file(task_id)
            if path:
                task = _parse_task_file(path)
                if task and task.get("notify_on_complete", True):
                    await _discord_notify_task_complete(
                        task_id, task.get("name", "Autonomy Task"), preview
                    )
        except Exception as e:
            logger.warning("Discord notify error for task #%s: %s", task_id, e)

    # Report task-level failures IN-BAND rather than raising. Raising sent the
    # item back through the queue's retry path, so a single timeout became up to
    # max_attempts full re-runs (3 x 600s on #36) before the scheduler's own
    # cooldown was ever consulted. run_task has already written the run record,
    # bumped failure_count and set the cooldown.
    status = result.get("status") or ("success" if result.get("success") else "failed")
    artifact_path = (f"autonomy-runs/{task_id}/{result.get('run_id')}.md"
                     if result.get("run_id") else "")
    out = {
        "status": status,
        "summary": (result.get("response_preview") or result.get("error") or "")[:500],
        "task_id": str(task_id),
        # The path is recorded UNCHANGED whether or not the file is on disk: it is a
        # path, and `pool.normalize_result` copies it verbatim into `runs.artifact_path`,
        # which other readers join on. The verdict goes beside it in `meta`, so "a run
        # record was written" and "there is a run record to read" stop being the same
        # claim — every one of the 97 scheduled-task `artifact_path` values in
        # `workers.db` was built from `run_id` alone without looking at disk (#1567).
        "artifact_path": artifact_path,
        "response": result.get("response_preview") or "",
        "meta": result.get("meta") or {},
    }
    if artifact_path and not _artifact_on_disk(artifact_path):
        from app.paths import DATA_ROOT
        logger.error("Task #%s: run record %s is not a non-empty file under the resolved "
                     "data root %s — the queue row records artifact_absent instead of "
                     "presenting it as readable", task_id, artifact_path, DATA_ROOT)
        out["meta"] = {**out["meta"], "artifact_absent": True}
    # #945 — this dict is a hand-written whitelist, and `claims` was not on it,
    # which severed the pilot's evidence bundle one hop upstream of the verifier:
    # `autonomy.run_task` sets `result["claims"]` for `EVIDENCE_PILOT_TASK_IDS`,
    # `pool.normalize_result` carries the key through, and `pool` verifies and
    # writes `claims_json` when it is present — but every autonomy run reached
    # the pool with no `claims` at all, so 4,922 scheduled-task rows carry an
    # empty bundle and `autonomy_health` has reported `runs_with_bundle: 0`
    # since the pilot shipped.
    #
    # Presence is the copy, not `out["claims"] = result.get("claims")`: the key
    # *being there* is the pool's pilot-scope switch (`normalize_result` maps an
    # absent key to `claims is None` → no bundle, and an empty list to a bundle
    # recording "the model emitted nothing" as a visible gap). Copying the key
    # unconditionally would not change behaviour today, but it would make the
    # adapter — not the model's output — decide which runs get scoped in.
    if "claims" in result:
        out["claims"] = result["claims"]
    # #623: the acceptance grade is lifted into `meta` by name — `meta` is the
    # only field `normalize_result` writes to the runs row for it
    # (`runs.meta_json`), and naming the key here means a run_task that stops
    # nesting it in `meta` still cannot lose it on this whitelist, which is
    # exactly how `claims` was lost above.
    if "acceptance_grade" in result:
        out["meta"] = {**out["meta"], "acceptance_grade": result["acceptance_grade"]}
    # #2619 — the script alarm, lifted by name for the same reason and by the same
    # mechanism as `acceptance_grade` above. It is the only thing that separates an
    # alarm night from a clean one on this row: a monitoring script's exit code
    # dies inside the agent turn, so three consecutive knowledge-health alarms
    # (2026-10-10..12, each report line 291 reading ABOVE FLOOR) were recorded
    # `status: success` with nothing in the row to
    # say otherwise. The producer sends the key on every run and the lift copies
    # it only when non-empty, so the row carries the field exactly on alarm runs —
    # absence keeps meaning "no script reported a nonzero exit".
    if result.get("script_alarm"):
        out["meta"] = {**out["meta"], "script_alarm": result["script_alarm"]}
    return out
