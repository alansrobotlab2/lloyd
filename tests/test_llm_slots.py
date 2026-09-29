"""A slot switched off in config.yaml is invisible on every surface.

The bug this pins is not a disagreement between surfaces — it is unanimity in
the wrong direction. Before `app/llm_slots.py`, six places each read a
hardcoded service table, so an engine stopped on purpose was listed as a
service, counted as `stopped` in the agent's own view of the machine, and named
in `unhealthy_services` on every poll, with no action that could ever clear it.
"""

import asyncio
import json
import pathlib

import pytest

from app import llm_slots
from app.config import CONFIG


REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def slots(monkeypatch):
    """Drive both switches without touching the live config."""
    def _set(secondary: bool, djev: bool):
        monkeypatch.setitem(CONFIG, "secondary_enabled", secondary)
        monkeypatch.setitem(CONFIG, "djev", {**(CONFIG.get("djev") or {}),
                                             "enabled": djev,
                                             "base_url": "http://127.0.0.1:8010"})
    return _set


# ── the definition itself ──────────────────────────────────────────────

def test_a_program_with_no_switch_is_always_enabled():
    """The default that keeps this module from becoming a second service
    registry. `agent-llm-primary` and `lloyd-backend` have no flag and must
    never acquire one by being absent from the table."""
    for program in ("agent-llm-primary", "lloyd-backend", "agent-qmd-daemon",
                    "something-invented-next-year"):
        assert llm_slots.is_enabled(program) is True
        assert llm_slots.is_slot(program) is False


def test_each_slot_names_the_flag_a_human_has_to_edit():
    """"disabled" is not actionable; the config key is."""
    assert llm_slots.slot_flag("agent-llm-secondary") == "secondary_enabled"
    assert llm_slots.slot_flag("agent-djev") == "djev.enabled"
    assert llm_slots.slot_flag("agent-llm-primary") is None


def test_the_reader_takes_a_config_so_it_can_be_asked_about_one_that_is_not_live():
    off = {"secondary_enabled": False, "djev": {"enabled": True}}
    assert llm_slots.is_enabled("agent-llm-secondary", off) is False
    assert llm_slots.is_enabled("agent-djev", off) is True
    assert llm_slots.enabled_slots(off) == [("djev.enabled", "agent-djev")]


# ── every surface follows it ───────────────────────────────────────────

def _surfaces():
    """What each surface currently shows, by supervisord program / alias."""
    from app.supervisor_client import all_services, infra_services
    from app.vllm_metrics import configured_engines
    return {
        "services_tab": set(infra_services()),
        "audited": set(all_services()),
        "engine_cards": set(configured_engines()),
    }


@pytest.mark.parametrize("secondary,djev", [(True, False), (False, True), (False, False)])
def test_a_disabled_slot_is_absent_from_every_surface(slots, secondary, djev):
    slots(secondary, djev)
    seen = _surfaces()

    for on, program, alias in ((secondary, "agent-llm-secondary", "secondary"),
                               (djev, "agent-djev", "djev")):
        for surface in ("services_tab", "audited"):
            assert (program in seen[surface]) is on, (
                f"{program} should {'appear in' if on else 'be absent from'} "
                f"{surface}; got {sorted(seen[surface])}")
        assert (alias in seen["engine_cards"]) is on, (
            f"{alias} engine card; got {sorted(seen['engine_cards'])}")

    # The unconditional services are never filtered by any of this.
    assert "agent-llm-primary" in seen["services_tab"]
    assert "lloyd-backend" in seen["audited"]


def test_the_services_tab_and_the_dashboard_panel_show_the_same_slots(slots):
    """Two endpoints, one question. They were separate hardcoded reads."""
    from app.routers.dashboard import _services
    from app.routers.services import get_services

    slots(secondary=False, djev=True)

    tab = json.loads(asyncio.run(get_services()).body)["services"]
    panel = _services()["services"] if isinstance(_services(), dict) else _services()
    panel_ids = {row["id"] for row in panel}
    tab_ids = {row["id"] for row in tab}

    assert "agent-llm-secondary" not in tab_ids
    assert "agent-llm-secondary" not in panel_ids
    assert "agent-djev" in tab_ids
    assert "agent-djev" in panel_ids


def test_a_disabled_slot_is_not_reported_unhealthy_to_the_agent(slots):
    """The surface that mattered most: `unhealthy_services` goes into the
    agent's context. A stopped-on-purpose engine listed there is a permanent
    fault it can neither explain nor fix."""
    from app.routers import mc_ui

    slots(secondary=False, djev=False)
    summary = mc_ui._summarize_dashboard()
    unhealthy = summary.get("unhealthy_services")
    if unhealthy is not None:                       # absent if supervisord is unreachable
        assert "agent-llm-secondary" not in unhealthy
        assert "agent-djev" not in unhealthy

    counts = (mc_ui._summarize_services() or {}).get("counts")
    if counts:
        from app.supervisor_client import all_services
        assert sum(counts.values()) == len(all_services())


# ── acting on one ──────────────────────────────────────────────────────

class _Req:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def test_starting_a_disabled_slot_names_the_flag_rather_than_calling_it_unknown(
        slots, monkeypatch):
    """404 "Unknown service" sends someone hunting for a typo. It is not
    unknown, it is switched off — and for the two GPU 2 slots, starting it from
    the UI would fight the other one for the card and be undone by the next
    boot reconcile anyway.

    The stubs below are not decoration. `service_action` calls the real
    `start_process` against the real supervisord socket, and the first version
    of this test relied on the 409 firing first to stay safe — so the moment it
    was run against a deliberately broken filter (proving it could fail, which
    is the only reason to write it), it STARTED THE LIVE SECONDARY on a box
    where djev already held GPU 2. It OOM'd on a 15.5 GiB cudaMalloc and
    supervisord parked it FATAL. A test that reaches production when the code
    under test is wrong is exactly backwards: the isolation has to come from
    the test, not from the assertion it is trying to make.
    """
    from fastapi import HTTPException
    from app.routers import services as services_router

    called = []
    for name in ("start_process", "stop_process", "restart_process"):
        monkeypatch.setattr(services_router, name,
                            lambda sid, _n=name: (called.append((_n, sid)), (True, "stub"))[1])
    service_action = services_router.service_action

    slots(secondary=False, djev=True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service_action(_Req({"serviceId": "agent-llm-secondary",
                                         "action": "start"})))
    assert exc.value.status_code == 409
    assert "secondary_enabled" in exc.value.detail

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service_action(_Req({"serviceId": "nope", "action": "start"})))
    assert exc.value.status_code == 404

    # Neither refusal may have reached supervisord, and the stubs prove it
    # rather than the absence of a side effect being assumed.
    assert called == [], f"service_action dispatched on a refused id: {called}"

    # The enabled slot still dispatches — the guard refuses the off one, not all.
    asyncio.run(service_action(_Req({"serviceId": "agent-djev", "action": "restart"})))
    assert called == [("restart_process", "agent-djev")]


# ── and there are no other readers ─────────────────────────────────────

def test_only_supervisor_client_reads_the_raw_registries():
    """A seventh reader is how this comes back. The registries include rows
    that must not be shown, so anything reading them directly re-creates the
    bug; `infra_services()` / `all_services()` are the way in."""
    offenders = []
    for path in sorted((REPO / "app").rglob("*.py")):
        if path.name == "supervisor_client.py":
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        if "_INFRA_SERVICES" in body or "_LLOYD_SERVICES" in body:
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        "these read the unfiltered service registries directly and will list a "
        f"switched-off slot: {offenders}. Use supervisor_client.infra_services() "
        "/ lloyd_services() / all_services() instead.")


# ── the two model rows' ports come from config, not from a second literal ──
#
# #1684 made the dispatch gate derive its probe URL from `models.<alias>.base_url`.
# This registry was the other owner that ruling named: it still carried 8096 / 8091
# as literals, while `workers/service_probe.py` ticks the outage streak off
# `infra_services()` and `workers/sources/scheduled_task.py` re-derives the URL it
# probes two lines later. A port move therefore left the gate answering on the new
# port and the probe holding a streak on the old one that can never end.

from app.config import MODEL_CONFIGS


def test_one_config_value_moves_the_registry_row_and_the_dispatch_gate_together(
        monkeypatch):
    """Clauses 1: the surface that counts an outage and the surface that pauses
    dispatch on it cannot name different ports for the same config value.

    Asserted as ONE test over BOTH surfaces on purpose. Two tests, each checking its
    own half against its own expectation, stay green while the halves disagree — and
    the disagreement is the whole bug: the alert said ':8096 closed' for 90 minutes
    about a port the engine had stopped serving a week earlier, beside a URL the
    engine does serve.
    """
    from app import supervisor_client as sc
    from workers.sources.scheduled_task import _primary_health_target

    monkeypatch.setitem(MODEL_CONFIGS["primary"], "base_url", "http://127.0.0.1:9999")

    row = sc.infra_services()["agent-llm-primary"]
    url, url_source = _primary_health_target()

    assert row[1] == 9999, f"registry row still probes the literal: {row}"
    assert url.endswith(":9999/health"), f"gate moved without the registry: {url}"
    assert f":{row[1]}" in url, (
        f"probe port {row[1]} and gate port disagree — the outage streak would be "
        f"unbounded: row={row} url={url}")
    assert sc.model_port_source("agent-llm-primary") == "models.primary.base_url"


def test_the_secondary_row_follows_its_own_alias_and_never_the_default(monkeypatch):
    """Clause 2: the secondary's port is `models.secondary.base_url`, read directly.

    The row has to be visible for this to say anything, so the slot is switched on the
    way the fixture at the top of this file does it — `config.yaml` ships
    `secondary_enabled: false`, and a test that asked about a hidden row would pass on
    its absence rather than on its port. The primary keeps its own port in the same
    call, which is what rules out the two readings that would otherwise look right:
    `models.<model.default>` for both rows, or `resolve_model_alias`, either of which
    puts the primary's port in the secondary's row.
    """
    from app import supervisor_client as sc

    monkeypatch.setitem(CONFIG, "secondary_enabled", True)
    monkeypatch.setitem(MODEL_CONFIGS["secondary"], "base_url", "http://127.0.0.1:9998")

    services = sc.infra_services()

    assert "agent-llm-secondary" in services, "fixture guard: the row must be visible"
    assert services["agent-llm-secondary"][1] == 9998, services["agent-llm-secondary"]
    assert services["agent-llm-secondary"][1] != 8091, "still the literal"
    assert services["agent-llm-primary"][1] == 8096, (
        "the primary's row moved with the secondary's config")
    assert services["agent-llm-secondary"][1] != services["agent-llm-primary"][1]
    assert sc.model_port_source("agent-llm-secondary") == "models.secondary.base_url"


def test_the_fallback_answers_8096_and_8091_and_says_it_was_the_fallback(monkeypatch,
                                                                        caplog):
    """Clause 3: config silent is not no answer — and it is not a config answer.

    Both shapes of silence are exercised: the lookup raising outright, and the alias
    resolving with no endpoint behind it (both `base_url` and `env` cleared), which is
    the shape a real config produces when someone moves a model to a remote endpoint
    and the registry is left to answer for a port nobody named.
    """
    import logging

    from app import supervisor_client as sc
    from app import config as app_config

    monkeypatch.setitem(CONFIG, "secondary_enabled", True)
    monkeypatch.setitem(MODEL_CONFIGS["primary"], "base_url", "")
    monkeypatch.setitem(MODEL_CONFIGS["primary"], "env", {})
    monkeypatch.setitem(MODEL_CONFIGS["secondary"], "base_url", "")
    monkeypatch.setitem(MODEL_CONFIGS["secondary"], "env", {})

    with caplog.at_level(logging.WARNING):
        services = sc.infra_services()

    assert services["agent-llm-primary"] == ("LLM Primary", 8096), services
    assert services["agent-llm-secondary"][1] == 8091, services
    for sid in ("agent-llm-primary", "agent-llm-secondary"):
        assert sc.model_port_source(sid) == sc.MODEL_PORT_FALLBACK_SOURCE, (
            f"{sid} printed a port with no way to tell it was not config's")
    assert sc.MODEL_PORT_FALLBACK_SOURCE != "models.primary.base_url"

    # And the raising shape, same contract.
    def boom(alias):
        raise RuntimeError("config unreadable")
    monkeypatch.setattr(app_config, "_get_model_cfg", boom)
    raised = sc.infra_services()
    assert raised["agent-llm-primary"][1] == 8096
    assert raised["agent-llm-secondary"][1] == 8091
    assert sc.model_port_source("agent-llm-primary") == sc.MODEL_PORT_FALLBACK_SOURCE

    # Five callers unpack the value as a 2-tuple — mc_ui.py, services.py twice,
    # dashboard.py and service_probe.py — so provenance travels BESIDE the value, never
    # inside it. Checked on both branches, because the branch that tempts a third slot
    # is the derived one: on the fallback path the code is still the old literal return.
    monkeypatch.setattr(app_config, "_get_model_cfg",
                        lambda alias: {"base_url": "http://127.0.0.1:9999"})
    for shape in (services, sc.infra_services()):
        for sid, value in shape.items():
            assert isinstance(value, tuple) and len(value) == 2, f"{sid}: {value!r}"
            display, port = value                   # the unpack those five sites do
            assert display and (port is None or isinstance(port, int)), f"{sid}: {value!r}"


def test_only_the_model_rows_are_derived_and_every_other_literal_stands(monkeypatch):
    """Clause 4: this is a two-row change, not a general derivation.

    djev, qmd and livekit keep their literals — their ports are not model endpoints,
    and whether the same derivation should reach them is a ruling this item explicitly
    scopes out. With the primary patched to an odd port, every other row must still
    print its own number, or the fix would have moved services nobody asked about and
    started probing them somewhere they do not live.
    """
    from app import supervisor_client as sc

    monkeypatch.setitem(MODEL_CONFIGS["primary"], "base_url", "http://127.0.0.1:9999")
    monkeypatch.setitem(CONFIG, "secondary_enabled", True)

    infra = sc.infra_services()
    lloyd = sc.lloyd_services()

    assert infra["agent-djev"] == ("djev (DiffusionGemma)", 8011), infra["agent-djev"]
    assert infra["agent-qmd-daemon"][1] == 8181
    assert infra["agent-livekit-server"][1] == 7880
    assert infra["agent-qmd-watcher"][1] is None
    assert infra["agent-tts"][1] is None
    assert infra["lloyd-agent-worker"][1] is None
    assert lloyd["lloyd-backend"] == ("Lloyd Backend", 8080)
    assert lloyd["lloyd-frontend"] == ("Lloyd Frontend", 5173)
    assert lloyd["lloyd-mcp"] == ("Lloyd MCP", 8500)
    for sid in ("agent-djev", "agent-qmd-daemon", "agent-livekit-server"):
        assert sc.model_port_source(sid) is None, (
            f"{sid} went through the model derivation")
