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
