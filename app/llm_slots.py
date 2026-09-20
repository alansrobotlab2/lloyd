"""The optional LLM slots, and whether each one is switched on.

Two engines are optional and share GPU 2's RTX 3090: the Qwen3.6-35B-A3B
secondary on :8091 and djev (DiffusionGemma NVFP4) on :8010/:8011. Each has a
switch in config.yaml, and *six* places need to agree about what that switch
means:

    server.py::_sync_llm_slots     starts or stops the supervisord program
    vllm_metrics.configured_engines   whether the engine gets a dashboard card
    routers/services.py            the Services tab, and its start/stop actions
    routers/dashboard.py           the dashboard's services panel
    routers/mc_ui.py               the service counts the AGENT reads, twice

They had no shared definition, and the failure that produces is not a
disagreement — it is unanimous silence. A slot switched off in config.yaml
stayed in every hardcoded service table, so it was listed as a service, counted
as `stopped` in the agent's own view of the machine, and named in
`unhealthy_services` on every poll, forever. Nothing was wrong; the whole
system just kept reporting that something was.

So the question "is this program supposed to be running?" is answered here and
nowhere else.

A program that is not an optional slot is always enabled. That default is what
keeps this module from becoming a second service registry: it says nothing
about `agent-llm-primary` or `lloyd-backend`, which have no switch and must
never acquire one by being absent from a table.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, TypeVar

# supervisord program -> (the flag as it is written in config.yaml, reader).
#
# The reader takes the loaded config rather than reaching for it, so a caller
# that already holds one (and the tests) can ask about a config that is not the
# live one. The flag string is carried for error messages and logs: "disabled"
# is not actionable, "`djev.enabled` is false in config.yaml" is.
_SLOTS: dict[str, tuple[str, Callable[[dict], bool]]] = {
    "agent-llm-secondary": (
        "secondary_enabled",
        lambda c: bool(c.get("secondary_enabled", False)),
    ),
    "agent-djev": (
        "djev.enabled",
        lambda c: bool((c.get("djev") or {}).get("enabled", False)),
    ),
}


def _config() -> dict:
    # Imported late: app.config pulls in the whole config surface and this
    # module is imported from routers that are themselves imported at boot.
    from app.config import CONFIG

    return CONFIG


def is_slot(program: str) -> bool:
    """Is this supervisord program an optional LLM slot at all?"""
    return program in _SLOTS


def slot_flag(program: str) -> str | None:
    """The config.yaml key that governs `program`, for messages and logs."""
    entry = _SLOTS.get(program)
    return entry[0] if entry else None


def is_enabled(program: str, config: dict | None = None) -> bool:
    """Should `program` be running, according to config.yaml?

    True for anything that is not an optional slot — see the module docstring.
    """
    entry = _SLOTS.get(program)
    if entry is None:
        return True
    return entry[1](_config() if config is None else config)


def slots(config: dict | None = None) -> list[tuple[str, str, bool]]:
    """`(flag, program, enabled)` for every optional slot, in a stable order."""
    cfg = _config() if config is None else config
    return [(flag, program, reader(cfg))
            for program, (flag, reader) in _SLOTS.items()]


def enabled_slots(config: dict | None = None) -> list[tuple[str, str]]:
    """`(flag, program)` for the slots that are switched on."""
    return [(flag, program) for flag, program, on in slots(config) if on]


_V = TypeVar("_V")


def visible(table: dict[str, _V], config: dict | None = None) -> dict[str, _V]:
    """`table` without the entries for slots that are switched off.

    The service registries are module-level constants, so this filters at READ
    time rather than at import time: `data/tool_overrides.yaml` and a plain
    edit-and-restart both change the answer, and a table filtered once at
    import would keep serving whatever was true when the process booted.
    """
    cfg = _config() if config is None else config
    return {k: v for k, v in table.items() if is_enabled(k, cfg)}


def visible_ids(ids: Iterable[str], config: dict | None = None) -> list[str]:
    """The same filter for a bare sequence of program names."""
    cfg = _config() if config is None else config
    return [i for i in ids if is_enabled(i, cfg)]


def describe(program: str, config: dict | None = None) -> dict[str, Any]:
    """Why a program is not visible, for an error a human has to act on."""
    return {
        "program": program,
        "is_slot": is_slot(program),
        "flag": slot_flag(program),
        "enabled": is_enabled(program, config),
    }
