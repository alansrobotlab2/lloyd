"""Config loading and model resolution helpers.

Loads `config.yaml` once at import, then merges `data/tool_overrides.yaml`
on top. config.yaml is read-at-boot and hand-edited / git-tracked; the
overrides file holds the UI-mutable state (server enabled, disabled_tools,
tool_search settings) and is the ONLY file the toggle routes write — a UI
click can no longer rewrite (and de-comment) the whole config, nor race a
hand-edit. `CONFIG` and `MODEL_CONFIGS` are mutable dicts; importers should
access `CONFIG` via attribute lookup on this module rather than copying
references.

The repo's `.env` (if present) is loaded into `os.environ` first, and
any `${VAR}` placeholders in config.yaml are then expanded against the
combined environment. Secrets (LiveKit API keys, Discord token, etc.)
live in `.env` — gitignored — and are referenced by name from the
committed `config.yaml`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import logging

import yaml

from app.atomic_io import atomic_write_text
from app.paths import LLOYD_HOME

logger = logging.getLogger("lloyd-config")

TOOL_OVERRIDES_PATH = LLOYD_HOME / "data" / "tool_overrides.yaml"


_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _load_env_file(path: Path) -> None:
    """Parse a simple `.env` file (KEY=VALUE per line) into os.environ.

    Only sets variables that aren't already in the environment, so values
    set by supervisord / shell still win over the file. Lines beginning
    with '#' are comments; blank lines are ignored.
    """
    if not path.exists():
        return
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Strip optional surrounding quotes
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            os.environ.setdefault(k, v)
    except OSError:
        pass


def _expand(value: Any) -> Any:
    """Recursively expand `${VAR}` placeholders in any string within a config tree.

    Unknown vars expand to empty string (matches shell behaviour). Other
    types pass through unchanged.
    """
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _merge_tool_overrides(config: dict) -> dict:
    """Overlay data/tool_overrides.yaml onto the boot config.

    Only the UI-mutable keys are honored: per-server `enabled` /
    `disabled_tools` and the `harness.tool_search` block. Anything else in
    the overrides file is ignored, so a stray write can't shadow hand-edited
    config. Missing or unparseable file → config unchanged.
    """
    if not TOOL_OVERRIDES_PATH.exists():
        return config
    try:
        overrides = yaml.safe_load(TOOL_OVERRIDES_PATH.read_text()) or {}
    except Exception as e:
        logger.warning("tool_overrides.yaml unreadable, ignoring: %s", e)
        return config
    for server, o in (overrides.get("mcp_servers") or {}).items():
        if server not in (config.get("mcp_servers") or {}):
            continue  # overrides can't introduce servers, only adjust them
        cfg = config["mcp_servers"][server]
        if "enabled" in o:
            cfg["enabled"] = bool(o["enabled"])
        if "disabled_tools" in o:
            cfg["disabled_tools"] = list(o["disabled_tools"] or [])
    ts = (overrides.get("harness") or {}).get("tool_search")
    if isinstance(ts, dict):
        config.setdefault("harness", {}).setdefault("tool_search", {}).update(ts)
    return config


def save_tool_overrides() -> None:
    """Persist the UI-mutable slice of CONFIG to data/tool_overrides.yaml.

    Replaces the old behavior of yaml.dump-ing the entire CONFIG back over
    config.yaml on every toggle.
    """
    out: dict = {"mcp_servers": {}}
    for name, cfg in (CONFIG.get("mcp_servers") or {}).items():
        entry: dict = {"disabled_tools": list(cfg.get("disabled_tools") or [])}
        if "enabled" in cfg:
            entry["enabled"] = bool(cfg["enabled"])
        out["mcp_servers"][name] = entry
    ts = (CONFIG.get("harness") or {}).get("tool_search")
    if isinstance(ts, dict):
        out["harness"] = {"tool_search": ts}
    TOOL_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        TOOL_OVERRIDES_PATH,
        yaml.dump(out, default_flow_style=False, allow_unicode=True, sort_keys=False),
    )


def _load_config() -> dict:
    _load_env_file(LLOYD_HOME / ".env")
    config_path = LLOYD_HOME / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _merge_tool_overrides(_expand(raw))


CONFIG = _load_config()
MODEL_CONFIGS = CONFIG.get("models", {})

def _get_model_cfg(model_name: str) -> dict:
    """Resolve the full model config dict for a model name or alias."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    for _name, c in MODEL_CONFIGS.items():
        if c.get("alias") == model_name:
            return c
    # An unknown name silently resolving to {} means the caller proceeds
    # with no env overrides — i.e. against the wrong endpoint. Make the
    # misroute greppable.
    logger.warning("unknown model %r — no config/env overrides applied", model_name)
    return {}


def _get_model_env(model_name: str) -> dict:
    """Return environment variable overrides for a model."""
    cfg = _get_model_cfg(model_name)
    return dict(cfg.get("env", {}))


def _model_base_url(model_name: str) -> str:
    """Return the ANTHROPIC_BASE_URL for a model, or '' for real Anthropic models."""
    cfg: dict = MODEL_CONFIGS.get(model_name, {})
    if not cfg:
        for name, c in MODEL_CONFIGS.items():
            if c.get("alias") == model_name:
                cfg = c
                break
    return cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")


def _resolve_model_name(model_input: str) -> str:
    """Resolve alias to full model name."""
    for name, cfg in MODEL_CONFIGS.items():
        if cfg.get("alias") == model_input:
            return name
    return model_input


def service_url(name: str, default: str = "") -> str:
    """Resolve an internal service endpoint from config.yaml `services:`.

    One registry for non-model service URLs (backend, lloyd-mcp aggregator,
    QMD daemon) so swapping a port is a config edit, not a multi-file grep.
    Model endpoints stay under `models:` — that remains their single source
    of truth.
    """
    services = CONFIG.get("services") or {}
    url = services.get(name)
    return str(url) if url else default


def default_model_base_url() -> str:
    """Base URL of the default model — the fallback when a caller has no
    explicit ANTHROPIC_BASE_URL in its resolved model env."""
    default = (CONFIG.get("model") or {}).get("default", "primary")
    cfg = _get_model_cfg(default)
    return (
        cfg.get("base_url")
        or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
        or "http://127.0.0.1:8096"
    )


def resolve_model_alias(name: str) -> str:
    """Route 'secondary' → 'primary' when secondary_enabled is false.

    Single switch for primary-only deployments. Callers that previously
    hardcoded 'secondary' (or its port) should pass through this helper
    before resolving base_url / model name.
    """
    if name == "secondary" and not CONFIG.get("secondary_enabled", False):
        return "primary"
    return name
