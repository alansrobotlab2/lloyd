"""Config loading and model resolution helpers.

Loads `config.yaml` once at import. `CONFIG` and `MODEL_CONFIGS` are
mutable dicts — the `/api/tool-toggle` route mutates `CONFIG` in place
then dumps it back to disk, so importers should access `CONFIG` via
attribute lookup on this module rather than copying references.

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

import yaml

from app.paths import LLOYD_HOME


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


def _load_config() -> dict:
    _load_env_file(LLOYD_HOME / ".env")
    config_path = LLOYD_HOME / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _expand(raw)


CONFIG = _load_config()
MODEL_CONFIGS = CONFIG.get("models", {})

def _get_model_cfg(model_name: str) -> dict:
    """Resolve the full model config dict for a model name or alias."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    for _name, c in MODEL_CONFIGS.items():
        if c.get("alias") == model_name:
            return c
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


def resolve_model_alias(name: str) -> str:
    """Route 'secondary' → 'primary' when secondary_enabled is false.

    Single switch for primary-only deployments. Callers that previously
    hardcoded 'secondary' (or its port) should pass through this helper
    before resolving base_url / model name.
    """
    if name == "secondary" and not CONFIG.get("secondary_enabled", False):
        return "primary"
    return name
