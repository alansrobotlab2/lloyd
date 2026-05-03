"""Config loading and model resolution helpers.

Loads `config.yaml` once at import. `CONFIG` and `MODEL_CONFIGS` are
mutable dicts — the `/api/tool-toggle` route mutates `CONFIG` in place
then dumps it back to disk, so importers should access `CONFIG` via
attribute lookup on this module rather than copying references.
"""

import yaml

from app.paths import LLOYD_HOME


def _load_config() -> dict:
    config_path = LLOYD_HOME / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


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
