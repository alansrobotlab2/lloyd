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

_LOCAL_MODEL_VARS = [
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_CUSTOM_MODEL_OPTION",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME",
]


def _get_model_cfg(model_name: str) -> dict:
    """Resolve the full model config dict for a model name or alias."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    for _name, c in MODEL_CONFIGS.items():
        if c.get("alias") == model_name:
            return c
    return {}


def _get_model_env(model_name: str) -> dict:
    """Get environment variable overrides for a model.

    If the model doesn't set ANTHROPIC_BASE_URL (i.e. it's a real Anthropic
    model, not a local one), clear any inherited local-model vars so the
    subprocess doesn't accidentally hit the Qwen server.
    """
    cfg = _get_model_cfg(model_name)
    model_env = dict(cfg.get("env", {}))

    if "ANTHROPIC_BASE_URL" not in model_env:
        for var in _LOCAL_MODEL_VARS:
            if var not in model_env:
                model_env[var] = ""

    return model_env


def _resolve_effort(model_name: str, think_level: str | None = None) -> str:
    """Resolve effort level for a model, with optional /think override.

    Resolution order for base effort:
      per-model `effort` config > global `agent.effort` > 'medium'

    /think override behavior:
      - "off" → force "low" (minimal thinking)
      - "low"/"medium"/"high" on Anthropic models → pass through
      - Any non-off level on local models → "high" (local is binary on/off)
    """
    cfg = _get_model_cfg(model_name)
    effort = (
        cfg.get("effort")
        or CONFIG.get("agent", {}).get("effort", "medium")
    )
    if think_level == "off":
        return "low"
    if think_level and think_level != "off":
        is_local = bool(cfg.get("env", {}).get("ANTHROPIC_BASE_URL"))
        return "high" if is_local else think_level
    return effort


def _resolve_thinking(model_name: str) -> dict | None:
    """Return the SDK `thinking` config for a model.

    Opus 4.7 requires explicit `thinking: {type: "adaptive"}` to enable thinking
    at all — the default with no config is thinking OFF. Opus 4.6 and Sonnet 4.6
    also accept adaptive as the recommended mode. Local Qwen models (vLLM /
    llama-server behind an Anthropic-compatible gateway) also accept the adaptive
    thinking flag and return visible thinking content — unlike Opus 4.7 which
    defaults `display` to "omitted". Return adaptive for all models.
    """
    return {"type": "adaptive"}


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
