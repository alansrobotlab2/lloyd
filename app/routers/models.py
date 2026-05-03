"""Model list + per-session model switch endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import CONFIG, MODEL_CONFIGS, _resolve_model_name
from app.sessions_io import mutate_session


router = APIRouter()


@router.get("/api/models")
async def get_models():
    _ordered = ["primary", "secondary"]  # always first in dropdown
    models = []
    for name, cfg in MODEL_CONFIGS.items():
        models.append({
            "name": name,
            "alias": cfg.get("alias", ""),
            "display_name": cfg.get("display_name", name),
            "provider": "local",
            "base_url": cfg.get("base_url", ""),
            "context_length": cfg.get("context_length", 0),
        })
    # Sort so primary and secondary always appear first
    def _sort_key(m):
        priority = 0 if m["name"] in _ordered else 1
        return (priority, m["name"])
    models.sort(key=_sort_key)
    return JSONResponse({
        "models": models,
        "default": CONFIG.get("model", {}).get("default", ""),
    })


@router.post("/api/model/switch")
async def switch_model(request: Request):
    data = await request.json()
    model = data.get("model", "")
    session_id = data.get("session_id", "")
    resolved = _resolve_model_name(model)
    if session_id:
        await mutate_session(session_id, lambda d: d.__setitem__("model", resolved))
    return JSONResponse({"success": True, "model": resolved})
