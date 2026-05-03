"""Usage/metrics endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import usage_store
from app.config import CONFIG, MODEL_CONFIGS


router = APIRouter()
logger = logging.getLogger("lloyd-server")


@router.get("/api/usage/summary")
async def usage_summary(hours: float = 0, days: float = 0):
    """Aggregated usage totals for a time window. No args = all-time."""
    data = usage_store.summary(
        hours=hours if hours > 0 else None,
        days=days if days > 0 else None,
    )
    return JSONResponse(data)


@router.get("/api/usage/windows")
async def usage_windows():
    """Return usage for both allocation windows (4h and 7d)."""
    four_h = usage_store.summary(hours=4)
    seven_d = usage_store.summary(days=7)
    return JSONResponse({
        "four_hour": four_h,
        "seven_day": seven_d,
    })


def _local_models() -> list[str]:
    return [name for name, cfg in MODEL_CONFIGS.items() if cfg.get("base_url")]


@router.get("/api/usage/history")
async def usage_history(period: str = "4h"):
    """Time-series data for charts. period: 4h, 24h, 7d, 30d."""
    excl = _local_models()
    if period == "4h":
        data = usage_store.history_buckets(hours=4, bucket_minutes=15, exclude_models=excl)
    elif period == "24h":
        data = usage_store.history_buckets(hours=24, bucket_minutes=60, exclude_models=excl)
    elif period == "7d":
        data = usage_store.history_daily(days=7, exclude_models=excl)
    elif period == "30d":
        data = usage_store.history_daily(days=30, exclude_models=excl)
    else:
        data = usage_store.history_buckets(hours=4, bucket_minutes=15, exclude_models=excl)
    return JSONResponse({"period": period, "buckets": data})


@router.get("/api/usage/models")
async def usage_models(hours: float = 0, days: float = 0):
    """Per-model breakdown."""
    excl = _local_models()
    data = usage_store.model_breakdown(
        hours=hours if hours > 0 else None,
        days=days if days > 0 else None,
        exclude_models=excl,
    )
    return JSONResponse({"models": data})


@router.get("/api/usage/recent")
async def usage_recent(limit: int = 20):
    """Most recent usage records."""
    data = usage_store.recent_requests(limit=limit)
    return JSONResponse({"records": data})
