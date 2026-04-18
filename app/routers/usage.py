"""Usage/metrics endpoints + live Anthropic rate-limit ping.

The ping endpoint refreshes the Claude Code OAuth token on 401 and caches
the rate-limit headers for 30s so the frontend can poll freely.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import usage_store
from app.config import CONFIG, MODEL_CONFIGS

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore


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
    """Return usage for both allocation windows (4h and 7d) plus allocations."""
    four_h = usage_store.summary(hours=4)
    seven_d = usage_store.summary(days=7)
    alloc = {
        "4h": {"tokens": 0, "cost_usd": 0},
        "7d": {"tokens": 0, "cost_usd": 0},
    }
    return JSONResponse({
        "four_hour": four_h,
        "seven_day": seven_d,
        "allocations": alloc,
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
    """Per-model breakdown (Anthropic models only)."""
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


# Cached rate-limit data from last Anthropic ping
_rate_limit_cache: dict = {}
_rate_limit_cache_ts: float = 0.0


def _get_anthropic_api_key() -> str | None:
    """Read OAuth token from Claude Code credentials."""
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return None
    try:
        creds = json.loads(creds_path.read_text())
        oauth = creds.get("claudeAiOauth", {})
        expires_at = oauth.get("expiresAt", 0)
        if expires_at and time.time() * 1000 > expires_at - 60_000:  # 1min buffer
            refreshed = _refresh_oauth_token(creds_path, creds)
            if refreshed:
                return refreshed
        return oauth.get("accessToken")
    except Exception:
        return None


_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"


def _refresh_oauth_token(creds_path: Path, creds: dict) -> str | None:
    """Use the refresh token to get a new access token, update credentials file."""
    import urllib.request
    oauth = creds.get("claudeAiOauth", {})
    refresh_token = oauth.get("refreshToken")
    if not refresh_token:
        return None
    try:
        payload = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _OAUTH_CLIENT_ID,
        }).encode()
        req = urllib.request.Request(
            _OAUTH_TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        new_access = data.get("access_token")
        new_refresh = data.get("refresh_token")
        new_expires = data.get("expires_in")
        if not new_access:
            return None
        oauth["accessToken"] = new_access
        if new_refresh:
            oauth["refreshToken"] = new_refresh
        if new_expires:
            oauth["expiresAt"] = int((time.time() + new_expires) * 1000)
        creds["claudeAiOauth"] = oauth
        creds_path.write_text(json.dumps(creds))
        logger.info("Refreshed OAuth access token successfully")
        return new_access
    except Exception as e:
        logger.warning(f"OAuth token refresh failed: {e}")
        return None


@router.get("/api/usage/ping")
async def usage_ping():
    """Ping Anthropic with a minimal haiku call to get real rate-limit utilization.

    Returns the unified rate-limit headers (5h/7d utilization, status, resets)
    plus our locally tracked stats. Caches for 30 seconds to avoid excessive calls.
    """
    global _rate_limit_cache, _rate_limit_cache_ts

    if time.time() - _rate_limit_cache_ts < 30 and _rate_limit_cache:
        return JSONResponse(_rate_limit_cache)

    if _anthropic_sdk is None:
        return JSONResponse({"error": "anthropic SDK not installed"}, status_code=501)

    api_key = _get_anthropic_api_key()
    if not api_key:
        return JSONResponse({"error": "No Anthropic credentials found"}, status_code=401)

    async def _do_ping(key: str):
        client = _anthropic_sdk.Anthropic(
            api_key=key,
            base_url="https://api.anthropic.com",
        )
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.messages.with_raw_response.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            ),
        )

    try:
        try:
            resp = await _do_ping(api_key)
        except _anthropic_sdk.AuthenticationError:
            logger.info("Ping got 401, attempting OAuth token refresh...")
            creds_path = Path.home() / ".claude" / ".credentials.json"
            creds = json.loads(creds_path.read_text())
            new_key = _refresh_oauth_token(creds_path, creds)
            if not new_key:
                return JSONResponse({"error": "OAuth token expired and refresh failed"}, status_code=401)
            api_key = new_key
            resp = await _do_ping(api_key)

        rl = {}
        for k, v in resp.headers.items():
            if "ratelimit" in k.lower():
                key = k.replace("anthropic-ratelimit-unified-", "")
                try:
                    rl[key] = float(v) if "." in v else int(v)
                except ValueError:
                    rl[key] = v

        msg = resp.parse()
        usage_store.record_usage(
            session_id="__ping__",
            model="claude-3-haiku-20240307",
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            cost_usd=None,
            duration_ms=None,
            duration_api_ms=None,
            num_turns=1,
        )

        local_models = [
            name for name, cfg in MODEL_CONFIGS.items() if cfg.get("base_url")
        ]
        local_5h = usage_store.summary(hours=5, exclude_models=local_models)
        local_7d = usage_store.summary(days=7, exclude_models=local_models)

        result = {
            "rate_limits": rl,
            "local_5h": local_5h,
            "local_7d": local_7d,
            "pinged_at": datetime.utcnow().isoformat(),
        }

        _rate_limit_cache = result
        _rate_limit_cache_ts = time.time()

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Usage ping failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
