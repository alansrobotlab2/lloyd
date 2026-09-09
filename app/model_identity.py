"""Verify that each model slot serves the model config claims it does.

Why this exists: a slot's identity lives in three places that can drift
apart — `models.<alias>` in config.yaml (the endpoint), the supervisord
program's `environment=MODEL=...` (which launcher branch runs), and the
process actually listening on the port. Nothing reconciled the third
against the first two. `server.py::_sync_secondary_llm_state` only ever
decided whether the secondary should be *running*, never *what* it is.

On 2026-09-06 a `git reset` from the automod/guardian machinery reverted
`start-secondary.sh` and `agent-llm-secondary.conf` to a revision whose
`MODEL=qwen35` branch serves Qwen3.5-**4B** on vLLM. The 35B kept running
because the process was never restarted, but any restart in that window
would have answered to the alias `secondary` with a model an order of
magnitude smaller, and every caller — chat, autonomy, post-session
capture — would have gone on reporting `model: secondary` as if nothing
had changed. A rollback that silently downgrades a model is worse than
one that fails loudly.

Declare the expected model with `models.<alias>.expect_model` in
config.yaml: a case-insensitive substring matched against whatever the
engine reports. Slots without it are reported as `unchecked`.

The probe covers both runtimes:
  * vLLM  — `/v1/models` entries carry `root` (the checkpoint directory).
  * llama.cpp — `/v1/models` carries only the `--alias`, so the real
    identity comes from `/props` (`model_path`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger("lloyd-server")

PROBE_TIMEOUT_SECONDS = 5.0

# Result of the most recent verification sweep, keyed by alias. Read by
# `GET /api/models/identity`; refreshed by the boot sweep and on demand.
LAST_RESULT: dict[str, dict[str, Any]] = {}


async def probe_served_model(base_url: str, *, timeout: float = PROBE_TIMEOUT_SECONDS) -> str:
    """Best-effort identity string for whatever is serving `base_url`.

    Returns a descriptor built from every identifying field the engine
    offers, or "" when the endpoint is unreachable. Never raises: an
    engine being down is a normal state (the secondary is stopped
    whenever `secondary_enabled` is false), not an error.
    """
    base = base_url.rstrip("/")
    parts: list[str] = []
    async with httpx.AsyncClient(timeout=timeout) as cli:
        try:
            resp = await cli.get(f"{base}/v1/models")
            if resp.status_code < 400:
                for entry in (resp.json().get("data") or []):
                    for field in ("root", "id"):
                        val = entry.get(field)
                        if isinstance(val, str) and val:
                            parts.append(val)
        except Exception as exc:  # unreachable, malformed, wrong shape
            logger.debug("model_identity: /v1/models on %s failed: %s", base, exc)

        # llama.cpp reports only the --alias above, so the checkpoint path
        # is the only thing that actually names the model.
        try:
            resp = await cli.get(f"{base}/props")
            if resp.status_code < 400:
                body = resp.json()
                for field in ("model_path", "model_alias"):
                    val = body.get(field)
                    if isinstance(val, str) and val:
                        parts.append(val)
        except Exception:
            pass  # vLLM has no /props — expected, not worth a log line

    # Preserve order, drop duplicates.
    seen: set[str] = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return " ".join(uniq)


async def verify_models() -> list[dict[str, Any]]:
    """Check every configured slot against its `expect_model`.

    Each row is `{alias, base_url, expect, served, status}` where status
    is one of:
      ok         — the endpoint reports the expected model
      MISMATCH   — it reports something else (the drift this module exists for)
      unreachable— nothing answered (engine stopped, still loading)
      unchecked  — no `expect_model` declared for this alias
    """
    from app.config import MODEL_CONFIGS

    rows: list[dict[str, Any]] = []
    for alias, cfg in (MODEL_CONFIGS or {}).items():
        base_url = (
            cfg.get("base_url")
            or (cfg.get("env") or {}).get("ANTHROPIC_BASE_URL", "")
        )
        expect = (cfg.get("expect_model") or "").strip()
        row: dict[str, Any] = {
            "alias": alias,
            "base_url": base_url,
            "expect": expect,
            "served": "",
            "status": "unchecked",
        }
        if not base_url:
            rows.append(row)
            continue

        served = await probe_served_model(base_url)
        row["served"] = served
        if not expect:
            row["status"] = "unchecked"
        elif not served:
            row["status"] = "unreachable"
        elif expect.lower() in served.lower():
            row["status"] = "ok"
        else:
            row["status"] = "MISMATCH"
        rows.append(row)

    LAST_RESULT.clear()
    LAST_RESULT.update({r["alias"]: r for r in rows})
    return rows


async def verify_models_with_retry(
    *, attempts: int = 6, delay_seconds: float = 15.0
) -> list[dict[str, Any]]:
    """Boot-time sweep that tolerates an engine still loading weights.

    Retries only while something is `unreachable` — a MISMATCH is
    conclusive on the first look and waiting would just delay the alarm.
    The 35B takes minutes to page 17 GB onto a 3090, so a single probe at
    startup would report `unreachable` for a perfectly healthy slot.
    """
    rows: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        rows = await verify_models()
        if not any(r["status"] == "unreachable" for r in rows):
            break
        if attempt < attempts:
            await asyncio.sleep(delay_seconds)

    for row in rows:
        if row["status"] == "MISMATCH":
            logger.error(
                "model identity MISMATCH: alias %r at %s expects %r but serves %r "
                "— check the supervisord program's MODEL= env and its start script; "
                "a rollback may have reverted the launcher",
                row["alias"], row["base_url"], row["expect"], row["served"],
            )
        elif row["status"] == "unreachable":
            logger.warning(
                "model identity: alias %r at %s did not answer after %d attempts",
                row["alias"], row["base_url"], attempts,
            )
        elif row["status"] == "ok":
            logger.info(
                "model identity ok: alias %r serves %r", row["alias"], row["expect"]
            )
    return rows
