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

The right model can still be served wrongly. `expect_kv_cache_dtype` and
`expect_kv_pool_tokens_min` check what the engine says its KV cache is
(`vllm:cache_config_info` on /metrics): the 2026-09-10 FP8 cutover — a
692,263-token pool where BF16 held 398,175 — is what fixed the 09-09 stall,
and a boot that lands back on BF16 answers every request correctly while
undoing it. `agent-services/bin/flash-next-bootfacts.sh` asserts the same two
things against the boot log.
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


async def probe_kv_config(base_url: str, *, timeout: float = PROBE_TIMEOUT_SECONDS
                          ) -> dict[str, Any] | None:
    """The engine's own account of its KV cache — dtype, pool, page — or None.

    Never raises, for the same reason `probe_served_model` does not.
    """
    from app.vllm_metrics import cache_config_from_text

    try:
        async with httpx.AsyncClient(timeout=timeout) as cli:
            resp = await cli.get(f"{base_url.rstrip('/')}/metrics")
        if resp.status_code >= 400:
            return None
        return cache_config_from_text(resp.text)
    except Exception as exc:
        logger.debug("model_identity: /metrics on %s failed: %s", base_url, exc)
        return None


def judge_kv(cfg: dict[str, Any], kv: dict[str, Any] | None,
             reachable: bool) -> tuple[str, str]:
    """(status, detail) for a slot's KV cache against its config.

    ok / REGRESSION / unreachable / unknown (answers, but publishes no
    cache_config_info) / unchecked (nothing declared).
    """
    expect_dtype = str(cfg.get("expect_kv_cache_dtype") or "").strip().lower()
    expect_min = int(cfg.get("expect_kv_pool_tokens_min") or 0)
    if not expect_dtype and expect_min <= 0:
        return "unchecked", ""
    if kv is None:
        if not reachable:
            return "unreachable", "engine did not answer"
        return "unknown", "engine publishes no vllm:cache_config_info"
    dtype = str(kv.get("cache_dtype") or "").lower()
    pool = kv.get("kv_cache_size_tokens")
    pool_s = f"{pool:,}" if isinstance(pool, int) else "?"
    problems: list[str] = []
    if expect_dtype and not dtype.startswith(expect_dtype):
        problems.append(f"cache_dtype={dtype or '?'} (expected {expect_dtype})")
    if expect_min > 0 and (not isinstance(pool, int) or pool < expect_min):
        problems.append(f"pool={pool_s} tokens (expected >= {expect_min:,})")
    if problems:
        return "REGRESSION", "; ".join(problems)
    return "ok", f"cache_dtype={dtype} pool={pool_s} tokens"


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
            "kv": None,
            "kv_status": "unchecked",
            "kv_detail": "",
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
        if cfg.get("expect_kv_cache_dtype") or cfg.get("expect_kv_pool_tokens_min"):
            row["kv"] = await probe_kv_config(base_url)
            row["kv_status"], row["kv_detail"] = judge_kv(cfg, row["kv"], bool(served))
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
        if not any(r["status"] == "unreachable" or r.get("kv_status") == "unreachable"
                   for r in rows):
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
        if row.get("kv_status") == "REGRESSION":
            logger.error(
                "KV cache REGRESSION: alias %r at %s serves %s — the FP8 cutover "
                "(2026-09-10) is what fixed the 09-09 stall; check KV_CACHE_DTYPE "
                "and VLLM_VENV in agent-services/supervisor/conf.d/"
                "agent-llm-primary.conf, then flash-next-bootfacts.sh",
                row["alias"], row["base_url"], row["kv_detail"],
            )
        elif row.get("kv_status") == "unknown":
            logger.warning("model KV: alias %r — %s", row["alias"], row["kv_detail"])
        elif row.get("kv_status") == "ok":
            logger.info("model KV ok: alias %r %s", row["alias"], row["kv_detail"])
    return rows
