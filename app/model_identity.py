"""Verify that each model slot serves the model config claims it does.

Why this exists: a slot's identity lives in three places that can drift
apart — `models.<alias>` in config.yaml (the endpoint), the supervisord
program's `environment=MODEL=...` (which launcher branch runs), and the
process actually listening on the port. Nothing reconciled the third
against the first two. `server.py::_sync_llm_slots` only ever
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

A slot can also be served without the half its config promises.
`models.<alias>.supports_vision: true` routes screenshots to the slot as
images, and it is true only while the conf keeps `LANGUAGE_MODEL_ONLY=0`: the
launcher defaults to text-only, so an A/B arm, a rollback or a reverted conf
leaves the flag pointing at an engine that refuses every image. The engine
publishes no modality list (`/v1/models` and `vllm:cache_config_info` carry
none), so `probe_image_input` asks it — one tiny image, one token — and
classifies the refusal with the same test the harness's turn-time latch uses
(#1420). A slot whose flag is not literally `true` is never sent one.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
import zlib
from typing import Any

import httpx

logger = logging.getLogger("lloyd-server")

PROBE_TIMEOUT_SECONDS = 5.0

# The image probe is a real generation, so it can queue behind a live turn's
# prefill; 5 s would read a busy engine as a broken one.
IMAGE_PROBE_TIMEOUT_SECONDS = 30.0
# Behind every interactive and worker request (lower runs sooner): the probe
# answers a boot-time question and must never cost a turn its place.
IMAGE_PROBE_PRIORITY = 10


def _solid_png(side: int = 32) -> bytes:
    """A grey `side`x`side` PNG. Not 1x1: Qwen's processor resizes to a
    multiple of its patch size and a degenerate image can be refused for its
    size, which would read as a missing tower."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    raw = b"".join(b"\x00" + b"\x80" * side for _ in range(side))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PROBE_IMAGE_URL = "data:image/png;base64," + base64.b64encode(_solid_png()).decode()

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


def expects_image_input(cfg: dict[str, Any]) -> bool:
    """The slot's own claim, read the way the image router reads it
    (`tool_images.model_supports_vision`): literally `true`, nothing else."""
    return cfg.get("supports_vision") is True


async def probe_image_input(base_url: str, model: str, *,
                            timeout: float = IMAGE_PROBE_TIMEOUT_SECONDS) -> tuple[str, str]:
    """(status, detail) from one image-bearing request. Never raises.

    ok          — the engine generated from an image
    REFUSED     — a 400 the harness's latch would read as "no image input"
    unreachable — the request did not connect
    unknown     — any other answer (a timeout, a 5xx, a 400 about something
                  else); reported, never called a refusal
    """
    from app.harness.tool_images import looks_like_multimodal_rejection

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "."},
            {"type": "image_url", "image_url": {"url": PROBE_IMAGE_URL}},
        ]}],
        "max_tokens": 1,
        "temperature": 0,
        "priority": IMAGE_PROBE_PRIORITY,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=timeout) as cli:
            resp = await cli.post(url, json=payload)
    except httpx.ConnectError as exc:
        return "unreachable", f"image probe did not connect: {exc}"
    except Exception as exc:
        return "unknown", f"image probe failed: {type(exc).__name__}: {exc}"[:300]
    body = " ".join((resp.text or "").split())[:300]
    if resp.status_code < 300:
        return "ok", "engine accepted an image"
    if resp.status_code == 400 and looks_like_multimodal_rejection(body):
        return "REFUSED", body
    return "unknown", f"HTTP {resp.status_code}: {body}"


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

    `image_status` is the same kind of verdict for image input, expected
    exactly when `supports_vision` is literally true (`probe_image_input`);
    otherwise `unchecked`, and no image is sent.
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
            "supports_vision": expects_image_input(cfg),
            "image_status": "unchecked",
            "image_detail": "",
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
        if not row["supports_vision"]:
            row["image_detail"] = "supports_vision is not true; no image sent"
        elif not served:
            row["image_status"], row["image_detail"] = "unreachable", "engine did not answer"
        else:
            row["image_status"], row["image_detail"] = await probe_image_input(
                base_url, str(cfg.get("alias") or alias))
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
                   or r.get("image_status") == "unreachable" for r in rows):
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
        if row.get("image_status") == "REFUSED":
            logger.error(
                "image input REFUSED: alias %r at %s — models.%s.supports_vision is "
                "true but the engine refuses images (%s); the launcher strips the "
                "vision tower unless LANGUAGE_MODEL_ONLY=0 (with MM_IMAGES_PER_PROMPT "
                "> 0) in agent-services/supervisor/conf.d/agent-llm-primary.conf — "
                "restore it, or set supports_vision: false",
                row["alias"], row["base_url"], row["alias"], row["image_detail"],
            )
        elif row.get("image_status") == "unknown":
            logger.warning("model image input: alias %r — %s", row["alias"], row["image_detail"])
        elif row.get("image_status") == "ok":
            logger.info("model image input ok: alias %r", row["alias"])
    return rows
