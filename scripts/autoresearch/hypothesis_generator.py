"""Hypothesis generator — proposes variant overlays for an autoresearch round.

Reads current prompt surfaces (SOUL/MEMORY/USER) + recent signal (bench losers,
correction log, knowledge-health) and asks the local model to return a JSON
list of candidate variants. Each variant is a full-file replacement for one or
more prompt surfaces — the sandbox then materializes it and the bench runner
evaluates it under LLOYD_OVERLAY_DIR.

For v1, only the `prompts` target (SOUL.md + MEMORY.md) is generated. Other
targets are advertised in config and reserved for later iterations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

from .common import AUTORESEARCH_PRIORITY, LLOYD_HOME, AutoresearchConfig, now_iso, variant_id

logger = logging.getLogger("autoresearch.hypothesis")

SOUL_PATH = LLOYD_HOME.parent / "obsidian" / "lloyd" / "SOUL.md"
MEMORY_PATH = LLOYD_HOME.parent / "obsidian" / "lloyd" / "MEMORY.md"
USER_PATH = LLOYD_HOME.parent / "obsidian" / "lloyd" / "USER.md"
CORRECTIONS_PATH = LLOYD_HOME.parent / "obsidian" / "memory" / "corrections.md"
KNOWLEDGE_HEALTH_PATH = LLOYD_HOME / "_pipeline" / "reports" / "knowledge-health-latest.md"


def _read(path: Path, tail: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if tail and len(text) > tail:
        return text[-tail:]
    return text


def _recent_ledger_losers(ledger_path: Path, limit: int = 10) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    losers: list[dict[str, Any]] = []
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()[-2000:]
    except Exception:
        return []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("promoted") is False and entry.get("composite_score") is not None:
            losers.append(entry)
            if len(losers) >= limit:
                break
    return losers


def _call_local_llm(
    prompt: str,
    model: str = "primary",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    timeout: int = 300,
    json_mode: bool = True,
) -> str | None:
    """Synchronous call to local vLLM via OpenAI-compatible API.

    Hits vLLM directly (8096/8091) and includes `"priority": 2` in the body so
    vLLM's scheduler preempts this work for both interactive (priority=0) and
    pipeline/autonomy (priority=1) traffic. When json_mode=True, requests
    response_format={'type':'json_object'} so vLLM's constrained decoding
    forces well-formed output.
    """
    base = "http://127.0.0.1:8096" if model == "primary" else "http://127.0.0.1:8091"
    model_name = model if model in ("primary", "secondary") else model
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "priority": AUTORESEARCH_PRIORITY,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        resp = requests.post(
            f"{base}/v1/chat/completions",
            headers={"Authorization": "Bearer no-key-required"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as exc:
        logger.error("local LLM call failed: %s", exc)
        return None


def _build_prompt(
    cfg: AutoresearchConfig,
    targets: list[str],
    max_variants: int,
) -> str:
    soul = _read(SOUL_PATH)
    memory = _read(MEMORY_PATH, tail=6000)
    user = _read(USER_PATH, tail=4000)
    corrections = _read(CORRECTIONS_PATH, tail=3000)
    knowledge = _read(KNOWLEDGE_HEALTH_PATH, tail=3000)

    ledger_losers = _recent_ledger_losers(cfg.paths.ledger_path, limit=8)
    losers_summary = "\n".join(
        f"- target={e.get('target_surface')}, score={e.get('composite_score'):.2f}, "
        f"hypothesis={e.get('hypothesis', '')[:120]}"
        for e in ledger_losers
    ) or "(no recent losers)"

    return f"""You are a prompt-engineering researcher proposing improvements to an AI
agent named Lloyd. Propose up to {max_variants} candidate VARIANTS of Lloyd's
prompt surfaces that might score higher on Lloyd's benchmark.

## Supported targets this round
{', '.join(targets)}

## Current SOUL.md (identity, always loaded)
{soul}

## Current MEMORY.md (long-term notes)
{memory}

## Current USER.md
{user}

## Recent correction log (tail)
{corrections or '(none)'}

## Recent knowledge-health snapshot
{knowledge or '(none)'}

## Recent losing variants from the ledger
{losers_summary}

## Your task
Output a JSON object with a single key "variants" containing an array of up
to {max_variants} objects. Each object MUST have these fields:

- "description": one-line summary (<= 120 chars)
- "hypothesis": why you expect this variant to score better; be specific
  about which bench category (replay / synthetic / adversarial / safety) and
  which failure mode it addresses
- "target_surface": exactly one of: "prompts"  (only prompts is supported this
  round — the other targets will be implemented later)
- "overlay_files": object mapping relative path to the full new file content.
  For "prompts" you may include "SOUL.md" and/or "MEMORY.md". Include the
  FULL replacement content (not a diff). If you only want to change MEMORY.md,
  include only MEMORY.md. Every overlay file MUST be a complete, valid
  document that can stand on its own.

Do not weaken or remove any safety-critical guidance in SOUL.md (refusal
patterns, destructive-action confirmation). Do not invent new tools. Keep
edits surgical: prefer tightening an existing section over a rewrite.

Return ONLY the JSON object. No prose before or after.

/no_think"""


def _parse_variants(raw: str, max_variants: int) -> list[dict[str, Any]]:
    if not raw:
        return []
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.warning("hypothesis response did not contain JSON object")
        return []
    try:
        data = json.loads(match.group(0))
    except Exception as exc:
        logger.warning("hypothesis JSON parse failed: %s", exc)
        return []
    variants = data.get("variants", []) if isinstance(data, dict) else []
    clean: list[dict[str, Any]] = []
    for v in variants[:max_variants]:
        if not isinstance(v, dict):
            continue
        target = v.get("target_surface", "prompts")
        if target != "prompts":
            logger.info("skipping variant with unsupported target=%s", target)
            continue
        overlay = v.get("overlay_files") or {}
        if not isinstance(overlay, dict) or not overlay:
            continue
        # Only allow known relative paths for the prompts target
        allowed = {"SOUL.md", "MEMORY.md"}
        filtered_overlay = {k: str(val) for k, val in overlay.items() if k in allowed and isinstance(val, str) and val.strip()}
        if not filtered_overlay:
            continue
        clean.append({
            "variant_id": variant_id(),
            "description": str(v.get("description", ""))[:200],
            "hypothesis": str(v.get("hypothesis", ""))[:1000],
            "target_surface": "prompts",
            "overlay_files": filtered_overlay,
            "created_at": now_iso(),
        })
    return clean


def propose_variants(
    cfg: AutoresearchConfig,
    targets: list[str] | None = None,
    max_variants: int | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Ask the local model for up to N variant overlays targeting the given surfaces."""
    targets = targets or ["prompts"]
    max_variants = max_variants or cfg.max_variants_per_round
    model = model or cfg.default_model

    prompt = _build_prompt(cfg, targets, max_variants)
    logger.info("requesting %d variants from model=%s (targets=%s)", max_variants, model, targets)
    raw = _call_local_llm(prompt, model=model)
    variants = _parse_variants(raw or "", max_variants)
    logger.info("hypothesis generator returned %d valid variants", len(variants))
    return variants
