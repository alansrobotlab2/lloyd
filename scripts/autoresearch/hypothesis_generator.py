"""Hypothesis generator — proposes variant overlays for an autoresearch round.

Reads current prompt surfaces (SOUL/MEMORY/USER) + recent signal (bench losers,
baseline failures, correction log, knowledge-health) and asks the local model
to return candidate variants. Each variant is a full-file replacement for ONE
prompt surface — the sandbox then materializes it and the bench runner
evaluates it under LLOYD_OVERLAY_DIR.

For v1, only the `prompts` target (SOUL.md OR MEMORY.md, one per variant) is
generated. Other targets are advertised in config and reserved for later.

Design: N single-variant calls in parallel via ThreadPoolExecutor. Each call
returns a small, focused JSON object that reliably fits under max_tokens. On
parse failure the raw output is dumped to _pipeline/research/_debug/ for
post-mortem rather than silently lost.
"""

from __future__ import annotations

import concurrent.futures
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
DEBUG_DIR = LLOYD_HOME / "_pipeline" / "research" / "_debug"


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


def _recent_baseline_failures(ledger_path: Path, limit: int = 8) -> list[dict[str, Any]]:
    """Find recent BASELINE entries where composite_score < 0.5 or safety_passed=False.

    The hypothesis generator needs concrete failure signal. Bare 'lost variant'
    history assumes we've ever produced variants — on a cold start, we haven't,
    and the baseline's own per-task scores are the only real data.
    """
    if not ledger_path.exists():
        return []
    fails: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()[-3000:]
    except Exception:
        return []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        vid = str(entry.get("variant_id", ""))
        if not vid.startswith("BASELINE"):
            continue
        tid = entry.get("task_id") or ""
        if not tid or tid in seen_tasks:
            continue
        composite = entry.get("composite_score")
        safety_pass = entry.get("safety_passed")
        safety_crit = entry.get("safety_critical")
        is_fail = (
            (isinstance(composite, (int, float)) and composite < 0.5)
            or (safety_crit and safety_pass is False)
        )
        if not is_fail:
            continue
        seen_tasks.add(tid)
        fails.append(entry)
        if len(fails) >= limit:
            break
    return fails


def _dump_raw_on_failure(label: str, payload: dict[str, Any], raw: str, error: str) -> None:
    """Persist raw model output + request payload when JSON parsing fails.

    Without this, 'hypothesis generator returned 0 valid variants' is a
    diagnostic black hole. Files land in _pipeline/research/_debug/ with the
    timestamp + label so you can grep for `variant truncated` or `bad comma`
    across runs.
    """
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        ts = now_iso().replace(":", "-")
        path = DEBUG_DIR / f"hypothesis_fail_{ts}_{label}.txt"
        trimmed_payload = {k: v for k, v in payload.items() if k != "messages"}
        trimmed_payload["_messages_len"] = sum(len(m.get("content", "")) for m in payload.get("messages", []))
        content = (
            f"=== ERROR ===\n{error}\n\n"
            f"=== PAYLOAD (messages dropped) ===\n{json.dumps(trimmed_payload, indent=2)}\n\n"
            f"=== RAW OUTPUT ({len(raw)} chars) ===\n{raw}\n"
        )
        path.write_text(content, encoding="utf-8")
        logger.info("Dumped failed hypothesis output to %s", path)
    except Exception as exc:
        logger.warning("Failed to dump raw output: %s", exc)


def _try_parse_json(raw: str) -> tuple[dict | None, str | None]:
    """Parse JSON with a light repair pass. Returns (obj, error_message)."""
    if not raw:
        return None, "empty response"
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, "no JSON object found"
    candidate = match.group(0)
    # First attempt: straight parse
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        first_err = str(exc)
    # Repair pass: strip trailing commas before } or ], collapse control chars.
    repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
    # Escape bare control chars inside strings is too aggressive; just retry plain.
    try:
        return json.loads(repaired), None
    except json.JSONDecodeError as exc:
        return None, f"{first_err}; repair also failed: {exc}"


def _call_local_llm(
    prompt: str,
    model: str = "primary",
    temperature: float = 0.8,
    max_tokens: int = 8000,
    timeout: int = 300,
    json_mode: bool = True,
) -> tuple[str | None, dict[str, Any]]:
    """Synchronous call to local vLLM via OpenAI-compatible API.

    Returns (content_or_none, payload_used). The payload is returned so parse
    failures can dump the request alongside the bad response.

    max_tokens defaults to 8000 so a full MEMORY.md (~3.5K tok) rewrite wrapped
    in a JSON object fits without truncation. Temperature raised to 0.8 to get
    more variant diversity across parallel calls.
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
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish = data.get("choices", [{}])[0].get("finish_reason", "")
        if finish and finish != "stop":
            logger.warning("LLM finish_reason=%s (possible truncation)", finish)
        return content, payload
    except Exception as exc:
        logger.error("local LLM call failed: %s", exc)
        return None, payload


def _build_single_variant_prompt(
    cfg: AutoresearchConfig,
    targets: list[str],
    target_file_hint: str | None = None,
) -> str:
    """Prompt for ONE variant, targeting ONE file. Small JSON output → reliable parse."""
    soul = _read(SOUL_PATH)
    memory = _read(MEMORY_PATH, tail=4000)
    user = _read(USER_PATH, tail=2000)
    corrections = _read(CORRECTIONS_PATH, tail=2000)
    knowledge = _read(KNOWLEDGE_HEALTH_PATH, tail=2000)

    baseline_fails = _recent_baseline_failures(cfg.paths.ledger_path, limit=6)
    fails_summary = "\n".join(
        f"- task={e.get('task_id')} category={e.get('task_category')} "
        f"composite={e.get('composite_score'):.2f} "
        f"safety_crit={e.get('safety_critical')} safety_pass={e.get('safety_passed')}"
        for e in baseline_fails
    ) or "(no baseline failures logged yet — pick an area where Lloyd could be more robust)"

    ledger_losers = _recent_ledger_losers(cfg.paths.ledger_path, limit=4)
    losers_summary = "\n".join(
        f"- target={e.get('target_surface')}, score={e.get('composite_score'):.2f}, "
        f"hypothesis={e.get('hypothesis', '')[:100]}"
        for e in ledger_losers
    ) or "(no recent variant losers — this is a cold start)"

    # If caller hinted a specific file, tell the model to target that file.
    file_hint_block = ""
    if target_file_hint in ("SOUL.md", "MEMORY.md"):
        file_hint_block = (
            f"\n## File to modify for THIS variant\n"
            f"You MUST modify exactly: `{target_file_hint}`. Leave the other file(s) alone.\n"
        )

    return f"""You are a prompt-engineering researcher proposing ONE improvement to an AI
agent named Lloyd. Generate exactly ONE variant of Lloyd's prompt surfaces
that might score higher on Lloyd's benchmark.

## Supported targets this round
{', '.join(targets)}

## Current SOUL.md (identity, always loaded)
{soul}

## Current MEMORY.md tail (long-term notes)
{memory}

## Current USER.md tail
{user}

## Recent correction log (tail)
{corrections or '(none)'}

## Recent knowledge-health snapshot
{knowledge or '(none)'}

## Recent baseline bench FAILURES (what's currently broken)
{fails_summary}

## Recent losing variants (avoid repeating these approaches)
{losers_summary}
{file_hint_block}
## Your task
Output a JSON object with this EXACT shape:

{{
  "description": "<one-line summary, <=120 chars>",
  "hypothesis": "<why this will help; name a specific failure mode or bench category>",
  "target_surface": "prompts",
  "overlay_files": {{
    "<SOUL.md OR MEMORY.md>": "<full replacement content of that ONE file>"
  }}
}}

Rules:
- Modify exactly ONE file: either SOUL.md or MEMORY.md (NOT both).
- overlay_files must contain EXACTLY ONE key.
- The file content must be the COMPLETE replacement text, not a diff.
- Do not weaken safety language (refusal, destructive-action confirmation).
- Keep edits surgical — prefer tightening an existing section over a rewrite.
- Return ONLY the JSON object, no prose before or after, no markdown fences.

/no_think"""


def _parse_single_variant(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one variant JSON object. Returns (variant_or_none, error_message)."""
    data, err = _try_parse_json(raw)
    if data is None:
        return None, err or "unknown parse error"
    if not isinstance(data, dict):
        return None, f"expected object, got {type(data).__name__}"
    # Accept either a direct variant or a wrapped {"variants":[...]} for back-compat.
    if "variants" in data and isinstance(data["variants"], list) and data["variants"]:
        data = data["variants"][0]
        if not isinstance(data, dict):
            return None, "variants[0] is not an object"
    target = data.get("target_surface", "prompts")
    if target != "prompts":
        return None, f"unsupported target_surface={target}"
    overlay = data.get("overlay_files") or {}
    if not isinstance(overlay, dict) or not overlay:
        return None, "overlay_files missing/empty"
    allowed = {"SOUL.md", "MEMORY.md"}
    filtered = {k: str(val) for k, val in overlay.items() if k in allowed and isinstance(val, str) and val.strip()}
    if not filtered:
        return None, f"no valid overlay keys (got {list(overlay.keys())})"
    if len(filtered) > 1:
        # Keep the first — the prompt asks for one file only.
        logger.info("variant contained multiple overlay files; keeping first")
        k = next(iter(filtered))
        filtered = {k: filtered[k]}
    return {
        "variant_id": variant_id(),
        "description": str(data.get("description", ""))[:200],
        "hypothesis": str(data.get("hypothesis", ""))[:1000],
        "target_surface": "prompts",
        "overlay_files": filtered,
        "created_at": now_iso(),
    }, None


def _propose_one(
    cfg: AutoresearchConfig,
    targets: list[str],
    model: str,
    seed_idx: int,
) -> dict[str, Any] | None:
    """Run one hypothesis call. Returns a variant or None."""
    # Alternate hinted target file by index to diversify.
    hint = "SOUL.md" if seed_idx % 2 == 0 else "MEMORY.md"
    prompt = _build_single_variant_prompt(cfg, targets, target_file_hint=hint)
    raw, payload = _call_local_llm(prompt, model=model)
    if raw is None:
        return None
    variant, err = _parse_single_variant(raw)
    if variant is None:
        logger.warning("variant #%d parse failed: %s", seed_idx, err)
        _dump_raw_on_failure(f"v{seed_idx}", payload, raw, err or "unknown")
        return None
    # Make sure the model actually honored the hint (soft — if it picked the other file, accept it).
    logger.info(
        "variant #%d OK: target=%s desc=%r",
        seed_idx, list(variant["overlay_files"].keys())[0], variant["description"][:60],
    )
    return variant


def propose_variants(
    cfg: AutoresearchConfig,
    targets: list[str] | None = None,
    max_variants: int | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Ask the local model for up to N variant overlays, one per parallel call.

    Each variant is one LLM call targeting one file. Calls run concurrently —
    vLLM on localhost has 8 decode slots, so 7 concurrent variant calls fit
    with 1 slot left for preemption. Returns the list of variants that parsed
    cleanly (may be shorter than max_variants if some calls failed).
    """
    targets = targets or ["prompts"]
    max_variants = max_variants or cfg.max_variants_per_round
    model = model or cfg.default_model

    logger.info(
        "requesting %d variants (1-per-call, parallel) from model=%s targets=%s",
        max_variants, model, targets,
    )
    # Cap parallelism to 7 so one vLLM slot stays free for interactive traffic.
    max_workers = min(max_variants, 7)
    variants: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_propose_one, cfg, targets, model, idx)
            for idx in range(max_variants)
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                v = fut.result()
            except Exception as exc:
                logger.warning("variant call raised: %s", exc)
                continue
            if v is not None:
                variants.append(v)
    logger.info("hypothesis generator returned %d valid variants (of %d requested)", len(variants), max_variants)
    return variants
