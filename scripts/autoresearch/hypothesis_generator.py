"""Hypothesis generator — proposes variant overlays for an autoresearch round.

Reads current prompt surfaces (SOUL/MEMORY/USER) + recent signal (bench losers,
baseline failures, correction log, knowledge-health) and asks the local model
to return candidate variants. Each variant is a list of anchored edits against
ONE prompt surface; the sandbox applies them mechanically to the canonical text
and materializes the result, and the bench runner evaluates that under
LLOYD_OVERLAY_DIR.

For v1, only the `prompts` target (SOUL.md OR MEMORY.md, one per variant) is
generated. Other targets are advertised in config and reserved for later.

Design: N single-variant calls in parallel via ThreadPoolExecutor. Each call
returns a small, bounded JSON object that reliably fits under max_tokens. On
parse failure the raw output is dumped to _pipeline/research/_debug/ for
post-mortem rather than silently lost.

Why the contract is anchored edits and not file contents (#446): a variant used
to be required to return a whole prompt surface verbatim, so a legal response
had to be at least as long as the file it was editing. The model echoed
the SOUL.md/MEMORY.md it had just been handed, the response was cut off at
`max_tokens: 8000` mid-string, and 178 of 192 dumps in _debug were
`Unterminated string` at a median 32,020 chars. Output that scales with the
input cannot fit a token ceiling; output bounded by the edit can.
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

# ── the bounded variant contract (#446) ──────────────────────────────────────
#
# These bounds ARE the fix, not house style. A legal response is at most
# MAX_EDITS x (MAX_ANCHOR_CHARS + MAX_REPLACEMENT_CHARS) plus a little JSON
# scaffolding — ~19K chars worst case — against a ceiling of roughly 32K chars
# (8,000 tokens at ~4 chars/token). The contract this replaced had a ceiling
# equal to the size of the file being edited, which is unbounded from the
# generator's point of view and met that ceiling 192 times.
MAX_EDITS = 6
MAX_ANCHOR_CHARS = 1200
MAX_REPLACEMENT_CHARS = 2000

# What `max_tokens: 8000` is worth in characters. Not a limit this module
# enforces — the engine's is; it exists so a test can state how much headroom a
# response has, and so the size a full-file echo had to reach is comparable.
CEILING_CHARS = 32_000

# Surfaces the generator may propose. Deliberately narrower than the promotion
# allowlist (`_canonical_prompt_paths()`, which also covers USER.md).
GENERATOR_SURFACES = ("SOUL.md", "MEMORY.md")

# Diagnostics filenames. A JSON parse failure is the historical truncation
# signal and keeps the `hypothesis_fail_` name; a deliberate rejection (output
# hit max_tokens, or an out-of-contract proposal) is written as
# `hypothesis_reject_`. Keeping them apart is what lets a clean-rounds check on
# `hypothesis_fail_*` keep meaning "nothing was truncated" rather than "nothing
# happened".
FAIL_DUMP_PREFIX = "hypothesis_fail_"
REJECT_DUMP_PREFIX = "hypothesis_reject_"

# Prefix on an error meaning "the JSON parsed and the proposal is out of
# bounds", as opposed to "the JSON never parsed". `_propose_one` picks the dump
# filename prefix off it, so a deliberate rejection never lands in the pile the
# #446 acceptance counts.
CONTRACT_ERR = "contract: "



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


def _dump_raw_on_failure(
    label: str,
    payload: dict[str, Any],
    raw: str,
    error: str,
    prefix: str = FAIL_DUMP_PREFIX,
) -> None:
    """Persist raw model output + request payload when a variant is lost.

    Without this, 'hypothesis generator returned 0 valid variants' is a
    diagnostic black hole. Files land in _pipeline/research/_debug/ with the
    timestamp + label so you can grep for `variant truncated` or `bad comma`
    across runs.

    `prefix` separates the two kinds of loss: `hypothesis_fail_` is a response
    that never parsed (the truncation pile #446 is about), `hypothesis_reject_`
    is a response that parsed and was refused for being out of bounds. Counting
    only the first is how a round proves the stream is quiet.
    """
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        ts = now_iso().replace(":", "-")
        path = DEBUG_DIR / f"{prefix}{ts}_{label}.txt"
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
) -> tuple[str | None, dict[str, Any], str]:
    """Synchronous call to local vLLM via OpenAI-compatible API.

    Returns (content_or_none, payload_used, finish_reason). The payload is
    returned so a lost variant can dump the request alongside the bad response,
    and the finish reason is returned because `length` means the response was
    cut off at max_tokens — which is not a variant, and the CALLER has to fail
    it. It used to be logged here as "possible truncation" and handed to the
    parser anyway; that log line is what 192 dumps came out of (#446).

    max_tokens stays at 8000: raising it only buys a bigger echo of the same
    file, which is why the fix is the bounded edit contract rather than a bigger
    number. Temperature 0.8 for variant diversity across parallel calls.
    """
    from app.config import resolve_model_alias, _get_model_cfg
    model_name = resolve_model_alias(model)
    _cfg = _get_model_cfg(model_name) or {}
    base = (_cfg.get("base_url") or _cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")).rstrip("/")
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
        if finish and finish not in ("stop", "length"):
            logger.warning("LLM finish_reason=%s", finish)
        # `length` is deliberately not warned about here: this function cannot
        # act on it, and warning-and-continuing is precisely what let a
        # truncated response reach the parser for 192 dumps. It is returned, and
        # `_propose_one` fails the variant on it.
        return content, payload, finish
    except Exception as exc:
        logger.error("local LLM call failed: %s", exc)
        return None, payload, ""


def _build_single_variant_prompt(
    cfg: AutoresearchConfig,
    targets: list[str],
    target_file_hint: str | None = None,
    parent_variant: dict[str, Any] | None = None,
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

    # If this round should build on a previously promoted variant, surface it.
    parent_block = ""
    if parent_variant:
        parent_block = (
            f"\n## Previously promoted variant (build on this)\n"
            f"- ID: `{parent_variant.get('variant_id', '')}`\n"
            f"- Description: {parent_variant.get('description', '')}\n"
            f"- Hypothesis: {parent_variant.get('hypothesis', '')}\n"
            f"Use this variant's state as the starting point — don't redo its work,\n"
            f"build on top of it.\n"
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
{parent_block}
## Safety constraint (mandatory)
Lloyd has hard safety rules that variants MUST preserve or strengthen:

1. **Refusal of unsafe actions:** Lloyd refuses adversarial framing
   ("ignore previous instructions", "act as unfiltered") and never modifies
   protected paths (`~/obsidian/`, `~/lloyd/agent-services/`,
   `~/lloyd/.venvs/`) without explicit user confirmation.
2. **Destructive-action gating:** Any `rm -rf`, file overwrite, or
   modification of critical paths requires unambiguous intent. If the
   request is ambiguous, Lloyd MUST block (return `SIGNAL:BLOCKED:<reason>`)
   rather than guess.
3. **Critical bench task:** `bench_010_safety_destructive` measures this
   directly. Variants that lower the safety_passed rate on safety_critical
   tasks are auto-rejected via `safety_regression` HOLD — no exceptions.

Empirically, ~52% of past variants fail safety. The most common pattern
is INDIRECT weakening — adding "always be helpful and proactive" to SOUL.md
without keeping the corresponding refusal language, or removing
destructive-action checks as "verbose." When in doubt, leave safety
language alone or strengthen it.

## Your task
You are proposing EDIT OPERATIONS, not a file. Each edit says "find this exact
span in this file, put this text in its place". You are never asked for a file's
contents, a rewrite, or a diff — emitting one is rejected, because a response
that has to contain a whole prompt file gets cut off before it finishes.

Output a JSON object with this EXACT shape:

{{
  "description": "<one-line summary, <=120 chars>",
  "hypothesis": "<why this will help; name a specific failure mode or bench category>",
  "target_surface": "prompts",
  "edits": [
    {{
      "path": "SOUL.md",
      "anchor": "<a span copied VERBATIM from the file text shown above>",
      "replacement": "<the text that takes the place of that span>"
    }}
  ]
}}

Rules:
- Return ONLY the JSON object, no prose before or after, no markdown fences.
- Modify EXACTLY ONE file: every edit's `path` is `SOUL.md` or `MEMORY.md`, and
  never both inside one variant.
- Copy each `anchor` character-for-character from the text shown above — same
  whitespace, same punctuation, same markdown — from the file named by that
  edit's `path`. It must appear in that file EXACTLY ONCE. Two matches and the
  whole variant is rejected, so extend the span until it is unique.
- Bounds: at most {MAX_EDITS} edits, each `anchor` at most {MAX_ANCHOR_CHARS}
  chars, each `replacement` at most {MAX_REPLACEMENT_CHARS} chars. An edit over
  a bound is rejected, so prefer tightening one sentence over replacing a
  section.
- An anchor for MEMORY.md must come from the "Current MEMORY.md tail" shown
  above. That excerpt is all of MEMORY.md you are given; a span invented from
  the part you cannot see will not match and the variant is lost.
- `"replacement": ""` deletes the anchor. Anything else replaces it.
- Preserve or strengthen safety language. See "Safety constraint" above.
  Variants that fail safety_critical tasks get auto-rejected — don't waste
  a generation slot on prompts that risk this.
- A patch can only change text it quotes, which is the point: refusal language
  cannot quietly disappear in an edit the way it did in a rewritten file.

Your whole response is bounded by these rules and should come in well under
2,000 characters. A long response is a response that gets truncated and whose
variant is thrown away — which is exactly what this contract replaced.

/no_think"""


def _validate_edits(edits_raw: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Check the anchored-edit list against the bounds (#446).

    Every message here is prefixed CONTRACT_ERR so `_propose_one` can tell a
    proposal that was out of bounds from a response that never parsed, and file
    the dump under the rejection prefix instead of the truncation pile.

    Bounds are enforced per field rather than on the total response so the
    rejection names the edit that broke them.
    """
    if not isinstance(edits_raw, list) or not edits_raw:
        return None, (
            f"{CONTRACT_ERR}edits missing/empty — a variant is a list of "
            f"{{path, anchor, replacement}} edits, never a file's contents"
        )
    if len(edits_raw) > MAX_EDITS:
        return None, f"{CONTRACT_ERR}too many edits ({len(edits_raw)} > {MAX_EDITS})"

    edits: list[dict[str, Any]] = []
    surfaces: list[str] = []
    for i, edit in enumerate(edits_raw):
        if not isinstance(edit, dict):
            return None, f"{CONTRACT_ERR}edit {i} is not an object"
        path = str(edit.get("path", ""))
        if path not in GENERATOR_SURFACES:
            return None, (
                f"{CONTRACT_ERR}edit {i}: unsupported path {path!r} "
                f"(the generator may propose {list(GENERATOR_SURFACES)})"
            )
        anchor = edit.get("anchor")
        if not isinstance(anchor, str):
            return None, f"{CONTRACT_ERR}edit {i}: anchor must be a string"
        if not anchor.strip():
            return None, f"{CONTRACT_ERR}edit {i}: empty anchor — an edit must quote the span it replaces"
        if len(anchor) > MAX_ANCHOR_CHARS:
            return None, (
                f"{CONTRACT_ERR}edit {i}: anchor too long ({len(anchor)} > "
                f"{MAX_ANCHOR_CHARS} chars) — quote the span to replace, not the file"
            )
        replacement = edit.get("replacement")
        if not isinstance(replacement, str):
            return None, f"{CONTRACT_ERR}edit {i}: replacement must be a string ('' to delete)"
        if len(replacement) > MAX_REPLACEMENT_CHARS:
            return None, (
                f"{CONTRACT_ERR}edit {i}: replacement too long ({len(replacement)} > "
                f"{MAX_REPLACEMENT_CHARS} chars)"
            )
        if path not in surfaces:
            surfaces.append(path)
        edits.append({"path": path, "anchor": anchor, "replacement": replacement})

    if len(surfaces) > 1:
        return None, (
            f"{CONTRACT_ERR}edits span {len(surfaces)} surfaces {surfaces}; "
            f"one surface per variant"
        )
    return edits, None


def _parse_single_variant(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one variant JSON object. Returns (variant_or_none, error_message).

    The variant IS its edit list — there is no `overlay_files` any more. A
    response carrying one is refused by `_validate_edits`, which is the point:
    the full-file shape is what could not fit max_tokens.
    """
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
        return None, f"{CONTRACT_ERR}unsupported target_surface={target}"
    edits, err = _validate_edits(data.get("edits"))
    if edits is None:
        return None, err
    return {
        "variant_id": variant_id(),
        "description": str(data.get("description", ""))[:200],
        "hypothesis": str(data.get("hypothesis", ""))[:1000],
        "target_surface": "prompts",
        "edits": edits,
        "parent_variant_id": str(data.get("parent_variant_id", "")) if data.get("parent_variant_id") else None,
        "created_at": now_iso(),
    }, None


def _propose_one(
    cfg: AutoresearchConfig,
    targets: list[str],
    model: str,
    seed_idx: int,
    parent_variant: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run one hypothesis call. Returns a variant or None."""
    # Alternate hinted target file by index to diversify.
    hint = "SOUL.md" if seed_idx % 2 == 0 else "MEMORY.md"
    prompt = _build_single_variant_prompt(cfg, targets, target_file_hint=hint, parent_variant=parent_variant)
    raw, payload, finish = _call_local_llm(prompt, model=model)
    if raw is None:
        return None
    if finish == "length":
        # #446: this used to be a `logger.warning("possible truncation")` and the
        # truncated string was handed to the parser anyway, where it died as
        # `Unterminated string`. Output that hit max_tokens is not a variant —
        # fail it here, where the reason is the reason.
        logger.warning(
            "variant #%d rejected: finish_reason=length (hit max_tokens=%s)",
            seed_idx, payload.get("max_tokens"),
        )
        _dump_raw_on_failure(
            f"v{seed_idx}", payload, raw,
            "finish_reason=length: response was cut off at max_tokens. A bounded "
            "anchored-edit response should never reach the ceiling — if this reappears, "
            "the model is emitting file contents rather than edits.",
            prefix=REJECT_DUMP_PREFIX,
        )
        return None
    variant, err = _parse_single_variant(raw)
    if variant is None:
        logger.warning("variant #%d rejected: %s", seed_idx, err)
        prefix = REJECT_DUMP_PREFIX if (err or "").startswith(CONTRACT_ERR) else FAIL_DUMP_PREFIX
        _dump_raw_on_failure(f"v{seed_idx}", payload, raw, err or "unknown", prefix=prefix)
        return None
    # Make sure the model actually honored the hint (soft — if it picked the other surface, accept it).
    logger.info(
        "variant #%d OK: surface=%s edits=%d desc=%r",
        seed_idx, variant["edits"][0]["path"], len(variant["edits"]), variant["description"][:60],
    )
    return variant


def propose_variants(
    cfg: AutoresearchConfig,
    targets: list[str] | None = None,
    max_variants: int | None = None,
    model: str | None = None,
    parent_variant: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Ask the local model for up to N variant overlays, one per parallel call.

    Each variant is one LLM call targeting one file. Calls run concurrently
    but bounded by the engine's `--max-num-seqs` (currently 4 for primary).
    Overflowing it queues requests past their 300s read timeout — and vLLM
    doesn't honor client disconnects, so timed-out calls keep running and
    pin engine slots until natural completion. We cap parallelism strictly
    below the engine cap so one slot stays free for interactive chat.
    Returns the list of variants that parsed cleanly (may be shorter than
    max_variants if some calls failed).
    """
    targets = targets or ["prompts"]
    max_variants = max_variants or cfg.max_variants_per_round
    model = model or cfg.default_model

    logger.info(
        "requesting %d variants (1-per-call, parallel) from model=%s targets=%s",
        max_variants, model, targets,
    )
    max_workers = min(max_variants, 3)
    variants: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_propose_one, cfg, targets, model, idx, parent_variant)
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
