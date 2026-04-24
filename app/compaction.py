"""Conversation compaction — manage context window pressure.

Based on Claude Code's client-side compaction strategy:
  - Token estimation: rough char/4 heuristic
  - Conversation truncation: keep system prompt + last N turns
  - Threshold-based auto-truncation for local models
  - Prompt cache awareness for cloud models

For cloud (Anthropic API): the API handles auto-compaction + cache_control.
We mainly estimate to decide when to be conservative.

For local (vLLM/llama-server): we must manage context client-side since
there's no API-level context management.

Both assume 256k context window.
"""

import json
import logging

logger = logging.getLogger("lloyd-server")

# Context window: assume 256k for both cloud and local.
CONTEXT_WINDOW = 256_000

# Reserve this many tokens for model output.
OUTPUT_TOKENS_RESERVED = 20_000

# Buffer before truncation fires.
# We truncate before hitting the limit so the model still has headroom.
TRUNCATION_BUFFER_TOKENS = 32_000

# Max tokens we'll allow in the prompt before truncating.
TRUNCATION_THRESHOLD = CONTEXT_WINDOW - OUTPUT_TOKENS_RESERVED - TRUNCATION_BUFFER_TOKENS
# = 256k - 20k - 32k = 204,000 tokens

# How many turns to keep after truncation.
TURNS_TO_KEEP = 12

# Rough token estimator: 1 token ≈ 4 chars for English text.
TOKENS_PER_CHAR = 4.0

# Token estimation for common message types to be more accurate.
SYSTEM_PROMPT_TOK_ESTIMATE = 20_000  # Lloyd's system prompt is ~20K
SKILL_PROMPT_TOK_ESTIMATE = 5_000   # Skills prefix ~5K


def estimate_tokens(text: str) -> int:
    """Rough token count. ~4 chars/token for English text."""
    return max(1, len(text) // TOKENS_PER_CHAR)


def estimate_message_tokens(message: dict) -> int:
    """Estimate token count for a single message dict."""
    if not message:
        return 0

    # Text content
    text = message.get("content", "")
    if isinstance(text, list):
        # Structured content (blocks)
        text = json.dumps(text)
    text_tokens = estimate_tokens(text if isinstance(text, str) else str(text))

    # Tool call overhead
    if "tool_calls" in message:
        text_tokens += estimate_tokens(json.dumps(message["tool_calls"])) * 3

    if "tool_result" in message:
        text_tokens += estimate_tokens(str(message.get("tool_result", "")))

    return text_tokens


def estimate_conversation_tokens(messages: list[dict], system_prompt: str) -> int:
    """Estimate total tokens in a conversation."""
    system_tokens = estimate_tokens(system_prompt) + SYSTEM_PROMPT_TOK_ESTIMATE + SKILL_PROMPT_TOK_ESTIMATE
    message_tokens = sum(estimate_message_tokens(m) for m in messages)
    return system_tokens + message_tokens


def truncate_conversation(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = TRUNCATION_THRESHOLD,
    turns_to_keep: int = TURNS_TO_KEEP,
) -> list[dict]:
    """Truncate conversation to fit within token budget.

    Keeps:
    1. System prompt (untouched)
    2. Last N turns (user + assistant pairs)
    3. Recent tool results

    Drops everything before the kept window.

    Returns truncated message list.
    """
    tokens = estimate_conversation_tokens(messages, system_prompt)

    if tokens <= max_tokens:
        return messages  # No truncation needed

    # Count turns from the end
    # A "turn" = user message + optional assistant response
    turns = []
    i = len(messages) - 1
    while i >= 0:
        if messages[i].get("role") == "assistant":
            # Found an assistant message — collect backwards to user
            turn = [messages[i]]
            i -= 1
            while i >= 0 and messages[i].get("role") in ("user", "tool"):
                turn.append(messages[i])
                i -= 1
            turns.append(list(reversed(turn)))
        else:
            i -= 1

    # Keep last N turns
    keep_turns = turns[-turns_to_keep:]
    keep_indices = set()
    for turn in keep_turns:
        for msg in turn:
            # Find original index
            for idx, m in enumerate(messages):
                if m is msg or (m.get("content") == msg.get("content") and m.get("role") == msg.get("role")):
                    keep_indices.add(idx)

    # Build truncated list
    truncated = []
    for i, msg in enumerate(messages):
        if i in keep_indices:
            truncated.append(msg)
        elif i > 0:  # Don't drop the first message (user's opening)
            truncated.append({
                "role": msg.get("role", "user"),
                "content": f"[... {estimate_tokens(json.dumps(msg)) if isinstance(msg.get('content'), str) else 0} tokens omitted — conversation truncated]",
            })

    return truncated


def should_truncate(messages: list[dict], system_prompt: str) -> bool:
    """Check if conversation needs truncation."""
    return estimate_conversation_tokens(messages, system_prompt) > TRUNCATION_THRESHOLD


def needs_auto_reset(messages: list[dict], system_prompt: str, is_local: bool) -> bool:
    """Determine if the session should be reset.

    For local models: hard reset at lower threshold (we can't rely on API).
    For cloud models: the API handles auto-compaction, but we still monitor
    for sessions that are getting very long.
    """
    tokens = estimate_conversation_tokens(messages, system_prompt)

    if is_local:
        # Hard threshold for local: 180k tokens
        return tokens > 180_000
    else:
        # Cloud models handle auto-compaction. Only hard-reset at 220k.
        return tokens > 220_000


def build_compaction_note(tokens: int, was_truncated: bool) -> str:
    """Build a note about compaction to add to the conversation."""
    if not was_truncated:
        return ""

    return f"[Compaction note: conversation was truncated at ~{tokens} tokens. Old context before the last {TURNS_TO_KEEP} turns has been removed.]"


# ---------------------------------------------------------------------------
# Local-model compaction — format conversation as text for non-resume APIs
# ---------------------------------------------------------------------------

def is_local_model(model: str) -> bool:
    """Check if a model URL points to a local/self-hosted endpoint.

    Cloud Anthropic URLs: api.anthropic.com, api2.anthropic.com
    All others (http://, https:// to custom hosts, or non-API hostnames)
    are treated as local/self-hosted.
    """
    if not model:
        return False
    lower = model.lower()
    if "api.anthropic" in lower:
        return False
    return True


# ---------------------------------------------------------------------------
# Chat template — configurable per model family. Local LLM APIs expect
# specific role markers and message delimiters; using the wrong ones means
# the model treats the whole conversation as raw text with no structural
# awareness. Each family defines (role_map, delimiter, join_char).
# ---------------------------------------------------------------------------

_CHAT_TEMPLATES = {
    # Qwen family — <|im_start|><|im_end|> delimiters
    "qwen": {
        "role_map": {"user": "user", "assistant": "assistant", "system": "system", "tool": "assistant"},
        "delimiter": "<|im_end|>\n<|im_start|>",
        "join": "\n",
        "wrap_open": "<|im_start|>assistant\n",
    },
    # Llama 3/4 family — <START/END> markers
    "llama": {
        "role_map": {"user": "user", "assistant": "assistant", "system": "system", "tool": "assistant"},
        "delimiter": "\n",
        "join": "\n",
        "wrap_open": "",
    },
    # Mistral — <|im_start|>…<|im_end|>
    "mistral": {
        "role_map": {"user": "user", "assistant": "assistant", "system": "system", "tool": "assistant"},
        "delimiter": "<|im_end|>\n<|im_start|>",
        "join": "\n",
        "wrap_open": "<|im_start|>assistant\n",
    },
}

# Default fallback for unknown model families.
_DEFAULT_TEMPLATE = _CHAT_TEMPLATES["qwen"]


def _get_template_for_model(model: str) -> dict:
    """Return the chat template config for a given model name."""
    lower = model.lower()
    if "qwen" in lower:
        return _CHAT_TEMPLATES["qwen"]
    if "llama" in lower or "metallama" in lower or "xwin" in lower:
        return _CHAT_TEMPLATES["llama"]
    if "mistral" in lower:
        return _CHAT_TEMPLATES["mistral"]
    # Default to Qwen (most common local model family right now)
    return _DEFAULT_TEMPLATE


def _format_message_block(msg: dict, template: dict) -> str:
    """Format a single message into template-specific text."""
    role = msg.get("role", "user")
    content = msg.get("content", "")

    # Extract text from structured content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
                elif btype == "tool_use":
                    name = block.get("name", "")
                    inp = json.dumps(block.get("input", {}))
                    parts.append(f"[tool_call: {name}({inp})]")
                elif btype == "tool_result":
                    result = block.get("content", "")
                    if isinstance(result, str):
                        parts.append(f"[tool_result: {result[:500]}]")
                    elif isinstance(result, list):
                        for b in result:
                            if isinstance(b, dict) and b.get("type") == "text":
                                parts.append(f"[tool_result: {b.get('text', '')[:500]}]")
                elif btype == "image":
                    parts.append("[image]")
                elif btype == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        parts.append(f"[thinking: {thinking[:2000]}]")
            elif isinstance(block, str):
                parts.append(block)
        content_str = "\n".join(parts) if parts else ""
    elif isinstance(content, str):
        content_str = content
    else:
        content_str = str(content)

    # Map role via template
    mapped_role = template["role_map"].get(role, role)
    return f"{mapped_role}\n{content_str}"


def format_conversation_for_local(messages: list[dict], model: str = "") -> str:
    """Format a conversation history as a text block for local LLM APIs.

    Converts structured message dicts into a template-specific conversation
    format so the model can parse the role boundaries correctly.

    Args:
        messages: List of message dicts (role + content).
        model: Model name for template selection (e.g. 'primary' → Qwen).
    """
    if not messages:
        return ""

    template = _get_template_for_model(model)
    blocks = []
    for msg in messages:
        blocks.append(_format_message_block(msg, template))

    # Join with delimiter, prepend opening for assistant context
    joined = template["join"].join(blocks)
    if template["wrap_open"]:
        joined = template["wrap_open"] + joined

    return joined


def check_local_compaction(session_path: str) -> dict:
    """Check if a local model session needs compaction.

    Reads the session JSON, estimates tokens, and returns compaction
    status. Used before each query() call for local models.

    Returns:
        {
            "is_local": bool,
            "needs_compaction": bool,
            "token_count": int,
            "message_count": int,
            "turn_count": int,
            "should_reset": bool,
            "recent_messages": list[dict],
        }
    """
    try:
        if not session_path or not session_path.exists():
            return {
                "is_local": True,
                "needs_compaction": False,
                "token_count": 0,
                "message_count": 0,
                "turn_count": 0,
                "should_reset": False,
                "recent_messages": [],
            }

        data = json.loads(session_path.read_text())
        messages = data.get("messages", [])

        if not messages:
            return {
                "is_local": True,
                "needs_compaction": False,
                "token_count": 0,
                "message_count": 0,
                "turn_count": 0,
                "should_reset": False,
                "recent_messages": [],
            }

        message_count = len(messages)
        turn_count = sum(1 for m in messages if m.get("role") == "user")
        token_count = estimate_conversation_tokens(messages, "")

        # Count last 30 turns (60 messages) for compaction check
        recent = []
        i = len(messages) - 1
        turns_found = 0
        while i >= 0 and turns_found < 30:
            if messages[i].get("role") == "user":
                recent.insert(0, messages[i])
                turns_found += 1
                if i > 0 and messages[i-1].get("role") == "assistant":
                    recent.insert(0, messages[i-1])
                    i -= 1
            i -= 1

        # Need compaction if over threshold
        needs_compaction = token_count > TRUNCATION_THRESHOLD

        # Hard reset if over 180k (local models can't rely on API)
        should_reset = token_count > 180_000

        return {
            "is_local": True,
            "needs_compaction": needs_compaction,
            "token_count": token_count,
            "message_count": message_count,
            "turn_count": turns_found,
            "should_reset": should_reset,
            "recent_messages": recent,
        }

    except Exception as e:
        logger.warning(f"Compaction check failed: {e}")
        return {
            "is_local": True,
            "needs_compaction": False,
            "token_count": 0,
            "message_count": 0,
            "turn_count": 0,
            "should_reset": False,
            "recent_messages": [],
        }

