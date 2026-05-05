#!/usr/bin/env python3
"""Extract real conversation prompts from session history for benchmarking."""

import json
import os
import re
import sys

SESSIONS_DIR = os.path.expanduser("~/lloyd/sessions")
OUTPUT_FILE = "/tmp/mtp-bench-prompts.jsonl"
MAX_SESSIONS = 8
MAX_CHARS = 80_000
VALID_ROLES = {"user", "assistant"}


def extract_messages(path):
    """Parse a lloyd session JSON file into OpenAI-style messages, ending at last user msg."""
    with open(path) as f:
        session = json.load(f)

    messages = []
    for msg in session.get("messages", []):
        role = msg.get("role", "")
        if role not in VALID_ROLES:
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            text = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        elif isinstance(content, str):
            text = content
        else:
            continue
        text = text.strip()
        if not text:
            continue
        text = re.sub(r'<system-reminder>.*?</system-reminder>', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'<available-deferred-tools>.*?</available-deferred-tools>', '', text, flags=re.DOTALL).strip()
        if not text:
            continue
        messages.append({"role": role, "content": text})

    # Collapse same-role runs
    cleaned = []
    for m in messages:
        if cleaned and cleaned[-1]["role"] == m["role"]:
            cleaned[-1]["content"] += "\n" + m["content"]
        else:
            cleaned.append(m)
    messages = cleaned

    # Must start with user
    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    # Truncate at last user message
    last_user = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), -1)
    if last_user < 0:
        return None
    messages = messages[: last_user + 1]

    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars > MAX_CHARS or total_chars < 100:
        return None

    return messages


def main():
    files = []
    for f in os.listdir(SESSIONS_DIR):
        if f.endswith(".json"):
            path = os.path.join(SESSIONS_DIR, f)
            files.append((os.path.getmtime(path), path, f))
    files.sort(reverse=True)

    prompts = []
    for _, path, fname in files:
        if len(prompts) >= MAX_SESSIONS:
            break
        try:
            messages = extract_messages(path)
        except Exception:
            continue
        if messages is None:
            continue
        total_chars = sum(len(m["content"]) for m in messages)
        prompts.append({"session": fname, "messages": messages})
        print(f"  {fname}: {len(messages)} msgs, ~{total_chars} chars", file=sys.stderr)

    with open(OUTPUT_FILE, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"\nExtracted {len(prompts)} prompts to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
