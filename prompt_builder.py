"""
Lloyd prompt builder — assembles system prompts for SDK sessions.

Combines:
- SOUL.md identity
- Memory (MEMORY.md, USER.md from memories/)
- Skills index
- Platform hints and timestamp
"""

import datetime
import os
from pathlib import Path

LLOYD_HOME = Path(__file__).parent
# Use LLOYD_HOME instead of Path.home() to avoid distrobox path mismatch issues
# where Path.home() resolves to the host home instead of the container's isolated home
_CANON_SOUL_PATH = LLOYD_HOME.parent / "obsidian" / "lloyd" / "SOUL.md"
_CANON_MEMORIES_DIR = LLOYD_HOME.parent / "obsidian" / "lloyd"
_CANON_SKILLS_DIRS = [
    LLOYD_HOME.parent / "obsidian" / "skills",
    LLOYD_HOME / "skills",
]


def _resolve_overlay(overlay_dir: str | Path | None) -> Path | None:
    """Normalize an overlay-dir argument (falls back to LLOYD_OVERLAY_DIR env var)."""
    raw = overlay_dir if overlay_dir is not None else os.environ.get("LLOYD_OVERLAY_DIR")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path if path.exists() else None


# Back-compat public names (some callers import these directly)
SOUL_PATH = _CANON_SOUL_PATH
MEMORIES_DIR = _CANON_MEMORIES_DIR
SKILLS_DIRS = _CANON_SKILLS_DIRS


def build_system_prompt(include_skills_index: bool = True, overlay_dir: str | Path | None = None) -> str:
    """Build the full system prompt for a Lloyd session.

    `overlay_dir` (or the LLOYD_OVERLAY_DIR env var as fallback) redirects
    SOUL.md / MEMORY.md / USER.md / skills/ reads to a variant sandbox, with
    fallthrough to the canonical vault for any file the overlay doesn't
    provide. Thread-safe when callers pass `overlay_dir` explicitly.
    """
    overlay = _resolve_overlay(overlay_dir)
    parts = []

    soul = _load_soul(overlay)
    if soul:
        parts.append(soul)

    memory = _load_memories(overlay)
    if memory:
        parts.append(f"<memory>\n{memory}\n</memory>")

    if include_skills_index:
        skills = _load_skills_index(overlay)
        if skills:
            parts.append(f"<available_skills>\n{skills}\n</available_skills>\nNote: relevant skill content is automatically injected into each user message as <context> when matched.")

    # Platform hints — NOTE: no timestamp here; a per-minute timestamp busts
    # vLLM's prefix cache, forcing full re-prefill of the system prompt every turn.
    # The model gets the current time via tool calls or conversation context instead.
    platform = (
        "Platform: Lloyd (Claude Agent SDK). "
        f"Home: {LLOYD_HOME}. "
        "Vault: ~/obsidian/. Knowledge notes go in ~/obsidian/knowledge/. "
        "All persistent notes, research output, and files created by the agent "
        "go in the vault (~/obsidian/), NOT in the lloyd project directory."
    )
    parts.append(platform)

    bg_tasks = (
        "Background bash tasks: pass run_in_background=true to Bash for any "
        "long-running command (builds, finds, deploys, monitoring loops). The "
        "tool returns a task_id and an output_file path immediately so you can "
        "keep working. When the command exits, a <task_notification> message "
        "appears in the conversation on a later turn carrying the same task_id, "
        "the final status, and the output_file path. Use Read on the output_file "
        "to inspect what the command produced. Do not respond directly to a "
        "<task_notification> unless its result changes your plan."
    )
    parts.append(bg_tasks)

    return "\n\n".join(parts)


def _load_soul(overlay: Path | None = None) -> str | None:
    """Load SOUL.md content, stripping any YAML frontmatter."""
    path = overlay / "SOUL.md" if (overlay and (overlay / "SOUL.md").exists()) else _CANON_SOUL_PATH
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("---"):
        end = content.find("\n---\n", 3)
        if end != -1:
            content = content[end + 5:].strip()
    return content or None


def _load_memories(overlay: Path | None = None) -> str | None:
    """Load MEMORY.md and USER.md — overlay dir takes priority, canonical as fallback."""
    parts = []
    for filename in ("MEMORY.md", "USER.md"):
        content = None
        if overlay and (overlay / filename).exists():
            content = (overlay / filename).read_text(encoding="utf-8").strip()
        elif (_CANON_MEMORIES_DIR / filename).exists():
            content = (_CANON_MEMORIES_DIR / filename).read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"## {filename}\n{content}")
    return "\n\n".join(parts) if parts else None


def _load_skills_index(overlay: Path | None = None) -> str | None:
    """Build a list of available skill names from skill directories (overlay first)."""
    dirs: list[Path] = []
    if overlay and (overlay / "skills").exists():
        dirs.append(overlay / "skills")
    dirs.extend(_CANON_SKILLS_DIRS)

    skill_names: list[str] = []
    seen: set[str] = set()
    for skills_dir in dirs:
        if not skills_dir.exists():
            continue
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name in seen:
                continue
            skill_file = entry / "SKILL.md"
            if skill_file.exists():
                skill_names.append(entry.name)
                seen.add(entry.name)
    if not skill_names:
        return None
    return "Available skills: " + ", ".join(skill_names)
