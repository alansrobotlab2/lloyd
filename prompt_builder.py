"""
Lloyd prompt builder — assembles system prompts for SDK sessions.

Combines:
- SOUL.md identity
- Memory (MEMORY.md, USER.md from memories/)
- Skills index
- Platform hints and timestamp
"""

import datetime
from pathlib import Path

LLOYD_HOME = Path(__file__).parent
SOUL_PATH = LLOYD_HOME / "SOUL.md"
MEMORIES_DIR = LLOYD_HOME / "memories"
SKILLS_DIRS = [
    Path.home() / "obsidian" / "skills",
    LLOYD_HOME / "skills",
]


def build_system_prompt(include_skills_index: bool = True) -> str:
    """Build the full system prompt for a Lloyd session."""
    parts = []

    # Identity
    soul = _load_soul()
    if soul:
        parts.append(soul)

    # Memory
    memory = _load_memories()
    if memory:
        parts.append(f"<memory>\n{memory}\n</memory>")

    # Skills index
    if include_skills_index:
        skills = _load_skills_index()
        if skills:
            parts.append(f"<available_skills>\n{skills}\n</available_skills>")

    # Platform hints
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    platform = f"Platform: Lloyd (Claude Agent SDK). Time: {now}. Home: {LLOYD_HOME}"
    parts.append(platform)

    return "\n\n".join(parts)


def _load_soul() -> str | None:
    """Load SOUL.md content, stripping any YAML frontmatter."""
    if not SOUL_PATH.exists():
        return None
    content = SOUL_PATH.read_text(encoding="utf-8").strip()
    if content.startswith("---"):
        end = content.find("\n---\n", 3)
        if end != -1:
            content = content[end + 5:].strip()
    return content or None


def _load_memories() -> str | None:
    """Load MEMORY.md and USER.md from memories/."""
    if not MEMORIES_DIR.exists():
        return None
    parts = []
    for filename in ("MEMORY.md", "USER.md"):
        filepath = MEMORIES_DIR / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## {filename}\n{content}")
    return "\n\n".join(parts) if parts else None


def _load_skills_index() -> str | None:
    """Build a list of available skill names from skill directories."""
    skill_names = []
    for skills_dir in SKILLS_DIRS:
        if not skills_dir.exists():
            continue
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            skill_file = entry / "SKILL.md"
            if skill_file.exists():
                skill_names.append(entry.name)
    if not skill_names:
        return None
    return "Available skills: " + ", ".join(skill_names)
