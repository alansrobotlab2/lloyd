"""Skills discovery endpoints."""

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.config import CONFIG


router = APIRouter()


@router.get("/api/skills")
async def get_skills():
    """List available skills from configured directories."""
    skills = []
    for dir_path in CONFIG.get("skills", {}).get("directories", []):
        expanded = Path(dir_path.replace("~", str(Path.home())))
        if not expanded.exists():
            continue
        for entry in sorted(expanded.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                content = skill_file.read_text(encoding="utf-8")
                fm = {}
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}

                meta = fm.get("metadata", {})
                if isinstance(meta, dict):
                    hermes_meta = meta.get("hermes", meta.get("openclaw", {}))
                else:
                    hermes_meta = {}

                skills.append({
                    "name": entry.name,
                    "description": fm.get("description") or hermes_meta.get("description", ""),
                    "category": fm.get("category") or hermes_meta.get("category", ""),
                    "enabled": True,
                    "configured": True,
                    "location": str(entry),
                })
            except Exception:
                continue

    return JSONResponse({"workspace": skills, "bundled": []})


@router.get("/api/skill-content")
async def get_skill_content(name: str):
    """Read SKILL.md content for a skill."""
    for dir_path in CONFIG.get("skills", {}).get("directories", []):
        expanded = Path(dir_path.replace("~", str(Path.home())))
        skill_file = expanded / name / "SKILL.md"
        if skill_file.exists():
            return JSONResponse({"name": name, "content": skill_file.read_text(encoding="utf-8")})
    raise HTTPException(status_code=404, detail=f"Skill not found: {name}")
