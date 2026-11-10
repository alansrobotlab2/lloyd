"""Skills discovery endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from agent_mcp.skills import find_active_skill, iter_active_skills

router = APIRouter()


def _skill_rows() -> list[dict]:
    """One row per live skill, off the single walker (#1294).

    This endpoint used to walk `config.yaml skills.directories` itself and parse the
    front matter its own way, and it was wrong in both directions at once on
    2026-09-21: it listed the five `status: archived` skills the prompt no longer
    advertises, and it *dropped* seven live ones. The drop is the expensive half.
    Those seven carry `metadata:\n  openclaw: null`, and

        hermes_meta = meta.get("hermes", meta.get("openclaw", {}))

    returns `None` for a key that is *present with a null value* — `dict.get`'s
    default only fires for an absent key — so the next line's
    `hermes_meta.get("category")` raised `AttributeError`, and the bare
    `except Exception: continue` reported that exception as "this skill does not
    exist". A Skills page that silently omits a skill is worse than one that shows
    it with a blank field, because every later reader trusts the page.

    Every field is therefore read with `or ""` off the walker's parsed front matter:
    a shape the parser did not expect costs a value, never a row. The `hermes` /
    `openclaw` fallback is gone rather than null-guarded, because it was carrying
    nothing — `test_no_live_skill_needs_the_hermes_fallback_the_route_dropped`
    re-measures that over the real roots, so if a skill ever does keep its metadata
    there, a test says so instead of the page quietly losing it.
    """
    return [
        {
            "name": active.name,
            "description": str(active.frontmatter.get("description") or ""),
            "category": str(active.frontmatter.get("category") or ""),
            "enabled": True,
            "configured": True,
            "location": str(active.directory),
        }
        for active in iter_active_skills()
    ]


@router.get("/api/skills")
async def get_skills():
    """List the live skills — the same set the prompt index advertises."""
    return JSONResponse({"workspace": _skill_rows(), "bundled": []})


@router.get("/api/skill-content")
async def get_skill_content(name: str):
    """Read SKILL.md content for a live skill.

    Resolved through `find_active_skill` — the walker's own by-name lookup — so
    the page cannot offer a skill its content endpoint then 404s on, and the
    endpoint cannot serve one the page would not list: a name that leaves a root
    (`../.archived/<x>`, `../../…`) or a quarantined skill is a 404. Until the
    2026-11-09 review this joined `name` onto each root unchecked.
    """
    active = find_active_skill(name)
    if active is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {name}")
    return JSONResponse({"name": active.name,
                         "content": active.skill_file.read_text(encoding="utf-8")})
