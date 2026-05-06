"""Shared todo-list validation used by both TodoWrite and ExitPlanMode.

Lives in `agent_mcp/_todo_validation.py` so it can be imported by both
[builtin_todo.py](builtin_todo.py) (the standalone TodoWrite tool) and
[builtin_plan.py](builtin_plan.py) (the ExitPlanMode tool whose commit
path replaces session.todos using the same shape).
"""

from __future__ import annotations

from typing import Any


VALID_STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed")


def validate_todos(raw: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate a todos payload. Returns (list, None) on success or
    (None, error_message) on failure.

    Each item must have non-empty `content` (str), a `status` from
    `VALID_STATUSES`, and a non-empty `activeForm` (str). An optional
    `stage` (int) is preserved if present (Plan B / Phase 2 stage grouping).
    Additional fields are dropped — the persisted shape is canonical.
    """
    if not isinstance(raw, list):
        return None, "todos must be an array"
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None, f"todos[{i}] must be an object"
        content = item.get("content")
        status = item.get("status")
        active_form = item.get("activeForm")
        if not isinstance(content, str) or not content.strip():
            return None, f"todos[{i}].content must be a non-empty string"
        if status not in VALID_STATUSES:
            return None, (
                f"todos[{i}].status must be one of {list(VALID_STATUSES)}, "
                f"got {status!r}"
            )
        if not isinstance(active_form, str) or not active_form.strip():
            return None, f"todos[{i}].activeForm must be a non-empty string"
        validated: dict[str, Any] = {
            "content": content,
            "status": status,
            "activeForm": active_form,
        }
        stage = item.get("stage")
        if stage is not None:
            if not isinstance(stage, int):
                return None, f"todos[{i}].stage must be an integer if provided"
            validated["stage"] = stage
        out.append(validated)
    return out, None
