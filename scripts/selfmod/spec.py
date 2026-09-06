"""What a self-modification round is allowed to change.

With no human review tier, this file is the entire control surface, and a
control surface that can quietly modify itself is not a control surface. Hence
three tiers rather than two:

  * **allowed** — ordinary code. Lands automatically once the gate passes.
  * **protected** — the gate, the guardian, and the restart/health path the
    rollback itself depends on. Lloyd may edit these, but the change must
    additionally survive a live rollback drill (gate rung 6). A bound on *how*
    it changes, not on *whether* — which is the difference between guarding a
    capability and amputating it.
  * **denied** — never landed by this loop, whatever the run spec says.
    `config.yaml` and `data/tool_overrides.yaml` because a round could disable
    the agent's own Bash/Edit tools and lock itself out with no Python change
    at all; `pytest.ini` and `.gitignore` because they define what the gate
    even sees; `.env*` because secrets; `web/**` because the gate does not
    build the frontend and therefore cannot verify it.

Note `requirements.txt` / `requirements.lock` are *allowed*, but only because
gate rung 3 builds a throwaway venv from them (btrfs reflink clone + `uv pip
install` of the delta) and boots the canary against it. Without that rung they
would belong in `denied`.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

ALLOWED_GLOBS: tuple[str, ...] = (
    "app/**",
    "agent_mcp/**",
    "workers/**",
    "eval/**",
    "tests/**",
    "scripts/**",
    "architecture/**",
    "prompt_builder.py",
    "autonomy.py",
    "prefetch.py",
    "usage_store.py",
    "server.py",
    "requirements.txt",
    "requirements.lock",
    "CLAUDE.md",
    "README.md",
)

PROTECTED_GLOBS: tuple[str, ...] = (
    "scripts/selfmod/**",
    "agent-services/guardian/**",
    "agent-services/systemd/**",
    "agent-services/supervisor/**",
    "agent-services/bin/**",
    "app/routers/health.py",
    "app/routers/selfmod.py",
    "app/supervisor_client.py",
    "app/lifecycle.py",
    "app/gitinfo.py",
)

DENIED_GLOBS: tuple[str, ...] = (
    ".gitignore",
    "pytest.ini",
    "config.yaml",
    "data/**",
    ".env",
    ".env.*",
    "web/**",
    ".git/**",
    ".venvs/**",
)


def _match(path: str, globs: tuple[str, ...]) -> bool:
    for pattern in globs:
        if fnmatch.fnmatch(path, pattern):
            return True
        # fnmatch's `*` crosses `/`, but `a/**` should also match `a/b/c.py`
        # and, for a directory pattern, the directory's own entries.
        if pattern.endswith("/**") and (
            path == pattern[:-3] or path.startswith(pattern[:-2])
        ):
            return True
    return False


def normalize(path: str) -> str | None:
    """Return a clean repo-relative POSIX path, or None if it escapes the repo."""
    if not path or not isinstance(path, str):
        return None
    p = path.strip().replace("\\", "/")
    if not p or p.startswith("/"):
        return None
    parts = PurePosixPath(p).parts
    if any(part == ".." for part in parts):
        return None
    parts = tuple(part for part in parts if part not in (".", ""))
    if not parts:
        return None
    return "/".join(parts)


def classify(path: str) -> str:
    """Return 'denied' | 'protected' | 'allowed' | 'unlisted' for one path.

    Order is the whole point: **denied beats everything**, including an
    explicit entry in a run spec's `writable_paths`. A round cannot widen its
    own permissions by asking nicely.
    """
    norm = normalize(path)
    if norm is None:
        return "denied"
    if _match(norm, DENIED_GLOBS):
        return "denied"
    if _match(norm, PROTECTED_GLOBS):
        return "protected"
    if _match(norm, ALLOWED_GLOBS):
        return "allowed"
    return "unlisted"


def classify_all(paths: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"allowed": [], "protected": [], "denied": [], "unlisted": []}
    for p in paths:
        out[classify(p)].append(p)
    return out


def check_scope(paths: list[str]) -> tuple[bool, str, dict[str, list[str]]]:
    """Gate rung 0's diff-scope check.

    Returns (ok, reason, buckets). `ok` is False if anything is denied or
    unlisted. Protected paths are permitted here — they are what turns on the
    rollback drill in rung 6, handled by the caller.
    """
    buckets = classify_all(paths)
    if buckets["denied"]:
        return False, f"denied paths in diff: {sorted(buckets['denied'])}", buckets
    if buckets["unlisted"]:
        return False, f"paths outside the writable set: {sorted(buckets['unlisted'])}", buckets
    return True, "in scope", buckets


def requires_drill(paths: list[str]) -> bool:
    """True when the diff touches the rollback path and must pass the drill."""
    return bool(classify_all(paths)["protected"])


def touches_requirements(paths: list[str]) -> bool:
    return any(normalize(p) in ("requirements.txt", "requirements.lock") for p in paths)


def validate_code_run_spec(spec: dict) -> str | None:
    """Validate a code round's run_spec. Returns None on success, else a reason.

    Layers over `scripts.autoresearch.common.validate_run_spec`, which only
    hard-checks that `mutation_scope.writable_paths` is a list. For code rounds
    the entries are repo-relative globs rather than the absolute .md paths the
    prompt pipeline produces, so they need their own shape check.
    """
    try:
        from scripts.autoresearch.common import validate_run_spec
    except Exception:  # pragma: no cover - import guard
        validate_run_spec = None  # type: ignore
    if validate_run_spec is not None:
        base = validate_run_spec(spec)
        if base:
            return base

    scope = (spec.get("mutation_scope") or {}).get("writable_paths")
    if not isinstance(scope, list):
        return "mutation_scope.writable_paths must be a list"
    for entry in scope:
        if not isinstance(entry, str):
            return f"writable_paths entry is not a string: {entry!r}"
        if normalize(entry.replace("**", "x")) is None:
            return f"writable_paths entry is not a safe relative path: {entry!r}"

    code = spec.get("code")
    if not isinstance(code, dict):
        return "missing required key: code"
    for key in ("base_commit", "branch"):
        if not code.get(key):
            return f"code.{key} is required"
    return None
