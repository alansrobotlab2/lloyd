"""The `subagents:` profiles in config.yaml (review 2026-09-24, P8).

A profile's prompt and its deny list are two statements about one toolbox, and
they had drifted: `read-only` told its child "You may use Read, Grep, Glob, and
Bash" while denying it Bash, so the child's first move was a refused call.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _profiles() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return cfg.get("subagents") or {}


def test_no_profile_prompt_names_a_tool_it_disallows():
    profiles = _profiles()
    assert profiles, "no subagent profiles in config.yaml"
    for name, prof in profiles.items():
        prompt = str(prof.get("system_prompt") or "")
        for tool in prof.get("disallowed_tools") or []:
            bare = str(tool).rsplit("__", 1)[-1]
            assert not re.search(rf"\b{re.escape(bare)}\b", prompt), (
                f"subagents.{name}.system_prompt names {bare!r}, which the "
                f"profile disallows")


def test_the_read_only_profile_is_parallel_safe_and_the_default_is_not():
    profiles = _profiles()
    assert profiles["read-only"].get("parallel_safe") is True
    assert not profiles["general-purpose"].get("parallel_safe")


def test_the_loop_and_the_child_read_the_same_parallel_safe_set(monkeypatch):
    from agent_mcp import builtin_task
    from app import mcp_discovery as D
    from app.config import CONFIG

    monkeypatch.setitem(CONFIG, "subagents", {
        "a": {"parallel_safe": True}, "b": {"parallel_safe": "yes"}, "c": {}})
    assert D.parallel_safe_task_profiles() == frozenset({"a"})
    assert D._get_harness_kwargs()["parallel_safe_task_profiles"] == frozenset({"a"})
    assert builtin_task._load_subagent_profile("a")["parallel_safe"] is True
    assert builtin_task._load_subagent_profile("b")["parallel_safe"] is False
    assert builtin_task._load_subagent_profile("c")["parallel_safe"] is False
