"""The Bash modules name the gate that exists (#695, merged finding).

`builtin_bash` and `builtin_fs` said Bash denial ran in
`app/inner_voice/heuristics.py` under `inner_voice.pretooluse_deny` — a module
deleted in the v3 rewrite and a key nothing reads. The docstring is the stated
boundary between "guarded" and "standalone, unguarded", so a dead pointer there
reads as a guard being present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_mcp import builtin_bash, builtin_fs  # noqa: E402

MODULES = [builtin_bash, builtin_fs]


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_docstring_names_the_live_gate(mod):
    doc = mod.__doc__
    assert "app/harness/safety.py" in doc
    assert "install_default_safety_hook" in doc


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_docstring_names_no_dead_control(mod):
    doc = mod.__doc__
    assert "heuristics.py" not in doc
    assert "pretooluse_deny" not in doc


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_docstring_still_says_standalone_dispatch_is_ungated(mod):
    assert "standalone" in mod.__doc__
    assert "no gate" in " ".join(mod.__doc__.split())


def test_the_named_gate_exists():
    from app.harness import safety
    assert callable(safety.install_default_safety_hook)
    assert (ROOT / "app" / "harness" / "safety.py").is_file()
    assert not (ROOT / "app" / "inner_voice" / "heuristics.py").exists()


# D5 (review 2026-09-24): the gates are registered fail-closed, so a gate that
# raises denies the call instead of letting it through.

def test_the_three_gates_are_registered_fail_closed():
    from app.harness import policy, safety
    from app.harness.hooks import HookRegistry

    reg = HookRegistry()
    safety.install_default_safety_hook(reg)       # safety + outbound content
    policy.install_policy_hook(reg, scope="worker:test")
    names = [cb.__name__ for _m, cb in reg._pre]
    assert names == ["_safety_pretool_cb", "_content_pretool_cb",
                     "_policy_pretool_cb"]
    assert reg._pre_fail_closed == [True, True, True]


def test_a_safety_checker_that_raises_denies_bash_and_logs_once(monkeypatch):
    import asyncio

    from app.harness import safety, telemetry
    from app.harness.hooks import HookRegistry

    seen = []
    monkeypatch.setattr(telemetry, "log_harness_event",
                        lambda sid, event, data, **kw: seen.append((event, data)))

    def _broken(*_a, **_k):
        raise ImportError("checker module failed to load")

    monkeypatch.setattr(safety, "check_bash_command", _broken)
    reg = HookRegistry()
    safety.install_default_safety_hook(reg)
    out = asyncio.run(reg.fire_pre_tool_use(
        session_id="s", tool_name="Bash", tool_input={"command": "ls"}))
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "_safety_pretool_cb" in hso["permissionDecisionReason"]
    assert "ImportError" in hso["permissionDecisionReason"]
    assert [e for e, _d in seen] == ["harness.hook_raised"]
    assert seen[0][1]["fail_closed"] is True
