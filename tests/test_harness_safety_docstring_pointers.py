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
