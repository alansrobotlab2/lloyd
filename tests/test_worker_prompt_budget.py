"""What an unattended turn carries, and how many iterations it gets.

Two independent silent losses, both measured on the 2026-09-11 autocode
rounds:

  * **USER.md.** ~20k tokens describing the person Lloyd works for, on every
    worker turn. A round is judged against an acceptance contract and nobody
    reads its reply, so that is 20k of a 262k window spent on something
    structurally irrelevant — and three rounds that day died at the wall.
  * **The iteration ceiling.** `autocode.DEFAULT_MAX_TURNS` is 150, the queue
    payload carried 150, and `_turn_budget` clamped it to 120 with nothing
    logged. Round 858 died at the cap before it could re-gate.
"""

from __future__ import annotations

import logging

import pytest

import prompt_builder
from app.routers.messages import _turn_budget
from app.sessions_io import NON_USER_PLATFORMS


# ---------------------------------------------------------------------------
# the prompt surface
# ---------------------------------------------------------------------------

def test_a_worker_platform_drops_user_memory():
    assert prompt_builder._memory_files_for("worker") == ("MEMORY.md",)
    assert prompt_builder._memory_files_for("autonomy") == ("MEMORY.md",)


def test_a_user_platform_keeps_everything():
    assert prompt_builder._memory_files_for("") == ("MEMORY.md", "USER.md")
    assert prompt_builder._memory_files_for("mission-control") == (
        "MEMORY.md", "USER.md")
    assert prompt_builder._memory_files_for("voice") == ("MEMORY.md", "USER.md")


def test_every_non_user_platform_is_covered():
    """Keyed on `sessions_io.NON_USER_PLATFORMS`, not on a private list —
    that is the one definition of "nobody is reading this", and a seventh
    reader with its own copy is how this class of bug comes back.
    """
    for platform in NON_USER_PLATFORMS:
        assert "USER.md" not in prompt_builder._memory_files_for(platform), platform


def test_the_drop_list_is_configurable(monkeypatch):
    from app.config import CONFIG

    harness = dict(CONFIG.get("harness") or {})
    harness["worker_prompt"] = {"drop_memory_files": []}
    monkeypatch.setitem(CONFIG, "harness", harness)
    assert prompt_builder._memory_files_for("worker") == ("MEMORY.md", "USER.md")


def test_an_unreadable_config_keeps_the_full_set(monkeypatch):
    """Fails open in both directions that matter.

    Dropping USER.md from a chat turn is a visible regression; carrying it
    on a worker turn is only a cost. So a config lookup that raises must
    return the full set, not the reduced one.
    """
    class _Exploding(dict):
        def get(self, *_a, **_k):
            raise RuntimeError("config unreadable")

    monkeypatch.setattr("app.config.CONFIG", _Exploding())
    assert prompt_builder._memory_files_for("worker") == ("MEMORY.md", "USER.md")


def test_an_unimportable_sessions_io_keeps_the_full_set(monkeypatch):
    """A bench or eval script imports `prompt_builder` with no app bootstrap.
    It must still build a prompt rather than raise, and the prompt it builds
    is the user one — nothing has told it otherwise.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "app.sessions_io":
            raise ImportError("no app bootstrap")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert prompt_builder._memory_files_for("worker") == ("MEMORY.md", "USER.md")


def test_load_memories_honours_the_file_list(tmp_path):
    (tmp_path / "MEMORY.md").write_text("mem body")
    (tmp_path / "USER.md").write_text("user body")

    both = prompt_builder._load_memories(tmp_path)
    assert "mem body" in both and "user body" in both

    only = prompt_builder._load_memories(tmp_path, files=("MEMORY.md",))
    assert "mem body" in only
    assert "user body" not in only


def test_the_worker_prompt_is_measurably_smaller(tmp_path):
    """The point of the change is tokens, so assert tokens."""
    (tmp_path / "MEMORY.md").write_text("m" * 1_000)
    (tmp_path / "USER.md").write_text("u" * 80_000)

    both = prompt_builder._load_memories(tmp_path) or ""
    worker = prompt_builder._load_memories(tmp_path, files=("MEMORY.md",)) or ""
    assert len(both) - len(worker) > 75_000


def test_build_system_prompt_accepts_and_uses_platform(monkeypatch):
    seen: list[tuple] = []

    def _fake(overlay=None, *, soul=None, files=("MEMORY.md", "USER.md")):
        seen.append(files)
        return "MEM"
    monkeypatch.setattr(prompt_builder, "_load_memories", _fake)
    prompt_builder.build_system_prompt(include_skills_index=False, platform="worker")
    assert seen and "USER.md" not in seen[-1]

    seen.clear()
    prompt_builder.build_system_prompt(include_skills_index=False)
    assert seen and "USER.md" in seen[-1]


def test_the_prompt_budget_line_names_the_platform(caplog):
    with caplog.at_level(logging.INFO, logger="lloyd.prompt"):
        prompt_builder.log_prompt_size({"soul": "x" * 100}, session_id="s1",
                                       platform="worker")
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "platform=worker" in line


def test_the_budget_line_reports_the_platform_not_a_prompt_component(caplog):
    """A local named `platform` — the "Platform: Lloyd (Claude Agent SDK)…"
    hints block — shadowed the parameter and printed itself into the log line
    as though it were the session's platform.

    The drop itself was unaffected, because `_memory_files_for` is called
    above the reassignment. That is what makes this worth a test rather than
    a fix: the behaviour depended on statement order that nothing named.
    """
    with caplog.at_level(logging.INFO, logger="lloyd.prompt"):
        prompt_builder.build_system_prompt(include_skills_index=False,
                                           session_id="s1", platform="worker")
    line = next(r.getMessage() for r in caplog.records
                if "PROMPT_BUDGET" in r.getMessage())
    assert "platform=worker" in line
    assert "Claude Agent SDK" not in line


def test_the_prompt_budget_line_says_user_when_no_platform(caplog):
    with caplog.at_level(logging.INFO, logger="lloyd.prompt"):
        prompt_builder.log_prompt_size({"soul": "x" * 100}, session_id="s1")
    assert "platform=user" in "\n".join(r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# the iteration ceiling
# ---------------------------------------------------------------------------

def test_a_worker_keeps_the_budget_it_asked_for():
    """Round 858's exact loss: 150 requested, 120 delivered, silently."""
    assert _turn_budget({"max_turns": 150}, platform="worker") == 150
    assert _turn_budget({"max_turns": 150}, platform="autonomy") == 150


def test_a_user_turn_keeps_the_old_ceiling():
    assert _turn_budget({"max_turns": 150}, platform="mission-control") == 120
    assert _turn_budget({"max_turns": 150}) == 120


def test_the_worker_ceiling_still_bounds():
    assert _turn_budget({"max_turns": 9_999}, platform="worker") == 200


def test_no_request_gets_the_default_on_every_platform():
    assert _turn_budget({}, platform="worker") == 60
    assert _turn_budget({}) == 60


def test_a_clamp_is_logged(caplog):
    """Silence is what made this cost a round: a budget smaller than the one
    asked for reads, from the outside, as the model giving up early.
    """
    with caplog.at_level(logging.WARNING, logger="app.routers.messages"):
        _turn_budget({"max_turns": 500}, platform="mission-control")
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "clamped" in joined
    assert "500" in joined


def test_no_clamp_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="app.routers.messages"):
        _turn_budget({"max_turns": 150}, platform="worker")
    assert not [r for r in caplog.records if "clamped" in r.getMessage()]


def test_the_default_budget_fits_under_the_worker_ceiling():
    """`autocode.DEFAULT_MAX_TURNS` past the ceiling is a wish, not a budget."""
    from app.config import CONFIG
    from workers.sources.autocode import DEFAULT_MAX_TURNS

    ceiling = int((CONFIG.get("agent") or {}).get("max_turns_ceiling_worker", 200))
    assert DEFAULT_MAX_TURNS <= ceiling, (
        f"autocode asks for {DEFAULT_MAX_TURNS} and would be clamped to {ceiling}")


def test_the_missing_key_falls_back_rather_than_raising(monkeypatch):
    from app.config import CONFIG

    agent = {k: v for k, v in (CONFIG.get("agent") or {}).items()
             if k != "max_turns_ceiling_worker"}
    monkeypatch.setitem(CONFIG, "agent", agent)
    assert _turn_budget({"max_turns": 9_999}, platform="worker") == 200
