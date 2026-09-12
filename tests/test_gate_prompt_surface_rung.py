"""The prompt-surface eval, as a gate rung rather than as four prompt lines.

A behavioural regression in the prompt surface passes every other rung: the
tests are green, tsc is clean, the canary boots, and the model has quietly
stopped reaching for `http_search`. So the check has to exist — but it used
to live in the autocode prompt, which placed it exactly wrong. The model ran
it from inside its own turn, against the live engine, while its own
150k-token round was the other tenant: 20 primary queries beside the round's
own iterations, twice, evicting the round's prefix both times.

As a rung it runs while the model is idle in `automod_gate_wait` — the one
window in a round when nothing else of its own is on the engine.
"""

from __future__ import annotations

import types

import pytest

from scripts.automod import gate as G


class _Report:
    def __init__(self, changed):
        self.changed_paths = list(changed)


def _gate(changed, item_id=None):
    g = G.Gate.__new__(G.Gate)
    g.report = _Report(changed)
    g.round_id = "SM_20260912_120000"
    g.item_id = item_id
    g.live = G.LIVE_ROOT
    g.python = G.LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    g._child_env = lambda: {}
    return g


@pytest.mark.parametrize("path", [
    "prompt_builder.py",
    "prefetch.py",
    "lloyd/SOUL.md",
    "lloyd/MEMORY.md",
    "lloyd/USER.md",
])
def test_the_rung_fires_on_a_prompt_surface_path(path):
    assert _gate([path])._touches_prompt_surface() is True


@pytest.mark.parametrize("path", [
    "app/harness/loop.py",
    "workers/pool.py",
    "web/src/api.ts",
    "tests/test_prompt_builder_overlay.py",
    "docs/prompt_builder.md",
])
def test_the_rung_does_not_fire_on_an_unrelated_path(path):
    assert _gate([path])._touches_prompt_surface() is False


def test_an_untouched_surface_records_a_skip_not_a_pass_by_silence():
    """A skip is a recorded rung that says so — the same rule the drill rung
    already follows, and for the same reason: a report listing eight rungs
    when nine exist makes a reader infer the ninth from its absence.
    """
    ok, msg, data = _gate(["app/harness/loop.py"]).rung_prompt_surface()
    assert ok is True
    assert data.get("skipped") is True
    assert "not touched" in data.get("reason", "")


def test_a_regression_fails_the_rung(monkeypatch):
    calls = []

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        rc = 0 if "run_tool_choice_eval.py" in " ".join(map(str, cmd)) else 1
        return types.SimpleNamespace(
            returncode=rc, stdout="tool_choice regression: http_search 0.9 -> 0.4",
            stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)

    ok, msg, data = _gate(["prompt_builder.py"]).rung_prompt_surface()
    assert ok is False
    assert data["compare_exit"] == 1
    assert "regression" in msg


def test_exit_2_is_not_a_pass(monkeypatch):
    """`compare_tool_choice.py` exits 2 when it had nothing to compare
    against. That is "unmeasured", and an auto-landing loop must not read
    unmeasured as safe.
    """
    def _fake_run(cmd, **kw):
        rc = 0 if "run_tool_choice_eval.py" in " ".join(map(str, cmd)) else 2
        return types.SimpleNamespace(returncode=rc, stdout="no prior run",
                                     stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)

    ok, msg, _ = _gate(["prefetch.py"]).rung_prompt_surface()
    assert ok is False
    assert "nothing to compare" in msg


def test_an_unknown_nonzero_exit_also_fails(monkeypatch):
    """The rung reads whatever the live script returns rather than a copy of
    its contract — #875's kept branch adds an exit 3.
    """
    def _fake_run(cmd, **kw):
        rc = 0 if "run_tool_choice_eval.py" in " ".join(map(str, cmd)) else 3
        return types.SimpleNamespace(returncode=rc, stdout="new failure mode",
                                     stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)
    ok, _, data = _gate(["prompt_builder.py"]).rung_prompt_surface()
    assert ok is False
    assert data["compare_exit"] == 3


def test_a_clean_comparison_passes(monkeypatch):
    def _fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout="no regression vs item9",
                                     stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)
    ok, msg, data = _gate(["prompt_builder.py"]).rung_prompt_surface()
    assert ok is True
    assert data["compare_exit"] == 0
    assert "no regression" in msg


def test_the_eval_runs_from_the_live_tree_not_the_worktree(monkeypatch):
    """The eval writes its baseline next to the script, and a baseline written
    inside a worktree is deleted with it — SM_20260908_165950's was.
    """
    cwds = []

    def _fake_run(cmd, **kw):
        cwds.append(kw.get("cwd"))
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)
    _gate(["prompt_builder.py"]).rung_prompt_surface()
    assert cwds and all(c == G.LIVE_ROOT for c in cwds), cwds


def test_the_label_names_the_item_when_there_is_one(monkeypatch):
    labels = []

    def _fake_run(cmd, **kw):
        parts = [str(c) for c in cmd]
        if "--label" in parts:
            labels.append(parts[parts.index("--label") + 1])
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)

    _gate(["prompt_builder.py"], item_id=377).rung_prompt_surface()
    assert labels and all(x == "item377" for x in labels)

    labels.clear()
    _gate(["prompt_builder.py"]).rung_prompt_surface()
    assert labels and all(x.startswith("gate-SM_") for x in labels)


def test_the_rung_is_in_the_ladder_between_tests_and_review():
    """Order is load-bearing: after `tests`, so a broken tree fails first and
    cheaply; before `review`, because a behavioural regression is a fact the
    reviewer should be able to see.
    """
    import inspect

    src = inspect.getsource(G.Gate.run)
    i_tests = src.index('("tests"')
    i_ps = src.index('("prompt_surface"')
    i_review = src.index('("review"')
    assert i_tests < i_ps < i_review


def test_the_autocode_prompt_no_longer_asks_the_model_to_run_it():
    """20 primary queries from inside the round's own turn, twice."""
    from workers.sources.autocode import PROMPT

    assert "run_tool_choice_eval.py" not in PROMPT
    assert "compare_tool_choice.py" not in PROMPT
    # ...and it says who does run it, so the model does not simply assume
    # the check stopped existing.
    assert "prompt_surface" in PROMPT
