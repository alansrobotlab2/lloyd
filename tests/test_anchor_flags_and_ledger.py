"""The budget and todo anchors each have an off switch (#769, clauses 1-2).

They are the injected text that fires most often on a long turn, and until
#769 neither could be turned off to measure what it is worth: the context
anchor had `harness.context_anchor.enabled`, the budget anchor had no switch
at all (only `max_turns <= 0`, which removes the cap itself), and the todo
anchor only the de-facto `todo_anchor_interval_iterations: 0`. The flags are
read at the emission decision, and a key that is absent keeps today's
behaviour. config.yaml has carried both keys, set `enabled: true`, since
2026-10-20 (#3050) — the absent-key tests below therefore build a private
`harness` copy with the keys stripped, which is what the code faces on a
config that predates them, and `test_config_yaml_carries_both_anchor_flags_*`
grades the shipped file itself.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
import yaml

from app import deadline_anchor
from app.config import CONFIG
from app.routers import messages as M


TODOS = [
    {"content": "Read the code", "status": "in_progress"},
    {"content": "Write the review", "status": "pending"},
]


def _texts(out) -> str:
    return "\n".join(str(m.get("content") or "") for m in out)


@pytest.fixture
def harness_cfg(monkeypatch):
    """A private copy of `harness` with neither flag present."""
    cfg = {k: v for k, v in (CONFIG.get("harness") or {}).items()
           if k not in ("budget_anchor", "todo_anchor")}
    monkeypatch.setitem(CONFIG, "harness", cfg)
    return cfg


@pytest.fixture
def clock(monkeypatch):
    state = {"t": 1000.0}
    monkeypatch.setattr(deadline_anchor.time, "monotonic", lambda: state["t"])
    return state


def _budget_run(clock, *, max_turns=20, timeout=1000.0):
    """Walk a chat-path anchor past 75%/90% of its cap and 70%/90% of its clock."""
    anchor = M._build_state_anchor("flag-test", max_turns=max_turns,
                                   deadline_seconds=timeout)
    out = []
    for i in range(1, max_turns + 1):
        clock["t"] += timeout / max_turns   # 5% of the wall clock per iteration
        out.extend(asyncio.run(anchor(i)))
    return out


def test_budget_anchors_fire_when_the_key_is_absent(harness_cfg, clock, monkeypatch):
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    text = _texts(_budget_run(clock))
    assert "Iteration 15 of 20" in text and "Iteration 18 of 20" in text
    assert text.count("s of this turn's 1000s budget remain") == 2


def test_budget_flag_false_silences_both_budget_clocks(harness_cfg, clock, monkeypatch):
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    harness_cfg["budget_anchor"] = {"enabled": False}
    out = _budget_run(clock)
    assert "<budget>" not in _texts(out), out


def test_budget_flag_false_silences_the_autonomy_task_anchor(harness_cfg, clock):
    from app import autonomy

    harness_cfg["budget_anchor"] = {"enabled": False}
    anchor = autonomy._build_task_anchor(1000, 20)
    out = []
    for i in range(1, 21):
        clock["t"] += 50
        out.extend(asyncio.run(anchor(i)))
    assert out == []


def test_budget_flag_true_is_the_same_as_absent(harness_cfg, clock, monkeypatch):
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    harness_cfg["budget_anchor"] = {"enabled": True}
    assert _texts(_budget_run(clock)).count("<budget>") == 4


def _todo_run(harness_cfg, monkeypatch):
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: list(TODOS))
    harness_cfg["todo_anchor_interval_iterations"] = 5
    anchor = M._build_state_anchor("flag-test")
    return [asyncio.run(anchor(i)) for i in range(1, 13)]


def test_todo_anchor_fires_when_the_key_is_absent(harness_cfg, monkeypatch):
    runs = _todo_run(harness_cfg, monkeypatch)
    assert any("<active_todos>" in _texts(out) for out in runs)


def test_todo_flag_false_silences_the_reminder(harness_cfg, monkeypatch):
    harness_cfg["todo_anchor"] = {"enabled": False}
    runs = _todo_run(harness_cfg, monkeypatch)
    assert not any("<active_todos>" in _texts(out) for out in runs)


# ── #3050: the two flags are materialised in config.yaml ────────────────────
#
# Everything above reads a `harness` dict the test builds, so none of it could
# notice whether the shipped file carries the keys at all — which is exactly
# the gap #3050 was filed for ("the off switches exist in code and are
# invisible in config"). These tests read `config.yaml` from the tree, the way
# `app.config` reads it at boot, and pin three things a later edit can quietly
# break: the keys are there and on, they are where the loader looks for them,
# and their comments still say what they silence and how much of the ledger
# that is.

ROOT = Path(__file__).resolve().parent.parent
CONFIG_TEXT = (ROOT / "config.yaml").read_text(encoding="utf-8")
CONFIG_YAML = yaml.safe_load(CONFIG_TEXT)

#: The two switches #769 made readable and #3050 wrote into config.yaml.
ANCHOR_FLAGS = ("budget_anchor", "todo_anchor")

#: The modules that gate an anchor on one of those names. `autonomy` composes
#: the builders defined in `deadline_anchor`, so the flag is read there.
FLAG_READERS = ("app/deadline_anchor.py", "app/routers/messages.py")

#: The 2026-10-20 tally of `harness.anchor_fired` events under
#: `~/lloyd-data/event_logs/` — re-measure with
#: `grep -h -- '"harness.anchor_fired"' ~/lloyd-data/event_logs/*.events.jsonl`
#: piped to `grep -oE '"anchor": "[a-z_]+"' | sort | uniq -c`. Both config.yaml
#: comments and `architecture/harness.md` quote these figures as a dated
#: measurement, so re-measuring means editing these constants and that prose
#: together; `test_the_shares_in_the_comments_match_the_counts_they_cite` is
#: what fails when one half moves alone.
LEDGER_TALLY_DATE = "2026-10-20"
LEDGER_TOTAL = 15_552
DEADLINE_BUDGET_FIRES = 579
ITERATION_BUDGET_FIRES = 1_259
TODO_FIRES = 377
CONTEXT_FIRES = 138
SCRATCHPAD_FIRES = 6
GOAL_CARD_FIRES = 13_193

#: The key that gates the loud anchor instead: `_observer_cfg()` reads
#: `inner_voice.observer`, and `goal_card_anchor` asks it.
GOAL_CARD_GATE_PATH = ("inner_voice", "observer", "goal_card_to_primary")


def _pct(count: int) -> float:
    return count / LEDGER_TOTAL * 100


def _key_line(rel_key: str) -> int:
    """The 1-based-free index of the two-space-indented `rel_key:` line."""
    lines = CONFIG_TEXT.splitlines()
    hits = [i for i, ln in enumerate(lines) if ln == f"  {rel_key}:"]
    assert len(hits) == 1, (
        f"`  {rel_key}:` matched {len(hits)} lines in config.yaml; the flag this "
        "file pins must be uniquely identifiable before its comment is read")
    return hits[0]


def _top_level_block(name: str) -> list[str]:
    """The lines of the column-0 `name:` block, up to the next column-0 key."""
    lines = CONFIG_TEXT.splitlines()
    top = [i for i, ln in enumerate(lines) if ln == f"{name}:"]
    assert len(top) == 1, f"`{name}:` matched {len(top)} lines in config.yaml"
    start = top[0]
    end = next((i for i in range(start + 1, len(lines))
                if lines[i] and not lines[i][0].isspace()
                and not lines[i].lstrip().startswith("#")), len(lines))
    return lines[start:end]


def _comment_block_above(key: str) -> str:
    """The `#` lines directly above `  key:`, flattened to one line."""
    lines = CONFIG_TEXT.splitlines()
    i = _key_line(key)
    block: list[str] = []
    while i > 0 and lines[i - 1].lstrip().startswith("#"):
        block.append(lines[i - 1].lstrip().lstrip("#").strip())
        i -= 1
    assert block, f"no comment block above `  {key}:` in config.yaml"
    return " ".join(" ".join(reversed(block)).split())


def test_config_yaml_carries_both_anchor_flags_set_enabled():
    """#3050 clause 1: both keys are in the shipped file, both on."""
    harness = CONFIG_YAML.get("harness") or {}
    for key in ANCHOR_FLAGS:
        assert isinstance(harness.get(key), dict), (
            f"`harness.{key}:` is not a mapping in config.yaml — the flag is "
            "still only in code, which is the gap #3050 was filed for")
        assert harness[key].get("enabled") is True, (
            f"`harness.{key}.enabled` is {harness[key].get('enabled')!r}, not true")
    # The shape the item's acceptance grep asks for: two two-space-indented
    # keys, and the BRE spelling the item first shipped with is a trap —
    # `grep -c 'budget_anchor|todo_anchor'` matches a literal pipe, so it
    # reads 0 both before and after the change.
    assert len(re.findall(r"(?m)^  (?:budget_anchor|todo_anchor):$",
                          CONFIG_TEXT)) == 2
    # And `app.config`, the loader the anchors actually read, has them.
    loaded = CONFIG.get("harness") or {}
    for key in ANCHOR_FLAGS:
        assert isinstance(loaded.get(key), dict), (
            f"app.config loaded no `harness.{key}` block from config.yaml")
        assert loaded[key].get("enabled"), (
            f"app.config loaded `harness.{key}.enabled` off, so booting this "
            "config would silence the anchor the file says is on")


def test_both_anchor_flag_keys_sit_inside_the_harness_block():
    """Clause 1's `inside the harness: block`, checked structurally.

    Indentation alone is not enough: a `  budget_anchor:` under some other
    top-level key would satisfy the grep and load as nothing, because
    `anchor_enabled` reads `CONFIG['harness'][name]`.
    """
    harness_lines = _top_level_block("harness")
    for key in ANCHOR_FLAGS:
        assert f"  {key}:" in harness_lines, (
            f"`  {key}:` is not inside the `harness:` block, so `anchor_enabled` "
            f"would read it as absent and default it on")
        outside = [ln for ln in CONFIG_TEXT.splitlines()
                   if ln == f"  {key}:" and ln not in harness_lines]
        assert not outside, f"`  {key}:` also appears outside `harness:`"


def test_the_config_key_names_are_the_names_the_code_looks_up():
    """The seam between the file and the reader, both directions.

    `anchor_enabled(name)` is called with a literal, so a rename on either
    side leaves a flag that reads as absent (silently defaulting to on) or a
    config key no code reads. Neither is visible until an ablation is run and
    does nothing.
    """
    looked_up: set[str] = set()
    for rel in FLAG_READERS:
        looked_up |= set(re.findall(
            r'anchor_enabled\(\s*"([a-z_]+)"', (ROOT / rel).read_text("utf-8")))
    assert looked_up == set(ANCHOR_FLAGS), (
        f"the code gates on {sorted(looked_up)} but config.yaml is expected to "
        f"carry {sorted(ANCHOR_FLAGS)}")
    harness = CONFIG_YAML.get("harness") or {}
    assert looked_up <= set(harness), (
        f"{sorted(looked_up - set(harness))} is read by the code but absent from "
        "config.yaml, so the switch cannot be flipped from the file")


def test_the_budget_anchor_comment_names_both_kinds_and_their_levels():
    """Clause 3: the comment says which `ANCHOR_TAG` kinds the flag silences."""
    comment = _comment_block_above("budget_anchor")
    for want in ("deadline_budget", "iteration_budget", "70%", "90%", "75%",
                 f"{DEADLINE_BUDGET_FIRES + ITERATION_BUDGET_FIRES:,}"):
        assert want in comment, (
            f"the `budget_anchor:` comment no longer names {want!r}: {comment}")
    assert "`<budget>`" in comment, (
        "the comment does not say which injected block the flag silences")


def test_the_todo_anchor_comment_names_the_active_todos_reanchor():
    """Clause 3, todo half: the comment names the re-anchor it silences."""
    comment = _comment_block_above("todo_anchor")
    for want in ("<active_todos>", "kind `todo`",
                 "`todo_anchor_interval_iterations`", f"{TODO_FIRES} fires"):
        assert want in comment, (
            f"the `todo_anchor:` comment no longer names {want!r}: {comment}")


def test_the_kinds_named_in_the_comments_are_the_ones_the_code_emits():
    """A comment that names a dead kind is a comment nobody can act on.

    The `ANCHOR_TAG` kinds are what `loop.py` writes to the ledger, so these
    are read out of the emitting modules rather than restated: rename a kind
    in the code and this fails beside the config comment.
    """
    budget_src = (ROOT / "app/deadline_anchor.py").read_text("utf-8")
    emitted_budget = set(re.findall(r'"kind":\s*"([a-z_]+)"', budget_src))
    assert emitted_budget == {"deadline_budget", "iteration_budget"}, emitted_budget
    budget_comment = _comment_block_above("budget_anchor")
    named_budget = {k for k in emitted_budget if k in budget_comment}
    assert named_budget == emitted_budget, (
        f"the `budget_anchor:` comment covers {sorted(named_budget)} of the "
        f"{sorted(emitted_budget)} kinds `deadline_anchor.py` emits under it")
    todo_src = (ROOT / "app/routers/messages.py").read_text("utf-8")
    assert '"kind": "todo"' in todo_src, (
        "messages.py no longer tags the re-anchor `todo`, so the comment and the "
        "ledger disagree about what `todo_anchor.enabled` silences")


def test_both_comments_date_their_share_and_name_the_loud_anchor():
    """Clause 4: neither comment may read as if it holds the 85%."""
    for key in ANCHOR_FLAGS:
        comment = _comment_block_above(key)
        assert LEDGER_TALLY_DATE in comment, (
            f"the `{key}:` comment states a share with no date; a growing "
            "denominator makes an undated percentage a lie within a month")
        assert "about 85%" in comment, (
            f"`{key}:` does not say how much of the ledger is NOT its switch")
        assert "goal_card" in comment, (
            f"`{key}:` does not name the anchor that dominates the ledger")
        assert "inner_voice.observer.goal_card_to_primary" in comment, (
            f"`{key}:` does not name the key that actually gates `goal_card`")
        assert "todo_anchor_interval_iterations" in comment, (
            f"`{key}:` does not say that `goal_card` rides the same cadence")
    gate = CONFIG
    for part in GOAL_CARD_GATE_PATH:
        assert isinstance(gate, dict) and part in gate, (
            f"config.yaml no longer carries `{'.'.join(GOAL_CARD_GATE_PATH)}`, "
            "which both comments name as the goal card's switch")
        gate = gate[part]


def test_the_shares_in_the_comments_match_the_counts_they_cite():
    """The percentages beside the config comments are arithmetic, not vibes.

    Each figure is pinned twice: the count against the total it was tallied
    from, and the rounded percent the comment actually prints. Editing one
    count without the other fails here rather than shipping a comment whose
    share no longer follows from its own numbers.
    """
    assert (DEADLINE_BUDGET_FIRES + ITERATION_BUDGET_FIRES + TODO_FIRES
            + CONTEXT_FIRES + SCRATCHPAD_FIRES + GOAL_CARD_FIRES) == LEDGER_TOTAL, (
        "the per-kind counts no longer add up to the tally total they cite")
    budget_pct = round(_pct(DEADLINE_BUDGET_FIRES + ITERATION_BUDGET_FIRES))
    todo_pct = round(_pct(TODO_FIRES))
    paired_pct = round(_pct(DEADLINE_BUDGET_FIRES + ITERATION_BUDGET_FIRES
                            + TODO_FIRES))
    goal_pct = round(_pct(GOAL_CARD_FIRES))
    assert budget_pct == 12 and todo_pct == 2, (budget_pct, todo_pct)
    assert paired_pct == 14 and goal_pct == 85, (paired_pct, goal_pct)
    budget_comment = _comment_block_above("budget_anchor")
    todo_comment = _comment_block_above("todo_anchor")
    assert f"about {budget_pct}%" in budget_comment
    assert f"about {paired_pct}%" in budget_comment
    assert f"about {goal_pct}%" in budget_comment
    assert f"about {todo_pct}%" in todo_comment
    assert f"about {goal_pct}%" in todo_comment
    assert f"{LEDGER_TOTAL:,}" in budget_comment, (
        "the budget comment no longer states the total its share is of")
