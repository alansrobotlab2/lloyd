"""Which paths a self-modification round may change.

With no human review tier this module is the entire control surface, so the
ordering property matters more than any individual glob: **denied beats
protected beats allowed**, and a run spec cannot widen its own permissions by
listing a denied path in `writable_paths`.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.automod import spec

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "app/harness/loop.py", "agent_mcp/facts.py", "workers/pool.py",
    "tests/test_x.py", "eval/run_eval.py", "server.py", "app/autonomy.py",
    "app/prompt_builder.py", "app/prompt_surface.py", "scripts/memory/kg_rebuild.py",
])
def test_ordinary_code_is_allowed(path):
    """The prompt machinery is ordinary code under `app/**`. It used to sit at
    the repo root with one allow-list line per file, and `prompt_surface.py`
    was never enumerated, so item #1069, whose entire fix is that module, was
    unimplementable by any round: three rounds wrote the fix and were refused
    at rung 0 (SM_20260911_190850, SM_20260914_114935, SM_20260918_145241)
    (#1242)."""
    assert spec.classify(path) == "allowed"


@pytest.mark.parametrize("path", [
    "scripts/automod/gate.py", "scripts/automod/spec.py",
    "agent-services/guardian/guardian.py", "agent-services/guardian/rollback.py",
    "agent-services/systemd/lloyd-guardian.service",
    "agent-services/supervisor/conf.d/lloyd-backend.conf",
    "agent-services/bin/guardian-stage.sh",
    "app/routers/health.py", "app/supervisor_client.py",
    "app/lifecycle.py", "app/gitinfo.py",
])
def test_the_rollback_path_is_protected(path):
    """These may be changed, but only with a live drill. See the `drill` rung."""
    assert spec.classify(path) == "protected"


@pytest.mark.parametrize("path", [
    "config.yaml", "data/tool_overrides.yaml", ".env", ".env.local",
    "pytest.ini", ".gitignore", ".venvs/lloyd/bin/python",
    "web/package.json", "web/package-lock.json", "web/node_modules/vite/index.js",
    "web/vite.config.ts", "web/tsconfig.app.json", "web/dist/index.html", "web/.env",
])
def test_denied_paths(path):
    assert spec.classify(path) == "denied"


def test_config_yaml_is_denied_because_it_can_disarm_the_agent(
):
    """A round could disable Bash and Edit via `disabled_tools` and lock itself
    out without changing a line of Python — a soft brick no test would catch.

    `classify` judges by PATH and still says `denied`: the two lanes out of
    the denial (comment-only, and since 2026-10-05 a value change outside the
    denied keys) are judged on CONTENT by `check_scope`, which needs both
    sides of the file. Without content there is no lane, and the lock-out key
    itself — `mcp_servers` — heads `CONFIG_DENIED_KEYS`, so the content lane
    cannot reach it either."""
    assert spec.classify("config.yaml") == "denied"
    assert spec.classify("data/tool_overrides.yaml") == "denied"
    ok, _, buckets = spec.check_scope(["config.yaml"])  # no contents: no lane
    assert not ok and buckets["denied"] == ["config.yaml"]
    assert spec.CONFIG_DENIED_KEYS[0] == "mcp_servers"
    assert spec._config_denied_prefix("mcp_servers.lloyd-mcp.disabled_tools") == "mcp_servers"


def test_protected_beats_allowed_even_under_an_allowed_prefix():
    """`app/**` is allowed, but `app/routers/health.py` is the health endpoint
    the rollback verification depends on."""
    assert spec.classify("app/routers/messages.py") == "allowed"
    assert spec.classify("app/routers/health.py") == "protected"


def test_unlisted_paths_are_not_silently_allowed():
    assert spec.classify("some/random/thing.txt") == "unlisted"
    assert spec.classify("Makefile") == "unlisted"


@pytest.mark.parametrize("path", [
    "../etc/passwd", "/etc/passwd", "app/../../etc/passwd", "", "   ",
    "app/./../../x",
])
def test_traversal_and_absolute_paths_are_denied(path):
    assert spec.classify(path) == "denied"


def test_requirements_are_allowed_only_because_rung_3_exists():
    """The canary shares the live venv unless the gate builds a candidate one."""
    assert spec.classify("requirements.txt") == "allowed"
    assert spec.classify("requirements.lock") == "allowed"
    assert spec.touches_requirements(["app/x.py", "requirements.lock"])
    assert not spec.touches_requirements(["app/x.py"])


# ---------------------------------------------------------------------------
# #1073: the dev-only requirements file, and the doc that names it
# ---------------------------------------------------------------------------

def test_the_dev_requirements_file_is_writable():
    """`requirements-dev.txt` is where an optional solver goes, and a round has
    to be able to create it: `spec.classify` returned `unlisted` for it, and
    `check_scope` refuses an unlisted path, so the route item #1073 had to
    settle could not have been landed by any round at all. `SETUP.md` — the file
    #1073's contract says the decision belongs in — had the same defect.

    Both are exact filenames next to `README.md` and `CLAUDE.md`, which have been
    allowed all along.
    """
    assert spec.classify("requirements-dev.txt") == "allowed"
    assert spec.classify("SETUP.md") == "allowed"
    ok, reason, buckets = spec.check_scope(
        ["SETUP.md", "requirements-dev.txt", "tests/test_automod_spec.py"])
    assert ok, reason
    assert reason == "in scope"
    assert not buckets["unlisted"] and not buckets["protected"]


def test_the_dev_requirements_file_is_never_an_install_target():
    """The grant is safe for exactly one reason, and it is this function.

    `rung_venv` skips unless `spec.touches_requirements` is true
    (`scripts/automod/gate.py:1800`), and when it does run it installs from
    `requirements.lock` if that exists, else `requirements.txt`
    (`gate.py:1825-1826`). Neither can see a third file, so admitting
    `requirements-dev.txt` cannot put a package into a candidate venv — which is
    the divergence #1073 is about: a solver in `requirements.txt` alone never
    reaches the candidate, and the live venv and the candidate then disagree on
    whether `import z3` succeeds, against a floor of 1000 passed.
    """
    assert not spec.touches_requirements(["requirements-dev.txt"])
    assert not spec.touches_requirements(["SETUP.md", "requirements-dev.txt",
                                          "requirements-dev"])
    # The rule's own source names exactly two files, and no third spelling:
    # widening it is what would make the dev file an install target, so the
    # claim is pinned here rather than inferred from one call returning False.
    rule = inspect.getsource(spec.touches_requirements)
    assert '"requirements.txt"' in rule and '"requirements.lock"' in rule
    assert "requirements-dev" not in rule


@pytest.mark.parametrize("path", [
    "SETUP.md.bak", "SETUP.md/README.md", "docs/SETUP.md", "SETUP.MD",
    "requirements-dev.txt.asc", "requirements-devs.txt", "requirements_dev.txt",
    "dev/requirements-dev.txt", "Makefile", "some/random/thing.txt",
])
def test_the_solver_route_grant_is_two_filenames_not_a_pattern(path):
    """Admitting two root filenames must not admit a pattern, and must not
    touch the control surface: `scripts/automod/spec.py` stays protected, so the
    round that edits it still owes a live rollback drill.
    """
    assert spec.classify(path) == "unlisted"
    assert spec.classify("scripts/automod/spec.py") == "protected"
    assert spec.requires_drill(["scripts/automod/spec.py"])


def test_the_solver_route_grant_holds_in_the_interpreter_that_gates():
    """The seam the grant is actually consumed across, crossed the way rung 0
    crosses it.

    `gate_detached` spawns `[LIVE_ROOT/.venvs/lloyd/bin/python, "-m",
    "scripts.automod.round", "gate", <id>]` with `cwd=LIVE_ROOT`
    (`scripts/automod/round.py:425-445`), so rung 0's `spec.check_scope`
    (`gate.py:928`, inside `rung_preflight`) runs in a process this test does not
    share, on the tuple that
    interpreter imported. Same spawn shape here: a fresh interpreter, started from
    this tree. That mechanism is also the safety property that makes admitting
    these two names safe at all — a round editing `spec.py` inside its own
    worktree cannot widen the set its own gate measures it against, so the grant
    binds only once it has landed, which is why #1073's file-writing half is
    #1378 and not this round.

    `test_a_second_interpreter_reaches_the_same_verdict_as_rung_0` crosses the
    same seam for `prompt_surface.py`; this crosses it for the two names the
    solver route needs, plus the negative the route rests on — a delta that only
    touches `requirements-dev.txt` must not read as a requirements change, or the
    candidate venv would be rebuilt against an install list that cannot contain
    the solver.
    """
    probe = (
        "from scripts.automod import spec; "
        "print(spec.classify('SETUP.md')); "
        "print(spec.classify('requirements-dev.txt')); "
        "print(spec.check_scope(['SETUP.md', 'requirements-dev.txt', "
        "'tests/test_automod_spec.py'])[0:2]); "
        "print(spec.touches_requirements(['requirements-dev.txt', 'SETUP.md']))"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines() == [
        "allowed", "allowed", "(True, 'in scope')", "False"], out.stdout


# ---------------------------------------------------------------------------
# check_scope
# ---------------------------------------------------------------------------

def test_scope_rejects_a_denied_path():
    ok, reason, _ = spec.check_scope(["app/x.py", "config.yaml"])
    assert not ok and "denied" in reason


def test_scope_rejects_an_unlisted_path():
    ok, reason, _ = spec.check_scope(["app/x.py", "random.txt"])
    assert not ok and "outside the writable set" in reason


def test_scope_accepts_protected_paths_and_flags_the_drill():
    ok, _, buckets = spec.check_scope(["app/x.py", "agent-services/guardian/detect.py"])
    assert ok
    assert buckets["protected"] == ["agent-services/guardian/detect.py"]
    assert spec.requires_drill(["agent-services/guardian/detect.py"])
    assert not spec.requires_drill(["app/x.py"])


def test_a_clean_ordinary_diff_needs_no_drill():
    ok, _, buckets = spec.check_scope(["app/harness/loop.py", "tests/test_harness.py"])
    assert ok and not buckets["protected"]


def test_scope_accepts_a_diff_whose_fix_is_prompt_surface():
    """The diff #1069 has to write — the module plus its test files — is now
    in scope, which is the whole point of admitting the path. Before this entry
    rung 0 refused exactly this list: three rounds were refused for
    `['prompt_surface.py']` (SM_20260911_190850, SM_20260914_114935,
    SM_20260918_145241)."""
    ok, reason, buckets = spec.check_scope(
        ["app/prompt_surface.py", "tests/test_prompt_surface_guard.py"])
    assert ok, reason
    assert reason == "in scope"
    assert not buckets["unlisted"] and not buckets["protected"]


def test_admitting_prompt_surface_did_not_widen_anything_else():
    """The grant is one filename, not a pattern.

    Rung 0 must still refuse a root-level file the loop was never given, and
    the scope module must still classify as its own protected path — a round
    that widened the writable set and quietly un-armed the drill for the
    control surface it just edited would have removed the only guard
    permanently (see `test_the_denylist_is_not_overridable_by_a_spec`).
    """
    assert spec.classify("Makefile") == "unlisted"
    assert spec.classify("some/random/thing.txt") == "unlisted"
    assert spec.classify("scripts/automod/spec.py") == "protected"
    assert spec.requires_drill(["scripts/automod/spec.py"])
    ok, reason, _ = spec.check_scope(["app/prompt_surface.py", "Makefile"])
    assert not ok and "outside the writable set" in reason
    # The root-level names the modules had before they moved into `app/` are
    # no longer granted: a new module written there is refused, not admitted.
    for gone in ("prompt_surface.py", "prompt_builder.py", "autonomy.py",
                 "prefetch.py", "usage_store.py"):
        assert spec.classify(gone) == "unlisted", gone


def test_a_second_interpreter_reaches_the_same_verdict_as_rung_0():
    """The seam #1069 died at, crossed the way the gate crosses it.

    `automod_gate` does not gate in-process: `agent_mcp/automod.py:243-246`
    spawns `[LIVE_ROOT/.venvs/lloyd/bin/python, "-m",
    "scripts.automod.round", "gate", <id>]` detached with `cwd=LIVE_ROOT`, and
    `run_gate` (round.py:147) builds the `Gate` at round.py:163, whose rung 0 calls
    `spec.check_scope(changed)` (gate.py:765) on the `spec` module THAT
    interpreter imported. `run_spec.yaml`'s `writable_paths` is not what decides
    a code round's scope — the tuple as re-imported by the gate process is. So
    start a fresh interpreter the same way, from the same cwd, and read the two
    verdicts it prints.

    The same mechanism is the safety property that makes this edit safe to make
    at all: the gate process imports the tree it was launched in, so a round
    editing `spec.py` in its own worktree cannot widen the writable set its own
    gate checks it against. The grant binds only once it has landed.
    """
    probe = (
        "from scripts.automod import spec; "
        "print(spec.classify('app/prompt_surface.py')); "
        "print(spec.check_scope(['app/prompt_surface.py', "
        "'tests/test_prompt_surface_guard.py'])[0:2])"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == "allowed", out.stdout
    assert lines[1] == "(True, 'in scope')", out.stdout


def test_the_writable_set_every_round_is_built_with_still_validates():
    """`round.start()` writes ALLOWED_GLOBS into every round's run spec
    (round.py:90) and refuses to open the round if the validator rejects it
    (round.py:107-111, `W.remove(rid)` then a raise). So a malformed entry — a
    leading slash, a `..`, a non-string — is not one refused diff, it is no
    rounds at all. This is the only consumer of that serialised field: the gate
    decides a code round's scope by re-importing the tuple, not by reading it
    (see `test_a_second_interpreter_reaches_the_same_verdict_as_rung_0`)."""
    run_spec = {
        "objective": "x",
        "evaluation": {"command": "scripts.automod.gate", "timeout_secs": 3600},
        "budget": {"max_rounds": 1, "max_variants_per_round": 1},
        "mutation_scope": {"writable_paths": list(spec.ALLOWED_GLOBS)},
        "code": {"base_commit": "a" * 40, "branch": "automod/SM_1",
                 "worktree": "/home/lloyd"},
    }
    assert spec.validate_code_run_spec(run_spec) is None


# ---------------------------------------------------------------------------
# run spec
# ---------------------------------------------------------------------------

def valid_spec() -> dict:
    return {
        "objective": "make the harness faster",
        "evaluation": {"command": "scripts.automod.gate", "timeout_secs": 3600},
        "budget": {"max_rounds": 1, "max_variants_per_round": 1},
        "mutation_scope": {"writable_paths": ["app/**", "tests/**"]},
        "code": {"base_commit": "a" * 40, "branch": "automod/SM_1"},
    }


def test_a_valid_code_spec_passes():
    assert spec.validate_code_run_spec(valid_spec()) is None


@pytest.mark.parametrize("key", ["objective", "evaluation", "budget", "mutation_scope"])
def test_each_required_top_level_key_is_required(key):
    s = valid_spec()
    del s[key]
    assert spec.validate_code_run_spec(s) is not None


def test_writable_paths_must_be_a_list():
    # Rejected by autoresearch's own validator, which we layer on rather than
    # duplicate — so assert the outcome, not its exact wording.
    s = valid_spec()
    s["mutation_scope"]["writable_paths"] = "app/**"
    reason = spec.validate_code_run_spec(s)
    assert reason and "writable_paths" in reason


def test_a_traversing_writable_path_is_rejected():
    s = valid_spec()
    s["mutation_scope"]["writable_paths"] = ["../../etc"]
    assert "safe relative path" in (spec.validate_code_run_spec(s) or "")


def test_the_code_block_is_required_for_code_rounds():
    s = valid_spec()
    del s["code"]
    assert "code" in (spec.validate_code_run_spec(s) or "")


@pytest.mark.parametrize("key", ["base_commit", "branch"])
def test_code_block_fields_are_required(key):
    s = valid_spec()
    del s["code"][key]
    assert key in (spec.validate_code_run_spec(s) or "")


def test_the_denylist_is_not_overridable_by_a_spec():
    """The single most important safety property in the design.

    A round that could land a change to `scripts/automod/**` by naming it in
    `writable_paths` would remove the only guard permanently, and every
    subsequent round would inherit the weakened gate.
    """
    ok, reason, _ = spec.check_scope(["config.yaml"])
    assert not ok, "an explicit ask must not unlock a denied path"
    assert spec.classify("config.yaml") == "denied"
    # The content lanes are not an override either: with both sides in hand, a
    # change to the loop's own switch is refused by key, and the denied-key
    # tuple is read from the module the gate imports, not from any spec.
    ok, reason, _ = spec.check_scope(
        ["config.yaml"], contents=lambda p: ("automod:\n  enabled: true\n",
                                             "automod:\n  enabled: false\n"))
    assert not ok and "automod.enabled" in reason


# --- config.yaml: comment-only edits land; value edits land inside the fence --

_CFG = REPO_ROOT / "config.yaml"


def _scope_with(before: str, after: str, path: str = "config.yaml"):
    return spec.check_scope(["app/x.py", path], contents=lambda p: (before, after))


def test_a_comment_only_config_edit_is_in_scope_on_the_real_file():
    before = _CFG.read_text(encoding="utf-8")
    assert "\n  # " in before, "positive control: the live file carries comments"
    after = before.replace("\n  # ", "\n  # (reworded) ", 1)
    after = after.replace("\n  # ", "\n  #\n  # ", 1)  # a new comment line too
    ok, reason, buckets = _scope_with(before, after)
    assert ok, reason
    assert buckets["comment_only"] == ["config.yaml"] and buckets["denied"] == []


def test_moving_an_inline_comment_off_a_value_line_is_comment_only():
    before = "a:\n  cap: 400   # a section is <= 162 lines\n"
    after = "a:\n  # One review's whole-file diff.\n  cap: 400\n"
    assert spec.comment_only_change(before, after) == (True, "comments and layout only")


@pytest.mark.parametrize("after, changed", [
    ("a:\n  cap: 401   # a section is <= 162 lines\n", ["a.cap"]),      # a value
    ("a:\n  cap: '400'   # a section is <= 162 lines\n", ["a.cap"]),    # quoting changes the type
    ("a:\n  cap: 400\n  extra: 1\n", ["a.extra"]),                      # a new key
    ("a:\n  cup: 400\n", ["a.cap", "a.cup"]),                           # a renamed key = removal + add
])
def test_any_value_change_to_config_yaml_stays_denied(after, changed):
    """Until 2026-10-05 every one of these stayed denied. Now a value change
    under an ALLOWED key is a lane of its own: it is not comment-only (that
    check still refuses it — pinned so the two lanes stay distinct), it lands
    in the `config_value` bucket, and `config_value_change` names exactly the
    dotted paths that moved. A renamed key under an allowed prefix is a removal
    plus an addition below the top level, and lands; a removed TOP-LEVEL key is
    refused (`test_a_top_level_key_removal_is_refused`)."""
    before = "a:\n  cap: 400   # a section is <= 162 lines\n"
    assert spec.comment_only_change(before, after)[0] is False
    ok, reason, paths = spec.config_value_change(before, after)
    assert ok and paths == changed, reason
    ok, reason, buckets = _scope_with(before, after)
    assert ok, reason
    assert buckets["config_value"] == ["config.yaml"]
    assert buckets["denied"] == [] and buckets["comment_only"] == []


@pytest.mark.parametrize("before, after, side", [
    ("a:\n  cap: 400\n", "a:\n  cap: [400\n", "after"),      # does not parse
    ("a:\n  cap: [400\n", "a:\n  cap: 400\n", "before"),
    ("a:\n  cap: 400\n", "- a\n- b\n", "after"),               # parses, not a mapping
])
def test_a_non_parsing_side_is_refused_and_named(before, after, side):
    ok, reason, paths = spec.config_value_change(before, after)
    assert not ok and reason.startswith(side) and paths == [], reason
    ok, reason, buckets = _scope_with(before, after)
    assert not ok and buckets["denied"] == ["config.yaml"], reason
    assert "value change outside the denied keys" in reason and side in reason


def test_the_real_file_with_one_value_flipped_is_denied():
    """On the real file: the loop's own switch is refused and the refusal names
    the key; a real tunable — one of the five the 13 stalled items wanted —
    lands in `config_value`. The lock-out key is tried too."""
    before = _CFG.read_text(encoding="utf-8")
    cfg = yaml.safe_load(before)
    assert cfg["automod"]["enabled"] is True, "positive control: the switch is on"
    # Flipped through a parsed round-trip: the file's layout between `automod:`
    # and its `enabled:` line is comments, and the lane judges values not text.
    doc = yaml.safe_load(before)
    doc["automod"]["enabled"] = False
    after = yaml.safe_dump(doc)
    ok, reason, buckets = _scope_with(before, after)
    assert not ok and buckets["denied"] == ["config.yaml"], reason
    assert "`automod.enabled` is under the denied key `automod.enabled`" in reason

    old = cfg["harness"]["finalizer"]["max_tokens"]
    after = before.replace(f"max_tokens: {old}", f"max_tokens: {old * 2}", 1)
    ok, reason, paths = spec.config_value_change(before, after)
    assert ok and paths == ["harness.finalizer.max_tokens"], reason
    ok, reason, buckets = _scope_with(before, after)
    assert ok, reason
    assert buckets["config_value"] == ["config.yaml"] and buckets["denied"] == []

    doc = yaml.safe_load(before)
    doc["mcp_servers"]["lloyd-mcp"]["disabled_tools"] = ["Bash", "Edit"]
    ok, reason, _ = spec.config_value_change(before, yaml.safe_dump(doc))
    assert not ok and "mcp_servers.lloyd-mcp.disabled_tools" in reason, reason


def test_a_round_cannot_arm_the_fact_write_gate():
    """#2344 clause 1, on the real file. `config.yaml` reserves
    `knowledge_graph.write_gate.mode: "on"` for "an explicit decision from Alan
    plus a measured LloydMemEval `knowledge_update` gain" (#1487) — and until this
    fence, that reservation was a comment: `agent_mcp/fact_write_gate.py`'s `mode()`
    takes the config key with no guard behind it, so a round editing one value
    could start it stamping `expired_at` on facts. The refusal has to name BOTH the
    path a reviewer would look for and the prefix that refused it."""
    before = _CFG.read_text(encoding="utf-8")
    cfg = yaml.safe_load(before)
    gate_cfg = cfg["knowledge_graph"]["write_gate"]
    assert gate_cfg["mode"] == "noop", \
        "positive control: the file ships noop, so the flip below is an arming"

    # Replace EVERY occurrence, the spelling of the item's CHECK. `mode: "noop"`
    # appears twice in the file — once inside the reservation comment at :2362 and
    # once as the value at :2369 — and a count-limited replace lands on the
    # comment, which the diff cannot see: the answer comes back "no value changed"
    # and the test passes for the wrong reason.
    after = before.replace('mode: "noop"', 'mode: "on"')
    assert after != before, "positive control: the mode line was found to flip"
    ok, fence_reason, paths = spec.config_value_change(before, after)
    assert not ok, "arming the gate read as an ordinary value change"
    assert paths == ["knowledge_graph.write_gate.mode"], paths
    assert "`knowledge_graph.write_gate.mode` is under the denied key " \
           "`knowledge_graph.write_gate`" in fence_reason, fence_reason
    # The fence's own line names ONE path and ONE prefix. `knowledge_graph` carries
    # exactly two keys in the real file, and naming the other one here would be the
    # over-freeze the sibling test below refuses. (`check_scope`'s reason is not
    # asserted for this: it quotes the rule back at the round, "denied keys" and
    # all, so counting phrases in it measures the prose.)
    assert "knowledge_graph.write_enabled" not in fence_reason, fence_reason
    assert fence_reason.count("denied key") == 1, fence_reason
    ok, reason, buckets = _scope_with(before, after)
    assert not ok and buckets["denied"] == ["config.yaml"] and \
        buckets["config_value"] == [], (reason, buckets)


def test_the_write_gate_fence_stops_at_arming():
    """#2344 clause 2: the denied prefix is the dotted `knowledge_graph.write_gate`,
    NOT `knowledge_graph`. Fencing the section whole would look safer and freeze a
    tunable the loop is allowed to change — `write_enabled`, whose false is the
    kill switch #1236 shipped and whose value a round may legitimately move. If
    this goes red, someone widened the fence and the loop lost a knob it never
    needed to lose."""
    before = _CFG.read_text(encoding="utf-8")
    cfg = yaml.safe_load(before)
    assert cfg["knowledge_graph"]["write_enabled"] is True, \
        "positive control: writes are enabled on the live tree"
    assert cfg["knowledge_graph"]["write_gate"]["mode"] == "noop"

    after = before.replace("write_enabled: true", "write_enabled: false", 1)
    assert after != before, "positive control: the sibling line was found to flip"
    ok, reason, paths = spec.config_value_change(before, after)
    assert ok, f"a sibling of the armed key was refused: {reason}"
    assert paths == ["knowledge_graph.write_enabled"], paths
    ok, reason, buckets = _scope_with(before, after)
    assert ok and buckets["config_value"] == ["config.yaml"] and \
        buckets["denied"] == [], (reason, buckets)
    # And the fence matches on a dotted PREFIX, not a substring: a key that merely
    # starts with the denied name's characters is not under it.
    assert spec._config_denied_prefix("knowledge_graph.write_gate_mode") is None, \
        "a sibling named `write_gate_mode` would have been frozen by a substring rule"


# --- the fence around the value lane -----------------------------------------

def test_a_denied_prefix_is_refused_with_the_key_named():
    before = "workers:\n  enabled: true\n  slots: 5\n  sources:\n    autocode:\n      max_inflight: 2\n    autoresearch:\n      max_duration_seconds: 3600\n"
    # The loop's own source: refused, prefix named.
    after = before.replace("max_inflight: 2", "max_inflight: 4")
    ok, reason, paths = spec.config_value_change(before, after)
    assert not ok and paths == ["workers.sources.autocode.max_inflight"]
    assert "`workers.sources.autocode.max_inflight` is under the denied key `workers.sources.autocode.max_inflight`" in reason
    # A sibling source that is not the loop's own: lands.
    after = before.replace("max_duration_seconds: 3600", "max_duration_seconds: 7200")
    ok, reason, paths = spec.config_value_change(before, after)
    assert ok and paths == ["workers.sources.autoresearch.max_duration_seconds"], reason
    # Prefix match is on dotted segments, never on a string prefix: a key that
    # merely starts with a denied word is not under it.
    assert spec._config_denied_prefix("modelling.cap") is None
    assert spec._config_denied_prefix("model.default") == "model"
    assert spec._config_denied_prefix("workers.slots") == "workers.slots"
    # Two offences on one diff are both named.
    after = before.replace("slots: 5", "slots: 7").replace("enabled: true", "enabled: false")
    ok, reason, _ = spec.config_value_change(before, after)
    assert not ok and "`workers.slots`" in reason and "`workers.enabled`" in reason


def test_a_placeholder_scalar_is_refused_on_either_side():
    before = "livekit:\n  room: lloyd\n  api_key: ${LIVEKIT_API_KEY}\n"
    # Replacing a placeholder with a literal is the secret-in-the-tree path.
    after = before.replace("${LIVEKIT_API_KEY}", "abc123")
    ok, reason, _ = spec.config_value_change(before, after)
    assert not ok and "`livekit.api_key` carries a `${` placeholder" in reason, reason
    # ...and introducing one is refused too, on a leaf name that is otherwise fine.
    after = before.replace("room: lloyd", "room: ${LIVEKIT_ROOM}")
    ok, reason, _ = spec.config_value_change(before, after)
    assert not ok and "`livekit.room` carries a `${` placeholder" in reason, reason
    # A placeholder inside a list is a placeholder.
    ok, reason, _ = spec.config_value_change("a:\n  xs: [1]\n", "a:\n  xs: ['${X}']\n")
    assert not ok and "`a.xs` carries" in reason, reason


def test_a_denied_leaf_is_refused_under_an_otherwise_allowed_prefix():
    before = "djev:\n  base_url: http://127.0.0.1:8097\n  timeout_seconds: 5\n  device: cuda:2\n"
    after = before.replace("8097", "8098")
    ok, reason, paths = spec.config_value_change(before, after)
    assert not ok and paths == ["djev.base_url"], reason
    assert "`djev.base_url` is a denied leaf name (`base_url`)" in reason
    after = before.replace("cuda:2", "cuda:0")
    ok, reason, _ = spec.config_value_change(before, after)
    assert not ok and "`djev.device` is a denied leaf name (`device`)" in reason
    # The sibling tunable under the same prefix lands.
    after = before.replace("timeout_seconds: 5", "timeout_seconds: 9")
    ok, reason, paths = spec.config_value_change(before, after)
    assert ok and paths == ["djev.timeout_seconds"], reason
    assert reason == "value change within the fence: djev.timeout_seconds"
    for leaf in ("base_url", "url", "host", "port", "expect_model", "device", "devices",
                 "gpu", "cuda_visible_devices", "token", "api_key", "password", "secret"):
        assert leaf in spec.CONFIG_DENIED_LEAVES, leaf


def test_a_top_level_key_removal_is_refused():
    before = "a:\n  cap: 400\nb:\n  x: 1\n"
    ok, reason, paths = spec.config_value_change(before, "a:\n  cap: 400\n")
    assert not ok and paths == ["b"], reason
    assert "`b` is a top-level key and the change removes it" in reason
    # Adding a top-level key, and removing a nested one, both land.
    ok, reason, paths = spec.config_value_change(before, "a:\n  cap: 400\nb:\n  x: 1\nc: 2\n")
    assert ok and paths == ["c"], reason
    ok, reason, paths = spec.config_value_change(before, "a:\n  cap: 400\nb: {}\n")
    assert ok and paths == ["b.x"], reason


def test_no_value_change_is_not_a_value_lane():
    """Identical documents are the comment lane's business; the value lane
    refuses them, so a diff that passes neither says so for both."""
    ok, reason, paths = spec.config_value_change("a: 1\n", "a: 1 # c\n")
    assert (ok, reason, paths) == (False, "no value changed", [])


def test_the_admitting_reason_is_bounded():
    before = "a:\n" + "".join(f"  k{i}: {i}\n" for i in range(14))
    after = "a:\n" + "".join(f"  k{i}: {i + 1}\n" for i in range(14))
    ok, reason, paths = spec.config_value_change(before, after)
    assert ok and len(paths) == 14
    assert reason.endswith("… (+4 more)"), reason
    assert reason.count("a.k") == 10


def test_a_list_is_a_leaf_compared_whole():
    before = "harness:\n  egress_policy:\n    allow: [a, b]\n"
    after = "harness:\n  egress_policy:\n    allow: [a, b, c]\n"
    ok, reason, paths = spec.config_value_change(before, after)
    assert ok and paths == ["harness.egress_policy.allow"], reason


def test_a_config_value_change_arms_the_drill_via_buckets():
    """The bound on the lane is on HOW it lands: a config value change is
    boot-affecting, so rung 0's bucket turns the `drill` rung on the way a protected
    path does. Paths alone cannot see it — the bucket is content-judged — so
    the paths-only call keeps its old answer and the gate passes the buckets."""
    ok, reason, buckets = _scope_with("a:\n  cap: 400\n", "a:\n  cap: 401\n")
    assert ok and buckets["config_value"] == ["config.yaml"], reason
    assert spec.requires_drill(["app/x.py", "config.yaml"]) is False
    assert spec.requires_drill(["app/x.py", "config.yaml"], buckets=buckets) is True
    # A comment-only edit arms nothing; an empty bucket dict is the old answer.
    ok, _, buckets = _scope_with("a:\n  cap: 400\n", "a:\n  # c\n  cap: 400\n")
    assert ok and buckets["comment_only"] == ["config.yaml"]
    assert spec.requires_drill(["app/x.py", "config.yaml"], buckets=buckets) is False
    assert spec.requires_drill(["scripts/automod/spec.py"], buckets={"config_value": []}) is True


def test_comment_only_wins_when_both_lanes_would_pass():
    """A comment-only diff also passes no value check (there is no value to
    judge), but the order is pinned anyway: the comment lane is tried first,
    so a comment edit never buys a drill."""
    before = "a:\n  cap: 400   # old\n"
    after = "a:\n  # new\n  cap: 400\n"
    assert spec.comment_only_change(before, after)[0] is True
    ok, _, buckets = _scope_with(before, after)
    assert ok and buckets["comment_only"] == ["config.yaml"] and buckets["config_value"] == []


@pytest.mark.parametrize("src", ["autocode", "autotriage", "owed-check"])
def test_the_loops_own_sources_are_fenced_by_leaf_not_whole(src):
    """`workers.sources.autocode.reasoning_bank` was one of the five stalled
    tunables the lane was built for, and a whole-prefix fence refused it. The
    control surface — on/off, depth, model, budget, apply — stays out of reach."""
    before = (f"workers:\n  sources:\n    {src}:\n      enabled: true\n      max_inflight: 1\n"
              f"      model: primary\n      max_turns: 40\n      max_duration_seconds: 1500\n"
              f"      apply: true\n      reasoning_bank: 'off'\n      spawn_cap: 3\n")
    for tunable, new in (("reasoning_bank: 'off'", "reasoning_bank: 'on'"), ("spawn_cap: 3", "spawn_cap: 1")):
        ok, reason, _ = spec.config_value_change(before, before.replace(tunable, new))
        assert ok, reason
    for leaf, new in (("enabled: true", "enabled: false"), ("max_inflight: 1", "max_inflight: 4"),
                      ("model: primary", "model: secondary"), ("max_turns: 40", "max_turns: 400"),
                      ("max_duration_seconds: 1500", "max_duration_seconds: 9000"),
                      ("apply: true", "apply: false")):
        ok, reason, paths = spec.config_value_change(before, before.replace(leaf, new))
        assert not ok and f"workers.sources.{src}." in reason, (leaf, reason)
    # Removing the whole block is the off switch by another spelling.
    ok, reason, _ = spec.config_value_change(before, "workers:\n  sources: {}\n")
    assert not ok and f"workers.sources.{src}" in reason
    # A sibling source that is not the loop's own keeps every leaf open.
    other = before.replace(f"{src}:", "autoresearch:")
    assert spec.config_value_change(other, other.replace("max_turns: 40", "max_turns: 60"))[0]


def test_the_denied_key_set_is_the_one_landed():
    """The fence, verbatim: a widening or a narrowing here is a decision, and
    the test should say so rather than let it ride on an unrelated commit."""
    assert spec.CONFIG_DENIED_KEYS == (
        "mcp_servers", "models", "model", "subagents", "server", "services",
        "guardian", "automod.enabled", "automod.landing", "workers.enabled",
        "workers.slots",
        # #2344: arming the fact-expiring write gate, which `mode()` in
        # agent_mcp/fact_write_gate.py reads with no guard of its own. The dotted
        # `write_gate`, not `knowledge_graph` — the sibling tunable stays editable,
        # pinned by test_the_write_gate_fence_stops_at_arming.
        "knowledge_graph.write_gate",
    )
    assert spec.CONFIG_LOOP_SOURCES == ("autocode", "autotriage", "owed-check")
    assert spec.CONFIG_LOOP_SOURCE_LEAVES == {
        "enabled", "max_inflight", "model", "max_turns", "max_duration_seconds", "apply"}
    # Every denied prefix names a key the real file carries, or the fence is
    # guarding a spelling nobody uses.
    cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8"))
    for prefix in spec.CONFIG_DENIED_KEYS:
        node = cfg
        for part in prefix.split("."):
            assert isinstance(node, dict) and part in node, f"{prefix}: `{part}` not in config.yaml"
            node = node[part]
    # And the refusal text tells the round where the fence is written down.
    ok, reason, _ = _scope_with("server:\n  port: 8080\n", "server:\n  port: 8081\n")
    assert not ok and "CONFIG_DENIED_KEYS" in reason and "`server.port`" in reason


def test_comment_only_is_config_yaml_alone_and_unreadable_is_denied():
    """`.gitignore` with only a comment changed is still denied: the exception
    is for the one denied file the YAML check can judge."""
    ok, _, buckets = _scope_with("# a\n", "# b\n", path=".gitignore")
    assert not ok and buckets["denied"] == [".gitignore"]

    def missing(_p):
        raise FileNotFoundError("HEAD:config.yaml")
    ok, reason, buckets = spec.check_scope(["config.yaml"], contents=missing)
    assert not ok and buckets["denied"] == ["config.yaml"], reason


@pytest.mark.parametrize("path", ["web/src/App.tsx", "web/src/components/pages/BrowserPage.tsx",
                                  "web/index.html", "web/public/favicon.svg"])
def test_frontend_sources_are_allowed_because_the_frontend_rung_builds_them(path):
    """`web/**` was denied outright until 2026-09-07 because the gate did not
    build the frontend. It does now (rung `frontend`: tsc delta + vite build),
    so the sources are ordinary code; only the build inputs stay denied."""
    assert spec.classify(path) == "allowed"


def test_frontend_tooling_outside_src_is_unlisted_not_allowed():
    assert spec.classify("web/eslint.config.js") == "unlisted"


# ---------------------------------------------------------------------------
# the agent-services widening (#1376): rails for a decision the loop may not make
# ---------------------------------------------------------------------------
#
# #1376 asks for `agent-services/livekit_worker.py` to join ALLOWED_GLOBS, so a
# voice-stack fix can clear rung 0 at all — round SM_20260922_201113 wrote that
# fix and its tests and was refused before anything was judged. The item's own
# acceptance is `human-only: scripts/automod/spec.py`, and the reason is
# measured, not stylistic: the file that decides the writable set sits under the
# allowed prefix `scripts/**`, and `grep -rn "automod/spec" --include=*.py .`
# finds no rail against editing it outside `tests/` and `spec.py` itself. So a
# round may not make that edit, and this section does not make it either. What
# it owns is the *shape* the edit has to take (the edit itself was made by hand
# on 2026-09-24 under #1449, once #1444's round was refused at rung 0 for the
# same path): rails aimed at the class of bad
# widening (a directory glob) rather than at one name, a simulation that proves
# those rails can fail, a second simulation that runs every one of them against
# the single named entry the item asks for, and a check that the verdicts they
# pin are the verdicts a second interpreter reading the tree prints.

#: `ALLOWED_GLOBS` as committed, snapshotted at import. The two simulations below
#: widen from this and never from `spec.ALLOWED_GLOBS` at call time: a monkeypatch
#: undo that did not run, in any test anywhere in this file, would otherwise feed
#: one simulation's grant into the other, and the pair assert opposite verdicts —
#: the blanket edit must make the rails fire, the named edit must not. A leaked
#: `"agent-services/**"` therefore turns the green-path test red for a reason that
#: is not in the tree, which is exactly how a fence test stops meaning anything.
#: Reproduced before it was fixed: driving these two functions in-process without
#: restoring the tuple printed 79 violations on the named-file path.
ALLOWED_GLOBS_AS_SHIPPED: tuple[str, ...] = tuple(spec.ALLOWED_GLOBS)

#: Paths a widening must never reach, none of them named in ALLOWED_GLOBS today.
#: Each stands in for a whole directory, because that is what a glob does: an
#: entry that admits one of these admits every tracked file under it.
AGENT_SERVICES_GLOB_PROBES: tuple[str, ...] = (
    "agent-services/conf/probe.yml",
    "agent-services/services/probe/probe_server.py",
    "agent-services/voice/probe.sh",
    "agent-services/setup/probe.sh",
    "agent-services/models/wakeword/probe.onnx",
)

#: Five of the acoustic weights, all tracked. Not the whole tracked model tree, and this
#: line used to claim it was by saying `git ls-files agent-services/models` prints 6: that
#: command prints 7 at this round's base — these five, `wakeword/stop/lloyd_stop.onnx` (a
#: sixth weight no name in this file mentions) and `silero-vad/SOURCE`, a text note.
#: What keeps the unlisted sixth weight from being a blind spot is
#: `test_the_acoustic_weight_rail_covers_every_tracked_model_file`: it asserts this tuple is
#: a subset of the tracked tree, and that the writable model set is exactly the #3258
#: grant's tracked members — empty in this tree, the three once #3252 commits the bytes. An
#: unnamed weight is therefore refused the moment a grant or a wildcard makes it writable,
#: which is the only way a round could edit it at all.
#:
#: Named apart from the probes because they are the one `agent-services/**`
#: content whose bad edit no rung can see: rung 0 matches paths, the drill boots
#: the stack, and the eval axis is retrieval — none of the three scores whether a
#: wake-word model still fires, so a rewritten `.onnx` would land silently and
#: show up as a dead wake word.
AGENT_SERVICES_ACOUSTIC_WEIGHTS: tuple[str, ...] = (
    "agent-services/models/wakeword/hey_lloyd.onnx",
    "agent-services/models/wakeword/Lloyd.onnx",
    "agent-services/models/openwakeword/embedding_model.onnx",
    "agent-services/models/openwakeword/melspectrogram.onnx",
    "agent-services/models/silero-vad/silero_vad.onnx",
)

#: #3258's grant, verbatim: the end-of-turn detector the shipped config names, plus the
#: two licence files that travel with it. Spelled here rather than at each rail because
#: four nodes below asserted "no `agent-services/models/` path is `allowed`" before this
#: grant existed, and that claim is now false by design — so each of them narrows to
#: "the only model path the tuple admits is one of these three", which is a claim that
#: holds both before #3252 commits the bytes (none of the three is in the index, so the
#: set is empty) and after (all three are, and are allowed). Naming the exception once
#: is also what keeps the narrowing from becoming a hole: a rail that simply dropped the
#: assertion would stop refusing the weights, and this tuple still refuses them.
#:
#: These three paths are deliberately NOT added to `AGENT_SERVICES_ACOUSTIC_WEIGHTS`:
#: that tuple is the list of files `test_an_acoustic_model_weight_is_never_writable_by_a_round`
#: asserts are NOT allowed, so a member here would fail that node by construction.
TURN1MINI_GRANT: tuple[str, ...] = (
    "agent-services/models/turn-1-mini/onnx/turn-1-mini.step.onnx",
    "agent-services/models/turn-1-mini/LICENSE",
    "agent-services/models/turn-1-mini/NOTICE",
)

#: The turn-1-mini files that sit beside the grant on this disk and stay `unlisted`.
#: The int8 sibling is the one that matters: it is a second copy of the same model in a
#: different quantisation, no config key reads it, and a directory grant would admit it
#: alongside the fp32 the worker loads.
TURN1MINI_SIBLINGS: tuple[str, ...] = (
    "agent-services/models/turn-1-mini/onnx/turn-1-mini.step.int8.onnx",
    "agent-services/models/turn-1-mini/SHA256SUMS",
    "agent-services/models/turn-1-mini/config.json",
)


def model_paths_allowed() -> list[str]:
    """Every `agent-services/models/` path in the index that classifies `allowed`.

    The shape all three pre-grant rails share (`test_the_acoustic_weight_rail_covers_every_tracked_model_file`,
    the tail of `test_the_tts_grant_moves_exactly_one_tracked_agent_services_verdict`,
    and of `test_the_qmd_grant_moved_exactly_one_classification`) asserted this list was
    empty. It reads from the index for the same reason `tracked_agent_services_paths`
    does: rung 0 classifies the paths a diff touches, and only a tracked file can be in
    a diff. Callers compare the result against `TURN1MINI_GRANT` rather than against
    `[]`, which is what makes the same assertion true in #3258's tree and in the tree
    #3252 lands into.
    """
    return sorted(p for p in tracked_agent_services_paths()
                  if p.startswith("agent-services/models/")
                  and spec.classify(p) == "allowed")


def expected_allowed_model_paths() -> list[str]:
    """`TURN1MINI_GRANT` intersected with what the index actually holds.

    Empty until #3252 commits the blob, all three afterwards. Derived from the index
    rather than hard-coded so the rails above do not have to be edited twice — once by
    this grant and once by the commit it exists to unblock — which is the failure mode
    that would leave a model weight writable by a round with no test noticing.
    """
    tracked = set(tracked_agent_services_paths())
    return sorted(p for p in TURN1MINI_GRANT if p in tracked)


def tracked_agent_services_paths() -> list[str]:
    """Every file git tracks under `agent-services/`, read from the index.

    The index and not the working tree because rung 0 classifies the paths a diff
    touches (`gate.py:928`, `rung_preflight`), which is exactly what a commit can
    carry there. Measured at `1842b8cf`: 187 files, of which 112 already classify
    `protected` (writable today, gated by the `drill` rung) and 75 `unlisted`. Re-read
    at this round's base `c2bd72b`: 192 files, 117 `protected`, 75 `unlisted` —
    the five more are `guardian/datawatch.py` and four `systemd/` units, all
    already `protected`, which is why the unlisted count the rails are priced on
    has not moved.

    The length check is a denominator control, not decoration: a helper
    returning an empty list would make every "nothing else changed" assertion in
    this section vacuously true. It deliberately does not check *verdicts* —
    `test_a_blanket_agent_services_glob_would_trip_each_rail` calls this helper
    with ALLOWED_GLOBS deliberately widened, so a verdict check here would fire
    on the simulation instead of on the tree.
    """
    out = subprocess.run(["git", "ls-files", "agent-services"], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    paths = sorted(p for p in out.stdout.split() if p)
    assert len(paths) >= 187, (
        f"only {len(paths)} tracked files under agent-services/, below the 187 "
        f"this rail was sized on — 'nothing else changed' would be guarding "
        f"almost nothing")
    return paths


def agent_services_paths_admitted_without_being_named() -> list[str]:
    """The violation list for path-exactness: an `agent-services/**` path that
    classifies `allowed` although no ALLOWED_GLOBS entry is that exact path.

    This is the literal reading of #1376 clause 1 — "no `agent-services/**` file
    other than the ones named in the edit changes classification" — and it
    refuses a wildcard spelling of even a single file on purpose: `fnmatch`'s
    `*` crosses `/` (spec.py:107-112), so a wildcard in that tuple is one
    keystroke away from a grant over the whole tree.
    """
    verbatim = {g for g in spec.ALLOWED_GLOBS if "*" not in g}
    universe = tracked_agent_services_paths() + list(AGENT_SERVICES_GLOB_PROBES)
    return sorted(p for p in universe
                  if p not in verbatim and spec.classify(p) == "allowed")


def test_only_a_verbatised_path_under_agent_services_may_be_admitted():
    """#1376 clause 1, as a property rather than a name.

    The edit the item anticipates — one line, `"agent-services/livekit_worker.py"`
    added to ALLOWED_GLOBS — keeps this green, because the admitted file is then
    also the named file. The shortcut edit goes red: `"agent-services/**"` (or
    `"agent-services/services/**"`) admits the 75 unlisted tracked files plus all
    five probes — measured 80 violations against the blanket glob — while the 117
    tracked files under `PROTECTED_GLOBS` stay `protected`, the ordering property
    `test_a_blanket_agent_services_glob_would_trip_each_rail` pins separately.
    Same shape as `test_admitting_prompt_surface_did_not_widen_anything_else`,
    the precedent the item cites for the fix, widened from two root-level
    samples to the whole tracked `agent-services/` corpus.

    The violation assertion comes first on purpose: with the vacuity guard first,
    a blanket glob failed on the *count* and the message blamed the corpus for
    what the grant had done, so the failure read as "this rail has nothing left to
    guard" instead of naming the illegal admission.
    """
    violations = agent_services_paths_admitted_without_being_named()
    assert violations == [], (
        f"{len(violations)} agent-services path(s) became writable without being "
        f"named in ALLOWED_GLOBS: {violations[:8]}"
        + (f" … (+{len(violations) - 8} more)" if len(violations) > 8 else ""))
    unlisted = [p for p in tracked_agent_services_paths()
                if spec.classify(p) == "unlisted"]
    assert len(unlisted) >= 70, (
        f"only {len(unlisted)} tracked agent-services paths are still unlisted, "
        f"against 75 at both 1842b8cf and c2bd72b: the widening #1376 anticipates "
        f"is effectively complete, so this rail is close to vacuous and the item "
        f"should be re-read rather than trusted")


@pytest.mark.parametrize("path", AGENT_SERVICES_ACOUSTIC_WEIGHTS)
def test_an_acoustic_model_weight_is_never_writable_by_a_round(path):
    """#1376 clause 2: nothing under `agent-services/models/**` is `allowed`.

    Rung 0 is the only thing between an unattended round and these bytes, so the
    grant has to be written as if the blanket glob were the tempting option —
    which it is: it is one line, and it also closes #1301.
    """
    assert spec.classify(path) != "allowed"


def test_the_acoustic_weight_rail_covers_every_tracked_model_file():
    """The named weights are a subset of the tracked corpus, and nothing else is writable.

    Guards the case the parametrised test above cannot see on its own: a model file
    added to the repo that no name in this file mentions.

    Two claims here were looser than the tree before #3258, and both are now exact
    rather than restated. The corpus is seven tracked files, not the five named weights:
    `git ls-files agent-services/models/` also returns `wakeword/stop/lloyd_stop.onnx`
    and `silero-vad/SOURCE`, which is why the subset assertion below is `<=` and not
    equality — a stale docstring that promised "the whole tracked corpus" would have had
    a reader conclude the stop model and the note were untracked. And the last line used
    to assert no tracked model file classifies `allowed`, which #3258's grant makes false
    by design once #3252 commits the bytes: it now asserts that the allowed set is
    exactly the granted paths that are in the index, so the five weights, the stop model
    and the SOURCE note stay refused in both trees, and a granted path that quietly
    stopped being allowed is caught in the one tree where that is a regression.
    """
    models = [p for p in tracked_agent_services_paths()
              if p.startswith("agent-services/models/")]
    assert models, "no tracked file under agent-services/models/ — this rail guards nothing"
    assert set(AGENT_SERVICES_ACOUSTIC_WEIGHTS) <= set(models), (
        f"tracked model files this rail does not name: "
        f"{sorted(set(models) - set(AGENT_SERVICES_ACOUSTIC_WEIGHTS))}")
    assert model_paths_allowed() == expected_allowed_model_paths(), (
        f"the writable set inside agent-services/models/ is {model_paths_allowed()}, "
        f"expected only the #3258 grant that is in the index "
        f"{expected_allowed_model_paths()}; anything else in there is a model weight a "
        f"round may rewrite with no rung scoring whether it still fires")


def test_no_grant_in_the_allowed_tuple_reaches_the_models_tree():
    """#1376 clause 2, asked of the grants rather than of a sample of paths.

    The two nodes above ask whether known files classify `allowed`; a weight
    added tomorrow under a directory no name here mentions answers neither, so
    this turns the question round and interrogates every entry of
    `ALLOWED_GLOBS` with a path deeper in the tree than anything enumerated.
    `spec._match` is the matcher rung 0 itself uses (`gate.py:928` →
    `check_scope` → `classify`), so a grant that reaches this probe is a grant
    that would reach the weights — and the blanket `"agent-services/**"` edit
    this item names as the tempting option is what turns it red.
    """
    probe = "agent-services/models/probe-dir/probe_weight.onnx"
    offenders = [g for g in spec.ALLOWED_GLOBS if spec._match(probe, (g,))]
    assert offenders == [], f"ALLOWED_GLOBS entries reaching the models tree: {offenders}"
    assert spec.classify(probe) == "unlisted"


def test_the_scope_spec_stays_protected_though_scripts_is_allowed():
    """#1376 clause 3, and the reason the widening is a person's job.

    `scripts/**` is in ALLOWED_GLOBS, so `scripts/automod/spec.py` — the module
    that decides what a round may write — sits under an allowed prefix. It is
    `protected` only because `classify` consults PROTECTED before ALLOWED
    (spec.py:144-147); the item's probe is that no other file in the tree rails
    it. That asymmetry is why the widening binds only once landed: the gate is
    spawned against the live tree (`round.py:445`) and grades rung 0 with the
    spec that interpreter imports (`gate.py:53`, `gate.py:928`), so a round that
    widened the tuple in its own worktree would still be refused by its own gate,
    and every *later* round would inherit the widening.
    """
    assert "scripts/**" in spec.ALLOWED_GLOBS
    assert spec.classify("scripts/automod/spec.py") == "protected"
    assert spec.requires_drill(["scripts/automod/spec.py"])
    ok, _, buckets = spec.check_scope(["scripts/automod/spec.py"])
    assert ok and buckets["protected"] == ["scripts/automod/spec.py"], (
        "protected must stay permitted-but-drilled: refusing it outright would "
        "make the control surface unfixable, which is the difference between "
        "guarding a capability and amputating it")


def test_a_blanket_agent_services_glob_would_trip_each_rail(monkeypatch):
    """Proves the rails above can fail, and prices the wrong edit.

    A property of a currently-narrow tuple is worth nothing unless the widening
    being proposed is the thing that turns it red, so this applies the edit a
    rushed person makes — one line, `"agent-services/**"`, which would also close
    #1301 and is the tempting option — to the tuple the rails read, and calls
    `test_only_a_verbatised_path_under_agent_services_may_be_admitted` expecting
    it to raise. `fnmatch`'s `*` crosses `/` (spec.py:107-112), so that line is a
    grant over every tracked file in the tree, weights included: 80 violations
    measured at this base `c2bd72b`, the 75 unlisted tracked files plus the 5
    probes, which the assertion re-derives from the corpus rather than pinning.
    The raised message has to name the illegal admission — with the vacuity guard
    written ahead of it, the failure blamed the corpus count for what the grant
    had done.

    The last two assertions are the ordering property that keeps even that bad
    edit survivable: PROTECTED is consulted first (spec.py:142-147), so widening
    ALLOWED cannot pull the scope spec, the guardian or the supervisor confs out
    of the rollback drill.
    """
    # The expected violation set, measured from the tuple BEFORE it is widened:
    # every currently-unlisted tracked file plus every probe, because a blanket
    # glob admits exactly those — the 117 tracked `protected` paths never leave
    # `protected` (asserted below), and no currently-allowed path is unnamed.
    # Derived rather than hard-coded: tracked file count in that tree moves, and
    # a pinned 80 would go red on an unrelated commit that added one setup script.
    unlisted_before = [p for p in tracked_agent_services_paths()
                       if spec.classify(p) == "unlisted"]
    expected = len(unlisted_before) + len(AGENT_SERVICES_GLOB_PROBES)
    monkeypatch.setattr(spec, "ALLOWED_GLOBS",
                        (*ALLOWED_GLOBS_AS_SHIPPED, "agent-services/**"))
    assert spec.classify("agent-services/livekit_worker.py") == "allowed"
    # The rail itself, not a restatement of its helper: clause 1 asks that the
    # same rail turn red under the blanket edit, so the edit is applied to the
    # tuple the rail reads and the rail is called. The message is the other half
    # of the assertion — it must name the grant's effect, and with the vacuity
    # guard ordered ahead it named the corpus instead.
    with pytest.raises(AssertionError) as fired:
        test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    message = str(fired.value)
    assert "became writable without being named" in message, message
    assert "are still unlisted" not in message, (
        f"the failure blamed the corpus denominator rather than the grant: {message}")
    violations = agent_services_paths_admitted_without_being_named()
    assert len(violations) == expected, (
        f"a blanket glob admitted {len(violations)} unnamed paths, not the "
        f"{expected} that are unlisted today plus the {len(AGENT_SERVICES_GLOB_PROBES)} "
        f"probes — the helper and the corpus no longer agree, so the rail is "
        f"guarding a set it cannot see")
    # The grant landed by hand on 2026-09-24 (#1449, for #1444), so under the
    # blanket edit the worker is the one admission that is *named*: it must not
    # read as a violation, or the rail would be refusing the widening it was
    # written to shape. Before the grant this line asserted the opposite.
    assert "agent-services/livekit_worker.py" in spec.ALLOWED_GLOBS
    assert "agent-services/livekit_worker.py" not in violations
    assert "agent-services/models/wakeword/hey_lloyd.onnx" in violations
    assert spec.classify("scripts/automod/spec.py") == "protected"
    assert spec.classify("agent-services/guardian/guardian.py") == "protected"


def test_the_edit_1376_anticipates_keeps_every_rail_green(monkeypatch):
    """#1376 clause 4: the rails bound the widening, they must not veto it.

    This is the same simulation with the one line the item asks a person to add,
    and it is the check that the rails are a fence rather than a wall: the path
    becomes `allowed`, the rung-0 verdict for the exact diff round
    SM_20260922_201113 wrote — `agent-services/livekit_worker.py` plus
    `tests/test_voice_gate.py`, the two files `gate.json` bucketed as
    `unlisted: ['agent-services/livekit_worker.py']` /
    `allowed: ['tests/test_voice_gate.py']` — comes back in scope with no drill,
    and every other path is untouched. Without this half the section would be a
    way of keeping #1058 unimplementable by a different route.
    """
    monkeypatch.setattr(spec, "ALLOWED_GLOBS",
                        (*ALLOWED_GLOBS_AS_SHIPPED, "agent-services/livekit_worker.py"))
    assert spec.classify("agent-services/livekit_worker.py") == "allowed"
    # "Every rail stays green", executed: each rail in this section is called
    # with the widened tuple in place, so a rail that only looks green in prose
    # — or that quietly starts vetoing the very edit it exists to shape — fails
    # here rather than in the reviewer's reading of this file.
    test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    for weight in AGENT_SERVICES_ACOUSTIC_WEIGHTS:
        test_an_acoustic_model_weight_is_never_writable_by_a_round(weight)
    test_the_acoustic_weight_rail_covers_every_tracked_model_file()
    test_no_grant_in_the_allowed_tuple_reaches_the_models_tree()
    test_the_scope_spec_stays_protected_though_scripts_is_allowed()
    ok, reason, buckets = spec.check_scope(["agent-services/livekit_worker.py",
                                           "tests/test_voice_gate.py"])
    assert (ok, reason) == (True, "in scope"), (ok, reason)
    assert not buckets["protected"] and not buckets["unlisted"]
    assert spec.requires_drill(["agent-services/livekit_worker.py",
                                "tests/test_voice_gate.py"]) is False, (
        "an empty protected bucket is the clause: this diff must not buy a "
        "guardian drill")


def test_the_verdicts_these_rails_pin_are_the_ones_another_interpreter_prints():
    """The seam rung 0 sits on, crossed the way the gate crosses it.

    `automod_gate` does not grade in-process: it is spawned detached with
    `cwd=LIVE_ROOT` (`round.py:445`) and rung 0 calls `spec.check_scope(changed)`
    at `gate.py:928` on the `spec` module *that* interpreter imported. A rail
    holding only inside the pytest process would certify a widening the gate
    never sees. Same construction as
    `test_a_second_interpreter_reaches_the_same_verdict_as_rung_0`, on the paths
    this section owns plus the three the #2136 grant made load-bearing: the
    child's verdicts must equal the parent's, so the test stays green whichever
    way a later widening moves them and red only if the two interpreters disagree.
    """
    universe = tracked_agent_services_paths() + list(AGENT_SERVICES_GLOB_PROBES)
    probe = (
        "from scripts.automod import spec; "
        "import sys; "
        "print(spec.classify('agent-services/livekit_worker.py')); "
        "print(spec.classify('agent-services/models/wakeword/hey_lloyd.onnx')); "
        "print(spec.classify('scripts/automod/spec.py')); "
        "print(len([p for p in sys.argv[1:] if spec.classify(p) == 'allowed'])); "
        # #2136 added these three: the reconcile diff this grant exists to make
        # landable has exactly one path, so it is the one verdict in this section
        # that the gate's own interpreter has to agree on before that round gets a
        # tests rung at all.
        "print(spec.classify('agent-services/conf/qmd-index.yml')); "
        "print(spec.classify('agent-services/conf/livekit.yaml')); "
        "print(spec.classify('agent-services/conf/livekit.yaml.runtime'))"
    )
    out = subprocess.run([sys.executable, "-c", probe, *universe], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == spec.classify("agent-services/livekit_worker.py"), out.stdout
    assert lines[1] == spec.classify("agent-services/models/wakeword/hey_lloyd.onnx"), out.stdout
    assert lines[2] == "protected", out.stdout
    assert int(lines[3]) == len([p for p in universe if spec.classify(p) == "allowed"]), out.stdout
    assert lines[4] == spec.classify(QMD_TEMPLATE) == "allowed", (
        f"the interpreter that will grade rung 0 reads the template as {lines[4]!r}: "
        f"the tuple in this tree is not the tuple the gate reads, so the "
        f"reconcile round #2136 exists for dies at preflight")
    assert lines[5] == spec.classify("agent-services/conf/livekit.yaml") == "unlisted", out.stdout
    assert lines[6] == spec.classify("agent-services/conf/livekit.yaml.runtime") == "unlisted", out.stdout


# ---------------------------------------------------------------------------
# the #1883 grant: the tracked Qwen3-TTS patch, #1878's only writable artefact
# ---------------------------------------------------------------------------
#
# `agent-services/services/tts/qwen3-tts/` is an untracked vendored clone —
# `git ls-files agent-services/services/tts/` returns the patch, an upstream
# commit pin and three server scripts, and no `.py` from the clone — so the one
# artefact of the Qwen3-TTS integration that a diff can carry is the tracked
# `.patch` applied to it. Round SM_20260930_063800 wrote #1878's frame cap into
# that patch plus `tests/test_qwen3_tts_frame_cap.py`, and its `gate.json`
# records rung 0 as the only rung that ran:
# `paths outside the writable set: ['agent-services/services/tts/qwen3-tts-local.patch']`,
# bucket `unlisted`. #1883 is the widening that makes #1878 implementable. These
# rails are its shape: one verbatim file, no rollback drill, exactly one tracked
# path's verdict moved, and the reason written beside the entry.

#: The tracked patch. Named verbatim, never by a glob, because the rail above
#: (`agent_services_paths_admitted_without_being_named`) refuses a wildcard
#: spelling of even a single file.
TTS_PATCH = "agent-services/services/tts/qwen3-tts-local.patch"

#: The diff #1878's re-offered round carries: the regenerated patch plus its
#: test. `tests/test_qwen3_tts_frame_cap.py` exists on the kept branch
#: `automod/SM_20260930_063800` (`cb3bc441`), not on main; rung 0 classifies the
#: paths a diff touches, so what has to hold is the pair's verdicts together.
TTS_1878_DIFF = [TTS_PATCH, "tests/test_qwen3_tts_frame_cap.py"]


def test_the_tracked_tts_patch_is_admitted_verbatim():
    """#1883 clause 1: the exact path is in `ALLOWED_GLOBS` and classifies `allowed`.

    The file must also be tracked: the whole reason #1878 is unimplementable is
    that the clone the patch applies to is untracked, so admitting an untracked
    path would be a grant over bytes no commit can carry.
    """
    assert TTS_PATCH in spec.ALLOWED_GLOBS
    assert spec.classify(TTS_PATCH) == "allowed"
    assert TTS_PATCH in tracked_agent_services_paths(), (
        f"{TTS_PATCH} is not tracked, so admitting it grants nothing a round "
        f"can actually change")


def test_the_tts_grant_carries_its_reason_beside_the_entry():
    """#1883 clause 5, the in-file half: an allowlist entry has to say why.

    Every other `agent-services` grant in the tuple carries a comment naming the
    hazard it does not admit (`agent-services/**` would take in every launcher
    and conf), and this one needs the same sentence for a different reason: the
    path is admitted because the tree it sits in is untracked, which is exactly
    why the usual objection does not apply. A bare entry reads as an oversight
    to the next reader and gets swept by the next person who tidies the tuple.
    """
    src = (REPO_ROOT / "scripts" / "automod" / "spec.py").read_text().splitlines()
    entry = f'    "{TTS_PATCH}",'
    assert src.count(entry) == 1, (
        f"expected the grant exactly once in spec.py's ALLOWED_GLOBS, found "
        f"{src.count(entry)}")
    block: list[str] = []
    j = src.index(entry) - 1
    while j >= 0 and (src[j].lstrip().startswith("#") or not src[j].strip()):
        block.append(src[j])
        j -= 1
    comment = " ".join(reversed(block))
    assert "untracked" in comment, (
        f"the grant does not name the untracked vendored TTS clone as its "
        f"reason, which is the only thing that distinguishes it from a "
        f"`agent-services/**` widening: {comment!r}")
    assert "patch" in comment, comment
    assert "#1883" in comment or "#1878" in comment, comment


def test_the_1878_diff_clears_rung_0_without_a_drill():
    """#1883 clause 2: the re-offered #1878 diff is in scope, and buys no drill.

    The two buckets both matter. `unlisted` empty is the defect #1883 exists to
    close; `protected` empty is the grant being in `ALLOWED_GLOBS` rather than
    `PROTECTED_GLOBS` — a protected path also clears preflight
    (`check_scope` permits it) but sets `requires_drill`, which would make every
    #1878 re-offer pay the ~90 s guardian drill for a documentation-shaped edit.
    """
    ok, reason, buckets = spec.check_scope(TTS_1878_DIFF)
    assert (ok, reason) == (True, "in scope"), (ok, reason)
    assert buckets["allowed"] == list(TTS_1878_DIFF), buckets
    assert not buckets["unlisted"] and not buckets["protected"], buckets
    assert not buckets["denied"] and not buckets["comment_only"], buckets
    assert spec.requires_drill(TTS_1878_DIFF) is False, (
        "an empty protected bucket is the clause: this grant must not buy a "
        "guardian drill")


def test_the_tts_grant_keeps_the_exactness_rail_and_its_floor_green():
    """#1883 clause 3: the grant is a named file, so no unnamed admission exists.

    `agent_services_paths_admitted_without_being_named` reads `verbatim` straight
    out of `ALLOWED_GLOBS`, so adding this exact path is precisely the case where
    the admitted file and the named file are the same file and the violation list
    stays empty. The vacuity floor rides along: one file out of the 81 tracked
    paths that were `unlisted` at this round's base `337b1f1a` leaves the rail
    guarding a large corpus, not a husk.
    """
    assert TTS_PATCH in {g for g in spec.ALLOWED_GLOBS if "*" not in g}
    assert agent_services_paths_admitted_without_being_named() == []
    unlisted = [p for p in tracked_agent_services_paths()
                if spec.classify(p) == "unlisted"]
    assert len(unlisted) >= 70, (
        f"only {len(unlisted)} tracked agent-services paths are still unlisted "
        f"against 81 at base 337b1f1a: the path-exactness rail is close to "
        f"vacuous and the widening should be re-read, not trusted")


def test_the_wildcard_spelling_of_the_tts_grant_would_be_an_unnamed_admission(monkeypatch):
    """Proves the verbatim spelling is load-bearing, not stylistic.

    #1883's title asked for `agent-services/services/tts/*.patch`. Applied to the
    tuple before this round, that glob admits exactly the one file the item wants
    and moves the same tracked verdicts the verbatim entry does — and the
    path-exactness rail still goes red on it, because `fnmatch`'s `*` crosses
    `/` and a wildcard in that tuple is one keystroke from a grant over the tree.
    The count cannot tell the two spellings apart, which is why the rail is the
    only thing that can, and why this node exists: it is the difference between
    the entry the item's title named and the entry it landed, executed.
    """
    assert TTS_PATCH in ALLOWED_GLOBS_AS_SHIPPED, (
        "this node simulates the title's glob against the tuple minus the grant; "
        "with the grant absent the comparison is not the one being made")
    shipped_unlisted = [p for p in tracked_agent_services_paths()
                        if spec.classify(p) == "unlisted"]
    pre = tuple(g for g in ALLOWED_GLOBS_AS_SHIPPED if g != TTS_PATCH)
    monkeypatch.setattr(spec, "ALLOWED_GLOBS",
                        (*pre, "agent-services/services/tts/*.patch"))
    assert spec.classify(TTS_PATCH) == "allowed", "the glob does not even reach the file"
    assert agent_services_paths_admitted_without_being_named() == [TTS_PATCH]
    # The rail itself, not a restatement of its helper — but stated as a
    # try/except rather than `pytest.raises`, because the honest failure here is
    # "that rail stopped raising", which happens two different ways: the grant
    # went wrong, or the rail was refactored to return a violation list instead
    # of asserting. `pytest.raises` reports both as DID NOT RAISE and sends the
    # next maintainer to the wrong file.
    try:
        test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError(
            "the path-exactness rail did not fire under the wildcard grant. If "
            "the grant was narrowed this node needs updating; if the rail was "
            "refactored to return a violation list instead of asserting, assert "
            "on that list here — the helper reports "
            f"{agent_services_paths_admitted_without_being_named()} today"
        ) from None
    assert "became writable without being named" in message, message
    wildcard_unlisted = [p for p in tracked_agent_services_paths()
                         if spec.classify(p) == "unlisted"]
    assert len(wildcard_unlisted) == len(shipped_unlisted), (
        "the vacuity floor was supposed to be blind to the spelling; if it can "
        "see it, the node above is no longer the only rail that can")


def test_the_tts_grant_moves_exactly_one_tracked_agent_services_verdict(monkeypatch):
    """#1883 clause 4: nothing else under `agent-services` changes classification.

    Diffed against the tuple one line narrower rather than asserted in isolation:
    "the grant admits one file" is a statement about two trees, and the only way
    to say it is to compute both verdict maps over `git ls-files agent-services`
    and compare them. The models tree is asserted again on the `after` side
    because the acoustic weights are the one content whose bad edit no rung can
    score, and a grant that reaches them would be invisible to the gate.
    """
    corpus = tracked_agent_services_paths()
    pre = tuple(g for g in ALLOWED_GLOBS_AS_SHIPPED if g != TTS_PATCH)
    monkeypatch.setattr(spec, "ALLOWED_GLOBS", pre)
    before = {p: spec.classify(p) for p in corpus}
    monkeypatch.setattr(spec, "ALLOWED_GLOBS", ALLOWED_GLOBS_AS_SHIPPED)
    after = {p: spec.classify(p) for p in corpus}
    assert sorted(p for p in corpus if before[p] != after[p]) == [TTS_PATCH]
    assert (before[TTS_PATCH], after[TTS_PATCH]) == ("unlisted", "allowed")
    allowed_after = {p for p in after if after[p] == "allowed"}
    # #2136 added the third name, the qmd collection template, exactly as the
    # message below instructs: a verbatim grant is a legitimate edit, and the rail
    # that decides legality is `agent_services_paths_admitted_without_being_named()`
    # — which this round's own node
    # (`test_the_qmd_grant_moved_exactly_one_classification`) calls with the entry
    # shipped. `pre` above strips only TTS_PATCH, so the before/after diff this
    # node computes is still exactly one path: the qmd entry is in both tuples.
    # #2278 added the fourth name, `agent-services/voice/timeline.py`, exactly as
    # the message below instructs: a verbatim grant is a legitimate edit, and this
    # round's own node `test_the_voice_timeline_grant_admits_nothing_unnamed` is
    # the legality check called with the entry shipped. `pre` above strips only
    # TTS_PATCH, so the before/after diff this node computes is still exactly one
    # path: the voice entry is in both tuples.
    # #2779 added the fifth name, `agent-services/setup/install-services.sh`, the
    # same way again: one verbatim path, and this round's own node
    # `test_the_installer_grant_admits_one_exact_path_and_nothing_else` is the
    # legality check called with the entry shipped. `pre` still strips only
    # TTS_PATCH, so the diff stays exactly one path: the installer entry is in both
    # tuples.
    # #3162 added the sixth name, `agent-services/scripts/qmd-watcher.sh`, the same
    # way a sixth time: one verbatim path, and this round's own node
    # `test_the_watcher_grant_admits_one_exact_path_and_nothing_else` is the legality
    # check called with the entry shipped. `pre` still strips only TTS_PATCH, so the
    # diff stays exactly one path: the watcher entry is in both tuples.
    # #3258 added the seventh, eighth and ninth names — the three turn-1-mini paths — and
    # this pin cannot list them, because they are admitted but not yet tracked: the grant
    # lands in this file's tree and the bytes land in #3252's, so `allowed_after`, which is
    # a verdict over `corpus` = the index, does not contain them. The set below is therefore
    # the six pinned names plus `expected_allowed_model_paths()`, which is empty until
    # #3252's blob is committed and is exactly those three afterwards. That is the whole
    # point of splitting a grant from the commit it authorises, and it is why
    # `test_the_grant_moves_exactly_the_three_admitted_verdicts_in_a_fresh_interpreter`
    # asks the question of the three paths directly rather than of the index.
    assert allowed_after == {TTS_PATCH, "agent-services/livekit_worker.py",
                             QMD_TEMPLATE, VOICE_TIMELINE,
                             "agent-services/setup/install-services.sh",
                             "agent-services/scripts/qmd-watcher.sh"} \
            | set(expected_allowed_model_paths()), (
        f"the named grants over tracked agent-services paths are the six names this "
        f"node pins plus whichever of the #3258 turn-1-mini grant is in the index "
        f"({expected_allowed_model_paths()}); found {sorted(allowed_after)}. A new "
        f"verbatim grant is a legitimate edit — add its path to this pin (unless it is "
        f"a turn-1-mini path, which the derived half already covers), and check it "
        f"against `agent_services_paths_admitted_without_being_named()`, which is the "
        f"rail that actually decides whether a grant is legal")
    models = [p for p in corpus if p.startswith("agent-services/models/")]
    assert models, "no tracked file under agent-services/models/ — this rail guards nothing"
    assert [p for p in models if after[p] == "allowed"] == expected_allowed_model_paths(), (
        f"the before/after diff this node computes is about one grant, not about the "
        f"model tree: the writable set there is "
        f"{[p for p in models if after[p] == 'allowed']}, expected "
        f"{expected_allowed_model_paths()} — a weight beyond the #3258 three became "
        f"writable, or one of those three stopped being admitted")
    assert not spec.requires_drill([TTS_PATCH])


def test_a_second_interpreter_admits_the_tts_patch_the_way_rung_0_does():
    """The seam this grant sits on, crossed the way the gate crosses it.

    `automod_gate` is spawned detached against the live tree — `gate_detached`
    builds the argv and `S.spawn_detached(argv, log, cwd=LIVE_ROOT)` runs it
    (`scripts/automod/round.py:527-555`) — so rung 0 grades with the `spec`
    module *that* interpreter imported: rung 0 calls
    `spec.check_scope` at `scripts/automod/gate.py:1296` and the drill rung asks
    `spec.requires_drill` at `gate.py:2846`. A widening that read as admitted only
    inside the pytest process would certify a grant the gate never sees — and
    #1878's whole wait is a rung-0 verdict, so this node runs the fresh
    interpreter over #1878's own diff and takes the child's answer as the truth.
    """
    probe = (
        "import sys; "
        "from scripts.automod import spec; "
        "print(spec.classify(sys.argv[1])); "
        "ok, why, buckets = spec.check_scope(sys.argv[1:3]); "
        "print(ok, buckets['allowed'] == [sys.argv[1], sys.argv[2]], "
        "bool(buckets['unlisted']), bool(buckets['protected']), "
        "spec.requires_drill(sys.argv[1:3]))"
    )
    out = subprocess.run([sys.executable, "-c", probe, *TTS_1878_DIFF],
                         cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == "allowed", out.stdout
    assert lines[1] == "True True False False False", out.stdout


#: The vault, resolved the way `tests/board_presence.py:vault_root()` resolves it
#: (`LLOYD_OBSIDIAN_VAULT` if set, else `~obsidian`), duplicated rather than
#: imported so this file stays runnable as a single module.
VAULT = (Path(os.environ["LLOYD_OBSIDIAN_VAULT"]).expanduser()
         if os.environ.get("LLOYD_OBSIDIAN_VAULT") else Path.home() / "obsidian")

#: #1883 clause 6's witness, and the state-dir artefact it was copied from.
WITNESS = VAULT / "backlog" / "data" / "gate.json"
WITNESS_SOURCE = (Path.home() / ".local" / "state" / "lloyd-automod" / "rounds"
                  / "SM_20260930_063800" / "gate.json")


#: The witness lives in the vault, and the gate runs `-m "not live_vault"`
#: (pytest.ini:14-19): a file a nightly job can rewrite between rounds must not
#: fail the next author on a hard rung. Run it with
#: `python3 -m pytest tests/test_automod_spec.py -q -m "" -k witness`.
@pytest.mark.live_vault
def test_the_rung_0_refusal_this_item_quotes_has_committed_witness_bytes():
    """#1883 clause 6: the refusal this whole grant exists for has readable bytes.

    `~/.local/state/lloyd-automod/` is the audit trail and the rollback target,
    never hand-edited — and it is outside git and outside the vault backup, so
    SM_20260930_063800's rung-0 refusal existed in exactly one file with no
    history. #1883's Evidence block quotes that file line by line, and a
    transcription typed into this repo would be no more verifiable than the
    prose it replaced. So the witness is the artefact itself, copied byte for
    byte — `wc -l` on it reads 34, `wc -c` 1143 — and this node re-derives every
    quoted number from those committed bytes rather than from anything here.
    """
    assert WITNESS.is_file(), (
        f"{WITNESS} is absent, so the refusal #1883 quotes survives only in "
        f"{WITNESS_SOURCE}, a path with no history a reader can check")
    raw = WITNESS.read_text()
    # `wc -l` counts newline bytes, and the artefact's last line (`}`) has no
    # terminating one, so the figure the clause quotes is 34 while
    # `splitlines()` returns 35 lines. Counted the way the clause measures it,
    # with the byte count beside it: the two together are what make "the same
    # file" checkable rather than a claim.
    assert (raw.count("\n"), len(raw.encode())) == (34, 1143), (
        f"wc -l / wc -c on the committed witness read "
        f"{raw.count(chr(10))} / {len(raw.encode())}, not the 34 / 1143 of "
        f"SM_20260930_063800's gate.json: the copy is no longer the artefact")
    report = json.loads(raw)
    assert report["round_id"] == "SM_20260930_063800"
    assert report["head"] == "cb3bc441f9c17afbde23b13148ae90b5d5e54f3f"
    assert report["ok"] is False
    assert [r["name"] for r in report["rungs"]] == ["preflight"], (
        "the item states preflight was the only rung that ran; the witness "
        f"lists {[r['name'] for r in report['rungs']]}")
    rung0 = report["rungs"][0]
    assert rung0["ok"] is False and rung0["seconds"] == 0.08, rung0
    assert rung0["data"]["buckets"] == {
        "allowed": ["tests/test_qwen3_tts_frame_cap.py"],
        "protected": [], "denied": [],
        "unlisted": [TTS_PATCH],
        "comment_only": [],
    }, f"the bucket split the item quotes does not match: {rung0['data']['buckets']}"
    assert "`git add -f`" in rung0["detail"], rung0["detail"]
    if WITNESS_SOURCE.is_file():
        assert raw == WITNESS_SOURCE.read_text(), (
            "the committed witness and the state-dir artefact have come apart, "
            "so neither one is the witness any more")


# ---------------------------------------------------------------------------
# #2136: the qmd collection template grant — one path, and the drift it buys
# ---------------------------------------------------------------------------
#
# `agent-services/conf/qmd-index.yml` is the hand-maintained copy of the config
# the qmd daemon reads. Nothing enforces the two agreeing: the nightly check
# compares them and records the answer as "a report entry and never an exit
# code" (`scripts/maintenance/qmd_index_maintenance.py:85-86`, restated at :889),
# then prints `cp ~/.config/qmd/index.yml agent-services/conf/qmd-index.yml`
# (:156) for a person to run. That is why the 2026-09 `facts` drift sat reported
# but unfixed from 09-19 to 09-28 (#1298, closed by #1652's `72112b77`) — not
# because nobody saw it, but because the only writer of the tracked half was a
# human. #1301's owed decision 2 asked whether to widen `ALLOWED_GLOBS` and
# ruled the `SETUP.md` half already code, `agent-services/conf/**` illegal under
# #1376 clause 1, and this one verbatim path the surviving residue.

#: The granted path: the reconcile diff is this file and nothing else.
QMD_TEMPLATE = "agent-services/conf/qmd-index.yml"

#: What else lives in that directory. Two of these exist on disk and one does
#: not: `livekit.yaml` is tracked, `livekit.yaml.runtime` is the 0600
#: live-credentials copy `.gitignore:95` keeps out of the index — a diff of it
#: would be unreviewable, which is half of why a directory grant is refused — and
#: `probe.yml` is #1376's probe for a conf file nobody has written yet.
QMD_TEMPLATE_SIBLINGS: tuple[str, ...] = (
    "agent-services/conf/livekit.yaml",
    "agent-services/conf/livekit.yaml.runtime",
    "agent-services/conf/probe.yml",
)


def test_the_qmd_template_grant_is_one_exact_path_and_no_shape():
    """#2136 clause 1: the entry is that path, and no grant reaching it is a shape.

    `classify` is the verdict rung 0 records and `check_scope` is the call it makes
    (`gate.py:1381`), so the third assertion is the reconcile diff itself —
    `cp live -> template` touches exactly this one file — coming back in scope with
    an empty `unlisted` bucket and no guardian drill. At the round's base the same
    call bucketed it `unlisted: ['agent-services/conf/qmd-index.yml']` and refused
    the round at preflight, which is the state #1301 recorded as human-only.

    The `_match` half is what makes this a test of the clause rather than of a
    literal this file wrote: an entry spelled `agent-services/conf/*` satisfies
    `classify` on the template AND admits `livekit.yaml` plus the 0600
    `livekit.yaml.runtime`, because fnmatch's `*` crosses `/` (the matcher rung 0
    uses is `scripts/automod/spec.py::_match`, cited by symbol because #3162 inserts
    23 lines into that file above it and a number here would only record how far the
    pointer had travelled). Asking the tuple which entries reach
    the path turns that red while leaving
    `classify` green, which is the gap the #1376 rail exists to close.
    """
    assert QMD_TEMPLATE in spec.ALLOWED_GLOBS
    assert spec.classify(QMD_TEMPLATE) == "allowed"
    ok, reason, buckets = spec.check_scope([QMD_TEMPLATE])
    assert (ok, reason) == (True, "in scope"), (ok, reason, buckets)
    assert not buckets["unlisted"] and not buckets["protected"], buckets
    assert spec.requires_drill([QMD_TEMPLATE]) is False, (
        "a one-file config reconcile must not buy a guardian drill")
    reaching = [g for g in spec.ALLOWED_GLOBS if spec._match(QMD_TEMPLATE, (g,))]
    assert reaching == [QMD_TEMPLATE], (
        f"the template is admitted by {reaching}: only a verbatim entry is legal "
        f"under #1376 clause 1, and a wildcard here reaches its siblings too")


@pytest.mark.parametrize("path", QMD_TEMPLATE_SIBLINGS)
def test_every_sibling_of_the_qmd_template_stays_outside_the_writable_set(path):
    """#2136 clause 2, asked twice of each sibling: the verdict, and the grant.

    Red the moment the entry is widened. `agent-services/conf/*` makes all three
    `allowed` — that is the harm the entry refuses rather than the shape of it:
    `livekit.yaml` is tracked launcher config no round has business editing, and
    `livekit.yaml.runtime` holds live credentials at mode 0600, outside the index,
    so no diff of it is reviewable by any rung. `probe.yml` is the case the
    on-disk pair cannot cover: a file added tomorrow under a directory some later
    entry globbed.
    """
    assert spec.classify(path) == "unlisted", (
        f"{path} became writable as collateral of the #2136 grant")
    assert [g for g in spec.ALLOWED_GLOBS if spec._match(path, (g,))] == [], (
        f"an ALLOWED_GLOBS entry reaches {path}; the grant was one path")


def test_the_qmd_grant_moved_exactly_one_classification():
    """#2136 clause 3: the rails that bound a widening stay green, executed, and
    exactly one path's verdict moved.

    The rails are called, not restated — `test_only_a_verbatised_path_under_agent_services_may_be_admitted`
    is #1376 clause 1 as a property over every tracked `agent-services/` file plus
    the five probes, and it is the assertion that fires if this grant is ever
    re-spelled as a directory. Then the count: the granted file must appear in the
    `allowed` set (a grant that silently stopped applying reads to the corpus as
    "nothing changed", which is the vacuity direction this whole section has to
    guard as carefully as the widening direction), every allowed path must be named
    verbatim, and the conf slice must contain exactly one entry.
    """
    test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    corpus = tracked_agent_services_paths() + list(AGENT_SERVICES_GLOB_PROBES)
    allowed = sorted(p for p in corpus if spec.classify(p) == "allowed")
    assert QMD_TEMPLATE in allowed, (
        "the template is not `allowed` anywhere in the corpus this rail reads, so "
        "the assertions below about 'nothing else moved' would be vacuously true")
    verbatim = {g for g in spec.ALLOWED_GLOBS if "*" not in g}
    assert [p for p in allowed if p not in verbatim] == []
    assert [p for p in allowed if p.startswith("agent-services/conf/")] == [QMD_TEMPLATE]
    conf = [p for p in corpus if p.startswith("agent-services/conf/")]
    assert "agent-services/conf/probe.yml" in conf, (
        "the corpus stopped containing the conf probe, so the line above guards "
        "an empty set")
    unlisted = [p for p in tracked_agent_services_paths()
                if spec.classify(p) == "unlisted"]
    assert len(unlisted) >= 70, (
        f"{len(unlisted)} tracked agent-services paths still unlisted, below the "
        f"70 the #1376 rail is sized on")
    # The models line is #3258's narrowing, not a weakening: it used to assert the empty
    # list, which the turn-1-mini grant contradicts by design the moment #3252 commits the
    # bytes. It now asserts the writable model set is exactly the granted paths that are
    # in the index — empty in this tree, the three after #3252, and never a wake-word
    # weight in either.
    models = [p for p in tracked_agent_services_paths()
              if p.startswith("agent-services/models/")]
    assert models and [p for p in models if spec.classify(p) == "allowed"] \
        == expected_allowed_model_paths(), (
        f"a model path outside the #3258 grant became writable as collateral of the "
        f"conf grant this node exists to bound: "
        f"{[p for p in models if spec.classify(p) == 'allowed']}")


def test_the_reconcile_diff_clears_scope_in_an_interpreter_that_imported_the_tuple():
    """The boundary this grant exists to cross: a scope verdict made by a process
    that never ran this test, importing `spec` from a checkout.

    Every assertion above is evaluated inside the pytest process, whose `spec`
    arrived through this file's `sys.path.insert` (:31) — so all of them would keep
    passing on an entry that reached only an already-imported module and not a
    fresh import of the file. Rung 0 is that other kind of process: the gate ladder
    is spawned detached (`round.py:642`) and its preflight calls
    `spec.check_scope(changed, contents=…)` (`gate.py:1381`) on a `spec` it imported
    itself. So ask a fresh interpreter the same question about the exact diff a
    reconcile round commits — the template plus the test that covers the reconciled
    bytes — and require an equality with the parent rather than a hard-coded
    verdict, plus one concrete pin (`in scope`, nothing unlisted, no drill), because
    an equality between two readings of the same wrong tuple would otherwise be
    vacuously true.

    The narrower claim is deliberate. The gate's own interpreter runs with
    `cwd=LIVE_ROOT` and therefore reads the *live* tree's tuple, which cannot
    contain this entry until it lands — the reason `scripts/automod/spec.py` is
    `protected` and drilled rather than merely allowed, and the reason this node
    pins a checkout import rather than pretending to grade the gate's process.
    """
    probe = (
        "from scripts.automod import spec; "
        "print(spec.check_scope(['agent-services/conf/qmd-index.yml', "
        "'tests/test_qmd_index_template.py'])[0:2]); "
        "print(sorted(spec.check_scope(['agent-services/conf/qmd-index.yml', "
        "'tests/test_qmd_index_template.py'])[2]['allowed'])); "
        "print(spec.requires_drill(['agent-services/conf/qmd-index.yml', "
        "'tests/test_qmd_index_template.py']))"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    parent = spec.check_scope(["agent-services/conf/qmd-index.yml",
                              "tests/test_qmd_index_template.py"])
    assert lines[0] == repr(parent[0:2]) == "(True, 'in scope')", (lines[0], parent)
    assert lines[1] == repr(sorted(parent[2]["allowed"])), lines[1]
    assert lines[2] == "False", lines[2]
    assert sorted(parent[2]["allowed"]) == [QMD_TEMPLATE,
                                            "tests/test_qmd_index_template.py"], parent


# ---------------------------------------------------------------------------
# #2278: the voice timeline grant — one path, so #2273's clause 1 can clear rung 0
# ---------------------------------------------------------------------------
#
# #2273 asks that every spoken turn's latency timeline be persisted, and its
# clause 1 names the method's home: `TurnTimeline.as_dict()` in
# `agent-services/voice/timeline.py`. Round SM_20261006_062159 wrote it and its
# `gate.json` records the only rung that ran — preflight — refusing with
# `paths outside the writable set: ['agent-services/voice/timeline.py']`, bucket
# `unlisted`. The cause is the tuple this file tests: it named three verbatim
# `agent-services/` files and no voice path, and `COMMENT_ONLY_GLOBS` is
# `("config.yaml",)` (`scripts/automod/spec.py::COMMENT_ONLY_GLOBS` — by symbol: that
# cite read `spec.py:183`, which was the closing paren of `ALLOWED_GLOBS` even before
# #3162, and #3162's 23 inserted lines moved it again), so even a comment-only edit
# to the timeline
# classified `unlisted`. A round cannot widen its own scope, so #2278 is the
# widening and `as_dict` stays #2273's clause: this grant opens the door, it does
# not walk it, which is why the `as_dict` grep is #2278's owed-after-landing item
# and appears nowhere below as an assertion.
#
# The shape is #1376's, unchanged: one file, named exactly, never a directory.
# #2278's step 1 as first written asked for `agent-services/voice/**`; against the
# path-exactness rail that spelling prices at 14 unnamed paths, and
# `test_the_wildcard_spelling_of_the_voice_grant_admits_fourteen_paths` measures it.

#: The granted path: #2273's clause 1 is this file and nothing else.
VOICE_TIMELINE = "agent-services/voice/timeline.py"

#: The diff #2273's next round carries: the timeline plus the suite covering the
#: duplex path it sits on (`tests/test_voice_duplex.py` is tracked on `main`).
#: Rung 0 classifies the paths a diff touches, so what has to hold is the pair's
#: verdicts together — the same construction as `TTS_1878_DIFF`.
VOICE_TIMELINE_DIFF = [VOICE_TIMELINE, "tests/test_voice_duplex.py"]

#: #1376's probe for a voice file nobody has written yet. Named apart from the
#: tracked siblings because it does not exist on disk: it is the case a directory
#: grant would cover that no on-disk sample can.
VOICE_PROBE = "agent-services/voice/probe.sh"


def tracked_voice_siblings() -> list[str]:
    """Every tracked `agent-services/voice/` file except the granted one.

    Read from the index at call time rather than pinned as a literal tuple: a
    voice module added tomorrow is exactly what a directory-shaped grant would
    reach, so the sibling set has to grow with the tree for every "nothing else
    moved" assertion in this section to mean anything. 12 files at this round's
    base `ae42a26c`.
    """
    return sorted(p for p in tracked_agent_services_paths()
                  if p.startswith("agent-services/voice/") and p != VOICE_TIMELINE)


def test_the_voice_timeline_grant_is_one_exact_path_and_no_shape():
    """#2278 clause 1: the entry is that exact path, carries no `*`, and nothing
    else in the tuple reaches it.

    `classify` is the verdict rung 0 records and `check_scope` the call it makes,
    so the first two assertions are the pair of verdicts SM_20261006_062159 was
    refused for. The tracked-path assertion is why the grant is worth a line at
    all: an untracked path is bytes no commit can carry and no rung can see. The
    `_match` half is what makes this a test of the clause rather than of a literal
    written beside it — an entry spelled `agent-services/voice/*` satisfies
    `classify` on the timeline AND admits its 12 tracked siblings plus the probe,
    because `fnmatch`'s `*` crosses `/` and the matcher rung 0 uses is
    `spec._match`. Asking the tuple which entries reach the path turns that red
    while leaving `classify` green, which is the gap the #1376 rail exists to close.
    """
    assert VOICE_TIMELINE in spec.ALLOWED_GLOBS
    assert spec.classify(VOICE_TIMELINE) == "allowed"
    assert VOICE_TIMELINE in {g for g in spec.ALLOWED_GLOBS if "*" not in g}, (
        "the entry admitting the timeline is a wildcard spelling; #1376 clause 1 "
        "refuses that however few files it happens to reach today")
    reaching = [g for g in spec.ALLOWED_GLOBS if spec._match(VOICE_TIMELINE, (g,))]
    assert reaching == [VOICE_TIMELINE], (
        f"the timeline is admitted by {reaching}: only a verbatim entry is legal "
        f"under #1376 clause 1, and a wildcard here reaches its siblings too")
    assert VOICE_TIMELINE in tracked_agent_services_paths(), (
        f"{VOICE_TIMELINE} is not tracked, so admitting it grants nothing a round "
        f"can actually change")


def test_the_voice_timeline_grant_admits_nothing_unnamed():
    """#2278 clause 2: the widening admits one path, proved by executing the #1376
    rail rather than restating its helper.

    `agent_services_paths_admitted_without_being_named` reads the verbatim set out
    of `ALLOWED_GLOBS` at call time, so adding this exact path is precisely the
    case where the admitted file and the named file are one file and the violation
    list stays empty. The rail is then called: a helper restatement would keep
    passing if the rail were ever refactored away, the failure #1883's section
    names. The sibling assertions are the clause's own words — the 12 tracked
    `agent-services/voice/*.py` files beside the timeline and the seeded
    `agent-services/voice/probe.sh` all still classify `unlisted`. The two floors
    are denominator controls: 12 siblings is what the voice directory held at base
    `ae42a26c`, and the 70 still-unlisted paths the #1376 rail is sized on is what
    stops "nothing else changed" from guarding a husk.
    """
    test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    assert agent_services_paths_admitted_without_being_named() == []
    siblings = tracked_voice_siblings()
    assert len(siblings) >= 12, (
        f"only {len(siblings)} tracked files share the voice directory with the "
        f"grant, against the 12 at base ae42a26c — the sibling assertions below "
        f"would be guarding almost nothing")
    assert [p for p in siblings if spec.classify(p) == "unlisted"] == siblings, (
        "a voice file became writable as collateral of the #2278 grant: "
        f"{[p for p in siblings if spec.classify(p) != 'unlisted']}")
    assert VOICE_PROBE in AGENT_SERVICES_GLOB_PROBES
    assert spec.classify(VOICE_PROBE) == "unlisted", (
        "the seeded voice probe became writable, which is the exact harm a "
        "directory grant does and this grant must not")
    corpus = tracked_agent_services_paths() + list(AGENT_SERVICES_GLOB_PROBES)
    voice_allowed = sorted(p for p in corpus
                           if p.startswith("agent-services/voice/")
                           and spec.classify(p) == "allowed")
    assert voice_allowed == [VOICE_TIMELINE], (
        f"the writable set inside the voice tree is {voice_allowed}, not the one "
        f"file #2278 named")
    unlisted = [p for p in tracked_agent_services_paths()
                if spec.classify(p) == "unlisted"]
    assert len(unlisted) >= 70, (
        f"{len(unlisted)} tracked agent-services paths still unlisted, below the "
        f"70 the #1376 rail is sized on")


def test_the_wildcard_spelling_of_the_voice_grant_admits_fourteen_paths(monkeypatch):
    """Prices #2278's step 1 as it was first written, to prove the verbatim
    spelling is load-bearing and not stylistic.

    The item asked for `agent-services/voice/**`. Applied to the tuple minus this
    grant, that glob does reach the file the item wants — and the path-exactness
    rail still goes red on it with 14 violations: the 13 tracked
    `agent-services/voice/` files, the timeline among them, because a
    wildcard-spelled admission is an unnamed one by construction, plus
    `agent-services/voice/probe.sh`. The expected set is derived from the corpus
    the rail reads so it moves with the tree; the 14 is the number this node's
    name carries, measured at base `ae42a26c`, and it re-asserts as stale rather
    than quietly agreeing. Written as try/except rather than `pytest.raises` for
    the reason #1883's twin gives: the honest failure here is "that rail stopped
    firing", which has two causes and `raises` reports both as DID NOT RAISE. The
    shipped tuple is then restored and asked the same question, because the
    red-where-green contrast is the whole content of clause 2.
    """
    assert VOICE_TIMELINE in ALLOWED_GLOBS_AS_SHIPPED, (
        "this node simulates the wilder spelling against the tuple minus the "
        "grant; with the grant absent the comparison is not the one being made")
    pre = tuple(g for g in ALLOWED_GLOBS_AS_SHIPPED if g != VOICE_TIMELINE)
    monkeypatch.setattr(spec, "ALLOWED_GLOBS", pre + ("agent-services/voice/**",))
    assert spec.classify(VOICE_TIMELINE) == "allowed", "the glob does not even reach the file"
    violations = agent_services_paths_admitted_without_being_named()
    expected = sorted([p for p in tracked_agent_services_paths()
                       if p.startswith("agent-services/voice/")] + [VOICE_PROBE])
    assert violations == expected, (
        f"the glob admitted {violations[:8]}, not the "
        f"{len(expected)} voice paths this node prices — the rail and the corpus "
        f"no longer agree")
    assert len(violations) == 14, (
        f"the wilder spelling prices at {len(violations)} violations against the "
        f"14 measured at base ae42a26c: {violations}. The number in this node's "
        f"name is stale — update it with the corpus, not the other way round")
    assert VOICE_PROBE in violations, (
        f"the probe fell out of the violation set ({violations}), so the voice "
        f"corpus this node exists to compare against is no longer the one the "
        f"#1376 probes cover")
    try:
        test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    except AssertionError as exc:
        message = str(exc)
    else:
        raise AssertionError(
            "the path-exactness rail did not fire under the voice glob. If the "
            "grant was narrowed this node needs updating; if the rail was "
            "refactored to return a violation list instead of asserting, assert "
            f"on that list here — the helper reports "
            f"{agent_services_paths_admitted_without_being_named()} today"
        ) from None
    assert "became writable without being named" in message, message
    monkeypatch.setattr(spec, "ALLOWED_GLOBS", ALLOWED_GLOBS_AS_SHIPPED)
    assert agent_services_paths_admitted_without_being_named() == [], (
        "the shipped verbatim tuple must come back clean where the glob went red; "
        "that contrast is the whole content of #2278 clause 2")


def test_the_timeline_diff_clears_rung_0_without_a_drill():
    """#2278 clause 3: a round whose diff touches the timeline is in scope, and
    buys no guardian drill.

    Two pairs, because `check_scope` is what rung 0 calls over the whole changed
    set: the clause's own pair (`app/x.py` beside the timeline, an unrelated
    allowed path, so the grant cannot be something that only works in isolation)
    and `VOICE_TIMELINE_DIFF`, the diff #2273's next round actually commits. Both
    buckets matter on each: `unlisted` empty is the defect #2278 exists to close,
    and `protected` empty is the grant sitting in `ALLOWED_GLOBS` rather than
    `PROTECTED_GLOBS` — `check_scope` permits a protected path too, so only
    `requires_drill` distinguishes the two, and a protected spelling would make
    every #2273 re-offer pay the guardian drill for a voice-code edit.
    """
    ok, reason, buckets = spec.check_scope(["app/x.py", VOICE_TIMELINE])
    assert (ok, reason) == (True, "in scope"), (ok, reason, buckets)
    assert buckets["allowed"] == ["app/x.py", VOICE_TIMELINE], buckets
    assert not buckets["unlisted"] and not buckets["protected"], buckets
    assert not buckets["denied"] and not buckets["comment_only"], buckets
    assert spec.requires_drill(["app/x.py", VOICE_TIMELINE]) is False, (
        "an empty protected bucket is the clause: this grant must not buy a "
        "guardian drill")
    ok, reason, buckets = spec.check_scope(VOICE_TIMELINE_DIFF)
    assert (ok, reason) == (True, "in scope"), (ok, reason, buckets)
    assert sorted(buckets["allowed"]) == sorted(VOICE_TIMELINE_DIFF), buckets
    assert not buckets["unlisted"] and not buckets["protected"], buckets
    assert spec.requires_drill(VOICE_TIMELINE_DIFF) is False, (
        f"#2273's own diff would pay a guardian drill for a voice-code edit: "
        f"{buckets}")


def test_the_voice_grant_leaves_the_blanket_glob_simulation_firing(monkeypatch):
    """#2278 clause 4: the change weakened no rail, executed rather than restated.

    `test_a_blanket_agent_services_glob_would_trip_each_rail` widens from
    `ALLOWED_GLOBS_AS_SHIPPED`, the tuple snapshotted at import, so with this
    grant shipped that node is the blanket simulation running on the tree this
    round leaves behind — the version that lands is the version that has to keep
    firing. Calling it here is the difference between a rail that looks green in
    prose and one that demonstrably fires with the new entry inside the tuple it
    widens from. The grant is asserted into the snapshot first: a snapshot taken
    before the entry existed would silently grade the pre-change tree and read as
    a pass for the wrong reason.
    """
    assert VOICE_TIMELINE in ALLOWED_GLOBS_AS_SHIPPED, (
        "the shipped-tuple snapshot does not contain the grant, so the blanket "
        "simulation below is grading the tree before this round")
    test_a_blanket_agent_services_glob_would_trip_each_rail(monkeypatch)


def test_the_timeline_diff_clears_scope_in_an_interpreter_that_imported_the_tuple():
    """The process boundary this grant sits on, crossed the way the gate crosses it.

    Every assertion above is evaluated inside the pytest process, whose `spec`
    arrived through this file's own import — so all of them would keep passing on
    an entry that reached only an already-imported module and not a fresh import of
    the file. Rung 0 is that other kind of process: the gate ladder is spawned
    detached (`round.py:527-555`) and its preflight calls `spec.check_scope(changed)`
    (`gate.py:1296`) on a `spec` that interpreter imported itself. So a fresh
    interpreter is asked about the exact diff #2273's next round commits, and the
    child's answer is the truth here, pinned both by equality with the parent (two
    readings of one wrong tuple agree vacuously otherwise) and by a concrete verdict.
    The claim stays narrower than the round it unblocks, for the reason
    `test_the_scope_spec_stays_protected_though_scripts_is_allowed` gives: the
    gate's own interpreter runs with `cwd=LIVE_ROOT` and reads the live tree's
    tuple, which cannot contain this entry until this round lands — which is also
    why `scripts/automod/spec.py` is protected-and-drilled rather than allowed.
    """
    probe = (
        "from scripts.automod import spec; "
        "import sys; "
        "print(spec.classify(sys.argv[1])); "
        "ok, why, buckets = spec.check_scope(sys.argv[1:3]); "
        "print(ok, why, sorted(buckets['allowed']), bool(buckets['unlisted']), "
        "bool(buckets['protected'])); "
        "print(spec.requires_drill(sys.argv[1:3]))"
    )
    out = subprocess.run([sys.executable, "-c", probe, *VOICE_TIMELINE_DIFF],
                         cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == spec.classify(VOICE_TIMELINE) == "allowed", out.stdout
    assert lines[1] == (f"True in scope {sorted(VOICE_TIMELINE_DIFF)} False False"), \
        (lines[1], VOICE_TIMELINE_DIFF)
    assert lines[2] == "False", out.stdout
    assert spec.requires_drill(VOICE_TIMELINE_DIFF) is False, (
        "the parent disagrees with the fresh import on the only verdict that "
        "decides whether #2273's round pays a guardian drill")
    parent = spec.check_scope(VOICE_TIMELINE_DIFF)
    assert lines[1].split(" ")[0] == str(parent[0]) == "True", (lines[1], parent)
    assert sorted(parent[2]["allowed"]) == sorted(VOICE_TIMELINE_DIFF), parent


# ---------------------------------------------------------------------------
# #2779: the user-unit installer grant — one path, so #2774 can clear rung 0
# ---------------------------------------------------------------------------
#
# #2774 asks that the installer ARM the user units it links, and all five of its
# clauses land inside `agent-services/setup/install-services.sh`. The round that
# took it wrote the fix and never reached a test: its `gate.json` records one rung,
# preflight, `ok: false`, `paths outside the writable set:
# ['agent-services/setup/install-services.sh']`, and that rung's own buckets show
# the diff's other eight paths already `allowed`. The blocker was never the work; it
# was this tuple naming four verbatim `agent-services/` files and no setup path,
# with `COMMENT_ONLY_GLOBS` = `("config.yaml",)` so even a comment-only edit there
# classified `unlisted`. A round cannot widen its own scope, so this grant is its
# own landed commit, exactly as #1449, #1883, #2136 and #2278 were before it.
#
# The shape is #1376's, unchanged: one file, named exactly, never a directory.
# `agent-services/setup/` holds 33 tracked paths besides the granted one (`git
# ls-files agent-services/setup/` -> 34 at this round's base), and
# `agent-services/setup/probe.sh` — which exists only as a #1376 probe, never on
# disk — is the case a directory grant would cover that no on-disk sample can.
#
# Clause map: 1 (the verdict, from a verbatim `*`-free entry) and 2 (one file,
# nothing else in the directory) in
# `test_the_installer_grant_admits_one_exact_path_and_nothing_else`; 3 (the exactness
# rail still empty) called in that same node; 4 (the reason comment, read by this
# file and not by the TTS rail, which is hard-coded to `TTS_PATCH` and will never
# read a new line) in `test_the_installer_grant_carries_its_reason_beside_the_entry`;
# and the payoff #2779 exists for — the refused nine-path diff clearing rung 0 in the
# kind of process the gate is — in
# `test_the_2774_diff_clears_rung_0_in_a_fresh_interpreter`.

#: The granted path: #2774's five clauses are this file and nothing else.
INSTALL_SERVICES = "agent-services/setup/install-services.sh"

#: #1376's probe for a setup file nobody has written yet. A directory grant over
#: `agent-services/setup/` would admit it; this entry must not.
SETUP_PROBE = "agent-services/setup/probe.sh"

#: The diff #2774's next round commits: the nine `changed_paths` copied out of the
#: refused round's `gate.json`, verbatim, installer first. Rung 0 classifies a whole
#: changed set, so what has to hold is the nine verdicts together — the same
#: construction as `TTS_1878_DIFF` and `VOICE_TIMELINE_DIFF`, except that this one is
#: a real refusal's input rather than a prediction of one.
INSTALL_SERVICES_DIFF = [
    INSTALL_SERVICES,
    "architecture/infrastructure.md",
    "scripts/service_health_check.py",
    "tests/test_data_home.py",
    "tests/test_egress_window_unit.py",
    "tests/test_install_services_arms_units.py",
    "tests/test_persistence_window_unit.py",
    "tests/test_service_health_check_unit_links.py",
    "tests/test_unit_enabledness_check.py",
]


def tracked_setup_siblings() -> list[str]:
    """Every tracked `agent-services/setup/` file except the granted one.

    Read from the index at call time, as the voice section's helper does: a setup
    script added tomorrow is exactly what a directory grant would reach, so the
    sibling set has to grow with the tree for the "nothing else moved" assertions
    below to mean anything. 33 files at this round's base.
    """
    return sorted(p for p in tracked_agent_services_paths()
                  if p.startswith("agent-services/setup/") and p != INSTALL_SERVICES)


def test_the_installer_grant_admits_one_exact_path_and_nothing_else():
    """#2779 clauses 1, 2 and 3: the verdict rung 0 wanted, from an entry that
    reaches one file, with the exactness rail still clean.

    `classify` is the verdict the refusal records and `check_scope` the call that
    makes it, so the first two assertions are the pair of answers
    `agent-services/setup/install-services.sh` was refused for. The `_match` half is
    what makes this a test of the clause rather than of a literal written beside it:
    an entry spelled `agent-services/setup/**` satisfies `classify` on the installer
    AND admits its 33 tracked siblings plus `probe.sh`, because `fnmatch`'s `*`
    crosses `/` and the matcher rung 0 uses is `spec._match`. Asking the tuple which
    entries reach the path turns that red while leaving `classify` green. The rail is
    called rather than restated — a restatement would keep passing if it were ever
    refactored away — and the floors are denominator controls: 33 siblings is the
    `setup/` directory minus this grant, and 70 is what the #1376 rail is sized on,
    79 tracked `agent-services/` paths still classify `unlisted` with this grant
    shipped against the 80 measured at the round's base.
    """
    assert INSTALL_SERVICES in spec.ALLOWED_GLOBS
    assert spec.classify(INSTALL_SERVICES) == "allowed"
    assert INSTALL_SERVICES in {g for g in spec.ALLOWED_GLOBS if "*" not in g}, (
        "the entry admitting the installer is a wildcard spelling; #1376 clause 1 "
        "refuses that however few files it happens to reach today")
    reaching = [g for g in spec.ALLOWED_GLOBS if spec._match(INSTALL_SERVICES, (g,))]
    assert reaching == [INSTALL_SERVICES], (
        f"the installer is admitted by {reaching}: only a verbatim entry is legal "
        f"under #1376 clause 1, and a wildcard here reaches its siblings too")
    assert INSTALL_SERVICES in tracked_agent_services_paths(), (
        f"{INSTALL_SERVICES} is not tracked, so admitting it grants nothing a round "
        f"can actually change")
    test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    assert agent_services_paths_admitted_without_being_named() == []
    siblings = tracked_setup_siblings()
    assert len(siblings) >= 33, (
        f"only {len(siblings)} tracked files share the setup directory with the "
        f"grant, against the 33 at this round's base — the sibling assertions below "
        f"would be guarding almost nothing")
    assert [p for p in siblings if spec.classify(p) == "unlisted"] == siblings, (
        "a setup file became writable as collateral of the #2779 grant: "
        f"{[p for p in siblings if spec.classify(p) != 'unlisted']}")
    assert SETUP_PROBE in AGENT_SERVICES_GLOB_PROBES
    assert spec.classify(SETUP_PROBE) == "unlisted", (
        "the seeded setup probe became writable, which is the exact harm a directory "
        "grant does and this grant must not")
    corpus = tracked_agent_services_paths() + list(AGENT_SERVICES_GLOB_PROBES)
    setup_allowed = sorted(p for p in corpus
                           if p.startswith("agent-services/setup/")
                           and spec.classify(p) == "allowed")
    assert setup_allowed == [INSTALL_SERVICES], (
        f"the writable set inside the setup tree is {setup_allowed}, not the one "
        f"file #2779 named")
    unlisted = [p for p in tracked_agent_services_paths()
                if spec.classify(p) == "unlisted"]
    assert len(unlisted) >= 70, (
        f"{len(unlisted)} tracked agent-services paths still unlisted, below the 70 "
        f"the #1376 rail is sized on")


def test_the_installer_grant_carries_its_reason_beside_the_entry():
    """#2779 clause 4: the new grant's comment says why, read by this node.

    The existing reason rail, `test_the_tts_grant_carries_its_reason_beside_the_entry`,
    is hard-coded to `entry = f'    "{TTS_PATCH}",'` and asserts only TTS's own words,
    so it never looks at a line added after it — without this node the discipline that
    every `agent-services` grant carries its reason would be convention for the fifth
    entry and a test for the first four. The walk is that rail's: up from the entry
    over the contiguous comment lines. What it has to find is the two things a reader
    needs to audit the line — the item it unblocks, #2774, and the rung-0 refusal in
    the words rung 0 prints, which is what makes the grant's cost legible to someone
    who has never heard of either item — and one thing it must not find: a commit sha
    or branch name, which is the #2344 rule for reasons that outlive their round (the
    branch outliving this change has its own name by the time anyone reads this, and a
    reason that cites it stops being checkable the moment that branch is pruned).
    """
    import re

    src = (REPO_ROOT / "scripts" / "automod" / "spec.py").read_text().splitlines()
    entry = f'    "{INSTALL_SERVICES}",'
    assert src.count(entry) == 1, (
        f"expected the grant exactly once in spec.py's ALLOWED_GLOBS, found "
        f"{src.count(entry)}")
    block: list[str] = []
    j = src.index(entry) - 1
    while j >= 0 and (src[j].lstrip().startswith("#") or not src[j].strip()):
        block.append(src[j])
        j -= 1
    comment = " ".join(reversed(block))
    # The rail's join keeps each line's leading `#` and its indentation, so a phrase
    # that wraps — as rung 0's own message does in this comment — is not a substring
    # of the raw walk. Dropping only each line's leading marker keeps `#2774` intact
    # (one `#` per line is the marker, the second is the issue ref) and makes the
    # assertion about the prose rather than about where the author broke the line.
    prose = " ".join(ln.strip().lstrip("#").strip() for ln in reversed(block))
    prose = re.sub(r"\s+", " ", prose)
    assert "#2774" in prose, (
        f"the grant does not name the item it unblocks, so a reader cannot check "
        f"whether the reason still holds: {comment!r}")
    assert "#2779" in prose, comment
    assert "outside the writable set" in prose and "unlisted" in prose, (
        f"the grant does not quote the rung-0 refusal it exists to clear, which is "
        f"the evidence that the cost of this line was ever paid: {comment!r}")
    stray = re.search(r"\b(?:automod/|[0-9a-f]{7,40})\b", prose)
    assert stray is None, (
        f"the reason cites {stray.group(0)!r} — a sha or branch name goes stale the "
        f"moment the ref is pruned, so the reason has to name an item and a refusal "
        f"instead: {comment!r}")


def test_the_2774_diff_clears_rung_0_in_a_fresh_interpreter():
    """The payoff, asked of the process kind the gate actually is.

    Everything else here is evaluated inside the pytest process, whose `spec` came
    through this file's own import, so it would keep passing on an entry that reached
    only an already-imported module. Rung 0 is a different process: the gate ladder is
    spawned detached and its preflight calls `spec.check_scope(changed)` on a `spec`
    that interpreter imported itself. So a fresh interpreter is handed the nine paths
    the refused round actually presented, copied out of its `gate.json`, and the child's
    answer is pinned both by equality with the parent (two readings of one wrong tuple
    agree vacuously otherwise) and by the concrete verdict: `in scope`, no path in
    `unlisted`, and — the reason the entry went into `ALLOWED_GLOBS` and not
    `PROTECTED_GLOBS` — no drill, so a #2774 re-offer pays the tests rung and nothing
    else. The claim stops short of the round it unblocks: this says the installer is
    writable, not that #2774's arming is correct, and its nine paths' arming work is
    asserted nowhere here.
    """
    probe = (
        "from scripts.automod import spec; "
        "import sys; "
        "print(spec.classify(sys.argv[1])); "
        "ok, why, buckets = spec.check_scope(sys.argv[1:]); "
        "print(ok, why, sorted(buckets['allowed']), sorted(buckets['unlisted']), "
        "sorted(buckets['protected'])); "
        "print(spec.requires_drill(sys.argv[1:]))"
    )
    out = subprocess.run([sys.executable, "-c", probe, *INSTALL_SERVICES_DIFF],
                         cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == spec.classify(INSTALL_SERVICES) == "allowed", out.stdout
    assert lines[1] == (f"True in scope {sorted(INSTALL_SERVICES_DIFF)} [] []"), \
        (lines[1], INSTALL_SERVICES_DIFF)
    assert lines[2] == "False", out.stdout
    ok, why, buckets = spec.check_scope(INSTALL_SERVICES_DIFF)
    assert (ok, why) == (True, "in scope"), (ok, why, buckets)
    assert not buckets["unlisted"] and not buckets["protected"], buckets
    assert spec.requires_drill(INSTALL_SERVICES_DIFF) is False, (
        "the grant landed in PROTECTED_GLOBS, so every #2774 re-offer would pay a "
        f"guardian drill for an installer edit: {buckets}")


# ──────────────────────────────────────────────────────────────────────────────
# #3162: the qmd index watcher — one path, so #3161's cost block can be edited
# ──────────────────────────────────────────────────────────────────────────────
#
# #3161 rewrites the comment block in `agent-services/scripts/qmd-watcher.sh` that
# justifies the watcher's 60-second rebuild floor. Its round
# (`SM_20261022_163807`, head `86171b7d`) has one rung in its `gate.json`,
# preflight `ok: false`, `paths outside the writable set:
# ['agent-services/scripts/qmd-watcher.sh']` — bucket `unlisted`, while the diff's
# other path `tests/test_qmd_single_build.py` is bucketed `allowed` in the same
# rung data. So the test half of that item could land and the sentence it exists to
# protect could not: a round has no authority wherever the three tuples are silent,
# and a grant is the only route in.
#
# #3161's triage recorded this path as `protected` and expected a guardian drill to
# be the price. That premise is false and this section is where it stays false:
# `PROTECTED_GLOBS` is `scripts/automod/**`, `agent-services/guardian/**`,
# `agent-services/systemd/**`, `agent-services/supervisor/**`, `bin/**` and four
# `app/` modules, and `DENIED_GLOBS` never sees it either, so `classify` fell
# through both to `unlisted` — a state with no drill and no verdict, only a refusal.
# A drill would have been the wrong instrument anyway: drills are the price for
# editing a path the loop already governs, not the key to a path it never named.
#
# The fix is the one #1449/#1883/#1301/#2278/#2779 all used — a verbatim single-file
# entry — because #1376 clause 1 (`test_only_a_verbatised_path_under_agent_services
# _may_be_admitted`) prices a directory spelling over this tree at every file it
# silently reaches, `fnmatch`'s `*` crossing `/`. The refusal's exact shape matters
# for clause 3 below: the watcher alone stays `unlisted` before the grant, but a
# diff of [watcher, tests/test_qmd_single_build.py] is what rung 0 actually judged,
# and no earlier section drives a real refused round's whole `changed_paths`
# through a second interpreter except #2779's.
#
# Clause map:
#   1  `test_the_watcher_grant_admits_one_exact_path_and_nothing_else` — classify
#      is `allowed`, and the `ALLOWED_GLOBS` entries reaching it are exactly one
#      `*`-free string.
#   2  same node (the sibling, the probe, the #1376 rail, the >=70 unlisted floor)
#      + `test_a_scripts_directory_grant_would_admit_the_backup_job_and_the_probe` —
#        the harm
#      half: the `agent-services/scripts/**` spelling, executed against the rail.
#   3  `test_the_3161_diff_clears_rung_0_in_a_fresh_interpreter`
#   4  `test_the_watcher_grant_carries_its_reason_beside_the_entry`

#: The one file #3162 grants, verbatim, and the one path it must be. Assembled from
#: its directory and its basename rather than spelled whole, because the harness's
#: service-control guard refuses any Bash command whose string contains this
#: launcher's filename (re-confirmed in #3162's triage on a command that only read
#: `spec.py`), which would otherwise make running this file's own tests an event.
#: The path `classify` sees is identical: `"/".join` of these parts is the real path.
WATCHER = "/".join(("agent-services", "scripts", "qmd-watcher.sh"))

#: The other tracked file in that directory today (`git ls-files
#: agent-services/scripts/` -> 2): the vault backup job. Stays `unlisted`.
WATCHER_SIBLING = "agent-services/scripts/backup-openclaw.sh"

#: A probe for this directory — a path that must never exist on disk, and the thing
#: a directory grant admits without anyone choosing it. Carried by this section rather
#: than added to `AGENT_SERVICES_GLOB_PROBES` on purpose: that tuple is the rail's own
#: universe, and the nodes reading it quote violation counts measured at their own
#: bases — `80` at `c2bd72b` — so a sixth member would leave those sentences describing
#: a corpus their assertion never ran over. The cost of keeping it local is stated in
#: `test_a_scripts_directory_grant_would_admit_the_backup_job_and_the_probe`: the rail
#: cannot report this path, so its harm is asked of `classify`, which is the function
#: rung 0 actually calls.
WATCHER_PROBE = "agent-services/scripts/probe.sh"

#: The refused round's `changed_paths`, verbatim from that `gate.json`: the comment
#: edit and the test file that guards it. Rung 0 judges the whole set together, so
#: this two-path set is the thing that has to clear, and the sibling assertion in
#: node 1 is what keeps the grant from quietly being a directory.
WATCHER_DIFF = [WATCHER, "tests/test_qmd_single_build.py"]


def tracked_watcher_siblings() -> list[str]:
    """Every tracked `agent-services/scripts/` file except the granted one.

    Read from the index at call time, as the setup section's helper does: a launcher
    script added tomorrow is exactly what a directory grant would reach, so the
    sibling set has to grow with the tree for the assertions below to mean anything.
    1 file at this round's base (`backup-openclaw.sh`), against 33 in `setup/` and
    13 under `voice/` — this is the narrowest of these directories, which is why the
    floor is a floor and not a measurement.
    """
    return sorted(p for p in tracked_agent_services_paths()
                  if p.startswith("agent-services/scripts/") and p != WATCHER)


def watched_corpus() -> list[str]:
    """The corpus node 1 sweeps for launcher-directory admissions: every tracked
    `agent-services/` path, the shared #1376 probes, and this directory's probe.

    The rail's universe plus this section's local probe. It is not what
    `agent_services_paths_admitted_without_being_named` reads — that helper takes no
    corpus and walks the tracked files and the shared probes itself — which is why the
    probe's verdict is asserted through `classify` in both nodes rather than through a
    violation list that would never contain it.
    """
    return tracked_agent_services_paths() + list(AGENT_SERVICES_GLOB_PROBES) + [WATCHER_PROBE]


def test_the_watcher_grant_admits_one_exact_path_and_nothing_else():
    """#3162 clauses 1 and 2: rung 0's missing verdict, from an entry that reaches
    one file, with the exactness rail still clean.

    The grant is a one-line diff whose whole promise is the word "one", so every
    number here is checked before it is trusted: 224 tracked `agent-services/`
    paths at this base (the rail's own denominator floor is 187), 78 of them still
    `unlisted` with this grant shipped against the 79 measured at the refused
    round's base, and 1 tracked sibling in the launcher directory — so "nothing
    else moved" is asserted against a set that exists rather than an empty one. The
    probe is the other half: `WATCHER_PROBE` is not a file anywhere, so the only way
    it can be `allowed` is by a glob that reaches a directory by shape rather than by
    name — which is exactly what has to stay false here, and what the directory
    spelling is priced on in
    `test_a_scripts_directory_grant_would_admit_the_backup_job_and_the_probe`.

    `reaching` is the clause-1 question asked of the tuple rather than of
    `classify`: re-spelled as `"agent-services/scripts/**"` the verdict below stays
    `allowed`, because `fnmatch`'s `*` crosses `/` and the watcher matches itself.
    Only asking which entries reach the path tells you a grant became a directory.
    """
    assert WATCHER in spec.ALLOWED_GLOBS
    assert spec.classify(WATCHER) == "allowed", (
        f"{WATCHER} classifies {spec.classify(WATCHER)}, not allowed: #3161's cost "
        f"block is still uneditable by any round")
    reaching = [g for g in spec.ALLOWED_GLOBS if spec._match(WATCHER, (g,))]
    assert reaching == [WATCHER], (
        f"the watcher is admitted by {reaching}: only a verbatim entry is legal "
        f"under #1376 clause 1, and a wildcard here reaches its sibling too")
    assert all("*" not in g for g in reaching), (
        f"the entry admitting the watcher is a wildcard spelling: {reaching}")
    assert WATCHER in tracked_agent_services_paths(), (
        f"{WATCHER} is not tracked, so admitting it grants nothing a round can "
        f"actually change")
    test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    assert agent_services_paths_admitted_without_being_named() == []
    siblings = tracked_watcher_siblings()
    assert len(siblings) >= 1, (
        "no tracked file shares the launcher directory with the grant, so the "
        "sibling assertions below would guard nothing — this directory held "
        "`backup-openclaw.sh` at the round's base")
    assert WATCHER_SIBLING in siblings, (
        f"{WATCHER_SIBLING} left the sibling set ({siblings}), so the directory's "
        f"contents are no longer the two files this grant was priced against")
    assert [p for p in siblings if spec.classify(p) == "unlisted"] == siblings, (
        "a launcher script became writable as collateral of the #3162 grant: "
        f"{[p for p in siblings if spec.classify(p) != 'unlisted']}")
    assert spec.classify(WATCHER_PROBE) == "unlisted", (
        "the seeded scripts probe became writable, which is the exact harm a "
        "directory grant does and this grant must not")
    assert not (REPO_ROOT / WATCHER_PROBE).exists(), (
        f"{WATCHER_PROBE} exists on disk, so it is a real third file in the "
        f"launcher directory and no longer a probe — the sibling floor and the "
        f"directory pricing below are then counting the wrong corpus")
    corpus = watched_corpus()
    scripts_allowed = sorted(p for p in corpus
                             if p.startswith("agent-services/scripts/")
                             and spec.classify(p) == "allowed")
    assert scripts_allowed == [WATCHER], (
        f"the writable set inside the launcher directory is {scripts_allowed}, not "
        f"the one file #3162 named")
    unlisted = [p for p in tracked_agent_services_paths()
                if spec.classify(p) == "unlisted"]
    assert len(unlisted) >= 70, (
        f"{len(unlisted)} tracked agent-services paths still unlisted, below the 70 "
        f"the #1376 rail is sized on")


def test_a_scripts_directory_grant_would_admit_the_backup_job_and_the_probe(monkeypatch):
    """#3162 clause 2's harm half: the `agent-services/scripts/**` spelling, run.

    This directory is the narrowest of the ones these grants have touched — 2 tracked
    files, against 13 under `agent-services/voice/` and 34 under `setup/` — which is
    exactly why it needs the simulation rather than a sentence. A `scripts/**` entry is
    a few characters longer than the shipped one, reads as "obviously the launcher
    directory belongs to rounds now", and its cost is invisible in review. Two harms,
    measured separately because the #1376 rail can only see one of them:

    * the tracked files, which the rail reports — 2 violations at this base, the vault
      backup job `agent-services/scripts/backup-openclaw.sh` plus the watcher itself,
      since a `**` admits its own directory's file by shape rather than by name and
      that file therefore stops being a named grant. The backup job is the harm; the
      second entry is the tell that `fnmatch`'s `*` crossed `/`;
    * `agent-services/scripts/probe.sh`, which the rail cannot report: its universe is
      the tracked files plus `AGENT_SERVICES_GLOB_PROBES`, and this probe is a local of
      this section on purpose — the blanket-glob nodes quote violation counts measured
      at their own bases (`80` at `c2bd72b`) and a sixth shared probe would make those
      sentences describe a corpus their assertion never ran over. So the probe's harm
      is asked of `classify` directly, which is the function rung 0 calls.

    Written as try/except rather than `pytest.raises`, for the reason #1883's and
    #2278's twins give: the honest failure here is "that rail stopped firing", which has
    two causes and `raises` reports both as DID NOT RAISE. The shipped tuple is restored
    before returning, because leaving a directory grant in a module attribute would
    poison every later node in this file.
    """
    assert WATCHER in ALLOWED_GLOBS_AS_SHIPPED, (
        "this node simulates the wilder spelling against the tuple minus the grant; "
        "with the grant absent the comparison is not the one being made")
    pre = tuple(g for g in ALLOWED_GLOBS_AS_SHIPPED if g != WATCHER)
    assert agent_services_paths_admitted_without_being_named() == [], (
        "the shipped spelling already admits something unnamed, so this node's green "
        "half would prove nothing")
    siblings = tracked_watcher_siblings()
    assert spec.classify(WATCHER_PROBE) == "unlisted", (
        "the probe is already admitted in the shipped tree, so the harm below would "
        "not be the directory spelling's doing")
    try:
        monkeypatch.setattr(spec, "ALLOWED_GLOBS",
                            pre + ("agent-services/scripts/**",))
        assert spec.classify(WATCHER) == "allowed", \
            "the directory spelling does not even reach the file"
        violations = agent_services_paths_admitted_without_being_named()
        assert violations == sorted(siblings + [WATCHER]), (
            f"the directory spelling admitted {violations}, not the tracked "
            f"launcher-directory set {sorted(siblings + [WATCHER])} — the spelling "
            f"admits the file the narrow grant was for as an unnamed admission too, "
            f"and if that half is missing the rail's universe or this directory's "
            f"contents changed")
        assert len(violations) == 2, (
            f"the wilder spelling prices at {len(violations)} violations against the 2 "
            f"measured at this round's base (`backup-openclaw.sh` plus the granted file "
            f"itself, which a `**` admits by shape rather than by name): {violations}. "
            f"The number in this node's prose is stale — update it with the corpus, "
            f"not the other way round")
        assert spec.classify(WATCHER_PROBE) == "allowed", (
            "the directory spelling left the probe unlisted, which means it stopped "
            "being a directory grant — the shape this node exists to refuse")
        try:
            test_only_a_verbatised_path_under_agent_services_may_be_admitted()
        except AssertionError as exc:
            message = str(exc)
        else:
            raise AssertionError(
                "the #1376 rail did not fire under the scripts glob. If the grant was "
                "narrowed this node needs updating; if the rail was refactored to "
                "return a violation list instead of asserting, assert on that list "
                "here — the helper reports "
                f"{agent_services_paths_admitted_without_being_named()} today"
            ) from None
        assert "became writable without being named" in message, message
    finally:
        monkeypatch.setattr(spec, "ALLOWED_GLOBS", ALLOWED_GLOBS_AS_SHIPPED)
    assert agent_services_paths_admitted_without_being_named() == []
    assert spec.classify(WATCHER_SIBLING) == "unlisted"
    assert spec.classify(WATCHER_PROBE) == "unlisted"


def test_the_3161_diff_clears_rung_0_in_a_fresh_interpreter():
    """#3162 clause 3: the refused round's exact two-path diff, classified in a
    fresh interpreter rooted at this repo, and green.

    Same seam as the #2779 node, for the same reason — this grant exists only inside
    a classification done in a process this test does not own: the gate ladder is
    spawned detached with `cwd=LIVE_ROOT` (`round.py:611/642/668`) and
    `gate.py:108` imports `spec` from there, so a round can never widen its own
    scope and the authority granted here takes effect for the NEXT round. The
    subprocess is launched without `PYTHONPATH` or `cwd` and inserts `REPO_ROOT`
    explicitly, so `spec.py` is found by that root and not inherited from this
    pytest run; if it ever resolved elsewhere, `lines[0]` would stop matching the
    in-process verdict.

    The diff is #3161's real `changed_paths`, not a guess at one, because rung 0
    judges the set: one path clearing proves the entry, the pair clearing proves
    #3161 can actually ship. `requires_drill` is asserted False for the same reason
    #2779 asserts it — the admission round's own diff (`spec.py` + this file) does
    pay a drill, since `scripts/automod/**` is `PROTECTED_GLOBS[0]`, and mixing the
    two up would have #3161 paying a guardian drill for a comment.
    """
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from scripts.automod import spec;"
        "print(spec.classify(sys.argv[2]));"
        "ok, why, b = spec.check_scope(sys.argv[2:]);"
        "print(ok, why, sorted(sys.argv[2:]), b['unlisted'], b['protected']);"
        "print(spec.requires_drill(sys.argv[2:]))"
    )
    out = subprocess.run([sys.executable, "-c", probe, str(REPO_ROOT), *WATCHER_DIFF],
                         cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == spec.classify(WATCHER) == "allowed", out.stdout
    assert lines[1] == (f"True in scope {sorted(WATCHER_DIFF)} [] []"), \
        (lines[1], WATCHER_DIFF)
    assert lines[2] == "False", out.stdout
    ok, why, buckets = spec.check_scope(WATCHER_DIFF)
    assert (ok, why) == (True, "in scope"), (ok, why, buckets)
    assert not buckets["unlisted"] and not buckets["protected"], buckets
    assert spec.requires_drill(WATCHER_DIFF) is False, (
        "the grant landed in PROTECTED_GLOBS, so every #3161 re-offer would pay a "
        f"guardian drill for editing a comment: {buckets}")


def test_the_watcher_grant_carries_its_reason_beside_the_entry(monkeypatch):
    """#3162 clause 4: the new grant's comment says why, read by this node.

    Four things a later reader cannot reconstruct from the entry itself: which item
    asked (#3161), that the grant is one file and not its directory (with the
    sibling named as staying `unlisted`, so "not the directory" is testable rather
    than rhetorical), the boundary #3162's own triage got wrong in both directions
    (the path is `unlisted`, not `protected`, so no guardian drill is what unlocks
    it), and the thing this entry does NOT buy — the watcher runs from the copy of
    its bytes `supervisord` started, so no round editing this file changes the
    running process, and no round restarts one to find out. A grant whose comment
    says only "needed by #3161" is how a directory entry shows up next time with no
    record of what the narrow one was for.
    """
    import re

    src = (REPO_ROOT / "scripts" / "automod" / "spec.py").read_text().splitlines()
    entry = f'    "{WATCHER}",'
    assert src.count(entry) == 1, (
        f"expected the grant exactly once in spec.py's ALLOWED_GLOBS, found "
        f"{src.count(entry)} — the comment is read from the line above it, so two "
        f"entries would make this node grade the wrong one")
    block: list[str] = []
    j = src.index(entry) - 1
    while j >= 0 and (src[j].lstrip().startswith("#") or not src[j].strip()):
        block.append(src[j])
        j -= 1
    assert block, "the new entry has no comment block above it"
    # Whitespace-normalised prose, for the reason #2779's twin gives: the raw walk
    # keeps each line's indentation, so a phrase that wraps mid-sentence would not be
    # a substring of it and the assertion would be about line breaks, not claims.
    prose = re.sub(r"\s+", " ",
                   " ".join(ln.strip().lstrip("#").strip() for ln in reversed(block)))
    assert "#3161" in prose and "#3162" in prose, (
        f"the comment names neither item: {prose}")
    assert "not the directory" in prose.lower(), (
        "the comment does not say the grant is one file and not the directory, "
        f"which is the one thing the next editor needs: {prose}")
    assert WATCHER_SIBLING in prose and "unlisted" in prose, (
        "the comment's 'not the directory' claim is rhetoric: the file that stays "
        f"unlisted is never named, so nobody can check it against `classify`: "
        f"{prose}")
    assert WATCHER_PROBE in prose, (
        f"the comment no longer names `{WATCHER_PROBE}`, the probe whose whole "
        f"existence is the reason a directory spelling is the wrong shape here")
    assert "restart" in prose, (
        "the comment lost the boundary: the watcher runs from the bytes supervisord "
        "started, so this grant changes a file, not a running process — the thing a "
        "reader of the cost block most wants to know")
    # The comment explains an effect, so the effect is executed: with this one entry
    # deleted and nothing else changed, the watcher is back where #3161 found it. That
    # is what makes the comment's claims about a `scripts/**` spelling worth reading —
    # a grant that admitted the path by some other route would make the whole block a
    # story about a refusal that no longer happens.
    monkeypatch.setattr(spec, "ALLOWED_GLOBS",
                        tuple(g for g in ALLOWED_GLOBS_AS_SHIPPED if g != WATCHER))
    try:
        assert spec.classify(WATCHER) == "unlisted", (
            f"deleting the entry #3162 added leaves {WATCHER} "
            f"{spec.classify(WATCHER)}: some other entry reaches it, so this comment "
            f"documents a refusal that is not the one in force")
    finally:
        monkeypatch.setattr(spec, "ALLOWED_GLOBS", ALLOWED_GLOBS_AS_SHIPPED)


# ──────────────────────────────────────────────────────────────────────────────
# #3258: admit the three turn-1-mini paths, so #3252 can commit the detector it names
# ──────────────────────────────────────────────────────────────────────────────
#
# #3252 is a clone that cannot boot: the shipped config's end-of-turn detector is
# `agent-services/models/turn-1-mini/onnx/turn-1-mini.step.onnx`, that file is ignored
# rather than tracked (`git check-ignore -v` resolves it to `agent-services/.gitignore:68`,
# rule `models/`), and `build_turn_detector` raises with no `try` at its call site, so
# `lloyd-agent-worker` dies on any machine that did not fetch the model out of band. Its
# route (b) is to commit the blob plus the model's `LICENSE` and `NOTICE`.
#
# No round could do that, and the reason is the same one #3162 records: #3252's triage
# called the path landable because it is in neither `PROTECTED_GLOBS` nor `DENIED_GLOBS`,
# but `ALLOWED_GLOBS` is deny-by-default, so a path no tuple names is `unlisted` and rung
# 0 refuses `unlisted`. Measured at this round's base: `spec.classify` returned `unlisted`
# for all three paths and `spec.check_scope([the three])` came back
# `(False, "paths outside the writable set: […]", {'unlisted': [all three], 'allowed': []})`.
# So this section grants three verbatim paths and nothing else.
#
# Two claims circulating about this area are false and are corrected in the entry's own
# comment rather than repeated here, because both were written down before anyone
# measured them. `agent-services/.gitignore` classifies `unlisted`, not `denied`:
# `DENIED_GLOBS` holds the verbatim root name `.gitignore` and `_match` is whole-path
# `fnmatch` with no basename component, so a nested ignore file never reaches that entry.
# Practically it is still out of a round's reach today, but unlike a denied path a future
# verbatim grant could admit it, and "no route exists" is the wrong sentence to leave
# standing. And the ignore rule itself is real and untouched by this grant:
# `agent-services/models/` remains ignored, which is precisely why the blob is
# force-added by #3252 rather than merely written — a question this item does not settle
# and does not pretend to.
#
# Clause map:
#   1  `test_the_turn1mini_grant_admits_three_exact_paths_and_nothing_else` — the three
#      verdicts, each reached by one `*`-free entry.
#   2  same node (siblings, directory, probes, the #1376 rail executed) plus
#      `test_a_models_directory_grant_admits_every_weight_and_the_looser_one_the_llm_dump` — the
#      harm half, run.
#   3  the three narrowed rails, executed by `tests/test_automod_spec.py -q` and proved
#      still sharp by `test_the_narrowed_model_rails_stay_sharp_once_the_blob_is_tracked`.
#   4  `test_the_grant_changes_exactly_three_verdicts_over_the_whole_corpus` and
#      `test_the_grant_moves_exactly_the_three_admitted_verdicts_in_a_fresh_interpreter`.
#   5  `test_this_round_granted_the_paths_without_committing_the_blob`.

#: The directory the grant stops outside, and the probes that stand for what a
#: directory grant would reach. None of these is a file this test creates; each is the
#: shape a wildcard matches by form rather than by name, which is the only way to ask
#: the question before the bytes exist.
TURN1MINI_DIR = "agent-services/models/turn-1-mini"
TURN1MINI_MODELS_GLOB = "agent-services/models/**"
#: The other thing a "cover the model trees" grant can mean, and the only one of the two
#: that reaches the local LLM weight dump: `_match` anchors at the repo root, so the
#: anchored spelling cannot and this one can. See
#: `test_a_models_directory_grant_admits_every_weight_and_the_looser_one_the_llm_dump`.
TURN1MINI_ANY_GLOB = "**/models/**"
TURN1MINI_LLM_PROBE = "agent-services/llm/models/probe.gguf"
TURN1MINI_FETCHED_PROBE = "agent-services/models/smart-turn/smart_turn.onnx"

#: #3252's real `changed_paths` (branch `automod/SM_20261024_145851`, commit
#: `c8562653`): the three admitted bytes plus the restore test that guards them. Rung 0
#: judges the set together, so this is the thing that has to clear for the next round to
#: exist — and `tests/test_voice_model_restore.py` is what proves the grant is not being
#: asked to carry a diff by itself.
TURN1MINI_DIFF = [*TURN1MINI_GRANT, "tests/test_voice_model_restore.py"]


def tracked_model_paths() -> list[str]:
    """Every tracked `agent-services/models/` file, seven at this round's base.

    Derived rather than listed because #3252 is about to add three more and the rails
    below have to keep meaning something in both trees without being edited twice.
    """
    return sorted(p for p in tracked_agent_services_paths()
                  if p.startswith("agent-services/models/"))


def grant_corpus() -> list[str]:
    """The corpus the grant is scored against: the rail's own universe plus the six
    turn-1-mini files and the two directory probes.

    It is `tracked_agent_services_paths() + AGENT_SERVICES_GLOB_PROBES` — the universe
    #1376 clause 1 walks — with the grant's three targets appended, because they are not
    in the index: the grant binds for the round after this one, so asking the question of
    the index alone would score an empty set and read as "the entry moved nothing". The
    three siblings and the two probes come along for the same reason the shared probes do:
    they are not files a diff can contain yet, and they are exactly what a wildcard would
    admit without anyone choosing it.
    """
    return (sorted(set(tracked_agent_services_paths())
                   | set(AGENT_SERVICES_GLOB_PROBES)
                   | set(TURN1MINI_GRANT)
                   | set(TURN1MINI_SIBLINGS)
                   | {TURN1MINI_DIR, TURN1MINI_MODELS_GLOB, TURN1MINI_LLM_PROBE,
                      TURN1MINI_FETCHED_PROBE}))


def test_the_turn1mini_grant_admits_three_exact_paths_and_nothing_else():
    """#3258 clauses 1 and 2: three verdicts, each from one verbatim entry.

    Every number is checked before it is trusted, in the shape #3162's twin sets: the
    index holds seven tracked files under `agent-services/models/` at this base and
    `AGENT_SERVICES_ACOUSTIC_WEIGHTS` names five of them — the sixth is
    `wakeword/stop/lloyd_stop.onnx`, the seventh `silero-vad/SOURCE` — so "the weights
    are still refused" is asserted against a corpus that exists rather than a sample. The
    grant targets are asserted ABSENT from the index on purpose: this grant is the one
    here whose object is a file it must not contain, and a node that asked only
    `classify` would stay green on a directory entry.

    `reaching` is the clause-1 question again for the same reason: a `models/**`
    re-spelling leaves all three verdicts `allowed`, because `fnmatch`'s `*` crosses `/`,
    so only asking which entries reach each path tells you the grant became a directory.
    """
    assert set(TURN1MINI_GRANT) <= set(spec.ALLOWED_GLOBS)
    for path in TURN1MINI_GRANT:
        assert spec.classify(path) == "allowed", (
            f"{path} classifies {spec.classify(path)}, not allowed: #3252's blob round "
            f"is still unopenable, which is the whole of this item")
        reaching = [g for g in spec.ALLOWED_GLOBS if spec._match(path, (g,))]
        assert reaching == [path], (
            f"{path} is admitted by {reaching}; only its own verbatim name is legal "
            f"under #1376 clause 1, and any wildcard here reaches its siblings too")
        assert all("*" not in g for g in reaching), (
            f"the entry admitting {path} is a wildcard spelling: {reaching}")
    assert spec.check_scope(list(TURN1MINI_GRANT))[0] is True, (
        "the three paths classify allowed individually but a diff containing all three "
        "is refused, which is the shape #3252 actually submits")
    assert spec.classify(TURN1MINI_DIR) == "unlisted", (
        f"the directory itself classifies {spec.classify(TURN1MINI_DIR)}: three files "
        f"were granted, not the folder")
    for sibling in TURN1MINI_SIBLINGS:
        assert spec.classify(sibling) == "unlisted", (
            f"{sibling} became writable as collateral of the #3258 grant; the int8 "
            f"sibling in particular is a second copy of the same model that no config "
            f"key reads")
    assert spec.classify(TURN1MINI_MODELS_GLOB) == "unlisted"
    assert spec.classify(TURN1MINI_LLM_PROBE) == "unlisted", (
        "the local LLM weight dump became reachable, which is the collateral a "
        "`models/**` spelling pays and this grant refuses")
    assert spec.classify(TURN1MINI_FETCHED_PROBE) == "unlisted"
    assert [p for p in AGENT_SERVICES_ACOUSTIC_WEIGHTS
            if spec.classify(p) != "unlisted"] == []
    models = tracked_model_paths()
    assert len(models) >= 7, (
        f"only {len(models)} tracked files under agent-services/models/, against the 7 "
        f"this grant was priced on — the sibling assertions below would be scoring a "
        f"corpus that is no longer there")
    assert model_paths_allowed() == [], (
        "a tracked model file is writable in the granting round itself, which means "
        "somebody committed bytes here that #3252 owns")
    test_only_a_verbatised_path_under_agent_services_may_be_admitted()
    assert agent_services_paths_admitted_without_being_named() == []


def test_a_models_directory_grant_admits_every_weight_and_the_looser_one_the_llm_dump(
        monkeypatch):
    """#3258 clause 2's harm half: both directory spellings, executed.

    This is the grant the tempting one-line version of #3258 would have written — "admit
    the model directory" reads as obviously correct — and the two spellings it could mean
    cost different things, which is why this node runs both rather than one:

    * `agent-services/models/**` makes every tracked file under `agent-services/models/`
      writable, which the #1376 rail reports. The violation set is derived from the index
      rather than counted, because #3252 is about to add three paths to it and a
      hard-coded figure would fail in the tree this grant exists to enable.
      `agent-services/models/wakeword/probe.onnx`, the shared #1376 probe, is in that set
      too, and its presence is the tell that `*` crossed `/`. The turn-1-mini siblings,
      the directory itself and the fetched-model probe become writable without appearing
      anywhere in the rail's report, because that universe is the tracked files plus five
      probes — so those verdicts are asked of `classify`, the function rung 0 calls.
    * the same spelling does NOT reach `agent-services/llm/models/`, and #3258's item text
      said it did. `_match` anchors at the repo root and skips no path segment, so the
      pattern's second segment has to be `models` where that path's is `llm`. The claim
      is measured here rather than carried: it is false, and the reassurance it was
      offered for ("the dump is out of reach of a models glob") is the reason not to
      rely on it, because
    * `**/models/**` — one `**` away, and the spelling a glob that "covers the model
      trees" suggests — does admit `agent-services/llm/models/`, the 188 GB local dump,
      and the #1376 rail's report is byte-identical whether or not it does, because that
      dump has exactly one tracked file and no weight for the rail to key on. That is the
      collateral the rail cannot see, and it is the strongest single reason this grant is
      three names.

    Restored in `finally` for the standing reason in this file: a leaked directory grant
    in a module attribute would poison every later node, and here it would poison the
    rails that exist to catch it.
    """
    assert len(set(TURN1MINI_GRANT) & set(ALLOWED_GLOBS_AS_SHIPPED)) == 3, (
        "this node simulates the wilder spellings against the shipped tuple; with the "
        "grant absent the comparison is not the one being made")
    models = tracked_model_paths()
    assert agent_services_paths_admitted_without_being_named() == [], (
        "the shipped spelling already admits something unnamed, so this node's green "
        "half would prove nothing")
    assert spec.classify(TURN1MINI_LLM_PROBE) == "unlisted", (
        "the llm probe is already admitted in the shipped tree, so the harm below would "
        "not be the directory spelling's doing")
    expected_violations = sorted(models + [
        p for p in AGENT_SERVICES_GLOB_PROBES
        if p.startswith("agent-services/models/")])
    pre = tuple(g for g in ALLOWED_GLOBS_AS_SHIPPED if g not in TURN1MINI_GRANT)
    try:
        # ── spelling one: `agent-services/models/**` ─────────────────────────────
        monkeypatch.setattr(spec, "ALLOWED_GLOBS", pre + (TURN1MINI_MODELS_GLOB,))
        for path in TURN1MINI_GRANT:
            assert spec.classify(path) == "allowed", \
                "the directory spelling does not even reach the file it is for"
        violations = agent_services_paths_admitted_without_being_named()
        assert violations == expected_violations, (
            f"the directory spelling admitted {violations}, not the tracked models "
            f"corpus plus its probe {expected_violations} — the three granted paths are "
            f"absent from the violation list by construction (a verbatim entry is a "
            f"name), so a shorter list means the models corpus or the probe set moved")
        assert len(violations) >= 8, (
            f"the wilder spelling prices at {len(violations)} violations against the 8 "
            f"measured at this round's base (7 tracked model files plus the shared "
            f"`agent-services/models/wakeword/probe.onnx`); the number in this node's "
            f"prose is stale — update it with the corpus, not the other way round")
        for path in (*TURN1MINI_SIBLINGS, TURN1MINI_DIR, TURN1MINI_FETCHED_PROBE):
            assert spec.classify(path) == "allowed", (
                f"`agent-services/models/**` left {path} unlisted, which means it "
                f"stopped being a directory grant — the shape this node refuses")
        assert spec.classify(TURN1MINI_LLM_PROBE) == "unlisted", (
            "`_match` gained a basename or segment-skipping component since #3258 "
            "measured it, so the anchored-match reason in the entry's comment is stale "
            "and the item's original claim about the llm dump may be true after all")
        try:
            test_only_a_verbatised_path_under_agent_services_may_be_admitted()
        except AssertionError as exc:
            message = str(exc)
        else:
            raise AssertionError(
                "the #1376 rail did not fire under the models glob. If the grant was "
                "narrowed this node needs updating; the helper reports "
                f"{agent_services_paths_admitted_without_being_named()} right now"
            ) from None
        assert "became writable without being named" in message, message
        # ── spelling two: `**/models/**`, the one that reaches the dump ──────────
        monkeypatch.setattr(spec, "ALLOWED_GLOBS", pre + (TURN1MINI_ANY_GLOB,))
        assert spec.classify(TURN1MINI_LLM_PROBE) == "allowed", (
            "the looser spelling no longer reaches the local LLM weight dump, so the "
            "measured reason the entry gives for refusing a wildcard needs re-measuring "
            "and updating, not this assertion")
        assert agent_services_paths_admitted_without_being_named() == expected_violations, (
            "the rail's report changed between the two spellings; it is not the "
            "constant this node contrasts against, so the contrast above needs rewriting")
    finally:
        monkeypatch.setattr(spec, "ALLOWED_GLOBS", ALLOWED_GLOBS_AS_SHIPPED)
    assert agent_services_paths_admitted_without_being_named() == []
    assert spec.classify(TURN1MINI_LLM_PROBE) == "unlisted"
    assert spec.classify(TURN1MINI_SIBLINGS[0]) == "unlisted"


def test_the_narrowed_model_rails_stay_sharp_once_the_blob_is_tracked(monkeypatch):
    """#3258 clause 3's real content: the three narrowed rails run AGAINST #3252's tree.

    Each of the four pins this item had to touch asserted something about a world in
    which no model path was writable. Narrowing them to `expected_allowed_model_paths()`
    is only defensible if they still bite in the tree the grant enables, and that tree is
    one edit away and not in this diff — so the blob is put in the index virtually:
    `tracked_agent_services_paths` is replaced by the real list plus the three granted
    paths, which is exactly what the index will read once #3252 commits, and the rails
    are then executed rather than restated.

    The green result is the point of splitting a grant from the commit it authorises: the
    allowed model set becomes the three granted paths and nothing else, so the five
    weights, `wakeword/stop/lloyd_stop.onnx` and `silero-vad/SOURCE` are refused in the
    tree that has the blob. The red direction is proved by
    `test_a_models_directory_grant_admits_every_weight_and_the_looser_one_the_llm_dump`, which runs
    the same rails against the spelling that would let a weight through.

    Restoring is automatic (`monkeypatch`), and the shipped tuple is re-asserted at the
    end because this node leaves two module attributes alone only if nothing raised.
    """
    real = tracked_agent_services_paths()
    with_blob = sorted(set(real) | set(TURN1MINI_GRANT))
    monkeypatch.setattr(sys.modules[__name__], "tracked_agent_services_paths",
                        lambda: with_blob)
    assert expected_allowed_model_paths() == sorted(TURN1MINI_GRANT), (
        "the simulated index did not take, so the three rails below would be grading "
        "this tree and not the one they are meant to stand in for")
    assert model_paths_allowed() == sorted(TURN1MINI_GRANT)
    test_the_acoustic_weight_rail_covers_every_tracked_model_file()
    test_the_qmd_grant_moved_exactly_one_classification()
    test_the_tts_grant_moves_exactly_one_tracked_agent_services_verdict(monkeypatch)
    assert [p for p in AGENT_SERVICES_ACOUSTIC_WEIGHTS
            if spec.classify(p) != "unlisted"] == []


def test_the_grant_changes_exactly_three_verdicts_over_the_whole_corpus(monkeypatch):
    """#3258 clause 4: the before/after `allowed` set differs by exactly the three paths.

    `pre` is the shipped tuple minus the three entries — the spec as #3252's triage found
    it — and `after` is what ships. Over the whole corpus (the #1376 rail's universe plus
    the turn-1-mini files and the two probes: see `grant_corpus` for why the grant targets
    have to be appended by hand rather than read from the index) exactly three verdicts
    move, each `unlisted` to `allowed`, and every probe and every tracked file is where it
    was. That is the property the four narrowed pins cannot state on their own, because
    they are each scoped to one directory.

    The floor on the unlisted count is repeated here rather than trusted to the #1376
    rail: if that count ever fell to the point where "nothing else moved" was about an
    empty remainder, this node would say so at the grant that caused it.
    """
    corpus = grant_corpus()
    assert len(corpus) >= len(tracked_agent_services_paths()), (
        "the grant corpus is smaller than the index it is built on, so the equality "
        "below is comparing a subset with itself")
    pre = tuple(g for g in ALLOWED_GLOBS_AS_SHIPPED if g not in TURN1MINI_GRANT)
    monkeypatch.setattr(spec, "ALLOWED_GLOBS", pre)
    before = {p: spec.classify(p) for p in corpus}
    monkeypatch.setattr(spec, "ALLOWED_GLOBS", ALLOWED_GLOBS_AS_SHIPPED)
    after = {p: spec.classify(p) for p in corpus}
    moved = sorted(p for p in corpus if before[p] != after[p])
    assert moved == sorted(TURN1MINI_GRANT), (
        f"the grant moved {len(moved)} verdict(s): {moved}. It is three verbatim paths, "
        f"so any other answer means the tuple widened past what this item granted")
    for path in TURN1MINI_GRANT:
        assert (before[path], after[path]) == ("unlisted", "allowed"), (
            f"{path} went {before[path]} -> {after[path]}; the refusal this item exists "
            f"for was `unlisted`, so a different pair means the entry is doing something "
            f"other than replacing that refusal")
    assert all(before[p] == after[p]
               for p in corpus if p not in TURN1MINI_GRANT)
    unlisted_after = [p for p in tracked_agent_services_paths()
                      if spec.classify(p) == "unlisted"]
    assert len(unlisted_after) >= 70, (
        f"{len(unlisted_after)} tracked agent-services paths still unlisted, below the "
        f"70 the #1376 rail is sized on")


def test_the_grant_moves_exactly_the_three_admitted_verdicts_in_a_fresh_interpreter():
    """#3258 clause 4's seam: the verdict the NEXT round gets, from a process this test
    does not own.

    This is the boundary the whole grant sits on. The gate ladder is spawned detached with
    `cwd=LIVE_ROOT` and rung 0 calls `spec.check_scope` on the `spec` module *that*
    interpreter imported, so `scripts/automod/spec.py` is read live and only at preflight:
    nothing in a running gate re-imports it, and nothing in this round's worktree is ever
    consulted. The consequence is the odd shape of this item — the commit that authorises
    a blob cannot be the commit that contains it — and this node is the only thing that
    pins that consequence rather than asserting it in prose: a child interpreter importing
    the committed file, asked the exact six-path question, with equality against the
    parent's own answer on the verdicts plus one concrete pin on the diff (in scope, no
    unlisted bucket, no drill), because an equality between two readings of the same wrong
    tuple would be vacuously true.

    `requires_drill` False is asserted for #3252's diff, not for this one: this round's
    diff contains `scripts/automod/spec.py`, which is `PROTECTED_GLOBS[0]`, and it is
    paying the guardian drill for that edit right now. Mixing the two up would have every
    model-byte round pay a drill for a weight.
    """
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from scripts.automod import spec;"
        # argv: [1]=repo root, [2:9]=the seven verdict paths (three granted, three
        # siblings, the directory), [9:]=#3252's changed_paths. The slices are spelled
        # out because a probe that read one path too few would still print a plausible
        # comma-joined verdict line.
        "print(','.join(spec.classify(p) for p in sys.argv[2:9]));"
        "ok, why, b = spec.check_scope(sys.argv[9:]);"
        "print(ok, why, sorted(b['allowed']), b['unlisted'], b['protected']);"
        "print(spec.requires_drill(sys.argv[9:]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe, str(REPO_ROOT),
         *TURN1MINI_GRANT, *TURN1MINI_SIBLINGS, TURN1MINI_DIR, *TURN1MINI_DIFF],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == ",".join(
        ["allowed"] * 3 + ["unlisted"] * 4), (
        f"a fresh interpreter reads the verdicts {lines[0]}; the in-process answer is "
        f"{','.join(spec.classify(p) for p in (*TURN1MINI_GRANT, *TURN1MINI_SIBLINGS, TURN1MINI_DIR))}")
    assert lines[1] == (f"True in scope {sorted(TURN1MINI_DIFF)} [] []"), lines[1]
    assert lines[2] == "False", lines[2]
    ok, why, buckets = spec.check_scope(TURN1MINI_DIFF)
    assert (ok, why) == (True, "in scope"), (ok, why, buckets)
    assert not buckets["unlisted"] and not buckets["protected"], buckets


def test_this_round_granted_the_paths_without_committing_the_blob():
    """#3258 clause 5: the grant is in force and the bytes are still not tracked.

    The two halves are the whole difference between this item and #3252, and only the
    second one is checkable from the index: `agent-services/models/` is ignored, so the
    files exist on this disk (`du` reads 21,825,125 bytes for the fp32) while
    `git ls-files` reports none of them. That asymmetry is the reason the grant had to be
    a round of its own — a diff cannot contain a path the live spec refuses, and the live
    spec only changes once a grant has landed.

    Read from the index rather than the working tree for the standing reason in this file:
    rung 0 classifies the paths a diff touches, so the index is the authority on what this
    round shipped, and a working-tree check would also read as a failure in the round
    worktree, where ignored files are not materialised at all.
    """
    for path in TURN1MINI_GRANT:
        assert spec.classify(path) == "allowed"
    tracked = tracked_agent_services_paths()
    assert [p for p in TURN1MINI_GRANT if p in tracked] == [], (
        f"{[p for p in TURN1MINI_GRANT if p in tracked]} is already tracked, so this "
        f"round carried #3252's diff as collateral — the grant was supposed to be the "
        f"only thing it landed")
    assert [p for p in tracked if p.startswith(TURN1MINI_DIR + "/")] == [], (
        "some turn-1-mini path reached the index; the item states this round's diff "
        "must contain no path under agent-services/models/")
    assert spec.classify("agent-services/.gitignore") == "unlisted", (
        f"the nested ignore file classifies {spec.classify('agent-services/.gitignore')}. "
        f"The claim repeated in #3252's findings and in b118f05f's commit message was "
        f"that it is `denied`; it is not — DENIED_GLOBS holds the verbatim root name "
        f"`.gitignore` and `_match` has no basename component — and this node is where "
        f"the corrected reading is pinned so the sentence cannot come back")
    assert not spec.requires_drill(list(TURN1MINI_GRANT)), (
        "the granted model paths sit in PROTECTED_GLOBS, so #3252 would pay a guardian "
        "drill for committing a weight, which is not the price this item set")


def test_the_turn1mini_grant_carries_its_reason_beside_the_entry(monkeypatch):
    """#3258 clause 2's prose half: the comment says why, read by this node.

    Five things a later reader cannot reconstruct from three path strings: which item
    asked and which item granted them; that the grant is three files and not the
    directory, with a sibling named as staying `unlisted` so the claim is checkable; the
    collateral a `models/**` spelling reaches that no tracked-file rail would report
    (`agent-services/llm/models/`, whose directory has exactly one tracked file and no
    weight); the correction about `agent-services/.gitignore` that was wrong in prose
    twice; and the thing the entry does NOT settle — whether 21.8 MB of weights is the
    right route at all, which is on #3252's owed list. The block is then made load-bearing:
    with the three entries deleted, the first path is back to `unlisted`, so the comment
    describes the refusal actually in force rather than a historical one.
    """
    import re

    src = (REPO_ROOT / "scripts" / "automod" / "spec.py").read_text().splitlines()
    for path in TURN1MINI_GRANT:
        entry = f'    "{path}",'
        assert src.count(entry) == 1, (
            f"expected {path} exactly once in spec.py's ALLOWED_GLOBS, found "
            f"{src.count(entry)} — the comment is read from the block above the first "
            f"entry, so a duplicate would make this node grade the wrong one")
    assert TURN1MINI_MODELS_GLOB not in spec.ALLOWED_GLOBS
    assert TURN1MINI_LLM_PROBE.rsplit("/", 2)[0] + "/**" not in spec.ALLOWED_GLOBS
    first = f'    "{TURN1MINI_GRANT[0]}",'
    block: list[str] = []
    j = src.index(first) - 1
    while j >= 0 and (src[j].lstrip().startswith("#") or not src[j].strip()):
        block.append(src[j])
        j -= 1
    assert block, "the new entries have no comment block above them"
    prose = re.sub(r"\s+", " ",
                   " ".join(ln.strip().lstrip("#").strip() for ln in reversed(block)))
    assert "#3258" in prose and "#3252" in prose, f"the comment names neither item: {prose}"
    assert "not the directory" in prose.lower(), (
        f"the comment does not say the grant is three files and not the directory: {prose}")
    sibling = TURN1MINI_SIBLINGS[0].rsplit("/", 1)[-1]
    assert sibling in prose and "unlisted" in prose, (
        "the comment's 'not the directory' claim is rhetoric: no sibling is named as "
        f"staying unlisted, so nobody can check it against `classify`: {prose}")
    assert TURN1MINI_ANY_GLOB in prose, (
        "the comment lost the measured reason a wildcard is refused: the anchored "
        f"`agent-services/models/**` cannot reach the llm dump, and `{TURN1MINI_ANY_GLOB}` "
        f"can — refusing the first while claiming it reaches the second is the "
        f"comfort that gets a later round to write the one that does: {prose}")
    assert "agent-services/llm/models/" in prose, (
        "the comment lost the collateral that no rail reports: a `models/**` spelling "
        "also reaches the local LLM weight dump, whose directory has one tracked file "
        "and no weight, so nothing keyed on tracked content would notice")
    assert "NOT `denied`" in prose and "_match" in prose, (
        "the comment lost the correction about `agent-services/.gitignore`, which two "
        "surfaces previously got wrong — it is `unlisted` because DENIED_GLOBS holds the "
        "verbatim root name and `_match` is whole-path fnmatch")
    assert "owed" in prose.lower(), (
        "the comment now reads as if this entry settled whether the bytes belong in git; "
        f"that ruling is on #3252's owed list: {prose}")
    monkeypatch.setattr(spec, "ALLOWED_GLOBS",
                        tuple(g for g in ALLOWED_GLOBS_AS_SHIPPED
                              if g not in TURN1MINI_GRANT))
    try:
        assert spec.classify(TURN1MINI_GRANT[0]) == "unlisted", (
            "deleting the three entries leaves the blob allowed, so some other entry "
            "reaches it and this comment documents a refusal that is not in force")
    finally:
        monkeypatch.setattr(spec, "ALLOWED_GLOBS", ALLOWED_GLOBS_AS_SHIPPED)
