"""Which paths a self-modification round may change.

With no human review tier this module is the entire control surface, so the
ordering property matters more than any individual glob: **denied beats
protected beats allowed**, and a run spec cannot widen its own permissions by
listing a denied path in `writable_paths`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.automod import spec

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "app/harness/loop.py", "agent_mcp/facts.py", "workers/pool.py",
    "tests/test_x.py", "eval/run_eval.py", "server.py", "autonomy.py",
    "prompt_builder.py", "prompt_surface.py", "scripts/memory/kg_rebuild.py",
])
def test_ordinary_code_is_allowed(path):
    """`prompt_surface.py` joined this list on 2026-09-18 (#1242). It is the
    same shape of file as `prompt_builder.py` — root-level, loaded-prompt
    machinery — but was never enumerated, so item #1069, whose entire fix is
    that module, was unimplementable by any round: three rounds wrote the fix
    and were refused at rung 0 (SM_20260911_190850, SM_20260914_114935,
    SM_20260918_145241)."""
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
    """These may be changed, but only with a live drill. See rung 6."""
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
    out without changing a line of Python — a soft brick no test would catch."""
    assert spec.classify("config.yaml") == "denied"
    assert spec.classify("data/tool_overrides.yaml") == "denied"


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
        ["prompt_surface.py", "tests/test_prompt_surface_guard.py"])
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
    ok, reason, _ = spec.check_scope(["prompt_surface.py", "Makefile"])
    assert not ok and "outside the writable set" in reason


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
        "print(spec.classify('prompt_surface.py')); "
        "print(spec.check_scope(['prompt_surface.py', "
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


@pytest.mark.parametrize("path", ["web/src/App.tsx", "web/src/components/pages/BrowserPage.tsx",
                                  "web/index.html", "web/public/favicon.svg"])
def test_frontend_sources_are_allowed_because_the_frontend_rung_builds_them(path):
    """`web/**` was denied outright until 2026-09-07 because the gate did not
    build the frontend. It does now (rung `frontend`: tsc delta + vite build),
    so the sources are ordinary code; only the build inputs stay denied."""
    assert spec.classify(path) == "allowed"


def test_frontend_tooling_outside_src_is_unlisted_not_allowed():
    assert spec.classify("web/eslint.config.js") == "unlisted"
