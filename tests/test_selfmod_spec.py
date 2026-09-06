"""Which paths a self-modification round may change.

With no human review tier this module is the entire control surface, so the
ordering property matters more than any individual glob: **denied beats
protected beats allowed**, and a run spec cannot widen its own permissions by
listing a denied path in `writable_paths`.
"""

from __future__ import annotations

import pytest

from scripts.selfmod import spec


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "app/harness/loop.py", "agent_mcp/facts.py", "workers/pool.py",
    "tests/test_x.py", "eval/run_eval.py", "server.py", "autonomy.py",
    "prompt_builder.py", "scripts/memory/kg_rebuild.py",
])
def test_ordinary_code_is_allowed(path):
    assert spec.classify(path) == "allowed"


@pytest.mark.parametrize("path", [
    "scripts/selfmod/gate.py", "scripts/selfmod/spec.py",
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
    "pytest.ini", ".gitignore", "web/src/App.tsx", ".venvs/lloyd/bin/python",
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


# ---------------------------------------------------------------------------
# run spec
# ---------------------------------------------------------------------------

def valid_spec() -> dict:
    return {
        "objective": "make the harness faster",
        "evaluation": {"command": "scripts.selfmod.gate", "timeout_secs": 3600},
        "budget": {"max_rounds": 1, "max_variants_per_round": 1},
        "mutation_scope": {"writable_paths": ["app/**", "tests/**"]},
        "code": {"base_commit": "a" * 40, "branch": "selfmod/SM_1"},
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

    A round that could land a change to `scripts/selfmod/**` by naming it in
    `writable_paths` would remove the only guard permanently, and every
    subsequent round would inherit the weakened gate.
    """
    ok, reason, _ = spec.check_scope(["config.yaml"])
    assert not ok, "an explicit ask must not unlock a denied path"
    assert spec.classify("config.yaml") == "denied"
