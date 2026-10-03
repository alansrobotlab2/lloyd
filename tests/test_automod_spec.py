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


# --- config.yaml: comment-only edits land, value edits never do -------------

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


@pytest.mark.parametrize("after", [
    "a:\n  cap: 401   # a section is <= 162 lines\n",        # a value
    "a:\n  cap: '400'   # a section is <= 162 lines\n",      # quoting changes the type
    "a:\n  cap: 400\n  extra: 1\n",                          # a new key
    "a:\n  cup: 400\n",                                      # a renamed key
    "a:\n  cap: [400\n",                                     # does not parse
])
def test_any_value_change_to_config_yaml_stays_denied(after):
    before = "a:\n  cap: 400   # a section is <= 162 lines\n"
    ok, reason, buckets = _scope_with(before, after)
    assert not ok and buckets["denied"] == ["config.yaml"], reason
    assert "comment-only edit to config.yaml is allowed; this one is not" in reason


def test_the_real_file_with_one_value_flipped_is_denied():
    before = _CFG.read_text(encoding="utf-8")
    after = before.replace("enabled: true", "enabled: false", 1)
    assert after != before, "positive control: the live file has an enabled flag"
    ok, _, buckets = _scope_with(before, after)
    assert not ok and buckets["denied"] == ["config.yaml"]


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

#: The acoustic weights themselves, all tracked
#: (`git ls-files agent-services/models` -> 6 files, these 5 plus a SOURCE note).
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


def tracked_agent_services_paths() -> list[str]:
    """Every file git tracks under `agent-services/`, read from the index.

    The index and not the working tree because rung 0 classifies the paths a diff
    touches (`gate.py:928`, `rung_preflight`), which is exactly what a commit can
    carry there. Measured at `1842b8cf`: 187 files, of which 112 already classify
    `protected` (writable today, with rung 6's drill) and 75 `unlisted`. Re-read
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
    """The five named weights are the whole tracked corpus, not a sample of it.

    Guards the case the parametrised test above cannot see on its own: a model
    file added to the repo that no name in this file mentions.
    """
    models = [p for p in tracked_agent_services_paths()
              if p.startswith("agent-services/models/")]
    assert models, "no tracked file under agent-services/models/ — this rail guards nothing"
    assert set(AGENT_SERVICES_ACOUSTIC_WEIGHTS) <= set(models), (
        f"tracked model files this rail does not name: "
        f"{sorted(set(models) - set(AGENT_SERVICES_ACOUSTIC_WEIGHTS))}")
    assert [p for p in models if spec.classify(p) == "allowed"] == []


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
    assert allowed_after == {TTS_PATCH, "agent-services/livekit_worker.py",
                             QMD_TEMPLATE}, (
        f"the named grants over tracked agent-services paths are exactly the three "
        f"this node pins; found {sorted(allowed_after)}. A new verbatim grant is "
        f"a legitimate edit — add its path to this pin, and check it against "
        f"`agent_services_paths_admitted_without_being_named()`, which is the "
        f"rail that actually decides whether a grant is legal")
    models = [p for p in corpus if p.startswith("agent-services/models/")]
    assert models, "no tracked file under agent-services/models/ — this rail guards nothing"
    assert [p for p in models if after[p] == "allowed"] == []
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
    uses is `spec._match`, spec.py:188-198). Asking the tuple which entries reach
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
    models = [p for p in tracked_agent_services_paths()
              if p.startswith("agent-services/models/")]
    assert models and [p for p in models if spec.classify(p) == "allowed"] == []


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
