"""Which paths a self-modification round may change.

With no human review tier this module is the entire control surface, so the
ordering property matters more than any individual glob: **denied beats
protected beats allowed**, and a run spec cannot widen its own permissions by
listing a denied path in `writable_paths`.
"""

from __future__ import annotations

import inspect
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
    this section owns: the child's verdicts must equal the parent's, so the test
    stays green whichever way a later widening moves them and red only if the two
    interpreters disagree.
    """
    universe = tracked_agent_services_paths() + list(AGENT_SERVICES_GLOB_PROBES)
    probe = (
        "from scripts.automod import spec; "
        "import sys; "
        "print(spec.classify('agent-services/livekit_worker.py')); "
        "print(spec.classify('agent-services/models/wakeword/hey_lloyd.onnx')); "
        "print(spec.classify('scripts/automod/spec.py')); "
        "print(len([p for p in sys.argv[1:] if spec.classify(p) == 'allowed']))"
    )
    out = subprocess.run([sys.executable, "-c", probe, *universe], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == spec.classify("agent-services/livekit_worker.py"), out.stdout
    assert lines[1] == spec.classify("agent-services/models/wakeword/hey_lloyd.onnx"), out.stdout
    assert lines[2] == "protected", out.stdout
    assert int(lines[3]) == len([p for p in universe if spec.classify(p) == "allowed"]), out.stdout
