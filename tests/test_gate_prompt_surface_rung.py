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

import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.automod import gate as G


class _Report:
    def __init__(self, changed):
        self.changed_paths = list(changed)


#: A round's worktree, laid out the way `scripts/automod/worktree.py` lays one
#: out — `…/lloyd-work/<round_id>/home/lloyd`, a sibling of the live checkout.
#: It is deliberately NOT `G.LIVE_ROOT`: a gate handed the same directory as
#: `live` and `worktree` would pass any assertion about which of the two the
#: eval was launched from, which is the assertion #1790 turns on.
ROUND_WORKTREE = (G.LIVE_ROOT.parent / "lloyd-work" / "SM_20260912_120000"
                  / "home" / "lloyd")


def _gate(changed, item_id=None, worktree=None, real_child_env=False):
    """A Gate with only what `rung_prompt_surface` and its trigger read.

    Constructing the real one shells out to git (`GateReport` wants
    `W.head(worktree)`), which the test does not need and the sandbox will not
    always give it.

    `worktree` exists because the rung reads that attribute (#1790), and
    `real_child_env` because two of the nodes below have to inspect the env the
    rung actually hands the eval; the default stub keeps every other node in this
    file from touching the filesystem.
    """
    g = G.Gate.__new__(G.Gate)
    g.report = _Report(changed)
    g.round_id = "SM_20260912_120000"
    g.item_id = item_id
    g.live = G.LIVE_ROOT
    g.worktree = Path(worktree) if worktree else ROUND_WORKTREE
    # The interpreter running this suite, not a literal `.venvs/` path: the
    # witness node below EXECUTES the command this fixture hands back, and a
    # venv path anchored to a tree the suite is not running in does not exist
    # there. The real gate passes its own venv python; the claim under test is
    # argv[1:], the relative script path, and the cwd beside it.
    g.python = Path(sys.executable)
    g._child_env = (
        (lambda root=None, **kw: G.Gate._child_env(g, root=root, **kw))
        if real_child_env else (lambda root=None, **_kw: {}))
    return g


#: Every path the trigger is supposed to fire on, read out of the gate's own
#: tuples rather than re-typed here. #1758: this list was the second copy of the
#: tuple, and the tuple gained `app/prompt_surface.py` while this copy could not
#: notice — a parametrized list that has to be edited in step with the data it
#: describes is the drift, not the check. The vault files are matched by
#: basename, so they are exercised at the path a real diff carries.
SURFACE_PATHS = (list(G.Gate.PROMPT_SURFACE_PATHS)
                 + [f"lloyd/{v}" for v in G.Gate.PROMPT_SURFACE_VAULT])


def test_the_trigger_list_covers_both_kinds_of_path():
    """Deriving the parametrized list from the tuples removed the second copy —
    and left the parametrization with nothing to check if both tuples were
    emptied, since a test parametrized over an empty list simply does not run.

    So the guard asserts something the derivation cannot guarantee: the trigger
    still covers at least one code module AND at least one loaded vault file. Those
    are the two ways a surface reaches the rung (full path, and basename for the
    vault files), and emptying either tuple is exactly the edit that would silent
    the rung for half the surface while the parametrized tests above went green
    on what was left. The count is printed beside it because a denominator is only
    informative next to the number.
    """
    code_paths = [p for p in SURFACE_PATHS if p.endswith(".py")]
    vault_paths = [p for p in SURFACE_PATHS if p.endswith(".md")]
    assert code_paths and vault_paths, (
        f"the prompt-surface trigger covers only "
        f"{'code' if code_paths else 'vault'} paths ({len(SURFACE_PATHS)} total: "
        f"{SURFACE_PATHS}); the other half of the surface is unscored")


@pytest.mark.parametrize("path", SURFACE_PATHS)
def test_the_rung_fires_on_a_prompt_surface_path(path):
    assert _gate([path])._touches_prompt_surface() is True


@pytest.mark.parametrize("path", SURFACE_PATHS)
def test_a_prompt_surface_edit_is_never_answered_by_reuse(path):
    """The reuse rule and the trigger must read one list. `_reuse_rule` used to
    spell the two paths out and match them exactly, so when the modules moved
    from the repo root into `app/` the trigger (which also matches basenames)
    kept firing while the reuse rule stopped recognising the edit — a round
    that changed the prompt would have been handed an earlier head's score."""
    assert G.Gate._reuse_rule("prompt_surface", [path]) is None


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

    ok, msg, data = _gate(["app/prompt_builder.py"]).rung_prompt_surface()
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

    ok, msg, _ = _gate(["app/prefetch.py"]).rung_prompt_surface()
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
    ok, _, data = _gate(["app/prompt_builder.py"]).rung_prompt_surface()
    assert ok is False
    assert data["compare_exit"] == 3


def test_exit_3_is_named_as_instrument_failure_not_a_regression(monkeypatch):
    """The message a round reads when the rung fails is the only place its exit
    codes are explained, so an unnamed 3 reads as just another regression — and
    a regression is what a round is trained to argue past.

    #875 clause 4 gives the control-set movement its own exit reason; this is
    the half of that which reaches the reader. `tests/test_compare_tool_choice.py`
    runs the real script and checks the code; this checks the label.
    """
    def _fake_run(cmd, **kw):
        rc = 0 if "run_tool_choice_eval.py" in " ".join(map(str, cmd)) else 3
        return types.SimpleNamespace(returncode=rc,
                                     stdout="control_correct_rate beyond floor",
                                     stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)

    ok, msg, _ = _gate(["app/prefetch.py"]).rung_prompt_surface()
    assert ok is False
    assert "3=INSTRUMENT FAILURE" in msg
    assert "re-run both sides" in msg
    # 1 stays the code that MEANS a regression, and the label must not blur it.
    assert "1=regression of the change" in msg
    assert "control" in msg.lower()


def test_a_clean_comparison_passes(monkeypatch):
    def _fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout="no regression vs item9",
                                     stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)
    ok, msg, data = _gate(["app/prompt_builder.py"]).rung_prompt_surface()
    assert ok is True
    assert data["compare_exit"] == 0
    assert "no regression" in msg


def test_the_eval_runs_from_the_round_worktree_not_the_live_tree(monkeypatch):
    """The launch cwd is the only thing that decides which `app/` gets imported.

    The command passes the script as a RELATIVE path, and
    `eval/run_tool_choice_eval.py` puts its own
    `Path(__file__).resolve().parent.parent` at the FRONT of `sys.path`, so the
    tree the script resolves into is the tree whose `build_system_prompt()` gets
    scored — the cwd, and nothing else. `_child_env`'s worktree `PYTHONPATH`
    sits behind that insert and loses. Launched from the live tree, as this rung
    did until #1790, every prompt-surface diff was scored against the SHIPPED
    prompt: green over a round that regressed tool choice, red over drift the
    round never caused.

    What the launch used to justify itself with was a layout `6426668b` ended on
    2026-09-22, when the run's baseline lived inside the checkout and
    SM_20260908_165950's went out with that round's worktree. Runtime data has
    since left the tree, the baseline goes to `EVAL_BASELINES_DIR`, and where it
    lands is pinned on its own by
    `test_the_live_data_root_is_named_so_the_baseline_outlives_the_round` instead
    of being assumed from here — which is what lets the launch move.
    """
    cwds = []

    def _fake_run(cmd, **kw):
        cwds.append(kw.get("cwd"))
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)
    gate = _gate(["app/prompt_builder.py"])
    assert gate.worktree != gate.live, (
        "this fixture gave the gate one directory for both roots, so the "
        "assertion below cannot tell which tree the eval was launched from")
    gate.rung_prompt_surface()
    assert cwds and all(c == gate.worktree for c in cwds), cwds


def test_the_child_the_rung_launches_imports_the_worktrees_prompt(tmp_path,
                                                                  monkeypatch):
    """A subprocess witness, not a kwarg check: launch on the pair the rung
    produces and read which `app.prompt_builder` the child resolved.

    The node above can only say the gate passed a path. The claim that matters —
    the one the rung exists for, that a prompt-surface edit is measured on the
    prompt the edit built — is about what the interpreter does with that path, and
    two things can undo it without touching this file: the eval's
    `sys.path.insert` moving behind the inherited `PYTHONPATH`, or the command
    gaining an absolute script path, which would make the cwd inert.

    So the pair is CAPTURED from the launch `rung_prompt_surface` actually makes —
    command, cwd and env alike — rather than rebuilt here, which is what lets this
    node stay green while the method under test goes back to the live checkout.
    The child is then asked where it found the module, in that same cwd and env,
    against two stub trees each holding an `app/prompt_builder.py`: running the
    real eval here would take an hour of model calls to answer a question about
    import precedence. `PYTHONPATH` is then pointed at the tree the child must NOT
    import from, so a green witness means the cwd beat `PYTHONPATH` — precisely the
    precedence the live launch lost.
    """
    gate = _gate(["app/prompt_builder.py"],
                 worktree=str(tmp_path / "round" / "home" / "lloyd"),
                 real_child_env=True)
    captured = []

    def _capture(cmd, **kw):
        captured.append(([str(c) for c in cmd], kw.get("cwd"), kw.get("env")))
        # 2, not 0: the rung stops after the eval, so exactly one launch is
        # issued and the command below is the eval's, with its relative path.
        return types.SimpleNamespace(returncode=2, stdout="", stderr="")
    monkeypatch.setattr(G, "_run", _capture)
    gate.rung_prompt_surface()

    assert len(captured) == 1, (
        f"the rung issued {len(captured)} launches; the witness wants the one "
        "the eval gets")
    cmd, cwd, rung_env = captured[0]
    # The label is the round id because this fixture carries no item id;
    # `test_the_label_names_the_item_when_there_is_one` owns the item form.
    assert cmd[1:] == ["eval/run_tool_choice_eval.py", "--label",
                       "gate-SM_20260912_120000"], cmd
    assert not Path(cmd[1]).is_absolute(), (
        "the launch moved to an absolute script path, which makes the cwd inert "
        "and this whole mechanism dead — the child would resolve `app/` from "
        "wherever the script lives, whichever tree it was launched from")
    assert Path(cwd) == gate.worktree, cwd
    assert Path(rung_env["PYTHONPATH"]) == gate.worktree, (
        "the rung no longer exports the candidate's tree on PYTHONPATH, the other "
        "half of what makes the live launch lose this race")
    assert rung_env.get("LLOYD_DATA"), (
        "the rung stopped naming a data root, which is what keeps the baseline "
        "this launch writes out of the worktree")

    round_tree = tmp_path / "round" / "home" / "lloyd"
    live_tree = tmp_path / "live"
    for tree in (round_tree, live_tree):
        (tree / "app").mkdir(parents=True, exist_ok=True)
        (tree / "app" / "__init__.py").write_text("", encoding="utf-8")
        (tree / "app" / "prompt_builder.py").write_text(
            f"TREE = {str(tree)!r}\n", encoding="utf-8")
    # A linked worktree reads as one: `.git` is a file there, a directory in the
    # main checkout. The witness must not trip app.paths' own worktree test into
    # treating a scratch directory as a checkout of something else.
    (round_tree / ".git").write_text("gitdir: /nowhere/.git/worktrees/w\n",
                                     encoding="utf-8")

    # The rung's own env with the one adversarial change a witness needs:
    # PYTHONPATH pointed at the tree the child must NOT import from. HOME and
    # LLOYD_DATA are the two variables that are not this claim's to carry — the
    # first points at a scratch home the suite also moves, and `app.paths` will
    # not import under it; the second's behaviour is the next node's subject.
    env = {k: v for k, v in rung_env.items() if k not in ("HOME", "LLOYD_DATA")}
    env["PYTHONPATH"] = str(live_tree)
    assert env["PYTHONPATH"] == str(live_tree), "the witness must be adversarial"
    probe = [sys.executable, "-c",
             "import app.prompt_builder as m; print(m.__file__)"]

    from_candidate = subprocess.run(probe, cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=180)
    assert from_candidate.returncode == 0, from_candidate.stderr[-600:]
    resolved = Path(from_candidate.stdout.strip()).resolve()
    assert resolved == (round_tree / "app" / "prompt_builder.py").resolve(), (
        f"launched from the round's worktree the child imported {resolved}, "
        "which is not the candidate's prompt — the rung would be scoring the "
        "shipped tree again")

    # The control, which is the half that makes this a witness and not a
    # tautology: byte-identical command and byte-identical env, with only the cwd
    # moved to the tree the rung used to launch from. PYTHONPATH still names the
    # ROUND's tree, so a child that ordered its paths any other way would import
    # the round's module and this assertion would fail: what moves is exactly what
    # the fix moves.
    from_live = subprocess.run(probe, cwd=live_tree, env=env,
                               capture_output=True, text=True, timeout=180)
    assert from_live.returncode == 0, from_live.stderr[-600:]
    assert Path(from_live.stdout.strip()).resolve() == (
        (live_tree / "app" / "prompt_builder.py").resolve()), (
        "with the live tree as cwd the child did not import the live prompt, so "
        "cwd is not what decides the import and the kwarg changed below fixes "
        f"nothing: {from_live.stdout!r} against {from_candidate.stdout!r}")
    assert Path(from_live.stdout.strip()) != resolved, (
        "both cwds resolved the same module, so nothing here distinguishes the "
        "candidate from the shipped tree")


def test_the_live_data_root_is_named_so_the_baseline_outlives_the_round(monkeypatch):
    """The other half of moving into the worktree, and the reason `live_data`
    stopped meaning "unset".

    Unset was only ever safe because of where the child ran: with the script
    resolving into `LIVE_CHECKOUT`, `app.data_root`'s rule 2 applied and the run
    landed on `~/lloyd-data`. From the worktree the same unset hits rule 3 — "any
    other checkout keeps its data inside itself" — so `EVAL_BASELINES_DIR`
    resolves to `<worktree>/.lloyd-data/eval/baselines/`, the baseline is deleted
    with the round, and `compare_tool_choice.py` then exits 2, "nothing to
    compare against", on a rung that just measured something.

    So the env handed to both evals names the production data root outright.
    `LLOYD_DATA` is therefore present — which is the opposite of what this node
    asserted before #1790, when its subject ran from the live tree and the key
    being ABSENT was what kept the baseline alive. What has not changed is what
    the node is for: the baseline must not be inside either tree. The fork
    variable is still absent, because a round must not inherit a developer's
    fork, and the live root is named instead of inherited.
    """
    envs = []

    def _fake_run(cmd, **kw):
        envs.append(dict(kw.get("env") or {}))
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)
    _gate(["app/prompt_builder.py"], real_child_env=True).rung_prompt_surface()
    assert len(envs) == 2, envs

    # `production_data_root()` is what the gate exports, and it is the answer the
    # gate PROCESS has: `ACCOUNT_HOME` is read at import, and the suite's conftest
    # isolates it to a scratch account dir. Comparing to a fresh interpreter's
    # `Path.home()` instead would demand that a gate run under the suite write its
    # baseline into the developer's real data root — the opposite of isolation.
    # What is pinned here is that the child runs with exactly what the gate named.
    from app.data_root import production_data_root
    live = production_data_root()
    repo = Path(__file__).resolve().parents[1]
    assert (live / "eval" / "baselines") != repo / ".lloyd-data" / "eval" / "baselines"
    for env in envs:
        assert Path(env["LLOYD_DATA"]) == live, (
            "the eval was handed a data root that is not the one the gate names as"
            " production's, so the baseline it writes is either the round's (gone"
            " with the worktree) or a developer's")
        assert "LLOYD_DATA_FORK" not in env

    # Where the baseline actually goes, read out of a child's own `app.paths`
    # under that env. `EVAL_BASELINES_DIR` is what `run_tool_choice_eval.py:497`
    # writes and `compare_tool_choice.py:76-78` reads back, so this is the
    # pairing itself and not a paraphrase of it. The probe runs in THIS tree —
    # the round's real worktree under the gate — because the synthetic
    # `ROUND_WORKTREE` above is a path with no code behind it, and it is a
    # worktree or the main checkout either way, which is the only fact the
    # resolver reads about the tree.
    repo = Path(__file__).resolve().parents[1]
    probe = [sys.executable, "-c",
             "import app.paths as p; print(p.EVAL_BASELINES_DIR); "
             "print(p.IS_WORKTREE)"]

    def _resolve(env):
        out = subprocess.run(probe, cwd=repo, env=dict(env, PYTHONPATH=str(repo)),
                             capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-600:]
        path, is_worktree = out.stdout.strip().splitlines()
        return Path(path), is_worktree == "True"

    named, named_worktree = _resolve(envs[0])
    assert named == live / "eval" / "baselines", named

    # The failure mode the naming exists to prevent, shown rather than asserted
    # away: same child, same tree, `LLOYD_DATA` removed — which is exactly what
    # `live_data` used to hand the eval.
    unset, unset_worktree = _resolve(
        {k: v for k, v in envs[0].items() if k != "LLOYD_DATA"})
    assert unset_worktree == named_worktree
    if named_worktree:
        assert unset == repo / ".lloyd-data" / "eval" / "baselines", unset
        assert unset != named, (
            "with LLOYD_DATA unset a worktree child still resolved production's "
            "baselines, so naming the root buys nothing and the premise of this "
            "node is the layout `6426668b` replaced — re-measure which rule "
            "`app.data_root` applies here")
    else:
        assert unset == named, (
            "the suite is running in the main checkout, where unsetting resolves "
            "production by rule 2; run it under a gate to see the rule-3 branch "
            "this node exists for")


def test_the_label_names_the_item_when_there_is_one(monkeypatch):
    labels = []

    def _fake_run(cmd, **kw):
        parts = [str(c) for c in cmd]
        if "--label" in parts:
            labels.append(parts[parts.index("--label") + 1])
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(G, "_run", _fake_run)

    _gate(["app/prompt_builder.py"], item_id=377).rung_prompt_surface()
    assert labels and all(x == "item377" for x in labels)

    labels.clear()
    _gate(["app/prompt_builder.py"]).rung_prompt_surface()
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
    # ...and the procedure the prompt points at says who does run it, so the
    # model does not assume the check stopped existing. That text lives in
    # the vault skill now (cut 4); `tests/test_prompt_pacing_and_ordering.py`
    # pins it there under the `live_vault` marker.
    assert "automod-change-own-code" in PROMPT
