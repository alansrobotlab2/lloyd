"""Guards added after turn 20260905_011748_iv84e4 — the looping code review.

That turn: a primary that ran six reformulations of one search, then chased
two symbols that do not exist, then hit a 120s Bash timeout; an observer that
watched 24 tool calls over six minutes without a single LLM judgment, then
spent its whole intervention budget in 88 seconds, one of those injects
asserting a finding the primary had never reported; and a cancel.

Every test here fails against the code as it stood that night. Run:
  .venvs/lloyd/bin/python -m pytest tests/integration/test_iv_loop_guards.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.inner_voice import guards
from app.inner_voice import observer as obs_mod
from app.inner_voice import observer_prompt as prompt_mod
from app.inner_voice.observer import (
    ObserverDecision,
    _apply_decision_guards,
    _fast_path_assistant_message,
    _fast_path_tool_result,
)
from fixture_iv_loop_turn import REAL_BASH_CALLS


# `_persist` records for real; keep the production usage.db clean.
RECORDED: list = []
obs_mod.record_inner_voice_observation = lambda **kw: (RECORDED.append(kw), len(RECORDED))[1]


def _sigs(commands):
    return [guards.tool_call_signature("Bash", {"command": c}) for c in commands]


# ---------------------------------------------------------------------------
# Repetition guard, replayed against the turn it was built for
# ---------------------------------------------------------------------------


def test_repetition_fires_on_the_real_loop():
    """Replay all 28 Bash calls in order; assert where the guard speaks.

    The guard is fed the same bounded ring the pretool hook maintains, and
    the ring is cleared on a fire exactly as the hook clears it — so this is
    the live firing sequence, not a per-call scoring pass.
    """
    ring: list = []
    fired_at: list[int] = []
    for idx, command in REAL_BASH_CALLS:
        ring.append(guards.tool_call_signature("Bash", {"command": command}))
        if len(ring) > 16:
            del ring[:-16]
        verdict = guards.repetition_verdict(ring)
        if verdict is not None:
            fired_at.append(idx)
            ring = []

    # 62 — the 4th grep of the search-reformulation cluster (messages 53-66).
    # 76 — the second pathological cluster, where the primary hunts
    # `build_subliminal_context` / `_messages_subliminal`, a symbol that is
    # never defined anywhere (see the fixture docstring, messages 72-79). The
    # ambient-path tokenisation used through 2026-09-04 could not see this
    # one: `alansrobotlab` padded every identifier set, so the containment
    # ratio was diluted and the real shared symbol never carried a match.
    # Both fires are inside a region the fixture labels pathological, and the
    # 15 healthy exploration calls before them stay silent — see
    # test_repetition_silent_through_healthy_exploration.
    assert fired_at == [62, 76], (
        f"expected fires at the two loop clusters (62, 76); got {fired_at}"
    )
    print("test_repetition_fires_on_the_real_loop: OK")


def test_repetition_silent_through_healthy_exploration():
    """Messages 4-51 are 15 calls of legitimate widening. Never fire there."""
    healthy = [c for i, c in REAL_BASH_CALLS if i <= 51]
    assert len(healthy) == 15
    ring: list = []
    for sig in _sigs(healthy):
        ring.append(sig)
        assert guards.repetition_verdict(ring) is None, sig.preview
    print("test_repetition_silent_through_healthy_exploration: OK")


def test_repetition_names_the_terms_the_primary_kept_chasing():
    loop = [c for i, c in REAL_BASH_CALLS if 53 <= i <= 64]
    verdict = guards.repetition_verdict(_sigs(loop))
    assert verdict is not None
    # The identifiers the primary rewrote its filter around six times.
    assert "iv_cancel_requested" in verdict.shared_terms
    assert "iv_inject_queue" in verdict.shared_terms
    content = guards.repetition_inject_content(verdict)
    assert "iv_cancel_requested" in content
    # The instruction that breaks the loop: an unchanged result is the answer.
    assert "ANSWER" in content
    print("test_repetition_names_the_terms_the_primary_kept_chasing: OK")


def test_repetition_needs_more_than_one_refinement():
    """Narrowing a search once is normal work, not a loop."""
    pair = _sigs([
        'grep -rn "iv_inject_queue" app/',
        'grep -rn "iv_inject_queue" app/ --include="*.py"',
    ])
    assert guards.repetition_verdict(pair) is None
    print("test_repetition_needs_more_than_one_refinement: OK")


# ---------------------------------------------------------------------------
# #523 — scan roots, filenames and filter operands cannot carry a near match
# ---------------------------------------------------------------------------
#
# The surviving false-fire channel measured on 2026-09-08: 75 firings in 37
# sessions whose NAMED terms were `_pipeline` 25, `trajectories` 9,
# `node_modules` 2 — the directories the primary was looking in, inside a
# message asserting it kept chasing one target. Restricting to sessions begun
# after the round-id fix (`5531f21`) left 9 firings in 3 sessions and not one
# naming a round id, so this channel, not that one, is what remains live.
#
# `SCAN_ROOT` is one path in three shapes so the calls really do share it;
# `_pipeline`, `trajectories` and `node_modules` are the tokens the histogram
# named, and `node_modules` arrives as a filter operand because that is the
# shape it appeared in (`--exclude-dir`, `grep -v`).

SCAN_ROOT = "/home/alansrobotlab/lloyd/_pipeline/vault-derived/trajectories"


def test_repetition_ignores_three_greps_sharing_only_a_scan_root():
    """Clause 1: three patterns, one root, no shared hunt — and no fire.

    Every token the three calls share is a directory of the root they were
    pointed at. Before #523 this fired on `('_pipeline', 'trajectories')`,
    because two shared identifiers is all a near match used to need.
    """
    cmds = [
        f"grep -rn iv_cancel_requested {SCAN_ROOT}",
        f"grep -rn build_subliminal_context {SCAN_ROOT}",
        f"grep -rn facts_idx {SCAN_ROOT}",
    ]
    sigs = _sigs(cmds)
    # Precondition: the calls DO share the scan root as identifiers — if the
    # tokenizer stopped seeing them this test would pass for the wrong reason.
    assert {"_pipeline", "trajectories"} <= (sigs[0].idents & sigs[2].idents)
    assert all(s.path_idents for s in sigs), "root must be classified as a path"
    assert guards.repetition_verdict(sigs) is None
    print("test_repetition_ignores_three_greps_sharing_only_a_scan_root: OK")


def test_repetition_never_names_a_path_operand_as_the_target():
    """Clause 2: a firing on grep/rg/find names the hunt, never the location.

    Five search shapes over the same root, one symbol really being chased. Each
    usable prefix is screened, not just the last: `threshold` is 2, so the first
    two calls can only ever have one prior between them and `sigs[:2]` is out of
    reach — `sigs[:3]`, `sigs[:4]` and `sigs[:5]` can all fire, and each one is
    asserted to, so a guard that goes quiet at the third shape but not the fifth
    fails here. No verdict at any prefix may name the root those calls share or a
    filter's operand; the one term all five aim at is the symbol.
    """
    cmds = [
        f"grep -rn iv_inject_queue {SCAN_ROOT} --exclude-dir=node_modules",
        f"grep -rn iv_inject_queue {SCAN_ROOT} -l | head -5",
        f"find {SCAN_ROOT} -name '*.py' | xargs grep -l iv_inject_queue",
        f"rg iv_inject_queue {SCAN_ROOT} --glob '!node_modules'",
        f"grep -rn iv_inject_queue {SCAN_ROOT} --include=*.py | head -40",
    ]
    sigs = _sigs(cmds)
    fired = checked = 0
    for i in range(3, len(sigs) + 1):
        verdict = guards.repetition_verdict(sigs[:i])
        checked += 1
        if verdict is None:
            continue
        fired += 1
        for location in ("_pipeline", "trajectories", "node_modules"):
            assert location not in verdict.shared_terms, (i, verdict.shared_terms)
    assert checked == 3, checked
    assert fired == checked, (
        f"a real hunt stopped firing at some prefix ({fired} of {checked}): {cmds}"
    )
    final = guards.repetition_verdict(sigs)
    assert final is not None and final.shared_terms == ("iv_inject_queue",), final
    print("test_repetition_never_names_a_path_operand_as_the_target: OK")


def test_repetition_ignores_three_greps_sharing_only_a_filename():
    """Clause 1's other measured shape: the shared operand is ONE file.

    #523's Claim names two token classes, not one. The second is the file a
    search is pointed at: three greps of
    `tests/integration/test_trajectory_extraction.py` for three different test
    names measured `('test_trajectory_extraction',)` as the shared terms both at
    triage (HEAD `a0a127a`) and at the start of this round (HEAD `f31e87f`) — the
    file being scanned, reported to the primary as the target it keeps chasing. A
    search's file operand is scope, so it is marked in full, stem included, and
    carries nothing. The control half pins the narrowing to searches: three READS
    of that same file (`head`/`sed`/`wc`) are one target revisited three ways and
    must still fire on its stem — that is the shape calibration message 76 has.
    """
    target = "tests/integration/test_trajectory_extraction.py"
    greps = _sigs([
        f"grep -rn test_extract_from_session {target}",
        f"grep -rn test_write_text_blob {target}",
        f"grep -rn test_scan_root_tokens {target}",
    ])
    # Precondition: the filename IS shared as an identifier, so the silence
    # below is the carrier rule and not the tokenizer failing to see it.
    assert "test_trajectory_extraction" in (greps[0].idents & greps[2].idents)
    assert all("test_trajectory_extraction" in s.path_idents for s in greps)
    assert guards.repetition_verdict(greps) is None
    assert guards.repetition_verdict(greps, ambient=frozenset()) is None

    reads = _sigs([
        f"head -30 {target}",
        f"sed -n '40,90p' {target}",
        f"wc -l {target}",
    ])
    verdict = guards.repetition_verdict(reads, ambient=frozenset())
    assert verdict is not None, "three reads of one file are one target"
    assert "test_trajectory_extraction" in verdict.shared_terms, verdict.shared_terms
    print("test_repetition_ignores_three_greps_sharing_only_a_filename: OK")


# ---------------------------------------------------------------------------
# #1026 — an opaque id cannot carry a near match on its own
# ---------------------------------------------------------------------------
#
# #523 closed the directory/filename channel; this is the other token class
# that used to carry a match alone. `_is_distinctive` granted the solo-carry on
# shape (`term.count("_") >= 2 or len(term) >= 16`), and two non-symbol shapes
# satisfied it: a 16-char hex id clears the length test exactly, and the
# identifier regex cannot start inside digits, so a session key splices into
# the underscore-rich fragment after it
# (`…/20260912_140009_autocode_7915.md` → `_140009_autocode_7915`).
#
# Measured live in `usage.db` (`inner_voice_observations`, action='inject',
# reason LIKE 'deterministic:%near-identical%'): 14 injects naming an opaque id,
# all of them dated 2026-09-12, the last reading
# `3 near-identical Bash calls for _043416_iv9aa2, a9a5bdae37eff3d4`. Both named
# terms were id artifacts; the primary was inspecting ONE commit from three
# ordinary angles.
#
# SHA is a git short-SHA-shaped token, not a commit in this repo — the token
# CLASS is the defect, which is also why the sequences below are shaped as
# inspection rather than as a hunt.

SHA = "a9a5bdae37eff3d4"
SESSION_DIGEST = (
    "/home/alansrobotlab/lloyd/_pipeline/vault-derived/sessions/"
    "2026-09-12/20260912_140009_autocode_7915.md"
)


def test_repetition_ignores_one_hex_commit_id_alone():
    """Clause 1a: three `git` calls naming one 16-hex id are not a loop.

    `-- lloyd/MEMORY.md` / `--stat` / `--format=%B` is how a commit gets read,
    and the only shared identifier is the id itself. The assertions before the
    verdict pin WHY it is silent: the id is shared as an identifier and is not a
    path operand, so neither the tokenizer nor #523's carrier rule is what
    stopped the fire — only the predicate.
    """
    sigs = _sigs([
        f"git show {SHA} -- lloyd/MEMORY.md",
        f"git show {SHA} --stat",
        f"git log -1 --format=%B {SHA}",
    ])
    assert all(s.idents == frozenset({SHA}) for s in sigs), [s.idents for s in sigs]
    assert not any(s.path_idents for s in sigs), "the id must not be read as a location"
    assert not guards._is_distinctive(SHA), "a hex id must not carry a match alone"
    # `ambient=frozenset()` so the silence cannot be attributed to the ambient
    # pass: with nothing stripped, the predicate is the only thing left.
    assert guards.repetition_verdict(sigs, ambient=frozenset()) is None
    print("test_repetition_ignores_one_hex_commit_id_alone: OK")


def test_repetition_ignores_a_digit_spliced_session_id_fragment_alone():
    """Clause 1b: three non-search reads of one session digest stay silent.

    The shared token here is `_140009_autocode_7915` — three underscores, so it
    cleared the old segment test — and it is a filename STEM, which #523 round 2
    (`47077d8`) already demoted from carrier to location for SEARCHES. These
    three calls are `wc -l` / `grep -c` / `tail -20`, the read shape that #523
    deliberately KEEPS firing (see
    `test_repetition_ignores_three_greps_sharing_only_a_filename`'s control
    half), so the read shape cannot be what silences them. It is the token class:
    the stem is an opaque session key, not a symbol. `test_trajectory_extraction`
    — the pinned control stem — has no digit run and still fires, which is the
    difference between these two tests.
    """
    sigs = _sigs([
        f"wc -l {SESSION_DIGEST}",
        f"grep -c Stop {SESSION_DIGEST}",
        f"tail -20 {SESSION_DIGEST}",
    ])
    stem = "_140009_autocode_7915"
    assert all(stem in s.idents for s in sigs), [s.idents for s in sigs]
    assert stem not in sigs[0].path_idents, (
        "precondition: a read's file is the target, so this is the shape that "
        "must keep firing for a real symbol — silence here is the token class"
    )
    assert not guards._is_distinctive(stem)
    assert guards.repetition_verdict(sigs, ambient=frozenset()) is None
    print("test_repetition_ignores_a_digit_spliced_session_id_fragment_alone: OK")


def test_repetition_still_fires_when_an_opaque_id_shares_with_a_real_symbol():
    """Clause 3: the weakening reaches the solo-carry branch and nothing else.

    Same three `git` shapes, same hex id — plus one hunted symbol in every
    command. Two shared non-path terms clear `min_overlap` without any reference
    to distinctiveness, so the verdict must come back and name BOTH: the id is
    no longer a solo carrier, it has not become invisible. A change that
    suppressed the id outright (a tokenizer deny-list, or read-shape
    suppression) fails here.
    """
    sigs = _sigs([
        f"git log --grep=iv_inject_queue {SHA}",
        f"git log --grep=iv_inject_queue --oneline {SHA}",
        f"git log --grep=iv_inject_queue -3 {SHA}",
    ])
    assert all(s.idents == frozenset({"iv_inject_queue", SHA}) for s in sigs)
    verdict = guards.repetition_verdict(sigs, ambient=frozenset())
    assert verdict is not None, "two shared terms still clear min_overlap"
    assert set(verdict.shared_terms) == {"iv_inject_queue", SHA}, verdict.shared_terms
    assert guards._is_distinctive("iv_inject_queue"), (
        "and the symbol keeps its solo-carry: the id is what was demoted"
    )
    print("test_repetition_still_fires_when_an_opaque_id_shares_with_a_real_symbol: OK")


def test_opaque_id_inspection_through_the_pretool_hook_produces_no_nudge():
    """The seam this change has to hold: hook → guard → what the primary reads.

    A predicate flipped in isolation proves nothing about the message the
    primary receives, and the guard is reached through `fire_pre_tool_use`,
    which keeps the ring, applies the ambient pass over more history than the
    comparison window holds, and renders the nudge. So drive the two opaque-id
    sequences through the real hook with no LLM available, then drive one real
    reformulation through the same hook in a fresh observer to prove the channel
    still speaks. Both halves are needed: the first alone would also pass if the
    hook stopped wiring the repetition guard at all.
    """
    from unittest.mock import patch
    from app.harness.hooks import HookRegistry
    from app.inner_voice.observer import install_observer

    cfg = obs_mod._observer_cfg()
    cfg.update({"pretool_llm_enabled": False, "fast_path_enabled": True})

    async def run(session_id: str, commands: list[str]) -> list:
        def no_llm(**kwargs):
            raise AssertionError("the repetition guard must not need an LLM call")

        chat_messages: list = []
        hooks = HookRegistry()
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg), \
             patch.object(obs_mod, "extract_goal_card", new=_no_goal_card):
            state = install_observer(
                hooks=hooks, session_id=session_id, turn_id=f"{session_id}_turn",
                user_request="inspect one commit, then a session digest",
                chat_messages_handle=chat_messages,
                cancel_event=asyncio.Event(), primary_model="primary",
            )
        with patch.object(obs_mod, "_post_chat_completion_with_tools", new=no_llm):
            for command in commands:
                await hooks.fire_pre_tool_use(
                    session_id=session_id, tool_name="Bash",
                    tool_input={"command": command},
                )
        obs_mod.close_observer(state)
        return chat_messages

    # The three SHA calls alone would stay silent even unfixed, and for a
    # reason worth stating: the observer computes `ubiquitous_identifiers` over
    # the whole ring it keeps, and in a ring where every call names the same id
    # that id reads as ambient and is dropped. The live fires were never that
    # shape — the primary inspected one commit INSIDE an ordinary working turn,
    # so the id appeared in three calls of eleven and was not ubiquitous. That
    # is the sequence replayed here, healthy calls first.
    inspection = [
        "git status --porcelain | head -20",
        "pytest tests/integration/test_iv_loop_guards.py -k repetition -q",
        "sed -n '1,40p' app/inner_voice/guards.py",
        "grep -rn resolve_model_alias app/config.py",
        "supervisorctl status | head -40",
        "git log --format='%h %s' -3",
        f"git show {SHA} -- lloyd/MEMORY.md",
        f"git show {SHA} --stat",
        f"git log -1 --format=%B {SHA}",
        f"wc -l {SESSION_DIGEST}",
    ]
    injected = asyncio.new_event_loop().run_until_complete(
        run("opaque_id_inspection", inspection)
    )
    assert injected == [], f"opaque ids carried a nudge to the primary: {injected}"

    hunt = [
        "grep -rn iv_inject_queue app/inner_voice",
        "grep -rn iv_inject_queue app/routers --include=*.py | head -40",
        'grep -rn "iv_inject_queue" workers/; echo EXIT=$?',
    ]
    spoke = asyncio.new_event_loop().run_until_complete(run("opaque_id_control", hunt))
    assert len(spoke) == 1, f"the same hook stopped catching a real hunt: {spoke}"
    assert "iv_inject_queue" in str(spoke[0].get("content")), spoke[0]
    print("test_opaque_id_inspection_through_the_pretool_hook_produces_no_nudge: OK")


def test_repetition_requires_a_carrier_that_is_not_a_path():
    """Clause 3: with `ambient=frozenset()` the carrier rule is load-bearing.

    Passing the empty ambient set means the ambient pass cannot be what makes
    the first three calls silent — `ubiquitous_identifiers` would have called
    the shared root turn-ambient on its own, and this pins that it does not
    have to. The last three commands are the same calls with one real symbol
    added to the hunt, and they must fire.
    """
    path_only = [
        f"grep -rn alpha_marker {SCAN_ROOT}",
        f"grep -rn beta_marker {SCAN_ROOT} --include=*.log",
        "grep -rn gamma_marker /home/alansrobotlab/lloyd/_pipeline/vault-derived/trajectories | head -5",
    ]
    assert all(s.path_idents for s in _sigs(path_only))
    assert guards.repetition_verdict(_sigs(path_only), ambient=frozenset()) is None

    with_symbol = [
        f"grep -rn iv_inject_queue {SCAN_ROOT}",
        f'grep -rn "iv_inject_queue" {SCAN_ROOT} --include=*.py',
        f"grep -rn iv_inject_queue {SCAN_ROOT} | head -40",
    ]
    verdict = guards.repetition_verdict(_sigs(with_symbol), ambient=frozenset())
    assert verdict is not None, "a real hunt sharing a root must still fire"
    assert "iv_inject_queue" in verdict.shared_terms
    assert not {"_pipeline", "trajectories"} & set(verdict.shared_terms)
    print("test_repetition_requires_a_carrier_that_is_not_a_path: OK")


def test_repetition_ignores_filter_operands():
    """Clause 4: `--exclude-dir=` and `grep -v` operands are boilerplate.

    Three greps differing only in the pattern and the length of the exclusion
    list share `__pycache__` and `node_modules` and nothing else. Both render
    forms are driven — `--exclude-dir=X` and a piped `grep -v X` — because the
    histogram showed the token arriving both ways.
    """
    dash = [
        "grep -rn alpha_marker app/ --exclude-dir=__pycache__ --exclude-dir=node_modules",
        "grep -rn beta_marker app/ --exclude-dir=__pycache__ --exclude-dir=node_modules --exclude-dir=.git",
        "grep -rn gamma_marker app/ --exclude-dir=__pycache__ --exclude-dir=node_modules",
    ]
    pipe = [
        "grep -rn alpha_marker app/ | grep -v node_modules | grep -v __pycache__",
        "grep -rn beta_marker app/ | grep -v node_modules | grep -v __pycache__ | head",
        "grep -rn gamma_marker app/ | grep -v node_modules | grep -v __pycache__ | wc -l",
    ]
    for cmds in (dash, pipe):
        sigs = _sigs(cmds)
        shared = set(sigs[0].idents) & set(sigs[2].idents)
        assert {"__pycache__", "node_modules"} <= shared, shared
        assert guards.repetition_verdict(sigs) is None, cmds
    print("test_repetition_ignores_filter_operands: OK")


def test_repetition_still_fires_on_one_symbol_over_different_files():
    """Clause 5 — the purpose clause, kept alive by the same edit as 1-4.

    One symbol reformulated across DIFFERENT targets (the calibration turn
    20260905_011748_iv84e4 shape) still fires and still names the symbol; and
    a byte-identical Bash repeat still fires as `exact`, which is the path a
    path-blind carrier rule could otherwise have broken.
    """
    reformulations = [
        "grep -rn iv_inject_queue app/inner_voice",
        "grep -rn iv_inject_queue app/routers --include=*.py | head -40",
        'grep -rn "iv_inject_queue" workers/; echo EXIT=$?',
    ]
    verdict = guards.repetition_verdict(_sigs(reformulations))
    assert verdict is not None, "the guard stopped catching a reformulated hunt"
    assert "iv_inject_queue" in verdict.shared_terms, verdict.shared_terms

    verbatim = _sigs([f"grep -rn iv_inject_queue {SCAN_ROOT}"] * 3)
    exact = guards.repetition_verdict(verbatim)
    assert exact is not None and exact.exact, "verbatim Bash repeat not caught"
    assert "the same call" in guards.repetition_inject_content(exact)
    print("test_repetition_still_fires_on_one_symbol_over_different_files: OK")


def test_repetition_catches_verbatim_repeats_of_any_tool():
    """Exact re-runs need no similarity heuristic — three identical Reads."""
    sigs = [
        guards.tool_call_signature("Read", {"file_path": "/a/b.py"})
        for _ in range(3)
    ]
    verdict = guards.repetition_verdict(sigs)
    assert verdict is not None and verdict.exact
    assert "the same call" in guards.repetition_inject_content(verdict)
    print("test_repetition_catches_verbatim_repeats_of_any_tool: OK")


def test_operand_classification_survives_two_shell_shapes():
    """The path rule reads operands by position, so it is only as good as its
    shell split — and each of these two shapes silently destroyed a real fire
    while every other test in this file stayed green.

    (a) A separator glued to the path with no space (`…py; echo`), which is how
        generated shell usually reads. Taken as one token, the trailing `;`
        defeats the file-suffix test, the whole path is marked as a location,
        and the file's own stem goes with it.
    (b) A flag whose value is inline (`--include='*'`) ahead of the pattern. If
        the split breaks that flag in two, the operand counter is off by one and
        the hunted symbol is read as the scan target instead.
    """
    glued = _sigs([
        "head -20 app/routers/_messages_subliminal.py; echo ===",
        "sed -n '20,120p' app/routers/_messages_subliminal.py; echo ===",
        "grep -n state app/routers/_messages_subliminal.py | head",
    ])
    v = guards.repetition_verdict(glued)
    assert v is not None, "three probes of one FILE must stay a repeat"
    assert "_messages_subliminal" in v.shared_terms, v.shared_terms

    inline_flags = _sigs([
        "grep -rn --include='*' \"iv_inject_queue\" app/inner_voice",
        "grep -rn --exclude='*.pyc' \"iv_inject_queue\" app/routers",
        "grep -rn --include='*.py' \"iv_inject_queue\" workers/ | head",
    ])
    v2 = guards.repetition_verdict(inline_flags)
    assert v2 is not None, "--include='*' must not shift the pattern into path position"
    assert "iv_inject_queue" in v2.shared_terms, v2.shared_terms
    print("test_operand_classification_survives_two_shell_shapes: OK")


def test_verbatim_bash_repeat_over_a_scan_root_still_fires_exact():
    """The path rule narrows the NEAR signal; re-running one command word for
    word is the other signal and stays untouched. Same command three times,
    aimed at the scan root #523 otherwise refuses to name."""
    cmd = f"grep -rn iv_inject_queue {SCAN_ROOT} --include=*.py"
    verdict = guards.repetition_verdict(_sigs([cmd] * 3))
    assert verdict is not None and verdict.exact
    assert "_pipeline" not in verdict.shared_terms, verdict.shared_terms
    assert "iv_inject_queue" in verdict.shared_terms, verdict.shared_terms
    assert "the same call" in guards.repetition_inject_content(verdict)
    print("test_verbatim_bash_repeat_over_a_scan_root_still_fires_exact: OK")


def test_find_name_glob_is_the_hunt_not_the_tree():
    """`find` marks every bare operand as a path, so its pattern has to be
    recovered from behind the flag: three of these are one pattern chased
    three times, and the directory they share must not be what the guard
    reports. `grep -e` takes the same shape for the same reason."""
    sigs = _sigs([
        "find /home/alansrobotlab/lloyd/_pipeline -name '*iv_inject_queue*' | head",
        "find /home/alansrobotlab/lloyd/_pipeline -name '*iv_inject_queue*' -type f",
        "find /home/alansrobotlab/lloyd/_pipeline -iname '*iv_inject_queue*' | wc -l",
    ])
    verdict = guards.repetition_verdict(sigs)
    assert verdict is not None, "one -name glob repeated three times is a loop"
    assert "iv_inject_queue" in verdict.shared_terms, verdict.shared_terms
    # Intersected with what the classifier itself marked, so this can fail: a
    # hand-picked name list here would also have to name the tokens the
    # identifier rules never produce (`vault`, `derived` are dropped — no
    # underscore, under 12 chars), and an assertion over those passes no matter
    # what the guard does.
    locations = sigs[0].path_idents | sigs[1].path_idents | sigs[2].path_idents
    assert "_pipeline" in locations, sorted(locations)
    assert not (set(verdict.shared_terms) & locations), (
        f"-name globs are the hunt, so no scan-root token may be named: "
        f"{verdict.shared_terms} vs {sorted(locations)}"
    )
    print("test_find_name_glob_is_the_hunt_not_the_tree: OK")


def test_repetition_ignores_different_tools():
    sigs = [
        guards.tool_call_signature("Read", {"file_path": "/a/b.py"}),
        guards.tool_call_signature("Grep", {"pattern": "/a/b.py"}),
        guards.tool_call_signature("Bash", {"command": "cat /a/b.py"}),
    ]
    assert guards.repetition_verdict(sigs) is None
    print("test_repetition_ignores_different_tools: OK")


# ---------------------------------------------------------------------------
# Fast-path rows must not count as judgment
# ---------------------------------------------------------------------------


def test_suppressor_sees_past_fast_path_rows():
    """The exact row sequence that defeated the suppressor on 2026-09-04.

    A tool_result inject, then the fast-path noops every iteration emits —
    one `assistant_message`, one `pretool` — then a second inject. Before
    this fix the walk-back stopped on the pretool bookkeeping row and
    reported "no prior inject".
    """
    prior = [
        {"trigger": "tool_result", "action": "inject", "reason": "drifting"},
        {"trigger": "assistant_message", "action": "noop",
         "reason": "fast-path: tool-dispatch-only iteration", "fast_path": True},
        {"trigger": "pretool", "action": "noop",
         "reason": "observation-only: pretool LLM disabled", "fast_path": True},
    ]
    assert guards.suppress_consecutive_inject(
        action="inject", prior_decisions=prior, is_terminal=False,
    )
    print("test_suppressor_sees_past_fast_path_rows: OK")


def test_suppressor_still_clears_on_a_judged_noop():
    """An LLM noop is a real look. It must still clear the suppressor."""
    prior = [
        {"trigger": "tool_result", "action": "inject", "reason": "drifting"},
        {"trigger": "tool_result", "action": "noop",
         "reason": "primary recovered", "fast_path": False},
    ]
    assert not guards.suppress_consecutive_inject(
        action="inject", prior_decisions=prior, is_terminal=False,
    )
    print("test_suppressor_still_clears_on_a_judged_noop: OK")


def test_every_deterministic_noop_is_tagged_fast_path():
    """A missed tag silently re-breaks the suppressor and the prompt window."""
    assert _fast_path_tool_result("Read", "hi", False, benign_seen=3, sample_every=5).fast_path
    assert obs_mod._fast_path_pretool("Read", {"file_path": "/x"}).fast_path
    assert obs_mod._fast_path_pretool("Bash", {"command": "ls -la"}).fast_path
    assert _fast_path_assistant_message("", [{"name": "Bash"}]).fast_path
    # The stall is a turn guard's now (`app/harness/turn_guards.py`); the
    # observer's fast path leaves it to the terminal branch, which skips an
    # iteration a guard already answered and judges it when guards are off.
    assert _fast_path_assistant_message("Now let me check the logs:", []) is None
    print("test_every_deterministic_noop_is_tagged_fast_path: OK")


def test_prior_decisions_block_drops_fast_path_noise():
    """At ~8 bookkeeping rows per iteration, the 8-slot window showed nothing.

    Reproduces the window as it stood at the third inject of the real turn:
    every slot a fast-path noop, and one real inject pushed out.
    """
    decisions = [{"trigger": "tool_result", "action": "inject",
                  "reason": "stuck in a loop", "fast_path": False}]
    for _ in range(4):
        decisions += [
            {"trigger": "assistant_message", "action": "noop",
             "reason": "fast-path: tool-dispatch-only iteration", "fast_path": True},
            {"trigger": "pretool", "action": "noop",
             "reason": "observation-only: pretool LLM disabled", "fast_path": True},
            {"trigger": "tool_result", "action": "noop",
             "reason": "fast-path: benign result (unsampled)", "fast_path": True},
        ]
    block = prompt_mod._format_prior_decisions(decisions)
    assert "stuck in a loop" in block, "the only real decision was crowded out"
    assert "fast-path" not in block
    assert "observation-only" not in block
    print("test_prior_decisions_block_drops_fast_path_noise: OK")


# ---------------------------------------------------------------------------
# Inject pacing
# ---------------------------------------------------------------------------


def test_cooldown_blocks_the_88_second_budget_burn():
    """Inject, three iterations, inject again — the real seq 66 -> 75 gap.

    The suppressor allows this (a judged noop sits between them). At a
    cooldown of 4 it is held, so the budget survives to the point where the
    drift is actually established.
    """
    prior = [
        {"trigger": "tool_result", "action": "inject", "reason": "loop", "fast_path": False},
        {"trigger": "assistant_message", "action": "noop", "fast_path": True},
        {"trigger": "tool_result", "action": "noop", "reason": "recovered", "fast_path": False},
        {"trigger": "assistant_message", "action": "noop", "fast_path": True},
        {"trigger": "assistant_message", "action": "noop", "fast_path": True},
    ]
    assert guards.iterations_since_last_inject(prior) == 3
    assert guards.inject_on_cooldown(prior, cooldown_iterations=4)
    assert not guards.inject_on_cooldown(prior, cooldown_iterations=3)
    print("test_cooldown_blocks_the_88_second_budget_burn: OK")


def test_cooldown_silent_before_any_inject():
    prior = [{"trigger": "assistant_message", "action": "noop", "fast_path": True}]
    assert guards.iterations_since_last_inject(prior) is None
    assert not guards.inject_on_cooldown(prior, cooldown_iterations=4)
    print("test_cooldown_silent_before_any_inject: OK")


def _state(**kw):
    from app.inner_voice.observer import ObserverState
    base = dict(
        session_id="loop_sess",
        turn_id="loop_turn",
        user_request="review the inner voice implementation",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        primary_model="primary",
        intervention_budget=3,
        cfg={"inject_cooldown_iterations": 4},
    )
    base.update(kw)
    return ObserverState(**base)


def _one_inject_ago(iterations: int) -> list[dict]:
    """Decision rows for "one inject, then `iterations` primary iterations".

    Shaped like the real turn: a judged noop lands shortly after the inject
    (seq 69 there — "Primary has received the config data it needed"), which
    is what clears `suppress_consecutive_inject` and lets the question reach
    the cooldown. Without that judged row the suppressor answers first and
    the cooldown is never consulted — which is correct, and is why this
    helper models the harder case.
    """
    rows: list[dict] = [
        {"trigger": "tool_result", "action": "inject", "reason": "loop",
         "fast_path": False},
    ]
    for i in range(iterations):
        rows.append(
            {"trigger": "assistant_message", "action": "noop", "fast_path": True}
        )
        if i == 0:
            rows.append(
                {"trigger": "tool_result", "action": "noop",
                 "reason": "primary acted on the nudge", "fast_path": False}
            )
    return rows


def test_cooldown_downgrades_a_discretionary_inject_end_to_end():
    state = _state()
    state.decisions_this_turn = _one_inject_ago(2)
    d = ObserverDecision(action="inject", reason="still drifting", content="focus")
    _apply_decision_guards(state, d, trigger="tool_result", tool_calls=[])
    assert d.action == "noop_inject_on_cooldown", d.action
    assert "2 primary iterations" in d.reason
    print("test_cooldown_downgrades_a_discretionary_inject_end_to_end: OK")


def test_cooldown_exempts_bypass_budget_and_terminal_injects():
    """Stall rescue and the repetition guard must never be rate-limited:
    they are what keeps a turn from ending or looping, and both are specific
    enough that firing twice is a real signal rather than a nag."""
    state = _state()
    state.decisions_this_turn = _one_inject_ago(1)
    rescue = ObserverDecision(
        action="inject", reason="stall rescue", content="continue", bypass_budget=True,
    )
    _apply_decision_guards(state, rescue, trigger="assistant_message", tool_calls=[])
    assert rescue.action == "inject", rescue.reason

    # A terminal iteration is the last chance to act; never hold it back.
    state2 = _state()
    state2.decisions_this_turn = _one_inject_ago(1)
    terminal = ObserverDecision(action="inject", reason="unmet criteria", content="finish")
    _apply_decision_guards(
        state2, terminal, trigger="assistant_message", tool_calls=[], is_terminal=True,
    )
    assert terminal.action == "inject", terminal.reason
    print("test_cooldown_exempts_bypass_budget_and_terminal_injects: OK")


def test_cooldown_releases_once_the_gap_is_real():
    state = _state()
    state.decisions_this_turn = _one_inject_ago(4)
    d = ObserverDecision(action="inject", reason="drift confirmed", content="refocus")
    _apply_decision_guards(state, d, trigger="tool_result", tool_calls=[])
    assert d.action == "inject", d.reason
    print("test_cooldown_releases_once_the_gap_is_real: OK")


# ---------------------------------------------------------------------------
# Failure payloads inside successful results
# ---------------------------------------------------------------------------


def test_subagent_max_turns_is_never_sampled_away():
    """The real payload: 4 minutes, 28 tool calls, nothing returned.

    300 bytes and is_error False, so both the size rule and the 1-in-5
    sampler skipped it and the observer never learned the Task failed.
    """
    payload = (
        '{"response": "\\n[stopped: max_turns]", "tools_used": ["Read", "Bash", '
        '"Bash", "Grep"], "description": "Review IV observer prompt"}'
    )
    assert guards.looks_like_failure_payload(payload)
    # benign_seen deliberately off the sampling boundary. The turn guard
    # answers it on every turn now (tests/test_turn_guards.py), so the
    # observer records it as answered rather than paying a call on it.
    fp = _fast_path_tool_result("Task", payload, False, benign_seen=3, sample_every=5)
    assert fp is not None and "turn guard" in fp.reason, fp
    print("test_subagent_max_turns_is_never_sampled_away: OK")


def test_bash_timeout_payload_escalates():
    payload = '{"error": "command timed out after 120000ms", "command": "grep -rn ..."}'
    assert guards.looks_like_failure_payload(payload)
    fp = _fast_path_tool_result("Bash", payload, False, benign_seen=3, sample_every=5)
    assert fp is not None and "turn guard" in fp.reason, fp
    print("test_bash_timeout_payload_escalates: OK")


def test_ordinary_results_still_fast_path():
    """The escalation must stay rare — it is unsampled and unconditional."""
    for content in [
        "total 48\ndrwxr-xr-x 1 alan alan 972 Sep 4 18:18 .",
        '{"response": "Here is the review you asked for.", "tools_used": ["Read"]}',
        "def build_tool_result_summary(tool_name, result_preview):",
    ]:
        assert not guards.looks_like_failure_payload(content), content
        d = _fast_path_tool_result("Bash", content, False, benign_seen=3, sample_every=5)
        assert d is not None and d.action == "noop"
    print("test_ordinary_results_still_fast_path: OK")


# ---------------------------------------------------------------------------
# The silent primary
# ---------------------------------------------------------------------------


def test_silent_streak_escalates_past_the_fast_path():
    """33 text-free iterations produced 33 zero-cost noops and no judgment."""
    calls = [{"name": "Bash"}]
    # Below the limit: still fast-pathed, still free.
    d = _fast_path_assistant_message("", calls, silent_streak=9, silent_streak_limit=10)
    assert d is not None and d.action == "noop" and d.fast_path
    # At the limit: escalate to LLM judgment.
    assert _fast_path_assistant_message("", calls, silent_streak=10, silent_streak_limit=10) is None
    # Disabled by config.
    d = _fast_path_assistant_message("", calls, silent_streak=99, silent_streak_limit=0)
    assert d is not None and d.action == "noop"
    print("test_silent_streak_escalates_past_the_fast_path: OK")


def test_silent_streak_block_does_not_presume_guilt():
    """A long quiet run is usually productive work. The prompt must say so,
    or a jumpy observer turns this into a nag every 10 iterations."""
    summary = prompt_mod.build_assistant_message_summary(
        22, "", [{"function": {"name": "Bash"}}], "tool_calls", silent_streak=12,
    )
    assert "SILENT STREAK" in summary and "iteration 12" in summary
    assert "that is common" in summary and "not by itself a problem" in summary
    # Absent unless escalated.
    plain = prompt_mod.build_assistant_message_summary(
        3, "", [{"function": {"name": "Bash"}}], "tool_calls",
    )
    assert "SILENT STREAK" not in plain
    print("test_silent_streak_block_does_not_presume_guilt: OK")


# ---------------------------------------------------------------------------
# The observer must be able to see what the primary ran
# ---------------------------------------------------------------------------


def test_tool_result_summary_carries_the_command():
    """Without this the observer sees a docstring with no context — which is
    how it concluded the primary had "discovered a subliminal injection bug"
    it had never mentioned."""
    docstring = '"""Subliminal-injection capture (#306).\n\nThree ephemeral injection sites...'
    summary = prompt_mod.build_tool_result_summary(
        "Bash", docstring, False,
        call_preview="cd ~/lloyd && head -20 app/routers/_messages_subliminal.py",
    )
    assert "head -20 app/routers/_messages_subliminal.py" in summary
    assert "Primary ran:" in summary
    print("test_tool_result_summary_carries_the_command: OK")


def test_tool_result_summary_unchanged_without_a_command():
    """Callers that cannot recover the args must still get a valid summary."""
    summary = prompt_mod.build_tool_result_summary("Read", "file body", False)
    assert "Primary ran:" not in summary
    assert "Tool Read returned result" in summary
    print("test_tool_result_summary_unchanged_without_a_command: OK")


def test_signature_preview_is_bounded():
    """Commands go into the observer's prompt; an unbounded one is a cost bug."""
    sig = guards.tool_call_signature("Bash", {"command": "echo " + "x" * 5000})
    assert len(sig.preview) <= 160
    print("test_signature_preview_is_bounded: OK")


def test_signature_ignores_key_order_and_description():
    """`description` is model narration, not intent — two calls that differ
    only there are the same call."""
    a = guards.tool_call_signature("Grep", {"pattern": "foo", "path": "app/", "description": "first try"})
    b = guards.tool_call_signature("Grep", {"path": "app/", "pattern": "foo", "description": "second try"})
    assert a.exact == b.exact
    print("test_signature_ignores_key_order_and_description: OK")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()


# ---------------------------------------------------------------------------
# The silent model swap
# ---------------------------------------------------------------------------


def test_observer_resolves_to_the_primary_endpoint():
    """Inner Voice must run on the primary model, and say so out loud.

    `inner_voice.model` read `secondary` from 2026-05-07 onward and resolved
    to primary the whole time, because `resolve_model_alias` rewrites
    secondary -> primary while `secondary_enabled` is false. When de893d7
    turned that flag on for the autonomy scheduler (2026-09-03), the observer
    moved to Qwen3.5-4B with no config change and no log line. On its first
    day there it intervened on 40% of LLM-judged events against primary's
    1.7%, fabricated a finding, and cancelled a turn.

    This asserts the resolved endpoint, not the config string, so the alias
    indirection cannot hide a swap again.
    """
    from app.config import CONFIG
    base_url, model_name = obs_mod._resolve_endpoint()
    primary_url = (CONFIG["models"]["primary"].get("base_url") or "").rstrip("/")
    assert model_name == "primary", (
        f"observer resolved to {model_name!r}; pin inner_voice.model to primary "
        "or record the iv_grade evidence for moving it"
    )
    assert base_url == primary_url, (base_url, primary_url)
    print("test_observer_resolves_to_the_primary_endpoint: OK")


def test_alias_rewrite_is_logged_once(caplog=None):
    """The rewrite that hid the swap must leave a trace in the log."""
    import logging
    from app import config as config_mod

    config_mod._ALIAS_REWRITES_LOGGED.clear()
    original = config_mod.CONFIG.get("secondary_enabled")
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    config_mod.logger.addHandler(handler)
    prior_level = config_mod.logger.level
    config_mod.logger.setLevel(logging.INFO)
    try:
        config_mod.CONFIG["secondary_enabled"] = False
        assert config_mod.resolve_model_alias("secondary") == "primary"
        assert config_mod.resolve_model_alias("secondary") == "primary"
        assert len(records) == 1, f"expected one log line, got {records}"
        assert "secondary_enabled is false" in records[0]
    finally:
        config_mod.logger.removeHandler(handler)
        config_mod.logger.setLevel(prior_level)
        config_mod.CONFIG["secondary_enabled"] = original
        config_mod._ALIAS_REWRITES_LOGGED.clear()
    print("test_alias_rewrite_is_logged_once: OK")


def test_new_guards_are_on_by_default():
    """A guard that ships disabled is a guard that does not exist. The
    stalled-progress gate sat behind `stalled_progress: false` through the
    entire incident."""
    cfg = obs_mod._observer_cfg()
    assert cfg["repetition_guard_enabled"] is True
    assert cfg["silent_iterations_before_review"] > 0
    assert cfg["inject_cooldown_iterations"] > 0
    assert obs_mod._todo_stewardship_cfg()["stalled_progress"] is True
    print("test_new_guards_are_on_by_default: OK")


# ---------------------------------------------------------------------------
# End to end: the real turn, replayed through the live hooks
# ---------------------------------------------------------------------------


async def _no_goal_card(*a, **kw):
    """Goal-card extraction is an LLM call at turn start; not under test."""
    return None


def test_real_turn_replayed_through_the_pretool_hook():
    """Drive the 28 Bash calls through `fire_pre_tool_use` and assert the
    observer speaks — with no LLM available at all.

    The point of doing this deterministically is that on the night in
    question the observer's three interventions all came from an LLM working
    from a 300-char result fragment, and one of them was a fabrication. This
    path cannot fabricate: it fires on the arguments or not at all.
    """
    from unittest.mock import patch
    from app.harness.hooks import HookRegistry
    from app.inner_voice.observer import install_observer

    async def no_llm(**kwargs):
        raise AssertionError("the repetition guard must not need an LLM call")

    cfg = obs_mod._observer_cfg()
    cfg.update({"pretool_llm_enabled": False, "fast_path_enabled": True})

    async def scenario():
        chat_messages: list = []
        hooks = HookRegistry()
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg), \
             patch.object(obs_mod, "extract_goal_card", new=_no_goal_card):
            state = install_observer(
                hooks=hooks, session_id="loop_replay", turn_id="loop_replay_turn",
                user_request="architecture and code review of inner voice",
                chat_messages_handle=chat_messages,
                cancel_event=asyncio.Event(), primary_model="primary",
            )
        with patch.object(obs_mod, "_post_chat_completion_with_tools", new=no_llm):
            for _idx, command in REAL_BASH_CALLS:
                await hooks.fire_pre_tool_use(
                    session_id="loop_replay",
                    tool_name="Bash",
                    tool_input={"command": command},
                )
        obs_mod.close_observer(state)
        return chat_messages

    injected = asyncio.new_event_loop().run_until_complete(scenario())

    # One nudge per pathological cluster — the search-reformulation loop and
    # the hunt for a symbol that does not exist. See
    # test_repetition_fires_on_the_real_loop for why there are two.
    assert len(injected) == 2, f"expected two nudges, got {len(injected)}"
    body = str(injected[0].get("content"))
    assert "iv_cancel_requested" in body or "iv_inject_queue" in body
    assert "ANSWER" in body
    assert "_messages_subliminal" in str(injected[1].get("content"))
    print("test_real_turn_replayed_through_the_pretool_hook: OK")


def test_healthy_turn_replayed_produces_no_nudge():
    """The same path over the exploration phase alone must stay silent."""
    from unittest.mock import patch
    from app.harness.hooks import HookRegistry
    from app.inner_voice.observer import install_observer

    cfg = obs_mod._observer_cfg()
    cfg.update({"pretool_llm_enabled": False, "fast_path_enabled": True})

    async def scenario():
        chat_messages: list = []
        hooks = HookRegistry()
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg), \
             patch.object(obs_mod, "extract_goal_card", new=_no_goal_card):
            state = install_observer(
                hooks=hooks, session_id="healthy_replay", turn_id="healthy_turn",
                user_request="architecture and code review of inner voice",
                chat_messages_handle=chat_messages,
                cancel_event=asyncio.Event(), primary_model="primary",
            )
        for idx, command in REAL_BASH_CALLS:
            if idx > 51:
                break
            await hooks.fire_pre_tool_use(
                session_id="healthy_replay", tool_name="Bash",
                tool_input={"command": command},
            )
        obs_mod.close_observer(state)
        return chat_messages

    injected = asyncio.new_event_loop().run_until_complete(scenario())
    assert injected == [], f"guard fired during healthy exploration: {injected}"
    print("test_healthy_turn_replayed_produces_no_nudge: OK")


def test_shared_terms_lead_with_the_rarest():
    """The message must name what is being chased, not the ambient path.

    Live turn 20260905_020747_ivfe5f named "alansrobotlab" — a component of
    the home directory in every command — ahead of the symbol the primary was
    actually looping on.
    """
    cmds = [
        "grep -rn zzq_phantom_handle_v3 /home/alansrobotlab/lloyd --include=*.py",
        "ls -la /home/alansrobotlab/lloyd; ls -d /home/alansrobotlab/lloyd/app",
        'grep -rn "zzq_phantom_handle_v3" /home/alansrobotlab/lloyd; echo EXIT=$?',
        "grep -rn --include='*' \"zzq_phantom_handle_v3\" /home/alansrobotlab/lloyd/app",
    ]
    verdict = guards.repetition_verdict(_sigs(cmds))
    assert verdict is not None
    assert verdict.shared_terms[0] == "zzq_phantom_handle_v3", verdict.shared_terms
    assert guards.repetition_inject_content(verdict).index("zzq_phantom_handle_v3") < 120
    print("test_shared_terms_lead_with_the_rarest: OK")


# ---------------------------------------------------------------------------
# #770: every guard decision is written with the guard's name
# ---------------------------------------------------------------------------


def _record_here(monkeypatch) -> list:
    """A recorder installed for this test only. The module-level RECORDED patch
    is set once at import, and another file in the same worker can replace the
    writer after that — which read as a guard that stamped nothing."""
    rows: list = []
    rec = lambda **kw: (rows.append(kw), len(rows))[1]  # noqa: E731
    monkeypatch.setattr(obs_mod, "record_inner_voice_observation", rec)
    # The turn guards write through `usage_store` directly.
    import usage_store
    monkeypatch.setattr(usage_store, "record_inner_voice_observation", rec)
    return rows


def test_the_repetition_guard_row_carries_its_safeguard_key(monkeypatch):
    """Driven through the real pretool hook, as the live fires were."""
    from unittest.mock import patch
    from app.harness.hooks import HookRegistry
    install_observer = obs_mod.install_observer
    recorded = _record_here(monkeypatch)

    cfg = obs_mod._observer_cfg()
    cfg.update({"pretool_llm_enabled": False, "fast_path_enabled": True})
    hunt = [
        "grep -rn iv_inject_queue app/inner_voice",
        "grep -rn iv_inject_queue app/routers --include=*.py | head -40",
        'grep -rn "iv_inject_queue" workers/; echo EXIT=$?',
    ]

    async def run():
        hooks = HookRegistry()
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg), \
             patch.object(obs_mod, "extract_goal_card", new=_no_goal_card):
            state = install_observer(
                hooks=hooks, session_id="safeguard_rep", turn_id="safeguard_rep_turn",
                user_request="find the queue", chat_messages_handle=[],
                cancel_event=asyncio.Event(), primary_model="primary",
            )
        for command in hunt:
            await hooks.fire_pre_tool_use(
                session_id="safeguard_rep", tool_name="Bash",
                tool_input={"command": command},
            )
        obs_mod.close_observer(state)

    asyncio.new_event_loop().run_until_complete(run())
    rows = [r for r in recorded if r["session_id"] == "safeguard_rep"]
    injects = [r for r in rows if r["action"] == "inject"]
    assert len(injects) == 1, rows
    assert injects[0]["safeguard"] == "repetition", injects[0]
    # The observation-only rows beside it are named too, not left to read as
    # model verdicts.
    assert {r["safeguard"] for r in rows if r["action"] == "noop"} <= {
        "observation_only", "fast_path"}, rows


def test_stall_rescue_and_the_guard_rewrites_name_themselves(monkeypatch):
    from app.harness.hooks import HookRegistry
    from app.harness.turn_guards import install_turn_guards
    recorded = _record_here(monkeypatch)
    hooks = HookRegistry()
    install_turn_guards(hooks, session_id="stall_key", turn_id="t",
                        chat_messages_handle=[])
    asyncio.new_event_loop().run_until_complete(hooks.fire_on_event({
        "type": "assistant_message", "text": "Now let me check the logs:",
        "tool_calls": [], "iteration": 1,
    }))
    assert [(r["action"], r["safeguard"]) for r in recorded] == [
        ("inject", "stall_rescue")], recorded

    # Terminal ambient upgraded to inject: the model chose ambient, the guard
    # chose inject, so the row is the guard's.
    state = _state()
    amb = ObserverDecision(action="ambient", reason="follow up later", content="")
    _apply_decision_guards(state, amb, trigger="assistant_message", tool_calls=[],
                           is_terminal=True)
    assert amb.action == "inject" and amb.safeguard == "stall_rescue_ambient", amb

    # Unattended terminal inject: Python wrote the words.
    state = _state(unattended=True)
    inj = ObserverDecision(action="inject", reason="deliver the report", content="report now")
    _apply_decision_guards(state, inj, trigger="assistant_message", tool_calls=[],
                           is_terminal=True)
    assert inj.action == "inject" and inj.safeguard == "unattended_content", inj

    # A model inject no guard touched stays unkeyed.
    state = _state()
    plain = ObserverDecision(action="inject", reason="drifting", content="refocus")
    _apply_decision_guards(state, plain, trigger="tool_result", tool_calls=[])
    assert plain.action == "inject" and plain.safeguard is None, plain


def test_persist_writes_the_key_and_marks_fast_path_rows(monkeypatch):
    state = _state()
    recorded = _record_here(monkeypatch)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(obs_mod._persist(
        state, ObserverDecision(action="inject", reason="stall", content="go",
                                bypass_budget=True, safeguard="stall_rescue"),
        "assistant_message"))
    loop.run_until_complete(obs_mod._persist(
        state, ObserverDecision(action="noop", reason="fast-path: x", fast_path=True),
        "tool_result"))
    loop.run_until_complete(obs_mod._persist(
        state, ObserverDecision(action="noop", reason="on task"), "assistant_message"))
    got = [r["safeguard"] for r in recorded]
    assert got == ["stall_rescue", "fast_path", None], got
