"""Backlog #393 — the repetition guard may only claim what it actually observes.

`guards.repetition_verdict` compares `ToolCallSignature`: tool name, normalized
arguments, and identifiers extracted from argument *values*. There is no result
field on a signature and result text is never compared — yet the injected text
asserted "and the result has not changed".

Measured on the three sessions behind the 2026-09-05 trajectories: the guard
fired 5 times, while a result-level check (token-set Jaccard > 0.85 of each tool
result against every earlier result) found ZERO unchanged-result repeats in two
of the three sessions — 0 near-duplicates in 66 calls (iv5174), 0 in 103
(ivbf4f). The only near-duplicate results anywhere in the sample were TodoWrite
echoes. The guard was matching *command shape* — repeated `grep -n … <symbol>`
probes over different symbols — and reporting it as repeated results.

Two costs: an inject that asserts the unobserved trains the primary to discount
injects, and nightly-skills-management (#83) Stage 2 treats injected corrections
as user corrections, so a false fire becomes a false skill.

Fix is #393 branch 2: the wording now states only the observable — N
near-identical queries issued. Branch 1 (a real result-hash check) is what earns
back the "result has not changed" sentence, and
`test_no_result_claim_without_a_result_field` releases the ban automatically if
`ToolCallSignature` ever grows a result-bearing field.

Run:
  .venvs/lloyd/bin/python -m pytest tests/integration/test_iv_repetition_wording.py -q
"""

from __future__ import annotations

import re
import sys
from dataclasses import fields
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))

from app.inner_voice import guards  # noqa: E402

REPO = "/home/alansrobotlab/lloyd"


# ---------------------------------------------------------------------------
# What counts as an unobserved claim
# ---------------------------------------------------------------------------

# A sentence asserting that the tool OUTPUTS did not change / were identical.
# Legal only once something on the signature actually carries result content.
_RESULT_CLAIM_RE = re.compile(
    r"\b(?:the\s+)?(?:results?|outputs?)\b"          # subject: the results…
    r"[^.]{0,24}?"                                   # …within the same clause…
    r"\b(?:has|have|had|is|are|was|were)\s*"
    r"(?:not|n't)\s+"                                # …denied a change…
    r"(?:changed|changing|differed|moving)\b",
    re.IGNORECASE,
)

# The exact sentence that shipped until #393, kept verbatim so a reverts-can't-
# come-back check does not depend on the regex above staying clever.
_SHIPPED_FALSE_CLAIM = "the result has not changed"


def _sig(tool: str, args: dict) -> guards.ToolCallSignature:
    return guards.tool_call_signature(tool, args)


def _relabelled(sigs, tool: str = "Grep"):
    """Same arguments/identifiers, under a tool name that near-matching allows.

    The counterfactual half of the `_EXACT_ONLY_TOOLS` pin: if the guard stayed
    silent for a path-addressed tool only because comparison is broken, this
    would stay silent too.
    """
    return [
        guards.ToolCallSignature(
            tool=tool, exact=s.exact, idents=s.idents, preview=s.preview
        )
        for s in sigs
    ]


def _all_inject_texts() -> list[tuple[str, str]]:
    """(label, rendered inject) for both verdict shapes."""
    near = guards.repetition_verdict([
        _sig("Bash", {"command": 'grep -rn iv_inject_queue app/ --include="*.py"'}),
        _sig("Bash", {"command": "ls -d /home/alansrobotlab/lloyd/app"}),
        _sig("Bash", {"command": 'grep -rn "iv_inject_queue" app/; echo EXIT=$?'}),
        _sig("Bash", {"command": "grep -rn --include='*' iv_inject_queue app/"}),
    ])
    assert near is not None and not near.exact, "fixture must produce a near verdict"
    exact = guards.repetition_verdict([
        _sig("Read", {"file_path": f"{REPO}/app/inner_voice/guards.py"})
        for _ in range(3)
    ])
    assert exact is not None and exact.exact, "fixture must produce an exact verdict"
    return [
        ("near", guards.repetition_inject_content(near)),
        ("exact", guards.repetition_inject_content(exact)),
    ]


# ---------------------------------------------------------------------------
# The false claim
# ---------------------------------------------------------------------------


def test_no_result_claim_without_a_result_field():
    """The inject cannot assert anything about results until it sees results.

    Self-releasing: if `ToolCallSignature` grows a field carrying result
    content (#393 branch 1), the ban lifts and this test only asks that the
    claim be honest either way.
    """
    observes_results = any(
        "result" in f.name.lower() or "output" in f.name.lower()
        for f in fields(guards.ToolCallSignature)
    )
    assert not observes_results, (
        "the signature now carries result content — branch 1 landed; update this "
        "test to check the claim against that field instead of banning it"
    )
    for label, content in _all_inject_texts():
        assert _SHIPPED_FALSE_CLAIM not in content, f"{label}: {_SHIPPED_FALSE_CLAIM!r}"
        assert not _RESULT_CLAIM_RE.search(content), f"{label}: {content}"
    print("test_no_result_claim_without_a_result_field: OK")


def test_near_repeat_says_queries_were_issued_not_results_repeated():
    """What IS observable: the primary issued N near-identical calls."""
    label, content = _all_inject_texts()[0]
    assert "near-identical" in content, content
    assert re.search(r"\b3\b", content), f"must count the calls: {content}"
    # A near match is not a verbatim re-run; it must not claim one.
    assert "the same call" not in content, content
    # The loop-breaking guidance survives — it just stops pretending to be a
    # measurement. The primary knows what its calls returned; the guard does not.
    assert "ANSWER" in content, content
    print("test_near_repeat_says_queries_were_issued_not_results_repeated: OK")


def test_exact_repeat_still_names_the_same_call():
    label, content = _all_inject_texts()[1]
    assert "the same call" in content, content
    assert "ANSWER" in content, content
    print("test_exact_repeat_still_names_the_same_call: OK")


def test_shared_terms_still_lead_the_message():
    """The rarest shared identifier must stay in the first line.

    `test_iv_loop_guards.test_shared_terms_lead_with_the_rarest` pins the
    ordering; this pins that the reworded lead-in did not push the named term
    past the point where the primary stops reading.
    """
    cmds = [
        "grep -rn zzq_phantom_handle_v3 /home/alansrobotlab/lloyd --include=*.py",
        "ls -la /home/alansrobotlab/lloyd; ls -d /home/alansrobotlab/lloyd/app",
        'grep -rn "zzq_phantom_handle_v3" /home/alansrobotlab/lloyd; echo EXIT=$?',
        "grep -rn --include='*' \"zzq_phantom_handle_v3\" /home/alansrobotlab/lloyd/app",
    ]
    verdict = guards.repetition_verdict([_sig("Bash", {"command": c}) for c in cmds])
    assert verdict is not None
    assert verdict.shared_terms[0] == "zzq_phantom_handle_v3", verdict.shared_terms
    content = guards.repetition_inject_content(verdict)
    assert content.index("zzq_phantom_handle_v3") < 120, content
    print("test_shared_terms_still_lead_the_message: OK")


# ---------------------------------------------------------------------------
# `_EXACT_ONLY_TOOLS` behaviour — pinned unchanged by #393
# ---------------------------------------------------------------------------

# Two distinctive identifiers in a shared directory component, so sibling calls
# of any tool share ≥ 2 identifiers at containment 1.0. That is a near match
# under the general rule; an exact-only tool must still stay silent.
_SHARED_DIR = f"{REPO}/scratch/iv_inject_queue_zzq_phantom_handle_v3/observer_prompt_fixtures"


def _path(name: str) -> str:
    return f"{_SHARED_DIR}/{name}"


# Per tool: three sibling calls that are NOT byte-identical but share the
# distinctive identifiers. Under near-matching every one of these fires.
_SIBLING_ARGS: dict[str, list[dict]] = {
    "Read": [{"file_path": _path(n)} for n in ("a.py", "b.py", "c.py")],
    "Write": [
        {"file_path": _path(n), "content": f"x = {i}\n"}
        for i, n in enumerate(("a.py", "b.py", "c.py"))
    ],
    "Edit": [
        {"file_path": _path(n), "old_string": f"old {i}", "new_string": f"new {i}"}
        for i, n in enumerate(("a.py", "b.py", "c.py"))
    ],
    "MultiEdit": [
        {"file_path": _path(n),
         "edits": [{"old_string": f"old {i}", "new_string": f"new {i}"}]}
        for i, n in enumerate(("a.py", "b.py", "c.py"))
    ],
    "NotebookEdit": [
        {"notebook_path": _path(n), "new_source": f"print({i})", "cell_id": f"c{i}"}
        for i, n in enumerate(("a.ipynb", "b.ipynb", "c.ipynb"))
    ],
    "Glob": [
        {"pattern": f"**/{stem}*.py", "path": _path("sub")}
        for stem in ("alpha", "beta", "gamma")
    ],
    "TodoWrite": [
        {"todos": [{"content": "wire iv_inject_queue",
                    "status": s, "activeForm": f"status {i}"}]}
        for i, s in enumerate(("pending", "in_progress", "completed"))
    ],
}


def test_exact_only_membership():
    """The set #393 was verified against is still the set.

    Superset check, not equality: adding a tool to exact-only is a legitimate
    tightening and must not fail this; dropping one silently re-arms the 19
    false injects of 2026-09-04 and must.
    """
    assert guards._EXACT_ONLY_TOOLS >= {
        "Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Glob", "TodoWrite",
    }, sorted(guards._EXACT_ONLY_TOOLS)
    print("test_exact_only_membership: OK")


def test_exact_only_tools_do_not_near_match():
    """Distinct targets of a path-addressed tool never fire — still."""
    for tool, arg_sets in _SIBLING_ARGS.items():
        assert tool in guards._EXACT_ONLY_TOOLS, f"{tool} dropped out of exact-only"
        sigs = [_sig(tool, a) for a in arg_sets]
        for i in range(2, len(sigs) + 1):
            verdict = guards.repetition_verdict(sigs[:i])
            assert verdict is None, f"{tool}: fired on siblings — {verdict}"
        # …and the same identifiers WOULD fire once near-matching is allowed, so
        # the silence is attributable to `_EXACT_ONLY_TOOLS` and nothing else.
        control = guards.repetition_verdict(_relabelled(sigs))
        assert control is not None, (
            f"{tool}: counterfactual failed — the fixture no longer shares "
            "enough for a near match, so the test above proves nothing"
        )
    print("test_exact_only_tools_do_not_near_match: OK")


def test_exact_only_tools_still_catch_verbatim_repeats():
    """Byte-identical re-runs of an exact-only tool must still fire, exactly."""
    for tool, arg_sets in _SIBLING_ARGS.items():
        once = _sig(tool, arg_sets[0])
        for n in (3, 4):
            verdict = guards.repetition_verdict([once] * n)
            assert verdict is not None, f"{tool}: {n} verbatim repeats ignored"
            assert verdict.exact, f"{tool}: verbatim repeat not marked exact"
            assert "the same call" in guards.repetition_inject_content(verdict)
    print("test_exact_only_tools_still_catch_verbatim_repeats: OK")


def test_firing_counts_are_unchanged_by_the_rewording():
    """Branch 2 changes text only — the guard's judgment must be bit-identical.

    Replays the real 28-call loop from turn 20260905_011748_iv84e4 through the
    same ring the pretool hook maintains and pins the fire points.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fixture_iv_loop_turn import REAL_BASH_CALLS

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
    assert fired_at == [62, 76], f"fire points moved: {fired_at}"
    print("test_firing_counts_are_unchanged_by_the_rewording: OK")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
