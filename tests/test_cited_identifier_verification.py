"""A cited identifier has to be resolved before a claim rests on it. Backlog #447.

A 2026-09-06 research card cited vLLM **PR #34223** as evidence that shared-expert
overlap had been built. The API record is ``[Kernels] Make GGUF linear method
allow 3d inputs`` — closed **unmerged** 2026-07-13 as a conflicting draft. The
card's claim was not slightly wrong: the work it described does not exist at that
number. The same card called Qwen3.5-27B an MoE model, and its HF config carries
no expert keys at all. Memory made the same class of mistake with a commit hash —
``96472427`` credited with the ``.bak`` rotation fix does not exist in the repo;
the pickaxe names ``bbe5689``.

The rule already existed as ``cited-identifier-verification``, but nothing outside
its own file pointed at it, so it fired only if keyword prefetch happened to match,
and no test asserted the research passes used it. Two halves are pinned here:

  * the passes resolve citations **before** they synthesize, as a pointer to that
    skill rather than a second copy of its recipes (two copies re-fragment — that
    is what the 2026-09-04 tool-name cleanup cost), and
  * the resolver behind the step actually catches a wrong number, including
    through the command line the skills advertise.

Assertions about the skills go through ``agent_mcp.skills._load_skill`` rather than
``read_text``: a skill whose frontmatter breaks, or whose ``status`` turns to
quarantined, is invisible to the model while still sitting on disk looking
maintained. That is the failure this item exists to prevent, so it is the failure
the test has to be able to see.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VAULT = Path.home() / "obsidian"
SCRIPT = ROOT / "scripts" / "verify_cited_identifier.py"

DEEP_RESEARCH = VAULT / "skills" / "deep-research" / "SKILL.md"
RESEARCH_AGENT = VAULT / "skills" / "research-agent" / "SKILL.md"
VERIFIER = VAULT / "skills" / "cited-identifier-verification" / "SKILL.md"

PASSES = [DEEP_RESEARCH, RESEARCH_AGENT]

#: The step the two passes must carry. Worded once, here and in the skills.
_STEP = re.compile(r"resolve every cited identifier", re.IGNORECASE)
#: Where each pass synthesizes. The step must land before it.
_SYNTHESIS = re.compile(r"Phase 4: Synthesis|Step 5: Synthesize")

#: Tells that belong to `cited-identifier-verification` alone. If one shows up in
#: a research pass, the pointer was replaced by a copy and the recipes are on
#: their way to disagreeing with each other.
_RECIPE_TELLS = ("num_experts", "api.github.com", "moe_intermediate_size")

#: The advertised invocation, as it appears in all three skills.
_INVOCATION = re.compile(r"(~/\S*verify_cited_identifier\.py)\s+(pr\s+.+)")


def _load_skill_body(path: Path) -> str:
    """The skill body as the live loader hands it to the model."""
    from agent_mcp.skills import _load_skill

    skill = _load_skill(path.parent)
    assert skill, (
        f"{path.parent.name} does not load through agent_mcp.skills._load_skill — "
        f"broken frontmatter or quarantined status, which means the model never "
        f"sees it no matter what the file says")
    return skill["body"]


def _load_resolver():
    spec = importlib.util.spec_from_file_location("verify_cited_identifier", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures: records frozen from api.github.com on 2026-09-09.
# test_the_frozen_fixtures_still_match_github re-checks them against the live
# API and skips if the network is down — a frozen fixture that has quietly
# drifted would otherwise assert something untrue.
# ---------------------------------------------------------------------------

#: The citation that fooled a research card. Deliberately wrong on purpose.
WRONG_PR = {
    "found": True,
    "repo": "vllm-project/vllm",
    "number": 34223,
    "title": "[Kernels] Make GGUF linear method allow 3d inputs",
    "state": "closed",
    "merged": False,
    "merged_at": None,
    "html_url": "https://github.com/vllm-project/vllm/pull/34223",
}
WRONG_CLAIM = "GDN + MoE shared-expert overlap"

#: A real overlap PR, open and unmerged, for the "right citation is not flagged"
#: and "the landed check bites" cases.
RIGHT_PR = {
    "found": True,
    "repo": "vllm-project/vllm",
    "number": 50233,
    "title": "[Model][Perf] Overlap mixed GDN decode and prefill recurrent kernels",
    "state": "open",
    "merged": False,
    "merged_at": None,
    "html_url": "https://github.com/vllm-project/vllm/pull/50233",
}
RIGHT_CLAIM = "GDN decode and prefill overlap"


# ---------------------------------------------------------------------------
# The two research passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skill", PASSES, ids=["deep-research", "research-agent"])
def test_the_pass_resolves_cited_identifiers_before_it_synthesizes(skill):
    """The step must exist *and* precede synthesis. After synthesis, a bad
    citation has already been written into the note's Summary."""
    assert skill.exists(), f"{skill} is missing"
    body = _load_skill_body(skill)
    step = _STEP.search(body)
    assert step, (
        f"{skill.parent.name} lost the pre-synthesis identifier-resolution step. "
        f"It is what stops a card citing a closed-unmerged PR as evidence.")
    synthesis = _SYNTHESIS.search(body)
    assert synthesis, f"{skill.parent.name} no longer has a recognizable synthesis step"
    assert step.start() < synthesis.start(), (
        f"{skill.parent.name} resolves identifiers after it synthesizes, which is "
        f"after the claims are already written")


@pytest.mark.parametrize("skill", PASSES, ids=["deep-research", "research-agent"])
def test_the_step_points_at_the_skill_instead_of_copying_it(skill):
    """One copy of the recipes. The 2026-09-04 phantom-tool cleanup is the
    precedent for what two copies become."""
    body = _load_skill_body(skill)
    assert "cited-identifier-verification" in body, (
        f"{skill.parent.name} must name the skill the step defers to, or the step "
        f"is an instruction with no procedure behind it")
    copied = [tell for tell in _RECIPE_TELLS if tell in body]
    assert not copied, (
        f"{skill.parent.name} carries the resolver's recipes ({copied}) instead of "
        f"pointing at cited-identifier-verification")


@pytest.mark.parametrize("skill", PASSES, ids=["deep-research", "research-agent"])
def test_the_command_the_pass_advertises_is_the_one_that_exists(skill):
    """A skill that names a script nobody wrote is a run that fails at the
    moment it matters — the same defect class as naming a tool that does not
    exist, which test_skill_tool_names already bans."""
    body = _load_skill_body(skill)
    hit = _INVOCATION.search(body)
    assert hit, f"{skill.parent.name} advertises no resolvable verify_cited_identifier command"
    script_path, argv = hit.group(1), hit.group(2)
    assert script_path.endswith("scripts/verify_cited_identifier.py"), script_path
    assert SCRIPT.exists(), f"{SCRIPT} is the script the skills point at"

    # Substitute the placeholders the skills are written with, then prove the
    # advertised invocation parses against the real parser.
    line = (argv.replace("<owner/repo>", "vllm-project/vllm")
                .replace("<number>", "34223"))
    line = re.sub(r"<[^>]*>", "a-claim-in-prose", line)
    vci = _load_resolver()
    parsed = vci.build_parser().parse_args(shlex.split(line))
    assert parsed.command == "pr", parsed
    assert parsed.repo == "vllm-project/vllm" and parsed.number == 34223, parsed
    assert parsed.claim == "a-claim-in-prose", parsed


def test_the_resolver_names_the_passes_that_call_it():
    """Reciprocal discoverability. The skill pointed at the passes one-way, so
    the passes fired it by luck; a reader arriving from either direction has to
    find the other."""
    assert VERIFIER.exists(), "cited-identifier-verification went missing"
    body = _load_skill_body(VERIFIER)
    related = body.split("## Related skills", 1)
    assert len(related) == 2, "the skill lost its Related skills section"
    for name in ("deep-research", "research-agent"):
        assert name in related[1], (
            f"{name} is not named in cited-identifier-verification's Related skills")


def test_the_deep_research_quality_gate_makes_it_a_completion_condition():
    """A step in the middle of a 5-phase methodology is advisory. A line in the
    gate is what makes skipping it visible."""
    body = _load_skill_body(DEEP_RESEARCH)
    gates = body.split("## Quality Gates", 1)
    assert len(gates) == 2, "deep-research lost its Quality Gates section"
    assert "cited-identifier-verification" in gates[1], (
        "citation resolution is not a completion gate")


# ---------------------------------------------------------------------------
# The resolver behind the step
# ---------------------------------------------------------------------------


def test_the_resolver_exists():
    assert SCRIPT.exists(), f"{SCRIPT} is missing"


def test_a_wrong_pr_number_is_caught():
    """The item's fixture: vLLM #34223 cited as shared-expert overlap. Four
    content words, none of them in the real title."""
    vci = _load_resolver()
    matched, unmatched = vci.title_matches_claim(WRONG_PR["title"], WRONG_CLAIM)
    assert not matched
    assert {"gdn", "moe", "overlap"} <= set(unmatched), unmatched

    reasons, _notes = vci.check_pr_claim(WRONG_PR, WRONG_CLAIM)
    assert reasons, "the false citation was not caught"


def test_the_wrong_citation_is_caught_through_the_advertised_command():
    """Not just the library call — the exit code the skill tells the model to
    read."""
    vci = _load_resolver()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(vci, "fetch_record", lambda *a, **k: dict(WRONG_PR))
    code = vci.main(["pr", "vllm-project/vllm", "34223", "--claim", WRONG_CLAIM])
    monkey.undo()
    assert code == vci.MISMATCH, f"exit {code}; a wrong citation must not exit 0"


def test_a_correct_citation_is_not_flagged():
    """The counter-case, so the resolver is a check and not a refusal. A
    paraphrase of the real title has to pass, or the model will learn to ignore
    the step."""
    vci = _load_resolver()
    reasons, _notes = vci.check_pr_claim(RIGHT_PR, RIGHT_CLAIM)
    assert reasons == [], reasons

    monkey = pytest.MonkeyPatch()
    monkey.setattr(vci, "fetch_record", lambda *a, **k: dict(RIGHT_PR))
    code = vci.main(["pr", "vllm-project/vllm", "50233", "--claim", RIGHT_CLAIM])
    monkey.undo()
    assert code == vci.OK


def test_claiming_work_landed_when_it_never_merged_is_a_distinct_verdict():
    """"#34223 introduced X" is a stronger claim than "PR #34223 exists", and the
    card needed the merge, not the number."""
    vci = _load_resolver()
    reasons, _ = vci.check_pr_claim(WRONG_PR, "GGUF linear method allow 3d inputs",
                                    assert_landed=True)
    assert reasons and any("unmerged" in r for r in reasons), reasons
    assert vci.check_pr_claim(WRONG_PR, "GGUF linear method allow 3d inputs")[0] == [], \
        "the same citation is fine once the landed-work assertion is dropped"

    monkey = pytest.MonkeyPatch()
    monkey.setattr(vci, "fetch_record", lambda *a, **k: dict(WRONG_PR))
    code = vci.main(["pr", "vllm-project/vllm", "34223", "--assert-landed"])
    monkey.undo()
    assert code == vci.NOT_LANDED


def test_an_unresolvable_citation_is_not_recorded_as_a_pass():
    """Exit 0 on a network failure is how a verification step quietly becomes a
    no-op: the model sees success and moves on."""
    vci = _load_resolver()
    miss = {"found": False, "repo": "vllm-project/vllm", "number": 99999999,
            "error": "no such PR in that repo"}
    reasons, _ = vci.check_pr_claim(miss, "anything")
    assert reasons

    monkey = pytest.MonkeyPatch()
    monkey.setattr(vci, "fetch_record", lambda *a, **k: dict(miss))
    code = vci.main(["pr", "vllm-project/vllm", "99999999", "--claim", "anything"])
    monkey.undo()
    assert code == vci.UNRESOLVED


def test_a_claim_with_no_content_words_cannot_be_verified():
    """The degenerate input: an empty claim has nothing to falsify, so the check
    passes vacuously. The skill answers this by requiring the claim be recorded,
    which is a gate on the step, not on this function."""
    vci = _load_resolver()
    matched, _ = vci.title_matches_claim(WRONG_PR["title"], "")
    assert matched is True


# ---------------------------------------------------------------------------
# The fixtures' own truth
# ---------------------------------------------------------------------------


def test_the_frozen_fixtures_still_match_the_records_on_github():
    """These assertions are only honest while the PRs say what is written above.
    Skips rather than fails when there is no network — but does not skip the
    claim away silently if the record has actually changed.

    Both directions matter: it is this probe, run before the round opened, that
    caught an earlier note describing #50233 as deferring ``finish_group`` when
    its title is about overlapping GDN decode and prefill kernels.
    """
    vci = _load_resolver()
    live = {n: vci.fetch_record("vllm-project/vllm", n) for n in (34223, 50233)}
    if any(not r.get("found") for r in live.values()):
        errs = [r.get("error") for r in live.values() if not r.get("found")]
        if any("could not reach" in str(e) for e in errs):
            pytest.skip(f"no route to api.github.com from this test run: {errs}")
        pytest.fail(f"the API answered and the frozen fixtures are wrong: {live}")
    assert live[34223]["title"] == WRONG_PR["title"]
    assert live[34223]["merged"] is False, "the false-citation fixture must stay unmerged"
    assert live[50233]["title"] == RIGHT_PR["title"]
