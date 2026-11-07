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
and no test asserted the research passes used it. Three halves are pinned here:

  * the passes resolve citations **before** they synthesize, as a pointer to that
    skill rather than a second copy of its recipes (two copies re-fragment — that
    is what the 2026-09-04 tool-name cleanup cost),
  * the resolver behind the step actually catches a wrong number, including
    through the command line the skills advertise, and
  * the step's own truth probe and the CLI read ONE field for whether GitHub
    answered at all. On 2026-11-07 the box's 60-per-hour anonymous budget for
    ``api.github.com`` was spent, both of the probe's fetches came back HTTP 403,
    and the probe reported fixtures that were correct as wrong — because it could
    only notice a socket failure, having string-matched ``"could not reach"``.

Assertions about the skills go through ``agent_mcp.skills._load_skill`` rather than
``read_text``: a skill whose frontmatter breaks, or whose ``status`` turns to
quarantined, is invisible to the model while still sitting on disk looking
maintained. That is the failure this item exists to prevent, so it is the failure
the test has to be able to see.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import shlex
import sys
import urllib.error
import urllib.request
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
        f"quarantined status, which means the model never sees it no matter what "
        f"the file says")
    # A frontmatter that fails to parse is swallowed: `_parse_frontmatter`
    # returns {} and the skill loads with an empty description and no tags, so
    # skills_search can never match it while it still reads as installed.
    assert str(skill.get("description") or "").strip(), (
        f"{path.parent.name} loads with an empty description — its frontmatter "
        f"does not parse, so prefetch can never surface the step this test pins")
    return skill["body"]


def _load_resolver():
    spec = importlib.util.spec_from_file_location("verify_cited_identifier", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResponse(io.BytesIO):
    """The file-like object ``urlopen`` hands to ``json.load``, usable as a
    context manager like the real thing, so a fetch can be exercised with no
    network and no credential in sight."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


#: One real record, as the API returns it, for the nodes that drive a whole
#: fetch through a stubbed ``urlopen``.
_OK_BODY = json.dumps({
    "number": 34223,
    "title": "[Kernels] Make GGUF linear method allow 3d inputs",
    "state": "closed", "merged": False, "merged_at": None,
    "html_url": "https://github.com/vllm-project/vllm/pull/34223",
}).encode()


# ---------------------------------------------------------------------------
# Fixtures: records frozen from api.github.com on 2026-09-09.
# test_the_frozen_fixtures_still_match_the_records_on_github re-checks them
# against the live API: it fails if the API ANSWERED and a record has drifted,
# and skips only when nothing was answered (no route, or a quota refusal) — a
# frozen fixture that has quietly drifted would otherwise assert something untrue,
# and a refusal says nothing about drift either way.
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
    Skips rather than fails when GitHub gave this run no ANSWER — no route, or a
    quota refusal — and never skips the claim away when the record actually
    changed.

    Both directions matter: it is this probe, run before the round opened, that
    caught an earlier note describing #50233 as deferring ``finish_group`` when
    its title is about overlapping GDN decode and prefill kernels.

    The arm that skips is deliberately the one keyed on ``vci.answered()`` rather
    than on ``found`` or on the error's wording. On 2026-11-07 the anonymous
    hourly budget for ``api.github.com`` on this box was exhausted, both fetches
    returned HTTP 403, and this node failed the suite claiming the fixtures were
    wrong when an authenticated request the same hour confirmed both titles and
    both merge states exactly as frozen. A refusal is not evidence about a record.
    """
    vci = _load_resolver()
    live = {n: vci.fetch_record("vllm-project/vllm", n) for n in (34223, 50233)}
    no_answer = {n: r.get("error") for n, r in live.items() if not vci.answered(r)}
    if no_answer:
        pytest.skip(f"api.github.com answered nothing to this run "
                    f"(no route or quota spent): {no_answer}")
    answered_wrong = {n: r for n, r in live.items() if not r.get("found")}
    assert not answered_wrong, (
        f"the API answered and the frozen fixtures are wrong: {answered_wrong}")
    assert live[34223]["title"] == WRONG_PR["title"]
    assert live[34223]["merged"] is False, "the false-citation fixture must stay unmerged"
    assert live[50233]["title"] == RIGHT_PR["title"]


def test_a_refusal_is_never_readable_as_an_answer(tmp_path, monkeypatch):
    """The fetch has to say WHICH kind of ``found: False`` it is returning, or
    every consumer guesses — and the one consumer that guessed read a rate limit
    as fixture drift on 2026-11-07. A 404 is an answer (there is no such PR); a
    403/429/5xx and a dead socket are not.

    ``urlopen`` is replaced rather than the network being trusted, so this node
    fails on a regression in the mapping and not on the weather outside.
    """
    vci = _load_resolver()

    def refuse(code):
        def _open(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, "refused", {}, None)
        return _open

    for code, is_answer in ((403, False), (429, False), (500, False), (404, True)):
        monkeypatch.setattr(urllib.request, "urlopen", refuse(code))
        rec = vci.fetch_record("vllm-project/vllm", 34223)
        assert rec["found"] is False, f"HTTP {code} reported as a record"
        assert vci.answered(rec) is is_answer, (
            f"HTTP {code}: answered() said {vci.answered(rec)}; a quota refusal "
            f"must not be an answer and a 404 must be one")

    def die(req, timeout=None):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(urllib.request, "urlopen", die)
    rec = vci.fetch_record("vllm-project/vllm", 34223)
    assert vci.answered(rec) is False, "a socket failure reported as an answer"
    # The frozen fixture dicts the other nodes hand in carry no `answered` key;
    # the predicate has to read them as answers, or the drift probe would skip
    # its own assertions on the way past a hand-built record.
    assert vci.answered(dict(WRONG_PR)) is True


def test_the_probe_finds_the_token_across_the_gate_child_env(tmp_path, monkeypatch):
    """The seam this whole fix crosses: the gate runs candidate tests with
    ``PATH=/usr/bin:/bin`` and ``HOME`` moved to the round home
    (``scripts/automod/gate.py::_child_env``, whose home farm symlinks every real
    entry but ``lloyd`` and ``lloyd-data``). So the only credential this box has —
    the token ``gh`` holds, since ``config.yaml``'s ``token:`` is empty and no
    ``GITHUB_TOKEN`` is exported — is reachable through ``$HOME`` and not through
    ``PATH``. That environment is pinned here, not assumed from a shell.

    ``shutil.which`` is stubbed to find nothing, which is what the gate child's
    PATH does on this box; the stub keeps the node about the fallback rather than
    about which directories happen to hold a ``gh``.
    """
    vci = _load_resolver()
    shim = tmp_path / ".local" / "share" / "mise" / "shims" / "gh"
    shim.parent.mkdir(parents=True)
    shim.write_text("#!/bin/sh\necho gho_faketoken\n", encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setattr(vci.shutil, "which", lambda name: None)
    assert vci.gh_cli(home=str(tmp_path)) == str(shim)
    assert vci.github_token({"HOME": str(tmp_path)}) == "gho_faketoken", (
        "the credential behind the gate child env's PATH was not found, so the "
        "fixture probe above would be spending the anonymous budget again")


def test_an_exported_token_beats_the_cli_and_no_token_is_not_an_error(
        tmp_path, monkeypatch):
    """Order and the empty case. An explicitly exported token is the caller's
    stated intent and wins over an installed CLI; and a box with neither gets an
    anonymous request rather than a crash — discovery runs inside every fetch, so
    a missing ``gh`` must not become a fetch failure on a fresh machine."""
    vci = _load_resolver()
    shim = tmp_path / ".local" / "share" / "mise" / "shims" / "gh"
    shim.parent.mkdir(parents=True)
    shim.write_text("#!/bin/sh\necho gho_from_cli\n", encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setattr(vci.shutil, "which", lambda name: None)
    assert vci.github_token({"HOME": str(tmp_path), "GITHUB_TOKEN": "gho_exported"}) \
        == "gho_exported"

    empty_home = tmp_path / "no-credential-here"
    assert vci.github_token({"HOME": str(empty_home)}) == ""

    # The unguarded arm: `fetch_record` calls `github_token()` with no dict, so
    # what it finds is whatever os.environ and the real home hold. Both are put
    # into the no-credential state here, and the fetch that follows must still
    # report the record it was given — anonymously, with no header invented.
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    seen: list[urllib.request.Request] = []

    def capture(req, timeout=None):
        seen.append(req)
        return _FakeResponse(_OK_BODY)

    monkeypatch.setattr(urllib.request, "urlopen", capture)
    rec = vci.fetch_record("vllm-project/vllm", 34223)
    assert rec["found"] is True and vci.answered(rec) is True, rec
    assert seen[-1].get_header("Authorization") is None, (
        "a box with no credential sent an Authorization header anyway")
