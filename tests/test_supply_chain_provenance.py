"""Clauses 4 and 5 of #688: the dispatch-time install-provenance check.

Two separate claims, deliberately tested apart:

* The *policy* (`evaluate_provenance` / `check_install_provenance`) — a
  distribution new to the dependency set is refused when the registry has never
  heard of it, when its first release is under 90 days old, or when it has fewer
  than two releases; the printed reason names which fact failed; and an explicit
  override lets it through. Tested against a scripted registry so the numbers are
  exact.
* The *matcher* (`find_install_commands`) — what counts as an install command.
  This is the half that decides the false-positive rate: a substring match on
  `pip install` fires on `grep 'pip install' notes.md`, and a matcher that fires
  on prose is a check people override reflexively, which is the same as no check.

The registry is always injected. Nothing in this file may touch pypi.org: the
gate runs on a tree that assumes no outbound access, and a live GET in a test is
a flake with a network in it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.harness import supply_chain as sc
from app.harness.supply_chain import (
    MappingRegistry, Thresholds, check_install_provenance, evaluate_provenance,
    format_provenance_verdicts,
)

NOW = dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc)
BACKGROUND = "20260924_054120_autonomy_688"
DEPENDENCY_SET = {"httpx": "requirements.txt", "fastapi": "requirements.txt",
                  "pyyaml": "requirements.txt", "mcp": "requirements.txt"}


def _registry(**facts):
    """`_registry(freshpkg=(True, "2018-01-01T00:00:00Z", 40))`, per name."""
    payload = {}
    for name, (exists, first, count) in facts.items():
        payload[name] = sc.RegistryFacts(
            name=name, exists=exists,
            first_release=(dt.datetime.fromisoformat(first)
                           .replace(tzinfo=dt.timezone.utc) if first else None),
            release_count=count, source="test", observed_at=NOW)
    return MappingRegistry(payload, name="test")


def _check(command, registry=None, *, session=BACKGROUND, deps=DEPENDENCY_SET):
    return check_install_provenance(command, session,
                                    registry=(registry if registry is not None
                                              else _registry()),
                                    dependency_set=deps, now=NOW)


# ── clause 4: the policy ────────────────────────────────────────────────────

def test_a_name_the_registry_has_never_heard_of_is_refused_as_unpublished():
    """The slop-squat signature: the model invented a name, so the registry 404s.

    This is the case a test-based gate is structurally blind to — the package
    installs, imports, and its tests pass — so it is the one case that must be
    refused rather than warned about.
    """
    result = _check("pip install graphy", _registry())

    assert result.refusal
    assert "graphy" in result.refusal
    assert "not published on pypi.org" in result.refusal, result.refusal


def test_first_release_under_ninety_days_is_refused_and_the_reason_names_that_fact():
    reg = _registry(freshpkg=(True, "2026-09-17T00:00:00Z", 30))

    result = _check("pip install freshpkg", reg)

    assert result.refusal and "freshpkg" in result.refusal
    assert "first published 7 day" in result.refusal, result.refusal
    assert "threshold: 90 days" in result.refusal
    assert "release(s) on pypi.org" not in result.refusal, (
        "the reason names the fact that failed, not every fact it looked at")


def test_fewer_than_two_releases_is_refused_and_the_reason_names_that_fact():
    reg = _registry(singlepkg=(True, "2019-01-01T00:00:00Z", 1))

    result = _check("pip install singlepkg", reg)

    assert result.refusal and "singlepkg" in result.refusal
    assert "1 release(s)" in result.refusal, result.refusal
    assert "first published" not in result.refusal


def test_exactly_two_releases_and_an_old_first_release_is_allowed():
    """The boundary is `count < 2`, so 2 passes — pinned because an off-by-one here
    blocks every single-release CLI a research agent legitimately wants, and a
    blocked-then-overridden check is a blocked check."""
    reg = _registry(newish=(True, "2019-01-01T00:00:00Z", 2))

    assert _check("pip install newish", reg).refusal is None


def test_exactly_ninety_days_passes_and_one_day_under_does_not():
    old = _registry(p=(True, "2026-06-26T12:00:00Z", 3))    # exactly 90 days
    under = _registry(p=(True, "2026-06-27T12:00:00Z", 3))  # 89 days

    assert _check("pip install p", old).refusal is None
    refused = _check("pip install p", under)
    assert refused.refusal and "first published 89 day" in refused.refusal


def test_a_name_already_declared_by_the_repo_is_never_checked_against_the_registry():
    """Clause 5's second half at the policy level: `requirements.txt` is read from
    the file, and a declared name is not "new", so re-installing what this repo
    already ships cannot be blocked by a registry fact — not even by `exists:
    false`, which is what a failed lookup degrades to."""
    result = _check("pip install httpx", _registry())

    assert result.refusal is None
    assert [r.name for r in result.declared] and not result.cleared, (
        "allowed by the declaration itself, not by a registry verdict")


def test_an_override_with_a_reason_lets_the_name_through_and_says_so_loudly():
    reg = _registry(researchy=(True, "2026-09-20T00:00:00Z", 1))

    result = _check('LLOYD_DEP_OVERRIDE="new library for the routing eval" '
                    'pip install researchy', reg)

    assert result.refusal is None, result.refusal
    assert [r.name for r in result.cleared] == ["researchy"]
    assert result.overrides == [("researchy", "new library for the routing eval")]
    text = format_provenance_verdicts(result)
    assert "OVERRIDE" in text and "routing eval" in text, (
        "an override that prints nothing leaves no trace to trend later")


def test_an_override_needs_a_reason():
    reg = _registry(researchy=(True, "2026-09-20T00:00:00Z", 1))

    bare = _check("LLOYD_DEP_OVERRIDE= pip install researchy", reg)
    quoted_empty = _check('LLOYD_DEP_OVERRIDE="" pip install researchy', reg)

    assert bare.refusal and quoted_empty.refusal
    assert "no reason" in bare.refusal, bare.refusal
    assert "first published" in bare.refusal, "the failed fact still has to be named"


def test_the_override_only_covers_the_names_on_its_own_segment():
    """Scope, per call: an override on one install does not licence a second one in
    the same command, or a single override would outlive the decision it was given
    for — `env KEY=val cmd` is scoped the same way."""
    reg = _registry(good=(True, "2026-09-20T00:00:00Z", 1),
                    bad=(True, "2026-09-21T00:00:00Z", 1))

    result = _check('LLOYD_DEP_OVERRIDE="eval" pip install good && pip install bad', reg)

    assert result.refusal and "bad" in result.refusal, result.refusal
    assert [r.name for r in result.cleared] == ["good"]


def test_a_chat_session_is_never_refused_and_its_registry_is_never_consulted():
    """The tool is Alan's terminal: the same command refused unattended is allowed
    for a person, and the interactive path pays nothing for the decision a human is
    present to make — not one HTTP round trip."""
    calls: list[str] = []

    class Spy(MappingRegistry):
        def lookup(self, name):
            calls.append(name)
            return super().lookup(name)

    reg = Spy({"graphy": sc.RegistryFacts(name="graphy", exists=True,
                                          first_release=NOW - dt.timedelta(days=2),
                                          release_count=1, source="test",
                                          observed_at=NOW)}, name="spy")

    chat = _check("pip install graphy", reg, session="chat-abc123")

    assert chat.refusal is None
    assert calls == [], "an attended turn must not pay for a lookup it will not use"


def test_an_unreachable_registry_never_blocks_but_never_reads_as_vetted():
    """A network timeout is the one failure mode that would block every legitimate
    install in the fleet, so the decision falls open. The half that has to hold for
    that to be honest: an allowed-unvetted name lands in `unvetted` with a note,
    never in `cleared` — otherwise an outage is a clean tree, the exact failure
    this file's scanner wrapper also refuses to make."""
    class Unreachable(MappingRegistry):
        def lookup(self, name):
            return sc.RegistryFacts(name=name, exists=False, source="unreachable")

    result = _check("pip install anythingnew", Unreachable({}, name="down"))

    assert result.refusal is None
    assert [r.name for r in result.unvetted] == ["anythingnew"]
    assert [r.name for r in result.cleared] == []
    assert "registry-unreachable" in result.notes[0], result.notes


def test_a_lookup_that_raises_is_a_note_and_an_unvetted_name_not_a_refusal():
    class Boom(MappingRegistry):
        def lookup(self, name):
            raise RuntimeError("socket said no")

    result = _check("pip install anythingnew", Boom({}, name="boom"))

    assert result.refusal is None
    assert [r.name for r in result.unvetted] == ["anythingnew"]
    assert "RuntimeError" in result.notes[0], result.notes


def test_a_vcs_direct_reference_is_unvetted_rather_than_refused_or_clean():
    """`name @ url` and `#egg=` refs name a project but carry no registry history to
    ask about, so the honest output is "I did not check this" — never a clean
    verdict, and never a denial (the 2026-08-24 thunderbird-mcp clone was legitimate
    work). #628 owns the destination."""
    result = _check("pip install mypkg @ git+https://github.com/x/y.git")

    assert result.refusal is None
    assert [r.name for r in result.unvetted] == ["mypkg"]
    assert "#628" in result.notes[0], result.notes


def test_a_npm_name_is_reported_but_not_vetted_because_the_facts_are_not_read():
    """The ecosystems #688's provenance half does not cover are surfaced, not
    silently allowed: `npm install` is outside the PyPI facts this check reads, and
    the report has to say so rather than look clean."""
    result = _check("npm install brandnew-package")

    assert result.refusal is None
    assert [r.name for r in result.unvetted] == ["brandnew-package"]
    assert "npm" in result.notes[0], result.notes


def test_the_printed_verdict_distinguishes_unvetted_from_allowed():
    reg = _registry(old=(True, "2019-01-01T00:00:00Z", 9),
                    squat=(False, None, None))
    result = _check("pip install old && npm install newjs && pip install squat", reg)

    text = format_provenance_verdicts(result)
    assert "ALLOW old" in text
    assert "UNVETTED newjs" in text
    assert "BLOCK squat" in text


def test_the_printed_verdict_keeps_a_declared_name_off_the_allowed_line():
    """A name the repo already declares is allowed by `requirements.txt`, not by a
    registry verdict. Printing it as `ALLOW` would claim a lookup cleared it, and
    printing nothing at all would read as "that was not an install command" — the
    two ways this output could mislead the person debugging a worker's refusal."""
    reg = _registry()
    result = _check("pip install httpx", reg)

    text = format_provenance_verdicts(result)

    assert "DECLARED httpx" in text, text
    assert "ALLOW httpx" not in text, "nothing was vetted, so nothing may say it was"
    assert reg.lookups == [], "and the line is backed by the file, not a request"


# ── clause 5: the matcher ───────────────────────────────────────────────────

@pytest.mark.parametrize("command", [
    "grep 'pip install foo' notes.md",
    'grep -rn "pip install bar" app/',
    "echo 'pip install evilpkg'",
    'echo "run `pip install thing` first"',
    "sed -i 's/pip install old/pip install new/' setup.sh",
    "cat install_notes.txt",
    "awk '/pip install baz/ {print}' README.md",
    "printf '%s\\n' 'pip install qux'",
    "rg 'poetry add somepkg' docs/",
    "awk '/uv add anotherpkg/ {print}' README.md",
    "printf '%s\\n' 'cargo add serde_json'",
    "grep -c 'npm install left-pad' README.md",
])
def test_a_command_that_only_mentions_an_install_verb_is_allowed(command):
    """The matcher parses the command; it does not substring-match verbs."""
    assert sc.find_install_commands(command) == [], command


@pytest.mark.parametrize("command,expected", [
    ("pip install graphy", ["graphy"]),
    ("pip3 install graphy", ["graphy"]),
    ("python -m pip install graphy", ["graphy"]),
    ("python3.11 -m pip install graphy", ["graphy"]),
    ("uv pip install graphy", ["graphy"]),
    ("uv add graphy", ["graphy"]),
    ("pipx install graphy", ["graphy"]),
    ("poetry add graphy", ["graphy"]),
    ("npm install left-pad", ["left-pad"]),
    ("npm install @scope/pkg", ["@scope/pkg"]),
    ("pnpm add left-pad", ["left-pad"]),
    ("cargo add serde_json", ["serde_json"]),
    ("pip install --user --break-system-packages youtube-transcript-api",
     ["youtube-transcript-api"]),
    ("FOO=bar sudo -n pip install graphy", ["graphy"]),
    ("env FOO=bar pip install graphy", ["graphy"]),
    ("timeout 60 pip install graphy", ["graphy"]),
    ("cd /tmp && pip install graphy", ["graphy"]),
    ("pip install graphy; pip install datapro", ["graphy", "datapro"]),
    ("pip install 'foo>=1,<2' 'bar[extra]==1.0'", ["foo", "bar"]),
    ("bash -c 'pip install graphy'", ["graphy"]),
    ("python -c \"import os; os.system('pip install graphy')\"", ["graphy"]),
    ('LLOYD_DEP_OVERRIDE="why" pip install graphy', ["graphy"]),
    ("uvx ruff@0.6.9 check .", ["ruff"]),
])
def test_every_install_spelling_reaches_the_policy(command, expected):
    """A name that never arrives can never be refused, so the parser's job is to
    deliver the same identity for every spelling of the same act."""
    found = sc.find_install_commands(command)

    assert [r.name for r in found] == expected, command
    # Ecosystem decides vettedness, and one ecosystem's names are not another's:
    # a cargo crate is not on PyPI, so `cargo add serde` is reported as unvetted
    # rather than cleared (see test_an_unsupported_ecosystem...).
    assert all(r.vetted == (r.ecosystem == "pypi") for r in found), command


def test_the_override_travels_with_its_own_request():
    found = sc.find_install_commands('LLOYD_DEP_OVERRIDE="eval" pip install graphy')

    assert found[0].override == "eval", found[0]
    assert sc.find_install_commands("pip install graphy")[0].override is None
    assert sc.find_install_commands("LLOYD_DEP_OVERRIDE= pip install graphy")[0].override == ""


@pytest.mark.parametrize("command", [
    "pip install -r requirements.txt",
    "pip install -r requirements.lock",
    "pip install --requirement requirements.txt",
])
def test_a_requirements_operand_is_not_mistaken_for_a_package(command):
    """`requirements.txt` is not a distribution. `pip3` and `uv` share pip's option
    table, so this has to hold for the aliases too."""
    found = sc.find_install_commands(command)

    assert found, "an -r operand expands to the file's own names"
    assert all("requirements" not in r.name and not r.name.endswith(".txt")
               for r in found), [r.name for r in found]


@pytest.mark.parametrize("command", [
    "pip install ./local-dir",
    "pip install ./dist/mypackage-1.0-py3-none-any.whl",
    "pip install mypackage-1.0.tar.gz",
    "pip install -e .",
    "npm install ./vendor/package",
    "pip install --upgrade pip",
])
def test_paths_local_artifacts_editables_and_the_bootstrap_package_are_not_names(command):
    """None of these is a registry project, so none may be classed as a new
    distribution — blocking `pip install -e .` would block the repo's own setup,
    which is the fastest possible way to make a guard unpopular."""
    assert sc.find_install_commands(command) == [], command


def test_names_coming_from_a_requirements_file_are_grounded_in_that_file():
    """The ground-truth rule: a candidate name is one this run actually read out of
    a real file. `requests` is in no requirements file of this repo, so a command
    naming it stays a candidate for the registry check; a name only present in prose
    never is."""
    declared = sc.read_dependency_set(sc._repo_root())
    norm = sc.normalize_dist_name
    assert "httpx" in declared and "nonexistentpkg" not in declared, sorted(
        declared)[:5]

    found = sc.find_install_commands("pip install -r requirements.txt")
    names = {norm(r.name) for r in found}
    assert names, "the file's own names must be the candidates"
    assert names <= set(declared), sorted(names - set(declared))[:5]
    assert "nonexistentpkg" not in names, (
        "a name in no dependency file is not made declared by reading one")

    assert sc.find_install_commands("grep -n 'nonexistentpkg' requirements.txt") == []


def test_the_declared_dependency_set_is_read_from_the_files_and_normalised():
    """Normalisation is what makes a declared name match a requested one: the file
    says `discord.py>=2`, the command says `discord.py`, and a comparison that
    misses the version suffix classes an installed package as new — a false block
    on the repo's own stack."""
    declared = sc.read_dependency_set(sc._repo_root())

    assert declared[sc.normalize_dist_name("discord.py")] == "requirements.txt"
    assert declared["pytest-xdist"] == "requirements.txt"
    assert "mcp" in declared
    assert not any(name.startswith("-") for name in declared), declared
    # Everything it returns must survive the matcher's own normaliser, or a
    # declared name could still be classed as new.
    assert all(sc.normalize_dist_name(n) == n for n in declared)


def test_the_documented_thresholds_are_the_ones_the_policy_uses():
    assert sc.DEFAULT_THRESHOLDS == Thresholds(min_first_release_age_days=90,
                                               min_release_count=2)


def test_evaluate_provenance_is_pure_about_the_thresholds():
    """The policy as a function of (facts, clock, thresholds), so the fixture eval
    and the dispatch path cannot drift apart on the same facts."""
    fresh = sc.RegistryFacts(name="x", exists=True,
                             first_release=NOW - dt.timedelta(days=10),
                             release_count=5, source="test", observed_at=NOW)

    verdict = evaluate_provenance(fresh, now=NOW)

    assert verdict.blocked and verdict.fact == "first-release-too-recent"
    assert not evaluate_provenance(fresh, now=NOW,
                                   thresholds=Thresholds(min_first_release_age_days=5,
                                                         min_release_count=2)).blocked, (
        "one place turns a fact into a refusal, so a threshold change moves the "
        "eval and the dispatch path together")

