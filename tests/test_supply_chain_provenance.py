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



# ── #1956: `python -m pip <read-only verb>` is not an install ────────────────

@pytest.mark.parametrize("command", [
    "python -m pip list", "python -m pip check", "python -m pip show httpx",
    "python -m pip freeze", "python3 -m pip list --outdated",
    "python -m pip --version", "python -m pip", "python -m pip config list",
    ".venvs/lloyd/bin/python -m pip list",
])
def test_a_read_only_module_pip_subcommand_parses_to_no_install(command):
    """The `-m pip` branch stripped an install verb when it saw one and parsed
    whatever followed otherwise, so `python -m pip list` was an install of a
    distribution named `list` — hard-denied four times on 2026-09-30 as "not
    published on pypi.org". The bare forms beside it always parsed to nothing."""
    assert sc.find_install_commands(command) == [], command
    assert sc.find_install_commands(command.split("-m ", 1)[-1]) == [], (
        "the bare spelling of the same command is the control")


@pytest.mark.parametrize("command", [
    "python -m pip install graphy", "python3.11 -m pip install graphy",
    "python -m pip download graphy", "python -m pip wheel graphy",
    "python -m pip uninstall graphy", "python -m pip --quiet install graphy",
])
def test_an_installing_module_pip_verb_still_yields_its_one_request(command):
    found = sc.find_install_commands(command)
    assert [r.name for r in found] == ["graphy"], (command, found)


# ── #3206 clause 1: the registry yields the requested VERSION's upload time, and
#    the cache codec round-trips it so a warm read gives the same fact as a cold one

#: A payload in the shape the live PyPI API answers: `releases: {version: [file, …]}`,
#: one `upload_time` per file.
_YDAY = NOW - dt.timedelta(days=1)
_NOW10 = NOW - dt.timedelta(days=10)
_SIX_MONTHS = NOW - dt.timedelta(days=182)
#: The oldest release in the payload. It is DATED AND KEPT: absence from the version map has
#: exactly one meaning ("this snapshot has no date for that version"), so a parser that
#: trimmed old releases out of the map would be minting that statement out of an age. The map
#: is capped by SIZE (`MAX_VERSION_DATES`) and never by date, for exactly that reason.
_OLDEST_PIN = "0.9.0"


def _mcp_payload() -> dict:
    """Four releases, one per state the policy can tell apart about a version.

    `0.9.0` is six months out (dated, older than the hold → a measured pass), `2.1.9` is ten
    days out (inside the hold → held), `2.2.0` was published yesterday (held), and `9.9.9` is
    listed with no `upload_time` at all, so it is ABSENT from the map — the same shape as a
    release the snapshot never saw, and answered the same way: unmeasurable, allowed, named.
    """
    return {"info": {"name": "mcp", "version": "2.2.0"},
            "releases": {
                "0.9.0": [{"upload_time": _SIX_MONTHS.strftime("%Y-%m-%dT%H:%M:%S+00:00")}],
                "2.1.9": [{"upload_time": (_NOW10).strftime("%Y-%m-%dT%H:%M:%S+00:00")}],
                "2.2.0": [{"upload_time": _YDAY.strftime("%Y-%m-%dT%H:%M:%S+00:00")}],
                "9.9.9": [{"packagetype": "sdist"}],
            }}



def test_the_parser_carries_the_upload_time_of_each_version_it_was_given():
    """#3206 clause 1, first half: the per-version timestamps survive the parse.

    #688's parser collapsed this whole map into `min(times)` plus a key count — two facts
    about the NAME — and threw the rest away. The exhibit this item exists for was a
    package whose first release was years old with 15 clean releases, so those two facts
    PASSED it; the only fact that sees the release published yesterday is the one keyed on
    the version, and it is in the payload already.

    Every DATED release is carried, however old it is. The parser filters on nothing,
    because absence from this map carries exactly one statement ("no date for that
    version") and a parser that dropped old releases would be minting that statement out of
    an age — which is the reading the review rung refused.
    """
    facts = sc.parse_pypi_payload("mcp", _mcp_payload(), observed_at=NOW)

    assert facts.version_upload_times["2.2.0"] == _YDAY, facts.version_upload_times
    assert facts.version_upload_times["2.1.9"] == _NOW10, facts.version_upload_times
    assert facts.version_upload_times[_OLDEST_PIN] == _SIX_MONTHS, facts.version_upload_times
    assert facts.latest_version == "2.2.0", facts.latest_version
    # The two facts #688 already measured are unmoved: this is an added field, not a
    # re-derivation, so a name that cleared the old check still clears it.
    assert facts.first_release == _SIX_MONTHS, facts
    assert facts.release_count == 4, "the version map must not change the release COUNT"


def test_a_release_with_no_timestamp_is_recorded_as_absent_which_means_unmeasurable():
    """#3206 clause 1/3 at the parser: a release with no date is ABSENT, and absence has ONE
    meaning.

    `0.9.0` and `2.2.0` have a date; `9.9.9` has a file with no `upload_time` at all. It is
    not in the map, because the map's rule is "a key is a release with a date". The previous
    shape of this field carried undated releases as `None` and dated-out releases as absent,
    and the cache codec then re-read the `None` as the *other* statement — "we could not date
    it" became "older than the horizon", a measured pass. One meaning closes that: the
    undated release and a release the snapshot missed are answered identically, allowed and
    named `version-age-unknown`, never `cleared`.
    """
    facts = sc.parse_pypi_payload("mcp", _mcp_payload(), observed_at=NOW)

    assert "9.9.9" not in facts.version_upload_times, facts.version_upload_times
    assert not facts.dates_version("9.9.9"), facts.version_upload_times
    verdict = evaluate_provenance(facts, now=NOW, version_spec="==9.9.9")
    assert not verdict.blocked and verdict.fact == "version-age-unknown", verdict.fact


def test_the_dated_map_is_capped_by_size_and_a_dropped_release_stays_unmeasurable(tmp_path):
    """#3206: `MAX_VERSION_DATES` bounds the CACHE ENTRY, and dropping a release may only
    ever make it unmeasurable — never older, never younger.

    The cap keeps the newest 400 dated releases, so what it can drop is a release older than
    400 other releases of the same name. For that drop to cost a hold, one distribution would
    have to publish 400 releases inside the 14-day window. What the cap is actually for is
    size: this file's own `parse_pypi_payload` docstring records a live payload holding 317
    release keys for one ordinary name, and an entry costs ~40 bytes a release, so without a
    cap the cache file would be unbounded.
    """
    cap = sc.MAX_VERSION_DATES
    # `1.0.0` is the oldest (cap+1 days), `1.{cap+1}.0` the newest (today).
    payload = {"info": {"name": "huge", "version": f"1.{cap + 1}.0"},
               "releases": {f"1.{i}.0": [{
                   "upload_time": (NOW - dt.timedelta(days=cap + 1 - i)).isoformat()}]
                   for i in range(cap + 2)}}
    facts = sc.parse_pypi_payload("huge", payload, observed_at=NOW)

    assert facts.release_count == cap + 2, "the cap must not change the release COUNT"
    assert len(facts.version_upload_times) == cap, len(facts.version_upload_times)
    assert facts.dates_version(f"1.{cap + 1}.0"), "the newest release must survive the cap"
    assert not facts.dates_version("1.0.0"), "the oldest release should be the one dropped"

    # The dropped tail is unmeasurable, and unmeasurable is ALLOWED with a note: the cap may
    # cost an install a hold it could never have applied, and nothing else.
    for spec in ("==1.0.0", "==1.1.0"):
        verdict = evaluate_provenance(facts, now=NOW, version_spec=spec)
        assert not verdict.blocked and verdict.fact == "version-age-unknown", (spec, verdict.fact)

    cache = tmp_path / "registry-cache.json"
    sc._write_cache(cache, "huge", facts)
    size = cache.stat().st_size
    assert size < 60_000, f"a capped entry still wrote {size} bytes — the cap is not bounding"


def test_the_registry_cache_round_trips_the_version_fact_so_a_warm_read_agrees(tmp_path):
    """#3206 clause 1, second half: the cache codec carries the version fact.

    `_write_cache`/`_read_cache` are a hand-written field-by-field projection, not an
    `asdict`. That is the whole trap: a field added to `RegistryFacts` and not to BOTH
    halves of the codec silently vanishes on every warm read for `CACHE_TTL_DAYS` = 7 days,
    so the hold would fire at most once a week per name — i.e. on the first install of a
    day-old release and on none after, which is the failure this clause exists to pin.

    Every version the payload mentions is priced on both readings, `==9.9.9` included. It is
    the release with no `upload_time`, and the review rung refused the commit that filtered
    it out of this loop instead of judging it: dropping a case from a comparison is exactly
    how a round-trip test stops testing the case where the codec has to say "no date".
    """
    cache = tmp_path / "registry-cache.json"
    cold = sc.parse_pypi_payload("mcp", _mcp_payload(), observed_at=NOW)
    sc._write_cache(cache, "mcp", cold)
    warm = sc._read_cache(cache, "mcp", max_age_days=sc.CACHE_TTL_DAYS, now=NOW)

    assert warm is not None, "the cache entry was not readable at all"
    assert warm.source == "cache", "a warm read must still say where its facts came from"
    assert warm.version_upload_times == cold.version_upload_times, (
        f"the version map did not survive the cache: cold={cold.version_upload_times} "
        f"warm={warm.version_upload_times}")
    assert warm.latest_version == cold.latest_version, (
        f"latest_version did not survive the cache: {cold.latest_version!r} -> "
        f"{warm.latest_version!r}; an unpinned install priced through a warm read would "
        "be judged against no release at all")
    specs = [f"=={v}" for v in sorted(_mcp_payload()["releases"])] + [None]
    assert len(specs) == 5, (
        f"the payload lists 4 releases plus the unpinned claim, so 5 claims must be priced: "
        f"{specs}")
    for spec in specs:
        cold_v = evaluate_provenance(cold, now=NOW, version_spec=spec)
        warm_v = evaluate_provenance(warm, now=NOW, version_spec=spec)
        assert cold_v.fact == warm_v.fact and cold_v.blocked == warm_v.blocked, (
            f"spec={spec!r}: cold read says {cold_v.fact} (blocked={cold_v.blocked}) and "
            f"warm read says {warm_v.fact} (blocked={warm_v.blocked}) — the second install "
            "of the same version would be decided by whether a cache file happened to exist")

def test_a_cache_blob_written_before_the_version_map_existed_fails_open(tmp_path):
    """#3206 clause 1's migration case: a pre-#3206 cache entry carries no version map.

    This is not hypothetical — every entry on this box written before today is shaped like
    this. Reading one must not become "every version is old" (silently permissive about the
    one thing the hold is for) nor "the pinned version is undated, refuse" (a stale cache
    file blocking a real install, which is #688's own four-denial false-block history).
    """
    import json as _json
    cache = tmp_path / "registry-cache.json"
    cache.write_text(_json.dumps({"mcp": {
        "exists": True, "first_release": _SIX_MONTHS.isoformat(), "release_count": 2,
        "observed_at": NOW.isoformat()}}), encoding="utf-8")
    warm = sc._read_cache(cache, "mcp", max_age_days=sc.CACHE_TTL_DAYS, now=NOW)

    assert warm is not None and warm.version_upload_times == {}, warm
    verdict = evaluate_provenance(warm, now=NOW, version_spec="==2.2.0")
    assert not verdict.blocked, verdict.reason
    assert verdict.fact == "version-age-unknown", (
        f"a pre-#3206 cache entry answered {verdict.fact!r}; the hold has no version fact "
        "here, so it must say so and let the install through")


# ── #3206 clause 2: the hold itself, and what the refusal is obliged to print

def test_a_pinned_version_younger_than_the_hold_is_blocked_under_its_own_fact():
    """#3206 clause 2: `version-too-recent` is the fact, and it is a threshold on the
    VERSION, not a re-statement of the 90-day name rule.

    The name facts pass this artifact — first release six months ago, 3 releases — so the
    only thing refusing it is the release's age. `min_release_age_days` defaults to 14,
    Rybka's own recipe shortened ("a week or two"), so that a two-week-old fix is held
    rather than a three-week-old one.
    """
    facts = sc.parse_pypi_payload("mcp", _mcp_payload(), observed_at=NOW)

    assert sc.Thresholds().min_release_age_days == 14, sc.Thresholds()
    verdict = evaluate_provenance(facts, now=NOW, version_spec="==2.2.0")

    assert verdict.blocked and verdict.fact == "version-too-recent", verdict.fact
    # The same facts at a version that clears the hold still pass, unchanged: the hold is
    # one more branch, not a stricter policy applied to everything.
    older = evaluate_provenance(facts, now=NOW, version_spec="==0.9.0")
    assert not older.blocked and older.fact == "cleared", older.fact
    # And the threshold is what moved, provably: at 14 the release is held, at 0 it is not.
    permissive = evaluate_provenance(
        facts, now=NOW, version_spec="==2.2.0",
        thresholds=sc.Thresholds(min_release_age_days=0))
    assert not permissive.blocked, (
        "min_release_age_days=0 did not disarm the hold, so the threshold is not what "
        "decides this refusal")


def test_the_refusal_names_the_distribution_the_version_the_age_and_the_release_date():
    """#3206 clause 2's print contract, checked by finding each field rather than by
    trusting a template.

    A refusal that names only the package sends the next attempt at the same version. The
    four fields are what let a reader decide in one pass: `mcp` / `2.2.0` / `1 day` /
    `2026-10-22`. It also has to say the NAME passed, or the reader hunts for a problem with
    the package that does not exist — and it must offer the vendor-the-artifact route
    before the override, because the held version is very often the CVE fix.
    """
    facts = sc.parse_pypi_payload("mcp", _mcp_payload(), observed_at=NOW)
    reason = evaluate_provenance(facts, now=NOW, version_spec="==2.2.0").reason

    _release_day = _YDAY.date().isoformat()
    for needle, what in (("mcp", "the distribution"), ("2.2.0", "the version"),
                         ("1 day", "the age in days"), (_release_day, "the release date")):
        assert needle in reason, f"the refusal does not name {what} ({needle!r}): {reason}"
    assert "14 day" in reason, f"the refusal does not state the hold: {reason}"
    assert "older and passed its own checks" in reason, (
        f"the refusal does not say the name itself cleared: {reason}")
    assert sc.OVERRIDE_ENV in reason, "the refusal does not offer the reason-carrying override"
    assert "pypi.org/simple/mcp" in reason, (
        f"the refusal does not point at the artifact's hashes for vendoring: {reason}")
    # An unpinned install is told the truth about what it would fetch, which is the
    # newest release — the shape that silently takes a day-old artifact.
    bare = evaluate_provenance(facts, now=NOW, version_spec=None)
    assert bare.blocked and "latest" in bare.reason, bare.reason


def test_only_an_exact_pin_names_a_version_and_everything_else_is_the_latest():
    """#3206 clause 2's naming rule: `==X` is a claim about X; everything else is a claim
    about the registry's latest release.

    `>=`/`~=`/`==1.*`/`@` all resolve at install time to the newest artifact that matches,
    so an unpinned or ranged install is the SHAPE THAT SILENTLY TAKES A DAY-OLD RELEASE and
    must not read as vetted. `==1.*` is a wildcard and names no artifact we could date.
    """
    facts = sc.parse_pypi_payload("mcp", _mcp_payload(), observed_at=NOW)

    assert sc.target_version(facts, "==2.1.1") == ("2.1.1", "pinned")
    assert sc.target_version(facts, "[cli]==2.1.1") == ("2.1.1", "pinned"), (
        "an extras suffix hid the exact pin, so `mcp[cli]==2.1.1` would be priced against "
        "the latest release instead of the one being installed")
    for spec in (None, "", ">=2.1.0", "<3", "==2.*", "~=2.0", "@2.2.0", "[cli]"):
        version, how = sc.target_version(facts, spec)
        assert (version, how) == ("2.2.0", "latest"), (
            f"spec={spec!r} resolved to {version!r}/{how}; a range resolves to the newest "
            "artifact, so it must be priced against the registry's latest release")
    assert sc.exact_pin("==2.1.1") == "2.1.1" and sc.exact_pin("==2.*") is None
    assert sc.exact_pin(">=2.1.0") is None and sc.exact_pin(None) is None


# ── #3206 clause 3: an age that cannot be MEASURED allows, with a note, and never
#    reads as vetted — the same shape as first-release-unknown

def test_a_pinned_version_the_registry_never_dated_is_allowed_and_says_so():
    """#3206 clause 3: the release exists, the pin names it, and it has no timestamp.

    Failing open here is not a shortcut: refusing an artifact because the registry sent no
    date for it would let a payload shape this box has never seen block every install of a
    real, six-month-old package. The honest output is ALLOW-with-a-note, and the note is the
    fact name — `cleared` would claim the age was checked.
    """
    facts = sc.parse_pypi_payload("mcp", _mcp_payload(), observed_at=NOW)
    verdict = evaluate_provenance(facts, now=NOW, version_spec="==9.9.9")

    assert not verdict.blocked, verdict.reason
    assert verdict.fact == "version-age-unknown", verdict.fact
    assert verdict.fact != "cleared", (
        "an age this check never measured was reported as cleared, which is the folding "
        "#688's finding exists to prevent")



def test_a_pinned_version_dated_older_than_the_hold_is_measured_and_cleared():
    """#3206 clause 3's other edge: a date inside the map that clears the hold is a
    MEASUREMENT, and it reads as `cleared` rather than as an unmeasured-age note.

    This is the branch that must NOT become `version-age-unknown`: collapsing it would make
    every install of an established package print "not vetted" until the note means nothing,
    which is how a check becomes theatre — and it is the branch every one of this repo's
    `requirements.lock` pins travels.
    """
    payload = {"info": {"name": "httpx", "version": "0.28.1"},
               "releases": {
                   "0.28.1": [{"upload_time": (NOW - dt.timedelta(days=3)).isoformat()}],
                   "0.27.0": [{"upload_time": (NOW - dt.timedelta(days=400)).isoformat()}],
               }}
    facts = sc.parse_pypi_payload("httpx", payload, observed_at=NOW)
    assert facts.dates_version("0.27.0"), (
        "a 400-day-old release was left out of the map: absence is reserved for a release "
        "with no date, and a pin of an old artifact must be priced on its real date")

    verdict = evaluate_provenance(facts, now=NOW, version_spec="==0.27.0")
    assert not verdict.blocked and verdict.fact == "cleared", verdict.fact
    # A payload whose files carry no timestamps at all yields an EMPTY map, and the parser
    # also has to report that it could not date the name — so the name fact fails open first
    # and is the fact that gets named. That is the right order (a name with no measurable age
    # says nothing about any release), and the version arm is pinned separately below, on
    # facts whose name IS dated so only the version can be the reason.
    undated = {"info": {"name": "httpx", "version": "0.27.0"},
               "releases": {"0.27.0": [{}], "0.26.0": [{}]}}
    empty = sc.parse_pypi_payload("httpx", undated, observed_at=NOW)
    assert empty.version_upload_times == {} and empty.release_count == 2, empty
    assert empty.first_release is None, "an undated payload must not report a first release"
    assert evaluate_provenance(empty, now=NOW, version_spec="==0.27.0").fact == (
        "first-release-unknown")

    name_dated_no_version_dates = sc.RegistryFacts(
        name="httpx", exists=True, first_release=_SIX_MONTHS, release_count=2,
        source="test", observed_at=NOW, version_upload_times={}, latest_version="0.27.0")
    older = evaluate_provenance(name_dated_no_version_dates, now=NOW, version_spec="==0.27.0")
    assert not older.blocked and older.fact == "version-age-unknown", older.fact


def test_a_cache_entry_older_than_the_threshold_cannot_date_anything_in_it(tmp_path):
    """#3206 clause 3, the cache's half: an entry older than the hold fails open by NAME.

    Two stale-cache states, answered differently, and the difference is the whole point:

    * An entry whose observation is inside the hold is a usable snapshot. A release it dates
      as 20 days old really is 20 days old — an upload time is an absolute instant and the
      cache never ages a number — so it is priced normally and `cleared` is earned.
    * An entry whose observation is OLDER than the hold threshold has no standing to date
      anything inside that window: a release published three days ago is invisible to a
      snapshot eight days old, and "absent" is not evidence that the missing release is old.
      That is `version-age-stale-cache` — allowed, named, never `cleared`. Refusing instead
      would let a stale cache file block a real install, which is #688's measured false-block
      shape (four denials, all four false), so staleness allows and says so.
    """
    cache = tmp_path / "registry-cache.json"
    stale = sc.RegistryFacts(
        name="mcp", exists=True, first_release=_SIX_MONTHS, release_count=3,
        source="pypi.org", observed_at=NOW - dt.timedelta(days=20),
        version_upload_times={"2.2.0": NOW - dt.timedelta(days=25)},
        latest_version="2.2.0")
    sc._write_cache(cache, "mcp", stale)
    warm = sc._read_cache(cache, "mcp", max_age_days=99, now=NOW)
    assert warm is not None and warm.version_upload_times, warm

    aged = evaluate_provenance(warm, now=NOW, version_spec="==2.2.0")
    assert not aged.blocked, aged.reason
    assert aged.fact == "version-age-stale-cache", (
        f"a snapshot observed {int((NOW - warm.observed_at).days)} days ago — older than the "
        f"{sc.DEFAULT_THRESHOLDS.min_release_age_days}-day hold it is being asked to price — "
        f"answered {aged.fact!r}; it may not report age it could not have seen")
    assert aged.fact != "cleared", "a stale cache entry read as vetted"

    # The same facts, observed inside the window, ARE a usable snapshot: the date is exact
    # however recent the read was, and this is what makes the fail-open above a judgement
    # about standing rather than a blanket refusal to use the cache.
    fresh = sc.RegistryFacts(
        name="mcp", exists=True, first_release=_SIX_MONTHS, release_count=3,
        source="pypi.org", observed_at=NOW - dt.timedelta(days=2),
        version_upload_times={"2.2.0": NOW - dt.timedelta(days=25)},
        latest_version="2.2.0")
    usable = evaluate_provenance(fresh, now=NOW, version_spec="==2.2.0")
    assert usable.fact == "cleared", usable.fact
    # ...and a release the entry never saw is still unknown, not old.
    missed = evaluate_provenance(fresh, now=NOW, version_spec="==2.99.0")
    assert not missed.blocked and missed.fact == "version-age-unknown", missed.fact


def test_the_new_facts_keep_unmeasured_ages_out_of_the_cleared_list():
    """#3206 clause 3 end to end: through the dispatch path, an unmeasured version age is
    UNVETTED, not `cleared` — while an inert target stays allowed.

    The distinction is #688's own finding, restated for the new facts: `cleared` is read
    downstream as "the registry vetted this", and folding a version the registry never
    dated into it converts a payload shape into a clean bill of health.
    """
    reg = MappingRegistry({
        "undated": sc.RegistryFacts(
            name="undated", exists=True, first_release=NOW - dt.timedelta(days=2_000),
            release_count=40, source="test", observed_at=NOW,
            version_upload_times={}, latest_version="1.0")}, name="test")

    result = _check("pip install undated==1.0", reg, deps={})

    assert not result.refusal, result.refusal
    assert [r.name for r in result.cleared] == [], (
        "an install whose version age could not be measured was reported as cleared")
    assert [r.name for r in result.unvetted] == ["undated"], result.unvetted
    assert any("version-age-unknown" in n for n in result.notes), result.notes
    assert "UNVETTED undated" in format_provenance_verdicts(result)

# ── #3206 clause 3, the half a cached snapshot cannot answer ────────────────────

def _cache_written_days_ago(tmp_path, name: str, days: int,
                            versions: dict[str, str | None], latest: str,
                            first_days_ago: int = 1_400, count: int = 42):
    """A registry-cache file as it sits on disk: written `days` ago, so its
    `observed_at` is `NOW - days` and every version in `versions` was dated *then*.
    """
    observed = NOW - dt.timedelta(days=days)
    cache = tmp_path / "registry-cache.json"
    sc._write_cache(cache, name, sc.RegistryFacts(
        name=name, exists=True,
        first_release=NOW - dt.timedelta(days=first_days_ago), release_count=count,
        source="pypi.org", observed_at=observed,
        version_upload_times={v: (observed - dt.timedelta(days=int(d)) if d is not None
                                  else None) for v, d in versions.items()},
        latest_version=latest))
    return cache



def test_a_release_published_after_the_cached_snapshot_re_reads_the_registry(tmp_path):
    """#3206 clause 3's real cure: a cached entry that cannot date the pin may not ANSWER
    for it, so the reader goes to the registry instead of failing the claim open.

    A cache entry three days old is inside `CACHE_TTL_DAYS` = 7, so it is served; its map
    holds the releases dated at write time and, by construction, nothing published since. A
    release that went out yesterday is absent from it for exactly the reason a release with
    no `upload_time` is absent. The first commit answered that with `version-age-stale-cache`
    on every warm read, which is honest and useless: the cache serves every repeat install of
    a name, so the guard would spend its life saying "unknown" and hold nothing.

    So the two halves are read together here: the claim the entry CAN date is served from it
    at zero cost, and the claim it cannot is re-read live and held.
    """
    cache = _cache_written_days_ago(tmp_path, "mcp", days=3,
                                    versions={f"2.{i}.0": 300 for i in range(5)},
                                    latest="2.1.4", count=42)
    warm = sc._read_cache(cache, "mcp", max_age_days=sc.CACHE_TTL_DAYS, now=NOW)

    assert warm is not None, "a 3-day-old entry should be inside the 7-day TTL"
    assert not warm.dates_version("2.99.0"), warm.version_upload_times

    # Absence never lies about age: read on its own, the entry can only say it could not
    # measure — never `cleared`, which is the veto-shaped clean the first commit was refused
    # for, and never a hold.
    verdict = evaluate_provenance(warm, now=NOW, version_spec="==2.99.0")
    assert not verdict.blocked and verdict.fact == "version-age-unknown", verdict.fact

    # The reader's rule, both directions, on the same warm entry.
    class _Reg:
        def __init__(self) -> None:
            self.fresh_calls: list[str] = []
            self.plain_calls: list[str] = []

        def lookup(self, name, *, fresh=False):
            if fresh:
                self.fresh_calls.append(name)
                return sc.RegistryFacts(
                    name=name, exists=True, first_release=NOW - dt.timedelta(days=1_400),
                    release_count=43, source="pypi.org", observed_at=NOW,
                    version_upload_times={f"2.{i}.0": NOW - dt.timedelta(days=300)
                                          for i in range(5)} | {"2.99.0": _YDAY},
                    latest_version="2.99.0")
            self.plain_calls.append(name)
            return sc._read_cache(cache, name, max_age_days=sc.CACHE_TTL_DAYS, now=NOW)

    reg = _Reg()
    held = sc._registry_facts(reg, "mcp", pin="2.99.0")
    assert held.dates_version("2.99.0"), "the live read did not reach the policy"
    assert reg.fresh_calls == ["mcp"] and reg.plain_calls == ["mcp"], (
        f"pin the entry could not date cost {reg.fresh_calls} live read(s): the release a "
        "hold exists for is precisely the one the entry cannot date")
    assert evaluate_provenance(held, now=NOW, version_spec="==2.99.0").fact == (
        "version-too-recent"), "the live facts do not hold the release either"

    # A pin the snapshot DID date is answered from the cache, with no round trip at all:
    # that is the whole content of this repo's own `requirements.lock`, and the economy of
    # the rule is part of its claim, not an afterthought.
    reg2 = _Reg()
    cheap = sc._registry_facts(reg2, "mcp", pin="2.0.0")
    assert cheap.dates_version("2.0.0") and reg2.fresh_calls == [], (
        f"a pin the cache dated went to the network anyway: {reg2.fresh_calls}")
    assert evaluate_provenance(cheap, now=NOW, version_spec="==2.0.0").fact == "cleared"


def test_the_version_claim_the_cache_cannot_answer_reads_the_registry_in_production(
        tmp_path, monkeypatch):
    """#3206 clause 3 made true on the shipped path, not only in the policy.

    The policy-level nodes prove what a fact means; they cannot prove which read supplied it.
    This node runs the real `PypiRegistry` against a cache file on disk and a fake transport,
    and asks the three questions that matter about cost and coverage:

    * a pin the cached snapshot cannot date fetches, and is HELD — a guard that only fired on
      a cold cache would fire about twice a month on this box, which #688 measured as the
      entire install traffic;
    * a pin it can date is answered from the cache with NO extra GET, because a date does not
      go stale and this repo's whole pinned set travels that path;
    * an unpinned install of a name NEW to the dependency set is priced on the registry's
      latest release, because that is the artifact pip will take, and a day-old latest is HELD
      (item step 3's ruling: an unpinned install must not read as vetted);
    * an unpinned install of a name ALREADY declared is journalled `declared` and asks nothing
      of the network — the boundary this round deliberately keeps, and the reason is printed in
      the note beside it.

    The clock is the real one here, unlike every node above: `PypiRegistry` stamps
    `observed_at` from wall time, so a fixture-dated release would be priced against a
    different now than the one it was written for.
    """
    import httpx  # noqa: F401 — the seam this node replaces

    monkeypatch.setenv("LLOYD_SUPPLY_CHAIN_CACHE_DIR", str(tmp_path))
    today = dt.datetime.now(dt.timezone.utc)
    fresh_version = "2.99.0"
    fresh_day = (today - dt.timedelta(days=1)).date().isoformat()
    calls: list[str] = []
    # What the registry knows WHEN ASKED. The snapshot the cache holds was written before
    # `2.99.0` existed, which is the only way that release can be missing from a map.
    live: dict[str, list[dict]] = {}

    # What the registry calls the latest release, mutable so one node can also ask the
    # unpinned question: `info.version` is the artifact an unpinned install resolves to.
    latest = {"v": "2.1.4"}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"info": {"version": latest["v"]},
                    "releases": {f"2.{i}.0": [{"upload_time": "2026-01-01T00:00:00"}]
                                 for i in range(5)} | dict(live)}

    def fake_get(url, **kw):
        calls.append(str(url))
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)

    reg = sc.PypiRegistry(ttl_days=sc.CACHE_TTL_DAYS,
                          cache_path=tmp_path / "registry-cache.json")
    # Warm it the way the name facts are warmed: a plain lookup, which the cache owns.
    reg.lookup("mcp")
    assert len(calls) == 1, calls
    assert not sc._read_cache(tmp_path / "registry-cache.json", "mcp",
                             max_age_days=sc.CACHE_TTL_DAYS, now=today).dates_version(
                                 fresh_version), "the warm-up snapshot must not know the pin"

    # A release list maps version → the LIST OF FILES, each with its own `upload_time`; a
    # bare string here is a malformed entry, which `release_version_times` rightly declines
    # to date — so the fixture has to look like the API, not like a summary of it.
    live[fresh_version] = [{"upload_time": fresh_day + "T00:00:00"}]
    held = sc.check_install_provenance(
        f"pip install mcp=={fresh_version}", BACKGROUND,
        registry=reg, dependency_set={"mcp": "requirements.txt"}, now=today)
    assert held.refusal is not None, (
        "a day-old pin of a declared name was allowed: the version fact was answered from a "
        "cache entry that had never seen the release")
    assert held.refusals[0][1].fact == "version-too-recent", held.refusal
    assert len(calls) == 2, f"the claim the cache could not date did not fetch: {calls}"

    # A pin the warm snapshot DOES date costs nothing: `2.1.0` is in the entry with a
    # 2026-01-01 upload time, and an upload time is an absolute instant, so the cached answer
    # is as true as a fresh one. This is the path every one of `requirements.lock`'s 180 pins
    # travels on a rebuild, so it must be free AND it must clear.
    old_pin = next(v for v in sorted(
        sc._read_cache(tmp_path / "registry-cache.json", "mcp",
                       max_age_days=sc.CACHE_TTL_DAYS, now=today).version_upload_times))
    cheap = sc.check_install_provenance(
        f"pip install mcp=={old_pin}", BACKGROUND, registry=reg,
        dependency_set={"mcp": "requirements.txt"}, now=today)
    assert len(calls) == 2, f"a pin the cache dated went to the network: {calls}"
    assert not cheap.refusal, cheap.refusal
    assert [r.name for r in cheap.declared] == ["mcp"], cheap.declared

    # An unpinned install of a DECLARED name asks nothing and holds nothing, and is journalled
    # `declared` — the outcome #688 already gives a name the box ships. The boundary is
    # deliberate: `_declared_hold` prices an EXACT pin, because that is the claim whose
    # artifact is knowable in advance, and refusing a bare `pip install <declared name>`
    # whenever any newer release exists is the class of false block #688's four denials were.
    # What this node pins is that the boundary is the SHIPPED behaviour and not an accident of
    # the registry being unreachable: `calls` does not move, so no failed lookup is hiding it.
    unpinned = sc.check_install_provenance(
        "pip install mcp", BACKGROUND, registry=reg,
        dependency_set={"mcp": "requirements.txt"}, now=today)
    assert len(calls) == 2, f"an unpinned declared install cost a registry call: {calls}"
    assert not unpinned.refusal, unpinned.refusal
    assert [r.name for r in unpinned.declared] == ["mcp"], unpinned.declared

    # A name NEW to the dependency set is a different claim: the artifact is the registry's
    # latest, so a day-old latest is held even though nothing was pinned. `latest["v"]` is now
    # the fresh release, which is exactly the state a name is in on its first day.
    latest["v"] = fresh_version
    fresh_name = "brand-new-utility"
    new_turn = sc.check_install_provenance(
        f"pip install {fresh_name}", BACKGROUND, registry=reg,
        dependency_set={"mcp": "requirements.txt"}, now=today)
    assert len(calls) == 3, f"the new name did not ask the registry: {calls}"
    assert new_turn.refusal is not None, (
        "an unpinned install of a new name took a day-old latest release without a word")
    assert new_turn.refusals[0][1].fact == "version-too-recent", new_turn.refusal
    assert fresh_version in new_turn.refusal, (
        f"the refusal did not name the version it held: {new_turn.refusal}")
