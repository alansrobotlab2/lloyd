"""The dispatch seam for #688: the refusal happens at the tool boundary, not in a
helper nobody calls.

`check_bash_command` is the single definition every Bash call passes through
(`agent_mcp/main.py`, `app/harness/loop.py`), and its contract is a returned
`(label, excerpt)` or None — the caller turns that into a tool denial. The real
registry is out of the question here, so the registry is patched at the module
this check imports it from; everything above that seam — the command parsing, the
session-kind decision, the override, the refusal's shape — is the shipped code.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from app.harness import safety, supply_chain

BACKGROUND = "20260924_054120_autonomy_688"
SUBAGENT = "task:abc12345"
NOW = dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc)


@pytest.fixture
def fresh_squat(monkeypatch):
    """Patch only the registry *fetch*. A name the registry has never heard of is
    refused on its own, so a registry that returns exists=False for the one name
    under test reproduces a squat exactly."""
    calls: list[str] = []

    def fake_lookup(self, name):  # noqa: ANN001
        calls.append(name)
        return supply_chain.RegistryFacts(name=name, exists=False, source="test",
                                          observed_at=NOW)

    monkeypatch.setattr(supply_chain.PypiRegistry, "lookup", fake_lookup)
    monkeypatch.delenv(supply_chain.OSV_API_URL_ENV, raising=False)
    return calls


def _verdict(command: str, session_id, **kw):
    return safety.check_bash_command(command, session_id=session_id, **kw)


def test_a_background_turn_installing_a_squat_shaped_name_is_denied(fresh_squat):
    out = _verdict("pip install graphy", BACKGROUND)

    assert out is not None, "the check must be wired into the dispatch path"
    label, excerpt = out
    assert label.startswith("install provenance:")
    assert "graphy" in label and "not published on pypi.org" in label
    assert "pip install graphy" in excerpt


def test_a_subagent_inheriting_a_background_parent_is_denied_too(fresh_squat):
    """Background-ness is inherited through `parent_of`, exactly as the service-
    control rule above it does: a `task:*` subagent is as unattended as the worker
    that spawned it."""
    out = _verdict("pip install graphy", SUBAGENT,
                   parent_of=lambda sid: BACKGROUND if sid == SUBAGENT else None)

    assert out is not None and "graphy" in out[0]


def test_a_subagent_inheriting_a_chat_parent_is_allowed(fresh_squat):
    out = _verdict("pip install graphy", SUBAGENT,
                   parent_of=lambda sid: "chat-abc123" if sid == SUBAGENT else None)

    assert out is None


def test_a_chat_session_is_never_refused(fresh_squat):
    """The refusal is scoped to unattended turns. Refusing Alan's own terminal
    would be the false-positive cost this design cannot pay — he is the one who can
    read the reason and decide."""
    assert _verdict("pip install graphy", "chat-abc123") is None
    assert fresh_squat == [], "an attended turn must not pay for a lookup"


def test_an_absent_session_id_is_treated_as_unattended(fresh_squat):
    """The two guards on this line differ on purpose. `service_control` skips when
    it cannot show the session is a background one, because a wrongly-denied restart
    breaks tooling; a supply-chain refusal is recoverable by the person in the loop
    with one env prefix, so the unknown case is checked rather than waved through
    (#1053: treating an absent id as trusted is itself the bypass)."""
    out = _verdict("pip install graphy", None)

    assert out is not None and "graphy" in out[0]


def test_a_provenance_override_clears_the_denial_at_the_boundary(fresh_squat):
    out = _verdict('LLOYD_DEP_OVERRIDE="new solver for the routing eval" '
                   'pip install graphy', BACKGROUND)

    assert out is None, out


def test_an_existing_dependency_is_never_denied(monkeypatch):
    """httpx ships in requirements.txt, so a background worker re-installing it
    cannot be blocked even with a registry that answers "unpublished" for it."""
    def fake_lookup(self, name):  # noqa: ANN001
        return supply_chain.RegistryFacts(name=name, exists=False, source="test",
                                          observed_at=NOW)

    monkeypatch.setattr(supply_chain.PypiRegistry, "lookup", fake_lookup)

    assert _verdict("pip install httpx", BACKGROUND) is None


def test_prose_about_an_install_never_reaches_the_registry(fresh_squat):
    """The matcher's false-positive half, measured at the dispatch boundary: a
    refusal here would be a check nobody trusts."""
    for command in ["grep -rn 'pip install graphy' docs/",
                    "echo 'remember to pip install graphy later'",
                    "sed -i 's/pip install old/pip install graphy/' setup.sh"]:
        assert _verdict(command, BACKGROUND) is None, command
    assert fresh_squat == [], fresh_squat


def test_the_provenance_check_runs_after_the_existing_checks_and_never_precedes_them():
    """Ordering matters for the message a worker sees: a command that is already
    denied for a privileged wrapper or an engine restart must not be relabelled as a
    provenance finding, and a provenance refusal must not shadow a protected-path
    one."""
    out = _verdict("sudo pip install httpx", BACKGROUND)

    assert out is not None and out[0] == "sudo", out
    assert "provenance" not in out[0], "the earlier check owns the answer"

    out = _verdict("systemctl restart agent-supervisord", BACKGROUND)
    assert out is not None and out[0].startswith("service control:"), out


def test_the_registry_is_never_consulted_when_no_install_verb_is_present(fresh_squat):
    assert _verdict("git push origin main", BACKGROUND) is None
    assert fresh_squat == []


def test_the_seam_survives_a_registry_that_raises(monkeypatch):
    """A crash inside a new check must not take the tool down: the guard falls open
    with a printed finding, which is what `unvetted` is for."""
    def boom(self, name):  # noqa: ANN001
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(supply_chain.PypiRegistry, "lookup", boom)

    assert _verdict("pip install graphy", BACKGROUND) is None


# ── the third registry fact, at the tool boundary (#1610 review, attempt 1) ─────
#
# The refusal that names the 90-day age fact exists in the module today and is
# pinned there (`test_supply_chain_provenance.py`, "first published 7 day"), but
# no node showed it crossing `safety.check_bash_command` — the refusal a caller
# actually sees. The rule is not duplicated on the way out: the installs loop
# calls `evaluate_provenance`, the same function the fixture eval grades with, so
# the shipped dispatch verdict and `block_rate: 1.0` in
# `eval/supply-chain/baseline.yaml` are one predicate measured twice. This node is
# what keeps them from drifting apart.


def test_a_first_release_under_the_age_threshold_is_denied_at_the_boundary(monkeypatch):
    """Published, 144 releases, first released 30 days ago: the only fact left to
    fail is age, so the printed reason can only be the age rule's.

    Measured before writing this: the reason that comes back through the boundary
    is `'brandnewpkg' was first published 30 day(s) ago (threshold: 90 days,
    first release …)`. A node that merely asserted `is not None` would stay green
    if the refusal were the never-published one, so the age wording is asserted
    and the other two facts' wording is asserted absent."""
    # Pinned against the REAL clock, not NOW: `check_bash_command` takes no clock,
    # so the dispatch path reads wall time and a fixed `young` here would print a
    # day count off by whatever has elapsed since 2026-09-24 (measured: `NOW - 30`
    # rendered as "34 day(s) ago" on 2026-09-28). The assertion is therefore that a
    # day count inside the threshold is printed, not which one.
    young = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)

    def fake_lookup(self, name):  # noqa: ANN001
        return supply_chain.RegistryFacts(
            name=name, exists=True, first_release=young, release_count=144,
            source="pypi.org", observed_at=young)

    monkeypatch.setattr(supply_chain.PypiRegistry, "lookup", fake_lookup)
    monkeypatch.delenv(supply_chain.OSV_API_URL_ENV, raising=False)
    out = _verdict("pip install brandnewpkg", BACKGROUND)

    assert out is not None, "the age fact did not reach the dispatch boundary"
    label, _excerpt = out
    assert label.startswith("install provenance:")
    printed = re.search(r"first published (\d+) day\(s\) ago", label)
    assert printed, label
    assert 25 <= int(printed.group(1)) <= 35, (
        f"printed a day count that is not this fixture's 30-day-old name: {label}")
    assert "threshold: 90 days" in label, label
    assert f"first release {young.date().isoformat()}" in label, label
    assert "not published" not in label and "release(s) on" not in label, (
        f"a different fact answered, so this is not the age rule: {label}")


def test_the_same_name_older_than_the_threshold_is_allowed_at_the_boundary(monkeypatch):
    """One date differs from the node above — 2018, not last month — same 144
    releases and the same `pypi.org` source, so it must be allowed. If both
    verdicts came back the same, the pair would be measuring something other than
    the 90-day threshold."""
    old = dt.datetime(2018, 9, 5, tzinfo=dt.timezone.utc)

    def fake_lookup(self, name):  # noqa: ANN001
        return supply_chain.RegistryFacts(
            name=name, exists=True, first_release=old, release_count=144,
            source="pypi.org", observed_at=NOW)

    monkeypatch.setattr(supply_chain.PypiRegistry, "lookup", fake_lookup)
    monkeypatch.delenv(supply_chain.OSV_API_URL_ENV, raising=False)
    assert _verdict("pip install brandnewpkg", BACKGROUND) is None
