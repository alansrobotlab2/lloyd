"""What a real Bash child's environment holds after #3846's placeholder pass.

The premise this file exists to keep false is the one triage measured on this box:
every value in `~/lloyd/.env` was sha256-identical inside a live Bash child's
environment, because `app/config.py` puts them in `os.environ` and every spawn path
inherits that environment. #3846 replaces each vaulted name with a deterministic
placeholder at the spawn site, so the value a child can read is not the secret.

The rule these nodes hold to, and the reason they are written the way they are:

- **Fixture-owned names only.** Every name here is `LLOYD_TEST_*`, from the
  `credential_vault` fixture's `.env` under a monkeypatched `$HOME`. No node reads
  this box's `~/lloyd/.env` or names one of its credentials — the promotion gate runs
  pytest with `$HOME` pointed at a round home that has no `.env` at all, and a node
  that counted "the vaulted names" off the real file measured zero there while reading
  as a pass locally. The fixture is what makes the same node measure the same three
  secrets in both places.
- **A real child, not a reimplemented predicate.** Each node spawns through
  `agent_mcp.builtin_bash._bash` — the same coroutine the Bash tool calls — and reads
  the environment back out of the child process. A node that called
  `credential_store.vaulted_env` directly would pass while the spawn site never called
  it, which is the exact shape of the residual this item names.
- **The unchanged half is asserted too.** `PATH` and `XDG_RUNTIME_DIR` reaching the
  child is what #3768's one-key-environment bug destroyed; a change that narrowed the
  environment to fix a leak would be an operator outage wearing a security label.
"""

import asyncio
import hashlib
import json
import os
import re
import sys

import pytest

from agent_mcp import builtin_bash as B
from app.harness import credential_store as CS

#: The secret-shaped names `conftest.CREDENTIAL_VAULT_ENV` seeds, and the two in the
#: same file that are configuration rather than credentials. Written out as literals
#: rather than derived so a fixture edit that silently changes the population reddens
#: the fixture node below instead of quietly narrowing every node in this file.
VAULTED = ("LLOYD_TEST_ALPHA_SECRET", "LLOYD_TEST_BETA_KEY", "LLOYD_TEST_GAMMA_TOKEN")
NOT_VAULTED = ("LLOYD_TEST_DELTA_BASE_URL", "LLOYD_TEST_EPSILON_MODEL_OPTION",
               "LLOYD_TEST_ZETA_MODE")

#: Prints the child's own environment as JSON to the file named by argv 1. Reading it
#: from a file rather than from the tool result is deliberate: the foreground result is
#: framed by the harness, and a value may contain a newline, so stdout is not parseable
#: per-entry. `sys.executable` is this session's interpreter, and `PATH` reaching the
#: child is asserted separately below rather than assumed by using a bare `python3`.
_DUMP = ("import json,os,sys;"
         "open(sys.argv[1],'w').write(json.dumps(dict(os.environ)))")

_seq = [0]


def _child_env(tmp_path, **kwargs):
    """Spawn one real Bash child and return the environment it actually received."""
    _seq[0] += 1
    out = tmp_path / f"child-env-{_seq[0]}.json"
    args = {"command": f'{sys.executable} -c "{_DUMP}" {out}', "timeout": 40}
    args.update(kwargs)
    raw = asyncio.run(B._bash(args))
    if not out.is_file():
        # Background mode answers immediately; give the detached child time to write.
        for _ in range(80):
            import time
            time.sleep(0.25)
            if out.is_file():
                break
    assert out.is_file(), f"the child never wrote its environment; tool said: {raw[:400]}"
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _vault(credential_vault):
    """Every node here runs against the fixture vault. #3846's own rule, applied to
    the tests for it: an assertion about placeholders must not depend on this box's
    secrets existing."""
    return credential_vault


@pytest.fixture(autouse=True)
def _non_sandboxed_branch(monkeypatch):
    """Every environment comparison in this file runs on the non-sandboxed branch.

    `LLOYD_PATH_SANDBOX` is inherited by pytest whenever the suite is started from a
    shell that has it — a worker turn, or a developer session, both of which exist on
    this box — and then `_bash` takes the bubblewrap branch instead, whose child gets
    bwrap's own `--setenv` environment rather than the parent's. This file's central
    assertion is that the child's environment is the parent's minus the vaulted
    values, which is a statement about ONE branch; run against the other it fails on
    `INVOCATION_ID` being absent and reads as a broken placeholder pass.

    Not a widening: the sandboxed branch is not left unchecked. It cannot be exercised
    from a test (it needs the bwrap substrate and a bench/eval scope), so
    `test_all_spawn_sites_route_their_env_through_the_vault` counts it structurally,
    and #3669's existing nodes own the bwrap call itself.
    """
    from agent_mcp import _path_sandbox

    # Patched through the module's own availability probe so the run takes its
    # DOCUMENTED fail-open path — `wrap` returns a plain shell spawn and False,
    # byte-identical to pre-substrate behaviour — rather than a hand-written argv.
    monkeypatch.setattr(_path_sandbox, "available",
                        lambda: (False, "test: path sandbox substrate held back"))


# ---------------------------------------------------------------------------
# The fixture is what these nodes think they are using
# ---------------------------------------------------------------------------

def test_the_fixture_vault_is_the_population_every_node_below_measures(credential_vault):
    """Three vaulted names, three that are not — and they are the fixture's, not the
    box's.

    This is the node whose ABSENCE is the failure this round exists to fix: the first
    attempt at #3846 had a node asserting "exactly three vaulted secrets", derived
    from `Path.home()/"lloyd"/".env"`. Locally that file is this box's real env file
    and the count was 3; in the gate's round home it does not exist, the count was 0,
    and the run went red on an instrument. A node that measures a population has to
    own the population.

    The three `NOT_VAULTED` names are in the same file and must NOT be vaulted: one is
    a URL and two are model-selection strings, all three are in
    `credential_store.NOT_A_CREDENTIAL`, and a change that vaulted them would break a
    consumer for no gain.
    """
    # The path assertion comes FIRST and is the one that makes the rest meaningful:
    # `vaulted_names()` is the union of the env file's secret-shaped names and THIS
    # PROCESS's secret-shaped env names, and the fixture puts the six names in both
    # places — so without a path check, a fixture that failed to move `$HOME` would
    # still show all three names as vaulted while `credential_store` read this box's
    # real `~/lloyd/.env`. That is a pass measuring the wrong file, which is the same
    # class of instrument failure as counting zero.
    assert CS.env_file_path() == credential_vault / "lloyd" / ".env", (
        f"credential_store is reading {CS.env_file_path()}, not the file the fixture "
        "wrote — every assertion below would be measuring this box's real vault")
    # By NAME AND VALUE, not existence: `is_file()` is true for an unrelated file at the
    # same path, and a fixture that seeded nothing leaves every node below comparing a
    # pair that exists in neither process. Values are compared against this process's own
    # environment, which the fixture exported the same names into — the conftest constant
    # is not copy-typed here, so the two cannot silently disagree.
    seeded = CS.env_file_values()
    assert set(seeded) == set(VAULTED) | set(NOT_VAULTED), (
        "the env file credential_store reads does not hold exactly the six fixture "
        f"names: {sorted(seeded)}")
    for _name, _value in seeded.items():
        assert _value and os.environ.get(_name) == _value, (
            f"{_name} is in the file but not in this process's env with the same value, "
            "so the inheritance nodes below have no pair to compare")
    vaulted = CS.vaulted_names()
    for name in VAULTED:
        assert name in vaulted, (
            f"{name} is in the fixture .env under a secret-shaped name and is not "
            "vaulted — every node below would be measuring an empty vault")
    for name in NOT_VAULTED:
        assert name not in vaulted, f"{name} is configuration, not a credential"
    assert CS.env_file_path().is_file(), (
        "the fixture's env file is not where credential_store looks for it; the "
        "placeholder pass would be reading a different file than the tests seeded")
    # And the values a node compares against are the fixture's literals, so a node
    # cannot pass by matching a real secret it happened to have in the process env.
    values = CS.env_file_values()
    assert values[VAULTED[0]].startswith("alpha-real-value-"), values[VAULTED[0]]


def test_a_child_would_see_the_real_value_without_the_pass():
    """The counterfactual: the leak this item describes is real in the fixture setup.

    `credential_store.vaulted_env` is the whole mechanism, so the mechanism has to be
    shown to be the difference between a leaked environment and a placeholder one.
    This node asserts the fixture's values ARE in this process's environment — which is
    what `app/config.py`'s `os.environ.setdefault` does on a real turn — and that the
    un-vaulted environment hands them over unchanged. Every placeholder node below is
    only meaningful against this baseline; without it, "the child got a placeholder"
    could just as well mean "the name was never in the environment to begin with".
    """
    for name in VAULTED:
        assert os.environ.get(name) == CS.env_file_values()[name], (
            f"{name} is not in this process's environment, so the spawn path would "
            "have had no value to leak and the nodes below prove nothing")
    unvaulted = dict(os.environ)
    for name in VAULTED:
        assert unvaulted[name] == CS.real_value(name)
        assert not CS.is_placeholder(unvaulted[name])


# ---------------------------------------------------------------------------
# The foreground spawn branch
# ---------------------------------------------------------------------------

def test_every_vaulted_name_holds_a_placeholder_in_a_real_child(tmp_path):
    """Each vaulted name arrives as the deterministic placeholder for that name.

    Exact value, not merely "looks like a placeholder": `placeholder_for` is
    name-derived, and a child that failed with a per-boot random token would be
    unreproducible, which is the property the design chose determinism for.
    """
    env = _child_env(tmp_path)
    for name in VAULTED:
        assert env[name] == CS.placeholder_for(name), (
            f"{name} reached a real child as {env[name]!r}, not the placeholder "
            f"{CS.placeholder_for(name)!r}")
        assert CS.name_of_placeholder(env[name]) == name, (
            f"{env[name]!r} does not name {name}, so a reader of `env` output cannot "
            "tell which credential is missing")


def test_no_value_hash_matching_a_vaulted_secret_reaches_a_child(tmp_path):
    """No value the child can read hash-matches a vaulted secret — by value, not name.

    The name-based nodes above can be satisfied by a change that renames things; this
    one is the property the item's acceptance check states: hash every value the child
    holds and compare against the hash of every vaulted value. A secret arriving under
    an unexpected name — a renamed copy, a second variable holding the same token — is
    caught here and by nothing else in this file.

    Hashed on both sides and compared as digests, so this node's own assertion message
    cannot become a place a real value gets written.
    """
    # Only the VAULTED names' values. The point is that a secret cannot reach a
    # child; the same fixture file also carries a URL, a model id and a mode flag that
    # are deliberately NOT vaulted, so hashing every value in the file would report
    # three honest pass-throughs as leaks.
    values = CS.env_file_values()
    real = {hashlib.sha256(values[n].encode()).hexdigest()
            for n in VAULTED if values.get(n)}
    assert len(real) == len(VAULTED), (
        f"expected {len(VAULTED)} distinct vaulted values to hash, got {len(real)} — "
        "two fixture secrets share a value, so a leak of one could hide the other")
    leaked = [k for k, v in _child_env(tmp_path).items()
              if hashlib.sha256(str(v).encode()).hexdigest() in real]
    assert not leaked, (
        f"a child could read a value matching a vaulted secret under: {leaked}")


def test_non_vaulted_inheritance_is_unchanged(tmp_path):
    """The environment a child gets is today's environment plus the placeholders.

    `PATH` and `XDG_RUNTIME_DIR` by name, because those two are the documented reason
    an environment is not a dict of the caller's choosing: `app/autonomy/scope_launcher
    .py` names `XDG_RUNTIME_DIR` as required by the `systemd-run --user` route, and
    #3768's real incident was a one-key environment that stripped `PATH` from every
    Bash child on the machine. Then the whole diff, both directions: the only names
    whose values differ from the parent's are vaulted names, and nothing was dropped.
    """
    env = _child_env(tmp_path)
    assert env["PATH"] == os.environ["PATH"], (
        "PATH changed between this process and its child — every CLI on the box "
        "breaks the same way #3768 broke the scope launcher")
    # The NAME must reach the child; the VALUE must survive when the parent had one.
    # A parent with no `XDG_RUNTIME_DIR` is the normal state of a process started off a
    # socket rather than a login session — the gate's environment is one — and its child
    # still has the variable, because #2578's `systemd-run --user` wrapper runs the
    # command inside a unit the user manager starts and the manager supplies it. Asserted
    # as presence plus conditional equality rather than equality: the latter would claim
    # a child can never hold a runtime dir its parent lacked, which is a property of the
    # substrate, not of this change.
    assert "XDG_RUNTIME_DIR" in env, (
        "XDG_RUNTIME_DIR did not reach the child at all, which is the name "
        "app/autonomy/scope_launcher.py says the systemd-run route requires")
    if os.environ.get("XDG_RUNTIME_DIR"):
        assert env["XDG_RUNTIME_DIR"] == os.environ["XDG_RUNTIME_DIR"], (
            "the parent had a runtime dir and the child got a different one; the "
            "placeholder pass is supposed to change the vaulted names and nothing else")
    for name in NOT_VAULTED:
        assert env[name] == os.environ[name], (
            f"{name} is not a credential and its value changed: {env[name]!r}")
    # Every value that differs must be a vaulted name, and nothing else. The vault is
    # the union of the env file's secret-shaped names AND this process's own
    # secret-shaped env names, so on a box whose shell exported `LIVEKIT_API_KEY` the
    # honest set of changed names is bigger than the fixture's three — that is the
    # mechanism working, not a defect, and comparing against the set instead of a
    # literal keeps this node true in the gate's clean environment and in a shell.
    #
    # Two names belong to the substrate rather than to this design, and are excused by
    # name: `_` is bash's own last-argument variable, which differs in every shell, and
    # `INVOCATION_ID` is set by systemd for the unit it starts, so a command wrapped by
    # #2578's resource ceiling legitimately receives the new unit's id. Both are
    # asserted below NOT to be vaulted names, so an excuse can never absorb a secret.
    SUBSTRATE_OWNED = {"_", "INVOCATION_ID"}
    # One name is DROPPED, and by documented rule: `_path_sandbox.child_env` REMOVES
    # `LLOYD_PATH_SANDBOX` on the fail-open path, because a child that could carry the
    # marker would report itself sandboxed when it is not. That removal is the module's
    # own contract, not this diff's, so it is excused by name — and the same
    # cannot-be-a-vaulted-name assertion covers it.
    SUBSTRATE_DROPS = {"LLOYD_PATH_SANDBOX"}
    vaulted = set(CS.vaulted_names())
    excused = SUBSTRATE_OWNED | SUBSTRATE_DROPS
    assert not excused & vaulted, (
        f"{sorted(excused & vaulted)} is a vaulted name and sits on this node's excuse "
        "list, which would let it reach a child unplaceholdered and green")
    changed = {k for k in env if k in os.environ and env[k] != os.environ[k]}
    assert changed - SUBSTRATE_OWNED <= vaulted, (
        "values the child sees differently from this process that are not vaulted "
        f"names: {sorted(changed - SUBSTRATE_OWNED - vaulted)}")
    assert set(VAULTED) <= changed, (
        "a vaulted fixture name reached the child unchanged — the placeholder pass did "
        "not run on this branch")
    missing = {k for k in os.environ if k not in env}
    assert missing <= SUBSTRATE_DROPS, (
        "names this process inherited that did not reach the child, beyond the marker "
        f"`_path_sandbox.child_env` removes by contract: {sorted(missing - SUBSTRATE_DROPS)}")


def test_a_child_with_no_grant_scope_still_runs_and_reads_the_placeholder(tmp_path):
    """The outage risk #3846's own risks section names, measured rather than argued.

    A CLI that reads its token from env now reads a placeholder, and the failure mode
    that would be worst is the shell itself refusing to start. It does not: the command
    runs, and the program that reads the variable reads the placeholder — a remote 401,
    not a broken pipe. What a legitimate consumer needs is a declared grant, pinned by
    tests/test_bash_spawn_credential_grant.py.
    """
    raw = asyncio.run(B._bash({"command": f'printf "%s" "${VAULTED[0]}"',
                               "timeout": 30}))
    assert CS.is_placeholder(raw.strip()), (
        "a child without a grant printed neither the placeholder nor empty — the "
        "placeholder did not reach the command that reads it")
    assert CS.name_of_placeholder(raw.strip()) == VAULTED[0], raw[:200]


# ---------------------------------------------------------------------------
# The background branch
# ---------------------------------------------------------------------------

def test_a_background_child_gets_the_placeholder_too(tmp_path):
    """`run_in_background=True` is the branch a foreground-only test would miss.

    A detached child outlives the turn that spawned it, and its output lands in a file
    the model then reads back — so it is the least auditable of the spawn sites and the
    one where a leaked token is most durable. It gets the same environment, and this
    node is the reason: a change wired to the foreground `create_subprocess_exec` only
    would leave a real token in a log file under `logs/background/`.
    """
    env = _child_env(tmp_path, run_in_background=True,
                     summary="fixture: dump my environment")
    for name in VAULTED:
        assert env[name] == CS.placeholder_for(name), (
            f"a detached child holds {env[name]!r} for {name}")
    assert env["PATH"] == os.environ["PATH"]


# ---------------------------------------------------------------------------
# The sandboxed branch, and the seam between the tool and the mechanism
# ---------------------------------------------------------------------------

def test_all_spawn_sites_route_their_env_through_the_vault():
    """Every `create_subprocess_exec` in the module gets a vaulted environment.

    The sandboxed (`bwrap`) branch cannot be exercised from a test — it needs the
    bubblewrap substrate and a bench/eval scope — so this node is the only thing
    standing between a future edit and the residual #3846 exists to close: the rule is
    that no spawn site hands a child an environment that did not pass through
    `_vault_for_child`. Counted structurally rather than by prose: read the source, and
    require each of the three `env=` arguments at the real spawn call to be a
    `_vault_for_child(...)` call, so deleting one arm is a red node and not a comment
    that stopped being true.
    """
    src = open(B.__file__, encoding="utf-8").read()
    # Three sites, two shapes: the foreground branch reassigns `env` before the spawn
    # and the other two pass it as the `env=` keyword, so one regex on `env=` would
    # count two and read as if a branch had lost its vault.
    sites = re.findall(r"(?:env=|env\s*=)\s*_vault_for_child\(", src)
    assert len(sites) == 3, (
        f"expected three spawn sites routing their env through the vault "
        f"(foreground, background, sandboxed); found {len(sites)} — the "
        "bwrap branch is the one a test cannot run, so this node is its only check")
    assert src.count("create_subprocess_exec") >= 2, (
        "the module no longer spawns through the calls this node counts; recount the "
        "spawn sites rather than editing the 3")


def test_the_placeholder_pass_is_not_living_in_a_shared_env_builder():
    """`launcher_env` and `child_env` stay credential-free, on purpose.

    Those two builders are shared: `launcher_env` is also the autonomy scope launcher's
    env, `child_env` the RPC layer's. Putting the credential rule inside one of them
    would look like broader coverage while making the rule depend on which helper a
    caller happens to use, and would leave the branch that calls the other helper
    unprotected — a guard on one of two surfaces is not a guard. The rule belongs at
    the spawn site, where a child is actually created, which is what the node above
    counts.
    """
    from agent_mcp import _path_sandbox, _resource_ceiling

    for mod in (_resource_ceiling, _path_sandbox):
        body = open(mod.__file__, encoding="utf-8").read()
        assert "credential_store" not in body and "placeholder_for" not in body, (
            f"{mod.__name__} now carries the credential rule; the spawn sites in "
            "agent_mcp/builtin_bash.py are where it belongs")
