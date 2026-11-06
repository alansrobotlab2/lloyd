"""The exception to the placeholder: a scope that holds a `credential:<NAME>` grant.

#3846 replaces every vaulted credential in a Bash child's environment with a
placeholder. That is only survivable if a legitimate consumer can ask for the real
value, and the shape it asks with is the one the gate already mints: a row in
`authority_grants` whose `tool_pattern` is `credential:<NAME>`, keyed to the SAME
scope string the egress gate and the pool derive (`agent_mcp.egress.scope_context`),
so one grant table says both "this scope may reach this host" and "this scope may hold
this credential".

Everything here is fixture-owned: the names are `LLOYD_TEST_*` and the vault is the
`credential_vault` fixture's `.env` under a monkeypatched `$HOME`, never this box's
own `~/lloyd/.env`. The real values a node compares against are the fixture's.

"""

import asyncio
import datetime as dt
import json
import sys

import pytest

from agent_mcp import builtin_bash as B
from agent_mcp import egress
from app.harness import credential_store as CS
from app.harness.policy import GrantStore

#: The three secret-shaped names `conftest.CREDENTIAL_VAULT_ENV` seeds. The grant
#: tests spend `ALPHA`; the other two stay ungranted and are the control that a grant
#: for one name does not restore its siblings.
ALPHA = "LLOYD_TEST_ALPHA_SECRET"
BETA = "LLOYD_TEST_BETA_KEY"
GAMMA = "LLOYD_TEST_GAMMA_TOKEN"

#: A child that prints its own environment as JSON. `printenv` would print the whole
#: environment without delimiters and cannot be parsed unambiguously; a value can
#: contain a newline. The dump is read from a file the command writes, not from
#: stdout, because the foreground result is prefixed with the harness's own framing.
_DUMP = ("import json,os,sys;"
         "open(sys.argv[1],'w').write(json.dumps(dict(os.environ)))")


def _child_env(tmp_path, monkeypatch, *, command_kwargs=None):
    """Spawn one real Bash child and return the environment it actually received."""
    _SEQ[0] += 1
    out = tmp_path / f"child-env-{_SEQ[0]}.json"
    args = {"command": f'{sys.executable} -c "{_DUMP}" {out}', "timeout": 40}
    args.update(command_kwargs or {})
    raw = asyncio.run(B._bash(args))
    if not out.is_file():                      # background mode answers at once
        for _ in range(80):
            import time
            time.sleep(0.25)
            if out.is_file():
                break
    assert out.is_file(), f"the child never wrote its dump; tool said: {raw[:400]}"
    return json.loads(out.read_text())


_SEQ = [0]


@pytest.fixture(autouse=True)
def _vault(credential_vault):
    """Every node in this file runs against the fixture vault, not this box's."""
    return credential_vault


@pytest.fixture(autouse=True)
def _non_sandboxed_branch(monkeypatch):
    """The credential must reach the child on the plain spawn branch, whatever the
    sandbox substrate does — and this file's nodes compare values, so they need one
    deterministic branch.

    Patched through `available()`, the module's own availability probe, so the run
    takes the documented fail-open path (`wrap` returns a plain shell spawn and False,
    which is byte-identical to pre-substrate behaviour) rather than a hand-written
    argv. `LLOYD_PATH_SANDBOX` is inherited by pytest whenever the suite starts from a
    shell that exports it — a worker turn, or a developer session — and the enforced
    branch gives the child bwrap's `--clearenv` environment instead of the parent's,
    where a granted credential could not appear at all. tests/
    test_bash_spawn_env_placeholders.py::test_all_spawn_sites_route_their_env_through_the_vault
    is what covers the sandboxed branch, structurally.
    """
    from agent_mcp import _path_sandbox

    monkeypatch.setattr(_path_sandbox, "available",
                        lambda: (False, "test: path sandbox substrate held back"))


@pytest.fixture()
def grant_db(tmp_path, monkeypatch):
    """A `workers.db` of this node's own, via `LLOYD_GRANT_DB`.

    `app.harness.policy.default_store()` honours that name precisely so a test can
    point the gate at a file without touching config; a node that minted a
    `credential:` grant against the live store would be writing real authority over
    real credentials from a test run.
    """
    path = tmp_path / "workers.db"
    monkeypatch.setenv("LLOYD_GRANT_DB", str(path))
    monkeypatch.setenv("LLOYD_EGRESS_DB", str(tmp_path / "egress.db"))
    return GrantStore(path)


@pytest.fixture()
def child_scope(monkeypatch):
    """The grant scope of a fake worker turn, derived and not invented.

    The scope is read back out of `egress.scope_context()` rather than hardcoded, so
    these nodes cannot drift from the mapping the pool uses: if
    `grant_scope_for_source` ever changes what a `scheduled-task` effect scope means,
    the grant a node mints moves with it, and the node stays a test of the credential
    rule instead of turning into a test of that mapping.
    """
    monkeypatch.setattr(egress, "_effect_scope",
                        lambda: "item:fixture-credential-probe:7")
    scope = egress.scope_context()["scope"]
    assert scope, ("scope_context() returned no grant scope for a worker-turn effect "
                   "scope, so a minted credential grant could not be attributed — "
                   "change the effect scope above, not this assertion")
    return scope


def _mint(grant_db, scope, name, *, tool_pattern=None, expires_at=None):
    """One live `credential:<NAME>` row, minted through the real writer.

    Through `GrantStore.mint` and not raw SQL because `mint` validates: an
    `expires_at` is mandatory and a `tool_pattern` has to survive
    `normalize_tool_name`. Passing `credential:` through that validator is itself part
    of what these nodes prove — that the prefix is a legal pattern and is not
    lower-cased or folded into a tool name on the way in.
    """
    return grant_db.mint(
        scope=scope,
        tool_pattern=tool_pattern or f"{CS.GRANT_PREFIX}{name}",
        issued_by="human:alan",
        expires_at=expires_at or (dt.datetime.now(dt.timezone.utc)
                                  + dt.timedelta(hours=1)),
        note="test: #3846 credential grant",
        minted_by="human")


# ---------------------------------------------------------------------------
# The granted direction
# ---------------------------------------------------------------------------

def test_a_granted_scope_receives_the_real_credential(grant_db, child_scope, tmp_path,
                                                      monkeypatch):
    """A live `credential:<NAME>` grant puts the real value in the child's env.

    The value a granted child sees is `credential_store.real_value(NAME)` — this
    process's own environment first, then the env file — and it is NOT the
    placeholder. `BETA` and `GAMMA` are in the same vault and stay placeholders, which
    is the half that matters most: a grant read as "this scope may hold credentials"
    rather than "this scope may hold THIS credential" would pass a test that only ever
    spent the granted name.
    """
    _mint(grant_db, child_scope, ALPHA)
    env = _child_env(tmp_path, monkeypatch)
    assert env[ALPHA] == CS.real_value(ALPHA), (
        f"a granted credential reached the child as {env[ALPHA]!r}")
    assert not CS.is_placeholder(env[ALPHA]), (
        "the child printed a placeholder for a name the scope holds a live grant for")
    for other in (BETA, GAMMA):
        assert CS.is_placeholder(env[other]), (
            f"{other} is not granted and still reached the child as {env[other]!r}")


def test_the_same_credential_is_a_placeholder_without_a_grant(grant_db, child_scope,
                                                              tmp_path, monkeypatch):
    """The counterfactual for the node above, on the same fixture and the same name.

    Without this the grant node proves nothing on its own: it could be that the value
    simply reached the child because the placeholder pass is not wired to this branch
    at all, which is the failure #3846 exists to close. Same vault, same scope, same
    command; the only difference is that no `authority_grants` row was written.
    """
    env = _child_env(tmp_path, monkeypatch)
    assert env[ALPHA] == CS.placeholder_for(ALPHA), (
        f"with no grant the child got {env[ALPHA]!r} rather than the placeholder — "
        "the grant node above would be measuring the same thing")
    assert env[ALPHA] != CS.real_value(ALPHA)


def test_an_expired_grant_returns_the_child_to_the_placeholder(grant_db, child_scope,
                                                                tmp_path, monkeypatch):
    """A spent or expired grant must not keep holding the door open.

    `GrantStore.live(scope=…)` is the only thing this node's child consults, so the
    row below is written and then stopped by its own `expires_at` — the same way a
    real one-shot grant retires. A `credential:` row that kept working past expiry
    would turn the placeholder design into a permanent grant with a stale timestamp on
    it, which is the state the whole write-once argument is meant to avoid.
    """
    _mint(grant_db, child_scope, ALPHA,
          expires_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1))
    assert grant_db.live(scope=child_scope) == [], (
        "the store itself reads an expired grant as live; the assertion below would "
        "then be testing nothing")
    env = _child_env(tmp_path, monkeypatch)
    assert env[ALPHA] == CS.placeholder_for(ALPHA), (
        f"an expired grant left {env[ALPHA]!r} in the child environment")


def test_the_grant_prefix_cannot_be_confused_with_a_tool_name(grant_db, child_scope,
                                                              tmp_path, monkeypatch):
    """Only `credential:<NAME>` releases a credential, and nothing else does.

    Three near-miss rows in one scope, none of which may release `ALPHA`:

    - `Bash:credential:<NAME>` — a compound pattern whose SUFFIX is the credential
      pattern. The credential reader takes `tool_pattern` whole, prefixed-matched
      against `credential:`, so a compound row is not a credential grant. If a later
      change reads the pattern by `endswith`, this row releases the secret and this
      node goes red.
    - `credential:*` — a wildcard that names no credential. A wildcard that released
      everything would make the registry the only thing standing between an unattended
      turn and every token on the box.
    - a plain `Bash` tool grant — what a scope holds for being allowed to run Bash at
      all, which must not be read as consent to hold a secret.

    A `credential:<OTHER>` row is also present, so the scope does have a live
    credential grant and the assertions below cannot pass because the table is empty.
    """
    _mint(grant_db, child_scope, BETA,
          tool_pattern=f"Bash:{CS.GRANT_PREFIX}{ALPHA}")
    _mint(grant_db, child_scope, BETA, tool_pattern=f"{CS.GRANT_PREFIX}*")
    _mint(grant_db, child_scope, BETA, tool_pattern="Bash")
    _mint(grant_db, child_scope, BETA)

    env = _child_env(tmp_path, monkeypatch)
    assert env[ALPHA] == CS.placeholder_for(ALPHA), (
        f"the child received {env[ALPHA]!r} for a name this scope was never granted "
        "by an exact `credential:` row")
    # And the one row that IS an exact match for BETA does release BETA, so the
    # assertion above is a statement about the prefix and not about the reader being
    # dead in this scope.
    assert env[BETA] == CS.real_value(BETA), (
        "an exact `credential:LLOYD_TEST_BETA_KEY` row failed to release its "
        "credential, so the ALPHA assertion above would pass for the wrong reason")


def test_a_scope_with_no_effect_scope_holds_no_credential_grant(grant_db, child_scope,
                                                                 tmp_path, monkeypatch):
    """An unattributed turn gets placeholders even when the SAME grant exists.

    The row is minted against `child_scope`; this node then blanks the effect scope,
    which is what a plain attended turn looks like — `scope_context()` returns an
    empty grant scope and `granted_credentials("")` is empty by contract. It is the
    direction the design is actually for: the grant is scoped to a worker, and the
    human's own shell is where the placeholder is the point.
    """
    _mint(grant_db, child_scope, ALPHA)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "")
    assert egress.scope_context()["scope"] == "", (
        "a blanked effect scope still resolved to a grant scope, so this node would "
        "be minting a grant the child can spend")
    env = _child_env(tmp_path, monkeypatch)
    assert env[ALPHA] == CS.placeholder_for(ALPHA), (
        f"an unattributed turn was handed {env[ALPHA]!r}")


def test_the_credential_prefix_is_a_valid_tool_pattern_end_to_end(grant_db,
                                                                   child_scope):
    """`credential:<NAME>` survives the grant table's own validator intact.

    Cheapest node in the file and it is load-bearing for every node above: `mint`
    runs `normalize_tool_name` on the pattern, and if that folded the name to lower
    case or rejected the colon, the rows above would be storing something the reader
    can never match — and every "the grant worked" assertion would be an accident of
    the reader being equally broken. Read back through `live()`, which is exactly the
    call `granted_credentials` makes.
    """
    _mint(grant_db, child_scope, ALPHA)
    rows = grant_db.live(scope=child_scope)
    assert [r["tool_pattern"] for r in rows] == [f"{CS.GRANT_PREFIX}{ALPHA}"], rows
    assert CS.granted_credentials(child_scope) == frozenset({ALPHA}), rows
