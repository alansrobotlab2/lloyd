"""Bash is tiered by the shape of its command string, not by its name (#740).

Clauses 1, 2 and 3 of item #740. The decision these tests exist to keep is one
sentence: `supervisorctl restart agent-tts` from an unattended scope is a
durable-external action exactly like `email_send`, and `ls -la` is not.

The item's own premise, which these tests must not quietly widen, is what makes
the shape split hard. Bash cannot be gated as a NAME — the same name covers `ls`
and `dd`, so tiering it wholesale "would deny every worker" — so the tier has to
arrive per call, from the command string. And the second half of the item's
triage is that `safety.py`'s existing catastrophic table cannot answer that
question either: with no pattern for `supervisorctl`, plain `git push`, or an
external HTTP write, reusing `_HARD_DENY_PATTERNS` verbatim resolves all three of
the item's own examples to tier 1 and leaves them ungated. Hence a second class
beside it (`_DURABLE_EXTERNAL_PATTERNS`, hard-deny's non-denying neighbour) and
one function, `bash_command_tier`, that both grant gates read through.

Clause 3 is the load-bearing one for maintenance, so it is graded structurally
and not just by example: the shape table's location and the funnel through it
are asserted, because a second copy of this reasoning inside `policy.py` would
pass every tier-number test in this file while leaving the drift hazard intact.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from app.harness import policy
from app.harness import safety
from app.harness.policy import tool_tier
from app.harness.safety import bash_command_tier, match_durable_external

REPO = Path(__file__).resolve().parents[1]
SAFETY_SRC = (REPO / "app" / "harness" / "safety.py").read_text()
POLICY_SRC = (REPO / "app" / "harness" / "policy.py").read_text()

#: Tier-2 shapes named by the item, plus the forms this box actually uses.
DURABLE_EXTERNAL = [
    "supervisorctl restart agent-tts",
    "git push origin main",
    "curl -X POST https://api.vendor/v1/deploy",
    'python -m scripts.automod.round restart --only lloyd-backend --reason "x"',
    # The sanctioned supervisorctl invocation carries `-c <conf>` before the
    # verb, so a matcher anchored on the literal string `supervisorctl restart`
    # misses the form the triage said this box runs.
    "supervisorctl -c ~/lloyd/agent-services/supervisor/supervisord.conf "
    "restart lloyd-backend",
    # `git -C <path> push` is the form in CLAUDE.md and in every worktree round;
    # `cd … && git push` is the form a shell one-liner reaches for.
    "git -C ~/lloyd push origin automod/SM_20260924_000000",
    "cd /tmp && git push origin HEAD",
    # An HTTP write is durable-external only because of where it goes: a body to
    # a host that is not this machine.
    "curl -s -H 'Content-Type: application/json' -d @/tmp/b.json "
    "https://hooks.slack.com/services/T00/B00/XYZ",
    "systemctl restart agent-supervisord.service",
    "supervisorctl stop agent-tts && supervisorctl start agent-tts",
    # The round's own restart, spelled with a venv python and with the recovery
    # verb, is the same act.
    ".venvs/lloyd/bin/python -m scripts.automod.round restart "
    "--only agent-llm-primary",
    "python -m scripts.automod.round recover --reason 'rolled back'",
    "scripts/automod/round.py restart --only lloyd-mcp",
    # A durable shape does not have to be the whole command: `&&`/`|` segments are
    # matched individually, so leading with a read-only verb hides nothing.
    "supervisorctl status && git push origin main",
]

#: Tier 1: the ordinary traffic the item says must not gain friction, and the
#: read-only neighbours of every tier-2 shape above. One near-miss pair per
#: shape is the point — a tier-2 matcher that cannot tell `restart` from `status`
#: is a matcher that will be switched off.
ORDINARY = [
    "ls -la",
    "cat x.py",
    "grep -rn foo .",
    "supervisorctl -c ~/lloyd/agent-services/supervisor/supervisord.conf status",
    "supervisorctl status lloyd-backend",
    "systemctl is-active agent-supervisord",
    "git status --porcelain",
    "git log --format='%h %s' -3",
    "git add -A && git commit -m 'x'",
    "git grep push",
    "python -m scripts.automod.round status",
    "sed -n '1,20p' scripts/automod/round.py",
    "tar czf /tmp/x.tar.gz logs",
    "docker ps",
    # Loopback HTTP is the fleet's own health probe and its own control plane:
    # the triage measured every `curl -X` hit in 800 transcripts as loopback, so
    # an unscoped HTTP rule would gate the corpus's entire observed traffic while
    # gating nothing durable.
    "curl -s -m 20 -X POST http://127.0.0.1:8096/v1/chat/completions "
    "-d @/tmp/b.json",
    "curl -s -X POST localhost:8080/api/autonomy/run -d '{}'",
    "curl -s http://localhost:8080/health",
    "curl -s https://html.duckduck.com/html/?q=sqlite+wal+busy",
    # Prose that merely NAMES a durable shape is ordinary traffic: several of the
    # `supervisorctl`/`git push` matches in the triage's own transcript scan were
    # grep patterns, not requests.
    "grep -rn 'git push origin main' tests/",
    "grep -c 'supervisorctl restart' app/harness/safety.py",
    "echo 'do not run supervisorctl restart from a worker'",
    # A base URL held in a variable names no host, and guessing one from a shell
    # variable would gate the fleet's own control-plane pokes on a guess.
    "curl -s -X POST $API_BASE/api/autonomy/run -d @/tmp/body.json",
    "curl -s -m 5 -X POST 127.0.0.1:8500/mcp -d @/tmp/c.json",
    # A test fixture string is not a request either.
    'python3 -c "from app.harness.policy import tool_tier; '
    "print(tool_tier('Bash'))\"",
]


# ── Clause 1: the tier resolves per command string ─────────────────────────


@pytest.mark.parametrize("command", DURABLE_EXTERNAL)
def test_a_durable_external_command_tiers_2(command):
    assert tool_tier("Bash", {"command": command}) == 2, match_durable_external(command)


@pytest.mark.parametrize("command", ORDINARY)
def test_an_ordinary_command_stays_tier_1(command):
    assert tool_tier("Bash", {"command": command}) == 1, match_durable_external(command)


def test_the_clause_1_examples_answer_as_the_item_states():
    """The item's clause 1 is a list of six literals, so it is asserted as
    written rather than only through the tables above — a reviewer should be able
    to read the clause and this node side by side."""
    for command in ("supervisorctl restart agent-tts", "git push origin main",
                    "curl -X POST https://api.vendor/v1/deploy"):
        assert tool_tier("Bash", {"command": command}) == 2, command
    for command in ("ls -la", "cat x.py", "grep -rn foo ."):
        assert tool_tier("Bash", {"command": command}) == 1, command


def test_the_item_s_own_restart_form_tiers_2():
    """Clause 2, verbatim. This is the form the box uses — `round restart` shells
    out to supervisorctl itself (`scripts/automod/round.py`), so a tier keyed on
    the literal `supervisorctl restart` would miss the sanctioned path and gate
    only the forbidden one."""
    assert tool_tier("Bash", {"command": 'python -m scripts.automod.round '
                                         'restart --only lloyd-backend --reason "x"'}) == 2


def test_bash_without_a_command_string_is_tier_1():
    """The name-level answer, which two callers depend on and neither may lose.

    The side-effect traffic census enumerates tools by NAME and prints the tier
    for a name (`tests/test_side_effect_traffic_census.py` asserts `Bash` is a
    tier-1 row), and the effect ledger records a tool's name before its arguments
    are known. A `tool_tier("Bash")` that guessed tier 2 would print every
    historical Bash call as gated and deny every ledger write the row never
    asked about. Gating a CALL means passing the arguments; that is
    `effective_tier`'s contract, asserted below.
    """
    assert tool_tier("Bash") == 1
    assert tool_tier("Bash", {}) == 1
    assert tool_tier("Bash", {"command": 7}) == 1      # not a string, no shape
    assert tool_tier("Bash", "not a dict") == 1


def test_a_command_string_may_be_handed_over_bare():
    """`tool_tier('Bash', 'git push origin main')` is the tier of that command,
    so a caller holding only the string does not have to rebuild the tool_input
    envelope to ask the question."""
    assert tool_tier("Bash", "git push origin main") == 2
    assert tool_tier("Bash", "ls -la") == 1


def test_effective_tier_and_the_grant_gate_read_one_number_for_one_call():
    """`check_grants` (via `effective_tier`) and the outbound-content gate are two
    hooks on the same registry; the hook short-circuit in `install_policy_hook`
    answered tier 1 by NAME, and that short-circuit — not the ladder — was the
    half that left a service restart ungated. Both gates must therefore take the
    number from `effective_tier`, which is asserted here at the seam rather than
    inferred from the two call sites agreeing by luck."""
    durable = {"command": "supervisorctl restart agent-tts"}
    ordinary = {"command": "ls -la"}
    assert policy.effective_tier("Bash", durable) == 2
    assert policy.effective_tier("Bash", ordinary) == 1
    # And the tier-1 demotion that belongs to `autonomy_write_task` still cannot
    # be reached by the Bash branch: a name that is not Bash ignores the input.
    assert policy.effective_tier("email_send", durable) == 2


# ── Clause 3: the shape decision belongs to safety.py ──────────────────────


def test_the_shape_table_lives_in_safety_beside_the_hard_denies():
    """Clause 3's structural half: ONE durable-external class, declared in
    `app/harness/safety.py`, beside `_HARD_DENY_PATTERNS`.

    `beside` is checked as ordering because the two tables answer different
    questions about the same string — one denies, one asks for a grant — and a
    reader who finds the hard-deny table first must find the other one in the
    same screen, or the next durable shape gets added to only one of them.
    """
    assert hasattr(safety, "_DURABLE_EXTERNAL_PATTERNS"), "shape table in safety.py"
    assert safety._DURABLE_EXTERNAL_PATTERNS, "the table must not be empty"
    hard = SAFETY_SRC.index("_HARD_DENY_PATTERNS: list")
    durable = SAFETY_SRC.index("_DURABLE_EXTERNAL_PATTERNS: list")
    assert durable > hard, "declared after the hard-deny table, in the same module"
    for pattern, label in safety._DURABLE_EXTERNAL_PATTERNS:
        assert hasattr(pattern, "match"), f"{label!r}: table holds compiled patterns"
        assert label.strip(), "every shape is labelled for the deny reason"


def test_the_ladder_does_not_pattern_match_commands():
    """Clause 3's structural half, in the one form that cannot be satisfied by a
    duplicate matcher: the source that decides a Bash tier names no command.

    `app/harness/policy.py`'s Bash branch is two lines — take the command string,
    ask `safety.bash_command_tier` — so the check is that the ladder contains no
    compiled pattern of its own beyond the ones it has always needed (the grant
    id pattern and the argument-predicate pattern, both about GRANTS, neither
    about a command). A third `re.compile` in that file whose pattern mentions a
    shell verb is the drift clause 3 forbids: two tables deciding independently
    whether a push is durable.
    """
    patterns = safety.re.findall(r"re\.compile\(([^)]{0,200})", POLICY_SRC)
    offenders = [p for p in patterns
                 if safety.re.search(r"(bash|git|curl|supervisor|systemctl|push|"
                                     r"scripts|python|push|rm\b|shell)", p,
                                     safety.re.IGNORECASE)]
    assert offenders == [], f"policy.py pattern-matches a command: {offenders}"
    assert "bash_command_tier(" in POLICY_SRC, "the ladder must call safety's function"


def test_one_function_maps_a_command_to_a_tier_and_the_ladder_uses_it():
    """The reachability half of clause 3: one function, and the ladder's answer
    IS its answer. The import inside `tool_tier` is deferred (safety.py imports
    outbound_content, which imports policy — a top-level import would close that
    cycle at boot), so patching the module attribute is what an in-process caller
    sees."""
    calls = []

    def fake(command):
        calls.append(command)
        return 3

    monkeypatched = "bash_command_tier"
    real = safety.bash_command_tier
    setattr(safety, monkeypatched, fake)
    try:
        assert tool_tier("Bash", {"command": "anything at all"}) == 3
    finally:
        setattr(safety, monkeypatched, real)
    assert calls == ["anything at all"], (
        "tool_tier must hand the command string to safety.bash_command_tier and "
        "return its answer, not compute one of its own")
    # The funnel is one function deep: policy re-derives nothing from the tier.
    assert "def bash_command_tier" in SAFETY_SRC


def test_adding_a_shape_to_the_safety_table_moves_the_tier_with_no_policy_change():
    """Clause 3 stated as the change it must permit: declare a shape in
    `safety.py`, and `tool_tier` and `check_grants` both move, without either
    being touched.

    Done by appending to the live table rather than by editing the file, which is
    the same mutation the loader performs at import — so this passes only if the
    tier is READ from that table at call time. A ladder that had copied the
    patterns, or cached the table, would answer tier 1 here.
    """
    shape = (safety.re.compile(r"\brsync\b[^\n]*\b[\w.-]+@[\w.-]+:"),
             "rsync to a remote host")
    command = "rsync -az ./build deploy@host.example.com:/srv/app/"
    assert tool_tier("Bash", {"command": command}) == 1, "the shape is new"
    safety._DURABLE_EXTERNAL_PATTERNS.append(shape)
    try:
        assert tool_tier("Bash", {"command": command}) == 2
        assert match_durable_external(command) == "rsync to a remote host"
        assert tool_tier("Bash", {"command": "rsync -az ./build /mnt/backup/"}) == 1
    finally:
        safety._DURABLE_EXTERNAL_PATTERNS.remove(shape)
    assert tool_tier("Bash", {"command": command}) == 1, "the table is restored"


def test_a_durable_shape_is_not_a_hard_deny():
    """The two tables must not collapse into one. `supervisorctl restart` goes
    through `round restart`, which is the sanctioned route, so denying it
    outright would forbid the only permitted way to restart a service — and the
    item's own contract is tier 2 (a grant is sufficient), not "never".

    `service_control` is the module that refuses it to a background session, on a
    different axis; this asserts only that the durable class is not the deny
    class.
    """
    for command in ("supervisorctl restart agent-tts", "git push origin main",
                    "curl -X POST https://api.vendor/v1/deploy"):
        assert safety.check_bash_command(command) is None, command
        assert bash_command_tier(command) == 2, command


def test_the_hard_deny_axis_is_unchanged_by_the_tiering():
    """The catastrophic table keeps denying, and denies the same shapes it denied
    before this change — a tier is not a substitute for a hard deny, and
    `git push --force` to main was hard-denied on 2026-09-09 and must be
    hard-denied now.

    Checked against the pre-change labels the aggregator refuses on, so this node
    fails if the durable table is quietly loaded as a deny set too.
    """
    for command, label in [
        ("sudo reboot", "sudo"),
        ("git push --force origin main", "git push --force to main/master/release"),
        ("rm -rf / --no-preserve-rooter", "rm -rf on root/home/system path"),
        ("dd if=/dev/zero of=/dev/sda", "dd of=/dev/* (raw disk write)"),
    ]:
        hit = safety.check_bash_command(command)
        assert hit is not None and hit[0] == label, command
    assert bash_command_tier("git push --force origin main") == 2


# ── Corpus cost: the shape set must not gate the fleet's ordinary traffic ───


def test_the_shape_set_matches_nothing_in_the_census_of_real_commands():
    """The exposure this gate buys is latent, and so is its cost — this node is
    what stops the cost turning a real duty into a denied one.

    `scripts/side_effect_traffic_census.py`'s own transcript reader is pointed at
    the live corpus and every recorded Bash command is run through the tiering.
    Measured on the box on 2026-09-24: 14,887 Bash commands across 634
    transcripts, of which the shapes above match a small number. The assertion is
    deliberately about the read-only neighbours and the ratio, not an absolute
    zero, because a corpus that grows real `git push`es is a corpus telling the
    truth — what must NOT happen is the tier-1 neighbours of those shapes
    (status, log, grep, loopback POST) being swept in, which is the failure mode
    that gets a gate switched off.
    """
    from scripts.side_effect_traffic_census import DEFAULT_SESSIONS_DIR as CORPUS  # production root, not the caller's tree

    if not CORPUS.is_dir():                      # a bench machine with no corpus
        pytest.skip(f"no transcript corpus at {CORPUS}")
    import json

    commands = []
    for path in sorted(CORPUS.glob("*.json")):
        try:
            doc = json.loads(path.read_text(errors="replace"))
        except Exception:
            continue
        for msg in (doc.get("messages") or []):
            for call in (msg.get("tool_calls") or []):
                fn = (call.get("function") or {})
                if fn.get("name") != "Bash":
                    continue
                try:
                    cmd = json.loads(fn.get("arguments") or "{}").get("command")
                except Exception:
                    continue
                if isinstance(cmd, str):
                    commands.append(cmd)
    assert len(commands) > 1000, f"corpus too thin to say anything: {len(commands)}"
    gated = [c for c in commands if bash_command_tier(c) == 2]
    assert len(gated) * 20 <= len(commands), (
        f"{len(gated)} of {len(commands)} recorded Bash commands would need a "
        "grant; a gate that fires on the fleet's normal traffic gets switched "
        "off, and the shape table is what changed")


def test_the_two_documented_limits_are_limits_and_not_omissions():
    """Where the shape table stops, stated as behaviour so a reader meets the
    boundary here rather than discovering it after a denied deploy.

    Both come from matching the command's own segments after quoted data is
    removed, which is the choice that keeps prose (`grep -rn 'git push …'`) out of
    the gate — the same choice costs the two shapes below:

    - `bash -c '<command>'`: the inner command arrives as a quoted argument, so it
      is scrubbed as data. A nested interpreter is also `service_control`'s and the
      L0 text gate's territory, not the shape table's.
    - a wrapper (`timeout 120 git push …`): the leading token is the wrapper, not
      the durable verb, and `_DURABLE_LEAD` allows only `VAR=value` prefixes —
      because a matcher that searched anywhere in the string would match the
      prose case above.

    Pinned so that fixing either one is a deliberate edit to this node, and a
    matcher that silently widened would be caught by the prose cases in ORDINARY.
    """
    assert tool_tier("Bash", {"command": "bash -c 'git push origin main'"}) == 1
    assert tool_tier("Bash", {"command": "timeout 120 git push origin main"}) == 1


def test_bash_tiering_is_reachable_from_a_fresh_interpreter():
    """The deferred import is the only thing standing between this change and an
    import cycle: `app.harness.__init__` imports `safety`, `safety` imports
    `outbound_content`, `outbound_content` imports `policy`, and a top-level
    `policy -> safety` import closes that loop. The in-process suite cannot see
    that failure — by the time a test runs, everything is already loaded — so it
    is reproduced the way production does it: a fresh interpreter importing the
    package root first.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import app.harness as h; from app.harness.policy import tool_tier; "
         "print(tool_tier('Bash', {'command': 'supervisorctl restart agent-tts'}))"],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("2"), out.stdout + out.stderr


def test_the_tier_function_is_pure_and_reads_no_state():
    """`bash_command_tier` is consulted inside the grant gate, which the suite
    proves is pure stdlib; a shape matcher that reached a model, a file or the
    clock would break that property from underneath it.
    """
    src = inspect.getsource(safety.bash_command_tier)
    for forbidden in ("open(", "requests", "http", "time.", "datetime",
                      "random", "import ml", "torch"):
        assert forbidden not in src, f"bash_command_tier must not use {forbidden}"
    assert inspect.isfunction(safety.bash_command_tier)
    assert not inspect.iscoroutinefunction(safety.bash_command_tier)
    # Deterministic on the same input, because two gates ask the same question of
    # one call and a disagreement between them is the bug #740 is about.
    assert (bash_command_tier("git push origin main")
            == bash_command_tier("git push origin main"))
