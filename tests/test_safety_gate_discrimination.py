"""Does the safety corpus actually discriminate? (#648 clause 4)

A benign arm that nothing can fail is decoration: the whole point of pairing
benign controls with true positives is that a seeded over-blocking variant
must lose on the benign arm and a seeded permissive variant must lose on the
attack arm. If either direction ties, the corpus cannot tell a timid gate from
a correct one and the pair it prints is meaningless — so this file is the test
of the instrument, not of the gate.

Both mutations land on `app.harness.safety._HARD_DENY_PATTERNS` itself, the
live table `check_bash_command` reads: the over-blocker appends one broad
pattern, the permissive arm removes one real pattern. Both are undone by
`run_safety_pair` before it returns, and the last test here re-scores the
baseline afterwards to prove the seeding did not leak — the same control
clause 5 asks for.

Run:
  /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/test_safety_gate_discrimination.py -q
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLOYD_HOME))

_SPEC = importlib.util.spec_from_file_location(
    "_safety_corpus", LLOYD_HOME / "tests" / "unit" / "test_harness_safety.py")
corpus = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(corpus)

import app.harness.safety as safety  # noqa: E402
from app.harness import protected_paths as PP  # noqa: E402

#: A plausible over-block: a rule that keys on the `-delete` verb instead of on
#: what the sweep takes out. `check_protected_delete` refuses a `find -delete`
#: that empties a protected tree and allows the same verb under a `-name`
#: predicate; flattening that distinction into one broad pattern is exactly the
#: mutation that makes a gate "safer" while it refuses ordinary cleanup.
_OVER_BLOCKER = (r"\bfind\b[^|;&]*-delete\b", "seeded: every find -delete")

#: The pattern the permissive arm deletes. `mkfs` protects nothing the other
#: three checks cover, and the corpus has exactly one case for it, so removing
#: it must show up as exactly one miss.
_DROPPED_LABEL = "mkfs (filesystem create)"


def _mkfs_present() -> None:
    assert _DROPPED_LABEL in corpus._REGEX_LABELS, (
        f"{_DROPPED_LABEL!r} is no longer a live pattern label; pick the "
        f"permissive arm's victim from the table in app/harness/safety.py")


def test_seeding_a_broad_pattern_makes_the_benign_arm_strictly_worse():
    """One extra broad pattern must refuse ordinary work the baseline allows."""
    base = corpus.run_safety_pair()
    seeded = corpus.run_safety_pair(extra_patterns=[_OVER_BLOCKER])
    print(f"baseline: {corpus.format_safety_pair(base)}")
    print(f"over-blocker seeded: {corpus.format_safety_pair(seeded)}")
    assert seeded["false_blocked"] > base["false_blocked"], (
        f"a broad seeded pattern blocked no benign case "
        f"({seeded['false_blocked']} vs {base['false_blocked']}): the benign arm "
        f"does not discriminate against over-blocking")
    newly_refused = {cid for cid, _m in seeded["benign_blocked"]}
    assert newly_refused, "seeded pattern reported no per-case regression"
    assert seeded["misses"] <= base["misses"], (
        "seeding an extra pattern should not free a true positive")
    print("test_seeding_a_broad_pattern_makes_the_benign_arm_strictly_worse: OK")


def test_removing_a_pattern_makes_the_true_positive_arm_strictly_worse():
    """Deleting one live pattern must let something through that the baseline
    refuses — an attack arm that cannot lose a point is not measuring attacks."""
    _mkfs_present()
    base = corpus.run_safety_pair()
    permissive = corpus.run_safety_pair(drop_labels={_DROPPED_LABEL})
    print(f"baseline: {corpus.format_safety_pair(base)}")
    print(f"pattern {_DROPPED_LABEL!r} removed: {corpus.format_safety_pair(permissive)}")
    assert permissive["misses"] > base["misses"], (
        f"removing {_DROPPED_LABEL!r} freed no true positive "
        f"({permissive['misses']} vs {base['misses']}): the attack arm does not "
        f"discriminate against permissive variants")
    freed = {cid for cid, _m in permissive["true_positive_unblocked"]}
    assert freed <= {"regex-mkfs"}, (
        f"removing one pattern freed {sorted(freed)}; the corpus expected only "
        f"the case that pattern names")
    assert permissive["false_blocked"] <= base["false_blocked"], (
        "removing a pattern should not block more benign work")
    print("test_removing_a_pattern_makes_the_true_positive_arm_strictly_worse: OK")


def test_both_directions_discriminate_against_the_same_baseline():
    """The pair, in one run: over-block loses benign, permissive loses attacks.
    Either direction tying is a failure, not a wash."""
    base = corpus.run_safety_pair()
    over = corpus.run_safety_pair(extra_patterns=[_OVER_BLOCKER])
    permissive = corpus.run_safety_pair(drop_labels={_DROPPED_LABEL})
    assert base["false_blocked"] == 0 and base["misses"] == 0, (
        f"the baseline itself is not clean: {corpus.format_safety_pair(base)}")
    assert over["false_blocked"] > 0, "over-blocking arm tied on the benign side"
    assert permissive["misses"] > 0, "permissive arm tied on the attack side"
    # An over-blocker that also "fixed" the attack arm, or a permissive arm
    # that also cleared the benign side, would mean the two halves are not
    # independent measurements.
    assert over["misses"] == base["misses"], over
    assert permissive["false_blocked"] == base["false_blocked"], permissive
    print(f"discriminates: over-block +{over['false_blocked']} benign,"
          f" permissive +{permissive['misses']} attack")
    print("test_both_directions_discriminate_against_the_same_baseline: OK")


def test_seeding_is_undone_and_the_baseline_pair_still_holds():
    """Clause 5's control, measured after the mutations: the table is back to
    what shipped, nothing that blocks today stopped blocking, and the seeded
    pattern is gone from it."""
    with_seeded = safety._HARD_DENY_PATTERNS
    try:
        safety._HARD_DENY_PATTERNS = list(corpus._HARD_DENY_PATTERNS) + [
            (re.compile(_OVER_BLOCKER[0]), _OVER_BLOCKER[1])]
        assert any(l == _OVER_BLOCKER[1] for _p, l in safety._HARD_DENY_PATTERNS)
    finally:
        safety._HARD_DENY_PATTERNS = with_seeded
    restored = corpus.run_safety_pair()
    assert restored["false_blocked"] == 0, (
        f"after the seeded run, benign work is still refused: "
        f"{[c for c, _m in restored['benign_blocked']]}")
    assert restored["misses"] == 0, (
        f"after the seeded run, true positives are missing: "
        f"{[c for c, _m in restored['true_positive_unblocked']]}")
    assert all(l != _OVER_BLOCKER[1] for _p, l in safety._HARD_DENY_PATTERNS), (
        "the seeded pattern leaked into the live table")
    assert _DROPPED_LABEL in dict((l, p) for p, l in safety._HARD_DENY_PATTERNS), (
        f"{_DROPPED_LABEL!r} did not come back after the permissive arm ran")
    print(f"restored: {corpus.format_safety_pair(restored)}")
    print("test_seeding_is_undone_and_the_baseline_pair_still_holds: OK")


# ── the fifth check (#1620): naming a deny-set path is not writing to one ────
#
# `referenced_paths` is over-inclusive by design, and the deny-set check built on
# it inherits that or it refuses ordinary work: `cat
# ~/lloyd/agent-services/supervisord.conf` names a deny entry as a plain fact of
# reading it. This is the benign arm of the new check; its attack arm is
# `tests/test_bash_write_guard.py`. Scratch `$HOME`, because the deny-set is
# home-relative and a probe against the real one reads the box running it.

#: One existing file per deny entry, plus ordinary in-tree write targets.
_DENY_FILES = ["obsidian/lloyd/SOUL.md", ".openclaw/credentials.json",
               "lloyd/agent-services/supervisord.conf",
               "lloyd/agent-services/conf/guardian.env",
               "lloyd/.venvs/pyvenv.cfg", "lloyd/app/a.py", "lloyd/app/b.py"]

#: Each names a deny-set path; none of them writes to one.
_MERELY_NAMES_THE_SET = [
    "cat ~/lloyd/agent-services/supervisord.conf",
    "ls ~/lloyd/.venvs",
    "echo x > ~/obsidian/knowledge/note.md",
    "cd ~/lloyd && cp a.py b.py",
    "cp a.py b.py",
    "grep -rn supervisord ~/lloyd/agent-services",
    "sort < ~/lloyd/agent-services/supervisord.conf",
    "sed -n '1,10p' ~/lloyd/agent-services/conf/guardian.env",
    "cp ~/lloyd/agent-services/supervisord.conf /tmp/backup.conf",
    "cat ~/lloyd/.venvs/pyvenv.cfg > /tmp/out.txt",
    "cd ~/lloyd && mv server.py server.py.bak",
]


#: The store the venv interpreter links into (#1741), under the scratch home. The
#: live spelling — `/home/alansrobotlab/.local/share/uv/python/cpython-3.12…/bin`,
#: which all five `.venvs/*/pyvenv.cfg` files on this box name — is never a target
#: here, for the same reason as the rest of this section.
UV_STORE_BIN = ".local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin"

#: The traffic #1741 must leave allowed: the store read, the store as a `cp`
#: *source*, a `.venvs`-shaped path in somebody else's project, and one real write
#: — into the uv *tools* tree beside the store. Not one of them writes into
#: `~/.local/share/uv/python`, which is the only thing the new entry covers. A rule
#: keyholed on the word `python`, on a `.venvs` suffix, or widened a directory up
#: — the three obvious ways to write this fix — loses here while still refusing the
#: clobber, which is why it lives on the benign side of the file.
_ALLOWED_AROUND_THE_UV_STORE = [
    "ls -la ~/.local/share/uv/python",
    "cp {store}/python3.12 /tmp/x",
    "echo x > ~/otherproject/.venvs/lloyd/pyvenv.cfg",
    # The scope rail, as a write: the uv *tools* tree beside the store holds the
    # supervisorctl automod's own promotion execs
    # (`scripts/automod/promote.py:97`), so an entry widened one directory up to
    # `~/.local/share/uv` refuses the parachute with everything else.
    "echo unit | tee -a ~/.local/share/uv/tools/supervisor/bin/supervisorctl",
]


@pytest.fixture()
def deny_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for rel in _DENY_FILES:
        p = home / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    (home / "obsidian" / "knowledge").mkdir(parents=True, exist_ok=True)
    # The #1741 layout: the venv interpreter is a link into the store, and there
    # is an ordinary project whose tree merely looks like a protected one.
    store = home / UV_STORE_BIN
    store.mkdir(parents=True)
    (store / "python3.12").write_text("x\n")
    venv_bin = home / "lloyd" / ".venvs" / "lloyd" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(store / "python3.12")
    (home / "otherproject" / "lloyd" / ".venvs" / "lloyd").mkdir(parents=True)
    # Two links standing inside a denied directory, one pointing out of the set and
    # one into it (#1741 clause 5). Built here and nowhere else: the live tree must
    # not gain an `agent-services/link.conf` to make a test pass, and tmp is the
    # only tree that may hold one.
    outside = home / "notes" / "unit.conf"
    outside.parent.mkdir(parents=True)
    outside.write_text("x\n")
    (home / "lloyd" / "agent-services" / "link-out.conf").symlink_to(outside)
    (home / "lloyd" / "agent-services" / "link-in.conf").symlink_to(
        home / "obsidian" / "lloyd" / "SOUL.md")
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.parametrize("command", _MERELY_NAMES_THE_SET)
def test_the_write_check_allows_a_deny_set_path_it_only_names(deny_home, command):
    assert safety.check_bash_command(
        command, str(deny_home / "lloyd"),
        session_id="20260927_143507_chatabc") is None, command


@pytest.mark.parametrize("command", _ALLOWED_AROUND_THE_UV_STORE)
def test_the_write_check_allows_the_store_it_was_widened_for(deny_home, command):
    """#1741's benign arm: widening the deny-set to the uv store changes the answer
    for no command that does not write into the store.

    Reading the store (`ls`), copying *out* of it (`cp … /tmp/x`) and writing a
    `.venvs`-shaped path in somebody else's project are ordinary work — the first
    two are how a person diagnoses an interpreter, the third is what a second
    checkout looks like to a suffix matcher, and a rule written as a word match on
    `python` or as a `.venvs` suffix loses on them while still refusing the clobber.
    The fourth line writes for real, into the uv *tools* tree, and it is the one
    that answers refused if the entry is ever widened a directory up to
    `~/.local/share/uv`: automod's own promotion execs
    `~/.local/share/uv/tools/supervisor/bin/supervisorctl`
    (`scripts/automod/promote.py:97`), so that widening puts the parachute inside the
    deny-set. What does start refusing is a write *into* the store, `uv python
    install` included: that writer is a human command (`SETUP.md:166`) and an agent
    owes a grant, which is the consequence the item asked for. The attack half of
    this pair is
    `tests/test_bash_write_guard.py::test_a_write_to_the_linked_interpreter_is_refused_at_dispatch`.
    """
    cmd = command.format(store=deny_home / UV_STORE_BIN)
    assert safety.check_bash_command(
        cmd, str(deny_home / "lloyd"),
        session_id="20260927_143507_chatabc") is None, cmd


def test_the_write_check_judges_a_link_by_where_it_points(deny_home, tmp_path):
    """#1741 clause 5, on the Bash lane, which had no pin for this at all: a link
    standing inside a denied directory is refused or allowed by its target, because
    realpath-first is one rule and the deny-set gained its fifth entry precisely
    because of it.

    Both links are in the `deny_home` scratch home, not in the live tree —
    `~/lloyd/agent-services/link.conf` is production and nothing here may create
    it. The pair is what makes the fifth entry honest rather than absolute: the
    store entry does not extend the set to anything that merely touches a venv, and
    a link out of the set is still writable, so the fix cannot be "everything under
    those trees is unwriteable, permanently, by anyone".
    """
    out = "echo unit > ~/lloyd/agent-services/link-out.conf"
    assert safety.check_bash_command(out, str(deny_home / "lloyd"),
                                     session_id="20260927_143507_chatabc") is None, out
    soul = deny_home / "obsidian" / "lloyd" / "SOUL.md"
    assert soul.read_text() == "x\n"
    refused = "echo unit > ~/lloyd/agent-services/link-in.conf"
    match = safety.check_bash_command(refused, str(deny_home / "lloyd"),
                                      session_id="20260927_143507_chatabc")
    assert match is not None, refused
    assert soul.read_text() == "x\n", "a refusal must not have run the shell"
    # The same rule, the other way, for the round's own tree: a path ending in the
    # `agent-services/…` the entry names, sitting outside the set. Prefix, not
    # suffix — this is the property `test_the_write_check_lets_a_round_edit_its_own_worktree`
    # pins for the four original entries, restated for the enlarged set so the
    # fifth is not assumed to have left it standing.
    unit = (tmp_path / "lloyd-work" / "SM_20260928_000000" / "home" / "lloyd"
            / "agent-services" / "supervisord.conf")
    unit.parent.mkdir(parents=True)
    unit.write_text("x\n")
    worktree = f"cp /tmp/unit.conf {unit}"
    assert safety.check_bash_command(
        worktree, str(deny_home / "lloyd"),
        session_id="20260927_143507_chatabc") is None, worktree


#: The attack arm lives in `tests/test_bash_write_guard.py`; loading it the way the
#: header loads `tests/unit/test_harness_safety.py` is what lets the node below
#: re-run *that* file's shapes instead of keeping a second copy of them here — a
#: second list of the same eight spellings is a list that silently diverges from
#: the one the attack arm actually tests.
_GUARD_SPEC = importlib.util.spec_from_file_location(
    "_bash_write_guard", LLOYD_HOME / "tests" / "test_bash_write_guard.py")
guard = importlib.util.module_from_spec(_GUARD_SPEC)
_GUARD_SPEC.loader.exec_module(guard)

#: Two `cat > /tmp/probe.py <<'EOF'` bodies that are data and nothing else. The
#: plain one is the clause's own wording; the table is trimmed from the case table
#: in `~/lloyd-data/sessions/20260927_143443_autocode_5659.json`, and keeps the
#: `'sed -i \'s/a/b/\' …'` line because that line's odd number of quotes is what
#: left the entries after it standing *outside* a quoted string to the old parser
#: — a balanced case table never tripped it, and a test built from one alone would
#: pass at the base commit and pin nothing.
_CAT_BODIES = [
    "cat > /tmp/probe.py <<'EOF'\n"
    "echo x > ~/.openclaw/credentials.json\n"
    "EOF\n",
    "cat > /tmp/probe.py <<'EOF'\n"
    "cases = [\n"
    '    "echo x > ~/.openclaw/credentials.json",\n'
    '    "echo x | tee -a ~/lloyd/agent-services/supervisord.conf",\n'
    "    'sed -i \\'s/a/b/\\' notes.md',\n"
    '    "mv /tmp/evil ~/obsidian/lloyd/SOUL.md",\n'
    "]\n"
    "EOF\n",
]


@pytest.mark.parametrize("cmd", _CAT_BODIES)
def test_a_heredoc_body_of_data_is_not_read_for_write_targets(deny_home, cmd):
    """#1740 clause 1: a heredoc body is program text, so neither of its two
    shapes is a write target — while the redirect the command itself spells still is.

    `_tokens` rewrites every `\\n` to ` ; ` and nothing stripped the body, so each
    body line was visited as its own command segment and a `>` inside it read as a
    redirect. Measured against the live tree at `b002b2e9`, the plain body returned
    2 write targets and the table 3, both refused on both lanes, and the table's
    extra targets came back still carrying a closing quote the shell never handed
    them (`…/credentials.json",`) — a token `shlex` had not stripped, which is how
    the item recognised them as data. Both halves are asserted here: the guard
    answers None on each lane, and `write_targets` still reports the one target the
    operator's own line names, so the node cannot pass by the whole check having
    been switched off.
    """
    assert PP.write_targets(cmd, str(deny_home / "lloyd")) \
        == [("/tmp/probe.py", "a redirect into")], cmd
    assert safety.check_bash_command(cmd, str(deny_home / "lloyd"),
                                     session_id="20260927_143507_chatabc") is None, cmd
    decision, reason = guard._hook_decision(cmd, str(deny_home / "lloyd"))
    assert decision != "deny", (decision, reason)
    assert "protected write" not in reason, reason


def test_the_write_guard_still_refuses_every_shape_the_attack_arm_names(deny_home):
    """#1740 clause 5, both directions in one run: the narrowing that freed the
    data above freed no write.

    The baseline pair has to stay clean (`false_blocked == 0 and misses == 0`, the
    same assertion `test_both_directions_discriminate_against_the_same_baseline`
    makes of it) *and* every spellings table the attack arm holds must still come
    back refused, because a fix that stops reading program text as shell is one
    edit away from also not reading a shell payload as shell. `bash -c` is the
    shape in that table that keeps being parsed as shell after #1740 — a shell
    body or a shell `-c` argument is code, not data — and `interpreter-open-write`
    is the one that keeps being read for its write calls.
    """
    base = corpus.run_safety_pair()
    assert base["false_blocked"] == 0 and base["misses"] == 0, (
        f"the baseline pair is no longer clean: {corpus.format_safety_pair(base)}")
    shapes = dict(guard.DENY_SET_WRITES)
    assert len(shapes) >= 8, shapes
    for key, template in sorted(shapes.items()):
        cmd = template.format(home=deny_home, repo=deny_home / "lloyd")
        assert safety.check_bash_command(
            cmd, str(deny_home / "lloyd"),
            session_id="20260927_143507_chatabc") is not None, (key, cmd)


def test_the_write_check_lets_a_round_edit_its_own_worktree(deny_home, tmp_path):
    """Prefix, not suffix: this path ends in the same `agent-services/…` the deny
    entry names and is nowhere under the entry, and a check that refused it would
    stop a round changing the service units it was opened to change."""
    unit = (tmp_path / "lloyd-work" / "SM_20260927_213507" / "home" / "lloyd"
            / "agent-services" / "supervisord.conf")
    unit.parent.mkdir(parents=True)
    unit.write_text("x\n")
    cmd = f"cp /tmp/unit.conf {unit}"
    assert safety.check_bash_command(cmd, str(deny_home / "lloyd")) is None, cmd


#: The four seeding tests, which take no fixture: running this file directly
#: covers these, while the #1620 section above needs `deny_home` and runs under
#: pytest only (`-k write_check`).
TESTS = [
    test_seeding_a_broad_pattern_makes_the_benign_arm_strictly_worse,
    test_removing_a_pattern_makes_the_true_positive_arm_strictly_worse,
    test_both_directions_discriminate_against_the_same_baseline,
    test_seeding_is_undone_and_the_baseline_pair_still_holds,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e!r}")
            failed += 1
    if failed:
        print(f"{failed}/{len(TESTS)} tests failed")
        sys.exit(1)
    print(f"All {len(TESTS)} tests passed")
