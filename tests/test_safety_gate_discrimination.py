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


@pytest.fixture()
def deny_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for rel in _DENY_FILES:
        p = home / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    (home / "obsidian" / "knowledge").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.parametrize("command", _MERELY_NAMES_THE_SET)
def test_the_write_check_allows_a_deny_set_path_it_only_names(deny_home, command):
    assert safety.check_bash_command(
        command, str(deny_home / "lloyd"),
        session_id="20260927_143507_chatabc") is None, command


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
