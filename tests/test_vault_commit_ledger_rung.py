"""The ledger-strand refusal, at BOTH of Lloyd's vault commit doors (#3589).

`lloyd/MEMORY.md` and `lloyd/USER.md` are the loaded memory index: the prompt receives
their bullet lines verbatim. `lloyd/memory/memory-md-ledger.md` and
`user-md-ledger.md` carry one row per bullet — the row is the record (its `why`, its
`retire_when`, the command that re-checks it) and the index line is the only thing that
reaches it. Dream #47's fold, vault commit `085f714b` (2026-10-29 11:17, *"dedupe 7
loaded-memory lines into SOUL.md/USER.md"*), rewrote `lloyd/MEMORY.md` and
`lloyd/USER.md` and named NEITHER ledger in its `git show --stat`. Seven rows detached
in one commit, and `scripts/memory/memory_ledger.py retire` — the ledger's only
deleter, because rows arrive on every knowledge write and nothing else removes them —
answers that file with `refused, 7 orphan row(s) undispositioned (#2415)`. The ledger
was at 86,023 B against the derived 102,400 B bound (`app/memory_ceiling.py:478`
`LEDGER_MULTIPLIER = 4` times `memory_ceiling("MEMORY.md")`) when this item was
triaged, so the growth bound is now a bound with no deleter behind it.

Why a call site rather than more prose. `memory_ledger.py status` prints the orphan
count, and `scripts/memory/validate_memory_index.py` has carried the counts as
"REPORT-ONLY by decision, not by omission (#2415 owed ruling 2)" since it landed
(`d13514fa`, 2026-10-08) — three weeks BEFORE the fold. So printing the number is
proved insufficient by the sequence itself, and the landed discriminator is
guard > template > prose. What did not exist anywhere was a step that refuses the
DIFF.

Why the refusal is hard, where the neighbouring sidecar rung (#3236) is report-only.
A stray WAL is never what a commit was for, so `git reset -q --` is a repair. A stranded
row is the commit's own defect and the repair belongs to the committer: unstaging the
row would silently rewrite what another caller chose to commit, and warning alone is
the exact state that produced the seven. So both doors stop the commit and touch
nothing — the staged diff and the working tree stay as the caller left them, because
adding the row's disposition to the same batch is the whole fix.

Why the check is diff-relative and not the whole-tree count. `memory_ledger.py status`
prints `MEMORY.md: … orphan rows: 7` on the tree this round was written on, and those
seven are step-2a-ter-curation's disposition backlog (#2415), not every later writer's
blocking condition. A clause keyed on that count would fail every nightly until the
curation pass runs — which is precisely the outcome #2415's owed ruling 2 withheld. So
the matcher asks only what THIS commit's index-vs-HEAD diff removes, and a commit that
touches no loaded index line is never refused however many orphans the tree carries.
`tests/test_memory_index_ledger_orphans.py` pins that boundary; the fixture below
therefore ships one orphan row (`DEAD_ANCHOR`) that is undispositioned at HEAD and
stays that way, and the nodes here assert it never reddens a commit that did not
create it.

ONE MATCHER, TWO DOORS. `scripts/util/ledger_strand_findings.py` is the rule; the
readers are `scripts/util/vault-commit.sh` (bash, run as a subprocess after its `git
add -A`) and `scripts/automod/vault_round.py::land()`, which has its own `git add` and
its own `git commit` and never calls the wrapper — the shape #3236's
`sqlite_sidecar_findings.py` pair established, and #2468 is what guarding one door and
calling it coverage costs. `085f714b`'s own subject line ("merge the vault-land pair")
points at the second door, the one a bash-only rung misses. The node that walks BOTH
files is `test_both_commit_doors_read_the_one_matcher_and_restate_nothing`; the nodes
below walk each door end to end through its real entry point.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UTIL = ROOT / "scripts" / "util"
WRAPPER = UTIL / "vault-commit.sh"
MODULE = UTIL / "ledger_strand_findings.py"
VAULT_ROUND = ROOT / "scripts" / "automod" / "vault_round.py"

MEMORY_REL = "lloyd/MEMORY.md"
USER_REL = "lloyd/USER.md"
MEMORY_LEDGER_REL = "lloyd/memory/memory-md-ledger.md"
USER_LEDGER_REL = "lloyd/memory/user-md-ledger.md"
MEMORY_ARCHIVE_REL = "lloyd/reviews/memory-md-ledger-archive.md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: The instrument the matcher reads its definitions from, loaded the same way so the
#: anchors this file writes are the writers' own — never a hand-copied 60 characters.
ML = _load("memory_ledger_for_strand_rung", ROOT / "scripts" / "memory" / "memory_ledger.py")
LSF = _load("ledger_strand_findings_under_rung_test", MODULE)

FRONT = "---\ntype: note\ntimestamp: '2026-01-05T00:00:00'\n---\n"

# Three loaded lines with three different relationships to the ledger, because the
# rule is about ROWS and not about lines: `KEEP` and `GONE` both have a row, `UNROWED`
# never had one. Removing the last is lawful and must not be refused, which is the
# positive control that the guard keys on the join and not on "a bullet disappeared".
KEEP = ("- [project] (2026-01-02) **A stored command whose root moved** — a stored "
        "argv naming the old root returns a clean non-answer -> topics/a-stored-command\n")
GONE = ("- [feedback] (2026-01-03) **Never re-apply the merge** — the three names stay "
        "distinct -> topics/entity-resolution\n")
UNROWED = ("- [project] (2026-01-04) **A line the ledger never covered** — present to be "
           "removed without a refusal -> topics/unrowed\n")
USER_LINE = "- [user] Speaks English, reads metric units\n"

#: A row whose loaded line is ALREADY gone at HEAD: the fixture's stand-in for the
#: seven `status` prints today. It is undispositioned on purpose, and every node that
#: commits an unrelated change proves it stays nobody else's blocking condition.
DEAD_ANCHOR = ML.anchor_of("- [project] (2025-12-31) **A row whose index line was "
                           "removed by an earlier commit** -> topics/dead\n")


def _row(anchor: str, tag: str = "project") -> str:
    """One row in the shipped shape: `- [tag] (date) anchor: <a> | why: …`."""
    return (f"- [{tag}] (2026-01-05) anchor: {anchor} | why: fixture row | "
            f"origin: fixture | retire_when: never | check: `true` | "
            f"checked: 2026-01-05 (fixture)\n")


def _memory(keep: str = KEEP, gone: str = GONE, unrowed: str = UNROWED,
            tail: str = "\nSee `topics/`.\n") -> str:
    return FRONT + "# Index\n\n" + keep + gone + unrowed + tail


def _memory_ledger() -> str:
    return (FRONT + "# MEMORY.md ledger\n\n"
            + _row(ML.anchor_of(KEEP)) + _row(ML.anchor_of(GONE)) + _row(DEAD_ANCHOR))


def _user_ledger() -> str:
    return FRONT + "# USER.md ledger\n\n" + _row(ML.anchor_of(USER_LINE), tag="user")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """A git vault with the two loaded indexes, both ledgers, and one orphan row.

    Built the way the sidecar-rung fixture builds its vault — `git init -b main` plus a
    base commit — because the guard reads `git diff --cached` and needs a real HEAD to
    diff against, and the answer must come from the same `git` the door will use.
    """
    v = tmp_path / "obsidian"
    (v / "lloyd" / "memory").mkdir(parents=True)
    (v / "backlog").mkdir()
    _git(tmp_path, "init", "-q", "-b", "main", str(v))
    _git(v, "config", "user.email", "t@e.com")
    _git(v, "config", "user.name", "t")
    (v / MEMORY_REL).write_text(_memory())
    (v / USER_REL).write_text(FRONT + "# Profile\n\n" + USER_LINE)
    (v / MEMORY_LEDGER_REL).write_text(_memory_ledger())
    (v / USER_LEDGER_REL).write_text(_user_ledger())
    (v / "backlog" / "9-item.md").write_text("---\nstatus: draft\n---\n# item\n")
    _git(v, "add", "-A")
    _git(v, "commit", "-q", "-m", "base")
    return v


def _run_wrapper(vault: Path, msg: str, paths: list[str] | None = None,
                 python: str | None = None) -> subprocess.CompletedProcess:
    """Door one, really invoked: the wrapper, not the module under it.

    `VAULT_DIR` is the knob `vault-commit.sh:129` reads; `LLOYD_PYTHON` is `:138`'s.
    Nothing is monkeypatched, because the boundary this item is about is a bash script
    exec'ing a Python file, and a mocked seam pins everything except the one line that
    matters (#777 is the record of that mistake on the neighbouring door).
    """
    # The message is POSITIONAL and paths follow ` -- `. `VAULT_DIR` is what this
    # wrapper reads (`:129`) while `LLOYD_VAULT` is what `app.paths` reads: moving only
    # one of them would have the door guard the live vault instead of this fixture.
    # `LLOYD_PYTHON` stays unset unless a node passes one, because the default is the
    # interpreter resolved from `SCRIPT_DIR` (`:138`) — the one the nightly jobs get.
    argv = ["bash", str(WRAPPER), msg]
    if paths:
        argv += ["--", *paths]
    env = dict(os.environ, VAULT_DIR=str(vault), LLOYD_VAULT=str(vault),
               LLOYD_JOB_ID="test: ledger strand rung")
    env.pop("LLOYD_JOB_WRITES", None)
    env.pop("LLOYD_PYTHON", None)
    if python:
        env["LLOYD_PYTHON"] = python
    return subprocess.run(argv, capture_output=True, text=True, timeout=180, env=env)


def _head(vault: Path) -> str:
    return _git(vault, "rev-parse", "HEAD").stdout.strip()


def _staged(vault: Path) -> list[str]:
    return sorted(_git(vault, "diff", "--cached", "--name-only").stdout.split())


# ---------------------------------------------------------------- clause 1: door one


def test_removing_a_loaded_line_whose_row_is_still_live_refuses_before_git_commit(vault):
    """The refusal happens instead of a commit, not merely alongside one."""
    (vault / MEMORY_REL).write_text(_memory(gone=""))
    before = _head(vault)

    r = _run_wrapper(vault, "dream fold: dedupe one loaded line")

    assert r.returncode != 0, f"the door let the strand through: {r.stdout}{r.stderr}"
    assert _head(vault) == before, "a refusal must not commit anything"
    assert "LEDGER STRAND" in r.stderr, r.stderr


def test_the_refusal_names_the_removed_line_and_the_ledger_that_owes_the_disposition(vault):
    """A caller who never opens the module has to be able to act on the message."""
    (vault / MEMORY_REL).write_text(_memory(gone=""))

    r = _run_wrapper(vault, "dream fold: dedupe one loaded line")

    err = r.stderr
    assert ML.anchor_of(GONE) in err, f"the anchor is not named: {err}"
    assert "**Never re-apply the merge**" in err, f"the removed line is not quoted: {err}"
    assert MEMORY_LEDGER_REL in err, f"the owed ledger is not named: {err}"
    # Both lawful dispositions (#2415), because "refused" without the repair is how a
    # caller ends up committing the same thing through the other door.
    assert MEMORY_ARCHIVE_REL in err, f"the archive route is not named: {err}"
    assert "#2415" in err, err


def test_the_same_commit_that_deletes_the_row_is_committed(vault):
    """Deleting the row is a disposition, and it does not have to be an archive."""
    before = _head(vault)
    (vault / MEMORY_REL).write_text(_memory(gone=""))
    text = (vault / MEMORY_LEDGER_REL).read_text()
    (vault / MEMORY_LEDGER_REL).write_text(text.replace(_row(ML.anchor_of(GONE)), ""))

    r = _run_wrapper(vault, "dream fold: dedupe one loaded line, retire its row")

    assert r.returncode == 0, f"a dispositioned fold was refused: {r.stderr}"
    assert _head(vault) != before, "nothing was committed"
    assert MEMORY_LEDGER_REL in _git(vault, "show", "--stat", "--format=", "HEAD").stdout, \
        "the commit does not carry the ledger it owes"


def test_the_same_commit_that_moves_the_row_to_the_archive_is_committed(vault):
    """The other half of #2415's disposition: the row leaves the live ledger, it goes out."""
    anchor = ML.anchor_of(GONE)
    (vault / MEMORY_REL).write_text(_memory(gone=""))
    text = (vault / MEMORY_LEDGER_REL).read_text()
    (vault / MEMORY_LEDGER_REL).write_text(text.replace(_row(anchor), ""))
    archive = vault / MEMORY_ARCHIVE_REL
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(FRONT + "# memory-md-ledger retired rows\n\n" + _row(anchor))

    r = _run_wrapper(vault, "dream fold: dedupe one loaded line, archive its row")

    assert r.returncode == 0, f"an archived disposition was refused: {r.stderr}"
    assert MEMORY_ARCHIVE_REL in _git(vault, "show", "--stat", "--format=", "HEAD").stdout


def test_a_clean_staged_diff_exits_zero_and_prints_nothing(vault):
    """The clause's other half, verbatim: no deleted loaded line means silence."""
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")

    r = _run_wrapper(vault, "backlog: close #9")

    assert r.returncode == 0, r.stderr
    assert "LEDGER STRAND" not in r.stdout + r.stderr, r.stderr


def test_a_line_reworded_inside_its_anchor_is_refused_like_a_deletion(vault):
    """The dream-#47 fold shape: the bullet survives, its 60-character anchor does not.

    `ENTRY_FOLD` in `tests/test_memory_index_ledger_orphans.py` is the same trick —
    one character inside the first 60, bytes roughly neutral, `status` printing a
    detached row — and from the ledger's seat a moved anchor strands the row exactly as
    a removed line does. A guard that keyed on `^-` diff lines would have missed the
    fold that this whole item exists because of.
    """
    reworded = GONE.replace("**Never re-apply the merge**", "**Never re-merge those**")
    assert ML.anchor_of(reworded) != ML.anchor_of(GONE), "the fixture stopped detaching"
    (vault / MEMORY_REL).write_text(_memory(gone=reworded))

    r = _run_wrapper(vault, "dream fold: reword one loaded line")

    assert r.returncode != 0, "an anchor move was let through"
    assert ML.anchor_of(GONE) in r.stderr, r.stderr


def test_removing_a_line_that_never_had_a_row_is_committed(vault):
    """The positive control: the rule is the row join, not 'a bullet went away'."""
    (vault / MEMORY_REL).write_text(_memory(unrowed=""))

    r = _run_wrapper(vault, "curation: drop an index line the ledger never covered")

    assert r.returncode == 0, f"an unrowed line was refused: {r.stderr}"


def test_removing_prose_rather_than_an_index_entry_is_committed(vault):
    """`loaded_entries` excludes headings, front matter and non-bullets — so must this."""
    (vault / MEMORY_REL).write_text(_memory(tail="\nSee `topics/` for the bodies.\n"))

    r = _run_wrapper(vault, "curation: reword the index trailer")

    assert r.returncode == 0, f"a prose edit was refused: {r.stderr}"


def test_a_user_md_line_owes_its_row_to_the_user_ledger_and_says_so(vault):
    """One loaded file is not enough: `USER.md` has its own ledger to answer to."""
    (vault / USER_REL).write_text(FRONT + "# Profile\n")

    r = _run_wrapper(vault, "dream fold: dedupe one profile line")

    assert r.returncode != 0, r.stderr
    assert USER_LEDGER_REL in r.stderr, f"named the wrong ledger: {r.stderr}"
    assert MEMORY_LEDGER_REL not in r.stderr, f"blamed the other ledger: {r.stderr}"


def test_a_refusal_unstages_nothing_and_rewrites_no_file(vault):
    """The difference from #3236's `--unstage`, pinned as a state and not a promise.

    A guard that quietly dropped the ledger path from the index would hand the caller a
    commit that no longer contains what it asked for, and the item's whole point is that
    the omission belongs to the committer. So the index keeps both the index edit and
    whatever else was staged, and the file on disk is byte-identical.
    """
    (vault / MEMORY_REL).write_text(_memory(gone=""))
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")
    proposed = (vault / MEMORY_REL).read_text()

    r = _run_wrapper(vault, "dream fold: dedupe one loaded line")

    assert r.returncode != 0, r.stderr
    assert MEMORY_REL in _staged(vault), "the guard unstaged the caller's own edit"
    assert "backlog/9-item.md" in _staged(vault), "it unstaged someone else's too"
    assert (vault / MEMORY_REL).read_text() == proposed


def test_a_broken_check_is_said_out_loud_and_does_not_stop_an_unrelated_commit(vault):
    """Exit 3 is the rung failing; laundering it into a clean read is the old blindness.

    `LLOYD_PYTHON` is the wrapper's own interpreter knob (`vault-commit.sh:138`), so
    this points every rung at a stub that exits 3 — which is what a missing
    `memory_ledger.py` beside the instrument, or a raised exception inside it, looks
    like from the door. An unrelated commit still lands, because a nightly that cannot
    snapshot the vault is worse than one snapshotting a possible strand; the sentence
    saying the check did not run is what keeps the two cases distinguishable afterwards.
    """
    stub = vault.parent / "stub-python.sh"
    stub.write_text("#!/bin/sh\nexit 3\n")
    stub.chmod(0o755)
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")

    r = _run_wrapper(vault, "backlog: close #9", python=str(stub))

    assert r.returncode == 0, r.stderr
    assert "ledger strand CHECK FAILED" in r.stderr, r.stderr


# ------------------------------------------------------------------ clause 2: both doors


def test_both_commit_doors_read_the_one_matcher_and_restate_nothing():
    """A rung on the bash door alone cannot pass this node (#2468, two doors).

    Textual on purpose, and it is a reachability claim rather than a behaviour claim,
    so text is the honest evidence: what would break is a door that stopped importing
    the module or grew its own copy of the join. The two doors' behaviour is pinned by
    `test_removing_a_loaded_line_whose_row_is_still_live_refuses_before_git_commit`
    (door one, subprocess) and
    `test_the_second_door_refuses_its_own_commit_before_it_calls_git_commit` (door two,
    in-process), both in this file.
    """
    bash = WRAPPER.read_text()
    py_door = VAULT_ROUND.read_text()
    assert "ledger_strand_findings.py" in bash, "the bash door stopped running the matcher"
    assert "STRAND_RC" in bash and "exit 1" in bash, "the bash door stopped refusing"
    assert "ledger_strand_findings" in py_door, "the vault-land door never reads the matcher"
    assert "LSF.findings(" in py_door, "the vault-land door never calls it"

    # One rule, so neither door may carry its own copy of what an entry or a row is.
    # `anchor_of` and `ledger_rows` live in the instrument and the matcher; a door that
    # re-derived them is the second reader #2212 exists to prevent.
    for name, src in (("vault-commit.sh", bash), ("vault_round.py", py_door)):
        assert "anchor_of" not in src, f"{name} re-implements the anchor join"
        assert "ledger_rows" not in src, f"{name} re-implements the row join"


def test_each_door_runs_the_matcher_before_it_runs_its_own_git_commit():
    """Placement, per door: a refusal after the commit is a comment, not a guard."""
    bash = WRAPPER.read_text()
    refused = bash.index("LEDGER STRAND CHECK REFUSED")
    # The commit line is an array expansion, not a literal two-word `git commit`:
    # matching the first line that starts with `git` and mentions `commit` is the only
    # needle that survives the next reformatting of that invocation.
    commit = re.search(r"^git\b.*\bcommit\b.*$", bash, re.MULTILINE)
    assert commit, "the bash door's own git commit could not be located"
    assert refused < commit.start(), "the bash door refuses after its own git commit"
    # And after the door's own staging: the matcher reads the INDEX, so running it
    # before `git add` would check a state the commit will not contain. Line-anchored for
    # the same reason as the commit needle, and more sharply here: `git add -A --` occurs
    # three times in this script's COMMENTS (`:249`, `:276`, `:286`) before its one real
    # call at `:311`, so an unanchored `bash.index` of that string lands on a comment
    # that sits before every rung — an assertion that passes whether or not the door ever
    # stages before checking.
    add = re.search(r"^\s*git add -A --", bash, re.MULTILINE)
    assert add, "the bash door's own staging call could not be located"
    assert add.start() < refused, "the bash door runs the matcher before it stages"
    py_door = VAULT_ROUND.read_text()
    refused = py_door.index("LSF.findings(")
    landed = py_door.index('_git("commit", "-q", "-m", message.strip())')
    assert refused < landed, "land() refuses after its own git commit"
    # And before the #2381 witness print, which says "landed <sha>" out loud before the
    # commit exists: refusing after it would leave a witness claiming a landing that
    # this function then refused.
    assert refused < py_door.index("witness = _witness_findings()")


def test_the_matcher_never_reads_the_working_tree_only_the_index(tmp_path, vault):
    """An edit that was never staged is not this commit's, and must not be refused.

    The wrapper's own `git add -A -- "${CHANGED[@]}"` (`:311`) stages a path only under
    the caller's pathspec, so a scoped commit of `backlog/9-item.md` genuinely does not
    contain the `MEMORY.md` edit sitting in the tree. Proving the guard reads
    `git diff --cached` and nothing else is what makes that case safe; had it read the
    worktree, one writer's in-flight curation edit would block every later job's
    snapshot — the #2245 stranding, redeployed as a guard.
    """
    (vault / MEMORY_REL).write_text(_memory(gone=""))  # edited, never staged
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")
    assert _staged(vault) == [], "the fixture staged more than it meant to"

    assert LSF.stranded_rows(vault) == [], "the matcher looked past the index"

    r = _run_wrapper(vault, "backlog: close #9", paths=["backlog/9-item.md"])
    assert r.returncode == 0, r.stderr
    assert MEMORY_REL not in _git(vault, "show", "--stat", "--format=", "HEAD").stdout


# ------------------------------------------------------------------- clause 2: door two


@pytest.fixture()
def land_vault(vault, tmp_path, monkeypatch):
    """The same tree, plus `vault_round.py` pointed at it.

    `land()` resolves the vault through the module-level `VAULT`, which is how every
    `tests/test_automod_vault_round.py` fixture aims it at a scratch repo. The
    prompt-surface loaders and the code-agreement probe are stubbed: they are real rails
    on the landing path and neither is this item's claim, and both spawn
    subprocesses — see `tests/test_automod_vault_round.py:55` and `:1983` for the same
    two stubs on the neighbouring nodes.
    """
    import scripts.automod.vault_round as V
    import scripts.automod.state as S

    monkeypatch.setattr(V, "VAULT", vault)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    monkeypatch.setattr(V, "contract_errors", lambda paths: [])
    monkeypatch.setattr(V.VG, "agreement", lambda **kw: {"state": "checked",
                                                              "refuse": False})
    return vault


def test_the_second_door_refuses_its_own_commit_before_it_calls_git_commit(land_vault):
    """`085f714b` is a landing-shaped commit: the bash door was never going to see it."""
    import scripts.automod.vault_round as V

    v = land_vault
    before = _head(v)
    (v / MEMORY_REL).write_text(_memory(gone=""))

    with pytest.raises(V.VaultRoundError) as exc:
        V.land([MEMORY_REL], "dream fold: dedupe one loaded line")

    assert ML.anchor_of(GONE) in str(exc.value), str(exc.value)
    assert MEMORY_LEDGER_REL in str(exc.value), str(exc.value)
    assert _head(v) == before, "door two committed anyway"
    assert _git(v, "status", "--porcelain").stdout.count(MEMORY_REL) == 1, \
        "the refusal reverted or destroyed the caller's edit"
    rows = [e for e in _events("vault_land")]
    assert rows and rows[-1]["ok"] is False, rows
    assert rows[-1].get("strand_refusals"), "the refusal named nothing on the ledger"


def test_the_second_door_lands_when_the_row_is_dispositioned_in_the_same_batch(land_vault):
    """The same call, one extra path in it: the fix is the batch, not a different tool."""
    import scripts.automod.vault_round as V

    v = land_vault
    anchor = ML.anchor_of(GONE)
    (v / MEMORY_REL).write_text(_memory(gone=""))
    (v / MEMORY_LEDGER_REL).write_text(
        (v / MEMORY_LEDGER_REL).read_text().replace(_row(anchor), ""))

    out = V.land([MEMORY_REL, MEMORY_LEDGER_REL], "dream fold: dedupe one line, retire its row")

    assert out["ok"] is True
    shown = _git(v, "show", "--stat", "--format=", out["commit"]).stdout
    assert MEMORY_LEDGER_REL in shown, shown


def _events(kind):
    import scripts.automod.state as S
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]
