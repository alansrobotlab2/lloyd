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
was at 86,023 B against the derived 102,400 B bound (`app/memory_ceiling.py:494`
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


# ── #3729: the SAME matcher also refuses the ADDITION direction ───────────────
#
# #3589 built this module to catch a commit that REMOVES a loaded line and leaves its
# row live, and its join computes `head_anchors - staged_anchors` — removal-only. The
# inverse was explicitly skipped: an anchor with no row "is either a line that never
# had a row (an `unrepresentable` anchor …)". That sentence is the hole. A line whose
# anchor holds the row parser's own field delimiter `|` can never HAVE a row, so a
# commit that adds one strands nothing today and leaves the index holding a loaded line
# no `retire` can ever reach — which is what 2026-11-02 shipped (`(2026-11-02) **`git
# show X | python3 - <<'PY'` reads no bytes…`, `memory_ledger.py status` printing
# `unrepresentable: 1` at it ever since). The write-path arm in
# `app/memory_ceiling.memory_write_error` stops the tool call; these nodes pin the
# commit door, which is what makes the run that landed the line SEE the refusal rather
# than read a report a day later.

#: An added line whose anchor trips `memory_ledger.representable`: the 60-character
#: anchor contains the row parser's field delimiter `|`.
PIPE_ADDED = ("- [project] (2026-01-07) **`git show HEAD | cat` shows no bytes** "
              "-> topics/pipe-in-anchor\n")

#: The same shape with a readable anchor: must still commit, or the door is a freeze.
CLEAN_ADDED = ("- [project] (2026-01-07) **A pipe-free hook lands fine** "
               "-> topics/pipe-free\n")

#: The OTHER unrepresentable shape: an anchor that opens AND closes on a backtick, which
#: no reader can tell from the legacy form where a row wrapped its anchor in backticks
#: (`memory_ledger._anchor_field` strips one such pair). 56 characters, so the whole hook
#: is inside the 60-character anchor and both backticks survive into it.
BACKTICK_ADDED = "- `a hook wrapped entirely in backticks cannot be retired`\n"


def test_adding_an_unrepresentable_line_refuses_at_door_one_before_git_commit(vault):
    """#3729 clause 4, door one: the wrapper stops a commit that ADDS such a line."""
    before = _head(vault)
    (vault / MEMORY_REL).write_text(_memory() + PIPE_ADDED)

    res = _run_wrapper(vault, "knowledge write: add one loaded line")

    assert res.returncode != 0, (res.stdout, res.stderr, "the commit went through")
    out = res.stdout + res.stderr
    assert "UNREPRESENTABLE ANCHOR REFUSED" in out, out
    assert "git show HEAD | cat" in out, out          # names the added line
    assert "field delimiter" in out, out              # and which rule it broke
    assert _head(vault) == before, "door one committed anyway"
    assert _staged(vault) == [MEMORY_REL], _staged(vault)


def test_the_addition_refusal_names_the_repair_and_causes_no_freeze(vault):
    """The refusal routes the writer, and a pipe-free addition still lands.

    A refusal whose advice cannot be followed is #2838's defect, so the message names
    the rewrite; and because the live index already holds one such line, the door may
    only report what the commit ADDS — proven here by committing the clean addition
    over a vault whose HEAD copy of the bad line stays untouched.
    """
    before = _head(vault)
    (vault / MEMORY_REL).write_text(_memory() + PIPE_ADDED)
    refused = _run_wrapper(vault, "knowledge write: add one loaded line")
    out = refused.stdout + refused.stderr
    assert "Rewrite the hook" in out, out

    (vault / MEMORY_REL).write_text(_memory() + CLEAN_ADDED)
    ok = _run_wrapper(vault, "knowledge write: add one readable line")
    assert ok.returncode == 0, (ok.stdout, ok.stderr)
    # Real witnesses, not a sha-shaped-string check: `len(rev-parse HEAD) == 40` cannot
    # fail on any git repo, so it proved nothing about the line this node is about. HEAD's
    # own copy of the index must carry the readable line, and must NOT have carried the
    # refused pipe line along with it — the door stages before it checks, so a line the
    # refusal left in the INDEX would ride into whichever commit went through next.
    assert _head(vault) != before, "the readable addition never reached HEAD"
    head_index = _git(vault, "show", f"HEAD:{MEMORY_REL}").stdout
    assert CLEAN_ADDED.strip() in head_index, head_index
    assert PIPE_ADDED.strip() not in head_index, "the refused line rode along into HEAD"


def test_the_matcher_reports_only_what_the_commit_adds(vault):
    """Diff-relative on the addition side too: an unrepresentable line at HEAD is not
    this commit's finding, so the rung cannot freeze the file it just refused."""
    LSF = _load("lsf_under_test", ROOT / "scripts" / "util" / "ledger_strand_findings.py")
    # Seed the bad line into HEAD with raw git, past the door: the point is a vault
    # that ALREADY holds an unrepresentable line, which is the state the live vault is
    # in today.
    (vault / MEMORY_REL).write_text(_memory() + PIPE_ADDED)
    _git(vault, "add", "-A")
    _git(vault, "commit", "-q", "-m", "seed: the line the door would have refused")
    assert ML.anchor_of(PIPE_ADDED) in {
        ML.anchor_of(e) for e in ML.loaded_entries(
            _git(vault, "show", f"HEAD:{MEMORY_REL}").stdout)}

    # A later commit that touches no loaded index line says nothing about that line.
    (vault / "backlog" / "10-other.md").write_text("---\nstatus: draft\n---\n# other\n")
    res = _run_wrapper(vault, "backlog: one unrelated item",
                       paths=["backlog/10-other.md"])
    assert res.returncode == 0, (res.stdout, res.stderr)

    # And the matcher's own answer for that vault is empty of addition findings.
    added = LSF.unrepresentable_additions(vault)
    assert [r["anchor"] for r in added] == [], added


def test_the_second_door_refuses_an_added_unrepresentable_line(land_vault):
    """Door two, in-process: the landing run is told, not a report the next day."""
    import scripts.automod.vault_round as V

    v = land_vault
    before = _head(v)
    (v / MEMORY_REL).write_text(_memory() + PIPE_ADDED)

    with pytest.raises(V.VaultRoundError) as exc:
        V.land([MEMORY_REL], "knowledge write: add one loaded line")

    assert "UNREPRESENTABLE ANCHOR REFUSED" in str(exc.value), str(exc.value)
    assert ML.anchor_of(PIPE_ADDED) in str(exc.value), str(exc.value)
    assert _head(v) == before, "door two committed anyway"


# ── #3818: the same matcher also COUNTS an added line that has no row ─────────
#
# The other half of the addition direction, and the opposite enforcement. An
# `unrepresentable` anchor is permanent load no row can ever retire, so a door that lets it
# through has let through something no later commit can repair — that one refuses. An
# UNROWED representable line is a step the writer still controls: measured 2026-11-05,
# `scripts/memory/memory_ledger.py status` printed `MEMORY.md: … 86 entries, 84 ledger rows
# -> without a row: 1`, and that line reached disk through a commit door. #2415's owed
# ruling 2 withheld the REQUIRE transition on purpose — a door that blocked every line
# until its row existed fails every nightly while undispositioned rows sit in the curation
# backlog — so this counts, prints the anchors, and commits. Whether it must REQUIRE the
# row is #3543/#3539's open ruling, which is exactly why the census exists: it is the
# measurement that ruling needs, not a pre-emption of it.

#: The report's own leading token. Column 0 like every other verdict in this family, and
#: the string both doors must print — a census worded differently at the two doors is a
#: property of the door, not of the rule.
CENSUS_HEAD = "UNROWED INDEX ADDS"


def test_an_added_line_with_no_row_lands_and_is_counted_with_its_anchor(vault):
    """#3818 clause 5, door one: rc 0, plus a count and the anchors printed.

    `CLEAN_ADDED` is the same line the no-freeze node above commits, deliberately: this
    node adds that commit OWES a row and shows the door reports it instead of blocking it.
    Three things have to be true together, and each is a different failure — the commit
    lands (a refusal would be the #2415 withheld rule), the count is printed (silence is
    the state that let the 2026-11-05 line through), and the anchor is printed (a bare
    count tells the curator nothing they can act on without opening the index).
    """
    before = _head(vault)
    (vault / MEMORY_REL).write_text(_memory() + CLEAN_ADDED)

    res = _run_wrapper(vault, "knowledge write: add one loaded line, row owed")

    assert res.returncode == 0, (res.stdout, res.stderr)
    assert _head(vault) != before, "the census blocked a commit it may only report"
    out = res.stdout + res.stderr
    assert CENSUS_HEAD in out, out
    assert "(report only, nothing is refused)" in out, out
    assert "1 loaded index line(s)" in out, out
    assert ML.anchor_of(CLEAN_ADDED) in out, out
    assert MEMORY_LEDGER_REL in out, "the census did not name the ledger that owes the row"
    # A census a reader could mistake for a block is its own failure mode.
    assert "LEDGER STRAND REFUSED" not in out and "UNREPRESENTABLE ANCHOR REFUSED" not in out, out


def test_a_commit_that_adds_no_loaded_index_line_prints_no_census(vault):
    """Clause 5's second half: silence is the clean signal, not a printed zero.

    The fixture's HEAD carries `UNROWED` — a loaded line with no row that is nobody's
    backlog in this commit — so a census keyed on the whole file would report 1 here and
    teach the reader to ignore the report. Diff-relative means exactly this: the count is
    what THIS commit adds, and a commit that adds nothing says nothing.
    """
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")
    _git(vault, "add", "-A")
    assert LSF.unrowed_additions(vault) == [], "the census read HEAD's lines as this commit's"

    res = _run_wrapper(vault, "backlog: close #9")

    assert res.returncode == 0, (res.stdout, res.stderr)
    assert CENSUS_HEAD not in res.stdout + res.stderr, res.stderr


def test_the_census_reads_the_staged_ledger_so_a_same_commit_row_is_not_reported(vault):
    """The join is against the LEDGER'S INDEX COPY, the way #3589's strand join is.

    Without the control this node cannot fail for the wrong reason: a census that skipped
    the row lookup entirely would also print nothing here and look correct. So the same
    vault is measured twice — once with the row unstaged, once staged — and the two answers
    are the difference this rung draws.
    """
    (vault / MEMORY_REL).write_text(_memory() + CLEAN_ADDED)
    (vault / MEMORY_LEDGER_REL).write_text(
        (vault / MEMORY_LEDGER_REL).read_text() + _row(ML.anchor_of(CLEAN_ADDED)))

    _git(vault, "add", "-A", "--", MEMORY_REL)          # the row is still only in the tree
    owed = LSF.unrowed_additions(vault)
    assert [r["anchor"] for r in owed] == [ML.anchor_of(CLEAN_ADDED)], owed

    _git(vault, "add", "-A", "--", MEMORY_LEDGER_REL)   # the row joins the same commit
    assert LSF.unrowed_additions(vault) == [], "a same-commit row did not cover the line"

    res = _run_wrapper(vault, "knowledge write: add one loaded line WITH its row")
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert CENSUS_HEAD not in res.stdout + res.stderr, res.stderr


def test_the_second_door_prints_the_same_census_text_and_still_lands(land_vault, capsys):
    """Door two, in-process: the SAME block, and the landing is not withheld (#2468).

    The expected text is built by the matcher itself, from the staged state before the
    landing, so this node fails on a door that paraphrases, reorders, or prints only a
    count — the two doors differ in everything else about how they commit (bash exec'ing a
    subprocess here, in-process with its own `git add` there), so the report is the one
    thing that must not differ.
    """
    import scripts.automod.vault_round as V

    v = land_vault
    before = _head(v)
    (v / MEMORY_REL).write_text(_memory() + CLEAN_ADDED)
    _git(v, "add", "-A")
    expected = LSF.census_report_text(LSF.unrowed_additions(v))
    assert CENSUS_HEAD in expected, (
        f"the census found nothing to report, so the comparison below could not fail: "
        f"{expected!r}")

    out = V.land([MEMORY_REL], "knowledge write: add one loaded line, row owed")
    err = capsys.readouterr().err

    assert out["ok"] is True, out
    assert _head(v) != before, "door two withheld a landing the census may only report on"
    assert expected in err, err
    assert ML.anchor_of(CLEAN_ADDED) in err, err


def test_the_backtick_rule_is_rendered_at_both_doors(land_vault):
    """The second unrepresentable shape, through both real doors.

    `addition_refusal_text` has two `why` branches and #3729's two door nodes seeded only
    the pipe-bearing one, so the backtick sentence — the one a writer of a short
    backticked hook actually meets — had never been rendered by a test. A door that
    crashed while building that branch would have shipped green.
    """
    import scripts.automod.vault_round as V

    v = land_vault
    before = _head(v)
    (v / MEMORY_REL).write_text(_memory() + BACKTICK_ADDED)

    res = _run_wrapper(v, "knowledge write: add a backticked hook")
    out = res.stdout + res.stderr
    assert res.returncode != 0, (out, "door one committed an unretirable line")
    assert "UNREPRESENTABLE ANCHOR REFUSED" in out, out
    assert "opens AND closes on a backtick" in out, out
    assert _head(v) == before, "door one committed anyway"

    with pytest.raises(V.VaultRoundError) as exc:
        V.land([MEMORY_REL], "knowledge write: add a backticked hook")
    assert "opens AND closes on a backtick" in str(exc.value), str(exc.value)
    assert _head(v) == before, "door two committed anyway"
