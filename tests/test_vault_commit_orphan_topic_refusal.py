"""The unlinked-topic-file refusal at the `vault-commit.sh` door (#3545).

`lloyd/MEMORY.md` is the loaded index; `lloyd/memory/<slug>.md` is the detail, and the
`→ topics/<slug>` hook line is the ONLY route it has into a prompt — it is how
`memory_read(file="topics/<slug>")` is reached. So a topic file no index line names is
a rule that exists and is never applied, and #2399 measured exactly that: six such
files sat unread by every prompt for a week while
`scripts/memory/validate_memory_index.py` printed `OK` nightly.

What this rail is NOT is the part worth stating, because every other instrument in the
family is report-only and one very specific decision is why:

  * it is not the validator's whole-tree count. #2415's owed ruling 2 withheld turning
    a nonzero unlinked count red deliberately, and `871da02a` / `3b962ad7` are the
    vault commits that prove printing was not enough — both committed while that script
    reported `ok: false`. What was missing was a call site on the DIFF;
  * it is not a whole-tree rule at all. Of the 29 topic-file adds under `lloyd/memory/`
    since 2026-09-25, 12 landed with no `→ topics/` line in their own commit (41%;
    6 of them from a `nightly-reflection: pre-flight …` sweep commit) — but 17 of the 29
    ALREADY link in the same commit. The healthy writer exists and already does the
    work, which is what makes a commit-boundary refusal satisfiable rather than the
    nightly-wide red #2415 withheld. Scope is therefore `--diff-filter=A`: only a file
    this commit ADDS is this commit's doing, so the sweep commit that carries someone
    else's orphan still lands (#2245's stranded-snapshot failure mode, and clause 4);
  * it is not a working-tree read. The wrapper supports pathspec-scoped commits
    (#1867 `f92b87d9`, #2245 `238c3bc8`), so a hook line that is dirty and unstaged is
    not in the commit — and the failure this item was filed about is a writer whose
    index line exists on disk while the commit it just made leaves the file red. The
    check reads `git show :lloyd/MEMORY.md`, which is what `git commit` will commit.

ONE MATCHER, TWO DOORS, the #3236/#3589 shape. `scripts/util/orphan_topic_findings.py`
is shelled out to by this wrapper (bash, no interpreter to share) and imported by
`scripts/automod/vault_round.py::land()` — which has its own `git add` and its own
`git commit` and never calls this wrapper, and is the door `automod_vault_land` goes
through (`agent_mcp/vault.py` has no commit site; the item named it wrongly at
filing). A guard on one of two write surfaces is not a guard (#2468), so the node that
walks this door lives beside the land tests in `tests/test_automod_vault_round.py`.

The ledger arm is deliberately narrow and its negative control is pinned below. Keying
coverage on a mention of the slug or the path shelters an orphan whose disposition is
itself in dispute: the live `lloyd/reviews/memory-md-retired.md` names the disputed
`replay-an-age-bearing-alarm-at-its-own-instant` path five times in CHECK COMMANDS and
remediation prose while its §5 file-survival claim is recorded FALSE at `:75`. Only a
`- **Topic file disposition:**` FIELD counts, and the fixture below plants the check
command with no field to prove the difference.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_UTILS = Path(__file__).resolve().parents[1] / "scripts" / "util"
_WRAPPER = _UTILS / "vault-commit.sh"
_MODULE = _UTILS / "orphan_topic_findings.py"


def _load_rung():
    # `scripts/util/` carries no `__init__.py`, so the import the two doors use is by
    # path — same as `tests/test_vault_commit_sidecar_refusal.py:103`.
    spec = importlib.util.spec_from_file_location("orphan_topic_findings", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OTF = _load_rung()

MEMORY_TEXT = """\
---
type: note
segment: lloyd
tags: [memory]
timestamp: '2026-11-01T00:00:00'
---
# Lloyd Long-Term Memory

## Key Knowledge

- [project] **The seeded line** — a rule that is loaded, so its file is reachable. → topics/seeded-rule
"""

SEEDED_TOPIC = "---\ntype: note\nsegment: lloyd\n---\n# topics/seeded-rule\n\nThe seeded detail.\n"

#: The live ledger's shape, planted with the disputed file named in a CHECK COMMAND and
#: with no disposition field. This is the row the triage measured at
#: `lloyd/reviews/memory-md-retired.md:68` and `:79`, and it is the negative control: a
#: rail that matched the path anywhere in the ledger would call this file dispositioned.
LEDGER_TEXT = """\
---
type: note
segment: lloyd
tags: [archive, memory-md, retired-lines]
timestamp: '2026-10-04T01:04:00'
---
# memory-md-retired

Archive of `lloyd/MEMORY.md` index lines retired by Step 2a-ter (§5).

## 2026-10-28 — nightly-reflection-knowledge-write (run 39, autonomy)

### Retired: `lloyd/MEMORY.md` `## Replaying an age-bearing alarm at its own fire instant`

- **Ledger row:** existed at `lloyd/memory/memory-md-ledger.md:33`. Deleted with the line, per §3.
- **Check command:** `test -e ~/obsidian/lloyd/memory/replay-an-age-bearing-alarm-at-its-own-instant.md && echo EXISTS || echo MISSING`
- **What moved:** the clause "`topics/replay-an-age-bearing-alarm-at-its-own-instant` stays on disk and untouched" is now FALSE of the tree.
"""

DISPUTED_SLUG = "replay-an-age-bearing-alarm-at-its-own-instant"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


@pytest.fixture()
def vault_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch vault holding the three paths this rail reads, in a base commit.

    `VAULT_DIR` is what the wrapper reads (`vault-commit.sh:129`
    `VAULT="${VAULT_DIR:-$HOME/obsidian}"`); `LLOYD_VAULT` is what `app.paths` reads.
    Both are pointed here — the matcher under test is a python module reached from the
    bash door, so an env var that moved only one of them would have the wrapper guard a
    vault this test is not asserting on, and guard the live one, of all things. Same
    pair as `tests/test_vault_commit_sidecar_refusal.py:139`.
    """
    repo = tmp_path / "vault"
    (repo / "lloyd" / "memory").mkdir(parents=True)
    (repo / "lloyd" / "reviews").mkdir(parents=True)
    monkeypatch.setenv("LLOYD_VAULT", str(repo))
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@e")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "lloyd" / "MEMORY.md").write_text(MEMORY_TEXT, encoding="utf-8")
    (repo / "lloyd" / "memory" / "seeded-rule.md").write_text(SEEDED_TOPIC,
                                                              encoding="utf-8")
    (repo / "lloyd" / "reviews" / "memory-md-retired.md").write_text(LEDGER_TEXT,
                                                                      encoding="utf-8")
    (repo / "seed.md").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-qm", "base").returncode == 0
    return repo


def _run_wrapper(vault: Path, msg: str, paths: list[str] | None = None,
                 job_id: str = "test: orphan topic") -> subprocess.CompletedProcess:
    env = dict(os.environ, LLOYD_VAULT=str(vault), VAULT_DIR=str(vault),
               LLOYD_PYTHON=sys.executable, LLOYD_JOB_ID=job_id)
    env.pop("LLOYD_JOB_WRITES", None)
    argv = ["bash", str(_WRAPPER), msg]
    if paths is not None:
        argv.append("--")
        argv.extend(paths)
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)


def _write(vault: Path, rel: str, text: str) -> str:
    """Create a file in the working tree and return its vault-relative path."""
    target = vault / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return rel


def _topic(slug: str, body: str = "The detail.") -> str:
    return (f"---\ntype: note\nsegment: lloyd\ntags: [memory]\n"
            f"timestamp: '2026-11-03T00:00:00'\n---\n# topics/{slug}\n\n{body}\n")


def _head_paths(vault: Path) -> list[str]:
    r = _git(vault, "show", "--name-only", "--format=", "HEAD")
    assert r.returncode == 0, r.stderr
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _tracked(vault: Path) -> list[str]:
    r = _git(vault, "ls-files")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


# ── clause 1: the refusal, and that it commits nothing ─────────────────────

def test_adding_an_unlinked_topic_file_is_refused_and_nothing_committed(vault_repo):
    """The acceptance check, re-run: the probe flips from committing to refusing.

    The scratch reproduction on 2026-10-29 staged a new `lloyd/memory/….md` with no
    link and no row, ran the wrapper, and got a COMMIT (`9b1a7d7`, HEAD advanced, the
    path in `git show --name-only HEAD`), printing only the #1867 unattributed-path
    notice. Both halves are asserted here: a non-zero exit that means nothing, and a
    refusal that is also a rollback.
    """
    v = vault_repo
    before = _git(v, "rev-parse", "HEAD").stdout.strip()
    path = _write(v, "lloyd/memory/triage-orphan-probe.md", _topic("triage-orphan-probe"))
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03")
    assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
    assert _git(v, "rev-parse", "HEAD").stdout.strip() == before, \
        "a refusal must not advance HEAD"
    assert path not in _tracked(v), "the refused path must not be in the index"
    assert path not in _head_paths(v), "the refused path must not be in the commit body"
    assert "ORPHAN TOPIC FILE" in r.stderr, r.stderr
    assert (v / path).exists(), "refusing a file into a commit is not deleting the writer's note"


def test_the_refusal_names_the_path_and_both_remedies_and_the_path_to_stage(vault_repo):
    """Clause 1's stderr and clause 3's second half, in one read of the same refusal.

    A message that said only "add the index line" would produce a second refused
    commit, because the failure this rail exists to catch is a line that is on disk and
    not staged. So the refusal has to name `lloyd/MEMORY.md` as something to STAGE in
    this commit, and name the ledger field for the writers whose content is being
    merged or retired rather than indexed (#3539's owed ruling: a retirement may not
    leave a file under `lloyd/memory/` with no index line).
    """
    v = vault_repo
    path = _write(v, "lloyd/memory/a-new-rule.md", _topic("a-new-rule"))
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03")
    err = r.stderr
    assert path in err, err
    assert "lloyd/MEMORY.md" in err, err
    assert "→ topics/a-new-rule" in err, err
    assert "lloyd/reviews/memory-md-retired.md" in err, err
    assert "Topic file disposition" in err, err
    assert "stage" in err.lower(), err
    assert "Nothing is unstaged and nothing is committed" in err, err
    # Both remedies, in the order the writer can act on them.
    assert "(a)" in err and "(b)" in err, err


def test_the_wrapper_shells_out_to_the_shared_matcher_and_restates_nothing(vault_repo):
    """The rule lives in one file, and the bash door only calls it.

    `vault-commit.sh` is bash and cannot import, so it runs the module; `land()` imports
    it (pinned in `tests/test_automod_vault_round.py`). Both are pinned to the same
    file, because if a copy ever appears, the two doors start disagreeing about what an
    orphan is — the #2468 failure with one more step, and the one this item exists to
    avoid repeating.
    """
    script = _WRAPPER.read_text(encoding="utf-8")
    assert "orphan_topic_findings.py" in script
    # The anchor is the exit-code TEST, not the path variable: the variable is assigned
    # one line above its own existence check, so a slice from there is a few lines of
    # `[ -f ]` boilerplate and says nothing about what the rung does with a refusal.
    # What matters is that the `-eq 1` branch ends in `exit 1` and not in the
    # `|| echo "… committing anyway"` every report-only rung in this file ends with.
    assert 'ORPHAN_TOPIC_RC" -eq 1' in script
    # 1600 chars is the slice that reaches past the unstage loop and the closing
    # refusal line; the window has to END on the `exit 1`, which is why it is measured
    # rather than the whole tail — a `committing anyway` from the next rung would then
    # be inside it and this node would fail for the wrong reason.
    _block = script.split('ORPHAN_TOPIC_RC" -eq 1')[1]
    assert "exit 1" in _block[:1600], "the rung must refuse, not merely report"
    assert 'committing anyway' not in _block[:_block.index("exit 1")]
    assert OTF.__file__ == str(_MODULE.resolve())
    assert OTF.TOPIC_DIR == "lloyd/memory"
    assert OTF.MEMORY_REL == "lloyd/MEMORY.md"
    assert OTF.LEDGER_REL == "lloyd/reviews/memory-md-retired.md"


# ── clause 2: the healthy same-commit pattern is untouched ─────────────────

def test_the_index_line_in_the_same_commit_commits(vault_repo):
    """17 of the 29 historical adds already do this; none of them may be refused.

    This is the satisfiability proof for the whole rail: the writer that hooks its file
    in the same commit is committed with exit 0 and both paths in the same commit body.
    """
    v = vault_repo
    path = _write(v, "lloyd/memory/a-new-rule.md", _topic("a-new-rule"))
    (v / "lloyd" / "MEMORY.md").write_text(
        MEMORY_TEXT + "\n- [project] **A new rule** — one line, one file. → topics/a-new-rule\n",
        encoding="utf-8")
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "ORPHAN TOPIC FILE" not in r.stderr, r.stderr
    committed = _head_paths(v)
    assert path in committed and "lloyd/MEMORY.md" in committed, committed


def test_the_ledger_files_under_memory_are_never_topic_files(vault_repo):
    """The `*-md-ledger.md` exclusion, at the door that writes ledgers.

    `scripts/memory/memory_ledger.py`'s `LEDGERS` map reads `lloyd/memory/memory-md-ledger.md`
    and `user-md-ledger.md` by name, and nothing links either from an index line — so a
    rail that keyed on `lloyd/memory/*.md` alone would refuse every ledger append,
    including the disposition append this rail's own remedy asks for.
    """
    v = vault_repo
    path = _write(v, "lloyd/memory/probe-md-ledger.md", "# probe ledger\n\n- anchor: x\n")
    r = _run_wrapper(v, "nightly: ledger append")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert path in _head_paths(v), _head_paths(v)


# ── clause 3: coverage is read from the STAGED blob ────────────────────────

def test_a_dirty_but_unstaged_index_line_does_not_suppress_the_refusal(vault_repo):
    """The hole this item was filed to close, pinned.

    The wrapper supports a scoped commit (#1867 `f92b87d9`, #2245 `238c3bc8`), so this
    invocation names only the topic file and the hook line stays dirty on disk — the
    commit that goes through would leave the file unlinked, which is precisely the state
    12 of the 29 historical adds shipped in. A check that read `lloyd/MEMORY.md` from
    the working tree would wave it through; the refusal below also has to say that the
    index file is the thing to stage, or the caller re-runs and is refused again.
    """
    v = vault_repo
    path = _write(v, "lloyd/memory/a-new-rule.md", _topic("a-new-rule"))
    (v / "lloyd" / "MEMORY.md").write_text(
        MEMORY_TEXT + "\n- [project] **A new rule** — one line, one file. → topics/a-new-rule\n",
        encoding="utf-8")
    before = _git(v, "rev-parse", "HEAD").stdout.strip()
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03", paths=[path])
    assert r.returncode != 0, (
        f"the line is on disk but not staged, so the commit would leave the orphan red\n"
        f"--- stderr ---\n{r.stderr}")
    assert _git(v, "rev-parse", "HEAD").stdout.strip() == before
    assert path not in _tracked(v)
    assert "lloyd/MEMORY.md" in r.stderr and "stage" in r.stderr.lower(), r.stderr


def test_the_ledger_staged_in_the_same_commit_is_what_counts(vault_repo):
    """The mirror of the node above for the ledger's coverage half.

    A disposition row staged in this commit disposes of the file, and the same row
    present only in the working tree does not decide anything about the commit's own
    content — the file it names is in this commit either way, so the ledger is read
    from the tree and the index is not. That asymmetry is deliberate and it is stated
    in `orphan_topic_findings._ledger_text`; this node is what keeps it from being
    quietly unified into one rule by a later reader.
    """
    v = vault_repo
    path = _write(v, "lloyd/memory/merged-into-another.md", _topic("merged-into-another"))
    ledger = v / "lloyd" / "reviews" / "memory-md-retired.md"
    ledger.write_text(
        LEDGER_TEXT + "\n### 2026-11-03 — merged, not indexed\n\n"
        "- **Topic file disposition:** `lloyd/memory/merged-into-another.md` — content "
        "merged into `lloyd/memory/seeded-rule.md`; the stub was never indexed.\n",
        encoding="utf-8")
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert path in _head_paths(v), _head_paths(v)


def test_a_check_command_naming_the_path_is_not_a_disposition(vault_repo):
    """The triage's warning, turned into a test.

    The live `lloyd/reviews/memory-md-retired.md` names
    `lloyd/memory/replay-an-age-bearing-alarm-at-its-own-instant.md` five times — as a
    check command (`test -e …/lloyd/memory/replay-….md`) and as remediation prose —
    while its §5 file-survival claim is recorded FALSE at `:75`. A rail that matched
    the path anywhere in the ledger would shelter the one file whose survival the ledger
    says nobody verified. The fixture plants exactly that row (`LEDGER_TEXT`), so this
    node fails the moment coverage becomes a substring match.
    """
    v = vault_repo
    assert DISPUTED_SLUG in LEDGER_TEXT and "lloyd/memory/" + DISPUTED_SLUG in LEDGER_TEXT
    assert OTF.dispositioned_slugs(LEDGER_TEXT) == set(), \
        "the planted row must NOT count as a disposition"
    path = _write(v, f"lloyd/memory/{DISPUTED_SLUG}.md", _topic(DISPUTED_SLUG))
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03")
    assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
    assert path in r.stderr, r.stderr


# ── clause 4: scope is the added path only; no ledger is no crash ─────────

def test_a_pre_existing_orphan_does_not_freeze_an_unrelated_commit(vault_repo):
    """The scope clause, which is what keeps this off the nightly-wide red #2415 withheld.

    An orphan is committed and tracked first, then a completely unrelated note lands.
    That second commit did not add anything under `lloyd/memory/`, so it is not its
    author's defect — and this wrapper also commits other writers' dirty state, so a
    whole-tree count here would strand every later job's snapshot exactly like #2245's
    ~20 minutes of stranded task #4 commits.
    """
    v = vault_repo
    orphan = _write(v, "lloyd/memory/someone-elses-orphan.md", _topic("someone-elses-orphan"))
    assert _run_wrapper(v, "sweep: another job's dirty state").returncode != 0
    # Commit it the way the rail permits: with its hook line staged alongside.
    (v / "lloyd" / "MEMORY.md").write_text(
        MEMORY_TEXT + f"\n- [project] **Someone else's rule.** → topics/{orphan.split('/')[-1][:-3]}\n",
        encoding="utf-8")
    assert _run_wrapper(v, "sweep: with its line").returncode == 0
    assert orphan in _head_paths(v)

    # Now the same tree, with the hook line REMOVED from the index while the topic file
    # stays tracked: a pre-existing orphan, and an unrelated commit that adds nothing
    # under lloyd/memory/ must still land beside it.
    (v / "lloyd" / "MEMORY.md").write_text(MEMORY_TEXT, encoding="utf-8")
    assert _run_wrapper(v, "sweep: unhook it").returncode == 0
    assert orphan in _tracked(v), "the topic file itself is never deleted by this rail"
    note = _write(v, "knowledge/unrelated-note.md",
             "---\ntype: reference\nsegment: knowledge\n---\n# note\n")
    r = _run_wrapper(v, "backlog: land an unrelated note")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "ORPHAN TOPIC FILE" not in r.stderr, r.stderr
    assert note in _head_paths(v), _head_paths(v)
    assert orphan in _tracked(v)


def test_an_absent_ledger_still_refuses_an_unlinked_add_without_crashing(vault_repo):
    """Clause 4's second arm: `absent or empty`, and it must not crash or silently skip.

    `lloyd/reviews/memory-md-retired.md` is created by the first retirement (§5), so a
    young vault has no such file at all. "Could not read the ledger" must never read as
    "nothing to check": the honest answer for an absent ledger is NO ROWS, which is the
    reading that refuses. A crash would be worse — the bash door treats an exit past 1
    as a broken instrument and commits whatever the index holds.
    """
    v = vault_repo
    (v / "lloyd" / "reviews" / "memory-md-retired.md").unlink()
    path = _write(v, "lloyd/memory/a-new-rule.md", _topic("a-new-rule"))
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03")
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "CHECK FAILED" not in r.stderr and "CHECK SKIPPED" not in r.stderr, r.stderr
    assert path in r.stderr and path not in _tracked(v), r.stderr


def test_an_empty_ledger_still_refuses_an_unlinked_add(vault_repo):
    """The same arm with the file present and empty — the other half of "absent or empty".

    A zero-byte ledger is what a run that opened the file and wrote nothing leaves
    behind, and it parses to no rows without an exception; the refusal has to come from
    the absence of a row, not from the absence of a file.
    """
    v = vault_repo
    (v / "lloyd" / "reviews" / "memory-md-retired.md").write_text("", encoding="utf-8")
    path = _write(v, "lloyd/memory/a-new-rule.md", _topic("a-new-rule"))
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03")
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "CHECK FAILED" not in r.stderr, r.stderr
    assert path in r.stderr, r.stderr


def test_a_modification_of_a_tracked_topic_file_is_not_an_add(vault_repo):
    """`--diff-filter=A` is doing real work, and `M` must not be swept in with it.

    Editing the body of a file HEAD already holds is the nightly's most common vault
    write. Refusing it would be the rail refusing a healthy commit — the owed-check's
    measured guard against exactly that ("a materially higher refusal rate than ~12 in
    29 means the rail is refusing healthy commits") starts from this arm.
    """
    v = vault_repo
    tracked = "lloyd/memory/seeded-rule.md"          # tracked, and linked at MEMORY.md
    assert tracked in _tracked(v) and tracked not in _head_paths(v)[-1:]
    (v / tracked).write_text(SEEDED_TOPIC + "\nA later run appended a worked case.\n",
                             encoding="utf-8")
    r = _run_wrapper(v, "nightly: knowledge write 2026-11-03")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert tracked in _head_paths(v), _head_paths(v)


# ── the module's own contract, at the seam the bash door reads ─────────────

def test_a_broken_check_exits_three_and_is_never_laundered_into_a_clean_zero(tmp_path):
    """Exit 3 is the family's "I could not look", and it is not exit 0.

    `--vault` at a plain directory is the instrument with no git under it. The bash door
    commits anyway on an exit past 1 and SAYS so; what it must never get is a 0, because
    a check that goes blind looks exactly like a vault with no orphans — the blindness
    that made `status`'s printed count and the validator's report-only arm insufficient.
    """
    not_a_repo = tmp_path / "not-a-vault"
    not_a_repo.mkdir()
    r = subprocess.run([sys.executable, str(_MODULE), "--vault", str(not_a_repo)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert "not a clean read" in r.stderr, r.stderr


def test_the_module_reports_findings_as_json_for_a_nonbash_caller(tmp_path):
    """`--json` is the shape the land door's dict reads, so its keys are pinned.

    `land()`'s ledger row keys `orphan_topic_refusals` off `rec["path"]`; a renamed key
    there would be a ledger row that names nothing, so the field names are asserted at
    the module rather than only through the two doors.
    """
    v = tmp_path / "v"
    (v / "lloyd" / "memory").mkdir(parents=True)
    _git(v, "init", "-q", "-b", "main")
    _write(v, "lloyd/memory/a-new-rule.md", _topic("a-new-rule"))
    _write(v, "lloyd/MEMORY.md", MEMORY_TEXT)
    _git(v, "add", "-A")
    r = subprocess.run([sys.executable, str(_MODULE), "--vault", str(v), "--json"],
                       capture_output=True, text=True, timeout=60)
    # No HEAD yet: the whole index is what this commit adds, so the rail still refuses.
    # The empty-set answer the sibling module gives for an unborn HEAD would be a
    # fail-open here, and an initial vault commit most often adds exactly a topic file.
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    rows = json.loads(r.stdout)
    assert [row["path"] for row in rows] == ["lloyd/memory/a-new-rule.md"], rows
    assert rows[0]["slug"] == "a-new-rule"
    assert rows[0]["memory_md"] == "lloyd/MEMORY.md"
    assert rows[0]["ledger"] == "lloyd/reviews/memory-md-retired.md"
    assert rows[0]["field"] == "topic file disposition"
