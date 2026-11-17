"""#4118: the k-sweep doc's archived copies, asked about where they are claimed.

`eval/measurements/context-rot-ksweep-2026-11-08.md:3` names the bytes its tables were
computed from and the vault copy that is supposed to outlive them. That copy did not
exist: `find ~/obsidian -iname '*ksweep*'` returned only backlog items, and the sole copy
of the run's bytes was the runtime directory `/home/alansrobotlab/lloyd-data/ksweep-2026-11-08/`,
which is gitignored and named by no groundskeeper store — nothing prunes it and nothing
preserves it, so the cliff verdict and the hedge rows were recomputable only while a
directory nobody manages happened to stand.

Nothing could see that rot. `eval/check_doc_matches_json.py` proves the doc equals the
JSON and reads only the artifact, the doc and the cliff sample (`git grep -in
"archiv\\|obsidian\\|backup" eval/check_doc_matches_json.py` is empty), and
`tests/test_context_rot_eval.py` pins the `Artifact` line's *text* — the vault path is one
of its assertions, at :948 — without ever asking the vault whether the file is where the
line says. So the documented invocation exited 0 on a doc whose stated second home was
absent. This file asks the vault.

One node per acceptance clause, plus the seam:

| clause | node |
|---|---|
| 1 the artifact's archived copy exists with the artifact's own bytes | `test_every_archived_copy_the_doc_names_holds_those_bytes` |
| 2 so does the cliff sample's, giving §How to re-run's "the archived copies" both referents | the same node, and `test_the_archived_copies_alone_re_run_the_documented_check` |
| 3 one `Artifact` line naming both by vault path | `test_the_doc_names_both_archived_copies_by_vault_path` |
| #4204 1 the repo-history ruling reads on that line as closed, not owed | `test_the_repo_history_ruling_is_recorded_as_closed_not_owed` |
| #4204 2 no `owed on #<n>` pointer survives in the doc, and none to #3989 anywhere in `eval/` | the same node, through `owed_pointers_in_eval` |

Clause 3's *retained* half — the runtime path, `e463040b…`, `2d2aaba3` and "No copy of the
JSON is ever added to this repo" still on that line, and the checker command line still
unchanged — stays pinned by the two nodes that already own it, untouched by this diff, for
#4118's clause 3 and for #4204's clauses 3 and 4 alike:
`tests/test_context_rot_eval.py::test_the_doc_names_where_the_bytes_live_and_promises_no_repo_copy`
and `::test_the_doc_names_the_checker_by_a_path_that_resolves_from_the_repo_root`.

Why these nodes are NOT marked `live_vault`: the reason `tests/test_archived_skill_artifacts.py`
gives under the same heading, for the same reason. A marked vault assertion is deselected
by the gate's own `-m "not live_vault"`, certifies nothing, and "nothing enforces it" is
precisely the failure mode this item was filed on. The coupling that buys is cheap here
because the bytes are immutable history: two `sha256sum`-pinned extracts, written once,
named by no writer, edited by no nightly job. The vault root comes from
`board_presence.vault_root()` (`LLOYD_OBSIDIAN_VAULT`, else `~/obsidian`), read per call —
never from `app.paths`, which re-anchors to a round's worktree, where these paths do not
exist and a check built on them would read nothing and pass.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import board_presence

ROOT = Path(__file__).resolve().parents[1]
KS_DOC = ROOT / "eval" / "measurements" / "context-rot-ksweep-2026-11-08.md"
CHECKER = ROOT / "eval" / "check_doc_matches_json.py"

#: The two byte-sets the `Artifact` line claims homes for: keyed by the archived copy
#: relative to the vault root, valued by the `sha256sum` its runtime source printed at
#: #4118's triage and re-read on 2026-11-14 at the head this round was cut from. Equality
#: with these constants IS clause 1's "byte-identical to the runtime artifact rather than
#: re-typed" — the hash is of the runtime file's bytes, so a copy that re-serialised the
#: JSON (one space, one key order, one reformatted float) fails here instead of quietly
#: becoming a second, slightly different artifact.
ARCHIVED = {
    "backlog/data/2026-11-09.3923-context-rot-ksweep-2026-11-08.json":
        "e463040b404ed7b743dc55e3595feb9a0a053656d776e5f89f1cb2767980459f",
    "backlog/data/2026-11-09.3923-conflicts-iv.json":
        "fc864503a74f8b6186ad154e673aff74dc296d1ab1a2f48f827a1bc77fe6a6f7",
}

#: The name the checker's `conflicts_sample_path()` looks for under
#: `$LLOYD_DATA/ksweep-2026-11-08/`, so the third node can stage the archived copy where
#: the tool will find it instead of the runtime file.
IV_NAME = "conflicts-iv.json"

#: The corpus the acceptance's absence check walks: `git grep ... -- eval/` reads every
#: tracked file under `eval/`, and this is that corpus off the round's own tree — see
#: `owed_pointers_in_eval()` for why the walk stands in for the literal `git grep`.
EVAL_CORPUS = ROOT / "eval"

#: An owned pointer to a CLOSED item is how a settled question gets re-opened: the next reader
#: sees a number, visits it, finds `status: done`, and either re-litigates the ruling or
#: re-proposes the edit the ruling refused. `tests/test_research_doc_claims.py:1792` and
#: `tests/test_architecture_coverage_doc_claims.py` each pin this shape for one corpus — a
#: research page and `architecture/` — and neither walks `eval/measurements/`, which is why
#: the clause below stood from 2026-11-09 to 2026-11-16 unchallenged. This is that corpus's
#: pin. #3989, the item the clause named, is `status: done` (completed 2026-11-09T08:33:13,
#: landed `d0b2011c`), and the ruling it called "owed" is recorded on #3923's `owed_settled`
#: as outcome `ruling` on 2026-11-16: NO `.gitignore` negation and no repo copy of the raw
#: JSON, because the vault placement retires the question.
OWED_POINTER = re.compile(r"owed on #\d+")

#: The exact clause the `Artifact` line carried until #4204 retired it, verbatim. The node
#: below asserts the pattern against THIS text before it trusts the pattern's absence from the
#: doc, the way `RETIRED_RESTORE_OWNERSHIP` is asserted at
#: `tests/test_research_doc_claims.py:1846`: an absence proved by a pattern that never matched
#: anything is not evidence, it is the guard wearing the bug's clothes.
RETIRED_OWED_POINTER = (
    "whether the raw file should get repo history anyway is a ruling owed on #3989, "
    "not a step this doc skipped")

#: The sentence that replaces it, asserted byte-for-byte so the record cannot be softened by
#: a later reword: the date and the holder make a subsequent owed-check pass discharge the
#: question instead of re-deciding it, "no `.gitignore` negation and no repo copy" names the
#: declined outcome so the next round cannot propose it again, and the whole-history clause
#: states what the bytes' provenance IS rather than what it lacks — the claim
#: `test_every_archived_copy_the_doc_names_holds_those_bytes` above keeps true of the copies,
#: and `test_the_archived_copies_alone_re_run_the_documented_check` keeps true with the runtime
#: directory gone. This node cannot re-measure the runtime bytes itself: `conftest` re-anchors
#: `LLOYD_DATA` to a scratch directory for every test, so an env-derived path inside the suite
#: reads a `/tmp` tree that never held them — which is why the runtime pair is re-run by the
#: documented command from the checkout, not from a node.
RULED_CLOSED = (
    "Whether the raw file should get repo history at all is **RULED closed 2026-11-16 by "
    "owed-check**: no `.gitignore` negation and no repo copy \u2014 the runtime bytes plus "
    "the two vault copies named on this line are the artifact's and the cliff sample's "
    "whole history.")

#: What the checker recomputes from these exact bytes: 96 cells is 2 shapes x 2 arms x 2
#: lengths x 4 ks x 3 seeds (the grid the doc states above its headline table), and the 8
#: curve rows are one per (shape, arm, length). Pinned so the subprocess below cannot
#: exit 0 having read nothing.
DOC_CELLS, DOC_CURVE_ROWS = 96, 8


def artifact_line() -> str:
    """The doc's single `Artifact` line: one line, so a stale home cannot survive beside
    a live one — the shape `tests/test_context_rot_eval.py:941` already pins."""
    lines = [ln for ln in KS_DOC.read_text(encoding="utf-8").splitlines()
             if ln.startswith("Artifact")]
    assert len(lines) == 1, (
        f"{KS_DOC.name}: expected exactly one line starting 'Artifact', found {len(lines)}")
    return lines[0]


def named_archives() -> list[str]:
    """The `backlog/data/*.json` paths that line names under `~/obsidian`, in order."""
    return re.findall(r"~/obsidian/(backlog/data/[^\s`;)]+\.json)", artifact_line())


def archived(rel: str) -> Path:
    """One archived copy under the vault root the doc points at — or a failure that names
    the path, so a missing home reads as a broken promise and not as a stack trace."""
    p = board_presence.vault_root() / rel
    assert p.is_file(), (
        f"{p} is named as an archived home by {KS_DOC.relative_to(ROOT)}:3 and is not there")
    return p


def test_the_doc_names_both_archived_copies_by_vault_path():
    """Clause 3. §How to re-run closes with "The archived copies named in the artifact
    line above are where those bytes outlive the runtime directory" — plural, and about
    both byte-sets the checker reads. One referent made that sentence true of the
    artifact and silently false of the cliff sample, which had no named home at all."""
    named = named_archives()
    assert sorted(named) == sorted(ARCHIVED), (
        "the Artifact line has to name both archived copies by vault path — the "
        f"artifact's and the cliff sample's — and it names {named}")


def test_every_archived_copy_the_doc_names_holds_those_bytes():
    """Clauses 1 and 2. A path in a doc is a promise about a file; this is the check that
    the promise holds, in the vault the line points at rather than the runtime directory
    it is meant to survive."""
    unnamed = set(named_archives()) - set(ARCHIVED)
    assert not unnamed, (
        f"the Artifact line names an archived copy no sha is pinned for: {sorted(unnamed)}")
    for rel in sorted(ARCHIVED):
        p = archived(rel)
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == ARCHIVED[rel], f"{p} holds {got}, not the artifact's {ARCHIVED[rel]}"


def test_the_archived_copies_alone_re_run_the_documented_check(tmp_path):
    """The acceptance is that the cliff verdict and the hedge rows stay recomputable once
    `$LLOYD_DATA/ksweep-2026-11-08/` is pruned, so this prunes it: the archived artifact
    goes to the checker as argv[1] and the archived cliff sample is staged under a scratch
    `LLOYD_DATA`, with no runtime directory in the environment at all. The vault copies
    have to be a home, not a citation.

    It is also the only node that crosses the repo↔vault boundary by *using* those bytes:
    the checker is a subprocess whose third input it resolves itself (`conflicts_sample_path()`
    has no inbound edge in the code graph because nothing in-process ever calls it), so
    until now nothing in the suite ever opened an archived copy of either file.
    """
    staged = tmp_path / "lloyd-data" / "ksweep-2026-11-08"
    staged.mkdir(parents=True)
    (staged / IV_NAME).write_bytes(archived("backlog/data/2026-11-09.3923-conflicts-iv.json")
                                   .read_bytes())

    p = subprocess.run(
        [sys.executable, str(CHECKER),
         str(archived("backlog/data/2026-11-09.3923-context-rot-ksweep-2026-11-08.json")),
         str(KS_DOC.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env={**os.environ, "LLOYD_DATA": str(tmp_path / "lloyd-data")})

    assert p.returncode == 0, f"{p.stdout[-1200:]}\n{p.stderr[-1200:]}"
    report = json.loads(p.stdout)
    assert report["ok"] is True and report["sample_read"] is True, report["failures"]
    assert report["conflicts_sample"] == str(staged / IV_NAME), report["conflicts_sample"]
    assert (report["cells"], report["curve_rows"]) == (DOC_CELLS, DOC_CURVE_ROWS), report


def owed_pointers_in_eval(pattern: "re.Pattern[str]" = OWED_POINTER,
                          root: Path = EVAL_CORPUS) -> list[str]:
    """Every line under `root` matching `pattern`, as `<path>:<line>`.

    The acceptance states this as `git grep -n 'owed on #3989' -- eval/` — same corpus, same
    pattern — but walked rather than shelled out. Two reasons, both learned in this repo.
    `subprocess` is on this suite's ban list for what a test may spawn
    (`tests/test_context_rot_eval.py:12-16`, rule `R-024`), and git plumbing is the mechanism
    that sank the previous round on this very doc: its `git add` died in the gate's candidate
    tree and left the doc edit out of the commit while the tests still passed. A walk of the
    round's own tree needs no gitdir to be honest about what it saw, and the node proves the
    walker fires — on a seeded line, and against the real corpus — before trusting an empty
    answer from it.
    """
    hits: list[str] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        try:
            body = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, ln in enumerate(body.splitlines(), 1):
            if pattern.search(ln):
                hits.append(f"{f}:{n}")
    return hits


def test_the_repo_history_ruling_is_recorded_as_closed_not_owed(tmp_path):
    """Clauses 1 and 2, on the doc's own text. The order is the point: the pattern is shown
    firing on the retired clause and the walker on a seeded line FIRST, then that clause and
    every pointer to #3989 are shown absent, then the ruling is shown present in the
    affirmative. An absence proved by an instrument that was never seen to work is not
    evidence, and a doc that had merely deleted the sentence without recording the decision
    would be the other way this rot is made. Both are refused here.
    """
    assert OWED_POINTER.search(RETIRED_OWED_POINTER), (
        "the fixture no longer carries the pointer this guard exists to catch, so its absence "
        "from the doc would prove nothing about anyone else's owed pointer")

    seed = tmp_path / "seed.md"
    seed.write_text(f"prose {RETIRED_OWED_POINTER}.\n", encoding="utf-8")
    assert owed_pointers_in_eval(root=tmp_path) == [f"{seed}:1"], (
        "the corpus walker does not fire on the very text it is meant to catch, so its empty "
        "answer over eval/ would be an artifact of the walker, not a clean tree")

    text = KS_DOC.read_text(encoding="utf-8")
    assert RETIRED_OWED_POINTER not in text, (
        "the `Artifact` line still points repo history at a ruling 'owed' on #3989, which is "
        "status: done — a closed item named as the owner of a settled question invites a later "
        "round to re-open it and re-propose the ignored-path edit the gate refuses")
    assert OWED_POINTER.findall(text) == [], (
        f"{KS_DOC.name} still names a numbered item as the owner of something owed "
        f"{OWED_POINTER.findall(text)}: record the ruling and its date, never an item that may "
        "since have closed")

    assert RULED_CLOSED in artifact_line(), (
        "clause 1's closed ruling is not stated in the affirmative on the `Artifact` line; "
        f"that line's tail reads {artifact_line()[-300:]!r}")

    for keep in ("/home/alansrobotlab/lloyd-data/ksweep-2026-11-08/context-rot-ksweep-2026-11-08.json",
                 "e463040b404ed7b743dc55e3595feb9a0a053656d776e5f89f1cb2767980459f",
                 "fc864503a74f8b6186ad154e673aff74dc296d1ab1a2f48f827a1bc77fe6a6f7",
                 "backlog/data/2026-11-09.3923-context-rot-ksweep-2026-11-08.json",
                 "backlog/data/2026-11-09.3923-conflicts-iv.json",
                 "`2d2aaba3`",
                 "No copy of the JSON is ever added to this repo"):
        assert keep in artifact_line(), (
            f"the retained half of the `Artifact` line lost {keep!r}, which clause 3 requires "
            "to stay on that line")

    # The acceptance's own check, over the corpus `-- eval/` names. Scoped to the ruled item:
    # two other lines under `eval/` carry `owed on #3419` / `owed on #1676` to OPEN items, and
    # those are other rounds' records to discharge, not this diff's to edit.
    corpus = [f for f in EVAL_CORPUS.rglob("*") if f.is_file()]
    assert corpus and KS_DOC in corpus, (
        f"the corpus is not the one the check is about: {len(corpus)} files under {EVAL_CORPUS}")
    assert owed_pointers_in_eval(re.compile(r"owed on #3989")) == [], (
        "`git grep -n 'owed on #3989' -- eval/` is not empty: "
        f"{owed_pointers_in_eval(re.compile(r'owed on #3989'))}")
