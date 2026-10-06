"""#1869: a measurement witness that lives in no history gets an archive clause.

A measurement item's claim is a number read off a live store — rows of a jsonl,
a count out of a db — and those paths sit under the data root, where
`git rev-parse --show-toplevel` exits 128 (`fatal: not a git repository`) while
`~/lloyd` and `~/obsidian` each report a toplevel. The bytes a claim was read
off therefore have no history anywhere: #1621's 44-row report sat on one disk
until #1756 committed it to `backlog/data/…jsonl` at `5c5e6826`. #1756's owed
ruling (2026-09-30T02:14:09, run `20260929_190733_owedcheck_16ba`) settled the
policy — a standing rule on the item-authoring side, not the review prompt — so
the rule is a function on the path that writes a confirmed contract
(`add_witness_artifact_clause`, called from `record_verdict` after
`cap_new_clauses`), not a prompt bullet: four of this item's five checks are
behaviour of the emitted clause list.

Magnitude at filing: 6 open items under `~/obsidian/backlog/*.md` cite an
out-of-tree bytes path and 0 of them name a vault artifact, so the rule is
prospective. Its stated limit holds too: the trigger is a probe on a path the
item NAMES, so an item quoting a store's numbers with no path at all is
unreachable by it.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, cluster as CL, state as S


def write_item(d: Path, item_id, *, status="draft", days_old=100, tags=("backlog",),
               name=None, body="Do the thing.", extra=None) -> Path:
    name = name or f"Item {item_id}"
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": "lloyd", "tags": list(tags), **(extra or {})}
    p = d / f"{item_id}-{name.lower().replace(' ', '-')[:30]}.md"
    p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
                 encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(CL, "CLUSTERS_PATH", tmp_path / "clusters.json")
    return d


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text())[0]


def _witness(tmp_path: Path, name: str = "iv-metrics.jsonl") -> Path:
    """A witness where a witness lives on the live box: under the data root,
    in a directory no git tree covers."""
    d = tmp_path / "lloyd-data" / "_pipeline" / "reflection"
    d.mkdir(parents=True)
    p = d / name
    p.write_text('{"llm_calls": 164, "miss_rate": null}\n' * 17)
    return p


def _git_repo(tmp_path: Path) -> Path:
    """A real repository, so 'inside a tree' is the probe's own answer and not
    this file's assertion about a path string."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / "data").mkdir()
    (repo / "data" / "report.jsonl").write_text('{"rows": 17}\n')
    return repo


def _git_commit(repo: Path, *rel_paths: str) -> None:
    """Put `rel_paths` under history for real, so a test may call them tracked.

    `in_git_tree` answers about a repository's WORKING TREE (`git rev-parse
    --show-toplevel`), which a bare `git init` plus one written file already
    satisfies — so a fixture that calls its copy "committed" while only writing
    it has pinned less than its prose claims. The witness rule's premise is that
    no history covers the bytes, so the in-tree copy here is committed and the
    caller asserts it with `git ls-files`.
    """
    subprocess.run(["git", "-C", str(repo), "add", "--", *rel_paths],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo),
                    "-c", "user.email=tests@lloyd.local", "-c", "user.name=Lloyd Tests",
                    "commit", "-q", "-m", "test fixture: pin the witness bytes"],
                   check=True, capture_output=True)


# The item as triage writes it: the witness path is in the body, and no clause
# says where its bytes go.
_BODY_1869 = ("Live run 2026-09-30 over `{witness}` printed 17 rows with "
              "`miss_rate` null in 9. The change is display-only.")
_AUTHORED = ["the printed count is pinned by a test — tests/test_x.py"]


def test_a_witness_in_no_git_tree_gains_one_clause_with_a_path_and_a_command(tmp_path):
    """Clause 1: the added clause carries both halves — where the bytes go, and
    how to recompute the quoted report from the committed bytes.

    The default probe is the production one, run against a real directory that
    no repository covers, so the trigger is measured rather than injected.
    """
    witness = _witness(tmp_path)
    assert B.in_git_tree(witness) is False, "fixture must sit outside every tree"
    assert B.in_git_tree(Path(__file__)) is True, "the probe works at all"

    before = list(_AUTHORED)
    out = B.add_witness_artifact_clause(_AUTHORED, _BODY_1869.format(witness=witness))

    assert len(out) == 2, f"exactly one clause is added, got {out}"
    assert out[0] == before[0], "the authored clause is kept first, untouched"
    assert _AUTHORED == before, "the caller's list is never mutated in place"
    added = out[1]
    artifact = f"{B.WITNESS_ARTIFACT_DIR}/{witness.name}"
    assert artifact == "backlog/data/iv-metrics.jsonl", artifact
    assert artifact in added, added
    assert f"`wc -l -c < {artifact}`" in added, (
        f"the clause must carry a shell command over the COMMITTED bytes, not a "
        f"promise to re-measure, and it must be the byte-varying form (#2267): "
        f"{added}")
    assert str(witness) in added, "and it names the witness it is archiving"
    assert len(added) <= B.CLAUSE_MAX_CHARS, "it survives clean_clauses intact"


def test_evidence_that_all_sits_in_a_git_tree_returns_the_list_unchanged(tmp_path):
    """Clause 2: nothing is invented when the bytes already have a history.

    The paths are inside a real `git init`'d repo and the real probe answers,
    so an unchanged list is the rule declining to fire, not the probe failing
    to find anything.
    """
    repo = _git_repo(tmp_path)
    in_tree = [repo / "data" / "report.jsonl", repo / "notes.txt", Path(__file__)]
    for p in in_tree:
        assert B.in_git_tree(p) is True, f"{p} is inside a tree"

    text = "Measured over " + ", ".join(f"`{p}`" for p in in_tree) + ": 17 rows."
    out = B.add_witness_artifact_clause(_AUTHORED, text)

    assert out == _AUTHORED, out
    assert not any("backlog/data/" in c for c in out), out


def test_the_trigger_is_the_probe_on_the_witness_not_the_data_root_string(tmp_path):
    """Clause 3: the same path decides differently when the probe does.

    A `~/lloyd-data` path with an injected probe that reports "in a tree" adds
    nothing, and a `tmp_path` witness — which really has no tree — adds the
    clause only when the probe says so. Both directions are needed: the first
    says the rule is not a substring match on the data root, the second says
    it is not keyed to one machine's layout either.
    """
    data_root_witness = "~/lloyd-data/_pipeline/reflection/iv-metrics.jsonl"
    tmp_witness = _witness(tmp_path, "usage-report.jsonl")

    said_in_a_tree = B.add_witness_artifact_clause(
        _AUTHORED, _BODY_1869.format(witness=data_root_witness),
        probe=lambda path: True)
    assert said_in_a_tree == _AUTHORED, (
        f"a path under the data root alone must not fire the rule: {said_in_a_tree}")

    said_no_tree = B.add_witness_artifact_clause(
        _AUTHORED, _BODY_1869.format(witness=tmp_witness), probe=lambda path: False)
    assert len(said_no_tree) == 2, said_no_tree
    assert "backlog/data/usage-report.jsonl" in said_no_tree[1]

    but_in_one = B.add_witness_artifact_clause(
        _AUTHORED, _BODY_1869.format(witness=tmp_witness), probe=lambda path: True)
    assert but_in_one == _AUTHORED, (
        "the probe is the whole trigger; the path's shape decides nothing")


def test_the_real_probe_follows_a_tilde_witness_to_whatever_tree_home_is(tmp_path,
                                                                         monkeypatch):
    """The same clause-3 property with the production probe instead of an
    injected one, so nothing about the trigger is only true of a fake.

    `~` is expanded when the witness is classified, so pointing `HOME` at a
    throwaway directory decides both ways with the shipped `in_git_tree`: an
    out-of-tree home gains the clause, a home that is itself a git repository
    does not. A rule keyed on the literal string `~/lloyd-data` cannot make
    those two differ, since the text is identical.

    Since #1889 the emitter first stats the bytes it would ask a round to copy
    (clause 3 there: existence is checked before a `probe() == False` is read as
    "no history"), so the witness file is written under BOTH homes here. Without
    those bytes the second half would pass for the wrong reason — a missing file
    instead of a tree that has history.
    """
    tilde = "~/lloyd-data/_pipeline/reflection/iv-metrics.jsonl"
    body = _BODY_1869.format(witness=tilde)
    home = tmp_path / "home"
    (home / "lloyd-data" / "_pipeline" / "reflection").mkdir(parents=True)
    (home / "lloyd-data" / "_pipeline" / "reflection"
     / "iv-metrics.jsonl").write_text('{"llm_calls": 164, "miss_rate": null}\n' * 17)
    monkeypatch.setenv("HOME", str(home))

    out = B.add_witness_artifact_clause(_AUTHORED, body)
    assert len(out) == 2 and "backlog/data/iv-metrics.jsonl" in out[1], out

    (tmp_path / "vh").mkdir()
    versioned_home = _git_repo(tmp_path / "vh")
    (versioned_home / "lloyd-data" / "_pipeline" / "reflection").mkdir(parents=True)
    (versioned_home / "lloyd-data" / "_pipeline" / "reflection"
     / "iv-metrics.jsonl").write_text('{"llm_calls": 164, "miss_rate": null}\n' * 17)
    monkeypatch.setenv("HOME", str(versioned_home))
    assert B._witness_target(tilde).is_file(), (
        "the bytes exist here too, so only the probe can be what stops the clause")
    assert B.add_witness_artifact_clause(_AUTHORED, body) == _AUTHORED, (
        "the same path text under a home that is a tree must not fire")


#: #2248's real shape: the sweep's row lives in the data root, and a canonical
#: extract of the SAME basename is already committed in the repo's fixtures.
#: Both facts were in the item, and the generator asked for a third copy anyway.
_PREFIX_MISS = "vllm_prefix_miss_2026-10-01.json"
_IN_TREE_DIR = "tests/fixtures"


def _prefix_miss_shape(tmp_path, monkeypatch):
    """Lay out #2248's shape on disk and return the two path strings.

    The in-tree half is a fixture committed inside a real `git init` repository
    (`_git_commit` runs `git add`/`git commit`, and the callers assert it through
    `git ls-files`), so the default `probe=in_git_tree` is the shipped predicate
    answering about a real tree and the word "committed" here means under
    history. Committing is not what makes the probe answer — it reads the working
    tree — but the rule's premise is that no HISTORY covers the witness bytes, so
    a fixture that calls its pinned copy committed has to have committed it. The
    out-of-tree half is under `$HOME/lloyd-data`, which is not a tree, and both
    files exist so only the basename comparison can be what stops the clause.
    """
    home = tmp_path / "home"
    live = home / "lloyd-data" / "vllm-prefix-miss"
    live.mkdir(parents=True)
    (live / _PREFIX_MISS).write_text('{"date": "2026-10-01", "miss_rate": 0.063}\n')
    monkeypatch.setenv("HOME", str(home))
    repo = _git_repo(tmp_path)
    (repo / _IN_TREE_DIR).mkdir(parents=True)
    committed = repo / _IN_TREE_DIR / _PREFIX_MISS
    committed.write_text('{"date": "2026-10-01", "miss_rate": 0.063}\n')
    _git_commit(repo, f"{_IN_TREE_DIR}/{_PREFIX_MISS}")
    return repo, str(committed), f"~/lloyd-data/vllm-prefix-miss/{_PREFIX_MISS}"


def _is_tracked(repo: Path, rel: str) -> bool:
    """True when `git ls-files` in `repo` lists `rel` — the word 'tracked'."""
    return rel in subprocess.run(
        ["git", "-C", str(repo), "ls-files"], check=True,
        capture_output=True, text=True).stdout.split()


#: A basename no checkout on this box can contain, so the temp repository below
#: is the ONLY tree that can supply the pin. The #2248 fixture name cannot carry
#: that claim: `_pinned_candidates` tries a relative citation against the CWD and
#: then against the repository root of the module (backlog.py:1245), so inside a
#: real checkout `tests/fixtures/vllm_prefix_miss_2026-10-01.json` also resolves
#: to this repo's own committed fixture, and the skip would fire with or without
#: the temp tree. The control node below uses this name instead, which is what
#: makes the temp tree load-bearing.
_CONTROL_PIN = "witness_pin_control.jsonl"


def test_the_temporary_tree_is_what_pins_the_bytes_a_control_shows(tmp_path, monkeypatch):
    """The temp repo is load-bearing: with a basename no other tree holds.

    Same skip, same default probe, one difference — `witness_pin_control.jsonl`
    exists in the temp repository and nowhere else on the fallback path, so if the
    clause still does not fire then the accepted in-tree path was the temp repo's,
    not the checkout the suite happens to run inside. Without this node the #2248
    shape below would pass in a checkout that carries its own fixture of that name
    and fail in one that does not, while reading as though it had proved the same
    thing both times.
    """
    live_root = Path(B.__file__).resolve().parents[2]
    assert not (live_root / _IN_TREE_DIR / _CONTROL_PIN).exists(), (
        f"positive control: {live_root} must not hold the control basename, or the "
        "fallback candidate, not the temp repo, would be supplying the pin")
    home = tmp_path / "home"
    (home / "lloyd-data" / "_pipeline" / "reflection").mkdir(parents=True)
    (home / "lloyd-data" / "_pipeline" / "reflection" / _CONTROL_PIN).write_text(
        '{"rows": 17}\n' * 3)
    monkeypatch.setenv("HOME", str(home))
    repo = _git_repo(tmp_path)                    # holds data/report.jsonl only
    (repo / _IN_TREE_DIR).mkdir(parents=True)
    pin = repo / _IN_TREE_DIR / _CONTROL_PIN
    pin.write_text('{"rows": 17}\n' * 3)
    _git_commit(repo, f"{_IN_TREE_DIR}/{_CONTROL_PIN}")
    monkeypatch.chdir(repo)
    assert _is_tracked(repo, f"{_IN_TREE_DIR}/{_CONTROL_PIN}")

    clauses = [f"the sweep's real row is in {_IN_TREE_DIR}/{_CONTROL_PIN}"]
    body = _BODY_1869.format(witness=f"~/lloyd-data/_pipeline/reflection/{_CONTROL_PIN}")
    assert B._tree_basenames("\n".join([*clauses, body]), B.in_git_tree) == {_CONTROL_PIN}, (
        "the control basename must be pinned, and by exactly one path")
    assert B.add_witness_artifact_clause(clauses, body) == clauses, (
        "the temp tree's committed copy is what suppressed the clause")
    # And the same witness with the pin removed from the haystack does fire, so
    # the suppression above is the pin's doing rather than the witness's shape.
    assert len(B.add_witness_artifact_clause(_AUTHORED, body)) == 2, (
        "drop the in-tree citation and the very same bytes gain a clause")


def test_a_basename_already_pinned_inside_a_git_tree_gains_no_clause(tmp_path,
                                                                     monkeypatch):
    """#2267 clause 1: bytes pinned by a tracked path are not a missing witness.

    The generator's own premise is "no history covers these bytes". Once a clause
    or the item text names a path `in_git_tree` accepts whose basename is the
    candidate's, that premise is false for this witness, and the clause ordering a
    copy asks a round to add a second, competing home for the same figures — which
    is what #2248 survived only because the grader marked the demand
    `unsatisfiable` and an amendment was ratified.

    Which tree answers here is #2248's real ambiguity, not this file's: the
    relative citation resolves against the CWD (the temp repo, chdir'd below) and,
    through `_pinned_candidates`' second candidate, against the checkout the suite
    runs in — which has its own committed `tests/fixtures/vllm_prefix_miss_*.json`.
    Both are committed copies, so the claim below holds either way. The node that
    isolates the temp tree, by naming a basename no other tree holds, is
    `test_the_temporary_tree_is_what_pins_the_bytes_a_control_shows` above, and it
    is what stops this one from passing for a reason it does not name.

    Named in the clause and named only in the body are the same case to this
    generator (both go through `_all_text`), and both are asserted: the re-triage
    stamp that let it fire in #2248 cleared the front matter, so the citation
    living in the body is not a hypothetical.
    """
    repo, committed, out_of_tree = _prefix_miss_shape(tmp_path, monkeypatch)

    in_clause = f"the sweep's real row is in {_IN_TREE_DIR}/{_PREFIX_MISS}"
    monkeypatch.chdir(repo)
    assert B._witness_target(f"{_IN_TREE_DIR}/{_PREFIX_MISS}").is_file(), (
        "positive control: the relative citation resolves from the repo root, so "
        "the skip is reading a real path and not an absent one")
    assert _is_tracked(repo, f"{_IN_TREE_DIR}/{_PREFIX_MISS}"), (
        "and the cited copy is under history, not merely written into a directory "
        "that happens to be a repository — `in_git_tree` cannot tell those apart, "
        "so the fixture has to")
    body = _BODY_1869.format(witness=out_of_tree)
    assert B.add_witness_artifact_clause([in_clause], body) == [in_clause], (
        "the demand re-fired over bytes a tracked fixture already pins")

    body_names_both = (_BODY_1869.format(witness=out_of_tree)
                       + f" The canonical extract sits at {committed}.")
    assert B.add_witness_artifact_clause(_AUTHORED, body_names_both) == _AUTHORED, (
        "the same basename named only in the body must skip too — the re-triage "
        "stamp that fired in #2248 had cleared the front matter")
    assert B._witness_target(out_of_tree).is_file() and not B.in_git_tree(
        B._witness_target(out_of_tree)), (
        "and the candidate really is the un-historied half, or this proves nothing")


def test_a_witness_whose_basename_no_tree_path_names_still_gains_one_clause(
        tmp_path, monkeypatch):
    """#2267 clause 2: the skip is a basename test, never a blanket skip.

    The same layout with the fixture's basename changed by one character must
    still produce exactly one clause naming where the bytes go — otherwise the
    guard would silence every witness demand in a `tests/fixtures`-heavy repo and
    read as a corpus with no un-archived evidence at all.

    The renamed copy is then CITED, by a clause, which is the whole point of the
    node: with nothing in the haystack naming an in-tree path, the generator
    would fire because its pin set is empty and this would pin "no in-tree token
    present" rather than the comparison. The positive control asserts the pin set
    is populated — one basename, and it is not the witness's — so the clause that
    comes out is the skip running against a real pinned copy and declining on the
    difference.
    """
    repo, _, out_of_tree = _prefix_miss_shape(tmp_path, monkeypatch)
    other = "vllm_prefix_miss_2026-09-23.json"
    (repo / _IN_TREE_DIR / _PREFIX_MISS).rename(repo / _IN_TREE_DIR / other)
    _git_commit(repo, f"{_IN_TREE_DIR}/{_PREFIX_MISS}", f"{_IN_TREE_DIR}/{other}")
    monkeypatch.chdir(repo)
    assert _is_tracked(repo, f"{_IN_TREE_DIR}/{other}"), (
        "the different-basename copy is under history too, or the control below "
        "is asserting a tree that holds nothing")

    pin_clause = f"the canonical extract is committed at {_IN_TREE_DIR}/{other}"
    authored = [_AUTHORED[0], pin_clause]
    body = _BODY_1869.format(witness=out_of_tree)
    pins = B._tree_basenames("\n".join([*authored, body]), B.in_git_tree)
    assert pins == {other}, (
        f"the generator must see exactly one pinned basename, the cited copy's: {pins}")

    out = B.add_witness_artifact_clause(authored, body)
    assert len(out) == 3, out
    assert f"backlog/data/{_PREFIX_MISS}" in out[2], out[2]
    assert out[:2] == authored, "the authored clauses are never displaced"


def test_a_witness_beyond_the_size_bound_is_skipped_like_an_absent_one(tmp_path,
                                                                      monkeypatch):
    """#2267 clause 3: a bound named as a constant, applied as a skip.

    A live probe today picks `~/lloyd-data/logs/services/agent-llm-primary.log` —
    3,138,311 bytes when triage measured it, 6,179,560 at this round's own probe
    run, growing and rotating daily — and asks a round to copy all
    of it into a vault directory that already holds 108 MB. The canonical extracts
    this convention actually commits are 169-263 KB, so the bound is 1 MiB: over
    it, the token is skipped exactly as an absent file is skipped — no clause, and
    the list returned as it came in.

    A size that cannot be read is a skip too, never a pass: `_witness_within_size_bound`
    answers False on `OSError`, because bytes the generator cannot measure are
    bytes it must not order copied. The node below asserts that branch through a
    dangling symlink, the shape a rotation leaves behind. (End to end the same
    path is already stopped one guard earlier by `_witness_is_on_disk`, so the
    branch has to be asked directly to be pinned at all.)
    """
    assert B.WITNESS_MAX_BYTES == 1_048_576, "the bound is a named constant"
    home = tmp_path / "home"
    services = home / "lloyd-data" / "logs" / "services"
    services.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    body = _BODY_1869.format(witness="~/lloyd-data/logs/services/agent-llm-primary.log")
    big = services / "agent-llm-primary.log"
    big.write_bytes(b"line\n" * (B.WITNESS_MAX_BYTES // 5 + 1))
    assert big.stat().st_size > B.WITNESS_MAX_BYTES, (
        "positive control: the fixture is over the bound, so the skip is what "
        "produced the unchanged list")
    assert B.add_witness_artifact_clause(_AUTHORED, body) == _AUTHORED, (
        "a 3 MB rotating log must not become a copy-this demand")

    big.write_bytes(b"line\n" * 200)
    assert big.stat().st_size < B.WITNESS_MAX_BYTES
    out = B.add_witness_artifact_clause(_AUTHORED, body)
    assert len(out) == 2 and "backlog/data/agent-llm-primary.log" in out[1], (
        "under the bound the same token still fires, so the bound is not a "
        f"blanket skip either: {out}")

    # The unreadable-size branch, asserted as the skip it is. A rotation leaves a
    # name that resolves to nothing behind; `stat()` raises, and the answer must
    # be "not within the bound", because bytes the generator cannot measure are
    # bytes it must not order copied. Asserting True here instead would have the
    # generator ask a round to copy a file no one has ever sized.
    rotated = services / "agent-llm-primary.log.1"
    rotated.symlink_to(services / "agent-llm-primary.log.2026-10-05")
    assert not rotated.is_file(), "positive control: the target really is absent"
    assert B._witness_within_size_bound(rotated) is False, (
        "an unreadable size is a skip, not a pass")
    assert B._witness_within_size_bound(big) is True, (
        "and the readable under-bound case is the one answering True")


def test_the_rederive_command_varies_with_the_committed_bytes(tmp_path):
    """#2267 clause 4: a proving command that can actually disagree.

    Every canonical extract is ONE line — `wc -l` printed 1 for all six
    `tests/fixtures/vllm_prefix_miss_*.json` while their bytes ranged 169,348 to
    263,748 — so the clause's "that output is the figure the item quotes" pinned
    nothing at all: any two extracts passed. The emitted command must therefore
    produce different output for two one-line extracts with different contents,
    which the line count alone never will. Line count is kept beside the byte
    count so the multi-line witnesses that legitimately quote a line figure
    (`session_witness_*.json`, whose own line count IS its figure) keep it.
    """
    a = tmp_path / "extract-a.json"
    b = tmp_path / "extract-b.json"
    a.write_text('{"date": "2026-10-01", "miss_rate": 0.063}\n')
    b.write_text('{"date": "2026-10-01", "miss_rate": 0.063, "window": "7d", '
                 '"calls": 1016}\n')
    for one_line in (a, b):
        assert one_line.read_text().count("\n") == 1, "both are one-line extracts"
    assert subprocess.run(f"wc -l < {a}", shell=True, capture_output=True,
                          text=True).stdout == \
        subprocess.run(f"wc -l < {b}", shell=True, capture_output=True,
                       text=True).stdout == "1\n", (
        "the vacuity this replaces: line count cannot tell these apart")

    out = [subprocess.run(B._rederive_command(str(p), "json"), shell=True,
                          capture_output=True, text=True).stdout
           for p in (a, b)]
    assert out[0] != out[1], f"the command is still vacuous: both printed {out[0]!r}"
    assert out[0].split() == ["1", str(a.stat().st_size)], (
        f"the figure must be readable, not a hash: {out[0]!r}")


def test_no_format_the_convention_archives_is_handed_a_vacuous_proving(tmp_path):
    """#2267 clause 6 as amended for this round: no archived format gets a
    line-count-only demand.

    The original clause ordered a copy of
    `~/lloyd-data/logs/services/agent-llm-primary.log` into `backlog/data`, to be
    re-derived with the vacuous `wc -l` this item exists to retire. The review
    called that unsatisfiable twice — the log grows daily (3,138,311 bytes at
    triage, 6,179,560 at this round's probe, 6,244,939 at the review's own read),
    so no figure a clause names for it survives the round that carries it — and
    this round amended the clause to the property below (ratification is the next
    review's). Both halves of the retired demand are already refused by the
    generator itself: the bytes are over `WITNESS_MAX_BYTES`, and the command it
    would have carried is the one clause 4 removed. What is left to pin is the
    property one format cannot establish alone — that the fix reaches every
    extension the convention takes, not just the json triage happened to cite —
    because a `.jsonl` quoted from its line count and a `.parquet` whose real
    report command the item never quoted both pass on a table of one.

    So: every format in `WITNESS_EXTS` is asked through the real generator here. A
    file format's clause must carry `wc -l -c`, and must NOT carry the bare
    `wc -l < artifact`; the three store formats keep the `sqlite_master` count,
    which varies with the bytes by construction.
    """
    d = tmp_path / "lloyd-data" / "_pipeline" / "reflection"
    d.mkdir(parents=True)
    for ext in sorted(set(B.WITNESS_EXTS)):
        witness = d / f"run.{ext}"
        witness.write_text('{"rows": 17}\n')
        assert B.in_git_tree(witness) is False, (ext, "fixture must be out of tree")
        out = B.add_witness_artifact_clause(_AUTHORED, _BODY_1869.format(witness=witness))
        assert len(out) == 2, (ext, out)
        artifact = f"{B.WITNESS_ARTIFACT_DIR}/run.{ext}"
        clause = out[1]
        if ext in B._SQL_WITNESS_EXTS:
            assert "sqlite_master" in clause, (ext, clause)
        else:
            assert f"`wc -l -c < {artifact}`" in clause, (ext, clause)
            assert f"`wc -l < {artifact}`" not in clause, (
                f"a .{ext} witness was handed the vacuous single-figure form: {clause}")


def test_a_clause_that_already_names_an_archive_adds_no_second_one(tmp_path):
    """The de-dup half of the trigger: the rule is one clause per item, and an
    item whose contract already routes the bytes (the `backlog/data/`
    convention, or an explicit vault path) is done asking."""
    witness = _witness(tmp_path)
    body = _BODY_1869.format(witness=witness)
    for clause in (f"commit the extract to `backlog/data/{witness.name}` in the vault",
                   "the mirror at `~/obsidian/knowledge/inner-voice/iv-metrics.jsonl` "
                   "carries the same bytes"):
        out = B.add_witness_artifact_clause([_AUTHORED[0], clause], body)
        assert out == [_AUTHORED[0], clause], out


def test_the_generated_clause_reaches_the_graded_contract(tmp_path, isolated):
    """Clause 5: `record_verdict` writes it into `acceptance_clauses`, and the
    post-landing backstop leaves it there.

    End to end through the real writer: the clause is generated between
    `cap_new_clauses` and `split_post_landing_clauses`, so the same regex that
    parks an authored "needs a day of traffic" clause gets its chance at this
    one — and does not move it, because it is worded over committed bytes with
    no time shape.
    """
    witness = _witness(tmp_path)
    assert B.in_git_tree(witness) is False, "fixture must sit outside every tree"
    p = write_item(isolated, 11, body=_BODY_1869.format(witness=witness))

    B.record_verdict(B.item_by_id(11), "confirmed", "the counts re-measure exactly",
                     acceptance="the report re-derives from committed bytes",
                     acceptance_clauses=list(_AUTHORED))

    fm = _fm(p)
    graded = fm["acceptance_clauses"]
    assert len(graded) == 2, graded
    generated = graded[1]
    assert ("backlog/data/iv-metrics.jsonl" in generated
            and "`wc -l -c < " in generated), generated
    assert not fm.get("human_clauses"), fm.get("human_clauses")
    assert "Moved to human clauses" not in p.read_text(), p.read_text()[-400:]
    assert B.split_post_landing_clauses([generated])[0] == [generated], (
        "the wording itself must survive the backstop, not its position in the list")


# ── #1889: only bytes that are on this disk can be a witness ─────────────────
#
# The rule above fires on a path an item NAMES, and until #1889 it accepted any
# token whose last segment looked absolute: the pattern made the `~`/`$HOME`
# prefix optional while the `/` was mandatory, so the tail of ANY relative path
# ending in a witness extension matched as a root-form one. #1886's body names
# the relative `chrome-extension/manifest.json`, the pattern returned
# `/manifest.json`, `in_git_tree` walked a nonexistent path up to `/` (where
# `git rev-parse --show-toplevel` exits 128) and reported "no history", and the
# emitter asked a round to `copy /manifest.json` — bytes that never existed on
# any machine, in a contract no vault-surface round can amend. So the two halves
# below are separate fixes: a token must be a real root-form path (clause 1),
# and a real root-form token must still be on the disk to count (clause 3).
# Together they leave the five legitimate clauses on the board — #1878, #1879,
# #1880, #1883 and #1884, whose witnesses all exist — firing exactly as before.

#: #1886's own body line 102, verbatim enough to reproduce the misparse. Its
#: only witness-extension path is the RELATIVE `chrome-extension/manifest.json`.
_BODY_RELATIVE_ONLY = (
    "**Same defect class, in code comments nobody is currently tracking:** "
    "`agent-services/guardian/datawatch.py:153` lists `web/.vite` among "
    "directories that \"merely hold a build or test cache\" on this tree, and "
    "`chrome-extension/manifest.json` is gitignored while its build output is "
    "tracked, so a fresh clone ships an unloadable extension.")

#: The three spellings a real out-of-tree store appears in, quoted from the
#: live items that fired the rule legitimately: #1879/#1880 name the first, #1884
#: the second, and #1883's `~/.local/state/lloyd-automod/...` store takes the
#: third form once `~` is resolved. All three must keep matching.
_OUT_OF_TREE_SPELLINGS = ("~/.local/state/lloyd-request-manifests/manifests/2026-09-30.ndjson",
                          "$HOME/lloyd-data-cutover-20260922-201723.log",
                          "/home/alansrobotlab/.local/state/lloyd-automod/promotions.jsonl")

_COPY_SRC_RX = re.compile(r"copy `([^`]+)` to `")


def _resolve(tok: str) -> Path:
    """`tok` as a file on this box: `$HOME` and `~` both stand for home.

    Spelled out here rather than calling the emitter's own resolver, so clause
    4's assertion is an independent reading of the emitted text.
    """
    return Path(os.path.expandvars(str(tok))).expanduser()


def test_a_relative_path_tail_yields_no_token_and_no_clause(tmp_path, monkeypatch):
    """#1889 clause 1: prose whose only witness-extension path is the relative
    `chrome-extension/manifest.json` yields no token from WITNESS_PATH_RX, and
    add_witness_artifact_clause returns the input clause list unchanged.

    Three readings of the same anchor. The positive control is the identical
    file name in its root form: an empty token list is the anchor refusing a
    relative tail, not a pattern that has stopped seeing `.json` at all. Then a
    mixed body — a real witness and the relative path in one paragraph — where
    the relative tail must contribute nothing even when a clause IS emitted, so
    the anchor is shown to hold next to a firing case rather than only in
    isolation. A URL's path is the second shape the old pattern could cut a
    token out of (`https://docs.example.com/run/report.jsonl`), and it is
    refused by the same boundary.
    """
    assert B.WITNESS_PATH_RX.findall(_BODY_RELATIVE_ONLY) == [], (
        f"a relative path's tail must not match as absolute: "
        f"{B.WITNESS_PATH_RX.findall(_BODY_RELATIVE_ONLY)}")
    assert B.WITNESS_PATH_RX.findall("`/chrome-extension/manifest.json`") == [
        "/chrome-extension/manifest.json"], "positive control: the root form matches"
    assert B.WITNESS_PATH_RX.findall(
        "Upstream `https://docs.example.com/run/report.jsonl` printed 17 rows.") == [], (
        "a URL's path is not a local witness either")

    out = B.add_witness_artifact_clause(_AUTHORED, _BODY_RELATIVE_ONLY)
    assert out == _AUTHORED, out
    assert not any("backlog/data/" in c for c in out), out

    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    (tmp_path / "h" / ".local").mkdir(parents=True)
    real = tmp_path / "h" / ".local" / "run.jsonl"
    real.write_text("{}\n" * 17)
    mixed = (_BODY_RELATIVE_ONLY + f" The live run over `{real}` printed 17 rows.")
    assert B.WITNESS_PATH_RX.findall(mixed) == [str(real)], (
        f"only the root-form path may be a token: {B.WITNESS_PATH_RX.findall(mixed)}")
    fired = B.add_witness_artifact_clause(_AUTHORED, mixed)
    assert len(fired) == 2, fired
    assert str(real) in fired[1] and "manifest.json" not in fired[1], fired[1]


def test_the_three_root_spellings_of_a_store_still_match_and_still_fire(tmp_path,
                                                                       monkeypatch):
    """#1889 clause 2: the anchor must not cost the rule its real reach.

    First half is the pattern over the three spellings the live items use, each
    matched IN FULL (the whole string, not a tail of it). Second half is the
    emitter over a body naming a store that really is out of tree in each of
    those shapes — one clause added in every case, and the clause names the
    spelling the body used.
    """
    for tok in _OUT_OF_TREE_SPELLINGS:
        assert B.WITNESS_PATH_RX.fullmatch(tok) is not None, f"{tok} must match whole"
        found = B.WITNESS_PATH_RX.findall(f"Live run over `{tok}` printed 17 rows.")
        assert found == [tok], f"{tok} must match in full, got {found}"

    def _fake_home(name: str) -> Path:
        home = tmp_path / name
        (home / ".local" / "state" / "req").mkdir(parents=True)
        return home

    fake = _fake_home("h1")
    (fake / ".local" / "state" / "req" / "m.ndjson").write_text("{}\n" * 17)
    monkeypatch.setenv("HOME", str(fake))
    tilde = B.add_witness_artifact_clause(
        _AUTHORED, "Over `~/.local/state/req/m.ndjson`: 17 rows.")
    assert len(tilde) == 2, tilde
    assert "~/.local/state/req/m.ndjson" in tilde[1], tilde[1]

    (fake / "cutover.log").write_text("cutover\n" * 4)
    dollar = B.add_witness_artifact_clause(
        _AUTHORED, "Over `$HOME/cutover.log`: 4 rows.")
    assert len(dollar) == 2, dollar
    assert "$HOME/cutover.log" in dollar[1], dollar[1]

    root_form = _witness(tmp_path, "usage.jsonl")
    absolute = B.add_witness_artifact_clause(
        _AUTHORED, f"Over `{root_form}`: 17 rows.")
    assert len(absolute) == 2, absolute
    assert str(root_form) in absolute[1], absolute[1]


def test_a_named_path_that_is_not_on_this_disk_gates_before_the_probe(tmp_path,
                                                                     monkeypatch):
    """#1889 clause 3: a `~`- or root-form path that does not exist on disk
    fires no clause, because the emitter checks existence before treating
    `probe() == False` as "no history", and the clause list comes back unchanged
    when that nonexistent path is the only candidate.

    Every probe here says "no git tree" — the answer that fired the rule for
    #1886, whose token was `/manifest.json` (quoted in the shape assertion, which
    asks only whether the pattern sees the shape; nothing here depends on what
    the filesystem root of the machine running this suite happens to hold). The
    three misses are built under `tmp_path`, one in each home spelling, so the
    only thing that can stop a clause is the stat, and the control under them is
    the same token with bytes written: that fires, on the probe that still says
    False.
    """
    empty = tmp_path / "empty-home"
    monkeypatch.setenv("HOME", str(empty))
    missing = ["~/lloyd-data/_pipeline/reflection/no-such-store.jsonl",
               "$HOME/nowhere/1999-01-01.ndjson",
               str(tmp_path / "gone" / "report.jsonl")]
    for tok in missing:
        assert B.WITNESS_PATH_RX.findall(f"`{tok}`") == [tok], (
            f"{tok} is still a token by shape; the gate must be existence")
        assert _resolve(tok).is_file() is False, f"fixture: {tok} must not exist"
        out = B.add_witness_artifact_clause(
            _AUTHORED, f"Live run over `{tok}` printed 17 rows.",
            probe=lambda path: False)
        assert out == _AUTHORED, f"a nonexistent witness fired: {out}"
    assert B.WITNESS_PATH_RX.findall("`/manifest.json`") == ["/manifest.json"], (
    "the shape #1886 was handed is still a shape the pattern sees")

    # And the same body with the bytes present does fire, so the list coming
    # back unchanged above is the stat and nothing else.
    (empty / "lloyd-data" / "_pipeline" / "reflection").mkdir(parents=True)
    (empty / "lloyd-data" / "_pipeline" / "reflection" / "no-such-store.jsonl").write_text(
        "{}\n" * 17)
    fired = B.add_witness_artifact_clause(
        _AUTHORED,
        "Live run over `~/lloyd-data/_pipeline/reflection/no-such-store.jsonl` "
        "printed 17 rows.", probe=lambda path: False)
    assert len(fired) == 2, fired


def test_every_clause_the_emitter_adds_names_bytes_that_exist(tmp_path, monkeypatch):
    """#1889 clause 4: for every clause the emitter does append, the path in its
    `copy `X`` resolves through expanduser() to an existing file.

    Two readings, because the property is about the CLAUSE, not only the trigger.
    First the three spellings, with the clause count asserted before the loop — a
    loop over zero emitted clauses would satisfy the property vacuously, which is
    how a rule that never fires looks identical to a rule that is checked. Then a
    body holding three candidate tokens — a nonexistent root form first in
    reading order, a real witness second, and the relative tail — where exactly
    one clause may exist, carrying exactly one `copy ` source (counted, not
    searched), and that source is the surviving token: no skipped candidate may
    leave its name in the artifact path or the prose. That is what pins the
    general guarantee rather than the three cases, since the clause is built from
    the one token the stat passed and nothing else can reach the text.
    """
    fake = tmp_path / "h"
    (fake / ".local" / "state").mkdir(parents=True)
    (fake / ".local" / "state" / "voice.log").write_text("line\n" * 3)
    (fake / "gate.json").write_text('{"rungs": 6}\n')
    monkeypatch.setenv("HOME", str(fake))
    bodies = ["Over `~/.local/state/voice.log`: 3 lines.",
              "Over `$HOME/gate.json`: 1 object.",
              f"Over `{fake / '.local' / 'state' / 'voice.log'}`: 3 lines."]
    n = 0
    for body in bodies:
        out = B.add_witness_artifact_clause(_AUTHORED, body)
        assert len(out) == 2, f"expected one emitted clause for {body!r}: {out}"
        for clause in out[len(_AUTHORED):]:
            src = _COPY_SRC_RX.search(clause)
            assert src, f"the clause no longer names its source: {clause}"
            assert _resolve(src.group(1)).is_file(), (
                f"clause orders bytes that do not exist: {src.group(1)!r} -> {clause}")
            n += 1
    assert n == 3, f"three clauses expected, checked {n}"

    loser = "~/nowhere-dir/dead-store.jsonl"
    winner = fake / ".local" / "state" / "live-store.jsonl"
    winner.write_text("{}\n" * 17)
    three = (f"Live run over `{loser}` printed nothing, "
             f"the real figures came from `{winner}`, "
             "and `chrome-extension/manifest.json` is unrelated.")
    assert B.WITNESS_PATH_RX.findall(three) == [loser, str(winner)], (
        f"fixture: two root-form tokens expected, "
        f"got {B.WITNESS_PATH_RX.findall(three)}")
    out = B.add_witness_artifact_clause(_AUTHORED, three)
    assert len(out) == 2, out
    added = out[len(_AUTHORED):]
    assert len(added) == 1, added
    clause = added[0]
    assert clause.count("copy `") == 1, f"one source expected in: {clause}"
    src = _COPY_SRC_RX.search(clause)
    assert src and src.group(1) == str(winner), f"{clause}"
    assert _resolve(src.group(1)).is_file(), clause
    for absent in ("dead-store", "manifest.json"):
        assert absent not in clause, f"a skipped candidate leaked into {clause}"


@pytest.mark.live_vault
def test_item_1886_contract_keeps_only_its_three_authored_clauses():
    """#1889 clause 5: the contract the mechanism corrupted is whole again.

    Reads the live vault (#1886's own item file), which no round under test
    controls — hence `live_vault`, the same mark `test_bench_audit_tasks.py`
    carries; the gate's `tests` rung deselects it (`TESTS_MARK_EXPR`), so the
    run cited for this clause is the one this round ran by hand. Its two halves:
    `acceptance_clauses` holds the three authored skill clauses with no witness
    clause among them and no `copy `/manifest.json`` line left in the prose, and
    re-running the emitter over that body now adds nothing — the proof that
    deleting the clause is not undone by the next triage of the same item.
    """
    p = sorted((Path.home() / "obsidian" / "backlog").glob("1886-*.md"))
    assert len(p) == 1, p
    text = p[0].read_text()
    fm, body = B._split_frontmatter(text)

    graded = fm["acceptance_clauses"]
    assert len(graded) == 3, graded
    assert not any("witness bytes have no history" in c for c in graded), graded
    assert not any("/manifest.json" in c for c in graded), graded
    assert "copy `/manifest.json`" not in body, body[body.find("Acceptance clauses"):]
    assert B.WITNESS_PATH_RX.findall(body) == [] or all(
        _resolve(t).is_file() for t in set(B.WITNESS_PATH_RX.findall(body))), (
        "any root-form token left in the body must at least be real bytes")
    assert B.add_witness_artifact_clause(list(graded), body) == list(graded), (
        "the emitter must not re-add a witness clause to #1886")


def test_the_docstring_carries_no_dated_ledger_count_about_the_vault_route():
    """#1987: the vault refusal row now writes per-clause verdicts, so the sentence
    saying it writes none — and its "(0 of 291 on 2026-09-30)" — had rotted."""
    doc = " ".join(B.add_witness_artifact_clause.__doc__.split())
    assert "0 of " + "291" not in doc
    assert "no per-clause verdicts" not in doc
    assert "VAULT_NO_AMENDMENT_ROUTE" in doc
