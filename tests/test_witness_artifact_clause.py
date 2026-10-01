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
    assert f"`wc -l < {artifact}`" in added, (
        f"the clause must carry a shell command over the COMMITTED bytes, not a "
        f"promise to re-measure: {added}")
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
    assert "backlog/data/iv-metrics.jsonl" in generated and "`wc -l < " in generated
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
