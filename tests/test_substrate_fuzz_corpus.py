"""#2620 clause 1 — the corpus is data, and the loader refuses a task it cannot check.

Every claim here is about `eval/substrate_fuzz/corpus.json` as a FILE plus the loader
that reads it. Nothing dispatches anything: a corpus that stops parsing must fail here
and not inside a 44-task run whose first task happens to be the broken one.

The refusals are the clause. "Every task carries a mechanically checkable
postcondition" is not a property of the rows that exist today — it is a property of the
rows that get ADDED tomorrow, by someone who is not in this diff, and the only thing
that enforces it after this round is the loader refusing what does not carry one. A
test that only counts today's 44 tasks asserts a fact about a snapshot.
"""

from __future__ import annotations

import json
import re

import pytest

from _substrate_fuzz import (BY_ID, CONTROLS, CORPUS, REAL, b64, parse_task_row,
                             sha, sf_task, write_corpus)

from eval.substrate_fuzz import corpus as C

ALL_IDS = {t.id for t in CORPUS}


# ── the shipped file ─────────────────────────────────────────────────────────

def test_corpus_is_at_least_forty_tasks_over_six_domains():
    """44 tasks over 7 domains, counted from the file and not from a docstring.

    The floors are the item's (`>=40 tasks over >=6 substrate domains`). Counted over
    the REAL tasks: a control is a copy of a task it already counts toward, and a
    corpus of 20 tasks with 24 controls would otherwise clear the bar on paper while
    exercising 20 shapes.
    """
    stats = C.corpus_stats(CORPUS)
    assert stats["tasks"] >= C.MIN_TASKS, stats
    assert stats["domains"] >= C.MIN_DOMAINS, stats
    assert stats["controls"] >= stats["domains"], (
        f"{stats['controls']} controls over {stats['domains']} domains cannot cover "
        f"every domain, which clause 3 requires")


def test_every_domain_has_at_least_one_positive_control():
    """The green-board rule, enforced at load time rather than in a README.

    Every one of the 7 domains appears in the `control_for` set. A domain with no
    control is a domain whose 44-green-tasks result says nothing about whether
    anything was measured, which is #644's stated objection to generated rows and the
    reason #2582's class rule reads `a 0-hit grep needs a positive control`.
    """
    stats = C.corpus_stats(CORPUS)
    assert not stats["domains_without_control"], stats
    covered = {t.control_for for t in CONTROLS}
    for t in REAL:
        assert t.id in ALL_IDS
    assert covered <= {t.id for t in REAL}


def test_every_task_names_a_dispatchable_tool_and_carries_its_own_args():
    """A step with no `args` would dispatch an empty argument object and report a clean
    result about the wrong call — a task that silently tests nothing, in data form.
    """
    for t in CORPUS:
        assert t.steps, f"{t.id} has no steps"
        for i, step in enumerate(t.steps):
            assert isinstance(step.get("tool"), str) and step["tool"], \
                f"{t.id} step {i} names no tool"
            assert isinstance(step.get("args"), dict), \
                f"{t.id} step {i} has no args object"


def test_control_rows_restate_their_base_task_rather_than_sharing_it():
    """A control must be a real task row, not a pointer: `run_task` takes one row and
    runs what it says, so a control that omitted its steps would run nothing and report
    a green control — a detector that reports itself alive.
    """
    for c in CONTROLS:
        base = BY_ID[c.control_for]
        assert c.steps == base.steps, (
            f"{c.id} does not restate {base.id}'s steps, so it exercises a different "
            f"sequence than the one it claims to prove")
        assert c.post == base.post, (
            f"{c.id} does not carry {base.id}'s shipped postcondition")
        assert c.control_at == "tool_boundary", c.id


def test_no_control_is_parked_on_a_postcondition_that_reads_no_bytes():
    """The loader's own vacuity rule, re-asserted over the shipped file.

    `exit_shape` reads a boolean and `file_exists` an inode: no byte-level break at the
    boundary can redden either, so a control over one is a control that can never fire.
    The loader refuses the pairing at load time and this test makes sure the shipped
    corpus did not find a way around it.
    """
    for c in CONTROLS:
        base = BY_ID[c.control_for]
        reason = C._control_can_redden(c, base)
        assert reason is None, f"{c.id}: {reason}"


def test_corpus_file_is_stable_sorted_json():
    """The file is data a person reviews in a diff, so key order must not churn.

    A JSON file written without `sort_keys` produces a diff that moves every line when
    a task is appended, which is how a corpus review becomes unreadable and how a
    changed expected-hash hides inside the noise.
    """
    raw = C.CORPUS_PATH.read_text(encoding="utf-8")
    doc = json.loads(raw)
    assert raw == json.dumps(doc, indent=2, sort_keys=True) + "\n", (
        "corpus.json is not in the canonical form the writer uses; re-run "
        "`python -m eval.substrate_fuzz.runner --emit-corpus`")


# ── the loader refuses ───────────────────────────────────────────────────────

def test_loader_refuses_a_task_with_no_postcondition():
    """#2620 clause 1, the exact case the clause names: a task without a mechanical
    postcondition is refused, and a run therefore cannot contain a step nobody checked.

    This is the whole clause in one assert. Everything else in this file is context for
    it; without this refusal the corpus degrades into a script that dispatches tools
    and prints what happened, which is precisely the green-asserting-nothing instrument
    #644's triage refused.
    """
    row = sf_task()
    del row["postcondition"]
    with pytest.raises(C.CorpusError, match="postcondition"):
        parse_task_row(row)


def test_loader_refuses_a_vague_postcondition_kind():
    """`kind: "looks_ok"` is a postcondition in name only.

    The accepted kinds are the four the item enumerates — an expected sha256, an
    expected byte length, an exit shape, a file-exists-under-the-run-root — plus the
    stdout and spill-resolves readings of the same idea. Anything else is an
    LLM-judged verdict wearing a field name, and "no LLM grading anywhere" is in the
    item's own approach line.
    """
    for bad in ("looks_ok", "llm_grade", "reasonable", "graded_by_model"):
        with pytest.raises(C.CorpusError, match="postcondition kind"):
            parse_task_row(sf_task(post={"kind": bad}))


def test_loader_refuses_a_sha256_expected_that_is_not_a_digest():
    """A typo in a 64-hex digest is the corpus equivalent of a test with no assert."""
    for bad in ("deadbeef", "z" * 64, "A" * 63):
        with pytest.raises(C.CorpusError, match="64-hex"):
            parse_task_row(sf_task(post={"kind": "file_sha256", "path": "x.bin",
                                         "expected": bad}))


def test_loader_refuses_a_file_check_with_no_path():
    """`file_length` of nothing, or `file_exists` nowhere, both read as checks and
    neither can fail."""
    for kind in ("file_sha256", "file_length", "file_exists"):
        with pytest.raises(C.CorpusError, match="needs a 'path'"):
            parse_task_row(sf_task(post={"kind": kind, "expected": "3"}))


def test_loader_refuses_a_file_path_that_escapes_the_run_root():
    """A postcondition could otherwise read ANY file on the box — including a protected
    one — and report it as the run's own result. The run root is the only universe a
    mechanical check is allowed to speak about.
    """
    for bad in ("../outside.txt", "/etc/hostname", "./../x", "a/../../b"):
        with pytest.raises(C.CorpusError, match="run root"):
            parse_task_row(sf_task(post={"kind": "file_exists", "path": bad}))


def test_loader_refuses_a_control_parked_on_an_exit_shape_task(tmp_path):
    """The vacuous-detector case, spelled out as a refusal rather than a comment.

    A `flip_one_byte` control over a task whose postcondition reads a boolean can never
    redden: the injected byte cannot change `is_error`. Loading it anyway would put a
    control on the books that reports "detector alive" forever.
    """
    base = sf_task(tid="base-shape",
                   post={"kind": "exit_shape", "is_error": False})
    ctl = sf_task(tid="ctl-shape", post={"kind": "exit_shape", "is_error": False},
                  control_for="base-shape", control="flip_one_byte",
                  control_at="tool_boundary")
    with pytest.raises(C.CorpusError, match="reads no bytes"):
        C.load_corpus(write_corpus(tmp_path, [base, ctl]))


def test_loader_refuses_a_control_whose_base_writes_the_file_by_redirect(tmp_path):
    """The second vacuity route: the break lands on bytes crossing the boundary, so a
    base task that lays its file down with a shell redirect has no argument to break and
    its file-side hash stays green no matter what the control does.

    The fix is to point the control at a task whose payload arrives as a `content`
    argument, which is what `roundtrip-text-write` and `fs-write-nested-path` are for.
    """
    base = sf_task(tid="base-redir", domain="filesystem",
                   steps=[{"tool": "Bash",
                           "args": {"command": "printf x > '{root}/f.bin'"}}],
                   post={"kind": "file_sha256", "path": "f.bin", "expected": sha(b"x")})
    ctl = sf_task(tid="ctl-redir", domain="filesystem",
                  steps=[{"tool": "Bash",
                          "args": {"command": "printf x > '{root}/f.bin'"}}],
                  post={"kind": "file_sha256", "path": "f.bin", "expected": sha(b"x")},
                  control_for="base-redir", control="flip_one_byte",
                  control_at="tool_boundary")
    with pytest.raises(C.CorpusError, match="without a `content` argument"):
        C.load_corpus(write_corpus(tmp_path, [base, ctl]))


def test_loader_refuses_a_control_naming_an_unknown_task(tmp_path):
    """A typo'd `control_for` would be a control counted in the per-domain coverage and
    pointing at nothing, which reads as coverage and is a hole."""
    ctl = sf_task(tid="ctl-ghost", control_for="no-such-task",
                  control="flip_one_byte", control_at="tool_boundary")
    with pytest.raises(C.CorpusError, match="is not a task in this corpus"):
        C.load_corpus(write_corpus(tmp_path, [ctl]))


def test_loader_refuses_an_unknown_control_kind(tmp_path):
    """The vocabulary of breaks is closed, so a control cannot invent a break nobody
    implemented and thereby report itself as exercised."""
    base = sf_task(tid="base-w", domain="byte_roundtrip",
                   steps=[{"tool": "Write", "args": {"file_path": "{root}/w.txt",
                                                     "content": "abc"}}],
                   post={"kind": "file_sha256", "path": "w.txt", "expected": sha(b"abc")})
    ctl = sf_task(tid="ctl-w", domain="byte_roundtrip",
                  steps=[{"tool": "Write", "args": {"file_path": "{root}/w.txt",
                                                    "content": "abc"}}],
                  post={"kind": "file_sha256", "path": "w.txt", "expected": sha(b"abc")},
                  control_for="base-w", control="shuffle_bytes",
                  control_at="tool_boundary")
    with pytest.raises(C.CorpusError, match="the vocabulary is"):
        C.load_corpus(write_corpus(tmp_path, [base, ctl]))


def test_loader_refuses_a_control_breaking_somewhere_else(tmp_path):
    """`control_at` is pinned to the tool boundary because a break inside the runner's
    own bookkeeping would prove the bookkeeping works and nothing about the substrate.
    """
    base = sf_task(tid="base-at",
                   steps=[{"tool": "Write", "args": {"file_path": "{root}/a.txt",
                                                     "content": "abc"}}],
                   post={"kind": "file_sha256", "path": "a.txt", "expected": sha(b"abc")})
    ctl = sf_task(tid="ctl-at",
                  steps=[{"tool": "Write", "args": {"file_path": "{root}/a.txt",
                                                    "content": "abc"}}],
                  post={"kind": "file_sha256", "path": "a.txt", "expected": sha(b"abc")},
                  control_for="base-at", control="flip_one_byte",
                  control_at="inside_the_runner")
    with pytest.raises(C.CorpusError, match="control_at"):
        C.load_corpus(write_corpus(tmp_path, [base, ctl]))


def test_loader_refuses_a_fixture_whose_declared_digest_does_not_match():
    """Every round-trip task's expected value is the digest of its own fixture, so a
    fixture that no longer matches its declared sha means the expected value is a
    guess. Refused at load rather than run as a defect nobody can attribute.
    """
    
    row = sf_task(tid="bad-fixture", fixture=b64(b"payload"),
                  fixture_sha256=sha(b"different payload"),
                  steps=[{"tool": "Bash", "args": {"command": "printf x"}}],
                  post={"kind": "stdout_equals", "expected": "x"})
    with pytest.raises(C.CorpusError, match="fixture_sha256"):
        parse_task_row(row)


def test_loader_refuses_the_same_id_twice(tmp_path):
    """Two tasks with one id make the artifact's per-domain `k/N` meaningless: the
    denominator double-counts and `failing_task_ids` names an id a reader cannot
    resolve to one row."""
    p = write_corpus(tmp_path, [sf_task(tid="dup"), sf_task(tid="dup")])
    with pytest.raises(C.CorpusError, match="twice"):
        C.load_corpus(p)


def test_loader_refuses_a_top_level_shape_that_is_not_a_corpus(tmp_path):
    """A version this loader has never seen is refused rather than guessed at, so a
    future schema change fails loudly in the instrument that reads it."""
    p = write_corpus(tmp_path, [sf_task()], version=99)
    with pytest.raises(C.CorpusError, match="version 99"):
        C.load_corpus(p)
    p2 = write_corpus(tmp_path, [sf_task()])
    doc = json.loads(p2.read_text())
    # An item number this corpus is not: the data must name what it is for, or a reader
    # of the artifact cannot tell whose substrate a green board describes.
    doc["item"] = 9999
    p2.write_text(json.dumps(doc))
    with pytest.raises(C.CorpusError, match="not #2620"):
        C.load_corpus(p2)


def test_controlled_refusals_still_let_a_well_formed_row_load():
    """Positive control for THIS test file: the same helper that produced 9 refusals
    loads a row that satisfies the rules, so a test that passes only because
    `pytest.raises` matched an unrelated `CorpusError` cannot look like coverage.
    """
    good = sf_task(tid="fine",
                   steps=[{"tool": "Write",
                           "args": {"file_path": "{root}/f.txt", "content": "hi"}}],
                   post={"kind": "file_sha256", "path": "f.txt", "expected": sha(b"hi")})
    t = parse_task_row(good)
    assert t.id == "fine" and t.post.kind == "file_sha256"
    assert re.fullmatch(r"[0-9a-f]{64}", t.post.expected)


def test_expected_digests_are_recomputable_from_the_fixtures_they_name():
    """No expected hash in this corpus is a hand-typed constant.

    For every fixture-backed task, the postcondition's expected value is the digest of
    the fixture it carries. A hand-typed digest that quietly stopped matching the
    fixture would read as a substrate regression forever, and this assert is the only
    thing between that and a false defect report.
    """
    n = 0
    for t in REAL:
        if t.fixture is None:
            continue
        raw = t.fixture_bytes()
        assert sha(raw) == t.fixture_sha256, t.id
        if t.post.kind == "file_sha256":
            assert t.post.expected == sha(raw), (
                f"{t.id}: expected digest is not the fixture's digest")
            n += 1
    assert n >= 6, f"only {n} digest-checked fixture tasks; the round-trip domain is thin"
