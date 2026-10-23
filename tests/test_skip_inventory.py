"""#2637: the suite's skip total is a measurement, and the measurement is pinned.

The gate's `tests` rung refuses any round above `PYTEST_MAX_SKIPPED = 40`, and on
2026-10-13 the box sat at 39-41 run to run: #2636's own round drew 41 with 18,600
passing tests and zero failures and was refused on suite-wide arithmetic. The fix
has to be measured in the same units the rail uses, or a later round buys headroom
by editing the constant.

Both states are committed here as data (`fixtures/skip_inventory_*.jsonl`, one
`{"path","reason","n"}` per node), produced by `/tmp/inv2637.py`'s protocol: the
round home, the rung's mark expr, `-n 8 --dist loadfile`, and `-rs` to name the
reasons. The artifacts, not a transcription, are what the assertions read.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import skip_inventory as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# clause 1's named call sites. The two `test_research_doc_claims.py` sites the
# item names by line number have no inline `pytest.skip` on this tree — the skip
# there belongs to the corpus-collection guard at :135, not to those nodes — so
# they are named for what they are and the real conversions are listed in full.
# rel -> (node, shape). Two shapes, both non-skipping, deliberately distinct:
#   "branch"    — absent data is a named state the node asserts on and returns from
#   "committed" — the node's data is now committed, so no absent state exists and
#                 the FULL claim is asserted on every run (the stronger shape, and
#                 only available where the artifact is small enough to ship)
CONVERTED = {
    "tests/test_automod_regression.py": (
        "test_the_recorded_noise_floor_is_what_the_code_expects", "branch"),
    "tests/test_eval_scorer.py": (
        "test_a_run_record_declares_whether_it_matched_production", "branch"),
    "tests/test_uptake.py": (
        "test_live_engine_scores_the_corpus_and_reports_every_way_precision_was_measured",
        "branch"),
    "tests/test_memory_eval.py": (
        "test_the_vault_witness_of_the_unbounded_baseline_reproduces_its_quoted_numbers",
        "committed"),
    "tests/test_trajectory_extraction.py": (
        "test_production_buckets_since_the_fix_have_no_duplicate_keys", "branch"),
}

# Same file, so it cannot share the dict key above; checked by the same two nodes.
EXTRA_CONVERTED = {
    "tests/test_trajectory_extraction.py": (
        "test_pre_fix_duplicate_buckets_are_frozen_not_growing", "branch"),
}


# Converted, and its inline `pytest.skip` is gone, but the before-inventory never
# saw it fire: the guard reads the VAULT copy, and the round home symlinks
# `~/obsidian` through to the real vault, so the file is present and the node
# passes in gate shape. It fires on a machine whose vault is mid-rewrite, which is
# exactly when a skip would hide the baseline from a reader. Named rather than
# quietly exempted, so the exemption is checkable.
NOT_OBSERVED_ON_THIS_BOX = {"tests/test_memory_eval.py"}

#: Converted by THIS round's commit, with no row in the before-inventory: its guard
#: skipped on the vault working tree being mid-rewrite, and the vault happened to
#: hold the file the day the inventory was measured. It therefore cannot join
#: CONVERTED — the falsifiability check below would rightly call an unobserved row
#: an unfalsifiable claim — and it is not a previous-round conversion either. It is
#: named here so the two structural checks still pin it, and a conversion cannot
#: hide by having been measured on a lucky day.
CONVERTED_UNOBSERVED = {
    "tests/test_memory_eval.py": (
        "test_the_vault_witness_of_the_unbounded_baseline_reproduces_its_quoted_"
        "numbers", "committed"),
}


def _before():
    return S.read_inventory(S.BEFORE)


def _body_of(path: Path, func: str) -> str:
    """The source of a top-level node, DECORATORS INCLUDED.

    The slice starts at the node's first decorator (or at its `def` when it has
    none), and that is load-bearing rather than decorative:
    `skip_sites` below parses this text and walks `decorator_list` to find a marker,
    so a slice that began at `def` would hand the parser a function with no
    decorators and a re-added `@pytest.mark.skipif` would be invisible. #2637's
    first review named the slice; its second review named the other half of the same
    gap — the slice alone bought nothing while the reader was a substring search for
    `pytest.skip`, a needle that cannot match either marker spelling
    (`pytest.mark.skipif` has `.mark.` where the call has nothing). Both halves are
    needed, so both are pinned here and in
    `test_the_skip_needle_sees_a_marker_and_leaves_an_assertion_alone`.
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "".join(src.splitlines(keepends=True)[start - 1:node.end_lineno])
    raise AssertionError(f"{func} is not a top-level node of {path}")


def test_no_converted_site_keeps_a_live_state_conditional_skip():
    """Clause 1 and clause 3's `no `pytest.skip`/`skipif` survives at that node`.

    Read out of the AST, not out of a grep: a marker above the name, an inline
    `pytest.skip`, and an `importorskip` all fail this, including one re-added under
    a different condition, and a node that was renamed out of the table fails the
    loop. The reader is `skip_sites`, shared with
    `tests/test_intel_pipeline_body.py`'s witness-node check, because the two nodes
    forbid the same thing at two surfaces and a second spelling of "what is a skip"
    is a second thing to get wrong.
    """
    for rel, (func, _shape) in (list(CONVERTED.items()) + list(EXTRA_CONVERTED.items())
                                + list(CONVERTED_UNOBSERVED.items())):
        path = ROOT / rel
        assert path.exists(), f"{rel} is gone; {func} has to be tracked or removed"
        body = _body_of(path, func)
        assert body, f"{func} has no readable body in {rel}"
        sites = S.skip_sites(body)
        assert not sites, (
            f"{rel}::{func} can still report itself skipped: {sites} — the whole of "
            "#2637 is that a live-state-conditional skip spends headroom to assert "
            "nothing, and it costs the suite one of 40 slots whether it is spelled "
            "as a call or as a marker")


#: One entry per marker spelling a round could actually write at a converted node, as
#: parse-only source: `skip_sites` reads these, nothing imports them, so the bodies
#: need no real names in scope. Each is a case the OLD needle could not see — the
#: substring it grepped for, `pytest.skip`, appears in none of these strings, because
#: a marker puts `.mark.` where a call puts nothing.
_MARKER_PROBES: dict[str, str] = {
    "skipif carrying the condition": (
        '@pytest.mark.skipif(NODE is None, reason="no entity-vector index here")\n'
        "def test_a_skipif_marker():\n    assert 1 == 1\n"),
    "bare skip carrying the reason": (
        '@pytest.mark.skip(reason="an absent fork is reported skipped, not passed")\n'
        "def test_a_bare_skip_marker():\n    assert 1 == 1\n"),
    # The binding itself (`_SKIP_UNLESS_NODE = pytest.mark.skipif(...)`) lives at
    # module level, OUTSIDE any node's body, so the probe carries only the use: the
    # reader's one look at a name-bound marker is the decorator's text, which is why
    # the rule is "does the decorator say skip" and not "does it resolve to a marker".
    "marker bound to a module-level name": (
        "@_SKIP_UNLESS_NODE\n"
        "def test_a_bound_marker():\n    assert 1 == 1\n"),
}

#: The two inline-call spellings, one of which the old needle did catch.
_CALL_PROBES: dict[str, str] = {
    "inline pytest.skip": (
        "def test_an_inline_call():\n"
        "    if NODE is None:\n"
        '        pytest.skip("no trajectories dir on this machine")\n'
        "    assert 1 == 1\n"),
    "importorskip": (
        "def test_an_importorskip():\n"
        '    pytest.importorskip("camplus")\n'
        "    assert 1 == 1\n"),
}

#: Source that must NOT read as a skip. The first node is the load-bearing one:
#: clause 4 requires the ~12 nodes in `tests/test_trajectory_extraction.py` that
#: assert `_live_buckets` raises `pytest.skip.Exception`, and a needle that flagged
#: that assertion would buy skip headroom by reddening the only coverage of a guard
#: this item may not touch. The last node is the shape clause 3 calls out by name: a
#: guard quietly replaced by a bare `return` is not a skip, and is caught instead by
#: `test_each_converted_node_earns_its_pass_in_whichever_shape_it_claims`.
_NOT_SKIP_PROBE = (
    # The real shape of `ABSENCE_EDGES` in tests/test_trajectory_extraction.py: a
    # parametrize table about skips, ids and all. Reading a decorator's ARGUMENTS for
    # the word skip would redden on this, and this is the coverage of the guard
    # clause 4 forbids touching — which is why the rule reads the decorator's NAME.
    "@pytest.mark.parametrize('guard,root', ABSENCE_EDGES,\n"
    "                       ids=['corpus/skips-on-absence', 'store/skips-on-absence'])\n"
    "def test_a_table_of_skip_reasons_is_not_a_skip(guard, root):\n"
    "    assert guard is not root\n"
    "@pytest.mark.parametrize('x', [1, 2])\n"
    "def test_asserting_about_someone_elses_skip(x):\n"
    "    with pytest.raises(pytest.skip.Exception) as caught:\n"
    "        guard_under_test()\n"
    '    assert "guard" in str(caught.value)\n'
    "def test_a_red_is_not_a_skip():\n"
    "    if NODE is None:\n"
    '        pytest.fail("absent data fails loudly here")\n'
    "    assert 1 == 1\n"
    "def test_a_plain_pass():\n"
    "    assert 1 == 1\n"
    "def test_a_bare_return_is_not_a_skip_either():\n"
    "    if NODE is None:\n"
    "        return\n"
    "    assert 1 == 1\n")


def test_the_skip_needle_sees_a_marker_and_leaves_an_assertion_alone():
    """The positive control #2637's second review demanded of its own needle.

    A reader that can only say "no skip" is worth nothing until it is shown able to
    say "skip", and the one this round shipped on `SM_20261013_173548` was worse than
    useless: it read as passing at all six converted nodes — which are genuinely
    clean — while the substring it grepped for appears in none of the marker
    spellings, so clause 3's skipif half was unpinned at every one of them. Nothing
    else in this file could have caught that; a probe of the reader is the only test
    that fails when the reader is blind. Both directions, over the parse-only source
    above:

    * every marker spelling is SEEN, and each probe is shown to hold none of the
      substring the old test searched for — the proof the new reader is strictly
      stronger and not merely different;
    * both inline-call spellings are SEEN;
    * `pytest.raises(pytest.skip.Exception)`, a `parametrize` table whose ids and
      reasons are all about skips, `pytest.fail`, a plain pass and a bare `return` are
      NOT seen. The first two are what keep clause 4 intact: asserting that someone
      else's guard skips — and tabulating which guard skips on which absent root — is
      coverage OF that guard, and ~12 real nodes read that way on purpose. A reader
      that flagged them would buy headroom by reddening a skip this item is forbidden
      to remove.
    """
    for name, probe in _MARKER_PROBES.items():
        assert "pytest.skip" not in probe, (
            f"the {name} probe now contains the substring `pytest.skip`, so the old "
            "test would have caught it too and this node no longer proves the new "
            "reader can see a marker")
        sites = S.skip_sites(probe)
        assert len(sites) == 1, (
            f"{name}: expected the marker to be seen exactly once, got {sites} — "
            "`pytest.mark.skipif` and `pytest.mark.skip` contain no substring "
            "`pytest.skip`, which is how clause 3's skipif half read as passing on "
            "round SM_20261013_173548")

    for name, probe in _CALL_PROBES.items():
        sites = S.skip_sites(probe)
        assert len(sites) == 1, f"{name}: not seen: {sites}"

    seen = S.skip_sites(_NOT_SKIP_PROBE)
    assert seen == [], (
        f"an assertion about a skip, a red, a pass or a bare return was read as a "
        f"skip: {seen} — clause 4 needs the `pytest.raises(pytest.skip.Exception)` "
        "nodes to keep pinning `_live_buckets`, so flagging them would buy headroom "
        "by destroying the coverage of a guard this item may not touch")


def test_each_converted_node_earns_its_pass_in_whichever_shape_it_claims():
    """A pass must be an assertion, in either shape.

    * `branch`: the absent-data arm holds an `assert` and then `return`s — the
      shape #2636's witness fix set, and required wherever the artifact is too
      live to commit. `return` alone would be a skip wearing an assert's clothes.
    * `committed`: no absent state exists, so the node must carry a top-level
      `assert` that runs unconditionally.

    The shape is declared per node above, so a future edit cannot quietly degrade
    a `branch` node into a bare `return` or a `committed` node into a vacuous one.
    """
    for rel, (func, shape) in (list(CONVERTED.items()) + list(EXTRA_CONVERTED.items())
                               + list(CONVERTED_UNOBSERVED.items())):
        body = _body_of(ROOT / rel, func)
        tree = ast.parse(body.strip())
        if shape == "branch":
            # The assert has to be IN the absent arm. Walking the whole `If` would also
            # accept `if absent: return / else: <asserts>`, which is the same node
            # reporting nothing when the data is gone — #2637's first review named that
            # gap. So: an `if not <something>` whose own body asserts and returns.
            hit = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                   and isinstance(n.test, ast.UnaryOp)
                   and isinstance(n.test.op, ast.Not)
                   and any(isinstance(x, ast.Assert) for st in n.body
                           for x in ast.walk(st))
                   and any(isinstance(x, ast.Return) for st in n.body
                           for x in ast.walk(st))]
            assert hit, (
                f"{rel}::{func} claims the branch shape but has no absent arm that "
                "asserts and then returns, so one data state reports nothing")
        elif shape == "committed":
            fn = tree.body[0]
            assert isinstance(fn, ast.FunctionDef), (
                f"{rel}::{func} did not parse to a function")
            n_asserts = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assert))
            assert n_asserts >= 2, (
                f"{rel}::{func} claims the committed shape but carries "
                f"{n_asserts} assertions, so the committed artifact is being "
                "read without being checked")
        else:  # pragma: no cover - table guard
            raise AssertionError(f"unknown shape {shape!r} for {rel}")


def test_the_absent_state_really_passes_when_the_data_is_gone(tmp_path, monkeypatch):
    """Drive the absent branch for real, rather than trusting the parse above.

    Both converted data-existence guards are re-pointed at an empty directory and
    the node must return cleanly. If an absent branch raised, or fell through into
    the present-state assertions, this is the node that says so.
    """
    from workers.sources import automod_regression as R
    import tests.test_automod_regression as T
    import tests.test_eval_scorer as E
    import app.paths as P

    monkeypatch.setattr(R, "NOISE_PATH", tmp_path / "absent-noise.json")
    T.test_the_recorded_noise_floor_is_what_the_code_expects()

    monkeypatch.setattr(P, "EVAL_BASELINES_DIR", tmp_path / "absent-baselines")
    E.test_a_run_record_declares_whether_it_matched_production(
        tmp_path=tmp_path / "inner")


def test_the_committed_before_inventory_is_the_measurement_not_a_transcription():
    """The before-artifact is data, and its numbers are read out of it.

    Pinned so the artifact cannot be quietly edited to make a later delta look
    better: the total it records, the machine-fact share, and the exact set of
    converted sites it counts are each asserted from the file's own rows.
    """
    rows = S.read_inventory(S.BEFORE)
    assert rows, f"{S.BEFORE} is empty"
    assert S.total(rows) == S.BEFORE_TOTAL, (
        f"{S.BEFORE} totals {S.total(rows)}, not the pinned {S.BEFORE_TOTAL} — "
        "an inventory artifact is evidence, so it is never edited to flatter a "
        "delta; re-run the probe and say so in the commit")
    assert S.total(S.machine_rows(rows)) == S.BEFORE_MACHINE, (
        f"machine-fact share moved: {S.total(S.machine_rows(rows))} vs pinned "
        f"{S.BEFORE_MACHINE}")
    # `rows` above is the raw dict stream; `S.live_rows` consumes the normalized
    # (path, reason, n) tuples `read_inventory` returns, and reading the dicts here
    # asked the classifier a question it cannot answer — which is how this node,
    # written to catch a transcription, itself went blind to one.
    observed = {r[0] for r in S.live_rows(S.read_inventory(S.BEFORE))}
    for rel in list(CONVERTED) + list(EXTRA_CONVERTED):
        if rel in NOT_OBSERVED_ON_THIS_BOX:
            continue
        assert rel in observed, (
            f"{rel} was not in the before-inventory's live-state population, so "
            "listing it as converted is unfalsifiable")


def test_the_converted_sites_are_rows_of_the_committed_inventory():
    """Clause 3's `names each before-inventory row this round converted`, read back
    out of the artifact instead of out of a table nothing checks.

    `S.CONVERTED` names a conversion by the `(path, reason)` row the measurement
    recorded, because a reason is the one field that survives a conversion — a line
    number does not. This node makes that naming cost something: every entry has to
    match exactly one committed row, and every path it names has to be a file this
    file's own table attributes to a NODE. A table nobody can falsify is the failure
    mode #2637 keeps producing, and the dead copy of it that sat in
    `tests/skip_inventory.py` unquoted for two rounds is the mild version.

    The two `test_trajectory_extraction.py` conversions are absent from
    `S.CONVERTED`, and that absence is the point: their recorded reason is the text
    the surviving `_live_buckets` guard also emits, so the artifact cannot tell them
    apart from a skip clause 4 forbids removing. They are pinned structurally, by
    `test_no_converted_site_keeps_a_live_state_conditional_skip`, which reads the
    node rather than the run.
    """
    rows = S.read_inventory(S.BEFORE)
    assert rows, f"{S.BEFORE} is empty"
    for rel, prefix in S.CONVERTED:
        hits = [r for r in rows if r[0] == rel and r[1].startswith(prefix)]
        assert len(hits) == 1, (
            f"{rel} has {len(hits)} before-inventory rows whose reason starts "
            f"{prefix!r}: the conversion this round claims cannot be tied to one "
            "measured skip, which is what makes 'I converted it' checkable")
        assert rel in CONVERTED, (
            f"{rel} is a row of the artifact and not a key of this file's table, so "
            "no node here pins what became of it")
    # The other direction: every path this round claims as converted is either an
    # artifact row or an explicitly unobserved one — never a file someone made up.
    for rel in CONVERTED:
        if rel in NOT_OBSERVED_ON_THIS_BOX:
            continue
        assert any(r[0] == rel for r in rows), (
            f"{rel} is claimed as converted and appears in no row of the "
            "before-inventory")


def test_no_machine_fact_skip_was_deleted_to_buy_headroom():
    """Clause 2, checked in the direction the failure actually happens.

    Clause 2 states the before/after comparison. Here it is enforced at the
    cheaper and harder-to-evade end: the machine-fact population the before
    inventory recorded is pinned by path, and this diff touches none of those
    files — so headroom cannot have come from deleting a voice-model, CAM++,
    chromium or HF-cache skip. If a later round edits one of those files this
    assertion names it as the one that has to be justified.
    """
    rows = S.read_inventory(S.BEFORE)
    machine_paths = {r[0] for r in S.machine_rows(rows)}
    assert machine_paths, "no machine-fact skips classified at all: the "  \
        "classifier is too narrow and clause 2 would be vacuous"
    changed = {str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py")
               if S.git_touches(ROOT, p)}
    assert not (changed & machine_paths), (
        f"this diff edits files whose skips the before-inventory classed as "
        f"machine facts: {sorted(changed & machine_paths)}")


AFTER_MACHINE = S.FIXTURES / "skip_inventory_after_machine_20261013.jsonl"


def test_every_machine_fact_skip_still_reports_the_same_named_skip():
    """Clause 2's before/after half, measured — scoped to the machine-fact files.

    The after-inventory is a gate-shape `-rs` run over exactly the five files the
    before-inventory's machine-fact population names (`INV_TAG=mfixed`, round
    home, same mark expr and `-n 8 --dist loadfile`), because clause 2's claim is
    about those skips and a whole-suite re-run is a different measurement that
    also drifts on live state. Two assertions, in the direction that matters:

    * no machine-fact `(path, reason)` pair was lost or renamed — that is the
      only way headroom could have been bought dishonestly;
    * the machine-fact instance count did not fall.

    The scoped run legitimately reports MORE pairs than the full suite did: under
    `--dist loadfile` with five files, two `test_voice_wake_and_turn.py` nodes
    surface with their own reason (`turn-1-mini model not present`, `streaming
    model not present`) instead of being attributed to a coarser skip, and both
    are machine facts. A drop or a rename is what fails here.

    The fold is a `Counter` sum, not a dict keyed by `(path, reason)`. An inventory
    is one row per pytest summary line, and the same node emits the same reason on
    more than one line — `tests/test_voice_wake_and_turn.py` records
    `smart-turn model not present` twice (`n` 3 and 1) and
    `tests/test_voice_streaming_asr.py` records `parakeet-realtime-eou-120m not
    present` five times — so a dict comprehension kept the LAST row's `n` and threw
    the rest away: it compared 10 against 10 while both artifacts really total 21
    instances, the review's finding on round SM_20261013_173548. A real loss of
    instances could then sit inside a passing run. `read_inventory` preserves the
    rows precisely so they can be summed; summing is what this node is for.
    """
    import collections

    assert AFTER_MACHINE.exists(), (
        f"{AFTER_MACHINE} missing — clause 2 asks for an after-inventory, and it "
        "is produced by running the probe, never skipped")
    before = S.read_inventory(S.BEFORE)
    after = S.read_inventory(AFTER_MACHINE)

    def _fold(rows):
        out: collections.Counter = collections.Counter()
        for p, rs, n in rows:
            if S.is_machine_fact(rs):
                out[(p, rs)] += n
        return out

    mb, ma = _fold(before), _fold(after)
    assert mb, "the before-inventory classified no machine-fact skips at all"
    # Summed, both artifacts really carry the whole pinned machine-fact population.
    # This is the line that the last-wins dict made false, and it is the difference
    # between "no pair disappeared" and "no instances went missing".
    assert sum(mb.values()) == S.BEFORE_MACHINE, (
        f"the before-inventory's machine-fact rows sum to {sum(mb.values())}, not the "
        f"pinned {S.BEFORE_MACHINE} — the fold lost rows, or the artifact moved")
    # 7 distinct `(path, reason)` pairs carry those 21 instances. Pinned as a number
    # because the pair set is what the two claims below are taken over: a re-worded
    # reason splits a pair into two and silently empties `lost` while instances move,
    # so the count is checked as well as the totals.
    assert len(mb) == 7, (
        f"the machine-fact population is {len(mb)} distinct (path, reason) pairs "
        f"carrying {sum(mb.values())} instances, not the 7 pairs and "
        f"{S.BEFORE_MACHINE} instances the before-inventory measured — a renamed "
        "reason splits a pair, and the drop-pair claim below would pass on the way")
    lost = sorted(set(mb) - set(ma))
    assert not lost, (
        f"machine-fact skips no longer report the same named skip: {lost} — "
        "deleting or re-wording one to buy skip headroom is clause 2's failure")
    assert sum(ma.values()) >= sum(mb.values()), (
        f"machine-fact instances fell from {sum(mb.values())} to "
        f"{sum(ma.values())} without any pair disappearing")
    shrank = sorted((pair, mb[pair], ma.get(pair, 0)) for pair in mb
                    if ma.get(pair, 0) < mb[pair])
    assert not shrank, (
        f"machine-fact skip pairs reporting fewer instances than before: {shrank} — "
        "a rename short enough to keep the pair's path but lose its reason is how a "
        "machine fact disappears from an inventory without being deleted")


#: The machine-fact skip population per file, read out of the before-inventory on
#: 2026-10-13 and committed here as a literal. Clause 4's "machine-fact skips are
#: untouched" is only checkable if the population has a number attached to it: a
#: re-run could otherwise lose a voice-model skip and still show "no machine-fact
#: file in the diff", because the count itself was nowhere pinned.
MACHINE_FACT_FILES = {
    "tests/test_desktop_module.py": 2,
    "tests/test_speaker_id_backend.py": 3,
    "tests/test_voice_asr_hotwords.py": 2,
    "tests/test_voice_streaming_asr.py": 5,
    "tests/test_voice_wake_and_turn.py": 9,
}


def test_the_machine_fact_population_is_a_closed_committed_set():
    """Clause 4's first half: the machine-fact files, by name and by instance count.

    Two assertions that meet in the middle. The committed before-inventory folds to
    exactly `MACHINE_FACT_FILES` — so the classifier cannot be widened later to
    re-label a live-state skip as a machine fact and then "convert" it, and cannot
    be narrowed to drop a real machine fact, without moving this literal. And this
    round's diff names none of those five files, which is the direction headroom
    could actually have been bought dishonestly.
    """
    import collections
    rows = S.read_inventory(S.BEFORE)
    seen: collections.Counter = collections.Counter()
    for p, reason, n in rows:
        if S.is_machine_fact(reason):
            seen[p] += n
    assert dict(seen) == MACHINE_FACT_FILES, (
        f"the committed before-inventory folds to {dict(sorted(seen.items()))}, not the "
        f"pinned {MACHINE_FACT_FILES} — either a machine fact was re-classified or the "
        "artifact moved, and #2637's headroom claim rests on this population")
    assert sum(MACHINE_FACT_FILES.values()) == S.BEFORE_MACHINE, (
        f"{sum(MACHINE_FACT_FILES.values())} vs the pinned machine share "
        f"{S.BEFORE_MACHINE}")
    # The other direction — "this diff names none of these files" — is
    # `test_no_machine_fact_skip_was_deleted_to_buy_headroom`'s, and is deliberately
    # not repeated here: one node owns the diff claim, so a later round that has to
    # justify editing a voice-model file is told so once, by name.


def test_the_skips_a_ruling_put_outside_this_item_still_carry_their_reason_text():
    """Clause 4's second half, read out of the source rather than trusted.

    Two skips survive this round because a ruling says they must, and each survives
    WITH its reason text: `tests/test_qmd_fork_landing.py:617`, whose own docstring
    states that an absent fork is reported skipped rather than passed, and the
    `_live_buckets` guard whose skip ~12 nodes assert with
    `pytest.raises(pytest.skip.Exception)`. Deleting either to buy headroom would
    turn a measured fact into a pass, so the reason string is what is pinned here.
    """
    fork = _body_of(ROOT / "tests/test_qmd_fork_landing.py",
                    "test_the_live_fork_still_declares_the_pair_the_outer_tree_names")
    assert "pytest.skip" in fork, (
        "the qmd-fork node no longer skips an absent fork — its docstring says an "
        "unrun cross-tree half must be reported skipped, never passed")
    assert "is not on disk here, so the fork's package.json" in fork, (
        "the qmd-fork skip lost the reason text that says WHY it is a skip")

    live_buckets = _body_of(ROOT / "tests/test_trajectory_extraction.py", "_live_buckets")
    assert 'pytest.skip("no trajectories dir on this machine")' in live_buckets, (
        "`_live_buckets` no longer skips with that exact reason: ~12 nodes in this file "
        "assert it raises `pytest.skip.Exception`, so re-wording or removing it is a "
        "silent change to ~12 other nodes")


def test_the_pinned_before_total_is_the_ceiling_the_rung_uses():
    """Clause 3's `pinned as BEFORE_TOTAL == 40 against the ceiling`.

    Pinned twice over, because the two halves can move apart: `S.BEFORE_TOTAL` is a
    constant, and the ceiling is read out of `scripts/automod/gate.py`'s shipped
    bytes, not out of the imported object — so a future bump of the ceiling with the
    constant left alone goes red here rather than quietly making the before-artifact
    look like it had headroom. The row total is re-read from the artifact too, so the
    three cannot disagree: #2637's headline is that the standing population sat
    EXACTLY on the rail, which is only a headline if both numbers are this one number.
    """
    src = (ROOT / "scripts" / "automod" / "gate.py").read_text(encoding="utf-8")
    ceiling = re.findall(r"^PYTEST_MAX_SKIPPED = (\d+)$", src, re.M)
    assert ceiling == ["40"], (
        f"gate.py should assign PYTEST_MAX_SKIPPED = 40 exactly once, got {ceiling}")
    rows = S.read_inventory(S.BEFORE)
    assert S.BEFORE_TOTAL == 40
    assert S.total(rows) == S.BEFORE_TOTAL == int(ceiling[0]), (
        f"the before-inventory totals {S.total(rows)} against a pinned "
        f"{S.BEFORE_TOTAL} and a ceiling of {ceiling[0]} — #2637's claim is that these "
        "were the same number, and it needs a re-measure, not an edit")
    assert S.BEFORE_MACHINE < S.BEFORE_TOTAL, (
        "every recorded skip would be a machine fact, leaving nothing this item "
        "could have converted")


#: One throwaway repo, four files, one claim each — see `_round_repo`.
_ROUND_REPO_FILES = ("untouched.py", "committed-half.py",
                     "worktree-half.py", "added-untracked.py")


def _round_repo(tmp_path: Path) -> tuple[Path, str]:
    """A throwaway repo shaped like a graded round: `main`, a branch, one commit.

    Returns `(repo, base_sha)`. The SHAPE is the point. Every claim in this file of
    the form "this diff touches none of those files" is graded by the gate's `tests`
    rung in a worktree whose round diff is already COMMITTED, so the control that
    pins the reader has to be in that shape too — a committed change measured from a
    merge-base, not an uncommitted edit measured against `HEAD`.

    The four files are the four things the reader can get wrong:
    `untouched.py` is in the base commit and nowhere else (a reader that says yes to
    everything is caught here); `committed-half.py` is edited by the round's own
    commit, so it is invisible to `git diff --name-only HEAD` and visible only to one
    read from the merge-base; `worktree-half.py` is edited but uncommitted, which is
    `round_base`'s "diffing the WORKTREE against it also covers the uncommitted half"
    claim; `added-untracked.py` was never `git add`ed, which is what
    `changed_paths`' second git call — `ls-files --others` — exists for.

    Every git call carries its own identity, skips hooks, and is handed no config
    file, because a positive control that only builds when a global `user.email`
    happens to exist is one that quietly stops controlling anything the moment it
    runs in the gate's isolated round home.
    """
    import os
    import subprocess

    repo = tmp_path / "round-repo"
    repo.mkdir()

    def git(*args: str) -> str:
        r = subprocess.run(
            ["git", "-C", str(repo),
             "-c", "user.name=skip-inventory",
             "-c", "user.email=skip-inventory@localhost",
             "-c", "commit.gpgsign=false", *args],
            capture_output=True, text=True, check=False,
            env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                 "GIT_CONFIG_SYSTEM": os.devnull, "GIT_TERMINAL_PROMPT": "0"})
        if r.returncode != 0:
            raise AssertionError(
                f"git {' '.join(args)} failed in the fixture repo: "
                f"{r.stderr.strip()[:200]}")
        return r.stdout.strip()

    git("init", "-q", "-b", "main", ".")
    for name in _ROUND_REPO_FILES[:3]:
        (repo / name).write_text(f"# {name}\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "--no-verify", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "automod/round")
    (repo / "committed-half.py").write_text(
        "# committed-half\n# edited by the round's own commit\n", encoding="utf-8")
    git("add", "committed-half.py")
    git("commit", "-q", "--no-verify", "-m", "the round's own commit")
    # The two states a round is caught in between commits: a dirty tracked file and
    # a brand-new file that has never been staged.
    (repo / "worktree-half.py").write_text(
        "# worktree-half\n# edited, not committed\n", encoding="utf-8")
    (repo / "added-untracked.py").write_text("# never git-added\n", encoding="utf-8")
    return repo, base


def test_the_diff_reader_the_pins_depend_on_answers_in_both_directions(tmp_path):
    """A guard that reads the diff must be shown able to say YES — on a tree it
    builds, not on the tree it happens to be sitting in.

    #2637's first review caught the class: an empty diff also passes, so the node was
    given a positive control. The control it shipped with was this pin file itself,
    on the argument that it "cannot leave this round's diff" — which holds only while
    the round that ADDS the file is open. From the moment #2637 merged, every clean
    checkout of this file has an empty tracked diff against its own merge-base: the
    gate's base probe, live HEAD, and any round that does not touch it. So the node
    #2637 wrote to prove its reader can say YES has answered only NO ever since, and
    failed the base probe of round `SM_20261013_182959` at base `be413a55` — #2667.

    A control has to be built to be a control. The synthetic round repo above is the
    same shape the rung grades in, and it is checked in both directions: three files
    the reader must name, one it must not, and the base it must measure from. What a
    regression to each half of the reader now costs:

      * `round_base` falling back to `HEAD` (the bug the module docstring names) ->
        `committed-half.py` reads untouched;
      * `changed_paths` losing `ls-files --others` -> `added-untracked.py` reads
        untouched;
      * `git_touches` answering one constant for every input -> the `untouched.py`
        assertion, or one of the three positive ones, goes red first.

    The two claims about the REAL tree are the ones that stay true on any tree, and
    they are the ones the pins below actually rest on: a base must be nameable from
    here, and a file clause 4 forbids must not read as touched.
    """
    repo, base = _round_repo(tmp_path)
    assert S.round_base(repo) == base, (
        f"the reader measures this round's diff from {S.round_base(repo)!r}, not from "
        f"the merge-base with main ({base}) — every 'this diff touches X' assertion in "
        "this file is then taken against the wrong tree")
    for name in ("committed-half.py", "worktree-half.py", "added-untracked.py"):
        assert S.git_touches(repo, repo / name), (
            f"the reader cannot see {name}, which this round's repo changed — an empty "
            "diff looks exactly like this from the inside, which is the failure "
            "#2637's first review named")
    assert not S.git_touches(repo, repo / "untouched.py"), (
        "the reader claims every file is touched, so 'this diff touches no "
        "machine-fact file' would be vacuous")
    assert S.round_base(ROOT) != "HEAD", (
        f"no base could be named from {ROOT}, so the diff would be read against "
        "HEAD and name nothing")
    assert not S.git_touches(ROOT, ROOT / "tests" / "test_voice_wake_and_turn.py"), (
        f"{ROOT} has a diff against its base that names a machine-fact file, so "
        "`test_no_machine_fact_skip_was_deleted_to_buy_headroom` cannot be read as a "
        "clean claim right now")


#: Parse-only source carrying the one marker shape clause 2 names that `skip_sites`
#: deliberately cannot see: an xfail spends no slot of the 40, so the shared reader
#: excludes it by design. The probe is a STRING, so parsing this file never sees it as
#: a decorator — which is exactly the asymmetry the node below leans on, and why it
#: checks the extractor on the probe first.
_XFAIL_MARKER_PROBE = (
    '@pytest.mark.xfail(reason="an xfail spends none of the 40")\n'
    "def test_an_xfail_marker():\n    assert 1 == 1\n")

#: The floor on this file's test-node count: the 12 nodes it carried at base
#: `be413a55` (measured by collecting it: 11 passed, 1 failed), plus the one #2667
#: adds here. Deletion is the one route to clause 1's green that no skip check can
#: see, because a deleted node skips nothing — it simply stops asking.
MIN_TEST_NODES_AT_2667 = 13


def _marker_leaf_names(src: str) -> list[str]:
    """The LAST name component of every decorator applied to a function in `src`.

    `@pytest.mark.xfail(...)` -> `xfail`; `@pytest.mark.skipif(c, reason=...)` ->
    `skipif`; `@_needs_node` -> `_needs_node`. Only the name, never the arguments:
    the same reason `skip_inventory._decorator_name` drops them — a marker's reason is
    a string of prose ABOUT a skip, and this file's own probe tables are full of that
    prose. Reads decorators on functions, so a string constant that merely shows a
    marker is not seen by it.
    """
    out: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                target = d.func if isinstance(d, ast.Call) else d
                out.append(target.id if isinstance(target, ast.Name)
                           else getattr(target, "attr", ""))
    return out


def test_clause_2_green_cannot_have_come_from_a_skip_or_a_deleted_node():
    """Clause 2: the named node PASSES — it is not skipped, xfailed, or removed.

    The two cheap ways to a green run of one node are a marker on it and a deletion
    beside it, and both are invisible to the gate's arithmetic in the way that
    matters: an xfail spends none of the 40 skip slots, and a deleted node reports no
    skip at all. So this node pins clause 2 over this file, which is the only file the
    item's clause can be satisfied or faked in.

    Three checks. `S.skip_sites` over the WHOLE file — the one definition of "what is
    a skip" this suite has, and it sees a marker and an inline call alike, while
    leaving this file's `pytest.raises(pytest.skip.Exception)` prose alone, so the
    check costs no coverage. A marker-NAME sweep for the xfail half, shown able to say
    YES on the probe above before it is allowed to say NO about this file. And a floor
    on the node count, which is the direction a `skip_sites` sweep cannot walk.
    """
    seen = _marker_leaf_names(_XFAIL_MARKER_PROBE)
    assert seen == ["xfail"], (
        f"the marker reader sees {seen} on a source whose only decorator is "
        "`@pytest.mark.xfail`, so its silence about this file would be blindness "
        "rather than an all-clear")

    src = (ROOT / "tests" / "test_skip_inventory.py").read_text(encoding="utf-8")
    sites = S.skip_sites(src)
    assert sites == [], (
        f"a node in this file can still report itself skipped: {sites} — clause 2 is "
        "that the fix makes the named tests pass, and a skip is not a pass")
    marked = [nm for nm in _marker_leaf_names(src) if "skip" in nm or "xfail" in nm]
    assert not marked, (
        f"a node in this file carries {marked}, which is clause 2's exclusion by "
        "another spelling")
    nodes = [n.name for n in ast.parse(src).body
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    assert len(nodes) >= MIN_TEST_NODES_AT_2667, (
        f"this file now holds {len(nodes)} test nodes, under the "
        f"{MIN_TEST_NODES_AT_2667} it held at #2667 — deleting a node is how a green "
        "run is bought without skipping anything")
    assert ("test_the_diff_reader_the_pins_depend_on_answers_in_both_directions"
            in nodes), (
        "the node clause 1 names is gone from this file, so clause 1 has been "
        "answered by removing the question")


# ── #2920 — the non-convertible ruling has to live in the code, not in a closed item ──
#
# Four skip sites were ruled permanently non-convertible, and the ruling existed only in
# the owed entry of a `done` item, which is where a ruling goes to become invisible: the
# next round reads a closed item's tail as history and re-files the question for a person.
# The nodes below keep the comment honest against the tree, in the direction that matters
# — a comment that claims a skip call, a count or a citation the code does not carry goes
# red. The process boundary is the one every prose claim in this file crosses: a comment in
# `skip_inventory.py` and a `pytest.skip` several files away are read by different people at
# different times, and nothing but these nodes connects them.

RULED = {
    "tests/test_qmd_fork_landing.py": "is not on disk here",
    "tests/test_trajectory_extraction.py": "no trajectories dir on this machine",
    "tests/_live_data.py": "absent at",
    "tests/test_retrieval_seed_anchoring.py": "no entity-vector index",
}


def _module_comment_text() -> str:
    """Every comment and docstring line in `skip_inventory.py`, joined.

    Read as text rather than as `__doc__` because the ruling is a `#` block, and a test that
    only read docstrings would pass on a module that lost it.
    """
    src = (ROOT / "tests" / "skip_inventory.py").read_text(encoding="utf-8")
    keep = [ln for ln in src.splitlines() if ln.lstrip().startswith("#")]
    return "\n".join(keep) + "\n" + _docstring_text(src)


def _docstring_text(src: str) -> str:
    """All docstrings in the module, concatenated — the second half of where a claim lives."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(n)
            if d:
                out.append(d)
    return "\n".join(out)


def _skip_call_literals(rel_path: str) -> list[str]:
    """Every string literal passed to a `pytest.skip(...)` call in a file.

    Literals of the CALL, found on the AST — so an anchor matching one of these is matched
    against the text of a skip call and not against the same words appearing in a docstring,
    a comment or an assert message elsewhere in the file.
    """
    src = (ROOT / rel_path).read_text(encoding="utf-8")
    found: list[str] = []
    for n in ast.walk(ast.parse(src)):
        if not isinstance(n, ast.Call):
            continue
        name = ast.unparse(n.func)
        if name.rsplit(".", 1)[-1] != "skip":
            continue
        for part in ast.walk(n):
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                found.append(part.value)
    return found


def test_the_ruling_block_names_the_four_sites_and_withholds_them_as_headroom():
    """Clause 1: the word and the four filenames sit together, and headroom is refused.

    Checked as one block rather than four independent greps, because the failure this guards
    is the partial one — a later edit keeps the word `non-convertible` somewhere and drops a
    file, or lists the four as ordinary commentary where a round reads them as fair game.
    """
    text = _module_comment_text()
    assert "non-convertible" in text.lower(), \
        "the module carries no ruling: #2920's four sites are back to being visible only in " \
        "a closed item's owed entry"
    window = text.lower().index("non-convertible")
    nearby = text[max(0, window - 1200): window + 3000]
    for rel in RULED:
        assert rel in nearby, f"{rel} is not named inside the non-convertible block"
    assert "headroom" in nearby.lower(), \
        "the block does not say a round may not count these as skip headroom, which is the " \
        "half of the ruling a round actually acts on"
    assert "never" in nearby.lower(), \
        "the block does not state the withholding as permanent"


def test_each_ruled_site_is_anchored_by_the_text_of_its_skip_call():
    """Clause 2: each anchor is a live skip-call argument in the file it names.

    Matched against the literals of `pytest.skip(...)` calls found on the AST, not against the
    file's text — so an anchor cannot satisfy the block by sharing words with a docstring, a
    comment, or an assert message, and a reworded skip goes red here instead of leaving the
    block pointing at a call that no longer says what it was quoted as saying.
    """
    text = _module_comment_text()
    for rel, anchor in RULED.items():
        assert anchor in text, \
            f"the block does not carry the anchor `{anchor}` for {rel}"
        literals = _skip_call_literals(rel)
        assert literals, f"{rel} exposes no pytest.skip(...) string literal at all"
        assert any(anchor in lit for lit in literals), (
            f"`{anchor}` is not an argument of any pytest.skip call in {rel}; the block "
            f"anchors it to text its skip no longer carries — anchors are skip-call text "
            f"precisely so a drifted line number cannot be the whole citation")


def test_each_ruled_site_carries_the_reason_the_ruling_rests_on():
    """Clause 3: four reasons, each with the specific thing that makes it unconvertible.

    The substrings are the load-bearing nouns of each reason, not whole sentences: a block
    that named the four files and skipped the *why* would leave a round able to re-litigate
    each one, which is what the item is asking this file to stop.
    """
    text = _module_comment_text()
    want = [
        ("skipped", "qmd's docstring ruling that an absent fork is reported skipped"),
        ("never", "…and never as a pass"),
        ("never-touch", "the qmd fork is a tree no round may touch"),
        ("machine state", "_live_buckets skips because of what this box holds"),
        ("1377", "_live_data.py is #1377's step-3 branch"),
        ("step-3", "…named as step 3, the branch that stays a skip"),
        ("SM_20260923_034250", "the round that promoted with kg.sqlite missing"),
        ("entity-vector", "the seed-anchoring node needs the vector index"),
        ("Qwen3", "…and the Qwen3 weights"),
        ("ranking", "…because the assertion it guards is a ranking"),
    ]
    missing = [why for needle, why in want if needle not in text]
    assert not missing, (
        f"the block states the four files but not: {missing} — a ruling without its reasons "
        f"gets re-argued by the next round that wants a conversion")


def test_the_ruling_counts_agree_with_the_tree_and_the_stale_ones_are_gone():
    """Clause 4: the numbers in the comment are the numbers on disk, and two old ones died.

    Re-measured here rather than quoted, because the figure already sitting in the module was
    wrong in its attribution, and the number was right for the wrong reason: there are 8
    `pytest.raises(pytest.skip.Exception)` sites, and not one of them asserts `_live_buckets`
    — they assert the live-data guards `tests/_live_data.py` exports (`require_live_data`,
    `require_live_volume`). So the block gives that guard no count at all and names its 2
    consumers instead, which is a claim that cannot drift, and the stale `:331`/`:163` reserve
    citations became their marker-reason texts.
    """
    text = _module_comment_text()
    traj = (ROOT / "tests" / "test_trajectory_extraction.py").read_text(encoding="utf-8")
    assert traj.count("pytest.raises(pytest.skip.Exception)") == 8, (
        "the count this node pins as the truth has moved; re-read the file and update both "
        "this figure and the block's sentence about what those sites guard")
    assert len(_live_bucket_consumers()) == 2, \
        "`_live_buckets` gained or lost a consumer; the block names them, so update both"
    # The attribution, checked rather than trusted. Each raise-site reaches a guard of
    # `tests/_live_data.py` — the parametrized ones through an `ABSENCE_EDGES` row naming a
    # module guard, the rest by calling one directly — so the chain is followed through module
    # functions instead of looking for a call the test body does not contain.
    holders = _raises_skip_functions()
    assert holders, "no site asserts a live-data guard's skip any more; the exclusion this " \
                    "comment documents has nothing left to exclude"
    unreachable = [f for f in holders if not _reaches_live_data_guard(f)]
    assert not unreachable, (
        f"these holders of a `pytest.raises(pytest.skip.Exception)` site reach no guard that "
        f"`tests/_live_data.py` exports: {unreachable} — the block attributes the 8 to those "
        f"guards, and that attribution is the thing this node exists to keep true")
    bucket = [f for f in holders if "_live_buckets" in _calls_in(f)]
    assert not bucket, (
        f"{bucket} now asserts `_live_buckets`'s skip, so the block's disclaimer that the 8 "
        f"are not that guard's count is false and both files need the same edit")
    for guard in ("require_live_data", "require_live_volume"):
        assert guard in text, \
            f"the block stopped naming the guard `{guard}` that the 8 sites assert"

    assert "~12 nodes" not in text, \
        "the stale '~12 nodes' claim is back — it attributed 8 assertion sites to a guard " \
        "that has 2 consumers and whose guard no `pytest.raises` site covers"
    assert "test_turn_options.py:331" not in text and \
        "test_worker_capability_allowlist.py:163" not in text, \
        "the drifted reserve line citations are back; the reserve is cited by marker reason"

    # #3197: the two figures the block carried for the skip rail. Both were true while the
    # rail was the flat `PYTEST_MAX_SKIPPED`, and `d085bf2c` (#3143) derived the rail from
    # the ledger instead, so the standing reading and the rail figure were both false on the
    # tree that inherited them — and the trigger they carried, "a round actually refused by
    # the 40 rail", could not have fired. Banned over the whole file, not just the block:
    # the same stale sentence sat on two dated lines outside it, and those now name the rail
    # as the flat figure it was then rather than as a standing ceiling.
    whole = (ROOT / "tests" / "skip_inventory.py").read_text(encoding="utf-8")
    assert "35-36" not in whole, \
        "the stale '35-36' standing reading is back: the bound is derived by " \
        "`_skip_ceiling`, so any standing count quoted in prose ages into a trigger that " \
        "cannot fire"
    assert "of 40" not in whole, \
        "a ceiling-of-40 figure is back in the module: 40 is the FLOOR of the derived " \
        "bound, not a ceiling any round is refused against"

    for reason in ("recording mode only (LLOYD_RECORD_TURN_OPTIONS=1)",
                   "recording mode only (LLOYD_RECORD_CAPABILITY_BASELINE=1)"):
        assert reason in text, f"the reserve lost its reason text `{reason}`"
        assert (ROOT / "tests" / ("test_turn_options.py"
                                  if "TURN_OPTIONS" in reason
                                  else "test_worker_capability_allowlist.py")
                ).read_text(encoding="utf-8").count(reason) == 1, \
            f"`{reason}` is no longer a single live marker reason in the file it cites"


def _live_bucket_consumers() -> list[str]:
    """Test functions that call `_live_buckets(...)`, by name, not by line.

    Named because the block names them: a line number for these two drifted once already
    between triage and this commit, which is exactly the failure clause 4 is about.
    """
    src = (ROOT / "tests" / "test_trajectory_extraction.py").read_text(encoding="utf-8")
    out = []
    for f in [n for n in ast.walk(ast.parse(src))
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if any(isinstance(c, ast.Call)
               and getattr(c.func, "id", getattr(c.func, "attr", "")) == "_live_buckets"
               for c in ast.walk(f)):
            out.append(f.name)
    return sorted(out)


def test_the_ruling_is_a_comment_and_the_module_surface_is_untouched():
    """Clause 5: recording a ruling must not become a behaviour change.

    The module's public surface is pinned by name and value, and the shipped floor constant
    is read out of `gate.py`'s bytes rather than the imported object, so a bump that left
    `BEFORE_TOTAL` alone would be red here. Also asserts every changed line of the diff (when
    one is available) is a comment — the boundary this clause exists to guard is the one
    between a note a round reads and a rule the leg enforces.
    """
    # Clause 5's other half, added by #3197: a round that wants more skip room has to be
    # pointed at a person, not at the constant. `#2214` is the item that rules raising the
    # figure human-only, and its note lives in `gate.py`'s comment on the floor, so the block
    # in `skip_inventory.py` has to carry the citation itself — a round that opened only this
    # file is exactly the round that would otherwise reach for the number.
    flat = " ".join(_non_convertible_block().split())
    assert "#2214" in flat, \
        "the block no longer cites #2214, so a round reading only skip_inventory.py has " \
        "no note that raising the floor is not its ruling to make"
    assert re.search(r"PYTEST_MAX_SKIPPED.{0,200}human ruling", flat), \
        "the block lost the sentence tying `PYTEST_MAX_SKIPPED` to a human ruling; the " \
        "citation alone does not say WHICH figure #2214 reserves"

    assert S.BEFORE_TOTAL == 40 and S.BEFORE_MACHINE == 21
    for fn in ("is_machine_fact", "read_inventory", "skip_sites", "machine_rows",
               "live_rows", "total", "changed_paths", "git_touches", "round_base"):
        assert callable(getattr(S, fn)), f"{fn} disappeared from the module's surface"
    for name in ("FIXTURES", "BEFORE", "CONVERTED"):
        assert getattr(S, name) is not None, f"{name} is gone"

    gate = (ROOT / "scripts" / "automod" / "gate.py").read_text(encoding="utf-8")
    assert re.findall(r"^PYTEST_MAX_SKIPPED = (\d+)$", gate, re.M) == ["40"], \
        "the floor constant moved while this ruling was being written; the block's bound " \
        "and test_the_pinned_before_total_is_the_ceiling_the_rung_uses both need the same " \
        "number, and #2214 reserves raising it to a person"

    same, why = _executable_ast_matches_round_base()
    assert same, (
        f"#2920 is meant to be a comment-only change and it is not: {why}")


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Drop every docstring node, so a reworded docstring cannot read as a behaviour change.

    Comments are already invisible to the parser; docstrings are not, and one of this item's
    corrections sits inside `skip_sites`' docstring.
    """
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            body.pop(0)
    return tree


def _executable_ast_matches_round_base() -> tuple[bool, str]:
    """(True, "") when `tests/skip_inventory.py` parses identically to the round base.

    Compared on the AST with docstrings stripped, not on lines: a line-level "is it a comment"
    test passes on a change that edited a string literal's innards and fails on one that
    reflowed prose, and the question clause 5 actually asks is whether anything executable
    moved. Uses `S.round_base`, the same base resolution the rails themselves use, so the node
    compares against the tree the round was opened from whether or not the work is committed.
    """
    rel = "tests/skip_inventory.py"
    base = S.round_base(ROOT)
    try:
        before = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:{rel}"],
                                capture_output=True, text=True, check=False)
    except OSError as exc:
        return False, f"could not read the base copy of {rel}: {exc}"
    if before.returncode != 0:
        return False, f"`git show {base}:{rel}` exited {before.returncode}: {before.stderr[:120]}"
    after = (ROOT / rel).read_text(encoding="utf-8")
    try:
        old = ast.dump(_strip_docstrings(ast.parse(before.stdout)))
        new = ast.dump(_strip_docstrings(ast.parse(after)))
    except SyntaxError as exc:
        return False, f"{rel} no longer parses: {exc}"
    if old == new:
        return True, ""
    return False, ("the AST differs from the round base, so an executable statement changed "
                   "alongside the ruling")


def _raises_skip_functions() -> list[str]:
    """Test functions holding a `pytest.raises(pytest.skip.Exception)` site, by name.

    The attribution check needs the enclosing function, not a line count: 8 is the number of
    sites and every one of them is a `require_live_data` guard, which is only visible per
    function.
    """
    src = (ROOT / "tests" / "test_trajectory_extraction.py").read_text(encoding="utf-8")
    out = []
    for f in [n for n in ast.walk(ast.parse(src))
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for c in ast.walk(f):
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "raises" and c.args
                    and "skip" in ast.unparse(c.args[0])):
                out.append(f.name)
                break
    return sorted(out)


def _live_data_guard_names() -> set[str]:
    """The guard names `tests/_live_data.py` exports — the population the 8 sites draw from."""
    ld = (ROOT / "tests" / "_live_data.py").read_text(encoding="utf-8")
    return set(re.findall(r"^def (require[a-z_]*)\(", ld, re.M))


def _calls_in(func_name: str) -> list[str]:
    """Every callee name inside one test function of `test_trajectory_extraction.py`."""
    return _TRAJ_CALLS.get(func_name, [])


def _module_calls() -> dict[str, list[str]]:
    """`{function name: names it reaches}` for every function in the trajectory test module.

    A name reached through a `@pytest.mark.parametrize` argument counts, because that is how a
    parametrized guard reaches its subject: `ABSENCE_EDGES` passes a guard as a *value*
    (`pytest.param(test_the_live_corpus_emits_one_file_per_sequence_pattern, "corpus")`), so the
    function under test is an input to the node and appears nowhere in its body as a call.
    """
    src = (ROOT / "tests" / "test_trajectory_extraction.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    consts = _module_constants(tree)
    out: dict[str, list[str]] = {}
    for f in [n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        reached = [getattr(c.func, "id", getattr(c.func, "attr", ""))
                   for c in ast.walk(f) if isinstance(c, ast.Call)]
        for d in f.decorator_list:
            for n in ast.walk(d):
                if isinstance(n, ast.Name):
                    reached.append(n.id)
                    reached += sorted(consts.get(n.id, ()))
        out[f.name] = reached
    return out


def _module_constants(tree: ast.AST) -> dict[str, set[str]]:
    """`{constant name: names its literal mentions}` for module-level list/tuple bindings.

    `ABSENCE_EDGES` is a table of `pytest.param(<guard function>, <root>)`, so the guards a
    parametrized node can be handed live in that binding and in nowhere else; without this the
    reachability walk stops at the table's name and reports a real attribution as false.
    """
    out: dict[str, set[str]] = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
    return out


_TRAJ_CALLS = _module_calls()


def _reaches_live_data_guard(func_name: str) -> bool:
    """True when a function reaches a guard `tests/_live_data.py` exports, through module fns.

    The reachability matters: the parametrized raise-sites do not call `require_live_data`
    themselves — an `ABSENCE_EDGES` row names a module guard, and that guard is what calls
    one — so a direct-call check reports a true attribution as false. That is the difference
    between this node and a grep.
    """
    exported = _live_data_guard_names()
    seen: set[str] = {func_name}
    frontier = [func_name]
    while frontier:
        name = frontier.pop()
        callees = set(_TRAJ_CALLS.get(name, ()))
        if callees & exported:
            return True
        for nxt in callees - seen:
            if nxt in _TRAJ_CALLS:
                seen.add(nxt)
                frontier.append(nxt)
    return False


# ── #3197: the block describes the skip rail as a mechanism, never as a figure ─────
#
# Two processes hold the two halves of the claim these nodes pin. A round's implement
# session reads the comment in `tests/skip_inventory.py`; the gate's detached `tests` rung
# applies the bound in `scripts/automod/gate.py`, deriving it from that machine's rounds
# ledger when the rung runs. Nothing at run time connects the two texts, so these nodes
# re-read `gate.py`'s bytes from disk: a rename of `_skip_ceiling` or a re-spelling of
# `PYTEST_SKIP_HEADROOM` has to turn a node red here rather than leave a round comparing
# its own skip count against a sentence that stopped describing the rail.
_BLOCK_TITLE = "four sites ruled NON-CONVERTIBLE"
_MAX_BOUND = re.compile(
    r"max\(\s*PYTEST_MAX_SKIPPED,\s*base_skips\s*\+\s*PYTEST_SKIP_HEADROOM\s*\)")
_FLOOR_CLAIM = re.compile(r"PYTEST_MAX_SKIPPED`?\s+is the FLOOR", re.I)
_RAIL_MENTION = re.compile(r"\b(ceiling|rail|floor|standing|headroom)\b", re.I)
#: Figures a line about the bound may still carry, because each names a thing rather than a
#: reading of the suite: an item reference, a round id, an env flag's value, #1377's step
#: number, or a date.
_CITED_NOT_MEASURED = re.compile(r"#\d+|SM_\d{8}_\d{6}|=\d+|step-\d|\b20\d\d-\d\d-\d\d\b")


def _non_convertible_block() -> str:
    """The NON-CONVERTIBLE comment block of `tests/skip_inventory.py`, located by its title.

    Read from the title line down the contiguous `#` run rather than from a line range: the
    block's own reserve says its line citations had already drifted twice, and a pin written
    against a range silently grades whatever text has since moved into it.
    """
    src = (ROOT / "tests" / "skip_inventory.py").read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(src) if _BLOCK_TITLE in ln), None)
    assert start is not None, \
        f"the block titled {_BLOCK_TITLE!r} is gone from tests/skip_inventory.py, and the " \
        f"ruling every other node in this file reads is gone with it"
    end = start
    while end + 1 < len(src) and src[end + 1].lstrip().startswith("#"):
        end += 1
    return "\n".join(src[start:end + 1])


def _gate_source() -> str:
    """`scripts/automod/gate.py` as bytes on disk — the far side of the seam these nodes span."""
    return (ROOT / "scripts" / "automod" / "gate.py").read_text(encoding="utf-8")


def test_the_block_names_the_derived_bound_in_both_trees_and_no_figure_in_either():
    """Clauses 2 and 3: the rail is a mechanism, spelled identically on both sides, unquantified.

    Three names have to appear in the block AND in `gate.py`'s text, and that pairing is what
    makes a rename of the ceiling turn this node red instead of quietly leaving a round with a
    stale sentence: the block can only satisfy both sides while `gate.py` still spells the
    mechanism the way the block quotes it. The figure test is per line rather than per file
    because the block legitimately carries numbers that are not readings of the suite —
    `#1377`'s step number, and the id of the round that promoted with a missing `kg.sqlite` —
    and the figure that has to stay out is the one that tells a round how much room it has.
    """
    block = _non_convertible_block()
    gate = _gate_source()
    for name in ("_skip_ceiling", "PYTEST_SKIP_HEADROOM", "PYTEST_MAX_SKIPPED"):
        assert name in block, \
            f"the block no longer names `{name}`, so it no longer describes the bound the " \
            f"tests rung actually applies"
        assert name in gate, \
            f"`{name}` is not in scripts/automod/gate.py's text, though the block still " \
            f"quotes it as the mechanism — one of the two is now lying"
    assert _MAX_BOUND.search(block), \
        "the block does not spell the bound as max(PYTEST_MAX_SKIPPED, base_skips + " \
        "PYTEST_SKIP_HEADROOM), which is the only spelling that says which side is the floor"
    assert _FLOOR_CLAIM.search(block), \
        "the block stopped naming PYTEST_MAX_SKIPPED as the FLOOR of that max — the " \
        "attribution #3197 exists to hold, and it was PYTEST_MAX_SKIPPED read as the ceiling " \
        "that made the stale figures believable"
    assert not re.search(r"under the ceiling \(`?PYTEST_MAX_SKIPPED", block), \
        "the pre-#3143 attribution is back verbatim: the standing count sitting under " \
        "PYTEST_MAX_SKIPPED as if that constant were the ceiling"
    quantified = [ln.strip() for ln in block.splitlines()
                  if _RAIL_MENTION.search(ln)
                  and re.search(r"\d", _CITED_NOT_MEASURED.sub("", ln))]
    assert not quantified, \
        f"these lines state the bound or a standing reading as a number: {quantified} — " \
        f"the bound is derived from the tree and the ledger, so any figure written here " \
        f"is false the next time either moves"
    for gone in ("35-36", "of 40", "the 40 rail"):
        assert gone not in block, f"the stale figure {gone!r} is back inside the block"


def test_the_reserve_waits_for_a_refusal_by_the_derived_bound_not_by_a_fixed_rail():
    """Clause 4: what would justify taking the reserve, in the units the rung refuses in.

    The trigger is a round that failed nothing and was refused anyway, over the DERIVED bound:
    that is the only event which shows the room is gone. A named figure as the trigger was
    worse than no trigger, because the derived bound made a refusal at the floor arithmetically
    impossible, and a condition that cannot fire reads to the next round as one that was tested
    and failed. Pinned both ways — positively as the zero-failure refusal over the derived
    ceiling with the two reason texts and the on-record absence, negatively as the fixed rail.
    """
    block = _non_convertible_block()
    trigger = " ".join(ln.lstrip("# ").strip() for ln in block.splitlines()
                       if "untaken" in ln or "refused" in ln)
    assert trigger, "the reserve's trigger sentence is gone from the block entirely"
    low = trigger.lower()
    assert "untaken" in low, \
        "the block no longer rules the two `recording mode only` skips untaken, which is " \
        "the half of the ruling a round acts on"
    assert "zero failures" in low, \
        "the trigger lost its zero-failure condition: a round that failed something is " \
        "refused for its failures, not for the rail"
    assert "derived ceiling" in low, \
        "the trigger no longer names the derived ceiling as what refuses, so the next " \
        "reader goes looking for a fixed rail the rung stopped applying in #3143"
    assert "40 rail" not in block.lower(), \
        "'40 rail' is back as the trigger, and #3143 derived the bound — a refusal there " \
        "cannot happen, which is exactly how the old sentence came to read as a check that " \
        "had been run and failed"
    assert "no such refusal is on record" in " ".join(block.split()), \
        "the block lost its on-record claim. If a round HAS been refused over the derived " \
        "ceiling with zero failures since this was written, that is the trigger to take the " \
        "reserve: take it and rewrite the sentence, do not delete the claim to get green"
    for reason in ("recording mode only (LLOYD_RECORD_TURN_OPTIONS=1)",
                   "recording mode only (LLOYD_RECORD_CAPABILITY_BASELINE=1)"):
        assert reason in block, \
            f"the reserve is no longer findable inside the block by its marker reason " \
            f"`{reason}`, which is how it is located without a line number"
