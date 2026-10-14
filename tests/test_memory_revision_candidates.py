"""#2717 clauses 1-3 — the revision-triggered candidate set, read-only.

ReHarness's backward-revision step only makes sense if you can name the lines that
were retrieved for a run that went badly. Step 1 of the item is the honest version of
that: one read-only report over an uptake window that lists, per retrieved item, the
runs it appeared in and what the dispute classifier said in each of them — with a
denominator printed beside every rate, and the single-dispute rule expressed as data
rather than as prose a later agent has to remember.

Two properties hold this together, and each is tested at the boundary it lives at:

* **The attribution is uptake's, not a second one.** Every fixture below builds its
  table by calling `app.uptake.build_uptake_table` on real `Turn` objects, so the rows
  the script consumes are the rows the shipped instrument emits — presence ids,
  `presence_source`, dispute counts and all. A test also reads the script's source and
  refuses any session-tree walk in it: the moment this file starts opening transcripts
  itself, its numbers stop being comparable with the tables the nightly writer already
  cites (`tests/test_uptake.py` owns the instrument; this file owns the reading of it).
* **The process boundary is crossed, not described.** One node runs the script as a
  subprocess over a table file and reads the dated artifact back off disk, because the
  artifact's whole purpose is to be opened by a job that is not this one.

Nothing here edits memory, and nothing here asks a model anything: the classifier is
never called, because the table arrives with its verdicts already recorded.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import uptake  # noqa: E402

SCRIPT = REPO / "scripts" / "memory_revision_candidates.py"


def _load_module():
    """Load the CLI script by path: it has no package and nothing imports it."""
    spec = importlib.util.spec_from_file_location("mrc2717", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mrc():
    """One loaded instance of the script, function-addressable for `monkeypatch`."""
    return _load_module()


# --------------------------------------------------------------------- fixtures
# Synthetic but built BY the instrument: the shape of a row is uptake's business, and
# a hand-written dict here would let the script pass against a table that does not
# exist. `s1#1` and `s1#2` are disputed; `s1#3` and `s2#1` were never cue-screened, so
# they carry no verdict at all — which is a different thing from "verified clean".

def _turns() -> list[uptake.Turn]:
    return [
        uptake.Turn(session="s1", ts=None, user_text="do it the big way", ordinal=1,
                    vault_context=["Alpha rule", "Beta rule"], injections_seen=1),
        uptake.Turn(session="s1", ts=None, user_text="no, the other way", ordinal=2,
                    vault_context=["Alpha rule", "Beta rule", "Gamma rule"],
                    injections_seen=1),
        uptake.Turn(session="s1", ts=None, user_text="carry on", ordinal=3,
                    vault_context=["Beta rule", "Gamma rule"], injections_seen=1),
        uptake.Turn(session="s2", ts=None, user_text="unrelated ask", ordinal=1,
                    vault_context=["Gamma rule"], injections_seen=0),
    ]


def _entries() -> list[uptake.Entry]:
    return [uptake.Entry(source="lloyd/USER.md",
                         text="Prefer the larger batch when the queue looks long"),
            uptake.Entry(source="lloyd/MEMORY.md",
                         text="Alarms are written where humans read")]


#: `s1#3` is the case a count cannot carry: the cue screen reached it and the engine
#: answered NOTHING, so its verdict is `None`. `s2#1` is the other blank — the screen never
#: reached it, so it is in the map not at all. Both are blank to a reader of the verdict
#: list, which is the inversion the per-run columns exist to prevent.
DISPUTED = {"s1#1": True, "s1#2": True, "s1#3": None}
CLEAN = {"s1#1": False, "s1#2": False}


def _table(flags: dict, *, with_map: bool = True) -> dict:
    """One real uptake table over the fixture corpus with `flags` as its verdicts.

    Built BY `uptake.build_uptake_table`, so the shape — including the persisted
    `uptake.DISPUTE_MAP_KEY` map a probe run now writes (#2717 clause 1) — is the
    instrument's and not this file's guess. `with_map=False` strips that key, which is what
    all three committed tables look like: every one of them predates it.
    """
    table = uptake.build_uptake_table(turns=_turns(), dispute_flags=flags,
                                      memory_entries=_entries())
    if not with_map:
        table.pop(uptake.DISPUTE_MAP_KEY, None)
    return table


def _by_item(rows) -> dict:
    return {r["item"]: r for r in rows}


# ------------------------------------------------------------- clause 1: the join

def test_a_prefetched_note_is_listed_with_every_run_it_appeared_in_and_its_verdict(mrc):
    """Clause 1 (prefetched half): the artifact names, for one retrieved note, each run
    uptake attributed it to and the classifier's verdict in that run.

    The run list is *the row's own list*, not a re-derivation: `presence_turn_ids` is
    what `app/uptake.py:2067` emits from the persisted `<vault-context>` block, and the
    only thing this script adds is the join to the verdicts that produced the table.
    Alpha is the two-dispute note; Gamma the one-dispute note; `s2#1` is in Gamma's list
    with a null verdict because the cue screen never sent it to the classifier.
    """
    table = _table(DISPUTED)
    alpha = _by_item(mrc.candidates(table, DISPUTED))["note:Alpha rule"]
    row = next(e for e in table["entries"] if e["entry"] == "note:Alpha rule")

    assert alpha["present_in_runs"] == row["presence_turn_ids"], alpha
    assert alpha["present_in_runs"] == ["s1#1", "s1#2"], alpha
    assert [(v["run"], v["verdict"]) for v in alpha["verdicts"]] == [
        ("s1#1", True), ("s1#2", True)], alpha
    assert alpha["presence_source"] == row["presence_source"], alpha
    assert alpha["kind"] == "note"

    gamma = _by_item(mrc.candidates(table, DISPUTED))["note:Gamma rule"]
    assert [(v["run"], v["verdict"]) for v in gamma["verdicts"]] == [
        ("s1#2", True), ("s1#3", None), ("s2#1", None)], gamma
    # The list above holds TWO kinds of blank and the columns tell them apart: `s1#3` was
    # screened and the classifier answered nothing, `s2#1` was never screened. Lumped, a run
    # nobody looked at reads as a run that cleared the note.
    assert gamma["disputed_run_ids"] == ["s1#2"], gamma["disputed_run_ids"]
    assert gamma["unresolved_run_ids"] == ["s1#3"], gamma["unresolved_run_ids"]
    assert gamma["unscreened_run_ids"] == ["s2#1"], gamma["unscreened_run_ids"]


def test_a_memory_index_line_is_listed_with_its_runs_and_its_upper_bound_flag(mrc):
    """Clause 1 (loaded-index half): a MEMORY.md/USER.md bullet is listed too, and the
    artifact says out loud that its presence is by construction.

    Uptake emits no per-turn ids for these rows — `presence_source` is
    `always_in_force:system_prompt` because the line is in every prompt — so "the runs it
    was present in" is the whole window, which is exactly why the item distrusts these
    rows and why the flag has to travel with the candidate rather than living in a doc
    somewhere.
    """
    table = _table(DISPUTED)
    line = next(r for r in mrc.candidates(table, DISPUTED)
                if r["item"].startswith("memory:lloyd/USER.md:"))
    assert line["kind"] == "memory_entry"
    assert line["presence_source"] == uptake.ALWAYS_IN_FORCE, line
    assert line["attribution_upper_bound"] is True, line
    assert line["present_in_turns"] == 4, line          # every turn in the window
    assert line["disputed_run_ids"] == ["s1#1", "s1#2"], line


def test_the_script_reads_uptake_rows_and_never_walks_the_session_tree_itself(mrc):
    """Clause 1's other half, graded on the source: no second attribution.

    `app/uptake.py` already owns the corpus walk, the turn key and the presence
    evidence; a fresh scan of `sessions/` here would produce numbers that could not be
    compared with the tables the nightly writer cites, and it would be a second thing to
    keep correct. So the module must import uptake, must call `build_uptake_table` for a
    refreshed window rather than reimplementing the join, and must contain no filesystem
    enumeration of its own.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))

    imported = {a.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for a in node.names}
    imported |= {a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                 for a in node.names}
    imported |= {node.module for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert any(m.rsplit(".", 1)[-1] == "uptake" for m in imported), sorted(imported)

    # Clause 2's mechanism, graded on the source. `human_turns`, `build_uptake_table` and
    # `classify_dispute` are how the PROBE decides what was disputed. Called from here they
    # would be a second measurement of the same thing, taken after the model or the prompt
    # moved, printed in the same column as the first — and the engine grading its own output,
    # which is #2269's objection. The first version of this file called `classify_dispute`
    # with the wrong arity and died on it; the fix is not a corrected call, it is no call.
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    banned = {"human_turns", "build_uptake_table", "classify_dispute",
              "classify_dispute_raw", "has_dispute_cue", "verdicts_by_turn"}
    assert not (called & banned), (
        f"the artifact re-measures the instrument here: {sorted(called & banned)}")

    # Enumeration is banned because it is how a second attribution would start. One named
    # exception: `newest_table_name` lists `uptake-*.json` to pick a table to READ, and the
    # corpus-root string check below still applies, so the exception cannot widen.
    forbidden = {"rglob", "iterdir", "listdir", "walk"}
    walked = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in forbidden}
    walked |= {node.func.id for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id in forbidden}
    assert not walked, f"second attribution would start here: {sorted(walked)}"
    # No string constant names the corpus root either: an enumeration spelled as
    # `Path(root).joinpath("sessions")` would clear the call check above.
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [s for s in strings if "sessions" in s], strings
    # The one directory listing the script may do is of the probe's own artifacts.
    globs = [n.args[0].value for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "glob" and n.args and isinstance(n.args[0], ast.Constant)]
    assert all(g.startswith("uptake-") for g in globs), globs


# ------------------------------------------------- clause 2: denominators and zeros

def _walk_rates(node, path="artifact"):
    """Every value the artifact presents as a rate, with the path it was reached by.

    Two shapes count: anything under the top-level `rates` mapping, and any key named
    exactly `rate` anywhere in the tree. A substring match on 'rate' is not usable — it
    catches `generated_on`, and a walk that reports a date as a bare number is a walk that
    will be relaxed away rather than fixed.
    """
    if isinstance(node, dict):
        found = [(f"{path}.{key}", key, value) for key, value in node.items()
                 if key == "rate"]
        for key, value in node.items():
            if key == "rates":
                found.extend((f"{path}.{key}.{name}", name, sub)
                             for name, sub in value.items())
            else:
                found.extend(_walk_rates(value, f"{path}.{key}"))
        return found
    if isinstance(node, list):
        return [hit for i, item in enumerate(node)
                for hit in _walk_rates(item, f"{path}[{i}]")]
    return []


def test_every_rate_in_the_artifact_arrives_with_its_denominator(mrc):
    """Clause 2: a rate without a denominator is the failure this instrument has a
    history of — #552's own report had to be re-read because a share was quoted over a
    set nobody had named.

    Mechanical rather than sampled: every key containing 'rate' anywhere in the artifact,
    candidates included, must hold a mapping with `numerator`, `denominator` and `value`,
    and none of them may be a bare float.
    """
    artifact = mrc.build_artifact(_table(DISPUTED), window_days=30,
                                  as_of="2026-10-14", source_table="fixture")
    seen = _walk_rates(artifact)
    assert seen, "no rate printed at all — the walk below would be vacuous"
    for where, key, value in seen:
        assert isinstance(value, dict), f"{where} is a bare number: {value!r}"
        assert {"numerator", "denominator", "value"} <= set(value), (where, value)
        assert isinstance(value["denominator"], int), (where, value)
    rates = artifact["rates"]
    # 3 prefetched notes (Alpha, Beta, Gamma) + 2 index lines = the 5 retrieved rows.
    assert rates["disputed_retrieved_rows"] == {
        "numerator": 5, "denominator": 5, "value": 1.0}, rates
    # Alpha (2), Beta (2) and both index lines (2) clear the floor; Gamma's single flag
    # is the one row the rule excludes.
    assert rates["rows_with_two_or_more_disputed_runs"] == {
        "numerator": 4, "denominator": 5, "value": 0.8}, rates


def test_a_window_with_no_dispute_prints_the_zero_finding_and_its_denominator(mrc):
    """Clause 2's zero half, and the half that actually matters: an empty table is not a
    finding.

    The item's own step 1 says "if the join yields 0 dispute-flagged items, that is the
    finding, written as such". So the zero case must name the denominator it is zero over —
    5 retrieved rows over this fixture (3 notes, 2 index lines), the 4 window turns, and
    the 2 turns the cue screen actually reached, since
    recall 0.43 is what makes a zero here a lower bound rather than a clean bill of
    health (`tests/test_uptake.py` measures that recall; this prints it).
    """
    artifact = mrc.build_artifact(_table(CLEAN), window_days=30,
                                 as_of="2026-10-14", source_table="fixture")
    # The candidates are still there, all marked ineligible: clause 3's rule is that a
    # thin row is EMITTED and marked, so a zero window must not silently empty the table.
    assert len(artifact["candidates"]) == 5, artifact["candidates"]
    assert not [c for c in artifact["candidates"] if c["eligible"]], artifact["candidates"]
    zero = artifact["zero_finding"]
    assert zero["disputed_retrieved_rows"] == 0, zero
    assert zero["retrieved_rows"] == 5, zero
    assert zero["window_turns"] == 4, zero
    assert zero["screened_turns"] == 2, zero
    assert "lower bound" in zero["statement"].lower(), zero
    assert "0 of 5" in zero["statement"], zero


def test_a_window_with_a_dispute_carries_no_zero_finding(mrc):
    """The control on the node above: a zero finding that always printed would be
    decoration, and a reader could not tell a clean window from a broken report."""
    artifact = mrc.build_artifact(_table(DISPUTED), window_days=30,
                                  as_of="2026-10-14", source_table="fixture")
    assert "zero_finding" not in artifact, artifact["zero_finding"]
    assert artifact["rates"]["disputed_retrieved_rows"]["numerator"] == 5


# ------------------------------------------- clause 3: the single-dispute rule as data

def test_a_note_disputed_in_one_run_is_emitted_marked_ineligible(mrc):
    """Clause 3: 'never narrow on a single dispute' is a field on the row, not a rule in
    prose. The item's pilot died on exactly this — 14 disputed note rows, every one at
    `disputes == 1` — and a number that only exists in a paragraph gets forgotten by the
    next agent that reads the table.

    The row is still EMITTED (ineligible is a marking, not a filter): the count is the
    evidence, and hiding it would make a thin window look like an empty one.
    """
    rows = _by_item(mrc.candidates(_table(DISPUTED)))
    gamma = rows["note:Gamma rule"]
    assert gamma["disputed_run_count"] == 1, gamma
    assert gamma["eligible"] is False, gamma
    assert gamma["eligibility"] == mrc.INELIGIBLE_BELOW_FLOOR, gamma
    assert gamma["min_disputed_runs"] == mrc.MIN_DISPUTED_RUNS == 2, gamma
    assert "single dispute" in gamma["eligibility_reason"], gamma


def test_the_eligibility_floor_is_the_boundary_and_not_an_upper_bound(mrc):
    """The boundary itself: two disputed runs clears the floor, one does not, and a
    disputed-run count is counted over RUNS, so the same turn cannot vote twice.

    `attribution_upper_bound` is checked here too: an always-in-force line reaching the
    floor on the window's own dispute count is not the same evidence as a note disputed in
    two runs it was actually retrieved for, and the row says which it is.
    """
    rows = _by_item(mrc.candidates(_table(DISPUTED)))
    assert rows["note:Alpha rule"]["disputed_run_count"] == 2, rows["note:Alpha rule"]
    assert rows["note:Alpha rule"]["eligible"] is True, rows["note:Alpha rule"]
    assert rows["note:Alpha rule"]["eligibility"] == mrc.ELIGIBLE
    assert rows["note:Alpha rule"]["attribution_upper_bound"] is False

    line = next(r for r in mrc.candidates(_table(DISPUTED))
                if r["kind"] == "memory_entry")
    assert line["disputed_run_count"] == 2 and line["eligible"] is True, line
    assert line["attribution_upper_bound"] is True, line


def test_the_eligibility_summary_is_split_so_an_all_upper_bound_pass_is_visible(mrc):
    """Clause 4's second half: the floor reported per ATTRIBUTION family, because pooled it
    hides the one pattern that has to be visible.

    The pilot's own numbers are the reason. Over the committed window every one of the 14
    disputed prefetched rows carried a SINGLE dispute — so nothing honestly attributed cleared
    the floor — while all 112 `always_in_force` rows showed disputes by construction, since a
    loaded index line is in every prompt. Pooled, that window reads "112 of 117 rows eligible":
    a filter that appears to be working while excluding every row whose evidence is real.

    So the split is built from a window with exactly that pathology — two disputed turns,
    `s1#1` and `s2#1`, which no single note appears in twice — and asserted on both families
    and on the text, where a nightly reader actually looks.
    """
    pathology = {"s1#1": True, "s2#1": True}
    artifact = mrc.build_artifact(_table(pathology), window_days=30,
                                  as_of="2026-10-14", source_table="fixture")
    split = artifact["eligibility_by_attribution"]
    assert set(split) == {"prefetched", "always_in_force"}, split

    prefetched = split["prefetched"]
    assert prefetched["rows"] == 3, prefetched
    assert prefetched["meeting_floor"] == 0, prefetched
    assert prefetched["rate"] == {"numerator": 0, "denominator": 3, "value": 0.0}, prefetched
    assert "per-run" in prefetched["attribution"], prefetched

    loaded = split["always_in_force"]
    assert loaded["rows"] == 2 and loaded["meeting_floor"] == 2, loaded
    assert loaded["rate"] == {"numerator": 2, "denominator": 2, "value": 1.0}, loaded
    assert "upper bound" in loaded["attribution"], loaded

    # The pooled rate would have called this window healthy-ish; the split does not.
    assert artifact["rates"]["rows_with_two_or_more_disputed_runs"]["numerator"] == 2, (
        artifact["rates"])
    rendered = mrc.render(artifact)
    assert "eligible[prefetched]" in rendered and "0/3" in rendered, rendered
    assert "eligible[always_in_force]" in rendered and "2/2" in rendered, rendered

    # And the fixture window, where two notes DO clear the floor: the families differ in kind,
    # not just in numbers, and neither is allowed to be rounded into the other.
    normal = mrc.build_artifact(_table(DISPUTED), window_days=30,
                                as_of="2026-10-14", source_table="fixture")
    assert normal["eligibility_by_attribution"]["prefetched"]["rate"] == {
        "numerator": 2, "denominator": 3, "value": 0.6667}, normal["eligibility_by_attribution"]
    assert normal["eligibility_by_attribution"]["always_in_force"]["meeting_floor"] == 2, (
        normal["eligibility_by_attribution"])


def test_a_note_present_in_untouched_runs_is_counted_and_its_cap_is_reported(mrc):
    """Presence counts what uptake counted, and says when the id list is shorter than the
    count.

    `presence_turn_ids` truncates at 20 by design (`app/uptake.py:2067`), so a note in 25
    runs prints 20 ids while `present_in_turns` stays 25. A reader comparing the two would
    otherwise conclude five runs went missing, so the artifact flags the truncation
    wherever it happens instead of letting the count and the list disagree in silence.
    """
    turns = [uptake.Turn(session="s9", ts=None, user_text=f"ask {i}", ordinal=i + 1,
                         vault_context=["Wide note"], injections_seen=1)
             for i in range(22)]
    table = uptake.build_uptake_table(turns=turns, dispute_flags={"s9#1": True},
                                      memory_entries=[])
    row = next(e for e in table["entries"] if e["entry"] == "note:Wide note")
    assert len(row["presence_turn_ids"]) == 20 and row["present_in_turns"] == 22, row
    wide = _by_item(mrc.candidates(table))["note:Wide note"]
    assert wide["present_in_turns"] == 22, wide
    assert len(wide["present_in_runs"]) == 20, wide
    assert wide["runs_truncated"] is True, wide


# ------------------------------------------------------- the denominators it reports

def test_the_artifact_counts_what_it_left_out_as_well_as_what_it_kept(mrc):
    """Denominator hygiene over the kinds: skills are rows uptake emits and this report
    does not grade, so their count is printed as an exclusion rather than dropped — the
    same rule #552 states for memory entries the entry grammar misses.

    Retrieval depth is also printed as a denominator per item, because step 1's rates are
    meaningless without the window they were taken over: this fixture is 4 turns, 2 of them
    disputed and both of those cue-screened, so every rate below reads `n/4`. The committed
    window this instrument actually ships over is 276 turns — a different denominator, and the
    reason the artifact prints its own rather than inheriting one from prose.
    """
    artifact = mrc.build_artifact(_table(DISPUTED), window_days=30,
                                  as_of="2026-10-14", source_table="fixture")
    assert artifact["window_days"] == 30, artifact
    assert artifact["generated_on"] == "2026-10-14", artifact
    assert artifact["source_table"] == "fixture", artifact
    assert artifact["counts"]["retrieved_rows"] == 5, artifact["counts"]
    assert artifact["counts"]["excluded_rows"]["skill"]["rows"] == 0, artifact["counts"]
    assert artifact["counts"]["excluded_rows"]["skill"]["why"], artifact["counts"]
    assert artifact["rates"]["disputed_turns_over_window"] == {
        "numerator": 2, "denominator": 4, "value": 0.5}, artifact["rates"]
    assert artifact["rates"]["screened_share_of_window"] == {
        "numerator": 2, "denominator": 4, "value": 0.5}, artifact["rates"]


def test_skill_rows_are_excluded_from_the_candidates_and_counted(mrc):
    """A skill row is uptake's, but it is not a retrieved memory line in the sense this
    item's step 1 names, so it must not be silently merged into the candidate set — and
    it must not vanish either: the count lands in `excluded_rows`.
    """
    turns = [uptake.Turn(session="s7", ts=None, user_text="use the skill", ordinal=1,
                         vault_context=[], injected_skills=["demo-skill"],
                         injections_seen=1),
             uptake.Turn(session="s7", ts=None, user_text="wrong", ordinal=2,
                         vault_context=[], injected_skills=["demo-skill"],
                         injections_seen=1)]
    table = uptake.build_uptake_table(turns=turns, dispute_flags={"s7#1": True,
                                                                  "s7#2": True},
                                      memory_entries=[], active_skills=["demo-skill"])
    assert any(e["kind"] == "skill" for e in table["entries"]), table["entries"]
    rows = mrc.candidates(table)
    assert not [r for r in rows if r["kind"] == "skill"], rows
    artifact = mrc.build_artifact(table, window_days=30, as_of="2026-10-14",
                                  source_table="fixture")
    assert artifact["counts"]["excluded_rows"]["skill"]["rows"] == 1, artifact["counts"]


# ------------------------------------------------------------- the CLI boundary

def _write_table(tmp_path: Path, table: dict) -> Path:
    path = tmp_path / "uptake-2026-10-14.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def test_the_cli_reads_a_table_file_and_writes_a_dated_artifact(tmp_path):
    """The process boundary the artifact exists to cross: a nightly job runs this script
    later and opens what it wrote. Run as a subprocess, then read the file back — no
    in-process shortcut, because the reader is a different process on a different day.

    The filename carries the date and the artifact repeats it, because step 1 is a
    dated record of one window: an artifact that overwrites itself cannot be compared
    with the next window, which is how the instrument at `eval/uptake/` works.

    `returncode == 0` is asserted before anything is opened, so a script that dies is
    named as the failure rather than surfacing two statements later as a missing file.
    """
    # `with_map=False` is the shape a committed table actually has: all three predate the
    # persisted map. The CLI still has to produce a readable artifact, and it has to say the
    # per-run column is unavailable instead of leaving it blank — a blank reads as zero.
    table_path = _write_table(tmp_path, _table(DISPUTED, with_map=False))
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--table", str(table_path),
         "--out-dir", str(out), "--as-of", "2026-10-14"],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    written = sorted(p.name for p in out.iterdir())
    assert written == ["memory-revision-candidates-2026-10-14.json"], written
    artifact = json.loads((out / written[0]).read_text(encoding="utf-8"))
    assert artifact["generated_on"] == "2026-10-14", artifact
    assert artifact["source_table"].endswith("uptake-2026-10-14.json"), artifact
    assert len(artifact["candidates"]) == 5, artifact["candidates"]
    # A table file records dispute COUNTS, not per-turn verdicts, so the per-run verdict
    # column has to say it is unavailable rather than guess. The refresh path is what
    # carries verdicts, and the artifact names which of the two produced it.
    assert artifact["verdicts"].startswith("unavailable:"), artifact["verdicts"]
    assert "uptake-2026-10-14" in proc.stdout


def test_the_cli_dry_run_prints_and_writes_nothing(tmp_path):
    """`--dry-run` is the mode the item's 'nothing edits' step-1 promise is kept in, so
    the absence of the file is the assertion, not the presence of a flag."""
    table_path = _write_table(tmp_path, _table(CLEAN))
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--table", str(table_path),
         "--out-dir", str(out), "--as-of", "2026-10-14", "--dry-run"],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert not out.exists(), list(out.iterdir()) if out.exists() else None
    assert "zero" in proc.stdout.lower(), proc.stdout
    assert "0 of 5" in proc.stdout, proc.stdout


def test_a_window_whose_verdicts_were_never_classified_says_so_rather_than_reading_zero(mrc):
    """The second process boundary: the artifact's producer either has the classifier's
    per-turn verdicts or it does not, and "does not" must never read as "found nothing".

    `--table` is the mode that can be run any time at zero cost; the verdicts it cannot have
    are per-turn, because a committed table records a COUNT per entry and the list of turn
    ids only in aggregate. So the artifact carries an `unavailable:` explanation in the
    `verdicts` field of the artifact and of every row, while the counts it really does have
    stay put — presence from the persisted block, disputed count from the table's own
    aggregate. A reader who wants per-run verdict text has to run the probe, and the
    artifact says so instead of quietly leaving the column blank.

    Asserted on both halves: what is missing is named, and what is present is not flattened.
    """
    # The state all three committed tables are in: `corpus` counted the disputes and no
    # per-turn map says which runs they were. Expressed as a map-less table, which is what
    # the instrument actually writes, not as a passed argument.
    artifact = mrc.build_artifact(_table(DISPUTED, with_map=False), window_days=30,
                                  as_of="2026-10-14", source_table="fixture")
    assert artifact["verdicts"].startswith("unavailable:"), artifact["verdicts"]
    alpha = _by_item(artifact["candidates"])["note:Alpha rule"]
    assert alpha["verdicts"].startswith("unavailable:"), alpha
    # The inversion this whole branch exists to prevent, in its worst form: an EMPTY
    # disputed list beside a row the table counts as disputed TWICE. With no map the per-run
    # column cannot be filled, so the artifact prints why instead of letting the blank read
    # as "looked, nothing disputed".
    assert alpha["disputed_run_ids"] == [] and alpha["disputed_run_count"] == 2, alpha
    assert alpha["unscreened_run_ids"] == alpha["present_in_runs"], alpha
    assert artifact["verdict_map"]["available"] is False, artifact["verdict_map"]
    finding = artifact["unavailable_finding"]
    assert finding["table"] == "fixture", finding
    assert "missing measurement" in finding["statement"], finding["statement"]
    assert "UNAVAILABLE" in mrc.render(artifact), mrc.render(artifact)[:400]
    # 2 of the fixture's 4 turns carry Alpha's block, and both were disputed.
    assert alpha["present_in_turns"] == 2, alpha
    assert alpha["disputed_run_count"] == 2, alpha
    assert alpha["eligible"] is True, alpha

    clean = mrc.build_artifact(_table(CLEAN, with_map=False), window_days=30,
                               as_of="2026-10-14", source_table="fixture")
    assert clean["verdicts"].startswith("unavailable:"), clean["verdicts"]
    assert "0 of 5" in clean["zero_finding"]["statement"], clean["zero_finding"]
    assert clean["zero_finding"]["window_turns"] == 4, clean["zero_finding"]


def test_refresh_window_hands_back_the_persisted_map_and_never_one_of_its_own(mrc, tmp_path,
                                                                              monkeypatch):
    """Clause 2's seam, named: `refresh_window` is where the classifier call used to be.

    The refused round's version walked the session store and called
    `uptake.classify_dispute(turn)` for every cue-screened turn. That call raised
    `TypeError: missing a required argument: 'user_text'` — `app/uptake.py:1254` takes
    `(prev_assistant, user_text)`, not a turn — and no node in this file named the function
    that crashed, so the artifact was dead on arrival in production and green in here. The
    replacement then over-corrected into the opposite defect: it returned `(table, {})`,
    `main()` passed that empty dict to `build_artifact` as a verdict map, and the default
    `--days` path printed "available, 0 turns, 0 disputed" over a table whose own map recorded
    two disputed turns. Three properties, all pinned here:

    * it hands back the map the TABLE persisted — read through `read_verdict_map`, never
      computed — so `main()`'s per-run column is the probe's own measurement;
    * a table that persisted no map yields `None`, not `{}`, which is what routes the
      artifact to its `unavailable` finding;
    * and an empty map handed to `build_artifact` as an override cannot read as an available
      one either, so the mis-wire is impossible in the shape rather than merely absent.

    It also names what it looked for when there is nothing to read: a missing corpus is a
    finding a reader can act on, not a `StopIteration` from a glob.
    """
    tables = tmp_path / "uptake"
    tables.mkdir()
    mapped = _table(DISPUTED)                      # writes uptake.DISPUTE_MAP_KEY
    legacy = _table(DISPUTED, with_map=False)      # the shape all three committed tables have
    (tables / "uptake-2026-09-20.json").write_text(json.dumps(legacy), encoding="utf-8")
    (tables / "uptake-2026-10-11.json").write_text(json.dumps(mapped), encoding="utf-8")
    monkeypatch.setattr(mrc.uptake, "UPTAKE_ARTIFACT_DIR", tables)

    assert mrc.newest_table_name() == "uptake-2026-10-11.json"
    assert mrc.resolve_table_path().name == "uptake-2026-10-11.json"

    table, verdicts = mrc.refresh_window(30)
    assert verdicts == dict(DISPUTED), (
        "refresh_window must hand back the map the table persisted, not an empty dict")
    assert table["corpus"]["disputes"] == mapped["corpus"]["disputes"]

    # The map-available branch, through the same arguments main() uses.
    artifact = mrc.build_artifact(table, window_days=30, as_of="2026-10-14",
                                  source_table="uptake-2026-10-11.json", verdicts=verdicts)
    assert artifact["verdicts"] == "per-turn", artifact["verdicts"]
    assert artifact["verdict_map"]["available"] is True, artifact["verdict_map"]
    assert artifact["unavailable_finding"] is None, artifact["unavailable_finding"]
    alpha = _by_item(artifact["candidates"])["note:Alpha rule"]
    assert alpha["disputed_run_ids"] == ["s1#1", "s1#2"], alpha

    # The table that persisted nothing yields None, and the artifact says so out loud.
    legacy_table, legacy_verdicts = mrc.refresh_window(30, table=tables / "uptake-2026-09-20.json")
    assert legacy_verdicts is None, "an absent map is None, never an empty dict"
    legacy_artifact = mrc.build_artifact(legacy_table, window_days=30, as_of="2026-10-14",
                                        source_table="uptake-2026-09-20.json",
                                        verdicts=legacy_verdicts)
    assert legacy_artifact["verdicts"].startswith("unavailable:"), legacy_artifact["verdicts"]
    assert legacy_artifact["verdict_map"]["available"] is False, legacy_artifact["verdict_map"]

    # And the shape that caused the mis-wire is refused on its own: an EMPTY override records
    # no turn, so it cannot be printed as an available map of zero disputes.
    shaped = mrc.build_artifact(table, window_days=30, as_of="2026-10-14",
                               source_table="uptake-2026-10-11.json", verdicts={})
    assert shaped["verdict_map"]["available"] is False, shaped["verdict_map"]
    assert shaped["unavailable_finding"] is not None, shaped["unavailable_finding"]
    assert shaped["candidates"][0]["disputed_run_ids"] == [], shaped["candidates"][0]

    empty = tmp_path / "none-here"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        mrc.newest_table_name(empty)
    assert "uptake-*.json" in str(exc.value) and str(empty) in str(exc.value), exc.value


def test_the_default_cli_reports_the_map_the_table_persisted(mrc, tmp_path, monkeypatch):
    """The path the review found untested: `main()` with NO `--table`.

    This is the invocation a nightly job actually types, and until now no node ran it, which
    is how `refresh_window` could hand `main()` an empty dict and the artifact could print
    "available, 0 turns, 0 disputed" over a table whose own map recorded two disputed turns.
    Asserted on the artifact `main()` writes, not on a call the test assembled: the
    map-available branch has to survive the argument wiring to count.
    """
    tables = tmp_path / "uptake"
    tables.mkdir()
    table = _table(DISPUTED)
    (tables / "uptake-2026-10-11.json").write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setattr(mrc.uptake, "UPTAKE_ARTIFACT_DIR", tables)
    out = tmp_path / "artifacts"

    rc = mrc.main(["--days", "30", "--out-dir", str(out), "--as-of", "2026-10-14"])
    assert rc == 0, "the default CLI must exit 0 on a window it can read"
    artifact = json.loads((out / "memory-revision-candidates-2026-10-14.json").read_text(encoding="utf-8"))

    # The name it prints is the FILE it opened. `live:30d` named a source no process read,
    # and `unavailable_finding.table` would have pointed a reader at a table that does not
    # exist.
    assert artifact["source_table"] == "uptake-2026-10-11.json", artifact["source_table"]
    assert artifact["verdicts"] == "per-turn", artifact["verdicts"]
    assert artifact["verdict_map"] == {
        "available": True, "source_key": uptake.DISPUTE_MAP_KEY, "turns_in_map": 3,
        "disputed_turns": 2, "engine_unanswered": 1, "keyed_on": "turn_id"}, artifact["verdict_map"]
    assert artifact["unavailable_finding"] is None, artifact["unavailable_finding"]
    alpha = _by_item(artifact["candidates"])["note:Alpha rule"]
    # The join the item exists for, arriving through the default path: the runs uptake said
    # the note was present in, and the verdict the probe recorded in each of them.
    assert alpha["present_in_runs"] == ["s1#1", "s1#2"], alpha
    assert alpha["disputed_run_ids"] == ["s1#1", "s1#2"], alpha
    assert alpha["verdicts"] == [{"run": "s1#1", "verdict": True},
                                 {"run": "s1#2", "verdict": True}], alpha
    assert alpha["unscreened_run_ids"] == [] and alpha["unresolved_run_ids"] == [], alpha
    assert alpha["eligible"] is True, alpha


def test_the_default_cli_says_unavailable_and_names_the_table_it_read(mrc, tmp_path,
                                                                     monkeypatch):
    """The same untrodden path over a map-less table: the finding must name the real file.

    Clause 2's `unavailable` requirement is worth nothing if the default invocation reaches a
    different branch than `--table` does, because the default is the one that gets run. Here
    the table on disk has no `uptake.DISPUTE_MAP_KEY`, so the artifact has to refuse to print
    a dispute column, name the file it opened, and keep the counts it really does have.
    """
    tables = tmp_path / "uptake"
    tables.mkdir()
    legacy = _table(DISPUTED, with_map=False)
    (tables / "uptake-2026-09-20.json").write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(mrc.uptake, "UPTAKE_ARTIFACT_DIR", tables)
    out = tmp_path / "artifacts"

    rc = mrc.main(["--days", "30", "--out-dir", str(out), "--as-of", "2026-10-14"])
    assert rc == 0, rc
    artifact = json.loads((out / "memory-revision-candidates-2026-10-14.json").read_text(encoding="utf-8"))

    assert artifact["verdicts"].startswith("unavailable:"), artifact["verdicts"]
    assert artifact["verdict_map"]["available"] is False, artifact["verdict_map"]
    finding = artifact["unavailable_finding"]
    assert finding["table"] == "uptake-2026-09-20.json", finding
    assert "missing measurement" in finding["statement"], finding["statement"]
    alpha = _by_item(artifact["candidates"])["note:Alpha rule"]
    assert alpha["disputed_run_ids"] == [], alpha
    assert alpha["disputed_run_count"] == 2, alpha      # the count it really has, kept
    assert alpha["unscreened_run_ids"] == alpha["present_in_runs"], alpha


def test_the_default_write_lands_under_the_data_root_never_the_checkout(mrc, tmp_path,
                                                                       monkeypatch):
    """The same seam as the grounding script's, because an artifact has one home for a reason.

    Every other CLI node here passed `--out-dir`, so the branch a nightly job actually takes —
    no `--out-dir` — had never written a byte anywhere in a test. A report dropped into the
    checkout is a stray the watchdog has to clean and does not survive a promote, so the
    default is exercised here with the data root pointed at a temp directory (measuring the
    real one would mean writing into it, which is not a test's business).
    """
    from app import paths  # noqa: E404
    tables = tmp_path / "uptake"
    tables.mkdir()
    (tables / "uptake-2026-10-11.json").write_text(json.dumps(_table(DISPUTED)),
                                                   encoding="utf-8")
    monkeypatch.setattr(mrc.uptake, "UPTAKE_ARTIFACT_DIR", tables)
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "data")

    rc = mrc.main(["--days", "30", "--as-of", "2026-10-14"])
    assert rc == 0, rc
    written = (tmp_path / "data" / "eval" / "memory-revision"
               / "memory-revision-candidates-2026-10-14.json")
    assert written.exists(), (mrc.default_out_dir(), list(tmp_path.iterdir()))
    artifact = json.loads(written.read_text(encoding="utf-8"))
    assert artifact["verdicts"] == "per-turn", artifact["verdicts"]
    assert not (Path(mrc.__file__).parents[1] / "eval" / "memory-revision").exists(), \
        "the default write must resolve through DATA_ROOT, not the checkout"
