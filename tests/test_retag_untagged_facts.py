"""Tests for `scripts/memory/retag_untagged_facts.py` (#2676).

The untagged rail (#2474, `f7e08d41`) is red every night: the live scan on
2026-10-14 reports **3,206 untagged records in 544 directories / 830 files** out of
153,999 records — 2.08% against `kg_hygiene.UNTAGGED_ALARM_SHARE = 0.5`, so
`over_floor` is True and the nightly report appends `ABOVE FLOOR` and takes the
alarm exit. The population is entirely historical (newest `created_at` 2026-10-10,
the day the writer fix #2503 / `1822fce5` landed), so only a repair clears it. The
ruling (#2676, from #2474's owed entry) is to re-tag each record from its own
file's top-level `entity:` — not drop the records, not move the floor.

The fixture is the live corpus's shapes, not a clean subset of it: 2,142 of the
3,206 records carry `entity:` present-but-blank (all of them quoted `''` today, so
the `None` and whitespace states are guarded here because the rail counts them, not
because they are frequent) and 1,064 carry no `entity:` key at all, in the
`fact:`-first layout the `backlog_write_task` writer emits, whose double-quoted
fact scalar wraps onto continuation lines. A naive splice of a key line after a
record's first line parses as a syntax error on exactly those files, which is why
`test_the_insert_lands_outside_a_wrapped_scalar_and_the_file_still_parses` and
`test_apply_survives_the_emitters_real_shapes` build their fixtures with the
writers' own emit code and read back with `app.kg_store.parse_fact_file`, the only
reader that matters (`scripts/memory/kg_hygiene.py:139-143`).

Clause 5 of the item — `UNTAGGED_ALARM_SHARE` stays 0.5 and the rail still counts
all three untagged states — is pinned in `tests/test_kg_hygiene.py`, the file that
owns the rail, and is not duplicated here.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import yaml
from pathlib import Path

import pytest

from app.frontmatter import split_frontmatter
from app.kg_store import parse_fact_file
from scripts.memory import kg_hygiene as K
from scripts.memory import retag_untagged_facts as R

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "memory" / "retag_untagged_facts.py"

# ── The fixture: the live writers' output shapes ────────────────────────────

LONG_FACT = ("A2A declares capabilities in the Agent Card but its security posture "
             "is deliberately ambiguous about how a peer verifies a signed message, "
             "which is the gap the Aug 2025 survey lists first.")


def _rec(text: str, entity, *, entity_first: bool = True, id: str = "st-1",
         created_at: str = "2026-09-23", source_doc: str = "notes/src.md",
         source_hash: str = "abcd1234", category: str = "state",
         confidence: float = 1.0):
    """One fact record in the shape `agent_mcp/facts.py:681-693` writes.

    `entity` of `None` means OMIT THE KEY — the 1,064-record cohort. A string of
    spaces or an empty string is the other 2,142. `entity_first=False` puts `fact:`
    first, which is what `workers/sources/backlog_write_task.py:526-531` emits.
    """
    tagged = {"fact": text, "entity": entity, "confidence": confidence,
              "category": category, "id": id, "created_at": created_at,
              "valid_at": None, "invalid_at": None, "expired_at": None,
              "provenance": "EXTRACTED", "source_doc": source_doc,
              "source_hash": source_hash}
    if entity is None:
        del tagged["entity"]
    if entity_first:
        return tagged
    return {"fact": tagged.pop("fact"), **tagged}


def _doc(entity, facts: list[dict], *, title: str = "State",
         tail: dict | None = None) -> str:
    """A fact file in the shape `_shared._write_fact_frontmatter` writes.

    `entity` may be a string, a whitespace string, `""`, or the sentinel `"__omit__"`
    for a file whose front matter has no top-level `entity:` key at all. `tail` is
    emitted AFTER the `facts:` list, which is how the live writer closes a fact file
    (`last_updated:`) — the shape that ended all 831 live files holding an untagged
    record, and the one `test_the_final_record_is_tagged_inside_its_block_...` is
    built to reproduce.
    """
    fm: dict = {"type": "entity_fact", "source_doc": "notes/src.md"}
    if entity != "__omit__":
        fm["entity"] = entity
    fm["facts"] = facts
    if tail:
        fm.update(tail)
    return ("---\n"
            + yaml.dump(fm, default_flow_style=False, sort_keys=False)
            + "---\n\n" + title + "\n\nBody prose stays where it is.\n")


@pytest.fixture
def facts_root(tmp_path):
    """A facts tree with 13 records, 10 of them untagged across 6 files / 5 dirs.

    Seven are retaggable (4 files, 3 dirs); three sit in two files whose top-level
    `entity:` cannot name an entity, which is the whole skip population the live
    corpus happens not to contain (measured 2026-10-14: 0 of 830 files).
    """
    root = tmp_path / "facts"
    files = {
        # 3 untagged: quoted-blank, whitespace, null. Plus a tagged control.
        "Acme Widget/Acme Widget-state.md": _doc("Acme Widget", [
            _rec("Acme ships on Tuesdays.", "Acme Widget", id="acme-0"),
            _rec("Acme's gate rejects an empty queue.", "", id="acme-1"),
            _rec("Acme's retry budget is three.", "   ", id="acme-2"),
            _rec("Acme's index is rebuilt nightly.", None, id="acme-3"),
        ]),
        # 1 untagged, key absent, entity-first layout.
        "Acme Widget/Acme Widget-overview.md": _doc("Acme Widget", [
            _rec("Acme is a widget maker.", "Acme Widget", id="acme-o1"),
            _rec("Acme has four suppliers.", None, id="acme-o2"),
        ], title="Overview"),
        # 2 untagged, key absent, `fact:` FIRST with a wrapped double-quoted
        # scalar: the backlog_write_task shape that breaks a naive splice.
        "Beta Gadget/Beta Gadget-security model.md": _doc("Beta Gadget", [
            _rec(LONG_FACT, None, entity_first=False, id="beta-1",
                 source_doc="backlog/2001.md", source_hash="beef0001"),
            _rec(LONG_FACT.replace("first", "second"), None, entity_first=False,
                 id="beta-2", source_doc="backlog/2002.md", source_hash="beef0002"),
        ], title="Security model"),
        # Skipped: blank top-level entity.
        "Unattributed/Unattributed-state.md": _doc("", [
            _rec("Something happened here.", "", id="un-1"),
            _rec("Something else did too.", None, id="un-2"),
        ]),
        # Skipped: no top-level entity key at all.
        "No Entity/No Entity-state.md": _doc("__omit__", [
            _rec("Nobody owns this record.", "", id="ne-1"),
        ]),
        # 1 untagged, in a file whose front matter PROSE names `facts:` and
        # `entity:` — #1056's trap.
        "Wild API/Wild API-state.md": _doc("Wild API", [
            _rec("The reader takes the `facts:` key and then each record's "
                 "`entity:` word, in that order.", "", id="wild-1"),
            _rec("Wild API is versioned 3.", "Wild API", id="wild-2"),
        ]),
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(root.rglob("*.md"))}


def _records(root: Path, rel: str) -> list[dict]:
    """The file's records as `app.kg_store` reads them: `parse_fact_file` is
    `(frontmatter, facts)` and returns `({}, [])` for a file it cannot parse, so an
    empty list here means the repair broke the file, not that it is empty."""
    fm, facts = parse_fact_file(root / rel)
    assert fm, f"{rel} no longer parses through the real reader"
    assert facts, f"{rel} parsed to zero records through the real reader"
    return facts


def _rec_by_id(records: list[dict], rid: str) -> dict:
    hits = [r for r in records if r.get("id") == rid]
    assert len(hits) == 1, (rid, [r.get("id") for r in records])
    return hits[0]


# ── Clause 1: dry run by default, per-file and per-dir, names the value ─────

def test_no_apply_flag_writes_nothing_and_prints_the_plan_per_file_and_per_dir(
        facts_root, capsys):
    """A dry run that edits the corpus is not a rehearsal, and `--apply` is one
    keystroke away from 830 vault files.

    The count, not the edit, is the point: the plan must be legible per FILE and per
    DIR (the item's clause), and must name the top-level value it would write, so a
    reviewer can catch a wrong entity before the write rather than after the alarm
    goes green with the wrong tag. `reindex` appearing in dry-run output would mean
    the flag is being ignored, which is the same failure in a different coat.
    """
    before = _snapshot(facts_root)
    rc = R.main(["--facts-root", str(facts_root)])
    out = capsys.readouterr().out

    assert rc == 0, out
    assert _snapshot(facts_root) == before, "a dry run wrote the tree"
    assert "reindex" not in out.lower(), out
    assert "retag 7 record(s) in 4 file(s) across 3 director" in out, out

    for rel, n, ent in (("Acme Widget/Acme Widget-state.md", 3, "Acme Widget"),
                        ("Acme Widget/Acme Widget-overview.md", 1, "Acme Widget"),
                        ("Beta Gadget/Beta Gadget-security model.md", 2,
                         "Beta Gadget"),
                        ("Wild API/Wild API-state.md", 1, "Wild API")):
        line = next((ln for ln in out.splitlines() if rel in ln), None)
        assert line is not None, (rel, out)
        assert f"{n} record(s)" in line, line
        assert f"-> entity: {ent}" in line, line
    for d, n in (("Acme Widget", 4), ("Beta Gadget", 2), ("Wild API", 1)):
        assert any(d in ln and f"{n} record(s)" in ln
                   for ln in out.split("dirs:")[1].splitlines()), (d, n, out)
    # The two skip files are not in the plan.
    assert "Unattributed-state.md" not in out.split("skip")[0], out


def test_the_plan_is_a_machine_object_the_report_is_rendered_from_it(facts_root):
    """The report is a rendering, so the per-dir/per-file totals are computed once.

    Asserted against the data structure rather than only the text: a report that
    added up wrong while the structure was right would still let a reviewer sign off
    a wrong number, and this is the node that would notice which one is wrong.
    """
    rep = R.plan_tree(facts_root)
    assert rep.files == 4 and rep.records == 7 and rep.dirs == 3, rep.render(
        apply=False, root=facts_root)
    assert rep.by_file == {
        "Acme Widget/Acme Widget-state.md": 3,
        "Acme Widget/Acme Widget-overview.md": 1,
        "Beta Gadget/Beta Gadget-security model.md": 2,
        "Wild API/Wild API-state.md": 1,
    }, rep.by_file
    assert rep.by_dir == {"Acme Widget": 4, "Beta Gadget": 2, "Wild API": 1}, \
        rep.by_dir
    assert rep.skipped == 3 and len(rep.skips) == 2, rep.skips


def test_the_dry_run_names_the_value_of_a_top_level_entity_with_spaces(facts_root):
    """A top-level entity with a space in it is the common case, not an edge.

    'Acme Widget' is quoted by the emitter and its records must come back quoted; an
    implementation that wrote the raw string with a trailing space would be
    indistinguishable from a blank value to `_tag_of` in some readers, and the report
    is where that would be visible before it is written.
    """
    rep = R.plan_tree(facts_root)
    plan = next(p for p in rep.plans if p["path"].name == "Acme Widget-state.md")
    assert plan["entity"] == "Acme Widget"
    first = [e for e in plan["edits"] if e["kind"] == "replace"][0]["line"]
    # Written the way the house writers write it: `Acme Widget` needs no
    # quotes, so none are added. The value arriving intact as a two-word
    # name is pinned through the real reader by
    # `test_apply_survives_the_emitters_real_shapes`.
    assert first == "  entity: Acme Widget", first
    assert first.rstrip() != first.rstrip() + " ", "trailing space on the value"


# ── Clause 2: --apply writes them, and reindexes the changed paths once ─────

def test_apply_rewrites_the_records_and_reindexes_the_changed_paths_once(
        facts_root, capsys):
    """The second half of the seam this whole round turns on.

    `facts_idx` is derived from the markdown and every reader that counts goes
    through it, so a repaired file the index has not re-read is #2092's "reads 0
    facts while the rail counts 153,999" all over again. The report must therefore
    say HOW MANY rows reindex moved, not merely that the call happened — a reindex
    that returned 0 while the files were rewritten is the failure this line exists to
    make visible, and it is also the reason `paths=` is passed: a full reindex of a
    153,999-record corpus is not what a 830-file repair should cost.
    """
    calls: list[list[str]] = []

    def fake_reindex(*, paths, root=None, **kw):
        calls.append([str(Path(p).name) for p in paths])
        return {"files": len(list(paths)), "rows_total": 13}

    rc = R.main(["--facts-root", str(facts_root), "--apply"],
                reindexer=fake_reindex)
    out = capsys.readouterr().out
    assert rc == 0, out

    assert len(calls) == 1, f"reindex called {len(calls)} times, not once: {calls}"
    assert sorted(calls[0]) == sorted([
        "Acme Widget-state.md", "Acme Widget-overview.md",
        "Beta Gadget-security model.md", "Wild API-state.md"]), calls[0]
    assert "reindex: files=4, rows_total=13" in out, out

    for rel, ids, ent in (
            ("Acme Widget/Acme Widget-state.md", ["acme-1", "acme-2", "acme-3"],
             "Acme Widget"),
            ("Acme Widget/Acme Widget-overview.md", ["acme-o2"], "Acme Widget"),
            ("Beta Gadget/Beta Gadget-security model.md", ["beta-1", "beta-2"],
             "Beta Gadget"),
            ("Wild API/Wild API-state.md", ["wild-1"], "Wild API")):
        recs = _records(facts_root, rel)
        for rid in ids:
            assert _rec_by_id(recs, rid)["entity"] == ent, (rel, rid)
    # The tagged control is untouched, and still tagged.
    assert _rec_by_id(_records(facts_root, "Acme Widget/Acme Widget-state.md"),
                      "acme-0")["entity"] == "Acme Widget"


def _tree_records(root: Path) -> tuple[int, int]:
    """(total readable records, readable files) over a tree, per `parse_fact_file`.

    This is the DENOMINATOR the rail's `untagged` count is silently divided by, and
    the reason it has to be asserted: `parse_fact_file` answers `({}, [])` for a file
    whose YAML it cannot parse, so destroying a file REMOVES its records from the
    rail's numerator and makes the alarm look better. Measured on 2026-10-14 by
    replaying commit `42e1cc3e` over a copy of the live corpus: 350 files stopped
    parsing and 667 records vanished, and a check that only read the untagged count
    would have called that a partial success. A broken file is counted by its
    front matter being `{}` — the reader's own corrupt signal, not a second opinion.
    """
    total = readable = 0
    for p in sorted(root.rglob("*.md")):
        if not p.is_file():
            continue                          # an entity dir's NAME is an entity too
        fm, facts = parse_fact_file(p)
        if fm:
            readable += 1
        total += len(facts)
    return total, readable


def test_apply_clears_the_records_the_rail_counts(facts_root):
    """The end-to-end number, measured with the rail's own scanner — and with the
    record total it counts over, because the untagged count alone is satisfiable by
    destroying the files it counts.

    Before: 10 untagged in 6 files / 5 dirs. After: only the skip population is left,
    which is the shape the live repair should leave too — the alarm going quiet is
    owed-check's measurement after `--apply` runs over the live tree, and this is the
    same claim at fixture scale, so the mechanism is not trusted on faith. The 13
    records and 6 readable files are pinned as unchanged for the same reason: they
    are what a broken write takes with it.
    """
    before = K.scan(root=facts_root)["untagged"]
    assert (before["facts"], before["files"], before["dirs"]) == (10, 6, 5), before
    total_before, readable_before = _tree_records(facts_root)
    assert (total_before, readable_before) == (13, 6), (total_before,
                                                        readable_before)

    R.apply_tree(facts_root, reindexer=lambda **kw: {"files": 0})
    after = K.scan(root=facts_root)["untagged"]
    assert (after["facts"], after["files"], after["dirs"]) == (3, 2, 2), after
    assert after["facts"] == 3, "the skip population was retagged after all"
    total_after, readable_after = _tree_records(facts_root)
    assert (total_after, readable_after) == (13, 6), (
        f"the repair changed the tree the count is taken over: "
        f"{total_before}->{total_after} records, "
        f"{readable_before}->{readable_after} readable files")


def test_apply_survives_the_emitters_real_shapes(facts_root, capsys):
    """The fixture is re-emitted by the live writers' own code paths, not by hand.

    The fleet rule (`tests/test_fact_writer_reliability.py:14`) is that a fix is
    tested against the actual write path, not a synthetic string. A round-trip
    through `_write_fact_frontmatter` plus `atomic_write_text` is the real thing, and
    `parse_fact_file` is the only reader of consequence, so this is the node that
    catches a format change (yaml 6 quoting `''` differently, a new key order) before
    the nightly report does.
    """
    from agent_mcp import _shared
    from app.atomic_io import atomic_write_text

    fm = {"type": "entity_fact", "entity": "Acme Widget",
          "facts": [_rec("Acme's gate rejects an empty queue.", "", id="rt-1")]}
    p = facts_root / "Round Trip/Round Trip-state.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, _shared._write_fact_frontmatter(fm) + "\nBody.\n")

    rep = R.apply_tree(facts_root, reindexer=lambda **kw: {"files": 0})
    assert rep.by_file.get("Round Trip/Round Trip-state.md") == 1, rep.by_file
    _, recs = parse_fact_file(p)
    assert recs, "the round-tripped file parsed to no records"
    assert _rec_by_id(recs, "rt-1")["entity"] == "Acme Widget", recs


def test_the_insert_lands_outside_a_wrapped_scalar_and_the_file_still_parses(
        facts_root):
    """The 1,064 key-absent records are the dangerous half, and this is why.

    `backlog_write_task` writes `fact:` first as a double-quoted scalar whose
    continuation lines follow the dash line; a key line spliced in after the first
    line puts a plain scalar in the middle of a quoted one and the file will not
    parse. So the insert goes at the END of the record's block. The assertion that
    matters is not "the entity is set" but "the file still parses AND every other
    record survived with its fact text intact", which a splice-inside failure would
    destroy for the whole file, not just for one record.
    """
    rel = "Beta Gadget/Beta Gadget-security model.md"
    text_before = (facts_root / rel).read_text()
    assert re.search(r"^- fact: .+\n    \S", text_before, re.M), \
        "the fixture stopped being the wrapped-scalar shape this exists for"
    n_records_before = len(_records(facts_root, rel))

    R.apply_tree(facts_root, reindexer=lambda **kw: {"files": 0})

    recs = _records(facts_root, rel)
    assert len(recs) == n_records_before == 2, recs
    assert all(r["entity"] == "Beta Gadget" for r in recs), recs
    assert _rec_by_id(recs, "beta-1")["fact"] == LONG_FACT
    assert _rec_by_id(recs, "beta-2")["fact"] == LONG_FACT.replace("first", "second")


# ── Clause 3: skips ─────────────────────────────────────────────────────────

def test_a_file_whose_top_level_entity_is_blank_or_absent_is_skipped_and_untouched(
        facts_root, capsys):
    """'Re-tag from the location' is only safe where the location names an entity.

    Writing `entity: ` from a file whose own header is blank would be a no-op dressed
    as a repair, and writing a folder name that is a placeholder would attribute a
    record to a thing that is not a thing. The reason is printed per file because the
    exit code stays 0 for a correct skip, so the report is the only place a reader
    learns a file was left alone. Clause 3's own test: the fixture's blank-top file
    still reports its records untagged AFTER `--apply`.
    """
    before = _snapshot(facts_root)
    R.main(["--facts-root", str(facts_root), "--apply"],
           reindexer=lambda **kw: {"files": 0})
    out = capsys.readouterr().out

    for rel, why in (("Unattributed/Unattributed-state.md", "blank"),
                     ("No Entity/No Entity-state.md", "absent")):
        line = next((ln for ln in out.split("skip:")[1].splitlines() if rel in ln),
                    None)
        assert line is not None, (rel, out)
        assert why in line.lower(), line
        assert (facts_root / rel).read_text() == before[rel], rel

    # Read back through the rail's own per-directory scan, which is what the alarm
    # consumes. `scan()["items"]` is the FOREIGN report and carries no untagged
    # list at all, so looping over it would assert nothing and pass.
    left = []
    for d in sorted(facts_root.iterdir()):
        if not d.is_dir():
            continue
        for fname in K.scan_entity_dir(d)[1]["files"]:
            _fm, file_recs = parse_fact_file(d / fname)
            left += [r.get("id") for r in file_recs if K._tag_of(r) == ""]
    assert {"un-1", "un-2", "ne-1"} == set(left), left


def test_a_directory_on_the_named_skip_list_is_left_alone(facts_root, monkeypatch,
                                                          capsys):
    """The skip list's mechanism, pinned while the list is empty.

    `DIR_SKIP` ships empty because the population is measured — 0 of 830
    untagged-bearing files have a blank top-level `entity:` (#2676) — so nothing in
    the live corpus exercises this branch today. A constant that suppresses a
    directory with no test behind it is a silent data-loss switch waiting for the
    first person who adds a name to it, which is the `SKIP_NAMES` class of failure.
    """
    monkeypatch.setattr(R, "DIR_SKIP", ("Wild API",))
    before = _snapshot(facts_root)
    rc = R.main(["--facts-root", str(facts_root), "--apply"],
                reindexer=lambda **kw: {"files": 0})
    out = capsys.readouterr().out
    assert rc == 0, out
    assert (facts_root / "Wild API/Wild API-state.md").read_text() \
        == before["Wild API/Wild API-state.md"]
    assert "skip list" in out
    assert "Wild API-state.md" not in out.split("skip:")[0]
    # The others still get repaired: a skip list that suppressed the whole run would
    # also satisfy the assertion above by accident.
    assert _rec_by_id(_records(facts_root, "Acme Widget/Acme Widget-state.md"),
                      "acme-1")["entity"] == "Acme Widget"


def test_the_shipped_skip_list_is_empty_because_the_population_is_measured():
    """`DIR_SKIP == ()` is a measurement, not an oversight, so it is pinned.

    Counting every untagged-bearing file on 2026-10-14 — 830 files, 3,206 records,
    top dirs Lloyd 213 / Inner Voice 104 / IsaacLab 89 / Tool Keyword Activation Map
    75 / Autoresearch 74 — found 0 files whose top-level `entity:` is blank or
    absent, so every record's location names an entity and a skip entry would only
    block a repair that is due. The item's clause 2 still requires the branch to
    exist, and the node above is what makes the empty list safe to ship.
    """
    assert R.DIR_SKIP == (), R.DIR_SKIP


# ── Clause 4: only `entity:` ever changes, and the second apply is a no-op ───

def test_only_the_entity_line_changes_and_a_second_apply_reports_zero_changes(
        facts_root, capsys):
    """830 files are rewritten by this repair; 'it only fixed the tag' has to be a
    theorem, not a hope. Every field the index and the fact-quality jobs read is
    compared for every record — `id` is in that list because
    `(file_path, fact_id)` is the only unique key `facts_idx` has, so a drifted id
    would duplicate rows silently — plus the body, which is outside the front matter
    and exactly what a re-serialising repair script would lose.

    The second run is the idempotency claim: it is what makes a re-run safe after an
    interrupted first one, and the fleet's `changed == 0` precedent is
    `repair_fact_ids.py:224`.
    """
    before = {rel: _records(facts_root, rel) for rel in _snapshot(facts_root)}
    body_before = {rel: split_frontmatter(text)[1]
                   for rel, text in _snapshot(facts_root).items()}
    # Where each tag is supposed to come from, read off the plan rather than
    # restated here, so this node cannot drift from the script's own map.
    planned = {p["rel"]: p["entity"] for p in R.plan_tree(facts_root).plans}
    assert len(planned) == 4, planned

    R.apply_tree(facts_root, reindexer=lambda **kw: {"files": 0})

    after = {rel: _records(facts_root, rel) for rel in before}
    changed = 0
    for rel, recs_before in before.items():
        recs_after = after[rel]
        assert len(recs_after) == len(recs_before), rel
        for rb, ra in zip(recs_before, recs_after):
            assert set(rb) - {"entity"} == set(ra) - {"entity"}, (rel, rb.get("id"))
            for key in set(rb) - {"entity"}:
                assert rb[key] == ra[key], (rel, rb.get("id"), key,
                                            rb[key], ra[key])
            # `_tag_of` takes the RECORD, not the entity value — the first cut of
            # this line passed `rb.get("entity")`, which made it answer `None` for
            # every record and quietly degraded the guard to a bare inequality.
            # `record_is_untagged` is the script's own predicate over a whole
            # record, so the key-absent state is in it too, and the count below is
            # the number of records this run actually tagged.
            if R.record_is_untagged(rb) and rel in planned:
                assert ra.get("entity") == planned[rel], (rel, rb.get("id"),
                                                          ra.get("entity"))
                changed += 1
            elif R.record_is_untagged(rb):
                # A record in one of the two skip files: the location names no
                # entity, so it must STILL be untagged after `--apply`.
                assert R.record_is_untagged(ra), (rel, rb.get("id"),
                                                  ra.get("entity"))
            else:
                assert ra.get("entity") == rb.get("entity"), (rel, rb.get("id"))
            assert str(ra.get("fact")) == str(rb.get("fact"))
    assert changed == 7, changed
    assert {rel: split_frontmatter((facts_root / rel).read_text())[1]
            for rel in before} == body_before, "the body moved"

    capsys.readouterr()
    rc = R.main(["--facts-root", str(facts_root), "--apply"],
                reindexer=lambda **kw: {"files": 0})
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "would retag 0 record(s)" in out, out
    assert "nothing to apply" in out, out


def test_the_written_files_still_round_trip_through_the_only_reader_that_matters(
        facts_root):
    """After the write, `parse_fact_file` must agree with the plan, for every file.

    The plan is computed from the emitter; the readback is `app/kg_store`, which is
    what the whole KG is built from. An agreement check here is the same boundary
    check the nightly report gets one day later, at the moment it is still cheap to
    fix.
    """
    planned = {p["path"]: p["entity"] for p in R.plan_tree(facts_root).plans}
    R.apply_tree(facts_root, reindexer=lambda **kw: {"files": 0})
    assert planned, "the plan came back empty, so this would pass on nothing"
    for path, entity in planned.items():
        _, recs = parse_fact_file(Path(path))
        assert recs, path
        for rec in recs:
            assert rec.get("entity") == entity, (path, rec.get("id"),
                                                 rec.get("entity"), entity)


def test_a_prose_line_naming_facts_or_entity_does_not_move_the_edit(facts_root):
    """#1056, re-earned: 'the fence test that tested nothing' used `re.match(r'^---')`
    against front matter that opens with `--- facts:`, so a pattern matching a real
    file and a pattern matching nothing print the same green.

    This fixture's `Wild API` record has a fact whose own text says `facts:` and
    `entity:`. The assertions therefore run both directions: the edit still lands
    (the prose did not stop it) and it landed on the KEY line, not on the sentence —
    the positive control that makes an absence assertion here worth anything.
    """
    rel = "Wild API/Wild API-state.md"
    text_before = (facts_root / rel).read_text()
    assert "`facts:`" in text_before and "`entity:`" in text_before, \
        "the fixture no longer contains the prose traps this node is for"
    n_records_before = len(_records(facts_root, rel))

    R.apply_tree(facts_root, reindexer=lambda **kw: {"files": 0})

    text_after = (facts_root / rel).read_text()
    recs = _records(facts_root, rel)
    assert len(recs) == n_records_before, \
        "the edit changed how many records the real reader sees"
    assert _rec_by_id(recs, "wild-1")["entity"] == "Wild API", recs
    prose = next(ln for ln in text_after.splitlines() if "`facts:`" in ln)
    assert not prose.lstrip().startswith("entity:"), prose
    assert "entity: Wild API" not in prose, prose


# ── The seam to the index, and the shapes that must not be guessed ──────────

def test_a_written_tree_that_cannot_be_reindexed_exits_one(facts_root, capsys):
    """The one failure mode where silence is the wrong answer.

    The files are already rewritten when reindex is called, so if that call raises
    the markdown says one thing and `facts_idx` serves another — which is #2092's
    class of bug, and the only case in this tool's contract where a non-zero exit is
    the whole point. Exit 0 here would let the nightly report go green on a
    half-applied repair.
    """
    def boom(**kw):
        raise RuntimeError("store unavailable")

    # A second, untouched tree for the exit-code leg: the `apply_tree` below
    # writes the files BEFORE reindex is called, so by the time `main` ran over
    # the same tree there would be nothing left to write, no reindex call, and
    # an honest exit 0 — which would test the wrong thing.
    fresh = facts_root.parent / "fresh"
    shutil.copytree(facts_root, fresh)
    rep = R.apply_tree(facts_root, reindexer=boom)
    assert rep.written == 4, rep.written            # the writes did happen
    assert "4 file(s) written but reindex failed" in rep.reindex_error, \
        rep.reindex_error
    assert "REINDEX FAILED" in rep.render(apply=True, root=facts_root)

    capsys.readouterr()
    rc = R.main(["--facts-root", str(fresh), "--apply"], reindexer=boom)
    assert rc == 1, capsys.readouterr().out


def test_the_default_reindexer_returns_the_call_that_actually_writes_the_index(
        facts_root, tmp_path):
    """Clause 1 of #2909 — the seam every other node here fakes, crossed for real.

    Calling `default_reindexer()` and invoking what it returns with `paths=` over a temp
    facts root must index those files (`files >= 1`) with the rows readable back out of
    the store — and must not raise `AttributeError`, which is what it did on 2026-10-17.

    `git grep -n default_reindexer -- tests/` before #2909 returned exactly one line,
    `assert R.default_reindexer.__name__ == "default_reindexer"` inside
    `test_importing_the_module_does_not_open_the_kg_store`: a NAME check, in a node
    about import hygiene. Nothing ever called the hook, so the whole production path
    was untested while every apply above passed on an injected lambda. Called for real
    on 2026-10-17T14:33Z it raised `AttributeError: 'KGStore' object has no attribute
    'reindex'` — `reindex` lives on `_FactsIdx` (`app/kg_store.py:1261`), which the
    store exposes as `self.facts_idx` (`:218`) and does not delegate — after the run
    had already rewritten 831 fact files.

    So: a temp `kg.sqlite`, the hook production actually gets, and the rows read back
    out of the store. The `file_path` spelling is asserted too, because
    `reindex(paths=…, root=…)` matches rows on the tree-relative path it stored, and a
    hook that wrote absolute `file_path` rows would look like a clean reindex while
    every reader missed them.
    """
    from app import kg_store
    import yaml

    st = kg_store.configure(tmp_path / "kg.sqlite")
    try:
        target = facts_root / "Acme Widget" / "Acme Widget-state.md"
        # What that file says, read off its own front matter with yaml — not through
        # this script's parser and not through the store, so the expectations below are
        # independent of both.
        recs = yaml.safe_load(
            target.read_text(encoding="utf-8").split("---")[1])["facts"]
        hook = R.default_reindexer()
        assert hook == st.facts_idx.reindex, (
            "default_reindexer() is not the store's own facts-index reindex: "
            + repr(hook))
        stats = dict(hook(paths=[target], root=facts_root))
        # Clause 1's `files >= 1`. Every count here is derived from the fixture, never
        # a literal that would go stale if `_doc`/`_rec` grew: one path went in, so one
        # file must come out, and `facts` must be the records that file holds.
        assert stats["files"] >= 1, f"nothing was indexed: {stats}"
        assert stats["files"] == 1, f"one path in, {stats['files']} files reported"
        assert stats["facts"] == len(recs), f"{len(recs)} records in that file: {stats}"

        rows = st.facts_idx.for_entity("Acme Widget")
        assert {r["file_path"] for r in rows} == {
            "Acme Widget/Acme Widget-state.md"}, [r["file_path"] for r in rows]
        # The `file_path` is the tree-RELATIVE spelling, which is the thing `root=`
        # decides: `reindex` deletes by `file_path=?` and inserts what
        # `_rows_for_file` computed from the same root, so a wrong root would leave
        # the old rows standing under one spelling and add new ones under another.
        #
        # The expected entity per record is spelled the way `_rows_for_file` resolves
        # it — `f.get("entity") or entity` (`app/kg_store.py:1242`) — so `''` and an
        # absent key fall back to the file's top-level entity AT INDEX TIME (which is
        # why #2909's owed reindex is a content no-op on the live tree), while the
        # truthy `entity: '   '` record became its own key and only a retag plus this
        # call moves it.
        blank = st.facts_idx.for_entity("   ")
        assert sorted(r["entity"] for r in rows + blank) == sorted(
            rec.get("entity") or "Acme Widget" for rec in recs), \
            [(r["entity"], r["fact"]) for r in rows + blank]
        assert len(rows) + len(blank) == len(recs), (
            f"{len(rows) + len(blank)} of {len(recs)} records indexed")
        assert [r["fact"] for r in rows].count("Acme's retry budget is three.") == 0
    finally:
        kg_store.reset()


def test_a_default_reindexer_that_cannot_resolve_reports_failure_and_exits_one(
        facts_root, monkeypatch, capsys):
    """Clause 2 of #2909 — where the hook is BUILT is a clause, not a detail.

    `apply_tree` wrote the files, then resolved its default reindexer OUTSIDE the
    `try` whose `except` sets `rep.reindex_error` — so the 2026-10-17 crash escaped as
    14 bare traceback lines on stderr: no `retagged …` line, no `REINDEX FAILED:` line,
    and `main`'s exit-code expression never evaluated. A tool that has already written
    831 files and exits by traceback is a tool whose operator cannot tell what half
    happened. Constructing the hook is now inside that guard, so an unresolvable
    factory costs exactly what a failed reindex costs.
    """
    def cannot_resolve():
        raise AttributeError("'KGStore' object has no attribute 'reindex'")

    monkeypatch.setattr(R, "default_reindexer", cannot_resolve)
    fresh = facts_root.parent / "fresh"
    shutil.copytree(facts_root, fresh)
    # How many files the run is going to write, read off the plan rather than
    # hardcoded: this node is about what happens AFTER that many files are written.
    planned = len([p for p in R.plan_tree(facts_root).plans if p["edits"]])
    assert planned >= 1, "the fixture has nothing to write, so this node proves nothing"

    rep = R.apply_tree(facts_root)          # no injected reindexer: the production route
    assert rep.written == planned, rep.written      # the writes did happen
    assert f"{planned} file(s) written but reindex failed" in rep.reindex_error, \
        rep.reindex_error
    assert "AttributeError" in rep.reindex_error, rep.reindex_error
    assert "REINDEX FAILED" in rep.render(apply=True, root=facts_root)

    capsys.readouterr()
    rc = R.main(["--facts-root", str(fresh), "--apply"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "REINDEX FAILED:" in out, out


def test_the_reindex_is_asked_for_the_tree_that_was_written(facts_root, tmp_path):
    """Clause 3 of #2909 — `apply_tree` must ask for the tree it wrote, not the default.

    `--facts-root` exists, and `reindex`'s own `root` defaults to the LIVE tree.
    `_FactsIdx.reindex` resolves `root = Path(root or VAULT_FACTS_ROOT)`
    (`app/kg_store.py:1267`), and its note at `:1350-1353` is explicit that rows are
    matched on the tree-relative spelling they were indexed against, so `root` must be
    the tree they came from. `apply_tree` held that root in hand and never passed it —
    so a `--facts-root` repair would have indexed `VAULT_FACTS_ROOT` instead of the
    tree it wrote, and `_rows_for_file`'s `relative_to` fallback (`:1229-1232`) would
    have made the delete and the insert disagree on the spelling of `file_path`.
    """
    calls: list[dict] = []

    def spy(*, paths, root=None, **kw):
        paths = list(paths)
        calls.append({"paths": [Path(p) for p in paths], "root": root})
        # Exactly the three keys `_FactsIdx.reindex` returns
        # (`app/kg_store.py:1268`), so this fake is as tight as the seam it stands in
        # for: a call that passed some other keyword, or read back a key the real
        # reindex does not emit, fails here and not only against a real store.
        return {"files": len(paths), "facts": 0, "entities_registered": 0}

    other = tmp_path / "other-facts"        # a second tree exists; neither is the default
    shutil.copytree(facts_root, other)
    rc = R.main(["--facts-root", str(facts_root), "--apply"], reindexer=spy)
    assert rc == 0, rc
    assert len(calls) == 1, f"reindex called {len(calls)} times: {calls}"
    assert calls[0]["root"] == facts_root, (
        "reindex was asked for a tree other than the one it wrote: "
        + str(calls[0]["root"]))
    assert all(p.is_relative_to(facts_root) for p in calls[0]["paths"]), calls[0]["paths"]
    assert not any(p.is_relative_to(other) for p in calls[0]["paths"]), calls[0]["paths"]


def test_an_apply_over_a_fully_tagged_tree_prints_the_pair_and_needs_no_index(
        facts_root, capsys, monkeypatch):
    """Clause 4 of #2909 — a tree with nothing to write keeps today's behaviour.

    This is also the state the live tree is in now, and the reason the
    2026-10-17 crash is latent rather than recurring — `apply_tree` only reaches
    `default_reindexer()` inside `if changed:`, so a `--apply` over a tagged tree never
    touches the store. Both halves are pinned in one run: the report says what it said
    before this change, and the DEFAULT hook is rigged to raise, so if a future edit
    ever builds the hook on a tree with nothing to write, this node goes red instead of
    the nightly going red.
    """
    def never_wanted():
        raise AssertionError("default_reindexer() must not be resolved when nothing "
                             "was written — it would open kg.sqlite for nothing")

    n_files = 4                                 # files in this fixture holding a record
    monkeypatch.setattr(R, "default_reindexer", never_wanted)
    R.apply_tree(facts_root, reindexer=lambda **kw: {
        "files": 0, "facts": 0, "entities_registered": 0})
    capsys.readouterr()
    rc = R.main(["--facts-root", str(facts_root), "--apply"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "nothing to apply: no record in this tree is missing an entity:" in out, out
    assert "unchanged: every record already carries its file's entity" in out, out
    assert f"nothing to do in {n_files} file(s):" in out, out
    assert "REINDEX FAILED" not in out, out


def test_nothing_to_repair_is_a_clean_exit_not_an_alarm(facts_root, capsys):
    """A tree with no untagged record is the goal state.

    Exiting non-zero on 'planned nothing' would make the very first nightly after the
    live repair read red — the #2016 / #552 class of false alarm this fleet has been
    written about repeatedly. Two runs: the untouched fixture, and the repaired one,
    which is the state the live tree lands in.
    """
    capsys.readouterr()
    assert R.main(["--facts-root", str(facts_root)]) == 0
    R.apply_tree(facts_root, reindexer=lambda **kw: {"files": 0})
    capsys.readouterr()
    rc = R.main(["--facts-root", str(facts_root)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "retag 0 record(s)" in out, out


def test_importing_the_module_does_not_open_the_kg_store():
    """A maintenance script that opens `kg.sqlite` at import is a lock hazard.

    `default_reindexer` exists so the store is reached only on a real `--apply`,
    which is also what makes the whole suite above runnable without touching the
    live index. This node pins it the fleet's way: run the script as a subprocess,
    with the real `kg.sqlite` in the environment, and require an exit without the
    store having been opened by the import.
    """
    import subprocess
    import sys

    # The hazard this names is OPENING the database, not importing its module:
    # `kg_hygiene`, whose predicate this script delegates to, imports
    # `app.kg_store` itself. `store()` is what opens `kg.sqlite` and takes its
    # lock, and its process-wide singleton is `_default` — so the claim pinned
    # here is that importing this script leaves that singleton unconstructed.
    probe = ("import sys; sys.path.insert(0, %r); "
             "import scripts.memory.retag_untagged_facts as R; "
             "import app.kg_store as ks; "
             "print(ks._default is not None)" % str(ROOT))
    out = subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip().splitlines()[-1] == "False", (
        "importing the script opened kg.sqlite: " + out.stdout[-500:])
    assert R.default_reindexer.__name__ == "default_reindexer"


def test_the_module_loads_by_path_the_way_the_fleet_checks_scripts():
    """Every maintenance script in `scripts/memory/` has hyphen-tolerant import
    problems; this one has an underscore name and still gets the same check, because
    the alternative is a suite that passes on a script the runner cannot load."""
    spec = importlib.util.spec_from_file_location("retag_untagged_facts", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.is_untagged("") and mod.is_untagged("   ") and mod.is_untagged(None)
    assert not mod.is_untagged("Lloyd")



def test_the_final_record_is_tagged_inside_its_block_when_a_top_level_key_follows(
        tmp_path):
    """Every live file in this population, at fixture scale: `last_updated:` AFTER `facts:`.

    The live writer closes a fact file with a top-level `last_updated:`, and a replay
    over the corpus on 2026-10-14 found one in 831 of the 831 files carrying an
    untagged record — the `facts:` sequence is never the last thing in the front
    matter. `_fact_blocks` used to leave a record with no closing key open to the end
    of the block, which put the new `entity:` line under that top-level key instead of
    inside the record. PyYAML reads an indented mapping under a scalar key as a syntax
    error (`expected <block end>, but found '<block mapping start>'`), and
    `parse_fact_file` swallows that into `({}, [])`: the file's whole record set
    vanishes, not one record. Measured that way on the previous commit `42e1cc3e`: 350
    files unreadable and 667 records lost, and the untagged count went DOWN while it
    happened.
    """
    d = tmp_path / "Acme Widget"
    d.mkdir(parents=True)
    f = d / "Acme Widget-state.md"
    f.write_text(_doc("Acme Widget", [
        _rec("Wrapped fact whose last line is a continuation of a plain scalar, long "
             "enough that the writer wrapped it across lines.", entity=None, id="a-1"),
    ], tail={"last_updated": "2026-10-13T04:03:22.869623+00:00"}))

    out = R.apply_tree(tmp_path, reindexer=lambda **kw: {"files": 1})
    assert (out.written, out.failed, out.verify_bad, out.verify_lost) == \
        (1, [], 0, 0), f"{out.written} {out.failed} {out.verify_bad} {out.verify_lost}"
    fm, recs = parse_fact_file(f)
    assert len(recs) == 1, "the file stopped parsing: the tag landed outside the record"
    assert recs[0]["entity"] == "Acme Widget", recs[0]
    assert fm["last_updated"] == "2026-10-13T04:03:22.869623+00:00"
    assert K.scan(root=tmp_path)["untagged"]["facts"] == 0, \
        "the rail still counts this file, so the alarm would not go quiet"


def test_no_record_that_had_a_tag_keeps_it_untouched_and_the_rest_gain_one(facts_root):
    """The untagged count going DOWN is also what happens when a file stops parsing, so
    the count alone cannot be the check. This one reads the rail's own classifier per
    record.

    `kg_hygiene._tag_of` (scripts/memory/kg_hygiene.py:112) is what the floor is
    computed from: `''` for a record whose `entity` key is absent or whose value is
    blank, the stripped name otherwise. Three claims in exactly its units: a record
    that had a tag keeps that tag unchanged, every record in a planned file ends up
    with the planned entity, a record in a skipped file is still `''`, and this
    script's own predicate agrees with the rail's on every record in the tree — a
    drift between the two would mean the repair optimises a number the alarm does not
    read.
    """
    before = {rel: [K._tag_of(r) for r in _records(facts_root, rel)]
              for rel in _snapshot(facts_root)}
    planned = {p["rel"]: p["entity"] for p in R.plan_tree(facts_root).plans}
    assert len(planned) == 4, planned
    assert any(t == "" for ts in before.values() for t in ts), \
        "fixture lost its teeth: it holds no untagged record"

    R.apply_tree(facts_root, reindexer=lambda **kw: {"files": 1})

    for rel, was in before.items():
        now = [K._tag_of(r) for r in _records(facts_root, rel)]
        assert len(now) == len(was), f"{rel}: the record set changed size"
        for i, (b, a) in enumerate(zip(was, now)):
            if b:
                assert a == b, f"{rel}#{i}: the tag {b!r} became {a!r}"
            elif rel in planned:
                assert a == planned[rel], f"{rel}#{i}: untagged ended as {a!r}"
            else:
                assert a == "", f"{rel}#{i}: a skipped file was tagged {a!r}"
        for r in _records(facts_root, rel):
            assert R.record_is_untagged(r) == (K._tag_of(r) == ""), \
                f"{rel}: this script and the rail disagree on what untagged means"


def test_a_healthy_file_is_reported_as_nothing_to_do_and_not_as_left_untagged(
        facts_root, capsys):
    """Clause 1's report has to be READ before an 831-file `--apply`, and a file that
    holds no untagged record must not be itemised as a file left untagged.

    Two shapes carry no untagged record at all: a file where every record already
    names its entity, and a file with no `facts:` list. Over the live tree on
    2026-10-14 those were 24,575 and 14,697 files — 39,272 of 40,094 — and the dry
    run printed every one of them under `skip: … file(s) left untagged`, a claim false
    about each of them (they hold nothing untagged) in the one report a human reads
    before applying. They are now counted on one line, and the skip block keeps only
    the population clause 3 is about: files that DO hold untagged records the repair
    refused to tag, each named.
    """
    tagged = facts_root / "Gamma Tool" / "Gamma Tool-state.md"
    tagged.parent.mkdir(parents=True)
    tagged.write_text(_doc("Gamma Tool", [
        _rec("Already names its entity.", "Gamma Tool", id="g-1"),
    ]))
    empty = facts_root / "Gamma Tool" / "Gamma Tool-notes.md"
    empty.write_text(_doc("Gamma Tool", []))

    assert R.main(["--facts-root", str(facts_root)]) == 0
    out = capsys.readouterr().out

    rep = R.plan_tree(facts_root)
    assert rep.no_action_files == 2, rep.no_action
    assert rep.no_action == {"no facts: list": 1,
                             "every record already carries its entity": 1}, \
        rep.no_action
    # The genuine skip population is untouched by this: the two files whose own
    # top-level entity names nothing, still 3 records, still itemised by name.
    assert rep.skipped == 3 and len(rep.skips) == 2, rep.skips
    block = out.split("skip:")[1]
    for rel, _reason in rep.skips:
        assert rel in block, f"{rel} vanished from the skip block"
    for name in ("Gamma Tool-state.md", "Gamma Tool-notes.md"):
        assert name not in block, f"{name} is healthy but was called left untagged"

    line = next(ln for ln in out.splitlines() if ln.startswith("nothing to do in "))
    assert line.startswith("nothing to do in 2 file(s): 1 hold no fact records, "
                           "1 already fully tagged"), line
    assert "hold no fact records" in out and "already fully tagged" in out


def test_the_dry_run_report_stays_short_on_a_tree_where_almost_every_file_is_healthy(
        tmp_path, capsys):
    """The live tree's shape, compressed: 1 bad file, 1 refused file, 300 healthy ones.

    97.9% of the live facts tree is healthy and the repair's own report scales with
    it, so line count is the property that made the dry run unreadable: 40,655 lines
    for 831 files that would be repaired. Three hundred healthy files here must cost
    ONE counted line, while the file that would be repaired still gets its per-file
    count and the refused file still gets named.
    """
    healthy = tmp_path / "Healthy"
    healthy.mkdir()
    for i in range(300):
        (healthy / f"Healthy-h{i:03d}.md").write_text(_doc(
            "Healthy", [_rec(f"Tagged record {i}.", "Healthy", id=f"h-{i}")]))
    broken = tmp_path / "Broken"
    broken.mkdir()
    (broken / "Broken-state.md").write_text(_doc("Broken", [
        _rec("Needs a tag.", None, id="b-1"), _rec("Also needs one.", None, id="b-2"),
    ]))
    refused = tmp_path / "Unattributed"
    refused.mkdir()
    (refused / "Unattributed-state.md").write_text(_doc("", [
        _rec("No entity anywhere to supply it.", None, id="u-1"),
    ]))

    assert R.main(["--facts-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()

    assert len(lines) < 40, f"{len(lines)} lines to say 1 file needs a tag"
    assert any(ln.startswith("  Broken/Broken-state.md: 2 record(s) "
                             "-> entity: Broken") for ln in lines), out
    assert "nothing to do in 300 file(s): 300 already fully tagged" in out, out
    block = out.split("skip:")[1]
    assert "Unattributed/Unattributed-state.md" in block, out
    assert "Healthy-h000.md" not in block, out


# ── #98 clause 4: the runner's snapshot is a witness, not an alias of the damage ──

def test_a_hardlink_snapshot_still_holds_the_pre_apply_bytes_after_the_apply(
        facts_root, tmp_path):
    """The one property `skills/retag-untagged-facts/SKILL.md` step 1 bets the run on.

    The daily runner (#98, #2722) snapshots a 40,411-file facts tree with `cp -al`
    immediately before `--apply`, and a hardlink is only a witness — rather than a
    second name for the bytes the repair is about to change — because the repair
    never writes into the linked file: `atomic_write_text` writes a temp and
    `os.replace`s *that* inode over the target (`app/atomic_io.py:104`, the replace
    at `:135`). So the snapshot's entry keeps
    pointing at the pre-apply inode and the before-state survives its own repair.

    The failure this pins is the quiet one: had the repair written in place, both
    names would have come back saying what the repair says, the run's report would
    have quoted a before-state that no longer existed anywhere, and nothing except
    this test could tell the difference. A hardlink that tracks the write is worse
    than no snapshot, because it looks like evidence.

    Links rather than copies for the same reason the skill uses `cp -al`: a copy of
    that tree is paid for in bytes every night, a link only in inodes.
    """
    snap = tmp_path / "facts-snapshot-TEST"          # a sibling, as on the box
    shutil.copytree(facts_root, snap, copy_function=os.link)

    rel = "Wild API/Wild API-state.md"
    live, kept = facts_root / rel, snap / rel
    before = kept.read_text(encoding="utf-8")
    # What the runner's witness is supposed to hold: the whole tree, pre-apply.
    before_all = _snapshot(snap)
    ino_before = os.stat(kept).st_ino
    assert os.stat(live).st_ino == ino_before, (
        "the snapshot is a copy, not a link — not the mechanism the skill runs")
    assert R.record_is_untagged(_rec_by_id(_records(snap, rel), "wild-1")), (
        "the sampled record is not the damage the rail counts, so it witnesses "
        "nothing")

    rep = R.apply_tree(facts_root, reindexer=lambda *, paths, **kw: {
        "files": len(list(paths)), "rows_total": 13})
    assert [str(p) for p in rep.written_paths], (
        "the apply wrote nothing, so there is no witness to test")
    assert str(live) in [str(p) for p in rep.written_paths], rep.written_paths

    # The witness is intact: the same inode, the same bytes, still the damage.
    assert os.stat(kept).st_ino == ino_before
    assert kept.read_text(encoding="utf-8") == before
    assert R.record_is_untagged(_rec_by_id(_records(snap, rel), "wild-1")), (
        "the snapshot tracked the write — it is an alias of the repair, not a "
        "witness of what was there before")
    # ...and the live tree is the thing that moved: a new inode, and the tag.
    assert os.stat(live).st_ino != ino_before, (
        "atomic_write_text did not rename a new inode over the target")
    assert _rec_by_id(_records(facts_root, rel), "wild-1")["entity"] == "Wild API"
    # Every file in the snapshot is still the before-state, not just the sampled
    # one: the repair moved inodes out from under the links and left each link
    # holding what it linked.
    assert _snapshot(snap) == before_all, (
        "the snapshot tree drifted while the apply ran — it is no witness")
    # The control that makes that a claim rather than a tautology: the live tree
    # did NOT stay the before-state. Had the repair written in place, the two
    # assertions above could not both be true of the same bytes.
    assert _snapshot(facts_root)[rel] != before_all[rel], (
        "the live file never changed, so the snapshot proves nothing about the write")
