"""The #1813 repair script must repair, refuse, and repeat — over a fixture tree.

`facts_idx` attributes each fact to the document it came from, and nine rows cannot answer
that: `SELECT COUNT(*) FROM facts_idx WHERE source_doc IS NULL` returned 9 of 119,171 at
2026-09-29T09:18Z with `provenance_coverage()` saying `source_doc_pct 99.99`. Session-distill
extraction wrote them with no `source_doc`; `83b6e0ab` closed the hole at write time
(`agent_mcp/facts.py:574`), and a refusal repairs nothing, which is what
`scripts/maintenance/repoint_null_source_docs.py` is for. The repair has to land in the
markdown because `reindex()` runs `DELETE FROM facts_idx` (`app/kg_store.py:1282`) — a
database-only edit is undone by the next rebuild.

These tests run the whole script against a fixture facts tree and a fixture store, because
the live run edits data under `~/lloyd-data` that no round may touch: `KGStore(path)` takes a
path and `update_file(path, root=…)` takes a root (`app/kg_store.py:1309`), so every seam the
script uses is injectable and the live run is this same code with production arguments.

What is pinned, clause by clause: the nine rows go to zero and each change is printed with
the FACT'S TEXT beside it (clause 1 — the text is the only thing letting a human check the
mapping, since the row→transcript split is corroborated at topic level only); an id the tree
does not carry and a run that matched nothing both exit non-zero (clause 2 — a 0-row match
must not read as success); an absent transcript writes nothing at all (clause 3 — a path
pointing nowhere is a worse claim than the NULL it replaced); a second run changes no byte
(clause 4); and no fact outside the nine moves, in the markdown or in its row (clause 5).

What is NOT pinned here, deliberately: that the two live transcript paths exist. Session
files rotate, so a test reading them would go red for a reason unrelated to this script. The
shipped map's SHAPE is pinned instead, and running it against production data is owed to the
owed-check job with the printed pairs eyeballed first (#1813's owed entries).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "maintenance" / "repoint_null_source_docs.py"

from app.kg_store import KGStore, parse_fact_file  # noqa: E402
from agent_mcp._shared import _write_fact_frontmatter  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rns = _load(SCRIPT, "repoint_under_test")

#: The nine rows #1813 names, as `(entity, category-file, fact_id)`. EIGHT files, not the
#: seven the item's prose says: `Lloyd Chrome Side Panel` keeps `stat-001` in its `state`
#: file and `buil-001` in its `build` file, while `automod promotion gate` keeps BOTH its
#: behaviour facts in ONE file — verified per-file on disk at triage (1+1+1+2+1+1+1+1 = 9).
NINE: tuple[tuple[str, str, str], ...] = (
    ("Mission Control Dashboard", "state", "stat-001"),
    ("automod promotion gate", "behavior", "beha-001"),
    ("automod promotion gate", "behavior", "beha-002"),
    ("automod round resume", "behavior", "beha-001"),
    ("lloyd web frontend test tier", "conventions", "conv-001"),
    ("lloyd backlog", "state", "stat-001"),
    ("Alan", "preference", "pref-089"),
    ("Lloyd Chrome Side Panel", "state", "stat-001"),
    ("Lloyd Chrome Side Panel", "build", "buil-001"),
)

#: One sentinel fact per touched file, with provenance already correct. Clause 5 is about
#: these: a repair that rewrites a whole front-matter block can move a neighbour's bytes,
#: and the only check that catches it is a fact the map never named.
SENTINEL = ("sent-001", "A neighbour fact that the map never names.",
            "/home/alansrobotlab/obsidian/memory/2026-09-20.md")

UNTYPED = "Untouched Entity"


def _index_store(tmp_path: Path, facts: Path) -> KGStore:
    """A fixture store over `facts`, at the same distance from the live data as `env`'s."""
    kg = KGStore(tmp_path / "kg-fixture.sqlite")
    kg.facts_idx.reindex(sorted(facts.rglob("*.md")), root=facts)
    return kg


def _fact(fact_id: str, text: str, source_doc, provenance="EXTRACTED",
          created_at="2026-09-28T00:54:38.823498+00:00", confidence=0.9) -> dict:
    """One fact entry, in the key order the live writer emits."""
    return {"fact": text, "confidence": confidence, "id": fact_id,
            "created_at": created_at, "valid_at": None, "invalid_at": None,
            "expired_at": None, "provenance": provenance, "source_doc": source_doc}


def _fact_file(facts_root: Path, entity: str, category: str, facts: list[dict],
               body: str | None = None) -> Path:
    """Write one fact file the way `agent_mcp/facts.py:409` writes it, and return its path.

    `body` defaults to the shape the live tree carries after a writer pass — a blank line
    then the heading — because that is what eight of the real files look like; the one node
    that needs the other shape passes it explicitly rather than editing this default.
    """
    directory = facts_root / entity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entity}-{category}.md"
    fm = {"type": "facts", "entity": entity, "category": category, "facts": facts,
          "last_updated": "2026-09-28T00:54:47.686571+00:00"}
    if body is None:
        body = (f"\n\n# {entity} - {category}\n\n**Entity:** {entity}\n"
                f"**Category:** {category}\n**Fact Count:** {len(facts)}\n")
    path.write_text(_write_fact_frontmatter(fm) + body, encoding="utf-8")
    return path


@pytest.fixture
def env(tmp_path):
    """A fixture facts tree carrying the nine NULL rows, plus witnesses that must not move.

    The files are built through the LIVE writer rather than hand-written YAML, so the
    fixture has the bytes the script will meet in production — including the front-matter
    shape the reindex parses. A hand-typed fixture would let the script pass against a tree
    that nothing else on this machine ever writes.
    """
    facts = tmp_path / "facts"
    by_file: dict[tuple[str, str], list[dict]] = {}
    for entity, category, fact_id in NINE:
        by_file.setdefault((entity, category), []).append(
            _fact(fact_id, f"the claim recorded by {entity} as {fact_id}", None))
    for (entity, category), entries in by_file.items():
        _fact_file(facts, entity, category, entries + [_fact(*SENTINEL)])
    # An entity the map never mentions, so "only the eight files moved" has a control that
    # was never a candidate rather than a file that happened to be skipped.
    _fact_file(facts, UNTYPED, "state",
               [_fact("stat-001", "a fact in a file no entry names",
                      "/home/alansrobotlab/obsidian/memory/2026-09-21.md")])
    kg = KGStore(tmp_path / "kg.sqlite")
    kg.facts_idx.reindex(root=facts, register_entities=False)
    assert len(kg._query("SELECT 1 FROM facts_idx WHERE source_doc IS NULL")) == 9, \
        "the fixture does not carry the defect it is meant to repair"
    yield SimpleNamespace(tmp=tmp_path, facts=facts, kg=kg)
    kg.close()


def _map(env, missing=()) -> tuple:
    """The shipped map with its `(entity, fact_id)` pairs intact and transcripts in `tmp`.

    The pairs come from `rns.REPOINTS` unchanged — a fixture that renamed an entity to make
    fake paths fit would be testing a map nobody ships. Only the third element moves, and
    `missing` leaves one of the two named transcripts out of existence so clause 3's guard
    has something real to refuse.
    """
    t1 = env.tmp / "T1.json"
    t2 = env.tmp / "T2.json"
    for path, tag in ((t1, "T1"), (t2, "T2")):
        if tag not in missing:
            path.write_text('{"messages": []}', encoding="utf-8")
    return tuple((entity, fact_id, str(t2 if entity == "Lloyd Chrome Side Panel" else t1))
                 for entity, fact_id, _ in rns.REPOINTS)


def _null_rows(env) -> list:
    return env.kg._query(
        "SELECT entity, fact_id FROM facts_idx WHERE source_doc IS NULL "
        "ORDER BY entity, fact_id")


def _bytes(env) -> dict[str, bytes]:
    return {str(p.relative_to(env.tmp)): p.read_bytes()
            for p in sorted((env.facts).rglob("*.md"))}


def _neighbours(env) -> dict[str, dict[str, str]]:
    """Every fact NOT named by the map, rendered as the writer would render it.

    Rendered, not compared as parsed dicts: clause 5 is about the markdown entry, and a
    re-dump that reorders keys or re-quotes a sentence yields a dict that still compares
    equal while the file on disk has changed.
    """
    named = {(e, f) for e, f, _ in rns.REPOINTS}
    out: dict[str, dict[str, str]] = {}
    for path in sorted(env.facts.rglob("*.md")):
        fm, facts = parse_fact_file(path)
        entity = fm.get("entity")
        out[str(path)] = {f["id"]: yaml.dump([f], default_flow_style=False, sort_keys=False)
                          for f in facts if (entity, f["id"]) not in named}
    return out


COLS = ("text_hash", "fact", "created_at", "provenance", "confidence",
        "valid_at", "expired_at", "invalid_at", "file_path", "category")


def _rows(env) -> dict[tuple[str, str], dict]:
    return {(r["entity"], r["fact_id"]): {c: r[c] for c in COLS} for r in
            env.kg._query("SELECT * FROM facts_idx ORDER BY entity, fact_id")}


# ── clause 1: the nine go to zero, and each change is printed with its text ───

def test_the_run_leaves_no_null_rows_and_prints_each_change_with_its_fact_text(env):
    """Clause 1, and why the print line carries the fact rather than only its id."""
    log: list[str] = []
    result = rns.run(env.facts, env.kg, apply=True, repoints=_map(env), out=log.append)

    assert result["ok"], result
    assert len(result["changed"]) == 9, [c["fact_id"] for c in result["changed"]]
    assert not _null_rows(env), _null_rows(env)

    for entity, category, fact_id in NINE:
        want = [p for e, f, p in _map(env) if (e, f) == (entity, fact_id)][0]
        line = [ln for ln in log if ln.startswith("CHANGE ")
                and f"entity={entity!r}" in ln and f"fact_id={fact_id}" in ln]
        assert line, f"no CHANGE line for {entity}/{fact_id}:\n" + "\n".join(log)
        assert "old=None" in line[0], line[0]
        assert f"new={want!r}" in line[0], \
            f"the line names a different target than the map for this row: {line[0]}"
        # The fact's own text, on the run's output. Without it the operator is asked to
        # approve a mapping between an opaque id and a path with nothing to compare against.
        assert [ln for ln in log if ln.startswith("    fact:")
                and f"{entity} as {fact_id}" in ln], \
            f"CHANGE for {entity}/{fact_id} printed no fact text"
    assert sum(1 for ln in log if ln.startswith("CHANGE ")) == 9, "\n".join(log)


def test_the_repaired_rows_carry_the_transcript_the_map_names_row_by_row(env):
    """The value written is the map's per row — not one path stamped over all nine."""
    rns.run(env.facts, env.kg, apply=True, repoints=_map(env), out=lambda s: None)
    rows = {(r["entity"], r["fact_id"]): r["source_doc"] for r in
            env.kg._query("SELECT entity, fact_id, source_doc FROM facts_idx")}
    for entity, fact_id, path in _map(env):
        assert rows[(entity, fact_id)] == path, (entity, fact_id, rows[(entity, fact_id)])
    nine = {(e, f) for e, f, _ in rns.REPOINTS}
    # Restricted to the nine: the touched files carry sentinel facts whose own source_doc is
    # a vault path, and counting those would report a third group that has nothing to do with
    # the map.
    chrome = {p for (e, _f), p in rows.items() if e == "Lloyd Chrome Side Panel"
              and (e, _f) in nine}
    rest = {p for (e, _f), p in rows.items() if e != "Lloyd Chrome Side Panel"
            and (e, _f) in nine}
    assert len(chrome) == 1 and len(rest) == 1 and chrome != rest, \
        "the two-group split the mapping rests on was flattened onto one transcript"


# ── clause 2: an unmatched id and a zero-match run both fail, loudly ──────────

def test_an_id_the_tree_does_not_carry_is_reported_unmatched_and_refused(env):
    """Clause 2, first half."""
    log: list[str] = []
    broken = _map(env) + (("No Such Entity", "stat-001", str(env.tmp / "T1.json")),)
    result = rns.run(env.facts, env.kg, apply=True, repoints=broken, out=log.append)

    assert not result["ok"], "an entry nobody carries still let the run claim success"
    assert [(u["entity"], u["fact_id"]) for u in result["unmatched"]] == \
        [("No Such Entity", "stat-001")], result
    assert any("UNMATCHED" in ln and "No Such Entity" in ln for ln in log), "\n".join(log)
    # A refusal is a refusal: the eight rows it COULD have repaired stay unrepaired, so the
    # operator fixes the map and runs once instead of half-repairing twice.
    assert len(_null_rows(env)) == 9, _null_rows(env)


def test_a_run_matching_no_entries_never_exits_zero(env):
    """Clause 2, second half: 0 matched is a wrong tree, not a clean bill of health."""
    empty = env.tmp / "empty-facts"
    empty.mkdir()
    log: list[str] = []
    result = rns.run(empty, env.kg, apply=True, repoints=_map(env), out=log.append)

    assert not result["ok"], result
    assert result["reason"] == "unmatched", result
    assert len(result["unmatched"]) == 9, result
    assert all(u["why"].startswith("no entity directory") for u in result["unmatched"]), result
    assert any("UNMATCHED" in ln for ln in log), "\n".join(log)
    assert len(_null_rows(env)) == 9, "the real tree was touched by a run over an empty one"


def test_main_exits_two_over_a_tree_carrying_none_of_the_nine(tmp_path, monkeypatch):
    """The exit code a scheduler sees, through the real CLI path rather than `run()`.

    `REPOINTS` itself is patched, not the two transcript constants: the map is built from
    them at import, so patching the names afterwards would leave the CLI checking the live
    transcripts and this node passing for a reason it does not state.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    transcript = tmp_path / "t.json"
    transcript.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rns, "REPOINTS", (("Some Entity", "stat-001", str(transcript)),))
    code = rns.main(["--facts-root", str(facts), "--db", str(tmp_path / "kg.sqlite"),
                     "--apply"])
    assert code == rns.EXIT_BLOCKED, code


def test_main_exits_zero_only_when_it_can_back_its_claim(tmp_path, monkeypatch):
    """The other side of the exit code: 0 means "checked", never "no error was raised"."""
    facts = tmp_path / "facts"
    transcript = tmp_path / "t.json"
    transcript.write_text("{}", encoding="utf-8")
    _fact_file(facts, "Some Entity", "state",
               [_fact("stat-001", "one claim with no attribution", None)])
    monkeypatch.setattr(rns, "REPOINTS", (("Some Entity", "stat-001", str(transcript)),))
    code = rns.main(["--facts-root", str(facts), "--db", str(tmp_path / "kg.sqlite"),
                     "--apply"])
    assert code == rns.EXIT_OK, code
    rows = KGStore(tmp_path / "kg.sqlite")._query(
        "SELECT source_doc FROM facts_idx WHERE source_doc IS NULL")
    assert not rows, rows


# ── clause 3: a missing transcript means nothing is written ───────────────────

def test_a_missing_transcript_writes_nothing_and_is_refused(env):
    """Clause 3: the guard runs before a single fact file is opened for writing."""
    before = _bytes(env)
    log: list[str] = []
    result = rns.run(env.facts, env.kg, apply=True,
                     repoints=_map(env, missing=("T2",)), out=log.append)

    assert not result["ok"], result
    assert result["reason"] == "missing_transcripts", result
    assert _bytes(env) == before, "a row was pointed at a transcript that does not exist"
    assert len(_null_rows(env)) == 9, "the store moved even though the tree did not"
    assert any("BLOCKED" in ln and "transcript does not exist" in ln for ln in log), \
        "\n".join(log)
    assert not any(ln.startswith("CHANGE ") for ln in log), \
        "changes were planned and printed after the guard had already refused"


def test_a_missing_transcript_refuses_even_in_dry_run(env):
    """The guard is not an apply-only check: a plan pointing at nothing is not a plan."""
    before = _bytes(env)
    result = rns.run(env.facts, env.kg, apply=False,
                     repoints=_map(env, missing=("T1",)), out=lambda s: None)
    assert not result["ok"] and result["reason"] == "missing_transcripts", result
    assert _bytes(env) == before


# ── clause 4: the second run is a report, not a rewrite ──────────────────────

def test_a_second_run_changes_no_byte_and_reports_every_entry_already_correct(env):
    """Clause 4: idempotence is what makes a one-shot script safe to re-run by hand."""
    rns.run(env.facts, env.kg, apply=True, repoints=_map(env), out=lambda s: None)
    before = _bytes(env)
    log: list[str] = []
    result = rns.run(env.facts, env.kg, apply=True, repoints=_map(env), out=log.append)

    assert result["ok"], result
    assert not result["changed"], [c["fact_id"] for c in result["changed"]]
    assert len(result["already"]) == 9, len(result["already"])
    assert _bytes(env) == before, "a run with nothing to do still rewrote the files"
    body = "\n".join(log)
    assert body.count("OK (already correct)") == 9, body
    assert "CHANGE " not in body, body


def test_a_dry_run_writes_nothing(env):
    """`--dry-run` is the default mode, and it is the eyeball step the item owes."""
    before = _bytes(env)
    log: list[str] = []
    result = rns.run(env.facts, env.kg, apply=False, repoints=_map(env), out=log.append)

    assert result["ok"] and result["reason"] == "dry_run", result
    assert len(result["changed"]) == 9, result
    assert _bytes(env) == before, "the dry run wrote"
    assert len(_null_rows(env)) == 9, "the dry run reindexed the store"
    assert any("DRY RUN" in ln for ln in log), "\n".join(log)


# ── clause 5: nothing outside the nine moves ─────────────────────────────────

def test_every_fact_outside_the_nine_is_unchanged_in_the_markdown_and_the_store(env):
    """Clause 5: a whole-block rewrite must leave its neighbours byte-for-byte alone."""
    named = {(e, f) for e, f, _ in rns.REPOINTS}
    markdown_before = _neighbours(env)
    rows_before = _rows(env)
    untouched = (env.facts / UNTYPED / f"{UNTYPED}-state.md").read_bytes()

    rns.run(env.facts, env.kg, apply=True, repoints=_map(env), out=lambda s: None)

    assert _neighbours(env) == markdown_before, \
        "a fact the map never named changed rendering in a file this script rewrote"
    rows_after = _rows(env)
    for key, before in rows_before.items():
        if key in named:
            continue
        assert rows_after[key] == before, key
    assert (env.facts / UNTYPED / f"{UNTYPED}-state.md").read_bytes() == untouched, \
        "a file the map never names was rewritten anyway"


def test_the_repaired_rows_keep_every_other_field(env):
    """The nine change one column each: attribution moves, the fact itself does not."""
    before = _rows(env)
    rns.run(env.facts, env.kg, apply=True, repoints=_map(env), out=lambda s: None)
    after = _rows(env)
    for key in sorted({(e, f) for e, f, _ in rns.REPOINTS}):
        assert after[key] == before[key], \
            f"{key}: text_hash/created_at/provenance/confidence must survive the re-point"


def test_only_the_eight_mapped_files_are_rewritten(env):
    """Eight files, not seven, and the run stops at eight.

    Pinned because the item's own count is wrong in this direction, and both Chrome facts
    landing in ONE file would leave `buil-001` NULL while every other clause still read
    green: the map keys on (entity, fact_id), so a file-name assumption is exactly where a
    silent miss hides.
    """
    files = sorted(p.relative_to(env.facts).as_posix() for p in env.facts.rglob("*.md"))
    assert files == [
        "Alan/Alan-preference.md",
        "Lloyd Chrome Side Panel/Lloyd Chrome Side Panel-build.md",
        "Lloyd Chrome Side Panel/Lloyd Chrome Side Panel-state.md",
        "Mission Control Dashboard/Mission Control Dashboard-state.md",
        f"{UNTYPED}/{UNTYPED}-state.md",
        "automod promotion gate/automod promotion gate-behavior.md",
        "automod round resume/automod round resume-behavior.md",
        "lloyd backlog/lloyd backlog-state.md",
        "lloyd web frontend test tier/lloyd web frontend test tier-conventions.md",
    ], files
    before = {p: p.read_bytes() for p in env.facts.rglob("*.md")}
    rns.run(env.facts, env.kg, apply=True, repoints=_map(env), out=lambda s: None)
    touched = sorted(str(p.relative_to(env.facts).as_posix())
                     for p, b in before.items() if p.read_bytes() != b)
    assert len(touched) == 8, touched
    assert f"{UNTYPED}/{UNTYPED}-state.md" not in touched, \
        "the control entity's file moved, so the run is not scoped to the map"
    assert len(touched) == len(set(touched))


def test_the_only_bytes_that_move_are_the_mapped_source_doc_lines(env):
    """Clause 5's byte half: the diff of a touched file is its `source_doc` lines only.

    A front-matter round-trip can satisfy every fact-identity test in this file and still
    rewrite the whole document — re-quote a string, re-order a key, re-flow a wrapped fact —
    leaving a repair whose diff is unreadable and whose only honest description is "the file
    was rewritten". So this grades the diff itself: each added or removed line must be a
    `source_doc:` line, and the total must be 9 out and 9 in.

    It is also how the splice detail gets caught. `agent_mcp/facts.py:407` splices the body
    at `content[find("---", 3) + 3:]`, which keeps the newline that ends the closing
    delimiter line; since the writer's output ends `"---\\n"`, that inserts a blank line
    between front matter and body. Copying it here would have moved 9 extra lines — none of
    them a `source_doc:` line — for a provenance fix.
    """
    import difflib
    before = _bytes(env)
    rns.run(env.facts, env.kg, apply=True, repoints=_map(env), out=lambda s: None)
    after = _bytes(env)

    moved = 0
    for name, old in before.items():
        for line in difflib.unified_diff(old.decode().splitlines(keepends=True),
                                         after[name].decode().splitlines(keepends=True), n=0):
            if line.startswith(("+++", "---")) or line[:1] not in "+-":
                continue
            body = line[1:].strip()
            assert body.startswith("source_doc:"), \
                f"{name}: the repair moved a line that is not a source_doc line: {body!r}"
            moved += 1
    assert moved == 18, \
        f"expected 9 removed nulls and 9 added paths and nothing else, got {moved} lines"


def test_a_file_the_writer_leaves_unreadable_is_reported_unstable(tmp_path, monkeypatch):
    """The post-apply re-read, made able to fail.

    `run()` re-plans from the bytes it just wrote because `reindex()` runs `DELETE FROM
    facts_idx`: a markdown entry that only *looks* repaired in the table comes back NULL on
    the next full rebuild, so the repair silently un-happens weeks later. Nothing in a
    single run can see that from the inside, so this makes the writer return content that
    reads back as no facts at all and requires the refusal.
    """
    facts = tmp_path / "facts"
    entity = "Body Loose Entity"
    path = _fact_file(facts, entity, "state", [_fact("stat-001", "one claim", None)])
    transcript = tmp_path / "T.json"
    transcript.write_text("{}", encoding="utf-8")
    kg = _index_store(tmp_path, facts)
    before = path.read_bytes()
    monkeypatch.setattr(rns, "_write_fact_frontmatter", lambda fm: "---\ntype: facts\n---\n")
    log: list[str] = []
    result = rns.run(facts, kg, apply=True,
                     repoints=((entity, "stat-001", str(transcript)),), out=log.append)
    assert not result["ok"], result
    assert result["reason"] == "unstable", result
    assert any("UNSTABLE" in ln for ln in log), "\n".join(log)
    # The write did happen — this is a post-check, not a pre-check — so record that it left
    # a changed file rather than pretending the refusal was a rollback.
    assert path.read_bytes() != before, \
        "an unstable run should have written; if it did not, the post-check is dead code"


def test_a_body_that_starts_immediately_after_the_delimiter_is_not_truncated(tmp_path):
    """The other half of the split: no blank line to preserve must not eat real content.

    The fixture bodies all start with a newline, so a split that simply dropped one
    character more would pass every other node here. A fact file whose heading begins on
    the line after `---` is the shape that would lose its first character — and unlike a
    blank line, a lost `#` is silent corruption of the document this repair exists to keep
    intact.
    """
    facts = tmp_path / "facts"
    entity = "Body Tight Entity"
    path = _fact_file(facts, entity, "state",
                      [_fact("stat-001", "a claim with no attribution", None)],
                      body="# Body Tight Entity - state\n\n**Entity:** Body Tight Entity\n")
    transcript = tmp_path / "T.json"
    transcript.write_text("{}", encoding="utf-8")
    kg = _index_store(tmp_path, facts)
    repoints = ((entity, "stat-001", str(transcript)),)
    assert rns.run(facts, kg, apply=True, repoints=repoints,
                   out=lambda s: None)["ok"]
    after = path.read_text()
    assert "# Body Tight Entity - state" in after, after
    assert not after.startswith("\n"), f"a newline was invented: {after[:20]!r}"
    assert after.count("\n---\n") == 1, "the delimiter was duplicated"


# ── the shipped map, as data ─────────────────────────────────────────────────

def test_the_shipped_map_names_exactly_the_nine_rows_and_two_transcripts():
    """The map's shape, checked without reading the machine it will eventually run on."""
    assert len(rns.REPOINTS) == 9
    assert len({(e, f) for e, f, _ in rns.REPOINTS}) == 9
    assert {(e, f) for e, f, _ in rns.REPOINTS} == {
        ("Mission Control Dashboard", "stat-001"),
        ("automod promotion gate", "beha-001"),
        ("automod promotion gate", "beha-002"),
        ("automod round resume", "beha-001"),
        ("lloyd web frontend test tier", "conv-001"),
        ("lloyd backlog", "stat-001"),
        ("Alan", "pref-089"),
        ("Lloyd Chrome Side Panel", "stat-001"),
        ("Lloyd Chrome Side Panel", "buil-001"),
    }
    transcripts = {t for _, _, t in rns.REPOINTS}
    assert transcripts == {rns.TRANSCRIPT_2026_09_27, rns.TRANSCRIPT_2026_09_23}
    assert len(transcripts) == 2
    for t in transcripts:
        assert t.startswith("/home/alansrobotlab/lloyd-data/sessions/") and t.endswith(".json")
    assert sum(1 for _, _, t in rns.REPOINTS if t == rns.TRANSCRIPT_2026_09_27) == 7
    assert sum(1 for _, _, t in rns.REPOINTS if t == rns.TRANSCRIPT_2026_09_23) == 2


def test_the_script_writes_through_the_live_writer_not_a_local_copy():
    """The item's "write through the in-tree path" clause, pinned as a call graph.

    A locally re-implemented YAML emit would keep these tests green while production fact
    files drifted from what `fact_add` writes, and the next unrelated `fact_add` would
    restyle the file as a side effect. So the script must CALL the read path's writer, not
    match its output by hand.
    """
    import inspect

    src = inspect.getsource(rns._rewrite)
    assert "_write_fact_frontmatter(" in src and "atomic_write_text(" in src, src
    assert rns._write_fact_frontmatter is _write_fact_frontmatter, \
        "the script imported a copy rather than the writer the read path uses"
