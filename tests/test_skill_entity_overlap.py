"""#2440 — the skill-catalog overlap instrument, keyed on resolved KG entities.

The thing being replaced is not a function, it is a paragraph. Autonomy task #83's Stage 5
screen is described in a runbook with no scorer and no threshold committed to git, and it has
produced three readings of one corpus in two nights: the 2026-10-07 pass scored
`ml-paper-writing` vs `research-paper-writing` at 0.03 and called the pair a name look-alike,
the 2026-10-08 pass scored it 0.327-0.358 and filed it as #2409 — 2,607 lines of two SKILL.md
files for one procedure (`research-paper-writing` among them, archived in place on
2026-10-08 by vault commit `17e447d8` as that item's outcome), which a human found by
reading one flagged pair. #2409 writes the
defect out: "The two numbers are not comparable and one of the two readings is wrong."

So every node here pins a property of a MEASUREMENT, not a verdict about any skill pair. The
one exception is deliberate and goes the other way: the committed fixture's ranking is
measured to FAIL on the live graph (lowest positive 0.2982, highest negative 0.4706), and the
node below asserts that failure with its numbers exact, because an instrument that hides a
bad result is the instrument #2440 exists to replace. Whether the suppressed entity key is
worth keeping is ruled on that printed number after landing (owed 1), not by a test.

Nothing in this file writes to the vault, to the production knowledge graph, or to any
skill's front matter — which is also what clause 5 tests rather than asserts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "skill_entity_overlap", REPO_ROOT / "scripts" / "skill_entity_overlap.py")
seo = importlib.util.module_from_spec(SPEC)
sys.modules["skill_entity_overlap"] = seo
SPEC.loader.exec_module(seo)

FIXTURE = REPO_ROOT / "eval" / "skill_entity_overlap_pairs.json"
BODIES = REPO_ROOT / "eval" / "skill_entity_overlap_pair_bodies.json"

#: The production figures for the committed fixture, measured on the live graph at
#: `~/lloyd-data/_pipeline/vault-derived/kg.sqlite` on 2026-10-08 and copied out of that run,
#: not recomputed from the committed extract. They are what the extract-based scoring below
#: has to reproduce: if the extract and the live store ever disagree, this table is where the
#: disagreement surfaces, which is the point of pinning the number rather than the ranking.
PRODUCTION_FIGURES = {
    "memory-capture|periodic-memory-capture-lloyd": {"entity_score": 0.7692,
                                                     "incumbent_score": 0.0816,
                                                     "n_a": 12, "n_b": 11},
    "ml-paper-writing|research-paper-writing": {"entity_score": 0.2982,
                                                "incumbent_score": 0.34,
                                                "n_a": 54, "n_b": 94},
    "claude-code|codex": {"entity_score": 0.4706, "incumbent_score": 0.4839,
                          "n_a": 12, "n_b": 13},
    "claude-code|opencode": {"entity_score": 0.1724, "incumbent_score": 0.3871,
                             "n_a": 12, "n_b": 22},
    "codex|opencode": {"entity_score": 0.1667, "incumbent_score": 0.3235,
                       "n_a": 13, "n_b": 22},
    "github-code-review|github-issues": {"entity_score": 0.2222, "incumbent_score": 0.3409,
                                         "n_a": 24, "n_b": 9},
    "github-code-review|github-pr-workflow": {"entity_score": 0.2647,
                                              "incumbent_score": 0.3864, "n_a": 24, "n_b": 19},
    "github-code-review|github-repo-management": {"entity_score": 0.1562,
                                                  "incumbent_score": 0.375, "n_a": 24,
                                                  "n_b": 13},
    "github-issues|github-pr-workflow": {"entity_score": 0.2727, "incumbent_score": 0.3636,
                                         "n_a": 9, "n_b": 19},
    "github-issues|github-repo-management": {"entity_score": 0.375, "incumbent_score": 0.4595,
                                             "n_a": 9, "n_b": 13},
    "github-pr-workflow|github-repo-management": {"entity_score": 0.2308,
                                                  "incumbent_score": 0.4, "n_a": 19,
                                                  "n_b": 13},
    "nightly-reflection-knowledge-analysis|nightly-reflection-knowledge-write": {
        "entity_score": 0.3111, "incumbent_score": 0.4091, "n_a": 34, "n_b": 25},
}


# ── fixtures ─────────────────────────────────────────────────────────────────

def make_graph(tmp_path: Path, *, entities=(), aliases=()):
    """A throwaway knowledge graph at `tmp_path/kg.sqlite`.

    Built with `KGStore`, never with `kg_store.configure()`: the latter repoints the process
    default, and a test that leaves it pointing at a temp file hands every later node in this
    session an empty graph and a confident set of zeroes.
    """
    from app import kg_store
    path = tmp_path / "kg.sqlite"
    store = kg_store.KGStore(path)
    for name in entities:
        store.entities.register(name)
    for surface, canonical in aliases:
        store.aliases.set(surface, canonical, kind="semantic", origin="manual")
    return path


def plant(root: Path, name: str, *, description: str, body: str,
          status: str = "active") -> Path:
    """One advertised skill on disk, in the shape `iter_active_skills` walks."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nstatus: {status}\n---\n{body}\n",
        encoding="utf-8")
    return directory


def run_report(*, skills_root: Path, store: Path, out: Path, pairs: Path | None = None,
               extra=(), data_root: Path | None = None):
    """Run the committed script the way the nightly will, across the process boundary.

    A subprocess, not `seo.main()`, because every clause that matters here is about what the
    artifact says on disk and stdout: the `unknown` verdict, the ranking line, the exit code,
    the byte-identity of two runs, and the fact that the run leaves the tree alone. An
    in-process call would also inherit whatever knowledge graph the last test in this session
    pointed the process default at, which is precisely how a coverage number starts meaning
    nothing. `$LLOYD_DATA` is pinned to a temp root so the default report path cannot land in
    the live `_pipeline` tree even if this file ever forgets to pass `--out`.
    """
    argv = [sys.executable, str(REPO_ROOT / "scripts" / "skill_entity_overlap.py"), "report",
            "--skills-root", str(skills_root), "--store", str(store), "--out", str(out), *extra]
    if pairs is not None:
        argv += ["--pairs", str(pairs)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if data_root is not None:
        env["LLOYD_DATA"] = str(data_root)
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
                          timeout=300)
    return proc.returncode, proc.stdout, proc.stderr


def read_report(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8"))


def snapshot(root: Path) -> dict:
    """`relative path -> sha256 of bytes`, for every file under `root`."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


SUBJECT_ENTITIES = ["Neo4j", "Agent memory", "Entity resolution", "Knowledge graph",
                    "Retrieval augmented generation", "Cypher", "Graph database"]


@pytest.fixture()
def corpus(tmp_path: Path):
    """A three-skill catalogue whose third skill resolves nothing at all.

    `alpha` and `beta` are deliberately above the scoring floor and share most of their
    subject, and `gamma` is the coverage case the clause is about: a body whose only
    resolvable-looking names are harness tools and path segments, so after suppression it has
    no entity set to compare. That is not a hypothetical — one of the 194 advertised skills
    resolves to zero entities on the live graph and 29 more sit below the scoring floor, so
    roughly a sixth of the catalogue has no comparable set — the `unknown` path is the shape
    of a real slice of it, not a corner.
    """
    skills = tmp_path / "skills"
    plant(skills, "alpha", description="Work on agent memory using a knowledge graph and Neo4j",
          body="Procedure over Neo4j, a Graph database, for Agent memory, Entity resolution, "
               "Knowledge graph construction and Retrieval augmented generation, written in "
               "Cypher when needed.")
    plant(skills, "beta", description="Graph-backed recall for agents via Neo4j",
          body="Steps touching Neo4j, the Graph database, Agent memory, Entity resolution and "
               "Retrieval augmented generation before any Knowledge graph edit lands.")
    plant(skills, "gamma", description="Rename a batch of files safely",
          body="Call Bash to move them, then Read the result and Write the log to "
               "~/obsidian/skills/Neo4j/notes.md. See skills/bash-runbook/SKILL.md and "
               "/tmp/Write/out.log for the same advice.")
    # Two skills the walker must NOT yield, so "every advertised skill" is a checked claim
    # rather than a count: `delta` is retired in place and `old` lives under the dot-prefixed
    # archive, which is exactly why neither appears in `skills_search` tonight.
    plant(skills, "delta", description="Retired experiment with Neo4j and Cypher",
          body="Neo4j, Cypher, Graph database, Agent memory, Entity resolution.",
          status="retired")
    plant(skills / ".archived", "old", description="An archived predecessor about Neo4j",
          body="Neo4j and Graph database and Cypher and Agent memory and Entity resolution.")
    store = make_graph(tmp_path, entities=SUBJECT_ENTITIES + ["Bash", "Read", "Write",
                                                              "Neo4j"],
                       aliases=[("Agent Craft", "Agent memory")])
    return skills, store, tmp_path


# ── clause 1: the projection, and the unknown verdict ────────────────────────

def test_every_advertised_skill_is_projected_and_coverage_is_printed(tmp_path, corpus):
    """Clause 1: every yielded skill gets an entity set, and the zero count is on stdout.

    Across the CLI, because "every skill the tool advertises" is a claim about the same
    walker every other surface uses: `corpus()` goes through
    `agent_mcp.skills.iter_active_skills`, so a skill filed under a dot-prefixed archive
    directory or carrying a quarantined `status:` is out of the projection for the same reason
    it is out of `skills_search` — and `zero_entity_skills` is printed before any pair, since
    it bounds every score under it.
    """
    skills, store, data = corpus
    out = tmp_path / "report.json"
    rc, stdout, stderr = run_report(skills_root=skills, store=store, out=out,
                               data_root=data / "lloyd-data")
    assert rc == 0, (stderr or stdout)

    report = read_report(out)
    assert report["n_skills"] == 3, report
    assert sorted(s["name"] for s in report["skills"]) == ["alpha", "beta", "gamma"], report
    assert "delta" not in report["skills"] and "old" not in report["skills"], (
        "a retired-in-place or archived skill entered the projection")
    assert "skills_scanned 3" in stdout, stdout
    assert "zero_entity_skills 1" in stdout, stdout

    by_name = {s["name"]: s for s in report["skills"]}
    assert "Neo4j" in by_name["alpha"]["entities"] and by_name["alpha"]["n_entities"] >= 5
    assert by_name["gamma"]["n_entities"] == 0, by_name["gamma"]
    assert report["entity_set_size_histogram"]["0"] == 1, report["entity_set_size_histogram"]


def test_a_pair_with_nothing_to_compare_is_unknown_and_never_a_clean_verdict(tmp_path, corpus):
    """Clause 1's rail: no denominator means `unknown`, in the report and on stdout.

    Three unscored rows, one per reason a pair can have no reading. `alpha | gamma` is the
    clause's own case — one side resolved nothing at all, which on the live catalogue is the
    ORDINARY fate of a skill whose subject the graph never indexed. On the live catalogue only
    1 of 194 advertised skills resolves zero entities, but 5,162 of the 18,721 pairs are
    unscoreable because a side sits below the floor — so `unknown` is the majority verdict on
    pairs, not a rarity. `neo4j-admin | cypher-cli` is the thin case:
    both sides resolved, both below the stated floor of 5, and 4 entities in common terms are
    not a measurement of similarity. The third row has no recorded set at all, which is what a
    hand-edited or truncated fixture row looks like. All three come back `unknown`, each
    naming its reason, and the word `not duplicates` appears nowhere in the output.

    The same rule runs on the corpus itself: the pair list is not filtered down to the
    measurable pairs, because a report that lists only pairs it could score reads as "nothing
    else in the catalogue overlaps" — the coverage hole reporting itself as a clean result,
    which is the failure mode this whole item is about.
    """
    skills, store, tmp = corpus
    five = ["Agent memory", "Graph database", "Neo4j", "Cypher", "Entity resolution"]

    def pair(a, b, sets=None, descs=("", "")):
        row = {"a": a, "b": b, "label": "not_duplicate",
               "label_reason": "synthetic: a pair nobody adjudicated, written to pin the word",
               "source": "#2440 synthetic",
               "descriptions": {"a": descs[0], "b": descs[1]}}
        if sets is not None:
            row["entity_sets"] = {"a": sets[0], "b": sets[1]}
        return row

    pairs = tmp / "pairs.json"
    pairs.write_text(json.dumps({"schema": 1, "as_of": "2026-10-08",
                                 "provenance": {"negatives": "synthetic"},
                                 "pairs": [
                                     pair("alpha", "gamma", (five, []),
                                          ("Agent memory over a Graph database",
                                           "A procedure whose subject the graph never heard")),
                                     pair("neo4j-admin", "cypher-cli", (["Neo4j"], ["Cypher"])),
                                     pair("unlogged-a", "unlogged-b")]},
                                indent=1) + "\n", encoding="utf-8")

    out = tmp / "out" / "entity-overlap.json"
    rc, stdout, stderr = run_report(skills_root=skills, store=store, out=out, pairs=pairs,
                                    data_root=tmp / "lloyd-data")
    assert rc == 0, (stderr or stdout)
    report = read_report(out)
    assert report["zero_entity_skills"] == 1, report

    rows = report["fixture"]["rows"]
    assert len(rows) == 3, rows
    assert [r["verdict"] for r in rows] == ["unknown"] * 3, rows
    assert [r["unknown_reason"] for r in rows] == [
        "no_coverage", "thin", "missing_entity_set"], rows
    assert "verdict=unknown [no_coverage]" in stdout, stdout
    assert "verdict=unknown [thin]" in stdout, stdout
    assert "not duplicates" not in stdout.lower(), stdout
    assert "not_duplicates" not in json.dumps(report).lower(), json.dumps(report)

    # The same rail on the corpus's own pair list: alpha|gamma is present, marked unknown,
    # and the count of unscored pairs is published rather than the list quietly trimmed.
    sample = {(r["a"], r["b"]): r for r in report["unknown_pairs_sample"]}
    assert ("alpha", "gamma") in sample, sorted(sample)
    assert sample[("alpha", "gamma")]["unknown_reason"] == "no_coverage", sample
    assert report["pairs_unknown_by_reason"]["no_coverage"] >= 1, report


def test_surface_forms_resolved_through_the_alias_table_reach_one_canonical(tmp_path):
    """The step the whole item rests on: `Agent Craft` is the same thing as `Agent memory`.

    Without this, three spellings of one thing stay three unrelated strings and every
    judgement downstream inherits the ambiguity — the talk's hinge, and the reason the
    projection reads the graph's alias table instead of the text's vocabulary. The test
    asserts equality with the canonical name rather than with a count, so an alias that
    resolved to a second entity would fail here instead of inflating a set quietly.
    """
    store = make_graph(tmp_path, entities=SUBJECT_ENTITIES,
                       aliases=[("Agent Craft", "Agent memory")])
    from app import kg_store
    kg = kg_store.KGStore(store)
    surfaces = seo.surface_map(kg)
    keys = seo.alias_surfaces(surfaces)
    resolved = seo.project("Built on Agent Craft and Agent memory alike.", surfaces,
                           alias_keys=keys)
    assert resolved["entities"] == {"Agent memory"}, resolved
    assert "agent craft" in surfaces and surfaces["agent craft"] == "Agent memory"


# ── clause 2: what must NOT count as a subject ───────────────────────────────

def test_a_skill_naming_only_harness_tools_and_paths_resolves_no_entities(tmp_path):
    """Clause 2: shared boilerplate is the biggest signal until you suppress it.

    `Bash`, `Read` and `Write` are registered entities in this graph — the live store carries
    them, and this node registers them too, so the fixture is the real shape and not a
    flattering one. Unsuppressed, every runbook that says "call `Read` first" would be
    credited with an opinion about Read, and triage measured the consequence of that on the
    live corpus: a naive longest-match pass put the one adjudicated real pair 8th of 12 at
    0.405, inside the 0.32-0.53 band of the `github-*` look-alikes it exists to separate. The
    path half is the same error in a different costume: `~/obsidian/skills/Neo4j/notes.md`
    names a directory a skill is filed in, not the thing the skill is about.

    The suppressed counters are asserted too, because a screen that silently stopped matching
    would look exactly like a corpus with no coverage, and those two facts need different
    responses.
    """
    skills = tmp_path / "skills"
    plant(skills, "shovel", description="Move files about",
          body="Call Bash to move them, then Read what moved and Write the log. The same "
               "advice is in ~/obsidian/skills/Neo4j/notes.md and skills/bash-runbook/SKILL.md "
               "and /tmp/Write/out.log; use Read again if unsure.")
    store = make_graph(tmp_path, entities=["Bash", "Read", "Write", "Neo4j", "Task", "Edit"])
    out = tmp_path / "report.json"
    rc, stdout, stderr = run_report(skills_root=skills, store=store, out=out,
                               data_root=tmp_path / "lloyd-data")
    assert rc == 0, (stderr or stdout)
    report = read_report(out)
    assert report["zero_entity_skills"] == 1, report
    shovel = [s for s in report["skills"] if s["name"] == "shovel"][0]
    assert shovel["entities"] == [], shovel
    assert "zero_entity_skills 1" in stdout, stdout
    # `tool` is counted before `path`, so the `Write` inside `/tmp/Write/out.log` lands in the
    # tool bucket and the path bucket keeps the two tokens whose only candidate was a real
    # entity spelled as a directory or file name. Both counters are asserted because both
    # screens are load-bearing, and neither number is a claim about how the text reads.
    assert report["suppressed_totals"]["tool"] >= 4, report["suppressed_totals"]
    assert report["suppressed_totals"]["path"] >= 2, report["suppressed_totals"]


def test_the_suppressed_tool_list_covers_every_tool_the_harness_serves():
    """Rail on clause 2: the suppression list cannot quietly fall behind a new tool.

    Scanned from the server modules that define the tools, so a `name="Foo"` added to any
    `agent_mcp/builtin_*.py` without a matching entry here fails this node rather than
    reappearing as a phantom entity in 195 projection sets. A hand-maintained list over an
    open set of served tools is exactly the pattern the 2026-09-20 class rule says cannot
    close a property on its own — so the list is paired with the check that keeps it honest,
    and the six tool names the graph registers with a capital letter are named explicitly,
    because those are the ones that resolve without an alias row.
    """
    served = set()
    for path in sorted((REPO_ROOT / "agent_mcp").glob("builtin_*.py")):
        served |= set(re.findall(r'^\s+name="([A-Za-z_][A-Za-z0-9_]*)",',
                                 path.read_text(encoding="utf-8"), re.MULTILINE))
    assert served, "no served tools found: the scan is broken, not the list"
    missing = sorted(s for s in served if s not in seo.HARNESS_TOOL_SURFACES)
    assert not missing, f"served tools missing from HARNESS_TOOL_SURFACES: {missing}"
    for capital in ["Bash", "Read", "Write", "Edit", "Grep", "Glob"]:
        assert capital in seo.HARNESS_TOOL_SURFACES, capital


# ── clause 3: the fixture, both scores, the ranking, the exit code ───────────

def test_the_committed_fixture_states_its_labels_and_how_they_were_reconstructed():
    """Clause 3, first half: every row carries its verdict, its reason, and its provenance.

    Twelve pairs, not thirteen. The pass that produced them reported **11 flagged pairs, of
    which exactly one was a real scope duplicate** — precision 1/12 — and #2409 names ten of
    the ten look-alikes: `claude-code`/`codex`/`opencode` at 0.53/0.42/0.35, the six `github-*`
    pairs that clear the incumbent's own 0.34 threshold, and the deliberate
    `nightly-reflection-knowledge-analysis`/`-knowledge-write` split at 0.455. Two adjudicated
    positives plus ten named negatives is twelve. The item's "13 pairs" counts an eleventh
    negative that neither #2409 nor run #83 ever names, and inventing a pair to hit that
    number would put a label nobody adjudicated into the instrument that judges the metric.

    Why the fixture is committed at all, and dated: a hand-adjudicated set is evidence about a
    corpus on a day, and the 2026-09-20 class rule says a hand-maintained allowlist cannot
    close a property over an open-set corpus. So `as_of`, `provenance` and one
    `label_reason` per row are load-bearing, not decoration — including the note that the
    2026-10-07 pass scored #2409's pair at 0.03 and the 2026-10-08 pass at 0.327-0.358, the
    disagreement #2409 records verbatim as "one of the two readings is wrong".
    """
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pairs = doc["pairs"]
    assert len(pairs) == 12, len(pairs)
    assert sum(1 for p in pairs if p["label"] == "duplicate") == 2, pairs
    assert sum(1 for p in pairs if p["label"] == "not_duplicate") == 10, pairs
    assert {(p["a"], p["b"]) for p in pairs} >= {
        ("ml-paper-writing", "research-paper-writing"),
        ("memory-capture", "periodic-memory-capture-lloyd"),
        ("claude-code", "codex"),
        ("nightly-reflection-knowledge-analysis", "nightly-reflection-knowledge-write"),
    }
    for pair in pairs:
        assert pair["label"] in ("duplicate", "not_duplicate"), pair
        assert len(pair["label_reason"].strip()) > 40, pair
        assert "2409" in pair["source"] or "2287" in pair["source"] or "83" in pair["source"], pair
    assert doc["as_of"] == "2026-10-08", doc["as_of"]
    prov = json.dumps(doc["provenance"]) + json.dumps(
        [p["source"] for p in pairs if p["label"] == "not_duplicate"])
    for phrase in ("11 flagged pairs", "1/12", "2026-10-07", "0.03"):
        assert phrase in prov, phrase
    assert "hand-maintained allowlist" in json.dumps(doc["provenance"]), doc["provenance"]


def test_the_committed_fixture_scores_the_recorded_figures_and_can_be_reprojected(tmp_path):
    """Clause 3, second half: the printed scores are the adjudication's, and checkable.

    Three claims, in the order a reader can check them.

    1. The shipped ranking is a FAIL line. Scored on subject-matter entities, the #2409 pair
       sits 8th of 12 at 0.2982 — inside the 0.26-0.47 band of the six `github-*` pairs that
       are labelled negatives — while `claude-code`/`codex` sits first at 0.4706. Both
       positives together beat only two negatives. That is measured, it is printed, and the
       run exits 0 anyway, because a threshold tuned after seeing these twelve labels would
       be the fixture grading itself.
    2. The figures are the PRODUCTION store's, resolved on 2026-10-08 and recorded in the
       fixture: `desc_jaccard`, the committed incumbent (the word-set Jaccard behind #83's
       0.34 screen), puts the pair 5th at 0.3400. Pinning both scores is the point of
       publishing them side by side — one instrument does not silently replace the other.
    3. Those recorded sets are re-derivable, not believed: `eval/skill_entity_overlap_pair_
       bodies.json` carries the 24 SKILL.md bodies and the fixture's own `graph` block carries
       the 189 canonical entities and 18 alias rows they resolve through. Re-projecting every
       body against a store built from that block reproduces all 24 sets and every score to
       four decimals, which is what makes `fixture_drift` a warning rather than a surprise.
    """
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))

    rows = {f'{r["a"]}|{r["b"]}': r for r in seo.fixture_rows(FIXTURE)}
    assert len(rows) == 12, rows
    assert all(r["incumbent_name"] == seo.INCUMBENT_NAME for r in rows.values()), rows
    assert {k: (v["entity_score"], v["incumbent_score"]) for k, v in rows.items()} == {
        "claude-code|codex": (0.4706, 0.4839),
        "claude-code|opencode": (0.1724, 0.3871),
        "codex|opencode": (0.1667, 0.3235),
        "github-code-review|github-issues": (0.2222, 0.3409),
        "github-code-review|github-pr-workflow": (0.2647, 0.3864),
        "github-code-review|github-repo-management": (0.1562, 0.375),
        "github-issues|github-pr-workflow": (0.2727, 0.3636),
        "github-issues|github-repo-management": (0.375, 0.4595),
        "github-pr-workflow|github-repo-management": (0.2308, 0.4),
        "memory-capture|periodic-memory-capture-lloyd": (0.7692, 0.0816),
        "ml-paper-writing|research-paper-writing": (0.2982, 0.34),
        "nightly-reflection-knowledge-analysis|nightly-reflection-knowledge-write":
            (0.3111, 0.4091),
    }, rows

    extract = json.loads(FIXTURE.read_text(encoding="utf-8"))["graph"]
    store = make_graph(tmp_path / "graph", entities=extract["entities"],
                       aliases=[tuple(row) for row in extract["alias_rows"]])
    from app import kg_store
    surfaces = seo.surface_map(kg_store.KGStore(store))
    alias_keys = seo.alias_surfaces(surfaces)
    bodies = json.loads(BODIES.read_text(encoding="utf-8"))["bodies"]
    assert seo.fixture_drift(FIXTURE, bodies, surfaces, alias_keys=alias_keys) == []
    for pair in doc["pairs"]:
        for side in ("a", "b"):
            got = seo.project(bodies[pair[side]], surfaces, alias_keys=alias_keys)["entities"]
            assert sorted(got) == pair["entity_sets"][side], (pair["a"], pair["b"], side)

    out = tmp_path / "report.json"
    rc, stdout, stderr = run_report(skills_root=tmp_path / "skills",
                                    store=store, out=out, pairs=FIXTURE,
                                    data_root=tmp_path / "data")
    assert rc == 0, (rc, stderr)
    assert read_report(out)["fixture"]["ranking_first_positives"] == 0, stdout
    assert "RANKING: FAIL" in stdout, stdout
    for name in ("ml-paper-writing | research-paper-writing",
                 "memory-capture | periodic-memory-capture-lloyd"):
        assert name in stdout, stdout


def test_two_runs_over_an_unchanged_corpus_are_byte_identical(tmp_path, corpus):
    """Clause 4's first half: the same bytes in, the same bytes out.

    The incumbent does not have this property in practice, which is the finding #2409 records
    verbatim — the 2026-10-07 pass scored the pair that cost 2,607 lines of tidying at 0.03,
    the 2026-10-08 pass scored it 0.327-0.358, and "one of the two readings is wrong". Both
    the stdout and the report file are compared as bytes, because the report is the artifact
    the nightly reads and a diff between two runs of the same tree is a bug whatever caused
    it: nothing here may consult a clock, a cache, or the operating system's directory order.
    """
    skills, store, data = corpus
    first_out, second_out = tmp_path / "one.json", tmp_path / "two.json"
    rc_a, out_a, _ = run_report(skills_root=skills, store=store, out=first_out,
                                data_root=data / "lloyd-data")
    rc_b, out_b, _ = run_report(skills_root=skills, store=store, out=second_out,
                                data_root=data / "lloyd-data")
    assert (rc_a, rc_b) == (0, 0), (out_a, out_b)
    assert out_a == out_b, (out_a, out_b)
    assert first_out.read_bytes() == second_out.read_bytes()


def test_rewording_one_description_moves_the_incumbent_and_not_the_entity_score(tmp_path,
                                                                               corpus):
    """Clause 4's second half: the key does not move when the prose is re-worded.

    One sentence of `beta`'s description is replaced by hand, the body untouched, and the pair
    is re-scored. The incumbent's number changes because it measures the description; the
    entity score and the entity set do not change because the body never moved. That is the
    property the 0.03-against-0.358 contradiction shows the incumbent lacks, and it is why the
    projection reads bodies while the reported incumbent reads descriptions — a key over the
    same text it is being compared on would make this test tautological instead of measured.
    """
    skills, store, data = corpus
    out = tmp_path / "report.json"

    def row_pair():
        rc, stdout, stderr = run_report(skills_root=skills, store=store, out=out,
                                   data_root=data / "lloyd-data")
        assert rc == 0, (stderr or stdout)
        report = read_report(out)
        pairs = {f'{r["a"]}|{r["b"]}': r for r in report["top_pairs"]}
        return pairs["alpha|beta"], report

    before, before_report = row_pair()
    beta = skills / "beta" / "SKILL.md"
    original = beta.read_text(encoding="utf-8")
    beta.write_text(original.replace(
        "description: Graph-backed recall for agents via Neo4j",
        "description: Recall for assistants powered by a graph store and Neo4j drivers",
        1), encoding="utf-8")
    after, after_report = row_pair()

    assert after["incumbent_score"] != before["incumbent_score"], (before, after)
    assert after["entity_score"] == before["entity_score"], (before, after)
    by_name = {s["name"]: s["entities"] for s in after_report["skills"]}
    original_by_name = {s["name"]: s["entities"] for s in before_report["skills"]}
    assert by_name["beta"] == original_by_name["beta"], (
        "a description rewording changed a body-derived entity set")


# ── clause 5: the run writes a report and nothing else ───────────────────────

def test_the_run_writes_one_report_file_and_leaves_every_skill_alone(tmp_path, corpus):
    """Clause 5: over a corpus copy, exactly one file appears and nothing else moves.

    The comparison is a hash of every file under the skills root against the same tree after
    the run, plus the names on disk — so a merge, a move into the dot-prefixed archive, a
    quarantine, or a re-written front matter all land here, and `status:` in particular is
    checked by value because that is the field a consolidation pass would touch. The item's
    route is explicit about why this belongs in a test rather than in prose: #83 and #58 own
    the merge decision, and an instrument that starts enforcing is an instrument nobody
    trusts the output of. The one file allowed to appear is the report, written outside the
    skills root.
    """
    skills, store, data = corpus
    before = snapshot(skills)
    def statuses():
        found = {}
        for manifest in skills.rglob("SKILL.md"):
            if ".archived" in manifest.parts:
                continue
            found[str(manifest.parent.name)] = re.search(
                r"^status:\s*(\S+)", manifest.read_text(encoding="utf-8"),
                re.MULTILINE).group(1)
        return found

    assert statuses() == {"alpha": "active", "beta": "active", "gamma": "active",
                          "delta": "retired"}, statuses()

    out = tmp_path / "out" / "entity-overlap.json"
    rc, stdout, stderr = run_report(skills_root=skills, store=store, out=out,
                                    data_root=data / "lloyd-data")
    assert rc == 0, (stderr or stdout)

    assert snapshot(skills) == before, "the scoring run modified the corpus"
    assert statuses() == {"alpha": "active", "beta": "active", "gamma": "active",
                          "delta": "retired"}, "the run changed a skill's front matter"
    created = sorted(p for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert created == [out], created
    assert f"report_written {out}" in stderr, stderr
    archived = sorted(m.parent.name for m in (skills / ".archived").rglob("SKILL.md"))
    assert archived == ["old"], f"a scoring run archived a skill: {archived}"


# ── the two committed fixtures must stay re-addable (#2548) ───────────────────
#
# Both artifacts are tracked AND ignored: root `.gitignore:42` is `*.json`, and
# `eval/.gitignore` negated only `dup_calibration.json`, so `git check-ignore -q
# --no-index` exited 0 for both of these and 1 for that control. `tests/
# test_chrome_extension_manifest_tracked.py` is the model — same root rule, same
# nested-negation device, same pair of assertions.

#: Repo-relative, deliberately spelled here rather than derived from `FIXTURE`/
#: `BODIES`: what is under test is the PATH, and `Path.relative_to` would quietly
#: re-derive a moved fixture into passing assertions about its new location.
OVERLAP_FIXTURES = (
    "eval/skill_entity_overlap_pairs.json",
    "eval/skill_entity_overlap_pair_bodies.json",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                          capture_output=True, text=True)


@pytest.mark.parametrize("rel_path", OVERLAP_FIXTURES)
def test_an_overlap_fixture_is_not_ignored(rel_path: str):
    """#2548 clause 1: `git check-ignore -q --no-index` exits 1 for both fixtures.

    Root `.gitignore:42 *.json` matched both files with no negation beside
    `!dup_calibration.json`, so both answered 0 (ignored) where the control answered 1.
    They are in the tree anyway — #2440 added them past the rule — which is exactly why
    this was invisible: the artifact is committed, the eval reads it, and the rule that
    refuses it is still live underneath.

    What that costs is NOT a dropped regeneration. Measured twice in a throwaway repo,
    `git add -A` stages a modification to a tracked file an ignore rule matches, and
    stages `rm` + rewrite as a modification too: git applies ignore rules to untracked
    paths only. The real exposure is a path that is not tracked yet — a renamed fixture
    family, a regenerated sibling, a fresh clone, any `git rm --cached` — which the next
    `git add -A` skips with no word. That is #1701's chrome manifest, and exit 1 here is
    the state in which the next copy of either fixture is addable by plain `git add -A`.

    `-q` is the flag that carries the assertion: bare `-v` exits 0 even when a negation
    matched, so exit code alone would pin nothing. The `-v` line belongs in the message,
    so a failure names WHICH rule answered instead of just the verdict.
    """
    res = _git("check-ignore", "-q", "--no-index", rel_path)
    assert res.returncode == 1, (
        f"{rel_path} is ignored (exit {res.returncode}), so a new copy of it is not "
        f"addable by `git add -A`. Last matching rule: "
        f"{_git('check-ignore', '-v', '--no-index', rel_path).stdout.strip()!r}; "
        f"expected a '!{rel_path.rsplit('/', 1)[-1]}' line from eval/.gitignore to win "
        "as the deeper file's rule"
    )


def test_both_overlap_fixtures_are_tracked_under_eval():
    """#2548 clause 2: both paths appear in `git ls-files eval/`.

    A negation alone is not the guarantee — an untracked file that merely stops being
    ignored is still absent from a fresh clone, and the overlap eval reads both
    artifacts as committed fixtures (`FIXTURE` and `BODIES` above, with the production
    figures copied out of the 2026-10-08 run). The listing is scoped to `eval/` because
    the clause is about where they live: a fixture moved out of that directory is a
    fixture this item did not agree to.
    """
    listed = _git("ls-files", "eval/").stdout.split()
    assert listed, (
        "`git ls-files eval/` returned nothing, so this node pinned an empty listing"
    )
    missing = [p for p in OVERLAP_FIXTURES if p not in listed]
    assert not missing, (
        f"{missing} not tracked under eval/ — a fresh clone ships an overlap eval that "
        f"cannot read its own fixtures; `git ls-files eval/` lists "
        f"{sorted(p for p in listed if p.endswith('.json'))}"
    )


def test_the_json_rule_still_hides_a_stray_file_beside_the_fixtures():
    """Positive control for clause 1, the shape
    `tests/test_chrome_extension_manifest_tracked.py::test_the_ignore_check_can_fail`.

    Exit 1 on the fixtures has two possible causes, and only one of them is the fix: a
    negation that matched, or a root `*.json` rule that stopped answering (a moved line,
    a broadened un-ignore, a `check-ignore` that errored). A stray JSON file in the same
    directory, with no negation of its own, must still exit 0.
    """
    res = _git("check-ignore", "-q", "--no-index", "eval/not-a-fixture.json")
    assert res.returncode == 0, (
        "root `.gitignore:42 *.json` no longer hides a stray JSON file dropped in "
        "eval/ — clause 1's exit 1 would then mean the rule went inert, not that a "
        "negation matched. Last matching rule: "
        f"{_git('check-ignore', '-v', '--no-index', 'eval/not-a-fixture.json').stdout.strip()!r}"
    )


def test_the_new_negations_name_two_exact_filenames_and_no_wildcard():
    """#2548 clause 4: the un-ignore is scoped to filenames, never to a pattern.

    `eval/.gitignore`'s own header states the rule this node enforces — "Scoped to
    filenames and never to `eval/*.json`: a runtime dump dropped here by accident stays
    ignored, which is what the root rule is for." One `!*.json` would undo that for the
    whole directory and the gate's scratch output with it, so the guard is on the shape
    of every negation line in the file, not only on the two added now. (That header line
    said "this one filename" while the file negated one; this item's own change to it is
    the edit quoted above, which is why the quote is verbatim rather than paraphrased.)
    """
    lines = (REPO_ROOT / "eval" / ".gitignore").read_text(encoding="utf-8").splitlines()
    negations = [ln.strip() for ln in lines if ln.strip().startswith("!")]

    overlap = [ln for ln in negations if "overlap" in ln]
    assert overlap == ["!skill_entity_overlap_pairs.json",
                       "!skill_entity_overlap_pair_bodies.json"], (
        f"the overlap negations are {overlap}, expected both exact filenames"
    )
    unscoped = [ln for ln in negations
                if any(c in ln for c in "*?[]") or ln.endswith("/")]
    assert not unscoped, (
        f"negation(s) {unscoped} in eval/.gitignore are patterns, not filenames: "
        "`!*.json` or `!eval/*.json` would re-include every stray runtime dump the root "
        "`*.json` rule exists to hide, which the header above them forbids"
    )
    assert any(ln.lstrip().startswith("#") and "#2440" in ln for ln in lines), (
        "no comment in eval/.gitignore names #2440, so the two lines carry no record of "
        "which instrument's fixtures they un-ignore"
    )


# ── #2549: the RANKING header names each arm's positive ranks ────────────────
#
# The RANKING line used to print scores only, so two arms that failed very
# differently read the same way. On the committed fixture the entity key puts the labelled
# positives at ranks [1, 5] with three negatives above rank 5, while `desc_jaccard` puts
# them at [10, 12] with all ten negatives above them — one word, `FAIL`, for two different
# readings of one corpus. #2440's own history is three such readings in two nights, which is
# why the header has to say which arm failed and by how much rather than that something did.
#
# The nodes below pin the per-arm ranks, the `unknown` exclusion, and that the PASS/FAIL
# semantics and the exit code did not move. The ranks being asserted are re-derived from the
# report's own rows by `expected_rank_line` below, so the assertion is a second measurement
# of the same rows and not an echo of the formatting under test.

#: The arms as the render must name them, with the row field each one ranks on. Spelled out
#: here instead of read from the module's own arm table: the clause is about the NAME a
#: nightly reader sees, so a renamed arm has to fail here, not quietly follow along.
RANK_ARMS = (("entity_jaccard", "entity_score"), ("desc_jaccard", "incumbent_score"))


def expected_rank_line(rows: list[dict], arm: str, score_key: str) -> str:
    """The rank line a reader would hand-derive from the report's `fixture.rows`.

    Recomputed here — measured rows only, ordered by that arm's score, ties by pair name —
    rather than copied out of the module, so this is the reader's arithmetic and the render
    is the thing being checked against it.
    """
    ranked = sorted((r for r in rows if r["verdict"] != "unknown"),
                    key=lambda r: (-r[score_key], r["a"], r["b"]))
    positives = [i for i, r in enumerate(ranked, 1) if r["label"] == "duplicate"]
    lowest = max(positives) if positives else None
    above = sum(1 for i, r in enumerate(ranked, 1)
                if r["label"] != "duplicate" and lowest is not None and i < lowest)
    tail = (f"{above} negative pair(s) outrank the lowest positive (rank {lowest})"
            if lowest is not None else "no measurable positive pair to rank against")
    return (f"RANKS {arm}: positives at ranks {positives} of {len(ranked)} ranked row(s) "
            f"({len(rows) - len(ranked)} unmeasured row(s) unranked); {tail}")


def rank_lines_of(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if ln.startswith("RANKS ")]


def synth_pair(a: str, b: str, *, a_set, b_set, desc_a: str, desc_b: str,
               label: str, reason: str) -> dict:
    """One fixture row written to pin a rank, not to describe a real skill pair."""
    return {"a": a, "b": b, "label": label, "label_reason": reason,
            "source": "#2549 synthetic: written to pin the rank header, not a #2409 pair",
            "descriptions": {"a": desc_a, "b": desc_b},
            "entity_sets": {"a": list(a_set), "b": list(b_set)}}


def write_pairs(path: Path, pairs: list[dict]) -> Path:
    path.write_text(json.dumps({"schema": 1, "as_of": "2026-10-11",
                                "provenance": {"negatives": "synthetic #2549 rows"},
                                "pairs": pairs}, indent=1) + "\n", encoding="utf-8")
    return path


#: Six canonical names, so both sides of every synthetic pair below clear the floor of 5 and
#: are measurable unless the row is the deliberately thin one.
SIX = ["E1", "E2", "E3", "E4", "E5", "E6"]

#: Four measured rows with hand-checkable scores. Entity arm: `rank-pos-a` 5/7 = 0.7143,
#: `rank-pos-b` 4/8 = 0.5, `rank-neg-a` 3/9 = 0.3333, `rank-neg-b` 2/10 = 0.2 — positives at
#: [1, 2], nothing above. Incumbent arm: 6/7 = 0.8571, 2/3 = 0.6667, 3/5 = 0.6, 2/4 = 0.5 —
#: the same [1, 2]. Both arms therefore PASS, and the FAIL fixture below is these rows with
#: one negative moved between the positives.
def measured_rows() -> list[dict]:
    return [
        synth_pair("rank-pos-a", "rank-pos-b", a_set=SIX,
                   b_set=["E1", "E2", "E3", "E4", "E5", "E7"],
                   desc_a="alpha one two three four five",
                   desc_b="alpha one two three four five six", label="duplicate",
                   reason="synthetic positive, ranked first under both arms"),
        synth_pair("rank-pos-b", "rank-pos-c", a_set=SIX,
                   b_set=["E1", "E2", "E3", "E4", "E8", "E9"],
                   desc_a="beta one two three", desc_b="beta one two",
                   label="duplicate",
                   reason="synthetic positive, ranked second under both arms"),
        synth_pair("rank-neg-a", "rank-neg-b", a_set=SIX,
                   b_set=["E1", "E2", "E3", "E10", "E11", "E12"],
                   desc_a="gamma one two", desc_b="gamma one two three four",
                   label="not_duplicate",
                   reason="synthetic negative, ranked third under both arms"),
        synth_pair("rank-neg-b", "rank-neg-c", a_set=SIX,
                   b_set=["E1", "E2", "E10", "E11", "E12", "E13"],
                   desc_a="delta one", desc_b="delta one two three",
                   label="not_duplicate",
                   reason="synthetic negative, ranked last under both arms"),
    ]


def test_the_ranking_header_prints_each_arm_s_positive_ranks(tmp_path):
    """Clauses 1 and 2: the header names both arms, their positive ranks, and the squeeze.

    Run over the committed fixture across the CLI, because the render is what the nightly
    reads. The two figures asserted verbatim are the item's own, measured off the twelve
    labelled rows: the entity key has its positives at [1, 5] with three negatives above
    rank 5 (`claude-code|codex`, `github-issues|github-repo-management` and
    `nightly-reflection-knowledge-analysis|-knowledge-write`), and the incumbent named
    `desc_jaccard` has them at [10, 12] with all ten negatives above rank 12. Two lines
    directly beneath the RANKING verdict, one per arm — that placement is the clause, so it
    is asserted by index and not by substring.
    """
    extract = json.loads(FIXTURE.read_text(encoding="utf-8"))["graph"]
    store = make_graph(tmp_path / "graph", entities=extract["entities"],
                       aliases=[tuple(row) for row in extract["alias_rows"]])
    out = tmp_path / "report.json"
    rc, stdout, stderr = run_report(skills_root=tmp_path / "skills", store=store,
                                    out=out, pairs=FIXTURE, data_root=tmp_path / "data")
    assert rc == 0, (rc, stderr)
    rows = read_report(out)["fixture"]["rows"]

    lines = stdout.splitlines()
    verdict_at = [i for i, ln in enumerate(lines) if ln.startswith("RANKING:")]
    assert len(verdict_at) == 1, lines
    i = verdict_at[0]
    assert lines[i + 1].startswith("RANKS entity_jaccard:"), lines[i:i + 4]
    assert lines[i + 2].startswith("RANKS desc_jaccard:"), lines[i:i + 4]

    assert ("RANKS entity_jaccard: positives at ranks [1, 5] of 12 ranked row(s) "
            "(0 unmeasured row(s) unranked); 3 negative pair(s) outrank the lowest positive "
            "(rank 5)" in stdout), stdout
    assert ("RANKS desc_jaccard: positives at ranks [10, 12] of 12 ranked row(s) "
            "(0 unmeasured row(s) unranked); 10 negative pair(s) outrank the lowest positive "
            "(rank 12)" in stdout), stdout
    for arm, score_key in RANK_ARMS:
        assert expected_rank_line(rows, arm, score_key) in stdout, (arm, stdout)


def test_an_unmeasurable_pair_gets_no_rank_and_never_counts_above_the_lowest_positive(
        tmp_path):
    """Clause 3: ranks are assigned over measured rows only, in both arms.

    The fifth row is the trap. Its two entity sets are four elements each — under the floor
    of 5, so the pair is `unknown` — but a `thin` row still carries the quotient of those two
    sets, and identical sets make it 1.0 in the entity arm and 1.0 in the incumbent arm too.
    Rank unmeasurable rows and that row takes first place under BOTH arms, pushing the real
    positives to [2, 3] and reporting one negative above the lowest positive for a pair this
    run never measured. The printed line has to say [1, 2] with nothing above, and has to
    count the row as unmeasured rather than silently dropping it.
    """
    pairs = write_pairs(tmp_path / "pairs.json", measured_rows() + [
        synth_pair("rank-thin-a", "rank-thin-b",
                   a_set=["E1", "E2", "E3", "E4"], b_set=["E1", "E2", "E3", "E4"],
                   desc_a="trap one two three four", desc_b="trap one two three four",
                   label="not_duplicate",
                   reason="synthetic unmeasurable row: below the floor, yet 1.0 both ways"),
    ])
    store = make_graph(tmp_path, entities=SIX + ["E7", "E8", "E9", "E10", "E11", "E12",
                                                 "E13"])
    out = tmp_path / "report.json"
    rc, stdout, stderr = run_report(skills_root=tmp_path / "skills", store=store,
                                    out=out, pairs=pairs, data_root=tmp_path / "data")
    assert rc == 0, (rc, stderr)
    rows = read_report(out)["fixture"]["rows"]

    trap = [r for r in rows if r["a"] == "rank-thin-a"][0]
    assert trap["verdict"] == "unknown" and trap["unknown_reason"] == "thin", trap
    assert trap["entity_score"] == 1.0 and trap["incumbent_score"] == 1.0, (
        "the trap row stopped carrying a top score in both arms, so this node no longer "
        "proves an unmeasured row is unranked:", trap)

    assert rank_lines_of(stdout) == [
        "RANKS entity_jaccard: positives at ranks [1, 2] of 4 ranked row(s) "
        "(1 unmeasured row(s) unranked); 0 negative pair(s) outrank the lowest positive "
        "(rank 2)",
        "RANKS desc_jaccard: positives at ranks [1, 2] of 4 ranked row(s) "
        "(1 unmeasured row(s) unranked); 0 negative pair(s) outrank the lowest positive "
        "(rank 2)",
    ], stdout
    for arm, score_key in RANK_ARMS:
        assert expected_rank_line(rows, arm, score_key) in stdout, (arm, stdout)


def test_the_ranks_are_printed_without_moving_the_ranking_verdict_or_exit_code(tmp_path):
    """Clause 4: PASS still means the lowest positive outranks the highest negative.

    Two synthetic corpora, run the way the nightly runs the real one. The first is the four
    measured rows, where both arms put the positives first: `RANKING: PASS`, ranks [1, 2]
    under both arms, exit 0. The second moves one negative between the positives of the
    entity arm (5/8 = 0.625, over the second positive's 4/8 = 0.5) while leaving the
    incumbent arm's positives first: the verdict is `RANKING: FAIL`, the entity arm prints
    [1, 3] with one negative above rank 3, and `desc_jaccard` still prints [1, 2] with
    nothing above. That divergence is the whole point of the change, and it is also what the
    scores-only header could not show. Neither run exits non-zero, and the key the JSON has
    always carried for the verdict — `ranking_first_positives`, pinned to 0 over the
    committed FAIL fixture above — still carries the pass bool: 1 for PASS, 0 for FAIL.
    """
    store = make_graph(tmp_path, entities=SIX + ["E7", "E8", "E9", "E10", "E11", "E12",
                                                 "E13"])
    squeezed = [row for row in measured_rows() if row["a"] != "rank-neg-a"] + [
        synth_pair("rank-neg-a", "rank-neg-c", a_set=SIX,
                   b_set=["E1", "E2", "E3", "E4", "E5", "E10", "E11"],
                   desc_a="zeta one two three four five six seven eight nine",
                   desc_b="zeta ten", label="not_duplicate",
                   reason="synthetic negative squeezed between the positives of the "
                          "entity arm only"),
    ]

    cases = [("pass", measured_rows(), "RANKING: PASS", 1),
             ("fail", squeezed, "RANKING: FAIL", 0)]
    for tag, pairs_rows, verdict, first_positives in cases:
        pairs = write_pairs(tmp_path / f"{tag}-pairs.json", pairs_rows)
        out = tmp_path / f"{tag}.json"
        rc, stdout, stderr = run_report(skills_root=tmp_path / "skills", store=store,
                                        out=out, pairs=pairs,
                                        data_root=tmp_path / "data")
        assert rc == 0, (tag, rc, stderr)
        assert verdict in stdout, (tag, stdout)
        assert read_report(out)["fixture"]["ranking_first_positives"] == first_positives, \
            (tag, read_report(out)["fixture"])
        rows = read_report(out)["fixture"]["rows"]
        for arm, score_key in RANK_ARMS:
            assert expected_rank_line(rows, arm, score_key) in stdout, (tag, arm, stdout)

    _, fail_stdout, _ = run_report(
        skills_root=tmp_path / "skills", store=store, out=tmp_path / "again.json",
        pairs=write_pairs(tmp_path / "fail-again.json", squeezed),
        data_root=tmp_path / "data")
    assert ("RANKS entity_jaccard: positives at ranks [1, 3] of 4 ranked row(s) "
            "(0 unmeasured row(s) unranked); 1 negative pair(s) outrank the lowest positive "
            "(rank 3)" in fail_stdout), fail_stdout
    assert ("RANKS desc_jaccard: positives at ranks [1, 2] of 4 ranked row(s) "
            "(0 unmeasured row(s) unranked); 0 negative pair(s) outrank the lowest positive "
            "(rank 2)" in fail_stdout), fail_stdout
