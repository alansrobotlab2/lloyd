"""The retired-name → durable-consumer lookup (#3538) — the matcher half.

The gate's other rungs ask the code tree. This one asks the vault, so these tests
are about the two ways a lookup over prose goes wrong: matching something that is
not the retired thing (a live path that merely ENDS in the retired one), and
reporting a mention as a consumer (a note that says the path is dead). #882's
triage measured ~11% precision on a prose-cite extractor, which is what the
absence-prose exemption and the root anchoring are here to answer — so this file
is written as the precision instrument, not as coverage.

The `clause N` prefixes are #3538's acceptance clauses, which the gate's review
rung grades one at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.automod import prose_consumers as PC
from scripts.automod import typecheck as TC

# The retired spellings come from history and from this machine's real home:
# #3538's triage measured both of them there, and an absolute path means nothing
# against a made-up root.
HOME = Path("/home/alansrobotlab")
RETIRED_ROOT = "~/lloyd/_pipeline/"
#: With its trailing slash: `Path()` renders one away, and the slash is what makes
#: a retired ROOT differ from a retired file.
ABS_ROOT = str(HOME / "lloyd/_pipeline") + "/"
RETIRED_STORE = "~/lloyd/_pipeline/vault-derived/kg.sqlite"
ABS_STORE = str(HOME / "lloyd/_pipeline/vault-derived/kg.sqlite")
VENV_ABS = str(HOME / "lloyd/.venvs/lloyd/bin/python")
LIVE_STORE = "~/lloyd-data/_pipeline/vault-derived/kg.sqlite"

#: The live instruction file #3538's triage named as the usable seed for clause 5.
SEED_TASK_FILE = HOME / "obsidian" / "autonomy" / "70-skill-lint.md"

#: One durable corpus, one builder — clause 5's fixture copy and clause 3's
#: lookalike have to be the same corpus or the two tests are not testing a thing.
CORPUS = {
    # A real consumer: a task file that stores the command verbatim.
    "autonomy/70-skill-lint.md": [
        "---", "name: skill-lint", "---", "",
        "## Command", "/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python "
        "/home/alansrobotlab/lloyd/scripts/skill_lint.py --all"],
    # The item's named non-consumer: the LIVE root, carrying the retired string
    # as a suffix. Must never be reported for a retired `~/lloyd/` path.
    "skills/kg-mention-classifier/SKILL.md": [
        "# classifier", "", f"Read `{LIVE_STORE}` first."],
    # A genuine consumer of the retired ROOT, in the instruction corpus: without
    # this the corpus would prove only that nothing matches, never that the
    # anchoring is what made the other two files not match.
    "skills/root-consumer/SKILL.md": ["# root", "", f"Write the dump under "
                                      f"`{ABS_ROOT}`."],
    # A live archive path that contains the retired ABSOLUTE path byte for byte.
    "skills/data-root-survey/SKILL.md": [
        "# survey", "", f"The dump is under `/archive{ABS_STORE}`."],
    # A file naming the retired root to say it is dead: a mention, not a consumer.
    "skills/stale-note/SKILL.md": [
        "# stale", "", f"The old root `{RETIRED_ROOT}` is retired; everything is "
        "under `~/lloyd-data/` now."],
    # Record corpus: correct to count, wrong to list as actionable.
    "backlog/882-prose-cites.md": ["# 882", "", f"Found by the resolver: `USER.md` "
                                   f"cited {ABS_STORE}."],
    "memory/learnings/USER.md": ["# user", "", "The venv lives in `.venvs/lloyd`."],
    # Neither corpus: nothing runs a project note.
    "projects/notes.md": ["# notes", "", f"Left-over backup at {ABS_STORE}."],
    # A generated report, out of corpus for the same reason.
    "autonomy/referential-integrity-latest.md": ["# report", "", f"cited {ABS_STORE}"],
}


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "obsidian"
    for rel, lines in CORPUS.items():
        p = v / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return v


@pytest.fixture
def trees(tmp_path):
    """`(head, base, changed, diff)` for a round that deletes one module-level
    function and repoints two path literals — a root and an interpreter."""
    base, head = tmp_path / "base", tmp_path / "head"
    body_base = (
        f'KG_DB_ROOT = "{RETIRED_ROOT}"\n'
        f'PY = "{VENV_ABS}"\n'
        "KEEP = 1\n"
        "def retired_worker_entrypoint():\n    return KEEP\n"
    )
    body_head = 'KG_DB_ROOT = "~/lloyd-data/_pipeline/"\nPY = "python3"\nKEEP = 1\n'
    for tree, body in ((base, body_base), (head, body_head)):
        (tree / "app").mkdir(parents=True)
        (tree / "app" / "m.py").write_text(body, encoding="utf-8")
    diff = "\n".join([
        "--- a/app/m.py", "+++ b/app/m.py", "@@ -1,4 +1,3 @@",
        f'-KG_DB_ROOT = "{RETIRED_ROOT}"',
        '+KG_DB_ROOT = "~/lloyd-data/_pipeline/"',
        f'-PY = "{VENV_ABS}"',
        '+PY = "python3"',
        " KEEP = 1",
        "-def retired_worker_entrypoint():",
        "-    return KEEP",
    ]) + "\n"
    return head, base, ["app/m.py"], diff


def _scan(trees, vault, **kw):
    head, base, changed, diff = trees
    return PC.scan(head, base, changed, diff, checkout=head, vault=vault,
                   home=HOME, **kw)


def _one_path(tmp_path, vault, literal: str):
    """A scan whose diff retires exactly one path spelling and nothing else."""
    tree = tmp_path / "t"
    (tree / "app").mkdir(parents=True, exist_ok=True)
    (tree / "app" / "m.py").write_text("X = 1\n", encoding="utf-8")
    return PC.scan(tree, None, ["app/m.py"], f'-KG = "{literal}"\n', checkout=tree,
                   vault=vault, home=HOME)


# ── clause 2: the candidate list, emitted whether or not it finds anything ───

def test_clause_2_the_candidate_list_names_everything_the_diff_retires(trees, vault):
    """The list is the rung's own claim about the diff, so a wrong one has to be
    readable in the round event rather than guessed at from an empty hit list."""
    res = _scan(trees, vault)
    by_name = {c["name"]: c for c in res.candidates}
    assert by_name["retired_worker_entrypoint"]["kind"] == PC.KIND_SYMBOL
    assert by_name["retired_worker_entrypoint"]["origin"] == "app.m"
    assert by_name[RETIRED_ROOT]["kind"] == PC.KIND_PATH
    assert by_name[VENV_ABS]["kind"] == PC.KIND_PATH
    assert res.counts["candidates"] == len(res.candidates)


def test_clause_2_a_retired_root_is_searched_in_every_root_resolved_spelling(
        trees, vault):
    """Prose cites a root as `~/…` or absolutely and the diff wrote one spelling;
    both are searched, and neither is ever widened or shortened into another path.
    """
    res = _scan(trees, vault)
    c = next(c for c in res.candidates if c["name"] == RETIRED_ROOT)
    assert sorted(c["forms"]) == sorted([RETIRED_ROOT, ABS_ROOT]), c
    assert not any(f.endswith("vault-derived/kg.sqlite") for f in c["forms"]), (
        "a ROOT candidate must not be widened into one of its own files")


def test_clause_2_a_name_that_survived_the_round_is_not_retired(trees, vault):
    """`KEEP` is in the diff's context and its value moved; prose naming it is
    fine. Putting it in the list would make every live file a hit."""
    res = _scan(trees, vault)
    assert "KEEP" not in {c["name"] for c in res.candidates}


def test_clause_2_a_candidate_too_short_to_search_is_recorded_as_refused(trees, vault,
                                                                         monkeypatch):
    """`env` is three characters and #882's ablation is what grepping one across
    the vault produces — so short names are refused, in writing. A silently dropped
    candidate is a hole a reader cannot audit, which is the opposite of clause 2."""
    monkeypatch.setattr(PC, "MIN_SYMBOL_CHARS", 26)
    res = _scan(trees, vault)
    c = next(c for c in res.candidates if c["name"] == "retired_worker_entrypoint")
    assert "forms" not in c and "shorter than 26 characters" in c["not_searched"], c
    assert res.counts["not_searched"] == sum(1 for x in res.candidates
                                             if "not_searched" in x)
    assert res.counts["searched"] == res.counts["candidates"] - \
        res.counts["not_searched"]


@pytest.mark.parametrize("literal", [
    "~/lloyd-data/{env}/x.py",      # a template, not a path
    "*/vault-derived/*.sqlite",     # a glob
    "https://example.com/lloyd/x",  # a URL with slashes in it
    "not a path at all",            # prose with spaces
    "kg.sqlite",                    # a bare file name: no root to resolve
])
def test_clause_2_a_removed_literal_that_cannot_be_a_path_is_refused(literal,
                                                                    tmp_path):
    """#882's `_TEMPLATE` refuses the same shapes: `{` and `*` in a candidate match
    a formatting instruction, not a file, and a rootless name is #882's 196 files."""
    assert PC.path_candidate(literal, checkout=tmp_path, home=HOME) is None


def test_clause_2_an_unreadable_vault_is_unevaluated_rather_than_clean(tmp_path,
                                                                       trees):
    """The corpus can be missing while the diff retires plenty. A rung that reads
    that as "no consumers" is the one failure mode this class cannot afford, so it
    is UNEVALUATED with its reason — the rule `pyright` runs on."""
    head, base, changed, diff = trees
    res = PC.scan(head, base, changed, diff, checkout=head,
                  vault=tmp_path / "not-here", home=HOME)
    assert res.status == PC.UNEVALUATED and res.reason


# ── clause 3: a hit carries the retired token, root and all ─────────────────

def test_clause_3_a_live_path_ending_in_the_retired_one_is_not_a_consumer(tmp_path,
                                                                         vault):
    """THE precision rule, and the exact mistake #3538's triage tripped: grepping
    the retired store without its root returned 196 vault files, almost all of them
    the LIVE root carrying the retired string as a suffix. `skills/kg-mention-
    classifier/SKILL.md:17` names `~/lloyd-data/_pipeline/vault-derived/kg.sqlite`
    and must not be reported to a round that retired `~/lloyd/...`."""
    res = _one_path(tmp_path, vault, RETIRED_STORE)
    assert not any("kg-mention-classifier" in h["file"] for h in res.hits), res.hits


def test_clause_3_a_longer_absolute_path_holding_the_retired_one_is_not_one(tmp_path,
                                                                           vault):
    """`/archive/home/alansrobotlab/lloyd/_pipeline/…` contains the retired
    absolute path byte for byte and is a different file. The grep DOES return this
    line — which is why the assertion is on the hits *and* on the grep having run:
    without the second half this test would pass on a matcher that saw nothing."""
    res = _one_path(tmp_path, vault, ABS_STORE)
    assert res.counts["grep_lines"] >= 1, res.counts
    assert not any("data-root-survey" in h["file"] for h in res.hits), res.hits


def test_clause_3_naming_the_retired_root_itself_is_a_hit(vault, tmp_path):
    """The counter-case to the two above, so what is being asserted is the
    anchoring and not a matcher that never fires: the retired absolute root,
    written as itself, is reported — with the retired token as the matched text."""
    res = _one_path(tmp_path, vault, ABS_ROOT)
    hits = {h["file"]: h for h in res.hits}
    assert "skills/root-consumer/SKILL.md" in hits, (res.hits, res.counts)
    assert "skills/data-root-survey/SKILL.md" not in hits, res.hits
    assert "backlog/882-prose-cites.md" not in hits, "record corpus is not listed"
    assert res.counts["record_hits"] >= 1, res.counts
    assert all(h["matched"] == ABS_ROOT for h in res.hits), res.hits


# ── clause 4: two corpora, two numbers ──────────────────────────────────────

def test_clause_4_instruction_and_record_hits_are_counted_separately(trees, vault):
    """`rewrite_vault_data_paths.py:14-17` rules that only task files and SKILL.md
    files are instructions and the rest is the record of what was true when
    written. #3538 measured the record corpus at ~91% of the files naming the venv
    spelling, so one number over both would be mostly noise by construction."""
    res = _scan(trees, vault)
    assert res.counts["instruction_hits"] >= 1, res.counts
    assert res.counts["record_hits"] >= 1, res.counts
    assert all(h["file"].startswith(("skills/", "autonomy/")) for h in res.hits), \
        "only the instruction corpus is listed as actionable: " + str(res.hits)
    assert not any(h["file"].startswith(("backlog/", "memory/", "projects/"))
                   for h in res.hits), res.hits
    assert res.top_record, "the record corpus is counted and its files named"


def test_clause_4_every_listed_hit_carries_what_a_fixer_needs(trees, vault):
    """`{name, kind, file, line, matched text}`: a file alone is not a consumer —
    whoever fixes the citation has to see the line and which name drew it."""
    res = _scan(trees, vault)
    assert res.hits
    for h in res.hits:
        assert set(h) == {"name", "kind", "file", "line", "matched", "text"}, h
        assert isinstance(h["line"], int) and h["line"] > 0, h
        assert h["matched"] in h["text"] and h["name"], h
        assert str(vault) not in h["file"], "hits are vault-relative"


def test_clause_4_a_line_saying_the_thing_is_dead_is_a_mention_not_a_consumer(
        trees, vault):
    """`skills/stale-note/SKILL.md` names the retired root in order to say it is
    retired. Counting that is #882's error — and it is the case a blocking rung
    would get wrong forever, which is why the exemption is shared with the
    resolver rather than reinvented here."""
    res = _scan(trees, vault)
    assert not any("stale-note" in h["file"] for h in res.hits), res.hits
    assert res.counts["instruction_exempt"] >= 1, res.counts


@pytest.mark.parametrize("rel,want", [
    ("skills/foo/SKILL.md", "instruction"),
    ("autonomy/70-skill-lint.md", "instruction"),
    ("autonomy/referential-integrity-latest.md", ""),
    ("autonomy/nested/70-skill-lint.md", ""),
    ("skills/_archived/old/SKILL.md", ""),
    ("backlog/94-x.md", "record"),
    ("knowledge/ai/x.md", "record"),
    ("memory/learnings/MEMORY.md", "record"),
    ("lloyd/CLAUDE.md", "record"),
    ("projects/x.md", ""),
])
def test_clause_4_the_corpus_split_is_a_function_of_the_path(rel, want):
    """A generated autonomy REPORT is out of corpus: nothing runs it, and a
    report's own quote of a dead path is the sweep noticing, not a consumer."""
    assert PC.classify(rel) == want


# ── clause 5: the seeded rename, and the seeded root that must stay silent ───

def test_clause_5_the_seeded_venv_rename_reports_a_real_autonomy_task_file(tmp_path,
                                                                          vault):
    """The acceptance check: a round that retires the `.venvs/lloyd/bin/python`
    spelling must name a task file that stores the command verbatim. The live
    `autonomy/70-skill-lint.md` is read from the vault into the fixture corpus, so
    the fixture cannot drift from the file it stands in for — and the gate itself
    must never write there."""
    if not SEED_TASK_FILE.exists():
        pytest.skip(f"the seeded task file is gone on this box: {SEED_TASK_FILE}")
    live = SEED_TASK_FILE.read_text(encoding="utf-8", errors="replace")
    assert VENV_ABS in live, (
        "the seed is #3538's own example; if it no longer names the spelling the "
        "clause needs a new seed, not a new assertion")
    (vault / "autonomy" / "70-skill-lint.md").write_text(live, encoding="utf-8")
    res = _one_path(tmp_path, vault, VENV_ABS)
    assert any(h["file"].startswith("autonomy/") and "70-skill-lint" in h["file"]
               for h in res.hits), (
        f"clause 5: a real ~/obsidian/autonomy/ task file must be reported; "
        f"hits={res.hits} counts={res.counts}")


def test_clause_5_the_seeded_retired_root_reports_nothing_live(tmp_path, vault):
    """The other half of the seed. #3538's triage found the retired data roots
    (`~/lloyd/_pipeline/` and friends) match ZERO live instruction files, because
    `rewrite_vault_data_paths.py` already repointed them. A rung that reported the
    repointed live files as consumers of the retired root would be refusing good
    rounds; the anchoring is what stops it, and this is the clause that says so."""
    res = _one_path(tmp_path, vault, RETIRED_ROOT)
    assert not any("kg-mention-classifier" in h["file"] for h in res.hits), res.hits
    assert not any("lloyd-data" in h["text"] for h in res.hits), res.hits
    assert any("root-consumer" in h["file"] for h in res.hits), (
        "a file still told to read inside the retired root IS a consumer, and "
        "reporting nothing for a retired root would be the other failure mode")


# ── the retired-name extractor ──────────────────────────────────────────────

def test_the_retired_set_is_a_difference_of_module_level_names(tmp_path):
    """`retired_module_symbols` keeps only what is GONE.
    `changed_module_symbols` reports a changed body too, and prose naming a
    renamed-but-still-present symbol is not broken — so the two questions cannot
    share one function."""
    base, head = tmp_path / "b", tmp_path / "h"
    (base / "app").mkdir(parents=True)
    (head / "app").mkdir(parents=True)
    (base / "app" / "m.py").write_text(
        "GONE = 1\ndef kept():\n    return 3\nclass T:\n"
        "    def also_gone(self):\n        pass\n", encoding="utf-8")
    (head / "app" / "m.py").write_text("def kept():\n    return 4\n",
                                       encoding="utf-8")
    got = TC.retired_module_symbols(head, base, ["app/m.py"])
    assert got == {"app.m": {"GONE", "T", "T.also_gone"}}, got


def test_a_deleted_module_retires_its_whole_interface(tmp_path):
    """`changed_module_symbols` needs a HEAD image to compare against and so
    reports nothing for a DELETED file — which is the largest retirement a round
    can make, and exactly the class this rung exists for."""
    base, head = tmp_path / "b", tmp_path / "h"
    (base / "app").mkdir(parents=True)
    (head / "app").mkdir(parents=True)
    (base / "app" / "gone.py").write_text("ENTRY = 1\n", encoding="utf-8")
    assert TC.retired_module_symbols(head, base, ["app/gone.py"]) == {
        "app.gone": {"ENTRY"}}


def test_an_unreadable_image_retires_nothing(tmp_path):
    """A syntax error in either image means the extractor cannot tell, and
    "cannot tell" is not evidence that a name is gone."""
    base, head = tmp_path / "b", tmp_path / "h"
    (base / "app").mkdir(parents=True)
    (head / "app").mkdir(parents=True)
    (base / "app" / "m.py").write_text("def broken(:\n", encoding="utf-8")
    (head / "app" / "m.py").write_text("X = 1\n", encoding="utf-8")
    assert TC.retired_module_symbols(head, base, ["app/m.py"]) == {}


def test_the_result_serialises_for_the_round_event(trees, vault):
    """The detail and the ledger row are written from this dict by the gate, on a
    thread with no human between the rung and `json.dumps`."""
    d = _scan(trees, vault).to_dict()
    assert json.loads(json.dumps(d))["status"] == PC.EVALUATED
    assert {"status", "candidates", "counts", "hits"} <= set(d)
    assert PC.detail(_scan(trees, vault)).startswith("observe-only")


# ── the ledger bound: a rename can be cited everywhere ──────────────────────

def test_one_retired_name_cited_in_more_files_than_the_cap_is_counted_in_full(
        tmp_path):
    """"The moved root is named in 697 files" (#1928's measurement of exactly this
    class) is the round this bound exists for. The clause is that one rename cannot
    flood the ledger, and the way to satisfy it without hiding anything is to cap
    the LISTED hits and count all of them: `hits` carries at most MAX_HITS_LISTED
    entries, `counts` always carries the true number, and the rung's line reports
    the true number, so the clipped detail never reads as a small fix."""
    v = tmp_path / "obsidian"
    n = PC.MAX_HITS_LISTED + 5
    for i in range(n):
        p = v / f"skills/s{i:02d}" / "SKILL.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# task {i}\n\nRun `{VENV_ABS}` --check.\n", encoding="utf-8")
    res = _one_path(tmp_path, v, VENV_ABS)

    assert res.counts["instruction_hits"] == n, res.counts
    assert len(res.hits) == PC.MAX_HITS_LISTED, len(res.hits)
    assert res.counts["hits_listed"] == PC.MAX_HITS_LISTED, res.counts
    # The files listed are a deterministic prefix, not whichever files the grep
    # happened to finish first: two runs of the same round have to agree.
    files = [h["file"] for h in res.hits]
    assert files == sorted(files)[:PC.MAX_HITS_LISTED], files
    line = PC.detail(res)
    assert f"instruction {n} actionable hit(s)" in line, line
    assert "corpus capped" not in line, (
        "the hits were capped, not the corpus — saying so would send a reader to "
        "the wrong knob")


def test_an_archived_skill_is_not_an_instruction(tmp_path):
    """The vault holds archives at every depth, and the live measurement of this
    rung on 2026-11-03 caught the bug the naive exclusion list has: the skills
    archive is `skills/.archived/`, not `skills/_archived/`, so one spelling left
    archived instructions in the actionable count — a file nothing loads cannot be
    broken by retiring a name, and every one of those is a false positive."""
    v = tmp_path / "obsidian"
    for rel in ("skills/live/SKILL.md", "skills/.archived/dead/SKILL.md",
                "backlog/_archived/old-note.md", "autonomy/_archived/9-old.md"):
        p = v / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# x\n\nRun `{VENV_ABS}` --check.\n", encoding="utf-8")
    res = _one_path(tmp_path, v, VENV_ABS)
    assert [h["file"] for h in res.hits] == ["skills/live/SKILL.md"], res.hits
    assert res.counts["instruction_hits"] == 1, res.counts
    assert res.counts["out_of_corpus_hits"] == 3, res.counts
    assert res.counts["record_hits"] == 0, res.counts
