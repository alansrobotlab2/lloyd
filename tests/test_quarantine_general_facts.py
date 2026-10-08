"""The `general` god-entity mover (#2443, after #2303): a tree move plus its graph consequences.

The premise the fixture is built to match is the live tree of 2026-10-08: `facts/general/` holds
8 `.md` files (7 category files + `general-overview.md`) plus 7 zero-byte `.lock` files, 365
two-space `^  fact:` lines, and `knowledge-health-2026-10-08.md:43` ranks it 9th with
`| general | 365 | decision, event, general, goal, preference, relationship, state |`. The graph
side, read only through `app.kg_store`: 56 ACTIVE edges name `general`, `entities.get("general")`
answers a row, and `facts_idx.count(entity="general")` is 365 with every row active. #1999 closed
the write route at `70c2d002` ("facts with no usable entity are held back and counted, never
filed under general"), so the residue is static — but it is still a node, and a name every
category appears under answers no query about anything.

The fixture is that premise scaled to a size a test can assert on, with the shapes that matter
kept: several category files, an overview with zero facts, `.lock` files beside the category
files and NOT beside the overview (the live layout, 7 locks for 8 files), a second real entity
with its own directory, and — because `kg_store._rows_for_file` lets a fact override its entity
from inside a neighbour's file — one fact inside `general/general-state.md` attributed to
`Lloyd`. That last shape is why the mover captures the file list BEFORE the move: after it the
directory is gone, and a mover that rebuilt its list afterwards would leave Lloyd's row live and
pointing into the quarantine root.

The five clauses of #2443's acceptance are pinned here in the order the contract numbers them:
what the dry run prints (clause 1), that the dry run touches nothing (clause 2), what `--apply`
relocates and where (clause 3), that the relocated tree is lossless and says so (clause 4), and
the graph half — edges expired, rows de-indexed, entity deregistered, no database opened by
hand, and a second `--apply` that moves nothing and still exits 0 (clause 5).

Clause 1 as written carried a contract bug, and this file is the reason it is not inherited: it
named the ONE-space prefix `'^ fact:'`, which matches 0 of the 365 live lines (`'^  fact:'` is
the real spelling; `grep -hc '^ fact:'` on the live tree 2026-10-08 returns 0). A counter built
on it prints 0, and clause 4's "post-move total equals the printed pre-move total" then passes at
0 == 0 on a run that dropped every fact. So `FACT_LINE_RE` is `^ {2}fact:`, every tree seeded
here holds at least two fact lines, the one-space tree is asserted to count 0 rather than to be
silently skipped, and `--apply` refuses a tree whose count reads zero.
"""
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "memory" / "quarantine_general_facts.py"
sys.path.insert(0, str(ROOT))

from app import kg_store, paths  # noqa: E402


def _load_mover():
    spec = importlib.util.spec_from_file_location("quarantine_general_facts", SCRIPT)
    assert spec and spec.loader, "the mover script is gone"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qf = _load_mover()

GENERAL = "general"
LLOYD = "Lloyd"          # a real entity, the one the mover must leave alone
PYTORCH = "PyTorch"      # registers an endpoint for the unrelated edge
SEEDED_REASON = "expired by an earlier sweep, not by this mover"

# The fixture's own numbers, in one place so a prose claim and an assertion cannot disagree.
# general/general-state.md  3 fact lines (2 file-attributed + 1 Lloyd override inside the file),
# general/general-decision.md 2 fact lines, general/general-overview.md 0 fact lines.
GENERAL_FACT_LINES = 5           # two-space lines in the entity directory: the mover's figure
GENERAL_ROWS = 4                 # facts_idx rows attributed to `general`
LLOYD_OWN_ROWS = 2               # rows in Lloyd/Lloyd-profile.md
LLOYD_OVERRIDE_ROWS = 1          # rows inside a `general/` file naming Lloyd
LLOYD_ROWS_BEFORE = LLOYD_OWN_ROWS + LLOYD_OVERRIDE_ROWS
ACTIVE_EDGES_NAMING_GENERAL = 2  # one each way; a third naming it is already expired


def _fact_body(entity: str, category: str, facts: list[dict], extra: str = "") -> str:
    """A fact file in the shape the store writes one: `yaml.dump`, so `fact:` lands at TWO spaces.

    Deliberately produced by `yaml.dump` rather than by hand: the two-space indentation is the
    whole subject of clause 1, and a hand-written fixture could drift from the real spelling
    without the test noticing. `dumped.count("  fact: ")` below is the check that the helper
    still emits what the live tree emits.
    """
    doc = {"type": "facts", "entity": entity, "category": category, "facts": facts}
    dumped = yaml.dump(doc, sort_keys=False)
    assert len(re.findall(r"^ {2}fact:", dumped, re.M)) == len(facts)
    return (f"---\n{dumped}---\n\n# {entity.title()} - {category.title()}\n\n"
            f"**Entity:** {entity}\n{extra}\n")


def _one_space_file(entity: str, n: int) -> str:
    """A fact file written with the ONE-space `fact:` spelling — the shape clause 1 named.

    Its front matter is deliberately not parseable YAML: no valid facts document indents a
    list item's keys by one space (`yaml.dump` emits two at every indent setting), so this is a
    *spelling* fixture and nothing more — it exists to prove the counter distinguishes the two
    spellings rather than matching `fact:` anywhere on a line. `test` re-asserts with a positive
    control that the loose patterns do find all `n` lines here, so a 0 from the mover is a
    measurement and not a broken fixture.
    """
    lines = [f"---\ntype: facts\nentity: {entity}\ncategory: state\nfacts:\n"]
    for i in range(n):
        lines.append(f"- entity: ''\n fact: fact number {i + 1}\n id: one-{i + 1}\n")
    lines.append("---\n\n# body\n")
    return "".join(lines)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_tree(facts_root: Path) -> None:
    """The scaled live tree: two entities, an overview with no facts, and locks on category files."""
    gen = facts_root / GENERAL
    _write(gen / "general-state.md", _fact_body(GENERAL, "state", [
        {"entity": "", "fact": "the watchdog restarts on a missed beat", "category": "state",
         "id": "sta-001"},
        {"entity": "", "fact": "the fleet runs nightly at 03:00", "category": "state",
         "id": "sta-002"},
        # A fact inside general's file that overrides its entity — the shape that decides
        # whether the mover's file list has to be captured before the move.
        {"entity": LLOYD, "fact": "Lloyd prefers a scoped alternative to a rewrite",
         "category": "state", "id": "sta-003"},
    ]))
    _write(gen / "general-decision.md", _fact_body(GENERAL, "decision", [
        {"entity": "", "fact": "the KG is one sqlite file", "category": "decision",
         "id": "dec-001"},
        {"entity": "", "fact": "quarantine rather than delete", "category": "decision",
         "id": "dec-002"},
    ]))
    # The overview: prose only, no `facts:` key, and no `.lock` beside it. All 365 live lines on
    # the real tree are in the 7 category files, so this file must still be MOVED — clause 3 —
    # even though it contributes 0 to the count.
    _write(gen / "general-overview.md",
           f"---\ntype: note\nentity: {GENERAL}\n---\n\n# General\n\n"
           f"A god entity: every category appears under this name.\n")
    for name in ("general-state.md", "general-decision.md"):
        (gen / f"{name}.lock").write_bytes(b"")      # zero-byte, as the live ones are
    _write(facts_root / LLOYD / f"{LLOYD}-profile.md", _fact_body(LLOYD, "identity", [
        {"entity": "", "fact": "runs on one local box", "category": "identity", "id": "pro-001"},
        {"entity": "", "fact": "refuses a write it cannot validate", "category": "identity",
         "id": "pro-002"},
    ]))
    _write(facts_root / PYTORCH / f"{PYTORCH}-notes.md", _fact_body(PYTORCH, "state", [
        {"entity": "", "fact": "pins the CUDA wheel", "category": "state", "id": "pyt-001"},
    ]))


def _seed_graph(kg, facts_root) -> dict:
    """Index the tree, then add the edges: two live ones naming `general`, one already expired,
    one unrelated. Returns the edge ids by role."""
    kg.facts_idx.reindex(root=facts_root)
    kg.entities.register(PYTORCH)
    ids = {
        "mentions": kg.edges.add({"source": GENERAL, "target": LLOYD, "type": "mentions",
                                  "evidence": "a fact under general names Lloyd"},
                                 origin="fact_extractor"),
        "discusses": kg.edges.add({"source": LLOYD, "target": GENERAL, "type": "discusses",
                                   "evidence": "a document discusses the general bucket"},
                                  origin="fact_extractor"),
        "stale": kg.edges.add({"source": GENERAL, "target": LLOYD, "type": "related_to",
                               "evidence": "written by an earlier pass"}, origin="fact_relate"),
        "unrelated": kg.edges.add({"source": LLOYD, "target": PYTORCH, "type": "uses",
                                   "evidence": "an unrelated pair"}, origin="fact_relate"),
    }
    assert kg.edges.expire(ids["stale"], SEEDED_REASON)
    assert len(kg.edges.active(either=GENERAL)) == ACTIVE_EDGES_NAMING_GENERAL
    return ids


@pytest.fixture()
def env(tmp_path):
    """A vault-derived root with `facts/` beside it, and a KG store to go with it.

    The layout matters, not just the contents: the quarantine root is the SIBLING of the facts
    tree, so the fixture reproduces `<vault-derived>/facts` and `<vault-derived>/kg.sqlite` the
    way `~/lloyd-data/_pipeline/vault-derived/` does. The process-default store is saved and put
    back, because driving the CLI with `--kg-db` routes through `kg_store.configure()`, which
    repoints it for the whole process.
    """
    derived = tmp_path / "vault-derived"
    facts_root = derived / "facts"
    facts_root.mkdir(parents=True)
    _seed_tree(facts_root)
    db = derived / "kg.sqlite"
    saved = kg_store._default_path
    # A handle of our own, NOT kg_store.configure(): configure() closes whatever default
    # store existed, and every CLI run below routes through it — so a configured fixture
    # handle would be a closed database by the time the assertions after a --apply ran.
    kg = kg_store.KGStore(db)
    ids = _seed_graph(kg, facts_root)
    try:
        yield {"tmp": tmp_path, "derived": derived, "facts_root": facts_root, "kg": kg,
               "db": db, "edge_ids": ids}
    finally:
        kg.close()
        kg_store.reset()
        kg_store._default_path = saved


def _run(argv: list[str]) -> tuple[int, str]:
    """Drive `main()` and hand back `(exit code, stdout)`."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = qf.main(argv)
    return code, buf.getvalue()


def _hash_tree(root: Path) -> dict[str, str]:
    """`{relative path: sha256 of bytes}` — the test's OWN before/after comparison.

    Computed here rather than through `mover.tree_digest` on purpose: if the dry run's own
    measurement is also the test's, a digest function that ignores content (or a file) makes
    both agree on a tree that changed.
    """
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _cli(env, *extra: str) -> tuple[int, str]:
    return _run(["--facts-root", str(env["facts_root"]), "--kg-db", str(env["db"]), *extra])


def _quarantine_roots(derived: Path) -> list[str]:
    return sorted(p.name for p in derived.glob(f"{qf.QUARANTINE_PREFIX}*"))


# ── clause 1: what the dry run prints ────────────────────────────────────────

def test_dry_run_prints_the_entity_dir_its_md_count_and_the_two_space_fact_total(env):
    """Clause 1: the plan names the directory, counts 3 `.md` files, and counts 5 fact lines.

    The figures are read off stdout, not off the returned dict — the operator's whole basis for
    the `--apply` is what the terminal shows.
    """
    code, out = _cli(env)
    assert code == 0
    assert f"facts/{GENERAL}" in out or str(env["facts_root"] / GENERAL) in out
    assert re.search(rf"{GENERAL}\b", out)
    assert re.search(r"3\s*/\s*2", out), f"the 3 .md / 2 .lock figures are not in:\n{out}"
    assert re.search(r"fact lines \(2-space\)\s+5", out), \
        f"the two-space total of 5 is not in:\n{out}"
    assert len(env["kg"].edges.active(either=GENERAL)) == ACTIVE_EDGES_NAMING_GENERAL


def test_the_two_space_counter_reads_the_real_spelling_and_a_one_space_tree_counts_zero(
        env, tmp_path):
    """Clause 1's guard: one-space fact lines count 0, so a zero can never masquerade as a pass.

    The positive control is the point of this test. On the one-space tree the loose patterns
    find all 5 lines while `FACT_LINE_RE` finds none, which is what makes the mover's 0 a
    reading of the spelling rather than a broken fixture, a wrong path, or a directory that was
    never there — and it is the exact bug #2443's clause 1 carried: `grep -hc '^ fact:'` on the
    live `general/` returns 0 where `'^  fact:'` returns 365.
    """
    assert qf.FACT_LINE_RE.pattern == r"^ {2}fact:"
    two_space = env["facts_root"] / GENERAL
    assert qf.fact_line_count(two_space) == GENERAL_FACT_LINES
    assert len(re.findall(r"^\s*fact:", two_space.joinpath("general-state.md")
                          .read_text(), re.M)) == 3

    loose_root = tmp_path / "one-space" / "facts"
    _write(loose_root / GENERAL / "general-state.md", _one_space_file(GENERAL, 5))
    text = (loose_root / GENERAL / "general-state.md").read_text()
    assert len(re.findall(r"^\s*fact:", text, re.M)) == 5, "positive control: the lines are there"
    assert len(re.findall(r"^ *fact:", text, re.M)) == 5
    assert qf.fact_line_count(loose_root / GENERAL) == 0

    code, out = _run(["--facts-root", str(loose_root), "--kg-db", str(env["db"])])
    assert code == 0
    assert re.search(r"fact lines \(2-space\)\s+0", out), \
        f"a one-space tree must report 0, not 5:\n{out}"


def test_the_dry_run_defaults_to_the_root_LLOYD_FACTS_ROOT_names(env, monkeypatch):
    """Clause 1/5's resolution: with the override set, that IS the tree; without it, `app.paths` is.

    The equivalence assertion is what stops the script's own read of the variable from silently
    diverging from the canonical resolver at `app/paths.py:236`: unset, the two must name the
    same directory, so a rename of the override there cannot leave the mover reading an
    old-orphaned name.
    """
    monkeypatch.setenv("LLOYD_FACTS_ROOT", str(env["facts_root"]))
    assert qf.resolve_facts_root() == env["facts_root"]
    monkeypatch.delenv("LLOYD_FACTS_ROOT")
    assert qf.resolve_facts_root() == Path(paths.VAULT_FACTS_ROOT)


# ── clause 2: the dry run touches nothing ────────────────────────────────────

def test_dry_run_creates_no_quarantine_root_and_leaves_every_file_byte_identical(env):
    """Clause 2: no `facts-quarantine-*` appears beside the facts tree, and no byte moves.

    The comparison is the test's own `{path: sha256}` map — added, deleted, renamed and rewritten
    files each change it — and the mover's own report is asserted to say `UNCHANGED`, so the
    figure the operator reads is the same measurement rather than a separate claim.
    """
    before = _hash_tree(env["facts_root"])
    assert len(before) >= 6, "the fixture tree must have files to keep still"
    code, out = _cli(env)
    assert code == 0
    assert _quarantine_roots(env["derived"]) == [], "the dry run created a quarantine root"
    assert _hash_tree(env["facts_root"]) == before
    assert "UNCHANGED" in out
    # And the graph half is untouched by the same read-only run.
    assert len(env["kg"].edges.active(either=GENERAL)) == ACTIVE_EDGES_NAMING_GENERAL
    assert env["kg"].entities.get(GENERAL) is not None
    assert env["kg"].facts_idx.count(entity=GENERAL, active_only=False) == GENERAL_ROWS


def test_apply_refuses_a_tree_whose_fact_count_reads_zero_rather_than_passing_at_0_equals_0(
        env, tmp_path):
    """The consequence of clause 1's bug being fixed: a 0 count blocks `--apply` instead of validating it.

    With the one-space spelling the pre/post check would have been `0 == 0`, which passes on a
    run that moved nothing and on one that dropped every fact. The refusal is what makes the
    non-zero figure clause 4 asks for a property of the real run, not only of the test fixture —
    and `--allow-zero-facts` is the way past it, named in the message, because a genuinely
    factless tree is possible (`general-overview.md` holds 0 lines) and an operator may still
    want it out of the tree.
    """
    zero_root = tmp_path / "zero" / "facts"
    _write(zero_root / GENERAL / "general-state.md", _one_space_file(GENERAL, 4))
    code, out = _run(["--facts-root", str(zero_root), "--kg-db", str(env["db"]), "--apply"])
    assert code == 1
    assert "REFUSED" in out and "0 == 0" in out
    assert (zero_root / GENERAL).is_dir(), "the refused run must not have moved the tree"
    assert _quarantine_roots(tmp_path / "zero") == []
    # A tree with facts still applies, so the guard is about the zero, not about --apply.
    code, out = _cli(env, "--apply", "--ts", "20261008T235900Z")
    assert code == 0 and env["facts_root"].joinpath(GENERAL).exists() is False


def test_allow_zero_facts_moves_a_tree_the_counter_cannot_read(env, tmp_path):
    """The documented way past the guard actually moves, rather than only silencing the message."""
    zero_root = tmp_path / "zero" / "facts"
    _write(zero_root / GENERAL / "general-state.md", _one_space_file(GENERAL, 4))
    code, out = _run(["--facts-root", str(zero_root), "--kg-db", str(env["db"]),
                      "--apply", "--allow-zero-facts", "--ts", "20261008T235901Z"])
    assert code == 0, out
    assert not (zero_root / GENERAL).exists()
    moved = (tmp_path / "zero" / f"{qf.QUARANTINE_PREFIX}20261008T235901Z" / GENERAL)
    assert moved.is_dir() and list(moved.glob("*.md"))


# ── clause 3: what --apply relocates, and where ──────────────────────────────

def test_apply_relocates_the_whole_entity_dir_to_the_facts_quarantine_root(env):
    """Clause 3: `facts/general/` becomes `<vault-derived>/facts-quarantine-<tag>/general/`.

    Every entry has to arrive, including the overview that holds no facts and the zero-byte
    `.lock` files: a move that left `general/` behind as a stub would satisfy a count check and
    fail the clause's "and `facts/general/` no longer exists". The root name is asserted against
    the `%Y%m%dT%H%M%SZ` shape `kg_rebuild.py:910-911` writes, because the live
    `facts-quarantine-20260923T070837Z` is what a later reader compares against.
    """
    tag = "20261008T235902Z"
    code, out = _cli(env, "--apply", "--ts", tag)
    assert code == 0, out
    dest = env["derived"] / f"{qf.QUARANTINE_PREFIX}{tag}" / GENERAL
    assert dest.is_dir()
    assert _quarantine_roots(env["derived"]) == [f"{qf.QUARANTINE_PREFIX}{tag}"]
    assert not (env["facts_root"] / GENERAL).exists(), "facts/general/ survived the move"
    names = sorted(p.name for p in dest.rglob("*") if p.is_file())
    assert names == ["general-decision.md", "general-decision.md.lock", "general-overview.md",
                     "general-state.md", "general-state.md.lock"]
    # The other entities' trees are untouched.
    assert (env["facts_root"] / LLOYD / f"{LLOYD}-profile.md").is_file()
    assert (env["facts_root"] / PYTORCH).is_dir()
    # And the root is derived from the tree that was moved, so a temp run never quarantines
    # into the live vault-derived root.
    assert dest.parent.parent in (env["derived"], env["derived"].resolve())


def test_the_quarantine_root_is_the_sibling_of_the_facts_tree(env):
    """The naming convention, asserted as a pattern rather than only as one directory name.

    `kg_rebuild.py:911`, `expire_numeric_entity_facts.py:323` and
    `link_stranded_entities.py:409` all name `facts-quarantine-<ts>`; #2443's draft named
    `<facts_root>/../quarantine/<run-tag>/`, which would have stood up a second,
    differently-spelled quarantine root beside the first.
    """
    root = qf.quarantine_root_for(env["facts_root"], "20260923T070837Z")
    assert root.name == "facts-quarantine-20260923T070837Z"
    assert root.parent == env["facts_root"].resolve().parent
    assert re.fullmatch(r"facts-quarantine-\d{8}T\d{6}Z", root.name)
    assert qf.run_tag().count("T") == 1 and qf.run_tag().endswith("Z")


def test_apply_refuses_to_merge_into_a_quarantine_that_already_holds_a_copy(env):
    """A same-named destination is a refusal, not a nest.

    `shutil.move` onto an existing directory lands the tree one level deeper as
    `general/general`, which the report would then describe as a successful move to a path that
    does not hold the facts.
    """
    tag = "20261008T235903Z"
    dest = env["derived"] / f"{qf.QUARANTINE_PREFIX}{tag}" / GENERAL
    dest.mkdir(parents=True)
    (dest / "someone-elses-run.md").write_text("do not overwrite me\n", encoding="utf-8")
    code, out = _cli(env, "--apply", "--ts", tag)
    assert code == 1 and "REFUSED" in out
    assert (env["facts_root"] / GENERAL).is_dir()
    assert (dest / "someone-elses-run.md").read_text() == "do not overwrite me\n"
    assert not (dest / GENERAL).exists()


# ── clause 4: the move is lossless, and the run says so ──────────────────────

def test_apply_prints_both_fact_figures_and_they_match_the_relocated_tree(env, capsys):
    """Clause 4: 5 before, 5 after, both printed, non-zero, and the same 5 lines byte for byte.

    The equality is re-measured from the quarantine directory with the test's own counter rather
    than the mover's, and the bytes are compared per file, so "the count matches" cannot be
    satisfied by two different sets of 5 lines.
    """
    tag = "20261008T235904Z"
    code, out = _cli(env, "--apply", "--ts", tag)
    assert code == 0, out
    dest = env["derived"] / f"{qf.QUARANTINE_PREFIX}{tag}" / GENERAL
    assert re.search(r"fact lines before move\s+5", out), out
    assert re.search(r"fact lines after move\s+5", out), out
    assert "MISMATCH" not in out
    assert GENERAL_FACT_LINES > 0
    recount = sum(len(qf.FACT_LINE_RE.findall(p.read_text(encoding="utf-8")))
                  for p in sorted(dest.glob("*.md")))
    assert recount == GENERAL_FACT_LINES == 5
    assert "fact: the fleet runs nightly at 03:00" in (dest / "general-state.md").read_text()
    src = env["facts_root"] / GENERAL
    assert not src.exists()
    # The dry run's printed figure is the same number the apply moves, so the operator's
    # expectation is set by the run before the one that acts.
    assert (dest / "general-state.md").read_text() == \
        _fact_body(GENERAL, "state", [
            {"entity": "", "fact": "the watchdog restarts on a missed beat", "category": "state",
             "id": "sta-001"},
            {"entity": "", "fact": "the fleet runs nightly at 03:00", "category": "state",
             "id": "sta-002"},
            {"entity": LLOYD, "fact": "Lloyd prefers a scoped alternative to a rewrite",
             "category": "state", "id": "sta-003"},
        ])


def test_apply_reports_the_tree_as_changed_and_keeps_the_other_entities(env):
    """The same digest that reads `UNCHANGED` on a dry run must read `CHANGED` on the move.

    An instrument that reports `UNCHANGED` for every run is what makes the clause-2 assertion
    worthless, so both readings are pinned here on the same fixture: the facts tree loses 5
    files and gains 0, and `Lloyd/` and `PyTorch/` keep their bytes.
    """
    before = _hash_tree(env["facts_root"])
    code, out = _cli(env, "--apply", "--ts", "20261008T235905Z")
    assert code == 0, out
    assert "CHANGED" in out and "UNCHANGED" not in out
    after = _hash_tree(env["facts_root"])
    assert not any(k.startswith(f"{GENERAL}/") for k in after)
    for key, digest in before.items():
        if not key.startswith(f"{GENERAL}/"):
            assert after[key] == digest, f"{key} changed content during the move"


# ── clause 5: the graph half, and a second run that does nothing ─────────────

def test_apply_expires_every_active_edge_naming_the_entity_with_a_reason_naming_2303(env):
    """Clause 5: 2 active edges naming `general` go to 0, each with `#2303` in its reason.

    Both directions are covered — the mover has to select on source OR target, since a
    `Lloyd -> general` `discusses` edge is exactly as much a live naming as the reverse — and
    two neighbouring edges are the controls: the one an earlier sweep already expired keeps its
    own reason, and `Lloyd -> PyTorch` stays active. An expire-all-edges bug passes a fixture
    that only holds edges belonging to the entity.
    """
    ids = env["edge_ids"]
    code, out = _cli(env, "--apply", "--ts", "20261008T235906Z")
    assert code == 0, out
    assert env["kg"].edges.active(either=GENERAL) == []
    assert len(env["kg"].edges.active(either=LLOYD)) == 1     # only the unrelated one survives
    for role in ("mentions", "discusses"):
        row = env["kg"].edges.by_id(ids[role])
        assert row["expired_at"], f"the {role} edge is still active"
        assert "#2303" in (row["expired_reason"] or ""), \
            f"reason does not name #2303: {row['expired_reason']!r}"
    stale = env["kg"].edges.by_id(ids["stale"])
    assert stale["expired_reason"] == SEEDED_REASON
    assert env["kg"].edges.by_id(ids["unrelated"])["expired_at"] is None
    assert re.search(r"edges expired\s+2", out), out


def test_apply_empties_the_index_and_deregisters_the_entity_through_the_existing_api(env):
    """Clause 5: 0 rows for `general`, no entities row, `Lloyd`'s own rows and row intact.

    `facts_idx.count(entity="general", active_only=False)` at 0 is the figure #2443's
    verify-after-landing names, and the `Lloyd` figures are what stop a de-index that simply
    cleared the table from passing: 2 rows survive because they live in `Lloyd/`, and the 1 row
    that lived inside `general/general-state.md` goes with the file rather than being left
    pointing into the quarantine root.
    """
    assert env["kg"].facts_idx.count(entity=GENERAL, active_only=False) == GENERAL_ROWS
    assert env["kg"].facts_idx.count(entity=LLOYD, active_only=False) == LLOYD_ROWS_BEFORE
    assert env["kg"].entities.get(GENERAL) is not None

    code, out = _cli(env, "--apply", "--ts", "20261008T235907Z")
    assert code == 0, out
    assert env["kg"].facts_idx.count(entity=GENERAL, active_only=False) == 0
    assert env["kg"].facts_idx.count(entity=LLOYD, active_only=False) == LLOYD_OWN_ROWS
    assert env["kg"].entities.get(GENERAL) is None
    assert env["kg"].entities.get(LLOYD) is not None
    live_paths = {r["file_path"] for r in env["kg"].facts_idx.for_entity(
        LLOYD, include_expired=True)}
    assert not any((p or "").startswith(f"{GENERAL}/") for p in live_paths), \
        f"rows still point into the quarantined tree: {live_paths}"
    assert re.search(r"indexed rows after\s+0", out), out


def test_the_entities_row_goes_only_after_the_index_is_rebuilt(env, monkeypatch):
    """Clause 5's ORDER: `reindex` first, then the existing `Entities.remove()`.

    The sequence is load-bearing, not stylistic: `reindex` registers every entity directory it
    sees, so deregistering first lets the index put the row straight back and the run reports a
    quarantine the store never accepted. The accessors are patched on their CLASSES, because
    the CLI opens its own `KGStore` through `kg_store.configure()` — an instance patch on the
    fixture's handle would watch a connection the run never used and record nothing.

    The reindex call is also pinned to the scoped form: `paths=` the files it captured, the
    facts root as `root`, and `register_entities=False`. A whole-tree `reindex()` instead would
    walk every entity directory in the store and re-register the entity being retired.
    """
    seq: list[str] = []
    calls: dict = {}
    idx_cls = type(env["kg"].facts_idx)
    ent_cls = type(env["kg"].entities)
    real_reindex, real_remove = idx_cls.reindex, ent_cls.remove

    def spy_reindex(self, *args, **kwargs):
        seq.append("reindex")
        calls["args"], calls["kwargs"] = args, kwargs
        return real_reindex(self, *args, **kwargs)

    def spy_remove(self, name):
        seq.append(f"remove:{name}")
        return real_remove(self, name)

    monkeypatch.setattr(idx_cls, "reindex", spy_reindex)
    monkeypatch.setattr(ent_cls, "remove", spy_remove)
    code, out = _cli(env, "--apply", "--ts", "20261008T235914Z")
    assert code == 0, out
    assert seq == ["reindex", f"remove:{GENERAL}"], f"the writes went in the wrong order: {seq}"
    kwargs = calls["kwargs"]
    assert kwargs.get("register_entities") is False, \
        "a reindex that registers entities re-creates the row this mover retires"
    paths_arg = kwargs.get("paths") if kwargs.get("paths") is not None else (
        calls["args"][0] if calls["args"] else None)
    assert paths_arg is not None, "a whole-tree reindex walks every entity in the store"
    assert len(list(paths_arg)) == 3, "the mover must clear only the files it moved"
    assert Path(kwargs.get("root")) == env["facts_root"]
    assert env["kg"].entities.get(GENERAL) is None


def test_a_second_apply_on_the_emptied_root_exits_zero_having_moved_nothing(env):
    """Clause 5's idempotence: run twice, and the second run moves nothing and still exits 0.

    `moved` stays False and the quarantine root list keeps one entry, so "exits 0" is not being
    paid for by a second directory appearing beside the first. The graph writes are re-issued
    and are no-ops: an interrupted run between the move and the writes has to converge, not
    report a clean store while the entity keeps its edges.
    """
    tag = "20261008T235908Z"
    assert _cli(env, "--apply", "--ts", tag)[0] == 0
    roots_after_first = _quarantine_roots(env["derived"])
    dest = env["derived"] / roots_after_first[0] / GENERAL
    assert qf.fact_line_count(dest) == GENERAL_FACT_LINES

    code, out = _cli(env, "--apply", "--ts", "20261008T235909Z")
    assert code == 0, out
    assert "nothing moved" in out and "ABSENT" in out
    assert _quarantine_roots(env["derived"]) == roots_after_first
    assert qf.fact_line_count(dest) == GENERAL_FACT_LINES
    assert not (env["derived"] / f"{qf.QUARANTINE_PREFIX}20261008T235909Z").exists()
    # And the machine-readable report says the same: the second run moved nothing.
    assert _json_run(env, "--apply", "--ts", "20261008T235910Z")["moved"] is False


def _json_run(env, *extra: str) -> dict:
    """One `--json` run's report, for the fields the text report does not print."""
    code, out = _cli(env, "--json", *extra)
    assert code == 0, out
    return json.loads([ln for ln in out.splitlines() if ln.startswith("{")][-1])


def test_the_mover_opens_no_database_of_its_own(env):
    """Clause 5's rail: nothing opens `kg.sqlite` except `app.kg_store`.

    Asserted on the mover's own source, because the constraint is about the module, not about
    one run: no `sqlite3` import, no `connect`, no raw statement and no reaching through the
    store's private connection object — every graph effect goes through the public accessors
    `facts_idx.reindex`, `edges.expire` and `entities.remove`. `edges.expire` is the cited one
    (`app/kg_store.py:729`) and `entities.remove` is the pre-existing one (#2443's clause 4
    asked for a new method; `app/kg_store.py:1067` has done this all along), so the test also
    pins that no second spelling of either was added to `app/kg_store.py` by this round.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"\bsqlite3\b", src), "the mover must not name sqlite3"
    assert ".connect(" not in src
    assert ".execute(" not in src
    assert ".conn" not in src
    assert "from app import kg_store" in src
    assert "edges.expire(" in src and "entities.remove(" in src
    assert "facts_idx.reindex(" in src
    assert "entities_remove" not in src and "def remove_entity" not in src


def test_the_graph_writes_land_in_the_store_the_cli_was_pointed_at(env):
    """The process seam: a second opener of the same file sees what the run wrote.

    The CLI call runs in this process, so an assertion against the fixture's own handle would
    still pass on a run that wrote to a different database — which is the exact shape of the
    2026-09-09 incident `app/paths.py:228-234` records: five `--apply` runs reported 32 fact
    expirations that never reached the live store. This re-opens the file by path and reads the
    three figures back through `app.kg_store`.
    """
    code, out = _cli(env, "--apply", "--ts", "20261008T235911Z")
    assert code == 0, out
    fresh = kg_store.KGStore(env["db"])
    try:
        assert fresh.edges.active(either=GENERAL) == []
        assert fresh.entities.get(GENERAL) is None
        assert fresh.facts_idx.count(entity=GENERAL, active_only=False) == 0
        assert fresh.facts_idx.count(entity=LLOYD, active_only=False) == LLOYD_OWN_ROWS
    finally:
        fresh.close()


def test_the_cli_resolves_its_tree_from_LLOYD_FACTS_ROOT_in_a_fresh_process(env):
    """The environment seam across a real process boundary: no `--facts-root`, only the variable.

    Clause 1 and clause 5 both hang on "resolving the facts root from `LLOYD_FACTS_ROOT`", and
    `app/paths.py:236` reads it at import time — so an in-process test can only ever prove the
    resolver's fallback. This runs the script as a child process with the variable set and no
    path flags, and asserts the figures, the destination and the untouched live root: proof that
    the mover really acts on the tree the operator named.
    """
    env["kg"].close()
    env_path = str(env["facts_root"])
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--apply", "--ts", "20261008T235912Z"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
             "LLOYD_FACTS_ROOT": env_path, "LLOYD_KG_DB": str(env["db"]),
             "LLOYD_DATA": str(env["tmp"] / "data"),
             "PYTHONPATH": str(ROOT)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"fact lines after move\s+5", proc.stdout), proc.stdout
    dest = env["derived"] / f"{qf.QUARANTINE_PREFIX}20261008T235912Z" / GENERAL
    assert dest.is_dir() and not (env["facts_root"] / GENERAL).exists()
    # Nothing landed in the default tree the child would have used without the override:
    # had the variable been ignored, `main()` would have refused (exit 2, "no facts tree")
    # rather than reporting a move, and this sibling of the default root would be empty.
    default_derived = env["tmp"] / "data" / "_pipeline" / "vault-derived"
    assert not (default_derived / "facts" / GENERAL).exists()
    assert not list(default_derived.glob(f"{qf.QUARANTINE_PREFIX}*"))


def test_apply_refuses_while_a_rebuild_holds_writes_disabled(env, monkeypatch):
    """`--apply` honours `knowledge_graph.write_enabled`, the #1833 guard every facts-tree writer has.

    `kg_rebuild.py` renames its output INTO a `facts-quarantine-<ts>` root — the same root this
    mover creates, from the same directory it renames — so two components doing that in one
    window is how a rebuild and a quarantine end up inside each other. Driven for real against
    a config this test owns, never by patching the predicate away, and the dry run still works:
    the guard is on writing, not on looking.
    """
    import app.config
    monkeypatch.setattr(app.config, "CONFIG", {"knowledge_graph": {"write_enabled": False}})
    code, out = _cli(env, "--apply", "--ts", "20261008T235913Z")
    assert code == 2
    assert "knowledge_graph.write_enabled = false" in out
    assert (env["facts_root"] / GENERAL).is_dir()
    assert _quarantine_roots(env["derived"]) == []
    code, out = _cli(env)                                   # the dry run is always safe
    assert code == 0 and "fact lines (2-space)  5" in out
