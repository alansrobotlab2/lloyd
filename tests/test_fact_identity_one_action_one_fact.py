"""#874 — one action addresses exactly one fact, and the loop reports only
signals and numbers it can actually see.

Every test here pins one acceptance clause of backlog #874. The defect class is
"a check that reports a verdict it cannot justify", and the fact store had two
of them at once: a writer that condemned a fact by `id` while ids are a
per-file counter (`app/fact_ids.next_fact_id` numbers within one
`<entity>-<category>.md`, so `fact-001` is one fact in a file and an alias for
several across an entity — 104 of the first 400 live entity dirs have an id
shared by more than one active fact, measured 2026-09-19), and a corrections
reader that read a file last written in May while the live log went unparsed.

The cost on record before this change: 2 planned actions invalidated 25 facts on
the live `Assistant` entity (29 active → 4), and `fact_entity_recall` moved
0.35 → 0.30 because of it, which `architecture/memory.md` cited as evidence of
quality movement. It was blast radius.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import threading
import yaml
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_mcp._shared as shared                     # noqa: E402
from agent_mcp import facts as facts_mod               # noqa: E402
from agent_mcp import retrieval                        # noqa: E402
from agent_mcp import fact_improvement as fi           # noqa: E402
from app import kg_store                               # noqa: E402


def _iso(days_ago: int) -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=days_ago)).isoformat()


def _write(root: Path, entity: str, category: str, facts: list[dict]) -> Path:
    """One fact file. Ids are taken verbatim, because the collision is the point."""
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    prepared = [{"fact": f["fact"], "confidence": f.get("confidence", 0.9),
                 "category": category, "id": f["id"], "created_at": f["created_at"],
                 "valid_at": f["created_at"], "expired_at": None, "invalid_at": None,
                 "provenance": "STATED", "source_doc": None} for f in facts]
    fm = {"type": "facts", "entity": entity, "category": category, "facts": prepared}
    path = d / f"{entity}-{category}.md"
    path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n",
                    encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fact tree + store + vault root in tmp, with every reader repointed."""
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    vault_root = tmp_path / "vault"
    (vault_root / "memory").mkdir(parents=True)
    (vault_root / "lloyd").mkdir()

    monkeypatch.setattr(shared, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(facts_mod, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(fi, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(fi, "CORRECTIONS_PATH", vault_root / "memory" / "corrections.md")
    monkeypatch.setattr(fi, "CORRECTIONS_BULLETS_PATH", vault_root / "lloyd" / "USER.md")
    monkeypatch.setattr(fi, "CORRECTIONS_SOURCES", None)
    monkeypatch.setattr(fi, "RECORD_DIR", tmp_path / "records")
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None

    st = kg_store.configure(tmp_path / "kg.sqlite")
    yield facts_root, st, vault_root
    kg_store.reset()


def _reindex(st, root):
    st.facts_idx.reindex(root=root)


def _marked(root: Path) -> list[tuple[str, str, str]]:
    """(file, id, field) for every marked fact on disk — read back, not returned."""
    out = []
    for path in sorted(root.rglob("*.md")):
        fm = yaml.safe_load(path.read_text()[3:path.read_text().find("\n---", 3)]) or {}
        for f in fm.get("facts") or []:
            for field in ("invalid_at", "expired_at"):
                if f.get(field):
                    out.append((path.name, f.get("id"), field))
    return out


# ── the two-file fixture every id-collision clause is graded against ─────────
# A fixture that puts the SAME two texts in both category files yields two
# equal-confidence pairs, which `_fact_resolve_apply` skips — it marks 0 and
# proves nothing. This one puts the loser's id on a fact the other file uses for its
# winner, which is what the live corpus looks like: state/fact-001 is condemned
# while usage/fact-001 must survive.

def _colliding_entity(root: Path) -> Path:
    d = _write(root, "Idcol", "state", [
        {"fact": "The indexer is disabled.", "confidence": 0.3, "id": "fact-001",
         "created_at": _iso(40)},
        {"fact": "The indexer is enabled.", "confidence": 0.9, "id": "fact-002",
         "created_at": _iso(3)},
    ]).parent
    _write(root, "Idcol", "usage", [
        {"fact": "The daemon is active.", "confidence": 0.9, "id": "fact-001",
         "created_at": _iso(40)},
        {"fact": "The daemon is inactive.", "confidence": 0.3, "id": "fact-002",
         "created_at": _iso(3)},
    ])
    return d


# ── clause 1: the non-condemned twin keeps its null marks ────────────────────

def test_fact_resolve_apply_leaves_the_noncondemned_twin_unmarked(tree):
    facts_root, st, _vault = tree
    _colliding_entity(facts_root)
    out = facts_mod._fact_resolve_apply({"entity": "Idcol"})
    marked = _marked(facts_root)
    assert ("Idcol-state.md", "fact-001", "invalid_at") in marked, marked
    assert ("Idcol-usage.md", "fact-001", "invalid_at") not in marked, \
        "usage/fact-001 shares state/fact-001's id but was never condemned"
    assert out["resolved"] == len(marked)


# ── clause 2: exactly 2 facts marked, and `resolved` equals that ─────────────

def test_fact_resolve_apply_marks_exactly_two_and_reports_what_it_marked(tree):
    facts_root, st, _vault = tree
    _colliding_entity(facts_root)
    out = facts_mod._fact_resolve_apply({"entity": "Idcol"})
    marked = _marked(facts_root)
    assert len(marked) == 2, marked
    assert out["resolved"] == 2 == len(marked)
    # The returned list names the file of each marked fact: a caller cannot
    # audit a count that does not say where the writes landed.
    assert sorted((m["file"], m["id"]) for m in out["facts"]) == [
        ("Idcol/Idcol-state.md", "fact-001"), ("Idcol/Idcol-usage.md", "fact-002")]


# ── clause 3/4: the improve loop's own writes ────────────────────────────────

def test_confidence_class_records_its_loser_with_invalid_at_not_expired_at(tree):
    facts_root, st, _vault = tree
    _colliding_entity(facts_root)
    _reindex(st, facts_root)
    plan = fi.plan_entity("Idcol")
    assert plan["actions"], "fixture must produce an actionable pair"
    action = plan["actions"][0]
    assert action["kind"] == "confidence"
    result = fi.apply_action(action, _iso(0))
    assert result["expired_count"] == 1, result
    marked = _marked(facts_root)
    assert len(marked) == 1, marked
    assert marked[0][2] == "invalid_at", \
        "a claim that should not have been recorded is invalid, not expired " \
        "(agent_mcp/facts.py `_fact_resolve_apply` is the semantics)"
    assert not any(f == "expired_at" for _, _, f in marked)


def test_no_single_action_reports_more_than_one_fact_on_a_colliding_entity(tree):
    facts_root, st, _vault = tree
    _colliding_entity(facts_root)
    _reindex(st, facts_root)
    rec = fi.run_improvement(apply=True, entities=["Idcol"])
    actions = [a for e in rec["per_entity"] for a in e["actions"]]
    assert actions, rec["per_entity"]
    for action in actions:
        assert action["changed"] <= 1, \
            f"one action condemned {action['changed']} facts: {action}"
        assert not action.get("error"), action
    # Every action that ran condemned exactly one fact, and the run's total is
    # the sum of those ones — the figure that used to be 25 for 2 actions.
    assert rec["actions_taken"] == len(actions) == len(_marked(facts_root))


def test_marking_holds_the_file_lock_across_its_whole_read_modify_write(tree,
                                                                        monkeypatch):
    """Scope was the bug; a lost update was the way it could come back.

    `_fact_add` and `_fact_invalidate` each wrap their read-modify-write in
    `locked_file`, because the extractor runs four worker threads and a chat turn
    can fire `fact_add` at the same moment — the later writer drops whatever the
    earlier one added (`app/atomic_io.py:247`). The shared marking route this
    change introduced read outside any lock and wrote with a bare
    `atomic_write_text`, so a fact added while a resolve was in flight vanished
    from the file: one fact per action, and the neighbour it lost was a neighbour
    of a different kind.

    The interleaving is forced, not raced, so both outcomes are deterministic:
    the marker's write is gated on an event, and a concurrent writer of exactly
    `_fact_add`'s shape is started while the marker is between its read and its
    write. Holding the lock makes that writer wait; not holding it lets the
    writer land its addition a moment before the marker overwrites the file with
    its own stale snapshot.
    """
    from app.atomic_io import atomic_write_text as real_write, locked_file

    facts_root, st, _vault = tree
    path = _write(facts_root, "Locky", "state", [
        {"fact": "The queue is disabled.", "confidence": 0.3, "id": "fact-001",
         "created_at": _iso(40)},
        {"fact": "The queue is enabled.", "confidence": 0.9, "id": "fact-002",
         "created_at": _iso(3)},
    ])
    _reindex(st, facts_root)

    in_write = threading.Event()
    proceed = threading.Event()

    def gated_write(target, text):
        in_write.set()
        try:
            assert proceed.wait(timeout=15), "the marker's write never resumed"
        finally:
            real_write(target, text)

    monkeypatch.setattr(facts_mod, "atomic_write_text", gated_write)

    holder_added = threading.Event()

    def concurrent_adder():
        with locked_file(path):
            raw = path.read_text(encoding="utf-8")
            fm = facts_mod._parse_fact_frontmatter(raw)
            fm["facts"].append({
                "fact": "The queue drained 4 items.", "confidence": 0.9,
                "category": "state", "id": "fact-003",
                "created_at": _iso(0), "valid_at": _iso(0), "expired_at": None,
                "invalid_at": None, "provenance": "STATED", "source_doc": None})
            body = raw[raw.find("---", 3) + 3:]
            real_write(path, facts_mod._write_fact_frontmatter(fm) + body)
            holder_added.set()

    marker = threading.Thread(target=facts_mod._fact_resolve_apply,
                              args=({"entity": "Locky"},),
                              daemon=True)
    adder = threading.Thread(target=concurrent_adder, daemon=True)
    try:
        marker.start()
        assert in_write.wait(timeout=10), "the marker never reached its write"

        adder.start()
        adder.join(1.0)
        assert not holder_added.is_set(), (
            "a concurrent fact_add completed while the marker still had its own "
            "snapshot in hand — the marker is not holding locked_file across its "
            "read and its write, so this write will drop that fact")
    finally:
        proceed.set()
    marker.join(15)
    adder.join(15)

    fm = facts_mod._parse_fact_frontmatter(path.read_text(encoding="utf-8"))
    by_id = {f["id"]: f for f in fm["facts"]}
    assert by_id["fact-001"].get("invalid_at"), "the condemned fact was not marked"
    assert "fact-003" in by_id, \
        "the fact added during the resolve is gone: lost update"
    assert not by_id["fact-003"].get("invalid_at")
    assert not by_id["fact-003"].get("expired_at")


# ── clauses 5-8: the corrections log the loop can actually read ──────────────

def test_bullet_format_corrections_yield_a_signal(tree):
    """`## corrections_log` entries in USER.md are `- 2026-09-11: …` bullets.

    Pointing a heading parser at that file returns a silent zero — which is how
    the loop reported "no corrections" while the operator's own corrections sat
    unread in the file that actually holds them.
    """
    facts_root, st, vault_root = tree
    _write(facts_root, "TTS", "state",
           [{"fact": "TTS built-in voices return 500 errors.", "confidence": 0.9,
             "id": "stat-001", "created_at": _iso(4)}])
    _reindex(st, facts_root)
    (vault_root / "lloyd" / "USER.md").write_text(
        "# User\n\n## corrections_log\n"
        f"- **{_iso(1)[:10]}:** TTS built-in voices were returning 500s; fixed.\n"
        "\n## other_business\n- nothing entity-shaped\n", encoding="utf-8")
    found = fi.read_correction_signals()
    assert [s["entity"] for s in found] == ["TTS"], found
    assert found[0]["corrections_path"].endswith("lloyd/USER.md"), found[0]


def test_record_corrections_path_names_the_file_that_yielded_the_signals(tree):
    facts_root, st, vault_root = tree
    _write(facts_root, "TTS", "state",
           [{"fact": "TTS built-in voices return 500 errors.", "confidence": 0.9,
             "id": "stat-001", "created_at": _iso(4)}])
    _reindex(st, facts_root)
    (vault_root / "memory" / "corrections.md").write_text("# Corrections\n", encoding="utf-8")
    (vault_root / "lloyd" / "USER.md").write_text(
        "## corrections_log\n"
        f"- {_iso(2)[:10]}: TTS service status was wrong.\n", encoding="utf-8")
    rec = fi.run_improvement(sources=("corrections",))
    assert rec["signals"] == 1, rec["signals"]
    assert rec["corrections_path"].endswith("lloyd/USER.md"), rec["corrections_path"]
    assert "corrections.md" not in rec["corrections_path"]
    assert rec["corrections_status"] == "signals"


def test_zero_distinguishes_an_empty_log_from_an_unreadable_one(tree, monkeypatch):
    """Three zeros, three meanings, three records.

    `[]` from the corrections reader used to mean any of: no log, an empty log,
    a log whose format it cannot parse, a log whose entries are all too old, or
    a store that cannot say what an entity is. The last one is the trap: an
    unreadable registry writes "nothing to correct" over "I could not tell".
    """
    facts_root, _st, vault_root = tree
    head = vault_root / "memory" / "corrections.md"
    user = vault_root / "lloyd" / "USER.md"
    user.write_text("# nothing here\n", encoding="utf-8")
    monkeypatch.setattr(fi, "_known_entities", lambda: {"tts": "TTS"})

    head.write_text("# Corrections Log\n", encoding="utf-8")
    assert fi.read_correction_signals() == []
    assert fi.last_corrections_read()["status"] == "empty"

    head.write_text("### 2026-09-10 — TTS something\n", encoding="utf-8")
    monkeypatch.setattr(fi, "_known_entities", lambda: {})
    assert fi.read_correction_signals() == []
    assert fi.last_corrections_read()["status"] == "registry_unreadable", \
        fi.last_corrections_read()

    monkeypatch.setattr(fi, "_known_entities", lambda: {"tts": "TTS"})
    head.unlink()
    assert fi.read_correction_signals() == []
    assert fi.last_corrections_read()["status"] == "missing"


def test_a_heading_outside_the_window_contributes_no_entity(tree, monkeypatch):
    """Undated-by-window means no entity, but it is counted — not silence."""
    facts_root, _st, vault_root = tree
    monkeypatch.setattr(fi, "_known_entities", lambda: {"gateway": "Gateway"})
    head = vault_root / "memory" / "corrections.md"
    head.write_text(
        "# Corrections Log\n\n"
        "## 2026-03-31 20:04 PDT — Skipped skill check before gateway restart\n"
        "## 2026-03-30 — Thorough autonomy audit\n", encoding="utf-8")
    (vault_root / "lloyd" / "USER.md").write_text("## corrections_log\n", encoding="utf-8")
    assert fi.read_correction_signals() == []
    read = fi.last_corrections_read()
    per = read["sources"][str(head)]
    assert per["status"] == "no_entries_in_window", per
    assert per["entries"] == 2 and per["outside_window"] == 2, per
    assert read["status"] == "no_entries_in_window"


# ── clause 9: any skill naming an absent repo script fails, generically ──────

def test_a_skill_naming_an_absent_repo_script_is_caught(tmp_path):
    """The rule is generic: it reads a skill's text, not one hardcoded pair.

    #376's vault half shipped a day without its code half because the only check
    of this kind was `tests/test_memory_improvement.py`, written for the
    fact-improvement pair. Nothing checked *any* skill.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("skill_lint",
                                                 ROOT / "scripts" / "skill_lint.py")
    skill_lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill_lint)

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "here.py").write_text("#\n", encoding="utf-8")
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)

    hits = skill_lint.check_script_paths(
        "Run `~/lloyd/scripts/here.py` then `~/lloyd/scripts/gone.py`.",
        skill_dir=skill_dir, repo_root=repo)
    paths = {h["path"] for h in hits}
    assert paths == {"scripts/gone.py"}, hits
    assert hits[0]["known_stale"] == "", "a new absent path is a failure, not a footnote"

    # A skill-relative path resolves inside the skill, and prose examples that
    # are not anchored to the repo are not claims about this checkout.
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "own.py").write_text("#\n", encoding="utf-8")
    assert skill_lint.check_script_paths("see scripts/own.py", skill_dir=skill_dir,
                                        repo_root=repo) == []
    assert skill_lint.check_script_paths("write tests/test_login.py for it",
                                        skill_dir=skill_dir, repo_root=repo) == []


def test_the_lint_report_and_totals_line_carry_the_missing_script_count(tmp_path,
                                                                        monkeypatch,
                                                                        capsys):
    """A rule that cannot report itself is a rule that silently does nothing.

    The check landed wired into `lint()` and its JSON, but not into either of the
    two places a human reads: `render_report` never counted its own new key into
    the table or the body — a run whose ONLY defect was an absent script printed
    `## ✅ Clean / All skills pass lint` — and `main()`'s totals line referenced
    `n_scripts`, a local of `render_report`, so every invocation raised
    `NameError` on the way out. Both shapes are asserted here over one fabricated
    result, with the report path redirected into `tmp_path`: writing it for real
    would put a test's fake skill into the vault's lint report.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("skill_lint",
                                                 ROOT / "scripts" / "skill_lint.py")
    skill_lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill_lint)

    result = {
        "generated_at": "2026-09-19T04:00:00", "total": 1,
        "dead": [], "missing_desc": [], "drift": [], "duplicates": [],
        "stale": [], "phantom": [],
        "missing_script": [{"name": "demo-skill", "path": "/x/SKILL.md",
                            "scripts": [{"path": "scripts/demo.py",
                                         "known_stale": ""}]}],
    }

    report = skill_lint.render_report(result)
    assert "| MISSING_SCRIPT (names a repo script absent from the tree) | **1** |" in report
    assert "`demo-skill`" in report and "`scripts/demo.py`" in report
    assert "All skills pass lint" not in report, report[-400:]

    monkeypatch.setattr(skill_lint, "REPORT_PATH", tmp_path / "skill-lint-report.md")
    monkeypatch.setattr(skill_lint, "lint", lambda: result)
    assert skill_lint.main() == 0, "main() raised — the totals line cannot reach stdout"
    totals = capsys.readouterr().out
    assert "missing_script=1" in totals, totals


def _load_skill_lint():
    import importlib.util
    spec = importlib.util.spec_from_file_location("skill_lint",
                                                 ROOT / "scripts" / "skill_lint.py")
    skill_lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill_lint)
    return skill_lint


def _scan_corpus(skill_lint, skills_dir: Path):
    """(skills scanned, {skill: [absent paths not in the ledger]})."""
    offenders: dict[str, list[str]] = {}
    scanned = 0
    for entry in sorted(skills_dir.iterdir()):
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue
        scanned += 1
        for hit in skill_lint.check_script_paths(
                skill_file.read_text(encoding="utf-8", errors="replace"),
                skill_dir=entry, repo_root=ROOT):
            if not hit["known_stale"]:
                offenders.setdefault(entry.name, []).append(hit["path"])
    return scanned, offenders


def test_no_shipped_skill_names_an_absent_repo_script():
    """The rule runs over the real skill corpus, with named pre-existing drift.

    An entry in `KNOWN_ABSENT_SCRIPTS` is a dated debt the rule found the day it
    was written; a path NOT listed there fails the suite.

    A missing corpus is a failure, not a skip. The earlier revision skipped when
    `~/obsidian/skills` was absent, which let the corpus half of clause 9 pass
    silently on a host without the vault — a gate that reports nothing on a host
    where its subject should exist is the same defect this item is about. The
    denominator is asserted too: scanning zero skills, or one, is not a corpus
    (194 `SKILL.md` files live here as of 2026-09-19), so the floor is well below
    any real tree and above an empty one.
    """
    skill_lint = _load_skill_lint()
    skills_dir = Path.home() / "obsidian" / "skills"
    assert skills_dir.is_dir(), (
        f"{skills_dir} is absent — this rule's subject is the shipped skill "
        "corpus, and a scan that ran over nothing is not a pass")
    scanned, offenders = _scan_corpus(skill_lint, skills_dir)
    assert scanned >= 20, f"only {scanned} skills scanned; that is not a corpus"
    assert not offenders, offenders


def test_the_absent_script_ledger_only_carries_drift_still_cited():
    """A named allowlist inside an enforced rule needs a retirement route.

    `KNOWN_ABSENT_SCRIPTS` is four entries today. Every one must still be cited
    by a shipped skill, or the entry is a stale allowance: the day a skill drops
    its bad path, the line proving it stays behind and the rule quietly permits a
    *new* citation of that path forever. The failure message names the move —
    delete the entry.
    """
    skill_lint = _load_skill_lint()
    skills_dir = Path.home() / "obsidian" / "skills"
    assert skills_dir.is_dir(), f"{skills_dir} is absent"
    cited: set[str] = set()
    for entry in sorted(skills_dir.iterdir()):
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue
        for hit in skill_lint.check_script_paths(
                skill_file.read_text(encoding="utf-8", errors="replace"),
                skill_dir=entry, repo_root=ROOT):
            if hit["known_stale"]:
                cited.add(hit["path"])
    stale = sorted(set(skill_lint.KNOWN_ABSENT_SCRIPTS) - cited)
    assert not stale, (
        f"KNOWN_ABSENT_SCRIPTS entries no longer cited by any skill: {stale} — "
        "retire them, or the ledger is allowing paths nobody asked it to allow")


# ── clause 10: a before/after number its own mechanism can move ──────────────

def test_record_carries_a_pair_count_its_own_writes_can_move(tree):
    facts_root, st, _vault = tree
    _write(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "confidence": 0.9, "id": "stat-001", "created_at": _iso(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
         "confidence": 0.9, "id": "stat-002", "created_at": _iso(2)},
    ])
    _reindex(st, facts_root)
    rec = fi.run_improvement(apply=True, sources=("drift",), days=3)
    assert rec["actions_taken"] == 1, rec["per_entity"]
    # Nonzero before, zero after: retiring the superseded claim removes the
    # pair, which is the thing `fact_entity_recall` cannot show.
    assert rec["pairs_before"] == 1, rec["pairs_before"]
    assert rec["pairs_after"] == 0, rec["pairs_after"]
    written = sorted((fi.RECORD_DIR).glob("*.json"))
    assert written, "the pass must be recorded, not just returned"
    on_disk = json.loads(written[-1].read_text(encoding="utf-8"))
    assert on_disk["pairs_before"] == 1 and on_disk["pairs_after"] == 0
    for key in ("corrections_paths_read", "corrections_status"):
        assert key in on_disk, key


def test_a_near_duplicate_is_counted_once_and_stays_a_subset_of_the_pairs(tree):
    """`pairs_before` is the pair count, not the pair count plus its own subset.

    The field ships next to a comment calling it "the loop's own denominator",
    and beside `near_duplicates`, which it then also counted: an expression of
    `C + (C - actable)` charges a pair the loop declines to touch twice — once as
    a pair and once as a near-duplicate. The clause-10 test above keeps
    `near_duplicates` at zero, where the two expressions agree, so the inflation
    had no test and the record's absolute count (the half anyone reads) was wrong
    by one for every pair the loop chose to leave alone. This fixture has one of
    each, which is where the two formulas disagree.
    """
    facts_root, st, _vault = tree
    _write(facts_root, "Pairct", "state", [
        # An opposing-terms pair: the loop acts on this one.
        {"fact": "The indexer is disabled.", "confidence": 0.3, "id": "stat-001",
         "created_at": _iso(40)},
        {"fact": "The indexer is enabled.", "confidence": 0.9, "id": "stat-002",
         "created_at": _iso(3)},
    ])
    _write(facts_root, "Pairct", "usage", [
        # A 6-of-8-token Jaccard pair, 0.75 > 0.6, with no opposing term in
        # either text: the detector's near-duplicate class, which this loop
        # reports and refuses to delete.
        {"fact": "The vault sync runs at 03:00 daily.", "confidence": 0.9,
         "id": "usag-001", "created_at": _iso(5)},
        {"fact": "The vault sync runs at 04:00 daily.", "confidence": 0.9,
         "id": "usag-002", "created_at": _iso(2)},
    ])
    _reindex(st, facts_root)

    plan = fi.plan_entity("Pairct")
    assert plan["contradictions"] == 2, plan["contradictions"]
    assert plan["near_duplicates"] == 1, plan
    assert len(plan["actions"]) == 1, plan["actions"]
    # One pair, one count: the subset is not added to its own superset. The
    # inflated expression gives 3 here.
    assert plan["pairs_before"] == plan["contradictions"] == 2, plan["pairs_before"]
    assert plan["near_duplicates"] <= plan["pairs_before"]


# ── clause 11: over-reach reads as a regression, a correction does not ───────

# `scripts/memory/measure_overreach_snapshot.py` builds a fixture on a SNAPSHOT
# copy (`LLOYD_FACTS_ROOT`/`LLOYD_KG_DB` pointed at a temp tree, so no production
# fact is touched), runs both writes through the real store, and prints one JSON
# object. Running it here rather than reimplementing it means the clause is pinned
# against the real `run_eval` scorer over the real `_rank` pool cap — the number
# the nightly comparison produces is the number asserted below.
_SNAPSHOT_SCRIPT = ROOT / "scripts" / "memory" / "measure_overreach_snapshot.py"


def _run_snapshot(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SNAPSHOT_SCRIPT), *args],
        capture_output=True, text=True, timeout=180, cwd=str(ROOT))


def test_the_snapshot_measurement_refuses_a_facts_root_pointing_at_the_live_tree():
    """Before its numbers mean anything: the script must run, and `--facts-root`
    naming the live fact tree outright must be refused, not honoured."""
    assert _SNAPSHOT_SCRIPT.exists(), f"{_SNAPSHOT_SCRIPT} is gone"
    # The live root as `app.paths` computes it in this same checkout — the script
    # runs with this ROOT as its cwd, so its own `app.paths` resolves identically.
    from app.paths import VAULT_FACTS_ROOT
    refused = _run_snapshot(["--facts-root", str(VAULT_FACTS_ROOT)])
    assert refused.returncode == 2, refused.stderr[-300:]
    assert "live fact tree" in refused.stderr, refused.stderr[-300:]


def test_the_snapshot_number_sees_over_reach_and_ignores_a_correction():
    """Clause 11, measured on a snapshot copy: expiring an eval-named entity's
    top-scoring fact yields a nonzero regression count, expiring a near-duplicate's
    older twin yields zero."""
    done = _run_snapshot(["--json"])
    assert done.returncode == 0, done.stderr[-400:]
    m = json.loads(done.stdout)

    # Each write touched exactly one fact — including on an entity whose two
    # category files share the id `fact-001` (clauses 1/2, through the real store).
    assert m["expired_by_overreach_write"] == 1
    assert m["expired_by_near_duplicate_write"] == 1

    # A pass over an unchanged corpus finds nothing: the detector cannot fire on
    # noise, so any nonzero below is the writes and nothing else.
    assert m["same_run_regressions"] == []

    # The over-reach: the entity the query expected is no longer in the answer.
    reg = m["regressions_after_writes"]
    assert len(reg) == 1, reg
    assert reg[0]["query"] == "Snapshot Alpha indexer throughput", reg
    assert reg[0]["entities"] == ["snapshot alpha indexer"], reg
    assert m["matched_entities_before"][reg[0]["query"]] == ["snapshot alpha indexer"]
    assert m["matched_entities_after"][reg[0]["query"]] == []

    # The correction: retiring the older twin of a near-duplicate is not a
    # regression, and that query still answers.
    assert not any(r["query"] == "Snapshot Beta queue port" for r in reg), reg
    assert m["matched_entities_after"]["Snapshot Beta queue port"] == ["snapshot beta"]

    # And this is where `fact_entity_recall` is the wrong instrument: it reads
    # 1.0 → 1.0 across the Beta write, which is why the loop needs the count above
    # as its gate and not that average.
    assert m["fact_entity_recall_before"]["Snapshot Beta queue port"] == 1.0
    assert m["fact_entity_recall_after"]["Snapshot Beta queue port"] == 1.0


def test_the_eval_scorer_counts_only_an_entity_that_left_the_answer():
    """`count_overreach_regressions`' own semantics, over hand-built records.

    It reports an expected entity that was in the fact answer before and is not
    now, on the query that named it; it stays silent when the entity is still
    there. The end-to-end half — that expiring one top-scoring fact actually
    ejects an entity from the ten-slot pool, and that retiring a near-duplicate's
    older twin does not — is the snapshot test above, which measures it through the
    real `_rank` rather than simulating it here.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)

    def _records(fact_entities: list[str]) -> list[dict]:
        scored = run_eval._score({"query": "q1", "expect_entities": ["Assistant"]},
                                 {"documents": [], "facts": [
                                     {"entity": e, "fact": f"claim about {e}"}
                                     for e in fact_entities]})
        return [{"query": "q1", "category": "c", "scoring": scored}]

    baseline = _records(["assistant"])
    overreach = _records([])                    # its only expected entity is gone
    assert run_eval.count_overreach_regressions(baseline, overreach) == [
        {"query": "q1", "category": "c", "entities": ["assistant"]}]

    retired_twin = _records(["assistant"])      # survivor still carries it
    assert run_eval.count_overreach_regressions(baseline, retired_twin) == []
    # And the average cannot see the difference — which is why this exists.
    assert [r["scoring"]["fact_entity_recall"] for r in baseline] == [1.0]
    assert [r["scoring"]["fact_entity_recall"] for r in overreach] == [0.0]


# ── clause 12: the doc cannot cite the number without the caveat ─────────────

def test_memory_md_names_the_blast_radius_behind_the_metric_move():
    doc = (ROOT / "architecture" / "memory.md").read_text(encoding="utf-8")
    section = doc[doc.index("## 4."):] if "## 4." in doc else doc
    assert "0.35 → 0.30" in section
    window = section[max(0, section.index("0.35 → 0.30") - 1200):
                     section.index("0.35 → 0.30") + 2500]
    for phrase in ("blast-radius", "category file", "25"):
        assert phrase in window, f"{phrase!r} missing beside the number it explains"
