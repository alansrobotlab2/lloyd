"""Backlog #376 — the `improve` feedback loop and the unified memory surface.

Three things are pinned here. The first two were absent when #376 landed; the
third (#699) is pinned in section 1b below: the drift slice must be the newest
writes rather than the alphabetically-first ones, and every run must state the
pool it selected from.

1. **An `improve`-equivalent.** A consumer that reads a *real* feedback
   signal, treats the existing contradiction detector as *evidence* rather
   than a verdict, and acts only through the two existing fact writers
   (`fact_resolve(auto_resolve=…)` / `fact_invalidate`) while recording
   before/after active-fact counts. It is dry-run by default.

   The detector alone is not a verdict: `_token_overlap > 0.6` fires on two
   facts phrased alike. That is exactly why `fact_resolve`'s `auto_resolve`
   stopped defaulting to true, and why every action this loop takes must
   carry an independent reason — the user contested the entity, or the
   `created_at` ordering says which fact is the current one.

2. **A unified entry point.** `remember` / `recall` / `forget` over the
   19 memory-family tools, so there is one verb per cognitive operation and
   the long-tail tools stay available as the escape hatch.

Run: .venvs/lloyd/bin/python -m pytest tests/test_memory_improvement.py
"""
import asyncio
import ast
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import fact_improvement as fi          # noqa: E402
from agent_mcp import facts                            # noqa: E402
from app import kg_store                               # noqa: E402
from app import paths as app_paths                     # noqa: E402


# ── fixture: a fact tree + store + vault, all in tmp_path ────────────────────

@pytest.fixture
def world(tmp_path, monkeypatch):
    """Temp facts tree, KG store and vault root, wired into every reader."""
    import agent_mcp._shared as shared
    from agent_mcp import retrieval

    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    vault_root = tmp_path / "vault"
    (vault_root / "memory").mkdir(parents=True)
    (vault_root / "lloyd").mkdir()

    monkeypatch.setattr(shared, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts_root)
    from agent_mcp import facts as facts_mod
    monkeypatch.setattr(facts_mod, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(fi, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(fi, "CORRECTIONS_PATH", vault_root / "memory" / "corrections.md")
    # The live log lives in `lloyd/USER.md`, and the reader consults both. Left
    # as None, `corrections_paths()` resolves the two constants above at call
    # time; pinning a list here would be a third path to keep in sync.
    monkeypatch.setattr(fi, "CORRECTIONS_BULLETS_PATH", vault_root / "lloyd" / "USER.md")
    monkeypatch.setattr(fi, "CORRECTIONS_SOURCES", None)
    monkeypatch.setattr(fi, "RECORD_DIR", tmp_path / "records")
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None

    st = kg_store.configure(tmp_path / "kg.sqlite")
    yield facts_root, st, vault_root
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None


#: Pass this as a fact's `created_at` or `source_doc` value to leave that key
#: out of the written front matter entirely. `_read_facts_cached` hands back the
#: YAML entries verbatim, so a fact whose file never carried a key comes back as
#: a dict that *omits* the key — which is not the same object as the key present
#: with value None. That distinction is the live shape, not a hypothetical: the
#: 2026-09-21 `pass@k` winner row (#1348) had neither `created_at` nor
#: `source_doc` among its keys (`['category', 'confidence', 'entity',
#: 'event_date', 'fact', 'id', 'provenance', 'source_file']`), so a guard
#: written `f["created_at"]` raises `KeyError` on the exact row it exists to
#: catch, and a fixture that could only write `created_at: None` could not
#: reproduce it at all.
OMIT = object()


def _write_facts(root, entity, category, facts):
    """One fact file with explicit per-fact fields (mirrors the real shape).

    `created_at` is required unless it is `OMIT`; `source_doc` defaults to
    `None` and is written as `None`, so every fixture here has always written a
    fact that carries `created_at` — the shape the confidence tests in section 2
    and 2b are built on, and the reason the attribution guard (#1348) does not
    disturb them. `OMIT` is the only way to write the keyless row the live
    corpus is mostly made of.
    """
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    prepared = []
    for i, f in enumerate(facts, start=1):
        created_at = f["created_at"]
        source_doc = f["source_doc"] if "source_doc" in f else None
        row = {
            "fact": f["fact"], "confidence": f.get("confidence", 0.9),
            # An explicit id wins: ids are per-file counters, so a test that
            # wants two category files to share `fact-001` — the collision the
            # live corpus actually has — has to be able to say so. Without the
            # override every helper-written id is category-prefixed and
            # collision-free, which is why the collision went untested.
            "category": category, "id": f.get("id") or f"{category[:4]}-{i:03d}",
            "invalid_at": None, "expired_at": None, "provenance": "STATED",
        }
        if created_at is not OMIT:
            row["created_at"] = created_at
            row["valid_at"] = created_at
        if source_doc is not OMIT:
            row["source_doc"] = source_doc
        prepared.append(row)
    fm = {"type": "facts", "entity": entity, "category": category, "facts": prepared}
    (d / f"{entity}-{category}.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n", encoding="utf-8")


def _active(st, entity=None):
    return st.facts_idx.count(entity=entity, active_only=True)


def _reindex(st, root):
    st.facts_idx.reindex(root=root)


def _days_ago(n, hours=0):
    return (datetime.now(timezone.utc) - timedelta(days=n, hours=hours)).isoformat()


def _filler(prefix, n, days=20, confidence=0.9):
    """`n` facts that pair with nothing — not each other, not an opposing term.

    Size a scan up to the bound without the size coming from claims the scan would
    flag. The detector pairs two facts above 0.6 token overlap, so 50 rows phrased
    alike are C(50,2) = 1,225 near-duplicate pairs in one category and a test that
    then asserted a pair count would be asserting nothing about the pair it was
    built for. Each row here carries the entity prefix and the word `marker`, and
    six tokens unique to itself, so the Jaccard overlap the detector computes
    between any two rows is 2/14 = 0.14 against its 0.6 trigger. The six are
    coinages and none is a member of `_OPPOSING_PAIRS`, so a filler row cannot
    become an opposing-terms pair either — nor pair with a test's real pair, which
    sits 0.083 from a filler row.
    """
    return [{"fact": f"{prefix} marker zts{i} qlu{i} vex{i} wray{i} yron{i} zura{i}",
             "created_at": _days_ago(days), "confidence": confidence}
            for i in range(n)]


# ── 1. signals: a real source, not an invented one ───────────────────────────

def test_correction_signal_names_the_entity_the_user_contested(world):
    """`memory/corrections.md` entries are the user's own corrections; the
    entity named in the entry is the one worth re-checking."""
    facts_root, st, vault_root = world
    # An entity with no facts has nothing to improve, and a heading that names
    # an unknown word must not become a fact edit — so TTS has to exist first.
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    # Relative dates, not literals: this fixture feeds a windowed read, and a
    # hard-coded heading is inside the window only until the calendar says
    # otherwise — the same trap #802 names for the live log, and one that a
    # green suite cannot see coming (`_ago`, section 1c).
    (vault_root / "memory" / "corrections.md").write_text(
        "# Corrections Log\n\n"
        f"## {_ago(1)} 07:00 PDT — TTS service status\n"
        "**Correction:** TTS is fixed; the built-in voices were returning 500s.\n\n"
        f"## {_ago(2)} 09:00 PDT — plain prose with no entity\n"
        "**Correction:** nothing entity-shaped here.\n",
        encoding="utf-8")
    found = fi.read_correction_signals()
    assert [s["entity"] for s in found] == ["TTS"], found
    assert found[0]["source"] == "corrections"


def test_drift_signal_finds_the_tree_a_writer_touched_today(world):
    """A fact file written in the last N days is a claim that may already be
    stale — it is the only automatic feedback source the store itself offers."""
    import os
    facts_root, st, _ = world
    _write_facts(facts_root, "FRESH", "state",
                 [{"fact": "FRESH is running on the box.", "created_at": _days_ago(40)},
                  {"fact": "FRESH runs two workers.", "created_at": _days_ago(2)}])
    _write_facts(facts_root, "STALE", "state",
                 [{"fact": "STALE is not running.", "created_at": _days_ago(60)}])
    _reindex(st, facts_root)
    old = datetime.now() - timedelta(days=30)
    stale_file = facts_root / "STALE" / "STALE-state.md"
    os.utime(stale_file, (old.timestamp(), old.timestamp()))
    os.utime(facts_root / "STALE", (old.timestamp(), old.timestamp()))
    found = fi.read_drift_signals(days=3)
    assert [s["entity"] for s in found] == ["FRESH"], found
    assert found[0]["source"] == "drift"
    # #1461: the backdated mtimes above no longer decide anything. Freshen
    # STALE's file the way the nightly rebuild does — rewritten, rows unchanged
    # — and it is still not drift: its newest row was created 60 days ago.
    os.utime(stale_file, None)
    os.utime(facts_root / "STALE", None)
    found = fi.read_drift_signals(days=3)
    assert [s["entity"] for s in found] == ["FRESH"], found


def test_collect_signals_dedupes_by_entity(world):
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(1)}])
    _reindex(st, facts_root)
    # Both sources must name TTS for the dedupe to be exercised: a row created
    # yesterday (drift reads `created_at` since #1461, and this row used to be
    # drift only through its file mtime) and a dated, in-window correction (an
    # undated heading names no entity since #802).
    (vault_root / "memory" / "corrections.md").write_text(
        f"## {_ago(1)} 07:00 PDT — TTS regression\n**Correction:** TTS broke again.\n",
        encoding="utf-8")
    assert "TTS" in {s["entity"] for s in fi.read_drift_signals(days=3)}
    assert "TTS" in {s["entity"] for s in fi.read_correction_signals()}
    signals = fi.collect_signals(sources=("corrections", "drift"))
    assert sum(1 for s in signals if s["entity"] == "TTS") == 1
    assert signals[0]["source"] == "corrections", signals


# ── 1b. #699: the drift slice must be the newest writes, and its size stated ──
#
# Nightly `--limit 40` took the ASCII-earliest 40 of the drifted pool — 1,594
# entities with a fact file written in the last 3 days, measured 2026-09-15 —
# so the pass never looked at today's writes: the records for 09-12, 09-13 and
# 09-14 hold the identical 40 entities with `actions_planned=0`, and that slice
# shared 0 of 38 entities with the 38 most-recently-written ones. Two defects,
# both pinned below: the ranking, and a report that gave no denominator for its
# zero. The overlap number itself moves with the nightly pool — re-measured
# against live `main` on 2026-09-16 it was 2 of 38 over 1,552 candidates, and
# 0 of 38 over 1,269 the morning of the same day — so what these tests pin is
# the ranking, not a number. Post-fix, the same live check on 2026-09-16 reads
# 38 of 38 over that same 1,552.


def _aged(facts_root, entity, days=0, hours=0):
    """Backdate every fact row of one entity so its drift age is explicit.

    Stamps each row's `created_at` — the field the drift signal reads since
    #1461 — and deliberately leaves the file mtime at "just now": the rewrite
    here is what the nightly rebuild does to every file, so a fixture that
    passes with fresh mtimes is one the mtime signal could not have passed.

    Fails when `entity` matches no fact file. A helper that silently no-ops on a
    missed target leaves that entity freshly written, so a fixture meaning "this
    one is 30 days old" quietly becomes "this one is new" and the test it feeds
    keeps passing while pinning nothing — the review rung of this item's first
    round found the helper doing exactly that, and clause 5 is the guard.
    """
    stamp = _days_ago(days, hours=hours)
    paths = list((facts_root / entity).glob("*.md"))
    assert paths, (
        f"_aged({entity!r}) matched no *.md under {facts_root}, so the intended "
        f"backdating never happened and that entity is still freshly written")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        end = text.index("\n---", 3)
        fm = yaml.safe_load(text[3:end])
        assert fm.get("facts"), f"{path} carries no fact rows to backdate"
        for row in fm["facts"]:
            row["created_at"] = stamp
        path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}{text[end + 1:]}",
                        encoding="utf-8")


def _five_drifted(facts_root, st):
    """Five entities that all drifted, aged 6 hours apart: E0's last write is 24 h
    ago, E1's 18 h, E2's 12 h, E3's 6 h, and E4's is now. Returned in the
    newest-first order the fixed ranking must produce."""
    for name in ("E0", "E1", "E2", "E3", "E4"):
        _write_facts(facts_root, name, "state",
                     [{"fact": f"{name} is running on the box.", "created_at": _days_ago(2)}])
    _reindex(st, facts_root)
    for i, name in enumerate(("E0", "E1", "E2", "E3", "E4")):
        _aged(facts_root, name, hours=(4 - i) * 6)
    return ["E4", "E3", "E2", "E1", "E0"]


def test_drift_limit_takes_the_newest_write_not_the_alphabetical_head(world):
    """#699 clause 1: two entities whose fact files have different mtimes and
    `limit=1` must return the *newer* one — even though the older one sorts
    first by name, which is precisely the slice the nightly pass selected."""
    facts_root, st, _ = world
    for name in ("ALPHA_STALE", "ZEBRA_FRESH"):
        _write_facts(facts_root, name, "state",
                     [{"fact": f"{name} is running on the box.", "created_at": _days_ago(2)}])
    _reindex(st, facts_root)
    _aged(facts_root, "ALPHA_STALE", days=2)   # ASCII-earliest, written 2 days ago
    _aged(facts_root, "ZEBRA_FRESH", hours=1)  # ASCII-latest, written an hour ago

    found = fi.read_drift_signals(days=3, limit=1)
    assert [s["entity"] for s in found] == ["ZEBRA_FRESH"], found
    assert found[0]["source"] == "drift"


def test_drift_signals_are_ranked_newest_first_as_a_list(world):
    """The ranking holds for the whole returned slice, not just the survivor of
    a limit of 1: five entities written 6 hours apart come back newest-first,
    and each `evidence` stamp agrees with the order it was ranked on."""
    facts_root, st, _ = world
    expected = _five_drifted(facts_root, st)

    found = fi.read_drift_signals(days=3, limit=5)
    assert [s["entity"] for s in found] == expected, found
    stamps = [s["evidence"] for s in found]
    assert stamps == sorted(stamps, reverse=True), stamps
    # The slice a truncated run takes is a prefix of the full ranking, so the
    # `limit=38` set is the 38 newest of the whole candidate pool by construction.
    full = fi.read_drift_signals(days=3, limit=10 ** 9)
    assert full[:3] == found[:3], (full[:3], found[:3])


def test_run_record_names_the_denominator_it_selected_from(world):
    """#699 clause 2: `actions planned=0` was a statement about 40 entities read
    as a statement about the tree. The record must carry the candidate total the
    run selected from, distinct from `signals`, so the scanned fraction is
    computable from the JSON a run already writes."""
    facts_root, st, _ = world
    expected = _five_drifted(facts_root, st)

    rec = fi.run_improvement(sources=("drift",), days=3, limit=2)
    assert rec["drift_candidates_total"] == 5, rec["drift_candidates_total"]
    assert rec["signals"] == 2, rec["signals"]
    assert rec["entities"] == expected[:2], rec["entities"]
    # The zero the item complained about, now readable as a zero over 2 of 5.
    assert rec["actions_planned"] == 0
    assert rec["signals"] < rec["drift_candidates_total"]

    saved = json.loads(Path(rec["record_path"]).read_text(encoding="utf-8"))
    assert saved["drift_candidates_total"] == 5, saved.keys()
    assert saved["drift_candidates_total"] != saved["signals"]

    # Explicit entities consult no drift population, so there is no denominator
    # to print — 0 would be a number the run never measured.
    named = fi.run_improvement(entities=["E4"], record=False)
    assert named["drift_candidates_total"] is None, named["drift_candidates_total"]


def test_cli_summary_reports_scanned_over_total(world, capsys, monkeypatch):
    """#699 clause 3: the stdout line is what a worker quotes in its report, so
    it carries the denominator too — `scanned=2 of 5`, not a bare `scanned=2`."""
    facts_root, st, _ = world
    _five_drifted(facts_root, st)
    spec = importlib.util.spec_from_file_location("fact_improvement_cli", _SCRIPT_PATH)
    cli = importlib.util.module_from_spec(spec)
    sys.modules["fact_improvement_cli"] = cli
    spec.loader.exec_module(cli)

    monkeypatch.setattr(sys, "argv", ["fact-improvement.py", "--sources", "drift", "--limit", "2"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "entities scanned=2 of 5 drift candidates" in out, out


def test_improvement_reports_the_denominator_in_its_record(world):
    """The record is the pass's only report: the nightly script prints it and
    writes it to `_pipeline/improvement/`. Pin that the denominator is in it.
    (It was pinned across the `improve` MCP tool's serialisation until that
    tool was retired on 2026-09-23; nothing calls the pass over MCP now.)"""
    facts_root, st, _ = world
    _five_drifted(facts_root, st)

    payload = fi.run_improvement(sources=("drift",), limit=2)
    payload = json.loads(json.dumps(payload))
    assert payload["drift_candidates_total"] == 5, sorted(payload)
    assert payload["signals"] == 2, payload["signals"]
    # #1383: the store verdict is read by whoever receives this payload, not
    # only by the script's exit code, so it has to survive the same
    # serialization. `improve()` is a pass-through, which is exactly why the
    # key is worth pinning here rather than assumed.
    assert payload["store_ok"] is True and payload["store_error"] is None, payload


def test_corrections_outrank_drift_when_every_drift_write_is_newer(world):
    """#699 clause 4: re-ranking drift by recency must not promote drift over a
    correction. `TTS` was contested by the user and its last write is 30 days
    old, so every drift mtime in the tree is newer — and it still comes first,
    followed by the newest drift, not the alphabetically-first one."""
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(30)}])
    for name in ("AAA", "BBB"):
        _write_facts(facts_root, name, "state",
                     [{"fact": f"{name} is running on the box.", "created_at": _days_ago(1)}])
    _reindex(st, facts_root)
    _aged(facts_root, "TTS", days=30)
    _aged(facts_root, "AAA", days=1)
    _aged(facts_root, "BBB", hours=1)
    # `_ago`, not a literal date: the corrections branch is windowed, so a
    # hard-coded heading silently stops being a correction (see 1c).
    (vault_root / "memory" / "corrections.md").write_text(
        f"## {_ago(1)} 09:00 PDT — TTS regression\n"
        "**Correction:** TTS broke again.\n", encoding="utf-8")

    # The docstring's premise — every drift mtime is newer than TTS's — is a
    # measured claim, not a fixture intention: if the 30-day backdating above had
    # missed its target, TTS would still be in the drift pool and the ordering
    # below would pass for the wrong reason (dedupe keeps corrections first).
    drift_pool = {s["entity"] for s in fi.read_drift_signals(days=3, limit=10 ** 9)}
    assert "TTS" not in drift_pool, sorted(drift_pool)
    assert {"AAA", "BBB"} <= drift_pool, sorted(drift_pool)

    signals = fi.collect_signals(sources=("corrections", "drift"), days=3, limit=2)
    assert [(s["entity"], s["source"]) for s in signals] == [
        ("TTS", "corrections"), ("BBB", "drift")], signals


# ── 1b'. #1461: drift reads per-fact created_at, and names a sweep or a stall ──
#
# The nightly rebuild rewrites the whole fact tree, so on 2026-09-25 11,806 of
# 11,807 entity dirs had a file modified inside the 3-day window: the "drift
# pool" was the corpus and `scanned=40 of 11806` named the tree, not a slice.
# The inverse was silent: a rebuild stopped for >3 days would print
# `signals=0` and exit 0. Every tree below is written in the test, so every
# file's mtime is "now" — the rebuild's shape — and only `created_at` differs.

def _dated_tree(facts_root, st, recent=(), old=()):
    """Entities with rows created 2 h ago (`recent`) and 30 days ago (`old`)."""
    for name in recent:
        _write_facts(facts_root, name, "state",
                     [{"fact": f"{name} is running.", "created_at": _days_ago(0, hours=2)}])
    for name in old:
        _write_facts(facts_root, name, "state",
                     [{"fact": f"{name} is running.", "created_at": _days_ago(30)}])
    _reindex(st, facts_root)


def _cli(monkeypatch, capsys, *argv):
    spec = importlib.util.spec_from_file_location("fact_improvement_cli_1461", _SCRIPT_PATH)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", ["fact-improvement.py", *argv])
    code = cli.main()
    return code, capsys.readouterr().out


def test_drift_pool_is_a_strict_subset_when_only_mtimes_are_fresh(world):
    """Clause 1: five dirs rewritten just now, two with rows created inside the
    window. The mtime signal called all five drift; the pool is the two."""
    facts_root, st, _ = world
    _dated_tree(facts_root, st, recent=("NEW1", "NEW2"), old=("OLD1", "OLD2", "OLD3"))

    candidates, census = fi._drift_scan(3)
    assert sorted(c["entity"] for c in candidates) == ["NEW1", "NEW2"], candidates
    assert census["corpus"] == 5 and census["pool"] == 2, census
    assert census["fraction"] == 0.4 and census["status"] == "slice", census

    rec = fi.run_improvement(sources=("drift",), days=3, limit=40, record=False)
    assert rec["drift_candidates_total"] == 2
    assert rec["drift_corpus_total"] == 5
    assert rec["drift_pool_fraction"] == 0.4
    assert rec["drift_status"] == "slice"


def test_a_pool_that_is_the_corpus_is_named_a_sweep(world, capsys, monkeypatch):
    """Clause 2: when every dir has a row created in the window (a rebuild
    re-stamped them), the count line carries the fraction and stdout says
    `sweep` — `of 5 drift candidates` alone read as a selection."""
    facts_root, st, _ = world
    _dated_tree(facts_root, st, recent=("A1", "A2", "A3", "A4", "A5"))

    code, out = _cli(monkeypatch, capsys, "--sources", "drift", "--limit", "2")
    assert code == 0, out
    assert "entities scanned=2 of 5 drift candidates (5/5 entity dirs = 100.00% of the corpus)" \
        in out, out
    assert "[drift] sweep: 5 of 5 entity dirs" in out, out
    assert "[drift] 0 candidates" not in out, out


def test_an_empty_pool_prints_a_stale_line_not_a_bare_zero(world, capsys, monkeypatch):
    """Clause 3: nothing created inside the window across a populated tree is
    a stopped writer. The record says `stale` with the newest row's date, and
    stdout says so beside the counts — the drift twin of #802's corrections
    line."""
    facts_root, st, _ = world
    _dated_tree(facts_root, st, old=("OLD1", "OLD2"))

    rec = fi.run_improvement(sources=("drift",), days=3, record=False)
    assert rec["drift_candidates_total"] == 0 and rec["signals"] == 0
    assert rec["drift_status"] == "stale", rec["drift_status"]
    assert rec["drift_newest_fact"][:10] == _days_ago(30)[:10], rec["drift_newest_fact"]

    code, out = _cli(monkeypatch, capsys, "--sources", "drift")
    assert "[drift] 0 candidates in window: no fact created in the last 3 days " \
           "across 2 entity dirs; stale since " + _days_ago(30)[:10] in out, out
    assert "[drift] sweep" not in out, out


def test_a_discriminating_pool_prints_neither_drift_caveat(world, capsys, monkeypatch):
    """The two caveats are verdicts, not decoration: a real slice prints the
    fraction and no `[drift]` line at all."""
    facts_root, st, _ = world
    _dated_tree(facts_root, st, recent=("NEW1",), old=("OLD1", "OLD2", "OLD3"))

    code, out = _cli(monkeypatch, capsys, "--sources", "drift")
    assert "entities scanned=1 of 1 drift candidates (1/4 entity dirs = 25.00% of the corpus)" \
        in out, out
    assert "[drift]" not in out, out


def test_drift_census_is_null_when_drift_was_not_consulted(world):
    """A run that walked no tree measured no corpus: every census key is None,
    never 0, for explicit entities and for `--sources corrections`."""
    facts_root, st, _ = world
    _dated_tree(facts_root, st, recent=("NEW1",))
    for rec in (fi.run_improvement(entities=["NEW1"], record=False),
                fi.run_improvement(sources=("corrections",), record=False)):
        for key in ("drift_candidates_total", "drift_corpus_total", "drift_pool_fraction",
                    "drift_status", "drift_newest_fact", "drift_undated_dirs"):
            assert rec[key] is None, (key, rec[key])


def test_drift_reads_front_matter_rows_only_and_counts_undated_dirs(world):
    """A `created_at:` line in the rendered body is not a row, and a dir whose
    rows carry no `created_at` at all is counted `undated`, never drift."""
    facts_root, st, _ = world
    _write_facts(facts_root, "KEYLESS", "state",
                 [{"fact": "KEYLESS has no stamp.", "created_at": OMIT}])
    body = facts_root / "KEYLESS" / "KEYLESS-state.md"
    body.write_text(body.read_text(encoding="utf-8")
                    + f"\ncreated_at: '{_days_ago(0)}'\n", encoding="utf-8")
    _dated_tree(facts_root, st, recent=("NEW1",))

    candidates, census = fi._drift_scan(3)
    assert [c["entity"] for c in candidates] == ["NEW1"], candidates
    assert census["undated"] == 1 and census["corpus"] == 2, census


# ── 1c. #802: the corrections window, its threading, and its stale marker ────
#
# `memory/corrections.md` has had one commit ever (the 2026-08-22 baseline) and
# its newest dated heading is 2026-05-08, yet the corrections branch of the
# signal union applied no date filter at all while the drift branch took
# `days=DRIFT_WINDOW_DAYS`. The two 2026-05-08 headings therefore entered every
# nightly run as if they were fresh corrections — and, because
# `_collect_signals` extends corrections into the union first, ahead of every
# drift candidate. A window constant exists now; what is pinned below is the
# four things that make it real: it filters in both directions, the caller's
# window is the one the reader uses, a wholly stale log is recorded as stale
# rather than as a zero, and the function's own prose names the route it is.
#
# Dates here are relative to an explicit `now` (`_ago`) or to the constant, so
# no fixture ages out of the window it is testing — the shape that made the two
# hard-coded headings above (#802's own hazard, noted at filing) rot within a
# fortnight of being written.

def _ago(days: int, now: datetime | None = None) -> str:
    """ISO date `days` before `now` (default: real now, UTC)."""
    base = now or datetime.now(timezone.utc)
    return (base - timedelta(days=days)).date().isoformat()


def _corrections_log(path, entries):
    """Write a `memory/corrections.md` carrying one dated heading per entry."""
    path.write_text(
        "# Corrections Log\n\n" + "".join(
            f"## {date} 07:00 PDT — {entity} service status\n"
            f"**Correction:** the {entity} status was wrong.\n\n"
            for date, entity in entries),
        encoding="utf-8")
    return path


def test_the_corrections_window_admits_an_in_window_entry_and_drops_an_older_one(world):
    """Both directions, pinned against an injected `now`.

    An entry 3 days old is a correction; one 15 days past the window is not an
    entry at all, but it is still counted, because a window that silently drops
    is indistinguishable from a log that is empty.
    """
    facts_root, st, vault_root = world
    now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    for name in ("TTS", "ZED"):
        _write_facts(facts_root, name, "state",
                     [{"fact": f"{name} is running on the box.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    head = _corrections_log(vault_root / "memory" / "corrections.md", [
        (_ago(3, now), "TTS"), (_ago(fi.CORRECTIONS_WINDOW_DAYS + 15, now), "ZED")])

    found = fi.read_correction_signals(now=now)
    assert [s["entity"] for s in found] == ["TTS"], (
        "an entry inside the window must survive and one past it must not", found)
    per = fi.last_corrections_read()["sources"][str(head)]
    assert (per["in_window"], per["outside_window"]) == (1, 1), per


def test_collect_signals_threads_its_corrections_window_into_the_read(world):
    """The window is an argument, not only a default.

    Before this, `collect_signals(days=…)` threaded the window to the drift
    branch alone and the corrections branch received none, so no caller could
    narrow or widen what a stale log contributes. The same log must yield a
    signal under a 60-day window and nothing under a 7-day one, and the read
    must report the window it actually used.
    """
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    _corrections_log(vault_root / "memory" / "corrections.md", [(_ago(40), "TTS")])
    (vault_root / "lloyd" / "USER.md").write_text(
        "# User\n\n## corrections_log\n", encoding="utf-8")

    narrow = fi.collect_signals(sources=("corrections",), corrections_days=7)
    assert narrow == [], "a 40-day-old correction must not pass a 7-day window"
    assert fi.last_corrections_read()["window_days"] == 7

    wide = fi.collect_signals(sources=("corrections",), corrections_days=60)
    assert [s["entity"] for s in wide] == ["TTS"], (
        "the same entry must pass a 60-day window — the argument, not the "
        "constant, decides what is admitted", wide)
    assert fi.last_corrections_read()["window_days"] == 60


def test_a_fully_stale_corrections_log_is_recorded_as_stale_not_as_zero(world):
    """"0 corrections" and "the log has been stale since 2026-05-08" are two
    different verdicts, and a run record that cannot say which it means is the
    instrument that reads as a quiet week while it is dead.
    """
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    newest = _ago(400)
    _corrections_log(vault_root / "memory" / "corrections.md", [(newest, "TTS")])
    (vault_root / "lloyd" / "USER.md").write_text(
        "# User\n\n## corrections_log\n\nStanding corrections only.\n", encoding="utf-8")

    rec = fi.run_improvement(sources=("corrections",))
    assert rec["signals"] == 0, rec["signals"]
    assert rec["corrections_status"] == "no_entries_in_window", rec["corrections_status"]
    # The marker: newest entry in the stale log, and the window that excluded it.
    assert rec["corrections_stale_since"] == newest, rec["corrections_stale_since"]
    assert rec["corrections_window_days"] == fi.CORRECTIONS_WINDOW_DAYS
    assert rec["corrections_path"] is None, (
        "no signal came from any log, so no log may be named as its source")

    # The other zero stays distinguishable from it: a log with nothing in it is
    # `empty`, and there is no date to name.
    (vault_root / "memory" / "corrections.md").write_text(
        "# Corrections Log\n", encoding="utf-8")
    fresh = fi.run_improvement(sources=("corrections",))
    assert fresh["corrections_status"] == "empty", fresh["corrections_status"]
    assert fresh["corrections_stale_since"] is None, fresh["corrections_stale_since"]
    assert fresh["corrections_window_days"] == fi.CORRECTIONS_WINDOW_DAYS


def test_cli_reports_a_stale_corrections_log_instead_of_a_bare_zero(world, capsys,
                                                                    monkeypatch):
    """The process boundary: the nightly job runs this script and quotes its
    stdout, so a dead channel has to be visible on the line it prints and in the
    record it persists — not only in a key inside the JSON nobody opens.
    """
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    newest = _ago(400)
    _corrections_log(vault_root / "memory" / "corrections.md", [(newest, "TTS")])
    (vault_root / "lloyd" / "USER.md").write_text(
        "# User\n\n## corrections_log\n\nStanding corrections only.\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("fact_improvement_cli_stale",
                                                 _SCRIPT_PATH)
    cli = importlib.util.module_from_spec(spec)
    sys.modules["fact_improvement_cli_stale"] = cli
    spec.loader.exec_module(cli)

    monkeypatch.setattr(sys, "argv", ["fact-improvement.py", "--sources", "corrections"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert f"log stale since {newest} (window {fi.CORRECTIONS_WINDOW_DAYS} days)" in out, out

    # The same verdict has to survive into the persisted record, since that is
    # what an audit reads a week later.
    record = sorted(Path(fi.RECORD_DIR).glob("*.json"))[-1]
    persisted = json.loads(record.read_text(encoding="utf-8"))
    assert persisted["corrections_stale_since"] == newest, sorted(persisted)
    assert persisted["corrections_window_days"] == fi.CORRECTIONS_WINDOW_DAYS


def test_the_corrections_reader_docstring_names_the_route_and_the_window():
    """The docstring used to assert that nothing routes `corrections.md` into
    *fact* quality while the function below it was exactly that route. Prose
    that denies the call path is how a dead channel survives: the reader, the
    record and the metric all said "no corrections" and nothing said "the
    channel is the fact-quality signal, and here is its window".
    """
    doc = inspect.getdoc(fi.read_correction_signals)
    assert "nothing routes" not in doc, doc
    for named in ("fact quality", "collect_signals", "run_improvement",
                  "CORRECTIONS_WINDOW_DAYS"):
        assert named in doc, f"docstring must name {named!r}: {doc}"


# ── 1c. an UNDATED correction is still a correction (#2198) ──────────────────
#
# The live log is `lloyd/USER.md`'s `## corrections_log`, and its bullets are
# standing prose with no date on the line — the section's own text says it holds
# standing corrections and not a write-changelog (`lloyd/USER.md:74`), and it is
# written both by hand and by the nightly reflection job. The reader used to answer
# that shape by returning `(None, "")`: the prose was discarded with the date, the
# consumer counted `undated` and `continue`d before the token scan, and the channel
# has returned 0 signals every night since it moved. What is pinned below: the prose
# survives (clause 1); one documented fallback date places the entry and can say *no*
# (clause 2); an entry admitted on the log's own clock is never reported as one the
# operator dated, in the read or in the record (clause 3); a zero-signal night names
# the count, the date and where the date came from, on stdout and in the record
# (clause 4); the heading-shaped log is never admitted by that clock, which is a
# decision and not an accident; and the run record the item's numbers were quoted
# from is re-derivable from committed bytes (clause 5).

def _standing_log(path, bullets, *, front="", mtime_days_ago=None):
    """Write a `## corrections_log` bullet log; `bullets` are the lines verbatim.

    Verbatim on purpose: whether a bullet carries a date is the variable under
    test, so the helper must not add one. `front` is the file's front-matter block
    ("" = none, which leaves the file's mtime as the only clock), and
    `mtime_days_ago` stamps the bytes so a test can pin where on the window the
    file itself sits without waiting for a night to pass.

    Either configured path may be handed to it, including `memory/corrections.md`,
    which the live log shapes as headings: the reader chooses heading- or
    bullet-parsing from the file's CONTENT (`_SECTION_HEAD_RE` finds no section in a
    bullet log), so a bullet log written at that path is the same code path a live
    bullet log takes and not a mis-shape. The heading shape's half of that — an
    undatable heading log, which can never be admitted by the file clock — is pinned
    by `test_an_undatable_heading_log_is_never_admitted_by_the_file_clock`.
    """
    body = ((front if front.startswith("---") else f"---\n{front}---\n")
            if front else "")
    body += "# User\n\n## corrections_log\n\nStanding corrections only.\n"
    for bullet in bullets:
        body += f"- {bullet}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    if mtime_days_ago is not None:
        epoch = (datetime.now(timezone.utc) - timedelta(days=mtime_days_ago)).timestamp()
        os.utime(path, (epoch, epoch))
    return path


def test_an_undated_corrections_bullet_returns_its_own_prose(world):
    """Clause 1: an undated entry keeps its text, or nothing downstream can place it.

    The old `return None, ""` was the second half of the bug: a caller that gets
    `""` cannot window the entry, cannot scan it for an entity, and cannot report
    what it dropped. The standing prose in the same section is still NOT an entry —
    the count, not just the text, is what keeps a run from crediting the operator
    with a mistake they never made.
    """
    _, _, vault_root = world
    log = _standing_log(vault_root / "lloyd" / "USER.md", [
        "**Standing**: a loaded-memory line names its source document",
        "**Standing**: TTS health verdicts require a synthetic end-to-end request"])
    entries, status = fi._corrections_entries(log)
    assert status == "undated", status
    assert len(entries) == 2, entries            # the prose paragraph is not an entry
    assert [d for d, _ in entries] == [None, None], entries
    assert entries[0][1] == ("**Standing**: a loaded-memory line names its "
                             "source document"), entries[0]
    assert "TTS health verdicts" in entries[1][1], entries[1]
    assert all(text for _, text in entries), "an undated entry must never return ''"


def test_undated_bullets_are_placed_by_the_log_own_fallback_date(world):
    """Clause 2: one documented clock — front-matter `updated:`, else `timestamp:`,
    else mtime — and it admits entries only inside `window_days`.

    Both directions, because the point of giving the reader a clock is that it can
    refuse: the failure mode the item named is a fallback that silently admits every
    old standing correction forever, and a `timestamp:` 90 days back must yield
    `in_window: 0` and no signal. The third leg (no front matter at all) is pinned
    too, since `lloyd/USER.md` today carries `timestamp:` and NO `updated:`, so the
    waterfall's order is the live behaviour and not a hypothetical.
    """
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS is running on the box.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    _write_facts(facts_root, "ZED", "state",
                 [{"fact": "ZED is running on the box.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    fresh = _standing_log(vault_root / "memory" / "corrections.md",
                          ["TTS built-in voices are the wrong default"],
                          front=f"updated: '{_ago(2)}T09:00:00'\n")
    stale = _standing_log(vault_root / "lloyd" / "USER.md",
                          ["TTS health verdicts require a synthetic request"],
                          front=f"timestamp: '{_ago(90)}T09:00:00'\n")
    got = fi.read_correction_signals()
    assert [s["entity"] for s in got] == ["TTS"], got
    assert got[0]["corrections_path"] == str(fresh)
    # The evidence says which clock placed it, so a reader of the signal alone can
    # see this is a standing correction and not something the operator dated today.
    assert got[0]["entry_date_origin"] == "front_matter:updated", got[0]
    assert got[0]["evidence"] == (
        f"[standing correction; placed by front_matter:updated date {_ago(2)}] "
        "TTS built-in voices are the wrong default"), got[0]
    # One 200-character budget, shared by both shapes: the label is spent from
    # inside it and not stacked on top, so a fallback-placed signal can neither
    # arrive over-long nor arrive unlabelled.
    assert len(got[0]["evidence"]) <= 200, len(got[0]["evidence"])
    assert got[0]["evidence"].startswith("[standing correction; placed by"), got[0]
    read = fi.last_corrections_read()["sources"]
    admitted = read[str(fresh)]
    assert (admitted["entries"], admitted["undated"], admitted["in_window"]) == (1, 1, 1)
    assert admitted["fallback_admitted"] == 1, admitted
    assert admitted["fallback_date"] == _ago(2), admitted
    assert admitted["fallback_origin"] == "front_matter:updated", admitted
    refused = read[str(stale)]
    assert (refused["entries"], refused["undated"], refused["in_window"],
            refused["outside_window"], refused["fallback_admitted"]) == (1, 1, 0, 1, 0), refused
    assert refused["fallback_date"] == _ago(90), refused
    assert refused["status"] == "no_entries_in_window", refused
    assert sum(1 for s in got if s["corrections_path"] == str(stale)) == 0, got
    # mtime leg: no front matter means the bytes' own stamp is the clock, and the
    # report says so rather than leaving the reader to guess which date it used.
    # `ZED` and not `TTS`: an entity a co-read log already claimed is deduped, and
    # a signal lost to dedupe would prove nothing about this clock.
    mtime_log = _standing_log(vault_root / "lloyd" / "USER.md",
                              ["ZED health needs a synthetic request"],
                              mtime_days_ago=5)
    got = fi.read_correction_signals()
    info = fi.last_corrections_read()["sources"][str(mtime_log)]
    assert info["fallback_origin"] == "mtime", info
    assert info["fallback_date"] == _ago(5), info
    assert info["fallback_admitted"] == 1, info
    zed = [s for s in got if s["entity"] == "ZED"]
    assert len(zed) == 1 and zed[0]["entry_date_origin"] == "mtime", got


def test_fallback_admitted_entries_are_never_reported_as_dated_ones(world):
    """Clause 3: the fallback gets its own count and its own status, on both sides.

    `undated` says how many lines carried no date; `fallback_admitted` says how many
    the clock placed inside the window; a log whose ONLY in-window content arrived
    that way reports `fallback_admitted`, never `entries_no_entity` or `signals` —
    while a log that ALSO has a genuinely dated in-window entry keeps a dated status
    and still counts its admitted bullet. The mixed log written below is that case:
    one dated entry and one undated one, both inside the window, so `in_window: 2` and
    `fallback_admitted: 1` — the count records the admission, and the status is not
    `fallback_admitted` either, because the window there is not filled only by the
    clock: one entry in it carries its own date. `fallback_admitted` as a status means
    "the file's date is the whole reason this log has in-window content", which is the
    case worth warning about, and nothing weaker. The two facts stay independent on
    purpose: the status says
    what KIND of evidence got in, the counts say how much of each kind. The vocabulary
    is one flat tuple, so the new member has to sit where its meaning sits: better than
    "nothing got in", worse than the same verdict earned from a date the operator wrote.
    """
    assert "fallback_admitted" in fi.CORRECTIONS_STATUSES, fi.CORRECTIONS_STATUSES
    order = fi.CORRECTIONS_STATUSES
    assert (order.index("entries_no_entity") < order.index("fallback_admitted")
            < order.index("no_entries_in_window")), order
    facts_root, st, vault_root = world
    for name in ("TTS", "ZED"):
        _write_facts(facts_root, name, "state",
                     [{"fact": f"{name} is running on the box.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    only_fallback = _standing_log(
        vault_root / "memory" / "corrections.md",
        ["TTS default voice is wrong", "**Standing**: TTS needs a synthetic check"],
        front=f"updated: '{_ago(2)}T09:00:00'\n")
    # A log with a real dated in-window entry AND an undated one, so the status has
    # to be the dated one and the fallback has to show up only as a count. `ZED`
    # here and `TTS` there because two logs naming one entity dedupe to one signal,
    # and a deduped signal would hide which entry the status was decided from.
    dated_too = _standing_log(
        vault_root / "lloyd" / "USER.md",
        [f"{_ago(3)}: ZED default is wrong", "**Standing**: ZED needs a synthetic check"],
        front=f"updated: '{_ago(2)}T09:00:00'\n")
    got = fi.read_correction_signals()
    assert sorted(s["entity"] for s in got) == ["TTS", "ZED"], got
    read = fi.last_corrections_read()["sources"]
    info = read[str(only_fallback)]
    assert info["status"] == "fallback_admitted", info
    assert (info["undated"], info["fallback_admitted"], info["in_window"]) == (2, 2, 2), info
    mixed = read[str(dated_too)]
    assert mixed["status"] not in ("fallback_admitted", "entries_no_entity",
                                   "no_entries_in_window"), mixed
    assert (mixed["undated"], mixed["fallback_admitted"], mixed["in_window"],
            mixed["outside_window"]) == (1, 1, 2, 0), mixed
    assert mixed["newest_entry"] == _ago(3), mixed   # the clock is not the newest entry
    assert [s["entity"] for s in got if s["corrections_path"] == str(dated_too)] == ["ZED"]
    assert next(s for s in got if s["entity"] == "ZED")["entry_date_origin"] == "entry"
    # The record carries the same distinction the read made, because the owed ruling
    # on whether standing corrections ever produce a USEFUL signal is made from these
    # records and not from a re-read of the log.
    rec = fi.run_improvement(apply=False, sources=("corrections",), limit=2,
                             max_actions=0, record=False)
    src = rec["corrections_sources"][str(only_fallback)]
    assert src["status"] == "fallback_admitted", src
    assert src["fallback_date"] == _ago(2) and src["fallback_admitted"] == 2, src
    assert rec["corrections_sources"][str(dated_too)]["fallback_admitted"] == 1, rec
    fallback_rec = rec["corrections_undated_fallback"]
    assert (fallback_rec["undated"], fallback_rec["admitted"]) == (3, 3), fallback_rec


def test_a_zero_signal_read_names_the_count_the_date_and_where_it_came_from(
        world, capsys, monkeypatch):
    """Clause 4: 0 signals with undated entries present has to explain itself.

    This is the half the run never printed: `memory/corrections.md` had a stale-line,
    so the pass printed one line and read as a two-source pass while the live log
    reported a bare `undated` and nothing else. The report must carry the undated
    count, the fallback date, and whether it came from front matter or the mtime —
    in the record AND on stdout, since a caveat only in the JSON nobody opens is the
    false green #802 was filed about.
    """
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS is running on the box.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    # Neither bullet names a registered entity, so the read legitimately yields zero
    # signals — the exact live shape of `lloyd/USER.md`, both entries prose rules.
    log = _standing_log(vault_root / "lloyd" / "USER.md", [
        "**Standing**: a loaded-memory line names its source document",
        "**Standing**: a health verdict needs a synthetic request"],
        front=f"timestamp: '{_ago(3)}T09:00:00'\n")
    # The co-read log, dated but far outside the window: the stale channel's own
    # line has to survive beside this one, and the file's date convention stays
    # relative so no future calendar can retire this assertion.
    _corrections_log(vault_root / "memory" / "corrections.md", [(_ago(400), "TTS")])
    assert fi.read_correction_signals() == [], fi.last_corrections_read()
    fb = fi.last_corrections_read()["undated_fallback"]
    assert fb["undated"] == 2 and fb["admitted"] == 2, fb
    assert fb["date"] == _ago(3) and fb["origin"] == "front_matter:timestamp", fb
    assert str(fb["undated"]) in fb["summary"] and fb["date"] in fb["summary"], fb
    assert "front matter" in fb["summary"], fb
    assert fi.last_corrections_read()["sources"][str(log)]["undated"] == 2
    # And the mtime leg says which leg it was, rather than borrowing the word
    # "front matter" for a timestamp no operator wrote.
    _standing_log(vault_root / "lloyd" / "USER.md",
                  ["**Standing**: a loaded-memory line names its source document"],
                  mtime_days_ago=4)
    fi.read_correction_signals()
    fb = fi.last_corrections_read()["undated_fallback"]
    assert fb["origin"] == "mtime" and "mtime" in fb["summary"], fb

    # The same process boundary #802 pinned: the nightly job quotes this script's
    # stdout, so the fallback has to reach the printed line and the persisted record,
    # not only a key inside the JSON nobody opens.
    spec = importlib.util.spec_from_file_location("fact_improvement_cli_undated",
                                                  _SCRIPT_PATH)
    cli = importlib.util.module_from_spec(spec)
    sys.modules["fact_improvement_cli_undated"] = cli
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", ["fact-improvement.py", "--sources", "corrections"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    printed = [ln for ln in out.splitlines() if ln.startswith("[corrections]")]
    assert any(_ago(4) in ln and "mtime" in ln and "undated" in ln for ln in printed), out
    record = sorted(Path(fi.RECORD_DIR).glob("*.json"))[-1]
    persisted = json.loads(record.read_text(encoding="utf-8"))
    assert persisted["corrections_undated_fallback"]["origin"] == "mtime", persisted
    # One undated bullet by this point (the rewrite above), placed by the mtime and
    # counted as admitted — the record has to carry both, or the night reads as a
    # bare zero with a date attached.
    src = persisted["corrections_sources"][str(log)]
    assert (src["undated"], src["fallback_admitted"], src["in_window"]) == (1, 1, 1), src
    assert src["fallback_date"] == _ago(4), src


def test_an_undatable_heading_log_is_never_admitted_by_the_file_clock(world):
    """The fallback is for a standing bullet, not for a log the reader cannot parse.

    `_corrections_entries`' heading branch drops a heading whose text carries no
    date, so a heading-shaped log contributes ZERO entries: there is nothing for the
    clock to place, the read stays `undated` with `fallback_date: null`, and no
    `undated_fallback` summary is produced for it. That is deliberate and pinned
    here rather than left to the shape of the code: admitting an undatable HEADING
    by the file's mtime is precisely the silent-admission failure #2198 was filed
    about, and the log that needed a clock is the bullet-shaped one.
    """
    _, _, vault_root = world
    (vault_root / "memory" / "corrections.md").write_text(
        "# Corrections Log\n\n## Standing rule about voices\n\nprose\n\n"
        "## Another rule with no date\n\nprose\n", encoding="utf-8")
    (vault_root / "lloyd" / "USER.md").write_text(
        "# User\n\n## corrections_log\n", encoding="utf-8")
    assert fi._corrections_entries(vault_root / "memory" / "corrections.md") == (
        [], "undated")
    assert fi.read_correction_signals() == [], fi.last_corrections_read()
    info = fi.last_corrections_read()["sources"][
        str(vault_root / "memory" / "corrections.md")]
    assert info["status"] == "undated" and info["entries"] == 0, info
    assert info["fallback_date"] is None and info["fallback_admitted"] == 0, info
    assert fi.last_corrections_read()["undated_fallback"]["undated"] == 0, (
        "a log that yielded no entries must not be counted as undated entries "
        "awaiting a clock")


def test_the_committed_witness_bytes_still_carry_the_quoted_two_silent_channels():
    """Clause 5: the record #2198's numbers came from has history and is re-derivable.

    This is the tree's copy of the vault witness clause 5 names,
    `backlog/data/20261004-210010-dryrun.json` (committed under #2199), and identity
    between the two is CHECKED below rather than asserted in prose: when the vault is
    reachable the bytes must compare equal, and under the gate they are not, because
    the run happens with HOME at the round home where no vault exists — which is
    exactly why a copy had to be committed here for a node to read at all. No hash is
    quoted in this file or in `tests/fixtures/.gitignore`: a bare object-id in a
    comment gets read as a commit by whatever validates citations, and this claim has
    a better witness than a digest. The clause's own re-derive is `wc -l` of the
    record whole, which is why the file is kept whole rather than cut down to the
    two blocks the item quotes: 1052 lines is the figure it asks for.

    What the bytes say, and what the change is a response to: the pass printed
    `signals: 40`, every one of them from drift (`drift_status: slice`,
    `drift_candidates_total: 1887`), while BOTH correction channels returned 0 —
    `memory/corrections.md` 10 entries newest 2026-05-08 and outside the window,
    `lloyd/USER.md` 2 entries with BOTH undated and the bare status `undated`. And no
    key in either source block names a fallback, because before this change there was
    no clock to report: that absence is the defect, recorded.
    """
    witness = ROOT / "tests" / "fixtures" / "improvement_20261004-210010-dryrun.json"
    # The clause names the vault copy, so where the vault is readable the copy in the
    # tree is only a witness if it is the same bytes. Skipped, not assumed, when it is
    # not there — which is the gate's own condition.
    vault_copy = Path.home() / "obsidian" / "backlog" / "data" / (
        "20261004-210010-dryrun.json")
    if vault_copy.exists():
        assert vault_copy.read_bytes() == witness.read_bytes(), (
            "the committed copy has drifted from the vault witness clause 5 names")
    # No `else: skip` — the re-derive below is the node's job and must run in the
    # gate too, where no vault exists at this path. What is conditional is only the
    # comparison against the out-of-band copy, never the figures themselves.
    raw = witness.read_text(encoding="utf-8")
    # `wc -l` counts NEWLINES, and `json.dump` leaves the last `}` unterminated, so
    # the clause's figure is one below the line count a split gives: 1052 by `wc -l`,
    # 1053 by `splitlines()`. Counted the way the clause counts, because a witness
    # pinned to the wrong convention is a witness that fails on someone else's box.
    assert raw.count("\n") == 1052, raw.count("\n")
    assert not raw.endswith("\n") and len(raw.splitlines()) == 1053, "as emitted"
    rec = json.loads(raw)
    # `apply: false` is the record's own dry-run key (the `-dryrun` in the filename is
    # derived from it), so the witness pins that nothing was written on the night the
    # numbers come from — the pass being discussed reported, it did not act.
    assert rec["signals"] == 40 and rec["apply"] is False, rec["signals"]
    assert rec["drift_status"] == "slice" and rec["drift_candidates_total"] == 1887
    assert rec["corrections_status"] == "undated", rec["corrections_status"]
    assert rec["corrections_stale_since"] == "2026-05-08", rec["corrections_stale_since"]
    assert rec["corrections_newest_entry"] == "2026-05-08", "both channels' newest date"
    assert rec["corrections_window_days"] == 30
    src = rec["corrections_sources"]
    by_leaf = {Path(p).parent.name + "/" + Path(p).name: v for p, v in src.items()}
    assert set(by_leaf) == {"memory/corrections.md", "lloyd/USER.md"}, sorted(src)
    # The dated channel was fine and merely stale; the live channel could not be read
    # at all. Two different silences, and the printed pass showed only the first.
    head = by_leaf["memory/corrections.md"]
    assert (head["status"], head["entries"], head["outside_window"], head["undated"],
            head["in_window"], head["signals"]) == (
        "no_entries_in_window", 10, 10, 0, 0, 0), head
    live = by_leaf["lloyd/USER.md"]
    assert (live["status"], live["entries"], live["undated"], live["in_window"],
            live["outside_window"], live["signals"]) == (
        "undated", 2, 2, 0, 0, 0), live
    assert all("fallback_date" not in v for v in src.values()), (
        "the pre-fix record has no clock to report — that absence is the premise")


# ── 2. the loop: evidence + reason, dry-run by default ───────────────────────

def test_equal_confidence_contradiction_needs_a_time_order_reason(world):
    """Detector says "these two disagree", confidences tie → the tie is broken
    by created_at, and only when the loser is clearly older."""
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK (tts.builtin).",
         "created_at": _days_ago(30), "confidence": 0.9},
        {"fact": "TTS built-in voices are broken and returning 500 errors (tts.builtin).",
         "created_at": _days_ago(2), "confidence": 0.9},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("TTS")
    assert plan["contradictions"] >= 1, plan
    assert len(plan["actions"]) == 1
    action = plan["actions"][0]
    assert action["kind"] == "superseded"
    assert action["loser_fact"].startswith("TTS built-in voices are working")
    assert "created" in action["reason"]           # states *why* this side lost
    assert plan["before_active"] == 2


def test_unequal_confidence_is_planned_as_the_confidence_class(world):
    """A low-confidence claim contradicted by a high-confidence one is a
    separate evidence class: the loser is the weak one regardless of which was
    written first."""
    facts_root, st, _ = world
    _write_facts(facts_root, "ASR", "state", [
        {"fact": "ASR wake word detection is working end to end.",
         "created_at": _days_ago(2), "confidence": 0.95},
        {"fact": "ASR wake word detection is broken.",
         "created_at": _days_ago(4), "confidence": 0.4},
    ])
    _reindex(st, facts_root)
    action = fi.plan_entity("ASR")["actions"][0]
    assert action["kind"] == "confidence"
    assert action["loser_fact"].startswith("ASR wake word detection is broken")


def test_one_action_cannot_touch_a_fact_it_did_not_condemn(world):
    """The collateral-damage class this loop exists next to: fact ids are
    per-file counters, so `fact-001` names several facts in one entity.
    `fact_resolve(auto_resolve=true)` selects losers by id and invalidated 25
    facts to change 2 on the live `Assistant` entity. An improve action may
    expire only the fact it names."""
    facts_root, st, _ = world
    loser = "Lloyd built-in voices are working and returning 200 OK (lloyd.voices)."
    _write_facts(facts_root, "Lloyd", "state", [
        {"fact": loser, "created_at": _days_ago(30)},
        {"fact": "Lloyd built-in voices are broken and returning 500 errors (lloyd.voices).",
         "created_at": _days_ago(2)},
    ])
    _write_facts(facts_root, "Lloyd", "usage", [
        {"fact": "Lloyd is used for nightly reflection runs.", "created_at": _days_ago(20)}])
    _write_facts(facts_root, "Lloyd", "preference", [
        {"fact": "Lloyd is preferred over the older assistant.", "created_at": _days_ago(20)}])
    _reindex(st, facts_root)
    # Both files carry a fact with id `state-001`/`usage-001` etc. — the point is
    # the plan condemns exactly one claim and exactly one fact must go.
    assert _active(st, "Lloyd") == 4
    rec = fi.run_improvement(apply=True, entities=["Lloyd"])
    assert rec["actions_taken"] == 1, rec["per_entity"]
    assert _active(st, "Lloyd") == 3
    survivors = [f["fact"] for f in
                 fi._get_facts_sync("Lloyd").get("facts", [])]
    assert loser not in survivors
    assert any("nightly reflection" in f for f in survivors), survivors
    assert any("broken and returning 500" in f for f in survivors), survivors
    assert any("preferred over the older" in f for f in survivors), survivors


def test_same_day_equal_confidence_pair_is_left_alone(world):
    """No basis to pick a winner: same confidence, written an hour apart. This
    is the false-positive class that made `auto_resolve` default to false."""
    facts_root, st, _ = world
    _write_facts(facts_root, "QMD", "state", [
        {"fact": "QMD index is current and queryable.", "created_at": _days_ago(1, 2)},
        {"fact": "QMD index is stale and not queryable.", "created_at": _days_ago(1, 1)},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("QMD")
    assert plan["contradictions"] >= 1
    assert plan["actions"] == []


def test_a_near_duplicate_pair_is_reported_not_deleted(world):
    """The detector's second trigger is token overlap alone — two facts that
    say nearly the same thing. Acting on those is where an `improve` loop eats
    useful facts (measured: 0.35 -> 0.30), so the pair is reported and left
    alone even when the confidences differ."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Transcripts", "usage", [
        {"fact": "Transcripts were extracted from the video with the VTT parser.",
         "created_at": _days_ago(20), "confidence": 0.9},
        {"fact": "Transcripts were extracted from the video with the parser.",
         "created_at": _days_ago(19), "confidence": 0.6},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Transcripts")
    assert plan["contradictions"] >= 1, plan
    assert plan["near_duplicates"] >= 1
    assert plan["actions"] == []
    rec = fi.run_improvement(apply=True, entities=["Transcripts"])
    assert rec["actions_taken"] == 0
    assert _active(st) == 2


def test_godnode_entity_is_refused_not_rescanned(world):
    """The detector refuses above FACT_GODNODE_THRESHOLD; the loop inherits the
    refusal instead of paying O(n²) for noise."""
    facts_root, st, _ = world
    from agent_mcp.retrieval import FACT_GODNODE_THRESHOLD
    facts = [{"fact": f"QMD detail number {i} differs from {i + 1}.",
              "created_at": _days_ago(20)} for i in range(FACT_GODNODE_THRESHOLD + 1)]
    _write_facts(facts_root, "QMD", "state", facts)
    _reindex(st, facts_root)
    plan = fi.plan_entity("QMD")
    assert plan.get("refused") is True
    assert plan["actions"] == []
    # The per-category retry (#1251) must not rescue THIS shape: every one of the
    # entity's facts sits in one category that is itself above the bound, so the
    # retry has nothing to scan and the entity-level refusal stands. `scan_scope`
    # says which scan answered — the entity-level one — and no category is
    # reported as scanned, so a wholly-refused entity cannot be read as a
    # partially-scanned one.
    assert plan["scan_scope"] == "entity", plan
    assert plan["categories_scanned"] == [], plan


def test_entity_over_bound_with_small_categories_is_scanned_by_category(world):
    """#1251 clause 1: an entity the entity-level scan refuses is not discarded
    when each of its category files sits under the bound.

    The live shape this replaces: `Assistant` is 55 facts across 5 files, largest
    26, and `plan_entity` returned it with zero actions because the refusal says
    "Pass a narrower `category`" and no caller ever had. Here 55 facts sit in 3
    files, and the pair to act on is inside one of them."""
    facts_root, st, _ = world
    _write_facts(facts_root, "MULTI", "state", _filler("MULTI", 26))
    _write_facts(facts_root, "MULTI", "event", _filler("MULTI", 23))
    _write_facts(facts_root, "MULTI", "preference", [
        {"fact": "MULTI cache eviction is enabled (multi.cache.evict).", "created_at": _days_ago(20)},
        {"fact": "MULTI cache eviction is disabled (multi.cache.evict).", "created_at": _days_ago(2)},
        *_filler("MULTI", 4),
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("MULTI")
    assert plan["refused"] is False, plan
    assert plan["scan_scope"] == "by_category", plan
    # `checked` covers exactly the facts in the scanned categories — all 55 here,
    # since no category of this entity is over the bound.
    assert plan["checked"] == 55, plan
    assert plan["categories_scanned"] == ["event", "preference", "state"], plan
    assert plan["categories_skipped"] == [], plan
    # And the recovered scan finds what the discarded one would have: the older
    # side of a pair the detector classified as opposing terms.
    assert plan["contradictions"] >= 1, plan
    assert len(plan["actions"]) == 1, plan
    act = plan["actions"][0]
    assert act["category"] == "preference", act
    assert act["kind"] == "superseded", act
    assert act["loser_fact"] == "MULTI cache eviction is enabled (multi.cache.evict).", act


def test_category_over_bound_is_skipped_and_named_in_the_plan(world):
    """#1251 clause 2: the retry still honours the bound per category, and the
    plan names what it did not scan.

    Without the name, a partial scan is indistinguishable from a full one: the
    pair count and `checked` are simply smaller, and the next reader has no way to
    tell "nothing opposed in the other 51 facts" from "those 51 were never read".
    """
    facts_root, st, _ = world
    from agent_mcp.retrieval import FACT_GODNODE_THRESHOLD
    _write_facts(facts_root, "MIXED", "state",
                 _filler("MIXED", FACT_GODNODE_THRESHOLD + 1))       # 51: unscanable
    _write_facts(facts_root, "MIXED", "event", [
        {"fact": "MIXED tailnet relay is working (mix.tailnet.relay).", "created_at": _days_ago(18)},
        {"fact": "MIXED tailnet relay is broken (mix.tailnet.relay).", "created_at": _days_ago(3)},
        *_filler("MIXED", 8),
    ])
    _write_facts(facts_root, "MIXED", "preference", _filler("MIXED", 5))
    _reindex(st, facts_root)
    plan = fi.plan_entity("MIXED")
    assert plan["refused"] is False, plan                 # partial coverage, not a refusal
    assert plan["scan_scope"] == "by_category", plan
    assert plan["categories_skipped"] == ["state"], plan
    assert plan["categories_scanned"] == ["event", "preference"], plan
    assert plan["checked"] == 15, plan                    # 10 + 5; the 51 never read
    assert len(plan["actions"]) == 1, plan
    assert plan["actions"][0]["loser_fact"] == "MIXED tailnet relay is working (mix.tailnet.relay).", plan

    # Distinguishability, asserted on the record FILE and with all three shapes in
    # one run side by side — that is what clause 2 actually claims, and it is the
    # artifact the nightly refusal count is read from (`per_entity[]` in
    # `_pipeline/improvement/*-dryrun.json`). Asserted on disk rather than on the
    # returned dict because a scope that survived only in memory would be exactly
    # as invisible there as the discarded refusal this change replaced.
    #
    # GODNODE: every fact in one 51-fact category — nothing scanable, so the
    # whole-entity refusal stands. ALLSMALL: over the bound in total, no category
    # over it, so the retry covers everything. MIXED: covered above.
    _write_facts(facts_root, "GODNODE", "state",
                 _filler("GODNODE", FACT_GODNODE_THRESHOLD + 1))
    _write_facts(facts_root, "ALLSMALL", "state", _filler("ALLSMALL", 26))
    _write_facts(facts_root, "ALLSMALL", "event", _filler("ALLSMALL", 23))
    _write_facts(facts_root, "ALLSMALL", "preference", _filler("ALLSMALL", 4))
    _reindex(st, facts_root)
    rec = fi.run_improvement(entities=["MIXED", "GODNODE", "ALLSMALL"])
    entries = {e["entity"]: e for e in _persisted_record(rec)["per_entity"]}
    # (refused, scan_scope, categories_skipped) — the triple the record exposes.
    triples = {name: (e["refused"], e["scan_scope"], tuple(e["categories_skipped"]))
               for name, e in entries.items()}
    assert triples["GODNODE"] == (True, "entity", ()), triples
    assert triples["MIXED"] == (False, "by_category", ("state",)), triples
    assert triples["ALLSMALL"] == (False, "by_category", ()), triples
    # Three distinct rows: a partial scan reads as neither a full scan nor a
    # refusal, and no two of the three collapse onto the same triple.
    assert len(set(triples.values())) == 3, triples


def test_entity_under_bound_is_still_scanned_whole_across_categories(world):
    """#1251 clause 4: the partition is a refusal-path fallback, not the scan.

    An entity at or under the bound keeps its cross-category pairs, which a
    category-partitioned scan structurally cannot find. Pair split across two
    categories: if the retry ran here the plan would come back empty, so the
    action below is the assertion that the entity-level scan answered."""
    facts_root, st, _ = world
    _write_facts(facts_root, "SPLIT", "state", [
        {"fact": "SPLIT ingestion watcher is enabled (split.ingest.watch).", "created_at": _days_ago(21)},
        *_filler("SPLIT", 19),
    ])
    _write_facts(facts_root, "SPLIT", "event", [
        {"fact": "SPLIT ingestion watcher is disabled (split.ingest.watch).", "created_at": _days_ago(4)},
        *_filler("SPLIT", 19),
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("SPLIT")
    assert plan["refused"] is False, plan
    assert plan["scan_scope"] == "entity", plan
    assert plan["checked"] == 40, plan                    # both categories, one scan
    assert len(plan["actions"]) == 1, plan                # the cross-category pair
    assert plan["actions"][0]["loser_fact"] == "SPLIT ingestion watcher is enabled (split.ingest.watch).", plan


def test_run_is_dry_run_by_default_and_acts_on_request(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK (tts.builtin).",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors (tts.builtin).",
         "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    _reindex(st, facts_root)

    dry = fi.run_improvement(sources=("drift",), days=3)
    assert dry["apply"] is False
    assert dry["actions_planned"] == 1 and dry["actions_taken"] == 0
    assert _active(st) == 2, "dry run must not touch a single fact"
    text = (facts_root / "TTS" / "TTS-state.md").read_text(encoding="utf-8")
    assert "expired_at: null" in text
    # A plan-mode pass that reports only a count is not a plan: the list of
    # what it would do, with the reason, is the whole point of running it.
    listed = dry["per_entity"][0]["actions"]
    assert len(listed) == 1 and listed[0]["planned"] is True and listed[0]["applied"] is False
    assert listed[0]["loser_fact"].startswith("TTS built-in voices are working")
    assert "created" in listed[0]["reason"]

    wet = fi.run_improvement(apply=True, sources=("drift",), days=3)
    assert wet["apply"] is True
    assert wet["actions_taken"] == 1
    assert wet["before_active"] == 2 and wet["after_active"] == 1


def test_run_record_carries_before_after_counts_and_is_persisted(world):
    facts_root, st, tmp_vault = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK (tts.builtin).",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors (tts.builtin).",
         "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    rec = fi.run_improvement(apply=True, sources=("drift",), days=3)
    for key in ("before_active", "after_active", "delta_active", "signals",
                "actions_planned", "actions_taken", "per_entity", "fact_entity_recall"):
        assert key in rec, key
    assert rec["delta_active"] == rec["after_active"] - rec["before_active"] == -1
    written = list((fi.RECORD_DIR).glob("*.json"))
    assert written, "the run must be recorded, not just returned"
    on_disk = json.loads(written[0].read_text(encoding="utf-8"))
    assert on_disk["before_active"] == 2 and on_disk["after_active"] == 1


# ── 2b. #700: a run record names the tree, the store and the commit ──────────
#
# Five `--apply` records from 2026-09-09 report 32 fact expirations that never
# reached the live knowledge graph: all 32 condemned facts are still active and
# `facts_idx` holds no row stamped 09-09. Nothing in those records said which
# fact tree or which sqlite file the run had acted on, and because `RECORD_DIR`
# is code-relative while `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB` are env-overridable, a
# run aimed at a copy lands in the same directory as a real one. A deletion
# loop's audit trail must not be able to read as a deletion that did not
# happen, so every record now describes itself. The four keys below are asserted
# on the FILE, because the file is the artifact a reader actually opens.

def _persisted_record(rec: dict) -> dict:
    """The JSON this run wrote under RECORD_DIR, not the returned dict."""
    path = Path(rec["record_path"])
    assert path.parent == Path(fi.RECORD_DIR), "the record must live under RECORD_DIR"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("apply", [False, True])
def test_record_names_the_fact_tree_store_and_commit_it_acted_on(world, apply):
    """The record is the only evidence of a deletion, so it has to name what it
    deleted from — dry-run and apply alike."""
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK (tts.builtin).",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors (tts.builtin).",
         "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    rec = fi.run_improvement(apply=apply, sources=("drift",), days=3)
    on_disk = _persisted_record(rec)
    assert on_disk["apply"] is apply
    # The tmp tree and the tmp store, never the live locations.
    assert on_disk["facts_root"] == str(facts_root)
    assert on_disk["kg_db"] == str(st.path)
    assert on_disk["facts_root"] != str(app_paths.VAULT_FACTS_ROOT_DEFAULT)
    assert on_disk["kg_db"] != str(app_paths.VAULT_KG_DB_DEFAULT)
    assert re.fullmatch(r"[0-9a-f]{40}", on_disk["git_head"]), on_disk["git_head"]
    assert on_disk["isolated"] is True, "a redirected run must say so"


def test_record_stamps_git_head_unknown_when_git_cannot_answer(world, monkeypatch):
    """`app.gitinfo.head_commit` returns None rather than raising; the key may
    never be absent, or a reader cannot tell 'no commit' from 'no such field'."""
    monkeypatch.setattr(fi, "_head_commit", lambda root: None)
    rec = fi.run_improvement(sources=("drift",), days=3)
    assert _persisted_record(rec)["git_head"] == "unknown"


def test_a_run_on_the_production_locations_is_not_marked_isolated(world, monkeypatch,
                                                                  tmp_path):
    """The stamp has to be able to say False, or every future record reads as a
    verification pass and the flag says nothing. Stands the production pair in
    for a tmp pair: what binds here is that the resolved tree and store equal
    the defaults the record compares against."""
    prod_facts = tmp_path / "prod-facts"
    prod_facts.mkdir()
    prod_db = tmp_path / "prod-kg.sqlite"
    monkeypatch.setattr(fi._paths, "VAULT_FACTS_ROOT_DEFAULT", prod_facts)
    monkeypatch.setattr(fi._paths, "VAULT_KG_DB_DEFAULT", prod_db)
    monkeypatch.setattr(fi, "FACTS_ROOT", prod_facts)
    kg_store.configure(prod_db)
    rec = fi.run_improvement(sources=("drift",), days=3)
    on_disk = _persisted_record(rec)
    assert on_disk["facts_root"] == str(prod_facts)
    assert on_disk["kg_db"] == str(prod_db)
    assert on_disk["isolated"] is False


def test_an_environment_redirect_still_reads_as_isolated(world, monkeypatch):
    """`LLOYD_FACTS_ROOT`/`LLOYD_KG_DB` move `app.paths.VAULT_FACTS_ROOT` and
    `VAULT_KG_DB` along with them, so comparing a run against THOSE constants is
    exactly how a copy certifies itself as production. The record compares
    against the built-in defaults, which the env cannot move."""
    facts_root, st, _ = world
    monkeypatch.setattr(fi._paths, "VAULT_FACTS_ROOT", facts_root)
    monkeypatch.setattr(fi._paths, "VAULT_KG_DB", st.path)
    rec = fi.run_improvement(sources=("drift",), days=3)
    assert _persisted_record(rec)["isolated"] is True


def test_the_default_paths_cannot_be_moved_by_the_environment(tmp_path):
    """The comparison target has to be env-immune or the flag is self-declared.
    Reloaded under an override: the overridable paths move, the defaults do
    not."""
    import importlib
    import os

    from app import paths as paths_mod
    expected = paths_mod.DATA_ROOT / "_pipeline" / "vault-derived"
    saved = {k: os.environ.get(k) for k in ("LLOYD_FACTS_ROOT", "LLOYD_KG_DB")}
    try:
        os.environ["LLOYD_FACTS_ROOT"] = str(tmp_path / "copy-facts")
        os.environ["LLOYD_KG_DB"] = str(tmp_path / "copy-kg.sqlite")
        moved = importlib.reload(paths_mod)
        assert moved.VAULT_FACTS_ROOT == tmp_path / "copy-facts"
        assert moved.VAULT_KG_DB == tmp_path / "copy-kg.sqlite"
        assert moved.VAULT_FACTS_ROOT_DEFAULT == expected / "facts"
        assert moved.VAULT_KG_DB_DEFAULT == expected / "kg.sqlite"
        assert moved.VAULT_FACTS_ROOT != moved.VAULT_FACTS_ROOT_DEFAULT
        assert moved.VAULT_KG_DB != moved.VAULT_KG_DB_DEFAULT
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(paths_mod)


def test_run_reports_the_metric_it_claims_to_move(world, monkeypatch):
    """The acceptance bar for #376: a change that cannot move
    `fact_entity_recall` is not this feature — so the run carries the number."""
    facts_root, st, _ = world
    seen = {}
    detail = {"score": 0.5, "queries": 2, "queries_answered": 1, "errors": 0,
              "per_query": [{"id": "q1", "fact_entity_recall": 0.5,
                             "fact_entities_matched": ["A"], "expected_entities": ["A", "B"],
                             "error": None},
                            {"id": "q2", "fact_entity_recall": None,
                             "fact_entities_matched": [], "expected_entities": [],
                             "error": None}]}

    def _fake_report(limit):
        seen["limit"] = limit
        return detail

    monkeypatch.setattr(fi, "_fact_entity_recall_detail", _fake_report)
    rec = fi.run_improvement(sources=("drift",), days=3, report_eval=True, eval_limit=7)
    assert rec["fact_entity_recall"] == 0.5
    assert seen["limit"] == 7
    # #702 clause 5: the evidence rides beside the score — how many queries
    # answered and the raw per-query hits — and it reaches the file.
    saved = _persisted_record(rec)
    assert saved["fact_entity_recall"] == 0.5
    assert saved["fact_entity_recall_detail"]["queries_answered"] == 1
    assert saved["fact_entity_recall_detail"]["queries"] == 2
    assert [q["fact_entity_recall"] for q in saved["fact_entity_recall_detail"]["per_query"]] == [0.5, None]
    skipped = fi.run_improvement(sources=("drift",), days=3)
    assert skipped.get("fact_entity_recall") is None, "off by default"
    assert skipped.get("fact_entity_recall_detail") is None, "null is the only 'did not run'"


def test_recall_detail_keeps_the_per_query_evidence_run_eval_returned():
    """#702 clause 5, on `run_eval`'s own row shape: the count of queries that
    answered excludes a query with no expected entities (its per-query value is
    None, which is not-run rather than zero), and every row's hits survive."""
    records = [
        {"id": "q1", "scoring": {"fact_entity_recall": 1.0, "fact_entities_matched": ["vllm"]},
         "expected": {"entities": ["vllm"]}, "error": None},
        {"id": "q2", "scoring": {"fact_entity_recall": 0.0, "fact_entities_matched": []},
         "expected": {"entities": ["qmd", "djev"]}, "error": None},
        {"id": "q3", "scoring": {"fact_entity_recall": None, "fact_entities_matched": []},
         "expected": {"entities": []}, "error": None},
    ]
    summary = {"overall": {"fact_entity_recall_avg": 0.5, "errors": 0}}
    detail = fi._recall_detail(records, summary)
    assert detail["score"] == 0.5 and detail["queries"] == 3
    assert detail["queries_answered"] == 2, detail
    assert [q["fact_entity_recall"] for q in detail["per_query"]] == [1.0, 0.0, None]
    assert detail["per_query"][0]["fact_entities_matched"] == ["vllm"]
    assert detail["per_query"][1]["expected_entities"] == ["qmd", "djev"]


def test_a_refused_entity_is_not_recorded_as_scanned_and_clean(world):
    """#702 clause 2: the record entry for a refused god-node carries the
    detector's reason and the facts it holds, and null pair counts — a scanned
    entity with nothing found carries 0 and no reason. The two shapes must not
    collapse, on the file the nightly refusal count is read from."""
    facts_root, st, _ = world
    from agent_mcp.retrieval import FACT_GODNODE_THRESHOLD
    _write_facts(facts_root, "GODNODE", "state",
                 _filler("GODNODE", FACT_GODNODE_THRESHOLD + 1))
    _write_facts(facts_root, "CLEAN", "state", _filler("CLEAN", 3))
    _reindex(st, facts_root)
    rec = fi.run_improvement(entities=["GODNODE", "CLEAN"])
    entries = {e["entity"]: e for e in _persisted_record(rec)["per_entity"]}
    god, clean = entries["GODNODE"], entries["CLEAN"]
    assert god["refused"] is True and god["contradictions"] is None and god["near_duplicates"] is None
    assert god["checked"] == FACT_GODNODE_THRESHOLD + 1 == god["before_active"], god
    assert god["skipped_reason"] and "category" in god["skipped_reason"], god
    assert clean["refused"] is False and clean["contradictions"] == 0
    assert clean["checked"] == 3 and clean["skipped_reason"] is None, clean


def test_cli_summary_counts_and_names_the_refusals(world, capsys, monkeypatch):
    """#702 clause 3: the stdout line is what the nightly run quotes, and
    `entities scanned=2` over a refused god-node was a coverage claim about
    facts nobody compared. The refusal comes off the count and is named."""
    facts_root, st, _ = world
    from agent_mcp.retrieval import FACT_GODNODE_THRESHOLD
    _write_facts(facts_root, "GODNODE", "state",
                 _filler("GODNODE", FACT_GODNODE_THRESHOLD + 1))
    _write_facts(facts_root, "CLEAN", "state", _filler("CLEAN", 3))
    _reindex(st, facts_root)
    spec = importlib.util.spec_from_file_location("fact_improvement_cli_refused", _SCRIPT_PATH)
    cli = importlib.util.module_from_spec(spec)
    sys.modules["fact_improvement_cli_refused"] = cli
    spec.loader.exec_module(cli)

    monkeypatch.setattr(sys, "argv", ["fact-improvement.py", "--entity", "GODNODE", "--entity", "CLEAN"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "entities scanned=1 refused=1 (GODNODE)" in out, out
    assert "entities scanned=2" not in out, out


def test_run_with_no_signals_changes_nothing(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "OLD", "state",
                 [{"fact": "OLD is disabled.", "created_at": _days_ago(90)}])
    _reindex(st, facts_root)
    import os
    old = datetime.now() - timedelta(days=40)
    target = facts_root / "OLD" / "OLD-state.md"
    os.utime(target, (old.timestamp(), old.timestamp()))
    os.utime(facts_root / "OLD", (old.timestamp(), old.timestamp()))
    rec = fi.run_improvement(apply=True, sources=("drift", "corrections"), days=2)
    assert rec["signals"] == 0 and rec["entities"] == []
    assert rec["actions_taken"] == 0
    assert _active(st) == 1


# ── #2078: a keyword-opposition match is a screen, not an authority ──────────
#
# #701's owed ruling: `opposing_terms:<a>/<b>` is a required screen and never
# the sole authority to expire a fact. `created_at` order says only which row
# was written last, so on the equal-confidence branch the whole basis was the
# detector reading its own keyword match twice. These nodes are graded against
# the two actions that guard planned after #701 landed, as the corpus wrote
# them: `_pipeline/improvement/20260930-210019-dryrun.json` and
# `20261001-210151-dryrun.json`, whose rows sit on disk under
# `_pipeline/vault-derived/facts/TTS/TTS-event.md` and `.../Kit/`.

#: The TTS pair, both rows at the 0.95 the live rows carry (equal, so the pair
#: reaches the age branch and never #701's floor), 3.4 days apart as recorded.
#: A success-path claim and a failure-path claim about one subsystem: the older
#: row names no identifier at all, so nothing here is about the same named
#: thing as the newer one.
_LIVE_TTS_OLDER = "TTS success JSON response reads were bounded in commit #96984."
_LIVE_TTS_NEWER = ("TTS failure during an outage results in a connection-refused "
                   "error and leaves only a `voice.log` line.")

#: The Kit pair, 0.95 on both live rows, 1.7 days apart. Two unrelated
#: subsystems: each names its own identifier-shaped predicate and neither names
#: the other's. The older row is also the reason the basis has to be a property
#: of the PAIR — it is itself a sentence saying something was "flipped ... from
#: false to true", so a basis keyed on one side's wording ("an explicit
#: supersession statement") would let it authorise its own expiry.
_LIVE_KIT_OLDER = ("Kit commit 9e5dd32c (OMPE-65433) flipped "
                   "`/rtx/dldenoiser/responsiveDenoising` from false to true.")
_LIVE_KIT_NEWER = ("The sim.has_gui property is always False in Isaac Lab "
                   "3.0.0-beta2.patch1, leading to a missing IsaacLab GUI tab.")


def test_a_kit_flag_and_an_isaac_lab_property_are_not_one_predicate():
    """The basis unit on its own, with no tree: what counts as "the same named
    thing", and what does not, at the level of the two live witnesses."""
    from agent_mcp.fact_improvement import _predicate_tokens, _shared_predicate

    # The two false witnesses: co-naming a subsystem in prose is not a
    # predicate, and each side naming its OWN flag is not co-naming either.
    assert _shared_predicate(_LIVE_TTS_OLDER, _LIVE_TTS_NEWER) is None
    assert _shared_predicate(_LIVE_KIT_OLDER, _LIVE_KIT_NEWER) is None
    # The witness that is a real supersession: the same flag, different value.
    assert _shared_predicate(
        "/rtx/dldenoiser/responsiveDenoising is false in the shipping build.",
        "/rtx/dldenoiser/responsiveDenoising is true in the current build."
    ) == "rtx/dldenoiser/responsivedenoising"
    # A dotted identifier is a predicate; a hyphenated English compound is not,
    # because `connection-refused` and `read-only` are how prose is written and
    # counting them is the "shared word is too loose" failure — both live TTS
    # witnesses say "TTS".
    assert _predicate_tokens("The read-only view of sim.has_gui is cached.") == {
        "sim.has_gui"}
    # A citation is a shared source, which is #1941's ONE reason, never a
    # predicate either of the facts is about.
    assert _predicate_tokens(
        "See https://github.com/isaac-sim/IsaacLab/pull/5941.") == set()


# ── #2199: the entity's own identifier is not a shared predicate ─────────────
#
# #2078 requires one identifier-shaped token BOTH facts name before an equal-
# confidence pair may be expired as `superseded`. Measured on the live corpus,
# that bar was free to clear for an entity written by a dotted, hyphenated id:
# `_PREDICATE_TOKEN_RE` cuts `anthropic.claude-code` at the hyphen and returns
# `anthropic.claude`, a FRAGMENT of the entity's own identifier, so every fact
# about that extension co-named every other one. The witness then named the
# alphabetically-first shared token, which is how the vacuous one displaced the
# real field `metadata.pinned` in the `reason` a reviewer reads
# (`backlog/data/20261004-210010-dryrun.json`, 1052 lines).
#
# The fix is name-based, not length-based: `_MIN_PREDICATE_TOKEN_LEN` stays 4,
# because `metadata.pinned` (15) and `anthropic.claude` (16) both clear it today.

#: The item's probe: two facts whose only co-naming is the entity's own dotted,
#: hyphenated identifier. Before #2199 this pair shared `anthropic.claude`.
_PROBE_OLDER = "anthropic.claude-code ships a TUI affordance"
_PROBE_NEWER = "anthropic.claude-code supports a planning mode"


def test_an_identifier_cut_at_its_hyphen_is_not_a_shared_predicate():
    """Clause 1: the item's probe returns None, and the reason is the cut.

    The guard is the truncation, not the dot: a dotted identifier the text
    actually writes that way is still a predicate, and a real predicate in the
    same sentence as a cut identifier survives with it."""
    from agent_mcp.fact_improvement import (
        _MIN_PREDICATE_TOKEN_LEN, _predicate_tokens, _shared_predicate)

    assert _shared_predicate(_PROBE_OLDER, _PROBE_NEWER) is None
    assert _predicate_tokens(_PROBE_OLDER) == set(), (
        "the extension id is still yielding its pre-hyphen fragment")
    # Not a length rule: `metadata.pinned` is longer than the fragment, so a
    # raised floor would have removed neither. The floor is untouched.
    assert _MIN_PREDICATE_TOKEN_LEN == 4
    assert "metadata.pinned" in _predicate_tokens("metadata.pinned was set")
    # A whole token beside a cut one is kept — the cut, not the dot, is excluded.
    assert _predicate_tokens("sim.has_gui is set by anthropic.claude-code only") == {
        "sim.has_gui"}
    # And an identifier the corpus really does write this way is still a token.
    assert _predicate_tokens("The split is documented in agent_mcp/facts.py.") == {
        "agent_mcp/facts.py"}


def test_the_pair_s_own_entity_name_and_its_aliases_are_not_predicates():
    """Clause 2: the exclusion narrows the basis without silencing it.

    A token that IS the subject — the entity's own name, a prefix of it, or a
    known alias — answers a different question than #2078 asked: this scan is
    already entity-scoped, so every pair in the entity's file co-names it. What
    the bar is for, a predicate both facts name ABOUT that entity, still counts,
    and so does a token LONGER than the name, which is a field of the subject."""
    from agent_mcp.fact_improvement import _shared_predicate

    # The entity's own dotted name, written whole in both rows.
    assert _shared_predicate("claude.code skipped the 09-30 sweep",
                             "claude.code installed nothing on 10-01",
                             entity="claude.code") is None
    # A PREFIX of it: the entity is `claude.code.extensions`, the rows co-name
    # the shorter spelling. Equality alone would have let this through.
    assert _shared_predicate("claude.code is pinned", "claude.code is unpinned",
                             entity="claude.code.extensions") is None
    # A KNOWN ALIAS the caller names, over an entity whose display name is prose.
    assert _shared_predicate("claude.code is pinned", "claude.code is unpinned",
                             entity="Claude Code", aliases=["claude.code"]) is None
    # A genuine predicate about that same entity still stands …
    assert _shared_predicate("claude.code reads metadata.pinned as true",
                             "claude.code reads metadata.pinned as false",
                             entity="Claude Code",
                             aliases=["claude.code"]) == "metadata.pinned"
    # … and so does a deeper field: longer than the name, so not the name.
    assert _shared_predicate("claude.code.enabled is true",
                             "claude.code.enabled is false",
                             entity="claude.code") == "claude.code.enabled"


def test_the_alias_look_up_in_the_store_and_a_missing_store_is_not_a_crash(world,
                                                                           monkeypatch):
    """The alias half crosses into `app.kg_store`, so it is tested across that.

    `aliases=` is the caller's own knowledge; the store's alias table is what
    says two spellings are one entity for a caller that did not pass anything.
    And `StoreUnavailable` is a real state on this box (#1236): the guard loses
    its alias half there and keeps the half read off the entity name — a missing
    store must not raise inside a nightly scan, and must not silently disable
    the exclusion the caller handed it either.
    """
    facts_root, st, _ = world
    st.aliases.set("claude.code", "Claude Code", kind="punct", origin="test")
    st.aliases.set("claude.code.plugin", "Claude Code", kind="semantic", origin="test")
    assert {r["surface"] for r in st.aliases.for_canonical("Claude Code")} == {
        "claude.code", "claude.code.plugin"}, "the alias rows did not land"
    # Both rows name the alias AND a real field: only the field may count.
    assert fi._shared_predicate("claude.code.plugin reads metadata.pinned",
                                "metadata.pinned says claude.code.plugin is set",
                                entity="Claude Code") == "metadata.pinned"
    # Both rows name ONLY the alias: nothing is predicated in common.
    assert fi._shared_predicate("claude.code.plugin skipped the sweep",
                                "claude.code.plugin installed nothing",
                                entity="Claude Code") is None

    monkeypatch.setattr(fi, "_store",
                        lambda: (_ for _ in ()).throw(fi._StoreUnavailable("no db")))
    # With no store that alias is unreadable, so the pair is admitted on it again
    # — the loss is real and bounded to the alias half.
    assert fi._shared_predicate("claude.code.plugin is pinned",
                                "claude.code.plugin is unpinned",
                                entity="Claude Code") == "claude.code.plugin"
    # A caller that handed its own alias in is not affected by the missing store.
    assert fi._shared_predicate("claude.code.plugin is pinned",
                                "claude.code.plugin is unpinned",
                                entity="Claude Code",
                                aliases=["claude.code.plugin"]) is None
    # The name half needed no store: `claude.code` and `Claude Code` are one name
    # written two ways, and `_name_form` says so without asking the database.
    assert fi._shared_predicate("claude.code is pinned", "claude.code is unpinned",
                                entity="Claude Code") is None


def test_the_witness_names_every_shared_token_not_the_alphabetically_first():
    """Clause 3: `sorted(shared)[0]` reported the weak token and hid the real
    field, so a reviewer reading `reason` could not tell a vacuous co-naming
    from a genuine one — #701's "name what admitted it" implemented as "name one
    thing that admitted it", with the alphabet choosing."""
    from agent_mcp.fact_improvement import _shared_predicate

    older = "The pin on claude.code held: metadata.pinned was true in extensions.json."
    newer = "The sweep skipped claude.code because metadata.pinned was set in extensions.json."
    witness = _shared_predicate(older, newer, entity="Claude Code",
                                aliases=["claude.code"])
    assert witness == "extensions.json, metadata.pinned", witness
    assert "metadata.pinned" in witness, "the real field was displaced again"
    assert "claude.code" not in witness, "the subject is still in its own witness"
    # One token, one name: the single-token string is that token, unchanged.
    assert _shared_predicate("metadata.pinned is true", "metadata.pinned is false",
                             entity="Claude Code") == "metadata.pinned"


def test_planning_and_the_write_seam_refuse_the_entity_name_pair_alike(world):
    """Clause 4: one pair, one verdict, at both seams.

    Before the entity was threaded through, the two could disagree by
    construction: `_shared_predicate(t1, t2)` took only the two texts, so the
    write seam in `apply_action` asked a strictly weaker question than the
    planner. A pair whose only co-naming is the entity's own alias is keyword-
    only at plan time and refused at write time, and the alias it declines on
    comes from the store both seams read."""
    facts_root, st, _ = world
    st.aliases.set("claude.code", "Claude Code", kind="punct", origin="test")
    _write_facts(facts_root, "Claude Code", "update", [
        {"fact": "The claude.code auto-update gate is disabled.",
         "created_at": _days_ago(30), "confidence": 0.95},
        {"fact": "The claude.code auto-update gate is enabled.",
         "created_at": _days_ago(0), "confidence": 0.95},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Claude Code")
    assert plan["pairs_before"] == 1, plan
    assert plan["actions"] == [], plan
    assert len(plan["keyword_only_flags"]) == 1, plan
    assert plan["keyword_only_flags"][0]["trigger"] == "opposing_terms:enabled/disabled"
    hand_built = {
        "entity": "Claude Code", "category": "update", "kind": "superseded",
        "loser_fact": "The claude.code auto-update gate is disabled.",
        "loser_id": "upda-001", "loser_confidence": 0.95, "winner_confidence": 0.95,
        "winner_fact": "The claude.code auto-update gate is enabled.",
        "winner_id": "upda-002",
        "loser_source_file": "Claude Code/Claude Code-update.md",
        "winner_source_file": "Claude Code/Claude Code-update.md",
        "reason": "opposing_terms:enabled/disabled; written 30.0 days later",
    }
    result = fi.apply_action(hand_built, _days_ago(0))
    assert result["expired_count"] == 0, result
    assert "keyword opposition" in result["skipped"], result
    assert _active(st, "Claude Code") == 2, "the write seam expired a keyword-only pair"


def test_a_genuine_predicate_admits_the_pair_at_both_seams(world):
    """The other direction of clause 4, so the agreement above is not two refusals:
    a pair that names a real predicate about the entity is planned AND written,
    and the `reason` names every token that survives the exclusion."""
    facts_root, st, _ = world
    st.aliases.set("claude.code", "Claude Code", kind="punct", origin="test")
    _write_facts(facts_root, "Claude Code", "update", [
        {"fact": "The claude.code auto_update.gate is disabled in extensions.json.",
         "created_at": _days_ago(30), "confidence": 0.95},
        {"fact": "The claude.code auto_update.gate is enabled in extensions.json.",
         "created_at": _days_ago(0), "confidence": 0.95},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Claude Code")
    assert plan["keyword_only_flags"] == [], plan
    assert len(plan["actions"]) == 1, plan
    action = plan["actions"][0]
    assert action["kind"] == "superseded", action
    reason = action["reason"]
    assert ("both facts name `auto_update.gate`, `extensions.json`, so the later "
            "write is about 2 predicates both facts name") in reason, reason
    assert "`claude.code`" not in reason, reason
    result = fi.apply_action(action, _days_ago(0))
    assert result["expired_count"] == 1, result
    assert _active(st, "Claude Code") == 1, result


def test_the_live_tts_pair_plans_nothing_and_is_flagged(world):
    """Clause 1 and 2, on the witness the record calls
    `opposing_terms:success/failure`, written 3.4 days later."""
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "event", [
        {"fact": _LIVE_TTS_OLDER, "created_at": _days_ago(3.4), "confidence": 0.95},
        {"fact": _LIVE_TTS_NEWER, "created_at": _days_ago(0), "confidence": 0.95},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("TTS")
    # The pair is still a contradiction — the screen found something, and that
    # finding is what is worth reporting. What it is no longer worth is a write.
    assert plan["pairs_before"] == 1, plan
    assert plan["contradictions"] == 1, plan
    assert plan["actions"] == [], plan
    assert len(plan["keyword_only_flags"]) == 1, plan
    flag = plan["keyword_only_flags"][0]
    assert flag["entity"] == "TTS", flag
    assert flag["trigger"] == "opposing_terms:success/failure", flag
    assert flag["older_fact"] == _LIVE_TTS_OLDER, flag
    assert flag["newer_fact"] == _LIVE_TTS_NEWER, flag
    assert _active(st, "TTS") == 2, "a flagged pair must not have been touched"


def test_the_live_kit_pair_against_an_isaac_lab_property_plans_nothing(world):
    """Clause 2, on the witness that paired two unrelated subsystems:
    `opposing_terms:true/false`, the Kit flag against Isaac Lab's `sim.has_gui`,
    written 1.7 days later. Each fact names an identifier; neither names the
    other's, which is what makes the pairing false and what the co-naming test
    is there to catch."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Kit", "event", [
        {"fact": _LIVE_KIT_OLDER, "created_at": _days_ago(1.7), "confidence": 0.95},
        {"fact": _LIVE_KIT_NEWER, "created_at": _days_ago(0), "confidence": 0.95},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Kit")
    assert plan["pairs_before"] == 1, plan
    assert plan["actions"] == [], plan
    assert len(plan["keyword_only_flags"]) == 1, plan
    flag = plan["keyword_only_flags"][0]
    assert flag["trigger"] == "opposing_terms:true/false", flag
    assert flag["older_fact"] == _LIVE_KIT_OLDER, flag
    assert flag["newer_fact"] == _LIVE_KIT_NEWER, flag
    assert _active(st, "Kit") == 2


def test_a_later_fact_naming_the_same_predicate_still_supersedes(world):
    """Clause 3: the age path narrowed, not deleted. Two rows on ONE flag, the
    later one 3 days newer, both at 0.95 — a real supersession of exactly the
    shape the live Kit witness has, minus the second subsystem. The write seam is
    in here too: `apply_action` may not refuse what the planner just planned."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Kit", "state", [
        {"fact": "/rtx/dldenoiser/responsiveDenoising is false in the shipping build.",
         "created_at": _days_ago(4), "confidence": 0.95},
        {"fact": "/rtx/dldenoiser/responsiveDenoising is true in the current build.",
         "created_at": _days_ago(1), "confidence": 0.95},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Kit")
    assert plan["keyword_only_flags"] == [], plan
    assert len(plan["actions"]) == 1, plan
    action = plan["actions"][0]
    assert action["kind"] == "superseded", action
    assert action["loser_fact"].startswith("/rtx/dldenoiser/responsiveDenoising is false")
    assert action["winner_fact"].startswith("/rtx/dldenoiser/responsiveDenoising is true")
    # The basis is in the reason beside the trigger (#701's rule that an
    # admitted action names what admitted it, now covering the non-lexical half
    # of the basis too).
    assert "rtx/dldenoiser/responsiveDenoising" in action["reason"], action
    rec = fi.run_improvement(apply=True, entities=["Kit"], record=False)
    assert rec["actions_taken"] == 1, rec["per_entity"]
    assert _active(st, "Kit") == 1, "the same-predicate supersession did not apply"


def test_the_run_record_reports_the_flagged_pairs_beside_the_pair_count(world):
    """Clause 1's reporting half: the withheld class is a number the record can
    be asked for, in the same field group as the denominator it is a subset of.
    Without it `actions_planned: 0` cannot tell a clean tree from a tree whose
    every pair rested on keyword-plus-write-order alone — the #1348 and #1941
    lesson, one refusal later."""
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "event", [
        {"fact": _LIVE_TTS_OLDER, "created_at": _days_ago(3.4), "confidence": 0.95},
        {"fact": _LIVE_TTS_NEWER, "created_at": _days_ago(0), "confidence": 0.95},
    ])
    _reindex(st, facts_root)
    rec = fi.run_improvement(sources=("drift",), days=3)
    assert rec["actions_planned"] == 0, rec["per_entity"]
    assert rec["pairs_before"] == 1, rec["pairs_before"]
    assert len(rec["keyword_only_flags"]) == 1, rec["keyword_only_flags"]
    assert rec["keyword_only_flags"][0]["older_fact"] == _LIVE_TTS_OLDER
    per_entity = rec["per_entity"][0]
    assert per_entity["entity"] == "TTS", per_entity
    assert per_entity["pairs_before"] == 1, per_entity
    assert [f["older_fact"] for f in per_entity["keyword_only_flags"]] == [_LIVE_TTS_OLDER]
    on_disk = _persisted_record(rec)
    assert len(on_disk["keyword_only_flags"]) == 1, on_disk["keyword_only_flags"]


def test_the_opposing_terms_screen_still_classifies_and_still_admits_confidence(world):
    """Clause 4: the narrowing is in the branch, not in the screen. The whole-word
    classifier still fires on `enabled` against `disabled`, and the pair it
    classifies still reaches the confidence basis at a gap of 0.3 — and the flag
    list stays empty for it, because a pair with a confidence basis was never
    asking the age question. `MIN_CONFIDENCE_GAP` is #701's, and the confidence
    nodes above and beside this one are that floor's own guards."""
    from agent_mcp import facts as facts_mod
    from agent_mcp.fact_improvement import REQUIRE_OPPOSING_TERMS
    assert REQUIRE_OPPOSING_TERMS is True
    assert facts_mod._opposing_reason(
        "The gate is enabled.", "The gate is disabled.") == "opposing_terms:enabled/disabled"
    assert facts_mod._opposing_reason(
        "A redeployed container.", "He re-enabled the sync.") is None
    facts_root, st, _ = world
    _write_facts(facts_root, "Gate", "state", [
        {"fact": "The gate is enabled.", "created_at": _days_ago(30), "confidence": 0.9},
        {"fact": "The gate is disabled.", "created_at": _days_ago(2), "confidence": 0.6},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Gate")
    assert len(plan["actions"]) == 1, plan
    assert plan["actions"][0]["kind"] == "confidence", plan
    assert plan["actions"][0]["loser_fact"] == "The gate is disabled.", plan
    assert plan["keyword_only_flags"] == [], plan


def test_an_assembled_superseded_action_with_no_shared_predicate_is_refused(world):
    """Clause 5's apply seam: the write path refuses a superseded action whose
    only basis is keyword opposition plus write order, even one that did not come
    from this planner — an action dict assembled by hand, or a record written by
    an older revision and applied later. A `skipped` refusal, not an error: the
    caller's loop reads a skip as "took none of these", and the #1255 invariant
    (a reported count equals the marks it wrote) is what that reading rests on."""
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "event", [
        {"fact": _LIVE_TTS_OLDER, "created_at": _days_ago(3.4), "confidence": 0.95},
        {"fact": _LIVE_TTS_NEWER, "created_at": _days_ago(0), "confidence": 0.95},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("TTS")
    assert plan["actions"] == [], plan
    hand_built = {
        "entity": "TTS", "category": "event",
        "kind": "superseded",
        "loser_fact": _LIVE_TTS_OLDER, "loser_id": "evnt-001",
        "loser_confidence": 0.95, "winner_confidence": 0.95,
        "winner_fact": _LIVE_TTS_NEWER, "winner_id": "evnt-002",
        "loser_source_file": "TTS/TTS-event.md", "winner_source_file": "TTS/TTS-event.md",
        "reason": "opposing_terms:success/failure; written 3.4 days later",
    }
    result = fi.apply_action(hand_built, _days_ago(0))
    assert result["expired_count"] == 0, result
    assert "keyword opposition" in result["skipped"], result
    assert result["traces_written"] == 0, result
    assert _active(st, "TTS") == 2, "the write seam expired a keyword-only pair"

# ── 2b. #701: what may authorise an expiry, and what a reviewer can see ──────
#
# `REQUIRE_OPPOSING_TERMS` is the only gate between a detected pair and a
# planned expiry, and until #701 the confidence basis beneath it was "the two
# numbers differ". The 2026-09-15 nightly board shows what that cost: of 215
# pairs scanned, the 2 that reached `opposing_terms` both became actions, both
# were false positives, and neither action's `reason` said which pair had fired
# — `plan_entity` dropped the detector's classification and rebuilt its own
# sentence. So now: an admitted action names its trigger, and a confidence
# difference smaller than MIN_CONFIDENCE_GAP condemns nothing.

def test_every_admitted_action_names_the_opposing_pair_that_admitted_it(world):
    """Both bases, one entity. The confidence action and the superseded action
    each open with the detector's own classification of its pair, so a reviewer
    reading a run record sees `opposing_terms:enabled/disabled` or
    `opposing_terms:working/broken` and can reject that pair — not merely the
    loser's text."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Trig", "state", [
        {"fact": "The gate is enabled.", "confidence": 0.9, "created_at": _days_ago(2)},
        {"fact": "The gate is disabled.", "confidence": 0.3, "created_at": _days_ago(30)},
    ])
    _write_facts(facts_root, "Trig", "usage", [
        {"fact": "The build is working (build.lloyd.ci).", "confidence": 0.9, "created_at": _days_ago(30)},
        {"fact": "The build is broken (build.lloyd.ci).", "confidence": 0.9, "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Trig")
    assert len(plan["actions"]) == 2, plan["actions"]
    by_kind = {a["kind"]: a for a in plan["actions"]}
    assert sorted(by_kind) == ["confidence", "superseded"], by_kind
    assert by_kind["confidence"]["reason"].startswith(
        "opposing_terms:enabled/disabled;"), by_kind["confidence"]["reason"]
    assert by_kind["confidence"]["loser_fact"] == "The gate is disabled."
    assert by_kind["superseded"]["reason"].startswith(
        "opposing_terms:working/broken;"), by_kind["superseded"]["reason"]


def test_a_confidence_gap_below_the_floor_is_reported_not_condemned(world):
    """The 2026-09-15 class, rebuilt with whole-word terms so that nothing but
    the gap can save it: 0.95 against 0.9 on a success-rate claim and a
    failure-cause claim — two compatible facts, one naming a rate, one naming a
    cause. MIN_CONFIDENCE_GAP is 0.1, so a 0.05 gap is not evidence and no
    action is planned."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Subgap", "state", [
        {"fact": "ALFWorld reached a 43% success rate on the tasks.",
         "confidence": 0.95, "created_at": _days_ago(20)},
        {"fact": "ALFWorld task failure is a rate the harness moves, not a "
                 "property of the model.", "confidence": 0.9, "created_at": _days_ago(10)},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Subgap")
    assert plan["contradictions"] == 1, plan["contradictions"]
    assert plan["actions"] == [], plan["actions"]
    # Declining for the floor is not the same as calling the pair a
    # near-duplicate. The detector classified it as an opposition; the figure
    # that reports that is the classification, not what this loop chose to do.
    assert plan["near_duplicates"] == 0, plan["near_duplicates"]
    assert _active(st) == 2


def test_the_confidence_floor_admits_a_gap_of_exactly_its_value(world):
    """The floor is `gap >= MIN_CONFIDENCE_GAP`, not `>`. The 2026-09-15
    working/broken hit had 0.9 against 1.0 — exactly the floor — so it stays
    admissible, and that is deliberate: whether it should be is the scope
    ruling on the item, which no threshold answers."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Floor", "state", [
        {"fact": "ALFWorld reached a 43% success rate on the tasks.",
         "confidence": 1.0, "created_at": _days_ago(20)},
        {"fact": "ALFWorld task failure is a rate the harness moves, not a "
                 "property of the model.", "confidence": 0.9, "created_at": _days_ago(10)},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Floor")
    assert len(plan["actions"]) == 1, plan["actions"]
    action = plan["actions"][0]
    assert action["kind"] == "confidence"
    assert action["loser_fact"].startswith("ALFWorld task failure"), action
    assert action["reason"].startswith("opposing_terms:success/failure;"), action["reason"]


def test_the_equal_confidence_age_basis_is_unmoved_by_the_floor(world):
    """MIN_CONFIDENCE_GAP bounds the CONFIDENCE basis only. An equal-confidence
    pair has a gap of 0.0 — below any floor — and its own basis is write order
    plus the predicate BOTH rows name, so a 28-day gap still plans a
    `superseded` expiry. #2078 narrowed that basis to pairs co-naming one
    identifier (`build.lloyd.ci` here) and did not move the floor: a pair of
    whole-word opposing terms still reaches this branch through
    `REQUIRE_OPPOSING_TERMS`, and only the basis is in question."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Ageway", "state", [
        {"fact": "The build is working (build.lloyd.ci).", "confidence": 0.9, "created_at": _days_ago(30)},
        {"fact": "The build is broken (build.lloyd.ci).", "confidence": 0.9, "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Ageway")
    assert len(plan["actions"]) == 1, plan["actions"]
    action = plan["actions"][0]
    assert action["kind"] == "superseded"
    assert action["loser_fact"] == "The build is working (build.lloyd.ci)."
    assert "created_at" in action["reason"], action["reason"]


# ── 2c. #1348: a bare confidence number may not beat an attributed fact ───────
#
# The 2026-09-21 nightly improve pass (`_pipeline/improvement/
# 20260921-210019-dryrun.json`, and reproducible live through
# `plan_entity('pass@k')` at this round's base) planned ONE action across 40
# entities and 207 contradiction pairs, and the action demoted the right row.
# On `pass@k` the detector paired — both rows read through `app.kg_store`, the
# only reader of the store:
#
#   stat-009  conf 0.9  created_at 2026-09-21T17:50:12.961951+00:00
#             source_doc knowledge/youtube/AI_Engineer/20260820-your-agent-
#                       evolved-your-evals-didnt-ameya-bhatawdekar-braintrust.md
#   fact-001  conf 1.0  created_at NULL  source_doc NULL
#
# `loser, winner = (f2, f1) if c1 > c2 else (f1, f2)` ranked them on the bare
# numbers, so the claim written that afternoon from a named talk note lost to a
# row with no date and no source document on `0.9 < 1.0`. Both rows read
# `provenance='EXTRACTED'`, so the provenance column cannot separate them; the
# two fields that can are `source_doc` and `created_at`, and this module had
# never consulted either — `grep -c source_doc agent_mcp/fact_improvement.py`
# returned 0 and `git log -S'source_doc'` on that file is empty, so the
# condition was never removed, it was never written.
#
# The guard is one-sided and has to stay that way. Measured through the store on
# 2026-09-21: 204,706 of 321,252 active rows (64%) carry neither field, and
# 162,175 of those sit at confidence >= 0.95. A rule barring every unattributed
# row from winning would suppress most of the actions the pass can take; what
# #1348 asks for is narrower — the winner may not be the row with no evidence
# while the loser has some.
#
# The two live claim strings are reused verbatim below, and the third test is
# the control for the first two: same entity, same texts, same confidences, both
# rows attributed → the action fires. So a 0-action result in the first two can
# only come from attribution, never from the pair failing to pair or from
# `MIN_CONFIDENCE_GAP` declining the 0.1 gap.

_PASSK_UNATTRIBUTED = ("The pass@k metric on a deterministic environment is "
                       "mathematically equivalent to the success rate of a replay agent.")
_PASSK_LOSER = ("A system can appear strong on pass@k but weak on pass^k, "
                "revealing variance as a failure mode.")
_PASSK_LOSER_SOURCE = ("knowledge/youtube/AI_Engineer/20260820-your-agent-evolved-"
                       "your-evals-didnt-ameya-bhatawdekar-braintrust.md")


def test_an_unattributed_winner_cannot_demote_an_attributed_loser(world):
    """Clause 1: the live shape, and it yields no action.

    The higher-confidence row carries neither `source_doc` nor `created_at`; the
    lower-confidence row carries `created_at` — either field alone is enough,
    which is what "carrying either" means. The pair is a contradiction the
    detector still reports (its count includes it, and the control test below
    proves the same pair is actable), it simply stops being actable: a demotion
    needs evidence on the winning side, and a bare `1.0` is not evidence.
    """
    facts_root, st, _ = world
    _write_facts(facts_root, "pass@k", "state", [
        {"fact": _PASSK_UNATTRIBUTED, "confidence": 1.0,
         "created_at": OMIT, "source_doc": OMIT},
        {"fact": _PASSK_LOSER, "confidence": 0.9, "created_at": _days_ago(1),
         "source_doc": OMIT},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("pass@k")
    assert plan["contradictions"] >= 1, plan
    assert plan["actions"] == [], plan["actions"]


def test_two_unattributed_rows_still_demote_on_confidence_alone(world):
    """Clause 2: the guard is asymmetric, so 64% of the corpus stays actable.

    Both rows lack `source_doc` and `created_at` — nobody can say where either
    came from — and the ordinary confidence basis decides, demoting the lower
    one exactly as it did before #1348. This is the majority shape on the board,
    so a symmetric guard ("no unattributed row ever wins") would have gutted the
    loop's whole output to fix one wrong line: the pass would report 40 entities
    and zero actions every night and look healthy doing it.
    """
    facts_root, st, _ = world
    _write_facts(facts_root, "PassBoth", "state", [
        {"fact": _PASSK_UNATTRIBUTED, "confidence": 1.0,
         "created_at": OMIT, "source_doc": OMIT},
        {"fact": _PASSK_LOSER, "confidence": 0.9, "created_at": OMIT, "source_doc": OMIT},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("PassBoth")
    assert len(plan["actions"]) == 1, plan["actions"]
    action = plan["actions"][0]
    assert action["kind"] == "confidence", action
    assert action["loser_fact"] == _PASSK_LOSER, action
    assert action["reason"].startswith("opposing_terms:success/failure;"), action["reason"]


def test_two_attributed_rows_still_demote_on_confidence_alone(world):
    """Clause 3: attribution on both sides leaves the confidence basis untouched.

    The same entity, the same two claim texts and the same 1.0 vs 0.9 gap as the
    test above, with `created_at` restored to both rows — the shape every other
    fixture in this file writes. The demotion fires and the loser is the
    lower-confidence row. This is also the control for
    `test_an_unattributed_winner_cannot_demote_an_attributed_loser`: it shows
    that pair does reach `opposing_terms`, does clear `MIN_CONFIDENCE_GAP`, and
    is declined only by the attribution guard.
    """
    facts_root, st, _ = world
    _write_facts(facts_root, "PassSourced", "state", [
        {"fact": _PASSK_UNATTRIBUTED, "confidence": 1.0, "created_at": _days_ago(20)},
        {"fact": _PASSK_LOSER, "confidence": 0.9, "created_at": _days_ago(1)},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("PassSourced")
    assert len(plan["actions"]) == 1, plan["actions"]
    action = plan["actions"][0]
    assert action["kind"] == "confidence", action
    assert action["loser_fact"] == _PASSK_LOSER, action


def test_the_attribution_guard_reads_a_row_that_omits_the_keys_entirely(world):
    """Clause 4: absent key, not `None` key — and reading it must not raise.

    `plan_entity` judges rows handed back by `_read_facts_cached`, which returns
    the parsed YAML entries verbatim: a fact whose file never carried
    `created_at` or `source_doc` is a dict that has no such key, exactly the
    live `fact-001` row. The first assertion is the positive control that this
    fixture really produced that shape rather than a `None` under the key — a
    guard written `f["created_at"]` would raise `KeyError` here instead of
    planning anything, and a guard written `if "created_at" in f` would answer a
    different question than the one the guard has to answer.

    Here the winner omits both keys and the loser's only evidence is
    `source_doc` — the other half of "carries either", and the half the live
    loser also satisfied.
    """
    facts_root, st, _ = world
    _write_facts(facts_root, "PassKeys", "state", [
        {"fact": _PASSK_UNATTRIBUTED, "confidence": 1.0,
         "created_at": OMIT, "source_doc": OMIT},
        {"fact": _PASSK_LOSER, "confidence": 0.9, "created_at": OMIT,
         "source_doc": _PASSK_LOSER_SOURCE},
    ])
    rows = yaml.safe_load((facts_root / "PassKeys" / "PassKeys-state.md")
                           .read_text(encoding="utf-8").split("---")[1])["facts"]
    winner_row = [r for r in rows if r["fact"] == _PASSK_UNATTRIBUTED][0]
    assert "created_at" not in winner_row and "source_doc" not in winner_row, (
        f"the fixture wrote the keys after all; the live row omits them: "
        f"{sorted(winner_row)}")
    loser_row = [r for r in rows if r["fact"] == _PASSK_LOSER][0]
    assert "created_at" not in loser_row, (
        f"the loser was meant to be attributed by source_doc alone: {sorted(loser_row)}")

    _reindex(st, facts_root)
    plan = fi.plan_entity("PassKeys")          # must not raise KeyError
    assert plan["contradictions"] >= 1, plan
    assert plan["actions"] == [], plan["actions"]


# ── 3. the unified verbs were retired into the tools they wrapped ─────────────
#
# `remember`/`recall`/`forget`/`improve` (#376) were routers layered on top of
# `fact_add`/`vault_recall`/`fact_invalidate`/`run_improvement`, each adding one
# guard. On 2026-09-23 the guards moved into the tools and the verbs went:
# `fact_add` already had the duplicate refusal (#499), `fact_invalidate` took the
# scope refusal and the default date, `recall` was `vault_recall` with a
# default, and the improvement pass is the nightly script's. These pin that the
# guards survived the move.

_RETIRED_VERBS = ("remember", "recall", "forget", "improve")


def test_the_retired_verbs_are_gone_from_the_catalog_and_every_table():
    from agent_mcp import main as M
    from agent_mcp import annotations as A
    names = {t.name for t in asyncio.run(M.list_tools())}
    tables = (A.READ_ONLY | A.DESTRUCTIVE | A.IDEMPOTENT | A.REPEAT_EXPECTED
              | A.PLAN_MODE_ALWAYS_ALLOWED)
    for verb in _RETIRED_VERBS:
        assert verb not in names, verb
        assert verb not in M._dispatch, verb
        assert verb not in tables, verb


def test_fact_add_adds_a_fact_once(world):
    facts_root, st, _ = world
    res = facts._fact_add({"entity": "Bernie", "category": "state",
                           "fact": "Bernie uses a mecanum drive."})
    assert res.get("success") is True, res
    assert _active(st, "Bernie") == 1
    again = facts._fact_add({"entity": "Bernie", "category": "state",
                             "fact": "Bernie uses a mecanum drive."})
    assert again.get("skipped") is True
    assert _active(st, "Bernie") == 1, "fact_add must not duplicate a fact"


def test_fact_invalidate_expires_only_facts_that_match(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "Bernie", "state", [
        {"fact": "Bernie uses a mecanum drive.", "created_at": _days_ago(20)},
        {"fact": "Bernie has a 5-lb Olympic plate mount.", "created_at": _days_ago(20)},
    ])
    _reindex(st, facts_root)
    res = facts._fact_invalidate({"entity": "Bernie", "fact_substring": "mecanum"})
    assert res.get("expired_count") == 1, res
    assert _active(st, "Bernie") == 1


def test_fact_invalidate_refuses_to_blank_an_entity(world):
    """A bare `fact_invalidate(entity=…)` is a blanket expire over every fact
    the entity has. Refused — the guard `forget` carried, now on the tool."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Bernie", "state",
                 [{"fact": "Bernie uses a mecanum drive.", "created_at": _days_ago(20)}])
    _reindex(st, facts_root)
    res = facts._fact_invalidate({"entity": "Bernie", "ended": "2026-09-23"})
    assert res.get("error"), res
    assert res.get("expired_count") == 0
    assert _active(st, "Bernie") == 1


def test_improvement_defaults_to_dry_run(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK (tts.builtin).",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors (tts.builtin).",
         "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    res = fi.run_improvement(sources=("drift",), days=3)
    assert res["apply"] is False and res["actions_taken"] == 0
    assert _active(st) == 2


def test_an_applied_improvement_records_what_it_acted_on(world):
    """The file that lands in `_pipeline/improvement/` has to say which store
    it deleted from, or the one entry point that can delete facts is the one
    whose record cannot."""
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK (tts.builtin).",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors (tts.builtin).",
         "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    res = fi.run_improvement(sources=("drift",), days=3, apply=True)
    assert res["actions_taken"] == 1 and _active(st) == 1
    on_disk = _persisted_record(res)
    assert on_disk["apply"] is True
    assert on_disk["facts_root"] == str(facts_root)
    assert on_disk["kg_db"] == str(st.path)
    assert re.fullmatch(r"[0-9a-f]{40}", on_disk["git_head"]), on_disk["git_head"]
    assert on_disk["isolated"] is True


# ── 4. the schedule half: a consumer nobody dispatches is dead text ───────────
#
# Everything above is reachable-but-unrunnable until something calls it on a
# schedule. That is not a hypothetical here: the 09-08 attempt at this item
# landed the *vault* half (autonomy task #84 + the `fact-improvement` skill)
# and aborted the code half, so the vault spent a day naming a script that did
# not exist — and #463 is the same failure class one level up, a consolidation
# pass described everywhere that runs nowhere.

_AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
_SKILL_PATH = Path.home() / "obsidian" / "skills" / "fact-improvement" / "SKILL.md"
_SCRIPT_PATH = ROOT / "scripts" / "memory" / "fact-improvement.py"


def _improvement_tasks() -> list[tuple[Path, dict]]:
    """Autonomy tasks whose `skill_name` is the fact-improvement pass."""
    if not _AUTONOMY_DIR.is_dir():
        pytest.skip("no vault on this box; the schedule lives in the vault")
    found = []
    for path in sorted(_AUTONOMY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.startswith("---"):
            continue
        try:
            front = yaml.safe_load(text.split("---", 2)[1]) or {}
        except yaml.YAMLError:
            continue
        if str(front.get("skill_name", "")) == "fact-improvement":
            found.append((path, front))
    return found


def test_the_improvement_pass_is_armed_on_a_schedule():
    """#376 asks for a *scheduled* consumer. `up_next` is the only status the
    scheduler dispatches (app/autonomy.py:419), so a task sitting in `draft` is a
    description of a feature, not one."""
    tasks = _improvement_tasks()
    assert tasks, "no autonomy task runs the fact-improvement skill"
    for path, front in tasks:
        assert str(front.get("status")) == "up_next", (
            f"{path.name} is status={front.get('status')!r}: the improvement "
            "pass is not scheduled. Arm it (status up_next) or archive it; a "
            "draft task is dead text, which is #463's failure class.")
        assert str(front.get("frequency")) in ("daily", "weekly"), front.get("frequency")


def test_the_skill_names_a_script_that_exists():
    """The wiring is only complete if the file the skill tells the worker to
    run is in the tree — the half that the 09-08 attempt left missing."""
    tasks = _improvement_tasks()
    assert tasks, "no autonomy task runs the fact-improvement skill"
    assert _SKILL_PATH.is_file(), _SKILL_PATH
    skill_text = _SKILL_PATH.read_text(encoding="utf-8", errors="replace")
    assert "scripts/memory/fact-improvement.py" in skill_text, (
        "the skill must name the command it runs")
    assert _SCRIPT_PATH.is_file(), (
        f"{_SCRIPT_PATH} is named by {_SKILL_PATH} but is not in the tree")


# ── #1326: the write half of `fact_resolve` is its own writer-classified verb ──
#
# `fact_resolve` sat in `READ_ONLY` while `auto_resolve=true` marked the weaker
# side of a pair `invalid_at`. That table is the predicate behind four refusals,
# and none of them fired: plan mode (`plan_mode_blocked_tools`), a bench or eval
# session (`_tool_sandbox.refusal`), a sessionless `call_tool` (#1053), and
# `MCPPool._retry_safe`, which re-sends a read-only call after a transport drop
# — so one dropped stream could re-mark a fact it had already marked. The
# marking now lives in `fact_resolve_apply`, a name absent from every annotation
# table and therefore a writer everywhere; `fact_resolve` only reports.

def _resolve_pair(facts_root):
    """A contradictory pair with one clearly weaker side, ids `stat-001`/`-002`."""
    _write_facts(facts_root, "Lloyd", "state", [
        {"fact": "the feature is enabled", "confidence": 0.9, "created_at": _days_ago(2)},
        {"fact": "the feature is disabled", "confidence": 0.5, "created_at": _days_ago(30)},
    ])


def _fact_rows(facts_root, entity="Lloyd", category="state"):
    raw = (facts_root / entity / f"{entity}-{category}.md").read_text(encoding="utf-8")
    return {f["id"]: f for f in yaml.safe_load(raw.split("---")[1])["facts"]}


def test_fact_resolve_apply_marks_the_weaker_side_and_names_the_files(world):
    """Clause 3, write half: it invalidates the loser, says which facts in which
    files, and the invalidation is visible to a reader without a reindex. A count
    that does not name the file cannot be audited — the reason #874 left the list
    in the payload."""
    from agent_mcp import facts as facts_mod
    facts_root, st, _vault = world
    _resolve_pair(facts_root)
    _reindex(st, facts_root)
    assert _active(st, "Lloyd") == 2, "the pair did not index as two active facts"
    out = facts_mod._fact_resolve_apply({"entity": "Lloyd"})
    assert out["resolved"] == 1, out
    assert sorted((m["file"], m["id"]) for m in out["facts"]) == [
        ("Lloyd/Lloyd-state.md", "stat-002")], out["facts"]
    rows = _fact_rows(facts_root)
    assert rows["stat-002"]["invalid_at"], "the lower-confidence fact was not invalidated"
    assert not rows["stat-002"].get("expired_at"), (
        "invalid is 'should not have been recorded'; expired is 'was true, no longer is'")
    assert "fact_resolve_apply" in (rows["stat-002"].get("invalid_reason") or ""), (
        "the audit line in the vault must name the verb that wrote it")
    assert not rows["stat-001"].get("invalid_at"), "the winner lost its fact too"
    # No `_reindex` between the mark and this count: an invalidation the index
    # still shows as active is not an invalidation, it is a footnote in one file.
    assert _active(st, "Lloyd") == 1, (
        "the marked fact is still active in facts_idx after fact_resolve_apply")


def test_fact_resolve_marks_nothing_even_when_asked_to_auto_resolve(world):
    """Clause 3, read half: the shape that used to write is now inert, and the
    file it used to rewrite does not change at all."""
    from agent_mcp import facts as facts_mod
    facts_root, _st, _vault = world
    _resolve_pair(facts_root)
    path = facts_root / "Lloyd" / "Lloyd-state.md"
    before = path.read_bytes()
    out = facts_mod._fact_resolve({"entity": "Lloyd", "auto_resolve": True})
    assert out["resolved"] == 0, out
    assert out["remaining"] >= 1, out
    assert out["contradictions"], "the report stopped reporting"
    assert "fact_resolve_apply" in json.dumps(out), (
        "a caller that asked for the write has to be told where it went")
    assert path.read_bytes() == before, "`fact_resolve` still writes"
    assert not [i for i, f in _fact_rows(facts_root).items()
                if f.get("invalid_at") or f.get("expired_at")]


def test_the_two_fact_verbs_split_over_the_mcp_module_seam(world):
    """The boundary a caller crosses is `agent_mcp.facts.call_tool`, not the
    private handler: the read must come back clean and the write must arrive."""
    from agent_mcp import facts as facts_mod
    facts_root, _st, _vault = world
    _resolve_pair(facts_root)
    path = facts_root / "Lloyd" / "Lloyd-state.md"
    before = path.read_bytes()
    read = asyncio.run(facts_mod.call_tool(
        "fact_resolve", {"entity": "Lloyd", "auto_resolve": True}))
    assert not (getattr(read, "isError", False) or getattr(read, "is_error", False)), read
    assert json.loads(read.content[0].text)["resolved"] == 0
    assert path.read_bytes() == before, "the read verb wrote to the tree"
    applied = asyncio.run(facts_mod.call_tool(
        "fact_resolve_apply", {"entity": "Lloyd"}))
    assert json.loads(applied.content[0].text)["resolved"] == 1
    assert path.read_bytes() != before


def test_fact_resolve_apply_is_a_registered_writer_and_resolve_stays_a_reader():
    """The classification is the fix: an unlisted name is a writer by the safe
    default in `annotations_for`, and the reader must stop advertising a
    parameter that no longer does anything."""
    import asyncio as _asyncio

    from agent_mcp import annotations as A
    from agent_mcp import facts as facts_mod
    assert "fact_resolve" in A.READ_ONLY
    assert "fact_resolve_apply" not in A.READ_ONLY | A.IDEMPOTENT | A.REPEAT_EXPECTED
    tools = {t.name: t for t in _asyncio.run(facts_mod.list_tools())}
    assert "fact_resolve_apply" in tools, sorted(tools)
    assert "auto_resolve" not in (
        tools["fact_resolve"].input_schema.get("properties") or {})
    assert tools["fact_resolve_apply"].input_schema.get("required") == ["entity"]
    for name in ("fact_resolve", "fact_resolve_apply"):
        assert len(tools[name].description or "") >= 60, name
        for pname, spec in (tools[name].input_schema.get("properties") or {}).items():
            assert (spec.get("description") or "").strip(), f"{name}.{pname}"
        assert "fact_resolve_apply" in (tools["fact_resolve"].description or "") \
            or "fact_resolve_apply" in (tools[name].description or ""), name


# ── #1383: an unreadable store must not report success ───────────────────────
#
# The 2026-09-23 nightly pass ran with no `kg.sqlite` at all and exited 0:
# `_known_entities()` and `_active_count()` turn every store failure into the
# `{}` / `-1` sentinels BY DESIGN (a missing count must never read as a zero),
# so `run_improvement` never raises, and the wrapper's only store signal was
# "did it throw" — which the design guarantees it never does. The sentinels
# stay. What must not stay silent is the verdict: one probe sets a flag and an
# error text, the record carries both, the wrapper prints the line beside its
# counts, and the exit code says 2. These tests pin that whole chain across
# both boundaries — the module call and the subprocess exit status.

def _point_the_store_at_nothing(monkeypatch, tmp_path):
    """Make `app.kg_store.store()` raise the real `StoreUnavailable`.

    Not `configure()` — that *provisions* an absent path on purpose (the
    writer's route); `store()` refuses one. Pointing the default path at a
    missing file and resetting the cached handle reproduces exactly what the
    09-23 nightly hit: `StoreUnavailable: no knowledge-graph database at …`.
    """
    missing = tmp_path / "no-store-here" / "kg.sqlite"
    monkeypatch.setattr(kg_store, "_default_path", missing)
    kg_store.reset()
    return missing


def test_record_carries_the_store_verdict_and_the_pass_still_plans(world, monkeypatch,
                                                                   tmp_path):
    """Clause 1: the returned record and its on-disk copy name the unreadable
    store — flag plus error text naming `StoreUnavailable` — from ONE probe,
    while the markdown half still produces its plan."""
    facts_root, _st, _vault = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK (tts.builtin).",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors (tts.builtin).",
         "created_at": _days_ago(2)},
    ])
    missing = _point_the_store_at_nothing(monkeypatch, tmp_path)
    probe = fi._store_probe()
    assert probe["store_ok"] is False
    assert "StoreUnavailable" in probe["store_error"], probe
    assert str(missing) in probe["store_error"], (
        "the verdict must name the path the failing probe could not open")
    # "Derived from ONE probe" is the clause, so count the probes: a pass that
    # asked the store twice could write a record whose flag and its counts
    # describe two different readings of it — the exact shape that let `-1`
    # read as a zero while some other line claimed the graph had been seen.
    calls = []
    real_probe = fi._store_probe

    def counting_probe():
        calls.append(None)
        return real_probe()

    monkeypatch.setattr(fi, "_store_probe", counting_probe)
    rec = fi.run_improvement(entities=["TTS"])
    assert len(calls) == 1, f"run_improvement probed the store {len(calls)} times"
    assert rec["store_ok"] is False
    assert rec["store_error"] == probe["store_error"], (
        "the record's verdict must come from the one probe, not a second read")
    # The markdown-driven half runs on a dead index — that is what makes this
    # partial success rather than a failed pass, and why exit 2 must still be
    # reachable after real planning happened.
    assert rec["actions_planned"] == 1, rec["per_entity"]
    # The sentinels propagate into the record unchanged (#1383's `Do not`).
    assert rec["before_active"] == -1 and rec["after_active"] == -1
    assert rec["delta_active"] is None
    on_disk = _persisted_record(rec)
    assert on_disk["store_ok"] is False
    assert "StoreUnavailable" in on_disk["store_error"]


def test_store_probe_reports_ok_and_the_sentinels_stay_when_it_cannot_answer(
        world, monkeypatch, tmp_path):
    """Clause 5: `_active_count()` still answers -1 and `_known_entities()`
    still answers {} when the store cannot — propagated, not removed. The
    probe must also be able to say ok against the live handle, or it is an
    unconditional alarm rather than a verdict."""
    facts_root, st, _vault = world
    assert fi._store_probe() == {"store_ok": True, "store_error": None}
    assert fi._active_count() == 0
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
         "created_at": _days_ago(2)}])
    _reindex(st, facts_root)
    assert fi._active_count() == 1
    assert fi._known_entities() == {"tts": "TTS"}
    _point_the_store_at_nothing(monkeypatch, tmp_path)
    assert fi._store_probe()["store_ok"] is False
    assert fi._active_count() == -1, "the -1 sentinel must stay"
    assert fi._known_entities() == {}, "the {} sentinel must stay"


def _run_wrapper(tmp_path, kg_db, args):
    """The wrapper as the nightly runs it: a real subprocess, the store moved
    with `LLOYD_KG_DB`, all runtime data pointed at tmp."""
    import os
    env = dict(os.environ)
    env["LLOYD_DATA"] = str(tmp_path / "data")
    env["LLOYD_FACTS_ROOT"] = str(tmp_path / "facts")
    env["LLOYD_KG_DB"] = str(kg_db)
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *args],
        capture_output=True, text=True, env=env, timeout=180,
        cwd=str(tmp_path))


def test_wrapper_warns_and_exits_2_when_the_store_was_unreadable(tmp_path):
    """Clauses 2+3 against the real process boundary: with `LLOYD_KG_DB`
    pointing at a missing file the pass prints the unreadable line beside its
    counts, keeps printing the sentinel counts line, and exits 2 — even though
    the drift/contradiction half completed."""
    db = tmp_path / "absent" / "kg.sqlite"
    proc = _run_wrapper(tmp_path, db, ["--entity", "Probe-Entity"])
    out = proc.stdout
    assert proc.returncode == 2, f"exit {proc.returncode}\nstdout:\n{out}\nstderr:\n{proc.stderr}"
    assert "[warn] knowledge-graph store unreadable:" in out, out
    assert "StoreUnavailable" in out, out
    assert "[facts] active -1 -> -1 (delta None)" in out, (
        "the sentinel line stays beside the warning, not replaced by it")
    assert "[store] ok" not in out, out


def test_wrapper_exits_0_and_calls_the_store_ok_on_the_control(tmp_path):
    """Clause 4: a real store at the `LLOYD_KG_DB` path flips every reading —
    store ok, no warning line, exit 0. This is what proves the change is a
    verdict and not an unconditional alarm."""
    db = tmp_path / "real" / "kg.sqlite"
    st = kg_store.KGStore(db)  # the writer's route: provision on purpose
    st.close()
    proc = _run_wrapper(tmp_path, db, ["--entity", "Probe-Entity"])
    out = proc.stdout
    assert proc.returncode == 0, f"exit {proc.returncode}\nstdout:\n{out}\nstderr:\n{proc.stderr}"
    assert "[store] ok" in out, out
    assert "unreadable" not in out.lower(), out
    assert "[facts] active 0 -> 0 (delta 0)" in out, out
    records = list((tmp_path / "data" / "_pipeline" / "improvement").glob("*.json"))
    assert len(records) == 1, records
    rec = json.loads(records[0].read_text())
    assert rec["store_ok"] is True and rec["store_error"] is None


def test_wrapper_exit_codes_keep_their_distinct_meanings(monkeypatch, capsys):
    """Clause 3's second half: 2 (store unreadable) and 3 (budget overrun) are
    distinct codes, and an overrun that COINCIDES with an unreadable store
    still reports 3 — the budget signal is never swallowed into the new one.
    Each case also checks the printed store line: `[store] ok` is owed only to
    a record that actually says so, and a code nobody can read beside the
    counts is the same false green in a different column."""
    def run(argv, drop=(), **rec_over):
        rec = {"apply": False, "signals": 0, "entities": [], "drift_candidates_total": None,
               "actions_planned": 0, "actions_taken": 0, "before_active": 0,
               "after_active": 0, "delta_active": 0, "per_entity": [],
               "store_ok": True, "store_error": None, "kg_db": "/x/kg.sqlite",
               "fact_entity_recall": None, "record_path": "/x/record.json"}
        rec.update(rec_over)
        for key in drop:
            rec.pop(key, None)
        spec = importlib.util.spec_from_file_location(
            "fact_improvement_script_exit_codes", _SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        monkeypatch.setattr(fi, "run_improvement", lambda **kw: rec)
        monkeypatch.setattr(sys, "argv", ["fact-improvement.py", *argv])
        code = mod.main()
        return code, capsys.readouterr().out

    code, out = run(["--entity", "E"])
    assert code == 0, "store ok, in budget: clean pass"
    assert "[store] ok (/x/kg.sqlite)" in out, out

    code, out = run(["--entity", "E"], store_ok=False, store_error="StoreUnavailable: gone",
                    before_active=-1, after_active=-1, delta_active=None)
    assert code == 2
    assert "[warn] knowledge-graph store unreadable: StoreUnavailable: gone" in out, out
    assert "[store] ok" not in out, out

    code, out = run(["--apply", "--entity", "E", "--max-actions", "5"],
                    apply=True, actions_planned=7, actions_taken=7)
    assert code == 3

    code, out = run(["--apply", "--entity", "E", "--max-actions", "5"],
                    apply=True, actions_planned=7, actions_taken=7,
                    store_ok=False, store_error="StoreUnavailable: gone",
                    before_active=-1, after_active=-1, delta_active=None)
    assert code == 3, "the budget overrun keeps code 3 even when the store was also unreadable"
    assert "[warn] knowledge-graph store unreadable:" in out, out

    # A record that carries no verdict at all gets neither line: the printed
    # half of the fix is a measurement, not a default for a missing key. Its
    # exit code is pinned to 0 here (a missing flag is not evidence of an
    # unreadable store either) so that if the wrapper ever starts demanding a
    # verdict, this test is what names the change.
    code, out = run(["--entity", "E"], drop=("store_ok", "store_error"))
    assert code == 0
    assert "[store] ok" not in out, out
    assert "unreadable" not in out, out


# ── #1654: an unreadable store must stop the WRITES, not the pass ────────────
#
# #1383 made an unreadable store loud — one probe, a verdict in the record, a
# warning line, exit 2 — and every one of those fires AFTER the pass has written.
# The apply loop never consults `store_verdict`, and the writes do not go to the
# unreadable store at all: `apply_action` stamps `expired_at` / `invalid_at` into
# the MARKDOWN fact files under `FACTS_ROOT` (its `_apply_fact_marks` call) and
# aims through `_get_facts_sync`, which is also the markdown read path; the index
# is only consulted for counts, where a dead store answers the `-1` sentinel. So
# an operator who ran `--apply` during a store outage retired facts, was told the
# store was unreadable only by the exit code they got afterwards
# (`scripts/memory/fact-improvement.py`, the `store_ok is False` → 2 rung, which
# is evaluated after `run_improvement` has already written), and got a record
# whose `before_active`/`after_active` were both `-1` — unable to show what it
# changed, while the index went on serving the facts it had just retired until a
# rebuild.
#
# The gate therefore belongs on the write path and nowhere else: #1383 pins that
# a dry run on a dead store still plans (`actions_planned == 1`) and still exits
# 2, so an outage must not silence the pass — only stop it from writing.

def _write_voices_pair(root, entity):
    """Write one contradictory entity and return its fact file path.

    The 30-day-old `working / 200 OK` claim loses to the 2-day-old `broken / 500
    errors` one on recency alone, so a plan over this entity is exactly ONE
    action — the unit the apply loop is required to take or leave.
    """
    _write_facts(root, entity, "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK (tts.builtin).",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors (tts.builtin).",
         "created_at": _days_ago(2)},
    ])
    return root / entity / f"{entity}-state.md"


def test_an_apply_pass_writes_nothing_when_the_store_could_not_be_read(world,
                                                                       monkeypatch,
                                                                       tmp_path):
    """Clause 1, both arms. On a readable store this call retires the loser; with
    the store unreadable it retires nothing and the fact file is byte-identical —
    which is the whole difference between an outage and a clean tree, and today
    only the clean tree is safe."""
    facts_root, st, _vault = world
    control_file = _write_voices_pair(facts_root, "TTSCtl")
    target_file = _write_voices_pair(facts_root, "TTS")
    _reindex(st, facts_root)

    control_before = control_file.read_bytes()
    applied = fi.run_improvement(apply=True, entities=["TTSCtl"])
    assert applied["store_ok"] is True, applied["store_error"]
    assert (applied["actions_planned"], applied["actions_taken"]) == (1, 1), \
        applied["per_entity"]
    assert control_file.read_bytes() != control_before, \
        "the readable-store arm must actually write, or the refused arm proves nothing"

    _point_the_store_at_nothing(monkeypatch, tmp_path)
    target_before = target_file.read_bytes()
    rec = fi.run_improvement(apply=True, entities=["TTS"])
    assert rec["store_ok"] is False
    assert rec["actions_planned"] == 1, (
        "the markdown half still plans on a dead store — that is what makes this "
        "a refusal rather than an empty pass:", rec["per_entity"])
    assert rec["actions_taken"] == 0, rec["per_entity"]
    assert target_file.read_bytes() == target_before, (
        "an apply pass that could not read the store retired a fact anyway")


def test_a_refused_apply_record_names_the_store_as_the_reason_it_wrote_nothing(
        world, monkeypatch, tmp_path):
    """Clause 2. A refused apply and an empty plan both report
    `actions_taken: 0`, and only the record can tell them apart: the refused one
    pairs `apply: true` with `store_ok: false` and a reason quoting the store's
    own error, and the readable pass that simply had nothing to do carries no
    such reason."""
    facts_root, st, _vault = world
    _write_voices_pair(facts_root, "TTS")
    # A near-duplicate pair, which the loop reports and never deletes: an apply
    # pass over it plans 0 and takes 0 on a healthy store.
    _write_facts(facts_root, "Transcripts", "usage", [
        {"fact": "Transcripts were extracted from the video with the VTT parser.",
         "created_at": _days_ago(20), "confidence": 0.9},
        {"fact": "Transcripts were extracted from the video with the parser.",
         "created_at": _days_ago(19), "confidence": 0.6},
    ])
    _reindex(st, facts_root)

    empty = fi.run_improvement(apply=True, entities=["Transcripts"])
    assert empty["store_ok"] is True and empty["store_error"] is None
    assert (empty["actions_planned"], empty["actions_taken"]) == (0, 0), empty["per_entity"]
    assert empty["writes_refused_reason"] is None, (
        "a pass with nothing to do must not borrow the refusal reason — that "
        "would make the new key a second spelling of `actions_taken: 0`")

    _point_the_store_at_nothing(monkeypatch, tmp_path)
    rec = fi.run_improvement(apply=True, entities=["TTS"])
    assert rec["apply"] is True and rec["store_ok"] is False
    assert (rec["actions_planned"], rec["actions_taken"]) == (1, 0), rec["per_entity"]
    why = rec["writes_refused_reason"]
    assert why and "unreadable" in why, rec
    assert "StoreUnavailable" in why, (
        "the reason must quote the probe's error, which is the only text naming "
        "the store the pass could not open:", why)
    on_disk = _persisted_record(rec)
    assert on_disk["apply"] is True and on_disk["store_ok"] is False
    assert on_disk["writes_refused_reason"] == why, (
        "the reason has to survive into the record an operator reads, not just "
        "the returned dict")


def test_a_dry_run_on_an_unreadable_store_still_plans_and_reports_no_refusal(
        world, monkeypatch, tmp_path):
    """Clause 3, the module-call half: the gate is on the write path ONLY. The
    #1383 dry-run behaviour is unchanged — same plan, same sentinels, same
    record — and the refusal key stays null, because a dry run was never asking
    to write and a reason there would read as an apply pass that was refused."""
    facts_root, st, _vault = world
    target_file = _write_voices_pair(facts_root, "TTS")
    _reindex(st, facts_root)
    _point_the_store_at_nothing(monkeypatch, tmp_path)
    before = target_file.read_bytes()

    rec = fi.run_improvement(entities=["TTS"])
    assert rec["apply"] is False and rec["store_ok"] is False
    assert rec["actions_planned"] == 1, rec["per_entity"]
    assert rec["actions_taken"] == 0
    assert rec["writes_refused_reason"] is None, rec["writes_refused_reason"]
    assert target_file.read_bytes() == before
    assert {a["planned"] for a in rec["per_entity"][0]["actions"]} == {True}, \
        rec["per_entity"][0]["actions"]


def test_the_cli_exits_2_and_writes_nothing_when_apply_meets_an_absent_store(tmp_path):
    """Clause 3, across the real process boundary: `--apply` with `LLOYD_KG_DB`
    pointing at a missing file still exits 2 and still prints the unreadable
    line — but it leaves the fact tree alone, so the exit code stops arriving
    after the writes it used to follow."""
    db = tmp_path / "absent" / "kg.sqlite"
    target_file = _write_voices_pair(tmp_path / "facts", "TTS")
    before = target_file.read_bytes()

    proc = _run_wrapper(tmp_path, db, ["--apply", "--entity", "TTS"])
    out = proc.stdout
    assert proc.returncode == 2, (
        f"exit {proc.returncode}\nstdout:\n{out}\nstderr:\n{proc.stderr}")
    assert "[warn] knowledge-graph store unreadable:" in out, out
    assert "[facts] active -1 -> -1 (delta None)" in out, out
    assert target_file.read_bytes() == before, (
        "the wrapper retired a fact through an unreadable store")


# ── #1544: a resolved contradiction leaves a trace naming the winner ─────────
#
# `fact_resolve_apply` invalidated the weaker fact and recorded nothing else: no
# winner, no counterparty. "What did this contradict, and who won?" had no answer
# anywhere downstream, and the canonical `conflicts_with` edge type
# (`app.kg_store.EDGE_TYPES`, canonical since 4f4432f3 / #546) held exactly 1 row
# — hand-filed through `fact_relate` — over 36,875 active edges, measured
# 2026-09-27 through `app.kg_store` on the live store.
#
# The trace is a record on the LOSER, not an edge row, because a contradiction
# pair has no two entities to be an edge's endpoints: `_resolve_scan` globs ONE
# entity directory and `_detect_contradictions_sync` pairs inside that one list,
# while `EdgeStore.add` raises `refusing self-loop edge` when `source == target`
# (`app/kg_store.py:719`) and `rewrite_endpoint` drops such a pair rather than
# rewriting it (`:867`). Measured the same day: 12,244 entity dirs, 0 of them
# holding more than one distinct `entity`; 48,488 edge rows, 0 with
# `source = target` in any state. Which shape an intra-entity contradiction takes
# in the graph was #1596's to settle, and it settled it markdown-only on
# 2026-09-30: neither guard relaxes and no fact-granularity node id is minted, so
# the trace is the representation of record and
# `test_a_resolution_mints_no_edge_row_under_the_markdown_only_ruling` is the
# permanent fence on that state — reopen only through the trigger written into
# `_contradiction_trace`'s own docstring (a `contradiction_trace_coverage` count
# above 0 AND a demonstrated traversal gain), not through a round's preference.

TRACE_KEY = "conflicts_with"          # the canonical edge type's own spelling
PAIR_FILE = "Lloyd/Lloyd-state.md"    # spelled as `retrieval.fact_source_file` does


def _write_adjudicable_pair(facts_root, *, winner_conf=0.9, loser_conf=0.5,
                            omit_from_loser=()):
    """One file, one adjudicable pair: `stat-001` beats `stat-002`.

    `enabled`/`disabled` is a member of `_OPPOSING_PAIRS` matched whole-word
    (#701), so the detector fires `opposing_terms` rather than the 0.6-overlap
    heuristic, which pairs facts that merely read alike. The YAML is written here
    rather than through `_write_facts` so a test can drop the loser's `id` — the
    one shape that makes `fact_identity` return None, and with it both the mark
    and the trace refuse to land.
    """
    def row(fid, text, conf, days):
        return {"fact": text, "confidence": conf, "category": "state", "id": fid,
                "created_at": _days_ago(days), "valid_at": _days_ago(days),
                "invalid_at": None, "expired_at": None, "provenance": "STATED",
                "source_doc": None}
    winner = row("stat-001", "the feature is enabled", winner_conf, 2)
    loser = row("stat-002", "the feature is disabled", loser_conf, 30)
    for key in omit_from_loser:
        loser.pop(key, None)
    fm = {"type": "facts", "entity": "Lloyd", "category": "state",
          "facts": [winner, loser]}
    d = facts_root / "Lloyd"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Lloyd-state.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# Lloyd\n", encoding="utf-8")


def _traced_facts(facts_root):
    """(file_name, fact_id) for every fact entry carrying a contradiction trace."""
    out = []
    for path in sorted(facts_root.rglob("*.md")):
        fm = yaml.safe_load(path.read_text(encoding="utf-8").split("---")[1]) or {}
        for f in fm.get("facts") or []:
            if isinstance(f, dict) and f.get(TRACE_KEY):
                out.append((path.name, f.get("id")))
    return out


def test_a_resolved_pair_leaves_exactly_one_trace_on_the_loser(world):
    """Clause 1: the mark and the trace are one event, on the fact that lost."""
    facts_root, _st, _vault = world
    _write_adjudicable_pair(facts_root)
    out = facts._fact_resolve_apply({"entity": "Lloyd"})
    assert out["resolved"] == 1, out
    assert out["traces_written"] == 1, out
    assert _traced_facts(facts_root) == [("Lloyd-state.md", "stat-002")], \
        "the pair's trace must sit on the loser and nowhere else"
    rows = _fact_rows(facts_root)
    assert rows["stat-002"]["invalid_at"], "the loser was not invalidated"
    assert not rows["stat-001"].get("invalid_at"), "the winner lost its fact too"
    trace = rows["stat-002"][TRACE_KEY]
    assert trace["resolved_at"] == rows["stat-002"]["invalid_at"], (
        "a trace stamped apart from the mark is a claim about a resolution the "
        "file does not record")
    assert trace["reason"] == "opposing_terms:enabled/disabled", trace


def test_the_trace_names_the_winner_by_a_pointer_that_resolves(world):
    """Clause 2: who won, answered from the loser's own record.

    `fact` is copied so the answer needs no second read, and `file` + `fact_id`
    are the handle `fact_identity` uses, so the winner is one fact and not every
    fact sharing a per-file counter id (#874). The pointer is asserted to land: a
    trace naming a file or an id that is not there would satisfy a "names both
    sides" reading of the clause while answering nothing.
    """
    facts_root, _st, _vault = world
    _write_adjudicable_pair(facts_root)
    facts._fact_resolve_apply({"entity": "Lloyd"})
    read = facts._fact_get({"entity": "Lloyd", "include_expired": True})
    losers = [f for f in read["facts"] if f.get(TRACE_KEY)]
    assert [f["id"] for f in losers] == ["stat-002"], read["facts"]
    trace = losers[0][TRACE_KEY]
    assert trace["type"] == TRACE_KEY and trace["entity"] == "Lloyd", trace
    assert trace["fact_id"] == "stat-001" and trace["confidence"] == 0.9, trace
    assert trace["file"] == PAIR_FILE, trace
    assert trace["fact"] == "the feature is enabled", trace
    assert (facts_root / trace["file"]).is_file(), (
        f"the trace names a file that is not there: {trace['file']}")
    winner = _fact_rows(facts_root)["stat-001"]
    assert (winner["fact"], winner["confidence"]) == (trace["fact"],
                                                      trace["confidence"])
    # The default read does not show it. An invalidated fact leaves the active
    # path, so the trace is reachable only through the read that asks for
    # expired/invalid facts — the shape the clause names, not an accident.
    assert [f["id"] for f in facts._fact_get({"entity": "Lloyd"})["facts"]] == [
        "stat-001"], "the loser is still on the active read path"


def test_a_second_resolve_apply_over_a_resolved_pair_adds_no_trace(world):
    """Clause 3, dedupe half: a settled pair stays settled, byte for byte."""
    facts_root, _st, _vault = world
    _write_adjudicable_pair(facts_root)
    path = facts_root / "Lloyd" / "Lloyd-state.md"
    facts._fact_resolve_apply({"entity": "Lloyd"})
    after_first = path.read_bytes()
    out = facts._fact_resolve_apply({"entity": "Lloyd"})
    assert out["resolved"] == 0 and out["traces_written"] == 0, out
    assert len(_traced_facts(facts_root)) == 1, "the same pair was traced twice"
    assert path.read_bytes() == after_first, "a second run rewrote a settled pair"


def test_a_run_that_marks_no_fact_traces_nothing(world):
    """Clause 3, zero-mark halves: equal confidences, and an unattributed loser."""
    facts_root, _st, _vault = world
    _write_adjudicable_pair(facts_root, winner_conf=0.7, loser_conf=0.7)
    equal = facts._fact_resolve_apply({"entity": "Lloyd"})
    assert equal["resolved"] == 0 and equal["traces_written"] == 0, equal
    assert _traced_facts(facts_root) == [], (
        "a pair with no basis to pick a winner must not name one")
    assert not [f for f in _fact_rows(facts_root).values() if f.get("invalid_at")]

    (facts_root / "Lloyd" / "Lloyd-state.md").unlink()
    _write_adjudicable_pair(facts_root, omit_from_loser=("id",))
    orphan = facts._fact_resolve_apply({"entity": "Lloyd"})
    assert orphan["resolved"] == 0 and orphan["traces_written"] == 0, orphan
    assert _traced_facts(facts_root) == [], (
        "a loser that cannot be marked cannot be traced either")
    assert orphan["unapplied"], "the run has to say it marked nothing, and why"


def test_a_resolution_mints_no_edge_row_under_the_markdown_only_ruling(world):
    """The fence on the half of #1544 that was never built — permanent by ruling.

    A `conflicts_with` edge for an intra-entity pair would need either
    `EdgeStore.add` to stop refusing self-loops or fact-granularity node ids — the
    two designs #1596 settled on 2026-09-30 with the answer markdown-only, each of
    which would act on all 48,488 edge rows and on `edges.nodes()`, from which
    `fact_neighbors` and `fact_search` derive their nodes. Neither was built: the
    trace shipped, the edge did not, and that is the decision rather than the
    open question this node was annotated as when the ruling was still owed.

    So the fence is permanent unless the reopen trigger is met —
    `contradiction_trace_coverage` reporting a trace count above 0 over the live
    corpus AND a demonstrated traversal gain, both halves, as written into
    `_contradiction_trace`'s docstring. Until then a red run here means the WRITE
    regressed by minting an edge row; it does not mean the ruling changed, and it
    is not licence to relax a guard to get to green.
    """
    facts_root, st, _vault = world
    _write_adjudicable_pair(facts_root)
    out = facts._fact_resolve_apply({"entity": "Lloyd"})
    assert out["traces_written"] == 1, out
    assert st.edges.active(types=[TRACE_KEY]) == []
    assert st.edges.count(active_only=False) == 0, "the resolve path wrote an edge row"


# ── #1596: the trace is on the advertised surface, not only in the write ──────
# #1596 also settled the ENDPOINT shape markdown-only (#1871); this header is
# about the advertised surface, and the ruling is named so an id sweep finds it.
#
# `fact_resolve_apply` has left a `conflicts_with` trace on every loser since
# #1544, and neither surface a client reads said so: the registered description
# promised only "mark `invalid_at` … and report which facts in which files it
# marked", and the `architecture/tools.md` row said the same in fewer words, while
# the result carried `traces_written` and the loser carried the winning fact. A
# caller therefore had to read the handler to learn that a resolution is
# recoverable at all (#1596 clause 1).

TRACE_COUNT_KEY = "traces_written"     # the key the result reports the count under


def _registered_tool(name: str):
    """One tool exactly as an MCP client receives it from `list_tools()`."""
    tools = {t.name: t for t in asyncio.run(facts.list_tools())}
    assert name in tools, sorted(tools)
    return tools[name]


def _tools_md_row(tool_name: str) -> str:
    """The single `architecture/tools.md` table row for `tool_name`."""
    rows = [ln for ln in (ROOT / "architecture" / "tools.md").read_text(
                encoding="utf-8").splitlines()
            if ln.startswith(f"| `{tool_name}` |")]
    assert len(rows) == 1, rows
    return rows[0]


def test_the_registered_description_names_the_trace_and_its_count():
    """Clause 1, first half: what a client can learn without reading the handler."""
    desc = _registered_tool("fact_resolve_apply").description or ""
    assert f"`{facts._CONTRADICTION_TRACE}`" in desc, (
        f"the advertised surface never names the trace key: {desc}")
    assert TRACE_COUNT_KEY in desc, (
        f"the advertised surface never names the returned count: {desc}")
    assert "loser" in desc, desc


def test_the_advertised_names_are_the_ones_the_write_produces(world):
    """Clause 1, the join: the description describes THIS handler's output.

    A description naming a key the result does not carry is worse than one naming
    nothing, so both promised names are checked against a real resolve run — the
    MCP registration on one side of the seam, the fact tree on the other.
    """
    facts_root, _st, _vault = world
    _write_adjudicable_pair(facts_root)
    desc = _registered_tool("fact_resolve_apply").description or ""
    out = facts._fact_resolve_apply({"entity": "Lloyd"})
    assert out["resolved"] == 1 and out[TRACE_COUNT_KEY] == 1, out
    assert f"`{facts._CONTRADICTION_TRACE}`" in desc, desc
    assert _traced_facts(facts_root) == [("Lloyd-state.md", "stat-002")], (
        "the description promises a trace the write does not leave")


def test_the_tools_md_row_says_the_same_thing():
    """Clause 1, second half: the human-readable row names trace and count too."""
    row = _tools_md_row("fact_resolve_apply")
    assert facts._CONTRADICTION_TRACE in row, row
    assert TRACE_COUNT_KEY in row, row
    assert "invalid_at" in row, "the row still has to say what the mark is"


#: #1596's ruling, recorded 2026-09-30 from measurement: a resolved intra-entity
#: contradiction stays **markdown-only** — the per-record `conflicts_with` trace is
#: the representation of record, neither `EdgeStore` self-loop guard relaxes, and no
#: fact-granularity node id is minted. The reopen trigger lives in
#: `_contradiction_trace`'s own docstring, and
#: `test_the_endpoint_ruling_is_stated_in_the_shipped_prose` is what keeps this
#: file's prose from drifting back into asking the question again.
RULING = "markdown-only"

#: Words that mark a prose block as being about the endpoint shape, which #1596
#: settled markdown-only. Any block carrying one has to state that ruling — that is
#: how a stale pointer fails this file instead of shipping.
ENDPOINT_POINTER_MARKS = ("self-loop", "fact-granularity", "edge row")

#: framings that presented the ruling as still owed to somebody. Each needle is
#: assembled from fragments so this file cannot be the hit for the grep that proves
#: the phrasing is gone — the reason `#1769`'s test splits its needle the same way.
OPEN_QUESTION_FRAMINGS = (
    "for a person to " + "rule on",
    "is open, " + "now",
    "puts to a " + "person",
    "when #" + "1596's ruling lands",
    "the test that has to " + "change",
    "ruling " + "#1596 now carries",
    "_while_" + "1593_is_unruled",
)


def _flat(text: str) -> str:
    """Prose with its line wrapping folded away, so a phrase is matched whole.

    Every instance of the framings below wrapped mid-phrase, and the line-oriented
    form of the tree-wide check that #1871's triage recorded as the premise grep
    reported nothing about the wrapped one while that claim was shipped.
    """
    return " ".join(text.split())


def _prose_blocks(src: str) -> list[str]:
    """Every module-level prose block in a file: unindented `#` runs and docstrings.

    Prose here travels in two shapes — a `#:` block above a module constant, or a
    header comment above a section of tests, and a docstring — and #1596's round put
    a pointer in both, so a check that reads only `inspect.getdoc` of one function
    sees one of the four it is grading. An indented `#` line is a step marker inside
    a test body rather than a pointer: sweeping those would turn the endpoint-shape
    check into a hunt for the word "markdown-only" under every comment that happens
    to mention an endpoint, which grades the prose of the test rather than the
    shipped claim.
    """
    blocks: list[str] = []
    run: list[str] = []
    for line in src.splitlines():
        if line.startswith("#"):
            run.append(line.lstrip("#").strip())
        elif run:
            blocks.append(" ".join(run))
            run = []
    if run:
        blocks.append(" ".join(run))
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                blocks.append(doc)
    return blocks


def _comment_above(src: str, name: str) -> str:
    """The comment block immediately above the module-level binding `name`.

    `_CONTRADICTION_TRACE`'s pointer is a `#:` comment, invisible to `getdoc`, and
    it is the one line of shipped prose an MCP client never sees but every future
    reader of the key does.
    """
    lines = src.splitlines()
    at = next((i for i, ln in enumerate(lines)
               if re.match(rf"^{name}\b", ln)), None)
    assert at is not None, f"{name} is no longer a module-level binding"
    block: list[str] = []
    j = at - 1
    while j >= 0 and lines[j].strip().startswith("#"):
        block.append(lines[j].strip().lstrip("#").strip())
        j -= 1
    return " ".join(reversed(block))


def _shipped_prose_blocks() -> list[tuple[str, str]]:
    """Every module-level prose block of both files this ruling is written into,
    as `(flattened, raw)`.

    The two files are the two that carry a pointer: `agent_mcp/facts.py` holds the
    builder, and this file holds the section header and the fence docstring. Raw is
    kept beside the flattened form because an assertion that only prints folded prose
    cannot show the line break a reader has to find.
    """
    return [(_flat(b), b) for f in (facts.__file__, __file__)
            for b in _prose_blocks(Path(f).read_text(encoding="utf-8"))]


def test_the_endpoint_ruling_is_named_in_both_prose_shapes():
    """#1871 clause 1: the shipped prose states the ruling, in BOTH shapes.

    `#1596 settled the endpoint shape **markdown-only** on 2026-09-30`, and no edge
    row will ever be minted for an intra-entity pair. Two shapes because the
    `_CONTRADICTION_TRACE` pointer is a `#:` comment — invisible to `getdoc`, never
    sent to an MCP client, and read by every human who opens the key.

    Falsified by reverting either block of `agent_mcp/facts.py` prose to what #1596's
    round shipped: the docstring then names no ruling and the comment offers the join
    key to "whoever rules on" the design.
    """
    doc = inspect.getdoc(facts._contradiction_trace) or ""
    flat_doc = _flat(doc)
    assert RULING in flat_doc, (
        "the trace docstring does not name the ruling, so a reader cannot tell "
        f"whether the endpoint shape is still owed: {doc}")
    assert "no edge row will ever be minted" in flat_doc, (
        "the docstring names the ruling but not what it forecloses, so a later "
        f"round can read 'markdown-only' as this week's convenience: {doc}")

    key_comment = _comment_above(
        Path(facts.__file__).read_text(encoding="utf-8"), "_CONTRADICTION_TRACE")
    assert RULING in key_comment, (
        "the trace-key comment no longer states the ruling — it is the pointer a "
        f"reader of the constant itself sees: {key_comment}")
    assert "no edge row" in key_comment, (
        "the trace-key comment offers the shared spelling as a join key again, "
        f"which is the framing that invited a round to use it: {key_comment}")


def test_the_endpoint_ruling_is_not_asked_as_an_open_question():
    """#1871 clause 2: nothing shipped still asks what #1596 settled markdown-only.

    The predecessor of this node asserted `"#1596's endpoint design" in src` — prose
    pinning an open item, which is a claim with an expiry date that expired the
    moment the ruling landed. What is pinned now is the opposite: the open framings
    are ABSENT. It fails when a round re-asks, which is the same event seen from the
    side that keeps shipping prose honest.

    Matching is whole-phrase on folded prose, and the same needles are re-run against
    the whole tree with `git grep -F`: every shipped instance wrapped mid-phrase,
    which is how a line-oriented grep reported nothing about the wrapped open-item
    framing while that claim was in the file.
    """
    own = Path(__file__).read_text(encoding="utf-8")
    for needle in OPEN_QUESTION_FRAMINGS:
        assert needle not in _flat(Path(facts.__file__).read_text(encoding="utf-8")), (
            f"{needle!r} is back in agent_mcp/facts.py: the endpoint shape was "
            "settled markdown-only by #1596 on 2026-09-30 and must not be re-opened "
            "in prose")
        assert needle not in _flat(own), (
            f"{needle!r} is back in this file's prose — the module header, this "
            "section's fence docstring or this node's own text")

    out = subprocess.run(
        ["git", "grep", "-rln", *sum([["-e", n] for n in OPEN_QUESTION_FRAMINGS], [])],
        cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode in (0, 1), out.stderr
    assert not out.stdout.strip(), (
        "these lines still present the endpoint shape as owed to someone:\n"
        + out.stdout)

    # A positive control on the instrument, not on the needles. #1769's lesson: a
    # `git grep -e` whose needle list is silently empty exits 1, which is the exact
    # output a clean sweep produces — so absence alone is not evidence the command
    # could have found anything. The same command shape, run on a phrase that IS
    # shipped, has to name the file; if it does not, the sweep above proves nothing.
    assert OPEN_QUESTION_FRAMINGS, "the framing list is empty, so the sweep proves 0"
    control = subprocess.run(
        ["git", "grep", "-rln", "-e", RULING, "--", "agent_mcp/facts.py"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert control.returncode == 0 and "facts.py" in control.stdout, (
        f"the same `git grep -e` shape cannot find a phrase that IS shipped "
        f"(rc={control.returncode!r}, out={control.stdout.strip()!r}): its silence "
        "above is not a clean bill")


def test_the_endpoint_ruling_trigger_names_the_counter_it_reads():
    """#1871 clause 3: the reopen trigger is bounded, and measurable from code.

    Stated inside `_contradiction_trace`'s own docstring: revisit the graph
    representation only if `contradiction_trace_coverage` reports a trace count above
    0 over the live corpus AND putting the pair in the graph demonstrates a traversal
    gain. Both halves, because the first alone buys the ability to express a row that
    nothing writes.

    Both halves AND the instrument are asserted inside ONE paragraph. The counter's
    name also appears in the paragraph above, describing what counts the records
    today, so the whole-docstring form of this check stayed green while I deleted the
    instrument from the trigger sentence alone — measured, not hypothesised.
    """
    doc = inspect.getdoc(facts._contradiction_trace) or ""
    trigger = [_flat(par) for par in doc.split("\n\n")
               if "above 0" in _flat(par) and "traversal gain" in _flat(par)]
    assert trigger, (
        "no paragraph of the docstring states both halves of the bounded trigger — "
        f"a trace count above 0 AND a demonstrated traversal gain: {doc}")
    assert "contradiction_trace_coverage" in trigger[0], (
        "the trigger states a threshold without the instrument that measures it, so "
        f"nobody can tell whether it has fired: {trigger[0]}")

    counter = ROOT / "scripts" / "memory" / "knowledge-health-report.py"
    assert "def contradiction_trace_coverage" in counter.read_text(
        encoding="utf-8"), (
        "the counter the trigger reads is gone or renamed: the first half of the "
        "trigger can no longer be measured, which silently un-bounds it")


def test_the_endpoint_ruling_sweep_covers_every_endpoint_pointer():
    """#1871 clause 5, first half: a stale pointer fails this file, it does not ship.

    #1596's round — which settled the endpoint shape markdown-only — put pointers in
    two shapes and this item's list named four of the
    five — the section header above these tests was the fifth, unpinned by anything,
    so a round that reworded it would have gone unnoticed and a round that did not
    left a fifth open-question block behind. So the rule is not "these four lines say
    the ruling" but a sweep: any module-level prose block in either file that carries
    an endpoint-shape mark — one of ENDPOINT_POINTER_MARKS —
    has to state the ruling in the same block.

    The count floor is the non-vacuity control, and it is on the denominator rather
    than on a needle: five blocks qualified when this landed, so four blocks found
    means a pointer stopped being read, which is the failure this sweep exists for.
    """
    blocks = [(flat, raw) for flat, raw in _shipped_prose_blocks()
              if any(m in raw for m in ENDPOINT_POINTER_MARKS)]
    assert len(blocks) >= 4, (
        f"the endpoint-shape sweep found only {len(blocks)} block(s) across the two "
        "files. It existed to police five (#: key comment, trace docstring, section "
        "header, fence docstring, two constant comments); a smaller count means a "
        "pointer shape stopped being read, and a sweep that reads none passes "
        "vacuously")
    for flat, raw in blocks:
        assert RULING in flat, (
            "a pointer about the endpoint shape states no ruling, which is the "
            f"stale-pointer state this item closed: {raw}")


def test_the_endpoint_ruling_cites_the_two_self_loop_guard_lines():
    """#1871 clause 5, second half: the guard line numbers in that markdown-only
    ruling's prose are the guards themselves.

    Both shipped pointers cited lines 704-705 of `app/kg_store.py` for guards already
    at 719 and 867 — the same rot class as a stale ruling, arriving on the same
    schedule, and invisible to a reader who trusts the citation while checking whether
    the self-loop refusal is still there. So every line this prose cites for the
    guards must be a line that really holds `if src == tgt`, and BOTH guards must be
    cited: one correct number still leaves the other unfindable.

    Read off the prose blocks, not the whole file: the first version scanned the raw
    source and failed on this file's own history comment, which quotes the rotted
    `704-705` deliberately to explain the rot. A check that grades its own commentary
    is not grading the prose.
    """
    guard_src = (ROOT / "app" / "kg_store.py").read_text(encoding="utf-8")
    guards = {i for i, ln in enumerate(guard_src.splitlines(), 1)
              if "if src == tgt" in ln}
    assert len(guards) == 2, (
        f"expected the two self-loop guards #1596's markdown-only ruling refuses to relax, found "
        f"{sorted(guards)} — this check is out of date with the file it reads")

    cited: set[int] = set()
    for flat in (f for f, _ in _shipped_prose_blocks() if "kg_store" in f):
        cited |= {int(n) for n in re.findall(r"kg_store\.py:(\d+)", flat)}
    assert cited, "neither file cites a guard line any more, so this checks nothing"
    assert cited <= guards, (
        f"prose cites {sorted(cited - guards)} which is not a self-loop guard line — "
        f"the guards are at {sorted(guards)}")
    assert cited == guards, (
        f"prose cites {sorted(cited)} but the guards are at {sorted(guards)}: the "
        "uncited one is unfindable by the next reader")


def test_the_endpoint_ruling_sweeps_every_mention_of_the_item_that_settled_it():
    """#1871 clause 5, third half: naming the item means stating its markdown-only
    decision.

    Narrower than the endpoint-mark sweep and aimed at the rot that produced this
    item: #1596's own round shipped four pointers naming #1593 as the authority for
    the endpoint representation while #1593 sat CLOSED with the ruling unmade. So any
    module-level prose block in either file that names #1593 or #1596 has to state the
    ruling in the same block — a closed item's id may appear beside a decision, never
    beside a question. No exemption list, because an exemption is the hole the next
    stale pointer is written into.

    The floor is the denominator: ten blocks named an item id when this landed, so a
    drop means a block stopped being read rather than the rot being fixed.
    """
    ITEM_IDS = ("#1593", "#1596")
    blocks = [(flat, raw) for flat, raw in _shipped_prose_blocks()
              if any(i in raw for i in ITEM_IDS)]
    assert len(blocks) >= 8, (
        f"the item-id sweep read {len(blocks)} block(s); ten qualified when this "
        "landed, so a smaller number is a prose shape going unread, not a cleanup")
    for flat, raw in blocks:
        assert RULING in flat, (
            "a block names the item that settled the endpoint shape but not what it "
            f"settled it to — the #1593-pointing-at-a-closed-item state again: {raw}")


def test_the_fence_names_itself_permanent_and_points_at_the_trigger():
    """#1871 clause 4: the fence reads as a decision, not as a to-do.

    Asserted about the shipped node rather than restated as a comment: its name must
    not claim the ruling is unmade, and its docstring must say the fence is permanent
    AND name the reopen trigger, so a reader who finds a red run learns to look for a
    WRITE regression instead of permission to relax a guard. Both halves are needed:
    a docstring that only says "permanent" hides the way back, and one that only names
    the trigger reads like an invitation.
    """
    fence = "test_a_resolution_mints_no_edge_row_under_the_markdown_only_ruling"
    node = globals().get(fence)
    assert node is not None, (
        f"{fence} is gone: the fence on the markdown-only ruling has to exist")
    assert "_while_" + "1593_is_unruled" not in fence, (
        "the fence's own name still claims the ruling is unmade")
    doc = _flat(inspect.getdoc(node) or "")
    assert RULING in doc, f"the fence docstring does not name the ruling: {doc}"
    assert "permanent" in doc, (
        f"the fence docstring does not say the fence is permanent: {doc}")
    assert "contradiction_trace_coverage" in doc and "above 0" in doc, (
        "the fence docstring does not name the bounded reopen trigger, so 'permanent' "
        f"has no stated end condition: {doc}")
    assert "and st." not in doc and "has to change" not in doc, (
        f"the fence docstring still tells the next round to edit it: {doc}")
