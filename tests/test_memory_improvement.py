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
import importlib.util
import inspect
import json
import re
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


def test_collect_signals_dedupes_by_entity(world):
    facts_root, st, vault_root = world
    _write_facts(facts_root, "TTS", "state",
                 [{"fact": "TTS built-in voices return 500 errors.", "created_at": _days_ago(4)}])
    _reindex(st, facts_root)
    (vault_root / "memory" / "corrections.md").write_text(
        "## TTS regression\n**Correction:** TTS broke again.\n", encoding="utf-8")
    signals = fi.collect_signals(sources=("corrections", "drift"))
    assert sum(1 for s in signals if s["entity"] == "TTS") == 1


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
    """Backdate one entity's fact file so its drift mtime is explicit.

    Fails when `entity` matches no fact file. A helper that silently no-ops on a
    missed target leaves that entity freshly written, so a fixture meaning "this
    one is 30 days old" quietly becomes "this one is new" and the test it feeds
    keeps passing while pinning nothing — the review rung of this item's first
    round found the helper doing exactly that, and clause 5 is the guard.
    """
    import os
    stamp = (datetime.now() - timedelta(days=days, hours=hours)).timestamp()
    paths = list((facts_root / entity).glob("*.md"))
    assert paths, (
        f"_aged({entity!r}) matched no *.md under {facts_root}, so the intended "
        f"backdating never happened and that entity is still freshly written")
    for path in paths:
        os.utime(path, (stamp, stamp))


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


# ── 2. the loop: evidence + reason, dry-run by default ───────────────────────

def test_equal_confidence_contradiction_needs_a_time_order_reason(world):
    """Detector says "these two disagree", confidences tie → the tie is broken
    by created_at, and only when the loser is clearly older."""
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30), "confidence": 0.9},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
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
    loser = "Lloyd built-in voices are working and returning 200 OK."
    _write_facts(facts_root, "Lloyd", "state", [
        {"fact": loser, "created_at": _days_ago(30)},
        {"fact": "Lloyd built-in voices are broken and returning 500 errors.",
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
        {"fact": "MULTI cache eviction is enabled.", "created_at": _days_ago(20)},
        {"fact": "MULTI cache eviction is disabled.", "created_at": _days_ago(2)},
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
    assert act["loser_fact"] == "MULTI cache eviction is enabled.", act


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
        {"fact": "MIXED tailnet relay is working.", "created_at": _days_ago(18)},
        {"fact": "MIXED tailnet relay is broken.", "created_at": _days_ago(3)},
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
    assert plan["actions"][0]["loser_fact"] == "MIXED tailnet relay is working.", plan

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
        {"fact": "SPLIT ingestion watcher is enabled.", "created_at": _days_ago(21)},
        *_filler("SPLIT", 19),
    ])
    _write_facts(facts_root, "SPLIT", "event", [
        {"fact": "SPLIT ingestion watcher is disabled.", "created_at": _days_ago(4)},
        *_filler("SPLIT", 19),
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("SPLIT")
    assert plan["refused"] is False, plan
    assert plan["scan_scope"] == "entity", plan
    assert plan["checked"] == 40, plan                    # both categories, one scan
    assert len(plan["actions"]) == 1, plan                # the cross-category pair
    assert plan["actions"][0]["loser_fact"] == "SPLIT ingestion watcher is enabled.", plan


def test_run_is_dry_run_by_default_and_acts_on_request(world):
    facts_root, st, _ = world
    _write_facts(facts_root, "TTS", "state", [
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
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
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
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
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
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

    def _fake_report(limit):
        seen["limit"] = limit
        return 0.5

    monkeypatch.setattr(fi, "_fact_entity_recall", _fake_report)
    rec = fi.run_improvement(sources=("drift",), days=3, report_eval=True, eval_limit=7)
    assert rec["fact_entity_recall"] == 0.5
    assert seen["limit"] == 7
    skipped = fi.run_improvement(sources=("drift",), days=3)
    assert skipped.get("fact_entity_recall") is None, "off by default"


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
        {"fact": "The build is working.", "confidence": 0.9, "created_at": _days_ago(30)},
        {"fact": "The build is broken.", "confidence": 0.9, "created_at": _days_ago(2)},
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
    pair has a gap of 0.0 — below any floor — and its basis is write order, so
    a 30-day gap still plans a `superseded` expiry exactly as it did before
    #701. A pair of whole-word opposing terms, so `REQUIRE_OPPOSING_TERMS` lets
    it through and only the basis is in question."""
    facts_root, st, _ = world
    _write_facts(facts_root, "Ageway", "state", [
        {"fact": "The build is working.", "confidence": 0.9, "created_at": _days_ago(30)},
        {"fact": "The build is broken.", "confidence": 0.9, "created_at": _days_ago(2)},
    ])
    _reindex(st, facts_root)
    plan = fi.plan_entity("Ageway")
    assert len(plan["actions"]) == 1, plan["actions"]
    action = plan["actions"][0]
    assert action["kind"] == "superseded"
    assert action["loser_fact"] == "The build is working."
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
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
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
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
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
    scheduler dispatches (autonomy.py:419), so a task sitting in `draft` is a
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
        {"fact": "TTS built-in voices are working and returning 200 OK.",
         "created_at": _days_ago(30)},
        {"fact": "TTS built-in voices are broken and returning 500 errors.",
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
    import subprocess
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
