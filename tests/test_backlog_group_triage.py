"""Group triage: one turn over a cluster, and the umbrella it leaves behind.

The single-item pass could only ever close one item per run and it filed
~2 for each; the loop manufactured sibling clusters (84 parents, 282
children by 2026-09-11) and had no way to reassemble them. A group triage
reads a cluster from `clusters.json`, closes the duplicates, retires the
stale, and folds what remains into ONE umbrella item that the implement
loop takes as ordinary confirmed work — and whose landing closes the
members. Nothing here adds a status; `group`/`members`/`duplicate_of` are
frontmatter keys and `umbrella`/`grouped` are tags.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, cluster as CL, state as S
from workers.sources import _common as C
from workers.sources import autotriage as M


def write_item(d: Path, item_id, *, status="draft", days_old=10, tags=("backlog",),
               name=None, body="Do the thing.", board="lloyd") -> Path:
    name = name or f"Item {item_id}"
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": board, "tags": list(tags)}
    p = d / f"{item_id}-{name.lower().replace(' ', '-')[:30]}.md"
    p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
                 encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(CL, "CLUSTERS_PATH", tmp_path / "clusters.json")
    return d


class _Item:
    def __init__(self, payload=None):
        self.payload = payload or {}


def _clusters(*groups, duplicates=None):
    out = []
    for ids in groups:
        out.append({"id": CL.cluster_id(ids), "item_ids": sorted(ids), "reason": "paths on 3 pair(s)",
                    "anchor_paths": ["loop.py"], "parent": 549,
                    "duplicates": [d for d in (duplicates or []) if d[0] in ids and d[1] in ids]})
    return {"schema": 1, "clusters": out}


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text())[0]


def _path(d: Path, item_id) -> Path:
    return next(d.glob(f"{item_id}-*.md"))


# ── which cluster ───────────────────────────────────────────────────────────

def test_select_cluster_needs_min_size_untriaged_ungrouped_survivors(isolated):
    for i in (1, 2, 3, 4, 5, 6):
        write_item(isolated, i)
    S.append_event({"event": "backlog_triage", "item_id": 3, "verdict": "stale"}, path=S.LEDGER_PATH)
    B.update_frontmatter(_path(isolated, 4), {"group": 99})
    B.tag_item(5, add=(B.NEEDS_HUMAN_TAG,))
    B.tag_item(6, add=("umbrella",))
    pick = B.select_cluster(S.LEDGER_PATH, _clusters([1, 2, 3, 4, 5, 6]), min_size=2)
    assert [m.id for m in pick[1]] == [1, 2]
    assert B.select_cluster(S.LEDGER_PATH, _clusters([1, 2, 3, 4, 5, 6]), min_size=3) is None


def test_select_cluster_trims_to_max_size_keeping_duplicate_pairs_together(isolated):
    for i in range(1, 11):
        write_item(isolated, i, days_old=100 - i)   # 1 is the oldest
    pick = B.select_cluster(S.LEDGER_PATH, _clusters(list(range(1, 11)), duplicates=[[9, 10]]),
                            min_size=2, max_size=4)
    assert [m.id for m in pick[1]] == [1, 2, 9, 10], "the duplicate pair first, then oldest"


def test_a_fully_judged_cluster_is_never_retaken_but_new_members_are(isolated):
    for i in (1, 2, 3):
        write_item(isolated, i)
    S.append_event({"event": "backlog_group_triage", "cluster_id": CL.cluster_id([1, 2, 3]),
                    "judged": {"1": "keep", "2": "keep", "3": "keep"}}, path=S.LEDGER_PATH)
    assert B.select_cluster(S.LEDGER_PATH, _clusters([1, 2, 3]), min_size=2) is None
    write_item(isolated, 4); write_item(isolated, 5)
    pick = B.select_cluster(S.LEDGER_PATH, _clusters([1, 2, 3, 4, 5]), min_size=2)
    assert [m.id for m in pick[1]] == [4, 5], "only the unjudged members of a recomputed cluster"


def test_the_largest_surviving_cluster_wins(isolated):
    for i in range(1, 8):
        write_item(isolated, i)
    pick = B.select_cluster(S.LEDGER_PATH, _clusters([1, 2], [3, 4, 5, 6], [7]), min_size=2)
    assert [m.id for m in pick[1]] == [3, 4, 5, 6]


def test_quarantined_self_spawned_items_are_admitted_to_group_mode(isolated):
    for i in (1, 2, 3):
        write_item(isolated, i, days_old=0, tags=("backlog", "spawned-by-triage"))
    assert B.select_candidate(S.LEDGER_PATH) is None, "held from the single pool"
    pick = B.select_cluster(S.LEDGER_PATH, _clusters([1, 2, 3]), min_size=2)
    assert [m.id for m in pick[1]] == [1, 2, 3]


# ── the contract ────────────────────────────────────────────────────────────

def test_group_schema_is_built_from_the_one_list():
    props = B.GROUP_TRIAGE_SCHEMA["properties"]
    assert props["items"]["items"]["properties"]["verdict"]["enum"] == list(B.GROUP_VERDICTS)
    assert set(B.RETIRING) <= set(B.GROUP_VERDICTS)
    assert props["umbrella"]["properties"]["surface"]["enum"] == list(B.SURFACES)
    assert B.FOLDED not in B.VERDICTS, "a fold is a state, not a judgement on the premise"


BLOCK = """...work...

GROUP_VERDICTS:
#1: duplicate_of #2 — same finding, #2 has the repro
#2: fold — the real one
#3: dup #2 — restates it
#4: already done — fixed in abc123
#5 -> fold
#6: keep — separate work
#99: fold — not in the cluster
UMBRELLA: #50
UMBRELLA_MEMBERS: #2 #5
SURFACE: code
CHECK: grep -n foo
EVIDENCE: The floor is dead.
ACCEPTANCE: the floor gates again
ACCEPTANCE_CLAUSES:
1. a test pins the floor
2. the ledger records it
SPAWNED: none
"""


def test_parse_group_verdict_regex_reads_each_spelling_and_fills_unjudged():
    out = M.parse_group_verdict(BLOCK, None, [1, 2, 3, 4, 5, 6, 7])
    assert out["source"] == "regex"
    v = out["items"]
    assert v[1] == {"verdict": "duplicate_of", "duplicate_of": 2, "evidence": "same finding, #2 has the repro"}
    assert v[3]["verdict"] == "duplicate_of" and v[3]["duplicate_of"] == 2
    assert v[4]["verdict"] == "already_done" and v[2]["verdict"] == "fold" and v[5]["verdict"] == "fold"
    assert v[6]["verdict"] == "keep"
    assert 99 not in v, "an id outside the cluster is dropped"
    assert out["unjudged"] == [7] and v[7]["verdict"] == "keep"
    u = out["umbrella"]
    assert u["item_id"] == 50 and u["members"] == [2, 5] and u["surface"] == "code"
    assert u["acceptance_clauses"] == ["a test pins the floor", "the ledger records it"]
    assert out["spawned"] == []


def test_parse_group_verdict_structured_wins_and_degrades_to_keep():
    structured = {"items": [{"item_id": 1, "verdict": "stale", "duplicate_of": 0, "evidence": "gone"},
                            {"item_id": 2, "verdict": "bogus", "duplicate_of": 0, "evidence": ""}],
                  "umbrella": {"item_id": 0, "members": [], "surface": "vault", "check": "",
                               "evidence": "", "acceptance": "", "acceptance_clauses": []},
                  "spawned": [7]}
    out = M.parse_group_verdict(BLOCK, structured, [1, 2])
    assert out["source"] == "structured" and out["items"][1]["verdict"] == "stale"
    assert out["items"][2]["verdict"] == "keep" and out["unjudged"] == [2]
    assert out["umbrella"]["surface"] == "vault" and out["spawned"] == [7]
    assert M.parse_group_verdict("no block here", None, [1]) is None


def test_duplicate_chains_resolve_to_the_survivor_and_cycles_become_keep():
    v = {1: {"verdict": "duplicate_of", "duplicate_of": 2, "evidence": ""},
         2: {"verdict": "duplicate_of", "duplicate_of": 3, "evidence": ""},
         3: {"verdict": "fold", "duplicate_of": 0, "evidence": ""},
         4: {"verdict": "duplicate_of", "duplicate_of": 5, "evidence": ""},
         5: {"verdict": "duplicate_of", "duplicate_of": 4, "evidence": ""},
         6: {"verdict": "duplicate_of", "duplicate_of": 77, "evidence": ""}}
    out = B._resolve_duplicates(v, {1, 2, 3, 4, 5, 6})
    assert out[1]["duplicate_of"] == 3 and out[2]["duplicate_of"] == 3
    assert out[4]["verdict"] == "keep" and out[5]["verdict"] == "keep", "a cycle"
    assert out[6]["verdict"] == "keep", "a target outside the cluster"


# ── one run, end to end ────────────────────────────────────────────────────

def _turn(text, *, files_umbrella=True, backlog_dir=None, structured=None, stop_reason="stop"):
    async def fake(prompt, **kw):
        fake.calls.append({"prompt": prompt, **kw})
        if files_umbrella:
            write_item(backlog_dir, 50, days_old=0, name="Umbrella for 2 5",
                       tags=("backlog", "umbrella", "spawned-by-triage"),
                       body="Umbrella for #2 #5, formed by automod group triage.")
        return {"text": text, "session_id": "sess_group", "stop_reason": stop_reason,
                "num_turns": 30, "errors": [], "structured": structured}
    fake.calls = []
    return fake


def _seed(isolated, ids=(1, 2, 3, 4, 5, 6, 7)):
    for i in ids:
        write_item(isolated, i, days_old=2, tags=("backlog", "spawned-by-autocode"))
    CL.write_clusters(_clusters(list(ids)))


def test_a_group_run_closes_duplicates_retires_folds_and_confirms_the_umbrella(isolated, monkeypatch):
    _seed(isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", _turn(BLOCK, backlog_dir=isolated))
    out = asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2})))
    assert out["status"] == "success"
    assert out["duplicates"] == 2 and out["retired"] == 1 and out["folded"] == 2 and out["kept"] == 2
    assert out["umbrella_id"] == 50

    # Duplicates: done, pointing at the survivor, with a stale triage row.
    fm1 = _fm(_path(isolated, 1))
    assert fm1["status"] == "done" and fm1["duplicate_of"] == 2 and fm1["autotriage_retired"] == "stale"
    assert "duplicate of #2" in _path(isolated, 1).read_text()
    # Retired: done.
    assert _fm(_path(isolated, 4))["status"] == "done"
    # Folded: draft, grouped, out of both pools.
    fm2 = _fm(_path(isolated, 2))
    assert fm2["status"] == "draft" and fm2["group"] == 50 and "grouped" in fm2["tags"]
    assert 2 not in {i.id for i in B.triage_pool(S.LEDGER_PATH)[0]}
    assert B.select_candidate(S.LEDGER_PATH).id in (6, 7), "kept items are released to single triage"
    # Umbrella: confirmed, up_next, members and clauses on disk, selectable.
    fm50 = _fm(_path(isolated, 50))
    assert fm50["status"] == "up_next" and fm50["members"] == [2, 5] and "umbrella" in fm50["tags"]
    assert fm50["acceptance_clauses"] == ["a test pins the floor", "the ledger records it"]
    pair = B.select_confirmed(S.LEDGER_PATH)
    assert pair[0].id == 50 and pair[1]["umbrella"] is True and pair[1]["members"] == [2, 5]
    # The ledger: one triage row per judged member (keep writes none), one summary.
    rows = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "backlog_triage"]
    assert sorted(e["item_id"] for e in rows) == [1, 2, 3, 4, 5, 50]
    assert {e["item_id"]: e["verdict"] for e in rows}[2] == B.FOLDED
    summary = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "backlog_group_triage"][-1]
    assert summary["judged"] == {"1": "duplicate_of", "2": "fold", "3": "duplicate_of", "4": "already_done",
                                 "5": "fold", "6": "keep", "7": "keep"}
    assert summary["umbrella_id"] == 50 and summary["verdict_source"] == "regex"
    # A second run finds nothing left to group.
    assert B.select_cluster(S.LEDGER_PATH, CL.load_clusters(), min_size=2) is None


def test_the_umbrella_is_never_merged_into_a_member_by_dedupe(isolated, monkeypatch):
    """The umbrella's description names its members, so it matches them
    all; the MCP writer refuses to merge an `umbrella`-tagged write."""
    from agent_mcp import backlog as BL
    assert "umbrella" in BL._NEVER_MERGE_TAGS


def test_fold_without_an_umbrella_on_disk_is_recorded_as_keep(isolated, monkeypatch):
    _seed(isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", _turn(BLOCK, files_umbrella=False, backlog_dir=isolated))
    out = asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2})))
    assert out["folded"] == 0 and out["kept"] == 4 and out["umbrella_id"] is None
    assert _fm(_path(isolated, 2))["status"] == "draft" and "group" not in _fm(_path(isolated, 2))
    summary = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "backlog_group_triage"][-1]
    assert summary["umbrella_missing"] is True


def test_a_single_fold_is_a_keep_and_the_umbrella_is_not_confirmed(isolated, monkeypatch):
    block = BLOCK.replace("#5 -> fold", "#5: keep — on its own")
    _seed(isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", _turn(block, backlog_dir=isolated))
    out = asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2})))
    assert out["folded"] == 0 and out["umbrella_id"] is None
    assert _fm(_path(isolated, 50))["status"] == "draft", "filed but not confirmed"
    assert B.select_confirmed(S.LEDGER_PATH) is None


def test_running_out_of_budget_twice_abandons_the_cluster_writing_nothing_on_items(isolated, monkeypatch):
    _seed(isolated)
    monkeypatch.setattr(C, "run_prompt_in_session",
                        _turn("...never reached the block", files_umbrella=False,
                              backlog_dir=isolated, stop_reason="max_turns"))
    out = asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2})))
    assert out["status"] == "skipped" and "incomplete" in out["summary"]
    assert B.select_cluster(S.LEDGER_PATH, CL.load_clusters(), min_size=2) is not None, "comes back once"
    out = asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2})))
    assert out["status"] == "success" and "abandoned" in out["summary"]
    assert B.select_cluster(S.LEDGER_PATH, CL.load_clusters(), min_size=2) is None
    assert all(_fm(_path(isolated, i))["status"] == "draft" and "group" not in _fm(_path(isolated, i))
               for i in range(1, 8))
    assert B.triaged_ids(S.LEDGER_PATH) == {}


def test_the_kill_switch_runs_the_single_path_unchanged(isolated, monkeypatch):
    _seed(isolated, ids=(1, 2, 3))
    write_item(isolated, 9, days_old=300)   # a real single candidate
    fake = _turn("VERDICT: stale\nSURFACE: code\nCHECK: c\nEVIDENCE: e\nACCEPTANCE: none\nSPAWNED: none\n",
                 files_umbrella=False, backlog_dir=isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    out = asyncio.run(M.execute(_Item({"group_triage": False})))
    assert out["item_id"] == 9 and out["verdict"] == "stale"
    assert "<cluster" not in fake.calls[0]["prompt"]


def test_a_qualifying_cluster_wins_over_the_single_pool(isolated, monkeypatch):
    _seed(isolated, ids=(1, 2, 3))
    write_item(isolated, 9, days_old=300)
    fake = _turn(BLOCK, files_umbrella=False, backlog_dir=isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    out = asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2})))
    assert out.get("cluster_id") == CL.cluster_id([1, 2, 3])
    assert "<cluster" in fake.calls[0]["prompt"] and fake.calls[0]["final_schema"] is B.GROUP_TRIAGE_SCHEMA


def test_the_group_prompt_fits_the_body_budget_for_eight_items(isolated, monkeypatch):
    ids = tuple(range(1, 9))
    for i in ids:
        write_item(isolated, i, days_old=2, body="x" * 20_000)
    CL.write_clusters(_clusters(list(ids)))
    fake = _turn("nothing", files_umbrella=False, backlog_dir=isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2, "body_chars": 30_000})))
    prompt = fake.calls[0]["prompt"]
    assert prompt.count("<item id=") == 8
    assert len(prompt) < 30_000 + 8 * 400 + len(M.GROUP_PROMPT), "per-item cap holds"


def test_unfold_umbrella_releases_members_and_records_why(isolated):
    write_item(isolated, 50, status="up_next", tags=("backlog", "umbrella"))
    B.update_frontmatter(_path(isolated, 50), {"members": [2, 5]})
    for i in (2, 5):
        write_item(isolated, i)
        B.update_frontmatter(_path(isolated, i), {"group": 50}, add_tags=("grouped",))
    with pytest.raises(ValueError):
        B.unfold_umbrella(50, "")
    out = B.unfold_umbrella(50, "the umbrella conflated two areas")
    assert out["released"] == [2, 5]
    assert "group" not in _fm(_path(isolated, 2)) and "grouped" not in _fm(_path(isolated, 2))["tags"]
    assert _fm(_path(isolated, 50))["members"] == []
    assert B.select_candidate(S.LEDGER_PATH).id in (2, 5)
    assert S.read_events(path=S.LEDGER_PATH)[-1]["event"] == "backlog_group_unfold"
