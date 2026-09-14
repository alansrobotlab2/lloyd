"""The clause budget: a contract a round can meet.

Every acceptance clause is graded on its own at the review rung, and one
unmet clause refuses the round. Measured over the week to 2026-09-14: 109 of
147 reviews refused, a per-clause not-met rate of 13%, so six clauses pass
together ~43% of the time and twelve ~19%. First reviews by clause count
promoted 3/16 at <=4, 21/52 at 5-8 and 3/11 at 9+, while 62 confirmations in
the week carried 9-12 clauses and umbrellas of 8-12 were 56 of 92 `up_next`.

So a contract triage writes is capped at `MAX_CLAUSES` (6) on both parse
paths and in `record_verdict`, the single prompt asks for at most
`SINGLE_MAX_CLAUSES` (5), the verdict row records what was dropped, and group
triage takes at most four members. A contract already on disk is read at the
old bound, or a round would be graded — and an item closed — on half of it.
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


def write_item(d: Path, item_id, *, status="draft", days_old=100, tags=("backlog",),
               name=None, body="Do the thing.", extra=None) -> Path:
    name = name or f"Item {item_id}"
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": "lloyd", "tags": list(tags), **(extra or {})}
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


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text())[0]


NINE = [f"behaviour {i} is pinned by a test" for i in range(1, 10)]


def _block(clauses=NINE) -> str:
    rows = "\n".join(f"{i}. {c}" for i, c in enumerate(clauses, 1))
    return ("...\nVERDICT: confirmed\nSURFACE: code\nCHECK: grep -n x\nEVIDENCE: it holds.\n"
            f"ACCEPTANCE: all of it\nACCEPTANCE_CLAUSES:\n{rows}\nHUMAN_CLAUSES: none\nSPAWNED: none\n")


def test_the_budget_numbers():
    assert B.MAX_CLAUSES == 6 and B.SINGLE_MAX_CLAUSES == 5 and B.READ_MAX_CLAUSES == 12
    assert M.DEFAULT_GROUP_MAX_ITEMS == 4


def test_both_prompts_state_the_cap(isolated):
    write_item(isolated, 7)
    single = M.render_prompt(B.item_by_id(7), ledger=S.LEDGER_PATH)
    assert f"At most {B.SINGLE_MAX_CLAUSES} clauses" in single
    assert "Three to six" not in single
    group = M.GROUP_PROMPT.format(n=2, reason="r", cluster_id="c", anchor_paths="a", parent="none",
                                  items="", max_clauses=B.MAX_CLAUSES, spawn_cap=1)
    assert f"at most {B.MAX_CLAUSES}" in group


@pytest.mark.parametrize("path", ["regex", "structured"])
def test_an_overlong_verdict_is_capped_and_says_how_much(path):
    if path == "regex":
        parsed = M.parse_verdict(_block())
    else:
        parsed = M.parse_verdict("", {"verdict": "confirmed", "surface": "code", "check": "c",
                                      "evidence": "e", "acceptance": "a",
                                      "acceptance_clauses": NINE, "spawned": []})
    assert parsed["source"] == path
    assert parsed["acceptance_clauses"] == NINE[:B.MAX_CLAUSES]
    assert parsed["clauses_dropped"] == 3
    assert M.parse_verdict(_block(NINE[:4]))["clauses_dropped"] == 0


def test_a_single_triage_writes_the_capped_contract_and_records_the_drop(isolated, monkeypatch):
    p = write_item(isolated, 7)

    async def turn(prompt, **kw):
        return {"text": _block(), "session_id": "s", "stop_reason": "stop", "num_turns": 9,
                "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(M.execute(_Item({"group_triage": False})))
    assert out["verdict"] == "confirmed"
    assert _fm(p)["acceptance_clauses"] == NINE[:B.MAX_CLAUSES]
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "backlog_triage"][-1]
    assert row["clauses_dropped"] == 3 and len(row["acceptance_clauses"]) == B.MAX_CLAUSES


def test_record_verdict_is_the_hard_cap_whatever_it_is_handed(isolated):
    p = write_item(isolated, 8)
    B.record_verdict(B.item_by_id(8), "confirmed", "real", acceptance="x", acceptance_clauses=NINE)
    assert len(_fm(p)["acceptance_clauses"]) == B.MAX_CLAUSES


def test_an_umbrella_is_capped_and_its_row_records_the_drop(isolated, monkeypatch):
    for i in (1, 2):
        write_item(isolated, i, days_old=2, tags=("backlog", "spawned-by-triage"))
    CL.write_clusters({"schema": 1, "clusters": [{"id": CL.cluster_id([1, 2]), "item_ids": [1, 2],
                                                  "reason": "r", "anchor_paths": [], "duplicates": []}]})
    rows = "\n".join(f"{i}. {c}" for i, c in enumerate(NINE[:8], 1))
    text = ("GROUP_VERDICTS:\n#1: fold — a\n#2: fold — b\nUMBRELLA: #50\nUMBRELLA_MEMBERS: #1 #2\n"
            f"SURFACE: code\nCHECK: c\nEVIDENCE: e\nACCEPTANCE: a\nACCEPTANCE_CLAUSES:\n{rows}\nSPAWNED: none\n")

    async def turn(prompt, **kw):
        write_item(isolated, 50, days_old=0, tags=("backlog", "umbrella", "spawned-by-triage"))
        return {"text": text, "session_id": "s", "stop_reason": "stop", "num_turns": 9, "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2})))
    assert out["umbrella_id"] == 50
    assert _fm(next(isolated.glob("50-*.md")))["acceptance_clauses"] == NINE[:B.MAX_CLAUSES]
    row = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e["event"] == "backlog_triage" and e["item_id"] == 50][-1]
    assert row["clauses_dropped"] == 2


def test_group_triage_takes_four_members_by_default(isolated, monkeypatch):
    ids = list(range(1, 8))
    for i in ids:
        write_item(isolated, i, days_old=20 - i)
    CL.write_clusters({"schema": 1, "clusters": [{"id": CL.cluster_id(ids), "item_ids": ids,
                                                  "reason": "r", "anchor_paths": [], "duplicates": []}]})
    seen = {}

    async def turn(prompt, **kw):
        seen["prompt"] = prompt
        return {"text": "no block", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    asyncio.run(M.execute(_Item({"group_triage": True, "group_min_items": 2})))
    assert seen["prompt"].count("<item id=") == 4
    assert B.select_cluster(S.LEDGER_PATH, CL.load_clusters(), min_size=2) is not None
    assert len(B.select_cluster(S.LEDGER_PATH, CL.load_clusters(), min_size=2)[1]) == 4


def test_the_configured_group_size_is_four():
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    assert raw["workers"]["sources"]["autotriage"]["group_max_items"] == 4


def test_a_contract_already_on_disk_is_read_whole(isolated):
    """56 umbrellas confirmed before the cut carry up to 12. Reading them at 6
    would grade a round against half the contract, close the item on the half
    it met, and `amend_clause` would write the truncated half back."""
    twelve = [f"clause {i}" for i in range(1, 13)]
    p = write_item(isolated, 9, status="up_next", extra={"acceptance_clauses": twelve})
    assert B.acceptance_clauses_of(None, _fm(p)) == twelve
    S.append_event({"event": "review", "round_id": "SM_A", "item_id": 9, "ok": True,
                    "clauses": [{"clause": 10, "verdict": "unsatisfiable"}]}, path=S.LEDGER_PATH)
    B.amend_clause(9, 10, "clause ten, satisfiable", "the grader said so", round_id="SM_A")
    after = _fm(p)["acceptance_clauses"]
    assert len(after) == 12 and after[9] == "clause ten, satisfiable" and after[11] == "clause 12"
