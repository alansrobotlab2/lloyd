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


def test_both_prompts_ask_each_clause_for_the_test_file_that_pins_it(isolated):
    """The grader downgrades a `met` with no test node to `partial`; a clause
    that names its test file tells the round where that node belongs."""
    write_item(isolated, 7)
    single = " ".join(M.render_prompt(B.item_by_id(7), ledger=S.LEDGER_PATH).split())
    assert "End each clause with the test file that pins it" in single
    assert 'ending with the test file that pins it, e.g. "— tests/test_workers_pool.py"' in single
    group = " ".join(M.GROUP_PROMPT.format(n=2, reason="r", cluster_id="c", anchor_paths="a",
                                           parent="none", items="", max_clauses=B.MAX_CLAUSES,
                                           spawn_cap=1).split())
    assert "each ending with the test file that pins it" in group
    assert 'ACCEPTANCE_CLAUSES: <numbered, one per line, each ending "— tests/<file>.py", or none>' in group
    desc = B.TRIAGE_VERDICT_SCHEMA["properties"]["acceptance_clauses"]["description"]
    assert "tests/<file>.py" in desc


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


def test_clauses_past_the_budget_are_kept_as_text_on_the_row_and_the_item(isolated, monkeypatch):
    """A count alone lost what the dropped clauses said, while the prose
    ACCEPTANCE still stated them."""
    p = write_item(isolated, 12)

    async def turn(prompt, **kw):
        return {"text": _block(), "session_id": "s", "stop_reason": "stop", "num_turns": 9,
                "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    asyncio.run(M.execute(_Item({"group_triage": False})))
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "backlog_triage"][-1]
    assert row["clauses_dropped_text"] == NINE[B.MAX_CLAUSES:]
    text = p.read_text()
    assert "not graded, not part of the contract" in text and NINE[-1] in text


# ── #1869: the witness clause fits the budget or does not go on ───────────────

def _witness_under(tmp_path: Path) -> Path:
    """A witness path the probe itself says is in no git tree."""
    d = tmp_path / "lloyd-data" / "_pipeline" / "reflection"
    d.mkdir(parents=True)
    p = d / "iv-metrics.jsonl"
    p.write_text('{"llm_calls": 164, "miss_rate": null}\n' * 17)
    assert B.in_git_tree(p) is False, p
    return p


def test_a_contract_already_at_the_cap_gains_no_witness_clause(tmp_path, isolated):
    """The rule takes a spare slot or none: a full contract comes back exactly
    as the cap left it, so a generated clause can never evict, truncate or
    reorder one that triage authored.

    The control beside it is the same body and the same helper with one slot
    spare, which does fire — so an unchanged list here is the budget declining
    the clause, not the trigger missing the path.
    """
    witness = _witness_under(tmp_path)
    body = (f"Live run 2026-09-30 over `{witness}` printed 17 rows with "
            "`miss_rate` null in 9.")
    at_cap = NINE[:B.MAX_CLAUSES]
    # The two halves of the guard, over the SAME body: with one slot spare the
    # helper fires, at `MAX_CLAUSES` it does not. Without the first line the
    # second could be the no-witness branch instead of the budget.
    spare = B.add_witness_artifact_clause(at_cap[:B.MAX_CLAUSES - 1], body)
    assert spare[:B.MAX_CLAUSES - 1] == at_cap[:B.MAX_CLAUSES - 1] and \
        len(spare) == B.MAX_CLAUSES and "backlog/data/" in spare[-1], (
        "with one slot spare the same body does fire", spare)
    assert B.add_witness_artifact_clause(list(at_cap), body) == at_cap, (
        "a full contract is returned clause for clause, in order")

    # And through the writer, with the witness in the body `record_verdict`
    # reads: nine authored clauses cap to six, and the rule adds nothing.
    p = write_item(isolated, 21, body=body)
    B.record_verdict(B.item_by_id(21), "confirmed", f"`{witness}` re-measured",
                     acceptance="x", acceptance_clauses=NINE)

    graded = _fm(p)["acceptance_clauses"]
    assert graded == NINE[:B.MAX_CLAUSES], graded
    assert not any("backlog/data/" in c for c in graded), graded


def test_a_contract_with_room_ends_at_or_below_the_cap(tmp_path, isolated):
    """Five authored clauses (the single-triage ask) stay five graded — `MAX_CLAUSES`
    is a ceiling the contract sits under, not a total the witness demand fills up.

    Rewritten by #2289, which moved the generated witness demand out of the graded
    contract and into the item's owed list: this node used to assert the opposite
    (`len(graded) == 6`, the demand last), because the demand arriving AFTER
    `cap_new_clauses` used to mean it took the slot the budget left free. The budget
    accounting it exists for is unchanged — five authored clauses, in their own order
    in front, with the total under the cap — and the demand now shows up in
    `human_clauses` instead, which is asserted here so a reader of this file learns
    where it went rather than finding it missing.
    """
    witness = _witness_under(tmp_path)
    body = f"`{witness}` holds the 17 rows the item quotes."
    five = [f"clause {i}" for i in range(1, B.SINGLE_MAX_CLAUSES + 1)]
    assert len(five) == B.SINGLE_MAX_CLAUSES

    p = write_item(isolated, 22, body=body)
    B.record_verdict(B.item_by_id(22), "confirmed", "real", acceptance="x",
                     acceptance_clauses=five)

    fm = _fm(p)
    graded = fm["acceptance_clauses"]
    assert graded == five, graded
    assert len(graded) == B.SINGLE_MAX_CLAUSES == 5 and len(graded) != B.MAX_CLAUSES, (
        "the graded contract sits under the cap; the witness demand is owed, so it "
        f"no longer fills the sixth slot: {graded[-1][:60]}")
    assert not any("backlog/data/" in c for c in graded), graded
    assert any("backlog/data/iv-metrics.jsonl" in c
               for c in (fm.get("human_clauses") or [])), fm.get("human_clauses")



def test_the_cap_holds_against_a_body_that_would_otherwise_fire(tmp_path, isolated):
    """#2267 clause 5: the brake is tested against a TRIGGERING body, and the
    caller's list comes back unmutated.

    The node above hands the generator the same body in both halves, which proves
    the budget declines the clause. It cannot prove the budget is what declined it
    for a body whose witness is small and uncollided — so this one passes the
    caller's own list object at `MAX_CLAUSES`, over a body that does fire one slot
    lower, and asserts three separate things the older node leaves open: the
    returned list equals the input clause for clause in the same order (no
    generated clause displaces or reorders an authored one), the input list OBJECT
    is still what it was (the generator copies, never appends in place, so a
    caller's list cannot gain a clause behind its back), and the same helper with
    one slot spare does fire — the positive control that keeps an unchanged list
    from being read as a broken trigger.

    The witness is 17 short lines: under `WITNESS_MAX_BYTES`, so it is the cap
    doing the work here and not the new size bound.
    """
    witness = _witness_under(tmp_path)
    assert witness.stat().st_size < B.WITNESS_MAX_BYTES, (
        "otherwise the size bound, not the budget, would be why nothing was added")
    body = (f"Live run 2026-09-30 over `{witness}` printed 17 rows with "
            "`miss_rate` null in 9.")
    authored = NINE[:B.MAX_CLAUSES]

    spare = B.add_witness_artifact_clause(authored[:B.MAX_CLAUSES - 1], body)
    assert len(spare) == B.MAX_CLAUSES and "backlog/data/" in spare[-1], (
        "positive control: this body fires when a slot is free", spare)

    out = B.add_witness_artifact_clause(authored, body)
    assert out == authored, "a full contract is returned clause for clause, in order"
    assert not any("backlog/data/" in c for c in out), out
    assert authored == NINE[:B.MAX_CLAUSES], (
        "the caller's list object was appended to in place")
    assert out is not authored, "the caller gets a copy, not the list it handed in"
    assert out[0] == NINE[0] and out[-1] == NINE[B.MAX_CLAUSES - 1], (
        "no authored clause was displaced from either end")
    # The same thing through the writer: nine authored clauses cap to six and the
    # witness clause is absent, with the cap's own drop record left as the only
    # account of what went.
    p = write_item(isolated, 24, body=body)
    B.record_verdict(B.item_by_id(24), "confirmed", f"`{witness}` re-measured",
                     acceptance="x", acceptance_clauses=list(NINE))
    assert _fm(p)["acceptance_clauses"] == NINE[:B.MAX_CLAUSES], _fm(p)


def test_a_contract_at_the_cap_gains_no_witness_clause_on_either_list(
        tmp_path, isolated):
    """#2289 clause 4: the brake still holds after the destination moved.

    #2289 changed only where the generated witness demand is PUT — the item's owed
    list instead of its graded contract — so what has to be re-pinned is that nothing
    about `MAX_CLAUSES` moved with it. Through the real writer, over a body that
    genuinely names an out-of-tree witness (`_witness_under` builds the file and the
    assertion below has `in_git_tree` say no tree holds it, so only the cap stands
    between this item and a demand): the six authored clauses come back as those six,
    in order, none evicted, truncated or reordered, with no witness demand in the
    graded contract — and none in `human_clauses` either, because the cap gates both
    destinations. An item at its cap has spent its budget: the clauses that fell off
    it are published unnumbered on the item rather than re-added one list over, which
    is exactly the quiet growth past its own prose that #1909's budget exists to stop.

    The second item is the positive control that makes the first half a brake test
    rather than a stopped trigger: one clause shorter, the same body gains nothing in
    the graded contract and gains its demand in the owed list. A change that simply
    stopped generating the demand would pass the at-cap half and fail here.
    """
    witness = _witness_under(tmp_path)
    assert B.in_git_tree(witness) is False, "the witness must sit outside every tree"
    assert witness.stat().st_size < B.WITNESS_MAX_BYTES, (
        "otherwise the size bound, not the budget, would be why nothing was added")
    body = f"the row came from `{witness}`, and nothing here names where those bytes go"
    at_cap = [f"route {i}: the mechanism is pinned by a test — tests/test_x{i}.py"
              for i in range(B.MAX_CLAUSES)]
    assert len(at_cap) == B.MAX_CLAUSES == 6

    p = write_item(isolated, 66113, body=body)
    B.record_verdict(B.item_by_id(66113), "confirmed", "re-measured, unchanged",
                     acceptance="x", acceptance_clauses=list(at_cap))
    fm = _fm(p)
    assert list(fm["acceptance_clauses"]) == at_cap, (
        "at the cap the graded contract comes back UNCHANGED: no generated clause, "
        "and no authored one evicted, truncated or reordered")
    assert not any("backlog/data/" in c for c in fm["acceptance_clauses"]), fm
    assert not any("backlog/data/" in c for c in (fm.get("human_clauses") or [])), (
        f"the owed list gains no witness clause at the cap either: "
        f"{fm.get('human_clauses')}")

    one_short = at_cap[:B.MAX_CLAUSES - 1]
    q = write_item(isolated, 66114, body=body)
    B.record_verdict(B.item_by_id(66114), "confirmed", "re-measured, unchanged",
                     acceptance="x", acceptance_clauses=list(one_short))
    fm2 = _fm(q)
    assert list(fm2["acceptance_clauses"]) == one_short, (
        "with a slot spare the graded contract still gains no witness demand — that "
        "is the whole of #2289")
    assert any("backlog/data/iv-metrics.jsonl" in c
               for c in (fm2.get("human_clauses") or [])), (
        "and the demand is owed, which is what proves the at-cap half is a brake and "
        f"not a generator that stopped firing: {fm2.get('human_clauses')}")

