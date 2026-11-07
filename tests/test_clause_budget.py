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

import ast
import asyncio
import re
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
    # #2513: the witness emitter now asks the VAULT whether a name is taken before it
    # writes the demand, so a node here that asserts the emitted target
    # (`backlog/data/iv-metrics.jsonl`) was reading the real `~/obsidian` — where that
    # witness was delivered long ago, so the name is genuinely taken and the emitter
    # answered with a dated sibling. An empty temporary vault is the fixture's own
    # claim: these nodes are about the BUDGET, and the name they expect is the free
    # name. Same isolation `tests/test_witness_artifact_clause.py` applies.
    (tmp_path / "vault" / "backlog" / "data").mkdir(parents=True)
    monkeypatch.setattr("app.paths.VAULT_ROOT", tmp_path / "vault")
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


# ── #2382: the cap gates the over-bound note as hard as it gates the clause ─────

def _witness_over(tmp_path: Path) -> Path:
    """An on-disk witness over `WITNESS_MAX_BYTES` in no git tree: the #2382 shape.

    Same relation to the bound as the corpus's own example —
    `~/.local/state/lloyd-automod/promotions.jsonl`, 30,785,438 bytes against
    `WITNESS_MAX_BYTES = 1_048_576` — at a size a test may write.
    """
    d = tmp_path / "state" / "lloyd-automod"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "promotions.jsonl"
    p.write_bytes(b'{"row": 1}\n' * (B.WITNESS_MAX_BYTES // 10 + 1))
    assert p.stat().st_size > B.WITNESS_MAX_BYTES, "the fixture is over the bound"
    assert B.in_git_tree(p) is False, p
    return p


def test_the_cap_gates_the_over_bound_note_as_hard_as_the_clause(tmp_path, isolated):
    """#2382 clause 5: a note gathered inside the token loop may not escape the cap.

    This is the trap the new note walks straight into. `MAX_CLAUSES` was checked AFTER
    the loop, on the strength of a loop that only ever produced one candidate; a note
    accumulated per-token and appended before that line would have restarted exactly
    the quiet growth past an item's own prose that the budget exists to stop (#1909),
    and it would have done it on the list nobody grades, which is worse — #2289
    clause 4 already closed that door for the copy demand, and a second destination
    discovered later is precisely how such a gate stops being a gate.

    So: at `MAX_CLAUSES`, over a body that genuinely cites an over-bound store, the
    call returns the contract clause for clause AND the caller's own list object
    unmutated, and `owed` is empty — no note. One slot spare, the same body does gain
    the note: without that control an empty `owed` reads as a trigger that broke.

    The writer halves matter because the cap the round is actually protected by is the
    one applied there: `cap_new_clauses` brings nine authored clauses to six, and the
    note must not arrive afterwards into `human_clauses` to re-grow what the cap just
    trimmed. The at-cap item gains nothing on either list; the one-short item gains
    the note owed, which is the same positive control one level up.
    """
    big = _witness_over(tmp_path)
    body = f"the row came from `{big}`, and nothing here names where those bytes go"
    at_cap = [f"route {i}: the mechanism is pinned by a test — tests/test_x{i}.py"
              for i in range(B.MAX_CLAUSES)]
    assert len(at_cap) == B.MAX_CLAUSES == 6
    before = list(at_cap)

    clauses, owed = B.add_witness_artifact_clause(at_cap, body, route_to_owed=True)
    assert clauses == before, (
        f"at the cap the contract is returned clause for clause, in order: {clauses}")
    assert at_cap == before, "the caller's list object was appended to in place"
    assert not any("backlog/data/" in c for c in clauses), clauses
    assert owed == [], (
        f"the cap gates the note too, so nothing is owed at all: {owed}")

    short = at_cap[:B.MAX_CLAUSES - 1]
    clauses2, owed2 = B.add_witness_artifact_clause(list(short), body,
                                                    route_to_owed=True)
    assert clauses2 == short, clauses2
    assert len(owed2) == 1 and "byte ceiling" in owed2[0] and str(big) in owed2[0], (
        f"positive control: one slot spare, the same body gains the note: {owed2}")

    p = write_item(isolated, 66115, body=body)
    B.record_verdict(B.item_by_id(66115), "confirmed", "re-measured, unchanged",
                     acceptance="x", acceptance_clauses=list(at_cap))
    fm = _fm(p)
    assert list(fm["acceptance_clauses"]) == at_cap, fm
    assert not any("WITNESS_MAX_BYTES" in c for c in (fm.get("human_clauses") or [])), (
        f"an item at its cap is owed no note either: {fm.get('human_clauses')}")

    q = write_item(isolated, 66116, body=body)
    B.record_verdict(B.item_by_id(66116), "confirmed", "re-measured, unchanged",
                     acceptance="x", acceptance_clauses=list(short))
    fm2 = _fm(q)
    assert list(fm2["acceptance_clauses"]) == short, fm2
    assert any("WITNESS_MAX_BYTES" in c for c in (fm2.get("human_clauses") or [])), (
        f"and with a slot free the note is owed, so the at-cap half is a brake: "
        f"{fm2.get('human_clauses')}")



# ── a contract that numbers itself (#3002) ─────────────────────────────────
# The number a clause is GRADED under is its position: `build_vault_prompt`
# renders `f"{i}. {clauses[i - 1]}"` and `merge_grading_chunks` matches the
# answer's `clause` value against that same index. A stored list that carries its
# own `1.`/`2.`/`4.`/`5.` therefore shows the grader TWO numbers per clause, and
# the grader answers the one it can read: #2991's `automod_vault_land` failed
# eight times as `no verdict for clause(s) 4 … [3 grading generations issued]`
# because its 4th clause is stored as `5. Exactly one activity-log line …` and
# the generation answered `{"clause": 5}` — an index outside the slice, which the
# merge drops (#2263), so a real `met` became a gap and the witness was reverted.
# The gaps in the stored numbering are not noise: #2991 has no `3.` because that
# clause moved to `human_clauses`.

SELF_NUMBERED = [
    "1. Both witness files exist at the named vault paths and `git ls-files` lists both.",
    "2. The note states the pre-landing reading with its figures.",
    "4. The note quotes `wc -l -c < backlog/data/x.json` matching the bytes it ships.",
    "5. Exactly one activity-log line naming that path and sha256 is appended to #2806.",
]
# Each stored text minus its own `N. ` enumerator and the one space after it.
UNNUMBERED = [c[3:] for c in SELF_NUMBERED]


def test_a_self_numbered_contract_is_stripped_to_its_content():
    """#3002 clause 1. Every entry of the list opens on an enumerator, so the
    enumerators are the list's numbering, not any clause's content, and the
    clause text the grader receives starts with its content."""
    assert B.clean_clauses(SELF_NUMBERED) == UNNUMBERED


def test_a_contract_read_from_an_item_is_stripped_and_keeps_its_clause_count(isolated):
    """The read path the grader actually uses — front matter through
    `acceptance_clauses_of` — not just the helper. Four entries stay four: the
    numbering is stripped, the clauses are not dropped."""
    p = write_item(isolated, 66120, extra={"acceptance_clauses": list(SELF_NUMBERED)})
    clauses = B.acceptance_clauses_of(None, _fm(p))
    assert clauses == UNNUMBERED, clauses
    assert len(clauses) == 4, f"a 4-entry list must grade as 4 clauses: {clauses}"


def test_the_write_path_stores_the_contract_unnumbered(isolated):
    """What stops the NEXT self-numbered contract. The triage prompt tells every
    triage to write `"1." "2."`, and a gap opens the moment one clause moves to
    `HUMAN_CLAUSES` or falls past the clause cap — which is exactly #2991's shape.
    Storing content makes the position the only number there is."""
    p = write_item(isolated, 66121)
    B.record_verdict(B.item_by_id(66121), "confirmed", "re-measured, unchanged",
                     acceptance="x", acceptance_clauses=list(SELF_NUMBERED))
    assert list(_fm(p)["acceptance_clauses"]) == UNNUMBERED, _fm(p)["acceptance_clauses"]


def test_a_clause_whose_leading_digits_are_a_measurement_survives_byte_for_byte():
    """#3002 clause 3. One entry opening `2. GB of free space …` among unnumbered
    siblings is a quantity, not clause 2 of a list: stripping it would delete a
    unit out of the contract, so the all-entries test is what makes stripping safe."""
    stored = ["The volume has 3 mounts",
              "2. GB of free space must remain after the sweep",
              "The note records the count"]
    assert B.clean_clauses(stored) == stored
    # A lone entry is no numbering either unless it opens at 1: nothing here
    # says "clause 2", and the digits are the measurement.
    lone = ["2. GB of free space must remain after the sweep"]
    assert B.clean_clauses(lone) == lone
    # A single clause that does number itself at 1 is still just a numbering.
    assert B.clean_clauses(["1. Both witness files exist at the named vault paths."]) == \
        ["Both witness files exist at the named vault paths."]


# ── #3098: one clause the writer stored as two rows ─────────────────────────
#
# Rows 3 and 4 below are item #3092's clause 3 verbatim as its confirmed
# `backlog_triage` ledger row still carries it in
# `~/.local/state/lloyd-automod/promotions.jsonl`: the guided finalizer wrote the
# clause's two lines as two array elements, so the contract holds SIX rows for
# FIVE clauses and row 4 opens with no enumerator at all. The grader is handed the
# rows and answers the clauses it can read (1..5), the merge maps them
# positionally, and the orphan row comes back `partial (downgraded: not graded)` —
# a blocking entry. Round `SM_20261021_013732` spent a review attempt on exactly
# that, with all five real clauses `met`. The other four rows keep their
# enumerators and lose their prose: the row COUNT is the defect.
WRAPPED_TAIL = ("MAX_SESSIONS , )` helper and fails if the block reverts to pending wording "
                "or drops the pass id or the eviction trip wire, so a later edit cannot "
                "silently reopen the question — tests/test_component_manifest_retention.py")
SIX_ROWS_FIVE_CLAUSES = [
    "1. `app/component_manifest.py`'s comment block above `MAX_SESSIONS` no longer presents "
    "the cap as undecided — tests/test_component_manifest_retention.py",
    "2. The same block records the ruling as made and which way it went — "
    "tests/test_component_manifest_retention.py",
    "3. A new node in `tests/test_component_manifest_retention.py` reads the block with the "
    "existing `_comment_above(",
    WRAPPED_TAIL,
    "4. The residency pricing the existing witness node grades survives the edit — "
    "tests/test_component_manifest_retention.py",
    "5. `tests/test_component_manifest_retention.py:664` no longer describes the cap "
    "decision as owed work — tests/test_component_manifest_retention.py",
]


def test_a_clause_stored_in_two_rows_is_read_as_the_one_clause_it_is():
    """#3098 clause 1. N+1 rows holding one enumerator-less row, with the others
    running a contiguous `1.`…`N.`, are N clauses: the orphan is the row above it
    wrapped, so it joins onto that row with one space and no clause word is lost."""
    assert len(SIX_ROWS_FIVE_CLAUSES) == 6
    assert sum(1 for c in SIX_ROWS_FIVE_CLAUSES if not B._CLAUSE_LINE.match(c)) == 1, \
        "the fixture must stay one orphan among five numbered rows"

    out = B.clean_clauses(SIX_ROWS_FIVE_CLAUSES)
    assert len(out) == 5, [c[:60] for c in out]
    # The join added exactly the one byte the writer removed: a space, where the
    # newline was. The two halves are otherwise byte-identical to what was stored.
    assert out[2] == SIX_ROWS_FIVE_CLAUSES[2][3:] + " " + WRAPPED_TAIL, out[2]
    assert "`_comment_above( MAX_SESSIONS , )`" in out[2], out[2]
    assert out[:2] == [SIX_ROWS_FIVE_CLAUSES[0][3:], SIX_ROWS_FIVE_CLAUSES[1][3:]]
    assert out[3:] == [SIX_ROWS_FIVE_CLAUSES[4][3:], SIX_ROWS_FIVE_CLAUSES[5][3:]]
    lost = [w for w in WRAPPED_TAIL.split() if w not in out[2]]
    assert not lost, f"the continuation lost {lost}"
    # Where the bound is, said out loud rather than left to the phrase "no word
    # lost": `CLAUSE_MAX_CHARS` caps a CLAUSE, so two halves whose joined length
    # passes it are cut to it exactly as a single clause that long already is —
    # #3092's own clause 3 joins to 329 characters, nowhere near it.
    halves = B._join_wrapped_clause(["1. " + "A" * 560, "B" * 60, "2. next clause"])
    assert len(halves) == 2 and len(halves[0]) == B.CLAUSE_MAX_CHARS, \
        [len(x) for x in halves]
    assert halves[0].startswith("1. " + "A" * 560 + " " + "B")


def test_the_join_lives_in_clean_clauses_so_the_writer_and_every_reader_agree(isolated):
    """#3098 clause 2. `clean_clauses` is the one funnel: the writer of a new
    contract (`cap_new_clauses`, called by `record_verdict`) and every reader
    (`acceptance_clauses_of`, over front matter AND over the ledger event) see the
    same joined clauses, so an already-poisoned stored row is repaired on read as
    well as on the next write."""
    joined = B.clean_clauses(SIX_ROWS_FIVE_CLAUSES)
    assert len(joined) == 5, joined

    # The writer: five kept, nothing silently past the cap. Before the join the
    # same input was seven rows on a six-clause contract and cap_new_clauses cut
    # a real clause out of the contract without saying so.
    kept, dropped = B.cap_new_clauses(SIX_ROWS_FIVE_CLAUSES)
    assert kept == joined and dropped == [], (len(kept), len(dropped))

    p = write_item(isolated, 3098)
    B.record_verdict(B.item_by_id(3098), "confirmed", "re-read from the ledger row",
                     acceptance="x", acceptance_clauses=list(SIX_ROWS_FIVE_CLAUSES))
    fm = _fm(p)
    assert list(fm["acceptance_clauses"]) == joined, fm["acceptance_clauses"]
    assert "`_comment_above( MAX_SESSIONS , )`" in p.read_text(), "the body block split too"

    # Every reader, over both places a contract is stored.
    assert B.acceptance_clauses_of({}, fm) == joined
    assert B.acceptance_clauses_of({"acceptance_clauses": SIX_ROWS_FIVE_CLAUSES}) == joined


def test_the_collapse_never_joins_a_list_that_is_not_one_clause_wrapped():
    """#3098 clause 3. Each shape that is NOT a wrapped clause is returned
    untouched, and `_self_numbered` keeps every answer it had before: a guard
    that widened here would merge two real clauses into one and the grader would
    never see the difference."""
    # The positive control, so a broken signature cannot pass by never firing.
    assert B._join_wrapped_clause(["1. a", "an orphan tail", "2. b"]) == \
        ["1. a an orphan tail", "2. b"]

    # (a) an all-unnumbered list is not a numbering one row short.
    assert B._join_wrapped_clause(UNNUMBERED) == UNNUMBERED
    assert B.clean_clauses(UNNUMBERED) == UNNUMBERED
    # (b) the measurement shape: one numbered row among unnumbered siblings
    #     carries TWO orphan rows, and `2. GB…` is a quantity, not clause 2.
    measured = ["The volume has 3 mounts",
                "2. GB of free space must remain after the sweep",
                "The note records the count"]
    assert B._join_wrapped_clause(measured) == measured
    assert B.clean_clauses(measured) == measured
    lone = ["2. GB of free space must remain after the sweep"]
    assert B._join_wrapped_clause(lone) == lone
    # (c) a leading orphan continues nothing, (d) a numbering with a gap did not
    #     come from one list, (e) two orphans is prose with stray enumerators.
    headless = ["an unnumbered fragment first", "1. a", "2. b"]
    gapped = ["1. a", "an orphan tail", "3. c"]
    two = ["1. a", "orphan one", "orphan two", "2. b"]
    for ragged in (headless, gapped, two):
        assert B._join_wrapped_clause(ragged) == ragged, ragged

    # `_self_numbered` is untouched by the join: the same six answers as at base.
    assert B._self_numbered(SELF_NUMBERED) is True
    assert B._self_numbered(UNNUMBERED) is False
    assert B._self_numbered(measured) is False
    assert B._self_numbered(lone) is False
    assert B._self_numbered([]) is False
    assert B._self_numbered(["1. one", "2. two", "4. four", "5. five"]) is True
    # And #3002's shape still strips to its content, join not firing on it.
    assert B.clean_clauses(SELF_NUMBERED) == UNNUMBERED


# ── #3347: a verdict trailer is not a clause the grader has to satisfy ──────────
#
# `record_verdict` takes `human_clauses=` and `spawned=` as parameters SEPARATE from
# `acceptance_clauses`, and keeps them out of that list on purpose. Autotriage ignored the
# separation for #3341 and wrote the two trailer lines INTO the clause list, so #3341's
# contract is five rows of which two no diff can ever satisfy. Two rounds were refused on
# those two rows alone (SM_20261025_233431 at 7ccdc67bb228, SM_20261026_004219 at
# 3718bc6d3754), with the gate's clause rows reading 1/2/3 `met` + `how_verified: ran` and
# 4/5 `partial` + note `not addressed by the grader`: `review.py` synthesises a `partial`
# for a clause the grader could not address, and `partial` blocks. Neither escape route was
# open from inside that round — `amend_clause` refuses unless the grader called the clause
# UNSATISFIABLE, and it never does (it never reaches the clause at all).
#
# The asymmetry is the one `acceptance_text` already knows about: a bare `- 'none'` dies in
# `clean_clauses` today, while `- 'HUMAN_CLAUSES: none'` survives it, because the
# punctuation strip that catches the first leaves the second intact. Both nodes below
# measure both sides, because a filter that drops too much is worse than no filter: it
# deletes a real requirement from grading and reports success.

REAL3 = ["the runner prints the tally", "the table names the column",
         "a node pins the refusal"]


def test_a_trailer_row_with_a_placeholder_value_stops_being_a_clause():
    """Clause 1: `HUMAN_CLAUSES: none` / `SPAWNED: none` are not requirements.

    #3341's two rows, plus a numbered and an `=`-spelled variant, because the row reaches
    `clean_clauses` carrying whatever numbering the model chose.
    """
    assert B.clean_clauses(["1. real clause", "HUMAN_CLAUSES: none",
                            "SPAWNED: none"]) == ["real clause"]
    assert B.clean_clauses(REAL3 + ["4. HUMAN_CLAUSES: none", "5. SPAWNED: none"]) == REAL3
    assert B.clean_clauses(REAL3 + ["human_clauses: n/a", "Spawned = NA"]) == REAL3
    assert B.clean_clauses(REAL3 + ["- HUMAN_CLAUSES: none", "* SPAWNED: nil"]) == REAL3
    assert B.trailer_placeholder_row("HUMAN_CLAUSES: none") is True
    assert B.trailer_placeholder_row("SPAWNED: none") is True


def test_a_clause_that_only_names_a_trailer_label_survives_byte_for_byte():
    """Clause 2: the false positive this filter must not commit.

    `SPAWNED: #401 #402` is a round's record of what it filed, and `the ruling the
    HUMAN_CLAUSES section defers` is a real requirement that happens to name a section.
    Dropping either silently un-grades a demand, which is the failure
    `POST_LANDING_RX`'s comment recites for the sibling filter: the clause list is the
    implementer's contract, and a filter that shaves it is worse than one that lets an
    ungradeable row through.
    """
    survivors = ["the ruling the HUMAN_CLAUSES section defers", "SPAWNED: #401 #402"]
    assert B.clean_clauses(survivors) == survivors
    assert B.clean_clauses(REAL3 + survivors) == REAL3 + survivors

    # A non-placeholder value under the label stays graded, deliberately: routing
    # `HUMAN_CLAUSES: re-run the owed check Monday` into the human list is a triage
    # question, and this is a placeholder guard, not a re-router.
    kept = ["HUMAN_CLAUSES: re-run the owed check Monday"]
    assert B.clean_clauses(kept) == kept
    # A bare label with no value is not the shape either: that is a header row, and
    # guessing which field it meant is not this filter's call.
    assert B.clean_clauses(["SPAWNED"]) == ["SPAWNED"]


def test_a_contract_read_from_an_item_already_on_disk_drops_the_stored_rows(isolated):
    """Clause 3: the READ side, which is what heals items filed before this landing.

    #3341's malformed rows are already committed in its front matter, so a writer-only fix
    would leave every item triaged before today broken forever. The file is therefore
    written the way a pre-fix autotriage left it — trailer rows sitting in
    `acceptance_clauses` and in `human_clauses` — and read back the way `item_contract`
    reads it, with no re-write in between.
    """
    p = write_item(isolated, 66130, extra={
        "acceptance_clauses": REAL3 + ["HUMAN_CLAUSES: none", "SPAWNED: none"],
        "human_clauses": ["HUMAN_CLAUSES: none", "SPAWNED: none"]})
    raw = _fm(p)
    assert "HUMAN_CLAUSES: none" in raw["acceptance_clauses"], (
        "the fixture stopped looking like #3341's file, so this node is no longer "
        "measuring a stored malformed contract")
    assert B.acceptance_clauses_of(None, raw) == REAL3
    # The trailer's own field: `human_clauses_of` shares `clean_clauses`, so the same rows
    # are absent there too instead of parking in a field nothing grades.
    assert B.human_clauses_of(None, raw) == []


def test_writer_and_reader_agree_on_what_a_trailer_row_is(isolated):
    """Clause 4: the boundary is the item FILE, and both ends must call it nothing.

    A writer that dropped the row while a reader kept it would make an item's contract
    depend on when it was written; the reverse would let a trailer reach the grader from a
    file an older writer produced. So write through `record_verdict` with the trailers among
    the clauses, then read the file both ways: as stored bytes, and through the reader. The
    first assertion is the writer's half and would fail alone if only the read path were
    fixed.
    """
    p = write_item(isolated, 66131)
    B.record_verdict(B.item_by_id(66131), "confirmed", "re-measured, unchanged",
                     acceptance="A: real one\nB: real two\nC: real three",
                     acceptance_clauses=REAL3 + ["HUMAN_CLAUSES: none", "SPAWNED: none"],
                     human_clauses=["HUMAN_CLAUSES: none"], spawned=[])
    stored = _fm(p)
    assert stored["acceptance_clauses"] == REAL3, stored["acceptance_clauses"]
    assert B.acceptance_clauses_of(None, stored) == REAL3
    assert B.human_clauses_of(None, stored) == [], (
        "the trailer survived into the human list, which is the field it should have been "
        f"written into as a VALUE — `none` is not one: {stored['human_clauses']}")
    # One answer from either end, bypassing the file entirely:
    assert B.acceptance_clauses_of({"acceptance_clauses": REAL3 + ["SPAWNED: none"]}) == REAL3

REAL_3504 = [
    (
        "1. `~/obsidian/backlog/data/2026-10-22.3120-lloyd-automod-promotions.jsonl` "
        "exists and `wc -l -c <` it returns 9 lines and 16,246 bytes, with every line "
        "one JSON object (no header or comment line) — "
        "backlog/data/2026-10-22.3120-lloyd-automod-promotions.jsonl"
    ),
    (
        "2. Composition is exactly 1 row with title `Lloyd data root damaged — "
        "promotions halted` and 8 rows with `Runtime data is being written into the "
        "code tree`, all carrying `"
    ),
]

# The four payload-key rows #3504's contract actually carries, in the order the
# stored file holds them, and #3616's cache key: a label, an underscore, and two
# 32-hex halves. All of them are one lowercase identifier, which is the shape.
PAYLOAD_KEYS = ["human_clauses:[", "spawned", "findings_filed_to", "item_id"]
HEX_CACHE_KEY = ("human_clauses_" + "5200331d648a5761878a246168183852_"
                 + "5200331d648a5761878a246168183852_" + "5200331d648a5761878a246168183852")
JUNK_ROWS = PAYLOAD_KEYS + ["1.", "1]}],", "or", "entity:", "contaminated_dirs",
                            "spawned:[", "human_clauses", "--rep",
                            "` `) ` ` ` ` ` _ ` ` ` ` ` , a .", HEX_CACHE_KEY]
# #3504's contract AS STORED: two `N. ` rows and four payload keys, six items.
SIX_ROWS_TWO_CLAUSES = list(REAL_3504) + PAYLOAD_KEYS
# The two real clauses exactly as the reader answers them: #3504's list is fully
# numbered once the junk is gone, so `_strip_self_numbering` takes the `N. `.
REAL_3504_UNNUMBERED = [c[3:] for c in REAL_3504]


def test_a_spliced_payload_key_row_stops_being_a_clause():
    """#3897 clause 1: one exported predicate, and #3504's six rows are its two.

    The reproduction the item was triaged on was `clean_clauses` answering 6 of 6
    on the stored list; the figure it has to answer now is 2 of 6. The predicate
    is module-level in `scripts/automod/backlog.py` (`names_a_demand`) because
    clause 3's other reader — the owed writer — has to give the same answer.
    """
    assert len(SIX_ROWS_TWO_CLAUSES) == 6, "the fixture is #3504's stored list"
    out = B.clean_clauses(SIX_ROWS_TWO_CLAUSES)
    assert out == REAL_3504_UNNUMBERED, [c[:70] for c in out]
    assert len(out) == 2, f"the reproduction figure is 2 of 6, got {len(out)}"

    # Every junk row, refused. `1.` and `1]}],` are the two #3504's
    # `human_clauses` held and `add_owed` booked; `or`, `entity:` and
    # `contaminated_dirs` are #2933's and #2474's; `spawned:[` is #2687's; the
    # punctuation orphan is one of #3205's; the cache key is #3616's.
    for row in JUNK_ROWS:
        assert B.names_a_demand(row) is False, row

    # Both arms on their own, so a guard that only ever fires one cannot pass.
    # Arm one is the conjunction — no path-like token AND no run of four
    # alphanumerics: `1]}],` fails it, and `a/b` is kept on the token alone even
    # though it has no run of four at all.
    assert B.names_a_demand("1]}],") is False
    assert B.names_a_demand("a/b") is True
    # Arm two is the bare lowercase key, and it is the only thing standing between
    # a spliced `spawned` and a graded clause: that row carries a run of seven.
    assert B.names_a_demand("spawned") is False
    assert B.names_a_demand("spawned and or are ordinary words in this clause") is True


def test_a_contract_read_from_an_item_already_on_disk_drops_the_payload_rows(isolated):
    """#3897 clause 2: the READ side heals a contract already committed to main.

    #3504's six rows are in its front matter now, so a writer-only fix would leave
    every item triaged before this landing graded against four rows nobody can
    satisfy. The item below is written the way the spliced verdict left it and read
    back with no rewrite in between; then the same payload is handed to the writer,
    which has to refuse it too — one answer from both ends of the file.
    """
    p = write_item(isolated, 66140, extra={
        "acceptance_clauses": list(SIX_ROWS_TWO_CLAUSES),
        "human_clauses": ["1.", "1]}],"]})
    raw = _fm(p)
    assert len(raw["acceptance_clauses"]) == 6, (
        "the fixture stopped being a stored malformed contract")
    assert raw["human_clauses"] == ["1.", "1]}],"], raw["human_clauses"]
    survivors = B.acceptance_clauses_of(None, raw)
    assert survivors == REAL_3504_UNNUMBERED
    # Anchored in literal text as well, so the equality above cannot be satisfied by
    # a pair of rows the same constant both supplies and expects: the two survivors
    # are #3504's extract-path demand and its composition count, and the four
    # payload rows are gone.
    assert survivors[0].startswith("`~/obsidian/backlog/data/2026-10-22.3120"), survivors[0][:40]
    assert survivors[1].startswith("Composition is exactly 1 row"), survivors[1][:40]
    assert B.human_clauses_of(None, raw) == [], (
        f"the payload rows are still human clauses: {raw['human_clauses']}")
    assert "spawned" in p.read_text(), "the guard reads; it does not rewrite the file"

    q = write_item(isolated, 66141)
    target = B.item_by_id(66141)
    assert target is not None, "the fixture item is not in the isolated board"
    B.record_verdict(target, "confirmed", "re-measured, unchanged",
                     acceptance="the extract is published and its figures match",
                     acceptance_clauses=list(SIX_ROWS_TWO_CLAUSES),
                     human_clauses=["1.", "1]}],"])
    stored = _fm(q)
    assert stored["acceptance_clauses"] == REAL_3504_UNNUMBERED, stored["acceptance_clauses"]
    assert stored.get("human_clauses") in (None, []), stored.get("human_clauses")


def test_the_guard_is_shape_based_and_never_length_based():
    """#3897 clause 4: the false positive this guard must never commit.

    A clause that stops mid-sentence is still a requirement, and the shortest
    survivor here is shorter than the longest rejected row — which is what says
    there is no length floor anywhere in it. #3504's clause 2 ends on a bare
    backtick because the JSON salvage cut it, and every candidate below is kept
    byte-for-byte, not repaired and not shaved.
    """
    survivors = [
        REAL_3504_UNNUMBERED[1],                       # … all carrying `
        REAL_3504_UNNUMBERED[0],                       # names two file paths
        "the new node names the file it edits — tests/test_clause_budget.py",
        "the clause mentions spawned and or as words mid-sentence",
        "facts — tests/test_kg_hygiene.py",
        "or.py",
        "a/b",
    ]
    for s in survivors:
        assert B.names_a_demand(s) is True, s
        assert B.clean_clauses([s]) == [s], s

    kept_shortest = min(survivors, key=len)
    refused_longest = max(JUNK_ROWS, key=len)
    assert len(kept_shortest) < len(refused_longest), (kept_shortest, refused_longest)


def test_the_predicate_is_case_sensitive_so_an_uppercase_header_row_survives():
    """#3897 clause 4: the lowercase keys go, the UPPERCASE label stays.

    `test_a_clause_that_only_names_a_trailer_label_survives_byte_for_byte` pins
    `clean_clauses(["SPAWNED"]) == ["SPAWNED"]`: a bare uppercase label is a header
    row, and guessing which field it meant is not this filter's call. A
    case-insensitive key rule would red that node by keeping `spawned` with it, so
    the arm is lowercase-only — which is also the shape of every row #3504 and
    #3654 leaked, and why the two answers coexist in one call.
    """
    assert B.clean_clauses(["SPAWNED"]) == ["SPAWNED"]
    assert B.names_a_demand("SPAWNED") is True
    assert B.names_a_demand("HUMAN_CLAUSES") is True
    assert B.names_a_demand("spawned") is False
    assert B.names_a_demand("human_clauses") is False
    # #3654's contract as stored: one real clause and the two labels that spliced in.
    assert B.clean_clauses(["1. The stage-list line", "human_clauses", "spawned"]) == \
        ["The stage-list line"]
    # The trailer filter is untouched and still answers only for a placeholder VALUE.
    assert B.trailer_placeholder_row("SPAWNED: none") is True
    assert B.trailer_placeholder_row("spawned") is False

# ── #3897 clause 5: a clause fixture cannot pass vacuously ──────────────────

# The field names a fixture writes a CONTRACT under. `clauses` is the review
# grader's own key and the ledger's `acceptance_clauses` its stored one, so all
# three reach a round: a junk-filled literal under any of them makes the node
# that holds it assert about nothing (see the vacuous pair
# `tests/test_automod_review_tools.py:35` / `tests/test_review_transport.py:1037`).
CLAUSE_FIXTURE_FIELDS = {"clauses", "acceptance_clauses", "human_clauses"}

CLAUSE_YAML_RX = re.compile(
    r"(?:acceptance_clauses|human_clauses|clauses)\s*:?\s*\n((?:[ \t]*-[ \t]+.*\n?)+)")

#: Clause-shaped rows a fixture may still hold although `names_a_demand` refuses
#: them — keyed by the TEST NODE that holds the row, never by the row text alone,
#: because a global text exemption would silently excuse the same text in any
#: future fixture. Each entry names the node whose assertion has the junk as its
#: SUBJECT: migrating those rows would delete the test rather than fix it.
#: `test_the_fixture_sweep_is_not_neutered` re-checks both halves of every entry —
#: the row is still refused, and it is still a row of THAT node — so an exemption
#: that stops being earned fails instead of rotting.
CLAUSE_FIXTURE_ROWS_UNDER_TEST = {
    ("test_a_contract_read_from_an_item_already_on_disk_drops_the_payload_rows", "1."):
        "#3504's first phantom owed row, written to prove the reader drops it",
    ("test_a_contract_read_from_an_item_already_on_disk_drops_the_payload_rows", "1]}],"):
        "#3504's second phantom, same node, same proof",
    ("test_a_closed_item_whose_human_clauses_are_payload_keys_derives_nothing", "1."):
        "the same two phantoms on the owed side, tests/test_owed_check.py",
    ("test_a_closed_item_whose_human_clauses_are_payload_keys_derives_nothing", "1]}],"):
        "the same two phantoms on the owed side, tests/test_owed_check.py",
    ("test_the_structured_path_carries_clauses_too", "none"):
        "the placeholder clause row that node asks `parse_review` to drop",
    ("test_clause_entries_that_are_not_objects_are_still_unreadable", "met"):
        "a grader VERDICT used as a clause row; that node asserts exactly that such a row "
        "grades nothing",
    ("test_a_turn_that_died_without_a_finalizer_closes_on_the_reviews_all_met", "met"):
        "the same shape on the finalizer side: five verdicts and no demand, which is the "
        "state the finalizer must still close on",
    ("test_the_fixture_sweep_is_not_neutered", "item_id"):
        "the planted collector probe's own inline-YAML row — this node plants junk to "
        "prove the collector sees it, so the junk is legitimately its subject",
    ("test_the_fixture_sweep_is_not_neutered", "or"):
        "the second planted inline-YAML row, same probe (#2933's two-letter shape)",
}


def _clause_elements(node, rows, depth=0):
    """Direct string elements of a list/tuple literal, into `rows` as (lineno, row).

    Direct only: recursing into dicts would sweep a grader ANSWER's own keys
    (`verdict`, `evidence_path`), which are not clauses and never reach
    `clean_clauses`. f-strings are skipped — their literal parts are fragments by
    construction.
    """
    if not isinstance(node, (ast.List, ast.Tuple)) or depth > 2:
        return
    for el in node.elts:
        if isinstance(el, ast.Constant) and isinstance(el.value, str):
            rows.append((node.lineno, el.value))
        elif isinstance(el, (ast.List, ast.Tuple)):
            _clause_elements(el, rows, depth + 1)


def _clause_rows_in_scope(nodes, local_params, rows):
    """Collect clause rows from the AST nodes of one scope (a test node, or module level).

    `local_params` maps a locally-defined helper's name to its positional parameter
    names, which is how a positional clause argument is recognised: the vacuous pair
    the sweep exists for wrote its clauses positionally (`write_item(items, 1, [c1..c5])`
    in `tests/test_automod_vault_round.py`), so a collector that read only keyword
    arguments and dict values would walk past them.
    """
    for n in nodes:
        if isinstance(n, ast.keyword) and n.arg in CLAUSE_FIXTURE_FIELDS:
            _clause_elements(n.value, rows)
        elif isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and k.value in CLAUSE_FIXTURE_FIELDS:
                    _clause_elements(v, rows)
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                named = ((isinstance(t, ast.Name) and t.id in CLAUSE_FIXTURE_FIELDS)
                         or (isinstance(t, ast.Subscript)
                             and isinstance(t.slice, ast.Constant)
                             and t.slice.value in CLAUSE_FIXTURE_FIELDS))
                if named:
                    _clause_elements(n.value, rows)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            params = local_params.get(n.func.id)
            if params:
                for i, arg in enumerate(n.args):
                    if i < len(params) and params[i] in CLAUSE_FIXTURE_FIELDS:
                        _clause_elements(arg, rows)
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            for m in CLAUSE_YAML_RX.finditer(n.value):
                for line in m.group(1).splitlines():
                    row = line.strip()[1:].strip()
                    if row:
                        rows.append((n.lineno, row))


def _clause_rows_from_source(src: str):
    """Every `(node_name, lineno, row)` this source attributes to a clause field.

    Scoped, so an exemption can name the node that earned it rather than the row
    text: rows found inside a function are credited to it, everything else to
    `<module>`. Nested helpers are skipped by the scope walk, which is deliberate —
    their bodies are their own scope and are walked separately.
    """
    tree = ast.parse(src)
    local_params = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_params[n.name] = [a.arg for a in n.args.args
                                    if a.arg not in ("self", "cls")]

    out = []
    for n in (tree.body):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows = []
            inner = [c for c in ast.walk(n)
                     if c is not n and not isinstance(c, (ast.FunctionDef,
                                                          ast.AsyncFunctionDef))]
            _clause_rows_in_scope(inner, local_params, rows)
            out += [(n.name, ln, row) for ln, row in rows]
        else:
            rows = []
            _clause_rows_in_scope(list(ast.walk(n)), local_params, rows)
            out += [("<module>", ln, row) for ln, row in rows]
    return out


def _all_clause_fixture_rows():
    """`(relpath, node_name, lineno, row)` for every clause fixture under tests/."""
    root = Path(__file__).resolve().parent
    out = []
    for f in sorted(root.rglob("*.py")):
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            for node_name, lineno, row in _clause_rows_from_source(src):
                out.append((str(f.relative_to(root)), node_name, lineno, row))
        except SyntaxError:
            continue
    return out


def test_no_clause_fixture_in_the_suite_is_a_row_the_reader_would_drop():
    """#3897 clause 5: every clause-shaped fixture row must name a demand.

    A fixture row the predicate refuses is worse than a missing fixture: the reader
    drops it silently, the contract the node then builds has fewer clauses than the
    test believes, and the assertions still pass — the shape that made
    `tests/test_review_transport.py:1037` pass vacuously behind
    `tests/test_automod_review_tools.py:35`, and the shape
    `tests/test_automod_vault_round.py`'s `write_item(items, 1, [c1..c5])` fixtures
    fell into at the previous attempt, where all five rows were refused so the item
    those nodes graded had no contract at all. So the whole `tests/` tree is swept
    and every row a fixture writes under `clauses`/`acceptance_clauses`/`human_clauses`
    — as a keyword argument, a dict value, an assignment, a POSITIONAL argument to a
    helper whose parameter carries that name, or an inline-YAML `- ` row — has to be a
    demand. Offenders were migrated to rows that say what they check (`["c"]` ->
    `["the flag defaults to off"]`, `["a"]` -> `["the skill version is bumped"]`,
    `[c1..c5]` -> five named demands); the rows still refused are the ones keyed by
    their node in `CLAUSE_FIXTURE_ROWS_UNDER_TEST`, whose junk IS that node's subject.
    """
    rows = _all_clause_fixture_rows()
    # An empty walk proves nothing, so the sweep carries its own floor. The number
    # in the message is the same measurement the assertion makes — deliberately not
    # a second figure in prose, so a fixture added later cannot make this node red
    # for having a stale count in a docstring.
    files = len({r[0] for r in rows})
    assert len(rows) >= 80, (
        f"the sweep collected {len(rows)} rows over {files} file(s) — it is not walking")
    assert files >= 8, f"the sweep reached {files} files — too few to mean anything"
    offenders = sorted({(f, node, row) for f, node, _, row in rows
                        if not B.names_a_demand(row)
                        and (node, row) not in CLAUSE_FIXTURE_ROWS_UNDER_TEST})
    assert not offenders, (
        f"{len(offenders)} clause fixture row(s) name no demand and would be dropped "
        f"by the reader: {offenders[:8]}")


def test_the_fixture_sweep_is_not_neutered():
    """#3897 clause 5, both directions: the sweep sees what it claims, and every
    exemption is still earned by the node that holds it.

    The first arm feeds the collector fixtures written the three ways real ones are
    written — a dict value, an inline-YAML block, and a POSITIONAL argument to a
    helper whose parameter is named `clauses` — and insists it found a junk row from
    each, so a collector that silently stopped matching any one of them cannot turn
    the sweep back on. The second arm reads the exemption table per node: a row may
    only sit there while it is BOTH refused by the predicate AND a row of the node
    the table names, so an exemption cannot leak to a new fixture that happens to
    reuse the text.
    """
    planted = _clause_rows_from_source(
        'def write_item(d, item_id, clauses):\n'
        '    return clauses\n'
        'write_item("/tmp", 1, ["- human_clauses:["])\n'
        'ev({"event": "backlog_triage", "acceptance_clauses": ["spawned"]})\n'
        'SRC = """\n'
        'status: draft\n'
        'acceptance_clauses:\n'
        '- item_id\n'
        '- or\n'
        '"""\n')
    planted_rows = [row for _, _, row in planted]
    for shape, want in (("positional", "- human_clauses:["),
                        ("dict value", "spawned"),
                        ("inline-YAML", "item_id"),
                        ("inline-YAML short word", "or")):
        assert want in planted_rows, f"the {shape} arm missed {want!r}: {planted_rows}"
    assert all(not B.names_a_demand(r) for r in planted_rows), planted_rows
    assert [node for node, _, _ in planted if node == "write_item"] == [], \
        "a helper's own body is not a fixture scope; it is walked as its own node"

    corpus = {(node, row) for _, node, _, row in _all_clause_fixture_rows()}
    for (node, row), why in CLAUSE_FIXTURE_ROWS_UNDER_TEST.items():
        assert not B.names_a_demand(row), f"{row!r} is exempt but no longer refused: {why}"
        assert (node, row) in corpus, (
            f"{row!r} is no longer a fixture row of {node}; drop its exemption: {why}")
