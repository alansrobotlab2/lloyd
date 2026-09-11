"""The triage pass may not feed itself.

`autotriage` exists to work through a stale backlog: it asks, of one old
item, whether the claim still describes the system. Step 6 of its prompt then
requires it to file whatever the item did not cover, because a finding that
lives only in a transcript is lost (#229).

Those two properties met in `select_candidate`, which takes the oldest
untriaged *open* item, and `OPEN_STATUSES` contains `draft` — the status
`backlog_write_task` writes. So every item triage filed re-entered the queue
it came out of, and the pass became its own supplier.

Measured over the loop's first 48 hours (2026-09-06 to 09-08):

    40 triage runs      28 items closed      78 items filed
    -> R = 1.95 new open items per item consumed
    -> +46 open items/day at the then-cadence of one run per 30 minutes
    -> open board 19 -> 122, of which 110 were the loop's own output

R > 1 is the whole bug. It does not matter how good the verdicts are or how
long it runs; the queue doubles instead of draining. Oldest-first ordering hid
it, because self-filed items sort to the back and the pass looks healthy until
the real backlog runs out — which on 2026-09-08 was 6 items away.

So: an item this loop filed is not a single-item triage candidate. The first
cut released it at 30 days, and by 2026-09-11 that was 291 items due to
re-enter the queue in October, each spawning ~2 more. Now the exits are the
ones that do not re-enter the queue: the nightly clustering pass, a group
triage `keep`, expiry (closed, tagged, reopenable by hand), and a human
reopen. See `backlog.is_quarantined` and `backlog.expire_stale_spawns`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, state as S
from workers.sources import _common as C
from workers.sources import autocode as I
from workers.sources import autotriage as M


def write_item(d: Path, item_id, *, status="draft", days_old=100, tags=("backlog",),
               name="A thing", body="Do the thing.", board="lloyd") -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": board, "tags": list(tags)}
    path = d / f"{item_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
                    encoding="utf-8")
    return path


def spawned_item(d: Path, item_id, *, days_old=0, tag="spawned-by-triage", **kw):
    """What `backlog_write_task` leaves behind: an open draft, freshly created."""
    return write_item(d, item_id, status="draft", days_old=days_old,
                      tags=("backlog", tag), **kw)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    # `new_worker_session` writes through `sessions_io.create_session`
    # now, and conftest's `_isolate_background_records` already points
    # that at a scratch dir for every test.
    return d


class _Item:
    def __init__(self, payload=None):
        self.payload = payload or {}


def _turn_that_files(text, spawn_ids, backlog_dir, *, stop_reason="stop", num_turns=12):
    """A triage turn that really writes the items it claims under SPAWNED.

    `existing_ids` verifies the ids against disk, so a fake that only prints
    numbers would be recorded as `spawned_unverified` and prove nothing.
    """
    async def fake(prompt, **kw):
        fake.calls.append({"prompt": prompt, **kw})
        for i in spawn_ids:
            spawned_item(backlog_dir, i, name=f"Spawn {i}")
        return {"text": text, "session_id": "sess_fake", "stop_reason": stop_reason,
                "num_turns": num_turns, "errors": []}
    fake.calls = []
    return fake


def _verdict(spawned="none"):
    return ("...analysis...\n\nVERDICT: stale\nSURFACE: code\nCHECK: grep -n foo\n"
            f"EVIDENCE: it moved.\nACCEPTANCE: -\nSPAWNED: {spawned}\n")


# ===========================================================================
# The edge that closed the loop
# ===========================================================================

def test_a_fresh_self_filed_item_is_not_a_triage_candidate(isolated):
    """The single assertion this whole file is about."""
    spawned_item(isolated, 500, days_old=0)
    assert B.select_candidate(S.LEDGER_PATH) is None


@pytest.mark.parametrize("tag", ["spawned-by-triage", "spawned-by-autocode"])
def test_both_producers_are_held(isolated, tag):
    """Triage files one tag and an implement round the other. Holding only the
    first would leave the hotter of the two producers wired straight back in —
    implement rounds filed 2.83 items per run to triage's 1.95."""
    spawned_item(isolated, 500, days_old=0, tag=tag)
    assert B.select_candidate(S.LEDGER_PATH) is None


def test_a_human_draft_is_still_a_candidate(isolated):
    """The gate keys on the spawn tags, not on `draft`.

    105 of the items on the board when this landed were drafts, and most of a
    stale backlog is exactly the material this pass exists to read. A rule
    that skipped drafts would switch the pass off rather than bound it.
    """
    write_item(isolated, 500, status="draft", days_old=0)
    assert B.select_candidate(S.LEDGER_PATH).id == 500


def test_a_self_filed_item_expires_instead_of_becoming_a_candidate(isolated):
    """Age no longer releases an item into triage; it releases it from the board."""
    spawned_item(isolated, 500, days_old=B.SPAWN_TRIAGE_MIN_AGE_DAYS + 1)
    assert B.select_candidate(S.LEDGER_PATH) is None
    out = B.expire_stale_spawns(S.LEDGER_PATH)
    assert [r["item_id"] for r in out] == [500]
    fm = yaml.safe_load(next(isolated.glob("500-*.md")).read_text().split("---")[1])
    assert fm["status"] == "done" and B.EXPIRED_TAG in fm["tags"] and fm.get("completed")
    assert any("expired" in str(l) and "reopen" in str(l) for l in fm["activity_log"])
    assert B.select_candidate(S.LEDGER_PATH) is None
    assert S.read_events(path=S.LEDGER_PATH)[-1]["event"] == "backlog_expired"


def test_the_expiry_boundary_is_the_documented_one(isolated):
    spawned_item(isolated, 500, days_old=B.SPAWN_TRIAGE_MIN_AGE_DAYS - 1)
    spawned_item(isolated, 501, days_old=B.SPAWN_TRIAGE_MIN_AGE_DAYS)
    assert [r["item_id"] for r in B.expire_stale_spawns(S.LEDGER_PATH)] == [501]


def test_expiry_never_touches_grouped_umbrella_needs_human_or_human_authored(isolated):
    old = B.SPAWN_TRIAGE_MIN_AGE_DAYS + 5
    write_item(isolated, 10, days_old=old)                                   # a human's draft
    write_item(isolated, 501, days_old=old, tags=("backlog", "spawned-by-triage", "grouped"))
    write_item(isolated, 502, days_old=old, tags=("backlog", "spawned-by-triage", "umbrella"))
    write_item(isolated, 503, days_old=old, tags=("backlog", "spawned-by-triage", B.NEEDS_HUMAN_TAG))
    assert B.expire_stale_spawns(S.LEDGER_PATH) == []
    assert all(i.status == "draft" for i in B.open_items())


def test_expiry_skips_triaged_implemented_and_landed_items(isolated):
    old = B.SPAWN_TRIAGE_MIN_AGE_DAYS + 5
    spawned_item(isolated, 501, days_old=old)
    spawned_item(isolated, 502, days_old=old)
    p = spawned_item(isolated, 503, days_old=old)
    S.append_event({"event": "backlog_triage", "item_id": 501, "verdict": "unverifiable"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 502, "phase": "started"},
                   path=S.LEDGER_PATH)
    text = p.read_text()
    p.write_text(text.replace("---\n\n#", f"{B.LANDED_MARKER}: abc123\n---\n\n#", 1))
    assert B.expire_stale_spawns(S.LEDGER_PATH) == []


def test_expiry_fires_once_and_a_reopen_clears_the_tag_and_releases_it(isolated):
    p = spawned_item(isolated, 500, days_old=B.SPAWN_TRIAGE_MIN_AGE_DAYS + 1)
    B.expire_stale_spawns(S.LEDGER_PATH)
    # A human sets the status back by hand (Mission Control, or an editor).
    text = p.read_text()
    p.write_text(text.replace("status: done", "status: draft", 1))
    moved = B.reconcile_statuses(S.LEDGER_PATH)
    assert {"item_id": 500, "from": "done", "to": "draft"} in moved
    fm = yaml.safe_load(p.read_text().split("---")[1])
    assert B.EXPIRED_TAG not in fm["tags"]
    assert B.select_candidate(S.LEDGER_PATH).id == 500, "released into the pool"
    assert B.expire_stale_spawns(S.LEDGER_PATH) == [], "a reopen is a decision; never re-expired"
    assert S.read_events(path=S.LEDGER_PATH)[-1]["event"] == "backlog_unexpired"


def test_expiry_can_be_switched_off(isolated):
    spawned_item(isolated, 500, days_old=B.SPAWN_TRIAGE_MIN_AGE_DAYS + 1)
    assert B.expire_stale_spawns(S.LEDGER_PATH, enabled=False) == []
    assert B.open_items()[0].status == "draft"


def test_a_group_triage_keep_releases_the_item_to_single_triage(isolated):
    spawned_item(isolated, 500, days_old=0)
    S.append_event({"event": "backlog_group_triage", "cluster_id": "c-1",
                    "judged": {"500": "keep"}}, path=S.LEDGER_PATH)
    assert B.select_candidate(S.LEDGER_PATH).id == 500


def test_real_work_still_outranks_everything(isolated):
    """Held items must not perturb the ordering of the ones that remain."""
    write_item(isolated, 10, days_old=50)
    write_item(isolated, 11, days_old=300)
    spawned_item(isolated, 500, days_old=0)
    assert B.select_candidate(S.LEDGER_PATH).id == 11


# ===========================================================================
# R < 1, asserted as arithmetic rather than hoped for
# ===========================================================================

def test_triage_cannot_feed_itself(isolated, monkeypatch):
    """One run in, nothing left behind that the next run may pick up.

    This is the regression that matters: before the quarantine, a run that
    filed three items handed the queue three replacements for the one it
    consumed, and the pass never ran out of work because it was the work.
    """
    write_item(isolated, 7, days_old=300)
    fake = _turn_that_files(_verdict("#401 #402 #403"), [401, 402, 403], isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)

    out = asyncio.run(M.execute(_Item({"max_turns": 90})))
    assert out["status"] == "success" and out["verdict"] == "stale"

    ledger_event = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ledger_event["spawned"] == [401, 402, 403], "the items really were filed"

    # The queue consumed one item and gained none it may act on.
    candidates, held = B.triage_pool(S.LEDGER_PATH)
    assert candidates == [] and held == 3
    assert B.select_candidate(S.LEDGER_PATH) is None


def test_without_the_quarantine_the_same_run_grows_the_queue(isolated, monkeypatch):
    """The counterfactual, so the test above cannot pass for a lazy reason.

    Same fixture, same run, the spawn tags unrecognised: the three filed
    items are candidates and the pass has more to do than when it started.
    """
    monkeypatch.setattr(B, "SPAWN_TAGS", frozenset())
    write_item(isolated, 7, days_old=300)
    monkeypatch.setattr(C, "run_prompt_in_session",
                        _turn_that_files(_verdict("#401 #402 #403"), [401, 402, 403], isolated))

    asyncio.run(M.execute(_Item({"max_turns": 90})))

    candidates, held = B.triage_pool(S.LEDGER_PATH)
    assert sorted(i.id for i in candidates) == [401, 402, 403] and held == 0


# ===========================================================================
# An empty queue has two meanings
# ===========================================================================

def test_the_pool_reports_what_it_is_holding(isolated):
    write_item(isolated, 10, days_old=300)
    spawned_item(isolated, 500, days_old=0)
    spawned_item(isolated, 501, days_old=0)
    candidates, held = B.triage_pool(S.LEDGER_PATH)
    assert [i.id for i in candidates] == [10] and held == 2


def test_an_exhausted_queue_says_whether_it_is_holding_anything(isolated, monkeypatch):
    """"Every open backlog item has been triaged" was true and misleading on a
    board of 122 where 106 were this loop's own drafts. The pass has to be
    able to say which of the two empties it is in, or nobody notices the
    board growing under a message that reads like completion."""
    spawned_item(isolated, 500, days_old=0)
    out = asyncio.run(M.execute(_Item()))
    assert out["status"] == "skipped"
    assert "1 self-filed item(s) held" in out["summary"], out["summary"]
    assert str(B.SPAWN_TRIAGE_MIN_AGE_DAYS) in out["summary"]


def test_a_genuinely_empty_queue_still_says_so_plainly(isolated):
    out = asyncio.run(M.execute(_Item()))
    assert out["status"] == "skipped"
    assert out["summary"] == "every open backlog item has been triaged"


# ===========================================================================
# The fan-out cap
# ===========================================================================

def _impl_prompt(**over):
    kw = dict(item_id=9, status="draft", priority="low", name="n", body="b",
              triaged_ago="an hour ago", surface="code", check="c", evidence="e",
              acceptance="a", clauses="    1. a", spawn_cap=I.SPAWN_CAP,
              round_label="item9", reoffer="", members="")
    kw.update(over)
    return I.PROMPT.format(**kw)


def test_both_prompts_state_the_cap():
    """A cap nobody is told about is not a cap."""
    triage = M.PROMPT.format(item_id=9, status="draft", priority="low", name="n",
                             body="b", age=1, spawn_cap=M.SPAWN_CAP)
    assert f"at most {M.SPAWN_CAP} separate items" in triage
    assert "Further findings from triage of #9" in triage

    impl = _impl_prompt()
    assert f"File at most {I.SPAWN_CAP}" in impl
    # Implement's cap is deliberately tighter than triage's: triage splits an
    # item into claims, so filing is its output; an implement round's output is
    # a landing, and its rounds filed 120 items against 7 closed.
    assert I.SPAWN_CAP < M.SPAWN_CAP


def test_the_overflow_item_keeps_the_anti_229_property():
    """The cap may not become a licence to drop findings. #229 said two claims
    "belong in two new items" and filed none; the answer to too many findings
    is somewhere for them to go, not fewer findings."""
    triage = M.PROMPT.format(item_id=9, status="draft", priority="low", name="n",
                             body="b", age=1, spawn_cap=M.SPAWN_CAP)
    assert "A finding that lives only in EVIDENCE is lost" in triage
    assert "cap on fan-out, not on honesty" in triage
    assert "nothing is dropped" in triage

    impl = _impl_prompt()
    assert "do not leave them only in your report" in impl


def test_the_autocode_prompt_sends_findings_to_the_parent_and_files_only_blockers():
    """The other half of the inflow: 120 implement-filed items against 7
    closed, and the re-runs of one item filing the same finding three times.
    A finding goes onto the item it came from; only a blocker is an item."""
    impl = _impl_prompt()
    assert 'backlog_write_task(task_id=9, description_mode="append"' in impl
    assert "## Findings" in impl
    assert 'first line "Blocks #9"' in impl and "`blocker`" in impl
    assert "merged_into: N" in impl
    assert "Further findings from implementing" not in impl, "the overflow item is gone"
    assert "SPAWNED: <ids of blocker items" in impl


def test_the_triage_prompt_tells_the_model_what_a_merge_means():
    """`backlog_tasks` has no text search; the check it used to ask for was a
    ritual. The tool does it now, and the prompt says how to read the answer."""
    triage = M.PROMPT.format(item_id=9, status="draft", priority="low", name="n",
                             body="b", age=1, spawn_cap=M.SPAWN_CAP)
    assert "merged_into: N" in triage and "force: true" in triage
    assert "run `backlog_tasks` to be sure" not in triage


def test_the_ledger_records_the_cap_and_the_overshoot(isolated, monkeypatch):
    """Recorded, not enforced — the items are on disk before the verdict is
    parsed, and unfiling them would destroy real findings. A number that can
    be watched is the honest version, on the one metric that told us the pass
    had inverted."""
    write_item(isolated, 7, days_old=300)
    ids = [401, 402, 403, 404, 405, 406]
    monkeypatch.setattr(C, "run_prompt_in_session",
                        _turn_that_files(_verdict(" ".join(f"#{i}" for i in ids)),
                                         ids, isolated))

    asyncio.run(M.execute(_Item({"max_turns": 90})))

    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["spawn_cap"] == M.SPAWN_CAP
    # cap + 1 is allowed: the N best, plus the single overflow item.
    assert ev["spawned_over_cap"] == len(ids) - (M.SPAWN_CAP + 1) == 2


def test_filing_within_the_cap_records_no_overshoot(isolated, monkeypatch):
    write_item(isolated, 7, days_old=300)
    monkeypatch.setattr(C, "run_prompt_in_session",
                        _turn_that_files(_verdict("#401 #402"), [401, 402], isolated))
    asyncio.run(M.execute(_Item({"max_turns": 90})))
    assert S.read_events(path=S.LEDGER_PATH)[-1]["spawned_over_cap"] == 0


# ===========================================================================
# What counts as a spawn
# ===========================================================================

def test_an_id_that_existed_before_the_run_is_a_merge_not_a_spawn(isolated, monkeypatch):
    """#370's finished row listed itself and a pre-existing #221 as spawns.
    The split is mechanical: the highest id on disk before the turn."""
    write_item(isolated, 7, days_old=300)
    spawned_item(isolated, 401, days_old=1, name="Already there")
    fake = _turn_that_files(_verdict("#401 #402 #7"), [402], isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(M.execute(_Item({"max_turns": 90})))
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["spawned"] == [402] and ev["merged"] == [401] and ev["id_floor"] == 401
    assert ev["spawned_unverified"] == [], "the item's own id is neither"
    text = next(isolated.glob("7-*.md")).read_text()
    assert "Filed as new items: #402" in text and "Merged findings into: #401" in text


def test_the_implementer_separates_merges_from_spawns_and_counts_findings(isolated, monkeypatch):
    write_item(isolated, 2, days_old=30)
    spawned_item(isolated, 300, days_old=2, name="Prior finding")
    S.append_event({"event": "backlog_triage", "item_id": 2, "verdict": "confirmed",
                    "check": "grep", "evidence": "still", "acceptance": "it passes"},
                   path=S.LEDGER_PATH)
    B.set_status(2, "up_next", "test: confirmed")
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    item_path = next(isolated.glob("2-*.md"))

    async def turn(prompt, **kw):
        # The round appends two findings onto its item and files one blocker.
        text = item_path.read_text()
        item_path.write_text(text.rstrip() + "\n\n## Findings (round SM_1)\n\n- one\n- two\n")
        spawned_item(isolated, 500, name="Blocker", tag="spawned-by-autocode")
        return {"text": "gate ok\n\nSPAWNED: #500 #300 #2\n", "session_id": "s",
                "stop_reason": "stop", "num_turns": 30, "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(I.execute(_Item()))
    assert out["status"] == "success"
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]
    assert ev["spawned"] == [500] and ev["merged"] == [300]
    assert ev["findings_appended"] == 2 and ev["spawned_over_cap"] == 0
    assert ev["spawned_unverified"] == []


def test_findings_are_counted_off_the_file_not_the_report():
    before = "# t\n\nbody\n\n## Findings (round A)\n\n- one\n"
    after = before + "\n## Findings (round B)\n\n- two\n- three\n\n## Notes\n\n- not a finding\n"
    assert B.count_findings(before, after) == 2
    assert B.count_findings(after, before) == 0, "never negative"
    assert B.count_findings("", "") == 0
