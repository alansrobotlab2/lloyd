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
import json
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
    spawned_item(isolated, 500, days_old=B.spawn_expiry_days() + 1)
    assert B.select_candidate(S.LEDGER_PATH) is None
    out = B.expire_stale_spawns(S.LEDGER_PATH)
    assert [r["item_id"] for r in out] == [500]
    fm = yaml.safe_load(next(isolated.glob("500-*.md")).read_text().split("---")[1])
    assert fm["status"] == "done" and B.EXPIRED_TAG in fm["tags"] and fm.get("completed")
    assert any("expired" in str(l) and "reopen" in str(l) for l in fm["activity_log"])
    assert B.select_candidate(S.LEDGER_PATH) is None
    assert S.read_events(path=S.LEDGER_PATH)[-1]["event"] == "backlog_expired"


def test_the_expiry_boundary_is_the_documented_one(isolated):
    spawned_item(isolated, 500, days_old=B.spawn_expiry_days() - 1)
    spawned_item(isolated, 501, days_old=B.spawn_expiry_days())
    assert [r["item_id"] for r in B.expire_stale_spawns(S.LEDGER_PATH)] == [501]


def test_expiry_reads_the_configured_bound_and_the_gauge_reads_the_same_one(isolated, monkeypatch):
    """One number, three readers. Expiry at 14 while the gauge measures
    `over_bound` against 30 would read 0 for the fortnight the sweep is
    supposed to have closed."""
    from scripts.automod import scorecard as SC
    monkeypatch.setattr(B, "spawn_expiry_days", lambda: 5)
    spawned_item(isolated, 500, days_old=6)
    gauge = SC._self_spawned_gauge([], isolated, now=datetime.now(timezone.utc).timestamp())
    assert gauge["bound_days"] == 5 and gauge["over_bound"] == 1
    assert [r["item_id"] for r in B.expire_stale_spawns(S.LEDGER_PATH)] == [500]


def test_the_expiry_bound_comes_from_config_and_falls_back_to_the_constant(monkeypatch):
    from app import config as CFG
    monkeypatch.setitem(CFG.CONFIG, "workers",
                        {"sources": {"autocode": {"expire_spawns_after_days": 14}}})
    assert B.spawn_expiry_days() == 14
    monkeypatch.setitem(CFG.CONFIG, "workers", {"sources": {"autocode": {}}})
    assert B.spawn_expiry_days() == B.SPAWN_EXPIRY_DAYS
    assert B.SPAWN_TRIAGE_MIN_AGE_DAYS == B.SPAWN_EXPIRY_DAYS, "the old name is an alias"


def test_expiry_never_touches_grouped_umbrella_needs_human_or_human_authored(isolated):
    old = B.spawn_expiry_days() + 5
    write_item(isolated, 10, days_old=old)                                   # a human's draft
    write_item(isolated, 501, days_old=old, tags=("backlog", "spawned-by-triage", "grouped"))
    write_item(isolated, 502, days_old=old, tags=("backlog", "spawned-by-triage", "umbrella"))
    write_item(isolated, 503, days_old=old, tags=("backlog", "spawned-by-triage", B.NEEDS_HUMAN_TAG))
    assert B.expire_stale_spawns(S.LEDGER_PATH) == []
    assert all(i.status == "draft" for i in B.open_items())


def test_expiry_skips_triaged_implemented_and_landed_items(isolated):
    old = B.spawn_expiry_days() + 5
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
    p = spawned_item(isolated, 500, days_old=B.spawn_expiry_days() + 1)
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
    spawned_item(isolated, 500, days_old=B.spawn_expiry_days() + 1)
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
    assert str(B.spawn_expiry_days()) in out["summary"]


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
              max_turns=I.DEFAULT_MAX_TURNS, gate_minutes=16, first_gate_by=23,
              round_label="item9", reoffer="", members="", human_clauses="",
              surface_rules="")
    kw.update(over)
    return I.PROMPT.format(**kw)


def _triage_prompt(item_id=9, **kw):
    item = B.Item(path=Path(f"/nonexistent/{item_id}-n.md"), id=item_id, name="n",
                  status="draft", priority="low", created="", body="b")
    return M.render_prompt(item, ledger=S.LEDGER_PATH, spawn_cap=kw.get("spawn_cap", M.SPAWN_CAP))


def test_both_prompts_send_findings_to_the_item_and_file_at_most_one():
    """A cap nobody is told about is not a cap — and since 2026-09-13 the two
    prompts carry the same rule. Triage's cap used to be three with an
    overflow item on top, and 80 of its first 100 confirmed runs filed at
    least one item the implementer would never read."""
    triage = _triage_prompt()
    assert 'backlog_write_task(task_id=9, description_mode="append"' in triage
    assert "## Findings (triage" in triage
    assert "at most 1 new item" in triage
    assert "Further findings from triage" not in triage, "the overflow item is gone"

    impl = _impl_prompt()
    assert f"File at most {I.SPAWN_CAP}" in impl
    # The asymmetry is abolished, not relaxed: one blocker per round, one
    # survivor per closing triage.
    assert I.SPAWN_CAP == M.SPAWN_CAP == 1


def test_a_kept_item_files_nothing_and_a_closing_one_files_last():
    """The verdict decides where a finding goes. `confirmed` was told "the
    item you are triaging is about to be closed", which was false."""
    text = " ".join(_triage_prompt().split())
    kept = text.index("`confirmed`, `unverifiable` or `not_code` — this item lives on")
    closing = text.index("`stale` or `already_done` — this item is about to be closed")
    assert kept < closing
    assert "File no new item; SPAWNED is `none`" in text[kept:closing]
    # a closing item's survivors look for an existing home before a new one
    tail = text[closing:]
    assert tail.index("already covers it") < tail.index("parent from `<origin>`") < tail.index("at most 1 new item")
    assert "item you are triaging is about to be closed" not in text


def test_the_cap_still_does_not_drop_findings():
    """The cap may not become a licence to drop findings. #229 said two claims
    "belong in two new items" and filed none; the answer to too many findings
    is somewhere for them to go, not fewer findings."""
    triage = _triage_prompt()
    assert "A finding that lives only in EVIDENCE is lost" in triage
    assert "cap on fan-out, not on honesty" in triage
    assert "nothing is dropped" in triage
    assert "Filing nothing is fine when there is nothing" in triage

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


def test_a_blocker_must_quote_an_artifact_it_re_read():
    """A blocker is the one thing an implement round files, and three of them
    carried evidence the artifact they named contradicted — each was trusted,
    and each became a fake blocker for another item:

      * #1091 quoted `FAIL review … (2 graded refusals, both blocking)`, a
        string `gate.py` has never contained in any commit, over a report
        showing attempt 1 of 2 and a ledger row saying the same;
      * #1169 read the *first* gate run's refusal as the current one and
        aborted a round whose `gate.json`, 20 s old, said `ok: true` with 5 of
        5 clauses met;
      * #1200 quoted `2 failed, 4969 passed … in 372.87s` for a `tests` rung
        that never ran — the round failed `frontend`, which precedes it — in a
        file that exists in no tree, after pytest had already answered
        `file or directory not found` in both trees.

    So the rule has to be the thing a reader holding only the item can check:
    the line copied out of the named artifact, the artifact re-read for the
    head it names, and any named file seen in the tree that ran.
    """
    impl = " ".join(_impl_prompt().split())
    # a quoted gate/ledger line is copied from the artifact, re-read from disk
    # for the round's current head, and the reader is told which path
    assert "re-read from disk" in impl
    assert "for the round's current head" in impl
    assert "name the path you copied from" in impl
    # no assertion that a rung produced output, and no named file, except from
    # that rung's own detail in the round's gate.json and from its tree
    assert "unless you read that rung's detail out of the round's `gate.json`" in impl
    assert "saw the file in the tree that rung ran in" in impl
    # the same re-read guards the abort, whose reason is stored unverified
    assert "re-read that report before filing a blocker or calling `automod_abort`" in impl


def test_the_triage_prompt_tells_the_model_what_a_merge_means():
    """`backlog_tasks` has no text search; the check it used to ask for was a
    ritual. The tool does it now, and the prompt says how to read the answer."""
    triage = _triage_prompt()
    assert "merged_into: N" in triage and "force: true" in triage
    assert "run `backlog_tasks` to be sure" not in triage


@pytest.mark.parametrize("filed,over", [(6, 5), (2, 1), (1, 0)])
def test_the_ledger_records_the_cap_and_the_overshoot(isolated, monkeypatch, filed, over):
    """Recorded, not enforced — the items are on disk before the verdict is
    parsed, and unfiling them would destroy real findings. With the overflow
    item gone there is no `+1`: a second filing is an overshoot."""
    write_item(isolated, 7, days_old=300)
    ids = list(range(401, 401 + filed))
    monkeypatch.setattr(C, "run_prompt_in_session",
                        _turn_that_files(_verdict(" ".join(f"#{i}" for i in ids)),
                                         ids, isolated))

    asyncio.run(M.execute(_Item({"max_turns": 90})))

    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["spawn_cap"] == M.SPAWN_CAP == 1
    assert ev["spawned_over_cap"] == over


def test_the_cap_rides_in_the_payload(isolated, monkeypatch):
    write_item(isolated, 7, days_old=300)
    fake = _turn_that_files(_verdict("#401 #402"), [401, 402], isolated)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(M.execute(_Item({"max_turns": 90, "spawn_cap": 2})))
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["spawn_cap"] == 2 and ev["spawned_over_cap"] == 0
    assert "at most 2 new item" in fake.calls[0]["prompt"]


# ===========================================================================
# Findings go onto the item
# ===========================================================================

def _confirmed(spawned="none"):
    return ("...analysis...\n\nVERDICT: confirmed\nSURFACE: code\nCHECK: grep -n foo\n"
            "EVIDENCE: still there.\nACCEPTANCE: foo is gone\n"
            "ACCEPTANCE_CLAUSES:\n1. foo is gone\nHUMAN_CLAUSES: none\n"
            f"SPAWNED: {spawned}\n")


def test_a_confirmed_triage_appends_its_findings_and_files_nothing(isolated, monkeypatch):
    path = write_item(isolated, 7, days_old=300)

    async def turn(prompt, **kw):
        text = path.read_text()
        path.write_text(text.rstrip() + "\n\n## Findings (triage 2026-09-13)\n\n- one\n- two\n")
        return {"text": _confirmed(), "session_id": "s", "stop_reason": "stop",
                "num_turns": 20, "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(M.execute(_Item({"max_turns": 90})))
    assert out["verdict"] == "confirmed"
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["findings_appended"] == 2 and ev["spawned"] == [] and ev["spawned_over_cap"] == 0
    after = path.read_text()
    assert "- one" in after and "- two" in after, "record_verdict kept the appended section"
    assert B.item_by_id(7).status == "up_next"


def test_a_stale_triage_appends_a_survivor_to_an_existing_item_as_a_merge(isolated, monkeypatch):
    write_item(isolated, 7, days_old=300)
    existing = spawned_item(isolated, 401, days_old=2, name="Covers it")

    async def turn(prompt, **kw):
        existing.write_text(existing.read_text().rstrip() + "\n\n## Findings (triage)\n\n- survivor\n")
        return {"text": _verdict("#401"), "session_id": "s", "stop_reason": "stop",
                "num_turns": 20, "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    asyncio.run(M.execute(_Item({"max_turns": 90})))
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["merged"] == [401] and ev["spawned"] == [] and ev["closed"] is True
    fm = yaml.safe_load(next(isolated.glob("7-*.md")).read_text().split("---")[1])
    assert fm["status"] == "done" and fm.get("completed"), "a triage close stamps completed"


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


# ===========================================================================
# The architecture reviewer files under its own tag, and the three readers
# of "the loop wrote this" deliberately disagree about it
# ===========================================================================

REVIEW_TAG = "spawned-by-review"


def test_a_review_finding_IS_a_triage_candidate(isolated):
    """The one asymmetry in the whole tag scheme, and the reason there are two
    frozensets rather than one.

    Quarantine asks: can this item answer the staleness question? An item
    triage filed cannot — it was written from a check that had just been run,
    so "is this still true?" is a re-run of that check. A review finding can:
    it describes the tree as of a commit up to a month old, and deciding
    whether it still holds is exactly what single triage is for.
    """
    spawned_item(isolated, 500, days_old=0, tag=REVIEW_TAG)
    assert B.select_candidate(S.LEDGER_PATH).id == 500
    fresh, held = B.triage_pool(S.LEDGER_PATH)
    assert [i.id for i in fresh] == [500] and held == 0


def test_quarantine_still_keys_on_exactly_the_four_loop_tags():
    """Widening `SPAWN_TAGS` instead of adding a second set would silently put
    every review finding out of reach of the pass that should judge it."""
    assert B.SPAWN_TAGS == {"spawned-by-triage", "spawned-by-autocode",
                            "spawned-by-autoimplement", "spawned-by-selfmod"}
    assert B.REVIEW_SPAWN_TAGS == {REVIEW_TAG}
    assert B.EVAL_SPAWN_TAGS == {EVAL_TAG}
    assert B.QUARANTINE_TAGS == B.SPAWN_TAGS | B.EVAL_SPAWN_TAGS
    assert B.LOOP_SPAWN_TAGS == B.SPAWN_TAGS | B.REVIEW_SPAWN_TAGS | B.EVAL_SPAWN_TAGS
    assert REVIEW_TAG not in B.quarantine_tags()


# ===========================================================================
# youtube-eval items are loop output (Alan, 2026-09-14)
# ===========================================================================

EVAL_TAG = "youtube-eval"


def _eval_switch(monkeypatch, on: bool):
    from app import config as CFG
    monkeypatch.setitem(CFG.CONFIG, "workers",
                        {"sources": {"youtube-digest": {"loop_spawned": on},
                                     "autocode": {"expire_spawns_after_days": 7}}})


def test_an_eval_item_is_quarantined_expired_and_counted(isolated, monkeypatch):
    """125 filed in a week, 101 still open, and none carried a `spawned-by-*`
    tag — so none was quarantined, expired or counted on the gauge."""
    from scripts.automod import scorecard as SC
    _eval_switch(monkeypatch, True)
    spawned_item(isolated, 500, days_old=1, tag=EVAL_TAG)
    spawned_item(isolated, 501, days_old=8, tag=EVAL_TAG)
    assert B.select_candidate(S.LEDGER_PATH) is None, "held out of single triage"
    assert B.triage_pool(S.LEDGER_PATH) == ([], 2)
    gauge = SC._self_spawned_gauge([], isolated, now=datetime.now(timezone.utc).timestamp())
    assert gauge["count"] == 2 and gauge["over_bound"] == 1
    out = B.expire_stale_spawns(S.LEDGER_PATH)
    assert [r["item_id"] for r in out] == [501]
    assert S.read_events(path=S.LEDGER_PATH)[-1]["spawned_by"] == EVAL_TAG


def test_an_eval_item_written_with_string_tags_is_still_recognised(isolated, monkeypatch):
    """The digest has written `tags` as a string that looks like a list."""
    from scripts.automod import scorecard as SC
    _eval_switch(monkeypatch, True)
    p = spawned_item(isolated, 500, days_old=8, tag=EVAL_TAG)
    p.write_text(p.read_text().replace("tags:\n- backlog\n- youtube-eval\n",
                                       "tags: '[youtube-eval, ai-engineer]'\n", 1))
    assert "'[youtube-eval" in p.read_text()
    gauge = SC._self_spawned_gauge([], isolated, now=datetime.now(timezone.utc).timestamp())
    assert gauge["count"] == 1
    assert [r["item_id"] for r in B.expire_stale_spawns(S.LEDGER_PATH)] == [500]


def test_the_eval_switch_off_puts_eval_items_back_to_a_humans(isolated, monkeypatch):
    _eval_switch(monkeypatch, False)
    spawned_item(isolated, 501, days_old=30, tag=EVAL_TAG)
    assert B.select_candidate(S.LEDGER_PATH).id == 501
    assert B.expire_stale_spawns(S.LEDGER_PATH) == []
    assert EVAL_TAG not in B.loop_spawn_tags() and EVAL_TAG not in B.quarantine_tags()


def test_the_two_predicates_answer_differently_for_a_review_finding(isolated):
    path = spawned_item(isolated, 500, days_old=0, tag=REVIEW_TAG)
    item = next(i for i in B.open_items(None) if i.id == 500)
    assert not B.is_self_spawned(item), "not held out of the triage pool"
    assert B.is_loop_spawned(item), "but still bounded by expiry and the gauge"
    assert path.exists()


def test_a_review_finding_nothing_picked_up_expires_on_the_same_bound(isolated):
    """Expiry asks the other question — did anything ever act on this? — and
    the answer is no for both producers, so the one bound applies to both."""
    spawned_item(isolated, 500, days_old=B.spawn_expiry_days() + 1, tag=REVIEW_TAG)
    out = B.expire_stale_spawns(S.LEDGER_PATH)
    assert [d["item_id"] for d in out] == [500]
    ev = [json.loads(l) for l in S.LEDGER_PATH.read_text().splitlines() if l.strip()]
    assert ev[-1]["event"] == "backlog_expired" and ev[-1]["spawned_by"] == REVIEW_TAG
    assert not B.open_items(None), "closed, with the text kept on disk"


def test_a_fresh_review_finding_is_not_expired(isolated):
    spawned_item(isolated, 500, days_old=1, tag=REVIEW_TAG)
    assert B.expire_stale_spawns(S.LEDGER_PATH) == []


def test_the_open_gauge_counts_review_findings(isolated, tmp_path):
    """Row 4's bound is about the size of the board, and a review draft sits on
    it exactly like a triage draft does."""
    from scripts.automod import scorecard as SC
    spawned_item(isolated, 500, days_old=1, tag=REVIEW_TAG)
    spawned_item(isolated, 501, days_old=1, tag="spawned-by-triage")
    gauge = SC._self_spawned_gauge([], isolated, now=datetime.now(timezone.utc).timestamp())
    assert gauge["count"] == 2


# ===========================================================================
# The mint is a prefix, so the expiry/gauge test is a prefix
# ===========================================================================
#
# `backlog_write_task` stamps `spawned-by-<whatever the session was>`, and the
# write-time dedupe has always asked for the PREFIX. The expiry/gauge side asked
# for six exact names instead, so 43 of the 412 open loop-tagged items on the
# board on 2026-09-17 — from 23 distinct minters — were loop output at the
# moment they were written and a human's item to every reader afterwards: never
# bounded by expiry, never counted by the gauge, and presented to triage as
# "a human's item, or a writer outside the loop"
# (`workers/sources/autotriage.py`, the fall-through in the origin block). The
# item that documents it, #1160, carries `spawned-by-autonomy-task-40` and was
# itself in that 43. See `LOOP_SPAWN_TAGS` for why quarantine stays exact.

PREFIX_ONLY = "spawned-by-anything-42"


def test_a_mint_nobody_enumerated_still_counts_as_loop_output(isolated):
    """Clause 1. A tag the reader never heard of is still the writer's own
    stamp: the prefix is what makes it loop output, not the name after it."""
    write_item(isolated, 600, days_old=0, tags=(PREFIX_ONLY,))
    item = next(i for i in B.open_items(None) if i.id == 600)
    assert B.is_loop_spawned(item), "expiry bounds it and the gauge counts it"
    assert B.loop_spawn_tag(item.tags) == PREFIX_ONLY, \
        "and the reader can name which minter produced it"


def test_a_forgotten_mint_expires_on_the_same_bound_and_names_its_minter(isolated):
    """Clause 2. The bound's whole purpose is to close what nothing picked up;
    an item the enumeration missed sits open past it forever, so this is the
    hole's actual cost — 44 open items on 2026-09-16 and 43 on 2026-09-17,
    measured through the production loader, none of them reachable by the pass
    whose entire job is to close them. The ledger row has to name the minter, or
    the next triage asks the same question this one did."""
    write_item(isolated, 601, days_old=B.spawn_expiry_days() + 1, tags=(PREFIX_ONLY,))
    out = B.expire_stale_spawns(S.LEDGER_PATH)
    assert [d["item_id"] for d in out] == [601]
    ev = [json.loads(l) for l in S.LEDGER_PATH.read_text().splitlines() if l.strip()]
    assert ev[-1]["event"] == "backlog_expired"
    assert ev[-1]["spawned_by"] == PREFIX_ONLY, \
        "an empty `spawned_by` would say 'a human closed this'"
    assert not B.open_items(None), "closed, with the text kept on disk"


def test_a_forgotten_mint_freshly_filed_is_not_expired(isolated):
    """The prefix admits 43 more items to the bound; it must not admit any of
    them before their time, since expiry is the ONLY automatic exit a draft
    that the sweep has not ranked can reach."""
    write_item(isolated, 602, days_old=1, tags=(PREFIX_ONLY,))
    assert B.expire_stale_spawns(S.LEDGER_PATH) == []


def test_the_quarantine_still_rejects_both_a_forgotten_mint_and_a_review(isolated):
    """Clause 3. The asymmetry, from the other side: recognition for expiry must
    not become recognition for quarantine. A prefix-tagged draft is not triage
    material (it was written from a check that had just been run, so "is this
    still true?" is a re-run of that check) — but neither was it before, and
    widening the quarantine would additionally hold out of single triage every
    `spawned-by-review` finding, which is the one shape that DOES answer the
    staleness question. So this asserts no change in either direction.

    `test_without_the_quarantine_the_same_run_grows_the_queue` is the
    counterfactual for the quarantine half: it empties `SPAWN_TAGS` and shows
    the queue growing, which only works while quarantine keys on the set.
    """
    for iid, tag in ((603, PREFIX_ONLY), (604, REVIEW_TAG)):
        write_item(isolated, iid, days_old=0, tags=(tag,))
        item = next(i for i in B.open_items(None) if i.id == iid)
        assert B.is_loop_spawned(item), "both are loop output to expiry"
        assert not B.is_self_spawned(item), "neither is held out of the triage pool"
    fresh, held = B.triage_pool(S.LEDGER_PATH)
    assert sorted(i.id for i in fresh) == [603, 604] and held == 0
    assert not any(t.startswith("spawned-by-") and t not in B.SPAWN_TAGS
                   for t in B.quarantine_tags()), "quarantine_tags is still an exact set"


def test_the_eval_switch_off_still_leaves_the_prefix_recognised(isolated, monkeypatch):
    """Clause 5. The kill switch exists to put YouTube evals back to being a
    human's items; it must not become a switch for the prefix too, or turning it
    off silently un-bounds the 43 forgotten mints as a side effect. Two
    mechanisms, one of which the switch governs."""
    _eval_switch(monkeypatch, False)
    write_item(isolated, 605, days_old=0, tags=(EVAL_TAG,))
    write_item(isolated, 606, days_old=0, tags=(PREFIX_ONLY,))
    eval_item = next(i for i in B.open_items(None) if i.id == 605)
    prefix_item = next(i for i in B.open_items(None) if i.id == 606)
    assert not B.is_loop_spawned(eval_item), "the kill switch keeps catching what it caught"
    assert B.is_loop_spawned(prefix_item), "and catches nothing else differently"


def test_the_shared_prefix_is_the_one_the_writer_stamps():
    """The boundary this whole item turned on: two processes, one string. The
    MCP server that merges a create asks `startswith(_SPAWN_TAG_PREFIX)`; the
    loop that expires the same file asks `startswith(SPAWN_TAG_PREFIX)`. They
    were two literals until #1160, which is exactly how the mint outgrew the
    reader without either side looking wrong on its own."""
    from agent_mcp import backlog as BL
    from app.backlog_tags import SPAWN_TAG_PREFIX, is_spawn_tag
    assert BL._SPAWN_TAG_PREFIX == SPAWN_TAG_PREFIX == "spawned-by-"
    assert is_spawn_tag(PREFIX_ONLY) and not is_spawn_tag("backlog")
    assert not is_spawn_tag("youtube-eval"), "the eval tag has no prefix: it is enumerated"


def test_a_review_write_is_merged_into_an_item_that_already_covers_it(isolated, tmp_path, monkeypatch):
    """The write-time dedupe keys on the `spawned-by-` PREFIX, so it caught
    this tag before the tag existed — pinned because the prefix is the only
    thing that makes that true, and someone will one day list the tags."""
    from agent_mcp import backlog as BL, backlog_similar as SIM
    d = isolated
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    monkeypatch.setattr(SIM, "DEDUPE_LOG", tmp_path / "dedupe.jsonl")
    monkeypatch.setattr(SIM, "dedupe_config", lambda: dict(SIM.DEFAULTS))
    name = "graph_refresh is advertised but never called by any tool"
    body = ("`agent_mcp/code_graph.py:210` defines it and nothing dispatches it; "
            "the staleness rule rebuilds on its own.")
    created = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    (d / "10-existing.md").write_text(
        "---\n" + yaml.dump({"status": "draft", "priority": "medium", "created": created,
                             "board": "lloyd", "tags": ["backlog"]}) +
        f"---\n\n# {name}\n\n{body}\n", encoding="utf-8")
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: [{"id": 10, "score": 0.9}])

    args = {"name": name, "description": body, "board": "lloyd", "tags": [REVIEW_TAG]}
    out = json.loads(BL._handle_write(dict(args)))
    assert out.get("merged_into") == 10, "a review re-run does not re-file its finding"
    assert "Merged finding" in (d / "10-existing.md").read_text()

    forced = json.loads(BL._handle_write({**args, "force": True}))
    assert forced.get("merged_into") is None and forced.get("task_id") != 10


# ===========================================================================
# A re-triaged item's second triage is told what was refused
# ===========================================================================

def test_the_origin_block_carries_the_refusal_of_a_retriaged_item(isolated):
    """Re-triage exists to write a contract a round can meet. A triage that
    is not shown which clauses the grader refused writes the same twelve."""
    write_item(isolated, 910, days_old=20)
    S.append_event({"event": "backlog_retriage", "item_id": 910, "round_id": "SM_REFUSED",
                    "outcome_detail": "review disagreement: clause 2 came back unmet on two reviews",
                    "findings": "clause 2 names a nightly job no pre-landing test can run",
                    "clauses": [{"clause": 1, "verdict": "met", "note": ""},
                                {"clause": 2, "verdict": "unmet", "note": "needs the nightly run"}],
                    "unmet_twice": [2]}, path=S.LEDGER_PATH)
    text = M.render_prompt(B.item_by_id(910), ledger=S.LEDGER_PATH)
    origin = text.split("<origin", 1)[1].split("</origin>", 1)[0]
    assert "RE-TRIAGED" in origin and "SM_REFUSED" in origin
    assert "no pre-landing test can run" in origin
    assert "clause 1 met; clause 2 unmet: needs the nightly run" in origin
    assert "drop them: clause(s) 2" in origin
    assert "A re-triaged item" in text and "write a NEW contract" in text


def test_an_item_never_retriaged_has_no_refusal_line(isolated):
    write_item(isolated, 911, days_old=20)
    assert "RE-TRIAGED" not in M.render_prompt(B.item_by_id(911), ledger=S.LEDGER_PATH).split(
        "</origin>", 1)[0]


def test_the_refusal_line_names_the_refused_clauses_and_the_kept_branch(isolated):
    write_item(isolated, 912, days_old=20)
    S.append_event({"event": "backlog_retriage", "item_id": 912, "round_id": "SM_KEPT",
                    "previous_clauses": ["the retry fires once", "the nightly job reports it"],
                    "unmet_twice": [2], "clauses": []}, path=S.LEDGER_PATH)
    origin = M.render_prompt(B.item_by_id(912), ledger=S.LEDGER_PATH).split("</origin>", 1)[0]
    assert "clause(s) 2. the nightly job reports it" in origin
    assert "1. the retry fires once | 2. the nightly job reports it" in origin
    assert "`automod/SM_KEPT`" in origin
