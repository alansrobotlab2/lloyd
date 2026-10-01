"""What an item still owes is settled by Lloyd, and nothing parks on Alan.

2026-09-27: 257 closed items carried `needs-human` and nothing ever came back
for the tag. A hand sweep found 95 already done, 30 moot, 20 waiting on a date,
68 policy calls Alan had delegated anyway, 63 pieces of leftover work, and 10
that needed his hands. Alan: "i don't want any more needs-human … lloyd can
approve his own choices now." These pin the replacement: the `owed` list
(`scripts/automod/owed.py`), the producers that write it, and the owed-check
job (`workers/sources/owed_check.py`) that settles it.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, owed as O, state as S
from workers.sources import owed_check as OC

ROOT = Path(__file__).resolve().parents[1]


def write_item(d: Path, item_id, *, status="draft", name="A thing", tags=("backlog",),
               extra=None) -> Path:
    fm = {"status": status, "priority": "medium", "board": "lloyd",
          "created": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
    if tags is not None:
        fm["tags"] = list(tags)
    fm.update(extra or {})
    path = d / f"{item_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\nBody.\n",
                    encoding="utf-8")
    return path


def fm_of(path: Path) -> dict:
    return B._split_frontmatter(path.read_text(encoding="utf-8"))[0]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


# ── The list ────────────────────────────────────────────────────────────────

def test_add_owed_appends_dedupes_and_tolerates_plain_strings(isolated):
    p = write_item(isolated, 1, extra={"owed": ["an old hand-written entry"]})
    assert O.add_owed(p, ["check the nightly", "check the nightly", ""], kind="check")
    assert not O.add_owed(p, ["check the nightly"]), "already owed: nothing written"
    entries = O.entries_of(fm_of(p))
    assert [e["what"] for e in entries] == ["an old hand-written entry", "check the nightly"]
    assert entries[1]["kind"] == "check" and entries[1]["since"]


def test_owing_items_orders_oldest_owed_first_and_honours_recheck_dates(isolated):
    future = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_item(isolated, 2, extra={"owed": [{"what": "b", "since": "2026-09-20T00:00:00"}]})
    write_item(isolated, 3, extra={"owed": [{"what": "a", "since": "2026-09-10T00:00:00"}]})
    write_item(isolated, 4, extra={"owed": [{"what": "later", "since": "2026-09-01T00:00:00",
                                             "recheck_after": future}]})
    write_item(isolated, 5, status="done", extra={"owed": [{"what": "closed items owe too",
                                                              "since": "2026-09-15T00:00:00"}]})
    assert [o.item.id for o in O.owing_items()] == [3, 5, 2]
    assert 4 in [o.item.id for o in O.owing_items(due_only=False)]


def test_a_recheck_date_is_clamped_to_thirty_days():
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    assert O.recheck_date("2027-06-01", now).startswith("2026-10-27")
    assert O.recheck_date("2026-09-20", now).startswith("2026-09-27T01"), "never in the past"
    assert O.recheck_date("not a date", now).startswith("2026-09-28")


# ── Applying an answer ──────────────────────────────────────────────────────

def _owing(isolated, item_id, whats, *, status="done", tags=("backlog",), **entry):
    p = write_item(isolated, item_id, status=status, tags=tags,
                   extra={"owed": [{"what": w, "kind": "check", "since": "2026-09-20T00:00:00",
                                    **entry} for w in whats]})
    return p, O.entries_of(fm_of(p))


def test_settled_records_the_evidence_and_clears_the_entry(isolated):
    p, entries = _owing(isolated, 10, ["confirm the nightly shows it", "still waiting"])
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "settled",
                                        "evidence": "run_82 at 13:00 shows it"}], item_id=10)
    fm = fm_of(p)
    assert [e["what"] for e in O.entries_of(fm)] == ["still waiting"], "unanswered stays owed"
    assert fm["owed_settled"][0]["evidence"] == "run_82 at 13:00 shows it"
    assert out["remaining"] == 1
    assert "owed-check" in fm["activity_log"][-1]


def test_recheck_sets_a_date_and_counts_until_it_must_be_ruled(isolated):
    p, entries = _owing(isolated, 11, ["a week of traffic"])
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "recheck", "evidence": "not yet",
                                  "recheck_after": "2026-10-04"}], item_id=11)
    e = O.entries_of(fm_of(p))[0]
    assert e["recheck_after"] and e["rechecks"] == 1
    p2, entries2 = _owing(isolated, 12, ["forever"], rechecks=O.MAX_RECHECKS)
    O.apply_verdict(p2, entries2, [{"n": 1, "outcome": "recheck", "evidence": "still nothing"}],
                    item_id=12)
    fm = fm_of(p2)
    assert O.entries_of(fm) == [], "a fifth recheck is refused: it is ruled on"
    assert fm["owed_settled"][0]["outcome"] == "ruling"


def test_work_files_a_draft_item_on_lloyd_and_links_it(isolated):
    p, entries = _owing(isolated, 20, ["the skill doc still says the old thing"])
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "work", "evidence": "SKILL.md:31",
                                        "follow_up": {"name": "Fix SKILL.md:31",
                                                      "body": "Say what the job does now."}}],
                          item_id=20)
    assert len(out["filed"]) == 1
    new = next(isolated.glob(f"{out['filed'][0]}-*.md"))
    nfm = fm_of(new)
    assert nfm["status"] == "draft" and nfm["board"] == "lloyd"
    assert "#20" in new.read_text()
    assert fm_of(p)["owed_settled"][0]["follow_up"] == out["filed"][0]


def test_a_placeholder_follow_up_is_never_filed(isolated):
    """The stand-ins owed-check has actually produced are never filed; an
    instruction-shaped follow-up still is.

    #1772 was filed as "Placeholder" / "Placeholder body" on 2026-09-28 beside a
    ruling that needed no follow-up, and #1932 as "ignore" / "ignore" on
    2026-09-30 — a word no denylist held, which is why the guard is structural now
    (`owed.is_real_follow_up`). All five forms are refused here: the two
    "Placeholder"/"TODO" ones by the word list, "ignore"/"IGNORE" by the
    name-equals-body test, "nothing to do" by the fragment test. A ruling still
    records, `work` stays owed, and no file appears — then the same pass on a
    second item, with a body that is an instruction, files exactly one draft.
    """
    before = sorted(p.name for p in isolated.glob("*.md"))
    stand_ins = [{"name": "Placeholder", "body": "Placeholder body"},
                 {"name": "TODO", "body": ""},
                 {"name": "ignore", "body": "ignore"},
                 {"name": " IGNORE ", "body": "Ignore."},
                 {"name": "skip", "body": "nothing to do"}]
    p, entries = _owing(isolated, 22, ["raise max_tokens?"] + [f"owed {i}" for i in range(1, 6)])
    out = O.apply_verdict(p, entries, [
        {"n": 1, "outcome": "ruling", "evidence": "x", "ruling": "no", "follow_up": stand_ins[0]},
    ] + [{"n": n, "outcome": "work", "evidence": "y", "follow_up": stand_ins[n - 2]}
         for n in range(2, 7)], item_id=22)
    assert out["filed"] == [], f"a stand-in form was filed: {out['filed']}"
    assert sorted(p.name for p in isolated.glob("*.md")) == before + [p.name], \
        "a refused stand-in still created a backlog file"
    fm = fm_of(p)
    assert fm["owed_settled"][0]["outcome"] == "ruling", "the ruling is still recorded"
    assert [e["what"] for e in O.entries_of(fm)] == [f"owed {i}" for i in range(1, 6)], \
        "unfiled work stays owed"
    for follow in stand_ins:
        assert not O.is_real_follow_up(follow), f"the guard let through {follow}"

    # The same pass, one body that is an instruction: it files, and files once.
    p2, entries2 = _owing(isolated, 23, ["the skill doc still says the old thing"])
    out2 = O.apply_verdict(p2, entries2, [
        {"n": 1, "outcome": "work", "evidence": "SKILL.md:31",
         "follow_up": {"name": "Fix SKILL.md:31", "body": "Say what the job does now."}}],
        item_id=23)
    assert out2["filed"] and len(out2["filed"]) == 1, out2["filed"]
    assert O.is_real_follow_up({"name": "Fix SKILL.md:31", "body": "Say what the job does now."})


def test_a_name_and_a_body_that_are_the_same_word_are_refused():
    """Answering one blank twice is a stand-in, whatever the word is.

    #1932 arrived as `ignore` / `ignore`: not on any denylist, and no list of
    stand-in words ends, because the corpus is whatever the model next types into
    an optional field (`owed._bare` compares the two answers with surrounding
    punctuation stripped, so `" IGNORE "` and `"Ignore."` are the same word too).
    `"Ignore."` is the case that proves this test does work the fragment test does
    not: that body carries a full stop, so its shape looks like a sentence, and
    only its equality with the name gives it away. The controls show no vocabulary
    is needed either side: a body that begins with the word "Ignore" and says
    something is filed, and a one-word name with a real body is filed.
    """
    assert not O.is_real_follow_up({"name": "ignore", "body": "ignore"})
    assert not O.is_real_follow_up({"name": " IGNORE ", "body": "Ignore."})
    assert not O.is_real_follow_up({"name": "unchanged", "body": "Unchanged"})

    assert O.is_real_follow_up({"name": "Triage the sweep",
                                "body": "Ignore the 3 stale rows."})
    assert O.is_real_follow_up({"name": "Rebase",
                                "body": "Rebase the fixture onto the current row count."})


def test_a_body_that_is_a_fragment_rather_than_a_sentence_is_refused():
    """#1933's second shape test, and the half of it that must NOT fire.

    `skip` / `nothing to do` was filed-adjacent junk: 13 characters with no
    sentence terminator anywhere in it. The conjunction matters as much as the
    rule, so both halves are pinned: a short body that does carry a terminator
    passes this test on its own ("Do it." is an instruction, however terse), and a
    body at or over `MIN_INSTRUCTION_CHARS` passes on length alone — the live
    replay in `owed.py` found a real 53-character follow-up (`#1915`) that only a
    conjunction leaves intact.
    """
    assert O.MIN_INSTRUCTION_CHARS == 40
    assert not O.is_real_follow_up({"name": "skip", "body": "nothing to do"})
    assert not O.is_real_follow_up({"name": "Fix the doc", "body": "see the note"})

    assert O.is_real_follow_up({"name": "Skip the check", "body": "Do it."}), \
        "a terminator-bearing body is not refused by this test on its own"
    over_floor = "Rename the sweep timer to match the job id"
    assert len(over_floor) >= O.MIN_INSTRUCTION_CHARS and "." not in over_floor
    assert O.is_real_follow_up({"name": "Rename the timer", "body": over_floor}), \
        "the rule is length AND no-terminator, not length alone"
    under_floor = "Rename the sweep timer to match its"
    assert len(under_floor) < O.MIN_INSTRUCTION_CHARS
    assert not O.is_real_follow_up({"name": "Rename the timer", "body": under_floor})


def test_follow_ups_past_the_cap_stay_owed(isolated):
    p, entries = _owing(isolated, 21, ["one", "two"])
    # The body is instruction-shaped on purpose: a stand-in here would be refused
    # by the guard and the node would test the wrong refusal.
    answers = [{"n": n, "outcome": "work", "evidence": "x",
                "follow_up": {"name": f"work {n}", "body": "Do the thing it names."}}
               for n in (1, 2)]
    out = O.apply_verdict(p, entries, answers, item_id=21, spawn_cap=1)
    assert len(out["filed"]) == 1
    assert [e["what"] for e in O.entries_of(fm_of(p))] == ["two"]


def test_outside_goes_on_the_one_list_and_never_on_a_tag(isolated):
    p, entries = _owing(isolated, 30, ["install the CA certificate"])
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "outside", "evidence": "needs sudo",
                                  "outside": "run update-ca-trust as root"}], item_id=30)
    fm = fm_of(p)
    assert O.entries_of(fm) == []
    assert B.NEEDS_HUMAN_TAG not in (fm.get("tags") or [])
    listed = O.outside_list()
    assert listed == [{"item_id": 30, "name": "A thing", "what": "install the CA certificate",
                       "needs": "run update-ca-trust as root", "since": listed[0]["since"]}]


def test_close_closes_an_open_item_through_the_shared_recorder(isolated):
    p, entries = _owing(isolated, 40, ["its attempts are spent: decide"], status="draft")
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "close", "evidence": "tried twice",
                                        "ruling": "tried twice; not worth a third"}], item_id=40)
    fm = fm_of(p)
    assert out["moved"] == "closed"
    assert fm["status"] == "done" and fm.get("completed") and fm["closed_by"] == "owed-check"
    assert "not worth a third" in fm["activity_log"][-1]


def test_close_and_reopen_leave_a_closed_item_alone(isolated):
    p, entries = _owing(isolated, 41, ["decide"], status="done")
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "reopen", "evidence": "x",
                                        "ruling": "try again"}], item_id=41)
    assert out["moved"] == "" and fm_of(p)["status"] == "done"


def test_reopen_without_an_attempt_on_record_puts_it_back_in_the_pool(isolated):
    p, entries = _owing(isolated, 42, ["decide"], status="draft")
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "reopen", "evidence": "x",
                                        "ruling": "worth a round"}], item_id=42)
    assert out["moved"] == "up_next" and fm_of(p)["status"] == "up_next"


def test_applying_invents_no_tags_field_and_strips_the_legacy_tag(isolated):
    p, entries = _owing(isolated, 50, ["x"], tags=None)
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "settled", "evidence": "e"}], item_id=50)
    assert "tags" not in fm_of(p)
    p2, entries2 = _owing(isolated, 51, ["x"], tags=("backlog", B.NEEDS_HUMAN_TAG))
    O.apply_verdict(p2, entries2, [{"n": 1, "outcome": "settled", "evidence": "e"}], item_id=51)
    assert fm_of(p2)["tags"] == ["backlog"]


# ── The producers ───────────────────────────────────────────────────────────

def test_no_code_adds_the_needs_human_tag():
    """The tag is retired as a destination: nothing in the loop, the routes or
    the workers adds it. Removal is still allowed (legacy files carry it)."""
    add = re.compile(r"\b(add_tags|(?<!_)tags|add)\s*=\s*\(\s*NEEDS_HUMAN_TAG"
                     r"|(?<!remove)\s\[\s*NEEDS_HUMAN_TAG\s*\]")
    hits = []
    for base in ("scripts", "workers", "app", "agent_mcp"):
        for f in (ROOT / base).rglob("*.py"):
            if "/tests/" in str(f):
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                tuple_add = "(NEEDS_HUMAN_TAG," in code.replace(" ", "") and "remove" not in code
                if add.search(code) or tuple_add:
                    hits.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()}")
    assert hits == [], hits


def test_a_protected_path_a_round_needed_is_owed_not_tagged(isolated):
    p = write_item(isolated, 60)
    added = B.record_human_paths(60, [{"path": "config.yaml", "reason": "needs a new key"}],
                                 round_id="SM_60")
    fm = fm_of(p)
    assert added == ["config.yaml"]
    assert B.NEEDS_HUMAN_TAG not in fm["tags"]
    assert [(e["kind"], e["what"]) for e in O.entries_of(fm)] == [
        ("path", "apply `config.yaml`: needs a new key")]


@pytest.mark.parametrize("verdict", ["not_code", "unverifiable"])
def test_a_parking_triage_verdict_is_a_decision_owed(isolated, verdict):
    p = write_item(isolated, 70)
    B.record_verdict(B.item_by_id(70), verdict, "it is a purchase, not code")
    entries = O.entries_of(fm_of(p))
    assert [e["kind"] for e in entries] == ["decide"]
    assert verdict in entries[0]["what"]


def test_a_confirmed_human_only_contract_is_a_decision_owed(isolated):
    p = write_item(isolated, 71)
    B.record_verdict(B.item_by_id(71), "confirmed", "real",
                     acceptance="human-only: config.yaml needs the key")
    assert [e["kind"] for e in O.entries_of(fm_of(p))] == ["decide"]


def test_a_retiring_verdict_owes_nothing(isolated):
    p = write_item(isolated, 72)
    B.record_verdict(B.item_by_id(72), "stale", "gone", close=True)
    assert O.entries_of(fm_of(p)) == []


# ── The job ─────────────────────────────────────────────────────────────────

def test_parse_answer_keeps_only_due_entries_and_known_outcomes():
    got = OC.parse_answer({"entries": [
        {"n": 1, "outcome": "settled", "evidence": "e"},
        {"n": 2, "outcome": "ask alan", "evidence": "e"},
        {"n": 9, "outcome": "settled", "evidence": "e"},
        {"n": "x", "outcome": "settled"}], "summary": "s"}, {1, 2})
    assert [a["n"] for a in got["entries"]] == [1]
    assert OC.parse_answer({"entries": []}, {1}) is None
    assert OC.parse_answer("not an object", {1}) is None


def test_the_prompt_says_nothing_goes_back_to_alan_and_names_every_outcome(isolated):
    p, _ = _owing(isolated, 80, ["confirm the nightly"])
    owing = O.owing_items()[0]
    text = OC.build_prompt(owing)
    assert "nothing may be handed \nback to him" in text or "handed back to him" in text.replace("\n", " ")
    for outcome in O.OUTCOMES:
        assert f"`{outcome}`" in text
    assert "1. [check" in text and "confirm the nightly" in text


def _fake_session(structured):
    async def fake(prompt, **kw):
        fake.calls.append({"prompt": prompt, **kw})
        return {"structured": structured, "session_id": "sess_owed", "stop_reason": "stop"}
    fake.calls = []
    return fake


class _Job:
    def __init__(self, payload):
        self.payload = payload


def test_execute_applies_the_answer_and_records_it(isolated, monkeypatch):
    from workers.sources import _common as C
    p, _ = _owing(isolated, 90, ["confirm it", "decide the knob"], status="done")
    fake = _fake_session({"entries": [
        {"n": 1, "outcome": "settled", "evidence": "server.err is clean since landing"},
        {"n": 2, "outcome": "ruling", "evidence": "no measured gain",
         "ruling": "leave the knob unset"}], "summary": "both settled"})
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    out = asyncio.run(OC.execute(_Job({"apply": True})))
    assert out["status"] == "success" and "1 ruling, 1 settled" in out["summary"]
    assert O.entries_of(fm_of(p)) == []
    assert "Edit" in fake.calls[0]["extra_disallowed"] and "Bash" not in fake.calls[0]["extra_disallowed"]
    rows = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "owed_check"]
    assert rows[-1]["apply"] is True and rows[-1]["item_id"] == 90


def test_a_dry_run_writes_nothing_and_is_not_offered_again(isolated, monkeypatch):
    from workers.sources import _common as C
    p, _ = _owing(isolated, 91, ["confirm it"])
    before = p.read_text()
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_session(
        {"entries": [{"n": 1, "outcome": "settled", "evidence": "e"}], "summary": "s"}))
    out = asyncio.run(OC.execute(_Job({"apply": False})))
    assert out["status"] == "success" and "dry run" in out["summary"]
    assert p.read_text() == before
    assert OC._dry_answered() == {91}
    assert asyncio.run(OC.execute(_Job({"apply": False})))["status"] == "skipped"


def test_no_answer_defers_the_item_without_spending_a_recheck(isolated, monkeypatch):
    from workers.sources import _common as C
    p, _ = _owing(isolated, 92, ["confirm it"])
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_session(None))
    out = asyncio.run(OC.execute(_Job({"apply": True})))
    assert out["status"] == "failed"
    e = O.entries_of(fm_of(p))[0]
    assert e["recheck_after"] and not e.get("rechecks")


def test_board_health_carries_the_owed_counts_and_the_outside_list(isolated):
    _owing(isolated, 95, ["a", "b"])
    p, entries = _owing(isolated, 96, ["needs sudo"])
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "outside", "evidence": "e",
                                  "outside": "sudo update-ca-trust"}], item_id=96)
    owed = B.board_health(S.LEDGER_PATH)["owed"]
    assert owed["items"] == 1 and owed["entries"] == 2 and owed["due"] == 2
    assert [o["item_id"] for o in owed["outside"]] == [96]


def test_the_source_is_registered_and_configured():
    from app.config import CONFIG
    from workers.sources import SOURCE_REGISTRY
    assert SOURCE_REGISTRY["owed-check"] is OC
    cfg = CONFIG["workers"]["sources"]["owed-check"]
    assert cfg["enabled"] is True and cfg["max_inflight"] == 1 and cfg["apply"] is True
    assert cfg["inner_voice"] is False


def test_one_write_that_untags_and_owes_keeps_what_it_owes(isolated):
    """The take-back rule acts on the item as it was, never on the caller's
    own update: the 2026-09-27 migration wrote decide entries and removed the
    legacy tag in one `update_frontmatter` call, and the take-back erased the
    entries it had just written."""
    p = write_item(isolated, 97, tags=("backlog", B.NEEDS_HUMAN_TAG),
                   extra={"owed": [{"what": "stale decision", "kind": "decide"}]})
    B.update_frontmatter(p, {"owed": [{"what": "fresh decision", "kind": "decide"}]},
                         remove_tags=(B.NEEDS_HUMAN_TAG,))
    fm = fm_of(p)
    assert [e["what"] for e in O.entries_of(fm)] == ["fresh decision"]
    assert fm["tags"] == ["backlog"]


def test_answering_a_legacy_tagged_item_keeps_its_unanswered_decisions(isolated):
    p = write_item(isolated, 98, tags=("backlog", B.NEEDS_HUMAN_TAG),
                   extra={"owed": [{"what": "one", "kind": "check"},
                                   {"what": "two", "kind": "decide"}]})
    O.apply_verdict(p, O.entries_of(fm_of(p)),
                    [{"n": 1, "outcome": "settled", "evidence": "e"}], item_id=98)
    assert [e["what"] for e in O.entries_of(fm_of(p))] == ["two"]


# ── #1909: the tick takes a batch, and an emptied open item does not survive ────

def test_a_tick_offers_several_due_items_oldest_first_one_row_each(isolated, tmp_path):
    """#1909 clause 4: the drain is bounded by items per TICK, not per session.

    Four owing items, their oldest entry aged 20 → 12 → 6 → 2 days: the tick takes
    three and the one it leaves for the next tick is the NEWEST, because the pool
    that needed draining on 2026-09-30 was 191 due entries across 76 items, a tick
    that reached one item settled maybe two entries, and 92 new entries were filed
    the same day. Each item gets exactly one queue row and the dedup key is per
    item — the old board-wide key is what turned a retry into a second row for the
    same item (#1418) — so a second tick adds nothing while those three are live.
    """
    from workers.queue import WorkQueue
    q = WorkQueue(tmp_path / "owed-batch.db")
    _owing(isolated, 11, ["read the run for #11"], since="2026-09-25T00:00:00")
    _owing(isolated, 12, ["read the run for #12"], since="2026-09-10T00:00:00")
    _owing(isolated, 13, ["read the run for #13"], since="2026-09-18T00:00:00")
    _owing(isolated, 14, ["read the run for #14"], since="2026-09-12T00:00:00")

    asyncio.run(OC.enqueue_if_due(q, {"apply": True}))
    rows = sorted((r for r in q.list_items(source=OC.NAME) if r.state == "queued"),
                  key=lambda r: r.id)
    assert [r.payload["item_id"] for r in rows] == [12, 14, 13], \
        "a batch has to lead with what has waited longest"
    assert len({r.dedup_key for r in rows}) == 3, "one row per item, keyed per item"
    assert q.has_live(f"{OC.DEDUP_KEY}:12") and not q.has_live(OC.DEDUP_KEY)
    asyncio.run(OC.enqueue_if_due(q, {"apply": True}))
    assert len(q.list_items(source=OC.NAME)) == 3, "a retry reuses the live row"


def test_the_batch_size_is_configuration_not_a_one_way_door(isolated, tmp_path):
    """`batch: 1` reproduces the old one-item tick, and an unset `batch` keeps the
    source's default, so widening the drain is a config edit and not a round."""
    from workers.queue import WorkQueue
    assert OC.DEFAULT_BATCH >= 3
    q = WorkQueue(tmp_path / "owed-one.db")
    _owing(isolated, 11, ["read the run for #11"], since="2026-09-25T00:00:00")
    _owing(isolated, 12, ["read the run for #12"], since="2026-09-10T00:00:00")
    _owing(isolated, 13, ["read the run for #13"], since="2026-09-18T00:00:00")
    asyncio.run(OC.enqueue_if_due(q, {"apply": True, "batch": 1}))
    rows = q.list_items(source=OC.NAME)
    assert len(rows) == 1
    assert rows[0].payload["item_id"] == 12


def test_an_open_item_whose_last_entry_settles_is_closed_not_left_a_draft(isolated):
    """#1909 clause 5: the sweep that empties the list is the sweep that closes.

    A `decide` entry on a draft — the shape a `human-only:` guard leaves behind —
    ruled `settled` with evidence: the entry goes, the list is empty, and the item
    used to stay a draft in exactly that state with nothing owed on it, invisible
    to a job that visits only items that DO owe something. It now closes as
    `owed-check` settled, the move the hand sweep made to 11 items on 2026-09-30
    morning, with the closing evidence on `completed_via`.

    The two ways a list empties WITHOUT the question being answered keep the item
    open: an entry ruled `outside` is a thing only a person can do, and one sent
    for recheck still owes its re-reading.
    """
    p, entries = _owing(isolated, 100, ["decide: is the CA check still owed?"],
                        status="draft")
    out = O.apply_verdict(p, entries,
                          [{"n": 1, "outcome": "settled", "evidence": "exited 0 on 09-28"}],
                          item_id=100)
    fm = fm_of(p)
    assert O.entries_of(fm) == [] and out["moved"] == "closed"
    assert fm["status"] == "done" and fm["closed_by"] == "owed-check"
    assert "exited 0 on 09-28" in str((fm.get("activity_log") or [])[-1]), \
        "the close has to say which ruling closed it"
    assert fm.get("completed"), "a close stamps `completed:` like every other closer"

    p2, e2 = _owing(isolated, 101, ["decide: is the CA check still owed?"], status="draft")
    out2 = O.apply_verdict(p2, e2,
                           [{"n": 1, "outcome": "outside", "evidence": "needs a person"}],
                           item_id=101)
    assert not out2["moved"] and fm_of(p2)["status"] == "draft", \
        "an entry only a human can do is still owed"

    p3, e3 = _owing(isolated, 102, ["recheck: re-read the live run"], status="draft")
    out3 = O.apply_verdict(p3, e3,
                           [{"n": 1, "outcome": "recheck", "evidence": "nothing ran since",
                             "recheck_after": (datetime.now(timezone.utc)
                                               + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")}],
                           item_id=102)
    assert not out3["moved"] and fm_of(p3)["status"] == "draft", \
        "an entry awaiting its re-reading is still owed"
    assert O.entries_of(fm_of(p3)), "and the entry is still on the list"


def test_a_row_settles_the_item_it_was_queued_for_not_the_oldest_one(isolated, tmp_path,
                                                                     monkeypatch):
    """The other half of the boundary: the queue row carries its item across into the
    session.

    A tick that offers several items while `max_inflight` is 1 runs the rows one
    after another, and an `execute` that re-picked "the oldest owing item" at claim
    time would point every session at whatever was oldest when it started — the
    second row would answer for an item nobody queued for it. The payload names the
    item, so the row that says #13 measures #13 even while the older #12 is still
    owing, and #12 keeps its entry unanswered.
    """
    from workers.queue import WorkQueue
    from workers.sources import _common as C
    q = WorkQueue(tmp_path / "owed-execute.db")
    p12 = write_item(isolated, 12, status="done",
                     extra={"owed": [{"what": "read run 90 for #12", "kind": "check",
                                      "since": "2026-09-05T00:00:00"}]})
    p13, _ = _owing(isolated, 13, ["read run 91 for #13"], since="2026-09-20T00:00:00")
    asyncio.run(OC.enqueue_if_due(q, {"apply": True}))
    rows = q.list_items(source=OC.NAME)
    assert len(rows) == 2, "the tick offered both items"

    fake = _fake_session({"entries": [{"n": 1, "outcome": "settled",
                                       "evidence": "run 91 at 07:00 shows it"}],
                          "summary": "settled"})
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    row = next(r for r in rows if r.payload["item_id"] == 13)
    out = asyncio.run(OC.execute(_Job(row.payload)))
    assert out["status"] == "success" and out["summary"].startswith("#13:"), \
        "the session measured the item its row named"
    assert "read run 91 for #13" in fake.calls[0]["prompt"]
    assert "read run 90 for #12" not in fake.calls[0]["prompt"]
    assert O.entries_of(fm_of(p13)) == []
    assert [e["what"] for e in O.entries_of(fm_of(p12))] == ["read run 90 for #12"], \
        "the older item is still owed, unanswered"


def test_a_row_whose_item_stopped_owing_before_it_ran_spends_no_session(isolated, tmp_path,
                                                                       monkeypatch):
    """A batch can be overtaken: an earlier session, or a hand sweep, settles the
    item between the tick and the claim. That row is skipped and runs no session —
    it must not fall through to some other item still on the board."""
    from workers.queue import WorkQueue
    from workers.sources import _common as C
    q = WorkQueue(tmp_path / "owed-gone.db")
    p, entries = _owing(isolated, 14, ["read the run for #14"], since="2026-09-20T00:00:00")
    p15, _ = _owing(isolated, 15, ["read the run for #15"], since="2026-09-21T00:00:00")
    asyncio.run(OC.enqueue_if_due(q, {"apply": True}))
    row = next(r for r in q.list_items(source=OC.NAME) if r.payload["item_id"] == 14)
    # An earlier pass (a hand sweep, or the session queued before this one) settles
    # it while this row sits in the queue.
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "settled", "evidence": "swept by hand"}],
                    item_id=14)
    fake = _fake_session({"entries": [], "summary": "should not run"})
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    out = asyncio.run(OC.execute(_Job(row.payload)))
    assert out["status"] == "skipped" and not fake.calls
    assert [e["what"] for e in O.entries_of(fm_of(p15))] == ["read the run for #15"], \
        "and it did not answer for the other item either"


def test_a_row_queued_before_the_item_id_existed_still_finds_the_oldest_item(isolated,
                                                                            monkeypatch):
    """A row queued by the previous code carries no `item_id`, so the upgrade needs
    no drain of the queue: `execute` falls back to the oldest-owing pick it made
    before batching."""
    from workers.sources import _common as C
    _owing(isolated, 15, ["read run 92 for #15"], since="2026-09-20T00:00:00")
    fake = _fake_session({"entries": [{"n": 1, "outcome": "settled", "evidence": "run 92"}],
                          "summary": "settled"})
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    out = asyncio.run(OC.execute(_Job({"apply": True})))
    assert out["status"] == "success" and out["summary"].startswith("#15:")
