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
    # #2013: a filing resolves citations against these and pins a revision;
    # neither may reach the live vault, the live repo or real git.
    monkeypatch.setattr(O, "CITE_ROOTS", {"vault": tmp_path / "no-vault", "repo": tmp_path / "no-repo"})
    monkeypatch.setattr(O, "_cite_revision", lambda: "")
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


def test_an_open_item_waiting_on_a_decision_is_offered_before_older_checks(isolated):
    """2026-10-05: 16 open items owed a `decide` sat behind 120 older post-landing
    checks. A decision on an open item holds a board slot; a check on a closed one
    does not."""
    future = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_item(isolated, 2, status="done", extra={"owed": [
        {"what": "old check", "kind": "check", "since": "2026-09-10T00:00:00"}]})
    write_item(isolated, 3, extra={"owed": [
        {"what": "open, but only a check", "kind": "check", "since": "2026-09-12T00:00:00"}]})
    write_item(isolated, 6, extra={"owed": [
        {"what": "spent; reopen or close", "kind": "decide", "since": "2026-10-05T00:00:00"}]})
    write_item(isolated, 7, extra={"owed": [
        {"what": "an older decision", "kind": "decide", "since": "2026-10-01T00:00:00"}]})
    write_item(isolated, 8, status="done", extra={"owed": [
        {"what": "a decision on a closed item holds no slot", "kind": "decide",
         "since": "2026-09-01T00:00:00"}]})
    write_item(isolated, 9, extra={"owed": [
        {"what": "not due yet", "kind": "decide", "since": "2026-09-01T00:00:00",
         "recheck_after": future},
        {"what": "a due check beside it", "kind": "check", "since": "2026-09-11T00:00:00"}]})
    assert [o.item.id for o in O.owing_items()] == [7, 6, 8, 2, 9, 3]


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
    # A contract a round can meet: the ledger keeps it in `up_next`. Without one
    # the reopen is refused and stays owed (tests/test_promotion_is_a_request.py).
    S.append_event({"event": "backlog_triage", "item_id": 42, "verdict": "confirmed",
                    "acceptance": "it passes"}, path=S.LEDGER_PATH)
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


def test_a_name_that_leads_with_none_is_not_a_follow_up():
    """2026-10-01: #1998, #2002 and #2020 were filed under these three names —
    the model saying "no follow-up" in the blank where one goes."""
    body = "The ruling is recorded on the parent item and nothing in the tree changes."
    for name in ("None — ruling closes the entry", "None — ruling needs no code",
                 "none - ruling leaves no engineering work", "N/A: nothing owed",
                 "Nothing: the ruling stands"):
        assert not O.is_real_follow_up({"name": name, "body": body}), name


def test_a_name_that_merely_starts_with_a_listed_word_still_files():
    body = "Change the cursor read in owed.py to the last settled row and pin it with a test."
    for name in ("Follow-up: fix the owed cursor", "None of the three callers pass session_id",
                 "Nothing-examined rows: count them in the scorecard", "Non-empty guard for owed"):
        assert O.is_real_follow_up({"name": name, "body": body}), name


# ── #1955: a settled field is bounded visibly, never cut silently ────────────

def _sentences(n: int, last: str = "") -> str:
    """Exactly `n` chars of short sentences, ending with `last` when given."""
    unit = "A sentence that explains the bound. "
    text = (unit * (n // len(unit) + 2))[:n - len(last)].rstrip()
    text = text + " " * (n - len(last) - len(text)) + last
    return " ".join(text.split()).ljust(n, "x")[:n]


def test_a_900_char_ruling_and_a_700_char_evidence_field_are_stored_whole(isolated):
    assert O.SETTLED_TEXT_LIMIT == 1200
    ruling, evidence = _sentences(900), _sentences(700)
    assert (len(ruling), len(evidence)) == (900, 700)
    p, entries = _owing(isolated, 60, ["decide the retention rule"])
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "ruling", "evidence": evidence,
                                  "ruling": ruling}], item_id=60)
    rec = fm_of(p)["owed_settled"][0]
    assert rec["ruling"] == ruling and len(rec["ruling"]) == 900
    assert rec["evidence"] == evidence and len(rec["evidence"]) > 500
    assert "artifact" not in rec, "nothing was cut, so nothing needs a witness"


def test_an_over_bound_ruling_is_cut_at_a_sentence_end_and_marked(isolated):
    ruling = _sentences(1300)
    p, entries = _owing(isolated, 61, ["decide the retention rule"])
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "ruling", "evidence": "measured it",
                                  "ruling": ruling,
                                  "artifact": "~/lloyd-data/sessions/x_owedcheck.json"}],
                    item_id=61)
    rec = fm_of(p)["owed_settled"][0]
    stored = rec["ruling"]
    assert stored.endswith(O.TRUNCATION_MARKER) and O.TRUNCATION_MARKER == " … [truncated]"
    kept = stored[:-len(O.TRUNCATION_MARKER)]
    assert 500 < len(kept) <= O.SETTLED_TEXT_LIMIT
    assert kept.endswith("."), "cut at the last sentence end"
    assert ruling.startswith(kept)
    assert rec["artifact"] == "~/lloyd-data/sessions/x_owedcheck.json"


def test_a_cut_with_no_sentence_end_falls_back_to_whitespace_never_mid_token():
    words = " ".join(["token%04d" % i for i in range(200)])       # no sentence end
    text, cut = O._bounded(words)
    assert cut and text.endswith(O.TRUNCATION_MARKER)
    kept = text[:-len(O.TRUNCATION_MARKER)]
    assert len(kept) <= O.SETTLED_TEXT_LIMIT
    assert kept.split()[-1] in words.split(), "the last kept token is whole"
    assert O._bounded("short") == ("short", False)


def test_a_cut_field_with_no_artifact_named_points_at_its_session(isolated):
    p, entries = _owing(isolated, 62, ["decide"])
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "settled", "evidence": _sentences(1300)}],
                    item_id=62, session_id="20261001_owedcheck_ab12")
    rec = fm_of(p)["owed_settled"][0]
    assert rec["evidence"].endswith(O.TRUNCATION_MARKER)
    assert rec["artifact"] == "20261001_owedcheck_ab12"


def test_the_bound_holds_end_to_end_through_parse_answer(isolated):
    """#1644's case: a 951-char ruling whose last sentence is clause (c)."""
    last = "Clause (c) reopen if a keep-ref ever appears."
    ruling = _sentences(951, last)
    assert len(ruling) == 951 and ruling.endswith(last)
    long = _sentences(1300)
    got = OC.parse_answer({"entries": [
        {"n": 1, "outcome": "ruling", "evidence": "e", "ruling": ruling},
        {"n": 2, "outcome": "ruling", "evidence": "e", "ruling": long,
         "artifact": "~/lloyd-data/sessions/y.json"}], "summary": "s"}, {1, 2})
    assert got["entries"][0]["ruling"] == ruling, "the dispatcher does not cut below the bound"
    assert len(got["entries"][1]["ruling"]) == 1300
    assert "artifact" in OC.OWED_SCHEMA["properties"]["entries"]["items"]["properties"]
    p, entries = _owing(isolated, 63, ["rule on retention", "rule on the other"])
    O.apply_verdict(p, entries, got["entries"], item_id=63)
    first, second = fm_of(p)["owed_settled"]
    assert first["ruling"].endswith(last) and len(first["ruling"]) == 951
    assert second["ruling"].endswith(O.TRUNCATION_MARKER)
    assert second["artifact"] == "~/lloyd-data/sessions/y.json"


# ── #2013: the child's copy of the owed line carries no rotting range ────────

@pytest.fixture
def cite_roots(tmp_path, monkeypatch):
    vault, repo = tmp_path / "vault", tmp_path / "repo"
    (vault / "skills" / "eval").mkdir(parents=True)
    (vault / "skills" / "eval" / "SKILL.md").write_text("x")
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "path.py").write_text("x")
    monkeypatch.setattr(O, "CITE_ROOTS", {"vault": vault, "repo": repo})
    monkeypatch.setattr(O, "_cite_revision", lambda: pytest.fail("no real git in a test"))
    return vault, repo


_FOLLOW = {"name": "Fix the cited paragraph",
           "body": "Change the helper so the paragraph is right. Check it with the unit test."}


def _child_body(isolated, new_id: int) -> str:
    return next(isolated.glob(f"{new_id}-*.md")).read_text(encoding="utf-8")


def test_a_filed_child_drops_the_range_keeps_the_path_and_labels_the_root(isolated, cite_roots):
    what = ("the copy at pkg/path.py:120-140 rots, as does skills/eval/SKILL.md:240-245 "
            "and probe1539.py:7")
    p, entries = _owing(isolated, 70, [what])
    assert O.is_real_follow_up(_FOLLOW)
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "work", "evidence": "seen",
                                        "follow_up": _FOLLOW}], item_id=70, revision="abc1234")
    body = _child_body(isolated, out["filed"][0])
    trailer = body[body.index("Filed by owed-check"):]
    assert ":120-140" not in body and ":240-245" not in body and "probe1539.py:7" not in body
    assert "pkg/path.py (repo-relative)" in trailer
    assert "skills/eval/SKILL.md (vault-relative)" in trailer
    assert "probe1539.py (root unresolved)" in trailer, "a root is never guessed"
    assert "[cited at abc1234]" in trailer
    # The parent keeps what it owed, byte for byte.
    assert fm_of(p)["owed_settled"][0]["what"] == what


def test_a_child_filed_with_no_revision_says_unpinned(isolated, cite_roots, monkeypatch):
    monkeypatch.setattr(O, "_cite_revision", lambda: "")
    p, entries = _owing(isolated, 71, ["the range at pkg/path.py:120-140 is stale"])
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "work", "evidence": "seen",
                                        "follow_up": _FOLLOW}], item_id=71)
    body = _child_body(isolated, out["filed"][0])
    assert "cited at an unpinned revision" in body and ":120-140" not in body
    assert "pkg/path.py (repo-relative)" in body


def test_an_unfiled_entry_keeps_its_text_byte_identical(isolated, cite_roots):
    what = "the range at pkg/path.py:120-140 is stale"
    p, entries = _owing(isolated, 72, [what])
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "work", "evidence": "seen",
                                  "follow_up": {"name": "None", "body": "None"}}],
                    item_id=72, revision="abc1234")
    assert [e["what"] for e in O.entries_of(fm_of(p))] == [what]


def test_cite_for_child_leaves_plain_prose_and_times_alone(cite_roots):
    text = "at 13:00 the run v2.6.0 finished: 3-4 rows, see config.yaml"
    assert O.cite_for_child(text, "r1") == text + " [cited at r1]"


# ── #2055: a clause a close never handed on is derived back into the queue ────
#
# `add_owed(human_clauses_of(...))` runs only in the landing writers, so an item
# closed any other way keeps its human clauses in front matter and never gets an
# `owed:` key, and `owing_items` read `owed` alone. #1999 is the case that proved
# it: `status: done`, three `human_clauses`, no `owed:`, closed 2026-10-01 15:47Z
# by "hand sweep 2026-10-01: landed 70c2d002" with no `## Automod landed` section
# in its body — so the ruling on what to do with the 365 facts filed under entity
# `general` had no item any pass visits. Four nodes below, one per clause.

#: Verbatim in shape from #1999's first `human_clauses:` entry: a data decision
#: over ~/lloyd-data that no diff can settle.
_DISPOSITION = ("Disposition of the 365 facts already filed under entity "
                "`general`: re-file, quarantine, or leave and de-score.")
_READSIDE = "De-score `general` on the read side as defence-in-depth?"


def _stranded(isolated, item_id, clauses, *, completed="2026-10-01T15:47:52"):
    """A `status: done` item that recorded `human_clauses` and has no `owed:` key:
    the shape #1999 was left in, with nothing but the close date to say when the
    clause stopped having an owner."""
    return write_item(isolated, item_id, status="done",
                      extra={"completed": completed, "human_clauses": list(clauses)})


def test_a_done_item_that_recorded_human_clauses_is_owing_again(isolated):
    """Clause 1: `entries_of` yields one due `check` entry for each `human_clauses`
    string that appears in neither `owed` nor `owed_settled`, and `owing_items`
    therefore returns a `status: done` item that has no `owed:` key at all.

    The derivation is at read time and writes nothing: the clause is already on the
    record, the reader simply was not looking where a hand-swept item puts it. The
    entry's `since` is the close that stranded it rather than today, because
    `owing_items` sorts oldest-first and a date invented now would push every one
    of these behind the entries that have genuinely been waiting.
    """
    p = _stranded(isolated, 1999, [_DISPOSITION, _READSIDE])
    fm = fm_of(p)
    assert "owed" not in fm and "owed_settled" not in fm, fm

    entries = O.entries_of(fm)
    assert [e["what"] for e in entries] == [_DISPOSITION, _READSIDE]
    assert [e["kind"] for e in entries] == ["check", "check"], \
        "a clause nobody recorded is a thing to check, not a path or a decision"
    assert [e["origin"] for e in entries] == [O.STRANDED_ORIGIN] * 2
    assert {e["since"] for e in entries} == {"2026-10-01T15:47:52"}

    owing = O.owing_items()
    assert [o.item.id for o in owing] == [1999], "the item is back in the queue"
    assert owing[0].due == [0, 1], "both clauses are due: nothing scheduled them away"

    # And the boundary the derivation does NOT cross: an OPEN item's `human_clauses`
    # are its own round's contract, which the landing route converts (only its
    # post-landing half) and the review rung grades. Deriving those would let the
    # owed job rule on — or `reopen` — a round that is still running.
    open_p = write_item(isolated, 1997, status="up_next",
                        extra={"completed": "", "human_clauses": [_DISPOSITION]})
    assert O.entries_of(fm_of(open_p)) == [], \
        "an open item's clauses are its round's contract, not an owed ruling"


def test_derivation_dedupes_both_lists_and_a_second_pass_writes_no_duplicate(isolated):
    """Clause 2, both halves.

    Half one, the dedupe: an item whose clauses the landing route DID record yields
    no additional entry, counted against `owed` AND against `owed_settled` — an
    item with `owed: [A]` and `human_clauses: [A, B]` owes A once, not twice, and an
    item with A already ruled in `owed_settled` does not owe A either, because a
    clause that has been ruled on is not stranded whatever its text went on to
    become.

    Half two, the idempotence: the owed pass run twice on a derived item writes no
    duplicate. The first pass materialises the clause it answered into
    `owed_settled`, which is also what takes it out of the derived set — a `recheck`
    needs a `recheck_after` somewhere real, and `owed` is the only place that holds
    one — so the second pass sees a settled text and derives nothing from it.
    """
    p_owed = write_item(isolated, 2001, status="done", extra={
        "completed": "2026-10-01T15:47:52", "human_clauses": [_DISPOSITION, _READSIDE],
        "owed": [{"what": _DISPOSITION, "kind": "check", "since": "2026-09-20T00:00:00"}]})
    entries = O.entries_of(fm_of(p_owed))
    assert [e["what"] for e in entries] == [_DISPOSITION, _READSIDE], \
        f"a recorded clause was derived a second time: {[e['what'] for e in entries]}"
    assert [e.get("origin", "") for e in entries] == ["", O.STRANDED_ORIGIN]

    p_settled = write_item(isolated, 2002, status="done", extra={
        "completed": "2026-10-01T15:47:52", "human_clauses": [_DISPOSITION, _READSIDE],
        "owed_settled": [{"what": _DISPOSITION, "outcome": "ruling",
                          "at": "2026-10-01T00:00:00", "evidence": "ruled in triage"}]})
    assert [e["what"] for e in O.entries_of(fm_of(p_settled))] == [_READSIDE]

    p3 = _stranded(isolated, 2003, [_DISPOSITION, _READSIDE])
    O.apply_verdict(p3, O.entries_of(fm_of(p3)),
                    [{"n": 1, "outcome": "settled", "evidence": "rows re-filed 2026-10-02"}],
                    item_id=2003)
    fm = fm_of(p3)
    assert [r["what"] for r in fm["owed_settled"]] == [_DISPOSITION]
    assert [e["what"] for e in O.entries_of(fm)] == [_READSIDE], \
        "answered once: the settled clause is gone and the other is still derived"

    # Numbered 1, not 2: after the first pass the settled clause is out of the
    # list, which is the dedupe doing its work on the numbering the session sees.
    O.apply_verdict(p3, O.entries_of(fm_of(p3)),
                    [{"n": 1, "outcome": "ruling", "evidence": "guard already in place",
                      "ruling": "the read-side guard at retrieval is sufficient"}],
                    item_id=2003)
    fm = fm_of(p3)
    assert [r["what"] for r in fm["owed_settled"]] == [_DISPOSITION, _READSIDE], fm["owed_settled"]
    assert len({r["what"] for r in fm["owed_settled"]}) == 2, "a duplicate settled record"
    assert O.entries_of(fm) == [], "two passes, nothing left owed, nothing written twice"

    # The writer keeps the same line: `add_owed` appends to what the item RECORDS,
    # so an unrelated write (a protected path, a parked decision) cannot silently
    # materialise a derived clause into front matter — it would freeze a read-time
    # derivation into a record whose `origin` says it was filed on purpose.
    p4 = _stranded(isolated, 2005, [_DISPOSITION])
    assert O.add_owed(p4, ["apply `config.yaml`: the key is still owed"], kind="path")
    recorded_after = fm_of(p4)["owed"]
    assert [e["what"] for e in O._recorded(fm_of(p4))] == [
        "apply `config.yaml`: the key is still owed"], recorded_after
    assert [e["what"] for e in O.entries_of(fm_of(p4))] == [
        "apply `config.yaml`: the key is still owed", _DISPOSITION]


def test_the_payload_labels_a_derived_clause_and_leaves_a_recorded_one_unlabelled(isolated):
    """Clause 3: the entry block the owed-check session is actually shown — built by
    `workers.sources.owed_check.build_prompt`, not by a field a caller could ignore —
    says which entries came from `human_clauses` and which a route recorded.

    One item, both kinds in the same list, because the distinction a ruling needs is
    the one it makes while reading: "a previous pass already handled this" is a
    legitimate answer for a recorded entry and a false one for a stranded clause
    that no pass has ever seen. The label is asserted per line so an entry cannot
    borrow the other's marker.
    """
    write_item(isolated, 2004, status="done", extra={
        "completed": "2026-10-01T15:47:52", "human_clauses": [_DISPOSITION],
        "owed": [{"what": "confirm the nightly ran", "kind": "check",
                  "since": "2026-09-20T00:00:00"}]})
    owing = O.owing_items()[0]
    assert len(owing.entries) == 2, owing.entries

    lines = [ln for ln in OC.build_prompt(owing).splitlines()
             if ln.startswith("1. [") or ln.startswith("2. [")]
    assert len(lines) == 2, lines
    labelled = [ln for ln in lines if "derived from human_clauses" in ln]
    assert len(labelled) == 1 and _DISPOSITION in labelled[0], lines
    assert "confirm the nightly ran" not in labelled[0], "the marker reached both entries"
    unlabelled = [ln for ln in lines if "derived from human_clauses" not in ln]
    assert len(unlabelled) == 1 and "confirm the nightly ran" in unlabelled[0], lines


def test_one_run_claims_at_most_the_named_constant_of_stranded_items(isolated, tmp_path):
    """Clause 4: the 454 stranded items on the board on 2026-10-02 cannot be taken
    in one run, and the bound is a named constant, `owed.MAX_STRANDED_PER_RUN`.

    `owing_items` sorts oldest-first and returns everything it is given, so without
    a bound the first tick after this lands reads 454 items, offers them, and puts
    every genuinely recorded entry behind a wall of historical debt. The bound is
    applied AFTER the sort, so which items get ruled on first is the wait they have
    actually done, and it never holds back an item with a recorded entry — the cap
    is on the back-fill, not on anything a route filed on purpose. Both sides of
    the boundary are pinned: the helper, and a real tick whose `batch` is widened
    to 500, which is the config edit the flood would otherwise need.
    """
    from workers.queue import WorkQueue
    assert O.MAX_STRANDED_PER_RUN == 5

    for i in range(O.MAX_STRANDED_PER_RUN + 3):
        _stranded(isolated, 3000 + i, [f"Rule on stranded thing {i}."],
                  completed=f"2026-09-{10 + i:02d}T00:00:00")
    _owing(isolated, 3100, ["a recorded entry, filed newest of all"],
           since="2026-10-01T12:00:00")

    ids = [o.item.id for o in O.owing_items()]
    assert ids == [3000, 3001, 3002, 3003, 3004, 3100], \
        "at most the constant in stranded items, oldest first, and a recorded item is never held back"

    q = WorkQueue(tmp_path / "owed-stranded.db")
    asyncio.run(OC.enqueue_if_due(q, {"apply": True, "batch": 500}))
    rows = [r for r in q.list_items(source=OC.NAME) if r.state == "queued"]
    assert len(rows) == O.MAX_STRANDED_PER_RUN + 1, \
        [r.payload["item_id"] for r in rows]


def test_a_derived_clause_ruled_outside_is_not_derived_or_listed_again(isolated):
    """#538, 2026-10-02: `derived_entries` deduped against `owed` and `owed_settled`
    but not `owed_outside`, so a stranded clause ruled `outside` was back in the
    queue on the next tick and each pass appended another entry to Alan's list —
    six in 28 minutes. An `outside` ruling takes the clause out of the derived set,
    and a clause ruled `outside` twice is one entry, restated, with its first date.
    """
    p = _stranded(isolated, 4000, [_DISPOSITION])
    entries = O.entries_of(fm_of(p))
    assert [e["what"] for e in entries] == [_DISPOSITION]
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "outside", "evidence": "e",
                                  "outside": "attach a drive"}], item_id=4000)
    assert O.entries_of(fm_of(p)) == [], "ruled outside: no longer stranded"
    assert 4000 not in [o.item.id for o in O.owing_items()]
    first = O.outside_list()
    assert [(o["item_id"], o["needs"]) for o in first] == [(4000, "attach a drive")]

    # The same clause answered `outside` again (a recorded entry can be): one row.
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "outside", "evidence": "e",
                                  "outside": "attach a drive, then run the collector"}],
                    item_id=4000)
    again = O.outside_list()
    assert [(o["item_id"], o["needs"]) for o in again] == [
        (4000, "attach a drive, then run the collector")]
    assert again[0]["since"] == first[0]["since"], "it has waited since the first ruling"
