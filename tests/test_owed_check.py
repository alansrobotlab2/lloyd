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
                     acceptance="human-only: .gitignore needs the entry")
    assert [e["kind"] for e in O.entries_of(fm_of(p))] == ["decide"]


# ── 2026-10-05: a review disagreement is owed-check's, and it may reword ────

def _contract(isolated, item_id, clauses):
    p = write_item(isolated, item_id, extra={
        "acceptance_clauses": list(clauses),
        "owed": [{"what": "its unattended attempts are spent; owed-check decides",
                  "kind": "decide", "since": "2026-10-05T11:47:04"}]})
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "finished"},
                   path=S.LEDGER_PATH)
    return p


def test_a_reopen_may_reword_the_clause_the_grader_called_wrong_as_written(isolated):
    """#2228: clause 5 named `backlog/data/gate.json`, #1883's witness; both
    reviews said the dated path was "the right call" and refused the clause "as
    literally written". Reopened alone, the next round meets the same refusal."""
    p = _contract(isolated, 90, ["a — tests/a.py", "copy the report to backlog/data/gate.json"])
    owing = O.owing_items()[0]
    out = O.apply_verdict(p, owing.entries, [{
        "n": 1, "outcome": "reopen", "evidence": "both reviews graded 1 clean",
        "ruling": "the literal path is another item's witness",
        "amend_clause": {"clause": 2, "text": "copy the report to backlog/data/2026-10-05.90-gate-witness.json"},
    }], item_id=90, session_id="sess")
    fm = fm_of(p)
    assert fm["acceptance_clauses"] == ["a — tests/a.py",
                                        "copy the report to backlog/data/2026-10-05.90-gate-witness.json"]
    assert out["moved"] == "reopened" and not O.entries_of(fm, derived=False)
    assert "was: copy the report to backlog/data/gate.json" in "".join(map(str, fm["activity_log"]))
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "clause_amended"]
    assert row and row[0]["by"] == "owed-check" and row[0]["clause"] == 2
    assert row[0]["old"] == "copy the report to backlog/data/gate.json"


@pytest.mark.parametrize("amend", [{"clause": 9, "text": "x"}, {"clause": 1, "text": "  "}])
def test_an_amendment_that_cannot_be_applied_leaves_the_decision_owed(isolated, amend):
    """A reopen whose clause was not reworded repeats the refusal, so neither half
    is applied and the entry stays for the next pass, counted as a recheck."""
    p = _contract(isolated, 91, ["a — tests/a.py"])
    owing = O.owing_items()[0]
    answer = {"n": 1, "outcome": "reopen", "evidence": "e", "ruling": "r", "amend_clause": amend}
    out = O.apply_verdict(p, owing.entries, [answer], item_id=91)
    fm = fm_of(p)
    if amend["text"].strip():
        assert out["moved"] == "" and fm["acceptance_clauses"] == ["a — tests/a.py"]
        (entry,) = O.entries_of(fm, derived=False)
        assert "amendment refused" in entry["note"] and entry["rechecks"] == 1
    else:       # no text is no amendment: a plain reopen
        assert out["moved"] == "reopened"


# ── 2026-11-08: two answers naming one entry are ambiguous, not last-wins ───

#: #3908's own clause 1 shape: a human-only delete no round diff can carry.
_PLANTED_CLAUSE = "delete the five fixture baselines under ~/lloyd-data/eval/baselines — human-only"

#: One owed-check answer set naming entry 1 twice, read from the row
#: `promotions.jsonl` carries at 2026-11-08T00:09:39Z for #3908: the answer
#: carrying the amendment, then a filler, both keyed `n: 1`.
_COLLIDING = [{"n": 1, "outcome": "reopen", "evidence": "no diff can be graded against clause 1",
               "ruling": "clause 1 is rewritten so the next round has something to grade",
               "amend_clause": {"clause": 1,
                                "text": "the sweep gains a store arm for a fixture baseline "
                                        "— scripts/groundskeeper/retention-sweep.py"}},
              {"n": 1, "outcome": "reopen", "evidence": "e", "ruling": "No further entry."}]

#: What the hold says, on the entry and in the returned notes (#3917 clause 4).
_COLLISION_NOTE = "answers collide on n=1, still owed"


def _colliding(isolated, item_id, clauses=(_PLANTED_CLAUSE,)):
    """One owed decision plus a graded clause list, so an amendment COULD be
    applied — the only way to prove it was not."""
    p = _contract(isolated, item_id, list(clauses))
    return p, O.entries_of(fm_of(p), derived=False)


def _events(event: str) -> list[dict]:
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == event]


def test_two_answers_naming_one_entry_leave_it_owed_rather_than_choosing_last(isolated):
    """#3917: `by_n = {int(a["n"]): a for a in answers}` collapsed #3908's two
    answers last-wins, so the amendment was dropped in silence, the row's
    `outcomes` still listed both, and the item reopened on the strength of a
    ruling that never reached it — which is what sent SM_20261108_001921 into the
    refusal the ruling had been written to avoid. An ambiguous answer set applies
    to neither answer."""
    p, entries = _colliding(isolated, 900)
    out = O.apply_verdict(p, entries, _COLLIDING, item_id=900, session_id="sess_collide")
    fm = fm_of(p)
    assert out["remaining"] == 1, "an ambiguous answer set settles nothing"
    assert O.entries_of(fm, derived=False), "the entry stays owed for the next pass"
    assert not fm.get("owed_settled"), "neither answer is recorded as settled"
    assert out["moved"] == "" and fm["status"] == "draft", "no reopen on a dropped ruling"
    assert out["amended"] == [] and not _events("clause_amended")
    assert out["notes"] == [f"#1 {_COLLISION_NOTE}"]
    assert fm["acceptance_clauses"] == [_PLANTED_CLAUSE], "the clause is not reworded"
    (entry,) = O.entries_of(fm, derived=False)
    assert entry["note"] == _COLLISION_NOTE
    assert entry["recheck_after"], "a date bounds the re-fire (#538), not the next tick"
    assert entry.get("rechecks", 0) == 0, (
        "a collision is a fault in the answer format, not a stubborn entry: "
        "MAX_RECHECKS must never close a debt over it")


def test_the_colliding_answer_that_arrives_last_is_dropped_too(isolated):
    """Choosing the FIRST answer would be the same bug mirrored, so the order of a
    colliding set must not be observable: both orderings leave one record."""
    a, ea = _colliding(isolated, 901)
    b, eb = _colliding(isolated, 902)
    first = O.apply_verdict(a, ea, _COLLIDING, item_id=901)
    second = O.apply_verdict(b, eb, list(reversed(_COLLIDING)), item_id=902)
    assert first["moved"] == second["moved"] == ""
    assert first["notes"] == second["notes"] == [f"#1 {_COLLISION_NOTE}"]
    assert ([e["note"] for e in O.entries_of(fm_of(a), derived=False)]
            == [e["note"] for e in O.entries_of(fm_of(b), derived=False)]
            == [_COLLISION_NOTE])
    assert (fm_of(a)["acceptance_clauses"] == fm_of(b)["acceptance_clauses"]
            == [_PLANTED_CLAUSE])
    assert not _events("clause_amended")


def test_only_the_entry_two_answers_named_is_held(isolated):
    """A collision is about the entry that was answered twice, not about the pass:
    every other entry is ruled on exactly as it was before, so one ambiguous
    answer cannot freeze an item with five debts."""
    p = write_item(isolated, 906, extra={"owed": [
        {"what": "collided", "kind": "check", "since": "2026-11-08T00:00:00"},
        {"what": "answered once", "kind": "check", "since": "2026-11-08T00:00:00"}]})
    out = O.apply_verdict(p, O.entries_of(fm_of(p)), [
        {"n": 1, "outcome": "settled", "evidence": "run_1 shows it"},
        {"n": 1, "outcome": "ruling", "evidence": "e", "ruling": "closed as unobservable"},
        {"n": 2, "outcome": "settled", "evidence": "the nightly shows it"}], item_id=906)
    fm = fm_of(p)
    assert [e["what"] for e in O.entries_of(fm)] == ["collided"]
    assert out["remaining"] == 1 and out["moved"] == ""
    assert [r["evidence"] for r in fm["owed_settled"]] == ["the nightly shows it"], \
        "the single answer to entry 2 is applied"
    assert out["notes"] == ["#1 answers collide on n=1, still owed",
                            "#2 settled: the nightly shows it"]


def test_a_colliding_answer_set_writes_no_acceptance_clauses_key_on_an_item_without_one(isolated):
    """#3908's front matter still carries no `acceptance_clauses` key — the one
    `amend_item_clause` writes — which is how the dropped amendment stayed
    invisible on the item a human reads. An item with no clause list to amend ends
    the pass as it started, and is not reopened either."""
    p = write_item(isolated, 903, extra={"owed": [
        {"what": "its unattended attempts are spent; owed-check decides",
         "kind": "decide", "since": "2026-11-08T00:00:00"}]})
    S.append_event({"event": "backlog_implement", "item_id": 903, "phase": "finished"},
                   path=S.LEDGER_PATH)
    out = O.apply_verdict(p, O.entries_of(fm_of(p), derived=False), _COLLIDING, item_id=903)
    fm = fm_of(p)
    assert "acceptance_clauses" not in fm
    assert out["moved"] == "" and out["amended"] == []
    assert O.entries_of(fm, derived=False) and not _events("clause_amended")


def test_the_owed_check_row_carries_amended_empty_so_a_summary_is_checkable(isolated, monkeypatch):
    """The #3908 row's `summary` said clause 1 was rewritten while the same ledger
    held no `clause_amended` event, because the row kept only `remaining`, `filed`
    and `moved`. `result.amended` is what that sentence gets checked against, and
    the job says the collision out loud — a dry run first, before an apply run
    consumes the pass."""
    from workers.sources import _common as C
    p, _ = _colliding(isolated, 904)
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_session(
        {"entries": list(_COLLIDING), "summary": "Entry 1 is reopened, not closed: clause 1 is rewritten"}))
    out = asyncio.run(OC.execute(_Job({"apply": True})))
    assert out["status"] == "success" and "answers collide on #1" in out["summary"]
    row = _events("owed_check")[-1]
    assert row["result"] == {"remaining": 1, "filed": [], "moved": "", "amended": []}
    assert row["collided"] == [1], "the row itself names the collision, beside `remaining: 0`"
    assert row["ok"] is True
    assert not _events("clause_amended")
    assert O.entries_of(fm_of(p), derived=False)


def test_the_owed_check_row_names_the_clause_an_amendment_rewrote(isolated, monkeypatch):
    """The other half of the same clause: when `amend_item_clause` runs its clause
    number is in the row, so the row and the `clause_amended` event agree instead
    of one of them being prose."""
    from workers.sources import _common as C
    p, _ = _colliding(isolated, 905, clauses=("a — tests/a.py", "b — tests/b.py"))
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_session(
        {"entries": [{"n": 1, "outcome": "reopen", "evidence": "both reviews refused b as written",
                      "ruling": "reopened on the reworded clause",
                      "amend_clause": {"clause": 2, "text": "b — tests/b2.py"}}],
         "summary": "clause 2 rewritten"}))
    out = asyncio.run(OC.execute(_Job({"apply": True})))
    assert "answers collide" not in out["summary"], "one answer for one entry is not a collision"
    row = _events("owed_check")[-1]
    assert row["result"]["amended"] == [2] and row["result"]["moved"] == "reopened"
    assert row["collided"] == []
    assert _events("clause_amended")[-1]["clause"] == 2
    assert fm_of(p)["acceptance_clauses"] == ["a — tests/a.py", "b — tests/b2.py"]


def test_a_dry_run_says_the_answers_collide_before_an_apply_run_is_spent(isolated, monkeypatch):
    """The collision is a property of the ANSWER SET, not of the write, so the dry
    run says it out loud: that is the pass a reader has before an apply run spends
    the item on a ruling that will be held."""
    from workers.sources import _common as C
    p, _ = _colliding(isolated, 907)
    before = p.read_text()
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_session(
        {"entries": list(_COLLIDING), "summary": "clause 1 is rewritten"}))
    out = asyncio.run(OC.execute(_Job({"apply": False})))
    assert out["status"] == "success" and "answers collide on #1" in out["summary"]
    assert p.read_text() == before, "a dry run still writes nothing"
    assert _events("owed_check")[-1]["collided"] == [1]


def test_parse_answer_carries_an_amendment_only_on_a_reopen():
    raw = {"entries": [
        {"n": 1, "outcome": "reopen", "evidence": "e", "amend_clause": {"clause": 2, "text": " new  text "}},
        {"n": 2, "outcome": "ruling", "evidence": "e", "amend_clause": {"clause": 1, "text": "x"}},
        {"n": 3, "outcome": "reopen", "evidence": "e", "amend_clause": {"clause": "two", "text": "x"}}],
        "summary": "s"}
    one, two, three = OC.parse_answer(raw, {1, 2, 3})["entries"]
    assert one["amend_clause"] == {"clause": 2, "text": "new text"}
    assert two["amend_clause"] == {} and three["amend_clause"] == {}


def test_the_prompt_carries_the_2026_10_05_ruling_and_keeps_outside_physical(isolated):
    _owing(isolated, 92, ["confirm the nightly"])
    text = " ".join(OC.build_prompt(O.owing_items()[0]).split())
    assert "the loop goes with its own recommendation" in text
    assert "Only something physical waits for him" in text
    assert "A protected path is NOT outside" in text and "spec.CONFIG_DENIED_KEYS" in text
    assert "A review disagreement" in text and "`amend_clause`" in text
    assert "A decision is never outside: make it." in text


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


def test_a_clause_ruled_under_its_old_number_is_not_derived_again(isolated):
    """2026-10-21: `clean_clauses` strips a self-numbered list's `N.` since #3002, so
    `human_clauses: ["1. X", "2. Y"]` derives `X` and `Y` — but rulings written before
    that recorded `1. X`. Compared as exact text, every such item was stranded again and
    owed-check re-ruled 155 of them (up to seven times each). A ruling under the old
    numbered text still answers the clause, in `owed_settled` and `owed_outside` alike,
    and a clause nobody ruled on is still derived."""
    p = write_item(isolated, 4100, status="done", extra={
        "completed": "2026-10-14T17:43:27",
        "human_clauses": [f"1. {_DISPOSITION}", f"2. {_READSIDE}", "3. Nobody ruled on this."],
        "owed_settled": [{"what": f"2. {_READSIDE}", "outcome": "ruling",
                          "at": "2026-10-18T06:19:18", "evidence": "e"}],
        "owed_outside": [{"what": f"1. {_DISPOSITION}", "needs": "hands",
                          "since": "2026-10-18T06:19:18"}]})
    assert [e["what"] for e in O.entries_of(fm_of(p))] == ["Nobody ruled on this."]


# ── The open item ruled `outside` (#3809) ───────────────────────────────────
#
# An `outside` ruling moves a clause out of `owed` and into `owed_outside`, and
# until 2026-11-05 that was the last word an OPEN item ever heard: `_recorded`
# reads `owed` alone and `derived_entries` returns [] for a non-closed status, so
# `entries_of` came back empty and `owing_items`' `if not entries: continue`
# skipped the item — while the human action the ruling asks for is exactly the
# thing that needs re-measuring. Measured 2026-11-05T19:1xZ on HEAD 01821481: 6
# open items carried an `owed_outside` row (#3119, #3359, #3446, #3466, #3477,
# #3782), all 6 with `entries_of(fm) == []` and none of them in the 157 items
# `owing_items(stranded_cap=None)` returns, and 4 of the 6 had been discharged by
# hand at 2026-10-30T03:01:07Z with nothing able to reach them again. The shape is
# `hand_runs()`'s (#3017), which was built open-items-only and deliberately
# unemptyable by an outcome vocabulary; `owed_outside` debt, the one with a human
# in the loop, never got it. These pin the fix: the row reads as one `decide`
# entry at read time, `done` on the row is the one field that clears both
# surfaces, and the row carries its own recheck bound (#538).

#: One human half, as the ruling that asks for it records it.
_OUTSIDE_ASK = "flip task 105 to `frequency: weekly` in the autonomy task file"


def _ruled_outside(isolated, item_id, *, status="draft", name="An outside debt") -> Path:
    """An item whose ONLY debt is one undischarged `owed_outside` row: the exact
    front matter an `outside` ruling leaves behind (`owed` is removed, not emptied)."""
    return write_item(isolated, item_id, status=status, name=name, extra={
        "owed_outside": [{"what": _OUTSIDE_ASK, "needs": "an edit in Alan's editor",
                          "since": "2026-11-05T04:35:47"}]})


def _owed_ids(now=None) -> list[int]:
    return [o.item.id for o in O.owing_items(now=now)]


def test_an_open_item_ruled_outside_is_selected_as_one_decide_entry(isolated):
    """#3809 clause 1: an open item carrying an undischarged `owed_outside` row is
    selected, as exactly one due `decide` entry whose text is the row's."""
    p = _ruled_outside(isolated, 5000)
    fm = fm_of(p)
    assert "owed" not in fm, "the ruling moved the clause out of `owed`"
    assert O._recorded(fm) == [], "the writer's view still records nothing"
    owing = [o for o in O.owing_items() if o.item.id == 5000]
    assert len(owing) == 1, "the human half keeps a queue slot"
    (owing,) = owing
    assert len(owing.entries) == 1 and owing.due == [0], "one entry, due now"
    (entry,) = owing.entries
    assert entry["kind"] == "decide", "nothing moves this item but the deciding job"
    assert entry["what"] == _OUTSIDE_ASK, "the entry is the row, not a paraphrase"
    assert entry["origin"] == O.OUTSIDE_ORIGIN
    assert entry["since"] == "2026-11-05T04:35:47", "it has waited since the ruling"

    # The other side of the seam: the session has to be told what it is measuring.
    prompt = " ".join(OC.build_prompt(owing).split())
    assert _OUTSIDE_ASK in prompt and "never discharged" in prompt, \
        "the prompt names the row as an undischargeable-by-prose debt"


def test_done_on_the_row_clears_it_from_the_queue_and_from_the_list(isolated):
    """#3809 clause 2: ONE field, `done`, governs both edges of the transition —
    the queue slot and Mission Control's list. #3782 is the case that shows the
    two must agree: its row is listed today, and a one-field hand move used to be
    able to make the debt vanish from the list while the queue never saw it at all.

    #3931 amends the QUEUE half of this pin. It used to read `5001 not in
    _owed_ids(), "off the queue"` — true only because nothing could ever read a
    discharged row again, which is the hole #3931 closes (measured
    2026-11-08T08:47Z: 5 open items sat `draft` with every row discharged and no
    pass able to close or re-rule them). `done` still takes the row off the
    UNDISCHARGED queue and off Mission Control's list; what is new is that the
    item comes back as one weekly `check` revisit that only the row's own recheck
    date holds off. The other two assertions stand untouched: the list stays
    empty, and a discharge is still not a close."""
    p = _ruled_outside(isolated, 5001)
    assert 5001 in _owed_ids() and [r["item_id"] for r in O.outside_list()] == [5001]
    fm = fm_of(p)
    B.update_frontmatter(p, {"owed_outside": [dict(fm["owed_outside"][0],
                                                   done="2026-11-05T18:00:00Z")]})
    assert O.outside_list() == [], "and off the one list Alan reads"
    assert fm_of(p)["status"] == "draft", \
        "a discharge is not a close: whether owed-check may close on one is its own ruling"
    (owing,) = [o for o in O.owing_items() if o.item.id == 5001]
    assert len(owing.entries) == 1 and owing.entries[0]["kind"] == "check", \
        "the undischarged debt is gone; what is left is a weekly verification"
    assert not O._holds_a_slot(owing), "verifying a discharge holds no board slot"
    assert O.defer(p, owing.entries, owing.due, why="no structured answer this pass")
    assert 5001 not in _owed_ids(), "off the queue until the recheck date it just stamped"


def test_one_outside_row_is_not_ruled_on_twice_in_a_day(isolated):
    """#3809 clause 3, the #538 bound. `derived_entries` bounds a re-ruling by
    deduping against `owed_outside`, but a synthesised entry IS the row, so that
    dedupe cannot bound it and an empty `recheck_after` is `_due()` on every tick —
    the shape commit dfe056f6 ("a clause ruled outside is one entry, not one per
    tick", #538: six filings in 28 minutes) exists to stop. The bound therefore has
    to live on the row."""
    now = datetime.now(timezone.utc)
    p = _ruled_outside(isolated, 5002)
    first = [o for o in O.owing_items(now=now) if o.item.id == 5002]
    again = [o for o in O.owing_items(now=now) if o.item.id == 5002]
    assert len(first) == len(again) == 1, "reading the board twice is not two slots"
    assert len(first[0].due) == len(again[0].due) == 1, "one row is one due entry"

    (owing,) = first
    assert O.defer(p, owing.entries, owing.due, why="no structured answer this pass")
    assert 5002 not in _owed_ids(now), "the date this pass stamped is one it honours"
    assert 5002 in _owed_ids(now + timedelta(hours=25)), "a day later it is measured again"
    assert fm_of(p)["owed_outside"][0]["recheck_after"], "the bound is on the row"


def test_a_synthesised_entry_is_never_materialised_into_the_owed_key(isolated):
    """#3809 clause 4, #2055: `defer` writes the entry list it is handed straight
    into `owed`, and `apply_verdict`'s `keep` does the same, so a pass over a
    synthesised entry would turn a read-time derivation into front matter — the
    exact side effect `_recorded`'s docstring and `add_owed` exist to prevent. The
    date goes on the ROW instead, which is also what makes the bound readable."""
    now = datetime.now(timezone.utc)
    when = (now + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")

    p = _ruled_outside(isolated, 5003)
    (owing,) = [o for o in O.owing_items(now=now) if o.item.id == 5003]
    assert O.defer(p, owing.entries, owing.due, why="no structured answer this pass")
    fm = fm_of(p)
    assert "owed" not in fm, "deferring the synthesised entry wrote it into `owed`"
    assert fm["owed_outside"][0]["recheck_after"], "the date went on the row"

    p2 = _ruled_outside(isolated, 5004)
    (owing2,) = [o for o in O.owing_items(now=now) if o.item.id == 5004]
    out = O.apply_verdict(p2, owing2.entries,
                          [{"n": 1, "outcome": "recheck", "evidence": "still daily",
                            "recheck_after": when}], item_id=5004)
    fm2 = fm_of(p2)
    assert "owed" not in fm2, "a recheck moved the row's date, not the row into `owed`"
    assert fm2["owed_outside"][0]["recheck_after"] == when, "the date the session named"
    assert out["remaining"] == 0 and fm2["status"] == "draft", \
        "the debt is still there, so the item stays open"


def test_an_outside_row_is_not_bounded_by_the_stranded_cap(isolated):
    """The synthesis is not the #2055 back-fill and must not spend one of its
    `MAX_STRANDED_PER_RUN` slots: a stranded clause is historical debt never seen
    by any pass, while an undischarged `outside` row is a live ruling on a named
    human action. With the cap applied to it, the 454-item bound would decide which
    of Alan's open debts gets measured, and the rest go quiet by the same route
    #3809 is about."""
    for i in range(O.MAX_STRANDED_PER_RUN + 2):
        _stranded(isolated, 6000 + i, [f"Rule on stranded thing {i}."],
                  completed=f"2026-09-{10 + i:02d}T00:00:00")
    _ruled_outside(isolated, 6100)
    ids = _owed_ids()
    assert 6100 in ids, "the outside row is not held back by the back-fill bound"
    assert sum(1 for i in ids if i >= 6000 and i < 6100) == O.MAX_STRANDED_PER_RUN


def test_a_verified_discharge_sets_done_on_the_row_not_a_prose_line(isolated):
    """The discharge has to be structural: 12 backlog files carry an
    `owed_outside discharged by hand: 2026-10-30T03:01:07Z …` line that `git grep
    -n discharg -- scripts/ app/ workers/` shows is parsed by NOTHING, which is why
    four of those items sat `draft` and unreachable for six days. `settled` on a
    synthesised entry is the answer that means "I measured it and Alan's half is
    done", and it writes `done` plus the evidence onto the row — never a
    `owed_settled` record for an entry that was never in `owed`."""
    now = datetime.now(timezone.utc)
    p = _ruled_outside(isolated, 5006)
    (owing,) = [o for o in O.owing_items(now=now) if o.item.id == 5006]
    O.apply_verdict(p, owing.entries,
                    [{"n": 1, "outcome": "settled",
                      "evidence": "autonomy/105-check-health-census-weekly.md:14 is "
                                  "`frequency: weekly` in HEAD"}], item_id=5006)
    fm = fm_of(p)
    row = fm["owed_outside"][0]
    assert row.get("done"), "the row carries its discharge"
    assert "frequency: weekly" in str(row.get("discharge")), "and what proved it"
    assert "owed_settled" not in fm, "no owed entry was settled, so none is recorded"
    assert "owed" not in fm
    assert 5006 not in _owed_ids(now) and O.outside_list() == [], "both surfaces clear"


def test_re_ruling_a_discharged_outside_row_restates_it_not_appends(isolated):
    """#3809 clause 5. The same-clause lookup in `apply_verdict`'s outside branch
    filtered on `not o.get("done")`, so a re-ruling of a clause whose row was
    already discharged fell through to `outside.append(ask)`: a second,
    undischargeable-in-place row for one clause, which `outside_list` then listed
    again. Harmless while open items were never re-ruled; a live double-count the
    moment the queue starts selecting them."""
    p, entries = _owing(isolated, 5005, [_OUTSIDE_ASK], status="draft")
    answer = [{"n": 1, "outcome": "outside", "evidence": "needs the editor",
               "outside": "edit the autonomy task file"}]
    O.apply_verdict(p, entries, answer, item_id=5005)
    B.update_frontmatter(p, {"owed_outside": [dict(fm_of(p)["owed_outside"][0],
                                                   done="2026-11-05T18:00:00Z")]})
    assert O.outside_list() == [], "discharged: off both surfaces"
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "outside",
                                  "evidence": "measured again: still `daily`",
                                  "outside": "edit the autonomy task file"}], item_id=5005)
    rows = fm_of(p)["owed_outside"]
    assert len(rows) == 1, f"one row per clause, restated: {rows}"
    assert not rows[0].get("done"), "the re-ruling took the discharge back"
    assert rows[0]["discharged_before"] == "2026-11-05T18:00:00Z", \
        "a retraction is written where humans read it, not overwritten silently"
    assert [r["item_id"] for r in O.outside_list()] == [5005], "one row listed"


# ── Revisiting a DISCHARGED `owed_outside` row (#3931) ──────────────────────
#
# `done` on a row clears the debt from both surfaces, and until now it also cleared
# the item from owed-check's sight for good: `outside_entries` skips a row carrying
# `done` (owed.py:264) and `owing_items` then drops the item on an empty entry list
# (owed.py:451-452). Measured 2026-11-08T08:47Z at HEAD 1202adc5 over 3858 items:
# exactly 5 OPEN items — #3119, #3359, #3446, #3466, #3477, all `status: draft` —
# had `entries_of(fm) == []` with every row discharged, four of them by hand at
# 2026-10-30T03:01:07Z. Four are satisfied on disk today (autonomy task 30 carries
# `preferred_hours: [20, 21]`, 48 `[8, 9]`, 74 `[11, 12]`, 82 `depends_on: 81` +
# `stale_bypass_hours: 36`, 103 `status: up_next`); #3119's demand RETURNED — its
# row records a 664,788,992 B `~/.cache/qmd/index.sqlite` at the 2026-10-30T04:18:50Z
# swap and `ls -l` on 2026-11-08 reads 1,073,299,456 B, +408 MB in 9 days, with the
# hand action live again. A discharge was the end of the record, so nothing could
# verify one, close on one, or notice one had been retracted. These pin the bounded
# revisit: one read-time `check` entry per such item, re-read no more than every
# `DISCHARGE_RECHECK_DAYS` (7) days, invisible to Mission Control's list, and a
# status move only over rows the writer re-reads as still discharged.

#: A discharge as the 2026-10-30 hand sweep left it. All 5 live rows are exactly
#: `what/needs/since/done` — none carries `recheck_after`, `rechecks` or
#: `discharged_before`, so the first read of this arm cannot require the field.
_DISCHARGE_STAMP = "2026-10-30T03:01:07Z"


def _discharged_outside(isolated, item_id, *, status="draft", done=_DISCHARGE_STAMP,
                        name="An outside debt discharged") -> Path:
    """An item whose ONLY debt is one `owed_outside` row carrying `done`."""
    return write_item(isolated, item_id, status=status, name=name, extra={
        "owed_outside": [{"what": _OUTSIDE_ASK, "needs": "an edit in Alan's editor",
                          "since": "2026-10-19T09:06:29", "done": done}]})


def test_an_item_whose_rows_are_all_discharged_is_revisited_as_one_check_entry(isolated):
    """#3931 clause 1: an OPEN item whose only debt is `owed_outside` rows that all
    carry `done` is selected as exactly one due `check` entry, origin
    `owed_outside`, whose text is the row's own."""
    p = _discharged_outside(isolated, 5010)
    fm = fm_of(p)
    assert "owed" not in fm and not fm.get("owed_settled"), \
        "the fixture is the census shape: no recorded debt but the discharged row"
    assert O._recorded(fm) == [], "and the writer's view still records nothing"
    owing = [o for o in O.owing_items() if o.item.id == 5010]
    assert len(owing) == 1, "a discharged row is still a debt somebody has to verify"
    (owing,) = owing
    assert len(owing.entries) == 1 and owing.due == [0], "exactly one entry, due on the first read"
    (entry,) = owing.entries
    assert entry["kind"] == "check", "verifying a discharge is a measurement, not a held slot"
    assert entry["origin"] == O.OUTSIDE_ORIGIN, "it is the row, not a recorded entry"
    assert entry["what"] == _OUTSIDE_ASK, "the entry is the row's own text, not a paraphrase"
    assert entry["since"] == "2026-10-19T09:06:29", "it has waited since the ruling"
    assert not O._holds_a_slot(owing), "a revisit leaves the item triageable"

    # The other side of the seam: the session has to be told it is re-measuring a
    # discharge. Telling it the #3809 line — "never discharged" — would be false.
    prompt = " ".join(OC.build_prompt(owing).split())
    assert _OUTSIDE_ASK in prompt and "discharged" in prompt, \
        "the prompt names the row as a discharge to verify"
    assert "never discharged" not in prompt, "the row HAS been discharged"


def test_the_discharge_revisit_is_due_at_most_once_every_seven_days(isolated):
    """#3931 clause 2: the bound is a week — not a tick, and not the #538 day. None
    of the 5 live rows carries `recheck_after`, so the first read is due; after a
    pass stamps it the item is not re-read at 6 days and is again at 8."""
    now = datetime.now(timezone.utc)
    assert O.DISCHARGE_RECHECK_DAYS == 7
    assert O.OUTSIDE_RECHECK_HOURS == 24, "the #538 floor is a separate constant"
    p = _discharged_outside(isolated, 5011)
    assert 5011 in _owed_ids(now), "a row with no `recheck_after` is due on the first read"
    (owing,) = [o for o in O.owing_items(now=now) if o.item.id == 5011]
    assert O.defer(p, owing.entries, owing.due, why="no structured answer this pass")
    assert 5011 not in _owed_ids(now + timedelta(days=6)), "six days on, it is not re-read"
    assert 5011 in _owed_ids(now + timedelta(days=8)), "eight days on, it is"
    stamped = datetime.fromisoformat(
        fm_of(p)["owed_outside"][0]["recheck_after"].replace("Z", "+00:00"))
    assert abs((stamped - now).total_seconds() - 7 * 86400) < 60, \
        "the date is the 7-day floor, not `defer`'s 24 hours"
    # The floor is a parameter of the ONE recheck-date function, defaulted from
    # `DISCHARGE_RECHECK_DAYS` — not a second function with its own clamp.
    assert (O.outside_recheck_date("", now, floor_hours=O.DISCHARGE_RECHECK_DAYS * 24)
            == (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert (O.outside_recheck_date("", now)
            == (now + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")), \
        "the default floor an UNDISCHARGED row gets is untouched"


def test_the_revisit_leaves_the_undischarged_and_closed_boundaries_as_they_were(isolated):
    """#3931 clause 3: the two existing boundaries are unchanged. An OPEN item with
    ANY undischarged row is still ruled on as one `decide` entry and is due again 25
    hours after a defer; a CLOSED item carrying a discharged row still yields
    nothing, which is the #538 pin `derived_entries`' dedupe rests on."""
    now = datetime.now(timezone.utc)
    p = _ruled_outside(isolated, 5012)
    (owing,) = [o for o in O.owing_items(now=now) if o.item.id == 5012]
    assert len(owing.entries) == 1 and owing.entries[0]["kind"] == "decide", \
        "an undischarged row is ruled on, not revisited"
    assert O.defer(p, owing.entries, owing.due, why="no structured answer this pass")
    assert 5012 not in _owed_ids(now)
    assert 5012 in _owed_ids(now + timedelta(hours=25)), "due again at 25 hours, not 7 days"

    closed = write_item(isolated, 5013, status="done", name="A closed outside ruling", extra={
        "completed": "2026-11-01T00:00:00",
        "owed_outside": [{"what": _OUTSIDE_ASK, "needs": "an edit in Alan's editor",
                          "since": "2026-10-19T09:06:29", "done": _DISCHARGE_STAMP}]})
    assert O.entries_of(fm_of(closed)) == [], "a CLOSED item's discharged row is a settled ruling"
    assert 5013 not in _owed_ids()

    write_item(isolated, 5014, name="One row open one discharged", extra={
        "owed_outside": [
            {"what": "swap `~/.cache/qmd/index.sqlite` before it fills the disk",
             "needs": "a swap with the daemon stopped", "since": "2026-10-19T09:06:29"},
            {"what": _OUTSIDE_ASK, "needs": "an edit in Alan's editor",
             "since": "2026-10-19T09:06:29", "done": _DISCHARGE_STAMP}]})
    (owing_m,) = [o for o in O.owing_items(now=now) if o.item.id == 5014]
    assert len(owing_m.entries) == 1, "any undischarged row is the debt; the revisit waits"
    assert owing_m.entries[0]["kind"] == "decide"
    assert owing_m.entries[0]["what"] == "swap `~/.cache/qmd/index.sqlite` before it fills the disk"


def test_the_revisit_writes_only_a_date_on_its_own_row(isolated):
    """#3931 clause 4: the revisit is a read. A pass that defers it must not
    materialise it into `owed` (#2055), must not re-stamp the discharge, and must
    not put an action Alan has already done back on Mission Control's list. A
    `recheck` answer naming a shorter wait is raised to the week, which is what
    makes the 7-day bound hold on the answer path and not only on `defer`."""
    now = datetime.now(timezone.utc)
    p = _discharged_outside(isolated, 5015)
    (owing,) = [o for o in O.owing_items(now=now) if o.item.id == 5015]
    assert O.defer(p, owing.entries, owing.due, why="no structured answer this pass")
    fm = fm_of(p)
    assert "owed" not in fm, "deferring the revisit wrote it into `owed`"
    assert fm["owed_outside"][0]["done"] == _DISCHARGE_STAMP, \
        "a read does not re-discharge: the row's `done` is byte for byte what it was"
    assert fm["owed_outside"][0]["recheck_after"], "only the date went on the row"
    assert O.outside_list() == [], "still invisible to the one list Alan reads"

    later = now + timedelta(days=8)
    (owing2,) = [o for o in O.owing_items(now=later) if o.item.id == 5015]
    out = O.apply_verdict(p, owing2.entries,
                          [{"n": 1, "outcome": "recheck",
                            "recheck_after": (later + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "evidence": "cannot verify the task file from here today"}],
                          item_id=5015, now=later)
    fm2 = fm_of(p)
    assert "owed" not in fm2, "a recheck moved the row's date, not the row into `owed`"
    assert fm2["owed_outside"][0]["done"] == _DISCHARGE_STAMP, "the discharge still stands"
    assert out["moved"] == "", "a recheck is not a close"
    assert 5015 not in _owed_ids(later + timedelta(days=1)), \
        "the named 2-hour wait was raised to the 7-day floor"
    assert 5015 in _owed_ids(later + timedelta(days=8)), "the next read is a week on"


def test_the_revisit_closes_only_on_rows_the_writer_re_reads_as_discharged(isolated):
    """#3931 clause 5: `close` is the one outcome that moves the status, and it may
    fire only over rows that STILL carry `done` in the front matter the writer
    re-reads — the same re-read `close_item` already performs before its hand-run
    refusal (owed.py:650-660). A discharge taken back between the read and the
    write is not a close; a re-ruling keeps #3809's existing shape, including the
    row's own `since`, which no code here re-stamps."""
    now = datetime.now(timezone.utc)

    p = _discharged_outside(isolated, 5020)
    (owing,) = [o for o in O.owing_items(now=now) if o.item.id == 5020]
    out = O.apply_verdict(p, owing.entries,
                          [{"n": 1, "outcome": "close",
                            "evidence": "task 30's file carries `preferred_hours: [20, 21]`",
                            "ruling": "the action is on disk; the item is done"}],
                          item_id=5020)
    assert out["moved"] == "closed", "every row still discharged: the close lands"
    fm = fm_of(p)
    assert fm["status"] == "done" and fm["closed_by"] == "owed-check"
    assert fm["owed_outside"][0]["done"] == _DISCHARGE_STAMP, "the row keeps its discharge"
    assert 5020 not in [r["item_id"] for r in O.outside_list()]

    p2 = _discharged_outside(isolated, 5021)
    (owing2,) = [o for o in O.owing_items(now=now) if o.item.id == 5021]
    B.update_frontmatter(p2, {"owed_outside": [
        {k: v for k, v in fm_of(p2)["owed_outside"][0].items() if k != "done"}]})
    out2 = O.apply_verdict(p2, owing2.entries,
                           [{"n": 1, "outcome": "close",
                             "evidence": "measured discharged at the read",
                             "ruling": "closing on the discharge"}], item_id=5021)
    assert out2["moved"] == "", "the write re-read the row and it no longer carried `done`"
    assert fm_of(p2)["status"] == "draft", "so the item stays open"
    assert "close refused" in str((fm_of(p2).get("activity_log") or [])[-1]), \
        "the refusal is written to the item naming why"

    p3 = _discharged_outside(isolated, 5022)
    (owing3,) = [o for o in O.owing_items(now=now) if o.item.id == 5022]
    O.apply_verdict(p3, owing3.entries,
                    [{"n": 1, "outcome": "outside",
                      "evidence": "the index is 1,073,299,456 B again, 9 days after the swap",
                      "outside": "swap the qmd index with the daemon stopped"}],
                    item_id=5022)
    rows = fm_of(p3)["owed_outside"]
    assert len(rows) == 1, f"one row per clause, restated: {rows}"
    assert not rows[0].get("done"), "the re-ruling took the discharge back"
    assert rows[0]["discharged_before"] == _DISCHARGE_STAMP, \
        "a retraction is written where humans read it, not overwritten silently"
    assert rows[0]["since"] == "2026-10-19T09:06:29", "and it keeps the row's own `since`"
    assert fm_of(p3)["status"] == "draft"
    # 5021 is on the list too: its row is the one this test simulated as retracted.
    listed = [r for r in O.outside_list() if r["item_id"] == 5022]
    assert len(listed) == 1, f"back on the human list, one row: {listed}"
    assert listed[0]["needs"] == "swap the qmd index with the daemon stopped", \
        "with the restated ask, not the discharged one"


# ── The pending hand-run (#3017) ────────────────────────────────────────────
#
# A decision can be RULED and still not EXECUTED. PS-3 is the case: seven entity
# pairs sit in `skills/autonomy-data-pipeline/SKILL.md`'s `RULED`, authorized by
# #2765 (2026-10-15T16:03Z) and ruled every-pair by #2986 (2026-10-18T23:41:30Z),
# and the write is a by-hand `--apply` against `~/lloyd-data` that no round and no
# autonomy task may make. Eight carriers went `draft → done` with the apply still
# unrun, because the sweep at the bottom of `apply_verdict` closes an open item the
# moment its last owed entry is settled. #3011's closure is the whole failure in
# four lines: at 2026-10-19T09:06:29Z owed-check answered its single apply-debt
# entry with `outcome: ruling`, `evidence: see entry 1`, `ruling: see entry 1 — one
# entry only`, and the same write moved it `draft → done`. Half an hour later the
# store still read 12 active `name-embedding:` edges and 0 live edges for every one
# of the seven pairs. A debt with no command and no receipt is settled by a
# two-word ruling. These pin the block that makes that impossible: `hand_run:`
# {set, cmd, receipt, since}, whose `receipt` is a command that fails until the run
# has happened.

#: The debt PS-3 is about, as its standing carrier records it.
APPLY_DEBT = ("run SD-stranded-1's --apply by hand: an attended, write-permitted "
              "session, outside a freeze")

#: The one command the hand-run is (`link_stranded_entities.py`'s apply command as
#: SKILL.md's SD-stranded-1 spells it). `cmd` is a record, never something this
#: module runs: only `receipt` is.
APPLY_CMD = ("cd ~/lloyd && .venvs/lloyd/bin/python scripts/memory/link_stranded_entities.py"
             " --min-facts 10 --sample 0"
             " --out ~/lloyd-data/_pipeline/stranded-proposals-<UTCts>.jsonl --apply")

#: PS-3's seven pairs, in the `"source -> target"` spelling the block uses because
#: it is the spelling `SD-STRANDED SETCHECK` prints.
PS3 = ["False Absence Guard Pattern -> Guard",
       "Failure Taxonomy Induction -> Failure Taxonomy",
       "Faros AI -> Faros",
       "Isaac-Lift-Soft-Franka -> Franka",
       "Perplexity Computer -> Perplexity",
       "giodl73-repo -> Repo",
       "OCR-Assisted ASR -> ASR"]


def _hand_run_item(isolated, item_id, *, status="draft", receipt="exit 1",
                   cmd=APPLY_CMD, pairs=None, hand_run=True):
    """An item owing the apply and carrying a `hand_run:` block whose receipt is
    `receipt` — `exit 1` while the by-hand run has not happened, `exit 0` after it
    (where it is the store probe reading active `name-embedding:` 12 + N with each
    of the seven pairs live at `expired_at IS NULL`)."""
    extra: dict = {"owed": [{"what": APPLY_DEBT, "kind": "check",
                             "since": "2026-10-19T09:02:39Z"}]}
    if hand_run:
        extra["hand_run"] = {"set": PS3 if pairs is None else pairs, "cmd": cmd,
                             "receipt": receipt, "since": "2026-10-19T09:02:39Z"}
    p = write_item(isolated, item_id, status=status, extra=extra)
    return p, O.entries_of(fm_of(p))


def test_a_ruling_cannot_settle_the_entry_of_a_hand_run_whose_receipt_still_fails(isolated):
    """#3017 clause 1: the answer that closed #3011 no longer settles anything."""
    p, entries = _hand_run_item(isolated, 300, status="draft")
    out = O.apply_verdict(p, entries,
                          [{"n": 1, "outcome": "ruling", "evidence": "authorized, not run",
                            "ruling": "see entry 1 — one entry only"}], item_id=300)
    fm = fm_of(p)
    assert out["moved"] == "" and fm["status"] == "draft", \
        "a ruled-but-unexecuted hand-run is not closeable"
    assert out["settled"] == 0 and O.SETTLED_KEY not in fm, \
        "the entry does not land in `owed_settled`"
    held = O.entries_of(fm)
    assert [e["what"] for e in held] == [APPLY_DEBT], "it stays in `owed`"
    assert held[0]["recheck_after"] and not held[0].get("rechecks"), \
        "dated so the item is not re-offered every tick, and no recheck is spent"
    assert "receipt 'exit 1' exited 1" in held[0]["note"]
    act = str((fm.get("activity_log") or [])[-1])
    assert "pending hand-run" in act and "exit 1" in act, \
        "the pass writes an activity line naming the failing receipt"


def test_a_close_ruling_on_a_pending_hand_run_is_held_the_same_way(isolated):
    """The other settle-shaped answer: `close` must not reach the closer either."""
    p, entries = _hand_run_item(isolated, 304, status="draft")
    out = O.apply_verdict(p, entries,
                          [{"n": 1, "outcome": "close", "evidence": "ruled long ago",
                            "ruling": "carry it on the board, no round needed"}], item_id=304)
    fm = fm_of(p)
    assert out["moved"] == "" and fm["status"] == "draft"
    assert [e["what"] for e in O.entries_of(fm)] == [APPLY_DEBT]


def test_close_item_refuses_a_pending_hand_run_and_says_which_receipt_failed(isolated):
    """#3017 clause 1 at the closer itself: the shared route, so a caller that
    never went through `apply_verdict` cannot close the item either."""
    p, _ = _hand_run_item(isolated, 302, status="up_next")
    assert O.close_item(p, "every owed entry is settled") is False
    fm = fm_of(p)
    assert fm["status"] == "up_next" and not fm.get("completed"), "the status is untouched"
    act = str((fm.get("activity_log") or [])[-1])
    assert "pending hand-run unexecuted" in act and "receipt 'exit 1' exited 1" in act, \
        "a reader can tell held from never-tried"


def test_a_passing_receipt_lets_the_ordinary_close_run(isolated):
    """#3017 clause 2: the control that proves the guard is an exit code, not the
    mere presence of a block. Same item, same ruling — only `receipt` differs, and
    it exits 0. The close proceeds exactly as it did before this guard existed: the
    entry lands in `owed_settled`, the sweep that empties the list closes the item,
    and `completed`/`closed_by` are written as every other close writes them."""
    p, entries = _hand_run_item(isolated, 301, status="draft", receipt="exit 0")
    out = O.apply_verdict(p, entries,
                          [{"n": 1, "outcome": "ruling",
                            "evidence": "applied 7 edges; active name-embedding = 19"}],
                          item_id=301)
    fm = fm_of(p)
    assert out["moved"] == "closed" and fm["status"] == "done"
    assert fm["closed_by"] == "owed-check" and fm.get("completed")
    assert O.entries_of(fm) == [], "the entry is gone from `owed`"
    assert fm["owed_settled"][0]["outcome"] == "ruling"
    assert "applied 7 edges" in fm["owed_settled"][0]["evidence"]

    p2, _ = _hand_run_item(isolated, 303, status="up_next", receipt="exit 0")
    assert O.close_item(p2, "the by-hand run has happened") is True
    assert fm_of(p2)["status"] == "done"


def test_hand_runs_lists_the_open_items_whose_receipt_fails_and_no_others(isolated):
    """#3017 clause 3: the attended readout, decided by an exit code alone."""
    _hand_run_item(isolated, 310, status="up_next")                    # failing, open
    _hand_run_item(isolated, 311, status="draft", receipt="exit 0")    # executed
    _hand_run_item(isolated, 312, status="done")                       # closed WITH the debt
    _hand_run_item(isolated, 313, status="up_next", receipt="")        # block, no receipt
    _hand_run_item(isolated, 314, status="up_next", cmd="")            # nothing to run
    write_item(isolated, 315, status="up_next")                        # no block at all
    listed = O.hand_runs()
    assert [h["item_id"] for h in listed] == [310, 313], \
        "a failing receipt and a missing one both hold; an executed, closed or " \
        "commandless block does not appear"
    h = listed[0]
    assert tuple(h) == O.HAND_RUN_FIELDS, \
        "the row's keys are the ones the client type is pinned against"
    assert h["name"] == "A thing" and h["since"] == "2026-10-19T09:02:39Z"
    assert h["set"] == PS3, "the pair set, in the spelling Step 2.5 prints"
    assert h["cmd"] == APPLY_CMD and h["cmd"].endswith("--apply")
    assert "receipt" not in h, "the readout names the run, not the probe that checks it"


def test_board_health_lists_a_pending_hand_run_beside_the_outside_list(isolated):
    """#3017 clause 3: on the same Mission Control readout that carries `outside`."""
    _hand_run_item(isolated, 95, status="up_next")
    p, entries = _owing(isolated, 96, ["run update-ca-trust as root"])
    O.apply_verdict(p, entries, [{"n": 1, "outcome": "outside", "evidence": "e",
                                  "outside": "run update-ca-trust as root"}], item_id=96)
    owed = B.board_health(S.LEDGER_PATH)["owed"]
    assert [h["item_id"] for h in owed["hand_runs"]] == [95]
    assert [o["item_id"] for o in owed["outside"]] == [96], "the two lists sit side by side"
    assert owed["items"] == 1 and owed["entries"] == 1, \
        "the hand-run item still owes its apply; the outside answer emptied the other"


# ---------------------------------------------------------------------------
# #3320: the LIVE carrier's OWN receipt, executed. Every hand-run test above
# writes its own probe (`receipt: "exit 1"`), so none ever asked whether the
# command #3320 actually carries can hold and release. Triage 2026-10-25 found
# the item asserting it carried a `hand_run:` block while the file carried
# none — `hand_run_of(fm) -> {}` — so dispatch offered it and `close_item`
# would have closed it around a corpus `stats()` says is absent. These run the
# real receipt: the registry probe, in a subprocess, inheriting an `LLOYD_DATA`
# pointed at a private tmp root (`app.paths` rule 1), so production
# `~/lloyd-data/research.db` is never opened. Held while `archived` < 314,
# silent at 314 — the same two readings #3289 pinned as a class rule, measured
# now through the block this round attached to the live item. And the held
# close, whose 300-char note bound measured on 2026-10-25 cut the live
# receipt's own text off mid-command (`…sys.exit(0 if n >= 314 els`), losing
# the reading `exited 1: ARCHIVED=0 …` that the line exists to record.


CARRIER_CMD = ("cd ~/lloyd && .venvs/lloyd/bin/python -m app.research_store import "
               "~/obsidian/lloyd/research-queue-archive.md")
CARRIER_RECEIPT = (
    "cd ~/lloyd && /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -c "
    "\"import sys; from app.research_store import store; s = store().stats(); "
    "n = s['by_status'].get('archived', 0); print('ARCHIVED=%d TOTAL=%d' % (n, "
    "s['total'])); sys.exit(0 if n >= 314 else 1)\"")
CARRIER_SET = ["314 archived rows from lloyd/research-queue-archive.md (the #3046 dedup corpus)"]


def _carrier_item(tmp_path, *, status="draft"):
    """An item carrying #3320's block verbatim: the restore `cmd`, the registry
    probe as `receipt`, `since` = the vault commit that attached it."""
    p = write_item(tmp_path, 3320, status=status, name="Hand-run carrier item")
    B.update_frontmatter(p, {"hand_run": {"set": CARRIER_SET, "cmd": CARRIER_CMD,
                                          "receipt": CARRIER_RECEIPT,
                                          "since": "2026-10-25T16:48:09Z"}})
    return p


def _seed_archived(root: Path, n: int, start: int = 0) -> None:
    """`n` topics settled `archived` in the registry under `root` — the corpus
    role, filled through the store's real API (`propose` + `finish`) so the
    receipt's `stats()` sees exactly what production's will after the import."""
    from app.research_store import ResearchStore
    st = ResearchStore(root / "research.db")
    for i in range(start, start + n):
        tid = st.propose(f"carrier seed {i} for the #3320 receipt",
                         proposed_by="test")["id"]
        st.finish(tid, "archived", note="seed for the #3320 receipt test")


def test_the_carrier_block_is_a_block_naming_the_import_and_the_probe(tmp_path):
    """#3320 clause 1 (the shape): `hand_run_of` returns a block, not `{}` — the
    failure mode #3320 was triaged from was the item ASSERTING this block while
    its front matter held none."""
    block = O.hand_run_of(fm_of(_carrier_item(tmp_path)))
    assert block, "hand_run_of returned {} — the exact #3320 triage failure"
    assert "python -m app.research_store import ~/obsidian/lloyd/research-queue-archive.md" \
        in block["cmd"], "the cmd names the #3085 restore, not a paraphrase of it"
    assert block["cmd"].startswith("cd ~/lloyd &&"), \
        ("from any other tree app.paths opens <worktree>/.lloyd-data/research.db, "
         "a gitignored store whose `existing: 0` is that empty file's answer, not "
         "production's (item clause 2)")
    assert "n >= 314" in block["receipt"] and "archived" in block["receipt"], \
        "the receipt probes the archived count and names its release bound"


def test_the_carrier_receipt_holds_the_item_while_the_corpus_is_absent(tmp_path, monkeypatch):
    """#3320 clause 2, holding half: the real receipt run against a registry
    with no corpus exits 1, so `hand_run_pending` returns the block WITH the
    receipt's own reading — `exited 1: ARCHIVED=0 …` — which is production's
    exact state (measured `ARCHIVED=0 TOTAL=134`, exit 1, 2026-10-25) until the
    operator runs the import."""
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))
    block, why = O.hand_run_pending(fm_of(_carrier_item(tmp_path)))
    assert block and block["cmd"] == CARRIER_CMD, \
        "the hold must be lifted by running the command, not by settling the entry"
    assert why.startswith("exited 1:") and "ARCHIVED=0" in why, \
        ("the why is the probe's own output — `exit 1` is only allowed as a "
         "receipt the test wrote, never as a result the carrier produces")


def test_the_carrier_receipt_releases_only_at_314_archived_rows(tmp_path, monkeypatch):
    """#3320 clause 2, release half: 313 archived rows still hold the item and
    say so; the 314th — the count the import reports (`inserted: 314`) — is the
    only thing that clears it."""
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))
    p = _carrier_item(tmp_path)
    _seed_archived(tmp_path, 313)
    block, why = O.hand_run_pending(fm_of(p))
    assert block and "ARCHIVED=313" in why, \
        "the bound is >= 314, not > 0: three quarters of a corpus is still a missing corpus"
    _seed_archived(tmp_path, 1, start=313)
    assert O.hand_run_pending(fm_of(p)) == ({}, ""), \
        "at 314 archived rows the block is spent: dispatch may offer the item and closes may land"


def test_a_held_close_on_the_carrier_keeps_draft_and_names_the_receipt_reading(tmp_path, monkeypatch):
    """#3320 clause 3: while the receipt fails, `close_item` leaves
    `status: draft` and writes an activity line naming BOTH the receipt command
    and its reading. Measured pre-fix, the 300-char note bound cut the live
    #3320 receipt (260 chars) off mid-command and dropped the reading entirely
    — the held close then recorded only that a command existed, which is one
    character short of distinguishing it from a close nobody attempted. (The
    `status` half was pinned for `exit 1` receipts by #3017's
    `test_close_item_refuses_a_pending_hand_run_and_leaves_the_status_alone`;
    this is that shape pinned for the carrier's real block at `status: draft`
    with the reading half added.)"""
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))
    p = _carrier_item(tmp_path, status="draft")
    assert O.close_item(p, "round reported every acceptance clause met") is False
    raw = p.read_text(encoding="utf-8")
    assert "\nstatus: draft\n" in raw, \
        "the held close moves nothing: the status line survives byte-for-byte"
    assert fm_of(p)["status"] == "draft"
    acts = "\n".join(str(a) for a in (fm_of(p).get("activity_log") or []))
    assert "pending hand-run unexecuted" in acts
    assert "exited 1: ARCHIVED=0" in acts, "the line carries the receipt's reading"
    assert CARRIER_RECEIPT in acts, \
        "and the receipt command whole — the old bound cut it at `n >= 314 els`"
    assert O.hand_run_of(fm_of(p))["since"] == "2026-10-25T16:48:09Z", \
        "the held close leaves the block itself untouched"


@pytest.mark.live_vault
def test_the_live_3320_item_carries_the_carrier_block():
    """#3320 clause 1, on the artifact the clause names: the live item's own
    front matter, not a test's copy of it. Marked `live_vault` (excluded from
    the gate): the item file is vault-owned and later writers append activity
    to it — but the `hand_run` block must survive those writes byte-equal to
    the shape these tests pin, and if an editor changes the restore command or
    drops the block, this line goes red naming the file. That is the tripwire
    for the failure #3320 was triaged from: an item whose prose promised the
    block while its front matter had none."""
    import os
    vault = Path(os.environ.get("LLOYD_OBSIDIAN_VAULT") or (Path.home() / "obsidian"))
    p = vault / "backlog" / "3320-run-the-3085-dedup-corpus-restore-as-a-hand-run-ca.md"
    block = O.hand_run_of(B._split_frontmatter(p.read_text(encoding="utf-8"))[0])
    assert block, f"{p} carries no hand_run block — the #3320 premise has returned"
    assert block["cmd"] == CARRIER_CMD and block["receipt"] == CARRIER_RECEIPT, \
        f"{p}'s block diverged from the shape tests/test_owed_check.py pins"


# ── #3897 clause 3: a string refused as a clause cannot be booked as owed ───────
#
# `clean_clauses` heals a contract on READ, but an owed entry is written by a
# different door: `add_owed` (the landing writers at `backlog.py:6785`, `:7217`
# and `:8307`) filtered only on `_text(w)` truthiness, so #3504's `human_clauses`
# fragments were booked exactly as its activity log records them — `still owed
# 1.; 1]}],` — and each cost an owed-check turn to re-discharge, out of the
# per-tick capacity #2055 bounded. Both doors now read `B.names_a_demand`, the
# one shape measurement `clean_clauses` applies.

def test_add_owed_refuses_a_what_that_names_no_demand(isolated):
    """The writer: a payload-key fragment is never booked, and a real clause is.

    The positive control is what makes the refusal measurable — without it a
    guard that wrote nothing to any item would pass this node.
    """
    p = write_item(isolated, 66150)
    assert not O.add_owed(p, ["1.", "1]}],", "spawned", "findings_filed_to"])
    assert not fm_of(p).get(O.OWED_KEY), fm_of(p).get(O.OWED_KEY)

    assert O.add_owed(p, ["rule on whether the mined rows still carry the token"])
    assert [e["what"] for e in O.entries_of(fm_of(p), derived=False)] == \
        ["rule on whether the mined rows still carry the token"]


def test_a_closed_item_whose_human_clauses_are_payload_keys_derives_nothing(isolated):
    """The reader: `derived_entries` (#2055) asks the same predicate itself.

    It gets its clauses through `human_clauses_of`, which already filters them, so
    a node that only asserted the empty answer would pass on the upstream fix
    alone. The second half of this node hands the same function a closed item with
    one real clause among the fragments and expects exactly that one entry back —
    which is also #3504's real shape: two fragments that must not reach the job,
    and the demand behind them that must.
    """
    fm = {"status": "done", "completed": "2026-11-04T08:45:09.581111",
          "human_clauses": ["1.", "1]}],"]}
    assert O.derived_entries(fm) == [], O.derived_entries(fm)

    fm["human_clauses"] = ["1.", "1]}],", "rule on whether the mining pass still carries the"]
    assert [e["what"] for e in O.derived_entries(fm)] == \
        ["rule on whether the mining pass still carries the"]
