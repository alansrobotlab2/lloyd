"""#2721: a vault grading whose guided restatement failed is graded from its own turn text.

The five transcripts in `tests/fixtures/vault_review_turn_text_2026-10-14-item2720.json`
are byte-exact `messages[-1]` text copied from the five grading generations item #2720's
`automod_vault_land` issued on 2026-10-14 — session ids `20261014_142122_review_d9a9`,
`_142156_review_40c8`, `_142229_review_0f73`, `_142432_review_1abe`,
`_142621_review_d5a3`, from `~/lloyd-data/sessions/`. The ledger rows those runs left
(`~/.local/state/lloyd-automod/promotions.jsonl` lines 21653-21655) read `clauses: []`,
one `blocking: true` with `kind: diverged`, and the verified 30-line skill edit was
reverted. Every one of these five texts still ends with a fenced JSON object carrying
`premise: "sound"` and `met` verdicts. The judgement existed five times; only the
transcription into the schema failed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.automod import backlog as B, review as RV
from scripts.automod import state as S
from tests.test_automod_review import (FIVE, _asked_from, _confirm, _met_answer,  # noqa: E402
                                       write_item)

#: One generation per line, which is why it is `.jsonl` and not `.json`: `*.json` is
#: git-ignored across this tree (".gitignore: All JSON: … may contain personal data"), so a
#: `.json` fixture here would silently never be committed and the tests would read as
#: missing-file failures on any other checkout. `.jsonl` is what the sibling witness fixture
#: `promotions_vault_review_rows_2026-10-01-item2038.jsonl` uses for the same reason.
FIXTURE = (Path(__file__).resolve().parent / "fixtures"
           / "vault_review_turn_text_2026-10-14-item2720.jsonl")
D5A3 = "20261014_142621_review_d5a3"
D9A9 = "20261014_142122_review_d9a9"
FORTY = "20261014_142156_review_40c8"
ZERO73 = "20261014_142229_review_0f73"
DIV_ERR = ("finalizer failed: generation diverged at 8192 tokens — every field this schema "
           "admits is capped, so the object running past them is malformed output, not a "
           "budget")
EMPTY_ERR = "finalizer failed: output is not JSON ('')"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """The same isolation `tests/test_automod_review.py` uses: a temp backlog and ledger.

    Re-declared rather than imported because it is `autouse` there, and an autouse fixture
    that reaches a module through an import is a different thing from the one pytest builds.
    """
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def _five_clause_item(isolated, item_id, tmp_path, clauses=None):
    """A #2335-shaped contract: five clauses, so the shrink plan is [1,2,3] then [4,5].

    The body of `tests/test_automod_review.py::_five_clause_item`, re-stated because that
    one is module-private and this file grades the same function from the other side.
    """
    texts = list(clauses or FIVE)
    write_item(isolated, item_id, clauses=texts)
    _confirm(item_id, acceptance="; ".join(texts), clauses=texts, surface="vault")
    (isolated / "skills" / "x").mkdir(parents=True, exist_ok=True)
    (isolated / "skills" / "x" / "SKILL.md").write_text("x\n", encoding="utf-8")
    return texts


def _fenced(payload) -> str:
    """A grader turn whose answer is the fenced JSON block a real one ends with."""
    return "```json\n" + json.dumps(payload) + "\n```"


def _turn_text(rows, *, premise="sound"):
    return _fenced({"premise": premise, "clauses": rows})


def _met_rows(indices):
    return [{"clause": i, "verdict": "met", "evidence_path": "skills/x/SKILL.md",
             "how_verified": "read"} for i in indices]


@pytest.fixture(scope="module")
def transcripts() -> dict:
    """The five generations keyed by session id, read from the committed JSONL."""
    return {json.loads(line)["session"]: json.loads(line)
            for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()}


def test_the_five_recorded_transcripts_reach_a_verdict_object_through_the_parser(transcripts):
    """The premise, measured on the real bytes rather than asserted.

    All five generations end with a `json.loads`-clean object whose clause rows carry a
    verdict word, including the one the ledger recorded as an empty completion. Had any of
    them truly held no object, the rescue this file tests would have nothing to read and the
    item's premise would be false.
    """
    assert len(transcripts) == 5, len(transcripts)
    for name, rec in transcripts.items():
        recovered = RV.recover_grading_from_turn_text(rec["turn_text"], [1, 2, 3, 4, 5])
        assert recovered is not None, f"{name}: its own turn text holds no verdict object"
        assert recovered["premise"] == "sound", name
        assert recovered["clauses"], name
        assert all(c["verdict"] in RV.RECOVERED_VERDICTS for c in recovered["clauses"]), name


def test_the_empty_completion_generation_yields_a_graded_clause_rather_than_no_verdict(
        transcripts):
    """Clause 1's named generation, the one whose `content` was the empty string.

    The ledger row for #2720's attempt 2 reads `kind: skipped` with `clauses: []` for
    `output is not JSON ('')`. That generation's turn text carries clause 1 spelled `n` and
    `verdict` — spellings the guided schema never asks for, which is the whole reason the
    restatement failed on it. Recovery must read it: one graded clause, the evidence the
    grader wrote, and no `evidence_line` it did not state.
    """
    recovered = RV.recover_grading_from_turn_text(transcripts[D5A3]["turn_text"], [1])
    assert recovered is not None
    assert len(recovered["clauses"]) == 1, recovered["clauses"]
    row = recovered["clauses"][0]
    assert row["clause"] == 1 and row["verdict"] == "met", row
    assert row["evidence_path"] == ("/home/alansrobotlab/obsidian/skills/"
                                    "runtime-store-schema-probe/SKILL.md:57-86"), row
    assert "evidence_line" not in row, "the transcript states a range, not one line"
    assert row["how_verified"] == "read", row


def test_recovery_reads_every_index_and_verdict_spelling_these_generations_wrote(
        transcripts):
    """Clause 2's first half, on the measured spellings rather than a guess at them.

    `id` came from `_142229_review_0f73`, `n` from `_142432_review_1abe`,
    `_142621_review_d5a3` and `_142122_review_d9a9`, and `status` where the schema asks for
    `verdict` from `_142156_review_40c8`. `clause` is what the schema wants and what a
    well-behaved grader writes, so all three spellings are read side by side here.
    """
    spellings = {
        "clause": {"clause": 1, "verdict": "met", "evidence_path": "skills/x/SKILL.md"},
        "id": {"id": 1, "verdict": "partial", "evidence_path": "skills/x/SKILL.md"},
        "n": {"n": 1, "status": "unmet", "evidence_path": "skills/x/SKILL.md"},
    }
    for spelling, row in spellings.items():
        got = RV.recover_grading_from_turn_text(_turn_text([row]), [1])
        assert got is not None, spelling
        assert got["clauses"][0]["clause"] == 1, f"{spelling}: index normalised to `clause`"
        want = row.get("verdict", row.get("status"))
        assert got["clauses"][0]["verdict"] == want, (spelling, got["clauses"][0])
    # And the real `status`-spelling transcript, where the key sits beside an `evidence`
    # leaf the review schema forbids: the forbidden leaf is dropped, the verdict is kept.
    both = RV.recover_grading_from_turn_text(transcripts[FORTY]["turn_text"], [4, 5])
    assert [c["clause"] for c in both["clauses"]] == [4, 5], both
    assert all(c["verdict"] == "met" for c in both["clauses"]), both
    assert all("evidence" not in c for c in both["clauses"]), "schema-forbidden leaves stay out"


def test_a_failed_restatement_is_graded_from_the_transcript_rather_than_recorded_empty(
        transcripts, isolated, monkeypatch, tmp_path):
    """Clause 1 end to end: #2720's five generations, replayed, produce five verdicts.

    The stub hands `grade_vault` the two slices its shrink plan issued, each carrying the
    turn text the run really produced for that slice and the diverged symptom it really
    reported — and the result is the opposite of the ledger row: five `met` clauses and a
    kind that is not `diverged`. The `[1, 2, 3]` transcript spells its index `n`, the
    `[4, 5]` one spells its verdict `status`, so the normalisation is crossed at the
    boundary that writes the row rather than only in the parser beside it.
    """
    texts = _five_clause_item(isolated, 721, tmp_path)
    detect = _asked_from(texts)

    def replay(**kw):
        slice_ = detect(kw["prompt"])
        by_slice = {(1, 2, 3): "20261014_142122_review_d9a9", (4, 5): FORTY}
        name = by_slice.get(tuple(slice_))
        assert name is not None, f"the stub was asked for a slice nobody graded: {slice_}"
        return {"ok": False, "error": DIV_ERR, "text": transcripts[name]["turn_text"]}
    monkeypatch.setattr(RV, "run_grader", replay)
    kind, why, clauses = RV.grade_vault(item_id=721, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind != RV.GRADER_DIVERGED, f"the blocking kind #2720 died on: {kind} — {why}"
    assert [c["clause"] for c in clauses] == [1, 2, 3, 4, 5], clauses
    assert all(c["verdict"] == "met" for c in clauses), clauses
    assert kind == "pass", why


def test_the_shrink_path_recovering_one_clause_never_invents_the_other_four(
        transcripts, isolated, monkeypatch, tmp_path):
    """The slice after a shrink asks for clause 1; only clause 1 may be filled from it.

    This is #2720's attempt-2 shape: the `[1, 2, 3]` call fails, the ladder re-asks clause 1
    alone, and `d5a3`'s text answers only that. Clauses 2-5 must come back `ungraded`, not
    borrowed from a generation that never saw them — a rescued `met` for a clause whose text
    was never in the prompt would be a verdict minted from nothing.
    """
    texts = _five_clause_item(isolated, 722, tmp_path)
    detect = _asked_from(texts)

    def shrunk(**kw):
        slice_ = detect(kw["prompt"])
        if slice_ == [1]:
            return {"ok": False, "error": EMPTY_ERR,
                    "text": transcripts[D5A3]["turn_text"]}
        return {"ok": False, "error": DIV_ERR, "text": "prose with no object in it"}
    monkeypatch.setattr(RV, "run_grader", shrunk)
    kind, why, clauses = RV.grade_vault(item_id=722, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    graded = {c["clause"]: c["verdict"] for c in clauses}
    assert graded == {1: "met", 2: "ungraded", 3: "ungraded", 4: "ungraded",
                      5: "ungraded"}, graded
    # `diverged`, because the ladder's re-ask of clause 2 diverged: the shipped kind rule,
    # unchanged by the rescue, which is what makes the row still refuse the land.
    assert kind == RV.GRADER_DIVERGED, f"four clauses never reached: {kind} — {why}"
    assert RV.RECOVERY_TAG in (why or ""), why
    assert "clause(s) 1" in why, f"the provenance names the one rescued index: {why}"


def test_a_recovered_met_without_evidence_path_reaches_the_land_as_partial(
        isolated, monkeypatch, tmp_path):
    """Clause 3 end to end: the downgrade is `parse_review`'s, applied to the rescued row.

    The vault prompt states in words that a `met` with no `evidence_path` under the vault is
    downgraded. A rescue that carried such a `met` through unchanged would hand the round a
    verdict the graded path could never produce, so recovery must go through the same rails
    — which it does by feeding `parse_review` the same object shape the decoder would have.
    """
    texts = _five_clause_item(isolated, 723, tmp_path)
    detect = _asked_from(texts)

    def no_path(**kw):
        slice_ = detect(kw["prompt"])
        rows = [{"clause": i, "verdict": "met", "how_verified": "read"} for i in slice_]
        return {"ok": False, "error": DIV_ERR, "text": _turn_text(rows)}
    monkeypatch.setattr(RV, "run_grader", no_path)
    kind, why, clauses = RV.grade_vault(item_id=723, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    graded = {c["clause"]: c["verdict"] for c in clauses}
    assert set(graded.values()) == {"partial"}, (
        f"a rescued `met` with no evidence path is a `partial`, never a `met`: {graded}")
    assert kind != "pass", f"no clause may pass on an ungrounded `met`: {kind}"


def test_a_partial_restatement_keeps_its_own_verdict_where_the_turn_text_disagrees(
        isolated, monkeypatch, tmp_path):
    """The narrow seam: a guided answer that DID transcribe a clause outranks the rescue.

    Clause 4 is asked at 2 clauses per call, the restatement comes back grading it `unmet`,
    and the turn text claims `met` for both 4 and 5. The decoder's `unmet` stands — recovery
    fills clause 5 and nothing else — because a transcribed row went through
    `parse_review`'s path and citation rails and a rescued row did not, so a rescued row can
    never overrule a verdict it could disagree with. The provenance tag then names clause 5
    alone.
    """
    texts = _five_clause_item(isolated, 724, tmp_path)
    detect = _asked_from(texts)

    def partial(**kw):
        slice_ = detect(kw["prompt"])
        if slice_ == [1, 2, 3]:
            return _met_answer(slice_)
        ans = _met_answer([slice_[0]])
        ans["structured"]["clauses"][0]["verdict"] = "unmet"
        ans["structured"]["clauses"][0]["note"] = "the cited line does not carry the claim"
        return {**ans, "text": _turn_text(_met_rows(slice_))}
    monkeypatch.setattr(RV, "run_grader", partial)
    kind, why, clauses = RV.grade_vault(item_id=724, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    graded = {c["clause"]: c["verdict"] for c in clauses}
    assert graded[4] == "unmet", f"the transcribed verdict outranks the rescue: {graded}"
    assert graded[5] == "met", f"only the gap the decoder left was rescued: {graded}"
    assert graded[1] == graded[2] == graded[3] == "met", graded
    assert kind == "retry", f"an unmet clause is not a pass: {kind}"
    assert RV.RECOVERY_TAG in (why or ""), why
    assert "clause(s) 5" in why, f"the provenance names exactly the rescued index: {why}"


def test_a_turn_text_with_no_object_leaves_the_kind_and_row_exactly_as_they_were(
        isolated, monkeypatch, tmp_path):
    """Clause 4, both shipped symptoms: an ungraded review still refuses the land.

    The expectations below are not read off the code — they are what
    `git show HEAD:scripts/automod/review.py` (the pre-change module) returns for these two
    stubs when it is loaded beside the changed one and both are run on the same input. A word
    changed here means the rescue leaked into a grading that had nothing to rescue. Each case
    grades its OWN item id, because the shrink ladder remembers per item: two reviews of one
    id would make the second one inherit the first one's budget, which is a different
    question from the one asked here.
    """
    cases = [
        (725, DIV_ERR, "diverged",
         "[clause(s) 1, 2, 3 asked at 3 clauses per call shrank to 1 clause per call] grader "
         "generation diverged answering clause(s) 1 (1 clause per call): finalizer failed: "
         "generation diverged at 8192 tokens — every field this schema admits is capped, so "
         "the object running past them is malformed output, not a budget [2 grading "
         "generations issued]"),
        (726, EMPTY_ERR, "skipped",
         "grader did not answer: finalizer failed: output is not JSON (\'\')"),
    ]
    for item_id, symptom, want_kind, want_why in cases:
        texts = _five_clause_item(isolated, item_id, tmp_path)
        detect = _asked_from(texts)

        def mute(**kw):
            asked.append(detect(kw["prompt"]))
            return {"ok": False, "error": symptom, "text": "prose, no braces at all"}
        asked: list[list[int]] = []
        monkeypatch.setattr(RV, "run_grader", mute)
        kind, why, clauses = RV.grade_vault(item_id=item_id, paths=["skills/x/SKILL.md"],
                                            diff="+x", vault=isolated)
        assert kind == want_kind, f"{symptom[:34]}…: kind {kind} != {want_kind}"
        assert clauses == [], f"{symptom[:34]}…: {clauses}"
        assert why == want_why, f"{symptom[:34]}…:\n{why!r}\n!=\n{want_why!r}"
        assert RV.RECOVERY_TAG not in why, "nothing was recovered, so nothing may claim it"
        assert asked[0] == [1, 2, 3], f"the first pass is the shipped 3-clause chunk: {asked}"


def test_an_outage_generation_that_still_delivered_its_object_is_rescued(
        isolated, monkeypatch, tmp_path):
    """The boundary the first review found was never reached: `grader_outage` keys on the
    SHAPE of the error, not on whether an answer arrived.

    A `ReadTimeout` raised after the turn wrote its fenced verdict object leaves that object
    in the session file, so refusing to read it on the theory that "no turn finished" throws
    away the same judgement #2721 exists to keep. An unreadable outage is the control in the
    test below, and stays a plain failure there.
    """
    _five_clause_item(isolated, 727, tmp_path)

    def timeout_with_text(**kw):
        return {"ok": False, "error": "finalizer failed: ReadTimeout: ",
                "text": _turn_text(_met_rows([1, 2, 3, 4, 5]))}
    monkeypatch.setattr(RV, "run_grader", timeout_with_text)
    kind, why, clauses = RV.grade_vault(item_id=727, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert [c["verdict"] for c in clauses] == ["met"] * 5, clauses
    assert RV.RECOVERY_TAG in (why or ""), why


def test_an_outage_that_delivered_nothing_readable_keeps_its_kind_text_and_no_rows(
        isolated, monkeypatch, tmp_path):
    """The control for the test above, and #2421's abstention rule: an outage with no
    readable object reports the outage kind, the error VERBATIM, and an empty clause list.

    The verbatim text is the half that matters for #2421 — the streak that ends in an
    abstention counts rows by that kind, so the rescue must add neither a clause row nor a
    word of its own prose to a turn that answered nothing.
    """
    _five_clause_item(isolated, 728, tmp_path)
    err = "finalizer failed: ReadTimeout: "

    def timeout_empty(**kw):
        return {"ok": False, "error": err, "text": ""}
    monkeypatch.setattr(RV, "run_grader", timeout_empty)
    kind, why, clauses = RV.grade_vault(item_id=728, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind == RV.GRADER_OUTAGE, why
    assert clauses == [], f"an unreadable outage credits no clause: {clauses}"
    assert err in why and RV.RECOVERY_TAG not in why, why


def test_a_failure_row_carries_the_verdicts_it_rescued_beside_its_failure_kind(
        isolated, monkeypatch, tmp_path):
    """The graded return is not the only row a rescue can write.

    #2720's attempt-2 shape: the chunk diverges, one clause is recovered from its prose, the
    rest stay ungraded, so `answer` re-asks and the review still fails. What must NOT happen
    is the row going back to `clauses: []` — #2720's ledger rows were unfixable precisely
    because they recorded zero verdicts for a generation that had answered one. The kind
    stays a failure and still refuses the land.
    """
    texts = _five_clause_item(isolated, 729, tmp_path)
    detect = _asked_from(texts)

    def one_clause_back(**kw):
        slice_ = detect(kw["prompt"])
        if slice_ == [1]:
            return {"ok": False, "error": EMPTY_ERR,
                    "text": _turn_text([{"clause": 1, "verdict": "met",
                                         "evidence_path": "skills/x/SKILL.md",
                                         "how_verified": "read"}])}
        return {"ok": False, "error": DIV_ERR, "text": "no object here"}
    monkeypatch.setattr(RV, "run_grader", one_clause_back)
    kind, why, clauses = RV.grade_vault(item_id=729, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind in RV.GRADER_FAILURE_KINDS, f"an ungraded review must still fail: {kind}"
    assert [c["clause"] for c in clauses] == [1, 2, 3, 4, 5], clauses
    assert clauses[0]["verdict"] == "met" and clauses[0]["recovered"] is True, clauses[0]
    assert [c["verdict"] for c in clauses[1:]] == ["ungraded"] * 4, clauses
    assert RV.RECOVERY_TAG in why, why


# ── the #2721 re-offer: the chunk bound had no test on the whole-chunk path ───

def test_a_whole_chunk_failure_credits_only_the_indices_that_chunk_asked_for():
    """Recovery may credit an index this generation was ASKED about, and nothing else.

    The rescue's own docstring (`review.py:1956-1958`) promises that an entry outside
    `chunk` is dropped, and the drop is implemented at `:1987` — but every recorded
    transcript in the fixture file above is passed with the chunk that produced it
    (d9a9 answers `n` 1,2,3; 40c8 answers 4,5; 0f73 answers 5; d5a3 answers 1), so
    the promise was never asserted on the shape #2720 is: `only=None`, the whole
    chunk failed. Measured by mutation — delete the `continue` at `:1987` and the
    only node in this tree that notices is the `only=missing` gap-filling case,
    because that is the only path a test ever handed an out-of-chunk index.

    Three entries in one object, three fates: 5 was asked and answered, 1 belongs to
    another chunk of the same contract, 7 is no clause of it at all.
    """
    text = _turn_text([
        {"clause": 5, "verdict": "met", "evidence_path": "skills/x/SKILL.md",
         "how_verified": "read", "note": "asked, and answered"},
        {"clause": 1, "status": "met", "evidence_path": "skills/x/SKILL.md",
         "note": "answered out of turn: another chunk owns index 1"},
        {"n": 7, "verdict": "met", "evidence_path": "skills/x/SKILL.md"},
    ])

    got = RV.recover_grading_from_turn_text(text, [4, 5])

    assert got is not None, "the object is readable; only its indices are out of bounds"
    assert [c["clause"] for c in got["clauses"]] == [5], (
        "recovery credited an index this chunk never asked about, so the row would "
        "carry a verdict for a clause nobody graded here")
    assert got["premise"] == "sound", "the premise reading survives the dropped rows"


@pytest.mark.parametrize("transcript,chunk", [(D9A9, [4, 5]), (FORTY, [1, 2, 3]),
                                              (ZERO73, [4])])
def test_a_real_transcript_never_answers_for_a_chunk_it_was_not_asked_about(
        transcripts, transcript, chunk):
    """Three real transcripts, each handed a chunk that is not its own: no verdicts.

    d9a9 answers 1,2,3; 40c8 answers 4,5; 0f73 answers 5. None of them may credit
    anything for the chunk named beside it — the parametrisation is what makes the
    bound a rule rather than one lucky pairing.
    """
    got = RV.recover_grading_from_turn_text(transcripts[transcript]["turn_text"], chunk)

    # Nothing creditable is NO rescue at all: `:2004` returns None on an empty entry
    # list, which leaves the caller's rows `ungraded` and the row's kind exactly what
    # a no-object failure would have left it — the land stays refused.
    assert got is None, (
        f"{transcript} answered for a chunk it was never asked about: "
        f"{[c['clause'] for c in (got or {}).get('clauses', [])]}")


def test_a_refused_review_stamps_rescued_rows_only_for_the_chunk_that_rescued_them(
        isolated, monkeypatch, tmp_path):
    """The land-level half: a decoder-graded clause must not wear the rescue marker.

    Chunk [1,2,3] restates cleanly through the guided decoder; chunk [4,5] fails
    with a turn text answering clause 4 AND clause 1. With the bound in place clause
    1's verdict is the decoder's and its row carries no `recovered` key, clause 4 is
    rescued and marked, and clause 5 — asked, never answered — stays `ungraded`. With
    the bound dead the rescued object re-states clause 1 and its row reads as though
    it survived only because its text was mined, which is #2721 clause 5's
    distinguishability promise failing in the opposite direction: the marker would
    point at the wrong clause.
    """
    texts = _five_clause_item(isolated, 723, tmp_path)
    detect = _asked_from(texts)

    def replay(**kw):
        slice_ = detect(kw["prompt"])
        if slice_ == [1, 2, 3]:
            return _met_answer([1, 2, 3])
        return {"ok": False, "error": DIV_ERR, "text": _turn_text([
            {"clause": 4, "verdict": "met", "evidence_path": "skills/x/SKILL.md",
             "how_verified": "read"},
            {"clause": 1, "verdict": "met", "evidence_path": "skills/x/SKILL.md",
             "how_verified": "read"}])}
    monkeypatch.setattr(RV, "run_grader", replay)
    kind, why, clauses = RV.grade_vault(item_id=723, paths=["skills/x/SKILL.md"],
                                        diff="+x", vault=isolated)

    by_clause = {c["clause"]: c for c in clauses}
    assert kind != "pass", "one chunk never answered; the land has to stay refused"
    assert "recovered" not in by_clause[1], (
        "a clause the guided decoder graded is stamped as rescued from turn text")
    assert by_clause[4]["verdict"] == "met" and by_clause[4].get("recovered") is True
    assert by_clause[5]["verdict"] == "ungraded", (
        "clause 5 was asked in the failed chunk and answered by nobody")
    assert RV.RECOVERY_TAG in (why or ""), why
