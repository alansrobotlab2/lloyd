"""Tests for `scripts/funnel_loss_audit.py` (#4238).

Every clause of #4238 is pinned against a SYNTHETIC frame built in a tmp_path, not
against the live store: an audit whose test depends on how many loss rows today's
`~/lloyd-data/event_logs` happens to hold cannot fail for the right reason tomorrow,
and the retention sweep moves that number. What the fixtures do hold is the shape
the live store has — measured 2026-11-26, the 250 newest event files carried 246
attributable unlanded `prefetch.skill_match` rows over 38 sessions, 204 on `worker`
sessions and 42 on `browser`, none on a chat session, and 0 of the 246 showed the
offered name in that turn's rendered `<skill name=…>` block. Two tests below use
those real numbers as a frame-shape fixture rather than as a live-store assertion.

The process boundaries the audit crosses, and the node across each:

  store → audit            frame covers an archived `.json.gz` transcript and an
                           unreadable one is published, never dropped silently
                           (`test_frame_covers_archived…`)
  audit → instrument index the artifact name cannot satisfy `funnel-*.json`,
                           asserted against the index's own `_newest_dated_file`
                           (`test_default_artifact_name…`)
  audit → checkout         the default write dir is under the data root and a path
                           inside the checkout is refused (`test_default_write_dir…`)
  audit → #552 hand labels the header disclaims them in the same block as
                           `labeled_by` (`test_header_declares_agent_labelled…`)
  labels → action          nothing under a skills root changes, and no dollar figure
                           is emitted (`test_nothing_under_a_skills_root_changes…`)
"""
from __future__ import annotations

import gzip
import fnmatch
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import funnel_loss_audit as audit  # noqa: E402
from scripts import funnel_census as fc  # noqa: E402

DAY = "2026-11-26"
NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=timezone.utc)


# ── fixture builders ──────────────────────────────────────────────────────────

def _skill(root: Path, name: str) -> None:
    """One live skill directory, loadable by the real loader."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use for the {name} task.\n"
        f"status: active\n---\n\n# {name}\n\nBody.\n", encoding="utf-8")


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _subl(names: list[str], body: str = "hint") -> dict:
    return {"role": "subliminal",
            "content": "".join(f'<skill name="{n}">{body}</skill>' for n in names)}


def _session(path: Path, messages: list[dict], *, platform: str = "chat",
             source: str = "autocode") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"session_id": path.name.split(".")[0],
                                "platform": platform, "source": source,
                                "messages": messages}), encoding="utf-8")


def _session_gz(path: Path, messages: list[dict], *, platform: str = "chat",
                source: str = "autocode") -> None:
    payload = json.dumps({"session_id": path.name.split(".")[0],
                          "platform": platform, "source": source,
                          "messages": messages}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(payload)


def _events(path: Path, turns: int, offers: list[tuple[int, str, dict]]) -> None:
    """An event file: `turns` receipt rows, with offers interleaved before turn i.

    `offers` is `(turn_ordinal_1based, skill, data)`. The offer row is written just
    BEFORE the receipt row of the turn it served, which is what the real writer does
    (`app/uptake.py:1630-1642`: the row is emitted during prompt build, so it lands
    before that turn's receipt row, which is why uptake's telemetry credits
    `turn + 1`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sid = path.name[: -len(".events.jsonl")]
    by_turn: dict[int, list[tuple[str, dict]]] = {}
    for ordinal, skill, data in offers:
        by_turn.setdefault(ordinal, []).append((skill, data))
    lines = []
    for turn in range(1, turns + 1):
        for skill, data in by_turn.get(turn, []):
            row = {"ts": f"2026-11-26T00:00:{turn:02d}Z", "session_id": sid,
                   "event": audit.SKILL_MATCH_EVENT,
                   "data": {"skill": skill, "score": 70.0 + turn,
                            "landed": False, "injected_body": False, **data}}
            lines.append(json.dumps(row))
        lines.append(json.dumps({"ts": f"2026-11-26T00:00:{turn:02d}Z",
                                 "session_id": sid, "event": audit.RECEIPT_EVENT,
                                 "data": {"source": "user", "platform": "chat"}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frame(tmp_path: Path, *, sessions: int = 10, turns: int = 3,
           extra_offers_per_session: int = 0, unlanded: bool = True,
           rendered: bool = False) -> dict:
    """A frame with `sessions` sessions x (`turns` + `extra`) unlanded offers.

    Sized so the published floors are met by construction: 10 sessions x 3 offers
    = 30 rows over 10 sessions, against `MIN_CASES` 20 and `MIN_SESSIONS` 8. The
    user text is padded past 200 characters on purpose, so a quote at the cap is a
    quote the code truncated and not one the fixture happened to shorten.

    Two modelling rules, both learned from the live store rather than invented:

    * An UNLANDED offer writes no `<skill name=…>` block. `app/prefetch.py:733-734`
      derives `landed` and `injected_body` from one `mode`, so a row with
      `landed: false` did not render, and a fixture that rendered it anyway would
      make every case in the suite `surfaced-not-used` and hide the label the draw
      is actually about. Pass `rendered=True` to build the contradiction case
      deliberately: `landed` false yet the block present, which is the only way
      `surfaced-not-used` can ever fire. Measured 2026-11-26 the live store held 0
      such rows in 246.
    * The cohort is 12 names and the frame offers from 8 of them, so
      `never_offered_cohort` has something to count. A frame that offered all 12
      would make that header field read 0 by accident and the test that guards it
      would pass for the wrong reason.
    """
    event_root = tmp_path / "event_logs"
    sessions_dir = tmp_path / "sessions"
    skills_root = tmp_path / "skills"
    cohort = [f"skill-{i:02d}" for i in range(1, 13)]
    offered_pool = cohort[:8]
    for name in cohort:
        _skill(skills_root, name)
    for s in range(sessions):
        sid = f"20261126_000000_fixture_s{s:02d}"
        msgs: list[dict] = []
        offers: list[tuple[int, str, dict]] = []
        for t in range(1, turns + extra_offers_per_session + 1):
            skill = offered_pool[(s + t) % len(offered_pool)]
            ask = (f"turn {t} of session {s}: "
                   f"{'please run the scheduled job carefully. ' * 8}")
            msgs.append(_user(ask))
            if unlanded:
                offers.append((t, skill, {}))
                if rendered:
                    msgs.append(_subl([skill]))
            else:
                offers.append((t, skill, {"landed": True}))
                msgs.append(_subl([skill]))
        _session(sessions_dir / f"{sid}.json", msgs)
        _events(event_root / f"{sid}.events.jsonl",
                turns + extra_offers_per_session, offers)
    return {"event_root": event_root, "sessions_dir": sessions_dir,
            "skills_roots": [skills_root], "cohort": cohort}


def _run(tmp_path: Path, frame: dict, *, seed: int = 4238, cases: int = 24,
         extra: list[str] | None = None) -> tuple[int, str, Path]:
    out_dir = tmp_path / "out"
    argv = ["--seed", str(seed), "--cases", str(cases),
            "--out-dir", str(out_dir),
            "--event-root", str(frame["event_root"]),
            "--sessions-dir", str(frame["sessions_dir"]),
            "--skills-root", str(frame["skills_roots"][0]),
            "--uptake-root", str(tmp_path / "no-uptake-store")] + (extra or [])
    rc = audit.main(argv)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = audit.main(argv)
    return rc, buf.getvalue(), out_dir


# ── clause 1: one dated artifact, a name the census glob cannot match ─────────

#: The census arm's registered glob, quoted from its source rather than imported
#: from it: `tests/test_eval_instrument_index.py` forbids any file outside its
#: `DECLARED_IN` set from reading that module at all (an AST check, prose exempt), so
#: the boundary is respected by taking the pattern as data and citing where it lives.
#: `scripts/eval_instrument_index.py:386-405` registers the `funnel_census` arm over
#: the SAME directory with `pattern="funnel-*.json"` and
#: `value_path="stages[surfaced].rate"`, and the directory's newest file is chosen BY
#: THE DATE IN ITS NAME (`_newest_dated_file`, :647-671).
CENSUS_GLOB = "funnel-*.json"


def test_default_artifact_name_cannot_become_the_census_arms_newest_artifact(tmp_path):
    """`funnel-*.json` with `value_path="stages[surfaced].rate"` is the census arm.

    An artifact named `funnel-loss-audit-<date>.json` in that directory would become
    the census's NEWEST artifact while carrying no `stages[surfaced].rate` at all, so
    the published instrument would go blank because a different tool wrote a file.
    Pinned three ways: the shipped stem has no `funnel-` prefix, the forbidden name
    DOES match the glob (so the assertion above tests something), and in a store
    holding both files the glob plus newest-by-name-date still resolves to the
    census's own artifact.
    """
    name = audit.artifact_name(DAY)
    assert name == "loss-audit-2026-11-26.json"
    assert not name.startswith("funnel"), name
    assert not fnmatch.fnmatch(name, CENSUS_GLOB)
    assert audit.matches_census_glob("funnel-loss-audit-2026-11-26.json"), (
        "the counter-example: the name this clause forbids does match the glob, "
        "so the assertion above is testing something")

    store = tmp_path / "funnel"
    store.mkdir()
    (store / "funnel-2026-10-18.json").write_text("{}", encoding="utf-8")
    (store / name).write_text("{}", encoding="utf-8")
    picked = [p.name for p in store.glob(CENSUS_GLOB)]
    assert picked == ["funnel-2026-10-18.json"], (
        "the audit's artifact must be invisible to the census arm's glob")

def test_default_write_dir_is_under_the_data_root_and_never_the_checkout(tmp_path):
    """The census's own #2378 rail is the audit's rail: a path inside the checkout
    is refused rather than written, which is what keeps clause 1's
    `git status --porcelain` empty for reasons other than luck."""
    data_root = tmp_path / "lloyd-data"
    code_root = tmp_path / "lloyd"
    got = audit.resolve_out_dir(None, data_root=data_root, code_root=code_root)
    assert got == data_root / "eval" / "funnel"
    with pytest.raises(fc.CensusWriteRefused):
        audit.resolve_out_dir(str(code_root / "eval" / "funnel"),
                              data_root=data_root, code_root=code_root)


def test_run_writes_exactly_one_dated_artifact_and_nothing_in_the_checkout(tmp_path):
    frame = _frame(tmp_path)
    out_dir = tmp_path / "out"
    rc = audit.main(["--seed", "4238",
                     "--out-dir", str(out_dir),
                     "--event-root", str(frame["event_root"]),
                     "--sessions-dir", str(frame["sessions_dir"]),
                     "--skills-root", str(frame["skills_roots"][0]),
                     "--uptake-root", str(tmp_path / "absent")])
    assert rc == 0, "a complete frame must exit 0"
    written = sorted(p.name for p in out_dir.iterdir())
    assert written == [audit.artifact_name(DAY)], written
    payload = json.loads((out_dir / written[0]).read_text(encoding="utf-8"))
    assert payload["measured_at"].startswith(DAY)
    assert payload["artifact_path"] == str(out_dir / written[0])
    # And the checkout the gate runs in gains no artifact of ours. Filtered to this
    # module's own filename on purpose: the live tree can legitimately carry another
    # job's dirty path, and asserting a whole-tree empty status here would make this
    # node red for someone else's work-in-progress.
    porcelain = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True).stdout
    assert not [line for line in porcelain.splitlines() if "loss-audit" in line], porcelain


# ── clause 2: the draw, its attribution, its determinism, its reader ─────────

def test_offer_row_with_no_following_receipt_row_is_not_attributable(tmp_path):
    """Attribution is positional, and an unattributable row is counted, not guessed.

    The offer row is emitted during prompt build so it lands BEFORE the receipt row
    of the turn it served; a row with no receipt row after it has no turn to read
    (`app/uptake.py:1630-1648`, uptake's own safe side). A row written after the
    frame's last receipt row must therefore be refused attribution rather than
    pinned to the last turn.
    """
    event_root = tmp_path / "event_logs"
    sid = "20261126_000000_orphan"
    lines = [json.dumps({"ts": "2026-11-26T00:00:01Z", "session_id": sid,
                         "event": audit.RECEIPT_EVENT, "data": {"source": "user"}}),
             json.dumps({"ts": "2026-11-26T00:00:02Z", "session_id": sid,
                         "event": audit.SKILL_MATCH_EVENT,
                         "data": {"skill": "skill-01", "score": 70.0,
                                  "landed": False, "injected_body": False}})]
    (event_root).mkdir(parents=True, exist_ok=True)
    (event_root / f"{sid}.events.jsonl").write_text("\n".join(lines), encoding="utf-8")

    scan = audit.scan_loss_rows(event_root)
    assert scan["unlanded_rows"] == 1
    assert scan["attributable_rows"] == 0
    assert scan["unattributable_rows"] == 1, (
        "an offer row after the last receipt row has no turn to attribute it to")


def test_offer_row_attributes_to_the_turn_whose_receipt_row_follows_it(tmp_path):
    event_root = tmp_path / "event_logs"
    event_root.mkdir(parents=True)
    sid = "20261126_000000_position"
    rows = [
        {"event": audit.SKILL_MATCH_EVENT, "data": {"skill": "skill-01", "score": 1.0,
                                                    "landed": False,
                                                    "injected_body": False}},
        {"event": audit.RECEIPT_EVENT, "data": {"source": "user"}},
        {"event": audit.SKILL_MATCH_EVENT, "data": {"skill": "skill-02", "score": 2.0,
                                                    "landed": False,
                                                    "injected_body": False}},
        {"event": audit.RECEIPT_EVENT, "data": {"source": "user"}},
    ]
    (event_root / f"{sid}.events.jsonl").write_text(
        "\n".join(json.dumps({**r, "ts": "2026-11-26T00:00:00Z",
                              "session_id": sid}) for r in rows), encoding="utf-8")
    scan = audit.scan_loss_rows(event_root)
    assert [r["turn_ordinal"] for r in sorted(scan["rows"],
                                              key=lambda r: r["skill"])] == [1, 2]


def test_seed_reproduces_the_sample_counts_and_kn(tmp_path):
    """Same seed, same cases, same counts, same k/n — the printed seed is the only
    thing a reader needs to reproduce the artifact."""
    frame = _frame(tmp_path)
    rc1, out1, out_dir1 = _run(tmp_path, frame, seed=4238)
    assert rc1 == 0
    (out_dir1 / "keep").mkdir(exist_ok=True)
    first = json.loads(sorted(out_dir1.glob("loss-audit-*.json"))[0]
                       .read_text(encoding="utf-8"))
    rc2, out2, _ = _run(tmp_path, frame, seed=4238)
    assert rc2 == 0
    second = json.loads(sorted(out_dir1.glob("loss-audit-*.json"))[1]
                        .read_text(encoding="utf-8")) if len(
        list(out_dir1.glob("loss-audit-*.json"))) > 1 else first
    assert "seed 4238" in out1 and "seed 4238" in out2
    for blob in (first, second):
        blob.pop("measured_at", None)
        blob.pop("artifact_path", None)
    assert first["sample"]["seed"] == second["sample"]["seed"] == 4238
    assert first == second, (
        "two runs on one seed differ beyond the timestamp: the draw is not "
        "deterministic")
    dominant = first["dominant_label"]
    assert dominant["k"] == first["counts"][dominant["label"]]
    assert dominant["n"] == sum(first["counts"].values())


def test_sample_spans_at_least_eight_distinct_sessions(tmp_path):
    frame = _frame(tmp_path, sessions=12, turns=4)
    rc, out, out_dir = _run(tmp_path, frame, cases=24)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    assert payload["sample"]["distinct_sessions"] >= audit.MIN_SESSIONS
    assert len({c["session"] for c in payload["cases"]}) >= audit.MIN_SESSIONS, (
        "a 24-case sample of a 246-row frame could otherwise be four sessions' bad "
        "afternoon, which is what the session cap exists to prevent")
    assert payload["sample"]["complete_cases"] >= audit.MIN_CASES


def test_frame_covers_archived_transcripts_and_prints_unusable_ones(tmp_path):
    """`.json.gz` stays in frame, and an unreadable transcript is a printed count.

    Two failure modes this node pins, because both silently shrink a measurement:
    the retention sweep gzips old transcripts in place, so a `.json`-only reader
    loses every loss row whose session has been archived; and one rotten member of a
    store is the difference between "this session is unusable" and "the audit died".
    Both are clause 2's `read_session` requirement and clause 3's printed `unusable`
    count, measured together because they are the same call.
    """
    frame = _frame(tmp_path, sessions=10, turns=3)
    sids = sorted(p.name[: -len(".json")] for p in frame["sessions_dir"].glob("*.json"))
    # Five of ten sessions archived, so 15 of the frame's 30 rows are reachable only
    # through the `.json.gz` path. With 24 cases wanted and one session corrupt, the
    # draw cannot avoid the archives, which is what makes this an assertion rather
    # than a hope that seed 4238 happens to select session 0.
    for sid in sids[:5]:
        path = frame["sessions_dir"] / f"{sid}.json"
        messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
        path.unlink()
        _session_gz(frame["sessions_dir"] / f"{sid}.json.gz", messages)
    # A session whose file exists and does not parse: unusable, not absent.
    (frame["sessions_dir"] / f"{sids[9]}.json").write_text("{not json", encoding="utf-8")

    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    sessions_in = {c["session"] for c in payload["cases"]}
    assert sessions_in & set(sids[:5]), (
        "a loss row whose transcript the sweep archived left the frame silently: "
        "read_session is gzip-transparent and is the only reader here")
    assert sids[9] not in sessions_in
    assert payload["sample"]["dropped"].get("unusable") == 1, payload["sample"]
    assert "unusable transcripts: 1" in out, out


# ── clause 3: every case carries its evidence ────────────────────────────────

def test_case_carries_its_full_evidence_and_a_verbatim_quote_from_its_own_turn(tmp_path):
    frame = _frame(tmp_path)
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    assert payload["cases"], "the frame was built to be complete"
    for case in payload["cases"]:
        for key in ("session", "turn_ordinal", "turn_id", "platform", "skill",
                    "score", "landed", "injected_body", "deciding_quote", "label"):
            assert key in case, (key, sorted(case))
        assert case["turn_id"] == f"{case['session']}#{case['turn_ordinal']}", (
            "uptake's turn key is `session#ordinal` (`app/uptake.py:496-500`); a "
            "case that cannot be grepped in uptake's table is not checkable")
        assert case["platform"], "the frame is machine-authored; the split is a clause"
        assert case["landed"] is not True, (
            "the draw is the unlanded population; a landed case in the frame is a "
            "filter that stopped working")
        assert isinstance(case["injected_body"], bool)
        quote = case["deciding_quote"]
        assert len(quote) <= audit.QUOTE_MAX, len(quote)
        assert quote not in ("", None)
        transcript = json.loads((frame["sessions_dir"]
                                 / f"{case['session']}.json").read_text())
        users = [m["content"] for m in transcript["messages"] if m["role"] == "user"]
        turn_text = users[case["turn_ordinal"] - 1]
        assert quote in turn_text, (
            "the quote must be a verbatim substring of THAT turn's user text, so a "
            "reader can grep it back to the transcript")
        assert "$" not in quote


def test_chat_and_machine_cases_are_published_separately(tmp_path):
    """97% of offers come from worker sessions, so one label would merge two
    populations: for a queue-authored job prompt `surfaced-not-used` means the
    scheduler's wording did not land, not that a person declined advice."""
    frame = _frame(tmp_path)
    for i, path in enumerate(sorted(frame["sessions_dir"].glob("*.json"))):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["platform"] = "worker" if i % 2 else "chat"
        path.write_text(json.dumps(doc), encoding="utf-8")
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    assert payload["platform_split"].get("chat", 0) > 0
    assert payload["platform_split"].get("worker", 0) > 0
    assert (payload["chat_cases"]
            == sum(1 for c in payload["cases"] if c["platform"] == "chat"))
    assert "chat 0 of" not in out or payload["chat_cases"] == 0


# ── clause 4: the floors and the four published counts ───────────────────────

def test_run_refuses_to_write_and_prints_the_shortfall_below_twenty_cases(tmp_path):
    """`MIN_CASES` is a refusal, not a warning: #2507 owed entry 1 asks for "the
    >=20 hand-labelled loss cases", and a 19-case artifact quoted beside that
    sentence is a different claim wearing the same words."""
    frame = _frame(tmp_path, sessions=6, turns=3)   # 18 rows over 6 sessions
    rc, out, out_dir = _run(tmp_path, frame, cases=24)
    assert rc == 2, out
    assert "REFUSED" in out
    assert "complete cases 18 < 20" in out, out
    assert "distinct sessions 6 < 8" in out, out
    assert not list(out_dir.glob("loss-audit-*.json")), "a refusal writes nothing"


def test_a_frame_of_fewer_than_eight_sessions_is_refused_too(tmp_path):
    frame = _frame(tmp_path, sessions=4, turns=8)   # 32 rows, only 4 sessions
    rc, out, out_dir = _run(tmp_path, frame, cases=24)
    assert rc == 2
    assert "distinct sessions 4 < 8" in out, out
    assert not list(out_dir.glob("loss-audit-*.json"))


def test_all_four_labels_publish_with_measured_zeros_shown_as_zero(tmp_path):
    """Every count in the artifact — always all four labels.

    `never-written: 0` is a sentence, not an absence: it says the audit looked and
    every offered name was in circulation, which is what lets a reader rule the
    produce leg out of the loss story. Measured 2026-11-26, 0 of the 35 distinct
    names on the unlanded rows was missing from the live cohort.
    """
    frame = _frame(tmp_path)
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    assert list(payload["counts"]) == list(audit.LABELS)
    assert sum(payload["counts"].values()) == payload["sample"]["complete_cases"]
    for label in audit.LABELS:
        assert isinstance(payload["counts"][label], int)
        assert f"{label}: {payload['counts'][label]}" in out, (
            f"{label} must be printed, at zero included")


def test_each_label_fires_on_its_own_evidence(tmp_path):
    """The taxonomy is a partition of what the transcript shows, so each of the
    four needs a fixture that produces it — a label no test can reach is a label
    that will silently measure zero forever.

    `written-not-surfaced` is the draw's own condition (`landed` is derived from the
    same `mode` that decides rendering, `app/prefetch.py:733-734`), so its fixture is
    an ordinary unlanded row. The other two require a CONTRADICTION: `landed` not
    true, yet the name present in that turn's rendered `<skill name=…>` block. That
    contradiction is measured on the live store at 0 of 246 attributable rows, so
    these three labels measure zero on today's data by construction, and the
    fixtures are how a future non-zero is checkable rather than an anecdote.
    """
    assert audit.LABELS == audit.LABELS[::-1][::-1]   # published order is the tie-break
    assert audit.label_case(skill="gone", in_cohort=False, rendered=False,
                            read_after_turn=False, dispute=False) == "never-written"
    assert audit.label_case(skill="skill-01", in_cohort=True, rendered=False,
                            read_after_turn=False,
                            dispute=False) == "written-not-surfaced"
    assert audit.label_case(skill="skill-01", in_cohort=True, rendered=True,
                            read_after_turn=False,
                            dispute=False) == "surfaced-not-used"
    assert audit.label_case(skill="skill-01", in_cohort=True, rendered=True,
                            read_after_turn=True,
                            dispute=True) == "used-and-harmful"
    assert audit.label_case(skill="skill-01", in_cohort=True, rendered=True,
                            read_after_turn=True, dispute=False) is None, (
        "the row says the offer never rendered, yet the artifact reached that turn, "
        "was read and drew no correction: there is no loss to label, and a fifth "
        "label would rename the contradiction instead of reporting it")
    assert set(audit.LABELS) == {"never-written", "written-not-surfaced",
                                 "surfaced-not-used", "used-and-harmful"}


def test_a_rendered_offer_that_drew_no_read_is_published_as_surfaced_not_used(tmp_path):
    """End-to-end for the contradiction path, not just the predicate."""
    # The contradiction frame: `landed` is false on every row and the block rendered
    # anyway, which `_frame(rendered=True)` builds for ALL sessions so the label
    # cannot depend on which sessions the seed happens to draw.
    frame = _frame(tmp_path, sessions=10, turns=3, rendered=True)
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    hits = [c for c in payload["cases"] if c["label"] == "surfaced-not-used"]
    assert len(hits) == 24, payload["counts"]
    assert all(c["evidence"]["rendered_in_that_turn"] is True for c in hits)
    assert payload["counts"]["written-not-surfaced"] == 0, (
        "with every row rendered, the draw's own label must go to zero: the two "
        "labels are a partition of one signal, not two names for the same rows")
    assert payload["contradiction"]["counts"].get(
        "landed not true, yet rendered in that turn") == 24, (
        "a rendered-but-unlanded row is the audit's one finding the draw does not "
        "imply, so it must be counted where a reader will see it")


def test_dominant_label_publishes_k_n_and_the_wilson_interval(tmp_path):
    """Wilson, not Wald, for the reason `eval/stats.py` gives: at k = n Wald
    collapses to width zero and would publish a 24/24 draw as a certainty."""
    from eval.stats import wilson_ci
    frame = _frame(tmp_path)
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    d = payload["dominant_label"]
    counts, n = payload["counts"], sum(payload["counts"].values())
    assert d["k"] == max(counts.values()) and d["n"] == n
    assert abs(d["rate"] - round(d["k"] / n, 4)) < 1e-9
    assert d["wilson_ci_95"] == [round(x, 4) for x in wilson_ci(d["k"], n)]
    assert f"k/n={d['k']}/{d['n']}" in out
    assert d["k"] == n, (
        "a synthetic frame of ordinary unlanded rows must be all one label, which "
        "is the live shape: 246 of 246 attributable rows on 2026-11-26")
    assert d["wilson_ci_95"][0] == pytest.approx(0.8620, abs=1e-3), (
        "a k=n interval that reaches 1.0 would be Wald's, not Wilson's")


def test_a_tie_is_broken_on_the_published_order_not_dict_order(tmp_path):
    counts = {"never-written": 0, "written-not-surfaced": 3, "surfaced-not-used": 3,
              "used-and-harmful": 0}
    d = audit.dominant_label(counts)
    assert d["label"] == "written-not-surfaced", (
        "a tie broken on dictionary order would move with the code that builds the "
        "counts, and then the artifact's headline would depend on a refactor")
    with pytest.raises(audit.LossAuditRefused):
        audit.dominant_label({label: 0 for label in audit.LABELS})


# ── clause 5: the header's claim about itself, and acting on nothing ─────────

def test_header_declares_agent_labelled_provisional_evidence_not_hand_labels(tmp_path):
    """`labeled_by` and the disclaimer in ONE block, because #552's ruling is about
    how a set like this gets quoted: `backlog/552-…:202` reserves steps 1 and 4 for
    hand labels ("do not skip step 4") since a same-family grader may agree with its
    own bad writing. A `labeled_by` field with the caveat elsewhere is a field a
    quoter can copy and a caveat a quoter can miss. Bound: #2441/#2556, pinned by
    tests/test_fact_write_gate.py.
    """
    frame = _frame(tmp_path)
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    block = payload["labeling"]
    assert block["labeled_by"] == f"agent:lloyd-transcript-audit-{DAY}"
    joined = json.dumps(block)
    assert "NOT hand labels" in joined
    assert "Alan" in joined, "the disclaimer must name whose labels these are not"
    assert "human-labelled set" in joined
    assert "#2441" in joined and "#2556" in joined
    assert "step 4" in joined and "552" in joined, (
        "without the reservation this block reads as a completed control")
    assert "provisional" in joined


def test_header_names_the_owed_human_control_with_both_triggers_measured(tmp_path):
    """The trigger pair is read through `uptake_hand_audit`, which owns the floors,
    so the audit cannot state a number the hand-audit has since moved. A fixture
    store keeps the measurement local: `ready_date` must survive as None when the
    corpus cannot yield a rate, which is that module's "no rate can be measured"
    and not this audit's invention."""
    from scripts import uptake_hand_audit as uha
    frame = _frame(tmp_path)
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    hc = payload["human_control"]
    assert hc["trigger_turns"] == uha.TRIGGER_TURNS == 500
    assert hc["min_candidates"] == uha.MIN_CANDIDATES == 15
    assert isinstance(hc["human_turns"], int) and isinstance(hc["candidates"], int)
    assert hc["human_turns_at_or_past_trigger"] is (hc["human_turns"] >= 500)
    assert hc["candidates_at_or_past_floor"] is (hc["candidates"] >= 15)
    assert f"human_turns {hc['human_turns']} vs trigger 500" in out
    assert "step 4" in hc["owed"] and ">=5" in hc["owed"]
    assert hc["ready_date"] is None or len(hc["ready_date"]) == 10, (
        "`ready_date` is that module's None-for-no-rate answer, passed through and "
        "not filled in here")


def test_nothing_under_a_skills_root_changes_and_no_dollar_figure_is_emitted(tmp_path):
    """Clause 6, pinned as an action test rather than a promise in a docstring.

    The skills snapshot covers the artifact bytes AND mtimes: a run that quietly
    edited a SKILL.md would be the failure mode the whole audit exists to avoid —
    measurement reaching for the thing it measures. The currency check is the census's
    own cost block, deliberately not read, and a quote whose turn text carried a
    figure is refused rather than published doctored.
    """
    frame = _frame(tmp_path)
    files = lambda: [p for p in frame["skills_roots"][0].rglob("*") if p.is_file()]
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files()}
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files()}
    assert after == before, "the audit acted on what it measures"
    assert "$" not in out
    for path in out_dir.glob("loss-audit-*.json"):
        text = path.read_text(encoding="utf-8")
        assert not audit._CURRENCY.search(text), text[:200]


def test_a_quote_that_would_carry_a_currency_figure_makes_its_case_unquotable(tmp_path):
    """No dollar figure is emitted, so the alternative is that a case with a
    currency-bearing turn has no quote published and is counted. Redacting the
    figure would be the worse option: the clause demands a VERBATIM quote."""
    frame = _frame(tmp_path, sessions=10, turns=3)
    sid = sorted(p.name[: -len(".json")]
                 for p in frame["sessions_dir"].glob("*.json"))[0]
    target = frame["sessions_dir"] / f"{sid}.json"
    doc = json.loads(target.read_text(encoding="utf-8"))
    for msg in doc["messages"]:
        if msg["role"] == "user":
            msg["content"] = ("migrate the billing job. The invoice is $4,120 and "
                              "the cap is $900, so stop before the ceiling is hit. "
                              "Then write the report carefully and name the file.")
    target.write_text(json.dumps(doc), encoding="utf-8")
    assert audit.choose_quote(
        "the run cost $4,120 in total for the whole month") is None, (
        "with no currency-free window the case must be refused, not redacted")
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    for case in payload["cases"]:
        assert case["session"] != sid or "$" not in case["deciding_quote"]
    assert not audit._CURRENCY.search(
        next(out_dir.glob("loss-audit-*.json")).read_text())


def test_run_prints_exactly_one_line_naming_the_dominant_label(tmp_path):
    frame = _frame(tmp_path)
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0
    hits = [line for line in out.splitlines()
            if line.startswith("dominant loss label:")]
    assert len(hits) == 1, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    assert hits[0] == (
        f"dominant loss label: {payload['dominant_label']['label']} "
        f"k/n={payload['dominant_label']['k']}/{payload['dominant_label']['n']} "
        f"wilson95=[{payload['dominant_label']['wilson_ci_95'][0]}, "
        f"{payload['dominant_label']['wilson_ci_95'][1]}]")


def test_never_offered_cohort_is_published_as_a_count_and_never_as_a_case(tmp_path):
    """The `offered` stratum's unit is an artifact, so it has no turn and no user
    text to quote: clause 2's per-case fields beside that stratum are the jointly
    unsatisfiable pair #875 warns about. It is named as a count instead."""
    frame = _frame(tmp_path, sessions=10, turns=3)
    for name in ("skill-09", "skill-10", "skill-11", "skill-12"):
        assert (frame["skills_roots"][0] / name / "SKILL.md").exists()
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    offered = {c["skill"] for c in payload["cases"]}
    assert offered & set(frame["cohort"]), "every case must name a cohort artifact"
    assert payload["never_offered_cohort"]["count"] >= 1
    assert payload["never_offered_cohort"]["cohort_size"] == len(frame["cohort"])
    scan = audit.scan_loss_rows(frame["event_root"])
    assert payload["never_offered_cohort"]["count"] == len(
        set(frame["cohort"]) - set(scan["offered_names"]))
    unstudied = {c["skill"] for c in payload["cases"]} - set(scan["offered_names"])
    assert not unstudied, f"a case names a skill no offer row carried: {unstudied}"
    assert "no turn to read" in payload["never_offered_cohort"]["basis"]
    assert f"never-offered cohort: {payload['never_offered_cohort']['count']}" in out


def test_artifact_prints_its_own_measured_denominator_and_no_census_stage_figure(tmp_path):
    """The census this supplements is reported `stale` by the instrument index — 30
    days old against a declared 7-day cadence — so the audit may not borrow its
    numbers as if they were current. It states its own frame instead, and the frame
    fields that carry it are required keys, not prose.
    """
    frame = _frame(tmp_path, sessions=10, turns=4)
    rc, out, out_dir = _run(tmp_path, frame)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    f = payload["frame"]
    assert f["offer_rows"] == 40 and f["unlanded_offer_rows"] == 40
    assert f["attributable_unlanded_rows"] == 40
    assert f["event_files_scanned"] == 10
    assert f["distinct_sessions_in_frame"] == 10
    assert "stale" in f["denominator_note"]
    text = json.dumps(payload)
    assert "21795" not in text and "29028" not in text, (
        "#2507's Oct-18 stage counts must not be carried as if current")
    assert "dominant_loss_stage" not in text


def test_frame_shape_fixture_matches_the_measured_live_shape(tmp_path):
    """The shape this suite builds its fixtures on, kept next to the numbers.

    Measured 2026-11-26 over the 250 newest event files: 246 attributable unlanded
    rows over 38 sessions; 204 rows on `worker` sessions and 42 on `browser`, 0 on a
    chat session; 0 of the 246 rendered in that turn's `<skill name=…>` block; 22 of
    38 sessions logged fewer receipt rows than the transcript held user messages.
    Those measurements are why three of the four labels are expected to read zero,
    why `platform_split` exists, and why `sample.ordinal_corroborated_cases` is
    published instead of the ordinal being presented as the transcript's own index.
    """
    frame = _frame(tmp_path, sessions=38, turns=7)
    rc, out, out_dir = _run(tmp_path, frame, cases=24)
    assert rc == 0, out
    payload = json.loads(next(out_dir.glob("loss-audit-*.json")).read_text())
    assert payload["frame"]["attributable_unlanded_rows"] == 38 * 7
    assert payload["sample"]["complete_cases"] == 24
    assert payload["counts"]["written-not-surfaced"] == 24
    assert payload["contradiction"]["counts"] == {}, (
        "a frame with no rendered-but-unlanded row must report the contradiction as "
        "empty, which is the live measurement rather than a silence")
