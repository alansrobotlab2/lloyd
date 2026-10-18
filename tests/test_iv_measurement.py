"""IV plan R3: the measurement loop — A/B split, human verdicts, weekly outcome
score, the tracked corpus, and the scorer's test-session exclusion."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3
from pathlib import Path

import pytest

from app import usage_store
from app.inner_voice import ab

ROOT = Path(__file__).resolve().parents[1]


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Human verdicts
# ---------------------------------------------------------------------------


def _row(**kw) -> int:
    base = dict(session_id="s1", turn_id="t1", sequence_in_turn=1,
                trigger="assistant_message", action="inject", content="go on")
    base.update(kw)
    return usage_store.record_inner_voice_observation(**base)


def test_a_verdict_is_stored_listed_and_cleared():
    rid = _row()
    assert usage_store.set_inner_voice_observation_verdict(rid, "up")
    got = usage_store.list_inner_voice_observations(session_id="s1")[0]
    assert got["verdict"] == "up" and got["verdict_at"]
    assert usage_store.set_inner_voice_observation_verdict(rid, None)
    got = usage_store.list_inner_voice_observations(session_id="s1")[0]
    assert got["verdict"] is None and got["verdict_at"] is None


def test_a_verdict_outside_the_pair_is_refused():
    rid = _row()
    with pytest.raises(ValueError):
        usage_store.set_inner_voice_observation_verdict(rid, "meh")
    assert not usage_store.set_inner_voice_observation_verdict(10**9, "up")


def test_the_route_answers_400_and_404():
    from fastapi import HTTPException
    from app.routers.inner_voice import set_observation_verdict

    rid = _row()
    assert _run(set_observation_verdict(rid, {"verdict": "down"}))["verdict"] == "down"
    with pytest.raises(HTTPException) as e:
        _run(set_observation_verdict(rid, {"verdict": "sideways"}))
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        _run(set_observation_verdict(10**9, {"verdict": "up"}))
    assert e.value.status_code == 404


def test_the_bubble_offers_thumbs():
    src = (ROOT / "web/src/components/ObservationBubble.tsx").read_text()
    assert "innerVoiceSetVerdict" in src and "isLabelable(obs.action)" in src


# ---------------------------------------------------------------------------
# A/B assignment
# ---------------------------------------------------------------------------

_CFG = {"enabled": True, "name": "iv-test", "start": "2026-09-25",
        "end": "2026-10-09", "fraction_on": 0.5}


def test_the_arm_is_a_pure_function_of_the_id():
    ids = [f"20260925_1200{i:02d}_{i:04x}" for i in range(60)]
    first = [ab.arm_for(i, name="iv-test") for i in ids]
    assert first == [ab.arm_for(i, name="iv-test") for i in ids]
    # A different experiment name reshuffles.
    assert first != [ab.arm_for(i, name="iv-other") for i in ids]


def test_the_split_is_roughly_the_fraction():
    ids = [f"20260925_{i:06d}_ab" for i in range(4000)]
    on = sum(ab.arm_for(i, name="iv-test", fraction_on=0.5) == "on" for i in ids)
    assert 1800 < on < 2200
    assert all(ab.arm_for(i, name="x", fraction_on=0.0) == "off" for i in ids[:200])


def test_assignment_only_inside_an_enabled_window():
    d = dt.date(2026, 9, 30)
    got = ab.assignment("20260930_120000_ab12", today=d, cfg=_CFG)
    assert got and got["inner_voice"] == (got["inner_voice_ab"]["arm"] == "on")
    assert got["inner_voice_evaluate_user_turns"] == got["inner_voice"]
    assert ab.assignment("x", today=dt.date(2026, 10, 9), cfg=_CFG) is None  # end exclusive
    assert ab.assignment("x", today=dt.date(2026, 9, 24), cfg=_CFG) is None
    assert ab.assignment("x", today=d, cfg={**_CFG, "enabled": False}) is None
    assert ab.assignment("x", today=d, cfg={"enabled": True}) is None  # no window


def test_a_new_chat_is_created_with_its_arm(monkeypatch, tmp_path):
    import app.sessions_io as sio

    monkeypatch.setattr(sio, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(ab, "ab_config", lambda: {**_CFG, "start": "2000-01-01",
                                                   "end": "2999-01-01"})
    _run(sio._save_session_meta("20260930_120000_ab12", "primary", preview="hi"))
    data = json.loads((tmp_path / "20260930_120000_ab12.json").read_text())
    assert data["inner_voice_ab"]["experiment"] == "iv-test"
    assert data["inner_voice"] == (data["inner_voice_ab"]["arm"] == "on")
    # A worker-shaped id is never enrolled.
    _run(sio._save_session_meta("20260930_120000_autocode_ab12", "primary"))
    w = json.loads((tmp_path / "20260930_120000_autocode_ab12.json").read_text())
    assert "inner_voice_ab" not in w and w["inner_voice"] is False


def test_the_ab_report_reads_outcomes_from_the_event_log(tmp_path):
    from scripts.iv_ab_report import load_sessions, report

    sessions, logs = tmp_path / "s", tmp_path / "e"
    sessions.mkdir()
    logs.mkdir()

    def make(sid, arm, stops, corr=False):
        (sessions / f"{sid}.json").write_text(json.dumps({
            "session_id": sid, "created_at": "2026-09-26T10:00:00",
            "inner_voice": arm == "on",
            "inner_voice_ab": {"experiment": "iv-test", "arm": arm},
            "messages": [{"role": "user", "content": "do it"},
                         {"role": "user", "content": "you didn't finish it" if corr else "thanks"}],
        }))
        lines = [json.dumps({"event": "brain1.result_message", "turn_id": f"t{i}",
                             "data": {"stop_reason": s, "num_turns": 3, "duration_ms": 2000}})
                 for i, s in enumerate(stops)]
        (logs / f"{sid}.events.jsonl").write_text("\n".join(lines) + "\n")

    make("a1", "on", ["stop", "stop"])
    make("a2", "off", ["stop", "max_turns"], corr=True)
    make("a3", "on", ["stop"])
    # Crossed over: assigned on, then switched off by hand.
    (sessions / "a4.json").write_text(json.dumps({
        "session_id": "a4", "created_at": "2026-09-26", "inner_voice": False,
        "inner_voice_ab": {"experiment": "iv-test", "arm": "on"}}))
    rep = report(load_sessions(sessions, "iv-test"), logs)
    assert rep["crossed_over"] == 1
    assert rep["arms"]["on"]["turns"] == 3 and rep["arms"]["on"]["bad_stop_rate"] == 0
    assert rep["arms"]["off"]["bad_stop_rate"] == 0.5
    assert rep["arms"]["off"]["correction_rate"] == 1.0
    assert rep["arms"]["on"]["median_iterations"] == 3


# ---------------------------------------------------------------------------
# The outcome scorer, weekly, without the test suite's sessions
# ---------------------------------------------------------------------------


def test_the_scorer_leaves_out_the_six_test_sessions(tmp_path, capsys):
    from scripts import iv_outcome_score as sc

    db = tmp_path / "u.db"
    c = sqlite3.connect(db)
    c.execute("""CREATE TABLE inner_voice_observations (id INTEGER PRIMARY KEY,
        session_id TEXT, turn_id TEXT, sequence_in_turn INT, trigger TEXT, action TEXT,
        reason TEXT, content TEXT, related_tool TEXT, input_tokens INT, output_tokens INT,
        cache_read INT, latency_ms INT, model TEXT, error TEXT, created_at TEXT, safeguard TEXT)""")
    for sid in ("test_sess", "real_one"):
        c.execute("INSERT INTO inner_voice_observations (session_id, turn_id, "
                  "sequence_in_turn, trigger, action, created_at) VALUES (?,?,?,?,?,?)",
                  (sid, "t", 1, "assistant_message", "inject", "2026-09-24T10:00:00"))
    c.commit()
    c.close()
    sc.main(["--db", str(db), "--event-logs", str(tmp_path), "--json"])
    assert json.loads(capsys.readouterr().out)["window"]["observations"] == 1
    sc.main(["--db", str(db), "--event-logs", str(tmp_path), "--json",
             "--include-test-sessions"])
    assert json.loads(capsys.readouterr().out)["window"]["observations"] == 2


def _recorder():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "iv_metrics_record_r3", ROOT / "scripts" / "iv_metrics_record.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_outcome_score_is_due_weekly(tmp_path):
    rec = _recorder()
    series = tmp_path / "s.jsonl"
    now = dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)
    assert rec.outcome_score_due(series, now=now)  # never run
    series.write_text(json.dumps({"recorded_at": "2026-09-27T00:00:00+00:00",
                                  "outcome_score": {"since": "x"}}) + "\n"
                      + json.dumps({"recorded_at": "2026-09-29T00:00:00+00:00"}) + "\n")
    assert not rec.outcome_score_due(series, now=now)
    assert rec.outcome_score_due(series, now=now + dt.timedelta(days=5))


# ---------------------------------------------------------------------------
# The tracked corpus
# ---------------------------------------------------------------------------


def test_the_seed_classifies_and_hand_labels(tmp_path):
    from scripts.iv_corpus import seed_rows

    day = tmp_path / "2026-09-05"
    day.mkdir()
    (day / "20260905_1_ivab.md").write_text(
        "user: find the chrome extension\n"
        "lloyd: It is in ~/lloyd/chrome-extension.\n"
        "user: [INNER VOICE] You found the extension but the user's actual question — how to load it\n"
        "lloyd: Open chrome://extensions and load it unpacked.\n"
        "user: [INNER VOICE] Stop: you have run the same call 3 times for x.\n"
        "lloyd: *(Inner Voice stopped this turn — looping)*\n")
    rows = seed_rows(tmp_path)
    kinds = [(r["kind"], r["label"]) for r in rows]
    assert ("model_inject", "helped") in kinds
    assert ("guard_repetition", None) in kinds and ("cancel", None) in kinds
    helped = next(r for r in rows if r["label"] == "helped")
    assert "chrome-extension" in helped["before"] and "unpacked" in helped["after"]


def test_the_seeded_corpus_is_in_the_tree_and_matches_the_review():
    rows = [json.loads(line) for line in
            (ROOT / "eval/iv/recovered-2026-08-22..09-21.jsonl").read_text().splitlines()]
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    assert kinds == {"guard_repetition": 242, "model_inject": 26, "cancel": 2}
    assert sum(r["label"] == "harmful" for r in rows) == 3


def test_export_appends_labelled_rows_once(tmp_path, monkeypatch):
    from scripts.iv_corpus import append_new, export_rows

    db = tmp_path / "usage.db"
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    rid = _row(session_id="x1")
    _row(session_id="x1", sequence_in_turn=2)  # unlabelled, not exported
    usage_store.set_inner_voice_observation_verdict(rid, "down")
    out = tmp_path / "labelled.jsonl"
    assert append_new(out, export_rows(db)) == 1
    assert append_new(out, export_rows(db)) == 0
    usage_store.set_inner_voice_observation_verdict(rid, "up")
    assert append_new(out, export_rows(db)) == 1  # a relabel replaces
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["label"] == "up"


def test_the_observer_replay_scores_only_what_it_can(tmp_path):
    """R4's harness: a labelled row with no recorded terminal text replays as
    an empty stop, where an inject is always right — so it is not scored."""
    from scripts.iv_observer_model_eval import load_cases, review_prompt

    rows = [
        {"kind": "model_inject", "label": "helped", "request": "load it",
         "before": "The extension is in ~/lloyd.", "session_id": "a"},
        {"kind": "model_inject", "label": "harmful", "request": "x",
         "before": "", "session_id": "b"},                     # no text: skipped
        {"kind": "model_inject", "label": "obsolete", "before": "y",
         "session_id": "c"},                                    # not scored
        {"kind": "guard_repetition", "label": None, "before": "z", "session_id": "d"},
    ]
    (tmp_path / "c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cases = load_cases(tmp_path)
    assert [(c["session_id"], c["_want"]) for c in cases] == [("a", "inject")]
    p = review_prompt(cases[0])
    assert "load it" in p and "TERMINAL" in p and "noop, inject, or cancel" in p


# ---------------------------------------------------------------------------
# The iv-chat-1 reading recorded in architecture/inner-voice.md §3 (#2963)
# ---------------------------------------------------------------------------

#: The repo doc is the live one. `workers/sources/arch_review.py::doc_slugs`
#: globs top-level `architecture/*.md`, and the vault's copy of the same name
#: (mtime 2026-08-24) has no §3 at all, so a test that read that file would be
#: grading a stale fork while the doc the cadence reviews went unread.
DOC = ROOT / "architecture" / "inner-voice.md"
VAULT_DOC = Path.home() / "obsidian" / "architecture" / "inner-voice.md"
SECTION_HEADING = "## 3. The measurement loop"

#: The window `config.yaml` armed, exactly as `window_open` reads it: `start`
#: inclusive, `end` exclusive, both local dates.
WINDOW = "[2026-09-25, 2026-10-09)"
#: The day the reading was taken (`iv_ab_report.py --experiment iv-chat-1`).
READ_DATE = "2026-10-18"
OWED_SENTENCE = "whose reading by `iv_ab_report` is still owed"
_RATE_WORDS = ("bad-stop", "bad_stop", "tool-error", "tool_error", "correction")


def _flat(text: str) -> str:
    """One line per line, markup and wrapping gone.

    The doc is wrapped at ~80 columns and its load-bearing phrases are bolded,
    so a match written against the prose has to survive both, or the check fails
    on a re-wrap rather than on a changed fact.
    """
    return " ".join(text.replace("`", "").replace("*", "").split())


def _section_3(text: str) -> str:
    """§3 and nothing else: from its heading to the next `## ` heading.

    Scoped because every clause says §3 — a ruling parked in the Review log
    would satisfy a whole-file `in` test while being invisible to the reader who
    opens the doc for the measurement loop.
    """
    assert SECTION_HEADING in text, f"no {SECTION_HEADING!r} heading in {DOC.name}"
    rest = text.split(SECTION_HEADING, 1)[1]
    end = rest.find("\n## ")
    assert end != -1, "§3 runs to the end of the file, so its span is unbounded"
    body = rest[:end]
    assert len(body) > 500, f"§3 extracted only {len(body)} chars — vacuous"
    return body


def _py_function(name: str, rel: str = "app/sessions_io.py") -> str:
    """The source of one top-level function, for claims about what code calls."""
    import ast

    tree = ast.parse((ROOT / rel).read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"no top-level {name}() in {rel}")


def _cited_span(rel: str, start: int, end: int) -> str:
    """The lines a doc citation points at, 1-based and inclusive."""
    lines = (ROOT / rel).read_text().splitlines()
    assert 0 < start <= end <= len(lines), f"{rel}:{start}-{end} is past EOF ({len(lines)} lines)"
    return "\n".join(lines[start - 1:end])


def test_section_3_records_the_zero_population_without_inventing_a_rate():
    """Clause 1: the reading is recorded, and zero turns buys no rate.

    Both arms really are 0 sessions / 0 turns, so the honest record is the zero
    plus the fact that `iv_ab_report` prints `null` for every rate. A bad-stop,
    tool-error or correction figure quoted from this experiment would be a
    division by nothing wearing the clothes of a measurement.
    """
    s = _flat(_section_3(DOC.read_text()))

    assert "both arms: 0 sessions, 0 turns" in s, s[:400]
    assert WINDOW in s, "the reading is not tied to the window it was taken over"
    assert READ_DATE in s, "a reading without its date is a claim about no particular day"
    assert "null" in s, "the report's own nulls are the evidence that no rate exists"
    assert "no arm-level bad-stop, tool-error or correction rate exists for iv-chat-1" in s

    for word in _RATE_WORDS:
        for sentence in (p for p in s.split(". ") if word in p):
            assert "%" not in sentence, f"{word} carries a percentage: {sentence!r}"
            for banned in ("precision", "lift", "0.0", "0.1"):
                assert banned not in sentence, \
                    f"{word} sentence carries a rate figure ({banned!r}): {sentence!r}"


def test_section_3_records_that_the_reopen_condition_ran_and_closed():
    """Clause 2: #2149's condition was a promise about this day, and it fired.

    The condition is quoted rather than paraphrased: the whole point is that the
    ruling is no longer open, so the sentence that closed it has to survive in
    the doc that now carries the verdict.
    """
    s = _flat(_section_3(DOC.read_text()))

    assert "#2149" in s, "the ruling's provenance is not recorded"
    assert "after the inner_voice.ab window end 2026-10-09, if iv_ab_report still " \
           "reads 0/0 arms, record-only is permanent" in s
    assert "permanent" in s
    assert "record-only" in s
    assert "#833" in s, "the ruling the condition guards is #833's; name it"
    assert "no observer-prompt change" in s.lower(), "permanence must say what it forbids"


def test_section_3_names_the_wiring_gap_and_its_citations_resolve():
    """Clause 3: the empty arms are a wiring fact, and the pointers are live.

    Three things have to hold together: the doc names both writers, the line
    ranges it cites really are those functions, and the code really does call
    ab.assignment from one place only — because the one place is the branch a
    chat cannot reach, since create_session has already written the file.
    """
    s = _flat(_section_3(DOC.read_text()))

    assert "_save_session_meta" in s and "create_session" in s and "ab.assignment" in s
    assert "app/sessions_io.py:1360-1368" in s and "app/sessions_io.py:398" in s

    cited_branch = _cited_span("app/sessions_io.py", 1360, 1368)
    assert "ab import assignment" in cited_branch and "_iv_ab(session_id)" in cited_branch
    assert "mission-control" in cited_branch
    assert _cited_span("app/sessions_io.py", 398, 398).startswith("def create_session(")

    create = _py_function("create_session")
    assert "assignment" not in create, "create_session stamps now, so the doc is stale"
    assert "assignment(" in _py_function("_save_session_meta") or \
           "ab import assignment" in _py_function("_save_session_meta")

    hits = [p.relative_to(ROOT) for p in (ROOT / "app").rglob("*.py")
            if "ab import assignment" in p.read_text()]
    assert [Path("app/sessions_io.py")] == hits, \
        f"ab.assignment is called from {hits}, not only from sessions_io"


def test_section_3_says_the_arms_are_unrecoverable_and_where_to_stamp():
    """Clause 4: ab.py promises a recompute the reading half does not perform.

    `iv_ab_report.py` selects sessions from the stored `inner_voice_ab` stamp and
    never recomputes from the session id, so an experiment that was never stamped
    reads 0/0 for the rest of its life — iv-chat-1 included. The doc has to carry
    that asymmetry, or the next window is armed the same way and fails the same way.
    """
    s = _flat(_section_3(DOC.read_text()))

    assert "inner_voice_ab" in s and "stamp" in s
    assert ":67" in s and ":144" in s, "the two lines that make the arms unrecoverable"
    assert "cannot be reconstructed after the fact" in s
    assert "future window must stamp at the single create writer (create_session)" in s

    report_src = (ROOT / "scripts" / "iv_ab_report.py").read_text()
    assert "assignment(" not in report_src, \
        "the report recomputes now, so the doc's unrecoverability claim is stale"
    assert "inner_voice_ab" in _py_function("load_sessions", "scripts/iv_ab_report.py")
    assert "inner_voice_ab" in _cited_span("scripts/iv_ab_report.py", 67, 67)
    assert "inner_voice_ab" in _cited_span("scripts/iv_ab_report.py", 144, 144)


def test_the_owed_sentence_is_gone_and_the_ruling_is_read_from_the_repo_doc():
    """Clause 5: the debt is paid in the file that owed it, and re-failing works.

    Two failure modes are pinned here. A rewrite that drops the ruling must go
    red — hence 0/0 and `permanent` are both required, not just the absence of
    the owed sentence. And a rewrite that points this check at
    ~/obsidian/architecture/inner-voice.md, the stale fork with no §3, must go
    red too — hence the section extractor is shown to fail on exactly that shape
    of text, written here rather than read from the vault so the proof does not
    depend on how long the fork stays stale.
    """
    text = DOC.read_text()

    assert OWED_SENTENCE not in text
    assert "still owed" not in text, "an owed-marker survived in the doc that just paid it"
    assert "#2963" in text, "the item that closed must be named by the text that closed it"

    s = _flat(_section_3(text))
    assert "0 sessions" in s and "permanent" in s and READ_DATE in s

    assert DOC == ROOT / "architecture" / "inner-voice.md"
    assert not str(DOC).startswith(str(VAULT_DOC.parent)), "grading the vault fork"
    fork_shape = "Intro prose only.\n\n## Some other section\n\nNothing here.\n"
    try:
        _section_3(fork_shape)
    except AssertionError as e:
        assert "heading" in str(e), f"failed for the wrong reason: {e}"
    else:
        raise AssertionError("_section_3 accepted a page with no §3: "
                             "pointing it at the vault copy would pass vacuously")


def test_the_grep_command_section_3_publishes_reproduces_the_call_site():
    """The command §3 tells an operator to run has to run.

    The first draft of this record published `git grep -n "ab.assignment" -- app/`,
    which returns nothing: the only call site is written
    `from app.inner_voice.ab import assignment`, so no pattern of that shape
    matches it. A verify command that exits 1 is worse than none — the reader
    concludes the call site is gone. So the command is extracted from the prose and
    executed, here rather than eyeballed, because the same mistake is one edit
    away for whoever rewrites this bullet.
    """
    import re as _re
    import subprocess

    commands = _re.findall(r"`(git grep [^`]+)`", _section_3(DOC.read_text()))
    assert len(commands) == 1, f"expected one published command in §3, got {commands}"

    proc = subprocess.run(commands[0], shell=True, cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"the command §3 publishes exits {proc.returncode}: {proc.stderr.strip()!r}")
    hits = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(hits) == 1, f"§3 says one line, the command returned {len(hits)}: {hits}"
    assert hits[0].startswith("app/sessions_io.py:1363:"), \
        f"the one hit is not the stamping call site: {hits[0]!r}"
    assert "import assignment" in hits[0]
