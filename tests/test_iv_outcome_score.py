"""#833 — Inner Voice interventions scored against turn outcomes from the event log.

A throwaway fixture root holds a `usage.db` with the real table schema and an
`event_logs/` directory; the shipped script is run against it. Pinned: the join
on (session_id, turn_id) and the per-class line (1), the label coming from the
event log and never from an IV column (2), guard injects as their own class (3),
counts without rates under 30 scored rows (4), and a read-only run (5).

#2964 added the four nodes at the tail. The first three read the comment above
`TEST_SESSION_IDS` as text, because it justified the exclusion with a ratio measured
2026-09-24 — 580 of 583 `observer_injected` events — that the fleet has since grown
past, and a comment whose only ground is a number goes false with no test turning red.
The fourth pins the behaviour the comment is not allowed to imply has changed either:
the exclusion still works, off by default, back on with `--include-test-sessions`.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import iv_outcome_score as S  # noqa: E402

_SCHEMA = """CREATE TABLE inner_voice_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
    sequence_in_turn INTEGER NOT NULL, trigger TEXT NOT NULL,
    action TEXT NOT NULL, reason TEXT, content TEXT, related_tool TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER,
    cache_create INTEGER, latency_ms INTEGER, model TEXT, error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, safeguard TEXT)"""


def _root(tmp_path, rows, turns, blobs=None):
    """rows: (sid, tid, action, reason, safeguard, created_at);
    turns: {(sid, tid): (stop_reason|None, [tool results])}."""
    db = tmp_path / "usage.db"
    with sqlite3.connect(db) as c:
        c.execute(_SCHEMA)
        for i, (sid, tid, action, reason, guard, at) in enumerate(rows):
            c.execute("INSERT INTO inner_voice_observations (session_id, turn_id, "
                      "sequence_in_turn, trigger, action, reason, safeguard, created_at) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (sid, tid, i, "assistant_message", action, reason, guard, at))
    logs = tmp_path / "event_logs"
    (logs / "blobs").mkdir(parents=True)
    for sha, text in (blobs or {}).items():
        (logs / "blobs" / f"{sha}.txt").write_text(text)
    by_session: dict[str, list[str]] = {}
    for (sid, tid), (stop, results) in turns.items():
        lines = by_session.setdefault(sid, [])
        for res in results:
            lines.append(json.dumps({"session_id": sid, "turn_id": tid,
                                     "event": "brain1.tool_result_received",
                                     "data": {"tool_call_id": "x", "result": res}}))
        if stop is not None:
            lines.append(json.dumps({"session_id": sid, "turn_id": tid,
                                     "event": "brain1.result_message",
                                     "data": {"stop_reason": stop}}))
    for sid, lines in by_session.items():
        (logs / f"{sid}.events.jsonl").write_text("\n".join(lines) + "\n")
    return db, logs


def _run(capsys, db, logs, *extra):
    assert S.main(["--db", str(db), "--event-logs", str(logs), "--json", *extra]) == 0
    return json.loads(capsys.readouterr().out)


T0 = "2026-09-20T10:00:00.000000"


def test_join_on_session_and_turn_prints_the_per_class_line(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "keep going", None, T0),
            ("s1", "t2", "inject", "keep going", None, T0),
            ("s1", "t3", "noop", "fine", None, T0),
            ("s2", "t1", "noop", "fine", None, T0)]
    turns = {("s1", "t1"): ("max_turns", []), ("s1", "t2"): ("stop", []),
             ("s1", "t3"): ("stop", []), ("s2", "t1"): ("max_turns", [])}
    db, logs = _root(tmp_path, rows, turns)
    rep = _run(capsys, db, logs)
    inj = rep["classes"]["inject[model]"]
    assert (inj["n"], inj["tp"], inj["fp"]) == (2, 1, 1)
    # s2/t1 ended badly with no intervention — the FN pool; the same turn id in
    # another session must not borrow s1/t1's inject.
    assert rep["bad_turns"] == 2 and rep["fn_pool"] == 1

    assert S.main(["--db", str(db), "--event-logs", str(logs)]) == 0
    text = capsys.readouterr().out
    for word in ("n", "TP", "FP", "precision", "recall", "FN pool"):
        assert word in text
    assert "inject[model]" in text


def test_since_until_bound_the_window(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "a", None, "2026-09-19T23:00:00.0"),
            ("s1", "t2", "inject", "b", None, "2026-09-20T12:00:00.0"),
            ("s1", "t3", "inject", "c", None, "2026-09-21T01:00:00.0")]
    turns = {k: ("stop", []) for k in (("s1", "t1"), ("s1", "t2"), ("s1", "t3"))}
    db, logs = _root(tmp_path, rows, turns)
    rep = _run(capsys, db, logs, "--since", "2026-09-20 00:00:00", "--until", "2026-09-21T00:00:00")
    assert rep["classes"]["inject[model]"]["n"] == 1


def test_outcome_comes_from_the_event_log_not_the_iv_reason(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "task succeeded, all tests pass", None, T0)]
    db, logs = _root(tmp_path, rows, {("s1", "t1"): ("max_turns", [])})
    rep = _run(capsys, db, logs)
    assert rep["classes"]["inject[model]"]["tp"] == 1
    assert rep["classes"]["inject[model]"]["fp"] == 0


def test_tool_error_label_reads_the_result_prefix_including_blobs(tmp_path, capsys):
    sha = "ab" * 32
    rows = [("s1", "t1", "inject", "x", None, T0), ("s1", "t2", "inject", "x", None, T0)]
    turns = {("s1", "t1"): ("stop", [{"$blob": sha, "size": 9000}]),
             ("s1", "t2"): ("stop", ["ok", "  fine"])}
    db, logs = _root(tmp_path, rows, turns, blobs={sha: "Traceback (most recent call last):\n"})
    rep = _run(capsys, db, logs, "--label", "tool_error")
    assert rep["label"] == "tool_error"
    assert (rep["classes"]["inject[model]"]["tp"], rep["classes"]["inject[model]"]["fp"]) == (1, 1)


def test_a_turn_with_no_result_message_is_unlabelled(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "x", None, T0)]
    db, logs = _root(tmp_path, rows, {("s1", "t1"): (None, ["ok"])})
    rep = _run(capsys, db, logs)
    c = rep["classes"]["inject[model]"]
    assert c["n"] == 0 and c["unlabelled"] == 1 and rep["turns_unlabelled"] == 1


def test_guard_injects_are_their_own_class(tmp_path, capsys):
    rows = [("s1", "t1", "inject", "deterministic: 3 near-identical Read calls", None, T0),
            ("s1", "t2", "inject", "you skipped the test run", None, T0),
            ("s1", "t3", "inject", "stall", "stall_rescue", T0)]
    turns = {k: ("stop", []) for k in (("s1", "t1"), ("s1", "t2"), ("s1", "t3"))}
    db, logs = _root(tmp_path, rows, turns)
    rep = _run(capsys, db, logs)
    assert rep["classes"]["inject[guard]"]["n"] == 2   # prose fallback + safeguard key
    assert rep["classes"]["inject[model]"]["n"] == 1


def test_a_small_class_prints_counts_and_no_rate(tmp_path, capsys):
    rows = [("s1", f"c{i}", "cancel", "stop", None, T0) for i in range(3)]
    rows += [("s1", f"i{i}", "inject", "go", None, T0) for i in range(30)]
    turns = {("s1", f"c{i}"): ("cancelled", []) for i in range(3)}
    turns.update({("s1", f"i{i}"): ("max_turns" if i < 6 else "stop", []) for i in range(30)})
    db, logs = _root(tmp_path, rows, turns)
    rep = _run(capsys, db, logs)
    cancel, inj = rep["classes"]["cancel"], rep["classes"]["inject[model]"]
    assert (cancel["n"], cancel["tp"]) == (3, 3)
    assert cancel["precision"] is None and cancel["recall"] is None and cancel["lift"] is None
    assert inj["precision"] == pytest.approx(0.2) and inj["recall"] == pytest.approx(6 / 9, abs=1e-3)

    assert S.main(["--db", str(db), "--event-logs", str(logs)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.strip().startswith("cancel"))
    assert "—" in line and "1.000" not in line


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_read_only_run_leaves_the_fixture_root_byte_identical(tmp_path):
    rows = [("s1", "t1", "inject", "x", None, T0), ("s1", "t1", "cancel", "x", None, T0)]
    db, logs = _root(tmp_path, rows, {("s1", "t1"): ("max_turns", ["Error: nope"])})
    before = _tree(tmp_path)
    for extra in ([], ["--json"], ["--label", "tool_error"]):
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "iv_outcome_score.py"),
                               "--db", str(db), "--event-logs", str(logs), *extra],
                              capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr
    assert _tree(tmp_path) == before
    assert "mode=ro" in (ROOT / "scripts" / "iv_outcome_score.py").read_text()


# ── #2964: the comment above TEST_SESSION_IDS, read as text ───────────────────────────
#
# The scorer excludes six sessions, and until #2964 the only ground its comment gave was a
# count: 580 of 583 `observer_injected` events on disk, measured 2026-09-24. The same six
# files are 580 of ~11,400 now, so the comment's consequence clause — an all-time run
# "would be scoring the suite's fixtures" — went false while every hard rung stayed green.
# That is the shape these nodes exist for: prose whose truth depends on a number nobody
# re-measures. They read the comment out of the source file, and each one is run against
# the pre-#2964 text below as its control, because a pin that has only ever passed cannot
# tell a justified exclusion from an unjustified one.

_SOURCE = (ROOT / "scripts" / "iv_outcome_score.py").read_text(encoding="utf-8")
_DOC = (ROOT / "architecture" / "inner-voice.md").read_text(encoding="utf-8")

#: The block above `TEST_SESSION_IDS` exactly as it stood at `78e6cf5d`, comment markers
#: stripped — the text #2964 replaced. Literal so the controls need no git: a control that
#: shells out to `git show` resolves to this round's own HEAD in a worktree and would then
#: be reading the text it is supposed to be testing against.
_COMMENT_BEFORE_2964 = """Test-suite output written into the LIVE event-log directory before conftest
isolated it. These six files hold 580 of the 583 `observer_injected` events
on disk (counted 2026-09-24), so an all-time run that joined them would be
scoring the suite's fixtures. Excluded from every run; `--include-test-sessions`
puts them back for anyone auditing that claim."""


def _comment_block_above(constant: str, source: str = _SOURCE) -> str:
    """The contiguous comment lines directly above `constant`'s declaration, as prose.

    Walks up from the declaration and stops at the first non-comment line, so a block the
    round did not write — or a line squeezed between the block and the constant, which is
    how a block quietly stops being one — shows up as an empty read rather than as a green.
    """
    lines = source.splitlines()
    decl = [i for i, ln in enumerate(lines) if ln.startswith(constant)]
    assert len(decl) == 1, (
        f"`{constant}` is declared {len(decl)} times in scripts/iv_outcome_score.py, so "
        "'the comment above it' names no one block and every pin below is ambiguous"
    )
    top = decl[0]
    while top > 0 and lines[top - 1].lstrip().startswith("#"):
        top -= 1
    block = " ".join(ln.lstrip("#").strip() for ln in lines[top:decl[0]])
    assert block.strip(), (
        f"no comment block could be read above `{constant}`, so the checks below would "
        "report a comment that says the right thing by having read nothing"
    )
    return block


def _sentences(block: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]


def _identity_reason(block: str) -> str | None:
    """The sentence that gives the exclusion its real reason, or None: suite output that
    landed in the LIVE event-log directory, excluded for what it IS, with volume named as
    what does not justify it."""
    for s in _sentences(block):
        if (re.search(r"test.suite", s, re.I)
                and re.search(r"LIVE event-log directory", s)
                and re.search(r"\bidentity\b", s, re.I)
                and re.search(r"\bnot\b[^,.]*\bvolume\b", s, re.I)):
            return s
    return None


def _stale_583_sentences(block: str) -> list[str]:
    """Sentences naming 583 without any marker that dates the count to shipping."""
    return [s for s in _sentences(block)
            if re.search(r"\b583\b", s)
            and not re.search(r"\b(held|then|shipping|shipped)\b", s, re.I)]


_CURRENT_FIGURE = re.compile(r"inside\s+([\d,]+)\b[^.]*?\bas of\s+(\d{4}-\d{2}-\d{2})")


def _current_figure(text: str) -> tuple[int, str] | None:
    """The `(denominator, date)` of the dated on-disk total, as a text carries it."""
    m = _CURRENT_FIGURE.search(text)
    return (int(m.group(1).replace(",", "")), m.group(2)) if m else None


def test_the_exclusion_is_justified_by_fixture_identity_not_by_volume():
    """Clause 1: the reason the comment gives has to survive the number changing.

    Unmarked and in this file rather than a doc-claim file because it is a comment in a
    script the gate imports, and the rung that reads prose is the one the gate runs.

    The reason must be in ONE sentence, carrying all four of: suite output, the LIVE
    event-log directory, identity, and volume named as not the ground. Split across two
    sentences the claim dissolves — "written into the live directory" alone was already in
    the pre-#2964 comment, and it was not a reason, which is exactly why that comment could
    rest on a ratio.
    """
    block = _comment_block_above("TEST_SESSION_IDS")
    reason = _identity_reason(block)
    assert reason is not None, (
        "nothing above `TEST_SESSION_IDS` justifies the exclusion as the six sessions being "
        f"the suite's own fixtures rather than a share of the volume ({block!r}). A reason "
        "made of a count is a reason that expires the day the fleet grows, and it expired "
        "between 2026-09-24 and 2026-10-18 (#2964)"
    )
    # Control: the block this round replaced, which named the live-directory origin but
    # grounded the exclusion on the count instead.
    assert _identity_reason(_COMMENT_BEFORE_2964) is None, (
        "the pin accepted the pre-#2964 comment, which justified the exclusion by how many "
        "events the six files held — so it is matching words, not the reason, and would "
        "pass again if the volume-based justification were put back"
    )


def test_the_580_of_583_count_is_worded_as_a_shipping_time_figure():
    """Clause 2: 580/583 is a fact about the day the scorer shipped, and the comment has to
    say so — because read as present tense it tells a reader that an all-time run without
    this exclusion would be scoring nothing but fixtures, which stopped being true when real
    traffic grew past it.

    Two directions, both run: no sentence may name 583 undated, AND the shipping-time
    wording has to still be there. The second is what stops this clause being satisfied by
    deleting the history, which would leave the 580 in the next sentence untethered from the
    583 it is counted against.
    """
    block = _comment_block_above("TEST_SESSION_IDS")
    assert not _stale_583_sentences(block), (
        f"{_stale_583_sentences(block)}: a sentence above `TEST_SESSION_IDS` names 583 with "
        "nothing dating it, which reads as the number of `observer_injected` events on disk "
        "now. It has not been since the fleet grew (#2964)"
    )
    shipping = [s for s in _sentences(block) if re.search(r"\b583\b", s)]
    assert shipping, (
        "583 no longer appears above `TEST_SESSION_IDS` at all, so the 580 in the current "
        "figure has nothing to be 580-of and 'worded as a shipping-time count' is being met "
        "by removing the shipping time"
    )
    assert all(re.search(r"\bat shipping\b|\bwhen the scorer shipped\b", s, re.I)
               for s in shipping), (
        f"the sentence naming 583 does not say it is a shipping-time count ({shipping})"
    )
    # Control: the replaced wording, present tense and undated.
    assert _stale_583_sentences(_COMMENT_BEFORE_2964), (
        "the finder let the pre-#2964 sentence through, so 'no line asserts 583 is the "
        "current total' is pinned by a rule that cannot see the claim it was written for"
    )


def test_the_current_denominator_is_dated_and_matches_the_architecture_doc():
    """Clause 3: the comment carries the current on-disk figure with its own date, and it is
    the same figure the architecture doc carries — because two carriers of one measurement
    that may diverge is how the code came to say 99.5% while the doc had been corrected to
    5%.
    """
    block = _comment_block_above("TEST_SESSION_IDS")
    doc = _current_figure(_DOC)
    assert doc is not None, (
        "architecture/inner-voice.md no longer states a dated on-disk total for "
        "`observer_injected` events, so the comparison below would be a code comment checked "
        "against nothing — the doc half of #2964 is what makes the number checkable"
    )
    code = _current_figure(block)
    assert code is not None, (
        f"the comment above `TEST_SESSION_IDS` states no dated on-disk total ({block!r}), so "
        "the only figure it carries is the shipping-time one and a reader has no way to "
        "notice it is stale (#2964)"
    )
    assert code == doc, (
        f"the script says {code[0]:,} events as of {code[1]} and architecture/inner-voice.md "
        f"says {doc[0]:,} as of {doc[1]} — one measurement, two carriers, already drifting "
        "apart, which is the failure mode #2964 exists to close"
    )
    assert code[1].startswith("2026-10"), (
        f"the current figure is dated {code[1]}, which is not the 2026-10 re-measurement "
        "#2964 recorded"
    )
    assert code[0] > 10_000, (
        f"the figure the comment calls current is {code[0]:,}, at or below the 583 the "
        "shipping-time count measured, so it is not the post-growth denominator"
    )
    dated = next(s for s in _sentences(block) if f"as of {code[1]}" in s)
    assert "580" in dated, (
        f"the sentence carrying the dated total does not also carry the 580 it is a fraction "
        f"of ({dated!r}), so the two counts are not readable as the same numerator"
    )
    # Control: the replaced block stated no current figure at all.
    assert _current_figure(_COMMENT_BEFORE_2964) is None, (
        "the finder read a dated current figure out of the pre-#2964 comment, so the pin is "
        "matching the shipping-time count and would pass on it"
    )


def test_the_default_run_drops_the_test_sessions_and_the_flag_keeps_them(tmp_path, capsys):
    """Clause 4: the fix is a comment, so what has to be shown is that the behaviour the
    comment describes is the behaviour that ships — the same six ids, dropped by default,
    kept on request, under the flag's real name.

    No node in this file covered the exclusion before, which is why a comment could rot for
    three weeks unnoticed from both ends: nothing read the prose, and nothing read the
    filter the prose described.
    """
    assert S.TEST_SESSION_IDS == frozenset(
        {"test_sess", "guard_sess", "loop_sess", "v52_sess", "loop_replay",
         "goal_test_sess"}
    ), (
        f"`TEST_SESSION_IDS` is now {sorted(S.TEST_SESSION_IDS)} — #2964 is a comment change, "
        "so a seventh id here (or one fewer) is a behaviour change riding in on a re-wording"
    )
    rows = [("s1", "t1", "inject", "go", None, T0),
            ("test_sess", "t1", "inject", "go", None, T0)]
    turns = {("s1", "t1"): ("stop", []), ("test_sess", "t1"): ("max_turns", [])}
    db, logs = _root(tmp_path, rows, turns)

    rep = _run(capsys, db, logs)
    inj = rep["classes"]["inject[model]"]
    assert (inj["n"], inj["tp"], inj["fp"]) == (1, 0, 1), (
        f"a default run scored {inj} — only s1/t1 should survive the exclusion, and s1/t1 "
        "ended `stop`, so it is a false positive; test_sess/t1 ended badly and would have "
        "added the TP that is not in these numbers"
    )
    kept = _run(capsys, db, logs, "--include-test-sessions")
    kinj = kept["classes"]["inject[model]"]
    assert (kinj["n"], kinj["tp"]) == (2, 1), (
        f"`--include-test-sessions` scored {kinj}: the flag exists to put the six back for "
        "someone auditing the number, and without its TP the audit cannot reproduce the "
        "580 the comment cites"
    )
    assert ("--include-test-sessions" in _SOURCE
            and "--keep-test-sessions" not in _SOURCE), (
        "the flag's name moved: the option that keeps the six is `--include-test-sessions`, "
        "and #2964's own fix text asked for a `--keep-test-sessions` that has never existed. "
        "Renaming it would break the other caller that spells it, "
        "`tests/test_iv_measurement.py`, which passes the option by name"
    )

