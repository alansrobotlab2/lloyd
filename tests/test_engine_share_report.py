"""#4240 clause tests for `app/engine_share.py`: the share, its floor, its ceiling.

The corpus is seeded through `app.transcript_entries` — the builders
`app/routers/messages.py` and `app/run_recorder.py` use themselves — and through
`usage_store.record_usage`, so the reader is pinned against the rows the writers
really persist. That matters because one turn writes TWO stats shapes: the
per-iteration block (which rides on the flushed text row AND on the tool-call row
of the same iteration, so an unguarded sum counts it twice) and the turn-total
`stream_stats` block (whose `duration_ms` is the whole turn, and which the item's
literal reading sums as if it were engine time).

Those are the two mistakes the triage measurement caught on live data, and they
are distinguishable by number. On the `autotriage` fixture the correct share is
0.300; skipping the de-dupe says 0.400; also taking the turn-total row says
1.400 — above 1.0, which is what the triage found in 1,258 of 1,503 live
sessions. Each fixture class is built so exactly one misreading produces its
published number.

A missing duration is the third failure mode and the one that reads silently as a
fast engine: `owed-check` carries transcripts with no stats at all (#1052's
reconstructed rows), and the report must answer with a coverage fraction and a
refusal, never a share.
"""

from __future__ import annotations

import json
import re
from datetime import date

import pytest

from app import engine_share, usage_store
from app.transcript_entries import (build_assistant_text_entry, build_tool_call,
                                    build_tool_call_entry,
                                    build_tool_result_entry)

TODAY = date(2026, 11, 17)


def _iter_stats(iteration: int, duration_ms: int) -> dict:
    """What the writers stamp on an iteration that ended in a tool call: the
    per-iteration block — `iteration` present, `num_turns` absent."""
    return {"input_tokens": 100, "output_tokens": 5, "cache_read": 50,
            "cache_create": 0, "duration_ms": duration_ms,
            "iteration": iteration, "model": "primary"}


def _turn_stats(duration_ms: int, num_turns: int) -> dict:
    """`stream_stats` as the eager path writes it on the turn's last row."""
    return {"input_tokens": 400, "output_tokens": 12, "cache_create": 0,
            "cache_read": 200, "cost_usd": None, "duration_ms": duration_ms,
            "num_turns": num_turns, "model": "primary"}


def _write_session(root, *, sid: str, day: str, source: str,
                   platform: str = "worker", iterations: list,
                   turn_ms: int, num_turns: int, tool_ms: int, tool_calls: int,
                   duplicate_stats: bool = True,
                   measured_turn_row: bool = True):
    """One transcript plus its usage row, shaped the way the writers shape them.

    `iterations` is the list of `(iteration, engine_ms)` blocks that reached a
    stats row. With `duplicate_stats` each block rides on BOTH a flushed text row
    and the tool-call row of its own iteration, exactly as the eager chat path
    writes an iteration that streamed prose and then called a tool.
    `measured_turn_row` adds the final assistant row whose stats are
    `stream_stats` — the turn-total block the scoping rule must reject.
    """
    entries: list[dict] = []
    stamp = f"{day}T09:00:00"
    for iteration, duration_ms in iterations:
        stats = _iter_stats(iteration, duration_ms)
        if duplicate_stats:
            entries.append(build_assistant_text_entry(
                f"thinking out loud {iteration}", timestamp=stamp,
                stats=dict(stats), turn_id="t1"))
        entries.append(build_tool_call_entry(
            build_tool_call(f"c{iteration}", "Bash", "{}"),
            timestamp=stamp, stats=stats, turn_id="t1"))
        entries.append(build_tool_result_entry(
            f"c{iteration}", "ok", timestamp=stamp, turn_id="t1"))
    if measured_turn_row:
        entries.append(build_assistant_text_entry(
            "the answer", timestamp=stamp, stats=_turn_stats(turn_ms, num_turns),
            turn_id="t1"))
    else:
        # #1052's reconstructed rows: `_tool_pair(..., evt=None)` and
        # `_unpersisted_pairs` persist a row with stats ABSENT, not zero, so the
        # transcript has content and the numerator has nothing.
        entries.append(build_assistant_text_entry(
            "the answer, with nothing persisted about how long it took",
            timestamp=stamp, turn_id="t1"))
    doc = {"session_id": sid, "title": sid, "created": stamp, "updated": stamp,
           "messageCount": len(entries), "platform": platform,
           "source": source, "messages": entries}
    root.joinpath(f"{sid}.json").write_text(json.dumps(doc))
    usage_store.record_usage(session_id=sid, model="primary",
                             input_tokens=400, output_tokens=12,
                             duration_ms=turn_ms, num_turns=num_turns,
                             tool_ms_total=tool_ms, tool_calls=tool_calls)


def _corpus(root, *, cls: str, n: int, **kw):
    """`n` copies of one session shape, so a class has an n worth publishing."""
    defaults = dict(iterations=[], turn_ms=10_000, num_turns=2, tool_ms=0,
                    tool_calls=0)
    defaults.update(kw)
    for i in range(n):
        _write_session(root, sid=f"20261116_{cls}_{i:04d}", day="2026-11-16",
                       source=cls, **defaults)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Five classes, each pinning a different property of the reader.

    autotriage      — the scoping rule: 2 of 3 counted iterations persisted,
                      engine 3,000 ms of a 10,000 ms turn → 0.300 every session.
    youtube-digest  — engine-bound: 7,000 ms of 10,000 → 0.700, n=35. The class a
                      decode arm pays for.
    autocode        — Kundel's compile-bound case: 1,500 ms engine, 7,000 ms tool
                      dispatch in a 10,000 ms turn → 0.150 engine / 0.700 tool.
    owed-check      — thin coverage: 4 of 5 transcripts carry no stats at all.
    bench           — the clock guard: one session's engine ms (8,000) exceeds its
                      turn (1,000), beside two honest ones (500/1,000).
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(usage_store, "DB_PATH", tmp_path / "usage.db")
    _corpus(sessions, cls="autotriage", n=40, iterations=[(1, 1000), (2, 2000)],
            turn_ms=10_000, num_turns=3, tool_ms=3_500, tool_calls=2)
    _corpus(sessions, cls="youtube-digest", n=35,
            iterations=[(1, 3000), (2, 4000)], turn_ms=10_000, num_turns=2,
            tool_ms=1_000, tool_calls=2)
    _corpus(sessions, cls="autocode", n=31, iterations=[(1, 1500)],
            turn_ms=10_000, num_turns=2, tool_ms=7_000, tool_calls=9)
    _corpus(sessions, cls="owed-check", n=4, turn_ms=9_000, num_turns=4,
            tool_ms=500, tool_calls=1, duplicate_stats=False,
            measured_turn_row=False)
    _write_session(sessions, sid="20261116_owed-check_0004", day="2026-11-16",
                   source="owed-check", iterations=[(1, 1000)],
                   turn_ms=10_000, num_turns=2, tool_ms=500, tool_calls=1)
    _corpus(sessions, cls="bench", n=2, iterations=[(1, 500)], turn_ms=1_000,
            num_turns=1, tool_ms=100, tool_calls=1)
    _write_session(sessions, sid="20261116_bench_0002", day="2026-11-16",
                   source="bench", iterations=[(1, 8000)], turn_ms=1_000,
                   num_turns=1, tool_ms=100, tool_calls=1)
    return {"sessions": sessions, "db": tmp_path / "usage.db"}


def _collect(corpus, **kw):
    return engine_share.collect(sessions_dir=corpus["sessions"],
                                db_path=corpus["db"], today=TODAY, **kw)


def _class(report, name):
    return next(c for c in report["classes"] if c["class"] == name)


def _row(text, name):
    return next(line for line in text.splitlines() if line.startswith(name))


def test_the_share_is_the_per_iteration_sum_not_every_stats_row(corpus):
    """Clause 1: p50, p90 and n per class, over the scoped and de-duped numerator.

    autotriage persists iterations 1 and 2 at 1,000 and 2,000 ms of a 10,000 ms
    turn, and each block is written twice (flushed text row and tool-call row), so
    a reader that skips the de-dupe says 4,000/10,000 = 0.400 and one that also
    takes the turn-total row says 14,000/10,000 = 1.400. Scoped and de-duped the
    answer is 0.300.
    """
    report = _collect(corpus)
    cls = _class(report, "autotriage")
    assert cls["sessions"] == 40 and cls["coverage"] == 1.0
    assert (cls["p50"], cls["p90"]) == (pytest.approx(0.3), pytest.approx(0.3))
    assert (cls["engine_ms"], cls["turn_ms"]) == (120_000, 400_000)
    # 2 persisted iterations per session against 3 counted by the harness: an
    # iteration that answers in prose alone never reaches a stats row.
    assert (cls["iterations"], cls["iterations_expected"]) == (80, 120)
    row = _row(engine_share.format_report(report), "autotriage")
    assert "40/40" in row and "0.300" in row and "120,000/400,000" in row
    assert "0.350" in row and "80" in row


def test_the_floor_refuses_a_share_the_transcripts_cannot_support(corpus):
    """Clause 2: a class whose rows carry no stats prints coverage and a refusal.

    Four of the five `owed-check` transcripts are the reconstructed shape — rows
    with stats ABSENT (#1052's choice), not zero — so 1 session of 5 carries a
    measurement. Publishing that as 1,000 ms over the class's 46,000 ms of turn
    would read as an engine that answers in 2 % of its turn; the report prints
    `1/5` and `coverage too thin to interpret` instead.
    """
    report = _collect(corpus)
    cls = _class(report, "owed-check")
    assert (cls["measured"], cls["sessions"]) == (1, 5)
    assert cls["refusal"] == "coverage too thin to interpret"
    assert cls["p50"] is None and cls["p90"] is None and cls["pooled"] is None
    row = _row(engine_share.format_report(report), "owed-check")
    assert "coverage too thin to interpret" in row and "1/5" in row
    # The proof the clause asks for: the refused row's share columns are dashes,
    # never a 0.000 that a reader would take for a measured instant engine. The
    # tool column still prints, over the one session that has a turn row at all
    # (500 ms of its 10,000 ms), because tool time is booked independently of
    # whether the engine stamped a span.
    assert "0.000" not in row and "0.050" in row
    assert row.count("—") == 3


def test_the_floor_is_a_number_and_not_a_mood(corpus):
    """The same class publishes when the floor is lowered under its coverage."""
    cls = _class(_collect(corpus, coverage_floor=0.2), "owed-check")
    assert cls["refusal"] is None
    assert cls["p50"] == pytest.approx(0.1)          # 1,000 ms of 10,000 ms


def test_an_engine_span_longer_than_its_turn_gets_no_ratio(corpus):
    """Clause 3: `engine_ms > turn_ms` is a clock disagreement, not a share.

    `bench_0002` persisted 8,000 ms of engine inside a 1,000 ms turn. It is named
    in the report and dropped from the numerator, the denominator and the
    percentiles alike, so the class publishes 0.500 — the honest sessions' share —
    instead of a ratio above 1.
    """
    cls = _class(_collect(corpus), "bench")
    assert cls["mismatched"] == ["20261116_bench_0002"]
    assert (cls["engine_ms"], cls["turn_ms"]) == (1_000, 2_000)
    assert cls["p50"] == pytest.approx(0.5)
    text = engine_share.format_report(_collect(corpus))
    assert "clock mismatch" in text and "20261116_bench_0002" in text
    assert "engine 8,000 ms > turn 1,000 ms" in text


def test_no_share_above_one_survives_anywhere(corpus):
    """Clause 3 as a property: no published ratio of any kind exceeds 1.0.

    Checked over every column of every class and over the two pooled lines,
    because the guard cannot be satisfied by dropping a session from the share
    while its broken clock still sits in a class total — the shape the first live
    run showed, where an excluded session left engine and tool shares summing to
    1.107 on one class.
    """
    report = _collect(corpus)
    ratios = []
    for cls in report["classes"]:
        ratios += [r for r in (cls["p50"], cls["p90"], cls["pooled"],
                               cls["tool_share"], cls["coverage"])
                   if r is not None]
    text = engine_share.format_report(report)
    # The two pooled lines print `= k/N = d.ddd`, so the pattern has to cross the
    # fraction to reach the ratio. Matched with `findall` and asserted by count:
    # a pattern that silently stops matching those lines would leave this test
    # checking only the per-class columns and still reporting green.
    pooled = re.findall(
        r"(?:pooled engine_ms|tool_ms_total)/turn_ms=[\d,]+/[\d,]+ = (\d\.\d{3})",
        text)
    assert len(pooled) == 2, (
        f"the check expects both pooled lines to print a ratio, got {pooled}")
    ratios += [float(v) for v in pooled]
    assert len(ratios) > 10, "the fixture publishes ratios to be checked"
    assert max(ratios) <= 1.0, ratios


def test_a_class_where_every_session_disagrees_says_why(corpus):
    """Two refusal reasons, not one blurred message."""
    _write_session(corpus["sessions"], sid="20261116_broken_0000",
                   day="2026-11-16", source="broken-clock",
                   iterations=[(1, 5000)], turn_ms=1_000, num_turns=1,
                   tool_ms=100, tool_calls=1)
    cls = _class(_collect(corpus), "broken-clock")
    assert cls["coverage"] == 1.0 and cls["shares"] == []
    assert "every measured one disagreed on the clocks" in cls["refusal"]
    assert "coverage too thin" not in cls["refusal"]


def test_tool_share_and_call_count_ride_beside_every_share(corpus):
    """Clause 5a: the remainder is printed with the share it does not explain."""
    text = engine_share.format_report(_collect(corpus))
    columns = _row_header(text)
    for heading in ("class", "n", "coverage", "p50", "p90", "engine_ms/turn_ms",
                    "tool_ms/turn", "tool_calls"):
        assert heading in columns, f"{heading} missing from the table header"
    autocode = _class(_collect(corpus), "autocode")
    assert (autocode["p50"], autocode["tool_share"], autocode["tool_calls"]) == (
        pytest.approx(0.15), pytest.approx(0.7), 279)
    # The caveat matters as much as the number: `tool_ms_total` is a sum of
    # per-call spans, so it is a floor on the remainder ONLY if no two calls
    # overlapped. The live autocode class measured 0.367 + 0.706 = 1.073 on
    # 2026-11-17, which is that overlap, not a broken clock.
    assert "a SUM of per-call spans" in text
    # Each publishable class also gets the two columns added up, so a reader can
    # see whether they can still describe one turn. Pinned off the report dict
    # rather than typed figures.
    assert (f"engine + tool over turn_ms: {autocode['pooled']:.3f} + "
            f"{autocode['tool_share']:.3f} = "
            f"{autocode['pooled'] + autocode['tool_share']:.3f}") in text


def _row_header(text):
    return next(line for line in text.splitlines() if "tool_calls" in line
                and "coverage" in line)


def test_the_report_ends_naming_arms_with_a_verdict(corpus):
    """Clause 5b: a ruling per throughput arm, naming the class it read."""
    text = engine_share.format_report(_collect(corpus))
    decision = text[text.index("\ndecision:\n"):]
    ruling_lines = [ln for ln in decision.splitlines() if ln.strip().startswith("#")]
    assert len(ruling_lines) == 4
    for item in ("#2144", "#3771", "#4128", "#2019"):
        assert any(item in ln for ln in ruling_lines), f"{item} has no ruling"
    for line in ruling_lines:
        assert "SUPPORTS" in line or "CAPS" in line, line
    assert "youtube-digest" in decision and "0.700" in decision
    # The counterweight the ruling has to state too: the class whose turn is
    # already 70 % tool dispatch is where the same win does not reach.
    assert "autocode" in decision


def test_a_decode_arm_is_capped_when_no_class_is_engine_bound():
    """The other reading of the same instrument: nothing at 0.5 means nothing to buy."""
    rulings = engine_share.decide(
        [{"class": "autocode", "sessions": 31, "p50": 0.15, "p90": 0.40,
          "tool_share": 0.70},
         {"class": "autotriage", "sessions": 40, "p50": 0.30, "p90": 0.30,
          "tool_share": 0.35}], min_class_n=30)
    by_item = {r["item"]: r for r in rulings}
    assert by_item["#2144"]["verdict"] == "caps"
    assert by_item["#4128"]["verdict"] == "caps"
    assert "0.500" in by_item["#2144"]["why"]
    assert by_item["#3771"]["verdict"] == "caps"


def test_a_prefill_ruling_never_claims_a_split_these_columns_cannot_make():
    """Prefill is inside the engine share but not separable from it: a turn keeps
    one `ttft_ms_first` (the first chunk of its first iteration), so no summing
    recovers a per-iteration prefill. The ruling says so in every class shape."""
    for publishable in ([{"class": "youtube-digest", "sessions": 35, "p50": 0.7,
                          "p90": 0.8, "tool_share": 0.1}], []):
        by_item = {r["item"]: r
                   for r in engine_share.decide(publishable, min_class_n=30)}
        assert by_item["#3771"]["verdict"] == "caps"
        assert "ttft" in by_item["#3771"]["why"]
        assert "#520" in by_item["#3771"]["why"]


def test_a_class_below_the_publication_bar_does_not_rule(corpus):
    """`n >= 30` is the item's own bar, so a class at n=18 decides nothing even at
    a share that would otherwise clear 0.5."""
    _corpus(corpus["sessions"], cls="deep-research", n=18,
            iterations=[(1, 9000)], turn_ms=10_000, num_turns=2,
            tool_ms=200, tool_calls=1)
    report = _collect(corpus)
    assert _class(report, "deep-research")["p50"] == pytest.approx(0.9)
    decision = engine_share.format_report(report)
    decision = decision[decision.index("\ndecision:\n"):]
    assert "deep-research" not in decision
    assert "youtube-digest" in decision          # the class that does clear the bar


def test_no_ruling_means_the_instrument_is_disabled(corpus):
    """#543 close-out discipline, the clause #2275 used."""
    report = _collect(corpus)
    report["decision"] = []
    assert "instrument is disabled" in engine_share.format_report(report)


def test_the_automod_line_prices_a_decode_win_against_the_round(corpus):
    """The worked example the item asks the report to reach, as printed numbers."""
    line = _row(engine_share.format_report(_collect(corpus)), "automod rounds")
    # autocode: 31 sessions x 1,500 ms engine of 10,000 ms turn = 0.150.
    assert "engine_ms/turn_ms=46,500/310,000 = 0.150" in line
    assert "2x decode win caps at 0.075" in line


def test_the_gate_ladder_seconds_are_never_a_denominator(corpus, monkeypatch):
    """#2378's reuse ruling in the direction that can fail: a ladder figure of
    any size moves no published number, because the ladder runs detached from
    every turn and its seconds are in a round's wall clock, not in a turn's.

    Proved by difference. A stubbed ledger reporting 9,000,000 s is allowed to
    change exactly one printed line — its own — while every class column, both
    pooled lines and the automod sentence come back identical. Asserting the
    stub reached the page first is what makes the rest falsifiable: without it,
    a report that never printed the ladder at all would pass a diff test for the
    wrong reason.
    """
    import scripts.automod.backlog as B

    def collect_with(ladder):
        monkeypatch.setattr(B, "gate_duration_stats", lambda path: ladder)
        return engine_share.collect(sessions_dir=corpus["sessions"],
                                    db_path=corpus["db"], today=TODAY)

    quiet = collect_with({"n": 0})
    loud = collect_with({"mean_s": 9_000_000.0, "median_s": 9_000_000.0,
                         "p90_s": 9_000_000.0, "n": 500})
    quiet_text = engine_share.format_report(quiet)
    loud_text = engine_share.format_report(loud)
    assert "median 9000000.0 s / p90 9000000.0 s over 500 rounds" in loud_text
    assert "no ledger history (n=0)" in quiet_text
    assert len(quiet_text.splitlines()) == len(loud_text.splitlines())
    changed = [(a, b) for a, b in zip(quiet_text.splitlines(),
                                      loud_text.splitlines()) if a != b]
    assert len(changed) == 1 and "SEPARATE denominator" in changed[0][1], changed
    keys = ("sessions", "measured", "engine_ms", "turn_ms", "tool_ms",
            "tool_calls", "coverage", "p50", "p90", "pooled")
    assert [[c[k] for k in keys] for c in loud["classes"]] == \
        [[c[k] for k in keys] for c in quiet["classes"]]
    # #2378 also ruled the two wait columns out of any sum: the ladder's own
    # waiting must not be priced as work a faster engine could remove.
    assert "rung_wait" not in loud_text and "waited_s" not in loud_text


def test_the_reused_ladder_helper_is_called_not_re_derived(corpus, monkeypatch):
    """A stubbed ledger proves the report reads the scorecard's own function."""
    calls = []

    def fake_stats(path):
        calls.append(str(path))
        return {"mean_s": 900.0, "median_s": 850.0, "p90_s": 1400.0, "n": 7}

    import scripts.automod.backlog as B
    monkeypatch.setattr(B, "gate_duration_stats", fake_stats)
    text = engine_share.format_report(_collect(corpus))
    assert calls and "median 850.0 s / p90 1400.0 s over 7 rounds" in text


def test_job_class_is_the_writer_that_made_the_session():
    """The class comes off the session's own metadata, never from a guess.

    `autonomy-task:24` and `autonomy-task:84` are one class (the mint rule puts
    the task number in `source`, and per-task classes all sit under the
    publication bar), and a session with no source is named by its platform,
    which is what separates the interactive paths from the worker ones.
    """
    assert engine_share.job_class({"source": "autotriage"}) == "autotriage"
    assert engine_share.job_class({"source": "autonomy-task:84"}) == "autonomy-task"
    assert engine_share.job_class({"source": "", "platform": "mission-control"}) \
        == "mission-control"
    assert engine_share.job_class({}) == "unknown"


def test_the_window_is_a_file_name_and_never_a_clock_subtraction(tmp_path,
                                                                 monkeypatch):
    """#1001: `usage.ts` is naive UTC and the session stamps are naive local.

    A file dated 20 days ago stays out of a 7-day window even though its own
    `updated` stamp says minutes ago, and a file dated today is in even though its
    rows are old. Nothing here compares the two stores' clocks, so neither frame
    can silently move the population — which is the trap #2275's finding recorded.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(usage_store, "DB_PATH", tmp_path / "usage.db")
    _write_session(sessions, sid="20261025_old_0000", day="2026-10-25",
                   source="old", iterations=[(1, 1000)], turn_ms=2_000,
                   num_turns=1, tool_ms=0, tool_calls=0)
    _write_session(sessions, sid="20261117_new_0000", day="2026-11-17",
                   source="fresh", iterations=[(1, 1000)], turn_ms=2_000,
                   num_turns=1, tool_ms=0, tool_calls=0)
    report = engine_share.collect(sessions_dir=sessions,
                                  db_path=tmp_path / "usage.db", today=TODAY)
    assert report["files_in_window"] == 1
    assert [c["class"] for c in report["classes"]] == ["fresh"]


def test_a_session_with_no_usage_row_is_not_a_denominator(tmp_path, monkeypatch):
    """A transcript with no turn clock is not a turn whose engine was instant."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(usage_store, "DB_PATH", tmp_path / "usage.db")
    _write_session(sessions, sid="20261116_orphan_0000", day="2026-11-16",
                   source="orphan", iterations=[(1, 1000)], turn_ms=5_000,
                   num_turns=1, tool_ms=0, tool_calls=0)
    conn = usage_store._conn()
    conn.execute("DELETE FROM usage")
    conn.commit()
    report = engine_share.collect(sessions_dir=sessions,
                                  db_path=tmp_path / "usage.db", today=TODAY)
    assert report["sessions_joined"] == 0 and report["classes"] == []
    assert "joined_to_usage=0" in engine_share.format_report(report)


def test_sweep_writes_one_dated_row_set_and_prunes_its_own(corpus, tmp_path):
    """The store is bounded by the module that writes it — #2275's precedent,
    because the retention sweep's table is a vault file a code round cannot
    stage."""
    out = tmp_path / "engine-share"
    out.mkdir()
    (out / "2026-01-01.json").write_text("{}")
    report = engine_share.sweep(sessions_dir=corpus["sessions"],
                                db_path=corpus["db"], out_dir=out, today=TODAY)
    assert (out / "2026-11-17.json").is_file()
    assert not (out / "2026-01-01.json").exists()
    assert report["pruned"]["files"] == 1
    stored = json.loads((out / "2026-11-17.json").read_text())
    assert stored["schema"] == engine_share.SCHEMA
    assert "text" not in stored
    engine_share.sweep(sessions_dir=corpus["sessions"], db_path=corpus["db"],
                       out_dir=tmp_path / "second", today=TODAY, write=False)
    assert not (tmp_path / "second").exists()


def test_the_cli_reads_the_two_stores_it_did_not_write(corpus, capsys):
    """The process boundary: run as a separate process by whatever schedules it,
    over rows two other modules left behind."""
    rc = engine_share.main(["--sessions-dir", str(corpus["sessions"]),
                            "--db", str(corpus["db"]), "--dry-run"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "decision:" in printed and "autotriage" in printed
    assert "report-only" in printed


def test_the_dead_column_is_reported_as_dead_until_it_is_filled(corpus):
    """`duration_api_ms` is null on every row these fixtures wrote, and the report
    prints that rather than hiding it — the figure the post-landing check watches
    grow without anybody running the query by hand."""
    report = _collect(corpus)
    assert report["usage_rows"] == {"rows": 114, "rows_with_api_ms": 0}
    assert "duration_api_ms filled on 0/114 rows" in engine_share.format_report(
        report)


def test_the_scoping_rule_rejects_the_turn_shape_by_both_keys():
    """`is_iteration_stats` needs both keys, so a later writer adding one to the
    other shape cannot quietly move the numerator."""
    assert engine_share.is_iteration_stats(_iter_stats(1, 10))
    assert not engine_share.is_iteration_stats(_turn_stats(10, 2))
    assert not engine_share.is_iteration_stats({"iteration": 1, "num_turns": 2,
                                                "duration_ms": 10})
    assert not engine_share.is_iteration_stats({"duration_ms": 10})
    assert not engine_share.is_iteration_stats(None)


def test_percentile_is_the_same_nearest_rank_the_gate_ladder_uses():
    """Matching `gate_duration_stats`, so the two tables' p90s are comparable —
    not a mean, and not a linear interpolation."""
    assert engine_share.percentile([], 0.5) is None
    assert engine_share.percentile([1, 2, 3, 4], 0.5) == 3
    assert engine_share.percentile(list(range(1, 11)), 0.9) == 9
    assert engine_share.percentile([0.5], 0.9) == 0.5
