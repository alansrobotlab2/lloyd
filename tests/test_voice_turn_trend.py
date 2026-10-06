"""The nightly voice-latency trend (#2273): percentiles with their denominator.

The store is one row per spoken turn (`app/voice_turns.py`, written by the LiveKit
worker). What this job owes is not the percentiles — any `statistics` one-liner has
those — it is the three things that decide whether a percentile may be believed: N
printed beside every one of them, the rows it could not read counted out loud, and an
explicit refusal to grade a population below the declared floor. A report that prints
`p90 1.4 s` over four turns is the rig-sampled number that argued the 2026-09-17 voice
rework, re-labelled as production.
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy
import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Every stage a microphone turn that replied leaves behind.
_FULL = {"speech_end": 0.0, "vad_close": 0.1, "turn_verdict": 0.2, "asr_done": 0.3,
         "embed_done": 0.15, "inject_sent": 0.05, "voice_turn": 0.05,
         "first_delta": 0.4, "first_clause": 0.3, "first_tts_byte": 0.2,
         "first_pushed": 0.1, "first_played": 0.05}
#: A typed turn spoken out loud: everything between the microphone and the injection is
#: absent, because there was no utterance to run it on. Measured on 2026-10-06, 7 of the
#: 8 `[latency]` lines the worker had ever emitted were exactly this shape.
_TYPED = {"first_delta": 0.4, "first_clause": 0.3, "first_tts_byte": 0.2,
          "first_pushed": 0.1, "first_played": 0.05}


@pytest.fixture(scope="module")
def vtt():
    # scripts/maintenance has no __init__.py — the same loader
    # tests/test_referential_integrity.py uses for this directory.
    path = ROOT / "scripts" / "maintenance" / "voice_turn_trend.py"
    # No skip: this diff ships the file, so its absence is a move or a rename, and the
    # answer to that is sixteen red nodes rather than sixteen silent skips.
    assert path.is_file(), f"{path} is not on disk — the trend script moved or was renamed"
    spec = importlib.util.spec_from_file_location("voice_turn_trend", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(*, label="voice", stages=None, eos=1.4, gap=0.5, interrupted=False,
         queued_behind=False, tools_ran=False, turn_id="t1"):
    return {"v": 1, "turn_id": turn_id, "room": "lloyd-20261006_063000",
            "label": label, "at": "2026-10-06T06:30:00.500Z",
            "epoch": 1791268200.5, "stages": _FULL if stages is None else stages,
            "eos_to_audio": eos, "max_gap_s": gap, "interrupted": interrupted,
            "queued_behind": queued_behind, "tools_ran": tools_ran,
            "spoken_chars": 40, "capped": False}


def _store(tmp_path, rows, *, extra_lines=()):
    path = tmp_path / "turns.jsonl"
    lines = [json.dumps(row) for row in rows] + list(extra_lines)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _out(capsys):
    return capsys.readouterr().out


def _blocks(stdout):
    """`{label: that population's printed block}`, keyed on its own `label`."""
    out = {}
    for chunk in stdout.split("── population ")[1:]:
        match = re.match(r"(.+): (\d+) row\(s\)", chunk)
        assert match, chunk.splitlines()[0]
        out[match.group(1)] = chunk
    return out


def _table(stdout):
    """`{row label: (N, missing%, p50, p90, p95)}` from the printed tables.

    Deliberately positional: a test that fails because a column moved is a test that
    should fail, because those columns are the report's contract with whoever reads it
    at 07:00.
    """
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[1].isdigit():
            out[parts[0]] = tuple(parts[1:])
    return out


def test_the_declared_floor_is_thirty_rows_and_the_gap_bound_is_eight_seconds(vtt):
    # Both numbers are the item's, and both live in the script that grades them rather
    # than in prose: 30 from the acceptance clause, 8 s from the voice latency plan's
    # bound on the silence inside a tool turn (the 2026-09-24 turn was silent for 44 s).
    assert vtt.MIN_ROWS_FOR_VERDICT == 30
    assert vtt.MAX_GAP_P90_BOUND_S == pytest.approx(8.0)


def test_the_table_prints_n_beside_every_percentile(vtt, tmp_path, capsys):
    store = _store(tmp_path, [_row(turn_id=f"t{n}") for n in range(3)])
    assert vtt.main(["--store", str(store)]) == 0
    table = _table(_out(capsys))
    assert set(vtt.STAGES) <= set(table), f"every stage of STAGES gets a row: {sorted(table)}"
    for stage, (n, _missing, p50, p90, p95) in table.items():
        assert n == "3", f"{stage}'s percentiles must say how many turns they rest on"
        assert "-" not in (p50, p90, p95), stage
    assert table["eos→audio"][0] == "3"
    assert table["max_gap_s"][0] == "3"


def test_a_malformed_line_is_skipped_and_counted_not_dropped_silently(vtt, tmp_path,
                                                                      capsys):
    # A row whose write was killed mid-append is exactly these three shapes. Skipping
    # them silently would make the printed N a survivor count.
    store = _store(tmp_path, [_row(), _row(), _row()],
                   extra_lines=['{"v":1,"stages":{', "[1,2,3]", "not json at all"])
    assert vtt.main(["--store", str(store)]) == 0
    out = _out(capsys)
    assert "3 row(s) read, 3 malformed line(s) skipped" in out
    assert _table(out)["eos→audio"][0] == "3", "the unreadable lines are not in the sample"


def test_below_the_floor_the_report_abstains_with_its_n_printed(vtt, tmp_path, capsys):
    rows = [_row(gap=44.0) for _ in range(2)]        # a breach, if it were graded
    assert vtt.main(["--store", str(_store(tmp_path, rows))]) == 0, \
        "an unmeasured night is not a failed one, or every quiet week is an incident"
    out = _out(capsys)
    assert "abstain" in out
    assert "n=2" in out, "the abstention says what it abstained over"
    assert "verdict: red" not in out and "verdict: green" not in out, \
        "below the floor neither verdict is available"


def test_the_floor_is_per_population_not_pooled(vtt, tmp_path, capsys):
    # 40 rows that all say `typed:user` is still zero microphone turns. Pooling them to
    # clear a floor would let typed traffic certify a voice verdict.
    store = _store(tmp_path, [_row(label="typed:user") for _ in range(40)]
                   + [_row(label="voice") for _ in range(2)])
    assert vtt.main(["--store", str(store)]) == 0
    blocks = _blocks(_out(capsys))
    assert set(blocks) == {"typed:user", "voice"}
    assert "abstain" in blocks["voice"]
    assert "verdict: green" in blocks["typed:user"]


def test_over_the_floor_a_breached_gap_bound_is_red_and_exits_nonzero(vtt, tmp_path,
                                                                      capsys):
    # 44 s is the 2026-09-24 incident's number, against the plan's 8 s bound.
    store = _store(tmp_path, [_row(gap=44.0) for _ in range(30)])
    assert vtt.main(["--store", str(store)]) == 1, \
        "a bad night has to show in the autonomy run log without anyone reading the table"
    out = _out(capsys)
    assert "verdict: red" in out
    assert "max_gap_s p90 44.000s > bound 8.0s" in out


def test_over_the_floor_inside_the_bounds_is_green(vtt, tmp_path, capsys):
    store = _store(tmp_path, [_row(gap=0.5) for _ in range(30)])
    assert vtt.main(["--store", str(store)]) == 0
    out = _out(capsys)
    assert "verdict: green" in out and "abstain" not in out
    assert "1 bound(s) graded" in out, \
        "only the declared bound is graded, and the line says how many that was"


def test_eos_to_audio_is_reported_as_an_owed_bound_not_an_invented_one(vtt, tmp_path,
                                                                       capsys):
    # The item's step 5 sets this bound from the first real week of rows. Until a week
    # exists the honest output is `not declared yet`, and a 300 s p90 must not be turned
    # red by a threshold nobody measured.
    assert vtt.EOS_TO_AUDIO_P90_BOUND_S is None
    store = _store(tmp_path, [_row(gap=0.5, eos=300.0) for _ in range(30)])
    assert vtt.main(["--store", str(store)]) == 0
    out = _out(capsys)
    assert "eos→audio p90: not declared yet (owed #2273)" in out
    assert "verdict: green" in out
    assert _table(out)["eos→audio"][2:] == ("300.000", "300.000", "300.000"), \
        "the number is still printed; only the verdict is withheld"


def test_a_typed_population_never_borrows_the_microphone_sample(vtt, tmp_path, capsys):
    store = _store(tmp_path,
                   [_row(label="typed:user", stages=_TYPED, eos=None) for _ in range(30)]
                   + [_row(label="voice") for _ in range(2)])
    assert vtt.main(["--store", str(store)]) == 0
    blocks = _blocks(_out(capsys))
    typed, voice = _table(blocks["typed:user"]), _table(blocks["voice"])
    assert typed["asr_done"][0] == "0" and typed["asr_done"][2] == "-", \
        "a stage no row in this population has prints no sample and no number"
    assert voice["asr_done"][0] == "2"
    assert typed["first_clause"][0] == "30", "the TTS-side stages keep their own N"
    assert typed["eos→audio"][0] == "0", "and no typed turn pretends to an end of speech"


def test_the_missing_stage_rate_and_the_interruption_share_are_printed(vtt, tmp_path,
                                                                       capsys):
    half = dict(list(_FULL.items())[:6])
    store = _store(tmp_path, [_row(stages=half), _row(stages=half, interrupted=True)])
    assert vtt.main(["--store", str(store)]) == 0
    out = _out(capsys)
    assert "missing stages: 12 of 24 cells (50.0%)" in out, \
        "half the cells of a half-recorded turn, printed rather than skipped"
    assert "interrupted 1 of 2 (50.0%)" in out
    assert "queued behind 0 of 2 (0.0%)" in out


def test_turns_are_counted_by_whether_tools_ran(vtt, tmp_path, capsys):
    # The population a #1163-class fix claims to move, printed before anyone argues it.
    store = _store(tmp_path, [_row(gap=6.0, eos=9.0, tools_ran=True, turn_id="a"),
                              _row(gap=0.4, eos=1.0, tools_ran=False, turn_id="b")])
    assert vtt.main(["--store", str(store)]) == 0
    out = _out(capsys)
    assert "tools ran: n=1 eos→audio p50 9.000s p90 9.000s" in out
    assert "no tools: n=1 p50 1.000s p90 1.000s" in out


def test_a_percentile_is_the_number_numpy_would_print(vtt):
    # The table is what a latency fix gets graded on, so the interpolation is pinned to
    # the definition a reviewer would recompute — an off-by-one here reads as a
    # regression nobody can reproduce.
    sample = [0.4, 1.1, 0.9, 2.7, 0.8, 1.4, 9.9]
    for q in (0.5, 0.9, 0.95):
        assert vtt._percentile(list(sample), q) == pytest.approx(
            float(numpy.percentile(sample, q * 100.0))), f"q={q}"
    assert vtt._percentile([], 0.5) is None, "no sample is not a zero"
    assert vtt._percentile([3.0], 0.9) == 3.0


def test_a_boolean_never_enters_a_sample(vtt):
    # `isinstance(True, int)` is true in Python; a row that wrote `true` into a latency
    # field would otherwise enter the sample as 1.0 and move a p50.
    rows = [_row(eos=1.0), _row(eos=None), {**_row(), "eos_to_audio": True}]
    assert vtt._numbers(rows, "eos_to_audio") == [1.0]


def test_an_absent_store_abstains_rather_than_reading_green(vtt, tmp_path, capsys):
    missing = tmp_path / "never-spoken" / "turns.jsonl"
    assert vtt.main(["--store", str(missing)]) == 0
    out = _out(capsys)
    assert "0 row(s) read" in out
    assert "abstain" in out and "verdict: green" not in out, \
        "a box that has not spoken yet is not a healthy box, it is an unmeasured one"


def test_the_json_output_and_the_table_are_one_measurement(vtt, tmp_path, capsys):
    store = _store(tmp_path, [_row(gap=0.5) for _ in range(30)])
    assert vtt.main(["--store", str(store), "--json"]) == 0
    payload = json.loads(_out(capsys))
    assert payload["rows_read"] == 30 and payload["malformed_lines"] == 0
    voice = payload["populations"]["voice"]
    assert voice["n"] == 30 and voice["verdict"] == "green"
    assert voice["stages"]["asr_done"]["n"] == 30
    assert voice["stages"]["asr_done"]["p90"] == pytest.approx(0.3)
    assert voice["missing_stage_cells"] == {"absent": 0, "possible": 360, "rate": 0.0}
    assert voice["flags"]["interrupted"] == {"n": 0, "share": 0.0}
    assert payload["bounds"] == {"max_gap_s_p90_s": 8.0, "eos_to_audio_p90_s": None}
    assert list(payload["populations"]) == ["voice"]

    assert vtt.main(["--store", str(store)]) == 0     # the same store, the same numbers
    assert _table(_out(capsys))["asr_done"] == ("30", "0.0%", "0.300", "0.300", "0.300")


def test_the_report_states_the_floor_it_used(vtt, tmp_path, capsys):
    # `--min-rows` exists for a synthetic rig (`scripts/voice/e2e_voice.py`'s population)
    # and for these tests. The nightly job never passes it, and the default is the floor.
    store = _store(tmp_path, [_row() for _ in range(2)])
    assert vtt.main(["--store", str(store), "--min-rows", "1"]) == 0
    out = _out(capsys)
    assert "floor for a verdict: 1 row(s) per label population" in out
    assert "verdict: green" in out


def test_the_committed_witness_reproduces_the_traffic_figures_this_item_quotes():
    """Clause 6: the traffic figures are re-derivable from bytes with history behind them.

    `backlog/data/lloyd-agent-worker.log` is a copy of `~/lloyd-data/logs/
    lloyd-agent-worker.log` truncated to the 9,303 lines that existed when the item was
    triaged on 2026-10-06, and it sits in the same place as the provenance-journal copy
    (#2225), the denials copy (#2190) and the promotions ledger copy (#1975) for the same
    reason: the numbers this whole item rests on were read out of a log file on a disk the
    retention sweep is reclaiming bytes from. Without a committed copy, "8 `[latency]`
    lines in thirteen days" is a sentence that can never be checked again — and it is the
    sentence that decided the ≥30-row floor, the per-population split, and the fact that
    the trend will abstain for weeks.

    The shell equivalent, run against the committed bytes:

        wc -l < backlog/data/lloyd-agent-worker.log                    # 9303
        grep -c "\\[latency\\]" backlog/data/lloyd-agent-worker.log     # 8
        grep -c "turn spoken" backlog/data/lloyd-agent-worker.log       # 11
    """
    from tests.board_presence import vault_root

    witness = vault_root() / "backlog" / "data" / "lloyd-agent-worker.log"
    assert witness.is_file(), (
        f"clause 6 has no bytes to re-derive from: {witness} does not exist. It is a copy "
        "of the live worker log truncated to its 2026-10-06 length (`head -9303 "
        "~/lloyd-data/logs/lloyd-agent-worker.log > ~/obsidian/backlog/data/"
        "lloyd-agent-worker.log`) and the only history the quoted figures have, so this "
        "node failing on a missing copy is the clause, not the environment")

    lines = witness.read_text(errors="replace").splitlines()
    assert len(lines) == 9303, f"the copy is {len(lines)} lines, not the 9303 triage measured"

    latency = [ln for ln in lines if "[latency]" in ln]
    assert len(latency) == 8, f"the witness holds {len(latency)} `[latency]` lines, not 8"
    assert sum("[latency] typed:user" in ln for ln in latency) == 7
    assert sum("[latency] voice " in ln for ln in latency) == 1
    spoken = [ln for ln in lines if "turn spoken" in ln]
    assert len(spoken) == 11, f"the witness holds {len(spoken)} `turn spoken` lines, not 11"
    assert sum(" voice turn spoken" in ln for ln in spoken) == 2

    # The shape finding the aggregator is built around, and it comes out of these bytes:
    # a typed turn spoken aloud has no `speech_end`, so `total()` returns None and the
    # line prints `eos→audio=?`. 7 of 8 — the reason one N per report is survivorship.
    assert sum("eos→audio=?" in ln for ln in latency) == 7, (
        "the typed-user lines no longer print `eos→audio=?`; the missing-stage split this "
        "job prints per population was justified by these seven lines")


def test_a_row_the_worker_wrote_is_a_row_the_nightly_process_reads(tmp_path):
    """The seam, crossed for real: writer here, trend job in another interpreter.

    Every other node in this file builds its rows as dicts. That pins the reader against
    the format, but the thing that actually has to hold is that the bytes
    `app.voice_turns.append_turn` puts on disk are the bytes this job parses a day later
    in a different process — the store and the reader have no shared call, only a file.
    So this node writes through the real writer, from a real `TurnTimeline`, and runs the
    script with `subprocess`.
    """
    import subprocess
    import sys

    from app import voice_turns
    from voice.timeline import TurnTimeline

    store = tmp_path / "turns.jsonl"
    for i in range(2):
        tl = TurnTimeline("voice")
        tl.mark("speech_end", 10.0)
        tl.mark("vad_close", 10.1)
        tl.mark("asr_done", 10.4)
        tl.mark("first_played", 11.4)
        tl.audio_pushed(11.4, 11.5, 0.4)
        voice_turns.append_turn(
            voice_turns.turn_row(tl, turn_id=f"seam{i:02d}", room="lloyd-seam",
                                 tools_ran=(i == 1), spoken_chars=25,
                                 epoch=1791268200.5 + i),
            path=store)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "maintenance" / "voice_turn_trend.py"),
         "--store", str(store), "--json"],
        capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-800:]
    payload = json.loads(proc.stdout)
    assert payload["rows_read"] == 2 and payload["malformed_lines"] == 0, payload
    voice = payload["populations"]["voice"]
    assert voice["n"] == 2 and voice["verdict"] == "abstain"
    assert voice["stages"]["asr_done"]["n"] == 2
    # 0.3, not 0.4: the row stores the STAGE DELTA (asr_done minus vad_close,
    # 10.4 - 10.1), not the offset from speech_end.
    assert voice["stages"]["asr_done"]["p50"] == pytest.approx(0.3)
    table = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "maintenance" / "voice_turn_trend.py"),
         "--store", str(store)],
        capture_output=True, text=True, timeout=120, check=False).stdout
    assert "tools ran: n=1" in table and "no tools: n=1" in table, table[-600:]
