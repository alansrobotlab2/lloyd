"""The Inner Voice metrics series (backlog #460): the grader's numbers, persisted.

`scripts/iv_grade.py` could always compute intervention rate, landed rate, miss rate
and the dropped-verdict count. What it could not do is leave them anywhere: as of
2026-09-14 `grep -rln "iv_grade" ~/obsidian/autonomy ~/obsidian/skills` returned
nothing, `~/lloyd/_pipeline/reflection/iv-metrics.jsonl` did not exist, and a live run
for 09-01..09-12 — 847 turns, 9,027 LLM calls, 353 dropped verdicts — evaporated with
the terminal that printed it. So the metric existed and the trend did not, which is why
#458 was written off one hand-run, why the observer's `model: primary` pin has been
"pending a re-run" since the secondary changed on 09-06, and why verdicts that hit
their async deadline and were filed as a silent `noop` have told nobody for a fortnight.

Every test here drives the real thing: a fixture `usage.db`, real copies of
`scripts/iv_grade.py` and `scripts/iv_metrics_record.py`, joined by a real shell pipe,
under a `HOME` aimed at the fixture. Nothing monkeypatches the seam, because the seam
*is* the change — a job whose documented command and executed command are two different
strings is exactly how a scheduled task ends up measuring nothing.

Run:
  .venvs/lloyd/bin/python -m pytest tests/test_iv_metrics_series.py -q
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
GRADER = SCRIPTS / "iv_grade.py"
RECORDER = SCRIPTS / "iv_metrics_record.py"
AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
#: The task file clause 1 requires. Globbed, not hardcoded, so renaming it does not
#: silently un-pin every test below.
TASK_GLOB = "86-*iv*metrics*.md"

# Loaded by path rather than by putting `scripts/` on sys.path: that directory holds
# 40+ modules, several of which shadow stdlib names, and a shadowed `json` or `re`
# would break unrelated tests at random.
_spec = importlib.util.spec_from_file_location("iv_metrics_record", RECORDER)
iv_metrics_record = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iv_metrics_record)  # type: ignore[attr-defined]

#: The pipeline exactly as the task file documents it, ending at its last flag.
#: Extracted from the prompt rather than hardcoded here, because what is under test is
#: that the string the runner hands the model is itself runnable.
PIPELINE_RE = re.compile(r"(cd ~/lloyd && python3 scripts/iv_grade\.py .*?--hours \d+)")

# Fixture rows sit on a fixed instant so the window test asserts "1 row, not 2" rather
# than "whatever the clock said when pytest ran". Local-naive, because `created_at` is
# written local-naive (#835) — this is the clock the grader filters on.
NOW_LOCAL = datetime.datetime(2026, 9, 14, 3, 0, 0)
#: How far UTC runs *ahead* of this box's wall clock on the fixture date. The whole of
#: #835 is this number, so it is measured from the fixture instant and the system's own
#: timezone rather than written down: the previous constant said `8 # PDT`, which was
#: wrong twice over — PDT is UTC−7, and a hardcoded value does not move when the
#: fixture does. A naive `datetime` read as local gives the offset in force on that
#: date, so a fixture that crosses a DST change moves with it.
LOCAL_UTC_AHEAD_HOURS = -(
    NOW_LOCAL.astimezone().utcoffset().total_seconds()) / 3600.0

_OBS_COLUMNS = """id, session_id, turn_id, sequence_in_turn, trigger, action,
                  reason, content, related_tool, input_tokens, output_tokens,
                  cache_read, cache_create, latency_ms, model, error, created_at"""


def _iso(moment: datetime.datetime) -> str:
    """The stored shape: ISO with a `T`, no offset, sub-second."""
    return moment.isoformat(timespec="microseconds")


def _make_repo(tmp_path: Path) -> Path:
    """A throwaway repo root: copies of both scripts and its own `usage.db`.

    Copied, not symlinked — `iv_grade.py` anchors the database at
    `Path(__file__).resolve().parents[1]`, and a symlink resolves back to the real
    checkout. That would leave clause 4's byte-comparison checking a file the run
    never opened.
    """
    repo = tmp_path / "lloyd"
    (repo / "scripts").mkdir(parents=True)
    for name in ("iv_grade.py", "iv_metrics_record.py"):
        shutil.copy2(SCRIPTS / name, repo / "scripts" / name)
    return repo


def _make_db(repo: Path, rows: list[dict]) -> Path:
    """A fixture `usage.db` carrying the live `inner_voice_observations` schema."""
    db = repo / "usage.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE inner_voice_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        turn_id TEXT NOT NULL, sequence_in_turn INTEGER NOT NULL,
        trigger TEXT NOT NULL, action TEXT NOT NULL, reason TEXT, content TEXT,
        related_tool TEXT, input_tokens INTEGER, output_tokens INTEGER,
        cache_read INTEGER, cache_create INTEGER, latency_ms INTEGER,
        model TEXT, error TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    for i, row in enumerate(rows, start=1):
        conn.execute(
            f"INSERT INTO inner_voice_observations ({_OBS_COLUMNS}) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, row.get("session_id", "s1"), row.get("turn_id", f"t{i}"),
             row.get("sequence_in_turn", i),
             row.get("trigger", "assistant_message"),
             row.get("action", "noop"), row.get("reason", ""), None, None,
             row.get("input_tokens", 1000), row.get("output_tokens", 5),
             row.get("cache_read", 0), 0, row.get("latency_ms", 400),
             row.get("model", "primary"), row.get("error"),
             _iso(row["created_at"])))
    conn.commit()
    conn.close()
    return db


def _call(*, since: str, repo: Path, out: Path,
          extra: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run the documented pipeline: `iv_grade.py --json --since X | iv_metrics_record.py`.

    Two real processes and a real pipe, launched by bash with `HOME` set to the fixture
    root so `cd ~/lloyd` resolves inside the fixture. The deliverable *is* a shell
    pipeline a model will paste, so the test pastes it too.
    """
    script = (f"cd ~/lloyd && python3 scripts/iv_grade.py --json --since '{since}'"
              f" | python3 scripts/iv_metrics_record.py --out '{out}'"
              + "".join(f" {e}" for e in (extra or [])))
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          check=False, cwd=str(repo.parent),
                          env={"HOME": str(repo.parent),
                               "PATH": "/usr/bin:/bin:/usr/local/bin"})


def _rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _near_local_now(text: str) -> bool:
    """Is `text` within five minutes of the local wall clock (not the UTC one)?"""
    then = datetime.datetime.fromisoformat(text)
    drift = abs((datetime.datetime.now() - then).total_seconds())
    return drift < 300


def _task_file() -> Path:
    matches = sorted(AUTONOMY_DIR.glob(TASK_GLOB))
    assert matches, (
        f"clause 1: no IV-metrics task file matching {TASK_GLOB} under "
        "~/obsidian/autonomy/ — the scheduled half of #460 is what makes the "
        "recorder run at all")
    return matches[0]


# ── clause 2: the documented command appends exactly one parsing row ──────────


def test_documented_pipeline_appends_exactly_one_json_object(tmp_path):
    """One run appends exactly one line, and that line parses as a JSON object.

    Acceptance clause 2, verified "by a one-shot manual run against a fixed
    `--since`" — automated: one matching fixture row in, one line out. Two lines
    would double every delta; a pretty-printed object would not parse as JSONL.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2)}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"

    result = _call(since=_iso(NOW_LOCAL - datetime.timedelta(hours=26)),
                   repo=repo, out=out, extra=["--hours 26"])

    assert result.returncode == 0, result.stderr
    assert out.exists(), "the series file was not created"
    rows = _rows_of(out)
    assert len(rows) == 1, f"expected exactly 1 appended row, got {len(rows)}"
    # `_rows_of` already json.loads each line, so "is a dict" is a fact about the
    # helper, not the row. What clause 2's one-shot run actually has to produce is a
    # row carrying the grader's numbers — a `{"written": true}` stub would satisfy a
    # type check and leave nothing to trend.
    row = rows[0]
    assert row["llm_calls"] == 1, row
    assert row["since"] == _iso(NOW_LOCAL - datetime.timedelta(hours=26))
    assert _numeric(row["dropped_verdicts"]) and _numeric(row["turns"])


def test_second_run_appends_one_more_row(tmp_path):
    """Two runs leave two rows — the file is appended, never rewritten.

    Clause 2 says each run appends exactly one line. A recorder that rewrote the
    file would leave a plausible single row and destroy the trend, which is the
    only thing #460 exists to create.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2)}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    since = _iso(NOW_LOCAL - datetime.timedelta(hours=26))

    first = _call(since=since, repo=repo, out=out, extra=["--hours 26"])
    rows_after_first = _rows_of(out)
    second = _call(since=since, repo=repo, out=out, extra=["--hours 26"])
    rows = _rows_of(out)

    assert first.returncode == 0 and second.returncode == 0, second.stderr
    assert len(rows_after_first) == 1
    assert len(rows) == 2, f"two runs must leave two rows, got {len(rows)}"
    assert [r["llm_calls"] for r in rows] == [1, 1]


# ── clause 3: the fields a delta needs, as numbers ────────────────────────────


def test_row_exposes_the_delta_fields_as_numbers(tmp_path):
    """turns, llm_calls, dropped verdicts, window bounds — end to end.

    Clause 3's list, against a fixture of known content: 2 turns, 2 LLM-call rows,
    one of them dropped at the 12.0 s deadline. The dropped count is *derived from
    `cost.errors`* — the grader buckets it as `error.split(':')[0]`, so the key is
    `timeout after 12.0s` — which is the point: nothing in `app/` reads the error
    column, so this row is the only place the drops ever get counted.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [
        {"created_at": NOW_LOCAL - datetime.timedelta(hours=3),
         "turn_id": "tA", "error": "timeout after 12.0s"},
        {"created_at": NOW_LOCAL - datetime.timedelta(hours=2),
         "turn_id": "tB", "trigger": "result", "action": "noop"},
    ])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"

    result = _call(since=_iso(NOW_LOCAL - datetime.timedelta(hours=26)),
                   repo=repo, out=out, extra=["--hours 26"])

    assert result.returncode == 0, result.stderr
    row = _rows_of(out)[0]
    for key in ("turns", "llm_calls", "dropped_verdicts", "error_total",
                "observations", "window_hours"):
        assert _numeric(row[key]), f"{key} is not numeric: {row[key]!r}"
    assert row["turns"] == 2, row["turns"]
    assert row["llm_calls"] == 2, row["llm_calls"]
    assert row["dropped_verdicts"] == 1, row["dropped_verdicts"]
    assert row["error_total"] == 1, "rows with no error must not count as errors"
    assert row["dropped_rate"] == pytest.approx(0.5)
    assert row["timeout_by_deadline"] == {"timeout after 12.0s": 1}
    assert row["since"] == _iso(NOW_LOCAL - datetime.timedelta(hours=26))
    assert row["landed_rate"] is None, "no inject in the fixture to score"
    assert row["miss_rate"] is None, "no session transcript to compare"
    assert row["models"] == ["primary"], "model names come from cost['models']"


def test_until_is_stamped_from_the_local_clock():
    """`until` bounds the rows, so it must be on the clock the rows are stored on.

    Clause 3 names the window bounds and clause 6 forbids the UTC skew from
    silently moving them. `datetime.utcnow()` in the recorder would put `until`
    `LOCAL_UTC_AHEAD_HOURS` ahead of the newest row it claims to bound — invisible in
    the row, and exactly what #835 is. Asserted against the live clock rather than a
    pinned one, because the defect is which clock, not what time.

    The second assert is unguarded. It previously read `if UTC_OFFSET_HOURS:` on a
    hardcoded constant, which is a permanently-true condition: the branch could never
    be skipped, so the guard documented nothing and a fixture move that changed the
    offset would not have flipped it. The offset is now measured, and this box could
    in principle be UTC, where `until` and UTC coincide — so the counterfactual is
    expressed as a distance from the *fixture's* offset, not as a truthiness test.
    """
    until = iv_metrics_record._local_now_text()
    assert _near_local_now(until), f"until={until} is not local wall clock"
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    drift = (now_utc - datetime.datetime.fromisoformat(until)).total_seconds()
    expected = LOCAL_UTC_AHEAD_HOURS * 3600
    assert abs(drift - expected) < 300, (
        f"until is {drift/3600:.2f} h from UTC; the fixture's offset says it should "
        f"be {expected/3600:.2f} h — a stamp in the wrong clock drifts by exactly "
        f"the offset, so this fails only if `until` is not local")


def test_row_carries_rates_as_numbers_when_the_grader_scored_something():
    """`landed_rate` and `miss_rate` reach the row as numbers, not prose.

    The end-to-end fixtures above leave both null (nothing to score), which is the
    honest value — but a field that is only ever null in tests is not a field
    anyone can take a delta of. This drives `_row` with a report shaped as
    `iv_grade.py:289-300` emits it (`_cost` at `:221-260`): `cost.errors` holds two
    deadline buckets plus one `http_error` and no `no reason` key, because the
    counter at `:257-259` only buckets rows whose `error` is truthy. The fixture
    matches that, and the asserts below spell out what each field must then be:
    `dropped_verdicts` and `error_total` are both 353 — every key in the map is a
    dropped call — while `timeout_by_deadline` keeps only the two `timeout` keys,
    excluding `http_error`.
    """
    report = {
        "scope": {"session": "all", "since": "2026-09-13T01:00:00",
                  "first": "2026-09-13T01:04:11", "last": "2026-09-14T02:59:02"},
        "cost": {"observations": 7000, "turns": 847, "llm_calls": 9027,
                 "input_tokens_per_turn": 81000, "observer_ms_per_turn": 412,
                 "models": {"primary": 9027},
                 "errors": {"timeout after 12.0s": 234,
                            "timeout after 5.0s": 118, "http_error": 1}},
        "precision_proxy": {"landed_rate": 0.955, "stranded": 3},
        "recall_proxy": {"miss_rate": 0.0,
                         "terminal_noops_with_a_following_user_message": 6400},
    }

    row = iv_metrics_record._row(report, window_hours=26.0, threshold=0.10,
                                threshold_source="default", window_rows=7)

    assert row["landed_rate"] == pytest.approx(0.955)
    assert row["miss_rate"] == pytest.approx(0.0)
    assert row["llm_calls"] == 9027
    assert row["dropped_verdicts"] == 353, row["dropped_verdicts"]
    assert row["error_total"] == 353
    assert row["dropped_rate"] == pytest.approx(0.0391, abs=1e-4)
    assert row["timeout_by_deadline"] == {"timeout after 12.0s": 234,
                                          "timeout after 5.0s": 118}
    assert row["flagged"] is False
    assert row["window_hours"] == 26.0
    assert row["models"] == ["primary"]


def test_window_hours_is_recorded_so_a_wrong_flag_is_visible(tmp_path):
    """`window_hours` is metadata; a wrong one cannot masquerade as a real change.

    The recorder cannot re-derive the span from the grader's output — the grader
    reports the bound it was handed, not the hours it covered. Storing the requested
    span beside `since`/`until` is what lets a reader notice a row gathered with the
    wrong flag instead of reading it as a regression.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2)}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    since = _iso(NOW_LOCAL - datetime.timedelta(hours=26))

    assert _call(since=since, repo=repo, out=out,
                 extra=["--hours 13"]).returncode == 0
    assert _call(since=since, repo=repo, out=out,
                 extra=["--hours 26"]).returncode == 0
    rows = _rows_of(out)
    assert [r["window_hours"] for r in rows] == [13.0, 26.0]
    assert rows[0]["llm_calls"] == rows[1]["llm_calls"] == 1


# ── clause 4: usage.db is untouched ──────────────────────────────────────────


def test_run_leaves_usage_db_byte_untouched(tmp_path):
    """Same bytes, same mtime, same `sqlite_master` — the read-only contract holds.

    Clause 4. `scripts/iv_grade.py:1-6` documents that nothing in the chat path may
    depend on it and opens it `mode=ro`; the reason the series is a JSONL file and
    not a new column is that a scheduled job with nobody watching must not be able
    to write the table the live observer is writing. This checks the three things a
    writer would change: content, mtime, and schema.
    """
    repo = _make_repo(tmp_path)
    db = _make_db(repo, [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2)}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"

    def _schema_fingerprint() -> str:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return hashlib.sha256("".join(
                str(r) for r in conn.execute(
                    "SELECT type,name,sql FROM sqlite_master ORDER BY name")
                .fetchall()).encode()).hexdigest()
        finally:
            conn.close()

    before_bytes = hashlib.sha256(db.read_bytes()).hexdigest()
    before_mtime = db.stat().st_mtime_ns
    before_schema = _schema_fingerprint()

    assert _call(since=_iso(NOW_LOCAL - datetime.timedelta(hours=26)),
                 repo=repo, out=out, extra=["--hours 26"]).returncode == 0

    assert hashlib.sha256(db.read_bytes()).hexdigest() == before_bytes
    assert db.stat().st_mtime_ns == before_mtime
    assert _schema_fingerprint() == before_schema
    assert not (repo / "usage.db-wal").exists(), "a WAL sidecar means a write happened"
    assert not (repo / "usage.db-shm").exists()


def test_the_recorder_itself_opens_no_database():
    """The append half has no sqlite in it, and its output path is a flag.

    Clause 4 says the job writes nothing to `usage.db`. The grader is allowed to read
    it; the recorder is the half that runs nightly unattended, so it is held to the
    stricter rule — no sqlite import, no database path anywhere in its code.
    """
    code = re.sub(r'"""[\s\S]*?"""|#[^\n]*', "", RECORDER.read_text())
    assert "import sqlite3" not in code
    assert "usage.db" not in code, "the recorder must not name the database at all"
    assert "--out" in RECORDER.read_text(), "the destination must not be hardcoded"


# ── clause 5: a stated threshold, compared over fields ────────────────────────


def test_breach_exits_2_when_the_recent_median_is_over_the_bound(tmp_path):
    """A sustained breach is an exit code, computed from JSONL fields.

    Clause 5: "flag the row when the last N rows exceed it, with the comparison
    expressed over JSONL fields rather than prose". Four prior rows at 0.20 plus
    this run's 1.0: median 0.20 > 0.10, so exit 2. Exit status rather than a
    sentence because the autonomy runner consumes exit codes reliably —
    `scripts/skill_verdicts.py:320` and `scripts/validate_handoff.py:70` use 2 for a
    refusal for the same reason, and `_detect_silent_failures` (`autonomy.py:26-32`) only catches
    prose it has been told to look for.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2),
                     "error": "timeout after 12.0s"}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    out.parent.mkdir(parents=True)
    for i in range(4):
        out.open("a").write(json.dumps({"since": f"prior-{i}",
                                        "dropped_rate": 0.20}) + "\n")

    result = _call(since=_iso(NOW_LOCAL - datetime.timedelta(hours=26)),
                   repo=repo, out=out, extra=["--hours 26"])

    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    row = _rows_of(out)[-1]
    assert row["dropped_rate"] == pytest.approx(1.0)
    assert row["flagged"] is True and row["breach"] is True
    assert row["threshold"] == pytest.approx(iv_metrics_record.DEFAULT_THRESHOLD)
    assert "BREACH" in result.stderr


def test_a_lone_spike_does_not_breach_and_a_healthy_row_stops_one(tmp_path):
    """Both directions of the median, since the median is the whole design choice.

    One spiked row among six healthy ones must not breach (a nightly report on
    single-night noise is a report nobody reads), and a healthy row must be able to
    stop an ongoing breach — under "every row over the bound" it never could, and the
    alert would fire forever after one bad night.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2),
                     "error": "timeout after 5.0s"}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    out.parent.mkdir(parents=True)
    since = _iso(NOW_LOCAL - datetime.timedelta(hours=26))

    # This run breaches (1.0 of 1 calls). Priors: one at 0.2 and four healthy, so
    # the median over [0.0,0.0,0.0,0.0,0.2] + [1.0] is 0.0 and the healthy majority
    # clears it — impossible under "every row over the bound".
    out.open("a").write(json.dumps({"since": "b", "dropped_rate": 0.2}) + "\n")
    for _ in range(4):
        out.open("a").write(json.dumps({"since": "h", "dropped_rate": 0.0}) + "\n")
    stopped = _call(since=since, repo=repo, out=out,
                    extra=["--hours 26", "--window-rows 5"])
    assert stopped.returncode == 0, (stopped.stdout, stopped.stderr)
    last = _rows_of(out)[-1]
    assert last["flagged"] is True and last["breach"] is False

    # The lone spike: six healthy priors, this run breaches, window 7 -> median 0.
    out.unlink()
    for _ in range(6):
        out.open("a").write(json.dumps({"since": "h", "dropped_rate": 0.0}) + "\n")
    spike = _call(since=since, repo=repo, out=out,
                  extra=["--hours 26", "--window-rows 7"])
    assert spike.returncode == 0, (spike.stdout, spike.stderr)
    assert _rows_of(out)[-1]["breach"] is False


def test_malformed_prior_rows_are_counted_not_read_as_healthy(tmp_path):
    """A truncated tail is reported, not mistaken for an absence of problems.

    The file is appended by a job that can be killed mid-write. If a non-parsing line
    simply vanished from the denominator, a series quietly rotting would report "no
    breach" — the same shape as this repo's other guards that read a missing input
    and still hand back a verdict.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2)}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text('{"dropped_rate": 0.02}\n{"dropped_rate": 0.02}\n{"dropped_ra\n')

    result = _call(since=_iso(NOW_LOCAL - datetime.timedelta(hours=26)),
                   repo=repo, out=out, extra=["--hours 26"])

    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text().splitlines()[-1])["malformed_prior_rows"] == 1
    assert "WARNING 1 unreadable prior row" in result.stdout
    assert "BREACH" not in result.stderr


def test_recorder_refuses_a_report_with_no_window_bound(tmp_path):
    """A windowless run is refused rather than stored as a comparable row.

    `iv_grade.py --json` with no `--since` reports scope "all time", whose
    `dropped_rate` would sit in the series beside nightly rows and make every delta
    meaningless. Exit 3 and no append: the failure has to be louder than a healthy
    row would be.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2)}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    grader = subprocess.run(
        [sys.executable, str(repo / "scripts" / "iv_grade.py"), "--json"],
        cwd=str(repo), capture_output=True, text=True, check=False)
    assert grader.returncode == 0, grader.stderr

    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "iv_metrics_record.py"),
         "--out", str(out)],
        input=grader.stdout, cwd=str(repo), capture_output=True, text=True,
        check=False)

    assert result.returncode == 3, (result.stdout, result.stderr)
    assert not out.exists(), "a windowless report must not create a series row"


# ── clause 6: the window bound cannot drop the newest hours (#835) ────────────


def test_local_naive_bound_keeps_the_newest_row_and_drops_the_old_one(tmp_path):
    """A 26-hour local bound selects by time: the 20-hour row is in, the 30's out.

    Clause 6 asks for a fixed window's `llm_calls` "computed both ways within the
    offset"; this is the honest way and the sibling below is the dishonest one. Both
    fixture rows cost an LLM call and are otherwise identical, so `llm_calls == 1`
    can only be true if the boundary was compared correctly — a bound that let
    everything through reads 2, and one that silently dropped part of the window
    reads 0.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [
        {"created_at": NOW_LOCAL - datetime.timedelta(hours=20), "turn_id": "new"},
        {"created_at": NOW_LOCAL - datetime.timedelta(hours=30), "turn_id": "old"},
    ])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"

    assert _call(since=_iso(NOW_LOCAL - datetime.timedelta(hours=26)),
                 repo=repo, out=out, extra=["--hours 26"]).returncode == 0

    row = _rows_of(out)[0]
    assert row["llm_calls"] == 1, (
        "a 26-hour window over local-naive rows must select only the 20-hour-old "
        f"row; got llm_calls={row['llm_calls']}")
    assert row["turns"] == 1


def test_a_utc_authored_bound_on_the_same_rows_would_have_measured_nothing(tmp_path):
    """The #835 defect reproduced on the same fixture, as the counterfactual.

    A bound authored in UTC is `utcnow - 26h` written as a naive string. Since UTC
    runs `LOCAL_UTC_AHEAD_HOURS` ahead of this box, that string reads
    `NOW_LOCAL - (26 - offset)` hours in local terms — *later* than the honest bound
    by the whole offset. Here it lands past both stored rows, so the honest count is
    0 and the run must fail loudly rather than record a clean-looking night.

    Pinned so nobody "simplifies" the task file's bound to `date -u` and quietly
    empties the series: the recorder stores whatever bound it is handed, so the bound
    the job passes is the only defence that exists.

    The offset is measured, not asserted as 7 or 8 — the first version hardcoded 8 and
    was wrong about this box (UTC−7), and the test still passed because the counterfactual
    survives at either value. What makes that a real pin is the precondition below: the
    geometry only discriminates while the dishonest bound lands *after* the newest stored
    row, so rather than pass vacuously on a box with a small offset, the test says so.
    """
    # Newest fixture row is NOW_LOCAL-20h; the dishonest bound is NOW_LOCAL-(26-offset)h,
    # which passes it only once 26-offset > 20, i.e. offset > 6 h. At or below 6 the two
    # bounds select the same rows and this test can no longer tell them apart — a fixture
    # or timezone move into that regime must fail here rather than quietly go green.
    assert LOCAL_UTC_AHEAD_HOURS > 6, (
        f"this box is only {LOCAL_UTC_AHEAD_HOURS} h behind UTC, so a `date -u` bound "
        "and the honest bound select the same rows and this test can't distinguish "
        "them — widen the fixture window past `26 + offset` hours rather than deleting "
        "the test")

    repo = _make_repo(tmp_path)
    _make_db(repo, [
        {"created_at": NOW_LOCAL - datetime.timedelta(hours=20), "turn_id": "new"},
        {"created_at": NOW_LOCAL - datetime.timedelta(hours=30), "turn_id": "old"},
    ])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    honest = NOW_LOCAL - datetime.timedelta(hours=26)
    utc_authored = (NOW_LOCAL + datetime.timedelta(hours=LOCAL_UTC_AHEAD_HOURS)
                    - datetime.timedelta(hours=26))
    assert utc_authored > honest, "the counterfactual bound must be the later one"

    result = _call(since=_iso(utc_authored), repo=repo, out=out,
                   extra=["--hours 26"])

    assert result.returncode == 3, (
        "a UTC-authored bound selected no rows, so the grader printed 'no "
        f"observations matched'; the run must fail loudly, not append zeros "
        f"(rc={result.returncode}, stdout={result.stdout[:120]!r})")
    assert not out.exists()


def test_the_documented_bound_is_local_wall_clock():
    """The task file's bound is `date -d`, and it says in words why not `date -u`.

    Clause 6's other half: the window must be *written* so the skew cannot drop
    hours. `date -u` in the pipeline would produce the empty-series case above on
    real data, so the exact string is pinned — with the reasoning, which is what
    stops the next editor from "fixing" it back.
    """
    body = _task_file().read_text()

    assert "date -d '26 hours ago' +%Y-%m-%dT%H:%M:%S" in body
    assert "date -u" in body, "the file must say not to use date -u"
    assert "#835" in body, "the bound's reasoning has to live in the file"


# ── clauses 1 and 2 across the scheduler seam ────────────────────────────────


def test_task_file_frontmatter_parses_and_is_schedulable():
    """Clause 1: frontmatter parses as YAML, house style, and can actually dispatch.

    Same style as `60-knowledge-health-report.md`. `status: up_next` and a
    resolvable `skill_name` are checked because `autonomy.py:510-521` makes either
    one missing a permanent silent skip — a task file that parses but never
    dispatches would satisfy the clause's grep and still measure nothing.
    """
    fm = yaml.safe_load(_task_file().read_text().split("---\n", 2)[1])

    assert fm["frequency"] == "daily" and fm["runs_per_day"] == 1
    assert fm["status"] == "up_next"
    assert fm["skill_name"] and (
        Path.home() / "obsidian" / "skills" / fm["skill_name"] / "SKILL.md"
    ).exists(), f"skill {fm['skill_name']!r} does not resolve"
    assert int(fm["id"]) == int(re.match(r"(\d+)-", _task_file().name).group(1))
    assert fm["preferred_hours"], "nightly work needs a window in the schedule"

    # The `model:` value has to name a configured model. The first version said
    # `eco`, which is no key under `config.models` and no alias — 30 of the other 33
    # task files say `primary`. `autonomy.run_task` hands an unmatched name to
    # `_get_model_env`, which returns `{}` rather than failing, so the run boots with
    # no `ANTHROPIC_BASE_URL` override and inherits whatever the parent had. A
    # nightly that silently runs on the wrong engine is worse than one that fails:
    # its numbers still look like measurements. Asserted against config.yaml so a
    # future `eco` tier is a green test, not a guess.
    models = (yaml.safe_load((ROOT / "config.yaml").read_text())
              .get("models") or {})
    names = {*(models.keys()), *(m.get("alias") for m in models.values())}
    assert fm["model"] in names, (
        f"model {fm['model']!r} is not a config.models name or alias {sorted(n for n in names if n)}")

    # And the inverse of "the task file exists": an ARMED task dispatches nightly,
    # so the script its body names has to exist in the live tree by then. The
    # previous round hit this exactly — the reviewer ran the documented command
    # against ~/lloyd and got `can't open file .../iv_metrics_record.py`, which is
    # why the task shipped `status: draft`. Arming and landing are one change; if
    # this ever fails, the fix is to land the script or disarm the task, never to
    # leave both halves half-done.
    assert fm["status"] != "up_next" or (ROOT / "scripts" / "iv_metrics_record.py").exists(), (
        "task is armed but scripts/iv_metrics_record.py is not in the live checkout — "
        "the nightly would dispatch into a FileNotFoundError")


def test_task_body_invokes_the_grader_with_an_explicit_window():
    """Clause 1's own pin: the body invokes `python3 scripts/iv_grade.py --json`.

    `grep -l "iv_grade" ~/obsidian/autonomy/*.md` is the clause's check; this adds the
    parts that make the grep mean something — `--json` (the machine-readable path,
    `scripts/iv_grade.py:267`), an explicit `--since` (`:266`), and the recorder at the other
    end of the pipe writing the named series file.
    """
    body = _task_file().read_text()

    assert "python3 scripts/iv_grade.py --json" in body
    assert "--since" in body
    assert "iv_metrics_record.py" in body
    assert "_pipeline/reflection/iv-metrics.jsonl" in body


def test_prompt_the_scheduler_builds_contains_a_runnable_command(tmp_path):
    """The command the *runner* sees is the command that works — executed end to end.

    This is the seam #460 would otherwise miss. `autonomy.py:674-694` builds a run's
    prompt from the frontmatter `description` plus the skill file: the markdown
    **body is not in it**. A task whose procedure lives only in the body passes
    clause 1's grep and then leaves the model to improvise, which is how a scheduled
    job ends up not running the thing it documents. So: build the real prompt with the
    real loader, take the pipeline out of it, run it under a HOME aimed at a fixture
    repo, and require a row.
    """
    import autonomy

    task = autonomy._parse_task_file(_task_file())
    assert task and not task.get("_yaml_broken"), "task file must parse cleanly"
    skill_content = autonomy._load_skill_content(task["skill_name"])
    assert skill_content, f"skill {task['skill_name']!r} did not load"
    prompt = autonomy._build_task_prompt(task, skill_content)

    match = PIPELINE_RE.search(prompt)
    assert match, (
        "the scheduler-built prompt carries no runnable iv_grade pipeline — the "
        "procedure has to live in the frontmatter description (which is injected), "
        "not only in the markdown body (which is not)")
    command = match.group(1)

    repo = _make_repo(tmp_path)
    _make_db(repo, [{"created_at": datetime.datetime.now()
                     - datetime.timedelta(hours=2)}])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    run = subprocess.run(
        ["bash", "-c", command.replace("--hours 26", f"--out '{out}' --hours 26")],
        capture_output=True, text=True, check=False, cwd=str(tmp_path),
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"})

    assert run.returncode in (0, 2), (run.stdout, run.stderr)
    assert out.exists(), (
        "the command as written in the prompt produced no row:\n"
        f"{run.stdout}\n{run.stderr}")
    rows = _rows_of(out)
    assert len(rows) == 1 and rows[0]["llm_calls"] == 1, rows


def test_task_file_threshold_matches_the_code_default():
    """The bound stated in prose equals the bound in code.

    Clause 5 requires a *stated* threshold. A number in a prompt and a different
    number in the script is worse than no number: the run report quotes the prose
    while the exit code follows the code. The measured baseline (0.039) is also
    required, because that is what makes the bound legible as "about 2.5x what we
    measure today" rather than an arbitrary 0.10.

    Alan has not chosen the final value or the alert channel — an open
    person-decision on #460 — so this pins the two copies agreeing, not the
    magnitude. Retune both together and it stays green.
    """
    text = _task_file().read_text()
    frontmatter, body = text.split("---\n", 2)[1], text.split("---\n", 2)[2]
    desc = str(yaml.safe_load(frontmatter).get("description", ""))

    # The bound is the number named *by the word "bound"*, in both halves: finding it
    # separately means neither the frontmatter the runner sees nor the body a human
    # reads can silently drift from the code.
    #
    # `body` is the markdown after the closing fence, not the whole file. Passed the
    # whole file, `re.search` returns the *frontmatter* match again — the first
    # iteration of this test did exactly that, so both iterations graded the same
    # string and the body's own "**Bound: 0.10**" was never located. Anchored on the
    # word `Bound:` at the start of a line so a stray "bound" in prose can't stand in.
    for label, chunk in (("description", desc), ("body", body)):
        stated = (re.search(r"[Bb]ound\D{0,24}?(\d\.\d+)", chunk) if label == "description"
                  else re.search(r"Bound:\s*(\d\.\d+)", chunk))
        assert stated, (f'the {label} must state the bound, e.g. "Bound: 0.10" — '
                        f"searched {len(chunk)} chars of the {label}")
        assert float(stated.group(1)) == pytest.approx(
            iv_metrics_record.DEFAULT_THRESHOLD), (
            f"{label} says {stated.group(1)}, DEFAULT_THRESHOLD is "
            f"{iv_metrics_record.DEFAULT_THRESHOLD}")
    assert "0.039" in text, "the measured baseline belongs next to the bound"


def test_the_live_series_has_two_dated_rows_and_a_computable_delta():
    """The item's own verification command, as an assertion rather than a skip.

    #460's acceptance: "the JSONL must exist with ≥2 dated rows" so a delta is
    computable. The previous version of this test bailed out with `pytest.skip` when
    there was no `usage.db` next to it, which in practice meant every gate run — the
    ladder tests a worktree, which has no database and no `_pipeline/`, so the clause
    was pinned by a test that never executed. Skips are not a way to keep a clause
    pinned.

    So the path is the *live checkout's*, resolved from `HOME` rather than from this
    file's own tree: the artifact is data, not source, and the round wrote its two
    seed rows into the same file the nightly job will append to. Same instant, same
    command the reviewer ran (`wc -l ~/lloyd/_pipeline/reflection/iv-metrics.jsonl`).

    *Two consecutive scheduled nights* is the item's own person-decision and is not
    claimed here; what this pins is that the file exists, every row is dated, and the
    delta is a field read.
    """
    series = Path.home() / "lloyd" / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    assert series.exists(), (
        f"no series file at {series} — the one-shot manual verification run is part "
        "of this round, and clause 2 is not met without it")

    rows = _rows_of(series)
    assert len(rows) >= 2, (
        f"clause 2 needs ≥2 dated rows so a delta exists; {series} has {len(rows)}")
    for row in rows:
        for field in ("since", "until", "recorded_at"):
            assert datetime.datetime.fromisoformat(row[field]), (
                f"{field}={row.get(field)!r} is not an ISO timestamp")
        assert _numeric(row["llm_calls"]) and _numeric(row["threshold"])
        assert _numeric(row["dropped_rate"]) or row["llm_calls"] == 0, (
            "a null dropped_rate is only honest for a night with no calls")

    # The delta itself — the thing #460 exists to make possible — computed the way
    # the task file documents Step 3, from fields, over the last two rows.
    delta = rows[-1]["dropped_rate"] - rows[-2]["dropped_rate"]
    assert _numeric(delta), delta
    assert rows[-1]["since"] != rows[-2]["since"], (
        "two rows with the same window bound are one measurement written twice, not a "
        "trend; the seed rows differ in `since` on purpose")


def test_a_consumer_compares_the_timeout_count_against_the_bound():
    """Clause 5's other half: something can read the count against the bound.

    The item says the timeout count has to sit in the row "so a threshold on it is a
    one-line check over the last N rows rather than a parse of prose". The previous
    version of this test computed that check and then asserted `isinstance(over,
    list)`, which cannot fail for any input — it asserted the type of a list
    comprehension, not that the comparison discriminates.

    So: hand `over_bound` four rows whose answers are known independently of the
    code — under the bound, over it, exactly on it, and one with no calls — and
    require exact membership. Then check the live rows agree with their own stored
    `flagged` field, so the shipped nightly decision and the consumer are the same
    function rather than two readings of the same number.
    """
    rows = [
        # under: 5 of 100 against 0.10 -> bound is 10 calls, 5 < 10
        {"llm_calls": 100, "threshold": 0.10, "dropped_verdicts": 5, "flagged": False},
        # over: 20 of 100 -> 20 > 10
        {"llm_calls": 100, "threshold": 0.10, "dropped_verdicts": 20, "flagged": True},
        # exactly on the bound: 10 of 100 -> 10 > 10 is False
        {"llm_calls": 100, "threshold": 0.10, "dropped_verdicts": 10, "flagged": False},
        # no traffic: no rate, so nothing to breach
        {"llm_calls": 0, "threshold": 0.10, "dropped_verdicts": 0, "flagged": False},
    ]
    for row in rows:
        assert iv_metrics_record.over_bound(row) is row["flagged"], row

    over = [r for r in rows if iv_metrics_record.over_bound(r)]
    assert [r["dropped_verdicts"] for r in over] == [20], (
        f"exactly one of the four rows is over the bound, got {over}")

    # The four constructed rows above are the check that discriminates. This half
    # asks whether the *real* rows agree with the `flagged` they stored at write
    # time, which is only meaningful where the file is. It cannot silently be a
    # no-op: the assert below is satisfied by an empty file only when the file
    # genuinely isn't there, and its message says which happened.
    series = Path.home() / "lloyd" / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    if not series.exists():
        assert not (Path.home() / "lloyd" / "usage.db").exists(), (
            f"{series} is missing but the database to grade it is here — a recorder "
            "that ran against a live db wrote no row")
        print(f"note: no live series at {series}; checked the consumer against "
              f"{len(rows)} constructed rows only")
    for row in _rows_of(series) if series.exists() else []:
        assert iv_metrics_record.over_bound(row) is bool(row["flagged"]), (
            f"row since={row['since']} stored flagged={row['flagged']} but the "
            f"consumer says otherwise — the nightly verdict and the reader would "
            "disagree about the same row")
