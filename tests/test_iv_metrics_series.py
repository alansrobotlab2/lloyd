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
import inspect
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from tests._live_data import require_live_data

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
GRADER = SCRIPTS / "iv_grade.py"
RECORDER = SCRIPTS / "iv_metrics_record.py"
#: The one fan-out a breach announcement goes through (#1145 clause 2). Copied into the
#: fixture repo below, because the recorder resolves it relative to its own `__file__`
#: and a test that cannot reach it would silently test the no-notify fallback.
GUARDIAN = ROOT / "agent-services" / "guardian"
AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
#: The task file clause 1 requires. Globbed, not hardcoded, so renaming it does not
#: silently un-pin every test below.
TASK_GLOB = "86-*iv*metrics*.md"

#: The two commands the LIVE guardian fan-out shells out to for the channels someone in the
#: room can experience: `_journal` runs `systemd-cat`, `_desktop` runs `notify-send`
#: (`agent-services/guardian/notify.py:203-217`). One tuple because "no test reaches the
#: machine" is ONE property of this suite. It was not treated as one on 2026-09-20: the
#: clause-5 guard checked only the mutes in the #1145 fixture's own environment, while
#: `_make_repo` — the fixture every *other* test uses — copied the live `notify.py`, and the
#: breaching fixture rows fanned out for real (20 lines into the host journal from 32 PIDs).
#: See `test_the_announcement_tests_reach_neither_systemd_cat_nor_notify_send`.
LIVE_CHANNEL_BINARIES = ("systemd-cat", "notify-send")

#: The variables `_channel_on` (`notify.py:47-65`) reads as the master mute for those two
#: channels. Named once so every fixture environment and the guard test agree on what
#: "closing the door" means.
ROOM_MUTE_VARS = ("LLOYD_JOURNAL_ALERTS", "LLOYD_DESKTOP_ALERTS")


def _room_mutes() -> dict:
    """The variables that keep a fixture's fan-out out of the room.

    `ROOM_MUTE_VARS` are `_channel_on`'s master mutes, read at dispatch time; an empty
    `DBUS_SESSION_BUS_ADDRESS` makes `_desktop` decline at its own pre-flight
    (`notify.py:214`) before it can look for `notify-send`. Every environment this suite
    hands a subprocess includes them, generic `_env` included — the clause-5 guard test
    proves against the LIVE fan-out that these are the variables that close it, and that
    with them unset it is open.
    """
    mutes = {name: "0" for name in ROOM_MUTE_VARS}
    mutes["DBUS_SESSION_BUS_ADDRESS"] = ""
    return mutes


#: The receipt the substituted fan-out appends to. One line per call, so a count of
#: announcements is a count of calls across the process boundary, at the one boundary the
#: recorder uses.
RECEIPT = "IV_TEST_FANOUT_RECEIPT"

#: Written to `agent-services/guardian/notify.py` of EVERY fixture repo this suite builds:
#: `_make_repo` is the only writer and `_fanout_check` builds on it, so the suite has one
#: fan-out and one file that could reach the room. Same constructor keywords and the same
#: `announce(title, body, level=...)` as the real class — pinned against the real signature
#: by `test_the_substitute_fan_out_is_the_real_ones_shape` — except that it records each call
#: to the file named by `$IV_TEST_FANOUT_RECEIPT` instead of putting anything in the room.
#:
#: It is deliberately NOT a copy of `notify.py`. The real `Notifier.announce` writes no file
#: at all — its two room channels ARE `systemd-cat` and `notify-send` — so there is nothing
#: to count, and a fixture that copied the live file was a fixture that alerted the machine
#: about a breach that never happened. `notify.py:56-62` records that accident reaching a gate
#: run on 2026-09-07; this suite repeated it on 2026-09-20 through `_make_repo`.
RECORDING_FANOUT = '''"""Substitute guardian fan-out: records, never notifies."""
import json
import os


class Notifier:
    def __init__(self, *, ledger, state_dir, vault_root, backend_url=None,
                 external=True, voice=True, voice_window=3600.0):
        self.ledger, self.state_dir, self.vault_root = ledger, state_dir, vault_root
        self.backend_url, self.external, self.voice = backend_url, external, voice
        self.voice_window = voice_window

    def announce(self, title, body="", level="info"):
        path = os.environ.get("%s")
        if path:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"title": title, "body": body, "level": level}) + "\\n")
        return {"journal": True, "desktop": True, "voice": False}

    def alert(self, level, title, body="", **kwargs):
        raise AssertionError("a bound breach must never route through alert(): "
                             + title)
''' % RECEIPT

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
    # The announcement reaches the guardian through a path computed from the recorder's
    # own `__file__`, so a fixture repo with no `agent-services/guardian/notify.py` takes
    # the "no notify.py" fallback and every announcement test would pass while proving
    # nothing. The file is therefore there — but it is `RECORDING_FANOUT`, never a copy of
    # the live one. This is the seam that leaked: the generic fixture is used by tests that
    # breach the bound with no announce-specific environment, and the live fan-out is on by
    # default, so a copied `notify.py` alerted the machine for a condition that never
    # happened. `gstate.py` and `policy.py` stay the real files: `announce_breach` reads
    # `policy.AUTOMOD_STATE`, `policy.GUARDIAN_STATE` and `policy.VAULT_ROOT` by name to
    # build the Notifier, and a fixture without them fails the import, which is how a broken
    # fan-out gets reported rather than read as a quiet night.
    guardian = repo / "agent-services" / "guardian"
    guardian.mkdir(parents=True)
    (guardian / "notify.py").write_text(RECORDING_FANOUT, encoding="utf-8")
    for name in ("gstate.py", "policy.py"):
        shutil.copy2(GUARDIAN / name, guardian / name)
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


def _env(home: Path) -> dict:
    """A subprocess environment that cannot reach the machine.

    `HOME` alone would be enough for the state paths, which all resolve under it, but the
    room mutes are named explicitly too: #1145 made this the first suite here that runs
    code writing OUTSIDE the repo, and a breaching fixture row is a breach nobody observed.
    `_room_mutes` is the same set the #1145 announcing runs use — one definition, so a
    generic test can't be the one that leaks (that is exactly what happened on 2026-09-20).
    `IV_METRICS_ANNOUNCE` is deliberately NOT set here: the recorder's own mute, and a test
    that carried it would satisfy "the recorder announces" by sending nothing.
    """
    return {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LLOYD_GUARDIAN_STATE": str(home / ".local/state/lloyd-guardian"),
        "LLOYD_AUTOMOD_STATE": str(home / ".local/state/lloyd-automod"),
        **_room_mutes(),
    }


def _call(*, since: str, repo: Path, out: Path,
          extra: list[str] | None = None,
          env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the documented pipeline: `iv_grade.py --json --since X | iv_metrics_record.py`.

    Two real processes and a real pipe, launched by bash with `HOME` set to the fixture
    root so `cd ~/lloyd` resolves inside the fixture. The deliverable *is* a shell
    pipeline a model will paste, so the test pastes it too.
    """
    script = (f"cd ~/lloyd && python3 scripts/iv_grade.py --json --since '{since}'"
              f" | python3 scripts/iv_metrics_record.py --out '{out}'"
              + "".join(f" {e}" for e in (extra or [])))
    # `env` replaces the fixture environment wholesale. The #1145 tests pass their own
    # because which variables are present IS the clause under test there — see
    # `_announcing_env` against `_env`.
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          check=False, cwd=str(repo.parent),
                          env=env if env is not None else _env(repo.parent))


def _rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_the_default_out_is_the_path_the_nightly_writes(tmp_path):
    """The recorder with no `--out` lands in `<repo>/_pipeline/reflection/iv-metrics.jsonl`.

    The task file's Step 1 command passes no `--out`, so the default *is* the nightly's
    destination and the item's acceptance names that exact path. `REPO_ROOT` derives
    from the script's own file, not the cwd, so the fixture is a copy of the script in
    a fake repo — which is what pins the derivation rather than re-reading the constant.
    Run without touching the live tree: nothing here writes to `$HOME/lloyd`.
    """
    fake = tmp_path / "repo"
    (fake / "scripts").mkdir(parents=True)
    shutil.copy(RECORDER, fake / "scripts" / "iv_metrics_record.py")
    report = json.dumps({"meta": {}, "scope": {"since": "2026-09-12T00:00:00",
                                               "until": "2026-09-13T00:00:00",
                                               "first": None, "last": None},
                         "coverage": {"observations": 1, "turns": 1},
                         "cost": {"llm_calls": 10, "observations_per_llm_call": 0.1,
                                  "tokens": {"in": 1, "out": 1, "total": 2},
                                  "by_trigger": {}, "errors": {}},
                         "prompt_following": {"landed_rate": 1.0, "miss_rate": 0.0}})

    result = subprocess.run(["bash", "-c",
                             f"echo {shlex.quote(report)} | python3 scripts/iv_metrics_record.py"],
                            capture_output=True, text=True, check=False, cwd=str(fake))

    assert result.returncode == 0, (result.returncode, result.stderr)
    written = fake / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    assert written.exists(), (
        "no --out was passed and the default wrote nowhere near "
        f"{written}; stdout={result.stdout[:160]!r}")
    assert len(_rows_of(written)) == 1


def test_the_alert_reaches_the_autonomy_runner(tmp_path):
    """The seam the round exists on: a breach makes the RUN get flagged, not just exit 2.

    The nightly reaches the recorder through an agent's Bash tool, so the process's
    exit status is a tool result and the task's own exit code is the agent's turn — 0
    whether or not anything was wrong. The one automated consumer that exists is
    `autonomy._detect_silent_failures`, applied to the run's terminal block
    (`autonomy.py:2268-2270`), whose patterns include `exit code [1-9]`. So this runs the
    real pipeline three ways — breach / one-night-flagged / healthy — feeds each run's
    actual stdout to that production detector, and requires that only the breach trips
    it. A breach that only moved an exit code nobody reads would pass every other test
    in this file and leave the item's own "something reads it against a bound" false,
    which is why it is asserted here rather than documented.
    """
    sys.path.insert(0, str(ROOT))
    from autonomy import _detect_silent_failures

    since = _iso(NOW_LOCAL - datetime.timedelta(hours=26))
    # 10 observations each, one of them carrying an `error`, so dropped/llm_calls is
    # 1/10 = 0.1, over the 0.05 default bound; the breach case adds three prior rows
    # at 0.20 so the median of the window is over it too. `healthy` has no error row.
    scenarios = {
        "breach": ({"error": "timeout after 5.0s"}, [0.20, 0.20, 0.20]),
        "flagged": ({"error": "timeout after 5.0s"}, []),
        "healthy": ({}, []),
    }
    lines = {}
    for label, (error_row, prior_rates) in scenarios.items():
        repo = _make_repo(tmp_path / label)
        rows = [{"created_at": NOW_LOCAL - datetime.timedelta(hours=2), **error_row}]
        rows += [{"created_at": NOW_LOCAL - datetime.timedelta(hours=3)}] * 9
        _make_db(repo, rows)
        out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
        out.parent.mkdir(parents=True)
        for i, rate in enumerate(prior_rates):
            out.open("a").write(json.dumps({"since": f"p-{i}",
                                            "dropped_rate": rate}) + "\n")

        result = _call(since=since, repo=repo, out=out, extra=["--hours 26"])
        row = _rows_of(out)[-1]
        assert iv_metrics_record.over_bound(row) is bool(row["flagged"]), (
            label, row)
        lines[label] = result.stdout
        if label == "breach":
            assert result.returncode == 2 and row["breach"] is True, result.stdout
        else:
            assert result.returncode == 0, (label, result.stdout, result.stderr)
            assert row["breach"] is False

    assert "exit code 2" in lines["breach"], (
        "the BREACH verdict line must name its own exit code — that literal is the "
        f"only token the runner's prose regex can match; got {lines['breach'][:200]!r}")
    assert _detect_silent_failures(lines["breach"]), (
        "the autonomy runner's own detector does not trip on a BREACH verdict line, "
        "so the alert has no automated surface at all")

    for label in ("flagged", "healthy"):
        assert "exit code" not in lines[label], (label, lines[label][:200])
        assert _detect_silent_failures(lines[label]) == [], (
            f"a {label} night must not trip the failure detector — a nightly that "
            f"cries wolf is how alerts get muted: {lines[label][:160]!r}")
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

    # `RULED_BOUND`, not a retired number: 353/9027 is the measured baseline, and the
    # assert on `flagged` below only means something while the baseline is compared
    # against the bound that is actually in force.
    row = iv_metrics_record._row(report, window_hours=26.0, threshold=RULED_BOUND,
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
    # Against the stripped `code`, not the raw file: the docstring above discusses
    # `--out` in prose, so a raw read would still pass with the flag deleted.
    assert '"--out"' in code, (
        "the destination must be a flag with a default, not a hardcoded write path")
    assert "add_argument(\"--out\"" in code or "add_argument('--out'" in code, (
        "--out must be the argparse destination flag")


# ── clause 5: a stated threshold, compared over fields ────────────────────────


def test_breach_exits_2_when_the_recent_median_is_over_the_bound(tmp_path):
    """A sustained breach is an exit code, computed from JSONL fields.

    Clause 5: "flag the row when the last N rows exceed it, with the comparison
    expressed over JSONL fields rather than prose". Four prior rows at 0.20 plus
    this run's 1.0: median 0.20 > the 0.05 bound, so exit 2. Exit status rather than a
    sentence because the autonomy runner consumes exit codes reliably —
    `scripts/skill_verdicts.py:633` and `scripts/validate_handoff.py:70` use 2 for a
    refusal for the same reason, and `_detect_silent_failures` (`autonomy.py:27-33`) only catches
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
    resolvable `skill_name` are checked because `autonomy.py:1145-1157` skips a task with no
    `skill_name` forever, and `RUNNABLE_STATUSES` (`autonomy.py:542`) never dispatches a status
    outside up_next/in_progress/failed — a task file that parses but never
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

    This is the seam #460 would otherwise miss. `autonomy.py:1333-1353` builds a run's
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
        env=_env(tmp_path))

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
    while the exit code follows the code. That is exactly what happened on
    2026-09-20: vault `1eb0735e` put 0.05 into the task file and the constant stayed
    0.10, so the nightly read one number and enforced another. The measured baseline
    (0.039) is also required, because that is what makes the bound legible as a
    margin over what is actually measured rather than an arbitrary number.

    This node pins the two copies *agreeing*; it cannot pin the magnitude — moving
    both together stays green here, which is why `test_the_code_default_is_the_ruled_bound`
    asserts the value itself. The channel a breach reaches is still an open
    person-decision (#1145).
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
    # string and the body's own bound line was never located. Anchored on the
    # word `Bound:` at the start of a line so a stray "bound" in prose can't stand in.
    for label, chunk in (("description", desc), ("body", body)):
        stated = (re.search(r"[Bb]ound\D{0,24}?(\d\.\d+)", chunk) if label == "description"
                  else re.search(r"Bound:\s*(\d\.\d+)", chunk))
        assert stated, (f'the {label} must state the bound, e.g. "Bound: '
                        f'{iv_metrics_record.DEFAULT_THRESHOLD}" — '
                        f"searched {len(chunk)} chars of the {label}")
        assert float(stated.group(1)) == pytest.approx(
            iv_metrics_record.DEFAULT_THRESHOLD), (
            f"{label} says {stated.group(1)}, DEFAULT_THRESHOLD is "
            f"{iv_metrics_record.DEFAULT_THRESHOLD}")
        assert float(stated.group(1)) == pytest.approx(0.05), (
            f"the {label} states {stated.group(1)}; the bound Alan ruled for #1145 is "
            "0.05 — 0.10 is the provisional value that never fired on the degraded "
            "09-05..09-11 median of ~0.053")
    assert iv_metrics_record.DEFAULT_THRESHOLD == pytest.approx(0.05), (
        "DEFAULT_THRESHOLD must be the ruled bound, 0.05")
    assert "0.039" in text, "the measured baseline belongs next to the bound"


#: The bound Alan ruled on #460. Named here rather than inlined so a failure names
#: the ruling that moved, not a number.
RULED_BOUND = 0.05
#: Per-local-day dropped-verdict rates measured off `usage.db` (2026-09-20), as the
#: ruling saw them: the healthy weeks sit at the first, the degraded week's 7-day
#: median at the second. 0.05 between them is what makes it an alert and not a
#: re-statement of the weather.
HEALTHY_WEEK_RATE = 0.014
DEGRADED_STRETCH_MEDIAN = 0.0526


def _constant_comment(name: str) -> str:
    """The `#:` block directly above a module-level constant, read off the source.

    The reason a bound was chosen lives in that comment and nowhere else — not in a
    docstring `--help` prints, not in the task file the agent reads — so a reader
    editing the number meets the reason or meets nothing. Returning the empty string
    when there is no block is deliberate: callers assert on it, so a constant that
    loses its comment is a failure and not a silent pass.
    """
    lines = RECORDER.read_text(encoding="utf-8").splitlines()
    anchor = next((i for i, ln in enumerate(lines)
                   if ln.startswith(f"{name} =")), None)
    assert anchor is not None, f"{name} is no longer a module-level constant"
    start = anchor
    while start > 0 and lines[start - 1].lstrip().startswith("#:"):
        start -= 1
    return "\n".join(lines[start:anchor])


def test_the_code_default_is_the_ruled_bound():
    """The magnitude is pinned, and so is the reason written beside it.

    `test_task_file_threshold_matches_the_code_default` can only say the prose and the
    code agree; move both to 0.20 and it stays green, which lets an un-ruling arrive
    dressed as a retune. So this asserts the ruled value itself — 0.05 — and then
    asserts the constant's own comment still carries what justifies it: Alan's ruling
    (#460), the measured baseline 0.039 over 2026-09-01..09-12, the degraded
    2026-09-05..09-11 stretch whose 7-day median (0.0526) sat under the retired 0.10
    and never alerted, and the healthy-week rate the bound must stay above.

    The arithmetic of the ruling is asserted too, because it is the whole reason the
    value is 0.05: healthy below it, the degraded median above it. A retune that
    breaks that ordering is a different guard, not a retuned one.
    """
    assert iv_metrics_record.DEFAULT_THRESHOLD == pytest.approx(RULED_BOUND), (
        f"DEFAULT_THRESHOLD is {iv_metrics_record.DEFAULT_THRESHOLD}, not the "
        f"{RULED_BOUND} Alan ruled on #460 — re-tuning it is a person-decision, and "
        "the task file in ~/obsidian/autonomy has to move in the same change")
    assert HEALTHY_WEEK_RATE < RULED_BOUND < DEGRADED_STRETCH_MEDIAN, (
        "the bound only means 'fires on a run of bad nights, quiet otherwise' while "
        "it sits between the healthy-week rate and the degraded stretch's median")

    comment = _constant_comment("DEFAULT_THRESHOLD")
    assert comment, "the constant lost its comment; the reason for a bound has to " \
                    "travel with the number"
    for token, why in (
            ("#460", "whose ruling it is"),
            ("0.039", "the measured baseline the bound is a margin over"),
            ("2026-09-01..09-12", "the window that baseline was measured over"),
            ("2026-09-05..09-11", "the degraded stretch the bound exists to catch"),
            ("0.0526", "that stretch's 7-row median, which 0.10 let pass"),
            ("0.10", "the retired bound, so the next reader knows it was retired")):
        assert token in comment, f"the comment beside the constant must state {why}"
    # The retired number must not come back as a live claim about today's traffic:
    # at 0.05 the "about 2.5x what we measure now" sentence is false (0.05 is ~1.3x
    # the 0.039 baseline), and a reader who believes it under-alarms.
    assert "2.5x" not in comment, (
        "the comment still scales the bound off the retired 0.10; 0.05 is ~1.3x the "
        "0.039 baseline, so the multiplier has to go or be re-measured")


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
    require_live_data(series, "the inner-voice metrics series", kind="file")

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
    code — over the bound, under it, exactly on it, and one with no calls — and
    require exact membership. Then check the live rows agree with their own stored
    `flagged` field, so the shipped nightly decision and the consumer are the same
    function rather than two readings of the same number.
    """
    # Every row below is 1,000 LLM calls, so the counts are the rates times a
    # thousand and each one is a real night rather than an arbitrary number: the
    # degraded 2026-09-05..09-11 median (0.0526 -> 53 calls), a healthy week
    # (0.014 -> 14), and the bound itself (0.05 -> 50). Against RULED_BOUND the bound
    # on 1,000 calls is 50 calls, so 53 is over and 50 is exactly on it — the three
    # answers a reader can compute without running anything. Retuning the bound moves
    # RULED_BOUND and these rows together; leaving the bound at 0.05 and editing a
    # `flagged` expectation here is the failure mode this file exists to catch.
    rows = [
        # the degraded stretch's median: 53 of 1000 -> 53 > 50
        {"llm_calls": 1000, "threshold": RULED_BOUND,
         "dropped_verdicts": round(DEGRADED_STRETCH_MEDIAN * 1000), "flagged": True},
        # a healthy week: 14 of 1000 -> 14 < 50
        {"llm_calls": 1000, "threshold": RULED_BOUND,
         "dropped_verdicts": round(HEALTHY_WEEK_RATE * 1000), "flagged": False},
        # exactly on the bound: 50 of 1000 -> 50 > 50 is False
        {"llm_calls": 1000, "threshold": RULED_BOUND,
         "dropped_verdicts": round(RULED_BOUND * 1000), "flagged": False},
        # no traffic: no rate, so nothing to breach
        {"llm_calls": 0, "threshold": RULED_BOUND,
         "dropped_verdicts": 0, "flagged": False},
    ]
    for row in rows:
        assert iv_metrics_record.over_bound(row) is row["flagged"], row

    over = [r for r in rows if iv_metrics_record.over_bound(r)]
    assert [r["dropped_verdicts"] for r in over] == [53], (
        f"exactly one of the four rows is over the bound, got {over}")

    # The four constructed rows above are the check that discriminates. This half
    # asks whether the *real* rows agree with the `flagged` they stored at write
    # time, which is only meaningful where the file is. It cannot silently be a
    # no-op: the assert below is satisfied by an empty file only when the file
    # genuinely isn't there, and its message says which happened.
    series = Path.home() / "lloyd" / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    if not series.exists():
        # This used to infer "a recorder ran against a live db and wrote no row"
        # from usage.db being present while the series is not. That inference is
        # sound only where the series was never written; on 2026-09-22 the tree was
        # deleted and _pipeline went with it while usage.db (tracked-adjacent, at
        # the repo root) stayed, so the premise holds and the conclusion is false.
        # The four constructed rows above already ran and are the discriminating
        # half; the live half is unanswerable, by name rather than by silence.
        require_live_data(series, "the inner-voice metrics series", kind="file")
    for row in _rows_of(series) if series.exists() else []:
        assert iv_metrics_record.over_bound(row) is bool(row["flagged"]), (
            f"row since={row['since']} stored flagged={row['flagged']} but the "
            f"consumer says otherwise — the nightly verdict and the reader would "
            "disagree about the same row")

# ─────────────────────────────────────────────────────────────────────────────
# #1145 — the bound is 0.05, and a breach announces itself from CODE.
#
# Clauses 2, 3 and 4 are about a process boundary: the recorder process sends the
# message, a human with a toast and a journal line reads it. So every test below drives
# the recorder as a real subprocess of the documented pipeline, and the ONE thing
# substituted is the fan-out module the recorder imports by path — `RECORDING_FANOUT`,
# which every fixture repo in this file gets from `_make_repo`.
#
# Why a substitute and not the live `Notifier`: `Notifier.announce` writes no file. Its
# journal channel shells out to `systemd-cat` (`notify.py:203-208`) and `_desktop` needs a
# session bus (`:210-217`), so there is no artifact to count — and a test that ran the real
# one would be posting to the machine's journal for a condition that never happened. The
# 2026-09-07 note at `notify.py:56-62` records exactly that accident on a gate run, in the
# live journal, labelled critical; this suite repeated it on 2026-09-20 because the generic
# fixture copied the live `notify.py` and its breaching rows alerted for real. So the
# substitute writes a receipt file, and
# `test_the_substitute_fan_out_is_the_real_ones_shape` pins it against the real signature
# so the substitution cannot drift. `IV_METRICS_ANNOUNCE` stays UNSET in the clause-2 and
# clause-3 tests: a mute cannot prove an alert fires by proving nothing was sent.
#
# The bound (clause 1) is coupled to the vault by the vault-reading
# `test_task_file_threshold_matches_the_code_default`
# above, which reads `~/obsidian/autonomy/86-*.md` and pins the front-matter copy, the body
# copy and `DEFAULT_THRESHOLD` as ONE value. Code-only or vault-only turns that test red —
# it is what killed round SM_20260916_093433, whose vault commit alone took the live suite
# from green to red. Both halves are in this round.
# ─────────────────────────────────────────────────────────────────────────────

#: `_announced` reads what that recording wrote. `RECORDING_FANOUT` and `RECEIPT` — the
#: substitute and the variable naming its receipt — are defined with the other fixture
#: constants at the top of the file, because `_make_repo` writes them for every test here,
#: not only for the ones below.


def _announced(out_dir) -> list[dict]:
    """The substituted fan-out's receipt: one dict per announce() CALL.

    Reads the process's receipt file, so it counts calls that crossed the process boundary
    and nothing else. Absent file means zero calls, which is the honest reading of an empty
    result in every test here: the pipeline's own exit code is asserted alongside, so a
    receipt-less run is either a quiet night or a stated failure, never an invisible one.
    """
    receipt = Path(out_dir) / "fanout.jsonl"
    if not receipt.exists():
        return []
    return [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _fanout_check(tmp_path: Path, *, raising: bool = False) -> Path:
    """A fixture checkout built by `_make_repo`, optionally with a fan-out that raises.

    It has to be `_make_repo` and not a parallel builder: the announcement resolves the
    fan-out as `<the recorder's own repo>/agent-services/guardian`, so whoever writes that
    file decides whether a test reaches the room — which is why the substitute is installed
    there rather than here, so no fixture can opt out of it. Building a second fixture for
    the announcing tests anyway is how two answers to that question existed at once on
    2026-09-20, the #1145 one recording and the generic one posting, and the generic one won:
    23 breach lines from 23 distinct PIDs landed in the host journal between 03:17 and 04:48
    (`journalctl --no-pager --since 2026-09-19 | grep -c dropped-verdict`), each one a
    warning toast about a breach that had not happened.

    `raising=True` injects a failing fan-out: `announce` raises on entry, which is clause 4's
    fault. Injected as text rather than by wrapping, so the recorder still imports a module
    named `notify` with the same class and signature — the only thing that changes is that
    the call blows up, as a dead session bus or an unreadable state dir would.
    """
    repo = _make_repo(tmp_path)
    notify = repo / "agent-services" / "guardian" / "notify.py"
    if raising:
        notify.write_text(RECORDING_FANOUT.replace(
            "    def announce(self, title, body=\"\", level=\"info\"):\n",
            "    def announce(self, title, body=\"\", level=\"info\"):\n"
            "        raise RuntimeError('session bus unreachable')\n"),
            encoding="utf-8")
    return repo


def _announcing_env(home: Path, repo: Path) -> dict:
    """Environment for an announcing run: everything off, and the receipt named.

    Set by construction, never inherited, because which variables exist IS clauses 3 and 5:
    - `IV_METRICS_ANNOUNCE` is absent. The recorder's own mute is production-dead, and a
      test that set it would satisfy "the recorder announces" by sending nothing.
    - Clause 5's mutes, from `_room_mutes()`: `LLOYD_JOURNAL_ALERTS`/`LLOYD_DESKTOP_ALERTS`
      are `0` and `DBUS_SESSION_BUS_ADDRESS` is empty. A real defence, not decoration: the
      substitute never constructs a real `Notifier`, so if the module substitution ever
      stopped working these are the variables standing between the suite and the machine's
      journal — and `test_the_mutes_close_the_live_fan_out_and_every_fixture_sets_them`
      proves they close it by running the REAL fan-out twice over decoy channel binaries,
      rather than merely being set.
    - Both state dirs point inside the fixture. `policy.py:166-171` reads each with
      `environ.get(name, default)` and no `or` fallback, so the empty string resolves to
      `Path("")` — the current working directory — and a real `Notifier` built on it would
      append `ledger.jsonl` and `ALERT.md` wherever it happened to be standing.
    - `HOME` points at the fixture, so the pipeline's own `cd ~/lloyd` lands in the fixture
      checkout and grades the fixture's `usage.db`, never the live one.
    """
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "PYTHONPATH": str(repo),
        "PYTHONDONTWRITEBYTECODE": "1",
        RECEIPT: str(home / "fanout.jsonl"),
        **_room_mutes(),
        "LLOYD_AUTOMOD_STATE": str(home / ".local/state/lloyd-automod"),
        "LLOYD_GUARDIAN_STATE": str(home / ".local/state/lloyd-guardian"),
    }


def _night(home: Path, *, rate: float, since: str, out: Path, repo: Path,
           extra: list[str] | None = None,
           env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """One nightly through the documented pipeline, for a day whose rate is `rate`.

    `rate` is applied as whole percentage points of 100 observer calls, so no assertion here
    is ever within a rounding tolerance of being wrong. `_call` runs `cd ~/lloyd &&
    iv_grade.py | iv_metrics_record.py` under `HOME=home`, which is the checkout the
    fixture built.
    """
    n_dropped = round(rate * 100)
    stamp = datetime.datetime.fromisoformat(since)
    blank = {"action": "noop", "created_at": stamp}
    dropped = dict(blank, error="observer refused")
    rows = [dict(dropped)] * n_dropped + [dict(blank)] * (100 - n_dropped)
    (repo / "usage.db").unlink(missing_ok=True)   # one night's db, never a growing one
    _make_db(repo, rows)
    env = _announcing_env(home, repo)
    env.update(env_extra or {})
    return _call(since=since, repo=repo, out=out, extra=extra, env=env)


def _nights(home: Path, *, rates: list, out: Path, repo: Path,
            extra: list[str] | None = None, env_extra: dict | None = None):
    """Several nightlies in sequence; returns (per-night counts, exit codes, rows, last run).

    The series file is never seeded. Each night's `dropped_rate` and `breach` come from the
    recorder's own previous output, so the file is evidence rather than an input and the
    assertion below is reading a chain, not a fixture.
    """
    per_night, codes, stored = [], [], []
    for index, rate in enumerate(rates):
        before = len(_announced(home))
        result = _night(home, rate=rate, since=f"2026-09-{15 + index:02d}T03:00:00",
                        out=out, repo=repo, extra=extra, env_extra=env_extra)
        assert result.returncode in (0, 2), (
            f"night {index + 1} exited {result.returncode}, not 0 or 2 — the pipeline "
            f"itself failed and every assertion downstream would be reading a run that "
            f"never recorded anything\nstdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}")
        per_night.append(len(_announced(home)) - before)
        codes.append(result.returncode)
        stored = _rows_of(out)
        last_run = result
    # The last run is returned because "the failure is legible in the run record" is a claim
    # about the run the chain just made, and a test that manufactured a seventh night to read
    # a line off would be asserting about a night the scenario never had.
    return per_night, codes, stored, last_run


def _real_notify_module():
    """Import the live guardian fan-out by the same path route the recorder uses.

    `announce_breach` loads `agent-services/guardian/notify.py` by file location because the
    guardian sits outside the package and is not importable by name. Two tests need the real
    module — one to pin this section's substitute against it, one to prove `announce` has no
    recording channel — and both must reach it the way production does rather than by a bare
    `import notify` that would silently pick up anything else on the path. `sys.path` is
    restored so the guardian's own module names (`policy`, `gstate`) stay unshadowed for the
    rest of the suite.
    """
    import importlib
    import sys as _sys

    gdir = str(ROOT / "agent-services" / "guardian")
    added = gdir not in _sys.path
    if added:
        _sys.path.insert(0, gdir)
    try:
        return importlib.import_module("notify")
    finally:
        if added:
            _sys.path.remove(gdir)


def test_the_default_bound_is_005_and_the_recorder_uses_it():
    """Clause 1: the constant, and why a bare number is not enough.

    `test_task_file_threshold_matches_the_code_default` above already reads
    `~/obsidian/autonomy/86-*.md` and fails if either copy of the bound disagrees with
    `DEFAULT_THRESHOLD`, so this test does not re-parse the markdown. It pins what that test
    cannot see: the provenance beside the constant. The number without the measurement is a
    constant nobody can re-litigate, and this file's own history is the argument — #460
    shipped 0.10 as a first guess and it never fired on a single recorded night, including
    the week the observer was measurably degraded.
    """
    assert iv_metrics_record.DEFAULT_THRESHOLD == 0.05
    assert iv_metrics_record.DEFAULT_WINDOW_ROWS == 7
    assert iv_metrics_record.MIN_BREACH_ROWS == 3, (
        "the floor is what makes the tighter bound safe; if it moved, the ruling this item "
        "records (median of 7, floor 3) would no longer describe the code")
    body = (ROOT / "scripts" / "iv_metrics_record.py").read_text(encoding="utf-8")
    assert "0.039" in body, (
        "the measured baseline the bound was set against has to live next to the constant")
    assert "0.0526" in body, (
        "so does the 7-row median of the degraded stretch the bound was set AT — the exact "
        "figure, the same one `test_the_code_default_is_the_ruled_bound` requires of that "
        "comment, or a reader cannot tell 0.05 from a second guess")


def test_six_nights_of_one_incident_announce_exactly_once_end_to_end(tmp_path):
    """Clauses 2 and 3, as one unbroken chain of real runs from an empty series file.

    Every other test here declares the previous night's verdict and shows the recorder
    honouring it. That is a real boundary only if the declaration the recorder reads is the
    one IT wrote the night before, so this chain seeds nothing: six nights of
    0.06/0.06/0.06/0.00/0.00/0.00 run through the real shell pipeline, and the series file
    grows only from the recorder's own output. The measured count is (0, 0, 1, 0, 0, 0).

    Each position rules something out alone. Nights 1-2 are the floor — a breach needs three
    numeric rows and there are one and two. Night 3 is the message. Nights 4-5 are
    one-per-breach: the median is still 0.06 and each row is still a breach, so a recorder
    that announces every breaching night answers (0,0,1,1,1,0) here. Night 6 is the rule the
    other way: the median of [0.06,0.06,0.0,0.0,0.0] is 0.0, the breach ENDS, and a recorder
    that latched "already announced" in a state file would stay silent through the next
    incident too. The stored verdicts read (False,False,True,True,True,False) — the
    announcement count and the file disagree in exactly one place, the nights a breach
    continues — and the exit codes (0,0,2,2,2,0) follow the median, not the announcement.
    """
    home = tmp_path
    repo = _fanout_check(tmp_path)
    out = home / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    per_night, codes, stored, last_run = _nights(home, rates=[0.06] * 3 + [0.0] * 3, out=out,
                                       repo=repo)
    assert per_night == [0, 0, 1, 0, 0, 0], (
        f"announcements per night were {per_night}, expected [0, 0, 1, 0, 0, 0] — one "
        "message for six nights of one incident, on the night it starts")
    assert codes == [0, 0, 2, 2, 2, 0], (
        f"exit codes were {codes}, expected [0, 0, 2, 2, 2, 0] — the exit code tracks the "
        "median and must not move with the announcement")
    assert [bool(r.get("breach")) for r in stored] == [
        False, False, True, True, True, False], (
        f"stored verdicts were {[r.get('breach') for r in stored]}")
    assert len(_announced(home)) == 1
    assert _announced(home)[0]["level"] == "warning", (
        "level=warning is the ruling; 'info' would be indistinguishable from the promotion "
        "notice and 'error' is the road to alert()")


def test_the_announcement_quotes_the_basis_the_verdict_medianed(tmp_path):
    """One decision, one row count: the toast and the verdict line report the same basis.

    A short series is the normal state, not an edge case — the live file held 6 rows when
    the bound tightened to 0.05 — and it is the only place the two texts can disagree,
    because `window_rows` records what was *asked for* (7) while `breach_basis_rows` records
    what was *medianed* (3). The toast printed the former beside a verdict line printing the
    latter, which is two copies of one number wearing a sentence: someone reading "median of
    7 rows" joins the alert to a decision the recorder never made. Asserted off one real
    run's two outputs rather than off a hand-built row, so `_verdict` and the message cannot
    drift apart without this going red.
    """
    home = tmp_path
    repo = _fanout_check(tmp_path)
    out = home / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    per_night, codes, stored, last_run = _nights(home, rates=[0.06] * 3, out=out, repo=repo)
    assert per_night == [0, 0, 1] and codes == [0, 0, 2], (per_night, codes)
    row = stored[-1]
    assert row["window_rows"] == 7 and row["breach_basis_rows"] == 3, (
        "this scenario only tests anything while the requested window and the medianed "
        f"basis differ; got window {row['window_rows']}, basis {row['breach_basis_rows']}")

    announced = _announced(home)
    assert len(announced) == 1, announced
    counts = re.findall(r"median of (\d+) row", last_run.stdout + announced[0]["body"])
    assert counts == ["3", "3"], (
        f"the verdict line and the toast reported bases {counts}, expected both '3' — the "
        f"verdict said {last_run.stdout.strip().splitlines()[-1]!r} and the toast said "
        f"{announced[0]['body']!r}")


def test_a_starting_breach_announces_once_and_a_continuing_one_stays_quiet(tmp_path):
    """Clause 2 read two ways at once: the file's own verdict, and the fan-out's count.

    The chain in the test above already proves the sequence end to end. This one checks the
    other half — that the STORED row and the announced message are the same decision, made
    the same night — because clause 2's "exactly one" is a statement about a row, and a
    reader of `_pipeline/reflection/iv-metrics.jsonl` must be able to reconstruct it. So the
    row that started the breach carries the verdict the fan-out was called with, and the row
    that merely continued it carries the verdict the fan-out never heard about.

    The rates are one spike (0.11) on three clean nights — the shape the tighter bound must
    NOT alert on — then four nights at 0.06. The spike row is `flagged` and is not a
    `breach`: one bad night is a report, a run of them is a fault. And the arithmetic is the
    point of the test, because the median of seven is slow to turn: with
    [0, 0, 0, 0.11, 0.06, 0.06] on the table the median is 0.03, still under the bound, so
    the breach starts on the SEVENTH night, not the fifth.
    """
    home = tmp_path
    repo = _fanout_check(tmp_path)
    out = home / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    per_night, codes, stored, last_run = _nights(
        home, rates=[0.0, 0.0, 0.0, 0.11, 0.06, 0.06, 0.06], out=out, repo=repo)
    assert per_night == [0, 0, 0, 0, 0, 0, 1], (
        f"announcements were {per_night}, expected the spike and the two first worsening "
        "nights to be silent and the message to land on the seventh run")
    flagged = [bool(r.get("flagged")) for r in stored]
    breach = [bool(r.get("breach")) for r in stored]
    assert flagged == [False, False, False, True, True, True, True], (
        f"per-row `flagged` was {flagged}; each bad night is flagged on its own rate")
    assert breach == [False, False, False, False, False, False, True], (
        f"`breach` was {breach}: the median of [0,0,0,0.11,0.06,0.06] is 0.03, under the "
        "bound, so the breach starts on the SEVENTH run where the window holds three 0.06s "
        "— and the announcement count has to agree with the file, not with a reader's guess")
    assert codes == [0, 0, 0, 0, 0, 0, 2], f"exit codes were {codes}"


def test_the_announcement_comes_from_the_recorder_with_no_model_in_the_loop(tmp_path):
    """Clause 3: no model, no prose, no `_detect_silent_failures` — and a person is told.

    The pre-existing half of this suite is `test_the_alert_reaches_the_autonomy_runner`,
    which feeds the recorder's verdict lines to `autonomy._detect_silent_failures` and
    proves the regex finds "exit code 2" in them. That is the channel this item exists to
    replace: the regex runs over the run's terminal block, so a model that
    paraphrases the Bash result loses the breach, and its only effect is a section in the
    run record. Here there is no model to paraphrase anything — one subprocess, stdin from
    a pipe, stdout to a file — and the fan-out still receives the message.

    Also asserted: the same verdict text that the regex looks for is still printed, so the
    new channel is additive and the run record is unchanged.
    """
    home = tmp_path
    repo = _fanout_check(tmp_path)
    out = home / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    per_night, codes, stored, last_run = _nights(home, rates=[0.06, 0.06, 0.06], out=out, repo=repo)
    assert codes == [0, 0, 2], f"exit codes were {codes}"
    assert per_night == [0, 0, 1], f"announcements were {per_night}"
    calls = _announced(home)
    assert len(calls) == 1, f"the fan-out saw {len(calls)} calls, expected exactly 1"
    assert stored[-1]["breach"] is True, (
        "the row itself carries the verdict, so a reader of the file can reconstruct the "
        "night without the run's prose")


def test_a_failing_fan_out_costs_neither_the_row_nor_the_exit_code(tmp_path):
    """Clause 4: the bell breaking must not become the reason nobody was told.

    A fan-out that raises — no guardian checkout, a permission-denied state dir, a dead
    session bus — sits between "the row is appended", "the exit code is 2" and "a person
    reads a toast". The first two are the recorder's contract and the third is what this
    item is FOR, so a broken announce has to degrade to the loud, visible failure it can
    still deliver (stderr + exit 2 + a stored breach) and never to a missing row or a clean
    exit. The failure is also printed, because a silently-swallowed announce exception is
    the same silence this item is fixing, one layer down.
    """
    home = tmp_path
    repo = _fanout_check(tmp_path, raising=True)
    out = home / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    per_night, codes, stored, last_run = _nights(home, rates=[0.06, 0.06, 0.06], out=out, repo=repo)
    assert codes == [0, 0, 2], (
        f"a raising fan-out changed the exit codes to {codes}")
    assert per_night == [0, 0, 0], (
        "the receipt is empty by construction here — this is a failed announcement, not a "
        "quiet one")
    assert len(stored) == 3, (
        f"the series holds {len(stored)} rows, expected 3: the row must survive its own "
        "broken alert")
    assert bool(stored[-1].get("breach")) is True, (
        "and it must still carry the verdict, or the next night cannot tell it was inside "
        "a breach")
    assert "breach announcement FAILED" in last_run.stderr, (
        "the failure has to be legible verbatim in the run record — that line is the only "
        f"evidence a reader gets that the room got nothing\nstderr: {last_run.stderr!r}")
    assert "exit code 2" in last_run.stdout and "exit code 2" in last_run.stderr, (
        "the verdict stays on both streams, since the exit code is invisible to the "
        "nightly and stdout is invisible to a shell that only shows stderr on failure")


def test_the_substitute_fan_out_is_the_real_ones_shape():
    """Why the receipt file is admissible evidence at all.

    Every clause-2/3/4 count in this file is taken from `RECORDING_FANOUT`, so its signature
    has to be the real one or the counts are counting a fiction. `announce_breach` constructs
    `Notifier(ledger=..., state_dir=..., vault_root=..., voice=False, voice_window=...)` and
    calls `announce(title, body, level="warning")`; the real `announce` returns a dict of
    per-channel booleans, which the recorder reports channel-by-channel. The live class is
    imported here by the same path route the recorder uses, so this pins the substitute
    against the thing it replaces rather than against this file's memory of it.

    Executed from the module source rather than read as text, and proven to be the file the
    fixture actually has: `Path` cannot hold a module, so the shape check and the checkout
    are forced to agree on one string.
    """
    notify_mod = _real_notify_module()
    real_ctor = inspect.signature(notify_mod.Notifier.__init__)
    stub_ns: dict = {}
    exec(RECORDING_FANOUT.replace('os.environ.get("%s")' % RECEIPT, "None"), stub_ns)
    fixture = _make_repo(Path(tempfile.mkdtemp()))
    assert (fixture / "agent-services" / "guardian" / "notify.py").read_text(
        encoding="utf-8") == RECORDING_FANOUT, (
        "the fixture's fan-out is not the module whose shape is being checked here, so this "
        "test pins one file while the subprocesses run another")
    stub_ctor = inspect.signature(stub_ns["Notifier"].__init__)
    for name, param in real_ctor.parameters.items():
        if name == "self":
            continue
        assert name in stub_ctor.parameters, (
            f"the substitute fan-out cannot construct a Notifier the way the recorder does "
            f"missing {name}")
    real_ann = inspect.signature(notify_mod.Notifier.announce)
    stub_ann = inspect.signature(stub_ns["Notifier"].announce)
    assert [p.name for p in real_ann.parameters.values()] == [
        p.name for p in stub_ann.parameters.values()], (
        "the substitute's announce signature drifted from the real one; every count in "
        "this section is then counting a method the recorder never calls")


def test_a_breach_announcement_writes_no_ledger_row_no_alert_file_and_no_task(tmp_path):
    """Clause 2's other half: `announce`, and specifically never `alert`.

    `alert()` is the recording fan-out — it appends `ledger.jsonl`, writes `ALERT.md`, and
    above a threshold files a backlog task, so a five-night breach through `alert` files five
    tasks and five ledger rows for one condition (`notify.py:118-160`, backlog gate at
    `:157-159`). `announce()` has none of those channels: its body is `_journal`, `_desktop`
    and `_speak` (`notify.py:162-188`). A sustained bound breach is a condition, not an
    incident, which is the ruling this item exists to implement.

    Two halves, because one would be a tautology. (a) The structural one: the REAL
    `Notifier.announce` is read and proven to reach none of alert's recording methods, so
    the property is a fact about the guardian's code and not about this repo's stub. (b) The
    behavioural one: a breach run leaves nothing in the two state directories its own
    environment names — the paths `_announcing_env` actually passes to the subprocess, not
    side directories nothing points at. `Notifier._backlog_task` writes into
    `policy.VAULT_ROOT`, a literal path rather than an environment variable, so the vault is
    NOT redirectable and is deliberately not asserted on: the claim that no task is filed is
    carried by (a).
    """
    home = tmp_path
    repo = _fanout_check(tmp_path)
    out = home / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    per_night, codes, _stored, last_run = _nights(
        home, rates=[0.06, 0.06, 0.06], out=out, repo=repo)
    assert codes == [0, 0, 2] and per_night == [0, 0, 1], (
        "this fixture has to BE a starting breach for a claim about one to mean anything")

    import inspect
    recording = {"_alert_file", "_ledger", "_backlog_task", "_vault_note"}
    src = inspect.getsource(_real_notify_module().Notifier.announce)
    reached = sorted(name for name in recording if f"self.{name}(" in src)
    assert not reached, (
        f"announce() reaches alert()'s recording channels {reached}; it is supposed to be "
        "the fan-out with no bookkeeping")

    for name in ("ledger.jsonl", "ALERT.md"):
        for directory in (home / ".local/state/lloyd-guardian",
                          home / ".local/state/lloyd-automod"):
            assert not (directory / name).exists(), (
                f"{directory / name} exists after a breach announcement: alert() wrote it, "
                "or something is filing incidents for a recurring condition")


def test_the_announcing_env_mutes_the_room_but_never_the_recorders_own_channel(tmp_path):
    """Clause 5's variable discipline, plus the mute being loud rather than silent.

    Two things are checked and they pull in opposite directions, which is why they sit in one
    test.

    (a) The room channels are muted by the environment every announcing run gets:
    `LLOYD_JOURNAL_ALERTS=0` and `LLOYD_DESKTOP_ALERTS=0`, which `_channel_on`
    (`notify.py:47-65`) treats as off, plus no session bus so `_desktop`'s pre-flight
    declines before it looks for `notify-send`. That is defence in depth, not the primary
    defence: the fan-out a fixture run resolves is the substitute `_make_repo` installs, so
    the live module is not reachable from inside a fixture at all. Whether those variables
    actually close the LIVE fan-out is measured, with a positive control, by
    `test_the_mutes_close_the_live_fan_out_and_every_fixture_sets_them`; whether a run reaches
    a channel process at all is measured by
    `test_the_announcement_tests_reach_neither_systemd_cat_nor_notify_send`.

    (b) `IV_METRICS_ANNOUNCE` — the recorder's own mute — is NOT among them, and
    `_announce_enabled()` reads it from the environment with no default other than "on". If it
    were set, every clause-3 assertion would pass while sending nothing. So the mute is run
    here to show what it does and does not cost: the announcement stops, exit 2 and the stored
    `breach` field survive, and stderr says both that nothing was announced and which variable
    did it. A silent return would be indistinguishable, in the run record, from a night that
    quietly decided not to alert.
    """
    env = _announcing_env(tmp_path, tmp_path / "lloyd")
    assert env["LLOYD_JOURNAL_ALERTS"] == "0"
    assert env["LLOYD_DESKTOP_ALERTS"] == "0"
    assert "IV_METRICS_ANNOUNCE" not in env, (
        "the recorder's own mute must be absent from every announcing test; set, it would "
        "satisfy 'a starting breach announces' by announcing nothing")
    # The positive control first: the environment above, nothing added, one chain.
    on = tmp_path / "unmuted"
    repo_on = _fanout_check(on)
    per_night, codes, _stored, _last = _nights(
        on, rates=[0.06, 0.06, 0.06], out=on / "iv-metrics.jsonl", repo=repo_on)
    assert per_night == [0, 0, 1], (
        f"with the mute unset the breach announced {per_night} times; clause 3 is the "
        "assertion that this absence is load-bearing, not decorative")

    # The negative control: the identical chain with ONLY the mute added, in its own fixture
    # so the receipt file cannot be shared. Mutating one variable and comparing two runs is
    # what makes the mute the difference rather than the fixture.
    off = tmp_path / "muted"
    repo_off = _fanout_check(off)
    per_night, codes, stored, last_run = _nights(
        off, rates=[0.06, 0.06, 0.06], out=off / "iv-metrics.jsonl", repo=repo_off,
        env_extra={"IV_METRICS_ANNOUNCE": "0"})
    assert codes == [0, 0, 2], (
        f"with the mute set the exit codes were {codes}, expected [0, 0, 2] — muting the "
        "alert must not mute the verdict")
    assert per_night == [0, 0, 0], (
        f"the mute let {per_night} messages through; it has to return before a notifier is "
        "constructed, because an announce that fires and is then discounted still puts a "
        "toast in the room")
    assert bool(stored[-1].get("breach")) is True, (
        "the row still carries the verdict, so the next night knows it was inside a breach "
        "it was never allowed to announce")
    assert "breach NOT announced" in last_run.stderr, (
        "the mute is LOUD: a silent return is indistinguishable, in the run record, from a "
        f"night that decided not to announce\nstderr: {last_run.stderr!r}")
    assert "IV_METRICS_ANNOUNCE" in last_run.stderr, (
        "and it names the variable, so a reader can tell a deliberate mute from a broken "
        f"fan-out\nstderr: {last_run.stderr!r}")


def test_the_announcement_tests_reach_neither_systemd_cat_nor_notify_send(tmp_path):
    """Clause 5, measured as invocations: a run that DOES announce starts no channel process.

    The refused first version of this guard asserted `"systemd-cat" not in RECORDING_FANOUT` —
    a string authored four hundred lines above in this same file, which could only fail if
    someone edited that literal. It passed in full on 2026-09-20 while the generic `_make_repo`
    fixture was copying the REAL `notify.py` into its fake home and 23 breach lines from 23
    distinct PIDs landed in the host journal (`journalctl --no-pager --since 2026-09-19 |
    grep -c dropped-verdict`) — 23 warning toasts about a breach that had not happened. A check
    that reads only its own fixture cannot see the other fixture, which is the pattern this
    file's own catalogue already names twice.

    So measure invocations instead, over a PATH of decoy binaries whose only behaviour is to
    record that they ran. The positive control is
    `test_the_mutes_close_the_live_fan_out_and_every_fixture_sets_them`: the same decoys under
    the live `Notifier.announce` DO get invoked, so a zero here is a verdict and not an artefact
    of a missing file or a mangled PATH — the denominator this file's clause-7 lesson says to
    print beside the numerator. And the run under test MUST announce, asserted before the zero:
    a run that alerted nobody would produce the same zero and prove the opposite of the clause.

    The last assertion is the structural half: what protects the machine is not the spelling of
    the substitute but that the fan-out a fixture installs has no way to start a process at all.
    Every channel in the live module is a `subprocess.run`, so that property fails the day a real
    channel call is added to a fixture fan-out, however the new command is spelled — and it
    cannot be satisfied by pointing at a file other than the one the runs executed.
    """
    room = tmp_path / "room"
    room.mkdir()
    bindir, hits = _room_bin(tmp_path)
    out = room / "iv-metrics.jsonl"
    repo = _fanout_check(room)
    # PATH and ROOM_HITS are the only additions: every other variable is the same
    # `_announcing_env` the clause-2/3/4 runs use, so this is that run and not a new one.
    per_night, codes, _stored, last_run = _nights(
        room, rates=[0.06, 0.06, 0.06], out=out, repo=repo,
        env_extra={"PATH": f"{bindir}{os.pathsep}/usr/bin{os.pathsep}/bin",
                   "ROOM_HITS": str(hits)})
    assert codes == [0, 0, 2], f"exit codes were {codes}, expected [0, 0, 2]: {last_run.stderr}"
    assert sum(per_night) == 1, (
        f"the run recorded {sum(per_night)} announcements; the zero below only means "
        "'isolated' if something was actually announced through the fan-out")
    assert not hits.is_file() or not hits.read_text(encoding="utf-8").strip(), (
        "an announcing run invoked a real channel binary while the suite believed it was "
        f"recording instead: {hits.read_text() if hits.is_file() else ''}")

    fan_out = (repo / "agent-services" / "guardian" / "notify.py").read_text(encoding="utf-8")
    assert fan_out == RECORDING_FANOUT, (
        "the fan-out in the fixture is not the source reasoned about below, so anything asserted "
        "of it is a claim about a file the runs never executed")
    spawnable = [tok for tok in ("import subprocess", "Popen", "os.system", "os.exec",
                                 "os.spawn", "os.fork") if tok in fan_out]
    assert not spawnable, (
        f"the fan-out the fixture installs can start a process ({spawnable}), so 'no channel "
        "binary is reached' is no longer a property of the code that runs — an environment "
        "variable would be the only thing left between a fixture breach and the host journal, "
        "which is exactly the arrangement that alerted the machine on 2026-09-20")


#: The driver the clause-5 proof runs: the LIVE `Notifier.announce`, in a subprocess, over
#: the same keyword-free positional route and the same `level="warning"` that
#: `announce_breach` uses. Nothing here fakes the fan-out — the whole point is that the real
#: one is inert under the mutes and live without them.
_LIVE_ANNOUNCE_DRIVER = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import notify
state = Path(sys.argv[2])
state.mkdir(parents=True, exist_ok=True)
notifier = notify.Notifier(ledger=state / "ledger.jsonl", state_dir=state,
                           vault_root=sys.argv[3], voice=False)
print(notifier.announce("IV metrics: clause-5 probe", "probe body", level="warning"))
'''


def _room_bin(tmp_path: Path) -> tuple[Path, Path]:
    """Executables named `systemd-cat` and `notify-send` that only record being run.

    The decoys are what make the clause-5 guard able to fail. The alternatives each answer
    nothing on this machine: grepping the fixture's own source asserts one literal against
    another literal authored in the same file, and `shutil.which` finds the real binaries
    because this box has both installed — so the only observable is whether a process
    invokes those names, and the only way to observe it is to be the thing it invokes.
    Decoys go on the FRONT of `PATH`, so the machine's real binaries are never run.
    """
    bindir = tmp_path / "room-bin"
    bindir.mkdir(parents=True, exist_ok=True)
    hits = tmp_path / "room-hits.txt"
    hits.touch()
    for name in LIVE_CHANNEL_BINARIES:
        script = bindir / name
        # Built as a concatenation, not an f-string: the shell needs `${VAR}` and an
        # f-string would eat the braces.
        script.write_text("#!/bin/sh\n"
                          "printf '%s\\n' '" + name + " $*' >> \"$ROOM_HITS\"\n",
                          encoding="utf-8")
        script.chmod(0o755)
    return bindir, hits


def test_the_mutes_close_the_live_fan_out_and_every_fixture_sets_them(tmp_path):
    """The clause-5 guard, run across a real process boundary in both directions.

    The first version of this check asserted that the string `systemd-cat` did not appear in
    a `notify.py` that this same file writes. That assertion cannot fail — the literal and
    the thing searched are both authored here, so editing one edits the other — and it sat
    green while the fixture builder copied the LIVE `notify.py` for every other test in the
    suite, whose breaching rows alerted for real: 23 lines into the host journal from 23
    distinct PIDs on 2026-09-20, a count `journalctl --no-pager --since 2026-09-19 |
    grep -c dropped-verdict` still gives. What was missing was the room. So this test puts
    the room there, as two decoy executables on the front of `PATH` named `systemd-cat` and
    `notify-send`, each of which only records that it was invoked. `shutil.which` cannot
    answer this question on this machine — both binaries are installed, so "would it have
    found them" is always yes —
    and reading the fixture's source answers nothing, which is exactly what just happened.

    Three claims, in this order:

    - **Positive control.** The live `Notifier.announce`, run as a subprocess with the mutes
      unset and a session bus present, reaches for both binaries. Without this the zero below
      is just a zero, and the class of bug this whole item is written about is a guard whose
      input is missing (`knowledge/software/guardian-data-damage-false-trip.md`).
    - **The mutes close it.** The same subprocess with `_room_mutes()` invokes neither. The
      door is closed by those variables, not by a sandbox that was inert anyway.
    - **Every fixture has them.** `_env` and `_announcing_env` both carry them, checked on
      the dicts the helpers return rather than on the source that builds them — the seam was
      two fixture builders disagreeing, so the guard is on the answer, not on the prose.
    """
    bindir, hits = _room_bin(tmp_path)
    # The LIVE fan-out, copied verbatim into a checkout of its own. Deliberately NOT
    # `_fanout_check`: this test's subject is the module that reaches the room, so it gets no
    # substitute at all.
    guardian = tmp_path / "live-guardian"
    guardian.mkdir()
    shutil.copy2(GUARDIAN / "notify.py", guardian / "notify.py")
    driver = tmp_path / "live_announce_driver.py"
    driver.write_text(_LIVE_ANNOUNCE_DRIVER, encoding="utf-8")

    def announce_live(extra_env: dict) -> list[str]:
        hits.write_text("", encoding="utf-8")
        run = subprocess.run(
            [sys.executable, str(driver), str(guardian), str(tmp_path / "state"),
             str(tmp_path / "vault")],
            capture_output=True, text=True, timeout=90,
            env={"PATH": f"{bindir}{os.pathsep}/usr/bin{os.pathsep}/bin",
                 "HOME": str(tmp_path), "ROOM_HITS": str(hits), **extra_env})
        assert run.returncode == 0, f"live announce driver died: {run.stderr[-400:]}"
        return [ln for ln in hits.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # The two binaries this test guards are the two the live channels shell out to — read out
    # of the live module, so a channel renamed or swapped re-pins `LIVE_CHANNEL_BINARIES`
    # rather than leaving this test guarding names nobody calls.
    live = _real_notify_module()
    for method, binary in zip(("_journal", "_desktop"), LIVE_CHANNEL_BINARIES):
        assert binary in inspect.getsource(getattr(live.Notifier, method)), (
            f"the live {method} no longer runs `{binary}`; LIVE_CHANNEL_BINARIES is guarding "
            "a name that is not the room, and this test proves nothing about the real one")

    # ── positive control: door open, the fan-out reaches the room.
    open_hits = announce_live({"DBUS_SESSION_BUS_ADDRESS": "/tmp/definitely-not-a-bus"})
    assert sorted(ln.split()[0] for ln in open_hits) == sorted(LIVE_CHANNEL_BINARIES), (
        "expected the LIVE fan-out to invoke both room binaries with the mutes unset — that "
        f"control is what makes the closed case below evidence; got {open_hits!r}. Either "
        "the channels changed shape or this test stopped exercising them: rewrite it, do not "
        "leave it green on an input it cannot read")

    # ── the same call, the same decoys, with clause 5's mutes: nothing reaches the room.
    assert announce_live(_room_mutes()) == [], (
        "the live fan-out reached the room WITH the mutes set — clause 5 is a mute that does "
        f"not mute, and every announcement test in this file is unguarded: {open_hits!r}")

    # ── and both fixture environments the suite actually hands to a subprocess carry them.
    for label, env in (("_env", _env(tmp_path / "home-a")),
                       ("_announcing_env",
                        _announcing_env(tmp_path / "home-b", _make_repo(tmp_path / "l2")))):
        missing = [name for name in ROOM_MUTE_VARS if env.get(name) != "0"]
        assert not missing, (
            f"{label} does not mute {missing}, so a fixture subprocess fans out into the "
            "machine's journal and desktop — which is the 2026-09-20 accident: 20 breach "
            "lines, 32 PIDs, for a condition that never happened")
        assert env.get("DBUS_SESSION_BUS_ADDRESS") == "", (
            f"{label} leaves a session bus present for `_desktop`, the one channel "
            "`LLOYD_DESKTOP_ALERTS` alone would not have closed")


# ── #995 clause 4: the grader's new per-call keys must not move this seam ─────
#
# `iv_grade.py` now also reports p50/p90/p99 of `latency_ms` per call, because its
# one printed latency figure was a per-turn SUM and #458 read it as a round-trip
# ("25.8 s/turn against a 12 s deadline"). The SUM's JSON key is this recorder's
# input — `iv_metrics_record.py:290` reads `cost["observer_ms_per_turn"]` — so the
# naming fix had to happen on the PRINTED label, not in the payload. Two series
# rows already carry that key, and `tests/test_iv_metrics_series.py` fixtures it at
# `:520`, so a rename would break the trend at the seam and leave the archived rows
# unjoinable to the new ones. This test is the guard on that decision, run across
# the real pipe.


def test_the_percentile_keys_leave_the_recorder_row_intact(tmp_path):
    """The documented pipe still appends one row, and `observer_ms_per_turn` is
    still its name and its value, while the grader adds the per-call keys beside it.

    Fixture: two completed calls, 100 ms and 300 ms, in one turn each — so the
    grader's SUM is (100+300)/2 = 200 per turn while its per-call percentiles over
    the same two rows are p50 100 / p90 300 / p99 300 (nearest-rank over n = 2).
    The two numbers differing by 2× in one fixture is the point: the row keeps the
    SUM, the new keys carry the round-trip, and neither is quietly redefined into
    the other. Run as a real `bash -c` pipe with the fixture `HOME`, exactly like
    every other end-to-end test in this file.
    """
    repo = _make_repo(tmp_path)
    _make_db(repo, [
        {"created_at": NOW_LOCAL - datetime.timedelta(hours=3), "latency_ms": 100},
        {"created_at": NOW_LOCAL - datetime.timedelta(hours=2), "latency_ms": 300},
    ])
    out = repo / "_pipeline" / "reflection" / "iv-metrics.jsonl"
    since = _iso(NOW_LOCAL - datetime.timedelta(hours=26))

    result = _call(since=since, repo=repo, out=out, extra=["--hours 26"])

    assert result.returncode == 0, (result.returncode, result.stderr[:400])
    rows = _rows_of(out)
    assert len(rows) == 1, f"the pipe appended {len(rows)} rows, expected 1"
    row = rows[0]
    assert row["llm_calls"] == 2, row
    assert row["observer_ms_per_turn"] == 200, (
        "the persisted SUM moved — either the grader redefined its key or the "
        f"recorder stopped reading it: {row['observer_ms_per_turn']!r}")
    assert "observer_ms_summed_per_turn" not in row, (
        "the persisted key was renamed; the archived rows in "
        "_pipeline/reflection/iv-metrics.jsonl would no longer join to the new ones")

    # The grader, same fixture and same bound, on its own — the keys the recorder
    # did NOT copy still have to exist for the next consumer, and the two views of
    # one fixture must agree on the SUM. `sys.executable` rather than the pipe's
    # `python3` because this call only reads keys; the pipe above is what pins the
    # interpreter a scheduled run actually gets.
    graded = subprocess.run(
        [sys.executable, str(repo / "scripts" / "iv_grade.py"),
         "--json", "--since", since],
        capture_output=True, text=True, check=False, cwd=str(repo))
    assert graded.returncode == 0, graded.stderr[:400]
    cost = json.loads(graded.stdout)["cost"]
    assert cost["observer_ms_per_turn"] == row["observer_ms_per_turn"] == 200, cost
    assert cost["latency_ms_per_call_n"] == 2, cost
    assert (cost["latency_ms_per_call_p50"], cost["latency_ms_per_call_p90"],
            cost["latency_ms_per_call_p99"]) == (100, 300, 300), cost
