"""#3415 — the #2670 enforce flip is a unit-env decision, and it has to survive a rollback.

The ceiling ships `shadow`: `app/scheduled_ceiling.mode()` returns `enforce` only when a
process carries `LLOYD_SCHEDULED_CEILING_MODE=enforce`, and a shadow turn records its
would-strip names to `scheduling/scheduled-ceiling.jsonl` while dispatching nothing away —
which is why a shadow transcript is indistinguishable from a turn with no ceiling at all.
That means the thing being decided is not in the code: it is two `environment=` lines in
`agent-services/supervisor/conf.d/`, read by supervisord at the process start.

So this file tests the three claims that are checkable from the tree, in the order the
risk runs:

1. **The flip is on the two units that dispatch scheduled turns, and on neither more nor
   less** (clause 1). Both the backend's autonomy ticker and a tool-opened turn under the
   aggregator can be the process that compiles a scheduled surface, so a flip on one unit
   alone leaves the other half of the traffic in shadow while the ledger still stamps
   whatever mode each process holds. `agent-livekit-server.conf` is a voice worker that
   dispatches no scheduled task, and the item says leave it alone — so "two of three" is a
   real, countable constraint, not a no-op.
2. **A process carrying only that variable answers `enforce`** (clause 2), in a subprocess,
   because that is the whole of what supervisord will do for the units.
3. **Removing the variable restores the exact pre-flip reading** (clause 3) — not just the
   word `shadow`, but a freshly appended ledger row that says `mode: shadow` and the four
   pre-flip bars still evaluating over the shadow rows, which is what makes the rollback
   in the conf comments a drill rather than a hope.

Two things this file does NOT do, on purpose:
  * It does not restart, signal or reload any supervisor program. A round cannot restart
    the process that reads the unit env (refused at dispatch), so the enforce traffic the
    flip will produce is owed to the post-landing reader, not simulated here.
  * It does not open the live grant store or the live ledger. The rows below are written
    to `tmp_path` and every ledger path is either passed explicitly or redirected through
    `LLOYD_SCHEDULED_CEILING_LEDGER` — the knob `app.scheduled_ceiling.ledger_path()`
    itself honours — so no reading here can touch `~/lloyd-data`.

Run: .venvs/lloyd/bin/python -m pytest tests/test_scheduled_ceiling_service_unit.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT / "agent-services" / "supervisor" / "conf.d"

#: The two units that can compile a scheduled surface, and the one that cannot.
FLIPPED = ("lloyd-backend.conf", "lloyd-mcp.conf")
UNFLIPPED = ("agent-livekit-server.conf",)

ENV_VAR = "LLOYD_SCHEDULED_CEILING_MODE"
LEDGER_ENV = "LLOYD_SCHEDULED_CEILING_LEDGER"

#: The 2026-10-30 reading of the live ledger, as the fixture's shape. 17 shadow dates, a
#: worst name at 3.7% against a 5.0% bar, 40 of 624 rows carrying a would-strip name. The
#: counts below are that shape at one-tenth the size so the arithmetic is checkable by
#: hand: 8 dates, one name at 2/40 = 5.0% (at the bar, not over it), and the enforce rows
#: the flip will start writing.
DATES = [f"2026-10-{d:02d}" for d in range(11, 19)]


def _conf(name: str) -> str:
    path = CONF_DIR / name
    assert path.is_file(), f"{name} is not in {CONF_DIR} — the counts below would be 0/0/0"
    return path.read_text(encoding="utf-8")


def _env_lines(text: str) -> list[str]:
    """supervisord's `environment=` lines, comments dropped.

    The comment blocks in these files explain the flip back to the reader, so a test that
    counted the variable anywhere in the file would pass on a conf whose only mention of
    it is prose saying it is not set — which is the exact shape of a half-taken rollback.
    """
    return [ln for ln in text.splitlines()
            if ln.strip().startswith("environment=")]


def _env_pairs(text: str) -> dict[str, str]:
    """The `environment=` lines parsed the way supervisord reads them: `K="v",K2="v2"`."""
    out: dict[str, str] = {}
    for line in _env_lines(text):
        for chunk in line.split("=", 1)[1].split(","):
            if not chunk.strip():
                continue
            key, _, value = chunk.partition("=")
            out[key.strip()] = value.strip().strip('"')
    return out


# ───────────────── clause 1: the flip is on the two units, and only those ─────────
@pytest.mark.parametrize("name", FLIPPED + UNFLIPPED)
def test_the_flip_is_a_countable_state_of_the_shipped_unit_files(name):
    """The counts the item names — 1, 1, 0 — read off the bytes supervisord reads.

    `agent-services/supervisor/supervisord.conf` includes
    `/home/alansrobotlab/lloyd/agent-services/supervisor/conf.d/*.conf`, so these ARE the
    live files, not a copy an installer overwrites: an edit here is the flip, and a test
    that read some other conf would pin a file nothing reads.
    """
    expected = 1 if name in FLIPPED else 0
    text = _conf(name)
    found = sum(ln.count(ENV_VAR) for ln in _env_lines(text))
    assert found == expected, (
        f"{name}: {found} assignment(s) of {ENV_VAR} on its environment= line, expected "
        f"{expected}. A second one is a duplicated pair (supervisord takes the last); a "
        f"missing one on a dispatching unit leaves that half of the scheduled traffic in "
        f"shadow while the ledger stamps rows with the other half's mode.")


@pytest.mark.parametrize("name", FLIPPED)
def test_the_flipped_units_set_enforce_on_the_line_that_already_carried_their_environment(name):
    """The value is `enforce`, and it rides the pre-existing line rather than a new one.

    Two distinct failures live here. A pair that reads `shadow` is a flip that does not
    flip — and it would still satisfy a test that only greps for the variable's name,
    which is why the value is asserted. And a second `environment=` line is not additive:
    supervisord's config parser keeps one value per key per program section, so a flip
    written as a new line elsewhere in the file can silently replace the line carrying
    HOME/PATH/PYTHONPATH instead of adding to it.
    """
    text = _conf(name)
    assert len(_env_lines(text)) == 1, (
        f"{name} now has {len(_env_lines(text))} environment= lines; supervisord reads one "
        f"per section and a second can displace the first")
    assert _env_pairs(text).get(ENV_VAR) == "enforce", (
        f"{name}: environment={ENV_VAR}="
        f"{_env_pairs(text).get(ENV_VAR)!r}, expected 'enforce'. Anything but the exact "
        f"word reads as shadow, which is the mode this item exists to leave behind")


def test_the_two_units_that_carry_it_are_the_two_that_dispatch_scheduled_turns():
    """The set, not just the count: a flip moved onto some third unit would satisfy 1/1/0.

    `agent-services/supervisor/supervisord.conf` includes every `.conf` in this directory,
    so the assertion is over the whole shipped fleet: exactly two units carry the
    variable, and they are the backend and the aggregator.
    """
    carriers = sorted(p.name for p in CONF_DIR.glob("*.conf")
                      if any(ENV_VAR in ln for ln in _env_lines(p.read_text(encoding="utf-8"))))
    assert carriers == sorted(FLIPPED), (
        f"units carrying {ENV_VAR}: {carriers}, expected {sorted(FLIPPED)}. The ceiling is "
        f"compiled in the process that dispatches the scheduled turn, and both the "
        f"backend's autonomy ticker and a tool-opened turn under the aggregator can be that "
        f"process; any other unit is a worker that dispatches no scheduled task")


def test_the_shipped_confs_describe_a_rollback_that_only_touches_the_variable():
    """The confs carry their own rollback instruction, and it matches what the code does.

    `mode()` is read per dispatch, so no stored state has to be unwound: deleting the pair
    and restarting the two programs returns the ledger to writing `mode: shadow`. A comment
    that named some other recovery (a migration, a deleted ledger) would send a
    3am-rollback reader to destroy the witness the gate reads, so the wording is pinned to
    the mechanism the drill below runs.
    """
    for name in FLIPPED:
        text = _conf(name)
        comments = [ln.lstrip("; ").strip() for ln in text.splitlines()
                    if ln.lstrip().startswith(";")]
        blob = " ".join(comments)
        # The comment names the ceiling in words rather than repeating the variable: the
        # file carries `LLOYD_SCHEDULED_CEILING_MODE` EXACTLY once, on the environment= line,
        # which is what `grep -c` returns for the two flipped units and 0 for the LiveKit
        # one. A second occurrence in a comment would make that grep read 2 and stop being
        # the flip's witness.
        assert "scheduled-task ceiling" in blob.lower(), (
            f"{name}: no comment says what the pair on the environment= line is for, so a "
            f"reader with only the file cannot tell the flip from a stray environment var")
        assert "#2670" in blob, (
            f"{name}: the comment does not cite the item the ceiling is #2670's, which is "
            "where a reader learns the shadow phase and its bars")
        assert sum(1 for ln in text.splitlines()
                   if ln.startswith("environment=")) == 1, name
        assert "restart" in blob.lower(), (
            f"{name}: the rollback note must say the program has to be restarted — the "
            f"variable is read at process start, so an edit alone changes nothing until "
            f"then, which is also why a round cannot test the live flip")
        assert "shadow" in blob.lower(), (
            f"{name}: the rollback note does not name the mode the system returns to")


# ───────────────── clause 2: the variable alone decides the mode ─────────────────
def _mode_in_subprocess(env: dict[str, str]) -> str:
    """`app.scheduled_ceiling.mode()` as a process that was booted with `env` and nothing
    else of interest.

    A subprocess, not a monkeypatched `os.environ`, because the thing under test is what a
    supervisord-started program sees: the unit's `environment=` line is applied at exec, and
    a test that patched the interpreter's environment proves the function reads a variable,
    not that the flip reaches a process. `env` is `os.environ` plus the caller's key — the
    venv, `PYTHONPATH` and `HOME` have to survive or the import fails for reasons unrelated
    to the ceiling.
    """
    child = dict(os.environ)
    for key, value in env.items():
        if value is None:
            child.pop(key, None)
        else:
            child[key] = value
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import app.scheduled_ceiling as s; "
         "print(s.mode())" % str(ROOT)],
        capture_output=True, text=True, timeout=120, env=child, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-800:]
    return proc.stdout.strip()


def test_a_process_carrying_only_the_variable_answers_enforce():
    """The whole of the flip, from the unit's point of view: one variable, one answer."""
    assert _mode_in_subprocess({ENV_VAR: "enforce"}) == "enforce"


@pytest.mark.parametrize("value", ["", "shadow", "SHADOW", "true", "1", "enforced", "off"])
def test_every_other_value_a_unit_might_carry_answers_shadow(value):
    """A typo must not become enforcement, and a blank must not either.

    The shapes here are the ways a conf edit goes wrong in practice: an empty value from a
    half-typed pair, a capitalized `SHADOW`, a near-miss word (`enforced`), and the boolean
    idiom (`true`/`1`) that supervisord's OWN boolean fields accept — `autostart=true` sits
    two lines below the flip in both files, so a reader who wrote
    `LLOYD_SCHEDULED_CEILING_MODE=true` would reasonably expect enforcement and must instead
    get the mode that refuses nothing.

    Measured, not assumed: `mode()` reads `os.environ.get(...).strip().lower()` and compares
    to `"enforce"`, so anything that is not that word after normalising answers shadow. The
    normalising variants are pinned in the node below rather than lumped in here, because
    they are the opposite claim — ` enforce ` and `Enforce` DO enforce.
    """
    assert _mode_in_subprocess({ENV_VAR: value}) == "shadow", (
        f"{ENV_VAR}={value!r} was read as enforcement: the ceiling's whole safety argument "
        f"for flipping is that an unrecognised value keeps the old behaviour")


@pytest.mark.parametrize("value", ["enforce", " enforce ", "Enforce", "ENFORCE"])
def test_a_write_of_the_word_enforce_takes_effect_whatever_the_case_or_padding(value):
    """Case and whitespace are normalised, so the flip cannot be silently half-taken.

    The symmetry the reviewer will look for: if `Enforce` answered shadow, the operator who
    typed it would believe enforcement was on while the ceiling refused nothing, and the
    ledger would read `shadow` at every row — which is detectable but only by someone who
    already knows to look. `mode()`'s `strip().lower()` makes a capitalisation choice
    impossible to get wrong, and both files ship the lower-cased form so the shipped value
    and the normalised value coincide.
    """
    assert _mode_in_subprocess({ENV_VAR: value}) == "enforce", (
        f"{ENV_VAR}={value!r} did not take effect: a flip an operator can mis-type into "
        f"shadow without noticing is worse than one they cannot type at all")


def test_the_variable_absent_answers_shadow():
    """The pre-flip state, as a subprocess the units do not start.

    This is the rollback arm measured at the mode level: the same interpreter with the key
    removed must not answer `enforce` from anything it cached or imported.
    """
    assert _mode_in_subprocess({ENV_VAR: None}) == "shadow"


# ───────────────── clause 3: the rollback drill, end to end ─────────────────
def _shadow_fixture(n_per_date: int = 5, *, strips: dict[str, list[str]] | None = None) -> list[dict]:
    """Eight dates of shadow rows in the shape the writer produces.

    `strips` maps a date to the names that date's rows should record; the worst name lands
    on two rows of one date, which is 2/40 = 5.0% — exactly at the bar, so the same fixture
    rolls to FAIL with one more reach and shows the bar is live on both sides.
    """
    rows: list[dict] = []
    for i, date in enumerate(DATES):
        for n in range(n_per_date):
            row = {"ts": f"{date}T06:{i:02d}:{n:02d}+00:00", "date": date, "mode": "shadow",
                   "task_id": 40 + (i % 7), "grant_scope": "autonomy-task:40",
                   "run_id": f"{date}-{n}", "envelope_size": 12, "n_calls": 3,
                   "would_strip": []}
            for name, dates in (strips or {}).items():
                if date in dates and n < 1:
                    row["would_strip"] = [name]
            row["n_would_strip"] = len(row["would_strip"])
            rows.append(row)
    return rows


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _gate_run(ledger: Path, env: dict[str, str]) -> tuple[int, str]:
    """The gate as a command, in a child carrying `env`.

    A subprocess again for the same reason as `mode()`: this is the reading an operator or
    the owed-check job takes, and it also gives the child a clean `LLOYD_DATA` so the
    drill's freshly-appended row can only be the one written into `tmp_path`.
    """
    child = dict(os.environ)
    # Hermetic by default: with no LLOYD_GRANT_DB the gate resolves `default_store()`
    # through `workers.queue.configured_db_path()`, which is the LIVE fleet queue, so a
    # reading would grade a refusal against however many grants exist on the box that
    # morning and a grant minted for a nightly job could flip the verdict. One empty DB per
    # call — every caller here passes a fresh `tmp_path`, so the store's per-path cache
    # cannot bleed a row between tests — means nothing is licensed unless this file puts a
    # row into that file first.
    for key, value in env.items():
        if value is None:
            child.pop(key, None)
        else:
            child[key] = value
    # After `env`, because `LLOYD_DATA` is itself one of the keys this helper stages.
    child["LLOYD_GRANT_DB"] = str(Path(child["LLOYD_DATA"]) / "grants-hermetic.db")
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.maintenance.scheduled_ceiling_flip_gate",
         "--ledger", str(ledger)],
        capture_output=True, text=True, timeout=180, cwd=str(ROOT), env=child)
    return proc.returncode, proc.stdout + proc.stderr


def test_the_rollback_drill_restores_the_pre_flip_reading_end_to_end(tmp_path, monkeypatch):
    """Remove the variable, and the next row says shadow and the old bars still decide.

    Four claims, in the order a rollback would need them:

      * the surface compiles `allowed_tools=None` under shadow, i.e. it dispatches nothing
        away — the property that makes the rollback safe to reach for at all;
      * a row appended after the removal carries `mode: "shadow"`, written by the real
        writer (`compile_surface(...).witness(...)`), not asserted about it;
      * the row lands in the file the env knob names, so the drill never touches
        `~/lloyd-data`;
      * the gate over those rows prints the four pre-flip bars and one PASS verdict whose
        exit code is 0, with `enforce rows: 0` — the reading is the pre-flip reading, not a
        post-flip reading that happens to be green.

    Then the flip's other side: with the variable set, the same call writes a row that says
    `enforce`, which is what makes the pair of readings a drill and not a coincidence.
    """
    sys.path.insert(0, str(ROOT))
    import app.scheduled_ceiling as sc

    monkeypatch.setenv("LLOYD_DATA", str(tmp_path))
    ledger = tmp_path / "scheduling" / "scheduled-ceiling.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(LEDGER_ENV, str(ledger))

    # (a) shadow compiles a surface that cannot refuse a name.
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert sc.mode() == "shadow"
    surface = sc.compile_surface(grant_scope="autonomy-task:40", task_id=40,
                                 disallowed_tools=["WebSearch"])
    assert surface.allowed_tools is None, (
        "shadow mode handed the harness an allow-list — a shadow turn must dispatch "
        "nothing away, which is the only reason 17 dates of shadow data were safe to take")

    # (b) + (c) a row appended now says shadow, and goes where the knob points.
    row = surface.witness(["WebSearch", "Bash"], run_id="rollback-drill")
    assert row is not None and row["mode"] == "shadow", row
    assert row["would_strip"] == ["WebSearch"], (
        "a shadow row that does not record the name it would have stripped is no witness "
        "at all — the per-name bars would silently lose this turn")
    written = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert len(written) == 1 and written[0]["mode"] == "shadow", (
        f"the drill's row went to {ledger} as {len(written)} row(s); the ledger is the "
        f"fleet's witness and this test must not be able to write it")

    # (d) the gate's pre-flip verdict over a shadow-only ledger, unchanged by the flip's
    # having been taken and rolled back an hour ago.
    rows = _shadow_fixture(strips={"automod_status": [DATES[2]]})
    _write(ledger, rows + [written[0]])
    rc, out = _gate_run(ledger, {ENV_VAR: None, "LLOYD_DATA": str(tmp_path),
                                 LEDGER_ENV: str(ledger)})
    assert rc == 0, out
    assert "verdict: PRE-FLIP PASS" in out, out
    assert "enforce rows: 0" in out, out
    # The fixture's eight dates PLUS the date the drill's own appended row carries (the
    # writer stamps `date` from the clock, so it is today, a ninth day). Asserting 8 here
    # would pin the test to a fixture the drill never actually wrote.
    assert f"shadow dates: {len(DATES) + 1}" in out, out
    assert "automod_status" in out, out

    # The flip side, same fixture, one variable apart.
    monkeypatch.setenv(ENV_VAR, "enforce")
    assert sc.mode() == "enforce"
    enforcing = sc.compile_surface(grant_scope="autonomy-task:40", task_id=40,
                                   disallowed_tools=["WebSearch"])
    assert enforcing.allowed_tools is not None, (
        "enforce mode handed out None, i.e. no allow-list: the flip would look enabled in "
        "the ledger and constrain nothing")
    row = enforcing.witness(["WebSearch", "Bash"], run_id="flip-drill")
    assert row["mode"] == "enforce", row
    assert row["would_strip"] == ["WebSearch"], (
        "an enforce row that does not record the refusal leaves POST-FLIP HEALTH with "
        "nothing to count")


def test_the_bars_the_rollback_restores_are_live_not_decorative(tmp_path):
    """One more reach of the bar-sitting name takes the restored reading to FAIL.

    The fixture the drill above runs on puts `automod_status` in 2 of 40 shadow rows =
    5.0% — at `NAME_RATE_CEILING`, which the code compares with `>`, so it passes. That is
    the live-data shape: the worst name on the real ledger was `automod_vault_land` at 23
    of 624 = 3.7%, and a name exactly at a bar is a name a bar has to be evaluated on
    rather than assumed. This node adds one reach, making 3 of 40 = 7.5%, and the reading
    the rollback is supposed to restore has to fall over it: a rollback that restored a
    reading which could not fail would restore nothing.

    Independent of the flip variable, which is why no subprocess env is staged here beyond
    the data root — the bars being alive is a property of the gate, and the drill's claim
    is that the rollback hands you those same bars.
    """
    rows = _shadow_fixture(strips={"automod_status": [DATES[2], DATES[3]]})
    assert sum(1 for r in rows if "automod_status" in r["would_strip"]) == 2, \
        "the fixture must start the name AT the bar (2/40 = 5.0%), not above it"
    extra = dict(rows[0], would_strip=["automod_status"], n_would_strip=1,
                 run_id="one-more-reach")
    rows[0] = extra
    ledger = _write(tmp_path / "over-bar.jsonl", rows)
    assert sum(1 for r in rows if "automod_status" in r["would_strip"]) == 3
    rc, out = _gate_run(ledger, {ENV_VAR: None, "LLOYD_DATA": str(tmp_path),
                                 LEDGER_ENV: str(ledger)})
    assert rc == 1, out
    assert "verdict: PRE-FLIP FAIL" in out, out
    assert "automod_status reached in 3 of 40 rows = 7.5%" in out, out
    assert "  automod_status" in out and "  OVER" in out, out


def test_a_pre_flip_failure_decides_the_exit_code_even_after_the_flip_is_taken(tmp_path):
    """The exit code answers both halves after the flip; the pre-flip line stays the old one.

    A ledger of eight dates whose 40 rows are ALL enforce (so the shadow date bar cannot be
    met) with a name refused in 24 of them — 60.0% of enforce rows: PRE-FLIP FAIL on
    `shadow dates 0 < 7`, POST-FLIP FAIL on the refusal rate, exit 1, and the two failure
    lines say which is which. The point of
    keeping them apart is #3415 clause 6 — a red post-flip reading must not trigger a
    revert until someone has named the refused name and said whether it was licensed, and
    that is only possible if the report says which half is red.
    """
    rows = []
    for i, date in enumerate(DATES):
        for n in range(5):
            rows.append({"ts": f"{date}T07:{i:02d}:{n:02d}+00:00", "date": date,
                         "mode": "enforce", "task_id": 40, "grant_scope": "autonomy-task:40",
                         "run_id": f"{date}-{n}", "envelope_size": 12, "n_calls": 3,
                         "would_strip": ["vault_overview"] if n < 3 else [],
                         "n_would_strip": 1 if n < 3 else 0})
    ledger = _write(tmp_path / "enforce-only.jsonl", rows)
    rc, out = _gate_run(ledger, {ENV_VAR: "enforce", "LLOYD_DATA": str(tmp_path),
                                 LEDGER_ENV: str(ledger)})
    assert rc == 1, out
    assert "verdict: PRE-FLIP FAIL" in out, out
    assert "shadow dates 0 < 7" in out, out
    assert "verdict: POST-FLIP HEALTH FAIL" in out, out
    assert "enforce refusal rate 60.0% (24 of 40 enforce rows) > 5.0%" in out, out
    assert "no live grant for that scope" in out or "LICENSED BY A LIVE GRANT" in out, out
