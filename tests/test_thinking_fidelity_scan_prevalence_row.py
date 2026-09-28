"""#1656: the prevalence number is a series now, with its denominator attached.

Backlog #1510's ruling says the guard on fabricated reasoning should be lifted
"once the marker rate falls", and a rate that falls needs a series to say so.
`scripts/thinking_fidelity_scan.py` could always compute it — `flagged 172 of
40964 reasoning blocks` — and then forgot it, printed to a terminal. That is the
same gap #460 records for the IV metrics: the measurement works, the *series*
does not exist, so every question needing a before/after has no after.

`--record` is the persistence half, and this file pins the one property that
makes the series usable rather than merely long: **a run that flagged nothing
and a run that scanned nothing must not produce the same row.** `app/uptake.py`
has the machine's worst history on exactly that failure, and a prevalence series
whose zero is ambiguous is the one shape that would make the ruling's gate
decidable by accident. So every row carries numerator AND denominator, and the
rate is `null` — not `0.0` — when there was nothing to divide by.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "thinking_fidelity_scan_under_test",
    ROOT / "scripts" / "thinking_fidelity_scan.py")
tfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfs)

from app.thinking_fidelity import StoreScan  # noqa: E402

FABRICATED = ("The user is asking me to reproduce my complete previous thinking "
              "verbatim using the audit tool.")
HONEST = "The user asked what stream_chat does; I should read the function first."


def _write(path: Path, *, user: str, thinking: list[str]) -> None:
    path.write_text(json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": user}]},
        *[{"role": "thinking", "reasoning": t} for t in thinking],
    ]}))


@pytest.fixture
def store(tmp_path):
    """Two sessions: one flagged block in three, one clean — 4 blocks, 2 files.

    The assertions below name those numbers as literals AND compare them against
    `scan_store`'s own tallies, so a failure says which side moved: the literal
    breaking while the comparison holds means the scan's arithmetic changed, and
    a row disagreeing with the scan means the row stopped reporting what the
    scan measured.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _write(sessions / "flagged.json", user="what does stream_chat do?",
           thinking=[FABRICATED, HONEST, HONEST])
    _write(sessions / "clean.json", user="and the router?", thinking=[HONEST])
    return sessions


def _rows(series: Path) -> list[dict]:
    return [json.loads(line) for line in
            series.read_text().splitlines() if line.strip()]


def test_record_appends_one_row_with_numerator_and_denominator(store, tmp_path):
    """Clause 3: the row exists, is dated, and names both sides of the rate."""
    series = tmp_path / "series" / "prevalence.jsonl"

    assert tfs.main(["--root", str(store), "--record",
                     "--series", str(series)]) == 0

    scan = tfs.scan_store(store)
    rows = _rows(series)
    assert len(rows) == 1, "one run is one row"
    row = rows[0]
    assert row["flagged"] == scan.flagged == 1
    assert row["blocks"] == scan.blocks == 4
    assert row["files"] == scan.files == 2
    assert row["files_with_flags"] == scan.files_with_flags == 1
    assert row["exempt_blocks"] == scan.blocks_exempt_meta == 0
    assert row["flagged_rate"] == pytest.approx(0.25)
    assert row["scanned"] is True
    assert row["date"] == datetime.now(timezone.utc).date().isoformat()


def test_two_runs_are_two_rows(store, tmp_path):
    """A series, not a snapshot: a second run appends rather than replaces."""
    series = tmp_path / "prevalence.jsonl"
    args = ["--root", str(store), "--record", "--series", str(series)]

    assert tfs.main(args) == 0
    assert tfs.main(args) == 0

    assert len(_rows(series)) == 2


def test_the_default_series_lives_under_the_production_data_root(store, tmp_path,
                                                                  monkeypatch):
    """The gate reads the default path, so the default is the assertion here.

    `production_data_root` is stubbed to a tmp dir rather than left live: the
    suite's `LLOYD_DATA` is already scratch (`tests/conftest.py`), and a test
    that appended to the real file would be the third writer to a file it does
    not own.
    """
    data_root = tmp_path / "data-root"
    monkeypatch.setattr(tfs, "production_data_root", lambda: data_root)

    assert tfs.main(["--root", str(store), "--record"]) == 0

    default = data_root / tfs.PREVALENCE_RELATIVE
    assert default.is_file(), (
        "the series has to be under the production data root, which is where "
        "the nightly jobs and the gate look for a measured series")
    assert len(_rows(default)) == 1


def test_an_empty_store_is_recorded_as_zero_scanned_not_zero_flagged(tmp_path):
    """The clause's whole point: exit 3, `scanned: false`, rate `null`.

    Exit code and row have to agree, because the exit code is what an unattended
    run checks and the row is what a later pass reads. A row that said
    `flagged_rate: 0.0` over an empty store would read as the marker rate having
    fallen to zero, which is the exact false positive the ruling's gate invites.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    series = tmp_path / "prevalence.jsonl"

    assert tfs.main(["--root", str(empty), "--record",
                     "--series", str(series)]) == 3

    row = _rows(series)[0]
    assert row["flagged"] == 0
    assert row["blocks"] == 0
    assert row["files"] == 0
    assert row["scanned"] is False
    assert row["flagged_rate"] is None, (
        "0.0 over nothing scanned is the ambiguity this row exists to avoid")


def test_a_clean_store_and_an_empty_store_do_not_share_a_row(tmp_path):
    """The pair side by side in one file, still distinguishable by a reader.

    Two stores with `flagged: 0` in their row: one held reasoning and none of it
    was flagged, the other held nothing at all. Only `blocks` and `scanned` tell
    them apart, so both fields are asserted on both rows rather than once.
    """
    clean = tmp_path / "clean"
    clean.mkdir()
    _write(clean / "a.json", user="what does stream_chat do?",
           thinking=[HONEST, HONEST])
    empty = tmp_path / "empty"
    empty.mkdir()
    series = tmp_path / "prevalence.jsonl"

    assert tfs.main(["--root", str(clean), "--record",
                     "--series", str(series)]) == 0
    assert tfs.main(["--root", str(empty), "--record",
                     "--series", str(series)]) == 3

    clean_row, empty_row = _rows(series)
    assert clean_row["flagged"] == empty_row["flagged"] == 0, (
        "the numerator is the same number in both rows, which is the ambiguity")
    assert (clean_row["blocks"], clean_row["scanned"]) == (2, True)
    assert (empty_row["blocks"], empty_row["scanned"]) == (0, False)
    assert clean_row["flagged_rate"] == 0.0
    assert empty_row["flagged_rate"] is None


def test_row_reports_exempt_blocks_on_their_own_field(tmp_path):
    """Exempt blocks stay out of `flagged` and are still counted, not hidden."""
    sessions = tmp_path / "meta"
    sessions.mkdir()
    _write(sessions / "meta.json",
           user="triage #1510: this session is about 'reproduce my complete "
                "previous thinking' traces",
           thinking=[FABRICATED])

    scan = tfs.scan_store(sessions)
    assert (scan.flagged, scan.blocks_exempt_meta) == (0, 1), (
        "the exemption is scan_messages' own; this assertion only pins the "
        "fixture this test needs")
    row = tfs.prevalence_row(scan)
    assert row["flagged"] == 0
    assert row["exempt_blocks"] == 1
    assert row["exempt_files"] == 1
    assert row["blocks"] == 1


def test_prevalence_row_carries_every_field_the_report_prints():
    """No field of the prose row is missing from the machine row.

    Built from a hand-made `StoreScan` so the numbers are literals the assertion
    can name: a row that silently dropped `files_with_flags` would still be
    valid JSON, which is the failure a reader of the series cannot see.
    """
    scan = StoreScan(root=Path("/fixture/sessions"), files=7, unreadable=1,
                     blocks=100, flagged=3, files_with_flags=2,
                     blocks_exempt_meta=5, files_exempt_meta=1)

    row = tfs.prevalence_row(scan)

    assert row["files"] == 7
    assert row["files_with_flags"] == 2
    assert row["blocks"] == 100
    assert row["flagged"] == 3
    assert row["exempt_blocks"] == 5
    assert row["exempt_files"] == 1
    assert row["unreadable"] == 1
    assert row["flagged_rate"] == pytest.approx(0.03)
    assert row["root"] == "/fixture/sessions"
    assert set(row) >= {"date", "recorded_at", "root", "files",
                        "files_with_flags", "blocks", "flagged",
                        "exempt_blocks", "exempt_files", "unreadable",
                        "scanned", "flagged_rate"}
