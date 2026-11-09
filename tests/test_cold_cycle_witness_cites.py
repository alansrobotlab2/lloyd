"""The #3860 cold-cycle witnesses are cited as committed vault files, never as scratch.

`COLD_P90_S` prices every cold dashboard cycle in this repo, and until #3972 the one
line that sourced it named a `meas-3860-r2.log` sitting in the scratch checkout root —
a file in no checkout, in a tree the retention sweep explicitly leaves alone (its
scratch-root rung counts a plain file as `not_a_dir`: "never touched by this rung",
`scripts/groundskeeper/retention-sweep.py:1835`). A reader who wanted to check the
budget against the measurement had to trust one machine that keeps no backup.

The three logs are in the vault now, under `backlog/data/`, and the repo side of #3972
is the four nodes here. Each one can fail, and two of them do at the shipped state:

* `test_no_shipped_source_sources_a_measurement_to_a_scratch_log` — clause 4. The cite
  SHAPE is what is banned, and it is scanned across `app`, `scripts`, `tests` and
  `web/src` INCLUDING this file, so the needle below is spelled as two adjacent
  literals no scan of its own line can read as a cite.
* `test_the_committed_witness_bytes_are_the_measured_ones` — clauses 1 and 2. The
  published md5 of each log is pinned as data, so byte-identity is a rail rather than
  a command somebody remembers to run, and `wc -c`/`wc -l` are read off the committed
  bytes, not off this file's copy of the figures.
* `test_the_committed_witnesses_still_carry_what_they_are_quoted_for` — clause 3. A
  witness that survives as a truncated tail is worse than a missing one, because the
  figures still print.
* `test_the_calibration_block_names_the_committed_witnesses_by_vault_path` — the other
  half of clause 4's "same calibration block": the paths and their figures have to sit
  in the block that prices the table, above it.

These four read the vault. `app/usage_store.py:1044` warns against that, and the
warning is about the FIGURES: an assertion whose number comes from whatever the
machine's vault happens to hold measures the machine. So nothing here derives a figure
from a file — the digests, line counts and byte counts are pinned above, and the vault
is only asked whether it still holds them. `tests/board_presence.py` settled the other
half of the policy on this box: a board-walking test FAILS when the vault is absent
rather than skipping, because a skip is where a lost corpus goes hidden, and its root
is `LLOYD_OBSIDIAN_VAULT`-overridable, which is what makes the no-vault branch
reachable by test rather than by chance.

Clause 5 is the three nodes in `tests/test_dashboard_cold_render.py`, which read THAT
file's own text (`Path(__file__)` at :1588, :1666 and :1976) and go red if a comment
edit costs the calibration table a `#:     serial,` row or reorders it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from tests.board_presence import vault_root

REPO = Path(__file__).resolve().parent.parent
CALIBRATION = REPO / "tests" / "test_dashboard_cold_render.py"

#: The three witnesses behind the cold-cycle calibration as committed to the vault on
#: 2026-11-09: vault-relative path, `wc -l`, `wc -c`, and the md5 of the committed
#: bytes. Those four figures are the ones #3972 publishes and triage re-measured on
#: 2026-11-09, so the digest is byte-identity stated as data — the vault cannot drift,
#: be truncated or be re-encoded without one of these nodes going red.
#:
#: Vault-RELATIVE is the spelling `app/usage_store.py`'s `REPREFILL_WITNESS` already
#: uses (`~/obsidian/` plus this is the file), and the tuple order is source order:
#: retriage 63/5,671, meas-3859 294/24,238, r2 11/995. `r2` is first because it is the
#: set `COLD_P90_S` is derived from.
WITNESSES: tuple[tuple[str, int, int, str], ...] = (
    ("backlog/data/meas-3860-r2.log", 11, 995, "8504fcb9385e9c5bf2eaeda8b6dbcedb"),
    ("backlog/data/meas-3859.log", 294, 24238, "2037e002fa1f2a34bd941e84221a3c42"),
    ("backlog/data/meas-3860-retriage.log", 63, 5671, "1454853dac088f6a4aed459ae61c7b07"),
)

#: Where a shipped citation to a log would be found. The scratch root is spelled in two
#: adjacent literals on purpose: this file is inside the scanned set, and a guard that
#: exempted itself would be a guard with a hole in it — the same reasoning
#: `tests/test_dashboard_cold_render.py` gives for excluding a node from its own search.
_SCRATCH_ROOT = "/lloyd" "-work"

#: The banned thing is a CITE, not the word: `app/paths.py` names the scratch root as a
#: constant, and legitimately so — that is where a round's worktree goes. What may not
#: exist is prose that sources a measurement to a `.log` file living only there, because
#: that figure stops being checkable the day the scratch tree is cleared.
SCRATCH_LOG_CITE = re.compile(re.escape(_SCRATCH_ROOT) + r"[^\s`)]*\.log")

SHIPPED_SOURCE_DIRS = ("app", "scripts", "tests", "web/src")
SHIPPED_SUFFIXES = {".py", ".ts", ".tsx", ".html"}


def _shipped_lines():
    """Every (repo-relative path, line number, text) of shipped source."""
    for part in SHIPPED_SOURCE_DIRS:
        root = REPO / part
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SHIPPED_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                yield str(path.relative_to(REPO)), n, line


def _committed_witness(rel_path: str) -> bytes:
    """The bytes the VAULT holds for one witness — and a failure, not a skip, if it cannot.

    `tests/board_presence.py:12` states the policy: the vault root is a
    `LLOYD_OBSIDIAN_VAULT`-overridable thing a test can move, so a skip guard would be
    a skip any test can walk into, and "an absent vault gets a red line naming the
    path instead of a green one that measured nothing." The root is resolved per call
    for that same reason.
    """
    root = vault_root()
    path = root / rel_path
    if not path.is_file():
        raise AssertionError(
            f"{path} is not there to be read. {rel_path} is the committed witness behind "
            f"`COLD_P90_S`: the vault root this run resolved is {root!r}, and a missing "
            "witness is the loss of the measurement, not a property of the box.")
    return path.read_bytes()


def test_no_shipped_source_sources_a_measurement_to_a_scratch_log():
    """No shipped line points a measurement at a log that lives only in the scratch tree.

    Fails at the shape that shipped: `tests/test_dashboard_cold_render.py:126` cited its
    cold-cycle log by absolute scratch path, which is neither a checkout artefact nor a
    swept one, so the number behind every cold budget had a source nobody could open.
    """
    hits = [f"{rel}:{n}" for rel, n, line in _shipped_lines()
            if SCRATCH_LOG_CITE.search(line)]
    assert not hits, (
        f"{hits}: a shipped line sources a measurement to a log under the scratch root. "
        "A plain file there is in no checkout and the retention sweep neither reclaims "
        "nor preserves it, so the figure becomes unverifiable whenever that tree is "
        "cleared. Commit the log under `backlog/data/` in the vault and cite it "
        "vault-relative, as #3972 did.")


def test_the_committed_witness_bytes_are_the_measured_ones():
    """The vault holds the exact bytes the calibration's figures were taken from.

    Byte-identity off a manual `cmp -s` is a claim that dies with the report, so the
    digest #3972 published for each source log is pinned here: `wc -c` and `wc -l`
    reproduce clause 2 off the same read, and a witness that was edited, truncated,
    re-wrapped or normalised on the way into the vault reddens this node by name
    instead of quietly becoming a different measurement.
    """
    for rel_path, lines, bytes_, digest in WITNESSES:
        raw = _committed_witness(rel_path)
        assert hashlib.md5(raw).hexdigest() == digest, (
            f"{rel_path}: the committed bytes hash to "
            f"{hashlib.md5(raw).hexdigest()}, not the {digest} that "
            f"meas-3860's runs produced. The witness is no longer the file the figures "
            "came from, and nothing but a re-measure can recover it.")
        assert len(raw) == bytes_, (
            f"{rel_path}: `wc -c` off the committed bytes is {len(raw)}, not the "
            f"published {bytes_}.")
        newlines = raw.count(b"\n")
        assert newlines == lines, (
            f"{rel_path}: `wc -l` off the committed bytes counts {newlines} newlines, "
            f"not the published {lines}.")


def test_the_committed_witnesses_still_carry_what_they_are_quoted_for():
    """Each witness still contains the report it is cited for, in the committed bytes.

    Size and digest aside, the three logs are quoted for three specific things, and a
    file can hold its byte count while losing the lines that matter — the r2 sets are
    what `COLD_P90_S` is derived from, meas-3859 is the board the drift was measured
    against, and the retriage log is the failure #3860 was filed from.
    """
    r2 = _committed_witness("backlog/data/meas-3860-r2.log").decode("utf-8")
    samples = re.findall(
        r"^set (\d) cyc (\d) load1 (\d+\.\d+) rc 0 \| cold cycle: (\d+\.\d+)s",
        r2, re.MULTILINE)
    assert len(samples) == 10, (
        f"the committed meas-3860-r2.log yields {len(samples)} `set N cyc M` cold "
        "cycles; the calibration's two CURRENT_RE_BASE rows are five cycles each, so "
        "fewer than ten means the set behind COLD_P90_S is not all there.")
    tails = sorted(float(s[3]) for s in samples)
    assert tails[-1] == 11.53, (
        f"the highest of the ten committed cycles is {tails[-1]} s, and the "
        "calibration's `COLD_P90_S` is the larger of the two sets' tails — the quoted "
        "figure is 11.53.")

    board_3859 = _committed_witness("backlog/data/meas-3859.log").decode("utf-8")
    drift_lines = [ln for ln in board_3859.splitlines() if "board: 3796 item files" in ln]
    assert len(drift_lines) == 13, (
        f"meas-3859.log yields {len(drift_lines)} `board: 3796 item files` lines, not "
        "the 13 that make up its serial sets.")

    retriage = _committed_witness("backlog/data/meas-3860-retriage.log").decode("utf-8")
    assert "assert 3799 <= 3794" in retriage, (
        "meas-3860-retriage.log no longer carries the `assert 3799 <= 3794` drift "
        "failure it is cited for — the red node that made this re-base necessary.")


def test_the_calibration_block_names_the_committed_witnesses_by_vault_path():
    """Each witness is cited, with its committed figures, in the block that prices it.

    The citation is the entire recovery route: the durations live in the vault, and a
    reader holding a fresh checkout can only reach them if the file quoting the figures
    says which committed file holds them. So each vault-relative path AND each
    `wc -l -c` figure has to appear in the block above the calibration table — a figure
    repeated anywhere else in a 2,400-line file is not a citation of the sets at the
    top of it, so the needles below are searched in that block only.
    """
    text = CALIBRATION.read_text(encoding="utf-8")
    table_at = text.find("#:     serial, ")
    assert table_at > 0, (
        "the `#:     serial,` table rows are gone, so there is no table for the "
        "witness cites to sit above — and "
        "`test_the_calibration_table_and_the_cycles_it_records_agree` pairs those rows "
        "with the recorded sets by position")
    block = text[:table_at]

    for rel_path, lines, bytes_, _digest in WITNESSES:
        at = block.find(rel_path)
        assert at >= 0, (
            f"{rel_path} is not cited above the calibration table in {CALIBRATION.name}. "
            "The committed witness behind `COLD_P90_S` has to be named by the file that "
            "quotes its figures, in the vault-relative spelling "
            "`app/usage_store.py`'s `REPREFILL_WITNESS` uses.")
        assert f"{lines} lines" in block, (
            f"{rel_path}'s committed line count ({lines}) is not quoted in the "
            f"calibration block of {CALIBRATION.name}, so the figure `wc -l -c` is "
            "meant to reproduce has nothing on the repo side to be checked against.")
        assert f"{bytes_:,} B" in block, (
            f"{rel_path}'s committed byte count ({bytes_:,}) is not quoted in the "
            f"calibration block of {CALIBRATION.name}, so the figure `wc -l -c` is "
            "meant to reproduce has nothing on the repo side to be checked against.")
