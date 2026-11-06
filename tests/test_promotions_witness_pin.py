"""The #3579 promotions witness, pinned to its own bytes.

`backlog/data/2026-10-30.3579-lloyd-automod-promotions.jsonl` in the vault is the
extract #3579 emitted: the 8 `review` rows and the 8 `gate` rows behind its
`external_blocker` finding, 16 lines / 42,465 bytes — what `wc -l -c` prints for it,
which is the figure the item and its clause quote. Nothing under `tests/` named that
file before this one: `git grep -n '2026-10-30.3579' -- 'tests/*.py'` returned nothing
at this round's base commit.

Why a pin is needed rather than a re-derivation. The rows were cut from the live
promotions ledger, and `LEDGER_ARCHIVE_AGE_DAYS = 14`
(`scripts/groundskeeper/retention-sweep.py`) folds a row out of that file two weeks
after it is written, leaving it only in the `promotions-archive-<YYYYMM>.jsonl.gz`
beside it. Measured on 2026-11-06, the day this file was written: the live ledger alone
holds 6 of these 16 rows, `promotions-archive-202610.jsonl.gz` holds the other 10, and
`promotions-archive-202609.jsonl.gz` holds none of them. The fold only ever moves rows
FORWARD, so the live-only count keeps falling toward zero. A future pass that
re-derives the witness from the live store therefore reads a smaller pair than the one
#3579 published and has no way to tell a sound witness from a broken one. These nodes
read the committed extract and nothing else, so the figure stays checkable with no
promotions store on the machine at all.

What this file adds over the guard that already exists:
`tests/test_failure_ledger_witness.py::test_every_dated_promotions_extract_is_committed_and_parses`
is a property of the directory — it already covers this artifact for being tracked,
sized under `WITNESS_MAX_BYTES`, and JSON-parseable. What no node asserted was the
CONTENT: that the extract is the 8/8 pair, that `external_blocker` distinguishes the
two halves, that the timestamps and item ids are the ones #3579 reported. That is the
half that decays into a file of the right size full of the wrong rows.

The vault is resolved through `tests.board_presence.vault_root()`, not a literal
`~/obsidian`, so a run under another account asks about *its* vault and a run with
`LLOYD_OBSIDIAN_VAULT` set can point the nodes at a throwaway one. The committed
extract is asserted to exist rather than skipped when missing: a witness behind a skip
is a claim with no witness, which is what these nodes exist to prevent.

Clause 5's union read is the one node here that reads this machine's promotions store
(the two nodes after it build a throwaway one rather than borrow it), and it is written
against two constraints. It streams line by line and never loads a file whole —
measured the same day, streaming the whole union costs 0.3 s, so the live file's ~31 MB
is not a reason to narrow the check to the archives. And it skips rather than fails when
no promotions store is present, because these nodes' subject is the committed extract.
Both behaviours are asserted by their own nodes below rather than left as prose about a
branch the gate never reaches.

It ships 10 nodes: `test_the_artifact_is_the_bytes_wc_prints`,
`test_the_sixteen_rows_parse_as_eight_review_and_eight_gate`,
`test_external_blocker_is_absent_from_review_rows_and_true_on_gate_rows`,
`test_created_at_never_goes_backwards_over_those_eight_item_ids`,
`test_every_row_is_still_in_the_promotions_ledger_union_byte_for_byte`,
`test_the_union_read_skips_rather_than_fails_when_no_store_is_present`,
`test_the_union_read_never_loads_a_store_file_whole`,
`test_the_pinned_facts_hold_with_the_ledger_reader_removed`,
`test_the_extract_is_committed_and_unchanged_in_the_vault` and
`test_the_docstring_names_every_node_this_file_ships` — the last being the rail that
keeps this paragraph's count and names true against the module's own bytes.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from tests.board_presence import live_ledger, vault_root

#: The extract #3579 emitted, named exactly as `scripts/automod/backlog.py`'s witness
#: emitter writes it: `<date>.<item>-lloyd-automod-promotions.jsonl`.
ARTIFACT_NAME = "2026-10-30.3579-lloyd-automod-promotions.jsonl"
#: The pair `wc -l -c` prints for it, and the pair every surface quoting this witness
#: repeats. 42,465 bytes is under `WITNESS_MAX_BYTES` = 1,048,576 by a wide margin, so
#: the extract may be read whole; the STORE beside it may not (see `_iter_lines`).
EXPECTED_LINES = 16
EXPECTED_BYTES = 42_465
#: Eight of each. The pair IS the finding: one `review` row per adjudication and the
#: `gate` row that carried its verdict, so a count that stops matching is a witness
#: that no longer shows what #3579 showed.
EXPECTED_REVIEW_ROWS = 8
EXPECTED_GATE_ROWS = 8
#: The last `created_at` in the file, and the ceiling the item quotes for it.
MAX_CREATED_AT = "2026-10-30T04:44:31Z"
#: The 8 `review` rows' item ids in file order. Seven distinct ids: 3565 appears twice,
#: because the round was adjudicated twice — which is why the ordered list is pinned
#: here and the SET of them is pinned below it, rather than one of the two alone.
REVIEW_ITEM_IDS = [2987, 2996, 3074, 3089, 3133, 3154, 3565, 3565]
#: `created_at` is emitted as `%Y-%m-%dT%H:%M:%SZ`. Parsed, not compared as strings:
#: a lexicographic check passes on a row whose offset is missing and fails on a
#: single-digit month, and neither is what "non-decreasing" is meant to catch.
STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def artifact() -> Path:
    """The committed extract, in the vault this run was asked about."""
    return vault_root() / "backlog" / "data" / ARTIFACT_NAME


def _raw() -> bytes:
    path = artifact()
    assert path.is_file(), (
        f"{path} is missing, so the #3579 figures have no bytes behind them any more")
    return path.read_bytes()


def _artifact_lines() -> list[bytes]:
    """The extract's rows as raw bytes, newline stripped, no line dropped.

    Split on `\\n` and nothing else: `bytes.splitlines()` also breaks on `\\x0b`,
    `\\x0c` and `\\x1c`, and a row split on one of those would no longer be
    byte-identical to the same row read back out of the ledger in clause 5.
    """
    return [line for line in _raw().split(b"\n") if line]


def _rows() -> list[dict]:
    out = []
    for n, line in enumerate(_artifact_lines(), start=1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            pytest.fail(f"row {n} of {ARTIFACT_NAME} does not parse as JSON: {exc}")
    return out


def _union_files(live: Path) -> list[Path]:
    """The live ledger plus every monthly archive beside it.

    `scripts/automod/landed_outcomes.py::ledger_files` is the shipped definition of
    that union and is asked for it, so this node's population is the population the
    landed-outcomes report counts — the module's own comment is that a reader naming
    one file "does not read a smaller window: it reads a population that no longer
    contains the events it exists to count".

    `live` is the seam this helper exists to have. The store node passes
    `tests.board_presence.live_ledger()`, NOT `scripts.automod.state.LEDGER_PATH`, and
    the difference is that node's whole reachability:
    `tests/conftest.py::_default_state_dirs_to_scratch` points `LLOYD_AUTOMOD_STATE` at
    an empty scratch directory for the session — as `board_presence`'s
    `timed_ledger_or_stop` says of the same effect, "`S.LEDGER_PATH` resolves to a file
    that does not exist" — so a node that asked the redirect where the promotions ledger
    is would be asking an empty directory and skipping in every gate run, which is a
    clause pinned by a node that never runs. Asking the home path is read-only. The two
    fixture nodes below pass their own `live` and never see this one.
    """
    from scripts.automod.landed_outcomes import ledger_files

    return ledger_files(live)


def _iter_lines(path: Path):
    """Yield one file's lines, streaming, gzip transparent.

    Nothing here may hold a ledger file in memory: `WITNESS_MAX_BYTES` = 1,048,576 is
    the ceiling an owed entry may order, and the live promotions ledger is ~31 MB
    (measured 2026-11-06). `test_the_union_read_never_loads_a_store_file_whole` is the
    rail, and it traps the two shapes this repo's own readers use — `read_text` /
    `read_bytes` on the path (`scripts/automod/state.py::_decode_ledger`) and
    `gzip.decompress(path.read_bytes())` (`landed_outcomes._decode_one`) — because
    reaching for the shipped decoder is exactly how a whole-file read gets reintroduced.
    """
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        for line in handle:
            yield line.rstrip(b"\n")


def _union_hits(live: Path, want: set[bytes]) -> set[bytes]:
    """Which of `want` appear, byte for byte, in the union rooted at `live`.

    Skips rather than fails when no store is present: the subject of these nodes is the
    committed extract, and a machine with no automod state dir must not read as a
    broken witness. The skip itself is pinned by
    `test_the_union_read_skips_rather_than_fails_when_no_store_is_present`.
    """
    present = [path for path in _union_files(live) if path.is_file()]
    if not present:
        pytest.skip(
            f"no promotions store under {live.parent}: the union this node checks "
            "against does not exist, and the extract is pinned by its own bytes")
    seen: set[bytes] = set()
    for path in present:
        for line in _iter_lines(path):
            if line in want:
                seen.add(line)
        if len(seen) == len(want):
            break
    return seen


# ── clauses 1-4: the extract's own facts ───────────────────────────────────


def test_the_artifact_is_the_bytes_wc_prints():
    """Clause 1: 16 lines and 42,465 bytes, and `wc` agrees that that is what it prints.

    The line count is `count(b"\\n")` because that is the definition `wc -l` uses; a
    file whose last row lost its terminator would keep parsing and still stop matching
    every `wc -l` figure anyone quotes for it, which is the drift this node is for.
    """
    raw = _raw()
    lines = raw.count(b"\n")
    assert lines == EXPECTED_LINES, (
        f"`wc -l` of {ARTIFACT_NAME} says {lines} lines, not the "
        f"{EXPECTED_LINES} its clause quotes")
    assert len(raw) == EXPECTED_BYTES, (
        f"`wc -c` of {ARTIFACT_NAME} says {len(raw):,} bytes, not the "
        f"{EXPECTED_BYTES:,} its clause quotes")
    assert len(_artifact_lines()) == EXPECTED_LINES, (
        f"{len(_artifact_lines())} non-empty rows, not {EXPECTED_LINES}: a row was "
        "added or the file gained a blank line, and the 8/8 pair below is now suspect")
    wc = shutil.which("wc")
    if wc is None:  # the equivalence is the claim; without `wc`, the Python figures stand
        return
    printed = subprocess.run([wc, "-l", "-c", str(artifact())],
                             capture_output=True, text=True).stdout.split()
    assert [int(printed[0]), int(printed[1])] == [EXPECTED_LINES, EXPECTED_BYTES], (
        f"`wc -l -c` prints {printed[:2]} for this file while the constants above say "
        f"{[EXPECTED_LINES, EXPECTED_BYTES]}: the Python measurement and the command "
        "every surface quotes have stopped agreeing")


def test_the_sixteen_rows_parse_as_eight_review_and_eight_gate():
    """Clause 2: every row parses, and the events are exactly 8 `review` + 8 `gate`.

    The whole-file event count is asserted as one equality rather than as two
    "at least" checks: a 17th row of either kind, or an event type that is neither,
    changes what the witness shows even though the totals above still read 16.
    """
    rows = _rows()
    assert len(rows) == EXPECTED_LINES, f"{len(rows)} rows parsed, expected {EXPECTED_LINES}"
    events = Counter(row.get("event") for row in rows)
    assert events == Counter({"review": EXPECTED_REVIEW_ROWS, "gate": EXPECTED_GATE_ROWS}), (
        f"the extract's events are {dict(events)}, not "
        f"{EXPECTED_REVIEW_ROWS} `review` + {EXPECTED_GATE_ROWS} `gate`: the pair is the "
        "finding, so a witness whose halves stopped pairing proves nothing about pairing")
    gate_rungs = Counter(row.get("rung") for row in rows if row.get("event") == "gate")
    assert gate_rungs == Counter({"review": EXPECTED_GATE_ROWS}), (
        f"the `gate` rows carry rungs {dict(gate_rungs)}; the extract is the 8 "
        "`review`-rung verdicts, so a row from another rung in here is a different "
        "measurement wearing this file's name")


def test_external_blocker_is_absent_from_review_rows_and_true_on_gate_rows():
    """Clause 3: `external_blocker` separates the two halves — absent on review, true on gate.

    Absent, not false, and the distinction is the whole clause: #3579's finding was
    that the REVIEW rung never emits the field while the GATE rung emits `true` on
    every one of these rows. A `review` row carrying `external_blocker: false` would
    satisfy a `.get(...) is not True` check while stating something the emitter does
    not do, so absence is checked on the key.
    """
    rows = _rows()
    review = [row for row in rows if row.get("event") == "review"]
    gate = [row for row in rows if row.get("event") == "gate"]
    assert len(review) == EXPECTED_REVIEW_ROWS and len(gate) == EXPECTED_GATE_ROWS, (
        f"{len(review)} review / {len(gate)} gate rows parsed, expected "
        f"{EXPECTED_REVIEW_ROWS} / {EXPECTED_GATE_ROWS}, so the split below is not the "
        "split the clause describes")
    carrying = [row.get("item_id") for row in review if "external_blocker" in row]
    assert carrying == [], (
        f"`external_blocker` appeared on review rows {carrying}: the finding is that "
        "the review rung does not emit the field at all")
    wrong = [row.get("round_id") for row in gate if row.get("external_blocker") is not True]
    assert wrong == [], (
        f"gate rows {wrong} carry an `external_blocker` that is not `true`; these 8 "
        "rows were chosen because every one of them is a promotion the gate refused "
        "for an external reason")


def test_created_at_never_goes_backwards_over_those_eight_item_ids():
    """Clause 4: stamps are non-decreasing, none exceeds 2026-10-30T04:44:31Z, ids as quoted.

    The ids are pinned twice over — the ORDERED list and the SET. Order is the ledger's
    append order, so a re-cut extract that reordered rows is a different document even
    when it holds the same eight; and 3565 appears twice, so a check written only
    against a set of 7 would also pass a witness that had silently dropped one of the
    two 3565 adjudications.
    """
    rows = _rows()
    stamps = [row.get("created_at") for row in rows]
    assert all(isinstance(s, str) and s for s in stamps), (
        f"a row is missing `created_at`: {stamps}")
    parsed = [datetime.strptime(s, STAMP_FORMAT) for s in stamps]
    backwards = [(n, stamps[n - 1], stamps[n])
                 for n in range(1, len(parsed)) if parsed[n] < parsed[n - 1]]
    assert backwards == [], (
        f"`created_at` goes backwards at {backwards[0]}: the extract is a slice of an "
        "append-only ledger, so an out-of-order row means these bytes were assembled "
        "rather than cut")
    assert max(parsed) == datetime.strptime(MAX_CREATED_AT, STAMP_FORMAT), (
        f"the newest stamp is {max(stamps)}, not the {MAX_CREATED_AT} the clause quotes "
        "as the end of the window")
    review_ids = [row.get("item_id") for row in rows if row.get("event") == "review"]
    assert review_ids == REVIEW_ITEM_IDS, (
        f"the review rows' item ids in file order are {review_ids}, not "
        f"{REVIEW_ITEM_IDS}")
    assert set(review_ids) == set(REVIEW_ITEM_IDS), (
        f"{set(review_ids)} is not {set(REVIEW_ITEM_IDS)}")


# ── clause 5: the union read, and the two behaviours it must keep ──────────


def test_every_row_is_still_in_the_promotions_ledger_union_byte_for_byte():
    """Clause 5: all 16 rows are still in the live ledger + monthly archives, byte for byte.

    Byte-for-byte, not field-for-field: the claim the archive makes
    (`scripts/groundskeeper/retention-sweep.py` §13) is that the gzip "holds the
    original line, byte for byte", and the folded rows are only findable again at all
    if the fold preserved the spelling. A row that was rewritten on the way into the
    archive would satisfy a parsed comparison and break the reproducibility the extract
    exists to provide.

    This node asks the store a question about the extract; it never answers a question
    about the extract FROM the store — the pin above is what stands when this one
    skips. Asking `live_ledger()` rather than the redirected `state.LEDGER_PATH` is what
    makes the branch reachable at all, and is explained on `_union_files`.
    """
    want = set(_artifact_lines())
    assert len(want) == EXPECTED_LINES, (
        f"the extract yields {len(want)} distinct rows, not {EXPECTED_LINES}: two rows "
        "are byte-identical, so a union membership count could be satisfied by one of "
        "them appearing twice")

    seen = _union_hits(live_ledger(), want)
    missing = sorted(m.decode("utf-8", "replace")[:120] for m in want - seen)
    assert not missing, (
        f"{len(want) - len(seen)} of the {EXPECTED_LINES} extract rows are no longer in "
        "the promotions ledger or any monthly archive beside it. §13 of the retention "
        "sweep says an archive is the one thing that removes a row and the gzip beside "
        "the ledger 'holds the original line, byte for byte', so a row missing from "
        f"the union has been deleted, not folded. First missing row: {missing[0]}")


def test_the_union_read_skips_rather_than_fails_when_no_store_is_present(tmp_path):
    """Clause 5's other half, asserted rather than promised: an empty state dir SKIPS.

    Built on a fixture because the branch is unreachable on a machine that has an
    automod state dir, and a skip nobody has watched fire is a node that has never been
    tested. Both sides of the guard are seen: no file at all skips, and an ARCHIVE with
    no live ledger beside it does not — the folded rows are the majority of this
    witness, so treating a missing live file as a missing store would skip the check
    that matters most.
    """
    want = set(_artifact_lines())
    live = tmp_path / "promotions.jsonl"
    with pytest.raises(pytest.skip.Exception) as caught:
        _union_hits(live, want)
    assert "no promotions store" in str(caught.value), (
        f"the absent-store branch raised {caught.value!r}, which is not the skip this "
        "node's wording promises: a reader would see a reason nobody wrote")

    half = sorted(want)[:2]
    (tmp_path / "promotions-archive-202610.jsonl.gz").write_bytes(
        gzip.compress(b"".join(line + b"\n" for line in half)))
    try:
        seen = _union_hits(live, set(half))
    except pytest.skip.Exception as exc:
        pytest.fail(
            "an archive with no live ledger beside it skipped the union check, so on a "
            f"machine mid-fold the node goes quiet instead of checking: {exc}")
    assert seen == set(half), (
        f"the archive-only union returned {len(seen)} of {len(half)} rows: the glob in "
        "`_union_files` is not reaching a monthly archive")


def test_the_union_read_never_loads_a_store_file_whole(tmp_path, monkeypatch):
    """Clause 5's streaming guarantee, trapped rather than described.

    The three calls that WOULD load a file whole are made to raise for the duration of
    the read: `Path.read_bytes`, `Path.read_text` and `gzip.decompress`. Those are not
    an arbitrary sample — they are the two shapes this repo's own ledger readers use
    (`scripts/automod/state.py::_decode_ledger` is `read_text`;
    `landed_outcomes._decode_one` is `gzip.decompress(path.read_bytes())`), and reaching
    for the shipped decoder is exactly how a whole-file read gets reintroduced into a
    node that currently streams. The fixture splits the extract the way the live ledger
    splits it today: 6 rows in the live file, 10 in a monthly gzip.
    """
    lines = _artifact_lines()
    live = tmp_path / "promotions.jsonl"
    live.write_bytes(b"".join(line + b"\n" for line in lines[:6]))
    (tmp_path / "promotions-archive-202610.jsonl.gz").write_bytes(
        gzip.compress(b"".join(line + b"\n" for line in lines[6:])))
    want = set(lines)

    def no_whole_file_read(*args, **kwargs):
        raise AssertionError(
            "the union read loaded a ledger file whole, which WITNESS_MAX_BYTES = "
            "1,048,576 exists to forbid against a ~31 MB live promotions ledger")

    monkeypatch.setattr(pathlib.Path, "read_bytes", no_whole_file_read)
    monkeypatch.setattr(pathlib.Path, "read_text", no_whole_file_read)
    monkeypatch.setattr(gzip, "decompress", no_whole_file_read)
    seen = _union_hits(live, want)
    assert seen == want, (
        f"the streamed fixture union returned {len(seen)} of {len(want)} rows, so the "
        "streaming path is not actually reading both the plain file and the gzip")


def test_the_pinned_facts_hold_with_the_ledger_reader_removed(monkeypatch):
    """The second witness the item asks for: the pin does not depend on the store.

    The live-store census over these 8 pairs is deliberately NOT asserted anywhere in
    this file — it is a decaying number by construction (6 of 16 rows sat in the live
    file when this was written, and the fold only moves forward), and pinning it would
    put a second copy of the rotting claim inside the guard meant to survive it. What is
    asserted instead is the dependency: with `ledger_files` made to raise, every
    extract-only node still passes, so none of them can be reading the store by another
    route and calling that a re-derivation.
    """
    import scripts.automod.landed_outcomes as landed_outcomes

    def unreachable(*args, **kwargs):
        raise AssertionError(
            "an extract-only node reached for the promotions ledger union; the point of "
            "this file is that these facts hold with no store present")

    monkeypatch.setattr(landed_outcomes, "ledger_files", unreachable)
    for node in (test_the_artifact_is_the_bytes_wc_prints,
                 test_the_sixteen_rows_parse_as_eight_review_and_eight_gate,
                 test_external_blocker_is_absent_from_review_rows_and_true_on_gate_rows,
                 test_created_at_never_goes_backwards_over_those_eight_item_ids,
                 test_the_extract_is_committed_and_unchanged_in_the_vault):
        node()


def test_the_extract_is_committed_and_unchanged_in_the_vault():
    """Clause 6: the bytes these nodes read are the bytes the vault has committed.

    `is_file()` is not enough and the sibling directory-wide node knows it: an extract
    whose history has gone is a scratch file, and history is the only thing that outlives
    the fold of the store the extract was cut from. Two checks, the second of which that
    node does not make — it compares HEAD to the index, while `diff --quiet` compares
    the WORKING TREE to the index, so a locally-modified extract that every node above
    is silently reading is caught here and only here.
    """
    root = vault_root()
    target = artifact()
    assert target.resolve().is_relative_to(root.resolve()), (
        f"{target} is not inside the vault at {root}, so the two `git` calls below would "
        "be grading some other tree")
    rel = f"backlog/data/{ARTIFACT_NAME}"
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", rel],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{rel} is not tracked in the vault at {root}: {tracked.stderr.strip()}")
    unchanged = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "--", rel], capture_output=True)
    assert unchanged.returncode == 0, (
        f"{rel} differs from the vault's committed copy, so the 16/42,465 figures above "
        "describe bytes that exist only on this machine and nobody can re-check")


def test_the_docstring_names_every_node_this_file_ships():
    """The rail on this file's own prose, in the shape the review rung asks for.

    The count and every node name are checked against the module's bytes. Retyping a
    number fixes one drift and leaves the next; a count this file recomputes is the only
    version that stays true when a node is added or renamed.
    """
    doc = sys.modules[__name__].__doc__ or ""
    names = sorted(name for name, value in globals().items()
                   if name.startswith("test_") and callable(value)
                   and getattr(value, "__module__", None) == __name__)
    unnamed = [name for name in names if name not in doc]
    assert unnamed == [], (
        f"{unnamed} ship in this file but are not named in its docstring, so the "
        "paragraph that enumerates them is now an undercount")
    stated = re.search(r"It ships (\d+) nodes", doc)
    assert stated, "the docstring no longer states its node count as `It ships N nodes`"
    assert int(stated.group(1)) == len(names), (
        f"the docstring states {stated.group(1)} nodes and this module defines "
        f"{len(names)}: the enumeration is prose, and only this node keeps it true")
    listed = sorted(set(re.findall(r"`(test_[a-z0-9_]+)`", doc)))
    assert listed == names, (
        f"the docstring lists {sorted(set(listed) - set(names))} that no longer exist "
        f"or omits {sorted(set(names) - set(listed))}")

