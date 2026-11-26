"""#3924: a dated promotions extract is proved against the ledger POOL, never the live file alone.

Every extract under `backlog/data/` was cut out of the automod promotions ledger, and
`LEDGER_ARCHIVE_AGE_DAYS = 14` (`scripts/groundskeeper/retention-sweep.py` §13) folds a row
out of that live file two weeks after it is written, leaving it only in the
`promotions-archive-<YYYYMM>.jsonl.gz` beside it. The fold only ever moves rows OUT of the
live file, so the live-only count of every extract is monotonically falling: a re-check
that reads the live ledger alone starts true and quietly becomes false, with no signal and
no way to tell a sound witness from a broken one. That proof is in use — the item's own
published line for `2026-11-02.3718` is 254 rows, and
`grep -F -f backlog/data/2026-11-02.3718-lloyd-automod-promotions.jsonl ~/.local/state/lloyd-automod/promotions.jsonl | wc -l`
returned **226** the day this file was written.

Measured 2026-11-08, over the dated promotions extracts committed that day:
**26** come back byte-for-byte out of the union of the live ledger plus its monthly
rotations, and **22 of those 26 are already short against the live file alone** — 4 are
still live-complete today (3367 23/23, 3683 10/10, 3619 3/3, 3461 2/2) and the fold is
taking them next. The pool today is live 19,236 rows plus `promotions-archive-202609.jsonl.gz`
26,417 and `-202610.jsonl.gz` 31,005; the live count is the one number in that list that is
not stable. The clearest single case is `2026-10-11.2444`: **0 of its 82 rows are in the
live file, and 82 of 82 are in the pool** — so a node satisfied by the live file alone is
not merely weaker, it is reading a file that no longer contains its subject at all.
Streaming the whole pool costs 0.35 s (measured), which is why this guard reads all of it.

The other **7** extracts are not store bytes at all and are exempted by name, in
`NON_VERBATIM_CUTS`, for what they are (a rollup, a compact re-serialisation, a projected
cut, a derived join). Their exemptions are not free: `_corpus_findings` asserts of an
exempted extract that it is genuinely SHORT in the pool, so the allow-list cannot be used
to excuse a file that the pool does in fact recover, and no extract can be exempted for
"passes against the live file today" — that is the decaying property this file exists to
ban, and it is why the four still-live-complete extracts above are checked, not excused.

What the union is, and which seam it is asked through. `scripts.automod.state` resolves its
state dir from `LLOYD_AUTOMOD_STATE` **at import**, and `tests/conftest.py`'s
`_default_state_dirs_to_scratch` points it at an empty scratch directory for the whole
session, so `state.LEDGER_PATH` here is a file that does not exist. A node that asked the
redirect where the promotions ledger is would ask an empty directory and SKIP in every gate
run: permanently green, permanently blind. So the union is asked through
`tests.board_presence.live_ledger()` — the home path, read-only — and handed to the shipped
reader `scripts.automod.landed_outcomes.ledger_files`, which is the same population the
landed-outcomes report counts (`state.LEDGER_ARCHIVE_GLOB` / `ledger_archives()`, the fix
for #3079 and #3179, which are this same bug one level up). `test_the_union_under_test_...`
pins that population against the glob on disk so a redirected seam cannot pass blind.

Discovery is the other half of "reads the right population", and the shipped pattern is not
enough. `_promotion_extract_findings` (`tests/test_failure_ledger_witness.py:810`) globs
`*.????-*promotions.jsonl`, and in `fnmatch` the tail of that pattern is LITERAL: it ends at
`promotions.jsonl`, so it reaches the `-lloyd-automod-promotions.jsonl` family and cannot
reach the `-promotions-witness.jsonl` family at all — including
`2026-10-08.2383-promotions-witness.jsonl`, which is one of the 7 cuts and has to be
reachable for its exemption to be checkable. This file's corpus pattern widens that one
`-` and covers both families; `test_the_corpus_glob_reaches_every_dated_promotions_extract`
pins the superset relation, and its family rail means a THIRD suffix family landing is a
finding rather than another silent drop, which is how 2383 and 2079 came to be uncovered in
the first place.

Nothing here edits a historical marker or restates a published figure: the extracts are read
and never written, and every number quoted above is recomputed by a node rather than
carried as an assertion, so the dated prose in the vault stays verbatim.

The corpus carries no size (#4584). Presence is the dated extracts on disk compared with
the same family in the vault's git index — two instruments, so a file lost from disk is red
and growth needs no edit — and freshness is the newest extract's own date against the live
ledger's newest row, inside `_RECENCY_WINDOW_DAYS`.

It ships 12 nodes: `test_the_corpus_glob_reaches_every_dated_promotions_extract`,
`test_an_extract_lost_from_disk_is_named_missing_against_the_vault_index`,
`test_the_loop_has_published_a_witness_extract_within_the_window_of_its_ledger`,
`test_the_recency_alarm_passes_inside_its_window_and_fails_beyond_it`,
`test_every_verbatim_extract_is_recoverable_byte_for_byte_from_the_ledger_union`,
`test_the_union_under_test_is_the_shipped_reader_behind_the_home_seam`,
`test_the_non_verbatim_allow_list_is_exactly_the_cuts_and_each_names_a_live_file`,
`test_a_store_missing_one_extract_row_is_named_short_by_the_corpus_check`,
`test_the_live_file_alone_cannot_recover_the_corpus_it_is_checked_against`,
`test_the_corpus_read_never_loads_a_store_file_whole`,
`test_the_corpus_check_skips_rather_than_fails_when_no_store_is_present` and
`test_the_docstring_names_every_node_this_file_ships` — the last being the rail that keeps
this paragraph's count and names true against the module's own bytes.
"""

from __future__ import annotations

import gzip
import pathlib
import re
import sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from tests.board_presence import live_ledger, vault_root

#: The witness directory, as `tests/test_failure_ledger_witness.py` names it — one
#: constant so a fixture cannot drift out of the directory the live nodes read.
_PROMOTION_SUBDIR = "backlog/data"

#: The corpus: every dated promotions extract, whichever naming family it belongs to.
#: `_promotion_extract_findings` uses `*.????-*promotions.jsonl`, whose fnmatch tail is a
#: literal `promotions.jsonl` and so cannot match `...-promotions-witness.jsonl`; widening
#: the tail to `*` is the one change that makes both families in scope, which clause 3's
#: own 2383 entry requires.
_CORPUS_GLOB = "*.????-*promotions*.jsonl"

#: The shipped pattern, kept here so the node below can pin the relation rather than
#: assert this file's glob is simply better.
_SHIPPED_CORPUS_GLOB = "*.????-*promotions.jsonl"

#: One extract of the family the shipped pattern cannot reach, as a string: the node that
#: uses it is demonstrating what the PATTERN reaches, and `fnmatch` answers that of a name
#: whether or not the file exists anywhere.
_WITNESS_FAMILY_SAMPLE = "2026-10-02.2079-promotions-witness.jsonl"

#: The two suffix families this directory has ever used. A hit whose name ends in neither
#: is a finding: an unknown family that quietly joins the corpus unchecked is the exact
#: way 2383 sat uncovered, so discovery widens only when a node is told about the family.
_KNOWN_SUFFIXES = ("-lloyd-automod-promotions.jsonl", "-promotions-witness.jsonl")

#: The item id is the field after the date in every dated extract name. Cut by pattern,
#: never from a roster: a node that named its subjects stops being a property of the
#: directory the day the 34th extract lands, which is how seven extracts stayed unpinned.
_ITEM_ID = re.compile(r"^\d{4}-\d{2}-\d{2}\.(\d+)-")

#: No corpus SIZE is carried here (#4584). The floor that stood in this slot was a hand-typed
#: count, re-derived by nobody, that had gone five extracts slack within three weeks — and a
#: bumped literal is the same hand-named node wearing a new number. Presence is instead the
#: dated extracts on disk against the same family as the vault's git index records it
#: (`_presence_findings`): loss from disk is red, growth is free. Its stated limit: a `git rm`
#: committed beside the deletion moves both instruments together, which is what the
#: named-presence nodes in this file and `tests/test_failure_ledger_witness.py` are for.

#: The freshness window, in days: the newest dated extract's own date may trail the live
#: ledger's newest row by at most this much. Its purpose: catch a fold season in which the
#: loop keeps writing (and folding) the ledger but publishes no witness extract, which a
#: presence check cannot see — a corpus frozen at any size reads complete forever. 14 is
#: `LEDGER_ARCHIVE_AGE_DAYS` (`scripts/groundskeeper/retention-sweep.py`): a window that
#: long means every row live at the newest extract has since been folded out of the live
#: file. The widest gap between consecutive extract dates through 2026-11-23 was 9 days
#: (11-08 to 11-17). Measured against the LEDGER, not the wall clock: an idle loop that
#: writes no rows owes no witness, and a test rung may not depend on today's date.
_RECENCY_WINDOW_DAYS = 14

#: Every exemption reason must say, in these words, what it is excusing. The point is not
#: the phrase; it is that an exemption has to state a property of the FILE rather than a
#: result the store happens to give, because the store's result is what decays.
_NOT_STORE_BYTES = "not store bytes"

#: The kinds of cut the corpus actually contains. An exemption naming none of them is not
#: describing a cut.
_CUT_KINDS = ("rollup", "re-serialis", "projection", "derived join")

#: The seven extracts that are not store bytes, by item id, each with what it is instead.
#: Measured 2026-11-08 against the pool, all seven are genuinely short in it — and the
#: corpus check re-asserts that on every run, so an entry cannot outlive its justification.
NON_VERBATIM_CUTS: dict[str, str] = {
    "2383": "A per-round gate summary shaped `{created_at, guards, item_id, ok}` that the "
            "ledger never emits: 0 of its 25 lines are store bytes, and the pool holds no "
            "row with that field set — a re-serialised cut, not store bytes.",
    "2759": "A rollup: one header row and one aggregate row per round over 50,141 ledger "
            "lines, followed by a deliberately raw block. Its own note publishes 65 verbatim "
            "lines of its 1,742, and the pool returns exactly 65 — a rollup, not store bytes.",
    "2992": "jq-compact re-serialisation: same fields as the ledger's rows, written without "
            "the store's `, ` separators, and its note says so in a section heading. 0 of its "
            "384 lines match byte-for-byte — re-serialised, not store bytes.",
    "3205": "The store's row plus a `source_store` provenance key the store never wrote, so "
            "all 4 lines differ from the same instant's ledger row at the first added key — "
            "a re-serialised cut, not store bytes.",
    "3753": "A field projection of the review rows, keys sorted and truncated to the grader "
            "census (`attempt, grader_model, has_grader_model_served, ...`) with no `event` "
            "key — a projection of store rows, not store bytes: 0 of 499 match.",
    "3756": "`round_start` rows re-keyed for the opening-instant census: same `ts` and "
            "round_id as the ledger's own row but a different key set and order, so only 2 "
            "of its 29 lines coincide — re-serialised, not store bytes.",
    "3684": "Says it in its own first line: `Whole rows of the store cannot be shipped: ... "
            "this file is a DERIVED JOIN, not an excerpt`. 0 of its 1,744 lines are store "
            "bytes — a derived join, not store bytes.",
}

#: Wording no exemption may rest on. The 14-day fold only removes rows from the live file,
#: so "it matches the live ledger today" is a claim with an expiry date, and an allow-list
#: that could grow on that basis would un-guard the corpus one extract at a time.
_FORBIDDEN_JUSTIFICATIONS = ("passes", "today", "live file", "live only")


def _data_dir(vault: Path) -> Path:
    return vault / _PROMOTION_SUBDIR


def _union_paths(live: Path) -> list[Path]:
    """Live ledger plus every monthly rotation, by the shipped reader.

    `scripts.automod.landed_outcomes.ledger_files` is asked for the union and `live` comes
    from `tests.board_presence.live_ledger()`, never from `scripts.automod.state.LEDGER_PATH`
    — the reasons are on the module docstring, and
    `test_the_union_under_test_is_the_shipped_reader_behind_the_home_seam` is the rail. The
    archives are globbed from `live`'s own directory, which is what lets the fixture nodes
    hand this a throwaway store.
    """
    from scripts.automod.landed_outcomes import ledger_files

    return ledger_files(live)


def _iter_lines(path: Path):
    """Yield one store file's lines, streaming, gzip transparent.

    No store file may be held in memory: the live promotions ledger measured 2026-11-08 is
    19,236 rows over tens of MB, several times the `WITNESS_MAX_BYTES` = 1,048,576 ceiling
    an owed entry is allowed to order. `test_the_corpus_read_never_loads_a_store_file_whole`
    is the rail, and it traps the two shapes this repo's own readers use —
    `Path.read_text` (`scripts/automod/state.py::_decode_ledger`) and
    `gzip.decompress(path.read_bytes())` (`landed_outcomes._decode_one`) — because reaching
    for the shipped decoder is how a whole-file read gets reintroduced here.
    """
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        for line in handle:
            yield line.rstrip(b"\n")


def _scan(files: list[Path], want: set[bytes]) -> set[bytes]:
    """Which of `want` appear, byte for byte, in `files`. One streaming pass, no early exit.

    No early exit on "everything found": the exempt extracts in `want` are never found in
    full by design, and an early break keyed on the verbatim half would leave the exempt
    half's counts wrong in `detail` — which is the number the allow-list's honesty depends
    on. The full pass costs 0.35 s measured over 76,658 lines.
    """
    seen: set[bytes] = set()
    for path in files:
        for line in _iter_lines(path):
            if line in want:
                seen.add(line)
    return seen


def _extract_rows(data: Path) -> dict[str, tuple[str | None, list[bytes], int]]:
    """Every corpus hit's item id, rows as raw bytes, and its `wc -l`.

    This is the only place this file reads a WHOLE file, and only ever an extract: the
    extracts are each under `WITNESS_MAX_BYTES` — the ceiling
    `test_every_dated_promotions_extract_is_committed_and_parses` already holds them to —
    while the store beside them is not. Call it before installing that node's read trap.

    Split on `\\n` and nothing else: `bytes.splitlines()` also breaks on `\\x0b`, `\\x0c` and
    `\\x1c`, and a row split on one of those is no longer byte-identical to the same row read
    back out of the ledger.
    """
    out: dict[str, tuple[str | None, list[bytes], int]] = {}
    for path in sorted(data.glob(_CORPUS_GLOB)):
        raw = path.read_bytes()
        rows = [line for line in raw.split(b"\n") if line]
        m = _ITEM_ID.match(path.name)
        out[path.name] = (m.group(1) if m else None, rows, raw.count(b"\n"))
    return out


def _corpus_findings(rows: dict[str, tuple[str | None, list[bytes], int]], live: Path,
                     allow: dict[str, str], *, live_only: bool = False
                     ) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """What is wrong with the corpus, and each extract's `(hits, rows)` in the store read.

    Takes the rows and a store path rather than reaching for the world itself, because the
    positive controls have to hand it a store with a row taken out — a claim about the
    guard that could only be made against the live ledger could not be made at all.

    `live_only=True` narrows the read to the live file. Nothing in the live nodes checks
    the corpus that way; it is the dial the control uses to show the difference this guard
    exists to make, and it is what proves a node reading less cannot pass.

    Skips rather than fails when no store is present: the subject of these nodes is the
    committed extract, and a machine with no automod state dir must not read as a broken
    witness. Both branches are pinned by
    `test_the_corpus_check_skips_rather_than_fails_when_no_store_is_present`.
    """
    verbatim: dict[str, set[bytes]] = {}
    exempt: dict[str, set[bytes]] = {}
    findings: list[str] = []

    for name, (item_id, file_rows, wc_lines) in rows.items():
        if not name.endswith(_KNOWN_SUFFIXES):
            findings.append(
                f"{name} is in the corpus glob but ends in none of {_KNOWN_SUFFIXES}: a new "
                "dated-promotions suffix family has joined the corpus, and no node here "
                "knows it, which is how 2383 and 2079 sat outside the shipped glob")
            continue
        if item_id is None:
            findings.append(
                f"{name} carries no item id after its date, so this guard cannot key it "
                "against `NON_VERBATIM_CUTS` or its `.md` sibling")
            continue
        if not file_rows:
            findings.append(f"{name} holds 0 rows, so its witness proves nothing about anything")
            continue
        if wc_lines != len(file_rows):
            findings.append(
                f"`wc -l` of {name} says {wc_lines} for {len(file_rows)} non-empty rows: the "
                "file lost a row terminator, so every `wc -l` figure quoted for it now "
                "counts one row short")
            continue
        distinct = set(file_rows)
        if len(distinct) != len(file_rows):
            findings.append(
                f"{name} holds {len(file_rows)} lines but only {len(distinct)} distinct "
                "ones, so a pool membership count could be satisfied by one row appearing "
                "twice — the row count and the hit count have to be different numbers")
            continue
        (exempt if item_id in allow else verbatim)[name] = distinct

    want: set[bytes] = set()
    for wanted in (*verbatim.values(), *exempt.values()):
        want |= wanted

    files = [live] if live_only else _union_paths(live)
    present = [path for path in files if path.is_file()]
    if not present:
        pytest.skip(
            f"no promotions store at {[str(p) for p in files]}: the union this guard "
            "checks against does not exist, and the extracts are pinned by their own bytes")

    seen = _scan(present, want)
    detail = {name: (len(wanted & seen), len(wanted))
              for name, wanted in (*verbatim.items(), *exempt.items())}

    for name, wanted in sorted(verbatim.items()):
        hits = len(wanted & seen)
        if hits != len(wanted):
            findings.append(
                f"{name} is recovered {hits} of its {len(wanted)} rows from the promotions "
                "ledger plus its monthly rotations, byte for byte. §13 of the retention "
                "sweep says the gzip beside the ledger 'holds the original line, byte for "
                "byte', so rows missing from the UNION were deleted rather than folded — or "
                "this extract is a cut, in which case it belongs in `NON_VERBATIM_CUTS` with "
                "a reason that says what it is instead")
    for name, wanted in sorted(exempt.items()):
        hits = len(wanted & seen)
        if hits == len(wanted):
            findings.append(
                f"{name} is exempt as a non-verbatim cut, but every one of its "
                f"{len(wanted)} rows IS in the pool byte for byte, so the exemption is "
                "excusing a recoverable extract and the corpus has fewer checked files than "
                "its count claims — remove the entry and let the recoverability check run")
    return findings, detail


def _indexed_names(vault: Path) -> set[str]:
    """The corpus family as the vault's git INDEX records it, by basename.

    The index, not a commit: `ls-files` is what the vault says it holds, independent of
    what is on disk, which is what makes it a second instrument. `--no-optional-locks`
    so the read never refreshes the index under a live vault.
    """
    import subprocess

    out = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(vault), "ls-files", "-z", "--",
         _PROMOTION_SUBDIR], capture_output=True, check=True).stdout
    names = set()
    for raw in out.split(b"\0"):
        rel = Path(raw.decode()) if raw else None
        if rel is not None and rel.parent.as_posix() == _PROMOTION_SUBDIR \
                and fnmatch(rel.name, _CORPUS_GLOB):
            names.add(rel.name)
    return names


def _presence_findings(data: Path, indexed: set[str]) -> list[str]:
    """Every dated extract the index lists that is not on disk, named by item id.

    One-directional on purpose: an extract on disk and not yet indexed is a round's
    work in progress, never a loss, and publishing extracts is a routine side effect of
    ordinary rounds — a guard that reddened on growth would gate every one of them.
    """
    on_disk = {path.name for path in data.glob(_CORPUS_GLOB)}
    if not indexed:
        return [f"the vault index lists no `{_CORPUS_GLOB}` file under {_PROMOTION_SUBDIR}, "
                "so there is no committed witness corpus for this file to guard at all"]
    missing = sorted(indexed - on_disk)
    if not missing:
        return []
    ids = sorted({(m.group(1) if (m := _ITEM_ID.match(name)) else name) for name in missing})
    return [f"{len(missing)} dated promotions extract(s) the vault index still lists are gone "
            f"from disk — items {', '.join('#' + i for i in ids)} ({', '.join(missing)}): a "
            "witness someone owes has left the tree"]


_EXTRACT_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.")


def _recency_finding(names: set[str], ledger_newest, *,
                     window_days: int = _RECENCY_WINDOW_DAYS) -> str | None:
    """Whether the newest dated extract trails the ledger's newest row by more than the window.

    `ledger_newest` is a `date`: the day of the live ledger's newest row. Reads each
    extract's date off its own NAME, the date every extract in this family carries.
    """
    from datetime import date

    dates = [date.fromisoformat(m.group(1)) for n in names if (m := _EXTRACT_DATE.match(n))]
    if not dates:
        return "no dated promotions extract is on disk, so the loop has published no witness"
    newest = max(dates)
    lag = (ledger_newest - newest).days
    if lag <= window_days:
        return None
    return (f"the loop keeps folding the ledger but has published no witness extract: the "
            f"newest dated promotions extract is {newest.isoformat()}, {lag} days behind the "
            f"ledger's newest row ({ledger_newest.isoformat()}), past the "
            f"{window_days}-day `_RECENCY_WINDOW_DAYS` window — a fold season with no witness")


def _ledger_newest_day(live: Path):
    """The `created_at` day of the live ledger's last row, read from the file's tail only."""
    import json
    from datetime import date

    with open(live, "rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 65536))
        tail = handle.read()
    for line in reversed([ln for ln in tail.split(b"\n") if ln.strip()]):
        try:
            stamp = json.loads(line).get("created_at") or ""
            return date.fromisoformat(str(stamp)[:10])
        except (ValueError, AttributeError):
            continue
    return None


# ── clause 1: discovery, and the population it reaches ─────────────────────


def test_the_corpus_glob_reaches_every_dated_promotions_extract():
    """Clause 1: the corpus pattern reaches BOTH naming families, and nothing silently does not.

    The shipped pattern `_promotion_extract_findings` uses cannot be the discovery here, and
    not for a stylistic reason: `fnmatch` treats the pattern's tail as literal, so
    `*.????-*promotions.jsonl` matches the `-lloyd-automod-promotions.jsonl` files and zero
    `-promotions-witness.jsonl` files, one of which (`2026-10-08.2383-...`) is a cut that
    clause 3 requires to be exempted BY NAME and re-checked for still existing. A node using
    the shipped pattern would report the narrower corpus, pass, and have dropped two extracts —
    the same silent narrowing this item is about, reproduced inside its own guard. So the
    first assertion is about the patterns themselves, where `fnmatch` can be asked, and the
    rest is about the directory.
    """
    assert not fnmatch(_WITNESS_FAMILY_SAMPLE, _SHIPPED_CORPUS_GLOB), (
        f"`{_SHIPPED_CORPUS_GLOB}` now matches a `{_WITNESS_FAMILY_SAMPLE}`-shaped name, so "
        "the reason this file widened its own pattern has gone and the two patterns should "
        "be collapsed back into one")
    assert fnmatch(_WITNESS_FAMILY_SAMPLE, _CORPUS_GLOB), (
        f"`{_CORPUS_GLOB}` no longer reaches the `-promotions-witness.jsonl` family, so "
        "2383 and 2079 have left the corpus without any node noticing")

    data = _data_dir(vault_root())
    assert data.resolve().is_relative_to(vault_root().resolve()), (
        f"{data} is not inside the vault, so the nodes below are reading some other tree")

    corpus = {path.name for path in data.glob(_CORPUS_GLOB)}
    shipped = {path.name for path in data.glob(_SHIPPED_CORPUS_GLOB)}
    assert shipped <= corpus, (
        f"{sorted(corpus ^ shipped)[:5]} are matched by one pattern and not the other in a "
        "direction that should not exist: the corpus pattern is a widening of the shipped "
        "one, so everything the sibling node sees has to be in scope here too")
    presence = _presence_findings(data, _indexed_names(vault_root()))
    assert presence == [], presence[0]
    assert corpus - shipped, (
        "no file the shipped pattern misses is on disk, so the `-promotions-witness.jsonl` "
        "family has gone and `NON_VERBATIM_CUTS`'s 2383 entry is naming a file that is not "
        "there — which the allow-list node also reports, and both readings mean the same loss")

    orphans = sorted(name for name in corpus if not name.endswith(_KNOWN_SUFFIXES))
    assert orphans == [], (
        f"{orphans[:3]} match the corpus glob but belong to no known suffix family: an "
        "unknown family would otherwise be counted in the corpus and never named as such")


# ── clause 1: the corpus-level pool proof, and clause 2's seam ─────────────


def test_every_verbatim_extract_is_recoverable_byte_for_byte_from_the_ledger_union():
    """Clause 1: every non-exempt extract's rows are ALL in the live ledger + rotations, byte for byte.

    `2026-10-11.2444-lloyd-automod-promotions.jsonl` is 82 rows and 0 of them are in the
    live file today, so this node cannot be satisfied by a live-only read; `2026-10-02.2079`
    is 2 rows and both are in a monthly archive. Byte-for-byte rather than parsed is the
    claim the archive makes about itself (§13: the gzip "holds the original line, byte for
    byte"), and it is the only claim that keeps the extract reproducible: rows that were
    re-spelled on the way into the rotation would satisfy a field comparison and no longer
    be findable by the `grep -F -f` proof every owed note quotes.

    The exempt extracts are checked in the same pass, in the opposite direction: each must
    come up SHORT. That is what makes `NON_VERBATIM_CUTS` a description of the files rather
    than a way of making the node green.
    """
    rows = _extract_rows(_data_dir(vault_root()))
    findings, detail = _corpus_findings(rows, live_ledger(), NON_VERBATIM_CUTS)
    assert len(detail) == len(rows), (
        f"only {len(detail)} of the {len(rows)} corpus hits reached the pool check, so the "
        f"rest were dropped by a name rule rather than by a stated exemption: {findings[:2]}")
    assert not findings, (
        f"{len(findings)} problem(s) across {len(detail)} extracts "
        f"({len(NON_VERBATIM_CUTS)} exempt as cuts): {findings[0]}")


def test_the_union_under_test_is_the_shipped_reader_behind_the_home_seam():
    """Clause 2: the population is `ledger_files(live_ledger())`, and it reaches a real rotation.

    Two separate failures are pinned here, and the second is the one that keeps the first
    honest. The population must be the shipped reader's — live ledger plus
    `LEDGER_ARCHIVE_GLOB`, the definition the landed-outcomes report uses, so this guard and
    that report can never be counting different stores. And the union must contain at least
    one monthly `.jsonl.gz` whenever this machine has one, because the blind alternative —
    asking `state.LEDGER_PATH`, which `tests/conftest.py` redirects to an empty scratch dir
    — returns a one-element list of a file that does not exist. A node reading THAT would
    skip in every gate run and read as green forever; this assertion is what makes passing
    blind impossible on a box that has rotations, and 26,417 of the 2026-10-11 extract rows
    only exist in the two rotations on this box.
    """
    from scripts.automod import state as S
    from scripts.automod.landed_outcomes import ledger_files

    live = live_ledger()
    union = _union_paths(live)
    assert union == ledger_files(live), (
        "_union_paths stopped being the shipped reader's answer, so this file is checking a "
        "population the landed-outcomes report does not count")
    assert union[0] == live, (
        f"the union's live component is {union[0]}, not the home ledger {live}: asking "
        "`state.LEDGER_PATH` instead is the redirect the conftest points at an empty scratch "
        "dir, and the node would skip in every gate run while reading as green")

    rotations = [path for path in union[1:]]
    on_disk = sorted(live.parent.glob(S.LEDGER_ARCHIVE_GLOB))
    assert rotations == on_disk, (
        f"the union's rotations are {[p.name for p in rotations]} while "
        f"`{S.LEDGER_ARCHIVE_GLOB}` under {live.parent} finds {[p.name for p in on_disk]}: "
        "one of the two is globbing somewhere else, which is the #3079 narrowing again")
    if on_disk:
        readable = [path for path in rotations if path.is_file()]
        assert readable, (
            f"{[p.name for p in on_disk]} exist under the home state dir but none is in the "
            "union under test, so the corpus check would have run against the live file "
            "alone and reported that as a pass")


# ── clause 3: the exemptions, and the wording they are not allowed to use ──


def test_the_non_verbatim_allow_list_is_exactly_the_cuts_and_each_names_a_live_file():
    """Clause 3: the exemptions are a named list of the cuts, rot-proofed in both directions.

    An entry has to name a file the corpus still discovers (a gone file fails the node, so
    the list cannot fossilise), has to say what the file IS in the words this list is
    written in, and must not rest on the one justification that decays — that a file matches
    the live ledger today. The fold only removes rows, so an allow-list that could grow on
    that basis would absorb the four still-live-complete extracts (3367, 3683, 3619, 3461)
    one at a time and leave the guard checking nothing while still counting 33 files. The
    fourth assertion names those four on purpose: they are checked, not excused.
    """
    data = _data_dir(vault_root())
    ids = {_ITEM_ID.match(name).group(1) for name in
           (path.name for path in data.glob(_CORPUS_GLOB)) if _ITEM_ID.match(name)}
    missing = sorted(set(NON_VERBATIM_CUTS) - ids)
    assert missing == [], (
        f"NON_VERBATIM_CUTS exempts {missing}, which the corpus glob no longer finds: an "
        "exemption for a file that has gone is a lie that keeps the node green")
    assert set(NON_VERBATIM_CUTS) == {"2383", "2759", "2992", "3205", "3753", "3756", "3684"}, (
        f"the allow-list is {sorted(NON_VERBATIM_CUTS)}, not the seven cuts #3924 measured "
        "as not-store-bytes on 2026-11-08 — an exemption added here has to be argued with a "
        "sentence about the file, not with a run that came out short")

    for item_id, reason in sorted(NON_VERBATIM_CUTS.items()):
        assert _NOT_STORE_BYTES in reason, (
            f"the {item_id} exemption does not say it is {_NOT_STORE_BYTES!r}: stating what "
            "the file is instead of store bytes is the whole content of an exemption")
        assert any(kind in reason for kind in _CUT_KINDS), (
            f"the {item_id} exemption names none of {_CUT_KINDS}, so it does not say what "
            "kind of cut it is")
        lowered = reason.lower()
        banned = [word for word in _FORBIDDEN_JUSTIFICATIONS if word in lowered]
        assert banned == [], (
            f"the {item_id} exemption rests on {banned}: the live file's answer is the "
            "decaying quantity this file exists to stop trusting, and the 14-day fold means "
            "an exemption justified that way is only ever getting deader")

    live_complete = {"3367", "3683", "3619", "3461"}
    assert not (live_complete & set(NON_VERBATIM_CUTS)), (
        "an extract that still matches the live ledger today has been exempted, which is the "
        "specific rot this assertion is here for")


# ── clause 4: the controls — the guard can see the difference, and can fail ─


def _fixture_extract(tmp_path: Path, rows: list[bytes]) -> Path:
    """A dated extract of `rows` in a throwaway vault directory.

    Item id 4999 is invented and is not in this vault (the real corpus tops out below it),
    so the fixtures can drive the same `_corpus_findings` the live nodes use without an
    allow-list entry of their own — and without ever writing into the real witness
    directory, whose bytes this round must not touch.
    """
    data = tmp_path / "vault" / _PROMOTION_SUBDIR
    data.mkdir(parents=True)
    (data / "2026-01-05.4999-lloyd-automod-promotions.jsonl").write_bytes(
        b"".join(row + b"\n" for row in rows))
    return data


def _fixture_rows(count: int) -> list[bytes]:
    import json

    return [json.dumps({"ts": 1790000000.0 + n, "created_at": "2026-01-05T00:00:00Z",
                        "event": "gate", "rung": "tests", "ok": True, "item_id": 4999,
                        "detail": f"fixture row {n}"}).encode()
            for n in range(count)]


def test_a_store_missing_one_extract_row_is_named_short_by_the_corpus_check(tmp_path):
    """Clause 4: the positive control — a pool with one row withheld makes the node name that extract.

    Three states of one fixture store, because "it failed" and "it always fails" are the
    same sentence to a reader who has only seen one of them, and a node that reads nothing
    has to be shown unable to pass:
    the full fixture is CLEAN (so the check is not rigged red), one row withheld names
    exactly that extract as 2 of 3, and a store holding none of the rows names it 0 of 3 —
    which is the shape `2444` has against the live file today, so the control is not a
    strawman, it is the live-only proof with the guard pointed at it.
    """
    rows = _fixture_rows(3)
    data = _fixture_extract(tmp_path, rows)
    live = tmp_path / "state" / "promotions.jsonl"
    live.parent.mkdir(parents=True)

    live.write_bytes(b"".join(row + b"\n" for row in rows))
    clean, detail = _corpus_findings(_extract_rows(data), live, {})
    assert clean == [], f"the complete fixture store did not come out clean: {clean}"
    assert detail == {"2026-01-05.4999-lloyd-automod-promotions.jsonl": (3, 3)}, detail

    live.write_bytes(b"".join(row + b"\n" for row in rows[:2]))
    findings, detail = _corpus_findings(_extract_rows(data), live, {})
    assert len(findings) == 1, f"withholding one row produced {len(findings)} findings: {findings}"
    assert "2026-01-05.4999-lloyd-automod-promotions.jsonl" in findings[0]
    assert "2 of its 3 rows" in findings[0], (
        "the finding does not count the shortfall, so a reader cannot tell which extract is "
        f"short by how much: {findings[0]!r}")
    assert detail["2026-01-05.4999-lloyd-automod-promotions.jsonl"] == (2, 3), detail

    live.write_bytes(b'{"ts": 1.0, "event": "gate", "unrelated": true}\n')
    findings, detail = _corpus_findings(_extract_rows(data), live, {})
    assert len(findings) == 1 and "0 of its 3 rows" in findings[0], (
        "a store holding none of the extract's rows did not read as a total shortfall, so "
        f"the guard cannot see the difference a live-only proof cannot: {findings}")


def test_the_live_file_alone_cannot_recover_the_corpus_it_is_checked_against():
    """Clause 4's other half, on the real store: the live file alone is short TODAY.

    Same rows, same check, one file narrower — the point being that the pool read and the
    live-only read are not interchangeable and the difference is already enormous. Measured
    2026-11-08: 22 of the 26 verbatim extracts are short against the live file alone, 2444's
    82 rows are all absent from it, and 3718 reads 226 of its published 254. The assertion
    is a floor, and the 14-day fold only ever moves it further true, so this node cannot be
    aged out by the very mechanism it demonstrates.

    Without it, a corpus node whose union read silently narrowed to the live file — the
    redirect, a gone glob, a store that stopped being mounted — could still return no
    findings on a smaller corpus, and the guard would read as green while proving nothing.
    """
    rows = _extract_rows(_data_dir(vault_root()))
    _, pool_detail = _corpus_findings(rows, live_ledger(), NON_VERBATIM_CUTS)
    _, live_detail = _corpus_findings(rows, live_ledger(), NON_VERBATIM_CUTS, live_only=True)

    checked = set(pool_detail)
    assert checked, "no extract reached the checks, so there is no corpus to compare"
    short = sorted(name for name in checked if live_detail[name][0] < live_detail[name][1])
    assert short, (
        f"every one of the {len(checked)} checked extracts is fully recoverable from the "
        "live ledger alone, so this node's control has stopped distinguishing the two reads "
        "and the corpus-level guard has nothing left to demonstrate")
    recovered_pool = sum(hits for hits, _ in pool_detail.values())
    recovered_live = sum(hits for name, (hits, _) in live_detail.items() if name in checked)
    assert recovered_pool > recovered_live, (
        f"the pool recovered {recovered_pool} rows and the live file {recovered_live} of "
        f"{sum(total for _, total in pool_detail.values())}: a pool read that is no wider "
        "than the live file is this item's bug, not its fix")
    assert len(short) >= 22, (
        f"only {len(short)} of {len(checked)} extracts are short against the live file, "
        "below the 22 measured 2026-11-08 — the fold only removes rows, so a drop here means "
        "rows were ADDED to the live ledger from the archives, which nothing in §13 does")


# ── clause 5: the read's shape, and the two branches around a missing store ─


def test_the_corpus_read_never_loads_a_store_file_whole(tmp_path, monkeypatch):
    """Clause 5: the corpus read streams, trapped rather than described.

    The three calls that WOULD load a store file whole are made to raise for the duration of
    the read: `Path.read_bytes`, `Path.read_text` and `gzip.decompress`. They are not an
    arbitrary sample — they are the two shapes this repo's own ledger readers use
    (`state._decode_ledger` is `read_text`, `landed_outcomes._decode_one` is
    `gzip.decompress(path.read_bytes())`), and reaching for the shipped decoder is exactly
    how a whole-file read gets reintroduced into a node that currently streams. The extract
    bytes are read BEFORE the trap, which is the file's real discipline: an extract is under
    `WITNESS_MAX_BYTES` by another node's assertion, the store is ~30x that.
    """
    rows = _fixture_rows(4)
    data = _fixture_extract(tmp_path, rows)
    live = tmp_path / "state" / "promotions.jsonl"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"".join(row + b"\n" for row in rows[:2]))
    (live.parent / "promotions-archive-202610.jsonl.gz").write_bytes(
        gzip.compress(b"".join(row + b"\n" for row in rows[2:])))

    parsed = _extract_rows(data)

    def no_whole_file_read(*args, **kwargs):
        raise AssertionError(
            "the corpus read loaded a ledger file whole, which WITNESS_MAX_BYTES = "
            "1,048,576 exists to forbid against a promotions ledger of tens of MB")

    monkeypatch.setattr(pathlib.Path, "read_bytes", no_whole_file_read)
    monkeypatch.setattr(pathlib.Path, "read_text", no_whole_file_read)
    monkeypatch.setattr(gzip, "decompress", no_whole_file_read)
    findings, detail = _corpus_findings(parsed, live, {})
    assert findings == [], f"the streamed fixture union did not come out clean: {findings}"
    assert detail["2026-01-05.4999-lloyd-automod-promotions.jsonl"] == (4, 4), (
        f"the streamed read returned {detail}, so the plain file and the gzip are not both "
        "being read line by line")


def test_the_corpus_check_skips_rather_than_fails_when_no_store_is_present(tmp_path):
    """Clause 5's two branches, asserted rather than promised.

    An empty state dir SKIPS: these nodes' subject is the committed extract, and a box with
    no automod state must not read as a broken witness. An ARCHIVE with no live ledger
    beside it RUNS the check: the folded rows are the majority of every October extract, so
    treating a missing live file as a missing store would go quiet exactly when the guard is
    most needed — mid-fold, which is the state this machine is in every month. Both branches
    are on a fixture because the first is unreachable on this box and the second is a race
    nobody can schedule.
    """
    rows = _fixture_rows(2)
    data = _fixture_extract(tmp_path, rows)
    parsed = _extract_rows(data)
    live = tmp_path / "empty-state" / "promotions.jsonl"

    with pytest.raises(pytest.skip.Exception) as caught:
        _corpus_findings(parsed, live, {})
    assert "no promotions store" in str(caught.value), (
        f"the absent-store branch raised {caught.value!r}, not the skip this file's wording "
        "promises: a reader would see a reason nobody wrote")

    live.parent.mkdir(parents=True)
    (live.parent / "promotions-archive-202609.jsonl.gz").write_bytes(
        gzip.compress(b"".join(row + b"\n" for row in rows)))
    findings, detail = _corpus_findings(parsed, live, {})
    assert findings == [], (
        f"an archive with no live ledger beside it did not come out clean, so mid-fold the "
        f"guard goes wrong instead of running: {findings}")
    assert detail["2026-01-05.4999-lloyd-automod-promotions.jsonl"] == (2, 2), detail


# ── #4584: presence without a hand-typed size, and freshness ───────────────


def test_an_extract_lost_from_disk_is_named_missing_against_the_vault_index(tmp_path):
    """The presence alarm still catches the loss it exists for, with no count anywhere.

    A constructed vault: three dated extracts committed, so disk and index agree and the
    check is clean; one removed from disk while the index still lists it, and the finding
    names that item; one added on disk and not yet indexed, and nothing reddens.
    """
    import subprocess

    vault = tmp_path / "vault"
    data = vault / _PROMOTION_SUBDIR
    data.mkdir(parents=True)
    names = ["2026-01-05.4997-lloyd-automod-promotions.jsonl",
             "2026-01-06.4998-lloyd-automod-promotions.jsonl",
             "2026-01-07.4999-promotions-witness.jsonl"]
    for name in names:
        (data / name).write_bytes(_fixture_rows(1)[0] + b"\n")
    git = ["git", "-C", str(vault), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "seed"], check=True)

    assert _indexed_names(vault) == set(names)
    assert _presence_findings(data, _indexed_names(vault)) == []

    (data / names[1]).unlink()
    found = _presence_findings(data, _indexed_names(vault))
    assert len(found) == 1 and "#4998" in found[0] and names[1] in found[0], found
    assert "#4997" not in found[0] and "#4999" not in found[0], found

    (data / names[1]).write_bytes(b"{}\n")
    (data / "2026-01-08.5000-lloyd-automod-promotions.jsonl").write_bytes(b"{}\n")
    assert _presence_findings(data, _indexed_names(vault)) == [], (
        "an extract on disk that is not yet indexed is growth, never a loss")


def test_the_loop_has_published_a_witness_extract_within_the_window_of_its_ledger():
    """The freshness alarm on the live tree: the newest dated extract's own date is within
    `_RECENCY_WINDOW_DAYS` of the live ledger's newest row. Skips where this box has no
    live ledger, like every store-reading node here: no ledger, nothing folding."""
    live = live_ledger()
    if not live.is_file():
        pytest.skip(f"no live promotions ledger at {live}: nothing is folding to owe a witness")
    newest = _ledger_newest_day(live)
    assert newest is not None, f"{live}'s tail carries no row with a `created_at` day"
    names = {path.name for path in _data_dir(vault_root()).glob(_CORPUS_GLOB)}
    finding = _recency_finding(names, newest)
    assert finding is None, finding


def test_the_recency_alarm_passes_inside_its_window_and_fails_beyond_it():
    """Both directions on constructed corpora, so the guard is pinned without waiting for
    the wall clock (or the ledger) to reach the window."""
    from datetime import date, timedelta

    ledger_day = date(2026, 1, 31)
    edge = ledger_day - timedelta(days=_RECENCY_WINDOW_DAYS)
    inside = {f"{edge.isoformat()}.4998-lloyd-automod-promotions.jsonl",
              "2025-12-01.4997-lloyd-automod-promotions.jsonl"}
    assert _recency_finding(inside, ledger_day) is None, (
        "an extract exactly at the window's edge is inside it")

    stale = edge - timedelta(days=1)
    beyond = {f"{stale.isoformat()}.4998-lloyd-automod-promotions.jsonl",
              "2025-12-01.4997-promotions-witness.jsonl"}
    finding = _recency_finding(beyond, ledger_day)
    assert finding is not None and stale.isoformat() in finding, finding
    assert "keeps folding the ledger but has published no witness extract" in finding, finding
    assert _recency_finding(set(), ledger_day) is not None, "an empty corpus is never fresh"


# ── the rail on this file's own prose ───────────────────────────────────────


def test_the_docstring_names_every_node_this_file_ships():
    """The review asks whether the prose can fail; this one can, and it keeps the numbers above honest.

    The count and every node name are checked against the module's bytes. Retyping a number
    fixes one drift and leaves the next; a count this file recomputes is the only version
    that stays true when a node is added or renamed, and the figures in the docstring are
    the ones a reader who never runs pytest will quote.
    """
    doc = sys.modules[__name__].__doc__ or ""
    names = sorted(name for name, value in globals().items()
                   if name.startswith("test_") and callable(value)
                   and getattr(value, "__module__", None) == __name__)
    unnamed = [name for name in names if name not in doc]
    assert unnamed == [], (
        f"{unnamed} ship in this file but are not named in its docstring, so the paragraph "
        "that enumerates them is now an undercount")
    stated = re.search(r"It ships (\d+) nodes", doc)
    assert stated, "the docstring no longer states its node count as `It ships N nodes`"
    assert int(stated.group(1)) == len(names), (
        f"the docstring states {stated.group(1)} nodes and this module defines "
        f"{len(names)}: the enumeration is prose, and only this node keeps it true")
    listed = sorted(set(re.findall(r"`(test_[a-z0-9_]+)`", doc)))
    assert listed == names, (
        f"the docstring lists {sorted(set(listed) - set(names))} that no longer exist "
        f"or omits {sorted(set(names) - set(listed))}")
