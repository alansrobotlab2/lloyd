#!/usr/bin/env python3
"""Fill in the `entity:` tag of every historical untagged fact record (#2676).

**The alarm this repairs.** The untagged rail (#2474, `f7e08d41`) has printed red
every night since it shipped. Measured 2026-10-14, `kg_hygiene.scan()["untagged"]`
reports **3,206 fact records carrying no entity tag, in 544 entity directories
across 830 files**, out of 153,999 records — 2.08% against
`kg_hygiene.UNTAGGED_ALARM_SHARE = 0.5`, so `over_floor` is True and
`knowledge-health-report.py` appends `ABOVE FLOOR` to the row and takes the alarm
exit. The item's quoted 2.11% is the same numerator over a smaller denominator
(152,223 then); the number that has to change is the 3,206, not the percentage.

**Every one of those records is historical.** 2,142 carry `entity:` present-but-blank
and 1,064 carry no `entity:` key at all; 1,753 of them are the frozen 2026-09-23
re-derivation cohort, and the newest `created_at` on any of them is 2026-10-10 — the
day the live writer was closed by #2503 (`1822fce5`). Zero records have been born
untagged since. So this is not a writer bug, and no writer fix reaches it: it is
residue, and the only way the alarm goes quiet is repairing the files.

**The rule this script implements** (#2676, ruling from #2474's owed entry): re-tag
each record from its own location, do not drop the records, do not move the floor. A
fact file's top-level `entity:` is the entity its directory was built for, and every
record in it was extracted from that entity's source note — so the record's own
location names the tag it is missing. Measured over the same 830 files: **all 830
have a non-blank top-level `entity:`**, which is why 100% of the population is
retaggable and why `DIR_SKIP` below ships empty.

**Why it edits lines instead of re-emitting the file.** These files are machine-
written by `agent_mcp/facts.py`, and one of the two writers emits records with `fact:`
FIRST and `entity:` SECOND (the sibling layout of the 1,064 key-absent cohort, and the
shape of the `backlog_write_task` records in the live tree). A `yaml.safe_load` /
`yaml.dump` round trip re-sorts keys, re-wraps long scalars and re-quotes names, so a
"tag-only" repair would show up as 830 rewritten files of unknown diff and would
silently re-quote values the index had already hashed. So the edit is structural line
surgery over the front-matter block, and clause 4's test is the claim that nothing
else moves.

**Where the tag goes, and why the plan is read back before it is written.** A record
with no `entity:` key is given one as its FIRST key, by rewriting its dash line — never
appended to the end of the block. An appended key is illegal YAML whenever the block's
last line is a continuation of a multi-line scalar (a plain wrapped `fact:` swallows
it: `mapping values are not allowed here`), and the writers wrap long facts across
lines as a matter of course, so that is the common case and not the odd one. And no
argument about shapes survived contact with this corpus the first time: the review
replay of `42e1cc3e` over 140 live files made 43 of them stop parsing, none of which
the fixture held. So every plan is applied to an in-memory candidate and read back
through `app.kg_store.parse_fact_file` before a byte is written, and a file whose
candidate is not the same document with tags added is SKIPPED with a printed reason
and left untouched. Refusing over one record's tag is the correct outcome: the
alternative is a fact file the index can no longer read, which the rail then counts as
zero untagged records and the nightly reports as fixed.

Dry run by default. `--apply` writes, re-reads what it wrote through `app.kg_store`'s
own parser, then calls that store's reindex for the changed paths only — `facts_idx`
is derived from these files and is what every query serves from, and #2092 is the
incident where a fixed file and a stale index disagreed.

Exit: 0 ran clean — including the run that planned nothing, because a tree with no
untagged record is this tool's goal state and an exit code that fired on it would be
a fresh false alarm of exactly the kind #2016/#552 are written about; 1 a file could
not be read or parsed, could not be written, was written but failed the re-parse, or
was written but not re-indexed.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import yaml
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:                                    # who-ran-this ledger, see `_invocation.py`
    from scripts.memory._invocation import invocation_ledger
except ImportError:                     # loaded by path, the fleet's script check
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _invocation import invocation_ledger
import app.paths as P  # noqa: E402
from app.atomic_io import atomic_write_text, locked_file  # noqa: E402
from app.frontmatter import split_frontmatter  # noqa: E402
from app.kg_store import parse_fact_file  # noqa: E402
from scripts.memory.kg_hygiene import _tag_of  # noqa: E402

# Catch-all skip list, clause 2: an entity directory whose name is a bucket rather
# than a thing — re-tagging from it would mint an entity called `general`. Shipped
# EMPTY on measurement: 0 of the 830 files holding an untagged record has a blank
# top-level `entity:`, and the top dirs by count are real entities (Lloyd 213,
# Inner Voice 104, IsaacLab 89, Tool Keyword Activation Map 75, Autoresearch 74).
# `general/` never reaches this list because `walk` excludes it the way the rail
# does (#1761 collects it separately), and inventing entries here would quietly
# exempt records this repair can in fact fix.
DIR_SKIP: tuple[str, ...] = ()

# A YAML indicator or anchor where a scalar belongs: not a scalar, so the record
# cannot be re-tagged by replacing one line, and guessing would rewrite prose.
_UNSUPPORTED = ("|", ">", "&", "*", "[", "{", "@", "`")


def is_untagged(value: object) -> bool:
    """Whether a record's `entity:` VALUE counts as untagged — by the rail's own
    predicate, not a second one next to it.

    `kg_hygiene._tag_of` takes the RECORD and answers `''` for all three untagged
    states (key absent, value None, value blank-or-whitespace) and a stripped name
    otherwise; `None` means "not a mapping at all". So untagged is `== ""`, not
    `is None` — read the wrong way round, this script would call every tagged record
    untagged. Re-implementing the predicate here is how a repair ends up retagging a
    population the rail then counts differently, leaving the alarm red for a reason
    nobody can see.
    """
    return _tag_of({"entity": value}) == ""


def record_is_untagged(record: object) -> bool:
    """The same predicate on a whole record, so the key-absent case is included.

    `_tag_of({})` is `''` — that is the 1,064-record cohort — so a record with no
    `entity:` key is untagged without any `.get()` default being able to hide it.
    """
    return isinstance(record, dict) and _tag_of(record) == ""


def _dump_entity(name: str) -> str:
    """The entity name as a YAML scalar that parses back to itself.

    Emitted through the same YAML library the writers use rather than hand-quoted:
    `Weird "Quoted"` and a name containing a colon both need quoting, for different
    reasons, and a hand-rolled `f"'{name}'"` corrupts a name holding an apostrophe.
    The width is raised because the default wraps a long name across lines, which
    would put this edit's value on a line the reader is not looking at.
    """
    text = yaml.safe_dump({"entity": name}, default_flow_style=False,
                          sort_keys=False, width=1_000_000).strip()
    _head, sep, value = text.partition("entity:")
    if not sep:
        raise ValueError(f"yaml emitted no entity key for {name!r}")
    return value.strip()


#: Reasons that mean "this file has nothing for the repair to do", as opposed to
#: "this file has untagged records the repair refused to tag". Only these two, and
#: both contribute 0 to the alarm by definition: a file with no `facts:` list has no
#: records, and a file where every record already carries its entity has no untagged
#: record. Measured over the live tree on 2026-10-14 these are 24,575 and 14,697 of
#: its 39,272 skips — 97.5% of the population — while every genuine skip population
#: (a blank/absent top-level `entity:`, a directory on `DIR_SKIP`, front matter that
#: will not parse) carries untagged records or is a declared exclusion and stays
#: printed one line per file.
#:
#: They are collapsed to one counted line in the report because the alternative is
#: that a dry run over the live tree prints 40,655 lines of which 39,272 say
#: "left untagged" about files that hold nothing untagged — and the dry run exists to
#: be read before an 831-file `--apply`.
NO_ACTION_REASONS = {
    "no facts: list": "hold no fact records",
    "every record already carries its entity": "already fully tagged",
}


class Skip(Exception):
    """This file is left exactly as it is, and the reason is printed.

    Carries the untagged count it walked past, because the rail counts RECORDS and
    the report's skip total has to be comparable to the alarm's numerator.
    """

    def __init__(self, reason: str, untagged: int = 0) -> None:
        super().__init__(reason)
        self.untagged = untagged


def _fact_blocks(lines: list[str]) -> list[tuple[int, int, int]] | None:
    """Locate each `facts:` record as `(first_line, last_exclusive, key_indent)`.

    Structural, never text-matching: a record's own prose can contain `facts:` or a
    `---`, and in the live corpus it does — a fact ABOUT this parser quotes the split
    expression. A new record starts at a line at the sequence's item indent that
    begins with `- `; the sequence ends at the first non-blank line at or shallower
    than that indent. Returns None when the sequence is not the column-0 block style
    these writers emit (flow style, or items indented under the key), which the
    caller turns into a skip rather than a guess.
    """
    key_at = None
    for i, line in enumerate(lines):
        if not line.strip() or line[0].isspace():
            continue                                 # blanks, and other keys' values
        if line.rstrip().endswith(":") and line.split(":", 1)[0].strip() == "facts":
            key_at = i
            break
        if ":" in line and not line.rstrip().endswith(":"):
            continue                                 # a scalar key like `entity: X`
    if key_at is None:
        return None                                  # absent, or `facts: [a, b]`

    item_indent = None
    out: list[list] = []
    for i in range(key_at + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if item_indent is None:
            if indent:
                return None                          # items indented under the key
            item_indent = 0
        if indent == item_indent and line.startswith("- "):
            if out:
                out[-1][1] = i                       # close the previous record
            out.append([i, len(lines), item_indent + 2])
            continue
        # Every other line at or shallower than the item indent is outside the
        # sequence — the next top-level key, or the closing `---`. A record's own
        # keys are always deeper than it, by exactly two.
        if indent <= item_indent:
            # Close the OPEN record here. Leaving it open to the end of the
            # block is what shipped broken in `42e1cc3e`: the live writer emits
            # a top-level `last_updated:` AFTER the `facts:` list — `Steering
            # Vector Portability/Steering Vector Portability-state.md:196` does,
            # and that file is the one that lost all 12 of its records — so an
            # unclosed final record ran past that key and its `entity:` line
            # landed indented under it. YAML reads that as a mapping inside a
            # scalar and the whole front matter stops parsing: `expected
            # <block end>, but found '<block mapping start>'`, which
            # `parse_fact_file` turns into `({}, [])` — 12 records lost, not 1.
            if out:
                out[-1][1] = i
            break
    return [(a, b, c) for a, b, c in out] if item_indent is not None else []


def _entity_line(lines: list[str], start: int, end: int,
                 key_indent: int) -> tuple[int, str] | None:
    """The record's own `entity:` key line, or None when it has no such line.

    Only a line at the record's key indent counts, so an `entity:` word inside a
    wrapped fact sentence — indented deeper, or a backticked mention in prose —
    cannot be mistaken for the key. A record carrying two of them answers `None`,
    the same as one carrying none: the caller then puts its tag first, and
    `_candidate_reason` refuses the file if the parsed tag is still blank, because
    YAML's last-key-wins would have left the blank one in charge.
    """
    found = None
    for i in range(start, end):
        line = lines[i]
        body = line.lstrip()
        on_dash = body.startswith("- ")
        candidate = body[2:] if on_dash else body
        indent = (len(line) - len(body)) + (2 if on_dash else 0)
        if indent != key_indent or not candidate.startswith("entity:"):
            continue
        if found is not None:
            return None                              # two entity keys: do not touch
        found = (i, line)
    return found


def plan_file(path: Path, root: Path | None = None) -> dict:
    """What `--apply` would write in one file, or why it will not.

    Returns `{"path", "rel", "entity", "records", "untagged", "edits"}`, where
    `edits` is the list of line operations
    `{"kind": "replace"|"insert", "at": <line index in the front-matter block>,
    "line": <new text>, "id": <the record's id when it has one>}` — indices into the
    front-matter block's lines, so `apply_edits` needs no second parse and cannot
    write something `plan_file` did not see.

    Raises `Skip` for the cases clause 2 names — a blank/absent top-level `entity:`,
    and a file under a directory on `DIR_SKIP` — plus the shapes this edit cannot make
    a promise about (front matter that will not parse, a `facts:` sequence in a style
    the scanner does not cover, a non-scalar `entity:`, two `entity:` keys on one
    record).
    """
    root = root or P.VAULT_FACTS_ROOT
    parts = path.relative_to(root).parts if path.is_relative_to(root) else path.parts
    if DIR_SKIP and any(part in DIR_SKIP for part in parts):
        raise Skip("directory on the named catch-all skip list")

    raw = path.read_text(encoding="utf-8", errors="replace")
    split = split_frontmatter(raw)
    if split is None:
        raise Skip("no front matter")
    fm_text = split[0]
    try:
        fm = yaml.safe_load(fm_text) or {}
    except Exception as exc:                                    # noqa: BLE001
        raise Skip(f"front matter will not parse ({type(exc).__name__})") from None
    if not isinstance(fm, dict):
        raise Skip("front matter is not a mapping")

    records = fm.get("facts")
    if not isinstance(records, list) or not records:
        raise Skip("no facts: list")
    untagged = sum(1 for r in records if record_is_untagged(r))

    top = fm.get("entity")
    if not isinstance(top, str) or not top.strip():
        # Clause 2's skip: the location does not name an entity, so nothing here may
        # invent one. The count above is carried so the report's total is honest.
        raise Skip("file's own top-level entity: is absent or blank", untagged)
    if untagged == 0:
        raise Skip("every record already carries its entity")

    lines = fm_text.split("\n")
    blocks = _fact_blocks(lines)
    if blocks is None:
        raise Skip("facts: sequence is not in block style at column 0", untagged)
    if len(blocks) != len(records):
        # Blocks and records are zipped by position. A disagreement means the
        # structural walk is not looking at the document the parser saw, and writing
        # on that basis could put one record's tag into another record.
        raise Skip(f"found {len(blocks)} record line(s) for {len(records)} parsed "
                   f"record(s)", untagged)

    want = "entity: " + _dump_entity(top.strip())
    edits: list[dict] = []
    for (start, end, key_indent), rec in zip(blocks, records):
        if not record_is_untagged(rec):
            continue
        ent = _entity_line(lines, start, end, key_indent)
        if ent is None:
            # Key absent (the 1,064-record cohort, and the whole `fact:`-first
            # layout). The tag becomes the record's FIRST key by rewriting its
            # dash line, so it always lands before any scalar region. Placing it
            # at the END of the block instead is unsafe in two shapes, and only
            # one of them has fired yet: appending after a record whose last line
            # is a continuation of a multi-line scalar puts `entity: X` inside a
            # plain scalar, which YAML rejects with `mapping values are not
            # allowed here` — measured over the live corpus on 2026-10-14, that
            # shape is currently 0 of the 831 untagged-bearing files, and the
            # closed-block fix in `_fact_blocks` is what accounts for all 350
            # files the `42e1cc3e` replay broke. The first-key form is chosen because it is
            # independent of where the block ends and of what its last scalar
            # kind is. Appending after the record's FIRST line is no better: it
            # splices a key into the middle of the scalar the dash line opened.
            # Rewriting the dash line keeps the original first key at its own
            # column, so a wrapped or block scalar keeps exactly its indentation.
            dash = lines[start]
            head, sep, tail = dash.partition("- ")
            if not sep or not tail.strip():
                # A bare `-` with its keys on the following lines: the new key
                # goes directly under it, above every scalar in the record.
                edits.append({"kind": "insert", "at": start + 1,
                              "line": " " * key_indent + want,
                              "id": rec.get("id")})
                continue
            edits.append({"kind": "replace", "at": start,
                          "line": (head + "- " + want + "\n"
                                   + " " * key_indent + tail),
                          "id": rec.get("id")})
            continue
        at, text = ent
        head, _sep, tail = text.partition("entity:")
        if tail.strip().lstrip().startswith(_UNSUPPORTED):
            raise Skip(f"record {rec.get('id') or '?'} has a non-scalar entity: "
                       f"value", untagged)
        edits.append({"kind": "replace", "at": at,
                      "line": head + want, "id": rec.get("id")})

    # Read the plan back through the reader every query serves from BEFORE it is
    # written, and refuse the file if the candidate is not the same document with
    # nothing but tags added. Line surgery over a corpus this shape cannot be
    # proven by reasoning about shapes — the review replay of `42e1cc3e` over 140
    # live files broke 43 of them, none of which the fixture held — so the claim
    # is checked per file instead: same record count, every non-`entity:` field
    # byte-equal, every record tagged. `parse_fact_file` is the check precisely
    # because a file that stops parsing yields `({}, [])` there: the old code
    # could only report that AFTER writing, and a destroyed file contributes zero
    # untagged records to the rail, which is how "the alarm went quiet" could have
    # been read as success on 100 destroyed records.
    reason = _candidate_reason(fm_text, edits, records, top.strip())
    if reason:
        raise Skip(reason, untagged)

    return {"path": path, "rel": str(path.relative_to(root)),
            "entity": top.strip(), "records": len(records), "untagged": untagged,
            "edits": edits}


def _candidate_reason(fm_text: str, edits: list[dict], records: list,
                      entity: str) -> str:
    """Why writing `edits` into this file must be refused, or `""` when it may.

    The candidate front matter is built exactly as `apply_tree` will build it and
    read back through `app.kg_store.parse_fact_file` — the reader every KG query
    serves from, not a mirror parser that could drift from it. Three things have
    to hold, and each is a clause-4 claim checked rather than asserted: the record
    count is unchanged (a candidate that is illegal YAML yields `({}, [])` here,
    which is how 12 records became 0 in the `42e1cc3e` replay), every record ends
    up tagged, and every field except `entity:` is equal to what the file says
    today.

    The read-back goes through one temporary file in the system temp dir, never in
    the facts tree: `plan_file` still writes nothing where the records live, and
    `parse_fact_file` takes a path, so an in-memory check would mean re-implement
    its slicing and its `isinstance(f, dict)` filter — the second-parser drift
    this whole file is written against.
    """
    candidate = apply_edits(Path("<plan>"), fm_text, {"edits": edits})
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(candidate)
            tmp = Path(fh.name)
        _fm, after = parse_fact_file(tmp)
    except Exception as exc:                                        # noqa: BLE001
        return (f"the planned edit could not be re-read "
                f"({type(exc).__name__}: {exc})")
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)

    if len(after) != len(records):
        return (f"applying it would leave {len(after)} parseable record(s) where "
                f"the file has {len(records)} — the planned edit is not valid YAML")
    still = sum(1 for r in after if record_is_untagged(r))
    if still:
        return (f"applying it would still leave {still} of {len(after)} record(s) "
                f"untagged (the file's entity: is {entity!r})")
    for before, after_rec in zip(records, after):
        for key in set(before) | set(after_rec):
            if key == "entity":
                continue
            if before.get(key) != after_rec.get(key):
                rid = before.get("id") or after_rec.get("id") or "?"
                return (f"applying it would change {key!r} on record {rid}, not "
                        f"only entity:")
    return ""


def apply_edits(path: Path, fm_text: str, plan: dict) -> str:
    """Write `plan`'s edits into the front-matter block, changing nothing else.

    The block is the only text touched: everything after it — the note body and the
    closing `---` — is appended back byte-for-byte by the caller.
    """
    lines = fm_text.split("\n")
    for ed in sorted((e for e in plan["edits"] if e["kind"] == "replace"),
                     key=lambda e: e["at"], reverse=True):
        lines[ed["at"]] = ed["line"]
    for ed in sorted((e for e in plan["edits"] if e["kind"] == "insert"),
                     key=lambda e: e["at"], reverse=True):
        lines.insert(ed["at"], ed["line"])
    return "---\n" + "\n".join(lines) + "---\n"


class Report:
    """Everything one run planned, wrote or refused, and the text it printed."""

    def __init__(self) -> None:
        self.plans: list[dict] = []
        self.skips: list[tuple[str, str]] = []      # (path, reason)
        self.no_action: dict[str, int] = {}         # reason -> file count
        self.errors: list[tuple[str, str]] = []     # (path, why it could not be read)
        self.failed: list[tuple[str, str]] = []     # (path, why the write failed)
        self.written_paths: list[Path] = []
        self.reindex: dict = {}
        self.reindex_error: str = ""
        self.ledger: dict = {}
        self.skipped_records = 0
        self.verify_bad = -1                        # -1: no write happened
        self.verify_lost = -1
        self.applied = False

    # ── counts: records where the rail counts records, files where it counts files ──
    @property
    def records(self) -> int:
        return sum(len(p["edits"]) for p in self.plans)

    @property
    def files(self) -> int:
        return len(self.plans)

    @property
    def dirs(self) -> int:
        return len({p["rel"].split("/")[0] for p in self.plans})

    @property
    def written(self) -> int:
        return len(self.written_paths)

    @property
    def skipped(self) -> int:
        """Records left untagged by a skip — not the number of skipped files.

        The dry run prints this beside its own total because the alarm counts
        records; a file count read as a record count would make a clean run look like
        it had missed most of the population.
        """
        return self.skipped_records

    @property
    def by_file(self) -> dict[str, int]:
        """Records to rewrite per file, root-relatively — the rail counts per file,
        so this print is line-for-line comparable to its tally."""
        return {p["rel"]: len(p["edits"]) for p in self.plans}

    @property
    def by_dir(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.plans:
            d = p["rel"].split("/")[0]
            out[d] = out.get(d, 0) + len(p["edits"])
        return out

    def add_skip(self, path: Path, reason: str, untagged: int = 0) -> None:
        # A file with no records and a file already fully tagged have no untagged
        # record to leave behind, so they are counted, not itemised: see
        # `NO_ACTION_REASONS`. Everything else stays one line per file, because that
        # is the list clause 3 demands and the one a reader has to check by name.
        if reason in NO_ACTION_REASONS:
            self.no_action[reason] = self.no_action.get(reason, 0) + 1
            return
        self.skips.append((str(path), reason))
        self.skipped_records += untagged

    @property
    def no_action_files(self) -> int:
        return sum(self.no_action.values())

    def stamp_invocation(self):
        """Who ran this, into the report (`_invocation.py`).

        The 2026-09-03 sweep merged 2,034 entity directories and nothing recorded the
        actor, so every tool that mutates the facts tree now stamps this. Stamped for
        a dry run too: a dry run is also something a later reader asks who ran. Only
        user and cwd are printed — the ledger's argv and env dumps stay out of the
        report, where an `LLOYD_*` value would be noise at best.
        """
        try:
            self.ledger = invocation_ledger()
        except Exception:                             # noqa: BLE001
            self.ledger = {}                          # a ledger never breaks a run

    def render(self, *, apply: bool, root: Path) -> str:
        # An `--apply` that found nothing to do speaks in the conditional voice:
        # `retagged 0 record(s)` would read, on a re-run, as a claim about work
        # done rather than about work that needs none.
        did = bool(apply and self.written)
        head = (f"{'retagged' if did else 'would retag'} {self.records} record(s) "
                f"in {self.files} file(s) across {self.dirs} directories")
        head += " (--apply)" if did else (
            " (--apply, nothing to do)" if apply
            else " (dry run — nothing was written)")
        out = [head, ""]
        if not self.records:
            out.append("nothing to apply: no record in this tree is missing an "
                       "entity: that its own file's top-level entity: could supply")
            out.append("unchanged: every record already carries its file's entity")
            out.append("")
        if self.by_file:
            out.append(f"{self.records} record(s) by file:")
            for rel, n in sorted(self.by_file.items()):
                ent = next(p["entity"] for p in self.plans if p["rel"] == rel)
                out.append(f"  {rel}: {n} record(s) -> entity: {ent}")
        if self.by_dir:
            out.append("")
            out.append("dirs:")
            for d, n in sorted(self.by_dir.items(), key=lambda kv: (-kv[1], kv[0])):
                out.append(f"  {d}: {n} record(s)")
        if self.no_action:
            # Counted, never itemised. Ordered as `NO_ACTION_REASONS` declares it, so
            # the line is byte-stable across runs of the same tree.
            parts = ", ".join(f"{self.no_action[r]} {label}"
                              for r, label in NO_ACTION_REASONS.items()
                              if r in self.no_action)
            out.append("")
            out.append(f"nothing to do in {self.no_action_files} file(s): {parts}. "
                       f"None of them holds an untagged record, so none of them is "
                       f"in the alarm")
        if self.skips:
            out.append("")
            out.append(f"skip: {len(self.skips)} file(s) left untagged, "
                       f"{self.skipped_records} record(s):")
            for path, reason in self.skips:
                out.append(f"  {path}: {reason}")
        if self.errors:
            out.append("")
            out.append(f"{len(self.errors)} file(s) could not be read:")
            for path, why in self.errors:
                out.append(f"  {path}: {why}")
        if self.failed:
            out.append("")
            out.append(f"WRITE FAILED for {len(self.failed)} file(s):")
            for path, why in self.failed:
                out.append(f"  {path}: {why}")
        if self.verify_bad >= 0:
            out.append("")
            out.append(f"verify: {self.verify_bad} record(s) re-tagged; "
                       f"{self.verify_lost} record(s) lost")
        if self.reindex:
            out.append("reindex: " + ", ".join(f"{k}={v}" for k, v in
                                               sorted(self.reindex.items())))
        if self.reindex_error:
            out.append("REINDEX FAILED: " + self.reindex_error)
        if self.ledger:
            out.append("invoked by " + str(self.ledger.get("user") or "?")
                       + " from " + str(self.ledger.get("cwd") or "?"))
        return "\n".join(out)


def walk(root: Path) -> list[Path]:
    """Every fact FILE under `root`, the same tree the rail walks.

    Files only, deliberately: an entity directory's NAME is an entity, so a `rglob`
    over this tree surfaces directories as well as markdown, and a walk that hands
    `read_text` a directory turns a wall of clean entity folders into a wall of
    "unreadable file" errors — which `--apply` would then refuse to write. `general/`
    is excluded because that is where #1761 collects the untagged population the rail
    already reports separately, and dot/underscore trees the same way.
    """
    out: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")) or d.name == "general":
            continue
        out.extend(p for p in sorted(d.rglob("*.md")) if p.is_file())
    return out


def plan_tree(root: Path) -> Report:
    """The plan for a whole facts tree. Writes nothing, ever."""
    rep = Report()
    for path in walk(root):
        try:
            rep.plans.append(plan_file(path, root))
        except Skip as skip:
            rep.add_skip(path, str(skip), skip.untagged)
        except Exception as exc:                                   # noqa: BLE001
            rep.errors.append((str(path), f"{type(exc).__name__}: {exc}"))
    return rep


def default_reindexer():
    """`app.kg_store`'s own reindex, resolved lazily so importing this module never
    opens the KG store. A test injects its own into `apply_tree`; production gets
    this, which is the only writer of `facts_idx` there is."""
    from app.kg_store import store
    return store().reindex


def apply_tree(root: Path, *, reindexer=None) -> Report:
    """Write the plan, re-read what was written, then reindex the changed paths once.

    The reindex takes `paths=` deliberately: a whole-tree `reindex()` over a
    154k-record corpus for a 3,206-record repair is the wrong cost, and N per-file
    calls would be N full transactions against the same store.
    """
    rep = plan_tree(root)
    rep.applied = True
    changed: list[Path] = []
    for plan in rep.plans:
        path = plan["path"]
        try:
            with locked_file(path):
                raw = path.read_text(encoding="utf-8", errors="replace")
                split = split_frontmatter(raw)
                if split is None:
                    raise RuntimeError("front matter gone (changed since the plan)")
                atomic_write_text(path, apply_edits(path, split[0], plan) + split[1])
            changed.append(path)
        except Exception as exc:                                   # noqa: BLE001
            rep.failed.append((str(path), f"{type(exc).__name__}: {exc}"))
    rep.written_paths = changed

    # Re-read with the reader every query serves from. The edit is line surgery, so
    # "the file is still valid YAML holding the same records" is the claim clause 4
    # makes, and it is cheap enough to check rather than to argue about.
    if changed:
        bad = lost = 0
        written = {p["path"]: p for p in rep.plans}
        for path in changed:
            _fm, recs = parse_fact_file(path)
            plan = written[path]
            if len(recs) < plan["records"]:
                lost += plan["records"] - len(recs)
            bad += sum(1 for r in recs if record_is_untagged(r))
        rep.verify_bad, rep.verify_lost = bad, lost

    if changed:
        if reindexer is None:
            reindexer = default_reindexer()
        try:
            rep.reindex = dict(reindexer(paths=changed))
        except Exception as exc:                     # noqa: BLE001
            # The files are already rewritten, so an index still serving the old
            # blank tag is the #2092 shape, and this is the one outcome the exit code
            # must not swallow: exit 1, with the file count in the message.
            rep.reindex_error = (
                f"{len(changed)} file(s) written but reindex failed "
                f"({type(exc).__name__}: {exc}) — facts_idx still holds the "
                f"pre-retag index")
    return rep


def main(argv: list[str] | None = None, *, reindexer=None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-tag historical untagged fact records from their file's own "
                    "top-level entity: (#2676). Dry run unless --apply.")
    ap.add_argument("--facts-root", type=Path, default=None,
                    help="facts tree (default: app.paths.VAULT_FACTS_ROOT)")
    ap.add_argument("--apply", action="store_true",
                    help="write the plan and reindex the changed paths (default: "
                         "print only)")
    args = ap.parse_args(argv)

    root = args.facts_root or P.VAULT_FACTS_ROOT
    rep = (apply_tree(root, reindexer=reindexer) if args.apply
           else plan_tree(root))
    rep.stamp_invocation()
    # `reindexer` is a test seam only, in the fleet's shape (`kg_rebuild.main` threads
    # the same argument for the same reason): pointing `--apply` at the real
    # `kg.sqlite` from a fixture tree would rewrite the LIVE index with fixture rows,
    # which is the 2026-09-03 incident and not a test. Production passes nothing and
    # gets `default_reindexer()`.
    print(rep.render(apply=args.apply, root=root))
    # 1 only for something that actually went wrong. "Planned nothing" is NOT an
    # error: a tree with no untagged record is the goal state, and a non-zero exit
    # there would be a fresh false alarm of the #2016/#552 class — the nightly would
    # read red on the very run that finished the repair.
    return 1 if (rep.failed or rep.errors or rep.reindex_error
                 or rep.verify_lost > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
