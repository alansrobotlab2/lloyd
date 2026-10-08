"""#2415 clauses 1 and 2: the fold's own byte check reports the ledger's orphans.

The 2026-10-07 dream #47 fold (vault `e72a3689`, "MEMORY.md 20,458→19,135 B") rewrote
index lines. It reported `bytes: 20458 → 19135, entries: 90 → 85` and passed
`scripts/memory/validate_memory_index.py`, yet it left rows in
`lloyd/memory/memory-md-ledger.md` whose `anchor:` named wording no longer in the file.
Re-measured here from those blobs with `memory_ledger`'s own join: **29 rows detached over
28 distinct anchors**, one anchor (`**Rebuild Regressions**…`) carrying two rows, in both
the post-fold state (33 rows, 90 → 85 entries) and the state the item was filed against
(43 rows, 85 entries). `status()` printed `orphan rows: 28` because that field is the
*anchor* set — so the count everyone quoted understated the rows by one, which is the same
class of error as not printing it at all. The index came in at 19,135 B, 1,323 B under the
figure the fold reported and still 6,465 B under its ceiling, so the byte check that ran had
nothing to complain about. Those rows are neither free space nor coverage:
`retire` only archives them, no live entry counts them, and each one used to certify a
`retire_when` check. Vault `972e265e` dispositioned all 29: 18 anchors re-pointed onto live
lines that had no row and 11 rows archived, which is exactly the 43 → 32 the ledger shows.

The two facts that make this a blind spot rather than a skipped step:

* `status()` already counted them — `python3 scripts/memory/memory_ledger.py status`
  printed `orphan rows: 28` all week. The count existed behind a call nobody in the fold
  path makes.
* The fold route runs `validate_memory_index.py`, which printed bytes and topic counts
  and had never read a ledger. And dream #47 shrank the file; a fold that had instead
  reworded lines at **equal bytes** — the commonest kind of tidy edit — would have
  printed the *identical* byte figure and still detached every anchor it touched.
  `test_a_byte_neutral_fold_of_one_line_...` pins that pair on the fixture.

Report-only, deliberately: a fatal count would have failed every nightly from
`e72a3689` until the disposition landed, turning an instrument into an outage. The
owed-check job rules on fatality once live data reads 0 and one fold has consumed the
printed number. An unreadable ledger is reported as `orphan rows unread (<reason>)`
instead and never enters `errors`: the ledger lives in the vault, and a vault I/O fault
must not stop the index that is prompt content from validating. It still prints a line,
because a count that silently disappears is how #2361's validator read as clean.

Fixture note, because it is the part a later reader will trip on: this store's topic
files live under `memory/`, not `topics/` (`mc.TOPICS_SUBDIR`), while index lines still
link them as `→ topics/<slug>`. So `memory/memory-md-ledger.md` is itself a topic file
that must be linked, or the fixture is red for a link reason and shows nothing about
ledgers.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "memory" / "validate_memory_index.py"
LEDGER_SCRIPT = REPO / "scripts" / "memory" / "memory_ledger.py"
VAULT = Path.home() / "obsidian"
#: Where the loaded index files actually live: the vault's `lloyd/`, which is the default
#: `--root` this script resolves for a real checkout and the one both fold skills pass.
VAULT_MEMORIES = VAULT / "lloyd"

#: The loaded index files that must each get a ledger figure. A report covering only the
#: first would satisfy every "the line carries a count" assertion below — and `USER.md`
#: is the file the curation trigger actually fires on, so it is the one a fold is most
#: likely to be editing and least likely to be counting.
INDEX_NAMES = ("MEMORY.md", "USER.md")

# `ledger_rows` and `anchor_of` are the parts `status()` itself joins with, imported
# rather than restated: an anchor computed a second way is a second instrument that can
# drift from the one whose number the validator prints.
_spec = importlib.util.spec_from_file_location("memory_ledger_under_test", LEDGER_SCRIPT)
memory_ledger = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(memory_ledger)

# ---------------------------------------------------------------- fixture corpus

#: The fixture `MEMORY.md`'s entry line, modelled on the repo's real 2026-09-22 class
#: rule and shortened to stay clear of the ceiling and the line-length error. It carries
#: the `-> topics/` hook `LINK_RE` requires.
ENTRY_ORIG = (
    "- [project] (2026-01-01) **A stored command whose root moved, and a job gate that "
    "counts it** — every stored command naming the old one reads as a clean non-answer "
    "-> topics/a-stored-command-whose-root-moved-and-a\n"
)
#: Dream #47's edit exactly: a comma become a colon inside the heading, the tail reworded
#: to the same 216 bytes. Exactly one character of the 59-character anchor changes, and
#: the join is `have - live` over those anchors — exact set equality — so the row
#: detaches while every byte count in the report stays exactly where it was.
ENTRY_FOLD = (
    "- [project] (2026-01-01) **A stored command whose root moved: the gate that counts "
    "it** — and it now prints the root it read, so a clean non-answer names that path "
    "-> topics/a-stored-command-whose-root-moved-and-a\n"
)
#: The two ledgers are files under `memory/`, so `check()` demands a link for each
#: (`topic file … no index line links it`) and no more links than files exist. These two
#: reference lines are what make the fixture a corpus the validator calls green; without
#: them every node below would assert against a run that was red for a link reason.
LEDGER_LINKS = (
    "- [reference] (2026-01-01) **The index's own row ledger** — every line's check "
    "lives here -> topics/memory-md-ledger\n",
    "- [reference] (2026-01-01) **The persona ledger** — the user-facing rows live here "
    "-> topics/user-md-ledger\n",
)
#: The fixture ledger's live row, written the way the writer writes one: the anchor is
#: `anchor_of` the index line. `test_the_fixture_anchor_is_the_writers_own` pins this
#: literal against the shipped function, so it cannot drift into a second definition.
ANCHOR_ORIG = "[project] (2026-01-01) **A stored command whose root moved,"
#: Two headings no fixture line begins with: the wording a fold replaced outright.
ANCHOR_DEAD_ONE = ("[project] (2026-01-01) **The uptake instrument scope** — a heading "
                   "no line in this fixture index begins with")
ANCHOR_DEAD_TWO = ("[project] (2026-01-01) **A guard on one write surface** — wording a "
                   "fold replaced with something else entirely")

ENTRY_USER = "- [user] Speaks English\n"
#: `anchor_of` strips the leading `- ` and nothing else, so the tag stays in the anchor.
ANCHOR_USER = "[user] Speaks English"

#: The one non-ledger topic file, beside the ledgers in `memory/` (see module docstring).
EXTRA_TOPIC = "memory/a-stored-command-whose-root-moved-and-a.md"


def _ledger_front(ledger_for: str) -> str:
    return (f"---\ntype: note\nscope: local\nsubject: memory-ledger/x\n"
            f"ledger_for: lloyd/{ledger_for}\n---\n\n")


def _row(anchor: str) -> str:
    return (f"- anchor: {anchor}\n"
            f"  checked_at: 2026-10-01\n"
            f"  retire_when: the entry is retired\n"
            f"  recheck_command: `grep -c x MEMORY.md`\n\n")


#: Three ledger states. `two-orphans` deliberately carries *two rows on one dead anchor*
#: — what a fold leaves behind when it rewords one line twice — so the row total and the
#: orphan total are different numbers (3 rows, 2 orphans) and a script that printed
#: distinct-anchor counts, or put a row count in the orphan slot, is caught here.
LEDGER_FIXTURES = {
    "one-orphan": {
        "MEMORY.md": _ledger_front("MEMORY.md") + _row(ANCHOR_ORIG) + _row(ANCHOR_DEAD_ONE),
        "USER.md": "",
    },
    "two-orphans": {
        "MEMORY.md": (_ledger_front("MEMORY.md")
                      + _row(ANCHOR_DEAD_ONE) + _row(ANCHOR_DEAD_TWO) + _row(ANCHOR_DEAD_TWO)),
        "USER.md": "",
    },
    "clean": {
        "MEMORY.md": _ledger_front("MEMORY.md") + _row(ANCHOR_ORIG),
        "USER.md": _ledger_front("USER.md") + _row(ANCHOR_USER),
    },
}

#: What each fixture must report per loaded file: (rows, orphans). `USER.md`'s ledger is
#: empty except in `clean`, so its two numbers move independently of `MEMORY.md`'s and a
#: report that described one file twice cannot pass.
EXPECTED = {
    "one-orphan": {"MEMORY.md": (2, 1), "USER.md": (0, 0)},
    "two-orphans": {"MEMORY.md": (3, 2), "USER.md": (0, 0)},
    "clean": {"MEMORY.md": (1, 0), "USER.md": (1, 0)},
}


def make_root(tmp_path: Path, fixture: str = "clean") -> Path:
    """The loaded files and their ledgers, written into `tmp_path/memories`.

    Written, not symlinked, because `check()` resolves a *missing* index with
    `Path(__file__).parents[2]/<name>` — its own location, not the root it was handed —
    so a tree holding only `MEMORY.md` would silently resolve `USER.md` against the repo
    and print real bytes for a file the fixture never made. Here both exist, so every
    byte figure asserted below is the fixture's own.
    """
    root = tmp_path / "memories"
    (root / "memory").mkdir(parents=True)
    (root / "MEMORY.md").write_text(ENTRY_ORIG + "".join(LEDGER_LINKS), encoding="utf-8")
    (root / "USER.md").write_text(ENTRY_USER, encoding="utf-8")
    (root / EXTRA_TOPIC).write_text("---\ntype: note\n---\n\n# note\n", encoding="utf-8")
    for name, rel in (("MEMORY.md", "memory/memory-md-ledger.md"),
                      ("USER.md", "memory/user-md-ledger.md")):
        (root / rel).write_text(LEDGER_FIXTURES[fixture][name], encoding="utf-8")
    return root


def validate(root: Path) -> subprocess.CompletedProcess:
    """One real run of the check the skills invoke, as a subprocess.

    Subprocess because `main()`'s exit status is part of the contract: calling `check()`
    in-process would have to re-implement what the script does with the return value,
    which is the half clause 1 puts a boundary across.
    """
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True, text=True,
    )


def output_of(res: subprocess.CompletedProcess) -> str:
    """`stdout + stderr`: findings go through `warn()` onto stderr, so a green run's
    report and a red run's diagnostics both have to be in scope for a selector."""
    return res.stdout + res.stderr


def summary_line(text: str) -> str:
    """The one `OK:`/`FAIL:` line for the root that was handed in.

    Fails if there is not exactly one: `main` prints it once, and a selector that quietly
    took the first of several would let a duplicated report pass as per-file coverage.
    """
    hits = [ln for ln in text.splitlines()
            if ln.startswith(("OK:", "FAIL:")) and "/MEMORY.md " in ln]
    assert len(hits) == 1, (
        f"expected exactly one OK/FAIL summary line naming MEMORY.md and its bytes, got "
        f"{len(hits)} in {text!r}")
    return hits[0]


def block_line(text: str, name: str) -> str:
    """The per-file line for `name` — the one carrying that file's own byte figure
    beside its counts.

    This, not the summary, is where `USER.md`'s numbers live: its bytes appear nowhere
    else in this output, which is exactly why the curation step's byte trigger fires on
    it while nobody counts its rows. Requiring a byte figure in the selector is the
    positive control that this is a coverage line and not a stray mention.
    """
    hits = [ln for ln in text.splitlines()
            if re.match(rf"^\s+{re.escape(name)}: \d", ln) and " B" in ln]
    assert len(hits) == 1, (
        f"expected exactly one report line carrying {name}'s own byte figure and its "
        f"ledger counts, got {len(hits)} in {text!r}")
    return hits[0]


def truth(root: Path, name: str) -> tuple[int, int]:
    """(rows, orphans) for `name`, from `memory_ledger`'s own parts.

    `status()`'s predicate is `have - live` over 60-character anchors — exact set
    equality, not a prefix join — restated here so a printed number can be checked
    against a second reading of the same rule rather than against itself.
    """
    stem = "memory-md" if name == "MEMORY.md" else "user-md"
    rows = memory_ledger.ledger_rows(
        (root / "memory" / f"{stem}-ledger.md").read_text(encoding="utf-8"))
    live = {memory_ledger.anchor_of(e)
            for e in memory_ledger.loaded_entries((root / name).read_text(encoding="utf-8"))}
    have = {r["anchor"] for r in rows}
    return len(rows), len(have - live)


def counts_of(line: str) -> tuple[int, int]:
    """(rows, orphans) read back off one report line, with the line in the failure.

    Both spellings the script prints: the summary hangs the pair off its byte figure as
    `— ledger: 32 rows, 0 orphan rows (report-only)`, while the per-file block spells the
    first noun out (`32 ledger rows`). A selector that read only one of them would report
    a missing count on a line that carries it, and a count that moved between the two
    spellings would read as absent rather than as changed.
    """
    orph = re.search(r"(\d+) orphan rows?", line)
    rows = re.search(r"(\d+) ledger rows?", line) or re.search(r"ledger: (\d+) rows?", line)
    assert rows and orph, f"this line is missing a row or an orphan count: {line!r}"
    return int(rows.group(1)), int(orph.group(1))


def bytes_of(line: str) -> int:
    """The byte figure on a line, printed either as `N B / M B` or as `N B,`."""
    m = re.search(r"(\d[\d,]*) B", line)
    assert m, f"this line carries no byte figure at all: {line!r}"
    return int(m.group(1).replace(",", ""))


# ---------------------------------------------------------------- the two claims

def test_the_fixture_anchor_is_the_writers_own():
    """The fixture ledger holds the anchor the writer would have written.

    `_row(ANCHOR_ORIG)` puts a 59-character literal into the fixture ledger, and every
    claim below about a row joining or detaching is only about the real join if that
    literal is what `anchor_of` derives from the index line. Checked against the shipped
    function rather than restated, because the point of this file is that the reading be
    `memory_ledger`'s and not a second one — which is #2248's defect in this script's own
    history: the hook rule re-implemented `memory-capture`'s regex pair, the writer began
    accepting bare `-> topics/...` hooks, and the validator failed 85 correctly-hooked
    entries for three months.

    `ANCHOR_USER` is the second half of that lesson: `anchor_of` strips the leading `- `
    and keeps the `[user]` tag, so a hand-written anchor that dropped the tag would join
    nothing and the fixture's clean state would be reporting zero for an unjoined row.
    """
    assert memory_ledger.anchor_of(ENTRY_ORIG) == ANCHOR_ORIG, (
        f"anchor_of() now derives {memory_ledger.anchor_of(ENTRY_ORIG)!r}, not the "
        f"fixture's {ANCHOR_ORIG!r}: the fixture ledger joins nothing, and a printed "
        "'0 orphan rows' would then mean the script prints zero for an unjoined row")
    assert memory_ledger.anchor_of(ENTRY_USER) == ANCHOR_USER, (
        f"anchor_of() derives {memory_ledger.anchor_of(ENTRY_USER)!r} from the USER "
        f"fixture line, not {ANCHOR_USER!r}")
    assert memory_ledger.anchor_of(ENTRY_FOLD) != ANCHOR_ORIG, (
        "the folded line's anchor still equals the fixture row's anchor, so the fold "
        "below detaches nothing and 'the count goes nonzero' would be fiction")


@pytest.mark.parametrize("fixture", sorted(EXPECTED))
def test_each_loaded_files_report_line_carries_its_live_ledger_counts(tmp_path, fixture):
    """Clause 1: the ledger's two counts beside the byte figure, for every loaded file.

    Three ledgers — 1 orphan of 2 rows, 2 orphans of 3 rows, none of 1 — and each loaded
    file's own report line now carries its row count and orphan count with the bytes it
    already printed. `USER.md` is checked in the same run, which is what "for each loaded
    index file" means: its ledger is empty in two of the three fixtures and holds one
    joined row in the third, so its figures move independently of `MEMORY.md`'s and a
    report describing only the first file cannot pass.

    Both halves of "same line or block" are pinned: the `OK:` line this script's callers
    quote carries `MEMORY.md`'s counts beside its `N B / M B` figure, and the per-file
    line carries that file's own bytes with its own counts.
    """
    root = make_root(tmp_path, fixture)
    res = validate(root)
    assert res.returncode == 0, (
        f"the fixture root was red before any ledger was read, so the counts below are "
        f"printed into a failing report: {output_of(res)}")
    text = output_of(res)

    summary = summary_line(text)
    assert bytes_of(summary) == (root / "MEMORY.md").stat().st_size, (
        f"the summary line's byte figure is not the fixture's, so it describes the "
        f"repo's own index rather than the tree under test: {summary!r}")
    assert counts_of(summary) == EXPECTED[fixture]["MEMORY.md"], (
        f"the quoted summary line does not carry MEMORY.md's fixture counts "
        f"{EXPECTED[fixture]['MEMORY.md']}: {summary!r}")

    for name in INDEX_NAMES:
        line = block_line(text, name)
        assert bytes_of(line) == (root / name).stat().st_size, (
            f"{name}'s ledger line reports bytes that are not the fixture's "
            f"{(root / name).stat().st_size}, so it describes some other tree: {line!r}")
        assert counts_of(line) == EXPECTED[fixture][name], (
            f"{name} printed {counts_of(line)} where its fixture ledgers hold "
            f"{EXPECTED[fixture][name]} (rows, orphans): {line!r}")
        assert truth(root, name) == EXPECTED[fixture][name], (
            f"the fixture itself is not the state this node claims: {truth(root, name)}")


def test_clean_ledger_reports_zero_rather_than_omitting_the_field(tmp_path):
    """The `0` is printed, not dropped — #2361's zero, and #1881's "including when 0".

    With every fixture row joined to a live line, both per-file lines must still carry
    `0 orphan rows`. A run that suppressed the count at zero would leave a fold with
    nothing to quote and a curator with no way to tell a dispositioned ledger from a step
    that never read one — #2361's exact failure in this same script: four months of `0
    topic files, 0/0 linked` that actually meant the validator had been run with no
    `--root` and never looked.

    `block_line` fails rather than returning nothing when a line is absent, so this node
    cannot be satisfied by the rest of a clean-looking report.
    """
    root = make_root(tmp_path, "clean")
    res = validate(root)
    assert res.returncode == 0, output_of(res)
    text = output_of(res)
    for name in INDEX_NAMES:
        line = block_line(text, name)
        assert counts_of(line)[1] == 0, line
        assert re.search(r"0 orphan rows?", line), (
            f"{name}'s clean ledger printed no zero, so a fold could not tell a clean "
            f"ledger from one the report never opened: {line!r}")
        assert "unread" not in line, (
            f"the clean fixture's ledger parsed but was reported unreadable: {line!r}")


def _load_validator():
    """The validator as an importable module, by path — the way it loads its own sibling.
    """
    spec = importlib.util.spec_from_file_location("validate_memory_index_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_counts_are_statuses_own_and_not_re_derived(tmp_path, monkeypatch, capsys):
    """Clause 1's mechanism: the printed numbers are `status()`'s, not the script's.

    The clause says the counts come from `memory_ledger.status()`, and the reason is not
    tidiness. `#2248` is this defect one level up in the file being edited: the hook rule
    re-implemented `memory-capture`'s pair instead of calling `hook_of_entry`, the writer
    began accepting bare `-> topics/...` hooks, and the validator went red over 85
    correctly-hooked entries for three months. Two readings of one rule drift apart
    unnoticed, and the store's orphan rule — `have - live` over exact 60-character
    anchors — is exactly the rule a second copy gets subtly wrong. This fixture's ledgers
    are too small to discriminate a subtly wrong count, so the test substitutes the
    instrument instead of observing it.

    The substituted `status()` reports `99 rows / 3 orphan rows` for `MEMORY.md` and `88 /
    0` for `USER.md`, with byte figures of its own — numbers the fixture cannot produce
    (it holds 1 row and 0 orphans, at 344 and 24 bytes). Any figure re-derived from the
    tree instead of read from the report shows up as the fixture's number and fails. The
    call is also asserted to name the tree being validated: a status of some other root
    would print a plausible number about the wrong corpus, which is #2361's zero coming
    back as a nonzero.

    The subprocess nodes cannot see a monkeypatch, so this one runs in-process against the
    seam `_memory_ledger`'s own docstring says exists for substitution. Report-only is
    pinned here as well: three substituted orphans must not move the exit status.
    """
    vi = _load_validator()
    calls: list = []
    FAKE = {"MEMORY.md": (99, 3, 12345), "USER.md": (88, 0, 6789)}

    def fake_status(memories_dir):
        calls.append(Path(memories_dir))
        return {name: {"bytes": b, "entries": 77, "rows": r,
                       "orphan_rows": ["a"] * o}
                for name, (r, o, b) in FAKE.items()}

    class _FakeLedger:
        status = staticmethod(fake_status)

    monkeypatch.setattr(vi, "_memory_ledger", lambda: _FakeLedger)

    root = make_root(tmp_path, "clean")
    assert truth(root, "MEMORY.md") == (1, 0) and (root / "MEMORY.md").stat().st_size != 12345, (
        "the fixture is not the small clean state the substituted numbers are meant to "
        "contradict, so 'echoed, not derived' below would discriminate nothing")

    assert vi.main(["--root", str(root)]) == 0, (
        "3 substituted orphan rows changed the exit status, which is the owed ruling and "
        "not this clause")
    out = capsys.readouterr().out

    assert calls == [root], (
        f"status() was called {len(calls)} time(s) {[str(c) for c in calls]!r}, not once "
        f"on the tree being validated ({root}); a count for some other corpus is a "
        "plausible number about nothing")
    for name, (want_rows, want_orph, want_bytes) in FAKE.items():
        line = block_line(out, name)
        assert counts_of(line) == (want_rows, want_orph), (
            f"{name} printed {counts_of(line)} instead of status()'s "
            f"({want_rows}, {want_orph}) — the figure was recomputed from the tree, "
            f"whose ledgers hold {truth(root, name)}: {line!r}")
        assert bytes_of(line) == want_bytes, (
            f"{name}'s byte figure came from the file rather than the report, so the "
            f"pair on this line is not one reading of one corpus: {line!r}")


def test_a_byte_neutral_fold_of_one_line_moves_the_orphan_count_at_equal_bytes(tmp_path):
    """Clause 2: the blind spot itself — identical bytes, nonzero orphans.

    `ENTRY_FOLD` is `ENTRY_ORIG` with one comma become a colon inside the heading and its
    tail reworded to the same 216 bytes. Run through the real validator it must send the
    printed orphan count from 0 to 1 while the printed byte total is *identical* — the
    pair that let dream #47 quote `bytes: 20458 → 19135` and `OK` as evidence that nothing
    had broken.

    Nothing here is assumed. Byte-neutrality is asserted against the encoded strings and
    again against the file's `st_size`; `memory_ledger`'s own join is consulted to prove
    the fold really did detach the row; and the byte figure is re-read off the report
    rather than trusted from the file, so a change in how bytes are printed cannot make
    the equality vacuous. The `OK:` verdict has to stay on the summary line too — that is
    the line a fold's completion note quotes, so a report that moved the verdict off it
    would leave the run quoting something else.
    """
    assert len(ENTRY_ORIG.encode("utf-8")) == len(ENTRY_FOLD.encode("utf-8")), (
        f"the fixture fold is no longer byte-neutral "
        f"({len(ENTRY_FOLD.encode('utf-8'))} vs {len(ENTRY_ORIG.encode('utf-8'))} B), so "
        "the equal-bytes half of this claim describes an edit shape that cannot occur and "
        "clause 2 is worth less than it sounds")
    assert "-> topics/a-stored-command-whose-root-moved-and-a" in ENTRY_FOLD, (
        "the fold drops the topic hook, which would make the run red for a hook reason "
        "and let a missing orphan count hide behind the hookless error")

    root = make_root(tmp_path, "clean")
    base = bytes_of(summary_line(output_of(validate(root))))
    assert truth(root, "MEMORY.md") == (1, 0), (
        "the baseline is not one joined row and no orphan, so a nonzero count after the "
        "fold would prove nothing about the fold")

    index = root / "MEMORY.md"
    before = index.read_bytes()
    index.write_bytes(before.replace(ENTRY_ORIG.encode("utf-8"),
                                     ENTRY_FOLD.encode("utf-8"), 1))
    assert index.read_bytes() != before, (
        "the fold found no occurrence of the entry line to replace, so nothing below "
        "describes a fold at all")
    assert index.stat().st_size == len(before), (
        "writing the fold changed the file's size, so the fold is not byte-neutral and "
        "the byte equality below is a different claim")
    assert truth(root, "MEMORY.md") == (1, 1), (
        "memory_ledger's own join says the fold detached no row, so the fixture no "
        "longer reproduces dream #47 and a nonzero printed count would be a bug")

    res = validate(root)
    assert res.returncode == 0, output_of(res)
    text = output_of(res)

    assert bytes_of(summary_line(text)) == base, (
        f"a byte-neutral fold moved the printed byte figure "
        f"({base} -> {bytes_of(summary_line(text))}); this node's premise has broken")
    assert counts_of(block_line(text, "MEMORY.md")) == (1, 1), (
        f"the fold left MEMORY.md reading {counts_of(block_line(text, 'MEMORY.md'))} "
        "where memory_ledger says 1 ledger row and 1 orphan row")
    assert summary_line(text).startswith("OK:"), (
        f"the verdict word left the line a fold quotes as its evidence: "
        f"{summary_line(text)!r}")


def test_the_orphan_count_never_changes_the_exit_status(tmp_path):
    """Report-only, measured at the one number that could make the fold route red.

    `main` returns 1 iff `errors` is non-empty, so the claim is that a nonzero orphan
    count leaves the return code at 0 while the report still says `1 orphan row`. The
    asymmetry is deliberate: a dangling hook is fatal because it is a defect in the file
    being committed, while an orphan row is a bookkeeping debt in a different file the run
    may not even be editing — and a fatal count on the day this landed would have failed
    every nightly from dream #47's fold until the disposition, turning an instrument into
    an outage. The owed-check job rules on fatality once live reads 0 and one fold has
    consumed the printed number.

    Both halves are asserted, because "does not change the status" is vacuously true of a
    script that reports nothing at all: with one orphan row the run is green, and with an
    oversized index the same script still goes red over bytes.
    """
    root = make_root(tmp_path, "one-orphan")
    res = validate(root)
    assert res.returncode == 0, (
        f"a nonzero orphan count turned the check red, which is the owed ruling rather "
        f"than this clause: {output_of(res)}")
    assert counts_of(block_line(output_of(res), "MEMORY.md")) == (2, 1)

    (root / "MEMORY.md").write_text(ENTRY_ORIG + "x" * 40_000 + "\n", encoding="utf-8")
    red = validate(root)
    assert red.returncode == 1, (
        "an oversized index no longer fails the byte check, so 'the orphan count does not "
        "change the status' would be true of a script that reports nothing")


def test_an_unreadable_ledger_is_printed_as_such(tmp_path):
    """Clause 1's last requirement: the reason, never a silent absence.

    A fixture ledger the loader cannot decode still gets its byte report, plus a line
    naming why there are no counts on it. The run stays green: the ledger lives in the
    vault, so a vault read that fails must not stop the index that is prompt content from
    validating — but it must not be read as coverage either. #2361's whole lesson is that
    a zero meaning "I did not look" is worse than no number, and a count that simply
    vanished is how `orphan rows: 28` went a day without anyone seeing it.

    One thing this node deliberately does **not** claim: that the *other* loaded file's
    counts survive. They do not. `memory_ledger.status()` reads each ledger at
    `scripts/memory/memory_ledger.py:199` outside the try that guards the index read, so a
    binary ledger raises `UnicodeDecodeError` out of `status()` itself, the validator's one
    `except Exception` catches it for the whole call, and `_ledger_block` then prints the
    reason *instead of* any per-file line — `USER.md`'s intact ledger goes uncounted
    alongside the broken one. Measured: the same fixture with only
    `memory/memory-md-ledger.md` overwritten with `\\xff\\xfe` yields a block of exactly one
    line, the reason. Clause 1 requires the reason to be printed rather than the count
    silently dropped, and it is; the collateral on the readable file is a defect in
    `status()`'s read, recorded on #2415 rather than fixed here.
    """
    root = make_root(tmp_path, "clean")
    (root / "memory" / "memory-md-ledger.md").write_bytes(b"\xff\xfe\x00binary junk")

    res = validate(root)
    assert res.returncode == 0, (
        f"an unreadable ledger made the check red: {output_of(res)}")
    text = output_of(res)
    assert "unread" in text and "orphan" in text, (
        f"nothing in the output says the orphan count could not be read, so a reader sees "
        f"a missing number and no reason: {text!r}")
    assert "UnicodeDecodeError" in text, (
        f"the reason names neither the failure nor its type, so the reader cannot tell a "
        f"corrupt ledger from a missing one: {text!r}")
    assert bytes_of(summary_line(text)) == (root / "MEMORY.md").stat().st_size, (
        "the byte figure went missing along with the ledger, so the fold's own byte check "
        "went down with the count it was never supposed to gate")
    assert "orphan rows unread" in text, (
        f"the missing counts are not labelled as the pair they are: {text!r}")


def test_the_counts_reach_stdout_where_a_fold_can_quote_them(tmp_path):
    """The report is only useful if the run that quotes it can see it on stdout.

    `main` prints the green report to stdout and findings to stderr; a count that landed
    on the wrong stream would satisfy every other node in this file and still never be
    copied into the completion note that is the whole purpose of the clause — §4 tells the
    curator to quote `status`'s lines into a `curation:` block, and a fold quotes what it
    sees. Both loaded files are asserted on stdout specifically, not on the merged output,
    and with the fixture's own figures so a constant could not pass.
    """
    root = make_root(tmp_path, "one-orphan")
    res = validate(root)
    assert res.returncode == 0, output_of(res)
    for name in INDEX_NAMES:
        block_line(res.stdout, name)      # asserts presence on STDOUT, not on stderr
    assert counts_of(block_line(res.stdout, "MEMORY.md")) == (2, 1)
    assert counts_of(summary_line(res.stdout)) == (2, 1)


@pytest.mark.live_vault
def test_the_live_ledgers_are_still_dispositioned():
    """The instrument aimed at the two files that actually ride in the prompt.

    Run explicitly: `pytest tests/test_memory_index_ledger_orphans.py -m live_vault -k
    live_ledgers`. The gate deselects `live_vault` (`scripts/automod/gate.py`'s
    `TESTS_MARK_EXPR`) because it reads the live vault, which is correct for a node whose
    claim is about live bytes — and the owed check for this landing is the same
    measurement over the live route, so nothing here is the only thing standing between a
    regression and a human.

    Both ledgers were dispositioned to 0 orphans by vault `972e265e` — 18 anchors
    re-pointed onto live lines that had no row, 11 rows archived with a reason each — so
    this node is a standing claim about ledger hygiene: if a fold detaches an anchor
    again, it goes red before that fold's note can quote a byte figure as proof that
    nothing broke. The `rows > 0` guard below is the positive control #2361 says an
    instrument needs: an empty ledger would make every zero in this file vacuous, and it
    is a real possibility in a tree rebuilt from scratch.
    """
    if not (VAULT_MEMORIES / "MEMORY.md").is_file():
        pytest.skip(f"no loaded index at {VAULT_MEMORIES} — nothing to measure")
    res = validate(VAULT_MEMORIES)
    text = output_of(res)
    assert res.returncode == 0, (
        "the live vault fails the fold's own byte or link check today, so this node has "
        f"nothing to report about: {text}")
    for name in INDEX_NAMES:
        rows, orphans = truth(VAULT_MEMORIES, name)
        assert rows > 0, (
            f"{name}'s live ledger holds no rows at all, so every zero asserted in this "
            "file would pass against an empty corpus")
        assert orphans == 0, (
            f"{name}'s ledger has {orphans} orphan rows again — disposition them (#2415 "
            "§5) rather than editing this assertion")
        line = block_line(text, name)
        assert counts_of(line) == (rows, 0), (
            f"{name}'s report line disagrees with its own ledger's {rows} rows / 0 "
            f"orphans: {line!r}")


def test_this_files_prose_does_not_outrun_its_measurements():
    """The #1881 rule: the numbers this file publishes are checked, not asserted.

    The module docstring puts two claims in front of a reader who will not re-measure
    them, and both decide what the next run concludes.

    * the scale and sign — 28 detached rows of 43, on a file that *shrank*. Had the file
      grown, or the count been a dozen, this would read as a loud failure somebody walked
      past rather than an unmeasured property. Re-measured below from the blobs the
      docstring names, as it says.
    * byte-neutrality — the claim that the edit shape which detaches an anchor need not
      move a byte. If it were false, the equal-bytes assertion above defends a case that
      cannot occur, and clause 2 is worth less than it sounds.

    Report-only is an argument, not a measurement, and is claimed only as far as the code
    agrees — by `test_the_orphan_count_never_changes_the_exit_status`.
    """
    doc = " ".join((__doc__ or "").split())
    assert "29 rows detached over 28 distinct anchors" in doc and "43 rows" in doc, (
        "the docstring no longer states the scale as measured, and above all no longer "
        "separates the two counts: rows and anchors differ by one here, and `status()` "
        "prints only the anchor set, so prose that quotes one as the other has already "
        "lost a row")
    assert "1,323 B" in doc and "85 entries" in doc, (
        "the fold's byte delta and entry count are gone from the prose: 'it shrank by "
        "1,323 B and nothing was caught' is the finding, and without the numbers a later "
        "run cannot tell the defect from a size change")
    assert "18 anchors re-pointed" in doc and "11 rows archived" in doc, (
        "the disposition's split is missing; it is what makes 29 dispositioned rows check "
        "against the ledger's 43 → 32, and the only figure that shows the work was not a "
        "blanket archive")

    if not (VAULT / ".git").exists():
        pytest.skip(f"no vault at {VAULT}: the fold's blobs cannot be re-measured here")

    def blob(rev: str, rel: str) -> str:
        r = subprocess.run(["git", "-C", str(VAULT), "show", f"{rev}:{rel}"],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"git show {rev}:{rel} failed: {r.stderr.strip()}"
        return r.stdout

    def detached_against(index_text: str, ledger_text: str) -> tuple[int, int, int, int]:
        """(rows, dead rows, distinct dead anchors, live entries)."""
        rows = memory_ledger.ledger_rows(ledger_text)
        live = {memory_ledger.anchor_of(e)
                for e in memory_ledger.loaded_entries(index_text)}
        dead = [r for r in rows if r["anchor"] not in live]
        return len(rows), len(dead), len({r["anchor"] for r in dead}), len(live)

    # The fold's own before/after, which is the causal claim rather than an association:
    # the same 33 rows join every entry one commit earlier and detach 29 of them one commit
    # later, so nothing between the two produced the orphans.
    led_rel = "lloyd/memory/memory-md-ledger.md"
    before = detached_against(blob("e72a3689^", "lloyd/MEMORY.md"), blob("e72a3689^", led_rel))
    after = detached_against(blob("e72a3689", "lloyd/MEMORY.md"), blob("e72a3689", led_rel))
    assert before == (33, 0, 0, 90), (
        f"before the fold: {before} as (rows, dead rows, dead anchors, entries) — every "
        "row must join an entry, or the fold did not create the orphans and the causal "
        "sentence above is wrong")
    assert after == (33, 29, 28, 85), (
        f"the fold left {after} as (rows, dead rows, dead anchors, entries), not (33, 29, "
        "28, 85); the count every clause is scoped by has moved")

    # The state the item was filed against, and the pair the module docstring publishes:
    # still 29 dead rows over 28 anchors once a later nightly write had grown the ledger to
    # 43 rows. `status()` printed `orphan rows: 28` because that field is the anchor set, so
    # one anchor carrying two rows (`**Rebuild Regressions**…`) hid a row from the very
    # count that was finally printed — understating the same damage by one, which is why
    # this file's own fixture ledger deliberately doubles a dead anchor too.
    filed = detached_against(blob("972e265e^", "lloyd/MEMORY.md"), blob("972e265e^", led_rel))
    assert filed == (43, 29, 28, 85), (
        f"at filing time: {filed} as (rows, dead rows, dead anchors, entries), not (43, "
        "29, 28, 85) — the docstring's figures, and the disposition's own scope, are then "
        "stale prose")

    # The disposition, read off the two ledgers rather than quoted from a commit message:
    # 18 anchors re-pointed onto lines that had no row, 11 rows archived, 29 dispositioned.
    rows_pre = memory_ledger.ledger_rows(blob("972e265e^", led_rel))
    rows_post = memory_ledger.ledger_rows(blob("972e265e", led_rel))
    idx_post = blob("972e265e", "lloyd/MEMORY.md")
    live_post = {memory_ledger.anchor_of(e) for e in memory_ledger.loaded_entries(idx_post)}
    pre_anchors = {r["anchor"] for r in rows_pre}
    post_anchors = {r["anchor"] for r in rows_post}
    re_pointed = {a for a in post_anchors - pre_anchors if a in live_post}
    assert (len(rows_pre), len(rows_post), len(re_pointed)) == (43, 32, 18), (
        f"the disposition reads {len(rows_pre)} → {len(rows_post)} rows with "
        f"{len(re_pointed)} anchors newly joining a live line; the docstring's '18 "
        "anchors re-pointed' and the 11 rows that left the ledger, which together are the "
        "29 dead rows above, do not follow from those numbers")
    assert len(rows_pre) - len(rows_post) == 11, (
        f"{len(rows_pre) - len(rows_post)} rows left the ledger, not the 11 the docstring "
        "says were archived with a reason each")
    # 29 rows left their anchors and those rows shared 28 anchors, so exactly 28 anchors
    # disappear from the ledger while 18 new ones appear: the row/anchor distinction again,
    # which is the pair `status()` conflates when it prints `orphan rows: 28`.
    assert len(pre_anchors - post_anchors) == 28, (
        f"{len(pre_anchors - post_anchors)} anchors left the ledger against 29 dispositioned "
        "rows, so the doubled anchor is not where the row/anchor gap the docstring names "
        "actually comes from")

    size_pre = len(blob("e72a3689^", "lloyd/MEMORY.md").encode("utf-8"))
    size_post = len(blob("e72a3689", "lloyd/MEMORY.md").encode("utf-8"))
    assert size_pre == 20458 and size_post == 19135 and size_pre - size_post == 1323, (
        f"the fold's byte figures are {size_pre} -> {size_post}, not the published "
        "20,458 -> 19,135 (down 1,323 B)")

    assert len(ENTRY_ORIG.encode("utf-8")) == len(ENTRY_FOLD.encode("utf-8")), (
        "the fixture fold is no longer byte-neutral, so this file's central claim — that "
        "the edit shape which detaches an anchor can leave the byte total alone — is "
        "false")
    assert memory_ledger.anchor_of(ENTRY_FOLD) != memory_ledger.anchor_of(ENTRY_ORIG), (
        "the fold's anchor survives, so the fixture reproduces nothing and the nonzero "
        "count asserted above would be a bug rather than a finding")


def test_the_fold_skills_still_invoke_this_script():
    """The instrument's reason to exist is that the fold route runs it.

    `validate_memory_index.py` is named by `skills/dream-consolidation/SKILL.md` (step
    1c) and `skills/nightly-reflection-knowledge-write/SKILL.md` (step 2a). Those files
    are in the vault, outside this repo, so the claim is made through a repo-side
    instrument that always runs: `git grep -l validate_memory_index -- scripts/` has to
    name the script itself, which is what proves the file a skill points at exists at the
    path the skill says. `hits.returncode == 0` is the positive control — a failing grep
    and an empty one both look like "no routes", which is #2361's shape again, and it is
    the same control `test_memory_ledger_bound.py` puts on its own corpus grep.

    The vault-side check that names both skills is `test_the_fold_skills_name_this_`
    `validator_by_path` below, marked `live_vault` because it reads live vault files.
    """
    hits = subprocess.run(
        ["git", "grep", "-l", "validate_memory_index", "--", "scripts/"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert hits.returncode == 0, (
        "git grep failed outright, so an empty list below would be a coverage gap rather "
        f"than an answer: {hits.stderr.strip()!r}")
    assert "scripts/memory/validate_memory_index.py" in hits.stdout, (
        f"the script is not findable by its own name under scripts/ ({hits.stdout!r}), so "
        "the path both fold skills cite resolves to nothing this repo can prove")


@pytest.mark.live_vault
def test_the_fold_skills_name_this_validator_by_path():
    """The route that gives this script its purpose, read where it lives.

    Run explicitly: `pytest tests/test_memory_index_ledger_orphans.py -m live_vault -k
    fold_skills`. The vault is a separate repository from this one, so which skills cite
    this script is live-vault state and a repo node must not assert it from memory: a
    file path that has moved reads as "not in the list" and a run that reports absence as
    a clean result is the #2361 failure wearing a different hat. Hence the marker, and
    hence the `-- skills/` scoping (a repo-wide grep would also match `tests/` files that
    invoke the script, including this one).

    The set assertion is containment, not equality: #2285 put a `validate_memory_index`
    clause into `dream-consolidation` §3e and other steps may route through
    `memory-capture`, so extra names are legitimate and only the two fold owners are
    required.
    """
    if not (VAULT / "skills").is_dir():
        pytest.skip(f"no vault at {VAULT}: the skill side is unverifiable here")
    named = {
        ln.strip() for ln in subprocess.run(
            ["git", "grep", "-l", "validate_memory_index", "--", "skills/"],
            cwd=VAULT, capture_output=True, text=True,
        ).stdout.splitlines() if ln.strip()
    }
    for skill in ("skills/dream-consolidation/SKILL.md",
                  "skills/nightly-reflection-knowledge-write/SKILL.md"):
        assert skill in named, (
            f"{skill} no longer names validate_memory_index.py, which is the only reason "
            "this script — rather than a louder check nobody invokes — is where an orphan "
            "count reaches a fold")
