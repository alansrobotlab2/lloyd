"""#1488 — the loaded-memory rationale ledger's coverage report.

`scripts/memory/memory_ledger.py status` is the deterministic half of the
proposed nightly curation: which loaded lines have no ledger row, which rows
name a line no longer loaded, and whether the file is inside its curation
margin. Read-only by design — the loaded files are written only through tools
that enforce the ceiling.

Run: .venvs/lloyd/bin/python -m pytest tests/test_memory_ledger.py
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("memory_ledger", ROOT / "scripts/memory/memory_ledger.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

USER = """---
segment: lloyd
---

# User

> **Ceiling**: a header rule about the file itself, carries no row.

## operational
- **Scope**: agent memory, knowledge and research notes go in `~/obsidian/`, NEVER `~/lloyd/`.
- **Two daily-note paths** — per-day session notes are FLAT.
- Direct, terse communication; no sycophantic language.
"""


def _dir(tmp_path, user=USER, ledger=None):
    d = tmp_path / "lloyd"
    (d / "memory").mkdir(parents=True)
    (d / "USER.md").write_text(user, encoding="utf-8")
    if ledger is not None:
        (d / "memory" / "user-md-ledger.md").write_text(ledger, encoding="utf-8")
    return d


def test_every_bullet_needs_a_row_and_the_header_needs_none(tmp_path):
    rep = ml.status(_dir(tmp_path))["USER.md"]
    assert rep["entries"] == 3 and rep["rows"] == 0
    assert len(rep["without_row"]) == 3
    assert all(not a.startswith("**Ceiling") for a in rep["without_row"])


def test_a_row_covers_its_line_and_an_edited_line_orphans_its_row(tmp_path):
    scope = ml.anchor_of("- **Scope**: agent memory, knowledge and research notes go in `~/obsidian/`")
    ledger = (f"- anchor: `{scope}` | why: keeps notes out of the code tree | origin: USER.md | "
              "retire_when: never | check: `true` | checked: 2026-09-20\n"
              "- anchor: `A line that was removed from the file` | why: x | checked: 2026-09-01\n")
    rep = ml.status(_dir(tmp_path, ledger=ledger))["USER.md"]
    assert rep["rows"] == 2
    assert scope not in rep["without_row"] and len(rep["without_row"]) == 2
    assert rep["orphan_rows"] == ["A line that was removed from the file"]
    assert rep["stalest_checked"] == [scope]


def test_a_row_written_through_memory_add_is_read(tmp_path):
    """memory_add prepends `[project] (date) ` to a topic entry."""
    scope = ml.anchor_of("- Direct, terse communication; no sycophantic language.")
    ledger = f"- [project] (2026-09-25) anchor: `{scope}` | why: tone | checked: 2026-09-25\n"
    rep = ml.status(_dir(tmp_path, ledger=ledger))["USER.md"]
    assert rep["rows"] == 1 and scope not in rep["without_row"]


def test_the_anchor_survives_an_edit_to_the_end_of_a_line():
    a = ml.anchor_of("- **Two daily-note paths** — per-day session notes are FLAT `~/obsidian/memory/`.")
    b = ml.anchor_of("- **Two daily-note paths** — per-day session notes are FLAT `~/obsidian/memory/YYYY`.")
    assert a == b and len(a) <= ml.ANCHOR_CHARS and a == a.strip()


def test_curate_flag_is_the_ceiling_minus_the_margin(tmp_path):
    ceiling = 16_384
    pad = "x" * (ceiling - ml.HEADROOM_BYTES - len(USER.encode()) + 1)
    rep = ml.status(_dir(tmp_path, user=USER + pad))["USER.md"]
    assert rep["ceiling"] == ceiling and rep["curate"] is True
    rep2 = ml.status(_dir(tmp_path / "b", user=USER))["USER.md"]
    assert rep2["curate"] is False and rep2["headroom"] == ceiling - len(USER.encode())


def test_status_writes_nothing(tmp_path):
    d = _dir(tmp_path)
    before = {p: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    ml.status(d)
    after = {p: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    assert before == after


# ── #1730 — the anchor field is delimited by ` | `, not by backticks ───────
#
# Both loaded files are written with code spans, so the 60-char span
# `anchor_of()` cuts routinely contains a backtick: 19 of the 54 loaded USER.md
# lines and 24 of the 85 loaded MEMORY.md lines on 2026-09-28. While the row
# grammar was backtick-delimited those lines could never be joined to a row at
# all, and a row written in good faith from `anchor_of()`'s own output came back
# truncated and read as a permanent orphan.

BACKTICK_LINE = ("- **Two daily-note paths** — per-day session notes are FLAT "
                 "`~/obsidian/memory/`, never in a per-month folder.")
# Same shape, but the 60-char span ENDS on the code-span backtick — the case
# where the legacy wrapper's closing backtick was the anchor's own last char.
ENDS_ON_BACKTICK = ("- **Render shaping is FIXED** (`agent-services/tts/shaping/x/`"
                    " — shipped; corrections go to the voice note)")
# A span containing the field delimiter cannot be written into a row at all.
PIPE_LINE = ("- **Two engines answer one queue** — primary | secondary, "
             "never both at once.")
# And a span that opens AND closes on a backtick is indistinguishable from the
# legacy wrapped form, so no reader can tell the wrapper from the anchor.
WRAP_LINE = ("- `djev` is the local decision engine on GPU 2 (`djev_decide` "
             "— its labels are code spans)")
CLEAN = "- Direct, terse communication; no sycophantic language."


def _user_with(entry: str) -> str:
    """The USER.md fixture with its third entry replaced by `entry`."""
    return USER.replace(CLEAN + "\n", entry + "\n")


def _row(anchor: str, checked: str = "2026-09-28") -> str:
    """A row as the 2a-ter template writes one: fields separated by ` | `, the
    anchor written verbatim and unwrapped."""
    return (f"- anchor: {anchor} | why: keeps behaviour pointed at one note | "
            f"origin: #1488 | retire_when: never | check: `true` | checked: {checked}\n")


def test_an_anchor_containing_a_backtick_round_trips_through_a_row():
    """Clause 1: anchor_of(line) → row → ledger_rows() is a round-trip even when
    the anchor crosses a code span. Under the backtick-delimited grammar every
    one of these came back shorter than it went in."""
    for line in (BACKTICK_LINE, ENDS_ON_BACKTICK):
        anchor = ml.anchor_of(line)
        assert "`" in anchor and ml.representable(anchor), anchor
        rows = ml.ledger_rows(_row(anchor))
        assert len(rows) == 1, rows
        assert rows[0]["anchor"] == anchor, (rows[0]["anchor"], anchor)
        assert rows[0]["checked"] == "2026-09-28"


def test_a_backtick_anchor_with_its_row_is_neither_uncovered_nor_orphaned(tmp_path):
    """Clause 2: `status` no longer reports a line that HAS a correct row as
    backfill owed, nor that row as an orphan."""
    anchor = ml.anchor_of(BACKTICK_LINE)
    rep = ml.status(_dir(tmp_path, user=_user_with(BACKTICK_LINE),
                        ledger=_row(anchor)))["USER.md"]
    assert rep["rows"] == 1 and rep["entries"] == 3
    assert anchor not in rep["without_row"], rep["without_row"]
    assert len(rep["without_row"]) == 2
    assert rep["orphan_rows"] == []
    assert rep["unrepresentable"] == []


def test_a_legacy_wrapped_row_still_covers_a_backtick_anchor():
    """The rows already in the vault's ledger were written wrapped in backticks
    — 5 of USER.md's 6 orphans on 2026-09-28 are exactly those rows — so the
    wrapped form has to keep parsing, including the anchor whose own last
    character IS a backtick, where a non-greedy strip would eat that too."""
    for line in (BACKTICK_LINE, ENDS_ON_BACKTICK):
        anchor = ml.anchor_of(line)
        wrapped = f"- anchor: `{anchor}` | why: written before #1730 | checked: 2026-09-20\n"
        rows = ml.ledger_rows(wrapped)
        assert len(rows) == 1 and rows[0]["anchor"] == anchor, (rows, anchor)


def test_status_separates_unrepresentable_lines_from_the_backlog(tmp_path, capsys):
    """Clause 3: a line no row can carry is not curator backlog — in the report,
    in --json, and on the printed line beside `without a row`."""
    pipe = ml.anchor_of(PIPE_LINE)
    assert "|" in pipe and not ml.representable(pipe)
    d = _dir(tmp_path, user=_user_with(PIPE_LINE))
    rep = ml.status(d)["USER.md"]
    assert rep["unrepresentable"] == [pipe]
    assert pipe not in rep["without_row"] and len(rep["without_row"]) == 2
    assert len(rep["without_row"]) + len(rep["unrepresentable"]) == rep["entries"]

    assert ml.main(["status", "--memories-dir", str(d), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["USER.md"]["unrepresentable"] == [pipe]
    assert len(out["USER.md"]["without_row"]) == 2

    assert ml.main(["status", "--memories-dir", str(d)]) == 0
    printed = capsys.readouterr().out
    assert "unrepresentable: 1" in printed and "without a row: 2" in printed
    assert f"    unrepresentable: {pipe}" in printed


def test_an_anchor_that_opens_and_closes_on_a_backtick_is_unrepresentable(tmp_path):
    """The other shape no row can carry: opening and closing on a backtick is
    indistinguishable from the legacy wrapped form, so the parser could not tell
    the anchor from its wrapper even in principle."""
    anchor = ml.anchor_of(WRAP_LINE)
    assert anchor.startswith("`") and anchor.endswith("`"), anchor   # fixture guard
    rep = ml.status(_dir(tmp_path, user=_user_with(WRAP_LINE)))["USER.md"]
    assert rep["unrepresentable"] == [anchor]
    assert anchor not in rep["without_row"]


def test_the_60_char_anchor_rule_survives_the_delimiter_change(tmp_path):
    """Clause 5: #1730 moved the row delimiter and nothing else. The anchor is
    still the first ANCHOR_CHARS normalised characters, so a span with no
    backtick still covers its line in the unwrapped form and in the legacy
    wrapped one, and the span still reaches past an edit at the end of the line
    — the behaviour `test_the_anchor_survives_an_edit_to_the_end_of_a_line`
    pins."""
    a = ml.anchor_of("- **Two daily-note paths** — per-day session notes are FLAT `~/obsidian/memory/`.")
    b = ml.anchor_of("- **Two daily-note paths** — per-day session notes are FLAT `~/obsidian/memory/YYYY`.")
    assert a == b and len(a) <= ml.ANCHOR_CHARS
    clean = ml.anchor_of(CLEAN)
    assert "`" not in clean and ml.representable(clean)
    rows = ml.ledger_rows(_row(clean) + f"- anchor: `{clean}` | why: tone | checked: 2026-09-20\n")
    assert [r["anchor"] for r in rows] == [clean, clean]
    rep = ml.status(_dir(tmp_path, ledger=_row(clean)))["USER.md"]
    assert rep["rows"] == 1 and clean not in rep["without_row"]
    assert rep["orphan_rows"] == []


SKILL = (Path.home() / "obsidian" / "skills" / "nightly-reflection-knowledge-write"
         / "step-2a-ter-curation.md")


def _skill_row_templates() -> list[str]:
    """Every row template the 2a-ter skill shows the nightly, with markdown's
    inline-code escaping undone. The nightly copies what this file literally
    shows into `lloyd/memory/user-md-ledger.md` and a separate process parses it
    back, so the document is the seam — and the vault is a live tree the same way
    the other tests that read `Path.home() / "obsidian"` treat it."""
    text = SKILL.read_text(encoding="utf-8").replace("\\`", "`")
    out = []
    for line in text.splitlines():
        s = line.strip().strip("`").strip().rstrip(".").strip()
        if re.match(r"^- (?:\[[a-z]+\] )?(?:\(\d{4}-\d{2}-\d{2}\) )?anchor:", s):
            out.append(s)
    return out


def test_the_curation_skill_writes_a_row_the_parser_reads():
    """Clause 4: a row built verbatim from the skill's own template, for an entry
    whose anchor contains a backtick, reads back with the whole anchor. The
    template must also not wrap the anchor: the legacy unwrapper would quietly
    repair a re-wrapped template for every anchor EXCEPT a hazard one, so a
    round-trip alone could not tell the two grammars apart."""
    templates = _skill_row_templates()
    assert templates, f"no `- anchor: …` row template found in {SKILL}"
    anchor = ml.anchor_of(BACKTICK_LINE)
    assert "`" in anchor and ml.representable(anchor), anchor
    for tmpl in templates:
        field = tmpl.split("anchor:", 1)[1].split("|", 1)[0].strip()
        assert not field.startswith("`"), f"the skill wraps its anchor again: {tmpl}"
        row = re.sub(r"<[^>]*>", lambda _: anchor, tmpl, count=1)
        rows = ml.ledger_rows(row + "\n")
        assert len(rows) == 1 and rows[0]["anchor"] == anchor, (row, rows)
