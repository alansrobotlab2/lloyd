"""#1488 — the loaded-memory rationale ledger's coverage report.

`scripts/memory/memory_ledger.py status` is the deterministic half of the
proposed nightly curation: which loaded lines have no ledger row, which rows
name a line no longer loaded, and whether the file is inside its curation
margin. Read-only by design — the loaded files are written only through tools
that enforce the ceiling.

Run: .venvs/lloyd/bin/python -m pytest tests/test_memory_ledger.py
"""
import importlib.util
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
