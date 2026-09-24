"""#834: nothing in the tree reads or documents the two Inner Voice tables
the store drops at init.

`usage_store.init_db` runs `DROP TABLE IF EXISTS inner_voice_critiques` and
`... inner_voice_interventions` on every boot — the v3 ensemble schema, whose
data is "intentionally discarded". Two things outlived it: `scripts/meta_review/
grading_queries.py`, a grading pass that SELECTed from both and so raised
`no such table` in every mode, and the docstring of `app/event_log.py`, the
module a reader lands on first for IV forensics, which described the dropped
tables in the present tense and never named `inner_voice_observations`, the
real store. The script was deleted as removed-by-design (its sibling
`prompt_diff.py` and `replay.py` are live and stay); the docstring was
rewritten.

The table names are read off the DROP statements rather than restated, so the
check follows the store: a third legacy table dropped tomorrow is covered by
the same assertion.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_DROP_TABLE = re.compile(r"DROP TABLE IF EXISTS\s+(\w+)")


def _dropped_tables() -> set[str]:
    src = (ROOT / "usage_store.py").read_text(encoding="utf-8")
    dropped = set(_DROP_TABLE.findall(src))
    assert {"inner_voice_critiques", "inner_voice_interventions"} <= dropped, (
        f"the store no longer drops the legacy tables this test is about: {dropped}")
    return dropped


def _mentions(path: Path, names: set[str]) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [f"{path.relative_to(ROOT)}: {n}" for n in sorted(names) if n in text]


def test_event_log_docstring_names_the_live_store_not_the_dropped_tables():
    doc = ROOT / "app" / "event_log.py"
    text = doc.read_text(encoding="utf-8")
    assert "inner_voice_observations" in text, (
        "app/event_log.py never names the table that actually holds observer decisions")
    # The one allowed mention is the parenthetical that says they are dropped;
    # a present-tense sentence about them is what #834 removed.
    for name in _dropped_tables():
        assert text.count(name) <= 1, f"app/event_log.py names {name} more than once"
        after = text.split(name, 1)[1][:200] if name in text else "dropped"
        assert "dropped" in after, (
            f"app/event_log.py describes {name} without saying it is dropped")


def test_no_script_under_meta_review_queries_a_dropped_table():
    """`grading_queries.py` could never return a row again — `outcome_addressed`
    appeared in no other file in the tree. A query module over the dropped pair
    must not come back under any name."""
    names = _dropped_tables()
    hits = [h for p in sorted((ROOT / "scripts" / "meta_review").glob("*.py"))
            for h in _mentions(p, names)]
    assert hits == [], f"scripts/meta_review still reads a table the store drops: {hits}"
    assert not (ROOT / "scripts" / "meta_review" / "grading_queries.py").exists(), (
        "grading_queries.py was deleted as removed-by-design (#834); re-target it "
        "at inner_voice_observations + event_logs/ if it is ever needed again")
