"""The MC autonomy tab and the task route enumerate ONE directory the same way — #1594.

`app/routers/mc_ui.py::_summarize_autonomy` feeds the autonomy tab that the agent
reads (`mc_get_state` → `GET /api/mc/navigate` detail, `mc_ui.py:126`). It used to
walk `_AUTONOMY_DIR.glob("*.md")` itself: skip `_config.md`, then count every file
whose first bytes are `---`, reading only the first 2000 bytes, and filing a file
with no `status:` key under `str(fm.get("status") or "scheduled")`.

The task route `GET /api/autonomy/tasks` gates the same glob on the task-FILENAME
pattern `^\\d+-` instead (`app/routers/autonomy.py:236`), as does the human dashboard
(`app/routers/dashboard.py:668`). Two gates on one directory means the looser one
decides the agent's number, in both directions, and it did:

* `~/obsidian/autonomy/meta-analysis-2026-06-03.md` is a prose note that already
  carries frontmatter and no `status:` key, so it was counted as a task and bucketed
  as `scheduled` — a status NO real task can report, because no autonomy task file
  says `status: scheduled` and the only `"scheduled"` literal in `app/`,
  `agent_mcp/` and `workers/` was that default. The lone `scheduled: 1` on the live
  board was the phantom, not a task.
* three real tasks were *dropped*, because their closing `---` sits past byte 2000
  and `len(parts) < 3` fired: `79-retention-sweep.md` (closes at byte 2734),
  `86-nightly-iv-metrics-series.md` (3373), `90-corpus-shape-trend.md` (2028).

So the tab reported `{'total': 30, ...}` against the 32 task files the route lists:
32 real tasks − 3 dropped + 1 phantom = 30. The summary is now built from the task
route's own enumeration and parse path — `autonomy_task_files()` /
`list_parsed_tasks()` — so one name gate, one parser and one status default decide
both surfaces.

Each test below writes a temporary autonomy directory and patches
`app.routers.autonomy._AUTONOMY_DIR` ONCE: if the summary were still carrying its
own `_AUTONOMY_DIR` constant, it would read the live vault instead of the fixture and
the comparison to the route would be vacuous.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.routers import autonomy as ROUTER  # noqa: E402
from app.routers import mc_ui  # noqa: E402

# Task-shaped: the filename is the task id, which is what makes it a task.
TASK_UP_NEXT = """---
id: 42
name: Morning brief triage
status: up_next
frequency: daily
---

# body
"""

# Report-shaped: a real report file with *legal* frontmatter — the frontmatter a
# vault-maintenance pass would add under "fix missing frontmatter" — and a status
# key, so the loose gate has nothing to reject it on but the name.
REPORT_WITH_LEGAL_FRONTMATTER = """---
id: 0
name: Skill Lint Report
status: up_next
tags:
- autonomy
- skill-lint
segment: autonomy
type: report
timestamp: '2026-09-27T00:42:52'
---

# Skill Lint Report — 2026-09-27T00:42:52
"""

# A task with no `status:` key at all. The task route reports it with the parse
# path's default; the summary must report the same string, not "scheduled".
TASK_WITHOUT_STATUS_KEY = """---
id: 43
name: Corpus shape trend
frequency: daily
---

# body
"""

# 200 top-level frontmatter keys, ~11 bytes each: pushes the closing `---` past
# byte 2000 the way the three live tasks above do. Keys the projection ignores.
_LONG_FM_PAD = "".join(f"pad_{i}: x\n" for i in range(200))
TASK_WITH_LONG_FRONTMATTER = (
    "---\nid: 79\nname: Retention sweep\nstatus: in_progress\n"
    + _LONG_FM_PAD
    + "---\n\n# body\n"
)


@pytest.fixture
def autonomy_tree(tmp_path, monkeypatch):
    """An empty autonomy directory at the path the route reads; returns a writer."""
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", dirn)
    return dirn


def _write(dirn: Path, **files: str) -> Path:
    """Write `{"42-task.md": text}` into the patched directory (kwargs can't hold
    the hyphenated task filenames, so a dict it is)."""
    for name, text in files.items():
        (dirn / name).write_text(text, encoding="utf-8")
    return dirn


def _route_tasks() -> list[dict]:
    """The tasks `GET /api/autonomy/tasks` lists, over HTTP, from the same tree."""
    app = FastAPI()
    app.include_router(ROUTER.router)
    resp = TestClient(app).get("/api/autonomy/tasks")
    assert resp.status_code == 200, resp.text
    return json.loads(resp.text)["tasks"]


def test_report_file_with_legal_frontmatter_is_never_counted_as_a_task(autonomy_tree):
    """Clause 1: the name gate, not the frontmatter gate, decides the tab.

    `42-task.md` and `report.md` are indistinguishable to the old gate — both open
    with `---`, both parse, both say `status: up_next`, both fit inside 2000 bytes —
    so it counted both and the tab reported two tasks where the board listed one.
    """
    _write(autonomy_tree, **{
        "42-task.md": TASK_UP_NEXT,
        "report.md": REPORT_WITH_LEGAL_FRONTMATTER,
    })

    out = mc_ui._summarize_autonomy()

    assert out["total"] == 1, (
        f"a report-shaped file with valid frontmatter was counted as a task: {out}"
    )
    assert out["by_status"] == {"up_next": 1}


def test_summary_and_task_route_agree_on_count_and_on_the_default_status(autonomy_tree):
    """Clause 2: one enumeration, not two, including the no-`status:` default.

    Half one: the summary total equals the number of tasks the route lists from the
    same directory. Half two: for a counted task with no `status:` key, the bucket
    the summary files it under is the string the route reports for that same task —
    which is the parse path's default, not the `or "scheduled"` the summary used.
    """
    _write(autonomy_tree, **{
        "42-task.md": TASK_UP_NEXT,
        "43-corpus-shape.md": TASK_WITHOUT_STATUS_KEY,
        "report.md": REPORT_WITH_LEGAL_FRONTMATTER,
    })

    out = mc_ui._summarize_autonomy()
    tasks = _route_tasks()

    assert [t["id"] for t in tasks] == [42, 43] or sorted(t["id"] for t in tasks) == [42, 43]
    assert out["total"] == len(tasks), (
        f"the tab counted {out['total']} while GET /api/autonomy/tasks lists "
        f"{len(tasks)} from the same directory"
    )
    route_status = next(t["status"] for t in tasks if t["id"] == 43)
    assert route_status == "draft", (
        "the task route no longer reports a status-less task as 'draft', so the "
        "string this test compares against needs updating with it"
    )
    assert out["by_status"].get("draft") == 1, (
        f"the status-less task was not filed under the route's default: {out}"
    )
    assert "scheduled" not in out["by_status"], (
        f"'scheduled' is a status no autonomy task file can carry; its bucket is "
        f"the phantom this fix removes: {out}"
    )


def test_task_whose_frontmatter_closes_past_byte_2000_is_counted(autonomy_tree):
    """Clause 3: the read window must not silently delete a real task.

    `79-retention-sweep.md`, `86-nightly-iv-metrics-series.md` and
    `90-corpus-shape-trend.md` are excluded from the tab today for this reason
    alone. The fixture asserts its own byte offset, because a padding mistake would
    leave this test passing for the wrong reason.
    """
    close = TASK_WITH_LONG_FRONTMATTER.find("---", 3)
    assert close > 2000, (
        f"fixture is not exercising the read window: frontmatter closes at byte {close}"
    )
    _write(autonomy_tree, **{
        "42-task.md": TASK_UP_NEXT,
        "79-retention-sweep.md": TASK_WITH_LONG_FRONTMATTER,
    })

    out = mc_ui._summarize_autonomy()

    assert out["total"] == 2, (
        f"a task was dropped because its frontmatter runs past byte 2000: {out}"
    )
    assert out["by_status"] == {"up_next": 1, "in_progress": 1}
    assert out["total"] == len(_route_tasks())


def test_the_frontmatter_2326_added_to_the_referential_integrity_report_keeps_it_out_of_the_tab(
        autonomy_tree):
    """#2326's own output, fed through the name gate: a conformant report is still not a task.

    The item's hazard was that the two fixes had to land together — #1594's UI gate and the
    generator's new fence — because a report under `autonomy/` that carries frontmatter used
    to be counted by a `head.startswith("---")` test. The gate moved to
    `_TASK_NAME_RE` first, so this node is not the discriminator: it is the cross-boundary
    check that the *specific bytes the generator now emits* — `render_report`'s real string,
    not a hand-written report-shaped constant — still fail the name test. The hand-written
    `REPORT_WITH_LEGAL_FRONTMATTER` above covers the shape; only the generator's own text
    covers a key that could confuse the parser (`type: note`, `segment: autonomy`, a quoted
    `generated_at`) or a dated copy landing in the `referential-integrity/` subdirectory that
    autonomy task #94's `cp` step makes each night and the glob never walks.
    """
    from scripts.maintenance import referential_integrity as ri

    text = ri.render_report([
        {"file": "autonomy/9-x.md", "line": 3, "target": "knowledge/gone.md",
         "verdict": "dangling", "why": "no file", "how_checked": "is_file"},
    ], None)
    assert text.startswith("---\n"), "the generator's fence is what this node is about"
    _write(autonomy_tree, **{"referential-integrity-latest.md": text})
    dated = autonomy_tree / "referential-integrity"      # task #94's `cp` destination
    dated.mkdir()
    (dated / "2026-10-07.md").write_text(text, encoding="utf-8")

    files = ROUTER.autonomy_task_files()
    assert [f.name for f in files] == [], (
        f"the report, or its dated copy, was enumerated as a task: {[f.name for f in files]}")
    assert mc_ui._summarize_autonomy()["total"] == 0, (
        "the tab counted the report even though the route did not — the two surfaces "
        "diverged back into the #1594 shape")

    # A real task beside it still counts, so the empty result above is the name gate and not
    # an enumeration that never reached the directory.
    _write(autonomy_tree, **{"43-corpus-shape-trend.md": TASK_WITHOUT_STATUS_KEY})
    assert [f.name for f in ROUTER.autonomy_task_files()] == ["43-corpus-shape-trend.md"], (
        "positive control: with the report present the gate stopped returning the one task")
