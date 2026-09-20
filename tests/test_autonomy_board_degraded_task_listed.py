"""The board lists a degraded autonomy task instead of dropping it — #1014 clause 4.

`app/routers/autonomy.py::_autonomy_parse` is the reader behind
`GET /api/autonomy/tasks`, which `web/src/api.ts` renders as the Mission Control
autonomy board. It parsed with a bare `yaml.safe_load` and
`except Exception: return None`, and the list handler does `if task is None:
continue`. So on a YAML-broken file the board omitted the task entirely while
the scheduler — same directory, graduated-recovery parser — loaded it and
dispatched it.

That is the silent direction of the exact incident this section of the
architecture doc exists to prevent (2026-05-28: 34 of 40 task files
dormant-killed). A degraded fleet now presents as "31 tasks, none broken" while
the scheduler runs 32. `validate_tasks.py` catches the YAML break itself, so the
alert exists; what did not exist was any signal that the board and the
scheduler were looking at different fleets.

The reader now goes through `parse_frontmatter_text` with the shared
`AUTONOMY_TASK_FIELDS` list, so a degraded task comes back as a normal-looking
row carrying `_yaml_broken: true`, and the listed count equals the count the
scheduler loads. The board's own write endpoint is unchanged and still rejects
such a file — it raises 500 when the parse says degraded, which is why the
board's writer was never a fourth loss vector.

The `task-write` 500 half is pinned here too, because "list it" and "don't
rewrite it" are one decision viewed from two ends: making a file visible on the
board must not simultaneously make it writable in place.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autonomy as SCHED
from app.routers import autonomy as ROUTER

BROKEN = """---
id: 3
name: nightly: reflection: signals
status: up_next
frequency: daily
preferred_hours: [1, 2, 3, 4]
depends_on: 42
model: eco
stale_bypass_hours: 24
---

# body
"""

HEALTHY_A = """---
id: 1
name: Morning brief triage
status: up_next
frequency: every-15min
---

# body
"""

HEALTHY_B = """---
id: 2
name: Knowledge health report
status: up_next
frequency: daily
---

# body
"""


@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    (dirn / "1-morning-brief.md").write_text(HEALTHY_A, encoding="utf-8")
    (dirn / "2-knowledge-health.md").write_text(HEALTHY_B, encoding="utf-8")
    (dirn / "3-nightly-signals.md").write_text(BROKEN, encoding="utf-8")
    # A non-task file: must stay out of BOTH counts, or "listed == loaded"
    # would be satisfiable by listing everything in the directory.
    (dirn / "_config.md").write_text("---\ntype: config\n---\n\nnotes\n", encoding="utf-8")
    monkeypatch.setattr(SCHED, "AUTONOMY_DIR", dirn)
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", dirn)
    return dirn


@pytest.fixture
def client(autonomy_dir):
    app = FastAPI()
    app.include_router(ROUTER.router)
    with TestClient(app) as c:
        yield c


def _task_files(dirn: Path) -> list[str]:
    """The scheduler's own notion of which files are tasks: `\\d+-` prefix."""
    return sorted(p.name for p in dirn.glob("*.md") if re.match(r"\d+-", p.name))


# ── clause 4 ─────────────────────────────────────────────────────────────────


def test_the_board_lists_a_degraded_task_with_a_flag(client, autonomy_dir):
    rows = client.get("/api/autonomy/tasks").json()["tasks"]
    degraded = [r for r in rows if r["id"] == 3]
    assert len(degraded) == 1, f"degraded task omitted from the board: {rows!r}"
    assert degraded[0].get("_yaml_broken") is True


def test_the_degraded_row_carries_the_fields_the_scheduler_used(autonomy_dir):
    """Listing it as an empty shell would satisfy "a row is present" and still
    hide the task: the board must show what the file says, recovered the same
    way the scheduler recovers it."""
    board = ROUTER._autonomy_parse(autonomy_dir / "3-nightly-signals.md")
    sched = SCHED._parse_task_file(autonomy_dir / "3-nightly-signals.md")
    assert board is not None and board["_yaml_broken"] is True
    assert board["preferred_hours"] == [1, 2, 3, 4] == sched["preferred_hours"]
    assert board["depends_on"] == 42 == sched["depends_on"]
    assert board["model"] == "eco" == sched["model"]
    assert board["stale_bypass_hours"] == 24 == sched["stale_bypass_hours"]


def test_the_listed_count_equals_the_count_the_scheduler_loads(client, autonomy_dir):
    """The whole clause in one equality: board and scheduler see one fleet."""
    rows = client.get("/api/autonomy/tasks").json()["tasks"]
    loaded = SCHED._all_board_tasks(autonomy_dir)
    assert len(rows) == len(loaded) == 3, f"board {len(rows)} vs scheduler {len(loaded)}"


def test_the_positive_control_that_makes_that_equality_mean_something(autonomy_dir):
    """`_all_board_tasks` genuinely loads the broken file — if it too returned
    2, the equality above would pass on two readers that both drop it."""
    loaded = SCHED._all_board_tasks(autonomy_dir)
    assert {t["id"] for t in loaded} == {1, 2, 3}
    assert len(_task_files(autonomy_dir)) == 3


def test_a_healthy_task_is_not_flagged(client):
    rows = client.get("/api/autonomy/tasks").json()["tasks"]
    healthy = [r for r in rows if r["id"] == 1]
    assert len(healthy) == 1
    assert not healthy[0].get("_yaml_broken")


def test_the_board_write_endpoint_still_refuses_a_degraded_file(client, autonomy_dir):
    """The other end of the same decision: visible, not rewritable in place."""
    before = (autonomy_dir / "3-nightly-signals.md").read_text(encoding="utf-8")
    resp = client.post("/api/autonomy/task-write",
                       json={"id": 3, "status": "draft"})
    assert resp.status_code == 409, f"expected 409, got {resp.status_code}: {resp.text}"
    assert (autonomy_dir / "3-nightly-signals.md").read_text(encoding="utf-8") == before


def test_the_board_write_endpoint_still_writes_a_healthy_file(client, autonomy_dir):
    """Guard-vs-ban control for the board half, same as the MCP half."""
    resp = client.post("/api/autonomy/task-write",
                       json={"id": 1, "description": "edited from the board"})
    assert resp.status_code == 200, resp.text
    text = (autonomy_dir / "1-morning-brief.md").read_text(encoding="utf-8")
    assert "edited from the board" in text
    fm = yaml.safe_load(text.split("---\n", 2)[1])
    assert fm["status"] == "up_next"
