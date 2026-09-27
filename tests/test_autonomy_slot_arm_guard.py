"""Arming a slot-bound autonomy task is refused while its slot is off — #1555.

`app/llm_slots.py` is the one authority on whether an optional LLM engine is meant
to be running, and until now neither status-write surface consulted it. That let
#85 (`Secondary Routing Eval`, `frequency: daily`) go `draft -> up_next` on
2026-09-26T16:38:14Z with `secondary_enabled: false` still in force — an
unattributed write against a park whose reason was in the commit that made it
(`a6c363d6`, for #1328: "#1328 uninstalled the secondary engine, so there is no
slot to measure; flipping the flag now OOMs GPU 2"). With the slot off,
`resolve_model_alias('secondary')` answers `primary`, so the dispatch is 120 trials
of one engine presented as a comparison, billed to the box's main model next to a
loaded 95 GiB table.

The eval already refused this from its own side (`EXIT_SLOT_DISABLED = 7`, pinned in
`tests/test_secondary_routing_eval.py`); the writers were the hole. So this guard
BLOCKS where its neighbour `status_change_note` only reports — because a job
claiming its own task is a legitimate write and arming a task onto a dead slot is
not. Both edges of the write are covered, and
`test_every_writer_that_records_a_status_change_also_checks_the_slot` is what stops
the next writer from being added to one surface only: a guard on one of two writers
is not a guard.

Task #85, the incident's subject, was itself retired by Alan's ruling of 2026-09-26
(#1577) and its task file deleted — `secondary_enabled` has been false since
`551e9044`, and djev is not a chat slot to re-scope onto. The guard is untouched by
that: what it refuses is any task naming a switched-off slot, which is why the
live-fleet node below scans every declaration in the fleet instead of asserting that
one particular task still makes one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import autonomy as A
from app import llm_slots
from app.routers import autonomy as ROUTER

REPO_ROOT = Path(__file__).resolve().parents[1]
VAULT_AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
SLOT = "agent-llm-secondary"
FLAG = "secondary_enabled"

TASK_BODY = "\n## Activity Log\n\n- 2026-09-20T03:00:00Z: run completed\n"


def _task_file(dirn: Path, task_id: int, name: str, status: str, **extra) -> Path:
    fm = {"type": "autonomy", "segment": "autonomy", "id": task_id, "name": name,
          "status": status, "frequency": "daily"}
    fm.update(extra)
    path = dirn / f"{task_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n{TASK_BODY}",
                    encoding="utf-8")
    return path


@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    import agent_mcp.autonomy as MCP
    monkeypatch.setattr(A, "AUTONOMY_DIR", dirn)
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", dirn)
    monkeypatch.setattr(MCP, "AUTONOMY_DIR", dirn)
    return dirn


@pytest.fixture
def client(autonomy_dir):
    app = FastAPI()
    app.include_router(ROUTER.router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def slot_off(monkeypatch):
    """Pin the slot OFF rather than reading the live flag: the guard's behaviour
    must not depend on what this machine happens to have booted."""
    monkeypatch.setattr(llm_slots, "is_enabled", lambda program, config=None: False)


# ── the rule itself ──────────────────────────────────────────────────────


def test_arming_a_slot_bound_task_is_refused_while_its_slot_is_off(slot_off):
    task = {"id": 85, "requires_slot": SLOT}
    reason = A.slot_arm_block(task, "up_next")
    assert reason is not None, "the writer would arm a job onto a slot that is not booted"
    assert FLAG in reason, f"the refusal must name the flag that has to change: {reason}"
    assert SLOT in reason


def test_parking_a_slot_bound_task_is_never_blocked(slot_off):
    """The guard must not stop stopping. #68's disable sat unattributed for a day
    (#1127); a guard that makes parking harder gets routed around within a week."""
    task = {"id": 85, "requires_slot": SLOT}
    for status in A.DISPATCH_STOPPING_STATUSES:
        assert A.slot_arm_block(task, status) is None, f"parking as {status} was blocked"


def test_a_task_that_declares_no_slot_is_untouched_by_the_guard(slot_off):
    """Denominator, not decoration: without this the guard could pass by refusing
    every write in the fleet, and `slots()` returning nothing would look identical
    to the guard working."""
    assert A.slot_arm_block({"id": 68}, "up_next") is None
    assert A.slot_arm_block({"id": 68, "requires_slot": ""}, "up_next") is None
    assert A.slot_arm_block(None, "up_next") is None


def test_the_guard_arms_the_task_once_the_slot_is_enabled(monkeypatch):
    """Both directions, or the guard is just a ban on this one task."""
    monkeypatch.setattr(llm_slots, "is_enabled", lambda program, config=None: True)
    assert A.slot_arm_block({"id": 85, "requires_slot": SLOT}, "up_next") is None


def test_the_slot_it_names_is_a_slot_the_engine_knows(slot_off):
    """A typo in `requires_slot` would make the guard refuse forever on a program
    that does not exist, and `flag` would fall back to the program name — so the
    flag named in the refusal has to be a real config key."""
    known = {program for _flag, program, _on in llm_slots.slots()}
    assert SLOT in known
    named_flag = next(f for f, program, _on in llm_slots.slots() if program == SLOT)
    assert named_flag == FLAG


# ── the two write surfaces ───────────────────────────────────────────────


def test_the_mcp_writer_refuses_to_arm_it_and_changes_nothing(autonomy_dir, slot_off):
    """`autonomy_write_task` is the surface an agent actually calls."""
    import agent_mcp.autonomy as MCP
    path = _task_file(autonomy_dir, 85, "Secondary Routing Eval", "draft",
                      requires_slot=SLOT, skill_name="secondary-routing-eval")
    before = path.read_text(encoding="utf-8")
    out = json.loads(MCP._handle_write({"id": 85, "status": "up_next"}))
    assert "error" in out, f"the tool armed the task: {out}"
    assert FLAG in out["error"]
    assert path.read_text(encoding="utf-8") == before, \
        "a refused write must leave the file byte-identical, status and log included"


def test_the_http_writer_refuses_the_same_write_with_409(client, autonomy_dir, slot_off):
    """The board / Mission Control surface, refused the same way. Two writers, one
    rule; the MCP half alone would leave the UI able to do the thing."""
    path = _task_file(autonomy_dir, 85, "Secondary Routing Eval", "draft",
                      requires_slot=SLOT, skill_name="secondary-routing-eval")
    before = path.read_text(encoding="utf-8")
    r = client.post("/api/autonomy/task-write", json={"id": 85, "status": "up_next"})
    assert r.status_code == 409, r.text
    assert FLAG in r.text
    assert path.read_text(encoding="utf-8") == before


def test_the_http_writer_still_writes_ordinary_fields_while_refusing(client, autonomy_dir,
                                                                     slot_off):
    """The refusal is about arming, not a freeze on the task: #68's lesson is that
    a parked task still has to be editable, or people edit the file by hand and
    lose the audit line entirely."""
    path = _task_file(autonomy_dir, 85, "Secondary Routing Eval", "draft",
                      requires_slot=SLOT, skill_name="secondary-routing-eval")
    r = client.post("/api/autonomy/task-write",
                    json={"id": 85, "priority": "high", "timeout_seconds": 5400})
    assert r.status_code == 200, r.text
    front = yaml.safe_load(path.read_text(encoding="utf-8").split("---\n")[1])
    assert front["priority"] == "high" and front["timeout_seconds"] == 5400
    assert front["status"] == "draft"


def test_the_http_writer_arms_it_when_the_slot_is_live(client, autonomy_dir, monkeypatch):
    monkeypatch.setattr(llm_slots, "is_enabled", lambda program, config=None: True)
    path = _task_file(autonomy_dir, 85, "Secondary Routing Eval", "draft",
                      requires_slot=SLOT, skill_name="secondary-routing-eval")
    r = client.post("/api/autonomy/task-write", json={"id": 85, "status": "up_next"})
    assert r.status_code == 200, r.text
    front = yaml.safe_load(path.read_text(encoding="utf-8").split("---\n")[1])
    assert front["status"] == "up_next"
    body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "status changed: draft -> up_next" in body, \
        "an allowed arm still records the transition (#1127)"


# ── coverage of the surface, so a third writer cannot be added silently ───


def test_every_writer_that_records_a_status_change_also_checks_the_slot():
    """Both writers that owe a status-change note (#1127) are exactly the writers
    that can arm a task from outside the scheduler, so the note-call is the
    enumerable denominator for the guard: add a third writer with a note and this
    goes red until it consults the slot too."""
    note_writers, guarded = [], []
    for path in [REPO_ROOT / "agent_mcp" / "autonomy.py",
                 REPO_ROOT / "app" / "routers" / "autonomy.py"]:
        text = path.read_text(encoding="utf-8")
        if "status_change_note(" in text:
            note_writers.append(path.name)
            if "slot_arm_block(" in text:
                guarded.append(path.name)
    assert len(note_writers) == 2, f"expected both write surfaces, found {note_writers}"
    assert guarded == note_writers, f"status writers with no slot check: {guarded}"


def test_the_guard_is_called_by_the_writers_and_by_nobody_else():
    """Exactly two call sites, both of them a write surface. The scheduler arms
    tasks through `_update_task_field` — retry recovery, a run completing, a stale
    chain re-arming — and guarding those would stop recovery running, which is how
    a guard gets switched off. A third call site is a new blocked path and should
    be a decision, not a diffusion."""
    callers = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("*.py")
        if not any(part.startswith(".") or part in ("node_modules", "tests")
                   for part in p.relative_to(REPO_ROOT).parts)
        and "slot_arm_block(" in p.read_text(encoding="utf-8")
    )
    # One definition (`app/autonomy.py`) and two call sites, both write surfaces.
    assert callers == ["agent_mcp/autonomy.py", "app/autonomy.py",
                       "app/routers/autonomy.py"], callers


def test_the_declaration_survives_a_degraded_parse():
    """`AUTONOMY_TASK_FIELDS` is what all three readers recover from a
    YAML-broken file. A recovered record missing `requires_slot` reads as a task
    that declares no slot — so the guard would find nothing to refuse, and the
    fails-open half of #1014 would reappear through the new field."""
    from agent_mcp._shared import AUTONOMY_TASK_FIELDS

    assert "requires_slot" in AUTONOMY_TASK_FIELDS


# ── the live declaration ─────────────────────────────────────────────────

#: The live fleet held 35 task files when this was measured on 2026-09-27. The floor
#: sits well below that and well above zero, because a scan of a directory nobody read
#: reports no declarations for exactly the reason a clean fleet does.
MIN_LIVE_TASK_FILES = 20


def _switched_off_declarations(autonomy_dir: Path) -> list[str]:
    """One entry per task file naming a program `llm_slots` says is switched off.

    The reader is the same authority the guard consults, so the live scan and the
    writers cannot disagree about what a declaration means. A file whose front matter
    will not parse falls back to a text search rather than reading as undeclared —
    the fails-closed rule `test_the_declaration_survives_a_degraded_parse` exists for.
    """
    out: list[str] = []
    for path in sorted(autonomy_dir.glob("*.md")):
        body = path.read_text(encoding="utf-8", errors="replace")
        try:
            front = yaml.safe_load(body.split("---\n")[1]) or {}
            declared = str(front.get("requires_slot") or "").strip()
        except Exception:  # noqa: BLE001 — a broken file must not hide a declaration
            found = re.search(r"^requires_slot:[ \t]*(\S+)[ \t]*$", body, re.MULTILINE)
            declared = found.group(1) if found else ""
        if declared and llm_slots.slot_flag(declared) and not llm_slots.is_enabled(declared):
            out.append(f"{path.name}: {declared}")
    return out


def test_no_live_task_declares_a_slot_the_box_has_switched_off(
        tmp_path, monkeypatch):
    """The part of the fix that lives outside this repo: the guard only fires for a
    task that declares its slot, so the declarations themselves have to be read.

    Until #1577 this asserted the reverse — that a task naming the instrument declared
    `SLOT`, the wiring a `draft -> up_next` would have needed. Alan's ruling deleted
    that task, so the live fleet names a switched-off program nowhere and the
    assertion inverts with the guard's purpose intact. It covers every program
    `app.llm_slots` knows about rather than one id, because `agent-djev` is the second
    optional slot and a `== SLOT` comparison would ignore a declaration naming it.
    """
    files = list(VAULT_AUTONOMY_DIR.glob("*.md"))
    assert len(files) > MIN_LIVE_TASK_FILES, (
        f"only {len(files)} task files under {VAULT_AUTONOMY_DIR}: that is a "
        "denominator failure, not evidence that nothing declares a dead slot")

    assert _switched_off_declarations(VAULT_AUTONOMY_DIR) == [], (
        "a live task names a slot this machine has switched off; the writers refuse to "
        "arm it and `scripts/autonomy/validate_tasks.py` reports it (#1577)")

    # The control measures the scan, not the live flag, so it sets the flag itself:
    # asserting the fixture is reported while `secondary_enabled` happens to be false
    # would fail on a box that boots the second engine and blame the scan for the
    # machine. Both directions are pinned, so what drives the verdict is the flag and
    # not the fixture's presence.
    fixture = _task_file(tmp_path, 85, "Secondary Routing Eval", "draft",
                         requires_slot=SLOT)
    monkeypatch.setattr(llm_slots, "is_enabled", lambda program, config=None: False)
    assert _switched_off_declarations(tmp_path) == [f"{fixture.name}: {SLOT}"], (
        "the scan cannot see a declaration, so the assertion above could not fail")
    monkeypatch.setattr(llm_slots, "is_enabled", lambda program, config=None: True)
    assert _switched_off_declarations(tmp_path) == [], (
        "the scan reports a declaration the flag has switched on, so it is not the "
        "flag that drives the verdict and the assertion above proves nothing")
