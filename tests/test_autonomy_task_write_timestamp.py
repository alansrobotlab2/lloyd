"""The instant `POST /api/autonomy/task-write` writes into a task's front matter
is the real wall-clock UTC instant — backlog #1128.

Why this file exists
--------------------
`app/routers/autonomy.py` built its stamp with
`datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")`. `datetime.now()` is naive — it
reads the *host's* wall clock — and the host is
`America/Los_Angeles (PDT, -0700)`, while the `Z` suffix asserts UTC. So every
task write through the Mission Control autonomy page (status change, save,
create: `web/src/api.ts` `autonomyWriteTask` ← `AutonomyPage.tsx`) recorded a
time 7 hours behind the moment it happened, in a form no reader can distrust.
Measured on 2026-09-20T00:06Z: the endpoint wrote `2026-09-19T17:06:25Z` while
the true instant was `2026-09-20T00:06:25Z` — a drift of 25,200.3 s, exactly
7.00 h.

That is not cosmetic, because the scheduler parses the field as UTC.
`autonomy._parse_iso` maps a trailing `Z` to `+00:00` — and stamps an
offset-free value as UTC too — and seven functions read the result:
`recover_stuck_tasks`, `run_task`'s "already in_progress; not starting a second
run" guard, `_is_task_due`, `_is_dependency_met`, `_in_failure_cooldown`,
`next_run_gap`, `_priority_key`. With a 25,200 s inflation, a task set to
`in_progress` from the UI is past its default 1800 s `timeout_seconds` the
instant it is written, so the recovery pass flips it back to `up_next` and logs
a false `Recovered from in_progress after 25200s` — observed, verbatim, while
confirming this defect — and the double-run guard can never trip for such a
task.

What is deliberately NOT changed here
-------------------------------------
`agent_mcp/autonomy.py` also writes `Z` stamps, but from
`datetime.datetime.now(datetime.timezone.utc)` — true UTC. It is the reference
correct form, and the scan below pins it so a later "make them consistent" edit
cannot quietly make it wrong. A `Z` on disk is therefore not this endpoint's
fingerprint, which is why no on-disk task file is rewritten by this change.

The same scan over this router tolerates one shape: `_to_iso`, in two places,
appends `Z` to a `datetime` that YAML produced from front matter. Those receivers
are parameters, not clock reads — re-serialising a value the file already
carried. If a *naive* (offset-free) timestamp reaches them they label it UTC too,
which is the same class of error one level down; that residual is recorded on
#1128, and clause 2 of the acceptance is scoped to clock reads in this module.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import autonomy as A
from app.routers import autonomy as ROUTER

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_SRC = REPO_ROOT / "app" / "routers" / "autonomy.py"
MCP_SRC = REPO_ROOT / "agent_mcp" / "autonomy.py"

TOLERANCE_SECONDS = 5           # the bound clauses 1 and 2 name
DRIFT_BEFORE_FIX_SECONDS = 25_200   # 7.00 h: how far the naive stamp sat off UTC


# ── the endpoint, driven over its real boundary ──────────────────────────────

@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    """`~/obsidian/autonomy` is a protected vault path, so every test here
    redirects the router's own global — the one `_autonomy_find_file`,
    `_autonomy_next_id` and `_autonomy_write_file` all resolve through — plus the
    scheduler's, for the recovery pass that reads what the write left behind."""
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", dirn)
    monkeypatch.setattr(A, "AUTONOMY_DIR", dirn)
    return dirn


@pytest.fixture
def client(autonomy_dir):
    app = FastAPI()
    app.include_router(ROUTER.router)
    with TestClient(app) as c:
        yield c


def _existing_task(dirn: Path, task_id: int = 68) -> Path:
    path = dirn / f"{task_id}-morning-brief-triage.md"
    path.write_text(
        "---\ntype: autonomy\nid: 68\nname: Morning Brief Triage\n"
        "status: up_next\nfrequency: every-15min\ntimeout_seconds: 1800\n"
        "created: '2026-04-02T00:00:00Z'\n---\n\n# Body\n",
        encoding="utf-8")
    return path


def _front_matter(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---\n", 2)[1])


def _drift_outside(value, earlier: datetime, later: datetime) -> float:
    """How far `_parse_iso(value)` sits outside the window the write happened in.

    The scheduler's own parser decides whether a task is stuck, so the claim is
    graded through it rather than through `fromisoformat`, and the window — a
    bracket captured around the POST, not one instant — is what makes the bound
    real: a value inside the bracket is off by at most the bracket's width.
    Returns 0.0 when the parsed instant falls inside it.
    """
    parsed = A._parse_iso(value)
    assert parsed is not None, f"_parse_iso could not read {value!r} at all"
    return max((earlier - parsed).total_seconds(),
               (parsed - later).total_seconds(), 0.0)


# ── clause 1: an update records the real instant ─────────────────────────────


def test_an_id_write_stamps_updated_with_the_true_utc_instant(client, autonomy_dir):
    """Clause 1: parsed by `autonomy._parse_iso`, within 5 s of
    `datetime.now(timezone.utc)`. Before the fix the same POST wrote a value
    25,200 s (7.00 h) off — 5,040 times this bound."""
    path = _existing_task(autonomy_dir)
    earlier = datetime.now(timezone.utc)
    r = client.post("/api/autonomy/task-write",
                    json={"id": 68, "status": "in_progress"})
    later = datetime.now(timezone.utc)
    assert r.status_code == 200, r.text

    raw = _front_matter(path)["updated"]
    assert _drift_outside(raw, earlier, later) <= TOLERANCE_SECONDS, (
        f"task-write stamped {raw!r}; the write happened between "
        f"{earlier.isoformat()} and {later.isoformat()}")


def test_saving_edits_without_a_status_change_stamps_the_true_instant(client,
                                                                      autonomy_dir):
    """The save path (`AutonomyPage.tsx`'s save button posts no status) stamps
    `updated` on the same line, so it is pinned on its own call."""
    path = _existing_task(autonomy_dir)
    earlier = datetime.now(timezone.utc)
    r = client.post("/api/autonomy/task-write", json={"id": 68, "priority": "high"})
    later = datetime.now(timezone.utc)
    assert r.status_code == 200, r.text
    fm = _front_matter(path)
    assert fm["priority"] == "high"
    assert _drift_outside(fm["updated"], earlier, later) <= TOLERANCE_SECONDS
    assert fm["status"] == "up_next", "a save must not move the status by itself"


def test_the_stamp_names_its_offset_instead_of_leaving_it_to_be_guessed(client,
                                                                       autonomy_dir):
    """The written value carries an explicit UTC designator. The neighbouring
    routers (`app/routers/backlog.py`, `sessions.py`, `messages.py`) write
    `isoformat()` with no `Z` and no offset — ambiguous rather than false, since
    reading them needs the host's zone — and this module keeps the `Z` form the
    scheduler and `agent_mcp/autonomy.py` already write. The point is that the
    label now denotes what it says."""
    path = _existing_task(autonomy_dir)
    client.post("/api/autonomy/task-write", json={"id": 68, "status": "in_progress"})
    raw = _front_matter(path)["updated"]
    assert raw.endswith("Z"), f"{raw!r} carries no UTC designator"
    assert A._parse_iso(raw).utcoffset() == timezone.utc.utcoffset(None)


# ── clause 2: a create records the real instant ──────────────────────────────


def test_creating_a_task_stamps_created_and_updated_with_the_true_utc_instant(
        client, autonomy_dir):
    """Clause 2: both fields, each parsed the scheduler's way, each within 5 s."""
    earlier = datetime.now(timezone.utc)
    r = client.post("/api/autonomy/task-write",
                    json={"name": "Timestamp Probe", "frequency": "daily"})
    later = datetime.now(timezone.utc)
    assert r.status_code == 200, r.text

    new_id = r.json()["task"]["id"]
    fm = _front_matter(autonomy_dir / f"{new_id}-timestamp-probe.md")
    for key in ("created", "updated"):
        assert _drift_outside(fm[key], earlier, later) <= TOLERANCE_SECONDS, (
            f"{key}: {fm[key]!r} sits outside the write window "
            f"{earlier.isoformat()}..{later.isoformat()}")


def test_the_created_stamp_survives_a_later_edit(client, autonomy_dir):
    """`updated` moves and `created` does not. The update branch re-writes
    `created` from the file, so a create that recorded the true instant has to
    still denote it after a subsequent save."""
    r = client.post("/api/autonomy/task-write",
                    json={"name": "Timestamp Probe", "frequency": "daily"})
    new_id = r.json()["task"]["id"]
    path = autonomy_dir / f"{new_id}-timestamp-probe.md"
    created = _front_matter(path)["created"]

    client.post("/api/autonomy/task-write", json={"id": new_id, "priority": "high"})
    assert _front_matter(path)["created"] == created


# ── clause 4: the scan over both writers ─────────────────────────────────────

_CLOCK_READS = {
    "datetime.now", "datetime.datetime.now",
    "datetime.utcnow", "datetime.datetime.utcnow",
}


def _dotted(node: ast.AST) -> str:
    """`datetime.datetime.now` for an Attribute chain; `''` for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_clock_read(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _dotted(node.func) in _CLOCK_READS


def _is_naive_clock_read(node: ast.AST) -> bool:
    """A system-clock read that names no timezone.

    `datetime.now()` with no argument is the host's local clock. Any
    `datetime.utcnow()` counts too whatever its arguments: it returns a UTC
    *value* carrying no `tzinfo`, which is precisely why formatting it with a
    `Z` is fine while comparing it to an aware `now(timezone.utc)` is not."""
    if not _is_clock_read(node):
        return False
    if _dotted(node.func).endswith("utcnow"):
        return True
    return not node.args and not node.keywords


def _assigned_values(tree: ast.AST) -> dict[str, list[ast.AST]]:
    """`name -> every expression assigned to it anywhere in the file`.

    Lets a stamp applied to a variable be traced to whatever produced the value,
    which is the difference between pinning the one line that was wrong today and
    pinning the shape of the bug. Matching by name across the file is deliberate:
    it over-approximates, so a shadowed or reused name is judged by its worst
    assignment rather than escaping the scan."""
    out: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out.setdefault(node.target.id, []).append(node.value)
    return out


def _z_stamps(src: str) -> list[dict]:
    """Every `strftime` whose format string ends in `Z`, with its subject
    resolved: the clock read it came from, if a clock read produced it.

    This is the whole of clause 4. The defect was a `Z`-suffixed stamp whose
    subject was the host's naive local clock, so the assertion is about the
    subject of every such stamp in the file — never about a line number."""
    tree = ast.parse(src)
    assigned = _assigned_values(tree)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "strftime"):
            continue
        fmt = node.args[0] if node.args else None
        if not (isinstance(fmt, ast.Constant) and isinstance(fmt.value, str)
                and fmt.value.endswith("Z")):
            continue
        receiver = node.func.value
        subjects = [receiver]
        if isinstance(receiver, ast.Name):
            # Resolve the variable; if nothing in the file assigns it, it is a
            # parameter — a value the file was handed, not a clock it read.
            subjects = assigned.get(receiver.id, []) or [receiver]
        found.append({
            "lineno": node.lineno,
            "receiver": receiver,
            "subjects": subjects,
            "clock_sourced": any(_is_clock_read(s) for s in subjects),
            "naive": any(_is_naive_clock_read(s) for s in subjects),
            "passthrough": not any(_is_clock_read(s) for s in subjects),
        })
    return found


def test_the_router_contains_no_naive_system_clock_read():
    """Clause 4, first half. `datetime.now()` with no timezone, and any
    `datetime.utcnow()`, are both naive reads of the clock: formatted with a `Z`
    they state an instant the host does not live in, and formatted without one
    they state an instant nobody can place. Neither belongs in this module — it
    is the file that stamps the field the scheduler compares against UTC."""
    tree = ast.parse(ROUTER_SRC.read_text(encoding="utf-8"))
    naive = [(node.lineno, _dotted(node.func))
             for node in ast.walk(tree) if _is_naive_clock_read(node)]
    assert naive == [], f"naive clock reads in {ROUTER_SRC}: {naive}"


def test_no_z_suffixed_stamp_in_the_router_comes_from_a_naive_clock():
    """Clause 4, first half, stated as the defect itself: a `Z` stamp formatted
    from a naive clock read. Traced through locals, so `now = datetime.now()` on
    one line and `now.strftime('%...Z')` on the next is caught as one bug.

    The tolerated remainder is `_to_iso`'s re-serialisation of a front-matter
    datetime — a parameter, pinned at exactly two occurrences so a third has to
    be justified rather than quietly added."""
    stamps = _z_stamps(ROUTER_SRC.read_text(encoding="utf-8"))
    offenders = [s["lineno"] for s in stamps if s["naive"]]
    assert offenders == [], (
        f"Z-suffixed stamps formatted from a naive clock read in {ROUTER_SRC}: "
        f"lines {offenders}")
    passthrough = [s["lineno"] for s in stamps if s["passthrough"]]
    assert len(passthrough) == 2, (
        "expected exactly the two `_to_iso` passthroughs, which append `Z` to a "
        "datetime read from front matter; found lines "
        f"{passthrough} of {[s['lineno'] for s in stamps]}")


def test_the_mcp_writer_still_stamps_true_utc():
    """Clause 4, second half: `agent_mcp/autonomy.py` is the reference correct
    form — a clock read with `datetime.timezone.utc` passed, then formatted with
    a `Z` — and this pins that it stays that way. It fails if those writers lose
    their timezone argument, and equally if they are removed, because then no `Z`
    stamp in that module comes from a clock read and the reference form is gone."""
    stamps = _z_stamps(MCP_SRC.read_text(encoding="utf-8"))
    naive = [s["lineno"] for s in stamps if s["naive"]]
    assert naive == [], (
        "the MCP writer's Z stamps must stay true UTC — lines "
        f"{naive} now format a naive clock read")
    clock_sourced = [s["lineno"] for s in stamps if s["clock_sourced"]]
    assert len(clock_sourced) >= 2, (
        "agent_mcp/autonomy.py is the reference for a correct Z stamp and had two "
        f"clock-sourced writers; found {len(clock_sourced)} of {len(stamps)} "
        f"Z stamps clock-sourced (lines {[s['lineno'] for s in stamps]})")


def test_the_router_keeps_at_least_one_utc_sourced_z_stamp():
    """The module retains a second `Z` writer besides the front matter: the audit
    line a dispatch-stopping status change appends (#1127). Counted, not located,
    so it survives the lines moving — and fails if that stamp stops being a clock
    read at all, which would mean the audit trail lost its timestamp."""
    stamps = _z_stamps(ROUTER_SRC.read_text(encoding="utf-8"))
    clock_sourced = [s["lineno"] for s in stamps if s["clock_sourced"]]
    assert len(clock_sourced) >= 1, (
        "the status-change audit stamp is a clock-sourced Z stamp; it has either "
        f"gone or stopped being a clock read ({len(stamps)} Z stamps in all)")
    assert not any(s["naive"] for s in stamps)


# ── the scan's own falsifiability ────────────────────────────────────────────


def test_the_scan_finds_the_shape_it_claims_to_find():
    """Without this, the two assertions above could be passing because the
    matcher never matched anything — the failure mode this vault has a whole
    catalogue of. The expression that produced this item, fed to the same code,
    must be reported."""
    naive_local = ("def handler():\n"
                   "    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')\n"
                   "    return now\n")
    stamps = _z_stamps(naive_local)
    assert len(stamps) == 1, "the Z-strftime itself was not found"
    assert stamps[0]["naive"], "the naive local form slipped through the scan"

    utc_named = ("def handler():\n"
                 "    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')\n"
                 "    return now\n")
    stamps = _z_stamps(utc_named)
    assert len(stamps) == 1 and stamps[0]["clock_sourced"]
    assert not stamps[0]["naive"], "the correct form was reported as a defect"

    utcnow = ("def handler():\n"
              "    naive = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')\n"
              "    return naive\n")
    stamps = _z_stamps(utcnow)
    assert stamps[0]["naive"], "utcnow() slipped through the scan"

    module_level = ("import datetime\n"
                    "NOW = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')\n")
    stamps = _z_stamps(module_level)
    assert stamps[0]["naive"], "the qualified module form slipped through the scan"

    handed_in = ("def _to_iso(val):\n"
                 "    return val.strftime('%Y-%m-%dT%H:%M:%SZ')\n")
    stamps = _z_stamps(handed_in)
    assert stamps[0]["passthrough"], "a value read from front matter was " \
                                     "reported as a clock read"
