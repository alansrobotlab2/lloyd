"""A numbered task file with no `id:` resolves its id from its own filename (#3625).

`GET /api/autonomy/health` rendered `task_id: 'None'` because the builder is
`str(t.get("id"))` and the task dict had no `id` key at all: the file
`105-check-health-census-weekly.md` carried its number under `task_id:` instead of
`id:`, which is a different key — vault git shows `8c781006` (2026-10-31 01:12)
removing the stray `task_id: 105` and adding `id: 105`. The population is fixed;
what was missing is the guard, and the reason it is not cosmetic is everything
downstream of the missing key. `app/autonomy.py`'s write-back looks the file up by
its `NN-` name prefix (`_find_task_file`), so for `task_id=None` it finds nothing
and returns at `:177-179` WITHOUT raising: the status, `last_run` and
`failure_count` writes for that task were dropped for as long as the file was
mis-keyed, and a task whose writes are dropped never becomes `done`, never records
its own failures, and reports whatever the file last happened to say.

Two facts decide the fix shape. The filename prefix is already this codebase's id
authority — `_find_task_file` resolves a task by `f"{id}-"` and
`agent_mcp/autonomy.py::_next_task_id` derives a new id from the same prefix — and
the prefix is the ONE number that cannot disagree with the file lookup. So the
loader fills the missing key from the filename and warns by name: the row becomes
attributable AND the dropped write path is restored. A bare `"?"` default at the
render site would have renamed the symptom and left the silent write loss exactly
where it was, which is why `tests/test_autonomy_task_id_resolution.py::test_a_status_update_on_a_task_loaded_that_way_reaches_the_file`
drives the write-back and not just the string.

The warning is the other half: an id inferred by the loader is a repaired read, and
a repair nobody is told about is how a data-entry defect sits invisible for a month.
`scripts/autonomy/validate_tasks.py`, the board's own linter, drops an id-less
numbered file from its id map with no message at all (`:569`,
`if tid: parsed[tid] = fm`), so the load-time warning is the only place this file
shape is spoken of. That linter and the MCP listing's `task_id: 0` render are
deliberately not changed here — the owed-check ruling on widening covers them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from app import autonomy as A

# A file with the front matter every task file carries and the one key this item is
# about: NO `id:`. `status`/`frequency`/`name` are what `compute_health` and the
# scheduler read, so the fixture is a task in every other respect — a fixture missing
# those too would prove nothing about a real mis-keyed file reaching the fleet.
ORPHAN_FM = {
    "type": "autonomy",
    "segment": "autonomy",
    "name": "Orphan",
    "status": "up_next",
    "frequency": "daily",
    "priority": "medium",
}
BODY = "\n## Activity Log\n\n- 2026-10-30T00:00:00Z: run completed\n"


@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    """A real autonomy dir the module under test actually looks at.

    `AUTONOMY_DIR` is the one path `_find_task_file` and `_iter_task_files` glob, so
    patching it is what makes the write-back test below a disk test rather than a
    monkeypatched tautology.
    """
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    monkeypatch.setattr(A, "AUTONOMY_DIR", dirn)
    return dirn


def _write(dirn: Path, fname: str, fm: dict) -> Path:
    """A task file with `fm` as front matter, written verbatim so `id:` can be absent."""
    path = dirn / fname
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n{BODY}",
                    encoding="utf-8")
    return path


def _front_matter(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---\n", 2)[1])


def test_a_numbered_file_with_no_id_key_loads_with_the_filename_number(autonomy_dir):
    """Clause 2: `t["id"]` is the file's own number, not None.

    Asserted through both doors a consumer can come in by — `_parse_task_file` for a
    single path and `_iter_task_files`, which is the call `app/routers/autonomy.py`
    uses to build the fleet's `tasks` list. Only the second one shows the fix reaching
    the payload rather than living in one function.
    """
    path = _write(autonomy_dir, "999-orphan.md", ORPHAN_FM)
    assert "id" not in _front_matter(path), "the fixture must not declare an id"

    task = A._parse_task_file(path)
    assert task is not None, "a task-shaped file must still parse after the fix"
    assert task["id"] == 999, (
        f"the loader handed back id={task.get('id')!r}, so every consumer reading "
        "t['id'] still sees None and the write-back lookup still cannot resolve")

    listed = {t["id"]: t for t in A._iter_task_files()}
    assert 999 in listed, f"_iter_task_files lost the file: {sorted(listed)}"
    assert listed[999]["name"] == "Orphan"


def test_a_declared_id_is_never_overridden_and_never_warns(autonomy_dir, caplog):
    """The repair fills a missing key and touches nothing else.

    The prefix is a fallback, not an authority that outranks the file: a file that
    declares `id: 999` under a `999-` name is the healthy case and must be silent, and
    the same call with the declared id differing from the name is left alone too —
    reconciling a disagreement is a different defect with its own owner
    (`scripts/autonomy/validate_tasks.py`). Without this node the loader could be made
    to overwrite every id from the filename and the clause above would still pass.
    """
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    declared = _write(autonomy_dir, "999-declared.md", {**ORPHAN_FM, "id": 999})
    assert A._parse_task_file(declared)["id"] == 999
    other = _write(autonomy_dir, "998-declared.md", {**ORPHAN_FM, "id": 4242})
    assert A._parse_task_file(other)["id"] == 4242
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [], (
        "a file that declares its id must not be warned about: the warning is the "
        "signal that a key was inferred, and noise here burries the real offender")


def test_the_load_warns_once_naming_the_file_by_name(autonomy_dir, caplog):
    """Clause 3: the inferred id is announced, with the offending filename in it.

    Exactly one WARNING record, and its message carries `999-orphan.md`. Named, not
    coerced: a silent repair is what let the mis-keyed file sit unnoticed while its
    write-backs went nowhere, and a warning that does not say WHICH file cannot be
    acted on — the reader would have to grep the board to find the one line to fix.
    """
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    _write(autonomy_dir, "998-second-orphan.md", ORPHAN_FM)
    _write(autonomy_dir, "999-orphan.md", ORPHAN_FM)
    A._iter_task_files()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    offenders = [m for m in warnings if "orphan" in m]
    assert len(offenders) == 2, (
        f"expected one warning per id-less numbered file, got: {warnings}")
    assert any("999-orphan.md" in m for m in offenders), offenders
    assert all(re.search(r"\b99[89]\b", m) for m in offenders), (
        f"a warning naming neither the file nor the number is unactionable: {offenders}")


def test_a_non_numbered_file_is_left_alone(autonomy_dir, caplog):
    """Reports and notes keep no id, because for them there is no number to resolve.

    `_config.md`, `meta-analysis-2026-06-03.md` and `skill-lint-report.md` are the
    three real id-less files on the board, and `_iter_task_files` skips every name whose
    leading digits are not followed by a hyphen — the same guard `_find_task_file` uses,
    and the reason the loader repairs an id only for a file the lookup can resolve.
    Inventing an id for them, or
    warning nightly about files that are not tasks, would make the new signal noise —
    and a warning the fleet reads every night about a file that is not a task is a
    warning that stops being read.
    """
    caplog.set_level("WARNING", logger="lloyd-autonomy")
    note = _write(autonomy_dir, "meta-analysis-2026-06-03.md", ORPHAN_FM)
    A._parse_task_file(note)
    assert "id" not in (A._parse_task_file(note) or {}), (
        "a file the scheduler never enumerates must not gain an id")
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == []


def test_a_status_update_on_a_task_loaded_that_way_reaches_the_file(autonomy_dir):
    """Clause 4: the write-back path resolves, so the status lands on disk.

    This is the consequence the item did not know it was describing. `_update_task_field`
    locates the file through `_find_task_file`, which matches on the `NN-` prefix and
    returns None for `task_id=None` — and then RETURNS SILENTLY at `app/autonomy.py:177-179`.
    The first half below is the fixed behaviour: the id the loader handed over finds
    `999-orphan.md` and the new status is on disk. The second half is the control that
    makes this node able to fail: the same write addressed by a None id must still be
    dropped, so a passing node means the id is what changed, not that the write path
    never drops anything.
    """
    path = _write(autonomy_dir, "999-orphan.md", ORPHAN_FM)
    task = A._parse_task_file(path)

    A._update_task_field(task["id"], status="paused")
    assert _front_matter(path)["status"] == "paused", (
        "the status write was dropped again — the id handed to the lookup does not "
        f"resolve to the file: A._find_task_file({task['id']!r}) -> "
        f"{A._find_task_file(task['id'])!r}")
    assert A._find_task_file(task["id"]) == path

    A._update_task_field(None, status="failed")
    assert _front_matter(path)["status"] == "paused", (
        "a None id must address no file at all: if this write reached the file, the "
        "control is no longer a control and the assertion above proves nothing")
