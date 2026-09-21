"""A task file's declared `grants:` block must survive the scheduler's own
corrupt-YAML recovery path (#724 clause 3).

Why this is a security property and not a parsing nicety: after #724 promoted
`autonomy_write_task` to tier 2, a task that legitimately re-arms another task
carries a `grants:` block naming that tool — the block IS the human's
authorization. The scheduler reads task files through `_parse_task_file`, whose
third recovery tier regex-extracts the fields in `fallback_fields` when
`yaml.safe_load` fails. A field absent from that tuple is silently dropped by
the recovered copy, so a corrupt #40 file would come back with no `grants`,
`validate_task_grants(task.get("grants"))` would see nothing, the task would
run ungated, and its one legitimate re-arm would be denied mid-nightly — a
silent behavioural change caused by a file that failed to parse.

The second half of the hazard is shape: `yaml.dump(default_flow_style=False)`
writes a list of maps in block style, one map per line group. The single-line
regex fallback recovers scalars and inline sequences only, so adding `grants`
to `fallback_fields` is necessary but NOT sufficient on its own — the recovery
has to reach a nested block value. Both halves are pinned here.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import autonomy as A  # noqa: E402
from app.harness.policy import GrantStore, sync_task_grants  # noqa: E402
from app.harness.policy import validate_task_grants  # noqa: E402

# Frozen: `sync_task_grants` takes an explicit `now`, so the expiry below is
# live against it and the test never depends on the wall clock.
NOW = dt.datetime(2026, 9, 20, 12, 0, tzinfo=dt.timezone.utc)

VALID_GRANTS = [{
    "tool": "autonomy_write_task",
    "expires_at": "2099-01-01T00:00:00+00:00",
    "quota": 10,
    "issued_by": "alan",
}]


EXTRA_GRANT = [{
    "tool": "calendar_create",
    "expires_at": "2026-10-01T00:00:00Z",
}]


def _frontmatter(**extra) -> str:
    fm = {
        "id": 40, "name": "Nightly Reflection Config", "status": "up_next",
        "frequency": "daily", "priority": "high",
        "skill_name": "nightly-reflection-config",
    }
    fm.update(extra)
    # yaml.dump is the scheduler's own writer (`_update_task_field`), so the
    # shape under test is the shape a real task file has on disk.
    return yaml.dump(fm, default_flow_style=False, allow_unicode=True)


def _corrupt(fm: str) -> str:
    """Break the YAML the way an agent-written file breaks most often: an
    unquoted colon inside a value. Every OTHER line stays valid, which is what
    makes per-line recovery the right repair."""
    return fm.replace("name: Nightly Reflection Config",
                      "name: config: nightly: retry", 1)


@pytest.fixture
def task_file(tmp_path):
    def write(fm_text: str, body: str = "\n## Activity Log\n") -> Path:
        path = tmp_path / "40-nightly-reflection-config.md"
        path.write_text(f"---\n{fm_text}---\n{body}", encoding="utf-8")
        return path
    return write


def test_grants_survives_the_regex_recovery_path(task_file):
    """A corrupt file whose grants block is intact must come back WITH the
    block — not with the block silently absent."""
    path = task_file(_corrupt(_frontmatter(grants=VALID_GRANTS)))
    fm = A._parse_task_file(path)
    assert fm is not None, "a task may degrade but never vanish"
    assert fm.get("_yaml_broken") is True, "the file really did fail to parse"
    assert "grants" in fm, "a dropped grants block is a dropped authorization"
    assert fm["grants"] == VALID_GRANTS


def test_recovered_grants_still_validate(task_file):
    """Recovery has to produce something `validate_task_grants` accepts, or the
    dispatcher's fail-closed rule parks the task instead of gating it — a
    different silent outage wearing the same file."""
    path = task_file(_corrupt(_frontmatter(grants=VALID_GRANTS)))
    fm = A._parse_task_file(path)
    specs, errors = validate_task_grants(fm.get("grants"))
    assert errors == [], errors
    assert [s["tool"] for s in specs] == ["autonomy_write_task"]


def test_a_recovered_task_is_runnable_not_parked(task_file, monkeypatch):
    """The end-to-end half: the recovered file must still be dispatchable. A
    malformed block parks the task (`_dispatch_blockers`); an UNREADABLE one
    must not be mistaken for either."""
    dirn = task_file(_corrupt(_frontmatter(grants=VALID_GRANTS))).parent
    monkeypatch.setattr(A, "AUTONOMY_DIR", dirn)
    ids = [t["id"] for t in A._all_runnable_tasks()]
    assert ids == [40], "a parse failure cost the fleet a nightly job"


def test_grants_is_in_the_recovery_field_list():
    """Clause 3, stated where the field list lives now. `grants` must be in the
    tuple the scheduler hands to the parser, and that tuple is
    `agent_mcp._shared.AUTONOMY_TASK_FIELDS`: #1014 replaced this function's
    private 25-name tuple with the shared constant so the MCP reader and the
    board cannot recover a different field set from the scheduler's, and a
    field absent from that constant is simply not in the dict the fallback
    returns. Both halves are checked — the membership, and that
    `_parse_task_file` passes the shared constant rather than a list of its
    own — because the second is what makes the first the scheduler's answer."""
    from agent_mcp._shared import AUTONOMY_TASK_FIELDS

    assert "grants" in AUTONOMY_TASK_FIELDS, (
        "a degraded task file would recover with no authorization block")
    src = (ROOT / "autonomy.py").read_text()
    start = src.index("def _parse_task_file")
    body = src[start:src.index("def ", start + 10)]
    assert "fallback_fields=AUTONOMY_TASK_FIELDS" in body, (
        "the scheduler stopped passing the shared field list, so the membership "
        "above is no longer the list it recovers from")


def test_a_scalar_list_keeps_its_old_loose_behaviour():
    """Negative control for the new block recovery. The tuple already holds
    list-ish fields the loose regex recovers today with STRING items; the new
    path must return None for anything whose first item is a bare scalar, so
    those values keep their shape."""
    from agent_mcp._shared import _recover_block_field

    fm_text = ("id: 7\n"
               "name: bad: yaml: here\n"
               "expected_error_patterns:\n"
               "  - TimeoutError\n"
               "  - ConnectTimeout\n")
    assert _recover_block_field(fm_text, "expected_error_patterns") is None
    assert _recover_block_field(fm_text, "grants") is None, (
        "a key that is not in the text must not be invented")


def test_a_nested_block_is_recovered_with_every_key():
    """Why the loose scan could not do this job: it needs text after the colon
    and reads one line, so a block-style `grants:` was invisible to it and the
    key vanished. The whole point of the block is the keys on the FOLLOWING
    lines — `expires_at` above all, since a grant with no expiry is the defect
    the whole grant design exists to prevent."""
    from agent_mcp._shared import _recover_block_field

    fm_text = ("name: bad: yaml: here\n"
               "grants:\n"
               "  - tool: autonomy_write_task\n"
               "    expires_at: '2026-10-01T00:00:00Z'\n"
               "    arg:\n"
               "      status: up_next\n"
               "status: up_next\n")
    got = _recover_block_field(fm_text, "grants")
    assert got is not None and len(got) == 1
    assert got[0]["expires_at"] == "2026-10-01T00:00:00Z"
    assert got[0]["arg"] == {"status": "up_next"}
    assert "status" not in got[0], "the block ran past the next top-level key"


def test_a_block_that_parses_to_garbage_is_not_recovered():
    """Recovery may restore a SHAPE, never an authorisation. A block whose items
    are not mappings returns None, so the caller keeps its existing behaviour
    and the validator still sees something it must reject."""
    from agent_mcp._shared import _recover_block_field

    assert _recover_block_field("grants:\nemail_send please\n", "grants") is None


def test_recovered_grants_are_materialisable(tmp_path, task_file):
    """End of the chain the clause is about: the recovered block is exactly what
    `sync_task_grants` writes at dispatch, so it must survive intact — a block
    that came back as tool names only would raise here, and dispatch fails
    closed on that, parking the task instead of gating it."""
    path = task_file(_corrupt(_frontmatter(grants=VALID_GRANTS)))
    task = A._parse_task_file(path)
    store = GrantStore(tmp_path / "grants-test.db")
    store.ensure_schema()
    added = sync_task_grants(store, task_id=40, scope="autonomy-task:40",
                             grants=task["grants"], now=NOW)
    assert added == 1
    live = store.live(scope="autonomy-task:40", now=NOW)
    assert [r["tool_pattern"] for r in live] == ["autonomy_write_task"]
    assert live[0]["expires_at"], "a recovered grant kept its expiry"
