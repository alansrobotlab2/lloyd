"""Shared fixtures for the #2620 substrate-fuzz tests. Not a test module.

Two things every one of the four files needs and neither should re-implement:

* `sf_task(...)` builds a synthetic corpus row and runs it through the SAME loader the
  shipped corpus goes through, so a test that wants a malformed task spends one line on
  it — the alternative is a test that stops checking refusals because building a task
  costs a paragraph, which is how a validator ends up unpinned.
* `sf_spill_dir` points the tool-result spill at a tmp directory. Arm A dispatches the
  real `Bash` tool, so a test run that did not redirect the spill would write into
  `SESSIONS_DIR` — the store `app/uptake.py` reads as the labeled corpus and the
  retention sweep ages (#1848). A test that adds rows to the corpus under test is worse
  than a test that fails.

Imported by `tests/test_substrate_fuzz_{corpus,runner,controls,safety}.py`, the same way
`tests/_live_data.py` is imported by the tests that need live-store reads.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.substrate_fuzz import corpus as sf_corpus  # noqa: E402
from eval.substrate_fuzz import runner as sf_runner  # noqa: E402

CORPUS = sf_corpus.load_corpus()
BY_ID = {t.id: t for t in CORPUS}
REAL = [t for t in CORPUS if not t.is_control]
CONTROLS = [t for t in CORPUS if t.is_control]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_s(s: str) -> str:
    return sha(s.encode())


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def sf_task(*, tid: str = "synthetic-task", domain: str = "byte_roundtrip",
            steps=None, post=None, **extra) -> dict:
    """A corpus row with the two required fields filled, ready to hand to `_parse_task`.

    Deliberately returns a dict and not a `Task`: the point is to change one field and
    watch the loader refuse it, and a `Task` cannot be built with a field missing.
    """
    row = {"id": tid, "domain": domain,
           "title": "synthetic task built by a test",
           "reason": "a test needed a shape and nothing more",
           "steps": steps if steps is not None else
           [{"tool": "Bash", "args": {"command": "printf x"}}],
           "postcondition": post if post is not None
           else {"kind": "exit_shape", "is_error": False}}
    row.update(extra)
    return row


def parse_task_row(row: dict):
    """Run `row` through the shipped per-task validator, cross-row checks included."""
    return sf_corpus._parse_task(row, seen_ids=set())


def write_corpus(tmp_path: Path, rows: list[dict], **top) -> Path:
    doc = {"version": 1, "item": 2620}
    doc.update(top)
    doc["tasks"] = rows
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


@pytest.fixture
def sf_spill_dir(tmp_path, monkeypatch):
    """Redirect the tool-result spill into `tmp_path`, and put it back after.

    Module-level for one reason: `tool_result_spill` reads `SESSIONS_DIR` through its own
    module binding at call time, so patching `app.paths.SESSIONS_DIR` changes nothing and
    a test that got it wrong would spill into the real store while looking green.
    """
    from app.harness import tool_result_spill as spill
    d = tmp_path / "spill"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(spill, "SESSIONS_DIR", d)
    assert spill.SESSIONS_DIR == d
    yield d

CONTROLLED_DOMAINS = sorted({c.domain for c in CONTROLS})

#: This module is a fixture, not a test: it loads the shipped corpus once so twenty-odd nodes
#: do not each pay for it, and the four test files import these names out of it. Declared, so
#: the re-exports are not read as an unused import by the gate's static rung.
__all__ = ["BY_ID", "CONTROLS", "CONTROLLED_DOMAINS", "CORPUS", "REAL", "b64",
           "parse_task_row", "sf_corpus", "sf_runner", "sf_spill_dir", "sf_task",
           "sha", "sha_s", "write_corpus"]
