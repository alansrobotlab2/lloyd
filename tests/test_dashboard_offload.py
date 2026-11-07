"""The dashboard's two GIL relievers (2026-11-05): a front-matter memo and
`board_health` in a child process. A py-spy profile put 62% of the backend's
samples in `_backlog` and 14% in `_autonomy` while `/health` stalled for up to
5 s."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from app.routers import dashboard as dash


def _item(d: Path, name: str, status: str) -> Path:
    p = d / f"{name}.md"
    fm = {"status": status, "priority": "medium", "board": "lloyd",
          "created": "2026-11-01T00:00:00+00:00", "tags": ["backlog"]}
    p.write_text(f"---\n{yaml.dump(fm)}---\n\n# {name}\n", encoding="utf-8")
    return p


def test_the_memo_rereads_a_changed_file_and_reuses_an_unchanged_one(tmp_path, monkeypatch):
    p = _item(tmp_path, "one", "draft")
    calls = []
    # The seam is `_frontmatter_from_block`: the parse the memoised
    # `_frontmatter` hands to `app.frontmatter.front_matter_of`. It was
    # `_frontmatter_parse` until #3860 moved the read and the fence rule into that
    # shared leaf. `_frontmatter_parse` still exists (the offload worker
    # re-exports it, and the loader guard's label cites it), but counting it here
    # would count a symbol the route no longer calls — the shape
    # test_dashboard_yaml_loader.py warns about: "an attribute that still has to
    # exist for the seam test to pass but is no longer what the reader calls".
    real = dash._frontmatter_from_block
    monkeypatch.setattr(dash, "_frontmatter_from_block",
                        lambda *a: calls.append(1) or real(*a))
    assert dash._frontmatter(p)["status"] == "draft"
    assert dash._frontmatter(p)["status"] == "draft"
    assert len(calls) == 1, "an unchanged file is not parsed twice"

    _item(tmp_path, "one", "up_next")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # a distinct mtime
    assert dash._frontmatter(p)["status"] == "up_next"
    assert len(calls) == 2


def test_a_caller_mutating_its_copy_does_not_change_the_next_read(tmp_path):
    p = _item(tmp_path, "two", "draft")
    dash._frontmatter(p)["status"] = "scribbled"
    assert dash._frontmatter(p)["status"] == "draft"


def test_a_missing_file_still_reads_as_no_front_matter(tmp_path):
    assert dash._frontmatter(tmp_path / "absent.md") == {}


def test_board_health_in_a_child_matches_the_in_process_call(tmp_path):
    from scripts.automod import backlog as B
    from app import board_health_offload as O

    d = tmp_path / "backlog"
    d.mkdir()
    for i, status in enumerate(("draft", "up_next", "done")):
        _item(d, f"{100 + i}-item", status)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("")

    child = O.run(ledger, d)
    here = json.loads(json.dumps(B.board_health(ledger, backlog_dir=d), default=str))
    assert child is not None
    assert set(child) == set(here)
    volatile = {"decisions"}            # carries the wall-clock window it was read over
    assert {k: v for k, v in child.items() if k not in volatile} == \
           {k: v for k, v in here.items() if k not in volatile}


def test_a_child_that_does_not_answer_in_time_reads_as_no_health(tmp_path):
    from app import board_health_offload as O
    d = tmp_path / "backlog"
    d.mkdir()
    assert O.run(tmp_path / "ledger.jsonl", d, timeout=0.01) is None


def test_off_production_data_the_dashboard_keeps_the_in_process_call(tmp_path, monkeypatch):
    import app.paths as paths
    from app import board_health_offload as O
    assert paths.IS_PRODUCTION_DATA is False          # conftest's data root
    monkeypatch.setattr(O, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    monkeypatch.setattr(dash, "_backlog_health_inproc", lambda d: {"inproc": True})
    assert dash._backlog_health(tmp_path) == {"inproc": True}
