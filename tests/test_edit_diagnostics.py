"""Diagnostics on the edit result, not on the gate three minutes later.

The delta rules are the interesting part. This tree carries ~69 tolerated
pyflakes findings; an absolute report would be noise on every edit, and the
model would learn to skip the block — which is worse than not having one.
"""

from __future__ import annotations

import json

import pytest

from agent_mcp import _edit_diagnostics as D, builtin_fs as FS

SID = "20260908_130000_diag"

CLEAN = "import os\n\n\ndef f():\n    return os.getcwd()\n"
BROKEN = "import os\n\n\ndef f():\n    return bar\n"


@pytest.fixture(autouse=True)
def _clean():
    FS.reset_read_records()
    yield
    FS.reset_read_records()


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)


def _text(res):
    return res.content[0].text


# ── the block itself ────────────────────────────────────────────────────────

def test_a_new_finding_is_reported_with_post_image_positions():
    block = D.python_block("/x/a.py", CLEAN.encode(), BROKEN)
    # Two, and both are real: dropping the only use of `os` makes the import
    # unused, which is exactly the second-order breakage this exists to show.
    assert 'tool="pyflakes" new="2"' in block
    assert "5:12: undefined name 'bar'" in block
    assert "1:1: 'os' imported but unused" in block
    assert block.startswith('<diagnostics file="/x/a.py"')


def test_a_clean_edit_reports_nothing():
    assert D.python_block("/x/a.py", CLEAN.encode(),
                          CLEAN.replace("getcwd", "getpid")) == ""


def test_a_pre_existing_finding_is_not_reported():
    """~69 of these exist in the tree; reporting them is how a block dies."""
    src = "import os\n\n\ndef f():\n    return 1\n"        # unused import
    assert D.python_block("/x/a.py", src.encode(), src + "\n") == ""


def test_a_finding_that_only_moved_is_not_new():
    """Position is not identity — otherwise inserting a line reports the file."""
    src = "import os\n\n\ndef f():\n    return 1\n"
    moved = "# a new first line\n" + src
    assert D.python_block("/x/a.py", src.encode(), moved) == ""


def test_a_second_copy_of_an_existing_finding_is_new():
    """A multiset, not a set: the duplicate is a real finding."""
    one = "def f():\n    return bar\n"
    two = "def f():\n    return bar\n\n\ndef g():\n    return bar\n"
    block = D.python_block("/x/a.py", one.encode(), two)
    assert 'new="1"' in block


def test_a_created_file_reports_everything():
    block = D.python_block("/x/a.py", None, "import os\n")
    assert 'new="1"' in block and "'os' imported but unused" in block


def test_a_syntax_error_is_always_reported_with_the_source_line():
    block = D.python_block("/x/a.py", CLEAN.encode(), "def f(:\n")
    assert 'syntax_error="true"' in block
    assert "pre_existing" not in block
    assert "def f(:" in block


def test_a_pre_existing_syntax_error_is_tagged_not_hidden():
    block = D.python_block("/x/a.py", b"def f(:\n", "def g(:\n")
    assert 'syntax_error="true"' in block
    assert 'pre_existing="true"' in block


def test_fixing_a_syntax_error_reports_no_flakes():
    """pyflakes never ran on the pre-image, so there is no baseline at all.

    Without this rule an edit that *fixed* the syntax would dump every
    tolerated finding in the file onto the model as if it had caused them.
    """
    assert D.python_block("/x/a.py", b"def f(:\n", "import os\nimport sys\n") == ""


def test_output_is_bounded():
    src = "".join(f"def f{i}():\n    return undefined_{i}\n" for i in range(50))
    block = D.python_block("/x/a.py", b"", src, max_lines=5)
    assert 'new="50"' in block
    assert "... and 45 more" in block
    assert len(block.splitlines()) == 8      # open tag + 5 + more + close


def test_only_dot_py_is_checked():
    assert D.python_block("/x/a.pyi", None, "import os\n") == ""
    assert D.python_block("/x/a.txt", None, "import os\n") == ""


def test_a_huge_file_is_skipped():
    big = "x = 1\n" * 400_000
    assert len(big) > D.MAX_SOURCE_BYTES
    assert D.python_block("/x/a.py", None, big) == ""


def test_nothing_raises_out_of_the_block(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("linter exploded")
    monkeypatch.setattr(D, "_python_block", boom)
    assert D.python_block("/x/a.py", None, "x = 1\n") == ""


# ── wired into the tool result ──────────────────────────────────────────────

async def test_an_edit_that_breaks_something_carries_the_block(bound, tmp_path):
    p = tmp_path / "m.py"
    p.write_text(CLEAN)
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "os.getcwd()",
                                      "new_string": "bar"})
    text = _text(res)
    assert res.is_error is False, text
    assert text.startswith(f"Edited {p} (1 replacement)")
    assert "<diagnostics" in text and "undefined name 'bar'" in text


async def test_a_clean_edit_carries_nothing(bound, tmp_path):
    p = tmp_path / "m.py"
    p.write_text(CLEAN)
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "getcwd", "new_string": "getpid"})
    assert "<diagnostics" not in _text(res)


async def test_a_write_carries_the_block_too(bound, tmp_path):
    p = tmp_path / "new.py"
    res = await FS.call_tool("Write", {"file_path": str(p), "content": BROKEN})
    assert "<diagnostics" in _text(res)


async def test_a_refused_edit_stays_a_clean_json_error(bound, tmp_path):
    """`text_result` sniffs a leading JSON object for `isError`."""
    p = tmp_path / "m.py"
    p.write_text(BROKEN)
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "bar",
                                      "new_string": "baz"})
    assert res.is_error
    json.loads(_text(res))            # must still parse


async def test_a_failed_edit_carries_no_block(bound, tmp_path):
    p = tmp_path / "m.py"
    p.write_text(CLEAN)
    await FS.call_tool("Read", {"file_path": str(p)})
    res = await FS.call_tool("Edit", {"file_path": str(p), "old_string": "nope",
                                      "new_string": "x"})
    assert res.is_error
    assert "<diagnostics" not in _text(res)
    json.loads(_text(res))


async def test_the_kill_switch_removes_the_block(bound, tmp_path, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "edit_diagnostics": {"python": False}})
    p = tmp_path / "m.py"
    res = await FS.call_tool("Write", {"file_path": str(p), "content": BROKEN})
    assert "<diagnostics" not in _text(res)


async def test_max_lines_comes_from_config(bound, tmp_path, monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "edit_diagnostics": {"python": True, "max_lines": 2}})
    p = tmp_path / "m.py"
    src = "".join(f"def f{i}():\n    return undefined_{i}\n" for i in range(10))
    res = await FS.call_tool("Write", {"file_path": str(p), "content": src})
    assert "... and 8 more" in _text(res)


async def test_a_linter_failure_never_fails_the_edit(bound, tmp_path, monkeypatch):
    p = tmp_path / "m.py"
    monkeypatch.setattr(D, "config", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    res = await FS.call_tool("Write", {"file_path": str(p), "content": "x = 1\n"})
    assert res.is_error is False
    assert p.read_text() == "x = 1\n"
