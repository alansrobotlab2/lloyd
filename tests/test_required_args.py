"""An absent required argument is refused, never silently defaulted to empty.

The failure these tests exist for is a *successful* destructive call. At base
`_edit` read `old_string`/`new_string` with `args.get(k, "")`, so a call that
omitted `new_string` — for instance one that sent `content`, the `Write` key,
by mistake — deleted its `old_string` and returned
`'Edited <path> (1 replacement)'`. `_write` read `content` the same way, so a
`Write` with no `content` truncated a 280-byte file to 0 bytes and returned
`'File written: <path> (0 chars)'`. Both tools' `inputSchema` declares the
argument required (`agent_mcp/builtin_fs.py`: `required: ["file_path",
"old_string", "new_string"]` for `Edit`, `["file_path", "content"]` for
`Write`), and nothing on the dispatch path validates arguments, so the handler
is the last place that can refuse.

Every test below Reads its target first where the lane would have to, so the
refusal it exercises is the argument one and not the read-before-write gate
firing by accident: at base these calls got past the gate and destroyed bytes
while reporting success.

The guard is key-presence (`k not in args`), never truthiness. `new_string: ""`
is the documented delete path — "pass an empty string to delete the matched
text" — and `content: ""` is the legitimate empty-file write, so
`if not args.get(k)` would refuse both of the calls that are supposed to work.
Key presence is also what keeps microcompaction compatible: it spills a big
body by replacing the *value* with `SHRUNK_ARG_MARKER` and leaves the key in
place, so a compacted call still satisfies the guard.

Each test goes through `FS.call_tool`, the `async def call_tool` seam at the
bottom of `agent_mcp/builtin_fs.py` that the MCP aggregator calls and that
picks the handler by tool name — which is why the AST graph reports no caller
for `_edit`/`_write` at all, and why the refusal can only be pinned from this
side of the seam.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from agent_mcp import builtin_fs as FS
from app.harness.microcompact import SHRUNK_ARG_MARKER, shrink_assistant_arguments

ROOT = Path(__file__).resolve().parents[1]

SID = "20261019_120000_reqargs"

ORIGINAL = "alpha\nKEEP ME\nomega\n"


@pytest.fixture(autouse=True)
def _clean():
    FS.reset_read_records()
    yield
    FS.reset_read_records()


@pytest.fixture
def bound(monkeypatch):
    """Bind a session id, as the aggregator's call path does for a real call."""
    monkeypatch.setattr(FS, "get_bound_session", lambda: SID)
    return SID


@pytest.fixture
def spill(monkeypatch, tmp_path):
    """Capture what microcompaction persists instead of filling a session dir."""
    saved: dict[str, str] = {}

    def _persist(text, *, tool_use_id, session_id):
        p = tmp_path / f"{tool_use_id}.txt"
        p.write_text(text)
        saved[tool_use_id] = text
        return str(p)

    monkeypatch.setattr("app.harness.microcompact.persist_for_compaction", _persist)
    return saved


def _text(res) -> str:
    return res.content[0].text


def _payload(res) -> dict:
    return json.loads(_text(res))


def _err(res) -> str:
    return _payload(res).get("error", "")


def _make(tmp_path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


async def _read(p: Path):
    """Satisfy the read-before-write gate, as the lane always does."""
    return await FS.call_tool("Read", {"file_path": str(p)})


# ── clause 1: Edit without new_string ───────────────────────────────────────

async def test_edit_omitting_new_string_names_it_and_writes_nothing(bound, tmp_path):
    """The item's measured case: file Read, old_string present, new_string absent."""
    p = _make(tmp_path, "c1.txt", ORIGINAL)
    await _read(p)

    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "KEEP ME\n"})

    assert _err(res) == "new_string is required"
    assert res.is_error
    assert p.read_text() == ORIGINAL, "the refusal must write no bytes"


async def test_edit_with_the_write_key_and_no_new_string_is_refused(bound, tmp_path):
    """A wrong-tool key is what made the absent argument destructive at base.

    `content` is `Write`'s key and `_edit` ignores it, so the handler saw
    `{file_path, old_string}`, deleted the matched text and reported
    `'Edited ... (1 replacement)'`.
    """
    p = _make(tmp_path, "c2.txt", ORIGINAL)
    await _read(p)

    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "content": "unused",
                                      "old_string": "KEEP ME\n"})

    assert _err(res) == "new_string is required"
    assert p.read_text() == ORIGINAL


# ── clause 2: Edit without old_string ───────────────────────────────────────

async def test_edit_omitting_old_string_names_it_and_writes_nothing(bound, tmp_path):
    p = _make(tmp_path, "c3.txt", ORIGINAL)
    await _read(p)

    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "new_string": "REPLACED\n"})

    assert _err(res) == "old_string is required"
    assert res.is_error
    assert p.read_text() == ORIGINAL


async def test_edit_omitting_old_string_with_replace_all_does_not_interleave(bound, tmp_path):
    """The path `original.replace("", new)` reaches only with `replace_all`.

    `original.count("")` is 7 on this 6-character file, so at base the
    multi-match check refused the call by accident; `replace_all: true` walked
    past that accident into `original.replace("", "-")` and interleaved
    `new_string` between every character.
    """
    p = _make(tmp_path, "six.txt", "abcdef")
    await _read(p)

    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "new_string": "-",
                                      "replace_all": True})

    assert _err(res) == "old_string is required"
    assert p.read_text() == "abcdef", "no new_string may be interleaved"


async def test_edit_omitting_both_strings_names_a_key_not_identical(bound, tmp_path):
    """Base answered this with `old_string and new_string are identical`."""
    p = _make(tmp_path, "c4.txt", ORIGINAL)
    await _read(p)

    res = await FS.call_tool("Edit", {"file_path": str(p)})

    assert _err(res) == "old_string is required"
    assert p.read_text() == ORIGINAL


# ── clause 3: Write without content ─────────────────────────────────────────

async def test_write_omitting_content_does_not_truncate_the_target(bound, tmp_path):
    body = "line\n" * 56  # 280 bytes, the size the item measured
    p = _make(tmp_path, "big.txt", body)
    await _read(p)

    res = await FS.call_tool("Write", {"file_path": str(p)})

    assert _err(res) == "content is required"
    assert res.is_error
    assert p.read_bytes() == body.encode(), "an existing target may not be truncated"
    assert p.stat().st_size == 280


async def test_write_omitting_content_does_not_create_the_target(bound, tmp_path):
    p = tmp_path / "never.txt"

    res = await FS.call_tool("Write", {"file_path": str(p)})

    assert _err(res) == "content is required"
    assert not p.exists(), "a refusal may not create the file either"


# ── clause 4: present-but-empty is accepted ─────────────────────────────────

async def test_edit_with_empty_new_string_still_deletes_the_match(bound, tmp_path):
    """`new_string: ""` is the documented delete path, so it must still work."""
    p = _make(tmp_path, "c5.txt", ORIGINAL)
    await _read(p)

    res = await FS.call_tool("Edit", {"file_path": str(p),
                                      "old_string": "KEEP ME\n",
                                      "new_string": ""})

    assert not res.is_error, _err(res)
    assert p.read_text() == "alpha\nomega\n"


async def test_write_with_empty_content_still_writes_an_empty_file(bound, tmp_path):
    p = tmp_path / "empty.txt"

    res = await FS.call_tool("Write", {"file_path": str(p), "content": ""})

    assert not res.is_error, _err(res)
    assert p.exists() and p.read_bytes() == b""


async def test_a_shrunk_edit_call_keeps_both_keys_and_is_not_refused_as_absent(bound, tmp_path, spill):
    """Microcompaction replaces the value and keeps the key, so the guard holds.

    The refusal this call must NOT get is the presence refusal; it is refused
    later, as `old_string not found`, exactly as it was before the guard.
    """
    p = _make(tmp_path, "shrunk.txt", "alpha\nomega\n")
    await _read(p)
    big = "O" * 20_000
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c0", "type": "function",
                "function": {"name": "Edit", "arguments": json.dumps({
                    "file_path": str(p),
                    "old_string": big,
                    "new_string": "N" * 20_000})}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "ok"}]
    out, shrunk, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id=SID)

    assert shrunk == 1
    args = json.loads(out[1]["tool_calls"][0]["function"]["arguments"])
    assert args["old_string"].startswith(SHRUNK_ARG_MARKER)

    res = await FS.call_tool("Edit", args)
    assert "is required" not in _err(res)
    assert "old_string not found" in _err(res)


# ── clause 5: existing error shape, no schema layer ─────────────────────────

async def test_every_refusal_uses_the_is_required_error_shape(bound, tmp_path):
    """`{"error": "<key> is required"}` — one key, the handler's own wording."""
    missing = re.compile(r"^[a-z_]+ is required$")
    p = _make(tmp_path, "c6.txt", ORIGINAL)
    await _read(p)
    cases = [
        ("Edit", {"file_path": str(p), "old_string": "KEEP ME\n"},
         "new_string is required"),
        ("Write", {"file_path": str(tmp_path / "another.txt")},
         "content is required"),
    ]
    for name, args, expected in cases:
        res = await FS.call_tool(name, args)
        payload = _payload(res)
        assert list(payload) == ["error"], f"{name}: {payload}"
        assert payload["error"] == expected
        assert missing.match(payload["error"])
        assert expected.split(" ")[0] in ("old_string", "new_string", "content")


async def test_no_schema_validation_layer_appears_on_the_dispatch_path():
    """`jsonschema` is imported by `app/harness/run_state.py` and nothing else.

    Switching global argument validation on was ruled NO on 2026-10-19 (the
    #1125 owed entry): it would reject the harness-injected `summary` and
    `_session_id` unless their stripping moved ahead of it, and would newly
    reject the proxied Thunderbird schemas. This per-handler presence check
    does not reopen that question, and this test fails if it ever does.
    """
    # The pattern is an anchored import statement, not the bare word and not a
    # plain string. Grepping the word matched this file's own prose and the
    # assertion refuted itself at the gate — the tree returned
    # ['app/harness/run_state.py', 'tests/test_required_args.py'] — and a plain
    # "import jsonschema" pattern refutes itself the same way, because the
    # pattern is then sitting in this file's argv list. -E with the anchor
    # matches a real import and matches neither spelling of the pattern here.
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-lE",
         r"^[ \t]*(import|from) +jsonschema", "--", "*.py"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    assert sorted(proc.stdout.split()) == ["app/harness/run_state.py"]

    source = (ROOT / "agent_mcp" / "builtin_fs.py").read_text()
    assert "jsonschema" not in source
    assert "validate" not in source.split("async def call_tool")[1]
