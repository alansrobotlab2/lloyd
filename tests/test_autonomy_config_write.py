"""#1326 — the scheduler-config write no longer travels under a read-only name.

`autonomy_config` sat in `READ_ONLY` while its `key`+`value` form called
`_write_config`, which whole-file re-dumped `~/obsidian/autonomy/_config.md`
from the parsed front matter alone — every byte below the closing `---` was
discarded on every set. Being read-only also meant the four refusals that key
off that table never fired: plan mode (`plan_mode_blocked_tools`), a bench or
eval session (`_tool_sandbox.refusal`), a sessionless `call_tool` (#1053), and
`MCPPool._retry_safe`, which re-sends a read-only call after a transport drop.

The fix splits the write into `autonomy_config_set`. These tests pin the two
halves of that split that a caller can observe: the writer keeps the body it
used to drop, and the read can no longer write.

Run: .venvs/lloyd/bin/python -m pytest tests/test_autonomy_config_write.py
"""
from __future__ import annotations

import asyncio
import json

import pytest
import yaml

import agent_mcp.autonomy as autonomy

# Copied byte-for-byte from the live `~/obsidian/autonomy/_config.md` on
# 2026-09-21 (14 lines, 219 bytes). The body below the closing `---` is the
# point: it repeats the key `segment`, and it carries a stray `---` line, so a
# writer that re-splits on the fence rather than keeping the tail would
# restructure it silently.
HEAD = (
    "---\n"
    "segment: autonomy\n"
    "autonomy_watermark: '{\"task_id\":\"24\",\"updated_at\":\"2025-07-15T03:00:00Z\"}'\n"
    "tags:\n- autonomy\n- config\n"
    "task_id: '25'\n"
    "type: autonomy-config\n"
    "updated: '2026-04-02T00:00:00Z'\n"
)
BODY = "segment: autonomy\n\n---\n\n"
#: The closing fence plus the body — the bytes a set must not touch.
TAIL = "---\n" + BODY
LIVE_CONFIG = HEAD + TAIL


def _is_error(result) -> bool:
    """Across the mcp 1.x/2.x field split `main._result_is_error` documents."""
    return bool(getattr(result, "isError", False) or getattr(result, "is_error", False))


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    """The live-shaped `_config.md`, in a directory the module is redirected to."""
    path = tmp_path / "_config.md"
    path.write_text(LIVE_CONFIG, encoding="utf-8")
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path)
    return path


def _front_matter(raw: str) -> dict:
    """Read the front matter the way the *reader* of the file must: the region up
    to the untouchable body, opening fence to closing fence."""
    assert raw.endswith(BODY), f"the body below the closing fence changed: {raw[-60:]!r}"
    head = raw[:-len(BODY)]
    assert head.startswith("---\n") and head.endswith("---\n"), head[:40]
    return yaml.safe_load(head[len("---\n"):-len("---\n")])


# ── clause 1: the writer writes, and keeps every byte of the body ────────────

def test_config_set_writes_the_key_and_keeps_the_body_byte_identical(cfg_file):
    before = cfg_file.read_text(encoding="utf-8")
    out = json.loads(autonomy._handle_config_set(
        {"key": "max_parallel", "value": "3"}))
    assert out.get("set") == "max_parallel", out
    after = cfg_file.read_text(encoding="utf-8")
    assert after.endswith(TAIL), (
        "the bytes below the closing `---` were rewritten: "
        f"before ended {before[-60:]!r}, after ends {after[-60:]!r}")
    fm = _front_matter(after)
    assert fm["max_parallel"] == "3", fm
    # The keys that were already there are still there — a set adds, it does
    # not replace the mapping.
    for key in ("segment", "task_id", "type", "autonomy_watermark"):
        assert key in fm, f"{key} disappeared from the front matter: {fm}"
    assert autonomy._read_config() == fm, (
        "`_read_config` no longer agrees with what is on disk")


def test_config_set_survives_a_second_set(cfg_file):
    autonomy._handle_config_set({"key": "a", "value": "1"})
    autonomy._handle_config_set({"key": "b", "value": "2"})
    after = cfg_file.read_text(encoding="utf-8")
    assert after.endswith(TAIL), after[-60:]
    fm = _front_matter(after)
    assert (fm["a"], fm["b"]) == ("1", "2"), fm


def test_config_set_keeps_a_prose_body_verbatim(cfg_file, tmp_path):
    """Any prose, not just the cruft the live file happens to carry."""
    prose = ("Somebody wrote the reasoning for the watermark here.\n"
             "It spans several lines, quotes a task id `#80`, and a line of its "
             "own is `---`, which is what defeated the old whole-file re-dump.\n")
    cfg_file.write_text(HEAD + "---\n" + prose, encoding="utf-8")
    autonomy._handle_config_set({"key": "paused", "value": "true"})
    after = cfg_file.read_text(encoding="utf-8")
    assert after.endswith("---\n" + prose), repr(after[-len(prose) - 20:])
    assert yaml.safe_load(after[:-len("---\n" + prose)])["paused"] == "true"


def test_a_front_matter_that_does_not_parse_is_refused_not_overwritten(cfg_file):
    """The module's own rule for a rewrite it cannot do safely (#1014): a
    fallback-parsed mapping re-emitted is the clobber, not the recovery."""
    cfg_file.write_text("---\n: this is not yaml [\n---\n" + BODY, encoding="utf-8")
    before = cfg_file.read_bytes()
    out = json.loads(autonomy._handle_config_set({"key": "a", "value": "1"}))
    assert "error" in out, out
    assert cfg_file.read_bytes() == before, "a broken front matter was rewritten anyway"


def test_a_file_with_no_closing_fence_is_refused_not_rewritten(cfg_file):
    """No closing fence means no place to put the prose, so there is nothing
    the writer can promise; it refuses rather than guessing."""
    cfg_file.write_text("---\nsegment: autonomy\n", encoding="utf-8")
    before = cfg_file.read_bytes()
    out = json.loads(autonomy._handle_config_set({"key": "a", "value": "1"}))
    assert "error" in out, out
    assert cfg_file.read_bytes() == before


# ── clause 2: the read half can no longer write ──────────────────────────────

def test_autonomy_config_with_key_and_value_writes_nothing_and_names_the_writer(
        cfg_file):
    """The whole point of the split: the shape that used to write now refuses,
    and says where the write went."""
    before = cfg_file.read_bytes()
    out = json.loads(autonomy._handle_config(
        {"key": "max_parallel", "value": "3"}))
    assert cfg_file.read_bytes() == before, "autonomy_config still writes"
    assert "autonomy_config_set" in json.dumps(out), out
    assert "error" in out, "a refused write must read as an error to the caller"


def test_autonomy_config_reads_the_whole_config_and_one_key(cfg_file):
    whole = json.loads(autonomy._handle_config({}))
    assert whole["segment"] == "autonomy", whole
    one = json.loads(autonomy._handle_config({"key": "task_id"}))
    assert one == {"task_id": "25"}, one
    missing = json.loads(autonomy._handle_config({"key": "nope"}))
    assert "error" in missing, missing
    assert cfg_file.read_bytes() == LIVE_CONFIG.encode(), "a read changed the file"


def test_the_read_half_dispatch_through_the_mcp_module(cfg_file):
    """The seam a caller crosses: `agent_mcp.autonomy.call_tool`, which is what
    `agent_mcp.main` routes an MCP tool call into."""
    read = asyncio.run(autonomy.call_tool("autonomy_config", {}))
    assert json.loads(read.content[0].text)["segment"] == "autonomy"
    refused = asyncio.run(autonomy.call_tool(
        "autonomy_config", {"key": "max_parallel", "value": "3"}))
    assert _is_error(refused), "the read half dispatched a write"
    assert "autonomy_config_set" in refused.content[0].text
    before = cfg_file.read_bytes()
    written = asyncio.run(autonomy.call_tool(
        "autonomy_config_set", {"key": "max_parallel", "value": "3"}))
    assert not _is_error(written), written.content[0].text
    assert cfg_file.read_text(encoding="utf-8").endswith(TAIL)
    assert autonomy._read_config()["max_parallel"] == "3"
    assert before != cfg_file.read_bytes()


def test_the_surface_advertises_the_writer_separately():
    """`tools/list` is what a model plans from: the reader must stop offering a
    `value`, and the writer must exist with both parameters required."""
    tools = {t.name: t for t in asyncio.run(autonomy.list_tools())}
    assert "autonomy_config_set" in tools, sorted(tools)
    assert "value" not in (tools["autonomy_config"].input_schema.get("properties") or {})
    assert "key" in (tools["autonomy_config"].input_schema.get("properties") or {})
    set_props = tools["autonomy_config_set"].input_schema.get("properties") or {}
    assert set(set_props) >= {"key", "value"}, set_props
    assert sorted(tools["autonomy_config_set"].input_schema.get("required") or []) == [
        "key", "value"]
    for name in ("autonomy_config", "autonomy_config_set"):
        assert len(tools[name].description or "") >= 60, name
        for pname, spec in (tools[name].input_schema.get("properties") or {}).items():
            assert (spec.get("description") or "").strip(), f"{name}.{pname}"
