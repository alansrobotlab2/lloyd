"""`vault_write` has an append route, and an overwrite names what it dropped (#4453).

The 2026-11-21 incident (`~/lloyd-data/sessions/20261120_214215_autonomy_c244.json`,
messages 153-154): a data-pipeline run appending its post-flight section to the day
run-log `memory/vault-maintenance/2026-11-21.md` could only do so by re-emitting the
whole file; the emission came back half as long, `vault_write` wrote exactly those
9,042 bytes and answered `{"success": true}`, and 170 lines of another run's section
were gone until the skill's own `git diff --numstat` noticed.

So, against a scratch vault (never `~/obsidian`):

* clause 1 — `mode: "append"` adds `content` after the existing bytes, and creates the
  file when it is absent;
* clause 2 — the pre-image survives as a byte-identical prefix and the size on disk is
  exactly `len(pre) + len(content)`;
* clause 3 — the loaded-memory ceiling is priced on pre-image + fragment, so an append
  that would push `USER.md` past its ceiling is refused even though the fragment alone
  is tiny;
* clause 4 — an overwrite's success result carries `removed_lines`, the integer count
  of pre-image lines absent from the new content (0 when nothing was dropped).

Deliberately not asked for and not pinned: a refusal on an overwrite that drops lines.
"""
from __future__ import annotations

import json
import types

import pytest

from agent_mcp import _change_ledger as CL
from agent_mcp import builtin_fs as FS
from app import memory_ceiling as MC

RUN_LOG = "memory/vault-maintenance/2026-11-21.md"
PRE_IMAGE = (
    "# Vault maintenance — 2026-11-21\n"
    "\n"
    "## Run 1 (another job's section)\n"
    "- step 1 ok\n"
    "- step 2 ok\n"
)
SECTION = "\n## Run 2\n- post-flight ok\n"


@pytest.fixture(autouse=True)
def _clean():
    FS.reset_read_records()
    CL.reset()
    yield
    FS.reset_read_records()
    CL.reset()


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    import agent_mcp.vault as V

    monkeypatch.setattr(V, "VAULT", tmp_path)
    monkeypatch.setattr(V, "AUDIT_LOG_DIR", tmp_path / "audit")
    monkeypatch.setattr(V, "AUDIT_LOG_FILE", tmp_path / "audit" / "writes.jsonl")
    monkeypatch.setattr(V, "get_bound_session", lambda: "")
    # A loaded-memory write POSTs a prewarm hint to the live backend; never from a test.
    monkeypatch.setattr(V, "_notify_prompt_surface_write", lambda path: None)
    monkeypatch.setattr(CL, "CHANGES_ROOT", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    memories = tmp_path / "lloyd"
    memories.mkdir()
    monkeypatch.setattr(MC, "MEMORIES_DIR", memories)
    return types.SimpleNamespace(root=tmp_path, mod=V)


def _write(scratch, rel: str, content: str, **extra) -> dict:
    return scratch.mod._vault_write({"path": rel, "content": content, **extra})


def _seed(scratch, rel: str, text: str) -> None:
    p = scratch.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ── clause 1 ─────────────────────────────────────────────────────────────────

def test_append_adds_content_after_the_existing_bytes(scratch):
    _seed(scratch, RUN_LOG, PRE_IMAGE)
    res = _write(scratch, RUN_LOG, SECTION, mode="append")
    assert res.get("success") is True, res
    assert (scratch.root / RUN_LOG).read_text(encoding="utf-8") == PRE_IMAGE + SECTION
    assert res["mode"] == "append"
    assert res["appended_bytes"] == len(SECTION.encode("utf-8"))


def test_append_to_a_missing_path_creates_the_file_holding_content(scratch):
    rel = "memory/vault-maintenance/2026-11-22.md"
    assert not (scratch.root / rel).exists()
    res = _write(scratch, rel, SECTION, mode="append")
    assert res.get("success") is True, res
    assert (scratch.root / rel).read_text(encoding="utf-8") == SECTION
    assert res["bytes"] == len(SECTION.encode("utf-8"))


def test_the_schema_advertises_the_mode(scratch):
    import asyncio

    tools = asyncio.run(scratch.mod.list_tools())
    props = next(t for t in tools if t.name == "vault_write").input_schema["properties"]
    assert props["mode"]["enum"] == ["overwrite", "append"]


def test_an_unknown_mode_writes_nothing(scratch):
    _seed(scratch, RUN_LOG, PRE_IMAGE)
    res = _write(scratch, RUN_LOG, SECTION, mode="prepend")
    assert res.get("success") is not True, res
    assert res["code"] == "INVALID_PARAM"
    assert (scratch.root / RUN_LOG).read_text(encoding="utf-8") == PRE_IMAGE


# ── clause 2 ─────────────────────────────────────────────────────────────────

def test_an_append_leaves_the_pre_image_untouched_as_a_prefix(scratch):
    # Multi-byte text on both sides: every size here is bytes, never characters.
    pre = PRE_IMAGE + "- naïve café — ok\n"
    frag = "\n## Run 2 — résumé\n- ✓ done\n"
    _seed(scratch, RUN_LOG, pre)
    pre_bytes = (scratch.root / RUN_LOG).read_bytes()

    res = _write(scratch, RUN_LOG, frag, mode="append")
    assert res.get("success") is True, res

    on_disk = (scratch.root / RUN_LOG).read_bytes()
    assert len(on_disk) == len(pre_bytes) + len(frag.encode("utf-8"))
    assert on_disk[:len(pre_bytes)] == pre_bytes
    assert on_disk[len(pre_bytes):] == frag.encode("utf-8")
    assert res["bytes"] == len(on_disk)
    assert res["removed_lines"] == 0


def test_an_append_is_audited_as_the_whole_file_it_landed(scratch):
    _seed(scratch, RUN_LOG, PRE_IMAGE)
    _write(scratch, RUN_LOG, SECTION, mode="append")
    rows = [json.loads(line) for line in
            (scratch.root / "audit" / "writes.jsonl").read_text().splitlines() if line]
    assert rows[-1]["bytes"] == len((PRE_IMAGE + SECTION).encode("utf-8"))
    assert rows[-1]["pre_sha256"], "the append's pre-image was not recorded"


def test_an_append_refuses_when_the_file_moved_after_the_guards_priced_it(
        scratch, monkeypatch):
    """The guards price pre-image + fragment; a writer landing between that read and
    the lock would make those bytes stale, so nothing is written."""
    _seed(scratch, RUN_LOG, PRE_IMAGE)
    real = scratch.mod._read_pre_bytes
    calls = {"n": 0}

    def racing(target):
        calls["n"] += 1
        if calls["n"] == 2:  # the in-lock read: someone else wrote first
            target.write_text(PRE_IMAGE + "- a concurrent line\n", encoding="utf-8")
        return real(target)

    monkeypatch.setattr(scratch.mod, "_read_pre_bytes", racing)
    res = _write(scratch, RUN_LOG, SECTION, mode="append")
    assert res.get("success") is not True, res
    assert res["code"] == "LOCK_TIMEOUT"
    assert (scratch.root / RUN_LOG).read_text(encoding="utf-8") == \
        PRE_IMAGE + "- a concurrent line\n"


# ── clause 3 ─────────────────────────────────────────────────────────────────

def test_an_append_is_priced_on_the_prospective_whole_file_against_the_ceiling(scratch):
    ceiling = MC.memory_ceiling("USER.md")
    line = "- [user] a ruling that stays loaded\n"
    pre = line * ((ceiling - 40) // len(line))
    assert ceiling - 80 < len(pre.encode()) <= ceiling - 40
    _seed(scratch, "lloyd/USER.md", pre)
    frag = "- [user] one more ruling that will not fit in the room left\n"
    assert len(pre.encode()) + len(frag.encode()) > ceiling

    # Priced on the fragment alone this would pass: the same bytes as an overwrite land.
    assert MC.memory_write_error(scratch.root / "lloyd" / "USER.md", frag) is None

    res = _write(scratch, "lloyd/USER.md", frag, mode="append")
    assert res.get("success") is not True, res
    assert "ceiling" in res["error"], res
    assert (scratch.root / "lloyd" / "USER.md").read_text(encoding="utf-8") == pre


def test_an_append_that_fits_under_the_ceiling_lands(scratch):
    pre = "- [user] a ruling that stays loaded\n" * 10
    _seed(scratch, "lloyd/USER.md", pre)
    res = _write(scratch, "lloyd/USER.md", "- [user] another\n", mode="append")
    assert res.get("success") is True, res


# ── clause 4 ─────────────────────────────────────────────────────────────────

def test_an_overwrite_names_how_many_pre_image_lines_it_dropped(scratch):
    _seed(scratch, RUN_LOG, PRE_IMAGE)
    shorter = PRE_IMAGE.replace("- step 1 ok\n", "").replace("- step 2 ok\n", "")
    res = _write(scratch, RUN_LOG, shorter)
    assert res.get("success") is True, res
    assert res["removed_lines"] == 2
    assert isinstance(res["removed_lines"], int)


def test_an_overwrite_that_drops_nothing_reports_zero(scratch):
    _seed(scratch, RUN_LOG, PRE_IMAGE)
    res = _write(scratch, RUN_LOG, PRE_IMAGE + "- step 3 ok\n")
    assert res["removed_lines"] == 0
    res = _write(scratch, "memory/new-note.md", "# fresh\n")
    assert res["removed_lines"] == 0, "a create dropped nothing"


def test_the_incident_shape_reports_its_removal(scratch):
    """171 pre-existing lines replaced by an emission that kept one of them."""
    pre = "".join(f"- run-1 line {i}\n" for i in range(171))
    _seed(scratch, RUN_LOG, pre)
    res = _write(scratch, RUN_LOG, "- run-1 line 0\n## Run 2\n- post-flight ok\n")
    assert res.get("success") is True, res  # reported, never refused
    assert res["removed_lines"] == 170


# ── the retry the pool may send ──────────────────────────────────────────────

def test_a_retried_append_with_the_same_call_id_lands_once(scratch, monkeypatch):
    """`vault_write` is badged idempotent, so `MCPPool._retry_safe` re-sends it after a
    transport drop carrying the same `lloyd/call_id`; the second copy must not append
    the section again. A NEW call id is a new append and lands."""
    from agent_mcp import _task_registry as TR

    monkeypatch.setattr(scratch.mod, "get_bound_session", lambda: "20261126_chat")
    _seed(scratch, RUN_LOG, PRE_IMAGE)
    tok = TR.current_call_id.set("call_1")
    try:
        first = _write(scratch, RUN_LOG, SECTION, mode="append")
        retry = _write(scratch, RUN_LOG, SECTION, mode="append")
    finally:
        TR.current_call_id.reset(tok)
    assert first.get("success") is True and retry.get("success") is True, (first, retry)
    assert retry.get("replayed") is True
    assert (scratch.root / RUN_LOG).read_text(encoding="utf-8") == PRE_IMAGE + SECTION

    tok = TR.current_call_id.set("call_2")
    try:
        again = _write(scratch, RUN_LOG, SECTION, mode="append")
    finally:
        TR.current_call_id.reset(tok)
    assert again.get("success") is True and "replayed" not in again
    assert (scratch.root / RUN_LOG).read_text(encoding="utf-8") == \
        PRE_IMAGE + SECTION + SECTION
