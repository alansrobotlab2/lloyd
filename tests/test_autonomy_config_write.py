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
import re

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


# ── #1672: the surface promised a knob no code reads ─────────────────────────
#
# `autonomy_config_set` used to invite `key='max_parallel'` and answer a write
# with `{"set", "value", "path"}`. Nothing in Lloyd reads that key, or any other
# key of `_config.md`: `_read_config()` has exactly two callers, both in
# `agent_mcp/autonomy.py`, and every other mention of the file in the tree is an
# exclusion — `app/autonomy.py`, `app/routers/autonomy.py`,
# `app/routers/dashboard.py` and `app/routers/mc_ui.py` all *skip* it, and
# `grep -rEl "_config\.md" ~/obsidian/skills` returns 0 files. Worse, the string
# `max_parallel` IS live-looking: `workers/sources/autoresearch.py` and
# `bench_mine.py` read a `max_parallel` of their own out of `config.yaml`
# (`CONFIG["workers"]["sources"]`), so a grep finds a consumer for a key that has
# none. The four tests below pin the tool surface against re-learning the
# misdirection: no unread key named, both descriptions saying what the file is,
# and the payload saying per key whether anything applies it.

#: A key an `inputSchema` description offers the caller: a snake_case word in
#: quotes or backticks. Only the `key` property is scanned — a `value`
#: description legitimately quotes a sample value, and `true` would match.
_SCHEMA_NAMED_KEY = re.compile(r"""['"`]([a-z][a-z0-9_]{2,})['"`]""")


def _tools() -> dict:
    return {t.name: t for t in asyncio.run(autonomy.list_tools())}


def test_the_key_schema_names_no_key_that_no_code_reads():
    """Clause 1. `tools/list` is served to every session, so the example it
    carries is the claim a caller plans from: naming `max_parallel` says
    "here is a scheduler knob", and it is false.

    The demand is structural rather than a grep for one string: any key the `key`
    property quotes must be one code actually reads, which today means the
    description may quote nothing."""
    applied = autonomy._CONFIG_KEYS_APPLIED_BY_CODE
    assert applied == frozenset(), (
        "the applied-key set is no longer empty, so this test's demand is a "
        "different one — update it to name the live key rather than dropping it")
    tools = _tools()
    for name in ("autonomy_config", "autonomy_config_set"):
        props = tools[name].input_schema.get("properties") or {}
        quoted = set(_SCHEMA_NAMED_KEY.findall(props["key"].get("description") or ""))
        assert quoted <= applied, (
            f"{name}.key advertises {sorted(quoted - applied)} — no code reads "
            f"them, so the example promises an effect the fleet will never apply")
    # Positive control that the scanner is not blind, and that this is what the
    # removed text looked like: the pre-fix line was exactly
    # "Config key to set, e.g. 'max_parallel'".
    assert _SCHEMA_NAMED_KEY.findall("Config key to set, e.g. 'max_parallel'") == [
        "max_parallel"], "the scanner matches nothing, so the check above is free"

    surface = json.dumps([
        {"name": t.name, "description": t.description, "schema": t.input_schema}
        for t in tools.values()])
    assert "max_parallel" not in surface, (
        "the decoy key is back on the autonomy tool surface. It is greppable in "
        "config.yaml under a DIFFERENT reader (workers/sources/autoresearch.py), "
        "which is what made it the worst possible example")


def test_both_config_tools_say_a_written_key_is_documentary_until_read():
    """Clause 2. Neither half may promise the scheduler will apply the value.

    The old read half was titled "Read autonomy scheduler configuration" and the
    old `value` description said "as a string the scheduler parses" — a promise
    about a parse that exists nowhere. So the demand here is that the word
    itself is gone from both descriptions, and that both carry the one shared
    claim (`_DOCUMENTARY_PHRASE`) that says what would end the limit: a reader."""
    tools = _tools()
    for name in ("autonomy_config", "autonomy_config_set"):
        desc = tools[name].description or ""
        assert "_config.md" in desc, f"{name} does not say which file this is about"
        assert autonomy._DOCUMENTARY_PHRASE in desc, (
            f"{name} does not carry the shared claim: {desc}")
        assert "scheduler" not in desc.lower(), (
            f"{name} still names the scheduler, which is the promise that no "
            f"reader exists to keep: {desc}")
        assert "the scheduler parses" not in json.dumps(tools[name].input_schema)
    assert autonomy._DOCUMENTARY_PHRASE in autonomy._config_key_effect("anything")[1], (
        "the payload and the description no longer say the same thing")


def test_both_config_descriptions_are_one_string_literal():
    """The claim only reaches a caller if the scan that guards it can see it.

    `_gist_losses` walks `agent_mcp/*.py` and reads `Tool(description=…)` nodes,
    skipping anything that is not an `ast.Constant` — so writing the description
    as `"…" + _DOCUMENTARY_PHRASE` was enough to make both tools vanish from the
    scan, and the failure it produced was in an unrelated assertion ("no longer
    losing guidance, remove from GIST_LOSS_EXCEPTIONS") about 3,000 km from the
    edit. The shared phrase is therefore a constant only for THIS file's
    assertions; in the source it is pasted, twice, on purpose."""
    import ast
    from pathlib import Path

    src = Path(autonomy.__file__).read_text(encoding="utf-8")
    seen = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Tool"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        name, desc = kw.get("name"), kw.get("description")
        if isinstance(name, ast.Constant):
            seen[name.value] = desc
    for name in ("autonomy_config", "autonomy_config_set"):
        assert name in seen, f"{name} is no longer declared the way the scan reads"
        assert isinstance(seen[name], ast.Constant), (
            f"{name}'s description is not a literal, so `_gist_losses` skips the "
            "tool and its gist is unguarded")
        assert autonomy._DOCUMENTARY_PHRASE in seen[name].value


def test_the_success_payload_says_whether_the_key_is_applied(cfg_file, monkeypatch):
    """Clause 3, across both seams. The write did happen — the key is on disk —
    so `{"set", "value", "path"}` was not a lie, it was an implication: a
    returned call naming a config file is read as a knob that took.

    The payload now says per key whether anything applies it, and the flag comes
    from `_CONFIG_KEYS_APPLIED_BY_CODE` rather than being hardcoded `False`: the
    second half of this test puts the key in that set and demands the SAME call
    flip to saying it is applied. Without that the field could never be wrong,
    and a field that can never be wrong reports nothing."""
    out = json.loads(autonomy._handle_config_set(
        {"key": "max_parallel", "value": "3"}))
    assert (out["set"], out["value"]) == ("max_parallel", "3"), out
    assert out["path"] == str(cfg_file), out
    # Bytes BELOW the closing fence — `TAIL` above still carries that fence,
    # `_split_config` hands back only what follows it.
    assert out["bytes_below_front_matter"] == len(BODY.encode("utf-8")), out
    assert out["applied_by_code"] is False, out
    assert "not read by any code" in out["effect"], out
    assert autonomy._read_config()["max_parallel"] == "3", (
        "the new field is a refusal in disguise: the key must still be written")

    monkeypatch.setattr(autonomy, "_CONFIG_KEYS_APPLIED_BY_CODE",
                        frozenset({"max_parallel"}))
    again = json.loads(autonomy._handle_config_set(
        {"key": "max_parallel", "value": "4"}))
    assert again["applied_by_code"] is True, again
    assert "will be applied" in again["effect"], again

    # The seam a caller actually crosses: `call_tool` out of `agent_mcp.main`,
    # which is where the JSON becomes text over MCP.
    over_mcp = json.loads(asyncio.run(autonomy.call_tool(
        "autonomy_config_set", {"key": "some_new_key", "value": "1"}
    )).content[0].text)
    assert over_mcp["applied_by_code"] is False, over_mcp
    assert over_mcp["path"] == str(cfg_file), over_mcp


def test_the_read_write_split_and_the_clobber_safe_round_trip_are_intact(cfg_file):
    """Clause 4. #1672 is a wording fix, so everything #1326 established has to
    survive it unchanged: the read refuses a `value` and is still in `READ_ONLY`
    behind those four refusals, the writer still requires both parameters, the
    tail below the closing fence still comes back byte for byte, and a front
    matter that cannot be round-tripped is still refused rather than rewritten.

    This is the control on the whole round: a test that cannot fail would let a
    payload change quietly re-merge the two tools or drop the body again."""
    from agent_mcp import annotations

    assert "autonomy_config" in annotations.READ_ONLY
    assert "autonomy_config_set" not in annotations.READ_ONLY, (
        "the writer is now annotated read-only, which is the #1326 bug returning")

    refused = json.loads(autonomy._handle_config({"key": "a", "value": "1"}))
    assert refused["error"] == autonomy._CONFIG_WRITE_MOVED, refused
    assert refused["use_tool"] == "autonomy_config_set", refused
    assert cfg_file.read_bytes() == LIVE_CONFIG.encode(), "the read half writes"

    tools = _tools()
    assert sorted(tools["autonomy_config_set"].input_schema.get("required") or []) == [
        "key", "value"]
    assert "value" not in (tools["autonomy_config"].input_schema.get("properties") or {})

    written = json.loads(autonomy._handle_config_set({"key": "paused", "value": "true"}))
    assert written["set"] == "paused", written
    after = cfg_file.read_text(encoding="utf-8")
    assert after.endswith(TAIL), f"the body below the fence moved: {after[-60:]!r}"
    assert _front_matter(after)["paused"] == "true", after

    cfg_file.write_text("---\n: this is not yaml [\n---\n" + BODY, encoding="utf-8")
    before = cfg_file.read_bytes()
    refused = json.loads(autonomy._handle_config_set({"key": "a", "value": "1"}))
    assert refused.get("yaml_broken") is True, refused
    assert cfg_file.read_bytes() == before, "an unparseable front matter was rewritten"


# ── #2096: the empty set is load-bearing, and the tier now says what it assumes ─
#
# `_CONFIG_KEYS_APPLIED_BY_CODE` is the single input that decides whether
# `autonomy_config_set` is a scheduler-authority write or a note in a markdown
# file. The answer on the other side of that decision is a different module:
# `app/harness/policy.py` never reads the name (`git grep
# "_CONFIG_KEYS_APPLIED_BY_CODE" app/harness/policy.py` → 0 hits), so nothing at
# either site connects them, and the tier the writer sits at was chosen while the
# set was empty. #2096 leaves the tier exactly where it is — a pointer, not a gate,
# because #2094 was closed with the set empty and no round may pre-make its answer —
# and makes the silence self-enforcing instead.
#
# Not a duplicate of `test_the_key_schema_names_no_key_that_no_code_reads` above,
# which asserts this same set is empty for its own reason (an `inputSchema`
# example may not quote a key no code reads). That node goes red on a key because
# the surface claim it pins changes; these two go red on the set/tier PAIRING,
# which is the claim nothing else holds. Both are supposed to fail on that day, for
# different reasons, and neither is a grep of the comment block: #2096's own ruling
# is that prose is not a test's job, so no node here reads autonomy.py's bytes.


def test_the_empty_applied_key_set_is_the_only_thing_keeping_the_writer_at_tier_1():
    """Clause 2: the pairing, asserted before either side hardens into habit.

    The biconditional is the whole node: `autonomy_config_set` may sit at tier 1
    exactly while no key of `_config.md` is applied by code. Today both sides hold
    and the node is quiet. The day a key joins the set, the writer is an unattended
    turn rewriting what the fleet runs while `effective_tier` still answers the
    tier a documentary note belongs at — tier 1 is where `check_grants` returns its
    allow before opening the grant store (`policy.py:1112-1113`) and the pre-tool
    callback short-circuits identically (`:1295-1296`), so nothing is consulted and
    no deny row is written — and that is the choice #2094 was closed for not
    making, which is why it must be red and must say so. The opposite drift is red
    too: a tier above 1 while every key is still documentary would gate a
    documentary markdown write, the class the traffic census puts vault writers in
    on purpose.

    Nothing here reads the comment block #2096 also landed. Pinning prose would
    make a rewording a regression and a deletion free; the pairing is the
    instrument.
    """
    from app.harness.policy import effective_tier

    applied = autonomy._CONFIG_KEYS_APPLIED_BY_CODE
    tier = effective_tier("autonomy_config_set",
                          {"key": "max_parallel", "value": "3"})

    assert bool(applied) == (tier != 1), (
        f"#2096's tripwire: `_CONFIG_KEYS_APPLIED_BY_CODE` holds {sorted(applied)} "
        f"and `effective_tier` answers {tier}, and those two answers have to be "
        "made together. If the set is no longer empty then a write to "
        "`_config.md` changes dispatch behaviour for the fleet, which is a "
        "scheduler-authority write and can no longer sit at tier 1 — the tier "
        "`check_grants` allows with the grant store still closed "
        "(`app/harness/policy.py:1112-1113`), so an unattended scope needs no "
        "grant and leaves no deny row. Raise the tool's tier (or take the key "
        "out of the set); do not inherit tier 1 past the change that made it "
        f"live. If you DID raise it and {tier} is your deliberate answer, this "
        "node is the one that has to be updated to pin the new pairing — #2094 is "
        "the item that question belongs on.")

    # Today's two halves, each read live from its own module, so a passing
    # biconditional above can only be the pair #2096 left behind and never a
    # coincidence of two values that both moved.
    assert applied == frozenset(), (
        "the pairing passed only because the tier moved as well; that is the "
        f"decision the node above demands be written down: {sorted(applied)}")
    assert tier == 1, (
        "the pairing passed only because a key joined the set, which is the same "
        f"decision: effective_tier answered {tier}")


def test_the_pointer_took_no_authority_scope(tmp_path):
    """Clause 3: #2096 names a consequence and changes no authority.

    Every claim here was true the moment before the comment landed, and has to
    stay true or this round quietly answered #2094: the writer is tier 1 by both
    tier functions, its name is in neither tier table, the tier-2→1 demotion branch
    still belongs to `autonomy_write_task` alone — so no argument shape of this
    tool, however scheduler-shaped, raises it — and the tier-1 path really does
    answer across the process boundary without opening the grant store, which is
    what makes "no grant consulted and no deny row" a fact rather than an
    assumption.

    The control at the end is why the allow means anything: from the same
    unattended scope, on the same tmp store, the schedule-changing
    `autonomy_write_task` call #2093 removed a grant for IS denied and leaves its
    deny row. A probe that cannot deny would call this whole node a pass on a
    broken harness.
    """
    from app.harness.policy import (TIER2_TOOLS, TIER3_TOOLS, GrantStore,
                                    SCHEDULE_STATE_TOOL, check_grants,
                                    effective_tier, tool_tier)

    cfg_call = {"key": "max_parallel", "value": "3"}
    assert tool_tier("autonomy_config_set", {}) == 1
    assert effective_tier("autonomy_config_set", cfg_call) == 1
    assert effective_tier("autonomy_config_set",
                          {"id": 40, "status": "up_next"}) == 1, (
        "a scheduler-shaped argument now raises this tool's tier, so #2094's gate "
        "question was answered by a round whose contract was a comment")
    assert "autonomy_config_set" not in TIER2_TOOLS
    assert "autonomy_config_set" not in TIER3_TOOLS
    assert SCHEDULE_STATE_TOOL == "autonomy_write_task", (
        "the demotion branch is no longer the only reason the tool stays at 1, so "
        f"check what moved: {SCHEDULE_STATE_TOOL!r}")

    store = GrantStore(tmp_path / "workers.db")
    store.ensure_schema()
    d = check_grants(store, scope="worker:autotriage",
                     tool_name="autonomy_config_set", tool_input=cfg_call)
    assert (d.allowed, d.grant_id, d.reason) == (True, None, ""), (
        f"the config write is no longer the tier-1 pass-through: {d}")
    assert store.dispatch_rows() == [], (
        "a tier-1 documentary write now consults the grant store, which is "
        f"#2094's gate — decided somewhere it should have been decided loudly: "
        f"{store.dispatch_rows()}")

    sched = check_grants(store, scope="worker:autotriage",
                         tool_name="autonomy_write_task",
                         tool_input={"id": 68, "status": "up_next"})
    assert sched.allowed is False, (
        "positive control broken: the grant gate denies nothing, so the allow "
        f"above proves nothing either: {sched}")
    assert "#68" in sched.reason, sched.reason
    assert [r["decision"] for r in store.dispatch_rows()] == ["deny"], (
        "the deny is not on the decision surface, so the asymmetry this node "
        "pins is invisible to whoever reads the store later")

def test_the_installed_hook_and_the_qualified_name_leave_the_writer_ungated(tmp_path):
    """Clause 3 again from the two boundaries the node above stops at.

    `test_the_pointer_took_no_authority_scope` calls `check_grants` directly, which
    is not how a turn reaches the gate. The dispatch path installs a callback whose
    FIRST act is its own `effective_tier(...) == 1` early return
    (`app/harness/policy.py:1285-1297`) — the half of the hazard the new comment
    cites that lives in the callback and not in `check_grants`, and the reason a
    direct call leaves it unmeasured. So this node drives the real
    `install_policy_hook` over a real `HookRegistry`.

    It also arrives under the name the MCP registry actually carries. A tool call
    reaches the harness as `mcp__<server>__<tool>` and `normalize_tool_name`
    (`policy.py:264-270`) unwraps it, while `_config.md` is written in the aggregator
    process; a tier pinned only on the bare name in one process would miss the same
    call moving under qualification, which is why the node reads both spellings
    through the tier function AND through the installed hook.

    Same asymmetry as the comment claims, same scratch store: the config write
    passes under either spelling and leaves no row at all, and the
    schedule-changing `autonomy_write_task` call that store and scope do deny.
    """
    from app.harness import HookRegistry
    from app.harness.policy import (GrantStore, effective_tier,
                                    install_policy_hook, normalize_tool_name)

    cfg_call = {"key": "max_parallel", "value": "3"}
    spellings = ("autonomy_config_set", "mcp__agent__autonomy_config_set")
    for name in spellings:
        assert normalize_tool_name(name) == "autonomy_config_set", (
            f"the unwrapping this node depends on is not what it was: {name} → "
            f"{normalize_tool_name(name)!r}")
        assert effective_tier(name, cfg_call) == 1, (
            f"qualification moved the tier: {name} answers "
            f"{effective_tier(name, cfg_call)}")

    hooks = HookRegistry()
    store = GrantStore(tmp_path / "grants.sqlite")
    store.ensure_schema()
    install_policy_hook(hooks, store=store, scope="worker:autotriage")

    for name in spellings:
        passed = asyncio.run(hooks.fire_pre_tool_use(
            session_id="s2096", tool_name=name, tool_input=cfg_call))
        assert passed == {}, (
            f"the installed hook no longer lets the documentary write through "
            f"under {name}: {passed}")
    assert store.dispatch_rows() == [], (
        "the callback consulted the grant store for a tier-1 documentary write, "
        f"which is #2094's gate arriving quietly: {store.dispatch_rows()}")

    denied = asyncio.run(hooks.fire_pre_tool_use(
        session_id="s2096", tool_name="autonomy_write_task",
        tool_input={"id": 68, "status": "up_next"}))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", (
        "the control is broken — the same hook denies nothing, so the two passes "
        f"above prove nothing either: {denied}")
    assert "#68" in str(denied), (
        f"the deny does not name the task it protected: {denied}")
    assert [r["decision"] for r in store.dispatch_rows()] == ["deny"], (
        "the asymmetry is not on the decision surface for whoever reads the store "
        f"next: {store.dispatch_rows()}")
