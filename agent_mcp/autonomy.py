#!/usr/bin/env python3
"""
Lloyd MCP Server: Autonomy — task management tools.

Provides tools for creating, listing, running, and managing autonomy tasks
stored in ~/obsidian/autonomy/ as markdown files with YAML frontmatter.

Tools: autonomy_tasks, autonomy_write_task, autonomy_get_task,
       autonomy_delete_task, autonomy_config, autonomy_config_set,
       autonomy_run_task
"""

import asyncio
import datetime
import json
import re
import sys
from pathlib import Path

import yaml
from mcp.types import Tool

from agent_mcp._shared import (
    AUTONOMY_TASK_FIELDS, parse_frontmatter_text, text_result, with_body_contract)

# One name rule for the one directory: a task file is `NN-slug.md`, and anything
# else that lives in `~/obsidian/autonomy/` is a report, a note, or the config
# block — whatever its frontmatter says. #1594 wrote that rule into
# `app/routers/autonomy.py::_TASK_NAME_RE` and this reader was pointed at it in
# the architecture table, but the import never happened, so `_handle_tasks` went
# on globbing the directory and excluding one filename by literal (#1692).
# Imported rather than copied: object identity is the thing that stops the two
# listings disagreeing again, and a test pins it.
from app.routers.autonomy import _TASK_NAME_RE

AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"


def _slugify(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name[:50]


def _parse_task_file(path: Path) -> dict | None:
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        # Resilient parse: bad YAML degrades to regex extraction instead of
        # returning None (a None here silently drops the task from listings —
        # the same failure mode that dormant-killed 34/40 scheduler tasks).
        #
        # `AUTONOMY_TASK_FIELDS` is the scheduler's list, imported rather than
        # re-typed: this reader used to carry 13 of its own names, so a broken
        # file came back here with no `preferred_hours`, `depends_on`, `model`,
        # `stale_bypass_hours`, `inner_voice`, `auto_advance`, `preemptible`,
        # `max_retries`, `failure_count`, `runs_per_day`, `last_attempt` or
        # `expected_error_patterns` — and since `_write_task_file` below writes
        # from this read, those 12 got written away. One file, one field list
        # (#1014).
        frontmatter = parse_frontmatter_text(
            parts[1],
            fallback_fields=AUTONOMY_TASK_FIELDS,
            log_label=f"autonomy:{path.name}",
        )

        def _to_iso(val):
            if val is None:
                return None
            if isinstance(val, datetime.datetime):
                return val.strftime("%Y-%m-%dT%H:%M:%SZ")
            return val

        return {
            "id": frontmatter.get("id", 0),
            "name": frontmatter.get("name", ""),
            "description": frontmatter.get("description", ""),
            "status": frontmatter.get("status", "draft"),
            "priority": frontmatter.get("priority", "medium"),
            "frequency": frontmatter.get("frequency", ""),
            "scheduled_at": _to_iso(frontmatter.get("scheduled_at", "")),
            "last_run": _to_iso(frontmatter.get("last_run")),
            "next_run": _to_iso(frontmatter.get("next_run")),
            "auto_advance": bool(frontmatter.get("auto_advance", False)),
            "preemptible": bool(frontmatter.get("preemptible", True)),
            # `board_id` used to be emitted here, defaulted to 4 for every task.
            # It is in no task file (0 of 33 at triage), read by no scheduler
            # path, and named by no board: the phantom was the tell that this
            # reader's field list had drifted from the scheduler's, so it goes
            # with the list (#1014).
            "inner_voice": frontmatter.get("inner_voice"),
            "created_at": _to_iso(frontmatter.get("created", frontmatter.get("created_at", ""))),
            "updated_at": _to_iso(frontmatter.get("updated", frontmatter.get("updated_at", ""))),
            "skill_name": frontmatter.get("skill_name", frontmatter.get("skill_path", "")),
            "agent_id": frontmatter.get("agent_id", "memory"),
            "model": frontmatter.get("model", ""),
            "timeout_seconds": frontmatter.get("timeout_seconds", 1800),
            "depends_on": frontmatter.get("depends_on"),
            "pipeline": frontmatter.get("pipeline"),
            "runs_per_day": frontmatter.get("runs_per_day"),
            "run_count": frontmatter.get("run_count"),
            "failure_count": frontmatter.get("failure_count", 0),
            # #1085. Worth more here than a complete listing: every
            # `autonomy_get_task` / `autonomy_tasks` answer comes through THIS
            # projection, and the file itself is not reachable across the process
            # boundary to check. A ceilinged task therefore came back as a
            # `status: up_next` row with no hold and no counter — a task that
            # appears never to have failed, to the one reader a human or another
            # agent consults to ask why it has been silent. `_write_task_file`
            # starts from the file's PRIOR frontmatter, so carrying them in does
            # not make this module a writer of them.
            "infra_failure_count": frontmatter.get("infra_failure_count", 0),
            "infra_rest_until": _to_iso(frontmatter.get("infra_rest_until")),
            "max_retries": frontmatter.get("max_retries", 3),
            "notify_on_complete": frontmatter.get("notify_on_complete", True),
            "preferred_hours": frontmatter.get("preferred_hours", []),
            "last_attempt": _to_iso(frontmatter.get("last_attempt")),
            "stale_bypass_hours": frontmatter.get("stale_bypass_hours"),
            "expected_error_patterns": frontmatter.get("expected_error_patterns") or [],
            "cron_id": frontmatter.get("cron_id"),
            # #1555. Read-side half only, but the half that decides whether the
            # guard fires: the UPDATE path below asks the projected dict for
            # `requires_slot`, and a task whose declaration this projection dropped
            # would read exactly like one that declares no slot — the fails-open
            # shape again, one layer in. `_write_task_file` starts from the file's
            # PRIOR frontmatter, so the key already survives the write untouched.
            "requires_slot": frontmatter.get("requires_slot"),
            "type": frontmatter.get("type", "autonomy"),
            # Carried through deliberately, not incidental: `_write_task_file`
            # refuses a record that only reached us by regex fallback, and the
            # only way it can know is if the read says so.
            "_yaml_broken": bool(frontmatter.get("_yaml_broken")),
            "body": parts[2] if len(parts) > 2 else "",
        }
    except Exception:
        return None


class BrokenFrontmatterError(Exception):
    """Refusal to round-trip a file that only parsed by regex fallback.

    This module is a read-modify-write path: `_write_task_file` starts from the
    file's existing frontmatter so unmodelled keys survive. That is exactly what
    makes a degraded read fatal — on a YAML-broken file the prior parse is a
    regex recovery over `AUTONOMY_TASK_FIELDS` and nothing else, so dumping it
    back rewrites the file from that subset. `AUTONOMY_TASK_FIELDS` is now the
    scheduler's full list, so a round-trip no longer destroys `preferred_hours`,
    `depends_on` or `model`; it would still drop every key outside the list
    (`category`, `segment`, `timestamp`, `paused_reason`, and anything added
    later) and would silently re-dump a file whose YAML is genuinely broken.

    A degraded record is fine to show; it is not fine to round-trip — the same
    rule `agent_mcp/backlog.py::save_task` and
    `app/routers/backlog.py::_reject_broken_fm` already enforce, and the reason
    the scheduler prefers to report an uninterpretable task rather than drop it
    (#1014)."""


def _refuse_broken(exc: Exception) -> str:
    """The tool-protocol shape of the refusal: an `error` the caller can read,
    not a traceback out of the dispatcher."""
    return json.dumps({"error": str(exc), "yaml_broken": True})


#: #2408: a task's `description` is the text the scheduler prompts the model with
#: (`app/autonomy.py::_build_task_prompt`), so every number in it is a claim about a
#: constant in this checkout. `scripts/automod/vault_round.py:630` has compared those
#: claims since #2317 — at a vault LAND, on the state of the landed file. The route the
#: prose actually travels is this module, and no commit had ever guarded it
#: (`git log -S'constant_quotes' -- agent_mcp/` is empty at #2408's base), so the
#: afternoon `01dea8bc` moved `LEDGER_ARCHIVE_AGE_DAYS` from 30 to 14 an
#: `autonomy_write_task(description=…)` could have gone on publishing 30.
def _stale_description_error(description) -> str | None:
    """Refuse a description quoting a tree constant at a value the tree does not have.

    Returns the tool-protocol refusal — the module's plain `{"error": ...}` string, the
    shape every other refusal here uses — or None when the description may be published.
    Never raised: a caller has to read this the way it reads the `slot_arm_block`
    refusal inside `_handle_write`. `_refuse_broken` is deliberately NOT reused, because
    its `yaml_broken: True` means "front matter only parsed by the regex fallback"
    everywhere on this box (`agent_mcp/_shared.py`, `app/routers/autonomy.py`,
    `tests/test_autonomy_write_refuses_broken_frontmatter.py`) and would send the caller
    off to fix YAML that is fine.

    The refusal names the constant and BOTH numbers, for the reason
    `vault_guards.autonomy_description_errors` gives: the fix is to type one of them, and
    a refusal that said only "disagrees" sends the author back to grep the script.

    **Judged on the caller's argument, never on the record.** `_write_task_file` is the
    one writer all three callers pass through — the right place for the `updated` stamp —
    but `_handle_write`'s update branch hands it the whole parsed record, description
    included, so checking the record there would refuse every write that parks a task
    whose prose is already stale. `draft`/`paused` are never dispatched, so that write is
    how a job gets stopped from this tool. The vault-side rail can afford to read the
    state of the file because a land is not how the engine stops a job; this half is the
    delta, and being a delta is what makes refusing cheap — the caller fixes the number in
    the retry it is already making, so it cannot deadlock.

    A resolver that cannot run (no git, a timeout) returns None rather than stopping the
    engine, and the number is still judged at the land. Zero resolved pairs also yields no
    refusal, which `constant_quotes.Report.resolved` exists to make visible on the
    vault-side witness and which this path deliberately does not police: most task
    descriptions legitimately name no constant, and refusing those would block ordinary
    writes on an instrument that had nothing to say.

    **Fail-open, and it cannot swallow anyone's error.** The item's stated failure mode is
    a guard that RAISES into the dispatcher, so everything the resolver does is inside the
    `try`, and the tuple covers what that work can actually raise: `OSError` walking the
    tree, `UnicodeDecodeError` reading a file, `subprocess.SubprocessError` if a harvest
    shells out, and `ValueError` because `mismatches` is the code that converts numbers
    here. Nothing in this module parses a caller's JSON — `grep -n json.loads
    agent_mcp/autonomy.py` returns only this sentence — so no exception class named in
    that tuple is ever raised on a caller's behalf, and no branch of it re-raises: a
    resolver that cannot answer yields None and the write proceeds, which is also what
    keeps a broken instrument from blocking the `status: draft` park that stops a job. The
    number is still judged at the land either way.
    """
    text = str(description or "")
    if not text.strip():
        return None
    import subprocess

    from app.paths import LLOYD_HOME
    from scripts.automod import constant_quotes as CQ

    try:
        found = CQ.mismatches(text, CQ.tree_constants(LLOYD_HOME)).found
    except (OSError, UnicodeDecodeError, subprocess.SubprocessError, ValueError):
        return None
    if not found:
        return None
    # The same sentence `vault_guards.autonomy_description_errors` refuses a land with,
    # so the two rails that police one number read alike to whoever gets refused.
    detail = "; ".join(f"its description quotes {name} as {quoted}; the tree says "
                       f"{actual}" for name, quoted, actual in found)
    return json.dumps({"error": (
        f"refusing to publish this description: {detail}. Retry with the tree's number, "
        "or land the code that moves it first — the description is what the task is "
        "prompted with, and the task file was not written.")})


def _write_task_file(task_dict: dict) -> Path:
    if task_dict.get("_yaml_broken"):
        raise BrokenFrontmatterError(
            f"task #{task_dict.get('id')} frontmatter only parsed by regex fallback; "
            "refusing to rewrite it (fix the file's YAML first)")
    task_id = task_dict.get("id", 0)
    name = task_dict.get("name", "unnamed")
    slug = _slugify(name)
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)

    existing = _find_task_file(task_id)
    path = existing if existing else AUTONOMY_DIR / f"{task_id}-{slug}.md"

    # Start from the file's EXISTING frontmatter so keys this module doesn't
    # model survive the write. The old code rebuilt frontmatter from the map
    # below alone, which destroyed `tags`, `segment`, `stale_bypass_hours`,
    # `expected_error_patterns`, `archived_reason` and anything else added
    # later — and, worse, substituted a DEFAULT for keys the caller omitted
    # (e.g. an update that didn't mention preferred_hours reset it to []).
    frontmatter: dict = {}
    if existing and existing.exists():
        try:
            from agent_mcp._shared import parse_frontmatter_text
            prior_parts = existing.read_text(encoding="utf-8").split("---\n", 2)
            if len(prior_parts) >= 3:
                prior = parse_frontmatter_text(
                    prior_parts[1], log_label=f"autonomy_write_task:{existing.name}")
                if isinstance(prior, dict):
                    frontmatter = {k: v for k, v in prior.items()
                                   if not str(k).startswith("_")}
        except Exception:
            pass

    updates = {
        "type": task_dict.get("type") or "autonomy",
        "id": task_dict.get("id", 0),
        "name": task_dict.get("name"),
        "description": task_dict.get("description"),
        "status": task_dict.get("status"),
        "priority": task_dict.get("priority"),
        "frequency": task_dict.get("frequency"),
        "agent_id": task_dict.get("agent_id"),
        "model": task_dict.get("model"),
        "auto_advance": task_dict.get("auto_advance"),
        "preemptible": task_dict.get("preemptible"),
        "pipeline_mode": task_dict.get("pipeline_mode"),
        "timeout_seconds": task_dict.get("timeout_seconds"),
        "max_retries": task_dict.get("max_retries"),
        "failure_count": task_dict.get("failure_count"),
        "skill_name": task_dict.get("skill_name", task_dict.get("skill_path")),
        "cron_id": task_dict.get("cron_id"),
        "runs_per_day": task_dict.get("runs_per_day"),
        "scheduled_at": task_dict.get("scheduled_at"),
        "last_run": task_dict.get("last_run"),
        "last_attempt": task_dict.get("last_attempt"),
        "next_run": task_dict.get("next_run"),
        "depends_on": task_dict.get("depends_on"),
        "preferred_hours": task_dict.get("preferred_hours"),
        "notify_on_complete": task_dict.get("notify_on_complete"),
        "pipeline": task_dict.get("pipeline"),
        "stale_bypass_hours": task_dict.get("stale_bypass_hours"),
        "expected_error_patterns": task_dict.get("expected_error_patterns"),
        "created": task_dict.get("created_at", task_dict.get("created")),
        "updated": task_dict.get("updated_at", task_dict.get("updated")),
    }
    frontmatter.update({k: v for k, v in updates.items() if v is not None})
    frontmatter = {k: v for k, v in frontmatter.items() if v is not None}
    # The stamp lives here, in the one writer both the create and the update path
    # pass through, rather than in the create branch's `"body": ""`: the pin reads
    # the live task directory, so it has to hold for whatever reaches disk, not
    # only for the one call site that happens to look like a creation.
    body = with_body_contract(task_dict.get("body", ""))
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{body}"
    path.write_text(content, encoding="utf-8")
    return path


def _find_task_file(task_id: int) -> Path | None:
    if not AUTONOMY_DIR.exists():
        return None
    patterns = [p for p in AUTONOMY_DIR.glob(f"{task_id}-*.md") if p.name != "_config.md"]
    return patterns[0] if patterns else None


def _next_task_id() -> int:
    if not AUTONOMY_DIR.exists():
        return 1
    max_id = 0
    for path in AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        name = path.name
        if "-" in name:
            id_str = name.split("-")[0]
            if id_str.isdigit():
                max_id = max(max_id, int(id_str))
    return max_id + 1


def _parse_run_file(path: Path) -> dict | None:
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        frontmatter = yaml.safe_load(parts[1])
        if not isinstance(frontmatter, dict):
            return None
        return {
            "run_id": frontmatter.get("run_id", 0),
            "task_id": frontmatter.get("task_id", 0),
            "status": frontmatter.get("status", ""),
            "duration_seconds": frontmatter.get("duration_seconds"),
            "started_at": frontmatter.get("started_at"),
            "completed_at": frontmatter.get("completed_at"),
            "body": parts[2] if len(parts) > 2 else "",
        }
    except Exception:
        return None


def _read_config() -> dict:
    config_path = AUTONOMY_DIR / "_config.md"
    if not config_path.exists():
        return {}
    try:
        content = config_path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) >= 2:
            frontmatter = yaml.safe_load(parts[1])
            if isinstance(frontmatter, dict):
                return frontmatter
    except Exception:
        pass
    return {}


#: #1672: the keys of `_config.md` that any code READS and therefore APPLIES.
#: Empty, measured not assumed: `_read_config()` has exactly two callers, both
#: in this file (`_handle_config`, `_handle_config_set`), and every other mention
#: of `_config.md` in Lloyd's own code is an exclusion — the scheduler
#: (`app/autonomy.py`), the board and the dashboard routers and `mc_ui` all *skip*
#: the file, and no skill references it. A key joins this set only when a change
#: both names it here and reads it there; until then a write is a note in a
#: markdown file. It is a set and not a comment so the reply can say which of the
#: two a given key is, instead of leaving the caller to infer an effect.
#:
#: #2096 — and this is the part the empty value above was hiding. This set is the
#: only thing that decides whether `autonomy_config_set` is a scheduler-authority
#: write or a note in a markdown file, and the authority answer is on the other
#: side of a process boundary, in `app/harness/policy.py`, which never reads this
#: name: `autonomy_config_set` is in neither `TIER2_TOOLS` nor `TIER3_TOOLS`, the
#: tier-2→1 demotion branch is hardcoded to `SCHEDULE_STATE_TOOL =
#: "autonomy_write_task"` (policy.py:306, :432-435), so no call shape of this tool
#: raises it, and `effective_tier` answers 1 for it — measured, not assumed:
#: `effective_tier("autonomy_config_set", {"key": "max_parallel", "value": "3"})
#: == 1`. At tier 1 `check_grants` returns its allow before the grant store is
#: opened (policy.py:1112-1113) and the pre-tool callback short-circuits the same
#: way (:1295-1296), so an unattended scope — `autonomy-task:40`,
#: `worker:autotriage` — writes `_config.md` with no grant consulted and no deny
#: row. Tier 1 is the right answer here, and it is right for exactly one reason:
#: every key of this file is documentary, which is the same class the traffic
#: census deliberately puts vault writers at (see
#: `tests/test_side_effect_traffic_census.py`). A non-empty set makes that false —
#: an unattended turn would then be rewriting what the fleet runs, at tier 1,
#: ungated — so adding a key here RE-OPENS a tier decision (#2094 asked it and was
#: closed with the set empty, so nobody owes this round an answer). Do not inherit
#: tier 1 past that change: make it again. The pairing is enforced, not hoped for —
#: `tests/test_autonomy_config_write.py::test_the_empty_applied_key_set_is_the_only_thing_keeping_the_writer_at_tier_1`
#: goes red the day the set and the tier disagree.
_CONFIG_KEYS_APPLIED_BY_CODE: frozenset[str] = frozenset()

#: The claim both halves of the tool pair carry about this file. It appears in
#: each description as a LITERAL — `_gist_losses` (app/harness/tests/
#: test_tool_search.py) reads `Tool(description=…)` by AST and skips anything
#: that is not a Constant, so a `+` here would silently drop both tools out of
#: the scan that exists to catch a gist losing its guidance. `tests/
#: test_autonomy_config_write.py` pins that both descriptions still say it, and
#: `_config_key_effect` below repeats it in the reply, so the two cannot drift
#: without a test going red.
_DOCUMENTARY_PHRASE = "documentary until some code reads them, and nothing does"


def _config_key_effect(key: str) -> tuple[bool, str]:
    """Whether a key written by `autonomy_config_set` does anything, and the
    sentence that says so (#1672).

    The success payload used to be `{"set", "value", "path"}`: it named a file
    and returned, which reads as a knob that took. For a key in
    `_CONFIG_KEYS_APPLIED_BY_CODE` a write genuinely is an effect; for one that
    is not, the only honest payload says so in the same object.
    """
    applied = key in _CONFIG_KEYS_APPLIED_BY_CODE
    if applied:
        return True, (f"`{key}` is read by code and will be applied on the next "
                      "read of the config.")
    return False, (
        f"`{key}` is not read by any code. Keys in `_config.md` are documentary "
        "until some code reads them, and nothing does: the only reader of this "
        "file is this tool pair — the scheduler, the board and the dashboard all "
        "skip it — so the write is a note in a markdown file. The scheduler's "
        "real tuning lives in `config.yaml` under `autonomy:`, which neither "
        "tool reaches.")


#: The front-matter fence of `_config.md`. The file is read with
#: `split("---\n", 2)`, so the *second* fence is the closing one and everything
#: after it is prose — a rewrite must not care what that prose contains, which
#: is why this splits rather than re-parses (#1326).
_CONFIG_FENCE = "---\n"


def _split_config(raw: str) -> tuple[str, dict, str] | None:
    """`(front_matter_text, parsed, tail)` for `_config.md`, or None.

    None means "this writer cannot promise a safe rewrite": no closing fence,
    prose above the opening one, or a front matter that does not parse as a
    mapping. `tail` is every byte below the closing fence, kept verbatim —
    `_write_config` used to rebuild the file from the parsed front matter alone,
    so any prose under it was discarded on every set.

    The boundary is the same `split("---\n", 2)` `_read_config` uses, so the two
    cannot disagree about where the front matter ends even when the tail itself
    contains a line that reads like a fence (the live file's does).
    """
    parts = raw.split(_CONFIG_FENCE, 2)
    if len(parts) < 3 or parts[0].strip():
        return None
    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    if not isinstance(frontmatter, dict):
        return None
    return parts[1], frontmatter, parts[2]


def _write_config(config: dict, tail: str = "") -> None:
    """Rewrite the front matter and re-emit `tail` untouched.

    `tail` defaults to "" only for a file that does not exist yet; every
    existing file must hand back the bytes `_split_config` read out of it.
    """
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    config_path = AUTONOMY_DIR / "_config.md"
    content = (f"---\n"
               f"{yaml.dump(config, default_flow_style=False, allow_unicode=True)}"
               f"{_CONFIG_FENCE}{tail}")
    config_path.write_text(content, encoding="utf-8")


# ── Tool definitions ──────────────────────────────────────────────────────────

async def list_tools():
    return [
        Tool(name="autonomy_tasks", description="Use to find scheduled autonomy tasks; to open one with its run history use autonomy_get_task. List the scheduled autonomy tasks, optionally filtered by status, frequency or agent. Returns task objects with their schedule and last-run state, not their full history.", inputSchema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter by status (draft, up_next, in_progress)"},
                "frequency": {"type": "string", "description": "Filter by frequency"},
                "agent_id": {"type": "string", "description": "Filter by agent_id"},
            },
        }),
        Tool(name="autonomy_write_task", description="Use only for recurring work; a one-off that outlives this turn goes to backlog_write_task, a subtask to Task. Every task re-runs on its frequency until its file is deleted — there is no run-once. Create or update (upsert) autonomy task. If id omitted → CREATE, if id provided → UPDATE.", inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Task ID to update (0 for create)"},
                "name": {"type": "string", "description": "Task name/title"},
                "description": {"type": "string", "description": "Task description"},
                "status": {"type": "string", "description": "Task status (draft, up_next, in_progress)"},
                "priority": {"type": "string", "description": "Task priority (low, medium, high)"},
                "frequency": {"type": "string", "description": "Task frequency"},
                "skill_name": {"type": "string", "description": "Skill slug (e.g. 'autonomy-data-pipeline'). Resolves to ~/obsidian/skills/<slug>/SKILL.md"},
                "agent_id": {"type": "string", "description": "Agent ID to run the task"},
                "model": {"type": "string", "description": "Model to use"},
                "timeout_seconds": {"type": "integer", "description": "Timeout in seconds"},
                "auto_advance": {"type": "boolean", "description": "Move the task to the next status automatically when a run succeeds"},
                "preemptible": {"type": "boolean", "description": "Allow the scheduler to interrupt this run for a higher-priority task"},
                "scheduled_at": {"type": "string", "description": "ISO timestamp for the next run; overrides the frequency for one cycle"},
                "depends_on": {"type": "integer", "description": "Task ID that must complete successfully before this one runs"},
                "pipeline": {"type": "string", "description": "Named pipeline this task belongs to; tasks in one pipeline run in order"},
                "activity_note": {"type": "string", "description": "Note to append to activity log"},
            },
        }),
        Tool(name="autonomy_get_task", description="Use to read one autonomy task's state and last runs before autonomy_run_task or autonomy_write_task. Get one autonomy task in full: its definition, schedule, dependencies and its most recent run records with exit status.", inputSchema={
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "Task ID to retrieve"}},
            "required": ["id"],
        }),
        Tool(name="autonomy_delete_task", description="Delete an autonomy task, or archive it by setting status back to draft. Archiving is reversible; deletion is not.", inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Task ID to delete"},
                "archive": {"type": "boolean", "description": "If true, set status to draft instead of deleting"},
            },
            "required": ["id"],
        }),
        Tool(name="autonomy_config", description="Read the autonomy config file `_config.md`: one setting with a key, all with none. It never writes; use autonomy_config_set. Keys there are documentary until some code reads them, and nothing does.", inputSchema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Config key to read (empty for all)"},
            },
        }),
        Tool(name="autonomy_config_set", description="Set one key in the autonomy config file `_config.md`: rewrites the front matter, prose below it survives. Keys there are documentary until some code reads them, and nothing does — the reply says whether yours is applied.", inputSchema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Config key to set (nothing applies it)"},
                "value": {"type": "string", "description": "New value, stored verbatim"},
            },
            "required": ["key", "value"],
        }),
        Tool(name="autonomy_run_task", description="Use only to fire an existing autonomy task now; to change what it does use autonomy_write_task. Run an autonomy task immediately, outside its schedule. The run happens in the background; use autonomy_get_task to see the result.", inputSchema={
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "Task ID to run"}},
            "required": ["id"],
        }),
        Tool(name="autonomy_health", description=(
            "Fleet health over the last N days: per-task runs, failure rate, timeouts, "
            "empty runs, [SILENT] rate, GPU-hours, wasted hours, consecutive failures, "
            "plus fleet totals. Use this to find tasks that are burning GPU without "
            "producing anything. It also answers the opposite question: every "
            "per-task row carries `hours_since_last_run` and `gap_ratio` (elapsed "
            "over the task's OWN declared frequency, independent of the window), "
            "and the top-level `stalled` list names every task more than one "
            "period past its own `next_run` whatever its status — so a "
            "`fail_rate: 0.0` no longer stands for 'healthy' about a job that is "
            "not running. Read the window before trusting a clean fleet number: "
            "`window_clamped_to_hours` and `oldest_input` in the `fleet` block "
            "appear "
            "when `workers.db` holds less history than `days` asks for, so a "
            "7-day verdict resting on 21 hours of post-rebuild state says so "
            "rather than implying a week of evidence. And an `idle_tasks` row "
            "is a task with no run in the window — its `fail_rate` and "
            "`silent_rate` are null, never 0.0."), inputSchema={
            "type": "object",
            "properties": {"days": {"type": "integer", "description": "Window in days (default 7, max 90)"}},
        }),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "autonomy_tasks":
        return text_result(_handle_tasks(arguments))
    elif name == "autonomy_write_task":
        return text_result(_handle_write(arguments))
    elif name == "autonomy_get_task":
        return text_result(_handle_get(arguments))
    elif name == "autonomy_delete_task":
        return text_result(_handle_delete(arguments))
    elif name == "autonomy_config":
        return text_result(_handle_config(arguments))
    elif name == "autonomy_config_set":
        return text_result(_handle_config_set(arguments))
    elif name == "autonomy_run_task":
        text = await _handle_run(arguments)
        return text_result(text)
    elif name == "autonomy_health":
        text = await _handle_health(arguments)
        return text_result(text)
    return text_result(json.dumps({"error": f"Unknown tool: {name}"}))


async def _handle_health(params: dict) -> str:
    """Proxy to the backend's /api/autonomy/health.

    agent_mcp runs in its own process, so the work queue singleton isn't
    initialised here — the backend owns it.
    """
    days = int(params.get("days") or 7)
    try:
        from agent_mcp._shared import make_http_client
        from app.config import service_url
        base = service_url("backend", "http://127.0.0.1:8080")
        async with make_http_client(timeout=30.0) as client:
            r = await client.get(f"{base}/api/autonomy/health", params={"days": days})
            if r.status_code >= 400:
                return json.dumps({"error": f"backend returned {r.status_code}",
                                   "body": r.text[:500]})
            return r.text
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _handle_tasks(params: dict) -> str:
    if not AUTONOMY_DIR.exists():
        return json.dumps({"tasks": []})
    status = params.get("status", "")
    frequency = params.get("frequency", "")
    agent_id = params.get("agent_id", "")
    tasks = []
    for path in AUTONOMY_DIR.glob("*.md"):
        # `matched at the start`, so this is `^\d+-`. Until #1692 the only thing
        # this loop excluded by name was the config block, and everything else it
        # accepted on frontmatter alone — so `meta-analysis-2026-06-03.md`, a prose
        # note with legal frontmatter and no `name:` key, reached the agent calling
        # `autonomy_tasks` as a task with `id: 0`, `name: ""` and `status: "draft"`:
        # 33 rows for the 32 task files on disk. The config block parses fine as a
        # task too; it was kept out by its name, not by any rule.
        if not _TASK_NAME_RE.match(path.name):
            continue
        task = _parse_task_file(path)
        if task is None:
            continue
        if status and task.get("status") != status:
            continue
        if frequency and task.get("frequency") != frequency:
            continue
        if agent_id and task.get("agent_id") != agent_id:
            continue
        tasks.append(task)
    return json.dumps({"tasks": tasks})


def _handle_write(params: dict) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    task_id = params.get("id", 0)

    if task_id == 0:
        name = params.get("name", "")
        if not name:
            return json.dumps({"error": "name is required when creating a task"})
        # Ahead of `_next_task_id` so the refusal is the first thing a bad create can do:
        # the id is derived from the highest one on disk, so writing nothing here leaves
        # the same id available, and a stale description never reaches the file at all.
        refusal = _stale_description_error(params.get("description", ""))
        if refusal:
            return refusal
        new_id = _next_task_id()
        task_dict = {
            "id": new_id,
            "name": name,
            "description": params.get("description", ""),
            "status": params.get("status", "") or "draft",
            "priority": params.get("priority", "") or "medium",
            "frequency": params.get("frequency", ""),
            "skill_name": params.get("skill_name", params.get("skill_path", "")),
            "agent_id": params.get("agent_id", "") or "memory",
            "model": params.get("model", ""),
            "timeout_seconds": params.get("timeout_seconds", 0) or 1800,
            "auto_advance": params.get("auto_advance", False),
            "preemptible": params.get("preemptible", True),
            "scheduled_at": params.get("scheduled_at", ""),
            "depends_on": params.get("depends_on") if params.get("depends_on", 0) else None,
            "pipeline": params.get("pipeline", ""),
            "created_at": now,
            "updated_at": now,
            "body": "",
        }
        _write_task_file(task_dict)
        return json.dumps({"task": task_dict})
    else:
        existing_path = _find_task_file(task_id)
        if not existing_path:
            return json.dumps({"error": f"Task #{task_id} not found"})
        task_dict = _parse_task_file(existing_path)
        if task_dict is None:
            return json.dumps({"error": f"Failed to parse task #{task_id}"})
        if task_dict.get("_yaml_broken"):
            # An update here is a read-modify-write of a file we could only
            # regex-recover: it would rewrite the frontmatter from the recovered
            # field list alone and drop every other key, on a task the scheduler
            # is still dispatching. Refuse and leave the file for a human, which
            # is what `agent_mcp/backlog.py::save_task` and the board's own
            # task-write endpoint already do (#1014).
            return _refuse_broken(BrokenFrontmatterError(
                f"task #{task_id} frontmatter only parsed by regex fallback; "
                "refusing to rewrite it (fix the file's YAML first)"))
        prior_status = task_dict.get("status")
        # The same rule on the update path, and on the ARGUMENT only. `task_dict` above
        # is the parsed file, so its `description` is what is already on disk: judging
        # that here would refuse every later write to a file whose prose went stale, and
        # `draft`/`paused` are never dispatched — this branch is how a job gets stopped.
        # The truthiness test mirrors the loop below, so an empty description is ignored
        # here exactly as it is ignored there.
        if params.get("description"):
            refusal = _stale_description_error(params["description"])
            if refusal:
                return refusal
        for key in ("status", "priority", "frequency", "skill_name", "agent_id", "model",
                     "scheduled_at", "pipeline", "description"):
            if params.get(key):
                task_dict[key] = params[key]
        if params.get("timeout_seconds", 0) > 0:
            task_dict["timeout_seconds"] = params["timeout_seconds"]
        if params.get("depends_on", 0) > 0:
            task_dict["depends_on"] = params["depends_on"]
        if "auto_advance" in params:
            task_dict["auto_advance"] = params["auto_advance"]
        if "preemptible" in params:
            task_dict["preemptible"] = params["preemptible"]
        # Arming a task whose measurement surface is an optional LLM slot is refused
        # while that slot is switched off — the writer-side half of the refusal the
        # eval already makes with `EXIT_SLOT_DISABLED = 7` (#1555). #85 went
        # `draft -> up_next` here-equivalent by an unattributed write with
        # `secondary_enabled: false` in force, which is a sweep of one engine twice
        # billed to the primary. Parking is never refused, so a job can always be
        # stopped from this tool.
        from app.autonomy import slot_arm_block
        blocked = slot_arm_block(task_dict, str(task_dict.get("status") or ""))
        if blocked:
            return json.dumps({"error": blocked})
        task_dict["updated_at"] = now
        # Two notes can land on this one write: the caller's own `activity_note`,
        # and the one this tool OWES when it moves `status` — `draft`/`paused` are
        # never dispatched (`app/autonomy.py:323`,
        # `workers/sources/scheduled_task.py:174`), and before #1127 an
        # `autonomy_write_task(status=...)` could park a task with no line naming
        # either value anywhere. Not a block: a status change is a legitimate
        # write, it just now has to say so in the task's own markdown.
        from app.autonomy import append_activity_line, status_change_note
        notes = [n for n in (
            status_change_note(prior_status, task_dict.get("status")),
            params.get("activity_note", ""),
        ) if n]
        if notes:
            body = task_dict.get("body", "")
            for note in notes:
                body = append_activity_line(body, note, now)
            task_dict["body"] = body
        _write_task_file(task_dict)
        return json.dumps({"task": task_dict})


def _handle_get(params: dict) -> str:
    task_id = params.get("id", 0)
    if task_id == 0:
        return json.dumps({"error": "id is required"})
    path = _find_task_file(task_id)
    if not path:
        return json.dumps({"error": f"Task #{task_id} not found"})
    task = _parse_task_file(path)
    if task is None:
        return json.dumps({"error": f"Failed to parse task #{task_id}"})
    runs_dir = AUTONOMY_DIR / "runs" / str(task_id)
    runs = []
    if runs_dir.exists():
        run_files = sorted(runs_dir.glob("*.md"), key=lambda p: p.name, reverse=True)
        for run_path in run_files[:10]:
            run = _parse_run_file(run_path)
            if run:
                runs.append(run)
    runs = sorted(runs, key=lambda r: r.get("started_at", "") or "", reverse=True)
    task["runs"] = runs
    return json.dumps({"task": task})


def _handle_delete(params: dict) -> str:
    task_id = params.get("id", 0)
    if task_id == 0:
        return json.dumps({"error": "id is required"})
    archive = params.get("archive", True)
    path = _find_task_file(task_id)
    if not path:
        return json.dumps({"error": f"Task #{task_id} not found"})
    if archive:
        task = _parse_task_file(path)
        if task:
            if task.get("_yaml_broken"):
                # Archiving is a write, and `draft` is a dispatch kill. Doing it
                # to a file we could only regex-recover would rewrite the
                # frontmatter from the recovered subset and drop every other key
                # — so the recovery path would itself be the clobber (#1014).
                return _refuse_broken(BrokenFrontmatterError(
                    f"task #{task_id} frontmatter only parsed by regex fallback; "
                    "refusing to rewrite it (fix the file's YAML first)"))
            from app.autonomy import append_activity_line, status_change_note
            prior_status = task.get("status")
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            task["status"] = "draft"
            task["updated_at"] = now
            # Archive IS a dispatch kill: `draft` is outside the runnable set, so
            # the third non-scheduler writer names the transition too (#1127
            # clause 6) — an archived task with no line saying so is indistinguishable
            # from one something clobbered. Already-`draft` tasks get nothing: the
            # status did not change.
            note = status_change_note(prior_status, "draft")
            if note:
                task["body"] = append_activity_line(task.get("body", ""), note, now)
            _write_task_file(task)
            return json.dumps({"success": True, "id": task_id})
        return json.dumps({"error": f"Failed to parse task #{task_id}"})
    else:
        path.unlink()
        return json.dumps({"success": True, "id": task_id})


#: #1326: the write left `autonomy_config`, whose name is a read and whose entry
#: in `agent_mcp.annotations.READ_ONLY` is the predicate behind four refusals
#: (plan mode, the bench/eval sandbox, a sessionless `call_tool`, and
#: `MCPPool._retry_safe` re-sending the call after a transport drop). A caller
#: that still sends `value` is pointed at the verb that does it.
_CONFIG_WRITE_MOVED = (
    "autonomy_config no longer writes. The write half is "
    "autonomy_config_set(key=..., value=...); this tool reads the whole config "
    "with no key, or one setting with a key alone.")


def _handle_config(params: dict) -> str:
    key = params.get("key", "")
    value = params.get("value", None)
    if value is not None:
        return json.dumps({"error": _CONFIG_WRITE_MOVED,
                           "use_tool": "autonomy_config_set"})
    if not key:
        return json.dumps(_read_config())
    else:
        config = _read_config()
        if key in config:
            return json.dumps({key: config[key]})
        return json.dumps({"error": f"Config key not found: {key}"})


def _handle_config_set(params: dict) -> str:
    """`autonomy_config_set`: one key into `_config.md`'s front matter, nothing else.

    A writer, deliberately — see `_CONFIG_WRITE_MOVED`. It rewrites the front
    matter and puts back the bytes below the closing fence exactly as it read
    them, and it refuses a file whose shape it cannot round-trip rather than
    re-emitting a fallback parse over it (the clobber rule `_write_task_file`
    applies to task files, which is #1014).
    """
    key = params.get("key", "")
    value = params.get("value", "")
    if not key:
        return json.dumps({"error": "key is required"})
    config_path = AUTONOMY_DIR / "_config.md"
    try:
        raw = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    except OSError as exc:
        return json.dumps({"error": f"cannot read {config_path}: {exc}"})
    split = _split_config(raw) if raw else ("", {}, "")
    if split is None:
        return json.dumps({
            "error": f"{config_path} has no parseable front matter to update "
                     "(no closing `---` fence, prose above the opening one, or a "
                     "front matter that is not a mapping); refusing to rewrite it",
            "yaml_broken": True})
    _frontmatter_text, config, tail = split
    config[key] = value
    try:
        _write_config(config, tail)
    except (OSError, yaml.YAMLError) as exc:
        return json.dumps({"error": f"could not rewrite {config_path}: {exc}",
                           "yaml_broken": True})
    applied, effect_note = _config_key_effect(key)
    return json.dumps({"set": key, "value": value, "path": str(config_path),
                       "bytes_below_front_matter": len(tail.encode("utf-8")),
                       # #1672: the file was named and the call returned, which a
                       # caller reads as a knob that took. The write did happen;
                       # whether it does anything is a separate fact, and it
                       # belongs in the same object rather than in a description
                       # read before the call.
                       "applied_by_code": applied,
                       "effect": effect_note})


async def _handle_run(params: dict) -> str:
    task_id = params.get("id", 0)
    if task_id == 0:
        return json.dumps({"error": "id is required"})
    try:
        from app.paths import LLOYD_HOME as lloyd_home
        if str(lloyd_home) not in sys.path:
            sys.path.insert(0, str(lloyd_home))
        from app.autonomy import run_task, run_trigger
        with run_trigger("mcp"):
            result = await run_task(task_id)
        return json.dumps(result)
    except ImportError as e:
        return json.dumps({"error": f"autonomy scheduler module not available: {e}"})
    except Exception as exc:
        return json.dumps({"error": str(exc), "task_id": task_id})

