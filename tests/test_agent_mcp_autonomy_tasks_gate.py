"""The `autonomy_tasks` MCP listing gates the autonomy directory on ONE name rule — #1692.

`agent_mcp/autonomy.py::_handle_tasks` is the payload behind the `autonomy_tasks`
tool, so its count is the number an agent quotes when it asks how many scheduled
jobs exist. Until #1692 it walked `AUTONOMY_DIR.glob("*.md")`, excluded exactly one
filename by literal, and accepted everything `_parse_task_file` returned — which is
a record for any file with frontmatter, whatever it is. Measured on the live vault
before the change (the probe in this item's acceptance check):

    33 tasks listed against 32 `NN-*.md` task files, `[0]` blank-name entries.

The extra row was `meta-analysis-2026-06-03.md` — a prose note carrying
`segment: autonomy`, `type: autonomy` and a timestamp, and no `id:`, `name:` or
`status:` key — which `_parse_task_file` projects to
`{id: 0, name: "", status: "draft", type: "autonomy"}` over an 8875-character
report body. `_config.md` is a second such file: it parses as a task and was kept
out only by the name literal, so the next note anyone writes into that directory
walks straight through.

The fix reuses the predicate #1594 introduced — `app/routers/autonomy.py::_TASK_NAME_RE`
(`re.compile(r"\\d+-")`, applied with `.match`, so `^\\d+-`) — by importing it. The
sibling file `tests/test_mc_summarize_autonomy_gate.py` (#1594) pins the same rule on
the Mission Control tab; this one pins it on the agent-facing tool, which is the
reader that was never enumerated.

Every test here writes a scratch autonomy directory and patches
`agent_mcp.autonomy.AUTONOMY_DIR` ONCE. `_handle_tasks` reads that module global, so
without the patch the assertions would be measuring the live vault — and a test that
measures the live vault passes or fails on what someone filed that day, not on the
code under test.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent_mcp.autonomy as MCP  # noqa: E402
from app.routers import autonomy as ROUTER  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

#: A task-shaped file: `NN-slug.md`, legal frontmatter, a status a task can carry.
TASK_42 = """---
id: 42
name: Morning brief triage
status: up_next
frequency: daily
---

# body
"""

#: A report-shaped file whose frontmatter is *legal* and whose `status:` is a real
#: status — the frontmatter the vault-maintenance "fix missing frontmatter" step
#: would add. To the pre-#1692 gate it was indistinguishable from a task.
REPORT_WITH_LEGAL_FRONTMATTER = """---
name: Skill Lint Report
status: up_next
tags:
- autonomy
- skill-lint
segment: autonomy
type: note
timestamp: '2026-09-27T00:42:52'
---

# Skill Lint Report — 2026-09-27T00:42:52
"""

#: Shaped from the file that produced the live phantom, `meta-analysis-2026-06-03.md`
#: read through `_parse_task_file`: frontmatter that parses, and no `id:`, `name:` or
#: `status:` key. Those three defaults are what turns the note into
#: `{id: 0, name: "", status: "draft"}`.
REPORT_SHAPED_LIKE_THE_PHANTOM = """---
tags:
- autonomy
- meta-analysis
segment: autonomy
type: autonomy
timestamp: '2026-07-06T14:36:37'
---
# Autonomy System Meta-Analysis & Remediation — 2026-06-03

Review of all autonomy tasks plus a system-wide meta-analysis.
"""

#: The config block: frontmatter with a settings map. `agent_mcp.autonomy` reads it
#: through `_read_config`, and it parses as a task too.
CONFIG_WITH_LEGAL_FRONTMATTER = """---
segment: autonomy
type: note
---

_settings:
  max_parallel: 3
"""

#: Frontmatter opened and never closed. `_parse_task_file` splits on `"---\\n"` and
#: bails on `len(parts) < 3`, so this is the case the listing legitimately drops.
TASK_WITH_UNCLOSED_FENCE = """---
id: 44
name: Unclosed fence
status: up_next
"""

#: Frontmatter that closes but is not valid YAML (the indented line breaks the
#: mapping, and the flow sequence it opens never closes). This one is NOT dropped:
#: the graduated recovery from #1014 extracts the fields by regex and flags the
#: record `_yaml_broken: True`, because a task the scheduler still dispatches must
#: stay visible to the reader a human asks "why has this job been silent". The
#: fallback extracts scalars as written, so its `id` arrives as the string "45".
TASK_WITH_MALFORMED_YAML = """---
id: 45
name: Malformed but closed
 bad_indent: [unclosed
status: up_next
---

# body
"""

#: The projection `_handle_tasks` emits, as of #1692. Clause 4 is a no-shape-change
#: clause, so the shape is pinned literally: a key added or dropped from the
#: projection has to be a deliberate edit to this list, not a side effect of
#: whatever the next round was doing to the listing.
TASK_DICT_KEYS = [
    "_yaml_broken", "agent_id", "auto_advance", "body", "created_at", "cron_id",
    "depends_on", "description", "expected_error_patterns", "failure_count",
    "frequency", "id", "infra_failure_count", "infra_rest_until", "inner_voice",
    "last_attempt", "last_run", "max_retries", "model", "name", "next_run",
    "notify_on_complete", "pipeline", "preemptible", "preferred_hours", "priority",
    "requires_slot", "run_count", "runs_per_day", "scheduled_at", "skill_name",
    "stale_bypass_hours", "status", "timeout_seconds", "type", "updated_at",
]


@pytest.fixture
def autonomy_tree(tmp_path, monkeypatch):
    """An empty autonomy directory at the path `_handle_tasks` reads; writer-shaped."""
    dirn = tmp_path / "autonomy"
    dirn.mkdir()
    monkeypatch.setattr(MCP, "AUTONOMY_DIR", dirn)
    return dirn


def _write(dirn: Path, **files: str) -> Path:
    """Write `{"42-task.md": text}` into the patched directory — a dict because
    hyphenated filenames cannot be keyword arguments."""
    for name, text in files.items():
        (dirn / name).write_text(text, encoding="utf-8")
    return dirn


def _tasks_via_mcp_seam() -> list[dict]:
    """The listing across the seam a caller actually crosses.

    `agent_mcp.autonomy.call_tool` is what `agent_mcp.main` routes an MCP tool call
    into, so this is the `autonomy_tasks` tool's answer and not just the function's.
    """
    result = asyncio.run(MCP.call_tool("autonomy_tasks", {}))
    assert not getattr(result, "isError", False), result
    return json.loads(result.content[0].text)["tasks"]


def _blank_name_ids(tasks: list[dict]) -> list:
    """The filter from the live acceptance probe: ids of entries with no usable name."""
    return [t["id"] for t in tasks if not str(t.get("name", "")).strip()]


def test_unnumbered_files_with_legal_frontmatter_are_not_listed(autonomy_tree):
    """Clause 1: the name decides the listing, not the frontmatter.

    `report.md` and `_config.md` are each a file the parser turns into a record with
    a real status — clause 2 below shows that directly for the phantom shape — so
    the only thing that can exclude them is the gate. `42-task.md` is the one entry
    that survives.
    """
    _write(autonomy_tree, **{
        "42-task.md": TASK_42,
        "report.md": REPORT_WITH_LEGAL_FRONTMATTER,
        "_config.md": CONFIG_WITH_LEGAL_FRONTMATTER,
    })

    tasks = _tasks_via_mcp_seam()

    assert [t["id"] for t in tasks] == [42], (
        f"a file whose name is not `NN-slug.md` was listed as a task: "
        f"{[(t['id'], t['name']) for t in tasks]}"
    )
    payload = json.dumps(tasks)
    assert "Skill Lint Report" not in payload, "report.md was listed as a task"
    assert "max_parallel" not in payload, "the config block was listed as a task"


def test_the_phantom_row_and_its_blank_name_are_gone(autonomy_tree):
    """Clause 2: the count-and-blank-name filter the live probe runs.

    One numbered task plus one frontmattted report — the live tree's shape at a
    32nd-of-the-size. The pre-fix answer was `(2, [0])`: the report counted, and its
    missing `name:` came back as the entry with no name to quote. The live tree
    measured `(33, [0])` for the same reason against 32 task files.

    The second half is what keeps this test from passing for the wrong reason:
    `_parse_task_file` must STILL return a task-shaped record for that report. If it
    ever returns None the exclusion would be the parser's doing and the name gate
    would be unobserved, which is the difference between a gate and a coincidence.
    """
    _write(autonomy_tree, **{
        "42-task.md": TASK_42,
        "meta-analysis-2026-06-03.md": REPORT_SHAPED_LIKE_THE_PHANTOM,
    })

    tasks = _tasks_via_mcp_seam()

    assert (len(tasks), _blank_name_ids(tasks)) == (1, []), (
        f"the listing carries a blank-name phantom: "
        f"{[(t['id'], t['name'], t['status']) for t in tasks]}"
    )
    phantom = MCP._parse_task_file(autonomy_tree / "meta-analysis-2026-06-03.md")
    assert phantom is not None, (
        "the parser stopped producing a record for the report, so the assertion "
        "above no longer exercises the name gate at all"
    )
    assert (phantom["id"], phantom["name"], phantom["status"]) == (0, "", "draft"), (
        "the report no longer projects to the phantom row, so the fixture no longer "
        "reproduces what the live probe measured"
    )


def test_the_gate_is_the_shared_predicate_and_not_a_name_literal():
    """Clause 3: one rule, imported, and applied in the loop that lists.

    Identity, not equality: `is` proves `agent_mcp` is reading the same compiled
    pattern the route compiles, so a change to either reader's rule is now one edit.
    The source assertions are the behavioural half — an imported constant nobody
    uses would satisfy `is` and leave the phantom exactly where it was.
    """
    assert MCP._TASK_NAME_RE is ROUTER._TASK_NAME_RE, (
        "agent_mcp carries its own copy of the task-name rule instead of importing "
        "`app.routers.autonomy._TASK_NAME_RE`"
    )

    source = inspect.getsource(MCP._handle_tasks)
    assert "_config.md" not in source, (
        "`_handle_tasks` still excludes a filename by literal; the exclusion is the "
        "shared predicate's job now"
    )
    assert "_TASK_NAME_RE" in source, (
        "`_handle_tasks` no longer applies the shared predicate to the glob it "
        "walks, so the import is decoration"
    )


def test_a_numbered_file_that_cannot_be_parsed_is_dropped_and_a_valid_one_keeps_its_shape(
        autonomy_tree):
    """Clause 4: exclusions only — the drop rule and the row shape are unchanged.

    "Cannot be parsed" is the unclosed-fence case: `_parse_task_file` splits on
    `"---\\n"` and returns None under `len(parts) < 3`, so `44-unclosed.md` is not a
    row. A *closed* but YAML-invalid block is a different case and stays listed with
    `_yaml_broken: True` — the graduated recovery of #1014, pinned here so nobody
    "fixes" the drop into swallowing the files that recovery exists to surface.

    The valid row is compared key-for-key against `TASK_DICT_KEYS`: this change is
    supposed to remove entries from the listing and nothing else, and a projection
    that quietly gained or lost a field would otherwise ride along with it.
    """
    _write(autonomy_tree, **{
        "42-task.md": TASK_42,
        "44-unclosed.md": TASK_WITH_UNCLOSED_FENCE,
        "45-malformed.md": TASK_WITH_MALFORMED_YAML,
    })

    tasks = _tasks_via_mcp_seam()

    assert MCP._parse_task_file(autonomy_tree / "44-unclosed.md") is None, (
        "an unclosed fence now parses to something, so the exclusion above is not "
        "the drop the clause pins"
    )
    # Keyed by name, not id: the regex fallback hands back scalars as written, so
    # the recovered record's `id` is the string "45" and its sort key would not
    # compare against the integer 42 the YAML path produces.
    by_name = {t["name"]: t for t in tasks}
    assert sorted(by_name) == ["Malformed but closed", "Morning brief triage"], (
        "the drop rule or the recovery rule moved: "
        f"{[(t['name'], t['_yaml_broken']) for t in tasks]}"
    )

    recovered = by_name["Malformed but closed"]
    assert recovered["_yaml_broken"] is True
    assert str(recovered["id"]) == "45"

    row = by_name["Morning brief triage"]
    assert sorted(row) == TASK_DICT_KEYS, (
        f"the listing's row shape changed beyond dropping rows: "
        f"added {sorted(set(row) - set(TASK_DICT_KEYS))}, "
        f"dropped {sorted(set(TASK_DICT_KEYS) - set(row))}"
    )
    assert (row["id"], row["name"], row["status"], row["frequency"]) == (
        42, "Morning brief triage", "up_next", "daily")
    assert (row["type"], row["_yaml_broken"]) == ("autonomy", False)


def test_the_architecture_sentence_about_the_name_rule_names_its_exceptions():
    """Clause 5: `architecture/autonomy.md` may no longer claim the rule is universal.

    The paragraph that opens "Every one of them skips a file whose name does not
    start with a digit" was false of the row directly above it in the reader table —
    `agent_mcp/autonomy.py` — and is still false of `app/routers/dashboard.py`, which
    gates on `path.name[:1].isdigit()` and so accepts `9x-notes.md`, a name the
    shared regex rejects because it also requires the hyphen after the digits. The
    sentence survives only as a claim with those exceptions named next to it.
    """
    doc = (REPO / "architecture" / "autonomy.md").read_text(encoding="utf-8")
    paragraphs = [p for p in doc.split("\n\n")
                  if "does not start with a digit" in p]

    assert paragraphs, (
        "the discussion of the name rule has been deleted rather than qualified; "
        "the sentence is the only place the divergences are written down"
    )
    for para in paragraphs:
        assert "Every one of them skips" not in para, (
            "the unqualified universal claim is still standing: it is false of the "
            "dashboard's gate and was false of agent_mcp until #1692"
        )
        assert "isdigit" in para, (
            "the paragraph does not name `app/routers/dashboard.py`'s "
            "`path.name[:1].isdigit()`, the reader that still counts a name the "
            "shared regex rejects"
        )
        assert "hyphen" in para, (
            "the paragraph does not say the shared regex requires a hyphen after "
            "the digits, which is what makes it stricter than a leading digit"
        )
        assert "_TASK_NAME_RE" in para, (
            "the paragraph does not name the one pattern the rule lives in"
        )


#: A finding-set `lint()` result, shaped like the one
#: `tests/test_skill_lint_report_frontmatter.py` uses, so the bytes this node puts on
#: disk come from the writer under test and not from a hand-typed fixture.
SKILL_LINT_RESULT = {
    "generated_at": "2026-09-29T01:45:33",
    "total": 1,
    "dead": [], "missing_desc": [], "drift": [], "duplicates": [],
    "stale": [], "phantom": [],
    "missing_script": [{"name": "demo-skill", "path": "/x/SKILL.md",
                        "scripts": [{"path": "scripts/demo.py", "known_stale": ""}]}],
}


def _run_skill_lint_into(dirn: Path, monkeypatch) -> Path:
    """`skill_lint.main()` with `REPORT_PATH` redirected into `dirn`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "skill_lint", REPO / "scripts" / "skill_lint.py")
    skill_lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill_lint)
    report = dirn / "skill-lint-report.md"
    monkeypatch.setattr(skill_lint, "REPORT_PATH", report)
    monkeypatch.setattr(skill_lint, "lint", lambda: SKILL_LINT_RESULT)
    assert skill_lint.main() == 0
    return report


def test_the_header_skill_lint_now_emits_does_not_make_the_report_a_task(
        autonomy_tree, monkeypatch):
    """#1826 clause 5: the report gains frontmatter and still is not a task.

    `scripts/skill_lint.py` starts writing a `type: note` block into
    `skill-lint-report.md` so `validate_okf.py` stops listing it. That is precisely
    the file shape this gate exists to survive — the ones above use a hand-typed
    `REPORT_WITH_LEGAL_FRONTMATTER`, and a hand-typed fixture freezes yesterday's
    writer. So the bytes here are the ones `main()` really writes, produced by
    running it with `REPORT_PATH` pointed at the scratch autonomy directory: the
    writer and the listing are both live, and the assertion is what the listing does
    with the real header.

    `_parse_task_file` must STILL return a record for that file (no `id:`, `name:` or
    `status:` key, so it projects to the `{id: 0, name: ""}` phantom row), which is
    what makes `[42]` a verdict about the name gate rather than about the parser
    giving up. `app/routers/dashboard.py`, the third reader of this directory, gates
    on `path.name[:1].isdigit()` and rejects the name for the same reason.
    """
    _write(autonomy_tree, **{"42-task.md": TASK_42})
    report = _run_skill_lint_into(autonomy_tree, monkeypatch)

    assert report.is_file() and report.read_text(encoding="utf-8").startswith("---\n"), (
        "the writer stopped emitting a header, so this node would pass on a report "
        "that never had one to be confused by"
    )

    tasks = _tasks_via_mcp_seam()
    assert [t["id"] for t in tasks] == [42], (
        f"the skill-lint report joined the task listing: "
        f"{[(t['id'], t['name']) for t in tasks]}"
    )
    assert (len(tasks), _blank_name_ids(tasks)) == (1, [])
    assert "Skill Lint Report" not in json.dumps(tasks)

    phantom = MCP._parse_task_file(report)
    assert phantom is not None, (
        "the parser stopped producing a record for the frontmattted report, so the "
        "assertions above are measuring the parser and not the name gate"
    )
    assert (phantom["id"], phantom["name"], phantom["status"]) == (0, "", "draft"), (
        "the emitted report no longer projects to the phantom row, so the fixture no "
        "longer reproduces what the live probe measures"
    )


def test_the_architecture_paragraph_about_alerts_names_where_they_land():
    """#2101: `architecture/autonomy.md` said alerts and completion notices "go to
    Discord" and that the guardian's fan-out "does not cover it". Discord is
    unconfigured on this box by decision, so that named a destination nothing reaches
    and hid the two branches an implementer needs: an alert falls back to the daily
    note, a completion notice is dropped. The paragraph is read here the same way the
    name-rule one is above, so a rewrite that loses a branch turns this red.
    """
    doc = (REPO / "architecture" / "autonomy.md").read_text(encoding="utf-8")
    start = doc.index("Alerts and completion notices are sent through")
    section = " ".join(doc[start:doc.index("## Fleet health", start)].split())

    assert "go to Discord" not in section, "the unqualified destination is back"
    assert "daily note" in section and "home_channel: null" in section
    assert "tests/test_autonomy_failure_alert.py" in section, "the tripwire is not named"
    # The two branches, kept distinct.
    assert "_survive_the_dropped_alert" in section and "append_daily_alert_line" in section
    assert "_discord_notify_task_complete" in section and "no fallback" in section
    # The read-back contract and its after-the-return witness.
    assert "readable back" in section and "not that the alarm can no longer be lost" in section
    assert "daily-note-appends.jsonl" in section and "daily_note_appends" in section
    # The guardian sentence, only in the form that is true today.
    assert "does not carry autonomy's alerts" in section
    assert "agent-services/guardian/daily_note.py" in section and "app/daily_note.py" in section

    # And each thing it names exists where it says.
    notify = (REPO / "app" / "discord_notify.py").read_text(encoding="utf-8")
    assert "def _survive_the_dropped_alert" in notify and "append_daily_alert_line" in notify
    assert "daily-note-appends.jsonl" in (REPO / "app" / "autonomy.py").read_text(encoding="utf-8")
    assert (REPO / "agent-services" / "guardian" / "daily_note.py").exists()
    assert (REPO / "app" / "daily_note.py").exists()


# ── #2408: a description's quoted constant is checked before the write, not at land ──

#: The constant these nodes quote, read live rather than hard-coded so the numbers in
#: this section cannot rot. `MAX_BODY_LINES` is `scripts/skill_lint.py:816`'s ceiling on
#: a SKILL.md body — the number `scripts/automod/vault_round.py` refuses a vault land
#: against, and the kind of figure an autonomy task description legitimately states.
STALE_PROBE_CONSTANT = "MAX_BODY_LINES"


def _tree_value() -> int:
    """The live value of `STALE_PROBE_CONSTANT`, with the control this section needs.

    `constant_quotes.mismatches()` resolves zero pairs for a name the tree does not
    carry exactly as it does for a quote that is correct, and zero pairs produces no
    refusal for either reason. A section that only asserted "no refusal" would
    therefore pass on a deleted or ambiguous constant, so every node here calls this
    first and an absent name is an instrument failure, loudly.
    """
    from scripts.automod import constant_quotes as CQ

    value = CQ.tree_constants(REPO).get(STALE_PROBE_CONSTANT)
    assert value is not None, (
        f"{STALE_PROBE_CONSTANT} is no longer a single-valued integer constant in this "
        "tree, so the quotes below resolve nothing and the assertions here would pass "
        "without the guard ever being consulted"
    )
    return int(value)


def _quote(value: int) -> str:
    """A task description stating `STALE_PROBE_CONSTANT` as `value`.

    `NAME (n)` is the spelling the resolver binds (`constant_quotes._AFTER_NAME_RX`),
    and the sentence carries no other ALL-CAPS token, so exactly one pair resolves —
    which is what makes a refusal here a verdict on the number rather than on prose
    shape.
    """
    return f"Reconcile the ledger window against {STALE_PROBE_CONSTANT} ({value})."


def _write_via_mcp(params: dict) -> dict:
    """`autonomy_write_task` across the seam a caller crosses, parsed.

    Same reason `_tasks_via_mcp_seam` exists above: `call_tool` is what
    `agent_mcp.main` routes into, so this is the refusal a caller actually receives.
    """
    result = asyncio.run(MCP.call_tool("autonomy_write_task", params))
    return json.loads(result.content[0].text)


def test_a_description_quoting_a_stale_constant_is_refused_and_nothing_is_written(
        autonomy_tree):
    """Clause 1: the create path refuses, names both numbers, and writes no file.

    #2317 put the quoted-number check on the vault-land route
    (`scripts/automod/vault_round.py:630`), which is where a human lands a description
    — but the description of an autonomy task is written most often by an agent
    through this tool, and `git log -S'constant_quotes' -- agent_mcp/` is empty at the
    base of this round: no commit had ever put the guard on that route. The failure it
    is fixed against is #2317's own: `01dea8bc` moved `LEDGER_ARCHIVE_AGE_DAYS` from 30
    to 14 and `autonomy/79-retention-sweep.md` went on saying 30 for three weeks — the
    drift #1573 and #1734 had each already repaired once by hand on that one file.
    """
    value = _tree_value()

    out = _write_via_mcp({"name": "Ledger reconciliation",
                          "description": _quote(value + 7),
                          "frequency": "daily"})

    assert "error" in out, (
        f"a description quoting {STALE_PROBE_CONSTANT} at {value + 7} was published "
        f"into a tree that says {value}: {out}"
    )
    assert list(autonomy_tree.glob("*.md")) == [], (
        "the tool refused and wrote the task file anyway, so the refusal is a message "
        "and not a refusal")

    err = out["error"]
    assert STALE_PROBE_CONSTANT in err, f"the refusal does not name the constant: {err}"
    assert str(value + 7) in err, f"the refusal does not name the quoted number: {err}"
    assert str(value) in err, f"the refusal does not name the tree's number: {err}"


def test_the_corrected_retry_writes_and_so_does_a_description_naming_no_constant(
        autonomy_tree):
    """Clause 2: the same call succeeds once the number is the tree's, and ordinary
    descriptions are untouched.

    The retry half is what makes this a guard rather than a veto — the caller fixes it
    by typing one digit, in the same call it was making. The no-constant half is the
    cost side: most task descriptions state no constant at all, and a write-path
    refusal that fired on those would be the engine's own nightly writers blocked by
    an instrument that resolved zero pairs.
    """
    value = _tree_value()

    stale = _write_via_mcp({"name": "Ledger reconciliation",
                            "description": _quote(value + 7)})
    assert "error" in stale, "the control half: the stale quote stopped refusing"

    fixed = _write_via_mcp({"name": "Ledger reconciliation",
                            "description": _quote(value)})
    assert "error" not in fixed, f"the corrected retry was refused: {fixed}"
    on_disk = sorted(autonomy_tree.glob("*.md"))
    assert len(on_disk) == 1, (
        f"the corrected retry wrote {len(on_disk)} files instead of one task")
    assert _quote(value) in on_disk[0].read_text(encoding="utf-8"), (
        "the retry wrote a file whose description is not the one that was passed")

    plain = _write_via_mcp({"name": "Queue sweep",
                            "description": "Sweep the queue nightly and log the count."})
    assert "error" not in plain, f"a description naming no constant was refused: {plain}"
    assert len(list(autonomy_tree.glob("*.md"))) == 2, (
        "the description-free create did not land beside the first one")


def test_a_status_only_update_on_a_file_whose_description_is_still_stale_writes(
        autonomy_tree):
    """Clause 3: the refusal sits on the delta, driven through the UPDATE path.

    This is the clause that decides where the guard can live. `_handle_write`'s update
    branch parses the existing file and hands `_write_task_file` the whole record,
    description included, so a check inside the writer would refuse every engine state
    write against a file whose prose is already stale — which is the routine
    `draft`/`paused` parking that must never be blocked. Here the file on disk carries
    the stale quote, so any check that reads the file rather than the call is refusing
    the park.

    The last block is the control that keeps this node from passing by being unhooked:
    the SAME call shape on the SAME file, with the stale description *supplied*, must
    still refuse. Without it, "status writes work" would also be satisfied by a guard
    that never fires at all.
    """
    value = _tree_value()
    stale_on_disk = f"""---
id: 42
name: Ledger reconciliation
status: up_next
frequency: daily
description: Reconcile the ledger window against {STALE_PROBE_CONSTANT} ({value + 7}).
---

# body

## Activity Log

- 2026-10-01T00:00:00Z: run completed
"""
    _write(autonomy_tree, **{"42-ledger.md": stale_on_disk})

    parked = _write_via_mcp({"id": 42, "status": "draft"})
    assert "error" not in parked, (
        f"parking a task was refused because of prose the call never touched: {parked}")
    text = (autonomy_tree / "42-ledger.md").read_text(encoding="utf-8")
    assert "status: draft" in text, "the park did not reach disk"
    assert f"{STALE_PROBE_CONSTANT} ({value + 7})" in text, (
        "the write silently corrected the stale description, so this node is no "
        "longer testing a stale file")

    noted = _write_via_mcp({"id": 42, "activity_note": "checked the queue"})
    assert "error" not in noted, f"an Activity Log line was refused: {noted}"
    assert "checked the queue" in (autonomy_tree / "42-ledger.md").read_text(
        encoding="utf-8")

    supplied = _write_via_mcp({"id": 42, "description": _quote(value + 7)})
    assert "error" in supplied, (
        "the update path stopped refusing a supplied stale description, so the two "
        "assertions above are measuring a guard that is not connected")
    assert "status: draft" in (autonomy_tree / "42-ledger.md").read_text(
        encoding="utf-8"), "the refused update changed the file anyway"


def test_the_constant_refusal_is_plain_error_json_and_not_the_broken_yaml_one(
        autonomy_tree):
    """Clause 4: one key, `error`, and it is returned rather than raised.

    `_refuse_broken` (`agent_mcp/autonomy.py:163`) is the other refusal this module
    can return, and its `yaml_broken: True` key means exactly one thing on this box —
    front matter that only parsed by the regex fallback (`app/routers/autonomy.py`
    turns that key into the Mission Control warning) — so reusing it here would send a
    caller off to fix YAML that is fine. The broken half of the pin is the sibling
    record below, which must STILL carry the marker: the two shapes are only
    distinguishable if both keep their own key set.

    "Never raised" is checked by the `call_tool` drive: an exception would leave
    `asyncio.run` and error this node with a traceback instead of a refusal, which is
    the failure mode `_refuse_broken` exists to avoid for the dispatcher.
    """
    value = _tree_value()
    stale = {"name": "Ledger reconciliation", "description": _quote(value + 7)}

    raw = MCP._handle_write(dict(stale))
    assert isinstance(raw, str), (
        f"the refusal is a {type(raw).__name__}, not the module's error-JSON string")
    parsed = json.loads(raw)
    assert set(parsed) == {"error"}, (
        f"the constant refusal carries keys beyond `error`: {sorted(parsed)} — "
        "`yaml_broken` on this box means broken front matter and nothing else")
    assert "yaml_broken" not in raw

    across_seam = _write_via_mcp(stale)
    assert "error" in across_seam, "the refusal did not survive call_tool"

    _write(autonomy_tree, **{"45-malformed.md": TASK_WITH_MALFORMED_YAML})
    broken = json.loads(MCP._handle_write({"id": 45, "status": "draft"}))
    assert broken.get("yaml_broken") is True, (
        "the broken-frontmatter refusal lost its marker, so a caller can no longer "
        "tell the two refusals apart: " + json.dumps(broken))


@pytest.mark.parametrize("boom", [OSError("no git here"), ValueError("bad number")],
                         ids=["oserror", "valueerror"])
def test_a_resolver_that_cannot_answer_still_lets_the_write_through(autonomy_tree,
                                                                   monkeypatch, boom):
    """The guard's own half of "never raised": a broken instrument must not stop writes.

    `constant_quotes.Report.resolved` exists because "0 mismatches" and "nothing to
    compare" look identical, and the vault-side rail logs its count for the `@live_vault`
    witness in `tests/test_constant_quotes.py`. This route records nothing when it has
    nothing to say, so the only thing standing between a failing harvest and a stopped
    engine is the `except` tuple in `_stale_description_error`. Both classes named here
    are ones that tuple has to answer for — walking the tree, and converting a number —
    and `ValueError` is the one a guard on a write path can least afford to let escape,
    because the write it would block may be the `status: draft` that stops a job.
    """
    import scripts.automod.constant_quotes as CQ

    # Read the number first: `_tree_value` harvests the tree itself, so patching before it
    # would have this node fail in its own fixture rather than in the guard.
    value = _tree_value()

    def _raise(*a, **k):
        raise boom

    monkeypatch.setattr(CQ, "tree_constants", _raise)

    out = _write_via_mcp({"name": "Ledger reconciliation", "description": _quote(value + 7)})
    assert "error" not in out, (
        f"with the resolver raising {type(boom).__name__} the write was refused, so a "
        "failing instrument has become an outage for task writes: " + json.dumps(out))
    assert len(list(autonomy_tree.glob("*.md"))) == 1, (
        "the write claimed success without writing the file")


def test_a_malformed_call_leaves_the_next_write_working(autonomy_tree):
    """A refused or malformed call cannot poison the writes that follow it.

    This module never decodes a caller's JSON — `grep -n json.loads
    agent_mcp/autonomy.py` finds only this node's prose and the helper's — so an argument
    payload that was never parsed reaches the create path as a dict with no `name`, which
    is an error answer rather than an exception. It is pinned because the guard now sits
    on that same first branch: whatever the branch returns for a call with no name, the
    next well-formed create has to succeed, and a refusal that left state behind would
    show up here rather than in a nightly run three days later.
    """
    value = _tree_value()

    junk = _write_via_mcp({"args": "not-a-json-object"})
    assert "error" in junk, (
        f"a call with no `name` returned {json.dumps(junk)}; the create path is expected "
        "to answer with an error, not to invent a task")
    assert list(autonomy_tree.glob("*.md")) == [], (
        "the nameless call wrote a task file, so a malformed request is a write")

    good = _write_via_mcp({"name": "Ledger reconciliation", "description": _quote(value)})
    assert "error" not in good, (
        f"the write after a malformed call was refused: {json.dumps(good)}")
    assert len(list(autonomy_tree.glob("*.md"))) == 1, (
        "the write after a malformed call did not land")

