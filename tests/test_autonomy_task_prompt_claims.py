"""An autonomy task may only instruct tools its rendered prompt delivers (#463).

`_build_task_prompt` (`app/autonomy.py:674`, sole caller `:968`) builds a worker's
prompt from the task's SKILL.md and its front-matter `description`, and never
from the markdown **body**. That makes the body a human-facing document that
looks like a specification: the `3. Fact Store Consolidation` phase in
`~/obsidian/autonomy/47-dream-consolidation.md` instructed per-entity
contradiction detection and auto-resolution over the knowledge graph, and no
#47 run had ever received it — `grep -l "Fact Store Consolidation"
~/lloyd/autonomy-runs/47/*.md` returned 0 of 4 records. The vault half of this
item deletes it and leaves exactly one named owner for contradiction resolution
over the graph (task #84 / `~/obsidian/skills/fact-improvement/SKILL.md`); the
last paragraph says how to tell whether that commit has landed.

The fleet-wide shape of the seam — every body-only instruction in all 32 task
files is dead text — is owned by `~/obsidian/backlog/951-*.md`. What this file
pins is the #47 instance and the seam's delivery property, so that neither can
quietly come back.

Seven of the fourteen tests read the live `~/obsidian` and carry `@live_vault`, which
the gate deselects (`pytest.ini:6-13`). The record, measured 2026-09-13: EVERY
automated pytest invocation in the tree passes `-m "not live_vault"`
(`gate.py:290`, `gate.py:918`, and the review rung's sandbox command at
`review.py:573`), so no automated run executes those six — they fire for a person,
or for a review that writes its own command. That gap is not this item's to fix and
is already owned, with a measured case list, by **backlog #979**
(`~/obsidian/backlog/979-live-vault-tests-run-on-no-automated-rung-so-a-pin.md`);
until it closes, read the seven live nodes as pins a human or a review executes, and
read the seven vault-free nodes below as the part the loop enforces by itself: the
extractor self-check, the `_build_task_prompt` mechanism test, the end-to-end
`run_task` delivery test, the two #1633 long-skill delivery nodes, the `CLAUSE_NODES`
self-check and the mark-scope self-check. (The round branch also
carried a clause-5 diff test and its surface-filter self-check; both read the
round's own `git diff` and were dropped when the file was landed by hand, #1413.
`fact_check` was retired from the tool catalog on 2026-09-23, so the fixtures below
use `fact_get` where the branch used it.)

What #1633 adds is not a claim about the task file but about the SKILL.md's length.
`app/skill_embed.py:4-8` records as prose that the chat route clips a skill to
`prefetch.SKILL_BODY_MAX` (6,000 characters) while the autonomy route "splice[s] the
whole file in with no cap". Nothing tested the second half, and the longest skill body
any vault-free node here hands to `_build_task_prompt` was `_SKILL_999`, 57 characters,
so a clip anywhere on the autonomy route would have dropped 10,915 of the 16,915
characters of `~/obsidian/skills/kg-mention-classifier/SKILL.md` — its whole Report
contract — with this file green. The `# ── #1633` section below the `run_task` node
carries why the enforced assertion is whole-body delivery rather than the preflight
facts this item was worded as grepping for.

Red was measured with `-m ""`, `pytest tests/test_autonomy_task_prompt_claims.py`,
on 2026-09-13 against the pre-fix vault (`git -C ~/obsidian checkout HEAD --` those
two paths): **4 failed, 8 passed**, all four the live claims, quoting the run —
"the #47 body instructs fact_check but the delivered prompt never mentions it",
"the task names a contradiction-resolution tool its dream skill runs no phase
for", "task: owners designated = none; want {'84'}", and "the skill stopped
declaring the fact store out of scope". `{'84', '463'}` was NOT one of the observed
messages — it is what the heading-as-prose variant of `_owner_designations` returns
on this file's own fixture, which is a different measurement and is labelled as
such where it is asserted. Against the fixed vault: **12 passed** (10 since the two round-scoped nodes
were dropped; re-measured 2026-09-24 with `-m ""`).

The gate-run nodes are green in both states by
design: they pin the mechanism and the file's own bookkeeping, not the vault's
contents, which is why a red in them means the seam moved rather than that the
deletion was undone.

`CLAUSE_NODES` maps each machine-gradable acceptance clause to the node that
pins it, because the review rung downgrades a `met` whose `test_node_id` is not
in a file the diff changed (`review.py:924-937`). Clause 4 has no node: its
deciding half is the sha the land commits, and `vault_round.py:314-346` grades
the review before `git add`/`git commit`, so no diff can carry it. That half is
the finalizer's, quoted after the land:

    git -C ~/obsidian log -1 --oneline -- autonomy/47-dream-consolidation.md
    git -C ~/obsidian log -1 --oneline -- skills/dream-consolidation/SKILL.md

both naming the sha `automod_vault_land(item_id=463)` returned, after
2026-09-12, with `git status --porcelain` empty on those two paths. The
deletion itself is the vault half of this item; until that commit exists the
sentence above about it is a promise, not a fact, and `git -C ~/obsidian show
HEAD:autonomy/47-dream-consolidation.md` still contains the instruction.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path

import pytest

from app import autonomy, prefetch

live_vault = pytest.mark.live_vault

ROOT = Path(__file__).resolve().parent.parent
VAULT = Path.home() / "obsidian"

TASK_FILE = VAULT / "autonomy" / "47-dream-consolidation.md"
SKILL_FILE = VAULT / "skills" / "dream-consolidation" / "SKILL.md"
TOOL_DEFS = ROOT / "agent_mcp" / "facts.py"

#: Repo-relative, because that is the form the gate compares a `test_node_id`
#: against: `gate.py:1109` builds `changed_tests` from `git diff --name-only`, and
#: `review.py:930-932` refuses a `met` whose node's file is not in that list. A bare
#: filename here would look right and downgrade every clause it names.
_FILE = f"tests/{Path(__file__).name}"

#: Which node pins which acceptance clause of #463. Nothing outside this file
#: reads it: the grader is handed the item's clauses as prose
#: (`review.py:item_contract`), not as this map, so its job is to keep the claim of
#: enforcement honest where the code lives rather than where a report is read once.
#: Clause 4 is absent because its deciding half — the sha the land commits — cannot
#: exist while the review that asks for it is running (`vault_round.py:314-346`).
CLAUSE_NODES = {
    1: f"{_FILE}::test_the_dream_task_instructs_no_fact_tool_the_skill_lacks",
    2: f"{_FILE}::test_every_fact_tool_the_dream_task_claims_reaches_the_prompt",
    3: f"{_FILE}::test_kg_contradiction_resolution_has_exactly_one_named_owner",
}

#: Transcribed from the item, not derived from the map above, so that deleting a
#: mapping is a red test. The round's second review (`SM_20260913_152649`, 2/2)
#: refused exactly this: a check that validated each entry but not the map's
#: coverage "stays green" when an entry disappears.
PINNED_CLAUSES = {1, 2, 3}
#: Clauses this file does not pin. Clause 4's deciding half is the vault sha the
#: land committed. Clause 5 ("the diff names no file under `agent_mcp/`") was a
#: property of the #463 round's own diff; the round's node read `git diff` against
#: a merge-base, which is meaningless once the file is on `main` (it was dropped
#: when the file was landed by hand on 2026-09-24, #1413).
FINALIZER_CLAUSES = {4, 5}
ITEM_CLAUSES = PINNED_CLAUSES | FINALIZER_CLAUSES

#: Tool names come out of the registration site rather than a hard-coded list,
#: so a new fact tool widens this check on its own.
_TOOL_NAME_RE = re.compile(r'Tool\(name="(fact_[a-z_]+)"')
_HEADING_RE = re.compile(r"(?m)^(#{2,4} .*)$")
_PHASE_RE = re.compile(r"^#{2,4}\s+Phase\s+\d")


def _fact_tools() -> set[str]:
    names = set(_TOOL_NAME_RE.findall(TOOL_DEFS.read_text(encoding="utf-8")))
    assert {"fact_get", "fact_resolve", "fact_invalidate"} <= names, (
        "the fact-tool registration site moved; fix the extractor, not this test")
    return names


def _body(text: str) -> str:
    """Markdown body: everything after the front matter's closing `---`."""
    parts = text.split("---\n")
    return text if len(parts) < 3 else "---\n".join(parts[2:])


_ACTIVITY_LOG_RE = re.compile(r"(?m)^#{1,4}\s*Activity Log\b")


def _instructions(text: str) -> str:
    """The instruction region of a task body: everything above `## Activity Log`.

    The runner appends that log itself (`_append_activity_log`, `app/autonomy.py:172`,
    called at `:1303-1315`), one line per run, so a tool name appearing there is a
    historical statement about a past run, not an instruction this run failed to
    receive. Scanning it would turn the tests red on a writer's changelog with no
    defect — a check reading input it cannot parse.
    """
    m = _ACTIVITY_LOG_RE.search(text)
    return text if m is None else text[:m.start()]


def _claims(text: str) -> set[str]:
    """Registered fact tools named anywhere in `text`."""
    return {n for n in _fact_tools() if n in text}


def _assert_claims_reach_prompt(instructions: str, prompt: str, where: str) -> None:
    """Clause 2's assertion, factored out so it has a fixture as well as a live
    file: every fact tool the instruction region names must appear in the prompt.

    #463's gate review measured the shape of the risk here — post-fix the #47 body
    names no fact tool at all, so the loop iterates an empty set and the node is
    green by construction until someone re-adds an instruction. That is what the
    clause asks for ("passes under either branch of clause 1"), but an assertion
    nobody ever feeds a positive case to is not evidence, so the extractor
    self-check feeds this exact function one of each: a body whose tool reached
    its prompt, and one whose did not.
    """
    for name in sorted(_claims(instructions)):
        assert name in prompt, (
            f"the {where} instructs {name} but the delivered prompt never mentions "
            f"it: delete the instruction or teach skills/dream-consolidation to do it")


def _activity_log(text: str) -> str:
    """The inverse of `_instructions()`: the runner's changelog region, or ""."""
    m = _ACTIVITY_LOG_RE.search(text)
    return text[m.start():] if m else ""


#: An owner designation is read off the paragraph that designates, not off the
#: whole file: `#84` and `skills/fact-improvement/SKILL.md` both appear elsewhere
#: in the task (the run-history line, the Paths section), and a substring test
#: over the whole body would be satisfied by any of those mentions while saying
#: nothing about who owns the surface.
#:
#: The unit is a markdown paragraph — a run of lines not broken by a blank line —
#: because that is the unit the prose uses to make a designation, and it is a
#: boundary in the file rather than a distance. The first cut used a
#: ±300/400-character window around the word `owner`, and #463's gate review
#: measured what that actually reached on these two files: `#47` from the
#: preceding paragraph (188 chars out) and `#38` from a later `(#38-40)` (336
#: chars out) — a window wide enough to swallow two neighbouring paragraphs' ids
#: and turn clause 3's live node red on prose that never named a second owner. A
#: window guesses where a sentence ends; a paragraph knows.
_OWNER_WORD_RE = re.compile(r"\bowners?\b", re.I)
_TASK_ID_RE = re.compile(r"#(\d+)")
_PROTOCOL_RE = re.compile(r"skills/([A-Za-z0-9._-]+)/SKILL\.md")


_HEADING_LINE_RE = re.compile(r"(?m)^\s{0,3}#{1,6}[ \t]")


def _paragraphs(text: str) -> list[str]:
    """Prose paragraphs: runs of lines separated by at least one blank line, with
    headings dropped.

    A heading is a title, not a sentence that designates anything. The live task
    file's section title is `## Structured fact store — not this task (removed
    2026-09-12, backlog #463)`, and a title that happened to contain the word
    `owner` would otherwise register its own id as a second owner — measured,
    not theorised: the fixture in `test_the_claim_extractor_actually_extracts`
    wrote exactly that title and the reader did exactly that."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        if not para.strip() or _HEADING_LINE_RE.match(para):
            continue
        out.append(para)
    return out


def _owner_designations(text: str) -> tuple[set[str], set[str]]:
    """(task ids, protocol skills) designated as owner anywhere in `text`.

    Scans only the paragraphs that contain the word `owner`, each to its own end.
    Name a second owner and its id or protocol enters the set; mention an
    unrelated id in the paragraph above or below and it does not. Over-reach runs
    toward a loud red and never toward missing a second owner: a sentence that
    *disclaims* ownership while naming an id in the same paragraph still
    registers it, which is pinned in `test_the_claim_extractor_actually_extracts`.
    """
    ids: set[str] = set()
    skills: set[str] = set()
    for para in _paragraphs(text):
        if not _OWNER_WORD_RE.search(para):
            continue
        ids.update(_TASK_ID_RE.findall(para))
        skills.update(_PROTOCOL_RE.findall(para))
    return ids, skills


def test_the_claim_extractor_actually_extracts():
    """The vault-side loops below iterate over `_claims(...)` of a body that
    currently names no fact tool, and over `_phase_bodies(...)` of a skill that
    currently runs no fact phase, so an empty result must be provably a real
    finding rather than a detector that stopped matching. No vault read; the gate
    runs it."""
    assert _claims("Use fact_resolve per entity, then fact_get and "
                   "fact_invalidate as needed.") == {
                       "fact_get", "fact_resolve", "fact_invalidate"}
    assert _claims("no tools here, only kg.sqlite and facts_idx.fact_id") == set()
    # The region split is what keeps the runner's own changelog out of the scan.
    assert _instructions("## Phase 1\nuse fact_get\n## Activity Log\nran and used "
                         "fact_get\n") == "## Phase 1\nuse fact_get\n"
    # And the phase reader needs its own teeth: a tool named in a subsection must
    # count for its phase, the phase must end at the next phase, and prose outside
    # any phase must not count — the three ways clause 1's other branch could be
    # mis-graded.
    skill = ("intro\n"
             "## Phase 2 — Synthesize\nmerge\n"
             "### Tools\nfact_get per entity\n"
             "## Phase 3 — Prune\ndelete\n"
             "## Out of scope\nfact_resolve is #84's\n")
    phases = _phase_bodies(skill)
    assert len(phases) == 2, f"the phase reader lost a phase: {phases!r}"
    assert _claims(phases[0]) == {"fact_get"}, (
        "a tool named in a subsection was not attributed to its phase")
    assert _claims(phases[1]) == set(), "the phase reader ran past its own section"
    assert "Out of scope" not in "\n".join(phases), (
        "out-of-scope prose counted as a phase: clause 1's second branch could be "
        "satisfied by writing a disclaimer instead of doing the work")
    # The log region, so the whole-file scan in the clause-1 test can say WHERE a
    # name it refuses is, instead of leaving a maintainer to work out whether the
    # task or its changelog drifted.
    text = "body says nothing\n## Activity Log\n- used fact_resolve once\n"
    assert _activity_log(text).startswith("## Activity Log")
    assert _claims(_instructions(text)) == set()
    assert _claims(_activity_log(text)) == {"fact_resolve"}
    # And the owner reader: it must return the designation, must be sensitive to a
    # SECOND owner being named — the case the clause-3 test's own name promises to
    # catch — and must not be fooled by a bare `#84` outside any owner sentence.
    ok = ('Contradiction resolution has exactly one owner: **autonomy task #84**, '
          'protocol `skills/fact-improvement/SKILL.md`, which runs daily.\n')
    assert _owner_designations(ok) == ({"84"}, {"fact-improvement"}), (
        "the owner reader cannot see a designation: the live assertions would then "
        "pass on a file that names nobody")
    second = ok + ('Backlog #999 is a second owner of the same surface, protocol '
                   '`skills/other-thing/SKILL.md`.\n')
    assert _owner_designations(second) == ({"84", "999"},
                                          {"fact-improvement", "other-thing"}), (
        "a second named owner is invisible, so 'exactly one' asserts nothing")
    assert _owner_designations("See backlog #84 for context.\n") == (set(), set()), (
        "the reader is a substring test in disguise: it would certify any file that "
        "mentions the id once")
    # The boundary, which is the whole point of scanning paragraphs. #463's gate
    # review measured a ±300/400-char window reaching `#47` one paragraph up and
    # `#38` one paragraph down, so a maintainer adding an unrelated sentence near
    # the owner prose would have turned clause 3's live node red without naming a
    # second owner. An id in the neighbouring paragraph must therefore stay out;
    # an id in the SAME paragraph must not. Red without the paragraph scoping.
    neighbours = ("The old phase is gone; no #47 run ever received it.\n\n"
                  + ok +
                  "\nThe dream skill is skills/dream-consolidation/SKILL.md, and "
                  "tasks (#38-40) share the nightly window.\n")
    assert _owner_designations(neighbours) == ({"84"}, {"fact-improvement"}), (
        "the reader reached outside the designating paragraph: clause 3 would go "
        "red on an edit that named no second owner")
    same_para = ok + " Backlog #999 is also mentioned, in passing.\n"
    assert _owner_designations(same_para) == ({"84", "999"}, {"fact-improvement"}), (
        "an id in the designating paragraph was dropped, so the reader is now "
        "reading less than the sentence that makes the designation")
    # Its known over-reach, pinned rather than hidden: the paragraph is the unit,
    # so a sentence that *disclaims* ownership while naming an id inside that same
    # paragraph still registers it. The error runs toward a loud red — an
    # over-reported owner turns a healthy file red and gets read — and never
    # toward missing a second owner, which is the direction that would matter.
    assert _owner_designations("There is no owner of this, not even #47.\n") == (
        {"47"}, set())
    # A heading is a title, not a designation, so a section title carrying the word
    # `owner` and an id of its own contributes nothing — the live file's title is
    # `## Structured fact store — not this task (removed 2026-09-12, backlog #463)`
    # and its body's designation sits under it. Measured: with headings treated as
    # prose this fixture returns `{'463', '84'}`, which is the same false second
    # owner the character window produced, arriving by a different route.
    assert _owner_designations("## Owner of the fact store (backlog #463)\n\n"
                               "The one owner is task #84, protocol "
                               "`skills/fact-improvement/SKILL.md`.\n") == (
        {"84"}, {"fact-improvement"}), (
        "a section heading's own id entered the owner set")
    # Clause 2's assertion itself, both ways. Post-fix the live body names no tool
    # and the live loop is empty, so without these two lines the node has never
    # been observed firing on a real claim.
    _assert_claims_reach_prompt("use fact_get per entity", "…fact_get…", "probe")
    with pytest.raises(AssertionError, match="instructs fact_get"):
        _assert_claims_reach_prompt("use fact_get per entity", "…prose only…",
                                    "probe")


def _heading_level(h: str) -> int:
    return len(h) - len(h.lstrip("#"))


def _phase_bodies(text: str) -> list[str]:
    """Body text under each numbered `## Phase N ...` heading, subsections included.

    A numbered phase's section runs to the next heading of the phase's own level or
    shallower, not merely to the next heading: a `### Tools` subsection under
    `## Phase 3` is part of Phase 3, and reading only up to the next heading of any
    level would drop a tool named there — which would fail clause 1's legitimate
    other branch (a genuinely added phase) as a false alarm rather than catch a real
    disagreement between the two files."""
    parts = _HEADING_RE.split(text)          # [pre, h1, body1, h2, body2, …]
    out: list[str] = []
    for i in range(1, len(parts) - 1, 2):
        head = parts[i]
        if not _PHASE_RE.match(head):
            continue
        level = _heading_level(head)
        chunk = [parts[i + 1]]
        for j in range(i + 2, len(parts) - 1, 2):
            if _heading_level(parts[j]) <= level:
                break
            chunk.append(parts[j])          # the nested heading itself
            chunk.append(parts[j + 1])
        out.append("\n".join(chunk))
    return out


def test_the_prompt_is_built_from_the_skill_and_description_never_the_body():
    """The mechanism behind #463, pinned directly so the tests below cannot pass
    vacuously: a tool named only in the body reaches no one.

    If #951 ever makes bodies live, this goes red — which is the signal to
    revisit this item, not to delete the assertion.
    """
    from app import autonomy

    task = {"id": 999, "name": "seam probe", "skill_name": "probe",
            "description": "Consolidates prose memory.",
            "body": "Use fact_resolve to close contradictions."}
    prompt = autonomy._build_task_prompt(task, "SKILL PROBE BODY: prose only.")
    assert "SKILL PROBE BODY" in prompt, "the skill stopped reaching the prompt"
    assert "Consolidates prose memory." in prompt, "description stopped reaching it"
    assert "fact_resolve" not in prompt, (
        "the runner now renders the task body, so a body-only instruction would be "
        "live: #463's deletion is no longer the right fix (see #951)")


_TASK_999 = {
    "id": 999, "name": "seam probe", "skill_name": "probe-skill",
    "description": "Consolidates prose memory only.",
    "body": "3. Fact Store Consolidation — run fact_resolve over every entity.",
}
#: The one skill-contract marker in the synthetic prompt, named because two
#: assertions reach for it: the engine must receive it, and the run record's window
#: must still be wide enough to show it.
SKILL_MARKER = "## Phase 1 \u2014 Orient"
_SKILL_999 = f"# probe-skill\n\n{SKILL_MARKER}\nread the prose files.\n"


def _drive_one_run(monkeypatch, tmp_path, *, task, skill_text):
    """Drive the real `autonomy.run_task` with the engine stubbed, and return
    everything the boundary produced: the message list handed to the engine, and
    the run record's markdown on disk.

    The stub is the one `tests/test_background_inner_voice.py` uses — same patch
    surface, same event stream — so this exercises the runner's own wiring rather
    than a parallel implementation of it. Nothing writes outside `tmp_path`: the
    task file, the activity log, the status field and the runs directory are all
    redirected.
    """
    import app.harness as harness
    import app.harness.mcp_pool as mcp_pool

    captured: dict = {}

    async def _run_query(messages, options):
        captured["messages"] = messages
        captured["options"] = options
        yield {"type": "text_delta", "text": "consolidated"}
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}}

    class Opts:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "999-x.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: dict(task))
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: skill_text)
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
    # The record must be the REAL writer: reading back its `## Prompt` section is
    # half of what this test is for. Only its destination moves.
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(harness, "run_query", _run_query)
    monkeypatch.setattr(harness, "RunOptions", Opts)
    monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
    monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "SYS")
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")

    # The id comes off the task the caller passed, not off `_TASK_999`: #1633's
    # long-skill node drives this same helper with its own task, and the id selects
    # the run-record directory (`runs/<id>/`) and the evidence-pilot branch of
    # `_evidence_prompt` (`app/autonomy.py`, `EVIDENCE_PILOT_TASK_IDS`). Both callers
    # use a non-pilot id, so what the engine receives is the built prompt unchanged.
    asyncio.run(autonomy.run_task(task["id"]))
    return captured


def test_the_runner_delivers_the_skill_and_records_what_it_delivered(monkeypatch,
                                                                     tmp_path):
    """The seam in full, at the boundary #463's own acceptance check is written
    against. Its triage clause says the fix is 'Verified by rendering the real
    prompt from the real task file', and the post-land human check reads a run
    record's `## Prompt` — so both ends have to be real, not just the return
    value of `_build_task_prompt`.

    The prior cut stopped at that return value, and #463's gate review named the
    gap: built prompt → worker turn (`app/autonomy.py:1119` builds the message list,
    `:1146` hands it to the engine) → the record's `## Prompt`, none of it
    asserted. This drives the real `run_task` and asserts on all three:

      * the message list the ENGINE received carries the skill and the
        description, and does not carry the body's `fact_resolve` — the defect
        one level above the one that caused this item;
      * the run record on disk shows the same prompt head it delivered.

    No vault read: the task and skill are synthetic and every write lands in
    `tmp_path`, so the gate runs this node.
    """
    out = _drive_one_run(monkeypatch, tmp_path, task=_TASK_999, skill_text=_SKILL_999)
    assert out.get("messages"), "run_task never reached the engine"

    # `content` must be a plain string before any `in` below means anything.
    # `run_task` appends strings (app/autonomy.py:1140) and the session writer later
    # replaces them with block dicts, so on a different message shape these
    # assertions would be testing membership in a list of dicts — always false,
    # and the first two would then fail for the wrong reason while the third
    # ("not in") would pass for the wrong reason. #463's gate review, attempt 2/2,
    # named exactly this; the type assertion is what makes it attributable.
    delivered = out["messages"][0]["content"]
    assert isinstance(delivered, str) and delivered, (
        f"engine message content is {type(delivered).__name__}, not the string the "
        "assertions below assume — re-read app/autonomy.py:1140 before trusting any of "
        "them")
    assert out["messages"][0]["role"] == "user"
    assert "## Phase 1 — Orient" in delivered, (
        "the skill stopped reaching the engine's message list: every claim in "
        "this file about what a run is handed would be about a string that no "
        "worker ever saw")
    assert "Consolidates prose memory only." in delivered
    assert "fact_resolve" not in delivered, (
        "the body reached the worker: a body-only instruction would be live "
        "again, which is the defect this item deletes rather than the fix "
        "(see #951 before changing this)")

    # The un-truncated artefact. #463's own post-land human check greps a run
    # record's `## Prompt`, and that field is `prompt[:500]` (`app/autonomy.py:1205`,
    # `:1225`, `:1252`, `:1265`) — a window, on #47's real prompt, that reaches
    # about a tenth of the way into its skill. The session store keeps the WHOLE
    # user row, which is the only on-disk artefact an ABSENCE claim can be made
    # from, so this asserts against it and the record gets the weaker claim.
    sessions = list((tmp_path / "sessions").glob("*.json"))
    assert len(sessions) == 1, f"expected one session, got {len(sessions)}"
    stored = json.loads(sessions[0].read_text(encoding="utf-8"))
    user_rows = [m for m in stored["messages"] if m.get("role") == "user"]
    assert user_rows, "the session stored no user row"
    stored_text = "\n".join(c.get("text", "") for c in user_rows[0]["content"])
    assert stored_text == delivered, (
        "the session's user row is no longer the prompt that was delivered, so a "
        "reviewer reading it is not reading what the worker read")
    assert "fact_resolve" not in stored_text, (
        "the body reached the worker and the record of it: a body-only instruction "
        "would be live again, which is the defect this item deletes rather than the "
        "fix (see #951 before changing this)")

    records = list((tmp_path / "runs" / "999").glob("run_*.md"))
    assert len(records) == 1, f"expected one run record, got {len(records)}"
    record = records[0].read_text(encoding="utf-8")
    assert "## Prompt" in record, "the record stopped showing what it delivered"
    head = record.split("## Prompt", 1)[1].split("## Response")[0]
    # The head is a PREFIX of what was delivered, asserted as a prefix rather than
    # for a particular paragraph inside it. The prior cut asserted this node's skill
    # heading was present and #463's gate review measured why that is a coin flip:
    # on the synthetic prompt the heading starts at index 480 of a 500-char slice, so
    # it cleared the ceiling by ONE character, and any growth in the silent-hint
    # preamble would have reddened this node for a change with no bearing on the
    # contract. Prefix-ness is the property; position inside a window is not.
    # Compared at full length, not over a slice of it: the first cut of this line
    # was `delivered.startswith(quoted[:200])`, which is satisfied by 200 matching
    # characters while the other ~300 have drifted — the exact kind of a test that
    # passes by reading less than it claims. Measured on the synthetic prompt: the
    # whole 501-char head is a prefix (`delivered.startswith(quoted)` is True and
    # the first difference between `delivered` and `quoted` is at 501, the end of
    # the slice), so comparing all of it costs nothing and cannot be satisfied by a
    # partial match.
    quoted = head.strip().rstrip(".").rstrip()
    assert quoted and delivered.startswith(quoted), (
        "the record's `## Prompt` is no longer the head of the prompt it delivered")
    assert len(quoted) <= 500, (
        f"the record shows {len(quoted)} chars, past the documented 500-char slice: "
        "re-measure what a reviewer of `## Prompt` can see before quoting this test "
        "as evidence for #463's post-land check")
    # And the window has to still REACH the skill, or the record is a prefix of
    # something nobody could act on: #463's human check greps `## Prompt` to see what
    # a run was told, and the skill is where the delivered contract lives. This is
    # the assertion the previous paragraph removed, restored with the reason it was
    # fragile measured alongside, so a red here is attributable rather than a mystery
    # (#463's review, attempt 2/2: with only the prefix check, the node never asks
    # whether the window covers the skill at all). On today's prompt the marker
    # starts at index 480 of a 500-char window — a margin of 20 characters. Growth in
    # the silent-hint preamble at `app/autonomy.py:948` past that margin reddens THIS
    # node, and this is the message that says so.
    assert SKILL_MARKER in head, (
        "the record's prompt head no longer reaches the delivered skill — it is a "
        "prefix of the prompt but shows a run nothing of its contract. Measured "
        "margin on the synthetic prompt: 19 chars (marker at 480, window 499 of a "
        "573-char prompt); if the silent-hint preamble grew, that is the cause, not "
        "this item")


# ── #1633: how much of the SKILL.md survives the trip ─────────────────────────
#
# `app/skill_embed.py:4-8` is the only place that states, and states as prose, how
# much of a SKILL.md each route carries: the chat turn-start injection clips to
# `prefetch.SKILL_BODY_MAX` (`app/prefetch.py:54`, pinned by the chat route at
# `tests/test_prefetch.py:1219-1233`), while `autonomy._build_task_prompt`
# "splice[s] the whole file in with no cap and carr[ies] it for the run". No node
# tested the second half of that sentence, and the `run_task` node above drives a
# 57-character fixture skill — so a clip added anywhere on the autonomy route would
# have cut #74's 16,915-character `skills/kg-mention-classifier/SKILL.md`
# (`~/obsidian`, measured 2026-09-28) down to its first 6,000 characters with this
# file green: 10,915 characters, 65% of the file, the whole Report contract.
#
# Why the tail and not the preflight facts the item names: those facts sit at
# characters 3,577 (`## Preflight facts`), 3,815 (`127.0.0.1:8096`), 4,387
# (`_endpoint_alive`) and 5,218 (`ps -eo pid,etime,cmd`) — every one of them INSIDE
# a 6,000-character clip. A grep for the markers alone would therefore stay green
# through exactly the regression that motivated the item, which is why the two nodes
# below pin whole-body delivery (subsuming the markers) and the live node under #979
# pins the markers individually.

#: Imported, not transcribed, so the fixture straddles whatever the cap is today and
#: a re-tuned cap reddens the fixture's own self-check rather than hiding it.
_CAP = prefetch.SKILL_BODY_MAX

#: One marker on each side of the cap. The head marker is what a clip KEEPS — its
#: presence proves nothing — and the tail marker is what a clip drops, so the tail is
#: the only one of the pair with teeth. The head exists to make the fixture's own
#: straddle check in `_long_skill` meaningful.
_LONG_HEAD_MARKER = "## Head clause 4a1e — inside the cap"
_LONG_TAIL_MARKER = "## Tail clause 9f3c — past the cap"

_LONG_TASK = {
    "id": 997, "name": "long-skill seam probe", "skill_name": "long-fixture",
    "description": "Reports today's classifier verdicts.",
    "body": "Append the archive index by hand.",   # never rendered: #463 / #951
}


def _long_skill() -> str:
    """A synthetic SKILL.md of the same order as #74's, with a marker either side of
    the cap.

    Sized against the file whose silent loss produced the item (16,915 characters,
    17,037 bytes, 2026-09-28) rather than against a round number, because the defect
    being pinned is proportional: at #74's size a 6,000-character cap is a 65% loss,
    and at 6,001 it is a rounding error. The three asserts at the end are the
    fixture's own teeth — a fixture that quietly stopped straddling the cap would
    otherwise turn both nodes below into a test that cannot fail.
    """
    filler = "".join(
        f"step {i:03d}: report the classifier's verdict, never the probe's exit "
        "code, and cite the file every row came from.\n"
        for i in range(140))
    skill = ("# long-fixture\n\n"
             f"{_LONG_HEAD_MARKER}\nThis section is what a clip keeps.\n\n"
             + filler + "\n"
             f"{_LONG_TAIL_MARKER}\n"
             "This section is what a clip drops, and it is the Report contract.\n")
    assert skill.index(_LONG_HEAD_MARKER) < _CAP, (
        f"the head marker moved to {skill.index(_LONG_HEAD_MARKER)}, past the "
        f"{_CAP}-character cap, so it no longer marks the half a clip keeps")
    assert len(skill) > _CAP + 6000, (
        f"the fixture is {len(skill)} characters, not meaningfully past the "
        f"{_CAP}-character cap — it has to stay long enough to stand in for #74's "
        "16,915-character skill or this node stops measuring the loss it names")
    assert skill.index(_LONG_TAIL_MARKER) > _CAP, (
        f"the tail marker sits at {skill.index(_LONG_TAIL_MARKER)}, inside the "
        f"{_CAP}-character cap: a clip would keep it, and the two nodes below could "
        "not fail")
    return skill


def test_a_skill_past_the_chat_cap_reaches_the_built_prompt_whole():
    """#1633 clause 1: the autonomy prompt is built from a long skill WHOLE, because
    today the only thing that says so is a sentence in `app/skill_embed.py`.

    `skill in prompt` is the assertion #463's gate review struck from the live node
    below, on the grounds that `_build_task_prompt` concatenates its argument and the
    substring test pinned the function's shape. That objection does not reach here:
    the claim being graded THERE was about tool names, for which whole-body presence
    said nothing, while the claim HERE *is* whole-body delivery. The node fails the
    instant a clip appears anywhere between the SKILL.md and the prompt — at the call
    site (`app/autonomy.py:3524`, where `record_skill_embed` currently records
    `embedded_chars=len(skill_content)` as a claim rather than a measurement) or
    inside the builder itself.

    Red was measured in this round's worktree, not argued. Putting
    `skill_content = skill_content[:6000]` at the top of the builder reddens this node
    on the tail assertion first, and 11 of the file's other 13 nodes stay green — the
    short-skill mechanism test and the short-skill delivery test among them, because a
    57-character fixture cannot be clipped. That is the gap this node exists to close,
    and `test_the_runner_delivers_a_long_skill_whole_to_engine_and_store` measured the
    mirror case. No vault read; the gate runs it.
    """
    skill = _long_skill()
    prompt = autonomy._build_task_prompt(_LONG_TASK, skill)

    assert _LONG_TAIL_MARKER in prompt, (
        f"the prompt stops at the cap: bytes past {_CAP} of a {len(skill)}-character "
        "skill are gone, which on #74's real skill is 65% of the file — the whole "
        "Report contract. `app/skill_embed.py:4-8` promises this route splices the "
        "whole file in uncapped; either keep that promise or fix the sentence and "
        "#624's cost model with it")
    assert skill in prompt, (
        f"the skill body reached the prompt only in part ({len(prompt)} chars built "
        f"from a {len(skill)}-char skill): no clip is expected on this route")
    assert _LONG_HEAD_MARKER in prompt, (
        "even the part inside the cap is gone, so this is a different failure from "
        "the one above — the skill stopped reaching the prompt at all")
    assert "Reports today's classifier verdicts." in prompt, (
        "the description stopped reaching a long-skill prompt")


def test_the_runner_delivers_a_long_skill_whole_to_engine_and_store(monkeypatch,
                                                                   tmp_path):
    """#1633 clause 2: the same promise across the two boundaries a clip can sit
    behind, with the real `run_task`.

    The node above pins the built string. A cap could equally be applied after the
    builder and before the engine, or the engine could be handed the whole prompt
    while the session store recorded a window of it — the second is not hypothetical,
    since the run record already records only `prompt[:500]` (`app/autonomy.py`).
    This drives the real runner with the same fixture and asserts the two artefacts
    that matter: the message list the engine received, and the session's stored user
    row, which the file's own docstring calls the only on-disk artefact an ABSENCE
    claim can be made from.

    Measured the other way round from clause 1's node: clipping the delivered message
    to `prompt[:6000]` at `app/autonomy.py`'s message list reddens THIS node alone
    (1 failed, 13 passed), because the builder still returns the whole body. The pair
    is what covers the route — neither node sees the other's cut point. No vault read;
    the gate runs it.
    """
    skill = _long_skill()
    out = _drive_one_run(monkeypatch, tmp_path, task=_LONG_TASK, skill_text=skill)
    assert out.get("messages"), "run_task never reached the engine"

    delivered = out["messages"][0]["content"]
    assert isinstance(delivered, str) and delivered, (
        f"engine message content is {type(delivered).__name__}, not the string the "
        "assertions below assume — re-read how `run_task` builds its message list "
        "before trusting any of them")
    assert skill in delivered, (
        f"the engine was handed less than the {len(skill)}-character skill: a clip on "
        "the delivery route is invisible to the builder-level node above, which is "
        "why this node exists")
    assert _LONG_TAIL_MARKER in delivered, (
        "the tail of the skill never reached the worker: everything it was supposed "
        "to do past the cap is dead text, and #74's preflight facts would be the "
        "first casualty on the real task")
    assert "Append the archive index by hand." not in delivered, (
        "the task body reached the worker on a long-skill prompt too: the #463 "
        "mechanism moved, see #951 before changing this")

    sessions = list((tmp_path / "sessions").glob("*.json"))
    assert len(sessions) == 1, f"expected one session, got {len(sessions)}"
    stored = json.loads(sessions[0].read_text(encoding="utf-8"))
    user_rows = [m for m in stored["messages"] if m.get("role") == "user"]
    assert user_rows, "the session stored no user row"
    stored_text = "\n".join(c.get("text", "") for c in user_rows[0]["content"])
    assert stored_text == delivered, (
        f"the store kept {len(stored_text)} of the {len(delivered)} characters that "
        "were delivered, so the transcript a reviewer or an owed-check reads is no "
        "longer what the worker read")
    assert len(stored_text) > _CAP, (
        f"the stored user row is {len(stored_text)} characters, at or under the "
        f"{_CAP}-character chat cap: the store became a window, and an absence claim "
        "made from it (the #1563 post-run check, the `:8000` re-derivation) is now "
        "unsound")

    # The run record, asserted for what it CANNOT show. Its `## Prompt` is a
    # `prompt[:500]` window; measured here it is 499 characters and the skill begins
    # at 476, so the record displays 23 characters of a 15,731-character skill and
    # neither marker survives the cut (the head marker starts at 492, eight characters
    # from the edge). A reviewer who reads `## Prompt` to decide whether a run got its
    # skill is reading a prefix that looks identical whether the body arrived or not —
    # exactly the reading #1563's owed post-run check would take if it read the record
    # instead of the session row. Pinned, not commented.
    records = list((tmp_path / "runs" / str(_LONG_TASK["id"])).glob("run_*.md"))
    assert len(records) == 1, f"expected one run record, got {len(records)}"
    record = records[0].read_text(encoding="utf-8")
    assert "## Prompt" in record, "the record stopped showing what it delivered"
    head = record.split("## Prompt", 1)[1].split("## Response")[0]
    quoted = head.strip().rstrip(".").rstrip()
    assert quoted and delivered.startswith(quoted), (
        "the record's `## Prompt` is no longer the head of the prompt it delivered")
    assert len(quoted) <= 500, (
        f"the record shows {len(quoted)} characters, past the documented 500-char "
        "slice: the assertion below assumes the window cannot reach the skill's tail")
    assert _LONG_TAIL_MARKER not in head, (
        "the record window now reaches the skill's tail, so `## Prompt` can be read "
        "as evidence of whole-body delivery after all: re-check #1633's owed post-run "
        "check (written to read the session row for exactly this reason) before "
        f"treating this red as a defect, then update it with the measured window — now "
        f"{len(quoted)} characters")


@live_vault
def test_every_fact_tool_the_dream_task_claims_reaches_the_prompt():
    """#463 clause 2, against the real task file: any fact tool the #47 body
    names must appear in the prompt a run is actually handed.

    This is the seam test for the vault→prompt boundary: `_parse_task_file`,
    `_load_skill_content` and `_build_task_prompt` are the same three calls the
    scheduler makes at `app/autonomy.py:968`, run here against the real files."""
    from app import autonomy

    task = autonomy._parse_task_file(TASK_FILE)
    assert task, "#47 stopped parsing as an autonomy task"
    skill = autonomy._load_skill_content(task.get("skill_name"))
    assert skill, f"skill {task.get('skill_name')!r} did not load"
    prompt = autonomy._build_task_prompt(task, skill)
    # No `skill in prompt` assertion here: `_build_task_prompt` concatenates its
    # argument, so that would pin the function's shape, not a behaviour (#463's
    # gate review). What matters is the claim loop below and the mechanism test.
    instructions = _instructions(_body(TASK_FILE.read_text(encoding="utf-8")))
    _assert_claims_reach_prompt(instructions, prompt, "#47 body")


@live_vault
def test_the_dream_task_instructs_no_fact_tool_the_skill_lacks():
    """#463 clause 1, as one boolean: either the task claims no
    contradiction-resolution tool, or the skill a run loads instructs one under a
    numbered phase. Today it is the former.

    Whole file, Activity Log included, because that is the check the item words —
    `grep -c "fact_resolve" <task file> == 0` — and a human running it reads the same
    bytes this test reads. The consequence is a convention, recorded in the task
    file's own prose and honoured by its changelog since 2026-09-12: an Activity Log
    line describes what a run did without naming the tools, so the file's grep stays
    clean for the right reason. Clauses 2 and 3 scan `_instructions()` instead,
    because those are about what a run fails to *receive*, and a changelog line is
    not an instruction."""
    text = TASK_FILE.read_text(encoding="utf-8")
    skill = SKILL_FILE.read_text(encoding="utf-8")
    claims = "fact_resolve" in text
    skill_runs_fact_phase = any(_claims(b) for b in _phase_bodies(skill))
    where = ""
    if claims and not skill_runs_fact_phase:
        where = " — in " + (
            "the instruction region" if "fact_resolve" in _instructions(text)
            else "an Activity Log line; the item's grep is whole-file, so reword "
                 "that line, which is the convention its prose records")
    assert (not claims) or skill_runs_fact_phase, (
        "the task names a contradiction-resolution tool its dream skill runs no "
        "phase for: delete the instruction or add the phase to "
        "skills/dream-consolidation/SKILL.md (item #463 clause 1)" + where)


@live_vault
def test_kg_contradiction_resolution_has_exactly_one_named_owner():
    """#463 clause 3, with the word "exactly" carrying its weight: the single
    designation of an owner — the task id and the protocol skill named around the
    word `owner` — is #84 and `skills/fact-improvement/SKILL.md`, in both files, in
    prose meant to survive. Naming a second owner turns this red; so does deleting
    the first.

    Scoped to `_instructions()`, not the whole body: the runner appends the Activity
    Log itself, so an owner named only in a writer's changelog line is a note about
    yesterday, not the standing pointer clause 3 asks for. Deleting the prose
    section while leaving that line would otherwise keep this green."""
    task_instr = _instructions(_body(TASK_FILE.read_text(encoding="utf-8")))
    skill = SKILL_FILE.read_text(encoding="utf-8")
    for name, text in (("task", task_instr), ("skill", skill)):
        ids, protocols = _owner_designations(text)
        assert ids == {"84"}, f"{name}: owners designated = {ids or 'none'}; want {{'84'}}"
        assert protocols == {"fact-improvement"}, (
            f"{name}: owner protocols = {protocols or 'none'}; want the one protocol")
    assert (VAULT / "skills" / "fact-improvement" / "SKILL.md").exists(), (
        "the named owner does not exist — the pointer is a dead link")


@live_vault
def test_the_dream_skill_keeps_the_fact_store_out_of_scope():
    """The half of clause 3 that a run can actually read: the SKILL.md itself
    says the graph is #84's surface, because the SKILL.md is what gets
    delivered."""
    skill = SKILL_FILE.read_text(encoding="utf-8")
    assert "Out of scope" in skill, "the skill stopped declaring the fact store out of scope"
    assert "#84" in skill and "fact-improvement" in skill, "owner not named in the skill"
    # The phase reader must stop at the phase boundary. On the landed skill that is
    # not reachable through a disclaimer: clause 1 chose the deletion, and the skill
    # names NO fact tool anywhere (`grep -coE 'fact_(get|add|resolve…)'` → 0), so
    # there is no tool-naming out-of-scope paragraph for a phase reader to swallow.
    # The reader's own teeth are the fixture in
    # `test_the_claim_extractor_actually_extracts`; here the property is stated as
    # the non-existence it is, below. Do not add a tool name to the skill to give
    # this test something to reach.
    phase_text = "\n".join(_phase_bodies(skill))
    assert "Out of scope" not in phase_text, (
        "the out-of-scope paragraph was read as part of a phase, so a disclaimer "
        "could satisfy clause 1's second branch by naming the tools it excludes")
    body_claims = _claims(_instructions(_body(TASK_FILE.read_text(encoding="utf-8"))))
    runs_a_fact_phase = any(_claims(b) for b in _phase_bodies(skill))
    if not body_claims:
        assert not runs_a_fact_phase, (
            "the task claims no fact-store phase while the skill runs one")
        assert not _claims(skill), (
            f"the dream skill names fact tools {sorted(_claims(skill))} while the "
            "task claims none: the fact store is #84's, and the skill should not "
            "name its tools at all")
    else:
        # Clause 1's other branch — a genuinely added phase — is allowed, and
        # would rewrite this file with the skill. What is not allowed is the two
        # files disagreeing, which is #463 verbatim.
        assert runs_a_fact_phase, "the task claims a fact-store phase its skill does not run"


@live_vault
def test_the_contract_lives_in_files_tracked_on_the_vault_main():
    """#463 clause 4, as far as it is checkable before the commit exists: the two
    files are tracked on the vault's `main`, which is the tree the autonomy
    runner reads and the land commits to. The sha and clean-porcelain half is the
    finalizer's post-land check — `vault_round.py:314-350` reviews before
    `git add`/`git commit`, so no diff can carry a sha at review time."""

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(("git", "-C", str(VAULT), *args),
                              capture_output=True, text=True, check=False)

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    assert branch.stdout.strip() == "main", f"vault on {branch.stdout.strip()!r}, not main"
    for rel in (TASK_FILE.relative_to(VAULT), SKILL_FILE.relative_to(VAULT)):
        assert git("ls-files", "--error-unmatch", str(rel)).returncode == 0, (
            f"{rel} is not tracked in the vault — the land would commit nothing")


#: #74's three preflight facts, each with what it stops a run from getting wrong.
#: Transcribed from `~/obsidian/skills/kg-mention-classifier/SKILL.md`, which is where
#: `~/obsidian` commit `5afb1fd4` ("make kg-mention-classifier SKILL.md the
#: authoritative run-to-run surface", #1563) put them: the 09-26 run had re-derived
#: `:8000` from memory and reported an outage that did not exist.
_PREFLIGHT_FACTS = {
    "127.0.0.1:8096": "the endpoint the run must probe",
    "_endpoint_alive": "the skill's own words for the only honest probe",
    "ps -eo pid,etime,cmd": "the writer check that tells a quiet window apart from a "
                            "dead pipeline",
}


@live_vault
def test_the_preflight_facts_of_the_kg_classifier_skill_reach_the_prompt():
    """#1633 clause 3, against the live skill, with each fact pinned by name.

    `_load_skill_content` and `_build_task_prompt` are two of the three calls the
    scheduler makes in `run_task`; the third, `_parse_task_file`, is exercised against
    the live tree by `test_every_fact_tool_the_dream_task_claims_reaches_the_prompt`.

    READ THE MARK — it is not decoration here. Every automated pytest invocation in
    this tree passes `-m "not live_vault"` (the `pytest.ini` markers block; measured
    node-by-node at the top of this file), so NO gate rung executes this node. An
    author editing these three facts out of the skill stays unpinned until #979
    closes, and that is the half of #1633 the item records as a ruling still owed, not
    as work this round did. What the loop DOES enforce is that the whole body survives
    the trip — the two nodes above — which is the property these markers depend on:
    they sit at characters 3,577 to 5,218 of a 16,915-character file, all inside the
    6,000-character chat cap, so no cap regression can take them and no marker-only
    test can see one.
    """
    skill = autonomy._load_skill_content("kg-mention-classifier")
    assert skill, (
        "kg-mention-classifier did not load. `_load_skill_content` resolves a slug to "
        "~/obsidian/skills/<slug>/SKILL.md and task #74 names it as its `skill_name`, "
        "so a real run would have died at `Skill not found` before a prompt was built")
    assert len(skill) > _CAP, (
        f"the live skill is {len(skill)} characters, no longer past the {_CAP}-char "
        "chat cap, so #1633's loss model (65% of the file) and the two enforced nodes "
        "standing in for it both need re-measuring")
    prompt = autonomy._build_task_prompt(
        {"id": 74, "name": "KG mention classifier",
         "skill_name": "kg-mention-classifier", "description": ""}, skill)
    assert skill in prompt, (
        f"the live skill reached the prompt only in part: {len(prompt)} characters "
        f"built from a {len(skill)}-character skill. The uncapped promise of "
        "`app/skill_embed.py:4-8` broke, and everything past the cut is dead text for "
        "the run")
    for fact, why in sorted(_PREFLIGHT_FACTS.items()):
        assert fact in prompt, (
            f"the #74 prompt no longer carries {fact!r} — {why}. Commit 5afb1fd4 put "
            "it in the skill because the 2026-09-26 run re-derived the endpoint from "
            "memory and reported an outage that was not there; removing it re-arms "
            "that failure. (#1633 clause 3 — and no automated rung runs this node; "
            "see #979)")


#: The nodes that read the live `~/obsidian`, and the nodes that do not — transcribed
#: from the decorators below in the same spirit as `PINNED_CLAUSES` above, so a mark
#: appearing or disappearing is a red rather than a quiet change to which half the
#: loop enforces. #1633 clause 4 is exactly this split: of the three nodes that round
#: added, only the preflight-facts node is on the live list, because the two delivery
#: nodes are synthetic and have to fire on every gate.
LIVE_VAULT_NODES = frozenset({
    "test_every_fact_tool_the_dream_task_claims_reaches_the_prompt",
    "test_the_dream_task_instructs_no_fact_tool_the_skill_lacks",
    "test_kg_contradiction_resolution_has_exactly_one_named_owner",
    "test_the_dream_skill_keeps_the_fact_store_out_of_scope",
    "test_the_contract_lives_in_files_tracked_on_the_vault_main",
    "test_the_map_covers_every_clause_the_item_numbers",
    "test_the_preflight_facts_of_the_kg_classifier_skill_reach_the_prompt",
})

ENFORCED_NODES = frozenset({
    "test_the_claim_extractor_actually_extracts",
    "test_the_prompt_is_built_from_the_skill_and_description_never_the_body",
    "test_the_runner_delivers_the_skill_and_records_what_it_delivered",
    "test_a_skill_past_the_chat_cap_reaches_the_built_prompt_whole",
    "test_the_runner_delivers_a_long_skill_whole_to_engine_and_store",
    "test_every_clause_names_a_node_that_exists",
    "test_only_the_nodes_that_read_the_vault_carry_the_live_mark",
})


def test_only_the_nodes_that_read_the_vault_carry_the_live_mark():
    """#1633 clause 4: which half of this file the loop enforces is itself asserted,
    not commented.

    Both directions are real failures. A mark dropped from a vault-reading node puts a
    file the round cannot control onto a hard gate, where it reddens the next author
    for a nightly writer's edit — the reason the `pytest.ini` marker block exists. A
    mark added to a synthetic node retires a pin in silence, which is the shape of
    the gap this item was filed from: the promise that the autonomy route carries a
    skill whole lived only in `app/skill_embed.py`'s docstring, where no rung reads
    it. `test_the_map_covers_every_clause_the_item_numbers` is the vault-reading twin
    of this check; this one needs no vault, so the gate runs it.
    """
    import sys
    mod = sys.modules[__name__]
    tests = {n for n in vars(mod)
             if n.startswith("test_") and callable(getattr(mod, n, None))}
    assert tests, "the module holds no test callables; this scan is measuring nothing"
    marked = {n for n in tests
              if any(getattr(m, "name", "") == "live_vault"
                     for m in getattr(getattr(mod, n), "pytestmark", ()))}
    assert marked == set(LIVE_VAULT_NODES), (
        f"the mark moved: {sorted(marked ^ set(LIVE_VAULT_NODES))} changed hands "
        "between the enforced half and the human half. A mark removed from a "
        "vault-reading node puts the vault on the gate; a mark added to a fixture node "
        "retires a pin nobody will notice. Reconcile the decorator with the two lists")
    enforced = tests - marked
    assert enforced == set(ENFORCED_NODES), (
        f"the enforced set drifted: {sorted(enforced ^ set(ENFORCED_NODES))}. The two "
        "lists are a transcription of the module's own nodes, and the module "
        "docstring's counts (\"Seven of the fourteen\") are a transcription of these")


def test_every_clause_names_a_node_that_exists():
    """The map is a claim of enforcement, so both ways it can rot are checked: a
    value that no longer resolves to a test in this file, and an entry that was
    quietly deleted. No vault read; the gate runs it."""
    import sys
    mod = sys.modules[__name__]
    assert set(CLAUSE_NODES) == PINNED_CLAUSES, (
        f"map covers {sorted(CLAUSE_NODES)}, expected {sorted(PINNED_CLAUSES)} — a "
        f"clause lost its pin, or the transcription above is stale")
    assert callable(getattr(mod, "test_the_dream_skill_keeps_the_fact_store_"
                             "out_of_scope", None)), "companion pin vanished"
    for clause, node in CLAUSE_NODES.items():
        assert "::" in node, f"clause {clause}: {node!r} is not a node id"
        fname, _, func = node.partition("::")
        assert fname == _FILE, f"clause {clause} points outside this file: {node}"
        assert callable(getattr(mod, func, None)), (
            f"clause {clause} claims a pin that does not exist: {node}")
        assert func.startswith("test_"), f"{node} is not a test"


@live_vault
def test_the_map_covers_every_clause_the_item_numbers():
    """Drift in the other direction: `PINNED_CLAUSES` is a transcription, so if the
    item grows or loses a numbered clause the copy above must go red rather than
    keep asserting the old contract. It compares clause NUMBERS only: an amendment
    that rewords a clause in place keeps its number and is not seen here. Reads the
    real backlog item the gate reads its clauses out of."""
    items = sorted((VAULT / "backlog").glob("463-*.md"))
    assert items, "the #463 item file is gone; the transcription has no source"
    text = items[0].read_text(encoding="utf-8")
    m = re.search(r"(?m)^\*\*Acceptance clauses\*\*[^\n]*\n((?:[0-9]+\..*\n?)+)", text)
    assert m, "the item no longer carries an Acceptance clauses block to transcribe"
    numbered = {int(n) for n in re.findall(r"(?m)^(\d+)\.\s", m.group(1))}
    assert numbered == ITEM_CLAUSES, (
        f"the item numbers {sorted(numbered)}; this file pins {sorted(PINNED_CLAUSES)} "
        f"and defers {sorted(FINALIZER_CLAUSES)} — reconcile before grading anything")
