"""A bench task may not enter the corpus without its grading contract (#2286).

Seven items in five weeks — #1589, #1724 (×3), #1968, #2174 (×2), #2228 (×2), #2285 —
each filed when `tests/test_autoresearch_judge.py::
test_every_live_bench_task_has_an_assertion_set` went red because a new
`~/obsidian/lloyd/bench/bench_NN_*.md` arrived with no key in
`eval/autoresearch_assertions.yaml`. Each closed by adding rows. The property was never
closed, because the only thing that enforced it was a test that runs *downstream* of
every writer: `git grep -ln autoresearch_assertions -- '*.py'` returns
`scripts/autoresearch/judge.py` and two test files — nothing on a write path reads the
table. So the corpus gained a task on 2026-10-05 the same way it always had (a
bench-mine candidate promoted into the vault, then swept into git by the nightly
pre-flight's unattributed dirty-state snapshot, vault `dc72ec91`), and the diagnosis
step — six nodes with the same signature, five weeks apart — was the test again.

Why the fix is a rail and not a skill: `grep -rln "lloyd/bench" ~/obsidian/skills
--include=SKILL.md` is a true zero (the same pattern hits three repo files), so no
installed skill owns bench authoring, and an instruction aimed at a corpus an
open-ended worker session writes is the defect the 2026-09-20 class rule names — "a
hand-maintained allowlist cannot close a property that must hold over an open-set
corpus". A refusal the writer cannot skip is the shape that closes it.

**One predicate, every lane.** Four surfaces can land a file under `lloyd/bench/`, and
the 2026-09-22 class rule ("a guard that lives on one of two write surfaces is not a
guard") is why this module is called from all four rather than from the one the item
named:

* `vault_write` — `agent_mcp/vault.py`, `_vault_write`
* `Write` / `Edit` — `agent_mcp/builtin_fs.py`, `_gate_check`
* Bash — `app/harness/protected_paths.py`, `check_bash_write_denied`
* the worker-artifact promotion route — `app/routers/workers.py`, the
  `bench-mine` branch of `/api/workers/pending/promote`, which writes
  `lloyd/bench/<id>.md` with `Path.write_text` and touches none of the tool lanes.
  This is the lane the recurrence actually used: `workers/sources/bench_mine.py`
  hands the spawned session a *prompt* that says to produce a bench task, and the
  promoted artifact is what becomes the file.

"Is this path a bench task file?" is decided by `app.harness.bench_corpus.corpus_target`
— the same realpath resolver the bench-corpus *read* deny uses — so the write rail and
the read deny move together if `autoresearch.bench_dir` ever moves or a second corpus
directory appears. `tests/test_bench_assertion_rail.py` pins both the four refusals and
the invariant that keeps them from drifting apart: every lane that asks the write
deny-set asks this rail too.

**The remedy is worded for the reader who can carry it out (#2718).** One property, four
lanes — but the *missing key* refusal used to be one string for every session that hit
it, and its remedy was "add a `<id>:` key to `eval/autoresearch_assertions.yaml`, then
retry". That is right for the one writer that can edit a tracked file of the code
checkout — an automod round, in its worktree — and it is an instruction a background
session must not receive, because on 2026-10-13 one did receive it and obeyed: session
`20261013_225121_benchmine_2a67` appended 8 lines to the tracked table with a `python3`
heredoc through `Bash`, after the `Write` it had attempted was refused by this very rail.
So `_missing_key` now asks `can_land_table_edit`, which is
`app.harness.live_git_writes.attended_session` plus the round-worktree case, and answers
a session that cannot land the edit with where its rows DO go — the staged draft — and
with an explicit do-not-retry / do-not-edit-the-table-through-another-tool. Both halves
matter and only one of them is this file: the same words aimed at a session that still
holds `Bash` reach the same heredoc, which is why #2718 declares a write-free ceiling for
`bench-mine` in `app/harness/capabilities.py` at the same time.

Deliberately **stricter than the test it pre-empts**: the coverage node exempts an entry
marked `graded: true`, and the rail does not. `graded: true` is the escape hatch the
class of items exists to catch — #1968's `test_bench_023_resolves_to_authored_checks_and_
not_a_graded_marker` and #2174's twin exist precisely to red a mapping that satisfies the
coverage node while leaving the task on the scalar judge — so the writing side requires
what `judge.assertions_for` actually resolves: a non-empty list of `{id, text}` rows.
No task in the table is marked `graded` today, so the stricter bar costs nothing; it
just cannot be quietly widened later.

And deliberately **not** a list of permitted ids: this module reads the table and holds
no exceptions of its own, because an exception list here would be the same
hand-maintained allowlist the item rules out, one file closer to the writer.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

#: A bench task file: `bench_<NN>_<slug>.md`, the shape `load_bench_tasks` globs and
#: `_bench_default_filename` mints. Deliberately anchored and `.md`-only: a scratch
#: note in the same directory (`README.md`, `bench-notes.md`) is author-facing prose,
#: not a graded task, and refusing it would make the rail something a writer routes
#: around.
BENCH_TASK_FILE_RE = re.compile(r"^bench_[0-9]+_[A-Za-z0-9._-]+\.md$")

#: Table path, for the refusal text. Read through `scripts.autoresearch.judge` at call
#: time (that module owns the format); this constant is only what the message names.
TABLE_REL = "eval/autoresearch_assertions.yaml"

_FIRST_BLOCK_ID_RE = re.compile(
    r"^---\s*\n(?:.|\n)*?^id:\s*([A-Za-z0-9._-]+)\s*$", re.MULTILINE)


def frontmatter_id(content: str | None) -> str:
    """The `id:` a written file declares inside its first frontmatter block, or "".

    Only used where a lane has the bytes in hand (the `vault_write` lane). The rail's
    primary key is the file's STEM, because that is what the corpus keys on for the
    lane that matters most: `/api/workers/pending/promote` rewrites
    `task_fm["id"] = dest_path.stem` before it writes, so the landed id *is* the stem.
    A hand-written file whose declared id differs from its stem is graded under the
    declared id (`load_bench_tasks` keys on it), so where the content is visible the
    rail checks both — it is never weaker than the stem test, and it costs the lane
    nothing it does not already parse.
    """
    if not content or not isinstance(content, str):
        return ""
    if not content.lstrip().startswith("---"):
        return ""
    match = _FIRST_BLOCK_ID_RE.search(content)
    return match.group(1) if match else ""


def assertion_key_defect(task_id: str, session_id: str | None = None) -> str:
    """"" when the table carries real assertion rows for `task_id`; else why not.

    `session_id` says who is being told, and only changes the WORDS of the missing-key
    answer (#2718, see `can_land_table_edit`); the verdict itself is the same for every
    session. Left empty it resolves the calling session the same way `record_refusal`
    already does, so no lane has to thread it and no lane can forget it.

    The answer is the judge's own answer — `judge.load_assertions()` for the table and
    `judge.assertions_for()` for the resolution — so the rail cannot disagree with the
    thing it is protecting. A `graded: true` mapping resolves to `None` there, and is a
    defect here, for the reason in this module's docstring.

    Fails closed. `judge.load_assertions` returns `{}` for a missing or unreadable file
    — correct for a judge, which then routes everything to the scalar path and still
    produces a score, and the wrong thing for a guard: an empty table read as "no keys"
    would refuse every bench write with a reason that blames the writer for the state of
    the repo. So the two states are separated, and the one the writer did not cause says
    so and names the path rather than naming a missing key.
    """
    task_id = (task_id or "").strip()
    if not task_id:
        return ""
    try:
        from scripts.autoresearch import judge
    except Exception as exc:  # noqa: BLE001 — an unimportable table is not a table that passed
        return _unreadable(task_id, f"scripts.autoresearch.judge is unimportable ({exc})")
    table_path = str(getattr(judge, "ASSERTIONS_PATH", TABLE_REL))
    try:
        table = judge.load_assertions()
    except Exception as exc:  # noqa: BLE001
        return _unreadable(task_id, f"{table_path} could not be read ({exc})")
    if not isinstance(table, dict) or not table:
        return _unreadable(
            task_id,
            f"{table_path} yielded no assertion keys at all"
            + ("" if os.path.isfile(table_path) else " and is not present at that path"))
    entry = table.get(task_id)
    if entry is None:
        return _missing_key(task_id, can_land_table_edit(session_id))
    try:
        rows = judge.assertions_for({"id": task_id}, table)
    except Exception as exc:  # noqa: BLE001
        return _unreadable(task_id, f"assertions_for({task_id}) raised ({exc})")
    if not rows:
        return (
            f"{task_id} has an entry in {table_path} but it is not an assertion set: "
            f"{_shape(entry)}. The judge scores that on the scalar rubric while every "
            "task with real rows scores binary, which is the fallback this rail exists "
            "to end — replace it with a list of `id:`/`text:` rows (one per rubric "
            "criterion, `conciseness` excepted) rather than a `graded: true` marker.")
    ids = [str(r.get("id")) for r in rows]
    if len(ids) != len(set(ids)):
        return (f"{task_id}: its assertion rows in {table_path} repeat an id ({ids}), "
                "so the binary judge would count one check twice")
    return ""


def bench_write_defect(path: str, cwd: str | None = None,
                       claimed_id: str = "", content: str | None = None,
                       session_id: str | None = None) -> str:
    """"" when writing `path` is allowed; the refusal reason when it is a keyless task.

    `session_id` reaches only the wording of a missing-key refusal; empty resolves the
    calling session. Every lane calls this with the first four arguments and nothing
    else, which is what keeps #2718's branch from being a rule each surface has to
    remember to pass.

    `path` may be vault-relative (`lloyd/bench/…`, as `vault_write` gets it), absolute,
    `~`-prefixed or relative to `cwd` — resolution is `bench_corpus.corpus_target`'s,
    the realpath one the read deny uses, so a symlinked parent and a `cd` cannot walk a
    file in without being seen. A path outside the corpus returns "" and this is never
    reached in the hot case.

    The stem is always checked. `claimed_id` (or `content`, from which the lane's own
    frontmatter id is read) adds the declared id where the bytes are available.
    """
    try:
        from app.harness import bench_corpus
    except Exception as exc:  # noqa: BLE001 — fail closed, for the reason above
        return _unreadable("(unresolved bench path)",
                           f"app.harness.bench_corpus is unimportable ({exc})")
    # Name before place, and in that order. The Bash lane asks this for EVERY write
    # target of EVERY command, and `corpus_target` reads config.yaml to learn where the
    # corpus is — measured here at ~70 ms per call, so on that hot path a command with a
    # few targets would pay fractions of a second for a check that cannot apply to it.
    # One regex match answers for all of them, and the ordering costs nothing in
    # correctness: a bench task is by definition a file named `bench_NN_*.md`, so a target
    # failing this test is not a task wherever it lives.
    if not BENCH_TASK_FILE_RE.match(os.path.basename(path.rstrip("/"))):
        return ""
    real = bench_corpus.corpus_target(path, cwd)
    if not real:
        return ""
    name = os.path.basename(real)
    if not BENCH_TASK_FILE_RE.match(name):
        return ""
    stem = name[:-len(".md")]
    defect = assertion_key_defect(stem, session_id)
    if defect:
        return defect
    extra = (claimed_id or "").strip() or frontmatter_id(content)
    if extra and extra != stem:
        return assertion_key_defect(extra, session_id)
    return ""


#: One path component of an automod round's tree: `~/lloyd-work/SM_20261014_190755/…`,
#: the layout `scripts/automod/worktree.py:round_dir` dictates. Matched on the component
#: and not on `Path.home()/lloyd-work` because a check that runs inside a round sees a
#: redirected `HOME` (`worktree.round_home`'s own docstring: that redirection is the
#: whole isolation model), and the component shape is the same either way.
_ROUND_DIR_RE = re.compile(r"(^|/)lloyd-work/SM_[0-9]{8}_[0-9]{6}(/|$)")


def _round_worktree_session(sid: str,
                            sessions_dir: "os.PathLike | str | None" = None) -> bool:
    """Whether this session started inside a self-modification round's worktree.

    Read off the session record's `start_cwd` through `app.session_cwd.read`, the
    existing reader for that key, because the round's own state
    (`~/.local/state/lloyd-automod/rounds/`) names a worktree and not a session, and a
    refusal is raised in a tool process that knows only the session id.
    `scripts/automod/round.py:155` stamps the round's turn with `directory=<worktree>`,
    so the recorded start directory is `…/lloyd-work/SM_<ts>/home/lloyd` (#2718). That
    reader is also what sanitises the id and drops a value pointing inside the live
    checkout, so this predicate inherits both instead of re-implementing either;
    `sessions_dir` is its test seam.
    """
    if not sid or ":" in sid:
        return False
    try:
        from app import session_cwd
        cwd = session_cwd.read(sid, sessions_dir=sessions_dir)
    except Exception:  # noqa: BLE001 — an unreadable record is not proof of a round
        return False
    return bool(_ROUND_DIR_RE.search(cwd or ""))


def can_land_table_edit(session_id: str | None = None) -> bool:
    """Whether the session being told may itself edit `TABLE_REL`, and so should be told to.

    Two cases, and one predicate each. **Attended**, from
    `app/harness/live_git_writes.attended_session` — that predicate and no other, so the
    rail does not grow a second notion of who is a person (#2718's own triage named the
    reuse; `agent_mcp/_tool_sandbox` was rejected for the same reason). **An automod
    round**, which edits the table in its own worktree and is the lane the whole
    self-modification loop exists to serve; a round has to be told the file's name or it
    cannot author the rows at all.

    Everyone else — a worker turn, an autonomy job, an MCP aggregator session, a
    colon-prefixed subagent — gets the answer that sends its assertion rows out with the
    draft instead of into a tracked file. On a failure to decide, that is the answer too:
    the branch that never tells a session to open a tracked file of the code checkout is
    the one to take when the predicate itself could not run.
    """
    sid = str(session_id or "").strip()
    if not sid:
        try:
            from agent_mcp._shared import get_bound_session
            sid = str(get_bound_session() or "").strip()
        except Exception:  # noqa: BLE001 — no bound session is the unattended case
            sid = ""
    try:
        from app.harness.live_git_writes import attended_session
        if attended_session(sid):
            return True
    except Exception:  # noqa: BLE001 — see the docstring: fail to the safe wording
        logger.debug("bench_assertion_rail: attended_session unavailable",
                     exc_info=True)
        return False
    return _round_worktree_session(sid)


def _missing_key(task_id: str, can_land: bool) -> str:
    """The refusal, written to survive its own transport and to fit its reader.

    `_shared._err` cuts a tool error at 500 characters, and the actionable half is the
    half that must not be cut: a refusal that loses its `id:`/`text:` shape sends the
    writer back to the test file to work out what was asked, which is the diagnosis step
    this rail replaces. So the table is named by its repo-relative path (`TABLE_REL`, 33
    characters against the 59 of this box's absolute one, and unambiguous because there
    is one checkout), and both wordings are measured against the transport rather than
    assumed short: substituting the longest task id in the real bench corpus
    (`bench_025_skill_invocation_never_ran_resume_chain`, 49 characters) gives 463
    characters for the attended answer and 416 for the background one, both under the
    cut. The node `test_neither_wording_is_cut_from_the_actionable_half`, in
    `tests/test_bench_assertion_rail.py`, re-measures both.

    Two wordings, one verdict (#2718). The table-naming half is kept verbatim for the
    sessions that can act on it, because re-wording it would cost the only lane that can
    land the edit its actionable instruction. For everyone else the same sentence is the
    hazard: on 2026-10-13 a bench-mine turn read it as permission to edit the tracked
    table and did, through `Bash`, after its `Write` was refused here. So the background
    wording names no path, gives no "add a key and retry", says where the rows DO go —
    the staged draft's own body, which is what a round later reads — and closes the
    "find another lane" door the transcript showed the model walking through.
    """
    if not can_land:
        return (
            f"{task_id} has no assertion set, and this session cannot add one: the "
            "assertion table is a tracked file of the code checkout, which only an "
            "automod round may edit. Put the rows in the staged draft's own body as "
            "`- id: <check-id>` / `text: <one falsifiable check>` lines and a round "
            "lands them from there. Do not retry this write, and do not edit the table "
            "through another tool.")
    return (
        f"{task_id} has no assertion set in {TABLE_REL}: an unkeyed task scores on the "
        "SCALAR rubric while every keyed task scores binary, and the two are averaged "
        "into one promotion decision. Add a "
        f"`{task_id}:` key holding `id:`/`text:` rows — one per rubric criterion except "
        "`conciseness`, NOT `graded: true`, which passes coverage and still scores "
        "scalar — then retry.")


def _unreadable(task_id: str, why: str) -> str:
    return (
        f"{TABLE_REL} could not be checked for {task_id}: {why}. The bench task is not "
        "being written, because a rail that cannot read its own input has no verdict to "
        "give. Report this rather than working around it.")


def _shape(entry: object) -> str:
    if isinstance(entry, dict):
        return f"a mapping with keys {sorted(entry)}"
    return f"a {type(entry).__name__}"


def record_refusal(where: str, tool: str, path: object, reason: str) -> None:
    """One durable denial row per refusal, never a decision of its own.

    Same convention and same swallow as the protected-path refusals in
    `agent_mcp/builtin_fs.py` and `agent_mcp/vault.py`: the journal is how anyone finds
    out a writer was stopped, and a journal that is down must not become a second
    verdict.
    """
    try:
        from app.harness import denial_journal
        session_id = ""
        try:
            from agent_mcp._shared import get_bound_session
            session_id = get_bound_session() or ""
        except Exception:  # noqa: BLE001 — the journal row is still worth having
            pass
        denial_journal.record(guard="bench_assertion_rail", where=where,
                              session_id=session_id, tool=tool,
                              reason=reason[:600], excerpt=str(path),
                              label="bench task without an assertion-set key")
    except Exception:  # noqa: BLE001
        logger.debug("bench_assertion_rail: denial journal unavailable", exc_info=True)
