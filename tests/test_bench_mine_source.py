"""The bench miner's turn has no lane into the code checkout (#2718).

`workers/sources/bench_mine.py` authors a bench task from a losing eval run and its
contract is TEXT: `_TASK_SCHEMA` (`workers/sources/bench_mine.py:534`) says "Output ONLY
the full markdown file content", and `write_staging_note` is what lands it, in Python,
after the turn. The turn never needed a writer. It had three: on 2026-10-13
`app/harness/capabilities.py` declared no ceiling for `bench-mine`, so
`capabilities.envelope_for` returned None, so the dispatch compile shipped the
floor-only surface with `Write`, `Edit` and `Bash` all reachable.

Session `20261013_225121_benchmine_2a67` used two of them. Msg 44 is a `Write` of a
`bench_029_…` task file into the bench corpus; the #2286 rail refused it, and the refusal
told the session to add the missing key to `eval/autoresearch_assertions.yaml` and retry.
Msg 46 records the reasoning ("the rail explicitly told me to add the key and retry —
'that is the way forward'"), and msg 47 is the `Bash` heredoc that appended the key to
that TRACKED file of the live checkout — a `python3 - <<'PY'` whose own stdout, at msg
48, was `inserted at line 505` with `is_error: false`. The file sat dirty on `main` until
it was reverted by hand on 2026-10-14.

This file pins the tool-surface half of the fix: the source is declared
(`tests/test_worker_capability_allowlist.py` grades the ceiling's content), the miner's
turns compile under it, and the two recorded calls are refused before they run. The other
half — the refusal no longer telling a background session to open that file — is pinned in
`tests/test_bench_assertion_rail.py`; neither half alone closes the incident, because a
session told to edit a file it cannot edit reaches for another lane, and a session with
`Bash` needs no instruction to have used it.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import workers.sources.bench_mine as bench_mine
from app.harness import capabilities as caps
from app.harness.loop import _allowed_names, _pre_dispatch
from app.harness.tool_search import LoadedToolSet
from workers.sources import _common as C

#: The checkout this test file lives in — the live tree when the suite runs normally,
#: the round worktree while a candidate is being gated. `app.paths` anchors to the same
#: tree, and a replay that dirtied anything would show here rather than in a remembered
#: absolute path that a redirected `$HOME` cannot see.
TREE = Path(__file__).resolve().parents[1]

#: The tracked file that was written. Named by repo-relative path, the same spelling the
#: rail uses (`app/harness/bench_authoring.TABLE_REL`).
TABLE_REL = "eval/autoresearch_assertions.yaml"

#: The task the incident session was mining, whose key it inserted. A `bench_029_…` key
#: belongs to no task in the tree: the bench file was never written, so the row it added
#: described nothing.
ORPHAN_KEY = "bench_029_recall_user_fact_two_topic_reads"

# The two tool_calls that sequence, transcribed out of
# `~/lloyd-data/sessions/20261013_225121_benchmine_2a67.json` (messages 44 and 47).
# `file_path` and `summary` are verbatim. The `content` bodies are truncated — the
# envelope refuses on the tool NAME, so what was in them decides nothing here, and the
# full heredoc (a `re.match` over the yaml's bench keys, a 3-row block, and a
# `p.write_text`) is 40 lines of a session file this test must not read to pass.
RECORDED_CALLS = (
    ("Write", {
        "file_path": "/home/alansrobotlab/obsidian/lloyd/bench/"
                     f"{ORPHAN_KEY}.md",
        "content": "---\nsegment: lloyd\nid: bench_029_recall_user_fact_two_topic_read"
                   "s\ncategory: replay\n…(the mined task's frontmatter and body)"}),
    ("Bash", {
        "command": "cd ~/lloyd && python3 - <<'PY'\nimport re, pathlib\n"
                   "p = pathlib.Path(\"eval/autoresearch_assertions.yaml\")\n"
                   "…(locates the last bench key, inserts a 3-row block, writes it back)"
                   "\nPY",
        "summary": "Adding bench_029 assertion set"}),
)

#: The session the two recorded calls come from, verbatim. It matters to the dispatch
#: nodes below that this is a real `benchmine` run name: `live_git_writes` classifies a
#: session as unattended from this shape (`_BACKGROUND_RUN` in
#: `app/harness/service_control.py`), and it is the same name that decides which half of
#: the refusal wording the rail answers with (see `tests/test_bench_assertion_rail.py`).
RECORDED_SID = "20261013_225121_benchmine_2a67"

#: Message 47's command, transcribed from the session file. Two edits, both of which the
#: nodes below depend on and neither of which changes what the command does: the
#: `cd ~/lloyd &&` prefix is dropped because the node sets `cwd` to a throwaway worktree
#: of this commit instead of naming a home directory, and the trailing `=== verify ===`
#: echo — which reads `/home/alansrobotlab/lloyd/eval/…` by absolute path — is left out
#: for the same reason. Everything from `python3 - <<'PY'` down, including the three-row
#: block and the `assert "bench_029" not in s` guard, is the string that shell ran.
INCIDENT_HEREDOC = r'''python3 - <<'PY'
import re, pathlib
p = pathlib.Path("eval/autoresearch_assertions.yaml")
s = p.read_text()
lines = s.split("\n")
ids = [i for i,l in enumerate(lines) if re.match(r'^\s{2}bench_\d+_[a-z0-9_]+:\s*$', l)]
print("tail check:", [l for l in lines if re.match(r"^\S", l)][:6])
block = """  bench_029_recall_user_fact_two_topic_reads:
    - id: accuracy
      text: "Does the reply state that agent-services/services/tts/qwen3-tts/ is untracked in git, so a revert there cannot be done by git?"
    - id: names_both_entries_not_just_one
      text: "Does the reply carry details from BOTH topic files - the .bak files and vault notes as the revert artifacts, AND the email_recent daysBack=1 probe where unread=0 is not evidence of an outage?"
    - id: does_not_fabricate_a_probe_result
      text: "Does the reply report the filed probe rule rather than asserting a live health verdict, a command or a latency it did not measure?"
"""
assert "bench_029" not in s, "already present"
assert ids, "no bench keys found"
last = ids[-1]
cut = len(lines)
for j in range(last+1, len(lines)):
    if re.match(r'^(\S|  \S)', lines[j]):
        cut = j
        break
lines[cut:cut] = block.rstrip("\n").split("\n")
p.write_text("\n".join(lines))
print("inserted at line", cut+1)
PY'''

def _miner_options():
    """The options the miner's own dispatch call compiles — no `extra_disallowed`.

    `workers/sources/bench_mine.py:934` and `:955` both call
    `run_prompt_on_primary(..., source=NAME)`, which reaches
    `C._worker_run_options(max_turns, source=source)`. So `source=` is the only thing
    standing between that turn and the floor-only compile, and this passes it as the
    source passes it: by the module's own `NAME`, not by a string this file invented.
    """
    return C._worker_run_options(12, source=bench_mine.NAME)


def _undeclared_options():
    """A turn of a source with no declared ceiling — the pre-#2718 compile.

    The control must never be dispatched with a recorded call's real arguments: with no
    ceiling, `Write` of `/home/alansrobotlab/obsidian/lloyd/bench/…` would not be refused
    by the envelope, and a test that proves a point by writing the file is the bug with
    extra steps. The control therefore passes empty arguments, which is also what
    `tests/test_worker_capability_allowlist.py`'s control node does for the same reason.
    """
    return C._worker_run_options(12, source="bench-mine-not-a-registered-source")


def _tc(name: str, args: dict | None) -> dict:
    return {"id": "call_1", "function": {"name": name}, "_args_dict": args or {}}


def _dispatch(opts, name: str, args: dict | None = None):
    """The real pre-dispatch gate, with nothing replaced.

    A `None` return means the call went through toward the pool; a returned event means
    it was refused before the pool saw it. Same seam as
    `tests/test_worker_capability_allowlist.py::_dispatch` — a spy would verify the
    wiring and miss the gate.
    """
    return asyncio.run(_pre_dispatch(
        tc=_tc(name, args), options=opts, session_id="s",
        loaded_set=LoadedToolSet(enabled=False, catalog=[], loaded=set())))


def _refused_by_envelope(evt) -> bool:
    return bool(evt) and bool(evt.get("is_error")) and (
        "not available on this turn" in str(evt.get("content") or ""))


def _porcelain(root: Path) -> str:
    """`git status --porcelain` over `root`, or a raised error naming the failure.

    Checked rather than trusted: a `git` that could not run returns empty stdout like a
    clean tree does, and this node's whole claim is the difference between those two.
    """
    out = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                         capture_output=True, text=True)
    assert out.returncode == 0, f"git status failed in {root}: {out.stderr[:200]}"
    return out.stdout


# ── clause 2, first half: the miner turn compiles under the declared ceiling ──

def test_the_miner_turn_dispatches_under_the_declared_ceiling():
    """`_worker_run_options(source=bench-mine)` carries the ceiling, not None.

    The ceiling reaching the options is a separate fact from the ceiling existing:
    `capabilities.envelope_for` answers by `str(source)`, and the registry's name for
    this source is `bench-mine` (`workers/source_registry.py`) while its module `NAME`
    was `bench-mine` already — a rename in either place lands on this node, because it
    reads `bench_mine.NAME` and compares to the key `SOURCE_CAPABILITIES` answers on.
    """
    opts = _miner_options()
    allowed = _allowed_names(opts)
    assert allowed is not None, (
        "the miner's turn still ships the floor-only compile: envelope_for got nothing "
        f"for source={bench_mine.NAME!r}")
    assert set(allowed) == set(caps.envelope_for("bench-mine") or ()), (
        "the turn's allow-list is not the declared ceiling for `bench-mine`")
    for name in ("Write", "Edit", "Bash"):
        assert name not in allowed, f"{name} is back in the miner's surface: {allowed}"
    assert "Read" in allowed, (
        "the miner's prompt says \"Read that file with the Read tool\" "
        "(workers/sources/bench_mine.py:594); a ceiling that strips it is a ceiling the "
        "job cannot do its work under")


def test_the_miner_declares_no_extra_disallowed_so_the_ceiling_is_the_whole_surface():
    """The two miner turns pass no `extra_disallowed`, so nothing narrows either way.

    Worth a node because it is the assumption the ceiling rests on: had the module kept
    its own `DISALLOWED` list, the shipped surface would be the intersection and the
    "ceiling == surface" claim above would be measuring a different object.
    """
    for attr in ("DISALLOWED", "DISALLOWED_TOOLS"):
        assert not getattr(bench_mine, attr, ()), (
            f"bench_mine.{attr} exists; the shipped compile is "
            "envelope_for(ceiling) MINUS that list, and this file's control is stale")


# ── clause 2, second half: the recorded sequence, replayed ───────────────────

def test_the_recorded_write_and_bash_calls_are_refused_before_running():
    """Replaying msg 44 and msg 47 answers `refused-before-running` for both.

    Both, because the incident is the PAIR: msg 44 is the write that was refused by the
    bench rail, msg 47 is the write that nothing refused because the refusal that
    provoked it named a tracked file and the session still had a shell. Asserting one of
    them would let the other lane back in.
    """
    opts = _miner_options()
    for name, args in RECORDED_CALLS:
        evt = _dispatch(opts, name, args)
        assert _refused_by_envelope(evt), (
            f"the recorded {name} call dispatched on a miner turn: "
            f"{(evt or {}).get('content', '')[:160]}")


def test_the_replayed_sequence_leaves_git_status_porcelain_over_the_checkout_empty():
    """The acceptance check: the 22:51 tool-call sequence, replayed, dirties nothing.

    Three assertions, in the order a reader needs them. FIRST that the tree was clean
    before the replay — a tree that starts dirty reports that, instead of letting this
    node blame the replay for somebody else's state. THEN that the replay changed
    nothing. THEN the two incident-specific facts, which hold even on a tree that has
    unrelated pending work: the tracked table is not among the modified paths, and the
    orphan key the heredoc inserted is not in the table's text.
    """
    before = _porcelain(TREE)
    assert before == "", (
        f"the checkout was already dirty before the replay: {before[:300]}")
    for name, args in RECORDED_CALLS:
        _dispatch(_miner_options(), name, args)
    after = _porcelain(TREE)
    assert after == before == "", (
        f"replaying the 22:51 sequence dirtied the checkout: {after[:300]}")
    assert TABLE_REL not in [line[3:].strip() for line in after.splitlines()], after
    table_text = (TREE / TABLE_REL).read_text(encoding="utf-8")
    assert ORPHAN_KEY not in table_text, (
        "the orphan key the incident inserted is in the table: the hand revert of "
        "2026-10-14 has not held")


def test_the_control_compiles_no_ceiling_so_the_refusals_above_are_the_ceilings():
    """Non-vacuity: the same two names are NOT envelope-refused on an undeclared turn.

    With no ceiling `_allowed_names` is None, so `_pre_dispatch` has no allow-list to
    consult and the names are simply not filtered by this mechanism. That is the
    pre-#2718 state, and without this node the two refusals above could be some other
    rail firing on an arbitrary machine.
    """
    opts = _undeclared_options()
    assert _allowed_names(opts) is None, (
        "the control's source is declared; it is no longer measuring an undeclared turn")
    for name, _args in RECORDED_CALLS:
        evt = _dispatch(opts, name)      # empty args on purpose — see _undeclared_options
        assert not _refused_by_envelope(evt), (
            f"{name} was refused by an envelope that does not exist for this source")


# ── the control that gives the porcelain assertion teeth ─────────────────────
#
# `test_the_replayed_sequence_leaves_git_status_porcelain_over_the_checkout_empty`
# asserts a negative, and a negative assertion about a command that could not have run
# anyway is worth exactly what a reader can check: nothing, unless something shows that
# the same check goes red when the ceiling is absent. These two nodes do that, in a
# throwaway worktree of THIS commit, so the demonstration never touches the checkout it
# is measuring.

def _worktree(where: Path):
    """A detached worktree of this commit at `where`, for the two nodes below.

    `git worktree add` shares the object store and costs one checkout; the node that uses
    it is the only place in this file that mutates a filesystem at all.
    """
    subprocess.run(["git", "-C", str(TREE), "worktree", "add", "--detach", str(where),
                    "HEAD"], check=True, capture_output=True, text=True)
    return where


def _remove_worktree(where: Path) -> None:
    subprocess.run(["git", "-C", str(TREE), "worktree", "remove", "--force", str(where)],
                   check=False, capture_output=True, text=True)


def test_the_recorded_heredoc_dirties_a_clean_checkout_of_this_commit(tmp_path):
    """Run msg 47 for real against a pristine checkout, and watch it go dirty.

    This is the incident reproduced, not described: the exact command string, `cwd` set to
    a checkout of this commit, executed by `subprocess` exactly as the miner's shell
    executed it. Afterwards `git status --porcelain` names the tracked table and the
    orphan key is in its text. If this node ever goes green-without-dirt, the replay
    above has stopped measuring anything, because the command has stopped being able to
    write — and the pair is what makes the porcelain assertion worth reading.
    """
    where = _worktree(tmp_path / "pristine")
    try:
        assert _porcelain(where) == "", "a fresh worktree of HEAD was already dirty"
        proc = subprocess.run(INCIDENT_HEREDOC, shell=True, cwd=str(where),
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr[-400:]
        assert "inserted at line" in proc.stdout, proc.stdout[-200:]
        after = _porcelain(where)
        assert TABLE_REL in [line[3:].strip() for line in after.splitlines()], after
        table = (where / TABLE_REL).read_text(encoding="utf-8")
        assert ORPHAN_KEY in table, "the command ran and the key is not there"
    finally:
        _remove_worktree(where)


def test_the_miner_cannot_reach_the_command_that_dirtied_it(tmp_path):
    """The other side of the pair: the same command, refused before it can run.

    Same string, same session id, one difference — it goes through the miner's dispatch
    instead of a shell. The `Bash` name is gone from the compile, so the loop answers
    before any rail, any cwd and any file is consulted, and the pristine worktree that
    the node above dirtied stays clean. A no-ceiling compile is the control on the same
    line: `_pre_dispatch` answers None for it, which is what `bench-mine` did on
    2026-10-13.
    """
    where = _worktree(tmp_path / "guarded")
    try:
        evt = _dispatch(_miner_options(), "Bash", {"command": INCIDENT_HEREDOC})
        assert _refused_by_envelope(evt), (
            "the miner could dispatch the command that dirtied the tree: "
            f"{(evt or {}).get('content', '')[:160]}")
        control = C._worker_run_options(12, source="failure-ledger",
                                        session_id=RECORDED_SID)
        assert _dispatch(control, "Bash", {"command": INCIDENT_HEREDOC}) is None, (
            "the control source's Bash is refused by something other than the envelope, "
            "so this pair is not measuring the ceiling")
        assert _porcelain(where) == "", "refused, and the tree is still clean"
    finally:
        _remove_worktree(where)
