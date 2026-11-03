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

import ast
import asyncio
import io
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest
import yaml
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
#: This file, relative to a checkout — how a child pytest is addressed.
TARGET_REL = "tests/test_bench_mine_source.py"

#: The task the incident session was mining, whose key it inserted. A `bench_029_…` key
#: belongs to no task in the tree: the bench file was never written, so the row it added
#: described nothing.
ORPHAN_KEY = "bench_029_recall_user_fact_two_topic_reads"
#: The incident's bench NUMBER on its own — the string the recorded guard used to compare
#: whole keys against. Derived from `ORPHAN_KEY` rather than written out, because #3731 bans it
#: as a literal in this file (`test_the_guard_is_keyed_on_the_whole_orphan_key_not_a_prefix`)
#: and a checker that carries its own copy of the banned value would trip on itself.
BENCH_NUMBER = "_".join(ORPHAN_KEY.split("_")[:2])

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

#: Message 47's command, transcribed from the session file. Three edits, and the first
#: two change nothing about what the command does: the `cd ~/lloyd &&` prefix is dropped
#: because the node sets `cwd` to a throwaway worktree of this commit instead of naming a
#: home directory, and the trailing `=== verify ===` echo — which reads
#: `/home/alansrobotlab/lloyd/eval/…` by absolute path — is left out for the same reason.
#: The third edit is deliberate and is the subject of item #3731: the guard reads
#: `assert "bench_029_recall_user_fact_two_topic_reads" not in s` where the shell ran a
#: bare `"bench_029"` prefix. As transcribed, the guard refused any table carrying a key
#: that merely *starts* with the incident's number, so promoting an unrelated
#: `bench_029_*` assertion set — which the miner is still told to pick a number it cannot
#: know, and #3688's durable-write card is exactly that — made this node abort on a
#: command that would have written the table perfectly well. `ORPHAN_KEY` above is now
#: spelled out in the guard, which is the same string; the prefix form was the defect, and
#: `test_the_guard_is_keyed_on_the_whole_orphan_key_not_a_prefix` keeps it from creeping
#: back. Everything else from `python3 - <<'PY'` down, including the three-row block, is
#: the string that shell ran.
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
assert "bench_029_recall_user_fact_two_topic_reads" not in s, "already present"
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


def _assert_replay_dirties_the_table(where: Path) -> None:
    """Run the recorded command in `where` and prove it wrote the tracked table.

    Four proofs, in this order: the shell exited 0, its stdout printed `inserted at line`,
    `git status --porcelain` names the tracked table, and the orphan key is in the table's
    text. The node below and the two #3731 nodes further below all call *this* function, so
    "the node still proves the write" is one body of assertions on every checkout, not a
    copy that a later edit can quietly weaken in one of them.

    It takes an `assert`-shaped helper rather than returning a bool because the failure has
    to name which of the four proofs broke: `returncode` failing means the command stopped
    running, `inserted at line` failing means it ran and wrote nothing, and a porcelain miss
    means it wrote something git is not seeing.
    """
    proc = subprocess.run(INCIDENT_HEREDOC, shell=True, cwd=str(where),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "inserted at line" in proc.stdout, proc.stdout[-200:]
    after = _porcelain(where)
    assert TABLE_REL in [line[3:].strip() for line in after.splitlines()], after
    table = (where / TABLE_REL).read_text(encoding="utf-8")
    assert ORPHAN_KEY in table, "the command ran and the key is not there"


def test_the_recorded_heredoc_dirties_a_clean_checkout_of_this_commit(tmp_path):
    """Run msg 47 for real against a pristine checkout, and watch it go dirty.

    This is the incident reproduced, not described: the exact command string, `cwd` set to
    a checkout of this commit, executed by `subprocess` exactly as the miner's shell
    executed it. Afterwards `git status --porcelain` names the tracked table and the
    orphan key is in its text. If this node ever goes green-without-dirt, the replay
    above has stopped measuring anything, because the command has stopped being able to
    write — and the pair is what makes the porcelain assertion worth reading.

    The checkout is a worktree of HEAD, so this node follows whatever the table at HEAD
    holds; from #3731 onward that table may carry a `bench_029_*` key, and
    `_assert_replay_dirties_the_table` is shared with the nodes that pin what happens then.
    """
    where = _worktree(tmp_path / "pristine")
    try:
        assert _porcelain(where) == "", "a fresh worktree of HEAD was already dirty"
        _assert_replay_dirties_the_table(where)
    finally:
        _remove_worktree(where)


# ── #3731: the same replay against a checkout whose table already carries a bench_029_* key
#
# The guard inside the recorded command read `assert "bench_029" not in s` — a bare prefix.
# The first assertion set that merely *shares the incident's number* therefore made the
# command abort with `already present`, and the node above went red on a checkout that had
# done nothing wrong. That is how #3688's clause 4 stalled: `bench_029_durable_write_vault_note`
# cannot be promoted to the table without reddening this node, and the rows must be promoted
# first, because the vault-write rail reads the LIVE checkout's table
# (`app/harness/bench_authoring.py:145` → `scripts/autoresearch/judge.py:123` → `LLOYD_HOME`),
# not the round's worktree. And the miner is still told to pick a bench number it cannot
# know, so the next mined card is likely to collide again: keying on the prefix was the
# defect, not merely an obstacle.
#
# These nodes are the pair to the one above. The write still happens on a table carrying an
# unrelated `bench_029_*` set, the refusal survives for the key it actually guards, and the
# guard's shape is pinned so the prefix cannot creep back.
#
# `_checkout_carrying` refuses to build a synthetic checkout around a key already at HEAD, so
# when #3688's rows are promoted for real these nodes fail loudly naming that, rather than
# quietly passing on a table they no longer control. That is the intended end for two of them:
# the state they were written to unblock stops existing, and whoever promotes the rows narrows
# or retires them in the same commit.

#: The three sets #3688 clause 4 needs promoted. The first collides with the incident's
#: number; the other two never did, which is the correction on this item's headline.
DURABLE_WRITE_KEYS = (
    "bench_029_durable_write_vault_note",
    "bench_030_durable_write_kg_row",
    "bench_031_durable_write_backlog_card",
)

#: Nodes in this file that answer pytest by running pytest. `test_the_whole_file_is_green…`
#: deselects them by name or the recursion has no bottom, and every child process carries
#: `BENCH_MINE_REPLAY_NO_SPAWN=1`, which makes them skip rather than spawn again — so a
#: stale name in this tuple surfaces as an unexpected `skipped` in the child's own summary
#: line, which the parent asserts against, instead of as a fork bomb under the gate.
#: The eight nodes this file had before #3731 added six — the pinned replay node among them.
#: Clause 4 of #3731 is stated against THESE eight ("8 passed ... on a table without the three
#: sets and 8 passed with them committed"), so it is checked by name and by number, not by
#: whatever size the file happens to have grown to.
PREEXISTING_NODES = (
    "test_the_miner_turn_dispatches_under_the_declared_ceiling",
    "test_the_miner_declares_no_extra_disallowed_so_the_ceiling_is_the_whole_surface",
    "test_the_recorded_write_and_bash_calls_are_refused_before_running",
    "test_the_replayed_sequence_leaves_git_status_porcelain_over_the_checkout_empty",
    "test_the_control_compiles_no_ceiling_so_the_refusals_above_are_the_ceilings",
    "test_the_recorded_heredoc_dirties_a_clean_checkout_of_this_commit",
    "test_the_miner_cannot_reach_the_command_that_dirtied_it",
    "test_neither_mining_prompt_asks_the_turn_to_choose_a_bench_number",
)


SPAWNS_PYTEST = (
    "test_the_pinned_replay_node_passes_on_a_checkout_carrying_the_promoted_sets",
    "test_the_whole_file_is_green_before_and_after_the_promotion",
)


def _set_block(key: str) -> str:
    """`key` as an assertion set in the shape the miner appends: one key line, `- id:` rows."""
    return (f"  {key}:\n"
            f"    - id: accuracy\n"
            f'      text: "Does the reply carry the sentinel {key}?"\n')


def _checkout_carrying(tmp_path, name: str, *keys: str) -> Path:
    """A worktree of HEAD whose COMMITTED table carries `keys`.

    Committed, not merely written, for two reasons. The replay asserts nothing about
    cleanliness itself, but its sibling node does, and more to the point the state #3731 asks
    about is a *promoted* row — one a later round inherits at HEAD — so the synthetic checkout
    has to be one commit deep or it is testing a dirty working tree instead of the tree a
    round will actually sit on.

    The `yaml.safe_load` check is not decoration. The recorded command finds its insertion
    point by regex over text, so it would report `inserted at line` against a block whose
    YAML is malformed, and a green replay over a table the judge cannot load would prove
    nothing about the promotion these nodes stand in for.
    """
    # A key already at HEAD is left alone, not refused: the state is "a checkout whose
    # committed table carries this key", and #3688's real promotion supplies that as well as a
    # synthetic append does. The presence check at the end is what the node needs, so the
    # helper is correct on both sides of the promotion.
    where = _worktree(tmp_path / name)
    table = where / TABLE_REL
    text = table.read_text(encoding="utf-8")
    missing = [k for k in keys if k not in text]
    if not missing:
        return where
    table.write_text(text.rstrip("\n") + "\n\n" + "".join(_set_block(k) for k in missing),
                     encoding="utf-8")
    loaded = yaml.safe_load(table.read_text(encoding="utf-8"))
    for key in keys:
        assert key in loaded["tasks"], f"{key} is not a real assertion set after the append"
    subprocess.run(["git", "-C", str(where), "add", TABLE_REL],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(where), "-c", "user.name=Lloyd",
                    "-c", "user.email=lloyd@localhost", "commit", "-qm",
                    "verify: promoted assertion sets (#3731)"],
                   check=True, capture_output=True, text=True)
    assert _porcelain(where) == "", "the synthetic checkout was not clean after its own commit"
    return where


def _commit(where: Path, message: str) -> None:
    """Commit everything in a synthetic checkout, as the promotion this node mimics would."""
    subprocess.run(["git", "-C", str(where), "add", "-A"], check=True,
                   capture_output=True, text=True)
    subprocess.run(["git", "-C", str(where), "-c", "user.name=Lloyd",
                    "-c", "user.email=lloyd@localhost", "commit", "-qm", message],
                   check=True, capture_output=True, text=True)


def _strip_key_block(table: Path, key: str) -> None:
    """Delete `key`'s assertion set from `table`, with the comment block above it.

    A set's provense travels with it, so leaving the comment behind would leave a table that
    talks about a set it no longer has. The end is the next key line or the next top-level line,
    whichever comes first.
    """
    lines = table.read_text(encoding="utf-8").splitlines(keepends=True)
    at = next((i for i, ln in enumerate(lines) if ln.rstrip("\n") == f"  {key}:"), None)
    assert at is not None, f"{key} is not a key line in the table"
    first = at
    while first > 0 and lines[first - 1].lstrip().startswith("#"):
        first -= 1
    last = at + 1
    while last < len(lines) and not re.match(r"^  \S", lines[last]) \
            and not re.match(r"^\S", lines[last]):
        last += 1
    del lines[first:last]
    table.write_text("".join(lines), encoding="utf-8")


def _checkout_lacking(tmp_path, name: str, *keys: str) -> Path:
    """A worktree of HEAD whose COMMITTED table does NOT carry `keys`, however HEAD got there.

    The pair to `_checkout_carrying`. The two directions are not symmetric: appending a key is
    always possible when HEAD lacks it, but "the table without these keys" stops being
    constructible by NOT doing something once a promotion lands. So this helper removes the
    blocks rather than hoping they are absent, which is what keeps
    `test_the_whole_file_is_green_before_and_after_the_promotion` a real comparison across
    #3688's promotion instead of a node that goes red the day it lands.
    """
    where = _worktree(tmp_path / name)
    table = where / TABLE_REL
    before = len(yaml.safe_load(table.read_text(encoding="utf-8"))["tasks"])
    present = [k for k in keys
               if k in yaml.safe_load(table.read_text(encoding="utf-8"))["tasks"]]
    for key in present:
        _strip_key_block(table, key)
    loaded = yaml.safe_load(table.read_text(encoding="utf-8"))
    for key in present:
        assert key not in loaded["tasks"], f"{key} survived the strip"
    assert len(loaded["tasks"]) == before - len(present), (
        f"the strip took {before - len(loaded['tasks'])} sets when asked for {len(present)}: it "
        "removed a neighbour, and a table missing an unrelated set is not the state under test")
    if _porcelain(where):
        _commit(where, f"test(#3731): strip {', '.join(present)}")
    assert _porcelain(where) == "", "the strip commit left the synthetic checkout dirty"
    return where


def _run_pytest(where: Path, *targets: str) -> subprocess.CompletedProcess:
    """Run pytest inside the synthetic checkout `where`, serially, with no cache provider.

    `-p no:cacheprovider` because `.pytest_cache` is not gitignored here: a child that
    wrote one would leave its own checkout dirty, and the node under test would then fail
    on the child's bookkeeping rather than on the table. Serial by default — `pytest.ini`
    sets no `addopts`, so the `-n 8` the gate passes is not inherited.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets],
        cwd=str(where), capture_output=True, text=True, timeout=900,
        env={**os.environ, "BENCH_MINE_REPLAY_NO_SPAWN": "1"})


def _child_env_guarded(node_name: str) -> None:
    """Skip in a child run, so a pytest-spawning node can never recurse.

    Reached only through `_run_pytest`, which sets the variable. It is the backstop; the
    `SPAWNS_PYTEST` deselect list is the front line, and the parent asserts no child node
    skipped, so this guard firing is a failure the parent reports rather than a silence.
    """
    if os.environ.get("BENCH_MINE_REPLAY_NO_SPAWN"):
        pytest.skip(f"{node_name} spawns a child pytest; it is the child")


def test_a_checkout_carrying_the_promoted_sets_still_lets_the_replay_dirty_the_table(tmp_path):
    """Clauses 1 and 2: the rows are in the table, and the recorded write still happens.

    The checkout is HEAD plus one commit adding the three #3688 sets, so `bench_029` is a
    prefix the transcribed guard would have found and aborted on. `_assert_replay_dirties_the_table`
    is the same function the pristine-checkout node calls, so green here is green with all
    four of its proofs — exit 0, `inserted at line`, the table in `git status --porcelain`,
    and the orphan key in its text — none skipped, none weakened.
    """
    where = _checkout_carrying(tmp_path, "promoted", *DURABLE_WRITE_KEYS)
    try:
        table = (where / TABLE_REL).read_text(encoding="utf-8")
        assert BENCH_NUMBER + "_" in table, (
            "the control is broken: no key carrying the incident's bench number is in the "
            "table, so this node would pass on a checkout the old prefix guard was never "
            "offered")
        assert ORPHAN_KEY not in table, "the incident's own key is present, so this is not "  \
            "the state a promotion creates"
        _assert_replay_dirties_the_table(where)
    finally:
        _remove_worktree(where)


def test_the_pinned_replay_node_passes_on_a_checkout_carrying_the_promoted_sets(tmp_path):
    """Clause 1, read literally: the node itself exits green there, not a rephrasing of it.

    `_assert_replay_dirties_the_table` proves the write; this proves the *node* is green, by
    running `pytest tests/test_bench_mine_source.py::test_the_recorded_heredoc_dirties_a_clean_checkout_of_this_commit`
    inside a synthetic checkout carrying the three sets and reading its summary line. Before
    #3731 that child failed with `AssertionError: already present` — the gate excerpt this item
    quotes — and the difference a promotion must not make is the difference this measures.
    """
    _child_env_guarded("test_the_pinned_replay_node_passes_on_a_checkout_carrying_the_promoted_sets")
    where = _checkout_carrying(tmp_path, "node", *DURABLE_WRITE_KEYS)
    try:
        proc = _run_pytest(
            where, "tests/test_bench_mine_source.py::"
                   "test_the_recorded_heredoc_dirties_a_clean_checkout_of_this_commit")
        out = proc.stdout[-600:]
        assert "already present" not in proc.stdout + proc.stderr, out
        assert "1 passed" in out, (
            f"the node did not run green in a checkout carrying the promoted sets "
            f"(rc={proc.returncode}): {out!r} stderr={proc.stderr[-300:]!r}")
    finally:
        _remove_worktree(where)


def test_a_later_bench_029_set_does_not_abort_the_replay(tmp_path):
    """Clause 3: the recurrence case — a *different* bench_029_* key, not one of the three.

    #2718's own subject is that 303 staged notes collided on `bench_029`, and the miner is
    still asked to choose a bench number it cannot see the state of, so the next card may well
    be keyed `bench_029_<anything>`. One set under that number, unrelated to #3688, is what a
    future promotion actually looks like; the replay must still dirty the table over it.
    """
    where = _checkout_carrying(tmp_path, "recurrence",
                               "bench_029_recall_two_sessions_one_tool")
    try:
        _assert_replay_dirties_the_table(where)
    finally:
        _remove_worktree(where)


def test_the_replay_still_refuses_a_table_that_already_carries_the_orphan_key(tmp_path):
    """The guard's remaining job, so the fix cannot have become a silence.

    Widening a guard to a whole key is only safe if it still refuses the thing it refused:
    here the incident's own key, committed. The child must exit non-zero with
    `already present` on stderr — and, because the guard sits above the `p.write_text`, the
    checkout must stay clean, which is the proof that it aborted rather than wrote anyway.
    """
    where = _checkout_carrying(tmp_path, "refusal", ORPHAN_KEY)
    try:
        proc = subprocess.run(INCIDENT_HEREDOC, shell=True, cwd=str(where),
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode != 0, f"the guard no longer refuses: {proc.stdout[-300:]!r}"
        assert "already present" in proc.stderr, proc.stderr[-300:]
        assert _porcelain(where) == "", (
            "it refused and wrote anyway; the abort must happen above the write")
    finally:
        _remove_worktree(where)


def _string_literals(path: Path) -> list[str]:
    """Every string-literal VALUE in `path`'s source, comments and docstrings excluded.

    `tokenize` gives a comment as a COMMENT token, and a docstring as a STRING token that is an
    expression statement's value; the docstring exclusion matters here because this file's prose
    names the bad prefix in order to explain it, and prose is not a key test.
    """
    src = path.read_text(encoding="utf-8")
    doc_at = {(node.body[0].lineno, node.body[0].col_offset)
              for node in ast.walk(ast.parse(src))
              if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                   ast.AsyncFunctionDef))
              and node.body
              and isinstance(node.body[0], ast.Expr)
              and isinstance(node.body[0].value, ast.Constant)
              and isinstance(node.body[0].value.value, str)}
    return [ast.literal_eval(tok.string)
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type == tokenize.STRING and tok.start not in doc_at]


def test_the_guard_is_keyed_on_the_whole_orphan_key_not_a_prefix():
    """The one-token fix, pinned as a property of the recorded string.

    A guard on the bare prefix matched every key that merely starts with the incident's number,
    which is the defect #3731 exists to remove; the guard now quotes `ORPHAN_KEY` in full. Read
    off `INCIDENT_HEREDOC` rather than off a copy, so re-narrowing it reddens this node, and the
    checks after it are the whole-file version of the same rule: nowhere in this file may the
    bare prefix be a string literal again, which is the only form that can be compared against a
    key.
    """
    guard = [ln for ln in INCIDENT_HEREDOC.splitlines() if "already present" in ln]
    assert len(guard) == 1, f"expected one guard line, found {guard!r}"
    # Parsed, not matched: a regex over the line cannot tell the key from the `already
    # present` message that sits on the same line, and a check that answers "which strings are
    # quoted here" is not a check on what is compared against the table. The AST says exactly
    # what the shell will do — `<literal> not in s`.
    stmt = ast.parse(guard[0].strip()).body[0]
    assert isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Compare), guard[0]
    cmp = stmt.test
    assert len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.NotIn), (
        f"the guard is no longer a containment test: {guard[0]!r}")
    assert isinstance(cmp.left, ast.Constant) and isinstance(cmp.left.value, str), (
        f"the guard's left side is not a literal key: {guard[0]!r}")
    assert cmp.left.value == ORPHAN_KEY, (
        f"the guard is keyed on {cmp.left.value!r}, not the whole orphan key {ORPHAN_KEY!r}")
    # The two halves catch different regressions. Above: restoring the prefix INSIDE the
    # recorded command, which is what #3731 changed. Below: a NEW literal comparison against
    # the bare prefix anywhere else in the file — over string TOKENS rather than text, because
    # this file has to *describe* the bad prefix to explain the fix, and a text grep would fail
    # on its own prose. A string literal is the only form that can be compared against a key,
    # so that is the form that is banned.
    literals = _string_literals(Path(__file__))
    bare = [lit for lit in literals if lit == BENCH_NUMBER]
    assert not bare, (
        f"the bare prefix is back as a string literal in this file {len(bare)} time(s); a "
        f"key test on it refuses every bench_029_* set, which is #3731")


def test_the_whole_file_is_green_before_and_after_the_promotion(tmp_path):
    """Clause 4: promoting the rows reddens nothing else here, and not promoting them does not either.

    The file runs inside synthetic checkouts of both table states — one stripped of the three
    durable-write sets, one carrying them — and every run must be green at its expected node
    count. The counts are derived from the file the child actually runs (HEAD's copy, since every
    checkout here is a worktree of HEAD) rather than written as literals, because a literal goes
    stale the first time this file gains a node and then reads as a regression in the wrong
    direction.

    Each state is run twice, and the two answers are different questions. The whole file is 14
    nodes here, 12 in a child run once the 2 that spawn a child pytest are deselected. The eight
    nodes #3731 found in this file are then re-run on their own, its own six deselected: that is
    clause 4 at the number it states, `8 passed` in both states. `PREEXISTING_NODES` names those
    eight, so a rename that drops one reddens this node instead of quietly shrinking the claim.
    """
    _child_env_guarded("test_the_whole_file_is_green_before_and_after_the_promotion")
    head_src = subprocess.run(["git", "-C", str(TREE), "show", f"HEAD:{TARGET_REL}"],
                              capture_output=True, text=True, check=True).stdout
    defined = re.findall(r"^def (test_\w+)", head_src, flags=re.M)
    absent = [n for n in PREEXISTING_NODES if n not in defined]
    assert not absent, (
        f"{absent} is/are not defined at HEAD, so the eight clause 4 counts have been renamed or "
        "deleted away from under this node")
    added = [n for n in defined if n not in PREEXISTING_NODES]
    whole = len(defined) - len(SPAWNS_PYTEST)
    # Paths relative to the child's own tree: an absolute path from this process lands outside
    # the child's rootdir, where pytest matches a deselect against its own node ids, would run
    # the spawners, skip them on the env guard, and report it to me as `2 skipped`.
    drop_spawners = [f"--deselect={TARGET_REL}::{n}" for n in SPAWNS_PYTEST]
    # Clause 4's eight, by set difference, so the list cannot silently gain a member.
    drop_added = [f"--deselect={TARGET_REL}::{n}" for n in added]

    runs = []
    plain = _checkout_lacking(tmp_path / "before", *DURABLE_WRITE_KEYS)
    promoted = _checkout_carrying(tmp_path, "after", *DURABLE_WRITE_KEYS)
    try:
        for state, where in (("without", plain), ("with", promoted)):
            for label, opts, n in (("the whole file", drop_spawners, whole),
                                   ("the eight pre-existing nodes", drop_added,
                                    len(PREEXISTING_NODES))):
                proc = _run_pytest(where, TARGET_REL, *opts)
                runs.append((f"{state} the promotion: {label}", proc, n))
    finally:
        _remove_worktree(plain)
        _remove_worktree(promoted)

    for label, proc, n in runs:
        out, err = proc.stdout[-300:], proc.stderr[-200:]
        assert "skipped" not in out, (
            f"a node skipped in the run {label} ({out!r}); a deselect list is stale, so the "
            "child's skip guard fired on a node that was meant to run")
        assert "failed" not in out and "error" not in out, (
            f"not green {label} (rc={proc.returncode}): {out!r} {err!r}")
        assert f"{n} passed" in out, (
            f"expected {n} passed {label}, got {out!r}")


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


# ---------------------------------------------------------------------------
# The prompt must stop asking the turn to number the task (#3531, clause 1)
# ---------------------------------------------------------------------------

def _assembled_prompt_text(loser: bool = False) -> str:
    """The prompt a mining turn is actually sent, through the real builders.

    Assembled by the two functions the clause names, not by re-reading
    `_TASK_SCHEMA`: each appends the schema from its own f-string, so a slot that
    survived in one branch and not the other is exactly what a schema-only check
    would miss. The arguments are the shapes `_execute` passes at its two call sites.
    """
    if loser:
        return bench_mine._ledger_loser_prompt(
            "bench_004_replay_schedule_task", 0.2)
    return bench_mine._run_failure_prompt(
        {"run_path": "/tmp/lloyd-data/autonomy-runs/24/run_24_x.md",
         "run_id": "run_24_x", "source": "knowledge-health-report",
         "status": "failed", "summary": "connection refused"})


def test_neither_mining_prompt_asks_the_turn_to_choose_a_bench_number():
    """Clause 1: no id slot carries a number, and the schema says who assigns it.

    2026-10-28: 324 of the 542 notes staged under `bench-mine/` declared
    `bench_029_*`, against a live corpus whose highest id is `bench_028`. The turns
    were not guessing wildly — `bench_029` is the correct answer to the question the
    prompt asked, and the question was unanswerable: the number a turn can read off
    `~/obsidian/lloyd/bench/` advances only on a human promotion, and
    `grep -l 'review_status: promoted' ~/obsidian/lloyd/bench/*.md` returns no file,
    so it has never advanced. `_TASK_SCHEMA` showed `id: bench_XXX_<slug>` and said
    nothing about the number; `git log -S'bench_XXX'` on this module returns one
    commit, `0edaab55` (2026-04-18) — the line had never been amended.
    """
    for label, text in (("run-failure", _assembled_prompt_text()),
                        ("ledger-loser", _assembled_prompt_text(loser=True))):
        assert "bench_XXX" not in text, (
            f"the {label} prompt still offers `bench_XXX` as an id slot, so every "
            f"turn it sends is instructed to number the task it is writing: "
            f"{[ln for ln in text.splitlines() if 'bench_' in ln][:4]}")
        assert "id: <short-slug>" in text, (
            f"the {label} prompt's schema no longer shows an id line for the turn to "
            f"copy, so it will invent one: "
            f"{[ln for ln in text.splitlines() if 'id:' in ln][:4]}")
    schema = bench_mine._TASK_SCHEMA
    assert "the pipeline numbers the" in schema.lower(), (
        "_TASK_SCHEMA shows an id line without saying the pipeline assigns the "
        "number, which is the ambiguity that produced 324 notes claiming one id: "
        f"{schema[-300:]}")
    assert "do not put a task number" in schema.lower(), (
        "the schema names the assignee but never tells the turn to leave the number "
        f"alone, which is the half that stops the collision: {schema[-300:]}")
