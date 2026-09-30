"""Where a session's Bash calls start when the call names no directory (#1906).

One convention, written by the code that mints a turn and read by the Bash tool when
an argument is absent. Both sides import this module, which is the only way to keep
the reader and the writers from drifting into two different conventions — the drift
that would put a worker's relative write back into the live checkout.

`start_cwd` is a key on the session's own record (`~/lloyd-data/sessions/<id>.json`,
or the directory a minter was handed); a round copies the same value into its
`round_start` ledger row (`scripts/automod/round.py:211`) as a second reading. The
session record is the session, and a mint that can reach the tool stamps it through
`stamp_new_session` below. The mints, with what each one is:

- a worker turn — `workers/sources/_common.py::new_worker_session`, for every worker
  source that runs through `run_prompt_in_session`, whose own backfill
  (`ensure_session_start_cwd`) covers a session handed in already minted;
- a turn that goes straight to the primary model —
  `workers/sources/_common.py::run_prompt_on_primary`, which mints its own
  `platform="worker"` session and is what `workers/sources/bench_mine.py:934`,
  `bench_mine.py:955` and `workers/sources/session_distill.py:300` run their turn on.
  It passes through `new_worker_session` not at all, which is how it stayed unstamped
  through two reviews;
- an autocode round turn — `scripts/automod/round.py::start` restamps that record with
  the round's worktree, so its relative writes land where its diff is;
- an automod review grader — `scripts/automod/review.py::write_session`, the single mint
  behind both the code grade and the vault-review grade;
- an autonomy run — `app/autonomy.py::run_task`, which mints its own
  `platform="autonomy"` session and therefore passes through none of the above;
- the two measurement harnesses that mint a `platform="worker"` session and hand it a
  Bash tool: `eval/run_rpc_eval.py::run_one` and
  `scripts/autoresearch/bench_runner_sdk.py::run_trial`.

The one mint deliberately NOT on that list is the automod canary's synthetic smoke turn,
`scripts/automod/canary_smoke.py::run` — because it does not need a stamp and stamping it
would cost something. It does not need one because the stack that serves that turn is not
this one: `scripts/automod/canary_config.py:211` and `:225` give both of the canary's
programs `directory={worktree}`, so supervisor starts the canary's own MCP server with the
round's worktree as its cwd, and an unstamped call inherits a directory that is already
outside the live checkout. Stamping it would cost the isolation the canary data root
exists to provide: `stamp_new_session` lays its scratch under `SCRATCH_ROOT`, and the
process that would call it on the canary's behalf is the gate, whose `SCRATCH_ROOT` is
production's `~/lloyd-data/session-cwd`. `resolved_for` records what follows for a record
in a non-default root. Two nodes hold the decision open rather than let it be forgotten:
`test_a_stamp_written_to_a_non_default_root_is_read_by_that_root_and_no_other` pins the
read/write asymmetry, and
`test_the_canary_smoke_turn_starts_outside_the_live_checkout_without_a_stamp` pins the
`directory=` that makes the exception safe together with the stamp that must not appear.

Whether the stamped list is complete is not a claim in prose: the AST walk in
`tests/test_session_start_cwd.py::test_every_session_mint_in_the_tree_stamps_one`
fails if a `create_session` call site appears in the tree without a stamp behind it.
`review.write_session` writes its record with a bare `write_text` and so is outside that
walk's reach; its own node is what holds its stamp.

A turn's start directory therefore travels with the thing the tool already resolves.

Why a start directory at all: `lloyd-mcp` is started by supervisord with cwd
`~/lloyd`, and `Bash` with no `cwd` argument spawns with `cwd=None`, which means
"inherit". So EVERY worker's default working directory was the live production
checkout. A grader handed `TMPDIR=~/lloyd-work/.t/05431072c3` built its fixture at
the RELATIVE path `.t/05431072c3/r1873` and put nine files on `main`; a nightly
probe wrote an unmeasured report into tracked `eval/uptake/`; datawatch alerted
hourly for eleven hours and never named a writer (#1906).

The fallback is deliberate and is the reason this is safe to land: a session with no
usable start directory gets exactly the behaviour it has today — the server's own
directory, however wrong that is — rather than a spawn that dies, or a directory
invented under it. An explicit absolute `cwd` outranks this on every session kind.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from app.paths import LLOYD_HOME, SESSIONS_DIR

META_KEY = "start_cwd"

# Outside every checkout by construction: `~/lloyd-data/session-cwd/`, next to the
# session records whose `start_cwd` points here, so one path answers "where was this
# session's Bash running" and retention can find the pair together. Never under
# `LLOYD_HOME`: a scratch inside a checkout is the bug, restated. Mutable so a test
# can point the convention at a temp root — and a test that does is itself the
# reason `path_for` exists as one function rather than a format string at two sites.
SCRATCH_ROOT = SESSIONS_DIR.parent / "session-cwd"

_UNTRUSTED = re.compile(r"[^A-Za-z0-9._-]")


def live_root() -> Path:
    """The production checkout, resolved the way the rest of the codebase resolves it.

    A method, not a constant, because the gate and the canary override `LLOYD_HOME`
    for their children: a check that read a literal `/home/alansrobotlab/lloyd` would
    call the canary's tree safe to write into while calling the scratch inside the
    real tree unsafe.
    """
    return Path(LLOYD_HOME).resolve()


def safe_name(session_id: str) -> str:
    """A session id made fit to be a filename.

    Every caller here passes an id this process generated, but the resolver reaches
    this with a value that came from a session record, and a record is data — a name
    like `../../etc/x` must not turn a read of the session dir into a read of
    anywhere else.
    """
    return _UNTRUSTED.sub("_", str(session_id or ""))[:120]


def path_for(session_id: str) -> Path:
    """Where this session's start directory lives, whether or not it exists yet."""
    return SCRATCH_ROOT / safe_name(session_id)


def _usable(path: Path) -> bool:
    """True when `path` can be a working directory, creating it if its parent is there.

    A deleted directory is recreated only when its root still exists, which is the
    difference between a retention sweep that pruned one session's scratch (recover
    it: the record still names it, and every later call expects it) and one that
    removed the root while the process held the old path (do not re-lay a data tree
    under a path that was just swept — `app/paths.py` is emphatic that import-time
    and incidental creation is what bit the last time).
    """
    if path.is_dir():
        return True
    if not path.parent.is_dir():
        return False
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def ensure(session_id: str) -> Path:
    """Create and return this session's start directory.

    Called explicitly by a minter, which knows it is making a session. The Bash
    resolver never calls this: re-creating a directory on the read path would be an
    incidental write from a tool that was only asked to run a command, and it would
    paper over the case the sweep-test below needs to see — a record naming a
    directory that is gone.
    """
    path = path_for(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def outside_live(directory: Path) -> bool:
    """True when writing into `directory` is not writing into the live checkout.

    The refused set is the directory itself and everything below it: a start cwd of
    `~/lloyd` reproduces the incident, and `~/lloyd/.t` is precisely the path that
    was alerted on, so a prefix match is not decoration.
    """
    target = Path(directory)
    if not str(target):
        return False
    root = live_root()
    try:
        resolved = target.resolve()
    except OSError:
        return False
    return resolved != root and root not in resolved.parents


def stamp(session_id: str, directory: Path, *,
          sessions_dir: Path | None = None) -> Path | None:
    """Record `directory` as the session's start cwd; refuse one inside the tree.

    Returns the directory written, or None when it was refused. Refusing is the
    purpose of the check: a storer that fell back to the live tree would reproduce
    the exact bug this closes, and a caller that handed it the tree is the caller
    with the bug — it should hear "nothing was written", not get a silent downgrade.

    Read-modify-write over the session JSON, so stamping never drops a key another
    writer owns. A record that does not exist is skipped rather than created:
    `app.sessions_io.create_session` and `review.write_session` are the writers of
    that shape, and minting a half-populated one from here would make this a second.

    `sessions_dir` names a non-default record location, because `run_grader` takes a
    `sessions_dir` precisely so a grading session can be minted elsewhere; a storer
    that could only write the default would miss exactly the records a test can point
    at, and those are the ones the round's grader actually reads.

    The value stored is the RESOLVED spelling, because that is the spelling the child
    reports: `pwd` prints the physical path, so a record written as
    `str(Path(symlinked_scratch))` and a node asserting `pwd == str(scratch)` disagree
    on any box where the data root is a symlink — and that disagreement looks exactly
    like the bug this module exists to close. Resolving here is also what makes the
    `outside_live` re-check in `read` a check of the same directory `stamp` approved:
    a symlink aimed inside the checkout is refused by the guard above, which resolves
    too, and is never stored under the spelling that would let it back in.
    """
    if not session_id or not outside_live(directory):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    stored = directory.resolve()
    path = (sessions_dir or SESSIONS_DIR) / f"{safe_name(session_id)}.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return None
    except (OSError, ValueError):
        return None
    meta[META_KEY] = str(stored)
    _atomic_write(path, meta)
    return stored


def stamp_new_session(session_id: str, *,
                      sessions_dir: Path | None = None) -> str | None:
    """Give a session that was just minted a start cwd, and never fail the mint over it.

    **The one call a minter makes.** `ensure` then `stamp`, in that order, with the
    failure swallowed: a mint that cannot lay a scratch keeps the behaviour it has
    always had (it inherits the server's cwd) instead of losing its turn to an
    `OSError` from a `mkdir`. Doing those three things inline at every mint is how
    five mints ended up with five subtly different error policies, and the sixth new
    one gets a seventh; the enumeration is therefore closed by
    `tests/test_session_start_cwd.py::test_every_session_mint_in_the_tree_stamps_one`,
    which walks the AST for this name at every `create_session` call site rather than
    trusting a list of mints kept by hand.

    `sessions_dir` is passed through to `stamp` for a mint that writes its record
    somewhere other than the default session store — a review grader minted into a
    round's canary root, for instance. It does not move the scratch: the directory
    created here is always under `SCRATCH_ROOT`, which is outside every checkout.
    """
    if not session_id:
        return None
    try:
        written = stamp(session_id, ensure(session_id), sessions_dir=sessions_dir)
        return str(written) if written else None
    except Exception:    # noqa: BLE001 — the mint outranks its start directory
        return None


def read(session_id: str, *, sessions_dir: Path | None = None) -> str | None:
    """The start cwd the record names, whether or not it still exists.

    Returns None for "nothing recorded" and for "the record could not be read", and
    never raises: a session whose JSON is mid-write or unparseable must run in the
    directory it ran in yesterday, not lose its Bash tool.

    A value that names a directory inside the live checkout is dropped here rather
    than trusted. Nothing in this tree writes such a value — `stamp` refuses it — but
    the resolver's whole job is to keep a worker out of that tree, and a session
    file is data: a hand-edited or stale record must not be able to put it back.
    """
    if not session_id:
        return None
    where = sessions_dir or SESSIONS_DIR
    try:
        meta = json.loads((where / f"{safe_name(session_id)}.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    value = meta.get(META_KEY)
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(os.path.expanduser(value))
    if not candidate.is_absolute() or not outside_live(candidate):
        return None
    return str(candidate)


def resolved_for(session_id: str, *,
                 sessions_dir: Path | None = None) -> str | None:
    """The directory to start in, or None meaning "behave as you always have".

    The resolver both sides read: `agent_mcp/builtin_bash.py` for a turn's Bash
    calls, `scripts/automod/review.py` for a grader turn whose start directory is
    the scratch the gate handed it. None is a real answer with two causes — nothing
    recorded, and a record naming a directory that cannot be made usable again — and
    in both cases the caller inherits the server's cwd exactly as it does today.

    That inheritance is not made worse here even though it is the live tree: a
    session with no record is a chat session, or a turn predating this change, and
    inventing a directory for those would redefine `pwd`, `ls` and every relative
    path for all of them in one promotion. The way a session stops being in that set
    is for its minter to call `stamp_new_session`, which every mint that can reach the
    tool now does (`tests/test_session_start_cwd.py::test_every_session_mint_in_the_tree_stamps_one`).

    `sessions_dir`, and what the Bash resolver does NOT do with it. The resolver at
    `agent_mcp/builtin_bash.py:144` calls this function with NO root, so it reads the
    store of the process that serves the turn — `app.paths.SESSIONS_DIR`, which each
    process resolves from its own `LLOYD_DATA` at import. This is a deliberate limit
    and not an oversight: a record written into some other root belongs to a stack that
    reads that root, and the only such stack on this box is a canary, which boots its
    own backend and MCP against `canary_config.canary_data_root(round_dir)` — the root
    `scripts/automod/canary.py:224` hands its smoke turn, and the root
    `review.write_session` would stamp into if a grading route were ever pointed at a
    canary. The live aggregator never serves a turn from that stack, and a resolver
    that guessed by scanning other roots would be choosing a session id's owner by
    collision order — two stacks, one `session_id`, and the answer depends on which
    directory `glob` met first. So the root is a caller-supplied argument for a
    process that knows it (a test, or a minter reading back its own write), and the
    live resolver has exactly one store to be wrong about.
    """
    value = read(session_id, sessions_dir=sessions_dir)
    if value is None:
        return None
    path = Path(value)
    return value if _usable(path) else None


def _atomic_write(path: Path, payload: dict) -> None:
    """Write the record through a temp file, so a crash cannot truncate a session.

    The same shape `agent_mcp/_rpc.py` and `gstate.write_json_atomic` use: the
    backend rewrites these files on every turn, and a torn session JSON would take a
    transcript down with the cwd key.
    """
    tmp = path.with_name(path.name + ".cwd.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
