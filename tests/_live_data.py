"""The one rule for a read-only guard whose live data is gone (backlog #1377).

Imported by every `test_*.py` that reads data on the machine — run
`grep -l "from tests._live_data import" tests/` for the list. No count of it is
written here: every enumeration has been stale by the next importer, and a number in
this file cannot be re-measured by the reader who needs it. Not a
`test_*.py` file, for the reason `tests/_relief_harness.py` gives — pytest does not
collect it, so a broken helper is a collection error in every importing suite rather
than a mysterious red run.

Why a shared rule
-----------------
Every guard in the importing files reads *data on the machine*, not files in the checkout:
`.gitignore` ignores `/sessions/` and `/_pipeline/`, so neither path is in git and
neither exists in a round's worktree. On 2026-09-22 a fixture teardown deleted the
production tree in 35 seconds (`tests/conftest.py:159-170` states it in code); the
tracked half came back from git by 13:02 PDT and the gitignored half only where a job
regenerated it, so `_pipeline/trajectories` and
`_pipeline/skills/reviews/verdicts.jsonl` stopped existing at all and
`~/lloyd/sessions` fell from the 3,000 files recorded on 2026-09-18 to 58.

Eight guards hard-asserted their presence. They therefore reproduced as red nodes
*at base* in every round's `tests` rung — `base_probe: probed 21 file(s) at base
1842b8cf: 106 already failing`, `external_blocker: true`, from
`~/.local/state/lloyd-automod/rounds/SM_20260922_201206/gate.json` — which blocks
every promotion after the one that noticed, not only that one.

The rule, and the line it must not cross
----------------------------------------
The absence of derived data is a property of the *machine* and gets a named skip whose
reason carries the missing path. Everything else still fails: a path that exists as
the wrong kind, a corpus that is present but empty or indiscriminate, a ledger that
exists but lost rows, a store whose classifier disagrees. A guard must therefore never
test existence itself and skip on the result — that is how a present corpus gets
skipped — it calls this function, which decides which of the two answers applies.
`test_an_absent_live_root_skips_every_guard_that_reads_it_by_name` and the tests
beside it pin both halves per guard.

What this does NOT decide: whether the automod gate's `tests` rung may go green on
named skips while a corpus it treats as required is absent. That is the fork backlog
#1377 step 3 leaves to a person.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def require_live_data(path: Path, what: str, kind: str = "dir") -> None:
    """Skip by name when a live-data root is genuinely absent; assert when it is not.

    `what` names the thing the guard reads (it appears in the skip reason beside the
    path) and `kind` is ``"dir"`` or ``"file"``. Two answers, deliberately:

      * the path does not exist -> ``pytest.skip`` naming it, because no round, no
        checkout and no re-run conjures derived data that was never regenerated;
      * the path exists but is not the kind of thing the guard reads -> a failing
        ``assert``, because that is a statement about the data, not the machine.
    """
    if not path.exists():
        pytest.skip(f"{what} absent at {path}: this machine holds no live copy to "
                    "read, so the guard has nothing it could check")
    if kind == "dir":
        assert path.is_dir(), f"{what} at {path} exists but is not a directory"
    else:
        assert path.is_file(), f"{what} at {path} exists but is not a file"


def require_live_volume(items, floor: int, root: Path, what: str,
                        noun: str = "files") -> None:
    """Skip a guard whose sample is below the size it needs to discriminate.

    Distinct from `require_live_data`: here the root is present and small, which is
    neither a machine with no data nor a full corpus. The reason names BOTH the floor
    and the observed count, because "vacuous" without the two numbers is unattributable
    — the whole reason `assert len(files) > 500, "so this is vacuous"` failed a round
    with no way to tell a 58-file store from a 0-file one.

    `noun` names what is being counted, because not every live store is counted in
    files: the autoresearch guards count *rounds* inside a frozen window, and a reason
    reading "holds 0 files" of a ledger would name the wrong unit. It defaults to
    "files" so the callers pinned before it existed keep their exact wording.
    """
    count = len(items)
    if count < floor:
        pytest.skip(f"{what} at {root} holds {count} {noun}, below the {floor}-{noun[:-1]} "
                    "floor under which this guard cannot discriminate either way")


# Declared ONCE, here, because two guards ask the same question of the same store:
# the corpus entity guard (`tests/test_eval_corpus_guard.py`) and the seed-anchoring
# guards (`tests/test_retrieval_seed_anchoring.py`). Two literals would let them
# disagree about what "below floor" means, which is the defect this line exists to
# prevent — pinned by
# `tests/test_eval_corpus_guard.py::test_both_kg_guards_route_through_the_one_floor_helper`.
# The number is the suite's own pre-existing bar (`test_the_default_route_is_the_live_store_the_eval_scores_against`
# asserted `len(names) > 1000` before this helper existed) and it sits two orders of
# magnitude under the last whole store — 26,390 entity rows, per `backlog/1381`'s close
# note — so it admits a plausible partial index and refuses a stub.
KG_ENTITY_FLOOR = 1000


def require_live_entity_volume(entity_names, root,
                               what: str = "the knowledge-graph store") -> None:
    """Skip a KG guard whose store is present but too small to discriminate.

    The third answer a live knowledge-graph store gets, after `require_live_data`'s
    two (absent -> named skip, wrong kind -> failing assert): a store that OPENS and
    holds a handful of entity rows. On 2026-09-22 a 30-entity / 0-edge stub appeared
    at `_pipeline/vault-derived/kg.sqlite`. Because the file existed, the absent-skip
    stopped firing: across the two guard files the whole-suite delta recorded in
    `backlog/1394` is 39 -> 30 skipped, 8722 -> 8731 passed, 0 -> 5 failed, the five
    naming 96 of the corpus's 100 `expect_entities` entries as unreachable or
    unresolvable, and the automod `tests` rung could not go green on any tree —
    verbatim from `~/.local/state/lloyd-automod/rounds/SM_20260923_033853/gate.json`:
    "every failure reproduces at base c2bd72b6 with this round's diff absent —
    PRE-EXISTING BREAKAGE", `external_blocker: true`. Total absence had NOT blocked
    it: `SM_20260923_034250` promoted at 03:51:45Z with `kg.sqlite` missing and 39
    skips. So it is the partial stub, not the data loss, that fails a guard, and
    deleting the stub to get the absence back would be the wrong move.

    Zero rows is deliberately NOT taken here. An empty `entities` table stays a
    FAILING verdict in the caller
    (`test_a_store_that_opens_with_no_entity_rows_is_refused_as_a_verdict`), because
    `KGStore(path)` CREATES an absent database and "no rows" is the created-empty
    false-clean that once made a worktree record `duplicate_rows: 0` about a store
    that was not there (`app/uptake.py`). This skips only on `0 < count < floor`.

    The reason names the floor, the observed count and the store path — the same
    attribution rule `require_live_volume` states, since a bare "vacuous" cannot be
    checked against the filesystem in the same breath as the skip.
    """
    count = len(entity_names)
    if 0 < count < KG_ENTITY_FLOOR:
        pytest.skip(f"{what} at {root} holds {count} entity rows, below the "
                    f"{KG_ENTITY_FLOOR}-entity floor under which this guard cannot "
                    "discriminate either way: rows left over from an unrelated stub "
                    "make every corpus expectation look unreachable, which is a "
                    "statement about the store, not about the corpus")
