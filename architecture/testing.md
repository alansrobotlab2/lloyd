---
segment: architecture
tags: [architecture, lloyd, testing, gate, pytest]
type: reference
status: implemented
date: 2026-09-24
---

# Lloyd — the test suite

~9,600 tests under `tests/`, run by every automod round's `tests` rung
([[automod]]) and by hand from a throwaway worktree. This doc is about the
suite's own structure: the two kinds of test in it, the rule for tests that read
data off the machine, and the isolation the parallel runner requires. The gate's
eleven rungs and the promotion flow are in [[automod]].

## Never run the suite against the production tree

`tests/conftest.py::_refuse_the_production_tree` raises `UsageError` if pytest is
invoked with `~/lloyd` as rootdir. This is not tidiness:

> On **2026-09-22 a fixture teardown deleted the production tree in 35 seconds** —
> the whole thing, mid-run.

Everything gitignored went with it and did not come back from git: `.venvs/`,
the model checkpoints, `agent-services/cert/`, `_pipeline/`, `sessions/`,
`eval/baselines/`, `qmd/`. Recovery took a day. Run from a worktree:

```bash
git worktree add --detach /tmp/lloyd-check HEAD && cd /tmp/lloyd-check && pytest -q
```

`LLOYD_ALLOW_LIVE_TREE_TESTS=1` exists for a human who means it. A round must
never set it.

## Two kinds of test

**Synthetic** tests build their fixture and pass on any machine. Most of the
suite. They are the coverage that survives.

**Live-data** tests read what is *on the machine* — `~/lloyd-data/sessions/`,
`~/lloyd-data/_pipeline/`, `~/lloyd-data/eval/baselines/`,
`~/lloyd-data/_pipeline/vault-derived/kg.sqlite`. Runtime data left the code
tree on 2026-09-22 (`6426668b`, [[data-home]]); the tree-relative names are
history — `6426668b` postdates this doc's first pass by three hours and rotted
them. These tests exist for a real
reason: they cross-check a figure published in a vault note against the store it
was derived from, so nobody can hand-edit a number into a published measurement
without a test going red.

The cost is that they encode the machine's state. When the machine changes, they
go red for a reason no commit caused — and because the `tests` rung is a gate for
*every* round, one such test blocks all promotion, not just the round that
noticed. On 2026-09-22 that was 98 nodes red at base.

Keep both. A file usually holds a synthetic majority and a small live section;
`test_uptake.py` is the shape to copy — ~70 synthetic tests, and a small live
section whose baseline cross-checks call the helpers below instead of
hard-asserting. `test_promotion_fp_rate.py` is what a file looks like *after*
retirement: 22 all-synthetic tests, its ten live checks deleted on 2026-09-22
(see below), the marked section headers still standing where they were — its
module docstring still describes those deleted tests (filed, backlog #1440).

## The rule for live data: `tests/_live_data.py`

Three functions, and the distinction between them is the whole point.

```python
require_live_data(path, what, kind="dir")   # absent -> named skip; wrong kind -> assert
require_live_volume(items, floor, root, what, noun="files")  # too small to discriminate -> skip
require_live_entity_volume(entity_names, root)  # KG store present but a stub -> skip
```

- **Absent** is a fact about the machine and gets a **named skip carrying the
  path**. No round, no checkout and no re-run conjures derived data that was
  never regenerated.
- **Present but the wrong kind** is a fact about the data and still **fails**.
- **Present but below a discriminating floor** skips naming **both** the floor
  and the observed count. "Vacuous" without the two numbers is unattributable —
  it cannot tell a 58-file store from a 0-file one.
- **Present but a stub** is the KG store's third answer (`3181b104`, backlog
  #1394): the entity guard skips only at `0 < rows < KG_ENTITY_FLOOR` (1000,
  declared once beside the helper), because `KGStore(path)` *creates* an absent
  database — "0 rows" is a created-empty false-clean and stays a **failing**
  verdict. It was a 30-entity stub sitting at `_pipeline/vault-derived/kg.sqlite`
  on 2026-09-23 that blocked every promotion, where total absence had not.

**A guard must never test existence itself and skip on the result.** That is how
a corpus that is present gets skipped. Call the helper and let it decide.

## When a live check should be retired instead

A skip says *this might run again*. If that is false, skipping is a lie that
accumulates.

Ask one question: **does the data this test reads refill on its own?**

- **Yes** — sessions accumulate, trajectories get mined, baselines get collected,
  the fact tree re-derives from the vault. A named skip is honest and temporary.
- **No** — the check is pinned to a *frozen window*: the FP-rate note's window
  ending `R_20260908_181458`, the thirteen `nightly-202609*` baselines of
  09-04..09-17,
  a 67-promotion corpus. That data is gone permanently.

For the second kind, **delete the check and annotate the note it guarded**.
Deleting alone silently downgrades a published number from machine-verified to
"someone typed it", with nothing recording the change — so the note itself says
so. Keep every synthetic test; what dies is the cross-check against a vanished
measurement, not coverage of the code.

Worked example, 2026-09-22: 36 checks retired across six files, their published
notes annotated (`knowledge/evaluation/autoresearch-promotion-fp-rate-at-threshold-0p5.md`,
backlog #608). Where a live test carried a contract that was never about the
destroyed data — `eval_trend_stats`'s `--strict` exit codes — the contract was
**rebuilt on a synthetic fixture first, then the original deleted**.

## The gate's floors, and why the skip cap exists

`scripts/automod/gate.py`:

| Floor | Value | Catches |
|---|---|---|
| `PYTEST_MIN_COLLECTED` | 1000 | a conftest import failure collecting everything and running none of it |
| `PYTEST_MIN_PASSED` | 1000 | collected ≠ run |
| `PYTEST_MAX_SKIPPED` | 40 | *"a round that skips its way to green is not a round that passed"* |

The skip cap is the one to respect when a round's fix converts failures into
skips. That is exactly the move it exists to catch, and the correct response is
almost never to raise it — it is to ask whether those tests can ever pass again,
and retire the ones that cannot. Raising the number should be the last resort and
should say in a comment what it is hiding.

## Marks and selection

`TESTS_MARK_EXPR = "not live_vault and not fault_injection"`. The gate deselects
both.

The trap: **a marked test is invisible to the gate but still red in a full run.**
`test_conversation_relations.py::test_live_trajectories_yield_material_co_access_signal`
carried `live_vault`, was red at head for days, and entered no gate report —
it surfaced only when someone ran the whole suite. Run unmarked periodically.

## Pre-existing failures block everyone

`rung_tests` probes the base commit before blaming a round: `_failures_at_base`
re-runs the failing node ids in a throwaway worktree, and failures present at base
are reported as `external_blocker: true` with the list under `external_failures`.
The round is re-offered rather than rejected.

Read `external_failures`, not `failed_node_ids` — the latter is empty on an
external block, which is easy to misread as "the gate did not say which test".

## Parallel isolation is a requirement, not a nicety

The rung runs `-n 8`. Tests from different files share a worker process, so
**anything process-global leaks**: logging levels and propagation, `TMPDIR`, cwd,
`os.environ`, module state, and any singleton a module caches at import.

A test that passes alone and fails in the full run is almost always this. Two
from 2026-09-22:

- `test_tool_effects.py` asserted on `caplog` records. Another file on the same
  worker had raised the `lloyd-mcp` logger, so `caplog` saw nothing and the test
  read "no suppression was logged" when one was. Fix: pin level, `propagate` and
  `disabled` for the duration.
- `test_system_health_check_vault_sync.py` globbed the shared temp dir and
  asserted no new scratch directory remained — a statement about the whole
  machine. A sibling worker's scratch directory failed it. Fix: give the test a
  `TMPDIR` of its own.

## Stubs must tolerate signature growth

A fake that pins an exact signature breaks the moment production grows a keyword,
and the breakage does not look like a signature problem.

`c0f12db` added a keyword-only `isolate_home` to `Gate._child_env` and updated
its call sites but not the three places that fake it. Two were test monkeypatches;
the third was `scripts/maintenance/guard_vacuity.py`, where the driver raised and
both verdicts printed `ERROR` — scoring three live gate floors **VACUOUS** for a
reason that had nothing to do with the guards. Stub with `**_kw`:

```python
g._child_env = lambda root=None, **_kw: {}
```

Same class: a fake Gate built with `Gate.__new__` skips `__init__`, so every
attribute the method under test reads has to be planted by hand.

## Guard vacuity

`scripts/maintenance/guard_vacuity.py` probes production guards by running each
against inputs it should PASS and BLOCK, and again with the guard mutated. A guard
whose real and mutant verdicts never differ is **VACUOUS** — it is not stopping
anything. Sites are found by **searching for the guard's own symbol**, never by a
pinned line number, because a number is a snapshot and the symbol moves with the
code.

## See also

[[automod]] for the rungs, the promoter and rollback · [[vault-protection]] for
why a test must never write to `~/obsidian` · [[harness]] for the agent loop the
suite exercises · [[data-home]] for where the live data moved on 2026-09-22.

## Review log

- 2026-09-24 — **stale**: the 09-22 data-home move (`6426668b`) rotted the
  live-data roots (now under `~/lloyd-data/`, [[data-home]]); `tests/_live_data.py`
  has gained a third function (`require_live_entity_volume`, `3181b104`); suite
  size re-measured at ~9,600 (gate `collected` 9,583–9,620, rounds
  `SM_20260924_09*`); the shape-to-copy example moved to `test_uptake.py` because
  `test_promotion_fp_rate.py`'s live checks were retired in `9185539e` and its
  header is now stale (#1440). Gate floors (1000/1000/40), `TESTS_MARK_EXPR`, the
  eleven rungs, `-n 8`, the `_refuse_the_production_tree` guard, the c0f12db
  stub story and the 09-22 incident record all verified current.
