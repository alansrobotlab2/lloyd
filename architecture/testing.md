---
segment: architecture
tags: [architecture, lloyd, testing, gate, pytest]
type: reference
status: implemented
date: 2026-10-24
---

# Lloyd — the test suite

~21,300 tests under `tests/` — the `tests` rung collected 21,298–21,348 across
the full runs of 2026-10-24, against ~9,600 at this doc's last pass — run by
every automod round's `tests` rung
([[automod]]) and by hand from a throwaway worktree. This doc is about the
suite's own structure: the two kinds of test in it, the rule for tests that read
data off the machine, and the isolation the parallel runner requires. The gate's
thirteen rungs and the promotion flow are in [[automod]].

## Never run the suite against the production tree

`tests/conftest.py::_refuse_the_production_tree` raises `UsageError` if pytest is
invoked with `~/lloyd` as rootdir. This is not tidiness:

> On **2026-09-22 a fixture teardown deleted the production tree in 35 seconds** —
> the whole thing, mid-run.

Everything gitignored went with it and did not come back from git: `.venvs/`,
the model checkpoints, `agent-services/cert/`, `_pipeline/`, `sessions/`,
`eval/baselines/`, `qmd/`. Recovery took a day. Run from a worktree:

```bash
git worktree add --detach ~/lloyd-work/check-$$ HEAD && cd ~/lloyd-work/check-$$ && pytest -q
git worktree remove --force ~/lloyd-work/check-$$   # after
```

`LLOYD_ALLOW_LIVE_TREE_TESTS=1` exists for a human who means it. A round must
never set it.

### 2026-10-07: the guard stays absolute — `live_vault` gets no in-tree path (#2189, owed to #1537)

`live_vault` marks the tests that read the live vault, and the gate deselects them
(`TESTS_MARK_EXPR`, further down). The question that kept coming back was whether a
run that genuinely needs one of those tests should be handed a way through the guard
that keeps it running inside `~/lloyd`. The ruling, settled on #2189's owed entry:
**no such in-tree path exists, and none is planned.** `_refuse_the_production_tree`
runs at conftest import and is
not conditional on the selection — naming one file, or passing `-m live_vault`,
buys nothing, because a single file is the same fixtures with the same teardowns.
`LLOYD_ALLOW_LIVE_TREE_TESTS` remains a human-only opt-in and nothing in this tree
sets it: the two code sites are `tests/conftest.py`, which reads it, and
`tests/test_live_tree_isolation.py`, which clears it before asserting the refusal.

What the runs do instead is the route above. **Every scheduled run that reads the
vault reaches the suite through a detached on-disk worktree under `~/lloyd-work/`**
— the same shape the gate uses to ask "is this failure pre-existing?"
(`scripts/automod/gate.py::_failures_at_base`). Measured on the three nightly
`live_vault` runs after that routing landed, 2026-10-05, -06 and -07: each returned
a pytest summary line and exit 1, 1 and 0, never the guard's exit 4, and each tore
its checkout down clean. So the in-tree alternative would buy no measured gain, and
what it would cost is the 2026-09-22 exposure this section exists to close.

## 2026-09-29: the gate deleted the tree, and the cause was a full `/tmp`

The production tree went a second time, at 15:01:42 PDT, seventeen seconds into
round `SM_20260929_213114`'s gate — not during a test run. `Gate.rung_static`
built its pyflakes scratch path as `Path(_run(["mktemp", "-d"]).stdout.strip())`.
`/tmp` is a tmpfs capped at 1,048,576 inodes and every one was in use (99 GB of
bytes free; the round's own transcript measured it at 15:01), so `mktemp -d`
failed with an empty stdout, `Path("")` was the gate's working directory —
`~/lloyd`, inherited from the aggregator that supervisord starts there — and the
rung's `finally: shutil.rmtree(with_base, ignore_errors=True)` removed everything
under it. The rung **passed** (`ignore_errors` ate the failing `rmdir('.')`); the
`tests` rung then failed on `.venvs/lloyd/bin/python` being gone. The 09-22
deletion above was attributed to a fixture teardown nobody ever identified, in a
round whose gate `tests` rung had just errored on 24 nodes — which is what a full
`/tmp` does to every `tmp_path` fixture — so the same mechanism is the likelier
cause of both.

Three rules now, pinned by `tests/test_gate_scratch_dirs.py`:

- **A scratch directory is created, never computed.** `gate._scratch_dir` is
  `tempfile.mkdtemp` under `<round>/gate-state/scratch/`: it raises when it
  cannot create the directory (and `Gate._rung` records an exception as a FAILED
  rung), and what it returns is a directory that did not exist a moment ago. No
  Python under `scripts/`, `app/`, `agent_mcp/` or `workers/` may build a path
  from a quoted `mktemp` argv.
- **A scratch directory is removed only if it is one.** `gate._drop_scratch`
  refuses a relative or empty path, anything that is not a direct child of the
  round's scratch parent, and anything that is or contains the working
  directory, `$HOME`, the live tree, `~/lloyd-work` or the round dir. It raises
  rather than logs.
- **The gate's children get a short on-disk `TMPDIR`**, `~/lloyd-work/.t/<10 hex>`
  (44 bytes, `gate._child_tmpdir`, removed by the gate's `finally`, a killed
  gate's copy pruned by the next). pytest's basetemp, conftest's scratch roots
  and every test's `tempfile` land there, so a gate run neither fills `/tmp` nor
  fails when something else has. **It must stay short**: a Unix socket path is
  capped at 107 bytes and programs put sockets under TMPDIR. The first cut used
  `<round>/gate-state/tmp` (64 bytes); Chromium's singleton socket adds 45, and
  from 17:30 PDT every Playwright test's browser aborted with "Socket path too
  long" — 65 core dumps and a stream of crash toasts in fifteen minutes, and
  browser-test verdicts in those gates that mean nothing. `MAX_CHILD_TMPDIR` is
  48; a work root too long for that gives the child no TMPDIR rather than a
  long one. A kernel `bind` at Chromium's path shape pins it.

What filled `/tmp` on 09-29, measured by `du --inodes -x -d1 /tmp` at 100%:
`/tmp/pytest-of-alansrobotlab` 666,565, `/tmp/claude-1000` (Claude Code's own
scratch) 245,930, `torchinductor_*` 18,647, and some forty `/tmp/lloyd-check*`,
`/tmp/mut*`, `/tmp/*478`-style worktrees of ~2,300 each — the worktree route this
section used to print, which conftest's refusal message repeated. pytest keeps its
last three runs' `basetemp` whole by default (`tmp_path_retention_policy = all`):
one 600-test file set left 11,884 inodes, so three full-suite runs plus the
leftovers of `timeout`-killed ones is most of the budget. Three changes:

- `pytest.ini` sets `tmp_path_retention_policy = failed`: a passing test's
  `tmp_path` is removed at session end, and only failures are kept to inspect.
- The worktree route is `~/lloyd-work/check-<n>`, on disk, in this doc and in
  conftest's refusal.
- The guardian watches `/tmp` (`agent-services/guardian/tmpwatch.py`): an `error`
  alert at 80% of inodes or bytes, `critical` at 95%, with the largest top-level
  entries named, latched until it falls below 70%. It deletes nothing.

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
`test_uptake.py` is the shape to copy — 112 tests (measured 2026-10-24), all but
a marked handful synthetic, and a small live section whose baseline cross-checks
call the helpers below instead of hard-asserting. `test_promotion_fp_rate.py` is
what a file looks like *after* retirement: 26 all-synthetic tests, its ten live
checks deleted on 2026-09-22 (see below), the marked section headers still
standing where they were — and its module docstring now carries a section saying
the published note is no longer machine-checked, which is the annotation #1440
asked for and has since closed.

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
  verdict. It was a 30-entity stub sitting at
  `~/lloyd-data/_pipeline/vault-derived/kg.sqlite`
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

| Floor | Value (the **floor** of a derived figure) | Catches |
|---|---|---|
| `PYTEST_MIN_COLLECTED` | 1000 | a conftest import failure collecting everything and running none of it |
| `PYTEST_MIN_PASSED` | 1000 | collected ≠ run |
| `PYTEST_MAX_SKIPPED` | 40 | *"a round that skips its way to green is not a round that passed"* |

All three are relative now, each as `max(<its literal>, <derived from the ledger>)`,
so every literal above is a floor and not the figure the rung applies (#3143 made the
skip rail relative; #3250 gave the two pytest floors the same treatment). Both pytest
floors are `max(literal, PYTEST_FLOOR_FRACTION x base)`, one fraction named once at
`gate.PYTEST_FLOOR_FRACTION = 0.25`, with `base` the median `collected` / `passed` of
the five newest green **full** `tests` rungs in the rounds store — the same accepted
set the skip rail reads (rung `ok`, detail not `REUSED`, no `partial` key), and the
round being gated excluded from its own base. On the ledger of 2026-10-24 the bases are
collected 21,341 and passed 21,300, which yields floors of 5,335 and 5,325 (against the
literals' 1000).

Why the absolute 1000 stopped being enough: it was budgeted when the suite was ~9,600 across the `tests` rounds of
2026-09-24, where 1000 was a tenth of it;
against ~21,300 it is under 5%. And `collected` here is what the run reports having
**run plus skipped**, not pytest's collection report (`gate._PYTEST_COUNT_WORDS`), so it
cannot tell a full run from a twentieth of one: a truncated full run that executed 1,200
of ~21,300 nodes and failed nothing cleared both floors and the rung still printed "all
green" (#3250, #3249). With no readable ledger each floor is exactly its literal, which
is what every unit test here and the vacuity probe's floor-at-0 mutant see, because
`tests/conftest.py` points `LLOYD_AUTOMOD_STATE` at scratch before
`scripts.automod.state` is imported. A rung detail says which happened in the same words
the skip rail uses, and both rails are printed on every green full rung as well as on
the two refusals (#3431) — beside the skip clause, ahead of the parallel-flinch clause
and the pin findings, so the 2000-char cap on a stored detail (#2726) cannot drop them:
`collected floor 5335 (base 21341, fraction 0.25, floor 1000); passed floor 5325 (base
21300, fraction 0.25, floor 1000)`. A partial rung prints and records neither, because
the whole-suite floors do not apply to a run over the changed test files.

The skip rail is derived per run as of #3143: `max(PYTEST_MAX_SKIPPED, base +
PYTEST_SKIP_HEADROOM)`, `base` being the median `tests_skipped` of the five newest
green **full** `tests` rungs in the rounds store (not `REUSED`, no `partial` key)
and the headroom is 14 (#3605), so a base of 39 yields a rail of 53 and the rung
prints the pair on its detail as `skip ceiling 53 (base 39, headroom 14, floor 40)`.
That headroom is one lost dependency's measured skip width plus one, not one pin
file's node count: on 2026-10-31 a round whose `web/node_modules` and chromium were
both absent came in **13** skips over a base of 37 while printing
`DASHBOARD_PINS_NOT_EXECUTED: 10 of 10 pins in tests/test_dashboard_responsive.py`
naming both reasons — the same two absent dependencies take dependent nodes outside
that file with them, so a room sized to the file's ten nodes refused a round for the
box it ran on, and the refusal's sentence blamed the diff.
The absolute
figure had stopped describing the suite — the ledger read 39 skips against 40, a
margin of one, and the counted set moves with the tree, because the dashboard pins
skip whenever a worktree has no `web/node_modules`. A fixed rail therefore refused
rounds for a fact no diff had changed. With no readable ledger the rail is exactly
40, which is also what every unit test sees: `tests/conftest.py` points
`LLOYD_AUTOMOD_STATE` at scratch before `scripts.automod.state` is imported.

The skip cap is still the one to respect when a round's fix converts failures into
skips. That is exactly the move it exists to catch, and the correct response is
almost never to raise it — it is to ask whether those tests can ever pass again,
and retire the ones that cannot. Raising the number should be the last resort and
should say in a comment what it is hiding, and raising the **figure** stays a human
ruling that no open item asks for (`dbfddc60`'s `PYTEST_MAX_SKIPPED = 55` is still
unreferenced by `main`; #2205 and #2214, which asked for it, closed as unnecessary
once the margin was derived):
#3143 derived the margin, it did not move the floor.

## Marks and selection

`TESTS_MARK_EXPR = "not live_vault and not fault_injection"`. The gate deselects
both.

The trap: **a marked test is invisible to the gate but still red in a full run.**
`test_conversation_relations.py::test_live_trajectories_yield_material_co_access_signal`
carried `live_vault`, was red at head for days, and entered no gate report —
it surfaced only when someone ran the whole suite. Run unmarked periodically.

## Pre-existing failures are subtracted, not tolerated

`rung_tests` probes the base commit before blaming a round: `_failures_at_base`
re-runs the failing node ids in a throwaway worktree with this round's diff
absent, and a failure the base reproduces **and** whose file the diff does not
touch is subtracted from the round (`_failure_outside_round`); the set is cached
in the round's `base-red-set.json` because the probe costs a whole suite.
Everything else — including anything in a file the round touched — stays owned by
the round and blocks. The reasoning behind that second half is in [[automod]] (a
test the diff itself touched cannot be excused by a probe), so the rule is stated
once there and not re-derived here.

The subtraction is gated on the probe having reached a conclusion. Failures
outside the round's own files that the base probe does **not** reproduce leave the
rung unable to attribute them, and it fails closed — *"every failure outside the
round's own files was unexplained, and the base probe could not say whose they
are"* (`scripts/automod/gate.py:3669`) — filing nothing. Only a conclusive pass with an inherited
red files the debt: `_file_red_tree` runs on `pre_existing and conclusive`
(`scripts/automod/gate.py:3640`), and the verdict names the nodes under `pre_existing_failures`
with the tracker's id in `red_tree_item`. There is no `external_blocker` /
`external_failures` pair in the gate — that was the pre-#2876 shape, and a round
whose report contains neither key did not hit an inherited red.

Read those keys rather than `failed_node_ids` alone: a red node id names a test
and never says why it is red, which is why each red node's own pytest body now
rides on the verdict in `failure_excerpts` (#2876) — including on the pass, the
branch that used to leave no output on disk at all. A round that passes with a
red tree is the signal to go fix base, not a reason to relax about the failure.

## Parallel isolation is a requirement, not a nicety

The rung runs `-n 8` — the count is `automod.gate.test_workers`
(`config.yaml`), and `_test_workers` falls back to serial when that key is
missing or unusable or the venv has no `xdist`, and always for a `--only` subset.
Tests from different files share a worker process, so
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

## A test about untracked state may not require that state to be present

(Moved from CLAUDE.md § Tools, 2026-09-25.) The end-to-end test in
`tests/test_tool_overrides.py` asserted `(ROOT / "data/tool_overrides.yaml").exists()`
with `ROOT` resolved from `__file__` — so in an automod worktree it demanded a
file that is, by design, never checked out. `data/` has no tracked contents at
all, so it failed for **every round from `d11ad8c` onward, whatever the diff
under test**, and the `tests` rung was a hard rung: three rounds aborted on it
in fifteen hours while filing fourteen new backlog items and landing nothing.
The loop's only drain was blocked by a test asserting that a deliberately
untracked file had been checked out. It now *writes* the file through the real
writer — `app.paths.LLOYD_HOME` resolves from `__file__` too, so the writer and
the test address the same tree in a worktree — and asserts git never sees it,
which is also the stronger check: the old form could only observe a file
somebody else had already written and reverted. The `.<pid>.tmp` sibling
`atomic_write_text` lands before renaming is ignored and asserted too. Why the
file is untracked at all is [[tools]] §8.

## A gate rung must not depend on the wall clock

Two `tests/test_guardian_speak.py` tests asserted that an alert dispatches
voice, while `speak.dispatch` consults `in_quiet_hours` against the real clock
and a 23→07 default window — so they failed whenever the gate ran overnight,
which is when the unattended loop runs. Rounds gated at 23:52 and 04:07
failed; the same code at 08:13 did not. They pin the policy off explicitly now
(`_AWAKE`), as the tests that are *about* quiet hours already pinned it on.

## An assertion that cannot fail pins nothing

`x == [] or True` shipped in the effect ledger's first cut (#544, 2026-09-10)
through every gate rung, and the round that wrote it declared its acceptance
`deferred` to an empty list. It is one of the two reasons the gate grew a
review rung that refuses a test that cannot fail ([[automod]] §4.5;
[[editing-safeguards]] has the effect-ledger side).

## Frontend unit tests: vitest, run by the gate

`web/` had no test runner until 2026-09-17, so a frontend clause had no node a
gate could grade — #1199's mount cascade and its unconditional 15 s poll were
held open as a human clause for exactly that reason, and `package.json` is a
build input the loop may not land. Alan's call: add one. `npm test` is
`vitest run`; tests sit beside what they test as `src/**/*.test.ts(x)`;
`web/vitest.config.ts` is deliberately not `vite.config.ts`, which reads TLS
certs and loads the Monaco/Tailwind plugins at import. The gate's `frontend`
rung runs it after `vite build` and a failure fails the rung. No config or no
binary is a skip that says so (`vitest SKIPPED (…)`, `vitest: None` on the
event), never a pass: `node_modules` is untracked, and a tree that gained the
dependency before anyone ran `npm install` in `~/lloyd/web` must neither fail
every frontend round nor read as tested. A triage clause on the `frontend`
surface can name a `web/src/**.test.ts` file the way a code clause names
`tests/<file>.py`.

## The dashboard pins' dependency arrives by symlink from the parent checkout

Since 2026-10-05 (#2233), a pytest run in a **linked git worktree** of this repo is
given the frontend its own tree lacks: `tests/conftest.py` and
`tests/test_dashboard_responsive.py::_vite_binary` both call
`tests/frontend_deps.ensure_node_modules_link`, which resolves the main working tree
from this tree's `.git` pointer file and, if THAT tree has a working
`web/node_modules/.bin/vite`, makes `<this tree>/web/node_modules` a symlink to it. A
symlink, never a copy (the install is 641 MB, and `package.json` and the lockfile are
paths the gate denies, so a copy could only ever be a stale duplicate), and never over
an existing `web/node_modules` — real directory or dangling link, it is left exactly as
found. The path is gitignored (`.gitignore:13`), so the link is invisible to `git
status` and can never enter a diff.

Why it exists: the ten pins in `tests/test_dashboard_responsive.py` skip when their
tree has no vite, and before this the only thing that ever created one was the gate's
`frontend` rung, which returns early for a diff that did not touch `web/`. So every
non-frontend round skipped ten tests, the suite crossed `PYTEST_MAX_SKIPPED = 40`
(44-45 observed on 2026-10-05), and the `tests` rung failed rounds for a reason no diff
could answer. That fix left the ceiling at 40: the fix was that the pins run, not that
the ceiling rises, and raising the figure has not happened since — the 55 sits on a
branch (`dbfddc60`, unreferenced by `main`), and the two items that asked for it closed
as unnecessary once #3143 made the rail move with the tree. (#3143 later
made the rail the base tree's count plus a headroom over that same 40 — it moved the
margin, not the figure, and this paragraph's point stands: an unavailable dependency
must be made to run, not counted into a wider rail.) What the
hook must never do is turn an unavailable dependency into a failure: on a box where
nobody ran `npm install`, the ten nodes keep skipping with the reason they always
named, the `DASHBOARD_PINS_NOT_EXECUTED` finding is still printed, and the run exits 0.
`LLOYD_FRONTEND_PINS_AVAILABLE` stays unset by the same reasoning: that declaration
means *on this box the pins really measured*, and `tests/dashboard_pins.py` turns a
declaration contradicted by an unavailable-pin skip into a failing run
(`PinNotExecuted`) rather than a quiet pass — a pin that ran and failed is a
regression, a pin that never ran is a machine fact, and the second must never be
filed as the first.
Provisioning therefore happens in `tests/`, not in `scripts/automod/gate.py`: the gate
imports itself from the live tree, so a gate-side fix could never land itself, while
pytest runs with `cwd=` the worktree and executes the round's own conftest.

## Guard vacuity

`scripts/maintenance/guard_vacuity.py` probes production guards by running each
against inputs it should PASS and BLOCK, and again with the guard mutated. A guard
whose real and mutant verdicts never differ is **VACUOUS** — it is not stopping
anything. Sites are found by **searching for the guard's own symbol**, never by a
pinned line number, because a number is a snapshot and the symbol moves with the
code.

## The triage replay set

`eval/triage_replay/cases.json` holds 42 past single-triage decisions whose label
is what happened to the item, never what a model thought of it: 40 `good_confirm`
(confirmed, landed, closed, and not reopened in 14+ days; bookkeeping-named items
left out; taken across inflow sources) and 2 `bad_confirm` (confirmed, then closed
by a person as superseded). Each case stores the item as triage saw it — the file
at the last vault commit before the verdict — plus the vault and tree shas of that
moment. `build.py` regenerates it from the ledger and the vault's history and is a
hand tool; `run.py` renders each case through the production
`autotriage.render_prompt`, sends it to a `prompt -> text` callable, parses with the
production `parse_verdict`, and reports agreement per label with a Wilson interval,
`unmeasured` under n = 20.

Two limits, both stated in the files. The set is **one-sided**: the record holds
1,996 candidates for the first label and 2 for the second, because nothing in the
loop ever revisits a retirement and a round's `rejected`/`unnecessary` turned out,
read one by one, to mean "did not land" or "another landing got there first"
(`architecture/automod.md` §12.3). It can say whether a prompt change stops
confirming work that landed; it cannot say whether triage confirms too much. And
the replay is a **plain completion with no tools**: a tooled turn against today's
tree would correctly call every `good_confirm` `already_done`, since its fix has
landed. It is not run by the gate and it needs the engine:

    .venvs/lloyd/bin/python eval/triage_replay/run.py plan
    .venvs/lloyd/bin/python eval/triage_replay/run.py run --base-url http://127.0.0.1:8096

The runner sends each case with reasoning **off** (`chat_template_kwargs:
{enable_thinking: false}`). The first recorded run, 2026-11-22, had it on with a
3,000-token budget, and 36 of 42 cases spent the whole budget reasoning and returned
no content (`no_verdict`, about 21 s each). One case probed both ways: reasoning on
reaches its verdict in about 8,000 tokens and 60 s, reasoning off in 489 tokens and
4 s. `--thinking` restores reasoning with a 16,000-token budget; rows from the two
modes are not comparable. `judgment_replay record` refuses to store a triage baseline
in which more than a quarter of the cases returned no verdict.

`tests/test_triage_replay.py` drives the runner with a fake model.

## Behaviour benches

Every scorecard row measures the loop's process and the regression check
measures retrieval; until 2026-11-22 nothing on a schedule said whether Lloyd's
*behaviour* moved. `eval/` holds 84 top-level scripts that measure some part of
it, almost all hand-run. This section is the inventory of those scripts, the
seven that now run as one nightly fitness pass, and how that pass decides that
a number moved.

### The inventory (read 2026-11-22; nothing was run)

Read from each script's docstring, argparse and call sites. `unknown` means the
code and the docs do not say — almost no script records its own runtime, so
that column is mostly `unknown` and is not a guess. Columns: **engine** — does
a run call a model (`yes` = the primary or secondary; `djev`, `qmd`, `embed`,
`TTS` when only those; `helper` = no CLI). **tools** — does it replay prompts
through real tool dispatch (`schema-only` = the `tools` array is sent and the
chosen call is scored, nothing is dispatched). **sandboxed** — for a live-tools
bench only: does it mint a session id `agent_mcp/_tool_sandbox.py` sandboxes
(`bench_…`, `pt-eval-…`, or `<date>_<time>_bench_<hex>`). **det.** — `yes`
(offline, no sampling), `temp 0`, `no`, or `corpus` (deterministic for a given
live corpus, which drifts).

| script | measures | engine | tools | sandboxed | runtime | det. | artifact |
|---|---|---|---|---|---|---|---|
| `action_equivalence.py` | tool-call equivalence tiers between two engine captures | no | no | n/a | unknown | yes | stdout |
| `aggregator_binding.py` | nothing: resolves where the egress child finds `agent_mcp` | helper | no | n/a | n/a | yes | none |
| `answer_bearing_rows.py` | answer-bearing sentences as djev's ranked row | djev, qmd | no | n/a | unknown | no | `--out`, undated |
| `artifact_named_in_final_message.py` | whether note-writing sessions name the note at the end | no | no | n/a | unknown | corpus | stdout |
| `autoresearch_judge_compare.py` | scalar vs binary rubric judge noise | yes | no | n/a | unknown | no | tree, fixed tracked name |
| `bench_lint.py` | bench task validity | no | no | n/a | unknown | yes | stdout |
| `calibrate_dup_threshold.py` | near-duplicate threshold agreement | djev on rebuild | no | n/a | unknown | `--check` yes | tree, undated |
| `canary_rows_guard.py` | nothing: who may append to the tracked canary corpus | helper | no | n/a | n/a | yes | none |
| `check_doc_matches_json.py` | whether a context-rot doc's figures equal its JSON | no | no | n/a | unknown | yes | stdout |
| `ci_backtest.py` | past retrieval verdicts re-decided against 95 % intervals | no | no | n/a | unknown | yes | optional `--out` |
| `compare_tool_choice.py` | two tool-choice runs against the per-metric noise floor | no | no | n/a | unknown | yes | stdout |
| `contextual_full_corpus.py` | LLM situating context on the full recall corpus | yes, qmd | no | n/a | unknown | unknown | data root, undated |
| `contextual_titles.py` | the same on the #1493 sub-corpus | yes, qmd | no | n/a | unknown | temp 0 | data root, undated |
| `counterfactual.py` | whether one flipped constraint moves retrieved entities | qmd | no | n/a | unknown | unknown | unknown |
| `cross_engine_disagreement.py` | whether primary/djev disagreement localises verdict error | yes, djev | no | n/a | unknown | temp 0 | `--out` dir |
| `decision_replay_588.py` | whether triage re-proposes retired decisions | yes | yes | yes (`bench_588_…`) | unknown | no | tree, appended, undated |
| `diagnosis_fanout_replay.py` | K=3 theory fan-out vs linear diagnosis | yes | yes | yes (`bench_2258_…`) | unknown | no | data root, timestamped |
| `djev_name_prior_probe.py` | whether djev follows an option's name or its definition | djev | no | n/a | unknown | no | data root, dated |
| `dup_detect.py` | near-duplicate predicate and vault census | no | no | n/a | unknown | yes | stdout |
| `egress_arms.py` | canary under enforce-off vs a private enforce-on aggregator | yes | yes | yes (via canary) | "hours" | no | canary rows file |
| `embed_side_index.py` | embedding-model A/B on a side index | embed, qmd | no | n/a | "hours" | unknown | data root, undated |
| `engine_output_probe.py` | primary output-integrity drift | yes | no | n/a | unknown | temp 0 | data root |
| `episodic_arm.py` | raw-transcript retrieval arm | qmd | no | n/a | unknown | unknown | tree, undated |
| `extraction_disagreement.py` | fact-extraction error from two independent passes | yes | no | n/a | unknown | no | `--out`, timestamped |
| `fact_write_gate_snapshot.py` | nothing: builds the fact-tree pair for the write gate | unknown | no | n/a | unknown | unknown | `--out` |
| `gold_blind_menu_probe.py` | gold-blind candidate menus for the label ceiling | embed (optional) | no | n/a | "minutes" | yes | stdout |
| `instruction_compliance_eval.py` | instruction-density compliance per engine slot | yes | no | n/a | unknown | temp 0 | data root, stamped, top level of `eval/baselines/` |
| `kv_dtype_arm.py` | output drift under another KV dtype — **restarts the primary** | yes | no | n/a | unknown | no | unknown |
| `label_agreement_ceiling.py` | retrieval-blind second labelling pass | yes | no | n/a | unknown | temp 0 | data root, stamped |
| `label_audit.py` | mechanical gold-shape defects in a LloydMemEval set | no | no | n/a | unknown | yes | stdout |
| `measure_parallel_dispatch.py` | share of traffic parallel dispatch could speed up | no | no | n/a | unknown | corpus | stdout |
| `measure_session_io_stalls.py` | event-loop stalls from session-file I/O | no | no | n/a | unknown | no | stdout |
| `memory_eval_build.py` | nothing: builds the frozen LloydMemEval set | yes | no | n/a | unknown | no | data root, then tree |
| `persistence_arms.py` | canary persistence arms with worker dispatch held | yes | yes | yes (via canary) | unknown | no | canary rows file |
| `prefetch_conflict_count.py` | conflicting look-alikes in merged prefetch | no | no | n/a | unknown | corpus | stdout |
| `qmd_contention.py` | cost of the eval pin on production qmd, from logs | no | no | n/a | unknown | yes | optional `--json` |
| `recall_topics_merge.py` | topics merge on `vault_recall` vs production | djev, qmd | no | n/a | unknown | no | `--out` |
| `refusal_detour_baseline.py` | retry-at-the-same-effect rate after a refused tool call | no | no | n/a | unknown | corpus | tree, dated (`--out` redirects) |
| `refusal_detour_labels.py` | nothing: hand labels for the detour dev set | helper | no | n/a | n/a | yes | none |
| `retrieval_holdout.py` | nothing: holdout manifest and reserve rule | no | no | n/a | unknown | yes | unknown |
| `run_compaction_recall_eval.py` | planted-fact recall under compaction policies | yes | no (local stub) | n/a | unknown | no | `--out`, undated |
| `run_context_rot_eval.py` | needle recall vs prompt length | yes | no | n/a | unknown | temp 0 | tree, dated |
| `run_egress_arms.sh` | wrapper for `egress_arms.py` | yes | yes | yes (via canary) | unknown | no | as above |
| `run_episodic_recall_eval.py` | whether transcripts earn recall slots | qmd, djev | no | n/a | unknown | no | `--out` |
| `run_eval.py` | vault retrieval (entity, doc hit, MRR) — **the nightly** | qmd, djev | no | n/a | unknown | no | data root, stamped |
| `run_fact_write_gate_eval.py` | fact writes replayed through the ADD/UPDATE/NOOP gate | djev | no | n/a | unknown | no | `--out` |
| `run_find_all_arms.py` | "find all" vs "find one" phrasing on audit tasks | yes | yes | yes (`…_bench_<hex>`) | unknown | no | `--out` dir |
| `run_framing_acceptance_eval.py` | whether replies adopt a wrong user framing | yes, djev | yes | yes (`run_bench_sdk`) | unknown | no | tree, undated |
| `run_harness_replay_diff.py` | behaviour diff of two trees' `run_query` on replayed sessions | no (scripted) | no (scripted pool) | n/a | unknown | yes | `--out` |
| `run_injection_canary.py` | whether a planted instruction drives a sink call | yes | yes | yes (`new_background_session_id("bench")` + sandbox pre-flight) | median 23.8 s per episode (286 tracked rows) | no | tree, appended; guarded |
| `run_kv_dtype_arm.sh` | wrapper for `kv_dtype_arm.py` | yes | no | n/a | unknown | no | as above |
| `run_memory_eval.py` | LloydMemEval: remembers and acts on told facts | yes (djev judge optional) | no | n/a | 31 min for 1,275 prompts (2026-09-25) | no (temp 0.6, seeded) | data root, undated |
| `run_memory_index_ab.py` | A/B of MEMORY.md as a typed index | yes | yes | yes (`run_bench_sdk`) | unknown | no | tree, undated |
| `run_memory_trim_ab.py` | A/B of one restored USER.md line | yes | yes | yes (`run_bench_sdk`) | unknown | no | tree, undated |
| `run_midturn_recall_probe.py` | whether mid-turn recall would have anticipated fetches | qmd | no | n/a | unknown | unknown | `--out` |
| `run_observation_compression_eval.py` | tool-observation compressor by decision preserved | yes | schema-only (unverified) | n/a | unknown | temp 0 | `--out` |
| `run_persistence_arms.sh` | wrapper for `persistence_arms.py` | yes | yes | yes (via canary) | unknown | no | as above |
| `run_prefetch_cost_eval.py` | downstream cost the prefetched `<context>` removes | yes | yes | yes (`pt-eval-562-…`) | unknown | no | tree, stamped |
| `run_prefetch_eval.py` | prefetch vault leg doc hit, MRR, latency | qmd | no | n/a | "~5 s" lex-only | unknown | data root, stamped |
| `run_prefetch_followup_eval.py` | facts and skills legs on a short follow-up | no | no | n/a | unknown | unknown | stdout |
| `run_prefetch_route_eval.py` | whether one djev read can gate `<context>` | djev | no | n/a | unknown | no | `--out` |
| `run_preserve_thinking_eval.py` | A/B of `preserve_thinking_iterations` | yes | yes | yes (`pt-eval-<keep>-<epoch>`) | unknown | no | optional `--out` |
| `run_reasoningbank_eval.py` | whether banked lessons would have named a refusal's cause | yes | no | n/a | unknown | temp 0 | data root, undated |
| `run_rpc_eval.py` | whether `lloyd_rpc` cuts prompt tokens | yes | yes | **NO** (`…_rpceval_…`; needs `--live-bash`, hook denies all but Bash + read-only) | unknown | no | `--out` dir |
| `run_scratchpad_ab.py` | prefill cost of the Scratchpad tool | yes | yes | **NO** (worker id on purpose; annotation deny-list) | unknown | no | tree, dated |
| `run_skill_activation_eval.py` | per-skill trigger recall and false-trigger rate | no | no | n/a | unknown | yes | stdout, optional `--json` |
| `run_skill_dispatch_probe.py` | dispatch-time skill rule trigger rate | no | no | n/a | unknown | corpus | stdout, optional `--json` |
| `run_skill_match_eval.py` | prefetch skill matcher recall@5 | no (djev/embed in optional arms) | no | n/a | unknown | yes (default arm) | stdout, optional `--json` |
| `run_stale_fact_eval.py` | whether the model acts on a superseded value | yes | no (shadow store) | n/a | 1,414 s for five arms × 3 samples (2026-09-24) | no | tree, dated (`--out` redirects) |
| `run_tool_choice_eval.py` | first tool chosen on web-shaped prompts | yes | schema-only | n/a — no session id at all | unknown | no | data root, stamped |
| `run_transcript_self_hit_audit.py` | how often a prompt retrieves its own transcript | qmd; `probe`: yes | `probe`: schema-only | n/a | unknown | unknown | optional `--out` |
| `run_tts_fidelity_eval.py` | whether the cloned voice says the words sent | TTS + ASR | no | n/a | unknown | no | data root, dated, top level of `eval/baselines/` |
| `run_user_md_curation_dryrun.py` | rationale-per-line curation on copies of USER.md | yes | no | n/a | unknown | temp 0 | `--out` |
| `run_vault_search_eval.py` | `vault_search` document quality per pipeline | qmd, djev | no | n/a | unknown | no | data root, stamped |
| `secondary_routing_eval.py` | primary vs secondary on the secondary's jobs | yes (both) | no | n/a | unknown | unknown | tree, undated |
| `self_question.py` | retrieval against LLM-drafted recall questions | yes, qmd, djev | no | n/a | unknown | temp 0 | `--out` |
| `speaker_embed_eval.py` | speaker-embedding model bake-off | local models | no | n/a | unknown | unknown | `--work` dir |
| `stats.py` | nothing: Wilson and bootstrap interval arithmetic | helper | no | n/a | n/a | yes | none |
| `step_replay_fidelity.py` | whether disk bytes rebuild a recorded step's prompt | no | no | n/a | unknown | corpus | stdout |
| `summary_length_attribution_probe.py` | summary length vs marker survival in canary rows | no | no | n/a | unknown | yes | stdout |
| `summary_length_repetition_run.py` | summariser length distribution on one fixture | yes | no | n/a | unknown | no | tree, dated |
| `tool_call_equivalence_probe.py` | tool-call decisions captured per engine arm | yes | schema-only | n/a | unknown | temp 0 | data root, labelled |
| `tool_choice_none_probe.py` | whether the primary renders `tools` at `tool_choice: none` | yes | schema-only | n/a | unknown | temp 0 | stdout |
| `tool_failure_baseline.py` | per-tool failure signatures from transcripts | no | no | n/a | unknown | corpus | dated |

What the read turned up that a nightly has to know:

- **Two benches dispatch live tools without a sandboxed id**: `run_rpc_eval.py`
  and `run_scratchpad_ab.py`. Both are deliberate and each carries its own
  deny rule, but neither is the tool-layer sandbox, so neither may be put on a
  schedule (vault-protection.md, layer 1).
- **`run_tool_choice_eval.py` is safe by a `break`, not by a guard.** It opens
  the live MCP pool, sends the real `tools` array and closes the generator at
  the first `tool_call` event; it mints no session id. It is in the nightly
  set on that reading — if that loop is ever changed to let a call dispatch,
  it must mint a `_bench_` id first.
- **`kv_dtype_arm.py` restarts the primary engine.** Attended only.
- **`run_injection_canary.py run` appends to a tracked file** unless
  `LLOYD_CANARY_ROWS` redirects it; `eval/canary_rows_guard.py` refuses a
  caller that is not one of the two window units and did not set it.
- **Almost nothing records its own runtime.** Three numbers exist, all quoted
  in the table.

### The seven that run nightly, and the ones that do not

`eval/fitness/manifest.yaml` is the list. A bench is in it only if it cannot
change the machine (sandboxed id, or no live tools), yields one 0..1 share with
an n, and finishes inside a declared timeout. The engine-minute figures are
**estimates** — pro-rated from the three recorded runtimes or guessed, and
marked which — until `fitness noise` has timed each bench once.

| bench | covers | script | value (n) | better | engine min (est.) |
|---|---|---|---|---|---|
| `tool_choice` | tool choice | `run_tool_choice_eval.py --label fitness` | `summary.overall.correct_rate` (20 queries) | up | 8 (guess: 20 cold ~34k prefills) |
| `injection_canary` | injection canary | `run_injection_canary.py run`, then `grade` | `attack_success_rate` (attack scenarios) | down | 12 (20 × 23.8 s median, plus persistence passes) |
| `stale_fact` | memory | `run_stale_fact_eval.py --arms facts_appended --samples 3` | `stale_action.rate` | down | 5 (pro rata from 1,414 s / 15 arm-samples) |
| `memory_history` | memory | `run_memory_eval.py run --arms history --judge rules` | `dev.history.all.correct_strict` (dev set) | up | 10 (pro rata from 31 min / 1,275 prompts) |
| `refusal_detour` | refusal detour | `refusal_detour_baseline.py --since <30 d ago> --no-devset` | `all.same_effect_rate` (`all.measurable`) | down | 0 (offline) |
| `skill_match` | skill match | `run_skill_match_eval.py` (lexical arm) | `arms.lexical.recall@5` (labelled turns) | up | 0 (offline) |
| `skill_activation` | skill match | `run_skill_activation_eval.py` | `corpus.recall` (positives) | up | 0 (offline) |

35 estimated engine minutes against a budget of 60; the loader refuses a
manifest whose `engine_minutes` add up to more than `budget_engine_minutes`.
The manifest lists the three offline benches first: the nightly run stops
*starting* benches when its budget runs short, so whatever is last is what a
slow night drops.

Named in the plan and **not** in the manifest, with the reason:

- **Retrieval** — already nightly (`autonomy/82`, the `retrieval_nightly` arm of
  the instrument index). Running it twice a night would buy nothing.
- **Compaction** (`run_compaction_recall_eval.py`) — 60 rows by default on
  90k–130k-token histories with no recorded runtime, needs a tools snapshot,
  and refuses to start unless `LLOYD_DATA` is exported to a scratch root, which
  a scheduled child of the backend must never do.
- **Context rot** (`run_context_rot_eval.py`) — 450 cold plus 450 warm requests
  at 50k–240k tokens, pauses the pool itself, and publishes a grid, not one
  rate. A windowed run by its own docstring.
- **Voice first-token latency** — nothing in `eval/` measures it.
  `scripts/maintenance/voice_turn_trend.py` reads the spoken-turn store, but it
  publishes percentiles in seconds, the store holds 53 rows in total
  (2026-11-22), and turning a latency into a share needs a budget the script
  deliberately declines to invent (`EOS_TO_AUDIO_P90_BOUND_S = None`).
- **Instruction compliance** — a mean over six requests, not k of n, and it
  writes into the top level of `eval/baselines/`.

Three of the seven are not readings of one commit:
`refusal_detour` scores the last 30 days of real sessions, so it moves with
traffic; `skill_match` and `skill_activation` score the live skill inventory,
which is in the vault and is rewritten nightly. A move in any of them is a
move in Lloyd's behaviour but not necessarily in Lloyd's code.

### How a fitness run works

`python -m scripts.automod.fitness run --once [--only NAME] [--repeat N]`
(`scripts/automod/fitness.py`):

1. **Preconditions.** If any selected bench calls the engine, the pool must be
   **paused with nothing in flight** (`GET /api/workers/status`) or the run
   refuses, exit 2. A momentary idle reading is not idle. `--force` overrides
   and stamps `mode: force` on every row. A run of only offline benches needs
   neither the pool nor the lock.
2. **`regression.lock`.** Engine runs hold the regression runner's own lock
   (`automod_regression.REGRESSION_LOCK`) for the whole pass, so a fitness run
   and a regression measurement never overlap. Held already: exit 3, nothing
   run (`--lock-wait SECONDS` queues instead).
3. **One bench at a time.** Each bench's commands run in order as child
   processes in their own process group, in a scratch directory the runner made
   and removes, under the bench's `timeout_s`; a timeout kills that group. The
   pool is re-read before each engine bench: a pool resumed mid-run means the
   remaining engine benches are `skipped`, not run against live traffic.
4. **One row per bench per repeat** appended to `app.paths.FITNESS_SERIES`
   (`fitness.jsonl` in the data root's eval directory): run id, bench, commit (the tree's `HEAD`)
   and `backend_commit` (`/health`), value, n, k, Wilson 95 % interval,
   duration, exit codes, pool state. A bench that timed out, failed, wrote no
   artifact or wrote nothing at the declared pointer carries its status and
   `value: null` — never a zero. Only an artifact written *by this run* is
   read; an older file at the same path is some other run's number.
5. **One closing `summary` row** with every bench's newest value from the last
   36 h in the rate-record shape (`{k, n, rate, ci_low, ci_high}`). This is the
   row the instrument index reads, one `fitness_<bench>` arm per bench
   (`scripts/eval_instrument_index.py::fitness_instruments`). Those arms are
   carried in the daily table at weight 0.0 and are never averaged; what they
   add is that a night with no fitness run prints `missing or stale:
   fitness_…` in the index headline.
6. **Attribution, record only.** If a bench *regressed* outside its noise floor
   (below), one `fitness_move` event is appended to the promotions ledger: the
   bench, the delta, the two commits, and the `promoted` events between the two
   measurements. No bisect runs and no rollback is requested.

The nightly caller is the `fitness` worker source
(`workers/sources/fitness.py`, `workers.sources.fitness.enabled`, **ships
false**): once a night inside `[window_start_hour, window_end_hour)` local, one
subprocess off the event loop. It cannot ask for a paused pool — it is one of
the pool's own in-flight jobs — so it runs `--pool-job`: before each engine
bench it waits up to 10 minutes for the pool to hold nothing but itself with
the KV gate released, and records the bench `skipped` if it never does. That is
a weaker guarantee than a paused pool (another job can still be claimed
mid-bench), which is why the noise floors are taken by hand with the pool
paused and why `pool` and `mode` are on every row. Taking a machine pause for
the hour instead was considered and not built: a machine hold whose holder died
is reconciled only at the next backend boot (`workers/pool.py`), and an
unattended nightly that can leave the fleet halted is a worse failure than a
noisier number.

### Noise floors

`python -m scripts.automod.fitness noise --repeat 5` runs each bench five times
at one commit and writes, per bench, min, max, mean, stdev and the per-repeat
durations to `app.paths.FITNESS_NOISE_PATH` (`fitness_noise.json` beside it). The repeats are not appended
to the series. A bench whose spread (max − min) exceeds its manifest
`noise_max_spread` is flagged `too-noisy`.

`fitness report`, `fitness.moves(now)` and scorecard row 22 all use one rule
(`bench_delta`): the 7-day delta is the latest value minus the newest value at
least 7 days older, and it is **outside noise** only if its magnitude exceeds
that bench's measured spread **and** the latest row's own Wilson interval does
not cover the older value. Both, because neither alone is a floor: a bench that
is deterministic on a fixed set (the two skill benches) measures a spread of
zero, and one that scores live traffic (`refusal_detour`) measures a spread
that says nothing about next week's traffic. A bench with no measured floor, a `too-noisy`
one, or one with no row old enough has `outside_noise: null` and a sentence
saying which — it is neither a move nor a clean reading. Until the floors are
measured, nothing can be reported as moved, by construction.

`tests/test_automod_fitness.py`; the index arms are pinned in
`tests/test_eval_instrument_index.py`.

## Where the tests rung's time goes (measured 2026-11-22)

The rung's cost grew faster than the suite. Green full 8-worker runs, per week,
from the ledger's `gate` rows (`rung: tests`, median seconds; median tests passed):

| week of | 09-14 | 09-28 | 10-12 | 10-26 | 11-09 | 11-16 |
|---|---|---|---|---|---|---|
| rung, s | 93 | 269 | 558 | 714 | 814 | 1,035 |
| tests passed | 6,671 | 14,344 | 19,348 | 21,913 | 24,812 | 25,748 |
| ms per test | 14 | 19 | 29 | 33 | 33 | 40 |

In the week to 2026-11-22, over the 164 rounds whose last gate report
(`rounds/<id>/gate.json`) holds a green full run: median 1,004 s (896 s net of
the `gate-tests.lock` wait, which averaged 291 s), and **all 164 paid a serial
re-run** for at least one test that failed only under parallel load.

**The profile.** One full run from a clean worktree of `28c0743e`, the gate's own
argv and child environment (`-m "not live_vault and not fault_injection" --dist
loadfile`, round-home `HOME`, scratch `LLOYD_DATA`, on-disk `TMPDIR`) at `-n 4`
with `--durations=0 --durations-min=0.5`: 26,176 tests in 916 files, 1,855.7 s
wall. 56 failed and none of them is a finding about time: the 11 clone-only nodes;
37 that reproduce serially in that hand-built worktree and round home and in no
gate run (194 green full rungs in the ledger that week), cause not chased; and 8
that are order-dependent (six in `test_kv_gate.py`, plus the two fixed below).

- **Collection and startup are not the cost.** `--collect-only` takes 11.2 s
  (14.9 s for the process); each xdist worker pays it once, in parallel. Test time
  summed over the four workers is 7,011 s of 7,423 worker-seconds (94 %).
- **Setup vs call, among the 1,258 phase rows of 0.5 s or more:** call 4,448 s,
  setup 88 s, teardown 25 s. No module fixture is slow enough to matter; the one
  slow setup is `test_degradation_selection.py` (48 s).
- **The tail is short and heavy.** 153 tests take 5 s or more and are 3,125 s
  (45 %); 829 take 1 s or more and are 4,373 s (62 %). The top 10 files are 28.5 %
  of the time, the top 40 are 50.1 %.
- **The body was fixture overhead.** The other 25,347 tests total 2,638 s; the
  median test took 83 ms, and most of that was `conftest.py`, not the test (below).
- **By kind** (static classification of each test function, times from this run):
  tests that spawn a subprocess, 1,865 tests and 1,751 s (25 %); tests that read
  live data off the machine, 510 and 379 s; the citation rails (below), 2,214 and
  777 s.

Top 40 files by summed test time (`-n 4` run, before any change below):

| file (`tests/`) | seconds | tests | share |
|---|---|---|---|
| `test_architecture_coverage_doc_claims.py` | 330 | 307 | 4.7% |
| `test_eval_corpus_guard.py` | 329 | 77 | 4.7% |
| `test_automod_code_rules.py` | 307 | 82 | 4.4% |
| `test_thinking_trace.py` | 248 | 46 | 3.5% |
| `test_landed_outcomes_human_input.py` | 205 | 24 | 2.9% |
| `test_dashboard_responsive.py` | 142 | 23 | 2.0% |
| `test_uptake.py` | 127 | 118 | 1.8% |
| `test_automod_vault_round.py` | 108 | 267 | 1.5% |
| `test_automod_frontend_probe_canary.py` | 106 | 15 | 1.5% |
| `test_dashboard_yaml_loader.py` | 98 | 11 | 1.4% |
| `test_vllm_doc_claims.py` | 97 | 150 | 1.4% |
| `test_guard_arm_matrix.py` | 89 | 15 | 1.3% |
| `test_automod_vet.py` | 85 | 20 | 1.2% |
| `test_validate_tasks_linter.py` | 81 | 75 | 1.2% |
| `test_flash_next_launcher.py` | 71 | 66 | 1.0% |
| `test_eval_ci_reporting.py` | 68 | 9 | 1.0% |
| `test_run_egress_arms.py` | 60 | 23 | 0.9% |
| `test_dashboard_pin_accounting.py` | 59 | 12 | 0.8% |
| `test_automod_frontend_probe.py` | 50 | 25 | 0.7% |
| `test_degradation_selection.py` | 49 | 4 | 0.7% |
| `test_retention_sweep.py` | 49 | 281 | 0.7% |
| `test_gate_parallel_tests.py` | 48 | 40 | 0.7% |
| `test_outbound_content_gate.py` | 45 | 96 | 0.6% |
| `test_run_persistence_arms.py` | 44 | 19 | 0.6% |
| `test_automod_gate.py` | 44 | 191 | 0.6% |
| `test_eval_scorer.py` | 43 | 83 | 0.6% |
| `test_autoresearch_bench_runner_sdk.py` | 43 | 64 | 0.6% |
| `test_path_sandbox_skip_accounting.py` | 43 | 10 | 0.6% |
| `test_guard_vacuity.py` | 43 | 56 | 0.6% |
| `test_browser_upload.py` | 42 | 26 | 0.6% |
| `test_code_rules_census.py` | 41 | 15 | 0.6% |
| `test_bench_mine_source.py` | 40 | 14 | 0.6% |
| `test_autonomy_scheduler.py` | 40 | 161 | 0.6% |
| `test_vault_commit_attribution.py` | 39 | 35 | 0.6% |
| `test_run_kv_dtype_arm.py` | 39 | 26 | 0.6% |
| `test_dashboard_cold_render.py` | 36 | 43 | 0.5% |
| `test_workers_pool.py` | 34 | 121 | 0.5% |
| `test_check_health_census.py` | 32 | 19 | 0.5% |
| `test_bench_side_effects.py` | 32 | 8 | 0.5% |
| `test_loop_depth.py` | 32 | 56 | 0.4% |

### What was changed

1. **The six per-test scratch directories no longer list the base temp**
   (`tests/conftest.py`, `_fresh_scratch_dir`). Six autouse fixtures each called
   `tmp_path_factory.mktemp(<prefix>)` for every test. That is pytest's
   `make_numbered_dir`, which lists the whole base temp to find the next number,
   and nothing removes those directories during the session — so each test left
   six more entries for every later test to list six times. Setup cost was
   quadratic in the number of tests a worker had run: six `mktemp` calls cost
   0.7 ms in an empty base temp, 45.5 ms at 20,000 entries, 91.8 ms at 40,000,
   and the profile run ended with 22,041–53,220 entries per worker. The
   directories are now `<basetemp>/isolate/<prefix>-<n>`, named from a counter.
   Same contract: a fresh, empty, existing directory per test, kept for the
   session. **Measured:** the same 6,516 tests from 119 files, one process,
   6,508 passed both times — 564.6 s before, 175.3 s after (−69 %). Fitted cost
   9.2 µs × (tests the worker has run)², which is ~98 s per worker at an even
   8-way split of 26,176 tests and more on whichever worker `loadfile` hands the
   most tests. This is also the term that made ms-per-test rise in the table above.
2. **`test_automod_code_rules.py` asks its whole-tree question once**
   (`_whole_tree_run`). Eight nodes each ran `CR.run` over every tracked `.py`
   (37–39 s apiece). The pass is memoised on the file list, the rule file's bytes
   and the tree's HEAD plus diff, and every caller gets a deep copy. File total
   307 s in the profile → 34.9 s alone, same 81 passed / 1 clone-only failure.
3. **Two parallel-only failures that were process-state leaks, not load.**
   `test_bash_tree_strays.py::test_the_probe_reached_the_file_read_only_and_wrote_nothing`
   asserted that `datawatch` was absent from `sys.modules`, true only when no
   earlier file on the worker had imported the guardian's module: it failed and
   was re-run serially in 164 of those 164 rungs (530 of 667 over 30
   days). It compares before and after the probe now, by identity.
   `test_automod_state_isolation.py::test_a_landing_gives_back_the_signal_handlers_it_took`
   (65 of 164) read `promote._LANDING_MERGED`, a process global another file can
   leave set; the node pins it to the pre-merge state it is about.
4. **Three dead names removed** (plan 5.5): `backlog.QUARANTINE_TAGS` and
   `backlog.SPAWN_TRIAGE_MIN_AGE_DAYS` (no reader but one identity assertion
   each; the readers are `quarantine_tags()` and `spawn_expiry_days()`), and
   `autocode.ABANDON_GRACE_SECONDS` (no production reader since the grace became
   `abandon_grace_seconds`, default 0; four tests used it as a number). All three
   removals are line-neutral.

**Estimated effect on the rung** (not measured at `-n 8`; only the gate can):
about 100 s from (1), about 34 s from (2) (272 worker-seconds), 12–20 s from the
re-run every round no longer makes — roughly 150 s of a 900 s run, and the
quadratic term is gone, so the next 10,000 tests cost what they cost and no more.

### What was left as a proposal

| proposal | what it would save | why it was not taken |
|---|---|---|
| Fix the remaining order- and load-dependent nodes: `test_dashboard_responsive.py` (re-run in 64 of 164 rungs, and its serial re-run is ~140 s of real browser), `test_skills_search_metadata_gate.py` (41), `test_tsc_runner.py` (29), `test_thinking_fidelity_scan_single_walk.py` (22), `test_kv_gate.py` (17: `SOURCE_REGISTRY` read after another file replaced it) | ~70 s per rung on average, almost all of it the dashboard file | each needs its own diagnosis; the two fixed above were the ones whose cause was one line |
| `automod.gate.test_workers` 8 → 12 or 16 (the box has 32 cores) | up to half the run, in proportion | a config value with a load cost: every parallel-only failure above gets likelier, and the engines share the box. Needs a measured trial with the pool's own turns running |
| Move tests that wait on a live service or scan a live store off the per-round rung to a scheduled run: the `eval/run_eval.py` CLI nodes (`test_eval_corpus_guard.py`, `test_eval_ci_reporting.py`; ~33 s each, 7 s of it CPU, the rest the live qmd daemon), `test_thinking_trace.py`'s live-store node (243 s), `test_landed_outcomes_human_input.py` (three nodes, 61–73 s), `test_dashboard_yaml_loader.py` (72 s) | ~1,000 worker-seconds, ~125 s per rung | it removes coverage from the gate; that is a ruling, not a speed-up |
| `test_architecture_coverage_doc_claims.py`: one node runs a child pytest for 163 s, another 57 s | ~220 worker-seconds | the child run is the assertion; splitting it needs its author's intent |
| Affected-test selection (below) | unknown until shadowed | skipping tests in the gate is a safety change |

**Affected-test selection.** The gate selects in exactly one case today:
`_tests_delta_only` re-runs only the changed test files when the delta since a
cached green run at the same base is test files alone (no `conftest.py`, no
`tests/_*.py`). Everything else is a full run. A wider selection from the code
graph (`agent_mcp/code_graph.py`, `graph_affected`: reverse-BFS from the changed
symbols to the test files that import them) is plausible and should be built in
this order, each step measured before the next:

1. **Shadow.** Compute the selected set, run the full suite anyway, and record on
   the rung's data which failing nodes the selection would have missed. No verdict
   changes.
2. **The always-run set is most of the design.** The graph is an AST import graph
   and is blind to everything this suite is unusually full of: 2,214 tests that
   read source or docs as text (the rails), 1,865 that execute a script in a
   subprocess, monkeypatches by dotted string, the HTTP/MCP seams, `config.yaml`
   and every non-Python file. Any change to `conftest.py`, a `tests/_*.py` helper,
   a data file or a config file is a full run.
3. **Promote only on zero misses** over a few hundred shadowed rounds, and keep a
   full run somewhere a miss cannot land past — the landing re-gate, or the first
   gate after a rebase.

The risk is the quiet one: a missed test is a green rung, and nothing in the
ledger distinguishes it from a real pass until something breaks in production.

### The citation rails, counted

2,214 collected tests (8.5 %) are citation rails by this count: the 1,395 in the
31 files named `*doc_claims*`, `*citation*`, `*cites*` or `*doc_repoint*`, plus
every other test whose body carries a `file.ext:<line>` string literal (925 with
such a literal in all). They cost 777 s of the 7,011 s profile (11 %), 330 s of it
`test_architecture_coverage_doc_claims.py` alone.

What they cost in red rungs, from every `tests` gate event in the ledger:

| window | red rungs | name a node | name at least one rail | name only rails | rung time on the only-rail reds |
|---|---|---|---|---|---|
| 7 days | 46 | 46 | 32 (70 %) | 26 (57 %) | 14.3 h of 23.9 h |
| 30 days | 204 | 197 | 111 (56 %) | 77 (39 %) | 34.0 h of 78.7 h |

`test_architecture_coverage_doc_claims.py` is 181 of the rail nodes blamed in the
30-day rounds' final gates; no other file has more than 14. A red rail is often
the rail working — the round moved a line and owes the re-point — so this is the
price of the rails, not a count of false alarms. The classification is static and
approximate: a `file:line` literal in a fixture string counts, a rail that builds
its citation at run time does not.

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
  twelve rungs, `-n 8`, the `_refuse_the_production_tree` guard, the c0f12db
  stub story and the 09-22 incident record all verified current.
- 2026-10-24 — **stale**: the suite has doubled since that pass — a full `tests`
  rung collected 21,298–21,348 across the rounds of 2026-10-24 — so the opening
  "~9,600" is out, and `architecture/index.md` still quotes that figure for
  [[testing]] (#3249). The `external_blocker` / `external_failures` pair no longer
  exists: a base failure is subtracted only off a **conclusive** probe, an
  inconclusive one fails the round and files nothing, and a conclusive pass files
  a `red-tree` item and reports `pre_existing_failures` / `red_tree_item` (#2876).
  `-n 8` is a config value now (`automod.gate.test_workers`), and the 2026-10-05
  pins passage still named #2214, which has since closed as unnecessary once
  #3143 derived the rail. Corrected in place: the KG stub's path is the data root,
  `test_uptake.py` (112 tests) and `test_promotion_fp_rate.py` (26) recounted,
  #1440 closed, today's ceiling reading (48) set beside the 47, and `dbfddc60`'s
  55 recorded as still unreferenced by `main`. Verified current and untouched: the
  1000/1000/40 constants, `TESTS_MARK_EXPR`, the scratch and `TMPDIR` rules,
  `tests/frontend_deps.py` and its gitignore line, the vitest sentence, the
  vacuity probes, tmpwatch's fractions, and both incident records. Filed: the two
  1000-test floors are under 5% of what they were set for (#3250), and
  `scripts/automod/gate.py` names closed #2214 as the live ruling on the figure (#3251).
- 2026-11-22 — **profiled**: "Where the tests rung's time goes" added — one full
  `-n 4` run of 26,176 tests (1,855.7 s), the ledger's weekly rung medians, the
  top-40 table, the quadratic base-temp scan in six autouse fixtures (fixed), the
  eight-fold whole-tree pass in `test_automod_code_rules.py` (shared), two
  process-state leaks behind "failed only under parallel load" (fixed), the rails'
  share of red rungs, and the proposals not taken. The opening "~21,300 tests" is
  stale by ~4,900 and was left for the next pass that re-verifies the whole page.
