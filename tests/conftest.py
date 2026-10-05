"""Test isolation from live runtime state.

The suite must not depend on the state of the running system. Two things
leaked in and broke tests that had nothing to do with them:

  * `config.yaml knowledge_graph.write_enabled`, which a knowledge-graph
    rebuild sets to false — six fact-write tests started failing because a
    rebuild was in progress on the machine.
  * `app.kg_store`'s process-default store, which points at the live
    database unless a test configures it.

Both are forced to a known value here. A test that wants the other value
patches it explicitly.
"""
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _default_state_dirs_to_scratch() -> None:
    """No pytest run reads or writes the machine's automod, guardian or request-manifest state.

    `scripts.automod.state` and the guardian's `policy` resolve their state
    dir from the environment **at import**, and default to the production
    path. `gate._child_env` sets both variables so a candidate's tests cannot
    reach the production ledger — but that covers the gate's own pytest only.
    A round's model runs the suite from Bash with a plain environment, thirty
    times a day (31 launches across 22 turns on 2026-09-18), and every one of
    those addressed `~/.local/state/lloyd-automod` by default, relying on each
    test to patch each path it touches. One did not: a `land()` whose leaked
    SIGTERM handler wrote `land_failed` for the fixture round `SM_L` into the
    live ledger, twice in one day, when the model `pkill`ed its own run.

    Set here, at conftest import, because that is the one moment that is
    before every test module's `from scripts.automod import state`. A caller
    that already chose a state dir (the gate, `review_tools`) keeps it.

    `LLOYD_MANIFEST_STORE` joined that list on 2026-09-19, for the same reason in
    a sharper form: `app/component_manifest.py` (#581) writes one NDJSON line per
    model request to `~/.local/state/lloyd-request-manifests` by default, and the
    first full-suite gate run that had the module in its tree put **48 fabricated
    lines** from fixture strings into that directory (`app/harness/finalizer.py`
    38, `app/secondary_models.py` 6, `app/harness/client.py` 4). The store is the
    artifact #581 exists to produce, and the clause that closes it is read off it
    — "24 h of mixed traffic read off the live store: zero manifest-write errors"
    — so a suite that writes there is a suite that corrupts its own acceptance
    evidence. Tests that own a store point the variable at their own `tmp_path`,
    as the three `test_component_manifest*`/`test_prompt_diff` files already do.

    `LLOYD_DAILY_NOTE_DIR` joined on 2026-09-21 for the same reason one level up
    the tree: `autonomy._append_fast_failure_alert` (#1209) appends a line to
    today's note under `~/obsidian/memory/` by default, and an existing
    failure-backoff test already drives a task through three sub-second failures
    without caring where an alert might go. Measured with this guard removed and
    the variable aimed at an empty directory: the scheduler, timeout, silence and
    grant-policy suites (173 tests, all passing) wrote one `2026-09-21.md`
    carrying one `Autonomy #` alert line, 387 bytes. A fixture writing "Autonomy
    #N failed 3 times in a row" into the live note is the same class of pollution
    as the manifest lines above, and worse, because the note is the surface a
    human reads to decide whether the fleet is healthy. Tests that assert on the
    alert point it at their own `tmp_path`.
    `LLOYD_EGRESS_DB` joined on 2026-09-21 with #628, and it is the sharpest case
    in this docstring: `agent_mcp/egress.py` records every destination the four
    web tools name, defaulting to the live `~/lloyd/workers.db`, and telemetry is
    ON by default — so a suite that fetches `example.com` in a fixture appends
    rows to the same table the item's own step 2 is derived from. The allow-list
    a human seeds is read off that table; fixture hosts in it are not noise, they
    are a forged inventory. Tests that assert on destinations point the variable
    at their own `tmp_path`, as `tests/test_egress_telemetry.py` does.
    `LLOYD_DAILY_NOTE_APPEND_LEDGER` joined on 2026-10-05 with #2213, as the
    ledger's half of the `LLOYD_DAILY_NOTE_DIR` pair: `app.autonomy._witness_daily_note_append`
    (#1799) appends one JSONL row per CONFIRMED alert line to
    `~/lloyd-data/alerts/daily-note-appends.jsonl` by default, so with the notes
    redirected and the ledger not, any suite test driving a confirmed append
    witnesses into the file the `daily_note_appends` health leg reads — green
    evidence no alarm produced (a single `mktemp -d` probe row made that leg
    report `witnessed (1 rows examined, 0 lost)` on 2026-10-04). The writer now
    refuses that combination outright; this default is the second half of the
    invariant, so the suite's own default resolves nowhere near the production
    ledger. Tests that assert on witness rows point it at their own `tmp_path`,
    as `tests/test_daily_note_append_witness.py` does.
    """
    scratch: Path | None = None
    for var, sub in (("LLOYD_AUTOMOD_STATE", "automod"), ("LLOYD_GUARDIAN_STATE", "guardian"),
                     ("LLOYD_MANIFEST_STORE", "request-manifests"),
                     ("LLOYD_DAILY_NOTE_DIR", "daily-notes")):
        if os.environ.get(var):
            continue
        if scratch is None:
            scratch = Path(tempfile.mkdtemp(prefix="lloyd-test-state-"))
            atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        (scratch / sub).mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(scratch / sub)
    if not os.environ.get("LLOYD_EGRESS_DB"):
        # A file, not a directory: the tuple above creates directories, and this
        # one is the sqlite path `agent_mcp.egress.db_path()` reads.
        if scratch is None:
            scratch = Path(tempfile.mkdtemp(prefix="lloyd-test-state-"))
            atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        os.environ["LLOYD_EGRESS_DB"] = str(scratch / "egress-events.db")
    if not os.environ.get("LLOYD_DAILY_NOTE_APPEND_LEDGER"):
        # Same file-shaped case (#2213), same scratch dir: the JSONL witness path
        # `app.autonomy._daily_note_append_ledger()` reads. The notes dir above
        # is redirected unconditionally, so WITHOUT this the pair-invariant skip
        # in `_witness_daily_note_append` would fire in every test that drives a
        # confirmed append — correct for pollution, but the un-override witness
        # path would have no in-suite coverage at all.
        if scratch is None:
            scratch = Path(tempfile.mkdtemp(prefix="lloyd-test-state-"))
            atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        os.environ["LLOYD_DAILY_NOTE_APPEND_LEDGER"] = str(
            scratch / "daily-note-appends.jsonl")


_default_state_dirs_to_scratch()


# `scripts/automod/vault_guards.py` judges a vault land by running the tree's
# vault-reading selection as a pytest subprocess (~70 s, one throwaway worktree
# and one whole vault mirror per call), and `vault_round.land` calls it. `land()`
# is called ~25 times by `tests/test_automod_vault_round.py` alone; left alone,
# every one of those nests a second full selection inside the suite. This variable
# is the nesting rule, not an off switch: production never sets it, so every real
# landing is probed, and the tests that pin the probe itself hand `agreement`
# roots of their own and clear it. Same shape as the block just above — the suite
# must not reach the machine, and here must not reach a second copy of itself.
os.environ.setdefault("LLOYD_VAULT_GUARD_PROBE", "1")


def _data_root_to_scratch() -> None:
    """No pytest run writes into the machine's data root (`~/lloyd-data`).

    `app.paths.DATA_ROOT` resolves at import and, in the production checkout,
    names the live sessions, `workers.db` and logs. Pointing `LLOYD_DATA` at a
    scratch directory here, before any test module imports `app.paths`, is what
    keeps every store the suite touches — including those no fixture isolates —
    off the live one. A caller that chose a root of its own (the gate sets the
    round's) keeps it; a caller that chose the LIVE root is refused, with the
    same opt-in as the tree guard below.
    """
    try:
        import pwd
        live = (Path(pwd.getpwuid(os.getuid()).pw_dir) / "lloyd-data").resolve()
    except Exception:  # pragma: no cover
        live = None
    chosen = os.environ.get("LLOYD_DATA")
    if chosen:
        if (live is not None and Path(chosen).expanduser().resolve() == live
                and os.environ.get(LIVE_TREE_OPT_IN) != "1"):
            raise pytest.UsageError(
                f"refusing to run the suite with LLOYD_DATA={chosen}, the live data "
                f"root. Leave LLOYD_DATA unset and conftest gives the run a scratch "
                f"root. If you are a human and you mean it, set {LIVE_TREE_OPT_IN}=1.")
        return
    scratch = Path(tempfile.mkdtemp(prefix="lloyd-test-data-"))
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    os.environ["LLOYD_DATA"] = str(scratch)

#: Set to "1" to run the suite against the production checkout anyway. Named on
#: the refusal below, because a guard whose way past it is undocumented gets
#: worked around by deleting the guard.
LIVE_TREE_OPT_IN = "LLOYD_ALLOW_LIVE_TREE_TESTS"


def _production_tree() -> Path | None:
    """`~/lloyd` as the ACCOUNT defines it, not as `$HOME` says it.

    `Path.home()` honours `$HOME`, and since 2026-09-22 `gate._child_env` sets
    `$HOME` to the round's own home precisely so that `Path.home()/"lloyd"` is
    the WORKTREE — that is the other half of this incident's fix. Reading
    production through `Path.home()` would therefore make this guard fire on
    every gate run and on nothing else: the loop would stop running its tests,
    which is the one failure worse than the one being prevented. The passwd
    entry is the anchor `$HOME` cannot move.

    `None` rather than a guess when the entry cannot be read: this guard exists
    to refuse a specific directory, and a guard that cannot name it has nothing
    to refuse.
    """
    try:
        import pwd
        return (Path(pwd.getpwuid(os.getuid()).pw_dir) / "lloyd").resolve()
    except Exception:  # pragma: no cover - no passwd entry, or an unresolvable home
        return None


def _refuse_the_production_tree() -> None:
    """The suite may not run with `~/lloyd` as its own tree.

    Every test that builds a repo, a home or a store resolves it from somewhere,
    and the two anchors are `app.paths.LLOYD_HOME` (from `__file__`) and
    `Path.home()/"lloyd"`. Both are the PRODUCTION tree when the suite is
    launched from `~/lloyd`, so a fixture's teardown is aimed at the running
    system rather than at a checkout nobody minds losing.

    On 2026-09-22 that happened. An implement round's gate `tests` rung had
    failed with 24 errors, and the model re-ran the full suite **in the live
    tree** to decide whether those failures were pre-existing — 9,716 node ids,
    rootdir `/home/alansrobotlab/lloyd`, starting 11:15:57. At 11:18:50
    something under it removed the tree: `.git`, `.venvs`, `qmd`, `data`,
    `web/node_modules`, the model weights, every tracked file, in about 35
    seconds. Three directories survived with their original inodes — `~/lloyd`,
    `web/` and `event_logs/` — because live writers re-created files inside them
    mid-walk and the closing `rmdir` hit ENOTEMPTY, which is the signature that
    identifies it as a recursive delete of the root rather than a `git clean`.

    No layer below could see it. `app/harness/protected_paths.py` parses Bash
    COMMAND STRINGS, and `pytest tests/` is not destructive on its face; the
    deletion happened inside the test process, in Python, which is not a tool
    call. So the refusal belongs at the one place every invocation passes
    through however it is spelled — conftest import, before a single fixture
    runs.

    Anchored on the passwd entry, not on `$HOME` — see `_production_tree`. A
    round's worktree is never refused, which is what keeps the gate working.

    The supported route is a throwaway checkout: `git worktree add --detach
    <path> HEAD` and run there. The gate already does exactly this
    (`gate._failures_at_base`), which is what makes "is this failure
    pre-existing?" a question nobody needs the live tree to answer.

    Not a `pytest.skip` and not a warning: both leave the run going. Not
    conditional on the selection either — a single file is the same fixtures
    with the same teardowns, and "how much of the suite" was never what made
    this safe or unsafe.
    """
    if os.environ.get(LIVE_TREE_OPT_IN) == "1":
        return
    production = _production_tree()
    if production is None or ROOT.resolve() != production:
        return
    raise pytest.UsageError(
        f"refusing to run the suite against the production tree at {production}. "
        f"A fixture teardown here deletes the running system, and on 2026-09-22 "
        f"one did — the whole tree in 35 seconds. Run from a throwaway checkout "
        f"instead, on disk and never under /tmp (a 1M-inode tmpfs that filled on "
        f"2026-09-22 and 2026-09-29): `git worktree add --detach "
        f"~/lloyd-work/check-$$ HEAD && cd ~/lloyd-work/check-$$ && pytest ...`, "
        f"and `git worktree remove` it after. To answer 'does this failure already "
        f"exist at base?', that IS the supported route (see "
        f"scripts/automod/gate.py::_failures_at_base). If you are a human and "
        f"you mean it, set {LIVE_TREE_OPT_IN}=1.")


def _provision_frontend_deps() -> None:
    """Give this tree the frontend dependencies its parent checkout already has (#2233).

    Called here, at import, rather than from `gate.py`, for the reason
    `tests/frontend_deps.py` gives in full: the gate runs pytest with
    `cwd=self.worktree`, so this file — the round's own — is the surface on which a fix
    to the round's environment can be judged by the round that changes it. And it runs
    per worker process, which is where the race the helper settles actually lives: the
    tests rung runs `-n <workers> --dist loadfile`, and every worker imports this file.

    What it can cost a run is one symlink, `web/node_modules`, pointing at the parent
    checkout's install (641 MB, gitignored, and identical by construction because
    `package.json` and the lockfile are paths the gate denies). What it cannot do is
    fail: a tree it cannot link keeps the skips it always had, so a box where nobody ran
    `npm install` is still green.
    """
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        # Named explicitly rather than left to pytest's `prepend` import mode, which
        # does put a conftest's own directory on the path: a hook that depends on an
        # import detail nobody verified is a hook that silently stops linking.
        sys.path.insert(0, tests_dir)
    try:
        import frontend_deps                      # sibling module of this file
    except ImportError:                           # a partial tree: not this hook's business
        return
    frontend_deps.ensure_node_modules_link(ROOT)


_refuse_the_production_tree()
_data_root_to_scratch()
_provision_frontend_deps()


@pytest.fixture(autouse=True)
def _no_voice_alerts_in_tests(monkeypatch):
    """The guardian's sixth channel synthesises speech and plays it aloud.

    `Notifier.alert` fans out to it like any other channel, so without this an
    ordinary `pytest tests/` would talk to the room — and, worse, would do it
    from a detached process that outlives the test. Muted for every test; the
    ones that exercise the channel assert on the dispatch decision instead.
    """
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "0")


@pytest.fixture(autouse=True)
def _memory_entries_unstamped_in_tests(monkeypatch):
    """`memory_tools.date_stamp_entries` is on in production (#622), and
    `memory_add` then prefixes each entry with its write date. The writer-lane
    and characterization tests assert entries byte for byte, so they run on the
    code default (off); the tests that exercise the stamp set it themselves.
    `typed_entries` (review 2026-09-24 P4) is held off for the same reason.
    """
    try:
        from app import config
    except Exception:
        return
    monkeypatch.setitem(config.CONFIG, "memory_tools",
                        {**(config.CONFIG.get("memory_tools") or {}),
                         "date_stamp_entries": False, "typed_entries": False})


@pytest.fixture(autouse=True)
def _no_desktop_or_journal_alerts_in_tests(monkeypatch):
    """The other two channels that reach the room from a test process.

    `Notifier`'s `external` gate suppresses vault notes and backlog tasks, and
    was added after drill rollbacks polluted the live vault on 2026-09-06. It
    never covered the toast or the journal line, because those need nothing
    but a session bus — so a test that builds a default `Notifier` and only
    cares whether *voice* dispatched still paints the user's screen.

    That is what happened on 2026-09-07: every self-mod gate run toasted
    `Lloyd guardian: real rollback / body` and wrote
    `STILL BROKEN :: 2026-09-06 liveness failed` to the live journal at
    priority 2, from the fixture strings in `test_guardian_speak.py` and
    `test_guardian_predicates.py`. A fake critical incident in the journal is
    worse than a stray toast: it is the record you consult *after* an
    incident, and it now contains fiction.

    DBUS is blanked as well as the switch flipped, so the nag unit's
    bash `notify-send` fallback — which reads the environment, not
    `LLOYD_DESKTOP_ALERTS` — cannot fire from a test either. Tests that assert
    on the channel's dispatch decision set these explicitly.
    """
    monkeypatch.setenv("LLOYD_DESKTOP_ALERTS", "0")
    monkeypatch.setenv("LLOYD_JOURNAL_ALERTS", "0")
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)


@pytest.fixture(autouse=True)
def _promoter_cannot_reach_the_live_backend(monkeypatch):
    """No test asks the RUNNING backend whether a round is in flight.

    `round.land` waits for the other round's turn by reading the live pool
    (`promote.wait_for_rounds`), up to 75 minutes. A test that reaches it
    unpatched is then a test whose duration depends on what production is
    doing — it hung the first time it ran beside a real round (2026-09-17).
    The discard port refuses at once, which reads as "pool state unreadable"
    and returns. A test about the promoter's HTTP patches `_get`, as before.

    One unreadable poll is enough here: production waits out
    `ROUNDS_UNREADABLE_POLLS` of them, a minute, which no test should spend.
    And the aggregator is as unreachable as the backend, so
    `promote.restart_needed` never asks the LIVE services what they have
    loaded — it fails closed to "restart", the landing every older test means.
    """
    try:
        from scripts.automod import promote
        monkeypatch.setattr(promote, "BACKEND", "http://127.0.0.1:9")
        monkeypatch.setattr(promote, "MCP_HEALTH", "http://127.0.0.1:9/health")
        monkeypatch.setattr(promote, "ROUNDS_UNREADABLE_POLLS", 1)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _review_grader_cannot_reach_the_live_backend(monkeypatch):
    """No test posts a grading turn to the RUNNING backend.

    On 2026-10-01 `automod.review.confirm` went to `shadow` in config.yaml
    (#2017). The review-rung tests stub the first reader (`RV.grade`) and were
    written while the policy was off, so none of them stubbed the second one:
    from that commit on, every refusal a fixture staged went through
    `Gate._review_confirm` to the real `review.run_grader`, which resolves the
    LIVE backend and writes its session into the live data root. 705 turns and
    32.3M input tokens in one day, 35-50 per gate `tests` run, each a worker
    session titled `review #7 (SM_REV)` grading "the thing happens once"
    against `(no diff)` — and nothing read the votes, because the same tests
    stub the ledger.

    The tripwire: `review.backend_url` is how `run_grader` and `cancel_grader`
    find the live backend when a caller named none, and it is the first thing
    `run_grader` does — before the session file. `pytest.fail` rather than a
    raise, because `review.confirm_refusal` reads a reader that raised as
    "upheld" and the test would pass having learned nothing. Not the discard
    port the promoter fixture uses: `run_grader` retries a refused connection.
    A test about the transport passes `backend=` or replaces `run_grader`.

    And the policy those tests assumed is pinned rather than inherited: the
    suite reads `automod.review.confirm` as `off`, the code default, whatever
    config.yaml carries this week. A test about `shadow` or `on` replaces
    `app.config.CONFIG` itself, after this fixture, and wins.
    """
    try:
        from app import config as C
        from scripts.automod import review as RV
    except Exception:
        return
    automod = dict(C.CONFIG.get("automod") or {})
    automod["review"] = {**(automod.get("review") or {}), "confirm": RV.CONFIRM_OFF}
    monkeypatch.setitem(C.CONFIG, "automod", automod)

    def _refuse(root=None):
        pytest.fail("a test reached scripts.automod.review.backend_url with no stub: it was "
                    "about to POST to the live backend. Replace `RV.run_grader` (or pass "
                    "`backend=`) — see _review_grader_cannot_reach_the_live_backend.")

    monkeypatch.setattr(RV, "backend_url", _refuse)


#: The commands a test must never be allowed to execute (#1853). Chosen by COMMAND
#: name, never by a later token: `app/harness/service_control.py:186` (systemctl) and
#: `:202` (pkill/killall) reduce an argv to a verdict the same way on the production
#: side, and matching a later token would refuse `tests/test_data_home.py:1404`, which
#: stops a whole fake fleet through `subprocess.run(["bash", "-c", program])` (its
#: runner, `tests/test_data_home.py:1330`) with a STUB `systemctl` first on PATH — there
#: the command name is `bash`, and the word `systemctl` sits inside one quoted token of
#: the program. `supervisorctl` is deliberately NOT in this set: a bare `supervisorctl
#: status` is a read the suite asks for on purpose (`tests/test_service_control_guard.py`
#: answers it ALLOWED), so refusing the name would refuse the query, not the change.
DANGEROUS_SUBPROCESS_COMMANDS = frozenset({"systemctl", "pkill", "killall"})


class LiveServiceControlRefused(RuntimeError):
    """A test asked to run one of `DANGEROUS_SUBPROCESS_COMMANDS`, and did not stub it.

    A `RuntimeError` subclass so a node that only expects "it refused" catches it
    without importing anything, and a node that wants to name which guard fired can
    catch this. The message carries the whole argv: the point of the error is that the
    reader can see WHICH command was reached for.
    """


def _dangerous_subprocess_command(cmd: object) -> str | None:
    """The command `cmd` would execute, if it is a refused one; else None.

    `cmd` is what `subprocess.run`/`subprocess.call` received as argv: a list/tuple
    whose first element is the program, or one string for a shell. The DECIDING field
    is the command name — argv[0]'s basename for a list, the first token's basename for
    a string — so `/usr/bin/pkill` is refused exactly as `pkill` is, and an absolute
    path never smuggles a danger word past the set. The string is split on whitespace,
    not with `shlex`: an unbalanced quote inside a shell program must not raise here,
    because this function runs on the way to commands that are about to run for real.
    """
    if isinstance(cmd, (list, tuple)):
        if not cmd:
            return None
        first = cmd[0]
    elif isinstance(cmd, (str, bytes, os.PathLike)):
        first = cmd
    else:
        return None
    first = first if isinstance(first, str) else os.fsdecode(first)
    if isinstance(cmd, (str, bytes, os.PathLike)):
        first = first.split(maxsplit=1)[0] if first.strip() else ""
    name = os.path.basename(first)
    return name if name in DANGEROUS_SUBPROCESS_COMMANDS else None


@pytest.fixture(autouse=True)
def _no_live_service_control_in_tests(monkeypatch):
    """No test may restart the supervisor, kill the sync or reload systemd for real.

    Three commands reach the machine instead of the checkout, all of them on the
    guarded names' ordinary path. `tick()` issues a real
    `subprocess.run(["systemctl", "--user", "restart", policy.SUPERVISORD_UNIT])` as
    soon as `sup_down_streak` passes `policy.SUPERVISORD_DOWN_STREAK` (3) —
    `agent-services/guardian/guardian.py:1233-1236`, `policy.py:54` — `_stop_sync`
    issues a real `pkill -f "sync --path …"` at `guardian.py:1054`, and the promoter
    reaches `systemctl --user daemon-reload` (`scripts/automod/promote.py:586`) and
    `systemctl --user restart lloyd-guardian` (`:595`) through its own thin `_run`
    wrapper (`promote.py:542`), which calls the module-global `subprocess.run`.

    Until #1853 the only protection was each test remembering its own
    `monkeypatch.setattr(subprocess, "run", fake_run)`, so it existed exactly where
    someone had already been bitten: the two nodes that drive the supervisord branch
    stub it themselves (`tests/test_guardian_pool_watch.py:357`,
    `tests/test_guardian_predicates.py:1080`), the first of those saying why in a
    comment (:344-350 — "a red node of that shape restarts production's supervisor
    from inside the gate's pytest run"), while the promoter's nodes defend the WRAPPER
    and not the call (`tests/test_automod_hardening.py:1091`, `:1116` stub `P._run`).
    A new module that stubs neither re-inherits the trap, which is what spent the
    implement turns of `SM_20260928_190007`. And `_refuse_the_production_tree` (this
    file, `:160`) already refuses to run the suite against the production tree, which is
    precisely why that is not an answer: a worktree isolates files, not
    `systemctl --user`.

    Two names are wrapped, and why those two is worth keeping exact. `check_output`
    reaches the guard through the module-global `run`, and `check_call` through the
    module-global `call` — measured on this interpreter (CPython 3.12.14), where
    `check_call` calls `call` and `call` calls `Popen`, so `check_call` does NOT pass
    through `run`. `Popen` is deliberately not wrapped: the tests that use it stub it
    separately (`tests/test_guardian_speak.py:214`, `:233`, `:258`), and wrapping the
    two module-level entry points covers every way the suite spells these three
    commands today.

    A node that installs its own stub still wins: its `monkeypatch.setattr` runs after
    this fixture's and `undo()` pops in reverse, so the existing per-node interceptions
    keep their behaviour untouched and this fixture is only the floor beneath them —
    which is what keeps `tests/test_guardian_vaultwatch.py`'s four nodes that run the
    real sync/backup/restore scripts (`:158`, `:189`, `:217`, `:225`) running them.
    """
    real_run, real_call = subprocess.run, subprocess.call

    def _refuse_or_forward(original, *args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        danger = _dangerous_subprocess_command(cmd)
        if danger is not None:
            raise LiveServiceControlRefused(
                f"refused to run {danger} from a test, which would reach the machine "
                f"and not this checkout: argv={cmd!r}. Stub it in the node "
                f"(monkeypatch.setattr(subprocess, \"run\", fake_run)) — see "
                f"tests/conftest.py::_no_live_service_control_in_tests. #1853")
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _refuse_or_forward(real_run, *a, **k))
    monkeypatch.setattr(subprocess, "call",
                        lambda *a, **k: _refuse_or_forward(real_call, *a, **k))


@pytest.fixture(autouse=True)
def _primary_engine_probe_is_not_consulted(monkeypatch):
    """No test's claim depends on whether the real primary engine is up.

    The pool's third claim gate (`workers/pool.py::_primary_hold_held`) asks
    `127.0.0.1:8096/health` before it will let an `autocode` round claim, so left
    live it would make every pool test's answer depend on whether the engine
    happened to be loading its weights while the test ran — and, worse, a test
    suite run while the engine is down would hold a source and then report it as
    a real result. Same hazard as the two fixtures above (a live process reaching
    into a test's verdict), and the same remedy taken from them: neutralise the
    outside signal, do not stub the code under test.

    Disabled rather than pointed at a discard port, and that distinction is the
    whole fixture: an unreachable engine is the gate's ON state, so redirecting
    the URL would engage the hold in every test and quietly change what each one
    asserts. The gate's own behaviour is covered in `tests/test_round_hold.py`,
    which re-enables it with a fake probe.
    """
    try:
        from workers import pool as P
        monkeypatch.setattr(P, "primary_hold_config", lambda: {"enabled": False})
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_automod_lock(tmp_path, monkeypatch):
    """No test reads the machine's automod lock to decide anything.

    `autocode._loop_is_free` asks whether a landing holds that lock before it
    will call anything free, so read unpatched, every `(True, "free")` the
    suite asserts would depend on whether a real landing happened to be in
    flight while the test ran — and on this box one usually is. Same shape of
    hazard as the fixture above (a live process's state reaching into a test's
    verdict), different mechanism and different file.

    It redirects rather than stubs: the lock file still exists and `S.Lock`
    still takes a real `flock` on it, so lock behaviour is tested on the real
    class, only not on production's file. A test that wants to *be* the holder
    constructs `S.Lock(its_own_path)`."""
    from scripts.automod import state as S
    monkeypatch.setattr(S, "LOCK_PATH", tmp_path / "lloyd-automod" / "lock")


@pytest.fixture(autouse=True)
def _writes_enabled_in_tests(monkeypatch):
    """Fact writes are on unless a test says otherwise."""
    try:
        from agent_mcp import facts as facts_mod
    except Exception:          # module not importable in this test's env
        return
    monkeypatch.setattr(facts_mod, "_writes_enabled", lambda: True, raising=False)


@pytest.fixture(autouse=True)
def _isolate_default_store(request, tmp_path_factory):
    """No test writes to the live knowledge-graph store.

    Tests that need a store call `kg_store.configure(...)` themselves; this
    only guarantees the *default* never resolves to the production file, and
    puts it back afterwards.
    """
    from app import kg_store
    original = kg_store._default_path
    kg_store.reset()
    kg_store._default_path = tmp_path_factory.mktemp("kg") / "kg.sqlite"
    yield
    kg_store.reset()
    kg_store._default_path = original


@pytest.fixture(autouse=True)
def _isolate_research_store(tmp_path_factory):
    """No test writes to the live research registry.

    Same shape and same reason as `_isolate_default_store` above: the default
    must never resolve to `~/lloyd/research.db`, because a test that proposes
    a topic would otherwise put it in front of the deep-research worker. Tests
    that want a registry call `research_store.configure(...)` themselves.
    """
    from app import research_store
    original = research_store._default_path
    research_store.reset()
    research_store._default_path = tmp_path_factory.mktemp("research") / "research.db"
    yield
    research_store.reset()
    research_store._default_path = original


@pytest.fixture(autouse=True)
def _isolate_background_records(tmp_path_factory, monkeypatch):
    """No test writes a session or an event log into the live tree.

    Recording became universal on 2026-09-10: every autonomy run and every
    `run_prompt_on_primary` worker turn now writes a session JSON and an event
    log. A test that drives one of those paths — and several do, for reasons
    that have nothing to do with recording — would otherwise leave a real
    transcript in `~/lloyd/sessions/`, where it shows up in the history list,
    in session recall and in the retention sweep.

    Same shape as the two isolators above: the *default* never resolves to the
    production directory. A test that patches these itself still wins, because
    its `monkeypatch` runs after this fixture's.
    """
    root = tmp_path_factory.mktemp("lloyd-records")
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", root / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", root / "event_logs")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", root / "event_logs" / "blobs")


@pytest.fixture(autouse=True)
def _isolate_usage_store(tmp_path_factory, monkeypatch):
    """No test writes a usage row into the live `usage.db`.

    `usage_store.DB_PATH` resolves from `__file__`, so in `~/lloyd` it is the
    production database behind the dashboard's token panel and its
    prefix-miss counter. Until 2026-09-10 only the chat path wrote rows; the
    background recorder writes one per run now, and several tests drive it.
    `usage_store._conn` reopens when `DB_PATH` moves, so the per-thread
    connection cache cannot carry a test's writes into the next one's file.
    """
    monkeypatch.setattr("app.usage_store.DB_PATH",
                        tmp_path_factory.mktemp("usage") / "usage.db")


@pytest.fixture(autouse=True)
def _isolate_backlog_dedupe(tmp_path_factory, monkeypatch):
    """No test appends to the live dedupe log or queries the live qmd daemon.

    `backlog_write_task` runs the write-time dedupe on every create, and its
    log path is a `Path.home()` literal. Tests that patched `BACKLOG_DIR` and
    nothing else — `test_backlog_okf_frontmatter.py`, `test_backlog_tags_shape.py`
    — wrote a real row per run and POSTed their fixture text to the daemon. On
    2026-09-13, 656 of `dedupe.jsonl`'s 1004 rows were fixtures ("A newly
    written task" 498, "Filed by a digest run" 158): the log that exists to
    tune the merge threshold was two-thirds noise.

    `dedupe_config` is pinned to the defaults so `config.yaml` cannot flip
    `merge` under a test. Test-local patches still win — their `monkeypatch`
    runs after this one — which is how `test_backlog_dedupe.py` feeds rows in.
    """
    try:
        from agent_mcp import backlog_similar as SIM
    except Exception:          # module not importable in this test's env
        return
    monkeypatch.setattr(SIM, "DEDUPE_LOG",
                        tmp_path_factory.mktemp("dedupe") / "dedupe.jsonl")
    monkeypatch.setattr(SIM, "semantic_candidates", lambda text, **kw: [])
    monkeypatch.setattr(SIM, "dedupe_config", lambda: dict(SIM.DEFAULTS))


@pytest.fixture(autouse=True)
def _recall_ranked_by_qmd_in_tests(monkeypatch):
    """No test's recall reaches the LIVE djev on GPU 2 unless it asks to (#1336).

    `agent_mcp.vault.recall_reranker()` is "djev" whenever `djev.enabled` is on in
    the live config, and a test that stubs qmd's reply with two rows or more would
    then send a real ranking to :8011 — production load and a result that depends
    on a GPU. Pinned to the cross-encoder path the existing recall tests were
    written against; `tests/test_recall_djev_ranker.py` opts back in, with djev
    stubbed.
    """
    import agent_mcp.vault as _vault
    monkeypatch.setattr(_vault, "RECALL_RERANKER", "qmd")
    # Same reason, other engine: config.yaml turns the topics merge on (#1456),
    # which drafts on the LIVE primary. `tests/test_recall_topics_merge.py` sets
    # the mode itself through config, which this block only defaults.
    monkeypatch.setitem(__import__("app.config", fromlist=["CONFIG"]).CONFIG,
                        "vault_recall", {"topics_merge": "off"})
    yield


@pytest.fixture(autouse=True)
def _entity_seeding_off_in_tests(monkeypatch):
    """Recall tests seed lexically unless they ask otherwise (#1486).

    `retrieval.entity_seeding` is on in the live config, and its semantic half
    would load a 0.6B embedder and read the production entity-vector index from
    any test that reaches `_vault_recall`. The env switch (not a monkeypatch) so
    the live-corpus subprocesses inherit it too; the tests of the feature patch
    `entity_linker.seeding_config` or clear the variable for their subprocess.
    """
    monkeypatch.setenv("LLOYD_ENTITY_SEEDING", "0")
    # #1485: the episodic floor too, so recall tests keep the request they pin
    # whatever `retrieval.recall.episodic_floors` says; its own tests override.
    monkeypatch.setenv("LLOYD_RECALL_EPISODIC", "0")


@pytest.fixture(autouse=True)
def _fact_write_gate_off_in_tests(monkeypatch):
    """Fact writes in tests never ask the live djev (#1487).

    `knowledge_graph.write_gate.mode` is `noop` in the live config, and every
    `_fact_add` would otherwise put a djev read in front of a fixture's write.
    The gate's own tests arm it with `monkeypatch.setenv(gate.MODE_ENV, …)` and a
    fake `djev.ask_sync`, which overrides this.
    """
    monkeypatch.setenv("LLOYD_FACT_WRITE_GATE", "off")


@pytest.fixture(scope="session", autouse=True)
def _isolate_djev_shadow(tmp_path_factory):
    """No test appends a row to the log the djev floors are calibrated from.

    `~/.local/state/lloyd-djev/shadow.jsonl` is what `eval/djev/replay.py
    --floors` reads, and step 5 of `architecture/djev.md` §9.3 is "let the
    shadow rows accumulate, then set the floor from them" — a corpus that is
    mostly fixtures yields a floor set from fixtures. Its three paths are
    `Path.home()` literals bound at import (`app/djev_shadow.py:68-71`), and
    three seams (`agent_mcp/backlog.py:529`, `agent_mcp/vault.py:1289`,
    `scripts/memory/entity_semantic_gate.py:225`) call the recorder from code
    that has no idea a test is running. Re-measured 2026-09-21 08:19Z over the
    live log: 36 of its 44 `dedupe` rows carried one of two fixture titles and
    27 of its 32 `rerank` rows carried an empty `meta`, in a file six hours
    old — nothing decays, `_write()` only appends.

    SESSION scope, not per test, because the recorder's worker is a daemon
    thread that drains whenever it likes and `_write()` re-reads `SHADOW_LOG`
    at write time (`:295`). `tests/test_djev_shadow.py` has patched these names
    per test since the recorder landed and still leaked: after its `monkeypatch`
    tears down, the module points at the home path again and the row arrives
    there. Its own per-test patch still wins while the test runs — a narrower
    patch applied later simply covers it — and restores TO this one afterwards.

    **And deliberately no teardown, for the same reason.** The first cut of
    this fixture restored the three names on session teardown and still left a
    row in `$HOME/.local/state/lloyd-djev/shadow.jsonl` after
    `HOME=$FH pytest tests/test_backlog_dedupe.py tests/test_backlog_spawn_loop.py`:
    neither module calls `flush()`, so a job whose djev read outlasts pytest's
    teardown lands wherever the module points then, and restoring put the home
    path back exactly when a drain was in flight. Nothing reads these globals
    after the session ends — the process exits — so the restore bought nothing
    and cost the isolation. Measured after dropping it: the file is not created
    at all.

    Paths, not `LLOYD_DJEV_SHADOW=0`, because that variable is itself asserted
    on: `test_djev_rerank_arm.py::test_the_eval_mutes_the_shadow_recorder`
    reads it out of the process to prove `eval/run_eval.py:35` set it, and an
    ambient mute makes that assertion hold with the line under test deleted. A
    redirected path silences the same writes without touching the switch.

    All three globals move together, because they are three independent leaks:
    `_write()` opens `SHADOW_LOG`, `_record_pending_drops()` writes
    `PENDING_DROPS`, and both `mkdir` `STATE_DIR`.

    Pinned by `tests/test_djev_shadow_isolation.py`, whose child-pytest node is
    #1324's reproduction (`HOME=$FH pytest tests/test_backlog_dedupe.py
    tests/test_backlog_spawn_loop.py` must leave no shadow log in `$FH`) run as
    an assertion, so the automod gate's own full-suite run — which inherits
    `HOME` from `_child_env` — is covered by the same property.
    """
    try:
        from app import djev_shadow
    except Exception:          # module not importable in this test's env
        return
    root = tmp_path_factory.mktemp("djev-shadow")
    djev_shadow.STATE_DIR = root
    djev_shadow.SHADOW_LOG = root / "shadow.jsonl"
    djev_shadow.PENDING_DROPS = root / "dropped_at_shutdown.json"


@pytest.fixture
def worker_turn_post(monkeypatch):
    """Capture the body of a worker's loopback POST, with no backend to POST to.

    `workers.sources._common.run_prompt_in_session` is the one entry every
    session-backed source uses (`deep-research`, `youtube-digest`, `autotriage`,
    `autocode`), and what it puts in `payload["text"]` is the whole of the
    instruction the turn gets. A test cannot assert that from a stubbed
    `run_prompt_in_session` — the source's own test replaces that function, so
    anything it does downstream is invisible there. This swaps only the
    transport: the function runs unpatched through its payload build, and the
    list returned is the JSON body in request order.

    Not autouse: it replaces `httpx.AsyncClient` globally for the test, which
    is exactly what a test that is verifying a real POST wants and what every
    other test would choke on. `tests/test_session_platform_checks.py` owns the
    variant that crosses the wire for real.

    `.report(3, 5)` says what each turn's `done` event reports, consumed one
    element per POST in turn order; an element of `None` means a `done` with no
    `num_turns` key at all. Calling it is optional and every test that skips it
    gets what it always got — one turn reporting one step — because the return
    value is still the same list of bodies, read by index and length exactly as
    before. It lives on the fixture rather than in a second stub because the
    number a turn reports is the fixture's business, and `#2087` is the reason:
    the copies of this stub that sources wrote for themselves could not express
    "the turn reported 7 steps", which is why nothing pinned the step count on a
    run row until that item.
    """
    import httpx

    class _Posts(list):
        """The captured bodies, plus the knob for what each turn reports."""

        def __init__(self) -> None:
            super().__init__()
            self._report: list = []
            self._configured = False

        def report(self, *num_turns) -> "_Posts":
            self._report = list(num_turns)
            self._configured = True
            return self

        def _next(self):
            """What the next turn's `done` reports, and loudly if it shouldn't.

            A test that scripts `(3, 5)` and then gets a third turn has a wrong
            expectation, not a stubbed environment: silently answering `1` would
            shift the sum the test is asserting and read as a product bug. So the
            exhaustion is an error, and only once `report()` has been called — a
            test that never configured anything keeps the historical answer.
            """
            if self._report:
                return self._report.pop(0)
            if self._configured:
                raise AssertionError(
                    "worker_turn_post.report() was given "
                    f"{self._report!r} and has run out, so the job ran more turns "
                    "than the test scripted — fix the test, do not assume 1")
            return 1

    captured = _Posts()

    class _Streamed:
        """A finished turn, in the SSE shape `_aiter_sse` parses."""

        status_code = 200

        def __init__(self, num_turns):
            self._num_turns = num_turns

        async def aread(self) -> bytes:
            return b""

        async def aiter_lines(self):
            payload = '{"response": "stubbed", "stop_reason": "end_turn"'
            if self._num_turns is not None:
                payload += f', "num_turns": {self._num_turns}'
            yield "event: done"
            yield "data: " + payload + "}"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            captured.append(dict(json or {}))
            return _Streamed(captured._next())

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return captured
