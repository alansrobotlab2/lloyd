"""Shared filesystem paths for the Lloyd backend."""

import logging
import os
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parent.parent
IS_WORKTREE = (LLOYD_HOME / ".git").is_file()

# The account's home as the passwd entry names it, which `$HOME` no longer does
# inside an automod gate: since 2026-09-22 the rungs that run candidate code set
# `HOME=<round>/home`, a symlink farm whose `lloyd` IS the worktree
# (`scripts/automod/worktree.py::ensure_round_home`). So `Path.home() / "lloyd"`
# names the worktree there, and a reader that falls back to "the live checkout"
# through it falls back to the tree it just found empty. So the rule is that a
# job which wants live data — to read a session, a log, a baseline — reaches it
# through here and not through `$HOME`. Writers use `DATA_ROOT`, which is why the
# sweep's root, not this, is what `--apply` deletes (#1415). "Nothing that writes
# should" was never true of this seam: four jobs write through it deliberately —
# the review grader's session file, the nightly extraction's lock/log/backups, the
# content hasher's index, the tool-override sync — and `architecture/data-home.md`
# names them. What is still missing, and #1415 does not close, is anything that
# ENUMERATES them: no constant or test separates a deliberate live writer from the
# next person who reached for the nearest live-path function.
#
# The resolution lives in `app.data_root`, the stdlib-only half of this module,
# because the jobs that cannot import THIS one — no venv, cron, a unit's `sh -c`
# child — were each re-implementing it and getting rule 2 below with neither the
# marker check nor rule 3. One of them is the script that deletes (#1415). The
# names are re-exported here unchanged, so every existing caller still resolves
# `app.paths.DATA_ROOT_MARKER`, `data_root_for_tree`, `production_data_root`,
# `resolve_data_root` and `DataRootMissing` at the place it always did.
from app.data_root import (
    ACCOUNT_HOME, DATA_ROOT_MARKER, DataRootMissing, PRODUCTION_DATA_ROOT,
    data_root_for_tree, production_data_root, resolve_data_root)

#: The names re-exported through `app.paths`, listed so the import above reads as
#: the API it is rather than as four unused imports a later tidy-up can delete —
#: the stdlib-only jobs import `app.data_root` directly, and a reader who removes
#: one of these lines breaks `app.paths`, not `app.data_root`. This is the
#: re-export list, not this module's whole surface: the constants below are its
#: other half.
__all__ = ["ACCOUNT_HOME", "DATA_ROOT_MARKER", "DataRootMissing",
           "PRODUCTION_DATA_ROOT", "data_root_for_tree", "production_data_root",
           "resolve_data_root"]

LIVE_CHECKOUT = ACCOUNT_HOME / "lloyd"

# ------------------------------------------------------------------ data root --
#
# Runtime data (transcripts, databases, logs, derived state) lives OUTSIDE the
# code tree. On 2026-09-22 a pytest fixture teardown deleted `~/lloyd` in 35 s
# and every gitignored byte in it went too: sessions, `_pipeline/`, the
# databases, the baselines. A tree that holds the data puts the data in reach of
# every `rm -r`, `git clean -x` and fixture aimed at the code.
# `architecture/data-home.md` is the long version.
#
# The layout under the root keeps the tree's old relative names (`sessions/`,
# `_pipeline/`, `workers.db`, …), so `~/lloyd/X` became `~/lloyd-data/X`.
#
# Resolution, first match wins:
#   1. `LLOYD_DATA` in the environment. The automod gate, the canary and the test
#      suite set it; nothing in production needs to, and nothing exports it —
#      a Bash child running a worktree's code must not inherit the live root.
#   2. The production checkout (`LLOYD_HOME == LIVE_CHECKOUT`, not a worktree)
#      uses `<passwd home>/lloyd-data`, read off passwd and never `Path.home()`:
#      under a gate's HOME that name is the round's.
#      The root must carry `DATA_ROOT_MARKER`. Before the 2026-09-22 cutover it
#      did not exist and production kept its data in the tree; that fallback is
#      gone, and a production checkout with no marked root refuses to start
#      rather than quietly writing a second copy of everything into the tree.
#   3. Any other checkout — the sandbox, a round's worktree, a scratch clone —
#      keeps its data inside itself, under `.lloyd-data/` (gitignored). That is
#      the isolation the canary was built on ("state follows the code"), kept as
#      the default so the unsafe direction is never what happens by omission.
# `DATA_ROOT_MARKER`, `PRODUCTION_DATA_ROOT`, `data_root_for_tree`,
# `production_data_root`, `DataRootMissing` and `resolve_data_root` are this
# module's names too — imported from `app.data_root` above, where the rules are
# written once so the stdlib-only jobs read them as well (#1415).
DATA_ROOT = resolve_data_root(env=os.environ.get("LLOYD_DATA"), lloyd_home=LLOYD_HOME,
                              is_worktree=IS_WORKTREE, live_checkout=LIVE_CHECKOUT,
                              production_root=PRODUCTION_DATA_ROOT)
IS_PRODUCTION_DATA = DATA_ROOT == PRODUCTION_DATA_ROOT

SESSIONS_DIR = DATA_ROOT / "sessions"
# Importing this module creates NOTHING under `DATA_ROOT` (#712). Until now
# `SESSIONS_DIR.mkdir(...)` ran here, so merely *collecting* the test suite laid
# directories into the tree it was collected from — a gate's full-suite rung
# imports the round's own checkout, so a container appeared in a tree no code
# had run in, and a reader that asked whether the container existed took that for
# a populated store. Creation moved to `ensure_dirs()` below, which every process
# that writes calls at boot.
EVENT_LOGS_DIR = DATA_ROOT / "event_logs"
USAGE_DB = DATA_ROOT / "usage.db"
WORKERS_DB = DATA_ROOT / "workers.db"
MC_STATE_PATH = DATA_ROOT / "mc-state.json"
TOOL_OVERRIDES_PATH = DATA_ROOT / "data" / "tool_overrides.yaml"
PIPELINE_DIR = DATA_ROOT / "_pipeline"
EVAL_BASELINES_DIR = DATA_ROOT / "eval" / "baselines"
VOICE_PROFILES_DIR = DATA_ROOT / "voice_profiles"
# Supervisor program logs for the engines and services (was agent-services/logs).
SERVICE_LOGS_DIR = DATA_ROOT / "logs" / "services"

VAULT_ROOT = Path.home() / "obsidian"

VAULT_DERIVED_ROOT = PIPELINE_DIR / "vault-derived"

# Runtime state directories. These were once spelled `Path.home() / "lloyd" /
# ...` at a dozen call sites, which pinned them to the *user's* lloyd checkout
# rather than to the code that is running — fatal for a self-modification
# canary, which would have claimed jobs from the live workers.db, written into
# the live sessions dir, and rewritten live autonomy task files. They hang off
# DATA_ROOT, whose rule 3 keeps a worktree's state inside the worktree.
AUTONOMY_RUNS_DIR = DATA_ROOT / "autonomy-runs"
TASKS_DIR = PIPELINE_DIR / "tasks"
LOGS_DIR = DATA_ROOT / "logs"
SCREENSHOTS_DIR = LOGS_DIR / "screenshots"
# Desktop computer use (agent_mcp/desktop): the lease file both the backend
# (which grants it from Mission Control) and the aggregator (which checks it
# before every action) read, and the last capture's frame for the Desktop tab.
DESKTOP_DIR = DATA_ROOT / "desktop"
DESKTOP_LEASE_PATH = DESKTOP_DIR / "lease.json"
# The mitigation drill's latest measured result per stop control (#703):
# `scripts/mitigation_drill.py` writes it, `GET /api/workers/status` reports it.
MITIGATION_DRILL_STATE = DATA_ROOT / "mitigation_drill.json"
# The autocode ReasoningBank (#1489): strategy items distilled from the review
# rung's refusals. A cache the automod ledger always rebuilds whole
# (`scripts/automod/reasoning_bank.py`); read only while
# `workers.sources.autocode.reasoning_bank` is on.
REASONING_BANK_PATH = DATA_ROOT / "automod" / "reasoning_bank.jsonl"

#: Everything `ensure_dirs()` creates, as data rather than as a literal inside the
#: function. `LOGS_DIR` is an ancestor of `SCREENSHOTS_DIR`, so a filesystem probe
#: cannot tell "created `logs`" from "`mkdir -p` made it on the way to
#: `logs/screenshots`" — this tuple is the claim that IS checkable per name, and
#: `tests/test_paths_ensure_dirs.py` compares it against the five constants the
#: writers actually use.
RUNTIME_STATE_DIRS: tuple[Path, ...] = (SESSIONS_DIR, AUTONOMY_RUNS_DIR, TASKS_DIR,
                                        LOGS_DIR, SCREENSHOTS_DIR)


def ensure_dirs() -> None:
    """Create the runtime-state directories this module's writers need.

    Every process that writes calls this: `server.py` registers it as a startup
    hook ahead of the autonomy ticker and the worker pool, and
    `agent_mcp/main.py` calls it in its own lifespan. It used to be an
    import-time `SESSIONS_DIR.mkdir(...)`, and that one line is why merely
    *collecting* the test suite wrote directories into a tree it had no business
    creating anything in — see the note above `SESSIONS_DIR`. The `LLOYD_HOME`
    anchoring above is untouched on purpose: a canary still resolves every name
    here inside its own tree; only the side effect moved.

    Idempotent, so it is safe on every boot and from any test. Several of these
    writers cover themselves anyway (`_task_registry` makes `TASKS_DIR` when a
    task registers, `autonomy._write_run_record` makes each per-task runs dir,
    `sessions_io.create_session` makes `SESSIONS_DIR`); they are listed here too
    so the directories exist at a known moment rather than at whichever writer
    happened to run first.

    What it creates is `RUNTIME_STATE_DIRS` above, not a literal here, so "these
    are the five" is checkable against the constants the writers use rather than
    against a copy of them inside this function.
    """
    for directory in RUNTIME_STATE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

# The fact tree (one dir per entity, markdown fact files). LLOYD_FACTS_ROOT
# lets a rebuild extract into a fresh tree without touching the live one.
#
# The `*_DEFAULT` pair below is the same location spelled WITHOUT the override,
# and exists so a run can name what it actually acted on (#700): five
# improve-pass `--apply` records from 2026-09-09 reported 32 fact expirations
# that never reached the live store — all 32 facts are still active — and the
# JSON could not say which tree or which store it described, because
# `RECORD_DIR` is code-relative while these two paths are env-overridable. A
# guard that compared a run against `VAULT_FACTS_ROOT` would be comparing it
# against the override, which is how a copy certifies itself as production.
VAULT_FACTS_ROOT_DEFAULT = VAULT_DERIVED_ROOT / "facts"
VAULT_FACTS_ROOT = Path(os.environ["LLOYD_FACTS_ROOT"]) if os.environ.get("LLOYD_FACTS_ROOT") \
    else VAULT_FACTS_ROOT_DEFAULT

# The knowledge-graph store: edges, aliases, entity registry and the fact
# index live in one SQLite file (app.kg_store). Nothing opens it except that
# module. LLOYD_KG_DB overrides the location for rebuilds and tests;
# `VAULT_KG_DB_DEFAULT` is the built-in location, env-immune for the reason
# above.
VAULT_KG_DB_DEFAULT = VAULT_DERIVED_ROOT / "kg.sqlite"
VAULT_KG_DB = Path(os.environ["LLOYD_KG_DB"]) if os.environ.get("LLOYD_KG_DB") \
    else VAULT_KG_DB_DEFAULT

# The fact write gate's decision log (#1487, agent_mcp/fact_write_gate.py):
# beside the store it judged, so a replay against a copy (LLOYD_KG_DB) logs
# into the copy's directory and never into the live one's.
FACT_WRITE_GATE_LOG = VAULT_KG_DB.parent / "fact-write-gate.jsonl"

# The research topic registry: what to research, what came of it, and the
# feedback that keeps a generator from re-proposing it (app.research_store).
# Under DATA_ROOT for the reason in the block above — a canary booting from a
# worktree must get its own empty registry, not claim the live one's topics.
# LLOYD_RESEARCH_DB overrides it for tests and rebuilds.
RESEARCH_DB = Path(os.environ["LLOYD_RESEARCH_DB"]) if os.environ.get("LLOYD_RESEARCH_DB") \
    else DATA_ROOT / "research.db"

# The legacy alias map, and a warning about its shape. The live alias table is
# `aliases` in app.kg_store (SQLite); this path is NOT an export target any
# more (#474). Store.export_json() is called only by the one-shot migration and
# by the pre-rebuild freeze, and both write timestamped dirs under
# _pipeline/backups/ — because a snapshot that sits inside the tree it claims to
# describe gets read as that tree. The copy that used to live here was written
# once on 2026-09-03 and never refreshed, and every alias "defect" since was
# measured off it at ~5x the live table. The constant stays so
# kg_migrate_to_sqlite's `--aliases` default can still name a pre-migration file;
# nothing in the running system writes or reads this path.
VAULT_FACTS_ALIASES = VAULT_FACTS_ROOT / "entity-aliases.json"
VAULT_SESSIONS_DIR = VAULT_DERIVED_ROOT / "sessions"
# Background sessions export here instead, and the split is about the qmd
# watcher: `agent-services/scripts/qmd-watcher.sh` indexes and *embeds*
# `sessions/` on every change. The ~70 session-backed worker transcripts a
# day that reach post-capture (the direct-path runs never do) would each be an
# embedding job over the machine talking to itself — against ~14 chats a day,
# drowning the corpus that answers questions about what the user and Lloyd
# discussed. Outside the watch: still exported, still greppable, not embedded.
VAULT_BACKGROUND_SESSIONS_DIR = VAULT_DERIVED_ROOT / "sessions-background"
VAULT_PENDING_RESEARCH_DIR = VAULT_DERIVED_ROOT / "pending-research"
VAULT_FEEDS_DIR = VAULT_DERIVED_ROOT / "memory" / "feeds"

# ---------------------------------------------------------------- which tree? --
#
# Everything above resolves inside the tree this module was imported from, and
# that is deliberate: a self-modification canary boots from a worktree and MUST
# get its own empty `sessions/` and `workers.db` rather than claim the live
# one's (the block above `AUTONOMY_RUNS_DIR`), and the canary imports its own
# tree on purpose — the `<round>/home/lloyd` layout exists precisely so that
# `LLOYD_HOME` and `HOME=<round>/home` are the same directory
# (`scripts/automod/worktree.py:1-10`). What was missing is a NAME for the fact,
# which is what #733 is about: a script a round runs scans `SESSIONS_DIR`, finds
# the worktree's empty one, and reports a clean, plausible, false result. The
# #529 replay printed `no session file matching '20260909_155011_backlogi_32bb'
# in …/SM_20260910_004029/home/lloyd/sessions` for a 500,934-byte file that is in
# the live checkout, and an hour later silently reported
# `cumulative_prompt_tokens: null` for a whole arm. `app/uptake.py:205-238` grew
# a private copy of this same detection after a landing was refused on a corpus
# of one canary turn (`bfa8bd1`), and `app/uptake.py:1413-1418` names a second
# instance where the worktree's absent `kg.sqlite` produced a table reporting
# `duplicate_rows: 0`. The name belongs here, one level up.
#
# The discriminator is on disk and is not an inference: `git worktree add` leaves
# `.git` as a one-line FILE (`gitdir: /…/.git/worktrees/<name>`) where the main
# checkout has a DIRECTORY. One `is_file()` answers it — no `$HOME` preference,
# no live-root computation, nothing to guess wrong. It deliberately does NOT
# retarget anything: the anchor stays exactly where it was, this only says where
# it is.
# (IS_WORKTREE itself is defined at the top of this module, beside LLOYD_HOME:
# the data-root rule reads it.)


def describe_tree() -> str:
    """One line naming the tree these paths resolve in: `… tree=live` or `… tree=worktree`.

    Put it in any aggregate artifact — a scan of `SESSIONS_DIR`, a row count, an
    uptake table — so a measurement carries the corpus it was measured on and a
    reader can tell "zero rows" from "zero rows because this is a worktree".
    `tree` is `worktree` exactly when this checkout is a linked git worktree, and
    `live` for every other tree, which in this repo means the main checkout: a
    tree with no `.git` entry at all is not a worktree either, and this module
    does not infer anything beyond that.
    """
    return (f"LLOYD_HOME={LLOYD_HOME} tree={'worktree' if IS_WORKTREE else 'live'}"
            f" DATA_ROOT={DATA_ROOT}")


if IS_WORKTREE:
    # Once per process, because the module body runs once — and this is the ONLY
    # thing the detection changes: every constant above is untouched. A WARNING,
    # not a raise: the canary boot legitimately imports its own tree and gates on
    # the subprocess returncode and health polling
    # (`scripts/automod/canary.py:104-113`), so refusing at import would break the
    # boot the layout was designed for. Scripts that aggregate over the dirs
    # below are what #733 asks to hear this, and they get it in their own log.
    logging.getLogger(__name__).warning(
        "app.paths is anchored to a git WORKTREE, not to the main checkout: %s."
        " SESSIONS_DIR=%s VAULT_DERIVED_ROOT=%s AUTONOMY_RUNS_DIR=%s all"
        " resolve inside it and are usually empty or partial, so scanning them is"
        " a measurement of nothing, not a clean result. Name the tree in whatever"
        " this prints: app.paths.describe_tree().",
        LLOYD_HOME, SESSIONS_DIR, VAULT_DERIVED_ROOT, AUTONOMY_RUNS_DIR,
    )
