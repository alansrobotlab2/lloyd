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
# through it falls back to the tree it just found empty. A READ of live data
# (sessions, logs, baselines) goes through this; nothing that writes should.
try:
    import pwd as _pwd
    ACCOUNT_HOME = Path(_pwd.getpwuid(os.getuid()).pw_dir)
except (ImportError, KeyError):
    ACCOUNT_HOME = Path.home()
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
DATA_ROOT_MARKER = ".lloyd-data-root"
PRODUCTION_DATA_ROOT = ACCOUNT_HOME / "lloyd-data"


def data_root_for_tree(tree: Path) -> Path:
    """Where a non-production checkout keeps its own runtime data."""
    return Path(tree) / ".lloyd-data"


def production_data_root() -> Path:
    """The live data root, for READERS that mean production on purpose.

    Writers use `DATA_ROOT`. This exists for the few readers whose whole job is
    live data wherever they run from — the review grader reading a round's
    sessions, the regression check reading the live store — and it is the same
    path whatever `$HOME` or `LLOYD_DATA` say.
    """
    return PRODUCTION_DATA_ROOT


class DataRootMissing(RuntimeError):
    """The production checkout found no marked data root."""


def resolve_data_root(*, env: str | None, lloyd_home: Path, is_worktree: bool,
                      live_checkout: Path, production_root: Path) -> Path:
    """The three rules above, as a pure function of what they read."""
    if env:
        return Path(env).expanduser()
    if lloyd_home == live_checkout.resolve() and not is_worktree:
        if not (production_root / DATA_ROOT_MARKER).is_file():
            raise DataRootMissing(
                f"{production_root} has no {DATA_ROOT_MARKER}: the production"
                " checkout keeps its runtime data there and refuses to fall back to"
                " the code tree. Restore it from ~/.lloyd-data-snapshots"
                " (scripts/backup/restore-data.sh), or set LLOYD_DATA explicitly."
            )
        return production_root
    return data_root_for_tree(lloyd_home)


DATA_ROOT = resolve_data_root(env=os.environ.get("LLOYD_DATA"), lloyd_home=LLOYD_HOME,
                              is_worktree=IS_WORKTREE, live_checkout=LIVE_CHECKOUT,
                              production_root=PRODUCTION_DATA_ROOT)
IS_PRODUCTION_DATA = DATA_ROOT == PRODUCTION_DATA_ROOT

SESSIONS_DIR = DATA_ROOT / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
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
