"""Shared filesystem paths for the Lloyd backend."""

import logging
import os
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parent.parent
SESSIONS_DIR = LLOYD_HOME / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

VAULT_ROOT = Path.home() / "obsidian"

VAULT_DERIVED_ROOT = LLOYD_HOME / "_pipeline" / "vault-derived"

# Runtime state directories. These were previously spelled `Path.home() /
# "lloyd" / ...` at a dozen call sites, which pinned them to the *user's*
# lloyd checkout rather than to the code that is running. That is a latent bug
# on its own (a second checkout silently shares the first one's state) and it
# is fatal for a self-modification canary: the canary boots from a worktree but
# would still have claimed jobs from the live workers.db, written into the live
# sessions dir, and rewritten live autonomy task files. Anchoring to LLOYD_HOME
# means state follows the code, which is what every other path here already did.
AUTONOMY_RUNS_DIR = LLOYD_HOME / "autonomy-runs"
TASKS_DIR = LLOYD_HOME / "_pipeline" / "tasks"
LOGS_DIR = LLOYD_HOME / "logs"
SCREENSHOTS_DIR = LOGS_DIR / "screenshots"

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
# Anchored to LLOYD_HOME for the reason in the block above — a canary booting
# from a worktree must get its own empty registry, not claim the live one's
# topics. LLOYD_RESEARCH_DB overrides it for tests and rebuilds.
RESEARCH_DB = Path(os.environ["LLOYD_RESEARCH_DB"]) if os.environ.get("LLOYD_RESEARCH_DB") \
    else LLOYD_HOME / "research.db"

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
IS_WORKTREE = (LLOYD_HOME / ".git").is_file()


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
    return f"LLOYD_HOME={LLOYD_HOME} tree={'worktree' if IS_WORKTREE else 'live'}"


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
