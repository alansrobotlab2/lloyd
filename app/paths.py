"""Shared filesystem paths for the Lloyd backend."""

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
VAULT_FACTS_ROOT = Path(os.environ["LLOYD_FACTS_ROOT"]) if os.environ.get("LLOYD_FACTS_ROOT") \
    else VAULT_DERIVED_ROOT / "facts"

# The knowledge-graph store: edges, aliases, entity registry and the fact
# index live in one SQLite file (app.kg_store). Nothing opens it except that
# module. LLOYD_KG_DB overrides the location for rebuilds and tests.
VAULT_KG_DB = Path(os.environ["LLOYD_KG_DB"]) if os.environ.get("LLOYD_KG_DB") \
    else VAULT_DERIVED_ROOT / "kg.sqlite"

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
