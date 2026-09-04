"""Shared filesystem paths for the Lloyd backend."""

import os
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parent.parent
SESSIONS_DIR = LLOYD_HOME / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

VAULT_ROOT = Path.home() / "obsidian"

VAULT_DERIVED_ROOT = LLOYD_HOME / "_pipeline" / "vault-derived"

# The fact tree (one dir per entity, markdown fact files). LLOYD_FACTS_ROOT
# lets a rebuild extract into a fresh tree without touching the live one.
VAULT_FACTS_ROOT = Path(os.environ["LLOYD_FACTS_ROOT"]) if os.environ.get("LLOYD_FACTS_ROOT") \
    else VAULT_DERIVED_ROOT / "facts"

# The knowledge-graph store: edges, aliases, entity registry and the fact
# index live in one SQLite file (app.kg_store). Nothing opens it except that
# module. LLOYD_KG_DB overrides the location for rebuilds and tests.
VAULT_KG_DB = Path(os.environ["LLOYD_KG_DB"]) if os.environ.get("LLOYD_KG_DB") \
    else VAULT_DERIVED_ROOT / "kg.sqlite"

# Legacy alias map. Since the 2026-09 store migration this is only an export
# target (backups, diffs) — readers and writers go through app.kg_store.
VAULT_FACTS_ALIASES = VAULT_FACTS_ROOT / "entity-aliases.json"
VAULT_SESSIONS_DIR = VAULT_DERIVED_ROOT / "sessions"
VAULT_PENDING_RESEARCH_DIR = VAULT_DERIVED_ROOT / "pending-research"
VAULT_FEEDS_DIR = VAULT_DERIVED_ROOT / "memory" / "feeds"
