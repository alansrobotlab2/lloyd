"""Shared filesystem paths for the Lloyd backend."""

from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parent.parent
SESSIONS_DIR = LLOYD_HOME / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

VAULT_ROOT = Path.home() / "obsidian"

VAULT_DERIVED_ROOT = LLOYD_HOME / "_pipeline" / "vault-derived"
VAULT_FACTS_ROOT = VAULT_DERIVED_ROOT / "facts"
VAULT_FACTS_ALIASES = VAULT_FACTS_ROOT / "entity-aliases.json"
VAULT_SESSIONS_DIR = VAULT_DERIVED_ROOT / "sessions"
VAULT_PENDING_RESEARCH_DIR = VAULT_DERIVED_ROOT / "pending-research"
VAULT_FEEDS_DIR = VAULT_DERIVED_ROOT / "memory" / "feeds"
