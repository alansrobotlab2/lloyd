"""Shared filesystem paths for the intel-pipeline package.

Kept local to the package because intel-pipeline is invoked standalone
(`cd ~/lloyd/scripts/intel-pipeline && python -m intel_pipeline`) and cannot
rely on `app.paths` being importable.
"""

from pathlib import Path

VAULT_ROOT = Path.home() / "obsidian"
KNOWLEDGE_DIR = VAULT_ROOT / "knowledge"

FEEDS_DIR = Path.home() / "lloyd" / "_pipeline" / "vault-derived" / "memory" / "feeds"
RAW_DIR = FEEDS_DIR / "raw"
STATE_FILE = FEEDS_DIR / "scanner-state.json"
VAULT_WRITTEN_STATE = FEEDS_DIR / "vault-written.json"
