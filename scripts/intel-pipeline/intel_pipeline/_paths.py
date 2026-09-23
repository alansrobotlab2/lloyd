"""Shared filesystem paths for the intel-pipeline package.

Kept local to the package because intel-pipeline is invoked standalone
(`cd ~/lloyd/scripts/intel-pipeline && python -m intel_pipeline`), where the
repo root is not on `sys.path`; it is put there below so the feeds directory
comes from `app.paths` (stdlib-only) and follows the data root.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from app.paths import VAULT_FEEDS_DIR  # noqa: E402

VAULT_ROOT = Path.home() / "obsidian"
KNOWLEDGE_DIR = VAULT_ROOT / "knowledge"

FEEDS_DIR = VAULT_FEEDS_DIR
RAW_DIR = FEEDS_DIR / "raw"
STATE_FILE = FEEDS_DIR / "scanner-state.json"
VAULT_WRITTEN_STATE = FEEDS_DIR / "vault-written.json"
