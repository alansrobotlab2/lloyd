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
# Append-only record of the FIRST model grade every item id ever received
# (backlog #2139). It sits beside the `intel-<date>.jsonl` day files because it
# is the same kind of thing: stage-2 output, keyed by item, read by the next
# scoring pass rather than by the writer. The day files are rewritten wholesale
# on every `--score` (`__main__.py:150`), so they cannot be the memory; this file
# is never rewritten, only appended to.
GRADE_STORE = FEEDS_DIR / "grades.jsonl"
