"""Self-modification loop: worktree → gate → promote, with guardian rollback.

The prompt-surface analogue is `scripts/autoresearch/`. This package is its
code-surface sibling and reuses that package's round/spec helpers where they
are sound (`round_id`, `write_run_spec`, `validate_run_spec`) while
deliberately NOT reusing `ledger_append` — see `scripts/selfmod/state.py`.
"""
