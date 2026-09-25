"""One definition of the tools an unattended turn may not call.

The two lists lived in `workers/sources/_common.py`, which is where the first
reader was. They have more readers now than that module has callers: the chat
router arms the ban for every non-user session, the review grader's deny list
must be a superset, and a `Task` child spawned by a worker turn has to inherit
them (review 2026-09-24, D4) — from the aggregator, which must not import the
worker package to learn a list of names. `_common` re-exports both, so every
existing `from workers.sources._common import WORKER_AUTOMOD_BAN` keeps working.

It lives in `app/` and imports only the standard library for the same reason
`app/backlog_tags.py` does: anything may read it, including the light modules
and the aggregator, without pulling a package in behind a tuple of strings.
"""

from __future__ import annotations

#: The self-modification loop is not a worker's to drive. Two turn paths need
#: it: `run_prompt_on_primary` bakes it in, and a session-backed source passes
#: it as `extra_disallowed`.
#: `tests/test_automod_hardening.py::test_worker_turns_cannot_drive_the_loop`
#: greps this file for these names, so this is where they live.
WORKER_AUTOMOD_BAN: tuple[str, ...] = (
    "automod_start", "automod_gate", "automod_gate_wait", "automod_land",
    "automod_abort", "automod_amend_clause", "automod_rollback",
    "automod_vault_land", "automod_vault_revert",
)

#: Minting an authority grant is not a worker's to do either (#534). Same
#: reasoning as the automod ban, named separately because it is enforced twice
#: on purpose: the tool is not advertised on a worker turn, AND the policy hook
#: denies the call if a local model emits it anyway. A turn subject to an
#: authority gate must not be able to write its way out of it — that would be
#: `bypassPermissions` with extra paperwork.
WORKER_GRANT_MINT_BAN: tuple[str, ...] = ("grant_create",)
