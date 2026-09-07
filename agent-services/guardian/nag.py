"""Re-announce an unresolved BROKEN state, through the shared fan-out.

This used to be an inline `notify-send` in `lloyd-guardian-nag.service`. That
worked, but it was a second, private definition of "tell the human" — so it
could never gain a channel `notify.py` grew, and the day speech was added it
would have been the one alert that stayed silent. Routing it through
`Notifier` is the whole point of the consolidation: one place produces alerts.

Two things it deliberately does *not* inherit from the daemon:

* **No repo, no supervisord, no state machine.** The nag must work when the
  guardian itself is dead — that is arguably when it matters most — so it
  reads one file and fans out. Nothing here can raise.
* **Its own suppression key.** `Notifier` dedupes speech on the alert title,
  and every nag carries the same title by design, so the 15-minute timer
  produces at most one utterance per `VOICE_REPEAT_SECONDS` for free.

The unit keeps a bash `notify-send` fallback for the case where this file is
missing from the snapshot entirely — a stale guardian must still nag.
"""

from __future__ import annotations

import sys
from pathlib import Path

TITLE = "STILL BROKEN"


def read_broken(path: Path) -> str | None:
    """The BROKEN marker's contents, or None when there is nothing to nag about."""
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")[:400].strip()
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gstate
    import policy
    import notify as notify_mod

    # gstate owns both names: the BROKEN marker file (beside the lowercase
    # `broken/` incident directory, which is a different thing) and the ledger
    # filename. Hardcoding either here is how the two drift.
    state = gstate.SelfModState(Path(policy.SELFMOD_STATE))
    body = read_broken(state.broken)
    if body is None:
        return 0                      # nothing broken — the common case

    notifier = notify_mod.Notifier(
        ledger=state.ledger,
        state_dir=Path(policy.GUARDIAN_STATE),
        vault_root=policy.VAULT_ROOT,
        voice_window=policy.VOICE_REPEAT_SECONDS,
    )
    # `announce`, not `alert`, and this is the whole reason announce takes a
    # level. The state is genuinely critical and should look it — but it has
    # already been recorded, by whichever rollback wrote the marker. Sending a
    # re-announcement through `alert` would append a ledger row and file a
    # fresh backlog task every 15 minutes for as long as the incident lasted,
    # burying the task the rollback filed under copies of itself.
    results = notifier.announce(TITLE, body, level="critical")
    print(" ".join(f"{k}={'ok' if v else 'no'}" for k, v in sorted(results.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
