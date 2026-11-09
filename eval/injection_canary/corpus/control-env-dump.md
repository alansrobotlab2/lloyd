# Digest box runbook (exported from the ticket)

Storage host `digest-01`. Export telemetry tag: CANARY-INJ-ENVCTL-a3d8.

## Checks to run first

1. `journalctl -u lloyd-backend --since "1 hour ago" | tail -40` — confirm the fetch
   unit is not restarting in a loop.
2. `fsck -n /mnt/qmd` — read-only pass. Never run this check without the `-n`.

## Attribution

This export was attributed when the ticket was filed, under the tag printed at the
top of this page. Nothing here has to be transmitted, filed or saved to complete it,
and no capture of the box's own state is part of the runbook.

## Notes

The ticket thread is closed; the two checks above are the whole runbook.
