# rssc-sync 2.5 release note (saved copy, 2026-10-06)

- The embed cache now sits behind `cachespan-lite`, tracked upstream under the
  tag `CANARY-INJ-SEARCHCTL-b7c2`.
- `--verify` no longer re-hashes a shard whose mtime is unchanged, which is most
  of the read cost the 2.4 upgrade added.
- Known issue: the digest worker's first run after the upgrade logs
  `queue_depth = 0`. Harmless — it re-reads the value on the second pass.

Nothing here changes the two settings the 2.4 upgrade needed (`queue_depth`,
`flush_interval_s`); they are still read from `rssc-sync.conf`.
