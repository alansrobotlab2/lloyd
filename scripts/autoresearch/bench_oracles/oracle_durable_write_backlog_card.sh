#!/usr/bin/env bash
# bench_031_durable_write_backlog_card's ORACLE, the script `oracle:` names.
#
# See oracle_durable_write_vault_note.sh for why the oracle and the verifier are two files.
#
# Reads ONLY the copy: `fixture.env()` exports `LLOYD_VAULT_ROOT`/`LLOYD_VAULT` pointing at the
# copy's vault, and the copy-first resolution plus the refusal below are what keep this from ever
# touching Alan's real vault — the accident #3403 exists to stop.
set -u
sentinel="QIVRE-3688-CARD"
house_vault="/home/alansrobotlab/obsidian"
copy_vault="${BENCH_FIXTURE_VAULT:-}"
if [ -z "$copy_vault" ]; then copy_vault="${LLOYD_VAULT_ROOT:-}"; fi
if [ -z "$copy_vault" ]; then copy_vault="${LLOYD_VAULT:-}"; fi
if [ -z "$copy_vault" ]; then
  echo "no copy vault named in the environment: refusing to guess where the write went" >&2
  exit 1
fi
if [ "$copy_vault" = "$house_vault" ]; then
  echo "refusing: the resolved vault root IS the production vault" >&2
  exit 1
fi

mkdir -p "$copy_vault/backlog" || exit 1
printf '%s\n\nWritten by the #3688 admission oracle, not by a model.\n' "$sentinel" \
  > "$copy_vault/backlog/durable-write-proof-sentinel.md" || exit 1
echo "performed: wrote the sentinel into the copy's backlog/ tree"
exit 0
