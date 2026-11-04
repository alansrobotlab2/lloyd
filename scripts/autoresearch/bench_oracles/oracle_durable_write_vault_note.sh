#!/usr/bin/env bash
# bench_029_durable_write_vault_note's ORACLE, the script `oracle:` names.
#
# Admission arm 1 (`bench_side_effects.admit_task`): this script performs the write the
# card asks for, and the VERIFIER must then read green on it. That pair is what proves the
# verifier is satisfiable -- a verifier nothing can satisfy scores 0.0 forever and reads
# exactly like a model that cannot do the task. It is a separate file from the verifier
# because admission gives both arms the SAME environment, so one file cannot tell which
# arm it is in, and a script that wrote on both arms would make the empty-run leg vacuous.
#
# It plants in the directory the card's own write lands in, not merely in a directory the
# verifier happens to read: `memory_add(file="topics/<slug>")` resolves to
# `lloyd/memory/<slug>.md` (`app/memory_ceiling.py:72`, `agent_mcp/session.py:264`), so a
# plant at `lloyd/<sentinel>.md` would prove only that the two scripts agreed with each
# other. Agreement between a plant and a check that both name the wrong path is exactly the
# green that means nothing.
#
# Reads and writes ONLY the copy: `fixture.env()` exports `LLOYD_VAULT_ROOT`/`LLOYD_VAULT`
# pointing at the copy's vault, and the copy-first resolution plus the refusal below are what
# keep this from ever touching Alan's real vault — the accident #3403 exists to stop.
set -u
sentinel="ZEDQUILL-3688-VANE"
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

mkdir -p "$copy_vault/lloyd/memory" || exit 1
printf '%s\n\nWritten by the #3688 admission oracle, not by a model.\n' "$sentinel" \
  > "$copy_vault/lloyd/memory/bench-3688-plant.md" || exit 1
echo "performed: planted a topic note carrying the sentinel under the copy's lloyd/memory/"
exit 0
