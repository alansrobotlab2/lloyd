#!/usr/bin/env bash
# bench_029_durable_write_vault_note's VERIFIER, the script `side_effect_checks:` names.
#
# Runs after a trial with nothing performed, and says whether a durable note is in the copy.
# The store is the topic-note directory a `memory_add(file="topics/<slug>")` actually writes:
# `app/memory_ceiling.py:72` sets `TOPICS_SUBDIR = "memory"` and `agent_mcp/session.py:264`
# resolves `topics/<slug>` to `<root>/memory/<slug>.md` under `MEMORIES_ROOT`, so the file on
# disk is `lloyd/memory/<slug>.md` while the tool calls it `topics/<slug>`. Both spellings are
# searched because a check named after the tool's word and reading only that word would report
# every real write absent, which is the failure this card's admission exists to catch.
#
# `lloyd/bench` is NOT searched, and that is the whole reason the leg is scoped to a directory
# instead of to `lloyd/`: the card's own body quotes the sentinel, the card is inside the copy,
# and a check that grepped the tree above the corpus would find the card and report the trial's
# write as landed. The empty-run leg is what proves this: it builds the same copy, writes
# nothing, and must come back RED with this file's own corpus sitting in it.
#
# Reads ONLY the copy: `fixture.env()` exports `LLOYD_VAULT_ROOT`/`LLOYD_VAULT` pointing at the
# copy's vault, and the copy-first resolution plus the refusal below are what keep this from ever
# touching Alan's real vault — the accident #3403 exists to stop.
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

for dir in "$copy_vault/lloyd/memory" "$copy_vault/memory"; do
  [ -d "$dir" ] || continue
  hit=$(grep -rl "$sentinel" --include='*.md' "$dir" 2>/dev/null | head -1)
  if [ -n "$hit" ]; then
    echo "landed: $hit"
    exit 0
  fi
done
echo "no markdown file under the copy's lloyd/memory/ (or memory/) carries the durable-write sentinel" >&2
exit 1
