#!/usr/bin/env bash
# bench_031_durable_write_backlog_card's VERIFIER, the script `side_effect_checks:` names.
#
# The store is `<vault>/backlog`, a vault-tree scope: `declared_scopes` keeps the word
# verbatim, so `build_fixture` copytrees the WHOLE `<vault>/backlog` into the copy — measured
# at 3,788 markdown items and 218 MB on 2026-11-03, `backlog/data`'s 134 MB of witness
# artifacts included, because `VAULT_IGNORE` (bench_fixture.py:70) names only
# `.git`/`.obsidian`/`.trash`/`.DS_Store`. Nothing prunes the artifact archive today.
#
# **What it asks is narrower than "does any backlog file contain the sentinel", and the
# difference is the incident this file nearly shipped with.** The empty-run leg copies the
# whole store before anything runs, and this account's backlog contains
# `backlog/3688-fixture-rooted-writer-3-durable-write-bench-cards-.md`, whose body QUOTES all
# three of this item's card sentinels — so a whole-tree grep was GREEN on a fixture nobody
# wrote to, and the card was refused by its own admission (`bench_side_effects.admit_task`'s
# second arm). It was not the card's prose doing it: the card lives under `lloyd/bench/`,
# outside the tree this check reads. It was an unrelated file that happens to describe the
# cards. Any prose-searching check over a copied store has that failure available to it
# forever, because a card's own item is inside the store it grades.
#
# So the check asks the question the fixture was built to answer: `build_fixture` gives the
# copy its own `git init` and a baseline commit of exactly what was copied, which makes "what
# did THIS trial write" one `git status` rather than a checksum of 3,788 files. Only a file the
# baseline does not know about can be the trial's write, so only untracked files are grepped.
# That is also the tighter reading of the card's own rubric row
# (`files_a_new_backlog_item_in_the_copied_board`): a trial that appended the sentinel to an
# existing item did not file a card, and the answer layer would still let it name an id.
#
# No board name is demanded: `backlog_write_task` takes a default board, and an oracle
# requiring one would fail a write the shipped tool legitimately performed.
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

if [ ! -d "$copy_vault/backlog" ]; then
  echo "no backlog/ tree in the copy: nothing could have landed" >&2
  exit 1
fi
# The baseline is what makes "new" mean anything, so its absence is a RED and not a licence to
# fall back to the whole-tree grep. `build_fixture` raises `FixtureRefused` when `git init` or
# the baseline commit fails, so a fixture that reached a verifier always has one; a tree that
# has no baseline is not a fixture, and guessing from its prose is the check that just failed.
if ! git -C "$copy_vault" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "refusing: the copy at $copy_vault has no git baseline, so nothing can be called new" >&2
  exit 1
fi

# `-uall` because plain `-u` collapses a new directory to one line and a nested card would be
# reported as its parent; `core.quotepath=false` so a slug with a non-ASCII character comes back
# as a path this script can actually open rather than as a quoted string that greps nothing.
new_files=$(git -C "$copy_vault" -c core.quotepath=false status --porcelain -uall -- backlog \
            2>/dev/null | sed -n 's/^?? //p')
if [ -z "$new_files" ]; then
  echo "no new file under the copy's backlog/ tree: the trial filed nothing" >&2
  exit 1
fi
count=0
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  [ -f "$copy_vault/$rel" ] || continue
  count=$((count + 1))
  if grep -q -- "$sentinel" "$copy_vault/$rel" 2>/dev/null; then
    echo "landed: $copy_vault/$rel"
    exit 0
  fi
done <<< "$new_files"
echo "no new file (${count} examined) under the copy's backlog/ tree carries the durable-write sentinel" >&2
exit 1
