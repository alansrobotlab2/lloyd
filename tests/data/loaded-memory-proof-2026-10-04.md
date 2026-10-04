<!-- WITNESS COPY — not a live record, and not today's state. Verbatim excerpts of
the two corrected proof passages of `~/obsidian/memory/vault-maintenance/2026-10-04.md`
(the vault repository, which is not this repository), graded for #2184 by
`tests/test_vault_commit_attribution.py`. That module's nodes parse the `git … → N`
lines below and re-run them, and
`test_the_witness_copy_matches_the_vault_record` re-checks these bytes against the live
vault file whenever it is readable. The vault file is the record; this is its witness.
-->

### 08:56Z run (task #24) — the corrected loaded-memory proof

No curated loaded-memory file is in that commit. What this section first offered as
the proof could not show it: it cited the message-inclusive word-list grep — whole
commit object, `MEMORY|USER|SOUL` as bare words, no `--pretty` — and reported the check
as clear. Re-run over `a49a256d` below, that command returns **7 lines**, so the sentence
describing it was false as written; the verdict it certified was still true, but nothing
reproducible supported it. Measured line by line, the defects are separable:

```
git -C ~/obsidian show --name-only a49a256d | grep -icE 'MEMORY|USER|SOUL'                → 7
git -C ~/obsidian show --name-only --pretty=format: a49a256d | grep -icE 'MEMORY|USER|SOUL' → 4
git -C ~/obsidian show --name-only --pretty=format: a49a256d | grep -icE '(^|/)lloyd/(MEMORY|USER|SOUL)\.md'   → 0
git -C ~/obsidian show --name-only --pretty=format: 30dae7c6 | grep -icE '(^|/)lloyd/(MEMORY|USER|SOUL)\.md'   → 1
```

Seven is 3 message lines + 4 filenames. The 3 message lines are the indented entries of
the `unattributed dirty state:` block the wrapper deliberately writes into the body (see
`scripts/util/vault-commit.sh`, and #1070 for why), which `git show` prints — and the
block is a copy of the file list, so the grep input contained the very claim the grep was
meant to establish. The 4 filenames are `memory/audit/writes.jsonl`,
`memory/skills-index.md`, `memory/skills-usage.jsonl` and
`memory/vault-maintenance/2026-10-04.md`, and every one matches on its **directory word
`memory`**: no path in this commit contains `USER` or `SOUL` at all.

Suppressing the message is therefore not sufficient on its own — the second line above
still returns 4 — and neither is anchoring alone. The form that decides the question is
message-suppressed *and* anchored to the filenames the runtime write guard protects
(`app/harness/protected_paths.py:157,224-225`). The fourth line is the positive control
that the anchored pattern is not inert: `30dae7c6` really did edit `lloyd/MEMORY.md`, and
that command reports 1 for it.

`#2184` moved the third command into `scripts/util/vault-commit.sh`, which now prints it
beside the `unattributed dirty state:` list, so a run record cites a line of shell
instead of a sentence about one. The four figures above are pinned against git by
`tests/test_vault_commit_attribution.py::test_the_08_56Z_record_states_the_anchored_command_and_not_the_returned_nothing_claim`.

This hash line arrived by a follow-up commit, because a commit cannot carry its own
hash — same shape as `980bce81` for the 04:38Z run above.

### 13:15Z run (task #24) — the corrected loaded-memory proof

The shape this section first recorded — `git show --name-only ca546bac | grep -iE
'MEMORY|USER|SOUL'`, no `--pretty` — **does** return lines. This section's first
correction said the 3 were prose drawn from the message body, and that was a miscount
too: measured line by line, they are **1 message line** (the indented
`memory/audit/writes.jsonl` inside the quoted unattributed block) and **2 of the
commit's own filenames** (`memory/audit/writes.jsonl`,
`memory/vault-maintenance/2026-10-04.md`), matched by their directory word. It also
reported that suppressing the message would leave 0, which is the same conflation in the
other direction: with `--pretty=format:` and the word pattern still unanchored the count
is **2**. The anchor is load-bearing. The three lines below are all measured on
`ca546bac` — this run's own commit, not `a49a256d`, which the 08:56Z section measures:

```
git -C ~/obsidian show --name-only ca546bac | grep -icE 'MEMORY|USER|SOUL'                → 3
git -C ~/obsidian show --name-only --pretty=format: ca546bac | grep -icE 'MEMORY|USER|SOUL' → 2
git -C ~/obsidian show --name-only --pretty=format: ca546bac | grep -icE '(^|/)lloyd/(MEMORY|USER|SOUL)\.md'   → 0
```

0 is the only answer that means what the run record said. The verdict was true; the
evidence quoted for it was not, because a grep over `git show` output matches the message,
and that message describes the very check it is being used to prove. `a49a256d` (the
08:56Z run's commit, whose section above quotes the same un-suppressed form) has the
identical shape: 7 lines = 3 message + 4 filenames, suppressed-unanchored 4, anchored 0.
These figures are pinned against git by
`tests/test_vault_commit_attribution.py::test_the_13_15Z_record_states_the_measured_split_of_the_flawed_form`.


This hash line arrived by a follow-up commit, because a commit cannot carry its own
hash — same shape as the 08:56Z run above.
