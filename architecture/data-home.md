---
segment: architecture
tags: [architecture, lloyd, data, safety, guardian, backup, automod]
type: reference
status: implemented
date: 2026-09-23
---

# The data home: runtime data lives outside the code tree

`~/lloyd` holds code. Everything Lloyd *produces* lives in `~/lloyd-data`. That
covers transcripts, the databases, `_pipeline/`, logs and baselines. `CLAUDE.md`
has the short version under "Runtime data lives in ~/lloyd-data, never in the
tree".

## Why

On 2026-09-22 an implement round re-ran the full test suite with `~/lloyd` as
its rootdir. A fixture teardown under it deleted the tree in about 35 seconds
(`architecture/testing.md`). Git brought back the code. Nothing brought back what
was gitignored:

- `sessions/`, which was 725 MB before and 19 MB after;
- `_pipeline/`;
- `usage.db`, `workers.db` and `research.db`;
- `eval/baselines/`;
- the logs.

The data went because it sat inside the code. Every `rm -r`, `git clean -x`
and test fixture aimed at the code could reach it.

## Layout

`~/lloyd-data` is its own btrfs subvolume. Under it, every name is the one it
had in the tree, so a path `~/lloyd/X` became `~/lloyd-data/X`:

```
~/lloyd-data/
├── .lloyd-data-root          # the marker app.paths requires in production
├── sessions/                 # + <sid>.changes/, <sid>.tool-results/
├── event_logs/
├── _pipeline/                # tasks, vault-derived (kg.sqlite, facts, exports), research, …
├── autonomy-runs/
├── logs/                     # server/mcp/frontend/agent-worker .log/.err, locks/, screenshots/
│   └── services/             # the engines' supervisor logs (was agent-services/logs/)
├── eval/baselines/
├── voice_profiles/
├── desktop/                  # the computer-use lease, created on first use
├── data/tool_overrides.yaml
├── usage.db  workers.db  research.db
└── mc-state.json
```

Some things stay in the tree because they are code, build output or a
rebuildable cache, not data: `.venvs/`, `qmd/`, model weights, `node_modules`,
`web/dist`, `graphify-out/` and `__pycache__`. `graphify-out/` is per tree on
purpose, because `code_graph` answers questions about one root. Automod and
guardian state was already outside the tree, in `~/.local/state/lloyd-*`.

## One resolver: `app.paths.DATA_ROOT`

Every data path hangs off `DATA_ROOT`:

- `SESSIONS_DIR`, `EVENT_LOGS_DIR`, `USAGE_DB`, `WORKERS_DB`;
- `PIPELINE_DIR`, `VAULT_DERIVED_ROOT`, `LOGS_DIR`, `SERVICE_LOGS_DIR`;
- `EVAL_BASELINES_DIR` and the rest.

`config.yaml` names data paths as `${LLOYD_DATA}/…`. `app.config` expands that
to the resolved root, whether or not the variable is exported.

The first rule that matches wins:

1. **`LLOYD_DATA`** in the environment. The gate, the canary and the test suite
   set it.
2. **The production checkout** uses `<passwd home>/lloyd-data`. The production
   checkout is the one where `LLOYD_HOME` is the passwd home's `lloyd` and is
   not a worktree. It is read from passwd and never from `Path.home()`, because
   under a gate's `HOME` that name belongs to the round. The root must carry
   `.lloyd-data-root`. Without the marker, `app.paths` raises `DataRootMissing`
   instead of falling back to the tree. A silent fallback would quietly start a
   second copy of everything inside the tree, which is the failure this whole
   move prevents.
3. **Any other checkout** (the sandbox, a worktree, a scratch clone) uses
   `<tree>/.lloyd-data`, which is gitignored. "State follows the code" is what
   kept a canary off the live `workers.db` before this change. It stays the
   default so that the unsafe direction never happens by omission.

Rule 3 reaches whatever imports `app.paths` — and `app.paths` needs the project
venv, so the jobs that run without it used to re-implement the resolution as
`${LLOYD_DATA:-~/lloyd-data}`, which is rule 2 with neither the marker check nor
rule 3. The one that mattered was the sweep, because it deletes: `--apply` from a
sandbox or a round worktree reached the live root, invisible to the delete guard
(a Python `unlink` is not an `rm`) and to the tripwire (#1415).

The rules therefore live in `app/data_root.py`: stdlib-only, importing nothing
from `app` and touching nothing but `pwd` and one `is_file()`. `app.paths`
imports and re-exports it, so `app.paths.DATA_ROOT` and every derived constant
are unchanged; a script with no venv calls
`app.data_root.resolve_data_root_for_tree()` and gets the same three rules, or
`DataRootMissing`. `scripts/groundskeeper/retention-sweep.py` is converted, and
its run prints the root it resolved in both dry-run and `--apply`, because every
number it reports is a count of that root.

The stdlib readers still to convert — `agent-services/guardian/{policy,datawatch}.py`,
`idle-worker.py`, `livekit_worker.py`, `snapshot-data.sh`, `restore-data.sh` —
each read `${LLOYD_DATA:-~/lloyd-data}` directly. They resolve the live root from
any tree, which is wrong but harmless for readers; the sweep was the one with
`unlink` behind it. Import the same accessor when one of them next changes.

**Nothing exports `LLOYD_DATA` in production, and nothing should.** The backend
and the aggregator find the root by rule 2. Their Bash children inherit their
environment. A child that ran a worktree's code with the live root exported
would write live data, which is the leak rule 3 exists to prevent.

Some consumers reach into live data *on purpose*: the review grader reading a
round's sessions, and the promoter's knowledge-graph probe. They use
`production_data_root()`. The name and the `app.paths` docstring both say
readers only — "nothing that writes should" — but four jobs write through it
deliberately: the grader's own session file, the nightly extraction's lock, log
and backups, the content hasher's index, and the tool-override sync. Nothing
lists them, so an accidental reach looks the same as a deliberate one (#1415).

## Isolation: who gets which root

| Who | `LLOYD_DATA` | Set by |
|---|---|---|
| backend, aggregator, timers | unset → `~/lloyd-data` (rule 2) | — |
| a gate rung running candidate code | `<round>/home/lloyd-data` | `Gate._child_env` (every call) |
| the `tool_choice` rung (live scripts, live baselines) | unset → production | `_child_env(live_data=True)` |
| the canary | `<round>/canary-home/lloyd-data` | `canary_config.canary_env` |
| pytest | a fresh mkdtemp | `tests/conftest.py::_data_root_to_scratch` |

The round home is a symlink farm over the real `$HOME`. `ensure_round_home`
skips `lloyd-data` (`HOME_LINK_SKIP`) and makes it a real, empty directory, so
`~/lloyd-data` in a round is the round's own. conftest refuses a run whose
`LLOYD_DATA` is the live root, with the same `LLOYD_ALLOW_LIVE_TREE_TESTS=1`
opt-in as the tree guard. The live-data cross-checks in `tests/` read through
`production_data_root()` and are read-only.

## Protection

- **Delete guard.** `app/harness/protected_paths.py` treats `~/lloyd-data`, the
  resolved `DATA_ROOT` and `~/.lloyd-data-snapshots` as protected roots, like
  the vault and the tree. It refuses `rm -r`, `find -delete`, `rsync --delete`,
  `mv` and `git clean` of the root or a top-level folder in it, and a
  `shutil.rmtree` one-liner, at both the hook and dispatch. Targeted deletes of
  single files stay allowed. This guard parses Bash command strings. It cannot
  see a delete made inside a Python process, which is how the 09-22 deletion
  happened. That gap is why the next two layers exist.
- **Snapshots.** `lloyd-data-snapshot.timer` (user) runs
  `scripts/backup/snapshot-data.sh` every hour. It takes a read-only btrfs
  snapshot into `~/.lloyd-data-snapshots/<UTC stamp>`.
  - A snapshot is atomic, so a WAL-mode SQLite database opens from it the way it
    opens after a power cut.
  - `/home` is not mounted `user_subvol_rm_allowed`, so nothing running as the
    user can delete a snapshot. Measured on 2026-09-22: `btrfs subvolume
    delete` returned EPERM and `rm -rf` returned EROFS.
  - Pruning therefore needs root. `scripts/backup/prune-data-snapshots.sh` keeps
    48 hourly and 14 daily snapshots and is installed as a system timer; its
    header has the three `sudo` lines.
  - The snapshot script refuses while the data tripwire is set, or when the root
    shrank below its last healthy measurement. A refusal exits 0 on purpose, so
    the timer never shows as failed — and the unit is `Type=oneshot`, so
    `systemctl --user show lloyd-data-snapshot.service` still reads
    `Result=success ExecMainStatus=0` after refusing. Neither the timer nor
    systemd is a surface for a silent refusal.
  - **What watches it:** the guardian's data check asks that directory how old
    its newest snapshot is every `policy.SNAPSHOT_CHECK_SECONDS` (900 s) and
    alerts `error` past `policy.SNAPSHOT_MAX_AGE_SECONDS` (3 h — three hourly
    periods, so one skipped hour is not an alarm), naming the stamp and its age.
    It stays quiet while the data tripwire is set, because that refusal is
    intended and already paged, and it is not armed on a machine with no data
    root at all — that box has nothing to snapshot and its own alert. An empty
    directory is reported when the root exists: present and listable is not the
    same as restorable.
  - A count of the directory cannot stand in for that age check. Pruning never
    deletes the newest snapshot whatever its age, so the entry count stays at 1
    or more through any length of outage. `datawatch.py snapshot-age [--tsv <dir>]`
    is the one measurement; `restore-data.sh` with no arguments prints the newest
    stamp's age through it, so what an operator sees is what would have alerted.
- **Tripwire.** `agent-services/guardian/datawatch.py` runs every guardian tick.
  It uses `vaultwatch`'s measurement and thresholds, and trips when:
  - the root is missing or was replaced;
  - the root lost its marker;
  - at least 10% and at least 200 files disappeared within 15 minutes;
  - a top-level folder of 20 or more files emptied.

  A trip pauses the pool, halts promotions, keeps a process listing, writes
  `data-tripped.json` and alerts critical. Only a human clears it
  (`datawatch.py clear`). The watch is not armed until it first sees a marked
  root. Once an hour it also lists any runtime name that has come back into the
  tree (`datawatch.stray_in_tree`), which means some writer still resolves its
  path off the code.
- **Restore.** `scripts/backup/restore-data.sh <stamp>` copies a snapshot into a
  side directory with reflinks, so it is instant, and integrity-checks every
  database. Swapping it in is a human step. The order is in the script's header.

## The move (2026-09-22)

`scripts/migrate_data_home.py` did the move once, dry run by default. It:

1. refuses while any process holds a source file open;
2. checkpoints every WAL;
3. makes reflink copies;
4. verifies file count, bytes, the sha256 of each database and
   `integrity_check`;
5. moves the originals to `~/lloyd-data-migration-hold/<stamp>/`, outside the
   tree;
6. writes the marker last.

`scripts/maintenance/rewrite_vault_data_paths.py` pointed the vault's
instructions at the new root: `autonomy/*.md` and `skills/**` only. `backlog/`,
`memory/`, `knowledge/` and `projects/` are a record and were left alone.

The supervisor reopens a program's log file only when it restarts that program,
and reopens its own log only when the unit restarts. So the cutover was one
stop of `agent-supervisord.service` with the guardian stopped, not a series of
`round restart`s.

## Review log

- 2026-09-23 — `current` (round `SM_20260923_204702`, #1415). The three rules moved
  into `app/data_root.py` and `app.paths` re-imports them, which converts the one
  stdlib job that deletes: `retention-sweep.py` now resolves by rule 3 in a
  sandbox or a worktree, refuses on an unmarked production root, and prints the
  root in both modes (`tests/test_retention_sweep.py`, five clauses). Supersedes
  the entry below on one point: "rule 3 does not reach the stdlib-only scripts" is
  no longer true of the sweep — it still names the five stdlib readers above,
  which reach the live root from any tree and are read-only.
- 2026-09-23 — Checked against `a9f5fef`: the three resolution rules
  and the marker, the nine `${LLOYD_DATA}` keys in `config.yaml`, the
  gate/canary/pytest isolation rows, the delete guard's refusals (probed live: the
  root, a top-level folder, a glob over one, `find` with `-delete`, `rsync
  --delete`, `mv`, `git clean`, an `rmtree` one-liner — a single file stays
  allowed), `vaultwatch`'s thresholds (10 %, 200 files, 900 s, a 20-file top
  dir), the hourly user snapshot timer and the root prune timer, both live, and
  the migration's six steps. Corrected: the `CLAUDE.md` heading quoted above, the
  `desktop/` child `265d597` added, and two claims stated past their coverage —
  rule 3 does not reach the stdlib-only scripts (#1415), and no monitor notices
  snapshots stopping (#1416).
