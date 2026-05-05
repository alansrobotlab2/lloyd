# Real-Time KG Pipeline via QMD Watcher + Secondary Model

## Context

Four skills react to vault document changes:

| Skill | Output |
|-------|--------|
| `autonomy-data-pipeline` | `facts-index.json`, `relations-index.json`, entity fact dirs |
| `conversation-relation-linking` | `conversation-relation-proposals.json` |
| `entity-resolution-sweep` | merged entity dirs, `fact-aliases.json`, rebuilt `facts-index.json` |
| `groundskeeper-loop` | vault structural repairs (wikilinks, frontmatter, refs); drains groundskeeper queue |

Previously these were autonomy tasks (#24, #51, #48, #33) that ran on fixed schedules and spent most ticks doing a watermark check and exiting with `NO_NEW_DATA`. Meanwhile, `~/agent-services/scripts/qmd-watcher.sh` already emits a real-time file-change signal (inotifywait + 2s debounce, then `qmd update && qmd embed`). And the **secondary model** (Gemma-4 26B on GPU 0, port 8091) is a separate vLLM instance that can process these runs in real time without contending with interactive primary-model traffic.

**Resulting design:** the four skills are now invoked directly by a standalone pipeline triggered by the qmd watcher, pinned to the secondary model. The task files (#24, #48, #51, #33) have been deleted — they're no longer in the autonomy list and aren't run by the scheduler. Only the realtime pipeline exercises these skills. The groundskeeper step runs last so it can clean up any structural issues exposed by the KG mining steps.

## Event Flow

```
inotify in ~/obsidian
    ↓ (2s debounce, existing)
qmd update && qmd embed
    ↓ (new: post-embed curl)
POST http://127.0.0.1:8080/api/autonomy/vault-change
    ↓
_run_realtime_step("data-pipeline")
_run_realtime_step("conversation-relation-linking")
_run_realtime_step("entity-resolution-sweep")
_run_realtime_step("groundskeeper-loop")
    (serialized, pinned to secondary model, watermark-gated inside each skill)
```

## Changes

### 1. Watcher hook — `agent-services/scripts/qmd-watcher.sh`

Append one `curl -fsS -m 2 -X POST http://127.0.0.1:8080/api/autonomy/vault-change || true` after the `qmd embed` line. Non-blocking, tolerant of backend being down.

### 2. New endpoint — `app/routers/autonomy.py`

`POST /api/autonomy/vault-change`. Fires-and-forgets an asyncio background task that runs the three-task pipeline. Returns `202` immediately so the watcher never blocks.

### 3. Step executor — `autonomy.py`

`_run_realtime_step(step_name, skill_path, model="secondary")` invokes a skill directly via the Claude Agent SDK. No task-file indirection. Writes a run record to `~/lloyd/autonomy-runs/rt-<step_name>/` and updates `~/lloyd/autonomy-runs/realtime-state.json` with `last_run` / status / duration.

### 4. Pipeline orchestrator (`run_vault_change_pipeline`)

Iterates `REALTIME_PIPELINE_STEPS` in order. Per-pipeline coalescing via `_realtime_lock` + `_rerun_pending`: if a vault-change event arrives while the pipeline is mid-run, set the flag; re-fire once on completion. Never more than one pending re-run.

**Entity-resolution safety throttle:** the entity-resolution step has no internal watermark gate (unlike data-pipeline and relation-linking). Only fire it if its `realtime-state.json` `last_run` was more than 30 minutes ago. This prevents hammering the heaviest step under rapid vault activity.

### 5. Task files deleted

`24-data-pipeline.md`, `48-entity-resolution-sweep.md`, `51-conversation-relation-linking.md` have been removed from `~/obsidian/autonomy/`. The skills themselves (`~/obsidian/skills/{autonomy-data-pipeline,conversation-relation-linking,entity-resolution-sweep}/SKILL.md`) remain and are the canonical source of behavior.

## Concurrency Model

- **Existing scheduler lock** (`_ticker_running`): untouched. Still serializes scheduled primary-model tasks.
- **New realtime lock** (`_realtime_pipeline_running`): serializes the #24→#51→#48 pipeline against itself. Does NOT contend with the scheduler lock — secondary-model traffic on port 8091 is independent of primary on 8096.
- **Coalescing**: `_rerun_pending` flag, read and cleared by the pipeline on completion.

## Out of Scope

- #25 Memory Capture, #33 Groundskeeper Loop, #36, #45, #55, #59, etc. — these do not mine vault changes into KG facts. Only #24, #51, #48 qualify. They remain on their current schedules.
- No changes to the qmd daemon, watermark files, or skill bodies. Skills already do their own change detection where applicable.
- No new config keys — the three target task IDs are hardcoded in the orchestrator (single-purpose pipeline).

## Reused Existing Code

- `autonomy.py:316-331` `_get_model_env()` — resolves `secondary`'s env vars.
- `RunOptions(priority=1)` on the harness call — vLLM preempts realtime runs in favor of interactive chat (priority=0).
- `autonomy.py:146-159` `_write_run_record()` + `autonomy.py:116-129` `_update_task_field()` — reused by `run_task_realtime()`.

## Verification

1. Restart backend: `supervisorctl ... restart lloyd-mc:lloyd-backend`.
2. Restart watcher: `supervisorctl ... restart agent-qmd-watcher`.
3. Edit a vault file: `touch ~/obsidian/lloyd/test-realtime.md`.
4. Tail logs — expect in order:
   - qmd-watcher: `Change detected → updating → embedding → Triggered vault-change pipeline → Ready`
   - server.log: `POST /api/autonomy/vault-change 202`
   - server.log: `Realtime step: data-pipeline (model=secondary)` → `conversation-relation-linking` → `entity-resolution-sweep`
5. Check run records: `ls ~/lloyd/autonomy-runs/rt-data-pipeline/ rt-conversation-relation-linking/ rt-entity-resolution-sweep/` should show new `run_*.md` files within the last minute.
6. Check `~/lloyd/autonomy-runs/realtime-state.json` for per-step `last_run` timestamps.
7. Concurrency check: while the realtime pipeline runs, confirm the scheduler is still running other tasks.
8. Coalescing check: rapid-fire 5 edits within 10s. Expect one or two pipeline runs (first + one coalesced rerun).
9. Entity-resolution throttle check: trigger two changes 5 minutes apart. Expect the step to run on the first and skip on the second (`skipped: recent_run`).
10. Watermark skip check: trigger a change on an irrelevant file. Expect the data-pipeline and relation-linking steps to run and exit quickly via their internal `NO_NEW_DATA` watermark.
