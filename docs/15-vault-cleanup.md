# 15 — Vault Cleanup: Agent State Out of Vault

> Moved auto-generated agent-state directories out of `~/obsidian/` into
> `~/lloyd/` where they belong, promoted structured facts into the vault as a
> first-class segment, and cleaned up QMD collections.

---

## Changes Summary

### 1. `_pipeline` moved out of vault

| Old | New |
|---|---|
| `~/obsidian/memory/_pipeline/` | `~/lloyd/_pipeline/` |

All scripts, skills, and code updated. Groundskeeper exclusions for
`memory/_pipeline/` removed from broken-link, orphan, and large-doc scans —
the path no longer exists in the vault.

### 2. `autonomy/runs` moved out of vault

| Old | New |
|---|---|
| `~/obsidian/autonomy/runs/` | `~/lloyd/autonomy-runs/` |

`autonomy.py` and `server.py` updated. Groundskeeper `autonomy/runs/` exclusions
removed. Not indexed in QMD (runs are ephemeral logs, not knowledge).

### 3. `lloyd/knowledge/` merged into vault

`~/lloyd/knowledge/ai/` contained 4 files. Merged into `~/obsidian/knowledge/ai/`:

- **Moved:** `agent-amnesia-self-learning.md`, `autoagent-research-verification.md`
- **Discarded (vault had richer versions):** `autoagent-self-improving-agents.md`, `self-improving-agents.md`

`~/lloyd/knowledge/` directory removed.

### 4. `facts` promoted to vault root segment

| Old | New |
|---|---|
| `~/lloyd/_pipeline/facts/` | `~/obsidian/facts/` |

Facts are structured knowledge, not pipeline state — they belong in the vault.
All `FACTS_ROOT` constants updated across Python code. Added to `VAULT_SEGMENTS`
in `mcp_server/memory.py` so vault search includes facts.

### 5. QMD collection updates

| Change | Detail |
|---|---|
| Added `facts` collection | `~/obsidian/facts`, 874 files |
| Removed `agents` collection | Empty (0 files), deleted 32 stale docs |
| Removed `autonomy-runs` collection | Runs are not searchable knowledge |
| Removed `_pipeline/**` ignore from `memory` | Path no longer in vault |
| Removed `memory/_pipeline/**` ignore from `subliminal` | Same |
| Removed `runs/**` ignore from `autonomy` | Path no longer in vault |

---

## Files Modified

### `~/lloyd/`

| File | Change |
|---|---|
| `server.py` | `_FACTS_ROOT` → `~/obsidian/facts`; `_RELATIONS_INDEX` updated; `_AUTONOMY_RUNS_DIR = ~/lloyd/autonomy-runs` |
| `autonomy.py` | `AUTONOMY_RUNS_DIR = Path.home() / "lloyd" / "autonomy-runs"` |
| `mcp_server/memory.py` | `FACTS_ROOT` → `~/obsidian/facts`; `_pipeline` removed from `VAULT_EXCLUDE_DIRS`; `"facts"` added to `VAULT_SEGMENTS` |
| `web/src/components/EntityGraph.tsx` | Node color prefix updated to `facts/` |
| `scripts/extract-trajectories.py` | `OUTPUT_DIR` updated |
| `scripts/mine-trajectories.py` | `TRAJECTORY_DIR`, `OUTPUT_DIR` updated |
| `scripts/groundskeeper/groundskeeper-survey.py` | `QUEUE_OUTPUT`, `FACTS_DIR` updated; stale exclusions removed |
| `scripts/groundskeeper/groundskeeper-weekly-summary.py` | Log/output/queue paths updated |
| `scripts/memory/batch-process-orphans.py` | Queue path updated |
| `scripts/memory/process-groundskeeper-queue.py` | Queue/log paths updated |
| `scripts/memory/rebuild_index.py` | `RELATIONS_INDEX`, `FACTS_INDEX`, `FACTS_DIR` updated |
| `scripts/autonomy/self_improve.py` | `METRICS_DIR`, `WATERMARKS_FILE`, `PENDING_IMPROVEMENTS_FILE` updated |
| `scripts/memory/next-gen-memory/nightly_extraction.py` | `FACTS_DIR`, `INDEX_FILE`, log, registry paths updated |
| `scripts/memory/next-gen-memory/fact_extractor.py` | `FACTS_DIR` updated |
| `scripts/memory/next-gen-memory/relations_index.py` | `index_file` updated |
| `scripts/memory/next-gen-memory/generate_relationship_proposals.py` | Relations index paths updated |

### `~/obsidian/`

- `skills/*/SKILL.md` — 34 files: all `_pipeline` and `autonomy/runs` path references updated
- `architecture/*.md` — path references updated
- `scripts/` — same path updates as lloyd scripts (exact copies)

### `~/.config/qmd/index.yml`

Added `facts` collection; removed `agents` and `autonomy-runs` collections;
removed three stale `ignore` entries.
