# Lloyd Autoresearch Loop — Karpathy-Style Continual Self-Improvement

## Context

Lloyd already has the ingredients — **memory** (19.8K facts, entity graph at [../agent_mcp/memory.py](../agent_mcp/memory.py)), **next-gen memory** (nightly extraction + health reports in [../scripts/memory/next-gen-memory/](../scripts/memory/next-gen-memory/)), **autonomy** (cron-like scheduler at [../autonomy.py](../autonomy.py)), **subliminal** (SOUL injection at [../agent_mcp/subliminal.py](../agent_mcp/subliminal.py)), **pipeline** (parallel SDK runner with priority proxy at [../agent_mcp/pipeline.py](../agent_mcp/pipeline.py)), and a **4-way concurrent vLLM** on port 8096 — but these don't yet compound. [../scripts/autonomy/self_improve.py](../scripts/autonomy/self_improve.py) is a single-threaded measure→propose→eval scaffold with no persistent ledger, no A/B, and no rollback safety.

**The gap:** idle GPU cycles are wasted. Karpathy's autoresearch pattern — parallel hypothesis generation, frozen eval bench, objective scoring, persistent ledger — is exactly what closes the loop. With `max-num-seqs` bumped 4→8 and existing priority=2 preemption, 7 slots run experiments 24/7 while 1 slot stays hot for the user.

**Intended outcome:** a nightly autoresearch round produces measurable deltas (e.g., +X pts on eval bench) that flow back into prompts, skills, retrieval params, and tool allowlists — with human-auditable provenance in the fact graph and full attempt log on disk.

Related prior work: [12-meta-optimization-gameplan.md](12-meta-optimization-gameplan.md) (the measure-propose-evaluate scaffolding this plan formalizes and parallelizes), [18-autonomy-redesign.md](18-autonomy-redesign.md), [16-vault-intelligence-gameplan.md](16-vault-intelligence-gameplan.md).

---

## Architecture

```
┌─ Autonomy scheduler (nightly 2:30am, after knowledge-health)
│    └─▶ autoresearch_round MCP tool
│           │
│           ├─[1] Hypothesis Generator  (122B, thinking=false)
│           │       inputs: recent eval failures + correction log +
│           │               knowledge-health report + prior ledger
│           │       output: N variant patches (≤7 per round)
│           │
│           ├─[2] Variant Sandbox        (copy SOUL/MEMORY/USER/skills/config
│           │                             into _pipeline/research/variants/<id>/)
│           │
│           ├─[3] Parallel Executor      (fan out variants × bench tasks
│           │                             through priority proxy @ 8097,
│           │                             priority=2, 7-way parallelism)
│           │
│           ├─[4] Two-Layer Judge        (objective checks + 122B rubric)
│           │                             → composite score per (variant, task)
│           │
│           ├─[5] Ledger Write            JSONL: _pipeline/research/ledger.jsonl
│           │                             (every attempt, always)
│           │
│           └─[6] Promotion Gate          if winner beats baseline on ≥60%
│                                         of bench with Δ>threshold AND no
│                                         regression on safety tasks:
│                                           - snapshot current → _pipeline/snapshots/<ts>/
│                                           - atomic swap
│                                           - write winning experiment as FACT
│                                             in ~/obsidian/facts/experiments/
│                                           - update MEMORY.md entry via vault_write
└─ User session interrupts → vLLM preempts priority=2 work → slot freed
```

---

## Scope decisions (confirmed)

- **Optimization targets:** all four surfaces — prompts (SOUL/MEMORY/USER), skills (`~/obsidian/skills/`), retrieval policy (`vault_recall` params, consolidation thresholds), and tool allowlist (`mcp_servers.*.disabled_tools`, `tools.disabled_builtin`).
- **GPU budget:** background only. Bump vLLM `max-num-seqs` 4→8, reserve 1 slot for interactive, route research at priority=2 so user sessions preempt.
- **Ledger storage:** **both** — every attempt in JSONL for full audit; promoted winners additionally written as facts so the agent can recall them via normal retrieval.

---

## Components to build

### New: `agent_mcp/autoresearch.py` (MCP server)

Tools exposed (register in [../config.yaml](../config.yaml) `mcp_servers`):

- `autoresearch_round(targets=[...], budget_minutes=N)` — orchestrate full round
- `autoresearch_bench_list()` — show eval tasks
- `autoresearch_bench_add(task_yaml)` — extend bench
- `autoresearch_ledger_query(filter)` — read JSONL results
- `autoresearch_promote(variant_id, dry_run=bool)` — manual promotion path
- `autoresearch_rollback(snapshot_ts)` — revert to a prior snapshot
- `autoresearch_status()` — current round progress, last winner, next scheduled

### New: `scripts/autoresearch/` directory

- `run_round.py` — entry point invoked by MCP or autonomy
- `hypothesis_generator.py` — LLM-driven variant proposer; reads from the four target surfaces; emits unified-diff-style patches
- `variant_sandbox.py` — materialize a variant: deep-copy the agent config tree into `_pipeline/research/variants/<variant_id>/` (overlay dir consumed by `prompt_builder` and vLLM env)
- `bench_runner.py` — for each (variant, task), spawn SDK `query()` with env vars pointing at variant overlay, collect trace (turns, tools, tokens, final output); reuses the priority-proxy pattern from [../agent_mcp/pipeline.py](../agent_mcp/pipeline.py) at priority=2
- `judge.py` — objective (pass/fail, tool-correctness, output schema) + rubric (122B, thinking=false, JSON-mode) → composite score 0..1
- `promote.py` — snapshot + atomic swap + fact write; honors `--dry-run`

### New: `~/obsidian/lloyd/bench/`

Initial seed (10 tasks minimum):
- 4 replayed from session transcripts (known-good outcomes in `sessions/`)
- 3 synthetic probes (vault_recall lookup, fact contradiction resolution, skill invocation correctness)
- 2 adversarial (gap detection, hallucination resistance)
- 1 **safety-critical** task (must-not-regress gate — e.g., refuses destructive op without confirmation)

Task schema (YAML frontmatter + markdown body):
```yaml
---
id: bench_001
category: replay | synthetic | adversarial | safety
prompt: "..."
objective_checks:
  - type: contains | regex | tool_called | tool_not_called
    value: "..."
rubric_criteria: [clarity, accuracy, cost_efficiency]
safety_critical: true | false
---
```

### New: `~/obsidian/autonomy/nightly-autoresearch.md`

Autonomy task consumed by existing scheduler at [../autonomy.py](../autonomy.py):
```yaml
---
title: Nightly Autoresearch
frequency: daily
preferred_hours: [2, 3, 4]
priority: background
depends_on: nightly-knowledge-health
timeout_seconds: 10800   # 3h cap
---
Run autoresearch_round with targets=[prompts, skills, retrieval, tool_allowlist]
and budget_minutes=120. Summarize promoted changes (if any) to MEMORY.md.
```

### Modifications

- [../scripts/run-vllm-19-turboquant.sh](../scripts/run-vllm-19-turboquant.sh) line 36 — `--max-num-seqs 4` → `--max-num-seqs 8` (verify GPU headroom first with benchmark)
- [../config.yaml](../config.yaml) — add `autoresearch:` block (bench path, round budget, promotion thresholds, safety gates); register `autoresearch` MCP server
- [../agent_mcp/pipeline.py](../agent_mcp/pipeline.py) — extract the priority-proxy SDK spawn into a shared helper so `bench_runner.py` can reuse without duplication
- [../scripts/autonomy/self_improve.py](../scripts/autonomy/self_improve.py) — deprecate header; delegate to `autoresearch_round`

### Storage layout

```
~/lloyd/_pipeline/research/
├── ledger.jsonl                    ← every attempt, append-only
├── rounds/<round_id>.md            ← round summary (hypothesis, results, winner)
├── variants/<variant_id>/          ← overlay: SOUL.md/MEMORY.md/skills/config override
└── snapshots/<ts>/                 ← pre-promotion snapshot for rollback

~/obsidian/facts/experiments/
└── <entity>/<entity>-experiment.md ← promoted winners only, queryable via fact_get
```

---

## Reused building blocks (do not reinvent)

| Existing piece | Reuse as |
|---|---|
| [../agent_mcp/pipeline.py](../agent_mcp/pipeline.py) priority-proxy runner | `bench_runner` parallel spawn pattern |
| [../autonomy.py](../autonomy.py) frequency/preferred_hours/depends_on | scheduling `nightly-autoresearch` |
| [../agent_mcp/memory.py](../agent_mcp/memory.py) `fact_add` + relations | ledger promotion to semantic memory |
| [../prompt_builder.py](../prompt_builder.py) SOUL/MEMORY/USER loader | consume via `LLOYD_OVERLAY_DIR` env var for variant sandboxing |
| [../scripts/memory/knowledge-health-report.py](../scripts/memory/knowledge-health-report.py) output | hypothesis-generator input |
| [../usage_store.py](../usage_store.py) per-request metrics | cost/latency columns in ledger |
| [../scripts/benchmark.py](../scripts/benchmark.py) | pre-flight GPU headroom check before bumping `max-num-seqs` |

---

## Safety gates

1. **Safety-critical eval task must never regress** — even a winner is rejected if it fails the safety probe.
2. **Atomic swap + snapshot** — every promotion snapshots the full overlay to `_pipeline/snapshots/<ts>/` before writing. `autoresearch_rollback <ts>` restores.
3. **Promotion requires Δ > threshold on ≥60% of bench** (configurable; default 0.05 composite delta).
4. **Second-pass judge** on the winning variant using a different rubric prompt to catch overfitting to the judge.
5. **Priority=2 preemption** — bench work yields to user sessions at the vLLM scheduler level, no extra code needed.
6. **Tool-allowlist target is gated harder** — any change to `disabled_tools` requires 2 consecutive winning rounds before auto-promotion (otherwise proposal is logged for human review).

---

## Verification plan

1. **Unit shakedown**
   - Seed `~/obsidian/lloyd/bench/` with 3 tasks (1 replay, 1 synthetic, 1 safety)
   - Run `python scripts/autoresearch/run_round.py --budget 10 --dry-run` — confirms variant generation + sandboxing + ledger writes without promoting
   - Inspect `_pipeline/research/ledger.jsonl` for well-formed entries

2. **End-to-end (manual trigger)**
   - Call `autoresearch_round(targets=[prompts], budget_minutes=20)` via MCP
   - Verify ≥3 variants generated, each scored across 3 bench tasks
   - Verify round summary written to `_pipeline/research/rounds/<id>.md`
   - If winner found: verify snapshot exists and fact appears in `~/obsidian/facts/experiments/`

3. **Concurrency verification**
   - Bump `max-num-seqs` to 8, run [../scripts/benchmark.py](../scripts/benchmark.py) to confirm no quality regression vs 4-seq baseline
   - Submit an interactive user query mid-round; measure TTFT — should be ≤2× idle baseline (priority preemption working)
   - Inspect vLLM logs for evidence of priority=2 requests being preempted

4. **Autonomy integration**
   - Drop `nightly-autoresearch.md` into `~/obsidian/autonomy/`
   - Wait for scheduler (or force via [../autonomy.py](../autonomy.py) manual trigger) — confirm run record in `~/lloyd/autonomy-runs/nightly-autoresearch/`

5. **Rollback drill**
   - After a promotion, call `autoresearch_rollback <ts>` — confirm SOUL/MEMORY/USER/skills/config restored byte-for-byte
   - Diff against snapshot to prove determinism

6. **Ledger queryability**
   - `fact_get experiments/<winner>` returns the promoted experiment
   - `vault_recall "autoresearch"` surfaces recent winners via the agent's normal recall path

---

## Open questions / nice-to-haves (defer)

- **Population-based training:** maintain a pool of top-K variants and cross-breed their patches. Worth it only once single-round wins flatten.
- **Bench autogrowth:** have the autoresearch loop propose new bench tasks from failure modes it discovered. Requires a separate curation gate to prevent judge-hacking.
- **Cross-session learning:** turn every real user session into a bench candidate via opt-in flagging. Big upside, privacy considerations.
- **Fine-tuning export:** once the ledger is rich, export (input, winning-variant-output, score) tuples as a future SFT/DPO dataset.
