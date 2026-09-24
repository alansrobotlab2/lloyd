# Episodic raw-transcript arm vs the recall path (#675)

Run 2026-09-24T21:05:10+00:00. Baseline `nightly-20260924-20260924-065033.json` (ran 2026-09-24T13:50:33.497410+00:00), re-derived today; the item's nightly-20260909 figures are quoted beside it (artifact lost in the 09-22 wipe).

## Setup

- Corpus: 662 exported chat transcripts, 2026-08-22 → 2026-09-24 (656 pre-wipe docs from ~/.cache/qmd/index.sqlite.bak-gemma-20260921 + 6 live exports).
- Turn kinds included: user, lloyd, tool_call, tool_result. user:/lloyd: open a turn; tool_call: and → [OK]/[ERROR] lines attach to the turn they follow.
- Expansion: k ∈ {0, 2, 5}, 1024 tokens per episode (ceil(chars/4)), nearest-first (before, after), whole turns, first misfit ends it.
- Gold: episode text contains an expect_entities name or an expect_docs path/stem (normalised substring).
- Retrieval: qmd structured search, lex+vec, lexMode or, rerank on, candidateLimit 40; top 10 episodes.
- Queries: 81 (the baseline's scored set); zero-gold (no label in any corpus turn, excluded from arm rates): **9**; eligible: 72.

## Headline (eligible queries unless marked)

| arm | n | entity_hit | doc_hit | entity_recall | doc_recall | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| nightly-20260909 (quoted, 20 q) | 20 | 0.5 | 0.85 | 0.4 | 0.55 | 0.447 | 0.536 |
| baseline today, all | 81 | 0.358 | 0.704 | 0.391 | 0.647 | 0.355 | 0.407 |
| baseline today, eligible | 72 | 0.389 | 0.750 | 0.382 | 0.660 | 0.380 | 0.434 |
| episodic k=0 | 72 | 0.694 | 0.361 | 0.666 | 0.260 | 0.475 | 0.556 |
| episodic k=0, eval-talk dropped | 72 | 0.597 | 0.347 | 0.566 | 0.248 | 0.451 | 0.510 |
| vault-doc text control k=0 | 72 | 0.736 | 0.486 | 0.745 | 0.338 | 0.516 | 0.566 |
| episodic k=2 | 72 | 0.694 | 0.389 | 0.666 | 0.275 | 0.485 | 0.565 |
| episodic k=2, eval-talk dropped | 72 | 0.597 | 0.375 | 0.558 | 0.262 | 0.459 | 0.520 |
| vault-doc text control k=2 | 72 | 0.792 | 0.556 | 0.806 | 0.385 | 0.596 | 0.646 |
| episodic k=5 | 72 | 0.694 | 0.389 | 0.666 | 0.275 | 0.485 | 0.566 |
| episodic k=5, eval-talk dropped | 72 | 0.597 | 0.375 | 0.558 | 0.262 | 0.459 | 0.521 |
| vault-doc text control k=5 | 72 | 0.806 | 0.556 | 0.816 | 0.404 | 0.617 | 0.668 |

The baseline's entity metrics come from the fact layer; the episodic and control rows use text containment. Compare episodic against the control row at the same k, not against the baseline row. The baseline's MRR/nDCG rank gold *documents*; the arm's rank gold *episodes*.

95% CIs (Wilson for hit rates, bootstrap for means):

- k=0: episodic entity_hit 0.694 [0.58, 0.789], eval-talk dropped 0.597 [0.482, 0.703], control 0.736 [0.624, 0.824]
- k=2: episodic entity_hit 0.694 [0.58, 0.789], eval-talk dropped 0.597 [0.482, 0.703], control 0.792 [0.684, 0.869]
- k=5: episodic entity_hit 0.694 [0.58, 0.789], eval-talk dropped 0.597 [0.482, 0.703], control 0.806 [0.7, 0.88]

## The miss subset (baseline entity_hit = 0)

52 queries. Expected entity present in any corpus turn: 36. Arm surfaced it at any k: 27 (0.519); with eval-talk episodes dropped: **21 (0.404, 95% CI [0.282, 0.539])**; of those, not also carried by the vault docs' own text: 3.

| query | category | in corpus | arm any k | arm any k (clean) | k=0 | k=2 | k=5 | vault-doc text |
|---|---|---|---|---|---|---|---|---|
| backlog-363 | single | yes | yes | yes | yes | yes | yes | yes |
| entity-resolution-sweep | single | yes | yes | yes | yes | yes | yes | yes |
| vault-recall | single | no | no | no | no | no | no | no |
| kg-maintenance-tasks | multi-hop | yes | yes | yes | yes | yes | yes | yes |
| qwen38-local-serving | multi-hop | yes | yes | yes | yes | yes | yes | yes |
| graph-quality | recent | yes | yes | yes | yes | yes | yes | yes |
| godnode-threshold | technical | yes | no | no | no | no | no | yes |
| relationships-location | technical | yes | yes | yes | yes | yes | yes | yes |
| memory-persistence | fuzzy | yes | yes | yes | yes | yes | yes | yes |
| robotics-projects | hard | yes | yes | yes | yes | yes | yes | yes |
| tts-voice-cloning | single | yes | yes | yes | yes | yes | yes | yes |
| wake-word-models | single | no | no | no | no | no | no | no |
| gpu-model-naming-collision | single | yes | yes | yes | yes | yes | yes | yes |
| browser-tool-validation | single | yes | yes | yes | yes | yes | yes | yes |
| three-d-printing-calibration | single | yes | no | no | no | no | no | yes |
| config-yaml-readonly | single | no | no | no | no | no | no | no |
| kg-dedup-key | single | yes | yes | no | yes | yes | yes | yes |
| dream-to-skill-edit | multi-hop | yes | yes | yes | yes | yes | yes | yes |
| automod-to-entity-guard | multi-hop | yes | yes | yes | yes | yes | yes | yes |
| counterfactual-to-trend-audit | multi-hop | yes | yes | no | yes | yes | yes | yes |
| autonomy-task-to-skill | multi-hop | yes | yes | yes | yes | yes | yes | no |
| voice-session-to-room | multi-hop | yes | yes | yes | yes | yes | yes | yes |
| watchdog-metacharacter | technical | no | no | no | no | no | no | no |
| yaml-scalar-block-indent | technical | no | no | no | no | no | no | no |
| json-filter-epoch-vs-iso | technical | no | no | no | no | no | no | no |
| bash-arrays-vs-strings | technical | no | no | no | no | no | no | no |
| git-signals-in-async-code | technical | no | no | no | no | no | no | no |
| docker-volumes-uv-cache | technical | no | no | no | no | no | no | no |
| regex-lookbehind-recall | technical | no | no | no | no | no | no | no |
| job-that-changes-its-own-code | fuzzy | yes | yes | yes | yes | yes | yes | yes |
| check-that-cannot-see-its-input | fuzzy | no | no | no | no | no | no | no |
| stop-auto-merging-entities | fuzzy | yes | no | no | no | no | no | yes |
| cheaper-model-every-turn | fuzzy | yes | yes | yes | yes | yes | yes | no |
| nightly-cannot-tell | fuzzy | yes | no | no | no | no | no | yes |
| facts-that-contradict | fuzzy | yes | no | no | no | no | no | no |
| browser-tool-falls-back | fuzzy | yes | yes | yes | yes | yes | yes | yes |
| skill-that-never-improves | fuzzy | yes | no | no | no | no | no | no |
| gpu-ram-thin-should-not-reboot | hard | yes | yes | yes | yes | yes | yes | yes |
| alarm-comes-back-after-fixed | hard | yes | yes | yes | yes | yes | yes | yes |
| numbers-differ-after-rebuild | hard | yes | yes | yes | yes | yes | yes | no |
| wrong-thing-broke-simultaneously | hard | no | no | no | no | no | no | no |
| djev-decision-engine-integration | hard | yes | no | no | no | no | no | yes |
| ambient-prefetch-ttl-reclaim | hard | yes | no | no | no | no | no | yes |
| primary-ram-floor-changed | hard | no | no | no | no | no | no | no |
| prompt-surface-open-set | hard | no | no | no | no | no | no | no |
| isaac-gr00t-n17 | recent | yes | no | no | no | no | no | yes |
| graph-rerank-ab-cache | recent | yes | yes | no | yes | yes | yes | yes |
| retrieval-seed-anchoring-contract | recent | yes | yes | no | yes | yes | yes | yes |
| eval-corpus-naming-conventions | recent | yes | yes | no | yes | yes | yes | yes |
| eval-north-star-candidate | recent | yes | yes | no | yes | yes | yes | yes |
| self-referential-check-catalogue | recent | no | no | no | no | no | no | no |
| eval-artifact-absolute-path | recent | no | no | no | no | no | no | no |

## Cost

- Episodic search latency (scratch store, same models, rerank on): p50 824 ms, p95 961 ms, mean 815.9 ms (baseline recall mean 540.928 ms).

- k=0: 7382.6 tokens per query for 10 episodes (2191.7 for the top 3), 1 turns per episode; the vault-doc control costs 4202.0.
- k=2: 8740.5 tokens per query for 10 episodes (2620.0 for the top 3), 1.68 turns per episode; the vault-doc control costs 5778.6.
- k=5: 8822.4 tokens per query for 10 episodes (2637.0 for the top 3), 1.74 turns per episode; the vault-doc control costs 7340.8.

## Per query

| query | zero-gold | base e_hit | base d_hit | base e_rec | base d_rec | base RR | base nDCG | k=0 e_hit/d_hit/e_rec/d_rec/RR/nDCG | k=2 e_hit/d_hit/e_rec/d_rec/RR/nDCG | k=5 e_hit/d_hit/e_rec/d_rec/RR/nDCG |
|---|---|---|---|---|---|---|---|---|---|---|
| backlog-363 | no | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 1/1/1.000/0.500/0.500/0.621 | 1/1/1.000/0.500/0.500/0.621 | 1/1/1.000/0.500/0.500/0.621 |
| entity-resolution-sweep | no | 0 | 1 | 0.000 | 1.000 | 1.000 | 0.798 | 1/1/1.000/0.500/1.000/0.993 | 1/1/1.000/0.500/1.000/0.993 | 1/1/1.000/0.500/1.000/0.993 |
| inner-voice | no | 1 | 1 | 1.000 | 0.500 | 1.000 | 0.968 | 1/1/1.000/1.000/1.000/0.997 | 1/1/1.000/1.000/1.000/0.997 | 1/1/1.000/1.000/1.000/0.997 |
| vault-recall | no | 0 | 1 | — | 1.000 | 0.111 | 0.301 | 0/1/—/0.500/0.111/0.362 | 0/1/—/0.500/0.111/0.362 | 0/1/—/0.500/0.111/0.362 |
| qmd | no | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1/1/1.000/1.000/1.000/0.962 | 1/1/1.000/1.000/1.000/0.980 | 1/1/1.000/1.000/1.000/0.980 |
| kg-maintenance-tasks | no | 0 | 1 | 0.000 | 0.200 | 0.100 | 0.289 | 1/0/0.400/0.000/0.500/0.651 | 1/0/0.400/0.000/0.500/0.651 | 1/0/0.400/0.000/0.500/0.651 |
| lloyd-vllm-rel | no | 1 | 0 | 1.000 | 0.000 | 0.000 | 0.000 | 1/0/1.000/0.000/1.000/1.000 | 1/0/1.000/0.000/1.000/1.000 | 1/0/1.000/0.000/1.000/1.000 |
| harness-tools | no | 1 | 0 | 0.500 | 0.000 | 0.000 | 0.000 | 1/1/1.000/0.500/1.000/1.000 | 1/1/1.000/0.500/1.000/1.000 | 1/1/1.000/0.500/1.000/1.000 |
| nightly-reflection | no | 1 | 1 | 1.000 | 0.667 | 1.000 | 0.898 | 0/1/0.000/0.333/1.000/0.815 | 0/1/0.000/0.333/1.000/0.815 | 0/1/0.000/0.333/1.000/0.815 |
| qwen38-local-serving | no | 0 | 1 | 0.000 | 0.667 | 1.000 | 0.790 | 1/1/1.000/1.000/1.000/0.984 | 1/1/1.000/1.000/1.000/0.984 | 1/1/1.000/1.000/1.000/0.984 |
| tgs-rag-state | no | 1 | 1 | 1.000 | 1.000 | 0.500 | 0.698 | 1/1/1.000/0.667/1.000/0.958 | 1/1/1.000/0.667/1.000/0.958 | 1/1/1.000/0.667/1.000/0.958 |
| graph-quality | no | 0 | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 1/0/0.500/0.000/1.000/0.832 | 1/0/0.500/0.000/1.000/0.832 | 1/0/0.500/0.000/1.000/0.832 |
| classifier-v4 | no | 1 | 1 | 0.500 | 0.500 | 0.250 | 0.431 | 0/1/0.000/0.500/0.333/0.500 | 0/1/0.000/0.500/0.333/0.500 | 0/1/0.000/0.500/0.333/0.500 |
| godnode-threshold | no | 0 | 1 | 0.000 | 0.500 | 0.053 | 0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/1/0.000/0.500/0.111/0.301 | 0/1/0.000/0.500/0.111/0.301 |
| relationships-location | no | 0 | 1 | 0.000 | 0.500 | 0.200 | 0.387 | 1/1/1.000/0.500/0.500/0.693 | 1/1/1.000/0.500/0.500/0.693 | 1/1/1.000/0.500/0.500/0.693 |
| memory-persistence | no | 0 | 1 | 0.000 | 0.500 | 0.333 | 0.500 | 1/0/0.333/0.000/0.250/0.458 | 1/0/0.333/0.000/0.250/0.458 | 1/0/0.333/0.000/0.250/0.458 |
| autonomy-pipeline | no | 1 | 1 | 0.667 | 0.500 | 1.000 | 1.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 |
| robotics-projects | no | 0 | 1 | 0.000 | 1.000 | 0.200 | 0.600 | 1/1/0.200/1.000/1.000/1.000 | 1/1/0.200/1.000/1.000/1.000 | 1/1/0.200/1.000/1.000/1.000 |
| this-week-autonomy | no | 1 | 1 | 0.667 | 1.000 | 0.062 | 0.000 | 1/1/0.333/1.000/0.500/0.564 | 1/1/0.333/1.000/0.500/0.564 | 1/1/0.333/1.000/0.500/0.564 |
| backlog-overview | no | 1 | 1 | 1.000 | 1.000 | 0.111 | 0.362 | 1/1/1.000/1.000/0.500/0.833 | 1/1/1.000/1.000/0.500/0.833 | 1/1/1.000/1.000/0.500/0.833 |
| tts-voice-cloning | no | 0 | 1 | 0.000 | 1.000 | 0.333 | 0.500 | 1/0/1.000/0.000/1.000/0.967 | 1/0/1.000/0.000/1.000/0.967 | 1/0/1.000/0.000/1.000/0.967 |
| wake-word-models | no | 0 | 1 | 0.000 | 1.000 | 0.062 | 0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 |
| thunderbird-mcp-toolset | no | 1 | 1 | 1.000 | 1.000 | 0.167 | 0.356 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 |
| grafana-monitoring-stack | yes | 1 | 0 | 1.000 | — | 0.000 | 0.000 | 0/0/0.000/—/0.000/0.000 | 0/0/0.000/—/0.000/0.000 | 0/0/0.000/—/0.000/0.000 |
| gpu-model-naming-collision | no | 0 | 1 | 0.000 | 1.000 | 0.125 | 0.316 | 1/0/1.000/0.000/1.000/0.997 | 1/0/1.000/0.000/1.000/0.997 | 1/0/1.000/0.000/1.000/0.997 |
| browser-tool-validation | no | 0 | 1 | 0.000 | 0.500 | 1.000 | 1.000 | 1/1/1.000/1.000/0.500/0.634 | 1/1/1.000/1.000/0.500/0.634 | 1/1/1.000/1.000/0.500/0.634 |
| three-d-printing-calibration | no | 0 | 0 | 0.000 | — | 0.000 | 0.000 | 0/0/0.000/—/0.000/0.000 | 0/0/0.000/—/0.000/0.000 | 0/0/0.000/—/0.000/0.000 |
| youtube-transcript-workflow | no | 1 | 1 | 1.000 | 1.000 | 0.143 | 0.398 | 1/0/1.000/0.000/0.500/0.695 | 1/0/1.000/0.000/0.500/0.695 | 1/0/1.000/0.000/0.500/0.695 |
| mcp-transport-error-recovery | no | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1/0/1.000/0.000/0.500/0.776 | 1/0/1.000/0.000/0.500/0.776 | 1/0/1.000/0.000/0.500/0.776 |
| config-yaml-readonly | no | 0 | 0 | — | 0.000 | 0.000 | 0.000 | 0/1/—/0.500/0.143/0.333 | 0/1/—/0.500/0.143/0.333 | 0/1/—/0.500/0.143/0.333 |
| model-alias-resolution | no | 1 | 0 | 1.000 | 0.000 | 0.000 | 0.000 | 1/0/0.500/0.000/0.250/0.526 | 1/0/0.500/0.000/0.250/0.526 | 1/0/0.500/0.000/0.250/0.526 |
| groundskeeper-queue-corruption | no | 1 | 1 | 1.000 | 1.000 | 0.143 | 0.333 | 1/0/1.000/0.000/1.000/0.785 | 1/0/1.000/0.000/1.000/0.785 | 1/0/1.000/0.000/1.000/0.785 |
| kg-rebuild-abandoned | no | 1 | 0 | 1.000 | 0.000 | 0.000 | 0.000 | 1/0/1.000/0.000/0.143/0.440 | 1/0/1.000/0.000/0.143/0.440 | 1/0/1.000/0.000/0.143/0.440 |
| kg-dedup-key | no | 0 | 1 | 0.000 | 1.000 | 0.053 | 0.000 | 1/0/1.000/0.000/0.500/0.631 | 1/0/1.000/0.000/0.500/0.631 | 1/0/1.000/0.000/0.500/0.631 |
| guardian-to-backlog | no | 1 | 1 | 1.000 | 1.000 | 0.333 | 0.500 | 1/0/1.000/0.000/0.500/0.787 | 1/0/1.000/0.000/1.000/0.967 | 1/0/1.000/0.000/1.000/1.000 |
| dream-to-skill-edit | no | 0 | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 1/1/1.000/0.500/0.250/0.479 | 1/1/1.000/0.500/0.250/0.479 | 1/1/1.000/0.500/0.250/0.479 |
| automod-to-entity-guard | no | 0 | 1 | 0.000 | 0.333 | 0.250 | 0.431 | 1/0/0.500/0.000/0.500/0.666 | 1/0/0.500/0.000/0.500/0.666 | 1/0/0.500/0.000/0.500/0.666 |
| memory-capture-to-kg | no | 1 | 0 | 0.500 | 0.000 | 0.000 | 0.000 | 1/0/0.500/0.000/0.125/0.316 | 1/0/0.500/0.000/0.200/0.431 | 1/0/0.500/0.000/0.200/0.431 |
| research-queue-to-vault-note | no | 1 | 1 | 1.000 | 0.667 | 1.000 | 0.798 | 1/0/1.000/0.000/1.000/0.777 | 1/0/1.000/0.000/1.000/0.777 | 1/0/1.000/0.000/1.000/0.777 |
| counterfactual-to-trend-audit | no | 0 | 1 | 0.000 | 1.000 | 0.333 | 0.500 | 1/0/1.000/0.000/0.167/0.356 | 1/0/1.000/0.000/0.167/0.356 | 1/0/1.000/0.000/0.167/0.356 |
| autonomy-task-to-skill | no | 0 | 1 | 0.000 | 0.333 | 0.143 | 0.333 | 1/1/0.500/0.333/0.143/0.389 | 1/1/0.500/0.333/0.143/0.389 | 1/1/0.500/0.333/0.143/0.389 |
| inner-voice-to-surface | no | 1 | 1 | 0.500 | 1.000 | 1.000 | 1.000 | 1/1/0.500/1.000/0.500/0.833 | 1/1/0.500/1.000/0.500/0.833 | 1/1/0.500/1.000/0.500/0.833 |
| entity-guard-to-alias-table | no | 1 | 1 | 1.000 | 0.333 | 0.111 | 0.301 | 1/0/1.000/0.000/0.333/0.598 | 1/0/1.000/0.000/0.333/0.598 | 1/0/1.000/0.000/0.333/0.598 |
| email-pipeline-to-daily-note | no | 1 | 1 | 1.000 | 1.000 | 0.333 | 0.571 | 0/1/0.000/0.500/1.000/0.807 | 0/1/0.000/0.500/1.000/0.807 | 0/1/0.000/0.500/1.000/0.807 |
| skill-mining-to-promotion | no | 1 | 1 | 0.500 | 1.000 | 1.000 | 0.920 | 1/0/0.500/0.000/1.000/1.000 | 1/0/0.500/0.000/1.000/0.850 | 1/0/0.500/0.000/1.000/0.850 |
| guardian-alert-to-retraction | no | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1/0/1.000/0.000/1.000/0.889 | 1/0/1.000/0.000/1.000/0.914 | 1/0/1.000/0.000/1.000/0.914 |
| voice-session-to-room | no | 0 | 1 | 0.000 | 0.500 | 1.000 | 1.000 | 1/1/0.500/1.000/1.000/0.807 | 1/1/0.500/1.000/1.000/0.899 | 1/1/0.500/1.000/1.000/0.899 |
| memory-md-clobber-to-guard | no | 1 | 1 | 1.000 | 1.000 | 0.333 | 0.500 | 1/0/1.000/0.000/0.500/0.715 | 1/0/1.000/0.000/0.500/0.715 | 1/0/1.000/0.000/0.500/0.715 |
| watchdog-metacharacter | yes | 0 | 1 | — | 1.000 | 1.000 | 1.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| yaml-scalar-block-indent | yes | 0 | 0 | — | 0.000 | 0.000 | 0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| json-filter-epoch-vs-iso | no | 0 | 1 | — | 1.000 | 0.333 | 0.500 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| bash-arrays-vs-strings | yes | 0 | 0 | — | — | 0.000 | 0.000 | 0/0/—/—/0.000/0.000 | 0/0/—/—/0.000/0.000 | 0/0/—/—/0.000/0.000 |
| git-signals-in-async-code | yes | 0 | 1 | — | 1.000 | 0.125 | 0.316 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| docker-volumes-uv-cache | yes | 0 | 0 | — | 0.000 | 0.000 | 0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| regex-lookbehind-recall | yes | 0 | 0 | — | — | 0.000 | 0.000 | 0/0/—/—/0.000/0.000 | 0/0/—/—/0.000/0.000 | 0/0/—/—/0.000/0.000 |
| gpu-daemon-ipc-timeout | no | 1 | 1 | 1.000 | 1.000 | 0.200 | 0.387 | 1/0/1.000/0.000/1.000/0.967 | 1/0/1.000/0.000/1.000/0.967 | 1/0/1.000/0.000/1.000/0.967 |
| job-that-changes-its-own-code | no | 0 | 1 | 0.000 | 1.000 | 0.333 | 0.500 | 1/0/1.000/0.000/0.333/0.615 | 1/0/1.000/0.000/0.333/0.615 | 1/0/1.000/0.000/0.333/0.615 |
| check-that-cannot-see-its-input | no | 0 | 0 | — | 0.000 | 0.000 | 0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| stop-auto-merging-entities | no | 0 | 1 | 0.000 | 1.000 | 0.143 | 0.333 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 |
| cheaper-model-every-turn | no | 0 | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 1/0/1.000/0.000/0.125/0.316 | 1/1/1.000/0.500/0.125/0.316 | 1/1/1.000/0.500/0.125/0.316 |
| write-that-ate-the-memory-file | no | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1/0/1.000/0.000/1.000/0.944 | 1/0/1.000/0.000/1.000/0.927 | 1/0/1.000/0.000/1.000/0.927 |
| nightly-cannot-tell | no | 0 | 1 | 0.000 | 1.000 | 0.056 | 0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 |
| facts-that-contradict | no | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0/1/0.000/0.500/0.333/0.500 | 0/1/0.000/0.500/0.333/0.500 | 0/1/0.000/0.500/0.333/0.500 |
| browser-tool-falls-back | no | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 1/1/1.000/1.000/1.000/1.000 | 1/1/1.000/1.000/1.000/0.949 | 1/1/1.000/1.000/1.000/0.949 |
| skill-that-never-improves | no | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 |
| gpu-ram-thin-should-not-reboot | no | 0 | 0 | 0.000 | — | 0.000 | 0.000 | 1/0/1.000/—/0.500/0.631 | 1/0/1.000/—/0.500/0.631 | 1/0/1.000/—/0.500/0.631 |
| alarm-comes-back-after-fixed | no | 0 | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 1/0/1.000/0.000/0.143/0.333 | 1/0/1.000/0.000/0.143/0.398 | 1/0/1.000/0.000/0.143/0.398 |
| numbers-differ-after-rebuild | no | 0 | 1 | 0.000 | 0.667 | 0.167 | 0.356 | 1/1/1.000/0.333/0.250/0.562 | 1/1/1.000/0.333/0.250/0.562 | 1/1/1.000/0.333/0.250/0.562 |
| memory-line-contradicts-vault | no | 1 | 1 | 1.000 | 0.500 | 0.071 | 0.000 | 1/0/1.000/0.000/0.500/0.710 | 1/0/1.000/0.000/0.500/0.724 | 1/0/1.000/0.000/0.500/0.724 |
| wrong-thing-broke-simultaneously | no | 0 | 0 | — | 0.000 | 0.000 | 0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| djev-decision-engine-integration | no | 0 | 0 | 0.000 | — | 0.000 | 0.000 | 0/0/0.000/—/0.000/0.000 | 0/0/0.000/—/0.000/0.000 | 0/0/0.000/—/0.000/0.000 |
| ambient-prefetch-ttl-reclaim | no | 0 | 1 | 0.000 | 1.000 | 1.000 | 0.807 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 |
| primary-ram-floor-changed | yes | 0 | 1 | — | 1.000 | 0.250 | 0.431 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| prompt-surface-open-set | no | 0 | 0 | — | 0.000 | 0.000 | 0.000 | 0/1/—/0.500/1.000/1.000 | 0/1/—/0.500/1.000/1.000 | 0/1/—/0.500/1.000/1.000 |
| isaac-gr00t-n17 | no | 0 | 1 | 0.000 | 1.000 | 1.000 | 0.852 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 | 0/0/0.000/0.000/0.000/0.000 |
| graph-rerank-ab-cache | no | 0 | 1 | 0.000 | 0.500 | 0.111 | 0.301 | 1/0/1.000/0.000/0.125/0.371 | 1/0/1.000/0.000/0.125/0.371 | 1/0/1.000/0.000/0.125/0.371 |
| retrieval-seed-anchoring-contract | no | 0 | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 1/0/1.000/0.000/0.500/0.631 | 1/0/1.000/0.000/0.500/0.631 | 1/0/1.000/0.000/0.500/0.631 |
| eval-corpus-naming-conventions | no | 0 | 1 | 0.000 | 1.000 | 0.167 | 0.423 | 1/1/1.000/0.500/0.167/0.396 | 1/1/1.000/0.500/0.167/0.459 | 1/1/1.000/0.500/0.167/0.459 |
| eval-north-star-candidate | no | 0 | 0 | 0.000 | — | 0.000 | 0.000 | 1/0/1.000/—/0.500/0.631 | 1/0/1.000/—/0.500/0.631 | 1/0/1.000/—/0.500/0.631 |
| self-referential-check-catalogue | no | 0 | 0 | — | 0.000 | 0.000 | 0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
| eval-artifact-absolute-path | yes | 0 | 0 | — | 0.000 | 0.000 | 0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 | 0/0/—/0.000/0.000/0.000 |
