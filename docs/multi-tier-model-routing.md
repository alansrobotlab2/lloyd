# Multi-Tier Model Routing

**Status:** Implemented — active in production since 2026-02-27
**Last updated:** 2026-02-28

---

## Motivation

Without routing, every prompt goes to `claude-sonnet-4-6` regardless of complexity.
This is:

- **Slower than necessary** for simple tasks (~1.5–8s when local Qwen answers in <300ms)
- **Costly** when simpler/cheaper models suffice
- **Inflexible** — no way to dial up power for genuinely hard problems

The model-router plugin (`extensions/model-router/`) selects the right model
automatically via the `before_agent_start` hook, with manual override always
available.

---

## Model Tiers

Active routing uses 3 tiers: **local → sonnet → opus**. Haiku is not in the routing
path (not eval-calibrated); it exists only as a TIERS entry for possible future use.

| Tier | Model | Provider | Context | Cost | Use for |
|------|-------|----------|---------|------|---------|
| **local** | `Qwen3.5-35B-A3B` | `local-llm` (port 8091) | 32k | Free | Simple Q&A, quick lookups, tool dispatch |
| **sonnet** | `claude-sonnet-4-6` | `anthropic` (native) | 200k | Medium | Code gen, analysis, multi-step tasks — default tier |
| **opus** | `claude-opus-4-6` | `anthropic` (native) | 200k | High | Architecture decisions, deep research, complex reasoning |

The local model endpoint is the OpenAI-compatible vLLM server at
`http://127.0.0.1:8091/v1`. Sonnet and Opus use the native Anthropic provider
via the `anthropic:manual` auth profile. No explicit `"anthropic"` provider block
is needed in `openclaw.json` — the router references model/provider IDs directly
and the native auth profile handles API access.

---

## Evaluation Results (2026-02-27)

Full results: [eval-local-vs-sonnet.md](eval-local-vs-sonnet.md)

### Summary: Local Qwen vs Sonnet 4.6

| Metric | Sonnet 4.6 | Local Qwen |
|--------|-----------|------------|
| Avg latency | 7,788ms | 2,587ms |
| Avg quality (Opus judge) | 4.64/5 | 4.12/5 |
| Tool selection accuracy | 84% | **100%** |
| Wins / Ties / Losses | 12 / 9 / 4 | 4 / 9 / 12 |

### Per-category breakdown

| Category | Sonnet score | Local score | Local latency | Assessment |
|----------|-------------|------------|--------------|------------|
| instruction-following | 5.0/5 | 3.4/5 | 263ms | **Route to Sonnet** — local fails word counting, character ops |
| tool-selection | 4.2/5 | **4.8/5** | 1,066ms | **Route to local** — local actually outperforms |
| code | 4.8/5 | 4.2/5 | 2,803ms | Route to Sonnet for complex generation; local ok for explain |
| multi-step-reasoning | 4.6/5 | 4.0/5 | 6,482ms | Route to Sonnet; quality gap noticeable at hard problems |
| edge-cases | 4.6/5 | 4.2/5 | 2,321ms | Mixed — local good at refusals, weak at numeric edge cases |

### Key findings

1. **Local Qwen is 3× faster** across all categories (7.8s → 2.6s avg).
2. **Tool selection is local's strongest suit** — it outscores Sonnet (4.8 vs 4.2).
   Ideal for the common case of "dispatch a tool call and summarize the result."
3. **Strict instruction-following is local's Achilles heel** — exact word counts,
   character-level transformations, precise output constraints → use Sonnet.
4. **Quality gap is small on most tasks** (4.12 vs 4.64 overall), but the gap widens
   on complex code generation and hard reasoning.
5. **Local Qwen is not a drop-in Sonnet replacement** as primary orchestrator today,
   but is appropriate for a significant portion of real workloads.

### Routing thresholds informed by eval

| Signal | Route to | Rationale |
|--------|----------|-----------|
| Tool dispatch + summarize | **local** | Local scores 4.8/5, 3× faster |
| Explain code (not write) | **local** | Comparable quality (code-03: local won) |
| Factual Q&A, short prompt | **local** | Fast + adequate |
| Precise constraints (word count, format) | **sonnet** | Local fails at 3.4/5 |
| Write/generate code | **sonnet** | Gap widens on complex generation |
| Architecture, research, deep analysis | **opus** | Explicit escalation only |

---

## Routing Strategy: 4-Stage Hybrid

Routing happens via the `before_agent_start` plugin hook, which fires before the
first LLM call and has access to session message history. It returns
`{ modelOverride, providerOverride }`.

### Stage 1 — Explicit overrides (zero latency)

Check the prompt for unambiguous routing commands:

| Signal | Route |
|--------|-------|
| Prompt starts with `/fast` or `/local` | **local** |
| Prompt starts with `/deep` or `/opus` | **opus** |
| Prompt starts with `/sonnet` | **sonnet** |

### Stage 2 — Feature-based scoring (zero latency)

Extract features from the prompt and session context, then score each tier. The
tier with the highest score wins if confidence meets the threshold.

**Prompt length signals:**

| Condition | Effect |
|-----------|--------|
| < 80 chars, no code blocks | local +3 |
| < 150 chars, no code blocks | local +2 |
| < 300 chars | local +1 |
| > 600 chars | sonnet +1 |
| > 1200 chars | opus +1 |

**Question type:**

| Pattern | Effect |
|---------|--------|
| Factual: *what is*, *who is*, *define*, *how many*, *yes or no*, etc. | local +3 |
| Reasoning: *why does*, *how can*, *explain why/how*, etc. | sonnet +2 |

**Intent signals:**

| Pattern | Effect |
|---------|--------|
| Tool dispatch (*search for*, *look up*, *find*, *read file*, *show me*) without code intent | local +4 |
| Code intent (*implement*, *write function/class*, *debug*, *refactor*, *fix code/bug*) | sonnet +4 |
| Code blocks present | sonnet +3 |
| Output constraints (*exactly N words*, *character by character*, *numbered list of exactly*) | sonnet +3 |
| Deep analysis (*architecture*, *design document*, *tradeoffs*, *compare and contrast*) | opus +4 |
| Summarize/translate (without code intent) | local +2 |

**Session context signals:**

| Condition | Effect |
|-----------|--------|
| Prior code tool usage (file/bash/edit/write/grep/glob) + short follow-up (< 150 chars) | sonnet +3 |
| Prior Opus tier + short follow-up (< 200 chars) | opus +3 |
| Session depth > 10 messages | sonnet +1 |

**Confidence formula:**

```
confidence = (topScore - runnerUpScore) / (topScore + 1)
```

If `confidence >= 0.7` and `topScore > 0`, use the feature-scored tier. Otherwise
fall through to Stage 3.

### Stage 3 — LLM classifier (for ambiguous cases)

When feature scoring is inconclusive, call local Qwen with a classification prompt.
Timeout: 1,500ms. On timeout/error, fall through to Stage 4.

```
System: You are a task router. Given a user message, respond with exactly one word —
the complexity tier:
- "local" — simple factual Q&A, greetings, quick lookups, tool dispatch, summarize,
  translate, reformat, short explanations
- "sonnet" — write/debug/refactor code, multi-step reasoning, precise instruction
  following
- "opus" — architecture design, deep research, complex analysis, tradeoff evaluation

Respond with ONLY the tier name, nothing else.
```

The prompt is truncated to 2,000 chars. Temperature: 0, max_tokens: 10.
Classifier confidence is set to 0.6 (lower than feature scoring).

### Stage 4 — Default fallback

If the classifier times out or returns an unrecognized response, route to the
configured default tier (**sonnet**).

---

## Implementation

### Plugin structure

```
extensions/model-router/
├── openclaw.plugin.json    # plugin metadata
├── config.json             # runtime config
└── index.ts                # 4-stage routing logic (before_agent_start hook)
```

### Model/provider mapping

```typescript
const TIERS = {
  local:  { modelOverride: "Qwen3.5-35B-A3B",  providerOverride: "local-llm" },
  haiku:  { modelOverride: "claude-haiku-4-5",  providerOverride: "anthropic"  },
  sonnet: { modelOverride: "claude-sonnet-4-6", providerOverride: "anthropic"  },
  opus:   { modelOverride: "claude-opus-4-6",   providerOverride: "anthropic"  },
};
```

Haiku is defined in the TIERS map but never selected by scoring or classifier —
only reachable if a future feature enables it.

### Configuration (`config.json`)

```json
{
  "enabled": true,
  "defaultTier": "sonnet",
  "classifierEnabled": true,
  "classifierTimeoutMs": 1500,
  "confidenceThreshold": 0.7,
  "verbose": true
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Master switch — disable for graceful no-op |
| `defaultTier` | string | `"sonnet"` | Fallback when all stages are inconclusive |
| `classifierEnabled` | boolean | `true` | Toggle LLM classifier (Stage 3) |
| `classifierTimeoutMs` | number | `1500` | Qwen classifier timeout |
| `confidenceThreshold` | number | `0.7` | Minimum confidence for feature scoring |
| `verbose` | boolean | `true` | Log non-default routing decisions at info level |

### Observability

Every routing decision is appended to `logs/routing.jsonl`:

```json
{
  "ts": "2026-02-28T02:13:08.065Z",
  "tier": "local",
  "reason": "feature-scored (local=3, confidence=0.75)",
  "confidence": 0.75,
  "classifierUsed": false,
  "latencyMs": 2,
  "promptLength": 78,
  "sessionDepth": 0
}
```

Typical routing overhead: 0–2ms for feature scoring, 100–400ms when classifier fires.
Logging failures are silent (non-fatal).

---

## Verification

1. **Check routing logs**: `tail -20 logs/routing.jsonl` — confirm decisions are
   being logged with expected tiers and reasons
2. **Test explicit overrides**: `/fast what is 2+2` → local; `/opus analyze this` → opus
3. **Test feature scoring**: short factual question → local; paste code block → sonnet
4. **Test classifier**: medium-length ambiguous prompt → check logs for `classifierUsed: true`
5. **Test default**: disable classifier in config, send ambiguous prompt → sonnet
6. **Re-run eval harness**:
   ```bash
   npx tsx tools/model-eval.ts --no-judge --category tool-selection
   ```

---

## Future Work

- **Haiku tier**: Run the eval harness against Haiku to calibrate its quality/latency
  profile. Once calibrated, enable Haiku in the scoring and classifier stages for
  moderate tasks (summarize, translate, reformat).
- **Re-run eval after model updates**: Local model version may improve (newer GGUF).
  Bump the eval suite if so.
- **User feedback loop**: Analyze routing decisions + session outcomes in
  `logs/routing.jsonl` to find which routes lead to re-prompts or corrections
  (implicit quality signal).
- **Cost tracking**: Add cost metadata to `logs/timing.jsonl` (local << Sonnet << Opus)
  to quantify savings over time.
- **Timing-profiler integration**: Extend `timing-profiler` to log the resolved model
  ID in `logs/timing.jsonl` for per-model latency auditing.
