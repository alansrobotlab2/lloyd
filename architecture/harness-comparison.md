---
segment: architecture
type: notes
tags: [anthropic,architecture,harness-comparison,our-system]

---

# Harness Architecture Comparison: Anthropic vs. Our System

## Executive Summary

Anthropic's GAN-inspired harness (planner→generator→evaluator with sprint contracts) achieves dramatic quality improvements over solo agent runs. Our current pipeline (orchestrator→planner→coder→reviewer→tester) has structural similarities but misses key innovations: evaluator separation,sprint contracts,and live app testing.

## Architecture Mapping

### Anthropic's Three-Agent System

```
┌─────────┐     ┌──────────┐     ┌──────────┐
│ Planner │────▶│ Generator│────▶│ Evaluator│
└─────────┘     └──────────┘     └──────────┘
                    ▲                  │
                    └──────┬───────────┘
                           │
                    Sprint Contracts
                    (negotiated before each sprint)
```

**Key characteristics:**
- Generator and Evaluator are **separate agents**
- Evaluator uses **Playwright MCP** for live app testing
- Sprint contracts negotiated **before** coding begins
- Context resets with structured handoff for long tasks

### Our Current Pipeline

```
┌────────────┐     ┌───────┐     ┌───────┐     ┌────────┐     ┌─────────┐
│Orchestrator│────▶│Planner│────▶│ Coder │────▶│Reviewer│────▶│  Tester │
└────────────┘     └───────┘     └───────┘     └────────┘     └─────────┘
```

**Key characteristics:**
- Orchestrator coordinates multi-agent workflows
- Reviewer does code review (static analysis)
- Tester runs test suites
- Wiggam nudge loop for context management (60s →5s poll,TASK_COMPLETE signal)

## Detailed Comparison

### 1. Agent Separation

| Aspect | Anthropic | Our System | Gap |
|--------|-----------|------------|-----|
| Build vs Critique | **Separate agents** (Generator,Evaluator) | **Reviewer** after Coder | ⚠️ Reviewer may have self-evaluation bias |
| Evaluator Tools | Playwright MCP (live app) | Code review,static analysis | ⚠️ No live application testing |
| Calibration | Explicit criteria,weighted scoring | 22% bug catch rate (memory/2026-03-07.md) | ⚠️ Calibration opportunity |

**Impact:** Anthropic's separation makes it tractable to tune skepticism independently from generation quality. Our reviewer effectiveness (22% bug catch rate) suggests calibration opportunity.

### 2. Sprint Contracts

| Aspect | Anthropic | Our System | Gap |
|--------|-----------|------------|-----|
| Pre-coding negotiation | ✅ Yes (done criteria before each sprint) | ❌ No | ⚠️ Scope ambiguity risk |
| Acceptance criteria | Concrete,testable thresholds | Implicit in requirements | ⚠️ May be ambiguous |
| Iteration structure | Sprint-based with clear boundaries | Continuous/linear | ⚠️ Less structured |

**Impact:** Sprint contracts prevent scope creep and ensure alignment before coding. Our system lacks this layer.

### 3. Context Management

| Aspect | Anthropic | Our System | Gap |
|--------|-----------|------------|-----|
| Strategy | Context resets (fresh agent + handoff) | Wiggam nudge loop (60s→5s poll) | ⚠️ Different approach |
| "Context anxiety" | Sonnet 4.5 had it,Opus 4.6 reduced it | Unknown | ? |
| Handoff artifact | Structured (state,next steps,questions) | TASK_COMPLETE signal | ⚠️ Less structured |

**Impact:** Both approaches manage long-running tasks but differently. Anthropic found resets > compaction. Our wiggam loop is novel but untested against their findings.

### 4. Evaluation Criteria

| Aspect | Anthropic | Our System | Gap |
|--------|-----------|------------|-----|
| Criteria | Design quality,Originality,Craft,Functionality | Code correctness,tests | ⚠️ Narrower scope |
| Weighting | Heavier on design/originality | Not specified | ⚠️ May undervalue quality |
| Scoring | Weighted,calibrated with few-shot | Pass/fail,bug count | ⚠️ Less nuanced |

**Impact:** Anthropic evaluates holistically (design,craft,originality). Our review focuses on correctness/tests.

### 5. Testing Approach

| Aspect | Anthropic | Our System | Gap |
|--------|-----------|------------|-----|
| Testing method | Playwright MCP (live app interaction) | Unit/integration tests | ⚠️ Static vs dynamic |
| Coverage | User-facing functionality | Code coverage | ⚠️ Different focus |
| Automation | Full browser automation | Test suite execution | ⚠️ Less comprehensive |

**Impact:** Anthropic's evaluator can test real user interactions. Our tester runs predefined tests.

### 6. Cost/Duration Tradeoffs

| Configuration | Anthropic | Our System |
|---------------|-----------|------------|
| Solo run | $9,20min (broken app) | Unknown |
| Full harness | $200,6hr (16 features) | Unknown |
| Simplified | $124,4hr (16 features) | Unknown |

**Impact:** Anthropic's harness costs 10-20x more than solo but produces working software. Our cost structure unknown.

## Structural Gaps Identified

### High Priority Gaps

1. **Evaluator Separation** ⚠️⚠️⚠️
   - **Issue:** Our reviewer may hav