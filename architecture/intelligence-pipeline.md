---
segment: architecture
type: reference
tags: [architecture]
date: 2026-03-22
status: planned
related:
  - projects/lloyd/architecture/index.md
  - projects/lloyd/architecture/autonomy-system.md
  - projects/lloyd/architecture/memory.md
  - projects/lloyd/architecture/agents.md

---

# Intelligence Pipeline Architecture

**Created:** 2026-03-22
**Backlog:** #178
**Status:** Planned (Phase 0 partially exists via GitHub Watch)

## Overview

The intelligence pipeline is a proactive internet monitoring and synthesis system that tracks developments across Alan's interest domains,scores relevance against active projects,and delivers actionable intelligence through graduated urgency channels.

**Design principle:** Scan broadly,filter aggressively,synthesize concisely. All filtering and scoring runs on local 122B — zero API cost for high-volume work.

**Execution model:** Leverages the autonomy system's GPU-gated idler and scheduled researcher tasks. Scanners run throughout the day whenever GPU is idle — not batched overnight. Scoring happens inline immediately after scanning (same task),so items are classified within minutes of discovery. Only the weekly synthesis is a fixed-schedule batch job.

## Architecture Diagram

```
                    ┌──────────────────────────────┐
                    │       Source Scanners         │
                    │  (autonomy researcher tasks)  │
                    ├──────────────────────────────┤
                    │ arXiv │ HN │ Reddit │ GitHub  │
                    │ RSS   │ YT │ Releases         │
                    └────────────┬─────────────────┘
                                 │ raw items
                                 ▼
                    ┌──────────────────────────────┐
                    │     Dedup & State Filter      │
                    │   memory/feeds/state files    │
                    └────────────┬─────────────────┘
                                 │ new items only
                                 ▼
                    ┌──────────────────────────────┐
                    │    Relevance Scoring (122B)   │
                    │  interest profile × item      │
                    │  → score + "why it matters"   │
                    └────────────┬─────────────────┘
                                 │ scored items
                                 ▼
                    ┌──────────────────────────────┐
                    │     Urgency Classification    │
                    │  🔴 interrupt (score ≥ 90)    │
                    │  🟡 briefing  (score ≥ 50)    │
                    │  🟢 weekly    (score ≥ 30)    │
                    │  ⚫ drop      (score < 30)    │
                    └────────┬───────┬───────┬─────┘
                             │       │       │
                    ┌────────┘       │       └────────┐
                    ▼                ▼                ▼
              ┌──────────┐   ┌────────────┐   ┌───────────┐
              │ Immediate│   │  Morning   │   │  Weekly   │
              │  Notify  │   │  Briefing  │   │ Synthesis │
              └──────────┘   └────────────┘   └───────────┘

              ┌──────────────────────────────────────────┐
              │          Engagement Tracking              │
              │  (Phase 4: which items Alan acts on)     │
              │  → tune interest profile over time       │
              └──────────────────────────────────────────┘
```

---

## Phase 1: Scanner Infrastructure

**Goal:** Build individual source scanners as autonomy tasks,each producing raw items in a common format.

### Common Item Format

Every scanner outputs items in a normalized structure before scoring:

```json
{
  "source": "arxiv|hn|reddit|github|rss|youtube|release",
  "id": "unique-item-id",
  "title": "Item title",
  "url": "https://...",
  "summary": "1-2 sentence description (from source or extracted)",
  "authors": ["name"],
  "tags": ["robotics","llm"],
  "published": "ISO-8601",
  "discovered": "ISO-8601",
  "raw_metadata": {}
}
```

### Scanner: arXiv

| Property | Value |
|----------|-------|
| **API** | `http://export.arxiv.org/api/query` (Atom feed,no auth) |
| **Categories** | `cs.RO`,`cs.AI`,`cs.CL`,`cs.LG`,`cs.CV` |
| **Keywords** | `humanoid`,`locomotion`,`sim-to-real`,`imitation learning`,`foundation model`,`embodied`,`quadruped`,`voice synthesis`,`text-to-speech`,`language model`,`quantization`,`agent`,`tool use` |
| **Frequency** | 1x/day (arXiv updates daily ~6pm ET) |
| **Rate limit** | 3-second delay between requests (arXiv policy) |
| **State** | `memory/feeds/arxiv-state.json` — last query date + seen paper IDs |
| **Volume** | ~10-30 papers/day after keyword filter |

**Query strategy:** Category filter + keyword search in title/abstract. Fetch last 48h to handle weekends. Deduplicate by arXiv ID.

```
GET http://export.arxiv.org/api/query?search_query=(cat:cs.RO+OR+cat:cs.AI)+AND+(ti:humanoid+OR+ti:locomotion+OR+ti:embodied)&sortBy=submittedDate&sortOrder=descendin