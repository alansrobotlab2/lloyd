---
segment: architecture
tags: [architecture]
type: reference
relations:
  related-to:
  - architecture/index.md
  - architecture/autonomy-system.md
  - architecture/memory.md
  - architecture/nightly-vault-maintenance.md

---

# Groundskeeper System Architecture

No changes needed.

**Created:** 2026-03-25  
**Status:** Active

## Overview

The Groundskeeper is a vault health and enrichment system that continuously scans the Obsidian knowledge vault for issues and enrichment opportunities. It evolved from the "Ralph Wiggum" survey scanner and absorbed the Vault Maintenance task (#32) into a unified two-loop architecture:

1. **Groundskeeper Fix Loop** — Automated mechanical repairs (autonomy task #33)
2. **Groundskeeper Research** — Deep web research for enrichment (autonomy task #34)

Both loops share a single scanner script and queue file,with the autonomy system dispatching them on different schedules.

## Architecture Diagram

```
                    +-------------------+
                    |  Groundskeeper    |
                    |  Survey Script    |
                    |  (groundskeeper-  |
                    |   survey.py)      |
                    +---------+---------+
                              |
                              v
                    +---------+---------+
                    |  Queue JSON File  |
                    |  (groundskeeper-  |
                    |   queue.json)     |
                    +---------+---------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
+-------------+-------------+     +-----------+-----------+
|   Fix Loop                |     |   Research Loop       |
|  (Task #33)               |     |  (Task #34)           |
|  Every 15 minutes         |     |  Hourly               |
|  Memory agent             |     |  Researcher agent     |
|  5 items per run          |     |  1 item per run       |
+-------------+-------------+     +-----------+-----------+
              |                               |
              v                               v
+-------------+-------------+     +-----------+-----------+
|  Audit Log                |     |  Audit Log            |
|  groundskeeper-log.jsonl  |     |  groundskeeper-       |
|                           |     |  research-log.jsonl   |
+---------------------------+     +-----------------------+

              |                               |
              +---------------+---------------+
                              |
                              v
                    +---------+---------+
                    | Weekly Summary    |
                    | (groundskeeper-   |
                    |  weekly-summary.py)|
                    +-------------------+
```

## Scanner (`groundskeeper-survey.py`)

The scanner performs a comprehensive vault scan across 11 categories grouped into 3 functional areas:

### Fix Categories (Handled by Fix Loop)

| Category | Description |
|----------|-------------|
| BROKEN_LINK | Wiki-links that don't resolve to existing files |
| STALE_FACT | Facts with `last_updated > 30 days` old |
| STALE_RELATION | Broken relation paths in frontmatter `relations:` blocks |
| MEMORY_HYGIENE | Missing file references in `MEMORY.md` |
| ORPHAN_FILE | Files with zero inbound wiki-links or relations |
| THIN_PROFILE | Entities with < 3 facts (no knowledge/projects references) |
| MISSING_FRONTMATTER | Files missing frontmatter or incorrect segment/type fields |
| LARGE_DOC | Documents over 300 lines that may need splitting |

### Enrichment Categories (Handled by Research Loop)

| Category | Description |
|----------|-------------|
| ENRICH_THIN_PROFILE | Entities with < 3 facts BUT 2+ references in knowledge/projects |
| ENRICH_STALE_TOPIC | Knowledge files not updated in > 7 days with 2+ inbound links |
| ENRICH_STUB | Files in knowledge/projects with < 200 characters of content |

### Scanner Workflow

1. **File Index Building** — Scans all `.md` files,building:
   - Basename index: `{lowercase_name_without_ext: [full_paths]}`
   - Relative path index: `{lowercase_rel_path_without_ext: full_path}`
   - Used for wiki-link resolution (handles various wiki-link formats)

2. **Relation Inbound Counting** — Parses frontmatter `relations:` blocks:
   - Extracts `related-to:` and `references:` arrays
   - Builds map of which files have inbound relations
   - Used for orphan file detection and enrichment prioritization

3. **Health Score Computation** — Weighted composite across 11 dimensions:
   ```
   Overall = Σ(score_i × weight_i)
   
   Weights:
   - BROKEN_LINK: 15%
   - STALE_FACT: 15%
   - MEMORY_HYGIENE: 10%
   - ORPHAN_FILE: 10%
   - THIN_PROFILE: 5%
   - STALE_RELATION: 10%
   - ENRICH_THIN_PROFILE: 5%
   - ENRICH_STALE_TOPIC: 5%
   - ENRICH_STUB: 5%
   - MISSING_FRONTMATTER: 10%
   - LARGE_DOC: 10%
   ```

4. **Queue Idempotency** — Preserves status