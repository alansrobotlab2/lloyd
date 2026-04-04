# Next-Gen Memory System - Implementation

Complete implementation of the next-gen memory system for OpenClaw.

## Directory Structure

```
obsidian/agents/memory/scripts/next-gen-memory/
├── README.md                 # This file
├── fact_extractor.py         # Fact extraction pipeline
├── relations_index.py        # Relations index generator
├── context_bundle.py         # Context retrieval tool
├── profile_synthesis.py      # Profile synthesis
├── mcp_tools.py              # MCP tools (9 tools)
├── nightly_extraction.py     # Nightly extraction job
├── knowledge_ingestion.py    # Knowledge ingestion
└── vault_bootstrap.py        # Full vault bootstrap
```

## Quick Start

### 1. Test Fact Extraction
```bash
cd ~/obsidian/agents/memory/scripts/next-gen-memory
python3 fact_extractor.py
```

### 2. Get Facts
```bash
python3 mcp_tools.py get-facts alan
python3 mcp_tools.py get-facts alan preferences
```

### 3. Add Facts
```bash
python3 mcp_tools.py add-fact alan preferences "New preference statement"
```

### 4. Get Relations
```bash
python3 mcp_tools.py get-relations "projects/lloyd/architecture/next-gen-memory-subsystem/phase-1-schema-design.md"
```

### 5. Context Bundle
```bash
python3 context_bundle.py "alan preferences and tools"
```

### 6. Get Profile
```bash
python3 profile_synthesis.py alan
```

### 7. Detect Contradictions
```bash
python3 mcp_tools.py detect-contradictions alan
```

### 8. Rebuild Index
```bash
python3 mcp_tools.py rebuild-index all
```

### 9. Run Nightly Extraction
```bash
python3 nightly_extraction.py
```

### 10. Detect Knowledge Gaps
```bash
python3 knowledge_ingestion.py detect-gaps
python3 knowledge_ingestion.py gap-report
```

### 11. Full Vault Bootstrap
```bash
python3 vault_bootstrap.py
```

## MCP Tool Integration

To integrate with OpenClaw MCP server, add these tools to `tool_services.py`:

```python
from obsidian.agents.memory.scripts.next-gen-memory.mcp_tools import MemoryMCPTools, register_mcp_tools

# In your MCP server setup
tools = MemoryMCPTools()

@mcp.tool()
def get_facts(entity: str, category: str = None, status: str = "current") -> dict:
    """Get facts for an entity/category."""
    return tools.get_facts(entity, category, status)

# ... repeat for all 9 tools
```

Or use the registration helper:
```python
register_mcp_tools(mcp)
```

## Nightly Cron Job

Add to crontab for nightly extraction at 2 AM PST:

```cron
0 2 * * * cd /home/alansrobotlab/obsidian/agents/memory/scripts/next-gen-memory && python3 nightly_extraction.py >> /home/alansrobotlab/obsidian/memory/nightly-extraction.log 2>&1
```

## Fact File Format

```yaml
---
type: facts
entity: alan
category: preferences
facts:
  - id: pref-001
    fact: Prefers concise, conversational responses
    confidence: 0.95
    status: current
    document_date: '2026-02-28'
    event_date: null
    source: agents/lloyd/USER.md
    ttl_category: permanent
last_updated: '2026-03-20T20:08:59.072047'
relationships: []
---

# Alan - Preferences

**Entity:** alan
**Category:** preferences
**Fact Count:** 1

## Facts

### pref-001

**Fact:** Prefers concise, conversational responses
**Confidence:** 0.95
**Status:** current
```

## Relation Types

- `implements` ↔ `designed-by`
- `supersedes` ↔ `superseded-by`
- `depends-on` ↔ `required-by`
- `derived-from` ↔ `produces`
- `related-to` ↔ `related-to` (symmetric)
- `conflicts-with` ↔ `conflicts-with` (symmetric)

## Entity Categories

### Alan
- `preferences` — Communication style, tools, environment
- `work` — Job, role, projects
- `household` — Living situation, roommates
- `relationships` — Family, friends
- `skills` — Technical skills
- `goals` — Goals and objectives

### Lloyd
- `configuration` — System configuration
- `architecture` — System design
- `operations` — Operational decisions

### Alfie
- `status` — Current project state
- `hardware` — Components, specs

### Work
- `decisions` — Work-related decisions
- `projects` — Project statuses

## Quality Thresholds

- Fact extraction accuracy: >85%
- Relationship precision: >80%
- Classification accuracy: >90%
- Index consistency: 100%

## Troubleshooting

### LLM Connection Failed
Check if local LLM is running on port 8097:
```bash
curl http://localhost:8097/v1/models
```

### Fact File Not Found
Ensure `memory/facts/` directory exists:
```bash
mkdir -p ~/obsidian/memory/facts/{alan,lloyd,alfie,work}
```

### Index Not Updating
Rebuild index manually:
```bash
python3 relations_index.py --rebuild
```

## License

Internal use only - OpenClaw project
