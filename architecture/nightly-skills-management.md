---
segment: architecture
relations:
  related-to:
  - architecture/nightly-vault-maintenance.md
  - tools.md
  - architecture/voice.md
  - memory.md
  - architecture/autonomy-system.md
  - architecture/evaluation-engine.md
  - architecture/index.md
  - architecture/infrastructure.md
  - autonomy/6-morning-briefing.md
  - architecture/nightly-reflection.md
  - skills.md
tags: [architecture]
type: reference

---















# Nightly Skills Management Architecture

An automated pipeline that mines Lloyd's session transcripts for reusable procedural knowledge,creates and maintains a skills library,and self-corrects incorrect or stale skills.

## Schedule

- **Time:** 3:00 AM PST (daily)
- **Agent:** `memory` (isolated session)
- **Model:** Claude Sonnet 4.6
- **Budget:** <$15 per run
- **Sequence:** Runs after [[nightly-vault-maintenance|reflection-vault]] (2am) and reflection-synthesis (1:30am),before [[nightly-reflection]] (4am)
- **Skill file:** [`nightly-skills-management/SKILL.md`](../../skills/nightly-skills-management/SKILL.md)

## Overview

Every interaction between Alan and Lloyd generates session transcripts stored as JSONL files. These transcripts contain procedural knowledge -- troubleshooting steps,corrections,behavioral rules,gotchas -- that would otherwise be lost between sessions. The skills management pipeline automatically surfaces this knowledge,packages it as reusable skills,and maintains the library over time.

This is a key component of Lloyd's self-improvement architecture,alongside [[nightly-reflection]] (mental models,MEMORY.md consolidation,config improvements) and [[nightly-vault-maintenance]] (structural hygiene).

## Four-Stage Pipeline

```
Session Transcripts (JSONL)          Skills Library
~/.openclaw/agents/*/sessions/       ~/obsidian/skills/
         |                                  ^
         v                                  |
+---------------------+          +----------------------+
|  Stage 1: Extract   |          |  Stage 3: Dedup &    |
|  extract-session-   |          |  Consolidate         |
|  log.py --hours 24  |          |  - Inventory all     |
|                     |          |    custom + built-in  |
|  Output: .log files |          |  - Flag overlaps     |
|  per session        |          |  - Merge redundant   |
+--------+------------+          +----------^-----------+
         |                                  |
         v                                  |
+---------------------+          +----------+-----------+
|  Stage 2: Evaluate  |--------->|  Create / Update     |
|  Read each .log,|          |  Skills              |
|  apply signal       |          |  ~/obsidian/skills/  |
|  detection criteria |          |  <name>/SKILL.md     |
+---------------------+          +----------+-----------+
                                            |
                                            v
                                 +----------------------+
                                 |  Stage 4: Effectiveness|
                                 |  Tracking             |
                                 |  - Read Stage 3 output|
                                 |  - Write effectiveness|
                                 |    metrics            |
                                 +----------------------+
```

### Data Flow

1. **Source**: Session JSONL files from `~/.openclaw/agents/main/sessions/`,`~/.openclaw/agents/memory/sessions/`,and `~/.openclaw/logs/cc-instances/`
2. **Extraction**: Python script produces one `.log` file per session in `~/obsidian/memory/skill-maintenance/YYYY-MM-DD/`
3. **Evaluation**: Each log is read and assessed against signal detection criteria
4. **Output**: New or updated skills in `~/obsidian/skills/<name>/SKILL.md`,report in `~/obsidian/memory/skill-maintenance/YYYY-MM-DD/report.md`

## Signal Detection Criteria

The evaluation stage looks for seven categories of skill-worthy patterns:

### 1. Corrections
Alan corrected Lloyd's behavior or approach. Extract as a guardrail -- update an existing skill with a warning,or create a new skill with the correct procedure.

### 2. Remember Requests
Alan explicitly asked to preserve a procedure ("from now on","always do X"). Create a skill capturing the procedure exactly.

### 3. Failure-to-Fix Chains
Something broke,investigation found a non-obvious root cause after multiple steps. Extract the diagnostic path and fix as a troubleshooting skill.

### 4. Behavioral Rules
Corrections that establish ongoing patterns ("never do X","always Y"). Extract as a guardrail skill or update an existing skill's constraints.

### 5. Troubleshooting Playbooks
Failure-to-fix chains where the root cause was not obvious from the symptom -- even one-time incidents. Extract as: symptom,investigation steps,root cause,fix.

### 6. Stale or Incorrect Skill Steps
Evidence that an existing skill's instructions are wrong or outdated. Two failure modes:
- **(A) Wrong skill selected**: Fix descrip