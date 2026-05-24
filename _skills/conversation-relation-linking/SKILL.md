# Conversation Relation Linking Skill

## Purpose
Extract relationship proposals between vault documents based on conversation co-access patterns.

## When to Use
- When analyzing how documents are accessed together during conversations
- When building knowledge graph connections from session data
- Part of nightly reflection pipeline (runs after session processing)

## Execution Steps
1. Run `--incremental` to extract new co-access pairs since last run
2. Run `--stats` to check for high-weight candidates (>= 0.8 aggregate weight)
3. If high-weight unclassified pairs exist, run `--classify` for LLM relationship typing
4. Run `--approve-strong` to auto-approve proposals with confidence >= 0.85 older than 48h
5. Report results: new proposals, classifications, auto-approvals, top 5 by confidence

## Success Criteria
- Script completes without errors
- Watermark advances correctly
- Proposals are properly classified and approved based on thresholds
