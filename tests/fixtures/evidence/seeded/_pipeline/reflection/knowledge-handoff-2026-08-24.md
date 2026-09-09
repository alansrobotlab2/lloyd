---
date: 2026-08-24
generated_at: 2026-08-24T05:47:00Z
---

# Knowledge Handoff — 2026-08-24

## Person: alan

### Mental Model — Decision Patterns
- Verifies state claims against disk before acting: the 08-22 "Entity Graph Status" session opened with `cat _relationships.json` (file-not-found) and the 08-22 systems check re-verified hardware (GPU count/driver/RAM) after migration instead of trusting prior inventory.
- Forensics-first incident handling: session "Entity Graph Status" reconstructed a full timeline (16:13 task start, 16:38 aliases truncation, 17:47 extraction end) with per-minute log attribution before deciding action.
- Never destructive against broken baselines: the 08-22 `entity-resolution-sweep --dry-run` crash was treated as caught-in-time; the 08-23 plan gates `--apply` behind a writer fix (pre-write .bak + path-scope exclusion) rather than re-running blindly.
- Re-verifies recovery outcomes independently: 08-23 "Entity Graph Status" re-ran the KG health check after the claimed restore; the check found `_relationships.json` still missing and 0 relationships — the handoff's "restored to 12,131" claim was an overclaim against the health report. Pattern: trust health-check output over prior claims.

### Mental Model — Communication Preferences
- Video-first research: 3 of 4 video sessions (08-17, 08-18, 08-20) were user-initiated video links; follow-up chat turns are terse (single-word "ok", "i mean", corrections) rather than re-briefing.
- Compound bash preference continued in ops sessions; terse corrections when imprecise ("i mean", 08-18).
- Unprompted status reporting is still a negative pattern — session "Entity Graph Status" (08-22) was user-initiated after the migration, not agent-pushed.

### Mental Model — Technical Preferences
- Local LLM + harness focus: 3 sessions on the DeepSeek MIT harness (repo, paper, portability to Python) — interest in agent-loop subsystems (compaction/guard/goal/spill/session log) and "everything is a plugin" as portable design patterns.
- Markdown over JSON for pipeline artifacts (reinforced by the handoff format being markdown, not JSON).
- Rust, modular/decoupled, self-hosted preferences unchanged — no new contradicting signals.

### Mental Model — Project Prioritization
- KG/entity-graph recovery is top priority: dominant thread across 08-22 → 08-23 (4+ sessions), and the 08-24 health check confirmed it is still unresolved (0 relationships).
- Qwen 3.8 27B was the dominant research topic (08-18 "DeepSeek moment" + 08-20 re-watch); 08-20 added agentic-engineering interest (Claude Code loop engineering + $75M founder setup).
- FreeCAD remains active (2 back-to-back 08-17 sessions + tutorial 08-17).

### MEMORY.md Additions
- KG health check 2026-08-24 04:13 UTC confirmed the entity graph is still destroyed: 0 total relationships, 0 active edges, `_relationships.json` still missing, 0 entity files in _pipeline/entities — the 08-23 handoff's "graph restored to 12,131" was an overclaim; only `entity-aliases.json` was restored via git (08-23); recovery via remote backup or rebuild via classify-relationships-v4.py (with pre-apply edge-count sanity check vs 12,131) is still pending.
- corrections_log 2026-08-24: 2026-08-23 handoff claimed the entity graph was "restored to 12,131 relationships," contradicting the same-night knowledge-health report (0 relationships, `_relationships.json` missing) and disk state; the graph is still destroyed as of the 08-24 health check. Rule: recovery claims must be verified against a fresh health check / disk before being written into handoffs or memory.
- Task frontmatter `timeout_seconds` is NOT read by pool.py — the effective cap is the source-level `max_duration_seconds`; entity-resolution-sweep (~30 min) poisoned 3× at the old 1200s cap; fix: commit `699b2d17` bumped the scheduled-task source cap 1200 → 1800s. Task authors must set timeouts at source level, never in frontmatter.
- 08-20 research: Claude Code "loop engineering" = automating verification against user-defined specs for unattended agent execution (3 levels; level 1 = single-goal loop via the `/goal` command); $75M founder's agentic-engineering setup — engineering markdown outweighs executing code, with a separate "context repo" holding agent-facing docs.
- 08-20 research: Qwen 3.8 27B video re-watch (same topic as the 08-18 "DeepSeek moment" session) plus Robotics Tech podcast on Optimus Gen 3 / Figure 03 / Unitree R1.
- memory-capture.log remains stale (last entry 2026-06-03) — flagged again 08-19; a one-time check of the memory-capture job's logging is still due at the next autonomy-task diagnosis pass.
- VTT-parser dedup fix is still pending: 08-19 DeepSeek Harness video VTT parse returned duplicate trailing lines (tail-echo) even though extraction succeeded; the 08-19 truncation guardrail (final-sentence-complete check + `[partial extraction]` marker) held with no new cut-offs since.
- Daily note `memory/2026-08-22.md` was created as a stub after the fact (post-migration log gap); its sessions section was filled from log evidence on 08-22 evening — treat its content as log-derived, not live-captured.
- RSSC monthly meetup: Michael Lynch confirmed for September 5, 2026, with a project-presentation slot on the event schedule (08-20 triage reply).
