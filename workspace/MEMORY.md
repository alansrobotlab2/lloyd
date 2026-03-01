# MEMORY.md - Long-Term Memory

## Obsidian Vault
- Path: /home/alansrobotlab/obsidian

## Alan's Roommates
- **Lisa Engelhardt** — closest friend (35 years), roommate for ~15 years. PA, loves her job but hates the bureaucracy/notes overhead. Works Mon/Tue/Thu/Fri (12-hr shifts), off Sun/Wed/Sat. Loves water, ocean, travel. When Alan says "Lisa" it's her.
- **Emilio Mendez** — close friend and roommate

## Local Context Model
- Model: Unsloth Qwen3.5-35B-A3B-GGUF (replaces previous Qwen3-0.6B setup)
- ~160 tps throughput
- nvFP4 quant under vLLM abandoned — not enough room for Triton + KV cache + autotuner simultaneously
- Used to populate initial session context; primary inference remains on Sonnet 4.6

## Claude (LLM) Setup
- Primary model: Claude Sonnet 4.6 via Anthropic directly (Max plan, OAuth auth)
- Claude Code CLI: use for file-heavy/multi-step tasks; Sonnet 4.6 or Opus for deeper reasoning

## Skills Directory
- Operational procedures live in `~/.openclaw/workspace/skills/`
- Check skills before improvising a complex multi-step task
- Add new skills as tasks are completed; refine when steps change
- **Key skills:** claude-code-subagent.md, obsidian-vault-maintenance.md, youtube-transcript.md, websearch.md, lloyd-voice-tui (SKILL.md in lloyd/skills/)

## Claude Code as Sub-Agent (Directive)
- Coding tasks that need to be spun off → use **`exec`** with `pty=true` + `background=true`, NOT `sessions_spawn`
- `sessions_spawn` = OpenClaw sub-agents (another Claude instance); `exec` = terminal CLI tools like Claude Code/Codex
- Always set `workdir` to the target project directory
- Always append notify command to prompt: `openclaw system event --text "Done: summary" --mode now`
- Never run in `~/.openclaw/` workspace
- Monitor with `process(action="log"/"poll"/"submit"/"kill", sessionId="XXX")`
- Use Claude Code for: new features, large refactors, multi-file edits, iterative work
- Do inline for: one-liner fixes, reading files, quick 1-3 file edits

### Claude Code Launch: Key Rules
- All exec/tmux calls are container-local (lloyd distrobox). Use `--dangerously-skip-permissions` flag.
- Write a `/tmp/<session>.sh` launch script → tmux exec → `sleep 4` → send prompt as keyboard input (never as CLI arg)
- Full procedure: `skills/claude-code-subagent.md`

## HuggingFace
- Read-only token: `REDACTED_HF_TOKEN`
- Note: there's also an older token in `projects/alfie/jetson-orin-nx-16gb/tensorrt-llm.md` (`REDACTED_HF_TOKEN`) — may be stale

## Alan's GitHub
- GitHub username: alansrobotlab2
- Note: confirmed by Alan directly — the correct account is alansrobotlab2, not alansrobotlab

## Alan's Dog
- Name: Gracie
- Breed: Great Dane (merle)
- Age: 9 years old
- Very important to Alan
- Great with people and children
- Loves playing with [[stompy-overview|Stompy]] in the backyard

## Alfie (Humanoid Robot) — Project Status
- GR00T N1.6 fine-tuning: Phase 3 complete (323 episodes, 10k steps, TensorRT FP16, 200ms on Orin AGX)
- Phase 4 next: closed-loop testing, behavior tuning, action horizon tuning, full ROS2 integration
- Mic stream crash: ✅ fixed. OAK-D camera: no longer in use on Alfie.
- Full details: `projects/alfie/gr00t/phase2-summary.md`, `projects/alfie/1-work-backlog.md`

## Project vs. Personal Directories (Obsidian)
- `lloyd/` in the vault root = Lloyd's personal/agent files (soul, memory, skills, identity)
- `projects/lloyd/` = the Lloyd **project** notes (architecture, backlog, GPU allocation, etc.)
- Always put project work items (backlogs, specs, plans) under `projects/lloyd/`, not `lloyd/`

## Stompy (Go2 Air)
- Unitree Go2 Air quadruped with hacked/custom firmware
- Named Stompy because his footsteps are very loud
- Lives under Alan's desk
- Gracie loves playing with him in the backyard
- Obsidian project notes: projects/stompy/

## Alan's Patterns & Preferences

Behavioral patterns observed over time — update as new ones emerge.

- Prefers concise check-ins; gets straight to the point
- When asking for a YouTube summary, wants core argument + key findings + takeaways (not a transcript recap)
- Likes skills written as simple `.md` files, not packaged directories
- Gives quick project status updates mid-conversation — capture them, don't interrogate them
- Usually picks up where things left off; context continuity matters to him
- When doing web research, always include links to the source pages used in the answer
- Vault knowledge files must capture ALL source URLs: inline in body AND in `sources:` frontmatter list (not just `source:` for one)

## QMD Memory System
- `includeDefaultMemory: true`, daily-notes manually added to index.yml (paths[] config not generating it — possible OpenClaw bug)
- Session memory: enabled, 30-day retention; MMR (λ=0.7) + temporal decay (halfLifeDays=30) active
- `MEMORY.md` is a symlink → indexed via obsidian vault collection (fine as-is)
- Watch: `daily-notes-main` entry in index.yml may be overwritten on gateway restart — re-add and run `openclaw memory index` if daily notes drop from search

## Lloyd Project Status
- **OpenClaw update pending:** 2026.2.25 available (security fixes: WebSocket auth, trusted-proxy bypass, macOS OAuth PKCE). Update: `sudo npm update -g openclaw`. Current: 2026.2.21-2
- Nightly memory consolidation cron running: 2am PST daily (job f53ad621)
- Android app: almost working; earbud button config still has issues
- Orpheus TTS: emotive tags (`<laugh>`, `<sigh>`, etc.) confirmed as the full expressive tag set; backlog item to evaluate for Lloyd's persona
- YouTube transcript skill built: `skills/youtube-transcript.md` (uv + youtube-transcript-api → local Qwen3.5 → knowledge file)
- Web tools (web_search, web_fetch) confirmed running via MCP

## Alan's Sister
- **Lori Hokanson** — Alan's sister
- Lives in Oceanside, CA (306 Justina Dr, 92057)
- Teacher at Bonsall West Elementary School (Bonsall USD)
- Husband: Ron Hokanson
- Previously: Lori Timm / Lori Williams (prior surnames)
