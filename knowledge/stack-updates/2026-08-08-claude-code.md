---
type: stack-update
source: claude-code
version: v2.1.226
previous_version: v2.1.224
updated_at: 2026-08-08T05:24:41Z
---

# Claude Code v2.1.226

## What Changed

### v2.1.226
- Bug fixes and reliability improvements (no detailed changelog provided)

### v2.1.225
- **Gateway spend-limit support**: Added gateway spend-limit support to Claude Code's usage warning. The limit-reached message now names the cap, its reset time, and the operator's message (requires the gateway on 2.1.225)
- **Workspace trust prompt**: Added a workspace trust prompt to `claude agents` for untrusted directories, matching the behavior of `claude`
- **OAuth token fix**: Fixed a transient 401 replacing a long-lived `CLAUDE_CODE_OAUTH_TOKEN` with a stored login's short-lived token, breaking headless sessions until restart
- **MCP OAuth fix**: Fixed MCP OAuth servers on macOS intermittently failing with a burst of 401 errors after a keychain read timed out
- **Auto mode fix**: Fixed auto mode counting a safety-filter refusal of its own permission check toward the consecutive-block limit; the action is still denied, but the model is now told to move on rather than retry
- **Cross-session messages**: Fixed cross-session messages staying parked without a notice or expiry in headless sessions and during startup
- **Conversation history**: Fixed conversation history breaking on Remote Control session resume after very large conversations were compacted
- **Agents list**: Fixed hovering over a session in another project in the agents list changing the directory the next agent starts in
- **Self-hosted runner**: Fixed `claude self-hosted-runner` registering and then failing every session when `--base-dir` cannot be created or written; it now exits at startup with a clear error
- **Web sessions**: Fixed Claude Code on the web sessions being misreported as stuck, re-sending a growing event backlog on every reconnect
- **Remote Control photos**: Improved Remote Control — photos attached from the Claude app are now shown to Claude directly instead of being read from disk with a separate tool call
- **VSCode Focus view**: Fixed Focus view folding away the latest to-do list, a pending question's context, and settled answers; thinking-only folds show "Thought for Ns" and re-collapse when their turn completes
- **SendMessage**: Can now start a conversation with Remote Control sessions on other machines by name (`ListAgents` shows them as `name [ref]`), instead of only replying after they message you first
- **SendMessage recipient**: A Remote Control recipient you already confirmed is never swapped for a same-named session on this machine when its own list couldn't be checked

## Relevant to Us
- The OAuth token fix (v2.1.225) addresses headless session breakage — directly relevant to autonomous agent operation
- MCP OAuth stability fix on macOS — relevant if we use MCP tools on macOS
- Self-hosted runner fix — directly relevant to our self-hosted runner setup
- Remote Control improvements — relevant to multi-machine agent orchestration

## Action Items
- [ ] Review gateway spend-limit integration if using gateway-based Claude Code
- [ ] Test self-hosted-runner with restricted `--base-dir` permissions
- [ ] Verify MCP OAuth server stability on macOS after upgrade
