---
type: stack-update
source: claude-code
version: v2.1.232
previous_version: v2.1.229
updated_at: 2026-08-14T06:07:00Z
---

# Claude Code v2.1.232

## What Changed

**Subagent forking on by default**: `subagent_type: "fork"` subagents now inherit the full conversation and prompt cache. Non-teammate agent spawns in interactive sessions run in the background by default.

**Cross-session messaging**: Type `@` in the prompt to mention another Claude session by name; Claude uses `SendMessage` to reach that session directly. `SendMessage` now delivers to a bare name that exactly matches one live session without confirmation. Interactive sessions keep unique names — name collisions get a `name-word-word` variant.

**GitLab support**: Bare `gitlab.com` repo URLs (including nested subgroups) now clone like `github.com` URLs. Clone auth-failure hints name your actual git host. Settings `additionalMarketplaces` and `allowedMarketplaces` accepted as aliases for `extraKnownMarketplaces` and `strictKnownMarketplaces`.

**Gateway/desktop**: The `desktop:` overlay now accepts every released Desktop setting (was 11 hand-listed keys), validated at boot against Desktop's own schema. Empty/malformed `email_domain` values and empty `managed.policies[].match.groups`/`admin.admin_groups` entries now fail at boot instead of silently matching no one or granting admin access.

**Security fixes**:
- PowerShell permission bypass: variable-writing parameters could silently overwrite `$PSDefaultParameterValues` and redirect later commands' file access — now fixed
- Windows Git Bash Cygwin-style symlink bypass: writes through them now require permission approval
- Nested git repositories inheriting trust from parent directory — each repo now requires its own trust confirmation
- MCP connections hanging for full 30-second connect timeout on server failure or malformed protocol-version reply — now fixed
- Remote Control sessions inheriting transcript/credentials from bridge inside cloud session — now fixed
- Hardened auto-generated cross-session messaging socket directory on shared `/tmp`: pre-planted symlinks or other users' directories are refused
- Hardened Linux filesystem sandbox against protected-path bypass
- `sandbox.ripgrep` now honored only from user, managed, and `--settings` settings; project settings can no longer override it

**Other notable fixes**:
- mTLS client certificate rotation now reloads cert/key automatically on connection errors (no restart needed)
- Stream idle timeout errors now recover on Bedrock, Vertex, and gateway deployments
- Remote Control reconnects for ~30 minutes after network blips, no longer drops after a few blips spread across an hour
- `/feedback` and `/bug` open immediately when invoked while Claude is responding
- `/code-review` at high/xhigh/max effort now runs in background agent like other levels
- `/plugin install plugin@marketplace` refreshes marketplace first
- Fixed `/update` and `/tui` refusing to restart while surviving work was running

## Relevant to Us

- **Subagent forking**: Directly relevant to our agent architecture work. Forked subagents inheriting full conversation + prompt cache could significantly reduce context re-transmission overhead.
- **Cross-session messaging**: `@` mentions and `SendMessage` enable direct inter-session communication — could be leveraged for our multi-agent orchestration patterns.
- **GitLab clone support**: If we add GitLab repos to our workflow, bare URL cloning now works like GitHub.
- **Gateway desktop overlay**: Now accepts full Desktop settings schema — relevant if we use gateway mode with desktop settings.
- **Security hardening**: The PowerShell bypass, Git Bash symlink bypass, and sandbox hardening are important if we run Claude Code in shared/multi-user environments.

## Action Items
- [ ] Evaluate subagent forking (`subagent_type: "fork"`) for our agent orchestration — test context cache inheritance benefits
- [ ] Test cross-session `@` messaging for multi-agent coordination workflows
- [ ] Review gateway desktop overlay settings — can we simplify our `desktop:` config now that all keys are accepted?
- [ ] Verify mTLS cert rotation works in our gateway deployment (no restart needed)
- [ ] Test Remote Control reconnect behavior in our network environment
