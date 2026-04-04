# Subliminal Injection: Analysis & Implementation Gameplan

## What It Is

`~/obsidian/agents/lloyd/subliminal.md` defines an **Operating Contract** for the `lloyd-prime` agent profile:

```yaml
agent: lloyd-prime
type: subliminal
injected: context-hook
```

The contract specifies:
- **Autonomous loop**: ASSESS → ACT → EVALUATE → REPEAT/SIGNAL on every turn
- **Signal syntax**: `SIGNAL:STAGE_COMPLETE`, `SIGNAL:TASK_COMPLETE`, `SIGNAL:BLOCKED:<reason>`
- **Rules**: every response must include a tool call or signal; no permission-seeking; 70% confidence threshold to act
- **Skill check**: `skills_search` must be the first tool call on any new task
- **Workspace isolation**: all code changes go through `agent-ws begin/submit`

The intent is to inject this behavioral spec into every LLM API call — ephemerally, not persisted in session history — so Lloyd operates autonomously without being told to each time.

---

## Existing Infrastructure

### The Hook That Already Exists: `pre_llm_call`

In `run_agent.py:6640`, there is already a `pre_llm_call` plugin hook that fires **before every single LLM API call** in the tool loop (not just once per user message):

```python
_pre_results = _invoke_hook(
    "pre_llm_call",
    session_id=self.session_id,
    user_message=original_user_message,
    conversation_history=list(messages),
    is_first_turn=(not bool(conversation_history)),
    model=self.model,
    platform=getattr(self, "platform", None) or "",
)
```

Any plugin returning `{"context": "..."}` gets that string appended to the ephemeral system prompt (`_plugin_turn_context`) for that API call only — it never touches the session DB or prompt cache. This is precisely what the subliminal frontmatter means by `injected: context-hook`.

### Plugin Registration

`hermes_cli/plugins.py` exposes `invoke_hook(hook_name, **kwargs)` which calls all registered plugins that implement that hook. A plugin registers by having a `pre_llm_call(...)` function. Discovery happens from `~/.hermes/plugins/`.

### Gateway Hooks (for Signal Parsing)

`gateway/hooks.py` provides an event system: `agent:start`, `agent:step`, `agent:end`, etc. These fire async handlers from `~/.hermes/hooks/<name>/handler.py`. The signal-parsing phase can live here.

### Agent Profile Context

`config.yaml` already has `agent.personalities` (the display skin layer) and `toolsets`. There is no concept of an "agent profile" (like `lloyd-prime`) yet — that mapping needs to be added.

---

## Multi-Phase Gameplan

### Phase 1 — Core Subliminal Plugin

**Goal**: Get the Operating Contract injected into every LLM call right now, hardcoded to lloyd's path.

**Implementation**:

Create `~/.hermes/plugins/subliminal/__init__.py`:

```python
from pathlib import Path
import yaml

SUBLIMINAL_PATH = Path("~/obsidian/agents/lloyd/subliminal.md").expanduser()

def pre_llm_call(**kwargs):
    if not SUBLIMINAL_PATH.exists():
        return None
    content = SUBLIMINAL_PATH.read_text(encoding="utf-8")
    # Strip YAML frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:].strip()
    return {"context": content}
```

**Register** via `config.yaml` adding `subliminal` to `known_plugin_toolsets.cli` (or `platform_toolsets.cli`).

**Test**: Start a session, confirm that the Operating Contract appears in effective_system but not in any stored message.

**Notes**:
- The content is injected at `run_agent.py:6764-6765` on every loop iteration
- It does NOT survive compression or get stored in state.db
- Prompt caching: this lands AFTER the stable cached prefix, so it does not break cache hits

---

### Phase 2 — Agent Profile Binding

**Goal**: Make the subliminal profile dynamic — selectable per-session or per-platform without hardcoding a path.

**Design**:

Add to `config.yaml`:
```yaml
agent:
  profile: lloyd-prime          # maps to ~/obsidian/agents/lloyd-prime/
  profile_dir: ~/obsidian/agents # base directory for all profiles
```

The plugin reads `config.agent.profile` at call time (config is already live-read from disk per turn), resolves the path, and loads any files named `subliminal.md` in that profile directory.

**Gateway integration**: Allow `agent_profile` to be set per-platform in `platform_toolsets`, so Discord sessions can use a different profile than CLI.

**Profile directory convention**:
```
~/obsidian/agents/lloyd/
  subliminal.md        # Operating Contract
  SOUL.md              # (optional) personality override
  skills/              # profile-local skills
```

**Why not personalities?** The existing `personalities` key just swaps the character voice. A profile is a different concept — it encodes behavioral contracts, not tone. Keep them separate.

---

### Phase 3 — Multi-Subliminal + Budget Capping

**Goal**: Support multiple subliminal files in a profile directory, with token budget enforcement so the injection doesn't bloat the context.

**Design**:

A profile directory can have multiple subliminals:
```
~/obsidian/agents/lloyd/
  subliminal.md          # Core operating contract (always injected)
  subliminal-agentic.md  # Additional rules for long agentic runs
  subliminal-coding.md   # Rules when code execution tools are active
```

Each file has frontmatter with:
```yaml
type: subliminal
priority: 10        # lower = higher priority
when_tools:         # only inject when these tools are in the active toolset
  - code_execution
  - terminal
token_budget: 500   # max tokens this file may consume (approximate by char/4)
```

The plugin:
1. Reads all `subliminal*.md` files in the profile dir
2. Filters by `when_tools` (cross-reference against tools passed in `pre_llm_call`)
3. Sorts by priority
4. Truncates to a total budget (default 3000 tokens ≈ 12000 chars)
5. Joins and returns

**Keyword freshness** (from the pi-coding-agent pattern): Track a set of keywords from the last N injections. If the current conversation hasn't drifted topics, skip re-injection of files that are stable. This saves tokens on long runs.

---

### Phase 4 — Signal Parsing & Hook Emission

**Goal**: When Lloyd emits `SIGNAL:TASK_COMPLETE` (or variants), the gateway should detect it and fire a hook event so other systems can react.

**Design**:

Add a new builtin hook at `gateway/builtin_hooks/signal_parser.py` registered for `agent:end`:

```python
import re

SIGNAL_RE = re.compile(r'\bSIGNAL:([A-Z_]+)(?::(.+))?\b')

async def handle(event_type, context):
    response = context.get("response", "")
    for match in SIGNAL_RE.finditer(response):
        signal_name = match.group(1)
        signal_arg  = match.group(2) or ""
        # Emit as a sub-event: signal:TASK_COMPLETE, signal:BLOCKED, etc.
        # The HookRegistry is not available here directly, so use a lightweight
        # pub/sub or write to a signals queue file for autonomy system to pick up.
        _handle_signal(signal_name, signal_arg, context)
```

Signals to handle:
- `SIGNAL:STAGE_COMPLETE` → log + notify via `display.background_process_notifications`
- `SIGNAL:TASK_COMPLETE` → mark matching autonomy task as done, advance `next_run`
- `SIGNAL:BLOCKED:<reason>` → log reason, optionally surface to user via chat

**Autonomy integration**: When `SIGNAL:TASK_COMPLETE` fires, the signal parser looks up the active autonomy task (stored in `session_entry` or via env var `HERMES_SESSION_KEY`) and calls `autonomy_write_task` to update its status.

---

### Phase 5 — Skill Check Enforcement

**Goal**: Enforce the subliminal rule that `skills_search` must be the first tool call when starting any new work — without relying on the LLM to follow the injected instruction.

**Design**:

Extend the plugin to also implement a `post_llm_call` hook (if added to run_agent) or use the existing `agent:step` gateway hook:

In the `agent:step` handler, check `iteration == 1` and `tool_names` does not contain `skills_search`. If so, inject a correction into the next turn via a mechanism similar to the `_honcho_turn_context` injection (line 6726).

Alternatively — and simpler — use **tool ordering hints** in the subliminal itself rather than enforcing in code. The Operating Contract already states this clearly enough that capable models will comply. Add code enforcement only if empirical observation shows the model skipping it.

**Skill creation nudge**: The subliminal says "write a new skill before TASK_COMPLETE if no match was found". The existing `skills.creation_nudge_interval` (config, currently 15 iterations) already nudges skill creation. Lower this to 8 for agentic profiles.

---

## Integration Points Summary

| What | Where in codebase | Hook/mechanism |
|------|-------------------|----------------|
| Inject subliminal every LLM call | `run_agent.py:6640` | `pre_llm_call` plugin |
| Agent profile config | `config.yaml` + plugin | `agent.profile` key |
| Multi-file budget capping | Plugin internal | char-budget + `when_tools` filter |
| Signal parsing | `gateway/builtin_hooks/` | `agent:end` hook handler |
| Task completion update | `plugins/autonomy/__init__.py` | Signal → autonomy task write |
| Skill check enforcement | Optional: `agent:step` hook | Iteration 1 tool list check |

---

## Recommended Build Order

1. **Phase 1** first — gets value immediately, zero risk, one file to create
2. **Phase 2** next — unlocks multi-profile before you need it
3. **Phase 4** before Phase 3 — signals are needed for autonomy; multi-subliminal is nice-to-have
4. **Phase 3** as needed — only necessary if token costs from injection become visible
5. **Phase 5** last — validate empirically whether the LLM follows the skill-check rule before adding enforcement overhead

---

## Open Questions

- Does `lloyd-prime` need to be distinct from the default CLI personality, or is `lloyd` the default and this just adds the behavioral layer on top? (If the latter, Phase 2 is optional for now.)
- Should signals strip themselves from the response before delivery to the user, or be displayed? (Probably display — they're useful audit trail.)
- The `agent-ws begin/submit` workflow referenced in the subliminal doesn't exist in the hermes tool inventory yet. Is that a future tool or an external script?
