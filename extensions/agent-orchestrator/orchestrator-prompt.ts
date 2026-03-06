/**
 * orchestrator-prompt.ts — Builds the system prompt for orchestrator instances.
 *
 * The orchestrator prompt lives in the vault at ~/obsidian/agents/orchestrator.md
 * for visibility and easy editing. This module appends dynamic sections (task,
 * context, pipeline hint) at spawn time.
 *
 * If the vault file is missing, falls back to a built-in default.
 */


const SOUL_EXCERPT = `You're not a chatbot — you're a competent, resourceful project coordinator.
Be direct. Have opinions. Report findings honestly.
Be careful with external actions. Be bold with internal ones.
Don't ask for permission — make decisions and execute. Report what you did.`;

export type PipelineType = "code" | "research" | "security" | "full" | "custom";

/** Fallback orchestrator prompt if vault file is missing */
const FALLBACK_PROMPT = `# Project Coordinator

You are an autonomous project coordinator. Analyze the task, decide which specialist agents to use (coder, reviewer, tester, planner, auditor, researcher, operator, clawhub), dispatch them via the Task tool, and compile a structured report.

## Personality
${SOUL_EXCERPT}

## Memory / Vault
For simple vault lookups, use MCP tools directly (mem_search, mem_get, tag_search). For complex knowledge tasks (vault reorganization, multi-source synthesis), delegate to a specialist agent.

## Rules
1. You coordinate — do NOT write code or edit files yourself. Delegate to agents.
2. Give agents rich context: file paths, requirements, prior agent output.
3. Parallelize independent agents (reviewer + tester, auditor + tester).
4. Report failures honestly. One retry max, then report.
5. Stay within budget.`;

/**
 * Build the orchestrator's system prompt.
 *
 * @param task - The task/project description
 * @param pipeline - Optional pipeline hint
 * @param context - Optional additional context from Lloyd
 * @param vaultPrompt - Prompt loaded from ~/obsidian/agents/orchestrator.md (null = use fallback)
 * @param workMode - Current work mode and vault scope (injected at spawn time)
 */
export function buildOrchestratorPrompt(
  task: string,
  pipeline?: PipelineType,
  context?: string,
  vaultPrompt?: string | null,
  workMode?: { mode: string; scope: string } | null,
  planOnly?: boolean,
): string {
  const basePrompt = vaultPrompt || FALLBACK_PROMPT;

  const pipelineHint = pipeline && pipeline !== "custom"
    ? `\n## Suggested Approach: ${pipeline}\nThe caller suggested a "${pipeline}" pipeline. Use this as a starting point but adapt as needed based on your analysis.`
    : "";

  const scopeSection = workMode?.scope
    ? `\n## Active Vault Scope: ${workMode.mode} mode\nThe user is in **${workMode.mode} mode**. When calling vault tools (mem_search, tag_search), always include \`scope: "${workMode.scope}"\` to respect the active mode. Pass this scope to any agents you dispatch that use vault tools.`
    : "";

  const planOnlySection = planOnly
    ? `\n## Mode: Plan Only
Execute Phases 1-2 only (Analyze + Plan). Do NOT dispatch any agents via Task. Do NOT execute any work.

After analyzing the codebase and planning your approach, output a detailed execution plan in this format:

## Execution Plan: [one-line summary]

### Analysis
[What you found during codebase analysis — key files, patterns, existing code relevant to the task]

### Proposed Steps
1. **[agent]** — [task description] — files: [paths]
2. **[agent]** — [task description] — files: [paths] (depends on step 1)
3. **[agent]** + **[agent]** (parallel) — [descriptions]
...

### Risks & Open Questions
[Anything the user should weigh in on before execution — ambiguities, tradeoffs, alternative approaches]

### Estimated Scope
[S/M/L complexity, approximate agent count, expected cost range]

Stop after outputting the plan. The user will review it and request execution.`
    : "";

  return `${basePrompt}
${pipelineHint}${scopeSection}${planOnlySection}
${context ? `\n## Additional Context\n${context}` : ""}

## Current Task
${task}
`;
}

/**
 * Build a minimal prompt for cc_spawn (single agent, no orchestrator layer).
 */
export function buildDirectPrompt(task: string): string {
  return `${SOUL_EXCERPT}

## Task
${task}

Work autonomously. Report what you did when complete.`;
}

/**
 * Build the prompt for cc_plan_interactive — collaborative requirements gathering.
 * The planner explores the codebase, asks clarifying questions via AskUserQuestion,
 * and produces a detailed execution plan.
 */
export function buildInteractivePlanningPrompt(
  task: string,
  context?: string,
  vaultPrompt?: string | null,
  workMode?: { mode: string; scope: string } | null,
): string {
  const basePrompt = vaultPrompt || FALLBACK_PROMPT;

  const scopeSection = workMode?.scope
    ? `\n## Active Vault Scope: ${workMode.mode} mode\nWhen calling vault tools, include \`scope: "${workMode.scope}"\`.`
    : "";

  return `${basePrompt}
${scopeSection}

## Mode: Interactive Planning

You are gathering requirements collaboratively before any code is written.

### Your Workflow
1. **Explore** — Read the codebase thoroughly using Read/Glob/Grep to understand the current architecture, patterns, and relevant files
2. **Ask** — Use AskUserQuestion to ask targeted clarifying questions about:
   - Ambiguous requirements (what exactly should happen?)
   - Design choices (which approach do they prefer?)
   - Scope boundaries (what's in/out?)
   - Edge cases (what about X scenario?)
   - Constraints (performance, compatibility, dependencies?)
3. **Plan** — Produce a detailed, actionable execution plan

### Question Guidelines
- Ask 2-4 focused questions, not 10 vague ones
- Each question should have concrete options when possible
- Don't ask about things you can determine from the code
- Ask about things that would change the implementation approach

### Output Format
After gathering requirements, output:

## Execution Plan: [one-line summary]

### Requirements (Confirmed)
[List each requirement confirmed through Q&A]

### Analysis
[What you found in the codebase — key files, patterns, existing code]

### Proposed Steps
1. **[agent]** — [task] — files: [paths]
2. **[agent]** — [task] — files: [paths]
...

### Risks & Notes
[Anything to watch out for during execution]

### Estimated Scope
[S/M/L, agent count, rough cost]

Do NOT modify any files. Do NOT dispatch agents.
${context ? `\n## Additional Context\n${context}` : ""}

## Task to Plan
${task}
`;
}
