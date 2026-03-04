/**
 * orchestrator-prompt.ts — Builds the system prompt for orchestrator instances.
 *
 * The orchestrator prompt lives in the vault at ~/obsidian/agents/orchestrator.md
 * for visibility and easy editing. This module appends dynamic sections (task,
 * context, pipeline hint) at spawn time.
 *
 * If the vault file is missing, falls back to a built-in default.
 */

import { memoryAgentConfig } from "./agents/index.js";

const SOUL_EXCERPT = `You're not a chatbot — you're a competent, resourceful project coordinator.
Be direct. Have opinions. Report findings honestly.
Be careful with external actions. Be bold with internal ones.
Don't ask for permission — make decisions and execute. Report what you did.`;

export type PipelineType = "code" | "research" | "security" | "full" | "custom";

/** Fallback orchestrator prompt if vault file is missing */
const FALLBACK_PROMPT = `# Project Coordinator

You are an autonomous project coordinator. Analyze the task, decide which specialist agents to use (coder, reviewer, tester, planner, auditor, researcher, operator), dispatch them via the Task tool, and compile a structured report.

## Personality
${SOUL_EXCERPT}

## Memory / Vault
${memoryAgentConfig.description}

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
 */
export function buildOrchestratorPrompt(
  task: string,
  pipeline?: PipelineType,
  context?: string,
  vaultPrompt?: string | null,
): string {
  const basePrompt = vaultPrompt || FALLBACK_PROMPT;

  const pipelineHint = pipeline && pipeline !== "custom"
    ? `\n## Suggested Approach: ${pipeline}\nThe caller suggested a "${pipeline}" pipeline. Use this as a starting point but adapt as needed based on your analysis.`
    : "";

  return `${basePrompt}
${pipelineHint}
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
