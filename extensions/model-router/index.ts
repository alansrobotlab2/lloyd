/**
 * index.ts — Hybrid 4-tier model router for OpenClaw
 *
 * Routes prompts to the optimal model tier:
 *   local (Qwen) → haiku → sonnet → opus
 *
 * Uses `before_agent_start` for session context access.
 *
 * Decision flow:
 *   1. Explicit overrides (/fast, /opus, /sonnet, /haiku)
 *   2. Feature-based scoring (prompt structure + session context)
 *   3. Local Qwen LLM classifier (for ambiguous cases)
 *   4. Default → sonnet
 *
 * All decisions logged to logs/routing.jsonl.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { readFileSync, appendFileSync } from "fs";
import { join } from "path";

// ── Types ────────────────────────────────────────────────────────────────

type Tier = "local" | "haiku" | "sonnet" | "opus";

interface RouterConfig {
  enabled: boolean;
  defaultTier: Tier;
  classifierEnabled: boolean;
  classifierTimeoutMs: number;
  confidenceThreshold: number;
  verbose: boolean;
  disabledTiers: Tier[];
}

interface TierScores {
  local: number;
  haiku: number;
  sonnet: number;
  opus: number;
}

interface RoutingDecision {
  tier: Tier;
  reason: string;
  confidence: number;
  classifierUsed: boolean;
  latencyMs: number;
}

const TIERS: Record<Tier, { modelOverride: string; providerOverride: string }> = {
  local: { modelOverride: "Qwen3.5-35B-A3B", providerOverride: "local-llm" },
  haiku: { modelOverride: "claude-haiku-4-5", providerOverride: "anthropic" },
  sonnet: { modelOverride: "claude-sonnet-4-6", providerOverride: "anthropic" },
  opus: { modelOverride: "claude-opus-4-6", providerOverride: "anthropic" },
};

const TIER_ORDER: Tier[] = ["local", "haiku", "sonnet", "opus"];

// ── Feature extraction ───────────────────────────────────────────────────

interface PromptFeatures {
  length: number;
  hasCodeBlocks: boolean;
  hasInlineCode: boolean;
  isQuestion: boolean;
  questionType: "factual" | "reasoning" | "instruction" | "none";
  hasToolDispatchIntent: boolean;
  hasCodeIntent: boolean;
  hasConstraints: boolean;
  hasDeepAnalysisIntent: boolean;
  hasSummarizeIntent: boolean;
  sessionDepth: number;
  priorToolsUsedCode: boolean;
  priorTierWasOpus: boolean;
}

const FACTUAL_PATTERNS = /\b(?:what is|what are|who is|who are|when did|when was|where is|how many|how much|yes or no|define|meaning of)\b/i;
const REASONING_PATTERNS = /\b(?:why (?:does|did|is|are|do)|how (?:does|did|do|can|should|would)|explain (?:why|how)|what (?:causes|would happen))\b/i;
const TOOL_DISPATCH_PATTERNS = /\b(?:search for|look up|find|list files|read file|check|fetch|get the|show me|look at)\b/i;
const CODE_INTENT_PATTERNS = /\b(?:implement|write (?:a |the )?(?:function|class|method|script|code|test)|debug|refactor|fix (?:this |the )?(?:code|bug|error)|add (?:a |the )?(?:feature|endpoint|method)|create (?:a |the )?(?:file|component|module))\b/i;
const CONSTRAINT_PATTERNS = /\b(?:exactly \d+|in \d+ (?:words|sentences|paragraphs|lines)|character by character|word by word|step by step list|numbered list of exactly)\b/i;
const DEEP_ANALYSIS_PATTERNS = /\b(?:architect(?:ure)?|design (?:document|system|pattern)|deep (?:analysis|research|dive)|trade-?offs?|compare and contrast|comprehensive (?:review|analysis)|strategic|long-term plan|evaluate (?:the )?pros and cons|system design)\b/i;
const SUMMARIZE_PATTERNS = /\b(?:summarize|summarise|translate|reformat|bullet points|in one sentence|tldr|tl;dr|brief overview|short summary)\b/i;

function extractFeatures(prompt: string, messages?: unknown[]): PromptFeatures {
  const lower = prompt.toLowerCase();
  const trimmed = prompt.trim();

  // Question type
  let questionType: PromptFeatures["questionType"] = "none";
  if (trimmed.includes("?") || FACTUAL_PATTERNS.test(lower) || REASONING_PATTERNS.test(lower)) {
    if (FACTUAL_PATTERNS.test(lower)) questionType = "factual";
    else if (REASONING_PATTERNS.test(lower)) questionType = "reasoning";
    else questionType = "instruction";
  }

  // Session context
  const msgArray = Array.isArray(messages) ? messages : [];
  const sessionDepth = msgArray.length;

  // Check prior tool usage and model tier from recent message history
  let priorToolsUsedCode = false;
  let priorTierWasOpus = false;
  for (const msg of msgArray.slice(-6)) {
    const m = msg as any;
    if (m?.role === "assistant" && m?.tool_calls) {
      for (const tc of m.tool_calls) {
        const name = tc?.function?.name ?? tc?.name ?? "";
        if (/file_|run_bash|code|edit|write|grep|glob/.test(name)) {
          priorToolsUsedCode = true;
        }
      }
    }
    if (m?.metadata?.model && /opus/i.test(m.metadata.model)) {
      priorTierWasOpus = true;
    }
  }

  return {
    length: trimmed.length,
    hasCodeBlocks: /```/.test(prompt),
    hasInlineCode: /`[^`]+`/.test(prompt) && !/```/.test(prompt),
    isQuestion: trimmed.includes("?") || questionType !== "none",
    questionType,
    hasToolDispatchIntent: TOOL_DISPATCH_PATTERNS.test(lower),
    hasCodeIntent: CODE_INTENT_PATTERNS.test(lower) || /```/.test(prompt),
    hasConstraints: CONSTRAINT_PATTERNS.test(lower),
    hasDeepAnalysisIntent: DEEP_ANALYSIS_PATTERNS.test(lower),
    hasSummarizeIntent: SUMMARIZE_PATTERNS.test(lower),
    sessionDepth,
    priorToolsUsedCode,
    priorTierWasOpus,
  };
}

// ── Feature scoring ──────────────────────────────────────────────────────

function scoreFeatures(f: PromptFeatures): { scores: TierScores; confidence: number } {
  const scores: TierScores = { local: 0, haiku: 0, sonnet: 0, opus: 0 };

  // Prompt length signals
  if (f.length < 80 && !f.hasCodeBlocks) scores.local += 3;
  else if (f.length < 150 && !f.hasCodeBlocks) scores.local += 2;
  else if (f.length < 300) scores.local += 1;
  else if (f.length > 600) scores.sonnet += 1;
  if (f.length > 1200) scores.opus += 1;

  // Question type
  if (f.questionType === "factual") scores.local += 3;
  else if (f.questionType === "reasoning") scores.sonnet += 2;

  // Intent signals
  if (f.hasToolDispatchIntent && !f.hasCodeIntent) scores.local += 4;
  if (f.hasCodeIntent) scores.sonnet += 4;
  if (f.hasCodeBlocks) scores.sonnet += 3;
  if (f.hasConstraints) scores.sonnet += 3; // Local fails these per eval
  if (f.hasDeepAnalysisIntent) scores.opus += 4;
  if (f.hasSummarizeIntent && !f.hasCodeIntent) scores.local += 2;

  // Session context signals
  if (f.priorToolsUsedCode && f.length < 150) {
    // Short follow-up in a code conversation → stay on Sonnet
    scores.sonnet += 3;
  }
  if (f.priorTierWasOpus && f.length < 200) {
    // Short follow-up to an Opus conversation → stay on Opus
    scores.opus += 3;
  }
  if (f.sessionDepth > 10) {
    // Deep conversations trend toward more capable models
    scores.sonnet += 1;
  }

  // Calculate confidence: normalized gap between winner and runner-up
  const sorted = TIER_ORDER.map((t) => ({ tier: t, score: scores[t] }))
    .sort((a, b) => b.score - a.score);
  const top = sorted[0].score;
  const runnerUp = sorted[1].score;
  const confidence = top === 0 ? 0 : (top - runnerUp) / (top + 1);

  return { scores, confidence };
}

// ── LLM classifier ──────────────────────────────────────────────────────

const CLASSIFIER_SYSTEM = `You are a task router. Given a user message, respond with exactly one word — the complexity tier:
- "local" — simple factual Q&A, greetings, quick lookups, tool dispatch ("search for X", "read file Y"), summarize, translate, reformat, short explanations
- "sonnet" — write/debug/refactor code, multi-step reasoning, precise instruction following
- "opus" — architecture design, deep research, complex analysis, tradeoff evaluation

Respond with ONLY the tier name, nothing else.`;

async function classifyWithQwen(
  prompt: string,
  timeoutMs: number,
): Promise<{ tier: Tier; latencyMs: number } | null> {
  const start = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetch("http://127.0.0.1:8091/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer none",
      },
      body: JSON.stringify({
        model: "Qwen3.5-35B-A3B",
        messages: [
          { role: "system", content: CLASSIFIER_SYSTEM },
          { role: "user", content: prompt.slice(0, 2000) },
        ],
        max_tokens: 10,
        temperature: 0,
      }),
      signal: controller.signal,
    });

    const data = (await resp.json()) as any;
    const raw = (data.choices?.[0]?.message?.content ?? "").trim().toLowerCase();
    const latencyMs = Date.now() - start;

    const tierMatch = raw.match(/\b(local|haiku|sonnet|opus)\b/);
    if (tierMatch) {
      // Remap haiku → local; haiku is only available via explicit /haiku override
      const tier = tierMatch[1] === "haiku" ? "local" : tierMatch[1] as Tier;
      return { tier, latencyMs };
    }
    return null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// ── Routing log ──────────────────────────────────────────────────────────

function logDecision(logPath: string, decision: RoutingDecision, promptLength: number, sessionDepth: number) {
  const entry = {
    ts: new Date().toISOString(),
    tier: decision.tier,
    reason: decision.reason,
    confidence: Math.round(decision.confidence * 100) / 100,
    classifierUsed: decision.classifierUsed,
    latencyMs: decision.latencyMs,
    promptLength,
    sessionDepth,
  };
  try {
    appendFileSync(logPath, JSON.stringify(entry) + "\n");
  } catch {
    // silently fail — logging should never break routing
  }
}

// ── Main router ──────────────────────────────────────────────────────────

export default function register(api: OpenClawPluginApi) {
  let config: RouterConfig;
  try {
    const raw = readFileSync(join(__dirname, "config.json"), "utf-8");
    config = JSON.parse(raw);
  } catch {
    config = {
      enabled: true,
      defaultTier: "sonnet",
      classifierEnabled: true,
      classifierTimeoutMs: 1500,
      confidenceThreshold: 0.7,
      verbose: true,
      disabledTiers: [],
    };
  }

  if (!config.enabled) {
    api.logger.info?.("model-router: disabled via config");
    return;
  }

  const logPath = join(__dirname, "../../logs/routing.jsonl");
  const disabled = new Set(config.disabledTiers || []);

  // Check if a tier is available (not disabled)
  const isTierEnabled = (tier: Tier) => !disabled.has(tier);

  // Find the best available tier, falling through to higher tiers if needed
  const fallbackTier = (preferred: Tier): Tier | null => {
    if (isTierEnabled(preferred)) return preferred;
    // Fall through to higher tiers
    const idx = TIER_ORDER.indexOf(preferred);
    for (let i = idx + 1; i < TIER_ORDER.length; i++) {
      if (isTierEnabled(TIER_ORDER[i])) return TIER_ORDER[i];
    }
    // Fall back to lower tiers
    for (let i = idx - 1; i >= 0; i--) {
      if (isTierEnabled(TIER_ORDER[i])) return TIER_ORDER[i];
    }
    return null;
  };

  if (disabled.size > 0) {
    api.logger.info?.(`model-router: active (hybrid 4-tier, disabled: [${[...disabled].join(", ")}])`);
  } else {
    api.logger.info?.("model-router: active (hybrid 4-tier, before_agent_start)");
  }

  api.on("before_agent_start", async (event, ctx) => {
    const start = Date.now();
    const prompt = event.prompt?.trim() ?? "";
    const lower = prompt.toLowerCase();

    // ── Skip framework commands ──────────────────────────────────────
    // Let OpenClaw handle its own slash commands without routing interference
    if (/^\/(?:new|reset|sessions?|help|clear|status|version|skills?|config)\b/.test(lower)) {
      return;
    }

    // ── Stage 1: Explicit overrides ──────────────────────────────────

    const explicitOverrides: [RegExp, Tier, string][] = [
      [/^\/(?:fast|local)\b/, "local", "explicit /fast or /local override"],
      [/^\/haiku\b/, "haiku", "explicit /haiku override"],
      [/^\/(?:deep|opus)\b/, "opus", "explicit /opus or /deep override"],
      [/^\/sonnet\b/, "sonnet", "explicit /sonnet override"],
    ];

    for (const [pattern, requestedTier, reason] of explicitOverrides) {
      if (!pattern.test(lower)) continue;
      const tier = fallbackTier(requestedTier) ?? config.defaultTier;
      const actualReason = tier !== requestedTier ? `${reason} (${requestedTier} disabled, fell through to ${tier})` : reason;
      const decision: RoutingDecision = {
        tier,
        reason: actualReason,
        confidence: 1,
        classifierUsed: false,
        latencyMs: 0,
      };
      if (config.verbose) api.logger.info?.(`model-router: → ${tier} (${actualReason})`);
      logDecision(logPath, decision, prompt.length, (event.messages ?? []).length);
      return TIERS[tier];
    }

    // ── Stage 2: Feature-based scoring ───────────────────────────────

    const features = extractFeatures(prompt, event.messages);
    const { scores, confidence } = scoreFeatures(features);

    // Find top enabled tier by score
    const enabledTiers = TIER_ORDER.filter(isTierEnabled);
    const topTier = enabledTiers.length > 0
      ? enabledTiers.reduce((best, t) => scores[t] > scores[best] ? t : best)
      : config.defaultTier;

    if (confidence >= config.confidenceThreshold && scores[topTier] > 0) {
      const resolvedTier = fallbackTier(topTier) ?? config.defaultTier;
      const decision: RoutingDecision = {
        tier: resolvedTier,
        reason: `feature-scored (${topTier}=${scores[topTier]}, confidence=${confidence.toFixed(2)})`,
        confidence,
        classifierUsed: false,
        latencyMs: Date.now() - start,
      };
      if (config.verbose && resolvedTier !== "sonnet") {
        api.logger.info?.(`model-router: → ${resolvedTier} (${decision.reason})`);
      }
      logDecision(logPath, decision, prompt.length, features.sessionDepth);
      return TIERS[resolvedTier];
    }

    // ── Stage 3: LLM classifier (ambiguous cases) ───────────────────

    if (config.classifierEnabled) {
      const result = await classifyWithQwen(prompt, config.classifierTimeoutMs);
      if (result) {
        const resolvedTier = fallbackTier(result.tier) ?? config.defaultTier;
        const decision: RoutingDecision = {
          tier: resolvedTier,
          reason: `classifier (${result.latencyMs}ms)`,
          confidence: 0.6,
          classifierUsed: true,
          latencyMs: Date.now() - start,
        };
        if (config.verbose) {
          api.logger.info?.(`model-router: → ${resolvedTier} (classifier, ${result.latencyMs}ms)`);
        }
        logDecision(logPath, decision, prompt.length, features.sessionDepth);
        return TIERS[resolvedTier];
      }
      if (config.verbose) {
        api.logger.info?.("model-router: classifier timeout/error, falling through to default");
      }
    }

    // ── Stage 4: Default ─────────────────────────────────────────────

    const defaultTier = fallbackTier(config.defaultTier) ?? config.defaultTier;
    const decision: RoutingDecision = {
      tier: defaultTier,
      reason: "default",
      confidence: 0,
      classifierUsed: config.classifierEnabled,
      latencyMs: Date.now() - start,
    };
    logDecision(logPath, decision, prompt.length, features.sessionDepth);
    return TIERS[defaultTier];
  });
}
