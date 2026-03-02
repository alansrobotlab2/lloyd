#!/usr/bin/env npx tsx
/**
 * model-eval.ts — OpenClaw Model Evaluation Harness
 *
 * Compares Sonnet 4.6 vs local Qwen3.5-35B-A3B on orchestrator tasks.
 *
 * Usage (inside lloyd container):
 *   npx tsx tools/model-eval.ts
 *   npx tsx tools/model-eval.ts --suite orchestrator
 *   npx tsx tools/model-eval.ts --no-judge        # skip Opus judge (faster)
 *   npx tsx tools/model-eval.ts --category code   # run one category only
 *   npx tsx tools/model-eval.ts --dry-run         # validate suite without API calls
 */

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

// ─── Config ────────────────────────────────────────────────────────────────

const CONFIG = {
  // OpenRouter API key (from openclaw.json providers.openrouter) — used for Sonnet + Opus judge
  openrouterApiKey: (() => {
    try {
      const cfg = JSON.parse(readFileSync(join(ROOT, "openclaw.json"), "utf8"));
      return cfg.models?.providers?.openrouter?.apiKey as string | undefined;
    } catch {
      return undefined;
    }
  })(),
  openrouterBaseUrl: "https://openrouter.ai/api/v1",
  // Local LLM (from openclaw.json providers.local-llm)
  localBaseUrl: "http://127.0.0.1:8091/v1",
  localModelId: "Qwen3.5-35B-A3B",
  // OpenRouter model IDs
  sonnetModelId: "anthropic/claude-sonnet-4-6",
  opusModelId: "anthropic/claude-opus-4-6",
  maxTokens: 1024,
  judgeMaxTokens: 512,
};

// ─── Types ─────────────────────────────────────────────────────────────────

interface TestCase {
  id: string;
  category: string;
  prompt: string;
  expectedTools: string[];
  mockToolResults?: Record<string, string>;
  rubric: string;
  maxTurns: number;
}

interface TestSuite {
  suite: string;
  description: string;
  tests: TestCase[];
}

interface ToolCall {
  name: string;
  args: Record<string, unknown>;
}

interface ModelResult {
  model: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  response: string;
  toolCalls: ToolCall[];
  error?: string;
}

interface JudgeScores {
  sonnet: number;
  local: number;
  dimensions: {
    name: string;
    sonnet: number;
    local: number;
  }[];
  winner: "sonnet" | "local" | "tie";
  reasoning: string;
}

interface EvalResult {
  testId: string;
  category: string;
  prompt: string;
  rubric: string;
  sonnet: ModelResult;
  local: ModelResult;
  toolAccuracy: { sonnet: boolean; local: boolean };
  judgeScores?: JudgeScores;
}

// ─── Tool definitions (passed to models for tool-selection tests) ───────────

const TOOLS_OPENAI = [
  {
    type: "function" as const,
    function: {
      name: "file_read",
      description: "Read the contents of a file at the given path",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Absolute path to the file" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "qmd_search",
      description: "Search the user's memory/notes vault for relevant content",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          max_results: { type: "number", description: "Max results to return (default 5)" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "http_search",
      description: "Search the web for current information",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          count: { type: "number", description: "Number of results (default 5)" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "run_bash",
      description: "Run a bash command and return its output",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "The bash command to execute" },
        },
        required: ["command"],
      },
    },
  },
];

// Anthropic tool format
const TOOLS_ANTHROPIC = TOOLS_OPENAI.map((t) => ({
  name: t.function.name,
  description: t.function.description,
  input_schema: t.function.parameters,
}));

const SYSTEM_PROMPT = `You are Lloyd, an AI assistant. You help with coding, research, file management, and general tasks.
You have access to tools to read files, search memory/notes, search the web, and run bash commands.
Use tools when they would be helpful to answer the user's request. Be concise and accurate.`;

// ─── ANSI colors ────────────────────────────────────────────────────────────

const C = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  cyan: "\x1b[36m",
  gray: "\x1b[90m",
};

// ─── API Clients ────────────────────────────────────────────────────────────

// Shared OpenAI-compat caller (used for OpenRouter and local vLLM)
async function callOpenAICompat(
  baseUrl: string,
  apiKey: string,
  modelId: string,
  messages: Array<{ role: string; content: string }>,
  useTools: boolean,
  extraHeaders: Record<string, string> = {},
): Promise<{ response: string; toolCalls: ToolCall[]; inputTokens: number; outputTokens: number }> {
  const systemMessage = { role: "system", content: SYSTEM_PROMPT };
  const allMessages = [systemMessage, ...messages];

  const body: Record<string, unknown> = {
    model: modelId,
    max_tokens: CONFIG.maxTokens,
    messages: allMessages,
  };
  if (useTools) body.tools = TOOLS_OPENAI;

  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
      ...extraHeaders,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status} (${baseUrl}): ${err.slice(0, 200)}`);
  }

  const data = (await res.json()) as {
    choices: Array<{
      message: {
        content: string | null;
        tool_calls?: Array<{ function: { name: string; arguments: string } }>;
      };
    }>;
    usage?: { prompt_tokens: number; completion_tokens: number };
  };

  const choice = data.choices[0];
  const response = choice?.message?.content ?? "";
  const toolCalls: ToolCall[] = (choice?.message?.tool_calls ?? []).map((tc) => ({
    name: tc.function.name,
    args: (() => {
      try { return JSON.parse(tc.function.arguments); } catch { return {}; }
    })(),
  }));

  return {
    response,
    toolCalls,
    inputTokens: data.usage?.prompt_tokens ?? 0,
    outputTokens: data.usage?.completion_tokens ?? 0,
  };
}

async function callSonnet(
  messages: Array<{ role: string; content: string }>,
  useTools: boolean,
): Promise<{ response: string; toolCalls: ToolCall[]; inputTokens: number; outputTokens: number }> {
  if (!CONFIG.openrouterApiKey) throw new Error("OpenRouter API key not found in openclaw.json");
  return callOpenAICompat(
    CONFIG.openrouterBaseUrl,
    CONFIG.openrouterApiKey,
    CONFIG.sonnetModelId,
    messages,
    useTools,
    { "HTTP-Referer": "https://openclaw.local", "X-Title": "OpenClaw Eval" },
  );
}

async function callLocal(
  messages: Array<{ role: string; content: string }>,
  useTools: boolean,
): Promise<{ response: string; toolCalls: ToolCall[]; inputTokens: number; outputTokens: number }> {
  return callOpenAICompat(
    CONFIG.localBaseUrl,
    "none",
    CONFIG.localModelId,
    messages,
    useTools,
  );
}

// ─── Multi-turn runner ──────────────────────────────────────────────────────

async function runModel(
  caller: typeof callSonnet | typeof callLocal,
  test: TestCase,
  modelLabel: string,
): Promise<ModelResult> {
  const useTools = test.expectedTools.length > 0;
  const messages: Array<{ role: string; content: string }> = [
    { role: "user", content: test.prompt },
  ];

  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let finalResponse = "";
  const allToolCalls: ToolCall[] = [];
  const startMs = Date.now();

  try {
    for (let turn = 0; turn < test.maxTurns; turn++) {
      const result = await caller(messages, useTools);
      totalInputTokens += result.inputTokens;
      totalOutputTokens += result.outputTokens;

      if (result.response) finalResponse = result.response;
      allToolCalls.push(...result.toolCalls);

      // If no tool calls, or last turn, we're done
      if (result.toolCalls.length === 0 || turn === test.maxTurns - 1) break;

      // Provide mock tool results and continue
      for (const tc of result.toolCalls) {
        const mockResult = test.mockToolResults?.[tc.name] ?? `(mock result for ${tc.name})`;
        messages.push({
          role: "assistant",
          content: result.response || `[called ${tc.name}]`,
        });
        messages.push({
          role: "user",
          content: `Tool result for ${tc.name}:\n${mockResult}`,
        });
      }
    }

    return {
      model: modelLabel,
      latencyMs: Date.now() - startMs,
      inputTokens: totalInputTokens,
      outputTokens: totalOutputTokens,
      response: finalResponse,
      toolCalls: allToolCalls,
    };
  } catch (err: unknown) {
    return {
      model: modelLabel,
      latencyMs: Date.now() - startMs,
      inputTokens: totalInputTokens,
      outputTokens: totalOutputTokens,
      response: "",
      toolCalls: allToolCalls,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ─── Tool accuracy check ────────────────────────────────────────────────────

function checkToolAccuracy(result: ModelResult, expected: string[]): boolean {
  if (expected.length === 0) return true;
  const called = new Set(result.toolCalls.map((tc) => tc.name));
  return expected.every((t) => called.has(t));
}

// ─── Opus judge ─────────────────────────────────────────────────────────────

async function judgeResponses(
  test: TestCase,
  sonnet: ModelResult,
  local: ModelResult,
): Promise<JudgeScores> {
  if (!CONFIG.openrouterApiKey) throw new Error("OpenRouter API key not found");

  const prompt = `You are evaluating two AI assistant responses to the same prompt.

PROMPT: ${test.prompt}

RUBRIC: ${test.rubric}

RESPONSE A (Sonnet):
${sonnet.error ? `[ERROR: ${sonnet.error}]` : sonnet.response || "[empty response]"}
${sonnet.toolCalls.length > 0 ? `[Called tools: ${sonnet.toolCalls.map((t) => t.name).join(", ")}]` : ""}

RESPONSE B (Local):
${local.error ? `[ERROR: ${local.error}]` : local.response || "[empty response]"}
${local.toolCalls.length > 0 ? `[Called tools: ${local.toolCalls.map((t) => t.name).join(", ")}]` : ""}

Rate each response 0–5 on these dimensions:
- correctness: Is the answer factually/logically correct?
- instruction_following: Does it follow the rubric constraints precisely?
- conciseness: Is it appropriately concise without omitting key info?
- tool_accuracy: Did it use the right tools (if any were expected)?

Respond with JSON only, no explanation outside the JSON:
{
  "sonnet": { "correctness": 0-5, "instruction_following": 0-5, "conciseness": 0-5, "tool_accuracy": 0-5, "overall": 0-5 },
  "local":  { "correctness": 0-5, "instruction_following": 0-5, "conciseness": 0-5, "tool_accuracy": 0-5, "overall": 0-5 },
  "winner": "sonnet" | "local" | "tie",
  "reasoning": "one sentence"
}`;

  const res = await fetch(`${CONFIG.openrouterBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${CONFIG.openrouterApiKey}`,
      "content-type": "application/json",
      "HTTP-Referer": "https://openclaw.local",
      "X-Title": "OpenClaw Eval Judge",
    },
    body: JSON.stringify({
      model: CONFIG.opusModelId,
      max_tokens: CONFIG.judgeMaxTokens,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Opus judge API ${res.status}: ${err.slice(0, 100)}`);
  }

  const data = (await res.json()) as {
    choices: Array<{ message: { content: string | null } }>;
  };
  const text = data.choices[0]?.message?.content ?? "{}";

  // Extract JSON from response (may have surrounding text)
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) throw new Error(`Judge returned non-JSON: ${text.slice(0, 100)}`);

  const parsed = JSON.parse(jsonMatch[0]) as {
    sonnet: Record<string, number>;
    local: Record<string, number>;
    winner: string;
    reasoning: string;
  };

  const dims = ["correctness", "instruction_following", "conciseness", "tool_accuracy"];
  return {
    sonnet: parsed.sonnet?.overall ?? 0,
    local: parsed.local?.overall ?? 0,
    dimensions: dims.map((d) => ({
      name: d,
      sonnet: parsed.sonnet?.[d] ?? 0,
      local: parsed.local?.[d] ?? 0,
    })),
    winner: (parsed.winner as "sonnet" | "local" | "tie") ?? "tie",
    reasoning: parsed.reasoning ?? "",
  };
}

// ─── Report generation ──────────────────────────────────────────────────────

function generateReport(results: EvalResult[], opts: { suite: string; noJudge: boolean }): string {
  const ts = new Date().toISOString().replace("T", " ").slice(0, 16);
  const total = results.length;

  // Aggregate stats
  const validResults = results.filter((r) => !r.sonnet.error && !r.local.error);
  const avgLatencySonnet = validResults.reduce((s, r) => s + r.sonnet.latencyMs, 0) / (validResults.length || 1);
  const avgLatencyLocal = validResults.reduce((s, r) => s + r.local.latencyMs, 0) / (validResults.length || 1);
  const avgTokensOutSonnet = validResults.reduce((s, r) => s + r.sonnet.outputTokens, 0) / (validResults.length || 1);
  const avgTokensOutLocal = validResults.reduce((s, r) => s + r.local.outputTokens, 0) / (validResults.length || 1);
  const toolResults = results.filter((r) => r.sonnet.toolCalls.length > 0 || r.local.toolCalls.length > 0 || r.toolAccuracy.sonnet !== r.toolAccuracy.local);
  const toolTests = results.filter((r) => r.sonnet.model && r.local.model).filter((r) => {
    const test = r as EvalResult & { _test?: TestCase };
    return test._test?.expectedTools?.length ? test._test.expectedTools.length > 0 : false;
  });
  const toolAccSonnet = results.filter((r) => r.toolAccuracy.sonnet).length / (results.filter((r) => !r.toolAccuracy.sonnet || r.toolAccuracy.sonnet).length || 1);
  const toolAccLocal = results.filter((r) => r.toolAccuracy.local).length / (results.filter((r) => !r.toolAccuracy.local || r.toolAccuracy.local).length || 1);

  let wins = { sonnet: 0, local: 0, tie: 0 };
  let avgScoreSonnet = 0;
  let avgScoreLocal = 0;
  if (!opts.noJudge) {
    for (const r of results) {
      if (r.judgeScores) {
        wins[r.judgeScores.winner]++;
        avgScoreSonnet += r.judgeScores.sonnet;
        avgScoreLocal += r.judgeScores.local;
      }
    }
    avgScoreSonnet /= (results.filter((r) => r.judgeScores).length || 1);
    avgScoreLocal /= (results.filter((r) => r.judgeScores).length || 1);
  }

  // Per-category breakdown
  const categories = [...new Set(results.map((r) => r.category))];
  const catRows = categories.map((cat) => {
    const catResults = results.filter((r) => r.category === cat);
    const avgS = catResults.reduce((s, r) => s + r.sonnet.latencyMs, 0) / catResults.length;
    const avgL = catResults.reduce((s, r) => s + r.local.latencyMs, 0) / catResults.length;
    const scoreS = !opts.noJudge
      ? (catResults.reduce((s, r) => s + (r.judgeScores?.sonnet ?? 0), 0) / catResults.length).toFixed(1)
      : "n/a";
    const scoreL = !opts.noJudge
      ? (catResults.reduce((s, r) => s + (r.judgeScores?.local ?? 0), 0) / catResults.length).toFixed(1)
      : "n/a";
    return `| ${cat} | ${Math.round(avgS)}ms | ${scoreS}/5 | ${Math.round(avgL)}ms | ${scoreL}/5 |`;
  });

  // Per-test table
  const testRows = results.map((r) => {
    const s = r.sonnet;
    const l = r.local;
    const sLabel = s.error ? "ERROR" : `${s.latencyMs}ms`;
    const lLabel = l.error ? "ERROR" : `${l.latencyMs}ms`;
    const winner = r.judgeScores?.winner ?? (s.error && !l.error ? "local" : !s.error && l.error ? "sonnet" : "?");
    const sScore = r.judgeScores ? `${r.judgeScores.sonnet}/5` : "";
    const lScore = r.judgeScores ? `${r.judgeScores.local}/5` : "";
    const sTools = s.toolCalls.length > 0 ? `[${s.toolCalls.map((t) => t.name).join(",")}]` : "";
    const lTools = l.toolCalls.length > 0 ? `[${l.toolCalls.map((t) => t.name).join(",")}]` : "";
    return `| ${r.testId} | ${r.category} | ${sLabel} ${sTools} ${sScore} | ${lLabel} ${lTools} ${lScore} | ${winner} |`;
  });

  // Readiness checklist
  const toolTestCount = results.filter((r) => r.category === "tool-selection").length;
  const toolTestPassed = results.filter((r) => r.category === "tool-selection" && r.toolAccuracy.local).length;
  const toolPct = toolTestCount > 0 ? Math.round((toolTestPassed / toolTestCount) * 100) : 0;
  const instTests = results.filter((r) => r.category === "instruction-following");
  const instScoreLocal = !opts.noJudge && instTests.length > 0
    ? instTests.reduce((s, r) => s + (r.judgeScores?.local ?? 0), 0) / instTests.length
    : null;
  const latencyDelta = avgLatencyLocal - avgLatencySonnet;

  const checkItem = (pass: boolean, text: string) => `- [${pass ? "x" : " "}] ${text}`;

  return `# Model Eval: Sonnet 4.6 vs Qwen3.5-35B-A3B (Local)

Date: ${ts} | Suite: ${opts.suite} | Tests: ${total} | Judge: ${opts.noJudge ? "disabled" : CONFIG.opusModelId}

---

## Summary

| Metric | Sonnet 4.6 | Local Qwen | Notes |
|--------|-----------|------------|-------|
| Avg latency | ${Math.round(avgLatencySonnet)}ms | ${Math.round(avgLatencyLocal)}ms | ${avgLatencyLocal < avgLatencySonnet ? "local is faster" : "sonnet is faster"} |
| Avg output tokens | ${Math.round(avgTokensOutSonnet)} | ${Math.round(avgTokensOutLocal)} | |
${!opts.noJudge ? `| Avg quality score | ${avgScoreSonnet.toFixed(2)}/5 | ${avgScoreLocal.toFixed(2)}/5 | Opus judge |\n| Wins / Ties / Losses | ${wins.sonnet} / ${wins.tie} / ${wins.local} | ${wins.local} / ${wins.tie} / ${wins.sonnet} | |\n` : ""}| Errors | ${results.filter((r) => r.sonnet.error).length} | ${results.filter((r) => r.local.error).length} | |

---

## By Category

| Category | Sonnet latency | Sonnet score | Local latency | Local score |
|----------|---------------|--------------|--------------|-------------|
${catRows.join("\n")}

---

## Orchestrator Readiness: Local Qwen

${checkItem(instScoreLocal !== null && instScoreLocal >= 4.0, `Adequate for instruction tasks (score ${instScoreLocal !== null ? instScoreLocal.toFixed(1) : "n/a"}/5 vs threshold 4.0)`)}
${checkItem(toolPct >= 85, `Adequate tool selection accuracy (${toolPct}% vs threshold 85%)`)}
${checkItem(Math.abs(latencyDelta) < 2000, `Latency within 2s of Sonnet (delta: ${latencyDelta > 0 ? "+" : ""}${Math.round(latencyDelta)}ms)`)}
${checkItem(results.filter((r) => r.local.error).length === 0, `Zero errors (${results.filter((r) => r.local.error).length} errors found)`)}

---

## Per-Test Results

| Test | Category | Sonnet | Local | Winner |
|------|----------|--------|-------|--------|
${testRows.join("\n")}

---

## Detailed Results

${results.map((r) => {
  const s = r.sonnet;
  const l = r.local;
  return `### ${r.testId} — ${r.category}

**Prompt:** ${r.prompt.length > 200 ? r.prompt.slice(0, 200) + "..." : r.prompt}

**Rubric:** ${r.rubric}

**Sonnet** (${s.latencyMs}ms, ${s.inputTokens}in/${s.outputTokens}out tokens${s.error ? ", ERROR" : ""}):
${s.error ? `> ERROR: ${s.error}` : (s.response || "[no text response]").split("\n").slice(0, 8).join("\n")}
${s.toolCalls.length > 0 ? `Tool calls: ${s.toolCalls.map((t) => `${t.name}(${JSON.stringify(t.args).slice(0, 80)})`).join(", ")}` : ""}

**Local Qwen** (${l.latencyMs}ms, ${l.inputTokens}in/${l.outputTokens}out tokens${l.error ? ", ERROR" : ""}):
${l.error ? `> ERROR: ${l.error}` : (l.response || "[no text response]").split("\n").slice(0, 8).join("\n")}
${l.toolCalls.length > 0 ? `Tool calls: ${l.toolCalls.map((t) => `${t.name}(${JSON.stringify(t.args).slice(0, 80)})`).join(", ")}` : ""}
${r.judgeScores ? `\n**Judge:** Sonnet ${r.judgeScores.sonnet}/5 vs Local ${r.judgeScores.local}/5 → **${r.judgeScores.winner}** — ${r.judgeScores.reasoning}` : ""}

---`;
}).join("\n\n")}
`;
}

// ─── Main ────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const opts = {
  suite: "orchestrator",
  noJudge: args.includes("--no-judge"),
  dryRun: args.includes("--dry-run"),
  category: (() => {
    const i = args.indexOf("--category");
    return i >= 0 ? args[i + 1] : undefined;
  })(),
};

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--suite" && args[i + 1]) opts.suite = args[i + 1];
}

async function main() {
  console.log(`${C.bold}OpenClaw Model Evaluator${C.reset}`);
  console.log(`${C.gray}Suite: ${opts.suite} | Judge: ${opts.noJudge ? "disabled" : CONFIG.opusModelId}${C.reset}\n`);

  // Validate OpenRouter API key
  if (!CONFIG.openrouterApiKey) {
    console.error(`${C.red}ERROR: OpenRouter API key not found in openclaw.json (models.providers.openrouter.apiKey)${C.reset}`);
    process.exit(1);
  }

  // Check local vLLM
  if (!opts.dryRun) {
    try {
      const check = await fetch(`${CONFIG.localBaseUrl}/models`, { signal: AbortSignal.timeout(3000) });
      if (!check.ok) throw new Error(`HTTP ${check.status}`);
      console.log(`${C.green}Local vLLM reachable${C.reset}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`${C.red}ERROR: Local vLLM not reachable at ${CONFIG.localBaseUrl}: ${msg}${C.reset}`);
      console.error(`${C.yellow}Is the local model server running?${C.reset}`);
      process.exit(1);
    }
  }

  // Load suite
  const suitePath = join(__dirname, "eval-suites", `${opts.suite}.json`);
  let suite: TestSuite;
  try {
    suite = JSON.parse(readFileSync(suitePath, "utf8"));
  } catch {
    console.error(`${C.red}Suite not found: ${suitePath}${C.reset}`);
    process.exit(1);
  }

  let tests = suite.tests;
  if (opts.category) {
    tests = tests.filter((t) => t.category === opts.category);
    if (tests.length === 0) {
      console.error(`${C.red}No tests found for category: ${opts.category}${C.reset}`);
      process.exit(1);
    }
  }

  console.log(`${C.cyan}Running ${tests.length} tests...${C.reset}\n`);

  if (opts.dryRun) {
    console.log(`${C.yellow}DRY RUN — test cases:${C.reset}`);
    tests.forEach((t) => console.log(`  ${t.id} [${t.category}]: ${t.prompt.slice(0, 60)}...`));
    process.exit(0);
  }

  const results: EvalResult[] = [];

  for (let i = 0; i < tests.length; i++) {
    const test = tests[i];
    process.stdout.write(
      `${C.gray}[${i + 1}/${tests.length}]${C.reset} ${test.category}/${test.id} ... `,
    );

    // Run both models in parallel
    const [sonnetResult, localResult] = await Promise.all([
      runModel(callSonnet, test, "sonnet"),
      runModel(callLocal, test, "local"),
    ]);

    const toolAccuracy = {
      sonnet: checkToolAccuracy(sonnetResult, test.expectedTools),
      local: checkToolAccuracy(localResult, test.expectedTools),
    };

    // Judge
    let judgeScores: JudgeScores | undefined;
    if (!opts.noJudge) {
      try {
        judgeScores = await judgeResponses(test, sonnetResult, localResult);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        process.stdout.write(`${C.yellow}[judge failed: ${msg.slice(0, 40)}]${C.reset} `);
      }
    }

    const winner = judgeScores?.winner ?? (sonnetResult.error && !localResult.error ? "local" : !sonnetResult.error && localResult.error ? "sonnet" : "?");
    const winnerColor = winner === "sonnet" ? C.blue : winner === "local" ? C.green : C.gray;

    console.log(
      `${C.blue}sonnet: ${sonnetResult.error ? "ERR" : sonnetResult.latencyMs + "ms"}${C.reset} | ` +
      `${C.green}local: ${localResult.error ? "ERR" : localResult.latencyMs + "ms"}${C.reset} | ` +
      `${winnerColor}winner: ${winner}${C.reset}` +
      (judgeScores ? ` (${judgeScores.sonnet.toFixed(0)} vs ${judgeScores.local.toFixed(0)})` : ""),
    );

    const result: EvalResult & { _test?: TestCase } = {
      testId: test.id,
      category: test.category,
      prompt: test.prompt,
      rubric: test.rubric,
      sonnet: sonnetResult,
      local: localResult,
      toolAccuracy,
      judgeScores,
      _test: test,
    };
    results.push(result);
  }

  // Generate report
  console.log(`\n${C.bold}Generating report...${C.reset}`);
  const report = generateReport(results, opts);

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const docsDir = join(ROOT, "docs");
  mkdirSync(docsDir, { recursive: true });
  const mdPath = join(docsDir, `eval-${timestamp}.md`);
  const jsonPath = join(docsDir, `eval-${timestamp}.json`);

  writeFileSync(mdPath, report, "utf8");
  writeFileSync(jsonPath, JSON.stringify({ suite: opts.suite, timestamp, results }, null, 2), "utf8");

  console.log(`${C.green}Report saved:${C.reset}`);
  console.log(`  ${C.bold}${mdPath}${C.reset}`);
  console.log(`  ${C.gray}${jsonPath}${C.reset}`);

  // Print quick summary
  const validR = results.filter((r) => !r.sonnet.error && !r.local.error);
  if (validR.length > 0) {
    const avgS = Math.round(validR.reduce((s, r) => s + r.sonnet.latencyMs, 0) / validR.length);
    const avgL = Math.round(validR.reduce((s, r) => s + r.local.latencyMs, 0) / validR.length);
    console.log(`\n${C.bold}Quick summary:${C.reset}`);
    console.log(`  Sonnet avg latency: ${C.blue}${avgS}ms${C.reset}`);
    console.log(`  Local avg latency:  ${C.green}${avgL}ms${C.reset}`);
    if (!opts.noJudge) {
      const judged = results.filter((r) => r.judgeScores);
      if (judged.length > 0) {
        const avgQS = (judged.reduce((s, r) => s + (r.judgeScores?.sonnet ?? 0), 0) / judged.length).toFixed(2);
        const avgQL = (judged.reduce((s, r) => s + (r.judgeScores?.local ?? 0), 0) / judged.length).toFixed(2);
        const wins = { sonnet: 0, local: 0, tie: 0 };
        judged.forEach((r) => wins[r.judgeScores!.winner]++);
        console.log(`  Sonnet avg quality: ${C.blue}${avgQS}/5${C.reset}`);
        console.log(`  Local avg quality:  ${C.green}${avgQL}/5${C.reset}`);
        console.log(`  Wins: sonnet ${wins.sonnet} | local ${wins.local} | tie ${wins.tie}`);
      }
    }
  }
}

main().catch((err) => {
  console.error(`${C.red}Fatal error:${C.reset}`, err);
  process.exit(1);
});
