#!/usr/bin/env node
/**
 * chat-startup-analysis.js — Timing & content analysis for new-chat startup runs.
 *
 * Correlates timing.jsonl, routing.jsonl, and session transcripts to show
 * exactly what happens during each new-chat run: routing decisions, prefill
 * inflation, token counts, and wall-time breakdown.
 *
 * Usage:
 *   node tools/chat-startup-analysis.js [--last N] [--session ID] [--all]
 */

const fs = require("fs");
const path = require("path");

const ROOT = path.join(process.env.HOME, ".openclaw");
const TIMING_LOG = path.join(ROOT, "logs/timing.jsonl");
const ROUTING_LOG = path.join(ROOT, "logs/routing.jsonl");
const SESSIONS_DIR = path.join(ROOT, "agents/main/sessions");

// ANSI colors
const C = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  bold: "\x1b[1m",
  cyan: "\x1b[36m",
  yellow: "\x1b[33m",
  green: "\x1b[32m",
  red: "\x1b[31m",
  magenta: "\x1b[35m",
  white: "\x1b[37m",
};

// ── Parse args ────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
let lastN = 10;
let filterSessionId = null;
let showAll = false;

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--last" && args[i + 1]) lastN = parseInt(args[++i], 10);
  else if (args[i] === "--session" && args[i + 1]) filterSessionId = args[++i];
  else if (args[i] === "--all") showAll = true;
  else if (args[i] === "--help" || args[i] === "-h") {
    console.log("Usage: node tools/chat-startup-analysis.js [--last N] [--session ID] [--all]");
    console.log("  --last N       Show last N new-chat runs (default: 10)");
    console.log("  --session ID   Analyze a specific session");
    console.log("  --all          Show all runs, not just startup/greeting runs");
    process.exit(0);
  }
}

// ── Read JSONL helper ─────────────────────────────────────────────────────

function readJsonl(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const lines = fs.readFileSync(filePath, "utf-8").split("\n").filter(Boolean);
  const results = [];
  for (const line of lines) {
    try { results.push(JSON.parse(line)); } catch {}
  }
  return results;
}

// ── Normalize timestamp to UTC ms ─────────────────────────────────────────

function toUtcMs(ts) {
  if (!ts) return 0;
  // JS Date() treats naive timestamps (no Z suffix) as local time,
  // which is correct since Python memory_prefill writes local time too.
  return new Date(ts).getTime();
}

function fmtTime(ts) {
  const d = new Date(ts);
  return d.toISOString().replace("T", " ").replace(/\.\d+Z$/, "");
}

function fmtMs(ms) {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${ms}ms`;
}

// ── Load data ─────────────────────────────────────────────────────────────

console.log(`${C.dim}Loading timing log...${C.reset}`);
const timingEntries = readJsonl(TIMING_LOG);

console.log(`${C.dim}Loading routing log...${C.reset}`);
const routingEntries = readJsonl(ROUTING_LOG);

// ── Extract run_end and memory_prefill events ─────────────────────────────

const runEnds = timingEntries
  .filter((e) => e.event === "run_end")
  .map((e) => ({ ...e, _utcMs: toUtcMs(e.ts) }));

const prefills = timingEntries
  .filter((e) => e.event === "memory_prefill")
  .map((e) => ({ ...e, _utcMs: toUtcMs(e.ts) }));

const routings = routingEntries.map((e) => ({ ...e, _utcMs: toUtcMs(e.ts) }));

// ── Identify startup/greeting runs ────────────────────────────────────────
// Criteria: roundTrips == 0 AND totalMs > 1000 (exclude fast heartbeats)
// OR: any run on a session if --all or --session is specified

let targetRuns;
if (showAll) {
  targetRuns = runEnds;
} else if (filterSessionId) {
  targetRuns = runEnds.filter((r) => r.sessionId === filterSessionId);
} else {
  targetRuns = runEnds.filter(
    (r) => r.roundTrips === 0 && r.totalMs > 1000
  );
}

// Sort by time descending, take last N
targetRuns.sort((a, b) => b._utcMs - a._utcMs);
targetRuns = targetRuns.slice(0, lastN);
targetRuns.reverse(); // show oldest first

console.log(
  `\n${C.bold}Found ${targetRuns.length} ${showAll ? "total" : "startup/greeting"} runs${C.reset}\n`
);

// ── Find session files ────────────────────────────────────────────────────

function findSessionFile(sessionId) {
  const files = fs.readdirSync(SESSIONS_DIR);
  // Prefer the live .jsonl, fall back to .jsonl.reset.*
  const live = files.find((f) => f === `${sessionId}.jsonl`);
  if (live) return path.join(SESSIONS_DIR, live);
  const archived = files.find((f) => f.startsWith(`${sessionId}.jsonl`));
  if (archived) return path.join(SESSIONS_DIR, archived);
  return null;
}

// Cache parsed session files to avoid re-reading
const sessionCache = new Map();

function readSessionMessages(sessionId, runStartMs, runEndMs) {
  const file = findSessionFile(sessionId);
  if (!file) return null;

  if (!sessionCache.has(file)) {
    sessionCache.set(file, readJsonl(file));
  }
  const lines = sessionCache.get(file);

  // Find the user+assistant message pair closest to the run's time window.
  // The run spans [runStartMs, runEndMs]. The user message should be near
  // runStartMs and the assistant response near runEndMs.
  const messages = lines.filter(
    (e) => e.type === "message" && e.message?.role
  );

  let bestUser = null;
  let bestAssistant = null;
  let bestDist = Infinity;

  for (let i = 0; i < messages.length; i++) {
    const entry = messages[i];
    if (entry.message.role !== "user") continue;
    const entryMs = new Date(entry.timestamp).getTime();
    const dist = Math.abs(entryMs - runStartMs);
    // Must be within 5s of run start
    if (dist < 5000 && dist < bestDist) {
      bestDist = dist;
      bestUser = entry;
      // Look for the next assistant message
      for (let j = i + 1; j < messages.length; j++) {
        if (messages[j].message.role === "assistant") {
          bestAssistant = messages[j];
          break;
        }
      }
    }
  }

  // Fallback: if no timestamp match, use first user+assistant pair
  if (!bestUser) {
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].message.role === "user") {
        bestUser = messages[i];
        for (let j = i + 1; j < messages.length; j++) {
          if (messages[j].message.role === "assistant") {
            bestAssistant = messages[j];
            break;
          }
        }
        break;
      }
    }
  }

  return { firstUser: bestUser, firstAssistant: bestAssistant };
}

// ── Correlate and display ─────────────────────────────────────────────────

for (const run of targetRuns) {
  const runStartMs = run._utcMs - run.totalMs;
  const isSessionInit = run.runId === run.sessionId;

  // Find closest routing decision (within ±3s of run start)
  const routing = routings.find(
    (r) => Math.abs(r._utcMs - runStartMs) < 3000
  );

  // Find memory_prefill for this session near run start (within ±10s)
  const prefill = prefills.find(
    (p) =>
      p.session_id === run.sessionId &&
      Math.abs(p._utcMs - runStartMs) < 10000
  );

  // Read session transcript — match by run timestamp
  const session = readSessionMessages(run.sessionId, runStartMs, run._utcMs);

  // ── Format output ───────────────────────────────────────────────────

  const shortId = run.sessionId?.slice(0, 8) ?? "unknown";
  const label = isSessionInit ? "session-init" : "run";

  console.log(
    `${C.bold}${C.cyan}── ${label} ${shortId} (${fmtTime(run._utcMs)}) ${"─".repeat(40)}${C.reset}`
  );

  // Routing
  if (routing) {
    const tierColor = {
      local: C.yellow,
      haiku: C.green,
      sonnet: C.cyan,
      opus: C.magenta,
    }[routing.tier] ?? C.white;

    console.log(
      `  ${C.dim}Route:${C.reset}    ${tierColor}${routing.tier}${C.reset} via ${routing.reason}`
    );
    console.log(
      `  ${C.dim}          promptLength=${routing.promptLength}, sessionDepth=${routing.sessionDepth}${C.reset}`
    );
  } else {
    console.log(`  ${C.dim}Route:${C.reset}    ${C.red}(no routing entry found)${C.reset}`);
  }

  // Prefill
  if (prefill) {
    console.log(
      `  ${C.dim}Prefill:${C.reset}  ${fmtMs(prefill.duration_ms)} — ` +
      `${prefill.context_chars} chars, ${prefill.vector_results ?? 0} vector results, ` +
      `${prefill.tag_docs ?? 0} tag docs, ${prefill.glm_keywords ?? 0} GLM keywords`
    );
    console.log(
      `  ${C.dim}          tier1=${prefill.tier1_count ?? 0}, tier2=${prefill.tier2_count ?? 0}${C.reset}`
    );
  } else {
    console.log(
      `  ${C.dim}Prefill:${C.reset}  ${C.red}(no prefill event found — timed out or skipped)${C.reset}`
    );
  }

  // Session content
  if (session?.firstUser) {
    const userMsg = session.firstUser.message;
    const textContent = userMsg.content
      ?.filter((c) => c.type === "text")
      .map((c) => c.text)
      .join("") ?? "";
    const hasMemoryContext = textContent.includes("<memory_context>");
    const isGreeting = /session was started via \/new|Session Startup sequence/i.test(textContent);
    const isCron = /\[cron:/.test(textContent);
    const promptChars = textContent.length;

    let promptType = "unknown";
    if (isGreeting) promptType = "greeting";
    else if (isCron) promptType = "cron-hook";
    else promptType = "user-message";

    console.log(
      `  ${C.dim}Prompt:${C.reset}   ${promptType} (${promptChars} chars)` +
      (hasMemoryContext
        ? ` ${C.yellow}+ memory_context prepended${C.reset}`
        : "")
    );

    if (session.firstAssistant) {
      const assistMsg = session.firstAssistant.message;
      const usage = assistMsg.usage ?? {};
      const model = assistMsg.model ?? "unknown";
      const provider = assistMsg.provider ?? "unknown";
      const outputText = assistMsg.content
        ?.filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("") ?? "";
      const outputChars = outputText.length;

      console.log(
        `  ${C.dim}Model:${C.reset}    ${provider}/${model}`
      );
      console.log(
        `  ${C.dim}Tokens:${C.reset}   input=${C.bold}${(usage.input ?? 0).toLocaleString()}${C.reset}, ` +
        `output=${(usage.output ?? 0).toLocaleString()} (${outputChars} chars)`
      );

      // Token breakdown: estimate system prompt vs user message
      const userTokensEst = Math.ceil(promptChars / 4); // rough char→token estimate
      const systemTokensEst = (usage.input ?? 0) - userTokensEst;
      if (systemTokensEst > 1000) {
        console.log(
          `  ${C.dim}Breakdown:${C.reset} ~${systemTokensEst.toLocaleString()} system prompt tokens + ~${userTokensEst.toLocaleString()} user tokens`
        );
      }

      if (hasMemoryContext) {
        const ctxMatch = textContent.match(/<memory_context>([\s\S]*?)<\/memory_context>/);
        const ctxChars = ctxMatch ? ctxMatch[1].length : 0;
        const originalChars = promptChars - ctxChars - "<memory_context></memory_context>".length;
        console.log(
          `  ${C.dim}Prefill:${C.reset}  +${ctxChars} chars prepended to user message ` +
          `${C.yellow}(${originalChars} → ${promptChars} chars, ${(promptChars / Math.max(originalChars, 1)).toFixed(1)}x)${C.reset}`
        );
      }
    } else {
      console.log(`  ${C.dim}Response:${C.reset} ${C.red}(no assistant response found)${C.reset}`);
    }
  } else {
    console.log(`  ${C.dim}Session:${C.reset}  ${C.red}(transcript not found)${C.reset}`);
  }

  // Compute true wall time from session transcript timestamps
  let trueWallMs = null;
  let sessionCreateMs = null;
  if (session?.firstUser && session?.firstAssistant) {
    const userTs = new Date(session.firstUser.timestamp).getTime();
    const assistTs = new Date(session.firstAssistant.timestamp).getTime();
    const msgToResponse = assistTs - userTs;

    // Check session creation time — only meaningful if session was just created
    const sessionFile = findSessionFile(run.sessionId);
    if (sessionFile && sessionCache.has(sessionFile)) {
      const entries = sessionCache.get(sessionFile);
      const sessionEntry = entries.find((e) => e.type === "session");
      if (sessionEntry?.timestamp) {
        sessionCreateMs = new Date(sessionEntry.timestamp).getTime();
      }
    }

    // Only show session→response gap if session was created within 30s of this run
    const isFirstRun = sessionCreateMs && (userTs - sessionCreateMs) < 30000;
    const fromCreate = isFirstRun ? assistTs - sessionCreateMs : null;
    trueWallMs = fromCreate ?? msgToResponse;

    let parts = [];
    if (fromCreate != null) {
      const createColor = fromCreate > 5000 ? C.red : fromCreate > 2000 ? C.yellow : C.green;
      parts.push(`session create → response: ${createColor}${C.bold}${fmtMs(fromCreate)}${C.reset}`);
    }
    parts.push(`message → response: ${fmtMs(msgToResponse)}`);
    console.log(`  ${C.dim}Transcript:${C.reset} ${parts.join(", ")}`);
  }

  // Timing profiler summary (note: totalMs starts from llm_input, AFTER hooks)
  const wallColor = run.totalMs > 5000 ? C.red : run.totalMs > 2000 ? C.yellow : C.green;
  console.log(
    `  ${C.dim}Profiler:${C.reset} ${wallColor}${fmtMs(run.totalMs)}${C.reset} ` +
    `(llm=${fmtMs(run.llmMs ?? 0)}, tool=${fmtMs(run.toolMs ?? 0)}, ` +
    `overhead=${fmtMs(run.overheadMs ?? 0)}, roundTrips=${run.roundTrips ?? 0})`
  );
  if (run.totalMs < (trueWallMs ?? 0) - 500) {
    const hookTime = (trueWallMs ?? 0) - run.totalMs;
    console.log(
      `  ${C.dim}          profiler misses ~${fmtMs(hookTime)} of pre-LLM hook time (starts after llm_input)${C.reset}`
    );
  }

  // Timeline reconstruction
  // Note: before_agent_start and before_prompt_build run BEFORE llm_input,
  // so they're not captured in totalMs. The timeline shows the true sequence.
  const phases = [];
  let hookTimeTotal = 0;

  if (routing) {
    phases.push({ at: routing.latencyMs, label: `before_agent_start: model-router → ${routing.tier} (${routing.reason})` });
    hookTimeTotal += routing.latencyMs;
  }

  if (prefill) {
    const prefillEnd = hookTimeTotal + prefill.duration_ms;
    phases.push({ at: prefillEnd, label: `before_prompt_build: prefill_context (${fmtMs(prefill.duration_ms)}, ${prefill.context_chars} chars)` });
    hookTimeTotal += prefill.duration_ms;
  }

  // llm_input fires here — profiler starts tracking totalMs
  phases.push({ at: hookTimeTotal, label: `llm_input (profiler starts here)`, dim: true });

  if (session?.firstAssistant) {
    const usage = session.firstAssistant.message?.usage ?? {};
    if (usage.input > 0) {
      phases.push({
        at: hookTimeTotal + run.totalMs,
        label: `LLM response (${(usage.input ?? 0).toLocaleString()} in → ${(usage.output ?? 0).toLocaleString()} out)`,
      });
    }
  }

  const endAt = hookTimeTotal + run.totalMs;
  if (phases.length > 1) {
    console.log(`  ${C.dim}Timeline (estimated):${C.reset}`);
    console.log(`    ${C.dim}0ms${C.reset}      chat.send received`);
    for (const p of phases) {
      const style = p.dim ? C.dim : "";
      console.log(`    ${C.dim}~${fmtMs(p.at)}${C.reset}   ${style}${p.label}${p.dim ? C.reset : ""}`);
    }
    console.log(`    ${C.dim}~${fmtMs(endAt)}${C.reset}   agent_end → response delivered`);
  }

  // Warnings
  if (run.roundTrips === 0 && run.totalMs > 1000 && session?.firstAssistant) {
    console.log(
      `  ${C.yellow}⚠ llm_output hook missing for ${session.firstAssistant.message?.provider ?? "local"} — ` +
      `LLM time misattributed as overhead${C.reset}`
    );
  }

  console.log();
}

// ── Summary stats ─────────────────────────────────────────────────────────

if (targetRuns.length > 1) {
  const times = targetRuns.map((r) => r.totalMs);
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const max = Math.max(...times);
  const min = Math.min(...times);

  console.log(`${C.bold}Summary (${targetRuns.length} runs)${C.reset}`);
  console.log(`  avg: ${fmtMs(Math.round(avg))}  min: ${fmtMs(min)}  max: ${fmtMs(max)}`);
}
