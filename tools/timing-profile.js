#!/usr/bin/env node
// timing-profile.js — OpenClaw interaction timing analyzer
// Reads session JSONL transcripts and shows a visual breakdown of where time goes.
//
// Usage:
//   node timing-profile.js                     # last 5 interactions in most recent session
//   node timing-profile.js --session <uuid>    # specific session
//   node timing-profile.js --interactions 10   # show last 10 interactions
//   node timing-profile.js --all-sessions      # all session files
//   node timing-profile.js --json              # JSON output

import { createReadStream, readdirSync, statSync } from 'fs';
import { join } from 'path';
import { createInterface } from 'readline';

const SESSIONS_DIR = '/home/alansrobotlab/.openclaw/agents/main/sessions';
const BAR_WIDTH = 24;

// ANSI colors
const C = {
  reset:   '\x1b[0m',
  bold:    '\x1b[1m',
  dim:     '\x1b[2m',
  red:     '\x1b[31m',
  green:   '\x1b[32m',
  yellow:  '\x1b[33m',
  blue:    '\x1b[34m',
  magenta: '\x1b[35m',
  cyan:    '\x1b[36m',
  gray:    '\x1b[90m',
};

// Parse CLI args
const args = process.argv.slice(2);
const opts = { session: null, interactions: 5, json: false, allSessions: false };
for (let i = 0; i < args.length; i++) {
  if      (args[i] === '--session'      && args[i+1]) opts.session     = args[++i];
  else if (args[i] === '--interactions' && args[i+1]) opts.interactions = parseInt(args[++i]);
  else if (args[i] === '--json')                      opts.json        = true;
  else if (args[i] === '--all-sessions')              opts.allSessions = true;
  else if (!args[i].startsWith('--'))                 opts.session     = args[i]; // positional = session uuid
}

// ─── File discovery ──────────────────────────────────────────────────────────

function getSessionFiles() {
  let files;
  try {
    files = readdirSync(SESSIONS_DIR);
  } catch {
    console.error(`Cannot read sessions dir: ${SESSIONS_DIR}`);
    process.exit(1);
  }

  // When a specific session is requested, also search reset archives
  const includeReset = !!opts.session;

  const jsonlFiles = files
    .filter(f => f.includes('.jsonl') && (includeReset || !f.includes('.reset.')))
    .map(f => {
      // live:  <uuid>.jsonl
      // reset: <uuid>.jsonl.reset.2026-02-22T02-23-54.279Z
      const isReset = f.includes('.reset.');
      const uuid = f.replace(/\.jsonl.*$/, '');
      const resetTs = isReset ? f.match(/\.reset\.(.+)/)?.[1]?.slice(0, 19) : null;
      return {
        path: join(SESSIONS_DIR, f),
        uuid,
        label: isReset ? `${uuid} (reset ${resetTs})` : uuid,
        mtime: statSync(join(SESSIONS_DIR, f)).mtimeMs,
      };
    })
    .sort((a, b) => b.mtime - a.mtime);

  if (opts.session) {
    const matched = jsonlFiles.filter(f => f.uuid.startsWith(opts.session));
    if (!matched.length) { console.error(`No session matching: ${opts.session}`); process.exit(1); }
    return matched;
  }
  if (opts.allSessions) return jsonlFiles;
  return jsonlFiles.slice(0, 1);
}

// ─── JSONL parsing ────────────────────────────────────────────────────────────

async function parseJsonl(filePath) {
  const records = [];
  const rl = createInterface({ input: createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    try { records.push(JSON.parse(line)); } catch { /* skip */ }
  }
  return records;
}

// ─── Interaction extraction ───────────────────────────────────────────────────
// Each record has { id, parentId, timestamp (ISO), type, message? }
// Messages are a linear linked list via parentId.
// We find user messages, walk forward to find the full interaction chain.

function extractInteractions(records) {
  // Build forward-child map (all record types, not just messages)
  const childrenOf = new Map();
  for (const r of records) {
    if (!r.parentId) continue;
    if (!childrenOf.has(r.parentId)) childrenOf.set(r.parentId, []);
    childrenOf.get(r.parentId).push(r);
  }

  // Find user messages (real messages, not things injected by system hooks at session start)
  // We include ALL user messages to get full coverage — system ones labeled separately
  const userMessages = records.filter(r =>
    r.type === 'message' &&
    r.message?.role === 'user'
  );

  const interactions = [];

  for (const userMsg of userMessages) {
    // Walk the linked list forward from this user message
    const chain = []; // only message-type records
    let cur = userMsg;
    let found_stop = false;
    const visited = new Set();

    while (cur && !visited.has(cur.id)) {
      visited.add(cur.id);
      if (cur.type === 'message') chain.push(cur);

      // Find child records in timestamp order
      const children = (childrenOf.get(cur.id) || [])
        .slice()
        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

      // Check if any message child is a final assistant stop
      const assistantStop = children.find(c =>
        c.type === 'message' &&
        c.message?.role === 'assistant' &&
        (c.message?.stopReason === 'stop' || c.message?.stopReason === 'endTurn')
      );

      if (assistantStop) {
        chain.push(assistantStop);
        found_stop = true;
        break;
      }

      // Move to first child (follow the linear chain)
      const nextChild = children[0];
      if (!nextChild) break;
      cur = nextChild;
    }

    if (chain.length >= 2 && found_stop) {
      interactions.push(chain);
    }
  }

  return interactions;
}

// ─── Tool result summarizer ───────────────────────────────────────────────────

function extractToolResultSummary(msg) {
  const toolName = msg.toolName || '';
  const text = (msg.content || []).find(c => c.type === 'text')?.text || '';
  const details = msg.details || {};

  if (toolName === 'mem_search') {
    try {
      const parsed = JSON.parse(text);
      const n = parsed.results?.length ?? 0;
      if (n === 0) return { ok: false, text: '0 hits' };
      const names = parsed.results.slice(0, 2).map(r => (r.path || '').split('/').pop());
      return { ok: true, text: `${n} hit${n > 1 ? 's' : ''}: ${names.join(', ')}` };
    } catch {}
  }

  if (toolName === 'mem_get') {
    try {
      const parsed = JSON.parse(text);
      if (!parsed.text) return { ok: false, text: 'empty' };
      return { ok: true, text: parsed.text.slice(0, 60).replace(/\n/g, ' ') };
    } catch {}
  }

  if (toolName === 'exec' || toolName === 'process') {
    const output = (details.aggregated || text || '').trim();
    if (details.exitCode != null && details.exitCode !== 0) {
      return { ok: false, text: `exit ${details.exitCode}: ${output.slice(0, 60)}` };
    }
    const lines = output.split('\n').filter(Boolean);
    const preview = (lines[0] || '').slice(0, 70);
    return { ok: true, text: lines.length > 1 ? `${preview}  (+${lines.length - 1} lines)` : preview };
  }

  if (toolName === 'read') {
    if (details.status === 'error' || details.error) {
      return { ok: false, text: (details.error || '').slice(0, 80) };
    }
    try {
      const parsed = JSON.parse(text);
      if (parsed.status === 'error') return { ok: false, text: (parsed.error || '').slice(0, 80) };
    } catch {}
    const firstLine = (text.split('\n')[0] || '').slice(0, 70);
    return { ok: true, text: firstLine || `${text.length} bytes` };
  }

  if (toolName === 'http_fetch' || toolName === 'http_search') {
    if (msg.isError) return { ok: false, text: text.slice(0, 80).replace(/\n/g, ' ') };
    const lines = text.split('\n').filter(Boolean);
    return { ok: true, text: (lines[0] || '').slice(0, 70) + (lines.length > 1 ? ` (+${lines.length - 1})` : '') };
  }

  if (msg.isError) return { ok: false, text: text.slice(0, 80).replace(/\n/g, ' ') };
  return { ok: true, text: text.slice(0, 80).replace(/\n/g, ' ') };
}

// ─── Timing analysis ──────────────────────────────────────────────────────────

function analyzeInteraction(chain) {
  const userMsg = chain[0];
  const finalMsg = chain[chain.length - 1];
  const totalMs = new Date(finalMsg.timestamp) - new Date(userMsg.timestamp);

  // Detect if this is a system-initiated interaction
  const firstText = userMsg.message?.content?.[0]?.text ?? '';
  const isSystem = firstText.startsWith('System:') || firstText.startsWith('A new session');
  // Strip conversation metadata prefix (webchat wraps every message with it)
  const stripped = firstText
    .replace(/^Conversation info.*?```\n\n/s, '')  // strip metadata block
    .replace(/^\[\w+ \d{4}-\d{2}-\d{2} \d{2}:\d{2} \w+\] /, '') // strip timestamp prefix
    .trim();
  const label = isSystem ? 'system/startup' : stripped.slice(0, 70).replace(/\n/g, ' ');

  // Build timeline segments
  const segments = []; // { kind: 'llm'|'tool', label, wallMs, cumulMs, actualMs, parallel }
  let roundTrip = 0;
  let prevT = new Date(userMsg.timestamp).getTime();
  let cumulMs = 0;
  let i = 1; // skip user message

  while (i < chain.length) {
    const rec = chain[i];
    const recT = new Date(rec.timestamp).getTime();
    const msg = rec.message;

    if (msg?.role === 'assistant') {
      // LLM round trip: from prev anchor to this assistant being written
      const llmMs = recT - prevT;
      roundTrip++;
      cumulMs += llmMs;
      const isStop = msg.stopReason === 'stop' || msg.stopReason === 'endTurn';

      // Count tool calls requested
      const toolCalls = (msg.content || []).filter(c => c.type === 'toolCall').map(c => c.name);

      // Token usage
      const usage = msg.usage || {};

      segments.push({
        kind: 'llm',
        round: roundTrip,
        wallMs: llmMs,
        cumulMs,
        toolCalls,
        isStop,
        model: msg.model,
        tokens: {
          input: usage.input || 0,
          output: usage.output || 0,
          cacheRead: usage.cacheRead || 0,
          cacheWrite: usage.cacheWrite || 0,
        },
        cost: usage.cost?.total || 0,
      });

      prevT = recT;
      i++;

    } else if (msg?.role === 'toolResult') {
      // Tool execution: from prev record (assistant or prior toolResult) to this result
      const wallMs = recT - prevT;
      cumulMs += wallMs;
      const details = msg.details || {};
      const actualMs = (typeof details.durationMs === 'number') ? details.durationMs : null;
      const resultSummary = extractToolResultSummary(msg);

      segments.push({
        kind: 'tool',
        toolName: msg.toolName || '?',
        wallMs,
        cumulMs,
        actualMs,
        overheadMs: actualMs !== null ? wallMs - actualMs : null,
        isError: msg.isError || false,
        resultSummary,
      });

      prevT = recT;
      i++;
    } else {
      i++;
    }
  }

  // Totals
  const llmSegments  = segments.filter(s => s.kind === 'llm');
  const toolSegments = segments.filter(s => s.kind === 'tool');
  const totalLlmMs   = llmSegments.reduce((sum, s) => sum + s.wallMs, 0);
  const totalToolMs  = toolSegments.reduce((sum, s) => sum + s.wallMs, 0);
  const overheadMs   = totalMs - totalLlmMs - totalToolMs;

  const ts = new Date(userMsg.timestamp);
  const timeStr = ts.toISOString().replace('T', ' ').replace(/\.\d{3}Z$/, ' UTC');

  return {
    timeStr,
    label,
    isSystem,
    totalMs,
    totalLlmMs,
    totalToolMs,
    overheadMs,
    roundTrips: roundTrip,
    segments,
    totalCost: llmSegments.reduce((sum, s) => sum + s.cost, 0),
  };
}

// ─── Visual rendering ─────────────────────────────────────────────────────────

function bar(ms, totalMs, color) {
  const fill = Math.max(1, Math.round((ms / totalMs) * BAR_WIDTH));
  const empty = BAR_WIDTH - fill;
  return color + '█'.repeat(fill) + C.gray + '░'.repeat(empty) + C.reset;
}

function fmtMs(ms) {
  if (ms >= 10000) return `${(ms/1000).toFixed(1)}s`;
  if (ms >= 1000)  return `${(ms/1000).toFixed(2)}s`;
  return `${ms}ms`;
}

function fmtPct(ms, total) {
  return `${((ms / total) * 100).toFixed(1)}%`;
}

function fmtCost(c) {
  return c > 0 ? `$${c.toFixed(4)}` : '';
}

function fmtCumul(ms) {
  // compact cumulative time for the Σ prefix column
  if (ms >= 60000) return `${Math.floor(ms/60000)}m${((ms%60000)/1000).toFixed(0).padStart(2,'0')}s`;
  if (ms >= 10000) return `${(ms/1000).toFixed(1)}s`;
  if (ms >= 1000)  return `${(ms/1000).toFixed(2)}s`;
  return `${ms}ms`;
}

function printInteraction(ix, n) {
  const { timeStr, label, totalMs, totalLlmMs, totalToolMs, overheadMs, roundTrips, segments, totalCost } = ix;
  const line = '━'.repeat(84);

  console.log(`\n${C.bold}${line}${C.reset}`);
  console.log(`${C.bold}${timeStr}${C.reset}  total: ${C.yellow}${fmtMs(totalMs)}${C.reset}  ${C.dim}${label}${C.reset}`);
  console.log(line);

  for (const seg of segments) {
    // Cumulative time prefix: "Σ 6.59s" right-aligned in 9 chars
    const cumulStr = `${C.gray}Σ${fmtCumul(seg.cumulMs).padStart(7)}${C.reset}`;

    if (seg.kind === 'llm') {
      const b = bar(seg.wallMs, totalMs, C.blue);
      const toolStr = seg.toolCalls.length ? C.dim + ` → ${seg.toolCalls.join(', ')}` + C.reset : '';
      const stopStr = seg.isStop ? C.green + ' ⏹ stop' + C.reset : C.cyan + ' ⟳ toolUse' + C.reset;
      const tokStr  = seg.tokens.output
        ? C.gray + `  (${seg.tokens.output} out` +
          (seg.tokens.cacheRead ? `, ${(seg.tokens.cacheRead/1000).toFixed(0)}k $hit` : '') +
          (seg.tokens.cacheWrite ? `, ${seg.tokens.cacheWrite} $write` : '') +
          ')' + C.reset
        : '';
      const costStr = seg.cost ? C.gray + ` ${fmtCost(seg.cost)}` + C.reset : '';
      console.log(`  ${cumulStr}  ${C.blue}LLM  #${seg.round}${C.reset}  ${b}  ${C.yellow}${fmtMs(seg.wallMs).padStart(7)}${C.reset}  ${fmtPct(seg.wallMs, totalMs).padStart(5)}${stopStr}${toolStr}${tokStr}${costStr}`);
    } else {
      const b = bar(seg.wallMs, totalMs, C.magenta);
      // Execution detail (actual vs wall for exec/process)
      let execDetail = '';
      if (seg.actualMs !== null) {
        execDetail = seg.overheadMs > 5
          ? C.gray + ` (${fmtMs(seg.actualMs)} actual + ${fmtMs(seg.overheadMs)} overhead)` + C.reset
          : C.gray + ` (${fmtMs(seg.actualMs)} actual)` + C.reset;
      }
      // Result summary
      let resultStr = '';
      if (seg.resultSummary) {
        const rs = seg.resultSummary;
        if (rs.ok) {
          resultStr = C.gray + `  → ${rs.text}` + C.reset;
        } else {
          resultStr = C.red + `  ⚠ ${rs.text}` + C.reset;
        }
      }
      const errStr = seg.isError ? C.red + ' ERR' + C.reset : '';
      console.log(`  ${cumulStr}  ${C.magenta}TOOL${C.reset}  ${seg.toolName.padEnd(16)} ${b}  ${C.yellow}${fmtMs(seg.wallMs).padStart(7)}${C.reset}  ${fmtPct(seg.wallMs, totalMs).padStart(5)}${errStr}${execDetail}${resultStr}`);
    }
  }

  console.log('  ' + '─'.repeat(82));
  const llmPct  = fmtPct(totalLlmMs,  totalMs);
  const toolPct = fmtPct(totalToolMs, totalMs);
  const ovhPct  = fmtPct(Math.max(0, overheadMs), totalMs);
  const costStr = totalCost > 0 ? `  ${C.gray}${fmtCost(totalCost)}${C.reset}` : '';
  console.log(
    `  Σ ${C.blue}LLM: ${fmtMs(totalLlmMs)} (${llmPct})${C.reset}` +
    `   ${C.magenta}Tools: ${fmtMs(totalToolMs)} (${toolPct})${C.reset}` +
    `   ${C.gray}Overhead: ${fmtMs(Math.max(0, overheadMs))} (${ovhPct})${C.reset}` +
    `   ${C.cyan}${roundTrips} round trip${roundTrips !== 1 ? 's' : ''}${C.reset}` +
    costStr
  );
}

function printSummary(analyses) {
  if (!analyses.length) return;
  console.log(`\n${'═'.repeat(72)}`);
  console.log(`${C.bold}SUMMARY${C.reset}  (${analyses.length} interactions)`);
  console.log('═'.repeat(72));

  const totals = analyses.map(a => a.totalMs).sort((a, b) => a - b);
  const llms   = analyses.map(a => a.totalLlmMs);
  const tools  = analyses.map(a => a.totalToolMs);
  const rounds = analyses.map(a => a.roundTrips);

  const avg = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const p95 = arr => { const s = [...arr].sort((a,b)=>a-b); return s[Math.max(0, Math.ceil(s.length * 0.95) - 1)]; };

  console.log(`  Total    avg: ${fmtMs(avg(totals))}  p95: ${fmtMs(p95(totals))}  min: ${fmtMs(totals[0])}  max: ${fmtMs(totals[totals.length-1])}`);
  console.log(`  LLM      avg: ${fmtMs(avg(llms))}  (${fmtPct(avg(llms), avg(totals))} of total)`);
  console.log(`  Tools    avg: ${fmtMs(avg(tools))}  (${fmtPct(avg(tools), avg(totals))} of total)`);
  console.log(`  Rounds   avg: ${avg(rounds).toFixed(1)}  max: ${Math.max(...rounds)}`);
  const totalCost = analyses.reduce((s, a) => s + a.totalCost, 0);
  if (totalCost > 0) console.log(`  Cost     total: ${fmtCost(totalCost)}  avg: ${fmtCost(totalCost / analyses.length)}`);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const sessionFiles = getSessionFiles();

  const allAnalyses = [];

  for (const sf of sessionFiles) {
    const records = await parseJsonl(sf.path);
    const interactions = extractInteractions(records);

    // Take last N interactions
    const slice = interactions.slice(-opts.interactions);

    if (!opts.json) {
      console.log(`\n${C.bold}Session: ${sf.label ?? sf.uuid}${C.reset}  (${interactions.length} interactions total, showing last ${slice.length})`);
    }

    const analyses = slice.map(chain => analyzeInteraction(chain));
    allAnalyses.push(...analyses);

    if (opts.json) continue;

    for (const ix of analyses) {
      printInteraction(ix);
    }
  }

  if (opts.json) {
    console.log(JSON.stringify(allAnalyses, null, 2));
    return;
  }

  if (allAnalyses.length > 1) {
    printSummary(allAnalyses);
  }
  console.log('');
}

main().catch(err => { console.error(err); process.exit(1); });
