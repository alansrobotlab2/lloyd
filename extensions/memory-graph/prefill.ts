/**
 * prefill.ts — Unified before_prompt_build hook.
 *
 * Runs tag matching (instant, in-memory) and vector search (async)
 * in parallel, merges and deduplicates results by path, fetches
 * content for top hits, and returns a single <memory_context> block.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import type { TagIndex } from "./tag-index.js";
import type { DocMeta } from "./scanner.js";
import type { McpStdioClient } from "./mcp-client.js";
import {
  extractUserQuery,
  simplifyQuery,
  extractTopicKeywords,
  matchKeywordToTag,
  extractKeywordsViaGLM,
  MIN_QUERY_LENGTH,
} from "./query.js";
import { formatUnifiedContext } from "./format.js";

// ── Constants ─────────────────────────────────────────────────────────

const PREFETCH_TIMEOUT_MS = 2000;
const CACHE_TTL_MS = 60_000;
const MAX_CONTEXT_CHARS = 8000;
const MAX_TOPICS = 3;
const MAX_DOCS_PER_TAG = 4;
const TIER1_THRESHOLD = 0.6;
const TIER2_THRESHOLD = 0.3;
const TIER1_MAX = 3;
const TIER2_MAX = 5;
const CONTENT_PER_DOC = 1500;

const LOG_FILE = join(process.env.HOME ?? "/root", ".openclaw", "logs", "timing.jsonl");

// ── Scoring Weights ───────────────────────────────────────────────────

const W_VECTOR = 0.55;
const W_TAG = 0.35;
const W_CROSS = 0.10;
const SYNTHETIC_VECTOR_SCORE = 0.5;

// ── Types ─────────────────────────────────────────────────────────────

interface CandidateDoc {
  path: string;
  doc?: DocMeta;
  vectorScore: number;
  tagScore: number;
  snippet?: string;
  sources: Set<"vector" | "tag" | "glm">;
  finalScore: number;
}

interface Tier1Doc extends CandidateDoc {
  content?: string;
}

// ── Logging ───────────────────────────────────────────────────────────

function log(record: object) {
  try {
    mkdirSync(dirname(LOG_FILE), { recursive: true });
    appendFileSync(LOG_FILE, JSON.stringify(record) + "\n");
  } catch { /* non-fatal */ }
}

// ── LRU Cache ─────────────────────────────────────────────────────────

const cache = new Map<string, { ts: number; result: string }>();

function getCached(key: string): string | null {
  const entry = cache.get(key);
  if (entry && Date.now() - entry.ts < CACHE_TTL_MS) return entry.result;
  return null;
}

function setCache(key: string, result: string) {
  cache.set(key, { ts: Date.now(), result });
  // Evict stale
  for (const [k, v] of cache) {
    if (Date.now() - v.ts > CACHE_TTL_MS) cache.delete(k);
  }
}

// ── Tag Match (sync, in-memory) ───────────────────────────────────────

interface TagMatchResult {
  tag: string;
  tagDocCount: number;
  docs: DocMeta[];
}

function runTagMatch(
  topicKeywords: string[],
  tagIndex: TagIndex,
): { matches: TagMatchResult[]; tagDocs: Map<string, { doc: DocMeta; tagScore: number }> } {
  const matchedTags = new Set<string>();
  const matches: TagMatchResult[] = [];

  for (const kw of topicKeywords) {
    const tag = matchKeywordToTag(kw, tagIndex);
    if (!tag || matchedTags.has(tag)) continue;
    matchedTags.add(tag);

    const docSet = tagIndex.tagToDocs.get(tag);
    if (!docSet || docSet.size === 0) continue;

    const docs = Array.from(docSet)
      .map((p) => tagIndex.docs.get(p))
      .filter((d): d is DocMeta => d != null)
      .sort((a, b) => b.importance - a.importance)
      .slice(0, MAX_DOCS_PER_TAG);

    matches.push({ tag, tagDocCount: docSet.size, docs });
    if (matches.length >= MAX_TOPICS) break;
  }

  // Compute normalized tagScore for each doc
  const allTagDocs = matches.flatMap((m) => m.docs);
  const maxIdf = Math.max(
    ...matches.map((m) => tagIndex.tagIdf.get(m.tag) ?? 0),
    1,
  );
  const maxImportance = Math.max(
    ...allTagDocs.map((d) => d.importance),
    1,
  );

  const tagDocs = new Map<string, { doc: DocMeta; tagScore: number }>();
  for (const m of matches) {
    const tagIdf = tagIndex.tagIdf.get(m.tag) ?? 0;
    for (const doc of m.docs) {
      if (tagDocs.has(doc.path)) continue; // keep first (highest importance topic)
      const tagScore =
        0.5 * (tagIdf / maxIdf) +
        0.5 * (doc.importance / maxImportance);
      tagDocs.set(doc.path, { doc, tagScore });
    }
  }

  return { matches, tagDocs };
}

// ── Merge & Rank ──────────────────────────────────────────────────────

function mergeAndRank(
  tagDocs: Map<string, { doc: DocMeta; tagScore: number }>,
  vectorResults: Array<{ path: string; score: number; snippet?: string }>,
): CandidateDoc[] {
  const candidates = new Map<string, CandidateDoc>();

  // Insert vector results
  for (const vr of vectorResults) {
    const existing = candidates.get(vr.path);
    if (existing) {
      existing.vectorScore = Math.max(existing.vectorScore, vr.score);
      existing.sources.add("vector");
      if (vr.snippet && !existing.snippet) existing.snippet = vr.snippet;
    } else {
      candidates.set(vr.path, {
        path: vr.path,
        vectorScore: vr.score,
        tagScore: 0,
        snippet: vr.snippet,
        sources: new Set(["vector"]),
        finalScore: 0,
      });
    }
  }

  // Insert/merge tag results
  for (const [path, { doc, tagScore }] of tagDocs) {
    const existing = candidates.get(path);
    if (existing) {
      existing.doc = doc;
      existing.tagScore = tagScore;
      existing.sources.add("tag");
    } else {
      candidates.set(path, {
        path,
        doc,
        vectorScore: SYNTHETIC_VECTOR_SCORE,
        tagScore,
        sources: new Set(["tag"]),
        finalScore: 0,
      });
    }
  }

  // Compute final scores
  for (const c of candidates.values()) {
    const crossBonus = c.sources.size > 1 ? W_CROSS : 0;
    c.finalScore = W_VECTOR * c.vectorScore + W_TAG * c.tagScore + crossBonus;
  }

  return Array.from(candidates.values()).sort((a, b) => b.finalScore - a.finalScore);
}

// ── MCP helper ────────────────────────────────────────────────────────

/** Call an MCP tool, rejecting if the AbortSignal fires first. */
async function callToolWithAbort(
  client: McpStdioClient,
  name: string,
  args: Record<string, unknown>,
  signal: AbortSignal,
): Promise<Array<{ type: string; text: string }>> {
  if (signal.aborted) throw Object.assign(new Error("Aborted"), { name: "AbortError" });
  return Promise.race([
    client.callTool(name, args),
    new Promise<never>((_, reject) =>
      signal.addEventListener(
        "abort",
        () => reject(Object.assign(new Error("Aborted"), { name: "AbortError" })),
        { once: true },
      )
    ),
  ]);
}

// ── Hook Factory ──────────────────────────────────────────────────────

export function createPrefillHook(
  getTagIndex: () => TagIndex,
  logger: { info?: (...args: any[]) => void; warn?: (...args: any[]) => void } | undefined,
  mcpClient: McpStdioClient,
) {
  return async (event: any, ctx: any) => {
    const t0 = Date.now();

    // ── Phase 0: Early exit ──────────────────────────────────────
    const userQuery = extractUserQuery(event.prompt);
    if (!userQuery || userQuery.length < MIN_QUERY_LENGTH) return;

    const vectorQuery = simplifyQuery(userQuery);
    const effectiveQuery = vectorQuery.length >= 4 ? vectorQuery : userQuery.slice(0, 80);
    const wordCount = effectiveQuery.split(/\s+/).filter((w) => w.length > 0).length;
    if (wordCount < 2) return;

    // Check cache
    const cached = getCached(effectiveQuery);
    if (cached) {
      logger?.warn?.("memory-prefill: cache hit");
      return { prependContext: cached };
    }

    logger?.warn?.(`memory-prefill: hook fired, query=${JSON.stringify(effectiveQuery)}`);

    const abortCtrl = new AbortController();
    const timer = setTimeout(() => abortCtrl.abort(), PREFETCH_TIMEOUT_MS);

    try {
      // ── Phase 1: Keyword extraction ────────────────────────────
      const topicKeywords = extractTopicKeywords(event.prompt);

      // ── Phase 2: Tag match (sync, instant) ─────────────────────
      const tagIndex = getTagIndex();
      const { matches: tagMatches, tagDocs } =
        tagIndex.docCount > 0
          ? runTagMatch(topicKeywords, tagIndex)
          : { matches: [], tagDocs: new Map<string, { doc: DocMeta; tagScore: number }>() };

      logger?.warn?.(`memory-prefill: tag match found ${tagDocs.size} docs from ${tagMatches.length} topics`);

      // ── Phase 3+4: Pipelined parallel async ───────────────────
      //
      // Initial memory_search and GLM run in parallel.
      // Extra GLM searches are chained off the GLM promise directly,
      // so they start as soon as GLM resolves — not after memory_search
      // also finishes. Total wall time ≈ max(search, glm + extra_searches).

      const matchedTagNames = new Set(tagMatches.map((m) => m.tag.toLowerCase()));

      const initialSearchPromise = callToolWithAbort(
        mcpClient, "memory_search",
        { query: effectiveQuery, max_results: 8, json_output: true },
        abortCtrl.signal,
      );

      const extraPhasePromise = extractKeywordsViaGLM(userQuery, abortCtrl.signal).then(
        async (keywords: string[]) => {
          const glmKeywords = keywords.filter((q) => q && q.length > 2).slice(0, 2);
          const extraQueries = glmKeywords.filter((kw) => !matchedTagNames.has(kw.toLowerCase()));
          if (extraQueries.length > 0) {
            logger?.warn?.(`memory-prefill: GLM extra queries: ${JSON.stringify(extraQueries)}`);
          }
          const extraResults = extraQueries.length > 0
            ? await Promise.allSettled(
                extraQueries.map((q) =>
                  callToolWithAbort(mcpClient, "memory_search", { query: q, max_results: 3, json_output: true }, abortCtrl.signal),
                ),
              )
            : [];
          return { glmKeywords, extraResults };
        },
        () => ({ glmKeywords: [] as string[], extraResults: [] as PromiseSettledResult<Array<{ type: string; text: string }>>[] }),
      );

      const [vectorResult, extraPhaseResult] = await Promise.allSettled([
        initialSearchPromise,
        extraPhasePromise,
      ]);

      // Collect all vector results
      const allVectorResults: Array<{ path: string; score: number; snippet?: string }> = [];

      if (vectorResult.status === "fulfilled") {
        const text = vectorResult.value.filter((b) => b.type === "text").map((b) => b.text).join("");
        let results: any[] = [];
        try {
          const parsed = JSON.parse(text);
          if (Array.isArray(parsed?.results)) results = parsed.results;
        } catch { /* ignore parse errors */ }
        logger?.warn?.(`memory-prefill: vector search returned ${results.length} results`);
        for (const r of results) {
          allVectorResults.push({ path: r.path, score: r.score ?? 0, snippet: r.snippet });
        }
      } else {
        logger?.warn?.(`memory-prefill: vector search failed: ${(vectorResult.reason as any)?.message}`);
      }

      const { glmKeywords, extraResults } = extraPhaseResult.status === "fulfilled"
        ? extraPhaseResult.value
        : { glmKeywords: [] as string[], extraResults: [] as PromiseSettledResult<Array<{ type: string; text: string }>>[] };

      for (const r of extraResults) {
        if (r.status === "fulfilled") {
          const text = r.value.filter((b) => b.type === "text").map((b) => b.text).join("");
          let results: any[] = [];
          try {
            const parsed = JSON.parse(text);
            if (Array.isArray(parsed?.results)) results = parsed.results;
          } catch { /* ignore */ }
          for (const vr of results) {
            allVectorResults.push({ path: vr.path, score: vr.score ?? 0, snippet: vr.snippet });
          }
        }
      }

      // ── Phase 5: Merge & rank ──────────────────────────────────
      const ranked = mergeAndRank(tagDocs, allVectorResults);

      if (ranked.length === 0) return;

      // ── Phase 6: Fetch content ─────────────────────────────────
      let tier1 = ranked.filter((c) => c.finalScore >= TIER1_THRESHOLD).slice(0, TIER1_MAX);

      // Fallback: promote top tagDocs if no tier 1 qualifies
      if (tier1.length === 0 && tagDocs.size > 0) {
        tier1 = ranked.slice(0, Math.min(2, ranked.length));
      }

      const tier2 = ranked
        .filter((c) => c.finalScore >= TIER2_THRESHOLD && !tier1.includes(c))
        .slice(0, TIER2_MAX);

      // Fetch full content for tier 1
      const tier1WithContent: Tier1Doc[] = [];
      if (tier1.length > 0) {
        const getResults = await Promise.allSettled(
          tier1.map((c) =>
            callToolWithAbort(mcpClient, "memory_get", { path: c.path }, abortCtrl.signal),
          ),
        );
        for (let i = 0; i < tier1.length; i++) {
          const c = tier1[i] as Tier1Doc;
          const r = getResults[i];
          if (r.status === "fulfilled") {
            const text = r.value.filter((b) => b.type === "text").map((b) => b.text).join("");
            if (text) c.content = text.slice(0, CONTENT_PER_DOC);
          }
          tier1WithContent.push(c);
        }
      }

      // ── Phase 7: Format & return ───────────────────────────────
      const context = formatUnifiedContext(tier1WithContent, tier2, MAX_CONTEXT_CHARS);
      if (!context) return;

      setCache(effectiveQuery, context);

      log({
        ts: new Date().toISOString(),
        event: "memory_prefill",
        sessionId: ctx?.sessionId,
        durationMs: Date.now() - t0,
        effectiveQuery,
        queryLength: userQuery.length,
        tagTopics: tagMatches.length,
        tagDocs: tagDocs.size,
        vectorResults: allVectorResults.length,
        glmKeywords: glmKeywords.length,
        tier1Count: tier1WithContent.length,
        tier2Count: tier2.length,
        contextChars: context.length,
      });

      return { prependContext: context };
    } catch (err: any) {
      if (err?.name !== "AbortError") {
        logger?.warn?.(`memory-prefill error: ${err?.message}`);
      }
    } finally {
      clearTimeout(timer);
    }
  };
}
