import { appendFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

const LOG_FILE = join(process.env.HOME ?? "/root", ".openclaw", "logs", "timing.jsonl");
const PREFETCH_TIMEOUT_MS = 1200;
const CACHE_TTL_MS = 60_000;
const GLM_SHORT_QUERY_THRESHOLD = 40;
const MAX_RESULT_CHARS = 8000;
const MIN_QUERY_LENGTH = 12;
const GATEWAY_URL = "http://127.0.0.1:18789/tools/invoke";
const GLM_URL = "http://127.0.0.1:8091/v1/chat/completions";
const GLM_MODEL = "Qwen3-30B-A3B-Instruct-2507";

function log(record: object) {
  try {
    mkdirSync(dirname(LOG_FILE), { recursive: true });
    appendFileSync(LOG_FILE, JSON.stringify(record) + "\n");
  } catch {
    // non-fatal
  }
}

/** Invoke a tool via the OpenClaw HTTP gateway (auth mode: none) */
async function gatewayInvoke(
  tool: string,
  args: Record<string, unknown>,
  signal: AbortSignal,
): Promise<any> {
  const resp = await fetch(GATEWAY_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ tool, args }),
    signal,
  });
  if (!resp.ok) throw new Error(`gateway ${tool} failed: ${resp.status}`);
  const data = (await resp.json()) as any;
  if (!data.ok) throw new Error(`gateway ${tool} error: ${data.error}`);
  return data.result;
}

/** Strip question filler words and extract dense keyword phrase for vector search */
const FILLER_WORDS = new Set([
  "hey", "hi", "what", "do", "you", "know", "about", "my", "work", "with", "the", "a", "an",
  "is", "are", "was", "were", "can", "could", "would", "should", "tell", "me", "i", "want",
  "to", "how", "when", "where", "why", "who", "which", "that", "this", "these", "those", "it",
  "be", "been", "being", "have", "has", "had", "does", "did", "will", "shall", "may", "might",
  "some", "any", "all", "more", "very", "just", "your", "our", "their", "its", "there",
  "give", "get", "let", "make", "see", "look", "show", "find", "help", "think", "say",
  "please", "thanks", "ok", "so", "and", "or", "but", "in", "on", "at", "of", "for", "from",
  "up", "out", "if", "no", "not", "as", "into", "through", "during", "before", "after",
]);

function simplifyQuery(query: string): string {
  const words = query
    .toLowerCase()
    .replace(/[?!.,;:'"]/g, "")
    .split(/\s+/)
    .filter((w) => w.length > 1 && !FILLER_WORDS.has(w));
  return words.slice(0, 6).join(" ");
}

/** Strip OpenClaw metadata envelope from prompt, extract clean user text */
function extractUserQuery(prompt: string): string | null {
  // Match "[Day YYYY-MM-DD HH:MM TZ] actual message" pattern
  const match = prompt.match(/\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+[\d:]+\s+\w+\]\s+([\s\S]+)$/);
  if (match) return match[1].trim();
  // Fallback: strip JSON blocks and return remainder
  const stripped = prompt.replace(/```json[\s\S]*?```/g, "").replace(/\s+/g, " ").trim();
  return stripped.length >= MIN_QUERY_LENGTH ? stripped.slice(-500) : null;
}

/** Use local GLM to extract 2-3 refined search keyword phrases from user text */
async function extractKeywordsViaGLM(
  userText: string,
  signal: AbortSignal,
): Promise<string[]> {
  const resp = await fetch(GLM_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: GLM_MODEL,
      max_tokens: 40,
      messages: [
        {
          role: "system",
          content: "You are a search keyword extractor. Always respond with ONLY a valid JSON array of 2-3 short keyword strings. No markdown, no explanation.",
        },
        {
          role: "user",
          content: `Message: ${userText.slice(0, 300)}`,
        },
      ],
    }),
    signal,
  });
  if (!resp.ok) return [];
  const data = (await resp.json()) as any;
  const text: string = data?.choices?.[0]?.message?.content ?? "";
  const arrayMatch = text.match(/\[[\s\S]*?\]/);
  if (!arrayMatch) return [];
  try {
    return JSON.parse(arrayMatch[0]) as string[];
  } catch {
    return [];
  }
}

/** Format search results + file contents into a compact prependContext block */
function formatContext(
  searches: Array<{ query: string; results: any[] }>,
  fileContents: Map<string, string>,
): string {
  if (searches.every((s) => s.results.length === 0) && fileContents.size === 0) return "";

  const lines: string[] = [
    "<memory_prefetch>",
    "Memory results pre-fetched based on your question. Use this context to answer directly if sufficient — skip memory_search unless you need information not shown here.\n",
  ];

  const seenPaths = new Set<string>();

  for (const { query, results } of searches) {
    if (results.length === 0) continue;
    lines.push(`**Search: "${query}"**`);
    for (const r of results) {
      const label = `${r.path} (score: ${r.score?.toFixed(2)})`;
      if (fileContents.has(r.path)) {
        lines.push(`\n--- ${label} ---`);
        lines.push(fileContents.get(r.path)!.slice(0, 1500));
        seenPaths.add(r.path);
      } else if (!seenPaths.has(r.path)) {
        lines.push(`\n--- ${label} (snippet) ---`);
        lines.push(r.snippet ?? "");
        seenPaths.add(r.path);
      }
    }
  }

  lines.push("\n</memory_prefetch>");
  const full = lines.join("\n");
  return full.length > MAX_RESULT_CHARS
    ? full.slice(0, MAX_RESULT_CHARS) + "\n[... truncated]\n</memory_prefetch>"
    : full;
}

// Simple LRU cache for recent prefetch results
const prefetchCache = new Map<string, { ts: number; result: string }>();

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.("memory-prefetch: active (gateway + GLM mode)");

  api.on("before_prompt_build", async (event, ctx) => {
    const t0 = Date.now();

    // 1. Extract clean user query from prompt
    const userQuery = extractUserQuery(event.prompt);
    if (!userQuery || userQuery.length < MIN_QUERY_LENGTH) return;

    // Simplify conversational question to dense keyword phrase for vector search
    const searchQuery = simplifyQuery(userQuery);
    const effectiveQuery = searchQuery.length >= 4 ? searchQuery : userQuery.slice(0, 80);

    // Skip prefetch when too few content words survive filler removal
    const wordCount = effectiveQuery.split(/\s+/).filter((w) => w.length > 0).length;
    if (wordCount < 2) {
      api.logger.warn?.("memory-prefetch: skipped — too few content words");
      return;
    }

    api.logger.warn?.(`memory-prefetch: hook fired, query=${JSON.stringify(effectiveQuery)}`);

    // Check cache before doing any network calls
    const cached = prefetchCache.get(effectiveQuery);
    if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
      api.logger.warn?.("memory-prefetch: returning cached result");
      return { prependContext: cached.result };
    }

    const abortCtrl = new AbortController();
    const timer = setTimeout(() => abortCtrl.abort(), PREFETCH_TIMEOUT_MS);

    try {
      // 2. PARALLEL: initial memory_search (simplified query) + GLM keyword extraction
      //    Skip GLM for short queries — the simplified string is sufficient
      const skipGlm = userQuery.length < GLM_SHORT_QUERY_THRESHOLD;
      const [initialSearchResult, refinedQueries] = await Promise.allSettled([
        gatewayInvoke("memory_search", { query: effectiveQuery, maxResults: 5 }, abortCtrl.signal),
        skipGlm ? Promise.resolve([]) : extractKeywordsViaGLM(userQuery, abortCtrl.signal),
      ]);

      const searchGroups: Array<{ query: string; results: any[] }> = [];

      api.logger.warn?.(`memory-prefetch: search settled: ${initialSearchResult.status}, glm: ${refinedQueries.status}`);
      if (initialSearchResult.status === "rejected") {
        api.logger.warn?.(`memory-prefetch: search error: ${(initialSearchResult.reason as any)?.message}`);
      }
      if (initialSearchResult.status === "fulfilled") {
        const results: any[] = Array.isArray(initialSearchResult.value?.details?.results)
          ? initialSearchResult.value.details.results
          : [];
        api.logger.warn?.(`memory-prefetch: search returned ${results.length} results`);
        searchGroups.push({ query: userQuery, results });

        // 3. memory_get top-scoring results in PARALLEL
        const topPaths = results
          .filter((r) => (r.score ?? 0) >= 0.7)
          .slice(0, 3)
          .map((r) => r.path as string);

        const getResults = await Promise.allSettled(
          topPaths.map((path) =>
            gatewayInvoke("memory_get", { path }, abortCtrl.signal),
          ),
        );

        const fileContents = new Map<string, string>();
        for (let i = 0; i < topPaths.length; i++) {
          const r = getResults[i];
          if (r.status === "fulfilled") {
            const text = (r.value?.content ?? [])
              .filter((c: any) => c.type === "text")
              .map((c: any) => c.text)
              .join("");
            if (text) fileContents.set(topPaths[i], text);
          }
        }

        // 4. Run extra searches from GLM-refined queries in PARALLEL
        const extraQueries =
          refinedQueries.status === "fulfilled"
            ? (refinedQueries.value as string[]).filter((q) => q && q !== userQuery).slice(0, 2)
            : [];

        if (extraQueries.length > 0) {
          api.logger.warn?.(`memory-prefetch: GLM extra queries: ${JSON.stringify(extraQueries)}`);
          const extraResults = await Promise.allSettled(
            extraQueries.map((q) =>
              gatewayInvoke("memory_search", { query: q, maxResults: 3 }, abortCtrl.signal),
            ),
          );
          for (let i = 0; i < extraQueries.length; i++) {
            const r = extraResults[i];
            if (r.status === "fulfilled") {
              searchGroups.push({
                query: extraQueries[i],
                results: Array.isArray(r.value?.details?.results) ? r.value.details.results : [],
              });
            }
          }
        }

        // 5. Format and inject as prependContext
        const context = formatContext(searchGroups, fileContents);
        if (context) {
          // Cache the result
          prefetchCache.set(effectiveQuery, { ts: Date.now(), result: context });
          // Evict stale entries (keep cache small)
          for (const [k, v] of prefetchCache) {
            if (Date.now() - v.ts > CACHE_TTL_MS) prefetchCache.delete(k);
          }

          log({
            ts: new Date().toISOString(),
            event: "memory_prefetch",
            sessionId: ctx.sessionId,
            durationMs: Date.now() - t0,
            effectiveQuery,
            queryLength: userQuery.length,
            searchCount: searchGroups.length,
            topPaths,
            contextChars: context.length,
            glm: !skipGlm && refinedQueries.status === "fulfilled" && (refinedQueries.value as string[]).length > 0,
          });
          return { prependContext: context };
        }
      }
    } catch (err: any) {
      if (err?.name !== "AbortError") {
        api.logger.warn?.(`memory-prefetch error: ${err?.message}`);
      }
    } finally {
      clearTimeout(timer);
    }
  });
}
