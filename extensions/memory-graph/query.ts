/**
 * query.ts — Shared query extraction and keyword processing.
 *
 * Single source of truth for text processing used by both the
 * unified prefill hook and tag matching. Merged from memory-prefetch
 * and memory-graph's duplicate implementations.
 */

import type { TagIndex } from "./tag-index.js";

// ── Constants ─────────────────────────────────────────────────────────

const GLM_URL = "http://127.0.0.1:8091/v1/chat/completions";
const GLM_MODEL = "Qwen3-30B-A3B-Instruct-2507";
const GLM_SHORT_QUERY_THRESHOLD = 40;
export const MIN_QUERY_LENGTH = 12;

// ── Filler Words (superset of both plugins' lists) ───────────────────

const FILLER_WORDS = new Set([
  "hey", "hi", "what", "do", "you", "know", "about", "my", "work", "with",
  "the", "a", "an", "is", "are", "was", "were", "can", "could", "would",
  "should", "tell", "me", "i", "want", "to", "how", "when", "where", "why",
  "who", "which", "that", "this", "these", "those", "it", "be", "been",
  "being", "have", "has", "had", "does", "did", "will", "shall", "may",
  "might", "some", "any", "all", "more", "very", "just", "your", "our",
  "their", "its", "there", "give", "get", "let", "make", "see", "look",
  "show", "find", "help", "think", "say", "please", "thanks", "ok", "so",
  "and", "or", "but", "in", "on", "at", "of", "for", "from", "up", "out",
  "if", "no", "not", "as", "into", "through", "during", "before", "after",
  // memory-graph extras
  "use", "using", "search", "memory", "related", "connected", "tag", "tags",
]);

// ── Query Extraction ──────────────────────────────────────────────────

/** Strip OpenClaw metadata envelope from prompt, extract clean user text. */
export function extractUserQuery(prompt: string): string | null {
  const match = prompt.match(/\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+[\d:]+\s+\w+\]\s+([\s\S]+)$/);
  if (match) return match[1].trim();
  const stripped = prompt.replace(/```json[\s\S]*?```/g, "").replace(/\s+/g, " ").trim();
  return stripped.length >= MIN_QUERY_LENGTH ? stripped.slice(-500) : null;
}

/** Simplify conversational question to dense keyword phrase for vector search. */
export function simplifyQuery(query: string): string {
  const words = query
    .toLowerCase()
    .replace(/[?!.,;:'"]/g, "")
    .split(/\s+/)
    .filter((w) => w.length > 1 && !FILLER_WORDS.has(w));
  return words.slice(0, 6).join(" ");
}

/** Extract topic keywords including bigrams for tag matching. */
export function extractTopicKeywords(prompt: string): string[] {
  const match = prompt.match(/\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+[\d:]+\s+\w+\]\s+([\s\S]+)$/);
  const text = match ? match[1].trim() : prompt.trim();

  const words = text
    .toLowerCase()
    .replace(/[?!.,;:'"()\[\]{}#]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !FILLER_WORDS.has(w));

  // Also look for bigrams from adjacent non-filler words
  const rawWords = text
    .replace(/[?!.,;:'"()\[\]{}#]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
  const phrases: string[] = [];
  for (let i = 0; i < rawWords.length - 1; i++) {
    if (
      !FILLER_WORDS.has(rawWords[i].toLowerCase()) &&
      !FILLER_WORDS.has(rawWords[i + 1].toLowerCase())
    ) {
      phrases.push(`${rawWords[i]} ${rawWords[i + 1]}`.toLowerCase());
    }
  }

  return [...phrases, ...words];
}

/** Match a keyword against the tag index. Returns actual tag name or null. */
export function matchKeywordToTag(keyword: string, idx: TagIndex): string | null {
  const kw = keyword.toLowerCase();

  for (const tag of idx.tagToDocs.keys()) {
    if (tag.toLowerCase() === kw) return tag;
  }

  const kwNoDash = kw.replace(/-/g, "");
  for (const tag of idx.tagToDocs.keys()) {
    if (tag.toLowerCase().replace(/-/g, "") === kwNoDash) return tag;
  }

  return null;
}

/** Use local GLM to extract 2-3 refined search keywords. Returns [] on failure. */
export async function extractKeywordsViaGLM(
  userText: string,
  signal: AbortSignal,
): Promise<string[]> {
  if (userText.length < GLM_SHORT_QUERY_THRESHOLD) return [];

  try {
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
    return JSON.parse(arrayMatch[0]) as string[];
  } catch {
    return [];
  }
}
