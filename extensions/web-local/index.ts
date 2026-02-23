import { createRequire } from "node:module";
import { execFile } from "node:child_process";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

// Resolve linkedom + readability from OpenClaw's own node_modules
const oclawRequire = createRequire(
  require.resolve("openclaw/plugin-sdk").replace(/\/plugin-sdk\/.*/, "/package.json"),
);
const { parseHTML } = oclawRequire("linkedom") as typeof import("linkedom");
const { Readability } = oclawRequire("@mozilla/readability") as {
  Readability: new (doc: any) => {
    parse(): { title: string; textContent: string; content: string } | null;
  };
};

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

const PYTHON = "/home/alansrobotlab/Projects/lloyd/.venv/bin/python";
const USER_AGENT =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const ACCEPT_LANG = "en-US,en;q=0.9";
const TIMEOUT_MS = 15_000;
const MAX_RESPONSE_BYTES = 2_000_000;
const DEFAULT_MAX_CHARS = 50_000;

function textResult(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

// ---------------------------------------------------------------------------
// web_search — Google via googlesearch-python package
// ---------------------------------------------------------------------------

interface SearchResult {
  position: number;
  title: string;
  url: string;
  snippet: string;
}

const SEARCH_SCRIPT = `
import json, sys
from googlesearch import search
query = sys.argv[1]
count = int(sys.argv[2])
results = list(search(query, num_results=count, advanced=True))
out = [{"title": r.title, "url": r.url, "snippet": r.description} for r in results]
print(json.dumps(out))
`;

function googleSearch(query: string, count: number): Promise<SearchResult[]> {
  return new Promise((resolve, reject) => {
    const child = execFile(
      PYTHON,
      ["-c", SEARCH_SCRIPT, query, String(count)],
      { timeout: TIMEOUT_MS, maxBuffer: 1024 * 1024 },
      (err, stdout, stderr) => {
        if (err) {
          reject(new Error(stderr?.trim() || err.message));
          return;
        }
        try {
          const raw = JSON.parse(stdout) as Array<{
            title: string;
            url: string;
            snippet: string;
          }>;
          resolve(
            raw.map((r, i) => ({
              position: i + 1,
              title: r.title,
              url: r.url,
              snippet: r.snippet,
            })),
          );
        } catch (e) {
          reject(new Error(`Failed to parse search output: ${stdout.slice(0, 200)}`));
        }
      },
    );
  });
}

// ---------------------------------------------------------------------------
// web_fetch — HTTP GET + Readability content extraction
// ---------------------------------------------------------------------------

const PRIVATE_IP_PATTERNS = [
  /^127\./,
  /^10\./,
  /^172\.(1[6-9]|2\d|3[01])\./,
  /^192\.168\./,
  /^0\./,
  /^169\.254\./,
  /^::1$/,
  /^fc00:/i,
  /^fd/i,
  /^fe80:/i,
];

function isPrivateHost(hostname: string): boolean {
  return (
    hostname === "localhost" ||
    PRIVATE_IP_PATTERNS.some((p) => p.test(hostname))
  );
}

async function fetchPage(
  url: string,
  extractMode: "markdown" | "text",
  maxChars: number,
): Promise<string> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`Invalid URL: ${url}`);
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`Only http/https URLs are supported, got ${parsed.protocol}`);
  }

  if (isPrivateHost(parsed.hostname)) {
    throw new Error(`Blocked: private/internal hostname "${parsed.hostname}"`);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      method: "GET",
      headers: {
        "User-Agent": USER_AGENT,
        Accept:
          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": ACCEPT_LANG,
      },
      signal: controller.signal,
      redirect: "follow",
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }

    const contentType = res.headers.get("content-type") ?? "";

    // Non-HTML: return raw text (truncated)
    if (!contentType.includes("html") && !contentType.includes("xml")) {
      const text = await res.text();
      const truncated = text.slice(0, maxChars);
      return truncated.length < text.length
        ? truncated + `\n\n[Truncated — ${text.length} chars total]`
        : truncated;
    }

    // Read body with size cap
    const chunks: Uint8Array[] = [];
    let totalBytes = 0;
    const bodyReader = res.body?.getReader();
    if (!bodyReader) throw new Error("No response body");

    while (true) {
      const { done, value } = await bodyReader.read();
      if (done) break;
      chunks.push(value);
      totalBytes += value.length;
      if (totalBytes > MAX_RESPONSE_BYTES) {
        bodyReader.cancel();
        break;
      }
    }

    const decoder = new TextDecoder("utf-8", { fatal: false });
    const html = decoder.decode(Buffer.concat(chunks));

    // Try Readability extraction
    const { document } = parseHTML(html);

    // Set a base URL so Readability can resolve relative links
    try {
      const baseEl = document.createElement("base");
      baseEl.setAttribute("href", url);
      document.head?.appendChild(baseEl);
    } catch {
      // non-critical
    }

    const article = new Readability(document).parse();

    if (article?.textContent) {
      const title = article.title ? `# ${article.title}\n\n` : "";
      const body = article.textContent.trim();
      const full = title + body;
      const truncated = full.slice(0, maxChars);
      return truncated.length < full.length
        ? truncated + `\n\n[Truncated — ${full.length} chars total]`
        : truncated;
    }

    // Fallback: strip tags, return body text
    const bodyEl = document.querySelector("body");
    const rawText = (bodyEl?.textContent ?? html.replace(/<[^>]+>/g, " "))
      .replace(/\s+/g, " ")
      .trim();

    const truncated = rawText.slice(0, maxChars);
    return truncated.length < rawText.length
      ? truncated + `\n\n[Truncated — ${rawText.length} chars total]`
      : truncated;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

const searchParams = {
  type: "object" as const,
  properties: {
    query: { type: "string" as const, description: "Search query" },
    count: {
      type: "integer" as const,
      description: "Number of results to return (1–10, default 5)",
      minimum: 1,
      maximum: 10,
    },
  },
  required: ["query"] as string[],
};

const fetchParams = {
  type: "object" as const,
  properties: {
    url: {
      type: "string" as const,
      description: "The URL to fetch (http or https)",
    },
    extractMode: {
      type: "string" as const,
      enum: ["markdown", "text"],
      description:
        'Extraction mode: "markdown" or "text" (default "markdown")',
    },
    maxChars: {
      type: "integer" as const,
      description: "Maximum characters to return (default 50000)",
      minimum: 1000,
      maximum: 200000,
    },
  },
  required: ["url"] as string[],
};

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.(
    "web-local: registering web_search + web_fetch (Google + HTTP/Readability)",
  );

  api.registerTool({
    name: "web_search",
    label: "Web Search",
    description:
      "Search the web using Google. Returns a list of results with title, URL, and snippet.",
    parameters: searchParams,
    async execute(
      _toolCallId: string,
      params: { query: string; count?: number },
    ) {
      const count = Math.min(Math.max(params.count ?? 5, 1), 10);
      try {
        const results = await googleSearch(params.query, count);
        if (results.length === 0) {
          return textResult(`No results found for "${params.query}".`);
        }
        const formatted = results
          .map(
            (r) =>
              `[${r.position}] ${r.title}\n    ${r.url}\n    ${r.snippet}`,
          )
          .join("\n\n");
        return textResult(formatted);
      } catch (err) {
        return textResult(
          `web_search error: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    },
  });

  api.registerTool({
    name: "web_fetch",
    label: "Fetch Web Page",
    description:
      "Fetch a URL and extract its readable content. Returns the main text content of the page.",
    parameters: fetchParams,
    async execute(
      _toolCallId: string,
      params: {
        url: string;
        extractMode?: "markdown" | "text";
        maxChars?: number;
      },
    ) {
      const mode = params.extractMode ?? "markdown";
      const maxChars = Math.min(params.maxChars ?? DEFAULT_MAX_CHARS, 200_000);
      try {
        const content = await fetchPage(params.url, mode, maxChars);
        return textResult(content);
      } catch (err) {
        return textResult(
          `web_fetch error: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    },
  });
}
