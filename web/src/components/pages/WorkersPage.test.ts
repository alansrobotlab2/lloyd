/**
 * WorkersPage Review tab — the review lane asks for the PENDING pile (#3548).
 *
 * Why a request-URL probe and not a render: `web/vitest.config.ts` pins
 * `environment: "node"` and `web/package.json` is a denied path for a round, so no DOM
 * package can be added here — see the note at the head of `BackgroundPage.test.ts`, the
 * same ruling on the same directory. The two claims that matter are therefore (a) the
 * URL the tab actually sends, captured off `globalThis.fetch`, and (b) what the tab
 * RENDERS as its count, pinned by the pure helper in `pendingLaneState.ts` plus a read
 * of the component's source showing it calls that helper rather than `items.length`.
 *
 * The defect: `api.workersPending(sourceFilter || undefined, 300)` returned the newest
 * 300 staged notes of EVERY status, because the route had no status filter. On the live
 * pile 2026-10-29 that reached 167 of 325 pending and dropped 158 back to 2026-09-22,
 * while da5c43de's `max_staged_pending` bound had made an unreachable pending note a
 * HALT rather than a backlog: `staged_pending_cap` skips a source's enqueue until a human
 * lowers the count, and the human could not see the notes.
 */
import { describe, it, expect, afterEach } from "vitest";
import source from "./WorkersPage.tsx?raw";
import { api } from "../../api";
import {
  PENDING_LANE_LIMIT,
  PENDING_LANE_STATUS,
  pendingLaneLabel,
  pendingLaneParams,
} from "./pendingLaneState";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

function captureFetch(payload: unknown): string[] {
  const urls: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    urls.push(String(input));
    return { json: async () => payload, ok: true, status: 200 } as Response;
  }) as unknown as typeof fetch;
  return urls;
}

describe("pendingLaneParams — the request the Review tab sends", () => {
  it("asks the server for the pending pile at the route's own ceiling", async () => {
    const urls = captureFetch({ items: [], sources: [] });
    await api.workersPending(pendingLaneParams(""));

    const url = new URL(urls[0], "http://localhost");
    expect(url.pathname).toBe("/api/workers/pending");
    expect(url.searchParams.get("status")).toBe("pending");
    expect(url.searchParams.get("status")).toBe(PENDING_LANE_STATUS);
    expect(url.searchParams.get("limit")).toBe("1000");
    expect(PENDING_LANE_LIMIT).toBe(1000);
    expect(url.searchParams.get("limit")).toBe(String(PENDING_LANE_LIMIT));
    expect(url.searchParams.has("source")).toBe(false);
  });

  it("keeps the source filter beside the status filter", async () => {
    const urls = captureFetch({ items: [], sources: [] });
    await api.workersPending(pendingLaneParams("bench-mine"));

    const params = new URL(urls[0], "http://localhost").searchParams;
    expect(params.get("source")).toBe("bench-mine");
    expect(params.get("status")).toBe("pending");
    expect(params.get("limit")).toBe("1000");
  });

  it("is the ONLY caller shape the tab may use, and an empty filter sends no source", async () => {
    // `''` is the all-sources state in ReviewPanel, and `if (opts.source)` is what keeps
    // it from becoming the string the server compares against a directory name.
    const urls = captureFetch({ items: [], sources: [] });
    await api.workersPending(pendingLaneParams("   "));
    expect(new URL(urls[0], "http://localhost").searchParams.has("source")).toBe(false);
  });
});

describe("pendingLaneLabel — 'N pending, showing M', never a window length alone", () => {
  it("names both totals when the cap bit", () => {
    const label = pendingLaneLabel(
      { pending: 325, returned: 12, dropped: 313, cap: 1000 },
      "all sources",
    );
    expect(label).toContain("325 pending");
    expect(label).toContain("showing 12");
    expect(label).toContain("313 dropped");
    expect(label).toContain("1000-row cap");
    // The 167-of-325 shape: a reader who saw only `12` would conclude a near-empty pile.
    expect(label).not.toMatch(/^[^0-9]*12[^0-9]*$/);
  });

  it("says nothing was hidden when nothing was hidden", () => {
    expect(pendingLaneLabel({ pending: 30, returned: 30, dropped: 0 })).toBe(
      "all sources — 30 pending, showing 30",
    );
    expect(pendingLaneLabel({ pending: 30, returned: 30 })).toBe(
      "all sources — 30 pending, showing 30",
    );
  });

  it("names the pile's depth when the window is empty but the pile is not", () => {
    // The old component rendered the words "no pending artifacts" here. Sixty-two notes
    // were pending at that exact moment on 2026-10-29, which is the sentence this change
    // exists to make impossible.
    const label = pendingLaneLabel(
      { pending: 62, returned: 0, dropped: 62, cap: 1000 },
      "bench-mine",
    );
    expect(label).toContain("bench-mine");
    expect(label).toContain("62 pending");
    expect(label).toContain("showing 0");
    expect(label).toContain("62 dropped");
  });

  it("falls back to the list length only when the server said nothing", () => {
    // A pre-#3548 backend, or a payload that failed to carry the counts: the tab still
    // has to print a number, and it must not invent a pending depth the server never sent.
    expect(pendingLaneLabel({ returned: 7 })).toBe(
      "all sources — showing 7 (pending count not reported)",
    );
  });
});

describe("WorkersPage source — the tab uses the lane, not a window length", () => {
  it("requests through pendingLaneParams rather than a hand-rolled 300", () => {
    expect(source).toContain("api.workersPending(pendingLaneParams(");
    expect(source).not.toMatch(/workersPending\(\s*sourceFilter\s*\|\|\s*undefined\s*,\s*300\s*\)/);
    expect(source).not.toMatch(/workersPending\([^)]*\b300\b/);
  });

  it("renders the count through the helper in more than one place", () => {
    // The SelectItem alone was not enough: it is only readable with the dropdown open.
    const uses = source.split("pendingLaneLabel(").length - 1;
    expect(uses).toBeGreaterThanOrEqual(3);
    expect(source).toContain("pendingLaneLabel(counts ?? { returned: items.length }");
  });

  it("keeps the counts in state apart from the returned list", () => {
    expect(source).toContain("d.pending_count");
    expect(source).toContain("d.dropped_by_limit");
    expect(source).toContain("useState<PendingCounts | null>(null)");
  });
});
