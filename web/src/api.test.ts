import { afterEach, describe, expect, it, vi } from "vitest";

import { api, sectionError, sectionOk } from "./api";

// The dashboard's independent-degradation rule rests on this pair. A section
// can be absent from the payload outright, not merely failed — a tab left
// open across a backend restart polls the new build with the old snapshot
// shape — and `sectionOk` answers "no" for both. So the message the else
// branch renders has to come from `sectionError`, which accepts `undefined`,
// and never from `section.error`, which dereferences it inside render and
// hands the whole page to the ErrorBoundary. These are the two behaviours
// #1273 makes load-bearing.
//
// The pytest suite is what the gate's test rung runs, so
// `tests/test_dashboard_doc_claims.py` has one node that shells out to vitest
// over this file — the behaviour is pinned under both runners, and the two
// cannot disagree about what `sectionError(undefined)` returns.
describe("sectionOk", () => {
  it("says no for a section that is missing outright", () => {
    expect(sectionOk(undefined)).toBe(false);
  });

  it("says no for a section that came back failed", () => {
    expect(sectionOk({ error: "boom" })).toBe(false);
  });

  it("says yes for a section that returned data", () => {
    expect(sectionOk({ uptime_s: 12, cpus: 8 })).toBe(true);
    expect(sectionOk([])).toBe(true);
  });
});

describe("sectionError", () => {
  it("reports a failed section's own message", () => {
    expect(sectionError({ error: "boom" })).toBe("boom");
  });

  it("answers for a missing section with something worth rendering", () => {
    // The assertion is non-empty-and-stable, not the exact wording: what the
    // panel must never do is throw, and what it must never do is render a
    // blank row. `automod` absent on an older backend is the case this exists
    // for, and it is indistinguishable from `{}` by the time it gets here.
    const missing = sectionError(undefined);
    expect(missing.length).toBeGreaterThan(0);
    expect(sectionError({})).toBe(missing);
  });

  it("coerces a non-string error rather than handing React an object", () => {
    expect(sectionError({ error: 500 })).toBe("500");
  });
});

// The landing hold: a 503 with `X-Lloyd-Landing` is the backend applying a
// code update, not a failure. `streamMessage` must hold the message, tell
// the UI once through `onLandingWait`, poll the drain flag, and resend the
// identical request when it clears — and a 503 *without* the header stays an
// ordinary error with no retry. The backend's
// `tests/test_pending_restart_visibility.py` pins the other half of the
// contract (the header, and the detail string the grader and worker parse).
describe("streamMessage landing hold", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  const landing503 = {
    ok: false,
    status: 503,
    body: null,
    headers: { get: (k: string) => (k === "X-Lloyd-Landing" ? "1" : null) },
    json: async () => ({ detail: "Lloyd is landing a code update; retry in 42s." }),
  };

  it("holds the message and resends it when the drain clears", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(landing503)
      .mockResolvedValueOnce({ ok: true, json: async () => ({ draining: false }) })
      .mockResolvedValueOnce({
        ok: false, status: 400, body: null,
        headers: { get: () => null },
        json: async () => ({ detail: "boom" }),
      });
    vi.stubGlobal("fetch", fetchMock);
    const onLandingWait = vi.fn();
    const onError = vi.fn();
    api.streamMessage("hi", "c1", undefined, { onLandingWait, onError });
    await vi.advanceTimersByTimeAsync(3000);
    expect(onLandingWait).toHaveBeenCalledTimes(1);
    expect(onLandingWait).toHaveBeenCalledWith("Lloyd is landing a code update; retry in 42s.");
    expect(onError).toHaveBeenCalledWith("boom");
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(String(fetchMock.mock.calls[1][0])).toContain("/automod/drain");
    // The resend is the same POST: same endpoint, same body.
    expect(fetchMock.mock.calls[2][1]?.body).toBe(fetchMock.mock.calls[0][1]?.body);
  });

  it("keeps waiting while the backend is mid-reboot", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(landing503)
      .mockRejectedValueOnce(new TypeError("fetch failed")) // the reboot itself
      .mockResolvedValueOnce({ ok: true, json: async () => ({ draining: false }) })
      .mockResolvedValueOnce({
        ok: false, status: 400, body: null, headers: { get: () => null },
        json: async () => ({ detail: "after" }),
      });
    vi.stubGlobal("fetch", fetchMock);
    const onError = vi.fn();
    api.streamMessage("hi", "c1", undefined, { onError });
    await vi.advanceTimersByTimeAsync(6000);
    expect(onError).toHaveBeenCalledWith("after");
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it("a 503 without the header is an ordinary error, not a hold", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce({
      ok: false, status: 503, body: null,
      headers: { get: () => null },
      json: async () => ({ detail: "plain 503" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const onLandingWait = vi.fn();
    const onError = vi.fn();
    api.streamMessage("hi", "c1", undefined, { onLandingWait, onError });
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
    expect(onError).toHaveBeenCalledWith("plain 503");
    expect(onLandingWait).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("stop during the hold aborts instead of resending", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn().mockResolvedValueOnce(landing503);
    vi.stubGlobal("fetch", fetchMock);
    const onAborted = vi.fn();
    const controller = api.streamMessage("hi", "c1", undefined, { onAborted });
    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await vi.advanceTimersByTimeAsync(3000);
    expect(onAborted).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
