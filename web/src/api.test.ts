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

// #2207 clause 4: the revoke must carry a body the server can check.
//
// Runnable pin for this clause: `tests/test_system_revoke_confirm.py::
// test_the_request_the_shipped_client_builds_is_the_request_this_route_accepts`.
// These nodes read the client; that one sends the client's own bytes at the real
// route, and pytest runs it in every checkout. `web/node_modules` is a denied path
// (`scripts/automod/spec.py`), so a review snapshot of this commit has no vitest
// binary and no diff can give it one — read these three as the client-side detail,
// and the pytest node as the proof the seam holds.
//
// `DELETE /api/system/clients/{name}` used to read no body at all, so the only
// confirmation in the path was the browser's `confirm()` dialog in SettingsPage —
// ceremony that exists only in a tab, and nothing at all to a script. The server
// cannot check who you are instead: no client certificate has been requested
// since 5e1351f3, so the one name available is the `x-client-fingerprint` header
// the caller chose to send. So the enforceable check is this body, and if the
// client stops sending it the honest click in the UI starts answering 400 —
// `tests/test_system_revoke_confirm.py` is the other end of that seam.
describe("revokeClient", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const revoked = { ok: true, status: 204, json: async () => ({}) };

  it("sends a JSON body whose confirm equals the client name", async () => {
    const fetchMock = vi.fn().mockResolvedValue(revoked);
    vi.stubGlobal("fetch", fetchMock);
    await expect(api.revokeClient("studio-mini")).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/system/clients/studio-mini");
    expect(init?.method).toBe("DELETE");
    expect(JSON.parse(String(init?.body))).toEqual({ confirm: "studio-mini" });
    expect(init?.headers?.['Content-Type']).toBe('application/json');
  });

  it("confirms the same client the path names, even when the name needs escaping", async () => {
    // The path is percent-encoded and the body is not, so a rewrite that
    // confirmed the encoded form would mismatch every name with a space in it
    // and the server — which compares the decoded path name to `confirm` —
    // would refuse a legitimate revoke.
    const fetchMock = vi.fn().mockResolvedValue(revoked);
    vi.stubGlobal("fetch", fetchMock);
    await api.revokeClient("studio mini");

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/system/clients/studio%20mini");
    expect(JSON.parse(String(init?.body))).toEqual({ confirm: "studio mini" });
  });

  it("surfaces the server's refusal detail rather than reporting a silent success", async () => {
    // The 400 is now a reachable outcome from this function, and the message a
    // person sees has to be the server's reason ("confirm must name the client")
    // rather than `revoke failed: 400`.
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false, status: 400,
      json: async () => ({ detail: 'revoke requires a JSON body whose "confirm" field names this client' }),
    });
    vi.stubGlobal("fetch", fetchMock);
    await expect(api.revokeClient("studio-mini")).rejects.toThrow(/confirm/);
  });
});

// ── #4292 clauses 1 and 2: NetworkState carries what #628's accessor already sends ──
//
// `app/routers/dashboard.py`'s `_network()` returns `agent_mcp.egress.network_report()`
// verbatim, and that report has carried two more keys than the typed reader declared:
// `policy.bash_record` (egress.py:1358, `bash_record_on()`) and the per-via split
// `per_via` (egress.py:1354). Probed live on 2026-11-22, `/api/dashboard` returned
// `policy` = {telemetry: true, enforce: false, bash_record: false, allow_entries: 7,
// permanent_allow_entries: 7} and one `per_via` row {via: 'tool', count: 494,
// distinct_hosts: 89, denies: 0, last_at: '2026-11-22T14:33:46+00:00', scopes: […]},
// which the browser threw away at the type boundary.
//
// Two runners, two halves of this pin, and both are needed:
//   * `tsc --noEmit` — the gate's `frontend` rung runs it as a delta, so a NEW type
//     error reddens the round. The `SAMPLE` literal and the `Exact<>` assertions
//     below are that half: they stop compiling if `bash_record` leaves the policy
//     block, if `per_via` gains or loses a key, or if a key's type drifts from what
//     egress.py:1327-1337 writes. "Exactly these six keys, and no others" is a
//     claim about a type, and a type is only checkable by the compiler.
//   * vitest — where types are erased and an interface has no runtime existence, so
//     the text pins on `api.ts?raw` are what a test run can actually fail on. `?raw`
//     is the device DashboardResponsive.test.ts uses for the same stated reason: this
//     project's tsconfig ships no node types, so `node:fs` does not type-check here.
import apiSource from "./api.ts?raw";
import type { NetworkState } from "./api";

/** The six keys the emit site's row dict writes — agent_mcp/egress.py:1327-1337:
 *  `via`/`count` at :1328, `distinct_hosts`/`denies` at :1329, `last_at` at :1330,
 *  `scopes` at :1337, over the grouped query that closes at :1349. */
const VIA_KEYS = ["via", "count", "distinct_hosts", "denies", "last_at", "scopes"];

/** What the same six keys must be typed as. `scopes` is the per-via list of
 *  `{scope, count}` built at egress.py:1326; `last_at` is `MAX(at)` over a
 *  `at TEXT NOT NULL` column (egress.py:134, :1345), so a row that exists has a
 *  string — never the null an outer join would produce. */
type ExpectedViaRow = {
  via: string;
  count: number;
  distinct_hosts: number;
  denies: number;
  last_at: string;
  scopes: { scope: string; count: number }[];
};

type ViaRow = NonNullable<NetworkState["per_via"]>[number];

/** True only when the two types are identical in both directions — the strict
 *  conditional-identity form, so a key ADDED to the reader fails it exactly as a
 *  key removed does. `extends`-shaped checks pass on a supertype and would let a
 *  seventh key in silently. */
type Exact<A, B> =
  (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;

/** The payload shape, as the live probe above returned it. Its annotation is the
 *  assertion: an omitted required key or a wrong row key is a compile error here. */
const SAMPLE: NetworkState = {
  window_days: 7,
  total: 494,
  distinct_hosts: 89,
  by_decision: { allow: 494, deny: 0, "grant-required": 0 },
  per_destination: [],
  per_scope: [],
  per_via: [{
    via: "tool", count: 494, distinct_hosts: 89, denies: 0,
    last_at: "2026-11-22T14:33:46+00:00",
    scopes: [{ scope: "worker:deep-research", count: 435 }],
  }],
  database_present: true,
  policy: { telemetry: true, enforce: false, bash_record: false,
            allow_entries: 7, permanent_allow_entries: 7 },
};

/** The interface's own block — the same range the item's check extracts with
 *  `awk '/export interface NetworkState/,/^}/'`. A `}` at column 0 ends it, so the
 *  nested object literals inside cannot close the match. */
const STATE_BLOCK =
  (apiSource.match(/export interface NetworkState \{[\s\S]*?^\}/m) ?? [""])[0];

/** One `key: type` list body, keys in source order, from inside `decl`. */
function keysOf(decl: string, name: string): string[] {
  const body = decl.match(new RegExp(`${name}:\\s*\\{([\\s\\S]*?)\\}`))?.[1] ?? "";
  return [...body.matchAll(/([A-Za-z_][A-Za-z0-9_]*)\s*:/g)].map((m) => m[1]);
}

describe("NetworkState per-via split", () => {
  it("extracts the interface block the item's awk check extracts", () => {
    // Without this the greps below could be reading an empty string and every
    // `not.toContain` would pass on nothing.
    expect(STATE_BLOCK.length).toBeGreaterThan(100);
    expect(STATE_BLOCK).toContain("database_present");
  });

  it("declares policy.bash_record beside the four keys the payload already carried", () => {
    // Clause 1. Order is the payload's, but the claim being pinned is the key
    // set, so this compares sorted — "the five keys the payload actually
    // carries", not "in this spelling order".
    expect(keysOf(STATE_BLOCK, "policy").sort())
      .toEqual(["allow_entries", "bash_record", "enforce",
                "permanent_allow_entries", "telemetry"].sort());
    expect(STATE_BLOCK).toContain("bash_record: boolean");
  });

  it("declares per_via as optional, so an older backend stays readable", () => {
    // Clause 2's optionality half: a tab open across a backend restart polls the
    // new build with the old snapshot shape (the `sectionOk` rule above), and a
    // required `per_via` would make that a type error the panel cannot catch.
    expect(STATE_BLOCK).toMatch(/per_via\?:/);
    const sample: NetworkState = { ...SAMPLE, per_via: undefined };
    delete (sample as { per_via?: unknown }).per_via;
    expect(sectionOk(sample)).toBe(true);
  });

  it("types a per_via row as exactly the six keys egress.py writes, and no others", () => {
    // Clause 2's exactness half. `Exact` is compile-time — the gate's tsc delta
    // is what makes it fail — and the runtime line beneath it is what makes the
    // fixture itself honest rather than a node that cannot fail.
    const rowIsExactlyTheSixKeys: Exact<ViaRow, ExpectedViaRow> = true;
    expect(rowIsExactlyTheSixKeys).toBe(true);
    const scopesAreScopeCountPairs:
      Exact<NonNullable<ViaRow["scopes"]>[number], { scope: string; count: number }> = true;
    expect(scopesAreScopeCountPairs).toBe(true);
    expect(Object.keys(SAMPLE.per_via![0]).sort()).toEqual([...VIA_KEYS].sort());
  });

  it("keeps all six key names inside the interface block itself", () => {
    // The row type is written inline, the way its two sibling lists already are,
    // so the awk block a reviewer extracts is the whole contract rather than a
    // pointer to a named type elsewhere in a 3000-line file.
    for (const key of VIA_KEYS) {
      expect(STATE_BLOCK).toMatch(new RegExp(`\\b${key}:`));
    }
  });
});
