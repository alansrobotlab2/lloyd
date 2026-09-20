import { describe, expect, it } from "vitest";

import { sectionError, sectionOk } from "./api";

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
