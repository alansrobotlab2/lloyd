import { describe, expect, it } from "vitest";

import * as autonomyHealth from "./autonomyHealth";

// `autonomyHealth` is the render-half module of the #713 evidence columns: it
// decides what one cell of the `refute` column says for one task row. It lives
// here rather than inline in `AutonomyPage.tsx` because the decision is the
// whole content of the column, and the runner that grades this directory is
// `environment: "node"` with no jsdom — a component cannot be mounted, so a
// rule that only exists inside JSX has no node the gate can grade.
describe("evidenceCellState", () => {
  it("reports a rate when the window holds checked claims", () => {
    expect(autonomyHealth.evidenceCellState({
      refuted_or_insufficient_rate: 0.25, claims_checked: 4,
    })).toEqual({ kind: "rate", text: "25%", bad: true });
  });

  it("calls a clean rate clean, and does not colour it as a finding", () => {
    expect(autonomyHealth.evidenceCellState({
      refuted_or_insufficient_rate: 0, claims_checked: 8,
    })).toEqual({ kind: "rate", text: "0%", bad: false });
  });

  it("says unevaluable for a null rate over runs that carried no bundle", () => {
    // This is the state the whole column exists to keep distinct: the pilot ran,
    // the model asserted nothing or the bundle never arrived, and there is no
    // verdict to report. Rendering `0%` here is the false-clean reading.
    expect(autonomyHealth.evidenceCellState({
      refuted_or_insufficient_rate: null, runs_without_bundle: 6,
    })).toEqual({ kind: "unevaluable", text: "n/a", bad: false });
  });

  it("says not-checked for a null rate over runs that were never checked", () => {
    // A task outside the #525 pilot: no rate, and no `runs_without_bundle`
    // either, because nothing was expected of it. `n/a` would claim a gap in a
    // measurement this task was never part of.
    expect(autonomyHealth.evidenceCellState({})).toEqual(
      { kind: "unchecked", text: "—", bad: false });
  });

  it("keeps a zero rate from a checked window out of the unevaluable branch", () => {
    // `0` is a number and `?? 0`-style flattening would turn a null into one, so
    // the branch has to test null-ness, not truthiness. One checked claim keeps
    // the cell a rate even when the count of no-bundle runs is large.
    expect(autonomyHealth.evidenceCellState({
      refuted_or_insufficient_rate: 0, claims_checked: 1,
      runs_without_bundle: 30,
    }).kind).toBe("rate");
  });
});

describe("noBundleState", () => {
  it("warns only when no-bundle runs outnumber the runs that carried one", () => {
    // The rate beside it describes a slice nobody chose exactly when this is
    // true, which is the rule #525 held for the payload and #713 holds for the
    // cell.
    expect(autonomyHealth.noBundleState({
      runs_without_bundle: 5, runs_with_bundle: 2,
    })).toEqual({ count: 5, warn: true });
    expect(autonomyHealth.noBundleState({
      runs_without_bundle: 2, runs_with_bundle: 5,
    })).toEqual({ count: 2, warn: false });
  });

  it("treats a task the rollup never saw as no gap, not as a gap of zero", () => {
    expect(autonomyHealth.noBundleState({})).toEqual({ count: 0, warn: false });
  });
});

describe("fleetEvidenceState", () => {
  it("leads with the coverage gap when the window is unevaluable", () => {
    expect(autonomyHealth.fleetEvidenceState({
      unevaluable: true, runs_without_bundle: 9, artifacts_with_refutations: 4,
      claims_checked: 3,
    })).toEqual({ tone: "unknown", text: "evidence: 9 run(s) unchecked" });
  });

  it("names misreported artifacts over a clean checked window", () => {
    expect(autonomyHealth.fleetEvidenceState({
      unevaluable: false, runs_without_bundle: 0,
      artifacts_with_refutations: 2, claims_checked: 12,
    })).toEqual({ tone: "bad", text: "2 artifact(s) misreported" });
  });

  it("says nothing was checked rather than reporting a clean fleet", () => {
    // claims_checked 0 with unevaluable false is the pre-#525 shape: the rate is
    // null because there is nothing to rate, not because the fleet is clean.
    expect(autonomyHealth.fleetEvidenceState({
      unevaluable: false, runs_without_bundle: 0,
      artifacts_with_refutations: 0, claims_checked: 0,
    })).toEqual({ tone: "unchecked", text: "evidence: not checked" });
    expect(autonomyHealth.fleetEvidenceState(undefined)).toEqual(
      { tone: "unchecked", text: "evidence: not checked" });
  });

  it("calls a window with refutation-free claims clean", () => {
    expect(autonomyHealth.fleetEvidenceState({
      unevaluable: false, runs_without_bundle: 1,
      artifacts_with_refutations: 0, claims_checked: 12,
    }).tone).toBe("clean");
  });
});

// `api.ts` declaration parity is pinned in `AutonomyPage.test.ts`, which reads
// the file through vite's `?raw` import: this runner has no `@types/node`, so
// `node:fs` does not type-check here, and a test that cannot compile is a test
// that never ran.
