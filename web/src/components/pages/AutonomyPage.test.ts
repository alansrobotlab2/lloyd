import { describe, expect, it } from "vitest";

import apiSource from "../../api.ts?raw";
import pageSource from "./AutonomyPage.tsx?raw";

// The autonomy tab's `refute` / `no-bundle` columns are the half of #713 that
// gets read: `compute_health` has returned `refuted_or_insufficient_rate` per
// task id since #525, and the tab still showed only fail/t-o/empty/silent/GPU-h/
// wasted/consec, so a rising refutation rate was visible only in `/health` JSON.
// There is no DOM in this runner (`environment: "node"`, no jsdom — adding one
// means `web/package.json`, a human-only build input), so what is pinned here is
// the two things that actually carry the data to a row: the payload type that
// declares the fields, and the component that renders them. An `interface` with
// no such field drops the field on the floor before any component can ask, which
// is precisely how #525's numbers stayed unread for four days.
describe("AutonomyHealthTask evidence fields", () => {
  const taskBlock = apiSource.slice(
    apiSource.indexOf("export interface AutonomyHealthTask {"),
    apiSource.indexOf("export interface AutonomyHealthClaimTally"),
  );

  it("declares the per-task refutation rate as a number OR null", () => {
    // Null is the unevaluable state and must stay expressible: a `number` here
    // would force a 0 into the column for a window with no checked claim, which
    // is the false-clean reading the column exists to prevent.
    expect(taskBlock).toMatch(
      /refuted_or_insufficient_rate\?: number \| null;/);
  });

  it("declares the bundle-coverage counts the unevaluable state is read from", () => {
    expect(taskBlock).toMatch(/runs_without_bundle\?: number;/);
    expect(taskBlock).toMatch(/runs_with_bundle\?: number;/);
  });

  it("declares the evidence keys on every block the tab reads, not just one", () => {
    // Positive control, and the reason it is here rather than implied: both
    // blocks above are cut with `indexOf`, and an `indexOf` that misses returns
    // -1 — which slices an EMPTY string, against which every `toContain` would
    // still have to fail, so a renamed interface would read as a missing field
    // rather than as the block no longer being found. Each block is therefore
    // shown to be non-empty by a key that must exist in it whatever else
    // changes, and a field the server does not send must not be declared here.
    expect(taskBlock).toContain("task_id: string;");
    expect(taskBlock).toContain("name: string | null;");
    expect(apiSource).toContain("evidence_unevaluable_reason");
    for (const invented of ["refuted_count", "worst_path", "misreported_total"]) {
      expect(apiSource, `api.ts declares ${invented}, which no route emits`)
        .not.toContain(`${invented}:`);
    }
  });
});

describe("AutonomyHealth artifact rollup", () => {
  const rollupBlock = apiSource.slice(
    apiSource.indexOf("export interface AutonomyHealthClaimRollup {"),
    apiSource.indexOf("/** An `idle_tasks` row"),
  );
  const entryBlock = apiSource.slice(
    apiSource.indexOf("export interface AutonomyHealthArtifactEntry {"),
    apiSource.indexOf("export interface AutonomyHealthClaimRollup"),
  );

  it("hangs the rollup off the payload the page already fetches", () => {
    const healthBlock = apiSource.slice(
      apiSource.indexOf("export interface AutonomyHealth {"));
    expect(healthBlock.slice(0, 2000)).toMatch(
      /artifacts\?: AutonomyHealthClaimRollup;/);
  });

  it("names the artifact, the tally, and every task that claimed about it", () => {
    expect(entryBlock).toMatch(/^  path: string;$/m);
    expect(entryBlock).toMatch(/refuted_or_insufficient: number;/);
    // The cross-task pair is the reason the rollup exists: the job that made
    // the claim and the job whose run refuted it are two different tasks.
    expect(entryBlock).toMatch(/task_ids: string\[\];/);
    expect(entryBlock).toMatch(
      /per_task: Record<string, AutonomyHealthClaimTally>;/);
  });

  it("keeps the unevaluable state in the type, not only the arithmetic", () => {
    expect(rollupBlock).toMatch(/unevaluable: boolean;/);
    expect(rollupBlock).toMatch(/unevaluable_reason: string \| null;/);
    expect(rollupBlock).toMatch(/runs_without_bundle: number;/);
    expect(rollupBlock).toMatch(/refuted_or_insufficient_rate: number \| null;/);
  });
});

describe("HealthStrip renders the evidence", () => {
  // Code, not prose. Every string the assertions below look for also appears in a
  // comment explaining why the column exists, so a page that declared nothing and
  // merely described the columns in a comment would pass a raw `toContain` — which
  // is what happened to the `evidenceCellState` import check at attempt 1, where
  // the page had a local wrapper and never named the helper at all. Line comments
  // and JSX comments are stripped first, so what remains is what the page runs.
  const code = pageSource
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, "")
    .split("\n").filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l)).join("\n");

  it("the comment stripper left real code behind, so it cannot pass by erasure", () => {
    expect(code).toContain("function HealthStrip");
    expect(code.length).toBeGreaterThan(pageSource.length / 2);
    // And it did remove the prose: this phrase is in a JSX comment above the chip,
    // so it is in `pageSource` and must be gone from `code`. Without both
    // directions the strip could silently remove nothing and every assertion below
    // would be back to matching on comments.
    const PROSE = "must not borrow the clean";
    expect(pageSource).toContain(PROSE);
    expect(code).not.toContain(PROSE);
  });

  it("adds the two columns beside fail/silent rather than replacing them", () => {
    for (const existing of ["fail", "t/o", "empty", "silent", "GPU-h", "wasted",
                            "consec"]) {
      expect(pageSource).toContain(`>${existing}<`);
    }
    expect(pageSource).toContain(">refute<");
    expect(pageSource).toContain(">no-bundle<");
  });

  it("renders the cell through the graded render rules, not an inline guess", () => {
    // Which of the three states a row is in is `autonomyHealth.ts`'s decision and
    // is pinned there; what is pinned here is that the page asks it rather than
    // reimplementing the branch beside the markup, where a second copy could
    // drift from the first.
    // The page imports the rule module and calls it on the row. `evidenceCell`
    // itself stays in the page because all it does is choose a colour and a
    // tooltip for a state the module already decided; what must not live there is
    // the decision, which is the thing the node runner can execute.
    expect(code).toMatch(
      /import \{[^}]*evidenceCellState[^}]*\} from "\.\/autonomyHealth"/);
    expect(code).toContain("evidenceCellState(t)");
    expect(code).toContain("{evidenceCell(t)}");
    expect(code).toContain("noBundleState(t).count");
    expect(code).toContain("fleetEvidenceState(health.artifacts)");
  });

  it("shows the no-bundle count itself, not only the rate it qualifies", () => {
    expect(code).toContain("noBundleState(t).count || \"\"");
  });

  it("lists the misreported artifacts with the tasks on both sides", () => {
    expect(code).toContain("Most misreported artifacts");
    expect(code).toContain("{e.refuted_or_insufficient}/{e.claims_checked}");
    expect(code).toContain("{e.task_ids.join(\", #\")}");
  });

  it("prints the rollup's unevaluable reason over the list", () => {
    expect(code).toContain("Artifact rollup unevaluable");
    expect(code).toContain("{health.artifacts.unevaluable_reason}");
  });
});
