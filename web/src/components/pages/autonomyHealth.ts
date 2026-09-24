// Render rules for the autonomy tab's evidence columns (#713).
//
// Extracted from `AutonomyPage.tsx` because the decision these encode is the
// entire content of the columns, and the runner that grades this directory is
// `environment: "node"` with no jsdom: a rule that exists only inside JSX has no
// node the gate can grade. The page keeps the markup; the states live here.
//
// The three states are the point. `compute_health` reports
// `refuted_or_insufficient_rate: null` for two entirely different situations — a
// piloted task whose runs carried no claims bundle, and a task outside the #525
// pilot that was never asked to carry any — and both of them would render as a
// blank cell if the branch tested truthiness instead of null-ness. A blank cell
// beside a green `fail` column reads as "fine", which is the false-clean reading
// this whole surface exists to make impossible.

import type { AutonomyHealthClaimRollup } from "../../api";

export interface EvidenceCellInput {
  refuted_or_insufficient_rate?: number | null;
  claims_checked?: number;
  runs_without_bundle?: number;
}

export interface EvidenceCell {
  /** `rate` = a measured share; `unevaluable` = the pilot ran but the window
   *  holds no checked claim; `unchecked` = outside the pilot, nothing expected. */
  kind: "rate" | "unevaluable" | "unchecked";
  text: string;
  /** True only for a measured rate above zero: a verdict, not a gap. */
  bad: boolean;
}

export function evidenceCellState(t: EvidenceCellInput): EvidenceCell {
  const rate = t.refuted_or_insufficient_rate;
  if (rate === null || rate === undefined) {
    return (t.runs_without_bundle ?? 0) > 0
      ? { kind: "unevaluable", text: "n/a", bad: false }
      : { kind: "unchecked", text: "—", bad: false };
  }
  return { kind: "rate", text: `${Math.round(rate * 100)}%`, bad: rate > 0 };
}

export interface NoBundleInput {
  runs_without_bundle?: number;
  runs_with_bundle?: number;
}

export interface NoBundleState {
  count: number;
  /** True when the no-bundle runs outnumber the ones that carried a bundle: the
   *  rate in the neighbouring cell then describes a slice nobody chose, and the
   *  count has to be loud enough to stop the reader trusting it. */
  warn: boolean;
}

export function noBundleState(t: NoBundleInput): NoBundleState {
  const count = t.runs_without_bundle ?? 0;
  return { count, warn: count > (t.runs_with_bundle ?? 0) };
}

export type FleetEvidenceTone = "clean" | "bad" | "unknown" | "unchecked";

export function fleetEvidenceState(
  art?: Pick<AutonomyHealthClaimRollup, "unevaluable" | "runs_without_bundle"
    | "artifacts_with_refutations" | "claims_checked">,
): { tone: FleetEvidenceTone; text: string } {
  if (!art || art.claims_checked === 0) {
    return { tone: "unchecked", text: "evidence: not checked" };
  }
  if (art.unevaluable) {
    return { tone: "unknown",
             text: `evidence: ${art.runs_without_bundle} run(s) unchecked` };
  }
  return art.artifacts_with_refutations > 0
    ? { tone: "bad", text: `${art.artifacts_with_refutations} artifact(s) misreported` }
    : { tone: "clean", text: `evidence clean · ${art.claims_checked} claims` };
}
