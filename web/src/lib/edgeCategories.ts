// Shared edge-category classification for the entity graph UI.
// The typed classifier emits ~25 distinct edge types; we collapse them into
// four semantic categories so the graph legend, filter toggles, and any
// sidebar chips all agree on color + grouping.

export type EdgeCategory = "structural" | "lineage" | "comparison" | "reference";

const EDGE_CATEGORY: Record<string, EdgeCategory> = {
  // Structural: how things compose
  uses: "structural",
  part_of: "structural",
  implements: "structural",
  depends_on: "structural",
  has_dependency: "structural",
  extends: "structural",
  runs_in: "structural",
  // Lineage: temporal / evolution
  supersedes: "lineage",
  created_by: "lineage",
  inspired_by: "lineage",
  follow_up: "lineage",
  // Comparison: alternatives, contrast
  competes_with: "comparison",
  competing_framework: "comparison",
  contrasts_with: "comparison",
  alternative_optimization: "comparison",
  compared_with: "comparison",
  independent_implementation: "comparison",
  benchmarked_on: "comparison",
  // Reference: discussion / weak semantic / default
  mentions: "reference",
  discusses: "reference",
  related_to: "reference",
  co_mentioned: "reference",
  wiki_link_co_occurrence: "reference",
  references: "reference",
  interacts_with: "reference",
  supports: "reference",
  produces: "reference",
};

export function categoryOf(type: string): EdgeCategory {
  return EDGE_CATEGORY[type] ?? "reference";
}

// Base RGB strings (no alpha); callers pick alpha per render state.
export const CATEGORY_COLOR: Record<EdgeCategory, string> = {
  structural: "59,130,246",   // blue
  lineage: "168,85,247",      // purple
  comparison: "239,68,68",    // red
  reference: "148,163,184",   // slate
};

// Tailwind classes for small chip badges in sidebars. Picking muted shades
// that match the graph colors without screaming.
export const CATEGORY_CHIP_CLASS: Record<EdgeCategory, string> = {
  structural: "bg-blue-400/10 text-blue-400",
  lineage: "bg-purple-400/10 text-purple-400",
  comparison: "bg-red-400/10 text-red-400",
  reference: "bg-slate-400/10 text-slate-500",
};
