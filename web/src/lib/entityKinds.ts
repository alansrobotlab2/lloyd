// Node colour palette for the entity graph, keyed on the entity's KIND
// (app/entity_kind.py) — which is what the graph legend names.
//
// This lives here rather than in EntityGraph.tsx so that consumers which only
// need the palette (MemoryPage's sidebar and detail panels) don't drag in
// three.js and react-force-graph-3d, which is what kept the graph off the
// lazy-loading path.

export const KIND_COLOR: Record<string, string> = {
  person:  "#F472B6",
  project: "#A78BFA",
  system:  "#F59E0B",
  concept: "#38BDF8",
  skill:   "#34D399",
  task:    "#FB923C",
  doc:     "#94A3B8",
  entity:  "#64748B",
};

export const KIND_ORDER = ["system", "project", "concept", "person", "skill", "task", "doc", "entity"] as const;
