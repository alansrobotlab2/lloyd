# 3D Memory Graph

## Context

The Memory tab in Mission Control renders an entity/knowledge graph in [EntityGraph.tsx](../web/src/components/EntityGraph.tsx) using `react-force-graph-2d`. The user wants a 3D view with drag-to-rotate and scroll-to-zoom, inspired by [the-palindrome.github.io/ml-knowledge-graph](https://the-palindrome.github.io/ml-knowledge-graph/).

The reference site is a custom three.js build (OrbitControls + LineSegments2 + instanced meshes). We don't need that lift — the current graph is already from vasturiano's `react-force-graph` family, and the same author ships `react-force-graph-3d` (v1.29.1, matching our installed 2D version) with a nearly identical API. It wraps `3d-force-graph` which bundles three.js and ships the exact controls requested:

- **Drag** → rotate (TrackballControls default, or OrbitControls via `controlType`)
- **Scroll** → zoom
- **Right-drag** → pan

This is a focused component swap, not an architectural change.

## Approach

Replace the `ForceGraph2D` usage inside [EntityGraph.tsx](../web/src/components/EntityGraph.tsx) with `ForceGraph3D`. The component's external API (`selectedNode`, `onNodeClick`, `onNodeDoubleClick`, `onNodeHover`) and its data flow (`api.entityGraph()`, edge filters, highlight dimming, degree calculation) stay identical — only the rendering primitives change.

### 1. Dependency

Install `react-force-graph-3d@^1.29.1`. No separate `three` install needed — it comes via `3d-force-graph`. Keep `react-force-graph-2d` in `package.json` for now in case we want a 2D/3D toggle later; remove in a follow-up if unused.

### 2. Component changes in [web/src/components/EntityGraph.tsx](../web/src/components/EntityGraph.tsx)

- **Import swap**: `ForceGraph2D` → `ForceGraph3D` (line 2).
- **Drop canvas rendering**: remove `nodeCanvasObject` (lines 274-329) and `nodePointerAreaPaint` (lines 331-339). Replace with a `nodeThreeObject` callback that returns a `THREE.Mesh`:
  - Entity nodes (`node.type === "entity"`): `OctahedronGeometry` (3D diamond) in gold `#F59E0B`.
  - Document nodes: `SphereGeometry` with radius from existing `nodeRadius()` (already size-by-degree).
  - Material: `MeshLambertMaterial` with `color` from existing `nodeColor()`.
  - For highlight/dimming: set `material.opacity` and `transparent = true`, mirroring the alpha logic at line 278.
- **Labels**: use `nodeThreeObjectExtend={true}` plus a `SpriteText` for labels (tiny library, same author — or inline via three.js `Sprite` with canvas texture). Gate visibility by `hasDimming && isLit || zoom threshold` to match current behavior at line 316.
- **Edges**: `ForceGraph3D` supports the same `linkColor`, `linkWidth`, `linkCurvature` props — keep `linkColorFn` / `linkWidthFn` as-is.
- **Forces**: `configureForces()` at lines 234-244 — the 3D variant runs the same d3-force simulation but in 3D, so `forceX(0)` / `forceY(0)` still work and we add `forceZ(0)` centering. Bump `distanceMax` modestly since 3D has more volume.
- **Fit / zoom**: replace `fgRef.current.zoomToFit(ms, px)` with `fgRef.current.zoomToFit(ms, px, () => true)` (3D version signature). `centerAt()` becomes `cameraPosition({ x, y, z }, lookAt, ms)`.
- **Controls**: default is TrackballControls — matches "drag up/down/left/right rotates" exactly. Add `controlType="orbit"` prop if we prefer an upright horizon (OrbitControls locks the up-vector). **Default to trackball** since the user described free rotation; easy to flip later.
- **Min zoom**: remove the `minZoom` prop (2D-only). 3D zoom is camera distance — clamp via `controls.minDistance` / `controls.maxDistance` if needed (probably not initially).
- **Background**: set `backgroundColor="rgba(0,0,0,0)"` to keep the transparent integration with the surrounding surface-0 panel.

### 3. Interaction parity

All of the following already go through props that `ForceGraph3D` also accepts — no logic changes needed:

- `onNodeClick` / `onNodeHover` / `onBackgroundClick`
- Double-click detection via `lastClickRef` (line 107, lines 386-408)
- Hover debounce (line 110, lines 375-382)
- Edge type filters (lines 100-104, lines 417-419)
- Highlight/dim sets (lines 171-194) — applied through material opacity instead of `ctx.globalAlpha`

### 4. UI chrome

- Keep legend, edge-filter checkboxes, reset button, and fit-all button (lines 453-546) unchanged.
- Update the fit button handler to use the 3D `zoomToFit` signature.

## Critical files

- [web/src/components/EntityGraph.tsx](../web/src/components/EntityGraph.tsx) — only file with substantive edits
- [web/package.json](../web/package.json) — add `react-force-graph-3d` dependency
- [web/src/components/pages/MemoryPage.tsx](../web/src/components/pages/MemoryPage.tsx) — no changes needed; consumes `<EntityGraph>` with stable props

## Out of scope (can be follow-ups)

- Auto-rotate, bloom, category-vs-metric node coloring modes, multiple layouts (force/hierarchical/cluster/radial), prerequisite/dependent filters — all present on the reference site but beyond the user's ask.
- 2D/3D toggle — straightforward to add if 3D performance is poor on very large graphs, but start with a straight swap.

## Verification

1. Install deps: `cd web && pnpm add react-force-graph-3d` (or npm/yarn — check the lockfile).
2. Vite HMR picks up the frontend change automatically; no supervisord restart needed. If HMR doesn't reload, restart `lloyd-mc:lloyd-frontend` per [CLAUDE.md](../CLAUDE.md).
3. Open Mission Control → Memory tab. Confirm:
   - Graph renders in 3D with visible depth (z-axis).
   - Drag rotates the graph. Scroll zooms in/out. Right-drag pans.
   - Single-click selects a node, double-click fires `onNodeDoubleClick`, clicking background deselects.
   - Hovering highlights the node and its neighbors; non-neighbors dim.
   - Edge-type filter checkboxes still toggle wiki-link / tag-cluster / has-facts edges.
   - Reset and fit-all buttons behave as before.
   - Legend and edge filter chrome remain in their current positions.
4. Sanity-check performance with the real dataset size from `api.entityGraph()` — if it chugs on the full graph, consider reducing `cooldownTicks` or lowering geometry segment counts.
