# Force-Directed Tag Graph — Implementation Spec

A complete specification for building an interactive tag co-occurrence graph visualization using `react-force-graph-2d` and `d3-force`. The graph renders inside a dashboard tab, showing tags from an Obsidian vault as nodes, with edges representing tags that appear together on the same document.

---

## Dependencies

```json
{
  "d3-force": "^3.0.0",
  "@types/d3-force": "^3.0.10",
  "react-force-graph-2d": "^1.29.1",
  "lucide-react": "any"
}
```

Imports from `d3-force`: `forceCollide`, `forceRadial`, `forceX`, `forceY`.

---

## Data Model

### Backend API

Endpoint: `GET /api/mc/tag-graph`

Returns `{ nodes, edges }`:

- **nodes**: `{ id: string, label: string, count: number }[]` — each tag, with `count` = number of documents containing that tag.
- **edges**: `{ source: string, target: string, weight: number }[]` — co-occurrence edges. Two tags share an edge if they appear on the same document. `weight` = number of shared documents.

Edge computation: for every document, generate all pairs of its tags. Accumulate pair counts across all documents.

### Frontend Graph Data

The API data is augmented client-side into `GData`:

```ts
interface GNode { id: string; label: string; count: number; isHub: boolean }
interface GLink { source: string; target: string; weight: number }
interface GData { nodes: GNode[]; links: GLink[] }
```

**Hub node**: A synthetic central node (`id: "__hub__"`, `label: "Vault"`, `count: 0`, `isHub: true`) is added. Hub links connect the hub to every tag node (weight 1). This creates a radial star topology with the hub at the center.

**Tag labels**: All tag node labels are prefixed with `#` (e.g., `#project`, `#ai`).

---

## Visual Design

### Color Palette

- **Background**: transparent (inherits from parent `bg-surface-0`, a dark theme background)
- **Node fill**: Deterministic hue from tag name hash, muted dark pastels: `hsl(hue, 30%, 55%)`
  - Hash: `hash = (hash * 31 + charCode) | 0` per character, `hue = ((hash % 360) + 360) % 360`
- **Node outline** (highlighted only): `#64748b` (slate-500), `lineWidth: 2 / globalScale`
- **Label text**: `#e2e8f0` (slate-200), monospace font (`ui-monospace, monospace`)
- **Dimmed elements** (not highlighted): `globalAlpha = 0.12` for nodes, `0.1` for labels

### Node Sizing

Radius is a function of document count:

```
radius = max(1.5, min(6, 1.5 + sqrt(count) * 0.75))
```

- Minimum: 1.5px
- Maximum: 6px
- Scale: square root of document count

### Hub Node

Rendered as a brain emoji (`🧠`) at `28 / globalScale` font size, always pinned at `(0, 0)`.

### Labels

Labels are shown when any of these conditions are met:
- `globalScale > 1.2` (zoomed in enough)
- `node.count >= 8` (high-traffic tag)
- Node is highlighted (hovered or selected neighbor)

Font size: `max(10 / globalScale, 2)`.

### Link Colors

| State | Color |
|-------|-------|
| Hub link, normal | `rgba(100,116,139,0.2)` |
| Hub link, dimmed | `rgba(71,85,105,0.03)` |
| Co-occurrence link, normal | `rgba(120,140,165,0.3)` |
| Co-occurrence link, highlighted | `rgba(168,183,204,0.85)` |
| Co-occurrence link, dimmed | `rgba(71,85,105,0.04)` |

### Link Widths

- Hub links: `0.3`
- Highlighted links: `1.5`
- Normal co-occurrence: `min(2, 0.3 + weight * 0.2)`

### Pointer Hit Area

Generous hit area for small nodes: `max(6, tagRadius(count) + 4)` for tag nodes, `16` for hub.

### Tooltip

HTML tooltip on hover:
- Hub: `<b>Vault Hub</b>`
- Tag: `<b>#tagname</b><br/><span style="color:#94a3b8;font-size:11px">N docs</span>`

---

## Interaction States

### 1. Default View (no selection, no hover)

- All nodes visible at full opacity
- All links visible at their normal colors
- Hub pinned at center, all other nodes free
- Standard force simulation running

### 2. Hover (no selection active)

When hovering over a tag node:
- The hovered node + all its immediate neighbors (via co-occurrence edges) are highlighted at full opacity
- All other nodes dim to `alpha 0.12`
- Links between highlighted nodes shown at highlighted color; all others dimmed
- Hub is included in hover highlighting (unlike selection)

### 3. Selection (click a tag node)

When clicking a tag node, it becomes the selected node. The graph transitions into an **isolated subgraph view**:

- **Selected node**: anchored at screen center `(0, 0)` via `fx/fy`
- **Neighbors**: all tags connected to the selected node via co-occurrence edges (hub links excluded from neighbor computation)
- **Hub**: treated as background — fades with everything else, hub links excluded from highlight set
- **Background nodes**: frozen in place at their current positions via `fx/fy`, dimmed to `alpha 0.12`
- **Forces**: completely reconfigured to only affect the active subgraph (selected + neighbors). Background nodes have zero collision, charge, and link strength.

**Clicking the same node again** deselects it, restoring the full graph view.
**Clicking the background** also deselects.
**Clicking the hub** deselects and resets the view.

### 4. Search

Text search in toolbar filters nodes by label substring match (case-insensitive). Matching nodes are highlighted; non-matching nodes dim.

---

## Animation System

All animations use `requestAnimationFrame` with **ease-out quadratic** easing:

```ts
const ease = 1 - (1 - t) * (1 - t);  // t goes from 0 to 1
```

A single `animFrameRef` (React ref holding the current animation frame ID) is used. Every new animation cancels the previous one via `cancelAnimationFrame(animFrameRef.current)`.

### Initial Load

On first data load (gated by `initialFitDone` ref):
1. Wait 300ms for warmup ticks to settle
2. `zoomToFit(1000, 50)` — animate the viewport to fit all nodes with 50px padding
3. After 1100ms (animation complete), capture `fgRef.current.zoom()` as the min zoom level (multiplied by 0.95 for slight margin)

No `onEngineStop` handler — the graph only auto-fits once on initial load, never again.

### Selection Animation (the core choreography)

When a tag node is clicked, a **single simultaneous animation** runs over **1200ms**:

#### Setup Phase (instant, before animation starts):

1. **Build neighbor set**: walk all co-occurrence links (excluding hub links) to find nodes connected to the selected node.
2. **Compute circle targets**: neighbors are sorted by document count (largest first), then assigned evenly-spaced angles around a circle centered at `(0, 0)`.
   - Base radius: `max(40, 15 + neighborCount * 4)`
   - Per-node distance factor: larger nodes orbit closer, smaller ones farther:
     ```
     factor = 1.4 - 0.8 * (tagRadius(count) - 1.5) / (6 - 1.5)
     ```
     - Smallest nodes (radius 1.5): factor = 1.4 (farthest)
     - Largest nodes (radius 6): factor = 0.6 (closest)
   - Target position: `(cos(angle) * dist, sin(angle) * dist)`
3. **Capture start positions**: record current `(x, y)` for the selected node and each neighbor.
4. **Freeze everything**: set `fx = x, fy = y, vx = 0, vy = 0` on ALL non-hub nodes (both neighbors and background). They'll be moved by the animation, not by forces.

#### Animation Phase (1200ms, ease-out quadratic):

Each frame simultaneously interpolates:
- **Selected node**: `fx` and `fy` from `(fromX, fromY)` to `(0, 0)`
- **Each neighbor**: `fx` and `fy` from `(sx, sy)` to `(tx, ty)` (their circle target)

```ts
node.fx = fromX * (1 - ease);
node.fy = fromY * (1 - ease);
for (const { nd, sx, sy, tx, ty } of neighborAnim) {
  nd.fx = sx + (tx - sx) * ease;
  nd.fy = sy + (ty - sy) * ease;
}
```

During this phase, `zoomToFit(1200, 80, filter)` runs in parallel (the library handles its own animation), filtering to only the neighbor set so the camera zooms to the forming cluster.

#### Settle Phase (after animation completes):

1. **Anchor selected node**: `fx = 0, fy = 0` (locked at center permanently)
2. **Unpin neighbors**: `fx = undefined, fy = undefined, vx = 0, vy = 0` — they're now at their circle positions but free to be nudged by forces
3. **Configure selection forces** (see Force Configuration below)
4. **Reheat simulation**: `d3ReheatSimulation()` — forces gently settle neighbors into final positions

### Deselection Animation

When deselecting (click same node, click background, or click hub):
1. Cancel any running animation
2. Unpin all non-hub nodes (`fx = undefined, fy = undefined`)
3. Restore default force configuration
4. Reheat simulation
5. `zoomToFit(1200, 50)` to animate back to the full graph view

---

## Force Configuration

Forces are managed through a `configureForces(activeIds?, selectedId?)` helper. When called with no arguments, it configures the default full-graph forces. When called with `activeIds` (the neighbor set) and `selectedId`, it configures isolated subgraph forces.

### Default Forces (full graph)

| Force | Configuration |
|-------|---------------|
| **Collision** | `forceCollide`, radius = `tagRadius(count) + 4` (hub: 20), strength 0.5, 1 iteration |
| **Charge** | `forceManyBody`, strength -80, distanceMax 400 |
| **Link distance** | Hub links: 180. Co-occurrence: `max(20, 60 - weight * 8)` |
| **Link strength** | Hub links: 0.015. Co-occurrence: `min(0.6, 0.15 + weight * 0.08)` |
| **Center** | Built-in center force (from react-force-graph-2d) |
| **CenterX/Y** | null (not used) |
| **Radial** | null (not used) |

### Selection Forces (isolated subgraph)

The `isActive(node)` predicate returns `true` only for nodes in `activeIds`. The hub is **never** active during selection — it's treated as background.

| Force | Configuration |
|-------|---------------|
| **Collision** | Active: radius = `tagRadius(count) + 10`, strength 0.5. Inactive: radius 0 |
| **Charge** | Active: strength -200, distanceMax 200. Inactive: strength 0 |
| **Link distance** | Hub links: 60 (irrelevant since strength is 0). Co-occurrence: `max(20, 60 - weight * 8)` |
| **Link strength** | Any link touching an inactive node: 0. Hub links: 0. Active co-occurrence: `min(0.6, 0.15 + weight * 0.08)` |
| **CenterX/Y** | `forceX(0)` and `forceY(0)` with strength 0.05 for active neighbors (0 for selected node, inactive, and hub) |
| **Radial** | `forceRadial` centered at (0,0) with size-dependent radius (same factor formula as animation targets). Strength 0.8 for active neighbors, 0 for selected/hub/inactive |
| **Center** | Disabled (`null`) — the built-in center force would pull ALL nodes including background |

### Simulation Parameters

Set on the `ForceGraph2D` component:

| Parameter | Value |
|-----------|-------|
| `d3AlphaDecay` | 0.025 |
| `d3VelocityDecay` | 0.8 (high damping for slow, deliberate movement) |
| `cooldownTicks` | 400 |
| `warmupTicks` | 80 |
| `minZoom` | Dynamically captured after initial fit (full-graph zoom level × 0.95) |

---

## Layout & Toolbar

The component fills its parent container (`flex-1 relative min-h-0 overflow-hidden`). A `ResizeObserver` tracks the container and passes `width`/`height` to the `ForceGraph2D` component.

### Toolbar (overlaid at top)

Semi-transparent bar: `bg-surface-0/80 backdrop-blur-sm border-b border-surface-3/20`

Contains:
1. **Search input**: filters/highlights nodes by label substring
2. **Reset button** (RotateCcw icon): clears selection + search, fits all nodes
3. **Fit button** (Maximize icon): fits all nodes to viewport
4. **Stats label**: `"{N} tags · {M} co-occurrences"` (right-aligned, 10px slate-500 text)

All zoom animations use 1200ms duration.

---

## Key Implementation Details

### Why `fx/fy` Animation Instead of Forces

Forces alone couldn't reliably distribute neighbors in a full 360° circle — they'd trail the direction of travel and clump to one side. The solution: bypass the simulation entirely during transition by animating `fx/fy` (d3-force's "fixed position" pins) via `requestAnimationFrame`. Forces only take over after nodes are already at their target positions.

### Why the Hub is Excluded from Selection Highlights

During selection, the graph shows an isolated subgraph. Including the hub would visually connect the selected cluster to the entire graph through hub links, breaking the isolation metaphor. The hub fades with the background.

### Why Background Nodes are Frozen

Setting `fx = currentX, fy = currentY` on background nodes prevents forces from moving them during selection. Without this, the strong radial/charge forces on the active subgraph would push background nodes around, causing visual noise.

### Node Drag

Node dragging is enabled. On drag end, the simulation is reheated (`d3ReheatSimulation()`) so neighboring nodes settle into new positions.

---

## File Structure

```
web/src/components/pages/GraphPage.tsx  — main component (all logic in one file)
web/src/api.ts                          — API client with TagGraphData types
backend/index.ts                        — /api/mc/tag-graph endpoint
web/package.json                        — dependencies
```

The component is registered in the app's page routing (`Layout.tsx` PAGES record) and navigation sidebar (`Sidebar.tsx` Page type union + nav item with `GitBranch` icon from lucide-react).
