# Tag Graph & Memory Tab — Implementation Spec

Interactive tag co-occurrence graph visualization using `react-force-graph-2d` and `d3-force`, embedded in the Memory tab. Tags from an Obsidian vault render as nodes; edges represent tags that appear together on the same document. The `TagGraph` component is driven by external props for bidirectional sync with the Memory tab's sidebar, search, and document viewer.

---

## Dependencies

```json
{
  "d3-force": "^3.0.0",
  "@types/d3-force": "^3.0.10",
  "react-force-graph-2d": "^1.29.1",
  "lucide-react": "any",
  "marked": "any"
}
```

Imports from `d3-force`: `forceCollide`, `forceRadial`, `forceX`, `forceY`.

---

## Data Model

### Backend API

**Endpoint**: `GET /api/mc/tag-graph`

Returns `{ nodes, edges }`:

- **nodes**: `{ id: string, label: string, count: number }[]` — each tag, with `count` = number of documents containing that tag.
- **edges**: `{ source: string, target: string, weight: number }[]` — co-occurrence edges. Two tags share an edge if they appear on the same document. `weight` = number of shared documents.

Edge computation: for every document, generate all pairs of its tags. Accumulate pair counts across all documents. Data sourced from vault index (`~/obsidian/**/*.md` frontmatter, 60-second cache).

### Frontend Graph Data

```ts
interface GNode { id: string; label: string; count: number }
interface GLink { source: string; target: string; weight: number }
interface GData { nodes: GNode[]; links: GLink[] }
```

Pure tag co-occurrence — no hub node, only tag nodes and edges.

**Props interface**:
```ts
interface TagGraphProps {
  selectedTag?: string | null;   // drives graph to select this tag node
  onTagClick?: (tag: string | null) => void;  // fired on click or deselect
}
```

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

### Labels

Labels are shown when any of these conditions are met:
- `globalScale > 1.2` (zoomed in enough)
- `node.count >= 8` (high-traffic tag)
- Node is highlighted (hovered or selected neighbor)

Font size: `max(10 / globalScale, 2)`.

### Link Colors

| State | Color |
|-------|-------|
| Normal | `rgba(120,140,165,0.3)` |
| Highlighted | `rgba(168,183,204,0.85)` |
| Dimmed | `rgba(71,85,105,0.04)` |

### Link Widths

- Highlighted links: `1.5`
- Normal co-occurrence: `min(2, 0.3 + weight * 0.2)`

### Pointer Hit Area

Generous hit area for small nodes: `max(6, tagRadius(count) + 4)`.

### Tooltip

HTML tooltip on hover:
- `<b>#tagname</b><br/><span style="color:#94a3b8;font-size:11px">N docs</span>`

---

## Interaction States

### 1. Default View (no selection, no hover)

- All nodes visible at full opacity
- All links visible at their normal colors
- All nodes free (no pinning)
- Standard force simulation running

### 2. Hover (no selection active)

When hovering over a tag node:
- The hovered node + all its immediate neighbors (via co-occurrence edges) are highlighted at full opacity
- All other nodes dim to `alpha 0.12`
- Links between highlighted nodes shown at highlighted color; all others dimmed

### 3. Selection (click a tag node)

When clicking a tag node, it becomes the selected node. The graph transitions into an **isolated subgraph view**:

- **Selected node**: anchored at screen center via `fx/fy` (computed with `screen2GraphCoords(w/2, h/2)`)
- **Neighbors**: all tags connected to the selected node via co-occurrence edges
- **Background nodes**: frozen in place at their current positions via `fx/fy`, dimmed to `alpha 0.12`
- **Forces**: completely reconfigured to only affect the active subgraph (selected + neighbors). Background nodes have zero collision, charge, and link strength.

**Clicking the same node again** deselects it, restoring the full graph view.
**Clicking the background** also deselects.

### 4. External Selection Sync

The `selectedTag` prop drives graph selection from outside (e.g., clicking a tag in the Memory tab sidebar). A `lastSelectedRef` prevents feedback loops between prop changes and click-driven `onTagClick` callbacks. When `selectedTag` changes:
- `null` → deselect animation
- `"tagname"` → select animation via `animateSelectById()`

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

### Selection Animation

When a tag node is clicked (or `selectedTag` prop changes), a **single simultaneous animation** runs over **1200ms**:

#### Setup Phase (instant, before animation starts):

1. **Build neighbor set**: walk all co-occurrence links to find nodes connected to the selected node.
2. **Compute screen center**: `fg.screen2GraphCoords(w / 2, h / 2)` gets the graph-space coordinates of the viewport center.
3. **Compute circle targets**: neighbors are sorted by document count (largest first), then assigned evenly-spaced angles around a circle centered at the screen-center point.
   - Base radius: `max(40, 15 + neighborCount * 4)`
   - Per-node distance factor: larger nodes orbit closer, smaller ones farther:
     ```
     factor = 1.4 - 0.8 * (tagRadius(count) - 1.5) / (6 - 1.5)
     ```
     - Smallest nodes (radius 1.5): factor = 1.4 (farthest)
     - Largest nodes (radius 6): factor = 0.6 (closest)
   - Target position: `(centerX + cos(angle) * dist, centerY + sin(angle) * dist)`
4. **Capture start positions**: record current `(x, y)` for the selected node and each neighbor.
5. **Freeze everything**: set `fx = x, fy = y, vx = 0, vy = 0` on ALL nodes (both neighbors and background). They'll be moved by the animation, not by forces.

#### Animation Phase (1200ms, ease-out quadratic):

Each frame simultaneously interpolates:
- **Selected node**: `fx` and `fy` from `(fromX, fromY)` to `(centerX, centerY)`
- **Each neighbor**: `fx` and `fy` from `(sx, sy)` to `(tx, ty)` (their circle target)

```ts
node.fx = fromX + (centerX - fromX) * ease;
node.fy = fromY + (centerY - fromY) * ease;
for (const { nd, sx, sy, tx, ty } of neighborAnim) {
  nd.fx = sx + (tx - sx) * ease;
  nd.fy = sy + (ty - sy) * ease;
}
```

During this phase, `fg.centerAt(centerX, centerY, 1200)` pans the camera to the selected node's destination.

#### Settle Phase (after animation completes):

1. **Anchor selected node**: `fx = centerX, fy = centerY` (locked at screen center permanently)
2. **Keep neighbors pinned**: `fx = tx, fy = ty` — they stay at their arranged circle positions
3. **Configure selection forces** (see Force Configuration below)
4. **Reheat simulation**: `d3ReheatSimulation()`

### Deselection Animation

When deselecting (click same node or click background):
1. Cancel any running animation
2. Unpin all nodes (`fx = undefined, fy = undefined`)
3. Restore default force configuration
4. Reheat simulation
5. `zoomToFit(1200, 50)` to animate back to the full graph view

---

## Force Configuration

Forces are managed through a `configureForces(activeIds?, selectedId?)` helper. When called with no arguments, it configures the default full-graph forces. When called with `activeIds` (the neighbor set) and `selectedId`, it configures isolated subgraph forces.

### Default Forces

| Force | Configuration |
|-------|---------------|
| **Collision** | `forceCollide`, radius = `tagRadius(count) + 4`, strength 0.5, 1 iteration |
| **Charge** | `forceManyBody`, strength -80, distanceMax 400 |
| **Link distance** | `max(20, 60 - weight * 8)` |
| **Link strength** | `min(0.6, 0.15 + weight * 0.08)` |
| **Center** | Built-in center force (from react-force-graph-2d) |
| **CenterX/Y** | null (not used) |
| **Radial** | null (not used) |

### Selection Forces

The `isActive(node)` predicate returns `true` only for nodes in `activeIds`.

| Force | Configuration |
|-------|---------------|
| **Collision** | Active: radius = `tagRadius(count) + 10`, strength 0.5. Inactive: radius 0 |
| **Charge** | Active: strength -200, distanceMax 200. Inactive: strength 0 |
| **Link distance** | `max(20, 60 - weight * 8)` |
| **Link strength** | Any link touching an inactive node: 0. Active co-occurrence: `min(0.6, 0.15 + weight * 0.08)` |
| **CenterX/Y** | `forceX(0)` and `forceY(0)` with strength 0.05 for active neighbors (0 for selected node and inactive) |
| **Radial** | `forceRadial` centered at (0,0) with size-dependent radius (same factor formula as animation targets). Strength 0.8 for active neighbors, 0 for selected/inactive |
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

## TagGraph Layout

Container: `w-full h-full relative overflow-hidden bg-surface-0 rounded-lg border border-surface-3/30`. A `ResizeObserver` tracks the container and passes `width`/`height` to the `ForceGraph2D` component.

Controls (bottom-right corner, `absolute bottom-2 right-2`):
1. **Reset button** (RotateCcw, 3×3): clears selection, fits all nodes, fires `onTagClick(null)`
2. **Fit button** (Maximize, 3×3): fits all nodes to viewport

No search input or stats — those live in the parent MemoryPage.

---

## Memory Tab Layout (MemoryPage)

The Memory tab (`MemoryPage.tsx`) provides a vault browser integrating the TagGraph with search and document viewing.

### Page Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  Search bar (BM25 full-text)                   Doc count  Tags │
├──────────┬──────────────────────────────┬──────────────────────┤
│ Left     │ Center                       │ Right               │
│ w-44     │ flex-1                       │ w-72                │
│          │                              │                     │
│ Tags tab │  ┌────────────────────────┐  │ Search results      │
│ or       │  │     TagGraph           │  │ (clickable cards)   │
│ Explorer │  │                        │  │                     │
│ tab      │  │                        │  │                     │
│          │  └────────────────────────┘  │                     │
└──────────┴──────────────────────────────┴──────────────────────┘
```

### Left Sidebar

Tab bar with two views:
- **Tags**: scrollable list of all vault tags with doc counts. Clicking a tag sets `activeTag`, which propagates to TagGraph via `selectedTag` prop and triggers a BM25 search for that tag name. Active tag scrolls into view automatically.
- **Explorer**: lazy-loaded tree view of vault directories. Top-level dirs auto-expand on mount (except `agents/`). Clicking a file opens the DocumentModal.

### Center

`<TagGraph selectedTag={activeTag} onTagClick={...} />` — when a graph node is clicked, it updates `activeTag`, switches to the Tags sidebar tab, and triggers a search.

### Right Panel

Search results panel (`w-72`, border-left). Shows results from BM25 full-text search via `GET /api/mc/memory/search`. Each result is a clickable card showing title and snippet. Panel content persists when tag is deselected (only cleared explicitly).

### Search Bar

Top-level search input with debounced BM25 query (300ms). Clearing the search also clears the active tag. Shows "searching..." indicator during API call.

### Document Modal

Full-screen overlay (`80% width/height`, `bg-black/60 backdrop-blur-sm`). Clicking outside (when not editing) closes it. Shows:

- **Header**: document title, file path, Edit/Close buttons
- **Frontmatter badges**: type badge (color-coded), status badge, tag pills
- **Summary**: italic text below badges
- **Content**: rendered markdown (via `marked`)

**Edit mode** (click Edit button):
- Content: full-height `<textarea>` replacing rendered markdown
- Tags: inline text input (comma-separated) replacing tag pills
- Summary: inline text input replacing summary text
- Save: calls `POST /api/mc/memory/save` with content + frontmatter merge, then re-reads the file
- Cancel: discards changes, exits edit mode

---

## Backend API Endpoints

All served from `extensions/mission-control/index.ts`.

### Graph Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mc/tag-graph` | GET | Tag co-occurrence graph. Returns `{nodes, edges}`. Nodes: one per unique tag with doc count. Edges: co-occurrence weight (number of shared docs). |
| `/api/mc/vault-graph` | GET | Document-level graph. Nodes: one per `.md` file (path, title, type, tags, folder). Edges: `[[wikilinks]]` resolved by path or basename. **Not currently rendered in UI.** |

### Memory Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mc/memory/stats` | GET | `{docCount, tagCount, types, topTags, lastRefresh}` |
| `/api/mc/memory/tags?limit=N` | GET | Top N tags sorted by doc count |
| `/api/mc/memory/search?q=&limit=` | GET | BM25 full-text search via `qmd search` CLI |
| `/api/mc/memory/browse?path=` | GET | List vault directory contents (entries with name, type, size, title, children) |
| `/api/mc/memory/read?path=` | GET | Read single vault file. Returns `{path, content, frontmatter, lineCount}` |
| `/api/mc/memory/save` | POST | Write vault file with frontmatter merge. Body: `{path, content, frontmatter}` |

---

## Key Implementation Details

### Why `fx/fy` Animation Instead of Forces

Forces alone couldn't reliably distribute neighbors in a full 360° circle — they'd trail the direction of travel and clump to one side. The solution: bypass the simulation entirely during transition by animating `fx/fy` (d3-force's "fixed position" pins) via `requestAnimationFrame`. Forces only take over after nodes are already at their target positions.

### Why Background Nodes are Frozen

Setting `fx = currentX, fy = currentY` on background nodes prevents forces from moving them during selection. Without this, the strong radial/charge forces on the active subgraph would push background nodes around, causing visual noise.

### Why Neighbors Stay Pinned

In the Memory tab context, the graph is one part of a busy layout. Keeping neighbors pinned after selection prevents subtle ongoing movement that competes for attention with the search results panel.

### Node Drag

Node dragging is enabled. On drag end, the simulation is reheated (`d3ReheatSimulation()`) so neighboring nodes settle into new positions.

---

## File Structure

```
web/src/components/pages/MemoryPage.tsx  — memory tab (3-column layout, tag sidebar, explorer, doc modal)
web/src/components/TagGraph.tsx          — reusable graph component (no hub, prop-driven selection)
web/src/api.ts                           — API client (TagGraphData, VaultGraphData, memory types)
backend/index.ts                         — all API endpoints (/api/mc/tag-graph, /vault-graph, /memory/*)
web/package.json                         — dependencies
```

MemoryPage is registered in the app's page routing (`Layout.tsx` PAGES record) and navigation sidebar (`Sidebar.tsx` — `Brain` icon).
