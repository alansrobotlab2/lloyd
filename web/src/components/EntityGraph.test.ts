import { describe, expect, it } from "vitest";

import lightsDecl from "../three-force-graph-lights.d.ts?raw";
import graphSource from "./EntityGraph.tsx?raw";

// Why a source-text test. This runner has no DOM (`environment: "node"`, no jsdom —
// adding one means `web/package.json`, a build input the automod gate denies), so
// `particleArrowFn` cannot be mounted and its return value inspected. What is pinned
// here is the two-directional shape that keeps arrow cones confined to the highlighted
// links, which is the behaviour #3816 had to keep while making the accessor's return
// type satisfy `LinkAccessor<…, Object3D>`: a link outside `highlightLinks` still gets
// NO cone, and a link inside it still gets one.
//
// Why "no cone" is still true when the accessor now returns an object for the excluded
// link: three-forcegraph digests `visibleLinks.filter(particlesAccessor)`
// (three-forcegraph.mjs:1394) before it calls this accessor at all — the call sits
// inside `onUpdateObj`, at :1341 — and `linkParticlesFn` below returns 0 particles for
// exactly that excluded set, so the empty `Object3D` is never asked for and never
// reaches the scene. Both halves are asserted below, because either one changing alone
// is what would put a stray cone on every dimmed edge.
describe("particleArrowFn keeps cones confined to highlighted links (#3816)", () => {
  // Code, not prose: every identifier below also appears in the comment above the
  // closure that explains it, so a component that described the rule and implemented
  // nothing would pass a raw `toContain`. Line and JSX comments are stripped first.
  const code = graphSource
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, "")
    .split("\n").filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l)).join("\n");

  it("the comment stripper left real code behind, so it cannot pass by erasure", () => {
    expect(code).toContain("const particleArrowFn = useCallback(");
    expect(code.length).toBeGreaterThan(graphSource.length / 2);
    // And it did remove prose: this phrase is in the comment above the closure, so it
    // is in the file and must be gone from `code`. Without both directions the strip
    // could remove nothing at all and every assertion below would be back to matching
    // on comments.
    const STRIPPED = "One arrow cone per link";
    expect(graphSource).toContain(STRIPPED);
    expect(code).not.toContain(STRIPPED);
  });

  const cut = (from: string, to: string, atLeast: number) => {
    const a = code.indexOf(from);
    const b = code.indexOf(to, a + 1);
    // An `indexOf` that misses returns -1 and slices an EMPTY string, against which
    // every `toMatch` below would still have to fail for the wrong reason: so each cut
    // is shown to have found both ends and to have produced real code.
    expect(a, `EntityGraph.tsx no longer contains ${from}`).toBeGreaterThan(-1);
    expect(b, `EntityGraph.tsx has no ${to} after ${from}`).toBeGreaterThan(a);
    const block = code.slice(a, b);
    expect(block.length, `${from} block looks gutted`).toBeGreaterThan(atLeast);
    return block;
  };

  const arrowBlock = cut(
    "const particleArrowFn = useCallback(", "const nodeTooltip = useCallback", 400);

  it("gives a link OUTSIDE the highlight set an empty Object3D, never undefined", () => {
    expect(arrowBlock).toMatch(/if \(!highlightLinks\.has\(/);
    expect(arrowBlock).toMatch(/return new THREE\.Object3D\(\);/);
    // The previous shape. `undefined` is what TS2322 rejected at the prop, and a
    // "fix" that restored it would send this node red rather than the build red.
    expect(arrowBlock).not.toMatch(/return undefined/);
  });

  it("still builds a real cone for a link INSIDE the set", () => {
    expect(arrowBlock).toMatch(/new THREE\.ConeGeometry\(/);
    expect(arrowBlock).toMatch(/return new THREE\.Mesh\(geo, mat\);/);
  });

  it("clears the type error with types, not with a suppression", () => {
    // The gate's build would go green on `@ts-ignore` just as happily as on the real
    // fix, and leave the defect invisible — which is what #3816 clause 5 forbids.
    for (const banned of ["@ts-ignore", "@ts-expect-error", "as any"]) {
      expect(arrowBlock, `particleArrowFn uses ${banned}`).not.toContain(banned);
    }
  });

  it("wires the accessor to the particle-object prop it was written for", () => {
    expect(code).toMatch(/linkDirectionalParticleThreeObject=\{particleArrowFn\}/);
  });

  it("gives 0 particles to the same set, so the empty object is never drawn", () => {
    const particlesBlock = cut(
      "const linkParticlesFn = useCallback(", "const particleArrowFn = useCallback(", 200);
    expect(particlesBlock).toMatch(/highlightLinks\.has\(/);
    expect(particlesBlock).toMatch(/\? 5 : 0;/);
  });
});

// The third error #3816 had to clear, which nobody had seen because TypeScript reports
// one failed JSX attributes check per element: `EntityGraph.tsx` carried the particle
// error, so the missing `lights` prop was never reported until the first one was fixed.
describe("the scene lights stay overridden and typed (#3816)", () => {
  const code = graphSource
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, "")
    .split("\n").filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l)).join("\n");

  it("keeps overriding the lights the library would otherwise default", () => {
    // The cheapest way to clear the JSX error was to delete this prop, and that would
    // silently fall back to the library's default lights — the flattening the comment
    // above the `lights` memo says these exist to avoid. So the prop must still be here.
    expect(code).toMatch(/lights=\{lights\}/);
    expect(code).toMatch(/new THREE\.AmbientLight\(/);
    expect(code).toMatch(/new THREE\.DirectionalLight\(/);
  });

  it("types the prop in our own source, so deleting that file is caught here", () => {
    // Read the augmentation rather than the component: an assertion that only looked at
    // EntityGraph.tsx stayed green when the .d.ts was deleted from a scratch copy, which
    // is the whole file going away without a node noticing.
    expect(lightsDecl).toMatch(/declare module "react-force-graph-3d"/);
    expect(lightsDecl).toMatch(/interface ForceGraphProps<NodeType = \{\}, LinkType = \{\}>/);
    expect(lightsDecl).toMatch(/lights\?: Light\[\];/);
    // A declaration is only worth what its value is checked at: `any[]` would silence
    // tsc the way a cast would, and leave the prop's contents unverified.
    expect(lightsDecl).not.toMatch(/lights\?: any/);
    for (const banned of ["@ts-ignore", "@ts-expect-error"]) {
      expect(lightsDecl, `the augmentation uses ${banned}`).not.toContain(banned);
    }
  });
});
