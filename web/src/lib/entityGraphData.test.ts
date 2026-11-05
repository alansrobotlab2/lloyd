import { describe, expect, it } from "vitest";
import { collectPositions, placeNodes } from "./entityGraphData";

const node = (id: string) => ({ id }) as {
  id: string; x?: number; y?: number; z?: number; fx?: number; fy?: number; fz?: number;
};
const link = (source: string, target: string) => ({ source, target });
const half = () => 0.5; // jitter of zero

describe("placeNodes", () => {
  it("pins everything and skips simulating when every node has a position", () => {
    const nodes = [node("a"), node("b")];
    const mode = placeNodes(nodes, [link("a", "b")], { a: [1, 2, 3], b: [4, 5, 6] }, half);
    expect(mode).toBe("settled");
    expect(nodes[0]).toMatchObject({ x: 1, y: 2, z: 3, fx: 1, fy: 2, fz: 3 });
  });

  it("keeps the known majority pinned and seeds a newcomer at its neighbours' centre", () => {
    const nodes = [node("a"), node("b"), node("c"), node("new")];
    const mode = placeNodes(
      nodes,
      [link("new", "a"), link("b", "new"), link("a", "c")],
      { a: [0, 0, 0], b: [10, 20, 30], c: [5, 5, 5] },
      half,
    );
    expect(mode).toBe("partial");
    const n = nodes[3];
    expect([n.x, n.y, n.z]).toEqual([5, 10, 15]);
    expect(n.fx).toBeUndefined();           // the newcomer is free to settle
    expect(nodes[0].fx).toBe(0);            // the known ones are not
  });

  it("fills a chain of newcomers inward from the placed end", () => {
    const nodes = [node("a"), node("b"), node("c"), node("n1"), node("n2")];
    placeNodes(
      nodes,
      [link("a", "n1"), link("n1", "n2")],
      { a: [3, 3, 3], b: [0, 0, 0], c: [1, 1, 1] },
      half,
    );
    expect(nodes[4]).toMatchObject({ x: 3, y: 3, z: 3 });
  });

  it("does not pin a minority, so a changed graph settles as a whole", () => {
    const nodes = [node("a"), node("b"), node("c")];
    const mode = placeNodes(nodes, [link("a", "b")], { a: [1, 1, 1] }, half);
    expect(mode).toBe("fresh");
    expect(nodes[0]).toMatchObject({ x: 1, fx: undefined });
  });

  it("clears a stale pin from a node the stored layout no longer covers", () => {
    const stale = { ...node("a"), fx: 9, fy: 9, fz: 9, x: 9, y: 9, z: 9 };
    const nodes = [stale, node("b"), node("c")];
    placeNodes(nodes, [], { b: [0, 0, 0], c: [1, 1, 1] }, half);
    expect(stale.fx).toBeUndefined();
    expect(typeof stale.x).toBe("number");  // placed inside the shape, not at 9
  });

  it("reports fresh with nothing known, and settled for an empty graph", () => {
    expect(placeNodes([node("a")], [], {}, half)).toBe("fresh");
    expect(placeNodes([], [], {}, half)).toBe("settled");
  });
});

describe("collectPositions", () => {
  it("rounds to 0.1 and skips nodes the simulation never placed", () => {
    const pos = collectPositions([{ id: "a", x: 1.26, y: -2.04, z: 3 }, { id: "b" }]);
    expect(pos).toEqual({ a: [1.3, -2, 3] });
  });
});
