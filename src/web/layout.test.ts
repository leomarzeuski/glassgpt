import { describe, it, expect } from "vitest";
import { Value } from "../engine/value";
import { trace } from "../engine/trace";
import { layoutGraph } from "./layout";

describe("layoutGraph", () => {
  it("posiciona todos os nós com coordenadas numéricas", () => {
    const out = new Value(2, { label: "a" }).mul(new Value(3, { label: "b" }));
    const { nodes, edges } = layoutGraph(trace(out));

    expect(nodes).toHaveLength(3);
    expect(edges).toHaveLength(2);
    for (const n of nodes) {
      expect(Number.isFinite(n.position.x)).toBe(true);
      expect(Number.isFinite(n.position.y)).toBe(true);
      expect(n.type).toBe("valueNode");
    }
  });
});
