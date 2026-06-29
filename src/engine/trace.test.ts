import { describe, it, expect } from "vitest";
import { Value } from "./value";
import { trace } from "./trace";

describe("trace", () => {
  it("captura todos os nós e arestas de uma expressão", () => {
    const a = new Value(2, { label: "a" });
    const b = new Value(3, { label: "b" });
    const out = a.mul(b); // nós: a, b, out -> 3 nós, 2 arestas
    out.label = "out";

    const g = trace(out);
    expect(g.nodes).toHaveLength(3);
    expect(g.edges).toHaveLength(2);

    const outNode = g.nodes.find((n) => n.label === "out");
    expect(outNode?.data).toBe(6);
    expect(outNode?.op).toBe("*");
  });

  it("não duplica um nó reutilizado", () => {
    const a = new Value(2, { label: "a" });
    const out = a.add(a); // a aparece 2x como operando, mas é 1 nó só
    const g = trace(out);
    expect(g.nodes).toHaveLength(2); // a, out
    expect(g.edges).toHaveLength(2); // a->out duas vezes
  });
});
