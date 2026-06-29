import { describe, it, expect } from "vitest";
import { Value } from "./value";
import { forwardGraph, backwardGraph, hasNonFinite } from "./graphController";

describe("forwardGraph / backwardGraph", () => {
  const build = () => new Value(3, { label: "a" }).mul(new Value(2, { label: "b" }));

  it("forwardGraph deixa todos os grads zerados", () => {
    const g = forwardGraph(build());
    expect(g.nodes.every((n) => n.grad === 0)).toBe(true);
  });

  it("backwardGraph preenche os grads", () => {
    const g = backwardGraph(build());
    const root = g.nodes.find((n) => n.op === "*")!;
    expect(root.grad).toBe(1);
    // grad de cada folha != 0
    expect(g.nodes.filter((n) => n.op === "").every((n) => n.grad !== 0)).toBe(true);
  });
});

describe("hasNonFinite", () => {
  it("detecta Infinity (divisão por zero)", () => {
    const g = backwardGraph(new Value(1).div(0));
    expect(hasNonFinite(g)).toBe(true);
  });

  it("é falso para grafo normal", () => {
    const g = backwardGraph(new Value(1).add(2));
    expect(hasNonFinite(g)).toBe(false);
  });
});
