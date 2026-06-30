import { describe, it, expect } from "vitest";
import { Value } from "./value";
import { forwardGraph, backwardGraph, hasNonFinite, backwardFrames } from "./graphController";
import { trace } from "./trace";

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

describe("backwardFrames", () => {
  // Expression: f = tanh(a*b + c) with a=1, b=0.5, c=0
  // 6 nodes: va, vb, mul, vc, add, tanh  → 7 frames total
  const buildExpr = () => {
    const va = new Value(1, { label: "a" });
    const vb = new Value(0.5, { label: "b" });
    const vc = new Value(0, { label: "c" });
    const out = va.mul(vb).add(vc).tanh();
    out.label = "f";
    return out;
  };

  it("frame count == number of nodes + 1", () => {
    const root = buildExpr();
    const nodeCount = trace(root).nodes.length;
    const frames = backwardFrames(root);
    expect(frames.length).toBe(nodeCount + 1);
  });

  it("frame 0 has all grads 0 (forward state)", () => {
    const root = buildExpr();
    const frames = backwardFrames(root);
    expect(frames[0].nodes.every((n) => n.grad === 0)).toBe(true);
  });

  it("last frame matches full backward result (all true grads)", () => {
    const root1 = buildExpr();
    const frames = backwardFrames(root1);
    const lastFrame = frames[frames.length - 1];

    const root2 = buildExpr();
    const bwGraph = backwardGraph(root2);

    // Both trace same structure in same DFS order → same positional grads
    expect(lastFrame.nodes.length).toBe(bwGraph.nodes.length);
    for (let i = 0; i < lastFrame.nodes.length; i++) {
      expect(lastFrame.nodes[i].grad).toBeCloseTo(bwGraph.nodes[i].grad, 10);
    }
  });

  it("each node's grad is revealed no earlier than the node it flows from (reverse-topo order)", () => {
    const root = buildExpr();
    const frames = backwardFrames(root);

    // Find the first frame (index) where each node's grad becomes non-zero
    const revealFrame = (nodeId: string): number => {
      for (let fi = 1; fi < frames.length; fi++) {
        const node = frames[fi].nodes.find((n) => n.id === nodeId);
        if (node && node.grad !== 0) return fi;
      }
      return frames.length; // never revealed (stays 0)
    };

    // edges: source=child, target=parent; grad flows parent→child
    // so parent (target) must be revealed in same or earlier frame than child (source)
    for (const edge of frames[0].edges) {
      expect(revealFrame(edge.target)).toBeLessThanOrEqual(revealFrame(edge.source));
    }
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
