import { Value } from "./value";
import { trace, type Graph } from "./trace";

/** Garante grads zerados (estado "antes do backward") e devolve o trace. */
export function forwardGraph(root: Value): Graph {
  zeroGrads(root);
  return trace(root);
}

/** Roda o backward e devolve o trace já com os gradientes. */
export function backwardGraph(root: Value): Graph {
  root.backward();
  return trace(root);
}

export function hasNonFinite(graph: Graph): boolean {
  return graph.nodes.some((n) => !Number.isFinite(n.data) || !Number.isFinite(n.grad));
}

function zeroGrads(root: Value): void {
  const visited = new Set<Value>();
  const visit = (v: Value) => {
    if (visited.has(v)) return;
    visited.add(v);
    v.grad = 0;
    for (const child of v.prev) visit(child);
  };
  visit(root);
}
