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

/**
 * Retorna um array de snapshots Graph mostrando o backprop nó a nó:
 *   frames[0]   = estado forward (todos os grads = 0)
 *   frames[1]   = root grad revelado
 *   frames[i]   = i-ésimo nó na ordem reverso-topológica com grad revelado
 *   frames[N]   = grafo completamente retropropagado (== backwardGraph)
 *
 * A ordem de revelação é reverso-topológica, garantindo que o grad de
 * um nó seja revelado somente após o nó de quem ele recebe o fluxo.
 */
export function backwardFrames(root: Value): Graph[] {
  // Build topological order (leaves first, root last) — same logic as Value.backward()
  const topo: Value[] = [];
  const visited = new Set<Value>();
  const buildTopo = (v: Value) => {
    if (visited.has(v)) return;
    visited.add(v);
    for (const child of v.prev) buildTopo(child);
    topo.push(v);
  };
  buildTopo(root);

  // reverseOrder: root first, leaves last (the propagation order)
  const reverseOrder = [...topo].reverse();

  // Run full backward to get every node's true gradient
  root.backward();
  const trueGrads = new Map<Value, number>();
  for (const v of topo) trueGrads.set(v, v.grad);

  // Reset grads to 0 for the initial "forward" frame
  for (const v of topo) v.grad = 0;

  const frames: Graph[] = [];

  // Frame 0: all grads 0
  frames.push(trace(root));

  // Reveal one node's true grad per frame in reverse-topological order
  for (const v of reverseOrder) {
    v.grad = trueGrads.get(v) ?? 0;
    frames.push(trace(root));
  }

  return frames;
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
