import { Value } from "./value";

export interface GraphNode {
  id: string;
  label: string;
  data: number;
  grad: number;
  op: string;
}

export interface GraphEdge {
  source: string; // operando (filho)
  target: string; // resultado (pai)
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export function trace(root: Value): Graph {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const ids = new Map<Value, string>();
  let counter = 0;

  const idOf = (v: Value): string => {
    let id = ids.get(v);
    if (id === undefined) {
      id = `n${counter++}`;
      ids.set(v, id);
    }
    return id;
  };

  const visited = new Set<Value>();
  const visit = (v: Value) => {
    if (visited.has(v)) return;
    visited.add(v);
    nodes.push({ id: idOf(v), label: v.label, data: v.data, grad: v.grad, op: v.op });
    for (const child of v.prev) {
      edges.push({ source: idOf(child), target: idOf(v) });
      visit(child);
    }
  };

  visit(root);
  return { nodes, edges };
}
