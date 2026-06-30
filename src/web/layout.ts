import dagre from "dagre";
import type { Node, Edge } from "@xyflow/react";
import type { Graph } from "../engine/trace";
import type { ValueNodeData } from "./ValueNode";

const NODE_W = 150;
const NODE_H = 72;

export function layoutGraph(graph: Graph): { nodes: Node<ValueNodeData>[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "LR", nodesep: 30, ranksep: 80 });

  for (const n of graph.nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of graph.edges) g.setEdge(e.source, e.target);

  dagre.layout(g);

  const nodes: Node<ValueNodeData>[] = graph.nodes.map((n) => {
    const pos = g.node(n.id);
    const data: ValueNodeData = { label: n.label, value: n.data, grad: n.grad, op: n.op };
    return {
      id: n.id,
      type: "valueNode",
      position: { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 },
      data,
    };
  });

  const edges: Edge[] = graph.edges.map((e, i) => ({
    id: `e${i}`,
    source: e.source,
    target: e.target,
    animated: true,
  }));

  return { nodes, edges };
}
