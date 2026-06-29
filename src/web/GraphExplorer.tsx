import { useMemo, useState } from "react";
import { ReactFlow, ReactFlowProvider, Background, Controls } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Value } from "../engine/value";
import { forwardGraph, backwardGraph } from "../engine/graphController";
import { layoutGraph } from "./layout";
import { ValueNode } from "./ValueNode";

const nodeTypes = { valueNode: ValueNode };

// Expressão de exemplo: f = tanh(a*b + c)
function buildExpression(a: number, b: number, c: number): Value {
  const va = new Value(a, { label: "a" });
  const vb = new Value(b, { label: "b" });
  const vc = new Value(c, { label: "c" });
  const out = va.mul(vb).add(vc).tanh();
  out.label = "f";
  return out;
}

type Phase = "forward" | "backward";

function Slider(props: {
  name: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label style={{ display: "flex", flexDirection: "column", fontSize: 12 }}>
      {props.name} = {props.value.toFixed(2)}
      <input
        type="range"
        min={-5}
        max={5}
        step={0.1}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function GraphExplorer() {
  const [a, setA] = useState(2);
  const [b, setB] = useState(-3);
  const [c, setC] = useState(1);
  const [phase, setPhase] = useState<Phase>("forward");

  const graph = useMemo(() => {
    const root = buildExpression(a, b, c);
    return phase === "backward" ? backwardGraph(root) : forwardGraph(root);
  }, [a, b, c, phase]);

  const { nodes, edges } = useMemo(() => layoutGraph(graph), [graph]);

  return (
    <div>
      <div style={{ display: "flex", gap: 16, alignItems: "flex-end", marginBottom: 12, flexWrap: "wrap" }}>
        <button onClick={() => setPhase("forward")}>Forward</button>
        <button onClick={() => setPhase("backward")}>Backward</button>
        <Slider name="a" value={a} onChange={setA} />
        <Slider name="b" value={b} onChange={setB} />
        <Slider name="c" value={c} onChange={setC} />
      </div>
      <div style={{ height: 480, border: "1px solid #1e293b", borderRadius: 8 }}>
        <ReactFlowProvider>
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView>
            <Background />
            <Controls />
          </ReactFlow>
        </ReactFlowProvider>
      </div>
    </div>
  );
}
