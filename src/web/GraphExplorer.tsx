import { useCallback, useEffect, useMemo, useState } from "react";
import { ReactFlow, ReactFlowProvider, Background, Controls } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Value } from "../engine/value";
import {
  forwardGraph,
  backwardFrames,
  hasNonFinite,
} from "../engine/graphController";
import type { Graph } from "../engine/trace";
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

const FRAME_INTERVAL_MS = 300;

export function GraphExplorer() {
  // Default values chosen so tanh(a*b+c) = tanh(0.5) ≈ 0.46 — clearly non-trivial grads
  const [a, setA] = useState(1);
  const [b, setB] = useState(0.5);
  const [c, setC] = useState(0);

  // Animation state: list of graph snapshots and current index
  const [framesList, setFramesList] = useState<Graph[]>(() => [
    forwardGraph(buildExpression(1, 0.5, 0)),
  ]);
  const [frameIdx, setFrameIdx] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // When sliders change, reset to forward graph of new values
  useEffect(() => {
    setFramesList([forwardGraph(buildExpression(a, b, c))]);
    setFrameIdx(0);
    setIsAnimating(false);
  }, [a, b, c]);

  // Animation timer: advance one frame every FRAME_INTERVAL_MS
  useEffect(() => {
    if (!isAnimating) return;
    if (frameIdx >= framesList.length - 1) {
      setIsAnimating(false);
      return;
    }
    const timer = setTimeout(() => {
      setFrameIdx((i) => i + 1);
    }, FRAME_INTERVAL_MS);
    return () => clearTimeout(timer);
  }, [isAnimating, frameIdx, framesList]);

  const handleForward = useCallback(() => {
    setIsAnimating(false);
    setFramesList([forwardGraph(buildExpression(a, b, c))]);
    setFrameIdx(0);
  }, [a, b, c]);

  const handleBackward = useCallback(() => {
    const frames = backwardFrames(buildExpression(a, b, c));
    setFramesList(frames);
    setFrameIdx(0);
    setIsAnimating(true);
  }, [a, b, c]);

  const displayGraph = framesList[frameIdx] ?? framesList[0];

  const { nodes, edges } = useMemo(() => layoutGraph(displayGraph), [displayGraph]);

  const nonFiniteDetected = hasNonFinite(displayGraph);

  return (
    <div>
      <div
        style={{
          display: "flex",
          gap: 16,
          alignItems: "flex-end",
          marginBottom: 12,
          flexWrap: "wrap",
        }}
      >
        <button onClick={handleForward}>Forward</button>
        <button onClick={handleBackward}>Backward</button>
        <Slider name="a" value={a} onChange={setA} />
        <Slider name="b" value={b} onChange={setB} />
        <Slider name="c" value={c} onChange={setC} />
      </div>

      {nonFiniteDetected && (
        <div
          role="alert"
          style={{
            marginBottom: 8,
            padding: "6px 12px",
            background: "#450a0a",
            border: "1px solid #e11d48",
            borderRadius: 6,
            color: "#fca5a5",
            fontSize: 13,
          }}
        >
          ⚠️ a expressão divergiu (NaN/Inf)
        </div>
      )}

      {/* Hidden probe used by tests to verify animation state */}
      <span
        data-testid="has-nonzero-grads"
        aria-hidden="true"
        hidden
        data-value={String(displayGraph.nodes.some((n) => n.grad !== 0))}
      />

      <div
        style={{ height: 480, border: "1px solid #1e293b", borderRadius: 8 }}
      >
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
