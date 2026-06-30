import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Value } from "../engine/value";
import { forwardGraph, backwardFrames, hasNonFinite } from "../engine/graphController";
import type { Graph } from "../engine/trace";
import { layoutGraph } from "./layout";
import { ValueNode, type ValueNodeData } from "./ValueNode";

const nodeTypes = { valueNode: ValueNode };
const FRAME_INTERVAL_MS = 320;
const INIT = { a: 1, b: 0.5, c: 0 };

// Expressão de exemplo: f = tanh(a*b + c)
function buildExpression(a: number, b: number, c: number): Value {
  const va = new Value(a, { label: "a" });
  const vb = new Value(b, { label: "b" });
  const vc = new Value(c, { label: "c" });
  const out = va.mul(vb).add(vc).tanh();
  out.label = "f";
  return out;
}

/** id -> {value, grad} for merging a frame's numbers onto stable positioned nodes. */
function numbersById(g: Graph): Map<string, { value: number; grad: number }> {
  return new Map(g.nodes.map((n) => [n.id, { value: n.data, grad: n.grad }]));
}

function Slider(props: { name: string; value: number; onChange: (v: number) => void }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2, fontSize: 12, color: "var(--muted)" }}>
      <span>
        {props.name} ={" "}
        <span className="mono" style={{ color: "var(--text)" }}>
          {props.value.toFixed(2)}
        </span>
      </span>
      <input
        type="range"
        min={-3}
        max={3}
        step={0.1}
        value={props.value}
        aria-label={`valor de ${props.name}`}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

function LegendChip(props: { color: string; label: string; desc: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--muted)" }}>
      <span style={{ width: 9, height: 9, borderRadius: 999, background: props.color, flexShrink: 0 }} />
      <span style={{ color: "var(--text)", fontWeight: 500 }}>{props.label}</span> {props.desc}
    </span>
  );
}

export function GraphExplorer() {
  const [a, setA] = useState(INIT.a);
  const [b, setB] = useState(INIT.b);
  const [c, setC] = useState(INIT.c);

  const [phase, setPhase] = useState<"forward" | "backward">("forward");
  const framesRef = useRef<Graph[]>([]);
  const [frameIdx, setFrameIdx] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // The graph currently shown (forward state, or the current backward frame).
  const [displayGraph, setDisplayGraph] = useState<Graph>(() =>
    forwardGraph(buildExpression(INIT.a, INIT.b, INIT.c)),
  );

  // Positions are computed once (structure is constant); only node DATA changes.
  // We update data in place via setRfNodes so React Flow never unmounts nodes,
  // and wire onNodesChange so it keeps node measurements.
  const initialLayout = useMemo(
    () => layoutGraph(forwardGraph(buildExpression(INIT.a, INIT.b, INIT.c))),
    [],
  );
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState<Node<ValueNodeData>>(initialLayout.nodes);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState<Edge>(initialLayout.edges);

  // Sync the displayed graph onto the React Flow nodes/edges (data only).
  useEffect(() => {
    const m = numbersById(displayGraph);
    setRfNodes((prev) =>
      prev.map((n) => {
        const v = m.get(n.id);
        return v ? { ...n, data: { ...n.data, value: v.value, grad: v.grad } } : n;
      }),
    );
    const backward = phase === "backward";
    setRfEdges((prev) =>
      prev.map((e) => ({
        ...e,
        animated: backward,
        style: {
          stroke: backward ? "var(--grad)" : "var(--hairline)",
          strokeWidth: backward ? 2 : 1,
        },
      })),
    );
  }, [displayGraph, phase, setRfNodes, setRfEdges]);

  // Reset to the forward graph whenever a slider changes. Depends ONLY on the
  // slider values so it never re-fires mid-animation.
  useEffect(() => {
    framesRef.current = [];
    setIsAnimating(false);
    setFrameIdx(0);
    setPhase("forward");
    setDisplayGraph(forwardGraph(buildExpression(a, b, c)));
  }, [a, b, c]);

  // Animation timer: advance one frame per tick while animating.
  useEffect(() => {
    if (!isAnimating) return;
    if (frameIdx >= framesRef.current.length - 1) {
      setIsAnimating(false);
      return;
    }
    const t = setTimeout(() => setFrameIdx((i) => i + 1), FRAME_INTERVAL_MS);
    return () => clearTimeout(t);
  }, [isAnimating, frameIdx]);

  // Paint the current backward frame as frameIdx advances.
  useEffect(() => {
    if (phase !== "backward") return;
    const frame = framesRef.current[frameIdx];
    if (frame) setDisplayGraph(frame);
  }, [phase, frameIdx]);

  const handleReset = useCallback(() => {
    framesRef.current = [];
    setIsAnimating(false);
    setFrameIdx(0);
    setPhase("forward");
    setDisplayGraph(forwardGraph(buildExpression(a, b, c)));
  }, [a, b, c]);

  const handleBackward = useCallback(() => {
    const frames = backwardFrames(buildExpression(a, b, c));
    framesRef.current = frames;
    setPhase("backward");
    setFrameIdx(0);
    setDisplayGraph(frames[0]);
    setIsAnimating(true);
  }, [a, b, c]);

  const totalSteps =
    framesRef.current.length > 0 ? framesRef.current.length - 1 : displayGraph.nodes.length;
  const revealed = phase === "backward" ? Math.min(frameIdx, totalSteps) : 0;
  const result = Math.tanh(a * b + c);
  const diverged = hasNonFinite(displayGraph);

  return (
    <div>
      {/* Expression + how-to */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "baseline", marginBottom: 10 }}>
        <span className="mono" style={{ fontSize: 16, color: "var(--text)" }}>
          f = tanh(<span style={{ color: "var(--data)" }}>a</span>·<span style={{ color: "var(--data)" }}>b</span> +{" "}
          <span style={{ color: "var(--data)" }}>c</span>) ={" "}
          <span style={{ color: "var(--data)" }}>{result.toFixed(4)}</span>
        </span>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 16, alignItems: "flex-end", marginBottom: 10, flexWrap: "wrap" }}>
        <button className="primary" onClick={handleBackward}>
          ▶ Animar backward
        </button>
        <button onClick={handleReset}>Resetar</button>
        <span data-testid="grad-status" className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
          gradientes calculados:{" "}
          <span style={{ color: revealed > 0 ? "var(--grad)" : "var(--muted)" }}>{revealed}</span>/{totalSteps}
        </span>
        <div style={{ flex: 1 }} />
        <Slider name="a" value={a} onChange={setA} />
        <Slider name="b" value={b} onChange={setB} />
        <Slider name="c" value={c} onChange={setC} />
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginBottom: 10 }}>
        <LegendChip color="var(--data)" label="valor" desc="— o número calculado em cada passo" />
        <LegendChip color="var(--grad)" label="grad" desc="— o quanto aquele número influencia o resultado f" />
      </div>

      {diverged && (
        <div
          role="alert"
          style={{
            marginBottom: 8,
            padding: "6px 12px",
            background: "rgba(255,92,122,0.12)",
            border: "1px solid var(--danger)",
            borderRadius: 8,
            color: "var(--danger)",
            fontSize: 13,
          }}
        >
          ⚠️ a expressão divergiu (NaN/∞) — puxe os valores de volta pra perto de zero.
        </div>
      )}

      <div style={{ height: 460, border: "1px solid var(--hairline)", borderRadius: 12, background: "var(--bg-2)" }}>
        <ReactFlowProvider>
          <ReactFlow
            nodes={rfNodes}
            edges={rfEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            proOptions={{ hideAttribution: true }}
            nodesDraggable={false}
            nodesConnectable={false}
          >
            <Background color="var(--hairline)" gap={20} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </ReactFlowProvider>
      </div>
    </div>
  );
}
