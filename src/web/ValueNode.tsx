import { Handle, Position } from "@xyflow/react";

export interface ValueNodeData {
  label: string;
  value: number;
  grad: number;
  op: string;
}

export function ValueNode({ data }: { data: ValueNodeData }) {
  const nonFinite = !Number.isFinite(data.value) || !Number.isFinite(data.grad);
  return (
    <div
      data-testid="value-node"
      data-nonfinite={nonFinite}
      style={{
        border: `2px solid ${nonFinite ? "#e11d48" : "#334155"}`,
        borderRadius: 8,
        padding: "6px 10px",
        background: "#0f172a",
        color: "#e2e8f0",
        fontFamily: "monospace",
        fontSize: 12,
        minWidth: 120,
      }}
    >
      <Handle type="target" position={Position.Left} />
      <div style={{ fontSize: 11, opacity: 0.7 }}>{data.label || data.op || "•"}</div>
      <div>data {data.value.toFixed(4)}</div>
      <div style={{ color: "#38bdf8" }}>grad {data.grad.toFixed(4)}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
