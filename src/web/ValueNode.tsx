import { Handle, Position } from "@xyflow/react";

export interface ValueNodeData extends Record<string, unknown> {
  label: string;
  value: number;
  grad: number;
  op: string;
}

const OP_WORDS: Record<string, string> = {
  "*": "× multiplica",
  "+": "+ soma",
  tanh: "tanh espreme",
  exp: "exp",
  ReLU: "ReLU",
};

/** Plain-language caption for a node, based on its role in the graph. */
function caption(label: string, op: string): { eyebrow: string; isResult: boolean } {
  if (label === "f") return { eyebrow: "resultado · f", isResult: true };
  if (op === "") return { eyebrow: `entrada · ${label || "?"}`, isResult: false };
  if (op.startsWith("**")) return { eyebrow: `^ potência ${op.slice(2)}`, isResult: false };
  return { eyebrow: OP_WORDS[op] ?? op, isResult: false };
}

function fmt(x: number): string {
  if (Number.isNaN(x)) return "NaN";
  if (!Number.isFinite(x)) return x > 0 ? "+∞" : "−∞";
  return x.toFixed(4);
}

export function ValueNode({ data }: { data: ValueNodeData }) {
  const nonFinite = !Number.isFinite(data.value) || !Number.isFinite(data.grad);
  const gradLit = data.grad !== 0 && Number.isFinite(data.grad);
  const { eyebrow, isResult } = caption(data.label, data.op);

  const borderColor = nonFinite
    ? "var(--danger)"
    : isResult
      ? "var(--data)"
      : "var(--hairline)";

  return (
    <div
      data-testid="value-node"
      data-nonfinite={nonFinite}
      style={{
        minWidth: 138,
        borderRadius: 10,
        padding: "8px 12px",
        background: "var(--panel)",
        border: `1px solid ${borderColor}`,
        boxShadow: gradLit
          ? "0 0 0 1px var(--grad), 0 0 18px -5px var(--grad)"
          : "none",
        transition: "box-shadow .25s ease, border-color .25s ease",
        fontFamily: "var(--font-ui)",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: "var(--muted)" }} />

      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: isResult ? "var(--data)" : "var(--muted)",
          fontWeight: 600,
        }}
      >
        {eyebrow}
      </div>

      <div className="mono" style={{ fontSize: 13, marginTop: 5 }}>
        <span style={{ color: "var(--muted)" }}>valor </span>
        <span style={{ color: "var(--data)" }}>{fmt(data.value)}</span>
      </div>
      <div className="mono" style={{ fontSize: 13, marginTop: 1 }}>
        <span style={{ color: "var(--muted)" }}>grad </span>
        <span style={{ color: gradLit ? "var(--grad)" : "var(--grad-dim)" }}>
          {fmt(data.grad)}
        </span>
      </div>

      <Handle type="source" position={Position.Right} style={{ background: "var(--muted)" }} />
    </div>
  );
}
