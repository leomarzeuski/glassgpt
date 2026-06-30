import { useState } from "react";
import { makeDemo, trainStep, type DemoState } from "../engine/trainingDemo";

const W = 520;
const H = 76;
const PAD = 8;

function Sparkline({ data }: { data: number[] }) {
  if (data.length < 2) {
    return (
      <div
        style={{
          height: H,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--muted)",
          fontSize: 13,
        }}
      >
        rode alguns passos pra ver a loss cair
      </div>
    );
  }
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const pts = data
    .map((v, i) => {
      const x = PAD + (i / (data.length - 1)) * (W - 2 * PAD);
      const y = PAD + (1 - (v - min) / range) * (H - 2 * PAD);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
      <polyline points={pts} fill="none" stroke="var(--data)" strokeWidth={2} strokeLinejoin="round" />
    </svg>
  );
}

export function TrainingDemo() {
  const [state] = useState<DemoState>(() => makeDemo(42));
  const [history, setHistory] = useState<number[]>([]);
  const [steps, setSteps] = useState(0);

  const step = (n: number) => {
    setHistory((h) => {
      const next = [...h];
      for (let i = 0; i < n; i++) next.push(trainStep(state));
      return next.slice(-400);
    });
    setSteps((s) => s + n);
  };

  const reset = () => {
    setHistory([]);
    setSteps(0);
  };

  const current = history[history.length - 1];
  const learned = current !== undefined && current < 0.2;

  return (
    <div
      style={{
        background: "var(--panel)",
        border: "1px solid var(--hairline)",
        borderRadius: 12,
        padding: 18,
      }}
    >
      <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap", marginBottom: 14 }}>
        <button className="primary" onClick={() => step(50)}>
          ▶ 50 passos
        </button>
        <button onClick={() => step(1)}>1 passo</button>
        <button onClick={reset}>Resetar</button>

        <div style={{ flex: 1 }} />

        <span style={{ fontSize: 13, color: "var(--muted)" }}>
          loss{" "}
          <span data-testid="loss" className="mono" style={{ fontSize: 18, color: learned ? "var(--data)" : "var(--grad)" }}>
            {current === undefined ? "—" : current.toFixed(4)}
          </span>
        </span>
        <span style={{ fontSize: 13, color: "var(--muted)" }}>
          passos <span className="mono" style={{ color: "var(--text)" }}>{steps}</span>
        </span>
        {learned && <span style={{ fontSize: 13, color: "var(--data)", fontWeight: 600 }}>✓ aprendeu o XOR</span>}
      </div>

      <Sparkline data={history} />
    </div>
  );
}
