import { useState } from "react";
import { makeDemo, trainStep, type DemoState } from "../engine/trainingDemo";

export function TrainingDemo() {
  const [state] = useState<DemoState>(() => makeDemo(42));
  const [history, setHistory] = useState<number[]>([]);
  const [steps, setSteps] = useState(0);

  const step = (n: number) => {
    let loss = 0;
    for (let i = 0; i < n; i++) loss = trainStep(state);
    setHistory((h) => [...h, loss].slice(-200));
    setSteps((s) => s + n);
  };

  const current = history[history.length - 1];

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
      <button onClick={() => step(1)}>1 passo</button>
      <button onClick={() => step(50)}>50 passos</button>
      <span data-testid="loss" style={{ fontFamily: "monospace" }}>
        loss: {current === undefined ? "—" : current.toFixed(4)}
      </span>
      <span style={{ fontSize: 12, opacity: 0.7 }}>passos: {steps}</span>
    </div>
  );
}
