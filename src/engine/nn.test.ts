import { describe, it, expect } from "vitest";
import { mulberry32 } from "./rng";
import { MLP, mseLoss, sgdStep } from "./nn";
import { Value } from "./value";

describe("mulberry32", () => {
  it("é determinístico para a mesma seed", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    expect(a()).toBe(b());
    expect(a()).toBe(b());
  });

  it("produz valores em [0,1)", () => {
    const r = mulberry32(1);
    for (let i = 0; i < 100; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe("MLP", () => {
  it("forward produz o número de saídas certo", () => {
    const mlp = new MLP(2, [4, 4, 1], mulberry32(0));
    const out = mlp.call([new Value(0.5), new Value(-0.5)]);
    expect(out).toHaveLength(1);
    expect(typeof out[0].data).toBe("number");
  });

  it("expõe todos os parâmetros", () => {
    const mlp = new MLP(2, [3, 1], mulberry32(0));
    // camada1: 3 neurônios * (2 pesos + 1 bias) = 9; camada2: 1*(3+1)=4
    expect(mlp.parameters()).toHaveLength(13);
  });
});

describe("treino reduz a loss", () => {
  it("MSE cai após passos de SGD", () => {
    const mlp = new MLP(2, [4, 4, 1], mulberry32(42));
    const xs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const ys = [-1, 1, 1, -1]; // XOR-ish

    const stepLoss = (): number => {
      const preds = xs.map((x) => mlp.call(x.map((v) => new Value(v)))[0]);
      const loss = mseLoss(preds, ys);
      loss.backward();
      sgdStep(mlp.parameters(), 0.05);
      return loss.data;
    };

    const first = stepLoss();
    let last = first;
    for (let i = 0; i < 200; i++) last = stepLoss();
    expect(last).toBeLessThan(first);
  });
});
