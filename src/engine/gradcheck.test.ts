import { describe, it, expect } from "vitest";
import { Value } from "./value";
import { gradCheck } from "./gradcheck";

const allOk = (rs: { ok: boolean }[]) => rs.every((r) => r.ok);

describe("gradCheck — gradiente analítico bate com o numérico", () => {
  it("add/mul: f = x*y + x", () => {
    const x = new Value(2, { label: "x" });
    const y = new Value(-3, { label: "y" });
    expect(allOk(gradCheck([x, y], ([a, b]) => a.mul(b).add(a)))).toBe(true);
  });

  it("tanh: f = tanh(x)", () => {
    const x = new Value(0.5, { label: "x" });
    expect(allOk(gradCheck([x], ([a]) => a.tanh()))).toBe(true);
  });

  it("div/pow/exp: f = exp(x) / (y^2)", () => {
    const x = new Value(0.7, { label: "x" });
    const y = new Value(1.3, { label: "y" });
    expect(allOk(gradCheck([x, y], ([a, b]) => a.exp().div(b.pow(2))))).toBe(true);
  });

  it("relu: f = relu(x) com x > 0", () => {
    const x = new Value(1.5, { label: "x" });
    expect(allOk(gradCheck([x], ([a]) => a.relu()))).toBe(true);
  });

  it("expressão composta (mini-neurônio): tanh(w*x + b)", () => {
    const w = new Value(0.8, { label: "w" });
    const x = new Value(-1.2, { label: "x" });
    const b = new Value(0.3, { label: "b" });
    expect(allOk(gradCheck([w, x, b], ([vw, vx, vb]) => vw.mul(vx).add(vb).tanh()))).toBe(true);
  });
});
