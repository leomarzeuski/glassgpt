import { describe, it, expect } from "vitest";
import { Value } from "./value";

describe("Value — forward", () => {
  it("soma e multiplica valores", () => {
    const a = new Value(2);
    const b = new Value(-3);
    expect(a.add(b).data).toBe(-1);
    expect(a.mul(b).data).toBe(-6);
  });

  it("aceita número cru como operando", () => {
    expect(new Value(2).add(5).data).toBe(7);
    expect(new Value(2).mul(5).data).toBe(10);
  });
});

describe("Value — backward (exemplo canônico do micrograd)", () => {
  it("calcula os gradientes corretos", () => {
    const a = new Value(2, { label: "a" });
    const b = new Value(-3, { label: "b" });
    const c = new Value(10, { label: "c" });
    const f = new Value(-2, { label: "f" });
    const e = a.mul(b);
    const d = e.add(c);
    const L = d.mul(f);
    L.backward();

    expect(L.data).toBe(-8);
    expect(f.grad).toBe(4);
    expect(d.grad).toBe(-2);
    expect(c.grad).toBe(-2);
    expect(e.grad).toBe(-2);
    expect(a.grad).toBe(6);
    expect(b.grad).toBe(-4);
  });

  it("acumula gradiente em nó reutilizado (b = a + a)", () => {
    const a = new Value(3, { label: "a" });
    const b = a.add(a);
    b.backward();
    expect(b.data).toBe(6);
    expect(a.grad).toBe(2); // db/da = 1 + 1
  });

  it("backward é idempotente (chamar 2x dá o mesmo grad)", () => {
    const a = new Value(3);
    const out = a.mul(a); // out = a^2, d/da = 2a = 6
    out.backward();
    out.backward();
    expect(a.grad).toBe(6);
  });
});
