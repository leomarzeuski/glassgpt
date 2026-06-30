import { describe, it, expect } from "vitest";
import { makeDemo, trainStep } from "./trainingDemo";

describe("trainingDemo", () => {
  it("a loss cai ao longo dos passos (seed fixa)", () => {
    const state = makeDemo(42);
    const first = trainStep(state);
    let last = first;
    for (let i = 0; i < 300; i++) last = trainStep(state);
    expect(last).toBeLessThan(first);
  });
});
