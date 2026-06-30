import { MLP, mseLoss, sgdStep } from "./nn";
import { mulberry32 } from "./rng";
import { Value } from "./value";

export interface DemoState {
  mlp: MLP;
  xs: number[][];
  ys: number[];
}

export function makeDemo(seed = 42): DemoState {
  const mlp = new MLP(2, [4, 4, 1], mulberry32(seed));
  const xs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  const ys = [-1, 1, 1, -1]; // XOR-ish
  return { mlp, xs, ys };
}

export function trainStep(state: DemoState, lr = 0.05): number {
  const { mlp, xs, ys } = state;
  const preds = xs.map((x) => mlp.call(x.map((v) => new Value(v)))[0]);
  const loss = mseLoss(preds, ys);
  loss.backward();
  sgdStep(mlp.parameters(), lr);
  return loss.data;
}
