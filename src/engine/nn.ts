import { Value } from "./value";

const randW = (rng: () => number) => rng() * 2 - 1; // [-1, 1)

export class Neuron {
  w: Value[];
  b: Value;
  nonlin: boolean;

  constructor(nin: number, nonlin = true, rng: () => number = Math.random) {
    this.w = Array.from({ length: nin }, () => new Value(randW(rng), { label: "w" }));
    this.b = new Value(0, { label: "b" });
    this.nonlin = nonlin;
  }

  call(x: Value[]): Value {
    let act: Value = this.b;
    for (let i = 0; i < this.w.length; i++) {
      act = act.add(this.w[i].mul(x[i]));
    }
    return this.nonlin ? act.tanh() : act;
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  neurons: Neuron[];

  constructor(nin: number, nout: number, nonlin = true, rng: () => number = Math.random) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, nonlin, rng));
  }

  call(x: Value[]): Value[] {
    return this.neurons.map((n) => n.call(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

export class MLP {
  layers: Layer[];

  constructor(nin: number, nouts: number[], rng: () => number = Math.random) {
    const sizes = [nin, ...nouts];
    this.layers = nouts.map(
      (_, i) => new Layer(sizes[i], sizes[i + 1], i < nouts.length - 1, rng),
    );
  }

  call(x: Value[]): Value[] {
    return this.layers.reduce((acc, layer) => layer.call(acc), x);
  }

  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }
}

export function mseLoss(preds: Value[], targets: number[]): Value {
  let loss = new Value(0);
  for (let i = 0; i < preds.length; i++) {
    const diff = preds[i].sub(targets[i]);
    loss = loss.add(diff.mul(diff));
  }
  return loss.div(preds.length);
}

export function sgdStep(params: Value[], lr: number): void {
  for (const p of params) p.data -= lr * p.grad;
}
