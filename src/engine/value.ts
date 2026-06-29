export interface ValueOpts {
  label?: string;
  op?: string;
  prev?: Value[];
}

export class Value {
  data: number;
  grad = 0;
  label: string;
  op: string;
  prev: Value[];
  _backward: () => void = () => {};

  constructor(data: number, opts: ValueOpts = {}) {
    this.data = data;
    this.label = opts.label ?? "";
    this.op = opts.op ?? "";
    this.prev = opts.prev ?? [];
  }

  private static toValue(x: Value | number): Value {
    return x instanceof Value ? x : new Value(x);
  }

  add(other: Value | number): Value {
    const o = Value.toValue(other);
    const out = new Value(this.data + o.data, { op: "+", prev: [this, o] });
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = Value.toValue(other);
    const out = new Value(this.data * o.data, { op: "*", prev: [this, o] });
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  pow(exponent: number): Value {
    const out = new Value(this.data ** exponent, {
      op: `**${exponent}`,
      prev: [this],
    });
    out._backward = () => {
      this.grad += exponent * this.data ** (exponent - 1) * out.grad;
    };
    return out;
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }

  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  exp(): Value {
    const out = new Value(Math.exp(this.data), { op: "exp", prev: [this] });
    out._backward = () => {
      this.grad += out.data * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, { op: "tanh", prev: [this] });
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, { op: "ReLU", prev: [this] });
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  /** Ordena topologicamente, zera grads, seta grad=1 na raiz e propaga. */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v.prev) build(child);
      topo.push(v);
    };
    build(this);

    for (const v of topo) v.grad = 0;
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}
