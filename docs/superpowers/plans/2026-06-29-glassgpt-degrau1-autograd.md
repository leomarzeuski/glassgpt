# GlassGPT Degrau 1 — Autograd + Visualizador do Grafo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir um motor de autograd escalar em TypeScript puro e um visualizador interativo do grafo de computação no navegador, mostrando os gradientes fluírem no backprop.

**Architecture:** Um **engine** em TS puro (sem DOM, zero bibliotecas de ML) implementa `Value` (autograd escalar), uma mini-rede neural, e utilitários de grafo. Uma camada **web** (React + React Flow) apenas *consome* o engine e desenha — ela nunca faz contas. Toda a matemática é testada no Node com gradient checking.

**Tech Stack:** Vite, React 19, TypeScript (strict), `@xyflow/react` (React Flow v12), `dagre` (layout), Vitest + Testing Library (jsdom). Deploy na Vercel.

## Global Constraints

- TypeScript em **strict mode**; `noUnusedLocals` e `noUnusedParameters` ligados.
- O **engine** (`src/engine/**`) é TS puro: **proibido** importar React, DOM ou qualquer biblioteca de ML. Só matemática.
- React Flow **apenas renderiza**; nenhuma operação matemática vive na camada web.
- Gradient checking: passo `h = 1e-5`, tolerância relativa `tol = 1e-4`.
- Node >= 20; gerenciador de pacotes **pnpm**.
- NaN/Inf em qualquer nó deve ser sinalizado visualmente (nó marcado).
- Cada degrau termina publicado numa URL pública da Vercel.

## File Structure

```
glassgpt/
  package.json            # deps + scripts (dev, build, test)
  tsconfig.json           # TS strict
  vite.config.ts          # Vite + Vitest (jsdom)
  index.html
  .gitignore
  src/
    main.tsx              # entry React
    App.tsx               # layout da página, monta GraphExplorer + TrainingDemo
    test/setup.ts         # jest-dom + polyfill ResizeObserver
    engine/               # === TS PURO, SEM DOM ===
      value.ts            # classe Value: add/mul/.../backward
      gradcheck.ts        # gradient checking por diferenças finitas (util de teste)
      rng.ts              # mulberry32 (RNG determinístico)
      nn.ts               # Neuron, Layer, MLP, mseLoss, sgdStep
      trace.ts            # trace(root) -> Graph {nodes, edges}
      graphController.ts  # forwardGraph, backwardGraph, hasNonFinite
      trainingDemo.ts     # makeDemo, trainStep (usa nn)
    web/                  # === REACT, só consome o engine ===
      layout.ts           # layoutGraph: Graph -> nodes/edges do React Flow (dagre)
      ValueNode.tsx       # nó customizado do React Flow
      GraphExplorer.tsx   # ReactFlow + Forward/Backward + sliders nas folhas
      TrainingDemo.tsx    # botões de passo + loss atual
  docs/superpowers/...    # spec + este plano
  README.md
```

Cada `*.ts(x)` de produção tem um `*.test.ts(x)` ao lado quando há lógica testável.

---

### Task 1: Scaffold do projeto + harness de testes

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `.gitignore`
- Create: `src/main.tsx`, `src/App.tsx`, `src/test/setup.ts`
- Test: `src/engine/smoke.test.ts`

**Interfaces:**
- Consumes: nada.
- Produces: harness funcionando (`pnpm test`, `pnpm dev`, `pnpm build`). `App` default export (placeholder).

- [ ] **Step 1: Criar `package.json`**

```json
{
  "name": "glassgpt",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "@xyflow/react": "^12.3.5",
    "dagre": "^0.8.5",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "@types/dagre": "^0.7.52",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.4",
    "jsdom": "^25.0.1",
    "typescript": "^5.7.2",
    "vite": "^6.0.5",
    "vitest": "^2.1.8"
  }
}
```

- [ ] **Step 2: Criar `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Criar `vite.config.ts`**

```ts
/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
```

- [ ] **Step 4: Criar `src/test/setup.ts`**

```ts
import "@testing-library/jest-dom";

// React Flow precisa de ResizeObserver, ausente no jsdom.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver =
  globalThis.ResizeObserver ?? (ResizeObserverStub as unknown as typeof ResizeObserver);
```

- [ ] **Step 5: Criar `index.html`, `src/main.tsx`, `src/App.tsx`, `.gitignore`**

`index.html`:
```html
<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GlassGPT — backprop visível</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`src/main.tsx`:
```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

`src/App.tsx` (placeholder; será substituído na Task 7):
```tsx
export default function App() {
  return <h1>GlassGPT</h1>;
}
```

`.gitignore`:
```
node_modules
dist
*.local
.vercel
```

- [ ] **Step 6: Criar smoke test `src/engine/smoke.test.ts`**

```ts
import { describe, it, expect } from "vitest";

describe("harness", () => {
  it("roda o vitest", () => {
    expect(1 + 1).toBe(2);
  });
});
```

- [ ] **Step 7: Instalar e rodar**

Run: `pnpm install && pnpm test`
Expected: 1 arquivo de teste, 1 teste, PASS.

- [ ] **Step 8: Verificar o build**

Run: `pnpm build`
Expected: build conclui sem erros, gera `dist/`.

- [ ] **Step 9: Commit**

```bash
git add package.json tsconfig.json vite.config.ts index.html .gitignore src/
git commit -m "chore: scaffold Vite + React + TS + Vitest"
```

---

### Task 2: `Value` — núcleo do autograd (add, mul, backward)

**Files:**
- Create: `src/engine/value.ts`
- Test: `src/engine/value.test.ts`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `class Value` com campos públicos `data: number`, `grad: number`, `label: string`, `op: string`, `prev: Value[]`, `_backward: () => void`.
  - `new Value(data: number, opts?: { label?: string; op?: string; prev?: Value[] })`
  - `add(other: Value | number): Value`
  - `mul(other: Value | number): Value`
  - `backward(): void` — zera os grads do grafo, seta o grad da raiz em 1, propaga.

- [ ] **Step 1: Escrever o teste que falha**

`src/engine/value.test.ts`:
```ts
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
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pnpm test src/engine/value.test.ts`
Expected: FAIL — `Cannot find module './value'`.

- [ ] **Step 3: Implementar `src/engine/value.ts`**

```ts
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
```

- [ ] **Step 4: Rodar e confirmar que passa**

Run: `pnpm test src/engine/value.test.ts`
Expected: PASS (5 testes).

- [ ] **Step 5: Commit**

```bash
git add src/engine/value.ts src/engine/value.test.ts
git commit -m "feat(engine): Value com add, mul e backprop"
```

---

### Task 3: Operações restantes + gradient checking

**Files:**
- Modify: `src/engine/value.ts` (adicionar métodos)
- Create: `src/engine/gradcheck.ts`
- Test: `src/engine/value.test.ts` (adicionar casos), `src/engine/gradcheck.test.ts`

**Interfaces:**
- Consumes: `Value` (Task 2).
- Produces:
  - Em `Value`: `neg(): Value`, `sub(other: Value | number): Value`, `pow(exponent: number): Value`, `div(other: Value | number): Value`, `exp(): Value`, `tanh(): Value`, `relu(): Value`.
  - `gradCheck(inputs: Value[], build: (inputs: Value[]) => Value, h?: number, tol?: number): { name: string; analytic: number; numeric: number; ok: boolean }[]`

- [ ] **Step 1: Escrever os testes que falham (ops + gradcheck)**

Adicionar em `src/engine/value.test.ts`:
```ts
describe("Value — operações derivadas", () => {
  it("neg, sub, div", () => {
    expect(new Value(3).neg().data).toBe(-3);
    expect(new Value(5).sub(2).data).toBe(3);
    expect(new Value(6).div(2).data).toBe(3);
  });

  it("pow, exp, tanh, relu (forward)", () => {
    expect(new Value(2).pow(3).data).toBe(8);
    expect(new Value(0).exp().data).toBeCloseTo(1, 10);
    expect(new Value(0).tanh().data).toBeCloseTo(0, 10);
    expect(new Value(-2).relu().data).toBe(0);
    expect(new Value(2).relu().data).toBe(2);
  });
});
```

Criar `src/engine/gradcheck.test.ts`:
```ts
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
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pnpm test src/engine/gradcheck.test.ts`
Expected: FAIL — `Cannot find module './gradcheck'`.

- [ ] **Step 3: Adicionar as operações em `src/engine/value.ts`**

Inserir estes métodos dentro da classe `Value` (depois de `mul`):
```ts
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
```

- [ ] **Step 4: Implementar `src/engine/gradcheck.ts`**

```ts
import { Value } from "./value";

export interface GradCheckResult {
  name: string;
  analytic: number;
  numeric: number;
  ok: boolean;
}

/**
 * Compara o gradiente analítico (do autograd) com o numérico (diferenças
 * finitas centrais) para cada input. `build` deve reconstruir a expressão a
 * partir dos MESMOS objetos Value passados em `inputs`.
 */
export function gradCheck(
  inputs: Value[],
  build: (inputs: Value[]) => Value,
  h = 1e-5,
  tol = 1e-4,
): GradCheckResult[] {
  const out = build(inputs);
  out.backward();
  const analytic = inputs.map((v) => v.grad);

  return inputs.map((v, i) => {
    const original = v.data;

    v.data = original + h;
    const plus = build(inputs).data;

    v.data = original - h;
    const minus = build(inputs).data;

    v.data = original;

    const numeric = (plus - minus) / (2 * h);
    const denom = Math.max(1, Math.abs(numeric), Math.abs(analytic[i]));
    const ok = Math.abs(numeric - analytic[i]) <= tol * denom;
    return { name: v.label || `x${i}`, analytic: analytic[i], numeric, ok };
  });
}
```

- [ ] **Step 5: Rodar e confirmar que passa**

Run: `pnpm test src/engine/value.test.ts src/engine/gradcheck.test.ts`
Expected: PASS (todos).

- [ ] **Step 6: Commit**

```bash
git add src/engine/value.ts src/engine/value.test.ts src/engine/gradcheck.ts src/engine/gradcheck.test.ts
git commit -m "feat(engine): ops (pow/div/exp/tanh/relu) + gradient checking"
```

---

### Task 4: Mini-rede neural (Neuron, Layer, MLP) + RNG + treino

**Files:**
- Create: `src/engine/rng.ts`, `src/engine/nn.ts`
- Test: `src/engine/nn.test.ts`

**Interfaces:**
- Consumes: `Value` (Tasks 2-3).
- Produces:
  - `mulberry32(seed: number): () => number` — RNG determinístico em [0,1).
  - `class Neuron` — `new Neuron(nin: number, nonlin?: boolean, rng?: () => number)`; `call(x: Value[]): Value`; `parameters(): Value[]`.
  - `class Layer` — `new Layer(nin: number, nout: number, nonlin?: boolean, rng?: () => number)`; `call(x: Value[]): Value[]`; `parameters(): Value[]`.
  - `class MLP` — `new MLP(nin: number, nouts: number[], rng?: () => number)`; `call(x: Value[]): Value[]`; `parameters(): Value[]`. Camada final é linear (nonlin=false).
  - `mseLoss(preds: Value[], targets: number[]): Value`
  - `sgdStep(params: Value[], lr: number): void`

- [ ] **Step 1: Escrever os testes que falham**

`src/engine/nn.test.ts`:
```ts
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
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pnpm test src/engine/nn.test.ts`
Expected: FAIL — `Cannot find module './rng'`.

- [ ] **Step 3: Implementar `src/engine/rng.ts`**

```ts
/** PRNG determinístico (mulberry32). Retorna valores em [0,1). */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
```

- [ ] **Step 4: Implementar `src/engine/nn.ts`**

```ts
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
  return loss;
}

export function sgdStep(params: Value[], lr: number): void {
  for (const p of params) p.data -= lr * p.grad;
}
```

- [ ] **Step 5: Rodar e confirmar que passa**

Run: `pnpm test src/engine/nn.test.ts`
Expected: PASS (todos).

- [ ] **Step 6: Commit**

```bash
git add src/engine/rng.ts src/engine/nn.ts src/engine/nn.test.ts
git commit -m "feat(engine): MLP, MSE, SGD e RNG determinístico"
```

---

### Task 5: Exportação do grafo (`trace`) + controlador (`forwardGraph`/`backwardGraph`/`hasNonFinite`)

**Files:**
- Create: `src/engine/trace.ts`, `src/engine/graphController.ts`
- Test: `src/engine/trace.test.ts`, `src/engine/graphController.test.ts`

**Interfaces:**
- Consumes: `Value` (Tasks 2-3).
- Produces:
  - `interface GraphNode { id: string; label: string; data: number; grad: number; op: string }`
  - `interface GraphEdge { source: string; target: string }`
  - `interface Graph { nodes: GraphNode[]; edges: GraphEdge[] }`
  - `trace(root: Value): Graph`
  - `forwardGraph(root: Value): Graph` — zera grads (sem backward) e devolve o trace.
  - `backwardGraph(root: Value): Graph` — roda `root.backward()` e devolve o trace (com grads).
  - `hasNonFinite(graph: Graph): boolean`

- [ ] **Step 1: Escrever os testes que falham**

`src/engine/trace.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { Value } from "./value";
import { trace } from "./trace";

describe("trace", () => {
  it("captura todos os nós e arestas de uma expressão", () => {
    const a = new Value(2, { label: "a" });
    const b = new Value(3, { label: "b" });
    const out = a.mul(b); // nós: a, b, out -> 3 nós, 2 arestas
    out.label = "out";

    const g = trace(out);
    expect(g.nodes).toHaveLength(3);
    expect(g.edges).toHaveLength(2);

    const outNode = g.nodes.find((n) => n.label === "out");
    expect(outNode?.data).toBe(6);
    expect(outNode?.op).toBe("*");
  });

  it("não duplica um nó reutilizado", () => {
    const a = new Value(2, { label: "a" });
    const out = a.add(a); // a aparece 2x como operando, mas é 1 nó só
    const g = trace(out);
    expect(g.nodes).toHaveLength(2); // a, out
    expect(g.edges).toHaveLength(2); // a->out duas vezes
  });
});
```

`src/engine/graphController.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { Value } from "./value";
import { forwardGraph, backwardGraph, hasNonFinite } from "./graphController";

describe("forwardGraph / backwardGraph", () => {
  const build = () => new Value(3, { label: "a" }).mul(new Value(2, { label: "b" }));

  it("forwardGraph deixa todos os grads zerados", () => {
    const g = forwardGraph(build());
    expect(g.nodes.every((n) => n.grad === 0)).toBe(true);
  });

  it("backwardGraph preenche os grads", () => {
    const g = backwardGraph(build());
    const root = g.nodes.find((n) => n.op === "*")!;
    expect(root.grad).toBe(1);
    // grad de cada folha != 0
    expect(g.nodes.filter((n) => n.op === "").every((n) => n.grad !== 0)).toBe(true);
  });
});

describe("hasNonFinite", () => {
  it("detecta Infinity (divisão por zero)", () => {
    const g = backwardGraph(new Value(1).div(0));
    expect(hasNonFinite(g)).toBe(true);
  });

  it("é falso para grafo normal", () => {
    const g = backwardGraph(new Value(1).add(2));
    expect(hasNonFinite(g)).toBe(false);
  });
});
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pnpm test src/engine/trace.test.ts`
Expected: FAIL — `Cannot find module './trace'`.

- [ ] **Step 3: Implementar `src/engine/trace.ts`**

```ts
import { Value } from "./value";

export interface GraphNode {
  id: string;
  label: string;
  data: number;
  grad: number;
  op: string;
}

export interface GraphEdge {
  source: string; // operando (filho)
  target: string; // resultado (pai)
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export function trace(root: Value): Graph {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const ids = new Map<Value, string>();
  let counter = 0;

  const idOf = (v: Value): string => {
    let id = ids.get(v);
    if (id === undefined) {
      id = `n${counter++}`;
      ids.set(v, id);
    }
    return id;
  };

  const visited = new Set<Value>();
  const visit = (v: Value) => {
    if (visited.has(v)) return;
    visited.add(v);
    nodes.push({ id: idOf(v), label: v.label, data: v.data, grad: v.grad, op: v.op });
    for (const child of v.prev) {
      edges.push({ source: idOf(child), target: idOf(v) });
      visit(child);
    }
  };

  visit(root);
  return { nodes, edges };
}
```

- [ ] **Step 4: Implementar `src/engine/graphController.ts`**

```ts
import { Value } from "./value";
import { trace, type Graph } from "./trace";

/** Garante grads zerados (estado "antes do backward") e devolve o trace. */
export function forwardGraph(root: Value): Graph {
  zeroGrads(root);
  return trace(root);
}

/** Roda o backward e devolve o trace já com os gradientes. */
export function backwardGraph(root: Value): Graph {
  root.backward();
  return trace(root);
}

export function hasNonFinite(graph: Graph): boolean {
  return graph.nodes.some((n) => !Number.isFinite(n.data) || !Number.isFinite(n.grad));
}

function zeroGrads(root: Value): void {
  const visited = new Set<Value>();
  const visit = (v: Value) => {
    if (visited.has(v)) return;
    visited.add(v);
    v.grad = 0;
    for (const child of v.prev) visit(child);
  };
  visit(root);
}
```

- [ ] **Step 5: Rodar e confirmar que passa**

Run: `pnpm test src/engine/trace.test.ts src/engine/graphController.test.ts`
Expected: PASS (todos).

- [ ] **Step 6: Commit**

```bash
git add src/engine/trace.ts src/engine/trace.test.ts src/engine/graphController.ts src/engine/graphController.test.ts
git commit -m "feat(engine): trace do grafo + forward/backward graph + hasNonFinite"
```

---

### Task 6: Primitivas de visualização — layout (dagre) + nó customizado

**Files:**
- Create: `src/web/layout.ts`, `src/web/ValueNode.tsx`
- Test: `src/web/layout.test.ts`, `src/web/ValueNode.test.tsx`

**Interfaces:**
- Consumes: `Graph`, `GraphNode` (Task 5); `@xyflow/react` (`Node`, `Edge`, `Handle`, `Position`, `ReactFlowProvider`).
- Produces:
  - `interface ValueNodeData { label: string; value: number; grad: number; op: string }`
  - `function ValueNode(props: { data: ValueNodeData }): JSX.Element` — nó com `data-testid="value-node"` e atributo `data-nonfinite`.
  - `function layoutGraph(graph: Graph): { nodes: Node[]; edges: Edge[] }` — nós do tipo `"valueNode"` com `data: ValueNodeData`.

- [ ] **Step 1: Escrever os testes que falham**

`src/web/layout.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { Value } from "../engine/value";
import { trace } from "../engine/trace";
import { layoutGraph } from "./layout";

describe("layoutGraph", () => {
  it("posiciona todos os nós com coordenadas numéricas", () => {
    const out = new Value(2, { label: "a" }).mul(new Value(3, { label: "b" }));
    const { nodes, edges } = layoutGraph(trace(out));

    expect(nodes).toHaveLength(3);
    expect(edges).toHaveLength(2);
    for (const n of nodes) {
      expect(Number.isFinite(n.position.x)).toBe(true);
      expect(Number.isFinite(n.position.y)).toBe(true);
      expect(n.type).toBe("valueNode");
    }
  });
});
```

`src/web/ValueNode.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { ValueNode, type ValueNodeData } from "./ValueNode";

const renderNode = (data: ValueNodeData) =>
  render(
    <ReactFlowProvider>
      <ValueNode data={data} />
    </ReactFlowProvider>,
  );

describe("ValueNode", () => {
  it("mostra valor e gradiente", () => {
    renderNode({ label: "x", value: 1.2345, grad: 0.5, op: "" });
    expect(screen.getByText(/1\.2345/)).toBeInTheDocument();
    expect(screen.getByText(/0\.5000/)).toBeInTheDocument();
  });

  it("marca nós com valor não-finito", () => {
    renderNode({ label: "", value: Infinity, grad: 0, op: "/" });
    expect(screen.getByTestId("value-node")).toHaveAttribute("data-nonfinite", "true");
  });
});
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pnpm test src/web/layout.test.ts`
Expected: FAIL — `Cannot find module './layout'`.

- [ ] **Step 3: Implementar `src/web/ValueNode.tsx`**

```tsx
import { Handle, Position } from "@xyflow/react";

export interface ValueNodeData {
  label: string;
  value: number;
  grad: number;
  op: string;
}

export function ValueNode({ data }: { data: ValueNodeData }) {
  const nonFinite = !Number.isFinite(data.value) || !Number.isFinite(data.grad);
  return (
    <div
      data-testid="value-node"
      data-nonfinite={nonFinite}
      style={{
        border: `2px solid ${nonFinite ? "#e11d48" : "#334155"}`,
        borderRadius: 8,
        padding: "6px 10px",
        background: "#0f172a",
        color: "#e2e8f0",
        fontFamily: "monospace",
        fontSize: 12,
        minWidth: 120,
      }}
    >
      <Handle type="target" position={Position.Left} />
      <div style={{ fontSize: 11, opacity: 0.7 }}>{data.label || data.op || "•"}</div>
      <div>data {data.value.toFixed(4)}</div>
      <div style={{ color: "#38bdf8" }}>grad {data.grad.toFixed(4)}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
```

- [ ] **Step 4: Implementar `src/web/layout.ts`**

```ts
import dagre from "dagre";
import type { Node, Edge } from "@xyflow/react";
import type { Graph } from "../engine/trace";
import type { ValueNodeData } from "./ValueNode";

const NODE_W = 150;
const NODE_H = 72;

export function layoutGraph(graph: Graph): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "LR", nodesep: 30, ranksep: 80 });

  for (const n of graph.nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of graph.edges) g.setEdge(e.source, e.target);

  dagre.layout(g);

  const nodes: Node[] = graph.nodes.map((n) => {
    const pos = g.node(n.id);
    const data: ValueNodeData = { label: n.label, value: n.data, grad: n.grad, op: n.op };
    return {
      id: n.id,
      type: "valueNode",
      position: { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 },
      data: data as unknown as Record<string, unknown>,
    };
  });

  const edges: Edge[] = graph.edges.map((e, i) => ({
    id: `e${i}`,
    source: e.source,
    target: e.target,
    animated: true,
  }));

  return { nodes, edges };
}
```

- [ ] **Step 5: Rodar e confirmar que passa**

Run: `pnpm test src/web/layout.test.ts src/web/ValueNode.test.tsx`
Expected: PASS (todos).

- [ ] **Step 6: Commit**

```bash
git add src/web/layout.ts src/web/layout.test.ts src/web/ValueNode.tsx src/web/ValueNode.test.tsx
git commit -m "feat(web): layout dagre + nó customizado com flag de não-finito"
```

---

### Task 7: Explorador do grafo (ReactFlow + Forward/Backward + sliders)

**Files:**
- Create: `src/web/GraphExplorer.tsx`
- Modify: `src/App.tsx`
- Test: `src/web/GraphExplorer.test.tsx`

**Interfaces:**
- Consumes: `Value` (Task 3), `forwardGraph`/`backwardGraph` (Task 5), `layoutGraph` (Task 6), `ValueNode` (Task 6), `@xyflow/react`.
- Produces: `function GraphExplorer(): JSX.Element`. `App` renderiza `GraphExplorer` dentro de uma seção com heading "GlassGPT".

- [ ] **Step 1: Escrever o teste que falha**

`src/web/GraphExplorer.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { GraphExplorer } from "./GraphExplorer";

describe("GraphExplorer", () => {
  it("renderiza os controles Forward/Backward e os sliders", () => {
    render(<GraphExplorer />);
    expect(screen.getByRole("button", { name: /forward/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /backward/i })).toBeInTheDocument();
    expect(screen.getAllByRole("slider").length).toBeGreaterThanOrEqual(3);
  });

  it("clicar em Backward não quebra a renderização", () => {
    render(<GraphExplorer />);
    fireEvent.click(screen.getByRole("button", { name: /backward/i }));
    expect(screen.getByRole("button", { name: /backward/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pnpm test src/web/GraphExplorer.test.tsx`
Expected: FAIL — `Cannot find module './GraphExplorer'`.

- [ ] **Step 3: Implementar `src/web/GraphExplorer.tsx`**

```tsx
import { useMemo, useState } from "react";
import { ReactFlow, ReactFlowProvider, Background, Controls } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Value } from "../engine/value";
import { forwardGraph, backwardGraph } from "../engine/graphController";
import { layoutGraph } from "./layout";
import { ValueNode } from "./ValueNode";

const nodeTypes = { valueNode: ValueNode };

// Expressão de exemplo: f = tanh(a*b + c)
function buildExpression(a: number, b: number, c: number): Value {
  const va = new Value(a, { label: "a" });
  const vb = new Value(b, { label: "b" });
  const vc = new Value(c, { label: "c" });
  const out = va.mul(vb).add(vc).tanh();
  out.label = "f";
  return out;
}

type Phase = "forward" | "backward";

function Slider(props: {
  name: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label style={{ display: "flex", flexDirection: "column", fontSize: 12 }}>
      {props.name} = {props.value.toFixed(2)}
      <input
        type="range"
        min={-5}
        max={5}
        step={0.1}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function GraphExplorer() {
  const [a, setA] = useState(2);
  const [b, setB] = useState(-3);
  const [c, setC] = useState(1);
  const [phase, setPhase] = useState<Phase>("forward");

  const graph = useMemo(() => {
    const root = buildExpression(a, b, c);
    return phase === "backward" ? backwardGraph(root) : forwardGraph(root);
  }, [a, b, c, phase]);

  const { nodes, edges } = useMemo(() => layoutGraph(graph), [graph]);

  return (
    <div>
      <div style={{ display: "flex", gap: 16, alignItems: "flex-end", marginBottom: 12, flexWrap: "wrap" }}>
        <button onClick={() => setPhase("forward")}>Forward</button>
        <button onClick={() => setPhase("backward")}>Backward</button>
        <Slider name="a" value={a} onChange={setA} />
        <Slider name="b" value={b} onChange={setB} />
        <Slider name="c" value={c} onChange={setC} />
      </div>
      <div style={{ height: 480, border: "1px solid #1e293b", borderRadius: 8 }}>
        <ReactFlowProvider>
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView>
            <Background />
            <Controls />
          </ReactFlow>
        </ReactFlowProvider>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Atualizar `src/App.tsx`**

```tsx
import { GraphExplorer } from "./web/GraphExplorer";

export default function App() {
  return (
    <main
      style={{
        maxWidth: 1000,
        margin: "0 auto",
        padding: 24,
        color: "#e2e8f0",
        background: "#020617",
        minHeight: "100vh",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <h1>GlassGPT — backprop visível</h1>
      <p>Um motor de autograd escrito do zero em TypeScript. Arraste os valores e veja os gradientes fluírem.</p>
      <section>
        <h2>Explorador do grafo</h2>
        <GraphExplorer />
      </section>
    </main>
  );
}
```

- [ ] **Step 5: Rodar e confirmar que passa**

Run: `pnpm test src/web/GraphExplorer.test.tsx`
Expected: PASS (2 testes).

- [ ] **Step 6: Verificação manual no navegador**

Run: `pnpm dev`
Expected: abrir o localhost; ver o grafo de `tanh(a*b + c)`; arrastar sliders muda os valores; clicar "Backward" mostra grads != 0 nos nós.

- [ ] **Step 7: Commit**

```bash
git add src/web/GraphExplorer.tsx src/web/GraphExplorer.test.tsx src/App.tsx
git commit -m "feat(web): explorador interativo do grafo (forward/backward + sliders)"
```

---

### Task 8: Demo de treino ao vivo

**Files:**
- Create: `src/engine/trainingDemo.ts`, `src/web/TrainingDemo.tsx`
- Modify: `src/App.tsx`
- Test: `src/engine/trainingDemo.test.ts`, `src/web/TrainingDemo.test.tsx`

**Interfaces:**
- Consumes: `MLP`, `mseLoss`, `sgdStep` (Task 4), `mulberry32` (Task 4), `Value` (Task 2).
- Produces:
  - `interface DemoState { mlp: MLP; xs: number[][]; ys: number[] }`
  - `function makeDemo(seed?: number): DemoState`
  - `function trainStep(state: DemoState, lr?: number): number` — roda 1 passo e devolve a loss.
  - `function TrainingDemo(): JSX.Element` — botões "1 passo"/"50 passos" e `data-testid="loss"`.

- [ ] **Step 1: Escrever os testes que falham**

`src/engine/trainingDemo.test.ts`:
```ts
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
```

`src/web/TrainingDemo.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TrainingDemo } from "./TrainingDemo";

describe("TrainingDemo", () => {
  it("mostra a loss após um passo", () => {
    render(<TrainingDemo />);
    expect(screen.getByTestId("loss")).toHaveTextContent("—");
    fireEvent.click(screen.getByRole("button", { name: /1 passo/i }));
    expect(screen.getByTestId("loss")).not.toHaveTextContent("—");
  });
});
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run: `pnpm test src/engine/trainingDemo.test.ts`
Expected: FAIL — `Cannot find module './trainingDemo'`.

- [ ] **Step 3: Implementar `src/engine/trainingDemo.ts`**

```ts
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
```

- [ ] **Step 4: Implementar `src/web/TrainingDemo.tsx`**

```tsx
import { useState } from "react";
import { makeDemo, trainStep, type DemoState } from "../engine/trainingDemo";

export function TrainingDemo() {
  const [state] = useState<DemoState>(() => makeDemo(42));
  const [history, setHistory] = useState<number[]>([]);

  const step = (n: number) => {
    let loss = 0;
    for (let i = 0; i < n; i++) loss = trainStep(state);
    setHistory((h) => [...h, loss].slice(-200));
  };

  const current = history.at(-1);

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
      <button onClick={() => step(1)}>1 passo</button>
      <button onClick={() => step(50)}>50 passos</button>
      <span data-testid="loss" style={{ fontFamily: "monospace" }}>
        loss: {current === undefined ? "—" : current.toFixed(4)}
      </span>
      <span style={{ fontSize: 12, opacity: 0.7 }}>passos: {history.length}</span>
    </div>
  );
}
```

- [ ] **Step 5: Adicionar a seção em `src/App.tsx`**

Adicionar o import no topo:
```tsx
import { TrainingDemo } from "./web/TrainingDemo";
```
E inserir, logo após a seção "Explorador do grafo" (antes de fechar `</main>`):
```tsx
      <section>
        <h2>Treino ao vivo</h2>
        <p>Uma MLP minúscula aprendendo XOR. Clique e veja a loss cair.</p>
        <TrainingDemo />
      </section>
```

- [ ] **Step 6: Rodar e confirmar que passa**

Run: `pnpm test src/engine/trainingDemo.test.ts src/web/TrainingDemo.test.tsx`
Expected: PASS (todos).

- [ ] **Step 7: Commit**

```bash
git add src/engine/trainingDemo.ts src/engine/trainingDemo.test.ts src/web/TrainingDemo.tsx src/web/TrainingDemo.test.tsx src/App.tsx
git commit -m "feat: demo de treino ao vivo (XOR) com loss decrescente"
```

---

### Task 9: README + build de produção + deploy na Vercel

**Files:**
- Create: `README.md`
- (Sem testes novos; deliverable = build verde + URL pública.)

**Interfaces:**
- Consumes: tudo das tasks anteriores.
- Produces: `README.md` e uma URL pública da Vercel.

- [ ] **Step 1: Rodar a suíte completa**

Run: `pnpm test`
Expected: TODOS os arquivos de teste PASS.

- [ ] **Step 2: Verificar o build de produção**

Run: `pnpm build`
Expected: `tsc --noEmit` sem erros + `vite build` gera `dist/`.

- [ ] **Step 3: Criar `README.md`**

```markdown
# GlassGPT — Degrau 1: backprop visível

Um motor de **autograd** (diferenciação automática) escrito **do zero em
TypeScript**, com um visualizador interativo do grafo de computação rodando no
navegador. Sem bibliotecas de ML — só a matemática.

Este é o primeiro degrau da escada GlassGPT: `autograd → makemore → GPT`.

## O que dá pra ver

- **Explorador do grafo:** monte `tanh(a*b + c)`, arraste os valores e clique em
  *Backward* para ver os gradientes fluírem de trás pra frente, nó a nó.
- **Treino ao vivo:** uma MLP minúscula aprendendo XOR; assista a loss cair.

## Como funciona

A classe `Value` (em `src/engine/value.ts`) embrulha um escalar e registra, em
cada operação, **como o gradiente flui de volta**. `backward()` ordena o grafo
topologicamente e aplica a regra da cadeia. A correção é **provada por gradient
checking**: cada gradiente analítico é comparado com a estimativa numérica por
diferenças finitas (`src/engine/gradcheck.ts`).

## Rodando

\`\`\`bash
pnpm install
pnpm dev     # app em http://localhost:5173
pnpm test    # suíte completa (inclui gradient checking)
\`\`\`

## Arquitetura

- `src/engine/**` — TS puro, sem DOM, zero libs de ML. Testável no Node.
- `src/web/**` — React + React Flow; apenas *consome* o engine e desenha.
```

- [ ] **Step 4: Commit do README**

```bash
git add README.md
git commit -m "docs: README do Degrau 1"
```

- [ ] **Step 5: Deploy na Vercel**

Login é interativo — no chat, rode com o prefixo `!`:
```
! npx vercel login
```
Depois faça o deploy de produção:
```bash
npx vercel --prod
```
Expected: a Vercel detecta Vite, builda e devolve uma URL pública (ex:
`https://glassgpt.vercel.app`). Abrir a URL e confirmar que o explorador e o
treino funcionam ao vivo.

- [ ] **Step 6: Adicionar a URL ao README e commitar**

Adicionar perto do topo do `README.md`: `**Demo ao vivo:** <URL>` e:
```bash
git add README.md
git commit -m "docs: link da demo ao vivo"
```

---

## Self-Review (preenchido pelo autor do plano)

**Cobertura do spec:**
- Engine puro `Value` + ops + backward → Tasks 2-3 ✔
- Mini-rede (Neuron/Layer/MLP), treino → Task 4 ✔
- `trace()` exportando o grafo → Task 5 ✔
- Visualização (React Flow), Forward/Backward, sliders → Tasks 6-7 ✔
- Treino ao vivo com loss caindo → Task 8 ✔
- Gradient checking → Task 3 ✔
- Teste de resposta conhecida (micrograd) → Task 2 ✔
- NaN/Inf sinalizado → `hasNonFinite` (Task 5) + `data-nonfinite` no ValueNode (Task 6) ✔
- Erros de `div`/`pow`: tratados de forma didática — valores não-finitos propagam e o nó fica vermelho (Tasks 5-6), em vez de lançar exceção. (Decisão consciente do spec: "erro é didático".)
- Stack Vite/React/React Flow/Vitest/Vercel → Tasks 1, 9 ✔
- Deploy na Vercel → Task 9 ✔

**Placeholders:** nenhum "TBD/TODO"; todo passo tem código/comando real.

**Consistência de tipos:** `Value` (campos e métodos), `Graph`/`GraphNode`/`GraphEdge`, `ValueNodeData`, `DemoState` e as assinaturas de `MLP`/`mseLoss`/`sgdStep`/`trainStep` batem entre as tasks que as consomem.
