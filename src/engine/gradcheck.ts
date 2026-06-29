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
