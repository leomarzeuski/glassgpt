import type { ReactNode } from "react";
import { GraphExplorer } from "./web/GraphExplorer";
import { TrainingDemo } from "./web/TrainingDemo";

function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontSize: 11,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        color: "var(--muted)",
        fontWeight: 600,
        marginBottom: 10,
      }}
    >
      {children}
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <div
      style={{
        flex: "1 1 180px",
        background: "var(--panel)",
        border: "1px solid var(--hairline)",
        borderRadius: 10,
        padding: "12px 14px",
      }}
    >
      <div className="mono" style={{ color: "var(--data)", fontSize: 13, marginBottom: 4 }}>
        {n}
      </div>
      <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 2 }}>{title}</div>
      <div style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.45 }}>{children}</div>
    </div>
  );
}

export default function App() {
  return (
    <main style={{ maxWidth: 1040, margin: "0 auto", padding: "48px 24px 80px" }}>
      <header style={{ marginBottom: 44 }}>
        <Eyebrow>GlassGPT · Degrau 1 de 3</Eyebrow>
        <h1 style={{ fontSize: 44, lineHeight: 1.05, maxWidth: 620 }}>Veja uma rede neural pensar.</h1>
        <p style={{ fontSize: 17, lineHeight: 1.6, color: "var(--text)", maxWidth: 640, marginTop: 16 }}>
          Toda IA aprende ajustando números a partir de um sinal chamado <strong style={{ color: "var(--grad)" }}>gradiente</strong> —
          o quanto cada número "tem culpa" pelo resultado. Normalmente isso é invisível. Aqui, acontece na sua frente,
          num motor de autograd escrito <strong>do zero em TypeScript</strong>.
        </p>
        <p style={{ fontSize: 13, lineHeight: 1.6, color: "var(--muted)", maxWidth: 640, marginTop: 12 }}>
          ℹ️ Pra quem é da área: autograd escalar estilo micrograd (reverse-mode), sem nenhuma biblioteca de ML, com a
          corretude provada por gradient checking (derivada analítica vs. numérica).
        </p>
      </header>

      <section style={{ marginBottom: 56 }}>
        <Eyebrow>O cálculo, passo a passo</Eyebrow>
        <h2 style={{ fontSize: 24, marginBottom: 14 }}>Explorador do grafo</h2>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
          <Step n={1} title="Mexa nas entradas">
            Arraste <span className="mono" style={{ color: "var(--data)" }}>a</span>,{" "}
            <span className="mono" style={{ color: "var(--data)" }}>b</span>,{" "}
            <span className="mono" style={{ color: "var(--data)" }}>c</span> e veja o resultado{" "}
            <span className="mono" style={{ color: "var(--data)" }}>f</span> recalcular.
          </Step>
          <Step n={2} title="Anime o backward">
            Clique em <span style={{ color: "var(--grad)", fontWeight: 600 }}>Animar backward</span> pra rodar a
            retropropagação.
          </Step>
          <Step n={3} title="Veja os gradientes acenderem">
            Os <span style={{ color: "var(--grad)" }}>grad</span> em âmbar acendem de trás pra frente — é a "culpa" de
            cada número fluindo do resultado até as entradas.
          </Step>
        </div>

        <GraphExplorer />
      </section>

      <section style={{ marginBottom: 40 }}>
        <Eyebrow>O aprendizado</Eyebrow>
        <h2 style={{ fontSize: 24, marginBottom: 8 }}>Treino ao vivo</h2>
        <p style={{ fontSize: 15, lineHeight: 1.6, color: "var(--muted)", maxWidth: 640, marginBottom: 18 }}>
          Uma rede minúscula tenta aprender o <strong style={{ color: "var(--text)" }}>XOR</strong> (responder +1 quando
          as entradas são diferentes, −1 quando são iguais). A <strong style={{ color: "var(--text)" }}>loss</strong> mede
          o erro: cada passo usa os gradientes pra ajustar os pesos e errar menos. Quanto menor, mais ela aprendeu.
        </p>
        <TrainingDemo />
      </section>

      <footer style={{ borderTop: "1px solid var(--hairline)", paddingTop: 20, fontSize: 13, color: "var(--muted)" }}>
        <span>
          Código aberto:{" "}
          <a href="https://github.com/leomarzeuski/glassgpt" target="_blank" rel="noreferrer">
            github.com/leomarzeuski/glassgpt
          </a>
        </span>
        <span style={{ margin: "0 10px" }}>·</span>
        <span>
          Escada: <span style={{ color: "var(--text)" }}>autograd</span> → makemore → GPT
        </span>
      </footer>
    </main>
  );
}
