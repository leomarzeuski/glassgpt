import { GraphExplorer } from "./web/GraphExplorer";
import { TrainingDemo } from "./web/TrainingDemo";

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
      <section>
        <h2>Treino ao vivo</h2>
        <p>Uma MLP minúscula aprendendo XOR. Clique e veja a loss cair.</p>
        <TrainingDemo />
      </section>
    </main>
  );
}
