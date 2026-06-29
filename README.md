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

```bash
pnpm install
pnpm dev     # app em http://localhost:5173
pnpm test    # suíte completa (inclui gradient checking)
```

## Arquitetura

- `src/engine/**` — TS puro, sem DOM, zero libs de ML. Testável no Node.
- `src/web/**` — React + React Flow; apenas *consome* o engine e desenha.
