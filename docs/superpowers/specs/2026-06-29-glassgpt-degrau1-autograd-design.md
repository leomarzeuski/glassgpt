# GlassGPT — Degrau 1: Autograd + Visualizador do Grafo de Computação

**Data:** 2026-06-29
**Status:** Design aprovado, pronto para planejamento de implementação
**Escopo deste spec:** Apenas o Degrau 1 da escada GlassGPT.

---

## Visão geral (a escada completa)

**GlassGPT** é uma série de 3 artefatos em TypeScript puro, cada um rodando no
navegador e cada um tornando **visível** o que acontece dentro do modelo. O
objetivo primário é **aprender LLMs por dentro de verdade**; a visibilidade no
LinkedIn vem como consequência de ensinar algo difícil e raro.

| Degrau | O que é | Visualização-estrela |
|---|---|---|
| **1. Autograd** | Motor de derivadas automáticas escalar (estilo micrograd) | Grafo de computação com valor + gradiente; backprop animado |
| **2. makemore** | Gerador de nomes char-level: bigram → MLP → 1ª self-attention | Embeddings 2D + distribuição do próximo caractere ao vivo |
| **3. GlassGPT** | Transformer/GPT completo no "tiny Shakespeare" | Attention heads acendendo durante a geração |

Cada degrau ganha seu **próprio spec → plano → implementação**. Este documento
cobre **só o Degrau 1**.

### Por que autograd escalar no Degrau 1 (decisão de escopo)

Cada nó do grafo guarda **um número** (não uma matriz). É de propósito:
- É o único jeito de visualizar o grafo de computação de forma bonita e
  intuitiva (com tensores o grafo vira uma sopa ilegível).
- É o melhor caminho pedagógico para entender backpropagation.
- A reescrita para **tensores** acontece no Degrau 2 e é, ela mesma, um momento
  de aprendizado importante — não é trabalho jogado fora.

---

## Princípio de arquitetura

O **engine** (matemática: autograd, backprop) é **TypeScript puro, sem React e
sem DOM** — testável no Node, com **zero bibliotecas de ML**. A camada de
**visualização** (React) apenas *consome* o engine e desenha. O engine não sabe
que o React existe.

```
glassgpt/
  engine/            # TS puro, sem DOM, zero deps de ML
    value.ts         # classe Value + operações + backward
    nn.ts            # Neuron, Layer, MLP (montados sobre Value)
    trace.ts         # trace(root) -> { nodes, edges } serializável
    *.test.ts        # vitest (rodam no Node)
  web/               # Vite + React + TS
    ...              # componentes de visualização que consomem o engine
```

> Nota: a estrutura exata de pastas (monorepo vs. app único com `src/engine`) é
> decidida no plano de implementação. O que é inegociável é a **fronteira**:
> engine puro, viz só consome.

---

## Componentes

### 1. `Value` — o núcleo do autograd

Embrulha um escalar e registra como o gradiente flui de volta.

```ts
class Value {
  data: number          // o valor escalar
  grad = 0              // a derivada acumulada (preenchida no backward)
  prev: Value[]         // Values que originaram este
  op: string            // operação que o criou: '+', '*', 'tanh', ...
  label?: string        // rótulo opcional para a viz
  _backward: () => void // empurra o gradiente para os pais
}
```

- **Operações:** `add`, `mul`, `pow`, `sub`, `div`, `neg`, `tanh`, `exp`,
  `relu`. Cada uma cria um novo `Value`, guarda os pais e registra `_backward`.
- **`backward()`:** ordena o grafo topologicamente, seta `grad = 1` na saída e
  percorre em ordem reversa chamando cada `_backward`. Gradientes **acumulam**
  (`+=`) para lidar com nós reutilizados.

### 2. `nn.ts` — mini-rede neural

`Neuron → Layer → MLP`, construídos só com `Value`. Permite treinar um exemplo
real (ajustar pontos 2D) e assistir gradientes + loss mudando. É o que deixa o
demo "vivo".

### 3. `trace.ts` — exportação do grafo

`trace(root): { nodes, edges }` caminha o DAG e devolve uma estrutura
serializável (`id`, `label`, `data`, `grad`, `op`) para a viz. Mantém o engine
livre de DOM.

### 4. Visualização (`web/`, React + React Flow)

- Renderiza o grafo de computação com **React Flow** (só posiciona/desenha os
  nós; a matemática continua 100% no engine).
- Cada nó mostra **label, valor e gradiente**; arestas ligam pais a filhos.
- **Forward** preenche valores; **Backward** roda `backward()` e **anima** os
  gradientes preenchendo na ordem topológica reversa, nó a nó.
- **Sliders** nos nós-folha: arrastar recomputa os valores ao vivo.
- **Demo de treino:** botão "1 passo de treino", número de loss, e a loss caindo
  a cada passo.

---

## Fluxo de dados

```
monta expressão (código/UI) → trace(root) → React desenha o grafo
   → [Forward] valores aparecem
   → [Backward] root.backward() → grad preenchido em cada nó
   → animação dos gradientes na ordem topológica reversa
```

---

## Correção e testes

A peça de rigor mais importante é o **gradient checking**: para cada operação e
para expressões compostas, estima-se o gradiente numericamente por diferenças
finitas e compara-se com o gradiente analítico do autograd.

```
grad_numérico ≈ (f(x + h) − f(x − h)) / (2h)
```

Bate dentro de tolerância (~`1e-5`) ⇒ a matemática está provada.

**Suíte de testes (vitest, no Node, sobre o engine puro):**
- Unitários por operação: forward + backward de cada op.
- **Gradient checking** automatizado (não só demo).
- **Teste de resposta conhecida:** replicar o exemplo canônico do micrograd do
  Karpathy e bater valor + gradiente nos números publicados.
- *(Stretch, degraus futuros):* paridade contra um exemplo exportado do PyTorch.

---

## Tratamento de erros (software educacional → erro é didático)

- Detectar **NaN/Inf** e mostrar o nó **vermelho** na viz (recurso de ensino,
  não bug escondido).
- Erros claros em `div` por zero e `pow` com base/expoente inválidos.
- O `backward()` assume DAG (sem ciclos); guardar contra ciclos no topo-sort.

---

## Stack

- **Vite + React + TypeScript** (engine e viz).
- **React Flow** para o grafo interativo (apenas renderização/layout).
- **vitest** para testes.
- **Deploy na Vercel** — cada degrau vira uma URL pública ao vivo.

---

## Marcos do Degrau 1

1. **Engine core** — `Value` + operações + `backward()`, com unitários e
   gradient checking passando (só Node, sem UI).
2. **Grafo estático** — `trace()` + render no React Flow de uma expressão fixa
   mostrando valores e gradientes.
3. **Interativo** — animação Forward/Backward + sliders nos nós-folha.
4. **Treino vivo** — MLP minúsculo ajustando pontos, botão "1 passo", loss
   caindo na tela.
5. **Polish + deploy na Vercel + README**.

### Definição de "pronto" (Degrau 1)

- URL ao vivo na Vercel onde dá para: montar uma expressão, ver o forward,
  clicar backward e assistir os gradientes fluírem, arrastar um peso e rodar
  passos de treino vendo a loss cair.
- Testes verdes, incluindo gradient checking e o teste de resposta conhecida.
- README limpo explicando como funciona (rascunho natural do post).

---

## Ângulo de conteúdo (objetivo secundário)

- **Um post forte quando o Degrau 1 sair**, em formato de história:
  *"Implementei backpropagation do zero em TypeScript e deixei o processo
  visível — aqui está o que aprendi"* + demo ao vivo + GIF dos gradientes
  fluindo.
- Repo público no GitHub com README caprichado.
- Depois, um carrossel ensinando **um** conceito (ex: "o que é um gradiente, na
  intuição") usando a própria visualização.

---

## Não-objetivos (Degrau 1)

- Tensores / autograd matricial (fica para o Degrau 2).
- Qualquer Transformer / attention (Degrau 3).
- Performance / treino de modelos grandes.
- Persistência, contas de usuário, backend.
