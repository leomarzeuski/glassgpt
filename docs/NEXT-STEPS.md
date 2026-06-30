# GlassGPT — Próximos passos (pra não esquecer)

> Lista viva de pendências. Última atualização: 2026-06-30.

## 1. GIF / clipe da demo (pro LinkedIn)

**Objetivo:** um clipe curto mostrando a animação nó-a-nó do backward (gradientes
âmbar acendendo de trás pra frente). É o visual que "para o scroll".

**Status:** existe um GIF rascunho gerado por automação de navegador
(`~/Downloads/glassgpt-backprop.gif`), mas ficou tosco — tem um "pulo" de scroll.

**Por que travou:** a automação de navegador tem limitações pra gravar isto —
cliques por coordenada não disparavam o botão de forma confiável, e o clique por
referência empurra a rolagem da página no meio da gravação.

**Caminho recomendado (rápido e impecável):** gravar manualmente a demo ao vivo
(https://glassgpt.vercel.app), agora que a animação está em 600ms:
- Mac: `⌘ + Shift + 5` → selecionar só a área do grafo → clicar "Animar backward"
  → parar. Postar o `.mov` (LinkedIn aceita vídeo) ou converter pra GIF.

**Alternativa (se quiser GIF automatizado):** ajustar o layout do explorador pra
caber o grafo inteiro sem precisar de scroll/zoom manual (ex: reduzir a altura do
canvas ou fixar o zoom do React Flow), aí a captura por automação fica estável.

## 2. Post do LinkedIn

- Finalizar depois de ter o clipe/GIF.
- Rascunho de texto já existe (formato história: "implementei backprop do zero em
  TS e deixei visível"). Incluir: demo ao vivo + repo + #MachineLearning #TypeScript.
- Anexar o clipe da animação (item 1).

## 3. Degrau 2 — makemore (próximo da escada)

A escada é `autograd → makemore → GPT`. O Degrau 1 (autograd + explorador) está
pronto e no ar.

**Escopo do Degrau 2:** gerador de nomes char-level — bigram → MLP → primeira
self-attention. É aqui que o autograd **escalar** vira **tensorial** (a reescrita
pra matrizes). Visualização-estrela: embeddings em 2D + distribuição do próximo
caractere ao vivo.

**Processo:** brainstorm → spec (`docs/superpowers/specs/`) → plano
(`docs/superpowers/plans/`) → execução TDD. Mesmo fluxo do Degrau 1.

## 4. Dívidas técnicas menores (não bloqueiam)

- `backwardFrames` (`src/engine/graphController.ts`) muta os grads do `Value` de
  entrada in-place; os chamadores precisam passar uma expressão recém-criada (os
  atuais já fazem). Adicionar um comentário `@param` documentando isso — bom fazer
  durante a reescrita tensorial do Degrau 2.
