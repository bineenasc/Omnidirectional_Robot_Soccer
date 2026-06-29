# Correções 1×0 — guia de execução (hoje)

Objetivo: obter um modelo 1×0 que realmente saiba aproximar-se da bola e
finalizar a partir de **qualquer** posição do campo, com uma métrica honesta,
antes de avançar para o 1×1.

## O que estava errado (diagnóstico)

1. **Eval enganoso** — `eval.py` rodava sempre na Fase 0 (bola a 5–20 cm do gol,
   robô já virado para ele). O "60% goal rate" media a tarefa trivial, não skill.
2. **Currículo travado** — promoção exigia 50% de gols, valor que o agente nunca
   atingia (saturava ~28%). Resultado: passava o treino quase todo nas fases fáceis.
3. **Sem demoção** — se promovido cedo demais, ficava preso numa fase difícil com
   reward baixo, sem recuar.
4. **Rollout PPO frágil** — `n_steps=1000` (= 1 episódio) dava estimativas de
   vantagem ruidosas.

## O que foi alterado

**`soccer_supervisor.py`**
- `_PROMOTE_THRESH` 0.50 → **0.40**, janela 40 → **30** (mais responsiva).
- Adicionada **demoção** (`_DEMOTE_THRESH = 0.10`): recua de fase se desabar.
- `set_phase()` e `freeze_curriculum()` — permitem ao eval fixar a dificuldade.

**`eval.py`** (reescrito o núcleo)
- Avalia numa **fase fixa** com o currículo **congelado** (métrica honesta).
- `EVAL_PHASE = None` → **varre as 8 fases** e imprime tabela goal/own/out/timeout.
- Distingue **timeout** (estourou o tempo) de gol/own-goal/out — antes ficava escondido.
- **Baseline de agente aleatório** (Fase 7) — exigido no documento de objetivos.
- `EVAL_FAST = True` para medir rápido (use False para ver o jogo em tempo real).

**`train.py`**
- **Warm-start** (`CONTINUE_FROM_FINAL = True`): continua a partir do
  `final_model_<robot>.zip` já treinado em vez de recomeçar do zero — com o
  currículo destravado, sobe para as fases difíceis muito mais rápido (poupa ~11 h).
- `n_steps` 1000 → **2048**, `batch_size` 250 → **256** (GAE mais estável).
- `N_EPOCHS` 13 → **8**, `STEPS_PER_EPOCH` 120k → **100k** (cabe num dia).
- **Backup automático** dos modelos atuais em `checkpoints/backup_1v0/` antes de
  continuar — os 11 h já treinados nunca são perdidos.

> A função de **reward não foi alterada** de propósito: já está bem afinada e
> mexer nela às cegas, sem tempo para A/B testar hoje, arriscaria piorar. Se
> depois das correções a taxa de **timeout** continuar alta nas fases finais, o
> próximo lever é tornar o termo "bola → gol" (×15, hoje só positivo) ligeiramente
> bilateral, ou adicionar um bónus de finalização perto do gol.

## Como rodar (passo a passo)

1. **Treinar** — em `controllers/shared_configs.py` ponha `MODE = "train"`,
   abra `worlds/soccer.wbt` no Webots e dê **play** (deixe correr, não pause).
   - Tempo estimado: ~2.5–3 h por robô (Viper depois Titan) ≈ 5–6 h no total.
   - Para iterar mais rápido, treine **só o Viper** primeiro: em `train.py`,
     `ROBOT_SEQUENCE = ["viper"]`.
2. **Avaliar** — ponha `MODE = "eval"`, play. Com `EVAL_PHASE = None` vai imprimir
   a tabela por fase + o baseline aleatório. Procure a coluna `goal%`.

## Critério para avançar ao 1×1 (decisão honesta)

Avance **somente** quando, no sweep com currículo congelado:
- **Fase 7 (aleatório): goal rate ≥ 50%**, e
- O modelo bate o **baseline aleatório** com folga em todas as fases, e
- `timeout%` baixo nas fases médias/altas (o robô finaliza, não só empurra).

Se isso for atingido, o `train_1v1.py` (que carrega estes `final_model_*.zip` e
estende a obs 24→30) parte de uma base sólida. Caso contrário, mais 1–2 epochs
de 1×0 valem mais que depurar o 1×1 sobre uma base fraca.
