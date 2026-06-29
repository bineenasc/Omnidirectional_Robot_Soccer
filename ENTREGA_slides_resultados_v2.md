# Slides de Resultados reescritos (honestos) — colar no deck

Numbers consistent with the rewritten article. Keep your Slidesgo template.

================================================================
## SLIDE 1 — "Results — 1v0 Training"
================================================================

**LEFT (image):**
- Insert `learning_curve.png` (reward + sample efficiency + goal rate).
  *(optional second image: `monitor_training.png` — Viper/Titan reward & episode length)*

**RIGHT (key numbers — honest):**
- Trained with PPO + an 8-phase competency curriculum (ball near goal → full-field random)
- Curriculum was **unblocked**: promotion 50%/40ep was unreachable → revised to **40%/30ep + demotion**
- Deterministic eval (fixed phase): **goal rate 37.5%**, **timeout 62.5%**, own-goal **0%**, ball-out **0%**
- Mean final ball→goal distance: **0.28 m** (brings ball close, doesn't finish)
- Best episode: **goal in 55 steps** (R = +320)

**BOTTOM (findings):**
- Robots learned to **approach the ball and align with the goal**; no ball-fleeing after the `max(0, Δ)` ball→goal lock
- **Key finding (honest):** reward stays high (~208) while the **goal rate falls (~80% → ~25%)** and episodes run long (~729 steps) → the policy **parks near a goal post** instead of finishing (reward-shaping local optimum)
- Titan (slower, heavier) follows the same trajectory as Viper → bottleneck is the **reward balance**, not the robot

> **Speaker note:** "High reward but low goals is the signature of reward hacking — the agent farms the dense shaping by stalling near the post. We diagnosed it and mapped the fix in future work."

================================================================
## SLIDE 2 — "Results — 1v1 Training"
================================================================

**LEFT (image):**
- Insert `field.png` (1v1 field top-view, Viper vs Titan with the ball)

**RIGHT (pipeline + result — honest):**
- Full 1v1 pipeline built: **warmup → phase1 → phase2 (scripted opp.) → phase3 (frozen self-play)**
- 1v0 policy transferred via **weight surgery** (obs extended 24 → 30; opponent features zero-init)
- Warmup OK: Viper adapted to the 30-D observation space
- **eval_1v1 (20 episodes):**

  | Outcome | Count |
  |---|---|
  | Viper wins | 0 |
  | Titan wins | 0 |
  | **Draws** | **20** |

**BOTTOM (honest takeaway):**
- The 1v1 **framework works**, but no goals are scored yet: the 1v0 finishing weakness is **compounded** by the opponent blocking the goal mouth → mutual stalling
- Needs the reward re-balance + longer training + true simultaneous self-play (future work)
- Live monitor during training: `py -3.13 monitor_1v1.py` (reads `episode_log_1v1.csv` in real time)

> **Speaker note:** "We deliberately report the draws rather than overclaim — the pipeline is complete and the cause is understood: finishing, not the adversarial setup."

================================================================
## Imagens necessárias (só as que vocês têm)
================================================================
- `learning_curve.png` e `monitor_training.png` → Slide 1
- `field.png` → Slide 2

Nenhuma imagem nova precisa ser gerada.
